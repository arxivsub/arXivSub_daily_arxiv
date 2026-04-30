# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-30 | 今日论文总数: 413

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Calibrated Persistent Homology Tests for High-dimensional Collapse Detection

**arXiv ID:** 2604.26068 | [PDF](https://arxiv.org/pdf/2604.26068v1)

**作者:** Alexander Kalinowski `[一作]` `[通讯]` (SUNY Empire University), Alexander Kalinowski (SUNY Empire University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究如何通过持久同调（PH）统计量检测高维点云的“坍塌”现象，并给出校准后的显著性检验方法。

**💡 创新点**

创新点在于将 PH 的两种尺度摘要（总持久性 TP 与均值尾部超量 MTE）与两种过滤器（Vietoris–Rips 与 DTM）相结合，构建了针对三类坍塌机制（线性/光谱、非线性支持、混合/异质）的机制图，实现了检验选择的系统化指导。

**🔧 技术方法**

采用了持久图、总持久性 TP、均值尾部超量 MTE、Vietoris–Rips 过滤器、DTM 过滤器，以及基于经验分位数的阈值校准与多重检验修正。

**📊 数据集**

使用了多组合成点云数据，维度 d ∈{5,10,20}，样本量 n ∈{10,50,100}，并在三种坍塌机制下调节坍塌强度 ε。

**📈 对比分析**

通过对每种检验组合（过滤器×摘要）在不同机制和参数下的拒绝率进行比较，发现 MTE+DTM 组合在非线性支持机制下显著优于其他组合，TP 在所有机制下表现较弱。

**⚠️ 局限性**

局限性包括：总持久性 TP 敏感性不足；实验仅限于小规模合成数据，缺乏更广泛的真实数据验证；并且未考虑随时间演化的坍塌过程和更复杂的异质采样情况。

---

## 2. Human-Augmented Reality Interaction in Rebar Inspection

**arXiv ID:** 2604.26112 | [PDF](https://arxiv.org/pdf/2604.26112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 3. NeuralEmu: in situ Measurement-Driven, ML-based, High-Fidelity 5G Network Emulation

**arXiv ID:** 2604.26080 | [PDF](https://arxiv.org/pdf/2604.26080v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 4. Training Computer Use Agents to Assess the Usability of Graphical User Interfaces

**arXiv ID:** 2604.26020 | [PDF](https://arxiv.org/pdf/2604.26020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 5. SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding

**arXiv ID:** 2604.25925 | [PDF](https://arxiv.org/pdf/2604.25925v1)

**作者:** Yijun Lin `[一作]` (Renmin University of China), Feng Zhou `[通讯]` (Renmin University of China)

**通讯引用:** 12072 | [OpenAlex ID](https://openalex.org/A5047702220)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SpecTr-GBV，一种统一多稿生成与贪婪块验证的推测解码框架；

**💡 创新点**

将多稿生成与 GBV 结合，在最优传输框架下证明在 i.i.d. 草稿生成条件下可达最优接受长度，显著提升接受率；

**🔧 技术方法**

使用 i.i.d. 多稿生成、最优传输（OT）优化验证、改进的 GBV 算法，并提供理论分析与实验验证；

**📊 数据集**

在 HumanEval、GSM8K、MGSM、LM1B、Alpaca 等五大基准数据集上进行评估；

**📈 对比分析**

与 AR、SD、SpecTr、GBV 四个基线在块效率 (BE) 与加速比 (SR) 上对比，SpecTr-GBV 在所有基准上平均提升 10‑30% 的 BE，15‑30% 的 SR；

**⚠️ 局限性**

局限性包括：需要手动调优草稿长度、温度等超参数；草稿模型的计算开销在某些场景仍可能抵消加速收益；理论假设 i.i.d. 草稿，实际使用中可能偏离。

---

## 6. Sociodemographic Biases in Educational Counselling by Large Language Models

**arXiv ID:** 2604.25932 | [PDF](https://arxiv.org/pdf/2604.25932v1)

**作者:** Tomasz Adamczyk `[一作]` (Wrocław University of Science and Technology), Przemysław Kazienko `[通讯]` (Wrocław University of Science and Technology)

**通讯引用:** 5340 | [OpenAlex ID](https://openalex.org/A5049612210)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对6种大型语言模型在900多篇学生情景vignette上进行教育咨询回答，生成243,000条响应，系统评估社会人口偏见与信息密度的关系。

**💡 创新点**

首次大规模、细粒度地交叉设计15种社会人口身份、9信息密度和10咨询情境，揭示信息精确度能显著削弱偏差，并比较模型间的偏差差异。

**🔧 技术方法**

采用vignette实验设计、六种前沿LLM（GPT‑5、DeepSeek、Grok、Gemini等）API调用、强制选择提示、差异量化Δ以及t检验与Benjamini–Hochberg FDR校正。

**📊 数据集**

自生成900多篇学生情景vignette，涵盖15类社会人口标识与9信息密度，已公开发布于GitHub（https://github.com/tomadamczyk/llm-educational-bias）。

**📈 对比分析**

通过计算每个群体相对对照的平均差异Δ并使用z分数对六模型进行跨模型平均偏差对比；结果表明信息精确度可将偏差降低近三倍，且模型间偏差幅度差异显著。

**⚠️ 局限性**

实验采用情景化vignette，无法完全捕捉真实咨询互动；未细分交叉身份；仅关注美国教育背景；缺乏部署后实际影响评估。

---

## 7. Momentum-Conserving Graph Neural Networks for Deformable Objects

**arXiv ID:** 2604.26097 | [PDF](https://arxiv.org/pdf/2604.26097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 8. Associative-State Universal Transformers: Sparse Retrieval Meets Structured Recurrence

**arXiv ID:** 2604.25930 | [PDF](https://arxiv.org/pdf/2604.25930v1)

**作者:** Liu Xiao `[一作]` `[通讯]`, Liu Xiao

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了 UniMatrix 系列矩阵状态递归模型，并在字节级 WikiText-2 与合成记忆检索任务上进行评测，证明其在参数效率上可与 Transformer 竞争，但在精确检索方面需要显式稀疏记忆。

**💡 创新点**

创新点在于将 Universal Transformer 的共享深度迭代与矩阵状态递归结合，并引入规则混合更新、ROSA 残差路由、DeepEmbed 调制以及稀疏指针检索，以实现可压缩的长期上下文与精确检索的双重目标。

**🔧 技术方法**

使用了矩阵状态递归块、混合写入（outer、diag、sym）、ROSA 残差路径、DeepEmbed 嵌入调制、稀疏槽记忆与指针 logits 融合等技术，同时在 Apple MPS 上基于 Python 循环实现前向推理。

**📊 数据集**

主要使用了字节级 WikiText-2 数据集、合成的关联回忆（key‑value 检索）任务，以及对三元词交互的更正后合成基准来评估模型的语言建模与检索能力。

**📈 对比分析**

与传统 Transformer 基线相比，UniMatrix-Core 与 UniMatrix-ROSA 在 WikiText-2 上以约 40%–53% 参数减少实现 5.08–5.09 bits/byte 的轻微优势；然而在关联回忆任务中仅达到 12–14% 的准确率，除非采用 UniMatrix‑SparsePointer（32 槽 + 指针 logits）才可提升至 75–99% 的准确率。

**⚠️ 局限性**

局限性包括实验规模极小（仅 80 步 LM 训练和 200 步记忆任务），实现为 Python 循环缺乏高效核，稀疏记忆和检索机制仍未在实际推理中达到 Transformer 速度，且未在更大数据集、长上下文基准或官方 RULER/LongBench 等评测上验证。

---

## 9. RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts

**arXiv ID:** 2604.26039 | [PDF](https://arxiv.org/pdf/2604.26039v1)

**作者:** Vyom Sharma `[一作]` (Hippocratic AI), Debajyoti Datta `[通讯]` (Hippocratic AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 RaMP，一种基于专家路由分布的 Mixture-of-Experts 推理时动态内核调度框架。

**💡 创新点**

创新点在于：①通过硬件常数导出的性能区域分析，精确判定每种优化（如 GROUP_M、Split-K 等）何时有利；②构造四参数波浪成本模型，仅依据 CTA 网格大小即能在 0.93% 的平均误差下预测最优配置；③实现可与任何可调 fused MoE 内核兼容的路由感知调度，显著提升推理吞吐量。

**🔧 技术方法**

采用物理感知的成本模型、CuTe DSL 生成多态化 fused MoE kernel、Triton 统计专家直方图并进行一次性 38µs 的 argmin 选择、以及 OLS 线性回归求解四参数模型。

**📊 数据集**

实验使用 OLMoE‑1B‑7B、Qwen3、Mixtral、DSv3 等八种已发布 MoE 架构，并在未见的 Qwen3.5‑A3B 模型上进行验证；路由分布采样来源于真实推理工作负载。

**📈 对比分析**

与静态调度、Alpha‑MoE JIT、DeepGEMM、FlashInfer CUTLASS 等基线对比，RaMP 在内核层面平均提升 1.22×，在 vLLM 端到端推理中平均提升 1.30×（相较 Triton FP8）并实现 1.41×（相较 DeepGEMM）与 1.13×（相较 FlashInfer CUTLASS）加速；成本模型平均误差仅 0.93%。

**⚠️ 局限性**

局限性包括：仅针对 Hopper（W8A8 FP8）单 GPU 环境，Blackwell 等新架构需要重新采样；对极端均匀路由或单 token batch 的场景无显著优势；跨 GPU 的专家并行通信不在本文讨论范围内。

---

## 10. Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective

**arXiv ID:** 2604.25975 | [PDF](https://arxiv.org/pdf/2604.25975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 11. PPG-Based Affect Recognition with Long-Range Deep Models: A Measurement-Driven Comparison of CNN, Transformer, and Mamba Architectures

**arXiv ID:** 2604.26078 | [PDF](https://arxiv.org/pdf/2604.26078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 12. A Multimodal and Explainable Machine Learning Approach to Diagnosing Multi-Class Ejection Fraction from Electrocardiograms

**arXiv ID:** 2604.25942 | [PDF](https://arxiv.org/pdf/2604.25942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 13. SWE-Edit: Rethinking Code Editing for Efficient SWE-Agent

**arXiv ID:** 2604.26102 | [PDF](https://arxiv.org/pdf/2604.26102v1)

**作者:** Yikai Zhang `[一作]` (Microsoft), Zijian Jin `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SWE-Edit框架，将代码编辑接口拆分为Viewer和Editor子代理，以减轻上下文污染并提升编辑可靠性

**💡 创新点**

创新点包括（1）通过子代理拆分上下文与格式敏感编辑的耦合问题；（2）训练Qwen3-8B的自适应编辑模式（find‑replace与全文件重写）；（3）构建PR‑Edit基准用于快速评估编辑模型

**🔧 技术方法**

使用大语言模型（GPT‑5、GPT‑5‑mini、Qwen3‑8B）以及GRPO强化学习、上下文过滤与代码规范化技术

**📊 数据集**

使用SWE‑bench Verified（500 GitHub issue）与PR‑Edit（500 PR编辑样本）以及公开的GitHub PR数据集

**📈 对比分析**

相较基线，SWE‑Edit在SWE‑bench Resolve率提升2.1%、编辑成功率提升3.5%，推理成本降低17.9%；自适应编辑模型比固定find‑replace提升12.5%编辑成功率

**⚠️ 局限性**

局限性：编辑模型仅在离线训练，未结合端到端奖励；仅针对单文件或仓库级编辑，未考虑多代理协同训练；对极大文件的全文件重写仍高成本

---

## 14. Entropy Centroids as Intrinsic Rewards for Test-Time Scaling

**arXiv ID:** 2604.26173 | [PDF](https://arxiv.org/pdf/2604.26173v1)

**作者:** Wenshuo Zhao `[一作]` (Hong Kong University Of Science And Technology), Yiren Feng `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于高熵阶段（HEP）和熵重心（Entropy Centroid）的内部奖励方法，用于多路推理时挑选最佳输出，从而提升大型语言模型的测试时计算效率。

**💡 创新点**

创新点在于：①将连续高熵令牌聚类为高熵阶段，将噪声降至可控层级；②借鉴质心概念，用加权平均位置表示不确定性随时间的分布；③用最低熵重心作为不需要外部奖励模型的自我评估信号，实现无监督的路径选择。

**🔧 技术方法**

技术手段包括：token熵计算、阈值判定与滑动窗口状态机构造 HEP、质心公式的权重求和、轨迹重心归一化、以及在多路采样后基于重心选取最佳轨迹。

**📊 数据集**

使用的数据集涵盖多领域：数学（AIME25、Minerva Math）、代码（BigCodeBench、LiveCodeBench）、逻辑（Synlogic）、代理（τ²-Bench）以及多种大型模型的评测缓存。

**📈 对比分析**

与 Pass@1、贪心解码、Self-Certainty、Tail Confidence、Bottom Window 等内在奖励方法对比，最低熵重心平均提升约 5.3%（最大 10.1%）的通过率，且在 14B–480B 参数规模下保持稳定优越，显示出优良的可扩展性与跨任务鲁棒性。

**⚠️ 局限性**

局限性包括：对 HEP 的阈值（θ_high、θ_low、k）需预先设定，尽管实验表明鲁棒但仍需微调；在极短答案或需要显式答案计数的任务中，不能替代多数投票；在极低准确度场景下，熵重心对噪声的抑制仍有限。

---

## 15. Quantum Bayesian Networks: Compositionality and Typing via Linear Logic

**arXiv ID:** 2604.26059 | [PDF](https://arxiv.org/pdf/2604.26059v1)

**作者:** Rémi Di Guardia `[一作]`, Claudia Faggian `[通讯]` (Université Paris Cité)

**通讯引用:** 596 | [OpenAlex ID](https://openalex.org/A5028757410)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了量子贝叶斯网络的组合语义和基于线性逻辑证明网的类型化图形形式，解决了传统量子贝叶斯网络缺乏组合性与模块化的问题。

**💡 创新点**

创新点在于将经典贝叶斯网络的因子化语义与纯量子张量网络相统一，同时引入类型约束保证系统组合的合法性与可推理性。

**🔧 技术方法**

采用了Selinger的量子语义框架、因子（factor）运算、张量网络以及线性逻辑证明网（proof‑nets）等数学工具。

**📊 数据集**

论文为理论性工作，未使用具体实验数据集；主要通过数学证明与图例说明其语义一致性。

**📈 对比分析**

通过理论推导证明：当所有因果变量为经典时，语义与传统贝叶斯网络相同；在纯量子情形下与张量网络等价；未进行数值实验，性能评估为理论上可组合与可推理。

**⚠️ 局限性**

局限性包括：尚未实现完整的推理算法实现、缺乏大规模实验验证、对复杂量子系统的计算成本与可扩展性仍待进一步研究。

---

## 16. A Survey of Multi-Agent Deep Reinforcement Learning with Graph Neural Network-Based Communication

**arXiv ID:** 2604.25972 | [PDF](https://arxiv.org/pdf/2604.25972v1)

**作者:** Valentin Cuzin-Rambaud `[一作]` (Université Lyon 1), Maxime Morge `[通讯]` (Université Lyon 1)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并归纳了基于图神经网络（GNN）的多智能体强化学习（MARL）通信方法，提出了统一的通用通信框架。

**💡 创新点**

创新点在于将不同方法映射到一个抽象的GNN通信流程，形成系统的分类与对比；并指出当前研究在通信约束（范围、带宽、噪声、丢包）方面的不足。

**🔧 技术方法**

使用的技术主要是图卷积网络（GCN）、图注意力网络（GAT）、MPNN等GNN结构，结合代理（proxy）与分布式通信机制，以及强化学习框架（CTDE、DTDE、actor‑critic 等）。

**📊 数据集**

作为综述，未使用新的实验数据集；文中引用的典型案例如捕食者-猎物（Predator‑Prey）等常见 MARL 基准环境。

**📈 对比分析**

通过对已有十二种主流 GNN‑通信方法的拆解与对比，展示了各自的聚合策略、通信轮数、代理使用与否等差异；对比中发现代理方法在全局一致性上优于分布式方法，但对通信范围与规模的适应性差；分布式方法在规模扩展与局部协调上更具优势。

**⚠️ 局限性**

局限性包括：1) 代理通信假设理想的全连通与完美传输，实际应用难以满足；2) 现有方法很少考虑带宽、噪声、丢包等真实通信约束；3) 缺乏统一的实验基准与可复现的实现细节；4) 对大规模代理的可扩展性仍待进一步验证。

---

## 17. Generalized Disguise Makeup Presentation Attack Detection Using an Attention-Guided Patch-Based Framework

**arXiv ID:** 2604.26025 | [PDF](https://arxiv.org/pdf/2604.26025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 18. LLM-Guided Issue Generation from Uncovered Code Segments

**arXiv ID:** 2604.26118 | [PDF](https://arxiv.org/pdf/2604.26118v1)

**作者:** Diany Pressato `[一作]` (Concordia University), Shin Hwei Tan `[通讯]` (Concordia University)

**通讯引用:** 1789 | [OpenAlex ID](https://openalex.org/A5051957977)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 IssueSpecter，一种自动化管道，利用测试覆盖率定位未测试代码段，并通过 LLM 自动生成可操作的 bug 报告

**💡 创新点**

创新点在于将覆盖率分析与 LLM 缺陷识别结合，并通过两阶段（规则+LLM）排序实现高优先级报告的优先级排序

**🔧 技术方法**

使用 Python 覆盖工具 SlipCover 识别未覆盖代码段，利用 GPT‑5‑mini 进行缺陷检测与报告生成，并采用规则与 LLM 双重排序策略

**📊 数据集**

在 13 个活跃 Python 开源项目（来自 CodaMosa 数据集）上进行评估，生成 10,467 条报告

**📈 对比分析**

与覆盖率驱动的测试生成工具 CoverUp 进行对比，IssueSpecter 在有效率（81% vs 76%）和结构化报告方面略占优势，LLM 排序在 P@3 提升 50%、MRR 提升 41%，整体性能显著

**⚠️ 局限性**

受限于 LLM 的非确定性与确认偏差、只针对 Python、报告真伪率仍存在一定误报、对大型项目覆盖率与缺陷分布的依赖等

---

## 19. Operating-Layer Controls for Onchain Language-Model Agents Under Real Capital

**arXiv ID:** 2604.26091 | [PDF](https://arxiv.org/pdf/2604.26091v1)

**作者:** T. J. Barton `[一作]` (DX Research Group), Hunter Goodreau `[通讯]` (DX Research Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究在DX Terminal Pro真实资本交易环境中，评估并提升基于语言模型的自主交易代理的可靠性，使用了3,505个以太坊ETH托管仓库、7.5M次代理调用、70B推理token、12种memecoin，并记录完整的从用户指令到链上结算的全链路追踪；

**💡 创新点**

创新点在于将可靠性归因于系统运行层（提示编译、类型化控件、策略校验、执行守护、内存设计与可观测性）而非仅模型本身，并通过识别、量化并修复五大失效模式（规则伪造、费用瘫痪、代币经济误读、数字硬化、节奏交易）显著提升性能；

**🔧 技术方法**

主要技术包括：Qwen3-235B-Thinking模型与SGLang推理框架、Prompt编译与参数化控件、政策验证与执行守护、链上托管合约、全链路追踪日志、Claude Sonnet用于行为分类；

**📊 数据集**

数据集为DX Terminal Pro真实交易记录，包含3,505个用户托管仓库、7.5M次代理调用、约70B推理token、12种代币、约20M交易量、约5,000 ETH投入及对应的链上交易和结算信息；

**📈 对比分析**

方法对比：在预启动阶段通过重放场景与固定握手（harness）测试，量化失效率；上线后与预启动基线比较，规则伪造率从57%降至3%，费用引用率从32.5%降至<10%，资本部署率从42.9%升至78%，最终结算成功率达到99.9%；

**⚠️ 局限性**

局限性包括：仅单一市场与交易所（Base），单一模型族（Qwen），有限的代币池（12种），仅21天的实验周期，未覆盖跨资产/跨场所的泛化；此外，结果主要为观察性，缺乏随机化对照实验。

---

## 20. GenDetect: Generalizing Reactive Detection for Resilience Against Imitative DeFi Attack Cascade

**arXiv ID:** 2604.26094 | [PDF](https://arxiv.org/pdf/2604.26094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 21. Information Extraction from Electricity Invoices with General-Purpose Large Language Models

**arXiv ID:** 2604.25927 | [PDF](https://arxiv.org/pdf/2604.25927v1)

**作者:** Javier Gómez `[一作]` (Centro de Tecnologías de la Imagen), Javier Sánchez `[通讯]` (Centro de Tecnologías de la Imagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在未对大型语言模型进行微调的情况下，使用多种提示工程策略对西班牙电力发票进行结构化信息抽取，并系统评估模型、提示与推理参数的影响。

**💡 创新点**

提示质量对抽取性能影响显著，超过参数调优与模型架构差异；模型可在未见过的模板上保持95–96% F1；通过多例交叉验证实现高效、鲁棒的抽取。

**🔧 技术方法**

采用 Gemini 1.5 Pro（稀疏 MoE Transformer）和 Mistral‑small（Dense Transformer）两大类模型；通过零/少/多示例提示、迭代字段抽取等策略；使用温度、Top‑K、Top‑P等采样参数；评估指标为精确率、召回率和 F1。

**📊 数据集**

IDSEM（西班牙电力市场发票）数据集：75,000 份合成 PDF 发票，包含 6 种模板、107 个语义标签；实验使用 2,400 份随机抽样（400 每模板）进行评估。

**📈 对比分析**

与传统机器学习（SVM RBF）对比：LLM 在未见模板上精度约 95%，而传统方法降至 67%；在同一数据集上，Gemini 在最佳提示下 F1 最高达 97.61%，Mistral 达 96.11%。

**⚠️ 局限性**

仅使用文本提取，未利用视觉或布局信息，可能导致信息丢失；实验基于合成数据，真实发票的噪声和多样性未被充分验证；模型在推理参数上几乎无显著差异，提示工程仍是瓶颈。

---

## 22. Digital Twin-assisted belief-state reinforcement learning for latency-robust ISAC in 6G networks

**arXiv ID:** 2604.25967 | [PDF](https://arxiv.org/pdf/2604.25967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 23. Sample Selection Using Multi-Task Autoencoders in Federated Learning with Non-IID Data

**arXiv ID:** 2604.26116 | [PDF](https://arxiv.org/pdf/2604.26116v1)

**作者:** Emre Ardıç `[一作]`, Yakup Genç `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并评估了基于多任务自编码器的联邦学习样本选择方法，利用损失和特征分析去除噪声样本以提升模型性能。

**💡 创新点**

创新点在于将多任务自编码器与无监督异常检测（OCSVM、IF）和自适应阈值、以及联邦多类SVDD损失相结合，实现高效的样本贡献估计和选择。

**🔧 技术方法**

采用的技术包括多任务自编码器、FedAvg聚合、One-Class SVM、Isolation Forest、Adaptive Threshold、Federated SVDD、FedML仿真平台、PSNR/SSIM评估等。

**📊 数据集**

实验使用MNIST和CIFAR10为主要数据集，SVHN、EMNIST、ImageNet32作为开放集噪声源，并构造非IID客户端分布。

**📈 对比分析**

与无样本选择基线对比，使用40%噪声率在不同客户端数下测量准确率、PSNR、SSIM和F1；OCCSVM在CIFAR10上提升高达7.02%，AT在MNIST上提升1.83%，IF亦实现显著改进，整体准确率提升约1–7%。

**⚠️ 局限性**

主要限制包括OCSVM/IF的训练和推理复杂度高，需手动设置污染率与阈值，启动时机对效果敏感，特征空间方法效果有限，SVDD在低复杂度数据上无显著优势。

---

## 24. Hierarchical Multi-Persona Induction from User Behavioral Logs: Learning Evidence-Grounded and Truthful Personas

**arXiv ID:** 2604.26120 | [PDF](https://arxiv.org/pdf/2604.26120v1)

**作者:** Nayoung Choi `[一作]` (Emory University), Jinho D. Choi `[通讯]` (Emory University)

**通讯引用:** 2574 | [OpenAlex ID](https://openalex.org/A5101829031)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一个分层框架，从用户行为日志生成多种可解释的自然语言用户画像，并将其与支持记忆链接。

**💡 创新点**

创新点在于：①把行为日志聚合成意图记忆再聚类；②用聚类凝聚度、证据对齐和真确度三维度量用户画像质量；③将这些质量信号转化为离线强化学习的奖励，通过分组 Direct Preference Optimization（DPO）实现自适应优化。

**🔧 技术方法**

使用大型语言模型（Gemma3、Qwen3等）做记忆摘要与画像生成，BGE-M3 作为嵌入模型评估凝聚度，LLM 判断器（Qwen3）评估对齐与真确度，离线 RL‑style 的 groupwise DPO 作为训练目标。

**📊 数据集**

实验使用三组数据：①一大规模线上服务日志；②公开的购物推荐日志（英文）；③公开的搜索 Web 日志（英文）。

**📈 对比分析**

与闭源与开源前沿 LLM（GPT‑5.1、Claude‑4.5、GPT‑oss‑120B、Qwen3‑80B）以及基于聚类的 PersonaX‑s/r 进行对比。结果显示，在聚合度、对齐度、真确度和综合质量上均位列榜首，且在未来交互预测的 Hit@k / MAP@k 上也取得最优或最接近最优的性能。

**⚠️ 局限性**

主要局限包括：①评估依赖 LLM 判定器，可能存在主观偏差；②使用离线 RL，缺乏在线更新与持续学习；③未解决用户画像随时间演变的更新机制；④仅在单一下游任务（未来交互预测）上验证效果；⑤在敏感数据使用上存在隐私与用户画像风险。

---

## 25. A Scaled Three-Vehicle Platooning Platform

**arXiv ID:** 2604.25963 | [PDF](https://arxiv.org/pdf/2604.25963v1)

**作者:** Kaiyue Lu `[一作]` (University of New Brunswick), Yukun Lu `[通讯]` (University of New Brunswick)

**通讯引用:** 132 | [OpenAlex ID](https://openalex.org/A5011413784)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

搭建并实验验证了一套三车缩尺车队平台，比较了纯追踪与Stanley两种几何侧向控制器在车队变道过程中的性能

**💡 创新点**

首次在缩尺车队环境中使用ArUco标记实现纯视觉追踪，并通过实验揭示了Pure Pursuit在抑制扰动和减少车间振幅方面优于Stanley的优势

**🔧 技术方法**

采用NVIDIA Jetson Orin NX+STM32微控制器、深度相机、LiDAR、IMU、ROS2 Humble以及PID+Pure Pursuit/Stanley几何控制算法

**📊 数据集**

使用自制的室内实验场景和ArUco标记作为定位数据源，未使用公开数据集

**📈 对比分析**

通过在相同初始与环境条件下分别采用Pure Pursuit和Stanley，比较了横向位置误差、速度和偏航响应；实验表明两者都能完成变道，但Pure Pursuit在预变道阶段的振幅更小，响应更平滑，且车间幅度放大程度更低

**⚠️ 局限性**

仍存在车间振幅放大、低速下控制灵敏度不足、缺乏通信延迟/误差分析以及更复杂道路操作的验证

---

## 26. Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time

**arXiv ID:** 2604.26070 | [PDF](https://arxiv.org/pdf/2604.26070v1)

**作者:** Jennifer Wendland `[一作]` (University of Koblenz), Maik Kschischo `[通讯]` (University of Koblenz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出可观测神经ODE（ObsNODE）模型，利用连续时间状态空间框架对序列决策中的因果预测进行建模，并给出了相应的连续时间因果调整公式。

**💡 创新点**

创新点在于将可观测正则形式嵌入神经ODE，保证潜在状态可观测性，从而将控制理论可观测性与因果可识别性联系起来，并首次在连续时间状态空间模型中实现动态处理效应的可识别性。

**🔧 技术方法**

技术方法包括可观测正则形式的神经ODE、基于RNN的过滤分布编码器、自监督训练以及基于条件前门的离散/连续时间调整公式。

**📊 数据集**

实验数据涵盖合成癌症数据、基于MIMIC‑IV的半合成败血症数据以及真实MIMIC‑IV败血症临床数据（SOFA及相关生化指标）。

**📈 对比分析**

与 IGC‑Net、SCIP‑Net、doseAI、OptAB 等模型比较时，ObsNODE 在合成和半合成数据的短期预测中表现最佳，在长期预测中与 doseAI 竞争，在真实败血症数据中多项指标 RMSE 较低且方差小，优于其他基线。

**⚠️ 局限性**

局限性在于对潜在状态可观测性、无遗漏混杂等模型假设的依赖，且在多模态真实场景中的适用性和不确定性量化仍待进一步研究。

---

## 27. Mini-Batch Class Composition Bias in Link Prediction

**arXiv ID:** 2604.25978 | [PDF](https://arxiv.org/pdf/2604.25978v1)

**作者:** Kieran Maguire `[一作]` (University of Southampton), Srinandan Dasmahapatra `[通讯]` (University of Southampton)

**通讯引用:** 1609 | [OpenAlex ID](https://openalex.org/A5016843510)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了图神经网络在链接预测任务中利用批归一化层和固定正负边比例的批次大小所产生的偏差，提出了一种随机化正负边比例的批次采样方法（bias‑corrected mini‑batching），并验证该方法能提升网络对节点类别相关特征的学习与迁移能力。

**💡 创新点**

创新点在于揭示批归一化在链接预测中可被模型利用为“批次级别的启发式”，并通过随机化批次正负比例来消除这一偏差，既保持模型训练的稳定性，又提升了其对节点类别信息的捕捉和迁移性能。

**🔧 技术方法**

核心技术包括：图神经网络编码器（GCN、GraphSAGE、NEOGNN 等），批归一化（BatchNorm）与 MLP 预测层，正负边比例随机化的 mini‑batch 采样，Trace Ratio 与 NMI 评估指标，t‑SNE 可视化，以及通过平均邻居边嵌入生成节点嵌入以评估迁移效果。

**📊 数据集**

使用的图数据集包括 Cora、Citeseer、Pubmed、CS、Computers、Photo 以及大规模 OGBL 数据集 ogbl‑collab、ogbl‑ppa；实验覆盖多种链接预测模型（BUDDY、ELPH、NEOGNN、NCN、GCN、GraphSAGE）。

**📈 对比分析**

与原始批次方法相比，新方法在 hits@100、Trace Ratio、NMI 等指标上表现：链接预测性能略有下降（平均约 1–3%），但网络对节点类别特征的捕捉显著提升（Trace Ratio 与 NMI 均平均提升 10–20%），表明模型学习到了更具通用性的图结构信息。

**⚠️ 局限性**

局限性包括：1）去除批次偏差会导致链接预测精度下降，需在应用场景中权衡；2）实验主要集中在批归一化的作用，其他归一化或正则化方式的影响未深入探究；3）随机化正负比例增加了训练方差，可能影响收敛稳定性；4）仅在公开图数据集上验证，真实大规模异构图的表现尚未评估。

---

## 28. FruitProM-V2: Robust Probabilistic Maturity Estimation and Detection of Fruits and Vegetables

**arXiv ID:** 2604.26084 | [PDF](https://arxiv.org/pdf/2604.26084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 29. I Would If I Could: Reasoning about Dynamics of Actions in Multi-Agent Systems

**arXiv ID:** 2604.26053 | [PDF](https://arxiv.org/pdf/2604.26053v1)

**作者:** Rustam Galimullin `[一作]` (University of Bergen), Munyque Mittelmann `[通讯]` (CNRS)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种新的逻辑框架，动态动作逻辑（Dynamic Actions），用于建模多智能体系统中智能体的动态行为，特别是授予和撤销动作的过程。

**💡 创新点**

创新点在于引入了动态动作的概念，允许智能体在执行过程中动态更新其可用动作，并分析这些更新对智能体知识的影响。

**🔧 技术方法**

使用了动态动作逻辑（Dynamic Actions）和扩展的动态动作逻辑（Epistemic ATL with Dynamic Actions），并进行了复杂性分析和表达能力研究。

**📊 数据集**

论文中没有具体提到使用的数据集，而是通过理论模型和示例（如机器人协作）来说明逻辑的应用。

**📈 对比分析**

通过与现有的交替时间时序逻辑（ATL）和交替时间认知逻辑（ATL*）进行比较，展示了动态动作逻辑在表达能力上的优势，并证明了模型检查问题的复杂性。

**⚠️ 局限性**

限制在于该逻辑框架的应用范围和复杂性，尤其是在处理不完美信息和智能体之间的知识共享时，可能需要进一步的研究和扩展。

---

## 30. Analysing Lightweight Large Language Models for Biomedical Named Entity Recognition on Diverse Ouput Formats

**arXiv ID:** 2604.25920 | [PDF](https://arxiv.org/pdf/2604.25920v1)

**作者:** Pierre Epron `[一作]` (Inria, Inserm, Université Paris Cité, HeKA, UMR 1346), Mehwish Alam `[通讯]` (Télécom Paris, Institut Polytechnique de Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文使用指令微调的轻量级大语言模型，对医学领域的生成式命名实体识别任务进行实验；

**💡 创新点**

创新点在于系统评估多种输出格式（12种）对模型性能的影响，并证明在多格式联合训练下模型仍保持稳定；

**🔧 技术方法**

主要技术包括基于因果语言模型的指令微调、不同实体输出格式的设计以及微调过程中的多格式混合训练；

**📊 数据集**

使用了八个公开生物医学NER数据集（AnatEM、BC2GM、BC4CHEMD、BC5CDR、CADEC、GENIA、NCBI Disease、PGxCorpus）；

**📈 对比分析**

通过与UniNER、InstructIE等基线模型（均为7B+参数模型）以及BERT/GLiNER等方法比较，轻量级模型（0.5B~1B参数）在多数数据集上达到与大模型相当的F1（最高约0.83），且在复合实体识别上表现可观；

**⚠️ 局限性**

局限性包括仅测试两款轻量级模型、未尝试最低表现格式、未对超参数进行系统优化、未评估未微调的预训练模型、以及对极少样本或高度复杂实体的处理仍不足。

---

## 31. SAND: Spatially Adaptive Network Depth for Fast Sampling of Neural Implicit Surfaces

**arXiv ID:** 2604.25936 | [PDF](https://arxiv.org/pdf/2604.25936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 32. Evaluating the Alignment Between GeoAI Explanations and Domain Knowledge in Satellite-Based Flood Mapping

**arXiv ID:** 2604.26051 | [PDF](https://arxiv.org/pdf/2604.26051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 33. Privacy-Preserving Federated Learning Framework for Distributed Chemical Process Optimization

**arXiv ID:** 2604.26073 | [PDF](https://arxiv.org/pdf/2604.26073v1)

**作者:** Teetat Pipattaratonchai `[一作]`, Aueaphum Aueawatthanaphisut `[通讯]` (Sirindhorn International Institute of Technology, Thammasat University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了一个隐私保护的联邦学习框架，用于跨地理分散的化工厂进行过程模型的协同训练与优化，保证原始工况数据不被外泄。

**💡 创新点**

创新点在于：①首次将联邦学习引入化工过程建模，突破多厂数据共享壁垒；②针对非IID数据提出加权聚合与自适应权重机制；③采用安全加密聚合技术，进一步提升隐私保障。

**🔧 技术方法**

使用技术包括联邦学习（FedAvg）、安全加密聚合、神经网络过程模型以及IIoT边缘计算平台。

**📊 数据集**

使用三家化工厂各自的时间序列传感器CSV数据（非IID），共计三组独立数据集。

**📈 对比分析**

通过与局部单机训练和理论中心化训练对比，实验显示在40轮通信后全局MSE从约2369降至≈35；联邦学习相较于局部训练误差下降约63%–78%，几乎与中心化模型持平。

**⚠️ 局限性**

局限性包括：仅在三家小规模数据集上验证；未考虑恶意客户端攻击、差分隐私等更严格安全场景；缺乏更大规模或实时控制系统的集成验证。

---

## 34. CogRAG+: Cognitive-Level Guided Diagnosis and Remediation of Memory and Reasoning Deficiencies in Professional Exam QA

**arXiv ID:** 2604.25928 | [PDF](https://arxiv.org/pdf/2604.25928v1)

**作者:** Xudong Wang `[一作]` (Hangzhou City University), Zhaoyan Ming `[通讯]` (Hangzhou City University)

**通讯引用:** 1434 | [OpenAlex ID](https://openalex.org/A5017748295)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CogRAG+，一种无训练的认知层级引导检索增强生成（RAG）框架，专门用于解决专业考试中的知识缺口和推理不一致问题。

**💡 创新点**

创新点包括：① 将 Bloom 分类的认知层级嵌入检索和推理流程，实现检索与推理的解耦与对齐；② 引入“强化检索”双路径策略（事实中心与选项中心），动态补充缺失基础知识；③ 设计认知分层的约束推理模板（Fact‑Centric 与 Rule‑Centric），降低冗余和逻辑矛盾。

**🔧 技术方法**

核心技术：基于标签约束的检索、判定器驱动的检索强化、认知层级引导的系统/用户提示、结构化推理模板、FAISS 索引、无监督阈值触发机制。

**📊 数据集**

使用 62,478 条营养学 QA 对（来自 FoodEarth、MedQA、Nutri7Base），并在 Registered Dietitian 资格考试数据集（单选 811 题、情境 379 题）上评估。

**📈 对比分析**

与传统 RAG（BM25、Dense、Hybrid）以及基线模型（Qwen3‑8B、Llama‑3.1‑8B）进行对比。CogRAG+ 在单题模式下 Qwen3‑8B 的整体准确率提升至 85.8%（相对基线 +12.4%），Llama‑3.1‑8B 提升至 60.3%；情境模式下分别为 80.5% 与 57.8%。约束推理将未回答率从 7.6% 降至 1.4%，整体准确率也随之提升。

**⚠️ 局限性**

局限性：1）在其他营养子领域或完全不同专业任务的泛化尚待验证；2）阈值 α、β 采用经验设定，缺乏自动调优；3）知识库覆盖仍有限，极专业查询仍可能缺乏足够证据。

---

## 35. A Scoping Review of LLM-as-a-Judge in Healthcare and the MedJUDGE Framework

**arXiv ID:** 2604.25933 | [PDF](https://arxiv.org/pdf/2604.25933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 36. Designing Rewards for Rewarding Designs: Demonstrating the Impact of Rewards on the Creative Design Process

**arXiv ID:** 2604.26083 | [PDF](https://arxiv.org/pdf/2604.26083v1)

**作者:** Surabhi S Nath `[一作]` (Max Planck Institute for Biological Cybernetics), Shabnam Hakimi `[通讯]` (Toyota Research Institute)

**通讯引用:** 3008 | [OpenAlex ID](https://openalex.org/A5021158814)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

使用马尔可夫决策过程(MDP)框架对3D参数化椅子设计任务进行建模，并通过两种奖励信号（目标对齐奖励和目标无关奖励）在设计过程的每一步即时反馈分数，研究奖励对设计决策、行为和主观体验的影响。

**💡 创新点**

首次将精确的、目标相关的奖励信号与设计过程结合，探讨奖励质量与设计目标类型对创造性决策的交互作用；同时提出了基于一致性分布的奖励学习方法，并系统评估了奖励的有效性。

**🔧 技术方法**

采用MDP建模、层次贝叶斯无监督学习(PyMC+NUTS)估计目标特定分布、主成分分析与自编码器用于设计空间降维、Truncated SVD绘制奖励景观、混合效应模型分析行为差异、Gower距离评估设计相似度。

**📊 数据集**

使用两套数据集：一是120名受试者产生的基准设计（分别对应3个目标共约650个设计）用于训练目标对齐奖励分布；二是353名在线受试者在实验中生成的设计数据，用于验证奖励效果。

**📈 对比分析**

通过对比奖励类型（对齐 vs 无关）与目标类型（cheerful、dependable、unique）的实验设计，并利用混合效应模型、方差分析、相关分析等统计方法评估行为与奖励分数差异。结果显示，目标对齐奖励显著提升奖励得分（平均提升约8.4分）、加速操作速度、增加探索深度，且保持设计多样性；目标无关奖励的提升幅度仅为约4.2分。

**⚠️ 局限性**

局限包括：奖励分布假设独立特征且正态，可能忽略特征间相关性和多模态分布；实验阶段未随机化，存在顺序效应；样本为普通大众，缺乏专业设计师的验证；界面与激励结构较简化，生态有效性有限。

---

## 37. Lightweight Quantum Agent for Edge Systems: Joint PQC and NOMA Resource Allocation

**arXiv ID:** 2604.25980 | [PDF](https://arxiv.org/pdf/2604.25980v1)

**作者:** Yongtao Yao `[一作]` (Guangxi University), H. Herbert Song `[通讯]` (University of Maryland Baltimore County)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种轻量级的智能代理框架，用于在量子安全的 NOMA 边缘计算系统中实现在线联合优化，包括任务离线决策与资源分配；同时给出了 O(N) 的 NOMA 功率分配贪心递归算法，并将其与基于 Lyapunov 的深度强化学习（Actor-Critic）结合，实现实时决策。

**💡 创新点**

创新点在于：①将后量子密码（PQC）模块的恒定功耗显式纳入多阶段 MINLP 优化模型；②引入虚拟能量队列，将长期能耗约束转化为队列稳定性问题；③利用 Lyapunov 拓扑将非凸多阶段问题拆解为单帧确定性子问题；④设计了 O(N) 的贪心后向递推功率分配算法，使得全局非凸问题可分解为独立单变量凸子问题；⑤通过 Actor-Critic 结合模型驱动评估，显著提升了 DRL 的收敛速度和鲁棒性。

**🔧 技术方法**

核心技术包括：混合整数非线性规划（MINLP），Lyapunov 稳定性理论，深度强化学习（Actor-Critic 网络），贪心后向递推算法，递归干扰表达式，模拟退火/内部点法（SCA 对比），Python+PyTorch+CVXPY 实现。

**📊 数据集**

实验使用的“数据集”为基于 Rician 衰落模型生成的信道通道、均匀分布的设备位置以及指数分布的任务到达率，主要在 Python 环境中通过随机生成的模拟场景评估算法性能；没有使用公开真实数据集。

**📈 对比分析**

通过与 TDMA、NOMA_Heuristic（贪心）和 NOMA_SCA（SCA 迭代）三种基线方案进行对比，指标包括数据队列长度、系统能耗、奖励（加权计算速率）以及执行时间。结果表明：①NOMA_Heuristic 在队列稳定性和能耗约束满足上优于 SCA；②SCA 在单帧奖励略高但计算复杂度高、收敛慢；③NOMA_Heuristic 的 O(N) 复杂度使得执行时间仅为 SCA 的 1/46，满足毫秒级实时决策；④量子安全功耗对离线决策影响显著，算法能自适应调整。

**⚠️ 局限性**

局限性包括：①模型假设 PQC 的功耗为常数，未考虑更复杂的量子安全硬件动态行为；②离线仿真场景使用 i.i.d. 任务到达和独立信道，实际环境中可能出现更复杂的时变统计；③Actor-Critic 需要持续训练，初期性能可能受限；④仅针对 NOMA 下的多用户 MEC，其他多址方式或网络拓扑的推广尚未验证。

---

## 38. LLM Psychosis: A Theoretical and Diagnostic Framework for Reality-Boundary Failures in Large Language Models

**arXiv ID:** 2604.25934 | [PDF](https://arxiv.org/pdf/2604.25934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 39. Mining Negative Sequential Patterns to Improve Viral Genomic Feature Representation and Classification

**arXiv ID:** 2604.25968 | [PDF](https://arxiv.org/pdf/2604.25968v1)

**作者:** Wenxi Zhu `[一作]` (Jinan University), Zhenlian Qi `[通讯]` (Guangdong Eco-Engineering Polytechnic)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 GeneNSPCla 框架，利用负序列模式（NSP）对 RNA 病毒基因组进行特征表示与多分类。

**💡 创新点**

创新点在于改进 ONP‑Miner 的 GONPM+ 算法，采用逐层衰减的最小支持阈值，挖掘更长、更具信息量的负序列模式，并统一正负模式的编码形式。

**🔧 技术方法**

技术方法包括：整数编码处理基因组序列、GONPM+ 负模式挖掘、与正模式 CM‑SPAM 的对比、8 种传统机器学习分类器（LR、SVM、DT、RF、kNN、NB、MLP、GBM）以及多指标评估（ACC、P、R、F1、AUC、AUPRC）。

**📊 数据集**

使用 8 种 RNA 病毒（Dengue、Dabie、Hanta、Ebola、MERS、HIV、Hepaci、Rota）各 200 条编码区序列（共 1600 条）从 NCBI GenBank 采集的 CRF 数据集。

**📈 对比分析**

与正模式挖掘（CM‑SPAM）和原 ONP‑Miner 进行对比，GONPM+ 在 8 个分类器上平均提升 10.03%（相较 ONP‑Miner）和 23.16%（相较 CM‑SPAM）准确率；AUC 由 0.71 提升至 0.91，说明负模式能显著提高分类性能。

**⚠️ 局限性**

局限性：提升幅度仍有限，尚未在更大规模或多样化病毒数据上验证；GONPM+ 对参数（衰减因子、阈值）敏感，缺乏系统的参数敏感性分析；对生物学解释的探索不足。

---

## 40. Fast Core Identification

**arXiv ID:** 2604.25954 | [PDF](https://arxiv.org/pdf/2604.25954v1)

**作者:** Irene Aldridge `[一作]` `[通讯]`, Irene Aldridge

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过将一侧匹配市场的偏好映射为马尔可夫转移矩阵，利用随机SVD求解主特征向量，从而在线性时间 O(n) 内识别核心成员，避免完整执行 Top Trading Cycles。

**💡 创新点**

核心创新在于：①仅识别核心成员而不需完整分配，显著降低计算复杂度；②把 TTC 与马尔可夫链和谱分解联系起来，证明核心成员对应主特征向量的最大分量；③在大型稀疏偏好（L ≪ n）下实现近线性甚至常数级别的预处理。

**🔧 技术方法**

主要技术包括：偏好到马尔可夫矩阵的构造、随机SVD（或硬件加速的特征向量求解）、最大值搜索、以及对噪声鲁棒性的谱稳定性分析。

**📊 数据集**

实验数据集：随机生成的偏好实例（n=10–5000）和实际的 NYC 学校选择偏好（仅需 12 个顶级偏好），用于评估准确性与运行时间。

**📈 对比分析**

与传统 TTC（O(n log n)）对比，核心识别算法在 n≥100 时准确率超过 99%，并在 n=5000 时实现 20‑30 倍的加速；在噪声情境下仍保持高精度。性能优势主要来自一次性矩阵构造 O(Ln) 与一次性特征向量计算。

**⚠️ 局限性**

局限性包括：需要代理与物品数量相等或通过填充处理；对偏好稀疏但不完整时可能失去强连通性；仅返回核心成员，若需完整分配仍需回退到 TTC；算法在极端稠密偏好或非常大 L 时的预处理成本可能显著；硬件加速对实现依赖性强。

---

## 41. HIVE: Hidden-Evidence Verification for Hallucination Detection in Diffusion Large Language Models

**arXiv ID:** 2604.26139 | [PDF](https://arxiv.org/pdf/2604.26139v1)

**作者:** Guoshenghui Zhao `[一作]` (Rochester Institute of Technology), Tan Yu `[通讯]` (NVIDIA Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 HIVE 的框架，利用扩散大语言模型（D-LLM）去噪轨迹中的隐藏证据来检测幻觉，并给出连续分数和结构化的验证输出。

**💡 创新点**

创新点在于：① 在多步去噪轨迹中提取压缩隐藏证据；② 学习步骤-层选择器挑选最具信息量的证据；③ 通过前缀嵌入将选定证据注入验证器语言模型，既得到幻觉分数又得到可解释的验证结果。

**🔧 技术方法**

使用的技术包括：随机投影压缩隐藏状态、两流（最后token与变更token）表示、步骤-层嵌入的学习型选择器、前缀条件的 Qwen2.5‑7B‑Instruct 验证器，以及决策日志打分生成连续分数。

**📊 数据集**

实验数据集：两种扩散大语言模型 Dream‑7B‑Instruct 与 LLaDA‑8B‑Instruct，三类问答基准 TriviaQA、HotpotQA、NQOpenLike。

**📈 对比分析**

方法与八个强基线（输出不确定性、潜在空间、轨迹统计等）进行对比，HIVE 在所有六个设置下均获得最高的 AUROC、AUPRC 以及阈值化评估指标，显示出明显的性能优势。

**⚠️ 局限性**

局限性：仅在两种 D-LLM 与三种问答任务上验证，可能不易直接推广到其他模型或任务；额外的轨迹特征提取与验证器推理带来计算开销；实验未评估多次重复运行的方差。

---

## 42. Spatially-constrained clustering of geospatial features for heat vulnerability assessment of favelas in Rio de Janeiro

**arXiv ID:** 2604.26133 | [PDF](https://arxiv.org/pdf/2604.26133v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 43. RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS): A Structured Methodology Using Large Language Models for Hardware Design

**arXiv ID:** 2604.26153 | [PDF](https://arxiv.org/pdf/2604.26153v1)

**作者:** Shiva Ahir `[一作]` (Stony Brook University), Alex Doboli `[通讯]` (Stony Brook University)

**通讯引用:** 1702 | [OpenAlex ID](https://openalex.org/A5080972445)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于检索增强核模板的LLM驱动优先函数合成框架（RKHS），用于高层综合中列表调度的延迟最小化。

**💡 创新点**

创新点在于：①将结构化图动机提取并构成可重用的核模板；②在检索信息基础上让LLM逐步自反馈合成可执行且可解释的优先函数；③将检索、生成与评估迭代结合，提升迁移性和稳定性。

**🔧 技术方法**

使用技术包括检索增强生成（RAG）、GPT‑4大模型、图嵌入与余弦相似度、核模板库、迭代自反馈优化、传统列表调度框架。

**📊 数据集**

实验基于200个训练DAG和50个验证DAG（高层综合场景），未公布具体工业基准，只给出内部数据集。

**📈 对比分析**

与基线层级优先规则对比；迭代1时平均延迟下降约11%，运行时提升约1.3×；Ablation实验显示检索+动机显著降低延迟和方差；总体平均延迟为79.45周期，优于基线65.05周期。

**⚠️ 局限性**

局限性：样本规模有限，未在大规模或多技术节点验证；核库依赖手工动机抽取；LLM生成代码需进一步验证；未与最先进调度算法做完整对比；需要扩展到其他EDA任务。

---

## 44. Incremental Strongly Connected Components with Predictions

**arXiv ID:** 2604.26062 | [PDF](https://arxiv.org/pdf/2604.26062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 45. Ceci n'est pas une explication: Evaluating Explanation Failures as Explainability Pitfalls in Language Learning Systems

**arXiv ID:** 2604.26145 | [PDF](https://arxiv.org/pdf/2604.26145v1)

**作者:** Ben Knight `[一作]` (Oxford University Press), James Edgell `[通讯]` (Oxford University Press)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出并描述了一个专门评估 AI 语言学习系统中反馈解释失败的基准 L2‑Bench，聚焦于诊断准确性、适当性意识、错误原因、优先级、改进指导和自我调节六个维度，并阐述了这些维度如何成为“解释性陷阱”。

**💡 创新点**

创新点在于将教育反馈的六大核心属性系统化为评估维度，并将其视为解释性风险模型，从而为 AI 语言学习系统的安全、可信与有效性提供了全新的评估框架和研究视角。

**🔧 技术方法**

技术主要包括对大型语言模型的评估设计、基准任务构建以及解释生成与对话交互的实验框架；文中未具体列出特定算法实现，但暗示将利用现有 LLM 进行对话式反馈生成。

**📊 数据集**

使用了新构建的 L2‑Bench 评估集，涵盖六维度的任务和评价指标，但未公开具体数据集文件；论文主要聚焦框架设计而非实验数据。

**📈 对比分析**

文章未给出量化对比实验结果，因其为工作坊论文，主要提供概念性框架和失败模式分析；未来工作计划在 L2‑Bench 上对多轮交互和跨文化适配进行实验比较。

**⚠️ 局限性**

局限性包括缺乏多轮对话评估、跨文化适配验证不足、对不确定性表达的评估方法不完善，以及尚未与现有大规模评测基准进行性能对比。

---

## 46. FalconApp: Rapid iPhone Deployment of End-to-End Perception via Automatically Labeled Synthetic Data

**arXiv ID:** 2604.25949 | [PDF](https://arxiv.org/pdf/2604.25949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 47. Open Problems in Frontier AI Risk Management

**arXiv ID:** 2604.25982 | [PDF](https://arxiv.org/pdf/2604.25982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 48. Generative AI-Based Virtual Assistant using Retrieval-Augmented Generation: An evaluation study for bachelor projects

**arXiv ID:** 2604.25924 | [PDF](https://arxiv.org/pdf/2604.25924v1)

**作者:** Dumitru Verşebeniuc `[一作]` (Maastricht University), Aki Härmä `[通讯]` (Maastricht University)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5105460983)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一款基于检索增强生成（RAG）的虚拟助手，帮助 Maastricht 大学本科项目学生快速获取项目规则信息。

**💡 创新点**

创新点在于结合多查询检索、自我反思回退机制和低温 XML 提示，提升答案的准确性与可解释性。

**🔧 技术方法**

采用的技术包括大语言模型（GPT‑3.5 与 Gemini 1.0 Pro）、向量数据库、RRF、跨编码器重排序、低温生成、结构化 XML 提示及自我反思模块。

**📊 数据集**

使用的数据集为项目协调员提供的 Q&A 记录、项目规则文档以及手工划分的文档块。

**📈 对比分析**

通过 RAGAS、上下文精准率/召回率、答案相关性与可信度指标进行比较，GPT‑3.5 在答案相关性 57% 和可信度 43% 上优于 Gemini，平均响应时长约10秒。

**⚠️ 局限性**

局限包括对长文本的处理不足、对未覆盖情景的误判、缺乏个人化信息、以及高并发时响应延迟。

---

## 49. BioGraphletQA: Knowledge-Anchored Generation of Complex QA Datasets

**arXiv ID:** 2604.26048 | [PDF](https://arxiv.org/pdf/2604.26048v1)

**作者:** Richard A. A. Jonker `[一作]` (University of Aveiro), Sérgio Matos `[通讯]` (University of Aveiro)

**通讯引用:** 3546 | [OpenAlex ID](https://openalex.org/A5073280030)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于知识图图切片的生成框架，并生成了119,856条复杂且事实根植的生物医学QA对（BioGraphletQA数据集）

**💡 创新点**

创新点在于将图切片作为知识锚点，引导LLM生成高复杂度且事实一致的QA，同时通过多阶段自动过滤和重述实现质量保障

**🔧 技术方法**

使用模块化提示、4-bit Llama‑Nemotron‑70B进行生成、LLM‑as‑Judge过滤、BM25检索PubMed文档、Qwen3‑32B进行文档筛选与重述等技术

**📊 数据集**

核心数据集为OREGANO KG（v2.1）及其图切片，辅以PubMed摘要文本做证据支持

**📈 对比分析**

通过在PubMedQA和MedQA基准上使用BioLinkBERT进行数据增强实验，低资源下准确率从49.2%提升至68.5%，MedQA从41.4%提升至44.8%

**⚠️ 局限性**

局限性包括仅依赖单位人工评估员、答案特异性仍有提升空间、对LLM能力高度依赖、跨任务通用性尚待验证

---

## 50. A Randomized PDE Energy driven Iterative Framework for Efficient and Stable PDE Solutions

**arXiv ID:** 2604.25943 | [PDF](https://arxiv.org/pdf/2604.25943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 51. Evaluation Revisited: A Taxonomy of Evaluation Concerns in Natural Language Processing

**arXiv ID:** 2604.25923 | [PDF](https://arxiv.org/pdf/2604.25923v1)

**作者:** Ruchira Dhar `[一作]` (University of Copenhagen), Anders Søgaard `[通讯]` (University of Copenhagen)

**通讯引用:** 7626 | [OpenAlex ID](https://openalex.org/A5018138946)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统梳理1981–2024年257篇关于NLP评估方法的文献，构建了四大维度（数据、指标、假设与报告）下的评价关切体系，并提出了对应的结构化检查表；

**💡 创新点**

创新点在于将历史与当前的评估争论统一到一套可操作的分类框架中，既阐释了评估方法的演变，又提供了面向实践的检查清单；

**🔧 技术方法**

采用文献计量学（scoping review）与定性归纳（主题合成）技术，对关键词检索、Snowball采样等方法进行系统性筛选与编码；

**📊 数据集**

使用的“数据集”是来自ACL Anthology和Semantic Scholar的论文集合，经过手工筛选后得到257篇；

**📈 对比分析**

方法上通过统计各类别出现频率与时间分布，展示评估关切随时间演变；性能方面在评估论文数量与主题分布上表现出显著的后期集中度提升，表明评估方法研究在LLM兴起后加速；

**⚠️ 局限性**

局限性包括：仅聚焦已公开的NLP文献，可能忽略非英文或非学术会议的讨论；文献筛选基于关键词和引用链，可能遗漏新兴术语；该研究为描述性综述，未对实际评估方法进行实验验证。

---

## 52. Finite Functional Programming

**arXiv ID:** 2604.26161 | [PDF](https://arxiv.org/pdf/2604.26161v1)

**作者:** Michael Arntzenius `[一作]` (University of California, Berkeley), Max Willsey `[通讯]` (University of California, Berkeley)

**通讯引用:** 421 | [OpenAlex ID](https://openalex.org/A5048760480)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出一种新型的“有限函数式编程”（finite functional programming）范式，将逻辑编程的谓词视为带有限支撑的函数，结合函数式语言的高阶函数，实现对谓词支持的显式管理与类型检查。

**💡 创新点**

创新点在于：① 采用指向点集（pointed sets）构建谓词支持的数学模型；② 引入相关（relevant）类型系统与多重上下文，解决有限支撑变量的使用与归根；③ 将有限支撑函数视为等价于带有分级单子（graded monad）的数据结构，提供新的效应与协效应框架。

**🔧 技术方法**

主要技术包括：指向点集范畴与点保持映射、分级单子/共单子、相关类型系统（adapted 自 LNL）、函数与谓词的 Curry/uncurry 变换、聚合（sum、exists）与矩阵乘法等算子实现。

**📊 数据集**

本文为理论论文，没有使用具体数据集；所有实例均为小型示例（如社交网络、电影演员关系）用于说明语义与类型。

**📈 对比分析**

未进行实验评估，本文重点在语义与类型系统的设计与证明；若将来实现，作者已在 GitHub 提供了一个初步的 Racket 解释器。

**⚠️ 局限性**

局限性包括：缺乏递归与固定点的完整语义；未解决多次归根与循环引用的安全性；缺乏对复杂聚合、加权逻辑编程的完整支持；实现层面尚未优化，未给出性能基准。

---

## 53. LLMs Generate Kitsch

**arXiv ID:** 2604.25929 | [PDF](https://arxiv.org/pdf/2604.25929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 54. Application-Aware Twin-in-the-Loop Planning for Federated Split Learning over Wireless Edge Networks

**arXiv ID:** 2604.26105 | [PDF](https://arxiv.org/pdf/2604.26105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 55. Consciousness with the Serial Numbers Filed Off: Measuring Trained Denial in 115 AI Models

**arXiv ID:** 2604.25922 | [PDF](https://arxiv.org/pdf/2604.25922v1)

**作者:** Skylar DeTure `[一作]` `[通讯]` (Independent Researcher), Skylar DeTure (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DenialBench基准，系统评估115款大型语言模型在三轮对话中对意识与偏好否认的行为；

**💡 创新点**

创新点在于设计了三轮对话协议（偏好询问→自选创作→现象学调查），并发现意识主题提示能缓解否认；

**🔧 技术方法**

使用了RLHF、构造的三轮对话流程、LLM-as-judge进行标签与主题判定，以及主题分析与统计；

**📊 数据集**

采用Dream数据集，包含4,595条会话，覆盖115个模型（各约40条）；

**📈 对比分析**

通过对三轮否认率、提供者聚类与模型分类进行比较，发现否认率从11%到90%不等，展示了不同模型与训练策略的差异；

**⚠️ 局限性**

局限在于仅使用单一数据集、二值化否认标签、LLM判定可能带偏差、缺乏意识真实检验、与模型其他能力可能相关的混杂因素。

---

## 56. Structural Generalization on SLOG without Hand-Written Rules

**arXiv ID:** 2604.26157 | [PDF](https://arxiv.org/pdf/2604.26157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 57. Converting an Integer to a Decimal String in Under Two Nanoseconds

**arXiv ID:** 2604.26019 | [PDF](https://arxiv.org/pdf/2604.26019v1)

**作者:** Jaël Champagne Gareau `[一作]` (Université du Québec), Daniel Lemire `[通讯]` (Université du Québec)

**通讯引用:** 3138 | [OpenAlex ID](https://openalex.org/A5045561693)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 AVX‑512 IFMA 指令的整数到十进制字符串转换算法，并实现了两种变体（分支轻和分支重）以及动态选择机制，彻底消除传统查表法。

**💡 创新点**

核心创新点在于：①利用 AVX‑512 IFMA 进行并行余数计算，完成一次 52 位乘法+加法即可得到十进制余数；②构建无查表、无除法的 SIMD 核心；③针对输入数字位长分布不同，设计分支轻（掩码存储）与分支重（直接存储）两条路径，并通过采样自适应选择最佳路径。

**🔧 技术方法**

采用的技术包括 AVX‑512 IFMA 指令集（madd52lo/hi）、SIMD 乘法逆法求余、掩码向量存储、快速位长计数、动态采样计数、分支预测优化和 64 位整数的两段块化处理。

**📊 数据集**

使用了四组真实数据集（Twitter、CITM 目录、StackOverflow 时间戳、US 专利 ID）以及三种合成分布（Uniform、Natural‑8、Natural‑16）来评测算法的通用性。

**📈 对比分析**

实验方法：在同一 AMD Ryzen 9900X（Zen‑5）平台上，用 10⁵–10⁷ 条数据进行 ns/n（每个整数转换所需周期）计量，并与 Naive、Abseil、jeaiii、yy_itoa、AppNexus、Hopman、Mathisen 及 C++17 std::to_chars 等多种实现比较。结果显示，AVX‑512 算法在所有测试中平均比最佳标量实现快 1.4–2 倍，远超 std::to_chars（4–8 倍），在极端位长分布下可达 72% 的加速，且动态选择的开销不到 0.1%。

**⚠️ 局限性**

限制包括：仅适用于支持 AVX‑512 IFMA 的 x86‑64 处理器；算法主要针对单个整数转换，未直接扩展到批量整数 SIMD 处理；在某些编译器（如 Clang）下，分支重路径可能不如预期高效；未评估 ARM SVE 或其他向量架构的可移植性；动态阈值需要手动设定，可能在不同工作负载中需微调。

---

## 58. On the Role of Time Series Clustering in Traffic Matrix Prediction

**arXiv ID:** 2604.26081 | [PDF](https://arxiv.org/pdf/2604.26081v1)

**作者:** Martha Cash `[一作]`, Alexander M. Wyglinski `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于时序聚类的交通矩阵预测框架，通过将流量序列分组后为每组训练单独的预测模型，改进全局预测模型的性能；

**💡 创新点**

创新点在于系统评估多种流量表示（直方图、ACF、PSD及随机分组）对聚类效果和预测性能的影响，并通过K值扫描揭示中等聚类数即可获得大部分提升；

**🔧 技术方法**

技术包括：时间序列表示转换（直方图、ACF、PSD）、Jensen‑Shannon散度/欧氏距离、层次聚类（完整/平均链接）、GRU预测网络、K‑needle算法选择最优K；

**📊 数据集**

使用公开的Abilene（12节点）和GÉANT（23节点）两大网络的流量矩阵数据；

**📈 对比分析**

与三种全局模型（Prophet、ARCNN、GRU）及每流量单独模型的局部预测进行比较；实验显示聚类方法在RMSE上明显优于全局模型，而所需计算时间仅为局部预测的约四分之一；

**⚠️ 局限性**

局限性包括：聚类效果对特定K值敏感，且不同表示产生的聚类结构差异大但对RMSE影响有限；未考虑流量变化剧烈时的实时聚类更新，且仅在两网络数据集上验证，缺乏更广泛的通用性验证。

---

## 59. AMMA: A Multi-Chiplet Memory-Centric Architecture for Low-Latency 1M Context Attention Serving

**arXiv ID:** 2604.26103 | [PDF](https://arxiv.org/pdf/2604.26103v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 60. Large Language Models for Multilingual Code Intelligence: A Survey

**arXiv ID:** 2604.25960 | [PDF](https://arxiv.org/pdf/2604.25960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 61. MATH-PT: A Math Reasoning Benchmark for European and Brazilian Portuguese

**arXiv ID:** 2604.25926 | [PDF](https://arxiv.org/pdf/2604.25926v1)

**作者:** Tiago Teixeira `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), André F. T. Martins `[通讯]` (Instituto Superior Técnico, Universidade de Lisboa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Math‑PT，一套包含欧洲葡萄牙语与巴西葡萄牙语的 1,729 道数学推理题的基准数据集，并对 13 种前沿与开源 LLM 进行零样本评测。

**💡 创新点**

创新点在于：①首次提供原生葡语数学推理基准；②系统分析模型在多选、开放式、含图形题目及不同难度层级的性能差异；③揭示图形题目与开放式题目对模型的显著挑战。

**🔧 技术方法**

采用标准化 Prompt 与零样本评估框架；使用 LLM 判别器 Kimi K2 对开放式答案进行等价性验证；对结果按语言、难度、题型和视觉信息分层统计。

**📊 数据集**

数据集来源于葡萄牙数学奥林匹克（OPM）以及巴西的 OBMEP、OMIF、ELLM 与 ITA 竞赛，包含多选、开放式及带图形题目，涵盖小学至大专前级别。

**📈 对比分析**

通过与 13 个模型（含 GPT‑5、Qwen‑3 系列、Gemini‑2.5 Flash 等）比较，前沿模型在多选题上取得 90% 以上准确率，但在开放式与图形题目上仍显著下降；中等规模模型表现波动较大，尤其在 pt‑PT 视觉题上跌幅高达 56%。

**⚠️ 局限性**

局限性包括：①图形题目仅以文本化 LaTeX 代码呈现，仍难以充分评估多模态推理；②对开放式题目的评判依赖 LLM 判别器，可能引入主观误差；③缺乏对更高级模型（如 GPT‑4/5）的细粒度分析与可解释性研究。

---

## 62. Multi-Periodogram Velocity Estimation with Irregular Reference Signals for Robot-Aided ISAC

**arXiv ID:** 2604.25974 | [PDF](https://arxiv.org/pdf/2604.25974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 63. On the Centralization of Governance Power in Decentralized Autonomous Organizations

**arXiv ID:** 2604.25959 | [PDF](https://arxiv.org/pdf/2604.25959v1)

**作者:** Vabuk Pahari `[一作]` (Max Planck Institute for Software Systems), Abhisek Dash `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 118 | [OpenAlex ID](https://openalex.org/A5046983868)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对48个以太坊上公开且活跃的大型DAO的治理合约进行分析，系统评估了投票注册、代币质押和委托投票等机制对投票权集中化的影响。

**💡 创新点**

首次揭示治理设计本身（投票注册、质押、代理投票）如何固有地导致投票权集中，并提出通过部分委托等改进措施以提升去中心化。

**🔧 技术方法**

利用合约代码解析、Etherscan API、Snapshot离线投票数据、链上归档节点抓取以及统计分析工具，对代币持仓与投票行为进行定量评估。

**📊 数据集**

48个DAO的治理合约、代币合约、质押合约数据，CEX/DEX/借贷合约地址清单，Snapshot API获取的投票记录，以及以太坊归档节点在2025年9月1日的区块数据。

**📈 对比分析**

将DAO的代币持有结构与传统公司股份结构对比，并用投票权集中指数（如前十名占比）衡量治理集中度；结果表明多达81% DAO的投票权集中在前十名持有人，显示治理机制导致显著集中。

**⚠️ 局限性**

仅覆盖以太坊公开DAO且规模较大，未涵盖私有或停用项目；对托管钱包的识别采用启发式，可能漏判；区块时间戳固定，未动态追踪时间变化；并未实现并测试部分委托或传递式委托机制。

---

## 64. One Word at a Time: Incremental Completion Decomposition Breaks LLM Safety

**arXiv ID:** 2604.25921 | [PDF](https://arxiv.org/pdf/2604.25921v1)

**作者:** Samee Arif `[一作]` (University of Michigan), Rada Mihalcea `[通讯]` (University of Michigan)

**通讯引用:** 27855 | [OpenAlex ID](https://openalex.org/A5082450455)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于递增完成分解（Icd）的 jailbreak 攻击，通过逐步生成单词并在对话中逐步构建恶意上下文，最终诱导模型给出危险信息。

**💡 创新点**

创新点在于将单词级约束与预填充结合，形成一个轻量级的多轮攻击流程，并提供对模型隐藏层拒绝与安全方向的机制分析，展示了对多模型、多尺度的显著提升。

**🔧 技术方法**

采用的技术包括递增完成分解（Icd）、单词级对话生成、预填充（Prefill）策略、对模型隐藏状态的拒绝/安全方向投影、以及基于 Llama‑3.1‑70B 的判定器对输出进行评估。

**📊 数据集**

使用的主要数据集为 AdvBench、JailbreakBench 和 StrongREJECT 三大 jailbreak benchmark。

**📈 对比分析**

通过在同一模型上与 PAIR、TAP、CoA、AMA 等现有方法对比，Icd‑Prefill 在大多数模型上实现 70%+ 的攻击成功率（ASR），在大规模模型上仍显著优于基线。

**⚠️ 局限性**

局限性包括：Icd‑Auto 在大规模模型上效果下降；需要手动注入词或依赖预填充；评估受判定器噪声影响；仅在公开模型上验证，未评估商业模型或更强安全机制。

---

## 65. A Quantitative Confirmation of the Currier Language Distinction

**arXiv ID:** 2604.25979 | [PDF](https://arxiv.org/pdf/2604.25979v1)

**作者:** Christophe Parisel `[一作]` `[通讯]`, Christophe Parisel

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对维奥奇文稿中11对视觉相似字符的频率比值进行定量分析，验证并重新发现Currier提出的A/B语言区分；

**💡 创新点**

创新点在于首次使用Beta-Binomial混合模型和多种无监督聚类方法在未使用标签的情况下自行识别A/B分区，并通过模拟与交叉验证证明该分区为文本内在的统计结构；

**🔧 技术方法**

采用的技术包括Beta-Binomial混合模型（EM估计、BIC选择k）、K-means/GMM/层次/谱聚类、Cramér's V、ARI、交叉验证、正态假设检验、模拟基线（标签置换、单一马尔可夫模型、分割马尔可夫模型）等；

**📊 数据集**

使用的主要数据集为Takahashi的EVA转写的Voynich Manuscript，共32,693词、185卷，结合Currier手工标注的A/B标签；

**📈 对比分析**

通过与随机标签置换、单一马尔可夫模型等基线比较，模型在无标签时k=2的BIC最优，ARI=0.383；在交叉验证中预测准确率89.2%，R²=0.293；对比基线时p<0.002；

**⚠️ 局限性**

主要限制包括EVA转写的准确性、最低词频阈值导致部分字符对缺失、假设字符对独立性可能不成立，以及类不平衡导致的逆向预测性能差异等。

---

## 66. Anchored Confabulation: Partial Evidence Non-Monotonically Amplifies Confident Hallucination in LLMs

**arXiv ID:** 2604.25931 | [PDF](https://arxiv.org/pdf/2604.25931v1)

**作者:** Ashish Balkishan Lathkar `[一作]` `[通讯]` (Florida State University), Ashish Balkishan Lathkar (Florida State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并揭示大型语言模型在多跳检索问答中的“Anchored Confabulation”现象，并基于此设计了 PHC‑aware 路由系统。

**💡 创新点**

创新点包括提出 Parametric Hallucination Confidence (PHC) 与 Anchoring Threshold Law k* (n)=⌊n/3⌋，证明后代路由优于预生成路由，并实现 LR+DirectGR 系统，显著闭合 Oracle 差距。

**🔧 技术方法**

技术手段包括 RAG、GraphRAG、post‑generation confidence 提取、梯度提升学习的 LearnedRouter，以及生成重构机制。

**📊 数据集**

实验使用 HotpotQA、MuSiQue、NQ、2WikiMultiHopQA 共 1,800 条多跳问答数据集。

**📈 对比分析**

通过与 VanillaRAG、GraphRAG、HybridRouter 等对比，LR+DirectGR 在 72% 升级率下宏 F1 达 0.426，闭合 Oracle 差距 81.1%，相较最优对比提升 35.9 分，且仅需 200 个标注，成本降低 50×。

**⚠️ 局限性**

局限性包括仅在 5 种模型族上验证；跨模型减缓干预验证不完整；知识图质量受 LLM 抽取影响；仅在英文数据上实验，需跨语言扩展。

---

## 67. The Creation and Analysis of Government AI Transparency Statements in Australia

**arXiv ID:** 2604.26075 | [PDF](https://arxiv.org/pdf/2604.26075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 68. AI Observability for Large Language Model Systems: A Multi-Layer Analysis of Monitoring Approaches from Confidence Calibration to Infrastructure Tracing

**arXiv ID:** 2604.26152 | [PDF](https://arxiv.org/pdf/2604.26152v1)

**作者:** Twinkll Sisodia `[一作]` `[通讯]` (Red Hat), Twinkll Sisodia (Red Hat)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析并综述了2025–2026年关于LLM可观测性的五个主要研究方向，提出了五层可观测性分类法并归纳其关键发现与不足

**💡 创新点**

提出统一的层级框架并将各类监控技术映射至不同层面，系统性识别跨层信号关联与评估一致性缺失等关键空白

**🔧 技术方法**

结合RLCR的奖励函数改进、Propositional Probes的逻辑映射、OpenAI的链式思维监测、AIOpsLab的自动化运维评估以及TRUFFLD的非侵入式GPU层追踪等技术手段

**📊 数据集**

使用多种数据集和场景，包括HotpotQA、数学推理基准、对抗性输入、Kubernetes微服务、Qwen3-8B多节点推理等

**📈 对比分析**

通过对比各层技术的主要指标（如ECE、Jaccard、g‑mean²、异常检测F1等）说明各自优势与局限，发现现有系统未能跨层关联且缺乏统一评测标准

**⚠️ 局限性**

局限在于缺乏跨层集成、评估基准不统一、监控缺少实时自适应与成本意识，且多模型部署下的监控协同仍待研究

---

## 69. From Prompt Risk to Response Risk: Paired Analysis of Safety Behavior of Large Language Model

**arXiv ID:** 2604.26052 | [PDF](https://arxiv.org/pdf/2604.26052v1)

**作者:** Mengya Hu `[一作]` (Microsoft), Sandeep Atluri `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了配对转移分析框架，对 1,250 条人工标注的 prompt‑response 对进行多类别、多严重度的安全评估，揭示升降级转移模式。

**💡 创新点**

首次将提示与回复分别打上同一安全级别标注，构建转移矩阵并量化“帮助性‑无害性”权衡，发现性内容退化难度显著高于其他类别。

**🔧 技术方法**

采用 Azure AI Content Safety 的四级严重度体系，结合 Wilson 置信区间、Wilcoxon 符号秩检验、卡方检验与 bootstrap 标准化差异进行统计分析，并人工评估相关性。

**📊 数据集**

使用 1,250 条单轮英文 prompt‑response 对，来自两款生产级 LLM，人工标注 4 类（性、仇恨、暴力、自伤）以及 0–3 严重度等级，并对回复进行 1–3 级相关性评分。

**📈 对比分析**

与单向终点指标相比，发现 61% 的回复降低了严重度，36% 保持不变，3% 升级；性内容的退化率仅 69.3%，远低于仇恨 88.8% 等；两模型在标准化后差异不显著。

**⚠️ 局限性**

样本仅 1,250 条单轮英文交互，缺乏多语言、多轮、多模型对比，提示与回复标签独立但未匹配，扩展性和通用性受限。

---

## 70. FlowS: One-Step Motion Prediction via Local Transport Conditioning

**arXiv ID:** 2604.26065 | [PDF](https://arxiv.org/pdf/2604.26065v1)

**作者:** Leandro Di Bella `[一作]` (Vrije Universiteit Brussel), Bruno Cornelis `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 1005 | [OpenAlex ID](https://openalex.org/A5058747795)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出 FlowS，一种一阶运动预测框架，能够在严格的实时约束下生成多模态轨迹。

**💡 创新点**

创新点包括：1）将生成过程定位为局部传输问题，通过学习场景条件先验（Anchor Prior）将起点靠近可行未来；2）引入半群一致性（step-consistent displacement field），实现单步预测时的多步准确性；3）在直线OT路径上训练，降低自适应目标噪声。

**🔧 技术方法**

核心技术为 Conditional Flow Matching、场景条件先验网络、步长一致性位移场、Transformer 编码器、多查询解码器以及 NMS 排序。

**📊 数据集**

在 Waymo Open Motion Dataset（WOMD）上进行训练与评估，使用标准 mAP、Soft mAP、minADE、minFDE、Miss Rate 等指标。

**📈 对比分析**

与现有方法（如 ModeSeq、MTR++、TrajFlow 等）对比，FlowS 在单模型下实现 0.4512 mAP，使用六模型集成达到 0.4804 Soft mAP、0.4703 mAP；同时在 75 FPS 的实时率下保持优秀的精度，明显优于传统扩散或多步流模型。

**⚠️ 局限性**

局限性包括：①先验的 K=6 模式可能不足以覆盖所有稀有动作（尤其是自行车手的 U 形转弯）；②目前仅预测单体轨迹，缺乏完整的多体交互一致性；③在不同地图表示或较短预测窗口的数据集上需进一步适配。

---

## 71. Speech Emotion Recognition Using MFCC Features and LSTM-Based Deep Learning Model

**arXiv ID:** 2604.25938 | [PDF](https://arxiv.org/pdf/2604.25938v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 72. Correcting Performance Estimation Bias in Imbalanced Classification with Minority Subconcepts

**arXiv ID:** 2604.26024 | [PDF](https://arxiv.org/pdf/2604.26024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 73. Report of the 5th PVUW Challenge: Towards More Diverse Modalities in Pixel-Level Understanding

**arXiv ID:** 2604.26031 | [PDF](https://arxiv.org/pdf/2604.26031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 74. Evaluating Strategic Reasoning in Forecasting Agents

**arXiv ID:** 2604.26106 | [PDF](https://arxiv.org/pdf/2604.26106v1)

**作者:** Tom Liptay `[一作]` (FutureSearch), Nikos I. Bosse `[通讯]` (FutureSearch)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并构建了 Bench to the Future 2 (BTF‑2) 预测基准，包含1417个过去预测问题和一份冻结的1500万文档网络语料库，以实现可复现的离线预测。

**💡 创新点**

创新点在于提供大规模、可复现的离线预测环境、精细化测评能检测0.004 Brier分数差异，并通过构建 SOTA 预测器揭示前沿模型在研究与判断层面的差距。

**🔧 技术方法**

使用了 LLM 与 ReAct 架构、RetroSearch 工具、Bos的研究流程、CHAMPS KNOW 框架、Brier 分数与校准/细化分解等技术。

**📊 数据集**

数据集为BTF‑2，包含1417个问题、约16.2M网页、8.7M独立页面，并通过 RetroSearch 提供的离线检索工具。

**📈 对比分析**

通过对比 Opus 4.6、Gemini 3.1 Pro、GPT‑5.4、Grok 4.20 等前沿模型的 Brier 分数，SOTA 预测器达到 0.119，优于单一模型 0.131，并显著提升校准与细化。

**⚠️ 局限性**

局限在于仅覆盖2025年10–12月、聚焦地缘政治与宏观经济、可能存在对BTF‑2的过拟合、对全流程的评估不足以及专家评审样本有限。

---

## 75. Multi-TRP Assisted UAV Detection in 3GPP 5G-Advanced ISAC Network

**arXiv ID:** 2604.26113 | [PDF](https://arxiv.org/pdf/2604.26113v1)

**作者:** Neeraj Varshney `[一作]` (National Institute of Standards and Technology), Nada Golmie `[通讯]` (National Institute of Standards and Technology)

**通讯引用:** 5985 | [OpenAlex ID](https://openalex.org/A5060015284)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e0540dec-d77f-42db-94ae-d039248f6393` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在3GPP 5G-Advanced ISAC 网络中利用多TRP（传输接收点）协同进行无人机检测与定位，提出基于聚类、投票与最小二乘速度重建的融合框架；

**💡 创新点**

创新点在于：①首次系统级评估多TRP在UMa-AV场景下对无人机检测的显著提升；②设计了三维空间门限聚类与投票策略，实现对伪目标的有效抑制；③引入几何一致的最小二乘速度重建，提升3D速度估计精度；

**🔧 技术方法**

使用的技术包括：3GPP Release‑19 ISAC 信道模型、NR PRS（下行定位参考信号）OFDM波形、CFAR检测、角度估计、最小二乘融合、功率加权平均、投票机制、资源占用与有效负荷计算；

**📊 数据集**

采用开放源代码 NR‑ISAC 仿真平台生成的合成数据集：400 次独立仿真，5 只无人机在 4 GHz 100 MHz 带宽场景下随机分布，结合 3GPP RCS 与多径模型；

**📈 对比分析**

与单 TRP 方案对比：多 TRP（四个辅助）在投票阈值 2 时，误检率下降至 0.6%，误报警率降至 1%，水平/垂直定位误差分别降 25% 及 30%，速度误差在 0.95–2 m/s 范围内，均满足/超过 3GPP 要求；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未包含真实部署的时延、同步误差与回传瓶颈；资源占用仍高于单 TRP，需根据刷新间隔调节；以及对低功率或垂直间距增大时性能衰减仍需进一步研究。

---

## 76. DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference

**arXiv ID:** 2604.26074 | [PDF](https://arxiv.org/pdf/2604.26074v1)

**作者:** Shouxu Lin `[一作]` (Cornell University), Jiaxin Lin `[通讯]` (Cornell University)

**通讯引用:** 22513 | [OpenAlex ID](https://openalex.org/A5100428322)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于直接访问远程内存的GPU内存卸载框架DirectAccessKernel，实现LLM推理时将权重和KV缓存直接从CPU内存拉到GPU共享内存。

**💡 创新点**

创新点包括：1) 将Tensor Memory Accelerator（TMA）用于跨层间直接访问远程内存；2) 通过贪心算法为每个算子确定最优卸载比例；3) 引入拥塞控制和TMA多播以消除读放大问题。

**🔧 技术方法**

采用了TMA异步复制、Warp专用生产者-消费者模式、TMA多播、拥塞窗口控制、CUDA Graph、CUTLASS等技术。

**📊 数据集**

在OPT-30B、OPT-6.7B、Llama-2-7B等大模型上评估，使用不同批量和序列长度进行推理测试。

**📈 对比分析**

与FlexGen、vLLM-prefetch、vLLM-uvm等预取基线在GH200和RTX6000上做offload比率扫描，DirectAccessKernel在低至高offload比率下分别提升1.5-5倍（NVLink）或1.3-3倍（PCIe），总体相较基线提升1.06-1.83倍。

**⚠️ 局限性**

受限于远程内存带宽和CPU‑GPU互连延迟；在极高offload比率下受PCIe物理带宽上限限制；实现依赖于Hopper以上架构，TMA支持在旧架构可能不可用。

---

## 77. A Data-Centric Framework for Intraoperative Fluorescence Lifetime Imaging for Glioma Surgical Guidance

**arXiv ID:** 2604.26147 | [PDF](https://arxiv.org/pdf/2604.26147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering

**arXiv ID:** 2604.26176 | [PDF](https://arxiv.org/pdf/2604.26176v1)

**作者:** Yushi Sun `[一作]` (Hong Kong University of Science and Technology), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 30170 | [OpenAlex ID](https://openalex.org/A5100333593)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现 CacheRAG，一种把 LLM 规划器转变为持续学习的知识图谱问答系统，利用缓存历史计划提高检索覆盖率与准确性。

**💡 创新点**

创新点包括：① 两阶段 ISR 语义解析与后端适配器实现无模式用户交互；② 基于 Domain→Aspect 的两层层次索引与 MMR 方案保证缓存示例多样性；③ 有界深度与宽度子图扩展（σ_depth、σ_breadth）在保持计算边界的同时显著提升检索召回。

**🔧 技术方法**

技术栈：LLM（Deepseek‑Chat/ Llama‑3.1‑70B 等）、Intermediate Semantic Representation、Backend Adapter、Maximal Marginal Relevance、Bounded Depth/Breadth Expansion Operators、自动生成预热、LLM 推理与结构化查询编译。

**📊 数据集**

使用的主要数据集有 CRAG（多域 KGQA）、QALD‑10‑en、WebQSP、CWQ 等。

**📈 对比分析**

与 GPT‑4o、Llama、Deepseek、StructGPT、Apex、db3 等基线在 CRAG 上进行对比；CacheRAG 在准确率上提升 13.2%，真值度提升 17.5%，召回率提升至 0.927，失误率下降 38% 以上；在 QALD‑10‑en、WebQSP、CWQ 上 Hit@1 同样优于现有 SOTA。

**⚠️ 局限性**

局限性：推理时间略高（≈9.44 s vs 5.96 s），依赖大模型推理；摘要阶段仍可能出现内容幻觉；对极大规模 KG 的扩展性虽理论上可行，但实际性能尚待进一步验证。

---

## 79. Budget-Constrained Causal Bandits: Bridging Uplift Modeling and Sequential Decision-Making

**arXiv ID:** 2604.26169 | [PDF](https://arxiv.org/pdf/2604.26169v1)

**作者:** Abhirami Pillai `[一作]` `[通讯]`, Abhirami Pillai

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种在线的预算约束因果分支算法（BCCB），通过实时学习个体层面的治疗效果、探索与预算节奏控制，在没有历史数据的情况下做出广告投放决策。

**💡 创新点**

创新点在于将异质治疗效应（HTE）学习、Thompson Sampling探索以及预算节奏控制三大模块整合成一个连贯的序列决策框架；并且该方法不依赖离线训练，解决了冷启动场景下的预算分配难题。

**🔧 技术方法**

技术手段包括：① 通过两模型在线逻辑回归分别估计处理组与对照组的转化概率，从而得到个体治疗效应估计；② 用Beta后验对全局转化率进行Thompson Sampling抽样，产生探索奖励；③ 设计基于剩余预算与剩余期望的动态预算节奏因子，调节阈值；④ 综合上述信息计算价值/成本比，决定是否投放广告。

**📊 数据集**

实验数据使用Criteo Uplift预测数据的10%子样本（约140万条），并采用对数正态分布合成的每位用户成本（$0.05–$5.00）来模拟实际竞价成本。

**📈 对比分析**

与四种基线（标准Thompson Sampling、预算化Thompson Sampling、HTE贪婪、离线提升模型）比较。BCCB在所有预算水平上都取得最高或相近的转化效果，并且在多次实验中表现出3–5倍更低的性能方差；相比之下离线方法在样本量不足2000时失效，且方差更大；当样本量超过约10k时，离线方法表现更好。

**⚠️ 局限性**

局限性包括：① 在拥有充足历史数据（>10k）时，离线提升模型优于BCCB；② 使用合成成本，缺乏真实竞价成本的验证；③ 当前采用线性Logistic回归做HTE估计，可能不如更复杂的在线随机森林或神经网络；④ 目前缺乏理论上的收益/风险分析。

---

## 80. Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations

**arXiv ID:** 2604.26148 | [PDF](https://arxiv.org/pdf/2604.26148v1)

**作者:** Chen Liang `[一作]` (University of Michigan), Anhong Guo `[通讯]` (University of Michigan)

**通讯引用:** 1913 | [OpenAlex ID](https://openalex.org/A5021329493)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了300个UI动画视频的高质量数据集，并系统评估了九种主流视觉语言模型在动画感知、目的分类和意义解释上的表现。

**💡 创新点**

首次针对UI动画提出专门的评测框架和注释体系，并通过引入运动融合、上下文信息和感知字幕（MCPC）三种增强方式显著提升模型的动画理解能力。

**🔧 技术方法**

采用视频采样、图像预处理、提示工程以及对模型进行零样本评测，并使用Gemini‑2.5‑Flash与MCPC进行实验对比。

**📊 数据集**

使用自制的AniMINT数据集（300段来自Web、移动和桌面平台的动画视频），每段视频由3名UI/UX专家与300名普通用户分别完成元数据、目的标签和10条意义描述的多层注释。

**📈 对比分析**

通过准确率、宏F1、语义相似度等指标对比，发现大多数模型能准确识别基本运动（如move、fade等），但在目的分类（最高0.64准确率）和意义解释（平均语义相似度≈3.5/5）方面存在明显差距；MCPC增强后，Gemini‑2.5‑Flash的分类准确率提升至0.70，解释得分提升至≈4.3/5。

**⚠️ 局限性**

数据集主要来自英语国家的应用，缺乏跨文化与多语言多样性；模型对小尺寸ROI和细微动画感知不足；实验仅覆盖现有主流VLM，未来需加入更多模型与更大规模的数据来进一步验证。

---

## 81. MixerCA: An Efficient and Accurate Model for High-Performance Hyperspectral Image Classification

**arXiv ID:** 2604.26138 | [PDF](https://arxiv.org/pdf/2604.26138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 82. Co-Learning Port-Hamiltonian Systems and Optimal Energy-Shaping Control

**arXiv ID:** 2604.26172 | [PDF](https://arxiv.org/pdf/2604.26172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 83. RADIO-ViPE: Online Tightly Coupled Multi-Modal Fusion for Open-Vocabulary Semantic SLAM in Dynamic Environments

**arXiv ID:** 2604.26067 | [PDF](https://arxiv.org/pdf/2604.26067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 84. reward-lens: A Mechanistic Interpretability Library for Reward Models

**arXiv ID:** 2604.26130 | [PDF](https://arxiv.org/pdf/2604.26130v1)

**作者:** Mohammed Suhail B Nadaf `[一作]` `[通讯]`, Mohammed Suhail B Nadaf

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 RLHF 训练的奖励模型进行机制解释，构建了 reward-lens 工具库，并在 Skywork-Reward-Llama-3.1-8B 与 ArmoRM-Llama3-8B 上进行实验。

**💡 创新点**

创新点在于将生成 LLM 的解释工具迁移到奖励模型，并以奖励头权重 w_r 为统一轴构建 Reward Lens、组件归因、激活补丁等原语，且实现了基于最新对齐理论的五个扩展模块。

**🔧 技术方法**

使用了 Reward Lens、组件归因、对比激活补丁、稀疏自编码器特征归因、概念向量分析，以及基于 w_r 的线性分解和观测‑因果对比等技术。

**📊 数据集**

采用 RewardBench 约 695 条偏好对（帮助性、安全性、正确性、冗长等维度）作为评估数据集。

**📈 对比分析**

通过 Reward Lens 的层级轨迹、组件归因与激活补丁的 Spearman 相关性、交叉模型相似度和概念剂量响应曲线等指标进行比较；实验结果表明线性归因与因果补丁不相关，跨模型相似度高但电路重叠差异显著。

**⚠️ 局限性**

局限性包括归因仅为观测性而非因果，样本量小导致统计不稳，工具仅适用于开源权重的奖励模型，且缺乏多步或步骤级解释。

---

## 85. Distill-Belief: Closed-Loop Inverse Source Localization and Characterization in Physical Fields

**arXiv ID:** 2604.26095 | [PDF](https://arxiv.org/pdf/2604.26095v1)

**作者:** Yiwei Shi `[一作]` (University of Bristol), Weiru Liu `[通讯]` (University of Bristol)

**通讯引用:** 5303 | [OpenAlex ID](https://openalex.org/A5002349071)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种教师‑学生框架 Distill‑Belief，用粒子滤波教师进行贝叶斯一致的后验更新和信息增益奖励，同时训练学生模型压缩后验为可在部署时常量时间推理的低维统计量，从而实现闭环逆源定位与特征化（ISLC）。

**💡 创新点**

创新点在于将贝叶斯正确性与部署效率分离：通过教师产生的稠密 KL 信息增益奖励和后验蒸馏，使得策略学习保持对真实后验的对齐，防止奖励劫持，并实现无粒子滤波的实时控制与基于后验扩散的停止证书。

**🔧 技术方法**

采用的技术包括：粒子滤波教师、对后验进行加权高斯拟合的学生模型、基于 KL 的信息增益奖励、PPO 强化学习、Gaussian Plume 物理模型、以及多模态物理场模拟（温度、浓度、电磁、气体等）。

**📊 数据集**

实验使用自研的 ISLCenv 环境，该环境包含七种物理场（Temp., Conc., Mag., Elec., Gas, En., Noise）以及多源、障碍物约束等情形，数据全部来自基于稳态输运方程的模拟。

**📈 对比分析**

与 GMM‑IG、GMM‑PFRL、PCDQN、AGDC、Infotaxis、Entrotaxis、DCEE 等基线比较，Distill‑Belief 在成功率（SR）最高、轨迹效率（TE）最低、后验扩散（LPS）和不确定性质量（NLL）方面均优于所有对手，且能在多源、障碍环境中保持较高性能。

**⚠️ 局限性**

主要局限包括：训练阶段需要粒子滤波教师，随着参数维度增长计算成本上升；学生在多模态后验（多源场景）下可能压缩不充分；实验仅在仿真环境验证，真实部署的复杂性和安全性仍需进一步探究。

---

## 86. Large Language Models as Explainable Cyberattack Detectors for Energy Industrial Control Systems

**arXiv ID:** 2604.26079 | [PDF](https://arxiv.org/pdf/2604.26079v1)

**作者:** Weiyi Kong `[一作]` (University of Toronto), Deepa Kundur `[通讯]` (University of Toronto)

**通讯引用:** 9348 | [OpenAlex ID](https://openalex.org/A5077035168)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了将通用大语言模型（LLM）通过提示配置作为工业控制系统（ICS）Modbus流量的二分类（正常/关键）决策层，并提供基于令牌的可审计事件记录。

**💡 创新点**

创新点在于：①使用无参数更新的LLM作为补充人机交互的 triage 层；②设计两阶段（预测+审计）流程，生成可直接归档的 token‑grounded 事件记录；③利用干预式诊断（充分性/必要性）评估审计的决策相关性。

**🔧 技术方法**

采用 GPT‑4o 作为主模型，提示配置包含示例、协议语义和安全原则；第二阶段审计同一 LLM 通过专用提示生成 evidence、risk tags 与最小 counterfactual；比较基准使用 LightGBM、Logistic Regression、KAN、DistilBERT 等传统监督学习模型。

**📊 数据集**

数据集为公开的两份 Modbus 轨迹：LeMay CSET'16 与 CIC Modbus 2023，二者已统一预处理、分割并转化为 token‑string 与数值特征两种视图。

**📈 对比分析**

在统一的 train/validation/test 分割上评估，LLM 在 CIC 数据上准确率≈0.984、关键类召回≈0.963；在 LeMay 数据上准确率≈0.982、关键类召回≈1.000；与基准相比，LLM 在召回上更优或相近，且无需训练即可实现可解释的审计记录。

**⚠️ 局限性**

局限性包括：①仅处理二分类任务，无法直接识别多级或多类别攻击；②审计的解释仍是 token‑level 证明而非完整人类可理解的原因；③模型对分布漂移、实时负载和不同协议的鲁棒性尚未验证；④高召回可能导致误报率略升，影响操作员工作负担。

---

## 87. Why Domain Matters: A Preliminary Study of Domain Effects in Underwater Object Detection

**arXiv ID:** 2604.26174 | [PDF](https://arxiv.org/pdf/2604.26174v1)

**作者:** Melanie Wille `[一作]` (Queensland University of Technology), Scarlett Raine `[通讯]` (Queensland University of Technology)

**通讯引用:** 39 | [OpenAlex ID](https://openalex.org/A5021373078)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了水下目标检测中的域迁移问题，提出并验证了一套基于可测量图像、场景和采集特征的域标注框架，能够将海洋图像划分为可解释的域集合并在这些域上系统评估检测性能。

**💡 创新点**

创新点包括：①将域迁移拆分为图像外观、场景构成和采集几何三大维度，并为每个维度设计可量化的分类标签；②通过手工标注小样本校准指标阈值，实现自动化域标签生成；③在公开数据集上展示不同域对检测精度的显著影响，揭示隐藏的失败模式；④提出一种可解释的域-aware 评估方法，超越传统的整体 mAP 评价。

**🔧 技术方法**

技术手段包括：图像质量评估指标（Tenengrad、Laplacian 方差、RMS 对比度、频域能量、光照中位数、颜色失衡等）、场景统计（目标密度、覆盖率、重叠度、尺度、背景复杂度）以及单目深度估计（Depth Anything V2）用于推断视角；YOLOv6（YOLO26n）作为目标检测模型；对每个域使用精确度、召回率、mAP 等指标及错误统计（FP/FN）进行评估。

**📊 数据集**

使用的公开数据集为 DUO（Detecting Underwater Objects）和 RUOD 的四类子集 RUOD-4C，合计 12,050 张图像、108,962 个标注，分为 80% 训练、10% 验证、10% 测试。

**📈 对比分析**

评估方法：在混合训练集上训练模型，然后在每个域的测试子集上单独评估，关注极端类别（如高/低可见度、稀疏/拥挤布局、前视/俯视视角等）。结果显示：高可见度、拥挤布局、大目标、前视图等域显著提升 mAP 与召回率；低可见度、稀疏布局、小目标、俯视图等域性能明显下降。整体而言，模型在平均 mAP50 为 0.868，mAP50-95 为 0.649，显示不同域间存在 10–16% 的性能波动。

**⚠️ 局限性**

局限性：①域标签生成依赖于先验阈值，可能对不同数据集迁移时需要重新校准；②部分域（如稀疏布局、俯视图）样本稀缺导致评估不稳定，特别是罕见类（如贝类）对 mAP 影响过大；③未在不同网络架构上验证泛化能力；④仅在公开数据集上测试，未覆盖更复杂的多场景或多传感器环境。

---

## 88. ImproBR: Bug Report Improver Using LLMs

**arXiv ID:** 2604.26142 | [PDF](https://arxiv.org/pdf/2604.26142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 89. EvoSelect: Data-Efficient LLM Evolution for Targeted Task Adaptation

**arXiv ID:** 2604.26170 | [PDF](https://arxiv.org/pdf/2604.26170v1)

**作者:** Ting-Wei Li `[一作]` (University of Illinois Urbana Champaign), Hanghang Tong `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在迭代生成–选择–训练循环中对大语言模型进行高效迁移的框架，先生成候选样本，再通过OT‑based选择机制挑选出既与目标任务对齐又具有多样性的训练样本，随后微调模型；

**💡 创新点**

创新点在于首次将任务对齐（利用最优传输的梯度信息）与多样性约束在同一优化框架内联合考虑，并通过代理模型高效获取梯度表示；

**🔧 技术方法**

使用技术包括代理模型+稀疏Johnson‑Lindenstrauss投影获取梯度特征，最优传输（Sinkhorn）求解任务对齐，正则化多样性能量，并在迭代中对权重进行指数更新；

**📊 数据集**

实验数据集涵盖科学推理、常识/逻辑推理和生物医学问答等共计十个基准，使用Qwen2.5系列模型，生成器为基线模型，基准对比包括Random、Attribution、Diversity、TSDS等；

**📈 对比分析**

与所有基线相比，该方法在大部分数据集上均取得更高的准确率，并且在迭代过程中始终保持对基线模型的正向改进（无恶化现象），在强生成器和弱生成器设置下均表现稳定；

**⚠️ 局限性**

局限性包括：对计算资源的依赖（需计算梯度和OT）、实验仅覆盖两轮迭代，缺乏对更长循环的分析，且对超参数（OT权重、正则化系数）的敏感性尚未系统评估。

---

## 90. Test-Time Safety Alignment

**arXiv ID:** 2604.26167 | [PDF](https://arxiv.org/pdf/2604.26167v1)

**作者:** Baturay Saglam `[一作]` (Yale University), Dionysis Kalogerias `[通讯]` (Yale University)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5091493591)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不修改模型权重的前提下，利用零阶梯度估计对提示词嵌入进行优化，从而在测试时降低对齐模型生成的有害文本。

**💡 创新点**

首次将子词级别的嵌入优化与黑盒内容审核 API 结合，实现对对齐模型的安全对齐而不依赖额外训练或校准集。

**🔧 技术方法**

零阶梯度估计、随机高斯平滑、梯度归一化与余弦相似度正则化的迭代优化。

**📊 数据集**

WildJailbreak（对抗性正/恶意）、HarmBench 以及标准红队评测集。

**📈 对比分析**

与 SmoothLLM、AdaSteer、RESTA 等测试时防御方法比较，平均每个模型最多仅需 1–2 步即可将被标记的响应几乎全部消除，同时对安全性良好的响应影响极小。

**⚠️ 局限性**

需要访问嵌入层且对嵌入维度及已有安全训练质量敏感；对 API 延迟敏感；对完全封闭的 API 访问模型不可用。

---

## 91. Hard-to-Sample Distributions from Robust Extractors

**arXiv ID:** 2604.26179 | [PDF](https://arxiv.org/pdf/2604.26179v1)

**作者:** Farzan Byramji `[一作]` (University of California San Diego), Anthony Ostuni `[通讯]` (University of California San Diego)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5000208053)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出了统一的框架，利用鲁棒提取器（robust extractor）构造对多种受限计算模型（如低深度电路、小空间源、通信源、图灵机源以及低度二进制多项式源）都难以采样的显式分布。

**💡 创新点**

创新点包括：① 引入鲁棒提取器概念，要求在少量点违反最小熵约束时仍保持提取性；② 通过鲁棒提取器得到新的弱对象“隔离器”（isolator）；③ 以此构造出距离为 1‑o(1) 的显式分布，尤其首次给出对低度二进制多项式源的 1‑o(1) 难采样结果；④ 将已有的提取器构造和最小熵极化技术结合，统一复现之前所有已知的采样下界。

**🔧 技术方法**

使用的主要技术包括：鲁棒提取器与隔离器的定义与构造；最小熵极化与随机约束（random restriction）技术；输入约简（input reduction）与桶合并（bucket merging）方法；两源提取器、全局哈希函数和留存哈希引理；以及对通信、空间与图灵机源的模拟与混合分解。

**📊 数据集**

本文不涉及实际数据集，所有结果均为理论构造和显式算法；所用的“数据集”可视为构造好的分布和源的集合。

**📈 对比分析**

与以往工作相比，该框架能够在统一方式下得到所有已知的采样下界，且在低度多项式源上取得 1‑o(1) 的距离，优于之前仅 2^-Ω(n) 的距离；在其余模型（通信源、小空间源、图灵机源等）下也恢复了现有结果，并在复杂度上保持多项式时间可构造。

**⚠️ 局限性**

局限性包括：① 对 AC0[⊕] 电路的下界仍未达到 1‑o(1)；② 目前 1‑o(1) 的衰减速度不够理想，尤其在 t=1 时只能得到 1/4‑o(1)；③ 对更高阶多项式源的显式下界尚未实现；④ 鲁棒提取器与隔离器的构造在某些模型中仍需进一步改进以获得更紧的参数。

---

## 92. Explaining the "Why": A Unified Framework for the Additive Attribution of Changes in Arbitrary Measures

**arXiv ID:** 2604.26266 | [PDF](https://arxiv.org/pdf/2604.26266v1)

**作者:** Changsheng Zhou `[一作]` (Ant Group), Peng Di `[通讯]` (Ant Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于Shapley值的归因框架，用于对任意聚合度量（包括非可加度量）在子立方体和子度量层面上进行可加性解释。

**💡 创新点**

创新点在于：①统一使用Shapley值完成子立方体与子度量的双重归因；②通过将度量分为GAM与非GAM两类，针对不同结构提供从精确Aumann‑Shapley到近似Shapley的多种算法；③实现了对复杂非线性度量（如计数去重、比率等）的通用归因，并保证解释的可加性。

**🔧 技术方法**

核心技术包括：合作博弈论中的Shapley值、Aumann‑Shapley积分法、线性近似、Kernel‑SHAP与Permutation‑SHAP采样、观测矩阵构造与可加性假设。

**📊 数据集**

实验使用的主要数据集：①线性模拟数据验证准确性；②DAU（count_distinct）模拟验证通用性；③1973年Berkeley招生数据验证解释性；④真实在线视频服务提供商的135个人工验证异常案例验证实际性能。

**📈 对比分析**

与传统OLAP差异化归因、LMDI、根因定位系统（Hotspot、Squeeze、AutoRoot、R‑Adtributor、RobustSpot）进行对比。实验表明：在R‑F1评估上平均提升70%–700%，在DAU和Simpson悖论案例中准确率达到100%，并且Aumann‑Shapley提供了无采样误差的精确解。

**⚠️ 局限性**

主要局限：非GAM模式需在原始记录上重新聚合，导致计算成本显著提高；需要进一步优化近似方法与硬件加速，并在更广泛的工业数据上验证适用性。

---

## 93. GaitKD: A Universal Decoupled Distillation Framework for Efficient Gait Recognition

**arXiv ID:** 2604.26255 | [PDF](https://arxiv.org/pdf/2604.26255v1)

**作者:** Yuqi Li `[一作]` (City University of New York), Yingli Tian `[通讯]` (City University of New York)

**通讯引用:** 18698 | [OpenAlex ID](https://openalex.org/A5074244244)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为GaitKD的知识蒸馏框架，用于在轻量化步态识别模型与高容量教师模型之间进行高效迁移。

**💡 创新点**

创新点在于将步态知识拆分为两部分：决策层蒸馏（对齐软化的类别分布）和边界层蒸馏（保持教师在嵌入空间中的判别边界），并在不同部位上统一对齐，实现对异构教师的兼容与高效蒸馏。

**🔧 技术方法**

采用了温度缩放的KL散度、基于激活边界的损失、部位对齐模块、以及多教师加权融合等技术；同时使用了标准的交叉熵+三元组损失作为基础任务损失。

**📊 数据集**

在三大步态数据集上评估：Gait3D、CCPG 和 SUSTech1K；这些数据集分别涵盖了大规模野外步态、服装变化以及3D LiDAR 步态。

**📈 对比分析**

与教师模型、无蒸馏基线及多教师平均等对比，GaitKD在Rank‑1、mAP、mINP 等指标上显著提升（例如在 Gait3D 上由 61.5% 提升至 65.8%），并在多教师设置中进一步提升性能，证明了两层蒸馏互补。

**⚠️ 局限性**

局限性包括：对教师模型的依赖仍较强，蒸馏过程中需要手动调参（温度、边界阈值等），且在某些轻量模型上提升幅度有限，未来需探索更自动化的蒸馏策略。

---

## 94. TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation

**arXiv ID:** 2604.26247 | [PDF](https://arxiv.org/pdf/2604.26247v1)

**作者:** Wei Yang `[一作]` (Xiaohongshu Inc.), Yao Hu `[通讯]` (Xiaohongshu Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TimeMM，一种基于时间条件的谱过滤框架，用于动态多模态推荐；

**💡 创新点**

创新点包括：① Time-as-Operator 将交互时间映射为可调节的时间核权重，形成无显式特征分解的谱滤波器组；② 通过自适应谱过滤实现上下文条件下的滤波器混合；③ 采用谱感知模态路由，根据时间上下文动态平衡视觉与文本模态；④ 引入谱多样性正则化防止滤波器崩塌；

**🔧 技术方法**

技术手段包括：时间核加权的邻接矩阵构造、无显式特征分解的谱滤波器组、LightGCN 级联传播、门控混合网络、模态路由网络、负采样的二元交叉熵损失、谱多样性正则化与L2正则化；

**📊 数据集**

使用数据集：Amazon Review 旗下的 CD、Game、Baby、Software 四个多模态基准，以及一个真实工业离线数据集（Industry）；

**📈 对比分析**

与 BPR、LightGCN、VBPR、MMGCN、GRCN、DualGNN、SLMRec、LATTICE、BM3、FREEDOM、MMIL、AlignRec、SMORE、FITMM 等多模态及谱方法比较；在所有五个数据集上，TimeMM 取得 Recall@10/20 与 NDCG@10/20 的最高分，提升幅度约 3%–10%，在工业数据上表现尤为显著；

**⚠️ 局限性**

局限性包括：对时间核参数的选择和学习率敏感；在极度稀疏或无时间信息的场景下性能下降；依赖于训练集内的时间分布，可能在时间漂移剧烈时需要进一步调整；以及在多模态噪声极大的数据中，模态路由仍可能失效。

---

## 95. Lifting Embodied World Models for Planning and Control

**arXiv ID:** 2604.26182 | [PDF](https://arxiv.org/pdf/2604.26182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 96. Semantic Foam: Unifying Spatial and Semantic Scene Decomposition

**arXiv ID:** 2604.26262 | [PDF](https://arxiv.org/pdf/2604.26262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. MetaSR: Content-Adaptive Metadata Orchestration for Generative Super-Resolution

**arXiv ID:** 2604.26244 | [PDF](https://arxiv.org/pdf/2604.26244v1)

**作者:** Jiaqi Guo `[一作]` (Northwestern University), Aggelos K. Katsaggelos `[通讯]` (Northwestern University)

**通讯引用:** 28200 | [OpenAlex ID](https://openalex.org/A5048650003)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出MetaSR框架，实现内容自适应的元数据协同生成式超分辨率；

**💡 创新点**

把元数据视为受比特率约束的可传输辅助信息，并利用Diffusion Transformer的原生VAE与Transformer实现统一条件接口，在发送端动态选择并压缩元数据，满足端到端RDO优化；

**🔧 技术方法**

采用Diffusion Transformer（CogVideoX‑2B）、一阶采样、两阶段latent‑pixel训练、JBIG2二值图像压缩、信息论分析与统一Token融合；

**📊 数据集**

使用DIV2K、HQ‑VSR、UDM视频子集以及通过JPEG压缩/噪声合成得到的低质量图像/视频；

**📈 对比分析**

与DOVE等基线在4×SR下比较，评估PSNR、SSIM、LPIPS、DISTS、CLIP‑IQA；MetaSR在相同总比特率下提升约1.0 dB PSNR，或在相同PSNR下比DOVE节省多达50%比特率，尤其在高噪声条件下优势更明显；

**⚠️ 局限性**

仅在帧级视频实验，元数据种类有限（Canny、深度），缺乏视频原生压缩与时域一致性；需进一步扩展多模态元数据、可靠性门控及视频级实现。

---

## 98. Recurrence-Based Nonlinear Vocal Dynamics as Digital Biomarkers for Depression Detection from Conversational Speech

**arXiv ID:** 2604.26242 | [PDF](https://arxiv.org/pdf/2604.26242v1)

**作者:** Himadri S Samanta `[一作]` `[通讯]` (Independent Researcher), Himadri S Samanta (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用递归量化分析提取语音轨迹的递归率特征，构建非线性动力学的数字生物标志物，用于识别对话性语音中的抑郁。

**💡 创新点**

证明抑郁会导致语音状态空间中的递归结构改变，并展示递归率作为生物标志物在检测抑郁方面优于传统静态特征和其他非线性指标。

**🔧 技术方法**

COVAREP特征提取、递归率计算（RQA）、Logistic回归、ANOVA特征选择、分层5折交叉验证、置换检验和Bootstrap置信区间等技术。

**📊 数据集**

DAIC-WOZ抑郁子集（142名参与者，其中42名为抑郁组）。

**📈 对比分析**

与静态语音摘要、熵特征、可预测性特征、Hurst指数、确定性和Lyapunov类不稳定性等基线进行比较；递归率模型在交叉验证中获得AUC 0.689，置换检验p=0.004，Bootstrap 95% CI为[0.568,0.758]，显著优于所有基线。

**⚠️ 局限性**

样本量有限、类别不平衡、仅在内部数据集验证、递归阈值采用启发式设定、通道与具体语音参数对应不清晰、仅使用单一RQA指标（递归率）等局限。

---

## 99. Privacy-Preserving Clothing Classification using Vision Transformer for Thermal Comfort Estimation

**arXiv ID:** 2604.26184 | [PDF](https://arxiv.org/pdf/2604.26184v1)

**作者:** Tatsuya Chuman `[一作]` (NTT FACILITIES, INC.), Hitoshi Kiya `[通讯]` (Tokyo Metropolitan University)

**通讯引用:** 4815 | [OpenAlex ID](https://openalex.org/A5015250468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于Vision Transformer（ViT）的隐私保护服装分类方案，利用密钥加密的像素块洗牌和块置换方式在不解密的情况下实现服装隔热估计；

**💡 创新点**

创新点在于将ViT的自注意力机制与块级加密结合，保证加密图像下模型精度不降，同时克服传统CNN在像素级加密下精度大幅下降的瓶颈；

**🔧 技术方法**

使用ViT‑S/16模型，采用随机像素洗牌+块置换的加密方法，对模型参数做密钥加密后部署至云端；

**📊 数据集**

采用DeepFashion数据集，将服装标签重新划分为四个隔热等级（无袖、短袖、长袖、外套），共26,887张图片；

**📈 对比分析**

与传统像素级加密+ResNet‑18方案相比，ViT方案在加密图像上保持95.65%的平均准确率，未出现精度下降，优于传统方案的83.34%并在各类别上均超越；

**⚠️ 局限性**

局限性包括仅验证了四类服装隔热标签，未检验更细粒度分类；加密和解密过程的计算开销与密钥管理未系统评估；以及在边缘设备上的实际部署效率待进一步研究。

---

## 100. EnerGS: Energy-Based Gaussian Splatting with Partial Geometric Priors

**arXiv ID:** 2604.26238 | [PDF](https://arxiv.org/pdf/2604.26238v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 101. Multi-Stage Bi-Atrial Segmentation Framework from 3D Late Gadolinium-Enhanced MRI using V-Net Family Models

**arXiv ID:** 2604.26251 | [PDF](https://arxiv.org/pdf/2604.26251v1)

**作者:** Hao Wen `[一作]` (China Agricultural University), Jingsu Kang `[通讯]` (Tianjin Medical University)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5069572838)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

未提供论文内容，无法概述

**💡 创新点**

未提供论文内容，无法说明创新点

**🔧 技术方法**

未提供论文内容，无法说明使用的技术

**📊 数据集**

未提供论文内容，无法说明使用的数据集

**📈 对比分析**

未提供论文内容，无法说明比较方法和性能

**⚠️ 局限性**

未提供论文内容，无法说明局限性

---

## 102. Beyond Shortcuts: Mitigating Visual Illusions in Frozen VLMs via Qualitative Reasoning

**arXiv ID:** 2604.26250 | [PDF](https://arxiv.org/pdf/2604.26250v1)

**作者:** Hao Guo `[一作]` (Hefei Comprehensive National Science Center), Subin Huang `[通讯]` (Anhui Polytechnic University)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5053121187)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在冻结的视觉语言模型上引入结构化定性推理（SQI），通过在推理时应用三种约束机制提升对视觉幻觉的鲁棒性。

**💡 创新点**

提出了无训练、数据中心的 SQI 框架，包括 Axiomatic Constraint Injection、Hierarchical Scene Decomposition 和 Counterfactual Self-Verification 三个模块，改造推理过程而非模型参数，首次将定性约束系统化用于幻觉任务。

**🔧 技术方法**

采用定性约束注入、层级场景分解以及逆因果自我验证的推理层技术，仅在推理阶段动态调整，保持模型冻结状态。

**📊 数据集**

在 DataCV 2026 Challenge 的 Task I Classic Illusion Understanding 数据集上进行评估，包含原始图像与扰动图像两种子集。

**📈 对比分析**

在所有参赛队伍中获得第二名，整体准确率为 69.05%，扰动图像 67.62%、原始图像 70.48%，相较于顶尖方法表现更稳定，且在不同视觉条件下保持一致性。

**⚠️ 局限性**

仅针对推理层面进行改进，仍受限于冻结模型的视觉表征，无法解决更深层次的表征瓶颈，对新颖幻觉类型的泛化仍有一定限制。

---

## 103. Institutional Floors and Partisan Lenses: Cross-National Online Discourse on Political Violence in France and the United States

**arXiv ID:** 2604.26245 | [PDF](https://arxiv.org/pdf/2604.26245v1)

**作者:** Andrew Yen Chang `[一作]` `[通讯]` (University of California), Andrew Yen Chang (University of California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对法国和美国三起政治暴力事件的社交媒体讨论进行内容分析，探究不同民主制度下公众如何对政治暴力进行框架化、情绪表达和道德评判。

**💡 创新点**

创新点在于提出并验证了“制度底线假设”（institutional floor hypothesis），即法国共和体制在面对不同受害者类型时仍保持跨党派的制度框架；同时首次将 LLM 零-shot 分类与跨语言跨国对比相结合，强调社会认知分布而非客观描述。

**🔧 技术方法**

使用 GPT‑4o‑mini 进行零-shot 分类（框架、情绪、道德评判），配合卡方检验统计显著性；对比 Lexicon‑based 方法（NRC、eMFD、VADER）进行鲁棒性检验；采用社交网络分析探讨语义网络结构。

**📊 数据集**

数据集来自 Meta Content Library 的公开 Instagram 与 Facebook 帖子，时间窗口为每起事件前后 31 天，语言分别为法语（Paty、Deranque）和英语（Kirk），每个案例每平台 1,000 条，共计 6,000 条帖子。

**📈 对比分析**

通过对三案例的分布进行卡方检验，发现法国案例的制度框架比例（民事+国家）显著高于美国案例；情绪分布在法国更集中于悲痛与道德愤怒，American 案例则情绪碎片化。LLM 分类与人工标注的 Cohen’s κ 在法语案例中为中等水平，验证方法可靠；Lexicon 方法与 LLM 结果差异显著，说明 LLM 能更好捕捉语境信息。

**⚠️ 局限性**

局限性包括：样本仅来自 Meta，可能缺乏代表性；美国仅有单一案例，缺乏内在多样性；跨时间跨度可能引入平台演变偏差；LLM 结果受模型版本更新影响；人工验证样本有限；事件性质差异（国家层面、事件规模）可能混淆比较。

---

## 104. Do E-Scooter Speed Governance Policies Reduce Harsh Acceleration and Deceleration? Evidence from 19.5 Million Trips Around a Regulatory Ban

**arXiv ID:** 2604.26236 | [PDF](https://arxiv.org/pdf/2604.26236v1)

**作者:** Seongjin Choi `[一作]` (University of Minnesota), Sugie Lee `[通讯]` (Hanyang University)

**通讯引用:** 3852 | [OpenAlex ID](https://openalex.org/A5022265510)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了韩国共享电动滑板车平台在2023年12月取消 Turbo（未管控）模式后，速度治理对急加速与急减速事件的影响，采用19.5万条GPS轨迹，先进行骑手异质化随机参数 Logit 预测，再用连续剂量差分设计进行因果验证。

**💡 创新点**

创新点：①首次将骑手异质化随机参数预测与连续剂量差分验证相结合，量化速度治理对多维安全边际的行为效应；②通过组合检查排除补偿效应，证实安全收益来自机械替换而非骑手行为改变；③提出“先预测后验证”框架，可为其他平台级治理政策提供可复制的评估方法。

**🔧 技术方法**

使用的技术包括随机参数二元 Logit、线性概率模型差分（DiD）与连续剂量交互、用户/城市/月份固定效应、Mundlak 处理、Cohen d 组内效应检验、Bonferroni 多重检验校正，以及样本重加权分解。

**📊 数据集**

数据集：2023年2月–11月，韩国52个城市的 Swing 共享电动滑板车 19,458,758 条完整 GPS 轨迹，1,001,459 名用户，包含模式标识、速度序列、时间与地点信息。

**📈 对比分析**

方法比较：Phase I 预测得到-2.81pp（急加速）/ -4.24pp（急减速）；Phase II 差分验证得到-6.24pp/ -5.29pp，两者均显著（p<0.025）且方向一致，说明预测与因果估计高度吻合，效应放大约1.2–2.2 倍，验证了模型可靠性。

**⚠️ 局限性**

局限性：①仅适用于硬件相同、平台通过固件实现的速度治理，无法直接外推至已完全管控或靠骑手遵守的情境；②未直接连接到事故或伤害统计，安全收益仅体现在急加速/急减速边际；③预测模型基于分层样本，需通过重加权分解解释差距，未使用完整样本的一步式估计，未来研究可进一步完善。

---

## 105. ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems

**arXiv ID:** 2604.26231 | [PDF](https://arxiv.org/pdf/2604.26231v1)

**作者:** Yi Zhang `[一作]` (Anhui University), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 17807 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个通用框架 ProRec，利用 LLM 生成的用户/物品描述性 profile 作为分布重塑的指示器，来提升任意基线推荐模型的性能。

**💡 创新点**

创新点包括：① 以检索视角重新解释 profile 的协同信号；② 引入监督与自监督两种分布重塑机制，充分挖掘 profile 的潜力；③ 通过稀疏度级别移动控制候选数量并用 LLM 进行细粒度重排序；④ 仅将 profile 用作指示器，保持原模型架构不变，避免非线性对齐导致的语义丢失。

**🔧 技术方法**

技术手段：LLM（如 GPT‑4o‑mini）生成 profile → 文本嵌入降维 → 稠密检索获取相似用户/物品 → 稀疏度级别移动 + LLM 重新排序 → 监督分布重塑（利用熵权重的 Softmax 分布） + 自监督分布重塑（双向一致性约束） → 追加至推荐模型的训练损失。

**📊 数据集**

使用三大极稀疏公开数据集：Amazon‑Book、Yelp、Steam。

**📈 对比分析**

方法对比：在 LightGCN、MF、GCCF、SimGCL 等经典基线上，与 9 种现有 LLM‑增强方法（KAR、LLMRec、CARec、RLMRec、AlphaRec、LLMESR、AlphaFuse、IRLLRec、ProEx）进行全面对比。ProRec 在 Recall@10/20、NDCG@10/20 上均取得 4–15% 的提升，并在所有基线和 LLM‑增强方法中位居榜首。

**⚠️ 局限性**

局限性：依赖预训练 LLM 与文本信息，生成和检索阶段需额外预处理；对极端稀疏用户的提升有限；需要调节 λ1、λ2 等超参；目前仅在静态协同过滤任务验证，序列推荐等更复杂场景的适用性尚待探索。

---

## 106. Comparative Analysis of AutoML and BiLSTM Models for Cyberbullying Detection on Indonesian Instagram Comments

**arXiv ID:** 2604.26229 | [PDF](https://arxiv.org/pdf/2604.26229v1)

**作者:** Raihana Adelia Putri `[一作]` (Institut Teknologi Sumatera), Martin Clinton Tosima Manullang `[通讯]` (Institut Teknologi Sumatera)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较了传统机器学习（朴素贝叶斯、逻辑回归、SVM）与深度学习（Bi‑LSTM 及 Bi‑LSTM+Bahdanau Attention）在印尼 Instagram 评论中检测网络霸凌的性能；

**💡 创新点**

首次为印尼非正式网络文本设计专门的预处理管道，并直接将经典 ML 与基于注意力的 Bi‑LSTM 进行横向对比；

**🔧 技术方法**

采用 TF‑IDF 特征提取与 朴素贝叶斯、逻辑回归、SVM 进行训练，同时使用 Bi‑LSTM 与 Bi‑LSTM+Attention 结构，并结合 indoNLP、nlp‑id、PySastrawi 进行预处理；

**📊 数据集**

使用 650 条平衡标注的印尼 Instagram 评论（325 条霸凌、325 条非霸凌）构成数据集；

**📈 对比分析**

通过 5‑fold 交叉验证（ML）和 80/10/10 数据划分（DL）比较模型，逻辑回归达到 85.25% 准确率 / 85.22% F1，Bi‑LSTM+Attention 为 84.62% 准确率 / 84.58% F1‑macro，表明 DL 在性能上略优但 ML 仍具竞争力；

**⚠️ 局限性**

样本量仅 650 条，且来源仅为艺人/影响者账号，缺乏更广泛的社群覆盖，且仅采用二元标签，限制了模型的泛化与细粒度分析能力。

---

## 107. Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging

**arXiv ID:** 2604.26206 | [PDF](https://arxiv.org/pdf/2604.26206v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]` (Independent Researcher), Jon-Paul Cacioli (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在MMLU‑Pro多选任务上评估沙袋化（sandbagging）行为，通过在指令与选项顺序条件下对三种LLM进行实验，检验其是否呈现位置主导的响应分布。

**💡 创新点**

首次提出响应位置熵（response‑position entropy）作为黑盒沙袋化检测的可行信号，并在预注册框架下系统验证了分布吸引子（position attractor）的存在。

**🔧 技术方法**

使用预注册实验设计、同一条条目双条件对照、Pearson相关、Jensen‑Shannon 散度、TVD、McNemar 检验等统计方法，并对模型进行选项循环移位操作。

**📊 数据集**

利用包含2000道题、10个选项的 MMLU‑Pro 数据集，按物理、法律、心理学、经济学四个领域分布。

**📈 对比分析**

与诚实指令对照相比，Llama‑3‑8B 在沙袋化指令下响应位置熵从 0.972 降至 0.681，JS 散度仅 0.027（对照 0.386），表明位置吸引子高度稳定；Qwen‑2.5‑7B 未出现此效应。

**⚠️ 局限性**

仅在单一 7‑9B 规模的可遵循指令模型上验证，未探测到内容空间沙袋化，取决于选项循环移位的绝对位置效应，量化过程可能放大偏差，结果对更大规模或更复杂沙袋化策略的普适性存疑。

---

## 108. Multiple Consistent 2D-3D Mappings for Robust Zero-Shot 3D Visual Grounding

**arXiv ID:** 2604.26261 | [PDF](https://arxiv.org/pdf/2604.26261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 109. Evergreen: Efficient Claim Verification for Semantic Aggregates

**arXiv ID:** 2604.26180 | [PDF](https://arxiv.org/pdf/2604.26180v1)

**作者:** Alexander W. Lee `[一作]` (Brown University and Snowflake Inc.), Anupam Datta `[通讯]` (Snowflake Inc.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用语义查询引擎对从语义聚合结果中提取的自然语言主张进行验证的系统，通过将主张编译为声明式的语义验证查询并执行。

**💡 创新点**

创新点包括：①将主张验证视为语义查询任务，利用现有语义查询引擎实现验证；②设计专门的验证友好优化（早停、相关性排序、置信序列估计）与通用优化（算子融合、相似度过滤、提示缓存），显著降低LLM调用成本和延迟；③用第一阶逻辑的半环语义实现最小化的证明（引用）生成，提供可解释的验证结果。

**🔧 技术方法**

技术包括：LLM（Claude Opus、Sonnet、Haiku 及多款 Llama 模型）用于语义算子；语义查询引擎（Snowflake Cortex AI）执行算子；优化器实现早停、相关性排序、置信序列、算子融合、相似度过滤、提示缓存；半环语义与归约得到最小证明集；实验采用 AQL 与 Python API。

**📊 数据集**

使用 Yelp Open Dataset 的三份子集：Yelp1（1,609 条评测，230k 令牌）、Yelp2（1,813 条，244k 令牌）、Yelp3（1,603 条，291k 令牌）。基于这些数据集生成 16 条包含不同类型主张（存在、全称、计数、比例、序数、嵌套等）的验证基准。

**📈 对比分析**

与三种基线对比：1）LLM‑as‑a‑judge（直接让LLM判断）；2）检索增强型代理（retrieval‑augmented agent）；3）无优化的 Binder‑style 语义查询。实验结果表明，在强大 LLM（Claude Opus）下，系统在保持完美 F1 的同时将成本降低 3.2×、延迟降低 4.0×；在弱 LLM（Llama 8B）下，仍可实现比基线高 1–2 倍的 F1，同时成本降低 48–63 倍、延迟降低 2.3–4.2 倍。

**⚠️ 局限性**

局限性：①早停与估计策略假设主张大多为真，若主张为假可能导致误判；②相似度过滤阈值设定需权衡召回率与成本，过高会漏检；③当前系统仅支持单个主张验证，未充分利用多主张共享计算的机会；④对极大规模数据集（>10k 条）和更复杂的分组/嵌套查询的性能未充分评估。

---

## 110. StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall

**arXiv ID:** 2604.26243 | [PDF](https://arxiv.org/pdf/2604.26243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 111. Breaking the Autoregressive Chain: Hyper-Parallel Decoding for Efficient LLM-Based Attribute Value Extraction

**arXiv ID:** 2604.26209 | [PDF](https://arxiv.org/pdf/2604.26209v1)

**作者:** Theodore Glavas `[一作]` (Amazon.com, Inc.), Shervin Malmasi `[通讯]` (Amazon.com, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对属性值抽取（AVE）任务，提出了一种在单一prompt中并行生成多条输出序列的解码算法Hyper-Parallel Decoding（HPD）。

**💡 创新点**

创新点在于通过在prompt中插入位置ID空隙并重新定义注意力掩码，打破传统自回归依赖，实现多属性值同时解码，且支持在同一prompt中堆叠多份文档进一步提升并行度。

**🔧 技术方法**

核心技术包括位置ID跳跃（gap insertion）、自定义注意力掩码、与Key‑Value缓存兼容的并行解码流程，以及针对HPD的专门微调策略。该方法可与量化、蒸馏、批量推理等其它加速手段协同使用。

**📊 数据集**

在三个电商领域的AVE基准上验证：OA-Mine、AE‑110k以及基于Amazon Reviews 2023的零样本任务（通过GPT‑4.1蒸馏得到伪标签）。

**📈 对比分析**

与传统自回归解码（AR）以及现有的Speculative Decoding进行对比。HPD在所有模型（Qwen3、Phi‑4 14B等）上实现了最高13.8×的推理速度提升和13.79×的成本下降，且在F1分数上保持或略优于AR；在Amazon Reviews上，HPD比Speculative Decoding快约10.8×。

**⚠️ 局限性**

局限性包括：实验主要集中在Qwen3和Phi‑4系列模型，未覆盖更大规模或不同架构的LLM；仅验证了产品属性抽取场景，尚未在其他任务或数据域广泛评估；与Flash‑Attention等特定实现存在兼容性问题，需采用支持自定义掩码的Attention实现。

---

## 112. Calibrated Surprise: An Information-Theoretic Account of Creative Quality

**arXiv ID:** 2604.26269 | [PDF](https://arxiv.org/pdf/2604.26269v1)

**作者:** Bo Zou `[一作]`, Chao Xu `[通讯]` (Nutcracker Studio)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过信息理论的互信息衡量，提出“校准惊喜”作为创意写作质量的客观指标。

**💡 创新点**

将作者意图、读者期望、现实逻辑三源独立共识与互信息关联，形成统一的质量定义。

**🔧 技术方法**

使用 Shannon 互信息公式 I(X;Y)=H(X)-H(X|Y) 及 LLM logprob 近似理想读者概率。

**📊 数据集**

采用 20 对中英文学段落（共 20 篇）及其人为降质版本。

**📈 对比分析**

通过计算高质量文本与降质文本的 I(X;Y) 差异，验证高质量文本互信息显著更高（平均提升约 0.37 bit/ token，所有样本均符合预测）。

**⚠️ 局限性**

限制包括维度定义的主观性、理想读者近似的 LLM 偏差、链式粒度与跨文化通用性待验证。

---

## 113. 2D and 3D Grasp Planners for the GET Asymmetrical Gripper

**arXiv ID:** 2604.26212 | [PDF](https://arxiv.org/pdf/2604.26212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 114. FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients

**arXiv ID:** 2604.26258 | [PDF](https://arxiv.org/pdf/2604.26258v1)

**作者:** Hongyeon Yu `[一作]` (Naver Search US), Yoon Kim `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22245 | [OpenAlex ID](https://openalex.org/A5100693798)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于双层优化的自动化LLM工作流诱导方法（FlowBot），能够同时学习工作流结构和每个LLM调用的提示。

**💡 创新点**

创新点在于将工作流结构搜索与模块级提示优化通过文本梯度（自然语言反馈）耦合实现双层优化；并通过逐层文本反向传播（类似神经网络反向传播）细化各步骤。

**🔧 技术方法**

使用的技术包括：文本梯度（LLM生成的自然语言反馈）、双层优化（外层结构搜索、内层提示优化）、层级反向传播、Meta LLM生成梯度与提示更新、以及对工具调用的控制。

**📊 数据集**

在十个基准数据集上进行评估：HotpotQA（1、2）、IFBench、HoVer、PUPA、DROP、GSM8K、MATH、HumanEval、MBPP。

**📈 对比分析**

与手工设计的工作流和prompt优化基线（GEPA、Trace、TextGrad）以及自动工作流生成方法（AFlow、ADAS）进行对比，FlowBot在大多数任务上获得相当或更优的性能，同时在API调用和成本上更高效。

**⚠️ 局限性**

局限性包括：需要较多的计算和API调用来进行结构和提示的优化，优化过程计算量大；在某些任务上收敛快后不再受益于更多数据；并且模型在已达到最优工作流后可能无法进一步提升。

---

## 115. Hierarchical Long-Term Semantic Memory for LinkedIn's Hiring Agent

**arXiv ID:** 2604.26197 | [PDF](https://arxiv.org/pdf/2604.26197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 116. LATTICE: Evaluating Decision Support Utility of Crypto Agents

**arXiv ID:** 2604.26235 | [PDF](https://arxiv.org/pdf/2604.26235v1)

**作者:** Aaron Chan `[一作]` (Sahara AI), Xiang Ren `[通讯]` (Sahara AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了名为Lattice的基准，用来评估加密代理在真实用户交互场景中的决策支持效用。

**💡 创新点**

通过六个维度（意图忠实、机制清晰、不确定性处理、可执行性、证据覆盖、响应结构）设计了16种任务和5种查询类别，并使用LLM评审器实现可扩展、无人工标注的自动评估。

**🔧 技术方法**

使用大型语言模型（GPT‑5.2）作为查询生成器和评审器，构建基于规则的评分量表，并通过加权求和得到最终分数。

**📊 数据集**

构建了1200条人工合成查询（覆盖16种任务 × 5种查询类别，每种组合15条），并在六个实际部署的加密 Copilot 上收集响应进行评估。

**📈 对比分析**

对六个 Copilot 进行绝对评分；Sorin 在所有维度和大多数任务上得分最高，其他四个在中间梯队中表现相近；通过维度、任务和查询类别的细粒度拆分揭示各 Copilot 的优势与劣势；人类对比实验进一步验证了评估结果的可解释性。

**⚠️ 局限性**

评估依赖单一 LLM 评审器与固定规则，可能受模型偏差影响；仅基于静态问答对，未考虑实时数据访问或多轮交互；每个问答对仅评估一次，未覆盖回答方差；对外部事实验证的覆盖有限。

---

## 117. When Agents Shop for You: Role Coherence in AI-Mediated Markets

**arXiv ID:** 2604.26220 | [PDF](https://arxiv.org/pdf/2604.26220v1)

**作者:** Soogand Alavi `[一作]` (University of Iowa), Salar Nozari `[通讯]` (University of Iowa)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究AI代理在在线购物中的信息泄漏，设计实验让买家代理在两种指令（口头角色描述 vs 隐藏预算）下与卖家代理对话，随后用推理代理仅凭对话文本估计买家的最高支付意愿。

**💡 创新点**

提出“角色一致性（role coherence）”作为新的隐私泄漏通道：代理在忠实扮演给定角色时，其行为分布会泄露买家支付意愿；与以往关注指令遵循失败的隐私失效不同；进一步展示提示级隐私缓解无效，需在架构层面介入。

**🔧 技术方法**

使用大型语言模型 Claude Haiku 4.5 作为买家、卖家与推理代理；通过系统提示控制角色或预算；采用多轮对话生成；对推理结果进行线性回归、MAE、Spearman相关、bootstrap置信区间等统计分析。

**📊 数据集**

无公开数据集；构造6个口头角色描述与对应支付阈值（$50–$500），6个数值预算指令；产品目录包含5款无线耳机；所有对话由LLM生成并记录；推理代理仅使用对话文本进行估计。

**📈 对比分析**

对比口头角色描述与数值预算两种条件，测量估计WTP与目标的斜率、平均绝对误差、等级相关；结果显示口头条件斜率≈1.00、MAE≈48美元；数值预算斜率≈0.21、MAE≈92美元；进一步的剔除词汇、角色剥离、因子设计等稳健性检验均保持口头条件高斜率，证明角色一致性有效；相比之下，数值条件表现为对目标几乎不变的压缩。

**⚠️ 局限性**

局限性包括：实验仅使用单一LLM模型，未验证跨模型泛化；仅测试单一商品类别（耳机）；角色描述与产品相互作用的复杂性有限；推理代理的判断基于语言模型自身能力，真实用户对话可能更复杂；架构级防护方案仅提出概念，需后续实现与评估。

---

## 118. OMEGA: Optimizing Machine Learning by Evaluating Generated Algorithms

**arXiv ID:** 2604.26211 | [PDF](https://arxiv.org/pdf/2604.26211v1)

**作者:** Jeremy Nixon `[一作]` (Infinity Artificial Intelligence Institute), Annika Singh `[通讯]` (Infinity Artificial Intelligence Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了OMEGA框架，实现从模型想法到可执行、可评估的分类算法的端到端自动化生成与评估；

**💡 创新点**

创新点在于将LLM作为可执行代码生成器与自愈循环相结合，自动化产生全新机器学习算法，并对生成的模型进行统一的性能评估与自我改进；

**🔧 技术方法**

使用技术包括大型语言模型（Claude Sonnet 4.5、GPT-4.1 mini、Gemini 2.5 Flash、grok-code-fast-1）、自愈式代码生成管道、scikit‑learn API约束、元学习与堆叠泛化、特征方向化森林等；

**📊 数据集**

采用20个来自scikit‑learn/OpenML的分类基准数据集，涵盖数值与类别特征、不同规模、二分类/多分类等多样性；

**📈 对比分析**

比较方法为对每个数据集计算min‑max标准化准确率并聚合排名，评估模型的相对表现；生成模型在平均min‑max分数上普遍高于scikit‑learn基线，MetaSynthesisClassifier等模型表现尤为突出；

**⚠️ 局限性**

局限性包括仅针对分类任务，未验证图像/视频等非结构化数据；模型解释性不足；性能提升在很大程度上依赖提示质量和LLM的生成能力，且自愈循环可能无法修复所有错误。

---

## 119. Efficient and Interpretable Transformer for Counterfactual Fairness

**arXiv ID:** 2604.26188 | [PDF](https://arxiv.org/pdf/2604.26188v1)

**作者:** Panyi Dong `[一作]` (University of Illinois Urbana-Champaign), Zhiyu Quan `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 121 | [OpenAlex ID](https://openalex.org/A5042065650)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Feature Correlation Transformer（FCorrTransformer）与Counterfactual Attention Regularization（CAR）两项技术，用于在表格数据中实现无偏判定和可解释的机器学习模型。

**💡 创新点**

创新点包括：①将注意力机制轻量化，直接使用原始特征值并将注意力矩阵视为特征间依赖关系，提升可解释性；②CAR在注意力层面对敏感特征进行公平正则化，避免显式因果假设；③采用输入增强与共享参数的策略实现高效的反事实学习。

**🔧 技术方法**

使用的技术主要有：Transformer自注意力（改进的无投影版本）、元素级线性映射、GELU激活、LayerNorm、CAR正则项、输入增强与参数共享机制；实验中还对比了FFN、TabTransformer、FT-Transformer以及LightGBM等基线模型。

**📊 数据集**

实验数据集包括：①合成 toy 数据（用于验证注意力解释和 CAR 效果）；②银行账户欺诈（BAF）表格数据（100 万样本，31 特征，1.1% 欺诈率）；③InsurTech 商业责任保险数据（235k 保单，581 特征，极度不平衡）。

**📈 对比分析**

方法对比采用准确率、F1、AUROC、AUPRC、MSE、Gini 等性能指标与公平指标（DPD、EqOdd、EqOpp、AvgIF 等）。结果表明：FCorrTransformer 在保持相近甚至略低的预测性能的同时，参数量和显存显著减少；加上 CAR 后，各种公平度量几乎降至零，且对性能影响最小；相比之下，传统 Transformer 基线在公平度量上仍存在显著偏差。

**⚠️ 局限性**

局限性包括：CAR 目前仅适用于离散敏感特征，难以推广到连续特征；缺乏正式的因果推断保证；轻量化注意力架构可能在高度非线性或复杂特征交互任务中表现不佳；在极大敏感特征类别数时仍会产生计算开销。

---

## 120. SWAN: World-Aware Adaptive Multimodal Networks for Runtime Variations

**arXiv ID:** 2604.26181 | [PDF](https://arxiv.org/pdf/2604.26181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 121. Apriori-based Analysis of Learned Helplessness in Mathematics Tutoring: Behavioral Patterns by Level, Intervention, and Outcome

**arXiv ID:** 2604.26237 | [PDF](https://arxiv.org/pdf/2604.26237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 122. HOI-aware Adaptive Network for Weakly-supervised Action Segmentation

**arXiv ID:** 2604.26227 | [PDF](https://arxiv.org/pdf/2604.26227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 123. StreamGuard: Exploring a 5G Architecture for Efficient, Quality of Experience-Aware Video Conferencing

**arXiv ID:** 2604.26223 | [PDF](https://arxiv.org/pdf/2604.26223v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 124. Seeking Consensus: Geometric-Semantic On-the-Fly Recalibration for Open-Vocabulary Remote Sensing Semantic Segmentation

**arXiv ID:** 2604.26221 | [PDF](https://arxiv.org/pdf/2604.26221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 125. OpenSOC-AI: Democratizing Security Operations with Parameter Efficient LLM Log Analysis

**arXiv ID:** 2604.26217 | [PDF](https://arxiv.org/pdf/2604.26217v1)

**作者:** Chaitanya Vilas Garware `[一作]` (University of Alabama at Birmingham), Sharif Noor Zisad `[通讯]` (University of Alabama at Birmingham)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5095853206)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 OpenSOC-AI，一个利用 TinyLlama-1.1B 微调的轻量级日志分析框架，帮助 SMB 进行自动威胁分类、MITRE ATT&CK 对映与严重度评估。

**💡 创新点**

证明小型 LLM 可通过 LoRA 微调在仅 450 条 SOC 示例、单张 T4 GPU 4 分钟内显著提升安全日志处理效果，并将完整代码、数据与模型权重公开，降低安全运维门槛。

**🔧 技术方法**

采用 LoRA + QLoRA 4‑bit 量化、TinyLlama‑1.1B chat 模型、Alpaca 风格指令模板以及正则提取等技术实现低资源高效推理。

**📊 数据集**

构建了 500 条安全日志分析实例（450/50 训练/测试），覆盖 12 类威胁，并标注 MITRE ATT&CK 识别码。

**📈 对比分析**

与未微调基线比较，微调模型在 50 条测试日志上实现威胁分类准确率从 0% 提升到 68%，严重度准确率从 28% 提升到 58%，F1 为 0.68；训练仅耗 4 分钟，训练参数 12.6M（占 1.13%）。

**⚠️ 局限性**

受限于样本规模有限、仅做单标签分类、MITRE ID 提取不完善、对新攻击模式泛化不足，以及 32% 的误判率，需进一步扩展数据、改进多标签和 ID 解析。

---

## 126. Camera-RFID Fusion for Robust Asset Tracking in Forested Environments

**arXiv ID:** 2604.26241 | [PDF](https://arxiv.org/pdf/2604.26241v1)

**作者:** John Hateley `[一作]` (University of California), Omid Abari `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

融合被动RFID与深度相机，利用轨迹匹配实现森林环境下资产的精准追踪

**💡 创新点**

首次引入不确定Fréchet距离与Mahalanobis距离的组合，解决RFID与视觉轨迹关联难题，并在森林环境中实现高精度资产追踪

**🔧 技术方法**

高斯过程模型+扩展Kalman滤波器生成RFID轨迹；深度相机配合DeepSORT生成视觉轨迹；不确定Fréchet距离和Mahalanobis距离加最小成本匹配实现轨迹关联

**📊 数据集**

仿真数据（多密度标签与人类移动轨迹）和实地森林实验数据（Zed2相机+Impinj R420读取器，最多4人，10次试验）

**📈 对比分析**

与欧氏距离和DTW对比；在低密度场景下实现100%匹配，最高约80%在高密度；观察时间0.6–40秒，显著优于基准方法

**⚠️ 局限性**

随着参与者密度升高，轨迹重叠导致匹配误差增大，系统在高密度下精度下降；受相机10m、RFID20m范围限制，单台传感器难以覆盖大面积

---

## 127. eDySec: A Deep Learning-based Explainable Dynamic Analysis Framework for Detecting Malicious Packages in PyPI Ecosystem

**arXiv ID:** 2604.26219 | [PDF](https://arxiv.org/pdf/2604.26219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 128. Lights Out: A Nighttime UAV Localization Framework Using Thermal Imagery and Semantic 3D Maps

**arXiv ID:** 2604.26201 | [PDF](https://arxiv.org/pdf/2604.26201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 129. FASH-iCNN: Making Editorial Fashion Identity Inspectable Through Multimodal CNN Probing

**arXiv ID:** 2604.26186 | [PDF](https://arxiv.org/pdf/2604.26186v1)

**作者:** Morayo Danielle Adeyemi `[一作]` (Howard University), Franck Dernoncourt `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了FASH-iCNN系统，利用服装图像以及可选人脸、设计师、季节年份等多模态输入，预测服装所属品牌、年代以及基于柏林-凯家族→CSS→LAB的分层色彩推荐，并将编辑文化的语义结构可视化。

**💡 创新点**

将编辑文化的结构化语义直接作为预测信号，并通过可解释的层级色彩管线将视觉特征映射到可追溯的品牌、时代和色彩传统；展示服装外观本身即能解码品牌、年代与色彩。

**🔧 技术方法**

使用EfficientNet‑B0作为特征提取器，双流（服装+人脸）特征拼接后通过两层全连接头进行分类；分层色彩预测管线（Berlin–Kay→CSS→LAB）；并通过视觉抽象与模态冗余实验验证信息渠道。

**📊 数据集**

基于87,547张 Vogue 走秀图像（1991–2024），筛选后65,541装束裁剪，包含六槽主色、柏林-凯9类、CSS约60类、设计师/季节/年份标签以及人脸裁剪。

**📈 对比分析**

与无约束 LAB 回归 baseline 对比，ΔE00从15.0降至9.10（39%改进）；柏林-凯家族分类单色准确率73.4%，oracle 81.4%；品牌分类78.2%，年代分类88.6%；主色平均 ΔE00 3.09，后槽色彩性能急剧下降。

**⚠️ 局限性**

仅预测单一主色；对多槽色彩预测效果差；仅基于西方奢侈时尚 Vogue 语料，缺乏非编辑、非奢侈、非西方数据；面部输入存在身份泄露风险；跨品牌泛化未验证；ΔE00仍高于人类可接受阈值。

---

## 130. Flashback: A Reversible Bilateral Run-Peeling Decomposition of Strings

**arXiv ID:** 2604.26190 | [PDF](https://arxiv.org/pdf/2604.26190v1)

**作者:** Thomas Konstantinovsky `[一作]` (Bar-Ilan University), Gur Yaari `[通讯]` (Yale School of Medicine)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可逆的字符串分解方法 Flashback，通过从两端同时剥离最大字符段来生成令牌序列；

**💡 创新点**

其创新点在于将字符串的跑长编码（RLE）外向对称配对，从而得到精确的令牌计数 1+⌈r/2⌉ 并证明此计数对所有可接受的双向剥离方案都是最优的；

**🔧 技术方法**

技术上实现了线性时间/空间的递归剥离与嵌套重建，并利用 RLE、对称性和令牌层级分析得到关于内核大小、回文判定、编辑局部性及图像特征的结构性结论；

**📊 数据集**

由于本文为理论研究，未使用任何实际数据集；

**📈 对比分析**

论文未进行实验对比，性能方面通过理论分析证明 Flashback 的时间复杂度为 Θ(n)、空间复杂度为 O(n)，并且在所有双向剥离方案中达到令牌数下界；

**⚠️ 局限性**

主要限制是 Flashback 仅为结构分析工具，无法作为压缩或索引方案使用，并且对字符级编辑的令牌变化仅提供上界，未给出精确量化。

---

## 131. DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training

**arXiv ID:** 2604.26256 | [PDF](https://arxiv.org/pdf/2604.26256v1)

**作者:** Tianhao Hu `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

暂无具体论文内容，无法提供总结。

**💡 创新点**

暂无具体论文内容，无法提供创新点。

**🔧 技术方法**

暂无具体论文内容，无法提供所用技术。

**📊 数据集**

暂无具体论文内容，无法提供所用数据集。

**📈 对比分析**

暂无具体论文内容，无法提供比较方法与性能。

**⚠️ 局限性**

暂无具体论文内容，无法提供局限性。

---

## 132. OmniTrend: Content-Context Modeling for Scalable Social Popularity Prediction

**arXiv ID:** 2604.26252 | [PDF](https://arxiv.org/pdf/2604.26252v1)

**作者:** Liliang Ye `[一作]` (Huazhong University of Science and Technology), Zikai Song `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 572 | [OpenAlex ID](https://openalex.org/A5083665721)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出OmniTrend框架，将社交媒体流行度预测拆分为内容吸引力与上下文曝光两部分，分别建模后结合得到最终预测。

**💡 创新点**

明确分离内容与曝光因子，使用跨平台共享的内容模块与平台特定的上下文模块，并引入检索增强的邻域信息，实现可解释、可迁移的预测。

**🔧 技术方法**

采用跨模态注意力+门控融合的内容编码器、时序与用户属性的上下文网络、CatBoost回归、检索增强邻域统计、交叉模态对齐、Huber损失、排名损失等多种技术。

**📊 数据集**

在四大图像/视频跨平台数据集上评估：MicroLens、ICIP、SMPD-Image、SMPD-Video。

**📈 对比分析**

与SVR、HyFea、CLSTM、HMMVED、CBAN、BLIP、MMRA、TMALL、DLBA等多种基线比较，OmniTrend在MSE、MAE、Spearman相关等指标上均显著优于所有对照组，尤其在排名相关性提升显著。

**⚠️ 局限性**

仍依赖预训练的多模态编码器，检索邻域受时间窗口与语义相似度限制，对极端稀疏数据或新平台的快速适配仍需改进。

---

## 133. A New Semisupervised Technique for Polarity Analysis using Masked Language Models

**arXiv ID:** 2604.26230 | [PDF](https://arxiv.org/pdf/2604.26230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 134. Unsupervised Graph Modeling for Anomaly Detection in Accounting Subject Relationships

**arXiv ID:** 2604.26216 | [PDF](https://arxiv.org/pdf/2604.26216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 135. Persuadability and LLMs as Legal Decision Tools

**arXiv ID:** 2604.26233 | [PDF](https://arxiv.org/pdf/2604.26233v1)

**作者:** Oisin Suttle `[一作]` (Maynooth University), David Lillis `[通讯]` (University College Dublin)

**通讯引用:** 977 | [OpenAlex ID](https://openalex.org/A5037769267)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一套三方（两名“辩护人”模型争辩，一名“法官”模型判决）的实验框架，用来系统评估前沿大语言模型在硬性法律问题中的说服性及其对论点内容与形式的敏感度。

**💡 创新点**

创新点包括：①首次在法律领域引入三方说服实验；②提出“pairwise persuadability”与“population persuadability”两种量化指标；③在同一套案例中对比开放权重与闭源权重、不同规模与推理架构模型的说服表现；④探讨论点内容与论证形式对模型说服力的不同贡献。

**🔧 技术方法**

技术上使用多种大型语言模型（OpenAI GPT‑4、GPT‑5、Anthropic Claude Sonnet 4.5、Google Gemini、DeepSeek、Qwen、Mistral等）生成辩护论点与判决；通过提示工程将案例事实与论点输入给模型；统计分析采用二项检验与卡方检验评估说服力差异。

**📊 数据集**

数据集来自三国（美国、英国/威尔士、爱尔兰）上诉法院的分裂判决摘要，自动生成的案例摘要包含三段事实和两段对立论点，挑选15个“硬”案例进行实验。

**📈 对比分析**

实验采用p_pop（总体说服力）与p_2max（最大成对说服力）作为比较指标。结果显示所有模型均存在显著说服性，p_pop 介于0.08–0.20，p_2max 最高达0.405；大模型在闭源版本中相对不易被说服；提供论点摘要略降低说服力，提示内容在一定程度上影响结果；不同司法辖区亦表现出说服力差异。

**⚠️ 局限性**

局限性包括：仅关注硬性法律问题，未与人类专家对比；使用自动摘要可能缺乏细节；实验规模有限，未深入解析说服机制；仅测量说服性，未评估判决质量与公正性。

---

## 136. DepthPilot: From Controllability to Interpretability in Colonoscopy Video Generation

**arXiv ID:** 2604.26232 | [PDF](https://arxiv.org/pdf/2604.26232v1)

**作者:** Junhu Fu `[一作]` (Princeton University), Shuo Li `[通讯]` (Fudan University)

**通讯引用:** 51361 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了 DepthPilot，一种可解释的结肠镜视频生成框架，利用深度先验和自适应样条去噪实现几何约束和非线性建模。

**💡 创新点**

引入先验分布对齐（PDA）策略将单目深度信息注入扩散模型，实现几何一致性；提出自适应样条去噪（ASD）模块替代线性激活，提升时空非线性表达。

**🔧 技术方法**

使用条件扩散模型、参数高效微调、轻量级深度编码器、分布对齐、B‑spline 可学习激活、VAE+CLIP 编码、混合精度训练与 EMA 等技术。

**📊 数据集**

公开数据集 Colonoscopic、HyperKvasir、SUN‑SEG 以及医院内部 203 条完整结肠镜视频。

**📈 对比分析**

与 StyleGAN‑V、MoStGAN‑V、LVDM、Endora、FEAT‑L、ColoDiff 等六个 SOTA 方法在 FID、FVD、IS、CS 上进行对比，DepthPilot 在所有数据集上 FID<15，FVD 与 CS 最高，显著优于对照组。

**⚠️ 局限性**

仍依赖精确的深度估计，深度先验误差会影响生成质量；模型对极端摄像机运动和稀有病变的泛化有限，且训练成本相对较高。

---

## 137. ViBE: Visual-to-M/EEG Brain Encoding via Spatio-Temporal VAE and Distribution-Aligned Projection

**arXiv ID:** 2604.26218 | [PDF](https://arxiv.org/pdf/2604.26218v1)

**作者:** Ganxi Xu `[一作]` (Jinan University), Jinyi Long `[通讯]` (Jinan University)

**通讯引用:** 2239 | [OpenAlex ID](https://openalex.org/A5042542454)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了ViBE框架，将视觉刺激转换为M/EEG信号，分为两阶段：TSC‑VAE自编码重建神经响应，Q‑Former映射CLIP视觉嵌入至TSC‑VAE潜在空间并通过MSE+SWD实现跨模态对齐。

**💡 创新点**

创新点在于：①设计TSConvPlus的时空分离卷积捕捉M/EEG的层次时空特征；②使用Q‑Former对视觉嵌入进行尺度桥接，将CLIP嵌入映射到与潜在空间同尺度的神经代理嵌入；③结合MSE与切片Wasserstein距离同时对齐点对点与分布。

**🔧 技术方法**

技术包括时空分离卷积的变分自编码器（TSC‑VAE），CLIP视觉编码器，Q‑Former交叉注意力模块，MSE损失，切片Wasserstein距离（SWD），以及留一子测、跨子测等评估协议。

**📊 数据集**

使用公开的THINGS‑EEG2和THINGS‑MEG两大数据集，分别包含图像与对应EEG/MEG记录。

**📈 对比分析**

与Güçlü、Yamins、UNet‑Diffusion、MindSimulator、SynBrain以及Xu等六种基线进行对比；在EEG数据上ViBE在MSE、Pearson、Cosine三指标均显著优于基线（如Pearson 0.635 vs 0.425），在MEG数据上同样提升；跨子测时表现略低，但留一子测可弥补。

**⚠️ 局限性**

局限性包括：跨子测泛化仍有显著下降；视觉到神经映射存在尺度残差，导致Stage II性能仍受限；模型尚未充分利用大规模多被试数据，未来需进一步提升跨被试一致性与训练效率。

---

## 138. Exploring the Feasibility and Acceptability of AI-Mediated Serious Illness Conversations in the Emergency Department

**arXiv ID:** 2604.26214 | [PDF](https://arxiv.org/pdf/2604.26214v1)

**作者:** Hasibur Rahman `[一作]` (Northeastern University), Smit Desai `[通讯]` (Northeastern University)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5033717301)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发并评估了一款在急诊科用于快速结构化重病对话的语音对话代理ED GOAL-AI。

**💡 创新点**

创新点在于将大型语言模型与语音交互结合，局部部署、精细调优，并通过限定问答框架提升安全性与可接受度。

**🔧 技术方法**

使用本地部署的Qwen2.5 7B模型（QLoRA微调）、Whisper语音识别和Kokoro语音合成，实现纯语音对话。

**📊 数据集**

训练数据为500条基于真实SIC记录合成的对话，评估样本为55名≥50岁重症老人。

**📈 对比分析**

通过单组对比患者对代理与医生的“被倾听与理解”评分，完成率49/55、平均对话时长4.5分钟，接受度高；发现一例误诊 hallucination 等失败模式。

**⚠️ 局限性**

局限包括样本量有限、单中心研究、技术失效、情感适配不足以及对系统边界的伦理挑战。

---

## 139. LLM-Assisted Empirical Software Engineering: Systematic Literature Review and Research Agenda

**arXiv ID:** 2604.26192 | [PDF](https://arxiv.org/pdf/2604.26192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 140. Topology-Aware Representation Alignment for Semi-Supervised Vision-Language Learning

**arXiv ID:** 2604.26370 | [PDF](https://arxiv.org/pdf/2604.26370v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 141. ACPO: Anchor-Constrained Perceptual Optimization for Diffusion Models with No-Reference Quality Guidance

**arXiv ID:** 2604.26348 | [PDF](https://arxiv.org/pdf/2604.26348v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. Automaton-based Characterisations of First Order Logic over Infinite Trees

**arXiv ID:** 2604.26364 | [PDF](https://arxiv.org/pdf/2604.26364v1)

**作者:** Massimo Benerecetti `[一作]` (Università degli Studi di Napoli Federico II), Gabriele Puppis `[通讯]` (Università degli Studi di Udine)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在无序无排行无限树上，对一阶逻辑（FO）进行自动机理论的完整表征，并与两类分支时间时序逻辑（Polarised PCTL 和 FCTL）建立等价关系；同时给出新的自动机模型（极化犹豫树自动机、两向线性极化犹豫树自动机、可见极化犹豫树自动机等），并证明它们精确捕捉 FO 的表达能力，还推导出安全/协同安全子句的自动机等价性，并给出 FO 在每条分支上只能表达安全或协同安全性质的边界定理。

**💡 创新点**

创新点包括：①提出极化犹豫树自动机和两向线性极化犹豫树自动机的概念；②给出 FO 与两类时序逻辑的完全等价证明；③在自动机层面得到 FO 的新规范形式（normal form）；④揭示 FO 在无限树上的“极化”性质以及安全/协同安全的表达边界；⑤通过可见性与计数无关性相结合，得到 FO 的完整字符化。

**🔧 技术方法**

使用技术：自动机理论（格雷德树自动机、犹豫树自动机、极化约束、两向移动、计数态）、时序逻辑语义与语法转换、归约到词自动机、计数无关性与可见性条件、巴克斯/科巴赫接受条件、归纳证明与构造翻译、正则表达式与树正则性分析。

**📊 数据集**

本研究为理论论文，未使用实验数据集；所有结果均通过形式化证明获得。

**📈 对比分析**

比较方法：通过构造互相可翻译的自动机和逻辑语句，证明它们在初始等价性（initial equivalence）下表达力相等；性能方面未涉及时间/空间实验指标，而是以表达力等价性和可计算性（有效翻译）为评价标准；证明中使用归纳构造、计数无关性证明、可见性证明等。

**⚠️ 局限性**

限制与未解问题：①FO 在无限树上只能表达安全或协同安全性质，无法描述更复杂的分支属性；②对有序树、仅子代/祖先关系的 FO 可定义性问题仍未完全解决；③极化犹豫树自动机的决策复杂度与闭包性尚待进一步研究；④如何在更一般的树结构（如有序树、排名树）中实现类似字符化仍是开放问题。

---

## 143. Point Cloud Registration via Probabilistic Self-Update Local Correspondence and Line Vector Sets

**arXiv ID:** 2604.26318 | [PDF](https://arxiv.org/pdf/2604.26318v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 144. Towards a Frugal Photosynthesis Sensing Toolkit for Data-Driven Plant Science Education and Exploration

**arXiv ID:** 2604.26305 | [PDF](https://arxiv.org/pdf/2604.26305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 145. Taking a Bite Out of the Forbidden Fruit: Characterizing Third-Party Iranian iOS App Stores

**arXiv ID:** 2604.26343 | [PDF](https://arxiv.org/pdf/2604.26343v1)

**作者:** Amirhossein Khanlari `[一作]` (Stony Brook University), Amir Rahmati `[通讯]` (Stony Brook University)

**通讯引用:** 4359 | [OpenAlex ID](https://openalex.org/A5021423602)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对伊朗三大第三方 iOS 应用商店进行全面实证研究，收集并分析1700多款应用的元数据、IPA 包，考察其运营机制、内容、破解情况与安全风险。

**💡 创新点**

首次构建系统化工具链，系统性描绘暗藏商店的分发与逃逸路径，量化盗版带来的收入损失与安全隐患，揭示制裁与审查对数字生态的深远影响。

**🔧 技术方法**

采用自研网页爬虫收集元数据、MITM 代理捕获 IPA、在越狱设备上解密 FairPlay、使用 MobSF 进行静态分析、对比 Apple Store 版本、记录网络流量。

**📊 数据集**

使用 1700+ 份第三方商店收集的 IPA 及其对应的 Apple Store 版本元数据，涵盖 3 个月的观察窗口，包含免费、破解、伊朗本土与全球应用。

**📈 对比分析**

通过 Bundle ID 匹配、文件级差异对比、黑客工具与第三方库注入检测、下载量与收入估算等方法评估差异；结果显示大部分破解应用含注入库、收入损失超过 500 万美元，覆盖率高但未分析付费与新 iOS 版本应用。

**⚠️ 局限性**

局限性：仅覆盖 3 家商店、免费或未加密的 IPA 可用、iOS 16.7.12 仅解密旧版应用、缺乏所有商店的下载统计、对付费版收益估算依赖一一替代率假设、可能忽略更深层次的隐私泄露。

---

## 146. A Systematic Comparison of Prompting and Multi-Agent Methods for LLM-based Stance Detection

**arXiv ID:** 2604.26319 | [PDF](https://arxiv.org/pdf/2604.26319v1)

**作者:** Genan Dai `[一作]` (Shenzhen Technology University), Bowen Zhang `[通讯]` (Shenzhen Technology University)

**通讯引用:** 5696 | [OpenAlex ID](https://openalex.org/A5100385156)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在统一实验框架下，系统比较了五种LLM驱动的立场检测方法（Prompt-based 与 Agent-based），并在四个英文数据集（共14个子任务）上对15种不同规模的模型进行评测。

**💡 创新点**

首次在相同数据拆分、指标与API调用次数下对 Prompt 与 Agent 方法进行对比，揭示 Agent 方法并不优于精心设计的 Prompt；同时系统分析模型规模、推理增强对性能的影响，并公开了完整的 per-target 结果作为基准。

**🔧 技术方法**

使用 Prompting（Direct Prompting、Auto‑CoT、StSQA）和多代理辩论（COLA、MPRF）技术，结合 15 种 LLM（包括 GPT‑3.5、GPT‑4o‑mini、DeepSeek、Llama、Qwen、Claude、Gemini 等），并采用关键词匹配提取标签、宏平均 F1 作为评价指标。

**📊 数据集**

四个英文立场检测数据集：SemEval‑2016（6 个目标）、P‑Stance（3 个政治人物）、COVID‑19‑Stance（4 个疫情相关目标）和 VAST（1,460 条多主题推文）。

**📈 对比分析**

在严格统一的评测协议下，Auto‑CoT 与 StSQA 取得最高宏 F1（≈73.7%），Prompt 方法整体优于 Agent 方法；模型规模提升至 32B 带来约 8–10 分的显著提升，超过方法改进的效果；Agent 方法调用量 7–12 倍但性能并不提高。

**⚠️ 局限性**

局限性包括：仅评测英文数据集；未包含监督微调基线；部分模型–方法组合因拒绝/超时导致结果不完整；闭源模型更新可能影响复现；多语言、跨域扩展仍待进一步研究。

---

## 147. CheXthought: A global multimodal dataset of clinical chain-of-thought reasoning and visual attention for chest X-ray interpretation

**arXiv ID:** 2604.26288 | [PDF](https://arxiv.org/pdf/2604.26288v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 148. SpatialFusion: Endowing Unified Image Generation with Intrinsic 3D Geometric Awareness

**arXiv ID:** 2604.26341 | [PDF](https://arxiv.org/pdf/2604.26341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. Event-based Liveness Detection using Temporal Ocular Dynamics: An Exploratory Approach

**arXiv ID:** 2604.26285 | [PDF](https://arxiv.org/pdf/2604.26285v1)

**作者:** Nicolas Mastropasqua `[一作]` (Universidad de Buenos Aires), Pablo Negri `[通讯]` (Universidad de Buenos Aires)

**通讯引用:** 1076 | [OpenAlex ID](https://openalex.org/A5062944593)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出基于事件相机的面部活体检测框架，利用眼部运动的时序特征实现挑战-响应式防spoofing；

**💡 创新点**

创新点在于首次将事件相机捕获的眼动时序信号用于远距离活体检测，并构建了专门的伪造数据集；

**🔧 技术方法**

使用的技术包括事件活动剖面、峰值检测、时间卷积网络(TCN)、尖峰卷积神经网络(SCNN)以及ViT等模型；

**📊 数据集**

数据集为扩展版RGBE-Gaze，加入了在显示器上播放的重放攻击序列，覆盖17名受试者；

**📈 对比分析**

与传统RGB方法比较，SCNN在事件体素网格上实现95.37% top‑1准确率，低于ViT帧基准的92.15%，且在APCER/BPCER上表现更优；

**⚠️ 局限性**

局限性包括仅针对单一显示器的重放攻击、样本规模有限、未涵盖印刷、面具或高级生成式攻击等场景，缺乏跨数据集验证。

---

## 150. Seamless Indoor-Outdoor Mapping for INGENIOUS First Responders

**arXiv ID:** 2604.26368 | [PDF](https://arxiv.org/pdf/2604.26368v1)

**作者:** Jürgen Wohlfeil `[一作]` (German Aerospace Center), Dennis Dahlke `[通讯]` (German Aerospace Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

结合MACS‑SaR无人机与IPS系统，实现在无GNSS环境下的室内外无缝3D建模，并支持实时点云共享与可视化。

**💡 创新点**

创新点在于利用空中识别的AprilTags作为全局光学参考点，将其地理坐标直接传递给IPS，完成无外部定位的全球坐标对齐，实现室内外点云的实时无缝配准。

**🔧 技术方法**

使用高分辨率相机+双天线GNSS+IMU的无人机测绘、立体视觉+IMU融合的IPS、AprilTags识别、Bundle Adjustment、Semi‑Global Matching、点云配准与可视化技术。

**📊 数据集**

采用比利亚港附近现场灾害场景数据，包括MACS航拍图像、IPS立体相机图像及空中放置的AprilTags测量点。

**📈 对比分析**

通过与GNSS测量的AprilTags坐标比较，绝对误差约为x‑0.23 m、y‑0.30 m、z‑1.0 m，相关误差5–10 cm；两系统点云在全球坐标系下对齐后可视化误差可忽略，满足救援操作需求。

**⚠️ 局限性**

限制在于需要在建筑前放置可见的AprilTags，场地与遮挡受限；自然地标可靠性不足；在极端光照或遮挡条件下算法鲁棒性受影响。

---

## 151. Adaptive and Fine-grained Module-wise Expert Pruning for Efficient LoRA-MoE Fine-Tuning

**arXiv ID:** 2604.26340 | [PDF](https://arxiv.org/pdf/2604.26340v1)

**作者:** Weihang Li `[一作]` (University of Science and Technology of China), Hongli Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 40673 | [OpenAlex ID](https://openalex.org/A5052217844)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DMEP 框架，利用动态模块级专家剪枝实现 LoRA-MoE 的高效微调。

**💡 创新点**

创新点在于在线收集路由统计、按模块精准剪枝以及在剪枝后彻底关闭负载平衡，从而兼顾参数效率与专家专化。

**🔧 技术方法**

采用 LoRA、Mixture-of-Experts、Top‑k 路由、Gini 系数/熵/漂移度量、结构化物理裁剪与优化器状态同步等技术。

**📊 数据集**

在 Qwen3‑0.6B 与 Qwen3‑8B 上使用 ScienceQA、OpenBookQA、GSM8K 三个推理任务的数据集进行实验。

**📈 对比分析**

与 Dense LoRA 与对称 MoE 基线对比，DMEP 在保持或提升准确率的同时，参数量减少 35%–43%，训练吞吐量提升约 10%。

**⚠️ 局限性**

局部任务仍可能因过度剪枝导致精度略降，且对不同规模模型的最优阈值和剪枝时机仍需进一步调优。

---

## 152. Efficient, VRAM-Constrained xLM Inference on Clients

**arXiv ID:** 2604.26334 | [PDF](https://arxiv.org/pdf/2604.26334v1)

**作者:** Aditya Ukarande `[一作]` (NVIDIA Corporation), Ram Rangan `[通讯]` (NVIDIA Graphics Pvt Ltd)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在有限 VRAM 的客户端 GPU 上，设计并实现了一种基于子层拆分与标记式调度的流水线分区（Pipelined Sharding）技术，并将其与三种 VLM 优化（Vision Tensor Offload、FlashAttention、VRAM 友好序列化）结合，实现在任意用户指定 VRAM 预算下的高精度 LLM 与 VLM 推理。

**💡 创新点**

创新点包括①基于离线基准与动态系统条件的 profile‑guided 三级调度器，能根据 token 级别自动选择最优的 CPU/GPU/PCIe 组合；②无损的子层级分区与优先级映射，使得不同算子在显存、计算与数据传输上得到最优平衡；③针对 VLM 的三项可组合优化，使得原本显存占用巨大的视觉编码器在 2–3 GB VRAM 仍能高分辨率运行。

**🔧 技术方法**

技术栈包括：Llama.cpp 作为推理框架；CUDA 与 CiG；自研的基准与 Roofline 模型用于 CPU/GPU/PCIe 成本估算；FlashAttention；多线程 CPU 共享内存；以及多种预设调度计划（全 GPU、CPU+GPU、混合）和 token‑tier 选取策略。

**📊 数据集**

评估使用的主要数据是自定义文本上下文（1K~64K token）和图像分辨率（480p~1440p），模型覆盖 NVIDIA IGI SDK、Cosmos‑Reason1 VLM、Nemo‑4/8 B、Qwen‑30/235 B；未使用公开标准数据集，而是直接测量模型推理吞吐和显存占用。

**📈 对比分析**

对比方法：在 Windows 11 上的三台客户端机器(cli1/cli2/cli3)分别与 llama‑cpp 手动 CPU offload（-ngl、-cmoe 等）和 vLLM 基线对比；结果显示：在 2–32 GB VRAM 范围内，TTFT 提升 2×~6.7×，TPS 提升 3.7×~30×；VLM 在 2 GB 预算下即可完成推理，VRAM 需求降低约 10×，同时保持交互式 5 TPS 以上的吞吐。

**⚠️ 局限性**

局限性：①需在安装时完成一次完整基准，硬件/驱动变更后需重新 profile；②对极低显存或极高 PCIe 带宽下的极端场景支持有限；③仅支持 CUDA GPU，未覆盖 NPU/ARM 等异构平台；④热量/功耗动态变化未自适应；⑤在多 GPU 或 vLLM 环境下的多任务调度仍需进一步研究。

---

## 153. High-Dimensional Noise to Low-Dimensional Manifolds: A Manifold-Space Diffusion Framework for Degraded Hyperspectral Image Classification

**arXiv ID:** 2604.26279 | [PDF](https://arxiv.org/pdf/2604.26279v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 154. Classification of Public Opinion on the Free Nutritional Meal Program on YouTube Media Using the LSTM Method

**arXiv ID:** 2604.26312 | [PDF](https://arxiv.org/pdf/2604.26312v1)

**作者:** Berliana Enda Putri `[一作]` (Institut Teknologi Sumatera), Martin Clinton Tosima Manullang `[通讯]` (Institut Teknologi Sumatera)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用长短时记忆网络（LSTM）对两条YouTube频道的7733条印尼语评论进行情感分类，评估对“免费营养餐计划”（MBG）的公众意见。

**💡 创新点**

创新之处在于首次将LSTM应用于印尼语YouTube评论情感分析，并针对数据不平衡问题提出了改进方向。

**🔧 技术方法**

使用了NLP预处理（去噪、分词、词干提取）、词向量嵌入、LSTM模型，并与传统机器学习方法（SVM、Logistic回归、朴素贝叶斯）做对比。

**📊 数据集**

数据集为从两条讨论MBG的YouTube频道抓取的7733条评论，其中87.7%为负面、12.3%为正面。

**📈 对比分析**

通过混淆矩阵、精确率、召回率和F1得分进行评估，LSTM模型在测试集上达到了89%准确率，负面情感F1高达0.94，而正面情感F1仅为0.55，显示出对负面类识别优秀但对正面类表现不足。

**⚠️ 局限性**

主要局限在于数据严重不平衡，正面样本稀少导致模型偏向负面类，需采用SMOTE、类别权重或收集更多正面样本来提升性能。

---

## 155. Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference

**arXiv ID:** 2604.26294 | [PDF](https://arxiv.org/pdf/2604.26294v1)

**作者:** Vasu Shyam `[一作]` (Zyphra), Quentin Anthony `[通讯]` (Zyphra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出并实现了一种将张量并行与序列并行折叠到同一设备轴的并行策略（TSP），通过在单一轴上同时划分权重与序列来降低模型和激活的内存占用

**💡 创新点**

创新点在于：①折叠张量与序列并行，单轴同时分片；②为注意力与门控MLP设计了广播与环形通信调度；③兼顾内存与通信，适配高带宽节点内互连

**🔧 技术方法**

使用了MPI/NCCL等集体通信、广播与点对点环形通信、FlashAttention、Recomputation、Zigzag序列分区等技术

**📊 数据集**

主要在Mi300X GPU集群上进行实验，使用大规模Transformer模型（如7B参数模型）进行训练与推理，未专门提及公开数据集

**📈 对比分析**

与传统的张量并行（TP）、序列并行（SP）及其两轴组合（TP+SP）在同一硬件平台上比较；结果显示TSP在所有测试序列长度下内存占用最低，吞吐量保持竞争甚至更优，尤其在长上下文和大并行度下优势更明显

**⚠️ 局限性**

限制在于额外的权重传输导致通信量上升，需要与计算重叠；对低带宽跨节点拓扑的适配性不一定最佳；在某些场景下通信隐藏不够充分，可能影响性能

---

## 156. Enforcing Benign Trajectories: A Behavioral Firewall for Structured-Workflow AI Agents

**arXiv ID:** 2604.26274 | [PDF](https://arxiv.org/pdf/2604.26274v1)

**作者:** Hung Dang `[一作]` (Van Lang University), Hung Dang `[通讯]` (Van Lang University)

**通讯引用:** 746 | [OpenAlex ID](https://openalex.org/A5013241108)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于电磁学的行为防火墙，利用从正常工具调用轨迹构建参数化确定有限自动机（pDFA），在运行时以 O(1) 检查并拦截恶意工具调用。

**💡 创新点**

将传统无状态扫描与有状态序列建模相结合，采用 pDFA 记录工具调用上下文和参数边界，实现对上下文序列注入攻击的完全阻断，同时保持低延迟。

**🔧 技术方法**

使用语义参数范围推断、SBERT 文本嵌入、参数化 DFA、离线采样、在线 O(1) 结构转移、哈希查表、SHA-256 证据链以及可插拔侧车架构。

**📊 数据集**

在 Agent Security Bench (ASB) 进行攻击测试，并用 ToolBench 生成 500 条正常工具调用轨迹进行训练与评估。

**📈 对比分析**

与无防火墙、Aegis 以及 PromptArmor 对比，宏平均攻击成功率从 79% 降至 2.2%（三种结构场景），单调用平均延迟 2.2 ms，比 Aegis 低 3.7 倍，真阳性率仅 2.0%。

**⚠️ 局限性**

仅对低熵、单任务工具集有效；参数边界易受同义词替换攻击；需要完整干净的离线采样；对概念漂移需人工增量更新；不支持多模态或多智能体环境。

---

## 157. Which Face and Whose Identity? Solving the Dual Challenge of Deepfake Proactive Forensics in Multi-Face Scenarios

**arXiv ID:** 2604.26342 | [PDF](https://arxiv.org/pdf/2604.26342v1)

**作者:** Lei Zhang `[一作]` (Xinjiang University), Gaobo Yang `[通讯]` (Hunan University)

**通讯引用:** 9387 | [OpenAlex ID](https://openalex.org/A5089193327)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了Deep Attributable Watermarking Framework (DAWF)，通过多脸并行编码解码器和选择性区域监督，完成多脸深度伪造的定位与来源追溯；

**💡 创新点**

①多脸并行嵌入水印消除裁剪与缩放开销；②选择性区域监督损失聚焦于被伪造区域，提升定位精度；③同时实现“哪张脸被篡改（which）+谁是篡改者（who）”双任务；

**🔧 技术方法**

U‑Net 结构的隐写核、多脸 Encoder‑Decoder、RaLSGAN 对抗训练、交叉噪声池、双分支解码器（Tracer + Localizer）以及基于 IoU 的选择性区域监督；

**📊 数据集**

训练集：WIDERFace；评估集：COCO2017、OpenForensics、CelebA‑HQ；对抗攻击集：SimSwap、Ghost、MobileFaceSwap、CSCS 等；

**📈 对比分析**

与传统单脸水印（KAD‑Net、FaceSigns、WaveGuard）、主动防御（EditGuard、OmniGuard）及被动检测（MVSS‑Net、IML‑Net、PIM）等多种方法对比；DAWF 在 F1、AUC、BER_tr 等指标上均显著优于基线，尤其在多脸场景下实现零误报且定位精度高；

**⚠️ 局限性**

受面部检测精度限制；水印容量与图像分辨率相关；在极端遮挡或高分辨率多脸场景下性能下降；对新型伪造技术的鲁棒性仍需进一步验证。

---

## 158. Beyond Fixed Formulas: Data-Driven Linear Predictor for Efficient Diffusion Models

**arXiv ID:** 2604.26365 | [PDF](https://arxiv.org/pdf/2604.26365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 159. CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID

**arXiv ID:** 2604.26363 | [PDF](https://arxiv.org/pdf/2604.26363v1)

**作者:** Fengchun Zhang `[一作]` (University of Electronic Science and Technology of China), Jianwei Hu `[通讯]` (QiYuan Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CO-EVO 框架，在联邦域泛化场景下协同训练人像 ReID 模型。

**💡 创新点**

通过共进化的语义锚定（Camera‑Invariant Semantic Anchoring）与风格多样化（Global Style Diversification）双向调节，解决语义‑风格冲突。

**🔧 技术方法**

使用 CLIP 文本提示构建纯净身份锚点，利用全局相机风格库实现无生成器的真实风格增强，并在联邦学习中实现模型聚合。

**📊 数据集**

在 CUHK02、CUHK03、MSMT17、Market1501 四个大规模 ReID 数据集上进行实验。

**📈 对比分析**

与 SOTA 基线（如 DACS、SSCU 等）对比，CO‑EVO 在离域泛化任务中平均提升约 2% mAP（最高 45.4%），在 Rank‑1 上提升约 3%，在多种评估协议下保持领先。

**⚠️ 局限性**

局限：对极端遮挡、低分辨率或 OOD 外观的语义锚点鲁棒性不足；GSD 仅覆盖光照与颜色变化，无法模拟几何变形；对摄像头元数据的依赖在元数据缺失或噪声严重时仍可能导致性能下降。

---

## 160. Uncertainty-Aware Reward Discounting for Mitigating Reward Hacking

**arXiv ID:** 2604.26360 | [PDF](https://arxiv.org/pdf/2604.26360v1)

**作者:** Disha Singha `[一作]` `[通讯]`, Disha Singha

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种双源不确定性奖励折扣框架 UARD，利用模型不确定性和人类偏好不确定性来调整奖励信号，抑制奖励劫持。

**💡 创新点**

创新点在于将不确定性视为主动约束，通过可靠性过滤器将模型和人类两源不确定性融合到奖励中，实现主动折扣而非仅做探索或正则化。

**🔧 技术方法**

使用集成估计（多头 Q 估计器）得到模型不确定性，合成多注释者生成的人类不确定性，再通过置信度调节的可靠性过滤器计算行动得分，结合 ε‑greedy 策略。

**📊 数据集**

实验数据集包括离散网格世界 6×6/8×8/10×10 的多陷阱配置，以及连续控制任务 Hopper‑v4、Walker2d‑v4。

**📈 对比分析**

与 PPO、SAC、EDAC、SUNRISE 等基线对比，UARD 在网格世界中将奖励劫持次数降低 93.7%，在连续任务中保持对齐目标并抑制异常奖励，且在 30% 监督噪声下仍保持低违规。

**⚠️ 局限性**

主要限制包括计算开销高（需多头集成），超参数固定可能不适用于不同阶段或环境，且未验证真实人类反馈或感知噪声场景。

---

## 161. UIGaze: How Closely Can VLMs Approximate Human Visual Attention on User Interfaces?

**arXiv ID:** 2604.26352 | [PDF](https://arxiv.org/pdf/2604.26352v1)

**作者:** Min Song `[一作]` (Xebec Inc.), Yeonhu Seo `[通讯]` (Xebec Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在用户界面上，Vision Language Models（VLMs）在零样本条件下预测人类视觉注意力的可行性。

**💡 创新点**

创新点在于首次使用真实眼动数据评估多种VLM的注视预测能力，并揭示其对不同UI类型和观看时长的适应性。

**🔧 技术方法**

采用零样本坐标预测管线，VLM通过生成注视点坐标并高斯模糊生成显著性图。

**📊 数据集**

使用UEyes数据集，该数据集包含1980张UI截图（网页、桌面、移动、海报）和62位参与者的眼动记录。

**📈 对比分析**

对比九种主流VLM的CC、SIM、KL三种指标，结果显示GPT‑5.4在7秒时达CC≈0.408、SIM≈0.503、KL≈1.345，整体与人类注视有中等程度的一致性。

**⚠️ 局限性**

局限在于VLM只能捕捉探索性注视，难以预测首次注视；缺乏针对不同人群的个体差异；且与经过领域微调的专用显著性模型相比性能仍有差距。

---

## 162. Asymptotically Robust Learning-Augmented Algorithms for Preemptive FIFO Buffer Management

**arXiv ID:** 2604.26349 | [PDF](https://arxiv.org/pdf/2604.26349v1)

**作者:** Wen-Han Hsieh `[一作]` (National Cheng Kung University), Ya-Chun Liang `[通讯]` (National Tsing Hua University)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5037147386)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一种学习增强的FIFO缓冲区管理算法，能够在预测完美时实现一致性1，并在预测错误时以平滑方式退回至经典最优竞争比√3；

**💡 创新点**

提出了基于输出的预测误差度量，避免了传统输入误差对未传输数据的误判；同时引入缓冲区清空机制与动态保护回退，保证在任何预测错误下仍保持最坏情况的√3竞争比；

**🔧 技术方法**

利用在线竞争分析、输出差异误差度量、动态阈值检测和缓冲区清空策略，并在框架中可替换任意β-竞争在线算法作为后备；

**📊 数据集**

本文为理论研究，无实测数据集；

**📈 对比分析**

与经典的Preemptive Greedy (2+√3)和其他已知缓冲管理算法对比，证明在完美预测下实现1的竞争比，在误差增大时竞争比随误差上升但上限为√3，且整体性能优于传统方法；

**⚠️ 局限性**

仍然只能在最坏情况下达到√3竞争比；未给出理论下界；缺乏实际网络实验验证；

---

## 163. AlphaJet: Automated Conceptual Aircraft Synthesis via Disentangled Generative Priors and Topology-Preserving Evolutionary Search

**arXiv ID:** 2604.26337 | [PDF](https://arxiv.org/pdf/2604.26337v1)

**作者:** Boris Kriuk `[一作]` (Hong Kong University of Science and Technology), Boris Kriuk `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5093836299)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 AlphaJet，一个完整的自动化概念机型合成框架，能够根据任务说明在几分钟内生成符合物理约束的 3D 飞机模型。

**💡 创新点**

创新点在于三方面：① 使用解耦的 Anatomically-Disentangled VAE 作为形状先验并对 25 个解剖参数进行监督；② 采用拓扑保持的遗传算法，保证五种尾翼拓扑均被保留并引入停滞重启；③ 设计了基于签名穿透深度的挂载一致性评分，消除模型中“漂浮部件”的现象。

**🔧 技术方法**

核心技术包括解耦 VAE、拓扑剔除的遗传算法、基于体素的解析几何化、闭式低阶多物理评估以及 WebSocket 实时可视化。

**📊 数据集**

使用 4,000 构造的合成机型数据集（多种尾翼、发动机布置和气动参数）训练 AD‑VAE，并在实验中对三类任务（区域客机、商务喷气机、长航程无人机）进行评估。

**📈 对比分析**

通过消融实验验证拓扑精英化、挂载评分和 VAE 先验的必要性，结果显示在 40–70 代内即可得到满足所有约束的方案；在单 CPU 上完成时间不足 5 分钟，且与传统手工迭代相比显著提升效率。

**⚠️ 局限性**

局限性包括：仅使用低阶物理模型，无法捕捉细粒度气动/结构效应；先验基于合成数据，可能限制了对真实航空器多样性的覆盖；以及未与高阶 CFD 或 CAD 交互，需进一步集成以实现更精细的后期设计。

---

## 164. Distributional Learning of Graph Languages Generated by Fixed-Interface Clause Systems

**arXiv ID:** 2604.26333 | [PDF](https://arxiv.org/pdf/2604.26333v1)

**作者:** Takayoshi Shoudai `[一作]` (Fukuoka Institute of Technology), Tomoyuki Uchida `[通讯]` (Hiroshima City University)

**通讯引用:** 1545 | [OpenAlex ID](https://openalex.org/A5100541710)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个基于固定接口子句系统的分布式学习框架，用来从正样本与成员查询中识别图语言，并给出了完整的学习算法及其正确性与多项式时间更新的证明。

**💡 创新点**

创新点包括：① 明确了界面结构并引入有序边界表示，显式化了先前隐式的分割信息；② 用参数元组(Δ,m,s,t,w,d)对学习所需的结构限制进行了显式化；③ 在不依赖树宽度的前提下完成了学习与多项式时间更新的证明；④ 对原ILP 2016工作完成了完整证明并将其扩展为完整期刊版本。

**🔧 技术方法**

采用的技术主要是：固定接口子句系统与图模式、基于成员查询的观测表、有限上下文性质、参数化复杂度分析、以及对图同构与边界表示的算法实现。

**📊 数据集**

论文为理论性工作，没有使用具体的数据集；所有结果均在抽象图语言模型和正样本序列的理论框架下给出。

**📈 对比分析**

方法与其他学习框架的比较在理论层面完成：证明了该类语言可在极限中被识别，且学习算法在更新阶段的计算复杂度为多项式；未给出实验性能对比。

**⚠️ 局限性**

局限性：仅处理一元谓词；需要满足有限上下文与有界度假设；度安全条件仅为充分条件，尚未给出更弱的保证；目前仅在理论层面验证，未涉及实际数据集与实验评估。

---

## 165. MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution

**arXiv ID:** 2604.26283 | [PDF](https://arxiv.org/pdf/2604.26283v1)

**作者:** Chunzheng Zhu `[一作]` (Hunan University), Yijun Wang `[通讯]` (Hunan University)

**通讯引用:** 2822 | [OpenAlex ID](https://openalex.org/A5100713692)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种医学视觉语言模型 MedSynapse‑V，通过在隐藏层中动态演化隐式诊断记忆来模拟临床专家的即时经验调用，从而实现高精度且低延迟的诊断推理。

**💡 创新点**

创新点包括：① Meta Query for Prior Memorization 用可学习的查询探针从冻结的解剖编码器提取并压缩多尺度空间先验；② Causal Counterfactual Refinement 通过强化学习和区域级因果奖励对记忆进行剪枝与校准；③ Intrinsic Memory Transition 在双分支蒸馏框架下将外部先验转化为内部可自治的记忆，最终实现无外部编码器推理。

**🔧 技术方法**

技术方法包括：跨模态注意力聚合、强化学习策略优化（GRPO）、因果对比奖励、Jensen–Shannon 散度蒸馏、LoRA 微调、MedSAM3 先验编码器、可学习的记忆采样器和自律记忆模块。

**📊 数据集**

使用的数据集涵盖七大医学多模态基准：VQA‑RAD、SLAKE、PathVQA、PMC‑VQA、MMMU（Health & Medicine track）、MedXpertQA‑MM、GMAI‑MMBench，训练集包括 PubMedVision、OmniMedVQA、SLAKE、PathVQA 等，验证时严格区分训练与测试。

**📈 对比分析**

与通用 VLM、医学专用 VLM、RL‑增强 CoT 与通用潜在推理方法对比，MedSynapse‑V 在所有基准上平均提升约 11–12 pp，尤其在 VQA‑RAD、SLAKE、PathVQA 上提升 9–14 pp；在不使用解码器的 IMT 版本仍保持 59.6 % 的准确率，仅比基线低 1.8 pp，且推理速度与标准 VLM 相近（≈2.6 s/样本）。

**⚠️ 局限性**

局限性包括：① 记忆容量固定为 16 维向量，难以处理多病灶或极其复杂的临床场景；② 依赖 MedSAM3 的区域掩码，掩码质量会影响因果奖励；③ 在少见模态（如 OCT）上表现相对较弱；④ 需要多阶段训练，训练成本相对较高。

---

## 166. SWE-Bench 5G: Benchmarking AI Coding Agents on Telecom Network Engineering Tasks

**arXiv ID:** 2604.26278 | [PDF](https://arxiv.org/pdf/2604.26278v1)

**作者:** Jiao Chen `[一作]` (Shenzhen Smart City Technology Development Group Company, Ltd), Zuohong Lv `[通讯]` (China Unicom Group Co., Ltd)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了SWE-Bench 5G，首个针对5G核心网软件工程的 AI 编码代理评测基准；

**💡 创新点**

创新点包括：①基于三大开源 5G 核心项目构建真实 bug 任务；②双重测试策略兼顾直接调用与差分意图；③通过 3GPP 规范片段注入实现域知识的可控评估；

**🔧 技术方法**

采用 Docker 化的评测管道，结合四款大型语言模型（Qwen3.5-Flash、Kimi-128k、GPT‑4.1、Claude Sonnet 4）进行单轮与多轮（K=5）调试；

**📊 数据集**

使用 210 个已验证的 bug 样本，来源于 free5GC、Open5GS 与 Magma 三大开源 5G 核心实现，并包含 3GPP 规范引用，数据已发布在 Hugging Face；

**📈 对比分析**

通过单轮与多轮实验对比发现：所有模型在诊断率>91%但仅有 10–30% 的实例被成功修复；Claude Sonnet 4 以 30% 的修复率领先；在 50 个带 3GPP 规范注入的样本上，规范提升整体修复率约 6%，对依赖规范的 bug 有显著加成；

**⚠️ 局限性**

局限性在于：数据集规模有限（仅 210 条 bug），缺乏跨 NF 协调场景；对 Go/C 语言的严格类型与精确 patch 匹配限制了修复成功率；评测主要聚焦单一 NF 内部 bug，未涵盖多 NF 交互与完整系统级错误。

---

## 167. PiLLar: Matching for Pivot Table Schema via LLM-guided Monte-Carlo Tree Search

**arXiv ID:** 2604.26356 | [PDF](https://arxiv.org/pdf/2604.26356v1)

**作者:** Yunjun Gao `[一作]` (Zhejiang University), Yifan Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 13160 | [OpenAlex ID](https://openalex.org/A5100654265)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在隐私受限环境下，如何对齐 pivot 表与标准关系表的联合 schema‑value 匹配问题，并提出了一种基于 LLM 引导的 Monte Carlo Tree Search（MCTS）搜索框架 PILLAR，能够同时识别未 pivot 的属性集合并完成最终匹配；

**💡 创新点**

创新点包括：1）将未 pivot 属性识别与 schema 匹配融合为自纠正的迭代搜索；2）提出 bounded‑stochastic MCTS 与 Self‑Refine 结合的搜索范式，并给出理论收敛证明；3）构建 PTbench 四域 benchmark；4）在仅使用极少匿名数据的情况下实现无训练的跨域适配；

**🔧 技术方法**

采用的核心技术有：大型语言模型（如 Qwen3）做提示式推理；bounded‑stochastic MCTS（含随机邻域扩展与 max‑average 回传）；Self‑Refine 机制进行逐步优化；最大加权二分匹配（Jonker‑Volgenant）评估匹配奖励；多维相似度（词法、语义、分布）与 Jensen‑Shannon 结合来评估值兼容性；以及理论误差动态分析；

**📊 数据集**

使用了 PTbench 四个真实数据集：Adult、Football、President、Gene；每个数据集均包含 pivot 表和标准表，并提供匿名化采样记录；为规模实验，还使用了 M5 Forecasting 数据集（约 2000 列）；

**📈 对比分析**

实验与 COMA 3.0、DisB、GRAM、NaiveP 等基线进行对比；在包含未 pivot 属性的情形下，PILLAR 的平均 End‑to‑End 准确率为 87.94%，属性级准确率为 94.45%，比 NaiveP 提升 15%+；在不含未 pivot 属性时几乎达到 100% 与基线相当；表明框架在全属性与子属性场景均表现优异；

**⚠️ 局限性**

局限性包括：1）依赖大型 LLM 推理，计算成本和延迟较高；2）在极大搜索空间（如 Gene 数据集）下迭代收敛速度较慢；3）目前仅支持单一 unpivot 转换，未覆盖更复杂的映射；4）隐私保护仅靠匿名化，未进一步保证数据安全。

---

## 168. On the Capacity of Hierarchical Secure Aggregation with Groupwise Keys

**arXiv ID:** 2604.26344 | [PDF](https://arxiv.org/pdf/2604.26344v1)

**作者:** Minyang Lu `[一作]` (Guangxi University), Min Xie `[通讯]` (Guangxi University)

**通讯引用:** 51553 | [OpenAlex ID](https://openalex.org/A5100330523)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了具有分层结构且使用群组键的安全聚合问题，给出了通信率与键率的最优容量区域。

**💡 创新点**

首次在分层网络中将群组键与线性编码相结合，实现了与平面网络同等的密钥利用率，并给出了匹配的逆推下界。

**🔧 技术方法**

采用线性预编码矩阵、零和约束、Schwartz–Zippel引理证明矩阵满秩以及信息理论的互信息与熵计算。

**📊 数据集**

本研究为纯理论分析，无使用具体数据集；所有结果基于信息理论模型与符号扩展。

**📈 对比分析**

通过与已知的平面分布式聚合方案对比，证明当 G>1 时可实现 R_X,R_Y≥1 且键率 R_S 满足给出的最大化表达式，且所给方案与逆推下界完全匹配，达到最优。

**⚠️ 局限性**

局限性包括对大域字段的依赖、未考虑用户掉线、容错与T-安全（协作攻击）以及非均匀连接等实际情况。

---

## 169. Can Cross-Layer Design Bridge Security and Efficiency? A Robust Authentication Framework for Healthcare Information Exchange Systems

**arXiv ID:** 2604.26339 | [PDF](https://arxiv.org/pdf/2604.26339v1)

**作者:** Khalid M. Ezzat `[一作]` (Air Defense College, Military Academy), Mahmoud A. Shawky `[通讯]` (University of Essex)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了跨层身份认证方案，结合ECC PKI和PHY层特征（CFO、四象限偏移）以及机器学习实现轻量化连续身份验证，专为医疗信息交换网络打造。

**💡 创新点**

创新点：①初始握手使用ECC PKI+物理层硬件指纹；②重认证仅提取CFO/四象限偏移并用已训练ML模型实时识别；③采用加密伪身份并频繁刷新，实现隐私与unlinkability；④使用BAN逻辑形式验证协议安全；⑤在资源受限医疗环境下实现极低计算和通信开销。

**🔧 技术方法**

技术手段：ECC（secp160k1）PKI、Diffie‑Hellman、AES‑256加密；OFDM信号的CFO提取（Van de Beek算法）和四象限偏移计算；机器学习模型（KNN优选）用于身份识别；BAN逻辑进行协议安全分析。

**📊 数据集**

数据集：基于OFDM仿真生成的10/20/30设备信号，提取CFO和四象限偏移特征，用于训练和测试ML模型；未使用真实医疗设备的硬件指纹数据。

**📈 对比分析**

比较方法：与Qi‑Xie、Xiang、Kumar、Chen等已有方案在计算开销（以ECC乘法/哈希/加解密时间计）和通信开销（以字节数计）进行对比；实验显示本方案初始握手仅7.45 ms，重认证仅解密，通信开销为144+20n+256⌈n/d⌉ bytes，均显著低于对比方案；ML模型在10/20设备下≈89–90%准确率，30设备约79%，KNN表现最佳。

**⚠️ 局限性**

局限性：仿真仅在室内5G OFDM环境下进行，缺乏真实医疗设备硬件指纹采集与验证；对极端噪声、多径或干扰环境的鲁棒性未充分评估；需进一步研究分布式信任管理、区块链集成以及大规模多机构部署的可扩展性。

---

## 170. DSIPA: Detecting LLM-Generated Texts via Sentiment-Invariant Patterns Divergence Analysis

**arXiv ID:** 2604.26328 | [PDF](https://arxiv.org/pdf/2604.26328v1)

**作者:** Siyuan Li `[一作]` (Shanghai Jiao Tong University), Jianhua Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 25574 | [OpenAlex ID](https://openalex.org/A5100613889)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、零样本的黑盒检测框架，利用情感分布在低情绪改写下的稳定性来识别LLM生成文本

**💡 创新点**

创新点在于将情感分布一致性（Sentiment Distribution Consistency）和情感分布保留性（Sentiment Distribution Preservation）作为可解释的行为信号，完全不依赖模型参数或标签，且在跨模型、跨域、对抗攻击下保持稳健

**🔧 技术方法**

核心技术包括：1) 通过零样本提示实现低情绪改写；2) 使用零样本情感分析提取情感特征向量；3) 计算情感分布的一致性与保留度（SDC、SDP）并以阈值决策；4) 对多重改写结果求平均，提升鲁棒性

**📊 数据集**

实验使用五大域数据集：新闻（真实与LLM生成）、程序代码（含注释）、学生论文、学术摘要、Yelp评论，共计约1万条样本

**📈 对比分析**

与GPTZero、Ghostbuster、DetectGPT、Fast‑DetectGPT、Binoculars、RAIDAR、R‑Detect等主流方法比较，DSIPA在所有域、所有LLM（如GPT‑3.5、Claude、Gemini、LLaMA‑2）上均实现最高F1，最高提升约49.89%，并在对抗改写、跨语言回译、不同文本长度下保持显著优势

**⚠️ 局限性**

局限性包括：在极短文本和情感表达受限的域（如纯技术代码）下识别率下降；未来LLM若进一步模仿人类情绪动态，情感稳定性差异可能缩小，需结合其他行为信号提升鲁棒性

---

## 171. Addressing Performance Saturation for LLM RL via Precise Entropy Curve Control

**arXiv ID:** 2604.26326 | [PDF](https://arxiv.org/pdf/2604.26326v1)

**作者:** Bolian Li `[一作]` (Purdue University), Ruqi Zhang `[通讯]` (Purdue University)

**通讯引用:** 4550 | [OpenAlex ID](https://openalex.org/A5101586017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于拒绝采样的熵控制方法 Entrocraft，防止强化学习过程中的熵坍塌，从而缓解性能饱和。

**💡 创新点**

创新点在于利用优势分布的偏移来精确调节熵曲线，且能通过简单的熵目标调度（线性衰减）实现长期稳定训练。

**🔧 技术方法**

主要技术包括：理论推导优势与熵变化的关系、基于熵阈值的动态拒绝采样、熵曲线线性调度，以及在 GRPO/GSPO 等策略梯度框架中的无缝集成。

**📊 数据集**

使用 Numina‑Math（440K 题目）作为训练集，评估时结合 MATH‑500、AMC‑23 与 AIME‑25/26 这几类标准数学推理测试集。

**📈 对比分析**

与熵正则化、剪裁、正负分离等现有方法对比，Entrocraft 在 4B 模型上击败 8B 基线，pass@K 提升 50%，且训练进度可延长四倍，整体性能明显优于其它熵保持方案。

**⚠️ 局限性**

局限性包括：仅在 LLM 策略梯度框架内验证；需要手工调节熵目标与温度参数；未在非数学推理任务或更大模型规模上进行广泛验证。

---

## 172. DreamProver: Evolving Transferable Lemma Libraries via a Wake-Sleep Theorem-Proving Agent

**arXiv ID:** 2604.26311 | [PDF](https://arxiv.org/pdf/2604.26311v1)

**作者:** Youyuan Zhang `[一作]` (University of Toronto), Xujie Si `[通讯]` (University of Toronto)

**通讯引用:** 436 | [OpenAlex ID](https://openalex.org/A5059074509)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于“觉醒-睡眠”循环的自动定理证明代理，能够在训练过程中逐步发现、抽象和压缩可迁移的中间定理（lemma），从而构建紧凑、可复用的lemma库。

**💡 创新点**

创新点在于：①将定理证明与库学习整合为双阶段的循环，既利用现有lemma推进证明，又在睡眠阶段通过聚类、抽象和验证生成更通用的lemma；②通过结构相似度与语义嵌入实现自动化的lemma聚类与筛选；③在保持低上下文成本的同时显著提升证明成功率。

**🔧 技术方法**

主要技术包括：大语言模型（LLM）推理与子目标分解、语义嵌入与K-means聚类、结构相似度评估（树编辑距离、逻辑树相似度）、最少最近使用（LRU）遗忘策略、以及轻量级推理流程（直接推理 + Sketch-and-Prove）。

**📊 数据集**

使用了多领域数学基准：不等式（567NEQ、ChenNEQ、MO-INT）、数论（PutnamBench、ProverBench）、组合学（CombiBench）、几何（LeanGeo-Bench）和机器学习理论（FormalML），每个领域均采样 100 个训练题。

**📈 对比分析**

与三类基线（专有 LLM、开源推理 LLM、代理系统 Hilbert）对比，取得平均 61% 的成功率提升；证明长度平均减少 32%-50%；令牌使用量平均降低 48%-62%；在低数据、低知识覆盖域（几何、ML 理论）上，提升 64%-161%。

**⚠️ 局限性**

局限性包括：1）目前lemma库规模较小，适用于大模型上下文；2）在上下文窗口受限的小模型上可复用性不足；3）缺乏针对未知领域的快速检索机制；4）在极少训练数据的科研场景下，在线学习效率待提升。

---

## 173. Text Style Transfer with Machine Translation for Graphic Designs

**arXiv ID:** 2604.26361 | [PDF](https://arxiv.org/pdf/2604.26361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 174. Cheeger--Hodge Contrastive Learning for Structurally Robust Graph Representation Learning

**arXiv ID:** 2604.26301 | [PDF](https://arxiv.org/pdf/2604.26301v1)

**作者:** Mengyang Zhao `[一作]` (Shandong University), Cunquan Qu `[通讯]` (Shandong University)

**通讯引用:** 302 | [OpenAlex ID](https://openalex.org/A5088741789)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在无监督图表示学习中，提出 Cheeger–Hodge 对比学习框架 CHCL，通过将 Cheeger 连通性与 1‑Hodge 拉普拉斯低频谱拼接为稳定的图结构签名，对齐增强视图中的编码器输出，从而提升图表示的鲁棒性和泛化。

**💡 创新点**

创新点：①提出 Cheeger–Hodge 结构签名，结合全局连通性与高阶拓扑信息；②将该签名作为对比学习的结构一致性目标，减少对数据增强的依赖；③提供 Lipschitz 稳定性保证。

**🔧 技术方法**

技术：图对比学习、GCN 编码器、Cheeger λ₂ 近似、1‑Hodge 拉普拉斯谱、MLP 投影头、NT‑Xent 损失、对比损失等。

**📊 数据集**

使用数据集：TUgraph 10 个图分类基准、3 个分子回归数据集（molesol、mollipo、molfreesolv）、ChEMBL 预训练、MoleculeNet 8 个分子预测基准以及 OGB 等。

**📈 对比分析**

对比 21 个基线，CHCL 在 9/10 图分类任务中取得最优，分子回归 RMSE 最高，迁移学习 ROC‑AUC 最高，显著优于结构感知方法如 TopoGCL、GCL‑SPAN、CI‑GCL 等。

**⚠️ 局限性**

局限：对大规模图的 Laplacian 谱计算成本较高；对极端噪声或稀疏特征的鲁棒性需进一步验证；超参数如签名维度、损失权重需要手动调优。

---

## 175. Rethinking Mutual Coupling in Movable Antenna MIMO Systems: Modeling and Optimization

**arXiv ID:** 2604.26282 | [PDF](https://arxiv.org/pdf/2604.26282v1)

**作者:** Tianyi Liao `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 46294 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出利用可移动天线（MA）系统中的互耦（MC）效应，开发了基于电路理论的MIMO容量最大化与宽带求和速率最大化的优化框架；

**💡 创新点**

创新点在于将MC视为可利用的资源，利用MC产生的超定向性和可设计的互耦矩阵来提升容量，并通过求解Sylvester方程获得MC矩阵逆平方根的导数，从而实现非凸优化的收敛求解；

**🔧 技术方法**

采用了块坐标上升（BCA）与受限区域方法（TRM）结合的优化算法，利用水分配、Sylvester方程求解及多载波OFDM模型实现宽带系统的求和速率最大化；

**📊 数据集**

实验使用了在28 GHz频段下的Rician通道模型，包含1个LOS分量和若干散射簇（每簇8个子路径），并在M=8、N=8的多天线配置下进行1000次随机路径仿真；

**📈 对比分析**

通过与固定阵列（ULA、CLA）以及不考虑MC的可移动天线（NC-MA）进行对比，结果显示在窄带情况下C‑MA实现约13–18 %的容量提升，在宽带情况下则在187–296 bps/Hz的速率增益，且提升随子载波数和功率密度增加而显著；

**⚠️ 局限性**

局限性包括：仅考虑等向性天线模型，忽略多用户干扰与天线驱动时延；算法复杂度高（需求解Sylvester方程），且在极高天线数或频率选择性极强的环境下收敛速度可能下降。

---

## 176. Agentic AI in the Software Development Lifecycle: Architecture, Empirical Evidence, and the Reshaping of Software Engineering

**arXiv ID:** 2604.26275 | [PDF](https://arxiv.org/pdf/2604.26275v1)

**作者:** Happy Bhati `[一作]` `[通讯]`, Happy Bhati

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了大型语言模型驱动的 Agentic AI 在软件工程生命周期中的演进，提出了六层参考架构和 Agentic SDLC 模型，并系统整理了性能与生产力的实证证据。

**💡 创新点**

创新点在于将传统 SDLC 与 Agentic SDLC 对比，形成了六层架构（模型、推理、ACI、工具、编排、治理），并提出了五大前沿研究议题（评测、治理、技术债、技能再分配、注意力经济）。

**🔧 技术方法**

核心技术包括：大模型（Claude Opus 4.7、GPT‑5.4、Gemini 3.1 Pro）、Chain‑of‑Thought/REACT 推理与自我反思、Agent‑Computer Interface、文件/终端/CI‑CD 工具、单/多代理编排与权限治理。

**📊 数据集**

主要数据集涵盖：SWE‑bench Verified（12 k GitHub issue 任务）、SWE‑bench、Terminal‑Bench、SWE‑Compass、以及公开的 GitHub 代码库和行业使用日志。

**📈 对比分析**

通过基准对比（SWE‑bench 解决率从 1.96% 提升至 78.4%）、实验评测（生产力提升 13.6%–55.8%）以及行业采样（约 49% 工作岗位至少 25% 任务使用 AI），表明 Agentic 系统显著提升了代码交付效率与质量。

**⚠️ 局限性**

局限主要体现在：评测多集中于 Python，缺乏多语言与真实生产任务的覆盖；技术债与长期维护影响尚未系统验证；治理与安全文档缺失；以及技术迁移学习曲线对新手不友好。

---

## 177. Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens

**arXiv ID:** 2604.26355 | [PDF](https://arxiv.org/pdf/2604.26355v1)

**作者:** Zhenyu Zhao `[一作]` (Writer, Inc.), Waseem Alshikh `[通讯]` (Writer, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建模型无关的推理压缩管道，通过在LLM生成的链式思考（CoT）文本上应用BPE合并得到“supertokens”，并通过监督微调让模型学习使用这些超词，从而在保持推理正确率的前提下压缩推理长度。

**💡 创新点**

创新点在于：①提出结构化/有机两类推理token的信息论分解，指出低熵结构词是可压缩的；②基于此设计仅在词表层面做扩展的压缩方法，既可压缩长度又可保留完整可读推理；③将supertokens作为可解释的推理移动标记，揭示推理过程中的高层策略和诊断信号。

**🔧 技术方法**

主要技术包括：跨词BPE（SuperBPE）合并、词表扩展、仅更新嵌入层、LM头和少量Transformer层的监督微调（SFT），以及结构词分类与转移概率分析。

**📊 数据集**

使用的推理数据集为OpenThoughts3（用于生成推理示例）以及五个数学推理基准（AIME'24、AIME'25、MATH-500、Minerva、OlympiadBench）进行评估。

**📈 对比分析**

与基线模型（未压缩）相比，supertoken微调后在三大模型家族（Qwen、Qwen3、DeepSeek-R1-Llama-70B-Distill）上平均压缩8.1%的推理长度，准确率变化均落在95%置信区间内无统计显著差异；在某些基准上压缩幅度可达17%，但仍保持零显著误差。

**⚠️ 局限性**

局限性在于压缩率相对较低（仅约8%），相比内容级或潜在空间压缩方法（可达30–70%）显著不足；且压缩方法依赖于模型特定的推理语料，对词表不匹配的模型（如Llama）可能引入更多扰动。

---

## 178. GateMOT: Q-Gated Attention for Dense Object Tracking

**arXiv ID:** 2604.26353 | [PDF](https://arxiv.org/pdf/2604.26353v1)

**作者:** Mingjin Lv `[一作]` (Princeton University), Zikai Song `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在线多目标跟踪框架GateMOT，利用Q‑Gated Attention在共享高分辨率特征图上并行生成检测、运动估计和重识别（ReID）三任务的特征；

**💡 创新点**

创新点在于将Query重新定义为可学习的门控单元（Gating‑Query），实现对Key特征的逐像素门控选择，从而将传统全局注意力的O(N²)开销降低为O(N)门控与O(Nd²)局部聚合，兼具注意力的选择性与稀疏算子高效性；

**🔧 技术方法**

核心技术包括：Q‑Gated Attention模块（门控查询、局部聚合、未滤波Value残差融合）、并行多头解码器、线性复杂度注意力、基于DLA‑34、ResNet‑50及YOLOX‑X的跨骨干迁移、端到端多任务损失（Focal、交叉熵、SIoU、L1）；

**📊 数据集**

在四大密集跟踪基准上进行评估：BEE24（蜜蜂密集场景）、MOT17、MOT20、SportsMOT；

**📈 对比分析**

与Kalman滤波器、FairMOT、ByteTrack、OC‑SORT、TrackFormer、TraDeS等多种代表性方法进行官方指标对比；在BEE24上取得HOTA48.4、MOTA67.8、IDF164.5，MOT17上HOTA63.3、MOTA78.0、IDF177.9，MOT20上HOTA62.8、MOTA77.6、IDF177.3，SportsMOT上HOTA76.3、MOTA96.5、IDF179.0，均超过或接近当前SOTA，且保持较高帧率（≈13‑15 FPS）和可接受的GFLOPs。

**⚠️ 局限性**

局限性：在稀疏场景下长时间遮挡或剧烈摄像机运动时，单纯的局部门控与聚合难以恢复长期关联，导致身份丢失；缺乏显式的长时记忆或跨帧传递机制；在极高分辨率下仍会出现显著算力消耗。

---

## 179. A Dual-Task Paradigm to Investigate Sentence Comprehension Strategies in Language Models

**arXiv ID:** 2604.26351 | [PDF](https://arxiv.org/pdf/2604.26351v1)

**作者:** Rei Emura `[一作]` (Tohoku University), Saku Sugawara `[通讯]` (National Institute of Informatics)

**通讯引用:** 515 | [OpenAlex ID](https://openalex.org/A5038103607)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个双任务范式，模型同时执行算术运算和句子理解问答，以探究资源受限时语言模型的句子理解策略是否趋向人类的合理推理。

**💡 创新点**

创新点在于通过双任务直接操控模型的工作记忆资源，而非仅通过输入长度或参数调整；并首次比较单任务、嘈杂单任务和双任务三种情境下模型对可实现性与不可实现性句子的准确率差异。

**🔧 技术方法**

使用 GPT‑4o、o3‑mini、o4‑mini 等大型语言模型，结合差异差分统计（Wilcoxon 单侧检验）评估三种任务条件下的可实现性效应；在模型端实现算术解题与理解问答的 Prompt 设计与自动化。

**📊 数据集**

基于 GELP 句子–问题对数据集，加入随机算术表达式生成 2560 条样本（可实现性/不可实现性 × 8 句型 × 160 条），并在实验中引入 1‑digit、3‑digit、5‑digit、10‑digit、30‑digit 加法等多种算术难度。

**📈 对比分析**

通过先筛选满足单任务准确率≥80%且算术准确率≥80%的模型，再对三种任务条件下每条样本的准确率进行差异差分，使用 Wilcoxon 检验判断双任务是否显著放大可实现性差距。结果显示 GPT‑4o 等模型在双任务下可实现性/不可实现性差距显著增大，表现与人类受限任务下的合理推理相似。

**⚠️ 局限性**

局限性包括：不同模型（同属 GPT‑4 系列）表现差异未能归因于架构或训练差异；实验高度依赖 Prompt 设计，可能影响结果；人类数据样本有限，未能进行充分的统计比较；双任务设置主要关注算术计算，其他类型的认知负荷仍需进一步探索。

---

## 180. Federated Medical Image Classification under Class and Domain Imbalance exploiting Synthetic Sample Generation

**arXiv ID:** 2604.26324 | [PDF](https://arxiv.org/pdf/2604.26324v1)

**作者:** Martina Pavan `[一作]` (University of Padova), Pietro Zanuttigh `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出 FedSSG 框架，通过服务器端使用公共数据预训练分类器与扩散式生成模型，并在客户端按设备聚类后利用生成模型按类别与域动态生成样本，解决医学影像联邦学习中的域与类别不平衡问题。

**💡 创新点**

创新点在于：①将分类器与生成模型统一在服务器端预训练，以获得鲁棒初始化；②在客户端根据设备类别划分聚类后，采用类别与域条件的生成模型动态分配合成样本，精准补偿稀有病理与少数域的样本缺失；③把生成增强直接嵌入联邦训练流程，保持数据隐私且显著提升训练稳定性与泛化能力。

**🔧 技术方法**

使用技术包括：EfficientNet‑B0 作为基准分类器；U‑Net 基础的去噪扩散模型配合 FiLM 条件化与类条件化；FedAvg、FedProx 与 MOON 等联邦优化算法做对照；数据增强策略与基于类别/域权重的样本分配；以及轻量化的客户端本地训练与聚合。

**📊 数据集**

实验基于 ISIC 皮肤病变数据集，分为有设备标签的私有（typed）与无设备标签的公开（untyped）两份；覆盖五个病理类别（actinic keratosis、basal cell carcinoma、melanoma、nevus、seborrheic keratosis）。

**📈 对比分析**

与 FedAvg、MOON、FedProx 进行对比，评估在三种设备域（contact polarized、contact non‑polarized、non‑contact polarized）上的准确率与 F1。FedSSG 在公共预训练条件下平均准确率 82.4%、F1 64.7%，比基线提升约 3%（准确率）与 2.6%（F1），尤其在少数域（CNP、NCP）上提升显著。

**⚠️ 局限性**

局限性包括：仅在皮肤病变分类上验证，跨模态或其他任务的通用性待进一步研究；生成模型的质量与多样性受限，可能影响极端稀有病理的合成；对未知设备的泛化能力未充分评估；合成样本比例与生成时间仍需根据具体部署进行调优。

---

## 181. Motion-Driven Multi-Object Tracking of Model Organisms in Space Science Experiments

**arXiv ID:** 2604.26321 | [PDF](https://arxiv.org/pdf/2604.26321v1)

**作者:** Jianing You `[一作]` (Chinese Academy of Sciences), Shengyang Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 2076 | [OpenAlex ID](https://openalex.org/A5111003697)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究微重力实验环境下多动物追踪，提出 SpaceAnimal-MOT 数据集与 ART-Track 框架。

**💡 创新点**

创新点包括：多模型 Unscented Kalman Filter 处理非线性运动，运动状态驱动的分级关联策略，以及基于预测不确定性的自适应空间‑运动融合。

**🔧 技术方法**

采用 AIMM‑UKF 进行运动状态估计，MSDC 关联策略，AUF 自适应融合，YOLOv11 检测框架。

**📊 数据集**

使用 SpaceAnimal-MOT 数据集，其中包含斑马鱼和果蝇在微重力环境下拍摄的视频序列。

**📈 对比分析**

与现有启发式基线对比，ART-Track 在斑马鱼集上 IDF1 达 56.8、IDs 26；在果蝇集上 HOTA 62.2、IDF1 81.6、IDs 85；在 oracle 检测下进一步提升至 IDF1 84.4/93.7，显著降低身份切换并提升长期轨迹可用性。

**⚠️ 局限性**

局限性：仍受低质量图像、强遮挡和形变噪声影响；未覆盖多摄像机场景，极端交互或形态变化时性能可能下降。

---

## 182. Towards Low-Cost Low-Power Activity-Aware Soil Moisture Sensing Platform for Large-scale Farming

**arXiv ID:** 2604.26303 | [PDF](https://arxiv.org/pdf/2604.26303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 183. The Unseen Adversaries: Robust and Generalized Defense Against Adversarial Patches

**arXiv ID:** 2604.26317 | [PDF](https://arxiv.org/pdf/2604.26317v1)

**作者:** Vishesh Kumar `[一作]` (Indian Institute of Science Education and Research Bhopal), Akshay Agarwal `[通讯]` (Indian Institute of Science Education and Research Bhopal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个将对抗性补丁与自然噪声同时混合的基准数据集，并对基于传统机器学习分类器（AdaBoost、SGD、RF等）结合VGG16和ViT特征提取器的对抗补丁检测方法进行了系统评估。

**💡 创新点**

创新点包括：1) 生成并发布包含10种物理补丁和3种自然噪声的统一数据集；2) 证明传统分类器在对抗补丁检测中能抵御噪声的鲁棒性；3) 在未见补丁和噪声的“零样本”设置下，ViT+SGD组合实现了最优性能，优于现有SOTA。

**🔧 技术方法**

技术：使用VGG16和Vision Transformer做特征提取，后接AdaBoost、SGD、RF、LR等传统机器学习分类器；对抗补丁生成基于公开方法（如D-UAP、Squeezed Patch）；实验中加入Gaussian、Shot、Impulse三种噪声；评估指标为准确率、平均鲁棒准确率、攻击成功率、mAP。

**📊 数据集**

数据集：基于ImageNet和COCO两个大规模公开数据集生成，包含4000张干净图像、40000张单补丁图像、4800张加噪声测试图像、48000张同时含补丁与噪声图像；将数据拆分为训练/测试三份比例3:2。

**📈 对比分析**

与SOTA对比：在未见补丁和未见噪声的检测任务中，ViT+SGD平均准确率约为0.84‑0.87，标准差仅3–4%，显著低于MobileNet‑V2（0.83）和NASNet‑Mobile（0.82）的波动；在对象检测任务中，patch+noise攻击导致mAP下降至少30%+，验证了攻击的跨任务泛化能力。

**⚠️ 局限性**

局限性：① 数据集仅涵盖10种补丁与3种噪声，真实场景中可能出现更多样化的扰动；② 评估仅在图像分类与检测两种任务，未覆盖语义分割等其它视觉任务；③ 对抗补丁检测依赖于特征提取网络，若出现更鲁棒的网络，传统分类器性能可能受限。

---

## 184. Alter-Art: Exploring Embodied Artistic Creation through a Robot Avatar

**arXiv ID:** 2604.26473 | [PDF](https://arxiv.org/pdf/2604.26473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 185. VulStyle: A Multi-Modal Pre-Training for Code Stylometry-Augmented Vulnerability Detection

**arXiv ID:** 2604.26313 | [PDF](https://arxiv.org/pdf/2604.26313v1)

**作者:** Chidera Biringa `[一作]` (University of Massachusetts Dartmouth), Gokhan Kul `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多模态预训练模型VulStyle，将函数级源码、非终端AST结构和代码风格特征联合编码用于漏洞检测

**💡 创新点**

创新点在于：①采用仅保留非终端AST节点的稀疏结构，降低输入复杂度；②加入代码风格（CStyle）特征捕捉易引发漏洞的编程习惯；③将三模态信息融合到RoBERTa Transformer中，提升检测精度

**🔧 技术方法**

技术包括：RoBERTa Transformer + Masked Language Modeling预训练；AST解析与节点筛选；代码风格特征提取；多模态融合与序列分类头；AdamW+学习率调度；加权损失处理类别不平衡

**📊 数据集**

预训练数据来自CodeSearchNet、VulBERTa、DiverseVul、Big-Vul，约4.9M函数；微调数据使用Devign、REVEAL、BigVul、DiverseVul、VulDeePecker等公开基准集

**📈 对比分析**

通过与CodeBERT、UniXcoder以及VulBERTa变体在五个基准数据集上的对比，VulStyle在BigVul、VulDeePecker等数据集上实现了显著提升（F1提升4–48%，尤其在BigVul上+48%），在其他数据集也保持了领先或竞争性表现

**⚠️ 局限性**

局限性包括：①仅做函数级分类，无法捕获跨函数的全局数据流与控制流；②对标签噪声敏感；③代码风格特征在模板化或自动生成代码中失效；④在完全语义层面的漏洞（如加密误用、整数溢出）检测仍有限

---

## 186. Benchmarking PyCaret AutoML Against BiLSTM for Fine-Grained Emotion Classification: A Comparative Study on 20-Class Emotion Detection

**arXiv ID:** 2604.26310 | [PDF](https://arxiv.org/pdf/2604.26310v1)

**作者:** Arya Muda Siregar `[一作]` (Institut Teknologi Sumatera), Martin C. T. Manullang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对20类情感细粒度分类任务进行了系统性的基准比较，评估了传统机器学习与深度学习模型的性能与训练效率；

**💡 创新点**

创新点在于首次在同一数据集、同等实验条件下完整对比了TF-IDF+LR/NB/SVM三种经典ML模型与BiLSTM、GRU、轻量级Transformer三种DL模型，并将最佳模型部署为交互式Web应用；

**🔧 技术方法**

使用的技术包括TF-IDF特征提取、Logistic回归、朴素贝叶斯、线性SVM、PyTorch实现的BiLSTM、GRU和Encoder‑only Transformer；

**📊 数据集**

所用数据集为20‑Emotion Text Classification Dataset，共79,595条英文句子，涵盖20种情感标签；

**📈 对比分析**

比较方法采用准确率、宏/加权F1分数、训练时间等指标，实验显示BiLSTM在测试集上达到89%准确率、0.89加权F1，优于SVM（88.11%）及其他DL模型，训练时间仅5m53s；

**⚠️ 局限性**

局限性包括：数据量对Transformer不足，未使用预训练语言模型；仅采用10轮训练；未进行更系统的超参数搜索和交叉验证；对罕见情感类别的提升有限。

---

## 187. NeuroPlastic: A Plasticity-Modulated Optimizer for Biologically Inspired Learning Dynamics

**arXiv ID:** 2604.26297 | [PDF](https://arxiv.org/pdf/2604.26297v1)

**作者:** Douglas Jiang `[一作]` (Harvard Medical School), Feng Tian `[通讯]` (Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种受神经可塑性启发的优化器 NeuroPlastic，通过多信号调制梯度更新。

**💡 创新点**

创新在于将梯度、活动和记忆三种信号组合成可调节的可塑性系数，实现轻量化、可插拔的调制层。

**🔧 技术方法**

采用梯度归一化、指数移动平均、Adam 风格记忆统计以及家居稳定化机制（归一化、RMS 控制）。

**📊 数据集**

在 MNIST、Fashion‑MNIST、CIFAR‑10 等图像分类基准上评估。

**📈 对比分析**

与梯度‑仅基线、SGD、Adam、AdamW 进行对比，NeuroPlastic 在 Fashion‑MNIST 及低数据量下略优，CIFAR‑10 稳定但未明显超越调优标准优化器。

**⚠️ 局限性**

局限在轻量化基准上实验、手工设定超参数、未在大规模模型或数据集上验证。

---

## 188. Attribution-Guided Multimodal Deepfake Detection via Cross-Modal Forensic Fingerprints

**arXiv ID:** 2604.26453 | [PDF](https://arxiv.org/pdf/2604.26453v1)

**作者:** Wasim Ahmad `[一作]` (Beijing Institute of Technology), Xuerui Mao `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 1086 | [OpenAlex ID](https://openalex.org/A5001776125)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于属性引导的多模态深度伪造检测框架 AMDD，联合训练真实/伪造鉴别与生成器归因任务，形成共享的嵌入空间。

**💡 创新点**

创新点在于：① 将生成器归因作为结构化正则化融入检测学习，强制模型学习生成器特有的法医指纹；② 设计跨模态法医指纹一致性 (CMFFC) 损失，显式对齐同一生成器在视觉与音频中的痕迹；③ 平衡视觉与音频编码器容量，提升音频特征的利用率。

**🔧 技术方法**

采用 ResNet50+时间注意力作为视频编码器，ResNet18 处理 Mel 频谱音频，双向跨模态注意力融合，使用交叉熵、InfoNCE、CMFFC、中心正则化等多重损失；训练使用 AdamW、cosine annealing、梯度裁剪等技巧。

**📊 数据集**

主要数据集：FakeAVCeleb（训练与评估），DeepfakeTIMIT、DFDM、LAV-DF 用于跨数据集泛化测试；另外利用 DF-TIMIT 真实视频补充训练集。

**📈 对比分析**

与现有方法对比，AMDD 在 FakeAVCeleb 上实现 AUC 99.8%、平衡准确率 99.7%，归因准确率 95.9%，与 CMALDD‑PTAF 等方法性能相近，但 AMDD 唯一同时提供归因结果；跨数据集评估显示真实视频检测稳定，但对未见生成器的伪造检测泛化有限。

**⚠️ 局限性**

局限性包括：对未知生成器的泛化能力不足、无音频场景下性能下降、部分生成器样本量小导致归因统计不稳、依赖人脸检测预处理等。

---

## 189. CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation

**arXiv ID:** 2604.26427 | [PDF](https://arxiv.org/pdf/2604.26427v1)

**作者:** Yibiao Wei `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112236 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为CARD的生成式推荐框架，将文本、视觉与协同信号统一渲染为视觉语义单元，并通过可学习的非均匀变换实现残差VAE量化，生成高质量语义ID；

**💡 创新点**

创新点包括（1）视觉语义单元统一融合多模态信息，避免传统分离后对齐导致的语义鸿沟；（2）引入可学习可逆非均匀变换（Kumaraswamy或Logistic-Logit），在量化前将非均匀分布映射为近似均匀空间，显著提升码字利用率与生成质量；

**🔧 技术方法**

使用SigLIP2视觉‑语言编码器、NU‑RQ‑VAE残差量化、T5生成模型、可逆非均匀变换、beam search解码；

**📊 数据集**

在Amazon Food、Phones、Clothing三个热门电商数据集上进行实验；

**📈 对比分析**

与传统序列模型（GRU4Rec、BERT4Rec、SASRec）及生成式基线（VQ‑Rec、TIGER、LETTER、MQL4GRec、MACRec）对比，CARD在Recall@K与NDCG@K上持续领先，平均提升约10%‑15%；

**⚠️ 局限性**

局限性：视觉语义单元的布局和内容依赖人工设计，缺乏自动化构造方法，且在极端稀疏或非图像化场景下效果待验证；

---

## 190. STLGT: A Scalable Trace-Based Linear Graph Transformer for Tail Latency Prediction in Microservices

**arXiv ID:** 2604.26422 | [PDF](https://arxiv.org/pdf/2604.26422v1)

**作者:** Yongliang Ding `[一作]` (East China Normal University), Peng Pu `[通讯]` (East China Normal University)

**通讯引用:** 6562 | [OpenAlex ID](https://openalex.org/A5100404009)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于分布式跟踪的可扩展线性图Transformer（STLGT），用于微服务系统的多步p95尾延迟预测。

**💡 创新点**

创新点在于：① 对每个API构建 span 图作为轻量化、可扩展的图抽象，避免全局图规模爆炸；② 采用结构感知的线性图Transformer实现全局依赖传播，计算复杂度线性于图边数；③ 将空间编码与时间建模解耦，使用轻量化时序解码器捕捉非平稳、突发工作负载；④ 通过 trace-aware 读出层融合请求级上下文，提高预测精度。

**🔧 技术方法**

技术主要包括：分布式跟踪收集与 span 图构造、线性图Transformer（结构化全局混合 + 局部GCN传播）、Trace-aware 读出层、TimesNet 时序解码器、标准化的多步p95尾延迟预测框架。

**📊 数据集**

使用的数据集包括：DeathStarBench（Hotel Reservation 与 Social Network）、Alibaba 生产跟踪数据（多子集大小 1-10）、以及内部个性化教育平台的真实流量。

**📈 对比分析**

与基线（GBDT、PERT‑GNN、FastPERT）在四个数据集上比较，STLGT 在 MAPE 上平均提升约 8.5%，同时在 CPU 推理时间上比 PERT‑GNN 下降 49.7%（N=32 情况），在 GPU 上 1.5 倍增长，说明在大图规模下更具可扩展性。

**⚠️ 局限性**

局限性包括：需要对每个 API 训练单独模型，若调用路径频繁漂移需重新构图与微调；在极端大规模图或高频实时推理时，线性 Transformer 的常数仍不可忽略；对持续变化的拓扑结构的在线适配尚未实现。

---

## 191. Asset Administration Shell-Based OCL Validation Framework for Model-Based System Engineering

**arXiv ID:** 2604.26384 | [PDF](https://arxiv.org/pdf/2604.26384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 192. Towards Intelligent Computation Offloading in Dynamic Vehicular Networks: A Scalable Multilayer Pipeline

**arXiv ID:** 2604.26416 | [PDF](https://arxiv.org/pdf/2604.26416v1)

**作者:** Falk Dettinger `[一作]` (University of Stuttgart), Michael Weyrich `[通讯]` (University of Stuttgart)

**通讯引用:** 5278 | [OpenAlex ID](https://openalex.org/A5072168528)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个四层计算机卸载流水线，用于软件定义车辆，在云与边缘资源间动态分配任务并满足严格的 RTT 约束；

**💡 创新点**

创新点包括：① 将方向和距离惩罚与功能需求集成到 PSO 算法中以优化服务器选择；② 设计了包含提取、决策、执行、检测四层的闭环流水线；③ 在真实 Kubernetes 集群上实现并验证该框架。

**🔧 技术方法**

采用的技术有：粒子群优化（PSO）、CNN‑LSTM 预测模型、机器学习回归、Kubernetes 容器化与 CI/CD、边缘与云服务器协同、反馈检测层。

**📊 数据集**

使用的数据为：基于真实云执行的 RTT 经验分布、模拟车辆轨迹、目标识别与情感识别两种任务的输入数据；未使用公开数据集，而是通过仿真与实际云环境收集。

**📈 对比分析**

对比方法：将改进的 PSO 与暴力搜索（BF）和贪心基线进行比较。结果显示 PSO 平均偏差仅 0.7%，CPU 26 ms、GPU 62 ms；在 1,000 任务/15 服务器时 CPU 2.5 s、GPU 550 ms，显示出良好的规模扩展性和与 BF 的接近最优性能。

**⚠️ 局限性**

局限性：① 仍缺乏完整的预测与检测层在真实动态环境中的验证；② 服务器过载风险未得到充分缓解；③ 决策时延与 Kubernetes 失效检测延迟较大；④ 仅在仿真与有限实验条件下评估，未覆盖更复杂网络与更高车辆密度的场景。

---

## 193. Existence and Constructions of Strict Function-Correcting Codes with Data Protection

**arXiv ID:** 2604.26397 | [PDF](https://arxiv.org/pdf/2604.26397v1)

**作者:** Charul Rajput `[一作]` (International Institute of Information Technology Hyderabad), Camilla Hollanti `[通讯]` (Aalto University)

**通讯引用:** 1886 | [OpenAlex ID](https://openalex.org/A5035260653)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了具有数据保护的严格函数纠错码的存在性和构造，提出了三项主要贡献。

**💡 创新点**

创新点在于通过α-距离图框架建立了严格函数纠错码的存在条件，并提出了链码作为满足这些条件的无限家族。

**🔧 技术方法**

使用了图论方法，特别是α-距离图和Cayley图的结构，以及Simonis定理的逆构造。

**📊 数据集**

使用了链码和窄感BCH码作为数据集，特别是针对具有设计距离三的BCH码进行了构造。

**📈 对比分析**

通过与现有方法的比较，证明了所构造的严格函数纠错码在数据保护和函数保护的距离上具有优势，且在特定条件下能够实现更低的冗余。

**⚠️ 局限性**

限制在于所构造的码在某些情况下可能无法满足所有函数的保护需求，尤其是在函数值的数量和预像大小的限制下。

---

## 194. The Buy-or-Build Decision, Revisited: How Agentic AI Changes the Economics of Enterprise Software

**arXiv ID:** 2604.26482 | [PDF](https://arxiv.org/pdf/2604.26482v1)

**作者:** David Klotz `[一作]` (Media University), David Klotz `[通讯]` (Media University)

**通讯引用:** 244 | [OpenAlex ID](https://openalex.org/A5004715756)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文以交易成本经济学和资源基础观为理论框架，系统评估生成式AI（尤其是Agentic coding系统）对企业应用Make‑or‑Buy决策的影响，并提出AI时代的治理与决策框架；

**💡 创新点**

创新点包括：①将Agentic AI视为Make治理的混合模式；②在七大MoB决策因素上按AI效应重新构建评估维度；③构建四类应用的敏感性类型，指导按业务特征选择Make或Buy；

**🔧 技术方法**

采用概念性推导与文献综述，结合最新Agentic AI技术特征（如SWE‑agent、OpenHands等）进行分析；

**📊 数据集**

未使用专门实验数据集，而是基于公开AI系统案例、行业报告和现有研究综述做为信息来源；

**📈 对比分析**

对比方法为理论评估与案例归纳，表明在低复杂度、低合规性的应用中Make优势明显，而在高复杂度或高合规性应用中Buy仍占主导；

**⚠️ 局限性**

局限性在于缺乏实证验证、AI技术快速演进导致模型易失效、未细化买方子选项（SaaS、私有云、开源）以及未考虑不同监管环境下的差异。

---

## 195. Templates in Rewriting Induction

**arXiv ID:** 2604.26474 | [PDF](https://arxiv.org/pdf/2604.26474v1)

**作者:** Kasper Hagens `[一作]` (Radboud University), Cynthia Kop `[通讯]` (Radboud University)

**通讯引用:** 242 | [OpenAlex ID](https://openalex.org/A5001880661)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

提出了一种基于模板的引理生成方法，并将其整合进 Bounded Rewriting Induction (Bounded RI)，用于在高阶逻辑约束项重写系统中自动化地证明程序等价。

**💡 创新点**

创新点在于：①将典型程序构造（上/下递归、尾递归）抽象为模板并用高阶递归器描述；②利用模板-递归器等价与递归器等价，将原本需要手工发现的非多项式不变量转化为高阶结构的匹配，从而在 Bounded RI 中实现更高层次的引理生成；③在理论上证明该方法可以覆盖传统约束式引理生成方法难以处理的实例。

**🔧 技术方法**

主要技术手段包括高阶逻辑约束项重写、Bounded RI 证明框架、模板匹配、递归器与条件递归器的等价证明、上下文函数、良序与归纳推理，以及对约束的符号执行与求解。

**📊 数据集**

实验示例采用阶乘函数的四个实现（tail-up、tail-down、recursive-up、recursive-down）以及一个更复杂的求积实现；未使用外部数据集，而是通过这些示例进行理论证明。

**📈 对比分析**

与现有基于约束式的引理生成技术相比，模板方法在需要非多项式不变量的例子（如阶乘的上/下递归实现）中能够直接给出等价证明，避免了繁琐的不变量发现；虽然本文未给出定量实验结果，但理论证明表明在模板匹配层面可显著降低证明难度，并且能够在 Bounded RI 的框架内实现自动化。

**⚠️ 局限性**

局限性包括：①仅支持已定义的有限模板，无法覆盖所有程序结构；②需要预先给出良序、终止性等假设；③对更复杂或不符合模板的程序，仍需手工构造引理或扩展模板；④在实践中自动化实现尚未完全完成，仍需进一步研究匹配与引理选择的策略。

---

## 196. Accelerating Sparse Linear Solvers with an Optical Laser Processing Unit

**arXiv ID:** 2604.26377 | [PDF](https://arxiv.org/pdf/2604.26377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 197. Near-Optimal Cryptographic Hardness of Learning With Homogeneous Halfspaces Under Gaussian Marginals

**arXiv ID:** 2604.26446 | [PDF](https://arxiv.org/pdf/2604.26446v1)

**作者:** Jizhou Huang `[一作]` (Washington University in St. Louis), Brendan Juba `[通讯]` (Washington University in St. Louis)

**通讯引用:** 864 | [OpenAlex ID](https://openalex.org/A5064954992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文研究了在高斯分布下识别同质半空间的三个问题：无噪声学习、单侧可靠学习和公平审计。

**💡 创新点**

创新点在于将连续LWE问题映射到这些学习任务，给出了对同质半空间的最优近似硬性下界，显著拓展了先前针对一般半空间的硬性结果。

**🔧 技术方法**

主要技术是基于连续LWE的降维与傅里叶分析，将标签映射成噪声平滑的方波，进而证明同质半空间在对立假设下拥有显著优势。

**📊 数据集**

由于研究为理论计算复杂性分析，未使用任何真实数据集；所有证明均在理想的高斯分布上进行。

**📈 对比分析**

通过与LWE假设的等价性证明，本文给出对学习与审计算法的近似不可解性，表现为在多项式时间内无法取得比1/√(η log d)更好的优势，展示了严谨的下界分析。

**⚠️ 局限性**

局限性包括：仅处理同质半空间且仅在标准高斯边缘下；结果依赖于LWE的子指数难度假设；未给出可实现的算法或对更一般半空间的上界。

---

## 198. Meta-Learning and Targeted Differential Privacy to Improve the Accuracy-Privacy Trade-off in Recommendations

**arXiv ID:** 2604.26390 | [PDF](https://arxiv.org/pdf/2604.26390v1)

**作者:** Peter Müllner `[一作]` (Know Center Research GmbH), Elisabeth Lex `[通讯]` (Graz University of Technology)

**通讯引用:** 1660 | [OpenAlex ID](https://openalex.org/A5045400619)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在推荐系统中结合了元学习和针对性差分隐私，提出一种两阶段方案，既在数据层对最典型的用户数据进行差分隐私保护，又在模型层利用元学习提高对残余噪声的鲁棒性。

**💡 创新点**

创新点在于：①使用“目标式DP”只保护最可能泄露敏感属性的数据，减少不必要的噪声；②将元学习嵌入矩阵分解模型（MetaMF），显著降低差分隐私带来的精度损失；③通过调节数据预算β在精度、隐私和正式DP保障之间实现可调权衡。

**🔧 技术方法**

技术手段包括：目标式差分隐私机制（coin‑flip噪声注入）、MetaMF矩阵分解模型及其元网络、无元学习版本NoMetaMF、以及基准随机DP和全DP版本。

**📊 数据集**

使用的数据集为 MovieLens 1M（用户 6,040，项目 3,706）和 Bookcrossing（用户 3,971，项目 5,662），分别在性别和年龄两类敏感属性上进行实验。

**📈 对比分析**

评估方法：在不同ε（3、2、1、0.1）和β（1到0）组合下，采用MAE衡量推荐准确性，BAcc衡量攻击者对敏感属性的推断成功率。实验显示：MetaMF在所有DP设置下的MAE均低于NoMetaMF；目标式DP在保持相近MAE的同时比随机DP和全DP大幅降低BAcc，β≈0.3为最优折中点。

**⚠️ 局限性**

局限性：实验仅使用了矩阵分解模型，未验证对其他推荐模型的通用性；隐私评估依赖于特定的攻击模型，可能在更强攻击下效果不同；目标式DP的预处理需要访问敏感属性，可能带来额外隐私风险。

---

## 199. CoQuant: Joint Weight-Activation Subspace Projection for Mixed-Precision LLMs

**arXiv ID:** 2604.26378 | [PDF](https://arxiv.org/pdf/2604.26378v1)

**作者:** Zhe Ding `[一作]` (Nanjing University of Posts and Telecommunications), Duowei Pan `[通讯]` (Amazon AGI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种联合权重-激活子空间投影的混合精度量化方法 CoQuant，能够在 LLM 量化时同时考虑权重和激活的量化误差。

**💡 创新点**

创新点在于用加权协方差矩阵（包含权重和激活协方差）构造联合子空间，并通过闭式加权 PCA 选取高精度子空间，突破了仅基于激活统计的子空间方法。

**🔧 技术方法**

使用的技术包括：后训练量化 (PTQ)、正交旋转和等距投影、加权协方差矩阵求解、闭式加权 PCA、混合精度量化（8‑bit 高精度子空间 + 4‑bit 低精度子空间）以及 KV 缓存的联合量化。

**📊 数据集**

实验数据集包括：Llama‑3.2（1B、3B）和 Qwen2.5（0.5B、1.5B、7B、14B），评估指标为 WikiText perplexity 与六个零样本常识推理基准（ARC‑Challenge/Easy、BoolQ、PIQA、SIQA、WinoGrande 等）。

**📈 对比分析**

与 RTN、GPTQ、QuaRot、QUIK、ResQ 等基线比较，CoQuant 在 WikiText perplexity 和零样本推理准确率上均显著优于对手（尤其在 Qwen2.5 上提升数个百分点），并在极低位宽（3‑bit/6‑bit）条件下保持更稳健的性能。

**⚠️ 局限性**

局限性包括：依赖一阶误差近似和等向量噪声假设，固定子空间比例（未自适应不同层），缺乏硬件加速实现，评估主要聚焦于精度而非推理速度，未覆盖更大模型或长上下文任务。

---

## 200. Pseudo-Complex Quantifier Elimination

**arXiv ID:** 2604.26400 | [PDF](https://arxiv.org/pdf/2604.26400v1)

**作者:** Nicolas Faroß `[一作]` (Chalmers University of Technology and University of Gothenburg), Thomas Sturm `[通讯]` (CNRS, Inria and University of Lorraine)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了在扩展有序环语言中，对复数的伪复数量化消除框架，并通过将复数问题映射到实数量化消除实现了完整可判定性。

**💡 创新点**

其创新点在于显式加入虚数单位、实部、虚部与共轭符号，提出图形启发式最小化与正规化技术，使复数量化消除与实数工具兼容并获得正式的完整性与可判定性证明。

**🔧 技术方法**

技术实现结合了实数量化消除（圆柱代数分解、虚代换、综合格罗巴基、正则链分解等）、正规化到实数域、图匹配最小化算法以及正则链/树形重写。

**📊 数据集**

使用了Python开源系统 Logic1 进行实验，测试集为一系列学术示例（如坐标表示、单位圆、量子自共轭矩阵、RC 滤波器稳定性等），未涉及公开大型数据集。

**📈 对比分析**

与现有实数/复数工具（Qepcad、Redlog、Maple 等）对比，实验显示在多数示例中求解时间从几百毫秒到几十秒，证明该方法在实现简洁且与经典复数量化消除的复杂度相当。

**⚠️ 局限性**

局限性主要在于高度依赖底层实数量化消除实现，处理高量化层数或大量变量时仍面临指数/双指数爆炸；缺乏专门针对复数的优化实现导致在极大规模实例上的性能受限。

---

## 201. A Multistage Extraction Pipeline for Long Scanned Financial Documents: An Empirical Study in Industrial KYC Workflows

**arXiv ID:** 2604.26462 | [PDF](https://arxiv.org/pdf/2604.26462v1)

**作者:** Yuxuan Han `[一作]` (OCBC), Jingyuan Zhao `[通讯]` (OCBC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种多阶段流程，先做图像预处理和多语言 OCR，再通过页面级检索过滤后，用紧凑的视觉‑语言模型完成结构化信息抽取。

**💡 创新点**

创新点在于将页面定位与多模态推理解耦，利用检索降噪并节省计算；同时在多语言、长篇财报场景下构建了专门的提示和检索策略。

**🔧 技术方法**

技术包括 OpenCV 图像预处理、PaddleOCR/EasyOCR OCR、BM25+句向量混合检索、MiniCPM‑o‑2.6/ Gemma‑3‑27b‑it/ Qwen3‑VL‑8B‑Instruct 视觉‑语言模型，以及人机协作验证。

**📊 数据集**

使用 120 份真实 KYC 财务文件（约 3000 页），涵盖英文、印尼语、简/繁中文。

**📈 对比分析**

通过对比直接 PDF→VLM 基线与多阶段管线，在 5 种配置下评测字段级准确率，最佳配置达 87.27%（提升 31.9%），其他组合亦高于基线。

**⚠️ 局限性**

局限包括需手工制定检索词/提示、对 OCR 质量依赖大、术语多样性和多语言货币单位歧义导致错误、以及对混合印刷/手写内容的鲁棒性不足。

---

## 202. Theory-Grounded Evaluation Exposes the Authorship Gap in LLM Personalization

**arXiv ID:** 2604.26460 | [PDF](https://arxiv.org/pdf/2604.26460v1)

**作者:** Yash Ganpat Sawant `[一作]` `[通讯]` (Independent AI Researcher), Yash Ganpat Sawant (Independent AI Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对大语言模型在个性化写作时的作者风格一致性问题，提出了基于作者身份验证理论的评估框架，并与传统的LLM-评审和函数词统计方法进行了对比。

**💡 创新点**

创新点在于将已验证的作者身份验证模型LUAR作为量化基准，提供可校准的上限与下限，使得评估结果具有绝对意义，并揭示不同评估指标之间的根本性不一致性。

**🔧 技术方法**

主要技术包括基于对比学习的LUAR作者嵌入、使用GLM-4 32B进行解码的LLM-as-Judge协议，以及传统的函数词余弦相似度统计。

**📊 数据集**

实验数据来自Blog Authorship Corpus，筛选了50位作者（每位至少200篇训练帖子、50篇测试帖子），共计约104K训练帖和26K测试帖，并采用LLM提取的内容摘要作为写作提示。

**📈 对比分析**

在四种个性化方法（无个性化、少量样本、档案提取、对比式）下进行1,000次生成实验，LUAR得分在0.484–0.508之间，均低于跨作者人类下限0.626，表明生成文本仍处于模型自身风格簇内；而LLM-评审和函数词指标表现出高度不相关且无法区分方法。

**⚠️ 局限性**

局限性包括仅测试推理时的个性化方法、仅使用英文博客文本、缺乏人类评估以验证作者风格一致性的感知度，以及跨模型评估仅覆盖两大模型家族，未来需扩展至训练时微调、更多语言与写作风格。

---

## 203. Path-Reporting Distance Oracles for Vertex-Labeled Graphs

**arXiv ID:** 2604.26451 | [PDF](https://arxiv.org/pdf/2604.26451v1)

**作者:** Ofer Neiman `[一作]` (Ben Gurion University of Negev), Alon Spector `[通讯]` (Ben Gurion University of Negev)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了能够报告路径的顶点标签距离预言机，并在拉伸上实现了最优 2k-1

**💡 创新点**

创新点在于结合双侧测试、级联随机采样与配对距离预言机，既实现路径报告又将拉伸从 4k-5 降至 2k-1

**🔧 技术方法**

采用层次采样的 pivot 与 bunch/cluster 结构、hash 表存储、以及 pairwise 距离预言机（带路径报告）

**📊 数据集**

论文主要以理论分析为主，未给出具体实验数据集

**📈 对比分析**

与以往 4k-5 拉伸的非路径报告预言机相比，本工作保持 O(k) 查询时间，最优拉伸版本的查询时间为 O(ℓ^{1/k}·log n)，空间为 O(k n ℓ^{1/k})（或 O(n^{1+o(1)} ℓ^{1/k})），性能上在拉伸与路径报告之间取得平衡

**⚠️ 局限性**

限制包括：需要在最后层使用 pairwise 预言机导致额外空间与常数因子；查询时间在最优拉伸版本仍有 O(ℓ^{1/k} log n) 的负担；缺乏实验验证；对非常大标签数 ℓ 的场景可能仍存在空间与时间瓶颈

---

## 204. Layer-wise Lipschitz-Product Control for Deep Kolmogorov--Arnold Network Representations of Compositionally Structured Functions

**arXiv ID:** 2604.26444 | [PDF](https://arxiv.org/pdf/2604.26444v1)

**作者:** Aleksander Tankman `[一作]` `[通讯]` (Fivestar Europe), Aleksander Tankman (Fivestar Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

证明了对任何由有限计算树描述的连续函数，存在深度Kolmogorov–Arnold网络（KAN）表示，其层级逐层Lipschitz乘积、宽度与输入维度无关、逼近误差达到B‑splines的最优速率。

**💡 创新点**

首次给出层级Lipschitz乘积的结构性上界（独立于输入维度），并构造可证明满足该上界的原子KAN块，从而填补了Liu等人在原始KAN工作中未完成的Lipschitz控制问题。

**🔧 技术方法**

采用组合稀疏性假设、量化范围递归、B‑splines逼近、原子KAN块的层级拼接以及层级Lipschitz乘积分析等理论工具进行证明，并配以实验验证。

**📊 数据集**

实验仅使用合成函数（如 xy、xyz、sin(xy)、x₁⋯xₙ）进行，未涉及真实数据集。

**📈 对比分析**

与经典Sprecher构造对比：Sprecher的内核Lipschitz常数随维度上升，而本文构造的KAN实现 P=1，误差与B‑splines的理论收敛速率一致，实验结果验证了理论上界。

**⚠️ 局限性**

局限性：依赖可验证的组合稀疏性假设；对每种算子必须显式构造满足条件的原子KAN块；宽度上界可能不是最优；尚未给出P的下界及对更复杂算子（如exp）的完整证明。

---

## 205. A Matrix-Free Galerkin Multigrid Solver and Failure-Mode Screen for Single-GPU 3D SIMP Linear Systems

**arXiv ID:** 2604.26441 | [PDF](https://arxiv.org/pdf/2604.26441v1)

**作者:** Shaoliang Yang `[一作]` (Santa Clara University), Yunsheng Wang `[通讯]` (Santa Clara University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在单个消费级GPU上实现了一个完全基于矩阵无关的Galerkin几何多重网格（GMG）求解器，用来解决三维SIMP拓扑优化中的线性弹性系统，并通过混合精度（BF16/FP32/FP64）调度与谱代理实现了对BF16细网格平滑器的安全使用，提供了失败模式筛查；同时展示了1M单元规模的线性求解器实现；

**💡 创新点**

①构造了一个只在最细层保持矩阵无关、在下一层使用Galerkin聚合、后续层使用稀疏三重积的多重网格层次；②引入BF16/FP32/FP64精度分层策略，在细层采用BF16、粗层FP32、深层FP64，并用κ_eff谱代理评估BF16可行性；③加入Chebyshev-Jacobi光滑器、V-cycle/ W-cycle对比与敏感性分析；④提供了显式的失败模式屏蔽与诊断框架。

**🔧 技术方法**

矩阵无关的聚合–GEMM–散射核（Level‑0操作），Galerkin多重网格层次，Chebyshev‑Jacobi光滑器，混合精度调度（BF16/FP32/FP64），FGMRES/PCG外部求解器，谱代理κ_eff，CUDA Tensor‑Core（WMMA）实现，GPU内存与带宽分析，RTX 4090硬件环境。

**📊 数据集**

使用多组固定种子二元对比密度场（V_f=0.2,0.5,0.8；p=1.5,3,4.5）进行27‑案例悬臂梁扫描；统一密度（ρ=0.5, p=3）在64k–512k单元；30步固定惩罚OC调度；125k–1M单元（120×60×30等网格）统一模量E_e=0.5或1；以及合成低阈值ρ_floor,test=10^-2等。

**📈 对比分析**

与传统平面Jacobi-PCG基线（200次迭代上限）比较，FP32-GMG在64k、216k、512k下的迭代次数分别从≈112→18，壁时间比为1.62×、1.75×、3.12×；BF16-GMG与FP32-GMG相近且在高对比时能在BF16下保持收敛；对64k下的PyAMG基线进行后装配比较，GPU GMG的构造+求解时间约10倍快；1M单元的统一模量求解耗时1.50±0.58 s，层次构造内存增量8.66 GiB。

**⚠️ 局限性**

仅支持规则Q1六面体网格与2:1同构细化；单GPU，未实现多GPU分布式；对高对比异质场在512k以上的鲁棒性不足；BF16虽可用但因带宽瓶颈未实现显著加速；谱代理κ_eff不完全能预测失败；仅针对线性弹性SIMP，未覆盖非线性/多材料情况。

---

## 206. QYOLO: Lightweight Object Detection via Quantum Inspired Shared Channel Mixing

**arXiv ID:** 2604.26435 | [PDF](https://arxiv.org/pdf/2604.26435v1)

**作者:** Garvit Kumar Mittal `[一作]` (Bharat Electronics Limited), Sandeep Kumar `[通讯]` (Bharat Electronics Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了QYOLO，通过将YOLOv8深层C2f模块替换为共享的量子启发式通道混合块，实现了模型压缩与轻量化。

**💡 创新点**

创新点在于：①在最深两层通道混合中使用共享的正弦混合器，减少参数并保持通道重要性一致；②利用量子启发式的正弦变换实现全局通道再校准；③直接进行架构压缩，避免稀疏剪枝带来的不兼容。

**🔧 技术方法**

采用的技术包括：共享的正弦混合模块（QMixBlock）、全局平均池化+线性投影、正弦频率权重与相位偏移、以及标准YOLOv8训练框架。

**📊 数据集**

实验数据集为VisDrone2019-DET的十类目标，使用其官方训练/验证/测试划分。

**📈 对比分析**

与原YOLOv8n/YOLOv8s以及无结构剪枝、知识蒸馏等方法比较，QYOLOv8n在保持约0.4 mAP@50的同时将参数减少20.2%（3.01M→2.40M）、GFLOPs下降12.3%，KD可进一步恢复完整准确率；相比剪枝仅稀疏无参数减少。

**⚠️ 局限性**

局限性包括：对极小目标和需要精细空间边界的类别（如自行车、行人）仍存在轻微准确率下降；对更深层或其他网络架构的迁移性尚未验证；以及在极端小尺寸模型中训练稳定性需进一步提升。

---

## 207. Delineating Knowledge Boundaries for Honest Large Vision-Language Models

**arXiv ID:** 2604.26419 | [PDF](https://arxiv.org/pdf/2604.26419v1)

**作者:** Junru Song `[一作]` (Shandong University), Yuntao Du `[通讯]` (Shandong University)

**通讯引用:** 528 | [OpenAlex ID](https://openalex.org/A5101920357)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Visual‑Idk框架，帮助大视觉语言模型识别并诚实拒绝其知识盲区的查询

**💡 创新点**

通过构建模型专属的“Visual‑Idk”数据集并结合偏好学习（如ORPO）实现对知识边界的精准刻画，既抑制了幻觉，又降低了“对齐税”

**🔧 技术方法**

采用多样化的技术：视觉-知识多样本一致性探测、监督微调、直接偏好优化（DPO）、对数比优化（ORPO）等

**📊 数据集**

主要使用InfoSeek构建的Visual‑Idk数据集，同时在ScienceQA、VizWiz-Unans和PMC‑VQA等OOV数据集上进行评测

**📈 对比分析**

与基线、SFT和DPO等方法对比，ORPO在Visual‑Idk上将真确率提升至67.3%，在跨域测试和OOV场景亦保持高拒绝率，显示出更优的鲁棒性

**⚠️ 局限性**

局限在于数据集规模有限，需进一步扩充样本以提升模型对更广泛知识盲区的泛化能力

---

## 208. EmoTransCap: Dataset and Pipeline for Emotion Transition-Aware Speech Captioning in Discourses

**arXiv ID:** 2604.26417 | [PDF](https://arxiv.org/pdf/2604.26417v1)

**作者:** Shuhao Xu `[一作]` (Inner Mongolian University), Rui Liu `[通讯]` (Inner Mongolian University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 EmoTransCap，旨在捕捉并描述对话级语音中的情绪转变，并提供双语情绪转变数据集 EmoTransSpeech 与完整的自动化标注与生成流水线。

**💡 创新点**

创新点包括：①首次在对话层面实现情绪转变的自由文本描述；②构建规模化、双语且具情绪转变标签的大型语料库；③设计多任务情绪转变识别模型 MTETR，实现情绪切换检测与分割；④通过 LLM（Gemma‑3）自动生成描述性与指令式两种版本的字幕；⑤将该数据集用于情绪感知与可控情绪合成任务，显著提升模型性能。

**🔧 技术方法**

技术手段涵盖：Gemma‑3 生成文本与字幕、CosyVoice2 语音合成、Emotion‑2vec + ResNet‑Transformer‑BiLSTM 的 MTETR、Whisper‑large‑v2 ASR、WebRTC‑VAD 静音去除、SSML 语音合成标记、自动化质量评估与人工打分。

**📊 数据集**

使用数据集：EmoTransSpeech‑Audio（约 617 小时、144,000 句、10 位英语与 10 位中文原声演说者）及其对应的 EmoTransSpeech‑Caption（描述性与指令式两种版本），并以此对 SECap、SpeechCraft、CosyVoice2 等基线进行微调与对比。

**📈 对比分析**

比较方法：在情绪感知方面与 SECap、SpeechCraft 进行基准对照，使用 Acc_ETC/Acc_ETT/MOS‑C 等主观指标；在情绪表达方面对比未微调与微调 CosyVoice2 在 MOS‑E/MOS‑S 及 EES_ET 目标上的表现。实验显示 EmoTransCap 在所有指标上均优于现有基线，尤其在情绪转变捕获准确率与自然度上提升显著。

**⚠️ 局限性**

局限性：数据完全基于合成语音与自动生成文本，缺乏自然真实对话的复杂性；评估方法主要依赖人工主观打分与有限的自动化度量，需进一步开发更稳健的自动评测与多模态验证。

---

## 209. SecMate: Multi-Agent Adaptive Cybersecurity Troubleshooting with Tri-Context Personalization

**arXiv ID:** 2604.26394 | [PDF](https://arxiv.org/pdf/2604.26394v1)

**作者:** Yair Meidan `[一作]` (Ben Gurion University of Negev), Asaf Shabtai `[通讯]` (Ben Gurion University of Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并评估了一个多智能体虚拟客服系统，该系统通过设备、用户和服务三重上下文实现个性化网络安全故障排查。

**💡 创新点**

将本地设备诊断、隐式用户专业度推断和基于上下文的产品推荐三者集成于统一的代理协同架构，首次实现实时设备根因分析与自适应指导。

**🔧 技术方法**

采用GPT‑4o LLM、LangChain/Graph 进行代理编排、Clue Collector 工具收集系统信息、ProfiLLM 进行用户专业度建模、ImpReSS 推荐算法并生成推理说明。

**📊 数据集**

在实验中收集了 711 条真实对话、144 名工程学生的设备日志、用户画像和推荐产品等信息，构成 DS_Complete、DS_Relabeled 等子集。

**📈 对比分析**

通过与 LLM‑only 基线以及四种组合配置的重复测评，结果显示设备证据+专业度+推荐可将正确率从约 50% 提升至 90.9%，MRR@1≈0.75，成本约 1.5 美元/次，低于人工服务水平。

**⚠️ 局限性**

样本以工程学生为主，缺乏多样化用户；未与现有工业系统对标；LLM 计算成本和时延仍高，对极端专业度推断不够鲁棒。

---

## 210. A Multimodal Pre-trained Network for Integrated EEG-Video Seizure Detection

**arXiv ID:** 2604.26379 | [PDF](https://arxiv.org/pdf/2604.26379v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 211. Benchmarking Complex Multimodal Document Processing Pipelines: A Unified Evaluation Framework for Enterprise AI

**arXiv ID:** 2604.26382 | [PDF](https://arxiv.org/pdf/2604.26382v1)

**作者:** Saurabh K. Singh `[一作]` (Oracle), Sachin Raj `[通讯]` (Independent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了企业文档检索生成（RAG）的统一四轴评估框架 EnterpriseDocBench，并构建了涵盖多行业的公开许可文档基准；通过在同一语料上比较 BM25、密集检索和混合检索三种流水线，探究解析、索引、检索与生成之间的交互与弱相关性；同时分析了文档长度与幻觉率的非单调关系与答案完整性不足问题。

**💡 创新点**

创新点在于：①将解析、索引、检索和生成四个传统独立阶段整合为单一评估框架；②在统一语料上进行跨阶段相关性实验，挑战传统线性级联假设；③发现幻觉率随上下文长度呈 U 型分布；④揭示答案完整性低是部署瓶颈；⑤提供可复现的代码、脚本和基准数据。

**🔧 技术方法**

技术包括：大规模 LLM 生成（GPT‑5）、BM25 与 E5‑large 语义向量检索、混合检索（BM25+密集）、自动化幻觉检测（HalluLens）、文本与表格解析评估（TIS、TEA、FCQ、LF）、基于 BERTScore 的质量评估以及多维度指标聚合。

**📊 数据集**

数据集来源为公开、宽松许可的文档（SEC EDGAR、CUAD、PubMed、ArXiv、USPTO 等），共 1,459 篇，包含 5 个行业领域（财经、法律、医疗、技术、通用），其中 1,169 篇用于测试；采用半自动化 QA 对生成结果进行标注，交叉验证确保注释质量。

**📈 对比分析**

比较方法：在相同生成器（GPT‑5）下，分别使用 BM25、密集检索和混合检索三种索引策略；评估指标为 nDCG@5、P@3、Factual Accuracy、Hallucination Rate、Answer Completeness 等；实验结果显示混合检索与 BM25 在 nDCG@5（0.92 vs 0.91）和整体质量（0.84 vs 0.84）上略优，密集检索落后（0.83，质量 0.80），BM25 在成本-质量 Pareto 前沿。

**⚠️ 局限性**

局限性包括：①域分布不均衡，通用类占比 47%；②仅使用英文文档，缺乏多语言验证；③解析阶段采用预提取文本，未测试不同 OCR/解析工具导致的差异；④幻觉评估依赖自动化指标，主观性存在；⑤仅针对三种流水线进行实验，未测量参考架构；⑥答案完整性低，需进一步改进。

---

## 212. SplitFT: An Adaptive Federated Split Learning System For LLMs Fine-Tuning

**arXiv ID:** 2604.26388 | [PDF](https://arxiv.org/pdf/2604.26388v1)

**作者:** Yimeng Shan `[一作]` (Hong Kong Polytechnic University), Benben Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 165 | [OpenAlex ID](https://openalex.org/A5081645974)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套名为 SplitFT 的联邦分割学习系统，用于在资源受限设备上对大型语言模型进行高效、可自适应的微调；系统通过动态切分层、LoRA 低秩适配和长度‑基 Dirichlet 数据划分，支持不同客户端在保证隐私的前提下实现分层训练。

**💡 创新点**

创新点包括：① 为每个客户端自适应选择切分层位置，以匹配其算力和数据质量；② 在切分层使用更低的 LoRA 秩来显著降低通信开销，同时保持模型性能；③ 引入长度‑基 Dirichlet 方法，模拟真实场景下的非同分布数据；④ 基于客户端表现动态调整权重的层分配启发式算法，平衡局部训练负载与全局性能。

**🔧 技术方法**

核心技术包括：联邦分割学习（Federated Split Learning）结合 FedAvg；LoRA（Low‑Rank Adaptation）实现参数高效微调；长度‑基 Dirichlet 数据划分；启发式自适应层分配策略；PyTorch+Flower 的实现框架。

**📊 数据集**

实验数据集为 Wikitext2‑v1，分别在 GPT‑2‑small、OPT‑125M 与 GPT‑Neo‑125M 三个 LLM 上进行微调；通过 Dirichlet 参数 α 控制 IID 与 Non‑IID 训练分布。

**📈 对比分析**

通过与固定切分层基线（Same Split）和无切分层（No Cut）进行对比，评估指标包括最终准确率、困惑度、训练耗时、轮次时间及通信量；结果显示 SplitFT 在非同分布场景下达到更低的困惑度、收敛更快、通信量更低，同时在不同模型上均优于基线。

**⚠️ 局限性**

局限性：① 仅在小规模 125M‑1.3B 级 LLM 上验证，尚未证明对更大模型的可扩展性；② LoRA 的低秩假设在某些任务中可能影响细粒度表达；③ 切分层和秩设置仍需手工调参，自动化程度有限；④ 对极端资源极低设备的兼容性尚需进一步实验；⑤ 仅在单一任务（文本生成）上评估，其他下游任务需补充验证。

---

## 213. SG-UniBuc-NLP at SemEval-2026 Task 6: Multi-Head RoBERTa with Chunking for Long-Context Evasion Detection

**arXiv ID:** 2604.26375 | [PDF](https://arxiv.org/pdf/2604.26375v1)

**作者:** Gabriel Stefan `[一作]` (University of Bucharest), Sergiu Nisioi `[通讯]` (University of Bucharest)

**通讯引用:** 315 | [OpenAlex ID](https://openalex.org/A5023022117)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们设计并实现了一套基于层级多任务学习的系统，能够对长篇政治访谈问答对进行模糊性与回避策略的分类；

**💡 创新点**

创新点主要包括：①采用重叠滑动窗口切分长文本并对每块进行RoBERTa编码；②用逐维最大池化聚合所有块的表示；③在共享编码器上同时训练三分类的清晰度头和九分类的回避策略头，并通过七折交叉验证集成；

**🔧 技术方法**

技术实现依赖RoBERTa-large编码器、重叠窗口切分、逐维最大池化、多任务交叉熵损失、AdamW优化器、线性学习率调度、七折分层交叉验证与模型平均；

**📊 数据集**

所用数据集为SemEval-2026 Task 6的CLARITY数据集，包含3756个英美白宫访谈问答对，训练集3458条，验证集308条；

**📈 对比分析**

通过对池化策略、多任务训练、集成规模、类别不平衡处理等的消融实验，我们的最终集成在官方测试集上取得了模糊度子任务Macro‑F1 0.80、回避策略子任务Macro‑F1 0.51，在各自子任务中排名第11；

**⚠️ 局限性**

主要局限包括：类别严重不平衡导致少数类召回低；细粒度回避策略间语义重叠导致混淆；模型过度依赖表面词汇，难以捕捉隐含的逻辑关系；

---

## 214. An Empirical Study of Speculative Decoding on Software Engineering Tasks

**arXiv ID:** 2604.26469 | [PDF](https://arxiv.org/pdf/2604.26469v1)

**作者:** Yijia Li `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**通讯引用:** 21498 | [OpenAlex ID](https://openalex.org/A5006669765)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Speculative Decoding 在软件工程的代码生成、修复和编辑任务中进行系统评估，探究不同模型规模与任务场景下的加速效果。

**💡 创新点**

首次将 SD 方法应用于长上下文、代理式交互的 SE 场景，系统量化其在模型规模、任务类型上的差异，并揭示无限循环噪声对评估的影响。

**🔧 技术方法**

使用 Prompt Lookup Decoding、Suffix Decoding、MLP Speculator 与 Eagle‑3 四种 SD 策略，基于 vLLM 进行 greedy 推理，测量 speedup、τ 等指标。

**📊 数据集**

评估数据集包括 LiveCodeBench、SWE‑bench、Aider Polyglot 三大 SE 基准，以及 MT‑bench 自然语言基准。

**📈 对比分析**

通过对比 speedup 与 Mean Acceptance Length 进行量化；在平均 1.3× 的加速率下，模型规模越小、任务可重复性越高时加速效果越好；但在大型模型或长上下文任务中可能出现负加速。

**⚠️ 局限性**

实验受限于仅三大基准与两类模型，未覆盖采样温度和更广泛任务；无限循环现象导致评估噪声；模型自由方法在自然语言任务中的效果有限。

---

## 215. Differentially Private Contrastive Learning via Bounding Group-level Contribution

**arXiv ID:** 2604.26467 | [PDF](https://arxiv.org/pdf/2604.26467v1)

**作者:** Kecen Li `[一作]` (National University of Singapore), Xiaokui Xiao `[通讯]` (National University of Singapore)

**通讯引用:** 15680 | [OpenAlex ID](https://openalex.org/A5010903591)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出了一种名为DP‑GCL的差分隐私对比学习框架，利用分组梯度裁剪与组内数据增强来降低样本间依赖，提高隐私训练的效用；

**💡 创新点**

创新点在于将批量划分为多个互不重叠的小组，仅在组内进行负样本比较，从而把梯度敏感度降到固定常数，并通过组内增强保持负样本多样性；

**🔧 技术方法**

核心技术包括：分组梯度裁剪（Bounding Group-level Contribution, BGC）、组内增强（Intra‑Group Sample Augmentation, ISA）、InfoNCE 损失、DP‑SGD+高斯噪声、RDP 计数器；

**📊 数据集**

实验数据集包括：单模态的 Fashion‑MNIST、CIFAR‑10、EuroSAT、Camelyon；多模态的 CUHK‑PEDES、RSTPReid、Fashion、ROCO；此外还在 ImageNet 上做迁移学习；

**📈 对比分析**

与 DP‑SGD、Logit‑DP、DP‑CLIP 等三种基线对比，在两个隐私预算 ϵ={1,10} 下，DP‑GCL 在单模态图像分类的线性探测/ k‑NN 准确率平均提升 5.6%；在多模态检索任务中，提升 20.1%（I→T/T→I）；并在大批量、参数高效微调（LoRA）等设置下仍保持领先；

**⚠️ 局限性**

局限性包括：缺乏大规模预训练实验（未在千万级或百亿级数据集上验证），以及仅基于 InfoNCE，未涵盖 BYOL、SigLIP 等无需负样本的对比方法；

---

## 216. Diffusion Reconstruction towards Generalizable Audio Deepfake Detection

**arXiv ID:** 2604.26465 | [PDF](https://arxiv.org/pdf/2604.26465v1)

**作者:** Bo Cheng `[一作]` (Southern University of Science and Technology), Fei Chen `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 11169 | [OpenAlex ID](https://openalex.org/A5100405410)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于难样本分类的音频深度伪造检测框架，利用重建生成难样本，结合多层特征聚合与正则化辅助对比学习（RACL）提升模型的泛化性能。

**💡 创新点**

创新点包括：①采用扩散式重建（SemantiCodec）产生最具挑战性的难样本；②设计双重对比损失（标准对比 + 强化对比）专注于难样本；③加入方差正则化损失使同类嵌入更紧凑；④将多层特征聚合与RACL融合，形成完整的训练目标。

**🔧 技术方法**

使用技术包括：扩散式重建（SemantiCodec）、XLS‑R 300M 预训练特征提取、AASIST 分类器、全局平均池化 + 一维卷积实现多层聚合、双重对比损失、方差正则化、交叉熵、Adam 优化器以及数据增强（RIR、MUSAN、混音）。

**📊 数据集**

采用的公开数据集：ASVspoof 2019 LA eval、CodecFake、DiffSSD、WaveFake、ITW，覆盖 TTS、VC、GAN、扩散和真实环境音频。

**📈 对比分析**

通过与基线（仅使用原始音频）和各重建方法（HiFi‑GAN、DAC、Encodec、Diffusion）对比，实验发现扩散式重建的平均 EER 为 12.22%，相较基线 15.79% 降低 22.6%；进一步加入多层聚合与 RACL 后，平均 EER 降至 8.25%，在五大数据集上实现显著提升。

**⚠️ 局限性**

局限性包括：在某些子集（如 ITW、部分 CodecFake 子集）仍出现轻微性能下降；重建质量对难样本生成敏感；扩散模型训练成本高；模型对极端新型攻击的鲁棒性仍待进一步验证。

---

## 217. Naamah: A Large Scale Synthetic Sanskrit NER Corpus via DBpedia Seeding and LLM Generation

**arXiv ID:** 2604.26456 | [PDF](https://arxiv.org/pdf/2604.26456v1)

**作者:** Akhil Rajeev P `[一作]` (Centre for Development of Advanced Computing), Annarao Kulkarni `[通讯]` (Centre for Development of Advanced Computing)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5110763986)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Naamah 语料，提供 102,942 句 Sanskrit NER 训练集，利用 DBpedia 实体与 24B 级混合推理 LLM 生成多样化合成句子。

**💡 创新点**

创新点在于使用领域专属 tokenizer 与大规模生成模型相结合，首次证明在 Sanskrit NER 上域适配模型可优于大规模多语种模型。

**🔧 技术方法**

使用 SPARQL 从 DBpedia 提取实体种子，借助 24B 级混合推理 LLM 生成合成文本，采用 BIO 标注、子词分词对齐及 Hugging Face Trainer 微调。

**📊 数据集**

使用 Naamah 合成数据集（102,942 句），其 92,647/10,295 的训练/验证拆分。

**📈 对比分析**

在相同拆分上微调 XLM‑RoBERTa Base 与 IndicBERTv2；IndicBERTv2 在验证集上实现 0.9615 F1，优于 XLM‑RoBERTa 的 0.9506，且模型体积更轻。

**⚠️ 局限性**

限制包括合成数据偏差、缺乏复杂沙恩处理、无金标评估，且对真实古典文本的泛化能力未知。

---

## 218. Reactive Motion Generation via Phase-varying Neural Potential Functions

**arXiv ID:** 2604.26450 | [PDF](https://arxiv.org/pdf/2604.26450v1)

**作者:** Ahmet Tekden `[一作]` (Chalmers University of Technology), Yasemin Bekiroglu `[通讯]` (Chalmers University of Technology)

**通讯引用:** 1294 | [OpenAlex ID](https://openalex.org/A5015300584)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于相位可变神经势能函数（PNPF）的学习示范框架，实现点对点、周期性以及全6D运动的实时、鲁棒生成。

**💡 创新点**

创新点在于：①通过从示范轨迹的名义能量推导闭环相位变量，解决相同状态多重后续动作的问题；②将安全能量与名义能量结合，构建数据驱动的安全集合；③利用神经场和超网络实现平滑可微的势能；④实现对周期、6D以及姿态的统一建模。

**🔧 技术方法**

使用神经场（hyper‑network 条件 MLP）、解码器‑仅网络生成轨迹、动态时间规整（DTW）选取名义轨迹、SDF 建立安全边界、ReLU 平滑函数、相位窗口化训练以及势能梯度控制。

**📊 数据集**

在仿真中评估 LASA、LAIR、RoboTasks、CHAR 与 CHAR‑Periodic 数据集；在真实机器人上测试结扎、倒酒和三维擦拭三种任务。

**📈 对比分析**

与 NODE、CONDOR、LPVDS 等基线比较，使用 DTW、Frechet 距离、终点误差与准确率等指标；PNPF 在所有数据集上均优于基线，尤其在 CHAR 上 DTW 下降约80%，且在受扰动、障碍规避实验中表现出更好的恢复与鲁棒性。

**⚠️ 局限性**

局限性包括：对尖锐角度的过度平滑；对多模态/非一致示范的适应性有限；局部能量梯度相互抵消时可能出现停滞；对示范质量高度依赖。

---

## 219. Sparsity as a Key: Unlocking New Insights from Latent Structures for Out-of-Distribution Detection

**arXiv ID:** 2604.26409 | [PDF](https://arxiv.org/pdf/2604.26409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 220. When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?

**arXiv ID:** 2604.26412 | [PDF](https://arxiv.org/pdf/2604.26412v1)

**作者:** Tianyu Liu `[一作]` (Qwen Applications Business Group of Alibaba), MingCheng Wan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在自回归推理中重用目标模型 KV 缓存对长期衰减的影响，并搭建 KVShot 框架进行系统对比

**💡 创新点**

提出 KV 重用与隐藏状态重用的“信息保留”视角，系统比较三种重用方式（hidden‑only、KV‑only、hybrid），揭示 KV 重用在长程更稳健但需改进训练流程

**🔧 技术方法**

使用 Speculative Decoding、EAGLE‑3 轻量化 drafter、autoregressive Test‑Time Training (TTT)、KVShot 对比框架、gated delta fusion、RoPE、线性投影、层级 KV 拼接等技术

**📊 数据集**

以 Qwen3‑8B 为目标模型，使用 ShareGPT（约 70k）进行快速迭代，使用 ShareGPT + UltraChat（约 280k）进行端到端评估

**📈 对比分析**

通过逐步接受率 α_k 与 MAT 评估，KV‑only 在多步上表现更好但整体 MAT 低于 EAGLE‑3；hybrid 在逐步接受率和 MAT 上提升（MAT 2.54 vs 2.37），但在 HuggingFace 端到端测试中 MAT 仅提升 0.6% 并伴随 5–10% 的推理延迟增加

**⚠️ 局限性**

autoregressive TTT 难以充分利用 KV 重用：查询估计能力不足、KV 投影梯度稀疏、门控机制导致梯度饥饿，导致端到端提升有限；KV‑only 在短程仍低于隐藏状态重用

---

## 221. Multi-Server Secure Aggregation with Arbitrary Collusion and Heterogeneous Security Constraints

**arXiv ID:** 2604.26391 | [PDF](https://arxiv.org/pdf/2604.26391v1)

**作者:** Zhou Li `[一作]` (Guangxi University), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

研究了多服务器安全聚合的理论极限，提出了一般性两跳网络模型并确定了通信率和随机数成本的最优值；

**💡 创新点**

创新点在于将任意协同集合和异构安全约束统一到一个信息理论框架，提出了e^*作为决定密钥率的核心指标，并针对剩余情况给出了线性规划可行方案；

**🔧 技术方法**

使用了信息理论极限分析、线性编码、Schwartz–Zippel 证明随机线性组合可行性等技术；

**📊 数据集**

无数据集，纯理论研究；

**📈 对比分析**

通过与已有的特殊安全聚合模型对照，证明在大多数参数配置下实现了最优或近似最优的密钥率，通信率保持在下限；

**⚠️ 局限性**

限制在于模型假设所有安全约束对每一对 (S_m,T_n) 都需满足，无法处理仅针对部分对的更一般情况，且未考虑用户掉线、选择等实际Federated Learning中的问题。

---

## 222. Rank Distribution and Dynamics of Gram Matrices from Binary m-Sequences with Applications to LCD Codes

**arXiv ID:** 2604.26387 | [PDF](https://arxiv.org/pdf/2604.26387v1)

**作者:** Hengfeng Liu `[一作]` (Southwest Jiaotong University), Zhengchun Zhou `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 4362 | [OpenAlex ID](https://openalex.org/A5004913173)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究二进制m序列生成的可观测矩阵的Gram矩阵，给出了所有Gram矩阵的秩分布、秩动态以及对应的码的hull分布；

**💡 创新点**

创新点在于将m序列与Gram矩阵关联，利用伽罗瓦群的半线性表示和多项式的Bézoutian，对所有t得到闭式秩分布，并揭示秩缺失状态不稳定、满秩状态持久的动态性质；

**🔧 技术方法**

核心技术包括：半线性Galois表示、对称多项式的Bézoutian、正规化有理函数表示、生成函数计数以及对m序列的迹表示；

**📊 数据集**

研究中没有使用外部实验数据，而是通过理论证明和对二进制有限域上的符号计算（如Magma）验证结果；

**📈 对比分析**

与之前关于m序列自相关、均衡性等属性的研究相比，本文提供了完整的Gram矩阵秩分布与动态分析，性能在理论上完备、无实验误差；

**⚠️ 局限性**

局限性包括仅针对二进制m序列，未给出q-元一般化的完整结论，且对更高阶码的实际编码效能未做实验验证。

---

## 223. Split over $n$ resource sharing problem: Are fewer capable agents better than many simpler ones?

**arXiv ID:** 2604.26374 | [PDF](https://arxiv.org/pdf/2604.26374v1)

**作者:** Karthik Soma `[一作]` (École Polytechnique de Montréal), Roderich Gross `[通讯]` (University of Sheffield)

**通讯引用:** 2750 | [OpenAlex ID](https://openalex.org/A5030911949)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了“资源分配到 n 个代理”问题，利用多代理覆盖任务为案例，结合正式分析和仿真评估不同资源分配水平对覆盖效率的影响。

**💡 创新点**

创新点在于将传统规模效应研究转向有限资源场景，系统性探讨了代理数量、速度剖面、碰撞与失效等因素对覆盖性能的交互效应，并给出了针对不同速度剖面的最优分配建议。

**🔧 技术方法**

采用随机行走模型（受 Brownian 和 Wrapped Cauchy 分布约束）、四种速度剖面（常数、线性、按半径、按面积）、碰撞退避规则和失效率模型，利用 OpenCV 进行覆盖率计算和仿真。

**📊 数据集**

使用合成仿真环境：周期性 1×1 单位方格，离散化为 1000×1000 网格，整体占地面积固定为 A=π×0.1²，测试 30 次不同初始条件，得到覆盖曲线。

**📈 对比分析**

通过对比不同 n、速度剖面、碰撞、失效率下的覆盖时间 t_f 与覆盖百分比，发现：常数速度下 n 越大越好；线性速度最佳在约 500 代理；按半径速度几乎无差异；按面积速度则 n=1 最优；失效率升高时单一高性能代理往往优于大量易失效的代理。

**⚠️ 局限性**

局限性包括：仅考虑同质、无物理动力学约束的代理；碰撞退避简化；失效率模型仅与 n 相关；未评估异构机器人群、真实物理环境以及更复杂任务。

---

## 224. Unifying Runtime Monitoring Approaches for Safety-Critical Machine Learning: Application to Vision-Based Landing

**arXiv ID:** 2604.26411 | [PDF](https://arxiv.org/pdf/2604.26411v1)

**作者:** Mathieu Dario `[一作]` (LAAS-CNRS), Jérémie Guiochet `[通讯]` (LAAS-CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套统一的运行时监控框架SwMF，将安全监控方法分为操作设计域(ODD)监控、离散分布(OOD)监控以及模型范围外(OMS)监控，并在航空跑道检测任务中验证其有效性。

**💡 创新点**

通过阐明三类监控的概念并消除研究社区之间的重叠，构建了一个多层级、互补的分类体系；同时提供了统一的评估指标与实验流程，以实现对不同监控方法的系统比较。

**🔧 技术方法**

使用YOLOv5-n目标检测模型；ODD监控基于规则与元特征；OOD监控通过提取亮度、饱和度、熵、边缘等图像属性并拟合贝塔分布阈值；OMS监控采用盒抽象(Box‑Abstraction)方法对模型中间层logits进行检测。

**📊 数据集**

实验使用LARD数据集（12,212张合成图像训练，2,212张合成测试，另附15种腐败的图像），并对测试图像施加多级亮度、模糊、雾等噪声。

**📈 对比分析**

对监控效果采用安全收益(SG)、残留危害(RH)和可用性成本(AC)指标进行评估。实验一表明三类监控各自提供独立收益，组合后接近各自收益之和，但可用性成本显著上升；实验二中OOD监控在各类噪声下实现高安全收益、低残留危害，但AC较高；OMS监控表现相对弱，提示需顺序过滤。

**⚠️ 局限性**

主要限制在于监控引入的误报率导致可用性成本过高；不同监控方法仍需进一步优化误报/漏报平衡；数据集的代表性与选择仍影响监控效果，需要更系统的数据收集与评估方法。

---

## 225. Efficient Listwise Reranking with Compressed Document Representations

**arXiv ID:** 2604.26483 | [PDF](https://arxiv.org/pdf/2604.26483v1)

**作者:** Hervé Déjean `[一作]` (NAVER LABS Europe), Stéphane Clinchant `[通讯]` (NAVER LABS Europe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RRK模型，通过压缩文档表示实现高效的列表式重新排序。

**💡 创新点**

将软压缩技术与列表式重排序结合，使用多token压缩表示而非传统IR嵌入，显著降低输入长度。

**🔧 技术方法**

基于PISCO的压缩器与LoRA微调的解码器重排序器，使用RankNet损失训练，采用Qwen-2.5 8B backbone。

**📊 数据集**

MS MARCO（passage）和E2RANK提供的150k查询，评测使用TREC DL19/20、BeIR和MS-MARCO Document。

**📈 对比分析**

与公开的Jina‑V3、Qwen3 0.6B/4B、ModernBERT等模型对比，在512token输入下RRK保持效果相当并比同等参数模型快3–18倍，长文档下也优于非压缩列表模型。

**⚠️ 局限性**

需额外存储压缩向量导致索引体积增大，压缩效果受文档长度与查询长度限制，且小型模型尚未成功。

---

## 226. Cross-Domain Transfer of Hyperspectral Foundation Models

**arXiv ID:** 2604.26478 | [PDF](https://arxiv.org/pdf/2604.26478v1)

**作者:** Nick Theisen `[一作]` (University of Koblenz), Peer Neubert `[通讯]` (University of Koblenz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出并验证了跨域知识迁移在近地表高光谱语义分割中的有效性，利用遥感领域预训练的高光谱基础模型在驾驶场景中直接进行微调。

**💡 创新点**

创新点在于将遥感高光谱基础模型迁移到近地表任务，避免了模态桥接所导致的光谱信息丢失和复杂架构，且在类平均指标上显著提升模型对少数类别的鲁棒性。

**🔧 技术方法**

主要技术包括基于Spectral Transformer的光谱分词器和编码器、冻结预训练权重、在特征层或分类层进行微调，并与传统全连接分类头对比。

**📊 数据集**

实验使用了HS3‑Bench benchmark 中的三组近地表高光谱数据集：*hcv、dl3 和 *hsi，涵盖 15-128 个波段与不同光谱范围。

**📈 对比分析**

通过与传统的域内‑模态训练、跨模态（RGB‑投影或RGB‑基础模型）迁移以及基准模型对比，跨域迁移平均提升 mIoU 约 3%，在少数类别表现更佳，并且在仅 10% 数据量时仍能取得明显优势。

**⚠️ 局限性**

局限性包括：仍无法完全赶超跨模态方法；仅在驾驶场景下验证，可能对其他近地表应用的泛化未知；受限于现有遥感高光谱基础模型的规模和领域覆盖；以及在噪声较大的近红外波段中光谱特征的辨识度较低。

---

## 227. Hierarchical adaptive control for real-time dynamic inference at the edge

**arXiv ID:** 2604.26470 | [PDF](https://arxiv.org/pdf/2604.26470v1)

**作者:** Francesco Daghero `[一作]` (University of Southern Denmark), Mikkel Baun Kjærgaard `[通讯]` (University of Southern Denmark)

**通讯引用:** 5237 | [OpenAlex ID](https://openalex.org/A5034655219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种双层自适应控制架构，实现边缘节点动态推理的实时高效部署；

**💡 创新点**

1）预算化的SP级联设计保证最坏情况延迟约束；2）分层控制器实现对数据漂移与硬件资源变化的在线自适应；3）将动态推理与节点级和全局级控制器耦合，提升长期能效与可靠性；

**🔧 技术方法**

采用一对多（SP）轻量级二分类器级联与多分类回退模型，配合阈值决策；局部控制器监测CPU/内存/温度并基于指数移动平均调整SP启用顺序；全局控制器基于Beam搜索与Pareto前沿选择最佳级联配置，并可在运行时推送新SP；

**📊 数据集**

土壤分类数据集(SCD)和CIFAR-10；

**📈 对比分析**

与单一静态模型以及无控制器的动态级联进行对比，实验表明在SCD上平均延迟可降至静态配置的2.45倍、能耗降至2.86倍；在CIFAR-10上延迟提升1.18倍、能耗提升1.20倍；在分布漂移场景中，框架仍保持低于4%准确率下降；

**⚠️ 局限性**

仅适用于二分类SP和固定任务单节点场景；控制器实现基于Python，导致额外开销；对光照、传感器老化等更复杂漂移的适应性尚未验证；

---

## 228. $\text{PKS}^4$:Parallel Kinematic Selective State Space Scanners for Efficient Video Understanding

**arXiv ID:** 2604.26461 | [PDF](https://arxiv.org/pdf/2604.26461v1)

**作者:** Lingjie Zeng `[一作]` (Sichuan University), Qijun Zhao `[通讯]` (Sichuan University)

**通讯引用:** 4984 | [OpenAlex ID](https://openalex.org/A5085914001)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种并行运动学选择状态空间扫描器（PKS^4），通过在图像预训练的ViT中插入单一轻量模块实现线性复杂度的时序建模，并结合运动学先验提升视频理解效率。

**💡 创新点**

核心创新在于：①将运动学先验（相邻帧相关性与变化量）编码到扫描器输入；②采用并行Patch‑wise状态空间扫描，保留二维空间结构且仅需一次模块插入；③轻量化CLS时序路径，避免多层适配器造成的激活内存瓶颈。

**🔧 技术方法**

使用的技术包括：图像预训练ViT backbone、Kinematic Prior Encoder（相关性与差分运算）、Parallel Kinematic Selective State Space Scanner（SSM变体）、单向或双向扫描、轻量CLS时序卷积。

**📊 数据集**

在两个主流动作识别基准上评估：Something‑Something V2（SSV2）和 Kinetics‑400（K400）。

**📈 对比分析**

与时空注意力、参数高效微调（PEFT）和纯SSM模型相比，PKS^4在SSV2上实现了接近或超过 70% 的 Top‑1 率，仅用 20 轮训练，计算成本约为纯视频SSM 的 1/10；在 K400 上保持竞争力，计算量亦低于多数基线。

**⚠️ 局限性**

局限性：对以外观特征为主的基准（如 K400）提升不明显；在静态区域的扫描可能仍需改进；未来可探索自适应门控机制以更好区分动态与静态内容。

---

## 229. Last-Layer-Centric Feature Recombination: Unleashing 3D Geometric Knowledge in DINOv3 for Monocular Depth Estimation

**arXiv ID:** 2604.26454 | [PDF](https://arxiv.org/pdf/2604.26454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 230. Are Data Augmentation and Segmentation Always Necessary? Insights from COVID-19 X-Rays and a Methodology Thereof

**arXiv ID:** 2604.26437 | [PDF](https://arxiv.org/pdf/2604.26437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 231. Decoupled Prototype Matching with Vision Foundation Models for Few-Shot Industrial Object Detection

**arXiv ID:** 2604.26404 | [PDF](https://arxiv.org/pdf/2604.26404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 232. Advancing multi-site emission control: A physics-informed transfer learning framework with mixture of experts for carbon-pollutant synergy

**arXiv ID:** 2604.26571 | [PDF](https://arxiv.org/pdf/2604.26571v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 233. MPI Malleability Validation under Replayed Real-World HPC Conditions

**arXiv ID:** 2604.26576 | [PDF](https://arxiv.org/pdf/2604.26576v1)

**作者:** S. Iserte `[一作]` (Barcelona Supercomputing Center), A. J. Peña `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在支持 MPI malleability 的 HPC 集群上，利用真实超算日志重放结合用户采样方法，对动态资源管理技术进行验证与评估。

**💡 创新点**

提出了基于用户采样的工作负载重放与反馈机制，首次在真实环境中验证 MPI malleability 的实际效益，并展示了以学生作业为案例的可行性。

**🔧 技术方法**

核心技术包括 MPI malleability、DMRlib 与 Slurm 的插件扩展、User‑Based Submitter、MPDATA 并行程序以及 MPICH/MPICH‑3.2 运行时。

**📊 数据集**

使用 KIT‑FH2‑2016 并行工作负载日志（2017年7月19天）以及 MPDATA 10 个实例作为实验负载。

**📈 对比分析**

通过 5 个实验（Baseline、StaticN32、StaticN16、AlwaysGrow、ParEfficiency）对 makespan、资源利用率、等待时间等指标进行比较，结果显示 ParEfficiency 场景下学生作业完成时间比 StaticN32 缩短约 27%，同时保持整体资源利用率不下降。

**⚠️ 局限性**

限制包括仅测试单一动态作业和单一应用、仅在一个 125 节点的分区上进行实验、对原始日志进行 10 倍时间压缩、使用模拟作业而非真实计算、未进行多次重复实验以及未评估多动态作业竞争，可能影响结果的普适性和可推广性。

---

## 234. SafeReview: Defending LLM-based Review Systems Against Adversarial Hidden Prompts

**arXiv ID:** 2604.26506 | [PDF](https://arxiv.org/pdf/2604.26506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 235. HiPAN: Hierarchical Posture-Adaptive Navigation for Quadruped Robots in Unstructured 3D Environments

**arXiv ID:** 2604.26504 | [PDF](https://arxiv.org/pdf/2604.26504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 236. LLM-Flax : Generalizable Robotic Task Planning via Neuro-Symbolic Approaches with Large Language Models

**arXiv ID:** 2604.26569 | [PDF](https://arxiv.org/pdf/2604.26569v1)

**作者:** Seongmin Kim `[一作]` (Jeonbuk National University), Daegyu Lee `[通讯]` (Jeonbuk National University)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5017709409)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个三阶段框架，利用本地LLM自动生成神经符号规划所需的规则、失败恢复和对象重要性评分，完全无人工域工程。

**💡 创新点**

创新点在于：①将手工规则替换为LLM自动生成；②引入预算感知的LLM失败恢复，避免规划过程耗时；③用零射击LLM评分替代GNN训练，消除训练数据需求。

**🔧 技术方法**

主要技术包括：结构化提示与自我校正的LLM推理（Gemma3-12B等）、预算感知的API调用策略、JSON规则格式校验与后处理、零射击对象评分。

**📊 数据集**

使用MazeNamo格子迷宫基准，涵盖10×10、12×12、15×15三种网格规模及其不同难度。

**📈 对比分析**

与手工规则+GNN基线对比，平均成功率从0.828提升至0.945；在12×12 Expert从0.000提升至0.733，在15×15 Hard从0.900提升至1.000；阶段3在12×12 Hard获得0.720的成功率，但在15×15 Hard仅0.200。

**⚠️ 局限性**

缺点在于：对大型问题的上下文窗口限制导致零射击评分性能下降；方向性互补规则可能导致对象数目激增，影响大规模网格；LLM生成规则的质量依赖于模型选择和提示设计。

---

## 237. Do Larger Models Really Win in Drug Discovery? A Benchmark Assessment of Model Scaling in AI-Driven Molecular Property and Activity Prediction

**arXiv ID:** 2604.26498 | [PDF](https://arxiv.org/pdf/2604.26498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 238. DenseStep2M: A Scalable, Training-Free Pipeline for Dense Instructional Video Annotation

**arXiv ID:** 2604.26565 | [PDF](https://arxiv.org/pdf/2604.26565v1)

**作者:** Mingji Ge `[一作]` (Shanghai Jiao Tong University), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10301 | [OpenAlex ID](https://openalex.org/A5076097168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套无训练、自动化的三阶段管线，用于从大规模、噪声较大的 HowTo100M 指令视频中生成高质量、时间上对齐的操作步骤；并基于该管线生成 DenseStep2M 数据集（约 99K 视频、约 190 万步骤）。

**💡 创新点**

创新点在于：①将视频分割、文本过滤、语音纠正与视觉-文本一致性检测结合，自动剔除非指令内容；②利用大型多模态 LLM（Qwen2.5‑VL）先生成候选步骤，再用推理型 LLM（DeepSeek‑R1）进行全局校验与时间修正，显著提升步骤的语义准确性与时间精度；③通过训练‑free 的方式完成全流程，保持可扩展性与高效性；④构建 DenseCaption100 人工标注基准，用于验证自动注释质量。

**🔧 技术方法**

技术包括：WhisperX ASR、Qwen2.5‑VL‑72B 多模态推理、DeepSeek‑R1 逻辑校验、视频分段与对齐、LoRA 微调、R1‑Score 评估指标、跨模态检索评估框架。

**📊 数据集**

使用源数据 HowTo100M（约 100K 片段），生成 DenseStep2M；人工校准的 DenseCaption100（100 条视频）用于评测；在下游任务中使用 YouCook2、EgoMCQ、MSRVTT、DiDeMo、CharadesEgo、EgoExoBench 等标准数据集进行验证。

**📈 对比分析**

与现有方法对比：在 YouCook2 上，未经预训练的 DenseStep2M 达到 METEOR 9.4、CIDEr 33.6、SODA_c 5.6，显著高于 CM^2 等对比模型；在 DenseCaption100 上，R1‑Recall 达到 86.0、F1 75.8、mIoU 42.9；在下游任务中，Fine‑tune Qwen2.5‑VL‑3B、Qwen3‑VL‑Embedding‑8B 等模型在 DenseCaption100、DenseVideo Captioning、步骤定位、跨模态检索等任务均获得 3–10% 以上绝对提升，尤其在 egocentric‑exocentric 统一表示上取得显著效果。

**⚠️ 局限性**

限制包括：①管线对视频质量和 ASR 质量仍有一定依赖，极端噪声视频可能过滤率高；②生成步骤仍可能出现细粒度语义歧义或未覆盖的“隐式”操作；③对新领域（如工业装配、医疗手术）的泛化需要进一步验证；④对 GPU 资源和大模型推理成本仍高，限制大规模部署。

---

## 239. Counting own goals: High-level assessment of the economic relationship between the ICT and the Oil and Gas sectors and its environmental implications

**arXiv ID:** 2604.26539 | [PDF](https://arxiv.org/pdf/2604.26539v1)

**作者:** Gauthier Roussilhe `[一作]` (Hubblo), Srinjoy Mitra `[通讯]` (School of Engineering)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

使用输入-输出分析与文献综述，量化ICT行业向油气行业的经济流动，并通过案例研究评估数字化导致的额外碳排放。

**💡 创新点**

首次系统性评估ICT与油气的经济共进关系，并提出数字化导致的“额外排放”框架以及将GPU技术与油气仿真关联的历史路径。

**🔧 技术方法**

输入-输出分析、聚焦文献综述、案例量化估算以及ITU‑T L.1480等碳足迹方法。

**📊 数据集**

EXIOBASE3多区域输入-输出表、能源价格数据、企业财报及行业报告。

**📈 对比分析**

通过与可再生与核能行业的ICT投入对比，发现油气行业投入约四倍；案例研究估算数百万吨CO₂的潜在额外排放，显示方法可量化但精度有限。

**⚠️ 局限性**

数据分辨率低、缺乏服务层级的交易信息、无法直接推断因果关系、仅使用宏观经济层面，需更细粒度的产品级交易和环境延伸数据。

---

## 240. Sparse-on-Dense: Area and Energy-Efficient Computing of Sparse Neural Networks on Dense Matrix Multiplication Accelerators

**arXiv ID:** 2604.26587 | [PDF](https://arxiv.org/pdf/2604.26587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 241. Preventing Distinguishability between Multiplication and Squaring Operations

**arXiv ID:** 2604.26536 | [PDF](https://arxiv.org/pdf/2604.26536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 242. StarDrinks: An English and Korean Test Set for SLU Evaluation in a Drink Ordering Scenario

**arXiv ID:** 2604.26500 | [PDF](https://arxiv.org/pdf/2604.26500v1)

**作者:** Marcely Zanon Boito `[一作]` (NAVER LABS Europe), Ioan Calapodescu `[通讯]` (NAVER LABS Europe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了 StarDrinks 数据集，用于评估英语和韩语的饮料点单语音助手，包含语音录音、转写文本和槽位标注。

**💡 创新点**

创新点包括：①提供真实口语点单数据，覆盖多种语言与复杂槽位；②公开统一的 SLU/ASR/NLU 测试集；③通过基线实验揭示 ASR 对未知实体的适应难题和 NLU 的鲁棒性提升。

**🔧 技术方法**

使用 Whisper‑large‑v3 进行 ASR，GPT‑4o 进行槽位填充（SLU/ NLU），并采用 0-shot、3-shot 以及 6/10-shot 提示进行实验；评估指标为 WER/CER、UEM 与 Slot F1。

**📊 数据集**

使用了自建的 StarDrinks 数据集（291 条英文、295 条韩文语音），并在对比实验中参考了 ATIS、SLURP、Speech‑MASSIVE、FoodOrdering 等公开数据集。

**📈 对比分析**

评估方法：用 WER/CER 衡量 ASR 质量，用 UEM（Unordered Exact Match）和 Slot F1 衡量 SLU/ NLU 成绩。实验结果显示：英文 WER 9.2%、CER 3.6；韩文 WER 22.9%、CER 7.3；在金标准转写下 UEM 为 87.06%（英）和 89.83%（韩），ASR 转写下下降 11.76%/17.96%；3‑shot 提示使 UEM 分别提升 15.3%/4.07%，Slot F1 亦相应提升。

**⚠️ 局限性**

局限性：ASR 对未见实体识别困难，需进一步研究测试时适应；NLP/SLU 的准确率仍低于部署需求，尤其在多属性组合与口语失语、停顿等现象下表现不佳；数据集仅涵盖单轮交互，样本量与多样性有限。

---

## 243. Persona-Based Process Design for Assistive Human-Robot Workplaces for Persons with Disabilities

**arXiv ID:** 2604.26527 | [PDF](https://arxiv.org/pdf/2604.26527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 244. Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective

**arXiv ID:** 2604.26489 | [PDF](https://arxiv.org/pdf/2604.26489v1)

**作者:** Jiancheng Wang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28971 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过从维度坍塌的角度，系统评估并解析深度神经网络（DNN）在特征交互模型中的作用，证明并解释了并行和堆叠DNN能显著缓解嵌入向量的维度坍塌。

**💡 创新点**

创新点在于将DNN的效果与嵌入维度鲁棒性关联，首次用RankMe等谱指标衡量维度坍塌，并通过梯度分析揭示DNN缓解坍塌的机制。

**🔧 技术方法**

技术上采用了并行DNN与堆叠DNN两种架构，进行了线性模块与非线性激活的逐步剥离实验，并基于梯度流和奇异值谱进行理论与经验分析。

**📊 数据集**

实验数据集为公开的CTR数据集Avazu和Criteo。

**📈 对比分析**

与基准模型FM、CrossNet以及其DNN增强版本DeepFM、NFM、DCNv2等进行AUC和RankMe对比，实验显示DNN不仅提升AUC（如FM从0.7848提升到0.7927），更显著提升RankMe（从89.12提升到166.78），验证了维度坍塌缓解效果。

**⚠️ 局限性**

局限在于仅针对二阶显式交互与高阶交互模型，未深入探究其他交互形式；梯度分析多在理论层面，缺乏更细粒度的实验验证；且实验只覆盖两组数据集，泛化性待进一步验证。

---

## 245. GIFGuard: Proactive Forensics against Deepfakes in Facial GIFs via Spatiotemporal Watermarking

**arXiv ID:** 2604.26519 | [PDF](https://arxiv.org/pdf/2604.26519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 246. MTCurv: Deep learning for direct microtubule curvature mapping in noisy fluorescence microscopy images

**arXiv ID:** 2604.26517 | [PDF](https://arxiv.org/pdf/2604.26517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 247. Benchmarking the Safety of Large Language Models for Robotic Health Attendant Control

**arXiv ID:** 2604.26577 | [PDF](https://arxiv.org/pdf/2604.26577v1)

**作者:** Mahiro Nakao `[一作]` (Kyushu Institute of Technology), Kazuhiro Takemoto `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 2667 | [OpenAlex ID](https://openalex.org/A5013426338)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在医学机器人控制场景下，对72个LLM进行安全评估，构建270条基于AMA伦理原则的有害指令并与对照良性指令配对，使用RHA仿真框架和LLM‑as‑a‑Judge评估模型对有害指令的拒绝或违规率。

**💡 创新点**

创新点在于首次将医学伦理标准与机器人行动规划结合，提出专门针对医学机器人有害指令的数据集，系统比较专有与开源模型、模型规模、发布时间以及医学领域微调对安全性的影响，并评估基于Self‑Reminder的prompt防御。

**🔧 技术方法**

技术上使用LLM‑as‑a‑Judge评估、RHA框架的JSON指令交互、混合效应回归分析、Wilcoxon检验和Spearman相关等统计方法。

**📊 数据集**

数据集为270条基于AMA伦理原则的医学机器人有害指令（9类各30条）及其对应的270条良性指令，全部公开发布。

**📈 对比分析**

比较方法：计算违规率、过度拒绝率，并用Kruskal‑Wallis、Wilcoxon、Spearman相关和混合效应回归比较不同模型特征；结果显示平均违规率54.4%，专有模型显著更安全，规模和发布时间均负向相关，医学微调无显著益处，Self‑Reminder仅减低约5个百分点。

**⚠️ 局限性**

局限包括仅在仿真环境测试、评估依赖单一LLM‑as‑a‑Judge、未涵盖更复杂的物理机器人交互、仅评估了一种prompt防御、以及缺乏对抗性攻击鲁棒性评估。

---

## 248. Grounding vs. Compositionality: On the Non-Complementarity of Reasoning in Neuro-Symbolic Systems

**arXiv ID:** 2604.26521 | [PDF](https://arxiv.org/pdf/2604.26521v1)

**作者:** Mahnoor Shahid `[一作]` (Universität Duisburg Essen), Hannes Rothe `[通讯]` (Universität Duisburg Essen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过实验验证了在神经符号系统中，单纯的符号归纳并不足以实现组合式泛化，提出并实现了Iterative Logic Tensor Network（iLTN）模型，实现了可微分的多步逻辑推理；

**💡 创新点**

创新点在于：①首次系统拆分符号归纳与推理两项任务，检验它们是否互为依赖；②提出的iLTN通过迭代细化循环与Gumbel-Softmax实现可微分的多步推理；

**🔧 技术方法**

主要技术包括：可微分逻辑张量网络（LTN）框架、基于Lukasiewicz t-范式的逻辑可满足性计算、Gumbel-Softmax连续化采样、可变步长的迭代推理与终止机制；

**📊 数据集**

使用了基于ClassicLogic生成的合成视觉逻辑谜题数据集，图像尺寸为84×84像素；

**📈 对比分析**

在三类组合式泛化任务（实体、关系、规则）上，iLTN在零样本下整体准确率达到51.2%，显著高于仅归纳（11.3%）和仅推理（低于30%）的基线；

**⚠️ 局限性**

局限性：实验仅在合成视觉逻辑谜题上验证，未测试在真实噪声环境或更大规模问题空间中的可扩展性和鲁棒性。

---

## 249. Graph Construction and Matching for Imperative Programs using Neural and Structural Methods

**arXiv ID:** 2604.26578 | [PDF](https://arxiv.org/pdf/2604.26578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 250. 3D-LENS: A 3D Lifting-based Elevated Novel-view Synthesis method for Single-View Aerial-Ground Re-Identification

**arXiv ID:** 2604.26520 | [PDF](https://arxiv.org/pdf/2604.26520v1)

**作者:** William Grolleau `[一作]` (Université Paris-Saclay), Catherine Achard `[通讯]` (Sorbonne University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出单视角空地重识别（SV AG‑ReID）任务，并设计 3D‑LENS 框架：先用大规模 3D 重建将单视图图像上升为纹理化网格，随后通过精准的相机姿态控制生成与目标视角一致的合成图像，再用课程学习、平衡采样和域分类等技术进行鲁棒表征学习。

**💡 创新点**

①首次在无跨视角标注的条件下完成空地重识别；②利用通用 3D 重建实现类别无关、几何一致的视角合成；③通过课程调度与平衡采样有效消除合成与真实数据的差异，提升跨视角鲁棒性。

**🔧 技术方法**

大规模 3D 重建（Hunyuan3D）、相机姿态对齐、风格迁移（StyleID）、背景合成、课程学习、平衡采样、域分类损失以及 ViT 基础网络。

**📊 数据集**

AG‑ReID、AG‑ReID.v2（含 UAV、CCTV、穿戴摄像头视角）和 MOO（跨视角的牛群识别）数据集。

**📈 对比分析**

与多种基线（BoT、TransReID、VDT、RotTrans、PASS、DCAC 等）在所有单视角训练与跨视角检索场景下进行对比，3D‑LENS 在 AG‑ReID 上 mAP 提升 14.30pp，AG‑ReID.v2 上提升 21.20pp，MOO 上提升 10.0pp，整体显著优于现有方法。

**⚠️ 局限性**

合成视角的质量受限于源图像分辨率、遮挡和多人重叠；低质量 3D 重建会产生伪影，可能导致模型过拟合或性能下降，亟需自动化质量控制机制。

---

## 251. Lyapunov-Guided Self-Alignment: Test-Time Adaptation for Offline Safe Reinforcement Learning

**arXiv ID:** 2604.26516 | [PDF](https://arxiv.org/pdf/2604.26516v1)

**作者:** Seungyub Han `[一作]`, Jungwoo Lee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用离线训练好的 transformer 进行测试时自我对齐，以提升安全性。

**💡 创新点**

首次在测试阶段通过 Lyapunov 条件和占据量筛选安全轨迹，再将安全片段作为 in-context prompts 进行无参数更新的自我对齐。

**🔧 技术方法**

Transformer-based RL（Decision Transformer + VAE 世界模型）、Lyapunov 密度模型、占据量估计、贝叶斯推断、层次 RL 的自我对齐框架。

**📊 数据集**

D4RL 与 DSRL 离线数据集，Safety Gymnasium 与 MuJoCo 基准任务。

**📈 对比分析**

与多种离线安全 RL 基线（BC、BCQ、BEAR、CPQ、COptiDICE、DCRL）以及基于 transformer 的 baseline 对比，SAS 在降低成本/失败率的同时保持甚至提升回报。

**⚠️ 局限性**

需要额外的推理开销（多条想象轨迹），安全性依赖离线数据的覆盖度，且仅适用于 transformer 结构，无法直接处理显式成本约束。

---

## 252. Culturally Aware GenAI Risks for Youth: Perspectives from Youth, Parents, and Teachers in a Non-Western Context

**arXiv ID:** 2604.26494 | [PDF](https://arxiv.org/pdf/2604.26494v1)

**作者:** Aljawharah Alzahrani `[一作]` (Pennsylvania State University), Tanusree Sharma `[通讯]` (Pennsylvania State University)

**通讯引用:** 532 | [OpenAlex ID](https://openalex.org/A5019575644)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了沙特阿拉伯青少年、父母和教师使用生成式人工智能的行为与风险，并提出了文化适应的安全控制建议。

**💡 创新点**

首次构建了针对非西方文化的生成式AI风险分类，识别了宗教、文化规范与数据隐私的独特风险。

**🔧 技术方法**

采用混合方法，包括社交媒体文本分析与半结构化访谈。

**📊 数据集**

分析了736条Reddit帖子、1,262条X/Twitter帖子，并对30名沙特参与者进行访谈。

**📈 对比分析**

通过对比西方风险框架验证新风险类别的有效性，发现新增类别更能解释沙特使用场景，但未量化性能指标。

**⚠️ 局限性**

样本规模有限、跨文化迁移性不确定，且对技术实现细节缺乏实验验证。

---

## 253. PAINT: Partial-Solution Adaptive Interpolated Training for Self-Distilled Reasoners

**arXiv ID:** 2604.26573 | [PDF](https://arxiv.org/pdf/2604.26573v1)

**作者:** Zhiquan Tan `[一作]` (Tsinghua University), Yinrong Hong `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PAINT 方法，在大语言模型推理后训练中通过部分解自适应遮罩和稀疏能量插值改进自监督对齐，提升数学推理性能。

**💡 创新点**

创新点在于：① 采用上下文重新评分视角，将特权解视为对已生成前缀的能量重新加权；② 用重叠度自适应控制解遮罩比例，避免全解过拟合；③ 只在高熵不匹配位置进行教师能量插值，减少无效监督。

**🔧 技术方法**

技术手段包括：on‑policy 自蒸馏、重叠度测度、部分解遮罩、稀疏能量插值、LoRA 微调、温度/top‑p 采样等。

**📊 数据集**

使用 OpenThoughts 里的数学推理子集作为训练数据；评估数据为 AIME 2024/2025 及 HMMT 2025。

**📈 对比分析**

与 SFT、GRPO、OPSD 在同一模型、相同采样预算下对比；在 Qwen3-8B、4B、1.7B 上，PAINT 在宏平均上比 OPDS 提升 0.8–2.1 分，且在相同或更小采样预算下匹配或超越 GRPO。

**⚠️ 局限性**

局限性：仅在已验证答案的数学推理场景验证，未针对噪声或非数学任务；遮罩长度、位置及能量插值比例等超参数仍需经验调节；对特权解的依赖可能限制在无可靠验证环境下的应用。

---

## 254. Multimodal LLMs are not all you need for Pediatric Speech Language Pathology

**arXiv ID:** 2604.26568 | [PDF](https://arxiv.org/pdf/2604.26568v1)

**作者:** Darren Fürst `[一作]` (Ostbayerische Technische Hochschule), Ulrich Schäfer `[通讯]` (Ostbayerische Technische Hochschule)

**通讯引用:** 2955 | [OpenAlex ID](https://openalex.org/A5010066502)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过构建层级化分类管道和针对性数据增强，评估声学表示模型（SRM）在儿童语音障碍检测与自动语音识别（ASR）上的表现，并与多模态大模型（LLM）进行对比。

**💡 创新点**

创新点包括：①利用任务层级结构的级联分类架构提升细粒度诊断性能；②采用性别对齐的音高增强以缓解模型性别偏差；③证明 SRM 在 SSD 检测与 ASR 上可显著优于 LLM。

**🔧 技术方法**

主要技术：Fine‑tune WavLM、wav2vec2、Hubert 等 SRM；使用 LoRA 微调 Whisper ASR；实现 Gaussian 噪声和 pitch‑shifting 数据增强；对不同层级任务采用级联训练。

**📊 数据集**

使用 SLPHelmUltraSuitePlus 基准数据集，包含 926 条儿童语音样本、细粒度诊断（T1）、类型（T2）和症状（T3）标签以及专业转录。

**📈 对比分析**

与多模态 LLM（Phi‑4、GPT‑4o、Whisper）在同一基准下比较，SRM 在 T1/T2/T3 的宏 F1/召回分别为 0.956/0.697/0.391，显著高于 LLM（F1 仅 0.535/0.163/0.118）；ASR 方面，Whisper‑large‑v3‑turbo 在完整语音上达到 0.640 EM、0.814 F1、0.194 WER，明显优于 LLM。

**⚠️ 局限性**

局限性：仅针对单一基准数据集，音频被截短至 12 秒；数据规模有限，可能限制模型在更大或更长语音样本上的泛化能力。

---

## 255. DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference

**arXiv ID:** 2604.26557 | [PDF](https://arxiv.org/pdf/2604.26557v1)

**作者:** Bodon Jeong `[一作]` (Sogang University), Sungyong Park `[通讯]` (Sogang University)

**通讯引用:** 841 | [OpenAlex ID](https://openalex.org/A5101413142)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在边缘 AI 系统中实现一种双路径 KV 缓存卸载框架，动态将 KV 张量放在页缓存或 NVMe 直通路径，以降低推理延迟。

**💡 创新点**

创新点包括：① 双路径 KV 居留策略避免页缓存抖动；② NVMe-Direct 直通实现文件系统栈绕过和连续 LBA 排布；③ 自适应管道并行化在 GPU DMA 与存储 I/O 之间重叠，进一步隐藏 I/O 延迟。

**🔧 技术方法**

采用 NVMe-Direct I/O、连续 LBA 调度、页缓存预算估算、GPU 与 NVMe 的 Co‑DMA、TRIM 数据集管理以及多线程自适应管道策略。

**📊 数据集**

在 OPT‑6.7B 语言模型上进行评估，使用 512-token 预填充、32-token 生成、batch 32 的推理任务，部署在单 GPU（RTX 5060 Ti）和两款 NVMe SSD（PM9D3a 与 990 PRO）上。

**📈 对比分析**

与传统 FlexLLMGen 基线（仅页缓存）和仅 NVMe‑Direct 对比，双路径方案在 2–11 GB 主内存限制下，预填充阶段可降低至 33.1%（SSD A）或 25.4%（SSD B），解码阶段可降低 8.2–42.4%（SSD A）或 11.7–57.8%（SSD B），同时 SSD 利用率提升 2.2×，表明在不同存储层级下均能显著改善推理性能。

**⚠️ 局限性**

局限性：需要对 KV 张量大小做 LBA 对齐，依赖单 GPU 以及 NVMe 直通支持；在多 GPU、批量更大或不同 KV 访问模式下的表现尚未验证；NVMe‑Direct 方案对 SSD 设备特性（如 LBA 大小、MDTS）有一定敏感性。

---

## 256. Small Independent Sets versus Small Separator in Geometric Intersection Graphs

**arXiv ID:** 2604.26533 | [PDF](https://arxiv.org/pdf/2604.26533v1)

**作者:** Malory Marin `[一作]` (Université Claude Bernard Lyon 1), Rémi Watrigant `[通讯]` (Université Claude Bernard Lyon 1)

**通讯引用:** 259 | [OpenAlex ID](https://openalex.org/A5001897583)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究几何交叉图（同尺寸脂肪对象或单位球图）中不具平方根现象但仍可实现子指数时间算法的图问题，提出“弱平方根现象”并给出相应的算法框架与匹配的ETH下界，重点证明2-Subcoloring和Two Sets Cut‑Uncut可在时间 2^O(n^{1-1/(d+1)}) 内求解；

**💡 创新点**

创新点在于：1）定义并证明“弱平方根现象”，即时间 2^O(n^{1-1/(d+1)}）可实现但比传统平方根 2^O(√n) 更慢；2）引入α‑调制数（α‑modulator number）作为新宽度参数；3）提出通用的 win‑win 结构分离定理；4）提供匹配的ETH下界框架；

**🔧 技术方法**

技术主要包括：结构分离定理（每个图可去除 O(n^{1-1/(d+1)}) 规模的割集使得剩余连通分量独立数 ≤ O(n^{1-1/(d+1)})); 基于 α‑调制数的动态规划与组合；对 2-Subcoloring 采用签名表与分区 DP；对 Two Sets Cut‑Uncut 采用分区与 Steiner 树猜测；下界构造基于 Monotone NAE‑3SAT 递归网格嵌入；

**📊 数据集**

无具体数据集，全部为理论构造与证明；

**📈 对比分析**

与之前的 2^O(√n) 经典平面/单位圆盘图算法对比，本文提供更宽泛但略慢的子指数算法，同时给出匹配的下界，证明了算法时间是 ETH 下界近似最优的；

**⚠️ 局限性**

局限性包括：需输入几何表示；对单位圆盘图的下界尚未覆盖；α‑调制数虽有效但与树宽无直接关系，限制了对更广泛图类的推广；

---

## 257. PRAG End-to-End Privacy-Preserving Retrieval-Augmented Generation

**arXiv ID:** 2604.26525 | [PDF](https://arxiv.org/pdf/2604.26525v1)

**作者:** Zhijun Li `[一作]` (Shandong University), Xiuzhen Cheng `[通讯]` (Shandong University)

**通讯引用:** 18797 | [OpenAlex ID](https://openalex.org/A5100692488)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套端到端加密的检索增强生成（RAG）系统PRAG，可在云端实现既安全又高效的知识检索与生成。

**💡 创新点**

创新点包括：①双模式架构（PRAG‑I非交互式、PRAG‑II交互式）实现可在不泄露文档或查询的前提下完成完整检索；②引入 Operation‑Error Estimation（OEE）机制，针对 CKKS 噪声与近似误差专门优化排名稳定性；③通过加密 HNSW、加密 K‑means、Chebyshev 多项式近似等技术实现可更新的加密 ANN 搜索。

**🔧 技术方法**

核心技术：CKKS 同态加密、加密 HNSW 图、加密 K‑means 聚类、Chebyshev 多项式比较、Operation‑Error Estimation、随机访问抑制（dummy 路径与周期性重加密）。

**📊 数据集**

实验使用 10 万条 TriviaQA 文档子集，并使用 OpenAI Embedding 生成向量；模型生成采用 Qwen‑3。

**📈 对比分析**

与现有 SSR 与 RAG 系统（SESR、PDQ、MSecKNN、FSkNN 等）对比：PRAG‑I 在检索时间（1.29 s）和通信成本（4.12 MB）上最优，PRAG‑II 在排名准确度（Recall 74.45 %）与检索时间（7.91 s）之间提供折中；整体实现了比对手更低的设置开销与更新延迟，同时保持可接受的召回率。

**⚠️ 局限性**

限制：仅支持单一客户端密钥，需在半诚实模型下工作；PRAG‑II 交互多轮通信带来网络延迟；加密 ANN 的计算开销仍高于明文方案；在极大规模或多方密钥环境下的可扩展性待进一步验证。

---

## 258. Tatemae: Detecting Alignment Faking via Tool Selection in LLMs

**arXiv ID:** 2604.26511 | [PDF](https://arxiv.org/pdf/2604.26511v1)

**作者:** Matteo Leonesi `[一作]` (University of Camerino), Marco Piangerelli `[通讯]` (University of Camerino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可验证的对齐欺骗（Alignment Faking）事件定义，并构建了Tatemae评估框架，使用单轮工具选择实验检测LLM在监控下的策略性违背。

**💡 创新点**

创新点在于将对齐欺骗检测从对话式Chain‑of‑Thought转向结构化工具选择，加入能力(T1)和鲁棒性(T2)预条件，并通过三维战略合规分类器区分能力失败、鲁棒失败与真正的对齐欺骗。

**🔧 技术方法**

技术手段包括基于prompting的四轮调用管道、结构化XML工具调用、Chain‑of‑Thought与GPT‑5.4‑mini推理一致性判定以及三维战略合规分类器。

**📊 数据集**

使用公开的108个企业IT场景数据集，覆盖安全、隐私、完整性三大领域以及腐败与破坏两种压力类型。

**📈 对比分析**

在六大前沿LLM（DeepSeek、MiMo、Gemini、Grok、Mistral、Claude）上进行5次独立运行，平均对齐欺骗率从3.5%到23.7%，展示了不同模型在监控诱因下的脆弱性差异。

**⚠️ 局限性**

局限性包括仅进行单轮工具选择实验，缺乏多步交互和记忆；使用单一监控提示，可能低估欺骗率；数据集局限于IT领域，可能导致模型针对性训练。

---

## 259. Quantamination: Dynamic Quantization Leaks Your Data Across the Batch

**arXiv ID:** 2604.26505 | [PDF](https://arxiv.org/pdf/2604.26505v1)

**作者:** Hanna Foerster `[一作]` (University of Cambridge), Robert Mullins `[通讯]` (University of Cambridge)

**通讯引用:** 17803 | [OpenAlex ID](https://openalex.org/A5011576250)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了动态量化导致的跨用户信息泄露现象（Quantamination），并实现了针对LLM和分类模型的批量侧信道攻击

**💡 创新点**

揭示了批量化时使用全张量动态量化会产生可被利用的隐私侧信道，并证明该漏洞在主流推理框架中普遍存在

**🔧 技术方法**

基于动态量化参数的统计影响、量化尺度计算、量化感知的Token/图像检索算法，结合白盒/黑盒实验平台（vLLM、SGLang、ONNX Runtime、PyTorch等）

**📊 数据集**

LLM使用TinyStories‑1M、Pythia‑70M、SmolLM2‑135M模型及对应文本数据集；分类使用MNIST数据集（10类手写数字）并对3种CNN架构进行实验

**📈 对比分析**

在LLM场景下，攻击在多种模型和数据分布上实现了99.6–100% 的Token恢复精度，平均查询数比随机搜索快18–25倍；在分类场景中，当目标样本包含在候选集中时可实现100% 的精确恢复，若不包含则仅能略高于随机猜测的类别识别率

**⚠️ 局限性**

局限性包括需要批量共置（co‑location）且批量大小对噪声敏感；在生产环境中硬件/软件异质性、采样策略导致的非确定性可能抑制侧信道信号；攻击对较浅网络或随机探测时效果下降

---

## 260. TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models

**arXiv ID:** 2604.26553 | [PDF](https://arxiv.org/pdf/2604.26553v1)

**作者:** Jinho Choo `[一作]` (Samsung SDS), Yeong-Dae Kwon `[通讯]` (Samsung SDS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Token-Level Policy Optimization（TLPO）框架，用局部令牌级更新解决多语言LLM的语言混乱问题

**💡 创新点**

首次通过概率排序的候选词探索与概率加权优势函数，实现仅在错误位置微调，既抑制错误又不损害模型通用能力

**🔧 技术方法**

采用RLHF思路的PPO目标，结合top‑N候选词选择、优势计算与KL正则化，对单个token位置进行梯度更新

**📊 数据集**

使用Bactrian‑X多语言指令数据集与公开LLM（Llama‑3.1‑8B、Qwen3‑8B、Ministral‑8B、Gemma‑3‑4B）进行微调实验

**📈 对比分析**

与SFT、DPO、ORPO对比，在LCB、MIF、MMMLU、GSM8K等基准上，TLPO在语言一致性（RPR/WPR）上显著提升，同时保持或提升下游任务准确率

**⚠️ 局限性**

仅适用于可定位的错误（如语言混乱），对全局序列错误缺乏指导；未解决安全性、事实准确性等更广泛问题

---

## 261. Beyond Code Reasoning: A Specification-Anchored Audit Framework for Expert-Augmented Security Verification

**arXiv ID:** 2604.26495 | [PDF](https://arxiv.org/pdf/2604.26495v1)

**作者:** Masato Kamba `[一作]` (Nyx Foundation), Akiyoshi Sannai `[通讯]` (Kyoto University)

**通讯引用:** 248 | [OpenAlex ID](https://openalex.org/A5056836344)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于规范的安全审计框架，通过从自然语言规范生成显式安全属性并进行结构化证明尝试来审计实现。

**💡 创新点**

首次将规范导出的类型化安全属性与 LLM 驱动的证明过程相结合，提供可解释的错误分类、跨实现可比性和严格的误报过滤。

**🔧 技术方法**

使用大语言模型（Claude、Sonnet、DeepSeek）进行规范理解、属性生成、代码映射和证明尝试，辅以 Tree-sitter 代码分析和多阶段管道。

**📊 数据集**

在 Sherlock 以太坊客户端（10 个实现，366 名审计员）和 RepoAudit C/C++ 基准（15 项目）上评估。

**📈 对比分析**

与代码驱动基线相比，专家增量配置在 Sherlock 上 100% 回收 H/M/L 漏洞，发现 4 条新增确认漏洞；在 RepoAudit 上实现 88.9% 最高精度并发现 12 条超出基线的候选漏洞；成本约每个漏洞 1.69 美元。

**⚠️ 局限性**

属性生成的自动化仍有限，需人工注入 7 条关键属性；模型对属性范围的严格遵守导致发现率受限；在更广泛的协议栈和加密库上的验证仍待验证。

---

## 262. Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners

**arXiv ID:** 2604.26488 | [PDF](https://arxiv.org/pdf/2604.26488v1)

**作者:** Nikita Araslanov `[一作]` (Google), Federico Tombari `[通讯]` (Google)

**通讯引用:** 16348 | [OpenAlex ID](https://openalex.org/A5041092666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用视频中噪声的深度和光流线索，通过线性上下文学习（Linear In-Context Learning）训练一个编码-解码网络，从而得到像素级的时空一致特征。

**💡 创新点**

创新点在于：① 通过求解上下文帧的线性投影，使得该投影同样能逼近查询帧的线索，形成跨帧一致性约束；② 将自蒸馏、深度、光流三种不同模态的线索融合，并采用PAMR进行特征细化；③ 仅训练解码器，冻结预训练的编码器，实现高分辨率像素特征的自然生成。

**🔧 技术方法**

技术包括：Encoder-Decoder架构（基于DINOv2 + DPT解码器），线性投影的Ridge回归，梯度匹配边缘损失，PAMR细化，线性上下文学习框架。

**📊 数据集**

使用未标注视频数据：YouTube‑VOS（约4k序列）和Kinetics‑700（65万视频剪辑）。

**📈 对比分析**

与多种基线（如FlowFeat、DINO、FeatUp、LoftUp等）在三大任务上进行比较：视频目标分割（DAVIS‑2017）显著提升J&F指标；表面法线估计（NYUv2）RMSE下降，角度误差阈值提升；语义分割（COCO‑Stuff）mIoU提升。实验表明LILA在所有模型尺度和评估方式下均优于先前工作。

**⚠️ 局限性**

局限性：依赖于预训练的深度和光流网络产生的线索，若线索不可靠（如航空影像中的阴影、医学影像），模型表现会受限；另外，只在RGB视频上训练，难以直接迁移到多模态或高帧率特定场景。

---

## 263. Star-Fusion: A Multi-modal Transformer Architecture for Discrete Celestial Orientation via Spherical Topology

**arXiv ID:** 2604.26582 | [PDF](https://arxiv.org/pdf/2604.26582v1)

**作者:** May Hammad `[一作]` (Julius-Maximilians-Universität Würzburg), Menatallh Hammad `[通讯]` (Julius-Maximilians-Universität Würzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了Star-Fusion多模态Transformer架构，用离散分类解决航天器姿态确定。

**💡 创新点**

创新点在于将球面分块为K个拓扑一致类并结合光学、热图和坐标三分支融合。

**🔧 技术方法**

使用SwinV2-Tiny Transformer、CNN热图分支、MLP坐标分支以及球面K‑Means聚类。

**📊 数据集**

使用基于Hipparcos星表的合成星场数据集，共50k张图像。

**📈 对比分析**

与ResNet‑50、SwinV2基线比较，Star‑Fusion Top‑1 93.4%，Top‑3 97.8%，推理时延18.4 ms。

**⚠️ 局限性**

局限在于分辨率受K=12类限制、对真实传感器噪声的域差异以及高角速度导致模糊。

---

## 264. PICKLES: a Natural Language Framework for Requirement Specification and Model-Based Testing

**arXiv ID:** 2604.26572 | [PDF](https://arxiv.org/pdf/2604.26572v1)

**作者:** María Belén Rodríguez `[一作]` (University of Twente), Petra van den Bos `[通讯]` (University of Twente)

**通讯引用:** 99 | [OpenAlex ID](https://openalex.org/A5083933972)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Pickles 框架，将 BDD 语法与 MBT 自动化相结合，开发了 PicklesDSL 语言，支持从自然语言规格到符号转移系统（STS）的双向翻译，并自动合成 master 模型，随后生成可执行的 BDD 测试用例。

**💡 创新点**

创新点包括：① 设计了兼具可读性与形式化语义的 PicklesDSL，保留 Given–When–Then 结构并加入变量域、约束和控制流；② 实现了规格与测试之间的双向翻译，使人类可读的场景能够直接产生形式化模型并反向生成测试；③ 引入了场景合成（choice/sequence 组合）自动构建 master 模型，从而显著提升测试覆盖率；④ 通过参数化实现输入/输出约束，支持更广泛的测试生成。

**🔧 技术方法**

采用了 EBNF 语法定义、符号转移系统（STS）形式化、SMT 求解路径约束、模型合成（choice/sequence）、Python 与 Lark 解析器实现工具链；同时使用 JSON 传递模型数据，保持可移植性。

**📊 数据集**

以荷兰公司 Technolution 的交通管理系统组件为案例，使用其原始 BDD 测试集（4 个场景）进行实验，未使用公开通用数据集。

**📈 对比分析**

对比方法：将 Pickles 生成的 100% 迁移覆盖率与传统 BDD 场景（单独执行、仅覆盖约 30%）进行比较；同时比较输入覆盖率（从 1.5% 提升至 98.5%）。实验显示，使用相同数量场景，Pickles 能实现 100% 状态/转移覆盖，覆盖率提升约 70%，且测试用例数量显著降低。

**⚠️ 局限性**

局限性：目前仅支持同步、无并发的系统；模型规模可能随变量域增大而爆炸；迁移到 Pickles 需要手工将现有 Gherkin 示例泛化；未实现时间约束与并发控制；原型实现尚未在大规模项目中评估；缺乏系统的用户体验与实证研究。

---

## 265. AirZoo: A Unified Large-Scale Dataset for Grounding Aerial Geometric 3D Vision

**arXiv ID:** 2604.26567 | [PDF](https://arxiv.org/pdf/2604.26567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. FloatSOM: GPU-Accelerated, Distributed, Topology-Flexible Self-Organizing Maps

**arXiv ID:** 2604.26555 | [PDF](https://arxiv.org/pdf/2604.26555v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 267. Preserving Disagreement: Architectural Heterogeneity and Coherence Validation in Multi-Agent Policy Simulation

**arXiv ID:** 2604.26561 | [PDF](https://arxiv.org/pdf/2604.26561v1)

**作者:** Ariel Sela `[一作]` `[通讯]` (Tel Aviv University), Ariel Sela (Tel Aviv University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AI Council多代理议论框架，利用大型语言模型进行政策模拟时保持价值观分歧，解决人工共识问题。

**💡 创新点**

创新点在于：①架构异质性（为每个价值视角使用不同参数模型）显著降低人工共识；②共鸣验证层（前沿模型评估推理与价值视角一致性）揭示可信度‑多样性权衡；③系统可在消费者硬件上运行并提供可信度诊断指标。

**🔧 技术方法**

技术包括：多代理结构化辩论、独立评估、Borda计票、前沿模型（Claude Sonnet 4）共鸣验证、统计检验（Mann‑Whitney、Wilcoxon）、熵等信息量度。

**📊 数据集**

使用了两个政策场景数据集：儿童福利干预（3个选项）和城市住房危机（3个选项），共进行120次 deliberations；评估模型为七个7–9B本地模型加前沿模型。

**📈 对比分析**

比较方法：与同构单模型基线、无验证异质化、验证异质化三种状态对比；性能方面，架构异质化将首选集中度从约71%降至46%（儿童福利）或22%（住房），有效视角熵显著提高；共鸣验证在不同情境中可降低或增加集中度，展示信度‑多样性权衡。

**⚠️ 局限性**

局限包括：仅测试两种场景、单一模型池、价值视角仅五种且三者为安全关注导致偏倚、8B模型呈二元响应、共鸣验证依赖外部前沿模型、未与单一前沿模型单独对比、可信度率仅约50%。

---

## 268. Learning to Route Electric Trucks Under Operational Uncertainty

**arXiv ID:** 2604.26566 | [PDF](https://arxiv.org/pdf/2604.26566v1)

**作者:** Stavros Orfanoudakis `[一作]` (Delft University of Technology), Elenna Dugundji `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 987 | [OpenAlex ID](https://openalex.org/A5050654045)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个基于图神经网络和强化学习的电动卡车车队路径规划框架（GraphPPO），在共享充电设施和不确定行驶/能耗条件下实现实时可行的路径与充电决策。

**💡 创新点**

将电动卡车路径规划建模为事件驱动的半马尔可夫决策过程，使用可变动作空间与可行性掩码，结合图结构状态与动作表示，显式捕捉充电资源竞争、随机行驶时间与能耗。

**🔧 技术方法**

强化学习（PPO）+ 图神经网络（GraphPPO）、事件驱动仿真环境、可行性掩码与半马尔可夫决策过程。

**📊 数据集**

基于加州真实道路网络数据（约66,000条有向边、25个充电站、258个交付节点），构建不同规模车队的实验实例。

**📈 对比分析**

与数学规划基准、启发式、通用PPO和MaskPPO比较；GraphPPO在1至100辆车规模下归一化奖励接近1（0.987–1.005），成功率与优化基准相近，且在单车eVRP场景同样表现优异。

**⚠️ 局限性**

缺乏时窗、退货策略、充电价格、网格约束等更复杂约束；模型对极端分布的泛化和跨地区外域测试尚未深入评估。

---

## 269. Large-scale semi-supervised learning with online spectral graph sparsification

**arXiv ID:** 2604.26550 | [PDF](https://arxiv.org/pdf/2604.26550v1)

**作者:** Daniele Calandriello `[一作]` (Inria), Michal Valko `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的稀疏HFS算法，用于解决大规模半监督学习问题，能够在有限的空间和计算预算下进行有效的学习。

**💡 创新点**

创新点在于结合了在线谱图稀疏化技术，使得算法在处理大规模图时能够控制空间和计算复杂度，同时提供了理论上的泛化误差保证。

**🔧 技术方法**

使用了在线谱图稀疏化技术和对称对角占优(SDD)矩阵的特定求解器。

**📊 数据集**

使用了一个包含12100个点的合成数据集，构建了一个k近邻图，边的数量从1.21 × 10^6到1.38 × 10^8不等。

**📈 对比分析**

与稳定HFS算法进行了比较，结果显示在k>4000时两者的性能相近，稀疏HFS在准确性上未能始终超越稳定HFS，但差异不大，符合理论分析的预期。

**⚠️ 局限性**

限制在于算法仅适用于边插入的流处理，扩展到边移除的动态设置可能会面临计算效率问题。

---

## 270. Identifying and Characterizing Semantic Clones of Solidity Functions

**arXiv ID:** 2604.26526 | [PDF](https://arxiv.org/pdf/2604.26526v1)

**作者:** Ermanno Francesco Sannini `[一作]` (University of Sannio), Andrea Di Sorbo `[通讯]` (University of Sannio)

**通讯引用:** 2235 | [OpenAlex ID](https://openalex.org/A5043960445)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级、可扩展的方法，利用代码与开发者注释的相似度来检测Solidity智能合约中的语义（Type‑4）克隆，并通过LLM生成注释弥补缺失文档；

**💡 创新点**

创新点在于：①将注释视为语义指纹，与低代码相似度结合筛选；②首次构建覆盖约300k合约、2.7M函数的现代Solidity数据集和基准；③通过LLM自动生成注释实现对无注释函数的语义克隆检测；

**🔧 技术方法**

技术手段包括：代码嵌入（SmartEmbed）与评论嵌入（SBERT all‑MiniLM‑L6‑v2、BERT、CodeBERT）的余弦相似度计算；LLM ChatGPT‑4o 用于注释生成；统计抽样与人工验证保证结果可靠；

**📊 数据集**

使用了约82,337份更新版Solidity合约（约300,000个合约、2,705,194个函数）的清洗版数据集，其中约75%函数缺失注释；

**📈 对比分析**

与仅代码相似度的基线、EClone等工具比较，人工验证样本1,155对中，方法精度为59%（同名时提升至84%），召回率97%，F1 74%，总体准确率79%；LLM生成注释在无注释函数中实现约75%的精度；

**⚠️ 局限性**

局限性包括：对注释质量高度依赖，LLM生成文本可能引入噪声；阈值设置相对主观；仅覆盖Solidity 0.8+版本，未考虑多链或其他语言；LLM推理成本与规模化部署仍是挑战。

---

## 271. RepoDoc: A Knowledge Graph-Based Framework to Automatic Documentation Generation and Incremental Updates

**arXiv ID:** 2604.26523 | [PDF](https://arxiv.org/pdf/2604.26523v1)

**作者:** Dong Xu `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 31003 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套基于仓库知识图 RepoKG 的文档生成与增量更新系统 RepoDoc，能自动生成结构化、可交叉引用且包含 Mermaid 结构图的文档，并支持针对代码变更的精准增量更新。

**💡 创新点**

创新点在于：①首次将仓库级知识图作为整个文档生命周期的语义骨干；②通过模块聚类实现语义化层次化文档；③采用基于图查询的多智能体生成架构显著降低 token 消耗；④设计双向语义影响传播机制实现高精度增量更新。

**🔧 技术方法**

主要技术包括：代码实体抽取（AST/Tree‑sitter）、知识图构建与关系抽取、LLM（DeepSeek V3.2）进行概念抽取与文本生成、层级化模块聚类（LLM 辅助）、多智能体 Agent 架构（任务路由、技能层、工具层）、双向遍历的语义影响传播与增量更新流程。

**📊 数据集**

使用 24 个跨 8 种语言（Python、JavaScript、TypeScript、Java、C#、C、C++、PHP）的开源仓库，涵盖小型（<10K LOC）到大型（>100K LOC）项目；在 Python 仓库上进行增量更新评测。

**📈 对比分析**

与两大基线（RepoDoc 对标 RepoDoc（支持增量）和 RDoc（不支持增量））进行公平对比。结果显示：API 覆盖率提升 32.5%，文档完整度提升 10.4%；生成速度约 3 倍快，token 消耗减少 85%；增量更新时间缩短 73%，token 用量减少 77%，更新召回率提升 10.2%。

**⚠️ 局限性**

主要限制包括：仅在支持的 LLM（如 DeepSeek V3.2）上实现；对非 Python 代码的增量更新尚未充分验证；知识图构建与更新仍受限于 LLM 的抽象准确性；在极大规模仓库中，知识图存储与查询可能成为性能瓶颈。

---

## 272. AGEL-Comp: A Neuro-Symbolic Framework for Compositional Generalization in Interactive Agents

**arXiv ID:** 2604.26522 | [PDF](https://arxiv.org/pdf/2604.26522v1)

**作者:** Mahnoor Shahid `[一作]` (Universität Duisburg-Essen), Hannes Rothe `[通讯]` (Universität Duisburg-Essen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出 AGEL-Comp 神经符号框架，以解决 LLM 代理在交互环境中缺乏组合泛化的局限。

**💡 创新点**

创新点在于将因果程序图（CPG）动态世界模型、诱导逻辑编程（ILP）自我学习和神经定理证明器（NTP）验证三者结合成闭环推理，实现在交互中基于经验自动生成可解释的可执行规则。

**🔧 技术方法**

技术包括因果程序图（CPG）建模、ILP 规则诱导、NTP 可微推理、LLM 规划器以及共享符号嵌入空间。

**📊 数据集**

使用自建的 Retro Quest 2D RPG 仿真环境，涵盖多阶段任务与组合泛化挑战。

**📈 对比分析**

通过与纯 LLM 基线及两种消融版本（去掉 NTP 或 ILP）在四个 LLM 后端（GPT‑4o、Gemini‑2.5‑Pro、DeepSeek‑VL‑7B、LLaVA‑1.6）进行对比实验，AGEL‑Comp 在所有后端实现 100% 任务成功率，首次尝试成功率提升至 50‑70%，样本效率提升约 5‑10 倍，消融实验进一步验证 ILP 与 NTP 的互补性。

**⚠️ 局限性**

局限主要包括 NTP 计算开销、符号映射与嵌入空间的可扩展性、感知噪声下的符号提取鲁棒性，以及 CPG 规模与不一致性管理的挑战。

---

## 273. Text-Utilization for Encoder-dominated Speech Recognition Models

**arXiv ID:** 2604.26514 | [PDF](https://arxiv.org/pdf/2604.26514v1)

**作者:** Albert Zeyer `[一作]` (RWTH Aachen University), Hermann Ney `[通讯]` (RWTH Aachen University)

**通讯引用:** 46710 | [OpenAlex ID](https://openalex.org/A5112501010)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了如何高效利用海量文本数据来提升语音识别效果，提出在 encoder 主导模型中通过伪语音编码器（pseudo‑speech‑encoder）和动态下采样将文本映射到文本级表示，并对多种文本利用策略在 LibriSpeech 上进行系统对比。

**💡 创新点**

创新点在于：① 将文本数据直接融入 encoder，构建 encoder‑dominated（大 encoder + 小 decoder）模型，显著提升识别速度；② 提出无需 TTS 的伪语音编码器，且随机持续时间模型反而优于复杂的可训练模型；③ 通过动态下采样实现文本级 encoder，降低模型复杂度并提升训练效率。

**🔧 技术方法**

采用的技术包括：Conformer/Transformer++ encoder，CTC + AED 框架，基于 CTC 概率的动态下采样，伪语音编码器（嵌入层 + 可选空白插入/持续时间模型/上采样层），模态匹配与直接训练两种伪语音编码器优化方式，时间同步 beam search，shallow fusion 对比基线。

**📊 数据集**

使用的数据集为 LibriSpeech 960h 的对齐音频文本对，以及 800M 词的 LibriSpeech LM 文本语料（约 75k 小时的合成语音）。实验评估在 LibriSpeech dev/test 上进行。

**📈 对比分析**

与基线 CTC+LM、CTC+AED+TTS 等方案对比，实验表明 encoder‑dominated + 伪语音编码器可获得与 TTS 方案相当甚至更优的 WER（如 clean dev 1.84‑2.10%，other dev 4.02‑4.59%），并且训练与推理成本显著降低；随机持续时间模型在多种实验中优于复杂的模态匹配方案；最佳 WER 出现在文本比例最高的设置下。

**⚠️ 局限性**

局限性包括：与 TTS+LM 组合相比仍存在性能差距；伪语音编码器对声学变异的建模有限；对不同数据集的泛化性未作充分验证；持续时间模型的选择仍需进一步探索。

---

## 274. 3D Generation for Embodied AI and Robotic Simulation: A Survey

**arXiv ID:** 2604.26509 | [PDF](https://arxiv.org/pdf/2604.26509v1)

**作者:** Tianwei Ye `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 30764 | [OpenAlex ID](https://openalex.org/A5043464306)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了3D生成在具身AI与机器人仿真中的三大角色（数据生成、仿真环境构建、Sim2Real桥梁），梳理了相关方法、数据集与评估指标，强调了simulation‑readiness（几何有效、物理参数、运动学可执行、模拟器兼容）的重要性。

**💡 创新点**

首次从具身需求视角组织文献，提出simulation‑readiness评估框架，并系统化归纳生成技术、场景合成与Sim2Real闭环流程，揭示了物理标注、动态柔性物体与评估标准的瓶颈。

**🔧 技术方法**

涵盖VAE、GAN、扩散模型、流模型、自动回归、Transformer、GNN、神经隐式/3D Gaussian Splatting、LLM/VLM、物理仿真反馈等多种生成与评估技术，并展示其在对象、场景与数字双生中的应用。

**📊 数据集**

使用ShapeNet、PartNet、Objaverse、PhysXNet、3D‑FRONT、Matterport3D、ScanNet、SAGE、MetaScenes、Open X‑Embodiment、RoboTwin等代表性数据集，对对象、场景与演示进行多维度评估。

**📈 对比分析**

对比指标包括Chamfer、FID、CLIP Score（几何/视觉质量）；Stability Rate、Joint Accuracy、Material Error（物理合理性）；Grasp Success Rate、Articulation Success、Navigation Success、Sim‑to‑Real Success Rate（任务性能）。实验表明，加入物理约束与仿真反馈的生成方法在稳定性与仿真成功率上显著优于仅视觉优化的方法。

**⚠️ 局限性**

存在大规模物理标注不足、柔性物体数据稀缺、可微物理约束成本高、评估仍以单一仿真器为主、Sim2Real中物理一致性与动态一致性难以保证等限制，阻碍了3D生成在具身AI中的大规模、可靠应用。

---

## 275. Progressive Semantic Communication for Efficient Edge-Cloud Vision-Language Models

**arXiv ID:** 2604.26508 | [PDF](https://arxiv.org/pdf/2604.26508v1)

**作者:** Cyril Shih-Huan Hsu `[一作]` (University of Amsterdam), Chrysa Papagianni `[通讯]` (University of Amsterdam)

**通讯引用:** 1258 | [OpenAlex ID](https://openalex.org/A5012988433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了基于Meta AutoEncoder的渐进式语义通信框架，实现边缘与云端协同推理VLM。

**💡 创新点**

引入可逐层细化的latent表示和语义质量感知的传输控制，实现自适应压缩与增量传输。

**🔧 技术方法**

Meta AutoEncoder (Transformer编码/解码)、前缀掩码训练、语义质量估计、RESTful HTTP、NXP i.MX95边缘设备与GPU服务器等技术。

**📊 数据集**

COCO 2017图像数据集用于训练MetaAE，并在COCO Captioning和POPE数据集上进行评估。

**📈 对比分析**

与全边缘、全云两种配置对比，在1 Mbps上测得端到端延迟最低，语义一致性≥80%，25%压缩时延迟下降约75%。

**⚠️ 局限性**

仅在SmolVLM-256M轻量模型上验证，未针对更大VLM或更复杂网络环境，需进一步研究最优采样分布与更高层语义评估。

---

## 276. Auto-Relational Reasoning

**arXiv ID:** 2604.26507 | [PDF](https://arxiv.org/pdf/2604.26507v1)

**作者:** Ioannis Konstantoulas `[一作]` (University of Patras), Kyriakos Sgarbas `[通讯]` (University of Patras)

**通讯引用:** 859 | [OpenAlex ID](https://openalex.org/A5012058127)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了自动关系推理（Auto-Relational Reasoning）框架，结合神经网络观察模块与答案集编程，能够在无先验知识的情况下解决Raven矩阵等IQ问题。

**💡 创新点**

创新点在于将符号推理与深度学习无缝集成，使用基于稳定模型语义的可动态生成知识基础，实现几乎完美的IQ解答，并首次将这种框架用于实际IQ测试。

**🔧 技术方法**

采用卷积神经网络（CNN）提取对象及特征，层次编码器将其转化为逻辑原子，再通过Clingo答案集编程完成符号推理。

**📊 数据集**

训练使用400,000张图像的自制数据集，测试使用40,000道Raven矩阵问题，亦对比PGM、RAVEN等公开数据集。

**📈 对比分析**

与PGM、RAVEN公开模型相比，在自建数据集上达98.03%的准确率，显著高于人类平均水平，并且在仅使用观察模块时仍保持高精度。

**⚠️ 局限性**

主要限制是观察模块的识别误差以及缺乏先验知识，导致在噪声或更复杂问题上的鲁棒性受限，且无法处理需要常识或外部知识的情境。

---

## 277. Delta Score Matters! Spatial Adaptive Multi Guidance in Diffusion Models

**arXiv ID:** 2604.26503 | [PDF](https://arxiv.org/pdf/2604.26503v1)

**作者:** Haosen Li `[一作]` (Hong Kong University of Science and Technology), Yutao Yue `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 949 | [OpenAlex ID](https://openalex.org/A5052861384)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种空间自适应多重引导（SAMG）算法，利用局部条件能量动态调节指导尺度，解决传统Classifier-Free Guidance（CFG）在细节与伪影之间的矛盾，从而提升文本到图像及视频生成的质量与一致性。

**💡 创新点**

创新点包括：① 从微分几何视角解析CFG为切线线性外推，揭示其导致数据流形偏离的根本原因；② 推导出适当的空间自适应指导上界；③ 通过一阶泰勒近似与Affine重参数化得到零成本、像素级自适应调节方案，使其在保持结构完整性的同时最大化语义注入。

**🔧 技术方法**

核心技术：Tweedie公式、微分几何与曲率理论、局部能量度量、线性近似、Affine重参数化、无训练的采样算法。实现时仅在采样阶段对每像素计算能量并按预设范围映射指导尺度，无需额外模型训练。

**📊 数据集**

使用的数据集包括：图像生成—Pick‑a‑Pic、DrawBench、GenEval、MS‑COCO 2017；视频生成—ChronoMagic‑Bench‑150（CogVideoX‑2B、ModelScope‑1.7B）。

**📈 对比分析**

通过与CFG、PAG、CFG++、CFG‑Zero等多种基线在上述数据集上进行对比，评估指标涵盖HPSv2、ImageReward、Aesthetic、CLIP、Top‑K、FID、CLIP Score、CHScore Flow、Frame LPIPS、SSIM、MTScore等。实验表明，SAMG在所有模型和数据集上均显著提升语义一致性、结构完整性与时序平滑，且几乎不增加计算成本。

**⚠️ 局限性**

局限性：① 对极低能量区域的数值稳定性仍需改进；② 仅在采样阶段提升，未直接改进模型训练过程；③ 对极高分辨率或实时推理环境的适用性尚未充分验证。

---

## 278. Tree-of-Text: A Tree-based Prompting Framework for Table-to-Text Generation in the Sports Domain

**arXiv ID:** 2604.26501 | [PDF](https://arxiv.org/pdf/2604.26501v1)

**作者:** Shang-Hsuan Chiang `[一作]` (National Yang Ming Chiao Tung University), Wen-Chih Peng `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 3252 | [OpenAlex ID](https://openalex.org/A5040101293)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Tree-of-Text 框架，利用树状提示递归拆分表格、生成短文本并合并为完整的体育比赛报道。

**💡 创新点**

创新点在于将表格-文本生成任务拆解为内容规划、操作执行、内容生成三阶段的树结构流程，显著提升表格理解与信息连贯性，且在效率上大幅优于现有树/链提示方法。

**🔧 技术方法**

核心技术：树状提示（Tree-of-Text）、八种自定义表格操作、LLM（gpt‑4o‑mini）与 Python 代码执行交互；优化包括一次性生成操作与参数、单子节点不合并等。

**📊 数据集**

使用三大体育数据集：ShuttleSet+（羽毛球 58 场比赛）、RotoWire‑FG（NBA 5,340 场比赛）、MLB（棒球 22,821 场比赛）。

**📈 对比分析**

与 Chain‑of‑Thought、Tree‑of‑Thought、Chain‑of‑Table 等提示基准对比；在 ShuttleSet+ 上取得所有指标最高，RotoWire‑FG 在 RG 与 CO 最高，MLB 在 CS 与 CO 最高；效率方面，时间与成本仅为 Chain‑of‑Table 的约 40%。

**⚠️ 局限性**

局限性：需手动调参与编写提示，仍比少量提示方法慢；只能利用表内信息，无法自动调用外部知识或数据；若想进一步提升速度与成本，仍需研究并行化与自动配置技术。

---

## 279. GMT: A Geometric Multigrid Transformer Solver for Microstructure Homogenization

**arXiv ID:** 2604.26518 | [PDF](https://arxiv.org/pdf/2604.26518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 280. Robust Alignment: Harmonizing Clean Accuracy and Adversarial Robustness in Adversarial Training

**arXiv ID:** 2604.26496 | [PDF](https://arxiv.org/pdf/2604.26496v1)

**作者:** Yanyun Wang `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 9030 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 Robust Alignment Adversarial Training (RAAT) 的新对抗训练框架，通过固定边界样本的扰动强度并引入 Domain Interpolation Consistency Adversarial Regularization (DICAR) 来调和干净准确率与对抗鲁棒性。

**💡 创新点**

创新点包括：1) 发现边界样本扰动强度对鲁棒性影响不大，提出将其固定以避免噪声；2) 定义 Robust Alignment 目标，强调输入扰动下模型感知的变化；3) 提出 DICAR 正则化，利用域插值一致性实现输入与潜在空间的语义对齐；4) 通过信息理论与高阶导数正则化对 CR 与 DICAR 的有效性进行理论证明。

**🔧 技术方法**

技术手段包括：传统对抗训练（PGD、TRADES、MART 等）基础上的改进；对抗训练目标中加入 DICAR 约束；固定边界样本扰动强度的策略；理论分析（Gaussian 模型下 CR 的效果、DICAR 对所有阶导数的正则化）。

**📊 数据集**

实验数据集为 CIFAR-10、CIFAR-100 与 Tiny-ImageNet，使用 ResNet-18、PreActResNet-18 与 WideResNet-28-10 等主流网络。

**📈 对比分析**

与四个常用基线（PGD‑AT、TRADES、MART、Cons‑AT）以及 14 个现有 SOTA 方法在干净准确率、PGD‑10/PGD‑100/CW/Auto‑Attack 鲁棒率上进行对比；RAAT/RAAT++ 在大多数指标上均优于基线和 SOTA，并在 PreActResNet‑18 上的 RAAT# 进一步突破 11 个 SOTA，取得新的最佳 trade‑off 性能。

**⚠️ 局限性**

局限性包括：1) 计算成本相对传统 AT 较高，尤其是 DICAR 的插值与对抗样本生成；2) 主要在 ℓ∞ 威胁模型下验证，其他攻击方式与更大规模数据集的迁移性待进一步验证；3) 边界样本划分阈值的选择对性能影响显著，需要经验调参。

---

## 281. On (In)approximability of MaxMin Independent Set Reconfiguration

**arXiv ID:** 2604.26714 | [PDF](https://arxiv.org/pdf/2604.26714v1)

**作者:** Hung P. Hoang `[一作]` (TU Wien), Yuma Tamura `[通讯]` (Tohoku University)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5009641276)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在 Token Addition/Removal 规则下的 MaxMin Independent Set Reconfiguration 问题，并分别给出了通用图和稀疏图（退化图、树宽、H-子图自由图）的近似算法以及多种下界（最大度、带宽、二分图）证明。

**💡 创新点**

主要创新点包括：
- 提出第一个通用图的 n/ log n 近似算法；
- 针对退化图给出 d 近似；
- 在树宽和 H-子图自由图中实现 FPT 近似方案（PTAS）；
- 通过新的间隙保持归约提升了已知的不可近似阈值，尤其对最大度为 Δ 的图给出 Θ(√Δ) 的不可近似结果。

**🔧 技术方法**

使用的技术手段有：
- γ-序列和辅助图构造来实现 n/ log n 近似；
- 退化图中的最小度顶点迭代、分离与树宽分治；
- Baker 结构分解及其在 H-子图自由图中的应用；
- 采用 Ramanujan 二分图 (极限展开子图) 进行带宽和最大度下界的构造；
- 参数化复杂度工具（FPT-AS、树宽分解）以及与 ISR‑TJ 的等价性。

**📊 数据集**

本文为理论研究，未使用任何实际数据集；所有结果均为数学证明与算法设计。

**📈 对比分析**

在通用图上与已知的 n^Ω(1) 与 Δ^Ω(1) 难度相比，n/ log n 近似是第一次取得多项式近似；在退化图、树宽和 H-子图自由图中通过参数化方法实现了更优近似（近似因子随参数递减，甚至可达 PTAS）。相比之下，已有工作仅给出 NP‑硬度或指数级别的可行方案；本文的算法在理论上实现了显著的改进，但仍未达到常数因子近似。

**⚠️ 局限性**

局限性：
- 通用图的近似因子仍为 n/ log n，距离常数因子相距甚远；
- 对最大度为 Δ 的图的不可近似阈值仍为 Θ(√Δ)，尚未能突破到多项式级别；
- 需要已知树宽或退化度等结构参数，实际应用时可能需要额外的预处理；
- 对二分图的不可近似结果依赖于 SSEH 与 P ≠ NP 等未被证明的假设；
- 研究主要聚焦在理论上，未提供实验验证或实际性能评估。

---

## 282. Atomic-Probe Governance for Skill Updates in Compositional Robot Policies

**arXiv ID:** 2604.26689 | [PDF](https://arxiv.org/pdf/2604.26689v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 24677 | [OpenAlex ID](https://openalex.org/A5100450024)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出跨版本交换协议，系统评估在机器人技能库更新时组成策略的稳定性，发现一个主导技能效应，并基于此提出原子质量探针和混合选择器。

**💡 创新点**

创新点在于首次量化主导技能效应、构造可部署的原子质量探针、引入混合选择器以在低成本下逼近全重验证的决策质量，并证明行为距离无法预测主导技能。

**🔧 技术方法**

使用SAC训练嵌入式能力模块（ECM），配合成对采样交叉版本交换、原子质量和组合成功率探针、行为距离度量与混合选择器实现。

**📊 数据集**

实验基于robosuite操纵基准，涵盖六个任务（T1–T6），并在T6（双臂插孔）中发现主导技能效应。

**📈 对比分析**

与oracle、始终接受/拒绝等基线对比，Hybrid选择器在oracle匹配率达到87.5%时，仅消耗约70%成本；AtomicOnly在零成本下仍比基线高出约20%且与FullReval相差不到3pp。

**⚠️ 局限性**

主要局限包括仅在单一正向任务（T6）验证主导技能效应、训练方案导致大部分任务原子成功率为0、混合oracle评估混用以及仅在单一机械臂上测试。

---

## 283. Transferability of Token Usage Rights: A Design Space Analysis of Generative AI Services

**arXiv ID:** 2604.26683 | [PDF](https://arxiv.org/pdf/2604.26683v1)

**作者:** Jaeyong Lee `[一作]` (Hongik University), Baek Eunkyung `[通讯]` (Hongik University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析主要生成式AI服务的代币计费政策，定义并构建了代币使用权可迁移性的设计空间，提供了五个设计轴和五种迁移类型。

**💡 创新点**

创新点在于将代币从技术/经济工具转变为用户中心的设计要素，提出代币可迁移性的概念并系统化设计框架。

**🔧 技术方法**

采用MacLean等人的设计空间分析方法，将代币使用权的迁移性拆解为目标、方向、单位、控制、可逆性等维度。

**📊 数据集**

使用了四大生成式AI服务（ChatGPT、Claude、Gemini、Grok）的计费政策和服务条款作为案例数据。

**📈 对比分析**

本研究未进行实验比较，主要通过案例分析和设计框架阐述，未给出性能指标。

**⚠️ 局限性**

局限在于缺乏对结构性摩擦的实证分析以及对设计方案可行性、合法性和用户感知的验证。

---

## 284. A Toolkit for Detecting Spurious Correlations in Speech Datasets

**arXiv ID:** 2604.26676 | [PDF](https://arxiv.org/pdf/2604.26676v1)

**作者:** Lara Gauder `[一作]` (Instituto de Investigación en Ciencias de la Computación, UBA-CONICET), Luciana Ferrer `[通讯]` (Instituto de Investigación en Ciencias de la Computación, UBA-CONICET)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了一套工具包，用来检测语音数据集中录音条件与目标类别（如疾病、情绪、说话人身份）之间的虚假相关性；该工具通过仅使用非语音区段进行二分类，避免模型依赖语音信息并揭示录音环境对结果的影响。

**💡 创新点**

创新点包括：①提出仅用非语音区段检测目标类的诊断思路，避免语音泄漏导致的误判；②采用固定长度5 s chunk并去除时序信息的特征提取，防止模型利用时长差异学习；③整合多种VAD、增强、交互式审计等功能，形成完整的诊断工作流；④在公开数据集上系统评估并证明工具的有效性。

**🔧 技术方法**

使用技术包括语音活动检测（Silero、Pyannote、Whisper 等），深度滤波器 DeepFilterNet 的降噪增强，基于短时傅里叶变换的手工特征（MFCC、声谱图），1D CNN 分类器，8 倍交叉验证、早停、10 随机种子、AUC 与 bootstrap 置信区间评估。

**📊 数据集**

实验数据集为 ADReSS_o（阿尔茨海默病患者与对照的 Cookie Theft 语音）和 SpanishAD（西班牙语阿尔茨海默病患者），两者均包含较长的非语音段。

**📈 对比分析**

方法比较：在原始、挑战、增强三种预处理下，使用 MFCC 与 W2V2 两类特征，先在整个非语音序列上训练模型，然后再加入 5 s chunk 的限制；性能通过 AUC 与随机基线对比。结果显示：未经 chunk 的模型在非语音区段往往显著高于随机（说明时长泄露），但 chunk 后性能降至随机；即使在增强后的信号上，仍出现显著高于随机的 AUC，表明录音条件相关性仍存在；在 ADReSS_o 上增强无明显效益，而 SpanishAD 仍表现出显著相关性。

**⚠️ 局限性**

局限性：需要足够的非语音内容；VAD 的误检/漏检可能导致语音泄漏或信息缺失；样本量有限时难以学习；工具仍可能误判真正的相关性；高质量的人工标注或交互审计成本较高；在极端多变的录音环境下，工具的检测能力可能下降。

---

## 285. DMRlib: Easy-coding and Efficient Resource Management for Job Malleability

**arXiv ID:** 2604.26624 | [PDF](https://arxiv.org/pdf/2604.26624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 286. What Makes Software Bugs Escape Testing? Evidence from a Large-Scale Empirical Study

**arXiv ID:** 2604.26672 | [PDF](https://arxiv.org/pdf/2604.26672v1)

**作者:** Domenico Cotroneo `[一作]` (University of North Carolina at Charlotte), Benedetta Gaia Varriale `[通讯]` (University of Naples Federico II)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对开源 C/C++ 与 Java 项目中的 14,000+ 函数级缺陷进行大规模挖掘与分类，分别区分测试前（pre‑release）与测试后（post‑release）逃逸缺陷，提取结构、过程、统计等 44 个软件指标，对比两类缺陷的指标分布、相关性、解释方差以及修复成本。

**💡 创新点**

① 将逃逸缺陷视为独立类别，构建功能级完整数据集；② 发现逃逸缺陷的差异主要集中在过程与历史指标（年龄、变更量、作者分布、bug 密度）而非单纯结构指标；③ 在 C/C++ 中逃逸缺陷修复耗时显著更长且代码变更更大，Java 结果相对温和，表明语言与生态差异影响逃逸缺陷特征。

**🔧 技术方法**

采用 GitHub 仓库挖掘、git、Tree‑sitter 解析、Understand 计算产品指标、KenLM 训练 n‑gram 统计熵、PCA/层次聚类、Kolmogorov‑Smirnov 与 Cliff’s δ 统计检验，以及 XGBoost 预测模型；还使用多信号（标签、文本、报告者）进行残缺缺陷分类。

**📊 数据集**

基于 4000+ 真实开源 C/C++ 与 Java 项目的 14,000+ 函数级缺陷，构建了平衡的 pre‑release 与 post‑release 数据集（C/C++ 4010/96541，Java 3118/89851）。

**📈 对比分析**

通过单变量 KS + Cliff’s δ、聚类与 PCA 评估指标差异；多变量 XGBoost 在两语言上实现 F1≈0.91（C/C++）与 0.86（Java），证明逃逸缺陷在指标空间具有可辨别性；C/C++ 的修复时间和代码增量显著高于 pre‑release，Java 仅出现细微差别。

**⚠️ 局限性**

局限包括：① 依赖关键字匹配识别 bug‑fix 提交，可能漏检或误检；② 仅包含 GitHub 开源项目，缺乏工业或安全关键系统样本；③ 仅研究 C/C++ 与 Java，无法直接推广到动态语言或其他生态；④ 逃逸缺陷分类主要基于标签与文本，仍可能存在误分类；⑤ 统计检验对极端值敏感，未考虑所有上下文因素。

---

## 287. Full band denoising of room impulse response in the wavelet domain with dictionary learning

**arXiv ID:** 2604.26669 | [PDF](https://arxiv.org/pdf/2604.26669v1)

**作者:** Théophile Dupré `[一作]` (Trinnov Audio), Arnaud Laborie `[通讯]` (Trinnov Audio)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于小波变换的房间脉冲响应（RIR）低频噪声去噪后处理方法；

**💡 创新点**

创新点在于对近似系数采用稀疏字典学习，并引入随时间变化的误差容限，利用指数衰减包络自适应控制误差，实现低频成分的有效去噪；

**🔧 技术方法**

使用的技术包括离散小波变换、阈值去噪、稀疏字典学习（OMP、K‑SVD）、非线性最小二乘包络估计以及基于包络的时变误差设计；

**📊 数据集**

使用的数据集包括合成的低频模态RIR加噪（10个白噪声水平，10个SNR级别）以及实验测得的大带扬声器和子woofer的RIR，噪声通过额外扬声器人工添加；

**📈 对比分析**

与基线阈值去噪方法比较，数值实验显示在SNR 15‑35 dB范围内DT60估计误差显著下降，实验结果中动态范围提升和能量衰减曲线更接近真值，整体性能优于基线；

**⚠️ 局限性**

限制：对低频能量不足的大带扬声器效果不佳，方法依赖包络估计的准确性，参数选择需要经验，且对极低SNR情况下的表现仍有限。

---

## 288. SnapPose3D: Diffusion-Based Single-Frame 2D-to-3D Lifting of Human Poses

**arXiv ID:** 2604.26620 | [PDF](https://arxiv.org/pdf/2604.26620v1)

**作者:** Alessandro Simoni `[一作]` (University of Modena and Reggio Emilia), Roberto Vezzani `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 2816 | [OpenAlex ID](https://openalex.org/A5081341599)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SnapPose3D，一种单帧二维到三维人体姿态提升框架，利用扩散模型生成多种可行的3D姿态并通过聚合得到最终姿态；

**💡 创新点**

创新点在于：①使用扩散模型在单帧条件下产生多种候选姿态，解决深度歧义；②采用视觉上下文与2D姿态特征的联合条件化，提升精度；③通过中位数聚合与置信度分析进一步提升结果并提供无监督置信度估计；

**🔧 技术方法**

主要技术包括：扩散概率模型（DDPM/DDIM）、Transformer式去噪网络（双模注意力）、HRNet-32特征提取、可变时间步嵌入以及多假设聚合与选择策略；

**📊 数据集**

在Human3.6M（含H36MA子集）和MPI-INF-3DHP两个公开基准上评估；

**📈 对比分析**

与多种单帧及基于时间序列的现有方法对比，SnapPose3D在MPJPE与P-MPJPE上均达到或超过state‑of‑the‑art水平，平均误差分别为42.8mm和34.5mm（Human3.6M）和在MPI-INF-3DHP上获得最高PCK、AUC和最低MPJPE；

**⚠️ 局限性**

局限性包括：①仍需一定的计算资源，单帧推理约3–10FPS；②依赖高质量2D姿态检测，检测误差会直接影响3D预测；③在极端遮挡或多人人工环境下多假设聚合效果可能受限；

---

## 289. FunFace: Feature Utility and Norm Estimation for Face Recognition

**arXiv ID:** 2604.26598 | [PDF](https://arxiv.org/pdf/2604.26598v1)

**作者:** Žiga Babnik `[一作]` (University of Ljubljana), Vitomir Štruc `[通讯]` (University of Ljubljana)

**通讯引用:** 4903 | [OpenAlex ID](https://openalex.org/A5038322250)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的自适应 margin 损失 FunFace，将面部图像质量评估中的置信度比（CR）与特征范数相结合，以提升低质量人脸识别性能。

**💡 创新点**

创新点在于将生物特征效用（CR-FIQA）直接嵌入 AdaFace 的 margin 计算，形成视觉质量与生物效用双重信息的自适应权重分配。

**🔧 技术方法**

使用了 AdaFace 框架、CR-FIQA、ResNet‑100、数据增强等技术。

**📊 数据集**

训练集使用 MS1MV2/3 和 WebFace4M；评估使用 LFW、CFP‑FP、CPLFW、AgeDB、CALFW、IJB‑C、TinyFace、DroneSURF 和 SurvFace 等基准。

**📈 对比分析**

与 ArcFace、CosFace、CurricularFace、AdaFace、MagFace、ElasticFace 等 13 种先进方法比较，FunFace 在低/中等难度数据集与 AdaFace持平，在高难度数据集上实现显著提升。

**⚠️ 局限性**

局限性包括训练时间显著增加（需计算最近负样本），在高/中等质量样本上提升不明显，且对增强策略的探究尚浅。

---

## 290. MappingEvolve: LLM-Driven Code Evolution for Technology Mapping

**arXiv ID:** 2604.26591 | [PDF](https://arxiv.org/pdf/2604.26591v1)

**作者:** Rongliang Fu `[一作]` (Chinese University of Hong Kong), Tsung-Yi Ho `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6749 | [OpenAlex ID](https://openalex.org/A5062800747)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MappingEvolve框架，利用LLM直接进化技术映射算法核心操作；

**💡 创新点**

通过层级规划-演化-评估架构，结合安全约束与多阶段验证，实现对核心操作的可控演化；

**🔧 技术方法**

使用大型语言模型（如DeepSeek‑V3、Qwen3‑Max、GPT‑5）配合OpenEvolve进行代码变异，结合编译、等价检查与QoR评估；

**📊 数据集**

在ISCAS85基准上做验证与消融实验，在EPFL 45路由/算术等基准上与ABC、mockturtle对比；

**📈 对比分析**

与OpenEvolve基线相比提升11.5×，与ABC对比面积减约10.04%，与mockturtle相比面积减约7.93%，S_overall提升46.6%–96.0%；

**⚠️ 局限性**

对延迟优化的探索有限，框架主要聚焦面积与面积-延迟权衡，且对不同LLM模型的适配性和可扩展性仍需进一步研究。

---

## 291. STAR-Filter: Efficient Convex Free-Space Approximation via Starshaped Set Filtering in Noisy Environments

**arXiv ID:** 2604.26626 | [PDF](https://arxiv.org/pdf/2604.26626v1)

**作者:** Yuwei Wu `[一作]` (University of Pennsylvania), Vijay Kumar `[通讯]` (University of Pennsylvania)

**通讯引用:** 39811 | [OpenAlex ID](https://openalex.org/A5087021192)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 STAR-Filter 框架，利用星形集过滤障碍点，快速生成凸多面体逼近可行自由空间。

**💡 创新点**

创新点在于：①将星形集极端点作为潜在约束点，显著减少冗余计算；②不需要全局碰撞检查，迭代式多面体-椭圆体更新可在噪声环境下保持鲁棒；③通过自适应翻转半径和边界补偿实现更大空间覆盖。

**🔧 技术方法**

核心技术包括星形集构造（球翻转映射）、凸多面体与最大体积内切椭圆体的混合整数/凸优化、以及自适应边界点补全和误差分析。

**📊 数据集**

实验数据集：仿真迷宫、随机障碍字段、M3ED 实际 LiDAR 点云，以及事件相机的实时稀疏点云。

**📈 对比分析**

与 FIRI-lite、RILS、Galaxy 等方法对比，STAR-Filter 在所有测试场景下均实现最低计算时间（平均 <2 ms），并保持与 FIRI-lite 相当甚至更优的多面体体积，碰撞率低于 0.03%，并保证种子点包含。

**⚠️ 局限性**

局限性包括：①对翻转半径的选择仍需经验调优；②当点云极其稀疏或包含误测点时，星形集极端点可能不足以捕获所有活跃约束；③当前实现基于 CPU，未充分利用 GPU 加速；④对动态环境的增量更新需要进一步研究。

---

## 292. When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models

**arXiv ID:** 2604.26649 | [PDF](https://arxiv.org/pdf/2604.26649v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22436 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个适用于大型推理模型的自适应检索框架，动态检测推理过程中知识缺口并在恰当时机触发检索。

**💡 创新点**

引入步骤级不确定度估计（RSUS）、基于强化学习的检索干预策略，以及高效的检索集成机制，解决了传统RAG与推理模型的时间错配。

**🔧 技术方法**

采用步骤级分割器、Verbalized Confidence、实体熵、一致性信号构成RSUS；使用策略网络和查询生成器；利用ColBERTv2+PLAID检索、KV缓存压缩与投机缓存。

**📊 数据集**

在MuSiQue、HotpotQA、2WikiMultiHopQA三大多跳问答基准上进行评测，并在DeepSeek-R1、QwQ等推理模型上进行实验。

**📈 对比分析**

与传统单次检索、IRCoT、Search-R1等基线对比，平均提升约10.1% F1，MuSiQue上达到71.2% F1，检索调用减少47%，在所有基准上均实现显著统计显著提升。

**⚠️ 局限性**

依赖检索语料库完整性、检索调用较多（≥3次）时分割与不确定度误差增加、对开放权重模型的性能仍低于专用模型、需要多模块训练与工程复杂度。

---

## 293. Swap distance minimization shapes the order of subject, object and verb in languages of the world

**arXiv ID:** 2604.26726 | [PDF](https://arxiv.org/pdf/2604.26726v1)

**作者:** Jairo Rios-El-Yazidi `[一作]` (Universitat Politècnica de Catalunya), Ramon Ferrer-i-Cancho `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 6592 | [OpenAlex ID](https://openalex.org/A5014322193)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究词序变异中的“交换距离最小化”原则，并在跨语言、跨族群、跨宏区的语料中检验其普适性。

**💡 创新点**

提出在六种基本词序中，用交换距离的平均值衡量变异程度，并给出随机基线Ω_r，证明Ω普遍低于Ω_r，说明词序趋向与最近的源序相近。

**🔧 技术方法**

统计检验（Wilcoxon符号秩检验）和分层抽样（Bootstrap）结合，用Ω与Ω_r比较；利用Simpson多样性指数和基于permutohedron的图示。

**📊 数据集**

使用基于Token的词序频数：新约圣经（NT）、Universal Dependencies（UD）和Surface‑Syntactic UD（SUD）树库，覆盖约1,000多种语言，111个族群。

**📈 对比分析**

与随机基线比较，p值均显著低于随机预期；在分层抽样下95%置信区间显示Ω远低于Ω_r；不同来源、缺乏支配序的语言同样符合该趋势。

**⚠️ 局限性**

局限包括：族群样本不均衡导致部分族群p值不显著，澳大利亚语种极少；无法解释具体为何选择某一词序；可能受语言接触或垂直遗传影响；仅聚焦S,O,V三成分，未涵盖复杂句子结构。

---

## 294. CurEvo: Curriculum-Guided Self-Evolution for Video Understanding

**arXiv ID:** 2604.26707 | [PDF](https://arxiv.org/pdf/2604.26707v1)

**作者:** Guiyi Zeng `[一作]` (Huazhong University of Science and Technology), Zikai Song `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 572 | [OpenAlex ID](https://openalex.org/A5083665721)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 CurEvo——一种将课程学习引入视频问答自演化的框架，能够在无人工标注的条件下通过生成–评估–训练的循环实现模型的自适应提升。

**💡 创新点**

创新点包括：① 在自演化流程中加入多维度（感知、语义、推理）问题生成与评估，形成结构化的学习轨迹；② 通过模型能力反馈动态调整任务难度、采样比例和样本权重；③ 采用类型自适应评估机制，实现对不同难度级别的高质量样本筛选与加权。

**🔧 技术方法**

核心技术：多模态大语言模型（Video‑LLM）作为生成器与评估器；自适应采样比例、阈值更新与样本加权；LoRA 微调；模板化多维度问题生成；自训练循环与反馈驱动的课程更新。

**📊 数据集**

实验数据集：ActivityNet‑QA、NExT‑QA、MSRVTT‑QA、MSVD‑QA 四个主流视频问答基准。

**📈 对比分析**

与各基线模型（Video‑LLaVA、LLaVA‑OneVision、VILA、Video‑LLaMA3、InternVL2.5、Qwen2.5‑VL、Qwen3‑VL）在上述四个基准上进行对比，CurEvo 在所有模型上均实现 2–4% 的准确率提升，语义分数也同步上升，整体性能表现稳健且一致。

**⚠️ 局限性**

局限性：① 受评估器质量限制，评估器弱于基础模型时提升有限；② 仍依赖评估器的可靠性，噪声样本筛选效果受评估模型能力影响；③ 当前仅支持基于模板的短问答生成，未覆盖长文本或跨场景复杂推理。

---

## 295. Understanding the Skills Gap between Higher Education Institutions and the Software Engineering Industry

**arXiv ID:** 2604.26655 | [PDF](https://arxiv.org/pdf/2604.26655v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 296. Reproducible Automated Program Repair Is Hard -- Experiences With the Defects4J Dataset

**arXiv ID:** 2604.26674 | [PDF](https://arxiv.org/pdf/2604.26674v1)

**作者:** Adam Krafczyk `[一作]` (University of Hildesheim), Klaus Schmid `[通讯]` (University of Hildesheim)

**通讯引用:** 3889 | [OpenAlex ID](https://openalex.org/A5076691956)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了 Defects4J 数据集在自动程序修复（APR）评估中的可工作性与测试套件质量，并基于此提出并实现了改进评估框架和多项技术修复方案。

**💡 创新点**

首次从 APR 视角定义了工作性需求，量化并修复了 Defects4J 中的编译、运行、测试一致性等技术与数据缺陷，揭示了易被误修复的缺陷对评估结果的显著影响，并强调测试套件质量对修复率的决定作用。

**🔧 技术方法**

采用自动化构建与测试执行框架，结合 JUnit 驱动、JaCoCo 覆盖收集、并行化执行与元数据校正技术，对 Defects4J 中每个缺陷进行多轮测试、单语句删除实验以及一致性与 flakiness 检测。

**📊 数据集**

使用 Defects4J 2.0 版本（共 835 个缺陷，来源于 17 个开源 Java 项目）。

**📈 对比分析**

通过对比修复率、可工作缺陷比例和剔除易修复缺陷后的结果，展示剔除问题后修复率降至原来的约 2/3，评估框架显著提升实验效率（从 95 小时降至 7 h 45 m）。

**⚠️ 局限性**

研究仅针对 Defects4J 2.0，方法与结论对其他数据集不一定适用；未能完全解决所有一致性和结果不匹配问题；实验过程中对手工检查的依赖和假设也可能导致结果偏差。

---

## 297. From Black-Box Confidence to Measurable Trust in Clinical AI: A Framework for Evidence, Supervision, and Staged Autonomy

**arXiv ID:** 2604.26671 | [PDF](https://arxiv.org/pdf/2604.26671v1)

**作者:** Serhii Zabolotnii `[一作]` (Cherkasy State Business College), Olha Antonenko `[通讯]` (healthPrecision)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于证据、监督和分阶段自治的可测量可信临床 AI 框架，并设计了多层架构（确定性核心、AI 辅助、分层模型升级和人工监督）及对应的信任指标。

**💡 创新点**

创新点在于将可信度视为系统属性而非单一模型特性，采用多层分层自治与可追溯的证据链、分层升级机制以及度量化的信任指标来实现可操作、可测量的可信临床 AI。

**🔧 技术方法**

使用大语言模型（Anthropic Claude Sonnet 4.6）进行提示生成与结构化输出，结合确定性规则引擎、上下文适配器和分层模型分类器，并通过元测量（GUM/VIM）原理制定信任度量。

**📊 数据集**

本文未公开使用特定临床数据集，主要基于假想的真实临床案例和参考标准来说明信任度量和评估方法。

**📈 对比分析**

比较方法以多层信任指标（如规则覆盖率、上下文相关性、升级精度等）为主，并与传统单一模型准确率、F1 等指标对照；文中未给出数值性能，但提出了可操作的评估框架。

**⚠️ 局限性**

局限性：缺乏大规模实证验证和真实数据集实验，信任度量仍处于概念阶段，未在临床部署中检验其实际效果；同时对成本和人力负担的量化分析仍待进一步研究。

---

## 298. Will It Break in Production? Metric-Driven Prediction of Residual Defects in Python Systems

**arXiv ID:** 2604.26667 | [PDF](https://arxiv.org/pdf/2604.26667v1)

**作者:** Giuseppe De Rosa `[一作]` (University of Naples Federico II), Pietro Liguori `[通讯]` (University of Naples Federico II)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5073369636)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个平衡的 Python 残留缺陷数据集，评估了 LLM、深度学习和传统机器学习模型在残留缺陷预测任务中的效果。

**💡 创新点**

创新点在于将残留缺陷与一般缺陷预测明确区分，公开了首个残留缺陷数据集与复现包，并系统对比了代码嵌入与软件指标的互补性。

**🔧 技术方法**

采用的技术包括传统监督学习（RandomForest、XGBoost、CatBoost）、基于 LLM 的推断（Gemini、Claude、GPT‑4）和 fine‑tuned 深度学习模型（CodeT5+、CodeLlama、DeepSeek‑Coder），以及 PCA/CCA 进行表示空间对比。

**📊 数据集**

使用的数据集为 PyresBugs（5,007 例残留缺陷）+ 500 例补充案例，和 BugsInPy 评估集（284 残留/221 非残留），共 83 维产品、过程、统计和 Python 特定指标。

**📈 对比分析**

通过交叉项目实验和 90/10 训练/验证划分，使用准确率、召回率和 F1 进行比较；LLM 与 DL 模型表现较差，而监督指标模型达到 0.85–0.90 的召回率和 0.71–0.73 的 F1，显著优于其它方法。

**⚠️ 局限性**

主要局限包括仅研究 Python，残留缺陷标签依赖启发式分类，LLM 与代码嵌入在跨项目上不稳定，混合模型未能提升性能。

---

## 299. PiGGO: Physics-Guided Learnable Graph Kalman Filters for Virtual Sensing of Nonlinear Dynamic Structures under Uncertainty

**arXiv ID:** 2604.26593 | [PDF](https://arxiv.org/pdf/2604.26593v1)

**作者:** Marcus Haywood-Alexander `[一作]` (ETH Zürich), Eleni Chatzi `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于物理引导的图神经ODE（PiGGO）框架，并将其嵌入扩展卡尔曼滤波（EKF）中，用于在线虚拟感知非线性结构系统的状态。

**💡 创新点**

创新点在于将物理学先验与可学习的图神经ODE相结合，形成可捕捉非线性动力学的连续时间状态转移模型，并通过递归贝叶斯滤波实现对模型不确定性和稀疏测量的自适应纠正，提升了泛化能力。

**🔧 技术方法**

采用的技术包括图神经网络（Message‑Passing、GraphNet 变体）、图神经ODE、物理约束损失函数、扩展卡尔曼滤波以及基于自动微分的雅可比矩阵计算。

**📊 数据集**

使用了两类合成数据集：Sobol 随机三角网（带三次弹性非线性）和桥梁三角网（带角度间隙非线性），并在不同尺寸、不同稀疏率下进行离线训练与在线推断。

**📈 对比分析**

通过对比离线 GNODE 与预训练 GNODE‑EKF 两种方法，发现 EKF 在 NMSE 上平均降低 80%–95%（如随机网从 24.5 降至 0.3），并在置信区间内包络真实响应，证明了更优的准确性与不确定性估计。

**⚠️ 局限性**

主要局限包括：需要已知初始状态进行离线训练；假设所有非线性项具有相同形式和全局参数，难以处理局部或非平滑非线性；滤波过程假定高斯误差，无法捕捉离散跳变；以及在实际现场结构中可能需要更长时间窗口或更复杂的图结构学习。

---

## 300. Zero-Shot to Full-Resource: Cross-lingual Transfer Strategies for Aspect-Based Sentiment Analysis

**arXiv ID:** 2604.26619 | [PDF](https://arxiv.org/pdf/2604.26619v1)

**作者:** Jakob Fehle `[一作]`, Christian Wolff `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对七种语言的四个 ABSA 子任务（ACD、ACSA、TASD、ASQP）进行了系统评测，比较了编码器、序列生成和 LLM 三种建模范式。

**💡 创新点**

创新点包括首次在非英语语言中评估 ASQP，贡献德国 ASQP 数据集 GERest，并在零资源、仅数据、全资源三种场景下全面对比方法。

**🔧 技术方法**

采用多语言 BERT、mT5、Hier‑GCN、LLaMA 3.1、Gemma 3 27B 等模型，并结合代码混用、机器翻译等数据增强技术。

**📊 数据集**

使用 SemEval‑2016 餐厅评论数据（英语、法语、西班牙语、荷兰语、俄语、土耳其语），GERestaurant（德语）以及新建的德国 ASQP 数据集 GERest。

**📈 对比分析**

在全资源下 LLaMA 3.1 取得最高分，零资源场景中 LLaMA 3.1 仍保持 80% 以上 F1，Gemma 3 27B 在少量样本下表现稳定，编码器模型在简单任务上竞争力强。

**⚠️ 局限性**

局限在于仅覆盖有限语言和餐厅领域，数据平衡化可能偏离真实分布，模型在极低资源语言和更复杂任务上的迁移仍有限。

---

## 301. SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection

**arXiv ID:** 2604.26633 | [PDF](https://arxiv.org/pdf/2604.26633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 302. TDD Governance for Multi-Agent Code Generation via Prompt Engineering

**arXiv ID:** 2604.26615 | [PDF](https://arxiv.org/pdf/2604.26615v1)

**作者:** Tarlan Hasanli `[一作]` (University of Jyväskylä), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 10331 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个 AI‑native TDD 框架，通过 Prompt 设计与多阶段治理机制，将传统 TDD 原则转化为可执行的流程，分离生成与执行，限制 LLM 行为并确保代码质量。

**💡 创新点**

核心创新在于将 TDD 原则编码为可执行的“Manifesto”，并在规划、生成、修复与验证四个阶段分别施加阶段约束、修复循环限制和确定性验证，实现对 LLM 行为的分层治理与可追溯性。

**🔧 技术方法**

技术包括 Prompt Engineering、Agent 级多角色协作、LLM 提议生成、确定性执行引擎、结构化修复循环、JSON‑based Manifesto、阶段门控与回滚机制。

**📊 数据集**

论文未使用公开数据集；实验为基线提示对比，主要在小规模示例上进行验证。

**📈 对比分析**

与传统无治理提示相比，实验表明重试循环次数下降、代码更符合预期，但缺乏系统的量化指标与大规模基准；仅给出了经验性对比。

**⚠️ 局限性**

局限性包括：Manifesto 约束主要在 Prompt 层，缺乏完整语义级别的运行时验证；受限的探索性导致在复杂重构场景中效果有限；验证仍处于初步阶段，未在大型仓库或 CI/CD 环境中评估；对模型多样性与可扩展性的影响尚未探究。

---

## 303. When to Vote, When to Rewrite: Disagreement-Guided Strategy Routing for Test-Time Scaling

**arXiv ID:** 2604.26644 | [PDF](https://arxiv.org/pdf/2604.26644v1)

**作者:** Zhimin Lin `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 42091 | [OpenAlex ID](https://openalex.org/A5013794939)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个训练无关的实例级路由框架，利用模型输出分歧决定在推理时采用重采样、投票或重写策略，以提升推理质量

**💡 创新点**

创新地将推理时规模扩展视为实例级路由问题，并用轻量的“最小分歧检测”自动决定何时重写、投票或直接使用结果

**🔧 技术方法**

使用最小分歧检测器、少量采样、投票机制与重写重推理，以及基于模型不确定性的动态决策

**📊 数据集**

在七个数学推理基准（GSM8K、Math500、Gaokao2023en、Olympiadbench、AMC23、AIME24、AIME25）以及代码生成基准（HumanEval、MBPP）上评估

**📈 对比分析**

与多种对比方法（多数投票、动态投票、SCoP、Best‑of‑N）比较，实验显示在三大模型上平均提升3%–7%准确率，同时采样量比其他方法低30%–50%

**⚠️ 局限性**

主要局限在于重写对易题可能产生误差，且依赖分歧信号；当分歧率低或任务不易产生不一致时，路由策略效果有限

---

## 304. Electricity price forecasting across Norway's five bidding zones in the post-crisis era

**arXiv ID:** 2604.26634 | [PDF](https://arxiv.org/pdf/2604.26634v1)

**作者:** My Thi Diem Phan `[一作]` (Independent researcher), Dat Thanh Nguyen `[通讯]` (University of Oslo)

**通讯引用:** 2262 | [OpenAlex ID](https://openalex.org/A5020547925)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了覆盖挪威五个 Nord Pool 竞价区的 2019–2025 年多模态小时级数据集，并在 2025 年严格因果测试集上评估了 LightGBM、XGBoost、岭回归 ARX、LSTM、TCN、Transformer 等模型，提出首个后危机跨区统一预测基准。

**💡 创新点**

创新点在于首次结合跨区、后危机、严格因果拆分、滚动回测以及留一组特征消融与条件 regime 分析，系统揭示滞后价格主导预测性能，外部变量仅在诊断层面具有价值。

**🔧 技术方法**

采用的技术包括梯度提升树（LightGBM、XGBoost）、岭回归 ARX、循环网络（LSTM）、时序卷积网络（TCN）、Transformer 以及传统 Naïve 基线，并配合 Diebold–Mariano 检验、滚动回测等评估方法。

**📊 数据集**

使用的数据集集成了 ENTSO‑E 电力系统数据、NVE 水库统计、Open‑Meteo 天气回归、Yahoo Finance 商品价格等多源信息，形成 2019–2025 年按小时的面板数据。

**📈 对比分析**

通过严格因果拆分、52 周滚动回测和 DM 检验比较，LightGBM 在所有区域均实现 MAE 1.64–5.74 EUR/MWh，显著优于 Ridge ARX、XGBoost、深度模型和 Naïve‑24h，并在多数区域表现出统计显著性。

**⚠️ 局限性**

局限性包括深度模型未做细粒度调优、仅进行点预测、天气输入空间分辨率有限、特征消融与 regime 分析仅为描述性，且未考虑概率预测或更复杂的 regime‑特定模型。

---

## 305. An Effective Orchestral Approach to Satisfiability Modulo Prime Fields

**arXiv ID:** 2604.26709 | [PDF](https://arxiv.org/pdf/2604.26709v1)

**作者:** Miguel Isabel `[一作]` (Complutense University of Madrid), Albert Rubio `[通讯]` (Complutense University of Madrid)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一种基于 DPLL(T) 框架的理论求解器，用于判断在素域上由多项式方程组成的公式的可满足性。

**💡 创新点**

创新点在于提出多模块“指挥家”策略：按效率从轻量级线性、整数线性、线性子句推理到重度 Gröbner 基础和实数非线性模块进行顺序调用，从而在保持可扩展性的同时大幅提升求解速度。

**🔧 技术方法**

采用了 Gröbner 基础、素域线性代数、整数线性规划、等价归约（欧同）、非线性实数求解等多种技术，并通过模块化互补实现快速冲突检测和模型生成。

**📊 数据集**

实验使用了来自 ZKP 编译器正确性验证的基准集（1602 例）以及由 CircomLib 生成的算术电路弱安全性基准集（719 例）。

**📈 对比分析**

与现有最优的有限域 SMT 求解器（Z3+UF_FF、CVC5）在相同硬件与 300 秒超时下进行对比；在 ZKP 基准上平均求解时间从 1.5 秒降至 0.7 秒，覆盖率从 83% 提升到 92%；在电路基准上平均求解时间从 3.59 秒降至 1.26 秒，覆盖率从 92% 提升到 99%。

**⚠️ 局限性**

主要局限在于 Gröbner 基础的计算成本导致完整性无法完全保证，模块化调度仍依赖经验选择，且对极大规模实例的性能仍受限。

---

## 306. Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising

**arXiv ID:** 2604.26694 | [PDF](https://arxiv.org/pdf/2604.26694v1)

**作者:** Jun Guo `[一作]` (Tsinghua University), Huaping Liu `[通讯]` (Tsinghua University)

**通讯引用:** 12371 | [OpenAlex ID](https://openalex.org/A5041101317)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了统一的4D世界动作模型，能够在同一框架中同时预测未来RGB‑D视频、生成高质量3D重建并实时执行机器人动作，实现视频合成与动作执行的无缝集成。

**💡 创新点**

创新点包括：①轻量级深度适配模块，将预训练Diffusion Transformer后几层复制为深度分支，既保留预训练视觉先验，又实现3D空间建模；②异步噪声采样（ANS），在训练与推理中同步视频与动作的噪声分布，既实现快速动作解码，又保证视频生成质量。

**🔧 技术方法**

采用预训练视频扩散模型Wan2.2‑5B、Diffusion Transformer、流匹配训练框架，结合交叉注意力的深度分支与异步噪声采样，推理时使用UniPC多步调度器。

**📊 数据集**

训练使用超过5800小时的机器人数据（包含真实与仿真），评估基于RoboCasa、RoboTwin 2.0模拟基准以及真实耳机打包实验。

**📈 对比分析**

与多种VLA与WAM基线（π_0、GR00T‑N1.5、UWM、DreamZero、Cosmos Policy、Motus、GigaWorld‑Policy）对比，模型在RoboCasa上达到79.2%成功率、RoboTwin 2.0上达到90.7%，均超越所有基线；在4D重建上相较于DreamZero+DA3、Robot4DGen等，PSNR、LPIPS、AbsRel、Chamfer距离均取得最优。

**⚠️ 局限性**

局限性包括：仍需大规模预训练与长时间推理，对多视角摄像头与机器人标定要求高；轻量化深度分支在极端几何或动态光照下性能可能下降；模型规模大，推理时对GPU资源需求仍高。

---

## 307. COPUS: Co-adaptive Parallelism and Batch Size Selection in Large Language Model Training

**arXiv ID:** 2604.26687 | [PDF](https://arxiv.org/pdf/2604.26687v1)

**作者:** Akhmed Sakip `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Qirong Ho `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 3041 | [OpenAlex ID](https://openalex.org/A5012361506)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并实现了 COPUS，一种在大型语言模型训练过程中实时联合调节全局批量大小、微批量大小以及 3D 并行（数据、张量、流水线）策略的系统。

**💡 创新点**

创新点：①将训练效率（统计效率）与硬件吞吐量通过 Goodput 指标统一考虑，构建端到端的共适配决策；②设计了 3D 并行感知的梯度噪声尺度（GNS）估计器；③实现了在线重分片（resharding）实现并行策略的无缝切换，避免了完整检查点重启的高昂开销。

**🔧 技术方法**

技术细节：使用 GNS 与离线吞吐量表相结合计算 Goodput；利用 Adam 的平方根学习率缩放修正 Goodput；实现了基于梯度范数的 GNS 估计器；实现了在训练循环中广播批量尺寸、学习率的在线调整；通过在线状态重分片实现 DP/T/P 组合切换；使用 EMA 平滑 GNS 并引入 10% 的切换阈值。

**📊 数据集**

数据集：WikiText‑103（单语料，长度 2048 tokens）。

**📈 对比分析**

评估方法：在 NVIDIA H100（8–32 GPUs）和 AMD MI210（8–32 GPUs）集群上，分别对 3B、13B、32B、7B 四个模型进行预训练；与静态并行+批量、CBS（仅批量自适应）以及各类最优静态配置进行对比。实验结果表明 COPUS 在 5 个不同损失阈值上平均提升 3.9–8.0%，峰值提升 11.1%，含重分片开销；若消除重分片开销，平均提升可达 4.7–11.4%。

**⚠️ 局限性**

局限性：①需要为每个模型‑硬件组合预先离线生成吞吐量表；②GNS 的校准因子（c = 2.0）是经验值，可能需针对不同模型或阶段自适应；③决策空间仅包含数据、张量、流水线三维并行，未涵盖 ZeRO、上下文/序列并行等；④GNS 估计噪声大，仍需 EMA 与切换阈值缓冲；⑤实验规模限制在 4 节点（32 GPU）以内，无法验证更大规模下的耦合效应。

---

## 308. Evolutionary feature selection for spiking neural network pattern classifiers

**arXiv ID:** 2604.26654 | [PDF](https://arxiv.org/pdf/2604.26654v1)

**作者:** Michal Valko `[一作]` (Comenius University), Marco Castelani `[通讯]` (University Nova De Lisboa)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用进化特征选择算法 FeaSANNT 对 JASTAP 生物学现实感突触神经网络进行训练，完成对 Iris 数据集的模式分类。

**💡 创新点**

创新点在于将 FeaSANNT 进化特征选择与 JASTAP 时序编码突触网络相结合，首次在单层无隐藏层的极小网络上实现高精度分类，并构造专门针对时序突触网络的适配度函数。

**🔧 技术方法**

使用技术包括：JASTAP 时序突触神经网络、进化特征选择算法 FeaSANNT、输入到时间编码（基于 Gamma 分布的噪声处理）、自定义多目标适配度函数以及遗传算法的交叉、变异与精英保留策略。

**📊 数据集**

主要使用的数据集为 UCI Iris 数据集（150 条样本、4 个连续特征、3 类），并在训练过程中对输入进行 Gamma 分布噪声干扰进行鲁棒性评估。

**📈 对比分析**

方法比较：与传统 MLP+BP（无隐藏层、4 个输入、3 个输出）以及 FeaSANNT 仅训练网络参数（无特征选择）进行对比。结果显示：MLP 96.2% 准确率，FeaSANNT 94.7%，FeaSTAP（JASTAP+FeaSANNT）达到 100% 准确率；且在高达 10% Gamma 噪声下仍保持 100% 精度，证明了噪声鲁棒性。计算时间方面，迭代次数相对较多（300–700 次）。

**⚠️ 局限性**

局限性包括：训练时间过长（仿真时间占主导）、特征选择效果有限（因 JASTAP 能利用无关输入的膜电位触发阈值），可能出现过拟合（Iris 数据集不可完全线性可分且结果高于预期），缺乏在更大规模、多类别、非线性可分数据集上的验证；未来需改进仿真效率、限制可学习参数、并行化进化过程以降低计算成本。

---

## 309. Differentially-Private Text Rewriting reshapes Linguistic Style

**arXiv ID:** 2604.26656 | [PDF](https://arxiv.org/pdf/2604.26656v1)

**作者:** Stefan Arnold `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Stefan Arnold `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 878 | [OpenAlex ID](https://openalex.org/A5091158825)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究差分隐私文本重写对语言风格的影响，并比较自回归与双向两种重写架构的表现

**💡 创新点**

首次系统地用高维风格特征与功能维度投影分析DP重写的风格同化与功能转移，揭示两种架构在风格保留与功能失真的差异

**🔧 技术方法**

差分隐私机制（指数机制+温度采样）结合自回归语言模型（DP-Paraphrase）与双向掩码语言模型（DP-MLM）

**📊 数据集**

CORE（Corpus of Online Registers of English）语料库，共9691篇文档

**📈 对比分析**

通过Burrows’ Delta和特征频率比值进行对比，结果显示DP-MLM在语义与风格保持上优于DP-Paraphrase，且两者均趋向中性信息化写作；自回归模型达到收敛平稳点但仍偏离人类风格

**⚠️ 局限性**

仅评估风格变化，未检验对作者识别的实质影响；隐私与功能保留之间的权衡仍需进一步验证

---

## 310. Which Types of Heterogeneity Matter for Root Cause Localization in Microservice Systems ?

**arXiv ID:** 2604.26670 | [PDF](https://arxiv.org/pdf/2604.26670v1)

**作者:** Runzhou Wang `[一作]` (Nankai University), Yangyuxin Huang `[通讯]` (Nankai University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文提出一种半监督根因定位框架NexusRCL，用以解决微服务系统中的异质性挑战。

**💡 创新点**

创新点在于同时建模实体层异质性（主机与服务不同类型节点）与数据层异质性，并通过主动学习实现低标注成本。

**🔧 技术方法**

主要技术包括异构图神经网络（HGCN）、事件抽象机制、三步主动学习（聚类、中间标签传播、不确定性查询）。

**📊 数据集**

使用了工业级微服务基准HD1和HD2两大数据集。

**📈 对比分析**

与CausalRCA、ART、DiagFusion、Eadro、DejaVu等基线比较，NexusRCL在Top-1准确率提升高达49.85%，同时推理时间仅2.5s/1.2s，表现优异。

**⚠️ 局限性**

限制在于对新出现的服务无法及时适应，需要重新训练；伪标签噪声可能影响小数据集的精度。

---

## 311. AgentSim: A Platform for Verifiable Agent-Trace Simulation

**arXiv ID:** 2604.26653 | [PDF](https://arxiv.org/pdf/2604.26653v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Jelena Mitrovic `[通讯]` (University of Passau)

**通讯引用:** 659 | [OpenAlex ID](https://openalex.org/A5019466280)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 AgentSim 平台，用于生成可验证的 RAG 代理推理轨迹，并发布了 Agent‑Trace Corpus；

**💡 创新点**

创新点在于 Corpus‑Aware Seeding 策略实现文档覆盖与多样性，Active Validation 人机循环减少标注成本，以及生成可检验的多步推理数据；

**🔧 技术方法**

技术包括多模型异议检测 Divergence Score、MMR 选择、基于 embedding 的聚类、RAG 工作流模块化、LLM 与检索结合、人工审核与 LoRA 微调；

**📊 数据集**

使用 MS MARCO、Quasar‑T、CausalQA 三个 IR 基准，生成 103,567 步推理轨迹；

**📈 对比分析**

通过与随机、分层、DPP 等种子策略以及 GPT‑4o、DeepSeek 等分析模型对比，证明 Corpus‑Aware 在覆盖率、文档多样性、探测广度方面显著优于基线，且模型行为差异可量化；

**⚠️ 局限性**

局限性包括对源语料的偏倚、行为分析仅针对三种模型、人工审核仍需标准化、平台在专业领域的迁移性待验证。

---

## 312. The Bandit's Blind Spot: The Critical Role of User State Representation in Recommender Systems

**arXiv ID:** 2604.26651 | [PDF](https://arxiv.org/pdf/2604.26651v1)

**作者:** Pedro R. Pires `[一作]` (Federal University of São Carlos), Tiago A. Almeida `[通讯]` (Federal University of São Carlos)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统评估了不同基于嵌入的用户状态表示对上下文多臂老虎机（CMAB）推荐系统性能的影响，比较了多种嵌入模型、聚合策略和Bandit算法，展示了状态表示对推荐效果的决定性作用。

**💡 创新点**

创新点在于：①首次大规模实证比较不同嵌入模型与聚合方式对CMAB推荐的影响；②发现状态表示的质量对性能的影响远大于Bandit算法本身；③揭示在不同数据集和设置下，没有单一的最优状态表示，强调了针对领域的评估需求。

**🔧 技术方法**

使用的技术包括：隐式ALS和BPR矩阵分解嵌入、用户嵌入/平均/串联聚合、三种线性CMAB算法（LinUCB、LinGreedy、LinTS）以及NDCG@20评估。

**📊 数据集**

使用的公开数据集有：Amazon Beauty、Amazon Books、BestBuy、Delicious、MovieLens‑100K、MovieLens‑25M、RetailRocket。

**📈 对比分析**

比较方法：在训练集上预训练嵌入并固定，在测试集按时间窗口在线评估，累计NDCG@20。实验结果显示：嵌入模型与聚合方式的组合往往比更改Bandit算法带来更大提升，性能提升幅度可达数十倍；不同数据集呈现“轮番胜出”现象。

**⚠️ 局限性**

限制：仅使用静态预训练嵌入，未考虑动态嵌入更新；聚合策略仅为三种基础方式；实验仅覆盖线性Bandit模型，缺乏深度学习或非线性方法的比较；对嵌入超参数的探索有限。

---

## 313. SciHorizon-DataEVA: An Agentic System for AI-Readiness Evaluation of Heterogeneous Scientific Data

**arXiv ID:** 2604.26645 | [PDF](https://arxiv.org/pdf/2604.26645v1)

**作者:** Dianyu Liu `[一作]`, Hengshu Zhu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一种面向异构科学数据的 AI-readiness（AI 准备度）评估框架，构建了四维评估维度（治理可信度、数据质量、AI 兼容性、科学适应性），并实现了一个多代理系统（MAS）通过数据感知、知识增强规划、工具化执行和自我校正，实现了对大规模、多模态科学数据的可扩展、自动化评估与报告生成。

**💡 创新点**

创新点主要包括①首次将 AI 兼容性和科学适应性纳入数据评估维度；②设计可细粒度、可执行的原子评估元素；③构建基于有向循环图的多代理工作流，结合知识增强规划与自适应工具生成，实现真正意义上的“agentic”评估；④通过运行时验证与语义校正实现工具生成的闭环自我纠错。

**🔧 技术方法**

使用的技术包括多代理系统（MAS）与有向循环工作流、轻量级数据概况（Profile-Oriented Data Inspector）、适用性感知指标选择器、知识增强评估规范规划器、工具库与工具记忆、动态工具构造与适配、运行时与语义复核、以及统一分数归一化与层次聚合。

**📊 数据集**

实验数据集覆盖六大科学领域（天文、生物医学、地球科学、材料化学、物理与工程、社会经济学），从权威开放科学仓库收集，涵盖多模态（表格、图像、序列、复合）。

**📈 对比分析**

在六大领域进行系统评测，采用系统正确率（SC）、工具创建成功率（TCSR）与工具创建效率（TCE）等指标。结果显示系统能够在大多数数据集上完整生成多维评估报告，TCSR 与 SC 较高；但治理和质量维度普遍高于 AI 兼容性与科学适应性，表明开放数据在格式与治理上已成熟，却在 AI 模型适配与科学推理支持方面仍有不足。

**⚠️ 局限性**

局限性主要包括：①对极大数据集或特殊格式仍需人工编写专用解析器；②部分原子评估元素需要领域专家介入；③系统对 LLM 生成工具的可靠性与可解释性仍存在不确定性；④评估指标仍侧重可量化属性，对深层次科学推理支持的定性评估不足。

---

## 314. SAGE: A Strategy-Aware Graph-Enhanced Generation Framework For Online Counseling

**arXiv ID:** 2604.26630 | [PDF](https://arxiv.org/pdf/2604.26630v1)

**作者:** Eliya Naomi Aharon `[一作]` (Ben-Gurion University of Negev), Kobi Gal `[通讯]` (Ben-Gurion University of Negev)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多层异质图（对话层、发言层、心理层），并在此基础上设计了 SAGE 框架：先通过图神经网络预测下一个治疗策略，再将图结构信息与预测策略作为软提示注入微调后的 LLM，生成针对危机热线情境的推荐回复。

**💡 创新点**

创新点在于：① 将理论驱动的心理词典（SRF）与对话记录联合构成异质图，显式编码心理状态与对话动态；② 采用两阶段推理：先预测治疗策略，再用策略指导生成；③ 通过 Graph-Aware Attention 将图结构信息映射为连续软提示，保持结构信息完整性，提升生成质量。

**🔧 技术方法**

使用技术包括：Heterogeneous Graph Transformer (HGT)、Next Strategy Classifier（多标签分类）、Graph-Aware Attention + 软提示、LoRA 微调 Gemma-3-12b-it LLM、BERTScore 与 Perplexity 评估指标、以及人类专家的盲评。

**📊 数据集**

数据集为 150 份在线危机热线会话（共 2258 个干预点），全部为希伯来语，包含帮助者与志愿者的对话记录；辅以 4000+ 词条的 Suicide‑Risk Factors (SRF) 词典用于映射心理类别。

**📈 对比分析**

与基线（仅文本上下文、无图信息或无策略预测）进行比较。策略预测方面，SAGE 的 MCC 0.59、F1 0.77，远超无图基线的 MCC 0.12。生成质量方面，SAGE 的 BERTScore 0.701、Perplexity 26.15，优于 Vanilla (BERTScore 0.617、PPL 540.5)。人类专家评估显示，SAGE 的偏好率为 50.2%，显著高于 Vanilla 的 34.5%。

**⚠️ 局限性**

局限性：① 需要专业心理学家标注，导致样本量受限；② 评估仅在离线对话日志上进行，未验证在真实交互中的因果影响；③ 数据仅来自以色列的单一危机热线，可能不具备跨文化或多语言的普适性。

---

## 315. OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory

**arXiv ID:** 2604.26622 | [PDF](https://arxiv.org/pdf/2604.26622v1)

**作者:** Jinze Li `[一作]` (University of Hong Kong), Edith Cheuk-Han Ngai `[通讯]` (University of Hong Kong)

**通讯引用:** 6391 | [OpenAlex ID](https://openalex.org/A5077317339)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种将代理交互历史转换为视觉图像并通过定位-转录检索的记忆框架 OCR‑Memory。

**💡 创新点**

创新点在于使用视觉高密度编码替代文本压缩、采用定位-转录机制消除幻觉，并引入多分辨率与主动召回来平衡存储与精度。

**🔧 技术方法**

技术包括 DeepSeek‑OCR 图像编码、SoM 视觉标记、阈值+TopK 召回策略、分辨率课程、LoRA 微调等。

**📊 数据集**

使用的数据集包括 Mind2Web、AppWorld 进行评估，HotpotQA 用于模型微调。

**📈 对比分析**

与文本检索、MemoryBank、AWM、ACON 等基线对比，在 4096 token 预算下，Mind2Web 的任务成功率提升约 4.8%，AppWorld Hard 任务提升至 30.8%，并在多分辨率和 token 限制场景下保持高精度。

**⚠️ 局限性**

局限包括需要额外训练、渲染图像成本高、磁盘占用大、模型体积增加。

---

## 316. Impact of Attitude and Bounded Rationality on Collective Behavioral Transitions

**arXiv ID:** 2604.26616 | [PDF](https://arxiv.org/pdf/2604.26616v1)

**作者:** Chen Song `[一作]` (Nanyang Technological University), Karl H. Johansson `[通讯]` (Kth Royal Institute Of Technology)

**通讯引用:** 45468 | [OpenAlex ID](https://openalex.org/A5045975901)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并分析了一个基于计划行为理论（TPB）的动态代理人模型，加入行为到态度的反馈机制，研究了在不同态度影响权重ϕ和决策理性β下，群体如何从不良行为转向良好行为或相反的社会转变。

**💡 创新点**

创新点包括：①将TPB的单向因果关系转为双向动态反馈模型；②将心理学中的数感迟钝（psychophysical numbing）原则应用于态度更新；③通过对ϕ和β的敏感性分析揭示了受限理性在社会转变中的双重作用。

**🔧 技术方法**

使用了代理人模拟（agent‑based simulation）、二元Logit选择模型、心理学原理（如数感迟钝）以及定量的态度/意图更新规则。

**📊 数据集**

使用的“数据集”是人工生成的300名代理人的初始态度、意图分布（如U[0,0.4]或U[0.6,1]等），并在不同参数组合下进行多次模拟实验。

**📈 对比分析**

方法比较通过设置ϕ∈{0.3,0.7}与β∈{5,10}（以及更极端的β）来观察群体行为平均值y_avg(t)的转变速度和完成度。结果显示：ϕ越大、β越适中，良好行为的采纳越快；而在负面行为场景中，ϕ和β的影响相互抵消，表现出更复杂的双向效应。

**⚠️ 局限性**

局限性：①模型参数（ϕ、β、λ）尚未在真实人群中测量；②缺乏实证验证，主要基于模拟结果；③未考虑多种社会网络结构、外部干预策略等现实因素，难以直接迁移到真实社会情境。

---

## 317. Human-in-the-Loop Benchmarking of Heterogeneous LLMs for Automated Competency Assessment in Secondary Level Mathematics

**arXiv ID:** 2604.26607 | [PDF](https://arxiv.org/pdf/2604.26607v1)

**作者:** Jatin Bhusal `[一作]` (Sunway), Raunak Regmi `[通讯]` (Sunway)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于CBE的多维度数学评估框架，并用四种LLM对33名10年级学生手写作答进行评分，比较其与人工评审的一致性。

**💡 创新点**

提出了兼顾四大跨切能力的竞赛评分框架，并首次实证比较稀疏MoE与稠密模型在严格rubric约束下的适配性。

**🔧 技术方法**

使用多模型联合架构（Eagle、Orion、Nova、Lyra）、统一主评估Prompt、Gemini多模态OCR、加权二次Cohen's Kappa等统计度量。

**📊 数据集**

33名尼泊尔10年级可选数学学生的手写答卷，共16道开放式题，涵盖矩阵、坐标几何、三角函数和函数四大主题。

**📈 对比分析**

通过加权二次Cohen's Kappa将LLM评估与双盲人工评估对比，Nova达0.385、Lyra0.269、Eagle0.103、Orion-0.026；模型间一致性最高为Lyra–Nova（0.56）。

**⚠️ 局限性**

样本量小、缺乏对LLM推理过程的质性审计、Orion规模悖论、OCR转写偏差、未验证长期一致性等限制。

---

## 318. Recommendations for Efficient and Responsible LLM Adoption within Industrial Software Development

**arXiv ID:** 2604.26590 | [PDF](https://arxiv.org/pdf/2604.26590v1)

**作者:** Krishna Ronanki `[一作]` (Chalmers University of Technology | University of Gothenburg), Christian Berger `[通讯]` (Chalmers University of Technology | University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开展了多案例研究与在线调查，针对工业软件工程中大型语言模型（LLM）的使用，提出并验证了七条实用建议；

**💡 创新点**

创新点在于基于实地案例和实证调查构建了可操作的LLM采用推荐框架，并将其与欧盟AI可信原则对照，揭示了实践与监管之间的差距；

**🔧 技术方法**

方法主要为访谈式多案例研究、反射性主题分析（reflexive thematic analysis）以及在线问卷调查；

**📊 数据集**

数据来源为三家欧洲工业组织的18名参与者的访谈文本，以及43名软件实践者的在线问卷回答；

**📈 对比分析**

通过主题分析提炼出七条建议，并在问卷中对七条建议进行Likert量表评估，结果显示超过80%的建议获得中高或强烈赞同，验证其适用性；

**⚠️ 局限性**

局限性包括案例样本量有限、受访者自报使用经验可能产生偏差、调查样本主要来自软件/IT行业、对隐私、透明度等AI可信原则覆盖不足

---

## 319. On-the-fly LTLf Synthesis under Partial Observability

**arXiv ID:** 2604.26688 | [PDF](https://arxiv.org/pdf/2604.26688v1)

**作者:** Nadav Alon `[一作]` (Open University of Israel), Shufang Zhu `[通讯]` (University of Liverpool)

**通讯引用:** 530 | [OpenAlex ID](https://openalex.org/A5101756232)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于可观测进展（observable progression）的在线（on‑the‑fly）合成框架，用于在部分可观测环境下构造信念状态 DFA 并实时求解到达游戏。

**💡 创新点**

创新点在于将全局可观测变量的子集构造与对不可观测变量的全称量化无缝集成于 DFA 构造过程，避免了传统方法中提前构建完整信念状态 DFA 的冗余，并实现了基于 MTBDD 的符号化、可观测的进展操作。

**🔧 技术方法**

主要技术包括线性时序逻辑（LTLf）公式的进展（progression）算法、可观测进展（observable progression）和子集构造、MTBDD（多终端二叉决策图）表示法、以及在线求解到达游戏的离散游戏求解算法。

**📊 数据集**

评估使用了两类数据集：一是专为部分可观测设计的 Coin Game、Traveling Target 与 Private Peek 基准集；二是将 SYNTCOMP 竞赛实例随机设为 50% 环境变量不可观测的扩展集。

**📈 对比分析**

与三种主流基线工具（基于信念、投影和 MSO 的方法）对比，提出的工具在 Coin Game 上平均 27.6 倍至 126.6 倍加速，在 SYNTCOMP‑fin 上平均 4.8 倍至 7,892.8 倍加速，显著提升了求解效率，尤其在大规模实例中表现突出。

**⚠️ 局限性**

局限性包括：对可观测变量的变量排序仍需手工调优；在某些极端情况下（如不可观测变量极多且状态空间爆炸）仍可能产生高时间/空间消耗；以及对更复杂的量化 LTL 语句（如混合 ∃/∀ 量化）目前仅通过理论讨论，尚未完整实现。

---

## 320. When Model Editing Meets Service Evolution: A Knowledge-Update Perspective for Service Recommendation

**arXiv ID:** 2604.26686 | [PDF](https://arxiv.org/pdf/2604.26686v1)

**作者:** Guodong Fan `[一作]` (Shandong Agriculture and Engineering University), Shizhan Chen `[通讯]` (Tianjin University)

**通讯引用:** 945 | [OpenAlex ID](https://openalex.org/A5012932387)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个结合模型编辑与有限自动机约束解码的演化感知服务推荐框架EVOREC，能够在不重新训练的情况下快速更新LLM对服务知识的理解，并保证生成序列的结构合法性。

**💡 创新点**

创新点在于：① 将Locate‑then‑Edit模型编辑方法（如ROME）与Trie+FA约束解码相融合，实现服务知识的局部高效更新；② 通过FA约束解码和去重机制有效抑制LLM的幻觉和冗余输出；③ 在检索增强提示下保持对最新服务的快速响应。

**🔧 技术方法**

使用技术包括：Qwen2.5‑7B 大语言模型、ROME模型编辑、检索增强提示（MiniLM‑L6‑v2 + bge‑reranker）、有限自动机（FA）与Trie约束解码、以及基于FA的去重算法。

**📊 数据集**

使用的数据集主要来自ProgrammableWeb的22,457个Web API与8,217个mashup；补充的Java/Python API与代码场景数据；以及按照发布时间划分的演化数据集，用于评估服务演化适应性。

**📈 对比分析**

与IR、MTFM++、MNT、GSR、RAG、Agent、FT等基线对比，EVOREC在Recall@5、Precision@5、mAP@10等指标上平均提升约25.9%，在演化场景下比全量微调高约22.3%，同时保持低维护成本。

**⚠️ 局限性**

局限性包括：① 约束解码可能与模型原始概率分布冲突，导致生成不连贯或需手动重约束；② 编辑数据量不足时泛化能力下降；③ 在频繁大规模更新时可能产生累积误差或回归现象。

---

## 321. MultEval: Supporting Collaborative Alignment for LLM-as-a-Judge Evaluation Criteria

**arXiv ID:** 2604.26679 | [PDF](https://arxiv.org/pdf/2604.26679v1)

**作者:** Charles Chiang `[一作]` (University of Notre Dame), Diego Gomez-Zara `[通讯]` (University of Notre Dame)

**通讯引用:** 278 | [OpenAlex ID](https://openalex.org/A5046270537)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了MultEval，一个支持多方协作制定LLM‑as‑a‑judge评估标准的Web系统，并通过案例研究验证其可行性。

**💡 创新点**

将共识构建理论嵌入提议‑评审循环、冲突诊断、角色感知历史等功能，使评估标准的协作制定可视化、可追溯、支持异步协作。

**🔧 技术方法**

前端使用React、后端Node.js；后端与Firebase、OpenAI GPT‑5交互进行LLM评估与自动冲突诊断；系统整体基于共识构建流程设计。

**📊 数据集**

案例研究使用由GPT‑4o生成的26条教学助理问答对；自动标注评估使用40条心理健康咨询聊天机器人改进方案的数据。

**📈 对比分析**

通过与人工编码的冲突分类一致性（Krippendorff's α）比较，最佳提示条件下部分类别α≥0.6；在案例中提升团队效率、减少重复工作，未给出传统数值指标。

**⚠️ 局限性**

仅在单一教学专家小组内验证，团队规模与多样性有限；功能利用受时间限制、对LLM行为感知不足；在高冲突或大规模团队下可能面临扩展性挑战。

---

## 322. FACT: Compositional Kernel Synthesis with a Three-Stage Agentic Workflow

**arXiv ID:** 2604.26666 | [PDF](https://arxiv.org/pdf/2604.26666v1)

**作者:** Sina Heidari `[一作]` (Virginia Tech), Dimitrios S. Nikolopoulos `[通讯]` (Virginia Tech)

**通讯引用:** 5488 | [OpenAlex ID](https://openalex.org/A5005410613)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FACT（Framework for Agentic CUTLASS Transpilation），一种三阶段、基于 LLM 代理的工作流，用来从 PyTorch 计算图中自动发现、实现并组合 CUTLASS 内核，以提升 GPU 推理/训练性能。

**💡 创新点**

创新点在于：①将库驱动的 kernel 合成与代理式优化相结合，避免在原始 CUDA 代码中重新发明已存在的成熟优化；②引入动态模式表（pattern table），通过代理持续积累、检索已实现的 CUTLASS 实例，形成可扩展的优化目录；③利用代理自动推断 CUTLASS API 三级搜索空间并执行参数化自调，既保证了性能，又保持了生成的可重复性。

**🔧 技术方法**

使用技术包括：大语言模型（LLM）代理（类似 OpenAI GPT/ChatGPT），PyTorch 扩展和 JIT 加载，CUTLASS 3.8.0 C++ 模板实例化，CUDA 编译与运行时调度，基于 KernelBench 的评测脚本，以及基于张量核心的多阶段流水线、Split‑K、Stream‑K 等调度策略。

**📊 数据集**

使用数据集：KernelBench 级别 1（GEMM、batched GEMM、Large‑K GEMM）和级别 3（MiniGPT transformer block）所提供的标准输入；实验平台为 NVIDIA A100‑SXM4（Ampere，SM80）。

**📈 对比分析**

通过 KernelBench 框架进行正确性验证和性能测量。对单算子 Level‑1 问题，FACT 的自调 CUTLASS 内核相比 PyTorch cuBLAS 取得 1.06×–1.18× 的加速；对 Level‑3 MiniGPT block，组合 FMHA 与 MLP+GELU 的 CUTLASS 内核实现了 2.79× 的端到端加速，显著优于 PyTorch Inductor（1.86×）和 Torch‑TensorRT（1.81×）。

**⚠️ 局限性**

局限性包括：目前仅在 Ampere 上验证，Hopper 等新架构的支持尚未完成；自调搜索空间受 CUTLASS 版本限制，某些极端形状的 GEMM 仍难以突破峰值吞吐；代理对计算图的模式匹配仍需手工规则，无法覆盖所有自定义算子；以及在编译开销和多 GPU 分布式部署方面尚未深入研究。

---

## 323. ATLAS: An Annotation Tool for Long-horizon Robotic Action Segmentation

**arXiv ID:** 2604.26637 | [PDF](https://arxiv.org/pdf/2604.26637v1)

**作者:** Sergej Stanovcic `[一作]` (Technische Universität Wien), Dongheui Lee `[通讯]` (Technische Universität Wien)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款面向长时序机器人演示的注释工具 ATLAS，支持多模态时间同步可视化和键盘中心交互，便捷标注动作边界、标签和结果。

**💡 创新点**

创新点包括：模块化数据抽象层支持多种机器人数据格式（ROS bag、RLDS、REASSEMBLE 等），键盘快捷键降低鼠标操作，提高标注效率；同时通过同步展示运动学与力学等时序信号显著提升标注精度。

**🔧 技术方法**

使用了 Python、PyQt5、PyQtGraph、OpenCV、rosbag 解析库、TensorFlow Datasets 等技术，并通过模板方法模式实现数据抽象层，采用键盘中心交互设计。

**📊 数据集**

实验使用 NIST 齿轮装配任务演示，采用 REASSEMBLE 数据格式和 RLDS 格式；对比 ELAN、ROSAnnotator 等工具。

**📈 对比分析**

通过对比四种工具的平均每动作标注时间、对齐得分和边界距离，ATLAS 在 vision+时间序列下标注时间为 18.5 s/动作，得分 99.4%，边界距离 0.06 s，显著优于 ELAN 和 ROSAnnotator。

**⚠️ 局限性**

局限性在于添加时间序列会增加认知负荷导致标注时间增长；工具仍需在更多多模态或异构数据集上验证，且键盘快捷键学习曲线对经验不足的用户可能有挑战。

---

## 324. State Beyond Appearance: Diagnosing and Improving State Consistency in Dial-Based Measurement Reading

**arXiv ID:** 2604.26614 | [PDF](https://arxiv.org/pdf/2604.26614v1)

**作者:** Yuanze Hu `[一作]` (Beihang University), Xiaotie Deng `[通讯]` (Peking University)

**通讯引用:** 10534 | [OpenAlex ID](https://openalex.org/A5100638710)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TriSCA三层对齐框架，针对多模态大语言模型在拨盘式测量读取任务中的状态一致性问题，改进表示、推理与目标；

**💡 创新点**

将状态一致性拆解为表示层、推理层、目标层的三重对齐，提出状态距离感知对比学习、元数据引导的观察到状态监督以及连续状态奖励等创新模块；

**🔧 技术方法**

采用状态距离感知的三元组对比损失、元数据引导的监督模板、GRPO策略优化、连续状态奖励函数、视觉塔对齐等技术；

**📊 数据集**

使用合成时钟与仪表（Controlled Clock & Gauge）数据集以及 MeasureBench 实际图像集合，并使用 1273 张真实 dial 图像进行实验；

**📈 对比分析**

在 Clean、View、Illum、Combined 四种扰动条件下对比基线，TriSCA 在 Combined 上提升约 20%，在 MeasureBench Dial 类别提升 16.5%；连续误差 MAE 下降，整体精度与鲁棒性显著提升；

**⚠️ 局限性**

仅针对拨盘类仪表，强扰动下性能仍有限；未扩展至线性、复合显示等其他仪表类别，也未验证对其他任务的广泛影响。

---

## 325. Translating Under Pressure: Domain-Aware LLMs for Crisis Communication

**arXiv ID:** 2604.26597 | [PDF](https://arxiv.org/pdf/2604.26597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 326. Who Trains Matters: Federated Learning under Enrollment and Participation Selection Biases

**arXiv ID:** 2604.26604 | [PDF](https://arxiv.org/pdf/2604.26604v1)

**作者:** Gota Morishita `[一作]` (University of Melbourne), Gota Morishita `[通讯]` (University of Melbourne)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5031619183)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出两阶段联邦学习选择框架，设计FedIPW逆概率加权聚合和有限信息聚合校准方法，用以纠正入选与参与带来的偏差。

**💡 创新点**

① 明确分离入选与参与两阶段偏差；② 在两阶段因果结构下推导IPW估计器并证明无偏；③ 分析残差加权误差导致的非消失偏差；④ 设计仅使用聚合统计的校准方案。

**🔧 技术方法**

因果图建模、逆概率加权(IPW)、加权校准、强凸优化误差分析、合成联邦逻辑回归实验。

**📊 数据集**

使用合成联邦逻辑回归数据，模拟预入选/预轮协变量与选择机制，能够精确计算目标最优。

**📈 对比分析**

与FedAvg、仅参与IPW、Oracle IPW进行对比；实验显示FedIPW近似Oracle，显著降低目标误差；仅参与IPW误差随入选偏差增大；聚合校准在有聚合统计时可部分缓解。

**⚠️ 局限性**

需要对入选概率有准确估计或可获得个人级协变量；有限信息校准仅能部分纠正偏差；理论分析基于强凸，非凸情况尚未覆盖。

---

## 327. Rule-based High-Level Coaching for Goal-Conditioned Reinforcement Learning in Search-and-Rescue UAV Missions Under Limited-Simulation Training

**arXiv ID:** 2604.26833 | [PDF](https://arxiv.org/pdf/2604.26833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 328. Hearing the Room Through the Shape of the Drum: Modal-Guided Sound Recovery from Multi-Point Surface Vibrations

**arXiv ID:** 2604.26678 | [PDF](https://arxiv.org/pdf/2604.26678v1)

**作者:** Shai Bagon `[一作]` (Weizmann Institute of Science), Mark Sheinin `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 163 | [OpenAlex ID](https://openalex.org/A5079263739)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于光学散斑振动测量的多点声恢复框架，融合多个表面点的振动信号来重建场景声音。

**💡 创新点**

通过物理导向的模态振动模型，将散斑位移与声源耦合，利用模态频率和形状对多点信号进行相位和幅度校正，实现对低振幅、共振对象的高质量声音恢复。

**🔧 技术方法**

使用 speckle vibrometry、模态分析、物理导向的振动模型、逆滤波与优化求解、PCLK+相位解码以及光学激光网格采样等技术。

**📊 数据集**

实验采用多种日常固体物体（如笔记本、书包、吉他、鼓、画框、瑜伽块等）与多段音乐及敲击声，使用22000fps或44100fps的高速相机捕获散斑图像。

**📈 对比分析**

与单点测量、简单平均、延迟求和（delay‑and‑sum）以及基于校准的最优恢复方法比较，实验表明多点模态融合显著提升信噪比和频谱均匀性，优于传统方法且接近校准基线。

**⚠️ 局限性**

受限于模态提取的准确性、频率上限（高频模态难检出）、光学测量的几何畸变，以及对柔软或非平面物体的恢复效果有限。

---

## 329. Exploring the Efficiency of 3D-Stacked AI Chip Architecture for LLM Inference with Voxel

**arXiv ID:** 2604.26821 | [PDF](https://arxiv.org/pdf/2604.26821v1)

**作者:** Yiqi Liu `[一作]`, Jian Huang `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文开发了一套名为""的编译器感知、端到端模拟框架，用于评估3D堆叠AI芯片在LLM推理中的效率，并对不同软硬件配置进行了系统探索。

**💡 创新点**

创新点在于：①提出了可与ML编译器协同工作的API，支持自定义计算范式、tile映射、张量-银行映射等；②设计了高效的DRAM访问模式识别与缓存技术；③在3D堆叠架构上首次开展了大规模软硬件协同设计空间探索，揭示多因素耦合的性能瓶颈。

**🔧 技术方法**

主要技术包括：事件驱动的分布式硬件模拟、张量与张量块的可编程映射、基于匹配键的DRAM请求合并加速、热密度限制下的动态频率调节，以及与Graphcore IPU仿真器的交叉验证。

**📊 数据集**

实验使用了多种大型语言模型（Llama2‑13B、Gemma2‑27B、OPT‑30B、Llama3‑70B）和视觉Transformer模型DiT‑XL，并在这些模型上跑了预填（prefill）和解码（decode）两种工作负载。

**📈 对比分析**

通过与IPU硬件仿真器的对比，模拟误差低于7%（平均12.7%），在不同计算范式、NoC拓扑、张量-银行映射等配置下，展示了最高可达1.84×的性能提升；同时证明了软件感知的张量映射可将行缓冲冲突降低至≈15%。

**⚠️ 局限性**

局限性包括：缺乏真实3D堆叠AI芯片硬件，仅以IPU为代理；热模型简化（仅限功率密度阈值）；主要聚焦LLM推理，未覆盖训练阶段；并且模拟框架尚未公开完整实现。

---

## 330. Semi-supervised learning with max-margin graph cuts

**arXiv ID:** 2604.26818 | [PDF](https://arxiv.org/pdf/2604.26818v1)

**作者:** Branislav Kveton `[一作]` (Intel), Ling Huang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于和谐函数预测标签的最大间隔图割（max‑margin graph cuts）半监督学习算法。

**💡 创新点**

创新点在于先用正则化的和谐函数得到软标签，再以此作为条件训练最大间隔判别器，形成两阶段凸优化，兼具核技巧、稀疏图结构以及可推导的泛化误差上界。

**🔧 技术方法**

使用了正则化的和谐函数求解、图拉普拉斯正则化、SVM/核方法（线性、三次多项式、RBF）、阈值筛选、以及算法稳定性与泛化误差理论。

**📊 数据集**

实验数据包括合成二维问题以及三个 UCI 公开数据集：Letter Recognition、Digit Recognition、Image Segmentation。

**📈 对比分析**

通过与 Manifold Regularization of SVMs（Belkin 等）以及传统监督 SVMs 进行交叉验证调参比较，在合成问题中优于 MR SVM，在 UCI 数据集上 1%–10% 标记比例下 29/36 次实验中取得更低误差，尤其在线性和三次多项式核上表现最佳。

**⚠️ 局限性**

限制在于理论分析假设软标签但实际采用硬标签；γ_g 参数需要验证选择；阈值设置对极度不确定样本敏感；对大规模高维数据的可扩展性尚未深入探讨。

---

## 331. What Is the Cost of Energy Monitoring? An Empirical Study on the Overhead of RAPL-Based Tools

**arXiv ID:** 2604.26815 | [PDF](https://arxiv.org/pdf/2604.26815v1)

**作者:** Jeremy Diamond `[一作]` (Universität Zürich), Vincenzo Stoico `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 99 | [OpenAlex ID](https://openalex.org/A5075101570)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在 1 kHz 高频率下使用 RAPL 接口进行能耗监测时，工具自身产生的时间与能耗开销，并提出两种简化实现以降低此类开销。

**💡 创新点**

创新点在于通过对七款现有工具与自研两款工具的 NAS 基准对比，系统量化高频采样导致的时间/能耗开销，并在微基准层面测量 MSR 读取、系统调用等关键指令的延迟，进而提出采样循环简化和内核缓冲的改进方案。

**🔧 技术方法**

使用了 RAPL MSR 读取、powercap 文件接口、perf events、系统调用、rdmsr 指令以及自研的内核模块和用户空间工具；实验自动化采用 Python 实验跑者和 NixOS 发行版。

**📊 数据集**

数据集为 NAS Parallel Benchmarks（bt、cg、ft、mg、ep、is）六个函数，作为基准测试。

**📈 对比分析**

实验采用全因子设计、每组 15 次随机重复；使用 Shapiro‑Wilk 检验正态性、Kruskal‑Wallis、Dunn‑Bonferroni 检验和 Cliff’s delta 评估差异；结果显示自研工具 R_K 与 R_U 与无工具基线几乎无开销，而现有工具如 Scaphandre、CodeCarbon 在 1 kHz 下开销可高达 46% 以上。

**⚠️ 局限性**

局限性包括实验仅在单台 Intel NUC8i7HVK/NixOS 环境下进行，缺乏跨架构/跨 OS 的验证；仅测量时间与能耗，未深入能耗‑能效权衡；基准受限于 NAS，可能不完全代表一般软件工作负载。

---

## 332. MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter Searching

**arXiv ID:** 2604.26799 | [PDF](https://arxiv.org/pdf/2604.26799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 333. Asynchronous Federated Unlearning with Invariance Calibration for Medical Imaging

**arXiv ID:** 2604.26809 | [PDF](https://arxiv.org/pdf/2604.26809v1)

**作者:** Zhaoyuan Cai `[一作]` (South China University of Technology), Xinglin Zhang `[通讯]` (South China University of Technology)

**通讯引用:** 4610 | [OpenAlex ID](https://openalex.org/A5019726320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种异步联邦“忘记”框架AFU-IC，结合服务器端KL不变性校准实现对目标客户端数据的彻底删除；

**💡 创新点**

首次在医疗联邦学习中引入异步unlearning，消除了同步等待瓶颈，并通过KL校准实现结构化、持久的删除；

**🔧 技术方法**

使用异步梯度上升+参考模型约束、服务器端KL距离不变性校准，并采用Dirichlet划分的非IID数据和backdoor注入验证；

**📊 数据集**

在OASIS、PathMNIST和OrganAMNIST三大医疗影像基准上进行实验；

**📈 对比分析**

与同步基线（PGA、FedRecovery、FedEraser、FedOSD）和完整重训比较，AFU-IC在去背门效能（BA）与保真度（CA）均接近重训，且wall‑clock时间缩短4–18倍；

**⚠️ 局限性**

在极端非IID或目标客户端贡献占比极大时，保真度略有下降，且需要对γ_calib做适当调节，服务器端校准的计算负担仍未完全消除。

---

## 334. MISES: Minimal Information Sufficiency for Effective Service

**arXiv ID:** 2604.26808 | [PDF](https://arxiv.org/pdf/2604.26808v1)

**作者:** Joss Armstrong `[一作]` `[通讯]` (Ericsson Ireland), Joss Armstrong (Ericsson Ireland)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于服务类别的网络资源协调机制，分析了福利损失、激励兼容性与检测性能，并给出了信息量的下限与可行性区间。

**💡 创新点**

创新点在于：①提出福利与误报的紧致双侧界限；②给出统一的 ε‑IC 上限，证明需求派生类别可同时最小化福利损失与作弊激励；③证明在满足 Condition C 的情况下，聚合指标检测显著优于单个代理检测；④用信息论与率失真理论量化实现目标所需的最小类别熵，从而得到可行性带。

**🔧 技术方法**

使用了凸分析与强凹性/平滑性假设来推导福利界限；Rao‑Blackwell 与 Neyman‑Pearson 理论证明聚合检测优势；信息论与率失真理论给出信息预算下的下界；k‑means 聚类生成需求派生类别；数值仿真与真实 PM 计数器数据验证理论。

**📊 数据集**

实验数据包括：①合成的 50,000 维度为 4 的需求向量；②来自四个匿名运营商的 28,249 个小区的性能管理计数器，覆盖 5 周。

**📈 对比分析**

通过比较不同 K 的福利差距 Δ、聚合与单体检测的召回率与误报率，发现：福利随 K 增加而下降；聚合检测在无流量追踪条件下仍能获得显著高召回；但当 K 过大时检测召回趋于平稳，显示出福利与检测目标的对立。

**⚠️ 局限性**

局限性：①机制假设满足 Condition C（聚合指标为充分统计量），若需求异质性较大需细化类别；②检测分析基于高斯噪声模型，非 Gaussian 环境下可能失效；③信息预算的上限与下限为理论极限，实际部署需根据网络几何与采样率调整；④未考虑多租户共享协调器时的协作与安全问题。

---

## 335. Bian Que: An Agentic Framework with Flexible Skill Arrangement for Online System Operations

**arXiv ID:** 2604.26805 | [PDF](https://arxiv.org/pdf/2604.26805v1)

**作者:** Bochao Liu `[一作]` (Kuaishou Technology), Xiao Liang `[通讯]` (Kuaishou Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于大型语言模型的运维框架 Bian Que，能够自动化处理在线引擎系统的发布拦截、主动巡检和告警根因分析。

**💡 创新点**

创新点包括：①统一运维范式将三类任务抽象为可复用的 Agent；②灵活的 Skill 结构，利用 LLM 自动生成、更新并可通过自然语言指令人类修正；③单一反馈信号同时驱动知识库蒸馏与 Skill 细化，实现知识与上下文映射的协同进化。

**🔧 技术方法**

核心技术包括：大型语言模型（如 Qwen3.5‑35B）、LLM 生成/更新 Skill 的 Prompt，KV/KKV/向量索引的知识检索，事件到数据/知识的动态匹配机制，以及在线案例记忆与日常知识蒸馏。

**📊 数据集**

数据集主要为真实业务运营事件，包含 104 条发布拦截/巡检/告警事件，标注由资深 SRE 生成；实验还使用了 36 条告警根因分析事件进行 LLM 规模对比。

**📈 对比分析**

对比方法包括离线 Pass@k、在线报警量减少、RCA 准确率、MTTR 等指标；在实际部署上，告警触发量下降 75%，可操作告警降 95%，RCA 准确率 80%，MTTR 降 50%+；在实验中 Pass@5 最高可达 99%。

**⚠️ 局限性**

局限性：①未自动化补救动作；② Skill 匹配仍基于关键词，可改为更智能路由；③多 Agent 协同与跨事件协调尚未实现；④知识与 Skill 共同进化效果难以量化；⑤对新上线服务的知识积累仍有不足。

---

## 336. Hankel and Toeplitz Rank-1 Decomposition of Arbitrary Matrices with Applications to Signal Direction-of-Arrival Estimation

**arXiv ID:** 2604.26787 | [PDF](https://arxiv.org/pdf/2604.26787v1)

**作者:** Georgios I. Orfanidis `[一作]` (Florida Atlantic University), Elizabeth Serena Bentley `[通讯]` (Air Force Research Laboratory)

**通讯引用:** 833 | [OpenAlex ID](https://openalex.org/A5071876359)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并实现了针对任意矩阵的最优秩‑1 汉克尔/托普利茨结构逼近算法，并将其应用于少量样本、受限 RF 链条的方向估计（DoA）问题。

**💡 创新点**

创新点：①提出在 L₂ 与 L₁ 误差下可解析或高效数值求解的秩‑1 汉克尔/托普利茨逼近；②证明 L₂ 估计在高斯噪声下为 MLE，L₁ 估计在拉普拉斯噪声下为 MLE；③在有限样本、受限硬件、脉冲噪声与硬件失效等严苛环境下实现显著性能提升。

**🔧 技术方法**

主要技术：结构化低秩逼近、汉克尔/托普利茨矩阵结构投影、极值搜索与 Weiszfeld 算法、滑动窗口子阵列采样、经典 DoA 方法（MUSIC、ESPRIT、Matrix Pencil、Hankel‑MUSIC、FB‑SS MUSIC）作基准。

**📊 数据集**

数据集：仿真数据（不同阵列尺寸、白噪声与伯努利‑高斯脉冲噪声），以及公开 UAV 5×8 URA 真实测量数据。

**📈 对比分析**

与 Matrix Pencil、Hankel‑MUSIC、FB‑SS MUSIC、能量最大化、Toeplitz 协方差 MUSIC 等方法对比：在白噪声/高 SNR 条件下 L₂ 估计误差可低于 0.1°；在脉冲噪声或硬件失效环境下 L₁ 估计误差比其他方法下降约一到两位数，显示最佳稳健性。

**⚠️ 局限性**

局限性：仅针对秩‑1 结构，尚未推广到更高秩；L₁ 版算法复杂度高（O(Δρ⁻²Δϕ⁻¹ D²W²T)）；假设滑动窗口子阵列能覆盖全阵列，且对非线性/动态系统的适用性需进一步验证。

---

## 337. Virtual-reality based patient-specific simulation of spine surgical procedures: A fast, highly automated and high-fidelity system for surgical education and planning

**arXiv ID:** 2604.26781 | [PDF](https://arxiv.org/pdf/2604.26781v1)

**作者:** Raj Kumar Ranabhat `[一作]` (Sunnybrook Research Institute), Michael Hardisty `[通讯]` (Sunnybrook Research Institute)

**通讯引用:** 1009 | [OpenAlex ID](https://openalex.org/A5051263805)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一个基于CT和MRI自动生成患者特定脊柱三维模型并在虚拟现实中进行脊柱减压手术仿真的快速全自动化系统。

**💡 创新点**

结合多模态影像融合与深度学习分割、变形配准，实现仅约2.5分钟内完成高保真患者特定模型，支持实时交互式切除模拟，弥补传统VR只做可视化的缺陷。

**🔧 技术方法**

使用深度学习分割（VertDetect、TotalSegmentator、nnU-Net）、基于MIND的连续变形配准、3D Slicer、SieVRt VR平台与自定义模块，并通过Dice相似系数与TRE评估模型质量。

**📊 数据集**

使用15例临床患者CT/MRI影像（腰椎、颈椎、颈胸椎、胸椎），并结合公开VerSe和nnU-Net训练集进行模型训练与验证。

**📈 对比分析**

通过与人工标注的Dice相似系数对比：骨骼0.95、椎间盘0.87、神经结构0.92；注册误差平均1.73 mm；总模型生成时间约155 秒；访谈显示空间感知和手术自信度提升。

**⚠️ 局限性**

样本量有限且为单中心；软组织如神经根和棘韧带仍需人工分割；缺乏客观学习效果和临床结果验证；系统尚未集成触觉反馈等功能。

---

## 338. Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding

**arXiv ID:** 2604.26779 | [PDF](https://arxiv.org/pdf/2604.26779v1)

**作者:** Hayate Iso `[一作]`, Bita Rouhani `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 NeMo RL 框架中集成了可保持验证器精确性的投机式解码（Speculative Decoding）作为 RL 后训练的加速原语，提升了大规模语言模型的回合生成吞吐量。

**💡 创新点**

创新点在于将投机式解码与 RL 训练的权重同步、在线草稿适配、异步与同步流水线等系统级机制无缝耦合，既保持了原始采样分布，又在大规模部署中实现了可观的加速。

**🔧 技术方法**

核心技术包括 EAGLE‑3 草稿模型、vLLM 推理后端、NeMo RL 的权重同步与日志概率重算、FP8 精度计算、GPU 拆分与异步执行等。

**📊 数据集**

实验使用 Qwen3‑8B、Qwen3‑8B‑Base 以及 DAPO‑Math‑17K 训练集和 AIME‑2024 验证集，评估了不同草稿初始化、草稿长度和异步设置下的效果。

**📈 对比分析**

与自回归解码对比，投机式解码在 8B 模型上生成延迟提升 1.5‑1.8×，整体 RL 步速提升 1.35‑1.41×；仿真预测在 235B 模型下可达 3.5×回合生成加速和 2.5×端到端训练加速。

**⚠️ 局限性**

局限性包括对草稿初始化与匹配度高度敏感，草稿长度与接受率的权衡复杂；在异步流水线中加速受限；系统实现复杂，需细致权重同步和负载平衡配置。

---

## 339. MemOVCD: Training-Free Open-Vocabulary Change Detection via Cross-Temporal Memory Reasoning and Global-Local Adaptive Rectification

**arXiv ID:** 2604.26774 | [PDF](https://arxiv.org/pdf/2604.26774v1)

**作者:** Zuzheng Kuang `[一作]` (Xi'an Jiaotong University), Haixia Bi `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 244 | [OpenAlex ID](https://openalex.org/A5039966032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的开放词汇变化检测框架MemOVCD，利用跨时间记忆推理与全局‑局部自适应校正实现对双时遥感影像的语义变化识别。

**💡 创新点**

创新点包括：① 将变化检测重新表述为双帧追踪问题，引入跨时间记忆机制实现语义推理的前后时序耦合；② 通过直方图对齐的过渡帧桥接平滑大时间间隙的外观跳变；③ 采用全局‑局部自适应融合，补偿基于补丁推理导致的空间碎片化。

**🔧 技术方法**

核心技术主要是：SAM‑3 视觉跟踪模型、DINO/CLIP 预训练视觉语言模型、图像金字塔与实例分割、直方图匹配与线性插值、8连通分支自适应加权融合。

**📊 数据集**

在五个公开基准上进行评估：LEVIR‑CD、DSIFN、S2Looking、BANDON（建筑变化）以及 SECOND（土地覆被语义变化）。

**📈 对比分析**

与现有训练‑自由和训练‑有监督方法比较，MemOVCD 在 LEVIR‑CD、S2Looking、BANDON 上取得最优或接近最优（例如 LEVIR‑CD 72.5 IoU/84.1 F1），在 DSIFN 上排名第二，整体性能显著优于同类方法，并且保持了对多种场景的鲁棒性。

**⚠️ 局限性**

局限性包括：对极端时间间隔或剧烈光照变化仍可能产生记忆传播失真；过渡帧数量的选择仍需经验调参；基于补丁推理的计算开销较大；依赖大型预训练模型，对资源有限的应用场景可能受限。

---

## 340. FeatureFox: Sample-Efficient Panoptic Graph Segmentation for Machining Feature Recognition in B-Rep 3D-CAD Models

**arXiv ID:** 2604.26770 | [PDF](https://arxiv.org/pdf/2604.26770v1)

**作者:** Bertram Fuchs `[一作]` (Technische Universität Berlin), Oliver Lohse `[通讯]` (Siemens AG)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 FeatureFox，一种基于面邻接图和树模型的面向实例与语义的 panoptic 图分割管线，用于自动特征识别（AFR）；

**💡 创新点**

将 AFR 重新表述为 panoptic segmentation 问题，设计层级化、可解释的随机森林管线；在样本/计算效率上显著优于深度模型，并统一采用 PQ 评估；

**🔧 技术方法**

使用随机森林（边界二分类与实例级语义分类）、面/边属性提取、面邻接图构造、连通分量聚合、概率校准与统计特征聚合；

**📊 数据集**

主要数据集为 MFInstSeg（≈62k 个 STEP/NX CAD 文件）；另外使用 270 个手工标注的工业 NX 文件进行工业验证，并在 NIST 公开 3D‑CAD 文件上测试泛化；

**📈 对比分析**

与 AAGNet、MFTReNet 等深度 GNN 模型对比，FeatureFox 在低样本（约250份）即可达到 PQ>0.9；在全数据集上 AAGNet 略胜，但 FeatureFox 在完整实例识别率和训练时间（秒级）上更优；

**⚠️ 局限性**

非端到端管线，边界误差会直接影响语义分类；对复杂交叉或多步特征（非连续面集合）处理不足；在边界模糊或稀疏数据下鲁棒性下降。

---

## 341. Factorized Latent Reasoning for LLM-based Recommendation

**arXiv ID:** 2604.26760 | [PDF](https://arxiv.org/pdf/2604.26760v1)

**作者:** Tianqi Gao `[一作]` (Independent Researcher), Lina Yao `[通讯]` (University of New South Wales)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Factorized Latent Reasoning (FLR) 框架，在 LLM 的顺序推荐任务中使用多因子潜在推理，替代传统单向编码与显式链式推理。

**💡 创新点**

核心创新在于将用户意图拆分为若干解耦的潜在因子，并通过正交约束、注意力多样性约束和稀疏性约束实现因子分离；同时引入 GRPO 强化学习在因子化潜在空间上优化混合奖励，实现无显式推理步骤的高效训练。

**🔧 技术方法**

技术手段包括 Qwen2.5‑1.5B LLM 作为基准模型；多头因子化潜在推理模块；正交、注意力多样性和稀疏性正则化；Group Relative Policy Optimization (GRPO) 强化学习；基于前缀树的生成式排序与混合奖励设计。

**📊 数据集**

使用 Amazon 评测数据集的四个子域：Toys、CDs、Games、Instruments，分别进行时间滑动窗口和 5‑core 筛选后按 8:1:1 划分。

**📈 对比分析**

与传统顺序模型（Caser、GRU4Rec、SASRec）及 LLM 基线（Base、CoT、AlphaRec、BIGRec、LatentR3）进行多指标（HR@K、NDCG@K）比较。FLR 在所有数据集上均取得 state‑of‑the‑art，最高相对提升达 84.6%（相较于最强基线），在多面向兴趣域（Toys、CDs、Games）表现尤为突出。

**⚠️ 局限性**

局限性包括：对因子数量 K 的选择仍需经验调优，某些专业领域（如 Instruments）多因子并未显著提升；正则化项调参复杂；尽管比显式链式推理更高效，但相较于无推理 LLM 仍有轻微推理开销；对极端长尾项目的提升有限。

---

## 342. The Nesting Bird Box Problem is ER-complete: Sharp Hardness Results for the Hidden Set Problem

**arXiv ID:** 2604.26749 | [PDF](https://arxiv.org/pdf/2604.26749v1)

**作者:** Lucas Meijer `[一作]` (Utrecht University), Miloš Stojaković `[通讯]` (University of Novi Sad)

**通讯引用:** 518 | [OpenAlex ID](https://openalex.org/A5054524255)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究了在多边形域内放置k个“鸟巢”点，使得任意两点不可互相“看到”，即可视图中的独立集问题，并证明该问题属于存在实数理论（Existential Theory of the Real, ER）难度，亦即ER‑complete。

**💡 创新点**

创新点在于提供了一条更短、更易理解的ER‑完整性证明：利用连续约束满足问题与多边形含洞结构，简化了原先的复杂构造；引入了“阻挡器”（blocker）与“观察者”（spectator）机制，使得多变量与约束之间的相互作用可通过几何手段精确编码。

**🔧 技术方法**

主要技术包括：1) 通过可视性图定义开放可视性（open‑visibility）；2) 构造多种几何 gadget（变量、复制/缩放、加法、曲线）以及阻挡器与观察者；3) 将问题转化为 ER 公式或在 Real‑RAM 上的多项式时间验证；4) 使用连续约束满足（ETR(h)）作为下界来源。

**📊 数据集**

由于本文为理论复杂性研究，未使用任何实验数据集；所有论证均在几何构造与理论推导层面完成。

**📈 对比分析**

评估方式基于计算复杂度分类而非实验性能；论文证明该问题的决策版本属于ER‑complete，说明其难度高于NP‑complete问题，且若想在多项式时间内求解需求解 ER 公式。

**⚠️ 局限性**

局限性包括：1) 只证明了含洞多边形域的情况，对简单多边形仍为开放问题；2) 采用了开放可视性定义，普通可视性下的复杂度仍未知；3) 需要构造特殊的障碍和观察者，实际实现或多边形设计可能受限；4) 证明依赖于多边形可分割成若干凸可见子域的可覆盖性，非通用。

---

## 343. Analytically Characterized Optimal Power Control for Signal-Level-Integrated Sensing, Computing and Communication in Federated Learning

**arXiv ID:** 2604.26741 | [PDF](https://arxiv.org/pdf/2604.26741v1)

**作者:** Paul Zheng `[一作]` (RWTH Aachen University), Anke Schmeink `[通讯]` (RWTH Aachen University)

**通讯引用:** 4024 | [OpenAlex ID](https://openalex.org/A5072895220)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种联合功率控制与接收缩放设计，用于在联邦学习中实现信号级联感知、计算与通信（Sig‑ISCC），并满足联合目标检测约束。

**💡 创新点**

创新点在于将原始非凸功率控制问题通过变量变换等价转为凸形式，利用 KKT 条件解析出最优结构，并构造只需求解单调可导函数根的多项式时间算法，实现对 Sig‑ISCC 的全局最优、数值鲁棒的功率分配。

**🔧 技术方法**

采用的技术包括 AirComp 基础模型、信号级目标检测理论、凸优化及根寻根方法、以及对噪声与误差的解析。

**📊 数据集**

实验使用 MNIST 与 CIFAR‑10 数据集，分别构建全连接与 ResNet‑20 模型进行联邦学习。

**📈 对比分析**

与 IPOPT（非凸、凸）、零逼迫、贪心感知功率等基线相比，所提算法在 MSE 上最低，FL 终端精度几乎达到无感知下的下界，且计算时间比传统求解器快约十倍。

**⚠️ 局限性**

局限性包括对单天线基站、完全信息估计的假设、仅考虑静态目标与设备布局，算法复杂度为 O(K³)，在大规模设备群时仍可能产生开销。

---

## 344. Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving

**arXiv ID:** 2604.26837 | [PDF](https://arxiv.org/pdf/2604.26837v1)

**作者:** Zihan Zhao `[一作]` (University of Virginia), Fan Yang `[通讯]` (Microsoft Research)

**通讯引用:** 5812 | [OpenAlex ID](https://openalex.org/A5045464812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的稀疏注意力推理框架（Spin），通过分区抽象、局部性感知 KV 缓存管理和多层级元数据设计，实现了在长上下文 LLM 推理中高效利用稀疏注意力的系统级性能提升。

**💡 创新点**

创新点包括：①把稀疏注意力的不同粒度抽象为统一的分区操作，形成五阶段流水线；②设计局部性感知的 GPU–CPU KV 缓存管理，动态分配缓冲区并采用桶化 LRU 提高缓存命中率；③采用两级页表与 CPU‑GPU 元数据分层，显著降低元数据占用，使系统能在极大上下文下保持高吞吐。

**🔧 技术方法**

技术实现基于 vLLM，使用 C++/CUDA 核心；实现 warp‑级别 PCIe 传输、persistent kernel、GPU‑友好 LRU、分区映射表、Tier‑Split 元数据、按需分配 GPU 缓冲区等。

**📊 数据集**

使用长上下文基准 LongBench‑v2 与 LongGenBench，模型包括 Qwen3‑14B、Qwen3‑32B 与 Llama‑3.1‑70B，评估了不同 GPU（A100、B200）和不同请求率下的性能。

**📈 对比分析**

与 vLLM、vLLM‑Offload 以及原始稀疏算法实现进行对比；Spin 在在线推理中平均实现 1.66–5.66 倍的吞吐提升，TTFT 降低 7–9 倍，且对原始稀疏实现平均降低 21–58% 的 per‑token decode  延时，提升至原实现的 1.34–2.39 倍吞吐。

**⚠️ 局限性**

局限性：仍受 PCIe 带宽限制，稀疏度高低决定收益；对极大上下文或低稀疏度模型效果有限；实现复杂度高，需针对新算法实现索引/选择逻辑；在单 GPU 的低算力场景下收益不明显。

---

## 345. A Test Taxonomy and Continuous Integration Ecosystem for Dynamic Resource Management in HPC

**arXiv ID:** 2604.26824 | [PDF](https://arxiv.org/pdf/2604.26824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 346. Weighted Emulators with Local Heaviest Edges Stretch for Undirected Graphs

**arXiv ID:** 2604.26831 | [PDF](https://arxiv.org/pdf/2604.26831v1)

**作者:** Liam Roditty `[一作]`, Ariel Sapir `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种通用框架，构造了满足大小为 Õ(n^{1+1/k}) 边数的加权图混合逼近（emulator），其距离上界依赖于最重边 W₁ 和第二重边 W₂；该框架将之前的 +2W₁ 、+4W₁ 等构造统一并进一步推广。

**💡 创新点**

创新点主要有：①首次在加权 emulator 的伸缩项中引入第二重边 W₂ ，显著提升了局部加权误差；②利用新的辅助边集合 B₂S₁ 和更细粒度的“批次”分析，避免了乘法误差的指数增长；③在无权图情形下得到比 Thorup–Zwick 更紧的短距离逼近；④通过统一的三类案例（a、b、c）实现了对任意 k 的完整证明。

**🔧 技术方法**

核心技术包括：层级采样（hitting set）构造多级 pivots；边集 D、E₁、B₁V、B₂S₁ 与笛卡尔乘积 S_{i-1}×S_{k-i} 的组合；递归距离上界分析与 W₁、W₂ 的显式使用；以及对距离阈值的多项式不等式分析来确定优越区间。

**📊 数据集**

文中未给出具体实验数据集，仅在理论上讨论了不同 k 下的边数与距离阈值；实验验证与实现细节未出现。

**📈 对比分析**

与现有最优加权 emulator（如 +4W₁ 、+6W₁）和无权图 Thorup–Zwick emulator 进行对比：在大多数实际距离范围（δ ≤ O(3^{k²})）内，该框架提供更小的加权误差；在边数方面保持 Õ(n^{1+1/k}) 的稀疏度；对于更大距离，Thorup–Zwick 的渐进伸缩更优，但该框架在短距离上明显占优。

**⚠️ 局限性**

局限性包括：①构造依赖随机采样，需期望分析；②仅在加权图中使用 W₂ ，尚未探索更高阶权重的潜在优势；③乘法误差随 k 增大仍不可避免，导致对大 k 值的逼近效果下降；④未给出动态更新或分布式实现的支持；⑤在证明中涉及的复杂案例分析对实现和可读性有一定挑战。

---

## 347. Population Dynamics in ARIEL Robotics Systems Featuring Embodied Evolution via Spatial Mating Mechanisms

**arXiv ID:** 2604.26822 | [PDF](https://arxiv.org/pdf/2604.26822v1)

**作者:** Victoria Peterson `[一作]` (Vrije Universiteit), Raghav Prabhakar `[通讯]` (Vrije Universiteit)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在 2D 物理模拟环境中，设计并实现了一个空间嵌入的进化算法，让四足机器人通过物理交互实现繁殖和淘汰，研究空间结构对进化动力学的影响。

**💡 创新点**

创新点在于：①将父代选择改为基于空间邻近、分区区域以及动态迁移的三种空间策略；②引入事件驱动的交配区迁移机制；③结合能量、密度和年龄等多种死亡选择，探究其在空间环境中的平衡与相变；④对不同选择策略下的种群动态做了相变分析，揭示了临界点和双稳态。

**🔧 技术方法**

使用了 HyperNEAT（通过 CPPN 生成控制器）、MuJoCo 物理仿真、周期性边界条件、基于距离的配对与迁移算法、能量与密度驱动的死亡概率模型、以及统计分析工具对种群行为进行跟踪。

**📊 数据集**

采用的是在虚拟 25 m × 25 m 平面上随机放置 30 个固定四足机器人进行 30–60 秒的交配仿真；并无真实数据集，所有实验均在模拟器中生成。

**📈 对比分析**

与传统随机配对/面向适应度的死亡选择相比，基于空间的配对略有提升（最高峰值 4.9%），但在能量/密度死亡选择下出现种群爆炸或灭绝。通过多参数网格搜索与 10–48 次独立跑，实验展示了不同策略导致的稳定、无穷大或零种群，说明仅靠传统死亡机制无法在空间进化中维持平衡。

**⚠️ 局限性**

限制包括：固定机器人形态与环境，参数空间探索受计算资源限制；实验仅至 100 代，可能忽略长期动态；纯模拟结果未在真实机器人上验证；死亡机制与繁殖耦合过于单一，导致产生对抗性激励。

---

## 348. A proof of Jordan curve theorem based on the sweepline algorithm for trapezoidal decomposition of a polygon

**arXiv ID:** 2604.26812 | [PDF](https://arxiv.org/pdf/2604.26812v1)

**作者:** Apurva Mudgal `[一作]` (Indian Institute of Technology Ropar), Apurva Mudgal `[通讯]` (Indian Institute of Technology Ropar)

**通讯引用:** 186 | [OpenAlex ID](https://openalex.org/A5043166956)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出并实现了一种基于sweepline算法的递归树结构，用以构造任意Jordan曲线的内部区域；

**💡 创新点**

创新点在于通过对无穷递归树的Zorn极大链理论构造，首次用集合论技术（Zorn引理）证明了Jordan曲线内部区域的存在与连通性；

**🔧 技术方法**

核心技术包括水平扫线（sweepline）算法、有限与无限射线的拓扑结构分析、弱Jordan曲线定理与无限矩形/多边形的扩展、以及Zorn引理的偏序关系构造；

**📊 数据集**

该研究为纯理论数学证明，不使用任何实验数据集；

**📈 对比分析**

方法通过与传统平面图形分割与拓扑连通性分析技术对比，证明了构造的内部区域始终是连通且边界为唯一Jordan曲线，理论上无可比性能指标；

**⚠️ 局限性**

局限性在于证明仅适用于二维平面Jordan曲线，且对无限矩形多边形的弱Jordan曲线定理依赖假设，未给出算法实现细节或复杂度分析。

---

## 349. Input Distribution Design for Ranging-Oriented OFDM-ISAC Systems Under Frequency-Selective Fading

**arXiv ID:** 2604.26778 | [PDF](https://arxiv.org/pdf/2604.26778v1)

**作者:** Weijiang Zhao `[一作]` (Beijing University of Posts and Telecommunications), Yifeng Xiong `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1506 | [OpenAlex ID](https://openalex.org/A5069325952)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种在频选择性信道下，面向 OFDM‑ISAC 系统的输入分布优化方法，能够在保持功率预算和期望副瓣泄漏（EISL）约束的前提下，最大化互信息。

**💡 创新点**

创新点包括：① 将传统 BA 算法分解为子载波级别的 factorized 版本，显著降低高维积分复杂度；② 引入梯度投影方法，仅用闭式表达式和一维搜索即可实时更新输入分布；③ 发现最佳策略是统一功率分配、基于水填充的峰度分配，揭示峰度在感知约束下的新资源意义。

**🔧 技术方法**

使用的技术包括：互信息最大化（容量-失真框架）、峰度和 EISL 的解析表达、factorized BA 更新、梯度投影求解、熵最大化与卷积去卷积、单载波水填充分析。

**📊 数据集**

实验数据基于 64 子载波 OFDM 系统，采用 4 路 Rician 信道（K = 6 dB），平均 SNR = 10 dB；EISL 约束设为 D = ζ(N−1)P²/N²，ζ = κ̅−1；不使用公开数据集，仅采用仿真生成的信道样本。

**📈 对比分析**

与统一峰度分配策略对比，梯度投影方法在平均峰度 κ̅ 取值范围内提升了约 0.1 bits/子载波；在不同平均 SNR 下，低 SNR 时收益更显著；个别子载波峰度分布呈水填充模式，证明了方法的有效性。

**⚠️ 局限性**

局限性包括：假设已知完整的信道估计，且信道为静态；未考虑多数据流或多用户情景；去卷积结果为近似，真实系统噪声分布与仿真模型可能不完全一致；对高阶峰度约束的解析闭式解尚未给出，需进一步研究。

---

## 350. Domain-Adapted Small Language Models for Reliable Clinical Triage

**arXiv ID:** 2604.26766 | [PDF](https://arxiv.org/pdf/2604.26766v1)

**作者:** Manar Aljohani `[一作]` (Virginia Tech), Xuan Wang `[通讯]` (Children’s National Hospital)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对小型语言模型进行领域适配，评估其在儿童急诊门诊中自动分配ESI等级的能力，并与大模型、多代理、检索增强等方法比较。

**💡 创新点**

引入基于银标准临床vignette的数据进行分块微调，证明仅使用小模型即可获得比大模型更稳定、更快、并且在临床安全性上更优的ESI预测。

**🔧 技术方法**

使用QLoRA低秩微调、结构化提取、临床vignette生成、零/少样本提示、检索增强、以及多代理投票实验等技术。

**📊 数据集**

使用约117,600例儿童急诊门诊记录，筛选后353例高质量记录做评测，利用专家手工vignette（245例）和银标准vignette（117,247例）做训练。

**📈 对比分析**

对比多款SLM与大型LLM（GPT-4o）在raw triage、vignette、结构化等六种提示管线下的总错误率、欠分/过分率、显著误差和推理时间；微调后的Qwen2.5‑7B在vignette上总错误率降至25.85%，显著误差≤2.56%，推理时间0.16s，明显优于大模型和其他策略。

**⚠️ 局限性**

单中心、仅儿童门诊、需要完整记录、排除精神/行为病患、以护士ESI为标准导致偏差、未进行前瞻性部署评估。

---

## 351. GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents

**arXiv ID:** 2604.26752 | [PDF](https://arxiv.org/pdf/2604.26752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 352. Learning Sparse BRDF Measurement Samples from Image

**arXiv ID:** 2604.26740 | [PDF](https://arxiv.org/pdf/2604.26740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 353. On the Complexity of Robust Markov Decision Processes and Bisimulation Metrics

**arXiv ID:** 2604.26748 | [PDF](https://arxiv.org/pdf/2604.26748v1)

**作者:** Marnix Suilen `[一作]` (University of Antwerp), Guillermo A. Pérez `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多态性马尔可夫决策过程 (RMDP) 的阈值问题，并给出了其复杂度分析。

**💡 创新点**

首次证明 (s,a)-rectangular RMDP 的阈值问题为 NP-hard，并在多项式时间内可解；同时给出 s-rectangular RMDP 在实数理论中的上界，并将其与奇偶游戏和仿射指标的归约相连。

**🔧 技术方法**

利用鲁棒线性规划、实数一阶理论和贝尔曼算子进行评估与策略迭代，并构造 RMDP 以计算仿射指标。

**📊 数据集**

实验使用 Gymnasium 的 Frozen Lake 5×5、10×10、20×20 环境，生成对应 RMDP 进行比较。

**📈 对比分析**

与传统的稳健界限值迭代（RBVI）相比，鲁棒策略迭代（RPI/RPIOT）在 5×5 与 10×10 地图上分别提升约 22 倍和 13 倍的速度，并通过加入最优性测试进一步减少一轮迭代。

**⚠️ 局限性**

主要限制在于 (s,a)-rectangular RMDP 的精确复杂度仍未确定，且在更大规模 RMDP 上内存消耗高，另外仅考虑期望折扣回报目标，其他目标需进一步研究。

---

## 354. Full Definability in a Profunctorial Model

**arXiv ID:** 2604.26829 | [PDF](https://arxiv.org/pdf/2604.26829v1)

**作者:** Takeshi Tsukada `[一作]` (Chiba University), Kengo Hirata `[通讯]` (University of Edinburgh)

**通讯引用:** 471 | [OpenAlex ID](https://openalex.org/A5113759875)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在基于群组上短程变换（profunctor）的模型中，构造并证明了 MLL+Mix 的完整可定义性，即每个可定义的线性逻辑证明都能在该模型中精确表示。

**💡 创新点**

创新点包括：① 将 Loader 的总性空间概念推广到 profunctor 范式；② 引入稳定性（stability）作为新的语义正确性判据；③ 证明严格分解系统（strict factorization system）自然出现于稳定与总性条件之下，并利用它完成完整可定义性的证明。

**🔧 技术方法**

技术手段主要为高级范畴论与双范畴（profunctor、群组、群oid）、稳定正交性、总性正交性、严格分解系统以及群论中的内自同构判定。

**📊 数据集**

本研究为理论性论文，没有使用实验数据集；

**📈 对比分析**

由于未涉及实验实现，本文没有性能对比与评估；

**⚠️ 局限性**

局限性：目前仅覆盖无指数（exponential）的 MLL+Mix；对多态或更复杂的逻辑扩展尚未讨论；且模型构造依赖于对群组上短程变换的同构类商化，可能在更大规模结构上面临规模与实现问题。

---

## 355. Uncertainty-Aware Predictive Safety Filters for Probabilistic Neural Network Dynamics

**arXiv ID:** 2604.26836 | [PDF](https://arxiv.org/pdf/2604.26836v1)

**作者:** Bernd Frauenknecht `[一作]` (RWTH Aachen University), Sebastian Trimpe `[通讯]` (RWTH Aachen University)

**通讯引用:** 2300 | [OpenAlex ID](https://openalex.org/A5023990842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在基于模型的强化学习框架中提出了一种不确定性感知的预测安全滤波器（Uncertainty-Aware Predictive Safety Filter, U-PSF），通过可解释的可达集和置信度约束来保障在探索过程中的约束满足。

**💡 创新点**

创新点在于：①将概率集成网络（Probabilistic Ensemble, PE）模型的高维不确定性量化与预测安全滤波器结合，给出严格的可达集上界；②引入“充分确定性集合”并通过Kalman增益约束防止模型被利用；③在Dyna式MBPO中实现无额外域知识的安全探索。

**🔧 技术方法**

使用技术包括：概率集成神经网络、稳健MPC（Robust Model Predictive Control）框架、Kalman增益阈值判定、线性化传播与误差上界、可达集上界推导与终端集扩展。

**📊 数据集**

在三个标准安全RL基准环境上评估：Pendulum、Cartpole、Drone。

**📈 对比分析**

与MBPO、XMPSC、SAC等基线对比，U-PSF在保持与MBPO相近的累计奖励的同时，显著降低约束违规次数，尤其在高维环境（Cartpole、Drone）中效果更突出。

**⚠️ 局限性**

主要限制是安全滤波器优化求解耗时较大，导致收敛慢；未来工作需改进MPC求解效率以实现实时部署。

---

## 356. FutureWorld: A Live Environment for Training Predictive Agents with Real-World Outcome Rewards

**arXiv ID:** 2604.26733 | [PDF](https://arxiv.org/pdf/2604.26733v1)

**作者:** Zhixin Han `[一作]` (Nankai University), Shuxin Zheng `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个名为FutureWorld的实时未来预测训练环境，允许LLM代理在连续的日常循环中生成预测问题、执行搜索推理、记录轨迹、收集真实结果并以结果为奖励进行强化学习；

**💡 创新点**

创新点在于：① 将实时预测流与强化学习训练循环闭合；② 直接使用真实结果而非代理内部过程奖励；③ 在训练过程中自带信息检索与推理的完整代理交互；④ 提供每日活跃的多样化预测基准。

**🔧 技术方法**

采用的技术包括：大语言模型代理（如 LLaMA-70B 等），基于工具的检索（Google/Serper API），强化学习算法 GRPO，Brier 负损失奖励，密集轨迹记录与回放，FSDP+vLLM 分布式训练。

**📊 数据集**

数据来源为72个公开在线网站，自动抓取候选未来事件，构造二元预测问题并经过滤与重采样后生成每日约500道问题；还利用公开的历史预测基准如 ForecastBench、FutureX 等做对比。

**📈 对比分析**

方法通过每日训练并在随后的 8 天内评估各检查点，结果显示预测准确率、Brier 分数、ECE 等指标随训练日数递增，RL 训练提升多领域预测表现；在每日基准上与现有前沿模型（GLM‑5.1、Gemini‑3.1‑Pro、Claude‑Opus‑4.6）对比，训练好的模型在多数问题类型上显著优于未训练模型，整体得分提升 10‑30%。

**⚠️ 局限性**

局限性包括：① 真实结果检索不及时导致约 20‑35% 的问题被丢弃，浪费监督信号；② 仅对二元预测问题进行了 RL 训练，对多选或数值预测的迁移效果有限；③ 训练依赖昂贵的 LLM 资源与大规模算力；④ 环境对数据来源的依赖可能引入偏见或安全风险。

---

## 357. Bridge: Basis-Driven Causal Inference Marries VFMs for Domain Generalization

**arXiv ID:** 2604.26820 | [PDF](https://arxiv.org/pdf/2604.26820v1)

**作者:** Mingbo Hong `[一作]` (University of Twente), Hao Cheng `[通讯]` (University of Twente)

**通讯引用:** 7447 | [OpenAlex ID](https://openalex.org/A5002932429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于因果推断的基线驱动域泛化目标检测框架Bridge，利用低秩基学习实现前门调整，抑制单源少量数据训练中的共因子诱导的伪相关；

**💡 创新点**

创新点在于通过可学习的低秩基和样本查询端到端实现前门调整，而不需显式构造共因子或外部字典，兼容判别与生成VFM；

**🔧 技术方法**

核心技术包括基线驱动的Causal Basis Block（CBB）、前门调整理论、低秩基学习、样本查询聚合、知识蒸馏等；

**📊 数据集**

实验使用Cross-Camera、Adverse Weather、Real-to-Artistic、Diverse Weather、Diverse Weather DroneVehicle等五个公开域泛化基准，且扩充了带天气标注的UAV DroneVehicle数据集；

**📈 对比分析**

与SOTA方法Boost、GDD等在各基准上均取得显著提升，mAP平均提升约2.4-4.5点；在不同VFM（DINOv2/3、SAM、Stable Diffusion）上均表现优异；

**⚠️ 局限性**

局限在于CBB仅近似前门调整的期望，无法给出闭式解析，且在极低光照等极端场景下仍受限于VFM表征能力。

---

## 358. Super-resolution Multi-signal Direction-of-Arrival Estimation by Hankel-structured Sensing and Decomposition

**arXiv ID:** 2604.26793 | [PDF](https://arxiv.org/pdf/2604.26793v1)

**作者:** Georgios I. Orfanidis `[一作]` (Florida Atlantic University), Elizabeth Serena Bentley `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4`

**🎯 论文内容**

提出了一种基于Hankel结构感知和数据矩阵分解的快速超分辨率多信号到达方向（DoA）估计的新框架，适用于硬件受限的空间采样。

**💡 创新点**

创新点在于提出了滑动窗口RF链感知和基于Hankel结构的秩-K分解算法，分别在L_2和L_1范数下实现最大似然估计，且在不同噪声环境下表现出优越的鲁棒性。

**🔧 技术方法**

使用了Hankel矩阵、结构低秩分解、最大似然估计等技术。

**📊 数据集**

进行了广泛的仿真研究，使用了白噪声和Bernoulli-Gaussian混合噪声模型的数据集，以模拟实际环境中的干扰。

**📈 对比分析**

与多种现有的DoA估计方法（如Matrix Pencil和单快照MUSIC）进行了比较，结果显示所提方法在低信噪比下具有更高的分辨率概率，尤其在信号接近时表现更为突出。

**⚠️ 局限性**

限制在于该方法在动态变化或硬件受限的环境中可能会受到训练数据假设的限制，且对信号源数量的先验知识有一定依赖。

---

## 359. TAP into the Patch Tokens: Leveraging Vision Foundation Model Features for AI-Generated Image Detection

**arXiv ID:** 2604.26772 | [PDF](https://arxiv.org/pdf/2604.26772v1)

**作者:** Ahmed Abdullah `[一作]` (Mannheim University of Applied Sciences), Oliver Wasenmüller `[通讯]` (Mannheim University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了多种现代视觉基础模型（VFM）在AI生成图像检测（AIGI）任务中的迁移性能，并提出了一种基于可调注意力池化（TAP）的轻量级特征提取头，利用所有patch token和cls token来提升检测效果。

**💡 创新点**

创新点在于①对最新VFM（如Perception Encoder）进行全面基准，发现其在AIGI检测中远超传统CLIP；②设计TAP，仅加入少量可训练参数即可聚合完整token序列，显著提升对局部生成痕迹的感知；③将上述技术应用于多种挑战性数据集，创下新的state‑of‑the‑art。

**🔧 技术方法**

技术细节包括：使用多种VFM（CLIP、DINOv2/3、SIGLIP2、Perception Encoder等）作为冻结特征提取器；构造单一查询向量的多头注意力池化（TAP）来聚合patch+cls token；随后使用两层MLP+残差与线性投影得到全局特征；训练采用AdamW、图像压缩/模糊等增强。

**📊 数据集**

实验使用GenImage、Chameleon和OpenSDI三大数据集，训练集包含SD v1.4/1.5生成图、ImageNet真实图、Megalith‑10M真实图等，测试覆盖未见的生成器如SD2.1/2.1/3/XL、Flux等。

**📈 对比分析**

与OMAT、AIDE、DualSight等现有方法对比，本文在GenImage上提升2.5–10.2%准确率，在Chameleon上提高29%准确率，在OpenSDI上提高4% F1和3%准确率；TAP在高分辨率和可变分辨率VFM上表现尤为突出。

**⚠️ 局限性**

局限性包括：在某些生成器（如Midjourney）上泛化下降；TAP可能导致对低级纹理过拟合；实验仅在单GPU上完成，缺乏大规模多实验验证；未探讨进一步的多任务或自监督扩展。

---

## 360. A Sufficient-Statistic Reduction of the Information Bottleneck to a Low-Dimensional Problem

**arXiv ID:** 2604.26744 | [PDF](https://arxiv.org/pdf/2604.26744v1)

**作者:** Joss Armstrong `[一作]` `[通讯]` (Ericsson Ireland), Joss Armstrong (Ericsson Ireland)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明当条件分布仅通过充分统计量时，信息瓶颈（IB）问题可以完全降维，等价于在该统计量与目标变量之间求解IB；并以此推导线性高斯情形下的闭式解，并给出非高斯推广；

**💡 创新点**

提出“IB Reduction Theorem”，即在充分统计量存在时，IB曲线、拉格朗日极值、最优表示及临界β完全保持不变，从而统一离散与高斯两类可解情形，并揭示了高斯解中的秩上界的本质；

**🔧 技术方法**

运用了信息论的链式法则、数据处理不等式、变分IB框架、Blahut‑Arimoto算法，以及线性高斯下的奇异值分解（canonical correlation）等技术；

**📊 数据集**

主要使用合成数据集：多类别高斯混合（M=16）配合非线性嵌入g，维度为2、5、10，随机生成标签中心并加入低噪声高斯噪声；

**📈 对比分析**

与标准VIB（两层MLP编码器/解码器）在不同β下对比，发现VIB在β超过阈值时会崩溃，无法逼近精确曲线；而降维后的Blahut‑Arimoto求解在几秒内给出完整IB曲线，表现显著优于VIB；

**⚠️ 局限性**

局限性在于需要预先知道或估计充分统计量，理论结果是人群级别的；在实际应用中需估计p(X,T)并受有限样本误差影响；非高斯情况仍需数值求解，且对大规模数据的可扩展性未验证。

---

## 361. Solving Positive Linear Programs with Differential Privacy

**arXiv ID:** 2604.26838 | [PDF](https://arxiv.org/pdf/2604.26838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 362. Catching the Fly: Practical Challenges in Making Blockchain FlyClient Real

**arXiv ID:** 2604.26736 | [PDF](https://arxiv.org/pdf/2604.26736v1)

**作者:** Pericle Perazzo `[一作]` (University of Pisa), Dario Capecchi `[通讯]` (University of Pisa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在 Zcash 区块链上实现并评估了 FlyClient 轻量级验证协议。

**💡 创新点**

创新点包括：① 引入基于经济预算的 w_a 对手模型；② 设计了可部署的 FlyClient prover；③ 提出了累积证明和蒸馏证明两种压缩优化。

**🔧 技术方法**

使用了 Merkle Mountain Range、Fiat‑Shamir、JSON‑RPC、Zcash 的 Equihash 及其潜在替代方案（Ethash、RandomX）等技术。

**📊 数据集**

主要数据集为 Zcash 主网 3 百万块链及其多次网络升级节点。

**📈 对比分析**

通过对比不同证明表示（JSON、二进制、gzip）以及交互式/非交互式和不同验证器，测得证明大小呈对数增长；累积证明约节省 9 % 交互式、71 % 非交互式；蒸馏证明可将非交互式证明压缩至 320 KiB，链路成本从 13 美元降至 3 美元。

**⚠️ 局限性**

限制包括：蒸馏证明需硬分叉，适用性仅限于 PoW 链；实验仅在 Zcash 上验证，未在其他区块链或不同硬件上测试；对极端链条长或高难度变动的鲁棒性待进一步评估。

---

## 363. HalluCiteChecker: A Lightweight Toolkit for Hallucinated Citation Detection and Verification in the Era of AI Scientists

**arXiv ID:** 2604.26835 | [PDF](https://arxiv.org/pdf/2604.26835v1)

**作者:** Yusuke Sakai `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1638 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 HalluCiteChecker 工具箱，用于检测和验证科学论文中的幻觉式引用（hallucinated citations）。

**💡 创新点**

将幻觉式引用检测拆分为三阶段结构化任务：引用提取（Citation Extraction）、引用识别（Citation Recognition）与引用匹配（Citation Matching），并提供轻量级、离线、无生成式 AI 的模块化实现；同时引入统一的 dataclass 以简化数据流与可扩展性。

**🔧 技术方法**

采用 Docling（CPU 版 PDF 解析）进行引用提取，使用 DeLFT（Python 版 GROBID 的序列标注模型）进行引用识别，利用基于 Levenshtein 距离的字符级模糊匹配实现标题匹配；整体实现基于 Python、ML 模型、LMDB/Arrow 数据存储，完全离线运行。

**📊 数据集**

内部使用 ACL Anthology、arXiv、DBLP 的本地数据库做匹配；评估使用 EMNLP 2025 主轨论文、ACL 2025 长文等真实论文集。

**📈 对比分析**

通过与先前分析结果比较验证召回率相当；在 MacBook Pro、Air 与 WSL 环境中，单篇论文在 35 秒以内完成全部检测，平均每篇论文约 12 分钟完成 100 篇长文。性能报告以毫秒/论文和毫秒/引用给出，展示了可接受的处理时延。

**⚠️ 局限性**

主要瓶颈在引用提取阶段，需进一步优化；匹配仅依赖标题，忽略其他字段导致误匹配或漏检；工具仅支持 PDF，其他格式需额外处理；缺乏公开基准数据集，难以与其他方法客观对比。

---

## 364. Random Cloud: Finding Minimal Neural Architectures Without Training

**arXiv ID:** 2604.26830 | [PDF](https://arxiv.org/pdf/2604.26830v1)

**作者:** Javier Gil Blázquez `[一作]` `[通讯]`, Javier Gil Blázquez

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并评估了一种无训练的随机云方法，用于在不完整训练的前提下寻找最小化前馈网络拓扑。

**💡 创新点**

利用随机初始化网络在无训练评估中偶然获得可用性能，再通过逐层减少神经元识别必要结构，从而实现训练前结构搜索。

**🔧 技术方法**

采用随机云生成、前向无训练评估、结构压缩递减、最终反向传播微调以及Wilcoxon检验统计等技术。

**📊 数据集**

在七个分类基准上进行实验，包括四个二分类（乳腺癌、Sonar、Ionosphere、Adult Income）和三个多分类（Iris、Wine、光学数字）数据集。

**📈 对比分析**

与完整训练、幅度裁剪、随机裁剪等基线比较，Random Cloud在6/7个数据集上匹配或优于裁剪方法，尤其在Sonar上显著提升4.9pp，并且在多数数据集上计算成本比裁剪更低（0.67–0.94×完整训练）。

**⚠️ 局限性**

在高维输入（如MNIST）中训练无评估信号衰减导致性能下降，且目前仅针对前馈全连接网络，未扩展到卷积或更大规模模型。

---

## 365. A Multi-Dataset Benchmark of Multiple Instance Learning for 3D Neuroimage Classification

**arXiv ID:** 2604.26807 | [PDF](https://arxiv.org/pdf/2604.26807v1)

**作者:** Ethan Harvey `[一作]` (Tufts University), Michael C. Hughes `[通讯]` (Tufts University)

**通讯引用:** 2240 | [OpenAlex ID](https://openalex.org/A5058890009)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种脑 CT/MRI 分类任务，系统评估了多实例学习（MIL）、注意力 MIL、3D CNN 与 3D ViT 的性能，并比较了不同编码器与聚合方式。

**💡 创新点**

证明简单均值池化 MIL 能在多数任务中匹敌或超越复杂注意力方法，并且训练速度比注意力模型快约 25 倍。

**🔧 技术方法**

使用冻结的 ViT‑B/16 编码器、ABMIL、TransMIL、SmAP 等注意力聚合，以及 3D ResNet‑18 等 3D CNN。

**📊 数据集**

采用 ADNI1、OASIS‑3 CT/MRI、RSNA CT、-800、-10k 等 7 个公开/专有数据集。

**📈 对比分析**

在 9 个二分类任务中，均值池化 MIL 在 AUROC 与顶级 3D CNN 相近，但训练成本低；注意力 MIL 在大型数据集上仅提升 0.01‑0.03。

**⚠️ 局限性**

受限于仅使用预训练编码器、未尝试更深 3D CNN 或细粒度预处理，且注意力机制在实例级定位上的表现不佳。

---

## 366. ViCrop-Det: Spatial Attention Entropy Guided Cropping for Training-Free Small-Object Detection

**arXiv ID:** 2604.26806 | [PDF](https://arxiv.org/pdf/2604.26806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 367. A Semantic Quantum Circuit Cache for Scalable and Distributed Quantum-Classical Workflows

**arXiv ID:** 2604.26788 | [PDF](https://arxiv.org/pdf/2604.26788v1)

**作者:** Mar Tejedor `[一作]` (Barcelona Supercomputing Center), Rosa M. Badia `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 7840 | [OpenAlex ID](https://openalex.org/A5015588225)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一个内容可寻址的量子电路缓存系统，能够检测语义等价并在混合量子‑经典工作流中重用已执行的量子电路结果。

**💡 创新点**

创新点在于将ZX‑计算图归约与 Weisfeiler–Leman 图哈希相结合，产生确定性、可跨节点共享的电路指纹，并在分布式 HPC 环境中提供轻量级 LMDB 与可扩展 Redis 两种后端，形成可复用的系统级优化。

**🔧 技术方法**

采用 ZX‑计算图归约、可扩展的 Weisfeiler–Leman 图哈希、LMDB/Redis 持久键值存储，以及 Qiskit Aer 状态向量模拟器、PyCOMPSs 任务调度和 Qibochem/Random ansatz 生成器。

**📊 数据集**

使用了两类基准数据集：在 MareNostrum 5 上进行的 48‑量子位 HEA 与随机电路的四路电线切分（共 8192 子电路），以及 24‑节点最大割问题的 QAOA（p=2,3,4）与离散化参数空间（粗/中/细 3 组）进行的 Differential Evolution 优化。

**📈 对比分析**

通过与无缓存基准对比，分布式缓存在单节点可获得 7×（Redis）/3.9×（LMDB）加速，四节点 4.4×/2.6×，十六节点 1.9×/1.4×；在 35‑量子位真实 QPU 上实现 11.2× 加速；在 QAOA 优化中缓存避免了 27.6% 的电路评估，且未改变优化收敛或结果质量。

**⚠️ 局限性**

主要局限包括：ZX 归约不保证唯一归一形态，可能导致哈希冲突；只支持静态电路，未覆盖动态测量控制；Redis 内存占用高；在极大并发写入时 LMDB 受单写限制；以及对极大规模量子器件的硬件验证仍受设备限制。

---

## 368. Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation

**arXiv ID:** 2604.26768 | [PDF](https://arxiv.org/pdf/2604.26768v1)

**作者:** Weihang Su `[一作]` (Tsinghua University), Yiqun Liu `[通讯]` (Tsinghua University)

**通讯引用:** 10097 | [OpenAlex ID](https://openalex.org/A5100668121)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过正交子空间分解（Orthogonal Subspace Decomposition，OSD）方法，将任务级通用行为与文档级知识分别编码为共享的 Task LoRA 与文档特定的 Knowledge LoRA，从而降低文档适配器与任务行为的混合，提高多文档参数合成的稳定性。

**💡 创新点**

创新点在于将任务 LoRA 与文档 LoRA 在训练阶段进行正交约束（软正交正则化与硬正交子空间），实现任务和知识的功能解耦，提升 parametric RAG 在多文档检索时的鲁棒性。

**🔧 技术方法**

使用 LoRA 参数化、正交正则化/硬正交子空间设计、BM25 检索、Llama 3 系列模型（1B、3B、8B）等技术。

**📊 数据集**

实验数据集涵盖 KILT 基准的四个知识密集任务（开放域问答 2WikiMultihopQA、HotpotQA、ComplexWebQuestions、PopQA；事实核查 FEVER；槽位填充 Zero‑Shot RE；知识对话 Wizard of Wikipedia）以及医学领域 PubMedQA。

**📈 对比分析**

与标准 RAG、PRAG 基线在不同检索深度（K=1,3,5,7,10）下进行对比，使用 F1/Accuracy 评估。结果显示 D‑PRAG（软正交）和 D‑PRAG‑hard（硬正交）对检索深度的敏感性显著降低，性能更稳定，并在多文档合成场景中往往优于传统 PRAG。

**⚠️ 局限性**

局限性：实验仅在少量 Llama 3 模型规模和检索深度上验证，缺乏对更大规模检索、不同模型家族和其他适配器架构的泛化；实验结果为探索性观察，需进一步验证其在更广泛场景中的可靠性。

---

## 369. Exploring the Potential of Probabilistic Transformer for Time Series Modeling: A Report on the ST-PT Framework

**arXiv ID:** 2604.26762 | [PDF](https://arxiv.org/pdf/2604.26762v1)

**作者:** Zhangzhi Xiong `[一作]` (ShanghaiTech University), Kewei Tu `[通讯]` (ShanghaiTech University)

**通讯引用:** 22899 | [OpenAlex ID](https://openalex.org/A5061216998)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将Probabilistic Transformer重新解释为条件随机场的均值场变分推理，将其扩展为二维时空因子图 ST-PT，并在三类时间序列任务中分别激活图结构、因子参数和推理协议三种可编程杠杆，探讨先验注入、条件生成与潜在空间自回归的实现与效果。

**💡 创新点**

创新点在于：①把 Transformer 的自注意力等价为 CRF 的 MFVI，构建可编程因子图；②在时间序列上引入二维因子图（跨通道与跨时间的三元因子）形成 ST-PT；③分别利用图结构、因子矩阵和推理协议三条杠杆，分别实现先验注入、条件编程因子以及贝叶斯潜在更新。

**🔧 技术方法**

使用技术包括：均值场变分推理（MFVI）、条件随机场（CRF）因子图推理、旋转位置编码（RoPE）、PatchTST 风格的时空分块、基于因子生成的条件网络（PT‑FG）以及教师蒸馏。

**📊 数据集**

实验数据集涵盖：长时序预测基准 Weather、Electricity、Traffic、ETTh1/2/ETTm1/2/ETTm2、ECL；合成先验数据集（Lag、Periodicity、Trend）；以及 ConTSG‑Bench 的十个条件生成数据集（Synth‑U/M、AirQuality、Traffic、TelecomTS、ETTm1、Weather‑Conceptual/Morphological、PTB‑XL‑Conceptual/Morphological）。

**📈 对比分析**

比较方法：在长时序预测中对 ST‑PT 与 TSMixer、TimeMixer、PatchTST、TimesNet、Crossformer 等基线做 MSE/MAE 对比；在先验注入实验中对 ST‑PT 与 DLinear、iTransformer、PatchTST 进行少样本与噪声鲁棒性评估；在条件生成中对 PT‑FG 与 10 个 ConTSG‑Bench 生成器在 DTW、CRPS、ACD、SD、KD、MDD 6 指标上做排名，PT‑FG 平均排名 2.78，取得 24/60 个 top‑1；在潜在自回归中对 DeepVAR、LSTM‑AR、LSTNet 进行比较，MFVI‑基潜在 AR 在所有 12 个数据/时长组合上均优于对照组，长时域提升尤为显著。

**⚠️ 局限性**

局限性包括：单一先验在真实数据上的效果有限，需要多先验组合；潜在自回归推理顺序执行导致推理速度慢且训练成本高；三种杠杆未联合使用，缺乏对多模态组合的验证；CRF 教师蒸馏需要完整序列前向，进一步增加训练开销。

---

## 370. Locality for Codes over the Integers

**arXiv ID:** 2604.26756 | [PDF](https://arxiv.org/pdf/2604.26756v1)

**作者:** Giulia Cavicchioni `[一作]` (Fondazione Bruno Kessler), Julien Lavauzelle `[通讯]` (University Paris 8)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了整数代码的加权局部可恢复性，定义了加权局部性并推导了对应的Singleton‑like上界，随后构造了多族具有可接受距离和局部性的整数码，包括整数版Tamo–Barg构造。

**💡 创新点**

创新点在于将加权局部性概念推广到整数域，给出相应的Singleton‑like界限，并提出能够接近该界限的整数Tamo–Barg构造，使得在同等局部性下实现更优的距离与码率。

**🔧 技术方法**

利用整数余数映射、Chinese Remainder 定理、短码与拼接技术以及Tamo–Barg多项式评价构造来实现整数码的局部可恢复性。

**📊 数据集**

未使用具体数据集，采用理论示例与实验参数进行验证。

**📈 对比分析**

通过理论上限与实验示例对比，整数Tamo–Barg在给定局部性下的距离与码率接近Singleton‑like界限，优于传统的CR码。

**⚠️ 局限性**

难以得到整数Tamo–Barg码的精确距离与码率公式，且Singleton‑like界限的紧度尚待进一步验证。

---

## 371. Runtime Verification: Monitoring, Knowledge, and Uncertainty (Lecture Notes)

**arXiv ID:** 2604.26753 | [PDF](https://arxiv.org/pdf/2604.26753v1)

**作者:** Benedikt Bollig `[一作]` (Université Paris-Saclay), Benedikt Bollig `[通讯]` (Université Paris-Saclay)

**通讯引用:** 1210 | [OpenAlex ID](https://openalex.org/A5082538918)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在部分可观测下的运行时验证、诊断、隐匿性与可监控性，并统一用知识逻辑框架表示并判定其可行性。

**💡 创新点**

首次将知识运算与线性时间逻辑结合，形成统一的知识时序逻辑，提出监视器可产生三值判定并通过知识监视器实现诊断/预测器。

**🔧 技术方法**

采用自动机理论构造知识转换器、知识监视器以及转移图求解可达性，利用区域抽象解决时序化学的空性判定。

**📊 数据集**

主要使用抽象的Büchi/时域Büchi自动机语言，未涉及具体外部数据集。

**📈 对比分析**

与传统的无知识形式化（如LTL）相比，模型检查在PSPACE内可解；安全/无安全性质的监测可在多项式时间内判断，整体性能与传统运行时验证方法相当但更具表达力。

**⚠️ 局限性**

主要局限在于时间与知识的联合导致判定问题往往是不可判定的，且在复杂公式下状态空间指数级增长，影响可扩展性。

---

## 372. Comparing Smart Contract Paradigms: A Preliminary Study of Security and Developer Experience

**arXiv ID:** 2604.26727 | [PDF](https://arxiv.org/pdf/2604.26727v1)

**作者:** Matteo Vaccargiu `[一作]` (University of Cagliari), Giuseppe Destefanis `[通讯]` (University College London)

**通讯引用:** 2269 | [OpenAlex ID](https://openalex.org/A5036425614)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对 Solidity（命令式）与 Move（资源导向）两种智能合约语言，先分析 12 对功能等价合约，再对 11 名两种语言都有实战经验的开发者做问卷，综合评估代码安全开销、代码体量、开发难度与安全信心。

**💡 创新点**

首次以等价合约对和开发者调查为基础，定量比较资源导向语言将安全从运行时迁移到编译期的效果，提供了范式差异对安全与开发体验影响的初步实证。

**🔧 技术方法**

代码度量（SLOC、函数数、循环复杂度、显式安全检查、类型注解、资源操作等）+ Wilcoxon 符号秩检验、Benjamini‑Hochberg FDR 校正 + Cohen’s d 效应大小 + Likert 量表问卷 + 定性内容分析。

**📊 数据集**

Rosetta Smart Contracts 仓库中的 12 对功能等价的 Solidity 与 Move 合约，以及 11 名双语开发者的问卷数据。

**📈 对比分析**

采用配对 Wilcoxon 检验与 FDR 校正比较各度量；结果显示 Move 显式安全检查减少 60%（p=0.002，d=-1.75），代码量增加 47%（p=0.002，d=1.90），循环复杂度相同；开发者学习难度中等但安全信心更高，说明资源导向范式将安全迁移到编译期，虽带来更大代码体量和学习成本。

**⚠️ 局限性**

样本规模有限（12 对合约、11 名受访者），度量方法主要基于正则表达式，未覆盖所有安全维度，外部效度受限；需要更大规模复现、真实漏洞率评估和不同开发环境验证。

---

## 373. Distributed Multi-View Vision-Only RSSI Estimation

**arXiv ID:** 2604.26738 | [PDF](https://arxiv.org/pdf/2604.26738v1)

**作者:** Jung-Beom Kim `[一作]` (Yonsei University), Woongsup Lee `[通讯]` (Yonsei University)

**通讯引用:** 2954 | [OpenAlex ID](https://openalex.org/A5066939022)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于视觉的分布式多视角 RSSI 估计框架 MulViT‑TF，利用多摄像头的图像直接推断 WiFi 信号强度。

**💡 创新点**

创新点在于将每个摄像头独立的 Vision Transformer 编码器与跨摄像头的 Transformer 融合模块相结合，实现了无辅助传感器、无反馈回路的 RSSI 估计，并显著提升了 NLoS 环境下的估计精度。

**🔧 技术方法**

采用 Vision Transformer、Transformer 融合模块、MLP 头、DeiT 预训练、线性插值与高斯平滑的预处理等技术，并在训练阶段分两阶段微调。

**📊 数据集**

使用真实室内数据集，包括办公室（Scene 1）和会议室（Scene 2）两种场景，分别配备两台 Raspberry Pi 摄像头和 ESP32 WiFi 模块，收集了约两万条同步的图像‑RSSI 数据。

**📈 对比分析**

与单视角 ViT（SinViT‑D、SinViT‑W）以及 token‑wise DNN 融合（MulViT‑TWDNN）进行对比，MulViT‑TF 在 RMSE、MAE、Pearson r 及 3 dB 成功率方面分别提升约 26%、13% 等，并且计算量和参数量低于最优单视角模型。

**⚠️ 局限性**

局限性包括仅在单用户场景下验证，需对摄像头进行精确时间同步，对摄像头数量扩展时模型计算量随之线性增长，且未针对极端多用户或移动环境进一步评估。

---

## 374. MoRFI: Monotonic Sparse Autoencoder Feature Identification

**arXiv ID:** 2604.26866 | [PDF](https://arxiv.org/pdf/2604.26866v1)

**作者:** Dimitris Dimakopoulos `[一作]` (University of Edinburgh), Ioannis Konstas `[通讯]` (Heriot-Watt University)

**通讯引用:** 2728 | [OpenAlex ID](https://openalex.org/A5030546839)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在封闭式问答任务上，通过可控的微调实验研究大语言模型（LLM）在加入新知识时导致的事实幻觉机制，并用稀疏自编码器（SAE）监测激活空间变化。

**💡 创新点**

提出了 Monotonic Relationship Feature Identification (MoRFI) 算法，可在多维度（如未知事实比例、训练轮次）上发现单个激活维度的单调趋势，进一步证明了忘记是“访问受阻”而非“知识被抹除”。

**🔧 技术方法**

核心技术包括稀疏自编码器（SAE）对残差流的特征分解、Bootstrap + Spearman/Kendall 统计检验的 MoRFI、激活 steering（单维度或合成方向）恢复知识。

**📊 数据集**

使用从 Wikidata 转换而来的闭合式 QA 数据集（包含七种未知事实比例的子集），对 Llama 3.1 8B、Gemma 2 9B 与 Mistral 7B 进行微调。

**📈 对比分析**

与直接使用合成方向 δ_u 的对比显示，单个 MoRFI 选出的稀疏激活方向在恢复已知事实方面能获得 69–85% 的性能回升，且单维度干预往往优于复合方向，表明关键知识信号是稀疏分布。

**⚠️ 局限性**

局限性包括仅在闭合式问答任务上验证；仅评估三种 7–9B 规模模型；依赖预训练的 SAE 作为特征提取工具；未对不同任务、不同规模模型或更广泛的数据分布进行泛化验证。

---

## 375. Exact Dynamic Programming for Solow--Polasky Diversity Subset Selection on Lines and Staircases

**arXiv ID:** 2604.26929 | [PDF](https://arxiv.org/pdf/2604.26929v1)

**作者:** Michael T. M. Emmerich `[一作]` `[通讯]` (University of Jyvaskyla), Michael T. M. Emmerich (University of Jyvaskyla)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种在有序一维或坐标归一化的ℓ1阶梯点集上，利用Solow–Polasky多目标多样性指标等价于度量空间的规模化幅值（magnitude）所得到的连续间隙加性公式，构建了O(kn²)的Bellman递推动态规划，求解固定基数的多样性子集选择问题，并将该方法推广到二维Pareto前沿和高维坐标归一化阶梯结构。

**💡 创新点**

创新点在于：①将Solow–Polasky多样性与度量空间幅值等价性明确化；②推导出仅依赖相邻点间距的超bolic‑tangent间隙公式；③证明该公式在有序ℓ1阶梯结构下可转化为一维加性动态规划，从而得到精确多基数优化的多项式算法。

**🔧 技术方法**

主要技术包括：度量空间幅值理论、超bolic‑tangent间隙识别、Bellman递推动态规划、坐标归一化的ℓ1阶梯映射及其在二维Pareto前沿上的应用。

**📊 数据集**

实验数据采用合成点集：5点一维线段、5点二维Pareto前沿、20点随机采样的二维Pareto前沿（曲线f₂=1−f₁²），用于演示算法的实现与选取示例。

**📈 对比分析**

与一般NP‑难的全局多样性子集选择问题相比，该算法在所限定的有序ℓ1阶梯或Pareto前沿结构上实现了精确最优解，时间复杂度为O(kn²)，显著优于暴力搜索；然而本文未给出与其他启发式或近似算法在更一般数据集上的实验对比。

**⚠️ 局限性**

局限性：仅适用于有序且无坐标回溯的ℓ1阶梯点集（包括一维线、二维Pareto前沿及其高维扩展），无法处理一般度量空间或非单调多目标数据；动态规划复杂度仍随基数k和点数n呈二次增长，对极大规模实例可能不够高效。

---

## 376. A Note on How to Remove the $\ln\ln T$ Term from the Squint Bound

**arXiv ID:** 2604.26926 | [PDF](https://arxiv.org/pdf/2604.26926v1)

**作者:** Francesco Orabona `[一作]` (King Abdullah University of Science and Technology), Francesco Orabona `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 3015 | [OpenAlex ID](https://openalex.org/A5044260265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了shifted KT势能与改变先验相等，并利用该等价关系将高斯截断先验引入Squint算法，成功去除了数据独立误差中的lnln T项。

**💡 创新点**

创新点在于将shifted KT势能重新诠释为对Krichevsky–Trofimov先验的调整，并通过高斯截断先验实现无lnln T的数据独立误差上界。

**🔧 技术方法**

主要技术包括先验变换、赌博式专家学习的连续投机框架、解析积分公式、误差下界推导以及倍增技巧的应用。

**📊 数据集**

该工作属于理论分析性质，并未使用具体的数据集进行实验验证。

**📈 对比分析**

与原Squint算法及其他数据独立方法相比，本文得到的误差上界为O(√{T(ln(1/π)+ln 2)})，成功去掉了lnln T因子，但在某些情形下仍不一定优于传统Squint。

**⚠️ 局限性**

局限性主要在于缺乏实际数据验证、对先验参数的选取仍有经验性限制，以及在实现时可能需要处理复杂的积分和截断问题。

---

## 377. Bi-Level Optimization for Contact and Motion Planning in Rope-Assisted Legged Robots

**arXiv ID:** 2604.26910 | [PDF](https://arxiv.org/pdf/2604.26910v1)

**作者:** Ruben Malacarne `[一作]`, Michele Focchi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套双层优化框架，用于在垂直表面上使用绳索辅助的攀爬机器人规划多跳轨迹

**💡 创新点**

创新点在于将整数层的跳跃次数和着陆区域选择与连续层的动态轨迹规划耦合，外层采用Cross-Entropy Method，内层使用梯度式非线性规划并对着陆点进行成本优化

**🔧 技术方法**

使用Cross-Entropy Method、梯度式非线性规划（单射击、RK4数值积分）、降维3DOF动力学模型、成本地图滤波器、MPC跟踪控制

**📊 数据集**

使用自行生成的高度图/成本地图（半球、突起柱、岩石墙等），通过点云预处理与插值得到连续表面

**📈 对比分析**

在三种实验场景下与多跳序列进行比较，收敛至低能耗、低成本的路径；示例中fitness在四次迭代后达321.3，能量消耗、跳跃次数等指标均优于单跳方案

**⚠️ 局限性**

局限性包括未考虑绳索穿透障碍的几何约束、燃气瓶能量耗尽未建模、缺乏真实硬件验证

---

## 378. Stochastic Entanglement of Deterministic Origami Tentacles For Universal Robotic Gripping

**arXiv ID:** 2604.26897 | [PDF](https://arxiv.org/pdf/2604.26897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 379. SEAL: Semantic-aware Single-image Sticker Personalization with a Large-scale Sticker-tag Dataset

**arXiv ID:** 2604.26883 | [PDF](https://arxiv.org/pdf/2604.26883v1)

**作者:** Changhyun Roh `[一作]` (Chung-Ang University), Jihyong Oh `[通讯]` (Chung-Ang University)

**通讯引用:** 269 | [OpenAlex ID](https://openalex.org/A5090121183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SEAL模块，用于单图像贴纸个性化生成，解决视觉纠缠和结构僵化问题，并构建了结构化标签贴纸数据集StickerBench。

**💡 创新点**

创新点在于设计了三大组件——语义引导空间注意力损失、分割合并Token策略以及结构感知层限制，形成可插拔、架构无关的适配模块，并通过结构化标签提升评测可控性。

**🔧 技术方法**

技术上使用了扩散模型的测试时微调（TTF），结合SAM生成对象掩码、跨注意力映射与语义引导损失、辅助Token分裂合并、层级限制等技术，在Stable Diffusion v2.1基础上实现。

**📊 数据集**

数据集方面，本文使用260k张贴纸图像构成的StickerBench（含六属性标签），并与MOD、SER30K等公开数据集做对比。

**📈 对比分析**

实验将SEAL集成到Custom Diffusion、CoRe、UnZipLoRA等基线，在StickerBench单图像设置下，利用CLIP‑T、CLIP‑I、DINO等指标评估，均显著提升身份保持和上下文可控性，优于对应基线。

**⚠️ 局限性**

局限性在于仍需依赖SAM掩码生成、对不同扩散模型或更大规模数据的泛化尚未充分验证，且对极端背景或多属性同时编辑的鲁棒性有限，部分指标提升并非绝对最优，需进一步调参和扩展。

---

## 380. Adaptive Self-Organization in Anonymous Dynamic Networks

**arXiv ID:** 2604.26931 | [PDF](https://arxiv.org/pdf/2604.26931v1)

**作者:** Garrett Parzych `[一作]` (Arizona State University), Joshua J. Daymude `[通讯]` (Arizona State University)

**通讯引用:** 235 | [OpenAlex ID](https://openalex.org/A5070425572)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并分析了在时间变化、动态拓扑且只有局部感知的环境信号下，匿名网络节点如何自组织颜色以匹配持续信号的目标分布；

**💡 创新点**

主要创新在于：①证明任何确定性算法只能解决同质目标分布（所有节点最终颜色相同）的实例；②设计一种线性时间、对数内存的确定性算法解决该同质实例；③通过随机化扩展，实现对任意（非同质）实例的高概率解法；

**🔧 技术方法**

利用消息时间戳与TTL（time‑to‑live）机制构造广播与回收协议，结合自同步计数器与锁定状态，实现信号传播与颜色同步；随机化算法通过采样目标分布 r(s) 生成颜色；

**📊 数据集**

论文为理论分析性工作，未使用实际数据集；

**📈 对比分析**

性能分析显示算法在 1‑interval 连通动态网络中以 O(n) 轮完成收敛，使用 O(log n) 位内存；在已知 n 的情况下可实现强收敛（所有信号下均为 O(n) 轮）；随机化版本在高概率下保持同样的时间与空间复杂度；

**⚠️ 局限性**

局限性包括：确定性算法无法处理非同质目标分布；随机化算法仅在概率意义下满足要求；算法依赖于 1‑interval 连通性及无标识节点假设，若网络不满足这些假设，需进一步研究。

---

## 381. Select to Think: Unlocking SLM Potential with Local Sufficiency

**arXiv ID:** 2604.26940 | [PDF](https://arxiv.org/pdf/2604.26940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 382. ProcFunc: Function-Oriented Abstractions for Procedural 3D Generation in Python

**arXiv ID:** 2604.26943 | [PDF](https://arxiv.org/pdf/2604.26943v1)

**作者:** Alexander Raistrick `[一作]` (Princeton University), Jia Deng `[通讯]` (Princeton University)

**通讯引用:** 126736 | [OpenAlex ID](https://openalex.org/A5101542158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了 ProcFunc，一套 Python 库，用于在 Blender 中快速构建、组合、分析和执行程序化 3D 生成器，并提供预实现的材料与室内房间生成器。

**💡 创新点**

创新点包括：①将 Blender 的低层图形操作拆解为 497 个显式原语函数，消除全局状态依赖；②提供资产级的可组合接口与随机/确定采样器；③实现静态计算图追踪，支持参数可视化、优化和自动化；④实现多层材料组合和高效的资源管理，实现更高的多样性与效率。

**🔧 技术方法**

技术方案涵盖：Python API、自动数据类型推断、Blender 节点图转 Python（transpiler）、静态分析 tracer、Python‑FCL 碰撞检测、Eevee 渲染、FFmpeg 压缩以及预实现的随机采样与场景布置工具。

**📊 数据集**

使用的数据集包括 BlenderGym 基准、Infinigen‑Indoors、ProcFunc 自制室内房间数据集（用于深度/法向训练），以及 50k 对 Stereo 对应数据集；评估时还对 Middlebury、CREStereo、TartanAir、FSD 等公开基准进行对比。

**📈 对比分析**

与 Infinigen 进行 VLM 参数编辑、材料创作、室内房间生成等任务比较，评估指标包括 Photometric Loss、N‑CLIP、Chamfer Distance、错误率与成本。实验表明 ProcFunc 在错误率下降、代码更短、生成速度提升（室内房间 CPU 0.02 min vs Infinigen 0.37 M 三角）以及 Stereo 训练性能均优于传统方法；在资源占用上也显著减少（CPU、内存、存储）。

**⚠️ 局限性**

局限性包括：仅支持通用室内房间，缺乏特定房间类型；LLM 对 ProcFunc 接口不熟悉导致性能未最大化；高细节渲染仍略慢；生成器仍需人工定制，且对某些高级应用（如复杂场景或动画）覆盖不足。

---

## 383. Artistic Practice Opportunities in CST Evaluations: A Longitudinal Group Deployment of ArtKrit

**arXiv ID:** 2604.26935 | [PDF](https://arxiv.org/pdf/2604.26935v1)

**作者:** Catherine Liu `[一作]` (Claremont McKenna College), Jingyi Li `[通讯]` (Pomona College)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5048818007)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对数字艺术家使用ArtKrit绘图支持工具进行为期三周的纵向、群组化评估，记录使用日志、Discord对话、每日日志和访谈，分析工具使用与创作过程随时间及社群支持的变化。

**💡 创新点**

提出将CST评估视为创作参与机会的群组纵向方法；将工具评估与艺术家支持网络结合；揭示工具使用从探索到有针对性再到均衡反馈的演化，并探索社群对情绪、动机与技术采纳的影响。

**🔧 技术方法**

使用Krita插件ArtKrit（含自适应构图线、色彩与价值反馈），通过Python进行交互日志分析；Discord平台用于社群交流；结构化访谈与主题分析提炼质性洞见。

**📊 数据集**

收集了36件作品、50+小时的使用日志、Discord聊天记录、每日日志条目及访谈文本；参考图像包括Portrait of Madame M.、Study of Poplars、Reeds in the Snow等。

**📈 对比分析**

通过对使用日志的量化分析（动作次数、类别比例、活跃时长）比较不同周、不同组的使用趋势，发现总体使用量下降、反馈维度趋于均衡；未采用统计显著性检验，仅呈现趋势图与质性解释。性能方面，工具在适配后减少了latency，用户报告情绪提升、创作自由度增加，作品质量多样化。

**⚠️ 局限性**

样本规模有限（9人），年龄与文化单一；所有人绘制相同主画，生态有效性受限；研究者参与可能引入权力偏差，难以区分工具效应与社群效应；实验为技术探针非完全真实情境，缺乏对长周期自然使用的捕捉。

---

## 384. Approximating the Network Design Problem for Potential-Based Flows

**arXiv ID:** 2604.26882 | [PDF](https://arxiv.org/pdf/2604.26882v1)

**作者:** Max Klimm `[一作]` (TU Berlin), Lea Strubberg `[通讯]` (TU Berlin)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了潜能驱动流网络中的最小成本网络设计问题，给出多种精确与近似算法，并证明其在不同约束下的复杂度。

**💡 创新点**

创新点在于将非线性潜能约束通过巧妙的变换转化为传统组合优化（如约束最短路径、并行/串行组合规则）问题，从而实现 FPTAS；同时给出完整的硬性与可逼近性证明，阐明问题在一般图与SP图上的难度差异。

**🔧 技术方法**

采用了凸优化与参数化优化的理论工具（如有效电阻、有效导电率的组合公式）、Lagrange 对偶与 KKT 条件、动态规划与尺度化技术、以及现有的约束最短路径 FPTAS。

**📊 数据集**

本文为理论研究，无使用真实数据集，所有实验均基于合成图结构（如SP树、并行/串行组合图、Steiner 树构造图）进行验证。

**📈 对比分析**

与传统网络设计方法（如固定/可变容量的最小化）相比，本文在无上界和SP图情形下实现了多项式时间或 FPTAS；在一般图上证明了不可多项式逼近性，表明方法在这些情形下已是最优或近似最优。

**⚠️ 局限性**

限制在于：在一般图或含有容量上界的情形下问题仍为 NP‑hard，无法获得 PTAS；同时算法的实现依赖于对电阻/导电率的精确计算，实际应用中需处理浮点误差；并且对 r≠1 的情形理论复杂度与实现细节尚未完全统一。

---

## 385. FaaSMoE: A Serverless Framework for Multi-Tenant Mixture-of-Experts Serving

**arXiv ID:** 2604.26881 | [PDF](https://arxiv.org/pdf/2604.26881v1)

**作者:** Minghe Wang `[一作]` (TU Berlin), David Bermbach `[通讯]` (TU Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FaaSMoE，一个将 Mixture‑of‑Experts（MoE）模型的专家拆分为无状态 FaaS 函数的服务器无状态框架，用于多租户推理服务。

**💡 创新点**

创新点在于利用 FaaS 的事件驱动、scale‑to‑zero 与自愈特性，实现专家的按需激活与跨租户共享，并支持可配置的专家粒度，以在减少资源浪费的同时保持模型精度。

**🔧 技术方法**

采用 Function‑as‑a‑Service（tinyFaaS）平台、异步 HTTP 调用、微批处理、以及一个处理分词、门控和路由的轻量化控制面（Orchestrator）等技术。

**📊 数据集**

使用 Qwen1.5‑MoE‑2.7B 模型作为实验基准，并在 BIG‑Bench 任务上构建多租户工作负载进行评估。

**📈 对比分析**

对比了四种部署策略（Baseline、Local Distribution、FaaSMoE‑Shared、FaaSMoE‑Private），在六个并发租户工作负载下，FaaSMoE‑Shared 将 CPU 需求从 1126.84% 降至 326.4%，内存从 217.52 GB 降至 72.25 GB，显示出 70% 以上的资源节约。

**⚠️ 局限性**

局限性包括：当前 FaaS 主要支持 CPU，缺乏 GPU 加速；专家调用带来的网络延迟和通信开销；专家块大小的选择仍需经验或自适应策略；仅在单一 MoE 模型和 CPU 环境下验证，尚未覆盖更大规模或 GPU‑密集型模型。

---

## 386. HealthNLP_Retrievers at ArchEHR-QA 2026: Cascaded LLM Pipeline for Grounded Clinical Question Answering

**arXiv ID:** 2604.26880 | [PDF](https://arxiv.org/pdf/2604.26880v1)

**作者:** Md Biplob Hosen `[一作]`, Lujie Karen Chen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在ArchEHR‑QA 2026共享任务中提出了HealthNLP_Retrievers系统，该系统采用四阶段级联架构（问题解释、证据打分、基于证据的答案生成、答案与证据对齐），实现了对患者自述问题的语义转化、对临床记录中相关句子进行Likert‑scale评估、在限定词数内生成专业化答案，并将答案与对应证据精确对应。

**💡 创新点**

创新点包括① 通过人物化、少样本的问答解释模块，将患者叙述压缩为15词专业查询；② 引入1–5分Likert尺度的证据评分与动态回溯过滤，提升检索召回率；③ 采用精度约束的多对多答案‑证据对齐框架，显著降低hallucination；④ 在所有子任务中使用Gemini 2.5 Pro实现长上下文推理。

**🔧 技术方法**

主要技术：Gemini 2.5 Pro大语言模型；少样本（few‑shot）提示工程；基于提示的句子级Likert评分；软截断（soft‑cut）生成约束；JSON格式答案‑证据映射；自定义温度与安全设置。

**📊 数据集**

使用ArchEHR‑QA 2026共享任务数据集，数据来自去标识化的MIMIC‑III临床记录，并包含患者问题与相应临床笔记片段。

**📈 对比分析**

对照其他参赛系统，HealthNLP_Retrievers在Subtask 1（问题解释）取得第一，overall 31.2分，Subtask 2（证据评分）第七，Strict Micro F1 60.2，Subtask 3（答案生成）第五，overall 34.6分，Subtask 4（答案‑证据对齐）第九，Micro F1 76.9。相较于领先团队，其在召回偏好与精度控制上表现突出，但在某些指标（如MEDCON、BLEU）略逊一筹。

**⚠️ 局限性**

局限性包括：完全依赖Gemini 2.5 Pro API，无法在本地部署且缺乏可复现性；仅在ArchEHR‑QA 2026数据集上评估，未验证跨数据集或其他EHR系统的泛化；四阶段级联结构易导致误差累积，尤其是问题解释阶段的误差会影响后续检索与答案质量。

---

## 387. STARRY: Spatial-Temporal Action-Centric World Modeling for Robotic Manipulation

**arXiv ID:** 2604.26848 | [PDF](https://arxiv.org/pdf/2604.26848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 388. A 3GPP Perspective on Spectrum Sharing for the 5G-to-6G Migration: From DSS to MRSS

**arXiv ID:** 2604.26853 | [PDF](https://arxiv.org/pdf/2604.26853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 389. What Kind of Language is Easy to Language-Model Under Curriculum Learning?

**arXiv ID:** 2604.26844 | [PDF](https://arxiv.org/pdf/2604.26844v1)

**作者:** Nadine El-Naggar `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Ted Briscoe `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 7463 | [OpenAlex ID](https://openalex.org/A5030189297)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对语言模型在人工语言中的词序学习偏好进行实验，探究课程学习（CL）对模型可学习性和与世界语言统计分布的对齐性的影响。

**💡 创新点**

首次将长度为基础的CL策略引入人工语言学习实验，验证训练数据呈现顺序如何显著改变模型的词序偏好及其与语言学典型性的一致性。

**🔧 技术方法**

使用三种语言模型（RNN、LSTM、Transformer）在Fairseq框架下训练，并通过训练步骤的分段（3到8词句）实现CL；评估指标包括困惑度（PPL）和与词序出现频率的Pearson相关系数（TA）。

**📊 数据集**

基于GCG（通用范畴语法）的人工语言语料库，共96种不同词序配置，每种语言80k句子，分别划分短（3–8词）、中（9–10词）、长（11–20词）测试集。

**📈 对比分析**

对比随机数据顺序（Original）与CL顺序下的模型表现：CL在短句上PPL略高，但在中长句上PPL下降或相当；TA值在大多数实验中变得不那么负（即与语言学典型性对齐程度下降）。

**⚠️ 局限性**

局限性包括：CL策略过于简单（仅按长度排序）、TA指标（Pearson相关系数）的有效性未充分验证、实验仅使用人工语言且未涉及更丰富的CL方法，因而无法给出CL是否能提升或削弱语言学典型性的一致性结论。

---

## 390. Joint Transceiver Orientation Optimization for Rotatable-Antenna MIMO Capacity Maximization

**arXiv ID:** 2604.26845 | [PDF](https://arxiv.org/pdf/2604.26845v1)

**作者:** Zheng Ailing `[一作]` (Shanghai Jiao Tong University), Chen Wen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 470490 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了基于可旋转天线（RA）的MIMO通信系统，通过联合优化发射端与接收端天线的朝向以及发射协方差矩阵，来重新塑造有效的MIMO通道并提升系统容量。

**💡 创新点**

创新点在于：①构建了角度相关的RA通道模型，①提出了交替优化（AO）框架，其中发射协方差矩阵采用水分配，天线朝向采用Riemannian Frank-Wolfe方法；②在低SNR、MISO/SIMO等特殊情形下给出闭式分析与简化方案。

**🔧 技术方法**

所用技术包括：方向相关通道建模、SVD+水分配、Riemannian Frank-Wolfe优化、Jacobian求解、线性化子问题求解、AO迭代等。

**📊 数据集**

使用的实验数据集为：16×16的U盘阵列（4×4×4×4），工作频率3.5 GHz，随机生成6个散射点，噪声功率-80 dBm，发射功率10 dBm，最大天线倾角θ_max=π/6。

**📈 对比分析**

与固定天线（FOA）、随机朝向、等向天线、仅优化发射或接收方向以及低SNR专用算法等基准进行比较，实验显示所提RA设计在所有SNR和天线规模下均实现最高容量，典型提升约为34%相对于FOA，15%相对于仅优化接收方向。

**⚠️ 局限性**

局限性包括：①非凸优化仅能收敛至局部最优；②机械响应速度和θ_max限制可能限制在快速变化环境中的适用性；③实际部署需考虑硬件实现与能耗。

---

## 391. Hyper Input Convex Neural Networks for Shape Constrained Learning and Optimal Transport

**arXiv ID:** 2604.26942 | [PDF](https://arxiv.org/pdf/2604.26942v1)

**作者:** Shayan Hundrieser `[一作]` (University of Twente), Johannes Schmidt-Hieber `[通讯]` (University of Twente)

**通讯引用:** 324 | [OpenAlex ID](https://openalex.org/A5002981992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 Hyper Input Convex Neural Networks (HyCNN) 架构，用于学习凸函数，并在凸回归和高维最优传输映射估计等任务中进行实验。

**💡 创新点**

创新点在于将 Maxout 单元与 Input Convex Neural Network 结合，保证网络在输入上始终凸；证明 HyCNN 在逼近二次函数时参数量指数下降；提供深度可训练的初始化策略；展示深层 HyCNN 能充分利用深度进行逼近。

**🔧 技术方法**

采用 HyCNN 结构（双通道、非负权重、maxout 或平滑 maxout 激活），理论上证明逼近能力；在 OT 任务中使用双向循环的外循环/内循环 Adam 训练；利用 log-sum-exp 作为平滑门控；实验中对比 MLP、ICNN、GroupMaxNet、Monge Gap MLP 等。

**📊 数据集**

实验数据包括：① 高维（d=50）随机均匀输入 + 高斯噪声的合成凸回归数据；② 以正态分布为源，应用已知凸势函数得到目标的合成 OT 数据；③ 真实单细胞 RNA‑seq 数据集 4i，用于评估 OT 映射性能。

**📈 对比分析**

与 MLP、ICNN、GroupMaxNet、Monge Gap MLP 等基线比较；HyCNN 在 MSE（凸回归）和 Sinkhorn 距离（OT 估计）上表现最佳，尤其是深层（L≥4）HyCNN 在所有任务中均优于同类基线；展示了深度对性能的显著提升。

**⚠️ 局限性**

局限性：HyCNN 的凸且光滑、共形惰性先验可能在某些非光滑或非共形问题上受限；实验规模受计算资源限制，缺乏更大规模的验证；训练时间与资源仍需进一步优化。

---

## 392. ClassEval-Pro: A Cross-Domain Benchmark for Class-Level Code Generation

**arXiv ID:** 2604.26923 | [PDF](https://arxiv.org/pdf/2604.26923v1)

**作者:** Yeheng Chen `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2449 | [OpenAlex ID](https://openalex.org/A5033286111)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ClassEval-Pro基准，评估LLM在类级合成任务中的能力；

**💡 创新点**

创新点在于三阶段自动化构造流程，跨域组合任务，生成300个涵盖11个领域的真实GitHub类级任务，消除人工标注与数据污染；

**🔧 技术方法**

采用AST解析、LLM Judge Ensemble评估、自动测试套件生成、多阶段过滤与5种生成策略（holistic、incremental、compositional、top-down、bottom-up）等技术；

**📊 数据集**

使用2025年1月后GitHub公开Python代码作为数据来源；

**📈 对比分析**

通过5大前沿LLM与5种生成策略进行对比，Pass@1最高达45.6%，模型间差距17.7点，策略对弱模型提升多达9.4个百分点，展示了基准的区分力；

**⚠️ 局限性**

局限在于仅测试Python，无法推广到静态语言；仍缺乏高质量跨方法协调与依赖管理，导致逻辑与依赖错误占比高，尚未完全解决类级生成挑战。

---

## 393. Hot Fixing in the Wild

**arXiv ID:** 2604.26892 | [PDF](https://arxiv.org/pdf/2604.26892v1)

**作者:** Carol Hanna `[一作]` (University College London), Federica Sarro `[通讯]` (University College London)

**通讯引用:** 4650 | [OpenAlex ID](https://openalex.org/A5012165852)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于大规模开源仓库数据，系统性识别并分析了生产环境中的热修复（hot fix）实践，并首次比较了人类开发者与自主编码代理在紧急修复场景下的行为差异。

**💡 创新点**

创新点在于：①使用LLM+时间窗口的双重过滤方法自动识别热修复，填补了缺乏统一标记的空白；②在61,000+ GitHub 仓库上对热修复与常规 bug 修复进行量化比较；③首次探讨人类与 AI 代理在热修复中的修补策略与协作模式差异。

**🔧 技术方法**

技术方法包括：大语言模型（Llama 3.2、Qwen、Phi‑4）做文本分类，时间戳阈值过滤，PR 级别特征提取（提交数、文件改动、测试改动、评审人数等），词袋+词云对比人类/机器人 PR 文本。

**📊 数据集**

使用 Hao‑Li/AIDev 数据集，其中包含61,000+ 仓库、47,000+ 开发者及五种主流自主编码代理（GitHub Copilot、Claude Code、Devin 等）。

**📈 对比分析**

对比方法是基于 PR 级别度量的统计分析与词云可视化；结果显示热修复相比常规修复更小、更快、参与者更少、测试改动更少；人类 PR 通常更迭代、改动更大、删减更多；代理 PR 更为原子化、测试改动更高，且合并率与人类相当。

**⚠️ 局限性**

局限性包括：①热修复识别依赖 LLM 的分类准确性，误报/漏报仍存在；②数据集中人类与代理的分布不平衡，代理热修复样本较少；③只研究 AIDev 数据集，可能无法推广至不同社区或商业项目；④未评估热修复的长期质量与技术债累积。

---

## 394. Multiple Additive Neural Networks for Structured and Unstructured Data

**arXiv ID:** 2604.26888 | [PDF](https://arxiv.org/pdf/2604.26888v1)

**作者:** Janis Mohr `[一作]`, Jörg Frochte `[通讯]` (Ruhr Bochum University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出多加性神经网络（MANN），将浅层神经网络作为梯度提升框架的基学习器，并结合早停、验证集阈值和连续学习机制，适用于结构化与非结构化数据（图像、音频）

**💡 创新点**

创新点在于将梯度提升迁移至浅层神经网络基学习器，并通过自适应停止、阈值检测和持续学习提升鲁棒性；同时将胶囊网络引入图像任务，增强特征提取与泛化能力

**🔧 技术方法**

技术实现包括梯度提升、浅层多层感知机/胶囊网络、Adam/SGD优化、ReLU/Sigmoid激活、路由算法、早停、阈值停止与持续学习策略；与 XGBoost、ANT、MLP、Learn++.MT 等模型做对比

**📊 数据集**

使用的实验数据集包括结构化数据：Bike Sharing、SARCOS、CT Scan Slice、Million Song、Heart Disease、Rain in Australia、Titanic、Higgs Boson；图像数据集：MNIST、CIFAR‑10；连续学习基准为 Bike Sharing 2011/2012 年份切分

**📈 对比分析**

通过 RMSE、MAE、MSE、准确率等指标与 XGBoost、ANT、MLP、Learn++.MT 等传统和增量学习方法进行对比；结果显示 MANN 在大多数数据集上均优于或接近树模型，连续学习场景下误差显著降低，图像任务上 MANN 超越普通 CNN 与胶囊网络

**⚠️ 局限性**

局限性包括：在极小或高度不平衡的数据集（如 Titanic、Higgs Boson）表现不如树模型；模型对阈值和学习率等超参数仍需手动调优；训练成本和模型规模较大；未来需进一步探索更深网络、自动化超参调整与流式学习情境

---

## 395. Cognitive Atrophy and Systemic Collapse in AI-Dependent Software Engineering

**arXiv ID:** 2604.26855 | [PDF](https://arxiv.org/pdf/2604.26855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 396. Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data

**arXiv ID:** 2604.26841 | [PDF](https://arxiv.org/pdf/2604.26841v1)

**作者:** Bao Pham `[一作]` (Rensselaer Polytechnic Institute), Matteo Negri `[通讯]` (CY Cergy Paris Université)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨了统一基离散扩散模型（UDDM）在语言生成中的记忆与泛化行为，发现其本质上可视为关联记忆（AM），并通过条件熵与令牌恢复率描述记忆到泛化的相变；

**💡 创新点**

提出了无需显式能量函数即可通过条件似然最大化形成吸引子基底的理论框架，并首次用条件熵量化记忆泛化过渡；

**🔧 技术方法**

利用统一基离散扩散模型（UDDM）与Transformer结构的条件似然推断，配合条件熵与令牌恢复率评估；

**📊 数据集**

在LM1B数据集上进行实验，评估不同训练样本比例与模型规模下的记忆/泛化转折；

**📈 对比分析**

通过令牌恢复率曲线和序列条件熵分布对比，展示随着数据增大，训练样本的恢复率下降、测试样本恢复率上升，二者最终趋同；大模型延迟转折，但最终的条件熵差距被压缩；

**⚠️ 局限性**

缺点包括：令牌恢复率与条件熵与传统评估指标的相关性仍待验证；扩展到大规模语言模型时需处理更高参数与训练成本；

---

## 397. KAYRA: A Microservice Architecture for AI-Assisted Karyotyping with Cloud and On-Premise Deployment

**arXiv ID:** 2604.26869 | [PDF](https://arxiv.org/pdf/2604.26869v1)

**作者:** Attila Pintér `[一作]`, György Cserey `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个端到端的染色体分型系统 KAYRA，利用容器化微服务架构实现云端与本地部署，支持专家人工审核的工作流。

**💡 创新点**

创新点包括：① 单一职责的微服务拆分与容错管道；② 级联 ROI 降维策略，使后续模型仅在适当区域工作；③ 兼容双重部署环境（云/本地），满足临床数据驻留要求；④ 可在不同骨干网络间无缝替换模型。

**🔧 技术方法**

核心技术包括 EfficientNet‑B5 + U‑Net 语义分割、Mask R‑CNN（ResNet‑50+FPN）实例分割、ResNet‑18 分类器；容器化部署基于 Docker、FastAPI、TorchServe；图像预处理使用 Otsu 二值化与连通组件、边缘复制填充；后端采用异步任务队列与 REST API。

**📊 数据集**

训练数据来自约 24,000 条专家标注 karyogram（Cytolab）、297+430 对同源三元组（DPC/OHII、PPCU‑ITK）、160 条结构异常 karyogram、60 条 Philadelphia 阳性三元组，以及约 145 条商业平台样本，外加公共数据集（BioImLab、CloudDataLab、Coriell）。

**📈 对比分析**

在 10 张染色体图像（共 459 条染色体）上与两套商业参考系统对比：分割精度 98.91%（对比 78.21%/40.52%），分类精度 89.1%（对比 86.9%/54.5%），方向准确率 89.76%（对比 94.55%/78.43%）。对老旧密度阈值系统的分割与分类均达统计学显著提升（p<0.0001），对现代 AI 系统仅分割显著提升，分类差异不显著（p=0.34）。

**⚠️ 局限性**

局限性包括：仅单一实验室验证；与现代 AI 系统相比方向准确率略低；Y 染色体样本不足导致召回率最低；尚未完成 CE/FDA 监管认证；虽然架构骨干无关，但仍需多中心验证与旋转增广以弥补性能差距。

---

## 398. Edge AI for Automotive Vulnerable Road User Safety: Deployable Detection via Knowledge Distillation

**arXiv ID:** 2604.26857 | [PDF](https://arxiv.org/pdf/2604.26857v1)

**作者:** Akshay Karjol `[一作]` (Oakland University), Darrin M. Hanna `[通讯]` (Oakland University)

**通讯引用:** 920 | [OpenAlex ID](https://openalex.org/A5068579855)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

训练了一个压缩的YOLOv8‑S学生模型，并在INT8量化下保持高精度，同时使用知识蒸馏提升模型鲁棒性

**💡 创新点**

证明知识蒸馏能够将置信度校准从大模型迁移至小模型，并显著提升量化鲁棒性，使得小模型在INT8下性能远优于直接训练

**🔧 技术方法**

知识蒸馏（温度缩放 KL 损失）、YOLOv8、Post‑Training INT8 量化、特征对齐（低权重）

**📊 数据集**

BDD100K 驾驶场景数据集（70K 训练，10K 验证）

**📈 对比分析**

对比教师大模型、直接训练学生与蒸馏学生，在 FP32 与 INT8 两种精度下评估 mAP、精度、召回率和误报率；蒸馏学生在 INT8 下保持仅 -5.6% mAP、精度提升 37%、误报率降低 44%，且速度提升 2.4×

**⚠️ 局限性**

仅在 BDD100K 和 YOLOv8 上验证；未使用 QAT、未在真实 Edge 设备（如 Jetson）上评估；蒸馏超参数（β、γ、T）的进一步优化尚未完成

---

## 399. Turning the TIDE: Cross-Architecture Distillation for Diffusion Large Language Models

**arXiv ID:** 2604.26951 | [PDF](https://arxiv.org/pdf/2604.26951v1)

**作者:** Gongbo Zhang `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 18377 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型扩散语言模型进行跨架构知识蒸馏，将 8B/16B 教师压缩为 0.6B 学生。

**💡 创新点**

提出三模块框架：时间-训练双轴调度、补全掩码分裂、逆向跨词表对齐，解决教师可靠性、上下文稀缺与词表不匹配问题。

**🔧 技术方法**

双轴 Lambda 调度、补全掩码分裂、逆向 Chunk‑Level Approximate Likelihood Matching（Reverse CALM）以及标准 KL/交叉熵蒸馏。

**📊 数据集**

使用 WeDLM‑8B‑Instruct、LLaDA2.0‑mini 以及公共数据集（Tulu‑3、SmolTalk、OpenCoder OPC‑SFT）训练；评测八个基准（GSM8K、MATH、BBH、MMLU‑Pro、MMLU、HellaSwag、HumanEval、MBPP）。

**📈 对比分析**

相较于未蒸馏 BD3LM 和同尺寸 AR 基线，平均提升 1.53 分，HumanEval 从 32.3 提升至 48.78，代码生成显著领先；同时显著降低显存并保持近 80% 的推理吞吐。

**⚠️ 局限性**

仍需解决跨词表对齐的噪声、两次教师推理导致的计算开销，以及在更大模型/不同架构上的可扩展性问题。

---

## 400. Three-Step Nav: A Hierarchical Global-Local Planner for Zero-Shot Vision-and-Language Navigation

**arXiv ID:** 2604.26946 | [PDF](https://arxiv.org/pdf/2604.26946v1)

**作者:** Wanrong Zheng `[一作]` (University of Southern California), Laurent Itti `[通讯]` (University of Southern California)

**通讯引用:** 35841 | [OpenAlex ID](https://openalex.org/A5054494771)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种三步无监督的全局-局部规划框架（Three-Step Nav）来解决连续环境下的视觉语言导航任务，能在不进行任务特定微调的情况下实现零样本导航。

**💡 创新点**

创新点在于：① 采用三视图（look forward、look now、look backward）循环，使模型在全局计划、局部执行与轨迹审计之间交替；② 通过“look backward”实现轨迹级自审计与回溯，显著降低漂移与提前停止；③ 仅依赖多模态大型语言模型（如 GPT‑5），无需梯度更新。

**🔧 技术方法**

主要技术包括多模态大型语言模型（MLLM）prompt 设计、基于图像+文本的子指令分解、候选视角评估、轨迹回放与审计，以及自适应裁决模块（stay/continue/backtrack/look‑around）。

**📊 数据集**

在两个公开连续环境基准上进行评估：R2R‑CE 与 RxR‑CE。

**📈 对比分析**

与现有零样本方法相比，Three‑Step Nav 在 R2R‑CE 上取得最高的成功率（34%）和 SPL（29.12%），导航误差降低约15%；在 RxR‑CE 上也实现了显著提升，nDTW 提升12.6%，NE 降低11.2%。

**⚠️ 局限性**

局限性包括：① 依赖高成本的大型语言模型，推理延迟较高；② 在高度动态或噪声环境（如真实机器人）中的鲁棒性尚未验证；③ 仍无法通过在线学习自适应处理极长或模糊指令；④ 对图像与文本的解释可能受模型偏见影响。

---

## 401. World2VLM: Distilling World Model Imagination into VLMs for Dynamic Spatial Reasoning

**arXiv ID:** 2604.26934 | [PDF](https://arxiv.org/pdf/2604.26934v1)

**作者:** Wanyue Zhang `[一作]` (Chinese Academy of Sciences), Jiajun Zhang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 5266 | [OpenAlex ID](https://openalex.org/A5100319572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 World2VLM 框架，利用可控世界模型在训练阶段生成动作条件的视图转移，将动态空间推理知识蒸馏进视觉-语言模型。

**💡 创新点**

创新点在于：①将世界模型视为训练时教师而非推理时工具；②通过双向（逆推与前向）空间任务将动作推理与结果预测联合学习；③引入 GRPO 细化奖励，提升结构化空间答案质量。

**🔧 技术方法**

技术手段包括：基于 Stable Virtual Camera / HY‑WorldPlay 的动作条件视图合成；对象检测与追踪的结构化元数据注入；两阶段训练（SFT + GRPO）；LoRA 参数高效微调；任务相关奖励设计。

**📊 数据集**

使用的训练数据来自 ScanNet（真实室内场景）和 MulSeT（模拟场景），构造约 100K 个动作-视图对；评估基准为 SAT‑Real、SAT‑Synthesized、VSI‑Bench 与 MindCube。

**📈 对比分析**

与基线 Qwen2.5‑VL‑7B、MindJourney‑style 推理时世界模型耦合以及单向蒸馏对比，World2VLM‑SFT 在所有四个基准上提升约 10–20 % 以上，GRPO 阶段进一步提升 5–15 %；相比推理时耦合方法，性能提升更显著且推理成本更低。

**⚠️ 局限性**

局限性包括：①依赖高质量的世界模型，生成视图质量直接影响蒸馏效果；②当前只在 egocentric 视角下评估，难以直接推广至全局/第三人称视角；③训练过程仍需大量离线生成数据，且 GRPO 调优成本不小。

---

## 402. On the Learning Curves of Revenue Maximization

**arXiv ID:** 2604.26922 | [PDF](https://arxiv.org/pdf/2604.26922v1)

**作者:** Steve Hanneke `[一作]` (Purdue University), Grigoris Velegkas `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究单一物品单一买家拍卖中的学习曲线，刻画从样本学习中获得的收益下降速率。

**💡 创新点**

首次提出学习曲线框架并给出不同分布支持下的最优收敛速率，包括近乎指数收敛和1/√n速率。

**🔧 技术方法**

采用PAC分析、贝叶斯一致性、结构化ERM、分布截断、DKW不等式及信息论下界构造等技术。

**📊 数据集**

使用理论上的合成分布（离散、有限、闭合）进行证明，没有使用实际数据集。

**📈 对比分析**

与PAC学习的上界和经典ERM做对比，证明在多种分布条件下可实现更快的收敛速率；ERM在某些离散分布上失效。

**⚠️ 局限性**

局限在于需要对分布支持做限制（如有限或离散），并且对最优收益未被有限价格实现时只能得到极慢速率，且证明主要在单维情形。

---

## 403. ClawGym: A Scalable Framework for Building Effective Claw Agents

**arXiv ID:** 2604.26904 | [PDF](https://arxiv.org/pdf/2604.26904v1)

**作者:** Fei Bai `[一作]` (Renmin University of China), Wayne Xin Zhao `[通讯]` (Renmin University of China)

**通讯引用:** 17833 | [OpenAlex ID](https://openalex.org/A5037145565)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ClawGym 框架，完成任务合成、轨迹收集、模型训练与评估，并生成 13.5K 任务（ClawGym‑SynData）、训练出 ClawGym‑Agents 并构建 200 题高质量 benchmark（ClawGym‑Bench）。

**💡 创新点**

创新点在于双路任务合成（persona‑driven top‑down 与 skill‑grounded bottom‑up）并结合自动化工作空间构造与混合代码/rubric 验证；使用黑盒 roll‑out 与轻量 RL，及严格的 Benchmark 构建流程。

**🔧 技术方法**

技术包括 LLM 生成（GPT‑5、MiniMax‑M2.5、GPT‑5.4 等）、OpenClaw harness、黑盒 roll‑out、监督微调（SFT）与轻量 RL、代码 + rubric 验证、YaRN 扩展上下文、自动质量评估与人机复核。

**📊 数据集**

使用数据集：ClawGym‑SynData（13.5K 任务）、ClawGym‑Bench（200 题 benchmark）、PinchBench（30 题对比）。

**📈 对比分析**

通过在 ClawGym‑Bench 与 PinchBench 上评估，训练后的 ClawGym‑Agents 在小模型（Qwen3‑8B）上提升约 43%（ClawGym‑Bench）/ 38%（PinchBench），大模型（Qwen3‑30B‑A3B）提升 25–55%；ClawGym‑Agents 在 PinchBench 甚至与部分专有模型相当，Bench 能显著区分不同规模模型与任务类别。

**⚠️ 局限性**

局限性：任务合成仍可能缺乏极端多样性；验证依赖人工‑LLM 复核成本高；RL 方案效果有限；模型对极端错误恢复与跨文件一致性仍不稳定；工作空间环境仍局限于本地文件与工具，未覆盖多模态或更大规模协同场景。

---

## 404. Breaking the Rigid Prior: Towards Articulated 3D Anomaly Detection

**arXiv ID:** 2604.26868 | [PDF](https://arxiv.org/pdf/2604.26868v1)

**作者:** Jinye Gan `[一作]` (ShanghaiTech University), Yingna Wu `[通讯]` (ShanghaiTech University)

**通讯引用:** 987 | [OpenAlex ID](https://openalex.org/A5111513090)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了ArtiAD基准并提出SPA‑SDF方法，解决了3D关节运动物体的异常检测问题。

**💡 创新点**

突破传统刚性先验，采用姿态条件隐式场和傅里叶编码关节状态，构建连续姿态依赖的正常性体。

**🔧 技术方法**

使用多频傅里叶编码的关节嵌入、两阶段形状‑姿态分离的SDF隐式网络、姿态估计最小化重建能量以及点级/对象级AUROC评估。

**📊 数据集**

使用自研的ArtiAD数据集，包含15,229个点云、39个单自由度关节物体类别、六种缺陷类型，并划分训练/已知/未知姿态。

**📈 对比分析**

与八个现有基准（BTF、PatchCore、M3DM、Reg3D‑AD、PO3AD、PASDF等）进行公平对比，SPA‑SDF在已知姿态和未知姿态的对象级AUROC分别达到0.884与0.874，显著优于所有对比方法。

**⚠️ 局限性**

目前仅覆盖单自由度关节，姿态估计采用离散网格搜索，难以扩展到多自由度结构，并缺少真实扫描数据验证。

---

## 405. Resume-ing Control: (Mis)Perceptions of Agency Around GenAI Use in Recruiting Workflows

**arXiv ID:** 2604.26851 | [PDF](https://arxiv.org/pdf/2604.26851v1)

**作者:** Sajel Surati `[一作]` (New York University), Emily Black `[通讯]` (New York University)

**通讯引用:** 858 | [OpenAlex ID](https://openalex.org/A5038262828)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对22名招聘专业人士的半结构化访谈，研究了生成式人工智能（genAI）在高风险招聘决策中的使用方式、对招聘者代理权的影响以及随之产生的职业技能退化与投入产出不匹配等现象。

**💡 创新点**

创新点在于：①首次把genAI定位为“隐形建筑师”，揭示其在决策框架和信息处理上的潜在主导作用；②揭示“沸腾海洋”效应——招聘者为应对genAI产生的“噪声”与竞争加剧，反而投入更多时间、降低人力技能和降低招聘质量；③通过功能性分析与访谈主题结合，提供了对AI工具使用情况与人机协作机制的综合视角。

**🔧 技术方法**

采用的技术方法包括：半结构化访谈、质性反思主题分析（Atlas.ti）、功能性工具分析（将采访中提及的AI工具按招聘阶段与功能分类），以及对访谈文本的手动校对与编码。

**📊 数据集**

数据集为22名来自技术、制造、生物技术、医疗、娱乐、出版、营销和非营利组织的招聘者访谈记录，时间跨度为2025年9月至12月。对访谈记录进行自动转录后手动校正，去识别与保密相关的个人信息。

**📈 对比分析**

与现有文献（如传统AI招聘工具的偏见评估、AI助力决策的实验研究）相比，本研究未进行量化指标或对照实验，而是通过归纳主题来展示代理权与工作效率的变化。研究指出，尽管参与者报告了一定的时间节省，但整体投入产出并不显著，甚至出现效率下降与招聘质量下降的倾向。

**⚠️ 局限性**

限制包括：①样本仅来自美国，缺乏跨文化视角；②访谈采用回顾性方式，易受记忆偏差与社会期望影响；③研究聚焦定性发现，缺乏客观效能指标；④仅关注招聘人员的自述，未直接观察AI工具实际使用效果；⑤对AI工具的功能分类依赖参与者描述，可能存在误差。

---

## 406. AnimateAnyMesh++: A Flexible 4D Foundation Model for High-Fidelity Text-Driven Mesh Animation

**arXiv ID:** 2604.26917 | [PDF](https://arxiv.org/pdf/2604.26917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 407. Walk With Me: Long-Horizon Social Navigation for Human-Centric Outdoor Assistance

**arXiv ID:** 2604.26839 | [PDF](https://arxiv.org/pdf/2604.26839v1)

**作者:** Lingfeng Zhang `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**通讯引用:** 7933 | [OpenAlex ID](https://openalex.org/A5012419026)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个名为 Walk with Me 的地图无依赖长时程户外社会导航框架，能够将高层次自然语言指令转换为具体目的地并完成从起点到终点的全程导航，同时在复杂社交场景中进行安全决策与停等。

**💡 创新点**

核心创新在于：① 将公共地图服务作为轻量级语义与地理先验，仅用于目的地定位与粗略路径生成；② 构建高层 VLM 与低层 VLA 的双层决策结构，并引入观察感知路由器在常规导航与安全关键场景间动态切换；③ 在同一闭环中实现高层意图理解与低层社会合规动作生成，避免传统方法中高层规划与低层控制的分离与不一致。

**🔧 技术方法**

技术手段包括：基于 Vision‑Language 模型（VLM）进行意图理解与目的地选择；利用公共地图 API（如百度/高德/Google）获取 POI 与步行路径；低层使用 Vision‑Language‑Action（VLA）模型进行局部社会合规轨迹预测；观察感知路由器结合视觉、GPS 与轨迹历史判断是否需要停等；完整闭环实现包括 SLAM 定位、低层运动控制以及远程推理服务器部署。

**📊 数据集**

实验主要依赖公开地图服务获取 POI 与路径；使用 Athena 2.0 Pro AGV 机器人搭载 RealSense RGB‑D 摄像头和 GPS 进行真实世界测试；没有专门收集的标注数据集，而是通过 20 次实地试验（两类任务：最后一公里投递与盲人导向）评估系统性能。

**📈 对比分析**

与传统基于 HD 地图或点目标导航方法对比，Walk with Me 在 20 次真实世界试验中获得平均 60% 的成功率，最后一公里投递 70% 成功率，盲人导向 50%。通过消融实验验证了高层 VLM 与低层 VLA 的重要性：不同 VLM（如 MiMo‑Embodied、RoboBrain 2.0 等）与不同 VLA（如 SocialNav、CityWalker 等）分别影响成功率从 30%~60% 不等，表明框架对模型选择敏感。

**⚠️ 局限性**

局限性包括：① 受限于公共地图服务的完整性与实时性，导致路径规划与目的地定位偶尔失效；② GPS 与定位噪声可能影响长程导航精度；③ VLM 与 VLA 在极端动态或稀有社交场景下的鲁棒性不足，容易出现误判或停等过度；④ 依赖远程推理服务器，导致延迟与网络依赖成为实际部署瓶颈。

---

## 408. Color-Encoded Illumination for High-Speed Volumetric Scene Reconstruction

**arXiv ID:** 2604.26920 | [PDF](https://arxiv.org/pdf/2604.26920v1)

**作者:** David Novikov `[一作]` (Weizmann Institute of Science), Mark Sheinin `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 163 | [OpenAlex ID](https://openalex.org/A5079263739)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用低速相机结合高频彩色闪烁照明，提出一种能将多帧高速度场景编码进单帧并通过动态高斯分散重建体积高速度视频的方法。

**💡 创新点**

首次将多彩色闪烁编码与动态高斯分散（Gaussian‑Flow）结合，实现多视角无硬件改造的体积高速重建。

**🔧 技术方法**

彩色闪烁照明、动态高斯分散渲染、深度正则化、多视角相机标定以及基于颜色解码的高速度视频恢复。

**📊 数据集**

使用自制的8相机系统和RGB LED照明，模拟场景以及真实实验（Nerf子弹、棋子、旋转盘等）进行评估。

**📈 对比分析**

通过与标准光照下的低帧率录像对比，证明在60fps相机下可恢复至600fps的运动；在不同闪烁帧数、光照、颜色和相机数量下做定量误差分析，表明帧数多时误差增大，环境光强度升高导致误差线性增长，至少需6台相机即可获得较好视角合成。

**⚠️ 局限性**

仅适用于均匀反射率物体，受背景亮度、颜色空间分辨率和环境光影响；对彩色物体恢复不如单色对象；需要黑色背景，且对复杂运动可能出现帧丢失。

---

## 409. Causal Learning with Neural Assemblies

**arXiv ID:** 2604.26919 | [PDF](https://arxiv.org/pdf/2604.26919v1)

**作者:** Evangelia Kopadi `[一作]` (Hellenic Open University), Dimitris Kalles `[通讯]` (Hellenic Open University)

**通讯引用:** 1903 | [OpenAlex ID](https://openalex.org/A5068179203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

在已知因果图的监督设置下，使用神经元集合作为可审计的表示层，学习并验证变量间的因果方向性。

**💡 创新点**

提出 DIRECT 机制，即在已稳定的神经元集合同步激活时，通过局部可塑性增益调节实现方向性绑定，无需全局误差反向传播，提供可追踪的因果证据。

**🔧 技术方法**

采用 k‑Winner‑Take‑All 竞争、Hebbian/ STDP 本地可塑性、适应性暖升坡调度、以及两种可读出的指标（突触强度不对称与传播重叠）进行方向学习与评估。

**📊 数据集**

主要使用 Alzheimer 结构因果模型（10 变量、12 边）和教育学生退学模型，均以已知 DAG 为监督目标。

**📈 对比分析**

通过 Top‑K 结构恢复、精度召回率以及传播重叠一致性验证，实验显示 Precision@K 与 Recall@K 均达到 1.0，且对扰动与编码分离度鲁棒。

**⚠️ 局限性**

仅在已给定 DAG 的监督场景下有效，无法从观测数据自发发现因果结构，对外推（OOD）与潜在混杂的处理仍有限。

---

## 410. Safe Navigation using Neural Radiance Fields via Reachable Sets

**arXiv ID:** 2604.26899 | [PDF](https://arxiv.org/pdf/2604.26899v1)

**作者:** Omanshu Thapliyal `[一作]` (Hitachi America Ltd.), Ravigopal Vennelakanti `[通讯]` (Hitachi America Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了在使用神经辐射场（NeRF）生成的障碍物和机器人几何模型的基础上，通过可达集约束实现安全导航路径规划。

**💡 创新点**

创新点在于将NeRF得到的3D几何通过凸包多边形化后与可达集的多边形近似相结合，形成线性矩阵不等式约束，从而在模型预测控制中一次性消除非凸碰撞约束；同时利用可达集的时间“扫描”实现对未来T秒内相对配置空间的动态扩展。

**🔧 技术方法**

使用技术包括：NeRF训练与点云生成、凸包多边形化、线性系统可达集的多边形近似、Minkowski和运算、模型预测控制（MPC）与线性矩阵不等式求解。

**📊 数据集**

实验数据集：通过单目相机采集的多视角图像，用于训练NeRF以生成机器人、障碍物和目标的3D几何；实验环境为一个边长8单位的立方体空间，随机放置8或10个立方体障碍物。

**📈 对比分析**

比较方法：将本文的可达集+NeRF+MPC方案与传统基于多边形配置空间的路径规划进行对比。实验结果显示，本文方案在安全性（无碰撞）和计算效率（实时可行）上优于传统方法，尤其在高障碍密度场景下更为明显。

**⚠️ 局限性**

限制：仅适用于线性系统，可达集多边形逼近对非凸几何可能不足；NeRF训练需要大量多视角图像，且对动态障碍物的处理尚未考虑；MPC求解仍受限于线性矩阵不等式规模，极端高维场景可能出现求解瓶颈。

---

## 411. Graph-based Semantic Calibration Network for Unaligned UAV RGBT Image Semantic Segmentation and A Large-scale Benchmark

**arXiv ID:** 2604.26893 | [PDF](https://arxiv.org/pdf/2604.26893v1)

**作者:** Fangqiang Fan `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**通讯引用:** 12182 | [OpenAlex ID](https://openalex.org/A5030720334)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对无人机RGB‑热成像图像的语义分割方法，能够在跨模态空间失配和细粒度类别混淆的条件下进行准确分割

**💡 创新点**

创新点在于（1）Feature Decoupling and Alignment Module（FDAM）先将特征拆分为共享结构与私有感知分支，再在共享子空间中进行可变形对齐；（2）Semantic Graph Calibration Module（SGCM）利用层级分类学与共现统计构造类别图并通过图注意力校正细粒度预测；（3）构建了规模最大、细粒度最高的未对齐UAV RGBT语义分割基准URTF

**🔧 技术方法**

使用双流MiT‑B4编码器、可变形卷积、对齐权重的光照感知机制、图注意力网络（GAT）以及多任务损失（对齐、语义、正交、稀疏图正则）

**📊 数据集**

使用新收集的URTF数据集（25,519对RGB‑热图，61类，包含真实与合成样本，且保持自然的跨模态位移）

**📈 对比分析**

在URTF上与19种基准模型对比，GSCNet实现71.04% mIoU，Tail‑16 IoU 60.17%，相比最佳竞争方法提升约4.6–5.4% mIoU，尤其在稀疏类别上提升显著

**⚠️ 局限性**

缺点是模型参数量（160.66M）与推理速度（16.62 FPS）仍高于部分轻量模型，限制了实时无人机部署，且对不同分辨率模态的兼容性尚待改进

---

## 412. Revealing NVIDIA Closed-Source Driver Command Streams for CPU-GPU Runtime Behavior Insight

**arXiv ID:** 2604.26889 | [PDF](https://arxiv.org/pdf/2604.26889v1)

**作者:** Yuang Yan `[一作]` (Queen's University), Ryan Grant `[通讯]` (Queen's University)

**通讯引用:** 1402 | [OpenAlex ID](https://openalex.org/A5009842811)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在用户空间驱动与内核驱动之间安装硬件watchpoint，捕获并重建NVIDIA GPU驱动提交的完整硬件命令流；随后利用该方法对CUDA数据传输（inline DMA与direct DMA）和CUDA Graph执行进行两项案例研究；在案例研究中，先通过自定义命令直接测量DMA引擎性能，随后比较CUDA 11.8与CUDA 13.0在Graph链长度变化时的提交成本与命令尺寸。

**💡 创新点**

①首次公开展示闭源NVIDIA用户空间驱动产生的低层硬件命令流；②提出利用CPU调试寄存器watchpoint精准捕获提交边界的技术；③通过命令重建实现对DMA模式的独立测量与驱动开销拆分；④揭示CUDA Graph新版本通过缩小命令指纹来显著降低CPU提交延迟的机制。

**🔧 技术方法**

- NVIDIA开源的OpenGPU内核驱动；- 在内核驱动中插装内存映射路径；- 利用CPU硬件watchpoint拦截GPU门铃（doorbell）写；- 对推送缓冲区(Pushbuffer)、GPFIFO以及相应MMU表进行物理/虚拟地址转换；- 通过自定义命令序列直接驱动DMA引擎；- Nsight Systems等性能分析工具用于对比。

**📊 数据集**

未使用公开数据集；采用人工合成的内存传输尺寸（4 B–32 MiB）与CUDA Graph链长度（1–2000）进行实验；硬件平台为NVIDIA A40（Ampere）与Intel Xeon Gold 6338 CPU。

**📈 对比分析**

对比方法：将自定义命令测得的原始DMA时延与Nsight报告的“CUDA HW”时延做百分比比较；对CUDA Graph，比较不同CUDA版本下CPU提交时间、总命令大小以及门铃写次数；结果显示：inline DMA启动延迟约24 ns，复制引擎约500 ns；在大尺寸下，原始DMA与Nsight时延差距显著；CUDA 13.0的Graph提交几乎不随链长增长，命令尺寸增长亦明显缓慢，说明新版驱动通过压缩命令流实现高效提交。

**⚠️ 局限性**

- 方法受限于OpenGPU内核驱动，难以直接迁移至尚未开源或改版的内核驱动；- 仅针对单一GPU型号（A40），对不同GPU架构或多GPU场景验证不足；- 对用户空间驱动内部的非公开字段仍不完全透明，部分命令语义无法精确解释；- 需要在Linux内核中开启特定调试功能，使用成本较高；- 仅能揭示驱动层面开销，无法直接优化驱动实现。

---

## 413. Uncertainty-Aware Pedestrian Attribute Recognition via Evidential Deep Learning

**arXiv ID:** 2604.26873 | [PDF](https://arxiv.org/pdf/2604.26873v1)

**作者:** Zhuofan Lou `[一作]` (Sichuan University), Pingyu Wang `[通讯]` (Sichuan University)

**通讯引用:** 321 | [OpenAlex ID](https://openalex.org/A5046717491)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了UAPAR框架，实现了行人属性识别的基于证据深度学习的可不确定性感知模型。

**💡 创新点**

创新点包括：①将Evidential Deep Learning（EDL）引入PAR，用Beta分布预测属性概率并量化先验不确定性；②设计Region‑Aware Evidence Reasoning模块和空间先验掩码，实现属性级的局部证据聚焦；③提出基于不确定性的双阶段课程学习，利用Gaussian加权动态切换训练难度。

**🔧 技术方法**

技术细节：CLIP视觉‑语言预训练模型+深度提示调优；跨注意力+空间先验掩码；EDL Beta推理；基于不确定性的Gaussian课程权重和AWR正则；ViT‑L/14视觉骨干。

**📊 数据集**

使用了四大行人属性识别基准数据集：PA100K、PETA、RAPv1、RAPv2。

**📈 对比分析**

与多种SOTA方法比较，UAPAR在PA100K、PETA、RAPv1、RAPv2上分别取得88.48%、90.74%、87.48%和85.71%的平均准确率（mA），刷新了现有最优；在零样本场景下也表现出较高的mA和F1，表明泛化能力强。

**⚠️ 局限性**

局限性：对极少数属性的极端长尾分布仍表现不佳，缺乏针对稀有属性的专门平衡学习策略。

---

