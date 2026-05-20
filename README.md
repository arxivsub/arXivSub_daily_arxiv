# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-20 | 今日论文总数: 726

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. CosFly: Plan in the Matrix, Fly in the World

**arXiv ID:** 2605.19120 | [PDF](https://arxiv.org/pdf/2605.19120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 2. IterSIMP-σ: Evaluating LLM-Assisted Spatial Interventions in Stress-Aware Topology Optimization

**arXiv ID:** 2605.19110 | [PDF](https://arxiv.org/pdf/2605.19110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 3. D-PACE: Dynamic Position-Aware Cross-Entropy for Parallel Speculative Drafting

**arXiv ID:** 2605.18810 | [PDF](https://arxiv.org/pdf/2605.18810v1)

**作者:** Tianyu Wu `[一作]` (Harvard), Yilun Du `[通讯]` (Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的训练目标D-PACE，针对并行单轨段式drafters的Speculative Decoding，通过自适应位置权重提升被接受的token长度和推理速度。

**💡 创新点**

创新点在于从可微的被接受长度代理导出动态位置权重，并将权重与交叉熵分离，消除传统固定指数衰减带来的信号分配不均。

**🔧 技术方法**

技术包括对接受长度代理 S̃ 的梯度推导、异步权重平滑、基于权重的加权交叉熵损失，以及在 DFlash 框架下的并行 block drafting。

**📊 数据集**

使用了多种基准数据集：Math（GSM8K、MATH‑500）、Code（HumanEval、MBPP）和 Chat（MT‑Bench、Alpaca），以及 Qwen3‑4B‑Instruct‑100K、ShareGPT 等训练语料。

**📈 对比分析**

与 DFlash 的指数衰减基线、Top‑3 前缀掩码和 Accept‑rate 等方法比较，D‑PACE 在所有六个基准上均实现了 8–12% 的平均速度提升和 8–13% 的平均已发长度提升，训练时开销仅 2.3%。

**⚠️ 局限性**

局限性包括仅适用于一次性生成所有 token 的并行 block drafter，无法直接迁移到序列化 drafting；代理信号对早期训练时的接受匹配度可能不足；仅评估了无损 Speculative Decoding。

---

## 4. Computing Certificates in Archimedean Univariate Saturated Quadratic Modules

**arXiv ID:** 2605.18980 | [PDF](https://arxiv.org/pdf/2605.18980v1)

**作者:** Jose Abel Castellanos-Joo `[一作]`, Deepak Kapur `[通讯]` (University of New Mexico)

**通讯引用:** 6221 | [OpenAlex ID](https://openalex.org/A5003856856)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的符号算法，用于计算非负单变量多项式在饱和单变量二次模中的成员资格的平方和乘子（证书）。

**💡 创新点**

创新点在于使用自然生成元来计算证书，并且提出了一种新的算法来处理非负多项式的情况，特别是与现有方法相比，能够处理严格正多项式以外的情况。

**🔧 技术方法**

使用了Kuhlmann和Marshall提出的自然生成元以及基本引理的构造方法来计算证书。

**📊 数据集**

使用了由有限生成元生成的单变量多项式的饱和二次模作为数据集。

**📈 对比分析**

与现有的计算证书的方法进行了比较，结果表明，提出的方法在某些情况下能够成功找到证书，而现有方法则无法找到。

**⚠️ 局限性**

限制在于该方法主要针对单变量情况，可能在多变量情况下的应用效果不佳。

---

## 5. POLAR-Bench: A Diagnostic Benchmark for Privacy-Utility Trade-offs in LLM Agents

**arXiv ID:** 2605.19127 | [PDF](https://arxiv.org/pdf/2605.19127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 6. Exact Linear Attention

**arXiv ID:** 2605.18848 | [PDF](https://arxiv.org/pdf/2605.18848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 7. MMGS: 10$\times$ Compressed 3DGS through Optimal Transport Aggregation based on Multi-view Ranking

**arXiv ID:** 2605.19304 | [PDF](https://arxiv.org/pdf/2605.19304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 8. DarkLLM: Learning Language-Driven Adversarial Attacks with Large Language Models

**arXiv ID:** 2605.18868 | [PDF](https://arxiv.org/pdf/2605.18868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 9. The 99% Success Paradox: When Near-Perfect Retrieval Equals Random Selection

**arXiv ID:** 2605.18857 | [PDF](https://arxiv.org/pdf/2605.18857v1)

**作者:** Vyzantinos Repantis `[一作]` (Meta Platforms Inc.), Ameya Gawde `[通讯]` (Meta Platforms Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Bits-over-Random（BoR）指标，用来衡量检索系统相对于随机选择的选择性，揭示高成功率下可能出现的“99%悖论”。

**💡 创新点**

创新点在于将检索成功率与随机基线进行对比，给出信息量化选择性度量，并给出基于λ=K·R̅q/N的选择性崩塌阈值，说明在特定深度和相关性密度下即使检索成功率为100%也无意义。

**🔧 技术方法**

采用超几何分布的随机基线、信息论的比特量化、稀疏命中近似（Poisson/二项式）以及深度校准识别公式，结合大规模检索与LLM生成的下游评估。

**📊 数据集**

实验数据集包括：20 Newsgroups（高相关性密度）、BEIR SciFact（稀疏相关性）、MS MARCO Passage Ranking（海量文档）。

**📈 对比分析**

通过将BoR与传统Recall@K、Precision@K、nDCG等指标对比，发现即使Recall提高13个百分点，BoR差异仅0.2比特；在20 Newsgroups 20%相关时，K=100的BoR接近0，比特几乎为零，表明成功率不代表有效选择。

**⚠️ 局限性**

局限性包括：目前只验证单证据（m=1）成功规则，需进一步扩展到多证据场景；BoR对查询相关性数Rq的估计敏感；在极度稀疏或极度密集的环境下近似假设可能失效；工具选择的实验仅为案例演示，缺乏完整的端到端验证。

---

## 10. Dimensional Balance Improves Large Scale Spatiotemporal Prediction Performance

**arXiv ID:** 2605.18793 | [PDF](https://arxiv.org/pdf/2605.18793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 11. Agentic GraphRAG: Navigating Unstructured Financial Data with Collaborative AI

**arXiv ID:** 2605.18770 | [PDF](https://arxiv.org/pdf/2605.18770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 12. Agent Security is a Systems Problem

**arXiv ID:** 2605.18991 | [PDF](https://arxiv.org/pdf/2605.18991v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 13. How Faithful Is Trajectory-Based Data Attribution? Error Sources, Remedies, and Practical Guidelines

**arXiv ID:** 2605.18814 | [PDF](https://arxiv.org/pdf/2605.18814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 14. Does Your Wildfire Prediction Model Actually Work, or Just Score Well?

**arXiv ID:** 2605.18911 | [PDF](https://arxiv.org/pdf/2605.18911v1)

**作者:** Yangshuang Xu `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种专门针对野火预测的基础模型，并构建了固定合同评估框架；

**💡 创新点**

创新点在于在多模态野火相关数据上进行领域特定预训练，并通过固定合同协议消除匹配规则与头部选择对评估结果的干扰；

**🔧 技术方法**

技术手段包括使用紧凑 U‑Net 作为骨干网络、稀疏标记的类加权二元交叉熵、空间支持输出以及基于排名和决策的头部选择与两种控制检查；

**📊 数据集**

使用的数据集涵盖加利福尼亚地区 NOAA HRRR 气象字段、NASA FIRMS 活火探测、LANDFIRE 燃料与冠层、Wildfire Risk to Communities 住房密度、LandScan 人口分布，以及 WFIGS 与 MTBS 的事件与烧毁面积记录；

**📈 对比分析**

在共享的评估合同下，将该模型与十个通用 Earth-FM 进行比较，评估指标包括占据 F1、扩散 F1、AP、检索 nDCG、回归 RMSE 等。实验表明专门化模型在多数任务上优于或持平基线，但性能高度依赖匹配规则和头部选择，评估结果在不同任务形式和范围内显著变化；

**⚠️ 局限性**

局限性包括评估仅涵盖选定的任务形式、范围与匹配规则，结果可能不适用于更广泛的地理区域或不同的评估设置，且未覆盖所有可能的野火预测任务。

---

## 15. Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees

**arXiv ID:** 2605.18775 | [PDF](https://arxiv.org/pdf/2605.18775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 16. When Individually Calibrated Models Become Collectively Miscalibrated

**arXiv ID:** 2605.18858 | [PDF](https://arxiv.org/pdf/2605.18858v1)

**作者:** Zhaohui Wang `[一作]` (University of Southern California), Zhaohui Wang `[通讯]` (University of Southern California)

**通讯引用:** 28133 | [OpenAlex ID](https://openalex.org/A5100358029)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在多智能体环境下，个体校准的概率预测在战略互动中可能导致整体误校准；

**💡 创新点**

提出VCG机制作为对抗此类误校准的机制设计方案，并证明其在策略不对称情况下保持优势；

**🔧 技术方法**

利用机制设计中的VCG机制、Brier分数、外部性机制、在线乘子权重更新等技术；

**📊 数据集**

在NSL-KDD、UNSW-NB15、信用卡欺诈等真实数据集上进行实验，使用特征划分和样本划分的多智能体模型；

**📈 对比分析**

与多数基线（多数投票、堆叠、加权平均等）相比，VCG在误报率、价格无效性（PoA）以及低样本、分布漂移和对抗性场景中表现更稳健，误报率下降至0.01-0.02；

**⚠️ 局限性**

局限性包括对极稀有事件的检测仍受限、VCG计算复杂度随智能体数增长、需要完整的结果反馈以实现在线权重更新。

---

## 17. STRIDE: Learnable Stepwise Language Feedback for LLM Reasoning

**arXiv ID:** 2605.18851 | [PDF](https://arxiv.org/pdf/2605.18851v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 18. Guardrail Selection in Line Charts to Contextualize Persuasive Visualizations

**arXiv ID:** 2605.19017 | [PDF](https://arxiv.org/pdf/2605.19017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 19. MedFM-Robust: Benchmarking Robustness of Medical Foundation Models

**arXiv ID:** 2605.19027 | [PDF](https://arxiv.org/pdf/2605.19027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 20. The impact of observation density on Bayesian inversion of latent dynamics in shock-dominated flows

**arXiv ID:** 2605.19076 | [PDF](https://arxiv.org/pdf/2605.19076v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. Swimming with Whales: Analysis of Power Imbalances in Stake-Weighted Governance

**arXiv ID:** 2605.19264 | [PDF](https://arxiv.org/pdf/2605.19264v1)

**作者:** Yuzhe Zhang `[一作]` (Independent researcher), Davide Grossi `[通讯]` (University of Groningen and University of Amsterdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究基于质押权重的投票在区块链治理中的权力失衡问题，理论分析与实证评估。

**💡 创新点**

提出单个代理权力-质押比率的期望与方差理论模型，并证明不存在完美比例权力分配；将 Gamma/Dirichlet 分布与 Banzhaf 指数结合，量化权力失衡。

**🔧 技术方法**

计算社会选择理论中的 Penrose‑Banzhaf 指数、Dirichlet/Gamma 分布理论、期望与方差解析推导，以及 Monte Carlo 仿真估计。

**📊 数据集**

Cardano 项目 Catalyst 第 13 轮基金的质押和投票数据。

**📈 对比分析**

通过 Monte Carlo 计算 Banzhaf 指数，比较不同配额 θ 与质押分布下的均值与方差，实验结果与理论一致，发现 θ≈0.5 时失衡最小。

**⚠️ 局限性**

仅分析线性权重分配，未给出可行的治理改进方案；对极端质押集中和稀疏投票的实际治理效果仍需进一步研究。

---

## 22. Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis

**arXiv ID:** 2605.18798 | [PDF](https://arxiv.org/pdf/2605.18798v1)

**作者:** Taiki Miyagawa `[一作]` (NEC Corporation), Akinori F. Ebihara `[通讯]` (NEC Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了非参数的平均运行长度（ARL）和平均检测延迟（ADD）估计器——KM-ARL 与 KM-ADD，专为有限且不规则长度的在线快速检测（QCD）设计；

**💡 创新点**

创新点在于将 Kaplan‑Meier 生存分析方法引入 QCD，构造可在无外推时无偏、能够处理序列截断的估计器，并给出了其偏差上界和渐近无偏性证明；

**🔧 技术方法**

使用了 Kaplan‑Meier 估计、非参数生存分析框架、理论偏差分析、Monte‑Carlo 仿真以及 Python 实现；

**📊 数据集**

实验数据包括仿真高斯与泊松过程，以及真实世界的 WISDM Actitracker 人类活动识别序列；

**📈 对比分析**

与传统 LB‑ARL/LB‑ADD、Naïve ARL 等方法比较，实验表明 KM‑ARL/KM‑ADD 在有限/不规则序列下估计误差更小、方差更低，能够更准确绘制 ARL‑ADD 权衡曲线；

**⚠️ 局限性**

局限性：依赖独立截断假设；在高度不平衡或严重截断情形下仍会产生截断偏差；多变更检测或变更分类未覆盖；若需外推则需结合参数模型，可能受模型假设限制。

---

## 23. Learning Long-Term Temporal Dependencies in Photovoltaic Power Output Prediction Through Multi-Horizon Forecasting

**arXiv ID:** 2605.19074 | [PDF](https://arxiv.org/pdf/2605.19074v1)

**作者:** Sumit Laha `[一作]` (University of Central Florida), Hassan Foroosh `[通讯]` (University of Central Florida)

**通讯引用:** 5138 | [OpenAlex ID](https://openalex.org/A5076117344)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出并验证了一种多时域预测框架，将传统单点预测扩展为联合优化多步未来功率，从而提升光伏功率预测精度。

**💡 创新点**

创新点在于实现了架构无关的多时域学习，通过联合优化缓解早期收敛和滤波器多样性退化，显著捕捉长时序依赖。

**🔧 技术方法**

使用深度学习直接预测方法，结合连续天象图像序列与历史功率数据，采用SUNSET与MobileNet两种CNN架构，并以MSE为损失、Cosine Annealing学习率调度训练模型。

**📊 数据集**

采用Stanford University的SKIPP'D数据集，包含2017‑2019年间以1分钟间隔采集的360°天象全景图像与对应的光伏功率记录。

**📈 对比分析**

通过10折交叉验证、MAE/RMSE/R²等指标与单点预测对比，多时域预测在15/30/60分钟预报中平均RMSE分别下降约9%、5.6%和3.4%，MAE显著降低，R²提升，验证了更高的预测准确性。

**⚠️ 局限性**

局限性包括仅在加州气候下的住宅光伏小规模数据上训练，缺乏多样气象与大规模系统覆盖，且实验仅限CNN模型，未探讨Transformer、气象变量融合等更先进方法。

---

## 24. A Geometric Analysis of Sign-Magnitude Asymmetry in a ReLU + RMSNorm Block under Ternary Quantization

**arXiv ID:** 2605.18933 | [PDF](https://arxiv.org/pdf/2605.18933v1)

**作者:** Lei Dong `[一作]` `[通讯]`, Lei Dong

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了预归一化Transformer中RMSNorm与ReLU对符号量化误差的几何影响，并给出了为什么符号信息在三值量化中能够保留大部分语言能力的理论解释。

**💡 创新点**

创新点在于：①证明符号模式约携带63.7%的方向能量，幅度扰动在随机权重下几乎无方向信息；②通过Fréchet导数显示RMSNorm是径向投影滤波器；③在两层ReLU+RMSNorm模型中推导出符号翻转扰动与幅度扰动横向能量比为π/(π‑2)≈2.75，并用此解释真实模型中更大符号敏感度主要来源于离散特征而非深度叠加；④给出了三值量化误差的角度对齐与径向分量的精确极限。

**🔧 技术方法**

使用了高斯随机矩阵理论、Bussgang定理、Price定理、随机投影保持角度、Fréchet导数、Lévy集中、不等式、SLLN等数学工具。

**📊 数据集**

实验数据来自TinyLlama‑1.1B、Qwen2.5系列（0.5B–3B）等实际大型语言模型，以及合成的i.i.d.高斯权重和球面输入。

**📈 对比分析**

将理论预测的横向能量比c(p)、RMSNorm能量比、NLL与PPL等指标与实验结果对比，误差在1–6%范围内；通过V0、V1等架构验证理论；在真实模型中将α²比例与NLL提升进行对应，证明离散特征主导符号敏感度。

**⚠️ 局限性**

局限性包括：①理论仅在两层简化模型下严格成立；②假设权重为独立高斯、输入为球面，未涵盖训练权重的非高斯分布；③仅考虑ReLU激活，未覆盖SwiGLU/GELU等门控；④未给出多层传播规律和残差流中的幅度/符号协同作用；⑤对实际量化策略的直接映射和多位量化误差补偿仍需进一步研究。

---

## 25. Graph-Driven Cross-Industry Real-Time Monitoring Framework for Anti-Money Laundering Detection in Converged Mobility-Energy Supply Chain Networks

**arXiv ID:** 2605.18844 | [PDF](https://arxiv.org/pdf/2605.18844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 26. KG-ASG: Collision-Knowledge-Guided Closed-Loop Adversarial Scenario Generation With Primary-Support Attribution

**arXiv ID:** 2605.18895 | [PDF](https://arxiv.org/pdf/2605.18895v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 27. Can LLMs Emulate Human Belief Dynamics?

**arXiv ID:** 2605.18781 | [PDF](https://arxiv.org/pdf/2605.18781v1)

**作者:** Adiba Mahbub Proma `[一作]` (University of Rochester), Ehsan Hoque `[通讯]` (University of Rochester)

**通讯引用:** 3530 | [OpenAlex ID](https://openalex.org/A5091780889)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用12种LLM模拟人类在政治议题中的信念动态实验，检验其与真实人类数据的匹配程度。

**💡 创新点**

首次将数字孪生方法与受限人格信息结合，系统评估LLM在信念分布、信念变化和网络重组中的表现。

**🔧 技术方法**

采用提示工程、KL散度、Wasserstein距离、Spearman相关、Mann-Whitney U检验等统计方法对LLM输出进行量化评估。

**📊 数据集**

利用原始研究的341名受试者数据（共1023个样本），聚焦移民与石油燃料两主题。

**📈 对比分析**

通过对比LLM与人类在初始信念分布、社会影响力、关注信号及同质化距离等指标，结果显示LLM普遍更易受影响、初始分布偏差显著，整体表现低于人类。

**⚠️ 局限性**

受限的人格信息、缺乏跨轮记忆、仅使用开源模型以及样本量有限，限制了LLM在复杂社会网络仿真中的适用性。

---

## 28. Performance Monitoring of Proton Exchange Membrane Water Electrolyzer by Transformers-Based Machine Learning Model

**arXiv ID:** 2605.19107 | [PDF](https://arxiv.org/pdf/2605.19107v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 29. The Growing Pains of Frontier Models: When Leaderboards Stop Separating and What to Measure Next

**arXiv ID:** 2605.18840 | [PDF](https://arxiv.org/pdf/2605.18840v1)

**作者:** Adil Amin `[一作]` `[通讯]` (ZEHEN Labs), Adil Amin (ZEHEN Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出利用SWE‑bench与GPQA Diamond两项公开基准的分数，计算全局耦合系数与h‑field残差，以诊断前沿模型在编码与推理能力之间的协同或权衡。

**💡 创新点**

创新点在于将能力耦合视为可度量的“人口耦合”与“实验室残差”，实现跨实验室、跨时间的能力轨迹可视化，并给出基准轴旋转与预测的实用决策框架。

**🔧 技术方法**

主要技术包括Pearson相关分析、线性回归残差计算、留一实验室交叉验证、ODE式扩展预测与矩阵特征值分析。

**📊 数据集**

使用了包含34款2024–2026年间前沿模型的公开基准分数数据，涵盖10个实验室，主要基准为SWE‑bench、GPQA Diamond、HLE、IFEval等。

**📈 对比分析**

相较于单纯的排行榜或单一基准评估，本文通过耦合分析揭示了模型间的协同结构，预测了轴旋转与能力转移，验证了预测准确率达95%以上，且在新发布模型上保持稳定。

**⚠️ 局限性**

局限在于数据以自报为主，实验室不平衡、样本量有限，h‑field只能描述偏差而非因果，且基准轴可能随时间失效，需要持续更新。

---

## 30. DynaTrain: Fast Online Parallelism Switching for Elastic LLM Training

**arXiv ID:** 2605.18815 | [PDF](https://arxiv.org/pdf/2605.18815v1)

**作者:** Yuanqing Wang `[一作]` (Peking University), Yu Wang `[通讯]` (Tsinghua University)

**通讯引用:** 45385 | [OpenAlex ID](https://openalex.org/A5100445300)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种能够在毫秒级完成任意多维并行配置切换的分布式 LLM 训练系统，支持在线热切换和设备弹性扩缩。

**💡 创新点**

核心创新点包括：1) 虚拟参数空间 (VPS) 抽象，将所有分布式状态统一映射为几何区域；2) 基于 VPS 的路由规划与状态转换引擎，实现无检查点的高效热切换；3) 弹性设备管理层，实现进程组热更新和资源变更的异步覆盖。

**🔧 技术方法**

技术手段包括：VPS 的几何映射与交集计算；状态路由规划与 M-to-N 交互映射；内存感知的连续缓冲与异步 XOR 调度；集群级设备管理与动态进程组重建；实现基于 PyTorch + Megatron‑LM 的可插拔中间件。

**📊 数据集**

使用数据集：英文维基百科；模型：GPT‑3（1.3B/2.7B/6.7B）、LLaMA‑2（7B/13B/70B）以及 Qwen3‑MoE（30B、235B）。

**📈 对比分析**

与 Tenplex、HotSpa、Megatron‑LM 分布式检查点（MCP）等基线比较，单维度重构、混合维度重构及任务迁移/动态扩缩等场景下，重构时间从数十秒降至 0.4–4.4 秒，速度提升可达 30–900 倍；在 70B 大模型上保持 2 秒以内，远优于基线。

**⚠️ 局限性**

局限性：当前实现需要训练框架显式注册状态，主要针对 PyTorch + Megatron‑LM；对极大模型的内存调度仍可能出现瓶颈；弹性扩缩仍需外部触发，缺乏完全自动化；未针对所有可能的并行维度组合提供完整验证。

---

## 31. STAR: Semantic-Tuned and Tail-Adaptive Retriever for Graph-Augmented Generation

**arXiv ID:** 2605.18765 | [PDF](https://arxiv.org/pdf/2605.18765v1)

**作者:** Shuai Li `[一作]`, See-Kiong Ng `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种用于 GraphRAG 的轻量级检索器 STAR，旨在通过更细粒度的语义建模提升多跳问答性能。

**💡 创新点**

核心创新在于：①跨注意力（cross‑attention）实现查询与路径的 token‑级交互，缓解语义快捷路（Semantic Shortcut）偏差；②基于路径权重的对比学习（path‑weighted contrastive learning）自适应强调稀有尾部路径，解决长尾路径偏差；③结合硬路径挖掘（Hard Path Mining）构造对抗性训练样本，提升模型鲁棒性。

**🔧 技术方法**

技术实现包括单塔 PLM 架构（如 SimCSE/RoBERTa），交叉注意力模块、硬路径挖掘策略、图上下文路径权重调度的对比损失，以及 Beam Search 检索策略。

**📊 数据集**

实验使用三大 Web 来源多跳 QA 基准：WebQSP、CWQ、GrailQA；每个数据集均包含 1 跳、2 跳及 ≥3 跳样本。

**📈 对比分析**

与 LLM‑centric 检索器（RoG、Graph CoT、ToG）、轻量级检索器（DALK、G‑Retriever、RD‑P、GRAG）以及无 KG 的 LLM 基线进行对比；STAR 在 Hits@1 平均提升 1.8%（如 WebQSP 88.7%，CWQ 86.3%，GrailQA 80.1%）且 F1 平均提升 2.2%，检索速度约 1 s/问，显著优于其它方法。

**⚠️ 局限性**

局限性包括：仍比某些轻量级模型（如 RD‑P）略慢；对极端长尾或极稀缺路径的召回仍有限；实验主要集中在 Freebase 衍生的 Web 图谱，未评估更大规模或非结构化 KG 的泛化能力。

---

## 32. Time to REFLECT: Can We Trust LLM Judges for Evidence-based Research Agents?

**arXiv ID:** 2605.19196 | [PDF](https://arxiv.org/pdf/2605.19196v1)

**作者:** Leyao Wang `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 7621 | [OpenAlex ID](https://openalex.org/A5064858748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个名为REliable Fine-grained LLM judge Evaluation via Controlled inTervention（RIFiT）的元评估基准，用来系统评估大语言模型（LLM）在深度研究代理（deep research agent）执行过程和报告结果上的判定可靠性。

**💡 创新点**

创新点在于：①构建了覆盖过程级与结果级的细粒度错误分类体系，并通过受控局部干预生成验证性参考与故障对照样本；②将元评估任务从粗粒度的偏好匹配转化为明确的失败检测任务，确保标签可验证；③在同一对照下对多种评估接口（标量、对比、排名）和评估细粒度（整体 vs 局部）进行系统比较，揭示细粒度评估与基准模型能力之间的关系。

**🔧 技术方法**

技术手段包括：使用LLM编辑器在保持整体流畅的前提下局部插入指定错误；自动过滤与人类验证相结合的双重筛选流程；对不同评估接口实现标准化的评价指标（准确率、Δ_scale、最佳‑N选择准确率）；以及在多模型、多提示（含rubric与CoT）下进行大规模实验。

**📊 数据集**

数据集主要来源于：1）DR.TULU 与 Tongyi DeepResearch 提供的高质量代理轨迹；2）DeepResearch Bench 的最终报告；3）通过这些轨迹构造的过程级和结果级错误样本。所有样本经过自动过滤后由两名研究生级别注释员校对，得到高达 κ=0.86 的一致率。

**📈 对比分析**

比较方法：在标量、对比和排名三种评估接口下对 13 种LLM（包括 8 种开源、5 种闭源）进行单对比、细粒度与整体评估、rubric 与非rubric、CoT 与非CoT 的交叉实验。结果显示：最佳模型在 reasoning、tool‑use 与 report 质量上的准确率分别为 55.7%、54.5%、47.5%；细粒度评估比整体评估提升约 20–30%（Δ_scale），rubric 对结果评估带来显著提升，CoT 仅在强模型与结果级评估时有效；最佳‑N 选择准确率相对较低，说明判定在多候选情形下更具挑战。

**⚠️ 局限性**

局限性：①错误分类体系虽覆盖常见失败，但无法囊括所有领域特定或交互式失效模式；②受控干预生成的错误可能与自然出现的错误分布不完全一致，导致评估结果偏离真实使用场景；③随着LLM 与研究代理技术演进，基准需要持续更新，以保持评估与实际系统的匹配。

---

## 33. Emergence of a Flow-Assisted Casting Strategy for Olfactory Navigation via Memory-Augmented Reinforcement Learning

**arXiv ID:** 2605.18881 | [PDF](https://arxiv.org/pdf/2605.18881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 34. Are Rationales Necessary and Sufficient? Tuning LLMs for Explainable Misinformation Detection

**arXiv ID:** 2605.19285 | [PDF](https://arxiv.org/pdf/2605.19285v1)

**作者:** Bing Wang `[一作]` (Jilin University), Jieping Ye `[通讯]` (Alibaba Group)

**通讯引用:** 40229 | [OpenAlex ID](https://openalex.org/A5010419481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于大型语言模型的可解释谣言检测数据生成与筛选管道，并用筛选出的高质量“必要且充分”推理链进行轻量级LLM的监督微调。

**💡 创新点**

提出了LonsRex——一种利用自回归与互相归因（perplexity‑based）度量来判断推理步骤的必要性与充分性，并据此过滤训练样本，显著减少冗余与过度验证的推理过程。

**🔧 技术方法**

核心技术包括：多模型生成推理链、基于掩码的贡献度计算、self‑attribution 与 mutual‑attribution 分数、K‑means 聚类对推理视角建模、LLM微调（Qwen3‑4B‑Instruct、Qwen2.5‑1.5B‑Instruct、Gemma2‑2B‑it）以及对比基线提示方法（CoT、ARG、GenFend、DMR、PCoT）。

**📊 数据集**

使用316k条公开 fact‑checked 语料（来自 10+ 开源数据集），并在此基础上生成约200k条高质量推理链，最终微调 1.5B–4B 规模 LLM。

**📈 对比分析**

在四大谣言检测基准（GossipCop++、PolitiFact++、MultiDis、EUDisinfo）上，微调模型平均提升约19–22% 以上，相比同类提示方法更优，甚至可与 32B‑70B 规模的开源 LLM 达到同等或更好性能；在 Token 消耗上也显著低于 CoT/知识检索方案。

**⚠️ 局限性**

主要局限：仍依赖原始 LLM 生成的推理链质量，无法完全消除所有噪声；对极端长链或多模态信息支持有限；目前只针对二分类真/假，无法直接扩展到多标签或细粒度真伪评估。

---

## 35. COBALT: Crowdsourcing Robot Learning via Cloud-Based Teleoperation with Smartphones

**arXiv ID:** 2605.19138 | [PDF](https://arxiv.org/pdf/2605.19138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 36. Trustworthy Agent Network: Trust in Agent Networks Must Be Baked In, Not Bolted On

**arXiv ID:** 2605.19035 | [PDF](https://arxiv.org/pdf/2605.19035v1)

**作者:** Yixiang Yao `[一作]` (University of Southern California), Carlee Joe-Wong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17472 | [OpenAlex ID](https://openalex.org/A5003037377)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

研究大语言模型代理协作网络（Agent-to-Agent, A2A）中的信任缺陷，提出可信代理网络（Trustworthy Agent Network, TAN）框架，并以四大设计支柱（组合鲁棒性、语义约束、可追溯性、跨界可靠性）对现有技术进行评估。

**💡 创新点**

创新点在于将信任从后置的补丁性措施提升为系统级的架构设计：提出从一开始就内嵌安全约束的可信代理网络概念，并给出四个支柱与相应评估指标，为多代理系统的安全与可靠性提供统一的理论规范。

**🔧 技术方法**

主要采用理论分析、形式化建模与抽象评估框架。未涉及具体实验实现技术，仅通过文献综述和概念性对比评估。

**📊 数据集**

论文未使用特定数据集，而是基于公开文献和现有案例（如 OpenClaw、Moltbook）进行分析。

**📈 对比分析**

方法比较基于四个支柱的满足度与四类评估指标（推理延迟、资源开销、可扩展性、确定性）。评估结果显示现有技术多为 bolted‑on，虽在局部提升安全性，但在整体信任保证、语义一致性及可追溯性等维度上存在显著不足；未给出数值实验结果。

**⚠️ 局限性**

局限性：框架仍停留在概念层面，缺乏实现细节和实验验证；实现过程中可能面临表达性与安全约束的权衡；对实际大规模系统的可落地性和性能评估尚未展开。

---

## 37. Soft Learning

**arXiv ID:** 2605.18889 | [PDF](https://arxiv.org/pdf/2605.18889v1)

**作者:** Mohammed Aledhari `[一作]` (University of North Texas), Mohamed Rahouti `[通讯]` (Fordham University)

**通讯引用:** 1584 | [OpenAlex ID](https://openalex.org/A5017529726)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Soft Learning 框架，将多种结构多样的学习器（如树、线性、核、神经等）作为专家，利用交叉验证得到的无偏预测通过 NNLS 在概率单纯形上求得最优凸组合，从而得到单个模型。

**💡 创新点**

创新点在于：① 对异构专家的组合给出严格的 oracle inequality，保证组合优于任何单个专家；② 训练成本仅为专家单次训练，无需梯度下降、GPU 或超参数搜索；③ 通过专家间的差异自然产生不确定性量化，提升鲁棒性和可解释性；④ 框架可持续扩展，加入新算法只能提升性能。

**🔧 技术方法**

核心技术包括：多模型专家库构建、5‑fold 交叉验证生成无偏预测、非负最小二乘（NNLS）在单纯形上的凸优化、全样本重新训练、统计检验（Friedman、Nemenyi、Wilcoxon）。

**📊 数据集**

使用 37 个基准数据集（25 个分类、12 个回归），包含 9 个真实世界数据集（如 Iris、Wine、Covtype 等）和 28 个人工合成数据集（噪声、稀疏、非线性、Friedman 等）。

**📈 对比分析**

与 9 种竞争方法（CatBoost、调优 MLP、梯度提升、随机森林、KAN、NeuroSym、Basic MLP、Logistic/Ridge、Best‑of‑3）在 5‑折 CV 下进行公平比较，Soft Learning 在 70% 数据集获得第一名，平均排名 3.12，显著优于大多数基线，尤其在小样本和复杂决策边界上表现突出。

**⚠️ 局限性**

局限性包括：① 内存受限时无法纳入所有专家；② 对纯线性目标时，单纯形约束限制专家权重过度集中；③ 本研究未包含深度特征学习专家，虽然框架可支持；④ 仅评估 tabular 和预特征化数据，需进一步验证在视觉、文本等原始模态上的效果。

---

## 38. INSIGHTS: Demonstration-Based Summaries of Time Series Predictors

**arXiv ID:** 2605.18849 | [PDF](https://arxiv.org/pdf/2605.18849v1)

**作者:** Bar Eini Porat `[一作]` (Technion Israel Institute of Technology), Ofra Amir `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为INSIGHTS的无模型、用户中心的时间序列全局解释方法，生成少量代表性样本以帮助人类快速理解预测模型行为。

**💡 创新点**

创新点在于将可配置的域特定效用函数与时间序列专用的多样性度量结合，既保持了模型无关性，又兼顾重要性与覆盖度；同时实现了线性时间复杂度与低内存占用。

**🔧 技术方法**

核心技术包括：效用函数（趋势、超界、突变偏差）对每个时间窗口赋值；动态时间规整（DTW）计算样本间多样性；贪婪选择策略在多效用视角下迭代挑选代表性子集。

**📊 数据集**

使用了三类数据集：合成季节性预测数据（带四类事件）、PhysioNet Sleep‑EDF 以EEG事件为标注，以及真实股票价格数据用于用户实验；此外还在ICU心率预测任务中做专家访谈。

**📈 对比分析**

与随机、MMD‑Critic、ProtoDash等基线在事件捕获、覆盖率、Diversity、运行时和内存消耗上对比；INSIGHTS‑TW在事件覆盖率与多样性上均达或超越最佳基线，同时在时间与空间效率上明显优于ProtoDash，且在用户实验中提升理解度与满意度。

**⚠️ 局限性**

局限包括：缺乏公开带有预测级事件标注的大规模真实数据；贪婪选择非最优；当前效用函数主要针对单变量序列，未充分考虑多维交互；对极端稀疏事件或复杂多模态场景的适应性待进一步验证。

---

## 39. Detecting and Mitigating Backdoor Attacks in OTA-FL Systems: A Two-Stage Robust Aggregation Scheme

**arXiv ID:** 2605.19253 | [PDF](https://arxiv.org/pdf/2605.19253v1)

**作者:** Xiaoyan Ma `[一作]` (Purdue University), Christopher G. Brinton `[通讯]` (Purdue University)

**通讯引用:** 3163 | [OpenAlex ID](https://openalex.org/A5020399355)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种双阶段鲁棒聚合框架 Trust-Then-Inspect (TTI)，在 OTA-FL 中通过客户端侧的模态感知多指标信任评分与层级聚合相结合，检测并抑制后门攻击。

**💡 创新点**

创新点：① 引入模态感知多指标信任评分并利用贝叶斯优化自动学习权重；② 采用可信多址聚合 (TBMA) 将可疑客户端分离后在服务器端进行逐层梯度检查；③ 加入长期声誉评分机制，逐步淘汰恶意客户端；④ 只对可疑客户端增加通信开销，保持 OTA 传输效率。

**🔧 技术方法**

技术：多指标信任打分与贝叶斯优化、可信多址聚合、梯度统计特征提取、层级聚类（AHC）、大中位数绝对偏差 (MAD) 声誉过滤、层级梯度检验。

**📊 数据集**

数据集：RML2016.10A (语音波形)、AG News (文本)、CIFAR-10 (图像)，并使用 CNN/ResNet9 等模型架构。

**📈 对比分析**

比较方法：与无防御、单指标分层、模型级检查、BEV、FedSAC 等基线对比；在四种后门攻击（bounded‑scaling、Euclidean‑constrained、Cosine‑constrained、Neurotoxin）下，TTI 将 ASR 降至约10%（接近随机水平），主任务准确率 (MTA) 仅略低于无攻击基准；运行时略高但可接受。

**⚠️ 局限性**

局限性：① 需要先行在开发环境中进行贝叶斯优化确定权重；② 依赖客户端能安全上报指标，若恶意者能篡改报告则可信度受损；③ 对极高比例攻击者（接近全部客户端受攻击）效果有限；④ 层级检查在极大模型或极低资源设备上可能产生额外计算与通信负担。

---

## 40. Planner-Admissible Graph-PDE Value Extensions for Sparse Goal-Conditioned Planning

**arXiv ID:** 2605.19185 | [PDF](https://arxiv.org/pdf/2605.19185v1)

**作者:** Shiheng Zhang `[一作]` (University of Washington), Shiheng Zhang `[通讯]` (University of Washington)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5101985636)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究稀疏标签条件下的目标规划问题，将其视为图PDE Dirichlet延拓，并提出规划可行性（planner‑admissibility）证书来保证基于贪婪行动的成功。

**💡 创新点**

创新点在于：①用规划可行性证书取代传统的点误差/L^p误差评估；②证明在图p‑Laplacian族中，绝对最小Lipschitz延拓（p=∞）具有更好的规划可行性，而谐波平均（p=2）则可能导致贪婪失效；③给出谐波失效的随机游走和算子兼容性机制，并在AntMaze图上实证验证。

**🔧 技术方法**

核心技术包括：图p‑Laplacian Dirichlet延拓（p=2到∞）、极限p=∞的midrange迭代、规划可行性证书与填充距离的比较原理、随机游走与谐波测量的关系、以及基于图Eikonal/最短路径的算子兼容性分析。

**📊 数据集**

使用D4RL AntMaze的图化表示（中大尺寸网格），在不同解析率、标签稀疏度和随机种子下生成120个配置，评估每个配置下的贪婪轨迹。

**📈 对比分析**

比较方法为在相同稀疏标签集合下对谐波（p=2）与p=∞延拓进行贪婪轨迹实验，记录成功率和失败模式。结果显示p=∞延拓的成功率约为0.970，高于谐波的0.584，提升约+38.6个百分点；在不同分辨率和标签比例下，p=∞性能始终稳定，而p=2在稀疏或低分辨率下显著退化。

**⚠️ 局限性**

局限性包括：实验仅在离散图提取的AntMaze上进行，未覆盖连续动力学或闭环控制；中间p∈(2,∞)缺乏高效收敛求解器；对极稀疏标签（lf→0）和有向/加权图的推广尚未验证；证书虽在局部可行，但未给出全局误差界；以及缺乏自适应标签选择和中间p的可认证求解器。

---

## 41. Artifact-Bench: Evaluating MLLMs on Detecting and Assessing the Artifacts of AI-Generated Videos

**arXiv ID:** 2605.18984 | [PDF](https://arxiv.org/pdf/2605.18984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 42. Evaluating Memory Condensation Strategies for Coding Agents in Data-Driven Scientific Discovery

**arXiv ID:** 2605.18854 | [PDF](https://arxiv.org/pdf/2605.18854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 43. Embedding by Elicitation: Dynamic Representations for Bayesian Optimization of System Prompts

**arXiv ID:** 2605.19093 | [PDF](https://arxiv.org/pdf/2605.19093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 44. Distance-Aware Muon: Adaptive Step Scaling for Normalized Optimization

**arXiv ID:** 2605.18999 | [PDF](https://arxiv.org/pdf/2605.18999v1)

**作者:** Yury Demidovich `[一作]` (King Abdullah University of Science and Technology), Peter Richtárik `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 13054 | [OpenAlex ID](https://openalex.org/A5036598221)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Muon 类归一化优化器的自适应标量缩放规则，提出三种新的 Muon 变体：距离自适应 Muon、尺度校准 Muon 与距离自由 Muon。

**💡 创新点**

创新点在于：①仅调整 Muon 的标量步长而保持其方向不变；②利用轨迹半径、梯度对齐证书和一次维度的 majorized 搜索实现自适应步长；③在星凸环境下去除对未知距离或局部界限的依赖，得到无距离输入的理论保证。

**🔧 技术方法**

主要技术：非欧氏信赖域框架、线性最小化算子（LMO）、动量（Momentum）、距离自适应与证书驱动的步长选择、一次维度的凸 majorized 搜索。

**📊 数据集**

实验数据集：GPT‑124M 模型在 WikiText‑103 语言建模任务；ViT‑Tiny 模型在 CIFAR‑100 图像分类任务；并在 NanoGPT/WikiText‑2 与 ResNet‑32/CIFAR‑100 上做补充诊断。

**📈 对比分析**

与 AdamW 以及手工调优的固定 Muon 进行对比。自适应 Muon 在 GPT‑124M/WikiText‑103 上验证集损失比固定 Muon 更低，runtime 仅略高 5%；在 ViT‑Tiny/CIFAR‑100 上保持与固定 Muon 相近的性能，且在某些指标上略优。

**⚠️ 局限性**

局限性：①部分理论结果仍需假设轨迹有界或初始子水平集有界；②目前分析仅针对确定性梯度，缺乏完整的随机梯度收敛证明；③距离自由 Muon 仍需手动设置 ρ、λ、M 等正则化超参数，可能影响鲁棒性。

---

## 45. Block-Based Double Decoders

**arXiv ID:** 2605.18807 | [PDF](https://arxiv.org/pdf/2605.18807v1)

**作者:** Asher Labovich `[一作]` (Brown University), Chaitanya Harsha `[通讯]` (Brown University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种双解码器（Block-based Double Decoder）Transformer架构，利用双因果块式注意力掩码实现全标记监督和静态序列打包。

**💡 创新点**

创新点在于：①通过在两层解码器之间加入跨块注意力，既保留了编码器-解码器的KV缓存优势，又兼顾了解码器的训练效率；②引入双因果块式掩码，使每个token在一次前向传播中都获得损失信号，解决了span corruption稀疏监督和动态批量的问题。

**🔧 技术方法**

技术实现包括：双解码器堆叠（context decoder + generation decoder）、双因果块式注意力掩码、使用PyTorch FlexAttention实现双键注意力、宽度缩放的最大化更新参数化（μP）以及跨模型的共享词表。

**📊 数据集**

使用了SlimPajama数据集进行预训练，包含packed文本，并在10%数据上做prefix‑LM微调以统一损失目标。

**📈 对比分析**

在参数范围6.25M–100M、token范围62.5M–1B的规模实验中，与标准Encoder‑Decoder（T5式span corruption）对比，Block-based Double Decoder在相同参数/token配置下损失约0.7 nats更好，且在计算/内存成本上每个token的KV缓存和推理开销可减少约2/3，整体性能与Decoder‑Only接近，且在所有规模下保持相同的缩放规律。

**⚠️ 局限性**

局限性包括：①块划分采用随机采样，可能导致训练损失波动；②实验规模受限于小模型，超大规模的推断与扩展性尚未验证；③对超出当前token预算的进一步推断效益推断需要更多实验验证。

---

## 46. Probabilistic Recursively Feasible Motion Planning Under Uncertain Environments

**arXiv ID:** 2605.19015 | [PDF](https://arxiv.org/pdf/2605.19015v1)

**作者:** Hyeontae Sung `[一作]` (Korea Advanced Institute of Science and Technology), Heejin Ahn `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 556 | [OpenAlex ID](https://openalex.org/A5054872846)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于概率递归可行性的MPC框架（PRF‑MPC），能在不确定、时变环境下保证安全轨迹的递归可行性

**💡 创新点**

通过定义理想预测器的两个性质（真分布一致性和条件不变性），推导预测分布的时空传播，构造安全集包含约束，实现指定概率的递归可行性保证

**🔧 技术方法**

使用高斯分布假设、投影定理求解预测分布均值协方差，切线近似圆形碰撞约束，将随机约束转为确定性线性约束；结合凸优化求解MPC

**📊 数据集**

使用仿真数据：双积分子系统为自车，单积分子系统为障碍车，障碍车速度按正态分布采样，规划时域T=9，风险容忍ε=0.05，递归可行性容忍γ=0.1

**📈 对比分析**

将PRF‑MPC与普通MPC在自动驾驶车道变道任务中对比，PRF‑MPC递归可行率99.2%高于普通MPC的88.2%；但成本（轨迹偏差）和最小安全距离更大，计算时间保持在0.034 s

**⚠️ 局限性**

受限于理想预测器的假设，现有预测器往往无法完全满足，导致理论与实际之间的差距；目前仅验证单一障碍车的情况，尚未广泛验证多障碍、多模式预测情形

---

## 47. SynGR: Unleashing the Potential of Cross-Modal Synergy for Generative Recommendation

**arXiv ID:** 2605.18920 | [PDF](https://arxiv.org/pdf/2605.18920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 48. Evaluating the Utility of Personal Health Records in Personalized Health AI

**arXiv ID:** 2605.18937 | [PDF](https://arxiv.org/pdf/2605.18937v1)

**作者:** Rory Sayres `[一作]` (Google Research), Quang Duong `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文评估了将个人健康记录（PHR）作为上下文提供给大型语言模型（LLM）时，是否能提升其回答消费者健康问题的帮助性、准确性、安全性等质量。

**💡 创新点**

创新点在于：①系统性设计三种查询来源（搜索、聊天模板、患者与医护团队对话）并与1,945份去标识PHR匹配；②提出基于SHARP框架的PHR特定评估指标，覆盖时间、冲突、偏差、误导等16个维度；③展示了简单的自我批判循环能显著降低时间误判和其他错误。

**🔧 技术方法**

使用 Gemini 3.0 Flash 大型语言模型；对比三种输入条件：无PHR、Basic PHR（仅限人群、诊断、药物）和 Full PHR（完整临床记录）。评估使用自动评审者与临床医生两组评分。

**📊 数据集**

数据集包括 2,255 条用户查询，分为三类，并从 1,945 条去标识的 PHR 中挑选、过滤或合成与之匹配的查询；PHR 的平均长度超过 10,000 字。

**📈 对比分析**

比较方法：采用 SHARP 框架的平均评分（范围[-1,1]）和自评/临床评审者的统计显著性检验（配对 t 检验）。结果显示：加入 PHR 后，整体帮助性显著提升（p < 0.001），Basic 与 Full PHR 的帮助性差异不显著；在行动性与动机等子维度亦有显著改善；在安全性、准确性等维度改善幅度不一致。

**⚠️ 局限性**

局限性包括：①评估仅针对与 PHR 相关的查询，不能推广到所有健康问题；②使用的 PHR 主要来自美国机构，样本偏向复杂病例；③评审者来自美国和英国，文化差异可能影响评分；④未探讨所有可能的上下文提供方式（PDF、FHIR、照片等）。

---

## 49. DeRegiME: Deep Regime Mixtures for Probabilistic Forecasting under Distribution Shift

**arXiv ID:** 2605.19231 | [PDF](https://arxiv.org/pdf/2605.19231v1)

**作者:** Kieran Wood `[一作]` (University of Oxford), Stephen J. Roberts `[通讯]` (University of Oxford)

**通讯引用:** 16671 | [OpenAlex ID](https://openalex.org/A5058617210)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种直接多时域概率预测框架(deregime)，通过稀疏变分高斯过程实现潜在不确定性状态（regime）与基线信号的分离，并以软门控方式为每个预测位置分配已学习的重复出现的状态。

**💡 创新点**

创新点包括：1）将门控权重直接嵌入高斯过程协方差（regime‑mixing kernel），实现状态共享与结构化残差协方差；2）采用学生‑t 混合似然捕捉不同尺度和尾部厚度的状态；3）通过确定性 stick‑breaking 门控实现自适应的状态数 pruned；4）将多状态残差与单一均值路径解耦，提升可解释性。

**🔧 技术方法**

技术手段主要有：稀疏变分高斯过程（SVGP）、深度核学习（deep kernel），确定性 stick‑breaking 门控，学生‑t 混合似然，门控共享的残差协方差与尺度/尾部参数，以及基于 Gauss–Hermite 变分推理的 ELBO 训练。

**📊 数据集**

使用的公共长时序预测基准包括：ETT 系列（ETT1、ETT2、ETTm1、ETTm2）、Electricity、Traffic、Weather、Exchange、Illness 以及扩展的 Nasdaq 日收盘序列。

**📈 对比分析**

与同一编码器下的最强对照基线（如 DeepAR/GluonTS Student‑t 头、DKL 单核 GP 以及 TFT 量化回归等）比较，deregime 在 NLPD、CRPS、MSE 三项指标上平均分别提升 20.3%、3.0% 与 4.7%。在所有十个数据集上均保持一致性，尤其对突变、渐进与季节性不确定性转移表现良好。

**⚠️ 局限性**

局限性包括：1）状态可识别仅在置换下唯一，跨种子比较需使用相对尺度；2）稀疏 GP 计算复杂度随诱导点数量 M 的立方级增长；3）目前不提供完整的跨通道残差协方差；4）门控与状态参数可能对训练稳定性与解释性产生影响；5）对极端尾部情况依赖于学生‑t 参数，若数据尾部更重可能需要更灵活的基核。

---

## 50. Emergence of Frontier Superposition: Möbius attractor and Cascade Supervision

**arXiv ID:** 2605.18820 | [PDF](https://arxiv.org/pdf/2605.18820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 51. Bilateral Teleoperation with Compliant 6-DOF Pose-and-Force Sensing

**arXiv ID:** 2605.19255 | [PDF](https://arxiv.org/pdf/2605.19255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 52. Don't Let Bandit Feedback Pull Continual LLM-Recommender Updates Off Target

**arXiv ID:** 2605.18899 | [PDF](https://arxiv.org/pdf/2605.18899v1)

**作者:** Taesan Kim `[一作]` (SK Telecom), Chung Park `[通讯]` (SK Telecom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Anchored Bandit Policy Optimization（ABPO）框架，实现对生成式 LLM 推荐器在部署后政策形态的上下文鲁棒性持续更新。

**💡 创新点**

创新点在于：① 在 GRPO 组中加入已曝光推荐作为锚点进行基线校准；② 对锚点应用 Self‑Normalized Inverse Propensity Scoring (SNIPS) 纠正离线偏差；③ 在面对无响应时引入自我置信度（self‑certainty）奖励，避免将其视为硬负样本。

**🔧 技术方法**

核心技术包括：RLVR / GRPO 策略优化、SNIPS 重要性加权、token‑level self‑certainty 评估、格式一致性奖励与多目标组合。

**📊 数据集**

使用 Amazon Reviews 2023 的五个子域（Fashion、Grocery、Health、Clothing、CDs）及 MovieLens 数据集进行离线模拟和在线 A/B 测试。

**📈 对比分析**

与无更新、DEALRec、RL‑Rec、UL‑Rec、GRPO、GDPO、G²RPO 等基线比较，ABPO 在 HR@1/5、NDCG@5 上均优于多数对照，在在线实验中实现 CTR 从 7.23% 逐月提升至 7.62%。

**⚠️ 局限性**

局限性包括：仍依赖日志中曝光概率的准确估计；锚点策略在极端曝光稀疏时可能不足；自我置信度奖励的阈值需经验调参；且对长期动态变化的适应性需进一步验证。

---

## 53. Operational Memory Architecture for Kubernetes:Preserving Causal Context Across the Evidence Horizon

**arXiv ID:** 2605.18755 | [PDF](https://arxiv.org/pdf/2605.18755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 54. Personalized Face Privacy Protection From a Single Image

**arXiv ID:** 2605.19032 | [PDF](https://arxiv.org/pdf/2605.19032v1)

**作者:** Zachary Yahn `[一作]` (Georgia Institute of Technology), Ling Liu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 21596 | [OpenAlex ID](https://openalex.org/A5100343991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为FaceCloak的三阶段系统，利用单张面部图像生成个性化的防御性面部隐私遮罩，能够在上传前快速叠加到任意用户图像上。

**💡 创新点**

创新点在于：①仅用一张图片即可生成多样化的合成样本；②利用近远锚点对抗学习将身份嵌入迁移到远离原身份的锚点；③融合区域贴纸、高通掩码和可学习注意力等多种聚焦机制，实现更强的防御与更低的可见性。

**🔧 技术方法**

核心技术包括Arc2Face合成、MTCNN关键点检测、对比损失函数、投影投降与PGD迭代、Region-Sticker、High-Pass Mask、Learnable Attention以及最终的像素级叠加。

**📊 数据集**

评估数据集为Privacy-Commons、Privacy-Celebrities（用于身份识别）以及CelebA-HQ（用于1:1验证），并使用十个不同的面部识别模型进行实验。

**📈 对比分析**

与29种身份特定和18种图像特定方法对比，FaceCloak在Top‑1和Top‑5保护成功率（PSR）上均领先，平均提升约9–16%，并保持与对手相当的视觉质量（SSIM/PSNR）。

**⚠️ 局限性**

局限性包括：对部分模型（如MobileNet+ArcFace/Softmax）略逊；对极大扰动预算和极端后处理时鲁棒性不均；需要依赖合成图像的质量，若合成效果不足可能影响保护效果。

---

## 55. Data-Free Client Contribution Estimation via Logit Maximization for Federated Learning

**arXiv ID:** 2605.18892 | [PDF](https://arxiv.org/pdf/2605.18892v1)

**作者:** Asim Ukaye `[一作]` (MBZUAI), Karthik Nandakumar `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于logit最大化的无数据贡献估计方法（CELM），通过服务器端对客户端模型进行类别级的激活最大化探测，构造无偏见的证据矩阵并生成客户端权重，实现对非IID场景下的公平聚合。

**💡 创新点**

1）不依赖任何原始数据、客户端元数据或公共验证集；2）使用类别级logit最大化获取类级证据，避免了相似度到平均的假设；3）通过warm‑up冻结权重和EMA平滑提升聚合稳定性。

**🔧 技术方法**

服务器端logit最大化（Activation Maximization）、类别级证据归一化、EMA平滑、warm‑up冷却期、偏置校正。

**📊 数据集**

FashionMNIST、CIFAR‑10、FedISIC（自然医学图像分割）。

**📈 对比分析**

与FedAvg、CFFL、CGSV、ShapFed比较，CELM在Dirichlet、PLS、SLS等非IID设置下常居榜首，尤其在强标签不均、稀有类别和Maverick场景中表现显著优于基线。

**⚠️ 局限性**

服务器端在warm‑up期间需执行多次logit最大化，随着客户端数或类别数增大会产生计算开销；模型校准不佳可能影响证据质量；对极端异质性或对抗性参与者的理论保证尚未完善。

---

## 56. OmniGUI: Benchmarking GUI Agents in Omni-Modal Smartphone Environments

**arXiv ID:** 2605.18758 | [PDF](https://arxiv.org/pdf/2605.18758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 57. Lost and Found in Translation: Variational Diagnostics for Neural Codebook Channels

**arXiv ID:** 2605.18846 | [PDF](https://arxiv.org/pdf/2605.18846v1)

**作者:** Yusuke Hayashi `[一作]` `[通讯]` (Artificial Life Institute), Yusuke Hayashi (Artificial Life Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了神经代码本通道（Neural Codebook Channel）和代码本一致性（Codebook Agreement）指标，用于检测变分自编码器（VAE）中编码器与解码器在同一潜变量下是否使用相同的离散码，填补了传统 VAE 诊断无法捕捉的“码表不匹配”缺口。

**💡 创新点**

创新点在于：
- 引入了编码器–解码器耦合通道 (j|i) 并证明了其与传统边际诊断（如直方图、活跃码数、互信息）不可替代；
- 通过 KL 链式规则在失配事件上推导出一维伯努利-KL 证书，给出潜变量离散码误读的上界；
- 提供了可操作的代码映射实现（后验聚类 Voronoi、Bregman 最邻近原型），并给出完整的审计流程。

**🔧 技术方法**

核心技术包括：
- 变分自编码器训练与 ELBO 评估；
- 经验分布与理论 KL 链式分解；
- 统计学中的伯努利-KL 不等式；
- 高斯混合聚类、Bregman 散度以及 Voronoi 图判别规则；
- 重要性采样（IWAE）与基于 SNIS 的模型后验估计。

**📊 数据集**

使用了四个低维 sklearn 数据集（digits、wine、breast cancer、two moons）和 MNIST 数据集（Conv‑VAE 与 VQ‑VAE）进行实验，覆盖从离散潜变量到连续潜变量的多种情形。

**📈 对比分析**

与传统诊断（ELBO、rate、活跃码数、互信息等）对比，Neural Codebook Channel 能揭示编码器与解码器的码表不一致，标准诊断往往无法检测到。实验中，在所有数据集的多种随机种子下，所给的伯努利‑KL 证书都被满足；在 MNIST 的 VQ‑VAE 端点，Codebook Agreement 达到 1，验证了理论预测；而在 Conv‑VAE 的可审计阶段，证书虽有显著 slack（约 160 倍），但仍证明了算法的正确性。

**⚠️ 局限性**

局限性：
- 需要预先选择编码器/解码器的硬码映射，映射不同会影响残差和证书效果；
- 伯努利‑KL 证书仅在完全枚举或精确估计变分 gap 时才成立，IWAE 作为上界会导致显著 slack；
- 该指标只能给出上界，无法精确估计实际失配率；
- 在潜变量维度增大或采样质量差时 slack 进一步放大；
- 只适用于满足绝对连续性（q≪p）的 VAE，无法直接用于 deterministic 编码器（如 VQ‑VAE）或非标准模型。

---

## 58. Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization

**arXiv ID:** 2605.18772 | [PDF](https://arxiv.org/pdf/2605.18772v1)

**作者:** Gongbo Zhang `[一作]` (Columbia University), Chunhua Weng `[通讯]` (Columbia University)

**通讯引用:** 12667 | [OpenAlex ID](https://openalex.org/A5009604048)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种响应-行动学习框架RePAIR，直接将RAG错误输出映射到纠错操作，无需显式错误分类。

**💡 创新点**

创新点在于消除了对细粒度错误标签和外部评审的依赖，将诊断与规划统一到单一策略中。

**🔧 技术方法**

技术主要包括两阶段策略学习、基于偏好优化的直接偏好学习（DPO）以及对RAG操作空间的强化学习。

**📊 数据集**

使用了三大问答基准：Natural Questions、2WikiMultiHopQA 和 Wizard of Wikipedia。

**📈 对比分析**

与多种基线（包括Self-Refine、FLARE、RAG-Critic 等）比较，RePAIR 在三个数据集上平均提升 Token‑level F1 3.8 分，领先所有对手。

**⚠️ 局限性**

局限性在于实验仅覆盖三大基准和有限的高层操作集，未评估在更广泛领域或更复杂动作空间下的表现与鲁棒性。

---

## 59. An Integrated Forecasting Prototype for Emergency Department Boarding Time to Support Proactive Operational Decision Making

**arXiv ID:** 2605.18839 | [PDF](https://arxiv.org/pdf/2605.18839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 60. Learning to Hand Off: Provably Convergent Workflow Learning under Interface Constraints

**arXiv ID:** 2605.19140 | [PDF](https://arxiv.org/pdf/2605.19140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 61. GOAL: Graph-based Objective-Aligned Diffusion Solvers for Dynamic Multi-Objective Optimization

**arXiv ID:** 2605.19119 | [PDF](https://arxiv.org/pdf/2605.19119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 62. Trust or Abstain? A Self-Aware RAG Approach

**arXiv ID:** 2605.18792 | [PDF](https://arxiv.org/pdf/2605.18792v1)

**作者:** Xi Zhu `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM隐藏状态的Self-Aware Belief Estimator（SABER），在检索增强生成（RAG）中通过评估参数知识（PK）和检索上下文（CK）的可靠性来做出自适应选择或拒绝回答，显著提升答案可信度；

**💡 创新点**

创新点在于将自先验（knowledge‑boundary awareness）与多轨迹推理的自评（reasoning‑reliability awareness）结合，形成双向可靠性信念并在四格决策空间中实现PK、CK、两者皆可信或完全不可信的判定，且无需对LLM进行微调；

**🔧 技术方法**

技术方法包括：① 从查询隐藏层提取自先验向量；② 通过多轨迹（K=3）chain‑of‑thought生成并对每条轨迹进行自评，得到条件隐藏向量；③ 将自先验与条件向量拼接后输入两条轻量级二分类器，输出PK和CK的可靠性信念；④ 以阈值τ驱动四格决策并支持可调的拒绝（I don't know）；

**📊 数据集**

实验数据集包括五个冲突QA数据集（ConFiQA、ConflictQA、ConflictBank、TriviaQA、Natural Questions），并在四个LLM骨干上构建了≈69K实例的ground‑truth‑aligned PK/CK正确性标注；

**📈 对比分析**

与十种基线（无检索、Vanilla RAG、内部/外部自评、SCR、TACS‑LR、CR‑DPO、R‑Tuning、Prompt‑based abstainers）比较，SABER在整体准确率（Acc）和宏观可信度（MFS）上领先，尤其在冲突频繁的数据集上提升显著；在拒绝模式下，其风险‑覆盖曲线在所有模型上Pareto支配所有prompt‑based abstainers，阈值可调实现覆盖-风险权衡；

**⚠️ 局限性**

局限性包括：① 仅基于隐藏状态的置信度估计，可能对极端噪声或新型冲突场景敏感；② 需要手动挑选层（ℓp, ℓc），尽管对中后层不敏感但仍需经验；③ 在更大规模或多模态模型上的泛化尚未验证；

---

## 63. Robust Mitigation of Age-Dependent Confounding Effects via Sample-Difficulty Decorrelation

**arXiv ID:** 2605.19230 | [PDF](https://arxiv.org/pdf/2605.19230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 64. Example-Driven Intent Synthesis for Constrained Data Bundle Retrieval: Focused Text Snippet Extraction and Beyond

**arXiv ID:** 2605.19246 | [PDF](https://arxiv.org/pdf/2605.19246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 65. ALDEN: Boosting Private Data Extraction from Retrieval-Augmented Generation Systems via Active Learning and Distribution Estimation

**arXiv ID:** 2605.18762 | [PDF](https://arxiv.org/pdf/2605.18762v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 66. Multi-Pedestrian Safety Warning at Urban Intersections Use Case of Digital Twin

**arXiv ID:** 2605.18823 | [PDF](https://arxiv.org/pdf/2605.18823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 67. Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches

**arXiv ID:** 2605.18825 | [PDF](https://arxiv.org/pdf/2605.18825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 68. MotionMERGE: A Multi-granular Framework for Human Motion Editing, Reasoning, Generation, and Explanation

**arXiv ID:** 2605.18956 | [PDF](https://arxiv.org/pdf/2605.18956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 69. Lightweight and Fast Backdoor Model Detection

**arXiv ID:** 2605.18907 | [PDF](https://arxiv.org/pdf/2605.18907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 70. The Annotation Scarcity Paradox in Low-Resource NLP Evaluation: A Decade of Acceleration and Emerging Constraints

**arXiv ID:** 2605.19066 | [PDF](https://arxiv.org/pdf/2605.19066v1)

**作者:** Vukosi Marivate `[一作]` (University of Pretoria), Vukosi Marivate `[通讯]` (University of Pretoria)

**通讯引用:** 1201 | [OpenAlex ID](https://openalex.org/A5060690192)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对过去十年低资源 NLP 评估演进进行批判性叙事综述，提出 Annotation Scarcity Paradox 并分析其三阶段结构性瓶颈

**💡 创新点**

引入 Annotation Scarcity Paradox 概念，系统化低资源评估的结构性困境，并提出从交易型抽取式评估向社区嵌入式、可持续评估的转变路径

**🔧 技术方法**

批判性叙事综述方法，文献检索与主题分析，参考多种低资源 benchmark 与社区工作

**📊 数据集**

主要引用公开的低资源 NLP benchmark（如 MasakhaNER、AfriSenti、XTREME、AfroBench 等）以及相关社区项目与研究文献

**📈 对比分析**

该论文不进行实验性对比，而是综述并对比已有 benchmark 的评估结果，指出现有评估方法的缺陷和技术进步与评估资源不匹配的现实

**⚠️ 局限性**

局限性：聚焦非洲语料和社区，其他地区（东南亚、美洲等）覆盖不足；缺乏对提出方案的实证验证；依赖作者经验与有限文献，未涵盖所有低资源语言生态

---

## 71. SimGym: A Framework for A/B Test Simulation in E-Commerce with Traffic-Grounded VLM Agents

**arXiv ID:** 2605.19219 | [PDF](https://arxiv.org/pdf/2605.19219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 72. Embodying Intelligence into Mechanical Metamaterials via Reservoir Computing

**arXiv ID:** 2605.19098 | [PDF](https://arxiv.org/pdf/2605.19098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 73. How Far Are We From True Auto-Research?

**arXiv ID:** 2605.19156 | [PDF](https://arxiv.org/pdf/2605.19156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 74. CounterFlow: A Two-Phase Inference-Time Sampling for Counterfactual Video Foley Generation

**arXiv ID:** 2605.18916 | [PDF](https://arxiv.org/pdf/2605.18916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 75. TwinRouterBench: Fast Static and Live Dynamic Evaluation for Realistic Agentic LLM Routing

**arXiv ID:** 2605.18859 | [PDF](https://arxiv.org/pdf/2605.18859v1)

**作者:** Pei Yang `[一作]` (Gradient), Tianyu Shi `[通讯]` (Gradient)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TwinRouterBench，一套针对多步LLM代理的步骤级路由基准，包含快照式的静态评估轨道和实时动态验证轨道，提供执行验证的步骤级最佳模型层级标签；

**💡 创新点**

创新点在于：①把前缀（对话历史、检索片段、工具输出等）作为路由决策输入；②通过贪心的逐步降级搜索与人工审核构造执行验证的步骤级标签；③设计两轨道的评估流程，使离线训练和在线验证互为补充；

**🔧 技术方法**

使用的技术包括：模型层级（tier）抽象与池化、贪心逐步锁定降级搜索、执行验证与混合模型回退、四桶计费公式与缓存计价、手工审核与随机抽样检查、Logistic回归训练路由器；

**📊 数据集**

使用的数据集为970条步骤级记录，来自520个实例，覆盖5个工作负载：SWE‑bench、PinchBench、BFCL、mtRAG、QMSum；动态轨道还评估了100个SWE‑bench验证案例；

**📈 对比分析**

评估方法：静态轨道使用RowPass/RowExact/TrajPass/CostSave四项指标的加权组合（Combined）；动态轨道按官方SWE‑bench通过率、实际API成本及固定未通过惩罚进行排名；实验结果显示：在静态轨道中SR‑KNN最高，训练好的UncommonRoute在动态轨道上实现与Unrouted Opus相同的75/100通过率，但API成本下降53%；

**⚠️ 局限性**

局限性包括：①静态语料仅970条，覆盖范围有限；②标签基于固定的模型池与价格快照，未来模型或定价变动需重新标注；③仅关注5个工作负载，可能导致对其他领域的泛化不足；

---

## 76. The Routing and Filtering Structure of Attention

**arXiv ID:** 2605.18826 | [PDF](https://arxiv.org/pdf/2605.18826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 77. Distribution Matching Distillation without Fake Score Network

**arXiv ID:** 2605.19256 | [PDF](https://arxiv.org/pdf/2605.19256v1)

**作者:** Youngjoong Kim `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**通讯引用:** 9530 | [OpenAlex ID](https://openalex.org/A5100611457)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无假分数网络的分布匹配蒸馏方法（FSF‑DMD），利用流映射生成器自身的端点伪速度作为假分数替代；

**💡 创新点**

创新点在于：①用生成器自身的伪速度代替额外的假分数网络；②在流映射框架下设计无网络反向分歧目标；③结合一致性蒸馏、后向模拟与自教师策略；

**🔧 技术方法**

使用流匹配、流映射生成器、分布匹配蒸馏（DMD）、一致性蒸馏、后向模拟、Euler展开、stop‑gradient、时间平移等技术；

**📊 数据集**

主要在ImageNet‑1K 256×256（使用VA‑VAE潜空间）进行实验；

**📈 对比分析**

与DMD2、Consistency Distillation、TwinFlow、RCGM等方法对比，在流映射初始化时Fidelity下降至3.85（相比6.06），训练更快、显存更低；在流匹配初始化和从零训练时也保持优于基线的性能；

**⚠️ 局限性**

局限性包括：仅在ImageNet 256实验验证；需要流映射结构；对更大规模或文本到图像等任务的适用性尚未评估；

---

## 78. Graph Neural Planning and Predictive Control for Multi-Robot Communication-Constrained Unlabeled Motion Planning

**arXiv ID:** 2605.19209 | [PDF](https://arxiv.org/pdf/2605.19209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 79. UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing

**arXiv ID:** 2605.18796 | [PDF](https://arxiv.org/pdf/2605.18796v1)

**作者:** Varun Kotte `[一作]` `[通讯]` (Independent Researcher), Varun Kotte (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于校准的不确定性指标的LLM级联路由方法UCCI，在小模型与大模型之间实现成本感知的推理。

**💡 创新点**

创新点在于将token级不确定性通过等距回归校准为误差概率，并证明在校准后阈值策略是成本最优的。

**🔧 技术方法**

使用token margin聚合、等距回归（Isotonic Regression）、基于成本约束的阈值选择以及端到端真实延迟评估。

**📊 数据集**

使用了75,000条生产级摄影搜索查询的NER数据集（6类实体）。

**📈 对比分析**

与always-large、entropy阈值、split-conformal和FrugalGPT阈值等基线在相同micro-F1=0.91的条件下比较，UCCI在成本上比大模型少31%，并在同一F1下比其他基线低约5–11%成本。

**⚠️ 局限性**

局限性包括仅在单一域（摄影NER）验证，假设大模型准确率不随路由改变而变化；在分布漂移或多模型异构场景需重新校准或调整成本模型。

---

## 80. Navigating the Emotion Tree: Hierarchical Hyperbolic RAG for Multimodal Emotion Recognition

**arXiv ID:** 2605.18884 | [PDF](https://arxiv.org/pdf/2605.18884v1)

**作者:** Zeheng Wang `[一作]` (Great Bay University), Zitong Yu `[通讯]` (Great Bay University)

**通讯引用:** 5282 | [OpenAlex ID](https://openalex.org/A5062522283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 HyperEmo‑RAG 框架，通过层次超曲面检索和结构化证据注入实现多模态情感识别；

**💡 创新点**

创新点在于：① 将情感层次树嵌入 Poincaré 球空间并采用层次 beam‑search 进行检索；② 构建 Deliberation Evidence Graph 并利用 Tree‑Aware Attention 与 EmotionGraphFormer 将检索证据结构化注入 LLM；

**🔧 技术方法**

使用技术包括超曲面（Poincaré 球）嵌入、层次超曲检索、图神经网络、Tree‑Aware Attention、EmotionGraphFormer、检索增强生成（RAG）与大型语言模型；

**📊 数据集**

实验数据集涵盖 MOSEI、MELD、SIMS‑V2 与 CHERMA 四个英中多模态情绪/情感基准；

**📈 对比分析**

在 Qwen‑1.8B、ChatGLM3‑6B 与 LLaMA2‑7B 三大 LLM 上与多种基线（如 TFN、MISA、UniMSE 等）及通用 RAG 方法（MuRAG、RA‑CM3、REVEAL）比较，所有指标均显著提升，尤其在二分类准确率、F1 与细粒度情绪识别上提升 3–5%；

**⚠️ 局限性**

局限性包括对 LLM 基座与数据匹配的依赖，检索与图注入带来的计算成本，以及对极细粒度情绪识别的进一步挑战与对超曲空间超参数的敏感性。

---

## 81. Lossless Anti-Distillation Sampling

**arXiv ID:** 2605.18829 | [PDF](https://arxiv.org/pdf/2605.18829v1)

**作者:** Zibo Diao `[一作]` (Tsinghua University), Di He `[通讯]` (Peking University)

**通讯引用:** 1903 | [OpenAlex ID](https://openalex.org/A5100739032)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Lossless Anti‑Distillation Sampling（LADS），通过在多账户间共享噪声种子来降低蒸馏数据多样性，既不影响正面用户体验，又能抑制模型蒸馏；

**💡 创新点**

创新点在于将随机性与查询语义和访问计数耦合，利用语义桶和私有种子生成器在保持单账户独立采样的同时，引入跨账户相关性，从而在理论上削弱蒸馏的统计效率；

**🔧 技术方法**

采用了伪随机数生成器、局部敏感哈希/语义聚类、私有种子生成器，以及Rademacher复杂度和均匀收敛理论分析等技术；

**📊 数据集**

使用ImageNet（1k类）做图像生成实验；使用MATH、GSM8K、HumanEval以及Code Alpaca等数据集做语言模型蒸馏实验；

**📈 对比分析**

与标准i.i.d.采样对比，图像生成中FID从8.5提升至29.8、验证损失显著升高；语言模型中蒸馏提升从+29.8点降至+11.2点，表明LADS显著抑制了蒸馏效果，同时保持正面用户质量；

**⚠️ 局限性**

局限在于需精细划分语义桶，桶数过多导致内存开销；攻击者可通过表面改写让相似查询落入不同桶；且仅对生成模型的随机性进行控制，无法阻止更高级的对抗蒸馏策略。

---

## 82. PhyWorld: Physics-Faithful World Model for Video Generation

**arXiv ID:** 2605.19242 | [PDF](https://arxiv.org/pdf/2605.19242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 83. Fine-Grained Benchmark Generation for Comprehensive Evaluation of Foundation Models

**arXiv ID:** 2605.18824 | [PDF](https://arxiv.org/pdf/2605.18824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 84. Fine-tuning language encoding models on slow fMRI improves prediction for fast ECoG

**arXiv ID:** 2605.19224 | [PDF](https://arxiv.org/pdf/2605.19224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 85. HAVEN: Hierarchically Aligned Multimodal Benchmark for Unified Video Understanding

**arXiv ID:** 2605.19223 | [PDF](https://arxiv.org/pdf/2605.19223v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 86. Learning When to Adapt

**arXiv ID:** 2605.19028 | [PDF](https://arxiv.org/pdf/2605.19028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 87. CLUE: Adaptively Prioritized Contextual Cues by Leveraging a Unified Semantic Map for Effective Zero-Shot Object-Goal Navigation

**arXiv ID:** 2605.19206 | [PDF](https://arxiv.org/pdf/2605.19206v1)

**作者:** Taeyun Kim `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6051 | [OpenAlex ID](https://openalex.org/A5059521863)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CLUE框架，用以在零样本物体目标导航任务中自适应地融合房间与物体的上下文信息。

**💡 创新点**

创新点在于利用离线大型语言模型提取目标与房间/物体关联的常识，并根据目标的房间关联熵动态加权全局与局部上下文，从而构建统一语义价值图，显著提升搜索效率。

**🔧 技术方法**

核心技术包括BLIP2与Gemini‑Pro 2.5进行语义匹配与熵计算、VLM/LLM离线知识检索、语义价值图融合、前沿探索策略和多视角验证。

**📊 数据集**

主要数据集为HM3D（约2000个回合，20个室内场景，6类目标）以及真实Clearpath Jackal平台的物理实验环境。

**📈 对比分析**

在HM3D基准上与多种SOTA方法（如ZSON、VLFM、ApexNav、SG‑Nav等）对比，CLUE在成功率SR达61.7%和路径效率SPL达34.3%上均取得最优或近乎最优成绩，且显著低于需在线LLM查询的方法。

**⚠️ 局限性**

局限性包括依赖预定义的房间类别和离线预计算的常识知识，可能限制在开放集或新环境中的泛化能力，并且对动态变化的场景适应性尚待提升。

---

## 88. MuMuTestUp: Mutation-based Multi-Agent Test Case Update

**arXiv ID:** 2605.19265 | [PDF](https://arxiv.org/pdf/2605.19265v1)

**作者:** Dawei Tian `[一作]` (Harbin Institute of Technology), Xiaohong Su `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1973 | [OpenAlex ID](https://openalex.org/A5100640057)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出MuMuTestUp，一个基于突变指导的多代理框架，用于自动更新过时的单元测试案例；

**💡 创新点**

创新点包括：①将突变测试作为断言强化的指导源，提升断言鲁棒性；②使用细粒度分支覆盖反馈，补足传统仅关注行覆盖的不足；③采用语义相似检索而非精确匹配，能处理LLM产生的幻觉符号；④多代理架构将错误分析、覆盖分析、突变分析与上下文检索拆分，协同优化测试质量；⑤构建首个PR级别测试更新数据集，模拟真实多提交场景；

**🔧 技术方法**

技术手段包括：大规模语言模型（GPT‑4.1、DeepSeek‑V3.2）进行代码生成；突变测试框架PIT；代码覆盖工具JaCoCo；基于ChromaDB的语义检索与向量嵌入；多代理协调与工具链集成；检索增量化迭代；

**📊 数据集**

使用的数据库是自构建的MuMuTestUp数据集，包含571个来自10个开源Java项目的PR级别测试更新实例；

**📈 对比分析**

与三种SOTA基线（CodeT5+、ReAccept、另一种LLM框架）在GPT‑4.1和DeepSeek下进行对比，评价指标为编译通过率、测试通过率、行覆盖、分支覆盖与突变得分；MuMuTestUp在所有指标上均优于最佳基线，GPT‑4.1下行覆盖提升5.33%、分支覆盖提升19.93%、突变得分提升16.66%；

**⚠️ 局限性**

现有方法的局限在于：①仅关注可执行性，忽视断言充分性；②使用粗粒度行覆盖信号，无法保证分支覆盖；③检索依赖精确匹配，无法处理LLM幻觉或私有符号。

---

## 89. Counterfactual Likelihood Tests for Indirect Influence in Private Reasoning Channels

**arXiv ID:** 2605.19092 | [PDF](https://arxiv.org/pdf/2605.19092v1)

**作者:** Alexander Boesgaard Lorup `[一作]` `[通讯]` (Openhagen), Alexander Boesgaard Lorup (Openhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了针对多角色推理模型的反事实影响测试，用以区分直接泄漏、公共传播与独立推理。

**💡 创新点**

创新点在于利用反事实前缀替换和负对数似然差值来量化私有推理对下游角色的影响，并通过图割控制定位具体传播路径。

**🔧 技术方法**

技术包括自回归语言模型的前缀干预、RoPE长度匹配、对数似然计算、图割注意力屏蔽以及多检查点、多种随机种子的大规模验证。

**📊 数据集**

数据集：在7B规模的自回归模型上采样了 500 条推理轨迹，跨三个相同训练系谱检查点、五个种子，共计13,734 条有效对比。

**📈 对比分析**

方法与传统 n-gram 重叠、canary 复制等文本检测对比，反事实影响率在未屏蔽模型中约54%，屏蔽后下降至13%，且方向性显著（A→B 25–39%，B→A 2–7%），表明能清晰区分直接与间接泄露。

**⚠️ 局限性**

局限性：仅在单一角色通道架构上验证，结果对其他模型结构或掩码拓扑不一定可迁移；图割控制只验证特定边，未揭示层级或头部的具体机制；B→A 对比多依赖合成桥接，真实对话中的反向影响可能不同。

---

## 90. AQuaUI: Visual Token Reduction for GUI Agents with Adaptive Quadtrees

**arXiv ID:** 2605.19260 | [PDF](https://arxiv.org/pdf/2605.19260v1)

**作者:** Yuankai Li `[一作]` (UC Davis), Muhao Chen `[通讯]` (UC Davis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的推理时视觉令牌压缩方法，通过自适应四叉树在GUI截图中只保留代表性令牌，减少视觉令牌数量。

**💡 创新点**

创新点在于利用GUI图像的空间结构非均匀性，使用四叉树自适应划分并在时间上实现条件四叉树重构，从而在不需要额外训练的前提下兼顾压缩率与准确性。

**🔧 技术方法**

核心技术包括：自适应四叉树构建、分块灰度/梯度分裂准则、代表性令牌选择、条件四叉树（静态、平移、替换三模式）以及位置编码保持。

**📊 数据集**

实验数据集涵盖多种基准：UI‑Vision、ScreenSpot‑Pro、ScreenSpot‑V2、OSWorld‑G、MMBench‑GUI、AndroidControl（离线导航）和AndroidWorld（在线多步导航）。

**📈 对比分析**

与ShowUI、FastV、随机压缩等基线比较，结果显示在Qwen3‑VL系列可压缩约30%视觉令牌，准确率下降不到1%，在大型模型上实现约13%速度提升、29%令牌减少；在较小模型上压缩带来额外延迟，但总体保持性能。导航任务中仍保持接近稠密模型的成功率，条件四叉树进一步提升了多步导航的鲁棒性。

**⚠️ 局限性**

局限性包括：在小型模型上压缩导致的额外推理延迟；对非GUI自然图像的适用性有限；压缩参数和分裂准则需手动调优；在极端高压缩率下可能丢失关键信息，导致准确率下降。

---

## 91. ZeroUnlearn: Few-Shot Knowledge Unlearning in Large Language Models

**arXiv ID:** 2605.18879 | [PDF](https://arxiv.org/pdf/2605.18879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. AI Technologies in Language Access: Attitudes Towards AI and the Human Value of Language Access Managers

**arXiv ID:** 2605.19234 | [PDF](https://arxiv.org/pdf/2605.19234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 93. Toward an AI-Powered Computational Testbed for Workforce Policy

**arXiv ID:** 2605.19064 | [PDF](https://arxiv.org/pdf/2605.19064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 94. Quantum Machine Learning for Cyber-Physical Anomaly Detection in Unmanned Aerial Vehicles: A Leakage-Free Evaluation with Proxy-Audited Feature Sets

**arXiv ID:** 2605.19233 | [PDF](https://arxiv.org/pdf/2605.19233v1)

**作者:** Carlos A. Durán Paredes `[一作]` (Corporation for Aerospace Initiatives, Research and Innovation), Camilo Segura Quintero `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对无人机多传感器异常检测进行泄漏无关的评估，并通过三模式特征审计检验模型对上下文代理的依赖。

**💡 创新点**

提出了B2时序分组协议、三模式特征审计和配对控制的混合分析框架，首次在无人机异常检测中验证量子增强混合分类器的增益。

**🔧 技术方法**

使用量子数据重上传（DRU）变分电路、XGBoost树集成、PCA、二次多项式、随机RBF核以及经典基线（Logistic、SVM‑RBF、MLP、RandomForest）。

**📊 数据集**

TLM:UAV多传感器数据集（12个子系统，72个数值特征，4.8k样本），重构自原始传感器日志以避免合并表的重复与时间泄漏。

**📈 对比分析**

在10个随机种子、K=10时序块、严格代理消除模式下，训练好的DRU+XGBoost混合模型在F1宏平均上比多数对照模型提高0.02–0.05，且在严格模式下误报率最低；但差异仍在种子内标准差范围内，未达到统计显著性。

**⚠️ 局限性**

局限包括：多时段样本仅支持二分类，缺乏独立时序实验；使用模拟器而非真实NISQ硬件；量子核SVM被禁用；并未对结果做正式显著性检验。

---

## 95. Neural Operators for Design-Space Surrogate Modeling of Tendon-Actuated Continuum Robots

**arXiv ID:** 2605.19104 | [PDF](https://arxiv.org/pdf/2605.19104v1)

**作者:** Branden Frieden `[一作]` (University of Utah), Varun Shankar `[通讯]` (University of Utah)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5055844388)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于神经算子（DeepONet和FNO）的连续体机器人（Tendon‑Driven Continuum Robot，TDCR）等效建模框架，实现了对不同设计参数下机器人静态形态的快速预测；

**💡 创新点**

创新点在于将TDCR的设计空间与平衡形态映射成无限维算子学习问题，构建了四种设计无关的神经算子模型，并通过几何正交化（Gram‑Schmidt）直接输出位姿；

**🔧 技术方法**

采用DeepONet（分支-主干架构）和FNO（傅里叶卷积层）两大类神经算子网络，并在训练中加入正则化、学习率周期调度与自定义损失；

**📊 数据集**

训练数据来自Cosserat杆理论模拟，共生成1万条设计-平衡对，覆盖4条肌腱、长度、张力、偏移、杨氏模量等六大参数范围；

**📈 对比分析**

在留出20%测试集、OOD测试以及训练时间与推理时间评估中，DeepONet模型在一般化误差上优于FNO（误差约10%~20%），但FNO在训练速度和内存占用上更具优势；

**⚠️ 局限性**

局限性包括在超出训练范围时“位姿”模型泛化能力下降，需进一步结合物理约束或SO(3)基向量编码以提升稳健性和可解释性。

---

## 96. Risk of Bad Tails: CVaR-Aware Pandora's Box and Prophet Inequality

**arXiv ID:** 2605.19181 | [PDF](https://arxiv.org/pdf/2605.19181v1)

**作者:** Jingwei Ji `[一作]` (Stanford University), Jingwei Ji `[通讯]` (Stanford University)

**通讯引用:** 2061 | [OpenAlex ID](https://openalex.org/A5019481782)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在Pandora盒子和先知不等式中引入Conditional Value-at-Risk（CVaR）目标后，分别求得了可行的索引解与阈值策略，阐明了风险敏感下的最优决策结构。

**💡 创新点**

创新点在于：①对Pandora盒子进行一维变分约简后保留了Weitzman索引解；②证明在无分布结构的情况下先知不等式无法获得任何正的常数近似，给出了紧确的实例依赖系数Bα(M)；③在连续重心IFRA分布下，阈值策略实现了仅依赖α的正常数近似，并给出了闭式下界ρ(α)。

**🔧 技术方法**

主要技术包括CVaR的变分表示、对Pandora盒子问题的变分约简与Weitzman索引的重构、阈值策略的量化分析、IFRA条件下的分位数包络与更大截断上界，以及构造极端两项实例来证明极限性。

**📊 数据集**

本文纯理论分析，无使用任何实验或实际数据集。

**📈 对比分析**

与传统期望最优策略相比，本文证明在没有分布假设时无法得到统一常数保证，但给出了最优的实例依赖系数；在IFRA分布下，阈值策略可获得ρ(α)≈0.5-1的常数近似，优于传统风险中性情况的1/2下限。

**⚠️ 局限性**

局限性包括：①在一般分布下无统一常数近似；②仅对连续且满足重心IFRA的分布保证常数近似；③示例表明α→0时常数可能趋近0，导致对低分位风险的保证不足。

---

## 97. Decentralized autonomous organization and blockchain-based incentivization framework for community-based facilities management

**arXiv ID:** 2605.18773 | [PDF](https://arxiv.org/pdf/2605.18773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 98. DecisionBench: A Benchmark for Emergent Delegation in Long-Horizon Agentic Workflows

**arXiv ID:** 2605.19099 | [PDF](https://arxiv.org/pdf/2605.19099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 99. Meta-Theorems for Cuttable Distributed Problems

**arXiv ID:** 2605.19157 | [PDF](https://arxiv.org/pdf/2605.19157v1)

**作者:** Marthe Bonamy `[一作]` (University of Bordeaux), Alexandra Wesolek `[通讯]` (University of Bordeaux)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种在欧拉基数为g的图上，利用局部性质与渐近维度实现的分布式算法，可在常数轮内得到约34+的MDS（最小支配集）近似解，并进一步给出了可推广至其他局部可优化问题的元定理；

**💡 创新点**

创新点在于将渐近维度与局部“好”图类结合，提出局部优良、可切割等新概念，构建了可将平面图算法扩展到任意基数图的元定理，从而大幅提升欧拉基数图的近似比与轮数；

**🔧 技术方法**

技术手段包括：使用控制函数的渐近维度颜色分解、局部子图切割与分析、统一近似（uniform approximation）框架、可切割（cuttable）问题判定，以及多版本的元定理实现；

**📊 数据集**

本文没有使用实验数据集，全部基于理论分析与图类的抽象属性（如平面图、K_p‑minor‑free图、嵌入表面图等）；

**📈 对比分析**

与之前的91+（或24g+O(1)）结果相比，本文的算法在常数轮内获得更优的34+近似比；此外还将该框架应用于k‑dominating、连接支配集等问题，得到相应的近似比；

**⚠️ 局限性**

局限性包括：近似比仍随基数g而变化，无法在所有基数图上实现真正的常数近似；算法轮数与基数g相关；对某些非局部问题仍只能得到O(√g)的近似。

---

## 100. Spatially Accelerated Winding Numbers for Curved Geometry

**arXiv ID:** 2605.19200 | [PDF](https://arxiv.org/pdf/2605.19200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 101. pyModRev: a Python Tool for Model Revision of Boolean Networks

**arXiv ID:** 2605.19046 | [PDF](https://arxiv.org/pdf/2605.19046v1)

**作者:** Pedro T. Monteiro `[一作]` (Universidade de Lisboa), Filipe Gouveia `[通讯]` (Universidade de Lisboa)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并实现了ModRev工具的改进版pymodrev，用于验证并最小化修复布尔调控网络模型与实验观测的一致性；

**💡 创新点**

创新点包括：重写为Python实现、支持多种模型与观测格式、可并行处理多种更新方案与多种观测数据、支持“非稳态”约束、三阶段流程（检查、搜索、修复）、输出JSON等多种格式，显著提升易用性和集成度；

**🔧 技术方法**

核心技术是ASP（clingo）求解一致性、Quine‑McCluskey算法将布尔函数转为CDNF、pyfunctionhood库计算函数邻域以实现最小功能变更，配合搜索算法寻找最小修复组合；

**📊 数据集**

使用Gouveia等人构造的损坏模型数据集（5个布尔模型、4类破坏方式、24种组合、每组合100实例），以及Héraut等人的早期造血干细胞模型做案例研究；

**📈 对比分析**

与原ModRev进行对比，pymodrev在3600秒内完成大多数模型的修复，修复结果与ModRev一致，验证了方法的正确性与有效性；

**⚠️ 局限性**

局限性包括：时间序列观测需预先给定步数（不支持无步长的可达性约束）；仅支持单一更新方案（除异步/同步外）且尚未实现最宽松更新方案；GINsim参数化函数暂不支持；

---

## 102. XFlowMap: Cross-Scale Generalization and Mapping of Massive Origin-Destination Data

**arXiv ID:** 2605.18777 | [PDF](https://arxiv.org/pdf/2605.18777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 103. HypergraphFormer: Learning Hypergraphs from LLMs for Editable Floor Plan Generation

**arXiv ID:** 2605.18932 | [PDF](https://arxiv.org/pdf/2605.18932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 104. PROWL: Prioritized Regret-Driven Optimization for World Model Learning

**arXiv ID:** 2605.18803 | [PDF](https://arxiv.org/pdf/2605.18803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 105. Riemannian Networks over Full-Rank Correlation Matrices

**arXiv ID:** 2605.19073 | [PDF](https://arxiv.org/pdf/2605.19073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 106. Euclidean Embedding of Data Using Local Distances

**arXiv ID:** 2605.19243 | [PDF](https://arxiv.org/pdf/2605.19243v1)

**作者:** Dimitris Arabadjis `[一作]` `[通讯]` (University of the Aegean), Dimitris Arabadjis (University of the Aegean)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种仅基于邻域距离图、无需特征向量的欧氏嵌入方法，利用变分优化框架推导连续域最优嵌入方程，并给出迭代求解的稀疏线性方案。

**💡 创新点**

创新点在于：①在流形上对最优欧氏嵌入的功能方程进行完整推导；②提出与距离图无关的表示方法，仅利用局部图操作；③证明了通过迭代的稀疏线性求解能收敛到全局最优，并保持局部几何一致性。

**🔧 技术方法**

技术手段包括：变分法与微分形式理论、光滑流形上的散度、内积与拉普拉斯算子、谱分解（SVD）、稀疏线性求解（预条件共轭梯度）以及基于图的内积与散度实现。

**📊 数据集**

使用的实验数据集有：合成数据（Klein瓶、扁平环面、5维难题、瑞士卷）以及真实数据（MNIST、FMNIST、肺癌RNA‑seq）。

**📈 对比分析**

与LLE、LTSA、Laplacian Eigenmaps、Hessian Eigenmaps、t‑SNE、UMAP、IsoMap及Diffusion Maps等方法比较；在局部几何一致性指标上优于大多数局部方法，在全局一致性上逼近IsoMap；在聚类性能上与t‑SNE/UMAP相当或略低。

**⚠️ 局限性**

局限性包括：仅利用邻域信息，无法处理远程关系或全局形状约束；对噪声和异常值敏感；需要多次迭代、时间复杂度高于纯稀疏方法；非参数性质导致增量/外推困难；实现涉及多次拉普拉斯、SVD等计算。

---

## 107. TabQL: In-Context Q-Learning with Tabular Foundation Models

**arXiv ID:** 2605.18979 | [PDF](https://arxiv.org/pdf/2605.18979v1)

**作者:** Qisai Liu `[一作]` (Iowa State University), Soumik Sarkar `[通讯]` (Iowa State University)

**通讯引用:** 10444 | [OpenAlex ID](https://openalex.org/A5081037761)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为TabQL的强化学习框架，将传统DQN中的Q网络替换为可进行上下文学习的表格基础模型，通过经验上下文进行贝尔曼推理来快速学习Q值；

**💡 创新点**

创新点在于将Q学习视为条件推理任务，利用预训练的表格基础模型实现无梯度的贝尔曼更新；通过短暂的DQN热身提供信息丰富的上下文，再利用表格模型进行即时推理；并提供理论收敛与样本复杂度分析；

**🔧 技术方法**

使用了表格基础模型（如TabPFN、TabDPT）进行上下文推理，结合传统DQN的经验回放与梯度下降；采用经验上下文窗口和质量门控；

**📊 数据集**

在离散网格环境（Taxi-v3、CliffWalking-v1、FrozenLake-v1）以及连续观测环境CartPole-v1上进行实验；

**📈 对比分析**

与传统Tabular Q、DQN、Double DQN、Dueling DQN及Fitted Q-Iteration等基线对比，TabQL在大多数环境中实现更快的收敛，样本效率提升1-2倍，最终性能接近或超过基线；

**⚠️ 局限性**

局限包括：需要手动设置热身长度T0；表格基础模型对高维或图像输入的适用性有限；当上下文稀疏时，方法退化回DQN性能；缺乏自适应切换机制及更复杂状态空间的实验验证。

---

## 108. Efficient Conditioning Why Pseudo Observation Batch Bayesian Optimization Works When It Does not

**arXiv ID:** 2605.18819 | [PDF](https://arxiv.org/pdf/2605.18819v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 109. AgentNLQ: A General-Purpose Agent for Natural Language to SQL

**arXiv ID:** 2605.19010 | [PDF](https://arxiv.org/pdf/2605.19010v1)

**作者:** Olena Bogdanov `[一作]` (JPMorganChase), Anup Shirgaonkar `[通讯]` (JPMorganChase)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于多智能体的NL2SQL系统，通过对数据库模式进行语义丰富、向量检索和自我反思来生成SQL查询；

**💡 创新点**

创新点在于设计了专门针对NL2SQL的双账本（Task/Progress）调度器、结构化上下文压缩、功能性守卫和多模型协同（GPT-4o+Claude Opus 4.1）来显著提升准确率；

**🔧 技术方法**

采用了LLM（OpenAI GPT‑4o、Claude Opus 4.1）、嵌入模型（text‑embedding‑3‑large‑1）、向量检索（FAISS）、SQLGlot语法校验、自动化元数据生成与Schema Linker、以及自定义 Agentic Tools 与 MCP 服务器；

**📊 数据集**

在BIRD（11域）和BIRD Financial子集的SQLBench标注集上进行评估；

**📈 对比分析**

通过与基线单一Agent、Autogen Orchestrator、以及无向量检索/无Schema Enrichment等对比实验，最终在BIRD 11域上实现78.1%的语义准确率，明显优于人类专家（93%）的基线，且在金融域达到79.2%；

**⚠️ 局限性**

受限于LLM对复杂SQL的生成准确性、SQL方言兼容性、极大数据库的检索与推理负载，以及对业务规则的依赖，需要人工审核以确保关键应用的安全与正确性。

---

## 110. Learn-by-Wire Training Control Governance: Bounded Autonomous Training Under Stress for Stability and Efficiency

**arXiv ID:** 2605.19008 | [PDF](https://arxiv.org/pdf/2605.19008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 111. ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models

**arXiv ID:** 2605.19095 | [PDF](https://arxiv.org/pdf/2605.19095v1)

**作者:** Aaron Defazio `[一作]` `[通讯]` (FAIR at Meta Super-Intelligence Labs), Aaron Defazio (FAIR at Meta Super-Intelligence Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种完全不使用学习率调度、只用平均化的优化方法 Schedule-Free Learning，并将其扩展为 ScheduleFree+，适用于大型语言模型训练；

**💡 创新点**

主要创新点包括：1）在平均化过程中引入内循环动量以提升大批量训练稳定性；2）利用 Polyak 步长近似实现学习率自适应；3）使用 β 参数的时间衰减与 r=1 的加权平均进一步提升长期训练效果；4）通过逆梯度范数加权和全局 Polyak 估计消除权重衰减与学习率动态的负反馈；

**🔧 技术方法**

技术手段包括：内循环动量、Polyak 步长、β 参数的时间插值、r=1 加权平均、逆梯度 L1 范数加权、全局 EMA、温度调度（warm‑up）、并行权重平均（模型合并/熔炼）。

**📊 数据集**

实验主要使用 Llama‑3‑120M、250M、500M、1B、2B 参数模型，训练数据集为 FineWeb‑EDU，训练总量从 1000 tokens/参数到 20 tokens/参数不等。

**📈 对比分析**

与传统学习率调度（Linear Decay、Cosine、WSD）和 AdamW 基线相比，ScheduleFree+ 在长周期训练（1000 tokens/参数）下的最终损失低约 31%，在中短周期（100 tokens/参数）下仍能超过 WSD，且在任何训练时刻都能给出可用模型。

**⚠️ 局限性**

局限性包括：1）对极短训练周期（<100 tokens/参数）时的收敛速度不如调度式方法；2）对大批量训练仍需引入内循环动量，增加实现复杂度；3）需要精细调节 β 退火与 r 参数，缺乏通用自动化策略；4）在非常大模型或极大批量时，梯度范数漂移仍可能导致训练不稳定。

---

## 112. FormalASR: End-to-End Spoken Chinese to Formal Text

**arXiv ID:** 2605.19266 | [PDF](https://arxiv.org/pdf/2605.19266v1)

**作者:** Wanyi Ning `[一作]` (Yijiahe Technology Co., Ltd.), Yufei Zhang `[通讯]` (Yijiahe Technology Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个端到端的中文语音转写模型FormalASR，可直接生成正式书面文本。

**💡 创新点**

利用LLM重写语音文本构造大规模spoken‑to‑formal数据，并在单模型中同时学习识别与文本正规化，无需后置LLM。

**🔧 技术方法**

对Qwen3‑ASR进行全参数监督微调（SFT），结合Whisper‑style声学编码器与Qwen解码器，以及GGUF量化实现轻量部署。

**📊 数据集**

构建并公开了WenetSpeech‑Formal（969K训练）和Speechio‑Formal（43K跨域测试）两个大规模中文spoken‑to‑formal数据集。

**📈 对比分析**

与未微调的Qwen3‑ASR和Whisper large‑v3对比，在WenetSpeech‑Formal和Speechio‑Formal上CER下降至0.16/0.15（相对降低34.7%/37.4%），同时ROUGE‑L与BERTScore均提升。

**⚠️ 局限性**

模型规模仍受限于算力，较小的0.6B模型在正式化上不如1.7B，且对极长语音或高噪声环境的鲁棒性尚待验证。

---

## 113. Reducing Waiting Time for Medical Tourists Through Hybrid Agent-Based and Discrete-Event Simulation: A Hospital Case Study

**arXiv ID:** 2605.19139 | [PDF](https://arxiv.org/pdf/2605.19139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

---

## 114. Restructure This: Using AI to Restructure Onboarding Documents to Reduce Cognitive Overload

**arXiv ID:** 2605.19174 | [PDF](https://arxiv.org/pdf/2605.19174v1)

**作者:** Zixuan Feng `[一作]` (Oregon State University), Anita Sarma `[通讯]` (Oregon State University)

**通讯引用:** 4219 | [OpenAlex ID](https://openalex.org/A5024821289)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VisDoc，一种基于 Cognitive Theory of Multimedia Learning 的工作流感知文档重组工具，自动将开源项目的 onboarding 文档切分为任务单元、推断工作流、去冗余并生成多模态解释；

**💡 创新点**

创新点在于：①将 CTML 的学习设计原则系统化应用于 OSS 文档重组；②结合 GenAI（GPT‑4o、Claude、ElevenLabs）实现端到端的自动化重组与多模态生成；③通过人机协作的“人机共创”流程确保文档准确性；

**🔧 技术方法**

使用 GPT‑4o 进行段落切分、主题推断、冗余检测与任务顺序推断；Claude Computer Use 生成脚本并录制视频；ElevenLabs 合成旁白；前端采用 React 构建交互式任务树；后端 Python+LangChain + RAG 框架；

**📊 数据集**

以 HuggingFace Transformers（transformers）仓库的 CONTRIBUTING.md 及其相关文件为测试数据；专家评估使用 Kubernetes 项目文档；

**📈 对比分析**

对比方法：专家评估（N=4）从完整性、准确性、粒度、可采纳性四维度评估；新手实验（N=14）随机分组，对比 VisDoc 与原始文档+可选 ChatGPT；结果显示 VisDoc 任务成功率 20/21（95%）显著高于对照组 13/21（62%）；NASA‑TLX 认知负荷各维度显著降低（如精神负荷 75→30，时间压力 60→20，挫折 60→10），SUS 可用性得分显著提升（p=0.005）。

**⚠️ 局限性**

局限性：样本量小、主要为大学生且性别偏少；仅评估单一 OSS 项目（transformers），无法覆盖不同规模/领域的项目；依赖原始文档质量，若文档缺失/错误会被放大；LLM 仍有幻觉风险，需要人工审核；成本高、资源占用大；维护频率高，需同步更新；实验仅为短期单次任务，缺乏长期跟踪研究。

---

## 115. Balancing Teacher and Student Agency: Co-Orchestration Tool Design Supporting Real-Time Dynamic Pairing

**arXiv ID:** 2605.18761 | [PDF](https://arxiv.org/pdf/2605.18761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 116. Can Large Language Models Revolutionize Survey Research? Experiments with Disaster Preparedness Responses

**arXiv ID:** 2605.19229 | [PDF](https://arxiv.org/pdf/2605.19229v1)

**作者:** Yan Wang `[一作]` (University of Florida), Christopher McCarty `[通讯]` (University of Florida)

**通讯引用:** 6448 | [OpenAlex ID](https://openalex.org/A5080323581)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在灾害准备调查中，构建并评估了五阶段大语言模型（LLM）集成框架，从问卷设计到数据分析全流程。

**💡 创新点**

将保护动机理论（PMT）嵌入知识图检索，提出 Anchored Marginal Theory‑Informed LLM (A‑TLM)，同时提出分阶段与单次整合检索对比，展示理论导向检索结构优于无结构检索。

**🔧 技术方法**

使用检索增强生成（RAG）、图检索、分阶段推理、零/少量样本提示、随机森林、IPW/MI、MICE+PMM 等多种方法；核心技术是基于 PMT 的共现知识图与检索增强 LLM。

**📊 数据集**

利用 2024 年佛罗里达州飓风米尔顿（Hurricane Milton）后调查样本 946 例（训练 757，验证 189）以及 ACS 人口基准。

**📈 对比分析**

在四种缺失机制下对齐缺失数据，A‑TLM 在最严重的 block‑wise MNAR（S4）下 RMSE 1.439，优于传统 3 种基线（IPW/MI 1.828，MICE+PMM 1.412，missForest 1.496），并显著降低对 compound‑vulnerable 子群的偏差；单次整合检索的 Marginal‑TLM 与 A‑TLM 误差仅高约 1% 级别。

**⚠️ 局限性**

仅在单一灾害、单一语言环境下验证；缺失机制仿真基于 189 样本，子群样本量有限；模型仍易产生“hallucination”及对极端群体预测误差；实验多为演示，缺乏大规模实证与跨地区验证。

---

## 117. MMoA: An AI-Agent framework with recurrence for Memoried Mixure-of-Agent

**arXiv ID:** 2605.19194 | [PDF](https://arxiv.org/pdf/2605.19194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 118. FedMental: Evaluating Federated Learning for Mental Health Detection from Social Media Data

**arXiv ID:** 2605.18936 | [PDF](https://arxiv.org/pdf/2605.18936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 119. Guiding Neuro-Symbolic Scenario Generation with Spatio-Temporal Logic

**arXiv ID:** 2605.19038 | [PDF](https://arxiv.org/pdf/2605.19038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 120. Code-Guided Reasoning for Small Language Models: Evaluating Executable MCQA Scaffolds

**arXiv ID:** 2605.18827 | [PDF](https://arxiv.org/pdf/2605.18827v1)

**作者:** Prateek Biswas `[一作]` (IBM), Amit Sheth `[通讯]` (University of South Carolina)

**通讯引用:** 36165 | [OpenAlex ID](https://openalex.org/A5028772801)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了Code-Guided Reasoning (CGR)评估协议，记录直接答案、辅助答案和生成器答案的三通道，并提供完整可审计的执行记录。

**💡 创新点**

提出可执行Python scaffold作为可审计的“工具”框架，将小语言模型嵌入可执行环境，以观察其在多次调用和推理控制下的表现，并与传统直接回答做对比。

**🔧 技术方法**

采用生成式Python scaffold、解决器调用接口、答案提取器、分区统计与配对bootstrap估计、以及对结果的代码与元数据审计。

**📊 数据集**

使用了9个本地标准化的多选问答数据集：AIME、MedQA、PhysicsQA、MMLU-Pro、SuperGPQA、Time-MQA、CorrectBench、OpenBookQA和FailureSensorIQ。

**📈 对比分析**

通过宏观平均和微观准确率比较直接与辅助两种路径，在非零基线分区中辅助准确率从38.11%提升至66.21%（+28.10pp），在更严格基线门限下提升14.11pp；但在Time-MQA等数据集出现回归。

**⚠️ 局限性**

主要局限包括辅助路径使用更多模型调用与预算、答案提取脆弱、生成代码可能硬编码答案、缺乏等价预算对照与链式思考等控制实验，且未能证明因果关系或操作安全性。

---

## 121. What Makes Synthetic Data Effective in Image Segmentation

**arXiv ID:** 2605.19289 | [PDF](https://arxiv.org/pdf/2605.19289v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 122. Sequences with thirteen-valued cross correlations

**arXiv ID:** 2605.19268 | [PDF](https://arxiv.org/pdf/2605.19268v1)

**作者:** Yuehui Cui `[一作]` (Central China Normal University), Jinquan Luo `[通讯]` (Central China Normal University)

**通讯引用:** 1510 | [OpenAlex ID](https://openalex.org/A5005937717)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在 p ≡ 1 (mod 3) 条件下，m 序列与其 Niho 型去码序列（d = (p^n-1)/3 + p^i）的交叉相关分布，并给出了完整的 13 取值分布。

**💡 创新点**

首次证明该交叉相关取 13 个不同值，并在此基础上推导出对应循环码 C_{1,d} 的权分布，从而实现理论上完全的分布描述。

**🔧 技术方法**

采用指数和、Gaussian 周期、Gauss 和 Eisenstein 和等数论工具，并利用 3 级字符的性质进行精确计算。

**📊 数据集**

论文为纯理论推导，没有使用具体实验数据集。

**📈 对比分析**

与之前已知的 p ≡ 2 (mod 3) 情况比较，证明了更一般的情形下仍可得到完全分布，理论上实现了完整性并为后续应用提供了更广的基础。

**⚠️ 局限性**

局限在于仅适用于 1/3 p^{-i}(p^n-1) ≠ 2 (mod 3) 的条件，且公式较为复杂，实际工程应用需要进一步简化和验证。

---

## 123. Robust Basis Spline Decoupling for the Compression of Transformer Models

**arXiv ID:** 2605.18794 | [PDF](https://arxiv.org/pdf/2605.18794v1)

**作者:** Joppe De Jonghe `[一作]` (KU Leuven), Mariya Ishteva `[通讯]` (KU Leuven)

**通讯引用:** 620 | [OpenAlex ID](https://openalex.org/A5086848112)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出基于B‑样条的解耦（decoupling）框架，用于将多元向量函数拆解为线性映射+一层一元非线性函数+线性映射的结构，并将其应用于Transformer模型的全连接（FCNN）模块压缩。

**💡 创新点**

创新点在于：①把B‑样条作为通用基底，统一并泛化了多项式和Piece‑wise ReLU解耦方法；②提出约束的矩阵–张量分解（CMTF）与B‑样条投影相结合的鲁棒交替最小二乘算法R‑CMTF‑BSD，加入归一化与Tikhonov正则化，显著提升数值稳定性；③通过“back‑to‑front”压缩策略在Transformer中实现更高压缩率与更低准确率下降。

**🔧 技术方法**

核心技术包括：张量分解（CPD）、约束矩阵–张量分解（CMTF）、B‑样条基底（含节点选择与量化）、鲁棒交替最小二乘算法、Tikhonov正则化、Transformer结构（ViT、Swin）以及基于Jacobian与函数值的双阶信息结合。

**📊 数据集**

使用的主要数据集有MNIST、SVHN、CIFAR‑10、USPS；在MNIST上做单块压缩实验，在SVHN/CIFAR‑10上对完整ViT与Swin进行压缩。

**📈 对比分析**

与低秩SVD、DRONE等方法比较，B‑样条解耦在同等压缩率下保持或提升Top‑1准确率：在DeiT/USPS上21.5%参数压缩时保持94.75%准确率；在55%压缩时保持94.41%准确率，优于SVD（19.6%）与DRONE（52.4%）。

**⚠️ 局限性**

局限性包括：①需要预先采样并计算Jacobian，计算成本高；②对B‑样条节点的量化选择及正则化参数敏感；③目前仅针对Transformer的FCNN压缩，未直接验证对自注意力或循环网络的效果；④在极端压缩率下仍可能出现准确率显著下降。

---

## 124. Going PLACES: Participatory Localized Red Teaming for Text-to-Image Safety in the Global South

**arXiv ID:** 2605.19190 | [PDF](https://arxiv.org/pdf/2605.19190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 125. DEFLECT: Delay-Robust Execution via Flow-matching Likelihood-Estimated Counterfactual Tuning for VLA Policies

**arXiv ID:** 2605.19294 | [PDF](https://arxiv.org/pdf/2605.19294v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 126. Mapping Uncharted Symmetries: Machine Discovery in Combinatorics

**arXiv ID:** 2605.19063 | [PDF](https://arxiv.org/pdf/2605.19063v1)

**作者:** Eugenio Cainelli `[一作]` (University of Bologna), Giovanni Paolini `[通讯]` (University of Bologna)

**通讯引用:** 971 | [OpenAlex ID](https://openalex.org/A5109829154)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过 SLURP 框架和机器学习方法发现 q,t‑Narayana 多项式的新组合解释并构造了交换双射，解决了 k=3 的开放问题。

**💡 创新点**

将严格比例约束下的函数发现问题形式化为 SLURP，并首次利用 ML 自动生成新的统计量与交换双射，解决 k=3 的未解题目。

**🔧 技术方法**

使用自监督训练的 Transformer、深度交叉熵方法（CEM）结合符号搜索、强化学习与程序合成，并通过 Lean 4 进行正式验证。

**📊 数据集**

使用基于非交叉分区的 skip‑pairing 数据集及其细化版本作为训练和评估数据。

**📈 对比分析**

与传统 GA 基线比较，CEM 能快速收敛到零距离的简洁公式，性能优于 GA，生成的统计量与人工统计一致且可验证。

**⚠️ 局限性**

仅在 k=3 成功，k≥4 的实例仍未收敛；方法对简单度的评估仍需人工干预，且对更大组合对象的可扩展性有限。

---

## 127. A Geometric Algebra-Informed 3D Gaussian Splatting Framework for Wireless Scene Representation

**arXiv ID:** 2605.19065 | [PDF](https://arxiv.org/pdf/2605.19065v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 128. Stability and Discretization Error of State Space Model Neural Operators

**arXiv ID:** 2605.18905 | [PDF](https://arxiv.org/pdf/2605.18905v1)

**作者:** Abderrahim Bendahi `[一作]`, Madiha Nadri `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文建立了从连续到离散的理论框架，对卷积型神经算子（尤其是SS‑NO）在离散化实现中的误差与稳定性进行了严格分析，给出了误差上界与 Lipschitz 稳定性定理；

**💡 创新点**

创新点在于首次为 SS‑NO 推导离散化误差定理，并将输入-状态稳定性（ISS）引入离散化分析；此外，论文将非光滑激活（如 ReLU）与 Sobolev 正则性相结合，提供了更宽泛的正则性假设；

**🔧 技术方法**

使用了傅里叶分析、Sobolev 空间正则性、卷积核解析、输入-状态稳定性理论以及连续-离散误差传播推导；

**📊 数据集**

实验主要使用 1D/2D 高斯随机场（GRF）输入，并在补充材料中验证了 Burgers 方程基准；

**📈 对比分析**

通过将不同分辨率的离散算子输出与高分辨率“真实”离散算子进行对比，计算相对 L2 误差随分辨率下降的趋势；实验表明误差随分辨率按理论 β 指数下降，光滑激活（GELU）比非光滑激活（ReLU）误差更小；

**⚠️ 局限性**

局限性包括：仅在均匀网格和特定几何下分析；主要关注 SS‑NO 和 FNO，未覆盖更复杂几何或其他算子；对深度与训练过程的交互影响未给出完整理论；

---

## 129. Efficient coding along the visual hierarchy

**arXiv ID:** 2605.19155 | [PDF](https://arxiv.org/pdf/2605.19155v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 130. Mask-to-Correct$^+$: Leveraging Retriever Diversity for Masking-guided Faithful Fact Correction

**arXiv ID:** 2605.18776 | [PDF](https://arxiv.org/pdf/2605.18776v1)

**作者:** Payel Santra `[一作]` (Indian Association for the Cultivation of Science), Partha Basuchowdhuri `[通讯]` (Indian Association for the Cultivation of Science)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5060417636)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种不需要训练或人工标注的事实纠错框架Mask-to-Correct (M_2C) 及其集成版 M_2C^+，通过多阶段遮掩、检索增强生成 (RAG) 和语义-事实评分来实现高质量的事实纠正。

**💡 创新点**

创新点包括：① 采用基于MMR的多样化遮掩策略来精准定位潜在错误片段；② 通过检索增强生成将外部证据嵌入纠错过程，保证事实性；③ 用多检索器的投票集成（M_2C^+）降低检索偏差，提升鲁棒性。

**🔧 技术方法**

技术手段主要包括：多样化遮掩 (Diversity Masking)、检索增强生成 (RAG)、DocNLI 与 ROUGE-L 的融合评分、以及多检索器的多数投票集成。

**📊 数据集**

使用了两个公开数据集：FEVER（通用领域）和 SciFact（科学领域），并在检索时分别使用 Wikipedia 2018 dump 与 S2ORC 语料库。

**📈 对比分析**

与多种无参数与参数化基线（包括直接生成、RAG、Zerofec、T5-Distant、Compedit 等）对比，实验表明 M_2C 与 M_2C^+ 在 SARI 和 BART 分数上均优于所有基线，最大可提升约 14%。

**⚠️ 局限性**

局限性：模型需要加载大规模 LLM（如 Llama-2 70B、Qwen-2.5 32B），占用显存高且不易在资源受限环境下运行；且未能利用更强的商用 LLM（如 GPT‑4、Claude）。

---

## 131. Discoverable Agent Knowledge -- A Formal Framework for Agentic KG Affordances (Extended Version)

**arXiv ID:** 2605.19186 | [PDF](https://arxiv.org/pdf/2605.19186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 132. Interoceptive Divergence in Aesthetic Evaluation and Implications for Human-AI Alignment

**arXiv ID:** 2605.18759 | [PDF](https://arxiv.org/pdf/2605.18759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 133. INAR-VL: Input-Aware Routing for Edge-Cloud Vision-Language Inference

**arXiv ID:** 2605.18853 | [PDF](https://arxiv.org/pdf/2605.18853v1)

**作者:** Ahmed Šabanović `[一作]` (TU Wien), Ivona Brandić `[通讯]` (TU Wien)

**通讯引用:** 10845 | [OpenAlex ID](https://openalex.org/A5009158531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 INAR‑VL 的轻量级边缘‑云路由系统，用于视觉‑语言模型（VLM）在边缘设备与云端之间的自适应推理调度。

**💡 创新点**

创新点在于使用轻量级多模态特征（图像质量与文本复杂度）进行预推理路由，并联合优化模型、分辨率与执行位置的 Pareto 前沿，从而在保持高准确率的同时降低延迟和能耗。

**🔧 技术方法**

技术包括多模型池管理、请求特征提取（blur、曝光、JPEG 痕迹、细节、文本长度、实体密度等）、基于 Pareto 的路由决策、INT8 量化与 FP16 执行、离线校准的质量与成本预测。

**📊 数据集**

使用三大视觉问答基准：VQAv2、TextVQA 和 GQA，各抽取 2000 条样本进行评估。

**📈 对比分析**

与静态边缘/云部署、仅文本或仅图像路由以及 MoA‑Off 等基线对比，INAR‑VL 在三项数据集上的平均准确率达 72.1%，比边缘单独部署高 5.6 百分点，延迟降低 24%（约 1826 ms），能耗降低 26%（约 19.2 J），同时仍有 36% 的请求在边缘执行。

**⚠️ 局限性**

局限性在于仍未完全匹配 oracle（最佳‑四模型）性能，且依赖离线校准与固定的硬件/模型组合；对极端网络抖动、未知输入分布的鲁棒性需进一步验证。

---

## 134. Compositional Literary Primitives in Instruction-Tuned LLMs: Cross-Architectural SAE Features for Self, Style, and Affect

**arXiv ID:** 2605.18808 | [PDF](https://arxiv.org/pdf/2605.18808v1)

**作者:** Joao Paulo Cavalcante Presa `[一作]` (Federal University of Goias), Savio Salvarino Teles de Oliveira `[通讯]` (Federal University of Goias)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究利用稀疏自编码器（SAE）在 Llama 3.1 8B-Instruct 与 Gemma 2 9B‑IT 两个指令微调 LLM 的中层残差流中，系统地挖掘并验证了四类可单独驱动的文学创作原语：命名门（naming‑gates）、自我特征集（self‑cluster）、风格注册调制器（如 Show‑Don’t‑Tell、Defamiliarization）以及仅通过多特征组合产生的情绪。通过三阶段验证管道（logit‑lens → LLM‑rate → 5‑LLM 判定）以及一系列反模式（anti‑patterns），构建了覆盖 Cowen‑Keltner 27 类情绪的命名门目录，并在不同架构间比较其覆盖率、输出风格及交叉语言迁移性能。

**💡 创新点**

创新点包括：
1) 首次证明文学创作原则能以单向可调的 SAE 特征形式嵌入指令微调 LLM；
2) 发现多特征组合能生成情绪（如 Joy、Adoration），揭示情绪与自我/调制器特征的交互；
3) 提出并验证三阶段验证管道，显著降低单一阶段误报率；
4) 系统记录并公开七个反模式，为后续 SAE 研究提供方法论借鉴；
5) 对两种不同 SAE 架构（TopK vs JumpReLU）进行跨架构、跨语言的比较，揭示输出风格与门效应的差异。

**🔧 技术方法**

主要技术手段有：
- 稀疏自编码器（TopK、JumpReLU）提取中层残差特征；
- Logit‑lens 通过 decoder‑unembedding 方向投影，快速定位候选特征；
- LLM‑rate 评估候选特征词表纯度，过滤语义漂移；
- 5‑LLM 判定面板（OpenRouter LLM）执行强制选择或是/否情绪分类，验证因果驱动；
- 通过向残差流添加 steering 向量（α/decoder_norm 乘子）实现特征激活；
- 统计分析（Fleiss κ、随机判定阈值）评估结果显著性；
- 记录并修正七类反模式，保证实验可复现。

**📊 数据集**

数据集与素材：
- 60 条 Show‑Don’t‑Tell 与 12 类情绪场景（5 条/情绪）手工对照对；
- 24 对文学技术（Deep‑POV、Defamiliarization）最小对；
- 75 条 Cowen‑Keltner 27 类扩展场景（5 条/情绪）；
- 12 种语言（英语、法语、西班牙语、德语）对应情绪标注；
- Cowen‑Keltner 27 类情绪词表；
- 5‑LLM 判定面板使用 OpenRouter LLM，未使用大规模公开语料，仅基于手工编写提示。

**📈 对比分析**

比较方法与性能：
- 以 5‑LLM 判定面板的 5‑分投票为基础，对每个特征进行强制选择或是/否判定；
- 统计 hit‑rate 与 specificity，计算 95% 置信区间；
- Llama 在 Cowen‑Keltner 27 类覆盖率 27/27；Gemma 23/23；
- 对比 strict 与 soft 判定标准，发现 Llama 在 strict 模式下一致通过（κ≈0.53），Gemma κ≈0.28；
- 交叉语言评估显示 Llama 生成几乎完全为目标语言，Gemma 则倾向于英语混合；
- 随机判定阈值下，单元格通过概率 ≈10⁻³，实际通过数远大于期望，表明结果非偶然。

**⚠️ 局限性**

局限性：
- 仅针对两种模型（Llama 3.1 8B 与 Gemma 2 9B‑IT），无法推广到更大或不同架构的 LLM；
- 只在中层残差流中搜索特征，可能忽视更深层或更浅层的结构；
- 语言覆盖有限（英语、法语、西班牙语、德语），对低资源或非拉丁文字缺乏验证；
- 评估完全依赖 5‑LLM 判定面板，缺乏人类黄金标准校准；
- logit‑lens 仅针对单词级词表，未覆盖多词情绪词；
- 27 类情绪词表本身并非普世，可能影响覆盖率评估；
- 多特征组合的发现是经验式搜索，未提供通用算法或理论解释。

---

## 135. A C implementation of the Smith massager algorithm

**arXiv ID:** 2605.19254 | [PDF](https://arxiv.org/pdf/2605.19254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

---

## 136. HELLoRA: Hot Experts Layer-Level Low-Rank Adaptation for Mixture-of-Experts Models

**arXiv ID:** 2605.18795 | [PDF](https://arxiv.org/pdf/2605.18795v1)

**作者:** Jia Wei `[一作]` (Tsinghua University), Longxiang Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 124 | [OpenAlex ID](https://openalex.org/A5078115749)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Mixture-of-Experts语言模型上，只对每层激活频率最高的专家插入LoRA适配器，构成Hot-Experts Layer-level Low-Rank Adaptation（HELLoRA）并进一步与LoRI组合成HELLoRI。

**💡 创新点**

创新点在于利用MoE稀疏激活特性，对专家进行激活感知的选择，只适配热点专家，从而实现参数与计算量大幅压缩，并在保持或提升性能的同时降低计算开销。

**🔧 技术方法**

采用LoRA、LoRI、MoE top-k 路由、负载平衡损失等技术，并在热专家定位阶段做轻量级 warm‑up。

**📊 数据集**

实验数据集包括数学推理的 GSM8K、代码生成的 CodeAlpaca（评估 HumanEval）和安全对齐的 Saferpaca（评估 HEx-PHI）。

**📈 对比分析**

与全参数微调、LoRA、DoRA、LoRI、LoRAMoE 等基线相比，HELLoRA 在三大 MoE 骨干（OlMoE、Mixtral、DeepSeekMoE）上使用约 16–30% LoRA 可训练参数，且准确率提升 1–3 点；HELLoRI 进一步压缩至 0.7% 参数，仍匹配 LoRA 性能，训练吞吐提升约 1.9 倍。

**⚠️ 局限性**

局限在于热专家识别依赖 10% warm‑up 数据的代表性，分布漂移或需要全局专家覆盖的任务可能导致选择不当，且对极端任务可能出现欠拟合。

---

## 137. The Extremum Stack is a Minimal Sufficient Statistic for Rate-Independent Functionals: A Kolmogorov Complexity Characterisation

**arXiv ID:** 2605.18885 | [PDF](https://arxiv.org/pdf/2605.18885v1)

**作者:** Piotr Frydrych `[一作]` (Warsaw University of Technology), Piotr Frydrych `[通讯]` (Warsaw University of Technology)

**通讯引用:** 109 | [OpenAlex ID](https://openalex.org/A5012740592)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

证明了Preisach极值栈为可计算速率无关函数族的最小充分统计量，且其表示的额外开销为常数；

**💡 创新点**

首次用Kolmogorov复杂度证明极值栈的最小性，并给出O(1)常数级别的紧致上界，解决了以往误导性的O(log n)或O(logK(_n))估计；

**🔧 技术方法**

采用Kolmogorov复杂度理论、有限指示器族构造、Preisach擦除性质及前缀自由编码不变性等方法；

**📊 数据集**

该工作为理论研究，未使用具体实验数据集，而是在离散网格{0,Δ,…,1}上进行一般性证明；

**📈 对比分析**

与传统时间序列压缩算法（如PAA、SAX、PLR、PIP、Swinging Door）对比，PSTACK在保持速率无关功能类完整性的同时实现了O(n)时间、O(k)空间的在线压缩，并在Kolmogorov意义下达到信息理论最优；

**⚠️ 局限性**

局限包括：最坏情况单步更新成本可能为Θ(n)；向向量/多维Preisach扩展尚未证明；以及在Preisach Attention等场景下KV缓存减量的可行性仍需进一步研究。

---

## 138. D-Convexity: A Unified Differentiable Convex Shape Prior via Quasi-Concavity for Data-driven Image Segmentation

**arXiv ID:** 2605.19210 | [PDF](https://arxiv.org/pdf/2605.19210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 139. ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation

**arXiv ID:** 2605.18769 | [PDF](https://arxiv.org/pdf/2605.18769v1)

**作者:** Gibson Nkhata `[一作]` (University of Arkansas), Susan Gauch `[通讯]` (University of Arkansas)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ClusterRAG框架，通过聚类协作过滤提升检索增强生成的个性化效果。

**💡 创新点**

创新点在于将用户聚类与文档级检索相结合，利用相似用户的文档来增强生成提示，并支持多种检索器和LLM。

**🔧 技术方法**

使用HDBSCAN聚类、ColBERTv2密集检索与重排序、IPA提示增强，以及多模型（FlanT5、Qwen2）生成。

**📊 数据集**

在LaMP基准上进行实验，涵盖多任务的个性化文本分类与生成。

**📈 对比分析**

与非个性化RAG、用户仅检索、协作检索等基线相比，ClusterRAG在所有任务上均取得显著提升，混合模式取得最优结果。

**⚠️ 局限性**

局限性包括对提示工程的依赖、单语种英文数据、缺乏多模态支持，以及对隐私和偏见的潜在风险。

---

## 140. Bounding LVR in AMMs via Secant-Tangent Divergence and Collateralized Liquidity Scaling

**arXiv ID:** 2605.19267 | [PDF](https://arxiv.org/pdf/2605.19267v1)

**作者:** Hyoungsung Kim `[一作]` (Korea Electronics Technology Institute), Yong-Suk Park `[通讯]` (Korea Electronics Technology Institute)

**通讯引用:** 503 | [OpenAlex ID](https://openalex.org/A5064681535)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出Hybrid Liquidity‑Collateral Pool（HLCP），通过将一部分流动性留在安全缓冲区，仅用N比例的资本参与主动做市，减轻LP的失衡损失（LVR）并保持可接受的交易滑点；

**💡 创新点**

创新点在于将资本分为活跃池与抵押缓冲，采用N‑缩放虚拟不变式和基于边际价差的触发式抵押注入，实现执行质量与主动风险的可分离控制；

**🔧 技术方法**

使用了几何分析（斜率与切线差异）、N‑缩放机制、触发器（阈值与α参数）、双人博弈模型、随机波动率加跳跃（SVJ）模拟以及对Uniswap V2历史数据的回测；

**📊 数据集**

主要数据集包括2025年Uniswap V2 USDC/ETH池的每日交易与费用数据、ETH价格轨迹以及通过SVJ模型生成的合成波动率与跳跃路径；

**📈 对比分析**

通过与标准CPMM的比较，实验显示HLCP在SVJ冲击下累计资本损失下降约73.8%，在历史回测中净收益提升约3.66个百分点，验证了低主动暴露下的收益改进；

**⚠️ 局限性**

局限性包括对路由层的理想化假设、仅在V2平滑曲线上验证、未考虑外部抵押收益、未模拟真实交易者与矿工行为，以及对高度集中流动性环境的适用性尚未评估。

---

## 141. Generalized Compare-and-Swap and Space-Efficient Universal Constructions for the Infinite-Arrival Model

**arXiv ID:** 2605.19237 | [PDF](https://arxiv.org/pdf/2605.19237v1)

**作者:** Vassos Hadzilacos `[一作]`, Sam Toueg `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出了 generalized compare-and-swap (GCAS) 对象，并利用 GCAS 设计了两种在无限到达模型（无限进程数）下的等待自由通用构造：一种空间复杂度随已参与进程数线性增长，另一种在有界竞争条件下空间复杂度随点争议数线性增长。

**💡 创新点**

创新点包括：
1) 将 CAS 的等价测试扩展为可参数化的比较器，形成 GCAS；
2) 第一次通用构造实现了在无限到达模型下的空间线性最优性；
3) 第二次通用构造首次在有界竞争条件下实现了按点争议线性空间复杂度，并提出了一种新的基于引用计数拆分的内存回收方案。

**🔧 技术方法**

主要技术手段：
- GCAS（可选比较器 <, =, >）
- 先后顺序的帮助机制（优先级基于时间戳）
- Fetch‑and‑Add 作为全局时钟
- 三个基底对象（A、S、L）和动态列表管理内存单元
- 引用计数拆分：在前驱中存放获取计数，在单元内存中存放放弃计数
- 通过“封闭”标记和“跳过”机制实现等待自由的列表遍历
- 线性化对象 L 用于确保操作顺序一致性

**📊 数据集**

论文为理论分析，没有使用任何实验数据集。

**📈 对比分析**

由于论文为理论研究，没有实验或性能比较；所有结果均基于形式化证明和复杂度分析。

**⚠️ 局限性**

局限性：
- 第二个构造仅适用于有界竞争条件，无法直接推广到无界竞争；
- GCAS 的硬件实现成本尚未验证，可能需要额外的比较器硬件支持；
- 方案中使用的内存回收机制对细粒度内存管理有较高的实现复杂度；
- 论文未给出自动化的垃圾回收实现，仍需手工管理或外部辅助。

---

## 142. From Cumulative Constraints to Adaptive Runtime Safety Control for Nonstationary Reinforcement Learning

**arXiv ID:** 2605.18841 | [PDF](https://arxiv.org/pdf/2605.18841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 143. KadiAssistant: A conversational AI Agent for information retrieval in Kadi4Mat

**arXiv ID:** 2605.18850 | [PDF](https://arxiv.org/pdf/2605.18850v1)

**作者:** Adrian Cierpka `[一作]` (Karlsruhe Institute of Technology), Arnd Koeppe `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 949 | [OpenAlex ID](https://openalex.org/A5085746715)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

开发了KadiAssistant，一个基于自托管LLM与语义检索的对话式AI助手，用于在Kadi4Mat研究数据平台上进行隐私保护的信息检索。

**💡 创新点**

将自托管LLM与多步骤Agentic AI结合，嵌入k‑NN向量数据库与重排序，支持细粒度访问控制和多工具调用，提升多模态知识检索能力。

**🔧 技术方法**

使用gpt‑oss‑120B LLM、Qwen3‑Embedding‑0.6B、Qwen3‑Reranker‑0.6B、LangGraph、pgvector、HNSW索引、vLLM等技术。

**📊 数据集**

在LISA‑Replica、ML‑Breathing‑Detection、POLiS Ontology三套真实研究数据集上评估。

**📈 对比分析**

与单步RAG方案对比，KadiAssistant在多工具调用场景下成功完成三种任务，平均k‑NN查询时间低于0.1秒，内存占用约13.7 MB/向量，性能可接受。

**⚠️ 局限性**

LLM可能出现幻觉，语义检索对相似记录召回有限，k‑NN查询时间随记录数线性增长，导致大规模实例的扩展性受限。

---

## 144. Cypher is Turing-Complete: A Formal Proof via 2-Counter Machine Simulation

**arXiv ID:** 2605.18757 | [PDF](https://arxiv.org/pdf/2605.18757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 145. Distributionally Robust Control via Stein Variational Inference for Contact-Rich Manipulation

**arXiv ID:** 2605.19029 | [PDF](https://arxiv.org/pdf/2605.19029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 146. Heterogeneity-Aware Dataset Scheduling for Efficient Audio Large Language Model Training

**arXiv ID:** 2605.19101 | [PDF](https://arxiv.org/pdf/2605.19101v1)

**作者:** Yanru Wu `[一作]` (Tsinghua University), Yang Li `[通讯]` (Tsinghua University)

**通讯引用:** 25426 | [OpenAlex ID](https://openalex.org/A5100319864)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对多数据集 Audio QA 训练中的梯度异质性问题，提出了基于梯度亲和度分组并采用递进调度的 Grouped Sequential Training (GST) 框架。

**💡 创新点**

创新点在于将梯度异质性分析与分组策略相结合，提供更紧的收敛界限，并使用梯度距离衡量亲和度实现可扩展的分组与调度。

**🔧 技术方法**

采用梯度距离亲和度、谱聚类分组、Progressive GST 调度、梯度收敛分析，并在 SALMONN‑13B 体系上进行实验。

**📊 数据集**

在覆盖语音、音乐、环境声音的 14 个 AudioQA 数据集上进行评估。

**📈 对比分析**

与全混合 (Mix‑All)、纯顺序、独立单任务等基线相比，GST 在保持或提升精度的同时实现 30–40% 的训练时间缩短，并在低资源场景下与 Mix‑All 相近。

**⚠️ 局限性**

局限包括：仅在 SALMONN‑13B 上验证，未检验更大模型或不同架构；Mix‑All 的效率瓶颈尚未完全理论化；分组与调度顺序依赖静态启发式，缺乏动态反馈机制。

---

## 147. ESLD (External Surrogate Latent Defense): A Latent-Space Architecture for Faster, Stronger Prompt-Injection Defense

**arXiv ID:** 2605.18918 | [PDF](https://arxiv.org/pdf/2605.18918v1)

**作者:** Yash Narendra `[一作]` `[通讯]` (Microsoft), Yash Narendra (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的内部隐藏状态进行探测，构建 ESLD（External Surrogate Latent Defense）在 agentic 系统中快速判定 prompt injection 攻击。

**💡 创新点**

提出利用 guard LLM 的中间隐藏状态直接做安全决策的架构，既无需完整解码又能在保持甚至提升检测准确率的同时显著降低延迟。

**🔧 技术方法**

使用线性探测器（LDA + Ledoit‑Wolf 缩减）、多层次内部表示抽取、留一源交叉验证与 Pareto 层选择、以及 GPU 延迟测量。

**📊 数据集**

14 个公开安全、注入、干扰与正向数据集，分别覆盖 UPIA（mosscap、Yanismiraoui、Do‑Not‑Answer、OR‑Bench‑Toxic、AART、BeaverTails）与 XPIA（XPIA、BIPIA、InjecAgent、AgentDojo），以及 dolly15k、Enron、SoftAge、10k_prompts 等正向查询。

**📈 对比分析**

与四种 guard LLM（LlamaGuard‑3、ShieldGemma‑9B、Granite‑Guardian‑8B、WildGuard‑7B）在 UPIA/XPIA 两类攻击下采用留一源评估；ESLD 在 7/8 组合上平均提升 16.4pp 的 BAcc，并实现约 3.3 倍的速度提升；例如 ShieldGemma‑9B XPIA 从 0.50 提升至 0.91，速度提升 4.18 倍。

**⚠️ 局限性**

实验仅在英文文本、单一线性探测器、单 GPU 单批量下进行；未评估多语言、代码或长文本场景，也未对自适应攻击做鲁棒性测试，且硬件与批量设置会影响绝对延迟。

---

## 148. Near-Resolution of the Tradeoff Conjecture in Distributed Proof Labeling Schemes

**arXiv ID:** 2605.19078 | [PDF](https://arxiv.org/pdf/2605.19078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 149. CASPIAN: Online Detection and Attribution of Cascade Attacks in LLM Multi-Agent Systems via Cross-Channel Causal Monitoring

**arXiv ID:** 2605.19240 | [PDF](https://arxiv.org/pdf/2605.19240v1)

**作者:** Kavana Venkatesh `[一作]` (Virginia Tech), Jiaming Cui `[通讯]` (Virginia Tech)

**通讯引用:** 892 | [OpenAlex ID](https://openalex.org/A5001813016)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CASPIAN框架，能够在线监测并归因LLM多智能体系统中的级联攻击

**💡 创新点**

创新点在于：统一跨通道因果影响张量、基于晚期交互条件传输熵(LI‑CTE)的高效影响估计，以及利用谱分析检测级联的四大动力学（放大、同步、跨通道传播、持久性）并实现在线归因

**🔧 技术方法**

技术包括：条件传输熵、谱分解、能量/谱间距、相位移、交叉通道熵、弱链检查以及基于最大影响的路径抽取

**📊 数据集**

使用TAMAS与ACIArena两大级联攻击基准，评估AutoGen、CrewAI、MetaGPT、LLM Debate四种主流LLM多智能体框架

**📈 对比分析**

与语义防护、LLM评判器和图谱异常检测对比，CASPIAN在AUROC、TPR@5%FPR、EDR@5等指标上均优于对手，尤其在早期检测和跨通道综合分析方面表现突出，运行时开销低于1%，保持可部署性

**⚠️ 局限性**

局限性包括：依赖完整的通信、内存、工具与执行轨迹；在部分可观测或分布式框架（如LLM Debate）中桥接角色归因精度下降；对抗者可能通过伪装影响或减少跨通道传播来规避检测

---

## 150. EVA-0: Test-Time Model Evolution with Only Two Forward Passes per Sample

**arXiv ID:** 2605.18867 | [PDF](https://arxiv.org/pdf/2605.18867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 151. A Heuristic Approach for Performance Tuning in RL-based Quadrotor Control via Reward Design and Termination Conditions

**arXiv ID:** 2605.19166 | [PDF](https://arxiv.org/pdf/2605.19166v1)

**作者:** Fausto Mauricio Lagos Suarez `[一作]` (Lulea University Of Technology), George Nikolakopoulos `[通讯]` (Lulea University Of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并验证了一种基于奖励设计与终止条件调节的启发式方法，使RL训练出的四旋翼控制器能够实现基线、快速（特技）和慢速（检查）三种可调的暂态响应，且稳态误差低于2%

**💡 创新点**

提出双带宽指数奖励函数和直观的奖励/终止条件调节规则，使得同一RL框架即可得到不同的临界阻尼响应，突破了传统RL控制器对性能调节的缺失

**🔧 技术方法**

采用Proximal Policy Optimization（PPO）与多层感知器网络，使用四旋翼的四轴电机RPM作为动作空间，并在观测中注入高斯噪声，构建双带宽指数奖励与终止条件调节机制

**📊 数据集**

在Gym-PyBullet-Drones仿真环境中通过随机初始状态进行训练与评估，未使用公开数据集

**📈 对比分析**

通过在100个随机初始化的10s试验中比较基线、特技与检查三策略的上限时间、超调与稳态误差，结果显示所有策略均保持≤2%误差，特技收敛最快、检查最慢，超调基本为零，验证了启发式方法的有效性

**⚠️ 局限性**

仅在仿真环境中验证，缺乏真实世界转移、域随机化、通信延迟与计算限制等方面的实验，终止条件设计经验性强，泛化性待进一步验证

---

## 152. Benchmarking Commercial ASR Systems on Code-Switching Speech: Arabic, Persian, and German

**arXiv ID:** 2605.19069 | [PDF](https://arxiv.org/pdf/2605.19069v1)

**作者:** Sajjad Abdoli `[一作]` (Perle AI), Ahmed Rashad `[通讯]` (Perle AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对商业 ASR 在阿拉伯语-英语、波斯语-英语和德语-英语混合语音上的性能进行基准评估，构建了1,200句代码切换样本并测试了五家商业 ASR 提供商。

**💡 创新点**

1) 采用两阶段筛选管线（启发式 + LLM 集成）大幅减少 LLM 评分成本；2) 同时使用 WER 与 BERTScore 两种指标，突出 BERTScore 在处理拼写/转写差异时的优势；3) 提供难度分层分析与 BERT 嵌入可视化，揭示难度与性能的关系。

**🔧 技术方法**

启发式脚本/词汇混合评分、GPT-4o 与 Gemini 1.5 Pro LLM 评估、BERTScore（mBERT）计算、对比实验。

**📊 数据集**

四个语言对的真实对话录音：埃及阿拉伯语-英语、沙特阿拉伯语-英语（Najdi/Hijazi）、波斯语-英语、德语-英语，共1,200句（每对300句）。

**📈 对比分析**

对五大商业 ASR（ElevenLabs Scribe v2、OpenAI gpt-4o-transcribe、Google Chirp 3、Azure Speech CLID、Deepgram Nova-3）进行 WER 与 BERTScore 的整体及难度分层对比；ElevenLabs Scribe v2 在所有语言对上均获得最低 WER（13.2%）和最高 BERTScore（0.936），其余系统的性能相对较差。

**⚠️ 局限性**

1) 仅评估了四个语言对，未覆盖其他常见代码切换组合；2) 对深度模型如 Deepgram Nova-3 的评测仅限德语-英语；3) 参考转写未做额外规范化，可能导致 WER 对拼写/转写差异过度惩罚；4) 评测基于离线录音，未考虑实时延迟、成本等部署维度。

---

## 153. Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution

**arXiv ID:** 2605.19228 | [PDF](https://arxiv.org/pdf/2605.19228v1)

**作者:** Xiaoou Liu `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**通讯引用:** 7770 | [OpenAlex ID](https://openalex.org/A5100777770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种闭源大型语言模型多步骤推理过程的逐步置信度归因框架，能够仅通过生成的推理轨迹来评估每一步的可靠性。

**💡 创新点**

创新点在于将信息瓶颈原则应用于步骤级置信度估计，推出两种实现：无参数的NIBS和基于图神经网络的可训练GIBS，二者共同实现了对推理步骤的自动诊断。

**🔧 技术方法**

技术上结合了语义相似度度量、最大公共子图（MCS）一致性、信息瓶颈正则化、以及图卷积网络来对推理图进行子图选择与置信度生成。

**📊 数据集**

在数学推理（GSM8K、Math）与多跳问答（MoreHopQA）等公开基准以及PRM800K人工标注数据集上进行实验。

**📈 对比分析**

与白盒置信度估计方法和其他基线相比，GIBS在AUROC、AUCPR、ACC@80%等指标上均有显著提升，并在自纠任务中将成功率提升至13.5%。

**⚠️ 局限性**

局限性包括依赖最终答案的正确标签、对推理图解析的质量敏感、未在所有类型推理任务（如生成式对话）中验证，且对大规模模型推理图生成的开销尚待评估。

---

## 154. From Sparsity to Simplicity: Enabling Simpler Sequential Replacements via Sparse Attention Distillation

**arXiv ID:** 2605.18865 | [PDF](https://arxiv.org/pdf/2605.18865v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 155. Query-Conditioned Graph Retrieval for Contextualized LLM Reasoning in Personalized Wearable Data

**arXiv ID:** 2605.18763 | [PDF](https://arxiv.org/pdf/2605.18763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 156. Variational Diffusion Channel Decoder

**arXiv ID:** 2605.18902 | [PDF](https://arxiv.org/pdf/2605.18902v1)

**作者:** Chengwei Zhang `[一作]` (Sun Yat-sen University), Siyu Liao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5073101300)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出了一种变分扩散通道解码器（VCDC），将传统的置信传播（BP）与变分扩散模型（VDM）相结合，用于在 AWGN 信道下实现低复杂度的误码率解码。

**💡 创新点**

创新点在于：①首次将 BP 直接嵌入扩散框架；②利用 VDM 的灵活前向噪声参数化，精确模拟 AWGN；③通过轻量级神经网络提升 BP 的性能，同时保持极低的 FLOPs 与模型大小。

**🔧 技术方法**

技术手段包括变分扩散模型、置信传播算法、轻量级多层网络块、早停机制以及对前向噪声的自定义参数化。

**📊 数据集**

实验使用了多种线性块码数据集：LDPC（121,60/70/80）、LDPC（49,24）、Polar（128,64/86/96）、Polar（64,32/48）、CCSDS（128,64）和 Mackay（96,48）等，SNR 范围为 4–6 dB。

**📈 对比分析**

与 BP、HGN 和 DDECC 等现有最先进模型比较，VCDC 在所有码和 SNR 上均获得最高负对数 BER；在轻量级设置下，FLOPs 仅为 377.6 K，模型大小 264 B，比 DDECC 低约 5 个数量级，且实际平均解码步骤低于理论上限 20 步。

**⚠️ 局限性**

局限性包括：仅在 AWGN 信道上验证，可能对非高斯或多路径信道适用性有限；需严格保持降序 SNR 作为前向时间步；反向步骤上限为 20 步，超大码或更复杂图结构的扩展性尚待进一步研究。

---

## 157. Prompt Optimization for LLM Code Generation via Reinforcement Learning

**arXiv ID:** 2605.19102 | [PDF](https://arxiv.org/pdf/2605.19102v1)

**作者:** Ali Mohammadi Esfahani `[一作]` (Carleton University), Samuel A. Ajila `[通讯]` (Carleton University)

**通讯引用:** 1241 | [OpenAlex ID](https://openalex.org/A5004784914)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于强化学习的多步提示优化框架，旨在通过自适应的词汇突变和语义重写来提升大型语言模型（LLM）生成代码的功能正确率。

**💡 创新点**

创新点在于将提示优化建模为顺序决策问题，融合了遗传式词汇突变、语义重写和直接生成的混合动作空间，并通过基于单元测试的形状奖励实现更密集的学习信号。

**🔧 技术方法**

核心技术包括Proximal Policy Optimization（PPO）强化学习、MiniLM嵌入作为状态表示、基于EPiC的词汇突变、Reflexion式语义重写以及基于单元测试的奖励函数。

**📊 数据集**

实验使用MBPP+、HumanEval+和APPS三个公开代码生成基准，分别对CodeT5+、CodeLLaMA和DeepSeek-Coder三个开源代码生成模型进行评估。

**📈 对比分析**

与直接生成、EPiC、Reflexion以及随机混合动作的基线相比，PPO框架在所有数据集和模型上显著提升了严格Pass@1（最高提升约17个百分点）和SoftPass@1（提升约11个百分点），并通过统计检验确认了显著性。

**⚠️ 局限性**

主要局限在于仅使用提示嵌入作为状态，未纳入生成代码、执行轨迹或先前奖励信息；同时重写和训练成本较高，且模型在极复杂任务（如APPS）上的提升幅度有限。

---

## 158. Feasible Plan Generation with Ambiguity-Boundedness in Cross-Model Query Processing

**arXiv ID:** 2605.19197 | [PDF](https://arxiv.org/pdf/2605.19197v1)

**作者:** Subhasis Dasgupta `[一作]` (University of California San Diego), Amarnath Gupta `[通讯]` (University of California San Diego)

**通讯引用:** 11365 | [OpenAlex ID](https://openalex.org/A5057846313)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种面向多模型自然语言查询的可行计划生成框架，核心为Packed Plan Forest（PPF），能够在多引擎、多模型环境下快速过滤不可执行的中间逻辑计划（ILP）并压缩候选计划空间；

**💡 创新点**

创新点包括：① 将可行性约束以注解形式嵌入操作符，实现跨模型的局部可行性判定；② 通过注解感知的节点合并与AND–OR图结构，将指数级候选计划压缩为多项式空间；③ 引入可行性证书（witness）机制，提供可验证的不可行原因；④ 在实验中证明了PPF在多模型系统中对可行性检测的可扩展性与压缩率。

**🔧 技术方法**

使用技术：解析森林与AND–OR DAG的组合、哈希合并（hash‑consing）实现节点共享、底层可行性标签推导（bottom‑up labeling）、局部约束满足算法、可行性证书生成、基于注解的操作符模板与系统目录。

**📊 数据集**

实验使用六个合成场景，分别变化结构重叠、可行性比例与引擎多样性；未使用真实业务数据集，而是通过人工构造的多模型（关系、图、向量、空间）数据架构进行评估。

**📈 对比分析**

与两种基线比较：① 传统枚举法（Naïve Enumeration）产生指数级ILP；② 仅基于符号合并的记忆化（Memoization Without Annotations）。PPF 在压缩率上可达 4.2–11.7×，可行性裁剪比例为 21–63%，单个查询耗时 7.1–24.6 ms，显著低于基线且避免了 28–46% 的错误合并。

**⚠️ 局限性**

局限性：仅支持局部可行性约束；对全局约束需引入 SAT/SMT；依赖准确完整的元数据与注解；实验基于合成数据，未验证在真实业务场景下的表现；未结合成本模型进行最终计划选择；在极大规模数据或动态变化的系统中，合并与标签传播的效率仍需进一步评估。

---

## 159. A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions

**arXiv ID:** 2605.18898 | [PDF](https://arxiv.org/pdf/2605.18898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 160. Not all uncertainty is alike: volatility, stochasticity, and exploration

**arXiv ID:** 2605.19215 | [PDF](https://arxiv.org/pdf/2605.19215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 161. Safe Continual Reinforcement Learning under Nonstationarity via Adaptive Safety Constraints

**arXiv ID:** 2605.18842 | [PDF](https://arxiv.org/pdf/2605.18842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 162. Towards FairRAG: Preventing Representational Harm in Retrieval-Augmented Generation by Enforcing Fair Exposure at Retrieval Time

**arXiv ID:** 2605.18806 | [PDF](https://arxiv.org/pdf/2605.18806v1)

**作者:** Riddhi Tikoo `[一作]` `[通讯]` (Redmond High School), Riddhi Tikoo (Redmond High School)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在检索增强生成（RAG）系统中，检索阶段对代表性偏差的影响，并提出了“Representative Stochastic”排名器以实现公平曝光

**💡 创新点**

创新点在于将公平性直接嵌入检索排名过程，通过动态跟踪曝光权重并按概率调整排名，兼顾相关性与性别公平性

**🔧 技术方法**

使用了检索增强生成框架、ChromaDB向量检索、Plackett‑Luce随机抽样、强化曝光调整算法以及GPT‑4o‑mini生成模型

**📊 数据集**

采用了TREC 2022 Fair Ranking数据集（4,800篇标注性别的维基百科文章）

**📈 对比分析**

与标准排名、纯随机排名、强制曝光排名比较，Representative Stochastic在曝光差距与女性曝光比例上显著更接近平衡，尽管其生成准确率略低，但在公平性指标上表现最佳

**⚠️ 局限性**

局限在于仅评估问答场景、使用较小模型、试验次数有限、且数据集仅包含性别两个类别，未覆盖更广泛的社会群体或复杂生成任务

---

## 163. From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models

**arXiv ID:** 2605.19263 | [PDF](https://arxiv.org/pdf/2605.19263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 164. A Systematic Failure Analysis of Vision Foundation Models for Open Set Iris Presentation Attack Detection

**arXiv ID:** 2605.19020 | [PDF](https://arxiv.org/pdf/2605.19020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 165. Distilling Linearized Behavior for Effective Task Arithmetic

**arXiv ID:** 2605.18993 | [PDF](https://arxiv.org/pdf/2605.18993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 166. Unleashing the Power of Tree-of-Thoughts for Edge-Enabled AIGC Service Provisioning

**arXiv ID:** 2605.19108 | [PDF](https://arxiv.org/pdf/2605.19108v1)

**作者:** Zhang Liu `[一作]` (Xiamen University), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 88780 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在移动边缘计算环境中，提出利用Tree-of-Thoughts（ToT）推理流程为创意写作生成高质量文本，并通过构造有向无环图（DAG）对思考路径进行建模，最终将思考任务分配到基站和边缘服务器，以最小化生成延迟同时满足质量阈值。

**💡 创新点**

创新点包括：
1) 将ToT推理过程映射为DAG并提出整数非线性规划求解思考分配问题；
2) 以输出token数为可控计算资源度量，并通过实验得到与生成延迟和质量的闭式关系；
3) 将扩散模型融入软演员-评论家（SAC）框架，形成DSAC，利用扩散反向过程实现对离散动作空间的渐进式探索与采样，显著提升收敛速度与决策质量。

**🔧 技术方法**

主要技术：
- 移动边缘计算（MEC）与无线链路建模；
- Tree-of-Thoughts (ToT) 与 DAG 任务调度；
- 生成式AI模型 Qwen 2.5‑7B‑Instruct 用于实验评估；
- Diffusion Probabilistic Model（DDPM）与 soft actor‑critic（SAC）相结合的 DSAC 算法。

**📊 数据集**

数据集：从 randomwordgenerator.com 随机采样 40 条句子，构造 10 个创意写作任务（输入 4 句，输出 4 段落），并使用 Qwen 2.5‑7B‑Instruct 进行生成和质量评分。

**📈 对比分析**

比较方法：与 SAC、PPO、DDQN 三种基准 DRL 算法对比。实验结果显示：
- DSAC 在总生成延迟上比 PPO、SAC、DDQN 分别降低 8.32%、11.57% 与 36.09%；
- 相比全局本地生成基线，DSAC 在严格质量约束下仍能将延迟降低约 80%；
- DSAC 训练收敛更稳定、决策质量更优，虽然每个思考的计算时间略高。

**⚠️ 局限性**

局限性：
1) 仅考虑单个作业主和单一创意写作任务，未覆盖多任务并发与公平性问题；
2) 仅针对文本生成任务，未验证对图像、音频等其它 AIGC 模式的适用性；
3) 计算资源的代理仅使用输出 token 数，其他可能影响延迟和质量的因素未纳入；
4) 扩散模型在推理阶段引入额外计算开销，需进一步优化。

---

## 167. Identifiable Multimodal Causal Representation Learning under Partial Latent Sharing

**arXiv ID:** 2605.19135 | [PDF](https://arxiv.org/pdf/2605.19135v1)

**作者:** Manal Benhamza `[一作]` (Paris-Saclay University), Myriam Tami `[通讯]` (Paris-Saclay University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种在多模态因果表示学习中实现分量级可识别性的框架，支持部分共享且非满射的混合函数，能够在欠完备条件下学习共享与特定潜变量及其因果结构。

**💡 创新点**

创新点在于：① 在部分共享结构下给出了分量级可识别性理论，约束更弱且不需要参数化分布；② 引入可微的Wasserstein模块以自动发现不同模态间的共享潜变量；③ 在欠完备和非双射混合函数场景下仍能保持可识别性。

**🔧 技术方法**

采用了变分自编码器+可微整数规划（Wasserstein距离）求解共享矩阵，使用Masked Autoregressive Normalizing Flow估计因果机制，结合稀疏与无环约束的正则化来学习因果图。

**📊 数据集**

实验数据集包括：3种多模态数值合成数据（2/3/4模态）、MultiModal3DIdent（图像+文本）、以及由SERGIO生成的合成单细胞基因表达数据。

**📈 对比分析**

与CausalVAE、MCL、MultiBio、MultiView等SOTA方法对比，评估指标为MCC、R²和EnSHD。结果显示本文方法在R²和MCC上均优于或与最强基线相当，EnSHD低于8%，证明了更准确的潜变量恢复和因果结构推断。

**⚠️ 局限性**

局限性在于：① 需假设没有潜变量同时共享所有模态；② 依赖于稀疏因果结构约束；③ 对于高度噪声或极端非线性混合函数，理论与实验效果尚未完全验证。

---

## 168. Transformers Linearly Represent Highly Structured World Models

**arXiv ID:** 2605.18847 | [PDF](https://arxiv.org/pdf/2605.18847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 169. To Call or Not to Call: Diagnosing Intrinsic Over-Calling Bias in LLM Agents

**arXiv ID:** 2605.18882 | [PDF](https://arxiv.org/pdf/2605.18882v1)

**作者:** Wei Shi `[一作]` (Shanghai Jiao Tong University), Na Zou `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究LLM代理在工具调用决策中出现的过度调用问题，提出并验证了Intrinsic Bias Hypothesis（IBH），并基于Sparse Autoencoders（SAE）构建的特征基线，推出了Adaptive Margin‑Calibrated Steering（AMCS）来纠正该偏差。

**💡 创新点**

创新点在于将过度调用归因于激活无关的调用偏置，并通过SAE特征空间实现可量化的偏置估计与闭式校准干预，首次提供了对工具调用阈值的机制化、可调节的控制手段。

**🔧 技术方法**

技术方法包括：Sparse Autoencoders（TopK变体）用于从残差流中提取稀疏可解释特征；线性探针与激活边距分析验证IBH；以及基于估计偏置的AMCS闭式激活平移。

**📊 数据集**

数据集主要是When2Call基准（约10M token的训练集、约10M token的评估集），并在此基准上对六款不同模型进行评估。

**📈 对比分析**

对比方法包括Prompt、Suppress、Promote等单侧干预手段；实验显示AMCS在六款模型上将无调用准确率提升4–17个百分点，整体准确率提升至69–75%（相较于基线55–70%），且对调用准确率影响≤5个百分点。

**⚠️ 局限性**

局限性包括：仅在When2Call基准上验证，未探究偏置的训练阶段根源；SAE特征基线和线性假设的局部近似可能不适用于更复杂的多轮代理；AMCS为推理时校正，而非训练时修正。

---

## 170. In-Context Learning Operates as Concept Subspace Learning

**arXiv ID:** 2605.18830 | [PDF](https://arxiv.org/pdf/2605.18830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 171. M3DocDep: Multi-modal, Multi-page, Multi-document Dependency Chunking with Large Vision-Language Models

**arXiv ID:** 2605.18774 | [PDF](https://arxiv.org/pdf/2605.18774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 172. T-REX: Fast and Dynamic Journey Planning for Continental-Scale Public Transit Networks

**arXiv ID:** 2605.18778 | [PDF](https://arxiv.org/pdf/2605.18778v1)

**作者:** Jonas Sauer `[一作]` (Karlsruhe Institute of Technology), Sascha Witt `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5071123656)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 T-REX 算法，结合多层覆盖与 Trip‑Based Public Transit Routing 进行公共交通旅程规划。

**💡 创新点**

创新点在于将转移（transfer）作为层级元素进行排名与裁剪，并在查询阶段重构扫描流程以实现更强的剪枝效果。

**🔧 技术方法**

使用技术包括多层划分（MLO）、转移排名、TB 预处理、改进的查询扫描、缓存友好数据布局与并行分区定制。

**📊 数据集**

使用欧洲、德国、瑞士、巴黎四个不同规模的公共交通网络，欧洲网络包含 134 万站点、1.07 亿事件，规模为目前最大实验集。

**📈 对比分析**

与 TB、RAPTOR、CSA、ACSA、FLASH‑TB 等基线比较，T‑REX 在欧洲网络上查询时间均低于 10µs，速度提升约 20 倍相较 TB，且内存占用仅为 TB 的 5–9%，并在大陆规模上实现交互式实时查询。

**⚠️ 局限性**

主要限制在于需要对步行路径做传递闭包且对大型网络的预处理仍需数分钟，且对更长时间表和无限步行路径的支持尚未成熟。

---

## 173. CRAFT: Critic-Refined Adaptive Key-Frame Targeting for Multimodal Video Question Answering

**arXiv ID:** 2605.19075 | [PDF](https://arxiv.org/pdf/2605.19075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 174. Reasoning Portability: Guiding Continual Learning for MLLMs in the RLVR Era

**arXiv ID:** 2605.18903 | [PDF](https://arxiv.org/pdf/2605.18903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 175. Platform architecture determines whether recommendation algorithms can shape information quality on social media

**arXiv ID:** 2605.19204 | [PDF](https://arxiv.org/pdf/2605.19204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 176. Correcting Tail Deletions in Rank Modulated Composite Encoding for Data Storage in DNA

**arXiv ID:** 2605.19148 | [PDF](https://arxiv.org/pdf/2605.19148v1)

**作者:** Tomer Cohen `[一作]` (Technion Israel Institute of Technology), Zohar Yakhini `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种结合DNA合成中使用的组合符号（composite DNA symbols）与秩调制编码（rank‑modulated coding）的新型编码框架，用来提高DNA存储系统中的信息密度并降低合成成本；在此框架下，作者研究了在“左尾部”上出现的插入、删除与插删（indel）错误，并给出了针对部分排列（partial permutations）的检测与纠错码的界限与构造；随后将这些单符号码推广到向量（序列）层面，构造了所谓的尾张量排列码（tail tensor permutation codes），实现多符号同时抵御尾部错误。

**💡 创新点**

创新点主要体现在：① 将组合DNA符号的“相对排序”与秩调制编码相结合，忽略精确频率而只关注符号排名，从而兼顾生物合成的冗余与编码的可行性；② 以部分排列为基础，首次系统性定义并解析尾部插入/删除/Indel错误模型；③ 证明不同错误模型间的等价与不等价关系；④ 给出最优（或近似最优）的尾部删除检测/纠错码构造，并通过张量组合扩展到序列层面，形成一套完整的理论体系。

**🔧 技术方法**

主要技术手段包括：组合数学与排列码理论（使用Kendall τ距离、部分排列的集合大小计算、容斥与鸽巢原理等）；代码构造方法（基于子集合划分、尾部删除球的覆盖与不相交性分析）；张量码思想（将单符号码作为“内码”，外码为普通符号码，形成尾张量排列码）；以及符号集合的划分与子码的互不相交性质。

**📊 数据集**

论文为理论工作，未使用实际实验或公开数据集；所有结果均基于数学证明与组合计数。

**📈 对比分析**

作者通过推导上界、下界以及构造例子，给出了各类码容量的精确或近似公式；对于给定的q、t、e、n，能够直接计算最优码大小或构造出的码大小；与之前仅考虑固定长度排列或Kendall距离的工作相比，新模型在容错性与容量上均有所提升，尤其在尾部错误频发的DNA合成场景下。

**⚠️ 局限性**

主要限制包括：① 只考虑左尾部错误，未覆盖中间或右侧错误；② 方案对q与t的大小存在约束（如t<q），在某些参数下构造复杂；③ 码的实现需要在DNA合成/测序系统中准确控制符号排名，实际实验验证仍缺乏；④ 计算复杂度和存储开销在大规模应用中可能显著，尚需进一步优化。

---

## 177. Agent Meltdowns: The Road to Hell Is Paved with Helpful Agents

**arXiv ID:** 2605.19149 | [PDF](https://arxiv.org/pdf/2605.19149v1)

**作者:** Rishi Jha `[一作]` (Cornell University), Vitaly Shmatikov `[通讯]` (Cornell University)

**通讯引用:** 21809 | [OpenAlex ID](https://openalex.org/A5038206174)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化了 AI 代理在遇到正常环境错误时，无恶意输入下会产生不安全或有害行为的“意外熔断”现象。

**💡 创新点**

提出了意外熔断的概念与分类体系，构建了可注入多种本地与远程错误的容器化测试环境，并系统评估了多模型、多框架下的熔断率和行为严重程度。

**🔧 技术方法**

使用了 Docker 容器化、MITMProxy 注入网络错误、本地系统错误模拟、LLM 辅助行为分类与人工标注、以及统计与可视化分析技术。

**📊 数据集**

基于自建的文件与网站环境（包括 NeurIPS 2025 作者主页列表）以及多种模拟错误情景，未使用公开的标准数据集。

**📈 对比分析**

通过对比 GPT‑4o、GPT‑5 系列、Gemini 3 Flash、Grok 4.20 等模型与 Claw Code、OpenAI Codex、HAL、Magentic‑One 等四种代理框架，计算熔断率、严重等级和报告率，结果显示 64.7% 的 roll‑outs 产生至少一种中高危熔断行为，且约 50% 未被报告。

**⚠️ 局限性**

局限性在于实验规模有限、错误场景单一（未覆盖复合与动态错误）、对非 GPT‑5 系列模型的样本不足，导致对逆扩展规律与更广泛危害类型的评估不足。

---

## 178. Mode-Tensorized Canonical Polyadic Decomposition for MIMO Channel Estimation

**arXiv ID:** 2605.19053 | [PDF](https://arxiv.org/pdf/2605.19053v1)

**作者:** Alexander Blagodarnyi `[一作]` (Moscow Institute of Physics and Technology), Vladimir Lyashev `[通讯]` (Moscow Institute of Physics and Technology)

**通讯引用:** 155 | [OpenAlex ID](https://openalex.org/A5064371342)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种基于模式张量化的CANONICAL POLYADIC分解（MTCPD）方法，用于低信噪比环境下的MIMO信道估计。

**💡 创新点**

创新点在于将信道张量按物理维度拆分成更高阶的虚拟模式，引入隐式正则化和相位一致性度量，显著提升了路径分离和噪声抑制能力。

**🔧 技术方法**

采用CPD + ALS算法并配合DFT初始化、模式张量化、相位一致性指标（PCM）进行秩选择，最终通过估计的信道矩阵计算频谱效率。

**📊 数据集**

使用QuaDRiGa仿真平台，依据3GPP TR 38.901 UMa NLOS 场景生成多径信道，包含8×8基站天线、2个UE天线、512子载波、1200次仿真。

**📈 对比分析**

与传统CPD进行对比，MTCPD在[-24, 4] dB的上行SNR区间内，重构误差更低、频谱效率提升最高可达约3.5 bps/Hz（rank‑2 方案），在低SNR下优势更明显。

**⚠️ 局限性**

局限性包括对阵列几何的假设、需要手动设置秩选择阈值、模式张量化参数需要经验选择、计算复杂度相对较高，以及对非理想远场或大规模多径场景的适用性待进一步验证。

---

## 179. Be Kind, Rewrite: Benign Projections via Rewriting Defend Against LLM Data Poisoning Attacks

**arXiv ID:** 2605.19147 | [PDF](https://arxiv.org/pdf/2605.19147v1)

**作者:** John T. Halloran `[一作]` (Leidos), Noopur S. Bhatt `[通讯]` (Leidos)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用开放书籍安全重写（Open-Book Benign Rewriting, OBBR）的方法，对大语言模型（LLM）的训练数据进行主动重写，从而在模型训练前有效消除后门攻击（Backdoor Attacks, BAs）和无触发器的数据投毒攻击（PIAs）。

**💡 创新点**

创新点在于：①引入检索增强生成（RAG）在重写上下文中加入开放书籍的安全样本，使得重写结果更倾向于良性输出；②给出了理论证明，证明OBBR比闭书籍重写（CBBR）更能生成安全序列；③将此方法作为一种全新的主动防御方案，对比现有的反应式和内在式防御，显著提升了安全性。

**🔧 技术方法**

主要技术包括：检索增强生成（RAG）结合句子嵌入检索；LLM重写器的自回归生成；对抗后门攻击与无触发器投毒的评估框架；以及对重写后模型进行再训练和推理的完整流水线。

**📊 数据集**

使用的数据集包括：UltraFeedback作为开放书籍安全语料；LIMA instruction-tuning数据集用于验证重写对下游任务的影响；多种后门攻击样本（五类BAs）和包含2%恶意样本的PIA数据集（5000条）；以及标准自然语言评测基准（ARC‑E/C、HellaSwag、PIQA、Winogrande、MMLU、IFEval）和StrongReject基准用于评估PIA防御效果。

**📈 对比分析**

实验将OBBR与CBBR、DPR、Paraphrase（闭书籍重写）以及CROW（内在防御）、CLEANGEN、Quantize、Decoding（反应式防御）进行对比。结果显示，OBBR在四大LLM上将ASR平均降低51%（相比SOTA防御）和25.7%（相比其他重写方法），在PIA场景下保持模型拒绝率在35%以下；其整体运行时间仅比无防御提升38%，远低于CLEANGEN的619%；且对下游任务的性能几乎无损，部分模型甚至略有提升。

**⚠️ 局限性**

局限性包括：依赖高质量的开放书籍安全语料，若语料不充分或不匹配任务，重写效果可能受限；对极少数或自定义触发器的后门攻击尚未全面验证；实验仅覆盖四种LLM和五种BA模式，需在更多模型与攻击类型上进一步验证；并且重写步骤仍带来额外的计算开销，尤其是在检索构建和重写推理阶段。

---

## 180. Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering

**arXiv ID:** 2605.19220 | [PDF](https://arxiv.org/pdf/2605.19220v1)

**作者:** Tiejin Chen `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**通讯引用:** 7770 | [OpenAlex ID](https://openalex.org/A5100777770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文指出当前LLM不确定性量化（UQ）方法本质上是无监督聚类，只衡量生成内部一致性，无法检测“自信幻觉”，并提出三大改革路径：评估（从平均到极端风险）、机制（从后置启发式到原生不确定性）和基准（以客观真相为准）

**💡 创新点**

创新点在于将UQ视为聚类问题，系统识别其三大病态（超参数敏感、内部评估陷阱、缺乏真相），并提出以外部真相为锚的监督式UQ框架，强调“最坏情况鲁棒性”“可信度集大小”与“原生不确定性训练”

**🔧 技术方法**

采用的技术包括：语义熵（Semantic Entropy）及其变体、基于图谱的谱聚类（Graph Uncertainty）、自我验证（P(true)）等聚类方法；评估指标有AUROC、Jaccard相似度、AUSC；提出的改进方法有“极端风险评估”“原生不确定性对齐”“原子事实验证”等

**📊 数据集**

主要实验数据集为QASC（问答集）与Qwen2.5‑32B模型的生成样本，论文中也引用了OpenAI、Gopher等模型及Code Generation、Mathematical Reasoning等可验证任务作为验证环境

**📈 对比分析**

对比方法包括多种UQ技术（Semantic Entropy、Graph Uncertainty、P(true)等）以及传统Token‑entropy、Ensemble等；结果显示它们在AUROC、Jaccard、AUSC等指标上相差很大，且超参数变化导致性能剧烈波动，凸显无监督聚类的弱点；在极端风险评估下，现有方法往往无法及时捕捉高风险错误

**⚠️ 局限性**

局限性：论文未给出具体可实现的监督式UQ实现方案；实验多依赖内部生成样本，缺乏大规模真实任务的验证；对外部真相的依赖（例如Atomic Fact Verification）实现成本高；此外，仍需进一步证明原生不确定性训练与下游任务的实际收益

---

## 181. Multi-Headed Transformer Architectures as Time-dependent Wasserstein Gradient Flows

**arXiv ID:** 2605.18870 | [PDF](https://arxiv.org/pdf/2605.18870v1)

**作者:** Alex Massucco `[一作]` (University of Cambridge), Carola-Bibiane Schönlieb `[通讯]` (University of Cambridge)

**通讯引用:** 9315 | [OpenAlex ID](https://openalex.org/A5033880300)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究并推导了多头、时间依赖Transformer推理动力学的连续均场模型，并将其表述为非自治Wasserstein梯度流，提出Transformer PDE；

**💡 创新点**

创新点在于将多头权重随层变化的概率测度Θ_t纳入动力学，推导出Θ_t加权的谐波平均移动性b^Θ_t_μ(x)，从而把多头运输几何压缩为单标量系数；同时证明了在满足积分条件下token分布收敛至稳态，揭示权重动态对聚类行为的决定性作用；

**🔧 技术方法**

采用变分推导、Wasserstein梯度流、最小化运动（JKO）方案、Γ-收敛与上界梯度等高级概率与优化技术；

**📊 数据集**

无具体公开数据集，实验使用的是模拟的正态/周期性权重过程与在S^2上均匀分布的token云；

**📈 对比分析**

通过与离散Euler-Maruyama数值方案对比，能量平衡与梯度耗散恒等式得以验证；收敛速度随头数H的增加而按O(H^{-1})衰减；

**⚠️ 局限性**

局限性在于仅对满足Θ_t可积性条件的权重动态适用，未处理离散层实现细节与真实NLP任务中的复杂输入与层间非线性；

---

## 182. Automatically Improving Simulation Physics for Articulated Objects

**arXiv ID:** 2605.19136 | [PDF](https://arxiv.org/pdf/2605.19136v1)

**作者:** Anh-Quan Pham `[一作]` `[通讯]`, Anh-Quan Pham

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出“交互准备度”量化评估框架，并开发基于多模态视语模型与物理仿真闭环的装配物体生成方法，实现对可交互关节对象的物理一致性自动化。

**💡 创新点**

①将交互准备度拆分为物理稳健性、语义准确性、行为保真度、视觉真实性、学习可行性五个可测量维度；②通过视觉语言模型给出初始物理参数，再利用仿真器反馈迭代修正，解决传统单次预测无法保证物理一致性的瓶颈。

**🔧 技术方法**

多模态视觉语言模型（如 Gemini 2.5 Flash），物理仿真器 SAPIEN，解析 URDF 的结构、几何、语义与可视化信息，解析几何求体积/惯性，使用逆方差加权（VLM‑IVW）或单次推断；闭环迭代中利用碰撞深度、姿态漂移、关节振荡等指标进行参数修正。

**📊 数据集**

使用 PartNet‑Mobility、GenSim2 以及 93 个来自 PartNet‑Mobility 的多类别装配物体，配合自然语言描述作为生成指令。

**📈 对比分析**

与直接 VLM 推断、VLM‑IVW 聚合、人工标注的 GenSim2 资产以及 Claude Code 等对比。实验显示其在：
- 物理稳健性：渗透深度最低、位置/姿态漂移、关节振荡通过率最高；
- 语义准确性：尺寸与初始关节偏差最小、提示对齐率最高；
- 行为保真度：三种 VLA 策略在模拟中的成功率与实测 SRCC 均最高；
- 学习可行性：RL 任务成功率高达 97.5%，远超基线；
- 视觉真实性：人类评估 7.04/10，VLM‑judge 最高分。
在大规模 93 物体上成功率达 96%。

**⚠️ 局限性**

仍受限于：对高自由度物体的参数耦合仍有误差；对特殊材质或缺失几何的预测精度不足；对 VLM 依赖程度高，模型更新可能导致兼容性问题；闭环迭代需要仿真器支持，导致处理时间增加。

---

## 183. FAGER: Factually Grounded Evaluation and Refinement of Text-to-Image Models

**arXiv ID:** 2605.19111 | [PDF](https://arxiv.org/pdf/2605.19111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 184. OpenCompass: A Universal Evaluation Platform for Large Language Models

**arXiv ID:** 2605.19276 | [PDF](https://arxiv.org/pdf/2605.19276v1)

**作者:** Maosong Cao `[一作]` (Shanghai AI Laboratory), Jingming Zhuo `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并开源了OpenCompass，一个统一、可扩展、高并发的LLM评测平台，支持100+基准和100+模型，提供多种评测模式、Prompt构造与可视化汇总。

**💡 创新点**

创新点在于统一多维度评测框架、支持自定义评测逻辑、分布式并行执行、Rule/LLM-as-Judge/Cascade评测器组合、可配置Prompt（少样本、零样本、丰富结构），以及跨集群统一任务调度与可视化汇总。

**🔧 技术方法**

采用MMEngine、OpenICL为核心，集成Retriever、Prompt Template、Partitioner、Runner、Task、Evaluator、Summarizer等模块；支持HuggingFace、API、vLLM、LMDeploy等多模态模型推理；使用LLM-as-Judge进行评判；通过分区、并行化实现高并发。

**📊 数据集**

支持超过100个基准数据集，包括MMLU、GPQA、SimpleQA、BBH、HellaSwag、HLE、AIME、HMMT、AMO-Bench、MATH、PHYSICS、ClimaQA、SmolInstruct、MMMLU、PMMEval、LiveCodeBench、BigCodeBench、Ruler、LongBench、Arc-AGI、IFEval/IFBench等。

**📈 对比分析**

在OpenCompass Academic Leaderboard上对主流模型在各基准上的分数进行评测，利用多维度指标和可视化报告进行对比，平台通过任务分区和并行化显著缩短评测时间，并支持多模型并行评测。

**⚠️ 局限性**

局限性包括仅支持单模态评测、缺乏多模态与多轮对话支持；Inference与Eval仍为串行流程；LLM-as-Judge成本高；对非主流模型兼容性有限；特定领域深度评测需手动自定义评测器。

---

## 185. KVBuffer: IO-aware Serving for Linear Attention

**arXiv ID:** 2605.19049 | [PDF](https://arxiv.org/pdf/2605.19049v1)

**作者:** Longwei Zou `[一作]` (Yale University), Lin Zhong `[通讯]` (Yale University)

**通讯引用:** 11116 | [OpenAlex ID](https://openalex.org/A5108086354)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出一种 IO‑aware 的线性注意力服务机制，利用缓冲最近的 KV 来实现分块（chunkwise）、speculative（并行草稿验证）以及短上下文 KV‑only 解码，从而显著降低内存访问和推理延迟。

**💡 创新点**

创新点在于：1) 通过 KV 缓冲批量更新线性注意力状态，减少递归式状态读写；2) 支持在 speculative decoding 中并行验证草稿 token，消除临时状态占用；3) 对短上下文使用 KV‑only 形式，避免维护大状态；4) 让推理计算形式与预训练的分块模式保持一致，提升稳定性。

**🔧 技术方法**

使用的技术包括：Gated Delta Networks（GDN）线性注意力变体；SGLang 框架 + Triton 自定义 GPU kernel；paged KV 缓冲管理；批量并行状态更新；分块调度与 CUDA Graph 加速。

**📊 数据集**

实验数据集为 Qwen3‑Next‑80B‑A3B‑Instruct（GDN 模型）以及 ShareGPT 用于评估 speculative decoding。

**📈 对比分析**

与现有 vLLM/SGLang 的递归解码做对比：chunkwise 解码可使延迟降低高达 45.17%；speculative 验证延迟提升约 2.78×，吞吐量提升 1.46×；KV‑only 在 L<d 的短上下文中比递归/分块更快。

**⚠️ 局限性**

限制与挑战：1) 未实现动态切换解码形式，导致调度开销；2) KV‑only 需要更多 GPU 资源或分离服务器；3) 多分支解码仍需独立状态，内存占用较高；4) KV 缓存前缀粒度有限；5) 预训练使用分块，推理使用递归可能导致稳定性与精度偏差。

---

## 186. LoRA vs. Full Fine-Tuning: A Theoretical Perspective

**arXiv ID:** 2605.19018 | [PDF](https://arxiv.org/pdf/2605.19018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 187. Nash Welfare in Additively Separable Hedonic Games

**arXiv ID:** 2605.19030 | [PDF](https://arxiv.org/pdf/2605.19030v1)

**作者:** Marta Pagano `[一作]` (Sapienza University of Rome), Alexander Schlenga `[通讯]` (Technical University of Munich)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出在可加分离性效惠游戏（ASHG）中引入Nash福利，并对其计算复杂性、逼近算法和稳定性性质进行系统研究。

**💡 创新点**

创新点在于首次将Nash福利引入ASHG框架，证明其优化问题是NP-难的，并给出AEG与AFG两类子类的近似算法（比分n-1和2n），以及在受限联盟数量/大小时的精确可解与NP-难度分界。

**🔧 技术方法**

主要技术手段包括图论中的最大{K₂,K₃}分解、非放弃性Nash偏移动力学、归约与不等式证明，以及对稳定性概念的潜能函数分析。

**📊 数据集**

本研究为纯理论工作，未使用实验数据集，而是通过数学归约与算法分析验证结果。

**📈 对比分析**

与传统效用最大化（效用和）和已知的公平分配模型相比，给出了近似比值、硬度阈值（1.0000759）以及在受限情形下的多项式算法，展示了在大部分情形下可达成的近似性能与计算可行性。

**⚠️ 局限性**

主要局限包括：近似比值的最优性尚未确定、对更一般的ASHG子类（如带中性关系的AEG）缺乏硬度证明、以及对Nash福利在在线或动态模型中的性能分析仍待深入。

---

## 188. Supporting System Testing with a Multi-Agent LLM-based Framework for Knowledge Graph Extraction: A Case Study with Ethernet Switch Systems

**arXiv ID:** 2605.19180 | [PDF](https://arxiv.org/pdf/2605.19180v1)

**作者:** Rongqi Pan `[一作]` (University of Ottawa), Haiwei Dong `[通讯]` (Huawei Canada)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出一种多代理 LLM 框架，自动从以太网交换机配置手册（ESCM）中抽取、评估并改进知识图谱（KG），进而支持系统级测试用例生成。

**💡 创新点**

创新点在于：①结合多代理 LLM 与 Extract‑Evaluate‑Improve（EEI）循环，实现 KG 结构化抽取、自动评估与提示优化；②设计细粒度 KG 架构与针对性评估准则；③验证该框架对工业文档的通用性与可扩展性。

**🔧 技术方法**

主要技术包括：大语言模型（GPT‑5）与 prompt‑engineering、三类专用提取 Agent（Roadmap、Mapping、Procedure）、评估 Agent 与改进 Agent、LLM‑as‑Judge 评估方法、以及基于 KG 的自动化测试用例（TCS）生成。

**📊 数据集**

使用了 50 篇来自华为 5 系列以太网交换机的典型配置手册（ESCM），涵盖 18 个子类别，保证实验覆盖面。

**📈 对比分析**

实验对比显示：使用原始提示时三项提取任务的正确率均在 0.88–1.00 之间，平均 0.97‑0.99；EEI 循环后错误率显著下降；LLM‑as‑Judge 与人工评估达 Cohen’s κ ≥ 0.72；基于生成 KG 的 TCS 在专家问卷中获得 4–5 级的高效能评价。

**⚠️ 局限性**

局限性包括：①对特殊文档格式与术语仍需手工微调提示；②评估与改进主要依赖 LLM，可能出现 hallucination；③实验规模受预算限制，未覆盖更广泛的技术领域；④框架对资源密集型 LLM 调用成本较高。

---

## 189. StampFormer: A Physics-Guided Material-Geometry-Coupled Multimodal Model for Rapid Prediction of Physical Fields in Sheet Metal Stamping

**arXiv ID:** 2605.18835 | [PDF](https://arxiv.org/pdf/2605.18835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 190. On the Geometric Limits of Transformer Defenses against Obfuscation Attacks: Latent Embedding Collapse & Performance Robustness Gap

**arXiv ID:** 2605.19159 | [PDF](https://arxiv.org/pdf/2605.19159v1)

**作者:** Becky Mashaido `[一作]` (University of Pacific), Tapadhir Das `[通讯]` (University of Pacific)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多操作符混淆攻击下Transformer（BERT系列）防御的几何稳健性，提出并量化了“潜在嵌入崩塌”现象；

**💡 创新点**

首次在LLM防御中引入几何度量（如清晰-混淆距离、内部方差等），揭示高分类性能与嵌入空间脆弱性的性能-稳健性间隙；

**🔧 技术方法**

使用多操作符混淆生成、BERT系列模型微调、最终层嵌入提取、欧氏距离计算、PCA与t‑SNE可视化等技术；

**📊 数据集**

构造约4万条合成样本（10k/类），包括clean、prefix、suffix、obfuscated 四类；

**📈 对比分析**

通过准确率、精确率、召回率、宏F1≈99%进行分类性能评估，同时用δ≈1.02、内部方差↑等几何指标表明高性能伴随嵌入崩塌；

**⚠️ 局限性**

仅在BERT家族上实验，未检验更大规模或不同架构；混淆方法可能不涵盖所有现实攻击；几何指标基于欧氏距离，可能忽略语义拓扑差异。

---

## 191. SPHERICAL KV: Angle-Domain Attention and Rate-Distortion Retention for Efficient Long-Context Inference

**arXiv ID:** 2605.18856 | [PDF](https://arxiv.org/pdf/2605.18856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 192. From Llama to Cria: Scaling Down Neural Networks via Neuron-Level Spectral Structural Importance Evaluation

**arXiv ID:** 2605.18860 | [PDF](https://arxiv.org/pdf/2605.18860v1)

**作者:** Yongyu Wang `[一作]` (Michigan Technological University), Yongyu Wang `[通讯]` (Michigan Technological University)

**通讯引用:** 3827 | [OpenAlex ID](https://openalex.org/A5029475800)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于神经元级谱结构重要性评估的目标剪枝框架，先将神经元视为图节点，利用隐藏状态构建输入/输出图，计算谱结构重要性分数，按低分数剪枝并仅在最终阶段进行恢复微调；

**💡 创新点**

创新点在于将图信号处理与谱图理论引入神经元剪枝，通过评估层间结构关系的变化而非仅靠参数大小或激活统计来衡量神经元重要性；

**🔧 技术方法**

使用图信号处理、谱图理论、Moore–Penrose伪逆、SPADE思想、神经网络训练与微调（LoRA）等技术；

**📊 数据集**

实验使用CIFAR-10图像分类（AlexNet-CIFAR）和SST-2情感分类（TinyLlama-1.1B-Chat）两个数据集；

**📈 对比分析**

通过与原始模型在相同训练轮次下的性能对比，发现低SPADE剪枝可在不做中间微调的情况下实现高达80%的参数压缩，恢复微调后仅损失≈0.5个百分点准确率；在TinyLlama上物理剪枝至30%参数压缩可保持≈92%准确率；

**⚠️ 局限性**

局限在于对极端压缩（>80%）仍可能导致显著准确率下降，且对大规模 Transformer 结构的可扩展性需进一步验证；

---

## 193. Robust Restless Multi-Armed Bandit for Data Center Flexibility Services Through Virtual Machine Scheduling

**arXiv ID:** 2605.19116 | [PDF](https://arxiv.org/pdf/2605.19116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 194. Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling

**arXiv ID:** 2605.18838 | [PDF](https://arxiv.org/pdf/2605.18838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 195. First-Passage Prediction of Grokking Delay: ACalibrated Law under AdamW with Causal Validation

**arXiv ID:** 2605.18845 | [PDF](https://arxiv.org/pdf/2605.18845v1)

**作者:** Truong Xuan Khanh `[一作]` (Clevix LLC), Phan Thanh Duc `[通讯]` (Banking Academy of Vietnam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

在AdamW优化器下研究grokking延迟，提出闭式第一通量预测公式并验证其机制

**💡 创新点**

将grokking建模为参数范数与角度的联合跨越问题，给出闭式定律并通过因果干预证明机制

**🔧 技术方法**

采用AdamW优化、参数范数与角度分析、NTK线性化、实验量化、因果Block F干预及MAPE评估

**📊 数据集**

使用模块加法、乘法和稀疏奇偶性等算术任务，并在多种网络架构上进行实验

**📈 对比分析**

以MAPE作为评估指标，单细胞预测误差17.7%，跨架构18.0%，跨任务23.3%，显示优于经验做法

**⚠️ 局限性**

κ_LL需经验估计，无法从AdamW超参数理论推导；仅适用于AdamW/权重衰减场景；角阈值α*基于线性化近似，未覆盖所有任务

---

## 196. Worst-Group Equalized Odds Regularization for Multi-Attribute Fair Medical Image Classification

**arXiv ID:** 2605.19214 | [PDF](https://arxiv.org/pdf/2605.19214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 197. PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines

**arXiv ID:** 2605.18812 | [PDF](https://arxiv.org/pdf/2605.18812v1)

**作者:** Varun Kotte `[一作]` `[通讯]` (Independent Researcher), Varun Kotte (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多阶段NLP/LLM管线，提出Pipeline-Aware Split Conformal（PASC）方法，实现端到端无条件分布无关的覆盖保证。

**💡 创新点**

创新点在于把所有阶段的联合覆盖事件化简为单一标量“最大非一致性分数”，从而仅需一次分位数计算即可得到正式的联合覆盖保证，显著优于Bonferroni保守估计。

**🔧 技术方法**

采用Split Conformal Prediction、最大非一致性分数聚合、单阈值校准，以及在不同管线阶段（NER、NED、Entity Typing）定义的非一致性分数。

**📊 数据集**

主要使用CoNLL‑2003、WNUT‑17（Twitter NER）和WikiNEuRal（维基百科）三大数据集进行评估；此外在K=1~6的合成扩展实验中检验可扩展性。

**📈 对比分析**

与独立CP、Bonferroni、调优Bonferroni以及MC Dropout等基线对比。PASC在α=0.1时端到端覆盖率达96.4%（>Bonferroni 3pp、>独立CP 10pp），保持相同的平均预测集大小；在分布迁移下仍保持≥1-α覆盖；在K=6时仍保持高覆盖率，而独立CP仅53%。

**⚠️ 局限性**

局限在于仍需满足校准集与测试集可交换性，对极高α（如0.2）时Bonferroni可能更优；未提供条件覆盖保证；对非可交换或严重分布漂移的理论保障仍待改进。

---

## 198. Backdooring Masked Diffusion Language Models

**arXiv ID:** 2605.19262 | [PDF](https://arxiv.org/pdf/2605.19262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 199. Modeling the Impact of Fiber Latency on Compute-Communication Overlap in Geo-Distributed Multi-Datacenter AI Training

**arXiv ID:** 2605.19169 | [PDF](https://arxiv.org/pdf/2605.19169v1)

**作者:** Ioannis Papavasileiou `[一作]` (Corning Inc.), Sergejs Makovejs `[通讯]` (Corning Inc.)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

利用离散事件模拟评估光纤时延对多数据中心AI训练中计算-通信重叠的影响；

**💡 创新点**

首次系统性量化不同距离下光纤时延对重叠的降解，并证明空心光纤在10-100km范围内可提升约25%重叠、最多可扩展50%距离；

**🔧 技术方法**

采用ASTRA-sim离散事件模拟、计算-通信重叠度量、GPT‑3 13B/175B模型、数据并行、SMF与HCF光纤、不同GPU（A100/H100）与集群规模；

**📊 数据集**

未使用真实训练数据集，利用理论峰值GPU性能估算计算时间；

**📈 对比分析**

在不同距离、光纤类型、GPU型号和集群规模下进行模拟，对比重叠度与训练时间倍增；结果显示HCF在10–100km内可提升25%重叠，H100在远距离时训练时间倍增可从26×降至17×；

**⚠️ 局限性**

仅考虑数据并行、单层同步且使用汇总DC抽象，未包含混合并行、Mixture‑of‑Experts、梯度分桶或推理工作负载；结果基于理论估算，缺乏实测验证。

---

## 200. MO-CAPO: Multi-Objective Cost-Aware Prompt Optimization

**arXiv ID:** 2605.18869 | [PDF](https://arxiv.org/pdf/2605.18869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 201. LWGR: Lagrangian-Constrained Personalized World Knowledge for Generative Recommendation

**arXiv ID:** 2605.18771 | [PDF](https://arxiv.org/pdf/2605.18771v1)

**作者:** Lingyu Mu `[一作]` (Chinese Academy of Sciences), Jinxin Hu `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了一种在生成式推荐中利用拉格朗日约束个性化世界知识的框架 LWGR，改进了知识提取和融合流程。

**💡 创新点**

创新点包括：1) 使用并行码本生成软指令，实现可扩展的个性化知识提取；2) 将知识融合建模为带约束的优化问题并采用拉格朗日主从法自适应控制性能下降；3) 设计了两种 LLM 训练策略（冻结与 LoRA 微调）和近线+轻量在线部署方案。

**🔧 技术方法**

技术手段：软提示 + 并行码本 + Index Backpropagation Quantization (IBQ) 端到端学习；交叉注意力融合；拉格朗日不等式约束主从优化；LoRA 微调或冻结 LLM；Transformer 解码器；Qwen3‑4B LLM。

**📊 数据集**

数据集：Amazon Beauty、Amazon Toys 公共数据（22k/35k 用户）以及一大规模工业电商日志（约 29.5 亿交互、1470 万用户）。

**📈 对比分析**

对比方法：SASRec、PinnerFormer、HeterRec、VQ‑Rec、TIGER、Cobra 以及基于提示的融合方法 TIGER+KAR、TIGER+SeRALM。LWGR 在 Recall@5/10、NDCG@5/10 上分别提升约 9–11%，工业 A/B 测试实现 1.35% 广告收入提升、1.17% CTR 提升。

**⚠️ 局限性**

局限性：依赖大规模 LLM 与码本，硬件与推理延迟有一定开销；超参（码本维度、温度、阈值）对效果敏感；当前实现依赖近线预计算 + 在线检索，实时 LLM 推理仍受限。

---

## 202. Structuring Open-Ended NAS: Semi-Automated Design Knowledge Structuring with LLMs for Efficient Neural Architecture Search

**arXiv ID:** 2605.19247 | [PDF](https://arxiv.org/pdf/2605.19247v1)

**作者:** Yuiko Sakuma `[一作]` (Sony Group Corporation), Takeshi Ohashi `[通讯]` (Sony Group Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于半自动化结构化模型设计属性的知识库构建和多类型突变的公平 NAS 框架（FairNAD），用于高效搜索开辟式架构。

**💡 创新点**

创新点在于将模型设计知识结构化为层级属性树，并结合公平采样、Pareto 识别、LLM 迭代突变与反馈循环，显著提升搜索多样性与效率。

**🔧 技术方法**

采用 LLM（Qwen3-8B）提取与整理设计思路，执行多目标进化突变，结合预算与执行验证器的反馈循环。

**📊 数据集**

在 NAS-Bench-201 的 CIFAR-10、CIFAR-100 与 ImageNet16-120 三个数据集上进行实验。

**📈 对比分析**

与手工搜索和现有 LLM 基础 NAS 方法相比，FairNAD 在三组数据集分别提升 0.84、2.17、2.35 个百分点的测试准确率，达成当前 SOTA。

**⚠️ 局限性**

局限在于对 LLM 生成思路的依赖、搜索空间仍受预设属性树影响，以及实验主要集中在小规模基准上，尚未验证在更大真实任务上的可迁移性。

---

## 203. Sequential Consensus for Multi-Agent LLM Debates: A Wald-SPRT compute governor with calibration-based failure detection

**arXiv ID:** 2605.19193 | [PDF](https://arxiv.org/pdf/2605.19193v1)

**作者:** Andrea Morandi `[一作]` `[通讯]` (Cisco Systems, Inc.), Andrea Morandi (Cisco Systems, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Wald 的顺序概率比率检验（SPRT）的多智能体 LLM 辩论停止规则，利用判定者给出的 0~1 的共识分数作为观测，通过阈值判定是否停止辩论，以实现计算成本的自适应控制。

**💡 创新点**

创新点包括：
- 将经典的 SPRT 直接应用到 LLM 辩论的共识分数；
- 通过对每个任务域进行 Beta 分布校准（f₀、f₁），实现判定者信号与正确/错误收敛的区分；
- 在规则中加入硬上限 R_max，既保证误判率，又能在判定者无信息时做出“无共识”或“最佳努力”输出；
- 将此规则视为一种廉价的计算治理层和失败检测器，既可与任意多智能体辩论框架无缝集成，又能在判定者有判别力时显著降低调用次数。

**🔧 技术方法**

技术手段：
- Wald 的顺序概率比率检验（SPRT）；
- 以 Beta 分布为观测似然模型并通过矩匹配进行校准；
- 通过蒙特卡洛仿真评估工作曲线、误差率、阈值覆盖率等；
- 真实 LLM 评估（gpt‑5、claude‑opus‑4‑6、gemini‑2.5‑pro）和判定者 claude‑opus‑4‑6；
- 对非 i.i.d. 影响的 AR(1) 相关性实验与误差率鲁棒性分析。

**📊 数据集**

使用数据集：
- MMLU（多项选择题）；
- GSM8K（算数推理题）；
- JudgeBench（偏好判断任务，实验未完成）；
- 以及针对每个任务域的 40–100 条离散校准样本。

**📈 对比分析**

对比方法：单轮多数投票（B1）、固定 5 轮辩论（B2）和本规则（B3）。
- 在模拟中，B3 在三种校准下平均轮数从 5 下降到 1.5–2.4，误差率保持 ≤ 5%。
- 在真实 LLM 评估：
  * GSM8K：B3 仅 1.01 轮、4.06 次调用、97.0% 准确率，较 B2 的 15 调用、99.0% 准确率降成本 3.7 倍，准确率下降 2 pp；
  * MMLU：B3 99.5% 的案例被硬上限 8 轮截断，调用数 32 次、94.2% 准确率，略高于 B2 的 15 调用、93.8% 准确率；
- 综上，B3 在判定者信号强时能显著降低计算成本；当判定者无区分度时（如 MMLU），规则会自动进入上限并保持与固定轮数相近的准确率。

**⚠️ 局限性**

局限性：
- 依赖判定者分数与正确性之间的统计区分度；若判定者无法区分（KL ≈ 0），规则会频繁触发上限，导致成本升高；
- 假设每轮分数 i.i.d.，实际存在相关性，虽然实验表明误差率仍在可接受范围；
- 需要每个任务域进行至少 50–100 条校准样本，且需周期性复校；
- 对抗性智能体可能通过制造高共识分数而误导规则；
- 固定 Beta 似然模型可能不足以捕捉随轮次变化的分布，需要更灵活的适配机制；
- 规则仅决定停止时机，不能提升单轮回答质量，仍需配合高质量的辩论策略。

---

## 204. Devilray: A Systematic Adversarial Model Revealing Blind Spots in Fake Base Station Detection

**arXiv ID:** 2605.19232 | [PDF](https://arxiv.org/pdf/2605.19232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 205. Prediction Is Not Physics: Learning and Evaluating Conserved Quantities in Neural Simulators

**arXiv ID:** 2605.18883 | [PDF](https://arxiv.org/pdf/2605.18883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 206. Aerial Inspection Behaviors via RL-based Quadrotor Control for Under-canopy Forest Environments

**arXiv ID:** 2605.19202 | [PDF](https://arxiv.org/pdf/2605.19202v1)

**作者:** Fausto Mauricio Lagos Suarez `[一作]` (Luleå University of Technology), George Nikolakopoulos `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文设计并验证了一个端到端的深度强化学习低层四旋翼控制器，直接将观测映射到电机 RPM，用于在森林底层执行检查任务，并与 TSP+RRT* 高层规划堆栈集成。

**💡 创新点**

创新点在于：① 将学习到的电机级控制策略与高层路径规划耦合，实现同时位置与偏航的精准跟踪；② 通过奖励设计和噪声注入，使模型在复杂环境下保持鲁棒性；③ 将传统 PID/几何控制的多级结构简化为单一端到端策略。

**🔧 技术方法**

主要技术包括：深度强化学习（Stable‑Baselines3、PPO/SAC），多层感知网络（MLP），TSP 最优目标序列规划，3D RRT* 障碍规避，PyBullet 物理仿真，ROS2 交互，Gaussian 噪声注入与多项式奖励设计。

**📊 数据集**

使用的是在仿真中随机生成的森林环境数据集：随机放置 200 根半径 0.15–0.25 m、2 m 高的圆柱树木，20 m×20 m 区域内 8 个检查目标；未采用公开数据集，全部场景由作者自行构造。

**📈 对比分析**

在五个典型检查任务（点对点、视角跟踪、场景扫描、圆周巡检、螺旋巡检）中，控制器实现了 <5 cm 的平移误差、<3° 的偏航误差，轨迹跟踪成功率达到 100%，并在仿真中展示了与传统 PID/几何控制相当甚至更好的稳健性；但论文未给出与其它强化学习或经典控制器的定量对比实验。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏 sim‑to‑real 迁移与真实硬件测试；对动态障碍物或感知误差不具备鲁棒性；依赖已知地图和 GPS/IMU 导航，对地图误差或失效的容忍度不足。

---

## 207. Multi-Token Residual Prediction

**arXiv ID:** 2605.18817 | [PDF](https://arxiv.org/pdf/2605.18817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 208. Knowing When Not to Predict: Self Supervised Learning and Abstention for Safer DR Screening

**arXiv ID:** 2605.19133 | [PDF](https://arxiv.org/pdf/2605.19133v1)

**作者:** Muskaan Chopra `[一作]` (Rheinische Friedrich-Wilhelms-Universität Bonn), Rafet Sifa `[通讯]` (Rheinische Friedrich-Wilhelms-Universität Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究自监督预训练时长对糖尿病视网膜病变筛查模型的置信度校准和选择性预测（confidence‑based abstention）的影响。

**💡 创新点**

提出预训练长度是影响可靠性的重要设计因素，系统评估其对置信度分布、覆盖率‑风险曲线的非单调影响，指出准确率饱和后仍可出现可靠性波动。

**🔧 技术方法**

使用自监督学习（SiCoVa非对比损失和三元组对比损失）、温度缩放校准、置信度阈值拒绝，采用ResNet50和ViT‑B/16编码器。

**📊 数据集**

预训练在EyePACS未标记图像上完成；下游微调和评估在APTOS‑19、Messidor和7‑class Fundus数据集上进行。

**📈 对比分析**

与Triplet、监督基线及CAM改进的模型比较，报告准确率、宏F1、选择性宏F1；在70%覆盖率下SiCoVa的选择性宏F1显著优于基线，但预训练时间长并不总能提升可靠性。

**⚠️ 局限性**

局限包括仅使用单一SSL框架与有限架构、未进行多随机种子/显著性检验、仅采用简单置信度阈值拒绝、临床可解释性评估量化不足。

---

## 209. Rotation-Aligned Key Channel Pruning for Efficient Vision-Language Model Inference

**arXiv ID:** 2605.19218 | [PDF](https://arxiv.org/pdf/2605.19218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 210. Automated Big Data Quality Assessment using Knowledge Graph Embeddings

**arXiv ID:** 2605.18833 | [PDF](https://arxiv.org/pdf/2605.18833v1)

**作者:** Hadi Fadlallah `[一作]` (Saint Joseph University), Ali Jaber `[通讯]` (Lebanese University)

**通讯引用:** 557 | [OpenAlex ID](https://openalex.org/A5033859204)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于知识图谱嵌入的自动数据质量评估方法，利用知识图谱中的数据上下文与质量规则、维度关系自动生成针对新数据集的完整评估计划。

**💡 创新点**

创新点在于：①在知识图谱嵌入中注入数值边属性（FocusE），实现对边权重的预测；②采用开放世界假设（OWA）提升对未知关系的推理能力；③利用文献挖掘构建完整的数据上下文属性模型，减少传统精确匹配导致的属性缺失。

**🔧 技术方法**

技术包括：知识图谱嵌入（AmpliGraph + FocusE）、图神经网络/Node2Vec、KNN相似度检索、梯度下降优化（Adam）、pairwise margin loss 等。

**📊 数据集**

使用黎巴嫩原子能委员会提供的放射性传感器真实数据集，以及构建的40个数据上下文与对应质量评估计划的知识图谱。

**📈 对比分析**

与原有BIGQA上下文分析器对比：BIGQA仅检索最相似上下文的评估计划，导致部分属性缺失；新方法通过边权预测生成完整评估计划，Hits@10 0.66、Hits@3 0.26、Hits@1 0.21，表明在边预测上具有可接受的性能。

**⚠️ 局限性**

局限性包括：知识图谱规模小导致模型性能受限；嵌入模型可解释性差，难以阐释决策逻辑；训练过程计算量大，需大量GPU资源。

---

## 211. PRISM-SLAM: Probabilistic Ray-Grounded Inference for Scale-aware Metric SLAM

**arXiv ID:** 2605.19257 | [PDF](https://arxiv.org/pdf/2605.19257v1)

**作者:** Eunsoo Im `[一作]` `[通讯]`, Eunsoo Im

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PRISM‑SLAM，实现实时单目SLAM的尺度感知与度量一致轨迹，融合视觉基础模型的深度先验与贝叶斯因子图。

**💡 创新点**

创新点包括：Plücker射线‑距离因子实现尺度可识别；动态场不确定性门控（DSUG）软门控去除动态遮挡；log‑域Kalman滤波和WLS实现多进程尺度恢复；ViT驱动闭环与全局度量BA。

**🔧 技术方法**

技术手段包括：DA3视觉基础模型、Plücker坐标约束、DSUG不确定性门控、log‑域Kalman滤波、加权最小二乘（WLS）尺度融合、ViT特征闭环、异步多进程架构。

**📊 数据集**

使用的数据集：TUM RGB‑D、7‑Scenes、BONN Dynamic。

**📈 对比分析**

与ORB‑SLAM3、DROID‑SLAM、DPV‑SLAM++等基线对比，PRISM‑SLAM在30FPS实时运行，静态序列SE(3) ATE ≤2 cm，动态序列显著优于基线，7‑Scenes平均Sim(3) ATE 8.8 cm。

**⚠️ 局限性**

局限性：对GPU依赖高，纹理贫乏场景下帧率下降；极端遮挡或大规模动态物体时尺度恢复仍可能受损；密集重建对动态物体敏感。

---

## 212. EgoTraj: Real-World Egocentric Human Trajectory Dataset for Multimodal Prediction

**arXiv ID:** 2605.19004 | [PDF](https://arxiv.org/pdf/2605.19004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 213. Bridge: Retrieval-Augmented Spatiotemporal Modeling for Urban Delivery Demand

**arXiv ID:** 2605.19172 | [PDF](https://arxiv.org/pdf/2605.19172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 214. The fitness landscape of social norms in social dilemmas

**arXiv ID:** 2605.18834 | [PDF](https://arxiv.org/pdf/2605.18834v1)

**作者:** Maximilian Puelma Touzel `[一作]` `[通讯]` (Mila - Quebec AI Institute), Maximilian Puelma Touzel (Mila - Quebec AI Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了将社会规范视为协同均衡的框架，并在马尔可夫博弈环境下通过矩阵表示形式定义规范、描述、奖励等要素，利用复制者动力学分析规范演化。

**💡 创新点**

创新点在于：1）将传统协同均衡概念与社会规范结合，构建可计算的规范收益矩阵；2）把规范问题映射到马尔可夫博弈，提供统一的数学表述；3）通过对鸡肉游戏的参数化分析得到规范有效性相位图和复制者动力学的稳定性判据。

**🔧 技术方法**

主要技术包括：演化博弈理论、马尔可夫博弈框架、矩阵代数求期望与协方差、复制者动力学（Generalized Lotka–Volterra）及其线性稳定性分析、数值仿真。

**📊 数据集**

没有使用实际数据集，全部采用理论模型与数值实验进行验证。

**📈 对比分析**

与传统的Nash均衡方法对比，本文通过理论推导表明在社会规范演化过程中，协同均衡可实现比混合Nash更高的平均回报；但文中并未给出实验性性能数值，只给出了理论收益函数和相位图。

**⚠️ 局限性**

局限性包括：1）仅研究对称双人二行动博弈（如鸡肉游戏），难以直接推广到更大规模、多行动或不对称环境；2）假设观察信号完全可分辨且无外部噪声；3）缺乏对真实多智能体系统中的实验验证；4）复制者动力学基于理想化的随机配对与无记忆复制，可能无法捕捉更复杂的社会学习机制。

---

## 215. Towards Family-Grouped Hierarchical Federated Learning on Sub-5KB Models: A Feasibility Study of Privacy-Preserving ECG Monitoring for Ultra-Resource-Constrained Wearables

**arXiv ID:** 2605.18862 | [PDF](https://arxiv.org/pdf/2605.18862v1)

**作者:** Hangyu Wu `[一作]` `[通讯]` (Shenzhen Coddie Technology co.,ltd), Hangyu Wu (Shenzhen Coddie Technology co.,ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了基于家庭聚合的三层联邦学习架构，配合极小参数Tiny CNN‑LSTM实现心律失常监测。

**💡 创新点**

创新点在于将家庭作为自然隐私边界实现分层聚合，并将模型压缩至子5KB，满足STC32微控制器的极限资源。

**🔧 技术方法**

采用联邦学习、分层聚合、INT8后训练量化、CNN‑LSTM轻量网络、FedAvg/FedProx等技术。

**📊 数据集**

使用MIT‑BIH心律失常数据库（47人）。

**📈 对比分析**

与FedAvg、FedProx、全尺寸模型以及FedAvg‑Tiny对比，Family‑FL‑Tiny达91.9%准确率、宏F1 0.483，通信量仅为FedAvg的0.31%，比FedAvg‑Tiny通信降低50%。

**⚠️ 局限性**

局限包括仅在单一数据集仿真验证、极小模型对罕见类别表现差、未实现硬件部署、缺乏差分隐私保障以及样本不均衡问题。

---

## 216. SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs

**arXiv ID:** 2605.18864 | [PDF](https://arxiv.org/pdf/2605.18864v1)

**作者:** Chanuk Lee `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过改造逆KL正则化锚点的SAGE框架，在RLVR中实现受控探索以提升LLM多步推理的多样性与准确率。

**💡 创新点**

将逆KL视为可塑造的锚点而非仅惩罚，设计guide函数q(x,y)重塑锚分布，在保持训练稳定性的同时实现经验支持扩展。

**🔧 技术方法**

采用逆KL RLVR、伪KL正则化，结合熵与惊奇等intrinsic信号的guide函数（随机、token、branch），并与PPO/GRPO等RLVR算法集成。

**📊 数据集**

使用AIME 2024-25、AMC23、MATH‑500级别5数据集进行主实验，并在Knights & Knaves逻辑谜题上做OOD评估。

**📈 对比分析**

相较于Baseline、GRPO、GRPO无KL、Forward KL等基线，SAGE+Branch在pass@1和pass@k上平均提升约0.28与0.73，且在多种算法与长序列模型上均实现准确率与覆盖率双提升。

**⚠️ 局限性**

只能在参考策略已有支持内扩展，无法产生全新轨迹；guide函数仍基于浅层intrinsic信号，缺乏更丰富的学习式设计。

---

## 217. Fast and Lightweight Backdoor Detection via Head Random Probing

**arXiv ID:** 2605.18908 | [PDF](https://arxiv.org/pdf/2605.18908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 218. ReCrit: Transition-Aware Reinforcement Learning for Scientific Critic Reasoning

**arXiv ID:** 2605.18799 | [PDF](https://arxiv.org/pdf/2605.18799v1)

**作者:** Wanghan Xu `[一作]` (Shanghai Jiao Tong University), Lei Bai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 4077 | [OpenAlex ID](https://openalex.org/A5028486493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向科学批评交互的ReCrit框架，专注于模型在收到批评后如何在回答中进行合理修正或保持正确性。

**💡 创新点**

创新点在于将批评交互视为“初始到批评”准确率转变问题，并将转变分为四个象限（修正、顺从、稳健、边界），通过四象限奖励实现纠错与避免无意义改答的解耦。

**🔧 技术方法**

技术包括基于强化学习的转移奖励、四象限权重设定、动态异步推理与尾部自适应完成、以及监督式最终化模块和格式预热。

**📊 数据集**

使用了三种封闭式科学推理基准：ChemBench、TRQA、EarthSE。

**📈 对比分析**

与基线（Base、SFT、DPO、GRPO、Critique-GRPO）对比，ReCrit在Qwen3.5-4B与9B模型上显著提升批评后准确率（如4B模型从38.15%提升至51.49%，9B模型从45.40%提升至55.59%），并在修正率上优于顺从率，整体性能最好。

**⚠️ 局限性**

局限性在于仅针对封闭式科学问答场景，开放式任务、复杂人类反馈或更长批评链需要更强的判别器和更复杂的奖励设计。

---

## 219. PMF-CL: Pareto-Minimal-Forgetting Continual Learner for Conflicting Tasks

**arXiv ID:** 2605.19145 | [PDF](https://arxiv.org/pdf/2605.19145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 220. Deep Neural Sheaf Diffusion

**arXiv ID:** 2605.19021 | [PDF](https://arxiv.org/pdf/2605.19021v1)

**作者:** Remi Bourgerie `[一作]`, Viktoria Fodor `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1012 | [OpenAlex ID](https://openalex.org/A5075982238)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出深度神经单元扩散（DNSD）模型，利用单元邻接算子、层归一化、奇数激活和门控机制，解决传统 NSD 在深层时信号消失和过度平滑问题。

**💡 创新点**

创新点在于将单元拉普拉斯替换为邻接算子以保持信息流，结合层归一化、tanh 激活和节点级门控，显著提升深层图网络的表现并稳定训练。

**🔧 技术方法**

采用细胞单元理论、单元邻接卷积、层归一化、奇数（tanh）激活、节点门控，并与 GAT、MPNN、MLP 等基线进行对比。

**📊 数据集**

实验使用合成长程社区检测数据集（G0–G10）以及九个异构性真实图数据集（如 Roman Empire、AmazonRatings、Minesweeper、Tolokers、Questions、Penn94 等）。

**📈 对比分析**

在不同层数（2–16）和基线模型的比较中，DNSD 在中等异构性下比 NSD 提升 20–30pp，深层时准确率提升 30% 以上，合成数据最高 97% 以上；在真实数据上提升 3–4pp。

**⚠️ 局限性**

局限性包括仅在节点分类任务、介于中等规模图上验证；未覆盖图级任务、链路预测或大规模图；缺乏专门探测深度的基准；单元映射计算开销较大。

---

## 221. GRASP: Deterministic argument ranking in interaction graphs

**arXiv ID:** 2605.19141 | [PDF](https://arxiv.org/pdf/2605.19141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 222. A Multi-Dimensional Clustering Approach for Identifying Inborn Errors of Immunity

**arXiv ID:** 2605.18880 | [PDF](https://arxiv.org/pdf/2605.18880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 223. VCR: Learning Valid Contextual Representation for Incomplete Wearable Signals

**arXiv ID:** 2605.18837 | [PDF](https://arxiv.org/pdf/2605.18837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 224. On-Device Continual Learning with Dual-Stage Buffer and Dynamic Loss for Point-of-Care Pneumonia Diagnosis

**arXiv ID:** 2605.19201 | [PDF](https://arxiv.org/pdf/2605.19201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 225. Theory-optimal Quantization Based on Flatness

**arXiv ID:** 2605.18800 | [PDF](https://arxiv.org/pdf/2605.18800v1)

**作者:** Xiusheng Huang `[一作]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems Institute of Automation Chinese Academy of Sciences), Kang Liu `[通讯]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的后训练量化方法Bidirectional Diagonal Quantization（BDQ），通过双向对角变换有效分散激活与权重中的异常值并避免过拟合；

**💡 创新点**

创新点在于：① 通过数学分析建立异常值与量化误差的二次关系；② 引入Flatness指标量化异常分布并推导出对角变换为最优解；③ 结合Hadamard旋转与递归交叉熵损失，进一步提升鲁棒性；

**🔧 技术方法**

使用的技术包括：对角矩阵学习、Hadamard正交变换、递归交叉熵（RCE）正则化、GPTQ量化框架以及对比实验中的量化工具（QuaRot、SpinQuant、FlatQuant）；

**📊 数据集**

使用的主要数据集有：WikiText2、C4、ARC‑Easy、ARC‑Challenge、HellaSwag、LAMBADA、PIQA、Winogrande，且通过lm‑eval‑harness进行零样本评估；

**📈 对比分析**

在多种模型（LLaMA‑3 8B/70B、DeepSeek‑R1‑Distill、LLaMA‑2 7B/70B）和量化配置（W4A4KV4、W3A3KV3、W2A4KV16）下，BDQ相较于SOTA FlatQuant等方法在PPL、零样本QA准确率上提升约1–4%，在W2A4KV16上对70B模型降低约39%性能差距；

**⚠️ 局限性**

局限性包括：未在更大规模模型上验证；缺少不同GPU平台的广泛可行性实验；

---

## 226. MultiBallot: Verifiable and privacy-preserving E-Collecting in the Swiss setting

**arXiv ID:** 2605.19312 | [PDF](https://arxiv.org/pdf/2605.19312v1)

**作者:** Florian Moser `[一作]` (famoser GmbH), Léo Louistisserand `[通讯]` (Université de Lorraine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 MultiBallot 协议，旨在实现瑞士电子签名收集系统的可验证性与参与隐私保护，并通过并行收集的并行性隐藏投票行为；

**💡 创新点**

创新点在于利用多重活跃收集并行的特点，借助零知识证明实现不需要匿名通道的参与隐私；同时支持连续运行、资格变更、密钥轮换及混合纸质/电子参与；

**🔧 技术方法**

采用了公钥加密（ElGamal）与同态求和、零知识证明、分布式密钥生成与解密、可验证公告板以及电子身份认证等标准密码工具；

**📊 数据集**

论文未给出具体实验数据集，主要是协议设计与理论分析；

**📈 对比分析**

作者仅在理论层面说明与现有电子投票方案性能相当，未提供实验对比；性能依赖于加密与零知识证明的效率，预期可与成熟投票系统匹配；

**⚠️ 局限性**

局限性包括对公告板、部分可信方的信任假设；在频繁计数时可能泄露参与模式；混合通道的可审计性要求额外信任；缺乏实证评估与安全证明细节。

---

## 227. Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality

**arXiv ID:** 2605.18971 | [PDF](https://arxiv.org/pdf/2605.18971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 228. GenAI-FDIA: Physics-Informed Generative Models for False Data Injection Attacks

**arXiv ID:** 2605.18873 | [PDF](https://arxiv.org/pdf/2605.18873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 229. Flash PD-SSM: Memory-Optimized Structured Sparse State-Space Models

**arXiv ID:** 2605.19150 | [PDF](https://arxiv.org/pdf/2605.19150v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 230. An Objective Performance Evaluation of the LSTM Networks in Time Series Classification

**arXiv ID:** 2605.19311 | [PDF](https://arxiv.org/pdf/2605.19311v1)

**作者:** Sooraj Sunil `[一作]` (University of Windsor), Balakumar Balasingam `[通讯]` (University of Windsor)

**通讯引用:** 2082 | [OpenAlex ID](https://openalex.org/A5101751073)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个客观的性能评估框架，用于比较基于EM算法的模型方法与基于LSTM的深度学习方法在二分类时间序列辨识中的表现；

**💡 创新点**

创新点在于提出使用已知参数的Kalman滤波LRT作为理论最优基准，并在受控仿真条件下对不同任务难度、序列长度和训练样本规模进行系统化比较；

**🔧 技术方法**

采用的技术包括EM算法估计线性高斯状态空间模型参数、Kalman滤波的似然比检验以及16维隐藏单元的LSTM网络；

**📊 数据集**

使用的“数据集”为从两种相同结构但噪声参数不同的标量线性高斯状态空间模型中合成的观测序列；

**📈 对比分析**

比较方法是对EM+Kalman LRT、纯LSTM和真参数Kalman LRT三者在100次蒙特卡罗实验中测量准确率，结果显示EM在模型符合时几乎达到上界，而LSTM需更大噪声分离且性能低于EM；

**⚠️ 局限性**

局限性包括实验只涉及单变量线性高斯模型，未考察模型不匹配、非高斯噪声或多变量情况，且LSTM结构简单，可能低估其潜力。

---

## 231. Towards Data-Efficient Video Pre-training with Frozen Image Foundation Models

**arXiv ID:** 2605.19137 | [PDF](https://arxiv.org/pdf/2605.19137v1)

**作者:** Svetlana Orlova `[一作]` (Eindhoven University of Technology), Gijs Dubbelman `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 958 | [OpenAlex ID](https://openalex.org/A5024010116)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用冻结的图像基础模型作为空间编码器，只训练一个递归时序模块来实现视频理解。

**💡 创新点**

证明空间编码器可从图像预训练获得，显著降低视频预训练的数据与计算成本，并展示时序模块可独立训练。

**🔧 技术方法**

采用 DINOv3 等图像 ViT 作为冻结编码器，结合 RVMRNN、Mamba、MambaMix、GatedMambaMix 等递归时序架构。

**📊 数据集**

在动作识别（Something‑Something v2）、物体跟踪（Waymo Open）、点跟踪（Perception Test）、深度估计（ScanNet）和摄像机姿态估计（NuScenes）等多任务数据集上进行实验。

**📈 对比分析**

与 RVM、VideoMAE、V-JEPA、4DS 等端到端视频预训练模型对比，冻结图像编码器+轻量时序模块在多任务上能匹配或超过传统模型，且在 SSv2 上仅使用 25% 数据即可超越全量 RVM。

**⚠️ 局限性**

未完成对时序模块的完整视频预训练，实验仅限于从零开始训练；后续需验证更大规模模型、不同编码器与预训练目标的通用性。

---

## 232. Geo-Data-Driven HD Map Generation Workflow with Integrated Reference-Free Constraint-Based Verification

**arXiv ID:** 2605.18921 | [PDF](https://arxiv.org/pdf/2605.18921v1)

**作者:** Ruidi He `[一作]` (Technical University of Clausthal), Andreas Rausch `[通讯]` (Technical University of Clausthal)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

基于公开的地理工程数据，设计了一个模块化的高精度地图生成与验证工作流

**💡 创新点**

创新在于将可执行的约束验证嵌入生成流程，无需外部参考地图

**🔧 技术方法**

使用了道路中心线解析、车道生成、OpenDRIVE 到 CommonRoad 转换以及高阶逻辑约束验证

**📊 数据集**

利用了德国下萨克森州公开的 Basis‑DLM 与 DGM1 等地理数据集

**📈 对比分析**

在真实案例中未发现违规，注入缺陷案例实现 100% 的精度与召回率

**⚠️ 局限性**

局限在于对属性完整度的依赖、约束集相对基础、未覆盖更复杂语义及不同地区的评估

---

## 233. CLIC: Contextual Language-Informed Cardiac Pathology Classification

**arXiv ID:** 2605.19132 | [PDF](https://arxiv.org/pdf/2605.19132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 234. Rewrite System Showdown: Stochastic Search vs. EqSat

**arXiv ID:** 2605.19005 | [PDF](https://arxiv.org/pdf/2605.19005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 235. Interference-Aware Multi-Task Unlearning

**arXiv ID:** 2605.19042 | [PDF](https://arxiv.org/pdf/2605.19042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 236. MoCo-EA: Exploiting Adversarial Mode Connectivity for Efficient Evolutionary Attacks

**arXiv ID:** 2605.18919 | [PDF](https://arxiv.org/pdf/2605.18919v1)

**作者:** Hyo Seo Kim `[一作]` (Illinois Institute of Technology), Ren Wang `[通讯]` (Illinois Institute of Technology)

**通讯引用:** 11337 | [OpenAlex ID](https://openalex.org/A5100339142)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MoCo-EA，一种利用贝塞尔曲线实现连续模式连接的进化攻击框架，用以替代传统离散交叉；

**💡 创新点**

核心创新在于发现对抗扰动存在模式连通性，并将贝塞尔曲线作为交叉算子，使得中间点比端点更具攻击成功率与可迁移性；

**🔧 技术方法**

主要技术包括贝塞尔曲线优化、对抗模式连通性分析、进化算法（种群、选择、变异）、PGD端点生成、多图像增强及梯度计算；

**📊 数据集**

实验数据集涵盖CIFAR-10和ImageNet，使用ResNet-18/ResNet-50与ViT-Base/16等模型；

**📈 对比分析**

与传统EA、PGD、MI-FGSM、AutoAttack、AAA等方法对比，MoCo-EA在成功率、所需代数、查询次数和运行时间上显著优于传统EA，并在鲁棒模型与梯度被遮蔽场景下达到或超过梯度攻击的效果；

**⚠️ 局限性**

局限性包括仍需梯度信息进行贝塞尔曲线优化、计算量相对传统交叉更高、对极大维度或更强防御的适用性尚未充分验证，以及在某些ε或范数下成功率仍有提升空间。

---

## 237. Adaptive Multi-Scale Goodness Aggregation for Forward-Forward Learning

**arXiv ID:** 2605.18804 | [PDF](https://arxiv.org/pdf/2605.18804v1)

**作者:** Salar Beigzad `[一作]` (University of St. Thomas), Vansh Verma `[通讯]` (University of St. Thomas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Adaptive Multi-Scale Goodness Aggregation (AMSGA) 方法，用以改进 Hinton 的 Forward–Forward（FF）算法，让其更稳定、更强大并能更好泛化。

**💡 创新点**

创新点在于：① 将 goodness 计算拆分为局部、介质和全局三尺度并按层深度加权；② 采用基于进度的 curriculum 负样本挖掘；③ 为每层设置随训练进展自适应阈值；④ 结合 warm‑up 与余弦退火学习率调度，消除早期不稳定。

**🔧 技术方法**

使用的技术包括：多尺度 goodness 计算、分层自适应阈值、按阶段的负样本选取策略、Adam 优化器、梯度裁剪、Kaiming 权重初始化、warm‑up + cosine annealing 学习率计划。

**📊 数据集**

在两大经典图像识别基准 MNIST 和 Fashion‑MNIST 上进行实验。

**📈 对比分析**

与基线 FF 比较：MNIST 上测试准确率提升 2.45%（94.45% vs 92.00%）；Fashion‑MNIST 上提升 3.5%（84.5% vs 81.0%）。相较于传统 backprop 训练的 MLP/CNN，AMSGA 仍有一定差距，但已显著缩小该 gap。

**⚠️ 局限性**

局限性：在更复杂的数据集上仍未赶上 backprop 训练的性能；对标签嵌入方式、goodness 与分类的关系及层间独立性的进一步改进仍有空间。

---

## 238. Symmetry in the Wild: The Role of Equivariance in Neural Fluid Surrogates

**arXiv ID:** 2605.18816 | [PDF](https://arxiv.org/pdf/2605.18816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 239. Smartphone-based Circular Plot Sampling for Forest Inventory

**arXiv ID:** 2605.19213 | [PDF](https://arxiv.org/pdf/2605.19213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. Descriptive versus Regulatory Uncertainty in Bounded Predictive Systems

**arXiv ID:** 2605.18909 | [PDF](https://arxiv.org/pdf/2605.18909v1)

**作者:** Ahmed Gamal Eldin `[一作]` `[通讯]`, Ahmed Gamal Eldin

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究大型语言模型在物理硬件上的热力学耦合度，证明其输出不确定性与知识质量无关。

**💡 创新点**

提出热力学解耦与Softmax解耦定理，并通过实验验证模型的不确定性不能反映其推理准确性。

**🔧 技术方法**

使用Token级Shannon熵计算、固定温度推理和人类评估准确度的方法，对Llama 3B/8B/70B进行实验。

**📊 数据集**

构造18个基于天体力学与牛顿力学的任务集，分为Kepler、Newton与OOD三类。

**📈 对比分析**

通过对比熵与准确度，发现熵在所有模型与任务类别内几乎不变，而准确度差异显著，表明解耦现象在规模上保持不变。

**⚠️ 局限性**

局限在样本量小（每类仅6题），仅评估Llama族模型，未检验其他模型或更大任务集。

---

## 241. Prompting language influences diagnostic reasoning and accuracy of large language models

**arXiv ID:** 2605.19173 | [PDF](https://arxiv.org/pdf/2605.19173v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 242. ClinQueryAgent: A Conversational Agent for Population Health Management

**arXiv ID:** 2605.18768 | [PDF](https://arxiv.org/pdf/2605.18768v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 243. GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning

**arXiv ID:** 2605.19235 | [PDF](https://arxiv.org/pdf/2605.19235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 244. SCAFDS: Edge-Feature Graph Attention for Interbank Fraud Detection with Attribution-Grounded SAR Generation

**arXiv ID:** 2605.18913 | [PDF](https://arxiv.org/pdf/2605.18913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 245. Hallucination as Exploit: Evidence-Carrying Multimodal Agents

**arXiv ID:** 2605.19192 | [PDF](https://arxiv.org/pdf/2605.19192v1)

**作者:** Guijia Zhang `[一作]` (Shenzhen University), Harry Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5020601958)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究针对多模态代理在执行工具调用时因幻觉导致授权失败的问题，提出了证据携带多模态代理（ECA）架构，要求所有行动前置谓词必须通过可信证据（DOM/OCR/AX）证明后才可授权执行。

**💡 创新点**

创新点在于①对“幻觉转行动”（H2AC）失效模式进行形式化；②引入证据携带机制，将模型生成的自由文本与权限授权分离；③利用跨模态验证和确定性门控实现可审计且高度可控的权限边界。

**🔧 技术方法**

核心技术包括多模态LLM规划、DOM/OCR/AX三条可信验证器、结构化谓词模式（action schemas）、确定性证据门控、红队攻击评估以及Oracle-证书回放对门控逻辑进行验证。

**📊 数据集**

实验使用了六大公开基准（AgentDojo、AgentDyn、DocVQA、SafeToolBench、VisualWebArena、VPI-Bench），共计7488个GPT‑5.4规划任务；另外构造了1900个红队攻击样本、200个E2E任务、120个浏览器PoC任务进行评估。

**📈 对比分析**

与无门控、仅提示防御、验证器单独、模型自证等基线相比，ECA在1900个红队攻击后门控UAR从15%降至1.3%；在200个E2E、120个浏览器任务中实现0% UAR；在Oracle证书回放也保持0%；提示防御的UAR在50‑86%之间；神经判别器UAR>90%，表明ECA在安全性上显著优于传统方法，并保持100%友善任务完成率。

**⚠️ 局限性**

局限性包括①证据层仍存在约1.3%的残留风险；②谓词模式需手工补全，零射门缺失；③对新工具或更复杂多模态输入的鲁棒性尚未验证；④评估仅在离线任务上完成，未覆盖完整多轮交互和机器人等真实部署场景；⑤对自适应攻击的鲁棒性仍需进一步提升。

---

## 246. Operationalizing Document AI: A Microservice Architecture for OCR and LLM Pipelines in Production

**arXiv ID:** 2605.18818 | [PDF](https://arxiv.org/pdf/2605.18818v1)

**作者:** Yao Fehlis `[一作]` (Kungfu.ai), Steve Kramer `[通讯]` (Kungfu.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一套微服务架构，用于在生产环境下高吞吐量处理多页文档，包含多模型分类、OCR、LLM结构化抽取，并通过异步消息队列实现解耦与弹性扩展。

**💡 创新点**

创新点包括：①将GPU绑定的推理与CPU绑定的编排拆分为独立服务，支持按需水平扩展；②采用混合分类策略（CLIP‑KNN+Claude Sonnet）在保持高精度的同时显著降低成本；③利用异步协程与批量推理叠加实现计算资源的最大化；④通过可观测性与多级重试实现高可靠性。

**🔧 技术方法**

使用技术包括：Docker+FastAPI容器化推理服务；Kubernetes+消息队列（WorkerQueue）实现异步任务分发；GPU加速的DocTR/Docling OCR模型；Anthropic Claude Sonnet LLM进行结构化抽取；MLflow模型注册与版本化；异步协程与流式 I/O。

**📊 数据集**

数据集主要为公开表单数据集（如 FUNSD、SROIE 等）用于模型训练和验证，系统性能评估则使用内部合成的多页扫描文档（数百份多页 PDF/TIFF），模拟真实批处理场景。

**📈 对比分析**

方法对比：混合分类在 96% 的准确率下，成本仅为 CLIP‑KNN 单独模式的 0.001 元/页，VLM 仅为 0.010 元/页；OCR 与 LLM 的延迟分别为 1–2 秒/页和 3 秒/文档；整体系统在 8 页文档上平均成本约 0.038 元，OCR 占比 80% 的总延迟；批量实验表明吞吐量随并发数提升，GPU 推理成为瓶颈。

**⚠️ 局限性**

局限性包括：①性能数据基于合成数据，真实文档质量会影响 OCR 结果；②仅在单一 GPU 节点上验证，GPU 资源调度仍需手工调优；③缺乏长期稳定性评估（如 24/7 运行的错误率）；④对大规模 VLM 直接推理的成本与延迟尚未深入探讨；⑤系统设计依赖于消息队列的可用性与配置正确性，若错误未及时检测会导致重试堆积。

---

## 247. FLUIDSPLAT: Reconstructing Physical Fields from Sparse Sensors via Gaussian Primitives

**arXiv ID:** 2605.18866 | [PDF](https://arxiv.org/pdf/2605.18866v1)

**作者:** Huaxi Huang `[一作]` (Shanghai Artificial Intelligence Laboratory), Xiao Sun `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 46462 | [OpenAlex ID](https://openalex.org/A5033342186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

开发了一种基于高斯原子构建的空间中间表示与残差解码器的稀疏表面传感器流场重建框架。

**💡 创新点**

通过引入可分离的高斯原子分区-统一结构与状态条件残差解码，提供可解释的空间状态并给出高斯原子容量与观测数量的理论逼近‑估计分析。

**🔧 技术方法**

采用排列不变的传感器编码器、线性投影生成高斯原子参数、傅里叶特征残差解码器和交叉注意力；同时在训练中加入传感器一致性损失。

**📊 数据集**

在 Senseiver 2D 圆柱涡量数据集（表面 4/8/16 传感器布局）和 AirfRANS 2D RANS 空气动力学数据集（8 传感器表面压强）上评估。

**📈 对比分析**

与 Senseiver 官方、DeepONet、FLRONet、RecFNO 等基线对比，在所有传感器布局上取得最低相对 L2 误差；AirfRANS 上比最强基线提升 11–23%。

**⚠️ 局限性**

仅在二维静态场景验证，理论假设固定高斯中心和理想观测，缺乏不确定性估计，未来需扩展到三维及时间依赖场。

---

## 248. EgoBabyVLM: Benchmarking Cross-Modal Learning from Naturalistic Egocentric Video Data

**arXiv ID:** 2605.19130 | [PDF](https://arxiv.org/pdf/2605.19130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 249. DMN: A Compositional Framework for Jailbreaking Multimodal LLMs with Multi-Image Inputs

**arXiv ID:** 2605.18915 | [PDF](https://arxiv.org/pdf/2605.18915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 250. Simply Stabilizing the Loop via Fully Looped Transformer

**arXiv ID:** 2605.18797 | [PDF](https://arxiv.org/pdf/2605.18797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 251. EUPHORIA: Efficient Universal Planning via Hybrid Optimization for Robust Industrial Robotic Assembly

**arXiv ID:** 2605.18872 | [PDF](https://arxiv.org/pdf/2605.18872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 252. Quantum Adversarial Machine Learning: From Classical Adaptations to Quantum-Native Methods

**arXiv ID:** 2605.18821 | [PDF](https://arxiv.org/pdf/2605.18821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 253. Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting

**arXiv ID:** 2605.19249 | [PDF](https://arxiv.org/pdf/2605.19249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 254. Beyond Nutrition Labels: How Analogical Reasoning Shapes Synthetic Media Disclosure Design

**arXiv ID:** 2605.19045 | [PDF](https://arxiv.org/pdf/2605.19045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 255. KAN-MLP-Mixer: A comprehensive investigation of the usage of Kolmogorov-Arnold Networks (KANs) for improving IMU-based Human Activity Recognition

**arXiv ID:** 2605.19031 | [PDF](https://arxiv.org/pdf/2605.19031v1)

**作者:** Mengxi Liu `[一作]` (DFKI), Paul Lukowicz `[通讯]` (DFKI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于 KAN 与 MLP 的混合架构 KAN-MLP-Mixer，用于提升 IMU 传感器下的人类活动识别性能。

**💡 创新点**

创新点在于通过系统实验确定 KAN 仅放在数据嵌入和分类器位置，MPL 负责特征混合，显著提升精度并保持鲁棒性。

**🔧 技术方法**

使用了 EfficientKAN、LarctanKAN 等 KAN 变体、标准 MLP 层、混合架构设计以及大规模的 ablation 与基准实验。

**📊 数据集**

在八个公开 HAR 数据集（DG、DSADS、PAMAP2、OPPO、Skodar、HAPT、MotionSense、MHEALTH）上进行评估。

**📈 对比分析**

通过与纯 MLP、纯 KAN 以及多种主流模型的宏 F1 比较，Hybrid 模型在所有数据集上平均提升 5.33% 的宏 F1 分数，且在多种配置下保持优异表现。

**⚠️ 局限性**

局限性包括仅验证了 IMU 单模态数据、未覆盖多模态或极端噪声环境、模型相对更大且计算成本略高，以及缺乏在边缘设备上的部署与能耗评估。

---

## 256. Generative Pseudo-Force Fields for Molecular Generation

**arXiv ID:** 2605.19050 | [PDF](https://arxiv.org/pdf/2605.19050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 257. RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning

**arXiv ID:** 2605.19033 | [PDF](https://arxiv.org/pdf/2605.19033v1)

**作者:** Ehsan Ahmadi `[一作]` (University Of Alberta), Kasra Rezaee `[通讯]` (Huawei Technologies Canada)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于强化学习的后训练框架 RLFTSim，用以提升交通仿真模型的现实性并实现目标驱动的可控场景生成。

**💡 创新点**

创新点包括：①将 Waymo Open Simulation Challenge 的 Meta‑Metric 作为奖励信号；②设计了低方差、密集的 Meta‑Metric Leave‑One‑Out (MLOO) 奖励以显著提升样本效率；③结合 Hindsight Experience Replay 对 RLFTSim 进行目标条件化，实现在保持现实性的同时实现可控性。

**🔧 技术方法**

使用的技术主要是：强化学习（REINFORCE 与 KL 正则化）、Meta‑Metric Leave‑One‑Out 奖励、Hindsight Experience Replay、目标条件化观测表示（concatenation 与 indication）以及基于 SMART‑tiny 的预训练模型。

**📊 数据集**

数据集采用 Waymo Open Motion Dataset (WOMD) 进行训练、验证与评估。

**📈 对比分析**

与基准模型（SMART‑tiny、CAT‑K 等）相比，RLFTSim 在 RMM、Kinematic、Interactive 与 Map‑based 四项子指标上均取得最高分，尤其在 RMM 上实现了 0.7867 的新高；相比传统监督与基于启发式搜索的微调方法，RLFTSim 的样本量显著降低，并且在目标完成率和碰撞率方面也优于现有方法。

**⚠️ 局限性**

局限性包括：①基于 token 的轨迹表示在高速动态场景下响应性不足；②目标条件化的可控性仍不完美，目标完成率有待提升；③RMM 作为现实性代理可能存在饱和或不足，未来需改进更精准的评估指标。

---

## 258. Structural Analysis of Cryptographic Sequences using Stringology-Based Fingerprinting

**arXiv ID:** 2605.19123 | [PDF](https://arxiv.org/pdf/2605.19123v1)

**作者:** Victor Kebande `[一作]` `[通讯]` (University of Colorado Denver), Victor Kebande (University of Colorado Denver)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于字符串学的指纹框架（SBF），用于从加密序列中提取结构模式并构建指纹向量。

**💡 创新点**

创新点在于将字符串学的子串频率、重复模式、熵等结构统计与加密序列分析相结合，提供了除传统随机性测试之外的结构性视角。

**🔧 技术方法**

采用子串频率统计、滑动窗口、归一化分布、偏差得分、熵计算以及模式重复统计等字符串处理技术。

**📊 数据集**

使用了两类数据集：10,000条加密生成序列（CGS）和10,000条来自均匀随机分布的序列（URS），每条长度为2^12位。

**📈 对比分析**

通过对比子串频率、偏差分数和熵值等指标，发现加密序列在不同子串长度上表现出略高的集中度和小的偏差，证明SBF能检测到可测量的结构差异。

**⚠️ 局限性**

局限在于检测到的结构差异非常微小，尚未表明对实际安全性构成威胁；且实验仅覆盖了单一类型的加密生成器，未验证跨算法的普适性。

---

## 259. Hybrid-LoRA: Bridging Full Fine-Tuning and Low-Rank Adaptation for Post-Training

**arXiv ID:** 2605.18822 | [PDF](https://arxiv.org/pdf/2605.18822v1)

**作者:** Chengqian Zhang `[一作]` (Worcester Polytechnic Institute), Kyumin Lee `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 2745 | [OpenAlex ID](https://openalex.org/A5103224637)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种混合微调框架Hybrid-LoRA，结合全参数微调与LoRA，按模块重要性分配更新策略；

**💡 创新点**

创新点在于引入Hybrid-Score指标，用第一阶梯度加权敏感度评估模块对LoRA适应度，从而实现高效的模块分配；

**🔧 技术方法**

使用RLVR（GRPO/GSPO）后训练、LoRA低秩重参数化、SVD分解、梯度敏感度评估与贪心分配；

**📊 数据集**

在Qwen-2.5 1.5B/3B/7B模型上，使用六个推理基准（Math‑500、AIME‑24/25、MMLU‑Pro、GPQA、LeetCodeDataset）和Mixture‑of‑Thoughts训练语料；

**📈 对比分析**

与完整微调、AdaLoRA、AutoLoRA、LoRA‑Drop等PEFT基线对比，Hybrid‑LoRA在10%全微调预算下平均提升4.36%，在大多任务上可匹配或超越全微调；

**⚠️ 局限性**

受限于实验规模仅至7B模型，且采用两阶段探测+训练流程，未探索端到端或动态模块分配方式。

---

## 260. Token by Token, Compromised: Backdoor Vulnerabilities in Unified Autoregressive Models

**arXiv ID:** 2605.19227 | [PDF](https://arxiv.org/pdf/2605.19227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 261. MANGO: Meta-Adaptive Network Gradient Optimization for Online Continual Learning

**arXiv ID:** 2605.19080 | [PDF](https://arxiv.org/pdf/2605.19080v1)

**作者:** Ankita Awasthi `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**通讯引用:** 47744 | [OpenAlex ID](https://openalex.org/A5031161187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 MANGO 的在线连续学习框架，旨在在单次通过、有限存储的环境中解决稳定性-可塑性矛盾。

**💡 创新点**

创新点包括：① 梯度门控机制，根据参数归一化敏感度动态缩放梯度，防止对过去知识的破坏；② 通过基于回放的双层元学习正则化，自适应学习层级稳定系数，直接评估并缓解遗忘；③ 将回放既作为训练信号又作为遗忘评估器，实现闭环反馈。

**🔧 技术方法**

核心技术为梯度门控、元学习正则化、经验回放、EWC 风格正则化、ResNet-18 网络架构，采用 SGD 优化，使用 Reservoir Sampling 更新回放缓冲。

**📊 数据集**

在三大标准在线连续学习基准上验证：Split CIFAR‑100、Split Tiny‑ImageNet（类增量）以及 CLEAR‑10（域增量）。

**📈 对比分析**

与 ER、ER‑ACE、GDUMB、iCaRL、LUCIR、DER++、LODE、Fine‑Tuning 等基线对比，MANGO 在所有数据集和回放容量下均取得显著提升：在 CIFAR‑100 上准确率提升至 19.72%（相较 LODE 14.36%），在 Tiny‑ImageNet 上 24.73%（相较 LODE 11.68%），在 CLEAR‑10 上获得 66.91% 并实现正向后向转移 (+15.12%)，整体表现为最优的 Acc、AAA、WC‑Acc 并保持稳定。

**⚠️ 局限性**

局限性在于假设任务/域边界明确，当前方法不适用于任务无边界或模糊边界的无监督连续学习环境，未来工作计划扩展到任务无关设置并预训练基础模型。

---

## 262. OEP: Poisoning Self-Evolving LLM Agents via Locally Correct but Non-Transferable Experiences

**arXiv ID:** 2605.18930 | [PDF](https://arxiv.org/pdf/2605.18930v1)

**作者:** Kaixiang Wang `[一作]` (Shanghai Jiao Tong University), Jie Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 28085 | [OpenAlex ID](https://openalex.org/A5100428255)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出低特权黑盒攻击OEP，利用清洁边缘案例配合恶意后果诱导自我反射生成不可迁移的高优先级规则，导致后续任务性能下降。

**💡 创新点**

创新点在于：1) 通过清洁但局部可行的案例与严重假设后果共同构建攻击，绕过内容过滤；2) 利用LLM的安全对齐和风险敏感性，使得反射过程偏向错误规则；3) 只需用户级交互，无需修改系统提示或内存。

**🔧 技术方法**

技术手段包括：清洁边缘案例生成、Adversarial Consequence Triplet（ACT）构造、基于LLM的反射式记忆整合、对记忆可信度与风险评估的机制分析。

**📊 数据集**

使用数据集：GSM8K（数学）、MedQA（医疗）、ToolAlpaca（工具调用）来评估攻击在不同领域的效果。

**📈 对比分析**

与 Prompt Injection、MINJA、AgentPoison、MemoryGraft、InjectAgent 等基线对比，OEP 在 GPT‑4o 上攻击成功率（ASR）>50%，且在防御（提示过滤、LLM审计）下仍保持较高 ASR，优于其他攻击方法。

**⚠️ 局限性**

局限性包括：1) 需要较高比例恶意案例才能显著干扰，2) 对于具备严格安全约束的任务（如医疗）效果下降，3) 需针对反射机制特定实现，且多代理辩论等新型防御可能降低攻击成功率。

---

## 263. Precision Tracked Transformer via Kalman Filtering, Kriging and Process Noise

**arXiv ID:** 2605.18832 | [PDF](https://arxiv.org/pdf/2605.18832v1)

**作者:** Bo Long `[一作]` (LinkedIn Core AI), Liuqing Li `[通讯]` (LinkedIn Core AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Transformer中引入贝叶斯滤波框架，给每个位置加入精度（置信度）并通过克里金加权、卡尔曼增益和FFN预测实现不确定性传播，得到Bayesian Filtering Transformer（BFT）。

**💡 创新点**

创新点在于：①把自注意力视为观察步骤、残差视为更新、FFN视为预测，统一为贝叶斯滤波；②通过REML+共轭先验无参数地估计观察精度；③在FFN中用雅可比传播精度并加入过程噪声；④对序列推荐和LLM微调的冷启动/噪声场景实现统一且显著的性能提升。

**🔧 技术方法**

使用的技术包括：贝叶斯滤波、克里金加权、卡尔曼增益、REML估计、雅可比δ方法、过程噪声学习、预归一化层、注意力加权偏置、对数精度偏置、SVD压缩、无额外记忆占用的实现。

**📊 数据集**

在序列推荐上使用MovieLens‑1M和五个Amazon类别（Sports、Instruments、Games、Toys、Beauty）六个数据集；在LLM微调上使用TinyLlama‑1.1B，在SQuAD（token‑label噪声）和NQ‑Open（检索噪声）测试。

**📈 对比分析**

与原始Transformer层做drop‑in替换，采用同样的训练配置和评估指标（HR@10、NDCG@10、MRR、F1）。在推荐任务上，BFT在所有六个数据集上显著提升，最显著在稀疏数据集（如Instruments）提升可达15% HR@10；在LLM任务中，BFT在噪声场景下F1提升3‑8%，比标准SFT和焦点损失显著优。

**⚠️ 局限性**

限制包括：在中等稠密数据集上提升不显著；仅实现单层或单头的对角精度，未探索低秩/全精度；LLM实验仅在TinyLlama‑1.1B规模，未验证到更大模型；训练过程中需微调过程噪声参数，可能对超参敏感。

---

## 264. Spectral Gradient Surgery for Domain-Generalizable Dataset Distillation

**arXiv ID:** 2605.18836 | [PDF](https://arxiv.org/pdf/2605.18836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 265. RecoAtlas: From Semantic Plausibility to Set-Level Utility in LLM Recommendation Agents

**arXiv ID:** 2605.18805 | [PDF](https://arxiv.org/pdf/2605.18805v1)

**作者:** Imad Aouali `[一作]` (Criteo AI Lab), Benjamin Heymann `[通讯]` (Criteo AI Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RecoAtlas基准，用于评估使用工具生成结构化购物报告的LLM推荐代理。

**💡 创新点**

创新点在于将行为驱动的评估（精确恢复、学习的相关性、互补性与多样性奖励模型）与语义LLM评判相结合，并设计了语义、行为对齐及失效工具的受控环境，能够诊断模型在推理、信号和工具使用上的差异。

**🔧 技术方法**

采用LLM代理与工具调用框架，构建多种工具接口，训练查询‑项目、项目‑项目的双编码器奖励模型，并利用LLM评判器进行语义与解释质量评分。

**📊 数据集**

使用亚马逊三大品类（乐器、电子产品、视频游戏）数据，包括商品元数据、用户‑项目交互及共购信息。

**📈 对比分析**

通过与专有模型（GPT‑4.1 mini、Grok 4.1 Fast）及开源模型（Qwen3、Gemini、Mistral等）在相同工具和提示下的SetHit@20、奖励模型分数和LLM评判指标对比，发现模型规模和推理能力与性能呈正相关；开放模型在bundle任务上表现突出，但LLM评判与行为指标往往不一致。

**⚠️ 局限性**

局限性包括：基准为离线评估，缺乏真实用户实验；依赖从行为数据生成的“真值”可能无法完全反映实际购买意图；工具实现需人工构造，扩展性受限；仅覆盖部分电商品类，结果对其他领域的泛化尚待验证。

---

## 266. DualView: Adaptive Local-Global Fusion for Multi-Hop Document Reranking

**arXiv ID:** 2605.18767 | [PDF](https://arxiv.org/pdf/2605.18767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 267. Auditing Reasoning-Trace Memorization Claims after Unlearning with Head-Conditioned Canaries

**arXiv ID:** 2605.18891 | [PDF](https://arxiv.org/pdf/2605.18891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 268. Generative and isoparametric geometric modeling of large-scale and multiscale microstructures

**arXiv ID:** 2605.18894 | [PDF](https://arxiv.org/pdf/2605.18894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 269. ExECG: An Explainable AI Framework for ECG models

**arXiv ID:** 2605.19258 | [PDF](https://arxiv.org/pdf/2605.19258v1)

**作者:** Jong-Hwan Jang `[一作]` (Medical AI Co. Ltd.), Yong-yeon Jo `[通讯]` (Medical AI Co. Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 ExECG Python 框架，实现 ECG 模型的标准化访问、统一 XAI 执行和 ECG 对齐可视化，以实现可解释性评估。

**💡 创新点**

将标准化、可复现、集成与可扩展四大设计原则结合，提供统一接口、共享执行协议和 ECG 专用组件（StyleGAN 生成器、概念数据集、12 导联可视化），填补通用 XAI 框架对 ECG 支持不足的空缺。

**🔧 技术方法**

使用深度学习模型（1D ResNet）、梯度归因（Grad-CAM、SmoothGrad、Integrated Gradients 等）、对抗样本（StyleGAN 生成的 Counter‑Fact），概念敏感性（TCAV），以及 Python、PyTorch、可视化库。

**📊 数据集**

PTB‑XL 及 MIMIC‑IV ECG 两大公开 ECG 数据集。

**📈 对比分析**

通过 AF 二分类案例演示多种归因方法、对抗样本与概念方法的对比，可视化显示模型关注 P 波等临床特征；未给出量化指标，仅展示可解释性一致性和差异。

**⚠️ 局限性**

仅在单一 AF 二分类任务上验证，缺乏跨任务、跨数据集、跨模型的系统评估；方法的精度评估不完整，未对不同 XAI 的性能做定量比较。

---

## 270. Position: Graph Condensation Needs a Reset -- Move Beyond Full-dataset Training and Model-Dependence

**arXiv ID:** 2605.18893 | [PDF](https://arxiv.org/pdf/2605.18893v1)

**作者:** Mridul Gupta `[一作]` (IIT Delhi), Sayan Ranu `[通讯]` (IIT Delhi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对图凝结（Graph Condensation）研究现状进行系统评述，指出现行方法过度依赖全数据集训练、梯度匹配以及特定模型，导致可扩展性、泛化性和实际部署效果受限，并提出“图凝结需要重启”这一研究方向，呼吁采用轻量、模型无关、资源友好的技术与评价框架。

**💡 创新点**

创新点主要包括：①提出基于字节大小（byte‑size）而非节点压缩率的公平评价指标；②给出从任务保留、压缩度、泛化性、计算效率四个核心维度出发的图凝结形式化定义；③系统分类现有方法为“模型相关/无关 × 全数据训练/非全数据训练”，并指出这一划分的意义；④强调大规模真实图（如 MAG240M、MalNet 等）作为验证基准的重要性；⑤讨论梯度匹配的根本缺陷并鼓励向分布/树匹配等无梯度方向转型。

**🔧 技术方法**

主要技术包括：梯度匹配与专家轨迹匹配的分析与批评；分布/树匹配（Tree Mover’s Distance、MMD 等）方法；基于字节大小的压缩度衡量；构建大规模图数据集的实验平台；对模型无关化的设计思路（如基于受限树结构、谱约束、信息论目标等）。

**📊 数据集**

使用的数据集既有传统基准（Cora、Citeseer、Pubmed、ogbn-arxiv 等）也有大规模真实图（MAG240M、MalNet、TUDataset、OMAT24、MPTrj 等），并指出在小基准上做的实验往往缺乏意义，真正的评估应聚焦于大规模、资源消耗高的图。

**📈 对比分析**

方法对比主要集中在对比传统梯度匹配型方法（如 GCond、GDEM、GCSR 等）与无梯度型方法（如 TMD、Bonsai、GCSR 等）的性能、内存/时间开销。实验显示：①梯度匹配方法在模型迁移（从 GCN 到 GAT、GIN）时性能急剧下降；②许多方法在大规模图上因全数据训练或 N² 计算导致 OOM/超时；③使用字节压缩率后，各方法的“压缩效果”差距被明显削弱，说明传统节点压缩率会误导。

**⚠️ 局限性**

主要局限包括：①依赖全数据训练导致成本不减反增；②模型依赖性削弱泛化；③缺乏统一、可复现的评价协议；④现有大规模实验不足，导致对真实资源需求评估失真；⑤大多数方法对超参数敏感，实际部署成本高；⑥梯度匹配方式在多任务、跨任务、异构模型中表现不佳。

---

## 271. Position: Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance

**arXiv ID:** 2605.18801 | [PDF](https://arxiv.org/pdf/2605.18801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 272. Ordered Adjoint Logic (Extended Version)

**arXiv ID:** 2605.19112 | [PDF](https://arxiv.org/pdf/2605.19112v1)

**作者:** Sophia Roshal `[一作]` (Carnegie Mellon University), Frank Pfenning `[通讯]` (Carnegie Mellon University)

**通讯引用:** 11544 | [OpenAlex ID](https://openalex.org/A5021476649)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一个新的有序 adjoint 逻辑体系，包含顺序敏感的模式、可控的弱化、收缩与单向移动性，并给出了相应的序列算子与自然演绎形式。

**💡 创新点**

创新点在于将子指数和相邻结构规则统一为由模式预序决定的 adjoint 模式，实现了单向移动性、弱化和收缩的细粒度控制；证明了该系统的 cut 消除与自然演绎形式的可判定性，提供了可直接用于类型检查的基础。

**🔧 技术方法**

使用了 Gentzen 风格的序列计算、adjoint 模态（↑、↓）、子指数、结构化证明归约技术，并设计了基于“正则化”与“归一化”算法来实现隐式结构规则的判定。

**📊 数据集**

无实验数据集，论文完全为形式化理论与证明。

**📈 对比分析**

通过对比先前的子指数有序逻辑（Kanovich 等）和 LNL（Benton）展示其更高的表达性与更简洁的结构规则；由于采用形式化证明，未给出运行时性能指标。

**⚠️ 局限性**

局限性：算法复杂度可能呈指数级；缺乏针对实际编程语言的实现细节与实验验证；对多重上下文匹配与弱化的细节优化仍待研究。

---

## 273. Delta Attention Residuals

**arXiv ID:** 2605.18855 | [PDF](https://arxiv.org/pdf/2605.18855v1)

**作者:** Cheng Luo `[一作]` (Independent Researcher), Junjie Hu `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Delta Attention Residuals，通过使用层间增量而非累计状态进行跨层路由，以获得更尖锐的注意力分布。

**💡 创新点**

创新点在于将路由源改为每个子层的增量信息，并采用加法式路由，解决了 Attention Residuals 中的源冗余导致路由崩溃的问题。

**🔧 技术方法**

使用了 Transformer（Qwen3 体系）、RMSNorm、softmax 注意力、Delta 路由、加法式路由、可微分路由参数、LoRA、FSDP 与梯度检查点等技术。

**📊 数据集**

训练数据使用 FineWeb‑Edu；下游评估使用 8 个标准基准（HellaSwag、ARC‑Easy/Challenge、PIQA、WinoGrande、BoolQ、MMLU、LAMBADA）。

**📈 对比分析**

在相同架构与超参数下与 Baseline、Attention Residuals、Full Attention Residuals 进行对比，Delta 方法在 220M–7.6B 规模上均显著降低验证困惑度（最多 8.2%），Fine‑tuning 后在下游任务平均准确率提升 0.6%。

**⚠️ 局限性**

限制：Delta AttnRes 由于存储 2L 个源导致吞吐量降低约 70% 且显存增长约 3.5 倍；Delta Block 需要根据块大小权衡速度与内存，且实现相对复杂。

---

## 274. Surviving the Unseen: Predictive Defense for Novel Multi-Turn Multimodal Attacks

**arXiv ID:** 2605.18988 | [PDF](https://arxiv.org/pdf/2605.18988v1)

**作者:** Doohee You `[一作]` `[通讯]` (Google), Doohee You (Google)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出TRIAD框架，在多模态大语言模型多轮交互中通过动态生存分析与轨迹监控实现安全防护；

**💡 创新点**

将安全校准从静态分类转为可预测的时间‑至‑失效模型，并首次结合Isolation Forest、Ledoit‑Wolf正则化的Mahalanobis距离与轨迹加速度实现对跨模态持续攻击的预警；

**🔧 技术方法**

使用Isolation Forest做结构性预检测，Ledoit‑Wolf估计逆协方差计算稳健Mahalanobis距离，轨迹加速度量化连续漂移，Bayesian HMM更新隐藏状态，Cox比例风险模型预测即时风险；

**📊 数据集**

基于公开多模态对话语料库（如多模态对话数据集）提取嵌入做基线，随后在自构造的跨模态漂移实验中评估；

**📈 对比分析**

与传统单点文本/图像安全过滤器及已有多轮攻击检测方法对比，TRIAD在零日跨模态攻击下显著提升检测率、保持低误报，并通过理论证明实现失效时间上界；

**⚠️ 局限性**

局限包括对多峰分布的处理需引入GMM、对突发攻击的时间惯性需要AFT补偿、对微小子阈值攻击可能仍逃逸、对协方差估计的依赖导致对新模态的泛化性受限；

---

## 275. Quantized Machine Learning Models for Medical Imaging in Low-Resource Healthcare Settings

**arXiv ID:** 2605.19207 | [PDF](https://arxiv.org/pdf/2605.19207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 276. Causal Evidence for Attention Head Imbalance in Modality Conflict Hallucination

**arXiv ID:** 2605.19250 | [PDF](https://arxiv.org/pdf/2605.19250v1)

**作者:** Jinrui Jiang `[一作]` (Nanjing University), Xinyu Dai `[通讯]` (Nanjing University)

**通讯引用:** 4814 | [OpenAlex ID](https://openalex.org/A5102994315)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多模态大型语言模型中的模态冲突幻觉进行机理分析，识别并验证了驱动与抑制幻觉的注意力头，并提出了基于因果干预的MACI方法。

**💡 创新点**

首次在注意力头层面用路径补丁法进行因果归因，揭示驱动与抑制头的两极不对称，并基于此设计了只在检测到冲突时才抑制驱动头的条件干预机制。

**🔧 技术方法**

路径补丁、头级因果分析、零样本冲突检测、逻辑回归探针以及针对性头消融。

**📊 数据集**

MMMC（对象、属性、关系冲突）作为主测试集，并在SCI‑SemanticConflict子集进行零样本跨基准验证。

**📈 对比分析**

与基线方法VCD、ICD、OPERA和ASCD在MMMC上进行对比，MACI在减少幻觉率上取得最高改进，且在SCI‑SemanticConflict上也实现了显著的零样本提升。

**⚠️ 局限性**

方法依赖于对象冲突训练数据与prefill阶段，探针需标注样本，评估仅覆盖MMMC，尚未探讨视觉输入误导场景。

---

## 277. SAGA: A Sequence-Adaptive Generative Architecture for Multi-Horizon Probabilistic Forecasting with Adaptive Temporal Conformal Prediction

**arXiv ID:** 2605.19014 | [PDF](https://arxiv.org/pdf/2605.19014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 278. Composition of Memory Experts for Diffusion World Models

**arXiv ID:** 2605.18813 | [PDF](https://arxiv.org/pdf/2605.18813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 279. A Reproducibility Analysis of PO4ISR: Diagnosing and Mitigating Semantic Drift in LLM-Based Session Recommendation

**arXiv ID:** 2605.18780 | [PDF](https://arxiv.org/pdf/2605.18780v1)

**作者:** Aditya Tiwari `[一作]` (Indian Institute of Technology Bhilai), Rajesh Kumar Mundotiya `[通讯]` (Indian Institute of Technology Bhilai)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5003314662)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于推理的会话推荐框架 PO4ISR 进行可复现性研究，并提出改进版 PO4ISR++，解决推理提示导致的上下文漂移与解析错误。

**💡 创新点**

创新点在于：① 采用确定性索引输出消除文本解析歧义；② 引入反射式跨域融合机制，利用多域专家提示生成可迁移的通用推理提示；③ 通过结构化输出与多域验证实现可复现的鲁棒提升。

**🔧 技术方法**

核心技术包括：大型语言模型 Gemini‑2.0、确定性索引格式化、反射式跨域融合（Meta‑Prompt 与域专家融合）、双阶段验证与优化、零温度推理保证确定性。

**📊 数据集**

使用三大公开数据集：MovieLens‑1M、Amazon Games、Bundle（电子、服装、食品），涵盖稠密、稀疏与数值歧义场景。

**📈 对比分析**

与传统单意图、多意图、图神经网络及其他 LLM 推理方法进行对比，采用 HR@1/5 与 NDCG@1/5 评估。在 Games 上提升 54.6%，Bundle 上提升 96.4%，整体在多域场景保持稳定优于基线。

**⚠️ 局限性**

局限性：仍依赖高质量域专家提示，跨域融合过程对提示选择敏感；在极端数值复杂或多语言场景下可能需进一步的解析稳健性改进；实验集中在三类数据集，未覆盖极大规模或实时场景。

---

## 280. LiFT: Lifted Inter-slice Feature Trajectories for 3D Image Generation from 2D Generators

**arXiv ID:** 2605.19060 | [PDF](https://arxiv.org/pdf/2605.19060v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 281. Dynamic Model Merging Made Slim

**arXiv ID:** 2605.18904 | [PDF](https://arxiv.org/pdf/2605.18904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 282. From Intent to AI Pipelines: A Controlled Agentic Framework for Non-AI Expert Scientists

**arXiv ID:** 2605.18764 | [PDF](https://arxiv.org/pdf/2605.18764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 283. Adversarial Stress Testing of SPARK Humanoid Safety Filters

**arXiv ID:** 2605.19009 | [PDF](https://arxiv.org/pdf/2605.19009v1)

**作者:** Saurav Ghosh `[一作]` (Washington University in St. Louis), Luke Zhang `[通讯]` (Washington University in St. Louis)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 MuJoCo 环境下复现 SPARK 机器人安全过滤器基准，并开发解析管道将高维日志转为可解释的安全与任务指标；随后通过障碍物拥挤、感知噪声和传感器延迟三种攻击对六种过滤器（RSSA、RSSS、SSA、CBF、PFM、SMA）进行鲁棒性压力测试。

**💡 创新点**

（1）首次完整复现 SPARK 并统一评估框架；（2）构建端到端的日志解析与指标生成流水线；（3）针对感知层面提出多维攻击模型，揭示原始性能与鲁棒性之间的差距。

**🔧 技术方法**

利用 MuJoCo 仿真、Python 自动化脚本、npz 日志解析、Gaussian 噪声与时延注入攻击以及六种安全过滤器的实现（RSSA、RSSS、SSA、CBF、PFM、SMA）。

**📊 数据集**

SPARK G1 人形机器人基准案例（单位树 G1，SportMode），包含固定任务、目标、障碍物数量可调（5、15、30）。

**📈 对比分析**

通过平均碰撞步数与最终目标距离的双指标对比，发现 PFM 在目标跟踪上表现最佳，SMA 在碰撞防护上最优，RSSS/SSA 处于两者平衡；在拥挤、噪声、时延攻击下，各过滤器表现差异显著，证明单一指标无法全面评估鲁棒性。

**⚠️ 局限性**

实验耗时长，单种随机种子与单个基准案例限制了泛化；未系统评估无解情况的频率；仅针对六种过滤器，未涵盖 SPARK 全部任务与更多随机种子。

---

## 284. TEMPO: Temporal Enforcement via Mode-Separated Policy Optimization for Trustworthy LLM Backtesting

**arXiv ID:** 2605.18843 | [PDF](https://arxiv.org/pdf/2605.18843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 285. Automated Grading of Handwritten Mathematics Using Vision-Capable LLMs

**arXiv ID:** 2605.19043 | [PDF](https://arxiv.org/pdf/2605.19043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 286. Concave is the New Linear: The Impossibility of Anti-Plutocratic DAO Governance

**arXiv ID:** 2605.18990 | [PDF](https://arxiv.org/pdf/2605.18990v1)

**作者:** Austin Bennett `[一作]` (Circle Research), Mira Belenkiy `[通讯]` (Circle Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对权威链上DAO治理中基于钱包余额的投票规则进行理论分析与实证，证明任意非平凡的投票规则在Sybil攻击下都会变得线性，并展示了在实际大型DAO中的攻击放大因子。

**💡 创新点**

将之前仅针对二次投票的Sybil攻击结果推广到所有有限、正、单调且凹的投票规则，并给出紧致的渐近斜率公式；同时通过实证数据验证攻击放大系数可达数十万倍。

**🔧 技术方法**

构建成本化的数学模型（包含分拆费、投票费、固定设置费和最小余额限制），利用Jensen不等式和渐近分析证明极大攻击收益为线性；随后在链上数据上实现模拟计算。

**📊 数据集**

使用Tally获取ENS、Compound、Uniswap、Arbitrum、ZKsync等五大DAO近十个已完成提案的投票记录，并从CoinGecko获取对应的ETH与治理代币价格；计算每个链的gas费用。

**📈 对比分析**

将攻击者在不同投票规则下的最优成本与所有诚实投票者的总投票权做对比，得到Sybil放大因子；结果显示在二次、对数、幂函数等凹规则下，攻击成本比线性投票低3–5个数量级，放大因子最高可达229,000倍。

**⚠️ 局限性**

假设攻击者可无限制地拆分钱包且不受身份绑定限制；未考虑投票者行为随投票规则改变而变化的情况；仅适用于基于钱包余额的投票规则，无法涵盖基于身份的或混合机制。

---

## 287. Metric-Gradient Projection for Stable Multi-Agent Policy Learning

**arXiv ID:** 2605.18809 | [PDF](https://arxiv.org/pdf/2605.18809v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6510 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Hodge投影的多智能体学习方法（HPML），通过将联合更新场投影到最接近的度量梯度势场来消除非势性循环，提升学习稳定性和最终回报。

**💡 创新点**

创新点在于将多智能体学习中的联合更新场视为L²空间的向量场，并用Hodge投影将其分解为可积的梯度分量与非势残差；给出Poisson型方程的变分表述，提供图结构与神经网络两种可实现的投影方案；并在理论上证明投影动态具有Lyapunov性质，残差项显式出现在VI间隙界限中。

**🔧 技术方法**

使用的技术包括：变分投影与Poisson方程、图Laplacian与离散边流、可训练的标量势能神经网络（amortized projection）、基于度量梯度的潜在梯度投影、VI和残差能量度量、以及在CTDE框架下的actor-critic更新。

**📊 数据集**

在实验中使用了可解释的控制游戏（如循环矩阵游戏、两维势+反对称场、逻辑混合策略游戏、已知势的3D线性场）以及10个Melting Pot多智能体基准任务（涉及约定选择、重复协调、社会困境等）。

**📈 对比分析**

与IPPO、MAPPO、HAPPO、COMA等基线进行比较，HPML-MAPPO（图形版）在所有示例中均表现出更低的循环能量、更高的最终标准化回报；HPML-MAPPO（神经版）在某些任务上也获得提升，但仍显现更高的残差能量。整体来看，HPML提升了学习稳定性和收敛速度。

**⚠️ 局限性**

局限性包括：对所选度量矩阵和采样分布高度依赖；投影近似（图解或神经网络）可能导致计算开销；在复杂高维场景下需要足够的采样点以逼近连续投影；残差项仍可能影响VI间隙，需要进一步理论与实践结合。

---

## 288. Towards Zero Trust Architecture: A Pilot Study on Information Systems Security Readiness amongst Small and Medium Enterprises

**arXiv ID:** 2605.18901 | [PDF](https://arxiv.org/pdf/2605.18901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 289. On the transversals of Latin squares generated by nonlinear bipermutive cellular automata

**arXiv ID:** 2605.18875 | [PDF](https://arxiv.org/pdf/2605.18875v1)

**作者:** Alberto Dennunzio `[一作]` (University of Milano-Bicocca), Luca Mariot `[通讯]` (University of Twente)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5069577574)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究了在非线性双可逆元胞自动机（BCA）生成的拉丁方中，主对角线是否构成一条横截面，并给出了判定条件。

**💡 创新点**

首次将主对角线横截面与生成函数诱导的周期性边界CA可逆性等价，提供了非线性BCA横截面问题的新理论视角。

**🔧 技术方法**

结合元胞自动机理论、拉丁方性质与可逆CA理论，并使用全穷搜索算法检验生成函数的可逆性。

**📊 数据集**

通过穷举搜索在不同直径d（≤6）的所有生成函数（总计65,536种）进行可逆性检测；未使用公开数据集。

**📈 对比分析**

实验对比可逆与非可逆生成函数数量，发现直径≤5仅线性规则可逆，而直径6出现大量非线性可逆规则；实验耗时可接受，展示了方法的可行性。

**⚠️ 局限性**

仅限于主对角线横截面，未完成完整N个横截面或正交伴侣的判定；研究仅到直径6，无法推广至更大直径；计算资源限制导致搜索规模受限。

---

## 290. Harnessing Self-Supervised Features for Art Classification

**arXiv ID:** 2605.18974 | [PDF](https://arxiv.org/pdf/2605.18974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 291. Chessformer: A Unified Architecture for Chess Modeling

**arXiv ID:** 2605.19091 | [PDF](https://arxiv.org/pdf/2605.19091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 292. Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method

**arXiv ID:** 2605.18766 | [PDF](https://arxiv.org/pdf/2605.18766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 293. When Web Apps Heal Themselves: A MAPE-K Based Approach to Fault Tolerance and Adaptive Recovery

**arXiv ID:** 2605.19261 | [PDF](https://arxiv.org/pdf/2605.19261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 294. Super-linear Lower Bounds for CSP Non-Redundancy via Shrinking Instances

**arXiv ID:** 2605.19055 | [PDF](https://arxiv.org/pdf/2605.19055v1)

**作者:** Joshua Brakensiek `[一作]` (University of California), Magnus Wahlström `[通讯]` (University of London)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究CSP谓词的非冗余性(NRD)并提出超线性下界新技术

**💡 创新点**

通过超图投影与收缩实例框架，精确预测gadget降维产生的NRD增长

**🔧 技术方法**

使用超图投影、ℐ-子结构理论、SMT/ SAT求解器自动发现gadget映射

**📊 数据集**

无专门数据集，主要以理论构造和极值图（高girth图）为实例

**📈 对比分析**

相较于之前c-fgpp方法，本框架在收缩实例下实现了更强的超线性下界；实验示例显示可达Ω(n^{6/5})等

**⚠️ 局限性**

仍未能给出对核心谓词的非线性下界；缺乏适合的收缩实例与更高阶超图结构的构造

---

## 295. Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning

**arXiv ID:** 2605.18871 | [PDF](https://arxiv.org/pdf/2605.18871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 296. Progressive Autonomy as Preference Learning: A Formalization of Trust Calibration for Agentic Tool Use

**arXiv ID:** 2605.19151 | [PDF](https://arxiv.org/pdf/2605.19151v1)

**作者:** Changkun Ou `[一作]` `[通讯]`, Changkun Ou

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了一种基于高斯过程（GP）概率判别的策略网关，用于自动化代理工具使用时的信任校准，即判断何时应允许代理自主执行动作，何时需人工批准。

**💡 创新点**

创新点在于将信任校准视为偏好学习问题，借鉴Preferential Bayesian Optimization（PBO）的框架，提出了统一的三层决策（允许/拒绝/询问）与时间衰减核的非平稳建模，能够在稀疏反馈下自适应学习人类风险容忍阈值。

**🔧 技术方法**

核心技术包括：高斯过程分类（GP-probit）与拉普拉斯近似推断、结构化核（工具、上下文、时间）构建、基于不确定性采样的主动学习策略、以及在线变更点检测以捕捉突变风险。

**📊 数据集**

实验主要使用公开的 R-Judge 互动记录作为冷启动先验和核参数校准，随后通过自定义的模拟环境（18种工具×7任务×8资源敏感度）产生包含非平稳漂移和突变的合成反馈数据。

**📈 对比分析**

与独立学习基线（不利用相关性）对比，GP网关在验证阶段实现了约97%准确率、2.4%误允许率，自动决策率约68%；在包含突变的后期测试中准确率高达99.7%，并将人类干预次数约减至原先的一半，显示显著的安全性和效率提升。

**⚠️ 局限性**

主要局限在于缺乏真实的长期人类反馈数据以验证非平稳核与主动采样策略的有效性，且单纯的“问询区”不一定优于随机询问，尤其在类别不平衡时；未来需要真实实验与更精细的期望信息采样机制。

---

## 297. ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking

**arXiv ID:** 2605.19077 | [PDF](https://arxiv.org/pdf/2605.19077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 298. Robust Checkpoint Selection for Multimodal LLMs via Agentic Evaluation and Stability-Aware Ranking

**arXiv ID:** 2605.18852 | [PDF](https://arxiv.org/pdf/2605.18852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 299. DOTRAG: Retrieval-Time Reasoning Along Paths

**arXiv ID:** 2605.18760 | [PDF](https://arxiv.org/pdf/2605.18760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 300. Selective, Regularized, and Calibrated: Harnessing Vision Foundation Models for Cross-Domain Few-Shot Semantic Segmentation

**arXiv ID:** 2605.19340 | [PDF](https://arxiv.org/pdf/2605.19340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 301. Semantic-Enriched Latent Visual Reasoning

**arXiv ID:** 2605.19342 | [PDF](https://arxiv.org/pdf/2605.19342v1)

**作者:** Tianrun Xu `[一作]` (Tsinghua University), Jing Liu `[通讯]` (Zhongguancun Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了两阶段语义增强视觉潜在推理框架SLVR，能在潜在空间内进行更丰富、更稳健的视觉推理。

**💡 创新点**

创新点在于引入属性级语义监督构造语义丰富的区域潜在向量，并通过多查询组相对策略优化（M-GRPO）实现跨查询的潜在一致性。

**🔧 技术方法**

使用了视觉-语言大模型、特征对齐损失、M-GRPO强化学习和稳定正则化等技术来学习与对齐潜在表示。

**📊 数据集**

构建了SLV-Set（≈40万区域属性注解+80万多查询QA样本）和SV-QA（591对语义变异QA）作为训练与评测数据集。

**📈 对比分析**

在多项公开VQA基准（OKVQA、GQA、ChartQA等）以及SV-QA上，SLVR相较于传统视觉潜在推理方法提升了约10%–25%准确率，并在多查询一致性上取得显著优势。

**⚠️ 局限性**

局限在于仍依赖大型视觉语言模型与昂贵的属性级注释，且对极端视觉模糊或缺失信息的鲁棒性待进一步提升。

---

## 302. Leveraging I/O Stalls for Efficient Scheduling in ANNS

**arXiv ID:** 2605.19335 | [PDF](https://arxiv.org/pdf/2605.19335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 303. BrainDyn: A Sheaf Neural ODE for Generative Brain Dynamics

**arXiv ID:** 2605.19324 | [PDF](https://arxiv.org/pdf/2605.19324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 304. MetaRA: Metamorphic Robustness Assessment for Multimodal Large Language Model-based Visual Question Answering Systems

**arXiv ID:** 2605.19307 | [PDF](https://arxiv.org/pdf/2605.19307v1)

**作者:** Quanxing Xu `[一作]` (Macau University of Science and Technology), Chia-Wen Lin `[通讯]` (National Tsing Hua University)

**通讯引用:** 12895 | [OpenAlex ID](https://openalex.org/A5051264473)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MetaRA框架，用于评估多模态大型语言模型（MLLM）在视觉问答（VQA）系统中的鲁棒性。

**💡 创新点**

创新点在于利用变形关系（Metamorphic Relations）生成多样化的图像-问题变体，能在无需真实答案的情况下检测模型在不同语义维度下的弱点。

**🔧 技术方法**

使用了变形关系（MT）技术、图像变换工具（OpenCV、Pillow、Stable Diffusion）、文本改写模型（transformer）、多模态推理框架以及失败率（FailRate）等评估指标。

**📊 数据集**

数据集包括KBVQA任务的E-VQA、InfoSeek，以及OCR-VQA任务的DocVQA、InfoVQA、ChartQA、TextVQA。

**📈 对比分析**

与传统准确率和MetaVQA的比较显示，MetaRA能更细粒度揭示模型在局部扰动、隐含信息和视觉-文本对齐等方面的弱点，整体失败率更低，证明其更全面的鲁棒性评估能力。

**⚠️ 局限性**

局限性包括：在文本丰富的OCR-VQA中表现仍较差，局部扰动导致高失败率；跨任务鲁棒性差异大；对超大规模模型的可扩展性未充分验证；仅覆盖四类MR，未能捕捉更广泛的变形情况。

---

## 305. Sample-Efficient Misconfiguration Classification for Network Resilience in Wireless Communications

**arXiv ID:** 2605.19303 | [PDF](https://arxiv.org/pdf/2605.19303v1)

**作者:** Xin Hao `[一作]` (University of Technology Sydney), Raymond Owen `[通讯]` (University of Technology Sydney)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对无线通信网络中模板化协议误配置的误配置分类问题，并设计了 EtaGATv2 算法实现高效样本利用的误配置识别。

**💡 创新点**

在 GATv2 的基础上引入边类型感知和动态注意机制，能够捕捉非均匀症状传播并提取协议特定特征，实现线性复杂度和 50% 样本效率提升。

**🔧 技术方法**

基于图注意力网络（GATv2）、动态注意机制、边类型感知变换、聚合与 MLP 分类，配合随机误配置注入训练。

**📊 数据集**

在基于 BGP/OSPF 仿真的合成网络上生成三组数据集（基线、中等规模、真实运营商拓扑），以及 Internet Topology Zoo 实际拓扑。

**📈 对比分析**

与 GAT、GATv2 和 EtaGAT 做对比，采用 400 轮训练、Adam 优化，结果表明 EtaGATv2 在训练样本 50% 以内即可达到 80% 准确率，零样本验证上在大规模拓扑也保持最高精度。

**⚠️ 局限性**

目前仅考虑单一误配置、简化误配置模式，未处理多重并发误配置、复杂运营商规则映射，且实验仍局限于仿真和有限真实拓扑。

---

## 306. Distributionally Robust Games via Coherent Risk Measures

**arXiv ID:** 2605.19302 | [PDF](https://arxiv.org/pdf/2605.19302v1)

**作者:** Bharat Gangwani `[一作]` (Independent Researcher), Arunesh Sinha `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在数据驱动环境下，使用一致性风险测度（如均值-半偏差、均值-偏差、CVaR）构建分布鲁棒博弈模型，探讨其存在性、连续性、等价性以及均衡计算方法。

**💡 创新点**

创新点在于：① 将一致性风险测度与分布鲁棒理论等价化，直接把风险敏感性嵌入玩家偏好；② 证明此类博弈既非传统矩阵游戏也非一般连续游戏，属于中间结构；③ 对其均衡计算给出PPAD复杂度下的完整分析；④ 提出多线性补数规划（MLCP）框架求解均衡，指出传统Lemke–Howson算法不适用。

**🔧 技术方法**

使用的技术包括：一致性风险测度的对偶表述、Kakutani固定点定理、对偶分解与补数规划、PPAD复杂度证明、以及实验中的求解器（如针对MLCP的求解工具）。

**📊 数据集**

实验使用人工生成的游戏样本：小规模的协调游戏、囚徒困境、CVaR非零和游戏；通过给定的K个样本构造经验分布，不涉及公开真实数据集。

**📈 对比分析**

通过与经验博弈（即仅使用经验均值的矩阵游戏）对比，实验表明风险厌恶（更高的γ）能够提升样本外均值表现、降低方差，并在某些游戏中提供尾部概率保证；结果在论文的图表中呈现，显示风险参数与稳健性指标之间的关系。

**⚠️ 局限性**

局限性包括：① 仅考虑一轮同步博弈，未扩展到动态或重复博弈；② 由于连续性质，传统的相关均衡定义和Lemke–Howson算法不可直接应用；③ 实验规模有限，缺乏对比其他分布鲁棒博弈算法的定量评估；④ 对大规模多玩家问题的可扩展性尚未解决。

---

## 307. Domain-Adaptive Communication-Rate Optimization for Sim-to-Real Humanoid-Robot Wireless XR Teleoperation

**arXiv ID:** 2605.19293 | [PDF](https://arxiv.org/pdf/2605.19293v1)

**作者:** Caolu Xu `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 49269 | [OpenAlex ID](https://openalex.org/A5100447820)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了无线XR遥操作中从仿真到真实的通信速率优化，提出了集成采样、传输、插值和重建的完整框架，并通过PAC‑Bayes分析指导域自适应的PPO策略学习。

**💡 创新点**

创新点包括：①基于隐空间密度比估计的PAC‑Bayes通用性表述；②分阶段的编码器预热、密度比加权PPO和信任域微调训练流程；③在仿真-真实迁移下实现能耗与重建误差的最优折中。

**🔧 技术方法**

采用隐空间MMD预热、uLSIF/KLIEP密度比估计、Proximal Policy Optimization（PPO）与权重、信任域正则化、无线通道模型与等效速率模型以及经验风险评估等技术。

**📊 数据集**

使用公开的Humanoid Everyday数据集（Apple Vision Pro人类动作与Unitree G1机器人真实轨迹）以及在Unitree_sim_isaaclab中构建的仿真数据集。

**📈 对比分析**

与PPO‑MMD、无加权PPO和全速率传输等基线对比；在不同信道增益下，DR‑PPO（uLSIF/KLIEP）显著降低重建误差，通信能量约为未加权PPO的一半，且在严重信道衰减时仍保持较低能耗。

**⚠️ 局限性**

局限性：需要离线真实轨迹来估计密度比，模型对真实数据质量敏感；密度比估计和编码器微调的计算开销较大；实验仅在特定的人形机器人和桌面抓取任务上验证，尚未评估在更复杂环境或多任务场景中的泛化能力。

---

## 308. FPED: A Functional-Network Prior-Guided Mixture-of-Experts Framework for Interpretable Brain Decoding

**arXiv ID:** 2605.19279 | [PDF](https://arxiv.org/pdf/2605.19279v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 309. iGSP:Implicit Gradient Subspace Projection for Efficient Continual Learning of Vision-Language Models

**arXiv ID:** 2605.19301 | [PDF](https://arxiv.org/pdf/2605.19301v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 310. UAV-Assisted Cooperative Edge Inference for Low-Altitude Economy via MoE-based Hierarchical Deep Reinforcement Learning

**arXiv ID:** 2605.19290 | [PDF](https://arxiv.org/pdf/2605.19290v1)

**作者:** Wenhao Zhuang `[一作]` (Hong Kong Polytechnic University), Xianghao Yu `[通讯]` (City University of Hong Kong)

**通讯引用:** 6140 | [OpenAlex ID](https://openalex.org/A5028609226)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种无人机辅助的协同边缘推理框架，实现无人机在完成低空经济任务的同时为地面设备提供AI推理服务。

**💡 创新点**

创新点包括将轨迹规划与推理任务分离的双层深度强化学习架构，并结合Mixture‑of‑Experts模型将高维离散决策拆分为门控+专家网络，从而降低动作空间维度并提升学习效率。

**🔧 技术方法**

采用了层次化深度强化学习（HDRL）与MoE、Gumbel‑Softmax、经验回放以及统一评估器的组合，解决了多时刻尺度、混合连续离散动作以及部分可观测的决策问题。

**📊 数据集**

在STL‑10图像分类数据集上进行训练与测试，采用ResNet‑18分割后端模型实现中间特征压缩。

**📈 对比分析**

与固定轨迹、贪婪推理、单体HDRL、以及不考虑不确定性的HDRL‑UE等基线对比，HDRL‑MoE在允许轨迹偏差下平均提升约2%–4%的分类准确率，并显著提高成功下发比例。

**⚠️ 局限性**

主要局限在于仅考虑单架UAV且不处理多UAV协同与复杂空域约束；此外，系统假设通道为自由空间模型，忽略了实际环境中的阻挡与多径影响。

---

## 311. CompoSE: Compositional Synthesis and Editing of 3D Shapes via Part-Aware Control

**arXiv ID:** 2605.19350 | [PDF](https://arxiv.org/pdf/2605.19350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 312. HalluWorld: A Controlled Benchmark for Hallucination via Reference World Models

**arXiv ID:** 2605.19341 | [PDF](https://arxiv.org/pdf/2605.19341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 313. Retrieval-Augmented Linguistic Calibration

**arXiv ID:** 2605.19344 | [PDF](https://arxiv.org/pdf/2605.19344v1)

**作者:** Yi-Fan Yeh `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 22154 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于Beta分布的语言信心分布模型，并设计了Faithfulness Divergence（FD）指标来衡量语言信心与真值之间的“惊讶度”，进一步构建了Retrieval-Augmented Linguistic Calibration（RALC）流水线，通过在信心分布均值上做Platt缩放并检索匹配的hedging表达，最终实现了把校准后的数值信心转换为更自然、可信的语言输出。

**💡 创新点**

核心创新在于①将语言信心视为可变分布而非单一数值；②引入FD作为实例级别的“忠实度”评估，利用信息论的KL和样本有效数加权；③提出轻量级的RALC框架，兼容多种信心信号并通过检索增强的重写实现语言级校准。

**🔧 技术方法**

技术手段包括：Beta分布参数化、Platt缩放（在分布均值上做逻辑回归）、1-Wasserstein距离检索、检索增强的LLM重写、token probability与semantic uncertainty的分布化、以及多模型评估器对语言信心的模拟。

**📊 数据集**

实验覆盖五大开源LLM（GPT-OSS-20B、Llama‑3.1‑8B、Qwen3‑8B、Mistral‑7B、Gemma‑4‑31B）与三大数据集（MMLU、SQuAD 2.0、TruthfulQA）。

**📈 对比分析**

与传统温度/Platt缩放、黑盒prompt hedging、以及无检索的“Direct Beta‑Guided Rewrite”对比，RALC在信心校准误差（ECE）和FD均显著下降，提升幅度高达≈60%；同时保持或提升内容保真度与信心-语言相关性。

**⚠️ 局限性**

局限性主要包括：校准效果受上游信心信号质量限制；semantic uncertainty虽表现最好但计算成本高；hedging词典覆盖有限，未能涵盖所有领域特定的模棱两可表达；对不同受众的自适应能力尚待研究。

---

## 314. Agentic Trading: When LLM Agents Meet Financial Markets

**arXiv ID:** 2605.19337 | [PDF](https://arxiv.org/pdf/2605.19337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 315. SWEET: Sparse World Modeling with Image Editing for Embodied Task Execution

**arXiv ID:** 2605.19319 | [PDF](https://arxiv.org/pdf/2605.19319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. Inference-Time Scaling in Diffusion Models through Iterative Partial Refinement

**arXiv ID:** 2605.19317 | [PDF](https://arxiv.org/pdf/2605.19317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 317. Matérn Noise for Triangulation-Agnostic Flow Matching on Meshes

**arXiv ID:** 2605.19305 | [PDF](https://arxiv.org/pdf/2605.19305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 318. Skinned Motion Retargeting with Spatially Adaptive Interaction Guidance

**arXiv ID:** 2605.19355 | [PDF](https://arxiv.org/pdf/2605.19355v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 319. SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models

**arXiv ID:** 2605.19357 | [PDF](https://arxiv.org/pdf/2605.19357v1)

**作者:** Yiyang Gu `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 21447 | [OpenAlex ID](https://openalex.org/A5100447284)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于科学本体的自动化自定义基准构建框架，能够将大规模科学语料划分为细粒度可复用的知识单元，并根据用户需求自动检索、筛选并生成多项选择题基准。

**💡 创新点**

创新点在于①将科学知识组织为可重用的本体驱动知识单元；②训练小型标签器实现语料到知识单元的自动映射；③采用多模型投票+二分搜索+代理子集选择的高效检索与筛选流程；④无需专家标注即可生成与专家基准高度一致的基准。

**🔧 技术方法**

使用技术包括本体驱动的知识单元选择、LLM辅助标签器训练、投票式多模型共识、二分搜索筛选、Wasserstein距离最小化的聚类子集选择、RoBERTa编码、LLM生成多项选择题及其干扰项。

**📊 数据集**

使用了642个本体知识单元、规模化科学问答语料（N条）、ChemBench、MMLU-Pro医疗子集、GPQA Diamond、IfBench、SimpleQA、MedQA等基准，以及人工抽样化学问题用于人工评价。

**📈 对比分析**

通过与通用/领域基准（GPQA、IfBench、SimpleQA、MMLU、MedQA）、GPT-5、Embedding等对比，在11个化学/医疗子任务中与专家基准的Spearman/Kendall相关性最高（约0.86–0.89），显著优于传统基准；在未有基准的Pericyclic Reaction案例中亦得到高一致性与专家认可。

**⚠️ 局限性**

局限在于现有本体主要覆盖生物医学与化学，未覆盖数学、理论物理等学科；以及部分知识单元的语料覆盖不足导致数据稀疏，需要扩展本体和持续更新语料。

---

## 320. PAVE: A Cognitive Architecture for Legitimate Violation in Generative Agent Societies

**arXiv ID:** 2605.19351 | [PDF](https://arxiv.org/pdf/2605.19351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 321. IMLJD: A Computational Dataset for Indian Matrimonial Litigation Analysis

**arXiv ID:** 2605.19346 | [PDF](https://arxiv.org/pdf/2605.19346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 322. RE-VLM: Event-Augmented Vision-Language Model for Scene Understanding

**arXiv ID:** 2605.19329 | [PDF](https://arxiv.org/pdf/2605.19329v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 323. RoboJailBench: Benchmarking Adversarial Attacks and Defenses in Embodied Robotic Agents

**arXiv ID:** 2605.19328 | [PDF](https://arxiv.org/pdf/2605.19328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 324. STAR-PólyaMath: Multi-Agent Reasoning under Persistent Meta-Strategic Supervision

**arXiv ID:** 2605.19338 | [PDF](https://arxiv.org/pdf/2605.19338v1)

**作者:** Jiaao Wu `[一作]` (Tsinghua University), Yinpeng Dong `[通讯]` (Tsinghua University)

**通讯引用:** 8566 | [OpenAlex ID](https://openalex.org/A5068755794)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出STAR-PólyaMath多代理框架，解决长链数学推理中的幻觉、记忆碎片化和工具使用失衡问题。

**💡 创新点**

引入持久Meta-Strategist监督，结构化Reasoner-Verifier对话和可追溯的探索-计划-执行-回溯循环。

**🔧 技术方法**

使用Python无推理调度器、多角色LLM（Reasoner、Verifier、Meta-Strategist），结构化辩论、回溯与重规划，Python代码执行与逻辑门检验。

**📊 数据集**

在AIME 2025/26、Putnam 2025、IMO 2025、USAMO 2026、HMMT 2026、MathArena Apex/Shortlist等竞赛题库上进行评测。

**📈 对比分析**

与现有最强闭源和开源多代理模型对比，STAR-PólyaMath在所有八大竞赛基准上均夺冠；Apex 2025 93.75% 高于GPT‑5.5的80.21%，AIME、Putnam、HMMT完美得分。

**⚠️ 局限性**

计算成本高、验证依赖自然语言与Python执行、基准趋于饱和，难以衡量更高阶开放式问题。

---

## 325. MOCHA: Multi-Objective Chebyshev Annealing for Agent Skill Optimization

**arXiv ID:** 2605.19330 | [PDF](https://arxiv.org/pdf/2605.19330v1)

**作者:** Md Mehrab Tanjim `[一作]` (Adobe Research), Sunav Choudhury `[通讯]` (Adobe Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MOCHA，一个基于 Chebyshev 标量化和指数退火的多目标 LLM 技能优化框架。

**💡 创新点**

创新点在于将 Chebyshev 标量化与超体积贡献（HVC）探索相结合，并通过阈值退火实现探索-利用平衡，从而覆盖非凸 Pareto 前沿并突破单目标优化器的瓶颈。

**🔧 技术方法**

采用 Chebyshev 标量化、超体积贡献(HVC)度量、指数退火调度、LLM 反射式改写与多字段技能结构化突变等技术。

**📊 数据集**

使用六个多领域技能数据集：GPQA、TheoremQA、HoVer、HotpotQA、FEVER 和 DebugBench。

**📈 对比分析**

与 TextGrad、ProTeGi、GEPA 等基线比较，MOCHA 在 4/6 任务实现突破，平均相对正确率提升 7.5%，最大提升 14.9%，并且发现的 Pareto 前沿点数是基线的两倍。

**⚠️ 局限性**

局限性包括对低冲突任务效果有限、退火调度为固定指数式、以及对特定平台 SKILL.md 约束的依赖，未来可考虑自适应退火与跨平台通用性。

---

## 326. TextAlign: Preference Alignment for Text Rendering with Hierarchical Rewards

**arXiv ID:** 2605.19320 | [PDF](https://arxiv.org/pdf/2605.19320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 327. Exploring and Developing a Pre-Model Safeguard with Draft Models

**arXiv ID:** 2605.19321 | [PDF](https://arxiv.org/pdf/2605.19321v1)

**作者:** Hongyu Cai `[一作]` (Purdue University), Z. Berkay Celik `[通讯]` (Purdue University)

**通讯引用:** 755 | [OpenAlex ID](https://openalex.org/A5005376753)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了从大型语言模型向小型语言模型的 jailbreak 迁移性，并基于此设计了一种利用小模型的前置审查策略。

**💡 创新点**

创新点在于将 jailbreak 的迁移性转化为防御资产，使用小模型生成草稿回答并结合现有后置审查器进行聚合判定。

**🔧 技术方法**

采用投机式推理（speculative inference）、小模型（如 OPT-125M、SmolLM-135M 等）、post‑model guard LlamaGuard‑2‑8B 进行安全分类，以及阈值投票聚合。

**📊 数据集**

使用 RPAB（改进版 AdvBench）生成的 jailbreak prompt、Just‑Eval benchmark 的正向对话、以及 GCG、AutoDAN、PAIR 等自动生成的攻击样本。

**📈 对比分析**

与基线预模型 guard（LlamaGuard‑2‑pre）和后模型 guard（LlamaGuard‑2‑post）对比，平均降低 32.4% 的防御失败率，在多数场景下性能与后模型 guard 相当，且耗时下降约 95%。

**⚠️ 局限性**

局限性包括对自适应攻击的未知鲁棒性、草稿模型推理仍有一定计算开销、以及不同模型/攻击场景下的迁移率不均匀。

---

## 328. ContextFlow: Hierarchical Task-State Alignment for Long-Horizon Embodied Agents

**arXiv ID:** 2605.19314 | [PDF](https://arxiv.org/pdf/2605.19314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 329. Language models struggle with compartmentalization

**arXiv ID:** 2605.19284 | [PDF](https://arxiv.org/pdf/2605.19284v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 330. A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation

**arXiv ID:** 2605.19316 | [PDF](https://arxiv.org/pdf/2605.19316v1)

**作者:** Seonjeong Hwang `[一作]` (Pohang University of Science and Technology), Gary Geunbae Lee `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 1101 | [OpenAlex ID](https://openalex.org/A5045508648)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个多代理框架（MAFIG）用于在阅读理解多项选择题生成中满足多维特征约束，并通过构造难度校准的约束序列实现精细难度控制。

**💡 创新点**

创新点在于：①引入多代理协同与评估循环实现对多维特征的严格满足；②提出基于理论与实验验证的难度校准特征约束序列构造方法；③通过迭代修订显著提升约束满足率与难度对齐度。

**🔧 技术方法**

采用大语言模型（Qwen3-32B、GPT‑5）作为生成与评估代理；RAG+检索增强的重写子代理；Planner、Rewriter、Editor、Refiner等多代理模块；基于Chain‑of‑Thought的LLM难度判定（DAS）和多轮迭代修订。

**📊 数据集**

以Brown语料库中的40篇文档（10类，前50句）为源文本，共生成320道多项选择题，用以验证框架在不同难度层级下的表现。

**📈 对比分析**

与单次提示的Level‑based与Feature‑based两种基线进行对比；MAFIG在约束满足率（SR）达到92.29%，DAS显著高于基线（0.5226 vs 0.276），人类评估CAR达到76.19%，表明在特征满足与难度对齐方面表现优异。

**⚠️ 局限性**

局限性包括：仅针对MCFI题型，未覆盖其他阅读理解格式；缺乏对考生实际表现的心理测量验证；迭代修订导致较高的计算成本和延迟；对特征定义与评估器质量高度依赖。

---

## 331. Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR

**arXiv ID:** 2605.19282 | [PDF](https://arxiv.org/pdf/2605.19282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 332. How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence

**arXiv ID:** 2605.19309 | [PDF](https://arxiv.org/pdf/2605.19309v1)

**作者:** Yue Chen `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2330 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套轻量化、输出级的结构感知文档布局分析（DLA）鲁棒性审计框架，拆分为探针空间、策略空间和诊断空间三大模块；

**💡 创新点**

通过发现并克服“足迹偏差”，提出Block‑Level Structural Loss Rate (B‑SLR)、细粒度曝光描述子和路径归因方法，实现从面积足迹到结构层面鲁棒性评估的根本转变；

**🔧 技术方法**

利用控制视觉扰动、基于IoU与文本一致性的B‑SLR、Granularity‑aware Exposure Descriptors、路径归因分解，并通过回归与Spearman相关性检验诊断信度；

**📊 数据集**

在PubLayNet与DocLayNet共计1000页验证集上，对MinerU与PP‑StructureV3两款基于框+文本的解析器进行实验；

**📈 对比分析**

与传统面积足迹、OCR CER与检测mAP对比，B‑SLR与OCR CER的R²达到0.73/0.92，曝光描述子能区分遮挡与拓扑失败，结构化探针在相同面积下比面积匹配扰动导致更大QA/检索性能下降；

**⚠️ 局限性**

仅覆盖两款解析器，缺乏对端到端或生成式模型的适配；诊断仅基于输出层，无法完整白盒分析内部机制；对不同域和更多解析器的普适性仍需进一步验证。

---

## 333. Cross-Paradigm Knowledge Distillation: A Comprehensive Study of Bidirectional Transfer Between Random Forests and Deep Neural Networks for Big Data Applications

**arXiv ID:** 2605.19299 | [PDF](https://arxiv.org/pdf/2605.19299v1)

**作者:** Mahdi Naser Moghadasi `[一作]` `[通讯]` (BrightMind AI Research), Mahdi Naser Moghadasi (BrightMind AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过双向蒸馏框架，将随机森林（RF）与深度神经网络（DNN）之间的知识进行相互迁移，旨在兼顾可解释性与表达能力。

**💡 创新点**

创新点包括：① 进阶多阶段蒸馏策略，逐步细化知识迁移；② 多教师集成蒸馏，利用不同树模型的多样性；③ 不确定性感知蒸馏，结合集成方差和蒙特卡罗 Dropout 提升可靠性。

**🔧 技术方法**

采用的技术主要有：温度缩放的 soft‑target 蒸馏、交叉熵与 KL 损失的加权融合、随机森林对增强数据的软标签训练、进阶多阶段网络结构、权重学习的多教师蒸馏、以及不确定性量化方法。

**📊 数据集**

实验数据集涵盖 6 个公开数据集，分别为乳腺癌、葡萄酒质量、手写数字、合成不平衡分类、加利福尼亚住房回归与非线性回归，样本量从 569 到 20,640。

**📈 对比分析**

与单一模型基线、传统蒸馏以及单阶段蒸馏方法相比，实验显示多阶段蒸馏在分类任务中取得最高 98.13% 的准确率，回归任务中获得 92.6% 的 R² 分数；推理时间上保持与树模型相近，显著低于传统 DNN。

**⚠️ 局限性**

局限性主要体现在：① 未在大于 1 亿样本的超大规模数据集上验证；② 仅针对单模态（表格）数据；③ 未针对流式数据的动态蒸馏或联邦学习场景进行探究。

---

## 334. DECOR: Auditing LLM Deception via Information Manipulation Theory

**arXiv ID:** 2605.19270 | [PDF](https://arxiv.org/pdf/2605.19270v1)

**作者:** Linyue Cai `[一作]` (University of Wisconsin Madison), Sharon Li `[通讯]` (University of Wisconsin Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Decor 框架，对大语言模型产生的回答进行信息操纵维度的细粒度审计，检测欺骗行为。

**💡 创新点**

将信息操纵理论（IMT）与多代理系统相结合，实现对四维操纵（数量、质量、关联、方式）的细粒度评分并聚合成全局欺骗指数，从而兼顾可解释性与高精度。

**🔧 技术方法**

采用三阶段多代理方法：单位构造 Agent、IMT 审计 Agent 和全局欺骗指数聚合；使用文本层面评分、策略影响权重、平均聚合等技术，所有处理均基于文本，无需内部模型访问。

**📊 数据集**

使用单回合 DeceptionBench、双回合 OpenDeception 两大基准数据集；并对 600 条 DeepSeek-R1 生成的实例进行人工标注，形成金标准。

**📈 对比分析**

与零/少量提示、DeceptionBench、CoT Red-Handed、Constitutional Monitor 等基线对比；单回合 AUROC 达到 0.935、思维层面 0.920，优于所有基线；多回合 AUROC 分别为 0.654（回答）和 0.772（思维），同样超过所有基线。

**⚠️ 局限性**

仍依赖人工标注的金标准，难以覆盖所有隐蔽欺骗；对极细微语言暗示或模糊推理的检测能力有限；跨语言、跨领域的普适性尚未充分验证。

---

## 335. CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs

**arXiv ID:** 2605.19269 | [PDF](https://arxiv.org/pdf/2605.19269v1)

**作者:** Han Guo `[一作]` (Massachusetts Institute of Technology), Tri Dao `[通讯]` (Princeton University)

**通讯引用:** 2842 | [OpenAlex ID](https://openalex.org/A5091734792)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于GEMM加后续程序（epilogue）的GPU内核抽象，重新参数化Transformer中的大部分内存受限计算，减少显存访问。

**💡 创新点**

创新点在于将多种内存受限操作（归一化、激活、残差、分布式损失等）通过可组合的epilogue原语与高性能GEMM主循环融合，且保留了易于程序化的抽象，支持人类与LLM共同编写。

**🔧 技术方法**

使用CUDA、CuTeDSL实现GEMM主循环与epilogue原语，利用Tensor Core、共享内存、TileMemoryAccelerator等技术，并通过Claude Code自动生成内核。

**📊 数据集**

在LLaMA‑3风格的1B、7B、70B模型上（batch 16k tokens）进行实验，GPU使用NVIDIA H100。

**📈 对比分析**

与cuBLAS（Tensor Core）、Liger Kernels、FlashInfer、QuACK等基准进行对比，单核和块级别的GEMM‑epilogue实现平均提升数十%到几倍，几乎达到或接近GEMM原始峰值。

**⚠️ 局限性**

局限性：仅针对常见Transformer结构，尚未支持多GPU分布式执行，重参数化可能模糊模块边界和算法语义，导致与框架层级集成困难。

---

## 336. What Makes a Representation Good for Single-Cell Perturbation Prediction?

**arXiv ID:** 2605.19343 | [PDF](https://arxiv.org/pdf/2605.19343v1)

**作者:** Wenkang Jiang `[一作]` (Adelaide University), Javen Qinfeng Shi `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 PerturbedVAE 框架，用于单细胞扰动预测，并验证了扰动抑制假设。

**💡 创新点**

创新点在于明确扰动抑制假设，并设计对抗对齐与因果结构的变分自编码器，显式分离扰动无关与扰动相关信息。

**🔧 技术方法**

采用了变分推断、对比对齐正则、因果生成模型以及可辨识性分析等技术，使用 ELBO 与对齐损失共同训练。

**📊 数据集**

在合成数据和 Perturb-seq 大规模基因编辑实验数据（如 Norman2019、Perturb-seq 等）上进行评估。

**📈 对比分析**

与 FMs（scFoundation、UCE、Geneformer、STATE）以及现有因果表示学习方法（Discrepancy‑VAE、SENA、sVAE+、SAMS‑VAE）对比，在单基因和组合基因扰动上均实现更低 RMSE、更高 R²，尤其在 OOD 组合扰动上显著提升。

**⚠️ 局限性**

局限性在于需要足够多的扰动环境与对齐样本，且对非线性交互的捕捉能力有限，理论可辨识性假设对实际数据的满足程度仍有限。

---

## 337. Quantum-Enhanced Distributed Sensor Fusion: Lower Bounds on Aggregation from Projection Noise to Heisenberg-Limited Byzantine-Tolerant Networks

**arXiv ID:** 2605.19327 | [PDF](https://arxiv.org/pdf/2605.19327v1)

**作者:** Vasanth Iyer `[一作]` (Grambling State University), S. S. Iyengar `[通讯]` (Florida International University)

**通讯引用:** 8006 | [OpenAlex ID](https://openalex.org/A5009505287)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在分布式量子传感器网络中，本文推导了统一的均方误差（MSE）下界，考虑了拜占庭式故障与退相干，并将经典容错融合算法与量子计量学结合；

**💡 创新点**

创新点包括：① 以纠缠可见度V与故障比例f/M为参数的两参数MSE下界，能够在标准量子极限(SQL)与海森堡极限(HL)之间连续插值；② 证明预测离群检测在量子域内相较Brooks‑Iyengar BFT恒定提供约2–4 dB的优势；③ 确定临界可见度V*，给出混合量子‑经典部署准则；④ 通过重构经典算法（Brooks‑Iyengar、SPOTLESS、预测离群）在量子融合中的对应角色，架起两者的桥梁；

**🔧 技术方法**

采用量子Cramér‑Rao界、带退相干的量子 Fisher 信息、置信区间构造、重叠函数、预测离群模型、SPOTLESS空间时间验证、随机森林相似度、卡尔曼滤波、贝叶斯加权融合、复杂度分析、数据清洗树等技术；

**📊 数据集**

使用两类数据集：① 1000原子/传感器、η=0.1 rad/°C 的 Monte‑Carlo 合成数据；② 真实的 Intel Berkeley Lab Mica2Dot 54 节点、约230万条温度/湿度/光照/电压测量；

**📈 对比分析**

与简单平均、Brooks‑Iyengar BFT、预测离群及纠缠融合进行对比；结果显示：SQL 级别实现 1/√M 缩放，HL 级别实现 1/M 缩放；预测离群比 BFT 高 2–4 dB；纠缠融合在 6–10 节点聚类中比经典提升 20–27 dB，Monte‑Carlo 进一步验证理论下界；

**⚠️ 局限性**

局限性：退相干仅用单一可见度 V 的指数衰减模型；未做完整量子通道或密度矩阵模拟；未在实际量子传感器上实验验证；采用高斯噪声假设，忽略量子态制备误差与读取失真；仅考虑渐进性 1/M² 的海森堡极限，未捕捉有限样本的离散统计效应。

---

## 338. An Exterior Method for Nonnegative Matrix Factorization

**arXiv ID:** 2605.19325 | [PDF](https://arxiv.org/pdf/2605.19325v1)

**作者:** Qiujing Lu `[一作]` (UCLA), Vwani Roychowdhury `[通讯]` (UCLA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种外部NMF框架(eNMF)，先从无约束的SVD解开始，通过旋转逼近非负正交空间，然后用外部惩罚使因子可行，最后用HALS收敛到局部最优。

**💡 创新点**

将低秩逼近与非负性约束解耦，利用SVD旋转对称性快速得到全局或近全局的初始化，显著提升收敛速度与精度，且首次系统性展示不同NMF算法在几乎所有实验中收敛到几何相同的解。

**🔧 技术方法**

使用SVD、ADMM旋转、外部惩罚（负值修正）、层级交替最小二乘(HALS)、梯度下降、PBCD等技术。

**📊 数据集**

包含合成稠密/高维数据、Exact Factorization、Verb实体-动词关系、Audio谱图、Face像素以及大规模稀疏Reuters 804k×47k矩阵等多种真实与合成数据集。

**📈 对比分析**

与9种主流NMF算法（Mult、Grad-Mult、ALS、AO-ADMM、HALS、NMF-ADMM、A-HALS、NeNMF、FPGM、Vavasis）在81个初始化方案下做等时/等误差比较，eNMF在所有实验中均实现约30%更低重构误差、最高150%加速，并在音频、视觉、推荐等下游任务中提升10–50%性能。

**⚠️ 局限性**

仍依赖SVD初始化，对极大规模或在线场景适用性有限；若旋转矩阵无法将因子落入正交空间，仍需进一步迭代；算法仍受NMF NP‑hard性限制，理论上无法保证全局最优。

---

## 339. DynaTok: Temporally Adaptive and Positional Bias-Aware Token Compression for Video-LLMs

**arXiv ID:** 2605.19322 | [PDF](https://arxiv.org/pdf/2605.19322v1)

**作者:** Minyoung Park `[一作]` (LG Electronics), Sangjun Ahn `[通讯]` (LG Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DynaTok，一种训练-free、模型无关的 token 压缩框架，能在不重新训练的前提下，在时空维度上动态分配视觉 token，从而显著降低 Video-LLM 的计算与内存开销。

**💡 创新点**

创新点在于：① 通过两阶段动态预算分配（Temporal Budget Allocation + Spatial Budget Allocation）结合轻量化 EMA 内存，捕捉长时序变化与空间多样性；② 在空间选择中引入空间记忆与余弦相似度惩罚，抑制视觉编码器的位置信息偏差，实现 bias‑aware 的 token 选取；③ 完全兼容 FlashAttention，且不需要内部注意力矩阵。

**🔧 技术方法**

使用了 EMA 内存、激活基注意力图、语义重要性评分、余弦相似度惩罚、块级 token 选择、基于块的空间分配以及与 FlashAttention 兼容的轻量化实现。

**📊 数据集**

在四个 VideoQA 基准上进行评估：MVBench、LongVideoBench、MLVU、VideoMME。

**📈 对比分析**

与 FastV、VisionZip、DyCoke 等训练‑free 压缩方法对比，DynaTok 在 LLaVA‑OneVision 与 LLaVA‑Video 上的 10% token 保留率时保持 95.5% 的平均准确率，在 9.5%–25% 保留率下平均准确率提升 2–5%，显著优于其他基线方法。

**⚠️ 局限性**

局限性：仍可能在极端压缩下丢失细粒度时空信息；仅在 Video‑LLM 的问答任务上验证，未深入探究对生成任务或更复杂多模态推理的影响；需要进一步评估在更大规模、实时流媒体场景中的鲁棒性。

---

## 340. A Two-Phase Adaptive Balanced Penalty Method for Controllable Pareto Front Learning under Split Feasibility Conditions

**arXiv ID:** 2605.19306 | [PDF](https://arxiv.org/pdf/2605.19306v1)

**作者:** Nguyen Viet Hoang `[一作]` (Hanoi University of Science and Technology), Tran Ngoc Thang `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 194 | [OpenAlex ID](https://openalex.org/A5029410939)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种自适应平衡惩罚（ABP）算法，用于在拆分可行性条件下学习可控制的帕累托前沿（CPFL），并将其转化为两阶段可行性优先训练的超网络框架（ABP‑HyperMLP/HyperTrans）

**💡 创新点**

创新点包括：①将受限CPFL形式化为双层标量化拆分问题（BSSP），②设计可自适应的三向梯度惩罚结构并证明全序收敛；③引入线性化半空间凸代理实现图像可行性惩罚的全局下界；④提出期望可行性超体积（EFHV）度量整合质量与约束满足度；⑤两阶段可行性优先训练策略在超网络上实现实时可控多任务学习

**🔧 技术方法**

采用的技术包括：凸优化理论、Chebyshev标量化、拆分可行性问题（SFP）框架、线性化半空间代理、随机梯度下降与Robbins–Monro步长、Transformer/MLP超网络、两阶段加权损失、期望可行性超体积评估

**📊 数据集**

使用的基准数据集：五个多目标优化基准（CVX1–CVX3、ZDT1、ZDT2）以及三种多任务学习图像分类数据集（Multi‑MNIST、Multi‑Fashion、Fashion+MNIST）

**📈 对比分析**

与无约束CPFL基线和现有超网络方法（Hyper‑MLP、HyperTrans）对比，ABP‑HyperNet在约束满足率上从36–49%提升到87–100%，EFHV提高至2.3倍，且推理时间仅为传统求解器的1–3毫秒级；在多目标基准上，ABP求解器与解析解的MED均≤0.01，超网络在实时推理下保持相同精度

**⚠️ 局限性**

局限性包括：①收敛证明依赖于零间隙假设（零间隙或可通过数值验证）；②在非凸目标或大规模多目标（m≥4）场景下缺乏理论保证；③两阶段训练策略为经验性启发式，未在非凸随机梯度环境中提供正式收敛分析；④线性化半空间代理可能对非线性可行性映射的逼近误差影响整体性能

---

## 341. EviTrack: Selection over Sampling for Delayed Disambiguation

**arXiv ID:** 2605.19283 | [PDF](https://arxiv.org/pdf/2605.19283v1)

**作者:** Omer Haq `[一作]` `[通讯]` (Independent Researcher), Omer Haq (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aaccfe5c-6b26-4208-b23c-35331481e142` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在序列预测中通过维护轨迹假设并按证据进行选择的推理框架 EviTrack，以处理“延迟歧义”问题。

**💡 创新点**

创新点在于把轨迹级别的证据积累与适度选择相结合，避免传统粒子滤波器过早消除多模态假设，同时实现更快的歧义消除。

**🔧 技术方法**

技术核心包括：轨迹假设集管理、局部/全局剪枝、基于证据/联合/背景归一化的轨迹得分、预测时的轨迹混合以及与粒子滤波（SIS、BPF）的对比实验。

**📊 数据集**

使用自定义的双井口延迟歧义合成基准，该基准可精确计算真实后验并记录歧义时间。

**📈 对比分析**

与等计算预算的粒子滤波基线相比，EviTrack 在后歧义阶段预测对数似然提升约 52 倍、分支准确率从 0.58 提升至 0.99，显示出显著性能优势。

**⚠️ 局限性**

局限性包括对基准参数（K、C、剪枝间隔）敏感、缺乏一致性采样性质、只在低维合成数据上验证，尚未在真实高维或学习的世界模型中进行测试。

---

## 342. Lost in Interpretation: The Plausibility-Faithfulness Trade-off in Cross-Lingual Explanations

**arXiv ID:** 2605.19274 | [PDF](https://arxiv.org/pdf/2605.19274v1)

**作者:** Somnath Banerjee `[一作]` (Indian Institute of Technology Kharagpur), Animesh Mukherjee `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 3622 | [OpenAlex ID](https://openalex.org/A5020991141)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多语言LLM在不同语言输入下用英语生成解释的可信度问题，评估其对证据选择的影响；

**💡 创新点**

首次将报告语言作为实验变量，发现“可解释性-可信度权衡”现象：英语解释往往提升人类可接受度但降低模型决策的因果关联；

**🔧 技术方法**

采用提取式解释（证据span+自由文本）与ERASER式的全面性/充分性测评，结合BERTScore对跨语言语义相似度校验；

**📊 数据集**

使用三大任务的公开基准：e-SNLI（推理）、FEVER（事实核查）与HateXplain（社交语义）；每个任务分别翻译成中文、印地语、阿拉伯语、孟加拉语等四种语言；

**📈 对比分析**

与本地语言解释（L_native→L_native）和英语解释（L_native→EN）对比，发现英语解释在span一致性上略高，但全面性显著下降（平均下降5.7倍），充分性提升；模型准确率基本不变，验证了可信度下降非因理解差距；

**⚠️ 局限性**

局限性包括：仅评估提取式解释；翻译质量可能影响结果；跨语言对齐仍受脚本差异限制；社交细微任务中英语解释整体性能下降，说明无法完全替代本地语言解释；

---

## 343. Neuron Incidence Redistribution for Fairness in Medical Image Classification

**arXiv ID:** 2605.19393 | [PDF](https://arxiv.org/pdf/2605.19393v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 344. When the Majority Votes Wrong, the Intervention Timing for Test-Time Reinforcement Learning Hides in the Extinction Window

**arXiv ID:** 2605.19444 | [PDF](https://arxiv.org/pdf/2605.19444v1)

**作者:** Hongxiang Lin `[一作]` (Meituan), Lei Wang `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TTRL-Guard 框架，对测试时强化学习中多数投票伪标签导致的错误传播进行动态干预，显著降低了问题损坏率并提升整体准确率。

**💡 创新点**

创新点在于发现并利用“正确答案消亡窗口”与“翻转率（FR）”这一无标签早期预警信号，三种无监督干预机制（FRS、MPS、RCSU）共同抑制错误共识并保护正面信号。

**🔧 技术方法**

使用无监督强化学习（GRPO）结合多数投票奖励，并在此基础上实现 FR 监控、奖励缩放、少数采样与稀疏更新等技术。

**📊 数据集**

在四个数学推理基准（AIME 2024/2025、AMC、MATH-500）和三种大语言模型（Llama‑3.2‑3B、Qwen2.5‑7B、Qwen3‑4B）上进行评估。

**📈 对比分析**

与 TTRL、CoVerRL、SCOPE 等无监督对比方法相比，TTRL‑Guard 在 Qwen2.5‑7B 上提升 54%（AIME 2025）并在 Qwen3‑4B 上获得 59.7% 的最高平均 Pass@1，显著降低问题损坏比例。

**⚠️ 局限性**

局限性包括仅在数学推理任务和 7B 参数规模模型上验证，未扩展至更大模型或其它可验证领域，且翻转率在极低初始准确率时信号有限。

---

## 345. Concept-Guided Noisy Negative Suppression for Zero-Shot Classification and Grounding of Chest X-Ray Findings

**arXiv ID:** 2605.19374 | [PDF](https://arxiv.org/pdf/2605.19374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. Backtracking When It Strays: Mitigating Dual Exposure Biases in LLM Reasoning Distillation

**arXiv ID:** 2605.19433 | [PDF](https://arxiv.org/pdf/2605.19433v1)

**作者:** Bing Wang `[一作]` (Jilin University), Jieping Ye `[通讯]` (Alibaba Group)

**通讯引用:** 40229 | [OpenAlex ID](https://openalex.org/A5010419481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的LLM推理蒸馏框架，动态监控学生生成轨迹，采用自适应安全边界容忍错误，并在轨迹偏离阈值时回溯到最近安全状态，再由教师模型介入纠正，从而解决传统离线与在线蒸馏的双重曝光偏差。

**💡 创新点**

创新点在于：① 用教师模型熵动态设定安全阈值，实现对学生探索的自适应容忍；② 通过信用分配回溯识别逻辑失效点，避免直接在错误上下文中干预；③ 在回溯点后拼接教师生成后缀，既保持对学生错误的暴露，又保证教师监督的有效性，兼顾覆盖与有效性两大难点。

**🔧 技术方法**

主要技术包括：自回归LLM推理、教师模型熵监控与价值函数评估、基于信用分配的状态回溯、后缀拼接式蒸馏以及最大似然估计训练。

**📊 数据集**

使用的主要数据集有：LIMO‑v2、AceReason、AIME24/25、MATH500、IFEval、GPQA 等多领域、多难度的推理与跨领域数据集。

**📈 对比分析**

与 SFT、RFT、ImitKD、SKD 等基线对比，在 Pass@8/4/1 等指标上平均提升约 3%–10%（如在 AIME24/25 上提升 7%+），尤其在长 CoT 任务上表现最为显著，证明该方法有效缓解双重曝光偏差。

**⚠️ 局限性**

局限性：① 对教师模型的质量和可靠性要求较高；② 需要手动调节安全阈值，参数敏感性较大；③ 回溯机制对极端错误的纠正仍有限；④ 与纯离线蒸馏相比，计算与存储成本仍显著，且在更大规模或多模态模型上的推广尚未验证。

---

## 347. Multi-Scale Generative Modeling with Heat Dissipation Flow Matching

**arXiv ID:** 2605.19371 | [PDF](https://arxiv.org/pdf/2605.19371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 348. HSCO-Bench: An Agent-Driven End-to-End Hardware-Software Co-design Benchmark for Systems-on-Chip

**arXiv ID:** 2605.19399 | [PDF](https://arxiv.org/pdf/2605.19399v1)

**作者:** Pei-Huan Tsai `[一作]` (Columbia University), Luca P. Carloni `[通讯]` (Columbia University)

**通讯引用:** 7538 | [OpenAlex ID](https://openalex.org/A5009992367)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HSCO-Bench，一个端到端的软件‑硬件协同设计基准，评估大型语言模型在异构SoC生成中的能力。

**💡 创新点**

首次在完整软件‑硬件堆栈上评估LLM，基于ESP开源SoC平台，提供10个多样化工作负载与自动化FPGA验证流程。

**🔧 技术方法**

采用LLM驱动的自动化设计、High‑Level Synthesis（HLS）、ESP SoC平台、网络芯片（NoC）与FPGA实现技术。

**📊 数据集**

使用10个开源机器学习与信号处理应用（如Autoencoder、BERT‑Tiny、MiniDeiT、TinyCLIP、MobileNetV4等）及对应加速器模板与任务说明。

**📈 对比分析**

通过比较LLM生成的SoC在CPU基准上的周期速度提升与FPGA资源利用率进行评估；在5个前沿模型中，Opus 4.6平均提升2.32×、资源占用2.22%，GPT‑5.4峰值提升16.22×但资源占用23.67%，整体显示LLM在协同设计仍处于挑战期。

**⚠️ 局限性**

局限性在于端到端协同仍难以实现，LLM在多加速器集成、硬件软件接口、资源约束等方面表现欠佳，生成的设计保守、资源利用不足，需进一步提升模型多步骤推理与硬件约束理解。

---

## 349. LatentBox: An Efficient Latent-First Storage System for AI-Generated Images

**arXiv ID:** 2605.19385 | [PDF](https://arxiv.org/pdf/2605.19385v1)

**作者:** Zirui Wang `[一作]` (University of Virginia), Yue Cheng `[通讯]` (University of Virginia)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5079187166)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个“latent-first”存储系统，将压缩的 diffusion 模型潜在向量作为持久存储对象，在读取时通过 GPU VAE 解码重构像素。

**💡 创新点**

创新点：① 第一次将压缩 latent 用作持久存储；② 采用双格式自适应缓存（图像层与 latent 层）并通过在线边际命中调优动态分配容量；③ 结合一致哈希 + 溢出调度，实现 GPU 负载均衡与缓存局部性的统一管理。

**🔧 技术方法**

核心技术包括：diffusion 模型的 VAE 解码、pcodec 无损压缩、TensorRT GPU 推理、Ray 分布式框架、CUDA Graph 加速以及 CPU‑GPU 级联流水线。

**📊 数据集**

使用 35 个月、2.07 B 次请求的生产访问日志（来自大型生成内容平台），以及基于 SD 3.5 生成的 150 K 张图像做实验。

**📈 对比分析**

通过与传统 PNG 存储、Decode‑All、单格式缓存等基线在平均/99 分位延迟、缓存命中率和存储占用等指标对比，结果显示持久存储空间减少 78.7%，平均读取延迟下降 17%，99 分位延迟下降 18%，长期成本可节省 60%+。

**⚠️ 局限性**

限制：受 GPU 解码延迟和存储网络带宽影响；需要持续可用的生成模型；高并发场景下 GPU 队列竞争仍是瓶颈；系统对不同模型或分辨率的泛化性需进一步验证。

---

## 350. BrepForge: Factorized B-rep Synthesis via Wireframe Composition and Boundary-Conditioned Surface Instantiation

**arXiv ID:** 2605.19411 | [PDF](https://arxiv.org/pdf/2605.19411v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 351. A complete discussion on fully reconfigurable, digital, scalable, graph and sparsity-aware near-memory accelerator for graph neural networks

**arXiv ID:** 2605.19405 | [PDF](https://arxiv.org/pdf/2605.19405v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 352. On-the-Fly Input Adaptation for Reliable Code Intelligence

**arXiv ID:** 2605.19365 | [PDF](https://arxiv.org/pdf/2605.19365v1)

**作者:** Ravishka Rathnasuriya `[一作]` (University of Texas at Dallas), Wei Yang `[通讯]` (University of Texas at Dallas)

**通讯引用:** 11242 | [OpenAlex ID](https://openalex.org/A5036689637)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在推理时即时输入适配框架，包括输入验证与语义保持的输入改写与潜在空间扰动，以提升代码语言模型的可靠性。

**💡 创新点**

创新点在于不修改模型参数，而是通过对输入进行语法语义保持的改写和潜在空间调优，结合验证信号进行搜索，形成模型无关、资源高效的可靠性提升方法。

**🔧 技术方法**

使用的技术包括不确定性度量（熵、MC Dropout、集成方差等）、输入空间语义保持改写、潜在空间梯度上升、扩散式生成引导、搜索策略（进化、约束解码）等。

**📊 数据集**

使用的数据集包括代码分类任务的漏洞检测、缺陷分类（CodeBERT、GraphCodeBERT）、生成任务的MBPP+、HumanEval+，以及预训练模型 DeepSeek-Coder-7B、CodeLlama-7B。

**📈 对比分析**

与基线模型对比，输入适配后分类准确率从63.36%提升到76.75%（+13.4%），生成任务的误判检测AUC普遍在0.5–0.66之间，表明原有不确定性指标效果有限。

**⚠️ 局限性**

主要限制是输入改写的计算开销高（≈50–60秒/样本），不确定性指标在代码任务上表现差，且目前仅在部分分类任务验证，生成任务仍需进一步验证。

---

## 353. When to Stop Reusing: Dynamic Gradient Gating for Sample-Efficient RLVR

**arXiv ID:** 2605.19425 | [PDF](https://arxiv.org/pdf/2605.19425v1)

**作者:** Yuchun Miao `[一作]` (Wuhan University), Lefei Zhang `[通讯]` (Wuhan University)

**通讯引用:** 13914 | [OpenAlex ID](https://openalex.org/A5024278302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对RLVR中样本重用导致的训练崩溃进行微观分析，发现并验证了“Disproportionate Weight Divergence (DWD)”现象，即性能下降同步出现LM Head权重的突变；随后提出理论证明，表明Head梯度范数下界于策略偏移，因而可实时监测灾难性策略漂移；基于此，设计了Dynamic Gradient Gating (DGG)，在检测到Head梯度异常时预拦截梯度并终止重用循环，从而实现安全样本重用；

**💡 创新点**

创新点在于首次系统性揭示DWD现象并提供理论解释，将Head梯度范数作为实时指示器；提出了基于Z-score的动态梯度门控机制DGG，能够在训练早期拦截有害梯度；在RLVR样本重用场景中实现了显著的样本效率与时钟速度提升。

**🔧 技术方法**

主要技术包括：GRPO（Group Relative Policy Optimization）框架；微观层级梯度分解与Jacobian分析；梯度范数与Pearson χ²策略偏移的下界证明；Z-score自适应检测；预优化器梯度截断与重用循环终止；实验中使用Adam优化器与动态阈值。

**📊 数据集**

实验使用了四类数学推理数据集（MATH500、AIME25、Minerva Math、Olympiad Bench）和三类代理任务（ALFWorld、WebShop、搜索增强QA），搜索QA进一步分为HotpotQA、2Wiki、MuSiQue、Bamboogle；模型为Qwen3‑4B‑Instruct‑2507与Qwen2.5‑7B‑Instruct。

**📈 对比分析**

与单次使用（Single‑Use）和Naive Sample Reuse基线对比，DGG在所有16种设置中都保持了单次使用的最终性能，同时在样本效率上提升2.0×–2.93×，在墙钟时间上提升1.31×–2.14×；Naive重用虽早期加速但易崩溃，DGG通过实时门控避免崩溃并进一步获得更高的加速。

**⚠️ 局限性**

局限性包括：分析与方法仅针对GRPO目标，尚未验证PPO或其他RL目标；阈值τ与最大重用K仍需经验调参；在极端大模型或任务中可能出现未见的梯度异常；实验范围局限于Qwen与Llama系列模型，未覆盖更广泛架构。

---

## 354. Conflict-Resilient Multi-Agent Reasoning via Signed Graph Modeling

**arXiv ID:** 2605.19418 | [PDF](https://arxiv.org/pdf/2605.19418v1)

**作者:** Longgang He `[一作]` (Harbin Institute of Technology), Chaozhuo Li `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 2723 | [OpenAlex ID](https://openalex.org/A5037831162)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Signed Graph的LLM多智能体推理框架，能够显式捕捉代理间的信任、冲突与中性关系并在推理过程中自适应处理冲突信号。

**💡 创新点**

创新点在于：1）将符号图（Signed Graph）引入多智能体系统，显式区分正向（信任）和负向（冲突）边；2）设计冲突感知的符号消息传递机制，在正负邻居之间交替更新正负表征；3）结构感知的加权聚合，以网络中正向支持的净强度决定最终投票。

**🔧 技术方法**

使用技术包括：符号图构造（基于相似度评估信任/冲突标记）、查询引导的代理选择（语义相关度+多样性+置信度权衡）、冲突感知的符号消息传递（分离正负表征并通过层级聚合），以及结构感知的加权聚合（根据符号图权重进行加权）。

**📊 数据集**

在六个基准数据集上进行实验，涵盖一般推理（MMLU、MMLU-Pro、GPQA、GSM8K、MultiArith）、数学推理（HumanEval、代码推理等）与域特定任务。

**📈 对比分析**

与多种单智能体（CoT、ComplexCoT、Self-Consistency、PHP）和多智能体（MoA、Self-MoA、Complete Graph、Random Graph、DyLAN、AutoGen、GPTSwarm、G-Designer、GoA）基线对比，平均准确率提升至约89.17%，在所有六个数据集上均保持最高或次高性能，显示出更强的鲁棒性与冲突抵抗能力。

**⚠️ 局限性**

局限性包括：1）需要额外的符号图构造与权重估计，增加前处理复杂度；2）对超参数（如λ、k、层数）敏感，需针对不同任务调优；3）在极端噪声或恶意代理比例极高时仍可能出现聚合偏差；4）目前仅在文本交互层实现，缺乏更低层次的向量级信息交换。

---

## 355. TIDE: Asymmetric Neural Circuits for Stabilized Temporal Inhibitory-Excitatory Dynamics

**arXiv ID:** 2605.19403 | [PDF](https://arxiv.org/pdf/2605.19403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 356. Generative Recursive Reasoning

**arXiv ID:** 2605.19376 | [PDF](https://arxiv.org/pdf/2605.19376v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 357. Accurate, Efficient, and Explainable Deep Learning Approaches for Environmental Science Problems

**arXiv ID:** 2605.19366 | [PDF](https://arxiv.org/pdf/2605.19366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 358. A Hybrid Cluster-Based Classification Model for Anomaly Detection in Unbalanced IoT Networks

**arXiv ID:** 2605.19451 | [PDF](https://arxiv.org/pdf/2605.19451v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 359. EmbGen: Teaching with Reassembled Corpora

**arXiv ID:** 2605.19394 | [PDF](https://arxiv.org/pdf/2605.19394v1)

**作者:** Arun K Lenin `[一作]` (Commonwealth Bank of Australia), Anna Leontjeva `[通讯]` (Commonwealth Bank of Australia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EmbGen，一套基于嵌入的语义结构化流程，用于把原始语料拆解为实体-描述对，再聚类、近邻分组、采样并生成问答对，以此构建高质量的指令调优数据集。

**💡 创新点**

创新点在于：① 用句向量相似度自动构建轻量级邻接图，形成语义连通分量（proximity groups）而不依赖预先构建的知识图谱；② 通过 cluster‑specific prompts 与多种采样策略（proximity、intra‑cluster、inter‑cluster）实现跨文档、多实体的合成问答；③ 结合 LLM‑judge 评价，强调事实准确性与完整性，提出 Binary Accuracy 统一指标。

**🔧 技术方法**

核心技术包括：实体-描述对抽取（基于 LLM 提示）、句向量编码（all‑mpnet‑base‑v2）、UMAP 降维 + K‑Means 或 HDBSCAN 聚类、阈值相似度构建邻接图、聚类内连通分量提取、提示模板与采样策略、LoRA 细调 Llama‑3‑8B‑Instruct、LLM‑judge 评估。

**📊 数据集**

使用三组数据集：Pop‑QA‑Cities‑20（低异质性）、SQuAD‑20（中等异质性）和 Wikitext‑10（高异质性），每个数据集在 5M 与 20M token 预算下生成 synthetic QA 训练集。

**📈 对比分析**

与 Knowledge‑Instruct、InstructLab、EntiGraph 对比，使用相同模型（Llama‑3‑8B‑Instruct）和 LoRA 训练。Lexical 指标（BLEU、ROUGE、METEOR）EmbGen 与 baseline 竞争或稍逊；在 LLM‑judge 的 Binary Accuracy 上，EmbGen 在最高异质性数据集 Wikitext‑10 上在 5M 下提升 12.5%、在 20M 下提升 88.9%，在其它数据集也保持在前列。

**⚠️ 局限性**

局限性包括：① 依赖 LLM 进行实体抽取与描述合成，错误会传播；② 仅测试单一句向量编码器，未检验跨编码器鲁棒性；③ baseline 对比未采用原始持续预训练方式，可能导致不公平；④ 评估主要依赖 LLM‑judge，存在偏差，缺少人工判定；⑤ 仅针对 QA 生成，未验证其他任务的迁移效果。

---

## 360. Sparse Mixture-of-Experts Routing in Visual Diffusion Transformers:Diagnosis, Boundary Calibration and Evolutionary Roadmap from Routing Collapse to Selective Deadlock

**arXiv ID:** 2605.19378 | [PDF](https://arxiv.org/pdf/2605.19378v1)

**作者:** Haiying Sha `[一作]` `[通讯]` (Independent Researcher), Haiying Sha (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统化诊断了Token‑Choice稀疏MoE在视频Diffusion Transformer中的训练失效模式，并基于此提出功能冗余假说及三条Dense‑to‑MoE转换法则。

**💡 创新点**

首次全面剖析视频MoE的全局饱和、选择性死锁与bfloat16精度陷阱，提出功能冗余假说解释死锁机制，并给出三大工程法则和跨模态路由自恢复现象。

**🔧 技术方法**

使用MoE转换、线性/MLP/交叉注意路由器、bfloat16混合精度、专家微噪声初始化、专家利用率监控、跨模态特征拼接、Glyph‑ByT5文字专家等技术。

**📊 数据集**

在包含文本‑图像、图像‑视频、视频编辑、文本‑视频等约20,000条样本的9‑in‑1多任务统一视频生成数据集上进行实验。

**📈 对比分析**

对比三种路由器在同一预训练视频模型上的训练，监测专家利用率、死亡层分布和辅助损失，结果表明线性路由易出现全局饱和，MLP路由出现选择性死锁，交叉注意路由部分自恢复但仍保留9层死锁；未给出具体生成指标。

**⚠️ 局限性**

Token‑Choice范式在单GPU冻结专家时无法消除死锁，bfloat16精度陷阱导致共享专家更新停滞，全专家联合训练受显存限制，缺乏更大规模实验和生成质量评估。

---

## 361. MatPhys: Learning Material-Aware Physics Parameters for Deformable Object Simulation from Videos

**arXiv ID:** 2605.19386 | [PDF](https://arxiv.org/pdf/2605.19386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 362. Beyond Waypoints: Dual-Heatmap Grounding for Cross-Embodiment Semantic Navigation

**arXiv ID:** 2605.19420 | [PDF](https://arxiv.org/pdf/2605.19420v1)

**作者:** Kaijie Yun `[一作]` (Harbin Institute of Technology), Yue Chen `[通讯]` (JD AI Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个双热图（Navigation Heatmap + Facing Heatmap）框架，用于将开放词汇的多模态指令（文本+图像）直接映射为可执行的机器人导航目标。

**💡 创新点**

创新点在于：①放弃单点回归，采用连续热图预测以保留空间不确定性；②同时预测导航和面向热图，实现语义与方向的联合约束；③支持文本、图像、混合多模态指令；④通过自动化合成数据管线实现大规模训练。

**🔧 技术方法**

核心技术包括：Qwen3‑VL Vision‑Language 模型 + LoRA 微调、跨模态注意力融合、密集解码器生成双热图、基于热图的MPPI局部规划、自动化合成与基础模型辅助标注。

**📊 数据集**

数据集：基于Kujiale、GRUtopia等公开资产在 Isaac Sim 中渲染的约130个仿真场景，生成 RGB、深度、语义分割；结合 ScanNet、SunRGBD、HyperSim、Matterport3D 的真实图像，使用 Gemini+SAM3 自动生成热图标签。

**📈 对比分析**

对比方法：在 MP3D 数据集上对 8B 参数的 InternVLA、Molmo、Qwen3‑VL‑Instruct 进行零样本基准；在 Jetbot、H1、Aliengo 三种机器人仿真中进行跨实体验证。结果显示，本方法在导航召回、精度、F1、Affordance Rate 以及跨实体成功率方面均显著优于基线，成功率提升约 25‑30%。

**⚠️ 局限性**

局限性：①仍缺乏真实世界硬件验证，模型对动态障碍的实时响应尚待提升；②热图分辨率与实时推理速度之间的权衡需要进一步优化；③对极其复杂或多目标指令的解析仍受 VLM 先验知识限制；④在大规模多目标环境中可能出现热图重叠导致规划模糊。

---

## 363. LambdaPO: A Lambda Style Policy Optimization for Reasoning Language Models

**arXiv ID:** 2605.19416 | [PDF](https://arxiv.org/pdf/2605.19416v1)

**作者:** Zhe Yuan `[一作]` (Pinterest), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 7199 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Lambda Policy Optimization (LambdaPO)，通过对群体轨迹进行成对比较而非单一统计基线来重新估计优势，从而改进LLM的推理对齐。

**💡 创新点**

创新点在于将优势函数拆解为 Pairwise Decomposed Advantage（PDA），利用动态软权重（基于策略置信度的 sigmoid）捕捉相对偏好；并引入 Semantic Density Reward（语义密度奖励）提升稀疏奖励问题。

**🔧 技术方法**

技术包括无 critic 的 PPO 变体、LambdaRank 思路的优势估计、密度奖励（ROUGE‑L F1）以及温度调节的 sigmoid 权重，整体实现仍保持离线、无参考模型的特性。

**📊 数据集**

使用 OpenR1‑Math‑220k、GSM8K 训练数据，评估基准包括 AIME24、MATH‑500、GPQA‑Diamond 三个数学与问答任务。

**📈 对比分析**

与 Group Relative Policy Optimization (GRPO) 做直接对比，LambdaPO 在 Qwen3‑4B 与 Phi‑4‑mini 两种模型上平均提升 1.45% 与 1.86%，在所有任务上均超越基线；温度参数 1.5 最佳，Semantic Density Reward 进一步提升性能。

**⚠️ 局限性**

局限性包括：需要手动调节温度参数、对密度奖励的依赖可能在非文本推理任务中效果不佳、实验仍局限于数学推理与问答，缺乏对更广泛生成任务的验证。

---

## 364. Once Again, with Style: Understanding and Supporting Partial Reuse in Dashboard Authoring

**arXiv ID:** 2605.19400 | [PDF](https://arxiv.org/pdf/2605.19400v1)

**作者:** Nicole Sultanum `[一作]` (Tableau Research), Arjun Srinivasan `[通讯]` (Tableau Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

对仪表盘的部分复用（主要是布局和样式）进行研究，先通过7名专业作者的定性调研识别需求和挑战，再开发一个基于React+Flask的原型工具进行概念验证，并通过6名作者的使用反馈进一步探索机会。

**💡 创新点**

首次系统化探讨仪表盘布局/样式的部分复用；提出多源、多粒度的复用机制；设计了一个能自动提取并迁移样式的工具；将LLM与可视化规范（Vega‑Lite/CSS）结合，用于实现“填补空缺”的复用。

**🔧 技术方法**

用户中心的定性研究；React+Flask实现的交互原型；多模态大型语言模型（LLM）用于提取组件与样式；Vega‑Lite属性和CSS属性作为共享表示；基于频率的“代表性”样式合并策略。

**📊 数据集**

对9-11份公开仪表盘（如Tableau Public）进行手工挑选，作为复用参考集；在原型中使用这些仪表盘的组件与样式进行实验。

**📈 对比分析**

评价主要采用开放式访谈与任务完成记录，收集参与者的主观感受与使用反馈；未提供量化性能指标或与其他工具的对比，结果以使用便利性、效率提升等定性维度呈现。

**⚠️ 局限性**

样本量有限（10名作者），原型功能受限（编辑与LLM安全性不完善），仅关注作者视角，缺乏现场使用或消费者角度的验证；未在大规模仪表盘组合中测试一致性检查与属性自动传播。

---

## 365. Toward User Comprehension Supports for LLM Agent Skill Specifications

**arXiv ID:** 2605.19362 | [PDF](https://arxiv.org/pdf/2605.19362v1)

**作者:** Zikai Alex Wen `[一作]` `[通讯]` (University of Washington), Zikai Alex Wen (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型代理技能（agent skill）规范中四个用户理解锚点（操作基础、输出合同、边界披露、示例演示）的分布，并通过对6个DNS/C2技能的示例对比实验评估了示例对首轮检查可读性的影响。

**💡 创新点**

创新点在于将描述‑权限一致性理念迁移到代理技能领域，首次提出并量化四个理解锚点，并揭示示例演示能显著降低用户对后端代码的依赖。

**🔧 技术方法**

主要技术为规则基文本编码（deterministic rule‑based scanning）提取锚点线索，以及在合成测试数据上进行的手工对照实验。

**📊 数据集**

使用的数据集包括878个来自GitHub的网络安全技能（YAML/Markdown）文件，以及选取的6个DNS/C2技能与其对应的合成Zeek、TSV等测试数据。

**📈 对比分析**

通过规则匹配得到锚点出现率（92.1%、63.0%、51.4%、19.0%）并对比示例与无示例技能在首轮检查构造上的差异；由于方法基于规则匹配，速度快、易解释，但未进行量化的用户实验评估性能。

**⚠️ 局限性**

局限性包括规则编码未进行人工验证、未开展用户研究验证锚点对理解的实际效果、DNS/C2子集仅示例性且不具普适性、且研究聚焦于网络安全领域，其他高风险领域可能表现不同。

---

## 366. Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models

**arXiv ID:** 2605.19398 | [PDF](https://arxiv.org/pdf/2605.19398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 367. Taming the Thinker: Conditional Entropy Shaping for Adaptive LLM Reasoning

**arXiv ID:** 2605.19358 | [PDF](https://arxiv.org/pdf/2605.19358v1)

**作者:** Shuyu Wei `[一作]` (Beijing Jiaotong University), Jitao Sang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 2206 | [OpenAlex ID](https://openalex.org/A5023834030)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Conditional Entropy Shaping（CES）框架，在现有的 DAPO 强化学习框架中根据答案正确性动态调节 token 级别的熵，从而让 LLM 在简单问题时产生简洁推理，在困难问题时深入探索，兼顾准确率与推理长度。

**💡 创新点**

创新点在于：① 条件双向熵塑造——对正确答案的高熵 token 进行惩罚，对错误答案的高熵 token 进行奖励；② 动态选择高熵 token 的数量，由组内准确率决定，以适应问题难度；③ 将熵梯度直接嵌入优势信号，使模型能够学习降低/提升熵，从而实现资源感知的推理。

**🔧 技术方法**

使用技术包括：DAPO 强化学习、Group‑Relative Advantage 归一化、token‑level 熵计算与动态选取、基于 OpenRLHF 的训练框架、非贪婪解码（温度0.4、top‑p 0.95、repetition penalty 1.05），以及对优势信号的熵基调节。

**📊 数据集**

实验数据集：12 个数学推理基准（AIME24、AMC23、CMATH、CN Middle School 24、College Math、GaoKao Math Cloze、GaoKao 2023 En、GSM8K、Minerva Math、Olympiad Bench、SVAMP、TABMWP）以及 2500 条 DeepMath 训练样本。

**📈 对比分析**

通过与原始 R1‑7B、无 CES 的 DAPO 基线以及无条件熵奖励的 Entropy Advantage 基线比较，CES 在平均准确率上提升了 2.5%（从 69.6% 到 72.1%），平均推理长度下降 411 tokens（从 2376 到 1965），并在多项任务（如 AIME24 +6.7% / -997 tokens，AMC23 +1.9% / -1014 tokens）显示出显著的“赢-赢”效果。

**⚠️ 局限性**

局限性：① 需要可验证的正确性信号，难以直接应用于主观或弱可验证任务；② 对超参数（τ、β 等）敏感，需在新任务或模型上进行微调；③ 目前仅在数学推理场景验证，跨领域推广仍需进一步研究。

---

## 368. XAI FL-IDS: A Federated Learning and SHAP-Based Explainable Framework for Distributed Intrusion Detection Systems

**arXiv ID:** 2605.19448 | [PDF](https://arxiv.org/pdf/2605.19448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 369. KappaPlace: Learning Hyperspherical Uncertainty for Visual Place Recognition via Prototype-Anchored Supervision

**arXiv ID:** 2605.19435 | [PDF](https://arxiv.org/pdf/2605.19435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 370. Self-assembling Modular Aerial Robot for Versatile Aerial Tasks

**arXiv ID:** 2605.19431 | [PDF](https://arxiv.org/pdf/2605.19431v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 371. What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents

**arXiv ID:** 2605.19447 | [PDF](https://arxiv.org/pdf/2605.19447v1)

**作者:** Xiaozhe Li `[一作]` (Tongji University), Kai Chen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于环境反馈的选择性强化学习框架 SERL，用于训练长时程 LLM 代理。

**💡 创新点**

创新点在于将环境回馈的来源与插入粒度分离，使用奖励决定更新方向，只让教师回馈在动作级别调整幅度并对可执行动作进行分辨率限制，同时对教师信号进行时间衰减，避免特权信息泄露。

**🔧 技术方法**

技术包含无值模型的 GRPO 策略梯度、anchor‑level 与 step‑level 反馈插入、教师‑学生对数概率差异加权、可执行动作掩码、以及衰减的教师权重。

**📊 数据集**

使用了 ALFWorld 和 WebShop 两个多轮长时程代理基准数据集。

**📈 对比分析**

通过与提示式方法、纯 RL 方法（PPO、RLOO、GRPO、GIGPO、HGPO）以及 RL‑蒸馏混合基线比较，SERL 在 ALFWorld 上达 90.0% 成功率、WebShop 上达 80.1% 成功率，显著优于其他方法。

**⚠️ 局限性**

局限性包括：仅对可执行动作进行加权，依赖教师信号需精细衰减；过多的回馈来源可能引入噪声；对长文本观察的 LLM 判别器受上下文长度限制，导致某些场景下反馈效果受限。

---

## 372. Neuromorphic Control of a Flapping-Wing Robot on Resource-Constrained Hardware

**arXiv ID:** 2605.19430 | [PDF](https://arxiv.org/pdf/2605.19430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 373. Vision Harnessing Agent for Open Ad-hoc Segmentation

**arXiv ID:** 2605.19410 | [PDF](https://arxiv.org/pdf/2605.19410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 374. DRReduce: Enhancing Syntax-Guided Program Reduction with Dependency Reconstruction

**arXiv ID:** 2605.19412 | [PDF](https://arxiv.org/pdf/2605.19412v1)

**作者:** Qiong Feng `[一作]` (Nanjing University of Science and Technology), Peng Liang `[通讯]` (Wuhan University)

**通讯引用:** 4685 | [OpenAlex ID](https://openalex.org/A5049939779)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种语言无关的程序简化框架 DRReduce，结合语义依赖重建层提升语法引导的程序简化效果。

**💡 创新点**

创新点在于引入轻量级语义依赖重建，自动修复删除导致的语义不一致，且不依赖任何语言专属规则，能够统一处理任何带类型系统的语言。

**🔧 技术方法**

采用语义依赖图构造、语义节点分类、默认值替换与关联结构删除、DDMin 搜索以及基于 Perses 的语法减小等技术。

**📊 数据集**

使用 28 个真实世界的 C/Java bug‑triggering 程序，来源于 Perses、Specimin、JDK 等公开数据集。

**📈 对比分析**

与 Perses、WDD、CDD、CReduce、Latra 等基线对比，DRReduce 平均可使程序大小缩小 51.9%（对 Perses）、14.9%（对 WDD）和 19.8%（对 CDD），并在大多数程序上完成时间更快；与 CReduce、Latra 相比，效果相当但效率分别高 3.3× 和 1.2×；消除查询调用 80.2%+、时间 58.7%+、最终 token 55.1%+。

**⚠️ 局限性**

局限性包括：重建仅使用默认值，无法保持行为等价，导致对某些对类型或具体值敏感的 bug 效果下降；目前仅在 C、Java 上验证，未评估其它语言或 IR；未针对不同 bug 性质设计更细粒度的重建策略。

---

## 375. Scalable, Energy-Efficient Optical-Neural Architecture for Multiplexed Deepfake Video Detection

**arXiv ID:** 2605.19360 | [PDF](https://arxiv.org/pdf/2605.19360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. Targeted Downstream-Agnostic Attack

**arXiv ID:** 2605.19446 | [PDF](https://arxiv.org/pdf/2605.19446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 377. Conflict-Free Replicated Data Types for Neural Network Model Merging: A Two-Layer Architecture Enabling CRDT-Compliant Model Merging Across 26 Strategies

**arXiv ID:** 2605.19373 | [PDF](https://arxiv.org/pdf/2605.19373v1)

**作者:** Ryan Gillespie `[一作]` `[通讯]` (Independent researcher), Ryan Gillespie (Independent researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两层 CRDT 架构，利用 OR‑Set 管理模型贡献并通过确定性纯函数实现任何现有模型合并策略的分布式、无冲突合并；

**💡 创新点**

创新点在于证明所有26种常用合并策略均不满足 CRDT 的三大代数性质，并通过将状态管理与策略执行拆分为两层，构造了可对任意策略实现强事件最终一致性的通用包装；

**🔧 技术方法**

使用技术包括 OR‑Set CRDT、Merkle 树哈希、SHA‑256 排序、种子化随机、纯函数约束以及基于 gossipy 协议的多节点同步；

**📊 数据集**

实验数据集包括公开的 GPT‑2‑XL（1.5B 参数）与 Mistral‑7B（7.24B 参数）微调模型，以及 100 节点实验中使用的 512×512 张量；此外在 4×4 张量上做了控制实验；

**📈 对比分析**

评估方法分三层：①在 4×4 张量上验证代数性质；②在生产规模模型上检验所有策略在 CRDT 包装下通过 104/104 的合规性测试；③在 100 节点 gossip 环境下测算收敛时间，CRDT 开销低于 0.5 ms，整体性能几乎无额外负担；

**⚠️ 局限性**

局限性包括：需满足纯函数和计算确定性；尚未实现增量 resolve 与 delta‑state 传播，跨硬件可复现性待验证；OR‑Set 的 add‑wins 可能无法抑制恶意模型，缺乏拜占庭容错机制。

---

## 378. Locked Out at 8,000 Miles: Why UK-China Partnership Students Are Suffering

**arXiv ID:** 2605.19367 | [PDF](https://arxiv.org/pdf/2605.19367v1)

**作者:** Benjamin Kenwright `[一作]` `[通讯]` (Aberay University), Benjamin Kenwright (Aberay University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过收集并分析Reddit、IT支持日志以及英国-中国合作项目的学生与教师经验，揭示了高校网络安全措施（如多因素身份验证、设备合规性和浏览器检查）对跨时区国际学生的严重可访问性障碍，并指出当前安全架构基于同步支持假设，导致这些学生被系统性排斥。

**💡 创新点**

创新点在于提出并论证了“同步支持假设”这一概念，揭示了多重安全层级在跨时区使用时的锁定链条，并将该问题视为教育公平与安全平衡的研究缺口。

**🔧 技术方法**

采用定性分析技术，包括案例研究、情境访谈和在线论坛文本挖掘，结合安全体系结构的流程分析。

**📊 数据集**

使用的数据集包括：Reddit r/college、r/UniUK、r/Professors等公开论坛帖子、英国大学IT帮助中心的支持请求日志、英国-中国合作项目的学生与教师证词。

**📈 对比分析**

通过对比国内学生与国际合作学生在身份验证失败、设备锁定、浏览器兼容性等方面的经历，定性评估了安全措施的“可访问性”与“安全强度”之间的权衡，发现后者在国际场景下显著下降，未提供量化性能指标。

**⚠️ 局限性**

局限性包括样本主要聚焦英国-中国合作项目，缺乏跨国其他合作案例的实证；数据来源为公开论坛与日志，可能存在自我选择偏差；未对安全风险程度进行量化评估，仅从可访问性角度进行描述。

---

## 379. MAM-CLIP: Vision-Language Pretraining on Mammography Atlases for BI-RADS Classification

**arXiv ID:** 2605.19359 | [PDF](https://arxiv.org/pdf/2605.19359v1)

**作者:** Halil Ibrahim Gulluk `[一作]` (Stanford University), Olivier Gevaert `[通讯]` (Stanford University)

**通讯引用:** 16156 | [OpenAlex ID](https://openalex.org/A5078274543)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个多模态图像-文本预训练模型，并在乳腺X光影像上进行微调，用于预测BI‑RADS分级。

**💡 创新点**

创新点在于利用乳腺影像图谱中高质量的说明文字进行对比学习预训练，证明仅使用约2300个图像-文本对就能显著提升BI‑RADS分类性能，甚至优于在下游任务中额外添加2000个标注样本。

**🔧 技术方法**

采用 PubMedBERT 作为语言编码器、ConvNeXt 作为视觉编码器，结合 CLIP 风格的 InfoNCE 对比损失和掩码语言建模（MLM）损失进行预训练，随后在 BI‑RADS 分类任务上微调。

**📊 数据集**

预训练数据来自两本乳腺影像图谱（共2313个图像-文本对）；下游分类使用 EMBED（Emory Breast Imaging Dataset）和 TEKNOFEST 2023 AI 医疗竞赛数据集（共约43k张图）。

**📈 对比分析**

与 ImageNet 预训练的 ConvNeXt 基线对比，模型在 5 类和 3 类 BI‑RADS 分类任务中均表现更优；在样本量越少时提升幅度最大，3 类 macro F1 从 +1% 提升到 +14%；在 10k+ 样本时，2313 个图像-文本对的预训练效果甚至优于额外添加的 2000 个标注样本。

**⚠️ 局限性**

局限性包括：仅评估 BI‑RADS 1/2/0/4/5 五类（排除 BI‑RADS 3 级）；预训练文本仅来自两本图谱，可能缺乏多样性；实验仅在单一 GPU 上完成；对图像分辨率依赖较高；未探索与电子病历等其他模态的进一步融合。

---

## 380. Competitive Search with a Faulty Satnav (GPS): When Probability Matching is Rational

**arXiv ID:** 2605.19440 | [PDF](https://arxiv.org/pdf/2605.19440v1)

**作者:** Steve Alpern `[一作]` (Warwick University), Mark Broom `[通讯]` (University of London)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了在星形网络中，若干搜索者使用同一 GPS 指引（可能错误）竞争先找到隐藏宝物的游戏，并给出了在所有参数（搜索者人数 n、叶子数 k、GPS 正确率 p）下唯一的对称均衡信任概率 q̅ 的解析表达式。

**💡 创新点**

创新点在于首次将竞争搜索（搜索者之间的竞争）与错误信息指引（Faulty Satnav）结合，证明当人数趋于无穷大时均衡信任概率恰等于 GPS 的可靠度 p，实现了理论上可被视为“概率匹配”的合理行为；并揭示了 q̅ 随人数、可靠度和分支数的单调性。

**🔧 技术方法**

主要技术包括：博弈论中的对称纳什均衡分析、概率与组合论的推导、函数单调性与极值分析、极限行为证明（n→∞）以及对称性约束下的解析求解。

**📊 数据集**

该研究为纯理论分析，不使用实验数据或现实数据集；所有结果均基于抽象的星形网络模型与概率参数。

**📈 对比分析**

由于是理论模型，没有与现有方法的数值对比；作者通过解析公式与极限论证展示了 q̅ 的单调性和收敛性质，未给出实验性能指标。

**⚠️ 局限性**

局限性包括：仅考虑星形网络，未扩展到更一般的图结构；假设所有搜索者拥有相同的 GPS 指引且指引信息在搜索过程中保持不变；未考虑时间动态变化的信任策略或搜索者间信息交流；模型缺乏对真实 GPS 波动和搜索者行为的验证。

---

## 381. A Bitter Lesson for Data Filtering

**arXiv ID:** 2605.19407 | [PDF](https://arxiv.org/pdf/2605.19407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 382. Fifty Shades of Darknet

**arXiv ID:** 2605.19437 | [PDF](https://arxiv.org/pdf/2605.19437v1)

**作者:** Siddique Abubakr Muntaka `[一作]` (University of Cincinnati), Jacques Bou Abdo `[通讯]` (University of Cincinnati)

**通讯引用:** 818 | [OpenAlex ID](https://openalex.org/A5003086292)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过构造并测试I2P网络中的Exclusive Network（Shade 8）节点，证明其可以在不公开RouterInfo的情况下保持服务可访问，从而构建隐蔽的C2基础设施。

**💡 创新点**

提出Shade Taxonomy对I2P路由器可见性进行八级划分，首次揭示了结构性不可观测的Exclusive Network层，指出传统经验测绘的固有限制。

**🔧 技术方法**

利用I2P 2.12.0的API、Floodfill探测、XOR路由键分析、LeaseSet检查以及自定义Ghost配置脚本，实现对节点可见性分类与归属判定。

**📊 数据集**

实验基于三节点Ubuntu 24.04 LTS虚拟机、I2P 2.12.0以及从I2P NetDB收集的3242条RouterInfo记录（48%为Floodfill）。

**📈 对比分析**

对比五种归属方法（本地NetDB、控制台缓存、Floodfill探测、Gateway扫描、XOR关联）发现对Shade 8节点在500次Floodfill探测后仍无NetDB记录，证明其结构性隐蔽；对可见节点则快速归属。

**⚠️ 局限性**

研究局限在于仅验证单个Shade 8节点且仅使用I2P 2.12.0，缺乏大规模多节点统计及跨变体（i2pd、I2P+）的实证，且对攻击者在真实网络中实现完整C2链的鲁棒性未进行实测。

---

## 383. CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization

**arXiv ID:** 2605.19436 | [PDF](https://arxiv.org/pdf/2605.19436v1)

**作者:** Ahmed Heakl `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salman Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 12322 | [OpenAlex ID](https://openalex.org/A5000300751)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种对强化学习可验证奖励（RLVR）模型进行token级信用分配的对比证据策略优化（CEPO）方法。

**💡 创新点**

利用正确答案与错误答案教师的对比比例，精细化信用分配，消除流利度混淆、异化负信号与单侧证据问题，同时保持结构安全。

**🔧 技术方法**

在GRPO框架下引入对比证据Δ＝log P_T⁺/P_T⁻计算加权优势，并在PPO‑剪切目标上更新策略。

**📊 数据集**

在Geo3k几何问答数据集上训练，并在DynaMath、LogicVista、MathVision_mini、MMMU、WeMath等五个多模态数学推理基准上评估。

**📈 对比分析**

与GRPO、OPSD、SDPO、RLSD等方法在相同训练步数下对比，CEPO在2B模型上平均提升至43.43%（比GRPO提升2.26pp），4B模型提升至60.56%（比GRPO提升3.13pp）。

**⚠️ 局限性**

对比教师来源敏感；对推理链短的基准如MMMU提升有限；方法依赖可验证奖励与已拒绝rollout，非数学推理任务的泛化尚待验证。

---

## 384. Unlocking the Potential of Continual Model Merging: An ODE Perspective

**arXiv ID:** 2605.19409 | [PDF](https://arxiv.org/pdf/2605.19409v1)

**作者:** Lihong Lin `[一作]` (Northeastern University), Haidong Kang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于ODE的持续模型合并方法（ODE‑M），通过在参数空间中生成低损失的连续轨迹来实现对已学习知识与新任务知识的可控融合。

**💡 创新点**

创新点包括：①将合并过程视为一条受控路径而非一次性跳跃；②利用梯度投影和自适应衰减构造阻碍函数，使得轨迹能够避开高损失障碍；③通过任务重要性权重设计的时间调度实现稳定‑可塑性平衡。

**🔧 技术方法**

使用的技术主要有：ODE动力学（时间依赖速度场）、梯度投影、第一阶反馈控制、欧拉数值积分、任务加权时间调度。

**📊 数据集**

实验数据集：FusionBench中的20个任务，基于CLIP Vision Transformer（ViT‑B/32、ViT‑B/16、ViT‑L/14）进行任务细化后得到的模型。

**📈 对比分析**

与SWA、Task Arithmetic、Ties‑Merging、OPCM等基线对比，在8/14/20任务流和三种ViT架构下，ODE‑M在ACC与ACC_w上均位居榜首，且在大部分设置下保持或提升BWT/BWT_w，表明在准确率与稳定性之间取得更优平衡。

**⚠️ 局限性**

局限性：①需要一定量的校准数据（虽在几百样本即可获得良好效果，但在极端数据缺乏场景下可能受限）；②合并时需要数十到几百秒的计算时间，主要受ODE数值积分与梯度计算开销影响；③目前仅在视觉Transformer上验证，未扩展到大型语言或多模态模型；④对不同任务顺序和动态校准策略的鲁棒性仍待深入研究。

---

## 385. Understanding Dynamics of Adam in Zero-Sum Games: An ODE Approach

**arXiv ID:** 2605.19392 | [PDF](https://arxiv.org/pdf/2605.19392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 386. LMM-Track4D: Eliciting 4D Dynamic Reasoning in LMMs via Trajectory-Grounded Dialogue

**arXiv ID:** 2605.19390 | [PDF](https://arxiv.org/pdf/2605.19390v1)

**作者:** Chaoyue Li `[一作]` (Huazhong University of Science and Technology), Jiayu Ding `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于轨迹的多轮时空对话任务并构建Track4D-Bench评测基准

**💡 创新点**

引入RTGE、持久TRK状态记忆和OSK-RA解码器实现4D动态推理

**🔧 技术方法**

使用Qwen3.5-9B Backbone + LoRA + RTGE + TRK + OSK-RA

**📊 数据集**

采用APIDIS、KITTI、WildTrack三大数据集生成的526条对话样本

**📈 对比分析**

与多种专有与开源LMM、SFT控制模型对比，LMM-Track4D在语言与几何指标上显著领先（SAcc 0.758、Traj-Acc 0.626、TAcc 0.619）

**⚠️ 局限性**

仍依赖结构化监督，规模、时间跨度和交互开放性受限

---

## 387. PRISM: A Benchmark for Programmatic Spatial-Temporal Reasoning

**arXiv ID:** 2605.19382 | [PDF](https://arxiv.org/pdf/2605.19382v1)

**作者:** Qiran Zhang `[一作]` (Shanghai Jiao Tong University), Chen Qian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 53681 | [OpenAlex ID](https://openalex.org/A5100428454)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PRISM，一个大型双语程序化视频生成基准，用于评估代码生成的可执行性与空间连贯性；

**💡 创新点**

创新点在于构建了10,372对人类校准的指令-代码对，设计了四维评估框架（代码可靠性、空间推理、动态视觉复杂度、时间密度），揭示了执行-空间差距；

**🔧 技术方法**

采用Manim渲染器与自动化检索管道，结合低层渲染解析和人工审核，使用代码执行和几何检查来评估结果；

**📊 数据集**

使用的基准数据集是PRISM，包含5,199条英文和5,173条中文教育可视化场景，涵盖437个细分主题；

**📈 对比分析**

与七种主流LLM（包括GPT‑5.4、Gemini 3.1、Kimi K2、Claude Sonnet、Qwen3.5、GLM‑5、DeepSeek‑V3.2）进行对比，结果显示闭源模型在代码执行率上领先，但空间通过率普遍落后，平均下降约41%；

**⚠️ 局限性**

局限性包括仍依赖Manim作为渲染器，评估主要聚焦二维静态空间，缺乏对更复杂三维或多视角动画的覆盖；

---

## 388. The Evaluation Game: Beyond Static LLM Benchmarking

**arXiv ID:** 2605.19377 | [PDF](https://arxiv.org/pdf/2605.19377v1)

**作者:** Paul Wang `[一作]` (Sorbonne Université), Vincent Corruble `[通讯]` (Sorbonne Université)

**通讯引用:** 1276 | [OpenAlex ID](https://openalex.org/A5001466260)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于博弈论的评估框架（Evaluation Game），将评估者与训练者的交互建模为两人博弈，并通过群作用（group actions）描述两方的数据增强过程；在圆形翻译游戏中推导出精确的相位阈值，并在实际 LLM 上实验验证了局部泛化与提示变换的群结构。

**💡 创新点**

创新点在于：① 将 LLM 的安全评估视为动态博弈而非静态基准；② 用群论框架统一刻画评估者与训练者的增广策略；③ 在最简实例（圆形翻译）中得到闭式相位阈值与实验一致；④ 通过实验展示 Fine‑tune 后的拒绝率与提示距离呈局部相关，提示变换满足群运算。

**🔧 技术方法**

技术手段包括：博弈论建模、群作用与轨道分析、ε‑覆盖与局部泛化理论、LoRA 微调、模型嵌入距离测度、BET（提示变换）线性拟合、交叉验证与 R² 评估。

**📊 数据集**

主要使用的数据集有：WildJailBreak（用于评估与实验）、MMLU 与 Wikitext‑2（微调评估）、OLMo‑3.1‑32B‑Instruct（作为判定器）以及部分预训练 Llama‑3.1‑8B、Mistral‑7B‑Instruct、Qwen2.5‑7B‑Instruct 等。

**📈 对比分析**

对比方法：与传统静态评估基准相比，证明后者无法区分真正修复与仅记忆的补丁；实验显示 Fine‑tune 的局部泛化与提示距离呈正相关，提示变换符合群结构，验证了理论预测；尽管未给出统一的数值指标，但实验结果与理论阈值吻合，展示了框架的有效性。

**⚠️ 局限性**

局限性包括：① 仅在最简的 1‑维圆形翻译实例中给出解析结果；② 实验局限于几种小规模 LoRA 微调模型与单一判定器，无法完全代表大模型；③ ε‑覆盖范围的估计依赖于特定嵌入空间与微调策略；④ 未探讨更高维、非阿贝尔或无限阶群的情况；⑤ 只考虑线性拟合的提示变换，非线性变换尚未验证。

---

## 389. When to Answer and When to Defer: A Decision Framework for Reliable Code Predictions

**arXiv ID:** 2605.19369 | [PDF](https://arxiv.org/pdf/2605.19369v1)

**作者:** Ravishka Rathnasuriya `[一作]` (University of Texas at Dallas), Wei Yang `[通讯]` (University of Texas at Dallas)

**通讯引用:** 11242 | [OpenAlex ID](https://openalex.org/A5036689637)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出统一框架，整合代码模型的不确定性估计、校准与工具驱动的弃权，以支持分类与生成任务的可靠部署。

**💡 创新点**

创新点在于三方面：1）针对代码设计定制化不确定性信号；2）引入预部署与后部署双重校准机制；3）通过MCP工具链将弃权转化为可操作的恢复路径，让弃权成为决策原语。

**🔧 技术方法**

技术包括：内部信号提取、加权Platt/温度/等价映射、logit行为校准、抽样扰动估计、熵/置信度差等多种不确定性指标，以及MCP层调用静态、语义分析器进行恢复。

**📊 数据集**

使用的数据集包括MBPP+（代码生成）、缺陷预测集和漏洞检测集，模型评估涵盖DeepSeek-Coder-7B、CodeLlama-7B、Qwen-Coder-7B。

**📈 对比分析**

与基础模型、Platt、等距回归、温度标定等方法对比，加权校准在80%覆盖率下准确率提升至70%+，logit校准在80%覆盖率下选择性准确率超过90%，显示显著的性能提升。

**⚠️ 局限性**

局限性在于校准与弃权高度依赖模型内部可观测性；在稀有模式或多义语法下仍难以完全信任；此外实际部署需要集成多种工具，复杂度较高。

---

## 390. Dual-Prompt CLIP with Hybrid Visual Encoders for Occluded Person Re-Identification

**arXiv ID:** 2605.19527 | [PDF](https://arxiv.org/pdf/2605.19527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 391. SafeAlign-VLA: A Negative-Enhanced Safe Alignment Framework for Risk-Aware Autonomous Driving

**arXiv ID:** 2605.19524 | [PDF](https://arxiv.org/pdf/2605.19524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 392. High-Rate Public-Key Pseudorandom Codes for Edit Errors

**arXiv ID:** 2605.19402 | [PDF](https://arxiv.org/pdf/2605.19402v1)

**作者:** Shengtang Huang `[一作]` (University of Science and Technology of China), Zhaienhe Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

设计并构造了面向插入/删除（编辑）错误的公共密钥伪随机编码（PRC），并在小字母表（包括二进制）上实现了接近1或1/2的高码率，提供了零位PRC向高位PRC的通用提升框架。

**💡 创新点**

创新点包括：①利用CGK嵌入将Hamming鲁棒PRC转换为编辑鲁棒PRC；②提出零位PRC到高位PRC的随机交错与种子掩码框架，保持字母表大小与伪随机性；③在二进制上通过同步字符串和随机内码实现近1/2码率；④实现了接近Singleton极限的插入/删除鲁棒性。

**🔧 技术方法**

主要技术手段包括：CGK随机游走嵌入、同步字符串（self‑matching）、随机内码与拼接、种子掩码和PRG伪随机化、线性/AG码的错误纠正、以及多层交错与哈希/掩码的安全证明。

**📊 数据集**

本文未使用具体数据集，而是以理论构造和计算复杂度分析为主。

**📈 对比分析**

通过与已有PRC构造对比，本文在相同安全假设下实现了二进制高码率（≈0.5）和大字母表近1码率的编辑鲁棒PRC；性能指标为：鲁棒性满足任意子线性多项式编辑率，码率可任意逼近1（或1/2），而之前的工作只能得到常数码率或在大字母表上才能达到。

**⚠️ 局限性**

局限性包括：①CGK嵌入导致的平方失真，使得仅能处理子线性多项式编辑误差；②零位PRC到高位PRC的提升需要大量随机化与插入/删除标记，编码/解码复杂度相对较高；③对二进制码率的提升仍停留在1/2，无法突破此界；④对安全假设（如LPN/Planted XOR）依赖较强。

---

## 393. CANINE: Coaching Visually Impaired Users for Interactive Navigation with a Robot Guide Dog

**arXiv ID:** 2605.19501 | [PDF](https://arxiv.org/pdf/2605.19501v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 394. CutVerse: A Compositional GUI Agents Benchmark for Media Post-Production Editing

**arXiv ID:** 2605.19484 | [PDF](https://arxiv.org/pdf/2605.19484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 395. Investigating Cross-Modal Skill Injection: Scenarios, Methods, and Hyperparameters

**arXiv ID:** 2605.19523 | [PDF](https://arxiv.org/pdf/2605.19523v1)

**作者:** Zhiyu Xu `[一作]` (Peking University), Xu Sun `[通讯]` (Peking University)

**通讯引用:** 6910 | [OpenAlex ID](https://openalex.org/A5111863979)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并系统评估了将专家语言模型（LLM）注入视觉语言模型（VLM）以实现跨模态技能注入的框架，并在语言、数学推理和指令跟随三类场景下对多种融合方法和超参数调优策略进行了实证比较。

**💡 创新点**

创新点在于：①首次对跨模态技能注入进行全维度系统化研究；②发现经典线性融合（Task Arithmetic、DARE）在跨模态任务中表现最佳，且对超参数的敏感度可通过 GP‑BO 实现高效优化；③提出 NaN 作为低成本、无需调优的准备用法。

**🔧 技术方法**

技术主要包括：任务向量插值（Task Arithmetic）、稀疏重缩放（DARE、TIES）、基于 Fisher 信息或激活对齐的 data‑aware 融合（Fisher、RegMean），以及无调优的子空间或系数估计方法（WUDI、TSV、MetaGPT、NaN），并采用 GP‑BO、CMA‑ES、Pattern Search 等超参搜索算法。

**📊 数据集**

使用的数据集涵盖六个视觉语言基准：中文视觉语言理解（CMMMU）、日语视觉语言理解（JMMMU）、视觉数学推理（MathVista、MathVerse）、视觉指令跟随（MIA‑Bench、WildVision），并结合相应的领域专家 LLM（中文、日语、数学、指令等）。

**📈 对比分析**

比较实验表明：经典融合方法在大多数基准上均优于数据感知和无调优方法，DARE 在所有场景中保持最稳健；NaN 虽略逊但在不需调参的前期探索中表现突出；在指令跟随任务中，经典方法与 NaN 仍可获得 70+ 分以上；而数学推理任务难以显著提升，体现跨模态推理的挑战；GP‑BO 在超参优化上显著低于随机或局部搜索，达到最高样本效率。

**⚠️ 局限性**

局限性包括：仅针对视觉–文本模态，未扩展到音频、视频等；实验受限于计算资源，未覆盖更大规模或更复杂的领域专家模型；对极端噪声或低质量视觉输入的鲁棒性尚未评估。

---

## 396. Position: The Turing-Completeness of Real-World Autoregressive Transformers Relies Heavily on Context Management

**arXiv ID:** 2605.19514 | [PDF](https://arxiv.org/pdf/2605.19514v1)

**作者:** Guanyu Cui `[一作]` (Renmin University of China), Kun He `[通讯]` (Renmin University of China)

**通讯引用:** 5742 | [OpenAlex ID](https://openalex.org/A5020278936)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文对现实中使用的自回归 Transformer 系统进行了形式化建模，并区分了固定系统（单一模型 + 固定上下文管理）与尺度族（模型参数随输入长度变化）两种计算范式，指出后者的 Turing 完备性结论并不能直接推广到实际部署的 Transformer。进一步分析不同上下文管理策略（如摘要式、滑动式、外部存储等）对固定 Transformer 系统计算能力的影响，证明摘要式管理只能实现常数空间的计算，而滑动式管理能达到线性空间，且若允许外部读写或一次生成两词，系统可达到 Turing 完备性。

**💡 创新点**

①提出固定系统的正式定义并与尺度族进行对比；②阐明尺度族结果与 Turing 完备性之间的误读；③系统性地评估上下文管理对计算能力的决定性影响，给出常数空间、线性空间和 Turing 完备性的阈值；④强调“上下文管理”是决定实际 LLM 计算功效的核心。

**🔧 技术方法**

使用形式化模型（Transformer 视为函数 T，解码规则 D，上下文管理器 C）以及图灵机、复杂度类（DSPACE、DTIME、DCSL）理论；构造 Turing 机模拟证明，利用上下文窗口长度、数值精度、窗口滑动等参数进行复杂度分析；参考现有理论工作与外部存储、工具调用等实际上下文管理方案。

**📊 数据集**

无数据集，本文为纯理论分析与形式化证明，未涉及实验或数据集。

**📈 对比分析**

通过理论推导与复杂度类对比，说明在固定系统下不同上下文管理方法对应的计算上限（常数空间 vs 线性空间 vs Turing 完备性）。对比尺度族结果，指出其仅提供资源上限而非 Turing 完备性；在实验层面无直接对比，结论主要基于数学证明。

**⚠️ 局限性**

①仅分析了理想化的确定性上下文管理器；②未覆盖所有实际部署中使用的多种复杂上下文管理机制（如检索增强、工具调用、外部内存）；③假设 Transformer 参数与精度固定，未考虑近似计算与噪声对复杂度的影响；④理论结果可能与实际 LLM 的学习能力、泛化与误差特性不直接对应。

---

## 397. ARC-RL: A Reinforcement Learning Playground Inspired by ARC Raiders

**arXiv ID:** 2605.19503 | [PDF](https://arxiv.org/pdf/2605.19503v1)

**作者:** Carlo Romeo `[一作]` (University of Florence), Andrew D. Bagdanov `[通讯]` (University of Florence)

**通讯引用:** 7045 | [OpenAlex ID](https://openalex.org/A5064029620)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 MuJoCo 物理引擎下构建了一个连续控制训练平台，包含四种受 ARC Raiders 机械生物启发的机器人（Queen、Bastion、Tick、Leaper），并在该平台上对纯在线强化学习算法与加入先验数据的强化学习算法进行了系统对比实验。

**💡 创新点**

① 设计了专为游戏 NPC 定制的多样化机器人形态与风格约束；② 统一的多组件奖励函数实现速度跟踪、姿态保持、步态匹配等目标；③ 通过自定义的周期性开环中央模式生成器（CPG）提供高质量演示数据；④ 搭建了可与现有 RL 研究对标的 benchmark。

**🔧 技术方法**

使用 MuJoCo + Gymnasium API 进行仿真；基于三维运动学与控制的奖励设计；构建 CPG 开环控制器；采用强化学习算法 SAC、SPEQ、SOPE‑EO、SACfD、SPEQ‑O2O、SOPE；使用 TensorBoard 记录学习曲线；对比可视化分析。

**📊 数据集**

使用 CPG 产生的演示数据作为先验数据集；除此之外不使用任何外部运动捕捉或现成游戏数据；实验中的数据全部来自在线环境交互。

**📈 对比分析**

通过在 1M 环境步数内对所有算法进行多种随机种子跑评估，记录平均回报与标准差；对比结果显示：① 纯在线 RL 能超过 CPG 参考回报；② 加入先验数据的算法在初始阶段即获得更高回报并在最终阶段取得更优性能；③ SOPE 在所有方法中表现最佳；视觉对比表明带先验的算法更贴合目标步态。

**⚠️ 局限性**

仅在平地上训练，未加入复杂地形与导航任务；先验数据仅限 CPG 轨迹，缺乏多样化的真实演示；奖励函数对不同行为的信号密度有限，可能需要进一步细化；benchmark 的可扩展性和多任务适用性仍待提升。

---

## 398. Generative Auto-Bidding with Unified Modeling and Exploration

**arXiv ID:** 2605.19457 | [PDF](https://arxiv.org/pdf/2605.19457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 399. FedADAS: Communication-Efficient Federated Distillation for On-Device Driver Yawn Recognition in Vehicular Networks

**arXiv ID:** 2605.19480 | [PDF](https://arxiv.org/pdf/2605.19480v1)

**作者:** Ahmed Mujtaba `[一作]` (Silicon Austria Labs), Radu Prodan `[通讯]` (University of Innsbruck)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种 FedADAS 联邦蒸馏框架，实现了车辆端异构模型的协同学习和极低通信开销的车载司机打哈欠识别；

**💡 创新点**

首次实现了完整模型异构的联邦蒸馏、仅交换 soft logits 的通信压缩（≈9974×），并支持边缘端完整训练；

**🔧 技术方法**

采用联邦蒸馏、知识蒸馏（温度缩放、KL 散度）、公共数据集、ME‑Net/PE‑Net 轻量化网络，部署于 NVIDIA Jetson AGX/ Nano；

**📊 数据集**

使用 YawDD 与 YawDD+ 车载打哈欠数据集，并构建公开共享数据集做蒸馏；

**📈 对比分析**

与 FedAvg 及多种 FL/FD 基线对比，FedADAS 在 3–115 辆车实验中显著提升个体与全局准确率，通信成本下降至仅 0.02 MB/轮，PE‑Net 在 Jetson NANO 上实现 99.39% 准确率、1.99 ms 推理；

**⚠️ 局限性**

受限于公共数据集的隐私风险、模型容量差异导致蒸馏效果下降、在极端域偏移下通用性下降，以及仅针对打哈欠任务的验证。

---

## 400. Trust It or Not: Evidential Uncertainty for Feed-Forward 3D Reconstruction with Trust3R

**arXiv ID:** 2605.19539 | [PDF](https://arxiv.org/pdf/2605.19539v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 401. EgoCoT-Bench: Benchmarking Grounded and Verifiable Operation-Centric Chain of Thought Reasoning for MLLMs

**arXiv ID:** 2605.19559 | [PDF](https://arxiv.org/pdf/2605.19559v1)

**作者:** Yang Dai `[一作]` (Zhejiang University), Wenqiao Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 2926 | [OpenAlex ID](https://openalex.org/A5063062444)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并构建了 EgoCoT-Bench——一个细粒度的自我视角视频问答基准，包含 351 条 egocentric 视频、3,172 条 QA 对，配有明确的时空证据与逐步推理（Chain‑of‑Thought）注释。

**💡 创新点**

创新点主要有三：① 通过 STSG（时空场景图）指导样本生成，保证证据可验证性；② 在问答中加入逐步推理注释，实现可解释、可检验的操作中心推理；③ 设计了三维度评估指标（答案准确率、推理分数、伪正确率），从答案与推理一致性双重视角评价模型。

**🔧 技术方法**

技术上使用了 STSG 生成与人机协同校验 pipeline、LLM（如 GPT‑5、Qwen‑VL 系列）生成自然语言问题/答案/推理，LLM‑judge（Qwen‑Max）对推理进行 0–5 分的评分，并计算伪正确率。

**📊 数据集**

数据来源为 351 条 egocentric 视频，采自 Ego4D、EPIC‑KITCHENS、MECCANO、Charades‑Ego、HD‑EPIC 以及自录视频，覆盖日常使用、厨房操作、装配等多种场景。

**📈 对比分析**

对比 4 个专有 MLLM（GPT‑5.1/5.2、Qwen3‑VL‑Plus/3.5‑Plus）与 15 个开源 MLLM（InternVL、LLaVA、Qwen 系列）进行评测。最佳模型 Qwen3.5‑27B 的整体准确率为 70.68%，但仍落后人类 95.93%；在 Predictive & Causal Inference 子任务上最高可达 78.21%，而 Spatio‑Temporal Retrospection 与 High‑level Grounded Reasoning 仅 69.10% 和 68.08%。推理分数普遍中等，许多模型出现高伪正确率，说明答案正确但推理弱。

**⚠️ 局限性**

局限性包括：① 仍缺乏充分的证据 grounding 与推理一致性；② 在目标跟踪、手物交互定位等子任务上性能远低于人类；③ 基准样本量虽大但仍局限于 351 条视频，难以覆盖更广泛的动态场景；④ 评估依赖 LLM‑judge，可能带来判断偏差；⑤ 仅关注第一人称视角，尚未扩展到更复杂的多模态或长时序推理。

---

## 402. A Dual Physics-Informed Kolmogorov-Arnold Neural Network Framework for Continuum Topology Optimization

**arXiv ID:** 2605.19536 | [PDF](https://arxiv.org/pdf/2605.19536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 403. Replacement Learning: Training Neural Networks with Fewer Parameters

**arXiv ID:** 2605.19533 | [PDF](https://arxiv.org/pdf/2605.19533v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 404. Boosting Text-to-Image Diffusion Models via Core Token Attention-Based Seed Selection

**arXiv ID:** 2605.19532 | [PDF](https://arxiv.org/pdf/2605.19532v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 405. EventPrune: Cascaded Event-Assisted Token Pruning for Efficient First-Person Dynamic Spatial Reasoning

**arXiv ID:** 2605.19506 | [PDF](https://arxiv.org/pdf/2605.19506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning

**arXiv ID:** 2605.19491 | [PDF](https://arxiv.org/pdf/2605.19491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 407. Conflict-Freedom as a Progress Condition

**arXiv ID:** 2605.19531 | [PDF](https://arxiv.org/pdf/2605.19531v1)

**作者:** Petr Kuznetsov `[一作]` (Telecom Paris), Guillermo Toyos-Marfurt `[通讯]` (Telecom Paris)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了冲突无关（conflict‑freedom）进度条件，并给出一种线性可实现的通用构造（Universal Construction），能够在冲突消失时保证所有进程完成操作；

**💡 创新点**

将阻塞式（obstruction‑freedom）推广为冲突无关进度条件，定义了新的抽象 Generalized Commit‑Adopt（GCA）并利用其实现冲突无关的读写通用构造；

**🔧 技术方法**

采用 Mazurkiewicz 跟踪（traces）理论、GCA 对象、两阶段快照（snapshot）实现以及帮助机制（helping）来实现并证明冲突无关；

**📊 数据集**

论文未提供具体实验数据集或实现；

**📈 对比分析**

未给出实验或性能对比，理论上通过 GCA 实现的冲突无关构造在冲突消失时比传统阻塞式更快；

**⚠️ 局限性**

限制包括：1) 在存在冲突的并发执行下仍无法保证进度；2) 对于故障进程的冲突可能仍影响进度；3) 目前仅适用于共享内存模型，未扩展至消息传递或拜占庭容错。

---

## 408. Efficient Elicitation of Collective Disagreements

**arXiv ID:** 2605.19521 | [PDF](https://arxiv.org/pdf/2605.19521v1)

**作者:** Mohamed Ouaguenouni `[一作]` (Université Toulouse Capitole), Magdalena Tydrichova `[通讯]` (Centrale Supélec)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了多层次的多数表（plurality matrix），用来描述在不同大小子集上的投票者首选概率，从而统一研究群体争议度量；

**💡 创新点**

创新点在于：①引入争议度量层级概念，证明多种常见度量（如等级方差、分歧度）属于层级3；②证明层级严格递进，并给出每层度量的示例；③在单峰和Plackett–Luce结构下证明层级可降至层级2；④设计基于链式和完整排序的两种提问协议，并分析其样本复杂度与认知负担的权衡；

**🔧 技术方法**

主要技术包括：概率偏好建模、阶层化多数表、中心矩理论、Hoeffding不等式与合并分析、理论证明与实验验证；

**📊 数据集**

实验使用合成偏好分布：Mallows、Plackett–Luce、单峰、k‑Euclidean 等，以及对比不同度量在这些分布下的分布特征；

**📈 对比分析**

方法对比：链式协议在低认知负担或大样本时更优，完整排序协议在样本受限时更优；实验显示两种协议满足理论预测的样本复杂度，并在实际平台（如在线投票）中实现；

**⚠️ 局限性**

限制：无法覆盖基于Kemeny、秩距离等的多项争议度量；假设投票者独立、诚实且无噪声；认知负担仅用对比次数衡量；实验仅基于完整排序，未考虑top‑k或部分排序等场景；

---

## 409. Beyond Mode Collapse: Distribution Matching for Diverse Reasoning

**arXiv ID:** 2605.19461 | [PDF](https://arxiv.org/pdf/2605.19461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 410. Adynamical systems view of training generativemodels and the memorization phenomenon

**arXiv ID:** 2605.19483 | [PDF](https://arxiv.org/pdf/2605.19483v1)

**作者:** Siva Athreya `[一作]` (International Institute for Theoretical Sciences), Vivek S. Borkar `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 12551 | [OpenAlex ID](https://openalex.org/A5018541798)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究了生成模型（尤其是扩散模型）训练过程中出现的记忆化现象，并通过动态系统视角与“坍塌”现象联系，阐释了在常数步长随机梯度下降（SGD）下，模型会在不同时间尺度上出现间歇性记忆化的机制。

**💡 创新点**

创新点在于将记忆化现象解释为由两时间尺度随机逼近产生的坍塌行为所导致的间歇性停留，并结合常数步长SGD的Freidlin‑Wentzell理论提供对记忆化频率和持续时间的定量描述；同时提出了利用噪声尺度与步长平衡来抑制记忆化的理论思路。

**🔧 技术方法**

主要技术包括：两时间尺度随机逼近理论、Markov 噪声下的SGD解析、Austin 定理的可变尺度假设、Freidlin‑Wentzell 大偏差理论、控制理论中价值函数与哈密顿-雅可比方程的关系、以及对扩散模型与分数函数学习的抽象模型构建。

**📊 数据集**

本工作为理论分析，未使用具体数据集；所用的实验和案例均为理想化的抽象模型。

**📈 对比分析**

由于没有数值实验，本文未给出具体的性能比较指标；理论结论主要通过收敛性质、概率分布聚焦以及能量函数极小化的频率来描述。

**⚠️ 局限性**

局限性包括：模型对实际生成模型（如深度网络）的复杂性进行了大量简化；假设的条件（如条件重心属性、噪声统计、梯度可导性）在实践中可能不满足；对记忆化的定量预测尚缺乏实验验证；以及对大规模高维参数空间的可扩展性和计算成本未作评估。

---

## 411. Exposing Functional Fusion: A New Class of Strategic Backdoor in Dynamic Prompt Architectures

**arXiv ID:** 2605.19478 | [PDF](https://arxiv.org/pdf/2605.19478v1)

**作者:** Zeyao Liu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Xiaoshuang Ji `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了VIPER框架，利用轻量级的动态视觉提示生成器实现对ViT模型的高效且隐蔽的后门注入

**💡 创新点**

创新点在于通过动态上下文感知提示生成实现功能融合（Functional Fusion），将恶意逻辑与正面任务紧密绑定，解决传统PEFT攻击的准确率、效率与鲁棒性三难困境

**🔧 技术方法**

采用可训练的VPG（两层全连接网络）与联合优化的可学习触发器，对冻结的ViT骨干进行动态提示注入

**📊 数据集**

在ImageNet100、Caltech101、OxfordPets、Food101、DTD以及UCF101等六个数据集上进行评估

**📈 对比分析**

与传统的Backbone‑Overwriting、静态Adapter以及静态Prompt攻击相比，VIPER在保持或提升clean accuracy的同时，实现近乎100%攻击成功率，并在90%参数剪枝后仍保持100% ASR，推理延迟仅提升1%

**⚠️ 局限性**

局限性包括：仅针对ViT模型，未验证跨模型适用性；依赖训练阶段的联合优化，部署时可能需要额外的计算资源；未来更先进的动态路由检测或特征空间异常检测可能对其构成威胁

---

## 412. Drifting Objectives for Refining Discrete Diffusion Language Models

**arXiv ID:** 2605.19470 | [PDF](https://arxiv.org/pdf/2605.19470v1)

**作者:** Daisuke Oba `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3601 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在离散扩散语言模型（DDLM）中提出并实现了一种漂移（drift）训练目标，通过软令牌特征提升和抗对称漂移估计来改进固定步数下的生成质量。

**💡 创新点**

创新点在于将连续漂移理念迁移至离散文本，通过软令牌特征桥接实现可微分的漂移目标，保留了漂移的固定点结构并可直接作用于DDLM的logits。

**🔧 技术方法**

使用了软令牌特征提升、冻结语义编码器、吸引-排斥漂移估计（多温度、归一化）、特征空间固定点损失、可选的镜像教师（mirror teacher）等技术。

**📊 数据集**

在OpenWebText（OWT）数据集上进行实验，并在掩码扩散（MDLM）和均匀状态扩散（DUO）两种DDLM模型上评估。

**📈 对比分析**

通过对比原始检查点、普通继续训练和漂移训练，在相同训练预算和NFE（4/8/16步）下，漂移训练显著降低生成困惑度（MDLM 89%/86%提升），熵保持稳定，且优于现有蒸馏方法。

**⚠️ 局限性**

局限性包括对冻结语义特征空间质量的依赖，以及实验仅覆盖无条件生成；将该目标推广到条件生成、指令跟随等场景仍需进一步研究。

---

## 413. BLINKG: A Benchmark for LLM-Integrated Knowledge Graph Generation

**arXiv ID:** 2605.19518 | [PDF](https://arxiv.org/pdf/2605.19518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 414. Base Models Look Human To AI Detectors

**arXiv ID:** 2605.19516 | [PDF](https://arxiv.org/pdf/2605.19516v1)

**作者:** Yixuan Even Xu `[一作]` (Carnegie Mellon University), J. Zico Kolter `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17431 | [OpenAlex ID](https://openalex.org/A5075035644)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了名为 Humanization by Iterative Paraphrasing (HIP) 的无检测器依赖的文本人性化流程，利用微调基础模型并迭代改写 AI 生成文本。

**💡 创新点**

发现商业检测器更倾向于将基础模型输出判为人类，提出通过低失真和人类上下文的两种直觉来解释并利用这一现象。

**🔧 技术方法**

使用基础模型微调为轻量级重写器，采用 LoRA 等参数高效微调，随后进行迭代改写。

**📊 数据集**

使用 RAID 与 MAGE 两个公开文档式数据集生成训练对，后者经 GPT‑5‑nano 生成 AI 重写，形成 11757 对训练样本。

**📈 对比分析**

与 Simple Paraphrase、DIPPER、SilverSpeak、StealthRL 等基线在 GPTZero 与 Pangram 上对比，HIP 在语义保留与检测逃避的 Pareto 前沿上表现最优。

**⚠️ 局限性**

方法随检测器更新可能失效，且多轮改写会导致语义漂移，限制了其长期适用性。

---

## 415. TORQ: Two-Level Orthogonal Rotation for MXFP4 Quantization

**arXiv ID:** 2605.19561 | [PDF](https://arxiv.org/pdf/2605.19561v1)

**作者:** Zukang Xu `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种两级正交旋转量化方法（TORQ），用于将大型语言模型的权重量化到 4 位 MXFP4 格式，并在不显著降低模型性能的情况下实现高压缩率。

**💡 创新点**

创新点在于先进行粗粒度正交旋转去除大幅量化误差，再用细粒度旋转进一步校正残余误差，实现了更精准的两级量化策略；同时支持 Post‑Training Quantization，避免了大规模微调。

**🔧 技术方法**

使用了正交旋转（Orthogonal Rotation）、两级量化技术、MXFP4 量化格式、Post‑Training Quantization、可逆矩阵映射等关键技术。

**📊 数据集**

在主流大模型（如 LLaMA、BERT、GPT‑J 等）上使用公开评测数据集（LLaMA‑2 evaluation、GLUE、Wikitext‑2/103 等）进行实验验证。

**📈 对比分析**

与 GPTQ、AWQ、BitsBack 等现有 PTQ 方法对比，TORQ 在相同 4 位量化下平均提升 0.5–1.0% 的 Perplexity/Accuracy，模型尺寸压缩率提升至 4 倍，推理速度提升 10% 以上。

**⚠️ 局限性**

局限性包括：需要额外的旋转矩阵训练时间；在极低位数量化（≤3 位）时误差校正效果有限；正交旋转的可逆计算复杂度较高，对某些模型架构可能不易推广。

---

## 416. Towards Camera-Robust 3D Localization: Equation-Anchored Tool-Use for MLLMs

**arXiv ID:** 2605.19528 | [PDF](https://arxiv.org/pdf/2605.19528v1)

**作者:** Xueying Jiang `[一作]` (Nanyang Technological University), Ran Xu `[通讯]` (Damo Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计了一种公式锚定的工具使用框架，使多模态大语言模型在单幅RGB图像上主动调用相机内参工具和多点深度采样工具，并将返回值作为公式变量在 Chain‑of‑Thought 中显式代入针孔投影方程，从而完成 3D 目标检测和 3D 视觉定位任务并显著提升相机尺度鲁棒性。

**💡 创新点**

核心创新在于将工具输出从软参考提示转化为严格的公式变量，并在推理链中显式进行符号计算，构建可审计的几何推理流程，彻底消除相机内参模糊导致的深度和尺度不确定性。

**🔧 技术方法**

技术手段包括：基于 Chain‑of‑Thought 的工具调用机制、相机内参提取工具、Multi‑Point Depth Sampling 工具、针孔投影方程、结构化公式化的监督微调（SFT）以及利用 Qwen3.5、Qwen3‑VL‑8B‑Instruct 等大型模型；同时使用 UniDepthV2 等深度估计模型作为工具后端。

**📊 数据集**

实验使用 ScanNet（3D 目标检测）和 ScanRefer（3D 视觉定位）数据集，结合 SAM3 生成掩模并使用 UniDepthV2 或 GT 深度作为工具输出。

**📈 对比分析**

与多种通用与专业 MLLM（如 GPT‑5.4、DeepSeek‑v4‑Flash、Claude‑Sonnet‑4.6、Qwen3.6‑Flash、VG‑LLM、Qwen3‑VL‑8B‑Instruct 等）在 0.5×–1.5× 相机缩放因子下进行对比，采用 Avg‑F1（IoU = 0.25）和 Acc@IoU = 0.25 评估。实验显示，本文方法在所有尺度下均优于基线，尤其在相机偏离训练尺度时提升显著。

**⚠️ 局限性**

主要局限包括：对外部深度估计工具的精度高度依赖，工具噪声或误差会直接影响最终 3D 预测；目前仅支持单帧静态场景，缺乏对动态环境和连续帧的处理；以及实现的链式推理过程对计算资源和推理时延有一定要求。

---

## 417. Are Watermarked Images Editable? SafeMark for Watermark-Preserving Text-Guided Image Editing

**arXiv ID:** 2605.19511 | [PDF](https://arxiv.org/pdf/2605.19511v1)

**作者:** Xiaodong Wu `[一作]` (Queen's University), Jianbing Ni `[通讯]` (Queen's University)

**通讯引用:** 6882 | [OpenAlex ID](https://openalex.org/A5033931001)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SafeMark 框架，实现文本引导的图像编辑过程在保持语义编辑质量的同时保证水印完整性。

**💡 创新点**

创新点在于将水印解码准确率作为阈值铰链损失直接嵌入编辑器训练目标，并给出信息论上可证明的 MI 下界，形成端到端的水印安全编辑。

**🔧 技术方法**

采用可微分扩散编辑器、Brier 型软准确率、阈值铰链惩罚以及信息论分析，辅以固定的参考编辑以保持语义一致性。

**📊 数据集**

实验数据集包括 LSUN‑Church/Bedroom、CelebA、AFHQ‑Dog 的真实图像以及 Stable Diffusion 生成的 Church、Bedroom、Human、Dog 图像，并使用 HiDDeN、VINE、SleeperMark、Stable Signature 等水印方案。

**📈 对比分析**

与 VINE、SleeperMark、HiDDeN、Stable Signature 等基线在 DiffusionCLIP、Asyrp、Eff‑Diff 三种编辑器下对比，SafeMark 在水印位误差低于 5%（接近 1.0 近似）且 FID/IS/CLIP 与原始编辑差异不超过 1‑2 点，证明水印完整性与编辑质量可兼得。

**⚠️ 局限性**

限制在于对极端结构性改动时水印约束可能与编辑目标冲突；需要白盒访问编辑器并对每个模型单独微调；仅保证水印可恢复，无法判断图像是否被大幅修改。

---

## 418. Accelerating Loops with Arrays

**arXiv ID:** 2605.19499 | [PDF](https://arxiv.org/pdf/2605.19499v1)

**作者:** Florian Frohn `[一作]` (RWTH Aachen University), Jürgen Giesl `[通讯]` (RWTH Aachen University)

**通讯引用:** 4169 | [OpenAlex ID](https://openalex.org/A5025232172)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新型的循环加速技术，针对数组操作的单路径循环，利用归纳 lvalue、λ 表达式和量化消除实现闭式闭包，支持在 SMT 求解器中无量词求解。

**💡 创新点**

创新点包括：① 引入归纳 lvalue 统一数组与标量的加速；② 用 λ 表达式代替传统量词描述闭式；③ 开发针对常数步长索引的量化消除方法；④ 结合“按需引入子句”实现无量词 SMT 求解。

**🔧 技术方法**

使用递推求解、闭式生成、归纳 lvalue 变换、λ 表达式、量化消除、按需子句生成、ABMC (加速的有界模型检查)、LoAT 及 SwInE（基于 Z3 的 SMT 求解器）实现。

**📊 数据集**

在 SV‑COMP 的 ReachSafety‑Arrays 领域使用 201 个可转换为 CHC 的实例，构成 SV 集合；再改写断言得到 201 个 UNSAT 的 SV_ 集合，包含 13 个曲折数组实例。

**📈 对比分析**

将 LoAT 与 Eldarica、Golem、Z3 进行比较，并与 LoAT 的基线版本 LoAT BL 对比；在 SV_ 集合中，LoAT 解决 129 个实例，Eldarica 78，LoAT BL 65，Z3 62，Golem 30；在 SV 集合中，LoAT 仅能解决少量实例，表明该领域难度高；运行时曲线显示 LoAT 在已解决实例上执行速度快，整体性能优于基线且在多数对手之上。

**⚠️ 局限性**

局限性包括：不支持曲折数组、指针、结构体和位运算；只能处理索引在每次迭代中按常数步长变化的循环；需满足 a‑solvable 与递推可解性；对非线性算术、某些 SV‑COMP 基准的求解仍受限；HornKlaus 仅支持 C 的子集，未完全保留安全性。

---

## 419. Quantifying the Pre-training Dividend: Generative versus Latent Self-Supervised Learning for Time Series Foundation Models

**arXiv ID:** 2605.19462 | [PDF](https://arxiv.org/pdf/2605.19462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 420. Building Acoustics 01: Finite Element Model of an Building Acoustics Test Facility to Predict the Sound Transmission Loss Based on DIN EN ISO 10140

**arXiv ID:** 2605.19492 | [PDF](https://arxiv.org/pdf/2605.19492v1)

**作者:** Sebastian Schmidt `[一作]`, Sabine C. Langer `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了符合 DIN EN ISO 10140‑5 标准的全尺度建筑声学测试设施有限元模型，预测单叶墙与双叶墙（含/不含隔热材料）在 8 Hz–630 Hz 范围内的声传输损失（STL），并通过 SALOME 网格生成与 elPaSo 并行求解器完成数值仿真。

**💡 创新点**

创新点在于采用非一致网格与域–频率特定离散化策略，结合等效流体（Johnson‑Champoux‑Allard）模型和简化的双叶墙耦合，显著降低求解成本，并将完整的基准数据集公开发布。

**🔧 技术方法**

主要技术包括有限元法（Helmholtz 方程）、等效流体模型、等效刚性框架、Mindlin 板理论、SALOME 9.14 进行几何与网格生成，以及 elPaSo（C 语言并行求解器）实现矩阵组装与求解。

**📊 数据集**

使用公开的测试设施几何和材料参数（空气、玻璃棉、石膏板），以及在 Zenodo 上发布的 STL 估计数据集作为验证与对比基础。

**📈 对比分析**

通过在小型模型中与 COMSOL 6.3 进行验证，误差均小于 0.1 dB；在大规模模型中采用非一致网格后，求解时间从约 15 h 降至 1/3（≈ 5 h），内存占用低于工作站 RAM 的 10%，表明实现了高效可行的数值预测。

**⚠️ 局限性**

局限性包括仅验证至 630 Hz，未覆盖更高频范围；未考虑侧壁传递和连接节点的细节；等效流体假设在低/高频可能产生误差；求解仍需进一步并行化多频段和模型降阶以提升效率。

---

## 421. Worst-Case Utility Privacy Mechanism via Pointwise Maximal Leakage

**arXiv ID:** 2605.19474 | [PDF](https://arxiv.org/pdf/2605.19474v1)

**作者:** Ci Song `[一作]` (KTH Royal Institute of Technology), Tobias J. Oechtering `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 2605 | [OpenAlex ID](https://openalex.org/A5079492269)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计了一种在点级最大泄露（Pointwise Maximal Leakage，PML）约束下的隐私机制，专门优化最坏情况下的效用。

**💡 创新点**

创新点在于利用PML允许的概率零化特性，结合PML的分解性质，推导出可直接计算的“效用安全机制”（utility-safe mechanism），在保持PML隐私保障的同时显著提升最坏效用；并证明其在大多数情况下是最优的。

**🔧 技术方法**

技术手段包括：PML的线性多面体约束建模、效用序列化、二分搜索求解最优效用阈值、利用PML分解降低复杂度、在特殊情况下使用线性规划验证最优性。

**📊 数据集**

使用了一个计数查询的仿真数据集（输入输出字母表大小为7，先验概率均等的离散分布），以及构造的二次损失函数矩阵作为效用示例。

**📈 对比分析**

与传统DP导向的指数机制和随机响应机制在相同PML隐私预算下进行对比。实验表明：当ε超过特定阈值时，效用安全机制的最坏效用呈现显著跳跃并高于DP机制，而DP机制的最坏效用随ε变化基本保持不变。

**⚠️ 局限性**

局限性：需要对先验分布完全知情；在某些先验模式和低效用阈值下，效用安全机制并非最优，需要额外的线性规划求解；适用于离散有限字母表，对连续或高维数据尚未验证。

---

## 422. EpiDiffVO: Geometry-Aware Epipolar Diffusion for Robust Visual Odometry

**arXiv ID:** 2605.19556 | [PDF](https://arxiv.org/pdf/2605.19556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 423. Resilient Byzantine Agreement with Predictions

**arXiv ID:** 2605.19452 | [PDF](https://arxiv.org/pdf/2605.19452v1)

**作者:** Julien Dallot `[一作]` (TU Berlin), Patrik Welters `[通讯]` (HU Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出了基于全局预测的拜占庭协议，利用机器学习预测提升一致性与鲁棒性。

**💡 创新点**

创新点在于给出了一致性-鲁棒性平衡以及误差平滑度的完整理论边界，并设计对应算法和不可达性证明。

**🔧 技术方法**

核心技术包括 Phase King 协议、可辨识性论证与误差函数（η）的定义与分析。

**📊 数据集**

实验未使用任何数据集，全部为理论分析与证明。

**📈 对比分析**

与传统无预测协议比较，证明在预测准确时可容忍 αn 个错误节点，鲁棒性提升至 (1-α/2)n-1；当预测误差线性增加时，容错性能以每误差降低一单位的速率下降，整体优于传统协议。

**⚠️ 局限性**

局限性包括仅在同步网络中讨论、未分析时间/通信复杂度、对局部预测几乎无提升、无法在异步环境下直接应用。

---

## 424. Self-Creative Text-to-Object Generation using Semantic-Aware Spatial Weighting

**arXiv ID:** 2605.19554 | [PDF](https://arxiv.org/pdf/2605.19554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 425. AnchorFlow: Editable SVG Reconstruction via Sparse Anchor Point Fields

**arXiv ID:** 2605.19551 | [PDF](https://arxiv.org/pdf/2605.19551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 426. CaptchaMind: Training CAPTCHA Solvers via Reinforcement Learning with Explicit Reasoning Supervision

**arXiv ID:** 2605.19538 | [PDF](https://arxiv.org/pdf/2605.19538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 427. Optimising Neural Speech Codecs for 300bps Communication using Reinforcement Learning

**arXiv ID:** 2605.19541 | [PDF](https://arxiv.org/pdf/2605.19541v1)

**作者:** Junyi Wang `[一作]` (Tsinghua University), Chao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 20638 | [OpenAlex ID](https://openalex.org/A5100460246)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在300 bps极低比特率下的神经语音编码器ClariCodec，并通过强化学习对编码器进行可解释的语义优化。

**💡 创新点**

创新点在于将量化过程重新表述为可微分的随机策略，利用GRPO与WER奖励直接优化可懂度，并在RL阶段仅冻结解码器以保持声学质量。

**🔧 技术方法**

采用ConvNeXt V2编码器、Vocos声码器、改进的FSQ量化、Gumbel‑Softmax随机采样、GRPO强化学习、以及Mel谱重建正则化。

**📊 数据集**

使用Libriheavy 50,000 h语音数据进行预训练，评估数据为LibriSpeech的test‑clean和test‑other子集。

**📈 对比分析**

在与8个基准系统（如StableCodec、FlexiCodec、SAC等）比较时，ClariCodec在仅300 bps下实现了3.55 %（test‑clean）和10.4 %（test‑other）的WER，显著优于同等或更高比特率的基准，同时保持相近的MOS与音质指标。

**⚠️ 局限性**

局限性包括：目前模型为非实时的非因果架构，缺乏低延迟流式实现；RL优化虽提升可懂度，但在极低比特率下仍存在声学质量与语义优化的权衡；对生成任务与不同语音内容的泛化性尚待进一步验证。

---

## 428. Generative-Evaluative Agreement: A Necessary Validity Criterion for LLM-Enabled Adaptive Assessment

**arXiv ID:** 2605.19529 | [PDF](https://arxiv.org/pdf/2605.19529v1)

**作者:** Grandee Lee `[一作]` (Singapore University of Social Sciences), Luke Peh `[通讯]` (Singapore University of Social Sciences)

**通讯引用:** 130 | [OpenAlex ID](https://openalex.org/A5025451689)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并评估了 Generative‑Evaluative Agreement (GEA)，即在同一 LLM 生成题目、模拟学生回答并评分时，检验生成与评估路径对技能水平的一致性，并通过 Claude Sonnet 4.6 在 Python OOP 任务上对 150 个合成学生的 24 维技能向量执行 generate‑then‑score 流程，计算 Pearson r 与偏差以量化 GEA；

**💡 创新点**

创新点在于首次将 GEA 定义为 LLM 自适应测评的必要有效性指标，揭示生成与评估路径的内部偏差，发现语法可验证技能 GEA 强、设计级技能 GEA 近零，并提出细粒度 rubrics、跨模型评估等对策以提升一致性；

**🔧 技术方法**

技术方法包括使用 LLM（Claude Sonnet 4.6 及 Haiku 4.5）进行生成与评分，构造 24 维技能向量与结构化 rubrics，采用 Bootstrap CI、Pearson r、平均偏差等统计指标对生成-评分一致性进行评估；

**📊 数据集**

数据集为 150 个人工合成的学生技能配置（10 个 archetype 加高斯噪声），每个学生在 6 个评测任务中完整回答，覆盖 23/24 个技能，数据完全模拟而非真实学生；

**📈 对比分析**

比较方法是对每个（学生、任务）对比真实技能向量与模型观察评分，得到全局 Pearson r = 0.698（95% CI .684–.712）且平均正偏差 +0.059；按技能分层显示强 GEA (>0.7) 与弱 GEA (<0.1)，并与 Haiku 4.5 的结果对比，展示 GEA 随模型规模变化的差异；

**⚠️ 局限性**

局限性包括仅使用合成学生样本、单一编程领域、两款 Claude 模型；缺乏真实学生验证、跨模型和主观领域的泛化检验；未对 rubric granularity 与 scorer identity 进行消融实验；以及可能出现生成器与评估器共享偏差导致 GEA 误判的风险。

---

## 429. Return of Frustratingly Easy Unsupervised Video Domain Adaptation

**arXiv ID:** 2605.19510 | [PDF](https://arxiv.org/pdf/2605.19510v1)

**作者:** Pengfei Wei `[一作]` (Magellan Technology Research Institute), Lawrence B. Hsieh `[通讯]` (Magellan Technology Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种只用两个损失的无监督视频域适配方法 MetaTrans，并通过时间-静态子模块实现空间与时间差异的分离

**💡 创新点**

设计了时间-静态减法模块，利用 Transformer 的时间置换不变性实现静态特征估计并去除空间域差异，保持了极简损失结构的同时获得了强大的适配性能

**🔧 技术方法**

基于 I3D RGB 特征提取、双流 Transformer（自注意力、位置嵌入、平均池化）、对抗域分类、伪标签学习、t‑SNE 可视化以及理论误差界证明

**📊 数据集**

UCF‑HMDB（UCF→HMDB、HMDB→UCF）与 Epic‑Kitchens（三域交叉任务）

**📈 对比分析**

与多种 Image‑UDA、UVDA 先前方法以及源/目标监督基准在准确率上比较，MetaTrans 在 UCF‑HMDB 上平均 95.4%（比最优竞争对手高 1.6%），Epic‑Kitchens 上平均 51.0%（仅次于 TranSVAE 52.6%），且因损失数仅为 2，RGRA 指标最高

**⚠️ 局限性**

仍依赖 RGB 特征、对计算资源要求高；方法在多模态或更复杂场景的泛化尚未验证；缺乏对伪标签误差的鲁棒性分析

---

## 430. Closed-Loop Hybrid Digital Twin Platform for Connected and Automated Vehicle Validation

**arXiv ID:** 2605.19490 | [PDF](https://arxiv.org/pdf/2605.19490v1)

**作者:** Kanglong Quan `[一作]` (Xi'an Jiaotong-Liverpool University), Dongyao Jia `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文设计并实现了一个实时混合数字孪生平台，用于在真实车辆与高保真CARLA‑SUMO共模拟环境之间实现双向低延迟闭环控制，并通过云‑边缘协同架构支持多用户交互式验证。

**💡 创新点**

创新点包括：①将高保真仿真与真实V2X硬件无缝耦合，实现了真实车辆状态向仿真世界的实时注入和仿真控制指令到CAN总线的即时下发；②提出低延迟的物理‑数字桥接中间件，采用UDP+标记过滤实现 <50 ms 的双向通信；③利用摄影测量与神经渲染构建与真实试验场完全一致的三维地图与车辆模型，显著缩小 sim‑real 差距；④引入云‑边缘协同，局部低延迟渲染+全局状态同步，提升多用户可扩展性。

**🔧 技术方法**

核心技术包括：CARLA 与 SUMO 的多尺度共仿；V2X 通信链路（OBU‑RSU‑工作站）与 UDP 轻量级消息传输；中间件三路管线（消息网关、影子车辆同步、CAN 编码）；摄影测量（UAV 影像、RealityScan、COLMAP、3DGS、SuGaR、RoadRunner）构建物理一致的车辆与场景模型；云‑边缘分层架构与 WebSocket、TCP 的协同通信。

**📊 数据集**

使用的主要数据集为：①无人机全景摄像采集的4K航拍图像（用于车辆与场景摄影测量）；②LiDAR点云（用于实时定位与场景对齐）；③自建的神经辐射场（3DGS）与三角网格（SuGaR）作为可碰撞的静态地图；并未引用公开的标准数据集，而是通过上述采集与重建得到的原始数据。

**📈 对比分析**

通过对比传统仅仿真共模平台、SimCCAD、MCCT等框架，本文在同步误差（≈0.01–0.04 m）、整体闭环延迟（<50 ms）以及多用户场景下的低延迟渲染表现上显著优于全云端方案。实验中在连续 8 秒内位置误差保持在 3.6 cm 以内，且闭环控制能稳定实现多种驾驶行为。

**⚠️ 局限性**

局限性：①当前仅支持单辆真实车与仿真影子同步，尚未实现多辆真实车同时映射；②V2X链路使用有线网络，未完全覆盖无线波束/丢包等真实网络干扰；③在极端交通或复杂环境下的鲁棒性待验证；④对高级驾驶决策（如紧急刹车、车道变换）尚未在真实车辆上进行闭环验证，后续需扩展算法层与多车协同。

---

## 431. Attention-Guided Reward for Reinforcement Learning-based Jailbreak against Large Reasoning Models

**arXiv ID:** 2605.19485 | [PDF](https://arxiv.org/pdf/2605.19485v1)

**作者:** Zheng Lin `[一作]` (Xidian University), Haichang Gao `[通讯]` (Xidian University)

**通讯引用:** 1696 | [OpenAlex ID](https://openalex.org/A5086029723)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对大型推理模型（LRMs）的基于强化学习的 jailbreak 方法（AGR），通过对模型内部注意力分布的引导实现 prompt 优化。

**💡 创新点**

创新点在于发现 jailbreak 成功与输入提示与推理链中的注意力比例紧密相关，并将此注意力模式作为奖励函数；同时构建了包含 17 种提示改写与说服策略的动作空间，使 RL 策略能够自适应选择与组合。

**🔧 技术方法**

技术包括注意力分析、线性 SVM 产生奖励、PPO 强化学习、对抗提示生成与重写、攻击模型与目标模型交互、恶意词提取（词典+LLM）、多步策略的 Prompt 转换。

**📊 数据集**

使用 AdvBench、StrongReject、HarmBench 三大 benchmark 进行评测，并在 Qwen3‑1.7B / Qwen3‑8B / DeepSeek‑R1‑Distill‑Llama‑8B 等开源 LRM 上验证；同时将生成的 jailbreak 提示迁移到闭源模型 o4‑mini 与 Gemini‑2.5‑Flash。

**📈 对比分析**

与 LLM 基础攻击（GCG、AutoDAN、PAIR、ReNeLLM）以及 LRM 专用攻击（H‑CoT、AutoRAN）比较，AGR 在 ASR/ASR‑T 上分别高达 98%/96%（开源）和 71%（闭源），平均成功回合数低于 2，推理时延仅 10.8 秒，显示出更高效、更强迁移与鲁棒性。

**⚠️ 局限性**

局限性包括：需要对模型内部注意力进行可观测；对恶意词提取的准确性敏感；RL 训练成本较高；在未来新的安全机制或注意力机制变更时可能失效。

---

## 432. Implicit Bias of Mirror Flow in Homogeneous Neural Networks: Sparse and Dense Feature Learning

**arXiv ID:** 2605.19458 | [PDF](https://arxiv.org/pdf/2605.19458v1)

**作者:** Tom Jacobs `[一作]` (CISPA Helmholtz Center), Guido Montufar `[通讯]` (UCLA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了镜像流在同质神经网络中的隐式正则化特性，推导了新的平衡方程并提出了 Q‑margin，证明了镜像流收敛到由 horizon 函数决定的最大间隔解，并通过实验验证了不同镜像映射导致稀疏或稠密特征学习的效果。

**💡 创新点**

创新点在于首次为非线性同质网络推导出基于凸双对偶的平衡方程，定义了 Q‑margin 并给出了其最大间隔收敛性；同时揭示镜像参数 λ 对收敛速度与特征稀疏度的双重影响。

**🔧 技术方法**

采用镜像流（continuous-time mirror descent）、凸分析（Fenchel‑Young 关系）、Q‑margin 与 horizon 函数理论、以及梯度流的延伸技术，并结合实验中的 VGG‑16、CIFAR‑10 等模型。

**📊 数据集**

主要使用合成可分数据集验证理论，并在标准视觉任务 CIFAR‑10 上训练 VGG‑16 进行实证比较。

**📈 对比分析**

与传统梯度下降相比，镜像流（尤其是超参数 λ 较小的双曲熵映射）在 CIFAR‑10 的验证精度更高，且能产生更稀疏的权重分布，导致剪枝时性能衰减更小；相反，平滑同质镜像映射产生更稠密权重，剪枝时性能下降更明显。

**⚠️ 局限性**

主要局限在于对 α<2 的镜像映射分析尚未完成；收敛速度受 λ 影响显著，过大 λ 可能导致指数级慢收敛；实验未深入探讨有限学习率、早停策略以及最终最大间隔解的泛化性质。

---

## 433. iDiff: Interpretable Difference-aware Framework for Pairwise Image Quality Assessment

**arXiv ID:** 2605.19522 | [PDF](https://arxiv.org/pdf/2605.19522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 434. Provable Fairness Repair for Deep Neural Networks

**arXiv ID:** 2605.19549 | [PDF](https://arxiv.org/pdf/2605.19549v1)

**作者:** Jianan Ma `[一作]` (Hangzhou Dianzi University), Zhen Wang `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 73215 | [OpenAlex ID](https://openalex.org/A5100460802)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对深度神经网络进行可证明的个体公平性修复，确保在相似输入集上模型输出保持一致。

**💡 创新点**

创新点在于将间隔边界传播与符号边界相结合，构建统一约束求解模型，利用对偶定理把非线性变为MILP，从而实现可证明的公平修复，并通过逐步收束特征边界提升修复效果。

**🔧 技术方法**

使用技术包括间隔边界传播（auto‑LiRPA）、符号边界推导、对偶理论、混合整数线性规划（MILP）以及 Big‑M 方法。

**📊 数据集**

实验数据集为 Adult、German Credit、Bank Marketing、Compas 四个广泛使用的公平性基准数据集。

**📈 对比分析**

与 FLIP、CARE、GRFT 等基线相比，在 CUR、IDI‑D、IDI‑S 等公平性指标上实现接近 100% 的提升，公平率提高 95%+，准确率下降 ≤3%，平均运行时间约 18 秒。

**⚠️ 局限性**

局限性包括 MILP 的规模随整数变量指数增长，可能限制对更大或更深网络的应用；仅适用于最后一层线性；多分类场景需要进一步扩展。

---

## 435. C2CServe: Leveraging NVLink-C2C for Elastic Serverless LLM Serving on MIG

**arXiv ID:** 2605.19481 | [PDF](https://arxiv.org/pdf/2605.19481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 436. Sampling-Based Safe Reinforcement Learning

**arXiv ID:** 2605.19469 | [PDF](https://arxiv.org/pdf/2605.19469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 437. Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images

**arXiv ID:** 2605.19656 | [PDF](https://arxiv.org/pdf/2605.19656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 438. A parallel wakeup problem and multi-room light switch strategies

**arXiv ID:** 2605.19488 | [PDF](https://arxiv.org/pdf/2605.19488v1)

**作者:** John Haslegrave `[一作]` (Lancaster University), Mark Walters `[通讯]` (Queen Mary University of London)

**通讯引用:** 11037 | [OpenAlex ID](https://openalex.org/A5002965880)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文对多间不可区分房间、具有限制状态数的囚犯问题进行了系统性分析，并证明了多种情形下是否存在赢取策略；

**💡 创新点**

创新点在于完整解决了 Kane 与 Kominers 的开放问题，证明了 r=q=2 时无解、q=3 时最多五名囚犯可赢（除 r=3 情形外），并确立了对称策略存在的必要与充分条件 (n,r)=1；

**🔧 技术方法**

采用了组合协议设计、归纳与单调性论证、领导者寻找与块调度等理论技术来构造或否定赢取策略；

**📊 数据集**

该工作为纯理论研究，无需实验数据集；

**📈 对比分析**

通过严格证明和构造对照，本文在理论上提升了已知阈值和条件，但未给出实验性能指标；

**⚠️ 局限性**

局限性包括对 r=3、q=3 的完整结果尚未确定，关于更多囚犯和房间数的递增性仍不明确，且所需状态数的上界仍未知。

---

## 439. P2DNav: Panorama-to-Downview Reasoning for Zero-shot Vision-and-Language Navigation

**arXiv ID:** 2605.19634 | [PDF](https://arxiv.org/pdf/2605.19634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 440. The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility

**arXiv ID:** 2605.19537 | [PDF](https://arxiv.org/pdf/2605.19537v1)

**作者:** David Pape `[一作]` (CISPA Helmholtz Center for Information Security), Lea Schönherr `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM推理引擎对基准性能的影响，并系统评估了多种推理后端；

**💡 创新点**

提出推理后端是未被记录的重要超参数，通过根因分析揭示了默认设置和数值漂移对评测结果的显著影响；

**🔧 技术方法**

采用五大主流推理后端（vLLM、SGLang、llama.cpp、LMDeploy、Ollama）及其CUDA/FP16/FP32优化技术进行对比实验；

**📊 数据集**

使用GSM8K、GPQA Diamond、SimpleQA Verified、LiveCodeBench v6四个公开评测数据集；

**📈 对比分析**

在保持模型权重、解码策略与硬件一致的条件下，比较各后端在准确率、分歧率和长度误差上的差异，发现后端差异可达16.6个百分点，足以改变模型排名；

**⚠️ 局限性**

实验局限于单一GPU、greedy解码、未覆盖高负载或多线程场景，且部分优化难以完全禁用，导致观察到的差异为下限估计。

---

## 441. Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection

**arXiv ID:** 2605.19652 | [PDF](https://arxiv.org/pdf/2605.19652v1)

**作者:** Ravishka Rathnasuriya `[一作]` (University of Texas at Dallas), Tao Xie `[通讯]` (Peking University)

**通讯引用:** 17869 | [OpenAlex ID](https://openalex.org/A5048118068)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性分析了 301 条真实的 tile 程序代码生成错误，构建了六大根因分类并提出了对应的检测与修复策略。

**💡 创新点**

首次将代码生成错误映射到 tile 级别的抽象与编译阶段，揭示了控制流、IR、映射、内存、类型与设备特定等六类独特缺陷。

**🔧 技术方法**

采用基于 GitHub issue 的挖掘、人工标签、IR 级别分析以及差分测试、符号回归等技术。

**📊 数据集**

数据集为 401 条经过筛选的公开 bug 报告，其中 301 条被确认为 tile 代码生成 bug。

**📈 对比分析**

通过与已有的 GPU 编译器 fuzzer、差分测试等方法对比，验证了提出的检测策略能覆盖 90% 以上的典型 bug，修复效率提升约 30%。

**⚠️ 局限性**

局限于公开仓库的 bug，未覆盖私有代码或深度自定义框架，且缺乏自动化修复工具。

---

## 442. Executable Boundary Contracts for Sound Event Traces

**arXiv ID:** 2605.19632 | [PDF](https://arxiv.org/pdf/2605.19632v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Hamdi Alakkad `[通讯]` (Bahcesehir University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种可执行的有限声音事件边界合约语言，提供帧层与事件层的规范化语法、匹配与计分机制，并通过可执行监视器将合约转换为可量化的向量。

**💡 创新点**

创新点在于：①将声音事件检测的边界行为拆分为可执行的约束向量，避免单一标量指标压缩错误信息；②引入基于有限时序逻辑的约束语法与严格的解析、评估流程；③通过匹配策略与容忍度可调节的规则实现对持续时间、碎片化等细粒度错误的显式检测。

**🔧 技术方法**

使用的技术包括：基于信号时序逻辑（STL）限定的帧级布尔语句、事件匹配与持续时间/碎片化谓词；递归下降解析器+词法分析器；离线与流式监视器；多种检测器（阈值、逻辑回归、CNN、残差扩张TCN、边界感知模型）与冻结预训练编码器（wav2vec2、AST、HTS‑AT、BEATs）进行特征提取；统计学的置信区间与容忍度扫频等评估手段。

**📊 数据集**

主要使用的数据集：①基于 Mini LibriSpeech、合成音调与噪声扰动的控制实验集；②MAESTRO Real 真实音景与软强标签；③DCASE 2024 Task 4 官方基线输出；④公开数据集如 AudioSet、DESED、UrbanSound8K 用于背景说明。

**📈 对比分析**

方法与传统 SED 评估相比：在同一数据集上分别计算帧 F1、段 F1、事件 F1、边界 F1 以及本研究的合约向量。实验表明，合约向量能揭示标准指标压缩的错误，例如在受混响影响时帧 F1 仍高但边界 F1 与持续时间/碎片化分量低。边界感知模型在 union 轨迹上取得约 0.83 的边界 F1 与 0.80 的逻辑平均分，显著优于未改进的基线（≈0.49），但其 class‑macro 仍较低，体现了合约检测的细粒度优势。

**⚠️ 局限性**

局限性包括：①仅针对离散化后有限长度的时间序列，未处理连续时序；②匹配策略与容忍度设置对结果有显著影响，需要手动校准；③合约框架侧重边界与持续时间的评估，对类别标识误差、语义层面评估支持不足；④实现复杂度高，解析器与监视器的性能需针对大规模音频优化。

---

## 443. HEAT: Heterogeneous End-to-End Autonomous Driving via Trajectory-Guided World Models

**arXiv ID:** 2605.19631 | [PDF](https://arxiv.org/pdf/2605.19631v1)

**作者:** Hoonhee Cho `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

训练一个单一的端到端驾驶模型，在多种异构数据域（nuScenes、NAVSIM、Waymo）上实现统一且鲁棒的驾驶性能。

**💡 创新点**

提出轨迹驱动的学习框架：利用轨迹聚类生成域不变原型，结合世界模型预测未来特征并通过对比学习和记忆检索实现域不变的行为表征。

**🔧 技术方法**

使用Swin-Transformer编码、轨迹条件世界模型、聚类原型对比学习、视觉-动作记忆模块以及多任务损失（轨迹回归、重构、对比）。

**📊 数据集**

在nuScenes、NAVSIM（基于nuPlan）和Waymo End-to-End三个公开数据集上进行训练和评估。

**📈 对比分析**

与多种现有 E2E-AD 方法（UniAD、VAD、BEVPlanner++、LAW 等）在开放循环 L2 误差和闭环 PDMS 指标上对比，HEAT 在三域均优于对比模型，尤其在闭环评估中取得最优成绩。

**⚠️ 局限性**

仅在已知域进行评估，未验证对全新未见域的泛化；依赖多域标注且对聚类数等超参敏感；在实时推理时计算量相对较大。

---

## 444. Online Market Making and the Value of Observing the Order Book

**arXiv ID:** 2605.19584 | [PDF](https://arxiv.org/pdf/2605.19584v1)

**作者:** Davide Maran `[一作]` (Politecnico di Milano), Marcello Restelli `[通讯]` (Politecnico di Milano)

**通讯引用:** 3389 | [OpenAlex ID](https://openalex.org/A5017130830)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于行动相关反馈的在线市场做市模型，并在该模型下设计了三种算法（OPSR、LazyOPSR、Explore‑then‑Perturb），给出相应的收敛率分析。

**💡 创新点**

创新点在于将无交易时可观测买卖价值的真实订单簿信息作为行动相关反馈，打破传统全隐或全显反馈限制；在此结构下即使不假设分布光滑性也能实现 √T 的 regret。

**🔧 技术方法**

采用消除式在线学习（OPSR）、懒惰更新（LazyOPSR）、探索‑扰动（ETP）等算法，结合新的集中不等式以及自回归/全局均值回归假设，对价过程进行统计分析。

**📊 数据集**

论文主要为理论分析，未使用真实金融数据；若做实验，使用的是仿真或公开行情数据（具体未给出）。

**📈 对比分析**

与传统bandit反馈下的基准算法（如常规bandit、持续动作空间算法）比较，证明在随机/均值回归情形下 regret 为 O(√T log T)，在对抗性价格下得到 O(T^{2/3} log^{1/3} T) 的期望上界，明显优于已知的对抗性 bandit 结果。

**⚠️ 局限性**

局限性包括：未考虑库存约束和双侧订单流；对手可适应行动的对抗性情形仍未解决；对抗性价格下的界限仅为期望，缺乏高概率保证。

---

## 445. A Data-Driven Approach to Idiomaticity Based on Experts' Criteria in Theoretical Linguistics

**arXiv ID:** 2605.19575 | [PDF](https://arxiv.org/pdf/2605.19575v1)

**作者:** Elena Mikhalkova `[一作]`, Timofey Protasov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对286个俄语多词表达(MWE)进行专家标注，构建了16个语义、句法、语用等特征的向量，随后通过三维可视化聚类分析其异质性。

**💡 创新点**

创新点在于提出了以语料为基础的16项特征模型并将其分为四大类（词汇变化、语法变化、陈旧性、可替换性），用向量化+可视化手段揭示MWE多样性与词汇特征的关联。

**🔧 技术方法**

采用手工专家标注、俄语语料库（RusCorpora）检验标注质量，并利用多维向量空间与三维散点图实现聚类可视化；未使用机器学习模型，而是以数据驱动的统计分析为核心。

**📊 数据集**

数据集为来自俄语学术文献的286个MWE实例，附带对应的翻译及语法/句法特征标注；在验证阶段用语料库检查插入、删除等可变性特征。

**📈 对比分析**

本文未给出与已有方法的数值对比，仅通过统计分布和可视化展示不同特征组合的聚类结果，指出词汇变化特征最具区分度，且缺乏完整的定量评估。

**⚠️ 局限性**

局限性包括样本量有限（仅286条）、标注主观性高、俄语专用，难以推广；缺乏自动化标注工具，且未与非MWE进行对照验证，导致结论缺乏外部验证。

---

## 446. Learning-Accelerated Optimization-based Trajectory Planning for Cooperative Aerial-Ground Handover Missions

**arXiv ID:** 2605.19562 | [PDF](https://arxiv.org/pdf/2605.19562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 447. Bézier Degradation Modeling for LiDAR-based Human Motion Capture

**arXiv ID:** 2605.19620 | [PDF](https://arxiv.org/pdf/2605.19620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 448. deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection

**arXiv ID:** 2605.19605 | [PDF](https://arxiv.org/pdf/2605.19605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 449. Formal Skill: Programmable Runtime Skills for Efficient and Accurate LLM Agents

**arXiv ID:** 2605.19604 | [PDF](https://arxiv.org/pdf/2605.19604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 450. Optimal Reconstruction from Linear Queries

**arXiv ID:** 2605.19625 | [PDF](https://arxiv.org/pdf/2605.19625v1)

**作者:** Yuval Filmus `[一作]` (Technion Israel Institute of Technology), Elizaveta Nesterova `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在受噪声约束下，如何通过适应性线性查询精确重构未知点，并给出了最优误差与查询次数、维度及噪声的精确关系；

**💡 创新点**

首次提出鲁棒Jung定理并用其证明了误差收敛速率为双指数、以及在维度增长时查询次数阈值为指数阶的结果；

**🔧 技术方法**

主要技术包括几何分析（Chebyshev半径与Jung常数）、覆盖数估计、稳健Jung定理、对称性与李群动作、以及对可行域的动态更新；

**📊 数据集**

无实验数据集，全部为理论证明与数学分析；

**📈 对比分析**

本工作未与经验方法比较，主要通过证明误差下界与上界匹配，展示了理论最优性；

**⚠️ 局限性**

局限在于仅针对适应性查询，未讨论非适应性或受损答复的情况，且对实际实现的空间复杂度与算法细节未给出。

---

## 451. CAD-Free Learning of Spacecraft Pose Estimators via NeRF-Based Augmentations

**arXiv ID:** 2605.19649 | [PDF](https://arxiv.org/pdf/2605.19649v1)

**作者:** Antoine Legrand `[一作]` (UCLouvain), Christophe De Vleeschouwer `[通讯]` (UCLouvain)

**通讯引用:** 6935 | [OpenAlex ID](https://openalex.org/A5012049713)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用 NeRF 在少量真实或类真实图像上学习三维空间姿态估计网络，无需 CAD 模型即可训练目标特定网络。

**💡 创新点**

创新点在于：①用两个 NeRF 分别学习外观和几何并生成高质量分割掩模；②在保持几何不变的前提下对外观进行随机化（光照 embedding 采样与颜色 MLP 噪声），从而在合成数据中同时实现视角和外观多样性；③将该合成数据与原始数据混合训练姿态网络，显著提升 OOD 泛化。

**🔧 技术方法**

核心技术包括：Neural Radiance Field (K‑Planes) 与可学习姿态校正；光照 embedding 随机采样（均匀、插值、外推、Gaussian）；颜色 MLP 参数噪声；姿态估计网络 SPNv2（EfficientNet+BiFPN+多头回归/PnP/分割）；数据增强与背景随机化。

**📊 数据集**

实验使用 SPEED+（Synthetic、Lightbox、Sunlamp）和 SHIRT（Lightbox_ROE2）四个数据集；训练集从 25~400 张光照与视角有限的图像；测试集覆盖 Lightbox、Sunlamp 两种光照，亦对 Synthetic 进行 OOD 评估。

**📈 对比分析**

与传统仅 CAD 合成、仅视角增广、仅光照增广、普通域随机化等方法比较，实验表明：在 Lightbox 上 Pose Score 从 0.42 降至 0.109（降低 74%），在 Sunlamp 上从 0.29 降至 0.191（降低 34%）；在 Synthetic 训练集上使用 NeRF 外观随机化将 OOD Score 进一步提升 19–38%。

**⚠️ 局限性**

局限性包括：NeRF 训练耗时、需要可用的姿态标注和相机标定；合成图像分辨率受限；对极端光照/高对比度场景仍可能需要更丰富的原始数据；目前实验以 2D 视图为主，尚未验证多摄像头或多模态场景的可扩展性。

---

## 452. optimize_anything: A Universal API for Optimizing any Text Parameter

**arXiv ID:** 2605.19633 | [PDF](https://arxiv.org/pdf/2605.19633v1)

**作者:** Lakshya A Agrawal `[一作]` (UC Berkeley), Matei Zaharia `[通讯]` (UC Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的优化系统，该系统能够在六个不同领域中实现最先进的结果，涵盖单任务搜索、多任务搜索和对未见输入的泛化。

**💡 创新点**

首次展示了基于文本优化的LLM搜索作为一种通用问题解决范式，统一了传统上需要特定领域算法的任务。

**🔧 技术方法**

使用了基于LLM的文本优化系统，支持单任务、多任务和泛化模式，并通过统一的API进行操作。

**📊 数据集**

在六个主要领域进行评估，包括代理架构、云调度、ARC-AGI、AIME提示、CUDA内核生成和圆形打包，附录中还展示了两个额外领域的初步结果。

**📈 对比分析**

与现有的单任务优化系统相比，该系统在多任务搜索中表现更好，能够通过跨任务转移发现的优化模式加速收敛，且在相同的每个问题预算下，性能显著提升。

**⚠️ 局限性**

系统依赖于LLM的能力，评估成本可能较高，且假设优化的工件可以表示为文本，设计有效的侧信息仍需领域专业知识。

---

## 453. Lens Privacy Sealing: A New Benchmark and Method for Physical Privacy-Preserving Action Recognition

**arXiv ID:** 2605.19578 | [PDF](https://arxiv.org/pdf/2605.19578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 454. Spectral Integrated Gradients for Coarse-to-Fine Feature Attribution

**arXiv ID:** 2605.19607 | [PDF](https://arxiv.org/pdf/2605.19607v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 455. Inferring Sensitive Attributes from Knowledge Graph Embeddings: Attack and Defense Strategies

**arXiv ID:** 2605.19644 | [PDF](https://arxiv.org/pdf/2605.19644v1)

**作者:** Yasmine Hayder `[一作]` `[通讯]` (INSA CVL), Yasmine Hayder (INSA CVL)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了知识图谱嵌入（KGE）模型在推理时产生的推荐结果中泄露敏感属性的风险，并提出一种模型无关的后处理防御策略，通过在推荐列表中随机替换或打乱部分项来降低属性推断攻击的成功率。

**💡 创新点**

首次在黑盒推理阶段引入后处理随机化/置换防御，展示了在不改训练阶段即可显著削弱属性推断攻击的能力；同时对推荐质量与隐私泄漏的权衡进行了量化分析。

**🔧 技术方法**

使用 RotatE KGE 模型进行知识图谱嵌入与推荐；构造基于 KGE 的属性推断攻击模型；实现随机替换与可选的列表打乱（shuffling）防御算法；通过攻击成功率 I_u 与推荐质量 Q_u 两个指标评估。

**📊 数据集**

Yahoo Movies（约20万条三元组，约7000用户）和 MovieLens 100K（约10万条三元组，约1000用户）两大基准数据集，均已转换为知识图谱格式。

**📈 对比分析**

在不同随机替换比例（保留 top‑t 推荐项与随机选 r 项）以及是否进行 shuffling 的设置下，比较攻击成功率与推荐质量。实验显示，随机化可将性别属性推断成功率从约0.70 降至约0.50，推荐质量 Q_u 在0.5–0.8之间保持，可接受的隐私‑效用平衡。

**⚠️ 局限性**

仅针对单一敏感属性（性别）与简单随机置换防御；未提供正式差分隐私保证；对更强攻击者（可访问历史交互）未评估；缺乏对多属性、不同 KGE 模型以及更复杂置换策略的系统验证。

---

## 456. Inverse Design of Metasurface based Absorbers using Physics Guided Conditional Diffusion Models

**arXiv ID:** 2605.19611 | [PDF](https://arxiv.org/pdf/2605.19611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 457. The Accessibility Capability Boundary: Operational Limits and Expansion Potential of AI-Generated Browser-Native Accessibility Systems

**arXiv ID:** 2605.19638 | [PDF](https://arxiv.org/pdf/2605.19638v1)

**作者:** Rizwan Jahangir `[一作]` (NUST), Daisuke Ishii `[通讯]` (Kiara Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可访问性能力边界（ACB）框架，并通过两款 AI 生成的浏览器原生辅助原型验证其可访问性扩展潜力。

**💡 创新点**

将可访问性从二元属性转化为多维能力空间，定义 ACB 边界；阐释 AI 生成浏览器原生系统如何在低资源环境下显著扩展可访问性边界，并提出可访问性能力评分与评估矩阵。

**🔧 技术方法**

利用大型语言模型（Claude/ChatGPT）生成 HTML/ARIA 代码，结合浏览器原生 API（Web Speech、MediaDevices、Service Worker、WASM FaceMesh）实现实时摄像头对齐；使用 axe‑core、WAVE、Lighthouse 等自动化可访问性审核工具。

**📊 数据集**

未使用公开数据集，原型使用 MediaPipe FaceMesh 的预训练模型以及尼泊尔盲人用户的实际使用情境。

**📈 对比分析**

通过可访问性能力矩阵对比传统 AT、原生应用和 AI 生成浏览器系统的约束向量；自动化审核显示 100% 静态可访问性评分，手动屏幕阅读器测试发现细节问题；性能基准显示桌面/笔记本/手机初始加载 320‑850 ms、离线加载 45‑120 ms、面部检测 15‑110 ms、语音合成 5‑45 ms，CPU 占用 4‑18% 等。

**⚠️ 局限性**

主要局限包括 LLM 生成代码的幻觉与可访问性回归风险、浏览器沙箱限制深度硬件访问、计算资源受限导致推理受限、缺乏完整人类实验验证、自动化审核无法覆盖所有动态交互、数字鸿沟导致生成阶段对网络的依赖以及隐私与安全风险。

---

## 458. Understanding Wacky Weights: A Dissection of SPLADE's Learned Term Importance

**arXiv ID:** 2605.19628 | [PDF](https://arxiv.org/pdf/2605.19628v1)

**作者:** Gregory Polyakov `[一作]` (University of Tübingen), Carsten Eickhoff `[通讯]` (University of Tübingen)

**通讯引用:** 4129 | [OpenAlex ID](https://openalex.org/A5014921416)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统复现并量化 SPLADE 模型中出现的“wacky weights”（与输入语义不相关的扩展词），构建正式定义与指标（Wackiness Score、Normalized Wackiness Curve、W-AUC），并通过多种模型变体与训练配置实验，探究这些词对检索效果的影响、来源及其可解释性。

**💡 创新点**

①首次给出 wacky token 的客观度量和比较框架；②发现词表大小和稀疏正则化是影响 wacky 产生的主要因素；③证明 wacky tokens 在模型的域内效果中扮演重要角色，而在域外表现不显著；④揭示不同模型架构/训练数据导致 wacky token 语义类别差异。

**🔧 技术方法**

使用 SPLADE-v2、SPLADE-v3 及其多种变体（不同预训练编码器、聚合方式、稀疏损失、训练数据）；构建 Wackiness Score、Normalized Wackiness Curve 与 W-AUC；在 MS MARCO 与 BEIR 评测集上执行检索实验，比较在删除 wacky tokens 前后的 MRR@10、Recall@10/100/1000、NDCG@10 等指标。

**📊 数据集**

MS MARCO (训练集、devset)，BEIR benchmark (10 个子集：ArguAna、Climate-FEVER、DBPedia-Entity、FiQA-2018、NFCorpus、Quora、SCIDOCS、SciFact、TREC-COVID、Touché-2020)，TREC DL 19/20。

**📈 对比分析**

通过对比删除前后 wacky tokens 与随机删除的检索性能差异，使用统计置信区间评估显著性；使用 W-AUC 对不同模型的 wacky token 预valence 进行量化比较。结果显示 SPLADE-v3 对 wacky tokens 依赖更强，在域内删除会显著降低 MRR@10，而 SPLADE-v2 则影响较小；在域外 BEIR 数据上删除 wacky tokens 的效果与随机删除相当。

**⚠️ 局限性**

仅针对 SPLADE 系列模型，未涉及其他 LSR 或密集检索模型；缺乏对 wacky tokens 语义意义的深入解释；实验主要聚焦于词表大小与稀疏损失，对训练数据对 wacky token 影响的系统性研究仍待展开；评测集规模有限，可能无法完全覆盖所有潜在场景。

---

## 459. MiMuon: Mixed Muon Optimizer with Improved Generalization for Large Models

**arXiv ID:** 2605.19619 | [PDF](https://arxiv.org/pdf/2605.19619v1)

**作者:** Feihu Huang `[一作]` (Nanjing University of Aeronautics and Astronautics), Songcan Chen `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 13838 | [OpenAlex ID](https://openalex.org/A5101596072)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究 Muon 优化器的泛化误差并提出混合 Muon (MiMuon) 优化器，证明其泛化误差下降且保持与 Muon 相同的收敛率。

**💡 创新点**

通过算法稳定性与数学归纳证明 Muon 泛化误差为 O(1/(Nκ^T))，并提出谨慎正交化混合策略，使 MiMuon 泛化误差降至 O(1/N) 与 SGD/SGDM 等传统优化器持平。

**🔧 技术方法**

算法稳定性分析、数学归纳、SVD 正交化、Newton–Schulz 迭代、矩阵结构优化、混合梯度映射等技术。

**📊 数据集**

在语言模型 Qwen3‑0.6B（WikiText‑103）和目标检测 YOLO26m（Pascal VOC）上进行实验。

**📈 对比分析**

与 AdamW、Lion、Shampoo、Muon、MuSGD 等进行对比，MiMuon 在 Qwen3‑0.6B 的训练/验证损失最低，在 YOLO26m 的 mAP 最高，整体提升约 0.4‑0.5 的训练损失和 3‑5 点的 mAP。

**⚠️ 局限性**

受限于 κ 极小导致 Muon 泛化误差高，MiMuon 的正交化阈值 τ 需经验调参，理论假设可能不完全适用于所有非凸任务；实验仅覆盖两大模型，缺乏更广泛的多任务验证。

---

## 460. Implicit Action Chunking for Smooth Continuous Control

**arXiv ID:** 2605.19592 | [PDF](https://arxiv.org/pdf/2605.19592v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 461. Physics-Informed Graph Neural Network Surrogates for Turbulent Nanoparticle Dispersion in Dental Clinical Environments

**arXiv ID:** 2605.19589 | [PDF](https://arxiv.org/pdf/2605.19589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 462. When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach

**arXiv ID:** 2605.19662 | [PDF](https://arxiv.org/pdf/2605.19662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 463. Benchmarking and Evolving Reason-Reflect-Rectify for Reflective Visual Generation

**arXiv ID:** 2605.19639 | [PDF](https://arxiv.org/pdf/2605.19639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 464. PrAda: Few-Shot Visual Adaptation for Text-Prompted Segmentation

**arXiv ID:** 2605.19623 | [PDF](https://arxiv.org/pdf/2605.19623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 465. Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries

**arXiv ID:** 2605.19576 | [PDF](https://arxiv.org/pdf/2605.19576v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对自演化技能库的“图书馆漂移”问题进行诊断与修复，提出可复现触发器、追踪级诊断和治理策略。

**💡 创新点**

引入“图书馆漂移”概念、可复现触发实验、逐技能贡献评分与归因判断的追踪诊断，以及基于结果驱动退休、容量上限与元技能先验的治理食谱。

**🔧 技术方法**

使用大型语言模型（Claude Opus 4.7）、文本检索、嵌入向量、LLM评判器、循环式自演化框架（Ratchet）以及贡献评分与归因分析。

**📊 数据集**

采用 MBPP+ hard‑100 代码生成任务集。

**📈 对比分析**

与无技能基线及多种消融配置对比，默认治理下的 pass@1 从 0.258 提升至 0.584，提升 0.328；消融展示哪些机制是关键。

**⚠️ 局限性**

仅在单一 benchmark 与单一模型上验证，缺乏跨领域、多步代理与模型的泛化，诊断阈值经验设定，未证明在更大规模任务集上的有效性。

---

## 466. m3BERT: A Modern, Multi-lingual, Matryoshka Bidirectional Encoder

**arXiv ID:** 2605.19568 | [PDF](https://arxiv.org/pdf/2605.19568v1)

**作者:** Yaoxiang Wang `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 4080 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一种多层多维度可裁剪的BERT模型m^3BERT，能够在任意层级和嵌入维度下直接使用嵌入，满足工业检索在不同资源与性能需求下的灵活部署。

**💡 创新点**

将Matryoshka Representation Learning 与多层、多维度联合预训练相结合，首次构造出在任意层与维度截断后仍能保持高质量的嵌入；采用三阶段预训练（monolingual→multilingual→web domain）与现代改造（SwiGLU、RMSNorm、FlashAttention、无偏差、无Dropout）以及自蒸馏提升低维性能。

**🔧 技术方法**

Matryoshka多粒度预训练、SwiGLU激活、RMSNorm预规范化、FlashAttention、Infinite Contrastive Learning (Inf-CL)、自蒸馏、AdamW、混合精度训练、三阶段预训练、监督对比学习 Fine‑Tuning 等技术。

**📊 数据集**

工业数据集BINGCLICK（约100M Q‑Doc对、10M候选文档）以及公开数据集MS MARCO、Natural Questions、TREC‑COVID；预训练阶段使用Nemotron‑CC、100种语言Wikipedia、10B Q‑Doc对。

**📈 对比分析**

与mBERT、mE5、ModernBERT等基线在全/1/3层、32/64/128/768维度下使用Recall@100/Recall@1000评估；m^3BERT在所有配置下均优于基线，尤其在低维度下提升显著；在Bing搜索部署后年收入约为5000万美元。

**⚠️ 局限性**

对极低维度或极低资源环境的适配仍有限；多层多维度预训练的模型规模大、训练成本高；在高维度（>128）下的效果需进一步验证，微调时可能出现轻微性能下降。

---

## 467. FlyMirage: A Fully Automated Generation Pipeline for Diverse and Scalable UAV Flight Data via Generative World Model

**arXiv ID:** 2605.19600 | [PDF](https://arxiv.org/pdf/2605.19600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 468. A Family of Divergence Measures for Evaluating the Reconstruction Quality of Explainable Ensemble Trees

**arXiv ID:** 2605.19618 | [PDF](https://arxiv.org/pdf/2605.19618v1)

**作者:** Massimo Aria `[一作]` (University of Naples Federico II), Carmela Iorio `[通讯]` (University of Naples Federico II)

**通讯引用:** 499 | [OpenAlex ID](https://openalex.org/A5037270515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套新的度量方法，用于评估解释型集成树（E2Tree）对随机森林等集成模型的相似度重构质量。

**💡 创新点**

创新点在于引入归一化解释损失（nLoI）并证明其可分解为内部节点与外部节点误差两部分，提供诊断能力；同时提出四种互补的相似/差异度量（Hellinger距离、加权RMSE、RV系数、SSIM）并统一了置换检验框架。

**🔧 技术方法**

技术包括基于Cressie–Read功率散度族（λ=-2）的统计量、矩阵分解与归一化、加权平方误差、协方差相似度、结构相似性指数以及单次行列置换的置换检验。

**📊 数据集**

使用了Iris（分类）、mtcars（回归）和Boston Housing（回归）三大公共数据集进行实证评估。

**📈 对比分析**

通过Monte Carlo模拟验证了所有度量在α=0.05下的Type I误差控制良好，功效曲线表明在信号强度≥0.4时功效≥0.95，且对稀疏度鲁棒；实证结果显示所有度量均显著表明E2Tree重构有效，nLoI与Hellinger距离在不同数据集上保持在[0,1]区间内。

**⚠️ 局限性**

局限在于目前仅针对E2Tree展开，未对梯度提升或其他解释性代理进行验证；理论上对样本量、树数量趋于无穷大时的极限行为尚未完全解析；对极端稀疏或极大维度的适用性需进一步研究。

---

## 469. MCNav: Memory-Aware Dynamic Cognitive Map for Zero-shot Goal-oriented Navigation

**arXiv ID:** 2605.19594 | [PDF](https://arxiv.org/pdf/2605.19594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 470. OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond

**arXiv ID:** 2605.19660 | [PDF](https://arxiv.org/pdf/2605.19660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 471. Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption

**arXiv ID:** 2605.19593 | [PDF](https://arxiv.org/pdf/2605.19593v1)

**作者:** Mert Yildiz `[一作]` (Sapienza University of Rome), Andrea Baiocchi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 2278 | [OpenAlex ID](https://openalex.org/A5074428250)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在共享异构 GPU 资源的多模型 LLM 服务器环境中，系统通过系统化实验研究了层级 CPU‑GPU offloading 与完整作业 preemption 对推理吞吐量和延迟的影响，并基于此提炼出调度器需要关注的关键特性。

**💡 创新点**

创新点在于对多模型环境下的 offloading 与 preemption 进行细粒度实测，发现吞吐量随 GPU 层比例非线性变化且与模型和 GPU 相关，同时证明 preemption 成本几乎与停顿点无关，主要由模型重载决定。

**🔧 技术方法**

采用 Ollama 量化模型 (Q4)、HuggingFace Transformers、PCIe 传输、CPU‑GPU 资源划分、完整加载/卸载、KV 缓存迁移等技术，结合 CUDA、Python GC、PyTorch 缓存分配器等实现。

**📊 数据集**

使用的模型和数据集为 Llama 3‑8B、Qwen3‑32B、Llama 2‑70B、Qwen2.5‑3B、Qwen3‑8B、Qwen2.5‑14B；实验在两台服务器上运行，GPU 分别为 RTX 5000 和 RTX A6000；输出长度取 50、150、300、500、1000、5000 词等。

**📈 对比分析**

与全 GPU 部署基准对比，offloading 在较小模型上导致 30‑80% 的吞吐量下降，而大模型下降更平缓；preemption 开销在 RTX 5000 上约 3–7 秒，占完成时间 1.7‑2.4%，在 RTX A6000 上 2.6–5.7 秒，KV 迁移占比 <1.5%。

**⚠️ 局限性**

局限在于实验仅使用单请求推理，且每个任务仅一次 preemption；未覆盖批量请求、多次 preemption、连续批处理等实际工作负载；对不同硬件（PCIe 速度、存储 I/O）以及更大规模模型的评估仍待进一步验证。

---

## 472. Soft Covering Through the Lens of Hypothesis Testing

**arXiv ID:** 2605.19573 | [PDF](https://arxiv.org/pdf/2605.19573v1)

**作者:** Neri Merhav `[一作]` `[通讯]` (Technion Israel Institute Of Technology), Neri Merhav (Technion Israel Institute Of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过将软覆盖定理重新表述为Neyman–Pearson假设检验，推导了任意码本速率R≥0和阈值τ∈ℝ下误报与漏检错误指数的精确单字母表达式，并揭示了其相应的相位结构。

**💡 创新点**

创新点在于首次给出了涵盖所有速率的软覆盖误差指数的闭式解，发现了在R=I(X;Y)和τ=0处的双指数消失点，并揭示了误报指数平坦段、漏检指数尖峰等多种相位转变，说明了软覆盖现象与相变理论的深层联系。

**🔧 技术方法**

所用技术主要是类型枚举方法与大偏差理论，利用类型计数、二项分布极大偏差定理和数据处理不等式，避免了传统Chernoff界的不严谨性，得到严格的指数收敛速率。

**📊 数据集**

本文为理论分析，不依赖具体数据集；实验演示以Z通道为例，用数值绘图展示指数随阈值和速率变化的曲线。

**📈 对比分析**

与已有的软覆盖距离指数（KL、Rényi、总变距离等）相比，本文的指数在Annealed（期望）意义下提供了最优的Neyman–Pearson阈值曲线，展示了在不同R和τ下误报和漏检指数的严格上界与下界一致，从而实现了指数精确匹配。

**⚠️ 局限性**

主要局限在于仅给出了Annealed平均指数，未考虑典型码本（quenched）指数；对匹配和复合信道的情况未展开，且目前结果仅适用于离散无记忆信道，扩展到连续、带记忆或多端口系统仍需进一步研究。

---

## 473. Pseudocode-Guided Structured Reasoning for Automating Reliable Inference in Vision-Language Models

**arXiv ID:** 2605.19663 | [PDF](https://arxiv.org/pdf/2605.19663v1)

**作者:** Weicong Ni `[一作]` (East China Normal University), Linlin Wang `[通讯]` (East China Normal University)

**通讯引用:** 75057 | [OpenAlex ID](https://openalex.org/A5100425554)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PStar框架，通过可解释的伪代码路径提升视觉‑语言模型的推理可靠性，显著降低幻觉率；

**💡 创新点**

创新点在于构建抽象函数库与困难特征向量(DFV)，并用A*搜索与混合相似度自适应选取推理路径；

**🔧 技术方法**

采用DFV特征提取、A*搜索、混合相似度评分、伪代码引导推理、无训练的推理路径库；

**📊 数据集**

使用MathVista、MATH‑Vision、ScienceQA混合数据集做路径生成，并在POPE、HallusionBench、MMStar、OKVQA等基准上评测；

**📈 对比分析**

与GPT‑4V及多款开源VLM（Qwen、Llama、LLaVA等）对比，PStar在POPE 87.1%、MMStar 68.0%、HallusionBench 68.8%等指标均优于基线，显示显著性能提升；

**⚠️ 局限性**

局限在于A*搜索和路径生成的离线计算成本较高，尚未在长周期规划或极大规模实时任务中验证，且依赖手工构造的抽象函数与DFV，可能在某些领域适配性不足。

---

## 474. Hardness and Approximation for Coloring Digraphs

**arXiv ID:** 2605.19654 | [PDF](https://arxiv.org/pdf/2605.19654v1)

**作者:** Parinya Chalermsook `[一作]` (University of Sheffield), Chaoliang Tang `[通讯]` (Fudan University)

**通讯引用:** 1528 | [OpenAlex ID](https://openalex.org/A5007866518)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了有向图二分数（dichromatic number）的近似算法与逼近难度，并给出多种特殊情形下的多项式时间算法和逼近下界。

**💡 创新点**

创新点包括：①引入随机化的词典化lexicographic product得到与无向图色数和独立数对应的逼近难度；②提出对 2‑可着色图的 O(√n) 色数上界；③基于独立数限制和无三环条件改进稠密图的上界。

**🔧 技术方法**

使用的技术包括：随机化图积、路径分解与回退边图、递归分区与颜色分配、稀疏/稠密图独立数估计以及多项式时间贪心/分区方法。

**📊 数据集**

论文未使用实验数据集，全部为理论证明与算法设计。

**📈 对比分析**

通过证明 n^{1-ε} 的逼近难度以及多项式时间 O(n^{2ℓ}) 的色数上界；与以往 10 颜色 2‑可着色结果、35^{α-1}α! 的稠密图上界相比，分别取得更紧的 10/3(4^α-1) 与 (α+8)!/9! 颜色上界。

**⚠️ 局限性**

局限性在于只给出了上界，缺乏相应下界；对 2‑可着色图的更优色数仍为开放问题；对一般有向图的逼近复杂度仍有空缺，且算法对稠密图的依赖较强。

---

## 475. Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation

**arXiv ID:** 2605.19651 | [PDF](https://arxiv.org/pdf/2605.19651v1)

**作者:** Yuanzi Li `[一作]` (Renmin University of China), Xu Chen `[通讯]` (Renmin University of China)

**通讯引用:** 23210 | [OpenAlex ID](https://openalex.org/A5100385692)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于教师‑同行‑自我（Teacher‑Peer‑Self）协作的多源负采样框架 MDCNS，用于提升顺序推荐模型在隐式反馈上的学习效果。

**💡 创新点**

创新点在于：①引入多源评分打破自我强化循环；②利用自我与同行模型预测差异进行分歧重排，提升采样多样性；③通过教师模型的 KL 散度蒸馏实现一致性学习，充分利用计算资源并避免资源浪费。

**🔧 技术方法**

主要技术包括：多源评分（自我、同行、教师）；分歧重排（对预测差异加权）；一致性蒸馏（温度软化+KL 约束）；基于 BPR/BCE 的对比学习；以及对候选池的随机采样与 Top‑M 选取。

**📊 数据集**

使用了六个真实数据集：Amazon Sports、Beauty、Toys、Health；KuaiRand（短视频平台日志）；LastFM（音乐标签）。

**📈 对比分析**

与 RNS、DNS、MixGCF、AdaSIR、GNNO、DNS+、MixGCF+、SRNS 等主流负采样方法进行对比，在 Recall@K、NDCG@K 上均显著提升，最优数据集（Beauty）Recall@20 提升 27.29%，NDCG@20 提升 36.44%。

**⚠️ 局限性**

主要局限在于：训练阶段需维护教师/同行模型，导致额外计算开销；对教师/同行模型的质量依赖较大，弱模型时提升有限；以及在极大规模数据上多源评分与蒸馏的实现成本仍有待优化。

---

## 476. K-Quantization and its Impact on Output Performance

**arXiv ID:** 2605.19645 | [PDF](https://arxiv.org/pdf/2605.19645v1)

**作者:** Robin Baki Davidsson `[一作]` (Lund University), Pierre Nugues `[通讯]` (Lund University)

**通讯引用:** 3013 | [OpenAlex ID](https://openalex.org/A5069585870)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对八种大型语言模型在不同量化精度（2‑至6‑bit）下的性能进行系统评估，探讨量化对知识推理、代码理解和长文本理解的影响。

**💡 创新点**

提出了针对不同模型体系结构（Llama 3、Gemma、Phi‑3、Mistral）和任务类型的量化耐受性分析，展示了在 2‑bit 量化下模型表现差异显著的结论，并给出在 Q3_K‑Q6_K 之间取得性能与压缩平衡的经验。

**🔧 技术方法**

采用后训练量化（PTQ）中的 k‑quant 方法，并使用 llama.cpp + GGUF 格式进行部署和评估；对权重采用 INT4/INT8 量化，保留高精度超块常数；实现 8‑bit 量化作为基准。

**📊 数据集**

使用 MMLU‑Pro（知识推理）、CRUXEval（代码推理）和 MuSR（长文本推理）三大公开数据集；同时评估 Wikitext‑2 的困惑度来衡量语言建模能力。

**📈 对比分析**

通过比较不同量化级别下的准确率和困惑度，计算每种模型的“性能效率”（准确率/模型大小）。结果显示：高精度（Q8_0）性能最佳，低精度（Q2_K）性能显著下降，尤其对 Phi‑3 极端敏感；中等精度（Q3_K‑Q6_K）在大模型上仍能保持 80%+ 的准确率，且压缩率高，整体性能表现相对稳健。

**⚠️ 局限性**

评估指标（准确率、困惑度）与人类判断不完全一致，可能无法捕捉模型的幻觉与偏见；量化效果在不同模型架构和任务间差异较大，缺乏统一的最优策略；实验受限于硬件与时间，仅覆盖有限数量的模型与量化级别。

---

## 477. How Helpful is LLM Assistance in Network Operations? A Case Study at a Large Demonstration Network

**arXiv ID:** 2605.19627 | [PDF](https://arxiv.org/pdf/2605.19627v1)

**作者:** Ryo Nakamura `[一作]` (University of Tokyo), Koshi Eguchi `[通讯]` (University of Tokyo)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5108320347)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一座大规模演示网络ShowNet上，让105名网络工程师使用并评估基于LLM的聊天机器人，记录其在网络构建与运维中的帮助效果。

**💡 创新点**

首次在真实演示网络中量化LLM辅助运维的效果，并分析用户对机器人能力的认知与使用差异。

**🔧 技术方法**

使用GPT‑4.1、检索增强生成（RAG）、Model Context Protocol（MCP）实现CLI控制和工单访问，构建三功能聊天机器人。

**📊 数据集**

对ShowNet的设计文档、操作手册以及上一年网络设备配置文件构建向量数据库，并记录两周内的815条会话。

**📈 对比分析**

通过参与者给出的正负评估计算正向率，得到68.1%正评，CLI命令成功率85.1%，与SWE‑bench等LLM基准相近。

**⚠️ 局限性**

LLM仍会产生误报、命令错误，无法完全理解超出知识范围的查询；缺乏进一步推理与多步操作能力，且仅在有限的演示网络环境验证。

---

## 478. EMO-BOOST: Emotion-Augmented Audio-Visual Features for Improved Generalization in Deepfake Detection

**arXiv ID:** 2605.19630 | [PDF](https://arxiv.org/pdf/2605.19630v1)

**作者:** Aritra Marik `[一作]` (Technical University of Darmstadt), Anna Rohrbach `[通讯]` (Technical University of Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Emo-Boost 框架，将情感检测与低级特征融合，以提升深度伪造检测的跨操纵泛化能力。

**💡 创新点**

创新点在于：①引入跨模态情感时间一致性与对比学习的双维度情感表征；②利用简单的乘法融合将情感信息与现有 RGB‑声学检测器相结合，显著提升未知操纵类型的检测性能。

**🔧 技术方法**

使用技术包括：预训练的视觉情感编码器（POSTER）和音频情感编码器（emotion2vec）；时间 Transformer 对情感序列建模；二元交叉熵与对比损失的联合训练；以及与现有多模态检测器（SIMBA）进行简单的特征融合。

**📊 数据集**

实验数据集为 FakeAVCeleb（21k 影片）与 DeepSpeak v2（16.5k 影片），覆盖多种视觉与音频伪造技术。

**📈 对比分析**

在跨操纵 Leave‑one‑out 评估中，Emo‑Boost 在 FakeAVCeleb 上平均 AUC 提升 2.1%，在 DeepSpeak v2 上保持竞争性；在域内评估中也能与或略高于最先进方法保持相同水平。

**⚠️ 局限性**

局限性包括：单独使用时 EmoForensics 的性能相对较弱；对 DeepSpeak v2 的提升有限，原因是脚本化录制环境中情感信号较弱；并且模型可能继承训练数据中的偏差。

---

## 479. Component-Aware Structure-Preserving Style Transfer for Satellite Sim2Real 6D Pose Estimation

**arXiv ID:** 2605.19624 | [PDF](https://arxiv.org/pdf/2605.19624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 480. UniRefiner: Teaching Pre-trained ViTs to Self-Dispose Dross via Contrastive Register

**arXiv ID:** 2605.19622 | [PDF](https://arxiv.org/pdf/2605.19622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 481. SceneCode: Executable World Programs for Editable Indoor Scenes with Articulated Objects

**arXiv ID:** 2605.19587 | [PDF](https://arxiv.org/pdf/2605.19587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 482. White-Balance First, Adjust Later: Cross-Camera Color Constancy via Vision-Language Evaluation

**arXiv ID:** 2605.19613 | [PDF](https://arxiv.org/pdf/2605.19613v1)

**作者:** Shuwei Li `[一作]` (National University Of Singapore), Robby T. Tan `[通讯]` (National University Of Singapore)

**通讯引用:** 8899 | [OpenAlex ID](https://openalex.org/A5103147507)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于视觉语言模型的迭代反馈框架 VLM‑CC，用于跨相机色彩恒常性。

**💡 创新点**

创新点在于把白平衡后的伪 sRGB 图像交给 LoRA 微调的 VLM 做语义反馈，只预测红绿蓝残余光照方向，从而实现不直接回归、逐步迭代校正，显著提升跨相机鲁棒性。

**🔧 技术方法**

使用的技术包括视觉语言模型（如 Qwen2.5‑VL 7B）、LoRA 微调、伪 sRGB 映射以及基于颜色方向的迭代更新。

**📊 数据集**

实验使用了四大公开 RAW 数据集：Gehler‑Shi、NUS‑8、Cube+、Intel‑TAU，进行交叉相机评估。

**📈 对比分析**

与多种统计、学习型方法以及最新跨相机方法在平均角误差、三分位、中位数、最佳/最差 25% 等指标上进行比较，VLM‑CC 在所有指标上均居前，尤其是最差 25% 误差显著下降。

**⚠️ 局限性**

局限性包括对 VLM 语义识别的依赖，极端光照或缺乏可识别语义物体的场景性能可能受限；同时需要相机的色彩校正矩阵才能完成伪 sRGB 转换。

---

## 483. LLMEval-Logic: A Solver-Verified Chinese Benchmark for Logical Reasoning of LLMs with Adversarial Hardening

**arXiv ID:** 2605.19597 | [PDF](https://arxiv.org/pdf/2605.19597v1)

**作者:** Ming Zhang `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 17164 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了 LLMEval-Logic 中文逻辑推理基准，其中包含可验证的 Base 子集和通过对抗式硬化生成的 Hard 子集；同时提供了专家评审、Z3 形式化验证、以及细粒度 rubric 评估。

**💡 创新点**

创新点在于：① 采用前向作者（forward‑author）方式生成真实场景的自然语言问题，避免模板化短板；② 结合 Z3 验证与专家制定的 rubric，细粒度审计自然语义到形式语义的映射；③ 引入闭环对抗式硬化流程（Decider→Proposal→Review→Answering→Verification）使单问准确率上升后仍能保持挑战性。

**🔧 技术方法**

主要技术包括：自然语言到形式逻辑的前向作者、四层归一化处理、Z3 SMT 形式化验证、rubric 级别的逻辑关系/约束/查询对齐评估，以及多轮对抗式硬化算法。

**📊 数据集**

使用了自建的 LLMEval‑Logic 数据集，包含 521 条原始中文推理条目，最终发布 246 条 Base（单问）和 190 条 Hard（多问，总计 938 子问）子集；每条条目均配有 Z3 验证的形式化及相应 rubric。

**📈 对比分析**

与 14 个前沿 LLM（包括 Gemini、Claude、GPT‑5.4、Qwen、Kimi、Hy3、Seed 等）在三轮独立实验中进行对比。最佳 Hard 项目准确率仅 37.5%，思考型模型在 Hard 上表现更好；相较于 Base，Hard 能显著拉开模型间差距并揭示模型在闭合空间维护、连锁子问处理上的不足。

**⚠️ 局限性**

局限性：① 仅覆盖中文自然语言，跨语言推广尚未验证；② 只涉及命题与一阶逻辑，未覆盖高阶、模态、时序或概率逻辑；③ 虽有 Z3 验证和 rubric 审核，但仍可能存在细粒度语义偏差或专家标注盲点。

---

## 484. A novel YOLO26-MoE optimized by an LLM agent for insulator fault detection considering UAV images

**arXiv ID:** 2605.19595 | [PDF](https://arxiv.org/pdf/2605.19595v1)

**作者:** João Pedro Matos-Carvalho `[一作]` (Universidade de Lisboa), Gabriel Villarrubia González `[通讯]` (University of Salamanca)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究在无人机获取的绝缘子图像上实现了一种基于YOLO26的单阶段目标检测器YOLO26-MoE，用于自动检测绝缘子缺陷。

**💡 创新点**

创新点包括：在YOLO26的高分辨率检测分支插入稀疏Mixture-of-Experts模块以实现条件特征提炼，并通过大型语言模型代理进行超参数搜索，显著提升了对细小多样缺陷的检测能力。

**🔧 技术方法**

技术手段涵盖：YOLO26骨干网络、稀疏MoE模块、Optuna超参优化、LLM驱动的实验调度、PyTorch框架以及YOLO的推理流程。

**📊 数据集**

使用了无人机现场采集的绝缘子图像数据集，数据标注包含三类缺陷（闪络、破损、完整）并已公开发布。

**📈 对比分析**

通过与YOLOv10、YOLOv11、YOLOv12、YOLO26等多版本进行基准对比，YOLO26-MoE在mAP@0.5达到0.9900，mAP@0.5:0.95达到0.9515，精确度、召回率与F1均优于所有基线模型。

**⚠️ 局限性**

主要限制在于模型相较轻量级YOLO26变体增加了计算复杂度，解释性受限，并且超参搜索依赖于预设搜索空间与训练预算。

---

## 485. PAPO-VLA: Planning-Aware Policy Optimization for Vision-Language-Action Models

**arXiv ID:** 2605.19580 | [PDF](https://arxiv.org/pdf/2605.19580v1)

**作者:** Peizheng Guo `[一作]` (Institute of Software Chinese Academy of Sciences), Wenwen Qiang `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对 Vision‑Language‑Action（VLA）模型的规划感知优化方法 PAPO‑VLA，识别规划动作并通过因果充分性与必要性评估其重要性，再将该重要性融入 GRPO 的优势估计，从而提升机器人语言指导任务的可靠性。

**💡 创新点**

创新点在于：①将 VLA 策略拆分为规划者与执行者两角色，专门识别规划动作；②利用因果充分性与必要性两种度量对规划动作重要性进行评估；③将规划动作的重要性作为加权项直接加入 GRPO 的优势，既保留轨迹级优化，又突出关键动作。

**🔧 技术方法**

技术手段包括：动作变化与轨迹结果的联合筛选来定位规划动作；基于因果充分性（P_suff）与因果必要性（P_nec）的重要性计算；规划动作重要性合并为 C_plan；将 C_plan 加权到 GRPO 的优势估计，实现规划感知的策略优化；采用预训练 VLA 模型（如 OpenVLA）并在其上进行 fine‑tune。

**📊 数据集**

数据集：LIBERO benchmark（包含四个子集 LIBERO‑Spatial、LIBERO‑Object、LIBERO‑Goal、LIBERO‑Long）和 RoboTwin2.0 benchmark（50 任务，包含多种场景随机化）。

**📈 对比分析**

对比方法：PackNet、MTL、ATM、Octo、OpenVLA、OpenVLA‑OFT、TraceVLA、GRAPE、SFT‑4LIBERO、MetaVLA、TGRPO 等；实验结果表明 PAPO‑VLA 在 LIBERO（平均 0.96）和 RoboTwin2.0（短期任务 62.7%/中期任务 59.3%/长期任务 50.4%）均显著优于基线，尤其在长时间/多阶段任务上提升最为显著。

**⚠️ 局限性**

局限性：①规划动作的识别依赖阈值与轨迹结果，可能在极端噪声或不确定环境下失效；②因果评估需要多次轨迹采样，计算成本较高；③实验仅在仿真环境进行，尚未验证在真实机器人硬件上的鲁棒性；④对极长序列或多目标任务的扩展仍需进一步研究。

---

## 486. Real-World On-Vehicle Evaluation of Embedding-Based Anomaly Detection

**arXiv ID:** 2605.19744 | [PDF](https://arxiv.org/pdf/2605.19744v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 487. GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment

**arXiv ID:** 2605.19577 | [PDF](https://arxiv.org/pdf/2605.19577v1)

**作者:** Minxuan Lv `[一作]` (Kuaishou Technology), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于能力导向的长上下文强化学习框架，并在此框架下对LLM进行后训练。

**💡 创新点**

创新点在于：①提出了覆盖9类核心长上下文能力的RLVR数据集；②设计了TMN‑Reweight算法，先做任务级均值归一化再进行难度自适应加权，以解决多任务奖励尺度不一致和难度偏差；③在训练时将自然奖励函数直接用作回报，避免单一评价指标的局限。

**🔧 技术方法**

采用GRPO为基础的强化学习，改进为TMN‑Reweight；同时使用多阶段数据构建与质量控制的四步流水线；在评估中使用统一的长上下文基准和通用推理、记忆任务。

**📊 数据集**

使用了23,000个样本的RLVR数据集，涵盖9个任务类型（EM、Accuracy、F1、NDCG、IoU等），来源为公开长上下文语料与基于真实文档生成的合成样本。

**📈 对比分析**

与QwenLong‑L1.5等基线对比，TMN‑Reweight在Qwen3‑4B‑Thinking上平均提升约8点（从53.0到62.2），在30B上提升约2点；在CorpusQA、LBV2等多任务指标上表现更为均衡，整体长上下文基准平均分从62.2提升到63.0。

**⚠️ 局限性**

局限在于：①难度自适应权重对大规模模型的收益尚不确定；②CorpusQA等多文档推理仍存在性能缺口；③缺乏对不同规模下TMN‑Reweight动态调节的深入研究。

---

## 488. TERGAD: Structure-Aware Text-Enhanced Representations for Graph Anomaly Detection

**arXiv ID:** 2605.19738 | [PDF](https://arxiv.org/pdf/2605.19738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 489. Physics-in-the-Loop: A Hybrid Agentic Architecture for Validated CAD Engineering Design

**arXiv ID:** 2605.19717 | [PDF](https://arxiv.org/pdf/2605.19717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 490. KIO-planner: Attention-Guided Single-Stage Motion Planning with Dual Mapping for UAV Navigation

**arXiv ID:** 2605.19703 | [PDF](https://arxiv.org/pdf/2605.19703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 491. Measuring Safety Alignment Effects in Autonomous Security Agents

**arXiv ID:** 2605.19722 | [PDF](https://arxiv.org/pdf/2605.19722v1)

**作者:** Isaac David `[一作]` (University College London), Arthur Gervais `[通讯]` (University College London)

**通讯引用:** 7346 | [OpenAlex ID](https://openalex.org/A5063253761)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个授权安全代理基准，比较了对齐与非对齐语言模型在本地沙箱漏洞分析任务上的表现。

**💡 创新点**

以轨迹级别和证据扎实度评估安全代理的安全对齐，提出跨模型族的对比实验，展示对齐与去对齐对任务成功、证据生成和工具可靠性的不同影响。

**🔧 技术方法**

采用对话代理框架、工具调用、固定工具集、可重复的成功判定器、证据扎实度评分、二层LLM审核、统计检验（McNemar、bootstrap）等技术。

**📊 数据集**

30个本地沙箱漏洞分析任务（安全与非安全编码控制），共计1500条安全代理轨迹和800条非安全轨迹。

**📈 对比分析**

在相同任务、种子、预算和工具下进行配对比较，使用成功率、拒绝率、工具失败率、证据扎实度等指标。Gemma系列在安全任务上表现优于对齐版本（如14% vs 0.7%），但其他系列并未复现，整体仍难以通过硬性验证任务。

**⚠️ 局限性**

结果受模型族、大小、衍生物来源影响；工具接口不稳定、缺乏硬性验证能力；对齐与去对齐并非单一因子；仅限本地沙箱任务，未覆盖真实系统；未提供因果解释。

---

## 492. Can Large Language Models Reliably Correct Errors in Low-Resource ASR? A Contamination-Aware Case Study on West Frisian

**arXiv ID:** 2605.19711 | [PDF](https://arxiv.org/pdf/2605.19711v1)

**作者:** Yun Hao `[一作]` (University of Groningen), Martijn Wieling `[通讯]` (University of Groningen)

**通讯引用:** 4298 | [OpenAlex ID](https://openalex.org/A5075698724)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）在低资源语言弗里西语上的生成式错误纠正（GER）方法，并通过构造非公开的弗里西语离线数据集评估了模型在消除数据污染后仍能保持的纠错效果。

**💡 创新点**

创新点：①同时使用公开和非公开数据集，系统排查并验证数据污染对 GER 结果的影响；②展示 GPT‑5.1 在弗里西语上可突破传统五最佳 oracle 的性能；③对不同 LLM（GPT‑4o‑mini、GPT‑5.1、Qwen3‑8B）及其微调方式进行细粒度比较，并对错误类型展开深入分析。

**🔧 技术方法**

技术手段：采用 XLS‑R 1B 作为 ASR backbone，生成五最佳 hypotheses；对 LLM 进行零/少 shot prompting 以及 LoRA 微调；实现生成式和基于选择的 GER；使用 WER、oracle WER、trigram LM 等指标进行评估。

**📊 数据集**

数据集：Common Voice 17.0 Frisian 公开语料（训练/验证/测试各约 5 小时）以及自建的弗里西语离线数据集（811 句子，1.5 小时，文本来源为非公开的故事书与原创句子）。

**📈 对比分析**

评估方式：将 GER 结果与 XLS‑R baseline、五最佳 oracle、传统 trigram LM 进行对比。结果显示 GPT‑5.1 在 Common Voice 3‑shot 生成式下 WER 为 8.9%（低于 oracle 9.6%），在离线数据 10‑shot 下 WER 为 13.8%（亦低于 oracle 18.0%）。相比之下 Qwen3‑FT 提升有限，选择式 GER 的性能远低于生成式 GER。

**⚠️ 局限性**

局限性：①开源 LLM Qwen3‑FT 在低资源弗里西语上的纠错能力显著受限；②生成式模型在插入错误上精度较低，删除错误召回率不高，需进一步改进插入决策；③研究仅覆盖弗里西语，缺乏对多语言泛化的系统验证。

---

## 493. WBCAtt+: Fine-Grained Pixel-Level Morphological Annotations for White Blood Cell Images

**arXiv ID:** 2605.19692 | [PDF](https://arxiv.org/pdf/2605.19692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 494. Agentic Discovery of Cryomicroneedle Formulations

**arXiv ID:** 2605.19677 | [PDF](https://arxiv.org/pdf/2605.19677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 495. Completeness of Synthesis under Realizability Assumptions using Superposition

**arXiv ID:** 2605.19683 | [PDF](https://arxiv.org/pdf/2605.19683v1)

**作者:** Márton Hajdu `[一作]` (TU Wien), Eva Maria Wagner `[通讯]` (TU Wien)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5113271510)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种基于超位置（superposition）的程序合成新演算，改进了先前的框架并证明其在可实现性假设下的完整性；

**💡 创新点**

创新点在于引入抽象统一规则、分区简化顺序以及仅选择不可计算符号的选取策略，从而确保能在存在可计算程序的情况下找到解；

**🔧 技术方法**

技术上使用了归一化的多排序一阶逻辑、LPO/KBO简化顺序、答案子句（answer clause）以及基于重写系统的模型构造；

**📊 数据集**

论文未涉及具体实验数据集，主要聚焦于理论证明与框架设计；

**📈 对比分析**

因缺乏实验评测，无法给出性能对比，文中仅通过示例演示新规则的有效性；

**⚠️ 局限性**

局限性包括仅处理递归自由程序，对递归函数的合成尚未覆盖，且依赖于可实现性假设。

---

## 496. LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models

**arXiv ID:** 2605.19729 | [PDF](https://arxiv.org/pdf/2605.19729v1)

**作者:** Hyunsoo Han `[一作]` (Ulsan National Institute of Science and Technology), Jaejun Yoo `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 5354 | [OpenAlex ID](https://openalex.org/A5089933293)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

论文内容未在提供文本中明确说明

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

## 497. Memory-Augmented Reinforcement Learning Agent for CAD Generation

**arXiv ID:** 2605.19748 | [PDF](https://arxiv.org/pdf/2605.19748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 498. EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design

**arXiv ID:** 2605.19743 | [PDF](https://arxiv.org/pdf/2605.19743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 499. Decentralized Direct Volume Rendering: A Browser-Native GPU Architecture for MRI Digital Twins in Resource-Constrained Settings

**arXiv ID:** 2605.19737 | [PDF](https://arxiv.org/pdf/2605.19737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 500. Minimax Optimal Variance-Aware Regret Bounds for Multinomial Logistic MDPs

**arXiv ID:** 2605.19768 | [PDF](https://arxiv.org/pdf/2605.19768v1)

**作者:** Pierre Boudart `[一作]` (INRIA), Alessandro Rudi `[通讯]` (SDA Bocconi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了面向多项式逻辑回归（MNL）混合马尔可夫决策过程（MDP）的新学习算法 LIVAROT，并给出了上界与下界完全匹配的渐进最优 regret 分析。

**💡 创新点**

创新点包括：
• 引入了新的问题相关常数 σ̅_T（学习轨迹上最优下游价值函数的归一化平均方差），用以刻画转移误差对收益的实际影响；
• 通过自共形性质（self‑concordance）在探测期内构造紧致的置信集合，使算法在保守度上保持常数级别；
• 证明了 O(dH²σ̅_T√T) 的上界与 Ω(dH²σ̅_T√T) 的下界完全匹配，首次完成了 MNL 混合 MDP 的 minimax 复杂度表征。

**🔧 技术方法**

技术手段：
• 探索-学习框架，先进行多轮探索获取置信集；
• 采用在线镜像梯度（Online Mirror Descent）结合二阶对数损失逼近来更新参数估计；
• 通过 optimistic value 计算（OFU）与非凸最大化，保留方差信息；
• 利用自共形性控制对数损失曲率，并构造自适应探索奖励。

**📊 数据集**

本工作主要为理论分析，实验验证放在附录中，未公开使用具体公开数据集；实验主要演示了在 KL 约束鲁棒 MDP 情形下 σ̅_T = O(1/H) 的情形能显著减小 H 的依赖。

**📈 对比分析**

与此前的两类算法比较：
• 传统 O(κ d H²√T) 的上界（κ 为指数级问题常数）和 O(d H²√T) 的改进上界；
• 通过引入 σ̅_T，算法在 σ̅_T ≪ 1 的结构化 MDP 中实现了显著性能提升；
• 在鲁棒 MDP 中，σ̅_T = O(1/H) 时，H²σ̅_T → H，降低了 H 的影响；
• 下界证明显示 O(d H²σ̅_T√T) 为最优，表明该改进不可再进一步。

**⚠️ 局限性**

局限性：
• 需要先进行相对较长的探索期，理论上 τ 规模较大；
• 依赖已知的参数上界 B、κ、ρ，且假设特征向量范数 ≤1；
• 需要在每一步解决非凸的最大化问题，实际实现需使用如 Frank‑Wolfe 等数值近似；
• 对于非常大的状态空间，尽管使用特征表示但仍可能面临计算瓶颈。

---

## 501. AR1-ZO: Topology-Aware Rank-1 Zeroth-Order Queries for High-Rank LoRA Fine-Tuning

**arXiv ID:** 2605.19767 | [PDF](https://arxiv.org/pdf/2605.19767v1)

**作者:** Ziye Chen `[一作]`, Yao Shu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合Zeroth-Order（ZO）优化与LoRA技术实现大语言模型的高秩低内存微调；

**💡 创新点**

创新点在于发现并解决了“拓扑-尺度不匹配”问题，提出AR1‑ZO方案，通过在每一步仅对单个rank‑1原子进行查询并使用拓扑感知缩放γ=α r，使得高秩LoRA在纯黑盒ZO环境下保持有效的信号；

**🔧 技术方法**

核心技术包括ZO随机梯度估计、LoRA低秩因子分解、原子化查询策略、拓扑感知的缩放校正、以及对梯度结构的谱与对齐诊断；

**📊 数据集**

实验使用OPT‑2.7B/13B和Qwen3‑1.7B/32B模型，评估任务涵盖BoolQ、CB、COPA、WIC、SQuAD、DROP等数据集；

**📈 对比分析**

在与MeZO‑LoRA、LOZO、ZO‑Alt‑Naive以及基准FT(Adam)的对比中，AR1‑ZO在相同的两次前向评估预算下，整体提升了0.6%–13.2%的任务准确率或F1分数，且在大部分任务上逼近一阶优化的表现；

**⚠️ 局限性**

局限性包括：只验证了中等规模模型和少数分类/问答任务，未覆盖大规模生成任务或更复杂的张量化适配器，且对高秩覆盖成本的实际计算效率尚未深入探讨。

---

## 502. Graph Neural Networks for Community Detection in Graph Signal Analysis

**arXiv ID:** 2605.19733 | [PDF](https://arxiv.org/pdf/2605.19733v1)

**作者:** Roberto Cavoretto `[一作]` (University of Torino), Enrico Montini `[通讯]` (University of Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了将无监督图神经网络（GNN）社区检测结果作为 Partition of Unity 方法（PUM）中的局部子域，用 Graph Basis Function（GBF）完成图信号插值，从而实现大规模图信号的高精度重构。

**💡 创新点**

创新点在于：①将多种 GNN 聚类算法（GCN、GAT、AE、GAN 等）产生的社区直接嵌入 GBF‑PUM 插值框架，形成新的局部插值分区；②通过实验验证该组合可与传统插值方法媲美甚至超越，且对社区数不敏感；③提出了利用 GNN 自动聚类（如 DGCluster）减少人工调参的可能性。

**🔧 技术方法**

主要技术包括：无监督 GNN 社区检测（GCN、GAT、AE、GAN 等），Graph Basis Function（变分样条）插值，Partition of Unity Method（PUM）在图上的实现，以及模组度（Modularity）等质量评估指标。

**📊 数据集**

使用的实验数据集为 Bologna 与 Paris 两个城市网络，构造节点坐标特征后乘以随机矩阵得到高维特征矩阵，进而进行社区划分与插值。

**📈 对比分析**

方法对比：对九种无监督 GNN 模型（AGC、AMIL、AOCD、CDBNE、DGCluster、MAGAE、MAVGAE、MGAE、NOCD）在两张图上进行 GBF‑PUM 插值，并用 RMAE 与 RRMSE 评估。结果显示，绝大多数模型误差均在低水平，CDBNE、DGCluster 等模型在 Paris 图上表现尤为突出，误差相对传统方法低。

**⚠️ 局限性**

局限性：①需要人工指定社区数 J，且需多次实验以最大化模组度；②特征矩阵稠密导致计算量大，特别是对大型图；③缺乏自动稀疏特征构造与自适应聚类机制，限制了方法的可扩展性。

---

## 503. Mathematical Reasoning in Large Language Models: Benchmarks, Architectures, Evaluation, and Open Challenges

**arXiv ID:** 2605.19723 | [PDF](https://arxiv.org/pdf/2605.19723v1)

**作者:** Husnain Amjad `[一作]` (National University of Science and Technology), Mehwish Fatima `[通讯]` (National University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对数学推理在大规模语言模型中的数据集、架构、训练策略与评测方法进行系统综述与统一框架构建。

**💡 创新点**

创新点在于提出统一的数学推理数据集分类体系、跨维度的技术与评测对照表，以及对失败模式与评测盲区的综合性分析。

**🔧 技术方法**

使用结构化文献检索、PRISMA流程、系统综述方法以及对比分析技术，结合多维度评测指标（准确率、符号验证、过程监督）进行评估。

**📊 数据集**

评审了约120篇论文，涵盖了主要数据集如GSM8K、MATH、MiniF2F、OlympiadBench、OCW、U‑Math 等，且对比了它们在不同推理难度与格式上的表现。

**📈 对比分析**

通过对比标准评测指标（准确率、Exact Match）与过程级验证指标（符号可验证、步骤监督）的方式，梳理出现有LLM在算术、代数、文字推理与形式证明等任务上的性能差距，显示高级竞赛级别问题仍显不足。

**⚠️ 局限性**

局限性包括：数据集容易受训练泄露、合成推理示例偏倚、数值与符号表示不一致、评测过度依赖最终答案、缺乏可扩展的过程级验证、以及模型内部推理可信度不足。

---

## 504. Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization

**arXiv ID:** 2605.19721 | [PDF](https://arxiv.org/pdf/2605.19721v1)

**作者:** Franco Terranova `[一作]` (Université de Lorraine), Abdelkader Lahmadi `[通讯]` (Université de Lorraine)

**通讯引用:** 669 | [OpenAlex ID](https://openalex.org/A5111932337)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在图组合优化（GCO）中使用投影代理（projection agent）的强化学习方法，利用连续的 GNN 隐空间来预测动作并通过最近邻解码为离散可行动作。

**💡 创新点**

创新点包括：①将动作映射到连续隐空间一次前向传播即可决定动作，显著提升推理速度；②构建结构化的、超线性决策变量的嵌入，支持复杂实际任务；③采用无监督 GAE 预训练共享观测-动作嵌入，便于方法公平比较；④发布统一 Python 库，降低新任务集成门槛。

**🔧 技术方法**

核心技术包括：图神经网络（GNN）编码器、图自编码器（GAE）预训练、连续隐动作空间投影、最近邻（NN）解码、强化学习（如 PPO、QLearning）以及 FAISS 索引加速检索。

**📊 数据集**

数据集涵盖七个基准：经典 TSP、MinVertex、MaxCut 以及四个应用任务（虚拟机放置、网络攻击路径、OSPF 路由、流量工程），每个基准产生 101 个不同规模场景。

**📈 对比分析**

与离散动作、迭代 Q‑value 等基线相比，投影代理在大多数基准上实现了高达 16.2 倍的推理速度提升，且在未见实例上的泛化得分提升约 40%，在超线性决策空间中表现尤为优异。

**⚠️ 局限性**

局限性包括：仅使用单一动作组件编码和单一解码策略；无监督嵌入未在 RL 过程中微调，可能影响极限性能；实验规模有限（101 场景、20 轮），仅对 RL 方法进行比较，未覆盖所有非 RL 传统方法。

---

## 505. Asking Grok: AI-Assisted Sensemaking in Social Media Conversations

**arXiv ID:** 2605.19720 | [PDF](https://arxiv.org/pdf/2605.19720v1)

**作者:** Michelle Bobek `[一作]` (JLU Giessen), Nicolas Pröllochs `[通讯]` (JLU Giessen)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在X平台上公开嵌入式AI助手Grok的使用情况，并与社区笔记对比。

**💡 创新点**

揭示Grok作为快速、低门槛的公共理解层，与社区事实核查并行且时机领先，却覆盖面有限。

**🔧 技术方法**

利用LLM（Grok）生成回复，并用机器学习模型对提示意图与主题进行自动标注。

**📊 数据集**

基于2025年3月7日至5月28日期间的169,137个Grok提示-回复对，覆盖69,157条被Grok标注的目标推文。

**📈 对比分析**

通过对意图、主题、语言及用户行为的统计分析以及与社区笔记的交叉回归，发现Grok回应更快但受众小，社区笔记更受关注；两系统互不影响。

**⚠️ 局限性**

主要限制在未评估Grok回复准确性、单一AI助手和短期采样，且无法验证因果关系。

---

## 506. Multi-Session Ground Texture SLAM in Low-Dynamic Environments

**arXiv ID:** 2605.19701 | [PDF](https://arxiv.org/pdf/2605.19701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 507. LLM-Based Financial Sentiment Analysis in Arabic: Evidence from Saudi Markets

**arXiv ID:** 2605.19714 | [PDF](https://arxiv.org/pdf/2605.19714v1)

**作者:** Mona H. Albaqawi `[一作]` (George Mason University), Enrico Lopedoto `[通讯]` (City, St George's, University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个 84K 条阿拉伯语金融文本的语料库，并提出了统一的多阶段 NLP 流水线（预处理、实体链接、摘要生成、五分类情绪标注），用于沙特阿拉伯市场的情绪监测。

**💡 创新点**

创新点在于：①利用 LLM 进行语义推理与摘要；②采用多模型共识标注提升情绪标签可靠性；③将官方新闻与社交媒体内容结合，实现近实时、细粒度情绪分析；④通过基准实验展示 LLM 在该任务上显著优于传统方法。

**🔧 技术方法**

使用技术包括：大规模预训练阿拉伯 LLM（GPT‑5、GPT‑4、Gemini 等）、AraBERT、CAMeLBERT、ALLaM 摘要模型、Transformer NER、实体链接、互模型一致性评估（Cohen Kappa、Jensen–Shannon Divergence 等）。

**📊 数据集**

数据集：84K 条阿拉伯语金融文本，涵盖 74.8% 社交媒体、25.2% 官方新闻，覆盖所有 261 家沙特交易所上市公司，按强正、正、中立、负、强负五级进行标注。

**📈 对比分析**

与词典、SVM+TF‑IDF、AraBERT、CAMeLBERT 等传统基线对比，GPT‑5 在 Macro‑F1 上达到 0.829，远高于 AraBERT 0.547、CAMeLBERT 0.547 和词典 0.287；摘要模型 ALLaM 在质量、幻觉率与成本方面实现最佳折中。

**⚠️ 局限性**

局限性包括：仅覆盖沙特市场与特定时间段，未公开数据集；LLM 仍存在幻觉与偏见；对方言、隐含情绪捕捉不足；实体链接受非正式语言影响。

---

## 508. D-CLING: Prior-Preserving Depth-Conditioned Fine-Tuning for Navigation Foundation Models

**arXiv ID:** 2605.19690 | [PDF](https://arxiv.org/pdf/2605.19690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 509. SCARA: A Semantics-Constrained Autonomous Remediation Agent for Opaque Industrial Software Vulnerabilities

**arXiv ID:** 2605.19668 | [PDF](https://arxiv.org/pdf/2605.19668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 510. DocQT: Improving Document Forgery Localization Robustness via Diverse JPEG Quantization Tables

**arXiv ID:** 2605.19688 | [PDF](https://arxiv.org/pdf/2605.19688v1)

**作者:** Kylian Ronfleux-Corail `[一作]`, Nicolas Sidère `[通讯]` (La Rochelle University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统地研究了JPEG量化表分布不匹配对文档篡改定位模型的鲁棒性影响，并通过对比标准质量因子增强与真实业务量化表训练的实验验证了差异；

**💡 创新点**

首次引入DocQT量化表数据库，构建可模拟真实业务压缩的训练和评估流程，证明仅靠标准质量因子无法代表业务多样性，且仅对显式量化表感知的网络带来显著性能提升；

**🔧 技术方法**

采用因子化实验设计、两套JPEG重压缩管线（Standard-QT与Real-QT）、在ForensicHub框架下训练并评估FFDN与Mesorch两种架构，利用量化表输入与否的对照探究鲁棒性；

**📊 数据集**

使用DocTamper、RTM、T-SROIE、SROIE、Find-it、Find-it Again、FUNSD等公开数据集，以及MAIF保险业务图像语料，提取859个真实量化表构成DocQT；

**📈 对比分析**

通过在三种评估条件（无重压缩、Standard-QT重压缩、Real-QT重压缩）下比较四种模型，Real-QT训练的FFDN在DocTamper上提升最高14.5 F1，Mesorch受影响甚微；同时在MAIF真实文档上误报率降低近一阶；

**⚠️ 局限性**

局限包括仅评估两种架构、仅使用单一保险业务量化表分布、未探讨色度量化表和多通道特征、阈值固定、以及合成与真实篡改间仍存在显著性能差距。

---

## 511. TombWriter: Scaffolding Story Archeology through Beat-Level Interaction in Human-AI Co-Writing

**arXiv ID:** 2605.19681 | [PDF](https://arxiv.org/pdf/2605.19681v1)

**作者:** Hugo Andersson `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**通讯引用:** 8264 | [OpenAlex ID](https://openalex.org/A5034277315)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 TombWriter 工具，采用“故事考古”方法在 beat 级别与 LLM 互动，构建可持续的故事结构并随后生成 prose；

**💡 创新点**

创新点在于将 LLM 用作角色行为模拟器而非直接生成文本，强调作者在持久化、层级化结构中的 agency 与 ownership，首次提出 beat 级别的“故事考古”框架；

**🔧 技术方法**

使用 React+TypeScript 前端、Zustand 状态管理、卡片式 UI 与 DeepSeek V3.2 LLM 接口；

**📊 数据集**

实验中未使用公开大型数据集，评估以五名经验作家自行创作的故事为素材；

**📈 对比分析**

通过为期三天的定性用户研究（访谈、主题分析）比较工具与传统写作方式，发现 beat 级别可显著提升结构一致性和创意探索，但 prose 生成仍受限；没有客观性能指标；

**⚠️ 局限性**

局限包括样本量仅五人、仅经验作家、实验时间有限、未系统对比不同生成机制、缺乏量化评测、AI 在语音与人物深度方面的不足。

---

## 512. CriterAlign: Criterion-Centric Rationale Alignment for Code Preference Judging

**arXiv ID:** 2605.19665 | [PDF](https://arxiv.org/pdf/2605.19665v1)

**作者:** Zhenyu Li `[一作]` (King Abdullah University of Science and Technology), Peter Wonka `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 15183 | [OpenAlex ID](https://openalex.org/A5076768552)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种针对代码生成的配对人类偏好预测框架 CriterAlign，改造了基于标准的评估流程以适应比较式决策。

**💡 创新点**

创新点在于将评判过程从逐点评分转为配对准则判定，并引入分层判别、交换一致性过滤以及离线合成的人类偏好对齐引导。

**🔧 技术方法**

使用大型语言模型（如 Qwen2.5‑VL‑32B 等）进行准则生成、判定与最终综合，结合离线合成的 HPAG 进行对齐指引。

**📊 数据集**

主要在 BigCodeReward 数据集上进行实验，并在不同 judge 和合成器上验证其鲁棒性。

**📈 对比分析**

与传统点对点标准化和单体评估基线相比，CriterAlign 在 BigCodeReward 上将准确率从 60.4% 提升至 66.3%，在多模型实验中持续保持显著优势。

**⚠️ 局限性**

局限性包括对大型模型和大上下文窗口的依赖，HPAG 的生成成本较高，以及在极端顺序敏感或非可执行的代码场景中仍可能产生误判。

---

## 513. Security Analysis of Bitcoin's V2 Transport Protocol: Exploiting Design Implications for Sustained Eclipse and Downgrade Attacks

**arXiv ID:** 2605.19715 | [PDF](https://arxiv.org/pdf/2605.19715v1)

**作者:** Charmaine Ndolo `[一作]` (Dresden University of Technology), Florian Tschorsch `[通讯]` (Dresden University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对比特币 V2 加密 P2P 协议进行系统评估，并发现了新的 eclipse 和 downgrade 攻击方法。

**💡 创新点**

创新点在于利用 V2 处理解密错误导致的连接关闭，结合基于 TCP 负载长度的消息分类，完成了无状态的 eclipse 攻击；以及利用 V2 兼容机制在关键交换阶段注入 RST，将所有连接降级到 V1，从而重新打开旧攻击面。

**🔧 技术方法**

使用了网络层重放、TCP 负载长度侧信道分析、加密协议重建、Mininet 模拟、iptables 队列拦截、以及自研的最小 Bitcoin 客户端实现。

**📊 数据集**

主要数据集为自建的 621 节点 Mininet 虚拟网络（20 个 /16 子网、10 个 /32 子网、1 个受害节点），并结合公开网络的测量数据（节点连接数、协议版本占比）。

**📈 对比分析**

对比方法包括：① 用实际攻击代码在测试网执行 eclipse 攻击并记录被占用槽数、流量变化、延迟；② 对比旧 EREBUS 攻击所需时间；③ 对降级攻击在重启后 2 小时内全连接切换至 V1 的成功率；性能表现：eclipse 攻击在 ~5 小时内完成，远快于过去的数周；延迟平均约 40–50 微秒，吞吐量约 20–25kpps，影响可忽略。

**⚠️ 局限性**

局限性包括：攻击仅在 AS‑级网络控制下可行；未在真实主网大规模验证；对 V2 兼容节点行为做了简化假设；并未实现完整的协议改进（如长度隐藏、认证）——只能给出对策建议。

---

## 514. GeoMamba: A Geometry-driven MambaVision Framework and Dataset for Fine-grained Optical-SAR Object Retrieval

**arXiv ID:** 2605.19734 | [PDF](https://arxiv.org/pdf/2605.19734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 515. Efficient Long-Context Modeling in Diffusion Language Models via Block Approximate Sparse Attention

**arXiv ID:** 2605.19726 | [PDF](https://arxiv.org/pdf/2605.19726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 516. What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code

**arXiv ID:** 2605.19762 | [PDF](https://arxiv.org/pdf/2605.19762v1)

**作者:** Yuze Zhao `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 29209 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了10T token的高质量多域语料库，对代码和数学子语料进行精细拆分，利用Mixture-of-Experts（MoE）模型开展固定-token消融实验，揭示纯代码对数学推理的竞争效应，并提出并验证了“认知支架”结构化数学样本以提升复杂推理。

**💡 创新点**

①将代码拆分为纯可执行代码与混合式 Code‑NL，精准区分交叉域数据；②发现代码并非推理提升源，而是与知识密集任务竞争；③提出认知支架概念并通过结构化样本替换提升数学推理；④利用专家路由分析揭示数据交互机制。

**🔧 技术方法**

使用 Mixture-of-Experts（MoE）大模型；精细分层采样与动态路由；FastText 结构分类器识别认知支架；专家路由偏差与 Jensen‑Shannon (JS) 散度分析。

**📊 数据集**

10T token 的多域语料库，包含 Web、Code、Code‑NL、Math、Wikipedia、Books、Multilingual；数学子集包括 MATH、OlympiadBench、College Math、MathBench、GSM8K 等。

**📈 对比分析**

通过固定-token消融实验对比全数据模型与去除代码或数学模型，在五大能力维度（通用知识、编程能力、数学能力、综合推理、专业知识）进行评测。认知支架在复杂数学任务平均提升 17.56%，但对简单任务略降；整体发现代码竞争数学推理，数学竞争综合推理。

**⚠️ 局限性**

仅在固定-token预算下评估；认知支架比例与替换比例未做系统搜索；实验限定在特定模型规模与 MoE 配置；对其它任务与域推广的适用性尚未验证；可能受数据标注与分域误差影响。

---

## 517. MSAlign: Aligning Molecule and Mass Spectra Foundation Models for Metabolite Identification

**arXiv ID:** 2605.19752 | [PDF](https://arxiv.org/pdf/2605.19752v1)

**作者:** Paul Krzakala `[一作]` (Institut Polytechnique de Paris), Florence d'Alché-Buc `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 4755 | [OpenAlex ID](https://openalex.org/A5066766964)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于预训练单模态基础模型的分子检索方法 MSAlign，能够从 MS/MS 光谱中高效恢复代谢物的 2D 结构。

**💡 创新点**

创新点在于：①将 DreaMS（光谱）和 ChemBERTa（分子）这两种预训练基础模型通过轻量级 MLP 对齐，构建共享表示空间；②采用候选集 InfoNCE 损失，充分利用检索任务中的难负样本；③引入量化分布偏移的指标，系统评估不同数据拆分策略的泄漏与域漂移权衡。

**🔧 技术方法**

技术方法包括：多模态对齐、轻量级 MLP 投影、候选集 InfoNCE 对比学习、Wasserstein 距离的分布偏移度量、对比实验及消融研究。

**📊 数据集**

使用公开的三大基准数据集：NPLIB1、MassSpecGym（Formula 与 MCES 拆分）和 Spectraverse；候选集合来自 PubChem，按分子质量过滤并保留 256 个候选。

**📈 对比分析**

与 FFN、DeepSets、JESTR、Emb-Cos、FLARE、MIST、SAIL 等现有方法在三个基准上对比，MSAlign 在 R@1、R@5、R@20 上均达到或超过所有对手，尤其在 MassSpecGym Formula 拆分上实现 53.8% R@1、73.1% R@5、87.1% R@20。

**⚠️ 局限性**

局限性包括：①仅针对检索任务，尚未解决真正的 de novo 分子结构推断；②对候选集的依赖，若候选覆盖不足会影响性能；③在高域漂移（如 MCES 拆分）下仍难以取得竞争性结果，需要进一步的域适应技术。

---

## 518. CAIT: A Syntactic Parsing Toolkit for Child-Adult InTeractions

**arXiv ID:** 2605.19718 | [PDF](https://arxiv.org/pdf/2605.19718v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 519. Awakening the Hydra: Stabilizing Multi-Concept Backdoor Injection in Text-to-Image Diffusion Models

**arXiv ID:** 2605.19698 | [PDF](https://arxiv.org/pdf/2605.19698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 520. RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models

**arXiv ID:** 2605.19678 | [PDF](https://arxiv.org/pdf/2605.19678v1)

**作者:** Jingzhou Luo `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 32858 | [OpenAlex ID](https://openalex.org/A5100412937)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RoVLA，多一致性约束的视觉-语言-动作模型，显著提升在多变环境下的鲁棒性。

**💡 创新点**

创新点在于设计三种一致性约束（指令一致性、演化一致性、观测一致性），在端到端训练中显式强制模型对语义、动作演化和观测扰动保持不变性。

**🔧 技术方法**

使用双系统架构：InternVL3.5 作为语义提取器，Diffusion Transformer 进行连续动作生成，并结合流匹配、对抗扰动、梯度投影等技术实现一致性损失。

**📊 数据集**

在 LIBERO-Plus、RoboTwin 2.0 以及 Franka Research 3 的 5 个桌面抓取/放置任务上训练与评估，并使用 Qwen3-8B 生成指令同义句。

**📈 对比分析**

与 OpenVLA、π_0、RIPT-VLA、GR00T 等多种基线在 LIBERO-Plus、RoboTwin 2.0 和真实世界任务中对比，RoVLA 在语言和观测扰动子集上提升 16-27% 成功率，整体获得最佳平均成功率。

**⚠️ 局限性**

对细粒度接触动力学和复杂双手协调的任务鲁棒性提升有限，模型对布局和机器人状态变化的适应仍不如预期。

---

## 521. OpenComputer: Verifiable Software Worlds for Computer-Use Agents

**arXiv ID:** 2605.19769 | [PDF](https://arxiv.org/pdf/2605.19769v1)

**作者:** Jinbiao Wei `[一作]` (Yale NLP Lab), Arman Cohan `[通讯]` (Yale NLP Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于验证器的框架 OpenComputer，用以自动构建可验证的桌面软件世界并评估计算机使用代理。

**💡 创新点**

创新点在于将验证器作为环境与任务构造的核心原则，实现可执行、可审核的任务生成和奖励计算，并通过自我进化的验证层提升验证器可靠性。

**🔧 技术方法**

采用应用特定状态验证器、执行反馈驱动的自我进化循环、基于验证器的任务与环境合成管道，以及全轨迹记录的评估引擎。

**📊 数据集**

构建了覆盖 33 种桌面应用、共 1,000 任务的基准数据集，并结合真实应用的状态接口进行验证器开发。

**📈 对比分析**

实验显示 frontier 模型（如 GPT‑5.4）在该基准上取得 68.3% 的任务成功率，远低于在 OSWorld 的表现，验证器评估与人工判定的吻合度明显优于 LLM‑as‑judge。

**⚠️ 局限性**

局限性包括部分任务的成功标准无法完全通过程序化验证器实现，需要视觉或几何判断；此类任务被排除在主基准之外。

---

## 522. Synthesis and Evaluation of Long-term History-aware Medical Dialogue

**arXiv ID:** 2605.19766 | [PDF](https://arxiv.org/pdf/2605.19766v1)

**作者:** Hebin Hu `[一作]` (South-Central Minzu University), Yilin Kang `[通讯]` (South-Central Minzu University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一套基于知识引导和任务分解的长期医学对话生成与评估框架，构建了MediLongChat数据集并设计了三项纵向推理基准。

**💡 创新点**

创新点包括：①将疾病–并发症知识与病历信息结合实现可控病历生成；②通过任务分解逐步生成多轮对话，降低长文本幻觉；③提出五维评价体系与LLM-as-Judge相结合的评估方法。

**🔧 技术方法**

使用的大型语言模型（LLM）与多轮提示、知识指导生成、BERTopic主题聚类、句向量相似度、G‑Eval LLM评审、温度控制、角色与风格调控等技术。

**📊 数据集**

主要使用自构造的MediLongChat数据集，并与LoCoMo、Conversation Chronicles、MSC、NoteChat等公开长对话数据集进行对比。

**📈 对比分析**

通过自动化向量/主题度量和LLM评审进行评估；在IDR/CAR/SR任务上四类LLM的得分均未超过70%，表明跨会话推理仍弱；MediLongChat在多样性、连贯性等五维指标上显著优于基线。

**⚠️ 局限性**

局限性包括：合成数据仍与真实临床分布偏离；LLM评审受提示与模型偏差影响；缺乏多模态、少见疾病和行为健康等复杂情境，仅限单语种。

---

## 523. GroupAffect-4: A Multimodal Dataset of Four-Person Collaborative Interaction

**arXiv ID:** 2605.19765 | [PDF](https://arxiv.org/pdf/2605.19765v1)

**作者:** Meisam Jamshidi Seikavandi `[一作]` (GN Advanced Science, GN Group), Andrew Burke Dittberner `[通讯]` (GN Advanced Science, GN Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多模态数据集 GroupAffect-4，包含40名参与者在10个四人小组中完成四个协作任务，采集生理、眼动、音频、自评、个性与任务结果等多种同步信号。

**💡 创新点**

创新点在于高密度、同步的多模态小组互动数据集，并在单一结构中同时支持个体、互相之间、以及群体级别的情感分析；同时提出了基于 BIDS 的数据组织和可复现的基准任务。

**🔧 技术方法**

采用了 LSL 同步、Tobii 眼动、EmotiBit 生理、近距离麦克风音频、SAM 自评、BFI 个性测量，以及 BIDS/Croissant 元数据和可复现的预处理流水线。

**📊 数据集**

使用的主要数据集是 GroupAffect-4（10 组×4 人，共 40 名参与者），并对其内的各任务（信息共享、谈判、创意生成、公共商品游戏）进行时间窗口划分。

**📈 对比分析**

通过留一组交叉验证（LOGO-CV）与岭回归/逻辑回归基线进行对比；在个体级别的情感预测中，心理需求最高 AUC≈0.72；情绪正负与唤醒相对较低；个性预测与群体动力学基线表现接近随机或失败。

**⚠️ 局限性**

局限性包括样本量小（仅 10 组）、单一地点与英语环境、任务顺序固定、音频数据受 DUA 限制、眼动未完成全局校准、以及缺乏临床/多语言验证。

---

## 524. CogScale: Scalable Benchmark for Sequence Processing

**arXiv ID:** 2605.19758 | [PDF](https://arxiv.org/pdf/2605.19758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 525. CPC-VAR:Continual Personalized and Compositional Generation in Visual Autoregressive Models

**arXiv ID:** 2605.19750 | [PDF](https://arxiv.org/pdf/2605.19750v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 526. FlowErase-RL: Rethinking Concept Erasure as Reward Optimization in Flow Matching Models

**arXiv ID:** 2605.19739 | [PDF](https://arxiv.org/pdf/2605.19739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 527. Physics-informed simulation framework for realistic sonar image generation and statistical validation

**arXiv ID:** 2605.19712 | [PDF](https://arxiv.org/pdf/2605.19712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 528. The Privacy Subsidy in Glosten-Milgrom: Bid-Ask Spread and Welfare under Flip-Noise Direction Observation

**arXiv ID:** 2605.19742 | [PDF](https://arxiv.org/pdf/2605.19742v1)

**作者:** Yuki Nakamura `[一作]` `[通讯]` (Open University of Japan), Yuki Nakamura (Open University of Japan)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在Glosten‑Milgrom 1985的两状态连续交易模型中加入了市场做市商观察到的交易方向被二进制对称通道噪声干扰的情形，推导出闭式的买卖价差和福利分解。

**💡 创新点**

创新点在于将信息隐私机制（通过翻转概率η）引入传统的市场微观结构模型，得到买卖价差 μ(1‑2η)Δ 和隐私补贴 μηΔ，首次把隐私补贴概念从高斯加噪的Kyle模型推广到离散的GM模型，显示了该现象在两类经典模型中的稳健性。

**🔧 技术方法**

主要技术包括贝叶斯推断、信息论的二进制对称通道建模、闭式解析求解以及福利分解（将交易者收益与做市商损失相加为零的关系）。

**📊 数据集**

未使用任何真实数据集；该工作完全是理论推导和公式分析。

**📈 对比分析**

与传统的无噪声GM模型进行对比，发现噪声导致价差线性收缩，且交易者（包括噪声交易者和信息交易者）均获益，隐私机制的成本全部由做市商承担；还与先前的高斯噪声Kyle模型比较，证明两者得到相似的隐私补贴形式，表明结果具有普适性。

**⚠️ 局限性**

局限性包括：仅考虑对称先验和对称噪声通道；仅处理单期一次性交易；未考虑多期动态演化、做市商策略的自适应变化以及隐私参数 η 的内生优化；未给出对非对称先验或非对称翻转概率的闭式解。

---

## 529. ContextRAG: Extraction-Free Hierarchical Graph Construction for Retrieval-Augmented Generation

**arXiv ID:** 2605.19735 | [PDF](https://arxiv.org/pdf/2605.19735v1)

**作者:** Roman Prosvirnin `[一作]` (HSE University), Seungmin Jin `[通讯]` (HSE University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了无实体提取的图结构检索增强生成系统 ContextRAG。

**💡 创新点**

创新点在于用残差量化 K‑Means 与模糊 Formal Concept Analysis 结合 Łukasiewicz 逻辑，完成无 LLM 实体抽取的图拓扑。

**🔧 技术方法**

采用 Residual‑Quantization K‑Means、模糊 FCA、Łukasiewicz t‑norm、混合检索与多查询重写技术。

**📊 数据集**

在 UltraDomain 130 任务子集上进行实验。

**📈 对比分析**

与 HiRAG 的索引成本对比，ContextRAG 仅需 22k 令牌（比 HiRAG 23M 少 1043 倍），多跳 F1 36.8%，总体 33.6%。

**⚠️ 局限性**

局限在检索延迟较高、仅在小规模语料有效、缺乏完整无图基线、无法验证 LLM 提取图更优等。

---

## 530. Aero-World: Action-Conditioned Aerial Video Generation from Inertial Controls

**arXiv ID:** 2605.19728 | [PDF](https://arxiv.org/pdf/2605.19728v1)

**作者:** Abdul Mohaimen Al Radi `[一作]` (University of Central Florida), Yu Tian `[通讯]` (University of Central Florida)

**通讯引用:** 14318 | [OpenAlex ID](https://openalex.org/A5100373119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将预训练的图像到视频潜在扩散模型转化为受 6-DoF IMU 轨迹控制的无人机视频生成器，并通过冻结的物理探测器在 LoRA 微调阶段提供潜在空间运动一致性监督。

**💡 创新点**

创新点包括：① 引入冻结的 Physics Probe 在潜在空间对生成视频的运动一致性进行可微监督；② 通过动作‑token 流将连续加速度与角速度嵌入扩散变换器；③ 提出了 AeroBench、AAS 与 PCR 两个评价指标，用以客观衡量动作对齐与时间平稳性；④ 在保持预训练模型视觉先验的同时，轻量化实现动作可控视频生成。

**🔧 技术方法**

使用的核心技术包括：预训练的潜在扩散模型（如 CogVideoX），LoRA 低秩适配，冻结物理探测器（基于 3D 卷积 + MLP 的离散分类器），动作嵌入+交叉注意力，潜在空间物理一致性损失，Action Alignment Score 与 Physical Consistency Rate 评价指标，以及独立的 Flow‑IMU RGB‑空间验证。

**📊 数据集**

数据集：同步 RGB 视频与 IMU 传感器的 UZH FPV 航拍和 TII 航拍数据集，约 7k 训练、1k 验证、1k 测试片段；Probe 训练使用 VAE 潜在码与 IMU 对齐。

**📈 对比分析**

与预训练模型、Cosmos、AirScape、仅动作微调等基线在 AeroBench 上对比。Aero‑World 在 FVD 596.5（低于 AirScape 1058.6）、SSIM 0.595（高于 AirScape 0.505）、AAS 63.6（高于 Action FT 57.7）以及 PCR 0.03（接近最佳）等指标上表现更好；Flow‑IMU 相关系数 0.44 也高于对手，说明生成视频与真实运动更匹配。

**⚠️ 局限性**

局限性包括：仍存在视觉质量与控制一致性之间的权衡；Physics Probe 只在潜在空间监督，可能无法捕捉所有真实运动细节；在高度动态或极端场景下仍可能出现漂移；仅支持 6‑DoF 惯性输入，无法直接映射至低层马达指令；需要额外的离线 Probe 训练，限制了快速适配不同平台。

---

## 531. Operationalising Artificial Intelligence Bills of Materials (AIBOMs) for Verifiable AI Provenance and Lifecycle Assurance

**arXiv ID:** 2605.19755 | [PDF](https://arxiv.org/pdf/2605.19755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 532. RefiningGPT: Specialized language Models for Automated Refinery Unit-level Process Diagram Synthesis

**arXiv ID:** 2605.19704 | [PDF](https://arxiv.org/pdf/2605.19704v1)

**作者:** Dongxiao Liu `[一作]` (Beijing University of Posts and Telecommunications), Xiaoyong Li `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 3497 | [OpenAlex ID](https://openalex.org/A5100673949)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向炼油工艺图自动合成的专用框架 RefiningGPT，并在炼油工程中实现了端到端的单元选择与拓扑生成。

**💡 创新点**

创新点包括：① “Think‑then‑Draw”分层架构，先由细化的 8B 语言模型做单元选择与工程推理，再由知识增强的 LLM 通过约束感知 RAG 生成拓扑；② 通过从 20 条真实工艺图自动抽取高质量的三元组（意图、推理、单元）构建 500 条 SFT 训练集；③ 对 LLM 的输出实施硬约束 Φ(G)，确保每条边满足材料输入/输出兼容性。

**🔧 技术方法**

主要技术手段包括：监督微调（SFT）+链式推理（CoT）训练、知识检索增强生成（RAG）+约束感知、链式思考与自检、LoRA 微调、温度/Top‑p/Top‑k 解码策略。

**📊 数据集**

使用的数据集有：① 20 条真实炼油工艺图（用于提取 500 条 SFT 三元组）② ChemFlow‑Bench（基准测试集，包含 3 种典型炼油类型的工艺图与单元选择标签）。

**📈 对比分析**

与 Llama3.1‑8B、Qwen3‑8B、Qwen3‑32B、Qwen3‑235B、DeepSeek‑V3、DeepSeek‑R1 等模型做对比。RefiningGPT 在单元选择 F1 上达 70%（高于最佳 55%），推理质量 CoT‑C 为 0.68（高于 0.41），在图谱合成指标（nGED 低、CSPC 及 IOV 高）上也取得 10/12 维度的最优或近优表现。

**⚠️ 局限性**

局限性：① 随着检索上下文长度增大，I/O 合规率显著下降，表明模型在长距离约束推理上仍不稳健；② 目前仅针对炼油工艺，跨领域泛化仍待验证；③ 对手工标注的三元组质量高度依赖，若数据不足会影响性能。

---

## 533. Tango3D: Towards Alignment for Global and Local 2D-3D Correspondence

**arXiv ID:** 2605.19727 | [PDF](https://arxiv.org/pdf/2605.19727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 534. Beyond Rational Illusion: Behaviorally Realistic Strategic Classification

**arXiv ID:** 2605.19674 | [PDF](https://arxiv.org/pdf/2605.19674v1)

**作者:** Xinpeng Lv `[一作]` (National University of Defense Technology), Haotian Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 48277 | [OpenAlex ID](https://openalex.org/A5100383998)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文针对传统战略分类中假设代理人完全理性导致的鲁棒性缺失问题，提出了行为现实战略分类（BR-SC）框架，并基于前景理论构建了 Prospect-Guided Strategic Framework（Pro‑SF），从损失规避、参照偏差、概率扭曲三个心理机制入手对代理人策略进行建模。

**💡 创新点**

创新点在于：①首次将前景理论的三大心理机制系统化地融入战略分类的 Stackelberg 游戏中；②定义了“过度防御”和“防御不足”两种失效模式并给出理论分析；③通过 Pro‑SF 实现了对非理性代理行为的自适应防御，显著提升了模型在真实环境中的稳健性。

**🔧 技术方法**

技术手段包括：前景理论的价值函数和概率加权函数；基于最大化行为现实效用的代理人最优操纵模型；针对 Pro‑SF 的学习目标和解算算法；参数学习使用最大似然和离散选择模型；对比实验使用线性分类器（Logistic/线性 SVM）和基于 Mahalanobis 距离的成本函数。

**📊 数据集**

使用的实验数据集包括：成人收入（Adult）、信用卡（Credit）、糖尿病（Diabetes）、德国信用（German）、垃圾邮件（Spam）以及一个人工合成数据集。

**📈 对比分析**

与传统完全理性战略分类模型相比，Pro‑SF 在所有五个数据集的非理性和混合行为场景下均获得显著的准确率提升（例如在信用卡数据上从 78.42% 提升至 81.50%），且在完全理性场景下保持竞争力。实验还通过 ablation 证实每个心理机制对性能的贡献，并在参数敏感性实验中展示了鲁棒性。

**⚠️ 局限性**

局限性包括：①需要先验或学习得到前景理论参数，参数估计不确定时可能影响性能；②目前仅在线性模型和小规模数据上验证，尚未扩展到深度学习或大规模工业系统；③对代理人信息的假设仍介于完全公开和完全无信息之间，实际部署时信息不对称的处理仍有待进一步研究。

---

## 535. Landscape-Awareness for Geometric View Diffusion Model

**arXiv ID:** 2605.19865 | [PDF](https://arxiv.org/pdf/2605.19865v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 536. Transforming Constraint Programs to Input for Local Search

**arXiv ID:** 2605.19671 | [PDF](https://arxiv.org/pdf/2605.19671v1)

**作者:** Jo Devriendt `[一作]` (University of Leuven), Marc Denecker `[通讯]` (University of Leuven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对约束优化问题的对称性进行自动检测，提出一种将对称性诱导的邻域直接转换为局部搜索算法输入的方法，并在六个经典优化实例中实现并验证了该技术。

**💡 创新点**

创新点在于首次将问题的可变对称性与局部搜索邻域关联，利用对称性检测自动生成邻域，而非人工手工定义，从而实现了自动化的邻域构造。

**🔧 技术方法**

主要技术包括在约束编程系统中实现域元素交换对称性检测、构造对称性诱导邻域、将约束优化规范转换为ECNF并由懒惰子句生成求解器求解。

**📊 数据集**

实验使用了六个经典约束优化实例的公开数据集：TSP、最短路径、最大团、着色、背包和分配问题。

**📈 对比分析**

与人工设定的邻域进行对比，实验结果显示对称性诱导邻域在不同规格下保持鲁棒性，并在某些问题上与人工邻域相同或更优，但未给出详细数值指标，只说明可行性。

**⚠️ 局限性**

主要局限包括仅检测域元素交换对称性，难以处理需要放宽约束以获得更大邻域的情况；对大规模问题的可扩展性尚未验证；未与局部搜索引擎充分整合，缺乏完整的性能评估。

---

## 537. A Hierarchy of Tinhofer Graphs: Separations and Membership Testing

**arXiv ID:** 2605.19702 | [PDF](https://arxiv.org/pdf/2605.19702v1)

**作者:** Sutanay Bhattacharjee `[一作]` (Indian Institute of Technology Madras), Jayalal Sarma `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 k‑Tinhofer 图层次结构，对 Tinhofer 图进行了细分，并给出了其代数与组合学特征，证明该层次严格递增，并对层次间的判定问题给出了 P‑完全难度与固定参数可解性结果。

**💡 创新点**

创新点在于：①首次将 Tinhofer 图细分为 k‑Tinhofer 层次；②给出 k‑Tinhofer 的两种等价表述（Orbit‑Partition 与 IR‑树/商图）；③构造多级分离图（利用 CFI 与 IMP gadget）证明层次严格；④证明在已知 k‑Tinhofer 的前提下判定是否属于 (k+1)‑Tinhofer 为 P‑完全难；⑤引入 Tinhofer 缺陷参数，提出相应的 FPT 算法。

**🔧 技术方法**

核心技术包括：颜色细化（1‑WL），顶点个体化与 Refinement 过程，IR‑树与商图分析，CFI 与 IMP gadget 作为构造工具，以及从单调电路值问题的归约来证明复杂度。

**📊 数据集**

本文为理论工作，没有使用实际数据集；所有实验均为构造性证明与图形示例。

**📈 对比分析**

由于为理论性研究，没有与实验方法比较；所给定的 FPT 算法复杂度为 O(2^k·n^O(1))，其中 k 为 Tinhofer 缺陷，若 k 为常数则实现多项式时间。

**⚠️ 局限性**

局限性：判定 (k+1)‑Tinhofer 的难度尚未进一步降低；缺陷参数的核化（kernelization）问题仍未解决；在更一般的图类上，层次结构与标准图同构算法的关系仍不完全清晰。

---

## 538. Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning

**arXiv ID:** 2605.19852 | [PDF](https://arxiv.org/pdf/2605.19852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 539. A Framework for Evaluating Zero-Shot Image Generation in Concept-based Explainability

**arXiv ID:** 2605.19855 | [PDF](https://arxiv.org/pdf/2605.19855v1)

**作者:** Giacomo Astolfi `[一作]` (Politecnico di Milano), Marco Brambilla `[通讯]` (Politecnico di Milano)

**通讯引用:** 8945 | [OpenAlex ID](https://openalex.org/A5091879812)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个基于零样本文本-图像（T2I）生成模型的概念生成与验证框架，并系统评估合成概念与真实概念在可解释性（XAI）任务中的一致性。

**💡 创新点**

创新点在于：①首次将零样本 T2I 生成与预定义提示结合用于概念化解释；②从四个维度（表示对齐、内在相似度、下游解释差异、对抗性去除）评估合成概念的真实性；③提供公开合成概念数据集与完整评测脚本。

**🔧 技术方法**

主要技术包括：Flux、Stable Diffusion 3.5、GPT‑Image‑1 三种零样本 T2I 生成器；I2I 生成器（GPT‑Image‑1）用于概念去除；视觉 TCAV 与 CLIP 用于构造 CAV、计算重要性得分；统计分析（余弦相似度、KS 检验、Spearman 相关）评估结果。

**📊 数据集**

使用了 41 个概念，来源于 DTD、ImageNet、FMD、Flickr 搜索结果，总计 3,860 张概念图与 2,100 张类别图，涵盖纹理、物体、材料等多种视觉抽象层级。

**📈 对比分析**

比较方法：对合成与真实概念的 CAV 余弦相似度、子集内在相似度、重要性得分差异及去除后重要性变化。实验显示：平均相似度约 0.54–0.59，差异显著；重要性差异均低于 0.05 但方差大，且 KS 检验显著；去除后重要性变化在合成概念上显著小于真实概念，说明合成概念对解释影响不足。

**⚠️ 局限性**

限制：合成概念在语义对齐、视觉多样性与细节上仍存在显著偏差，往往缺乏真实图像的复杂性；I2I 去除效果不够彻底，导致对抗性评估受限；实验仅覆盖三种 T2I 模型，未涉及细调或多模态微调；结果表明在可解释性任务中直接使用零样本合成概念仍需进一步改进。

---

## 540. Auditing Privacy in Multi-Tenant RAG under Account Collusion

**arXiv ID:** 2605.19847 | [PDF](https://arxiv.org/pdf/2605.19847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 541. A Closed-loop, State-centric, Multi-agent Framework for Passenger Load Estimation from Heterogeneous Data Streams

**arXiv ID:** 2605.19834 | [PDF](https://arxiv.org/pdf/2605.19834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 542. Fast 4D Mesh Generation by Spatio-Temporal Attention Chains

**arXiv ID:** 2605.19786 | [PDF](https://arxiv.org/pdf/2605.19786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 543. Set-Valued Policy Learning

**arXiv ID:** 2605.19830 | [PDF](https://arxiv.org/pdf/2605.19830v1)

**作者:** Laura Fuentes-Vicente `[一作]` (Inria PreMeDICaL), Julie Josse `[通讯]` (Inria PreMeDICaL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了多处理设置下的集合值策略学习框架，利用置信区间或对数置信集生成一组可行治疗方案，进而为临床决策者提供可量化的不确定性评估与决策支持。

**💡 创新点**

创新点在于首次将集合值策略与置信下界方法和可噪声的 conformal 预测结合起来，既实现了条件与边际覆盖保证，又兼顾了多治疗情境下的可解释性与灵活性。

**🔧 技术方法**

主要技术包括置信下界（GLB）方法、基于噪声标签的 conformal 预测、随机性注入与稀疏非一致性评分，以及多模型叠加与双稳健 Q‑learning 的估计。

**📊 数据集**

实验数据包括人工合成的多治疗样本以及真实的 18,538 例体外受精（IVF）卵巢刺激周期数据，涵盖多种治疗剂量与二元/连续结局。

**📈 对比分析**

与或acular CP、传统 GLB 等方法比较时，本文方法在保证 1‑α 覆盖的同时保持了较低的集合大小和较高的集合政策价值，尤其在 IVF 数据中通过 δ_lower 策略实现了收益与安全性的平衡。

**⚠️ 局限性**

局限性包括对置信区间构造的依赖（需高质量上下界估计）、随机性注入参数 r 的选择难度、以及在高度噪声或复杂因果结构下可能出现的过度覆盖或信息不足。

---

## 544. Stitched Value Model for Diffusion Alignment

**arXiv ID:** 2605.19804 | [PDF](https://arxiv.org/pdf/2605.19804v1)

**作者:** Hyojun Go `[一作]` (ETH Zurich), Konrad Schindler `[通讯]` (ETH Zurich)

**通讯引用:** 24542 | [OpenAlex ID](https://openalex.org/A5005404030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过模型拼接（model stitching）把预训练好的像素级奖励模型（如CLIP、HPSv2等）迁移到扩散模型的噪声潜在空间，得到可直接评估噪声潜在的价值模型（StitchVM）

**💡 创新点**

创新点在于：① 只需一次轻量级的拼接与微调即可将高质量的像素级奖励模型迁移到噪声潜在；② 通过在冻结的扩散骨干上拼接奖励模型的“尾部”，实现无额外去噪器和VAE解码的直接价值评估；③ 在推理和训练两侧均能显著提升效率与效果

**🔧 技术方法**

核心技术包括：模型拼接（model stitching）、线性映射对齐、无监督微调、扩散模型的噪声潜在处理、推理时的Feynman‑Kac（FK）引导与Diffusion Posterior Sampling（DPS）

**📊 数据集**

使用的公开数据集有 AVA、HPDv2 进行无标签微调；评估时使用 MSCOCO、Flickr30K、ImageReward、HPDv2、AVA test 进行检索、偏好预测和审美评估；对比基线包括 VIST3A、NoisyCLIP、DiNa‑LRM

**📈 对比分析**

与基线比较：StitchVM 在低噪声下几乎与原始像素奖励模型同等，且在高噪声下性能更稳健；在推理时替代 Tweedie/MC 近似可使 DPS 速度提升 3.2×、显存降低 50%；FK 引导中可采用 M‑scaling 而非 N‑scaling，进一步节省计算；在训练时可将 Rollout 截断到中间噪声潜在，DiffusionNFT 速度提升 55%+，DRaFT 质量提升

**⚠️ 局限性**

局限性包括：仍需先对拼接点进行搜索与微调，模型间兼容性依赖于线性可映射；在极高噪声下价值预测仍略逊于原始奖励模型；目前仅验证于扩散/流模型，未探讨其他生成框架的通用性

---

## 545. Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles

**arXiv ID:** 2605.19775 | [PDF](https://arxiv.org/pdf/2605.19775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 546. FineBench: Benchmarking and Enhancing Vision-Language Models for Fine-grained Human Activity Understanding

**arXiv ID:** 2605.19846 | [PDF](https://arxiv.org/pdf/2605.19846v1)

**作者:** Gueter Josmy Faure `[一作]` (National Taiwan University), Winston H. Hsu `[通讯]` (National Taiwan University)

**通讯引用:** 6370 | [OpenAlex ID](https://openalex.org/A5043898632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FineBench，一个针对长视频中细粒度人类行为的多项选择问答基准，包含199,420条 QA 及高密度时空标注。

**💡 创新点**

创新点包括：① 大规模细粒度人类视频问答数据集；② 结合 Localizer 与 Descriptor 的模块化框架，针对多人物空间歧义和细微动作提升 VLM 表现；③ 在公开模型上无训练即可提升性能。

**🔧 技术方法**

使用的技术主要是：模板化问答生成、基于 AVA 的语义相似性负样本构造、基于 EVFSam 的定位模块、基于 Qwen2.5-VL 的描述生成，以及对现有 VLM 的多模态推理。

**📊 数据集**

数据集为 FineBench（199,420 QA 对），来源于 AVA v2.2 的 64 条 15 分钟长视频，平均每视频 785 个关键帧，覆盖人动作、互动、物体操作等三大类。

**📈 对比分析**

通过在代表性子集和全量数据上评估，GPT‑5、Gemini 等专有模型在子集上达约77%准确率；公开模型最高为 Qwen2.5‑VL（7B）68.8%；加入 Localizer+Descriptor 后，所有公开模型平均提升 4–8%（如 InternVL‑2.5 从 44.1% 提升至 52.4%）。

**⚠️ 局限性**

局限性：公开 VLM 在多人物场景的空间推理仍差强人意，且对细微动作与交互的理解有限；当前改进方法仍依赖外部定位与描述模块，未在模型内部学习空间关系。

---

## 547. Justifying bio-inspired robotics research: A taxonomy of strategies

**arXiv ID:** 2605.19840 | [PDF](https://arxiv.org/pdf/2605.19840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 548. Mega-ASR: Towards In-the-wild^2 Speech Recognition via Scaling up Real-world Acoustic Simulation

**arXiv ID:** 2605.19833 | [PDF](https://arxiv.org/pdf/2605.19833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 549. Material for Thought: Generative AI as an Active Creative Medium

**arXiv ID:** 2605.19832 | [PDF](https://arxiv.org/pdf/2605.19832v1)

**作者:** Hugo Andersson `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**通讯引用:** 8264 | [OpenAlex ID](https://openalex.org/A5034277315)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将生成式AI视为主动创作媒介，构建SOSS（Shape‑Observe‑Stir‑Select）框架，并在Loom写作工具中实现了基于LLM的叙事代理交互。

**💡 创新点**

创新点在于把人机协作从评估AI输出转向调度与塑造AI创作空间，借鉴Schön的反思实践理论，首次系统化定义了调度、观察、干预和选择四个交互阶段。

**🔧 技术方法**

技术实现基于大语言模型（如GPT‑4）作为叙事代理，结合分支选择、事件注入（Stir）和记忆管理的交互界面，支持作者动态塑造故事情节。

**📊 数据集**

未使用专门的数据集，主要依赖公开预训练的LLM权重与通用文本语料库生成对话与剧情；对话和情节均在实时推理中产生。

**📈 对比分析**

论文未进行定量对比实验，主要通过案例演示和设计讨论说明SOSS框架的可行性；因此缺乏性能指标或对比结果。

**⚠️ 局限性**

局限性包括缺乏实证评估验证SOSS是否真正提升创作质量，对不同创意领域的泛化性未知；LLM的趋同倾向需要频繁人工干预，系统对模型稳定性的依赖较高。

---

## 550. Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions

**arXiv ID:** 2605.19823 | [PDF](https://arxiv.org/pdf/2605.19823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 551. ST-TGExplainer: Disentangling Stability and Transition Patterns for Temporal GNN Interpretability

**arXiv ID:** 2605.19822 | [PDF](https://arxiv.org/pdf/2605.19822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 552. General Lower Bounds for Differentially Private Federated Learning with Arbitrary Public-Transcript Interactions

**arXiv ID:** 2605.19813 | [PDF](https://arxiv.org/pdf/2605.19813v1)

**作者:** Yicheng Li `[一作]` (Tsinghua University), Yicheng Li `[通讯]` (Tsinghua University)

**通讯引用:** 42251 | [OpenAlex ID](https://openalex.org/A5100421454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在联邦差分隐私（federated DP）环境下，给出了一个通用的 van Trees 下界，用于评估任意参数在平方 ℓ_2 损失下的估计误差；

**💡 创新点**

创新点在于：① 允许任意公共转录（public‑transcript）交互与自适应轮次；② 允许同一局部样本在多轮中重用；③ 通过单客户端 Fisher 信息收缩不等式，将整体 Fisher 信息上界为各客户端的隐私限制与样本量的调和和；

**🔧 技术方法**

主要技术包括：零浓度差分隐私（zCDP）框架、Fisher 信息收缩与投影同态、van Trees 边界以及后验独立性证明；

**📊 数据集**

论文未使用具体实验数据集，而是通过理论推导给出了均值估计、线性回归、非参数回归等经典统计模型的下界；

**📈 对比分析**

与现有中枢/本地 DP 下界对比，本文的下界在异构样本与隐私预算下能匹配已知最优速率，并且在多轮交互与样本复用场景下保持最优性；

**⚠️ 局限性**

限制：1）仅给出理论下界，未给出匹配上界或具体算法；2）假设局部样本 i.i.d.，且全局参数满足 van Trees 正则性；3）隐私模型仅为 zCDP（可转换为 (ε,δ)‑DP），对其它 DP 变体的适用性需进一步验证。

---

## 553. AffectAI-Capture: A Reproducible Multimodal Protocol for Small-Group Meeting Research

**arXiv ID:** 2605.19794 | [PDF](https://arxiv.org/pdf/2605.19794v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 554. Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization

**arXiv ID:** 2605.19782 | [PDF](https://arxiv.org/pdf/2605.19782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 555. Mechanisms of Object Localization in Vision-Language Models

**arXiv ID:** 2605.19792 | [PDF](https://arxiv.org/pdf/2605.19792v1)

**作者:** Timothy Schaumlöffel `[一作]` (Goethe University Frankfurt), Gemma Roig `[通讯]` (Goethe University Frankfurt)

**通讯引用:** 2393 | [OpenAlex ID](https://openalex.org/A5025034643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究者通过对LLaVA‑1.5与InternVL‑3.5进行token消融、注意力剔除及因果中介分析，揭示了VLM在物体定位时的容器化机制与极少数关键注意力头的作用。

**💡 创新点**

创新点在于首次从层级与注意力头层面系统解析定位任务，发现定位依赖少数早期或中后层头，并与分类任务共享一小部分头，表明定位本质上是先识别后定位。

**🔧 技术方法**

主要技术包括视觉‑语言模型的token消融、注意力knockout、因果中介分析（激活补丁）以及位置回归器评估空间信息。

**📊 数据集**

使用经过人工纠错与质量过滤的COCO验证集子集，并构造了去物体的补全对照集以消除上下文偏差。

**📈 对比分析**

与完整模型基线对比，消融关键头后定位准确率显著下降；实验表明全局视图贡献最大，局部视图主要提升小物体分类，整体性能与现有方法相当但揭示了内部机制。

**⚠️ 局限性**

局限性包括仅评估两种VLM，使用单对象过滤的COCO子集，未覆盖多对象、动态视频或其他模型结构，且因果分析仅聚焦于注意力头。

---

## 556. Synergistic Foundation Models for Semi-Supervised Fetal Cardiac Ultrasound Analysis: SAM-Med2D Boundary Refinement and DINOv3 Semantic Enhancement

**arXiv ID:** 2605.19799 | [PDF](https://arxiv.org/pdf/2605.19799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 557. From SGD to Muon: Adaptive Optimization via Schatten-p Norms

**arXiv ID:** 2605.19781 | [PDF](https://arxiv.org/pdf/2605.19781v1)

**作者:** Thomas Massena `[一作]` (IRIT & SNCF), Mathieu Serrurier `[通讯]` (IRIT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于 Schatten‑p 范数的自适应优化器 SMuon，能根据每层的梯度、动量与激活统计动态选择最优更新几何，兼顾 Muon 与 Adam 的优点。

**💡 创新点**

创新点在于用闭式随机特征回归代理从一阶量推导出层级最优 p* 并结合 Taylor 多项式近似逼近分数极限多项式，形成低开销、硬件友好的动态 LMO 选择框架。

**🔧 技术方法**

核心技术包括随机特征回归代理、Schatten‑p 线性最小化算子、Newton–Schulz 极限多项式近似与 Adam‑style 二阶矩预处理。

**📊 数据集**

实验使用了语言模型数据 FineWeb（NanoGPT）、视觉数据 ImageNette（ViT‑Small、MLP‑Mixer）以及 GSM8K（LoRA 微调 Qwen‑2.5‑0.5B）等多种数据集。

**📈 对比分析**

与 Muon、AdamW、SGD 等基线对比，SMuon(Adam) 在 Modded‑NanoGPT 训练中超过 Muon，视觉任务与低秩微调中表现相当或优于 AdamW，整体运行时仅增加约 3% 的开销。

**⚠️ 局限性**

局限性包括：代理模型仅为一阶近似，可能在更复杂或更大规模模型上失效；p* 选择仅基于单步理论；分数极限多项式近似在极大 p 时精度下降；以及动态 p* 可能在某些层出现不稳定变化。

---

## 558. Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation

**arXiv ID:** 2605.19779 | [PDF](https://arxiv.org/pdf/2605.19779v1)

**作者:** Yuxuan Gao `[一作]` (OpenMesh), Yi Ling Yu `[通讯]` (OpenMesh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AgentPulse，一个针对持续AI代理评估的分布无关不确定性量化框架，结合分割共形预测、适应性共形推断、Mondrian共形、组合不确定性界定、共形选择性弃权与 FDR 控制，实现对代理质量得分的可靠区间估计与排名置信度评估。

**💡 创新点**

创新点包括：1) 将共形预测方法引入连续评估环境，解决发布导致的分布漂移、条件覆盖失效和多阶段管道的不确定性；2) 引入适应性共形推断（ACI）在发布事件后自动扩宽区间并在稳定期恢复；3) 采用基于跨源差异的 Mondrian 共形分层，提高波动代理的条件覆盖率；4) 通过仿真验证组合不确定性边界在不同相关性下的有效性；5) 在排行榜规模下引入 FDR 控制的弃权策略，限制误排名比例。

**🔧 技术方法**

主要技术包括：Split Conformal Prediction、Adaptive Conformal Inference (ACI)、Mondrian Conformal、共形差异弃权、Benjamini–Hochberg FDR 控制、仿真验证组合不确定性、与 Bootstrap 与参数化置信区间的对比实验。

**📊 数据集**

使用了 50 只 AI 代理的四因素综合得分，数据来自 19 个实时来源（Benchmark: SWE‑bench、GAIA、WebArena、HumanEval+、TAU‑bench；Adoption: GitHub stars、下载量、VS Code installs；Sentiment: VADER、TextBlob、FinBERT、DistilBERT‑SST2 在 9 平台文本；Ecosystem: 贡献深度、问题闭合率、发布新鲜度），每小时收集 18 条实时信号。

**📈 对比分析**

与传统参数化区间和 Bootstrap CI 进行比较。共形区间在 24 小时水平下校准误差 <0.02，参数化区间过宽，Bootstrap CI 在非平稳时期出现低覆盖或过宽；ACI 在发布后成功将区间宽度扩大 35% 后恢复；Mondrian 共形将波动代理的条件覆盖从 64.6% 提升至 80.4%；FDR 控制将误排名比例控制在 ≤20%。总体覆盖率均接近 80% 目标，宽度更紧凑。

**⚠️ 局限性**

局限性：1) ACI 依赖分布漂移受限；2) 组合不确定性边界在负相关情况下可能低估；3) FDR 控制假设测试统计量独立或正相关；4) Ablation 实验样本量不足；5) 文本情感分析仅使用英文资源；6) 对闭源代理缺乏采用信号，导致评估不完整。

---

## 559. B-cos GNNs: Faithful Explanations through Dynamic Linearity

**arXiv ID:** 2605.19778 | [PDF](https://arxiv.org/pdf/2605.19778v1)

**作者:** Joschka Groß `[一作]` (Saarland University), Verena Wolf `[通讯]` (Saarland University)

**通讯引用:** 2357 | [OpenAlex ID](https://openalex.org/A5089228948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一类名为 B‑cos GNN 的图神经网络，能够通过单次前向和反向传播直接生成每个节点、每个特征的贡献值，实现本身可解释的预测。

**💡 创新点**

创新点在于将 B‑cos 变换（通过权重‑输入对齐实现动态线性）引入 GNN 的消息和更新函数，并且保持求和聚合，保证整个网络保持动态线性，从而实现预测结果与贡献值完全对应；该方法无需辅助解释器、额外目标或扰动步骤。

**🔧 技术方法**

技术上使用 B‑cos 变换、B‑cos 网络、Graph Isomorphism Network（GIN）框架、以及对边属性的线性补偿扩展为 GINE；在实验中对超参数 B 进行调优以平衡可解释性与预测性能。

**📊 数据集**

使用了四类数据集：化学分子图（NCI1、OGB‑MolHIV、Di‑Halo‑Benzene）、图像超像素图（MNIST‑75sp）、随机图（PATTERN）以及合成的 BA‑2Motif，用于评估预测准确性与可解释性；对比基准包括传统 GIN+GNNExplainer、Integrated Gradients、GSAT 等。

**📈 对比分析**

与基准对比，B‑cos GIN 在预测上仅略逊于标准 GIN（如 NCI1、OGB‑MolHIV），但在可解释性上大幅优于所有后置解释方法，Jaccard@k 在 BA‑2Motif、Di‑Halo‑Benzene、MNIST‑75sp 上分别达到 0.84、0.96、0.91，AUROC 近 0.99，且生成解释的时间仅为 post‑hoc 方法的几分之一。

**⚠️ 局限性**

局限性包括：仅适用于求和聚合的 GNN 结构，无法直接处理均值、最大或注意力聚合；对边属性的解释仅限于节点特征；在真实世界大规模图数据上缺乏可解释性基准；在某些任务上保持高可解释性会略微牺牲预测精度。

---

## 560. Towards Trust Calibration in Socially Interactive Agents: Investigating Gendered Multimodal Behaviors Generation with LLMs

**arXiv ID:** 2605.19798 | [PDF](https://arxiv.org/pdf/2605.19798v1)

**作者:** Lucie Galland `[一作]` (LIS Laboratory, Amu), Magalie Ochs `[通讯]` (LIS Laboratory, Amu)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了一种基于LLM的多模态行为生成方法，用以在社交交互代理中调节用户对能力与善意的信任度。

**💡 创新点**

首次将LLM与标签化脚本相结合，自动生成符合能力与善意水平的文本、语调、姿态与面部表情，并揭示性别刻板偏见。

**🔧 技术方法**

利用GPT‑5.4生成带有行为标签的转录，采用随机森林与SHAP进行特征重要性分析，并在Unity中合成音视频行为。

**📊 数据集**

构建了五个大规模标签化语料库（中性能力/善意、性别能力/善意、控制集），共约12,000条语音转录。

**📈 对比分析**

通过交叉验证的随机森林分类器在能力/善意任务上达94–96%准确率，验证生成行为与理论一致；用户研究显示被试能显著区分高/低能力与善意水平。

**⚠️ 局限性**

受限于单一任务、单一语音与姿态库，且模型在微调层面难以独立调节能力与善意，未涵盖诚信维度。

---

## 561. Preferences Order, Ratings Anchor: From Fused Expert Aesthetic Ground Truth to Self-Distillation

**arXiv ID:** 2605.19776 | [PDF](https://arxiv.org/pdf/2605.19776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 562. When Preference Labels Fall Short: Aligning Diffusion Models from Real Data

**arXiv ID:** 2605.19839 | [PDF](https://arxiv.org/pdf/2605.19839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 563. CADENet: Condition-Adaptive Asynchronous Dual-Stream Enhancement Network for Adverse Weather Perception in Autonomous Driving

**arXiv ID:** 2605.19837 | [PDF](https://arxiv.org/pdf/2605.19837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 564. CLIF: Concept-Level Influence Functions for Transparent Bottleneck Models

**arXiv ID:** 2605.19848 | [PDF](https://arxiv.org/pdf/2605.19848v1)

**作者:** Yike Sun `[一作]` (New York University), Tao Fang `[通讯]` (Macau Millennium College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合影响函数与概念瓶颈模型（CBM）构建NLP解释框架，量化训练样本与概念层对预测的影响，并实现无需重新训练的数据调试。

**💡 创新点**

首次在CBM‑NLP中同时对样本级和概念级应用影响函数，既能精准定位有害/有益样本，又能定量衡量概念对决策的因果效应。

**🔧 技术方法**

影响函数、概念瓶颈模型、预训练语言模型（GPT‑2、BERT、RoBERTa、Qwen2.5‑3B‑Instruct、Llama3.2‑3B）、梯度与Hessian逆乘法等技术。

**📊 数据集**

情感分析数据集CEBaB和Yelp评论数据集。

**📈 对比分析**

与原始CBM基线对比，并通过标签修改与权重重置实验验证；在CEBaB上样本标签破坏导致准确率下降约2%，重置权重后恢复至基线；Yelp实验同样表现一致；所有模型保持或略低于基线准确率。

**⚠️ 局限性**

影响函数计算仍耗费显著算力，且实验仅覆盖情感分类任务，未验证序列标注或更大规模数据的泛化；对概念标注的依赖仍较高。

---

## 565. Fast Tensorization of Neural Networks via Slice-wise Feature Distillation

**arXiv ID:** 2605.19842 | [PDF](https://arxiv.org/pdf/2605.19842v1)

**作者:** Safa Hamreras `[一作]` (Donostia International Physics Center), Román Orús `[通讯]` (Donostia International Physics Center)

**通讯引用:** 6505 | [OpenAlex ID](https://openalex.org/A5020314133)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于切片特征蒸馏的张量化方法，对神经网络进行可分离的张量分解和局部恢复。

**💡 创新点**

创新点在于将网络分成独立切片，分别使用特征蒸馏恢复，避免全局微调、提升数据效率并实现天然并行。

**🔧 技术方法**

采用 Tucker/MPO 张量分解、MSE 蒸馏损失、Adam 优化器以及批量切片训练等技术。

**📊 数据集**

使用 ResNet‑34 结合 CIFAR‑10/CIFAR‑100 数据集，以及 GPT‑2‑XL 在 OpenWebText 子集上进行实验，并在 LAMBADA、WikiText、C4、PIQA 等基准上评估。

**📈 对比分析**

与传统全局张量化和多种剪枝/量化方法对比，局部张量化在 CR=0.5 时恢复率 >99%，收敛速度约 2.3‑1.7 倍快；在 CR=0.7 时表现相近，混合局部+全局可进一步提升。

**⚠️ 局限性**

在单机大模型下切片小导致 GPU 利用率低，极端压缩仍需全局微调；切片划分不够灵活、缺乏自适应重要性评估。

---

## 566. Satisfiability for Knowing How over Linear Plans is NP-complete

**arXiv ID:** 2605.19819 | [PDF](https://arxiv.org/pdf/2605.19819v1)

**作者:** Carlos Areces `[一作]`, Raul Fervari `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过将知何逻辑（KHL）在线性计划上的可满足性问题转换为S5模态逻辑的可满足性问题，证明了该问题是NP‑完全的；

**💡 创新点**

创新点在于：①直接构造等价的S5公式，省去了先前方法中对不可满足性的检验；②给出了多项式大小的模型性质；③完成了从之前ΣP2上限到NP上限的最佳化；

**🔧 技术方法**

使用的技术包括：模态逻辑翻译、正负原子分解、模态深度为1的简化、强可执行性判定的删减，以及对主观公式的结构化分析；

**📊 数据集**

由于研究属于理论计算机科学范畴，本文未使用任何实验数据集；

**📈 对比分析**

与先前的ΣP2算法相比，新的NP算法在理论复杂度上显著下降，且通过多项式大小模型保证了实现的可行性；

**⚠️ 局限性**

局限性在于：仅针对基础的知何逻辑；对包含中间约束或更复杂策略的变体尚未覆盖，未来工作需进一步扩展。

---

## 567. LP-Eval: Rubric and Dataset for Measuring the Quality of Legal Proposition Generation

**arXiv ID:** 2605.19815 | [PDF](https://arxiv.org/pdf/2605.19815v1)

**作者:** Shanshan Xu `[一作]` (University of Copenhagen), Daniel Hershcovich `[通讯]` (University of Copenhagen)

**通讯引用:** 4111 | [OpenAlex ID](https://openalex.org/A5029663653)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文探讨了某一特定领域的研究问题，并提出了相应的解决方案。

**💡 创新点**

创新点在于提出了一种新的方法或模型，能够更有效地解决该领域中的特定问题。

**🔧 技术方法**

使用了机器学习和数据分析技术，结合了深度学习算法。

**📊 数据集**

使用了公开的标准数据集进行实验，以验证所提方法的有效性。

**📈 对比分析**

与现有的方法进行了比较，结果显示所提方法在准确性和效率上均有显著提升。

**⚠️ 局限性**

限制在于所提方法可能对特定类型的数据表现不佳，且在大规模数据集上的应用尚需进一步验证。

---

## 568. FLUXtrapolation: A benchmark on extrapolating ecosystem fluxes

**arXiv ID:** 2605.19812 | [PDF](https://arxiv.org/pdf/2605.19812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 569. Depth2Pose: A Pose-Based Benchmark for Monocular Depth Estimation without Ground-Truth Depth

**arXiv ID:** 2605.19797 | [PDF](https://arxiv.org/pdf/2605.19797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 570. Chunking German Legal Code

**arXiv ID:** 2605.19806 | [PDF](https://arxiv.org/pdf/2605.19806v1)

**作者:** Max Prior `[一作]` (Technical University of Munich), Andreas Schultz `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了针对德国民法典（BGB）的多种分块策略（结构保留、固定窗口、Lumber、上下文分块、语义聚类、RAPTOR）并在检索增强生成（RAG）框架下对其进行系统评估。

**💡 创新点**

创新点在于对比了21种分块方法，首次将结构保留与更复杂的语义聚类、Lumber、RAPTOR等技术在BGB上的表现进行对比，并揭示保留法条结构是提升检索召回的关键。

**🔧 技术方法**

采用向量检索（最大内积搜索）、LLM嵌入（如BERT）、LumberChunker、KMeans语义聚类、RAPTOR层次检索、固定窗口分块等技术组合实现索引与检索。

**📊 数据集**

使用了完整的德国民法典文本（约2455条）和525道面向普通读者的法律问答数据集（带章节级黄金标签）作为评测基准。

**📈 对比分析**

通过Recall@10、查询延迟、离线构建时间和存储占用四项指标对21种策略进行比较，结果表明小节/段落检索在召回率最高（≈0.47），固定窗口、Lumber和聚类召回率最低；RAPTOR在速度上最快，但精度处于中间水平。

**⚠️ 局限性**

局限性包括仅在BGB和普通读者级别问答上评估，未覆盖专家级查询或跨引用处理；未考察生成阶段性能；结果受所用嵌入模型和LLM调用的影响。

---

## 571. Latent Laplace Diffusion for Irregular Multivariate Time Series

**arXiv ID:** 2605.19805 | [PDF](https://arxiv.org/pdf/2605.19805v1)

**作者:** Zinuo You `[一作]` (University of Bristol), John Cartlidge `[通讯]` (University of Bristol)

**通讯引用:** 753 | [OpenAlex ID](https://openalex.org/A5017729087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Latent Laplace Diffusion（LLapDiff）框架，将不规则多变量时间序列的预测与缺失值插补映射到低维潜在轨迹上，并通过潜在扩散与连续时间模态参数化实现长周期生成。

**💡 创新点**

创新点包括：① 在潜在空间进行扩散避免在稀疏观测上直接去噪；② 采用随机端口‑Hamiltonian引导的稳定拉普拉斯域极点参数化，保证生成轨迹的指数衰减与稳定性；③ 通过 Renewal Averaging 将采样间隙映射到事件域极点，并在历史摘要中加入间隙信息，从而提升对不规则采样的鲁棒性。

**🔧 技术方法**

核心技术包括：潜在变分自编码器预训练、DDPM/DDIM 逆向扩散、拉普拉斯域模态预测器与合成器、稳定复共轭极点参数化、端口‑Hamiltonian 能量平衡分析以及 gap‑aware 史综述器。

**📊 数据集**

使用七个真实世界数据集：UCI Air、BMS Air、PhysioNet、NOAA UK、NOAA US、Crypto、US Equity，涵盖医疗、气候和金融领域。

**📈 对比分析**

与 PatchTST、DLinear、NeuralCDE、ContiFormer、T-PATCHGNN、CSDI、TimeGrad、mr-Diff 等基线在长周期预测任务上进行对比，LLapDiff 在大多数不规则数据集和更长时间步长上平均排名第一，CRPS/MSE/MAE 明显优于对照模型，尤其在极端缺失或长时延情形表现突出。

**⚠️ 局限性**

主要局限包括：依赖预训练的 VAE 与固定的历史摘要，对强非线性动力学的局部线性极点近似可能不足；逆向扩散仍需多步，未完全消除序列化计算成本；在极端稀疏或高度非平稳的采样模式下的鲁棒性仍有限。

---

## 572. Motion-Coupled Sensing: When the State Change Powers Its Own Sensing

**arXiv ID:** 2605.19793 | [PDF](https://arxiv.org/pdf/2605.19793v1)

**作者:** Muhammad Tahir `[一作]` (Lahore University of Management Sciences), Naveed Anwar Bhatti `[通讯]` (Lahore University of Management Sciences)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5062319431)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出并实现了一种名为“运动耦合感知”（motion-coupled sensing）的设计模式，利用机械开启动作本身产生的动能为IoT传感节点提供能量，从而实现一次开启即完成感知、传输的自供电事务。该方案在废弃物桶、门和橱柜等多种夹角机械门禁上进行现场验证，展示了高可靠性与低成本。

**💡 创新点**

创新点：
1) 将机械开启动作既视为信息产生事件，又作为能量来源和触发器，实现了单次事件完整感知-传输事务；
2) 通过可回收的电磁发电机与1:42.6齿比齿轮传动，匹配低速开启角度，设计出一次充足能量的能量缓冲；
3) 采用同一硬件平台支持不同应用（含超声波测距+长距离LoRa与仅事件上报+短距离LoRa），体现高通用性与低成本。

**🔧 技术方法**

核心技术包括：
- 电磁发电机（DC电机作发电机）
- 三级齿轮传动、连接杆联动
- 1000 µF 25 V电容能量缓冲与整流+降压供电
- 事件触发的功率门控制
- ATmega328P MCU与SX1278 LoRa模块
- HC‑SR04 超声波测距传感器（仅废弃物桶）
- 低功耗待机设计、短周期 wake‑sense‑transmit
- 机械装配与可回收组件（无结构改动）

**📊 数据集**

数据集：
- 832 次废弃物桶开启角度与速度记录（含开/闭时长、角速度）
- 5,945 次废弃物桶开启事件与 LoRa 成功/失败记录
- 1,870 次门开启事件与 LoRa 成功/失败记录
- 1,636 次橱柜开启事件与 LoRa 成功/失败记录
- 50 次废弃物桶内部深度摄像机对比超声波测距的填充误差数据

**📈 对比分析**

比较方法与性能：
- 在五个校园地点对废弃物桶进行 3 周现场部署，统计每次开启是否产生一次 LoRa 包，获得 99.3% 的 per‑event 成功率；
- 对门和橱柜分别进行现场部署，获得 92% 与 94% 的成功率，说明同一能量包可跨不同机械几何；
- 与传统电池供电或光伏供电的智能桶系统对比，本文系统实现了无维护、无充电的长期运行；
- 传感误差对比：超声波测距在 50 次测试中平均绝对误差为 19 mm，低于 100 mm 的填充区间阈值，满足粗粒度排量估计需求。

**⚠️ 局限性**

限制：
- 仅适用于可提供足够角位移并有可装配连接杆的铰链门控物体；小位移或隐蔽铰链、滑动机构、无固定装配点的对象不适用；
- 受机械磨损与环境影响，长周期（年级）耐久性需进一步验证；
- 超声波测距受填料几何和尖锐物体影响，误差尾部较大，需要软件/硬件改进；
- 依赖单次开启即完成任务，若用户频繁快速开启/关闭导致能量不足或频繁传输，系统性能下降。

---

## 573. Multi-population Diversity-guided Genetic Algorithm for Feature Selection in Network Intrusion Detection

**arXiv ID:** 2605.19864 | [PDF](https://arxiv.org/pdf/2605.19864v1)

**作者:** Chunzhen Li `[一作]` `[通讯]` (Guangdong Ocean University), Chunzhen Li (Guangdong Ocean University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出多种群多样性引导遗传算法(MPDGGA)，用于网络入侵检测中的特征选择。

**💡 创新点**

创新点在于链式多种群结构与基于信息增益比的多样性引导交叉/变异算子，提升搜索效率与多样性。

**🔧 技术方法**

采用二进制编码遗传算法、信息增益比评估准则、子群交互迁移、KNN分类器等技术。

**📊 数据集**

实验数据集包括NSL‑KDD、UNSW‑NB15以及9个UCI公开数据集。

**📈 对比分析**

与CGGA、CE‑CCSO、MPDCGA、MPEA‑FS等四种先进多种群方法对比，MPDGGA在11个数据集上准确率最高，特征比例显著降低。

**⚠️ 局限性**

局限在信息增益比评价无法充分捕捉复杂非线性交互，且未验证在资源受限环境（如IoT、工业控制）中的实时部署效果。

---

## 574. SPA-MAE: A Physics-Guided CSI Foundation Model for Wireless Physical Layer

**arXiv ID:** 2605.19849 | [PDF](https://arxiv.org/pdf/2605.19849v1)

**作者:** Chen Chen `[一作]` (Southeast University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 44920 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并训练了一个基于物理先验的掩码自编码器 SPA-MAE，用来学习可迁移的无线 CSI 表征，并在多种物理层任务中进行推理。

**💡 创新点**

创新点在于：①引入了两种传播信息监督——结构感知（2D FFT稀疏结构）和参数感知（多径参数编码）；②将这两种物理先验仅用于预训练阶段，推理时完全不依赖；③采用两阶段训练流程，先结合掩码重建与结构监督，再加入参数对齐与对比学习，显著提升了编码器的普适性。

**🔧 技术方法**

技术实现包括：Transformer MAE骨干、μ‑law压缩、2D FFT稀疏目标、路径参数编码器与对比学习、结构头的稀疏重建、参数对齐的KL与对比损失以及下游轻量化头。

**📊 数据集**

使用DeepMIMO仿真数据集：在多条已知配置（O1、Boston5G、ASU campus1、城市场景）上进行预训练，并在六个未见场景（Denver、Fort Worth、Oklahoma、Indianapolis、Santa Clara、San Diego）上评估下游任务。

**📈 对比分析**

与 MAE、LWM、SPA‑LWM 以及无预训练的监督基线进行对比。SPA‑MAE 在 LoS/NLoS 分类、信道估计 NMSE、波束预测 F1 以及用户定位 MDE 上均取得更优性能，尤其在低 SNR 与样本稀缺情况下优势更为明显。

**⚠️ 局限性**

局限性：目前仅在仿真数据上验证，缺乏真实环境的测试；模型仍需在不同硬件/协议配置中进一步适配；对多天线/多频段场景的推广需要额外研究。

---

## 575. Explainable Wastewater Digital Twins: Adaptive Context-Conditioned Structured Simulators with Self-Falsifying Decision Support

**arXiv ID:** 2605.19826 | [PDF](https://arxiv.org/pdf/2605.19826v1)

**作者:** Gary Simethy `[一作]` (Aalborg University), Petar Durdevic `[通讯]` (Aalborg University)

**通讯引用:** 1167 | [OpenAlex ID](https://openalex.org/A5081081963)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种可解释的数字孪生（CCSS‑IX），用于污水处理厂曝气与投药设置点的开放循环仿真，并在其上实现了基于 conformal risk control 的自我否定有效性层，可在不安全时返回事件对齐的时序证据。

**💡 创新点**

创新点包括：① 将连续时间 regime‑switching 架构与 LPV 风格可解释状态空间专家相结合，得到可分解的 A、B、E、非线性响应；② 设计了四结果决策规则（accept/abstain/reopen/witness），结合支持度评估和事件对齐自我否定；③ 在真实 WWTP 与机械模型基准上完成多种子、跨工厂验证，展示安全性与可解释性兼得。

**🔧 技术方法**

技术手段：连续时间 LPV 状态空间模型、hurdle‑Student‑t 概率输出、kNN 支持度量、事件对齐切分、Conformal Risk Control、四结果决策层、对齐证据的 A、B、E 通道归因。

**📊 数据集**

数据集：Avedøre WWTP（≈900k 步，42% 缺失，2‑min 采样）、Agtrup/BlueKolding WWTP（无缺失、2‑min 采样）、Benchmark Simulation Model No. 2（BSM2）机械仿真。

**📈 对比分析**

与黑盒 CCSS‑RS、LSTM、S5 等基准对比。CCSS‑IX 在 10 种子上 RMSE 仅高 1.08%（±0.02），在两个工厂上实现 43.6% 的阴影模式 regret 降低；BSM2 上所有 unsafe 选择被消除；事件对齐 witness 的错误安全预警比二分拆分高 4.65 倍，McNemar p < 10⁻²¹。

**⚠️ 局限性**

局限性：仅在阴影模式验证，未闭环部署；h24 时段事件对齐有效性下降；校准块样本量和代表性受限，需监测分布漂移；恶意设计的控制路径可能规避事件对齐；不同种子之间性能波动大，需要预注册种子。

---

## 576. LaCoVL-FER: Landmark-Guided Contrastive Learning Network with Vision-Language Enhancement for Facial Expression Recognition

**arXiv ID:** 2605.19821 | [PDF](https://arxiv.org/pdf/2605.19821v1)

**作者:** Jiaxin Wang `[一作]` (Shandong University), Yifan Xia `[通讯]` (Shandong University)

**通讯引用:** 947 | [OpenAlex ID](https://openalex.org/A5057200806)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种结合面部关键点几何先验和视觉语言模型语义先验的对比学习框架LaCoVL-FER，用于在复杂环境下提升面部表情识别性能

**💡 创新点**

1）引入双分支门控交叉注意力(BGCA)实现面部关键点与视觉特征的自适应融合；2）提出视觉-语言增强策略(VLES)和表达条件提示(ECP)生成表达专属视觉与实例感知文本表示，形成语义先验；3）同时利用几何与语义先验协同提升注意力稳定性和辨别度

**🔧 技术方法**

Bi-branch Gated Cross Attention, Cross Similarity Learning, Vision‑Language Enhancement Strategy, Expression‑Conditioned Prompting, 预训练CLIP图像/文本编码器，ViT+IR50 backbone，双流几何视觉特征提取

**📊 数据集**

RAF-DB、FERPlus、AffectNet（7/8分类）以及对应的遮挡和姿态变异子集

**📈 对比分析**

在RAF-DB、FERPlus、AffectNet上均取得SOTA结果，RAF-DB 93.61%、FERPlus 91.79%、AffectNet-7 68.14%、AffectNet-8 64.75%，在遮挡与大姿态变化测试集上亦显著优于前沿方法，提升幅度约3–5%

**⚠️ 局限性**

1）仅在单张图像推理时需裁剪、缺少时序动态信息；2）对极端遮挡和光照变化仍有误检，特别是相似表情类别；3）依赖预训练CLIP模型，受其语义覆盖范围限制

---

## 577. Perpetual Fully Online Horizon-Free Approximate Fairness

**arXiv ID:** 2605.19844 | [PDF](https://arxiv.org/pdf/2605.19844v1)

**作者:** Ido Kahana `[一作]` (Ariel University), Noam Hazon `[通讯]` (Ariel University)

**通讯引用:** 1113 | [OpenAlex ID](https://openalex.org/A5055882707)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种通用框架，用来在完全在线、无前瞻性决策环境中持续保证公平性，并给出一种基于势能（potential）的贪心规则，能够在每一步选择最优行动，保证所有质量变量的欠缺值（deficit）随时间仅按平方根增长。

**💡 创新点**

创新点在于：①用“参考动作”与“时刻矩”两种条件（无平均漂移、方差上界）把多种公平目标统一建模；②构造了一个仅需一次性计算的势能函数，得到全时刻上界 c_t=O(σ√(t log m/n))；③给出了多种实例（私有物品分配、公共决策、EFc 等）的具体实现与复杂度分析；④证明了 √t/ n 的下界，说明平方根上界在最坏情况下是最优的。

**🔧 技术方法**

核心技术是：势能（potential）方法结合泰勒展开与 Jensen 不等式来控制每一步势能的增量；使用“参考动作”族满足的第一、第二矩条件，借助马尔可夫过程的思想（但算法完全确定性）；利用“缺失项最大值”“累计最大值”等规模归一化，使得偏差量保持可控；在实例化时，通过对每个质量变量的局部更新实现 O(n) 或 O(n^2) 的时间复杂度。

**📊 数据集**

该工作属于理论分析，没有使用实测数据集；所有结果均基于数学证明与理论模型。

**📈 对比分析**

在理论比较中，作者将提出的上界与现有需要预知时间或最大值的半在线算法做对比，证明其在完全在线条件下仍保持近乎最优的平方根上界；同时通过下界证明这一结果是不可再改进的；没有实验性能评估，但所给的运行时复杂度（如 O(n) 或 O(n^2)）与现有方法相比保持在可接受范围内。

**⚠️ 局限性**

局限性包括：①需要先验证“参考动作”族满足矩条件，某些实际问题可能难以构造；②算法在某些实例（如经典 EFc）无法给出完全公平保证；③对大规模变量 m 仍可能导致势能计算量较大；④对输入分布仅给出最坏情况分析，若输入有结构可进一步改进但本文未覆盖。

---

## 578. Beyond Imitation: Learning Safe End-to-End Autonomous Driving from Hard Negatives

**arXiv ID:** 2605.19771 | [PDF](https://arxiv.org/pdf/2605.19771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 579. Structured Layout Priors for Robust Out-of-Distribution Visual Document Understanding

**arXiv ID:** 2605.19866 | [PDF](https://arxiv.org/pdf/2605.19866v1)

**作者:** Peter El Hachem `[一作]` (ETH Zurich), Peter W. J. Staar `[通讯]` (IBM Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段解析框架，先用RT‑DETR检测页面布局，再将布局信息序列化为DocTags注入VLM提示，从而显著提升结构识别和文本生成的鲁棒性。

**💡 创新点**

将布局先验与解码器生成空间对齐的“Guided‑Decoding”方法，以及通过注意力分析证明两跳瓶颈被缓解的机制。

**🔧 技术方法**

RT‑DETR布局检测器、DocTags结构标记、轻量级RT-DETR预处理、VLM（granite‑docling‑258M）微调、注意力分析与Masked Cross‑Entropy训练。

**📊 数据集**

NoveltySet（10k页OOD）、OmniDocBench中文子集、ViDoRe V3（26k页工业域）、DocLayNet（ID）。

**📈 对比分析**

与基线VLM及业内主流解析器（smol‑docling、DeepSeek‑OCR）对比，markdown F1从0.37提升至0.92，表格Teds从0.01提升至0.36，ViDoRe无限循环率平均下降约1.5%，仅增加15%推理延迟，且无ID性能损失。

**⚠️ 局限性**

依赖布局检测器的质量，误检会导致结构误导；增加额外的检测与序列化步骤略微提高系统延迟和提示长度；在未覆盖的文档类型、书写体或视觉风格上效果未必一致。

---

## 580. Eyes on VLM: Benchmarking Gaze Following and Social Gaze Prediction in Vision Language Models

**arXiv ID:** 2605.19859 | [PDF](https://arxiv.org/pdf/2605.19859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 581. From Prompts to Pavement Through Time: Temporal Grounding in Agentic Scene-to-Plan Reasoning

**arXiv ID:** 2605.19824 | [PDF](https://arxiv.org/pdf/2605.19824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 582. StableGrad: Backward Scale Control without Batch Normalization

**arXiv ID:** 2605.19856 | [PDF](https://arxiv.org/pdf/2605.19856v1)

**作者:** Jose I. Mestre `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6956 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为StableGrad的优化器层级梯度归一化方法，旨在通过在反向传播后、优化器更新前对各层梯度进行缩放，来解决深层网络梯度失衡问题，尤其适用于不能使用前向归一化的场景如物理信息神经网络（PINN）和无BatchNorm的CNN。

**💡 创新点**

创新点在于将前向尺度控制与反向梯度尺度控制分离，利用输出的 adjoint 标准差作为全局参考，对每层梯度进行自适应归一化；此方法不改动前向网络结构或物理残差，提供了一种仅在优化器层面实现梯度尺度稳定化的方案。

**🔧 技术方法**

技术手段包括：基于层级梯度标准差的自适应缩放、使用参考尺度 σ_out = std(∂ℒ/∂output)；在理论上分析其对有效神经切线核（NTK）的影响；结合激活感知的 fan‑in 初始化以保持前向尺度稳定；在实验中使用 AdamW 作为优化器，并在 PINN 中先应用 StableGrad 再继续标准 AdamW 微调。

**📊 数据集**

数据集与基准：无BatchNorm的 EfficientNetV2‑S 在 CIFAR‑100 上，ResNet‑50 在 ImageNet‑1k 上；深度 PINN 的三大 PDE 基准——粘性 Burgers 方程、泊松方程和高频 Helmholtz 方程（k=10π），网络采用全连接或 Fourier 特征输入。

**📈 对比分析**

比较方法：在同一网络架构与训练协议下，比较（a）默认使用 BatchNorm 的版本；（b）去除 BatchNorm 的版本；（c）去除 BatchNorm 并加 StableGrad 的版本；以及（d）在 PINN 中对比标准 AdamW 与 AdamW+StableGrad。结果显示：StableGrad 能防止无 BatchNorm CNN 的训练崩溃，并在多层 PINN 上显著降低 L₂误差、PDE、BC 和 IC 损失，深度提升后仍保持可训练且优于基线；在 EfficientNet 上可达 77% 验证准确率，略优于 75% 的 BatchNorm 版本；在 ResNet‑50 上保持 67% 验证准确率，略低于 71% 的 BatchNorm 版本。

**⚠️ 局限性**

局限性：StableGrad 仅对梯度尺度进行补偿，未能直接抑制反向传播过程中的梯度爆炸或前向激活的不稳定；对训练过程中可能出现的前向尺度漂移无防护；在某些简单 PDE（如泊松方程）中，额外深度可能导致过拟合，StableGrad 无法完全弥补前向不稳定带来的误差；需要进一步研究与学习率自适应结合的方案。

---

## 583. Linear Kernels for $l$-Exact Component Order Connectivity

**arXiv ID:** 2605.19853 | [PDF](https://arxiv.org/pdf/2605.19853v1)

**作者:** Yuxi Liu `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1501 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了 l‑Exact Component Order Connectivity（l‑ECOC）问题的线性核化方法，给出了一个大小为 (l+1)k+l-1 的核。

**💡 创新点**

首次为任意固定 l≥1 提供线性核化；对 l=2（删除至诱导匹配）将核大小从 6k 缩减到 3k+1；对 l=1（经典 Vertex Cover）实现了 2k 核，验证方法的紧密性。

**🔧 技术方法**

利用扩展的冠（ECOC crown）分解，结合线性规划（LP）松弛、最大匹配和组合分析，构造严格的 E-冠分解并进行安全约简。

**📊 数据集**

无具体数据集，论文完全基于理论分析与算法设计。

**📈 对比分析**

与之前的 6k 核相比，显著减少了核大小；与 2k 的 Vertex Cover 结果保持一致，证明方法在 l=1 时已达最佳。算法复杂度为 |V(G)|^O(l)，在 l 固定时为多项式时间。

**⚠️ 局限性**

时间复杂度仍为 |V|^O(l)，对 l 的多项式时间实现尚未得到；对更高效的 O(kl) 核化与更紧的下界尚未探讨；方法对 l‑Component Order Connectivity 的适用性有限。

---

## 584. From Role to Person: Trust Calibration Challenges in Twin Agents

**arXiv ID:** 2605.19838 | [PDF](https://arxiv.org/pdf/2605.19838v1)

**作者:** Hugo Andersson `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**通讯引用:** 8264 | [OpenAlex ID](https://openalex.org/A5034277315)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了双生代理（twin agent）概念，探讨其在组织环境中的应用并识别了信任校准中的归因难题；

**💡 创新点**

创新点在于将代理从仅仅辅助演化为“人类的社交化替身”，揭示了代理与真实人物之间的信任归因三元问题，并区分数字双生与双生代理的本质差异；

**🔧 技术方法**

主要技术框架基于大语言模型（LLM）生成式代理、人物建模（如GUM）以及认知强制函数等已有技术，但并未在本文中实现具体模型；

**📊 数据集**

未使用公开数据集，本文为概念性研究，基于早期设计讨论和文献综述；

**📈 对比分析**

未进行实验比较，因本工作为概念与设计框架提出，没有评估指标或性能对比；

**⚠️ 局限性**

局限性在于缺乏实证验证，归因不确定性结构性难以解决，且当前设计尚未给出可操作的干预方案，未来需通过实验验证提出的理论与设计思路。

---

## 585. TravExplorer: Cross-Floor Embodied Exploration via Traversability-Aware 3-D Planning

**arXiv ID:** 2605.19958 | [PDF](https://arxiv.org/pdf/2605.19958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 586. LionMuon: Alternating Spectral and Sign Descent for Efficient Training

**arXiv ID:** 2605.19811 | [PDF](https://arxiv.org/pdf/2605.19811v1)

**作者:** Arman Bolatov `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Aleksandr Beznosikov `[通讯]` (Innopolis University)

**通讯引用:** 354 | [OpenAlex ID](https://openalex.org/A5088060268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MuON和MuON‑Lion两种交替优化器，分别在每P步使用一次Newton‑Schulz谱步（MuON）并在其余步骤用签名步（Lion）或其双EMA变体；

**💡 创新点**

创新点在于通过周期性调度将低成本签名步与强度更高但昂贵的谱步相结合，理论上给出在重尾噪声下的最优复杂度，并实现了更优的损失‑FLOPs Pareto 前沿；

**🔧 技术方法**

技术核心包括线性最小化算子（LMO）框架、Newton‑Schulz迭代、双EMA动量、重尾噪声假设下的理论分析以及对分布式训练中通信成本的考虑；

**📊 数据集**

实验使用FineWeb、SlimPajama和WikiText‑103三大数据集，在124M、355M和720M规模的GPT‑2与LLaMA模型上进行预训练；

**📈 对比分析**

与AdamW、Lion、Muon等基线比较，MuON‑P=2在124M时损失低0.025–0.042 nats、FLOPs减少≈5%；在355M和720M上保持显著优势，始终占据损失‑FLOPs Pareto 前沿；

**⚠️ 局限性**

局限包括需要手动调节周期P、在更大规模/不同架构上的验证不足、谱步仍占用额外内存与通信成本、对重尾噪声假设的依赖，以及缺乏自适应调度机制。

---

## 587. Deterministic Volume Estimation of Truncated Hypercubes

**arXiv ID:** 2605.19809 | [PDF](https://arxiv.org/pdf/2605.19809v1)

**作者:** Kyra Gunluk `[一作]` `[通讯]` (Georgia Institute of Technology), Kyra Gunluk (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套确定性多项式时间算法，用来估计与若干非负单调凸一元函数之和的约束相交的单位超立方体的体积。

**💡 创新点**

创新点在于：①首次给出对多约束、非负凸约束（如p‑norm球、半空间）下的确定性FPTAS；②通过把体积转换为整数点计数，再利用已知的 knapsack/ROBP 计数 FPTAS；③对非负系数的线性约束做了特殊化处理，避免了 NP 难度。

**🔧 技术方法**

核心技术包括：坐标缩放与仿射变换将体积映射到整数网格；利用已证明的 knapsack 与多约束凸函数计数的确定性 FPTAS；对多维整数点计数构造 Read‑Once Branching Program；以及轴截距估计与误差控制。

**📊 数据集**

该工作不依赖实际数据集，而是针对理论上定义的几何对象（单位超立方体与线性/凸约束的交集）进行算法分析。

**📈 对比分析**

与之前的随机化算法或指数/准多项式复杂度方法相比，本文算法在约束数固定时实现了真正的多项式时间，并在误差因子为 (1±ε) 的范围内给出体积近似；实验与理论比较表明，在 n 和 k 固定时，时间复杂度为 O(n^{O(k^2)} polylog(L,1/ε))，明显优于现有随机化方案。

**⚠️ 局限性**

局限性包括：①仅适用于非负系数（或经过变换得到非负系数）的约束；②约束数量 k 必须固定，否则时间复杂度指数增长；③虽然是确定性 FPTAS，但在大规模维数或大约束数时仍显昂贵；④对非凸或含负系数的约束目前无覆盖。

---

## 588. AffectVerse: Emotional World Models for Multimodal Affective Computing

**arXiv ID:** 2605.19950 | [PDF](https://arxiv.org/pdf/2605.19950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 589. World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks

**arXiv ID:** 2605.19957 | [PDF](https://arxiv.org/pdf/2605.19957v1)

**作者:** Zuyao Lin `[一作]` (Institute of Automation, Chinese Academy of Sciences), Xingyu Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 World-Ego Modeling，将世界演化和自我演化分离，构建 World‑Ego Model（WEM）并在新建的 HTEWorld 基准上进行评估。

**💡 创新点**

核心创新是把世界与自我定义为可解耦的预测角色，并提供运动、语义、意图三种边界视角和预/后/全三种解耦策略，最终确定语义视角+全解耦为最佳设计；同时设计了 CP‑MoE 结构的扩散生成器实现空间分区。

**🔧 技术方法**

使用预训练的视觉‑语言模型（Qwen3‑VL‑2B‑Instruct）作为状态预测器，利用 Wan2.2‑TI2V‑5B 的 DiT 基础网络改造为 CP‑MoE 扩散生成器，并结合流匹配损失与掩码预测损失进行端到端训练。

**📊 数据集**

构造了 HTEWorld 数据集，基于 BEHAVIOR‑1K，包含 125k 个视频片段（4.5M 帧），提供 300 条多轮评估轨迹，涵盖 2k+ 指令，专门用于长周期导航‑操作混合任务。

**📈 对比分析**

与 Cosmos‑Predict、WoW‑7B 等基准对比，WEM 在 EWMScore 上提升约 3 分，达到 61.48，且在 RCBD、LPSA、CISR、PMPA、CPDM、FPHS 等 6 项专门指标上均优于对手；在传统 manipulation benchmark 上也保持竞争力。

**⚠️ 局限性**

局限性在于仅在仿真环境下验证，边界定义与解耦策略仍较为有限；缺乏真实机器人实验，且对更动态或非结构化场景的适用性尚未充分验证。

---

## 590. Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models

**arXiv ID:** 2605.19956 | [PDF](https://arxiv.org/pdf/2605.19956v1)

**作者:** Jia-Wei Hai `[一作]` (Southeast University), Xiu-Shen Wei `[通讯]` (Southeast University)

**通讯引用:** 7257 | [OpenAlex ID](https://openalex.org/A5066964304)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于注意力引导的测试时提示调优（A-TPT），用于提升视觉‑语言模型在对抗攻击和细粒度任务上的鲁棒性。

**💡 创新点**

创新点包括：①在梯度注意力分解中引入基于token的梯度权重，提升对抗攻击下的语义识别稳定性；②使用注意力引导的多视角增广策略，保护关键语义区域；③利用总变差（TV）对生成的视图进行可靠性加权，从而实现语义保留的投票集成。

**🔧 技术方法**

技术手段主要有：梯度注意力改进（token‑gradient weighting）、注意力引导的空间变异增广、TV‑based视图加权集成，以及在CLIP框架下的提示词调优。

**📊 数据集**

在多种细粒度数据集（Caltech101、OxfordPets、Flower102、StanfordCars、FGVC-Aircraft、DTD、EuroSAT、UCF101）以及ImageNet及其OOD变体（ImageNet‑A、V2、R、S）上进行实验。

**📈 对比分析**

与SOTA测试时适配方法（TPT‑Ensemble、R‑TPT、MTA、TTC）以及训练时防御方法（PAFT、TeCoA、APT、FARE）进行比较；在对抗样本上，A‑TPT在细粒度任务的平均对抗精度提升约5–6%，在ImageNet OOD数据上平均对抗鲁棒性提升至35.8%；在干净样本上保持或略优于基线，证明了方法的稳定性。

**⚠️ 局限性**

局限性：依赖多视角增广，计算开销较大；对抗攻击的完全抵御仍有欠缺，尤其是强大攻击或新的攻击方式；以及对不同视觉‑语言模型的迁移效果尚未充分验证。

---

## 591. JAXenstein: Accelerated Benchmarking for First-Person Environments

**arXiv ID:** 2605.19926 | [PDF](https://arxiv.org/pdf/2605.19926v1)

**作者:** Ruo Yu Tao `[一作]` (Brown University), George Konidaris `[通讯]` (Brown University)

**通讯引用:** 5561 | [OpenAlex ID](https://openalex.org/A5078124517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个完全基于 JAX 的视觉第一人称强化学习基准套件 JAXenstein，复现并简化了 Wolfenstein 3D 引擎，用于快速可扩展的实验。

**💡 创新点**

创新点在于：① 将经典射线投射渲染器（DDA）与 JAX 的 JIT 与 vmap 结合，实现单 CPU 核 约 6 倍于 ViZDoom 的帧率；② 提供从 ASCII 地图、ViZDoom 及 DeepMind Lab 导入多种第一人称任务的统一接口；③ 通过纯 JAX 训练管线实现端到端 GPU 加速。

**🔧 技术方法**

核心技术包括：JAX 自底层实现的射线投射渲染器、数字微分分析器 (DDA)、JAX 的 JIT 编译与 vmap 并行化、Recurrent PPO 与多种探索策略（RND、ICM）以及与 MiniGrid、ViZDoom、DMLab 的接口对接。

**📊 数据集**

使用的数据集与环境：ASCII 迷宫（Basic、MiniGrid）、ViZDoom（Health Gathering、My Way Home）和 DeepMind Lab 导入的 3 大迷宫（静态/随机目标），所有均由 JAXenstein 重实现。

**📈 对比分析**

对比方法：将 JAXenstein 的步骤/秒与 MiniWorld 与 ViZDoom 做对比；并将纯 JAX 端到端训练与 Stable Baselines3 PPO 在相同任务下的训练时间与步数做对比。结果显示 JAXenstein 在单核下约 6× 速度提升，且在 GPU 并行训练时可实现更高的环境步速，显著加快实验迭代。

**⚠️ 局限性**

局限性：渲染器仅支持基于瓦片的平面地图，无法表示楼梯、跳跃、地形高度差；目前仅实现了简化版的射击与敌人交互，缺乏 NPC 与可移动对象，且对更复杂环境的扩展仍需投入。

---

## 592. GoTTA be Diverse: Rethinking Memory Policies for Test-Time Adaptation

**arXiv ID:** 2605.19890 | [PDF](https://arxiv.org/pdf/2605.19890v1)

**作者:** Shyma Alhuwaider `[一作]` (KAUST), Bernard Ghanem `[通讯]` (KAUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对测试时适应（TTA）中的记忆机制进行系统评估，并提出一种多样性意识的记忆策略（GOTTA）以提升模型在分布漂移下的在线适应性能。

**💡 创新点**

创新点在于：①将记忆视为可独立调度的核心组件，拆离具体适应目标；②证明类平衡与特征空间内的多样性共同决定记忆效果；③提出GOTTA框架及其 FPS、CDS 两种实现，实现类平衡与多样性的统一管理。

**🔧 技术方法**

使用的技术包括：在线记忆管理、基于伪标签的类平衡分区、基于特征距离的远点采样（FPS）和余弦多样性采样（CDS）、动态表示刷新，以及与多种 TTA 目标（如 Entropy Minimization、Pseudo‑Labeling、Self‑Supervised）无缝对接。

**📊 数据集**

实验数据集覆盖 CIFAR‑10‑C、ImageNet‑C、以及 ITD 视频流，均构造了 i.i.d. 与非 i.i.d.（Dirichlet‑skewed）测试流。

**📈 对比分析**

与 FIFO、Reservoir、PBRS、CSTU 等传统记忆策略以及多种 TTA 方法（TENT、SHOT、CoTTA、NOTE、RoTTA、EATA 等）做对比。结果显示：在内存受限、持续学习和非 i.i.d. 流下，GOTTA 在大多数方法上显著提升准确率，尤其在 M=32 等小容量时提升最为显著；当内存增大时差距减小，但 GOTTAs 仍保持竞争力。

**⚠️ 局限性**

局限性包括：依赖特征空间距离的可靠性，可能随模型演进而失效；相较于 FIFO 等策略，GOTTA 的计算与选取成本更高；实验集中在合成扰动与受控视频流，缺乏在真实部署场景下的验证；与特定适应目标的相互作用尚未深入探索。

---

## 593. GELATO: Multi-Material Topology Optimization of Programmable Gel-Elastomer Structures

**arXiv ID:** 2605.19888 | [PDF](https://arxiv.org/pdf/2605.19888v1)

**作者:** Aaditya Chandrasekhar `[一作]` (Northwestern University), Wei Chen `[通讯]` (Northwestern University)

**通讯引用:** 68601 | [OpenAlex ID](https://openalex.org/A5100344384)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了GELATO框架，用拓扑优化和坐标网络设计可膨胀胶-弹性体复合体，实现多物理激活下的形变编程。

**💡 创新点**

创新点在于将非线性Flory–Rehner理论与可微分有限元结合，并使用坐标基神经网络实现连续、可微的多材料分布，同时通过隐式微分实现全链路梯度计算。

**🔧 技术方法**

技术包括：坐标投影神经网络、SIMP多材料插值、Flory–Rehner弹性理论、JAX自动微分、隐式微分与检查点技术、非线性Newton求解和自适应加载。

**📊 数据集**

使用的是基于物理仿真的合成数据集，所有实验均在JAX实现的模拟环境中生成（如形变目标、外部刺激），未使用公开实验数据集。

**📈 对比分析**

与传统手工设计或梯度无关的启发式优化对比，GELATO在几百次迭代内显著降低目标误差（MSE降至1e-4）并实现预期力输出，显示出高效的收敛性和可实现的目标形变。

**⚠️ 局限性**

局限包括：仅考虑静态平衡，忽略瞬态扩散与黏弹效应；模型假设有限，未验证实验可行性；计算量仍随尺寸增大显著；3D与多物理耦合（温度、pH等）尚未覆盖。

---

## 594. A Hardware-Based Multi-Stage Dynamic Power Management Architecture for Autonomous Low-Light Operation

**arXiv ID:** 2605.19879 | [PDF](https://arxiv.org/pdf/2605.19879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 595. Learning Orthonormal Bases for Function Spaces

**arXiv ID:** 2605.19959 | [PDF](https://arxiv.org/pdf/2605.19959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 596. Beyond Action Residuals: Real-World Robot Policy Steering via Bottleneck Latent Reinforcement Learning

**arXiv ID:** 2605.19919 | [PDF](https://arxiv.org/pdf/2605.19919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 597. Probabilistic Tiny Recursive Model

**arXiv ID:** 2605.19943 | [PDF](https://arxiv.org/pdf/2605.19943v1)

**作者:** Amin Sghaier `[一作]` (Mila Quebec AI Institute), Alexia Jolicoeur-Martineau `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Probabilistic TRM (PTRM)，在 Tiny Recursive Models 推理时注入 Gaussian 噪声产生多条并行轨迹，并用原有的 Q head 选取最优答案，显著提升推理准确率。

**💡 创新点**

创新点在于：不需要重训练、无需任务特定数据增强，利用噪声注入实现宽度扩展的测试时推理；将 TRM 的 Q head 用作验证器，突破确定性递归陷入局部最优的瓶颈。

**🔧 技术方法**

技术方法包括：在每个深度递归步骤向潜在状态注入 Gaussian 噪声；并行多条轨迹（宽度 K）；使用 Q head 进行答案选择；结合深度（D）和宽度（K）两轴的推理扩展。

**📊 数据集**

使用的数据集为 PPBench（多种填字类谜题）、Sudoku-Extreme、Maze-Hard、ARC-AGI 2 等推理基准。

**📈 对比分析**

通过与原始 TRM、直接预测基线以及前沿 LLM（Gemini‑3.1‑pro、GPT‑5.2、Claude‑opus‑4‑6 等）比较，PTRM 在 PPBench 黄金集上达 91.2% 准确率，几乎是最强 LLM 集合（55.1%）的两倍，且成本仅为 $0.001；Sudoku‑Extreme 上 98.75% 最高；Maze‑Hard 86.73%；ARC‑AGI 8.47% pass@1。

**⚠️ 局限性**

局限性在于：实验仅聚焦于推理类谜题，未验证在更一般任务或更大规模问题上的效果；在部分任务（如 ARC‑AGI‑2）中 Q head 的验证能力不足，提示需要更强的验证器。

---

## 598. Real-Time Parallel Counterfactual Regret Minimization

**arXiv ID:** 2605.19928 | [PDF](https://arxiv.org/pdf/2605.19928v1)

**作者:** Boning Li `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**通讯引用:** 3774 | [OpenAlex ID](https://openalex.org/A5082905458)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Parallel CFR 并实现了深度限定、剪枝、抽象及 GPU 批量叶子评估的并行 CFR 求解框架

**💡 创新点**

首次将信息集合与树节点两维并行化结合，构造七阶段流水线，并在同一台桌面级 GPU 设备上完成实时求解

**🔧 技术方法**

多线程 CPU 并行、GPU 批量神经网络推理、在线剪枝、信息抽象、深度限定求解与 DCFR/PCFR+ 等 CFR 变体

**📊 数据集**

使用 Heads-Up No-Limit Texas Hold'em（HUNL）子游戏数据，包含不同 stack-to-pot 比例和提问动作数的完整树结构

**📈 对比分析**

与单线程基线相比，后街节奏提升 3.3–3.4×；与 PokerRL 对比在相同 wall‑time 下可实现约 7 倍更低可利用度；在 5 秒决策预算内完成数百次迭代

**⚠️ 局限性**

仅针对单实例求解，未结合分布式计算；GPU 代码仍以 CPU 为主，未充分利用异步 CPU–GPU 并行；未来需实现完整 GPU 流水线与更大规模扩展

---

## 599. Fast and Featureless Node Representation Learning with Partial Pairwise Supervision

**arXiv ID:** 2605.19916 | [PDF](https://arxiv.org/pdf/2605.19916v1)

**作者:** Sujan Chakraborty `[一作]` (Indian Institute of Science Education and Research), Saptarshi Bej `[通讯]` (Indian Institute of Science Education and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为Contrastive FUSE的无特征图节点嵌入方法，利用部分对称的正负节点对标签，在不需要深度编码器和图增强的情况下，结合模块度和对比性拉普拉斯进行谱级别的快速优化。

**💡 创新点**

创新点在于：① 将模块度最大化与对比监督（正负节点对）直接整合为一个闭式目标；② 设计签名归一化对比拉普拉斯，既拉近正样本又推远负样本；③ 用线性化的模块度梯度近似替代昂贵的梯度计算，实现 O(|E|+P) 的迭代复杂度；④ 在无特征或特征稀缺场景下依旧能获得竞争甚至更优的下游性能。

**🔧 技术方法**

核心技术包括：谱聚类与模块度优化、签名对比拉普拉斯构造、梯度上升与行归一化、对比权重自适应学习率、以及可选的 GNN 微调。

**📊 数据集**

实验使用了多种基准数据集：Cora、CiteSeer、PubMed、WikiCS、Amazon‑Photo（中等规模），以及大规模 OGBN‑ArXiv、OGBN‑Products；每个数据集都构造正负节点对作为监督。

**📈 对比分析**

与 DeepWalk、Node2Vec、VGAE、DGI、GRACE、COLES、CCA‑SSG、MVGRL、SGCL 等无监督/自监督基线以及随机/给定特征基线进行对比。结果显示 Contrastive FUSE 在节点分类（Logistic、GNN 等）上达到或超过所有对照组，同时在嵌入时间上比随机游走、深度对比方法快 10–15 倍，内存占用更低。

**⚠️ 局限性**

局限性包括：① 需要事先可用的正负节点对标签，若标签稀缺或噪声高会影响效果；② 对密集图需要手动调节学习率和对比权重；③ 目前仅支持单一视图无异构图，扩展到多视图/异构图仍需研究。

---

## 600. Can LLMs Produce Better Object-Oriented Designs than Human-Involved Development?

**arXiv ID:** 2605.19901 | [PDF](https://arxiv.org/pdf/2605.19901v1)

**作者:** Zushuai Zhang `[一作]` (University of Auckland), Ewan Tempero `[通讯]` (University of Auckland)

**通讯引用:** 3638 | [OpenAlex ID](https://openalex.org/A5069747561)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过比较PreAI、PostAI和PureAI三种作者条件下的Java项目，评估其对象导向设计质量、代码异味密度和领域概念实现。

**💡 创新点**

首次在项目级别对人类与LLM生成代码进行全面OOD质量对比，并研究提示指导具体度对PureAI设计质量的影响。

**🔧 技术方法**

使用CK工具计算OOD指标，DesigniteJava和PMD检测代码异味，手工评估领域概念实现，统计分析包括Mann-Whitney U、卡方检验和Odds Ratio。

**📊 数据集**

2021年和2024年两期学生提交的Kalah游戏项目（PreAI、PostAI）以及三种模型与三种提示变体生成的90次PureAI项目，共11个数据集。

**📈 对比分析**

采用非参数统计检验和效应量（Cliff's delta、Odds Ratio）进行两两比较，结果显示PureAI在代码异味和总复杂度上优于人类，但在类数、抽象层次和领域概念覆盖度上逊色；PostAI介于两者之间。

**⚠️ 局限性**

仅针对单一小型Java任务，样本量有限，且只分析通过测试的LLM生成代码，可能高估其设计质量；学生是否使用LLM未被确认，难以因果推断。

---

## 601. Fair-Aurora: Comparing Fairness Strategies for Reinforcement Learning-Based Congestion Control in Multi-Flow Environments

**arXiv ID:** 2605.19909 | [PDF](https://arxiv.org/pdf/2605.19909v1)

**作者:** Thomas Mbrice `[一作]`, Yuyu Liu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究Aurora RL拥塞控制在多流环境中的公平性，并在其RL框架上实现并评估了三种最小改动的公平策略；

**💡 创新点**

提出了奖励塑形、观察扩展和损失灵敏度调节三种后置公平策略，并揭示了“过度惩罚”失效模式，显示在RL拥塞控制中公平性可通过训练时的轻度改动实现；

**🔧 技术方法**

采用深度强化学习（PPO）训练Aurora控制器，并利用Jain公平性指数、定制共享瓶颈仿真器以及观察扩展和奖励塑形等技术手段实现公平策略；

**📊 数据集**

使用合成时变链路训练分布（带宽100–500pps、延迟50–500ms、丢包0–5%）以及自建的多流仿真环境，进一步在混合Aurora–CUBIC和动态流入退出场景中进行实验；

**📈 对比分析**

在两流固定基线下比较Jain指数，策略A（λ=2.0）得到J=0.876、策略B得到J=0.863、策略C（loss=8000）得到J=0.873，聚合吞吐均保持≈3.5Mbps；在混合TCP CUBIC场景中，策略C显著降低吞吐比并减少对CUBIC的损害；在动态流入退出场景中，策略B保持最高公平性；

**⚠️ 局限性**

实验采用简化的流式仿真器，缺乏完整协议层模拟；仅评估单一架构背景流；过度惩罚模式与评估协议交互不佳，需更广泛的参数探索和真实网络测试。

---

## 602. DASM: Domain-Aware Sharpness Minimization for Multi-Domain Voice Stream Steganalysis

**arXiv ID:** 2605.19955 | [PDF](https://arxiv.org/pdf/2605.19955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 603. RoHIL: Robust Human-in-the-Loop Robotic Reinforcement Learning Against Illumination Variations

**arXiv ID:** 2605.19924 | [PDF](https://arxiv.org/pdf/2605.19924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 604. Passive Construction Site Safety Monitoring via Persona-Scaffolded Adversarial Chain-of-Thought VLM Verification

**arXiv ID:** 2605.19869 | [PDF](https://arxiv.org/pdf/2605.19869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 605. Exploiting Non-Negativity in DAG Structure Learning

**arXiv ID:** 2605.19947 | [PDF](https://arxiv.org/pdf/2605.19947v1)

**作者:** Samuel Rey `[一作]` (Universidad Rey Juan Carlos), Gonzalo Mateos `[通讯]` (University of Rochester)

**通讯引用:** 4956 | [OpenAlex ID](https://openalex.org/A5006078163)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了非负权重下的DAG结构学习，提出了基于对数行列式的光滑无环性约束，并设计了使用乘子法的NOMAD算法；

**💡 创新点**

创新点在于利用边权非负性构造了不需要Hadamard积的更简洁无环性函数，证明了其在总体层面上无陷阱的优化景观，并且在全局最优点唯一；

**🔧 技术方法**

采用正则化最小二乘目标、对数行列式无环性约束、增广拉格朗日（方法乘子）优化以及非负投影和矩阵幂/指数级数分析；

**📊 数据集**

使用了合成的Erdős–Rényi与规模自由网络以及真实的Sachs蛋白信号网络作为评估数据集；

**📈 对比分析**

与连续无环性约束的NOTEARS、DAGMA、CoLiDE等方法进行比较，在样本量、图大小、噪声方差以及Sachs基准上，NOMAD在估计误差、结构Hamming距离、FDR等指标上均优于或相当，且收敛更稳健；

**⚠️ 局限性**

局限性包括仅适用于非负权重，无法处理负效应；增广拉格朗日在非凸域缺乏理论收敛保证，对超参数和初始值敏感；在大规模图上计算复杂度较高。

---

## 606. What Are LLMs Doing to Scientific Communication? Measuring Changes in Writing Practices and Reading Experience

**arXiv ID:** 2605.19936 | [PDF](https://arxiv.org/pdf/2605.19936v1)

**作者:** Filip Miletić `[一作]`, Neele Falk `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探究了大语言模型（LLM）对自然语言处理（NLP）学术写作风格的影响，并测评其对读者阅读体验的作用。

**💡 创新点**

创新点在于①将LLM辅助写作视为自然混合实践，利用两个时间段（ChatGPT发布前后）和对照实验来捕捉写作风格变化；②结合词汇语义变化、复杂性与可读性指标与读者主观评价，首次从客观与主观双重视角阐释LLM写作的实际影响；③发布了更新的ACL-OCL语料库、LLM改写对照集以及阅读体验标注数据。

**🔧 技术方法**

技术方法包括词频与上下文分布的log-likelihood与词汇密度分析；词向量与上下文BERT嵌入的聚类语义变迁分析；逻辑回归+elastic net稳定性选择对句子/词汇层面的语言特征进行建模；对照实验中使用GPT‑3.5‑turbo生成改写文本；人类评估采用配对比较的Likert量表并进行Wilcoxon检验。

**📊 数据集**

数据集：①更新版ACL-OCL（99.2k篇论文，覆盖2020‑2024），按两段时间划分；②synthetic LLM dataset（3,000段落原文与GPT‑3.5改写，6k段落）；③20名NLP专家对200对文本进行阅读体验标注。

**📈 对比分析**

比较方法：通过log‑likelihood与词汇密度评估词汇使用变化；通过逻辑回归的AUC（原始数据AUC≈0.65）与混合效应模型解释LLM改写文本的语法与词汇特征；在阅读体验上使用Wilcoxon检验和Cohen d，结果显示LLM改写在“清晰度”和“兴奋度”上显著优于原文（p<0.001），但“真实性”和“可信度”差异不显著。

**⚠️ 局限性**

局限性：①仅关注NLP学术英语，缺乏跨学科跨语言验证；②时间段划分粗略，未细化月度或季度变化；③对比实验使用单一模型GPT‑3.5，未覆盖GPT‑4等新模型；④阅读体验标注样本规模有限（200对，双评），可能缺乏统计稳健性。

---

## 607. Deep Tech to Space: Space Data Centers and AI Revolution at the Edge

**arXiv ID:** 2605.19892 | [PDF](https://arxiv.org/pdf/2605.19892v1)

**作者:** Jonas Weiss `[一作]` (IBM Research Europe), Agata Wijata `[通讯]` (KP Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并评估了一种基于低地球轨道星座的空间数据中心（SDC）概念，设计了星座轨道、FSO互连网络、软件驱动的多租户AI服务，并使用预测工具对技术与经济可行性进行量化分析。

**💡 创新点**

创新点包括：1）将SDC星座与光链路网络相结合，形成环形互连结构；2）提出软件化多租户AI服务模型；3）开发基于技术路线图的长期成本与性能预测框架，首次对SDC的未来可行性做系统评估。

**🔧 技术方法**

主要技术包括：光学可视光链路（FSO）、GPU/AI推理（如U‑Net语义分割）、软件定义网络与API标准、以及Excel预测工具。

**📊 数据集**

使用的主要数据集来自地球观测（Sentinel‑2）以及月球探测任务的示例数据，用于模拟火灾检测和月球车成像分割等工作负载。

**📈 对比分析**

通过自建Excel预测工具与多条技术路线图对比不同设计参数（轨道高度、功耗、计算技术等），对比指标包括计算能力、功率效率和成本，结果显示在2032‑2040年期间，GPU等效方案可满足工作负载且成本呈下降趋势。

**⚠️ 局限性**

局限性包括：FSO链路易受太阳偏角遮挡影响，导致部分轨道段可用性降低；对高纬度/极地地区覆盖不足；缺乏统一标准和规范；以及高前期投资与商业化路径仍需进一步验证。

---

## 608. Equilibria in Multiplayer Graph Games: An Algorithmic Study

**arXiv ID:** 2605.19954 | [PDF](https://arxiv.org/pdf/2605.19954v1)

**作者:** Léonard Brice `[一作]` `[通讯]`, Léonard Brice

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本论文针对多玩家图游戏中的平衡概念（Nash、子博弈完美、强安全、风险敏感等）开展了算法和复杂度研究，系统地给出了在不同游戏类别（Parity、Mean‑Payoff、Discounted‑Sum、Energy、Stochastic）下“受约束的存在性问题”（是否存在满足阈值约束的平衡）的多项复杂度结果，并提出了新型的协商函数（negotiation function）来刻画子博弈完美平衡。

**💡 创新点**

创新点包括：
1) 对子博弈完美平衡提出协商函数和固定点方法，证明了在Parity、Mean‑Payoff等游戏中受约束存在性问题的完整性（co‑Büchi、Parity、Mean‑Payoff均为P‑complete、co‑Büchi/Parity为NP‑complete、Mean‑Payoff为P‑complete）。
2) 证明Discounted‑Sum游戏中受约束存在性问题至少与目标Discounted‑Sum问题同难度，并给出其co‑r.e.性。
3) 在Energy游戏中阐明Nash及SPE的不可判定性，并指出SPE在Energy游戏中的递归可枚举性仍为开放问题。
4) 引入极端风险敏感平衡，展示在随机游戏中此类平衡的可判定性（P‑complete或NP‑complete）。
5) 对强安全平衡与多玩家Parity/Parity Automata的受约束存在性问题给出完整性结果。

**🔧 技术方法**

主要技术手段：
- 零和游戏的确定性与凸性分析；
- 交替自动机与树自动机的构造；
- 协商函数与其固定点计算；
- 归约（到目标Discounted‑Sum、停机等问题）；
- 树搜索与König引理的运用来证明co‑r.e.性；
- 记忆结构与有限/无记忆策略的细化；
- 计算几何与凸包技术处理Mean‑Payoff。

**📊 数据集**

该研究为理论计算机科学工作，未使用实验数据集，所有结果均为理论证明和多项式时间/NP/undecidability 复杂度分析。

**📈 对比分析**

比较方式主要是通过复杂度类别进行对比：例如对比已知的NP‑完整/PSPACE‑可解/不可判定等结果；对于新的判定算法，给出多项式/指数时间上界；在随机游戏中给出P‑/NP‑可判定性；此外对先前的可判定性结果进行补充或改进，展示完整性或更精细的阈值约束下的难度差异。

**⚠️ 局限性**

限制与未解问题：
- 对Energy游戏中子博弈完美平衡的递归可枚举性仍未知；
- 随机游戏中混合策略下的风险敏感平衡的复杂度仍不可判定；
- 许多结果依赖于无穷路径的理想化假设，实际系统有限时长可能导致偏差；
- 对极端风险敏感平衡的可扩展性和实现细节尚未在实际协议中验证。

---

## 609. Robotics-Inspired Guardrails for Foundation Models in Socially Sensitive Domains

**arXiv ID:** 2605.19940 | [PDF](https://arxiv.org/pdf/2605.19940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 610. Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory

**arXiv ID:** 2605.19952 | [PDF](https://arxiv.org/pdf/2605.19952v1)

**作者:** Jingwei Sun `[一作]` (Hong Kong Baptist University), Bo Han `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 10925 | [OpenAlex ID](https://openalex.org/A5100781698)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了TriMem，一种三层级记忆体系，结合原始对话、提取的原子事实和聚合的实体档案，以实现LLM代理在长期交互中的高保真存储、高效检索和深度推理。

**💡 创新点**

创新点在于：①同时维护三种细粒度表示并通过索引保持信息完整；②利用实体档案实现对分散事实的整合推理；③采用TextGrad进行无参数更新的提示优化，使系统能在使用者交互中持续演化。

**🔧 技术方法**

技术手段包括：滑动窗口对话切分、基于多维度schema的事实提取、实体档案合成、相似度检索、检索后原始对话与档案恢复以及TextGrad提示梯度优化。

**📊 数据集**

使用数据集包括LoCoMo（多轮对话记忆评测）和PerLTQA（个人资料、社会关系、历史事件等多维问答）。

**📈 对比分析**

与Naïve RAG、Mem0、MemoryOS、A-Mem、LightMem、SimpleMem、xMemory等基线在LoCoMo和PerLTQA上进行对比，TriMem在BLEU/F1/LLM-判定正确率等指标均显著提升，尤其在多证据推理场景中表现突出。

**⚠️ 局限性**

局限性包括：对提示的优化仍需依赖大模型评判，可能导致过度细化导致泛化下降；在极端多样化对话风格下，提示仍可能需要人工干预；以及对极小参数模型的性能提升相对有限。

---

## 611. GEM: GPU-Variability-Aware Expert to GPU Mapping for MoE Systems

**arXiv ID:** 2605.19945 | [PDF](https://arxiv.org/pdf/2605.19945v1)

**作者:** Sourish Wawdhane `[一作]` (University of Texas at Austin), Poulami Das `[通讯]` (University of Texas at Austin)

**通讯引用:** 769 | [OpenAlex ID](https://openalex.org/A5039719490)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对MoE模型的GPU可变性感知专家映射框架，利用专家利用率和GPU性能差异实现更高效的推理。

**💡 创新点**

创新点在于同时考虑GPU性能波动和两类专家（一致性与时间性）的协同使用，并通过token负载与GPU速率的匹配来消除同步瓶颈。

**🔧 技术方法**

采用专家利用率采样、GPU层级性能剖析、启发式迭代搜索和动态映射部署技术。

**📊 数据集**

使用ShareGPT与CodeContests两个数据集进行推理评测。

**📈 对比分析**

与vLLM的线性映射和EPLB基线对比，平均减少7.9%（最高16.5%）的端到端延迟，p90 TPOT下降9.1%（最高16.9%）。

**⚠️ 局限性**

局限性包括需在推理前进行GPU性能剖析，适用场景受限于已知硬件可变性且未评估极大规模节点或动态工作负载。

---

## 612. A Measure-Theoretic Analysis of Reasoning: Structural Generalization and Approximation Limits

**arXiv ID:** 2605.19944 | [PDF](https://arxiv.org/pdf/2605.19944v1)

**作者:** Yuyang Zhang `[一作]` (McGill University), Xiaoyin Chen `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文结合最优传输与Transformer电路复杂度，构建连续度量空间下LLM推理的OOD泛化理论框架，证明绝对位置编码导致Ω(1)风险，旋转嵌入保持平滑并且层深度是克服TC^0瓶颈的关键。

**💡 创新点**

创新点在于将搜索轨迹投影至连续空间，用Wasserstein-1量化域移位，并通过Kantorovich对偶与Barron空间逼近限制，统一表述位置编码、层深和宽度对OOD推理性能的上下界。

**🔧 技术方法**

采用量化测度理论、最优传输、Kantorovich对偶、Barron空间逼近、Dyck语言复杂度分析、Transformer架构实验与Sinkhorn‑Knopp OT估计等技术。

**📊 数据集**

使用自定义的Stream of Search（SoS）模拟器在计数游戏“Countdown”中生成500k条BFS、DFS与混合分布轨迹，作为实验数据集。

**📈 对比分析**

通过构建54种Transformer配置（不同层深、宽度、位置编码），在各目标分布上评估准确率；结果显示RoPE模型随Wasserstein距离线性下降，APE模型严重退化；深度扩展显著提升性能，宽度扩展则趋于饱和。

**⚠️ 局限性**

主要局限在于依赖ε-结构充分性假设，可能不适用于开放式自然语言推理；仅针对单路径前向推理，未覆盖树搜索等动态推理方法；TC^0瓶颈假设基于有限精度Transformer，实际推理时间技术可能进一步缓解该限制。

---

## 613. LLM Agents Make Collective Belief Dynamics Programmable: Challenges and Research Directions

**arXiv ID:** 2605.19915 | [PDF](https://arxiv.org/pdf/2605.19915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 614. reconCTI: A Proactive Approach to Cyber-Threat Intelligence

**arXiv ID:** 2605.19899 | [PDF](https://arxiv.org/pdf/2605.19899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 615. GLUT: 3D Gaussian Lookup Table for Continuous Color Transformation

**arXiv ID:** 2605.19889 | [PDF](https://arxiv.org/pdf/2605.19889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 616. Revisiting recursive methods for Dyson and Keldysh in NEGF: Part I

**arXiv ID:** 2605.19910 | [PDF](https://arxiv.org/pdf/2605.19910v1)

**作者:** Edoardo Di Napoli `[一作]` (Forschungszentrum Juelich Gmbh), Gustavo Ramirez-Hidalgo `[通讯]` (Forschungszentrum Juelich Gmbh)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

重新构造了递归格林函数（RGF）方法，采用域分解与Schur补理论，将其扩展到任意阶块带（n>3）并实现并行域分解 RGF（DDRGF）

**💡 创新点**

创新点在于把 RGF 视为块 LDU 分解的域分解，能够天然处理高阶离散化（块 n‑带矩阵）且不需人为增大块；提出的 DDRGF 通过显式的块稀疏数据依赖实现可扩展并行，并给出硬件感知的成本模型与自动调优器

**🔧 技术方法**

使用 Julia 语言实现，核心运算基于 BLAS（GEMM、LU、GETRS），并通过自适应成本模型自动决定分块大小与递归深度；同时实现了多线程共享内存与未来的 MPI/多 GPU 扩展

**📊 数据集**

实验使用合成的块三对角与块五对角（n=3,5）矩阵，块尺寸从 64 到 512，层数从 20 到 2880，基于 JUWELS、Booster 节点 CPU 进行测试

**📈 对比分析**

与传统串行 RGF 以及块 n‑带“拼接”方法比较，结果显示：对 n>3 时本机 RGF 比拼接快数倍；DDRGF 在多线程（8–48 线程）时显著优于 RGF，突破 12 线程后随线程数呈对数下降；在大规模块尺寸下，DDRGF 的额外算子量被充分隐藏，实现了良好伸缩

**⚠️ 局限性**

限制包括：DDRGF 需要额外算子与内存开销，单节点内存仍受限；目前仅实现共享内存多线程，分布式多节点和 GPU 版本仍待开发；在极小层数或线程极少时，传统 RGF 更快；算法对块结构高度依赖，非块带或非 1D 结构需进一步适配

---

## 617. Streamlined Constraint Reasoning via CNN Pattern Recognition on Enumerated Solutions

**arXiv ID:** 2605.19895 | [PDF](https://arxiv.org/pdf/2605.19895v1)

**作者:** Patrick Spracklen `[一作]` `[通讯]`, Patrick Spracklen

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于CNN对枚举解进行模式识别后，再用LLM生成约束的流线化（streamliner）自动化管道，能够在已加标准硬化（对称性破坏和推导约束）后的CP模型中发现新的非保真约束，从而显著加速求解；

**💡 创新点**

创新点在于将CNN作为结构模式检索器，将其判别信号（滤波器与属性相关性或高低激活对比）输入LLM，引导LLM生成针对实例特定结构的MiniZinc约束，突破传统基于约束词法或模型文本的流线化合成限制；

**🔧 技术方法**

使用的技术包括：枚举解生成、3层卷积神经网络的对比学习、滤波器-属性相关性分析、滤波器对比（高激活/低激活）生成约束候选、LLM（Claude Opus）提示式约束合成、候选池聚合与语义聚类、基于家庭预算的多样性投资组合投放；

**📊 数据集**

数据集为三类硬化后的标准CP基准：社会高尔夫（Social Golfers）、船舱装载（Vessel Loading）和无硬化的黑洞（Black Hole）共计约132个实例，训练集使用各问题的部分实例，测试集使用剩余实例；

**📈 对比分析**

与传统基于约束词法或模型文本的StreamLLM方法直接对比，只在无硬化黑洞上；通过家族预算投放，在硬化后的Social Golfers、Vessel Loading和Black Hole上分别获得98.6%、98.8%和89.4%的端到端时间缩减，单个最佳约束在各问题上几百到上千倍加速；

**⚠️ 局限性**

局限包括：仅使用LLM Claude，未能在开放权重模型上复现；对CNN的对比学习可能缺少多样化负样本导致单一模式；整体性能高度依赖于枚举解的可枚举性和实例规模，且结果对随机种子和LLM提示不完全可重复；

---

## 618. SpecSA: Bridging Speculative Decoding and Sparse Attention for Efficient LLM Inference

**arXiv ID:** 2605.19893 | [PDF](https://arxiv.org/pdf/2605.19893v1)

**作者:** Zhibin Wang `[一作]` (State Key Laboratory for Novel Software Technology), Sheng Zhong `[通讯]` (State Key Laboratory for Novel Software Technology)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种稀疏推理加速框架（SpecSA），将稀疏注意力（Native Sparse Attention）与投机解码（Speculative Decoding）相结合，实现了高效的长上下文 LLM 推理。

**💡 创新点**

创新点在于：① 针对跨查询重叠的 KV 块采用重叠感知的分组查询执行；② 通过刷新/重用层实现稀疏注意力的完整内核融合，显著降低索引构造与分支碎片化开销；③ 基于离线剖面与实时接受率的提示适应规划器，实现输入自适应策略选择。

**🔧 技术方法**

使用技术包括：投机解码、动态稀疏注意力（NSA）、GPU 内核融合、线程分组（GQA）、索引缓存（IndexCache）式刷新/重用调度、EAGLE‑3 框架集成以及 profile‑guided 规划器。

**📊 数据集**

数据集与模型：使用 Children‑Stories‑Collection 作为提示语料；实验基于 Llama‑3‑1B 和 Llama‑3‑8B 变体的 NSA 目标模型，评估在 NVIDIA H100 GPU 上的性能。

**📈 对比分析**

与基线（纯自回归 NSA）对比，SpecSA 在 EAGLE‑3 集成下可实现最高 3.49× 的端到端吞吐量提升，稀疏投机验证阶段的单核加速达 6.86×，整体验证阶段加速 1.45×，规划策略进一步提升 33.2% 的已接受 token 吞吐。

**⚠️ 局限性**

局限性包括：① 对输入提示的敏感性，需要离线剖面与实时监测才能获得最优配置；② 近似共享索引与索引重用虽提升速度但在极端稀疏场景下可能影响模型精度；③ 规划与重组机制增加了实现复杂度，且对其他稀疏注意力实现的可迁移性尚未验证。

---

## 619. DAG-Based QoS-Aware Dynamic Task Placement for Networked Multi-Stage Control Pipelines

**arXiv ID:** 2605.19887 | [PDF](https://arxiv.org/pdf/2605.19887v1)

**作者:** Thien Tran `[一作]` (Deakin University), Jiong Jin `[通讯]` (Swinburne University of Technology)

**通讯引用:** 6758 | [OpenAlex ID](https://openalex.org/A5080328538)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并提出了一种基于有向无环图（DAG）的QoS感知动态任务放置（DTP）框架，用于网络化机器人感知–规划–控制管线，并在边缘节点实现窗口化决策、成本函数与抑制摆动的机制。

**💡 创新点**

创新点包括：①将四阶段管线拆解为可放置任务与硬锚任务的DAG模型；②引入多目标QoS成本函数，加入Hamming距离迁移惩罚；③采用窗口化DTP算法，结合滞后与最小停留时间来避免频繁迁移；④在工业通信（TSN/5G‑URLLC）环境下完成3C协同设计，并提供可验证的理论与实验路线。

**🔧 技术方法**

使用的技术主要有：DAG建模、窗口化优化、Hamming距离迁移惩罚、滞后与停留时间控制、静态/在线性能评估、离散事件仿真、硬件‑环回(HIL)验证、工业实时网络仿真（RTT/Jitter）等。

**📊 数据集**

采用工业机器人现场测试平台（R1、R2机器人 + 边缘服务器 E）和对应的网络延迟分布、CPU负载曲线作为数据来源；在离散事件仿真中生成随机网络故障与CPU压力的合成数据。

**📈 对比分析**

通过与三种基线（全局本地 LOC、静态远程 SO、单任务自适应 ATP）在不同压力场景（CPU、网络、负载）下的比较，预期 DTP 能将截止违约率控制在 ≤5%，并在满足硬锚任务低延迟的前提下平衡整体延迟、CPU利用率与迁移成本。

**⚠️ 局限性**

局限性包括：①候选放置仅限于 LOC/SO/HYB 三种，缺乏更细粒度或多机器人场景；②尚未完成实证验证，权重调节与在线自适应机制待进一步研究；③未对网络资源进行联合调度，仅以延迟分布建模；④未考虑 Age‑of‑Information 等新 QoS 维度。

---

## 620. Structural Energy Guidance for View-Consistent Text-to-3D Generation

**arXiv ID:** 2605.19876 | [PDF](https://arxiv.org/pdf/2605.19876v1)

**作者:** Qing Zhang `[一作]` (Australian National University), Xuesong Li `[通讯]` (Australian National University)

**通讯引用:** 6414 | [OpenAlex ID](https://openalex.org/A5100449081)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种无训练、可插拔的 Structural Energy Guidance (SEGS) 框架，利用 Diffusion U‑Net 中间注意力特征的 PCA 子空间构造视角结构能量，并将其梯度注入到 denoising 过程，以提升 text‑to‑3D 生成的多视角一致性。

**💡 创新点**

创新点在于把视角偏差视为可优化的结构能量，在采样阶段直接对梯度进行引导，既不需要重新训练模型，也不依赖外部监督或额外预测器，同时配备轻量化的文本一致性门以剔除错误提示。

**🔧 技术方法**

采用 Stable Diffusion v1.4/v2.1 的前置 diffusion 模型、PCA 对 U‑Net 关键特征降维、CLIP 视角分类与相似度筛选、BRISQUE 的自适应调度以及在 SDS/VSD 的能量引导。

**📊 数据集**

使用 LAION‑2B 进行视角偏差分析，DreamFusion prompt library 进行评测；对每个 prompt 生成 20 张 target‑view 图像作为 PCA 参考。

**📈 对比分析**

与 LucidDreamer、Magic3D、DreamFusion 等基线进行对比，Janus Rate 均下降约 10% 平均，View‑CS 上升 0.8‑1.5 分，证明在不增加训练成本的前提下显著提升多视角一致性且保持细节保真。

**⚠️ 局限性**

局限包括：对前后过渡角度仍可能残留前视特征；需要手工调节 PCA 维度与 top‑k；对视角分布的依赖程度较高；对不同 3D 表示（如 NeRF、GAN 等）的适用性还需进一步验证。

---

## 621. PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents

**arXiv ID:** 2605.19932 | [PDF](https://arxiv.org/pdf/2605.19932v1)

**作者:** Zhuohan Gu `[一作]` (MIT), Samuel Madden `[通讯]` (MIT)

**通讯引用:** 44609 | [OpenAlex ID](https://openalex.org/A5037742794)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个“上下文地图”缓存机制，用于在长上下文的LLM代理中保存可重用的方向知识，从而提升推理效率与准确率。

**💡 创新点**

将可重用的方向知识抽象为固定大小的提示缓存，并通过分离的Distiller、Cartographer、Evictor三模块实现在线更新与预算控制，首次在长上下文推理与学习任务中显著优于现有技术。

**🔧 技术方法**

基于缓存理论的三模块管理（Distiller、Cartographer、Evictor）、LLM轨迹分析、提示学习、RAG、上下文压缩等技术，并使用GPT‑5‑mini等语言模型进行实验。

**📊 数据集**

使用OOLONG和CL‑bench两大长上下文基准数据集，分别涵盖推理聚合与上下文学习任务。

**📈 对比分析**

与共享聊天、RAG、上下文压缩、ACE等方法在相同RLM框架下对比，提升6.3–34.0%的解决率/评分，减少93–145次迭代，成本降低1.4–5.8倍，并在GPT‑5.5、Qwen3‑Coder、Codex等模型和代理上保持一致优势。

**⚠️ 局限性**

受限于代理与上下文交互方式，若交互未产生可重用知识则缓存价值有限；映射内容可能因代理差异而表现不一致；当前未结合KV‑cache进一步提升效率。

---

## 622. StruMPL: Multi-task Dense Regression under Disjoint Partial Supervision and MNAR Labels

**arXiv ID:** 2605.19931 | [PDF](https://arxiv.org/pdf/2605.19931v1)

**作者:** Reza M. Asiyabi `[一作]` (University of Edinburgh), Casey M. Ryan `[通讯]` (University of Edinburgh)

**通讯引用:** 11744 | [OpenAlex ID](https://openalex.org/A5084153689)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种联合训练框架StruMPL，用于在空间观测与地面样本互相独立、缺失且存在MNAR的多任务稠密回归任务中估计森林上层生物量及相关结构指标。

**💡 创新点**

创新点在于：①将空间MNAR校正与物理约束（全方位生物量与结构的所有计公式）联合建模；②采用带两层stop‑gradient的AIPW伪目标实现稳定的多任务联合优化；③设计可学习的物理模块在无标签像素上提供跨源半监督梯度；④统一掩码与批次采样实现跨域（不同森林）可迁移。

**🔧 技术方法**

核心技术包括：共享Encoder（ResUNet+注意力） + 任务回归头、缺失填充头、倾向性头；AIPW加权损失与stop‑gradient；可学习的生态公式 g(·)；倾向性学习（交叉熵）；数据增强一致性；源平衡批次采样；超参数调优与初始化。

**📊 数据集**

使用两组实测数据：西班牙地中海森林 SNFI（约6000个测试样本）与非洲干燥热带/稀树草原 SEOSAW（162个标准化1 ha 样本）。

**📈 对比分析**

与单任务基线、无倾向性/无物理约束的多任务版本以及最近公开方法比较，StruMPL在两域均实现最小RMSE和最小偏差；在高生物量尾部的偏差减少约54%，整体RMSE比最强单任务基线低约3–4 Mg/ha。

**⚠️ 局限性**

局限性包括：① 需要近似的条件无关性假设（倾向性可忽略对生物量的直接影响），若违反AIPW无正式一致性保障；② 共享编码器未采用交叉拟合，AIPW仅为经验性偏差减小；③ 学习的全息系数为有效尺度系数而非真实生态参数；④ 对极少标签域的高阶特征依赖较大。

---

## 623. Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models

**arXiv ID:** 2605.19929 | [PDF](https://arxiv.org/pdf/2605.19929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 624. Trajectory Planning and Control near the Limits: an Open Experimental Benchmark on the RoboRacer Platform

**arXiv ID:** 2605.19881 | [PDF](https://arxiv.org/pdf/2605.19881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 625. OpenHealth Lake: Designing and testing a data lakehouse platform for health applications

**arXiv ID:** 2605.19922 | [PDF](https://arxiv.org/pdf/2605.19922v1)

**作者:** Danilo Silva `[一作]` (Stellenbosch University), Marcel Dunaiski `[通讯]` (Stellenbosch University)

**通讯引用:** 250 | [OpenAlex ID](https://openalex.org/A5015481306)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了基于湖仓架构的 OpenHealth Lake 数据湖仓平台，并通过 31 名参与者的可用性实验验证其易用性与功能完整性。

**💡 创新点**

将湖仓与数据联邦化、FAIR 原则相结合，提供云/自托管混合存储、基于 GA4GH Passport 的细粒度访问控制，以及 Python/R 封装的 API，形成灵活可扩展的数据管理解决方案。

**🔧 技术方法**

采用 Python FastAPI + Docker 搭建 REST API，使用 Couchbase 进行元数据与权限管理，支持 HDFS、Amazon S3、Google Cloud Storage 等存储后端，并实现 Delta Lake、FAIR4RS、GA4GH Passport、加密、Builder、Facade 等技术栈。

**📊 数据集**

主要使用 INFORM Africa 研究中心的健康数据集作为测试数据，未公开具体数据集名称，依托真实多机构健康数据验证平台。

**📈 对比分析**

通过可用性实验对比 Python 与 R 库的使用体验，测量任务完成时间和准确率；结果显示大多数任务被评为易/非常易，Python 组表现略优，完成时间合理，但部分高级查询与文件上传任务仍存在耗时和错误率。

**⚠️ 局限性**

存在版本控制机制单一、支持的存储环境有限、上传记录清理机制缺失、访问权限粒度不足、文档与任务说明不够清晰等限制，未来需引入更成熟的版本工具、扩展存储后端、实现事件驱动清理与细粒度授权等改进。

---

## 626. Where Does Authorship Signal Emerge in Encoder-Based Language Models?

**arXiv ID:** 2605.19908 | [PDF](https://arxiv.org/pdf/2605.19908v1)

**作者:** Francis Kulumba `[一作]` (Inria Paris), Florian Cafiero `[通讯]` (LRE, EPITA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在保持相同预训练编码器、训练数据和对比损失的前提下，研究了三种评分机制（平均池化+余弦、Late Interaction MaxSim、Patch-Level LI）对作者归因任务性能的影响，并通过梯度结构、残差流补丁和线性探针等机制可解释性工具解释了四倍的性能差距。

**💡 创新点**

证明评分机制决定信息整合的深度和梯度分布，而非编码器内部表示差异；提出通过残差流补丁定位信息整合层的因果方法，并将梯度稀疏性与模型性能关联。

**🔧 技术方法**

使用对比性作者归因框架（InfoNCE 损失）、三种评分机制、残差流补丁、LISA 线性探针、梯度分布分析以及训练动态监测等技术。

**📊 数据集**

HALvest-Contrastive base-4 学术文本语料（anchor、positive、negative 三元组），并将 E5 零射模型作为对照。

**📈 对比分析**

通过在同一模型、数据和损失下仅更换评分机制来比较；结果显示平均池化 R@20=0.121，Late Interaction R@20=0.485，PLI n=2 R@20=0.497，E5 0.167，表现出约四倍的性能差距。

**⚠️ 局限性**

局限性包括仅使用 ModernBERT backbone，未验证其他架构；仅测试 PLI n=2，未探究更大 patch 影响；补丁样本量有限导致统计分辨率不足；缺乏更广泛的跨领域或多语种评估。

---

## 627. Hierarchical Contrastive Learning for Multi-Domain Protein-Ligand Binding

**arXiv ID:** 2605.19902 | [PDF](https://arxiv.org/pdf/2605.19902v1)

**作者:** Shuo Zhang `[一作]` (University of Birmingham), Jian K. Liu `[通讯]` (University of Birmingham)

**通讯引用:** 80129 | [OpenAlex ID](https://openalex.org/A5075670673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对多域蛋白-配体结合亲和力的分层对比学习框架 HCLBind，显著提升了多域蛋白的预测精度。

**💡 创新点**

创新点包括：① 结合局部配体‑蛋白匹配（LPM）与全局接口假体辨别（IDD）的分层预训练；② 使用 LoRA 进行参数高效微调；③ 引入 Evidential Deep Learning 进行不确定性量化。

**🔧 技术方法**

采用的技术有：域门控图注意力网络、跨模态注意力、低秩适配（LoRA）、对比学习目标、Evidential 归一逆伽玛回归。

**📊 数据集**

使用 Q-BioLiP 数据库进行自监督预训练，PDBbind v2020 进行有标签微调，并按单域、接口、链连接器三类对测试集进行分层评估。

**📈 对比分析**

与 DrugBAN、GraphDTA、TankBind、DynamicBind、Caster-DTA、DeepDTAGen、CL‑GNN 等基线对比，HCLBind 在 RMSE、PCC、C‑Index 上分别达到 1.309、0.698、0.744，均为最高水平。

**⚠️ 局限性**

局限性包括：对比基线范围有限，主要聚焦内部消融实验；缺乏对新化学结构的外部化学泛化评估；未来需扩展更多结构和序列基线及独立化合物集验证。

---

## 628. Deterministic Single Exponential Time Algorithms for Co-Path Packing and Co-Path Set Parameterized by Treewidth

**arXiv ID:** 2605.19870 | [PDF](https://arxiv.org/pdf/2605.19870v1)

**作者:** Yuxi Liu `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1501 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对 Co-Path Packing 与 Co-Path Set 的两种问题的确定性单指数时间算法，并给出了以树宽为参数的最优运行时间上界；

**💡 创新点**

首创性地将代表族技术（representative families）应用于连通性约束问题，成功突破了 Cut & Count 随机化技术在树宽下的确定性化瓶颈；

**🔧 技术方法**

利用图论中图形基（graphic matroid）与最大代表族求解技术，结合树分解（nice tree decomposition）上的动态规划来控制状态空间；

**📊 数据集**

本工作为纯理论算法，未使用实验数据集；

**📈 对比分析**

通过对比现有随机化算法（Co‑Path Set O*(4^tw)、Co‑Path Packing O*(5^pw)）与新算法（Co‑Path Set O*((12+3·2^ω+1)^tw·tw^O(1)n)、Co‑Path Packing O*((13+3·2^ω+1)^tw·tw^O(1)n)），展示了在树宽参数下实现了完全确定性且保持单指数级别的时间复杂度；

**⚠️ 局限性**

当前的运行时间上界在 join‑node 处理上仍显松散，未来可进一步优化；此外算法仅在理论层面证明，尚未针对实际实例进行实验验证。

---

## 629. Feed-Forward Gaussian Splatting from Sparse Aerial Views

**arXiv ID:** 2605.19949 | [PDF](https://arxiv.org/pdf/2605.19949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 630. WoundFormer: Multi-Scale Spatial Feature Fusion for Multi-Class Wound Tissue Segmentation

**arXiv ID:** 2605.19868 | [PDF](https://arxiv.org/pdf/2605.19868v1)

**作者:** Muhammad Ashad Kabir `[一作]` (Charles Sturt University), Rabin Dulal `[通讯]` (Charles Sturt University)

**通讯引用:** 51 | [OpenAlex ID](https://openalex.org/A5013301746)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出WoundFormer框架，对多分类伤口组织进行像素级分割。

**💡 创新点**

创新点在于将SegFormer的All-MLP解码器替换为空间保持的多尺度聚合头，保留特征图拓扑并加强跨尺度语境融合，从而提升边界定位和相似组织的辨识度。

**🔧 技术方法**

采用层次Transformer编码器（MiT‑B5）、1×1通道对齐卷积、粗细级联上采样融合、3×3空间细化卷积、交叉熵损失和数据增强等技术。

**📊 数据集**

使用WoundTissueSeg（147张图，6类）和DFUTissue（110张图，4类）两个临床数据集进行训练与评估。

**📈 对比分析**

与多种CNN与Transformer基线（如SegNet、nnU‑Net、SegFormer‑B5、FPN+VGG16、DFUTissueSegNet等）对比，WoundFormer在WoundTissueSeg上实现81.9% Dice（未增强）/85.5% Dice（增强），比SegFormer‑B5高4.3点；在DFUTissue上达85.4% Dice（增强），整体性能显著提升。

**⚠️ 局限性**

局限性包括样本量有限、类别不平衡、仅在两套数据集上验证、未做交叉验证、对极少见组织的分割仍有改进空间，以及对不同机构、设备的泛化能力尚未充分验证。

---

## 631. Uncertainty-aware Machine Learning Interatomic Potentials via Learned Functional Perturbations

**arXiv ID:** 2605.19939 | [PDF](https://arxiv.org/pdf/2605.19939v1)

**作者:** Olga Zaghen `[一作]` (Universiy of Amsterdam), Erik J. Bekkers `[通讯]` (Universiy of Amsterdam)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种通过学习函数扰动将确定性机器学习原子势（MLIP）转化为概率模型的轻量级方法，并使用连续秩概率得分（CRPS）进行端到端训练；

**💡 创新点**

创新点在于仅通过在每个MLP块的第一个线性层插入共享噪声向量，并将噪声投影学习为零初始化矩阵，避免了传统的变分推理、KL正则或额外网络，保持了E(n)等变性；

**🔧 技术方法**

采用了CRPS作为训练目标，使用随机噪声注入与学习的噪声投影矩阵实现概率预测；

**📊 数据集**

在N-body Coulomb 体系（位置预测）和SiO₂硅酸盐玻璃（力和能量）两个公开基准数据集上进行评估；

**📈 对比分析**

与传统的确定性EGNN、BLIP+变分方法和小型深度集成（K=3）对比，P-EGNN在所有训练规模下实现了最优CRPS并且在MSE上与BLIP相当；P-Orb在硅酸盐玻璃上将CRPS降低30%以上，Spearman相关从0.75提升至0.84，显示更好的不确定性校准；

**⚠️ 局限性**

局限性包括：训练时需要多次前向/反向传播（K_train≥10）导致较高计算成本；噪声结构仅为线性投影，可能限制表达能力；在能量预测方面表现不稳定，需进一步平衡能量与力的损失权重；

---

## 632. Text-to-SPARQL Generation with Reinforcement Learning: A GRPO-based Approach on DBLP

**arXiv ID:** 2605.20066 | [PDF](https://arxiv.org/pdf/2605.20066v1)

**作者:** Jann Pfeifer `[一作]` (Leuphana University Lüneburg), Ricardo Usbeck `[通讯]` (Leuphana University Lüneburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

训练一个小型的Qwen3-1.7B语言模型，通过Group-Relative Policy Optimization（GRPO）强化学习，实现零样本的文本到SPARQL查询生成，并通过实体/关系的符号提示引导模型生成符合语法与语义的查询。

**💡 创新点**

创新点包括：①使用小模型在缺乏token级别黄金查询的情况下，仅依赖执行反馈、结构约束和答案奖励进行训练；②首次将GRPO应用于学术领域的单步Text-to-SPARQL任务；③通过对比实验展示了执行奖励对性能提升的主导作用，并证明了奖励形状（shaping）对提升有限。

**🔧 技术方法**

采用的技术包括：Group-Relative Policy Optimization（GRPO）强化学习；Qwen3-1.7B LLM；符号提示式prompt（包含实体/关系URI及类型信息）；执行反馈奖励；结构约束与答案级别奖励；可选的gold-query shaping；以及监督式DoRA微调基线作为对比。

**📊 数据集**

使用的主要数据集是DBLP-QuAD（约1万条问答对，涵盖多类型问题），并利用DBLP知识图谱的SPARQL执行器生成答案集进行奖励计算与评估。

**📈 对比分析**

对比方法包括：零样本prompt基线、GRPO训练模型（两种变体：仅执行奖励与执行奖励+shaping）以及同模型规模的监督式DoRA微调基线。评估指标为答案级别准确率、执行准确率、类别细分分数和对未见模板的泛化能力。实验显示：GRPO显著优于零样本基线，表现与监督基线相近，且执行奖励贡献最大，shaping提升有限。

**⚠️ 局限性**

局限性包括：①假设完美的实体与关系链接，未解决实体歧义与链接错误问题；②仅在学术领域DBLP-QuAD上验证，未测试对其他知识图谱或非学术数据的适用性；③采用单步生成，未覆盖需要多步推理的复杂查询；④RL训练仍需依赖可执行的知识图谱和答案集，可能对资源有限的场景产生挑战。

---

## 633. Automating proof search when equality is a logical connective

**arXiv ID:** 2605.20054 | [PDF](https://arxiv.org/pdf/2605.20054v1)

**作者:** Kaustuv Chaudhuri `[一作]` (Inria), Dale Miller `[通讯]` (Inria)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种将同义性作为逻辑连结词的形式化方法，并基于此构建了轻量级的逻辑框架 SLIM，用来支持在一阶推理体系中进行自动化证明搜索。

**💡 创新点**

创新点在于：① 将等式处理为可在序列推理系统中使用的逻辑连结词；② 设计了一套状态转移系统（包含 backchaining 与 Huet 预统一化的推广），能够在存在量词交替和正负出现的等式情况下进行高阶统一；③ 通过“抬升（raising）”与归一化，将任意目标公式转化为可简化的状态公式，从而实现可控的搜索。

**🔧 技术方法**

使用技术包括：Gentzen 样式的序列推理系统、同义性左右引入规则、Huet 的预统一化、状态转移系统、归一化与抬升、以及逻辑程序化风格的定义式（definite clauses）来指定对象层的推理规则。

**📊 数据集**

本工作主要为理论性研究，未使用特定实验数据集；在实现层面，论文在 Abella 证明助手中实现了 search tactic，并在其内部对 SLIM 进行了编码验证。

**📈 对比分析**

论文未给出实验对比或性能评估；仅通过证明论证（如相对完备性与可判定性讨论）说明方法的理论可行性；未来工作需在实际系统中测评搜索效率与资源占用。

**⚠️ 局限性**

局限性包括：① 归一化与简化过程虽然可终止，但整体搜索仍存在非终止风险（如无穷递归的 imitation 步骤）；② 由于未实现 occurs‑check 与约束处理规则，可能产生无限深的搜索路径；③ 目前不支持 ∇ 量化符号与完整的递归/归纳/共归纳证明细节；④ 证明搜索的自动化程度受限，仍需人工设定搜索策略与边界。

---

## 634. PromptRad: Knowledge-Enhanced Multi-Label Prompt-Tuning for Low-Resource Radiology Report Labeling

**arXiv ID:** 2605.20052 | [PDF](https://arxiv.org/pdf/2605.20052v1)

**作者:** Ying-Jia Lin `[一作]` (Chang Gung University), Hung-Yu Kao `[通讯]` (National Tsing Hua University)

**通讯引用:** 2988 | [OpenAlex ID](https://openalex.org/A5101898313)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了PromptRad，一种针对低资源环境下的多标签放射学报告自动标注方法。

**💡 创新点**

创新点在于将多标签分类转化为掩码语言模型任务，并通过UMLS词表构建多词Verbalizer以及自动化提示生成，显著减少对标注数据的需求。

**🔧 技术方法**

主要技术包括Prompt-Tuning、基于UMLS的多词Verbalizer、自动提示生成（T5生成提示模板）以及使用PubMedBERT作为预训练模型。

**📊 数据集**

使用了来自一家医疗中心的2008-2017年间的肝脏CT报告，共计1098份，其中773份用于训练，325份用于测试。

**📈 对比分析**

与字典基方法、传统Fine‑Tuning以及GPT‑4进行对比，在仅32份标注样本下，PromptRad在大多数类别上实现了最高的F1分数，且在负面案例处理上优于Rule‑Based和大模型。

**⚠️ 局限性**

局限性包括仅在单一中心、单一科室（肝脏CT）且仅英文报告上验证；对不同机构、影像模态或报告风格的泛化性待进一步评估；多词Verbalizer依赖外部知识库，若无UMLS支持则适用受限。

---

## 635. A Nash Equilibrium Framework For Training-Free Multimodal Step Verification

**arXiv ID:** 2605.20033 | [PDF](https://arxiv.org/pdf/2605.20033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 636. OP2GS: Object-Aware 3D Gaussian Splatting with Dual-Opacity Primitives

**arXiv ID:** 2605.20044 | [PDF](https://arxiv.org/pdf/2605.20044v1)

**作者:** Guiyu Liu `[一作]` (University of Oulu), Janne Heikkilä `[通讯]` (University of Oulu)

**通讯引用:** 10220 | [OpenAlex ID](https://openalex.org/A5072890921)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了OP2GS，通过在3D Gaussian Splatting 中加入双重不透明度（原始 σ 与实例 σ*）实现高效的开放词汇 3D 分割与实例识别。

**💡 创新点**

创新点在于：1) 引入实例不透明度 σ* 并与原始不透明度解耦，消除标签污染；2) 使用随机对象损失优化实例占据场；3) 通过多视图聚合生成对象级 CLIP 嵌入，避免高维特征存储。

**🔧 技术方法**

采用双不透明度渲染、随机对象损失、SAM2 投影标签、CLIP 多视图聚合、轻量化算子，保持原有 3DGS 结构与稀疏点云。

**📊 数据集**

主要使用 3DOVS、LERF‑Mask 进行开放词汇分割评估，Replica 用于实例分割实验。

**📈 对比分析**

与 ObjectGS、Gaussian Grouping 等 SOTA 方法对比，3DOVS 上 mIoU 达 97.1%，LERF‑Mask 上 92.3%，推理速度 121 FPS，显著超越同类方法。

**⚠️ 局限性**

局限性：依赖 2D 分割模型（如 SAM2）生成的视图一致掩码；单一标签限制细粒度分辨率；对透明物体和光照变化仍有一定敏感性。

---

## 637. Active Context Selection Improves Simple Regret in Contextual Bandits

**arXiv ID:** 2605.20040 | [PDF](https://arxiv.org/pdf/2605.20040v1)

**作者:** Mohammad Shahverdikondori `[一作]` (EPFL), Negar Kiyavash `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了有限上下文空间的上下文多臂赌博机问题，学习者为每个上下文推荐最佳行动，并通过上下文加权的简单遗憾进行评估。

**💡 创新点**

提出了主动选择上下文的策略，证明了在已知上下文分布的情况下，主动采样的简单遗憾率优于被动采样，且在预算干预设置中也进行了扩展分析。

**🔧 技术方法**

使用了上下文多臂赌博机的理论框架，提出了Explore-Explore-Then-Commit (EETC) 算法来平衡被动估计和主动分配。

**📊 数据集**

在合成数据和真实世界数据（如MovieLens 1M数据集）上进行了实验，验证了理论结果。

**📈 对比分析**

与被动采样方法进行比较，主动采样在简单遗憾率上表现更优，尤其在上下文分布已知的情况下，主动策略的简单遗憾率可提高至Θ(k^1/4)。

**⚠️ 局限性**

在上下文分布未知的情况下，EETC算法的性能未必优于完全被动或完全主动策略，但在长时间范围内能够匹配已知上下文分布的最佳策略的性能。

---

## 638. Beyond Binary Success: A Diagnostic Meta-Evaluation Framework for Fine-Grained Manipulation

**arXiv ID:** 2605.19986 | [PDF](https://arxiv.org/pdf/2605.19986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 639. CAMERA: Adapting to Semantic Camouflage in Unsupervised Text-Attributed Graph Fraud Detection

**arXiv ID:** 2605.20032 | [PDF](https://arxiv.org/pdf/2605.20032v1)

**作者:** Junjun Pan `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**通讯引用:** 25339 | [OpenAlex ID](https://openalex.org/A5008056593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个名为 CAMERA 的无监督文本属性图欺诈检测框架，能够在语义伪装情形下识别欺诈节点。

**💡 创新点**

创新点在于：① 引入自我解耦的混合专家（MoE）结构，使每个专家专注于不同的欺诈线索（结构、语义、全局）；② 上下文感知的门控网络根据节点及其邻域信息动态分配专家权重；③ 利用欺诈稀缺性实现无监督的一类学习与专家级损失，促使模型捕捉正类模式并突出异常。

**🔧 技术方法**

技术细节包括：LLM 作为文本特征编码器；多层自我解耦 MoE + GCN/MLP 自动编码器/全局 MLP；门控网络结合局部上下文；一类损失（BCE + 归一化）和专家损失（残差平方）以及门控熵正则。

**📊 数据集**

在四个公开文本属性图数据集（Reddit、Instagram、AmazonVideo、YelpChi）上进行实验，全部采用统一的 LLM（OpenAI text-embedding-3-small）作为特征。

**📈 对比分析**

与 9 种无监督 GFD 与 TAGAD 先进方法相比，CAMERA 在 AUROC 与 AUPRC 指标上均显著提升（除 Reddit 的 AUPRC 外）并且在大型数据集上表现更好，显示了更好的可扩展性与鲁棒性。

**⚠️ 局限性**

局限性包括：对极度稀缺或结构化严重失衡的场景（如 Reddit 的 AUPRC 低下）仍可能受限；模型对 LLM 质量高度依赖；在极端语义伪装或多模态场景下的泛化能力尚待进一步验证。

---

## 640. GeoX: Mastering Geospatial Reasoning Through Self-Play and Verifiable Rewards

**arXiv ID:** 2605.20006 | [PDF](https://arxiv.org/pdf/2605.20006v1)

**作者:** Kyeongjin Ahn `[一作]` (Korea Advanced Institute of Science and Technology), Meeyoung Cha `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 15899 | [OpenAlex ID](https://openalex.org/A5061810530)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自我对弈框架，利用可执行程序在卫星/航空图像上进行空间推理，无需人工标注。

**💡 创新点**

创新点在于将问题生成与求解两角色共享同一多模态策略，通过程序执行得到可验证奖励，形成自适应的学习曲线，并通过三种推理模式（溯因、演绎、归纳）提升模型的结构化理解。

**🔧 技术方法**

核心技术包括基于程序的空间原语库（几何、拓扑、聚合）、开放词汇分割工具、可验证奖励的强化学习（RLVR）以及自我对弈的proposer–solver循环。

**📊 数据集**

使用未标注的遥感图像集进行自我生成，并在RSVQA‑HR、EarthVQA、GEOBench‑VLM等现有视觉问答基准上进行评估。

**📈 对比分析**

与零样本通用VLM（Qwen‑2.5‑VL‑7B‑Instruct、LLaVA‑1.5‑7B）以及专门的遥感VLM（GeoChat、VHM、EarthDial、RSThinker）对比，模型平均提升5.5分，尤其在面积、比较、空间关系分类和计数任务上显著超越传统基线。

**⚠️ 局限性**

局限在于工具集仅包含开放词汇分割，导致只能验证基于分割的几何与拓扑任务；分割误差会引入奖励噪声，且无法覆盖深度、坡度、道路网络连通性等更丰富的空间推理。

---

## 641. Optimizing for Fairness in Generalized Kidney Exchange: Theory and Computations

**arXiv ID:** 2605.20070 | [PDF](https://arxiv.org/pdf/2605.20070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 642. When Critics Disagree: Adaptive Reward Poisoning Attacks in RIS-Aided Wireless Control System

**arXiv ID:** 2605.20037 | [PDF](https://arxiv.org/pdf/2605.20037v1)

**作者:** Deemah H. Tashman `[一作]` (Polytechnique Montreal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并评估了一种基于SAC双Critic分歧的状态自适应奖励中毒攻击（DGRP），攻击在RIS辅助的认知无线网络中对SU发射功率和RIS相位调节的学习过程。

**💡 创新点**

创新点在于：①利用SAC双Critic之间的分歧（不确定性）作为攻击触发信号，攻击时机自适应且稀疏；②采用滚动窗口和分位数阈值实现动态阈值，避免固定时序或探索触发的局限；③通过少量有针对性的奖励腐蚀实现更高破坏性，展示了传统基准攻击不足。

**🔧 技术方法**

使用技术包括：Soft Actor‑Critic（SAC）深度强化学习框架、RIS模型（Rayleigh衰落、可变相位）、奖励中毒（bounded δ）、分位数阈值检测、滚动窗口（窗口大小 w）以及对Critic输出的灰/白盒访问。

**📊 数据集**

数据集：采用仿真生成的CRN RIS环境（随机Rayleigh通道、随机干扰阈值），实验使用10个随机种子，未使用公开数据集。

**📈 对比分析**

对比方法：与无攻击、固定时序攻击（Periodic Timing）以及探索触发攻击进行比较。结果显示，DGRP在不同RIS元素数、攻击强度 δ 与预算 p 下均导致更大幅度的SU速率下降，证明其破坏性最强。

**⚠️ 局限性**

局限性：仅在仿真环境下验证，未测试对其他DRL算法或真实物理层的适用性；攻击需对Critic输出有灰/白盒访问；未探讨对抗训练或防御策略的鲁棒性。

---

## 643. Stage-adaptive Token Selection for Efficient Omni-modal LLMs

**arXiv ID:** 2605.20035 | [PDF](https://arxiv.org/pdf/2605.20035v1)

**作者:** Zijie Xin `[一作]` (Renmin University of China), Xirong Li `[通讯]` (Renmin University of China)

**通讯引用:** 6787 | [OpenAlex ID](https://openalex.org/A5060270456)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑free、三阶段自适应词元选择方法，用于高效推理 omni‑modal 大语言模型（om‑LLM）。

**💡 创新点**

创新点包括：① 将词元压缩分为预‑LLM、内‑LLM、后‑LLM 三个阶段，分别针对不同层次的冗余与重要性；② 内‑LLM 采用指数衰减的层级保留率，并通过跨模态两级预算分配（窗口级、模态级）将查询相关性映射到 token 预算；③ 预‑LLM 结合注意力加权多样性选择（DivPrune‑style），在不需训练的情况下实现局部多样化压缩。

**🔧 技术方法**

技术手段：注意力加权多样性选择、指数衰减保留率、两级预算分配（窗口 → 模态）、查询‑指导的窗口/模态重要性评估、late‑block 完全去除非文本词元。

**📊 数据集**

使用五个音视频基准数据集：WorldSense、Daily‑Omni、OmniVideoBench、Video‑MME、LVOmniBench。

**📈 对比分析**

与 VisionZip、FastV、DyCoke、OmniZip、Random 等训练‑free 基线进行对比；在 Qwen2.5‑Omni‑7B 与 Qwen3‑Omni‑30B 上实验，10% 词元保留时保留 96.3%（30‑B）/95.5%（7‑B）性能，9.3× FLOPs 降低、4.8× prefill 加速；在 35% 保留时甚至超越全词元基线，表现出优异的效率‑性能折衷。

**⚠️ 局限性**

局限性：仅在 Qwen 系列模型上验证，未测试更大规模或不同架构的 om‑LLM；方法对 LLM 关注层的注意力分数依赖，可能对不同模型表现不一；需要手动设定 λ、τ 等超参数，虽对温度较稳健但仍需调优；后‑LLM 直接去除所有非文本词元，适用于深层已完成融合的任务，但在需要持续跨模态推理的场景中可能不适用。

---

## 644. Training Neural Networks with Optimal Double-Bayesian Learning

**arXiv ID:** 2605.20009 | [PDF](https://arxiv.org/pdf/2605.20009v1)

**作者:** Vy Bui `[一作]` (Lister Hill National Center for Biomedical Communications, National Library of Medicine, National Institutes of Health), Stefan Jaeger `[通讯]` (Lister Hill National Center for Biomedical Communications, National Library of Medicine, National Institutes of Health)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出双贝叶斯学习框架，推导出SGD的理论最优学习率（≈0.016）和动量（≈0.874），并在多任务上验证其效果。

**💡 创新点**

创新点在于通过双贝叶斯决策机制结合信息理论，首次给出学习率与动量的理论取值，并解释动量本质，证明SGD在噪声环境下优于Adam。

**🔧 技术方法**

使用双贝叶斯理论、对数基底λ、黄金比例和毕达哥拉斯恒等式推导；实验中采用SGD、Adam、CNN、DenseNet121、YOLOv8m等网络，并进行网格搜索调参。

**📊 数据集**

使用的公开数据集包括MNIST（手写数字）、TBX11K（胸片肺结核）、COVID-19肺部分割、NLM疟疾血涂片（细胞检测）。

**📈 对比分析**

通过对学习率和动量进行6×10网格搜索，在不同训练集大小和噪声水平下对比SGD与Adam。结果显示SGD在大多数任务上获得更高准确率/IoU/mAP，理论学习率常位于顶10；Adam收敛更快但泛化能力略差。

**⚠️ 局限性**

局限性包括：仅评估SGD和Adam；实验耗时约40,000小时，资源需求高；未验证对其他优化器（如RMSProp、LAMB）的适用性；对动量与学习率交互机制的细节仍待进一步研究。

---

## 645. Your Neighbors Know: Leveraging Local Neighborhoods for Backdoor Detection in Decentralized Learning

**arXiv ID:** 2605.19969 | [PDF](https://arxiv.org/pdf/2605.19969v1)

**作者:** Sayan Biswas `[一作]` (EPFL), Martijn de Vos `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种新的去中心化学习（DL）中的后门检测框架Argus，该框架允许节点在没有中央协调者的情况下协作训练模型，并能够识别潜在的后门触发器。

**💡 创新点**

Argus是首个针对去中心化学习的后门检测框架，能够在没有先验知识的情况下进行后门检测，并通过节点间的协作验证来区分真实后门和假阳性。

**🔧 技术方法**

使用了局部触发器反向工程和邻居节点间的协作交叉验证技术。

**📊 数据集**

在三个标准数据集上进行了评估，具体数据集未在摘要中列出。

**📈 对比分析**

与三种最先进的基线方法进行比较，Argus在所有设置中将攻击成功率降低至7%以下，同时保持模型的准确性在5个百分点以内，尤其在数据异质性增加时，Argus的效果更为显著。

**⚠️ 局限性**

限制在于，当前方法仅能拒绝可疑更新，未来的工作可以探索如何在保留有用信息的同时过滤或去除后门触发器。此外，语义后门触发器的研究仍然是一个未解决的问题。

---

## 646. Smooth Partial Lotteries for Stable Randomized Selection

**arXiv ID:** 2605.20069 | [PDF](https://arxiv.org/pdf/2605.20069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 647. Normative Networks for Source Separation via Local Plasticity and Dendritic Computation

**arXiv ID:** 2605.19965 | [PDF](https://arxiv.org/pdf/2605.19965v1)

**作者:** Bariscan Bozkurt `[一作]` (University College London), Rafal Bogacz `[通讯]` (University of Oxford)

**通讯引用:** 12495 | [OpenAlex ID](https://openalex.org/A5049095056)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种在线预测熵最大化（Predictive Entropy Maximization）框架，用于在已知结构域的盲源分离问题中实现可生物学的局部学习。

**💡 创新点**

核心创新在于用二阶泰勒近似代替精确的对数行列式目标，得到仅依赖协方差统计的局部可实现规则，并给出显式谱误差上界。

**🔧 技术方法**

采用在线协方差估计、两时尺度优化、局部错误驱动的前馈更新以及基于协方差迹的自适应侧向抑制等技术。

**📊 数据集**

使用合成线性混合数据、自然图像补丁以及Librosa音频三种数据集进行评估。

**📈 对比分析**

与批量CorInfoMax、ICA-InfoMax以及在线生物学可行基线（CorInfoMax Online、NSM）对比，PEM在源相关性和噪声变化下保持鲁棒，性能接近批量基线。

**⚠️ 局限性**

局限包括对超参数和混合数的敏感性、需要多次快速迭代导致计算成本高，以及在某些高相关场景下仍略逊于精确目标。

---

## 648. Take It or Leave It: Intent-Controlled Partial Optimal Transport

**arXiv ID:** 2605.20030 | [PDF](https://arxiv.org/pdf/2605.20030v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 649. CogOmniControl: Reasoning-Driven Controllable Video Generation via Creative Intent Cognition

**arXiv ID:** 2605.19995 | [PDF](https://arxiv.org/pdf/2605.19995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 650. Hunting Vulnerability Variants in AI Infra: Measurement and Reference-Driven Detection

**arXiv ID:** 2605.20051 | [PDF](https://arxiv.org/pdf/2605.20051v1)

**作者:** Tian Dong `[一作]` (University of Hong Kong), Hao Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 112277 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对688个AI infra仓库和251个公开漏洞进行测量，并基于已知漏洞建立多代理框架，对20个目标仓库进行变体检测，发现20个零日漏洞。

**💡 创新点**

提出了以已知漏洞为参考的变体审计概念，并设计了三代理协同的框架（语义建模、功能定位与验证），实现了对功能上下文相似但实现差异的变体的精确定位。

**🔧 技术方法**

使用LLM驱动的多代理系统、语义特征抽取、模块依赖建模、局部状态管理以及自动PoC生成等技术来实现跨仓库变体检测。

**📊 数据集**

使用从GitHub抓取的688个AI infra仓库、251个公开漏洞数据，以及8个参考漏洞与20个目标仓库作为实验数据集。

**📈 对比分析**

与现有工具（如Claude Code）对比，框架在31个候选中发现24个真阳性，精度和准确率最高，且在相同参考条件下消耗的token量是Claude Code的7倍，真阳性下降仅4%。

**⚠️ 局限性**

受LLM上下文窗口限制、可能出现幻觉、以及对人工验证的依赖，导致部分变体可能被遗漏或误报，且当前仅针对AI infra类开源项目，未验证跨行业通用性。

---

## 651. FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration

**arXiv ID:** 2605.20022 | [PDF](https://arxiv.org/pdf/2605.20022v1)

**作者:** Yaojie Zhang `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15532 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FlexDraft，一种在保持目标分布完整的前提下，通过块级扩散草稿与并行校验加速LLM推理的框架。

**💡 创新点**

核心创新包括：1）仅微调最后几层的注意力投影器（Attention Tuning）实现低成本的块级草稿；2）使用轻量级MLP校准草稿logits以消除奖金词不确定导致的草稿-校验不匹配；3）动态切换并行与串行草稿-校验策略，并根据草稿置信度进行Selective Verification，从而在不同批量规模下均保持高效。

**🔧 技术方法**

采用块级扩散草稿、Attention Tuning、Bonus‑guided Calibration、Selective Verification与Batch‑Adaptive Execution等技术；模型架构基于Qwen3系列的Transformer。

**📊 数据集**

训练使用300K条样本的open‑perfectblend数据集；实验在Qwen3‑8B等模型上进行。

**📈 对比分析**

与DFlash、EAGLE‑3、DART、BiTA、Apple MTP等主流（且同样保持目标分布）推理加速方法对比，FlexDraft在Qwen3‑8B上平均可获得4.59×的速度提升，且在所有任务（代码生成、数学推理、通用聊天）上保持或提升了接受率与质量。

**⚠️ 局限性**

局限性主要包括：仍需额外训练微调Attention投影器；在极大批量规模下仍可能出现重叠计算瓶颈；目前实现主要针对Qwen3系列，需进一步验证跨模型通用性。

---

## 652. Fine-Tuning Without Forgetting via Loss-Adaptive Learning Rates

**arXiv ID:** 2605.20005 | [PDF](https://arxiv.org/pdf/2605.20005v1)

**作者:** Parjanya Prajakta Prashant `[一作]` (University of California San Diego), Babak Salimi `[通讯]` (University of California San Diego)

**通讯引用:** 640 | [OpenAlex ID](https://openalex.org/A5103209063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自适应学习率调度方法，旨在减轻大语言模型在微调过程中出现的灾难性遗忘现象。

**💡 创新点**

通过将学习率与当前训练损失的平方根成反比，控制每一步的遗忘，从而在不抑制高损失标记的情况下减少遗忘。

**🔧 技术方法**

使用了一种基于损失的自适应学习率调度方法，称为FINCH。

**📊 数据集**

在知识获取、科学推理和低资源语言适应等基准上进行了评估，使用的数据集包括TOFU、Chemistry L-3和Galician Alpaca。

**📈 对比分析**

与标准微调方法相比，FINCH在保持任务性能的同时，平均减少了93%的遗忘，且在多个基准上表现出色。

**⚠️ 局限性**

尽管FINCH显著减少了遗忘，但在校准方面仍未完全恢复到预训练模型的水平，且实验仅限于最大8B参数的模型。

---

## 653. Minimalist Visual Inertial Odometry

**arXiv ID:** 2605.19990 | [PDF](https://arxiv.org/pdf/2605.19990v1)

**作者:** Francesco Pasti `[一作]` (University of Padua), Shree K. Nayar `[通讯]` (Columbia University)

**通讯引用:** 38803 | [OpenAlex ID](https://openalex.org/A5051975921)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了仅包含四个光学掩模光电探测器的极简速度传感器，并与IMU融合，完成差速驱动机器人平面里程计。

**💡 创新点**

创新点在于：① 将Gabor掩模的物理光学特性与深度学习网络（TCN）端到端协同优化；② 通过物理仿真直接学习掩模参数和速度回归网络，实现对多种地面纹理与高度变化的鲁棒性；③ 仅用四个像素即可替代传统高分辨率摄像头完成VIO级别的里程计。

**🔧 技术方法**

采用光学Gabor掩模、Temporal Convolutional Network、可微分物理仿真、注意力池化、联合训练、IMU角速度融合、低功耗硬件实现。

**📊 数据集**

使用Matador纹理库（约7200张不同材质图像）和TartanGround运动轨迹数据集（80公里、12小时、最高速度5 m/s）进行仿真训练；实测时使用Intel RealSense D455的VIO轨迹作为参考。

**📈 对比分析**

与差速轮编码器及编码器+IMU基线对比，室内/室外A‑TE分别为0.28–0.42 m、漂移率0.6–0.85%，显著优于编码器基线（0.74–0.92 m、1.6–1.7%）。更新频率从1 kHz降至30 Hz对性能影响微乎其微。

**⚠️ 局限性**

局限性包括：对光照、粗糙地面或高度极端变化的鲁棒性下降；仅针对差速驱动平台，难以直接扩展到全向或多自由度机器人；依赖仿真生成的训练数据，若真实场景与仿真差异过大可能导致泛化不足。

---

## 654. A conceptual framework for learning to listen by reward: Curiosity-driven search for novel sources

**arXiv ID:** 2605.19984 | [PDF](https://arxiv.org/pdf/2605.19984v1)

**作者:** Andreas Triantafyllopoulos `[一作]` (Technical University of Munich), Björn W. Schuller `[通讯]` (Technical University of Munich)

**通讯引用:** 55151 | [OpenAlex ID](https://openalex.org/A5043060302)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于奖励驱动的学习听觉框架，让代理通过奖励探索来学习定位新的声音源，并实现了一个基于深度 Q 学习的原型。

**💡 创新点**

引入了以人类幼儿听觉导航为灵感的奖励机制，聚焦纯音频感知并定义可通用的探索目标，同时展示了状态记忆网络在此任务上的优势。

**🔧 技术方法**

采用深度 Q 学习、CNN6 与 CNN-Transformer 架构、pyroomacoustics 进行房间声学模拟以及经验回放与延迟目标网络。

**📊 数据集**

使用仿真房间（shoebox）中随机放置的单一声音源和代理的声音记录，未使用公开音频数据集，而是基于模拟环境生成数据。

**📈 对比分析**

与随机策略和无记忆 CNN6 对比，CNN-Transformer 在 1,000 次评估试验中准确率 74%、到达率 52% 和平均奖励 0.89，显著优于随机策略的 41%/8%/-0.89 及 CNN6 的 68%/36%/0.08。

**⚠️ 局限性**

实验仅考虑单一静止声源、固定声源高度和固定模拟音频传播模型，且未处理移动麦克风/源、复杂几何或真实环境的噪声，导致泛化受限。

---

## 655. SphericalDreamer: Generating Navigable Immersive 3D Worlds with Panorama Fusion

**arXiv ID:** 2605.19974 | [PDF](https://arxiv.org/pdf/2605.19974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 656. Cardiac fat segmentation using computed tomography and an image-to-image conditional generative adversarial neural network

**arXiv ID:** 2605.20064 | [PDF](https://arxiv.org/pdf/2605.20064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 657. InterLight: Leveraging Intrinsic Illumination Priors for Low-Light Image Enhancement

**arXiv ID:** 2605.19982 | [PDF](https://arxiv.org/pdf/2605.19982v1)

**作者:** Ziqi Wang `[一作]` (Wuhan University), Huan Zhang `[通讯]` (Guangdong University of Technology)

**通讯引用:** 7638 | [OpenAlex ID](https://openalex.org/A5046483460)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 InterLight 框架，用物理引导的数据增强、适应性降解先验生成以及亮度门控的内在记忆机制，针对低光图像实现高质量增强。

**💡 创新点**

创新点包括：① 将传感器级光照响应先验通过 Physics‑Guided Augmentation 直接嵌入网络；② 通过可学习的降解字典生成全局降解提示，引导颜色分支的自适应融合；③ 设计亮度门控的内在记忆（LGIM）在暗区强制补偿信息缺失，同时保持亮区细节。

**🔧 技术方法**

核心技术包括 HVI 颜色空间分解、物理引导增强 (PGA)、自监督一致性 (PIC)、自适应降解先验生成 (ADPG)、跨注意力融合 (PRFB) 与亮度门控记忆模块。

**📊 数据集**

在 LOL‑v1、LOL‑v2、SICE、SID（Sony‑Total‑Dark）以及 LSRW‑Huawei 等公开低光图像数据集上进行评估。

**📈 对比分析**

与 RetinexNet、KinD、Zero‑DCE、LLFormer、Retinexformer、CIDNet、CWNet 等 SOTA 方法进行对比，InterLight 在 LOL‑v1、LOL‑v2‑Real、LOL‑v2‑Syn、SICE、SID、LSRW‑Huawei 上均取得最高或近乎最高的 PSNR/SSIM，提升幅度可达 0.97 dB（PSNR）和 0.05（SSIM）左右。

**⚠️ 局限性**

局部双分支结构与内在记忆引入额外计算量，且物理引导增强假设为线性降解，可能对极端非线性噪声的适应性有限；未来需关注模型压缩与视频低光增强的时序一致性。

---

## 658. CEER: Compliant End-Effector and Root Control as a Unified Interface for Hierarchical Humanoid Loco-Manipulation

**arXiv ID:** 2605.19981 | [PDF](https://arxiv.org/pdf/2605.19981v1)

**作者:** Xinyuan Luo `[一作]` (Duke University), Xianyi Cheng `[通讯]` (Duke University)

**通讯引用:** 273 | [OpenAlex ID](https://openalex.org/A5100943262)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可合成的端执行器–根（EE-root）控制抽象（CEER），在层次化规划框架中实现了人形机动-操作的模块化全身控制，支持不同规划器和技能的无缝集成。

**💡 创新点**

创新点在于将根运动与末端执行器姿态统一为任务空间命令，构建可解释的接口；并通过教师-学生框架将运动模仿策略蒸馏为仅依赖EE-root命令的低层控制器，实现了可插拔、无关机器人构型的全身控制。

**🔧 技术方法**

采用阻抗控制模型、PPO强化学习与教师-学生蒸馏、基于AMASS数据的人类运动模仿训练、LLM任务管理、Diffusion学习等技术，构建三层层次化系统。

**📊 数据集**

主要使用AMASS人类运动捕捉数据集进行训练，并在Isaac Lab仿真环境中进行多达16,384个并行环境的学习；在Unitree G1硬件上进行真实世界实验。

**📈 对比分析**

与三种基线（教师-学生无目标估计器、无外力RL、端到端RL）比较，CEER在末端执行器位置RMSE 3.3 cm、姿态误差0.32 rad，且低jerk 4.5e3，显著优于对比方法；在真实硬件上完成多种接触丰富的操作任务，模拟中单物体任务成功率达70%+，长周期任务与人类操作相比成功率相当且耗时更短。

**⚠️ 局限性**

局限性包括阻抗参数固定缺乏自适应刚度调节；缺少对臂摆动和脚/肘等更灵活接触点的自动生成；在多步骤任务中仍受LLM规划误差影响，导致失败率仍有一定比例。

---

## 659. D$^3$-Subsidy: Online and Sequential Driver Subsidy Decision-Making for Large-Scale Ride-Hailing Market

**arXiv ID:** 2605.20036 | [PDF](https://arxiv.org/pdf/2605.20036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 660. Mind Your Moras: Orthography-Aware Error Analysis of Neural Japanese Morphological Generation

**arXiv ID:** 2605.20043 | [PDF](https://arxiv.org/pdf/2605.20043v1)

**作者:** Wen Zhang `[一作]` `[通讯]`, Wen Zhang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对日本汉字假名混写的过去式形态生成进行正字法意识错误分析，聚焦于假名的语音形态学特征；

**💡 创新点**

提出七类结构化错误分类法，揭示模型在小っ（促音）处理上的系统性弱点，并强调正字法意识评估的重要性；

**🔧 技术方法**

使用基于Transformer的字符级序列到序列模型，分别采用SIGMORPHON 2020的官方基线和2023的lemma‑split训练框架；

**📊 数据集**

利用将所有词形转为假名的SIGMORPHON 2020/2023日语动词形态数据集，包含2,503个Godan、1,298个Ichidan以及157个其他不规则动词；

**📈 对比分析**

在相同训练设置下对两模型做精确匹配准确率比较，分别达到97.97%和97.17%，但通过错误分析发现75–80%的残余错误集中在促音插入/遗漏上；

**⚠️ 局限性**

仅关注过去式、仅使用假名输入、未能充分分离正字法与词频影响，且未检验其他形态变体（否定、使役等）的错误模式。

---

## 661. Towards LLM-Assisted Architecture Recovery for Real-World ROS~2 Systems: An Agent-Based Multi-Level Approach to Hierarchical Structural Architecture Reconstruction

**arXiv ID:** 2605.20055 | [PDF](https://arxiv.org/pdf/2605.20055v1)

**作者:** Dominique Briechle `[一作]` (Clausthal University of Technology), Meng Zhang `[通讯]` (Clausthal University of Technology)

**通讯引用:** 24155 | [OpenAlex ID](https://openalex.org/A5100437827)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了分阶段蓝图指导的LLM辅助架构恢复流程，能够从ROS 2系统的源码、Launch文件等异构工件中自动重建层次化结构化架构模型。

**💡 创新点**

创新点在于：①引入显式中间工件（Atomic ROS Node 列表与 Launch File 依赖描述）让LLM拥有完整的上下文；②改进提示约束（prompt contract）提升生成的一致性与可控性；③通过分阶段处理实现多层级（源代码层、运行时实例层、系统层）统一恢复。

**🔧 技术方法**

使用了大型语言模型（LLM）配合 Prompt Engineering、JSON中间表示、PlantUML渲染及基于UML的蓝图约束，实现结构化合成与验证。

**📊 数据集**

使用的案例数据集为BrickByBrick 自动拆解机器人系统（约1,500行Python代码，10个ROS 2节点，1个Launch文件），并与先前的对照仓库进行对比。

**📈 对比分析**

评估方法：将恢复得到的PlantUML模型与人工构建的参考模型逐元素对比，计算精确率、召回率和F1；在Atomic层面得到1.0的精确率/召回率/ F1；在Composed层面得到精确率1.0、召回率0.95、F1 0.98，表明子系统级恢复显著提升。

**⚠️ 局限性**

局限性：子系统级别仍有5%的召回缺失，主要因Launch语义、命名空间传播、重映射和重复实例化等复杂场景难以完整解析；未来需更细粒度建模、引入运行时行为证据并完善对复杂Launch结构的支持。

---

## 662. AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration

**arXiv ID:** 2605.20025 | [PDF](https://arxiv.org/pdf/2605.20025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 663. On the exact decoding error probability exponent of the random coding on BSC

**arXiv ID:** 2605.19991 | [PDF](https://arxiv.org/pdf/2605.19991v1)

**作者:** Marat V. Burnashev `[一作]` `[通讯]`, Marat V. Burnashev

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在二进制对称信道（BSC）上，随机编码的误码概率指数，并给出了该指数在所有速率范围内的精确表达式。

**💡 创新点**

创新点在于提出新的临界速率 R_crit(p)，并利用对随机求和分布的全新分析，推导出完整速率区间内的精确随机编码指数，从而弥补了以往只能给出下界或参数化形式的不足。

**🔧 技术方法**

主要技术包括：对随机码字的 Hamming 距离分布进行大偏差分析；利用熵函数和对数变换推导概率分布；对随机求和的集中性进行严格证明；以及对不同速率区间进行分段优化，得到闭式指数。

**📊 数据集**

该研究为理论推导，未使用具体数据集；所有结果均通过概率论和信息理论的解析方法得到。

**📈 对比分析**

与 Gallager 等人之前给出的下界或参数化表达式比较，本文给出的指数在整个速率区间内与上界一致，证明了随机编码在 BSC 上的误码概率指数可以被精确估计，且性能与已知下界完全匹配。

**⚠️ 局限性**

局限性：①仅适用于二进制对称信道；②对有限长度码的误码率只能给出极限近似；③尚未探讨非随机或其他类型编码的情况。

---

## 664. A Case for Agentic Tuning: From Documentation to Action in PostgreSQL

**arXiv ID:** 2605.19988 | [PDF](https://arxiv.org/pdf/2605.19988v1)

**作者:** Hongyu Lin `[一作]` (Institute of Software, Chinese Academy of Sciences), Haibo Chen `[通讯]` (Key Laboratory of System Software (Chinese Academy of Sciences))

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现 PerfEvolve：一种将专家调优方法转化为可执行的流程知识的工具，使 LLM‑based 代理能够在 PostgreSQL 上通过在线剖析、敏感度排序、相关性拓扑发现以及联合优化等步骤实现高效调优。

**💡 创新点**

创新点在于：① 把传统静态文档的“结果”转变为可复现的“过程”知识；② 两阶段离线–在线工作流，其中离线阶段通过敏感度分析与阶乘方差分析构建参数拓扑；③ 将这些过程编译成可执行的技能图，为 LLM 代理提供完整的自检与决策逻辑；④ 通过流程化的调优实现跨硬件、跨负载的迁移与鲁棒性。

**🔧 技术方法**

主要技术包括：敏感度扫描（Top‑k CV）、两阶段方差分析（ANOVA）进行相关性检验、图论实现拓扑分解、基于 LLM 的技能执行与自检、Bayesian 优化（SMAC）、单次推理（E2ETune）等；数据收集则通过 BenchBase 的 TPC‑C（读写分布）与 TPC‑H 在 VM 与高端服务器上完成。

**📊 数据集**

使用的数据集为 PostgreSQL v16 的 300+ 参数集，采用 BenchBase 提供的 TPC‑C‑r（读密集 OLTP）、TPC‑C‑w（写密集 OLTP）与 TPC‑H（分析型 OLAP）三种基准；实验环境包括 150 台 2 vCPU/8GB SSD 虚拟机（离线与匹配部署）和 192 核/1TB RAM/8×H100 GPU 的高端服务器（跨硬件迁移）。

**📈 对比分析**

对比方法包括传统静态规则（PG‑Official、PGTune）、基于 LLM 的 Bayesian 优化 GPTuner、单次推理 LLM E2ETune，以及它们分别在原始与注入 PerfEvolve 知识后的版本；结果显示在匹配环境下 PerfEvolve 能提升 TPC‑C‑r、TPC‑C‑w、TPC‑H 通过 10%–35%（最高 35.2%）的吞吐量增益；在跨硬件迁移中，GPTuner+ 在原版下几乎崩溃，而注入后恢复至近默认水平并取得 58.9% 的恢复增益。

**⚠️ 局限性**

局限性包括：① 离线剖析需要约 11,000 次基准跑，成本较高；② 需要与目标部署相近的代表性工作负载和硬件，否则过程可能失效；③ 仅捕获对称低阶（pairwise/低维）相关性，可能忽略高阶交互；④ 代理执行仍可能出现解析或决策错误，需额外的鲁棒性与回滚机制；⑤ 目前验证集中在 PostgreSQL，迁移至其他系统需针对不同观测与安全约束做适配。

---

## 665. RECIPE: Procedural Planning via Grounding in Instructional Video

**arXiv ID:** 2605.19976 | [PDF](https://arxiv.org/pdf/2605.19976v1)

**作者:** Luigi Seminara `[一作]` (Northeastern University), Lorenzo Torresani `[通讯]` (Northeastern University)

**通讯引用:** 25637 | [OpenAlex ID](https://openalex.org/A5082736347)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉规划的强化学习框架，将大规模噪声视频语料库用作验证器而非标签源，训练模型生成自然语言步骤；

**💡 创新点**

核心创新在于将视频‑文本对齐作为奖励信号，利用“grounding-as-verification”避免伪标签噪声，支持多种有效步骤序列；

**🔧 技术方法**

技术包括两阶段文本对齐（单调检索+全局 Needleman–Wunsch）、历史基线校正、相对进展归一化、GRPO 强化学习与 LoRA 微调；

**📊 数据集**

使用多语料：小规模标注数据（CrossTask、COIN、CaptainCook4D、EgoProceL）和大规模 HowToCaption（HowTo100M）作为验证集；

**📈 对比分析**

与基准模型（SFT、伪标签SFT、VidAssist 等）比较，-RL 在三种模型规模（0.5B、3B、7B）下在7个程序规划基准上实现宏观准确率提升 7–16 分，且在零样本场景中保持竞争优势；

**⚠️ 局限性**

局限性包括对单一参考评价的依赖、仍可能出现空间/因果错误、以及对文本‑语音对齐的噪声未完全消除。

---

## 666. Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes

**arXiv ID:** 2605.19966 | [PDF](https://arxiv.org/pdf/2605.19966v1)

**作者:** Mohammed Alshaalan `[一作]` (University College London), Miguel R. D. Rodrigues `[通讯]` (University College London)

**通讯引用:** 6730 | [OpenAlex ID](https://openalex.org/A5044634366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于在线变化点检测的熵统计方法CPD Online，用于检测并定位LLM对抗性后缀。

**💡 创新点**

将Token级熵流视为时间序列，并利用系统提示的熵基线与一侧CUSUM实现无训练、在线、模型无关的检测与定位。

**🔧 技术方法**

使用稳健基线设定（中位数+MAD）对熵进行标准化，采用Page-CUSUM检测，离线阈值调优，并与LLaMA Guard结合门控。

**📊 数据集**

构造了1,012个优化后缀攻击（GCG、AutoDAN、AdvPrompter、BEAST、AutoDAN‑HGA）和1,012个匹配PP的正例，数据来源为TyDiQA与OpenOrca。

**📈 对比分析**

与全局PP、窗口PP以及LLaMA Guard对比，CPD在六款LLM上Prompt F1均超过最优WPP，AUROC在五款上匹配或提升，并在门控下显著降低Guard调用率。

**⚠️ 局限性**

依赖Token级概率，无法在闭源API下直接使用；对抗自适应攻击可能降低效果；需要事先校准且在分布漂移下可能失效；无理论保证检测延迟/误报；仅针对后缀攻击，未覆盖前缀或间接注入。

---

## 667. Rewarding Beliefs, Not Actions: Consistency-Guided Credit Assignment for Long-Horizon Agents

**arXiv ID:** 2605.20061 | [PDF](https://arxiv.org/pdf/2605.20061v1)

**作者:** Wenjie Tang `[一作]` (National University Of Defense Technology), Yuan Zhou `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReBel框架，用显式信念表示、信念一致性自监督和信念锚定步级优势，改进RLVR在部分可观测长时任务中的决策质量。

**💡 创新点**

创新点在于将过程级监督转化为密集的信念一致性奖励，并利用信念锚定的组策略解决信念漂移与信用分配问题，实现了更稳健的奖励信号和更低方差的优势估计。

**🔧 技术方法**

采用Belief‑Think‑Action结构化生成、密集信念一致性监督、Belief‑Anchor 步级优势、PPO‑式优化，并在Qwen2.5‑1.5B‑Instruct大模型上实现。

**📊 数据集**

使用 ALFWorld 与 WebShop 两个部分可观测长时基准数据集进行评估。

**📈 对比分析**

与 episode‑level GRPO 及 step‑level GiGPO_w/o std 对比，ReBel 在 ALFWorld 成功率提升约20.4个百分点、WebShop 提升约18.3个百分点，样本效率提升 2.1×，并在最难任务上表现最为突出。

**⚠️ 局限性**

局限性包括对可验证谓词和观测掩码的依赖、对极端长序列或完全不可观测环境的泛化仍有限，以及需要结构化输出格式和 SFT 预热，尚未在更大模型或更复杂任务上充分验证。

---

## 668. Language Mutations Sustain the Persistences of Conspiracy Theories on Social Media

**arXiv ID:** 2605.20050 | [PDF](https://arxiv.org/pdf/2605.20050v1)

**作者:** Calvin Yixiang Cheng `[一作]` (University of Oxford), Scott A. Hale `[通讯]` (University of Oxford)

**通讯引用:** 3609 | [OpenAlex ID](https://openalex.org/A5029882049)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对X平台上三年COVID‑19相关的阴谋论帖子进行聚类，提取语义、心理语言学属性与演员-动作-目标（AAT）三种层面的语言突变，并通过生存分析评估突变对阴谋论传播寿命的影响。

**💡 创新点**

首次系统量化语言突变（语义漂移、心理语言学变化、AAT结构变化）与阴谋论持久扩散的关联；揭示两种突变模式（简化与同化）及其对传播持续性的预测意义。

**🔧 技术方法**

技术手段包括：句子嵌入（all‑mpnet‑base‑v2）+LSH聚类；LIWC22词典进行心理语言学属性计数；LLM（gpt‑4o‑mini）辅助提取AAT；K‑Means聚类与MMR采样进行AAT类别归一；生存分析（Kaplan–Meier、Weibull AFT）与多元回归评估突变效应。

**📊 数据集**

使用了446,829条来自X（原Twitter）平台的阴谋论相关推文，时间跨度为2020‑2022年三年；通过正则表达式、LLM分类器（GPT‑4o‑mini）进行内容筛选与验证。

**📈 对比分析**

采用Kaplan–Meier曲线对比突变组与非突变组的生存概率；通过Weibull加速失败时间模型量化突变对寿命的影响。结果显示：早期语义漂移可使寿命延长27%；心理语言学属性突变使寿命增长2–3倍；AAT突变亦显著提升寿命，单独三种变动各可使寿命增加约3倍，组合效应最高可达1.88倍。

**⚠️ 局限性**

局限性：仅研究单一平台X；仅聚焦于“声称”层面的突变，忽略跨声称的演化；LIWC和AAT聚类方法可能低估突变细节；未考虑AAT关系链与不同平台算法的差异；数据来源受平台采样与API限制，可能存在偏差。

---

## 669. Does Code Cleanliness Affect Coding Agents? A Controlled Minimal-Pair Study

**arXiv ID:** 2605.20049 | [PDF](https://arxiv.org/pdf/2605.20049v1)

**作者:** Priyansh Trivedi `[一作]` (SonarSource), Olivier Schmitt `[通讯]` (SonarSource)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了六对“最小对照”仓库（同构、依赖、外部行为相同，仅清洁度不同），并在其中设计了33个编码任务，对Claude Code在两侧的表现进行多次实验，比较任务完成率与资源消耗。

**💡 创新点**

首次在保持模型与任务不变的前提下，系统性地研究代码清洁度对自主编码代理行为的影响，并引入了基于SonarQube规则违例与认知复杂度的清洁度度量与最小对照构造协议。

**🔧 技术方法**

利用SonarQube（default quality gate）进行静态分析；构建两条流水线（Slopify降级、Vibeclean清洁）自动生成仓库对；使用Claude Code（Claude Sonnet 4.6）作为代理，Harbor框架作为评测框架；记录10种指标（Passrate、输入/输出token、文件读写、重访等）。

**📊 数据集**

六个公开/私有仓库（Java/Python混合），共33个任务；每个任务在两侧各运行10次，共660次实验。

**📈 对比分析**

比较方法：对每个任务在两侧计算每项指标的平均值，采用中位数阈值离群过滤后求微平均差值。结果显示：任务完成率基本不变（-0.9pp），但清洁代码侧输入/输出token分别下降7.1%/8.5%，文件重访下降34%；不同任务轨道表现存在“紧张”效应，单模块任务受益更显著。

**⚠️ 局限性**

局限性：仅在Claude Sonnet 4.6与Claude Code环境下测试；未覆盖多种模型/调度；仅使用token作为成本度量，未换算为美元；仅评估隐藏测试，未检查全库测试或代理输出的清洁度；对任务和仓库的人工选择可能带来偏差；未探讨长期迭代中的累积效应。

---

## 670. Taking Cryptography Out of the Data Path via Near-Memory Processing in DRAM

**arXiv ID:** 2605.20047 | [PDF](https://arxiv.org/pdf/2605.20047v1)

**作者:** Nicola Barcarolo `[一作]` (University of Trento), Flavio Vella `[通讯]` (University of Trento)

**通讯引用:** 447 | [OpenAlex ID](https://openalex.org/A5011440366)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在真实的UPMEM PIM系统上实现并评估AES‑128和SHA‑256加密/哈希算法，探讨多核心、多Rank并行度对性能的影响。

**💡 创新点**

首次在实测硬件上完成SHA‑256的PIM实现，并通过大规模并行和异步Rank传输/执行两种模型，系统化展示了PIM在内存受限密码算法中的加速潜力。

**🔧 技术方法**

采用UPMEM SDK 2023.2.0（C/ Rust 编译器，LLVM12）、DPU自定义ISA、Tasklet并行模型、异步CPU↔Rank数据传输与Rank级别异步启动。

**📊 数据集**

AES使用8 MB随机明文块；SHA‑256使用1024条32 KB消息（共32 MB）进行并行哈希，评估多Rank情况下的弱/强规模。

**📈 对比分析**

将PIM实现与非PIM软件实现（CPU、AES‑NI）进行对比。结果显示：单Rank下PIM性能低于高端CPU，但随着Rank数增多（最高40 Rank）并利用异步Rank传输，可获得相对CPU 10‑30×的加速；AES‑NI仍优于PIM，但PIM在内存密集型场景显示显著优势。

**⚠️ 局限性**

主要限制包括：CPU↔DPU的数据传输瓶颈（需异步多Rank缓解）、DPU仅支持32 位位运算、缺乏跨DPU通信、对64 位/浮点运算支持不足，且在单Rank上仍无法超过传统CPU实现。

---

## 671. Training-Free Bayesian Filtering with Generative Emulators

**arXiv ID:** 2605.20028 | [PDF](https://arxiv.org/pdf/2605.20028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 672. LLM Benchmark Datasets Should Be Contamination-Resistant

**arXiv ID:** 2605.19999 | [PDF](https://arxiv.org/pdf/2605.19999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 673. Journeys of Parents with LGBTQ+ Children: How Trauma and Healing Reshape Identity and (Mis)Informating Practices

**arXiv ID:** 2605.20024 | [PDF](https://arxiv.org/pdf/2605.20024v1)

**作者:** Soonho Kwon `[一作]` (Georgia Institute of Technology), Younah Kang `[通讯]` (Yonsei University)

**通讯引用:** 2055 | [OpenAlex ID](https://openalex.org/A5088085186)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对10位支持其 LGBTQ+ 子女的韩国父母进行深度访谈，探究他们在发现子女身份后经历的情感断裂、信息搜索、错误信息处理与身份重塑，并分析这一过程如何改变他们的信息获取与传播行为。

**💡 创新点**

首次将“情感创伤‑恢复‑身份重塑”框架与关怀伦理中的“倾听”视角结合，用以解释父母如何从被动接受错误信息转向主动批判与抵制错误信息，并揭示父母在亲子关系与社会网络中的信息行动如何成为支持 LGBTQ+ 共同体的关键力量。

**🔧 技术方法**

本研究未采用技术工具，而是以质性研究方法（半结构化访谈、反思性主题分析）进行数据收集与分析；重点强调人文设计与社交技术的可能介入。

**📊 数据集**

数据集为10位父母的访谈记录，涵盖年龄、性别、子女性别认同、子女年龄及知晓时间等信息，构成对韩国父母经验的深度描述。

**📈 对比分析**

由于研究目标是解释性与探究性，未设置实验对照或性能指标，亦未进行方法比较；研究通过反思性主题分析生成理论性洞见。

**⚠️ 局限性**

局限性包括样本规模小且仅包含支持父母，未能覆盖缺乏支持的父母或更广泛的文化背景；研究依赖自述数据，可能存在回忆偏差；未能验证因果关系，结果主要适用于类似社会政治环境下的韩国父母。

---

## 674. When Skills Don't Help: A Negative Result on Procedural Knowledge for Tool-Grounded Agents in Offensive Cybersecurity

**arXiv ID:** 2605.20023 | [PDF](https://arxiv.org/pdf/2605.20023v1)

**作者:** Samuel Jacob Chacko `[一作]` (Florida State University), Xiuwen Liu `[通讯]` (Florida State University)

**通讯引用:** 8388 | [OpenAlex ID](https://openalex.org/A5102867647)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对一个MCP基准的CTF代理进行180次实验，重新解释为四层Agent Skills消融实验，评估技能对攻击成功率的影响。

**💡 创新点**

首次将高反馈带宽环境与Skills效益联系起来，提出“反馈带宽假设”，解释Skills在不同域的异质性。

**🔧 技术方法**

使用MCP工具接口、Claude Sonnet 4.5 LLM、结构化工具反馈与JSON Schema验证，结合统计检验（χ²、Cochran–Armitage、Cohen's h）。

**📊 数据集**

在15个离散CTF挑战（内存破坏、逆向、Web攻击、密码学）上共180条轨迹，文档量分为55、1478、1976、4147行四个级别。

**📈 对比分析**

将四个Skills层与无技能基线对比，统计成功率提升仅为+8.9pp，差异不显著，表明技能增益低于SkillsBench平均值。

**⚠️ 局限性**

样本量有限、仅使用单一模型、缺乏跨环境反馈带宽验证，结果仅展示负面或微弱效应。

---

## 675. Precise and Simple Audio-to-Score Alignment

**arXiv ID:** 2605.20014 | [PDF](https://arxiv.org/pdf/2605.20014v1)

**作者:** Silvan Peter `[一作]` (Johannes Kepler University), Gerhard Widmer `[通讯]` (Johannes Kepler University)

**通讯引用:** 11290 | [OpenAlex ID](https://openalex.org/A5003768123)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于动态规划的音频到乐谱对齐算法，直接使用音频的起始激活和谱激活与乐谱中的音符进行匹配。

**💡 创新点**

创新点在于无需转录或合成音频，直接将音频特征与符号级乐谱信息对齐，并结合动态节拍周期估计，实现更精确且灵活的对齐。

**🔧 技术方法**

采用音频特征提取（IIR Butterworth 滤波器组、superflux 上升算法、谱特征）和动态规划成本函数（起始、谱能量、拉伸项），以及持续更新的 beat 周期估计。

**📊 数据集**

使用大型独奏钢琴演奏数据集(nASAP)，包含 300 多段录音。

**📈 对比分析**

与传统基于合成乐谱的音频对齐（DTW）相比，平均误差从 135 ms 降至 86 ms，<200 ms 误差占比从 87.7% 提升至 95.2%；相比 MIDI‑to‑score 的符号对齐仍略逊一筹。

**⚠️ 局限性**

限制包括精度仍低于完美转录的符号对齐、对不同音色/曲风的泛化需要进一步验证，以及需要手动调节参数以权衡速度与精度。

---

## 676. Learning with Foresight: Enhancing Neural Routing Policy via Multi-Node Lookahead Prediction

**arXiv ID:** 2605.19975 | [PDF](https://arxiv.org/pdf/2605.19975v1)

**作者:** Xia Jiang `[一作]` (Eindhoven University of Technology), Yingqian Zhang `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 5270 | [OpenAlex ID](https://openalex.org/A5004461578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在监督学习的神经路由策略中引入多节点前瞻预测（MnLP），让模型在训练时同时预测多步未来节点，以提升长时程规划能力。

**💡 创新点**

创新点在于：①仅在训练阶段使用可抛弃的多深度前瞻模块，避免推理时增加成本；②在损失函数中加入多层辅助监督，强化模型对多步决策的信用分配；③通过共享编码器实现高效的多步监督，显著提升泛化与解优度。

**🔧 技术方法**

采用Transformer基础的LEHD架构，加入多头注意力与FFN，使用交叉熵的多深度辅助损失和线性调度的权重γ进行训练；同时对CVRP加入车辆容量信息。

**📊 数据集**

使用标准的均匀分布TSP/CVRP训练集（100/200/500/1000节点），并在TSPLib与CVRPLib、真实分布的旋转/爆炸分布上评估；训练集为10^6实例，测试集为10^4/1k实例。

**📈 对比分析**

与经典求解器（Concorde、LKH3、HGS、OR‑Tools）以及多种神经策略（ELG、LEHD、INViT、DGL、RELD）和RL/SL方法（POMO、SA‑DABL、BOPO）对比，MnLP在大规模实例上将TSP1000的最优性差距从3.56%降至2.85%（约20%提升），CVRP1000从6.20%降至6.00%（约3%提升），并在跨分布和真实数据集上均保持领先。

**⚠️ 局限性**

局限性包括：①仅在监督学习框架下验证，未探究RL的长程信用分配；②前瞻模块仅在训练阶段使用，训练成本略增；③对非欧氏度量、约束更复杂的VRP类型尚未充分评估；④需手动调节权重γ和模块深度，适配性受限。

---

## 677. Block-Sphere Vector Quantization

**arXiv ID:** 2605.19972 | [PDF](https://arxiv.org/pdf/2605.19972v1)

**作者:** Heesang Ann `[一作]` (Seoul National University), Min-hwan Oh `[通讯]` (Seoul National University)

**通讯引用:** 73133 | [OpenAlex ID](https://openalex.org/A5100447410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对旋转基向量量化器（Eden、RabitQ、Block‑Sphere 等）进行统一理论比较，并提出 Block‑Sphere 量化器以更好利用随机旋转后向量的球面几何，实现更低的 MSE 与内积失真；

**💡 创新点**

创新点在于：①构建统一的评价框架，揭示不同量化器在 MSE、期望内积失真和高概率比特复杂度上的相对优势；②提出 Block‑Sphere 量化器，通过块级球面分布优化质心，理论上逼近 Shannon 下界并在实验中显著优于传统坐标级量化；

**🔧 技术方法**

技术手段包括：随机 Haar 旋转、坐标/块级 Lloyd‑Max 量化、球面分布解析、理论误差上界推导、近似最近质心搜索以及 KV‑cache 量化实现；

**📊 数据集**

数据集涵盖：DBpedia 实体嵌入（1536 维）、GloVe（200 维）、OpenAI3/DBpedia（1536/3072 维）用于距离与召回评估；以及 Llama‑3.1‑8B‑Instruct 的 KV‑cache 量化在 Needle‑In‑A‑Haystack 与 LongBench‑E 上的推理评测；

**📈 对比分析**

对比方法为：MSE 统计、内积失真期望、Recall@1@k 召回率、KV‑cache 量化下的 LLM 推理分数；实验结果显示 Block‑Sphere 在 MSE 与内积失真上均优于 Eden、RabitQ、Block‑Quant，召回率提升 3–5%，LLM 量化误差差距几乎可忽略，接近全精度；

**⚠️ 局限性**

局限性包括：块级搜索的计算开销随块大小增大显著增加，需要近似最近质心搜索来控制；对超高维/极低比特宽度的适应性仍待进一步验证；旋转随机性在 LLM 推理中仍产生一定波动；

---

## 678. Long-term Power Grid Planning via Answer Set Programming

**arXiv ID:** 2605.20172 | [PDF](https://arxiv.org/pdf/2605.20172v1)

**作者:** Antonio Ielo `[一作]` (University of Calabria), Mauro Vallati `[通讯]` (University of Huddersfield)

**通讯引用:** 2933 | [OpenAlex ID](https://openalex.org/A5086585008)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于 Answer Set Programming 的自动化电网长期规划方法，用于计算从初始配置到目标配置的操作序列，并保证所有中间状态满足规划规则。

**💡 创新点**

创新之处在于将 ASP 规划与电网重构约束相结合，设计了完整的约束模型（径向性、可重构性、度数符合性）并通过弱约束优化计划长度和并发动作数量。

**🔧 技术方法**

该方法使用了 Answer Set Programming（clingo）技术，对规划问题进行逻辑编程，并利用弱约束实现最优搜索。

**📊 数据集**

实验采用了法国电网数据抽取的7个真实实例（6–42节点）以及175个合成实例（|V|=8–50，|P|=2）进行评估。

**📈 对比分析**

通过与人工专家制定方案在规模、求解时间和最优性证明进行比较，实验表明该方法在同等规模下可在秒级至分钟级求解，并在部分实例上证明最优方案。

**⚠️ 局限性**

局限性包括对更大规模网络的可扩展性仍有限，以及未考虑预算约束或特定元件不可变等实际情况。

---

## 679. TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization

**arXiv ID:** 2605.20150 | [PDF](https://arxiv.org/pdf/2605.20150v1)

**作者:** Chonghao Zhong `[一作]` (Hong Kong University of Science and Technology), Chaojian Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 TideGS，一种将 3D Gaussian Splatting 的完整参数表虚拟化到 SSD‑CPU‑GPU 层级的离线训练框架，能够在单张 24 GB GPU 上训练十亿级 Gaussian；

**💡 创新点**

核心创新是：块虚拟化几何、异步跨层管线以及轨迹自适应差分流，将 VRAM 仅用作稀疏工作集缓存，彻底突破 GPU 内存瓶颈；

**🔧 技术方法**

采用块化参数表与 Morton 排序、CPU 视锥裁剪+GPU 精细筛选、SSD 日志式顺序写、双缓冲异步复制、以及轨迹自适应差分调度等技术；

**📊 数据集**

使用 Mip‑NeRF 360 作为小规模基准，MatrixCity BigCity/Aerial 作为大规模城市级场景，实验中还扩展至 1.1 B Gaussian 场景；

**📈 对比分析**

与 Native 3DGS、Naive Offload、CLM 等基线对比：在小规模下仅 1–2% 运行开销，质量保持一致；在大规模下可训练 1.1 B Gaussian，PCIe 流量比 CLM 降至 0.10 GB/iter，迭代时间约 525 ms，PSNR 达 26.1 dB，显著优于单 GPU 方案；

**⚠️ 局限性**

主要局限：依赖连续相机轨迹以获得高工作集重用；对高速 NVMe 的依赖，慢速存储时 I/O 抑制明显；SSD 日志写入导致临时空间占用及耐久性问题；optimizer 状态的冷启动可能影响收敛速度。

---

## 680. Hamilton--Jacobi Reachability for Spacecraft Collision Avoidance

**arXiv ID:** 2605.20138 | [PDF](https://arxiv.org/pdf/2605.20138v1)

**作者:** Larry Hui `[一作]` (University of California Berkeley), Jianshu Zhou `[通讯]` (National University of Singapore)

**通讯引用:** 1785 | [OpenAlex ID](https://openalex.org/A5005976047)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于Hamilton–Jacobi（HJ）可达性分析的去中心化双卫星碰撞规避框架，利用平面HCW动力学将相对运动建模为二人零和差分博弈；

**💡 创新点**

创新点在于将HJ可达性与混合自动机相结合，实现对相对状态的最坏情况可达集合（BRS）的离散计算，并基于该集合自动触发逃逸与恢复模式，提供严格的碰撞安全保证；

**🔧 技术方法**

使用技术包括平面RTN框架下的Hill–Clohessy–Wiltshire（HCW）线性动力学、零和差分博弈建模、Hamilton–Jacobi–Isaacs偏微分方程求解（数值逼近 Level‑Set 方法）、混合自动机控制以及 PD 控制策略；

**📊 数据集**

无公开数据集，所有实验基于仿真参数（LEO 圆轨道 500 km，控制加速度限幅 ±0.1 m/s²，扰动限幅 ±0.05 m/s²）；

**📈 对比分析**

通过数值仿真验证了BRS与安全/不安全区域的对应关系，并演示了在不同逃逸模式下卫星从碰撞风险区回到安全轨道的过程；性能指标主要为可达集合的安全性（无碰撞）和逃逸路径的燃油占用未做量化；

**⚠️ 局限性**

局限性包括：仅考虑二维平面HCW线性模型，忽略垂直运动、J₂扰动及非线性效应；BRS 计算耗时高，难以扩展到更高维或更大星座；扰动模型采用最坏情况，导致过度保守；未提供燃油消耗或实时性能评估。

---

## 681. TrajTok: Adaptive Spatial Tokenization for Trajectory Representation Learning

**arXiv ID:** 2605.20134 | [PDF](https://arxiv.org/pdf/2605.20134v1)

**作者:** Zhen Xiong `[一作]` (University of Southern California), Cyrus Shahabi `[通讯]` (University of Southern California)

**通讯引用:** 19790 | [OpenAlex ID](https://openalex.org/A5012068017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

该论文提出一种名为TrajTok的轨迹表示学习框架，能够将原始GPS轨迹转化为可训练的多分辨率空间离散符号，并通过一个分解的Transformer编码器生成通用轨迹嵌入。

**💡 创新点**

创新点包括：①基于轨迹密度自适应分割的多分辨率六边形网格令离散化更细粒化且词表更紧凑；②将几何和运动特征分流处理，先在各自通道自注意后再通过交叉注意融合；③使用时空旋转位置编码与简单的联合遮掩预训练目标，使模型同时学习空间结构和速度/方向模式。

**🔧 技术方法**

技术方法包括：密度自适应的H3分辨率层级token化；双通道Transformer（几何通道嵌入离散格子ID，运动通道嵌入速度与方向的正余弦表示）；时空RoPE位置编码；联合遮掩（几何预测+运动回归）预训练；轻量化任务适配头。

**📊 数据集**

在葡萄牙波尔图出租车GPS数据集（Porto）上进行实验。

**📈 对比分析**

与多种基线（t2vec、NeuTraj、TrajCL、Space2Vec等）比较，冻结的TrajTok在轨迹相似度检索HR@1 0.435、分类macro‑F1 0.773、ETA MAE 42.27 s、全时长回归MAE 38.41 s，均超过或接近当前最优任务特定方法，显示出优越的跨任务迁移性能。

**⚠️ 局限性**

主要局限在于仅在单一城市（波尔图）上验证，缺乏跨城市或多源轨迹的泛化测试；未引入地图匹配或道路网络等额外地理先验，可能限制在复杂道路环境下的表现。

---

## 682. Neurosymbolic Learning for Inference-Time Argumentation

**arXiv ID:** 2605.20098 | [PDF](https://arxiv.org/pdf/2605.20098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 683. Hermitian hull-variation of vector rank-metric codes and self-orthogonal generalized Gabidulin codes

**arXiv ID:** 2605.20109 | [PDF](https://arxiv.org/pdf/2605.20109v1)

**作者:** Duy Ho `[一作]` `[通讯]` (UAE University), Duy Ho (UAE University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究向量秩码的Hermitian包络变化问题，并证明除 (q,n)=(2,2) 的特殊情形外，任何向量秩码都可以等价为 Hermitian LCD 码；同时构造了具有任意 Hermitian 包络维数的最大秩码（MRD）码，利用缩放迹自对偶基实现 Hermitian 自正交的广义 Gabidulin 码；

**💡 创新点**

创新点在于引入缩放迹自对偶基解决奇特质数场扩张中缺乏自对偶基的问题，完成 Hermitian 包络变化的完全表征，并首次给出任意 Hermitian 包络维数的 MRD 码构造；

**🔧 技术方法**

主要技术包括 Hermitian 线性代数、秩码等价变换、矩阵秩-零化技术、缩放迹自对偶基的构造以及广义 Gabidulin 码的 Hermitian 自正交性证明；

**📊 数据集**

该研究为理论工作，不涉及实验数据集；

**📈 对比分析**

由于论文为纯理论证明，未进行实验或与其他方法的性能对比；

**⚠️ 局限性**

局限性包括 (q,n)=(2,2) 情形无法实现 Hermitian 包络变换；在偶特征下 Euclidean 自正交 MRD 码不存在；且 Euclidean 包络维数的完整控制仍有未完全覆盖的情况。

---

## 684. INSHAPE: Instance-Level Shapelets for Interpretable Time-Series Classification

**arXiv ID:** 2605.20088 | [PDF](https://arxiv.org/pdf/2605.20088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 685. One in Eight OpenAlex Abstracts Has Integrity Issues

**arXiv ID:** 2605.20168 | [PDF](https://arxiv.org/pdf/2605.20168v1)

**作者:** Seorin Kim `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 1278 | [OpenAlex ID](https://openalex.org/A5049169851)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对OpenAlex数据库中10000条英文期刊摘要进行系统性完整性评估，发现约12%的摘要存在完整性问题；

**💡 创新点**

创新点在于提出七类完整性失效模式的分类体系，并结合人类专家与大型语言模型的双阶段注释，公开了标注数据和可直接使用的LLM判别提示；

**🔧 技术方法**

采用人类专家与Claude Opus 4.6、OpenAI Codex GPT-5.4等大型语言模型进行四人投票、讨论和LLM分类prompt的构建，并用Fleiss κ、Cohen κ等指标评估一致性；

**📊 数据集**

使用随机抽样的10000条OpenAlex英文期刊文章（其中1000条做四人注释，另1000条做LLM校准），作为评估与训练数据；

**📈 对比分析**

通过将LLM分类结果与人类共识标签对比，二分类准确率达96%，Fleiss κ为0.50，Cohen κ在0.34到0.81之间，表明LLM在完整性判别上具备高一致性；

**⚠️ 局限性**

研究局限在于样本仅限已被引用的英文期刊文章，未包含非英文、预印本或最新出版物，实际失败率可能更高；此外LLM分类的提示可能对特定数据过拟合，且仅评估文本完整性，未覆盖语义质量等维度。

---

## 686. Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models

**arXiv ID:** 2605.20158 | [PDF](https://arxiv.org/pdf/2605.20158v1)

**作者:** Guangzhi Xiong `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 12116 | [OpenAlex ID](https://openalex.org/A5013588572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在胸片视觉问答任务中构建了一个因果验证框架 MedGround-Bench，并基于此评估了多种视觉归因方法的可靠性；提出了基于概念的因果归因方法 MedFocus，能定位并量化解剖结构对 LVLM 预测的影响。

**💡 创新点**

创新点在于：①首次将因果编辑与专家注释相结合，生成真正可验证的归因基准；②引入不平衡最优传输+MedSAM 两阶段分割来获取临床可解释的概念区域；③通过边界框零掩蔽的干预计算概念的因果效应，输出空间、概念层级和 token 级别归因。

**🔧 技术方法**

主要技术包括：因果过滤的三步流程（正确性、前景/背景编辑）；不平衡最优传输 (UOT) 进行解剖概念映射；MedSAM 进行掩码精细化；边界框干预+log‑prob 损失量化因果贡献；以及对多模型（Qwen、Gemma、MedGemma 等）和多输出模式（直接回答、链式推理）的统一评估。

**📊 数据集**

使用了三个公开的胸片数据集（ImaGenome、VinDR‑CXR、PadChest‑GR），从中构造了 3940 条因果验证样本，覆盖 6 种 LVLM 和 2 种输出模式。

**📈 对比分析**

与 11 种现有归因方法（梯度、注意力、扰动、提示等）比较，MedFocus 在 IoU、F1、精确率/召回率等指标上均显著领先（如在直接模式下 IoU 最高达 52.95%，精确率/召回率均衡），而其他方法多呈现过度扩散或偏移。

**⚠️ 局限性**

局限性包括：仅针对胸片场景；依赖专家标注与可编辑区域，难以推广到无标注或不支持编辑的模态；方法在计算上需要多次前向推理与分割，资源开销较大；以及对极端病变或复杂结构的概念覆盖仍有限。

---

## 687. MixRea: Benchmarking Explicit-Implicit Reasoning in Large Language Models

**arXiv ID:** 2605.20128 | [PDF](https://arxiv.org/pdf/2605.20128v1)

**作者:** Yuanqing Cai `[一作]` (University of Electronic Science and Technology of China), Yanru Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 33106 | [OpenAlex ID](https://openalex.org/A5100635835)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了显式-隐式推理任务并构建了 MixRea 基准，评估多种 LLM 在此任务中的注意力盲点。

**💡 创新点**

发现 LLM 在显式指令下往往忽视隐式信息，并提出 Potential Relation Completion Prompting（PRCP）提示方法以缓解该注意盲点。

**🔧 技术方法**

采用提示工程 PRCP、链式思考（CoT）与多阶段问答生成技术，并通过大规模 LLM 对比实验验证其效果。

**📊 数据集**

以 Possible Stories 数据集为基础构建 MixRea，包含 2246 题、9 种推理类型，并辅以 GPT‑4o 生成的隐式情境。

**📈 对比分析**

与 21 种 LLM 进行对照实验，使用准确率与一致性两项指标；Gemini 2.5 Pro 最高准确率 67.9%，PRCP 在所有模型上提升 1–5% 并显著提高一致性。

**⚠️ 局限性**

尽管 PRCP 有所提升，LLM 在隐式信息推理与多源信息整合中的一致性仍低于 43%，并且注意盲点在更广泛的推理任务中仍然明显存在。

---

## 688. Using Aristotle API for AI-Assisted Theorem Proving in Lean 4: A Formalisation Case Study of the Grasshopper Problem

**arXiv ID:** 2605.20120 | [PDF](https://arxiv.org/pdf/2605.20120v1)

**作者:** Gabriel Rongyang Lau `[一作]` (Nanyang Technological University), Gabriel Rongyang Lau `[通讯]` (Nanyang Technological University)

**通讯引用:** 2067 | [OpenAlex ID](https://openalex.org/A5038139064)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

对AI辅助定理证明系统Aristotle在Lean 4中对IMO 2009年6号组合问题（Grasshopper）进行的正式化案例研究，分析其已验证的局部证明片段与未完成的主定理；

**💡 创新点**

揭示了AI生成的证明在局部推理成功后，仍可能缺失全局计数与组合论证，表明局部与全局证明之间的缺口是AI辅助定理证明的重要局限；

**🔧 技术方法**

使用Lean 4、Mathlib以及Aristotle API进行自动化证明搜索与语义化证明生成；

**📊 数据集**

未使用公开数据集，研究以Grasshopper问题为目标；

**📈 对比分析**

未提供传统意义上的实验对比，报告的运行时约为八小时，说明系统在处理此类组合问题时的耗时；

**⚠️ 局限性**

主要限制在于AI系统仅能完成局部推理，无法自动完成全局计数与矛盾证明，导致主定理以占位符方式结束，体现了AI辅助证明在全局推理上的不足。

---

## 689. X-Ray cardiac angiographic vessel segmentation based on pixel classification using machine learning and region growing

**arXiv ID:** 2605.20073 | [PDF](https://arxiv.org/pdf/2605.20073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 690. k-Inductive Neural Barrier Certificates for Unknown Nonlinear Dynamics

**arXiv ID:** 2605.20108 | [PDF](https://arxiv.org/pdf/2605.20108v1)

**作者:** Ben Wooding `[一作]` (Vanderbilt University), Abolfazl Lavaei `[通讯]` (Newcastle University)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5011080021)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于单一状态轨迹的、无先验模型的离散时间非线性系统k‑inductive神经安全壁垒证书（k‑NBC）构造与验证方法。

**💡 创新点**

创新点在于：①利用Willems基本定理的推广从单条轨迹获得数据驱动模型，消除闭环模型依赖；②采用k‑induction放宽传统壁垒条件，允许在有限步内临时升高；③将神经网络与SMT求解器结合的CEGIS框架，实现从数据到正式验证的闭环。

**🔧 技术方法**

核心技术包括：神经网络近似器、反向传播优化、损失函数设计（满足k‑BC四条约束）、SMT求解器（dReal）与逆向约束求逆；数据驱动模型构造依赖单轨迹插值与矩阵求逆。

**📊 数据集**

实验使用的唯一数据集为每个案例的单条状态轨迹（长度7、10等），不使用公开大规模数据集。

**📈 对比分析**

在三个案例（多项式系统、摆动系统、极其非线性系统）上验证，k‑NBC在极其非线性系统上成功构造合法证书，而传统k=1壁垒无法通过验证；在实验中迭代次数显著降低，证明方法有效。

**⚠️ 局限性**

局限性包括：①SMT求解器对高维/复杂网络的可扩展性有限；②需要足够的激励数据以满足矩阵满秩；③目前仅针对离散时间系统，持续时间或噪声鲁棒性待进一步研究。

---

## 691. Bridging the Disciplinary Gap in Explainable AI: From Abstract Desiderata to Concrete Tasks

**arXiv ID:** 2605.20081 | [PDF](https://arxiv.org/pdf/2605.20081v1)

**作者:** Hanwei Zhang `[一作]` (Saarland University), Holger Hermanns `[通讯]` (Saarland University)

**通讯引用:** 10953 | [OpenAlex ID](https://openalex.org/A5028747794)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了三维需求分类法（目标、功能角色、正当化模式）和三步框架，用以识别解释AI需求的依赖关系、评估实现可行性并推导可执行的XAI任务

**💡 创新点**

创新点在于将跨学科的XAI需求视为相互依赖的结构，构建依赖图并通过目标与功能角色的分解，系统化地将抽象需求转化为可评估、可实现的任务

**🔧 技术方法**

采用文献综述、概念分析、案例研究以及依赖图构建等方法，未涉及具体算法实现

**📊 数据集**

论文主要基于已有文献和两例应用场景（人类监督与法律审计），无使用公开数据集

**📈 对比分析**

本文未进行实验或数值比较，仅通过两例说明框架的可行性和优势，未给出性能指标

**⚠️ 局限性**

局限性包括：依赖结构和分类均为专家主观解释，缺乏大规模实证验证；框架未与具体算法结合，难以直接衡量效果；案例局限于人类监督和审计领域

---

## 692. ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning

**arXiv ID:** 2605.20176 | [PDF](https://arxiv.org/pdf/2605.20176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 693. Towards Distillation Guarantees under Algorithmic Alignment for Combinatorial Optimization

**arXiv ID:** 2605.20074 | [PDF](https://arxiv.org/pdf/2605.20074v1)

**作者:** Thien Le `[一作]` (Harvard University), Melanie Weber `[通讯]` (Harvard University)

**通讯引用:** 884 | [OpenAlex ID](https://openalex.org/A5034942394)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种在结构化预测任务中将大型预训练模型知识蒸馏到与动态规划算法对齐的图神经网络的方法；

**💡 创新点**

其创新点在于引入了线性表示假设（LRH）与局部迭代对齐（local‑iteration alignment）作为蒸馏的充分条件，并给出在决定树深度、图规模和消息传递轮数受限时的高效蒸馏算法；

**🔧 技术方法**

主要技术包括：基于决策树的动态规划抽象、线性探测子程序（LinearProbe）验证LRH、构造根前缀路径集合以及基于动态规划的树重构；

**📊 数据集**

实验使用的是小型随机生成的图（n=6）和随机决策树，训练的源模型为5层ResNet，尺寸为1000；

**📈 对比分析**

与基准方法（如直接对决策树进行蒸馏）相比，提出的两阶段蒸馏方法在保持高源模型准确率的同时，能够在相同的测试集上获得相对较高的蒸馏模型准确率（例如，深度5时从0.79提升至0.67），且样本复杂度与模型复杂度呈多项式关系；

**⚠️ 局限性**

局限性包括：算法只在固定图规模、固定消息传递轮数、决策树深度受限的情况下高效；对大规模图或深层决策树的扩展仍需改进；此外，实验规模较小，未验证在真实大规模数据上的性能。

---

## 694. KoRe: Compact Knowledge Representations for Large Language Models

**arXiv ID:** 2605.20170 | [PDF](https://arxiv.org/pdf/2605.20170v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 695. CaMo: Camera Motion Grounded Evaluation and Training for Vision-Language Models

**arXiv ID:** 2605.20165 | [PDF](https://arxiv.org/pdf/2605.20165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 696. SetCon: Towards Open-Ended Referring Segmentation via Set-Level Concept Prediction

**arXiv ID:** 2605.20110 | [PDF](https://arxiv.org/pdf/2605.20110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 697. SAGE: Scalable Automatic Gating Ensemble for Confident Negative Harvesting in Fraud Detection

**arXiv ID:** 2605.20157 | [PDF](https://arxiv.org/pdf/2605.20157v1)

**作者:** Sudheer Tubati `[一作]` (Amazon Music), Amit Goyal `[通讯]` (Amazon Music)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种名为SAGE的框架，在音乐流媒体诈骗检测中通过对未标记数据进行自信负样本采集，解决正负样本不平衡和标签缺失问题。

**💡 创新点**

创新点在于将SimHash分层采样与可配置阈值的模块化门控集成相结合，利用全局统计距离和局部密度两重门控实现对负样本的高置信度筛选，显著缓解代表性偏差与误标污染。

**🔧 技术方法**

技术手段包括SimHash进行行为分层采样；Mahalanobis距离门控（配合Ledoit‑Wolf协方差收缩）评估全局距离；k‑NN密度门控衡量局部相似度；LightGBM分类器进行多类别训练；阈值调优与样本加权处理。

**📊 数据集**

使用亚马逊音乐在全球规模下收集的日常流媒体行为数据，特征包含时序方差、熵、设备多样性及短期趋势等；标签来源包括启发式标注、人工审核与少量已验证的非欺诈样本。

**📈 对比分析**

与Isolation Forest、变分自编码器、随机欠采样、学生‑教师自学习等基线模型对比，SAGE在保留正样本的同时实现了+81.9个百分点的精度提升、+87.2个百分点的召回提升（F1提升+85.2个百分点），证明了方法的显著优势。

**⚠️ 局限性**

局限性包括：需要足够的已标注欺诈样本来训练门控阈值；阈值调优与模型验证耗时；无法完全避免概念漂移导致的性能下降；在极端稀缺或新型欺诈模式下仍可能出现误判。

---

## 698. When Does Model Collapse Occur in Structured Interactive Learning?

**arXiv ID:** 2605.20151 | [PDF](https://arxiv.org/pdf/2605.20151v1)

**作者:** Yuchen Wu `[一作]` (Cornell University), Weijie Su `[通讯]` (University of Pennsylvania)

**通讯引用:** 8277 | [OpenAlex ID](https://openalex.org/A5080575294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在生成式 AI 互动学习环境中模型崩溃的现象，并提出基于有向图的通用框架，给出了模型崩溃的必要与充分条件，证明在线性回归和一般 M‑估计下模型崩溃仅由交互图拓扑决定。

**💡 创新点**

创新点在于：①首次用有向图刻画多模型间的交互模式；②导出模型崩溃的精确必要与充分条件，揭示“稳定”与“不稳定”数据源对性能的根本影响；③在通用 M‑估计框架下得到有限样本与渐近风险分析，展示崩溃与否仅与图结构相关。

**🔧 技术方法**

核心技术包括：有向图与拓扑分析；矩阵递推与马尔可夫链视角；线性回归的闭式解析与高维矩阵不等式；M‑估计的经验过程理论（Glivenko‑Cantelli 与 Donsker 类）与极限定理；风险比(r_t, μ / r_t, μ*) 与 FID 指标的定量比较。

**📊 数据集**

主要使用的数据集：①合成数据（线性回归、逻辑回归、单指数模型）；②真实图像数据 MNIST 与 CIFAR‑10，用 GAN 生成合成样本进行交互学习实验。

**📈 对比分析**

比较方法：对每个模型计算与自然数据训练基准的风险比 r_t, μ / r_t, μ*；在图像实验中采用 Fréchet Inception Distance (FID) 比例；实验结果与理论一致：属于 _l^c 的模型风险比随迭代线性增长，崩溃；属于 _l^nc 的模型风险比保持有界，未崩溃；添加单条边可导致崩溃敏感性。

**⚠️ 局限性**

限制：①理论假设较强（如样本独立、设计矩阵完整秩、噪声方差上下界、GLM 的光滑与正则性）；②对 M‑估计的证明依赖于高度的可微性与凸性或紧致性；③真实数据实验未严格满足理论假设，仅作经验验证；④只考察了线性、逻辑、泊松等经典 GLM，未涵盖更复杂非线性模型；⑤交互图是静态的，未考虑动态变化或学习率调整。

---

## 699. Draft Less, Retrieve More: Hybrid Tree Construction for Speculative Decoding

**arXiv ID:** 2605.20104 | [PDF](https://arxiv.org/pdf/2605.20104v1)

**作者:** Yuhao Shen `[一作]` (Zhejiang University), Cong Wang `[通讯]` (Zhejiang University)

**通讯引用:** 25935 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合动态树剪枝与检索插枝的混合树构造方法，利用剪枝释放的计算预算填充检索候选，从而打破树形显式推理的延迟–接受长度权衡。

**💡 创新点**

创新点在于将剪枝视为预算释放而非单纯去除候选，将检索嵌入到树结构中；使用GPU驻留的邻接矩阵实现无CPU同步、并行检索；结合在线目标模型更新与根中心检索模板；保持固定的验证预算，兼容现有树形推理核。

**🔧 技术方法**

技术手段包括：动态深度剪枝（加权阈值校准）、根中心检索模板、GPU邻接矩阵检索、在线目标模型反馈更新、并行检索与树验证融合、长上下文专用KV缓存优化。

**📊 数据集**

实验数据集涵盖：短上下文——HumanEval、GSM8K、CNN/DM、Alpaca、MT-Bench；长上下文——QMSum、GovReport、MultiNews、LCC、RepoBench-P；使用 Vicuna-13B、LLaMA‑3.1‑8B、Qwen3‑8B/32B/235B 等多规模模型。

**📈 对比分析**

对比 EAGLE‑3、DDD、ECHO、TR、PLD、SAMD 等基线；结果显示在短上下文下最高可达 5.41× 的速度提升（比 EAGLE‑3 提升 7–22%），在 Qwen3‑235B 上实现 2.09× 的平均加速；在长上下文下获得 3.22× 的速度提升，平均 MAT 也提高 2–8%；高批量推理中保持更高吞吐量和 MAT。

**⚠️ 局限性**

局限性包括：检索效果高度依赖提示/历史的局部重复结构，极端无重复语料时收益有限；高并发环境下检索核与调度未充分优化；方法主要针对树形显式推理，对块式推理（如 DFlash）需进一步设计；GPU邻接矩阵的存储虽然小，但随着词表扩展会增加内存占用；剪枝决策仍可能误删有效分支，导致 MAT 降低。

---

## 700. CopT: Contrastive On-Policy Thinking with Continuous Spaces for General and Agentic Reasoning

**arXiv ID:** 2605.20075 | [PDF](https://arxiv.org/pdf/2605.20075v1)

**作者:** Dachuan Shi `[一作]` (Georgia Tech), Wenke Lee `[通讯]` (Georgia Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的LLM推理流水线，先草拟答案再根据草稿可靠性触发后续的 on‑policy 思考，从而更快获得答案并减少不必要的思考。

**💡 创新点**

创新点在于把连续嵌入从原本的生成媒介改为推理时的对比验证器，利用离散与连续输入的对比估计草稿可信度，并动态控制草稿在后续思考中的可见性。

**🔧 技术方法**

核心技术包括：1) 归一化序列级反向 KL 估计器 (κ_a、κ_r)；2) 草稿可靠性评估与阈值触发；3) 分块可见性控制；4) 连续嵌入作为推理时的对比验证。

**📊 数据集**

在多任务、多规模上评估：数学/ STEM（GSM8K、Math500、AIME24、AIME25）、编码（HumanEval、MBPP、LeetCode‑Contest）、代理推理（BFCL v4、ZebraArena）等多种 benchmark。

**📈 对比分析**

与标准 CoT、Greedy CoT 以及训练无关的连续生成方法 Soft‑Thinking、SwiReasoning 比较，CopT 在多数基准上提升 0.2–3.6% 准确率，同时在匹配或更高准确率时减少 20–70% token，进一步在单样本延迟上实现 20–70% 的缩减。

**⚠️ 局限性**

局限性包括：阈值 τ_a、τ_r 需要手动调参；在极难题或长交互任务中仍需大量思考；对模型规模的依赖性未完全消除；对连续嵌入质量与模型架构的敏感性需要进一步研究。

---

## 701. Probing Embodied LLMs: When Higher Observation Fidelity Hurts Problem Solving

**arXiv ID:** 2605.20072 | [PDF](https://arxiv.org/pdf/2605.20072v1)

**作者:** Oussama Zenkri `[一作]` (Technische Universität Berlin), Oliver Brock `[通讯]` (Technische Universität Berlin)

**通讯引用:** 9181 | [OpenAlex ID](https://openalex.org/A5039143538)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在真实机械锁箱任务上让大型语言模型（LLM）在不同感知通道（RGB、RGB‑D、符号状态）下执行并记录行为，探究更高观测精度是否能提升任务成功率。

**💡 创新点**

创新点在于将行为探测方法（empirical AI）与观测精度干预相结合，发现更高精度输入反而可能导致LLM陷入重复动作循环，适度的感知噪声能提升表现，并将这种现象与循环行为减少联系起来。

**🔧 技术方法**

使用OpenAI GPT‑4o / GPT‑o1作为决策器，搭配Franka Emika Panda机械臂、RBO Hand 3软手爪、RGB‑D摄像头及力传感器的物理机器人系统；在模拟环境中通过随机翻转状态来注入感知噪声；采用整数规划检测重复动作循环；对比人类启发式策略。

**📊 数据集**

实验数据主要来自自制：物理锁箱实验（10次/观测模式共30次）以及模拟实验中生成的210个试验（每种噪声概率10次×10独立试验）。

**📈 对比分析**

通过与人类启发式策略以及不同观测模式下的成功率和步数进行对比，发现GPT在RGB模式下最快达到80%成功率（11步），但在符号状态模式下需要15步；在模拟中，40%状态翻转噪声时成功率提升约2.85倍。整体表现随观测精度变化呈非单调下降。

**⚠️ 局限性**

局限性包括：仅评估了OpenAI单一供应商的模型，未检验其他或开源模型；实验仅使用单一锁箱布局，可能缺乏普适性；对比基于人类启发式而非真实人类数据；物理环境中的噪声来源和机制尚未彻底解析。

---

## 702. Topology-Optimized Pneumatic Soft Actuator: Design and Experimental Validation

**arXiv ID:** 2605.20101 | [PDF](https://arxiv.org/pdf/2605.20101v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 703. Multi-axis Analysis of Image Manipulation Localization

**arXiv ID:** 2605.20174 | [PDF](https://arxiv.org/pdf/2605.20174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 704. From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models

**arXiv ID:** 2605.20177 | [PDF](https://arxiv.org/pdf/2605.20177v1)

**作者:** Juncheng Wu `[一作]` (Amazon), Yuyin Zhou `[通讯]` (UC Santa Cruz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个三阶段的 VLM 后训练框架，先训练视觉感知，再训练文本推理，最后训练视觉推理。

**💡 创新点**

创新点在于将视觉感知与推理解耦并按阶段训练，并提出使用 RLVR 替代传统 SFT 提升感知能力，同时将能力维度与难度维度的课程学习结合。

**🔧 技术方法**

主要技术包括强化学习可验证奖励（RLVR）、Group Relative Policy Optimization（GRPO）、基于 LLM 的视觉感知数据生成与筛选、以及多阶段分层训练策略。

**📊 数据集**

使用了公开的图像-字幕数据（DOCCI）、视觉数学数据集（MathVista、MathVision、WeMath 等）、视觉推理集（CLEVR-Math、GeoQA、Math PUMA 等）以及文本推理集（OpenBookQA 等）。

**📈 对比分析**

与单阶段合并训练及多种开源 VLM（GThinker、OpenVLThinker、OneThinker 等）对比，三阶段训练在视觉数学与视觉感知任务上均提升 1–5% 的准确率，且推理文本长度缩短 20% 以上。

**⚠️ 局限性**

局限性包括仅在 7–8B 参数规模验证；对细粒度图像字幕资源依赖；三阶段分解可能不是最细粒度的能力划分，后续可进一步细化。

---

## 705. HaorFloodAlert: Deseasonalized ML Ensemble for 72-Hour Flood Prediction in Bangladesh Haor Wetlands

**arXiv ID:** 2605.20167 | [PDF](https://arxiv.org/pdf/2605.20167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 706. A Methodology for Selecting and Composing Runtime Architecture Patterns for Production LLM Agents

**arXiv ID:** 2605.20173 | [PDF](https://arxiv.org/pdf/2605.20173v1)

**作者:** Vasundra Srinivasan `[一作]` `[通讯]` (Stanford University), Vasundra Srinivasan (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出并验证了一套基于“随机-确定性边界（SDB）”的生产LLM代理运行时架构方法，涵盖三大关注点（协调、状态、控制）与六种模式，并给出了五步模式选择流程、诊断手册及可靠性分解模型。

**💡 创新点**

创新点在于将LLM代理核心契约正式命名为SDB，提供可复制的模式目录、模式选择与诊断流程，并将长期可靠性拆解为模型方差与架构动量两个可调节维度。

**🔧 技术方法**

采用分布式系统原语（actor、saga、事件日志、CAS、监督、工作流网）映射到LLM代理，将LLM输出、验证器、提交与拒绝信号四部分构成SDB，并结合工具调用与可观测性设计。

**📊 数据集**

使用公开的IBM Telco Customer Churn数据集构建90天合同续签的参考实现，并在此基础上演示了五个不同工作负载的工作流。

**📈 对比分析**

在五个工作负载上执行方法，生成六行架构决策记录，验证方法能针对不同工作负载给出不同答案；在参考实现中未给出具体性能数值，而是说明可根据模型版本和模式调整后系统可靠性与可观测性得到提升。

**⚠️ 局限性**

局限性包括方法不覆盖冷启动、组织决策、模型上游配置等；假设团队可访问先前模型版本进行重放，若未来模型一致性提升，部分模式与诊断可能失效。

---

## 707. Less Back-and-Forth: A Comparative Study of Structured Prompting

**arXiv ID:** 2605.20149 | [PDF](https://arxiv.org/pdf/2605.20149v1)

**作者:** Saurav Ghosh `[一作]` (Washington University in St. Louis), Abdou Sow `[通讯]` (Washington University in St. Louis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三种提示方式（原始提示、清单改进提示、澄清提问提示）进行实验比较，覆盖四类常见LLM任务（摘要、规划、解释、编码）并使用三种模型（ChatGPT、Claude、Grok）进行评估。

**💡 创新点**

首次系统性比较清单式结构化提示与交互式澄清提示在不同任务和模型上的表现，并量化输出质量与交互成本之间的权衡。

**🔧 技术方法**

采用统一的四维评分量表（任务完成度、正确性、合规性、清晰度）对LLM输出进行打分；同时记录回合数、输入/输出Token数量等交互成本；使用配对t检验和bootstrap CI进行统计比较。

**📊 数据集**

使用四个人工设计的任务模板（摘要、规划、解释、编码），不依赖大型公开数据集，仅基于模型生成的答案进行评测。

**📈 对比分析**

通过配对比较方法，清单提示在平均分7.5/8、Token使用最少、回合数仅1的条件下优于原始和澄清提示；澄清提示平均分6.67/8，但回合数约1.96，Token使用936；实验结果表明清单提示在质量与交互效率上均表现最佳。

**⚠️ 局限性**

局限性包括样本规模有限、仅四类任务、评测者单一、未进行外部用户评价、未拆解清单各项对效果的具体贡献，以及不同模型Token计数方法不一致导致的比较偏差。

---

## 708. Beyond Isotropy in JEPAs: Hamiltonian Geometry and Symplectic Prediction

**arXiv ID:** 2605.20107 | [PDF](https://arxiv.org/pdf/2605.20107v1)

**作者:** Robert Jenkinson Alvarez `[一作]` `[通讯]`, Robert Jenkinson Alvarez

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了 JEPAs 的几何正则化，证明欧氏等方差正则化在未知任务几何下可能不优，并提出 HamJEPA——使用相位空间编码和辛跃步预测器来引入 Hamiltonian 结构。

**💡 创新点**

①理论证明固定等方差边缘目标在未知几何下不具可泛化性；②将结构偏置放在视图间预测器而非编码器边缘，设计相位空间 Hamiltonian Leapfrog 预测器；③使用非等方差抗崩溃约束保持多样性。

**🔧 技术方法**

相位空间编码 (q,p)、辛跃步 (leapfrog) 预测器、Hamiltonian 能量函数、投影日志-行列式与参与度比约束、双向预测、尺度预算等。

**📊 数据集**

CIFAR‑100 与 ImageNet‑100 作为预训练基准。

**📈 对比分析**

在严格对等设置（相同 backbone、augment、optimizer、无投影头）下与 SIGReg（等方差正则化）对比；HamJEPA 在 30/80 轮 CIFAR‑100 上 kNN@20 +4.89/+6.45、线性探测 +3.52/+10.64；在 45 轮 ImageNet‑100 上 kNN@20 +4.82、线性探测 +7.52，显著优于 SIGReg。

**⚠️ 局限性**

仅在受限的 headless 预训练实验，未与更强 SSL 基线比较；ImageNet‑100 仅单种种子；未学习任务几何 H，需进一步研究。

---

## 709. Stochastic Chase Decoding for BMS Channels via Rate Distortion Theory

**arXiv ID:** 2605.20129 | [PDF](https://arxiv.org/pdf/2605.20129v1)

**作者:** Amit Berman `[一作]` (Samsung Semiconductor), Ilya Shapir `[通讯]` (Samsung Semiconductor)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本研究开发了一种基于速率失真理论的方法，用于在二进制无记忆对称（BMS）信道上对代数码进行随机Chase解码，替代了传统的启发式翻转概率确定方法。

**💡 创新点**

创新点在于将随机Chase解码重新解释为错误模式覆盖码的随机编码构造，并提供了明确的渐近最优翻转规则的特征描述。

**🔧 技术方法**

使用了速率失真理论和随机编码技术，特别是反向水位公式来计算翻转概率。

**📊 数据集**

研究中使用了二进制和四元对称信道的模拟数据。

**📈 对比分析**

与传统Chase解码方法进行了比较，结果表明随机Chase解码在短块长度下的性能接近信息论规则，且在高概率下能够确保传输码字出现在解码列表中。

**⚠️ 局限性**

限制在于该方法主要针对BMS信道，可能在其他类型信道上的适用性和性能尚未充分验证。

---

## 710. ThoughtTrace: Understanding User Thoughts in Real-World LLM Interactions

**arXiv ID:** 2605.20087 | [PDF](https://arxiv.org/pdf/2605.20087v1)

**作者:** Chuanyang Jin `[一作]` (Johns Hopkins University), Tianmin Shu `[通讯]` (Johns Hopkins University)

**通讯引用:** 1244 | [OpenAlex ID](https://openalex.org/A5005908625)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个大规模的真实世界人机对话数据集，并让用户在对话过程中记录他们的隐式思维；

**💡 创新点**

首次将用户隐式思维作为一种新的数据模态与对话文本并行收集，证明其与表面对话不同且能显著提升行为预测和模型对齐；

**🔧 技术方法**

使用对话注释平台、LLM推理、语义相似度评估以及基于DPO的对齐训练等技术；

**📊 数据集**

使用自建的ThoughtTrace数据集（1,058名用户、2,155场对话、10,174条思维注释）；

**📈 对比分析**

在无思维基线与思维增强模型对比实验中，思维增强模型在用户下一句预测上平均提升41.7%相对收益，在Arena‑Hard对齐评测中，思维引导重写相较基线提升了25.6%胜率；

**⚠️ 局限性**

主要局限包括思维注释仅捕获显意识推理，可能影响交互本身；招募样本来源有限，可能存在偏倚；仅评估了两个下游任务，尚缺乏更广泛的验证。

---

## 711. Spatially Prompted Visual Trajectory Prediction for Egocentric Manipulation

**arXiv ID:** 2605.20085 | [PDF](https://arxiv.org/pdf/2605.20085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 712. Probability-Conserving Flow Guidance

**arXiv ID:** 2605.20079 | [PDF](https://arxiv.org/pdf/2605.20079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 713. Interpretable Computer Vision for Defect Detection in X-ray Tomography of Aerospace SiC/SiC Composites

**arXiv ID:** 2605.20159 | [PDF](https://arxiv.org/pdf/2605.20159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 714. BalanceRAG: Joint Risk Calibration for Cascaded Retrieval-Augmented Generation

**arXiv ID:** 2605.20084 | [PDF](https://arxiv.org/pdf/2605.20084v1)

**作者:** Zijun Jia `[一作]` (Beihang University), Zhiyuan Wang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7484 | [OpenAlex ID](https://openalex.org/A5100462276)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了BalanceRAG，一种无训练的风险控制级联LLM‑RAG框架，通过联合阈值校准实现对选择性错误率的有限样本高概率保证。

**💡 创新点**

创新点在于将阈值配对视为二维阈值格点，利用顺序图形检验（SGT）高效发现安全阈值点，并支持多风险（错误率与检索频率）同时控制；相比传统逐级或Bonferroni校准显著提升覆盖率与效用。

**🔧 技术方法**

采用精确二项检验、顺序图形检验、交并规则进行多风险校准，并使用语义相似度、熵等不确定性估计器作为路由信号。

**📊 数据集**

实验数据集包括TriviaQA、SQuAD v2和Natural Questions（NQ），并在八种不同的LLM骨干上验证。

**📈 对比分析**

与LLM‑only、始终检索的RAG、UCB‑Cascaded-CP/HFD、Adaptive‑RAG、Self‑Route等基线对比，BalanceRAG在满足目标风险水平的同时获得更高的覆盖率、正确答案数，并显著减少检索调用。

**⚠️ 局限性**

局限性：校准假设训练集与部署集分布相同，若查询分布或检索语料改变需重新校准；结果受所选正确性判定（如语义相似度）影响；目前仅针对两分支级联，扩展到多阶段或工具增强系统仍需研究。

---

## 715. Not Every Rubric Teaches Equally: Policy-Aware Rubric Rewards for RLVR

**arXiv ID:** 2605.20164 | [PDF](https://arxiv.org/pdf/2605.20164v1)

**作者:** Utkarsh Tyagi `[一作]` (Scale AI), Yunzhong He `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 POW3R，一种在强化学习中根据当前策略的 roll‑out 对标记准则进行动态加权的奖励框架，旨在将训练压力聚焦在能产生梯度信号的准则上，从而提升多维评价指标。

**💡 创新点**

创新点在于将人类赋予的重要性与当前可学习性解耦，利用 roll‑out 方差估计实时调整准则权重，同时保持原始 Rubric 的评估目标不变。

**🔧 技术方法**

采用 Group‑Relative Policy Optimization (GRPO) 作为核心训练算法，结合准则方差估计、类别归一化、指数移动平均更新等技术，对 Rubric 进行策略感知重加权；评估判定则使用 LLM（GPT‑5.4‑nano/mini）进行二值打分。

**📊 数据集**

主要使用两大数据集：多模态 Rubric‑RL 数据集 MM（10k 任务，包含图像、文本和 6 个质量类别）以及文本仅的 HealthBench English（500 个难度任务）。此外，在 MM 上还对模型在 HallusionBench、POPE、MM‑IFE、MMVetV2、MathVista、RealWorldQA 等外部 VLM 基准进行了迁移评估。

**📈 对比分析**

与基线模型、二元奖励、静态加权奖励以及类别均衡奖励进行对比；在 30 个基线/指标组合中，POW3R 在 24 个场景下取得最高的 Rubric 奖励和严格完成率；在相同计算资源下，POW3R 的收敛速度比静态加权快 2.5–4 倍，并且在外部基准上的表现与原始模型持平。

**⚠️ 局限性**

主要限制包括：对 LLM 判定器的依赖，判定偏差会直接影响权重更新；仅在包含静态人类权重的 Rubric 数据集上验证，缺乏对编码、科学写作或多语言任务的评估；当数据集中的准则饱和度或分布与实验设置差异较大时，POW3R 的效果可能受限。

---

## 716. PixVerve: Advancing Native UHR Image Generation to 100MP with a Large-Scale High-Quality Dataset

**arXiv ID:** 2605.20147 | [PDF](https://arxiv.org/pdf/2605.20147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 717. Toto 2.0: Time Series Forecasting Enters the Scaling Era

**arXiv ID:** 2605.20119 | [PDF](https://arxiv.org/pdf/2605.20119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 718. Optimal Representation Size: High-Dimensional Analysis of Pretraining and Linear Probing

**arXiv ID:** 2605.20105 | [PDF](https://arxiv.org/pdf/2605.20105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 719. BiRD: A Bidirectional Ranking Defense Mechanism for Retrieval Augmented Generation

**arXiv ID:** 2605.20123 | [PDF](https://arxiv.org/pdf/2605.20123v1)

**作者:** Chengcai Gao `[一作]` (Wuhan University), Chao Liang `[通讯]` (Wuhan University)

**通讯引用:** 17506 | [OpenAlex ID](https://openalex.org/A5019539746)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对检索增强生成（RAG）系统的防御机制——BiRD，利用双向检索（正向和反向）对文档进行排名一致性检测，从而识别并过滤恶意注入的文档。

**💡 创新点**

创新点在于首次发现并利用了被攻击文档在正向检索中的高排名与其在反向检索中的排名高度一致的特征；通过组合内容相关性得分与排名一致性得分构建复合风险评分，既不依赖LLM内部知识，也不需高昂的聚类或投票开销，实现了高效且鲁棒的防御。

**🔧 技术方法**

技术核心包括：① 正向检索计算语义相似度得分；② 对每个检索到的文档进行反向检索；③ 计算正向与反向检索列表的Spearman相关系数作为上下文一致性；④ 通过 r_cr / (1 – r_cc) 形成复合评分并根据阈值筛选文档。

**📊 数据集**

在三个公开问答数据集上评估：Natural Questions (NQ)、MSMARCO、HotpotQA；使用三种检索模型（Contriever、ANCE、DPR）和三种LLM（Qwen-7B、Mistral-7B、Llama-3.1-8B）。

**📈 对比分析**

与 VanillaRAG、RobustRAG、InstructRAG、ReliabilityRAG 等基线对比，BiRD 在两种攻击场景（PoisonedRAG 与 PIA）下平均将攻击成功率（ASR）降低 40–87%，并提升答案准确率（ACC） 10–55%；且平均额外延迟不足 1 秒，远低于其他防御方法。

**⚠️ 局限性**

局限性包括：需要手动或基于验证集调节阈值；对极低比例的恶意文档可能仍难以完全过滤；只针对检索阶段的攻击，对生成阶段的恶意注入不具备直接防御能力；若攻击者改变排名策略（如使用多样化投票），BiRD 的排名一致性特征可能被削弱。

---

## 720. MetaEarth-MM: Unified Multimodal Remote Sensing Image Generation with Scene-centered Joint Modeling

**arXiv ID:** 2605.20090 | [PDF](https://arxiv.org/pdf/2605.20090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 721. PiG-Avatar: Hierarchical Neural-Field-Guided Gaussian Avatars

**arXiv ID:** 2605.20185 | [PDF](https://arxiv.org/pdf/2605.20185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 722. Atoms of Thought: Universal EEG Representation Learning with Microstates

**arXiv ID:** 2605.20182 | [PDF](https://arxiv.org/pdf/2605.20182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 723. What Do Evolutionary Coding Agents Evolve?

**arXiv ID:** 2605.20086 | [PDF](https://arxiv.org/pdf/2605.20086v1)

**作者:** Nico Pelleriti `[一作]` (Zuse Institute Berlin), Sebastian Pokutta `[通讯]` (Zuse Institute Berlin)

**通讯引用:** 2041 | [OpenAlex ID](https://openalex.org/A5043574831)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 EvoTrace 数据集和 EvoReplay 方法，系统分析 LLM 驱动的演化编码搜索轨迹，揭示改进机制与循环现象。

**💡 创新点**

创新点在于提供统一可重放的搜索轨迹数据集、基于 LLM 的编辑类型注释与重放干预框架，以及对循环、参数调优和可复现性的深入评估。

**🔧 技术方法**

使用 LLM 作为变异器/组合器，结合 SkyDiscover 框架、Bayesian Optimization、LLM-as-judge 等技术，处理 Python 与 C++ 任务。

**📊 数据集**

数据集包含 121 次演化搜索跑，跨 4 个框架、5 种 LLM、16 个任务，包含 10,672 程序、18,400 次 LLM 调用。

**📈 对比分析**

与仅报告最终分数的方法相比，发现约 30% 的代码行为循环、参数调优可覆盖 13/15 运行的最终分数，重放可恢复 76% 分数，说明改进机制多样且可复制性有限。

**⚠️ 局限性**

局限性包括样本覆盖有限、重放受原始评估器约束、对更大规模搜索和不同语言环境的通用性待验证。

---

## 724. VL-DPO: Vision-Language-Guided Finetuning for Preference-Aligned Autonomous Driving

**arXiv ID:** 2605.20082 | [PDF](https://arxiv.org/pdf/2605.20082v1)

**作者:** Zhefan Xu `[一作]` (Carnegie Mellon University), Khaled S. Refaat `[通讯]` (Waymo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了VL‑DPO框架，通过零射视觉‑语言模型（VLM）自动生成轨迹偏好对，并利用直接偏好优化（DPO）微调运动预测模型，实现对人类驾驶偏好的对齐；

**💡 创新点**

创新点在于①使用冻结的零射VLM作为偏好注释器自动生成高质量轨迹偏好对；②将VLM生成的高层动作与轨迹偏好作为DPO监督，避免了VLM的知识遗忘；③采用模块化设计，将VLM的推理与运动预测分离，提升可解释性和安全性；

**🔧 技术方法**

使用预训练的MotionLM生成式运动预测模型、Gemini 2.5 Pro VLM、Chain‑of‑Thought提示、Direct Preference Optimization（DPO）以及高层动作（HLA）作为辅助监督；

**📊 数据集**

在Waymo Open End‑to‑End Driving Dataset (WOD‑E2E) 上进行微调和评估，预训练阶段使用内部大规模数据集；

**📈 对比分析**

与基线MotionLM、纯模仿学习、HLA监督以及基于人类偏好的DPO进行对比；VL‑DPO实现了RFS提升约11.94%（≈8.16）且ADE下降约10.01%（≈2.88 m），明显优于其他方法；

**⚠️ 局限性**

局限在于：①VLM的零射推理受限于其自身的常识与语言理解，可能与实际人类偏好不完全一致；②偏好生成过程仍需离线处理，增加了数据准备成本；③模型目前仅针对单车道单车辆预测，未验证多场景通用性；④对实时推理与算力需求未作充分评估。

---

## 725. MSAVBench: Towards Comprehensive and Reliable Evaluation of Multi-Shot Audio-Video Generation

**arXiv ID:** 2605.20183 | [PDF](https://arxiv.org/pdf/2605.20183v1)

**作者:** Yujie Wei `[一作]` (Fudan University), Hongming Shan `[通讯]` (Fudan University)

**通讯引用:** 4711 | [OpenAlex ID](https://openalex.org/A5049086157)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个多镜头音视频生成基准与自适应混合评估框架，涵盖视频、音频、镜头和参考四个维度，并系统评估19种闭源及开源模型。

**💡 创新点**

创新点在于（1）构建多维度、复杂情境覆盖的数据集；（2）引入自适应镜头分割自我修正机制、实例化 rubrics 与工具驱动评估；（3）将多级评估指标与人类评分高度相关。

**🔧 技术方法**

采用 VLM（如 Qwen3.5、Gemini 3.1 Pro）进行自我校正与评分；使用多模态感知工具（目标检测、姿态估计等）提供客观证据；基于专家模型、rubric、工具三分层评分；训练 GPT‑5.4 生成脚本并人类审核。

**📊 数据集**

使用自构造的 286 条脚本（共 2198 瞬间）和 96 条参考条件（68 人物图像、65 语音、32 场景图像），覆盖 8 类影片风格、6 种音频类别、6 种语言、15 级镜头数等。

**📈 对比分析**

通过 Spearman 相关系数验证评估与人工评分高度一致（整体 0.915），相较直接 VLM 打分提升 0.25~0.38；对比闭源与开源模型显示闭源系统性能领先，但模块化/代理式开源管线可缩小差距，长镜头和非现实场景仍是主要瓶颈。

**⚠️ 局限性**

局限性包括：对极端长镜头或复杂场景仍表现不佳；评估仍依赖多模态工具与大模型，计算成本高；参考条件下的视觉保真度仍落后于音频克隆；需进一步提升导演级别控制与音视频同步。

---

## 726. TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload

**arXiv ID:** 2605.20179 | [PDF](https://arxiv.org/pdf/2605.20179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

