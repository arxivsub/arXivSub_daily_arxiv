# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-07 | 今日论文总数: 561

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. The Impact of Vocabulary Overlaps on Knowledge Transfer in Multilingual Machine Translation

**arXiv ID:** 2605.04196 | [PDF](https://arxiv.org/pdf/2605.04196v1)

**作者:** Oona Itkonen `[一作]` (University of Helsinki), Jörg Tiedemann `[通讯]` (University of Helsinki)

**通讯引用:** 8696 | [OpenAlex ID](https://openalex.org/A5082417280)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多语言机器翻译框架下，作者通过在域外场景中引入相关语言（瑞典语）与不相关语言（芬兰语）以及联合词表与非联合词表，系统评估词表重叠对知识迁移效果的影响。

**💡 创新点**

创新点在于首次对比联合与非联合词表对跨语言知识迁移的具体贡献，并通过域外设置与辅助语言实验证明：语言相似性和域匹配对迁移效果的影响大于词表重叠。

**🔧 技术方法**

使用技术包括MarianNMT Transformer‑base架构、SentencePiece BPE+Byte‑Fallback tokenizer、prefixing实现非联合词表、marian‑vocab抽取词表、BLEU/ChrF评估以及自定义训练脚本。

**📊 数据集**

使用数据集为Europarl German‑English（半量）与OpenSubtitles2024 Swedish‑English/Finnish‑English（辅助语言），测试集为OpenSubtitles2024 German‑English。

**📈 对比分析**

通过对比基线单语模型、1M/2M训练行数、联合/非联合词表与不同辅助语言的组合，评估BLEU/ChrF得分。实验结果显示，联合词表+相关语言提升约6 BLEU，非联合词表仍优于基线，相关语言优于不相关，表明即使无词表重叠知识迁移仍然发生。

**⚠️ 局限性**

局限性包括：仅涉及两组语言对与单向翻译，数据量和词表规模有限，实验未覆盖多语言或更广域场景，且评估仅基于BLEU/ChrF，缺乏更丰富的质量分析。

---

## 2. Resource Utilization of Differentiable Logic Gate Networks Deployed on FPGAs

**arXiv ID:** 2605.04109 | [PDF](https://arxiv.org/pdf/2605.04109v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 3. Endogenous Regime Switching Driven by Scalar-Irreducible Learning Dynamics

**arXiv ID:** 2605.04054 | [PDF](https://arxiv.org/pdf/2605.04054v1)

**作者:** Sheng Ran `[一作]` (Washington University in St. Louis), Sheng Ran `[通讯]` (Washington University in St. Louis)

**通讯引用:** 4042 | [OpenAlex ID](https://openalex.org/A5071518409)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并验证了一种基于标量不可约（scalar‑irreducible）动力学的自适应学习框架，演示了该框架如何实现内生的、持续的动力学模式切换。

**💡 创新点**

创新点在于提出将学习动力学从传统的标量可约（gradient‑flow）框架转向标量不可约结构，利用内在的旋转分量实现自我调节的机制，突破了外部调度对学习模式切换的依赖。

**🔧 技术方法**

技术手段包括：①Helmholtz分解对学习动力学的结构分类；②分层时序动力学建模（快层为 FitzHugh–Nagumo 动态，慢层为受“压力”调节的可旋转学习规则）；③基于内部状态统计的“坏度”指标，用以驱动慢层结构更新。

**📊 数据集**

本研究未使用传统机器学习数据集，而是通过数值仿真在低维连续动力学系统中验证理论，所用的“数据”为内部状态序列与参数轨迹。

**📈 对比分析**

与传统标量可约学习（梯度下降）以及外部强制扫描相比，标量不可约模型能够持续产生非周期性的内部切换，保持系统的“坏度”较低；实验展示了在相同初始条件下，标量不可约模型的切换频率和多样性显著优于两种对照。

**⚠️ 局限性**

局限性包括：①模型极为简化，仅在二维动力学中验证；②缺乏在实际任务或大规模数据上的实证；③目前尚未给出理论上对切换频率与系统性能的严谨分析；④在更高维或非平稳环境下的稳定性与可扩展性待进一步研究。

---

## 4. Towards Formal Verification of Hybrid Synchronous Programs with Refinement Types

**arXiv ID:** 2605.04377 | [PDF](https://arxiv.org/pdf/2605.04377v1)

**作者:** Serra Z. Dane `[一作]` (University of Michigan), Jean-Baptiste Jeannin `[通讯]` (University of Michigan)

**通讯引用:** 1014 | [OpenAlex ID](https://openalex.org/A5002069299)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

在 Zélus 语言中定义了混合同步程序的零交叉判定，并通过改进的细化类型系统实现对连续动力学和离散重置的形式化验证。

**💡 创新点**

创新点在于给出零交叉的正式模块化定义，扩展了 Zélus 的操作语义和类型系统，并证明了类型安全性。

**🔧 技术方法**

采用细化类型（Refinement Types）、微分动态逻辑（dL）以及 Zélus 的同步语义进行形式化验证。

**📊 数据集**

无公开数据集；实验仅包含两个示例程序（水箱和自动刹车）。

**📈 对比分析**

未进行性能比较；验证通过类型检查完成，验证成本取决于 SMT 求解器的效率。

**⚠️ 局限性**

局限在于仅支持不变式安全性质、缺少演化域约束、仅处理由零交叉触发的离散事件，且对 Zélus 语法有依赖。

---

## 5. From Video-to-PDE: Data-Driven Discovery of Nonlinear Dye Plume Dynamics

**arXiv ID:** 2605.04535 | [PDF](https://arxiv.org/pdf/2605.04535v1)

**作者:** Cesar Acosta-Minoli `[一作]` (GEDES, Universidad del Quindio), Sayantan Sarkar `[通讯]` (State University of New York at Buffalo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套从灰度染料渗流视频到可模拟的偏微分方程的完整管线，包含图像预处理、漂移估计、弱形式稀疏回归、iPINN系数细化和块重采样校准；

**💡 创新点**

创新点在于利用弱形式稀疏回归避免对噪声视频直接求导，预先将由质心估计得到的整体漂移作为已知的运输项，随后通过物理信息网络和Bootstrap重采样分别校准非线性梯度项与扩散项，并通过几何诊断（质心误差、等值前沿半径）选择最优模型；

**🔧 技术方法**

使用的技术包括基于高斯核的弱形式SINDy、STLSQ稀疏回归、全连接tanh网络的iPINN、顺序块Bootstrap以及前沿和质心误差的几何评估，并证明所识别的方程可通过Cole–Hopf变换线性化；

**📊 数据集**

实验数据来自一段约34秒的顶视灰度染料渗流视频，原始帧数约1000帧，经过裁剪、归一化、Gaussian平滑后映射到200×200像素网格；

**📈 对比分析**

通过与传统的仅含漂移和扩散项的阿德维克–扩散基线进行对比，验证窗口的相对RMSE降至约6%，前沿半径误差和质心误差亦显著低于基线，说明所识别模型在像素级和几何层面均优于传统模型；

**⚠️ 局限性**

主要局限包括仅在单一实验视频上验证，所识别的系数仅以图像坐标为单位，漂移假设为均匀向量，预处理选择（裁剪、平滑等）会影响梯度特征，且未考虑空间变异的速度场，未来需在多实验、物理校准及更复杂流动场景下进一步验证。

---

## 6. Queue-Aware and Resilient Routing in LEO Satellite Networks Using Multi-Agent Reinforcement Learning

**arXiv ID:** 2605.04448 | [PDF](https://arxiv.org/pdf/2605.04448v1)

**作者:** Mudassar Liaq `[一作]` (University of Manitoba), Peng Hu `[通讯]` (University of Manitoba)

**通讯引用:** 71550 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种队列感知的多智能体深度强化学习（MA‑DRL）框架，用于在低地球轨道（LEO）卫星网络中进行分布式路由决策。

**💡 创新点**

创新点包括：①将队列延迟和链路可靠性（resilience score）纳入路由奖励函数，实现对拥塞与鲁棒性的双重优化；②在每颗卫星上部署本地智能体，利用邻近卫星的队列信息进行局部决策，从而避免传统全局最短路径算法对频繁拓扑变化的高计算与信令开销；③在训练阶段采用集中式全局 Q‑网络，随后分布到各卫星并进行在线微调，兼顾全局视野与局部适应。

**🔧 技术方法**

使用的技术主要有：Double Deep Q‑Network（DDQN）强化学习、经验回放、目标网络、ε‑greedy 探索策略、Huber 损失、Adam 优化器；在仿真中构建了 Starlink Shell 1 典型参数的仿真环境，并在此基础上实现了 MA‑DRL、SARSA 与 Dijkstra 三种算法。

**📊 数据集**

实验数据集为基于 Starlink Shell 1 的仿真数据：72 条轨道、1584 颗卫星、200 站地面终端、10 W/20 W 的发射功率、500 MHz 带宽等硬件与物理层参数，仿真覆盖了不同背景流量、拥塞与链路鲁棒性场景。

**📈 对比分析**

与传统 Dijkstra 最短路径和集中式 SARSA 的对比表明：MA‑DRL 的平均端到端延迟为 49.31 ms（Dijkstra 为 38.54 ms），但在需要频繁重计算时的决策开销约为 Dijkstra 的 50%；在不同重计算频率下，MA‑DRL 的路径变化率与 Dijkstra 低得多，保持更稳定的延迟；在鲁棒性得分方面，Dijkstra 最高，而 MA‑DRL 与 SARSA 的得分相近，略低于全局视野算法。

**⚠️ 局限性**

局限性包括：①由于只利用本地队列与邻近信息，MA‑DRL 的鲁棒性低于全局最短路径算法；②实验仅在仿真环境中验证，缺乏真实卫星网络的部署与验证；③未考虑链路失效、天候扰动等更复杂的物理层失效模型；④在线微调依赖于持续的数据流，可能在高动态场景下需要更频繁的模型更新。

---

## 7. Position: the Stochastic Parrot in the Coal Mine. Model Collapse is a Threat to Low-Resource Communities

**arXiv ID:** 2605.04127 | [PDF](https://arxiv.org/pdf/2605.04127v1)

**作者:** Devon Jarvis `[一作]` (University of Witwatersrand), Stefano Sarao Mannelli `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨生成式模型在多代训练中出现的模型崩溃现象及其对低资源社区的环境与文化影响，并呼吁学术界加强关注和研究。

**💡 创新点**

将模型崩溃与公平、低资源社区利益相结合，提出新的研究视角与行动呼吁，强调其对技术民主化的威胁。

**🔧 技术方法**

主要采用文献综述、理论分析与概念框架构建，未进行实验或算法实现。

**📊 数据集**

参考公开数据集与案例（如GPT‑2、CC‑100等）和已有研究，未产生新的数据集。

**📈 对比分析**

无实验比较，本文为立场论文，未给出具体性能指标或基准。

**⚠️ 局限性**

缺乏实证验证，研究范围受限于现有文献，未深入探讨模型崩溃检测与缓解的技术细节。

---

## 8. Coupled-NeuralHP: Directional Temporal Coupling Between AI Innovation Exposure and Public Response

**arXiv ID:** 2605.04194 | [PDF](https://arxiv.org/pdf/2605.04194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 9. Efficient Handwriting-Based Alzheimer,s Disease Diagnosis Using a Low-Rank Mixture of Experts Deep Learning Framework

**arXiv ID:** 2605.04079 | [PDF](https://arxiv.org/pdf/2605.04079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 10. One Pool, Two Caches: Adaptive HBM Partitioning for Accelerating Generative Recommender Serving

**arXiv ID:** 2605.04450 | [PDF](https://arxiv.org/pdf/2605.04450v1)

**作者:** Wenjun Yu `[一作]` (Hong Kong Baptist University), Amelie Chi Zhou `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 463 | [OpenAlex ID](https://openalex.org/A5015692437)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于强化学习的自适应HBM分配与请求调度框架，以协同管理生成式推荐模型（GR）中嵌入缓存与KV缓存的竞争资源，提升推理尾部延迟；

**💡 创新点**

创新点在于：①三层PPO控制器（冻结基策略+在线残差适配+突发恢复）实现32 µs的在线决策；②仅在EMB缓存边界做增删，不移动KV状态，避免关键路径干扰；③融合KV与EMB命中率与负载的综合调度器，打破传统单一侧向调度的局限；

**🔧 技术方法**

核心技术包括：强化学习（PPO）、在线残差适配与恢复控制、GPU HBM边界微调、分布式调度与负载感知、异步内存重填与带宽节流；

**📊 数据集**

实验使用三大工业数据集：Taobao、Amazon‑Video‑Games、Amazon‑Books；模型采用Meta HSTU框架，GPU为32节点A100；

**📈 对比分析**

与KV‑Opt、KV‑EMB‑Opt等基线相比，在三种工作负载（Steady、Trend、Burst）下P99延迟下降24–38%，SLO满足率≥93.5%，且决策开销仅32 µs；

**⚠️ 局限性**

局限性包括：仅验证在A100集群上的性能，未覆盖更高HBM容量或不同GPU架构；RL模型训练耗时与超参敏感；对极端极低热点比例或非常大序列长度的适配仍待进一步验证。

---

## 11. Parallel Prefix Verification for Speculative Generation

**arXiv ID:** 2605.04263 | [PDF](https://arxiv.org/pdf/2605.04263v1)

**作者:** Yuncheng Yao `[一作]` (Duke University), Danyang Zhuo `[通讯]` (Duke University)

**通讯引用:** 639 | [OpenAlex ID](https://openalex.org/A5007943946)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 PArallel pRefix Speculative Engine（PARSE）的框架，利用并行前缀验证在语义层面加速大型语言模型推理。

**💡 创新点**

核心创新在于使用自定义注意力掩码和多重对话模板后缀，在单次前向推理中同时评估所有前缀的正确性，从而消除传统语义级验证的序列依赖。

**🔧 技术方法**

技术实现包括：1) 轻量级草稿模型生成完整答案；2) 目标模型通过阈值置信度进行全答与多前缀判定；3) 自定义注意力掩码实现并行前缀验证；4) 与 token‑level 方案（如 EAGLE‑3）无缝组合。

**📊 数据集**

在 Qwen3‑235B‑A22B‑FP8 作为目标模型、Qwen3‑8B 或 GLM‑4.7‑FP8 作为草稿模型上进行实验；基准覆盖 MMLU‑Redux、MMLU‑Pro、GPQA、MATH、GSM8K、HumanEval、MBPP、MT‑Bench 等多种任务。

**📈 对比分析**

相较于基线（纯自回归）和单独的 token‑level 方案，PARSE 在所有基准上实现了 1.25×–4.3× 的吞吐量提升；与 EAGLE‑3 组合后可达 1.6×–4.5×；与 SpecReason 等语义级验证方法对比，PARSE 在准确率保持一致的同时吞吐量显著更高。

**⚠️ 局限性**

主要局限在于草稿模型与目标模型的兼容性——在跨模型族（不同 tokenizer、预训练数据）时，接受率下降导致速度提升受限；此外，前缀验证的粒度受 Δ 选择影响，过粗粒度可能导致更多错误需要全量重写。

---

## 12. ipc_shared_ptr: A Publish/Subscribe-Aware Smart Pointer for Cross-Process Object Lifetime Management

**arXiv ID:** 2605.04226 | [PDF](https://arxiv.org/pdf/2605.04226v1)

**作者:** Takahiro Ishikawa-Aso `[一作]` (TIER IV Incorporated), Shinpei Kato `[通讯]` (University of Tokyo)

**通讯引用:** 4700 | [OpenAlex ID](https://openalex.org/A5101496340)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了跨进程对象生命周期管理机制ipc_shared_ptr，支持ROS 2真零复制IPC，并在Agnocast中实现该机制。

**💡 创新点**

将Birrell的引用列表专门化到pub/sub域，只在订阅者0↔1切换时更新全局元数据，降低通信开销；采用单写者（kernel module）架构实现数据平面与控制平面一致性，消除交叉平面竞争。

**🔧 技术方法**

两级引用计数、单写者元数据管理（内核模块）、POSIX MQ 通知、共享内存堆映射、ROS 2 QoS与Transient Local、控制平面与数据平面分离、用户空间库。

**📊 数据集**

使用Autoware v1.7.1开源ROS 2堆栈作为基准工作负载（227主题，平均3.4订阅者等），并采用固定1KB消息尺寸进行基准测试。

**📈 对比分析**

通过覆盖主题数、订阅者数量和发布速率的三维实验，测量publish、receive和E2E延迟。Agnocast在Autoware规模下E2E p99.9维持几十微秒，低于iceoryx2；但在高订阅者数时，由于每订阅者的POSIX MQ系统调用导致O(S)成本，Agnocast可能逊色。

**⚠️ 局限性**

受单写者瓶颈和POSIX MQ通知导致O(S)延迟；需要内核模块；对高订阅者数量的负载不如冰箱；基准使用固定消息大小，可能低估了不规则消息对Agnocast的优势。

---

## 13. Enhancing the interpretability of spatially variable N2O model predictions with soft sensors during wastewater treatment

**arXiv ID:** 2605.04082 | [PDF](https://arxiv.org/pdf/2605.04082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 14. Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa

**arXiv ID:** 2605.04177 | [PDF](https://arxiv.org/pdf/2605.04177v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 15. Toward Human-AI Complementarity Across Diverse Tasks

**arXiv ID:** 2605.04070 | [PDF](https://arxiv.org/pdf/2605.04070v1)

**作者:** Yuzheng Xu `[一作]` (University of Tokyo), Rishub Jain `[通讯]` (NIT Agartala)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了跨知识、事实性、长上下文推理与欺骗检测的多域基准，并评估了人类、AI以及人机协作（混合化、Top‑2辅助和子任务委派）的监督效果。

**💡 创新点**

证明即使在高性能AI（GPT‑5‑mini）主导的情形下，通过信度路由与针对性辅助也能实现微量的互补提升，且人机误差在不同任务类型上系统性不重叠，揭示了提升人机互补的潜在空间。

**🔧 技术方法**

使用置信度校准（等距回归）、阈值路由、基于AI自报置信的Top‑2答案显示、子任务拆分与重组成分，比较了人类与AI在不同模式下的准确率。

**📊 数据集**

共收集了1,886道题，覆盖9个公开数据集（如 FACTS Search、QuALITY、Big‑Bench、GPQA Diamond、Humanity’s Last Exam、SHADE‑Arena、Web of Lies 等），其中低置信度子集为191道题。

**📈 对比分析**

在全量测试集（952道题）中，最佳混合化策略仅提升0.4pp（从68.9%到69.3%）；Top‑2辅助在低置信度子集中将人类准确率从28.4%提升至38.3%（优于AI的37.7%），子任务委派在可分解任务中提升10–25pp，但在欺骗检测中无效；总体而言提升有限，但证实人机误差分离可为进一步优化提供方向。

**⚠️ 局限性**

局限包括样本量不足（功效仅5.4%）、使用的基准可能已被GPT‑5‑mini预训练泄漏、实验仅使用公开数据集和普通参与者（非专家），且仅评估单一AI模型与两种辅助方式，未覆盖更广阔的路由信号与辅助界面。

---

## 16. Regularized Centered Emphatic Temporal Difference Learning

**arXiv ID:** 2605.04100 | [PDF](https://arxiv.org/pdf/2605.04100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 17. Adaptive Consensus in LLM Ensembles via Sequential Evidence Accumulation: Automatic Budget Identification and Calibrated Commit Signals

**arXiv ID:** 2605.04236 | [PDF](https://arxiv.org/pdf/2605.04236v1)

**作者:** Roberto Medina `[一作]` `[通讯]` (Independent Researcher), Roberto Medina (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于迭代集成的决策停止方法DASE，用多轮投票和自适应终止阈值来决定何时给出答案并可生成可审计的推理轨迹。

**💡 创新点**

主要创新在于（1）通过“右墙/左墙”提交类型实现与单呼叫置信度互补的路由分区；（2）设计了基于持续性阈值和空间战场（DASE‑Spatial）的自适应停止启发式；（3）提供完整的机器可读审计记录。

**🔧 技术方法**

技术上采用：多轮提示式大模型集成、投票与投票一致性检测、全局频率回退、空间战场终止规则、持续性阈值策略以及对比实验中的统计检验（McNemar+BH‑FDR）。

**📊 数据集**

使用的基准数据集包括：AIME 2010‑2023（N≈261）、AIME‑300（N=300）以及GPQA‑Extended（N=546）等。

**📈 对比分析**

与Self‑Consistency、Debate‑Dense、Debate‑Sparse、BoN‑V以及Claude Opus 4.6等基线比较。DASE在GPQA‑Extended上达到70.0%（与Debate‑Dense相当）但仅需10%注入带宽；在AIME‑300上使用W=8达到65.0%，比Debate‑Sparse高约6pp，且自适应停止是主要贡献。路由分区的置信度差距与Opus相当（≈25pp）。

**⚠️ 局限性**

局限性包括：路由分区的比较仅在AIME 2010‑2023上完成，GPQA‑Extended的路由实验尚未完成；W（战场宽度）选择为探索性结果；缺少更丰富注入的对称消融；模型大小与token预算共同变化导致最优W不可归因；在高实时交互场景中延迟过高；最新单呼叫模型的绝对准确率已超过DASE。

---

## 18. SCOUT: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States

**arXiv ID:** 2605.04496 | [PDF](https://arxiv.org/pdf/2605.04496v1)

**作者:** Zhenliang Zhang `[一作]` (Peking University), Xiaojun Wan `[通讯]` (Peking University)

**通讯引用:** 9743 | [OpenAlex ID](https://openalex.org/A5029568096)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Scout，一种主动信息搜寻框架，用于在百万级长文本中进行高效、准确的推理，将文档视为可交互的环境，利用分离的认知状态进行答案生成。

**💡 创新点**

核心创新在于：① 将探索历史与推理状态解耦，形成稀疏的、可追溯的认知状态；② 用出处锚定的认知单元保证信息真实性；③ 通过状态层面的差距诊断实现对齐、进展感知与充分性控制；④ 设计多分辨率信息搜寻动作与状态管理工具，形成自适应的主动采样策略。

**🔧 技术方法**

技术实现包括：基于 POMDP 的决策框架；ReAct 风格的多步骤 LLM 交互；专门的认知状态管理工具；来源锚定的语义抽取；状态差距诊断（gap detection）；以及多分辨率文本浏览动作（粗细级别的文本探索）。

**📊 数据集**

主要评测使用两类公开长文档基准：① 需要跨段多跳推理的长文本推理基准（类似 BigBench Hard Reasoning / Long‑Document Multi‑Hop QA）；② 以百万级真实文档（小说、代码库）为主的超长文本推理基准（类似 Long‑Document Benchmark）。

**📈 对比分析**

实验采用统一的后端（Claude‑Sonnet‑4.5），与领先的长文本 LLM（Gemini‑3‑Pro、GPT‑5.1 等）以及现有的专用长文本代理（ReadAgent、GraphReader、MemAgent、Claude‑Code）进行对比。结果显示，Scout 在两大基准上均取得最高准确率，并在 token‑efficiency（准确率/token 费用）上优于所有竞争模型，token 费用相对单次 LLM 推理可降低约 8 倍，且在 1M+ token 长度下保持性能稳定。

**⚠️ 局限性**

主要局限包括：① 多步骤 LLM 调用导致相较单次推理的时延更高；② 当前不支持跨查询共享知识，需对同一文档多次完整搜寻；③ 仅验证了信息稀疏、纯文本的场景，对信息密集或多模态长文档的适用性尚未探索。

---

## 19. Constructing Suffixient Arrays Revisited

**arXiv ID:** 2605.04258 | [PDF](https://arxiv.org/pdf/2605.04258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 20. OpenCLAW-Nexus: A Self-Reinforcing Trust Framework for Byzantine-Resilient Decentralized Federated Learning

**arXiv ID:** 2605.04091 | [PDF](https://arxiv.org/pdf/2605.04091v1)

**作者:** Wenyang Jia `[一作]` (Peking University), Kai Lei `[通讯]` (Peking University)

**通讯引用:** 253512 | [OpenAlex ID](https://openalex.org/A5071127149)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OpenCLAW-Nexus，一个基于折扣式 Beta 信誉模型的自强化信任框架，用于去中心化联邦学习中的参与者选择、加权聚合和模型验证。

**💡 创新点**

创新点：将信誉评估统一为单一可持续更新的 Beta 模型，形成闭环信任循环；设计了信誉加权聚合 Rep‑FedAvg 与信誉权重 BFT 共识；无需中心根数据集，改用公共验证基准与分布式评估。

**🔧 技术方法**

采用折扣 Beta 信誉模型、信誉加权聚合 Rep‑FedAvg、信誉权重 PBFT 共识、记录级 DP‑SGD、Kademlia DHT、Gossip 广播、分布式评估与加权投票、分布式硬件能力检测等技术。

**📊 数据集**

主要使用 CIFAR‑10 数据集（ResNet‑18 预训练模型），划分为公共验证集和隐藏测试集。

**📈 对比分析**

与 FedAvg、FLTrust、BALANCE、Krum、Trimmed Mean 等基线对比。实验显示 Rep‑FedAvg 在 20% Byzantine 环境下可达 72.6% 准确率（≈ 0.5pp 接近 FLTrust），在 300 Sybil 攻击下模型验证正确率 84.2%（比 PoW‑PBFT 及 Stake‑PBFT 高 21.4pp 和 36.6pp）。

**⚠️ 局限性**

局限性：评估仅针对可验证的监督任务；依赖公共验证基准，存在基准过拟合风险；记录级 DP 仅保护样本隐私，未覆盖参与者身份；对自适应攻击与动态成员的理论证明不足；统计显著性分析未充分展开。

---

## 21. Accountable Agents in Software Engineering: An Analysis of Terms of Service and a Research Roadmap

**arXiv ID:** 2605.04532 | [PDF](https://arxiv.org/pdf/2605.04532v1)

**作者:** Christoph Treude `[一作]` (Singapore Management University), Christoph Treude `[通讯]` (Singapore Management University)

**通讯引用:** 5297 | [OpenAlex ID](https://openalex.org/A5077658936)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对九大 AI 编码助手与代理工具的 14 份服务条款进行定性比较，揭示了当前合同框架下的所有权、责任与数据治理分配模式，并提出了面向可追溯性、治理感知与责任建模的研究路线图。

**💡 创新点**

创新点在于首次将 AI 开发工具的治理文档纳入系统化分析框架，明确了“输出所有权”与“使用责任”共置的普遍现象，并从实践角度指出了合同与实际代理工作之间的脱节，为后续责任建模与技术治理提供了方向。

**🔧 技术方法**

采用了定性文本分析与对照编码方法，对条款中的所有权、责任、数据治理和可接受使用等四大维度进行聚类与对比；并结合案例说明对委托、赔偿与责任上限的差异化处理。

**📊 数据集**

使用的数据集为 14 份公开可获取的服务条款（OpenAI、GitHub Copilot、Anthropic、AWS CodeWhisperer、JetBrains、Cursor、Replit、Sourcegraph、Google 等），覆盖 9 个主流 AI 编码工具。

**📈 对比分析**

比较方法为双人编码并讨论不确定条款，最终形成跨供应商的责任与权利映射。该方法揭示了 99% 的条款均将输出所有权授予用户，但责任与赔偿多由用户承担；与传统手工编码相比，代理工具在责任分配上表现出更大差异，表明现行合同模式不足以应对高自治场景。

**⚠️ 局限性**

局限性包括：①仅分析条款文本，未进行法律解释或合规性评估；②未收集开发者或组织在实际使用中的感知与行为数据；③缺乏量化指标衡量代理工具对责任转移的具体影响，导致研究更偏理论与策略层面。

---

## 22. Resilient AI Supercomputer Networking using MRC and SRv6

**arXiv ID:** 2605.04333 | [PDF](https://arxiv.org/pdf/2605.04333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 23. Lookahead Drifting Model

**arXiv ID:** 2605.04060 | [PDF](https://arxiv.org/pdf/2605.04060v1)

**作者:** Guoqiang Zhang `[一作]` (University of Exeter), W. Bastiaan Kleijn `[通讯]` (Victoria University of Wellington)

**通讯引用:** 9724 | [OpenAlex ID](https://openalex.org/A5087492771)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在每次训练迭代中按顺序计算并利用多组漂移项（lookahead drifting）的生成模型训练方法。

**💡 创新点**

创新点在于：①将漂移项分层计算，利用前一层生成的分布来构造下一层漂移项，从而获取高阶梯度信息；②通过多层漂移的堆叠实现类似残差网络的跳跃连接，提升信息传递效率；③保持漂移项的反对称性，保证当生成分布与真实分布相等时漂移项为零。

**🔧 技术方法**

主要技术包括：漂移模型（drifting model）与拉普拉斯核的梯度匹配；按层顺序迭代计算漂移项；使用 DINOv3 编码器辅助训练；采用 FID 作为评价指标；在实验中对比标准漂移模型与 lookahead 漂移模型。

**📊 数据集**

使用的公开数据集为 CIFAR‑10；实验也包含了简化的 toy 示例。

**📈 对比分析**

与标准漂移模型做对比，利用相同的网络结构、学习率和优化器；结果显示在单 GPU 训练下，lookahead 漂移（k=1）在 30k/40k/50k 迭代阶段的 FID 分别为 17.43/17.12/18.81，均显著低于基线（30.15/29.65/29.67）。

**⚠️ 局限性**

局限性包括：仅在 toy 示例和 CIFAR‑10 上验证，缺乏更大规模数据（如 ImageNet）的实验；随着 k 增大，计算量和内存需求急剧上升；对超参数（如温度 τ、kernel 选择）的敏感性未做深入探讨。

---

## 24. Predict-then-Diffuse: Adaptive Response Length for Compute-Budgeted Inference in Diffusion LLMs

**arXiv ID:** 2605.04215 | [PDF](https://arxiv.org/pdf/2605.04215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 25. Telegraph English: Semantic Prompt Compression via Structured Symbolic Rewriting

**arXiv ID:** 2605.04426 | [PDF](https://arxiv.org/pdf/2605.04426v1)

**作者:** Mikhail L. Arbuzov `[一作]` (Independent Researcher), Alexey A. Shvets `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为Telegraph English（TE）的提示压缩协议，通过将自然语言重写为符号丰富、结构化的短语，从而在保留信息的前提下实现约50%的标记压缩；

**💡 创新点**

创新点在于：①将压缩与语义分块视为同一操作，每行只包含一个原子事实；②使用符号词汇和标签实现逻辑与关系的显式表达；③压缩比自适应，能根据信息密度动态调整；④压缩后文本既可直接输入LLM，又可被索引、检索与动态更新；

**🔧 技术方法**

技术手段包括：基于预先定义的语法v5的LLM压缩器（o4-mini），12点质量检查门、六步推理序列；使用符号词汇、标签化的文法；以及在评估中使用GPT-4.1生成多选问答并与压缩文本比较；

**📊 数据集**

使用的数据集为LongBench‑v2，过滤后得到339份文档，拆分为4,081个块级问答对（key_facts）和801个细节级问答对（fine_facts）；

**📈 对比分析**

评估方法是将原始文本与TE压缩文本分别输入不同模型（GPT‑4.1、GPT‑4o‑mini、GPT‑4.1‑nano等），测量多选题准确率；结果显示TE在key_facts上与GPT‑4.1的准确率仅低0.9%，但在fine_facts上优势高达3–11个百分点；与LLMLingua‑2的50%压缩比相比，TE在所有模型上均保持更高准确率，尤其在小模型和细节任务中表现突出；

**⚠️ 局限性**

局限性包括：①压缩需依赖LLM调用，导致延迟和成本；②评估仅基于OpenAI专有模型，缺乏可复现性；③仅支持英文，符号词汇迁移到其他语言困难；④未系统评估压缩模型对不同LLM的敏感性；⑤生成的问答对与评估模型相同，可能存在偏差；⑥动态上下文管理和多轮推理效果尚未在实践中验证。

---

## 26. DiffCap-Bench: A Comprehensive, Challenging, Robust Benchmark for Image Difference Captioning

**arXiv ID:** 2605.04503 | [PDF](https://arxiv.org/pdf/2605.04503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 27. Model synthesis and identifiability analysis of stiff chemical reaction systems with inVAErt networks

**arXiv ID:** 2605.04134 | [PDF](https://arxiv.org/pdf/2605.04134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 28. Simultaneous CNN Approximation on Manifolds with Applications to Boundary Value Problems

**arXiv ID:** 2605.04126 | [PDF](https://arxiv.org/pdf/2605.04126v1)

**作者:** Hanfei Zhou `[一作]` (Fudan University), Lei Shi `[通讯]` (Fudan University)

**通讯引用:** 11653 | [OpenAlex ID](https://openalex.org/A5100427526)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在紧致黎曼流形上使用卷积神经网络（CNN）进行同时逼近和解椭圆边值问题的理论与实验方法。

**💡 创新点**

创新点包括：① 同时 Sobolev 逼近理论，证明逼近误差仅依赖流形内在维度；② 多通道 CNN 架构实现常数深度且参数量优越；③ 物理信息 CNN（PICNN）框架，引入谱边界损失，解决传统 PINN 低阶边界约束导致的不稳定性；④ 通过 FFT 实现高阶 Sobolev 边界正则化，避免了高阶积分核。

**🔧 技术方法**

技术方法：ReLU–ReQU CNN 架构（单通道与多通道），有限差分/有限元的卷积核逼近，谱分解与边界 Laplace–Beltrami 频率权重，FFT 计算频域边界损失，Adam 训练，GeLU 激活提高可微性。

**📊 数据集**

使用的实验数据集为两类流形子域：上半球（S² 上的二维子域）和上半托罗伊（T² 上的二维子域），每个流形具有圆形或闭合曲线边界，采样点来自均匀网格。

**📈 对比分析**

比较方法：将谱 Sobolev 边界损失与传统 L² 边界惩罚进行对比，评价指标为相对 L² 与相对 H² 误差，实验结果显示谱损失在两种流形上均实现更低误差、收敛更快、训练曲线更平稳。

**⚠️ 局限性**

局限性：① 对高维或复杂边界（非光滑、分块）仍需进一步理论与实现扩展；② 训练过程对学习率、采样密度高度敏感，可能出现优化不稳定；③ 需要频域算子谱信息，若无闭合形式或可预计算基，FFT 方案难以直接应用。

---

## 29. Hardware-Aware Neural Feature Extraction for Resource-Constrained Devices

**arXiv ID:** 2605.04282 | [PDF](https://arxiv.org/pdf/2605.04282v1)

**作者:** Francesco Tosini `[一作]` (Politecnico di Milano), Diana Trojaniello `[通讯]` (EssilorLuxottica)

**通讯引用:** 992 | [OpenAlex ID](https://openalex.org/A5079923755)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练了一个名为Gideon的神经网络特征提取器，可在微控制器级别的STM32N6芯片上实现实时运行。

**💡 创新点**

创新点在于将量化稳定性、内存占用与数据流规律直接纳入训练目标，采用知识蒸馏结合可微架构搜索，并用Affine层代替BatchNorm、控制描述子维度以及自适应阈值来显著提升INT8鲁棒性。

**🔧 技术方法**

使用的技术包括Relational知识蒸馏（KL损失在自相似矩阵上）、可微神经架构搜索（DNAS + Gumbel-Softmax）、焦点损失变体、Adaptive阈值策略以及Affine层替换BatchNorm。

**📊 数据集**

训练数据来自TUM‑VI 29000张图像，评估使用HPatches基准测试集。

**📈 对比分析**

与SuperPoint、ORB等方法对比，Gideon在STM32N6上实现9.003 ms/111 fps、内存占用不到1.5 MB，INT8版本几乎无性能损失甚至略优；在HPatches上可重复率≈0.52、光照正确率≈93.7%、视角正确率≈61.4%，显著优于ORB且速度快于SuperPoint。

**⚠️ 局限性**

局限性包括仅验证特征级别性能，未集成完整SLAM流水线；训练未做量化感知优化，量化改进可能受统计波动影响；适用性受限于特定MCU/NPU平台，需在更多硬件和更复杂场景中进一步验证。

---

## 30. Climate-based Pre-screening of Self-sustaining Regreening Opportunities in Drylands: A Case Study for Saudi Arabia

**arXiv ID:** 2605.04206 | [PDF](https://arxiv.org/pdf/2605.04206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 31. Road Risk Monitor: A Deployable U.S. Road Incident Forecasting System with Live Weather and Road-Level Tiles

**arXiv ID:** 2605.04242 | [PDF](https://arxiv.org/pdf/2605.04242v1)

**作者:** Anton Ivchenko `[一作]` `[通讯]`, Anton Ivchenko

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套可部署的美国道路事故预测系统，涵盖数据采集、模型训练、实时天气适配、瓦片服务以及运行时打包与服务启动。

**💡 创新点**

创新点在于将多源全国道路安全数据与实时天气结合，采用双尺度模型（H3全局基线+道路段预测），并提供完整的从数据到可生产部署的全流程实现。

**🔧 技术方法**

采用H3空间分辨率、特征工程（天气、时间、历史计数）、梯度提升树模型、FastAPI接口、瓦片渲染、脚本化运行时打包等技术。

**📊 数据集**

使用的数据集包括FARS致命车祸数据、NOAA ISD-Lite气象数据、TIGER/Line道路几何、US-Accidents道路事件、NWS实时天气API以及H3空间索引。

**📈 对比分析**

通过在留年测试集上评估基线模型，获得较高的AUROC和平均精度；段级模型在内部同管道留年评估中表现出极高的区分度，但并未作为主要科学主张。

**⚠️ 局限性**

局限性包括基线模型仅基于致命车祸数据，段级模型受采样负样本和历史计数特征的影响，实时适配仅使用天气信息，缺乏交通流量、工地等更丰富的环境信号。

---

## 32. QUIVER: Cost-Aware Adaptive Preference Querying in Surrogate-Assisted Evolutionary Multi-Objective Optimization

**arXiv ID:** 2605.04267 | [PDF](https://arxiv.org/pdf/2605.04267v1)

**作者:** Florian A. D. Burnat `[一作]` `[通讯]` (University of Warwick), Florian A. D. Burnat (University of Warwick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种成本感知的交互式进化多目标优化算法QUIVER，能够在目标评估和不同偏好查询（对比查询PS和不等价调整IA）之间动态分配预算，旨在最小化决策者最终推荐解的效用后悔；

**💡 创新点**

核心创新在于将评估与多模态偏好查询统一为行动空间，并通过价值-信息（VOI）与成本的比值来选择下一个行动，从而实现信息增益最大化与成本最小化的平衡；

**🔧 技术方法**

采用NSGA-II作为优化后端，使用粒子滤波对线性效用权重进行后验推断，利用蒙特卡洛估计信息增益和评价价值的期望，依据VOI/成本阈值做行动决策；

**📊 数据集**

在标准多目标测试套件DTLZ（DTLZ2）和WFG（WFG4、WFG9）上进行实验，假设决策者权重服从Dirichlet分布并产生带噪声的PS与IA反馈；

**📈 对比分析**

与Eval‑only、PS‑only、IA‑only、固定调度和随机调度等基线（均采用相同NSGA-II后端）对比，结果显示在难度较高的WFG4、WFG9上QUIVER的效用后悔分别为2.14和2.82，比分母约25%更优；在DTLZ2上所有方法均达近零后悔；

**⚠️ 局限性**

局限包括：假设线性效用仅能覆盖凸 Pareto 前沿，无法处理非凸或分离前沿；使用合成决策者而非真实人类，忽略疲劳、噪声随时间变化等因素；粒子滤波维度和计算开销在极大规模问题上可能受限；

---

## 33. A Regulatory Governance Framework for AI-Driven Financial Fraud Detection in U.S. Banking: Integrating OCC, SR 11-7, CFPB, and FinCEN Compliance Requirements for Model Development, Validation, and Monitoring Lifecycles

**arXiv ID:** 2605.04076 | [PDF](https://arxiv.org/pdf/2605.04076v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 34. Are Multimodal LLMs Ready for Clinical Dermatology? A Real-World Evaluation in Dermatology

**arXiv ID:** 2605.04098 | [PDF](https://arxiv.org/pdf/2605.04098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 35. TSCG: Deterministic Tool-Schema Compilation for Agentic LLM Deployments

**arXiv ID:** 2605.04107 | [PDF](https://arxiv.org/pdf/2605.04107v1)

**作者:** Furkan Sakizli `[一作]` `[通讯]` (Independent Researcher), Furkan Sakizli (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Token‑Context Semantic Grammar (TSCG)，一种基于编译器的工具架构语法压缩技术，能将 JSON 形式的工具定义转换为紧凑文本表示，从而在保持语义完整性的同时显著减少 token 消耗并提升模型的工具调用准确率。

**💡 创新点**

创新点在于：①将协议层的 JSON 与 LLM 输入格式做结构性映射，解决了“协议不匹配”导致的能力崩塌；②设计了八个可组合的确定性压缩算子，理论上可实现 ≥51% 的压缩，并在实践中达到了 50–72%；③提出格式-压缩分解法，证明格式转换是提升性能的主因；④构建了首个规模化工具压缩基准 TAB，并在 12 种模型（4B–32B 以及 3 个 frontier）上验证，提供四类模型行为分类与部署建议。

**🔧 技术方法**

使用的技术包括：BPE tokenizer 兼容的表述选择、约束优先布局 (CFL)、因果前向排序 (CFO)、语义密度最大化 (SDM)、分隔符角色优化 (DRO)、因果闭包原则 (CCP)、因果访问得分 (CAS) 与选择性锚点复制 (SAD‑F) 等算子；在实验中使用 OpenAI GPT‑4o/5.2、Anthropic Claude Sonnet 4、Ollama 上运行的 Phi‑4、Mistral 7B、Gemma 3‑4B/12B、Llama‑3.1‑8B、Qwen‑3‑4B/14B、Qwen‑2.5‑Coder 32B 等模型；使用的评测数据集包括自构建的 TAB benchmark（5 个场景、约 12,000+ 调用）、Berkeley Function Calling Leaderboard、GSM‑8K 以及实测 MCP 服务器等。

**📊 数据集**

数据集主要包括：
- TAB Benchmark：5 个工具使用场景（A–E）共计约 12,560 API 调用，覆盖 10–100 个工具、20–50 任务；
- Berkeley Function Calling Leaderboard (BFCL)：约 60 次调用；
- GSM‑8K 语义推理任务（200 次调用）；
- 轻量与重量 MCP 服务器（轻量 43–100 工具，重量 43 工具）用于规模性验证；
- 额外的 2,520 次 json‑text 与 840 次 30B 评测等。

**📈 对比分析**

对比方法：在同一模型、同一工具集下分别使用自然 JSON（Native FC）、完整 JSON 文本（json‑text）、TSCG 编译后文本（TSCG 或 TSCG+SAD）。
- 对 frontier 模型，TSCG 在 Scenarios A/B 的平均提升为 +10.9 百分点（Accuracy‑Retained Ratio 108–181%），同时 50–72% token 节省；
- 对 4B–14B 小模型，TSCG 将 0–49% 的工具调用成功率提升至 65–90%，主要依靠格式转换；
- 在 BFCL 上验证，ARR 108% 与 46.8% token 节省；
- 在实测 MCP 服务器上，TSCG 在重型 schema 上仍保持 +5 百分点；
- 在文本基线下，压缩效应几乎为零或负值，表明压缩收益主要来自格式迁移。

**⚠️ 局限性**

局限性包括：
- 仅验证了工具选择与参数提取，未覆盖生成质量、对话连贯性或完整任务完成；
- 对多语言支持缺乏评测；
- 仅在 12 种模型上验证，未知对未测试架构（如 Mamba、RWKV、思维模型）效果；
- 对 Frontier 模型的版本演进和 API 漏洞更新敏感，需要持续重新验证；
- 在 Class‑1 模型下，TSCG 对文本模式会导致准确率下降；
- 对比其他压缩方案（如 MCP Code Mode、wrapper‑tool）尚无直接对齐基准；
- benchmark 自构造，虽然通过 BFCL 验证但仍需第三方多样化验证。

---

## 36. StableI2I: Spotting Unintended Changes in Image-to-Image Transition

**arXiv ID:** 2605.04453 | [PDF](https://arxiv.org/pdf/2605.04453v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 37. Root-Cause-Driven Automated Vulnerability Repair

**arXiv ID:** 2605.04251 | [PDF](https://arxiv.org/pdf/2605.04251v1)

**作者:** Hulin Wang `[一作]` (Arizona State University), Tiffany Bao `[通讯]` (Arizona State University)

**通讯引用:** 929 | [OpenAlex ID](https://openalex.org/A5076987446)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于根因驱动的漏洞修复代理Kumushi，解决 LLM 修复中定位不准与浅层修复的问题。

**💡 创新点**

创新点在于采用多样化动态缺陷定位（通过 AFL++ 生成变体轨迹）与证据加权排名，将修复焦点聚焦于真正相关的函数；并引入双层评估指标（自动或acular + 人工专家评测）以区分真正根因修复。

**🔧 技术方法**

技术包括：AFL++ 动态变体挖掘、CodeQL 静态数据流与调用图分析、证据加权（OWA 与 noisy‑OR）排名、LLM 代理迭代生成与验证（编译、PoC 回放、测试套件、fuzz 变体），以及结构化专家评测流程。

**📊 数据集**

使用了 178 个 C/C++ 项目中的漏洞基准（来自 Yu 等的公开数据集），涵盖 30 个项目、不同规模和多种漏洞类型。

**📈 对比分析**

与三种基线（仅 LLM、符号执行加 LLM、通用编码代理）比较，Kumushi 在三种 oracle 下可行补丁率 85% 以上，显著优于符号执行工具；在人类专家评估中根因修复率 84.9% vs. 77.8%，且在可比 bug 组中 63.6% 的主观优胜率（p=0.0065）。

**⚠️ 局限性**

局限包括：依赖于可达的 crash‑anchored 证据，无法定位跨越字节码或其他不透明边界的缺陷；需要手工专家评测的高成本；以及在高难度或极端触发条件下缺乏足够的变体多样性导致定位覆盖不足。

---

## 38. Explaining and Preventing Alignment Collapse in Iterative RLHF

**arXiv ID:** 2605.04266 | [PDF](https://arxiv.org/pdf/2605.04266v1)

**作者:** Etienne Gauthier `[一作]` (Inria), Michael I. Jordan `[通讯]` (University of California, Berkeley)

**通讯引用:** 179946 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了迭代RLHF的动态反馈循环，证明标准的（无参数调度项）策略优化导致“alignment collapse”，并提出了foresighted policy optimization（FPO）机制，通过加入参数调度惩罚恢复Stackelberg自我纠正动态；

**💡 创新点**

创新点在于把政策对奖励模型参数的影响形式化为参数调度梯度，并给出可实现的TracIn近似，进而将此项加入策略更新，避免奖励模型误差被放大；

**🔧 技术方法**

使用了Stackelberg双层优化框架、影响函数分析、TracIn估计、LoRA微调、Best‑of‑N拒绝采样、Llama‑3.2‑1B与DeBERTa‑v3等大模型；

**📊 数据集**

实验数据集包括UltraFeedback prompts、TruthfulQA评估集，以及使用冻结的Llama‑3.3‑70B作为偏好oracle；

**📈 对比分析**

通过与标准RLHF比较，Relaxed FPO在TruthfulQA上获胜率56.6%（p=0.014），Practical FPO 50.9%（p=0.41），Relaxed优于Practical（p=0.076），并在控制实验中成功收敛至人类理想点；

**⚠️ 局限性**

主要局限在于对奖励模型的强凸假设不适用于大规模神经网络，以及实验规模与模型规模相对有限，未来需在更大模型和非凸设置中进一步验证。

---

## 39. $p$-adic Manifold Learning and Benchmark Tasks from Impartial Games

**arXiv ID:** 2605.04374 | [PDF](https://arxiv.org/pdf/2605.04374v1)

**作者:** Tomoki Mihara `[一作]` `[通讯]`, Tomoki Mihara

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在 p-奇数域上进行流形学习的方法，利用 p-奇数的非阿基米德度量来估计零点集合，并将其应用于判断三堆 Nim 游戏的赢/输位置。

**💡 创新点**

创新点包括：
1) 设计了 p-奇数版 kd‑Trie 作为最近邻搜索数据结构；
2) 结合 Mahler 级数的有限阶截断实现稀疏样本的零点估计；
3) 把 Mahler 展开解释为 p-奇数 Fourier 系列，进而推广到任意原像阿贝尔群；
4) 在 Nim 基准上展示了该方法在稀疏数据下仍能避免过拟合的实用性。

**🔧 技术方法**

核心技术：p-奇数 kd‑Trie、p-奇数 Mahler 级数（有限阶截断）、p-奇数 Fourier 变换（Iwasawa‑Amice 同构）、有限秩近似与样本稠密度分析。

**📊 数据集**

使用了三堆 Nim（D=3）在每堆牌数 <100 的位置集合作为实验数据，样本为所有失利位置共 7984 个，进一步划分为随机与全部位置的子集进行测试。

**📈 对比分析**

与最简单的“始终返回真”方法做对比：后者在混合样本中 99.90% 正确率。该方法在混合样本中达 99.80%，在随机赢局中 12.02%，在全部赢局（≤64×1024²）中 25.73%。
这表明该算法在大多数混合测试中几乎等同于基线，在单纯赢局检测上虽低于 10% 但已明显优于随机猜测，显示出一定的有效性。

**⚠️ 局限性**

局限性：
1) 需要样本在目标集合中具有足够的 E‑稠密性；若样本极为稀疏则估计不可靠；
2) 对 p‑奇数连续性假设过强，非连续情形下无法直接应用；
3) 目前仅在低维（D=3）和 Nim 这种特定游戏上验证，扩展到更高维或其他游戏尚需进一步研究；
4) 计算复杂度随维度和精度 E 上升而显著增加。

---

## 40. EngThrive: Make It Fast and Easy to Do Great Work

**arXiv ID:** 2605.04259 | [PDF](https://arxiv.org/pdf/2605.04259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 41. On the Architectural Complexity of Neural Networks

**arXiv ID:** 2605.04325 | [PDF](https://arxiv.org/pdf/2605.04325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 42. GraphPI: Efficient Protein Inference with Graph Neural Networks

**arXiv ID:** 2605.04376 | [PDF](https://arxiv.org/pdf/2605.04376v1)

**作者:** Zheng Ma `[一作]` (University of Waterloo), Ali Ghodsi `[通讯]` (University of Waterloo)

**通讯引用:** 17235 | [OpenAlex ID](https://openalex.org/A5040035859)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 GraphPI 框架，将蛋白质推断视为三元图中的节点分类问题；

**💡 创新点**

创新点在于使用半监督自训练的 GNN（GraphSAGE 变体）对蛋白–肽–PSM 三元图建模，并利用伪标签与硬负采样实现无监督训练；

**🔧 技术方法**

采用图神经网络、Pseudo-label 生成、self‑training、Percolator PSM 特征及硬负解码等技术；

**📊 数据集**

训练使用多份公开 Human ProteomeXchange 数据集（如 PXD004789、PXD005388 等），测试涵盖 iPRG2016、UPS2、18Mix、Yeast、Hela‑3T3 等；

**📈 对比分析**

与 Epifany、Fido、PIA、DeepPep 等传统方法对比，GraphPI 在大多数数据集上取得更高的 pAUC 与 ROC 曲线性能，同时推断速度至少快 10 倍；

**⚠️ 局限性**

局限在于仍依赖已有算法（如 Epifany）生成伪标签，且对共享肽高比例的数据集性能受限；

---

## 43. Densification and forecasting of Sentinel-2 time series from multimodal SAR and Optical satellite data using deep generative models

**arXiv ID:** 2605.04239 | [PDF](https://arxiv.org/pdf/2605.04239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 44. Misrouter: Exploiting Routing Mechanisms for Input-Only Attacks on Mixture-of-Experts LLMs

**arXiv ID:** 2605.04446 | [PDF](https://arxiv.org/pdf/2605.04446v1)

**作者:** Zekun Fei `[一作]` (Nankai University), XiaoFeng Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4563 | [OpenAlex ID](https://openalex.org/A5075707588)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种仅通过输入干扰来操纵Mixture-of-Experts (MoE) LLM路由机制的攻击框架，从而在远程服务上诱导模型生成不安全响应。

**💡 创新点**

创新点包括：①在不访问或修改模型的前提下，通过构造与路由相关的对比信号识别弱对齐专家；②采用两阶段优化（先路由、后输出）平衡路由控制与输出质量；③将路由信号与高效专家偏好相结合，实现跨模型、跨服务的迁移攻击。

**🔧 技术方法**

使用技术：基于GCG的梯度式输入优化；路由统计与对比（U_i^harm、U_i^comp、U_i^benign）和重采样；加权路由损失和联合损失；对专家选择进行加权调度。

**📊 数据集**

实验数据集：AdvBench、StrongREJECT；以及人工构造的三类路由分析数据集（harm、comp、benign）用于估计专家活跃频率。

**📈 对比分析**

与GCG、FFA、Jailbroken、WildPrompt等基线进行对比；在白盒下ASR从 2% 提升至 24%；在黑盒+FFA下平均 ASR 提升至 39.7%，比最强基线提升 20%+；路由损失显著降低，说明路由控制更有效。

**⚠️ 局限性**

局限性：依赖GCG的梯度搜索导致生成的提示不够自然，影响迁移；只使用单一 surrogate 模型，迁移到更大规模或结构差异显著的服务时效果下降；未给出完全无监督的黑盒实现，且训练与优化成本较高。

---

## 45. A Provably Convergent and Practical Algorithm for Gromov--Wasserstein Optimal Transport

**arXiv ID:** 2605.04175 | [PDF](https://arxiv.org/pdf/2605.04175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 46. Budgeted LoRA: Distillation as Structured Compute Allocation for Efficient Inference

**arXiv ID:** 2605.04341 | [PDF](https://arxiv.org/pdf/2605.04341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 47. Online Nonstochastic Prediction: Logarithmic Regret via Predictive Online Least Squares

**arXiv ID:** 2605.04364 | [PDF](https://arxiv.org/pdf/2605.04364v1)

**作者:** Chih-Fan Pai `[一作]` (University of California San Diego), Yang Zheng `[通讯]` (University of California San Diego)

**通讯引用:** 11641 | [OpenAlex ID](https://openalex.org/A5061103594)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在部分观测的线性动态系统中进行在线预测，目标是最小化累积平方预测损失，并与最佳的Luenberger预测器进行竞争。

**💡 创新点**

提出了一种无约束的在线最小二乘法（FM-POLS），通过定制的预测提示来稳定学习过程，能够在不确定干扰下实现对比最佳Luenberger预测器的对数后悔。

**🔧 技术方法**

使用了无约束的在线最小二乘法（FM-POLS）框架，并设计了基于模型的Luenberger提示和无模型的多项式滤波提示。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在部分观测的线性动态系统（PO-LDS）下的在线预测。

**📈 对比分析**

与经典的固定增益观察者相比，FM-POLS在不确定干扰下提供了自适应的实例最优在线预测器，性能表现出对数后悔，且在实验中显示出优于传统滤波器的效果。

**⚠️ 局限性**

限制在于对于复杂的边际特征值的情况仍然开放，且模型自由的提示构造在某些情况下可能不够有效。

---

## 48. OPENJ: A Conceptual Framework for Open-Source Digital Human Modeling and Ergonomic Assessment in a CAD Environment

**arXiv ID:** 2605.04270 | [PDF](https://arxiv.org/pdf/2605.04270v1)

**作者:** Sinan Bank `[一作]` (Colorado State University), Casey E. Eaton `[通讯]` (Auburn University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个面向工业工作场所的开源数字人模型与人体工学评估框架（Open-Jane/Joe），通过两层架构整合可扩展的仿真核心与FreeCAD可视化前端；

**💡 创新点**

首次在开源生态中统一实现可扩展的人体模型、逆运动学姿势预测、标准人体工学评估插件以及CAD环境集成，且采用插件化与可脚本化的设计；

**🔧 技术方法**

核心使用Python、URDF与YAML描述人体模型，Pinocchio+Pink实现差分IK，SciPy的SLSQP做优化预测；采用Monte Carlo +凸包计算可及性，Ray‑casting实现视线遮挡；通过FreeCAD插件完成可视化；

**📊 数据集**

主要利用美国陆军ANSUR II人类测量数据库及de Leva的体节参数（BSP）进行尺寸与质量的标定；

**📈 对比分析**

对比商业平台（Jack、DELMIA）与公开数据，验证可及面误差≤2 cm、NIOSH LWL误差≤2 %，RULA/REBA/OWAS等评分80 %匹配；实时差分IK维持30+FPS，优化求解时间可接受；

**⚠️ 局限性**

局限包括：受限于ANSUR II的美国军人样本；缺乏手指与完整脊柱细分；未实现碰撞检测；未考虑主观舒适度、环境热负荷及认知负担；不支持实时运动捕捉。

---

## 49. Structural Equivalence and Learning Dynamics in Delayed MARL

**arXiv ID:** 2605.04345 | [PDF](https://arxiv.org/pdf/2605.04345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 50. Governed Collaborative Memory as Artificial Selection in LLM-Based Multi-Agent Systems

**arXiv ID:** 2605.04264 | [PDF](https://arxiv.org/pdf/2605.04264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 51. From Language to Logic: A Theoretical Architecture for VLM-Grounded Safe Navigation

**arXiv ID:** 2605.04327 | [PDF](https://arxiv.org/pdf/2605.04327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 52. Extending Differential Temporal Difference Methods for Episodic Problems

**arXiv ID:** 2605.04368 | [PDF](https://arxiv.org/pdf/2605.04368v1)

**作者:** Kris De Asis `[一作]` (Openmind Research Institute), Jiamin He `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一般化的差分 TD（Differential TD）算法，使奖励中心化能够在包含终止状态的情节（episodic）环境中使用，并通过潜能奖励塑形保证最优策略不变；随后将该算法与线性 TD 等价并在流式深度强化学习框架中实现。

**💡 创新点**

创新点在于：① 将奖励中心化视为基于潜能的奖励塑形，推导出终止差分值定义；② 证明该定义保持最优策略顺序并扩展差分 TD 至情节问题；③ 展示差分 TD 与具有输出偏置的线性 TD 等价，从而继承其理论收敛性。

**🔧 技术方法**

使用的技术包括：差分 TD、奖励中心化、潜能奖励塑形、输出层偏置（bias unit）等价、线性与非线性函数逼近、PopArt 输出归一化、流式深度强化学习算法（Stream Q(λ)、Stream AC(λ)）。

**📊 数据集**

实验数据集包括：10×10网格世界（两种奖励设置）；MinAtar 套件（Asterix, Breakout, Freeway, Seaquest, SpaceInvaders）；MuJoCo 物理仿真环境（Ant-v4, HalfCheetah-v4, Hopper-v4, Humanoid-v4, Walker2d-v4）；DeepMind Control Suite 的 Reacher 环境（易、难两个版本）。

**📈 对比分析**

比较方法：将差分版本与未中心化的 Q‑learning/AC‑learning 以及 PopArt 归一化版本在同一环境下训练，并在 100/30/30 次实验中记录完成率、回报等指标。结果显示：在网格世界痛苦奖励下差分 Q‑learning 明显优于基线；在 MinAtar 中差分 Stream Q(λ) 在多数环境提升，而 PopArt 结果不稳定；在 MuJoCo 中差分 Stream AC(λ) 在 Ant、HalfCheetah 明显提升，其他环境表现相当；Reacher 痛苦版亦显著提升。

**⚠️ 局限性**

局限性：需要额外调节学习率参数 η，增加超参数搜索成本；在稀疏奖励或奖励差异不大的环境中收益有限；部分环境中 PopArt 归一化可获得更好或相当的性能；理论证明主要基于线性 TD 等价，非线性情况下的收敛性和稳定性仍需进一步探讨。

---

## 53. Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping

**arXiv ID:** 2605.04308 | [PDF](https://arxiv.org/pdf/2605.04308v1)

**作者:** Kaustubh Pethkar `[一作]` (New Jersey Institute of Technology), Yingcong Li `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于马尔可夫链的词表扩展框架，将新词映射到已有词上，并通过嵌入微调实现零遗忘的知识增量；

**💡 创新点**

通过令新词在原模型的马尔可夫转移矩阵中仅增加状态，证明样本复杂度随映射稀疏度线性；实现了一种仅更新嵌入的高效算法；

**🔧 技术方法**

马尔可夫过程建模、稀疏词嵌入映射、嵌入微调（Embedding Tuning）、理论样本复杂度分析；

**📊 数据集**

算术算子基准、100个合成词数据集、WikiText、跨语言（西班牙语、德语、阿拉伯语）Wikipedia 语料；

**📈 对比分析**

与全量微调、LoRA、Prompt Tuning 等对比；实验显示嵌入微调在保持原模型性能（零遗忘）的同时，在新任务上达到或超过传统方法，样本效率显著提升；

**⚠️ 局限性**

假设新词出现概率均等、模型足够表达，忽略新词与旧词的交互；在有限容量模型或频率分布不均时，理论与实践可能不完全匹配；

---

## 54. FlowEval: Reference-based Evaluation of Generated User Interfaces

**arXiv ID:** 2605.04165 | [PDF](https://arxiv.org/pdf/2605.04165v1)

**作者:** Jason Wu `[一作]` (Purdue University), Titus Barik `[通讯]` (Apple)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FlowEval 框架，通过比较生成 UI 与参考网站的交互流来自动评估 UI 的功能正确性。

**💡 创新点**

创新点在于将参考基准的相似度指标（DTW、eBLEU、WMD）与视觉化的交互轨迹相结合，实现可解释、可扩展且与人工评审高度相关的评估方法。

**🔧 技术方法**

采用 Vision‑based Computer Use Agent（UI‑TARS）捕获交互轨迹，使用 CLIP‑style 嵌入向量进行特征化，随后计算 DTW、eBLEU、WMD 等相似度；还使用 Elo 排名与 MLLM‑judge 进行对比。

**📊 数据集**

参考数据集为 WebVoyager（15 个网站）和 REAL v2（12 个高保真复制），并手工筛选得到 147 条与 116 条交互任务。

**📈 对比分析**

方法通过人类专家 429 组评分与三种指标和 MLLM‑judge 进行对照；WMD 与人类排序的 Spearman 相关系数 0.96、κ=0.46、约 73% 的一致率，明显优于 MLLM‑judge。

**⚠️ 局限性**

局限性包括仅关注任务支持而非美观/可访问性；依赖参考覆盖，缺乏创造性设计评估；仅在单次无交互迭代的一次性生成环境下测试；人类评审样本小；CUA 的表现可能影响评估稳定性；生成模型的安全与过滤需求。

---

## 55. A Physics-Aware Framework for Short-Term GPU Power Forecasting of AI Data Centers

**arXiv ID:** 2605.04074 | [PDF](https://arxiv.org/pdf/2605.04074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 56. Ground4D: Spatially-Grounded Feedforward 4D Reconstruction for Unstructured Off-Road Scenes

**arXiv ID:** 2605.04435 | [PDF](https://arxiv.org/pdf/2605.04435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 57. Conditional Flow-VAE for Safety-Critical Traffic Scenario Generation

**arXiv ID:** 2605.04366 | [PDF](https://arxiv.org/pdf/2605.04366v1)

**作者:** Zimu Gong `[一作]` (University of Michigan), Raquel Urtasun `[通讯]` (Waabi Innovation Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建一种基于条件潜空间流匹配的VAE框架，用于从常规初始化中生成逼真的安全关键交通场景。

**💡 创新点**

创新点包括：①在VAE潜空间学习将常规情景映射为安全关键情景的流匹配变换；②结合真实与仿真数据实现可扩展的数据驱动生成；③通过手工标签与流时间步长实现场景难度可控。

**🔧 技术方法**

技术方案包括：条件VAE、rectified flow匹配Transformer、Transformer骨干网络（含地图与演员编码）、两阶段训练、手工难度标签和流时间步长控制。

**📊 数据集**

使用自研的近两万条高速/城市驾驶数据，其中约500条真实安全关键片段，约1万条仿真安全关键片段，训练时按比例混合真实与仿真样本。

**📈 对比分析**

与基线VAE、VAE+Curated、Strive等方法对比，在minSTTC、Near-Miss率、JSD、碰撞率、重构误差等指标上表现更优，能生成更多安全关键场景且保持较高真实性。

**⚠️ 局限性**

局限性包括：真实安全关键数据稀缺导致模型对仿真数据的依赖；仿真与真实之间存在域差距；难度控制只能通过标签和流步长，无法精确指定具体动作；未来需要更高保真仿真和更丰富的数据采集。

---

## 58. Jordan-RoPE: Non-Semisimple Relative Positional Encoding via Complex Jordan Blocks

**arXiv ID:** 2605.04217 | [PDF](https://arxiv.org/pdf/2605.04217v1)

**作者:** Yaobo Zhang `[一作]` (Ningxia University), Yaobo Zhang `[通讯]` (Ningxia University)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5036981188)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于复杂Jordan块的相对位置编码（Exact Jordan‑RoPE），在传统RoPE的旋转相位上加入了与距离耦合的多项式项，实现了距离调制相位的原始注意力核

**💡 创新点**

创新点在于将复数特征值与nilpotent响应耦合在同一缺陷Jordan块中，产生新的预softmax特征d·cos(ωd)和d·sin(ωd)，并区分了严格的一参数群律、受限（bounded shear）以及尺度归一化的变体

**🔧 技术方法**

利用线性、平移不变、连续的群作用理论构造Jordan块；实现实化四维块、对齐查询/键的共变作用；对比RoPE、ALiBi、直接求和等基线；在Transformer中使用受限或尺度归一化版本

**📊 数据集**

在基于固定核的混合目标、结构化序列探针、合成距离调制相位语言模型以及WikiText‑103字节级语言模型上进行实验

**📈 对比分析**

在固定核和结构化序列任务中，Jordan‑RoPE在距离调制相位目标上表现最佳；在合成LM中，受限Jordan优于RoPE、直接求和和RoPE+ALiBi；在WikiText‑103小规模实验中，Scaled‑exact c=1在Jordan家族内取得最低平均损失，但RoPE+ALiBi仍表现最好

**⚠️ 局限性**

受限Jordan破坏了一参数律；实现中使用位置‑级因子化，可能引入绝对尺度补偿；实验规模有限，未覆盖大模型、长上下文自然语言任务；并未完全验证在更广泛任务中的优势

---

## 59. Adapt to Thrive! Adaptive Power-Mean Policy Optimization for Improved LLM Reasoning

**arXiv ID:** 2605.04066 | [PDF](https://arxiv.org/pdf/2605.04066v1)

**作者:** Yiming Huang `[一作]` (Harbin Institute of Technology), Chuanyi Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 762 | [OpenAlex ID](https://openalex.org/A5103171964)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种自适应的RLVR算法APMPO，利用可调的幂均值目标和基于奖励反馈的自适应裁剪来提升大型语言模型的推理能力。

**💡 创新点**

创新点在于：①将幂均值作为通用目标函数，根据模型表现动态在算术均值和几何均值之间切换；②设计反馈自适应裁剪(FAC)，根据奖励统计的稳定性动态调节裁剪阈值，改进传统固定裁剪。

**🔧 技术方法**

使用的技术包括：强化学习与可验证奖励(RLVR)、Power-Mean Policy Optimization、Feedback-Adaptive Clipping、基于KL散度的正则化、AdamW优化器以及大规模GPU并行训练。

**📊 数据集**

实验数据集涵盖数学推理（MATH、MATH500、AIME24/25、AMC23、Minerva、OlympiadBench）、SQL生成（BIRD-Train/Dev、Spider-Dev）以及多模态推理（Geometry3K）。

**📈 对比分析**

与GRPO、DAPO、GMPO等RLVR基线进行对比，APMPO在所有数学推理基准上均获得最高Pass@1分，平均提升约2–3点；在SQL和多模态任务中亦取得最高Pass@1/Pass@16成绩，显示出显著的性能优势。

**⚠️ 局限性**

主要限制包括：①实验仅在1.5B和3B参数规模模型上验证，缺乏更大规模模型的评估；②依赖可验证的奖励信号，若奖励难以定义则算法适用性受限。

---

## 60. Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment

**arXiv ID:** 2605.04363 | [PDF](https://arxiv.org/pdf/2605.04363v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在表格基础模型中针对标签偏移的无训练后期自适应方法DistPFN及其温度缩放改进版本DistPFN‑T，能够在推理时调整模型预测以补偿训练与测试标签分布差异。

**💡 创新点**

核心创新在于：①仅使用训练先验和模型自身预测的后验来构造调整因子，无需估计测试先验或重新训练；②引入基于交叉熵的温度缩放来根据训练先验与预测分布的差异动态调节调整强度，从而避免过度修正。

**🔧 技术方法**

技术手段包括：表格ICL模型TabPFN（和其变体LoCalPFN、TabICL、TabPFN‑v2），后验概率归一化、比例重加权、温度缩放（利用交叉熵或其他距离度量计算温度），以及在批量或单实例预测下的平均分布调整。

**📊 数据集**

使用了OpenML公开的253个表格分类数据集，并通过逆频率过采样生成多种标签偏移场景（β从0到5）。

**📈 对比分析**

与基线TabPFN、LoCalPFN、TabICL以及传统标签偏移校正方法EME、BBE进行了比较。实验显示：DistPFN在无偏移场景下保持与原模型相当的准确率，且在标签偏移β≥0.5时平均提升约1–3%，DistPFN‑T在更大偏移下提升更显著；同时在ECE和精度指标上也取得了校准与判别性能的改进。

**⚠️ 局限性**

局限性在于仅针对标签偏移有效，对特征分布漂移（feature shift）无处理；此外，方法假设训练先验已充分代表训练集标签分布，若训练样本极度失衡或分布噪声大，调整效果可能受限。

---

## 61. Confronting Label Indeterminacy in Automated Bail Decisions

**arXiv ID:** 2605.04073 | [PDF](https://arxiv.org/pdf/2605.04073v1)

**作者:** Cor Steging `[一作]` (University of Groningen), Tadeusz Zbiegień `[通讯]` (Jagiellonian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在预审保释决策中标签不确定性问题，并对五种标签补全方法在三种机器学习模型上的影响进行评估

**💡 创新点**

创新点在于系统性比较不同标签补全策略对模型表现和内部决策的影响，并从法律哲学角度阐释其规范含义

**🔧 技术方法**

使用逻辑回归、随机森林与XGBoost三种标准模型，并实现了Correct labels、Detention-as-failure、Observed only、Observed+IP、Nearest Neighbor五种补全方法

**📊 数据集**

采用宾夕法尼亚州统一司法系统 2016-2020 年 90,732 案例的公开匿名数据集

**📈 对比分析**

通过在 25 个平衡训练子集上训练并评估，使用 Matthew's Correlation Coefficient（MCC）比较模型；结果表明标签补全方法对预测分布和特征重要性影响大于模型选择，观测+IP 与 Observed 结果相近，而 Correct、Detention-as-failure 与 Nearest Neighbor 则产生显著差异

**⚠️ 局限性**

局限性包括：对标签不确定性的定义仍不完全系统；仅使用了预审保释相关特征，未考虑公共安全风险；未进行公平性度量与专家标注方法的实验；模型可解释性有限，缺乏人机结合部署方案

---

## 62. Hierarchical Visual Agent: Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning

**arXiv ID:** 2605.04304 | [PDF](https://arxiv.org/pdf/2605.04304v1)

**作者:** Qihua Dong `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31886 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Hierarchical Visual Agent (HierVA) 用管理器-工作者框架处理图表问答，结合缩放、技能即时注入和上下文压缩实现高级图表推理。

**💡 创新点**

通过层级分工、视觉上下文裁剪与蒸馏、以及按需技能路由，避免多步推理中视觉/文本上下文冗长增长，提升多子图、多步推理精度。

**🔧 技术方法**

结合多模态LLM（如Qwen3VL-A22B）作为管理器和工作者，使用可视化工具（zoom-in）、代码执行工具、简洁技能库，并实现计划-执行-蒸馏循环。

**📊 数据集**

在CharXiv reasoning子集、ChartQA以及合成多子图数据集上进行评测。

**📈 对比分析**

与直接、CoT、CoT-Plan、Thinking-with-Images等基线比较，HierVA在CharXiv上实现64.2%准确率，比最强基线高1.8-5.3个百分点，尤其在多步推理类任务上显著提升。

**⚠️ 局限性**

受限于底层视觉模型的细粒度识别、技能库不完整、对全局约束的拆解不佳以及多步执行导致延迟和工程复杂度。

---

## 63. Gradient Flow Structure and Quantitative Dynamics of Multi-Head Self-Attention

**arXiv ID:** 2605.04279 | [PDF](https://arxiv.org/pdf/2605.04279v1)

**作者:** Ayan Pendharkar `[一作]` `[通讯]`, Ayan Pendharkar

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

将多头Transformer自注意力建模为单位球面上的梯度流，并在此框架下研究能量、聚类速度和熵的演化。

**💡 创新点**

提出“径向阴影”概念揭示多头间的几何耦合；给出多头能量单调性、单头能量单调性的充分条件（径向优势）及其临界温度；推导异构头聚类速率的超加性与ReLU与Softmax在不同阶段聚类时间的理论对比；给出精确的注意力熵产生等式。

**🔧 技术方法**

使用梯度流理论、投影恒等式、Lambert W函数、矩阵正交性与近似正交性分析、等角度与正交头简化，结合能量方法和符号推导。

**📊 数据集**

本文为理论研究，未使用实际数据集，主要采用随机/等角度假设与正交头的理想化设置进行证明。

**📈 对比分析**

通过解析推导显示：多头在超加性条件下聚类速率优于等强度头；Softmax在γ≈0时驱动更快（O(n)），ReLU在后期更快（O(n log d)）；总体上多头与单头相比可加速收敛，理论上提升了聚类效率。

**⚠️ 局限性**

局限性包括：仅在标量头、等角度/正交token的理想化设置下证明；径向阴影阻碍单头能量单调性；未证明多头动力学的稳定终点；对近似正交、非正交矩阵的分析仍开放；对ReLU非光滑完整证明尚未完成。

---

## 64. Beyond Fixed Thresholds and Domain-Specific Benchmarks for Explainable Multi-Task Classification in Autonomous Vehicles

**arXiv ID:** 2605.04299 | [PDF](https://arxiv.org/pdf/2605.04299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 65. Physics-Guided Regime Unmixing

**arXiv ID:** 2605.04247 | [PDF](https://arxiv.org/pdf/2605.04247v1)

**作者:** Paula Pacheco `[一作]` (GVT-CONAE), Juan B. Cabral `[通讯]` (GVT-CONAE)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种物理引导的混合解混合方法（PGRU），通过可观察的物理特征估计每个像素的标量ξ_i，以在合理的情况下激活非线性混合。

**💡 创新点**

创新点在于通过可观察的场景特性引导非线性混合的激活，而不是依赖于重建误差，提供了可解释的混合状态图。

**🔧 技术方法**

使用了广义双线性模型（GBM）、后非线性混合模型（PPNM）和Hapke辐射传输模型，通过学习的注意力机制进行组合。

**📊 数据集**

在Samson、Jasper Ridge和Urban三个广泛使用的基准数据集上进行了实验。

**📈 对比分析**

与线性混合模型（LMM）、GBM和PPNM进行比较，PGRU在所有数据集上均表现出更低的重建误差，尤其在Samson和Jasper Ridge上，rRMSE显著降低，且物理一致性ρ均超过0.90。

**⚠️ 局限性**

当前的局限性在于尚未对每个组件的贡献进行单独验证，物理模型集的相对重要性、特征引导的ξ_i激活和空间正则化的作用仍需通过系统的消融研究进行解耦。

---

## 66. phys-MCP: A Control Plane for Heterogeneous Physical Neural Networks

**arXiv ID:** 2605.04256 | [PDF](https://arxiv.org/pdf/2605.04256v1)

**作者:** Stefan Fischer `[一作]` (University of Luebeck), Sebastian Otte `[通讯]` (University of Luebeck)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种子基物理神经网络（PNN）的统一控制平面架构，支持在边缘、雾、云等层面对异质PNN资源进行发现、匹配、调用、监控与生命周期管理。

**💡 创新点**

创新点在于将Model Context Protocol (MCP) 扩展为子基物理计算的能力、时序、生命周期、遥测、数字孪生绑定等多维描述；并通过原型演示跨子基、外部服务及真实湿脑API的可移植性、容错匹配与低开销执行。

**🔧 技术方法**

技术实现：Python原型，基于MCP 的发现与调用；核心组件包括 Capability Registry、Task‑to‑Substrate Matcher、Invocation Manager、Lifecycle Manager、Telemetry Collector、Twin Synchronization Manager；实现三类示例子基（DNA/化学、湿脑、忆阻/光子）以及 HTTP‑外部化执行路径。

**📊 数据集**

数据集：未使用传统机器学习数据集，而是采用三类示例子基的模拟/仿真数据和 Cortical Labs 的真实湿脑 API 进行控制平面功能验证。

**📈 对比分析**

比较方法与性能：①使用完整匹配器与随机、模态仅、时序仅等基线进行任务匹配准确率对比，完整匹配器在 7/7 任务上正确匹配；②在五种故障场景下验证容错（fallback 或拒绝）行为；③测量控制平面开销，三类本地后端平均开销 <1 ms（相对 1.17×‑3.67×），外部 HTTP 路径平均往返 8.96 ms，显示低额外成本。总体性能表现良好。

**⚠️ 局限性**

局限性：原型仅单机实现，缺乏大规模分布式部署评估；子基覆盖仅为三类典型示例，未涵盖 iontronic、微流控、机械/声学等其他子基；数字孪生精度与真实物理行为差异；安全、授权、共享资源管理等细节尚未深入探讨。

---

## 67. Semantic Reverse Engineering Legacy Software Applications with ChatGPT, Gemini AI, and Claude AI

**arXiv ID:** 2605.04114 | [PDF](https://arxiv.org/pdf/2605.04114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 68. GEM: Graph-Enhanced Mixture-of-Experts with ReAct Agents for Dialogue State Tracking

**arXiv ID:** 2605.04449 | [PDF](https://arxiv.org/pdf/2605.04449v1)

**作者:** Ziqi Zhu `[一作]` (Amazon Web Services), Iman Abbasnejad `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了GEM框架，用图神经网络与T5编码解码器通过Mixture‑of‑Experts动态路由，并结合ReAct代理式推理实现对MultiWOZ 2.2的对话状态追踪；

**💡 创新点**

创新点包括：①在DST任务中首次将图结构推理与序列模型通过专家路由相结合；②利用ReAct代理进行结构化值生成；③实现轻量化专家激活，显著降低参数与延迟；

**🔧 技术方法**

使用技术：Graph Attention Network（GAT/GATv2）处理对话图，BERT‑Base作节点嵌入；T5‑Small（70M）做序列生成；Mixture‑of‑Experts路由器与BERT‑based域分类器；ReAct代理式推理与LLM（Llama‑3.1‑8B、Claude‑3.7‑Sonnet）进行少样本值生成；检索增强与密集向量检索；

**📊 数据集**

数据集：MultiWOZ 2.2（10,438对话，7域，13意图/25槽）；

**📈 对比分析**

与现有SOTA（TOATOD、D3ST、Diable等）对比，GEM取得65.19% JGA、97.65% JTA，显著优于传统模型且参数量仅为271M（GNN）+70M（T5）+LLM；相对单一LLM端到端方案（Claude 0-shot 32.5% JGA），性能提升约32%点；在延迟方面，GEM约12 ms/轮；

**⚠️ 局限性**

限制：仍依赖大型LLM进行值生成，导致推理延迟高；图结构的构建与更新需要手工设计，难以自动化；在跨数据集或更复杂域场景下的泛化能力尚未充分验证；ReAct代理在逻辑推理深度与可解释性方面仍有提升空间。

---

## 69. Joint Optimization of Trajectory Control, Resource Allocation, and Task Offloading for Multi-UAV-Assisted IoV

**arXiv ID:** 2605.04436 | [PDF](https://arxiv.org/pdf/2605.04436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 70. Probabilistic Classification and Uncertainty Quantification of Sahara Desert Climate Using Feedforward Neural Networks

**arXiv ID:** 2605.04286 | [PDF](https://arxiv.org/pdf/2605.04286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 71. Intermediate Representations are Strong AI-Generated Image Detectors

**arXiv ID:** 2605.04358 | [PDF](https://arxiv.org/pdf/2605.04358v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 72. Pro$^2$Assist: Continuous Step-Aware Proactive Assistance with Multimodal Egocentric Perception for Long-Horizon Procedural Tasks

**arXiv ID:** 2605.04227 | [PDF](https://arxiv.org/pdf/2605.04227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 73. Structured 3D Latents Are Surprisingly Powerful: Unleashing Generalizable Style with 2D Diffusion

**arXiv ID:** 2605.04412 | [PDF](https://arxiv.org/pdf/2605.04412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 74. ClusterLess: Deadline-Aware Serverless Workflow Orchestration on Federated Edge Clusters

**arXiv ID:** 2605.04310 | [PDF](https://arxiv.org/pdf/2605.04310v1)

**作者:** Reza Farahani `[一作]` (TU Wien), Radu Prodan `[通讯]` (University of Innsbruck)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种面向联邦多边缘K8s集群的基于deadline的无服务器工作流编排框架CLU；

**💡 创新点**

创新点在于同时考虑工作流依赖、deadline约束、资源异构与跨集群协同，并通过分层（本地主控+超级主控）实现动态调度与调度模式选择；

**🔧 技术方法**

技术主要包括OpenFaaS作为无服务器执行平台，Argo实现工作流管理，Kubernetes实现容器编排，基于心跳和负载阈值的超级主控选举，四种执行模式（warm、warm scaling、cold scaling、offloading）和基于EDF的跨集群调度；

**📊 数据集**

使用真实六集群边缘测试床，包含64台异构节点（Jetson、Raspberry Pi、x86虚拟机等），并重现了两条真实工作流（文本转语音加速检测与回归模型调优）作为基准；

**📈 对比分析**

与四种基线（原生K8s、单集群CLI、RRX、RNX）对比，CLU在多种负载情境下完成时间平均降低约40%，deadline满足率从<50%提升至>90%，违约时间缩短至个位秒级；

**⚠️ 局限性**

局限性在于仅评估了两类工作流，未考虑状态迁移与多目标优化；算法复杂度与超主控延迟在高峰期仍可能受限，且对网络链路质量变动的鲁棒性待进一步验证。

---

## 75. Adaptive Diagonal Loading for Norm Constrained Beamforming

**arXiv ID:** 2605.04342 | [PDF](https://arxiv.org/pdf/2605.04342v1)

**作者:** Manan Mittal `[一作]` (Stony Brook University), Andrew C. Singer `[通讯]` (Stony Brook University)

**通讯引用:** 6681 | [OpenAlex ID](https://openalex.org/A5028083945)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于Kantorovich不等式的白噪声增益(WNG)约束自适应对角加载方法，用于快速变化的噪声环境下的麦克风阵列 beamforming

**💡 创新点**

创新点在于通过解析式将WNG下限映射到协方差矩阵条件数上，并给出三种可扩展的加载估计模式（Trace、Gershgorin、Exact EVD），实现了确定性且最小的对角加载，避免了传统方法的经验参数选择和过度抑制

**🔧 技术方法**

主要技术包括：Kantorovich不等式推导、条件数约束对角加载、基于矩阵分块的GSC架构、以及不同复杂度的特征值估计技术

**📊 数据集**

使用模拟的均匀线性阵列（M=15）在“出生-死亡”动态干扰场景下的合成声学数据，包含随机出现的干扰源和固定的目标源，采用滑动窗口估计SCM

**📈 对比分析**

与传统的Cox后置权重缩放方法以及全知Capon beamformer比较，实验显示Trace模式可保证WNG不低于设定阈值，Gershgorin模式性能几乎与Exact EVD相当，同时计算量大幅降低；Exact EVD模式在输出SINR和MSE上最优；Cox方法则表现最差

**⚠️ 局限性**

局限性包括：Exact EVD模式在大阵列下计算量过大，Trace模式在极端snapshot不足时可能过度加载导致性能下降；Gershgorin估计受基底变换影响，在GSC架构下可能出现不一致性

---

## 76. Not All That Is Fluent Is Factual: Investigating Hallucinations of Large Language Models in Academic Writing

**arXiv ID:** 2605.04171 | [PDF](https://arxiv.org/pdf/2605.04171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 77. Constrained Extreme Gradient Boosting for Adapting Reduced-Order Models

**arXiv ID:** 2605.04130 | [PDF](https://arxiv.org/pdf/2605.04130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 78. RemoteZero: Geospatial Reasoning with Zero Human Annotations

**arXiv ID:** 2605.04451 | [PDF](https://arxiv.org/pdf/2605.04451v1)

**作者:** Liang Yao `[一作]`, Yuhui Zheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种无盒子监督的地理空间推理框架 RemoteZero，通过生成-裁剪-验证循环让模型在未标注遥感图像上自我演化并定位目标。

**💡 创新点**

创新点在于利用多模态大模型的“眼（验证）>手（生成）”优势，将几何监督替换为内部语义一致性奖励，并在每一轮迭代中将前一轮模型作为验证器实现自监督演进。

**🔧 技术方法**

核心技术包括多模态大语言模型（Qwen3‑VL‑8B）、Group Relative Policy Optimization (GRPO) 强化学习、LoRA 微调、语义验证器、面积惩罚与上下文裁剪策略。

**📊 数据集**

使用 EarthReason 远程感知推理数据集进行训练与评估，且不使用任何地面真值框坐标。

**📈 对比分析**

与传统基于监督的 RemoteReasoner、通用 MLLM 以及专业遥感模型比较，RemoteZero 在 Acc@0.5 上达到 71.29%（超远程感知基线 3.18pp），gIoU 仍略低于监督模型，证明自验证奖励能在无监督场景下实现竞争性能。

**⚠️ 局限性**

局限性包括：1）语义验证奖励难以精准约束边界，导致定位精度不足；2）自演进过程可能累积前轮误差；3）裁剪验证难以捕获全局空间关系，需要更强的全局‑局部验证机制。

---

## 79. YOTOnet: Zero-Shot Cross-Domain Fault Diagnosis via Domain-Conditioned Mixture of Experts

**arXiv ID:** 2605.04528 | [PDF](https://arxiv.org/pdf/2605.04528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 80. Continual Distillation of Teachers from Different Domains

**arXiv ID:** 2605.04059 | [PDF](https://arxiv.org/pdf/2605.04059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 81. Undetectable Backdoors in Model Parameters: Hiding Sparse Secrets in High Dimensions

**arXiv ID:** 2605.04209 | [PDF](https://arxiv.org/pdf/2605.04209v1)

**作者:** Sarthak Choudhary `[一作]` (University of Wisconsin Madison), Somesh Jha `[通讯]` (University of Wisconsin Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种在预训练多层图像分类器中植入不可检测后门的攻击方法——Sparse Backdoor。

**💡 创新点**

创新点在于：①将后门注入过程归约为在每个全连接层插入稀疏方向的结构化扰动；②通过添加独立高斯抖动将后门模型与一个“干净参考”模型相区分；③证明该后门模型与参考模型在白盒条件下不可被任何多项式时间区分，等价于稀疏PCA检测难题，从而提供正式的不可检测性保证。

**🔧 技术方法**

采用的技术包括：梯度优化的输入触发器、稀疏方向的随机采样与传播、结构化稀疏权重扰动、独立高斯抖动、稀疏PCA难题的安全归约与证明。

**📊 数据集**

在 CIFAR‑10、SVHN 和 GTSRB 三个公开图像分类数据集上评估，模型架构包括 ConvNet、ResNet‑18 与 Vision Transformer（ViT‑Small）。

**📈 对比分析**

与三种主流检测方法（Neural Cleanse、FeatureRE、UNICORN）对比，平均区分优势仅为 0.12，接近随机猜测；攻击成功率（ASR）在 CIFAR‑10 上均超过 93%，在其他数据集也保持 70%+；正常精度损失仅 1.5–8.5 点。对比实验显示 Fine‑tune 等传统缓解措施对该后门效果不稳定，易失效。

**⚠️ 局限性**

局限性包括：①只针对包含全连接层预测头的网络；②需要满足“边际正则化”和“抖动校准”等经验假设；③目前仅在图像分类任务验证，未扩展至语言、语音或完全无 FC 的模型；④实验验证基于特定架构与数据集，理论证明对其他网络结构的适用性尚未完整验证。

---

## 82. Optimally Covering Large Triangles with Homothetic Unit Triangles

**arXiv ID:** 2605.04111 | [PDF](https://arxiv.org/pdf/2605.04111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 83. Probabilistic Floating-Point Round-Off Analysis via Concentration Inequalities

**arXiv ID:** 2605.04232 | [PDF](https://arxiv.org/pdf/2605.04232v1)

**作者:** Yichen Tao `[一作]` (University of Michigan), Jean-Baptiste Jeannin `[通讯]` (University of Michigan)

**通讯引用:** 1014 | [OpenAlex ID](https://openalex.org/A5002069299)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对浮点运算的舍入误差进行概率性分析，给出在指定置信水平下的误差阈值；

**💡 创新点**

将Taylor展开与正负分解结合并利用集中不等式，提出无绝对值的误差上界，并通过范围划分进一步优化；

**🔧 技术方法**

使用FPTaylor的Taylor展开、正负分解算法、Markov/中心矩不等式、区间划分和符号期望计算，构建ProbTaylor工具；

**📊 数据集**

基准涵盖PAF、FPBench等多项式与分式表达式，采用统一分布、截断正态和双指数分布，并扩展输入范围做对比；

**📈 对比分析**

与PAF、PrAn及FPTaylor比较，ProbTaylor在运行时间上提升数十到数百倍，得到的阈值与现有工具相当甚至更紧；

**⚠️ 局限性**

仅支持加减乘除，无法处理三角、指数等超多项式运算；二阶误差估计仍是性能瓶颈，对分布细分的效果依赖经验。

---

## 84. LCM: Lossless Context Management

**arXiv ID:** 2605.04050 | [PDF](https://arxiv.org/pdf/2605.04050v1)

**作者:** Clint Ehrlich `[一作]` (Voltropy PBC), Theodore Blackman `[通讯]` (Voltropy PBC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种确定性的上下文管理架构 LCM，并将其集成到开源编码代理 Volt 中，以提升 LLM 在长上下文任务中的记忆与推理性能。

**💡 创新点**

创新点在于将递归上下文管理从模型自治转为引擎控制，采用无损摘要 DAG、三阶压缩升级、文件外部引用以及确定性迭代工具（LLM‑Map/Agentic‑Map）等结构化操作。

**🔧 技术方法**

技术包括持久化存储（PostgreSQL）、有向无环图摘要、异步压缩阈值、文件引用与探索摘要、确定性递归升级以及并行批处理工具等。

**📊 数据集**

使用 Oolong 长上下文基准（拆分版）作为评测数据集，并以 Opus 4.6 为基准模型。

**📈 对比分析**

在 Oolong 长上下文评测中，将配备 LCM 的 Volt 与 Claude Code v2.1.4 以及原始 Opus 4.6 进行对比；结果显示 Volt 在 32K 以上上下文长度中均优于 Claude Code，平均绝对分数提升约 4.5 分，1M 上下文仍保持优势。

**⚠️ 局限性**

局限性包括潜在的训练数据污染、对极端上下文长度的评测受限，以及需要进一步验证在更大规模或不同任务上的泛化能力。

---

## 85. Experiment-as-Code Labs: A Declarative Stack for AI-Driven Scientific Discovery

**arXiv ID:** 2605.04375 | [PDF](https://arxiv.org/pdf/2605.04375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 86. MuCALD-SplitFed: Causal-Latent Diffusion for Privacy-Preserving Multi-Task Split-Federated Medical Image Segmentation

**arXiv ID:** 2605.04108 | [PDF](https://arxiv.org/pdf/2605.04108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 87. Assessing Generalisation Capability of Machine Learning Models for Intrusion Detection

**arXiv ID:** 2605.04407 | [PDF](https://arxiv.org/pdf/2605.04407v1)

**作者:** Md Zakir Hossain `[一作]` (Australian National University), Tom Gedeon `[通讯]` (Curtin University)

**通讯引用:** 7806 | [OpenAlex ID](https://openalex.org/A5030379402)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估机器学习模型在入侵检测中的泛化能力，采用同一数据集与交叉数据集评估 Random Forest、Logistic Regression 和 Naive Bayes 在 UNSW‑NB15 与 TON_IoT 上的表现。

**💡 创新点**

首次揭示入侵检测模型在不同数据集间存在显著泛化缺口，并将其与情感计算与人本 AI 的跨域适应问题相类比，指出需要构建可自适应、可泛化的安全模型。

**🔧 技术方法**

使用监督学习分类器（Random Forest、Logistic Regression、Naive Bayes）、特征预处理（标签编码、Min–Max 归一化）、特征相关性与重要性分析，以及同一数据集与交叉数据集评估框架。

**📊 数据集**

公开网络流量标签数据集：UNSW‑NB15 与 TON_IoT。

**📈 对比分析**

通过训练-测试拆分（70/30）在同一数据集评估，Random Forest 在 UNSW‑NB15 取得 95.08% 准确率，TON_IoT 取得 99.79%；Logistic Regression 与 Naive Bayes 分别次之。交叉数据集评估时，Random Forest 仅下降到约 38‑40% 的准确率，Logistic Regression 在 UNSW→TON 方向保持约 77% 的准确率，但整体性能仍显著低于同一数据集，说明泛化差距突出。

**⚠️ 局限性**

仅使用两数据集且交叉时仅保留 7 个公共特征导致信息损失；缺乏域适应、特征对齐等技术；实验规模有限，未涵盖更广泛的网络环境与攻击类型。

---

## 88. Decision Evidence Maturity Model for Agentic AI: A Property-Level Method Specification

**arXiv ID:** 2605.04093 | [PDF](https://arxiv.org/pdf/2605.04093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 89. MP-ISMoE: Mixed-Precision Interactive Side Mixture-of-Experts for Efficient Transfer Learning

**arXiv ID:** 2605.04058 | [PDF](https://arxiv.org/pdf/2605.04058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 90. Improving Medical VQA through Trajectory-Aware Process Supervision

**arXiv ID:** 2605.04064 | [PDF](https://arxiv.org/pdf/2605.04064v1)

**作者:** Halil Ibrahim Gulluk `[一作]` (Stanford University), Olivier Gevaert `[通讯]` (Stanford University)

**通讯引用:** 16100 | [OpenAlex ID](https://openalex.org/A5078274543)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在六个医学视觉问答基准上，先利用COMCTS算法自动生成推理轨迹，随后基于这些轨迹进行两阶段训练：先做监督微调，再用轨迹感知奖励（DTW）进行GRPO强化学习，以提升模型的推理质量和答案准确性。

**💡 创新点**

创新点在于：①首次公开六大医学VQA数据集的推理轨迹；②提出基于句子嵌入序列DTW距离的过程奖励，弥补传统仅用最终答案匹配奖励的缺陷；③在GRPO框架中将过程奖励与精确匹配奖励相结合，形成更细粒度、连续的训练信号。

**🔧 技术方法**

使用的技术包括：COMCTS多模型协同生成推理链；LLM（DeepSeek-R1）做推理链评估；Sentence-Transformer提取推理步骤嵌入；DTW计算过程相似度；GRPO（Group Relative Policy Optimization）强化学习；LoRA微调Qwen2.5-VL-3B视觉语言模型。

**📊 数据集**

数据集包括：VQA-RAD、SLAKE-VQA、PathVQA、PMC-VQA、OmniMed-VQA、VQA-MED；并在此基础上构造了包含推理链的扩展数据集。

**📈 对比分析**

与只用监督微调（SFT）和仅使用精确匹配奖励的基线相比，加入DTW过程奖励后平均准确率从0.598提升至0.689，平均BERTScore从0.845提升至0.881，平均ROUGE‑L从0.665提升至0.748；实验表明DTW奖励在所有评估指标上均有显著提升。对比Needleman‑Wunsch（NW）对齐奖励，NW虽可提升性能但未能在DTW基础上进一步提升。

**⚠️ 局限性**

局限性包括：①过程奖励仍依赖于高质量的参考推理链，若参考推理链不足或有偏差，奖励效果受限；②仅衡量推理轨迹相似度，未逐步验证每一步的正确性；③方法在医学域表现良好，但跨域推广尚待验证。

---

## 91. Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion

**arXiv ID:** 2605.04291 | [PDF](https://arxiv.org/pdf/2605.04291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. Worst-Case Discovery and Runtime Protection for RL-Based Network Controllers

**arXiv ID:** 2605.04373 | [PDF](https://arxiv.org/pdf/2605.04373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 93. InterFuserDVS: Event-Enhanced Sensor Fusion for Safe RL-Based Decision Making

**arXiv ID:** 2605.04355 | [PDF](https://arxiv.org/pdf/2605.04355v1)

**作者:** Mustafa Sakhaia `[一作]`, Maciej Wielgosza `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将动态视觉传感器（DVS）集成到多模态感知框架InterFuser中的端到端自动驾驶系统；

**💡 创新点**

创新点在于：①为DVS设计了基于Transformer的token融合策略，②使用RGB预训练权重进行DVS权重初始化以加速收敛，③引入基于占用图的模型驱动碰撞检测与红灯违规控制，实现了硬件感知与规则约束的混合安全策略；

**🔧 技术方法**

主要技术包括：多模态CNN特征提取（RGB/DVS使用ResNet，LiDAR使用PointNet++/BEV CNN），Transformer Encoder/Decoder进行全局融合，PID控制器结合安全验证模块，损失函数融合轨迹、交通灯与占用图监督，Spiking Neural Network与Asynchronous Fusion Transformer的未来方向；

**📊 数据集**

使用CARLA模拟器中的Town05 Long Routes基准数据集，包含RGB、LiDAR和合成的DVS事件序列；

**📈 对比分析**

在CARLA Leaderboard上实现了77.2的Driving Score、100%路程完成率，碰撞率低，红灯违规率极低，优于多数同类方法（如M2DA、ReasonNet）在路程完成上的表现；

**⚠️ 局限性**

局限性包括：目前仍采用将事件转换为同步帧的做法，未能充分利用DVS微秒级时序信息；系统对极端光照下的鲁棒性虽提升但仍有改进空间；以及在真实世界部署时对硬件时延和能耗的评估不足。

---

## 94. Rigid homotopies for sampling from algebraic varieties: a Waring structure complexity model

**arXiv ID:** 2605.04302 | [PDF](https://arxiv.org/pdf/2605.04302v1)

**作者:** Abigail R. Jones `[一作]`, Jose Israel Rodriguez `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了刚性同调（rigid homotopy）在具有Waring表示的多项式系统上的平均复杂度，并首次给出了相应的实现与实验结果；

**💡 创新点**

创新点在于将平均情况多项式时间的理论扩展到Waring表示的结构化输入，证明了其条件数的上界仅随表示长度r缓慢增长，且在r趋近无穷时退化为纯粹的n、D多项式依赖；

**🔧 技术方法**

采用刚性同调框架、γ-数（split γ-number）分析、概率估计的Frobenius范数、黑盒评估模型以及随机高斯采样等技术；

**📊 数据集**

使用合成的随机Waring系统（不同维度n、次数D、表示长度r），在实验中平均生成100个实例进行测试；

**📈 对比分析**

与基于启发式恒定步长的追踪方法比较，发现其成功率在大多数情况下达到100%，但平均步长远小于启发式步长，实验显示刚性同调在理论上可实现多项式复杂度，实际运行中受步长保守性影响；

**⚠️ 局限性**

局限性包括步长选择过于保守导致迭代次数巨大、理论上大常数导致估计值偏大、目前仅针对单解追踪而非全解计算，并未充分利用多项式神经网络等更广泛的评估模型。

---

## 95. LLMs Uncertainty Quantification via Adaptive Conformal Semantic Entropy

**arXiv ID:** 2605.04295 | [PDF](https://arxiv.org/pdf/2605.04295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 96. Interpreting V1 Population Activity via Image-Neural Latent Representation Alignment

**arXiv ID:** 2605.04309 | [PDF](https://arxiv.org/pdf/2605.04309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 97. EdgeRazor: A Lightweight Framework for Large Language Models via Mixed-Precision Quantization-Aware Distillation

**arXiv ID:** 2605.04062 | [PDF](https://arxiv.org/pdf/2605.04062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 98. Contextual Memory-Enhanced Source Coding for Low-SNR Communications

**arXiv ID:** 2605.04400 | [PDF](https://arxiv.org/pdf/2605.04400v1)

**作者:** Ziqiong Wang `[一作]` (Zhejiang University), Rongpeng Li `[通讯]` (Zhejiang University)

**通讯引用:** 5630 | [OpenAlex ID](https://openalex.org/A5014842979)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种内置上下文记忆的记忆增强源编码（MASC）框架，用于在低信噪比环境下提升分离源码-信道码（SSCC）的鲁棒性。

**💡 创新点**

创新点在于将多阶 n‑gram 记忆与 Transformer 的自回归建模结合，并引入 Mixture‑of‑Memory‑Experts Router（MMER）实现稀疏、状态自适应的记忆专家路由，从而在编码/解码两端共享记忆以提升概率估计质量并减少错误传播。

**🔧 技术方法**

技术包括 Transformer 主干、Parameterized Contextual Memory（PCM）哈希记忆、MMER 路由、LLM 驱动的算术编码/解码、ECCT 信道解码、BERT/ BLEU 评价指标。

**📊 数据集**

使用 Europarl 英文文本语料库进行训练、验证和测试。

**📈 对比分析**

通过与 Huffman、LLM‑AC+ECCT、ICD 等 SSCC 基线以及 DeepSC、UT、UT+量化等 JSCC 基线进行对比，评估 BLEU‑1/4 和语义相似度指标；实验结果显示，MASC 在 Rayleigh 和 AWGN 信道中均优于所有基线，尤其在低 SNR 区域提升显著。

**⚠️ 局限性**

局限性包括：目前仅针对文本数据验证；记忆设计仍相对简单，缺乏更丰富的多模态记忆与自适应路由机制；对极低信噪比下的误码率仍有进一步提升空间。

---

## 99. Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs

**arXiv ID:** 2605.04065 | [PDF](https://arxiv.org/pdf/2605.04065v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 100. AI and Suicide Prevention: A Cross-Sector Primer

**arXiv ID:** 2605.04321 | [PDF](https://arxiv.org/pdf/2605.04321v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 101. Self-Prompting Small Language Models for Privacy-Sensitive Clinical Information Extraction

**arXiv ID:** 2605.04221 | [PDF](https://arxiv.org/pdf/2605.04221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 102. Reward-Guided Semantic Evolution for Test-time Adaptive Object Detection

**arXiv ID:** 2605.04531 | [PDF](https://arxiv.org/pdf/2605.04531v1)

**作者:** Lihua Zhou `[一作]` (Hong Kong Institute of Science and Innovation Chinese Academy of Sciences), Zhen Lei `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 27270 | [OpenAlex ID](https://openalex.org/A5109299788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练、无反向传播的测试时自适应方法 RGSE，用于修正视觉语言模型在分布偏移下的文本嵌入与视觉特征的语义不对齐。

**💡 创新点**

创新点在于把文本嵌入优化视为演化搜索：生成噪声扰动候选、用当前与历史视觉提议的余弦相似度作为奖励、并用奖励加权平均融合得到新的文本嵌入，从而高效、直接纠正语义偏差。

**🔧 技术方法**

采用演化搜索思想、奖励引导的权重融合、历史视觉嵌入缓存、全局文本嵌入银行，以及基于 Grounding DINO 的多模态检测框架。

**📊 数据集**

在 FoggyCityscapes、PASCAL‑C 及 COCO‑C 这三套受噪声/天气影响的测试时自适应基准上进行评估。

**📈 对比分析**

与传统 Faster R‑CNN 自适应方法以及最新的 VLM 自适应技术（TDA、BCA、BCA+、MPMT、HisTPT 等）对比，RGSE 在所有基准下均取得最高 mAP_50（如 FoggyCityscapes Swin‑B 上 39.63、PASCAL‑C Swin‑B 上 73.88、COCO‑C Swin‑B 上 43.25），同时保持低额外计算量（≈ 180 ms/图，≈ 3.8 GB 显存）。

**⚠️ 局限性**

局限在于对每个类别维护历史视觉缓存会随类别数增大导致显存占用线性增长；且当前仅在文本嵌入上做改动，未考虑视觉编码器的潜在偏移。

---

## 103. Constraint-Aware Execution Planning for Hybrid Space-Ground Compute Workloads

**arXiv ID:** 2605.04052 | [PDF](https://arxiv.org/pdf/2605.04052v1)

**作者:** Subhadip Mitra `[一作]` `[通讯]` (RotaStellar), Subhadip Mitra (RotaStellar)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了约束感知执行（CAE）系统，自动化卫星空间-地面计算工作负载的放置、传输和调度。

**💡 创新点**

创新点在于结合轨道物理、资源约束和自适应 FEC 的四阶段规划，并保证规划的确定性与可重现性。

**🔧 技术方法**

使用 SGP4 轨道传播、太阳日蚀检测、地面站通道预算、DAG 调度、成本模型和自适应 FEC 等技术。

**📊 数据集**

使用公开的 TLE 数据、CelesTrak 卫星目录、12 站地面站信息以及五种工作负载预设数据集。

**📈 对比分析**

与传统的单独观测或通信调度方法相比，CAE 在 12 小时规划窗口内完成计划不超过 2 秒，能够满足所有工作负载并显著降低传输量，表现优于现有方法。

**⚠️ 局限性**

局限性包括贪心调度未保证全局最优、只规划单颗卫星、缺乏动态重规划、热模型简化以及 SGP4 预测误差随时间累积。

---

## 104. The Scaling Properties of Implicit Deductive Reasoning in Transformers

**arXiv ID:** 2605.04330 | [PDF](https://arxiv.org/pdf/2605.04330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 105. Critical Windows of Complexity Control: When Transformers Decide to Reason or Memorize

**arXiv ID:** 2605.04396 | [PDF](https://arxiv.org/pdf/2605.04396v1)

**作者:** Sarwan Ali `[一作]` (Columbia University), Sarwan Ali `[通讯]` (Columbia University)

**通讯引用:** 583 | [OpenAlex ID](https://openalex.org/A5064858842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

研究Transformer在组合任务中的复杂性控制如何随训练时间变化，揭示权重衰减在特定窗口内决定模型走向推理或记忆化解。

**💡 创新点**

提出关键窗口（critical‑window）概念，显示训练期间权重衰减的时序决定性，并给出两时尺度理论解释该现象，挑战传统的静态超参数设定。

**🔧 技术方法**

使用大规模CPU实验跑数千条训练轨迹，设计时间局部权重衰减调度，自定义condensation指数与桥梁对齐诊断，并在线性化Transformer模型中推导两时尺度动力学。

**📊 数据集**

以anchor‑function组合任务为主实验数据集（K=16，M=8），并对比grokking（模数算术）、SCAN等任务验证现象的任务特异性。

**📈 对比分析**

通过对比全时常量衰减、窗口式衰减、不同初始化与预算位置的实验，发现中期窗口可在仅使用25%正则化预算的情况下达到与全时相同的OOD准确率，早期窗口几乎无效；在4层、SGD等变体中现象仍保持一致。

**⚠️ 局限性**

局限于可达推理与记忆解的任务，未在SCAN、grokking等任务体现；理论基于线性化模型，缺乏对大模型和更广泛任务的泛化；小初始化时 basin 收缩导致实践建议需重新评估。

---

## 106. Science discussions of retracted articles on Bluesky: public scrutiny or misinformation spreading?

**arXiv ID:** 2605.04334 | [PDF](https://arxiv.org/pdf/2605.04334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 107. Counterfactual identifiability beyond global monotonicity: non-monotone triangular structural causal models

**arXiv ID:** 2605.04413 | [PDF](https://arxiv.org/pdf/2605.04413v1)

**作者:** Pengcheng Tan `[一作]` (East China Normal University), Dehui Du `[通讯]` (East China Normal University)

**通讯引用:** 552 | [OpenAlex ID](https://openalex.org/A5014788258)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一类非单调三角形结构因果模型（Non-monotone triangular SCM）以及对应的可学习逆模型CausalInverter，用以实现无全局单调性限制下的因果可辨识与对数推断。

**💡 创新点**

核心创新在于将全局单调性替换为两条更弱但足够的假设：机制层可逆性（mechanism-wise invertibility）和逆传输不随父变量上下文变化（context-independent inverse transport）。该组合等价于外生同构（exogenous isomorphism），从而保证在共享顺序的三角模型中完整的反事实可辨识。

**🔧 技术方法**

技术方法包括：1) 结构化参数化——每个机制使用可逆单调流与上下文相关的符号门；2) 训练目标：观测负对数似然加上循环一致性正则与逆传输稳定性正则；3) 逆向推断流程——先将观测样本映射回外生空间，再执行干预并递归递归回推；4) 采用可学习的三角层与Transport‑Stability正则化来实现上述理论条件。

**📊 数据集**

使用的数据集包括：1) 合成机制（全单调、阈值翻转、平滑翻转）共108种配置；2) 物理交互任务MuJoCo的Push与Door两套低维动态轨迹。

**📈 对比分析**

与多种基线（线性动力学、MLP、AR、GRU、Transformer、条件流）进行对比。结果显示：在Door任务（强非单调性）中，CausalInverter实现了事件级完美恢复且在连续角度误差上优于其他完美事件模型；在Push任务（弱非单调性）中，CausalInverter在连续误差上与强基线相当，但事件准确率不及线性/MLP模型，整体表现略逊。整体来看，在高非单调性环境下其结构化逆提供了最稳健的事件级恢复。

**⚠️ 局限性**

局限性：对弱非单调性任务（如Push）可能导致过度正则化，表现不如灵活的低数据状态预测器；在所有连续指标上并非最优；模型复杂度相对较高，需要额外的门控与正则化。

---

## 108. Capabilities of Auto-encoders and Principal Component Analysis of the Reduction of Microstructural Images; Application on the Acceleration of Phase-Field Simulations

**arXiv ID:** 2605.04229 | [PDF](https://arxiv.org/pdf/2605.04229v1)

**作者:** Seifallah Fetni `[一作]` (University of Liège), Anne Marie Habraken `[通讯]` (University of Liège)

**通讯引用:** 4949 | [OpenAlex ID](https://openalex.org/A5024593098)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建基于相场仿真数据的框架，利用自编码器和PCA实现微观结构图像的低维压缩，并使用LSTM/GRU进行时序预测，从而加速相场模拟并实现对微观演化的预测。

**💡 创新点**

创新点在于：①将自编码器与PCA两阶段组合用于非线性图像压缩，显著提升压缩率（1/196）同时保持80%以上精度；②在压缩后的潜在空间上训练LSTM实现短期预测，证明可在无需高性能计算的情况下快速预测未来帧；③提供了完整的框架与公开数据与脚本，便于复现与推广。

**🔧 技术方法**

技术包括：相场模拟（Cahn-Hilliard方程），自编码器（单层/双层）、PCA、LSTM、GRU、MinMaxScaler等预处理，使用Keras/TensorFlow训练模型。

**📊 数据集**

使用约10000个二元合金的相场模拟样本（浓度、迁移率、梯度能系数各变动范围不同），随机抽取7000个样本用于训练，数据以RGB图像形式存储。

**📈 对比分析**

与单层自编码器或PCA单独使用相比，双阶段自编码器+PCA在相同压缩率下MSE降低；LSTM在预测5帧时验证损失≈0.0082，GRU性能略差。总体显示自编码器+PCA+LSTM在预测精度与计算资源上均优于传统方法。

**⚠️ 局限性**

局限性包括：①第一阶段自编码器压缩需大规模HPC资源；②预测时间窗口有限（最多10帧），难以长期预测；③框架仍依赖相场仿真生成数据；④线性PCA在高度非线性图像上表现受限；⑤未探讨更高效的时序模型（如Transformer）。

---

## 109. NoisyCausal: A Benchmark for Evaluating Causal Reasoning Under Structured Noise

**arXiv ID:** 2605.04313 | [PDF](https://arxiv.org/pdf/2605.04313v1)

**作者:** Zhi Xu `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31886 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 NoisyCausal 这一新型评测基准，用于检验大型语言模型在带结构化噪声（如干扰变量、数值扰动、潜在混杂等）下的因果推理能力，并设计了一套基于图的模块化推理框架：先让 LLM 提取变量并构造因果图，再将图结构与观察值整理成结构化提示，供 LLM 进行最终推理。

**💡 创新点**

创新点：①首次系统性地在因果推理基准中引入可控的结构化噪声；②将显式因果图嵌入 LLM 推理流程，形成图引导提示，提升解释性与鲁棒性；③通过对图结构、提示设计与噪声组合的 ablation，揭示结构化引导对因果推理的决定性作用。

**🔧 技术方法**

技术：大型语言模型（如 GPT‑3.5、GPT‑4、LLaMA 等）与自定义提示策略；基于结构因果模型（SCM）生成真实因果图；结构化噪声注入机制；图引导推理框架（变量提取→图构建→结构化提示→LLM 推理）；多样化提示设计（Edge‑Only、Natural Prompt 等）。

**📊 数据集**

数据集：NoisyCausal（约10,617 题/答对，包含清洁与六类噪声组合），并在外部数据集 Cladder 进行泛化测试。

**📈 对比分析**

与传统提示（vanilla）、Chain‑of‑Thought、Tree‑of‑Thought、ReAct、Reflexion、Causal CoT 等方法对比：Graph‑Guided 在清洁数据上准确率 80.7%，噪声条件下保持 73–77%，远超 GPT‑4（62.8%）和 Causal CoT（73.4%）；在 Cladder 上无额外调优即获得 82.3%，说明方法具有良好的跨数据集泛化能力。

**⚠️ 局限性**

局限性：①对 LLM 变量提取与图构建的错误易导致连锁误差；②缺乏端到端联合训练，未能对提取/推理模块进行优化；③基准为合成数据，真实世界多模态或开放域任务的适用性尚待验证；④对噪声的模拟仍有限，未覆盖所有可能的真实噪声模式。

---

## 110. From Priors to Perception: Grounding Video-LLMs in Physical Reality

**arXiv ID:** 2605.04515 | [PDF](https://arxiv.org/pdf/2605.04515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 111. Transformation Categorization Based on Group Decomposition Theory Using Parameter Division

**arXiv ID:** 2605.04056 | [PDF](https://arxiv.org/pdf/2605.04056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 112. Information Coordination as a Bridge: A Neuro-Symbolic Architecture for Reliable Autonomous Driving Scene Understanding

**arXiv ID:** 2605.04475 | [PDF](https://arxiv.org/pdf/2605.04475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 113. Layerwise LQR for Geometry-Aware Optimization of Deep Networks

**arXiv ID:** 2605.04230 | [PDF](https://arxiv.org/pdf/2605.04230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 114. Second-Order FALQON Parameter Transfer for the Max-Cut Problem on 3-Regular Graphs

**arXiv ID:** 2605.04253 | [PDF](https://arxiv.org/pdf/2605.04253v1)

**作者:** Gabriel Fernandes Thomaz `[一作]` (Instituto de Pesquisas Eldorado), Evandro Chagas Ribeiro da Rosa `[通讯]` (UFSC)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在3-正则图上的最大割问题，探讨第二阶FALQON参数迁移的可行性。

**💡 创新点**

首次证明在第二阶FALQON中将小规模图的反馈参数迁移到大规模图可获得更高逼近比，显著降低参数发现成本。

**🔧 技术方法**

采用第二阶离散时间控制的FALQON算法、参数扫描、交叉评估与热力学模拟的Max-Cut基准。

**📊 数据集**

使用了节点数从6到24的20个随机3-正则图集合。

**📈 对比分析**

将迁移参数与在目标图上本地优化的参数对比，发现迁移方案在16层时的逼近比往往高出10-20%，且时间步更大。

**⚠️ 局限性**

仅在结构相似的3-正则图上验证，且图规模受限于模拟深度，未验证对非正则图或更大规模图的通用性。

---

## 115. Joint Semantic Token Selection and Prompt Optimization for Interpretable Prompt Learning

**arXiv ID:** 2605.04425 | [PDF](https://arxiv.org/pdf/2605.04425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 116. täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies (Extended Version)

**arXiv ID:** 2605.04172 | [PDF](https://arxiv.org/pdf/2605.04172v1)

**作者:** Pranav Srinivasan `[一作]` (University of Michigan), Yatin A. Manerkar `[通讯]` (University of Michigan)

**通讯引用:** 188 | [OpenAlex ID](https://openalex.org/A5027050350)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文为可编程内存层（PMH）提出了基于ISA的内存一致性模型（MCM），并构建了参数化的微架构实现模型，对其进行机器检查证明，验证实现满足该MCM。同时提供了一系列 litmus test、工具以及对程序员使用 MCM 的指导。

**💡 创新点**

创新点包括：① 首个兼顾缓存事件、回调与幻象地址的 ISA 级 MCM；② 通过前缀封闭（prefix‑closure）axioms 使得 MCM 可通过归纳方式进行机器检查；③ 设计的实现模型既支持架构师灵活调整预取/置换策略，又满足形式化验证者的可验证性，满足双向需求。

**🔧 技术方法**

采用的技术主要是形式化方法：axiomatic MCM 定义、Dafny 编写细粒度实现模型、SMT 证明（Z3）、Alloy 模型检查、环境化转移（environmental transitions）以及归纳不变量与细化证明（refinement）。

**📊 数据集**

没有使用真实数据集，全部采用人工设计的 litmus test 作为验证样本。

**📈 对比分析**

方法上通过对实现模型的所有可能执行进行机器检查，证明它们均满足 MCM；未做硬件性能评估，关注的是一致性正确性而非运行时速度。

**⚠️ 局限性**

局限性：证明规模大、证据复杂，可能难以推广到更大规模或其他架构；模型假设了特定的缓存一致性协议和预取/置换策略；对极端预取/置换行为的覆盖仍有限；以及未考虑多线程并发的运行时性能评估。

---

## 117. Autonomous Laparoscope Control through Unified Mechanics-Based Representation of Multimodal Intraoperative Information

**arXiv ID:** 2605.04408 | [PDF](https://arxiv.org/pdf/2605.04408v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 118. FMI_SU_Yotkova_Kastreva at SemEval-2026 Task 13: Lightweight Detection of LLM-Generated Code via Stylometric Signals

**arXiv ID:** 2605.04157 | [PDF](https://arxiv.org/pdf/2605.04157v1)

**作者:** Elitsa Yotkova `[一作]`, Preslav Nakov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

提供了一个示例，演示如何在 LuaLaTeX 或 XeLaTeX 中使用 ACL 样式文件进行排版

**💡 创新点**

演示了多语言文本（印地语、阿拉伯语）的支持，并展示了如何在 ACL 格式下编写参考文献

**🔧 技术方法**

使用了 ACL 样式文件、LuaLaTeX、XeLaTeX

**📊 数据集**

未使用任何实验数据集

**📈 对比分析**

该文档仅为排版示例，不包含实验或性能比较

**⚠️ 局限性**

缺乏实际实验内容，无法评估效果；仅限于演示排版与语言支持

---

## 119. Symmetry-induced quantum-inspired parallelism of classical dynamic systems

**arXiv ID:** 2605.04204 | [PDF](https://arxiv.org/pdf/2605.04204v1)

**作者:** Mikhail Erementchouk `[一作]` (University of Michigan), Pinaki Mazumder `[通讯]` (University of Michigan)

**通讯引用:** 9834 | [OpenAlex ID](https://openalex.org/A5068645158)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了利用经典非线性动力系统中的对称性实现多任务并行计算，提出并验证了“对称性诱导并行性”机制，利用松弛自旋网络（R‑DIM）完成布尔函数的多重评估；

**💡 创新点**

创新点在于：①提出对称性而非叠加原理可实现多重并行计算；②证明松弛自旋网络在单一系统内可同时完成多个布尔函数评估；③将布尔函数映射到Ising模型，利用单参数对称变换生成位翻转链，从而实现同步多评估；

**🔧 技术方法**

使用松弛自旋动力学模型（R‑DIM）及其目标函数（Relaxed Cut Function）；通过对称变换（相位圆平移）和Ising Hamiltonian构建布尔门（AND/OR、全加器、N 位加法器）；采用数值仿真验证多评估结果；

**📊 数据集**

实验使用随机生成的 32 位整数加法案例，配合自定义权重（包括去除线性相关的修正），对不同权重方案进行多次模拟，统计收敛成功率；

**📈 对比分析**

比较方法：在不同权重设置下测量两位数加法的收敛概率；结果显示：去除线性相关权重后成功率从低于 30% 提升至超过 90%，单独逐位处理时更高；总体性能表明在合理权重下，系统能稳定收敛到正确的多重评估结果；

**⚠️ 局限性**

限制包括：①对称性诱导并行性需要布尔函数在生成的位翻转链上保持单调性（isotonicity），不满足则出现链破坏；②系统存在动力学瓶颈，导致非理想的稳态；③对更复杂函数或更大规模网络的收敛性仍未保证，需要进一步研究是否可通过增加内部自旋或调整对称变换实现可靠并行。

---

## 120. HUGO-CS: A Hybrid-Labeled, Uncertainty-Aware, General-Purpose, Observational Dataset for Cold Spray

**arXiv ID:** 2605.04257 | [PDF](https://arxiv.org/pdf/2605.04257v1)

**作者:** Stephen Price `[一作]` (Worcester Polytechnic Institute), Danielle L. Cote `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 842 | [OpenAlex ID](https://openalex.org/A5003132026)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 HUGO-CS 这一大规模冷喷实验数据集，并提出了 HUGO 混合标注框架用于从文献中提取实验数据。

**💡 创新点**

创新点包括：① 通过 Hierarchical Risk Mitigation (HRM) 将 LLM 提取与有针对性的人为校正结合，显著提升标注效率与准确性；② 生成了迄今规模最大的 4,383 条实验记录（144 个特征）数据集；③ 提供了可复现的开源工具和代码。

**🔧 技术方法**

技术手段：使用 GPT‑4o‑mini 进行结构化提取；MinerU 将 PDF 转为 Markdown；HRM 对提取结果进行语法、结构、异常值和覆盖率四层筛选；Propose‑Inspect‑Review (PIR) 进行字符串映射、化学成分解析和单位标准化；最终发布 Python/JSON 数据和脚本。

**📊 数据集**

数据集来源：从 1,124 篇冷喷相关文献中手工/自动提取，包含 4,383 条实验记录，其中 1,765 条为完全人工标注的金标子集。

**📈 对比分析**

评估方法与性能：与先前最大 137 条记录的数据集相比，规模提升 30 倍；在未被 HRM 标记的 80 条手工验证样本中，LLM 的提取精度为 89.61%，召回率为 86.25%；在两组演示模型中，屈服强度预测 MAE 36.6 MPa、R² 0.66；微硬度预测 R² 0.65、MAE 87.12 HV。

**⚠️ 局限性**

局限性：低风险错误仍可能残留；部分特殊工艺（如多元混合粉末、激光重熔等）无法完全映射到现有 schema；图表信息提取仍不足，导致缺失实验；对异常值的统计筛选依赖阈值，可能误删或保留错误数据。

---

## 121. RetentiveKV: State-Space Memory for Uncertainty-Aware Multimodal KV Cache Eviction

**arXiv ID:** 2605.04075 | [PDF](https://arxiv.org/pdf/2605.04075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 122. Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search

**arXiv ID:** 2605.04057 | [PDF](https://arxiv.org/pdf/2605.04057v1)

**作者:** Zhen Liu `[一作]` (Xi'an Jiaotong University), Jingwen Fu `[通讯]` (Zhongguancun Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型进行可执行代码的神经网络架构搜索，并通过因子分离编辑实现结构化迭代进化

**💡 创新点**

提出SPARK框架，将编辑过程拆分为因子路由（ASR）、修订指令（RC）和因子条件补丁（SAR），从而显著降低功能耦合导致的失效率

**🔧 技术方法**

基于因子化代码标记的结构化编辑、可执行性快速检验（语法、接口、形状）、LLM路由/指令/编辑提示以及Archive‑based演化骨干

**📊 数据集**

CLRS算法推理基准（10个经典算法任务）

**📈 对比分析**

在与EvoPrompting、OpenEvolve、FunSearch、EoH同一搜索骨干和LLM编辑器下比较，SPARK在DFS上仅需57次评估即可达83.74% OOD，较EvoPrompting提升28.1×；跨10任务平均OOV 83.92%，MAC与基线相近

**⚠️ 局限性**

对因子划分的前置假设、对LLM提示成本的敏感性、仅适用于基于可执行代码的NAS，缺乏对更大规模或多模态任务的验证

---

## 123. LUCAS-MEGA: A Large-Scale Multimodal Dataset for Representation Learning in Soil-Environment Systems

**arXiv ID:** 2605.04323 | [PDF](https://arxiv.org/pdf/2605.04323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 124. Probing Structural Mathematical Reasoning in Language Models with Algebraic Trapdoors

**arXiv ID:** 2605.04352 | [PDF](https://arxiv.org/pdf/2605.04352v1)

**作者:** Igor Rivin `[一作]` `[通讯]`, Igor Rivin

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估基于SL(3,ℤ)子群构造的结构推理基准，探究模型对结构知识的掌握与自我否定能力。

**💡 创新点**

通过引入验证者‑证明者非对称的子群构造与四类行为评分，揭示模型在开放可判定性问题上的校准水平。

**🔧 技术方法**

结合Aschbacher分类、Congruence子群判定、符号计算（Sympy）、结构化推理轨迹记录等技术。

**📊 数据集**

使用自定义生成的SL(3,ℤ)实例（四类子群构造）以及SL(2,ℤ)正负样本作为测试数据集。

**📈 对比分析**

对GPT Pro和Gemini两模型进行手工推理轨迹对比，GPT在152分钟内识别并回避未决策，Gemini则崩溃；整体表现显示结构推理仍弱于简单命题推理。

**⚠️ 局限性**

样本量极小，仅两模型单轨迹；基准缺乏大规模统计；未验证对工具、不同语言模型的泛化。

---

## 125. Reproduction Test Generation for Java SWE Issues

**arXiv ID:** 2605.04320 | [PDF](https://arxiv.org/pdf/2605.04320v1)

**作者:** Toufique Ahmed `[一作]` (IBM), Martin Hirzel `[通讯]` (IBM)

**通讯引用:** 4803 | [OpenAlex ID](https://openalex.org/A5079080602)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了TDD-Bench-Java公共基准和针对Java的e‑Otter++生成重现测试的工作流，

**💡 创新点**

创新点在于首个针对Java的仓库级重现测试基准与完整的LLM驱动生成方案（结合定位、上下文、迭代反馈与多样化提示），

**🔧 技术方法**

主要技术包括大语言模型（Claude‑Sonnet‑4.5、GPT‑5.2）+定位/上下文化生成、执行反馈迭代、异质提示和测试选择器，

**📊 数据集**

使用了250个公开实例的TDD‑Bench‑Java以及150个IBM内部专有项目实例作为评估数据集，

**📈 对比分析**

通过fail‑to‑pass率评估，与Python同类方法相比，e‑Otter++在公开基准上达43.6%（Claude）/46.4%（GPT）fail‑to‑pass，且在专有数据上仅提升至20%（加入提示后），

**⚠️ 局限性**

局限性包括对专有项目的适用性差（短/欠缺描述、频繁新增文件导致定位困难）、对大型修复的覆盖不足以及模型对新文件/类的预测不准，

---

## 126. ANDRE: An Attention-based Neuro-symbolic Differentiable Rule Extractor

**arXiv ID:** 2605.04193 | [PDF](https://arxiv.org/pdf/2605.04193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 127. Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures

**arXiv ID:** 2605.04178 | [PDF](https://arxiv.org/pdf/2605.04178v1)

**作者:** Aaron Jarmusch `[一作]` (University of Delaware), Sunita Chandrasekaran `[通讯]` (University of Delaware)

**通讯引用:** 823 | [OpenAlex ID](https://openalex.org/A5009614578)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于系统微基准的阶段中心分析模型，对 NVIDIA Blackwell (B200) 与 AMD CDNA3 (MI300A) GPU 架构进行建模与验证，并迁移到前一代 H200/MI250X。

**💡 创新点**

首次构建并验证针对 Tensor Memory、TMA、5th‑gen Tensor Core 的 B200 模型与 Infinity Cache、VGPR/占用驱动的 MI300A 模型，并通过参数化实现跨平台可迁移。

**🔧 技术方法**

利用微基准提取硬件参数，阶段中心与波前中心的分析公式，结合占用、压缩、同步等阶段，进行精确执行时间预测。

**📊 数据集**

验证使用 21/27 个自研微基准以及 Rodinia 3.1、SPEChpc 2021 Tiny 等应用基准；还包含 H200、MI250X 的前代验证。

**📈 对比分析**

与 naive roofline 对比，模型 MAE 在 B200 为 1.3%、MI300A 为 0.09%；相较于 naive 的 95%+误差，模型显著提高；在应用基准上，B200 12.5% MAE，MI300A 1.3% MAE；跨平台迁移时误差升高但仍优于 roofline。

**⚠️ 局限性**

仅适用于规则计算与存储访问，稀疏/指针跳跃、极短 kernel 的误差较大；未建模缓存替换、多 GPU 互连热阻等；需要针对新硬件重新测量参数。

---

## 128. Laundering AI Authority with Adversarial Examples

**arXiv ID:** 2605.04261 | [PDF](https://arxiv.org/pdf/2605.04261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 129. Balanced Aggregation: Understanding and Fixing Aggregation Bias in GRPO

**arXiv ID:** 2605.04077 | [PDF](https://arxiv.org/pdf/2605.04077v1)

**作者:** Zhiyuan Zeng `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 18086 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了GRPO风格强化学习中不同的梯度聚合规则，尤其提出Balanced Aggregation（BA）方案；

**💡 创新点**

BA通过在正负样本内按token平均后再按样本数加权，解决了token聚合的符号-长度耦合偏差与序列聚合的等权偏差；

**🔧 技术方法**

采用RLVR框架下的GRPO训练，使用PPO裁剪、优势归一化及token/序列聚合方法；

**📊 数据集**

在数学推理数据集DAPO-17k和Polaris上进行训练，并在Math‑500、AIME 2024/25、OlympicBench、Minerva‑MATH和LivecodeBench等六个评测基准上测试；

**📈 对比分析**

与token-agg和seq-agg对比，BA在多模型、多数据集上均表现出更好的训练稳定性和更高的最终准确率，尤其在最终训练步骤保持更优性能；

**⚠️ 局限性**

仍受响应长度分布、正负长度差异等因素影响，需进一步探索在极端长度方差或高维奖励场景下的表现。

---

## 130. Investigating Trustworthiness of Nonparametric Deep Survival Models for Alzheimer's Disease Progression Analysis

**arXiv ID:** 2605.04063 | [PDF](https://arxiv.org/pdf/2605.04063v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 131. LEGO: LoRA-Enabled Generator-Oriented Framework for Synthetic Image Detection

**arXiv ID:** 2605.04445 | [PDF](https://arxiv.org/pdf/2605.04445v1)

**作者:** Yutong Xiao `[一作]` (University of Electronic Science and Technology of China), Caiyan Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1679 | [OpenAlex ID](https://openalex.org/A5071796939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LEGO框架，用生成器特定的LoRA模块和路由器实现AI生成图像的检测

**💡 创新点**

核心创新在于将检测拆分为生成器特定的低秩模块学习和输入自适应的路由融合，形成可扩展的模块化设计

**🔧 技术方法**

采用冻结的CLIP视觉编码器、LoRA低秩适配、MLP路由器、注意力融合与多阶段训练

**📊 数据集**

训练数据仅包含<30,000>张来自Stable Diffusion、ProGAN、ADM的图像，评估在AIGIBench和Chameleon等公开基准上

**📈 对比分析**

与Effort、TriDetect、HiDA-Net等SOTA方法对比，LEGO在AIGIBench平均ACC 80.2%、AP 89.6%以及Chameleon ACC 83.6%等指标上均实现领先，且训练样本量显著更少

**⚠️ 局限性**

局限性包括对极度新颖或未见生成器的识别仍需额外LoRA模块，且在极端后处理或多模态混合生成场景中鲁棒性待进一步验证

---

## 132. Imagery Dataset for Remaining Useful Life Estimation of Synthetic Fibre Ropes

**arXiv ID:** 2605.04262 | [PDF](https://arxiv.org/pdf/2605.04262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 133. Reddit's Globalization over Twenty Years: Inferring Community Time Zone from Activity Timestamps

**arXiv ID:** 2605.04371 | [PDF](https://arxiv.org/pdf/2605.04371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 134. Demystifying Manifold Constraints in LLM Pre-training

**arXiv ID:** 2605.04418 | [PDF](https://arxiv.org/pdf/2605.04418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 135. Geometry-Aware Neural Optimizer for Shape Optimization and Inversion

**arXiv ID:** 2605.04474 | [PDF](https://arxiv.org/pdf/2605.04474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 136. MedFabric and EtHER: A Data-Centric Framework for Word-Level Fabrication Generation and Detection in Medical LLMs

**arXiv ID:** 2605.04180 | [PDF](https://arxiv.org/pdf/2605.04180v1)

**作者:** Tung Sum Thomas Kwok `[一作]` (University of California, Los Angeles), Guang Cheng `[通讯]` (University of California, Los Angeles)

**通讯引用:** 2710 | [OpenAlex ID](https://openalex.org/A5043707940)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一个数据驱动的医学LLM造假生成与检测框架，能够生成与真实回答结构相似的词级造假文本，并基于此训练检测器。

**💡 创新点**

创新点包括：①三阶段造假生成管线（LLM重写、证据条件造假、结构相似度与SPPO双重质量控制）；②模块化词级检测器（Text2Table分解、Word Masking & Filling、Hybrid Sentence Pair Evaluation）实现对细粒度造假的精准判别。

**🔧 技术方法**

主要技术涉及：LLM重写与提示工程、ROUGE-L结构相似度筛选、SPPO偏好优化、RAG检索式填充、嵌入相似度评估以及LLM辅助推理。

**📊 数据集**

使用自构造的 MedFabric 数据集（基于 MedHallu 重写并造假）以及原始 MedHallu 数据集进行实验。

**📈 对比分析**

与多种SOTA方法（LLM-as-a-judge、SAPLMA、RelD、TSV、GCA）对比，整体 F1 提升约 15%，在结构相似度高的样本中检出率提升至约 68.6%，表现优于现有检测器。

**⚠️ 局限性**

局限性在于生成过程高度依赖 LLM，可能影响可复现性；对非医学领域的泛化未充分验证；以及模型仍需人工评估以确保高质量训练样本。

---

## 137. Two Integration Pathways in Human-Centered Requirements Engineering: A Systematic Mapping Study of Structural Gaps

**arXiv ID:** 2605.04132 | [PDF](https://arxiv.org/pdf/2605.04132v1)

**作者:** Imen Benzarti `[一作]` (École de Technologie Supérieure), Darine Amayed `[通讯]` (University of Quebec at Chicoutimi)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统映射研究对56篇人本需求工程论文进行归纳与分析，提出了两条整合路径（认知‑形式化与参与‑迭代），揭示其结构性分裂与研究空白，并给出跨路径研究议程。

**💡 创新点**

创新点在于：①提出认知‑形式化与参与‑迭代两条路径的对照分类，体现了学科、框架与工件的分离；②识别出缺失的翻译机制（Layer 3）及评估不足；③基于此提出经验中心化需求工程（XCRE）的四层框架与研究优先级。

**🔧 技术方法**

使用的技术包括系统映射方法、关键字检索、两阶段筛选、双人独立提取、开放编码构建主题、统计与交叉分析，以及将研究归类为路径与桥接研究。

**📊 数据集**

数据集为56篇原始研究，来源于Engineering Village（Compendex/Inspec）检索、前向/后向滚雪球筛选，涵盖1993‑2024年、47种会议/期刊，包含多学科、不同需求生命周期阶段。

**📈 对比分析**

比较方法主要是统计分布与评估率对比：认知‑形式化路径评估率54%，覆盖多阶段；参与‑迭代路径评估率29%，集中于获取阶段。两路径在框架类型、学科取向、工件类型上无交叉，显示结构分离。

**⚠️ 局限性**

局限性包括：①评估不足（仅39%被评估）；②缺乏跨路径融合案例；③语言仅限英语，可能遗漏非英语文献；④未给出统一的性能指标或实验验证；⑤对Layer 3翻译机制的识别仍基于少量案例，需进一步验证。

---

## 138. DeFed-GMM-DaDiL: A Decentralized Federated Framework for Domain Adaptation

**arXiv ID:** 2605.04324 | [PDF](https://arxiv.org/pdf/2605.04324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 139. Learning reveals invisible structure in low-rank RNNs

**arXiv ID:** 2605.04115 | [PDF](https://arxiv.org/pdf/2605.04115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 140. SemiConLens: Visual Analytics for 2D Semiconductor Discovery

**arXiv ID:** 2605.04067 | [PDF](https://arxiv.org/pdf/2605.04067v1)

**作者:** Kavinda Athapaththu `[一作]`, Yong Wang `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

无法获取论文内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法进行方法对比与性能评估

**⚠️ 局限性**

无法确定局限性

---

## 141. FlatASCEND: Autoregressive Clinical Sequence Generation with Continuous Time Prediction and Association-Based Pharmacological Testing

**arXiv ID:** 2605.04071 | [PDF](https://arxiv.org/pdf/2605.04071v1)

**作者:** Chris Sainsbury `[一作]` (University of Glasgow), Andreas Karwath `[通讯]` (University of Birmingham)

**通讯引用:** 2112 | [OpenAlex ID](https://openalex.org/A5026210150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并训练了一个14.5M参数的FlatASCEND自回归临床序列模型，能够基于患者临床前缀生成多步连续时间的医疗轨迹并对干预药物做出响应。

**💡 创新点**

创新点包括：平面复合标记(token) + 零膨胀对数正态时间头 + 以自由运行的scheduled sampling实现多步生成；以及基于患者级差分的药理关联测试框架，评估模型生成的轨迹与已知药理关系的一致性。

**🔧 技术方法**

使用技术：GPT‑2风格Transformer（ALiBi位置编码）、分量嵌入、序数Earth Mover’s Distance辅助损失、ZILN时间头、free‑running scheduled sampling、患者级Wilcoxon检验和置换检验等。

**📊 数据集**

数据集：专有糖尿病登记（61K患者）、公开数据集MIMIC‑IV（ICU）、INSPECT（门诊）以及eICU‑CRD（零样本跨站迁移测试）。

**📈 对比分析**

与一元/二元基线对比，FlatASCEND在Jaccard、模式崩溃率、教师强迫困惑度等指标上达到0.889–0.954（MIMIC‑IV/INSPECT），零样本eICU‑CRD下降至0.682，微调后恢复至0.820；在药理关联测试中，患者级差分检验成功恢复6/10机制性关联，prompt‑shuffle消除时序效应，显示对临床上下文的真正依赖。

**⚠️ 局限性**

局限性包括：门诊长期时间跨度预测不足、仅能捕获相关性而非因果关系、无临床可解释性验证、对零样本迁移高度敏感、依赖专有开发数据不可重复、基线评估指标无法区分学习与采样模型。

---

## 142. AsymmetryZero: A Framework for Operationalizing Human Expert Preferences as Semantic Evals

**arXiv ID:** 2605.04083 | [PDF](https://arxiv.org/pdf/2605.04083v1)

**作者:** Tadhg Looram `[一作]` (PortexAI), Steven Dillmann `[通讯]` (Stanford University)

**通讯引用:** 169 | [OpenAlex ID](https://openalex.org/A5116367400)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 AsymmetryZero 框架，将人类专家偏好转化为可复用的语义评估合同，并在 Harbor 代理环境中比较前沿模型与紧凑模型五人评审团在同一合同下的评估表现。

**💡 创新点**

创新点在于：① 以合同为核心的评估设计，统一模型与代理的评估逻辑；② 通过评审团而非单一 LLM 实现可审计的语义评分；③ 量化评审团规模对成本、延迟与判定一致性的折衷。

**🔧 技术方法**

使用技术包括：基于专家编写的 rubric（标准）与评审团聚合规则、Inspect 与 Harbor 运行 harness、LLM 评审模型、阶跃混合模型解析评审一致性、成本/延迟指标统计。

**📊 数据集**

使用的数据集为自研 PORTEX-COMPOSITE benchmark，包含 75 个前沿任务、4 类解决器（Opus-4.6、Gemini-3.1-Pro、GPT-5.4、Grok-4.20），共 4 轮评估。

**📈 对比分析**

比较方法是将同一评估合同下的五人前沿评审团与五人紧凑评审团在所有任务上对齐，并评估：criterion 级别的一致率（前沿 84%/紧凑 31%）、内部分歧率（前沿 9%/紧凑 31%）、任务级原始分数相关系数（≈0.88）以及成本/延迟（紧凑约 97% 费用下降、82% 延迟下降）。

**⚠️ 局限性**

局限性包括：未对评审结果进行人类标注验证；仅在单一 Harbor/Terminus‑2 代理设置下评估；任务多为 STEM 领域，缺乏对业务流程等多样场景的覆盖；评估时设定 10 轮上限导致部分空结果；成本与延迟基于 OpenRouter 提供的统计，未完全对账；框架的可靠性高度依赖合同质量。

---

## 143. Beyond Ability: The Four-Fold Spectrum of Power and the Logic of Full Inability

**arXiv ID:** 2605.04452 | [PDF](https://arxiv.org/pdf/2605.04452v1)

**作者:** Shanxia Wang `[一作]` (Henan Normal University), Shanxia Wang `[通讯]` (Henan Normal University)

**通讯引用:** 271 | [OpenAlex ID](https://openalex.org/A5078371286)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了全无能（Full Inability）概念，扩展了联盟逻辑（Coalition Logic）中的否定能力，构建了完整的四分支战略谱（Full Control、Positive Determination、Adverse Determination、Full Inability），并对其代数结构、对称性、以及在可玩模型中的凸性等性质进行形式化与证明；随后引入CLFI逻辑，将全无能作为原语实现，给出其公理化、可化简性、完备性、保守性及与原联盟逻辑相同的P‑SPACE复杂度。

**💡 创新点**

创新点包括：① 将传统的单一否定能力拆解为更细粒度的全无能，揭示其内部结构；② 构造四分支战略谱并证明其在α‑duality下形成Klein四元群对称性；③ 证明全无能在效应力集上形成顺序凸集，提供区间稳定验证；④ 将全无能作为原语加入联盟逻辑，保持表达力与复杂度不变，同时获得更直接的证明理论工具。

**🔧 技术方法**

主要技术手段有：效应力函数与可玩模型的语义；α‑duality与正则性推导的逻辑语义；代数对称性（Klein四元群）与贝尔朗双位阶结构的映射；凸性与格理论的顺序论证；以及通过消元翻译实现CLFI与基础联盟逻辑的等价性证明。

**📊 数据集**

无实验数据集，本工作为完全形式化理论研究，无需实证数据。

**📈 对比分析**

本文未进行算法性能对比，仅给出理论复杂度结果：CLFI的可满足性问题与传统联盟逻辑保持P‑SPACE‑complete，证明了保守扩展的复杂度不升高。

**⚠️ 局限性**

局限性包括：① 依赖可玩模型与α‑duality假设，某些性质在一般模型下失效；② 仅覆盖静态联盟逻辑，未扩展到时序或知识形式；③ 只处理定性“全无能”，未提供量化或概率化度量；④ 目前的理论尚未与实际多智能体系统或安全规范结合验证。

---

## 144. A Self-Attentive Meta-Optimizer with Group-Adaptive Learning Rates and Weight Decay

**arXiv ID:** 2605.04055 | [PDF](https://arxiv.org/pdf/2605.04055v1)

**作者:** JiangBo Zhao `[一作]`, ZhaoXin Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 MetaAdamW，一种在 AdamW 基础上加入自注意力机制和元学习的优化器，可对不同参数组动态调节学习率与权重衰减，并通过多目标元学习目标实现更优泛化。

**💡 创新点**

创新点在于：① 用轻量 Transformer 对参数组特征进行自注意力加权，实时生成组级学习率和权重衰减；② 设计结合梯度对齐、验证下降和泛化差距的三项元目标；③ 引入任务优先级注入的 homoscedastic uncertainty 权重，实现自动平衡且可注入领域知识；④ 采用细粒度分组与多版本特征，提升适配性。

**🔧 技术方法**

使用技术包括：Transformer Encoder、AdamW、元学习（双层优化）以及 homoscedastic uncertainty 与优先级注入的权重机制；同时实现特征提取、特征门控与多任务加权。

**📊 数据集**

实验数据集涵盖：ETTh1 时序预测、WikiText-2 语言建模、Multi30k 翻译、CIFAR-10 图像分类和 IMDB 情感分析。

**📈 对比分析**

通过与标准 AdamW 进行同一学习率、权重衰减配置的对比，MetaAdamW 在 Transformer 任务中显著降低验证误差并提前停止，减少训练时间；在非 Transformer 任务中提升准确率但训练时间略增，总体表现优于基线。

**⚠️ 局限性**

局限性在于：需针对不同任务调节元学习频率、参数组策略和 Transformer 规模；在大规模模型上的可扩展性未验证；部分任务训练时间显著增加，且元更新额外计算开销需进一步优化。

---

## 145. Covariance-Aware Goodness for Scalable Forward-Forward Learning

**arXiv ID:** 2605.04346 | [PDF](https://arxiv.org/pdf/2605.04346v1)

**作者:** Xiaoyi Jiang `[一作]` (King's College London), Kai Xu `[通讯]` (King's College London)

**通讯引用:** 6130 | [OpenAlex ID](https://openalex.org/A5086469511)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在无梯度反向传播的前向-前向算法中改进好感度提取，提出双轴协方差好感度（BiCovG）、轻量级特征对齐层（FAL）以及可调梯度范围的混合好感度块（HGB），实现深度网络的高效训练。

**💡 创新点**

创新点在于：①利用跨通道投影与多尺度空间聚合双轴提取第二阶统计，显著提升好感度的判别性；②引入零初始化的特征对齐层解决梯度隔离块间的表征失配；③设计可变块大小的混合好感度块在保持内存优势的同时逐步逼近全BP性能。

**🔧 技术方法**

采用前向-前向学习框架，结合BiCovG的1×1卷积投影、RMSNorm、RMSPool、Logistic Fusion与可调块梯度传递的混合好感度块（HGB），实现无BP深度CNN训练。

**📊 数据集**

在CIFAR‑100、Tiny‑ImageNet（64×64，200类）和ImageNet‑100（224×224，100类）三大数据集上进行实验验证。

**📈 对比分析**

与之前的FF与BP-free方法相比，BiCovG+FAL在ImageNet‑100上实现73.01%准确率，Tiny‑ImageNet上50.30%；通过HGB（m=4）进一步提升至83.98%（仅比全BP低3.6%），同时将峰值GPU内存降低约47%。

**⚠️ 局限性**

限制：仅在100类的ImageNet‑100上测试，未验证在完整ImageNet‑1K上的表现；FAL的效果对数据集和网络深度差异显著，HGB的梯度范围与混合精度、异构硬件的兼容性仍需进一步探索。

---

## 146. Coral: Cost-Efficient Multi-LLM Serving over Heterogeneous Cloud GPUs

**arXiv ID:** 2605.04357 | [PDF](https://arxiv.org/pdf/2605.04357v1)

**作者:** Yixuan Mei `[一作]` (Carnegie Mellon University), K. V. Rashmi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3175 | [OpenAlex ID](https://openalex.org/A5109919225)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向异构云GPU的多LLM低成本调度系统，联合优化资源分配与模型部署，并实现动态自适应

**💡 创新点**

通过无损的两阶段分解，将模型放置离线预计算为可复用的Serving Template，显著减少在线求解时间，首次实现多模型协同、跨区域资源调度的整体最优

**🔧 技术方法**

整数规划（ILP）用于离线模板生成与在线分配；混合流水线+数据并行模型布局；ZeroMQ+NCCL分布式通信；vLLM作为执行引擎

**📊 数据集**

利用Azure Code、Azure Conversation、BurstGPT三大请求日志；对六大LLM（Qwen‑3 32B/235B、GPT‑OSS 20B/120B、Phi4‑14B、Llama‑3 70B）在多种GPU配置上进行实验，并使用模拟器进行大规模验证

**📈 对比分析**

与Helix、Cauchy、SkyServe、SageServe等基线比较，平均可降低成本2.79×、提升goodput 2.39×；在线ILP求解仅需0.1–10 s；在资源紧张场景下仍保持高吞吐

**⚠️ 局限性**

局限于prefill–decode分离模式，仅支持流水线与数据并行；离线模板生成需要离线profiling和较长预处理时间；对极端GPU异构或新硬件支持有限；对模型规模与内存需求仍有硬件上限

---

## 147. Faster Iterative $φ$ Queries on the Positional BWT

**arXiv ID:** 2605.04244 | [PDF](https://arxiv.org/pdf/2605.04244v1)

**作者:** Paola Bonizzoni `[一作]` (University of Milano-Bicocca), Younan Gao `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5071941755)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种新的分解方案，将 PBWT 中的单倍型行分解为若干细化区间（refined segments），并基于此设计两种空间‑时间权衡的数据结构，以支持迭代的 φ 查询。

**💡 创新点**

①新颖的细化分解算法保证每个细化区间与其前驱行最多重叠常数个区间；②总细化区间数上界为 O(r + n + ⌈(r + n)/d-1⌉)。基于此实现了两种更高效的 φ 查询方案，突破了先前的 O((r + n)log n) 位空间 + O(k loglog n) 时间的限制。

**🔧 技术方法**

细化分解算法、父子关系维护、后继查询、位向量+ rank/select 结构、稀疏数组、分块、分解理论，以及 PBWT 的 Run‑Tops 结构。

**📊 数据集**

论文未给出实验数据，理论上针对英国生物银行、1000 Genomes 等常见单倍型面板。

**📈 对比分析**

与之前的 μ‑PBWT 与 move 结构比较；第一权衡在空间上与前者相同但查询时间从 O(k loglog n) 减少到 O(loglog min(r,n)+k)，第二权衡进一步将空间降到 O(log r + log n)，查询时间保持 O(k loglog n)；在现代基因组数据（r ≪ n）上表现更优。

**⚠️ 局限性**

细化分解算法实现复杂、常数因子可能较大；未在真实数据上评估，理论复杂度对 r 的依赖在 r 非常大时可能影响；仅针对迭代 φ 查询，对其他 PBWT 关键操作（如前向/后向跳步）未作改进。

---

## 148. Nearly-Tight Bounds for Zonotope Containment and Beyond

**arXiv ID:** 2605.04183 | [PDF](https://arxiv.org/pdf/2605.04183v1)

**作者:** Friedrich Eisenbrand `[一作]` (EPFL), Ruben Skorupinski `[通讯]` (EPFL)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了凸体包含问题，尤其聚焦于内层为锥体（zonotope），外层由 membership oracle 给出的任意凸体。作者提出了一种基于随机采样与稀疏化的算法，能够在多项式时间内获得一个 O(√d) 的近似因子，并在假设 Talagrand 猜想成立的情况下得到最优 Θ(√(d/ log d)) 的近似。进一步，论文证明了 Δ‑modular 锥体可以线性稀疏化，进而实现最优的 O(Δ√(d/ log d)) 包含算法。作者还给出了针对所有锥体的通用下界 Ω(√(d/ log d))，并对一般凸体给出匹配的 Ω(d/ log d) 下界，说明了 Barvinok 存在的多面体近似结果在 oracle 模型下不可多项式实现。

**💡 创新点**

创新点主要包括：① 将采样与 Talagrand 稀疏化相结合，首次给出 O(√d) 的随机近似算法；② 在 Δ‑modular 锥体上证明了 Talagrand 猜想，取得线性生成元稀疏化；③ 给出所有锥体的通用下界 Ω(√(d/ log d))，并证明对一般凸体的下界 Ω(d/ log d) 与上界匹配；④ 明确指出 Barvinok 的多面体近似方法在 oracle 模型下无法实现，从而划定了算法可行性的边界。

**🔧 技术方法**

核心技术包括：采样与反集中概率（Anti‑concentration on hypercubes）；Talagrand 稀疏化与 Spectral sparsification（Batson‑Spielman‑Srivastava 的矩阵稀疏化）；凸体极点/支撑函数的极值方法；John 位置与 Urysohn 不等式的体积与平均宽度估计；极体与极角的对偶关系；随机采样构造多面体近似；以及复杂度下界构造中的随机化与极值论证。

**📊 数据集**

该工作为纯理论研究，未使用任何实验数据集，所有结果均基于数学证明与符号计算。

**📈 对比分析**

与已有的下界（Ω(√(d/ log d)) 与 Ω(d/ log d)）保持一致，且在假设下给出了匹配的上界。对于锥体，算法在多项式时间内实现了最优的 Θ(√(d/ log d)) 近似；对于一般凸体，随机化多项式时间算法实现了 O(d/ log d) 的近似，并证明了该因子不可进一步改进。

**⚠️ 局限性**

局限性包括：① 仍需假设 Talagrand 猜想才能得到最优锥体结果；② 对非 Δ‑modular 锥体的稀疏化结果尚未给出；③ 结果仅适用于中心对称或通用 convex bodies 的 oracle 模型，缺乏对实际输入（如多面体描述）的直接应用；④ 由于使用随机化技术，算法在实际实现中可能需要较高的样本量或对 oracle 的高效访问。

---

## 149. Submodular Ground-Set Pruning: Monotone Tightness and a Non-Monotone Separation

**arXiv ID:** 2605.04428 | [PDF](https://arxiv.org/pdf/2605.04428v1)

**作者:** Alan Kuhnle `[一作]` (Texas A&M University), Alan Kuhnle `[通讯]` (Texas A&M University)

**通讯引用:** 617 | [OpenAlex ID](https://openalex.org/A5083575259)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了子模最大化中的“保留性裁剪”问题，旨在从海量集合中预先筛选出一个小核心集合，保证在任何后续预算下仍能找到近似最优解。

**💡 创新点**

提出并证明了多项理论极限：对单调子模，贪心算法在任何裁剪预算下都能获得 1-1/e 的保留因子，且该阈值是最优；对非单调子模，在基数约束下实现 1/2‑ε 的保留因子，超越了已知的最大化比例与硬性阈值，显示裁剪本质上比直接优化更易。进一步将非单调结果推广到背包约束，并给出可行的窗口式与连续贪心算法。

**🔧 技术方法**

采用贪心、阈值贪心、窗口保留、密度贪心等经典子模技术，结合随机化构造与概率不等式进行硬性证明，并利用连续双贪心等新方法验证上界。

**📊 数据集**

实验使用：随机 Erdős–Rényi 与植入分区图、真实 SNAP 社交网络（Facebook、Wiki‑Vote）进行 MaxCut 评估；HotpotQA 题集与 MuSiQue 试点用于大语言模型上下文选择。

**📈 对比分析**

与标准贪心、QuickPrune、COMBHelper、随机采样等基线对比；在 MaxCut 上保留因子在 0.99 以上，IP 求解速度提升约 620 倍；在 LLM 上，非单调代理裁剪在 38.7% 的问题中提升 F1/EM，MuSiQue 上提升黄金检索召回率 9.2%。

**⚠️ 局限性**

在非单调子模下，1/2 与 1-1/e 之间仍有空隙；仅针对基数与背包约束，未覆盖更一般约束；对代理函数的准确性与迁移性仍需进一步验证。

---

## 150. Safety by Invariance, Liveness through Refinement: Heterogeneous Contract Framework for Co-Design of Layered Control

**arXiv ID:** 2605.04222 | [PDF](https://arxiv.org/pdf/2605.04222v1)

**作者:** Yoshinari Takayama `[一作]` (Paris-Saclay University), Adnane Saoud `[通讯]` (University Mohammed VI Polytechnic)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种异构假设-保证（AG）合同框架，用于分层控制体系的安全与活性（liveness）协同设计，结合显式参考调度器（ERG）和模型预测控制（MPC）实现对混合储能系统的安全约束保证与能量目标达成；

**💡 创新点**

创新点在于：①将安全-活性分解引入AG合同，实现跨时域的垂直细化与时间兼容性；②设计基于ERG的实时参考滤波器，实现对连续时间安全约束的前向不变性；③提供一套完整的理论证明与仿真验证，展示该框架在实际电网储能系统中的可行性；

**🔧 技术方法**

技术包括：假设-保证合同理论、垂直细化与时间兼容性约束、模型预测控制（MPC）、显式参考调度器（ERG）、输入状态稳定性（ISS）分析、鲁棒递归可行性和控制不变性验证；

**📊 数据集**

实验以混合储能系统（电池+超级电容）为案例，使用该系统的物理参数和负载波形进行仿真验证，并未使用公开数据集；

**📈 对比分析**

与传统基于QP的控制栏过滤（如CBF-QP）相比，所提方法在保持低层跟踪器不变的前提下实现安全约束；仿真结果显示系统始终满足电压、电流等安全约束，并在预定时间内实现电池SOC目标，性能优于单一层控制或无约束方案；

**⚠️ 局限性**

局限性包括：①基于ISS的收敛时间上界保守，导致对采样周期的需求过高；②需要已知或可预测的负载信息以实现前馈取消，限制了对未知扰动的鲁棒性；③目前仅适用于单机层级结构，尚未推广到分布式或多智能体系统；

---

## 151. Hierarchical Support Vector State Partitioning for Distilling Black Box Reinforcement Learning Policies

**arXiv ID:** 2605.04254 | [PDF](https://arxiv.org/pdf/2605.04254v1)

**作者:** Senne Deproost `[一作]` (Vrije Universiteit Brussel), Ann Nowé `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 8474 | [OpenAlex ID](https://openalex.org/A5064553018)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于支持向量机的状态空间划分方法（SVSP），通过构建可解释子策略树来模仿深度强化学习中的黑盒策略。

**💡 创新点**

创新点在于使用线性SVM进行层次化划分，并利用价值网络（critic）对子策略的性能进行评估，从而显著减少子策略数量并提升回报；相比传统的Voronoi划分，SVSP实现了更紧凑、更可解释的决策边界。

**🔧 技术方法**

使用的技术包括：深度强化学习（TD3）训练的原始策略；价值网络（critic）进行回报评估；线性支持向量机用于状态划分；层次化决策树结构；线性子策略的拟合与评估。

**📊 数据集**

实验数据集为Gymnasium的LunarLanderContinuous环境，采集自1000条TD3训练得到的状态-动作对，并随机抽取30%作为分区训练集。

**📈 对比分析**

通过与原始TD3和Voronoi State Partitioning（VSP）两种基线方法对比，SVSP在LunarLander上的平均回报为166.3±31.07，分别比TD3高约2.8%和VSP高约7.4%；子策略数量从VSP的约56条降低到仅10条，减少了82.1%。

**⚠️ 局限性**

局限性包括：仅在单一控制任务上验证，缺乏对更复杂或多任务环境的评估；树的最大深度固定为3，未探究更深层次划分的潜力；使用线性SVM导致对非线性边界的捕捉受限；决策树训练过程对阈值和超参数敏感；实验中出现较大方差，提示需进一步优化和稳定化。

---

## 152. Single-Position Intervention Fails: Distributed Output Templates Drive In-Context Learning

**arXiv ID:** 2605.04061 | [PDF](https://arxiv.org/pdf/2605.04061v1)

**作者:** Bryan Cheng `[一作]`, Jasper Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过激活干预实验，证明LLM的ICL任务身份在输出模板上是分布式的，单点干预无法实现任务转移，只有在约30%网络深度同时替换所有示例输出token时才能达到高转移。

**💡 创新点**

其创新点在于首次发现单位置干预完全失败，提出并验证分布式输出模板假设，揭示ICL任务信息在多示例输出之间分散存储，并对格式高度敏感。

**🔧 技术方法**

采用了激活干预、线性探针、噪声注入、格式兼容性分析和因果追踪等技术，并在四种不同Transformer架构上进行跨模型复现。

**📊 数据集**

使用了包含程序化、数值和语义三类的八个少样本任务（如 uppercase、repeat_word、linear_2x、sentiment 等）的自建任务集，并在每对任务之间进行5-shot提示实验。

**📈 对比分析**

通过转移率 τ 与干扰率 δ 的量化评估，发现多点干预在层8时可实现约96%转移率（95%置信区间 [87%,99%]），而单点干预为0%；格式兼容性实验表明只有输出模板兼容时才出现显著转移。

**⚠️ 局限性**

局限性包括仅在确定性转换任务上验证，缺乏对复杂推理任务的评估；实验主要关注输出模板匹配，未深入解析内部机制；单点干预失败的原因可能受方法限制，模型规模扩展性仍待进一步探究。

---

## 153. Lightweight Vulnerability Detection from Code Metrics and Token Features

**arXiv ID:** 2605.04260 | [PDF](https://arxiv.org/pdf/2605.04260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 154. Deep Wave Network for Modeling Multi-Scale Physical Dynamics

**arXiv ID:** 2605.04198 | [PDF](https://arxiv.org/pdf/2605.04198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 155. FASQ: Flexible Accelerated Subspace Quantization for Calibration-Free LLM Compression

**arXiv ID:** 2605.04084 | [PDF](https://arxiv.org/pdf/2605.04084v1)

**作者:** Ye Qiao `[一作]` (University of California, Irvine), Sitao Huang `[通讯]` (University of California, Irvine)

**通讯引用:** 822 | [OpenAlex ID](https://openalex.org/A5050532440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FASQ框架，利用产品量化压缩LLM权重，实现27–49% FP16模型大小的连续压缩区间，并在单张RTX 3090上实现实时解码。

**💡 创新点**

创新点在于：①无校准数据的产品量化；②通过调节子向量尺寸与码表大小提供连续的压缩-质量曲线；③设计无LUT的解码核和双缓冲LUT的预填充核，解码速度超越FP16，突破传统标量量化的速度瓶颈。

**🔧 技术方法**

使用产品量化（PQ）、k‑means聚类、CUDA自定义核（LUT‑free GEMV、双缓冲LUT GEMM）以及split‑K并行技术。

**📊 数据集**

在Meta‑Llama‑3‑8B、Qwen3‑8B、Qwen3.5‑9B‑Base等模型上进行zero‑shot任务和WikiText‑2困惑度评测。

**📈 对比分析**

与GPTQ、AWQ、SmoothQuant、QuIP、RTN等基线比较，FASQ在相同压缩率下准确率相当或更好；解码吞吐率提升1.6–4.3×，在RTX 3090上实现45.2 tok/s（有效4位）或51.8 tok/s（有效3位），内存占用比FP16低2.56–2.80×。

**⚠️ 局限性**

主要限制是预填充阶段仍慢于cuBLAS；未针对更大模型进行自适应参数分配或代码簿微调，且当前仅在单GPU上验证。

---

## 156. DAO-enabled decentralized physical AI: A new paradigm for human-machine collaboration

**arXiv ID:** 2605.04522 | [PDF](https://arxiv.org/pdf/2605.04522v1)

**作者:** Mark C. Ballandies `[一作]` (University of Zurich), Claudio J. Tessone `[通讯]` (University of Zurich)

**通讯引用:** 4538 | [OpenAlex ID](https://openalex.org/A5020270223)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了 DAO‑Enabled Decentralized Physical AI（DePAI）框架，整合区块链、DAO、加密经济学与数字民主理念，构建了一个从能源、感知、连接、存储/计算、模型到机器人等垂直堆栈的去中心化物理‑数字系统治理与执行流程。

**💡 创新点**

创新点包括：①将 DAO 与 DePIN 结合，首次系统化阐述物理基础设施与 AI 的民主协同；②提出了“价值敏感设计+持续适应治理”的安全与伦理方法；③在数字民主研究与 DAO 设计之间搭建双向桥梁，提出更完善的协商与投票机制；④将 AI 助手与 DAO 结合，实现提案自动化、投票建议与风险检测。

**🔧 技术方法**

主要技术手段包括：区块链技术（PoW/PoS、智能合约）、DAO 治理框架（多种投票机制如二次投票、平方根投票、Futarchy 等）、加密经济学（代币激励、经济多重性）、去中心化物理基础设施网络（DePIN）以及 AI 组件（大语言模型、NLP、图神经网络）。

**📊 数据集**

未使用传统意义上的实验数据集；研究以文献综述、案例分析（如 MakerDAO、Helium、Ethereum DAO Hack 等）和理论模型为主。

**📈 对比分析**

由于本工作为概念与框架性研究，未进行基准实验或性能对比；作者通过对比已有 DAO 与 DePIN 实例，阐述理论可行性，并指出需要后续实验验证的关键指标（如网络去中心化程度、激励合规性、投票参与率）。

**⚠️ 局限性**

局限性包括：①缺乏实证验证与性能评估；②代币激励可能导致中心化、激励失配与外部攻击风险；③对法律合规性与链外执行机制的依赖；④在实际部署中对人类监督与 AI 安全性的保障仍未完善；⑤数字民主与 DAO 之间的交互设计尚不成熟，需进一步研究。

---

## 157. Orchestrating Serverless Applications in the Edge Cloud Space Continuum: What Breaks and What is Next?

**arXiv ID:** 2605.04316 | [PDF](https://arxiv.org/pdf/2605.04316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 158. FLUID: Continuous-Time Hyperconnected Sparse Transformer for Sink-Free Learning

**arXiv ID:** 2605.04421 | [PDF](https://arxiv.org/pdf/2605.04421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 159. Binary Image-Based Intrusion Detection for Operational Technology Networks: Extending the SPHBI Methodology from IoT to Modbus TCP

**arXiv ID:** 2605.04250 | [PDF](https://arxiv.org/pdf/2605.04250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 160. Evaluation Cards for XAI Metrics

**arXiv ID:** 2605.04410 | [PDF](https://arxiv.org/pdf/2605.04410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 161. When Context Hurts: The Crossover Effect of Knowledge Transfer on Multi-Agent Design Exploration

**arXiv ID:** 2605.04361 | [PDF](https://arxiv.org/pdf/2605.04361v1)

**作者:** Saranyan Vigraham `[一作]` `[通讯]` (Meta), Saranyan Vigraham (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多智能体软件设计任务中，作者系统地评估了不同知识工件（转录、拓扑、设计文档、反模式、代码、无上下文、无关文档）对设计探索的影响，发现了“交叉效应”：同一工件在某些任务中显著提升探索范围，而在其他任务中则大幅削弱探索。

**💡 创新点**

创新点在于：①首次证明上下文并非始终有益，而是取决于任务的基线探索度；②提出基线探索度（无上下文实验得到）作为预测工件效应的诊断指标；③区分自然收敛（训练数据偏好）与诱导收敛（提示强制）两种机制，解释交叉效应的根源。

**🔧 技术方法**

技术手段包括：使用 Claude Sonnet 4 的5人团队进行并行推理与综合；通过“直接评估”（Direct Tradeoff Assessment）利用 LLM 读取团队讨论，统计已知权衡的覆盖率；利用提示强度控制收敛压力；统计分析使用 Welch t 检验、Cohen d、Pearson/Spearman 相关系数。

**📊 数据集**

数据集由 10 个软件设计任务组成（5个通用任务如计数器、LRU 缓存；5个领域任务如 Kubernetes 运算符、数据库存储引擎），每个任务都有人工定义的权衡列表；实验共计 2,700+ 运行，涵盖 7 种工件类型和多种提示强度。

**📈 对比分析**

对比方法：将每种工件条件下的权衡覆盖率与无上下文基线对比，计算差异并进行统计显著性检验；展示完整的 10×7 效果矩阵；通过相关性分析验证基线探索度与工件效应的逆相关；在诱导收敛实验中观察工件效应随提示强度的变化。性能表现：在基线探索度低（如计数器）时，反模式和转录可将覆盖率提升 20–30 倍；而在探索度高的任务中，代码和转录会导致 20–50% 的下降。

**⚠️ 局限性**

局限性包括：①仅使用 Claude Sonnet 4，结果可能不适用于其他 LLM；②已知权衡列表的主观性可能影响覆盖率绝对值；③评估使用与生成同一 LLM 可能产生自相关偏差；④实验仅关注设计探索，不评估最终代码质量或系统性能；⑤温度 0.5 与 5 人团队的设置可能影响一般化；⑥无人工注释验证，部分结论需进一步人类评估。

---

## 162. Sparse Autoencoder Decomposition of Clinical Sequence Model Representations: Feature Complexity, Task Specialisation, and Mortality Prediction

**arXiv ID:** 2605.04072 | [PDF](https://arxiv.org/pdf/2605.04072v1)

**作者:** Chris Sainsbury `[一作]` (University of Glasgow), Andreas Karwath `[通讯]` (University of Birmingham)

**通讯引用:** 2112 | [OpenAlex ID](https://openalex.org/A5026210150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在 FlatASCEND 这 14.5 M 参数的电子健康记录（EHR）基础模型上，作者系统地训练 TopK 稀疏自编码器（SAE），通过解释隐藏层的稀疏特征来揭示模型在 transformer 深度中从 token 检测器到分布式临床概念的逐层抽象，并在多数据集上评估 SAE 特征与稠密表示在离散事件（死亡）与连续量化（住院时长）预测中的任务依赖特化；同时提出 delta‑mode 干预降低 SAE 重建噪声、探究特征因果性；在 eICU‑CRD、MIMIC‑IV、INSPECT 三个公开数据集上进行零样本迁移与时间窗口泄漏安全评估，发现 SAE 在完整序列死亡预测上略优，但在泄漏安全窗口下稠密表示更好，且对长度预测稠密表示占优；对特征可重复性、扰动实验显著性、下游模型简化以及数据偏向等方面提出局限。

**💡 创新点**

首次将稀疏自编码器系统地应用于 EHR 基础模型，揭示 Transformer 深度中概念抽象的层次化模式，并发现特征稀疏化与预测任务类型之间的交互特化；提出 delta‑mode 干预显著降低重建噪声，并在特征扰动实验中验证部分因果关联。

**🔧 技术方法**

TopK 稀疏自编码器（SAE）训练、残差流特征提取、delta‑mode 介入方法、线性与岭回归等下游评估、离散与连续预测的 AUC/R² 对比。

**📊 数据集**

FlatASCEND 训练集：MIMIC‑IV 与 INSPECT；零样本迁移验证集：eICU‑CRD；所有数据均来自 PhysioNet/Stanford。

**📈 对比分析**

与稠密隐藏层表示、Bag‑of‑Tokens 基线以及不同时间窗口（48 h ICU、1 y/3 y 产出）进行对比；在完整序列死亡预测中 SAE AUC≈0.93‑0.96（优势），连续预测长度时 R² 负值（劣势）；在泄漏安全窗口中稠密表示 AUC≥0.88，SAE 仅略逊。

**⚠️ 局限性**

特征重现率仅 21%，扰动实验未达显著性；下游模型过于简单；数据以 ICU 为主，缺乏多样化外部验证；单一 14.5 M 参数模型，缺少对大规模 EHR 模型的普适性验证；模型缺乏种族/族裔信息，公平性评估受限。

---

## 163. RaguTeam at SemEval-2026 Task 8: Meno and Friends in a Judge-Orchestrated LLM Ensemble for Faithful Multi-Turn Response Generation

**arXiv ID:** 2605.04523 | [PDF](https://arxiv.org/pdf/2605.04523v1)

**作者:** Ivan Bondarenko `[一作]` (Novosibirsk State University), Mikhail Kulakov `[通讯]` (Novosibirsk State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个由七个不同大模型组成的异质集成系统，并用 GPT-4o-mini 进行实例级别的选择，完成了SemEval‑2026 Task 8 B 的多轮检索增强生成任务。

**💡 创新点**

通过多模型多提示的异质集成和轻量级 LLM 评判器的结合，实现了比任何单一模型更高的条件化谐波均值；同时提出了 7B 领域适配模型 Meno‑Lite‑0.1 以实现成本性能折中。

**🔧 技术方法**

使用两种提示策略（迭代改进的系统提示和类别感知少样本提示）、七个跨供应商与规模的 LLM、GPT‑4o‑mini 作为评判器、vLLM 推理等技术。

**📊 数据集**

在 MTRAGEval Task B 的测试集（507 例，包含 FiQA、IBMCloud、CLAPnq、Govt 四个领域）上进行评估。

**📈 对比分析**

与官方基线 gpt‑oss‑120b (HM_3 0.639) 对比，系统达 0.7827，排名第一，提升绝对值 0.1437（约 22.5% 相对提升）。

**⚠️ 局限性**

主要限制包括：七个生成器与评判器导致计算成本高、实时延迟大；评判器依赖 GPT‑4o‑mini，可能对不同模型失效；空上下文的“无答案”捷径降低了对真实可回答性检测的挑战。

---

## 164. Nsanku: Evaluating Zero-Shot Translation Performance of LLMs for Ghanaian Languages

**arXiv ID:** 2605.04208 | [PDF](https://arxiv.org/pdf/2605.04208v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 165. Awaking Spatial Intelligence in Unified Multimodal Understanding and Generation

**arXiv ID:** 2605.04128 | [PDF](https://arxiv.org/pdf/2605.04128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 166. Membership Inference Attacks for Retrieval Based In-Context Learning for Document Question Answering

**arXiv ID:** 2605.04116 | [PDF](https://arxiv.org/pdf/2605.04116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 167. FL-Sailer: Efficient and Privacy-Preserving Federated Learning for Scalable Single-Cell Epigenetic Data Analysis via Adaptive Sampling

**arXiv ID:** 2605.04519 | [PDF](https://arxiv.org/pdf/2605.04519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 168. CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays

**arXiv ID:** 2605.04509 | [PDF](https://arxiv.org/pdf/2605.04509v1)

**作者:** Gyujin Sim `[一作]` (POSTECH), Sunghyun Cho `[通讯]` (POSTECH)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CoherentRaster框架，在光场显示上实现高分辨率实时渲染；

**💡 创新点**

创新点在于三项：1）跨视角属性复用以消除相邻视角的冗余计算；2）视角一致性重排以恢复warp级内存合并；3）子像素级光栅化直接生成光场图像，省去全帧渲染；

**🔧 技术方法**

使用3D高斯雾化（3D Gaussian Splatting）+子像素级光栅化+视角聚类+视角一致性映射；

**📊 数据集**

Synthetic Blender与Mip-NeRF 360两大数据集，分别用于合成与真实场景的光场渲染；

**📈 对比分析**

与完整帧3DGS、Subpixel‑3DGS、MPI等基线对比，CoherentRaster在4K 71视角上实现约23 FPS，速度比完整帧3DGS提升7.6×，质量与全帧3DGS相当；

**⚠️ 局限性**

局限于视角聚类内局部一致性，易在高频镜面或快速动态场景中出现伪影，仅适用于静态场景；

---

## 169. UAV as Urban Construction Change Monitor: A New Benchmark and Change Captioning Model

**arXiv ID:** 2605.04409 | [PDF](https://arxiv.org/pdf/2605.04409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 170. Temporal Reasoning Is Not the Bottleneck: A Probabilistic Inconsistency Framework for Neuro-Symbolic QA

**arXiv ID:** 2605.04243 | [PDF](https://arxiv.org/pdf/2605.04243v1)

**作者:** Tran Quang Liem `[一作]` `[通讯]` (VNUHCM - HIGH SCHOOL FOR GIFTED), Tran Quang Liem (VNUHCM - HIGH SCHOOL FOR GIFTED)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种神经符号时序问答框架，利用概率不一致信号（PIS）将语义感知与符号推理分离，实现一步级错误定位与结构修复。

**💡 创新点**

创新点在于将符号可信区间与神经网络的经验深度学习推断出的本体不确定性相融合，形成可判定的 PIS，用于检测并纠正结构错误。

**🔧 技术方法**

采用 Evidential Deep Learning 估计神经感知的不确定性，使用 Allen 互补区间代数构建符号可信区间，结合 Monte Carlo Tree Search 和 Blackboard 架构进行决策与修复。

**📊 数据集**

在 Synthetic Temporal-200、TempReason、TimeX-NLI 以及极具挑战性的 TRACIE 数据集上进行评估。

**📈 对比分析**

在零样本设置下与神经-仅、符号-仅、无 PIS 的混合基线对比，完全结构化数据上实现 100% 准确率；半结构化得到 75.1% 以上；在 TRACIE 上约 50%；相较于无 PIS 的混合模型提升约 6.7%。

**⚠️ 局限性**

主要局限在于对初始文本到事件结构的提取质量高度依赖，难以处理高度叙事化、隐式结构的文本；阈值设定固定且缺乏细粒度自适应机制。

---

## 171. Tightly-Coupled Estimation and Guidance for Robust Low-Thrust Rendezvous via Adaptive Homotopy

**arXiv ID:** 2605.04481 | [PDF](https://arxiv.org/pdf/2605.04481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 172. Characterizing Students' LLM Usage Behaviors and Their Association with Learning in Critical Thinking Tasks

**arXiv ID:** 2605.04534 | [PDF](https://arxiv.org/pdf/2605.04534v1)

**作者:** Minju Park `[一作]` (University of British Columbia), Cristina Conati `[通讯]` (University of British Columbia)

**通讯引用:** 8189 | [OpenAlex ID](https://openalex.org/A5024712028)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两学期的研究导向课程中收集学生对LLM使用的自我报告，构建了底层数据驱动的七类使用分类，探究不同使用频率和使用类型与三次期中考试成绩的关联；

**💡 创新点**

首次将LLM使用行为细化为七种支持类型并按学生主动性划分为学生主导、LLM主导和理解支持，揭示使用频率和使用方式对学习表现的非线性影响；

**🔧 技术方法**

采用底层分类编码（基于学生自述）并进行统计描述与组间比较（低/高使用率、学生主导/LLM主导），无需复杂机器学习模型；

**📊 数据集**

使用68名本科生在两学期内提交的论文阅读与批判任务的作业记录、LLM使用报告以及三次期中考试分数；

**📈 对比分析**

通过描述性统计与两两比较，发现未使用LLM者平均成绩高于使用者，且高使用率组低于低使用率组；在期中1阶段差异显著，随后差距收敛，显示使用方式对学习表现影响随课程进度减弱；

**⚠️ 局限性**

样本量有限，子组样本更小，缺乏显著性检验；数据仅来自单一课程，缺乏跨学科验证；自报使用可能低估真实使用率，且缺乏对LLM使用质量的细粒度测评。

---

## 173. Connecting online criminal behavior with machine learning: Using authorship attribution to analyze and link potential online traffickers

**arXiv ID:** 2605.04080 | [PDF](https://arxiv.org/pdf/2605.04080v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 174. Enabling Real-Time Training of a Wildfire-to-Smoke Map with Multilinear Operators

**arXiv ID:** 2605.04164 | [PDF](https://arxiv.org/pdf/2605.04164v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 175. Evaluating Patient Safety Risks in Generative AI: Development and Validation of a FMECA Framework for Generated Clinical Content

**arXiv ID:** 2605.04085 | [PDF](https://arxiv.org/pdf/2605.04085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 176. Material Database Agent: A Multimodal Agentic Framework for Scientific Literature Mining

**arXiv ID:** 2605.04278 | [PDF](https://arxiv.org/pdf/2605.04278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 177. Time series causal discovery with variable lags

**arXiv ID:** 2605.04081 | [PDF](https://arxiv.org/pdf/2605.04081v1)

**作者:** Bruno Petrungaro `[一作]` (Queen Mary University of London), Anthony C. Constantinou `[通讯]` (Queen Mary University of London)

**通讯引用:** 1710 | [OpenAlex ID](https://openalex.org/A5042011218)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于Tabu搜索的时间序列因果结构学习算法，允许每条边学习其特定的时滞并对时滞进行惩罚；

**💡 创新点**

创新点在于：①将每条边的时滞作为可优化参数，②在分解BIC评分中加入节点有效样本量和显式延迟惩罚，③给出时间展开后无环保证和并行实现方案；

**🔧 技术方法**

采用的技术包括：Tabu搜索+增量邻域评估、分解BIC+延迟正则化、GLM/线性回归局部模型、并行邻域评分与稀疏矩阵运算；

**📊 数据集**

实验数据集包括：①多种设置的合成多变量时间序列（N≤24、T≤10,000）；②真实英国COVID‑19日常数据（46变量、861观测，最大时滞6）；

**📈 对比分析**

与传统固定滞后或无时间约束的结构学习方法相比，本方法在F1、SHD、BSF和MAE_lag等指标上表现更好；短滞后情形下恢复率高，样本量增大、AR相关强时性能进一步提升；在COVID数据中得到主导为短滞后、但也有若干长滞后边的结构；

**⚠️ 局限性**

局限性包括：仅使用线性或GLM局部模型，未支持同一时刻边或非线性关系；缺乏对潜在混杂和更复杂缺失值处理的扩展；未来需加入非线性模型、时间序列特定的缺失处理和混杂变量建模。

---

## 178. Anatomy of a failure: When, how, and why deep vision fails in scientific domains

**arXiv ID:** 2605.04231 | [PDF](https://arxiv.org/pdf/2605.04231v1)

**作者:** Ji-Hun Oh `[一作]` (University of Illinois Urbana-Champaign), Rohit Bhargava `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 10899 | [OpenAlex ID](https://openalex.org/A5023245938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

分析深度学习在红外医学成像中的失效，比较红外与 H&E 图像的分类性能，揭示其机制并尝试多种鲁棒化策略。

**💡 创新点**

首次发现光谱信息与 DL 简单性偏差冲突导致红外模型退化为 1D 频谱分析的根本原因，并系统评估了现有鲁棒化方法的局限。

**🔧 技术方法**

使用 ResNet50 基础网络、IRM、V‑REx、DRO、特征去偏、虚拟 H&E 翻译、Grad‑CAM++、SHAP、CKA、ECE 及多种不确定性估计等技术。

**📊 数据集**

基于 51 块前列腺组织样本的 10 通道红外与对应 H&E 图像，采用 70/30 患者级拆分、15 折交叉验证的实验设置。

**📈 对比分析**

与 H&E 基线相比，红外模型在测试集上的准确率下降约 10%，多数鲁棒化方法仅提升 1–3%，而虚拟 H&E 融合后提升约 8%，但仍低于纯 H&E。

**⚠️ 局限性**

局限包括仅针对二分类任务、样本量有限、标签依赖 H&E、未探究多分类或分割等场景，且现有鲁棒化策略无法根本解决红外模型的简单性偏差问题。

---

## 179. The Anatomy of Silent Data Corruption: GPU Error Pattern Study and Modeling Guidance

**arXiv ID:** 2605.04213 | [PDF](https://arxiv.org/pdf/2605.04213v1)

**作者:** Chung-Hsuan Tung `[一作]` (Duke University), Sanjay Gongalore `[通讯]` (NVIDIA)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对一款生产级数据中心GPU进行大规模门级stuck-at故障注入（约三百万模拟器小时），通过63个CUDA微基准评估并提取SDC（Silent Data Corruption）的类型、位翻转分布和warp级空间相关性。

**💡 创新点**

首次系统量化GPU SDC 的真实分布：NaN/±INF仅占1%，单比特翻转不到40%，零化占比50%，并发现多位翻转和warp对齐周期性模式，为高层SDC建模和软件注入提供分布感知参考。

**🔧 技术方法**

使用门级电路仿真结合单stuck-at故障模型、微基准生成器（MT、UTP、LFSR）以及自定义的错误模式提取流程；构建分布感知软件注入模板以复现门级观察到的错误特征。

**📊 数据集**

数据集为63个CUDA微基准（包含加法、乘法、FMA、GEMM等多种运算和数据类型UINT8/UINT32/FP8/FP16/FP32/BF16/TF32），每个基准采用随机或伪随机输入，输出范围从1.8 KB到64 MB。

**📈 对比分析**

与以往基于软件层面随机或单比特翻转注入的方法相比，本研究通过门级仿真得到更逼真的错误分布；实验结果显示约25 k个SDC案例与28 k DUE/挂起事件，揭示了控制逻辑与数据缓存单元对SDC影响的差异。

**⚠️ 局限性**

局限性包括仅使用两SM简化的GPU模型、仅覆盖stuck-at故障（未考虑瞬态或延迟缺陷）、数据类型有限、缺乏对真实生产GPU的直接验证以及未评估多SM级联的影响。

---

## 180. Designing a double deep reinforcement learning selection tool for resilient demand prediction

**arXiv ID:** 2605.04068 | [PDF](https://arxiv.org/pdf/2605.04068v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 181. SWAN: Semantic Watermarking with Abstract Meaning Representation

**arXiv ID:** 2605.04305 | [PDF](https://arxiv.org/pdf/2605.04305v1)

**作者:** Ziping Ye `[一作]` (Amazon), Ninareh Mehrabi `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SWAN，一种基于抽象意义表示（AMR）的文本水印框架，能够在句子生成时将水印嵌入语义结构，并通过提示式LLM生成与AMR匹配的句子；检测时使用AMR解析与一比例z检验实现无训练、鲁棒的水印识别。

**💡 创新点**

创新点在于将水印嵌入AMR图的结构层面，而非传统的词表偏好或句子向量，使得即使句子被同义改写或重写，只要保持语义关系，水印仍能被检测到；并实现了训练‑free、纯提示式的生成与检测流程。

**🔧 技术方法**

技术手段包括：1) 构造AMR模板库并抽象化为占位符；2) 采用LLM（DeepSeek‑R1‑Distill‑Qwen‑14B）进行提示式生成并进行拒绝采样；3) 用S2match计算候选句子AMR与模板的相似度；4) 用amrlib解析器将文本转为AMR；5) 用一比例z检验统计判定文本是否包含水印。

**📊 数据集**

数据集与资源：RealNews（C4子集）用于评估生成与检测效果；MASSIVE‑AMR用于构建AMR模板库；Pegasus、Parrot、Claude 3.7 Sonnet用于生成改写攻击；amrlib的BART‑large AMR解析器用于解析。

**📈 对比分析**

对比方法包括token‑level SynthID‑Text、sentence‑level SemStamp及其改进k‑SemStamp；在未改写文本上SWAN的AUC≈99.1、TPR@1%≈91.6、TPR@5%≈97.6，与SemStamp相当或略逊；在改写（Pegasus、Parrot、Claude）场景下SWAN的AUC≈98.1‑98.3、TPR@1%≈81‑86、TPR@5%≈92‑95，显著优于基线，鲁棒性提升约13.9%点；生成文本质量（连贯性、流畅度、词汇多样性）与基线相当。

**⚠️ 局限性**

局限性包括：1) 依赖AMR解析器的准确性，解析误差会导致召回下降或误报；2) 仅在英文新闻文本上验证，低资源语言或专业领域的适用性不明；3) AMR模板库需保密，若泄露可被绕过；4) 对极端改写（句子拆分/合并）或未来更强的对抗式重写仍可能存在弱点。

---

## 182. dtour: a steerable tour de vis through high-dimensional data

**arXiv ID:** 2605.04306 | [PDF](https://arxiv.org/pdf/2605.04306v1)

**作者:** Fritz Lekschas `[一作]` (Ridge AI), Nezar Abdennur `[通讯]` (UMass Chan Medical School)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种统一的多模式高维可视化界面（dTour），集成了概览、引导、手动和全景（grand）四种投影巡航模式；

**💡 创新点**

通过圆形滑块实现投影路径的平滑、可调节自由度与可控性，允许用户在任意时刻在不同巡航模式之间无缝切换；

**🔧 技术方法**

利用WebGPU/WebGL实现GPU加速渲染、Catmull‑Rom曲线与Gram‑Schmidt正交化的投影插值、主角角距离（Grassmannian）度量、TypeScript/React/Anywidget多平台实现；

**📊 数据集**

在Fashion MNIST、CyTOF单细胞免疫数据、3M arXiv 文本嵌入以及多种嵌入方法的序列投影上进行了验证；

**📈 对比分析**

与传统的单一2D投影、静态散点矩阵、小倍数、手动投影以及现有的grand/tour软件相比，dTour在百万级点集上保持≥60 FPS（≤5 M点），支持平滑播放、即时选择与多模型/多参数比较；

**⚠️ 局限性**

仍需先生成关键帧序列，无法一次性覆盖所有高维方向；在极大数据量或极高维空间中仍受内存和CPU‑GPU通信开销限制；对投影序列的解释性依赖于用户对关键帧选择策略的设计。

---

## 183. LAWS: Learning from Actual Workloads Symbolically -- A Self-Certifying Parametrized Cache Architecture for Neural Inference, Robotics, and Edge Deployment

**arXiv ID:** 2605.04069 | [PDF](https://arxiv.org/pdf/2605.04069v1)

**作者:** Gregory Magarshak `[一作]` `[通讯]`, Gregory Magarshak

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种名为LAWS（从实际工作负载中符号学习）的推理时架构，可以在不修改任何训练神经网络的情况下部署。LAWS维护一个动态增长的参数化专家库，自动从观察到的推理查询中创建计算模式，并对其正确性进行形式化认证。

**💡 创新点**

LAWS的创新点在于自我认证定理：训练网络的权重编码了一个Lipschitz常数Λ(W)，可以在没有额外训练或推理的情况下认证每个专家的有效性。此外，LAWS严格推广了三种先前的方法：KV缓存、专家混合和手动符号AI。

**🔧 技术方法**

使用了Lipschitz常数和概率语言前缀树（PLT）度量来定义路由半径，并通过参数化专家实现了高效的推理。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到LAWS可以在个人电脑、机器人和车辆等设备上部署，这些设备可以根据需要下载所需的专家并从本地工作负载中学习。

**📈 对比分析**

与现有方法相比，LAWS在推理效率和能量消耗上表现出显著优势。LAWS在高命中率下可实现高达10^4倍的能量减少，并且专家库的增长速度为O(2^H log N)，在K个合作单元的情况下，收敛速度比单个单元快Ω(K/log K)。

**⚠️ 局限性**

主要限制在于Lipschitz常数Λ(W)可能对于深层网络来说很大，这会导致有效性半径对于较大错误δ非常小。此外，定理假设工作负载分布是平稳的，而实际工作负载分布会随着时间变化。

---

## 184. Towards Self-Referential Analytic Assessment: A Profile-Based Approach to L2 Writing Evaluation with LLMs

**arXiv ID:** 2605.04298 | [PDF](https://arxiv.org/pdf/2605.04298v1)

**作者:** Stefano Bannò `[一作]` (University of Cambridge), Mark Gales `[通讯]` (University of Cambridge)

**通讯引用:** 15250 | [OpenAlex ID](https://openalex.org/A5050766679)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证一种自我参照式（intra‑learner）分析评分框架，用以评估AES中LLM与单人评分者在识别相对强弱方面的表现

**💡 创新点**

创新点在于将评估焦点从跨学员排名转向同一学习者内部差异，结合Rasch模型校准参考分数并构建正负反馈的双分类任务，提供更具诊断性的评价

**🔧 技术方法**

技术包括两面Rasch模型、Krippendorff’s α筛选、LLM零样本提示、加权平均分数提取、标准化差异计算及F₀.₅二分类评估

**📊 数据集**

数据集为ICNALE GRA，包含140篇L2写作、80名评分者、十个分析维度及整体得分，具有极高的交叉评分密度

**📈 对比分析**

比较方法先用传统rank‑based指标（SRC、QWK）评估整体一致性，再用自我参照框架下的二分类F₀.₅衡量诊断性；实验表明LLM在识别负向弱点上优于单人评分者，而评分者在正向强点识别上更佳，整体平均F₀.₅分别为33.71、28.29、30.85、31.35

**⚠️ 局限性**

局限性：仅使用单一ICNALE GRA数据集，样本量相对有限；仅评估零样本提示策略；ELF背景下的泛化性尚未验证

---

## 185. Deep Reprogramming Distillation for Medical Foundation Models

**arXiv ID:** 2605.04447 | [PDF](https://arxiv.org/pdf/2605.04447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 186. Agent Island: A Saturation- and Contamination-Resistant Benchmark from Multiagent Games

**arXiv ID:** 2605.04312 | [PDF](https://arxiv.org/pdf/2605.04312v1)

**作者:** Connacher Murphy `[一作]` (Stanford University), Connacher Murphy `[通讯]` (Stanford University)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5085865031)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 Agent Island 多人对战模拟环境，让语言模型在合作、冲突与劝说中竞争。

**💡 创新点**

创新点在于构建动态、赢家为王的基准，避免传统基准的饱和与污染，并采用 Bayesian Plackett–Luce 模型评估模型技能。

**🔧 技术方法**

使用了 Bayesian Plackett–Luce 评估、Gibbs 采样、线性回归分析同一提供者偏好以及私聊、公开推销等多轮交互机制。

**📊 数据集**

使用了包含 2000+ 场游戏、24 种不同模型的日志数据集，并公开了 JSON 日志与复现代码。

**📈 对比分析**

通过 Bayesian Plackett–Luce 计算后验技能均值与可信区间，排行榜显示最高模型后验均值远超第二名，并通过两两对比得到显著的技能优势。

**⚠️ 局限性**

局限包括低风险设置导致行为可能不代表高风险场景、未考虑对手匹配效应以及模型可能随对手组合改变技能分布。

---

## 187. Validity-Calibrated Reasoning Distillation

**arXiv ID:** 2605.04078 | [PDF](https://arxiv.org/pdf/2605.04078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 188. Revocation-Ready CP-ABE Key Management for Blockchain-Based IoT Data Sharing

**arXiv ID:** 2605.04280 | [PDF](https://arxiv.org/pdf/2605.04280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 189. Topology-Constrained Quantized nnUNet for Efficient and Anatomically Accurate 3D Tooth Segmentation

**arXiv ID:** 2605.04201 | [PDF](https://arxiv.org/pdf/2605.04201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 190. Towards a Zero-Trust Supply-Chain Assurance Rubric for ORAN RIC Applications

**arXiv ID:** 2605.04249 | [PDF](https://arxiv.org/pdf/2605.04249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 191. Learning-based Statistical Refinement for Denoising

**arXiv ID:** 2605.04332 | [PDF](https://arxiv.org/pdf/2605.04332v1)

**作者:** Rihuan Ke `[一作]` (University of Bristol), Rihuan Ke `[通讯]` (University of Bristol)

**通讯引用:** 161 | [OpenAlex ID](https://openalex.org/A5085427471)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于学习的统计校正方法，用于在不知道噪声分布或干净图像的情况下，对已给出的去噪结果进行改进。

**💡 创新点**

创新点在于：①不需要噪声模型或配对清晰样本，利用噪声统计信息构造辅助信号；②引入一致性判别器 G_ω 与后验网络 R_θ，形成可训练的约束；③通过约束学习实现对任意已有去噪器的无监督提升。

**🔧 技术方法**

技术手段包括：贝叶斯推理与条件像素独立假设、辅助随机信号构造、深度网络实现 G_ω 与 R_θ、基于内积的约束损失、梯度重缩放与惩罚法优化。

**📊 数据集**

使用的数据集：400张 180×180 的训练图像（无干净样本），在 BSD68 与 Set12 上进行测试；噪声类型覆盖高斯、泊松、椒盐、混合泊松-高斯等。

**📈 对比分析**

与多种无监督方法（Noise2Self、LMFS、BM3D 等）以及监督基线比较。实验显示：对高斯噪声时 Refined BM3D 与监督模型差距 <0.5 dB；对泊松、椒盐和混合噪声时提升 0.6–5 dB，显著优于现有无监督方案。

**⚠️ 局限性**

局限性：需要额外训练 G_ω 与 E 两个网络，训练时间约为监督方法的 2.5 倍；方法仅针对条件像素独立噪声，无法直接处理强相关噪声；对辅助信号的选择和超参数设置较敏感。

---

## 192. A cross-modal network for facial expression recognition

**arXiv ID:** 2605.04439 | [PDF](https://arxiv.org/pdf/2605.04439v1)

**作者:** Chunwei Tian `[一作]` (Harbin Institute of Technology), Shichao Zhang `[通讯]` (Guangxi Normal University)

**通讯引用:** 14093 | [OpenAlex ID](https://openalex.org/A5100764178)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出跨模态网络CMNet，融合面部结构信息与生物对称信息，提升面部表情识别效果。

**💡 创新点**

创新点在于三大模块：跨模态增强模块（融合结构与对称信息）、显著面部信息精炼模块（多域注意力提取关键特征）以及半脸对齐优化机制（通过对称损失保证左右半脸一致性），共同提升模型鲁棒性。

**🔧 技术方法**

技术上基于ResNet-18骨干，采用残差学习、通道与空间注意力、分块划分、交叉熵与对称性损失等深度学习手段。

**📊 数据集**

使用多种公开数据集：FER2013、RAF‑DB、AffectNet（7/8 类）、CAER‑S 以及 SFEW 2.0。

**📈 对比分析**

与 33 种公开方法对比，CMNet 在 RAF‑DB 达到 89.11% 最高准确率，在 AffectNet‑8 达到 61.29%，在跨域与情境敏感数据上表现最优，整体性能显著优于 SCN、LAENet‑SA 等主流方法。

**⚠️ 局限性**

局限性主要包括：对大规模对称假设依赖、对动态视频场景验证不足以及模型仍需在更小样本或高噪声环境下进一步提升鲁棒性。

---

## 193. Sequential Strategic Classification with Multi-Stage Selective Classifiers

**arXiv ID:** 2605.04202 | [PDF](https://arxiv.org/pdf/2605.04202v1)

**作者:** Ziyuan Huang `[一作]` (University of Michigan), Mingyan Liu `[通讯]` (University of Michigan)

**通讯引用:** 11831 | [OpenAlex ID](https://openalex.org/A5101967011)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了多阶段战略分类问题，构建了一个包含选择性分类器的顺序级别模型，并推导了激励长期改进（而非作弊）的理论条件与设计原则。

**💡 创新点**

创新点在于：①首次提出多阶段、可弃权的分类器序列，捕捉代理人在不同难度级别间的动态决策；②在改进与欺骗两种行动同时存在的情景下，分析并得到最优即时行动的闭式解；③给出一系列可实现的阈值递增设计，保证在长期内代理人倾向于投入真实努力。

**🔧 技术方法**

主要技术手段包括：将问题建模为连续状态空间的马尔可夫决策过程；利用最优即时决策（myopic policy）求解闭式最佳行动；将即时最优策略映射为可数状态马尔可夫链，分析稳态分布；在此基础上开展大量数值仿真，并使用强化学习算法验证混合与纯策略的效果。

**📊 数据集**

文章未使用公开真实数据集，而是通过模拟生成具有可调参数（如阈值、退化因子、成本、奖励等）的实验数据来验证理论和设计。

**📈 对比分析**

通过对比两种贪婪策略（不改进NI与不作弊NG）在级别分布、平均属性和平均效用三方面的长期指标，仿真结果表明：①NG在所有指标上均优于NI；②在混合策略中，当改进比例介于30%–70%之间时，代理人可获得更高的级别聚集和效用；③决策者的准确率随改进比例增加而提升。

**⚠️ 局限性**

局限性主要包括：①仅分析贪婪即时策略，未研究全局最优或前瞻性策略的行为与影响；②未考虑公平性、社会福利等宏观指标；③模型假设简化（如已知参数、退化比例恒定），在现实应用中需进一步验证与扩展。

---

## 194. Beyond Rigid Geometries: The Spline-Pullback Metric for Universal Diffeomorphic SPD Representation Learning

**arXiv ID:** 2605.04406 | [PDF](https://arxiv.org/pdf/2605.04406v1)

**作者:** Tushar Das `[一作]` (National Institute of Technology Jamshedpur), Koushlendra Kumar Singh `[通讯]` (National Institute of Technology Jamshedpur)

**通讯引用:** 642 | [OpenAlex ID](https://openalex.org/A5014940645)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出一种可学习的 Spline-Pullback Metric（SPM），利用受约束的 B 样条实现 SPD 矩阵空间的全局可微且可逼近任意增函数的差异度量；

**💡 创新点**

创新点在于用 B 样条构造全局微分同胚映射，兼顾单调性、可逆性与局部谱建模，且理论上可逼近任意 C^1 单调映射；

**🔧 技术方法**

采用 B 样条差分同胚、谱/Cholesky 变换、闭式梯度推导与不对称谱扰动来保证数值稳定性和梯度 Lipschitz；

**📊 数据集**

在 HDM05、FPHA 与 Radar 三个真实数据集上进行实验；

**📈 对比分析**

与 AIRM、LE、LC、PCM、ALEM 等传统基线在 Linear Probe、SPDNet（MLR）和深度 RResNet 上对比，SPM 在所有 9 种配置中均取得最高准确率；

**⚠️ 局限性**

局限在于仅针对 SPD 矩阵空间，尚未充分探索初始化与正则化策略，对极小样本或高维情形的鲁棒性仍待验证。

---

## 195. HERCULES: Hardware-Efficient, Robust, Continual Learning Neural Architecture Search

**arXiv ID:** 2605.04103 | [PDF](https://arxiv.org/pdf/2605.04103v1)

**作者:** Matteo Gambella `[一作]` (Politecnico di Milano), Manuel Roveri `[通讯]` (Politecnico di Milano)

**通讯引用:** 4039 | [OpenAlex ID](https://openalex.org/A5035547226)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 HERCULES 的统一 NAS 框架，旨在同时满足硬件效率、鲁棒性和持续学习三大目标。

**💡 创新点**

创新点在于把效率、鲁棒性与持续学习三大轴构建成同等重要的多目标优化，形成十二项挑战（labors），并将动态神经网络作为关键的适应机制，使 NAS 能在资源受限、环境噪声和任务漂移下保持高性能。

**🔧 技术方法**

技术上结合了搜索空间扩展（包括基元、早停、扩展策略等）、多目标进化/强化学习搜索、硬件感知评估（如 LUT、模拟器噪声、能耗/延迟模型）、鲁棒性估计（对抗攻击、IMC 噪声、分布漂移）以及持续学习策略（逐步扩容、重放/正则化），并通过代理模型和 Surrogate 推断加速搜索。

**📊 数据集**

主要使用的公开数据集包括 ImageNet、CIFAR‑10/100、TinyImageNet、MNIST、SVHN、ImageNet‑C、ImageNet‑V2、ImageNet‑A、NICO、P‑MNIST、R‑MNIST、COCO、VOC、TIMIT、LibriSpeech 等，覆盖分类、检测、语音、强化学习、连续学习等多种任务。

**📈 对比分析**

论文通过对现有 NAS 方法的系统性比较，展示了在单一目标（准确率）之外，对效率、鲁棒性与持续学习的综合评估是必要的；但未给出统一实验结果，而是提供了框架设计与多维度性能指标（准确率、能耗、鲁棒分数、平均增量准确率等）作为未来验证的基准。

**⚠️ 局限性**

主要局限包括：搜索成本仍高昂，缺乏真实硬件反馈的闭环；鲁棒性评估往往采用昂贵的对抗攻击或噪声仿真，无法在搜索阶段实时准确；动态 NAS 仍处于概念化阶段，尚未在大规模任务上验证；整体框架需要统一的 benchmark（HERCULES‑Bench）才能实现可复现、可比较的评估。

---

## 196. Constraint-Enhanced Reinforcement Learning Based on Dynamic Decoupled Spherical Radial Squashing

**arXiv ID:** 2605.04185 | [PDF](https://arxiv.org/pdf/2605.04185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 197. A Multi-Agent Consensus Protocol for Stable Software Remodularization

**arXiv ID:** 2605.04188 | [PDF](https://arxiv.org/pdf/2605.04188v1)

**作者:** Ahmed F. Ibrahim `[一作]` (Western University), Ahmed F. Ibrahim `[通讯]` (Western University)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5111720376)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将软件模块聚类视为多代理分布式共识问题，提出并实现了 Asymmetric Monotonic Concession Protocol（AMCP）以在协商中兼顾凝聚度与演化稳定性。

**💡 创新点**

创新点在于把多属性重构转化为代理协商框架，AMCP在正式证明终止性、符合 Zeuthen 交涉策略、并保证局部 Pareto 满足的基础上，引入显式稳定性预算并通过协商动态调整。

**🔧 技术方法**

主要技术包括基于 TurboMQ 与 MoJo 的两种效用函数、AMCP 协商协议（单模块迁移、最小化让步比）、形式化证明与 Python 实现；同时使用邻域剪枝与局部搜索。

**📊 数据集**

实验数据集涵盖合成基准、真实 Java 系统 Xwork 1.0/1.1、Apache Ant、JFreeChart、JUnit、Tomcat、Lucene、Log4j、Spark 核心，以及跨语言 Flask；共计十余个系统。

**📈 对比分析**

通过与 Bunch（单目标遗传算法）和 CC/G（集成聚类）对比，实验显示在松散稳定预算下，AMCP 与最优解相当；在严格预算下，AMCP 能“断路”并严格限制变更，且步数显著减少，整体性能优于传统工具在稳定性约束下的表现。

**⚠️ 局限性**

局限性包括仅考虑单模块迁移导致只能达到局部最优、未评估多步或多代理扩展、缺乏大规模工业代码库验证，以及未来计划的导航启发式与集群级操作尚未实现。

---

## 198. Towards Robust LLM Post-Training: Automatic Failure Management for Reinforcement Fine-Tuning

**arXiv ID:** 2605.04431 | [PDF](https://arxiv.org/pdf/2605.04431v1)

**作者:** Lingzhe Zhang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**通讯引用:** 120511 | [OpenAlex ID](https://openalex.org/A5100391240)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RFT-FaultBench基准并构建了RFT-FM框架，用于自动化RFT过程中的故障检测、诊断与补救

**💡 创新点**

首次系统化细粒度RFT故障基准，证明故障可被观测且可区分，并提出一体化的自动故障管理方法

**🔧 技术方法**

采用基于RFT特征的IVS评分、时序动态故障指纹与基于代理的训练干预三阶段模型

**📊 数据集**

使用RFT-FaultBench（779次训练，16种故障类型，易/难两种设置，包含22,549步和1,457,288条轨迹记录）

**📈 对比分析**

在异常检测上实现87.96%（易）/73.88%（难）F1，故障诊断宏F1分别达到85.51%/42.16%，并实现46.25%异常缓解率，均优于多种基线

**⚠️ 局限性**

对难度更高的细粒度故障仍易遗漏，自动补救效果波动大，且整体修复力度有限

---

## 199. Actionable Real-Time Modeling of Surgical Team Dynamics via Time-Expanded Interaction Graphs

**arXiv ID:** 2605.04169 | [PDF](https://arxiv.org/pdf/2605.04169v1)

**作者:** Vincenzo Marco De Luca `[一作]` (University of Trento), Andrea Passerini `[通讯]` (University of Trento)

**通讯引用:** 4122 | [OpenAlex ID](https://openalex.org/A5066187890)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过构建时空扩展的交互图并结合图神经网络，实现了对手术团队互动动态的实时建模与预测，能够在手术过程中预测手术时间偏差并给出可操作的对策。

**💡 创新点**

创新点主要有：①提出时空扩展图神经网络（Time-Expanded Relational Neural Network），将时间维度直接嵌入图结构，避免小样本数据中过拟合；②设计双层反事实分析框架，既能在拓扑层面给出最小交互调整建议，又能在行为层面给出可解释的语言特征改动方案；③使用多模态可解释行为抽象（eGeMAPS与行为类）与团队角色信息，保持模型可解释性。

**🔧 技术方法**

核心技术包括：多模态特征提取（语音、姿态、人机交互）、时间窗口分割、广播式通信建图、时空扩展图构造、图卷积/图注意力网络、LSTM/MHA时序建模、反事实生成的组合优化与贪婪算法。

**📊 数据集**

使用的主要数据集为MM-OR（多模态模拟膝关节置换手术记录），包含27个手术、4-6名成员，采用多摄像头、环境麦克风，配合人工校正的分离与转录结果。

**📈 对比分析**

与基线方法（MLP、RF、LSTM、MHA、GCN、GAT、MHA+GCN/GAT）进行leave-one-team-out的10次随机种子评估，采用macro-F1指标。结果表明时空扩展GCN在所有配置中取得最高宏F1≈70.1%，相较于最优传统图+时序模型提升约3个百分点，证明该方法在小样本、动态团队建模中的优势。

**⚠️ 局限性**

主要局限包括：①手术时间作为团队绩效的代理指标仍不完全；②反事实建议的实际可行性和临床效用需要专家验证；③模型依赖于高质量的多模态录音与人工校正，部署到真实OR可能面临噪声、语言多样性和设备限制；④目前只评估模拟手术，未覆盖真实复杂手术流程。

---

## 200. Detecting Deepfakes via Hamiltonian Dynamics

**arXiv ID:** 2605.04405 | [PDF](https://arxiv.org/pdf/2605.04405v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 201. How Do Ice Shelves Calve? Peridynamic Modeling of Ice Shelf Fracture Driven by Wave Erosion, Basal Melting, and Buoyancy Flexure

**arXiv ID:** 2605.04365 | [PDF](https://arxiv.org/pdf/2605.04365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 202. Disentangled Learning Improves Implicit Neural Representations for Medical Reconstruction

**arXiv ID:** 2605.04234 | [PDF](https://arxiv.org/pdf/2605.04234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 203. Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI Evaluation

**arXiv ID:** 2605.04135 | [PDF](https://arxiv.org/pdf/2605.04135v1)

**作者:** David Gringras `[一作]` (Harvard University), Misha Salahshoor `[通讯]` (Harvard University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性评估了2022-2026年期间医学、法律、编程、教育和科学推理领域的LLM能力评估论文，量化其与当时技术前沿（frontier）之间的距离、配置报告不足以及结论泛化到“AI”层面的比例。

**💡 创新点**

首次以预注册、跨域的方式将“publication elicitation gap”拆解为时间滞后、同家族层级滞后和配置欠报三维，并提供了对应的可复现检查清单与可视化工具；同时在多尺度能力评估框架（Epoch、Chatbot Arena Elo、Artificial Analysis）下验证结果稳健性。

**🔧 技术方法**

采用OpenAlex检索、基于LLM的两阶段提取管线（V4F与配套补充提示）完成论文筛选与信息抽取；利用预注册的统计方法（Wilcoxon、线性回归、混合效应模型）对距前沿距离、配置披露率、类级声明比例等指标进行分析。

**📊 数据集**

核心数据集为112,303篇与LLM关键词匹配的OpenAlex记录，其中18,574篇符合纳入标准，4,766篇可获取PDF并完成全文抽取；论文涉及GPT、Claude、Gemini、Llama等主要模型，覆盖五大应用领域。

**📈 对比分析**

与前沿模型进行对比时，使用Epoch AI Capabilities Index 作为基准，计算每篇论文的“gap”值；结果显示中位距前沿+10.85（≈1.4倍Claude Sonnet至Opus 4.5差距），每年扩大+5.53；配置披露率仅3.2%，类级声明比例为52.5%，且逐年上升（OR=1.23/年）。

**⚠️ 局限性**

局限性包括：仅测量与前沿的差距，未检验在前沿配置下重新执行的结果；抽取错误及评估日期推测可能引入误差；仅覆盖英文论文和五大领域；未对期刊与会议的出版周期偏差做细致校正；以及对模型命名、版本一致性依赖外部资源。

---

## 204. A Mean Curvature Approach to Boundary Detection: Geometric Insights for Unsupervised Learning

**arXiv ID:** 2605.04274 | [PDF](https://arxiv.org/pdf/2605.04274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 205. Thinking fast and slow -- decision intelligence for power systems

**arXiv ID:** 2605.04228 | [PDF](https://arxiv.org/pdf/2605.04228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 206. Exploring the Output of Software Testing Tools through a Visual Comparative Analysis

**arXiv ID:** 2605.04189 | [PDF](https://arxiv.org/pdf/2605.04189v1)

**作者:** Brandon Lit `[一作]` (University of Waterloo), Thomas Driscoll `[通讯]` (University of Waterloo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对50种软件测试工具的输出进行可视化比较分析。

**💡 创新点**

首次系统性识别并归纳测试工具的共用界面元素与可视化模式。

**🔧 技术方法**

采用可视化对比分析与模式识别技术。

**📊 数据集**

采集了44个CLI工具和6个GUI工具在四种主流编程语言下的输出数据。

**📈 对比分析**

通过人工与可视化方法对输出进行编码与对比，揭示了色彩与布局使用趋势。

**⚠️ 局限性**

仅覆盖50个工具且语言范围有限，缺乏自动化分析与更广泛工具集。

---

## 207. Optimize-at-Capture: Highly-adaptive Exposure Controlling for In-Vehicle Non-contact Heart-rate Monitoring

**arXiv ID:** 2605.04397 | [PDF](https://arxiv.org/pdf/2605.04397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 208. Efficiently Aligning Language Models with Online Natural Language Feedback

**arXiv ID:** 2605.04356 | [PDF](https://arxiv.org/pdf/2605.04356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 209. Quantum-Resistant Networks: A Review of Primitives, Protocols and Best Practices

**arXiv ID:** 2605.04129 | [PDF](https://arxiv.org/pdf/2605.04129v1)

**作者:** Elisa Bertino `[一作]` (Purdue University), Attila A. Yavuz `[通讯]` (University of South Florida)

**通讯引用:** 1359 | [OpenAlex ID](https://openalex.org/A5055359585)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化研究后量子时代网络架构，构建统一的四维分类体系，评估多种关键分发与管理架构，提出最佳实践与未来研究方向。

**💡 创新点**

首次将后量子关键分发架构做系统化梳理；提出统一的加密基础、分发架构、信任模型、生命周期四维分类；对多场景部署进行交叉比较，揭示关键差距与研究空白。

**🔧 技术方法**

综合文献综述与架构分析方法，采用多维度分类法和安全性评估框架；对现有协议（TLS、Kerberos、IPsec等）进行架构映射；使用后量子攻击模型评估安全性；给出基于安全性、可扩展性、可维护性等维度的比较。

**📊 数据集**

该工作为综述性质，不使用实验数据集；主要引用NIST标准化KEM/签名、已发布协议规范以及相关研究成果作为参考。

**📈 对比分析**

通过构建安全性、可扩展性、可用性、成本等评估维度，对比表格形式展示各类架构的优势与局限；未给出具体实验性能数据，主要以理论分析和已有实现的经验为依据。

**⚠️ 局限性**

缺乏实证实验验证；统一威胁模型仍不完善；动态迁移过程的细粒度分析不足；部分建议依赖未来技术（如MPC、QKD）且实现原型缺失；对不同部署场景的可操作性评估仍待进一步研究。

---

## 210. ARMATA: Auto-Regressive Multi-Agent Task Assignment

**arXiv ID:** 2605.04225 | [PDF](https://arxiv.org/pdf/2605.04225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 211. Order Flow Exclusivity and Value Extraction Mechanisms: An Analysis of Ethereum Builder Centralization

**arXiv ID:** 2605.04471 | [PDF](https://arxiv.org/pdf/2605.04471v1)

**作者:** Ao Zhang `[一作]` (Tsinghua University), Yongwei Wu `[通讯]` (Tsinghua University)

**通讯引用:** 41173 | [OpenAlex ID](https://openalex.org/A5100611600)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对以太坊 PBS 架构下的 Builder 市场进行系统性研究，定义并量化 Exclusive Order Flow (EOF) 的排他性，构建监督学习模型识别非原子 MEV 流，并基于时间演化分析 Builder 集中度与收益机制的关系，最终证明 Builder 集中化是 PBS 本身的演化结果。

**💡 创新点**

① 结合 KL 散度和流量加权的排他性指标，首次实现对所有 Order Flow 的客观排他性评估；② 通过人工标注 210 个高收益合约并训练 Random Forest，自动识别 322 条非原子 MEV 流，显著扩大已知的非原子 MEV 规模；③ 将 EOF 与收益机制两维度映射到 Builder 市场的四个历史阶段，揭示“鸡蛋‑蛋”困境的动态解耦。

**🔧 技术方法**

1) KL 散度+流量加权的排他性度量；2) 随机森林监督分类器（随机树 Ensemble）；3) 统计分析：HHI、Pearson 相关、Power‑law 拟合；4) 传统交易特征提取与合约层级聚合特征。

**📊 数据集**

以太坊主网区块 18037988–23264565（2023‑09 至 2025‑08）共 5,226,578 个区块，889,227,817 笔交易；从中提取 152,466,013 条 swap 交易，构成 164,249 条 Order Flow；对前 1,000 名最高 bribe 合约进行人工标注（210 条），并对全部合约进行特征提取。

**📈 对比分析**

EOF 识别通过阈值 F1‑score 最优化，阈值 108.03，F1 = 0.78；监督模型准确率 92.06%，随机森林单棵树平均 91.61%。在全部合约中发现 75 条 EOF（占交易相关收入 70.53%），和 322 条非原子 MEV（占 22.99%）。与之前仅 13 条非原子 MEV 的方法相比，新增 309 条，提升 2380% 的覆盖率。

**⚠️ 局限性**

① 只对前 1,000 名 bribe 合约进行监督学习，无法覆盖低频长尾流；② 依赖人工标注，标注规模有限；③ 仅聚焦以太坊主网，缺乏跨链或其他公链的验证；④ EOF 排他性指标对极低频交易仍存在噪声，可能需要更长时间窗口或更细粒度的检验；⑤ 研究侧重收益与集中度关联，未深入探讨实际经济影响或治理方案。

---

## 212. DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation

**arXiv ID:** 2605.04458 | [PDF](https://arxiv.org/pdf/2605.04458v1)

**作者:** Bryan Li `[一作]` (University of Pennsylvania), Laura Dietz `[通讯]` (University of New Hampshire)

**通讯引用:** 1738 | [OpenAlex ID](https://openalex.org/A5027260515)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种三阶段流水线（生成-聚类-筛选），自动生成高质量、文档根源的 QA nugget 集合，用于跨语言报表评估。

**💡 创新点**

①将 nugget 表示从传统声明式转为 QA 对，解耦信息需求与满足内容；②在聚类阶段专门对问题进行语义匹配，聚合多语言答案；③利用 19 项质量准则训练 SVM 进行 nugget 重要性排序，显著提升评估可靠性。

**🔧 技术方法**

使用 Claude 3.5 Sonnet 生成 QA 对、Llama 3.3 70B 进行答案验证与聚合器推理、BGE‑large‑en‑v1.5 进行问题嵌入、SVM 进行重要性预测，检索阶段采用 PLAID‑X；整体框架依赖 LLM、嵌入模型与传统机器学习。

**📊 数据集**

主要实验基于 TREC 2025 RAGTIME（英文、阿拉伯语、中文、俄语新闻）和 2024 NeuCLIR（中俄法）报表生成任务的 16/59 题目与系统；同时对照 GINGER 基线、人工 nugget 集合与官方手工评测。

**📈 对比分析**

通过与人工评测的 Spearman ρ、Kendall τ、加权 τ 等指标比较，Claude 生成的 nugget 在 RAGTIME 上得到 ρ≈0.90、τ≈0.73 的高相关性；与手工 nugget 的相关性高于 GINGER；使用 Llama 则性能明显下降；对最优系统子集的相关性几乎不变，证明鲁棒性。

**⚠️ 局限性**

局限性包括：①若同一 LLM 同时用于生成与评测，易出现“循环评判”导致偏高分；②对低资源语言和极端文本仍有适配难度；③依赖强大 LLM 与手工设计的质量准则，成本较高；④聚类阈值与 SVM 训练需人工调参；⑤评估仍无法完全替代人工对细粒度语义的把握。

---

## 213. Angle-I2P: Angle-Consistent-Aware Hierarchical Attention for Cross-Modality Outlier Rejection

**arXiv ID:** 2605.04541 | [PDF](https://arxiv.org/pdf/2605.04541v1)

**作者:** Muyao Peng `[一作]` (Huazhong University of Science and Technology), Qiong Liu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 62641 | [OpenAlex ID](https://openalex.org/A5100345153)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文未提供具体内容，无法确定研究目标。

**💡 创新点**

无法识别创新点。

**🔧 技术方法**

无法确定使用的技术。

**📊 数据集**

无法确认使用的数据集。

**📈 对比分析**

无法说明比较方法与性能。

**⚠️ 局限性**

未提及研究限制。

---

## 214. RLearner-LLM: Balancing Logical Grounding and Fluency in Large Language Models via Hybrid Direct Preference Optimization

**arXiv ID:** 2605.04539 | [PDF](https://arxiv.org/pdf/2605.04539v1)

**作者:** Qiming Bao `[一作]` (University of Auckland), Michael J. Witbrock `[通讯]` (University of Auckland)

**通讯引用:** 3535 | [OpenAlex ID](https://openalex.org/A5057995059)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RLearner-LLM 框架，利用 Hybrid-DPO 在知识密集型生成任务中对语言模型进行强化学习对齐。

**💡 创新点**

核心创新是双信号混合奖励（NLI 逻辑一致性 + verifier 流畅度）和自适应奖励组合（H_A/H_M），解决了传统 DPO 的“对齐税”和 verbosity 偏差问题。

**🔧 技术方法**

技术包括 Direct Preference Optimization (DPO)、DeBERTa-v3 NLI 评分、LLM verifier、长度惩罚、ACR 筛选，以及 LoRA 微调和多模态 Gemma 4 的参数映射。

**📊 数据集**

使用 13,211 条学生编写的题目-解释对作为 SFT 语料，评测数据覆盖 5 个学术领域（生物学、医学、法学）共 500 个问答对。

**📈 对比分析**

与 SFT、单信号 DPO、迭代 ILearner‑LLM（K=5）及 GPT‑4o-mini 进行比较。Hybrid‑DPO 在 15 个架构/领域组合中 11 个获得 NLI 提升（最高 6×），在 5 个领域中部分超过迭代 ILearner‑LLM；在对比实验中对自身 SFT 获胜 95%，但对 GPT‑4o‑mini 长输出仍受 verbosity 偏差影响。

**⚠️ 局限性**

局限性包括：在某些领域（如 Qwen3‑Cardiff、Auckland Law、Med‑Y1）提升有限；NLI 评价与训练信号相同，存在循环偏差；使用学生生成的 SFT 数据可能限制逻辑深度与答案覆盖率。

---

## 215. SADE: Symptom-Aware Diagnostic Escalation for LLM-Based Network Troubleshooting

**arXiv ID:** 2605.04530 | [PDF](https://arxiv.org/pdf/2605.04530v1)

**作者:** Kuan-Hao Tseng `[一作]` (University of Sydney), Suranga Seneviratne `[通讯]` (University of Sydney)

**通讯引用:** 2299 | [OpenAlex ID](https://openalex.org/A5038376039)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SADE（Symptom‑Aware Diagnostic Escalation）框架，利用 LLM 代理在网络故障排查中按阶段式方法逐步收集证据并做出诊断；

**💡 创新点**

其创新点在于把 Cisco 的层层排查思路编码为显式政策，区分证据获取与假设决策，并引入症状到故障族映射与技能库，实现了更结构化、更高效的诊断流程；

**🔧 技术方法**

技术上采用 Claude Sonnet 4.6 作为 LLM 核心，结合 Kathara‑Emulator 的 MCP 接口、专门的 Python 辅助脚本、15 个定制的 Claude 技能文件以及一个手工维护的故障族索引；

**📊 数据集**

实验使用公开的 NIKA 基准数据集（共 640 起事故，覆盖 12 种网络场景），并在 523 个保留测试样本上进行评测；

**📈 对比分析**

与 ReAct+GPT‑5 和同一后端的 Claude‑Code 基线按相同的 20 步转限进行对比，SADE 在根因 F1（0.77 vs 0.44/0.55）、检测准确率（0.85 vs 0.68/0.67）和整体评审分数（4.32 vs 3.93/3.80）等指标均显著提升，同时工具调用次数更少、提交成功率更高；

**⚠️ 局限性**

局限性包括：对 NIKA 的故障注入可靠性依赖手工校验，技能库及索引仍需人工维护，推理过程相对耗费 token，且仅在 Kathara 仿真环境下验证，缺乏对真实硬件特性与运营监控的适配。

---

## 216. Velox: Learning Representations of 4D Geometry and Appearance

**arXiv ID:** 2605.04527 | [PDF](https://arxiv.org/pdf/2605.04527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. Distilling Bayesian Belief States into Language Models for Auditable Negotiation

**arXiv ID:** 2605.04507 | [PDF](https://arxiv.org/pdf/2605.04507v1)

**作者:** Zongqi Cui `[一作]` (Emory University), Baihan Lin `[通讯]` (Icahn School of Medicine at Mount Sinai)

**通讯引用:** 929 | [OpenAlex ID](https://openalex.org/A5018612055)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 BOND 框架，先用 LLM 进行 Bayesian 估计得到对手优先级的后验分布，再将后验信息与决策行为通过 8B 学生模型实现可审计的谈判代理。

**💡 创新点**

创新点在于把语言模型当作似然估计器，将 Bayesian 推理嵌入对话决策，并通过蒸馏把后验分布以可检验的文本标签形式输出，从而实现模型行为的透明审计。

**🔧 技术方法**

技术主要包括：LoRA 微调的 Llama-3.1-8B 作为似然评分器；离散 Bayesian 更新；菜单规划器根据后验期望对手效用计算最佳提议；以及 8B 学生模型的标签化输出蒸馏。

**📊 数据集**

使用 CaSiNo 露营场景谈判数据集，评估对手的六种优先级顺序，包含 150 条测试对话共 1054 个对话轮次。

**📈 对比分析**

与 70B 结构化 Chain-of-Thought 基线对比：学生模型的 Brier 分数为 0.114，低于教师的 0.085 和基线的 0.194；接受率 F1 分别为 0.908（学生）/0.897（教师）/0.947（基线）。总体来看，学生在后验校准与可审计性方面优于基线，决策质量略逊。

**⚠️ 局限性**

主要限制在于蒸馏后学生模型对后验的使用与决策的因果耦合较弱，后验修正对行为影响有限；需进一步设计约束解码或附加规划损失以强化后验驱动决策。

---

## 218. A Hybrid Method for Low-Resource Named Entity Recognition

**arXiv ID:** 2605.04489 | [PDF](https://arxiv.org/pdf/2605.04489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 219. CAR: Query-Guided Confidence-Aware Reranking for Retrieval-Augmented Generation

**arXiv ID:** 2605.04495 | [PDF](https://arxiv.org/pdf/2605.04495v1)

**作者:** Zhipeng Song `[一作]` (Dalian University of Technology), Heng Qi `[通讯]` (Dalian University of Technology)

**通讯引用:** 143223 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对检索增强生成（RAG）中的文档重排序问题，提出了 CAR（Confidence-Aware Reranking）框架，利用生成器采样的一致性作为置信度指标，对候选文档进行后验校正，实现查询导向、无训练、插件式的重排序。

**💡 创新点**

创新点包括：① 以生成器的采样一致性捕捉文档对生成器不确定性的影响；② 引入查询阈值 QT 和置信度边际 CM，控制何时及如何修正排名；③ 采用稳定分箱排序，在保持基准相对顺序的前提下对文档进行提升/降级；④ 完全不需要对检索器、重排器或生成器进行任何训练，兼容多种检索/生成模型。

**🔧 技术方法**

技术手段包括：多次生成采样、语义聚类（双向蕴含判定）来估计置信度；后验分箱重排序；在 BEIR 基准上使用 Qwen2.5‑7B‑Instruct、Llama‑3‑8B‑Instruct、GLM‑4‑9B‑Chat、InternLM2.5‑Chat‑7B 等 LLM；评估指标采用 NDCG@5 与 F1。

**📊 数据集**

使用了四个 BEIR 数据集：NQ、FEVER、SCIDOCS、TREC‑COVID。

**📈 对比分析**

与 BM25、Contriever 检索、LLM 重排器（YesNo、QLM、RankGPT）以及监督式重排器（ColBERT、Cross‑Encoder、RankT5）进行对比。CAR 在所有基线上均无性能下降，弱重排器（如 YesNo）提升显著（约 +25%），强重排器提升微小但稳定。不同检索器、不同 LLM 基础上均表现一致；与下游生成质量 F1 的 Spearman ρ 达 0.964，说明排名提升可有效转化为生成质量提升。

**⚠️ 局限性**

局限性包括：① 需要多次采样和双向蕴含判断，导致额外推理成本；② 依赖查询阈值 QT 与置信度边际 CM 两个超参数，需在验证集上调优，可能对某些场景不够自适应；③ 置信度估计基于语义一致性，可能在领域特定、低资源语言或多模态场景中效果欠佳；④ 主要验证文本检索/生成，未探讨多模态、代码检索等更复杂的 RAG 场景。

---

## 220. Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding in Gaussian Splatting

**arXiv ID:** 2605.04506 | [PDF](https://arxiv.org/pdf/2605.04506v1)

**作者:** Binh Long Nguyen `[一作]` (Queensland University of Technology), Peyman Moghadam `[通讯]` (Queensland University of Technology)

**通讯引用:** 2035 | [OpenAlex ID](https://openalex.org/A5008586469)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了基于高斯喷射（Gaussian Splatting）的实例级开放词汇3D场景理解框架 Ilov3Splat，能够通过自然语言查询精准定位和分割任意对象。

**💡 创新点**

创新点在于：①联合优化几何与语义，使用多分辨率哈希嵌入（MHE）高效编码语言对齐特征；②引入实例判别特征场并通过SAM掩膜对比学习实现实例级语义；③采用两阶段3D聚类（语义+空间）实现高质量实例分组。

**🔧 技术方法**

使用技术包括 Gaussian Splatting、CLIP 语言编码、DINO 边界正则化、SAM 模式掩膜、对比学习、哈希编码、HDBSCAN/DBSCAN 聚类、轻量级 MLP 投影。

**📊 数据集**

主要数据集：LERF（用于开放词汇3D物体检索）和 ScanNet（用于无类别3D实例分割）。

**📈 对比分析**

与 OpenGaussian、LangSplat、LEGaussians 等基线相比，Ilov3Splat 在 LERF 的 mAcc 及 mIoU 均显著提升，尤其在“tea”与“ramen”等场景中达到最优；在 ScanNet 的 mAcc 也突破同类方法，mIoU 亦保持竞争力。

**⚠️ 局限性**

局限性主要体现在稀疏场景和大物体上，多视角掩膜监督不足时可能导致实例表示不一致。

---

## 221. CCL-D: A High-Precision Diagnostic System for Slow and Hang Anomalies in Large-Scale Model Training

**arXiv ID:** 2605.04478 | [PDF](https://arxiv.org/pdf/2605.04478v1)

**作者:** Yida Gu `[一作]` (University of Chinese Academy of Sciences), Dingwen Tao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 2808 | [OpenAlex ID](https://openalex.org/A5063703614)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

本文提出一种名为CCL-D的高精度诊断系统，用于实时检测和定位大规模模型训练中的慢/挂通信异常。

**💡 创新点**

创新点在于结合跨层级Send/Recv计数与速率指标、轻量分布式追踪以及基于CPU的零拷贝测量，显著提升诊断精度与效率。

**🔧 技术方法**

技术实现包括跨层度量、分布式追踪框架、Trace ID/Probing Frame结构、CPU驱动度量、决策分析器等。

**📊 数据集**

实验使用Llama2‑7B、Llama3.1‑8B、BaiLing‑5B、BaiLing‑80B等大模型，部署在4,000 GPU集群。

**📈 对比分析**

与Bisection、Stack、RAS、Greyhound、C4D等基线比较，CCL‑D在检测/定位时间分别为5‑6分钟/1分钟，覆盖率高，诊断准确率近100%，整体开销低于0.5%。

**⚠️ 局限性**

局限性在于仅针对CCL层面，需要集成NCCL/RCCl修改，且在极短暂或非通信相关的故障上表现有限。

---

## 222. KEET: Explaining Performance of GPU Kernels Using LLM Agents

**arXiv ID:** 2605.04467 | [PDF](https://arxiv.org/pdf/2605.04467v1)

**作者:** Joshua H. Davis `[一作]` (University of Maryland), Abhinav Bhatele `[通讯]` (University of Maryland)

**通讯引用:** 4054 | [OpenAlex ID](https://openalex.org/A5081506338)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建了一套基于大型语言模型的代理框架，自动解析 Nsight Compute 生成的 GPU 核心性能剖面，生成自然语言报告并给出针对性优化建议。

**💡 创新点**

创新点在于将 LLM 与多阶段代理流程（代码分析→剖面选择→指标挑选→剖面分析→聚合与复核）结合，能够自适应地选取最有信息量的指标与剖面，并通过可解释的中间结果实现对性能瓶颈的系统性识别和定制化优化。

**🔧 技术方法**

采用了 GPT‑OSS‑120B 及 GPT‑5.1 等 LLM、Nsight Compute 的 Python Report Interface 读取剖面数据、可选的 DrGPU 规则引擎，并在多轮交互中使用专门的 Prompt 模板实现代理角色的协作。

**📊 数据集**

使用 Rodinia、LULESH、XSBench 等基准集合中的 15 个 CUDA 核心（共计 2,727 行代码）以及多种 GPU 架构（H100、V100、A100）和多种调参组合（块尺寸、寄存器数、网格类型）生成的剖面数据集。

**📈 对比分析**

与 Code‑Only、Code+Data、DrGPU‑Only、LLM+DrGPU 等基线方法在多选题（MCQ）和代码优化（OPT）两项下进行对比，结果显示所提出的框架在 MCQ 上平均得分提升至 82%（高于 79%），在 OPT 上平均 speedup@1 达到 1.4‑1.5×，并在多核优化中实现 14× 的速度提升，证明其在理解剖面数据和生成有效优化建议方面优于传统规则或单一 LLM 方案。

**⚠️ 局限性**

局限性包括对 LLM 质量的依赖，较大模型推理成本，若剖面数据不足或不完整可能导致报告失真；当前仅支持 NVIDIA Nsight Compute，且对新架构的适配仍需通过更新 LLM 知识库；在复杂的并行调参场景下，建议的块/网格尺寸仍可能出现误差。

---

## 223. DALight-3D: A Lightweight 3D U-Net for Brain Tumor Segmentation from Multi-Modal MRI

**arXiv ID:** 2605.04518 | [PDF](https://arxiv.org/pdf/2605.04518v1)

**作者:** Nand Kumar Mishra `[一作]` (Dr. Bhimrao Ambedkar University), Dr Manu Pratap Singh `[通讯]` (Dr. Bhimrao Ambedkar University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级3D脑肿瘤分割框架DALight-3D。

**💡 创新点**

创新点在于将深度可分离卷积、扫描仪感知归一化、跨切片注意力和自适应跳连融合四项模块集成到U-Net结构中。

**🔧 技术方法**

采用深度可分离3D卷积、ScannerAwareNorm、Cross‑Slice Attention、SSFB等技术。

**📊 数据集**

使用Medical Segmentation Decathlon Task01_BrainTumour（多模态T1、T1ce、T2、FLAIR）数据集。

**📈 对比分析**

与标准3D U‑Net、Attention U‑Net、Residual 3D U‑Net和V‑Net在同一训练配置下比较，DALight‑3D在2.22M参数下获得0.727平均Dice，性能优于所有基线。

**⚠️ 局限性**

局限在于仅在单一基准数据集验证，扫描仪归一化使用代理标识符，未评估跨数据集泛化，且实验未多次重复。

---

## 224. HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks

**arXiv ID:** 2605.04525 | [PDF](https://arxiv.org/pdf/2605.04525v1)

**作者:** Nandiraju Gireesh `[一作]` (Peking University), He Wang `[通讯]` (Peking University)

**通讯引用:** 296781 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种层次化的长时程规划框架HDFlow，结合扩散模型和纠正流模型完成机器人长距离任务规划

**💡 创新点**

创新点在于将高层利用扩散模型生成稀疏子目标并通过能量模型(EBM)和流形投影实现精确引导，同时低层采用纠正流模型快速生成稠密轨迹；并在世界模型中加入对比学习与逆动力学，构建结构化潜在空间

**🔧 技术方法**

使用扩散模型、能量模型(EBM)、纠正流模型、对比学习、逆动力学、MPC框架以及流形投影技术

**📊 数据集**

在四个家具装配任务（FurnitureBench）、RLBench 18 任务、OGBench 视觉任务及真实 Franka R3 机器人上进行评估

**📈 对比分析**

与传统模仿学习、单一扩散规划器以及其他层次化扩散模型对比，HDFlow 在所有任务中均取得最高成功率，且低层纠正流模型实现了显著的推理速度提升

**⚠️ 局限性**

依赖成功与失败演示数据集，数据收集成本高；对流形投影的近似与高维潜在空间误差仍是潜在瓶颈

---

## 225. High-Fidelity Single-Image Head Modeling with Industry-Grade Topology

**arXiv ID:** 2605.04524 | [PDF](https://arxiv.org/pdf/2605.04524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 226. SpecPL: Disentangling Spectral Granularity for Prompt Learning

**arXiv ID:** 2605.04504 | [PDF](https://arxiv.org/pdf/2605.04504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 227. Example-Based Object Detection

**arXiv ID:** 2605.04501 | [PDF](https://arxiv.org/pdf/2605.04501v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 228. Stream-T1: Test-Time Scaling for Streaming Video Generation

**arXiv ID:** 2605.04461 | [PDF](https://arxiv.org/pdf/2605.04461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 229. Harnessing Linguistic Dissimilarity for Language Generalization on Unseen Low-Resource Varieties

**arXiv ID:** 2605.04500 | [PDF](https://arxiv.org/pdf/2605.04500v1)

**作者:** Jinju Kim `[一作]` (Sungkyunkwan University), David R. Mortensen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2310 | [OpenAlex ID](https://openalex.org/A5059859009)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个两阶段语言泛化框架，先通过无标注无并行数据的 TOPPing 方法挑选高资源源方言，再使用轻量级 VAÇAÍ-Bowl 双分支架构学习方言特定与通用特征，显著提升零样本下低资源方言的句法任务表现。

**💡 创新点**

创新点在于：①不依赖预先标注或并行语料的自动源方言选择（TOPPing）①使用词重叠与嵌入距离两种独立指标保留方言多样性；②在 VAÇAÍ-Bowl 中同时捕捉方言不变特征与方言特定特征，并通过对抗训练区分两者，避免传统对齐方法导致的“对齐诱发失败”。

**🔧 技术方法**

技术包括：基于多语言预训练模型（mBERT/XLM‑R）的 CLS 表示、Token‑Overlap 与嵌入距离的方言相似度评估、轻量级双分支 MLP 编码器、梯度反转对抗训练、CKA 与 t‑SNE 等可视化分析。

**📊 数据集**

使用 DialectBench 中十个无训练数据的低资源方言，源方言从 DialectBench 与 Universal Dependencies 选取；评估任务为依存句法解析（UAS/LAS）和词性标注（F1）。

**📈 对比分析**

与基线（单纯微调、对齐训练）和 LangRank 传统源选方法对比，TOPPing+VAÇAÍ-Bowl 在 mBERT 上平均提升 UAS 约 50.6%，XLM‑R 上 58.6%；在大多数方言中均击败对齐基线，并在对齐导致性能下降的案例中实现恢复。

**⚠️ 局限性**

局限性包括：需为每个目标方言单独挑选源方言，增加计算开销；方法仍依赖预训练模型对高资源方言的良好表示；在极低资源或极不相关方言的源方言覆盖不足时可能效果受限。

---

## 230. Stabilizing LLM Supervised Fine-Tuning via Explicit Distributional Control

**arXiv ID:** 2605.04468 | [PDF](https://arxiv.org/pdf/2605.04468v1)

**作者:** Xinyu Wang `[一作]` (East China Normal University), Xiaoling Wang `[通讯]` (East China Normal University)

**通讯引用:** 11627 | [OpenAlex ID](https://openalex.org/A5100344619)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“Anchored Learning”框架，利用动态移动的锚点在离线微调过程中显式控制模型分布的漂移，并通过蒸馏方式逼近该锚点。

**💡 创新点**

创新点在于：①将锚点定义为当前模型与冻结的SFT模型的插值，实现动态、局部的分布更新；②提供每步KL距离上界的理论保证，等价于分布空间的信任域约束；③在纯离线设置下兼具RL的稳定性与SFT的高效性。

**🔧 技术方法**

核心技术包括：概率/对数空间插值、基于蒸馏的目标逼近、KL散度分析与上界证明、两级（外循环+内循环）训练框架。

**📊 数据集**

实验使用的任务数据集包括：iGSM、MedCalc、IFEval；评估一般能力的基准包含：MMLU-Pro、Countdown、MBPP、HumanEval 及其扩展。

**📈 对比分析**

与标准SFT及多种减记忆衰减基线（Low‑SFT、KL‑SFT、Self‑SFT、Iter‑SFT、STM、DFT）进行对比，Anchored Learning 在保持近乎最优的目标任务表现（如 iGSM 75.2%）的同时，显著降低灾难性遗忘（从 >53% 降至 <5%），整体位于性能‑稳定性 Pareto 前沿。

**⚠️ 局限性**

主要局限是额外的计算开销：每轮需构造锚点并执行一次前向传播蒸馏，训练延迟与显存约提升 1.5–2 倍，可能不适合极度资源受限环境。

---

## 231. Gradient Scaling Effects in Adaptive Spectral PINNs for Stiff Nonlinear ODEs

**arXiv ID:** 2605.04502 | [PDF](https://arxiv.org/pdf/2605.04502v1)

**作者:** Isabela M. Yepes `[一作]` (Harvard University), Pavlos Protopapas `[通讯]` (Harvard University)

**通讯引用:** 5525 | [OpenAlex ID](https://openalex.org/A5011315981)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在弹性杆摆（spring‑pendulum）等刚性非线性 ODE 中，初始条件（IC）嵌入函数对 Physics‑Informed Neural Networks（PINNs）优化条件的影响，比较了指数门和线性门在固定与自适应傅里叶谱模型中的表现。

**💡 创新点**

创新点在于揭示 IC 门的时间依赖梯度缩放对 NTK 产生显著影响，从而改变优化收敛性，并通过实验验证不同门函数在不同刚度下的性能切换。

**🔧 技术方法**

采用了自适应 Fourier 特征网络（adaptive spectral PINN）、Neural Tangent Kernel（NTK）分析、Adam 优化器、Wilcoxon 符号秩检验与 Holm 校正等技术。

**📊 数据集**

使用的“数据集”是基于弹性杆摆 ODE 的高精度数值解（通过 DOP853 产生的参考轨迹）作为对比基准。

**📈 对比分析**

通过对比指数门和线性门在不同刚度 k（20、30、50、60 及更大）下的相对 L^2 误差和最大点误差，配合 20 个随机种子并进行配对 Wilcoxon 检验，发现指数门在中等刚度时更优，而线性门在高刚度时表现更好。

**⚠️ 局限性**

局限性包括只考虑单一 ODE 系统、仅测试两种门函数、固定训练设置以及未探究更复杂 PDE 或更广泛的谱结构。

---

## 232. An Evaluation of Chat Safety Moderations in Roblox

**arXiv ID:** 2605.04491 | [PDF](https://arxiv.org/pdf/2605.04491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 233. Pen-Strategist: A Reasoning Framework for Penetration Testing Strategy Formation and Analysis

**arXiv ID:** 2605.04499 | [PDF](https://arxiv.org/pdf/2605.04499v1)

**作者:** Yasod Ginige `[一作]` (University of Sydney), Suranga Seneviratne `[通讯]` (University of Sydney)

**通讯引用:** 2299 | [OpenAlex ID](https://openalex.org/A5038376039)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个针对渗透测试的推理数据集，并训练了两模型（策略生成模型和步骤/工具分类模型）以实现本地化、可解释且高效的渗透测试流程。

**💡 创新点**

创新点在于：① 将逻辑推理与策略生成结合，利用强化学习（GRPO）显著提升模型的策略质量；② 引入专门的步骤与MCP工具分类器，减少工具错误与执行失败；③ 通过本地化部署实现数据隐私与低成本高效运行。

**🔧 技术方法**

技术方法包括：Qwen‑3‑14B 大语言模型+LoRA+GRPO 强化学习；双头 CNN 语义分类器；低成本本地推理与工具桥接（MCP）。

**📊 数据集**

数据集来自 240 台 Hack‑The‑Box 与 VulnHub 机器，包含 2,165 条推理样本（手工与自动标注），覆盖策略、步骤、工具与结果。

**📈 对比分析**

在策略推理上相较基线提升 87%（GEval 0.73 vs 0.39），在步骤/工具分类上准确率 82.9% 远超 GPT‑5 等商用模型；在集成至 PentestGPT、AutoPentester、VulnBot 时，子任务完成率提升 47.5%–52.5%；在 CTFKnow 与 PicoCTF 上亦显著提升。

**⚠️ 局限性**

局限性：仍受模型规模限制，部分复杂场景下策略选择仍不理想；仅覆盖 240 台机器，数据规模有限；需要进一步扩展工具集合与多模型协同，以提升跨环境的泛化能力。

---

## 234. Quadrature-TreeSHAP: Depth-Independent TreeSHAP and Shapley Interactions

**arXiv ID:** 2605.04497 | [PDF](https://arxiv.org/pdf/2605.04497v1)

**作者:** Ron Wettenstein `[一作]` (Reichman University), Peng Yu `[通讯]` (Shopify)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于Gauss–Legendre积分的树模型Shapley值与交互值计算方法，显著提高了树深度无关的数值稳定性和计算效率。

**💡 创新点**

创新点在于将Shapley值表述为加权Banzhaf多项式，通过固定点Gauss–Legendre积分实现高阶交互的精确、低阶点计算，并消除对树深度的依赖。

**🔧 技术方法**

采用加权Banzhaf多项式、Gauss–Legendre积分、深度优先遍历、SIMD/GPU向量化和固定点计算来实现高效的树解释。

**📊 数据集**

在12个XGBoost基准数据集（Adult、CalHousing、CovType、Fashion‑MNIST）上进行评估，涵盖小/大/稀疏三种模型配置。

**📈 对比分析**

与原TreeSHAP、GPUTreeSHAP和TreeSHAP‑IQ比较，CPU上实现1.06×–10.59×加速，GPU上实现1.84×–6.95×加速；在高阶交互时可达1209×的速度提升。

**⚠️ 局限性**

局限在于实验仅针对XGBoost决策树，未检验极高维特征或极深树的极限情况；对非XGBoost树模型或多分类场景的适用性尚未验证。

---

## 235. Towards General Preference Alignment: Diffusion Models at Nash Equilibrium

**arXiv ID:** 2605.04494 | [PDF](https://arxiv.org/pdf/2605.04494v1)

**作者:** Jiaming Hu `[一作]` (Boston University), Ioannis Ch. Paschalidis `[通讯]` (Boston University)

**通讯引用:** 6659 | [OpenAlex ID](https://openalex.org/A5075696701)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Diffusion Nash Preference Optimization（DNPO），一种基于 NASH 学习的自我对弈式文本到图像扩散模型对齐方法。

**💡 创新点**

创新点在于直接优化偏好概率的相对优势，摆脱 Bradley–Terry 假设，统一了参考策略与前一策略正则化，并通过对数逻辑损失实现统一框架。

**🔧 技术方法**

技术方法包括自我对弈的在线镜像下降、对数逻辑对数损失、Gaussian 逆过程重参数化以及多模型自动偏好投票。

**📊 数据集**

使用了 Pick‑a‑Pic v1 偏好数据进行训练，并在 Pick‑a‑Pic、Parti‑Prompts、HPSV2 等数据集上进行评估。

**📈 对比分析**

与在线 SFT、Diffusion‑DPO、SPIN、SEPPO 等基线比较，DNPO 在 SD1.5 与 SDXL 上平均赢率和自动评价指标均显著提升，特别是 SDXL 上提升约 7%–10%。

**⚠️ 局限性**

局限性包括仅依赖五个评分模型的平均排名来构造偏好，缺乏更丰富的偏好信号，并且在线采样效率与多样性待进一步改进。

---

## 236. How Does Thinking Mode Change LLM Moral Judgments? A Controlled Instant-vs-Thinking Comparison Across Five Frontier Models

**arXiv ID:** 2605.04488 | [PDF](https://arxiv.org/pdf/2605.04488v1)

**作者:** Sai Sourabh Madur `[一作]` `[通讯]`, Sai Sourabh Madur

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较五大前沿推理型LLM在同一模型检查点下，开启“思考模式”是否会改变其道德判断与自我标记的伦理框架；

**💡 创新点**

首次在单一权重、不同思考开关下进行对比实验，揭示思考模式对跨模型一致性与人口敏感度的潜在影响；

**🔧 技术方法**

采用链式推理（Chain‑of‑Thought）启用、API调用与结构化JSON提示，结合Krippendorff α、Cohen κ、配对Wilcoxon检验等统计工具；

**📊 数据集**

构建了100个情景的多类别基准（包括典型的电车难题、道德基础理论、同义句对、人口敏感度与当代议题），并公开了所有原始API响应；

**📈 对比分析**

在同一模型权重下切换思考模式，比较即时模式与思考模式的二元判决一致性、框架分布、同义句一致率、人口不一致系数，发现整体跨模型α从0.78升至0.79，难题集上α从0.08提升至0.23，但差异仍属统计不显著；

**⚠️ 局限性**

限制包括思考模式的计算量在各供应商之间差异巨大、样本量仅为3、基准仅涵盖英语西方场景、模型训练数据无法观测，且难题集可能已被模型记忆，导致结果受“记忆化”影响。

---

## 237. Data-dependent Exploration for Online Reinforcement Learning from Human Feedback

**arXiv ID:** 2605.04477 | [PDF](https://arxiv.org/pdf/2605.04477v1)

**作者:** Zhen-Yu Zhang `[一作]` (RIKEN), Masashi Sugiyama `[通讯]` (RIKEN)

**通讯引用:** 22528 | [OpenAlex ID](https://openalex.org/A5072744508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DEPO——一种基于历史比较数据构建探索奖励的在线人类反馈强化学习方法

**💡 创新点**

创新点在于利用历史偏好对比直接计算不确定性奖励，避免传统基于当前策略的期望难以估计的问题

**🔧 技术方法**

采用表示空间中的上置信度探索机制、偏好优化目标与理论 regret 上下界分析

**📊 数据集**

在常用的RLHF基准数据集（如OpenAI Feedback、Alpaca等）上进行实验

**📈 对比分析**

与现有在线RLHF基线（DPO、PPO‑HF等）对比，DEPO在多项评测中均表现出更高的样本效率和更快的收敛速度

**⚠️ 局限性**

局限性在于对历史数据覆盖度依赖较大，若历史样本极度稀疏或分布不均，探索奖励可能失效

---

## 238. Automated Formal Proofs of Combinatorial Identities via Wilf-Zeilberger Guidance and LLMs

**arXiv ID:** 2605.04472 | [PDF](https://arxiv.org/pdf/2605.04472v1)

**作者:** Beibei Xiong `[一作]` (East China Normal University), Lihong Zhi `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 1325 | [OpenAlex ID](https://openalex.org/A5026679766)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为WZ-LLM的神经-符号框架，用于在Lean 4中自动化证明组合恒等式

**💡 创新点**

将Wilf–Zeilberger（WZ）方法的证明计划转化为可执行的Lean证明草图，并训练专门的LLM推理器（WZ-Prover）来完成子目标；同时通过Lean核验证循环与强化学习提升推理器性能

**🔧 技术方法**

Wilf–Zeilberger算法、SageMath等符号计算、大型语言模型（LLM）、动态采样策略优化（DAPO）、专家迭代验证与数据扩增

**📊 数据集**

307条手工形式化的组合恒等式（Seed集），随后从两本经典教材中提取的1020条候选恒等式，构成训练与验证集；L​CI‑Test（100条经典恒等式）为主基准，外部基准包括CombiBench（100条）与PutnamBench‑Comb（36条）

**📈 对比分析**

在pass@32（每个问题最多32次解码）下与DeepSeek‑V3、MA‑LoT、Kimina‑Prover‑Distill、DeepSeek‑Prover‑V2、Goedel‑Prover‑V2等基线对比；WZ‑LLM在L​CI‑Test上实现34/100（比基线高约3–4倍），在CombiBench上提升至16/100，PutnamBench‑Comb提升至3/36，展示了两条推理路径（符号分解+子目标证明与直接证明）的互补优势

**⚠️ 局限性**

对WZ方法的适用性有限，未覆盖所有组合恒等式；LLM推理器在极长的证明链或复杂边界条件时仍可能失败；实验受限于固定的pass@32预算，无法完全体现推理效率与成本

---

## 239. CRAFT: Counterfactual-to-Interactive Reinforcement Fine-Tuning for Driving Policies

**arXiv ID:** 2605.04470 | [PDF](https://arxiv.org/pdf/2605.04470v1)

**作者:** Keyu Chen `[一作]` (Tsinghua University), Sifa Zheng `[通讯]` (Tsinghua University)

**通讯引用:** 930 | [OpenAlex ID](https://openalex.org/A5036282981)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CRAFT 框架，利用密集反事实代理与基于交互的残差校正对预训练的驾驶策略进行闭环微调。

**💡 创新点**

创新性地将闭环 RL 梯度拆解为代理与残差两项，在相同状态分布上使用反事实优势作为密集代理，并通过交互关键事件提供残差校正，辅以 EMA 教师的自蒸馏稳定训练。

**🔧 技术方法**

采用代理–残差梯度分解、组归一化反事实优势、双重裁剪残差更新、异向 KL 自蒸馏、EMA 教师，以及在 CARLA Bench2Drive 上的 on‑policy RL 训练技术。

**📊 数据集**

使用 Bench2Drive 及其扩展的 LEAD、Longest6 V2 场景，全部基于 CARLA 仿真环境。

**📈 对比分析**

与 PPO、REINFORCE++、GRPO 三个基线在相同协议下比较，CRAFT 在三种策略架构上均实现最大的闭环性能提升，尤其在安全关键指标（VRU 交互、障碍规避、交叉口转向）上显著优于基线。

**⚠️ 局限性**

在跨场景迁移时对路线完成率提升有限，且对未知交互分布的鲁棒性仍受限；极少样本的稀疏事件仍需要更细粒度的残差估计。

---

## 240. Robust Inverse Quadratic Error Decay with Meshing and Beam Search for Random Subset Sum

**arXiv ID:** 2605.04465 | [PDF](https://arxiv.org/pdf/2605.04465v1)

**作者:** Edwin Chen `[一作]` (Portland State University), Christof Teuscher `[通讯]` (Portland State University)

**通讯引用:** 27617 | [OpenAlex ID](https://openalex.org/A5070808670)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究随机子集和问题，提出一种基于 MITM 的 beam search 算法，构造 O(B/w) 网格并实现 O(n w log w) 时间的近似解。

**💡 创新点**

证明在概率 1‑δ 下仅需 O(log w) 步即可构造网格；通过分阶段 beam search 实现期望误差 O(B/(n w²))；将网格思想与期望误差框架结合，给出可实现的逆二次误差衰减。

**🔧 技术方法**

使用区间分桶与裁剪、均值场假设、离散逻辑递推、锚点与多目标 beam、检查点回溯、理论分析与实验验证等技术。

**📊 数据集**

采用多种 i.i.d. 分布（均匀、双峰、正态、对数正态、Student t、Cauchy）以及随机实例，模拟不同尾部目标。

**📈 对比分析**

与传统启发式（GA、SA、PSO、AOA、Tabu、FPTAS）对比，误差‑时间曲线表明本算法误差下降快、常数小，超过对手多数量级；实验显示误差符合 O(B/(n w²)) 并对分布鲁棒。

**⚠️ 局限性**

依赖均值场与 i.i.d. 假设，未给出下界；对极端尾部目标时 Phase B 可能被耗尽；对非对称或相关分布未证明；实现中需手动分割阶段，理论上对大 n 依赖。

---

## 241. Discovering Sparse Counterfactual Factors via Latent Adjustment for Survey-based Community Intervention

**arXiv ID:** 2605.04460 | [PDF](https://arxiv.org/pdf/2605.04460v1)

**作者:** Fatima Ashraf `[一作]`, Yan Shang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种“按需对齐”框架，利用固定基准的非负矩阵分解（NMF）将混合类型的调查数据映射到低维稀疏潜在空间，在此空间中通过Shapley值筛选关键潜在因子，再将其映射回可控调查变量，并通过熵正则化的最优传输（OT）配合加权ℓ₂,₁稀疏正则，学习出在保持调查可行性约束下能将目标人群转移至参考人群的稀疏、可解释的群体级干预策略。

**💡 创新点**

创新点包括：① 将政策可行性与分布级对齐统一到一个固定基准潜在空间；② 通过Shapley引导的潜在因子重要性转移，确保可控变量优先级与潜在结构紧密关联；③ 在干预学习中引入熵正则化的OT与共享杠杆稀疏正则，既保证分布级别的群体迁移，又保持干预的稀疏性与可解释性。

**🔧 技术方法**

技术手段主要包括：固定基准非负矩阵分解（NMF）、逻辑回归代理模型与Shapley值归因、熵正则化最优传输（OT）配合加权ℓ₂,₁稀疏正则、基于投影的交替优化算法，以及混合类型调查数据的可行性约束投影。

**📊 数据集**

实验使用的真实数据集包括：北京公共交通碳激励调查（1021受访者、38个变量）、VTA 2013上车调查（9654受访者、85个变量）；此外在MNIST数字分类任务中以数字3→8为源-目标迁移做交叉验证。

**📈 对比分析**

与多种基线（单一Shapley杠杆、Top‑k均匀干预、最大覆盖率干预、仅基于结果的稀疏干预）以及消除Shapley权重、稀疏正则或OT对齐的消融实验相比，完整方法在目标人群转换率、OT对齐提升、干预稀疏度等指标均表现最佳（例如在北京调查中实现61次转换、8个激活杠杆、OT差距下降0.2425）。与现有单体或分布级对抗方法（CEILS、DCE）对比亦显示本方法在群体级、政策可行性方面更具优势。

**⚠️ 局限性**

局限性包括：① 依赖固定基准NMF，若潜在结构变化或缺乏非负性约束时效果可能受限；② 需先构建可靠的可控变量与不可控变量划分，对新调查可能难以适配；③ 计算量随着样本数和潜在维度增长显著，尤其在高维像MNIST的像素级干预中；④ 代理逻辑回归与Shapley归因仅捕获局部线性关系，可能忽略非线性因果效应；⑤ 在高度非线性或交互复杂的调查场景下，稀疏正则可能导致欠拟合。

---

## 242. Deployment-Relevant Alignment Cannot Be Inferred from Model-Level Evaluation Alone

**arXiv ID:** 2605.04454 | [PDF](https://arxiv.org/pdf/2605.04454v1)

**作者:** Varad Vishwarupe `[一作]` (University of Oxford), Ivan Flechais `[通讯]` (University of Oxford)

**通讯引用:** 1590 | [OpenAlex ID](https://openalex.org/A5061337880)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文论证仅靠模型级评估无法推断部署时的对齐性，并对现有基准进行结构化审计，揭示其对交互级属性的缺失；

**💡 创新点**

提出按评估层级索引对齐声明的框架、交互级对齐八维度量表、对基准的系统化审计以及模型-支架双盲压测，证明支架效能与模型相关；

**🔧 技术方法**

采用双人编码审计、Cohen κ统计、固定权重多支架压测、对模型行为进行手工标注；

**📊 数据集**

审计基于16个主流基准（如TruthfulQA、MT‑Bench、CURATe、τ‑bench等），压测使用Claude‑Opus‑4.7、GPT‑4o、Llama‑4‑Scout生成的180条对话；

**📈 对比分析**

对比显示所有基准均未测量验证支持维度，且压测中验证支架对Claude提升到极限而对GPT‑4o无效，说明单一模型级分数不足以说明部署对齐；

**⚠️ 局限性**

局限在于基准样本有限、维度表面向HCI可能存在偏差、压测规模有限，且仍需进一步构建系统级评估方法与标准。

---

## 243. CombOL: a Library for Practical Enumeration and Boltzmann Sampling of Combinatorial Classes

**arXiv ID:** 2605.04629 | [PDF](https://arxiv.org/pdf/2605.04629v1)

**作者:** Casper Asbjørn Eriksen `[一作]` (University of Southern Denmark), Daniel Merkle `[通讯]` (Bielefeld University)

**通讯引用:** 2968 | [OpenAlex ID](https://openalex.org/A5078883427)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

开发了Combinatorial Objects Library（Combol）用于多变量组合类的枚举与Boltzmann采样，支持通过简洁字符串语法指定类、自动生成生成函数、编译采样器，并提供Python接口；

**💡 创新点**

创新点包括：动态精度浮点计算保证采样无偏差；早期拒绝（early‑rejection）策略在尺寸受限采样中显著提升效率；记录随机选择序列以实现迭代采样，避免递归带来的开销；

**🔧 技术方法**

技术实现基于Rust核心与Python绑定，使用Symbolica做符号计算、Rug做高精度数值、Newton迭代算法求计数序列、Boltzmann抽样框架、错误传播与动态精度控制、序列化/反序列化与多线程并行；

**📊 数据集**

实验使用经典组合类（如二叉树、一般树、UB树）和化学分子模型（异位同位素分子）作为示例，没有使用公开数据集；

**📈 对比分析**

通过 Pearson χ² 检验对采样均匀性进行统计验证，结果无显著偏差；与传统拒绝采样相比，早期拒绝在尺寸约束 [40,60] 的二叉树采样中平均加速 2.44 倍（95% 置信区间 2.01-2.94）；动态精度在大多数测试中未超过双精度，误差影响可忽略；

**⚠️ 局限性**

局限性：仅支持无标签组合类；对极大样本或特定结构仍可能需要更高精度；未实现标签类、复杂约束、完整极限分析、排名/逆序等功能；实验仅覆盖有限类，未评估更复杂或实际应用场景的性能与可扩展性。

---

## 244. Constructions of locally repairable codes via concatenated codes

**arXiv ID:** 2605.04618 | [PDF](https://arxiv.org/pdf/2605.04618v1)

**作者:** Hengfeng Jin `[一作]` (Nankai University), Fang-Wei Fu `[通讯]` (Nankai University)

**通讯引用:** 3239 | [OpenAlex ID](https://openalex.org/A5063946169)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过多层级码的串接（concatenated codes）构造了一系列二进制局部可恢复码（LRC），并提供了对外码（outer code）选取的系统性准则。

**💡 创新点**

创新点在于：① 用 F4 上的线性码作为外码，利用其重量分布直接推导二进制 LRC 的重量分布；② 在 r=2 的条件下，展示了多类经典极大码（如 Griesmer 码、完美码、几乎完美码、Johnson 近似完美码）可直接得到满足 Griesmer‑like、sphere‑packing‑like、Johnson‑like 边界的最优二进制 LRC；③ 给出构造过程中的具体矩阵映射与权重计算公式，使得构造从存在性证明转为显式可构造。

**🔧 技术方法**

主要技术包括：多层级码理论、线性码的权重分布与Krawtchouk多项式、射影几何中的 m‑维子空间与点集、以及各种容量/距离下界（Singleton‑like、Griesmer、C‑M、sphere‑packing、Johnson）。

**📊 数据集**

该工作为纯理论构造，无使用实际数据集；所有结论均基于码理论中的已知参数表（如 F4 上的 MDS 码、宏基特码、Solomon‑Stiffler 码、Hamming 码、Hexacode 等）和公开的码表。

**📈 对比分析**

通过与已知上界（如 Singleton‑like、Griesmer‑like、sphere‑packing‑like、Johnson‑like）对比，构造出的码在大多数参数组合下均实现了等号（即最优）。在表格与例子中给出的具体码长、维度、距离等参数均表明所构造的 LRC 达到了或接近理论极限。

**⚠️ 局限性**

局限性包括：① 目前仅针对局部可恢复度 r=2 的情况给出完整构造与分析；② 对于 r>2 的情形，外码与内码的关系更加复杂，尚缺乏统一的选取准则；③ 论文中大部分构造均基于 F4 上的已知极大码，若要扩展到更高域或更大参数，需要寻找或证明新的极大码。

---

## 245. TajikNLP: An Open-Source Toolkit for Comprehensive Text Processing of Tajik (Cyrillic Script)

**arXiv ID:** 2605.04583 | [PDF](https://arxiv.org/pdf/2605.04583v1)

**作者:** Mullosharaf K. Arabov `[一作]` (Kazan Federal University), Mullosharaf K. Arabov `[通讯]` (Kazan Federal University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5099178332)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了 TajikNLP，一个面向塔吉克语西里尔文字的完整开源 NLP 库，提供从清洗、规范化、分词、形态分析、词性标注、词干提取、词形还原到情感分析等完整处理流水线；

**💡 创新点**

创新点包括首个完整支持塔吉克语西里尔文字的统一处理框架、面向其黏着形态的统一形态学引擎、以及同时集成传统词典规则与现代神经子词/词向量模型，并公开发布了四套大型语料；

**🔧 技术方法**

技术手段涵盖基于规则和词典的形态分析器与词性标注器、BPE 子词分词、预训练 Word2Vec/FastText 词向量、字典情感分析器，以及 Python 模块化流水线设计；

**📊 数据集**

使用的数据集为 52.5k 条标注词性语料、3.5k 条情感词典、5.6k 条地名 gazetteer 与 3.8k 条人名数据库，全部以 Apache‑2.0 许可公开于 Hugging Face Hub；

**📈 对比分析**

在 1,500 句人工标注测试集上评估，词性标注器 F1≈0.86，统一词形还原 F1≈0.88，词干提取 F1≈0.90；整个库通过 616 条单元测试，代码覆盖率达 93%，表明功能健壮且可复现；

**⚠️ 局限性**

局限性包括词典覆盖不足导致稀有词处理有限、缺乏句法/依存解析组件、使用静态词向量无法捕捉上下文语义，以及目前仅提供字典情感分析，缺乏监督式分类与跨脚本翻译支持。

---

## 246. Benchmarking POS Tagging for the Tajik Language: A Comparative Study of Neural Architectures on the TajPersParallel Corpus

**arXiv ID:** 2605.04576 | [PDF](https://arxiv.org/pdf/2605.04576v1)

**作者:** Mullosharaf K. Arabov `[一作]` (Kazan Federal University), Mullosharaf K. Arabov `[通讯]` (Kazan Federal University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5099178332)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了首个塔吉克语词性标注基准，比较了经典 BiLSTM‑CRF 与多种多语言 Transformer 模型的表现。

**💡 创新点**

创新点在于首次使用 LoRA 进行低资源适配，验证多语言模型在缺乏塔吉克语预训练数据时的迁移能力，并提出首个可复现的塔吉克语 POS 基准。

**🔧 技术方法**

采用 BiLSTM‑CRF、XLM‑RoBERTa‑large、mBERT、ParsBERT、ruBERT，使用 LoRA 对 Transformer 进行参数高效微调。

**📊 数据集**

使用 TajPersParallel 平行词典语料库（约 43,819 条词条）作为训练、验证和测试集。

**📈 对比分析**

通过准确率、宏 F1 和加权 F1 进行比较，mBERT+LoRA 取得最佳效果（准确率 0.651，宏 F1 0.111，加权 F1 0.618），其余 Transformer 模型也显著优于 BiLSTM‑CRF。

**⚠️ 局限性**

局限在于仅对孤立词进行词性分类，缺乏上下文信息导致对少数词性类别无法识别，且训练集极度不平衡，易导致模型偏向主流词性。

---

## 247. VL-UniTrack: A Unified Framework with Visual-Language Prompts for UAV-Ground Visual Tracking

**arXiv ID:** 2605.04574 | [PDF](https://arxiv.org/pdf/2605.04574v1)

**作者:** Boyue Xu `[一作]` (Nanjing University), Gangshan Wu `[通讯]` (Nanjing University)

**通讯引用:** 7527 | [OpenAlex ID](https://openalex.org/A5101546753)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一的UAV-地面视觉跟踪框架 VL-UniTrack，利用单一共享编码器同时提取两视图特征，并通过视觉-语言几何提示和提示引导的跨视图适配器实现视图特定特征学习，辅之以置信度调制互相蒸馏损失以提升鲁棒性。

**💡 创新点**

核心创新点包括：
- 统一编码器消除传统两流方法的特征隔离，提升跨视图交互；
- 视觉-语言几何提示模块（VLGP）将 CLIP 文本和视觉上下文融合生成可学习的几何提示，显式引导特征对齐；
- 提示引导的跨视图适配器（PCVA）在注意力机制中注入提示，实现几何感知的特征交互；
- 置信度调制互相蒸馏损失（CMD）动态防止低置信度视图误导学习。

**🔧 技术方法**

主要技术手段包括 Transformer 共享编码器、CLIP 文本/视觉编码器、跨视图注意力、提示学习、置信度切换蒸馏、基于 L1、Focal、GIoU 的多任务损失。

**📊 数据集**

在 UGVT（UAV‑Ground Visual Tracking）基准数据集上进行训练与评测，该数据集包含200+对同步视角的跟踪序列。

**📈 对比分析**

与 DiMP50、SiamFC++、TransT、KeepTrack、MVCL、XTrack、SSTrack 等 SOTA 跟踪器进行对比。VL-UniTrack 在 UAV 视图 PR/SR 分别提升至 0.856/0.889，地面视图 PR/SR 提升至 0.872/0.905，平均性能超过所有对手；同时保持 30 FPS 的实时推理速度。

**⚠️ 局限性**

局限性：
- 由于模型参数约 98M，虽然已实现 30 FPS，但在资源受限的 UAV 边缘设备上仍需进一步压缩；
- 长期跟踪中若两视图同时出现失效或严重干扰，鲁棒性仍有待提升。

---

## 248. Neural-Guided Domain Restriction to Accelerate Pseudospectra Computation for Structured Non-normal Banded Matrices

**arXiv ID:** 2605.04550 | [PDF](https://arxiv.org/pdf/2605.04550v1)

**作者:** Amit Punia `[一作]` (Jai Narain Vyas University), Madan Lal `[通讯]` (Jai Narain Vyas University)

**通讯引用:** 80 | [OpenAlex ID](https://openalex.org/A5101409569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用神经网络预测非正交带状矩阵的伪谱敏感区域，进而只在这些区域进行昂贵的奇异值分解，以加速伪谱计算

**💡 创新点**

将矩阵全局特征与局部欧氏距离特征结合，设计了多尺度 Fourier 编码与双通道网络结构，并通过层级粗细预测和阈值校准实现高召回的自适应域限制

**🔧 技术方法**

深度学习（全连接、SiLU 激活）、Fourier 特征编码、残差块、二值交叉熵、Adam 优化、形态学膨胀等技术

**📊 数据集**

从随机带状非正交矩阵生成的 500 组训练样本（64×64，带宽 1–4），每组在 100×100 网格上计算伪谱标签；测试集 50 组相同分布的矩阵

**📈 对比分析**

与完整网格 SVD 计算和随机抽样对比，平均覆盖率 99.8%、召回率 99.5%，仅占网格 15.9%，实现 2.45 倍实际加速（理论 6.28 倍）

**⚠️ 局限性**

仅针对带状非正交矩阵验证，需先产生伪谱标签产生前期计算成本，且在更复杂或稀疏结构矩阵时性能可能下降

---

## 249. Temporal Structure Matters for Efficient Test-Time Adaptation in Wearable Human Activity Recognition

**arXiv ID:** 2605.04617 | [PDF](https://arxiv.org/pdf/2605.04617v1)

**作者:** Zishu Zhou `[一作]` (Hohai University), Xuanyao Jie `[通讯]` (Hohai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种轻量级、无需反向传播的测试时自适应框架SIGHT，针对可穿戴人体活动识别（WHAR）中的跨用户分布漂移，利用特征空间的预测惊喜和几何引导的迁移策略在线更新模型预测。

**💡 创新点**

创新点包括：①把活动持久性与转移视作特征条件化的信号，生成预测惊喜来决定是否放松时间惯性；②使用特征偏移方向与原型对齐的几何注意力来指导转移路由，同时结合流级习惯先验；③在无梯度更新的前提下实现原型和习惯向量的在线软更新，保持低计算和内存开销。

**🔧 技术方法**

采用的技术主要有：基于源模型线性头权重初始化的原型库；特征归一化与余弦距离计算的预测惊喜；几何注意力权重与习惯向量的概率归一；软原型更新与源原型的弹性锚定；所有操作均为一次前向计算，无需梯度或优化器状态。

**📊 数据集**

实验使用了两个真实世界的连续自由生活WHAR数据集——HARTH v2.0（31名受试者的加速度数据）和CAPTURE‑24（100 Hz采样的10 秒窗口），采用跨受试者评估协议，保持时间顺序进行自适应。

**📈 对比分析**

与多种现有TTA基线（如TENT、NOTE、SAR、OATTA、OFTTA、ACCUP、COA‑HAR）以及源模型对比，SIGHT在HARTH上平均MF1提高至76.73%（比最强基线高5.68pp），在CAPTURE‑24上平均MF1提升至48.77%（比最强基线高4.20pp）。同时在推理时间和内存占用上接近最小，优于需要梯度或多次前向的基线。

**⚠️ 局限性**

局限性：①对参数β、τ、η_μ的敏感性在CAPTURE‑24等高变异数据集上仍需谨慎；②目前仅在单一时间窗口内考虑特征偏移，未显式建模长序列动态；③在极端噪声或频繁转移场景下，预测惊喜阈值的选择可能影响稳定性。

---

## 250. Library learning with e-graphs on jazz harmony

**arXiv ID:** 2605.04622 | [PDF](https://arxiv.org/pdf/2605.04622v1)

**作者:** Zeng Ren `[一作]` (Ecole Polytechnique Federale de Lausanne), Martin Rohrmeier `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了基于 e‑graph 的库学习框架，从爵士和弦进程中无监督地学习可重用的和声模式，并生成压缩的解析程序。

**💡 创新点**

首次将消歧解析与库学习统一到等价饱和框架中，通过反向统一生成候选抽象并在共同解析与库学习空间内最小化 MDL 目标，解决两者相互依赖的问题。

**🔧 技术方法**

采用等价饱和（e‑graph）与下推逻辑编程实现 CYK 解析、反向统一、模板程序式 DSL、成本集分析，并以 MDL 原则评估库与解析成本。

**📊 数据集**

使用三首爵士曲目（Red Clay、Valse Hot、Sunny）的和弦进程作为小规模验证集，后续计划扩展至更大多样的曲集。

**📈 对比分析**

通过与各曲独立局部库学习比较，使用压缩率（CR）衡量；全局共享库学习将总解析长度从 87 降到 27，归一化 CR 为 1.5，优于局部学习 1.16，然而仅在三首曲子上实验，未在大规模数据上评估。

**⚠️ 局限性**

当前实现受限于效率，难以在大规模语料上完成全局库学习；压缩性未必保证音乐可解释性，尤其在小样本情况下可能产生非传统解析；使用的爵士和声语法简化，未覆盖更复杂的和声现象。

---

## 251. Practical validation of synthetic pre-crash scenarios

**arXiv ID:** 2605.04564 | [PDF](https://arxiv.org/pdf/2605.04564v1)

**作者:** Jian Wu `[一作]` (Volvo Cars), Jonas Bärgman `[通讯]` (Chalmers University of Technology)

**通讯引用:** 1390 | [OpenAlex ID](https://openalex.org/A5021204321)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出并实现了基于贝叶斯 ROPE 的实用等价性检验框架，新增了分箱统计量 θ 与 Θ，用于评估合成预碰撞场景与真实场景在自动驾驶安全影响评估中的代表性。

**💡 创新点**

创新点在于将分箱与权重化相结合，构造可解释的局部与整体分布差异度量，并通过 ROPE 直接判定实用等价性，弥补了传统显著性检验无法证明等价的缺陷。

**🔧 技术方法**

技术方法包括贝叶斯分布拟合（正态、对数正态、伽马等）与留一交叉验证、KNN 采样加权、重采样、分箱与权重函数设计、HDI 与 ROPE 比较。

**📊 数据集**

使用了 200 个 QUADRIS 参考场景、866 个 PCM 事故重建场景（原始与加权版）以及 7,888 个基于 SCM 的合成场景作为案例验证。

**📈 对比分析**

通过计算 θ、Θ 的 95% HDI 并与预设 ROPE 阈值比较，评估等价性；结果显示加权 PCM 与参考数据等价，原始 PCM 与 SCM 未通过，验证了加权修正的有效性并提供了诊断信息。

**⚠️ 局限性**

局限性包括未考虑指标间相关性、缺乏系统化指标选择准则、对先验与 ROPE 设定敏感且需专家判断、重采样有限导致 SCM 结果可能不具代表性。

---

## 252. Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation

**arXiv ID:** 2605.04559 | [PDF](https://arxiv.org/pdf/2605.04559v1)

**作者:** Ruijun Chen `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 43859 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 BLADE 框架，用动态贝叶斯估计的自我进化目标分布，直接在 LLM 推荐系统中优化非可微的列表级指标。

**💡 创新点**

创新点在于：① 用贝叶斯动态估计不断更新 BoN 目标，克服静态参考的无差别监督和梯度衰减；② 通过自我进化的目标保持梯度信息，突破静态上限；③ 在不额外计算开销的情况下实现持续优化。

**🔧 技术方法**

技术包括：大型语言模型（Llama‑3.2‑1B‑Instruct）、Best‑of‑N（BoN）对齐、Group Relative Policy Optimization (GRPO)、贝叶斯动态估计（Beta 分布更新）、强化学习与对齐策略。

**📊 数据集**

数据集：Amazon CDs & Vinyl、Steam、Goodreads。

**📈 对比分析**

与 SFT、DPO、List‑DPO、BoN Alignment 等基线在 Recall@5、NDCG@5 等指标上对比，BLADE 在所有数据集上显著提升，特别在多样性和公平性指标上取得领先，同时打破静态 BoN 的性能上限。

**⚠️ 局限性**

局限性包括：① 需要手动调节动态系数 τ；② 对更大模型或极端资源受限场景的适用性尚未充分验证；③ 训练仍需生成一批候选列表，计算成本相对较高；④ 可能在极端稀疏数据上性能下降。

---

## 253. Benchmarking LLMs on the Massive Sound Embedding Benchmark (MSEB)

**arXiv ID:** 2605.04556 | [PDF](https://arxiv.org/pdf/2605.04556v1)

**作者:** Cyril Allauzen `[一作]` (Google), Ke Wu `[通讯]` (Google)

**通讯引用:** 1538 | [OpenAlex ID](https://openalex.org/A5014326389)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统评估多模态大型语言模型（如 Gemini 2.5/3 Flash、GPT‑4o‑mini‑audio 等）在 Massive Sound Embedding Benchmark（MSEB）中的表现，并通过统一的 prompt 模板和 RAG 方案，将音频任务转化为文本或结构化输出。

**💡 创新点**

首次将 MSEB 作为全面基准，量化音频‑文本平衡（audio‑text parity），并对比音频原生与 cascaded 两种架构在多种任务（转写、检索、推理、分类、分割等）上的优势与不足；同时提出“headroom”指标评估音频输入与文本输入的性能差距。

**🔧 技术方法**

使用 prompt 设计、结构化 JSON 输出、检索增强生成（RAG）管道、密集嵌入模型（Gemini Embedding）以及多任务评估工具箱（MSEB Toolkit）等技术；在评估中结合 Whisper、GPT‑4o‑transcribe、ElevenLabs 等 ASR 前端。

**📊 数据集**

基于 MSEB 的公开数据集，包括 SVQ（多语言语音问答）、Speech‑MASSIVE（意图槽位标注）、FSD50K（环境声音分类）以及来自 Wikipedia 的文档检索集合；多种语言与噪声环境均被覆盖。

**📈 对比分析**

通过对每个任务的标准指标（WER、MAP、NDCG、F1、MRR 等）进行量化比较，发现：① 语音转写上 GPT‑4o‑transcribe 领先；② 检索与重排序任务中 Gemini 2.5/3 Flash 与 cascaded 方案表现相近；③ 语音原生模型在推理任务上逐渐逼近文本基准，但整体仍存在显著的 modality gap；④ 评估成本与延迟各模型差异明显，GPT 系列在吞吐量与费用上更具优势。

**⚠️ 局限性**

局限性包括：① 仍存在明显的音频‑文本差距，尤其在噪声和低资源语言上性能不稳定；② 评估仅覆盖小规模参数版本（如 GPT‑4o‑mini、Gemini 3 Flash），缺乏大参数前沿模型的验证；③ 可能存在数据泄漏（test data contamination），导致部分任务的高分不可靠；④ 评估主要基于零样本或少量示例，未充分探索 fine‑tuning 或自监督训练的潜力。

---

## 254. Counter-Dyna: Data-Efficient RL-Based HVAC Control using Counterfactual Building Models

**arXiv ID:** 2605.04555 | [PDF](https://arxiv.org/pdf/2605.04555v1)

**作者:** Jan Marco Ruiz de Vargas `[一作]` (Technical University of Munich), Christoph Goebel `[通讯]` (Technical University of Munich)

**通讯引用:** 1119 | [OpenAlex ID](https://openalex.org/A5030903555)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种名为Counter-Dyna的基于Dyna的模型驱动强化学习方法，用于HVAC控制，通过重新设计代理模型仅预测受控制变量的状态并采用按时间顺序的训练情景，显著提升了数据效率；

**💡 创新点**

创新点在于将状态空间拆分为受控与不可控变量，构建计数事实代理模型（CSM）仅预测区间温度，避免对天气、电价等不可控变量建模，并在训练中严格按时间顺序进行，消除了时间旅行问题；

**🔧 技术方法**

使用了Dyna式模型驱动强化学习、因果推断构建CSM、深度强化学习算法PPO/SAC，以及BOPTEST仿真框架进行仿真；

**📊 数据集**

采用BOPTEST的bestest hydronic heat pump测试案例，并加入2024年比利时电价数据作为动态价格；

**📈 对比分析**

与模型自由的10周、50周训练以及基线控制器对比，Counter-Dyna在仅5-10周训练即可实现50周模型自由的性能，成本节约5.3%至17%，且在不同季节表现稳定；

**⚠️ 局限性**

局限在于依赖完美预测、单一MLP建筑模型、BOPTEST简化场景、对真实环境重置方式不现实，且未考虑不确定天气/价格预测、安全约束和迁移学习等。

---

## 255. InterMesh: Explicit Interaction-Aware End-to-End Multi-Person Human Mesh Recovery

**arXiv ID:** 2605.04554 | [PDF](https://arxiv.org/pdf/2605.04554v1)

**作者:** Kaili Zheng `[一作]` (Tsinghua University), Ji Wu `[通讯]` (Tsinghua University)

**通讯引用:** 6011 | [OpenAlex ID](https://openalex.org/A5029547618)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 InterMesh 框架，显式将人-环境交互（HOI）信息融入端到端的人体网格恢复管线，利用 HOI 检测器生成交互特征并通过 Contextual Interaction Encoder 与 Interaction-Guided Refiner 模块加强查询的语义表示。

**💡 创新点**

创新点在于：1）首次将预训练的零射击 HOI 检测器直接用于多人人体网格恢复，提供人-物体和人-人交互语义；2）设计轻量化的 Contextual Interaction Encoder 与 Interaction-Guided Refiner，将交互上下文逐层融入查询，提升姿态与形状估计；3）实现显式交互建模，突破传统基于注意力的隐式关系建模局限。

**🔧 技术方法**

技术包括：DETR 结构的端到端 HMR（Encoder-Decoder 与查询），ViT-Base backbone，DAB-DETR 的 reference point 机制，使用 EZ-HOI 作为零射击交互检测器，Multi-Head Self-Attention 与 Cross-Attention 的交互编码器和精炼器，SMPL 参数回归以及多任务损失。

**📊 数据集**

使用多个人体交互数据集：3DPW、MuPoTS、CMU Panoptic、Hi4D 与 CHI3D 进行训练与评估，并在 AGORA、BEDLAM、COCO、MPII、CrowdPose、Human3.6M 等数据集上进行预训练。

**📈 对比分析**

与 SOTA 方法（SAT-HMR、ROMP、BEV、PSVT、Multi-HMR 等）进行对比，InterMesh 在 3DPW、MuPoTS、CMU Panoptic、Hi4D、CHI3D 上均取得显著提升：如 CMU Panoptic 上 MPJPE 降 8.3mm（9.9%），Hi4D 上 MPJPE、PA-MPJPE、PVE 分别下降 8.2%、6.8%、8.8%，CHI3D 上 MPJPE 降 2.4mm，整体提升体现了交互语义对姿态与形状的积极作用。

**⚠️ 局限性**

主要限制包括：对 HOI 检测器的性能高度依赖，检测误差会直接影响网格恢复；交互特征生成与编码增加计算开销，尚未实现实时推理；目前仅评估在已标注交互场景的数据集，缺乏在更广泛、非交互或极端遮挡场景中的鲁棒性验证。

---

## 256. The Newsworthiness of Brazilian Distress: A Peak Analysis on Time Series of International Media Attention to Disasters in Brazil

**arXiv ID:** 2605.04552 | [PDF](https://arxiv.org/pdf/2605.04552v1)

**作者:** Brielen Madureira `[一作]` (Leipzig University), Mariana Madruga de Brito `[通讯]` (Helmholtz Centre for Environmental Research)

**通讯引用:** 2275 | [OpenAlex ID](https://openalex.org/A5060930814)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对德国媒体2000-2024年关于巴西火灾和滑坡的新闻进行峰值检测与时间序列分析，验证与EM-DAT和S2iD灾害数据库的对齐情况。

**💡 创新点**

首次将时间序列峰值分割方法与外部灾害数据库对齐，揭示国际媒体报道与官方灾害记录的互补性，并探讨不同灾害类型（火灾聚合性 vs 滑坡事件化）的报道差异。

**🔧 技术方法**

使用SciPy的find_peaks进行时间序列峰值检测、文本主题模型筛选、手工误差分析，并进行时间对齐与数据库匹配。

**📊 数据集**

德国wiso-net新闻聚合器筛选出的约13万篇极端气候事件新闻（聚焦巴西火灾、滑坡），以及EM-DAT与S2iD（巴西官方灾害数据库）记录。

**📈 对比分析**

对齐率约为28%（与EM-DAT或S2iD匹配的新闻事件占总新闻峰值的比例），手工验证显示绝大多数匹配为真阳性，但召回率未知；峰值特征分布显示火灾聚合性报道多、滑坡报道更细粒度。

**⚠️ 局限性**

召回率未知、地理定位仅依赖关键词过滤导致可能的误报与漏报、火灾报道多聚合导致事件混淆、对齐仅基于时间而非内容、对峰值检测超参数敏感。

---

## 257. A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints

**arXiv ID:** 2605.04595 | [PDF](https://arxiv.org/pdf/2605.04595v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 258. Hard CNF Instances for Ideal Proof Systems

**arXiv ID:** 2605.04544 | [PDF](https://arxiv.org/pdf/2605.04544v1)

**作者:** Tuomas Hakoniemi `[一作]` (University of Helsinki), Iddo Tzameret `[通讯]` (Imperial College London)

**通讯引用:** 242 | [OpenAlex ID](https://openalex.org/A5066962007)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

在Ideal Proof System (IPS) 的框架下，研究了读一次无偏见代数分支程序(roABP) 的证明系统 roABP- 对 CNF 公式的下界，证明了任意变量顺序下都存在指数级大小的证明不可行。

**💡 创新点**

创新点在于将可行插值（feasible interpolation）方法与 roABP 的宽度分解相结合，得到对 CNF 公式的非平凡指数下界；并且通过一次提升（lifting）构造，使得下界对所有变量顺序都成立。

**🔧 技术方法**

主要技术包括：1）将 IPS 归约为 Nullstellensatz 形式并用 roABP 计算系数；2）在给定变量顺序下对 roABP 进行层分解，构造低维空间；3）构造单调 span‑program 作为插值函数；4）利用已知的单调 span‑program 下界实现 CNF 下界；5）使用一致性子公式进行一次提升，消除变量顺序限制。

**📊 数据集**

该工作为理论论文，无实验数据集，所有结果均为抽象证明与构造。

**📈 对比分析**

与此前仅针对单一代数实例或特殊 IPS 片段的下界相比，本文得到对任意 CNF 公式在所有变量顺序下的指数下界；此外证明展示了 roABP- 能模拟树形 Polynomial Calculus，且对 Tseitin 和函数鸽笼原理给出了多项式宽度的上界。

**⚠️ 局限性**

局限性包括：1）下界仅适用于 roABP- 这一较弱的 IPS 片段；2）未能推广至更强的 IPS 片段（如常数层 IPS、全阶 IPS）；3）方法在实际复杂度上仍受限于变量顺序的固定性，尽管通过提升消除了这一限制，但对更一般的证明系统仍缺乏对应技术。

---

## 259. HeterSEED: Semantics-Structure Decoupling for Heterogeneous Graph Learning under Heterophily

**arXiv ID:** 2605.04594 | [PDF](https://arxiv.org/pdf/2605.04594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 260. IntenBot: Flexible and Imprecise Multimodal Input for LLMs to Understand User Intentions for Casual and Human-Like HRI

**arXiv ID:** 2605.04585 | [PDF](https://arxiv.org/pdf/2605.04585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 261. Lightning Unified Video Editing via In-Context Sparse Attention

**arXiv ID:** 2605.04569 | [PDF](https://arxiv.org/pdf/2605.04569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 262. Stage-adaptive audio diffusion modeling

**arXiv ID:** 2605.04547 | [PDF](https://arxiv.org/pdf/2605.04547v1)

**作者:** Xuanhao Zhang `[一作]` (China Pharmaceutical University), Chang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 12127 | [OpenAlex ID](https://openalex.org/A5007581833)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了音频扩散模型训练中的阶段感知策略，提出了基于自监督特征进展的阶段感知训练机制。

**💡 创新点**

创新点在于统一使用SSL进度信号调度三种互补的训练策略：SSL引导衰减、自适应时间步采样以及结构感知正则化，从而实现训练过程的动态优化。

**🔧 技术方法**

技术手段包括冻结的自监督音频编码器进行SSL特征引导，Beta分布自适应采样时间步，以及基于CKA相似度的图平滑正则化在Transformer块间施加结构约束。

**📊 数据集**

实验数据集涵盖文本条件音频生成的AudioSet与FreeSound训练、AudioCaps评估，以及音频超分辨率的VCTK训练和VCTK‑test评测。

**📈 对比分析**

与传统stable‑audio‑tools基线以及Make‑An‑Audio、AudioLDM、Tango等方法比较，stage‑aware机制在FAD、LSD、SISNR等指标上均显著提升，证明训练效率与生成质量均得到改善。

**⚠️ 局限性**

局限性包括在波形级指标SISNR上提升不明显，可能由于潜在空间与波形空间的语义对齐不一致，以及所提策略在更大规模模型和多模态任务上的通用性仍需进一步验证。

---

## 263. Guidelines for Designing AI Technologies to Support Adult Learning

**arXiv ID:** 2605.04616 | [PDF](https://arxiv.org/pdf/2605.04616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 264. VocalParse: Towards Unified and Scalable Singing Voice Transcription with Large Audio Language Models

**arXiv ID:** 2605.04613 | [PDF](https://arxiv.org/pdf/2605.04613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 265. From Diffusion to Rectified Flow: Rethinking Text-Based Segmentation

**arXiv ID:** 2605.04590 | [PDF](https://arxiv.org/pdf/2605.04590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. UniPCB: A Generation-Assisted Detection Framework for PCB Defect Inspection

**arXiv ID:** 2605.04635 | [PDF](https://arxiv.org/pdf/2605.04635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 267. An Axiomatic Analysis of Proportionality Notions in Approval-Based Multiwinner Voting

**arXiv ID:** 2605.04612 | [PDF](https://arxiv.org/pdf/2605.04612v1)

**作者:** Chris Dong `[一作]` (Hasso Plattner Institute, Universitaet Potsdam), Jannik Peters `[通讯]` (Shanghai University Of Finance And Economics)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对approval-based多winner选举中比例代表性概念进行公理化刻画，重点研究PJR+和EJR+，并给出其最弱/最强的公理集合；

**💡 创新点**

首次用公理方法区分传统PJR/EJR与其+版本，证明单调性是区分两者的关键，并提出见证者基础的比例定义、merge-proofness等新公理，获得PJR+与EJR+的唯一性刻画；

**🔧 技术方法**

采用公理化方法、下限法配额、可证明性分析、组合与证明技巧、见证者框架等技术；

**📊 数据集**

未使用真实数据，主要使用合成实例和理论构造；

**📈 对比分析**

通过理论证明展示PJR+和EJR+在满足一组最小公理下是唯一的；实验部分（模拟）显示+版本比原始版本更具辨别力；

**⚠️ 局限性**

仅覆盖基于可见性/一致性类的比例概念，未涵盖过度代表性、核心等；依赖下限法配额，未探讨Droop配额；仅针对认可投票，未扩展到序数或随机化设置。

---

## 268. SensingAgents: A Multi-Agent Collaborative Framework for Robust IMU Activity Recognition

**arXiv ID:** 2605.04608 | [PDF](https://arxiv.org/pdf/2605.04608v1)

**作者:** Naiyu Zheng `[一作]` (City University of Hong Kong), Zhimeng Yin `[通讯]` (City University of Hong Kong)

**通讯引用:** 1964 | [OpenAlex ID](https://openalex.org/A5027785880)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 SensingAgents 多代理系统，用位置专属分析员、动态静态辩论和决策代理实现 IMU 活动识别。

**💡 创新点**

创新点在于：①将每个身体部位分配专门的分析员并通过工具调用实现精准信号抽取；②采用动态–静态辩论机制以结构化方式解决跨传感器冲突；③提供可解释的链式推理，显著提升透明度。

**🔧 技术方法**

使用技术包括：大型语言模型（Claude Sonnet 4.6、GPT‑5‑mini 等）与 ReAct 调度、预设的 FFT、峰值检测、能量分布等信号处理工具，以及 LangGraph 进行多代理协调。

**📊 数据集**

使用的数据集为公开的 Shoaib 多位置 IMU 数据集（10 位受试者、7 个日常动作、5 个身体位置）。

**📈 对比分析**

在零样本设置下与单代理、代码生成多代理、传统 SVM、IMU‑BERT 等基线比较，准确率达 79.5%，比单代理高约 40 pp，超出深度学习模型在未见用户时 9 % 的性能，且保持跨用户一致性。

**⚠️ 局限性**

局限性包括：仅在受控实验数据上验证，缺乏对自由生活、任意姿态或缺失传感器的评估；工具设计高度依赖人工专业知识，扩展性受限；在极短窗口内的姿态/楼梯方向区分仍易混淆。

---

## 269. SWE-WebDevBench: Evaluating Coding Agent Application Platforms as Virtual Software Agencies

**arXiv ID:** 2605.04637 | [PDF](https://arxiv.org/pdf/2605.04637v1)

**作者:** Siddhant Saxena `[一作]` (BaseThesis Labs), Vinayaka Jyothi `[通讯]` (QwikBuild)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个68指标的全栈软件评测基准SWE‑WebDev Bench，用于评估AI应用构建平台从需求获取到部署交付的全流程能力。

**💡 创新点**

创新点在于：①将评测维度拆分为交互模式（ACR/AMR）、机构角度（PM/工程/运维）和复杂度分层（T4/T5）；②引入嵌入式“canary”需求检测，以区分真实理解与模板匹配；③结合主指标与诊断指标，精准定位平台缺陷。

**🔧 技术方法**

采用LLM评判（Claude 3.5 Sonnet）与自动化工具（Lighthouse、k6、npm audit）相结合的分层评分体系，平台则基于多代理或单射程生成技术。

**📊 数据集**

使用了三领域（教育、现场服务、金融AI）共六个标准化提示，嵌入80条canary需求，形成18个ACR评测单元，评测对照六大AI构建平台。

**📈 对比分析**

通过对六个平台在三领域18个ACR单元的多维评分，结果显示无平台工程得分超过60%，即便最高平台在关键指标（如前端工程）也低于70%，体现整体缺口仍大。

**⚠️ 局限性**

局限性包括：评测样本规模有限（仅18个ACR单元，AMR仅单平台评估）；部分评估与作者关联平台相关；未实现完全盲测；缺乏统计显著性检验，需进一步扩展与独立复现。

---

## 270. Autonomous Synchronization of Discrete-Time Heterogeneous Multiagent Systems

**arXiv ID:** 2605.04627 | [PDF](https://arxiv.org/pdf/2605.04627v1)

**作者:** Wei Hu `[一作]` (Beihang University), Quanyi Liang `[通讯]` (Beihang University)

**通讯引用:** 194 | [OpenAlex ID](https://openalex.org/A5103193369)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并分析了离散时间异质多智能体系统的自主同步问题，并给出了满足同步的控制设计与同步速率估计；

**💡 创新点**

1) 给出了离散时间LTV系统的渐近解耦充分条件，2) 在此基础上提出了对异质系统的同步条件，3) 将同步条件简化为只依赖于平均动力学矩阵的谱半径，显著降低了对系统同质性的要求；

**🔧 技术方法**

利用图论、线性代数（Jordan 标准形、矩阵范数）、递推不等式与黎卡提不等式等数学工具，构造控制增益并证明系统的渐近解耦；

**📊 数据集**

采用人工设计的四节点加权无向网络（Laplacian矩阵已给出）以及三维状态、单输入的系统矩阵进行数值仿真验证；

**📈 对比分析**

通过与前人基于同质条件或强烈同质约束的同步条件对比，证明了本方法在异质程度更高时仍能实现同步，仿真结果表明实际收敛速率与理论上限值 r* 近似一致，远快于 0.7^t 的衰减；

**⚠️ 局限性**

仍需全局信息（图拓扑、平均动力学矩阵）来选取耦合强度和 Riccati 方程解，缺乏完全分布式实现；

---

## 271. AuditRepairBench: A Paired-Execution Trace Corpus for Evaluator-Channel Ranking Instability in Agent Repair

**arXiv ID:** 2605.04624 | [PDF](https://arxiv.org/pdf/2605.04624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 272. Advancing Aesthetic Image Generation via Composition Transfer

**arXiv ID:** 2605.04609 | [PDF](https://arxiv.org/pdf/2605.04609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 273. Beyond Retrieval: A Multitask Benchmark and Model for Code Search

**arXiv ID:** 2605.04615 | [PDF](https://arxiv.org/pdf/2605.04615v1)

**作者:** Siqiao Xue `[一作]` (Alipay), Hang Yu `[通讯]` (Alipay)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了CoREB，一个多任务、污染有限的代码检索与重排序基准，并发布了针对CoREB的 fine‑tuned 重排序器；

**💡 创新点**

创新点在于：①将 LiveCodeBench 的问题通过对抗性改写，消除表面记忆并保持语义；②在检索任务中引入分级相关性标签（正样本、硬负样本、易负样本），打破传统二元评测；③构建完整的两阶段检索‑重排序流水线，并在此基准上首次实现三向任务均提升的重排序器；

**🔧 技术方法**

技术手段包括：对抗性改写 + LLM 代码生成（Gemini、Claude）用于生成多语言候选；多种嵌入模型（代码专用与通用）进行一次阶段检索；通过 LoRA 对 Qwen3‑Reranker 进行任务内 fine‑tune，得到 CoREB‑Reranker；评测采用 nDCG@10、Recall@10、MRR 等指标；

**📊 数据集**

数据集主要是改写后的 LiveCodeBench（涵盖 Python、Java、C++、Ruby、Go），生成的文本与代码语料，以及与 CodeSearchNet、APPS、CosQA、CodeFeedback 等公开数据混合用于重排序训练；

**📈 对比分析**

对比方法：在两阶段检索‑重排序框架下，对 11 种嵌入模型和 5 种重排序器进行评估；结果显示代码专用模型在各任务中普遍优于通用模型；无单一模型在三任务中统治；但短关键词查询几乎无效，低资源语言表现不佳；fine‑tuned CoREB‑Reranker 成为唯一在所有任务上均正向提升的重排序器；

**⚠️ 局限性**

局限性包括：短关键词检索性能仍低至零；低资源语言（如 Go、Ruby）仍落后；硬负样本侵入率高，影响评测细粒度；基准仍未覆盖所有主流编程语言；模型规模与性能不呈单调关系，需进一步研究；

---

## 274. Reference-based Category Discovery: Unsupervised Object Detection with Category Awareness

**arXiv ID:** 2605.04606 | [PDF](https://arxiv.org/pdf/2605.04606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 275. PINSIGHT: A Comprehensive Threat Exploration of Domain-Adaptive Wi-Fi based PIN Code Inference

**arXiv ID:** 2605.04570 | [PDF](https://arxiv.org/pdf/2605.04570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 276. Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination

**arXiv ID:** 2605.04568 | [PDF](https://arxiv.org/pdf/2605.04568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 277. DiCLIP: Diffusion Model Enhances CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation

**arXiv ID:** 2605.04593 | [PDF](https://arxiv.org/pdf/2605.04593v1)

**作者:** Zhiwei Yang `[一作]` (Fudan University), Zhijian Song `[通讯]` (Fudan University)

**通讯引用:** 4095 | [OpenAlex ID](https://openalex.org/A5111697224)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了DiCLIP，一种利用Stable Diffusion增强CLIP的弱监督语义分割框架，能够在仅有图像级标签的条件下生成高质量CAM并训练分割网络。

**💡 创新点**

创新点在于：① 通过视觉相关增强（VCE）模块将扩散模型的空间一致性注入CLIP注意力，消除CLIP过平滑导致的细粒度信息缺失；② 通过文本语义增强（TSA）模块构建键值缓存，实现视觉-文本的patch‑wise知识检索，弥补CLIP在视觉与文本模态上的稀疏密集知识不足；③ 结合非参数Attention Clustering Refinement（ACR）和动态适配器，实现无训练和训练高效的CAM生成与分割。

**🔧 技术方法**

使用技术包括CLIP、Stable Diffusion 2.1、ViT视觉编码器、UNet自注意力提取、ACR聚类、键值缓存检索、Transformer分割头、AdamW优化等。

**📊 数据集**

在PASCAL VOC 2012和MS COCO 2014两个常用语义分割基准数据集上进行实验验证。

**📈 对比分析**

与最新的单阶段和多阶段CLIP基准方法（如WeCLIP、WeakCLIP）以及其他传统WSSS方法对比，DiCLIP在VOC上取得78.8%/78.9% mIoU（比WeCLIP提升约3.7%），在COCO上达到48.7% mIoU，且训练时间仅为115分钟，显著低于现有SOTA（约270–360分钟）。

**⚠️ 局限性**

主要局限是键值缓存完全基于Stable Diffusion生成的合成图像，可能带来合成偏差；在扩散模型未充分覆盖的类别或极为复杂场景下性能可能受限。

---

## 278. From Parameter Dynamics to Risk Scoring : Quantifying Sample-Level Safety Degradation in LLM Fine-tuning

**arXiv ID:** 2605.04572 | [PDF](https://arxiv.org/pdf/2605.04572v1)

**作者:** Xiao Wang `[一作]` (Northeastern University), Daling Wang `[通讯]` (Northeastern University)

**通讯引用:** 1940 | [OpenAlex ID](https://openalex.org/A5035378456)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首先从参数动态视角分析了LLM在细调过程中出现安全降解的机制，并基于此提出了样本级安全风险量化方法SQSD。

**💡 创新点**

创新点包括：①发现安全降解是由参数累积漂移朝危险方向导致；②提出通过样本导致的参数更新投影差值来连续量化每个训练样本的风险；③用一阶泰勒展开将参数更新与模型输出偏好关联；④证明SQSD在不同模型架构、规模及LoRA/全参数微调间具有良好迁移性。

**🔧 技术方法**

主要技术手段有：参数动态追踪、方向投影（安全与危险方向）、LoRA梯度更新、模块归一化、参数更新的泰勒近似、通过DPO、SFT构造安全/危险方向、Safety Score与ASR评估。

**📊 数据集**

使用的数据集包括 PKU‑SafeRLHF‑10K（构造安全方向）、Aegis 与 BeaverTails（构造危险方向）、Alpaca 与 Dolly（细调训练数据）、以及 CatHarmfulQA、AdvBench、HEx‑PHI（安全评估）。

**📈 对比分析**

通过与 Reward Model、Bi‑Anchor、Self‑Inf‑N、LARF 等基线方法对比，SQSD 在 12 个实验配置中均保持 ASR 单调递减且 ASR 差异（S1‑S5）最大，显示出更强的风险识别与区分能力；迁移实验进一步证明其在不同架构、参数规模和微调方式上的稳定表现。

**⚠️ 局限性**

主要局限包括：① SQSD 的效果依赖于用于计算方向的模型对安全/危险方向的敏感性，初始化不当会降低可靠性；② 方向构造方法目前仍局限于特定安全/危险数据集，可能不具备普适性；③ 目前未将 SQSD 直接集成进安全微调算法，如何在实际训练中利用风险评分仍待研究。

---

## 279. RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs

**arXiv ID:** 2605.04563 | [PDF](https://arxiv.org/pdf/2605.04563v1)

**作者:** Hanum Ko `[一作]` (Sungkyunkwan University), Jungrae Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 13488 | [OpenAlex ID](https://openalex.org/A5100354952)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种基于范围标识符（RID）的元数据级错误纠正框架 RangeGuard，以在 GPU HBM 内存中对深度学习模型进行可靠性增强。

**💡 创新点**

创新点在于将纠错关注点从位级别转向数值范围，利用 RID 在有限的冗余预算内实现多位错误的有界近似校正，并通过多种范围映射实现对不同数值格式的适配。

**🔧 技术方法**

采用的技术包括 RID、符号级 Reed–Solomon 码（4b 双符号修正/8b 单符号修正）、指数域范围映射、基于物理地址的 Map Tag 选择以及 GPU 内存控制器的兼容实现。

**📊 数据集**

实验使用了 ResNet‑50、Llama‑3.1‑8B 和 Llama‑3.2‑1B 三个模型的权重与激活，并在 ImageNet‑1K 与 ARC‑Easy 评测集上评估。

**📈 对比分析**

通过与基线 SEC‑DED、Weight Nulling、VAPI 等方案在多种 DRAM 错误模式（单比特、相邻双比特、16/32 比特突发、全芯片）下的误差覆盖率和模型准确率比较，RangeGuard 在相同 16‑bit 冗余下可容忍 64+ 位错误、在 BER=10⁻⁶ 时保持 LLM 准确率>90%，并仅产生 0.008% 的 IPC 损失。

**⚠️ 局限性**

局限性包括对范围映射参数（σ）的敏感性、需要在控制器中存储多份 RangeMap 以及对极端大范围错误（如全芯片错误）仍需额外检测/恢复机制。

---

## 280. Power Distribution Bridges Sampling, Self-Reward RL, and Self-Distillation

**arXiv ID:** 2605.04542 | [PDF](https://arxiv.org/pdf/2605.04542v1)

**作者:** Akiyoshi Tomihari `[一作]` (University of Tokyo), Issei Sato `[通讯]` (University of Tokyo)

**通讯引用:** 2768 | [OpenAlex ID](https://openalex.org/A5060421432)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了“功率分布”(power distribution)，并证明它既是功率采样的目标分布，也是自奖励KL正则化强化学习的闭式最优解，也是自蒸馏的教师分布，从而将昂贵的在线采样成本迁移到离线训练。

**💡 创新点**

创新点在于：① 将功率采样、RL和自蒸馏统一为同一分布；② 证明局部（token级）近似无法复现序列级功率采样；③ 通过自奖励RL得到功率分布，进而推导出离线的功率自蒸馏方法；④ 给出自蒸馏对自奖励锐化与真实奖励改进的可解释性（协方差条件）。

**🔧 技术方法**

核心技术包括：自奖励KL正则化强化学习、马尔可夫链蒙特卡洛（MH）功率采样、序列级信息理论（Renyi 熵）分析、前向KL蒸馏（最大似然）、经验量化实验。

**📊 数据集**

主要使用的公开数据集是 MATH、HumanEval、MBPP、GPQA；在实验中重点展示了 Qwen2.5-Math-7B 在 MATH500 子集上的效果，并用 Phi‑3.5‑mini‑instruct 进行对比。

**📈 对比分析**

与基线比较：标准自回归采样、token‑level 温度采样、最佳‑N 选取等。结果显示：功率采样提升了自奖励 r_self；当 r_self 与真实奖励 r^⋆ 正相关时，功率采样和自蒸馏也提升了 r^⋆；功率自蒸馏在离线训练后，使用简单温度采样即可达到或超过在线功率采样的性能，且推理成本大幅下降。

**⚠️ 局限性**

局限性包括：自蒸馏的提升受限于基模型能力，若基模型弱则提升有限；研究聚焦于有限长度的自回归模型，未考虑更大规模或不同架构；对奖励函数的依赖仅限于自奖励，未探究更丰富的外部奖励；实验未覆盖所有可能的模型/数据组合。

---

## 281. A Blockchain-as-a-Service Solution for TAFES-Compliant Verification of Fair Trade Certifications

**arXiv ID:** 2605.04600 | [PDF](https://arxiv.org/pdf/2605.04600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 282. GTF: Omnidirectional EPI Transformer for Light Field Super-Resolution

**arXiv ID:** 2605.04581 | [PDF](https://arxiv.org/pdf/2605.04581v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 283. SAMIC: A Lightweight Semantic-Aware Mamba for Efficient Perceptual Image Compression

**arXiv ID:** 2605.04560 | [PDF](https://arxiv.org/pdf/2605.04560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 284. Event-Based Early Warning of Vineyard Disease Risk from Environmental Time Series

**arXiv ID:** 2605.04548 | [PDF](https://arxiv.org/pdf/2605.04548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 285. Open-Source Image Editing Models Are Zero-Shot Vision Learners

**arXiv ID:** 2605.04566 | [PDF](https://arxiv.org/pdf/2605.04566v1)

**作者:** Wei Liu `[一作]` (Tencent Inc), Rui Chen `[通讯]` (Tencent Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了三种公开的图像编辑模型（Qwen‑Image‑Edit、FireRed‑Image‑Edit、LongCat‑Image‑Edit）在不进行任何任务特定微调的情况下，利用自然语言提示和确定性解码器，完成稠密视觉预测任务（单目深度、表面法向、语义分割）的零样本性能；并将其结果与现有的指令微调模型（Vision Banana、Marigold）以及公开基准进行对比，形成可复现的公开基准；

**💡 创新点**

首次在零样本条件下，将图像编辑模型转化为稠密预测器，证明图像编辑预训练本身已包含非平凡的视觉理解能力；通过对三款独立训练模型的横向对比，验证了该能力是图像编辑范式的共性而非单一模型特有；

**🔧 技术方法**

使用自然语言提示 + 决定性解码器（灰度深度、标准法向RGB编码、颜色指令分割）；采用光度加权提取深度、仿射对齐、自动轴对准（48种变换）等技术；

**📊 数据集**

NYUv2（深度、法向）、DIODE Indoor/Outdoor（深度）、Cityscapes（语义分割）

**📈 对比分析**

采用与Vision Banana、Marigold等基准相同的评价指标（δ1、AbsRel、RMSE、平均/中位角误差、A11/A22/A30、mIoU、像素准确率）。结果显示：FireRed在NYUv2法向平均误差17.69°（接近Vision Banana）；LongCat在NYUv2深度δ1=0.822；Qwen在DIODE Indoor深度δ1=0.868；Cityscapes 19类mIoU 25.7、7类mIoU 49.5，表明零样本下已取得非零性能。

**⚠️ 局限性**

受限于提示设计的鲁棒性、对颜色编码/RGB格式的严格要求、未涵盖实例分割、对光照/尺度等变化的适应性不足、仅评估静态图像、未测试视频或文本到图像生成器。

---

## 286. Delay-Aware Large-Small Model Collaboration over LEO Satellite Networks

**arXiv ID:** 2605.04565 | [PDF](https://arxiv.org/pdf/2605.04565v1)

**作者:** Mingyu Guo `[一作]` (Pengcheng Laboratory), Liang Li `[通讯]` (Pengcheng Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向低轨卫星网络的延迟感知大‑小模型协同方案，联合优化任务分配和路由策略；

**💡 创新点**

创新点在于：①考虑异构计算资源与有限ISL带宽的真实环境；②将任务分配与路由建模为离散化、部分可观测的多智能体决策问题；③通过离线训练的QMIX‑MARL和在线二分搜索（BS）相结合的BS‑MARL算法实现高效、分布式最优决策；

**🔧 技术方法**

使用技术包括：多智能体强化学习（MARL）、QMIX混合网络、离线政策训练与在线二分搜索、离散化Dec‑POMDP建模、离线离线策略融合等；

**📊 数据集**

采用仿真Walker星座（8平面×8卫星）和基于ResNet18/ResNet101的遥感图像分类任务作为实验数据；

**📈 对比分析**

与小模型本地处理、集中式大模型处理、以及固定α=0.5的等分拆方案比较，实验表明该方案平均服务延迟可降低约31%–32%，在不同数据量和计算能力下均保持优势；

**⚠️ 局限性**

局限性包括：①实验仅基于仿真，缺乏真实卫星平台验证；②模型更新依赖大模型推理后传输更新包，仍存在模型精度与更新开销权衡；③对高动态轨道变化和突发任务到达的鲁棒性尚未完全评估。

---

## 287. Efficient Geometry-Controlled High-Resolution Satellite Image Synthesis

**arXiv ID:** 2605.04557 | [PDF](https://arxiv.org/pdf/2605.04557v1)

**作者:** Vlad Vasilescu `[一作]` (University POLITEHNICA Bucharest), Teodor Costachioiu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对高分辨率卫星图像合成，本文提出了一种利用跳跃连接特征与窗口交叉注意力实现几何控制的方法。

**💡 创新点**

创新点在于只使用模型的 skip 连接特征并结合窗口化交叉注意力，以轻量级方式实现几何控制，避免了全参数训练。

**🔧 技术方法**

采用预训练的 Stable Diffusion 文本到图像扩散模型，并在其编码器上插入冻结副本，利用窗口交叉注意力计算混合权重，最终在解码阶段融合。

**📊 数据集**

实验使用 GeoSynth 提供的 512×512 RGB 卫星图像数据集（每张图对应文本描述和 OSM 贴图），共 3000 张样本。

**📈 对比分析**

与 ControlNet、Uni-ControlNet、SmartControl、GeoSynth 等控制方案比较，WCA 在 FID、LPIPS 及 CLIP‑IQA 的自然度与真实度上表现最佳，同时采样速度和内存占用处于中等水平。

**⚠️ 局限性**

限制主要体现在缺乏可靠的图像与 OSM 对齐度量、预训练 CLIP 在专业场景下表现不足，以及在更复杂多模态控制场景下的可扩展性问题。

---

## 288. Z-Opt: A Near-Optimal Reduced-Complexity Two-Dimensional Grassmannian Constellation

**arXiv ID:** 2605.04545 | [PDF](https://arxiv.org/pdf/2605.04545v1)

**作者:** Kotaro Shigenaga `[一作]` (Yokohama National University), Naoki Ishikawa `[通讯]` (Yokohama National University)

**通讯引用:** 1126 | [OpenAlex ID](https://openalex.org/A5088245484)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了两种在二维Grassmann流形（𝒢(2,1)）上构造Grassmannian星座的方法（S-Opt 与 Z-Opt），并给出了对应的低复杂度检测器；

**💡 创新点**

创新点在于：①S-Opt 利用已知的单位球面最优打点（Bloch 球面）直接获得达到 Fejes–Tóth 上界的星座；②Z-Opt 通过在球面上层叠规则多边形，将优化变量降至 O(√C) 并实现 O(N) 检测复杂度；

**🔧 技术方法**

采用了球面打点理论、Fejes–Tóth 上界推导、KD-tree 最近邻搜索、SVD 预估、结构化投影等技术；

**📊 数据集**

使用模拟数据：Rayleigh 漂移信道下的 M=1、T=2、不同接收天线 N 以及各类星座（Man-Opt、Exp-Map、Cube-Split、Grass-Lattice、S-Opt、Z-Opt）进行 SER 与最小弦距评估；

**📈 对比分析**

与传统方法（GLRT 检测器、Man-Opt、Exp-Map 等）比较，S-Opt 与 Z-Opt 的最小弦距均接近上界，SER 与 GLRT 基准相同；S-Opt 的检测时间为 O(N+log₂C)，Z-Opt 为 O(N)，而 GLRT 为 O(NC)，空间复杂度也从 O(C) 降至 O(√C)；

**⚠️ 局限性**

局限性在于仅针对 𝒢(2,1)（M=1, T=2）情况，Z-Opt 只适用于其特定结构，且对更大 B 或更高维流形的扩展尚未给出；

---

## 289. Active Contact Sensing for Robust Robot-to-Human Object Handover

**arXiv ID:** 2605.04610 | [PDF](https://arxiv.org/pdf/2605.04610v1)

**作者:** Linfeng Li `[一作]`, David Hsu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出主动感知框架，用机器人探测运动与力反馈来区分人类的紧握与偶触，实现稳健的机器人到人类物体递交。

**💡 创新点**

创新性地将贝叶斯线性回归与信息增益优化相结合，利用模型不确定性驱动主动感知动作，并通过鲁棒线性规划判定抓握是否牢固。

**🔧 技术方法**

采用贝叶斯线性回归、信息增益优化、鲁棒线性规划、力传感与运动控制算法。

**📊 数据集**

使用YCB数据集中的30种多样刚性物体。

**📈 对比分析**

与两种基线（阈值释放和负载下降释放）比较，实验中成功率达97.5%，比基线高约30%且主观工作量与可用性相近。

**⚠️ 局限性**

仅限刚性物体、仅依赖力感知、主动探测动作可能略延长递交时间。

---

## 290. UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding

**arXiv ID:** 2605.04543 | [PDF](https://arxiv.org/pdf/2605.04543v1)

**作者:** Yepeng Weng `[一作]` (University of Tokyo), Takehisa Yairi `[通讯]` (University of Tokyo)

**通讯引用:** 3414 | [OpenAlex ID](https://openalex.org/A5012762510)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种统一的验证框架 UniVer，用于在 Speculative Decoding 中同时优化多路草稿（multi‑draft）与多步（multi‑step）的验证，显著提升接受率与推理吞吐量。

**💡 创新点**

创新点在于将树结构的验证建模为条件 Optimal Transport 问题，利用前缀接受概率作为动态缩放因子，实现水平与垂直依赖的联合优化，并通过两阶段（分配 + 决策）并行化计算。

**🔧 技术方法**

使用技术包括：条件 OT、前缀接受概率、贪心采样（Top‑m‑1 + 残留样本）、Top‑down 概率传播、后序遍历决策、以及对比实验中的标准吞吐量评估。

**📊 数据集**

实验数据集涵盖 Spec‑Bench 的六大任务（对话、翻译、摘要、问答、数学推理、检索增强生成）以及在 Vicuna‑7B‑v1.3 与 Llama3.1‑8B‑Instruct 上的多种树结构与采样温度设置。

**📈 对比分析**

与基线方法 RRSw、Traversal（基于递归拒绝采样）以及 Greedy（单步 OT）进行对比，UniVer 在不同树规模、拓扑和温度下平均提升 4.2%–8.5% 的接受长度（τ），吞吐量提升约 7%（相较于 RRSw）且优于 Greedy，证明了其在多维优化上的优势。

**⚠️ 局限性**

局限性包括：在采样温度接近 0 时几乎无优势；验证性能高度依赖于特定的混合采样策略与树结构，未对更广泛的采样策略或自适应树形进行探索。

---

## 291. Right Model, Right Time: Real-Time Cascaded-Fidelity MPC for Bipedal Walking

**arXiv ID:** 2605.04607 | [PDF](https://arxiv.org/pdf/2605.04607v1)

**作者:** Franek Stark `[一作]` (German Research Center for Artificial Intelligence), Frank Kirchner `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种多相全身模型预测控制（MPC），在前期使用完整身体动力学模型，在后期使用简化单刚体模型，以实现双足步态的实时控制；

**💡 创新点**

创新点在于将完整模型与简化模型结合成多相预测框架，并在acados中仅用少量SQP迭代即可实时求解，同时保持足够的预测精度；

**🔧 技术方法**

采用非线性MPC、SQP求解器acados、前向欧拉离散、关节/触碰/力矩约束、Wrench cone约束，并利用Pinocchio+CasADi生成机器人动力学；

**📊 数据集**

实验数据基于MuJoCo仿真，使用18-DOF的HyPer-2机器人进行评估，并未使用公开数据集；

**📈 对比分析**

通过改变模型阶段比例α和SQP迭代次数，比较求解时间、步态高度误差与速度误差；结果表明，使用3次SQP可在10 ms内求解，α≈0.6时步态稳定且误差最小，α>0.7会导致失稳；

**⚠️ 局限性**

局限性包括：未能系统分离α与预测期长度的影响；未在真实硬件上验证；高α值导致预测期缩短并引起失稳；离散化步长对性能影响不完全可控；需要进一步完善多相设计与实验评估。

---

## 292. From Beats to Breaches:How Offensive AI Infers Sensitive User Information from Playlists

**arXiv ID:** 2605.04724 | [PDF](https://arxiv.org/pdf/2605.04724v1)

**作者:** Stefano Cecconello `[一作]` (University of Padova), Pier Paolo Tricomi `[通讯]` (University of Padova)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于音乐播放列表的个人信息推断工具musicPIIrate，并针对该威胁设计了防御框架JamShield。

**💡 创新点**

创新点在于将深度集（DeepSets）与图神经网络（GNN）相结合，充分利用播放列表的集合与图结构信息，显著提升推断精度；同时首次提出对抗性播放列表注入作为轻量级防御手段。

**🔧 技术方法**

使用了深度学习技术，包括多层感知机、深度集、简化图卷积网络（SGC）等；在防御方面实现了特征噪声/消除以及针对攻击模型的对抗性播放列表注入。

**📊 数据集**

实验基于Tricomi等人公开的Spotify数据集，涵盖739名用户、约10,000份播放列表、200,000首歌曲和55,000名艺术家。

**📈 对比分析**

通过5折用户级交叉验证、宏F1评分与传统基线（逻辑回归、随机森林、KNN等）对比，GNN‑DeepSet在15个属性任务中赢得9个基线，平均F1≈0.4–0.6；JamShield在注入1–4份假播放列表后，平均F1下降约10%。

**⚠️ 局限性**

局限性包括仅使用单一Spotify数据集，未验证跨平台通用性；数据假设静态，未考虑用户行为随时间变化；对习惯与人格等属性的推断表现不稳定，防御效果受假播放列表多样性限制。

---

## 293. FaithfulFaces: Pose-Faithful Facial Identity Preservation for Text-to-Video Generation

**arXiv ID:** 2605.04702 | [PDF](https://arxiv.org/pdf/2605.04702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 294. Multi-Level Bidirectional Biomimetic Learning for EEG-Based Visual Decoding

**arXiv ID:** 2605.04680 | [PDF](https://arxiv.org/pdf/2605.04680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 295. When Graph Traversal Meets Structured Preferences: Unified Framework and Complexity Results

**arXiv ID:** 2605.04701 | [PDF](https://arxiv.org/pdf/2605.04701v1)

**作者:** Guozhen Rong `[一作]` (Changsha University of Science and Technology), Yongjie Yang `[通讯]` (Saarland University)

**通讯引用:** 728 | [OpenAlex ID](https://openalex.org/A5032235006)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究将候选人视为图顶点，通过不同的图搜索范式（BFS、DFS、LexBFS、LexDFS、MCS、MNS）解释投票者的偏好排序，并探讨在给定结构限制（如边数或度数上限、树形支持）下，是否存在满足所有排序的图支持；

**💡 创新点**

提出了一个统一的框架，将传统的偏好结构限制（如单峰、单交叉）与图搜索范式联系起来；证明了在边数或度数受限的情况下，六种主流图搜索的支持问题均为NP‑hard，并在DFS上给出了多项式时间算法；

**🔧 技术方法**

使用图搜索的四点排序特征、可归约构造、硬度证明（从 Vertex‑Cover 归约）、DFS 的树形支持判定算法以及 attachment digraph 等工具；

**📊 数据集**

论文主要基于理论证明，并未使用具体实验数据集，全部以图构造与归约为主；

**📈 对比分析**

与已有的单峰单交叉等领域对比，证明了对比问题的复杂性（NP‑hard）与已知多项式可解情况（如单峰性单树、MCS/MNS 的树形支持可多项式解）；在DFS树形支持上给出多项式算法，展示了该类问题的可解性；

**⚠️ 局限性**

局限性：BFS 与 LexBFS 的受限支持问题复杂性仍未确定；研究聚焦于理论复杂度，缺乏对实际数据或多样化偏好结构的实证评估；

---

## 296. ITBoost: Information-Theoretic Trust for Robust Boosting

**arXiv ID:** 2605.04671 | [PDF](https://arxiv.org/pdf/2605.04671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 297. ReflectDrive-2: Reinforcement-Learning-Aligned Self-Editing for Discrete Diffusion Driving

**arXiv ID:** 2605.04647 | [PDF](https://arxiv.org/pdf/2605.04647v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 298. Paraphrase-Induced Output-Mode Collapse: When LLMs Break Character Under Semantically Equivalent Inputs

**arXiv ID:** 2605.04665 | [PDF](https://arxiv.org/pdf/2605.04665v1)

**作者:** Aofan Liu `[一作]` (Peking University), Jingxiang Meng `[通讯]` (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实现了ParaConsist框架，用于评估大型语言模型（LLM）在内容不变的提示变体下的输出一致性，发现模型往往会在提示改写后失去预期的回答模式（输出模式崩溃）。

**💡 创新点**

创新点在于：①构建900条提示变体基准（150个原始查询 × 5种变体）覆盖四种任务类型；②设计Semantic Consistency Score（SCS）将答案一致性、语义相似度和长度稳定性三维度量组合；③系统性地量化提示变体导致的输出模式崩溃，并揭示任务结构比模型身份更能预测崩溃风险。

**🔧 技术方法**

技术方法包括：Prompt变体生成（利用GPT-4o-mini进行词汇替换、句法重构与语义扩展）；多模型推理（GPT-4.1-mini、GPT-4o-mini、Claude Haiku 4.5、Claude Sonnet 4.5、Gemini 2.5 Flash）；基于句子-BERT的语义相似度计算；统计分析（Kruskal‑Wallis、Mann‑Whitney U、Cohen's d）。

**📊 数据集**

使用的数据集为自构的ParaConsist基准，包含70道ARC‑Challenge多项选择题、30条电影评论情感分析、30条AG News分类与20条XSum摘要，总计900条提示；此外引用了公开数据集ARC-Challenge、AG News、XSum、电影评论等。

**📈 对比分析**

通过在上述五种API模型上执行900条提示（共4,500次推理），使用SCS及其组成部分（答案一致性AC、语义相似度SS、长度稳定性CS）评估一致性；结果显示Claude Sonnet 4.5在SCS上最高（0.411），Gemini 2.5 Flash最低（0.256）；整体AC仅约22%，表明大多数变体回答未保留目标标签。

**⚠️ 局限性**

局限性包括：①基准规模相对有限（仅900条，且任务比例不平衡）；②提示变体生成器可能存在偏差；③仅在温度为0的确定性推理下测试，未覆盖随机性导致的波动；④答案一致性采用完整词匹配，可能低估可恢复的标签信息；⑤未覆盖更高能力层模型，需进一步验证跨模型一致性结论。

---

## 299. FAAST: Forward-Only Associative Learning via Closed-Form Fast Weights for Test-Time Supervised Adaptation

**arXiv ID:** 2605.04651 | [PDF](https://arxiv.org/pdf/2605.04651v1)

**作者:** Guangsheng Bao `[一作]` (Zhejiang University), Yue Zhang `[通讯]` (Westlake University)

**通讯引用:** 18116 | [OpenAlex ID](https://openalex.org/A5100333758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FAAST框架，使用前向无梯度的闭式快权重进行测试时监督适应。

**💡 创新点**

创新在于将预训练表示固定后，用Moore-Penrose伪逆闭式求解关联映射，并压缩为快权重，完全消除梯度和上下文存储需求。

**🔧 技术方法**

采用快权重、伪逆求解、谱过滤、插值融合以及读出投影技术，并将其集成到预训练分类器与语言模型中。

**📊 数据集**

实验使用CIFAR‑10、mini‑ImageNet、SST‑2、IMDB、WikiText‑103以及IWSLT2017等数据集。

**📈 对比分析**

与零样本、线性投影、全微调、k‑NN、Softmax内存、LoRA等基线对比，FAAST在准确率/困惑度上与或优于梯度方法，学习时间降低约95%，内存占用减少多达95%。

**⚠️ 局限性**

依赖预训练表示的质量，难以处理需要层次推理或长期依赖的任务，对结构化预测或多模态任务尚未充分验证。

---

## 300. Securing the Web with HSTS-Enforced

**arXiv ID:** 2605.04642 | [PDF](https://arxiv.org/pdf/2605.04642v1)

**作者:** Aaron van Diepen `[一作]` (Delft University of Technology), Fernando Kuipers `[通讯]` (Delft University of Technology)

**通讯引用:** 5869 | [OpenAlex ID](https://openalex.org/A5069179616)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并实现了 HSTS‑Enforced 机制，默认开启 HSTS 并通过 HTTP‑Required 指标（HTTPREQ DNS 记录和预加载列表）实现安全的 HTTP 访问；

**💡 创新点**

创新点在于将 HSTS 从 opt‑in 迁移为 opt‑out，允许站点在安全默认的基础上显式放宽，同时引入可验证的 DNSSEC 记录和逆向预加载列表来防止 TLS 降级攻击并避免用户追踪；

**🔧 技术方法**

使用了 Chromium 浏览器改造、libunbound DNSSEC 验证、PowerDNS/BIND/Unbound DNS 服务器、HTTPS 与 HTTP 服务、以及自定义的 HTTPREQ DNS 记录；

**📊 数据集**

使用自建的 Docker 化测试环境，包括多种 HTTP/HTTPS 配置的 Web 服务器、递归/权威 DNS 服务器，并通过模拟 20 ms RTT 的网络延迟进行测评；

**📈 对比分析**

评估方法包括安全性验证（阻止 TLS‑Stripping、正确处理受限 HTTP）、网络负载测量（DNS 记录大小 ≤ 400 B）、连接延迟测量（典型 3 RTT，最差 6 RTT≈134 ms）以及磁盘/内存占用对比；性能表现接近无显著开销，只有在需要验证 HTTPREQ 时产生可忽略的延迟；

**⚠️ 局限性**

限制主要在于：需要 DNSSEC 的部署（对不支持 DNSSEC 的域或仅使用 HTTP 的站点需要使用预加载列表），预加载列表规模未知，过渡期需要协调运营商与客户端；此外，若客户端不支持 HSTS‑Enforced，仍会受到 TLS‑Stripping 攻击。

---

## 301. Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models

**arXiv ID:** 2605.04638 | [PDF](https://arxiv.org/pdf/2605.04638v1)

**作者:** Mingda Li `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 40501 | [OpenAlex ID](https://openalex.org/A5100418162)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于大语言模型生成任务的梯度基不确定性量化方法SemGrad和HybridGrad，并在多个问答数据集上验证其有效性。

**💡 创新点**

创新点在于：①首次将梯度从参数空间转移到语义空间；②通过Semantic Preservation Score（SPS）识别语义保持的隐藏层和位置；③对输出做熵加权以抑制语言冗余；④设计HybridGrad融合语义梯度与参数梯度以兼顾高偏差与低偏差场景。

**🔧 技术方法**

核心技术包括：梯度基不确定性估计、语义保持嵌入的选择、熵权重重排、语义梯度与参数梯度的加权融合；实现时仅需在一次前向传播后计算梯度，无需采样。

**📊 数据集**

实验使用了三大问答数据集：SciQ、TriviaQA（单答案）和TruthfulQA（多答案），以及三款开源LLM（Llama3.1-Instruct8B、Qwen3-Instruct4B、Mistral-Nemo-Instruct12B）。

**📈 对比分析**

与11种基线方法（包括LN-PE、P(True)、Self-Con、Deg、INSIDE、S.E.、S.D.、M.I.、G-NLL、SAR、ExGrad）对比，SemGrad在多答案数据集TruthfulQA上提升AUROC约+3–7个百分点，HybridGrad在所有数据集均实现最高平均AUROC，表现最稳定。

**⚠️ 局限性**

限制包括：①在低方差单答案场景下，SemGrad偶尔不如参数梯度；②语义保持嵌入的选取仍依赖经验，跨模型迁移需要额外验证；③对非常长文本或多轮对话时的计算开销尚未彻底评估。

---

## 302. Exact Dual Geometry of SOC-ICNN Value Functions

**arXiv ID:** 2605.04722 | [PDF](https://arxiv.org/pdf/2605.04722v1)

**作者:** Kang Liu `[一作]` (Xi'an Jiaotong University), Wei Peng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 256210 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了第二阶锥输入凸神经网络（SOC‑ICNN）的几何性质，证明其一阶和局部二阶信息可以直接由其值函数的对偶变量读取。

**💡 创新点**

创新点在于利用SOC‑ICNN的精确SOCP值函数表示，从对偶角度获得完整的子梯度集合、方向导数以及在非退化区域可解析的梯度和Hessian，提供了白盒推理的几何原语。

**🔧 技术方法**

技术方法包括：SOCP值函数的拉格朗日对偶分析、结构化读取映射、对偶多重解的最小范数选取、凸分析（Danskin定理）以及局部仿射–曲率分解。

**📊 数据集**

实验使用随机初始化的SOC‑ICNN网络进行诊断性测试，并在构造的二维网络上验证退化情况；未使用公开数据集。

**📈 对比分析**

通过与自动微分（autodiff）对比，验证了梯度和Hessian在数值精度（10^-15级误差）上完全一致，并在白盒推理实验中展示了与PyTorch实现相近的性能，且白盒Newton方法的导数构造成本更低。

**⚠️ 局限性**

局限性包括：仅适用于SOC‑ICNN，退化点仍需额外处理；对高维大规模网络的求解与对偶映射计算仍有复杂度挑战；实际应用中需结合具体推理任务进一步优化。

---

## 303. CodeEvolve: LLM-Driven Evolutionary Optimization with Runtime-Enriched Target Selection for Multi-Language Code Enhancement

**arXiv ID:** 2605.04677 | [PDF](https://arxiv.org/pdf/2605.04677v1)

**作者:** Ajay Krishna Borra `[一作]` (Salesforce), Shuchita Singh `[通讯]` (Salesforce)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于大型语言模型的代码性能与质量优化框架 CodeEvolve，结合运行时分析、进化搜索、MCTS、自动代码精炼与多阶段评估。

**💡 创新点**

核心创新包括：基于 JFR 的运行时引导目标选择与加权组件图；先验验证的多目标进化搜索与 MCTS 引导的代码精炼；支持多语言（Java/Apex）的统一评估管线。

**🔧 技术方法**

技术栈涵盖：Java Flight Recorder、加权组件图、进化算法、蒙特卡洛树搜索、LLM 代码生成、静态分析、单元测试、LLM 语义审查。

**📊 数据集**

使用真实企业代码：Salesforce Monolith Java 热点函数集（7 个）以及 Salesforce Apex 代码进行消融实验。

**📈 对比分析**

通过与单次 LLM 调用的 RPBD、基线 OpenEvolve 以及消融配置比较，CodeEvolve 在 7 个热点函数上平均 15.22× 的速度提升，5/7 函数优于 RPBD，消融实验显示最终系统在 20 次迭代中平均产出 19.5 个合法程序、KPI 分数 0.8977。

**⚠️ 局限性**

局限性：依赖评估器（编译、测试、静态检查）的准确性；提示体积受限导致上下文精简；在超大规模代码库中目标与上下文构造难度上升；对简单模式替换的迭代搜索开销可能不划算。

---

## 304. Physical Adversarial Clothing Evades Visible-Thermal Detectors via Non-Overlapping RGB-T Pattern

**arXiv ID:** 2605.04675 | [PDF](https://arxiv.org/pdf/2605.04675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 305. AI-Aided Advancements in Autonomous Underwater Vehicle Navigation

**arXiv ID:** 2605.04672 | [PDF](https://arxiv.org/pdf/2605.04672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 306. ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC

**arXiv ID:** 2605.04709 | [PDF](https://arxiv.org/pdf/2605.04709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 307. From Reach to Insert: Tactile-Augmented Precision Assembly under Sub-Millimeter Tolerances

**arXiv ID:** 2605.04649 | [PDF](https://arxiv.org/pdf/2605.04649v1)

**作者:** Xinpan Meng `[一作]` (Chinese Academy of Sciences), Long Cheng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 471052 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种双阶段的视觉引导到触觉响应的模仿与强化学习框架，用于高精度、紧公差的插装任务；

**💡 创新点**

创新点包括将触觉基准化与分组采样嵌入非对称的演员-评论家结构，实现对接触强度的精细调控，并通过触觉群采样平衡经验分布，显著提升样本效率与安全性；

**🔧 技术方法**

使用扩散策略进行抓取与定位阶段，Soft Actor-Critic 进行插装阶段；采用异构演员-评论家架构、触觉基准化、触觉群采样、人工干预日志、奖励稀疏与评论家预热；

**📊 数据集**

基于5种孔形状（方形、圆形、六边形、L形、三角形）和3种公差（1.5 mm、0.25 mm、0.05 mm）的实验数据，收集100条人类演示用于训练并通过人机干预补充经验；

**📈 对比分析**

与纯模仿学习、仅模仿+强化学习（无触觉）进行对比；在1.5 mm公差下三种方法均接近100%，在0.25 mm下纯模仿仅24%成功，模仿+RL 79%，完整方法 79%；在0.05 mm下纯模仿 0%，模仿+RL 52%，完整方法 67%，同时最大接触力与扭矩分别下降了60%和44%；

**⚠️ 局限性**

仍受限于单件插装、对高精度触觉传感器与昂贵硬件的依赖、在极低公差下成功率未达100%，对不同任务的泛化能力与多件装配的可扩展性尚未验证。

---

## 308. SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning

**arXiv ID:** 2605.04712 | [PDF](https://arxiv.org/pdf/2605.04712v1)

**作者:** Lirui Luo `[一作]` (Peking University), Qing Li `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了Mixture-of-Experts (MoE) 策略在连续控制强化学习中的光谱可塑性衰退，并提出一种基于NTK理论的SPHERE正则化器来缓解该问题。

**💡 创新点**

创新点在于将光谱可塑性定义为经验神经切线核（eNTK）的谱熵有效秩，利用层级分解得到可计算的下界，并设计了对加权专家特征矩阵的Parseval正则化，使其在训练中能持续保持光谱均匀性。

**🔧 技术方法**

主要技术包括NTK/经验NTK理论、谱熵有效秩、块对角近似、Kronecker代理、Parseval正则化以及在MoE策略的最后隐藏层加权特征矩阵上实施光谱收缩操作。

**📊 数据集**

使用MetaWorld（CW10子集）和HumanoidBench（H1五个任务）作为评估数据集。

**📈 对比分析**

与PPO、Top‑K MoE、Dense‑MoE、DS‑MoE及若干可塑性缓解方法（LN、C‑CHAIN、CBP、PW）对比，SPHERE在MetaWorld上平均成功率提升133%，在HumanoidBench上提升50%，同时在整个训练过程中保持更高的谱熵有效秩。

**⚠️ 局限性**

局限性包括：依赖一系列理论近似（如块对角近似、Kronecker代理）；实验仅覆盖连续控制任务的MoE策略；未在大规模预训练基础模型或更复杂任务上验证。

---

## 309. Budget-aware Auto Optimizer Configurator

**arXiv ID:** 2605.04711 | [PDF](https://arxiv.org/pdf/2605.04711v1)

**作者:** Kang Liu `[一作]` (Xi'an Jiaotong University), Jianchen Hu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2598 | [OpenAlex ID](https://openalex.org/A5100389881)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于预算的块级优化器配置框架（BAOC），通过采样梯度统计量并构造风险评估，求解混合整数线性规划，在给定显存和更新时延预算下为每个参数块分配合适的优化器配置。

**💡 创新点**

首次将优化器配置视为可分配资源，并提出了块级梯度异质性指标和对应的风险函数，利用预算约束的MILP实现自动化、可解释的配置决策。

**🔧 技术方法**

梯度稀疏采样、指数移动平均统计、梯度方向稳定性、尺度各向异性、量化误差评估等指标；混合整数线性规划求解；多种优化器（AdamW、Adam、SGD、Adafactor等）与不同位宽（32/16/8位）组合。

**📊 数据集**

ViT（ImageNet-1K）、GPT-2小模型（Alpaca、GSM8K）、T5-base（Alpaca、GSM8K）、UNet扩散模型、Llama-3.2-1B/3B（Alpaca、GSM8K）。

**📈 对比分析**

与多种基线（AdamW8/16、Adam-mini、GaLore、Adafactor、SPAM、Muon、COSMOS）对比；在显存约为原始 AdamW16 的 50% 时，BAOC 在准确率/困惑度/损失上保持与全状态优化器相当，同时显存占用减半、训练时延提升不大；在更大显存预算下性能可进一步提升。

**⚠️ 局限性**

风险模型为线性加和，未考虑块间交互；配置空间有限，未覆盖最新优化器；在线重规划可能导致训练不稳定；与分布式训练框架（ZeRO、FSDP）集成尚未评估；未提供最终验证指标的直接优化。

---

## 310. Ultra Low-Power SDM-based Circuit-Switching for Networks-on-Chip

**arXiv ID:** 2605.04679 | [PDF](https://arxiv.org/pdf/2605.04679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 311. Differentiable Chemistry in PINNs for Solving Parameterized and Stiff Reaction Systems

**arXiv ID:** 2605.04708 | [PDF](https://arxiv.org/pdf/2605.04708v1)

**作者:** Miloš Babić `[一作]` (CD Laboratory for Physics-driven Machine Learning in Industrial Applications), Stefan Posch `[通讯]` (CD Laboratory for Physics-driven Machine Learning in Industrial Applications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

结合可微化学求解器与 Physics‑Informed Neural Networks（PINN），构建端到端可训练框架，用于求解含详细化学动力学的刚性反应系统；

**💡 创新点**

在 PINN 中首次引入可微化学求解器、残差加权、硬约束等多项改进，解决了刚性反应扩散 PDE 的收敛与精度问题；

**🔧 技术方法**

使用 Physics‑Informed Neural Networks、可微化学后端（Reactorch）、残差加权策略、参数化网络结构，以及 Adam 优化器；

**📊 数据集**

采用氢燃烧的一步 Arrhenius 机制生成的初值、边值、反应‑扩散 PDE 三类问题的数值参考解作为数据集；

**📈 对比分析**

与 vanilla PINN 以及基于因果关系的 loss‑scaling PINN 进行对比，使用 MAE 与相对 L₂ 误差评估；所提框架在所有测试场景下误差均显著降低，性能优越；

**⚠️ 局限性**

仅在一阶 Arrhenius 机制上验证，未涉及多步或详细化学机制；可扩展性、训练复杂度等仍需进一步研究。

---

## 312. Threshold-Guided Optimization for Visual Generative Models

**arXiv ID:** 2605.04653 | [PDF](https://arxiv.org/pdf/2605.04653v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 313. Entropy and Distributed Source Coding of Connected Soft Random Geometric Graphs

**arXiv ID:** 2605.04703 | [PDF](https://arxiv.org/pdf/2605.04703v1)

**作者:** Oliver Baker `[一作]` (University of Bristol), Carl P. Dettmann `[通讯]` (University of Bristol)

**通讯引用:** 2949 | [OpenAlex ID](https://openalex.org/A5050617429)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在连接阈值以上的软随机几何图（SRGG）的分布式压缩，建立了SRGG的Slepian-Wolf速率区域，证明了新的极限定理和渐近均分性质。

**💡 创新点**

创新点在于首次在信息论背景下研究了SRGG的分布式压缩，特别是在节点连接范围随节点数量增加而减小的情况下。

**🔧 技术方法**

使用了信息谱理论和随机分箱技术来研究SRGG的熵和分布式压缩。

**📊 数据集**

使用了独立均匀分布的点集作为SRGG的节点，具体的连接函数形式为p_n(r) = p(r/s(n))，其中s(n)是一个递减的稀疏性序列。

**📈 对比分析**

通过建立极限定理和渐近均分性质，证明了SRGG的熵速率与其信息内容以高概率收敛，比较了不同编码器数量下的分布式压缩性能。

**⚠️ 局限性**

限制在于尚未研究在压缩单元数量与节点数量成比例增长的情况下的分布式压缩算法，以及如何将结果扩展到有损分布式压缩的情形。

---

## 314. Average Attention Transformers and Arithmetic Circuits

**arXiv ID:** 2605.04683 | [PDF](https://arxiv.org/pdf/2605.04683v1)

**作者:** Lena Ehrmuth `[一作]` (Leibniz University Hanover), Laura Strieker `[通讯]` (Leibniz University Hanover)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

在理论上建立了平均注意力Transformer与无穷通路算术电路（R0[]）之间的对应关系

**💡 创新点**

首次将Transformer的表达能力与算术电路类相匹配，并证明Transformer可以模拟任意R0[]电路

**🔧 技术方法**

使用半环、环、域的抽象代数框架、注意力函数设计和电路编码技术

**📊 数据集**

无数据集，纯理论证明

**📈 对比分析**

通过形式化证明而非实验比较，展示Transformer能在2K层实现深度K的电路模拟，理论复杂度保持在多项式时间

**⚠️ 局限性**

局限在于仅适用于理论模型，未考虑实际实现与噪声、仅支持平均/硬注意力，且不支持仅使用前馈网络的激活函数

---

## 315. CHE-TKG: Collaborative Historical Evidence and Evolutionary Dynamics Learning for Temporal Knowledge Graph Reasoning

**arXiv ID:** 2605.04652 | [PDF](https://arxiv.org/pdf/2605.04652v1)

**作者:** Shuai-long Lei `[一作]` (University of Science and Technology Beijing), Xu-Cheng Yin `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 4160 | [OpenAlex ID](https://openalex.org/A5074514262)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 CHE-TKG 通过分别构建历史证据图和演化动力学图，联合学习两种互补的预测视角，实现时序知识图谱推理。

**💡 创新点**

创新点在于双视图协同学习框架、关系分解与对比对齐机制，显式捕获并利用历史证据与演化动力学的互补预测信号。

**🔧 技术方法**

采用图神经网络（GAT、GCN+GRU）、关系分解、多视图编码器、对比学习（InfoNCE）和 ConvTransE 等技术。

**📊 数据集**

在 ICEWS14s、ICEWS18、ICEWS05-15、GDELT 四大时序知识图谱基准上进行评测。

**📈 对比分析**

与 RE-NET、RE-GCN、RETIA、TiRGN、LogCL、DyMemR、HisRES 等强基线对比，CHE-TKG 在 MRR、Hits@1/10 上均取得显著提升（约 3%–4%）。

**⚠️ 局限性**

模型结构复杂，训练成本高；对规则检索的依赖可能受限于规则质量；在极端稀疏或非规则演化场景下性能提升有限。

---

## 316. A Framework of Secure Source Coding using Mutual Information Security Criterion: Universal Coding, Strong Converse Theorem

**arXiv ID:** 2605.04720 | [PDF](https://arxiv.org/pdf/2605.04720v1)

**作者:** Yasutada Oohama `[一作]` (University of Electro-Communications), Bagus Santoso `[通讯]` (University of Electro-Communications)

**通讯引用:** 174 | [OpenAlex ID](https://openalex.org/A5031357176)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

构建了一个固定长度源加密框架，利用互信息衡量泄露，给出了可靠与安全传输的必要充分条件，并证明了强逆定理。

**💡 创新点**

创新点在于：①把互信息作为安全准则；②在该准则下证明了强逆定理；③给出了对任意源分布与密钥分布都有效的通用加密/解密方案。

**🔧 技术方法**

采用信息谱方法、Birkhoff‑von Neumann 定理、类型分析以及随机仿射编码来实现编码与加密，并进行误差与信息泄露的指数上界估计。

**📊 数据集**

论文为理论研究，未使用具体数据集，仅基于离散无记忆源和无噪声通道的概率模型。

**📈 对比分析**

通过对比传统最大互信息或猜测概率等安全指标，证明在任意源/密钥分布下能实现最优指数性能；与已有结果相比，扩展了可实现的安全级别并提供了通用性。

**⚠️ 局限性**

局限性：仅适用于离散无记忆源和无噪声通道；未考虑失真或更一般的源模型；实现细节和复杂度未在实验层面验证。

---

## 317. Every Step Counts: Step-Level Credit Assignment for Tool-Integrated Text-to-SQL

**arXiv ID:** 2605.04719 | [PDF](https://arxiv.org/pdf/2605.04719v1)

**作者:** Yaxun Dai `[一作]` (Soochow University), Pingfu Chao `[通讯]` (Soochow University)

**通讯引用:** 502 | [OpenAlex ID](https://openalex.org/A5088523664)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出FineStep框架，实现工具增强型Text-to-SQL的细粒度步骤级信用分配；

**💡 创新点**

创新点在于结合独立的过程奖励、结果奖励折扣、过程平滑和多维优势估计，实现对中间推理步骤的精准奖励分配；

**🔧 技术方法**

采用强化学习中的GRPO改进策略、工具调用回调、步骤级优势估计以及正则化KL损失等技术；

**📊 数据集**

在BIRD和Spider（包括其变体Spider-Syn、Spider-Realistic、Spider-DK）数据集上进行实验；

**📈 对比分析**

与GRPO、DAPO、GSPO、GIGPO等基线相比，FineStep在4B、8B、30B-A3B模型上平均提升EX约3.25%、2.00%、1.60%，并在多种解码策略和难度等级下表现更佳；

**⚠️ 局限性**

局限性在于框架专注于SQL的结构化特点，过程奖励主要基于规则和执行结果，缺乏对模糊或非典型推理路径的灵活性，未来需引入学习型过程奖励模型并拓展至更广泛的推理领域。

---

## 318. On Minimum CADs for Algebraic Sets in Dimension Three

**arXiv ID:** 2605.04718 | [PDF](https://arxiv.org/pdf/2605.04718v1)

**作者:** Lucas Michel `[一作]` `[通讯]` (University of Liège), Lucas Michel (University of Liège)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

证明在三维空间中，所有闭且有幕形（尤其是代数集）的有限族都有最小的 Cylindrical Algebraic Decomposition (CAD)。

**💡 创新点**

首次给出三维空间内代数集存在最小 CAD 的正性存在定理，并提出闭且幕形作为充分条件。

**🔧 技术方法**

利用 CAD 减少、树重写、连通性与局部边界连通性、拓扑粘合等数学工具构造证明。

**📊 数据集**

本研究为纯理论性质证明，不使用实验数据集。

**📈 对比分析**

论文未进行实验对比，仅在结论中指出未来实现最小 CAD 的算法与投影算子设计方向。

**⚠️ 局限性**

结论仅适用于三维空间，四维及更高维的情况存在反例且仍未解决。

---

## 319. On the Complexity of Minimum Riesz s-Energy Subset Selection in Euclidean and Ultrametric Spaces

**arXiv ID:** 2605.04715 | [PDF](https://arxiv.org/pdf/2605.04715v1)

**作者:** Michael T. M. Emmerich `[一作]` (University of Jyvaskyla), André Deutz `[通讯]` (Leiden University)

**通讯引用:** 2324 | [OpenAlex ID](https://openalex.org/A5085426879)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了离散Riesz s能量子集选择问题的计算复杂性，证明在一般度量空间下NP‑hard，并进一步证明在欧氏平面上若指数s为输入时仍NP‑hard；在有限ultrametric空间则可通过动态规划在O(nk²)时间内求得最优子集；同时阐明大s极限时能量最小化与最小对距（MPD）问题的等价关系。

**💡 创新点**

创新点在于：①将已知的度量空间NP‑hard性推广到欧氏平面，并利用能量阈值技术给出完整的归约；②首次给出ultrametric空间的精确动态规划算法，揭示层级度量的可解性；③阐释Riesz能量在大s极限下收敛于MPD，提供了两种不同目标之间的桥梁。

**🔧 技术方法**

技术手段包括：多项式约简（从k‑Clique或几何独立集归约）、欧氏距离阈值与对数估计的能量分离、ultrametric树结构的分治递推、动态规划状态压缩、以及对s、k之间关系的解析式上界。

**📊 数据集**

该工作为理论性研究，不涉及实验数据集，所有结果均通过证明与构造实例得出。

**📈 对比分析**

在理论层面，本文通过归约证明了欧氏平面下的NP‑hard性；在可解性方面，ultrametric空间给出O(nk²)的动态规划算法；并通过极限分析说明当s→∞时问题与MPD等价。未进行实验对比或性能评测。

**⚠️ 局限性**

局限性包括：①固定指数s（s>0）时欧氏空间的NP‑hard性仍未解决；②一维有序欧氏点集的Riesz能量问题仍开放；③未给出近似或参数化算法，仅提供精确复杂度与可解性边界。

---

## 320. Feature importance analysis for patient management decisions

**arXiv ID:** 2605.04666 | [PDF](https://arxiv.org/pdf/2605.04666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 321. Vol-Mark: A Watermark for 3D Medical Volume Data Via Cubic Difference Expansion and Contrastive Learning

**arXiv ID:** 2605.04705 | [PDF](https://arxiv.org/pdf/2605.04705v1)

**作者:** Jiangnan Zhu `[一作]` (Kyushu University), Yujie Gu `[通讯]` (Kyushu University)

**通讯引用:** 4546 | [OpenAlex ID](https://openalex.org/A5015909551)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种针对3D医学体数据的可逆零水印方案Vol-Mark，用于在远程医疗中保护数据所有权和完整性。

**💡 创新点**

创新点在于结合对比学习的体积特征提取器与三维整数小波变换的立方差值展开(c-DE)嵌入，实现低失真可逆嵌入和双重验证（完整性+所有权）。

**🔧 技术方法**

采用3D ResNet‑18+对比学习提取特征，Henon映射加密，3D整数小波变换和差值展开进行嵌入，主观上使用多种统计测试。

**📊 数据集**

在MSD（Medical Segmentation Decathlon）数据集的Brain Tumours、Liver、Pancreas三类任务上进行训练和测试。

**📈 对比分析**

与现有3D-DTCWT和ADCL-ZW等方法在Gaussian噪声、JPEG压缩、几何变换及其混合攻击下进行对比，Vol-Mark在大多数攻击情形下ACC>0.90、NC>0.9、BER<0.1，显著优于基线。

**⚠️ 局限性**

局限性包括对极端尺寸变形（如大幅裁剪/旋转）的鲁棒性仍有限，对不同体素格式或压缩率的适应性需进一步验证。

---

## 322. Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization

**arXiv ID:** 2605.04700 | [PDF](https://arxiv.org/pdf/2605.04700v1)

**作者:** Zheng Fang `[一作]` (Wuhan University), Zhijin Ge `[通讯]` (Xidian University)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5101417184)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了音频语言模型（ALM）的 jailbreak 过程，分析了 token 级梯度分布并提出了基于 token‑aware 的稀疏梯度优化方法 TAGO。

**💡 创新点**

创新点在于发现梯度能量高度不均匀，仅少数 token 贡献大，并基于此提出在每一步只保留高能量 token 的稀疏更新；同时设计了模型兼容的前缀模板并抑制模型提前终止。

**🔧 技术方法**

采用了 token‑aligned 梯度测量、稀疏梯度掩码、教师强制前缀匹配、EOS 终止抑制、早停策略以及白盒梯度优化技术。

**📊 数据集**

使用的评估数据集为 AdvBench‑50（100 条 TTS 转换的有害音频）和 HarmBench（200 条有害指令转语音）。

**📈 对比分析**

与 Direct、SpeechGuard、AdvWave、Post‑hoc prune 等基线对比，TAGO 在 Qwen3‑Omni、Qwen2.5‑Omni、LLaMA‑Omni 上均保持 86%–87% 的 ASR_l 与 100% 的 ASR_r，且仅保留 25% 的 token 仍能保持高效，证明稀疏更新并不影响攻击效果。

**⚠️ 局限性**

局限性包括前缀模板可能无法完全匹配所有有害查询、缺乏自适应目标机制，且实验仅在白盒场景下进行，未验证黑盒或多样化 TTS 的稳健性。

---

## 323. From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models

**arXiv ID:** 2605.04678 | [PDF](https://arxiv.org/pdf/2605.04678v1)

**作者:** Yihan Lin `[一作]` (Renmin University of China), Jing Zhang `[通讯]` (Renmin University of China)

**通讯引用:** 17861 | [OpenAlex ID](https://openalex.org/A5100345321)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了在视觉‑语言‑动作（VLA）模型中使用潜在动作监督的效果，提出两种监督视角（图像‑基于潜在动作与动作‑基于潜在动作）并在统一基线下实现四种集成策略，随后在模拟与真实机器人任务上进行对比实验。

**💡 创新点**

创新点在于将潜在动作监督划分为“轨迹正则化”和“目标空间统一”两大视角，并统一实现四种监督策略；同时揭示了图像‑基于潜在动作在长周期与场景泛化上优越、动作‑基于潜在动作在复杂运动上更好，并证明直接监督离散Token比连续表示更有效。

**🔧 技术方法**

采用Qwen3‑VL‑2B作为VLM骨干，结合VQ‑VAE形式的图像/动作潜在动作模型；通过隐式对齐、显式直接解码、显式条件解码和动作‑到‑Token映射四种技术实现潜在动作监督，并对比离散Token与连续回归的监督方式。

**📊 数据集**

实验使用LIBERO‑Long、RoboTwin 2.0两个仿真基准以及JAKA实物机器人在抓取、堆叠与放置等任务。

**📈 对比分析**

在统一基线下对四种策略进行公平对比：图像‑基于潜在动作在长周期任务提升约8–10%；动作‑基于潜在动作在复杂运动任务提升约17%；离散Token监督相较连续回归平均提升2–3%；在混合任务联合训练中显著减少负迁移，提升整体性能。

**⚠️ 局限性**

研究仅覆盖部分潜在动作监督与集成方案，未尝试更强的潜在动作模型；实验仅在单臂机器人平台进行，缺乏对多臂或更复杂环境的验证。

---

## 324. Contact Matrix: Enhancing Dance Motion Synthesis with Precise Interaction Modeling

**arXiv ID:** 2605.04662 | [PDF](https://arxiv.org/pdf/2605.04662v1)

**作者:** Xuhai Chen `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 36067 | [OpenAlex ID](https://openalex.org/A5100712539)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一个两阶段的双人舞反应运动生成框架，包含分体编码+共享解码的PartFusion‑VQ和联合生成运动与接触矩阵的RCDiff。

**💡 创新点**

创新点在于：①使用体部特定代码表与共享解码器实现有限数据下的姿态一致性；②在扩散模型中同时生成运动和接触矩阵，并在推理时通过接触引导修正采样，显著提升交互真实性。

**🔧 技术方法**

技术手段包括VQ‑VAE分体编码与共享解码、Transformer‑based 条件扩散模型、接触矩阵的二值化预测与得分引导、SMPL‑X姿态表示以及Librosa提取的音乐特征。

**📊 数据集**

实验使用新构建的DD100双人舞数据集（约3.24小时、10种舞蹈风格），并在此数据集上训练与评估。

**📈 对比分析**

与Duolando、InterFormer、ReGenNet等方法在DD100上对比，RCDiff实现了FID_k 8.89、FID_cd 8.01、BED 0.4606等指标的显著提升，显示出更真实的运动质量、更好的交互协调和更精准的节奏同步。

**⚠️ 局限性**

局限性包括对极端动作的生成仍不稳定；接触预测仅为二值化，未捕捉接触强度与持续时间；模型对长序列的鲁棒性还有提升空间。

---

## 325. CAST: Mitigating Object Hallucination in Large Vision-Language Models via Caption-Guided Visual Attention Steering

**arXiv ID:** 2605.04641 | [PDF](https://arxiv.org/pdf/2605.04641v1)

**作者:** Qiming Li `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 16932 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

提出了一种训练无关、可插拔的视觉注意力引导方法 CAST，利用 caption 查询对注意力头的激活模式差异来提升 LVLM 的细粒度视觉感知，从而显著降低对象幻觉。

**💡 创新点**

创新点在于：①首次揭示 caption 与非 caption 查询对视觉注意力模式的显著差异；②通过无训练的探针识别 caption‑guided 注意力头；③预先计算注意力输出偏移向量，在推理时对这些头进行方向性调整，实现低延迟的幻觉抑制。

**🔧 技术方法**

核心技术包括：Transformer 关注机制分析、二分类探针（SVM）用于识别 caption‑guided 头；计算注意力输出差异向量（shift vector）；推理时对选定头加上调节因子 α 的向量偏移；多量化评价与超参数搜索。

**📊 数据集**

使用 5 个主流基准：POPE、CHAIR、MMHal-Bench、MHumanEval（GPT‑4/Human 评估）和 MME；在 LLaVA‑1.5‑7B、Qwen‑VL‑Chat、LLaVA‑NeXT 等 3 种 LVLM 上进行验证；探针训练样本来源为 1000 张多任务 VQA 图像与对应 caption。

**📈 对比分析**

与 VCD、OPERA、PAI、VTI 等现有无训练或解码策略对比，CAST 在 5 个基准上平均提升 5–6% 的准确率/召回率、减少 6–7% 的幻觉率；并且在推理时仅略微增加 5–10% 的延迟，保持了文本生成的流畅度和一致性。

**⚠️ 局限性**

局限性包括：①需要先在特定 LVLM 上预先计算偏移向量，适配性受模型架构限制；②超参数 α 与 K 的选择对性能敏感；③对极端非 caption 查询或完全不同的视觉场景可能效果不佳；④仍未能完全消除所有类型的幻觉，仅侧重对象级幻觉。

---

## 326. Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation

**arXiv ID:** 2605.04723 | [PDF](https://arxiv.org/pdf/2605.04723v1)

**作者:** Shereen Elsayed `[一作]` (University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (University of Hildesheim)

**通讯引用:** 17345 | [OpenAlex ID](https://openalex.org/A5039470755)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了ConvRec，一种基于层次卷积的属性与上下文感知序列推荐模型。

**💡 创新点**

创新点在于使用分层下采样卷积替代自注意力，保持线性计算与存储复杂度，同时通过逐层聚合邻近交互实现全局序列表示。

**🔧 技术方法**

采用多层1D卷积、残差连接、层归一化、时间间隔编码以及全连接投影，训练使用二元交叉熵和负采样。

**📊 数据集**

使用四个亚马逊电商数据集：Beauty、Games、Fashion、Men（含视觉特征、类别、品牌等属性）。

**📈 对比分析**

与BERT4Rec、SASRec、TiSASRec、CosRec、S^3-Rec、SASRec++、CARCA、ProxyRCA等基准对比，ConvRec在HR@10和NDCG@10上均优于所有基线，提升幅度约1–7%。

**⚠️ 局限性**

局限性在于对极长序列的下采样可能导致信息丢失，且在仅使用单一卷积层时性能下降，未来需探索更灵活的分辨率或混合注意力机制。

---

## 327. Not Every Subject Should Stay: Machine Unlearning for Noisy Engagement Recognition

**arXiv ID:** 2605.04713 | [PDF](https://arxiv.org/pdf/2605.04713v1)

**作者:** Alexander Vedernikov `[一作]` `[通讯]`, Alexander Vedernikov

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在已训练的参与度识别模型上实现基于主体的机器遗忘，通过先筛选平均训练损失最高的主体，然后对模型头部进行轻量级更新，比较与完整重训练的效果。

**💡 创新点**

将机器遗忘框架应用于主体级别的后置数据清理；提出用平均训练损失对主体进行风险评分，并以oracle retraining作为对照，验证轻量级头部更新在降低成本的同时能显著恢复重训练收益。

**🔧 技术方法**

使用TCCT‑Net结构，基于交叉熵平均损失进行主体排序；对已训练模型进行特征提取器冻结、头部正则化、对抗遗忘损失的轻量级再训练。

**📊 数据集**

使用DAiSEE和EngageNet两大以主体为单位的参与度识别基准数据集。

**📈 对比分析**

通过对比baseline、unlearned、oracle及naive finetune的准确率评估；在K=3场景下，unlearned以0.22×的计算成本恢复了89.3%（DAiSEE 92.5%）的oracle提升，准确率与oracle相近；在DAiSEE同样达到92.5%的恢复。

**⚠️ 局限性**

仅针对单一backbone、单一评分规则、有限的忘记集(K∈{1,3,5})，遗忘效果高度依赖于主体选择质量；未提供精确删除保证，且在更大规模或不同任务上的表现尚不确定。

---

## 328. UVMarvel: an Automated LLM-aided UVM Machine for Subsystem-level RTL Verification

**arXiv ID:** 2605.04704 | [PDF](https://arxiv.org/pdf/2605.04704v1)

**作者:** Junhao Ye `[一作]` (Southeast University), Zhe Jiang `[通讯]` (Southeast University)

**通讯引用:** 19841 | [OpenAlex ID](https://openalex.org/A5033815877)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 UVMarvel，一个基于大型语言模型的自动化 UVM 验证框架，能够从子系统级 RTL 及其规格自动生成完整的 UVM 测试平台，并通过信号跟踪与 Verilog 修补提升刺激覆盖率。

**💡 创新点**

创新点在于：① 引入统一的中间表示（IR）与可扩展的总线协议库，使 LLM 能够准确理解多协议子系统的结构与握手逻辑；② 通过信号跟踪与 Verilog 修补生成“过滤 DUT”，显著缩小 LLM 关注范围，提升覆盖率；③ 综合利用多模型 LLM（GPT‑4.1、Claude‑4.5、Gemini‑2.5）验证鲁棒性。

**🔧 技术方法**

主要技术包括：大型语言模型（ChatGPT API、Claude、Gemini）、中间表示（IR）与总线协议库、信号跟踪器、Verilog 修补模板、覆盖率分析与迭代刺激生成。

**📊 数据集**

使用了六个工业级子系统基准（Watchdog、Pwrctrl、Cordic、IdleControl、LPctrl、Busremap），每个基准均包含 APB、AHB、AXI、P‑Channel、Q‑Channel 等异构总线接口。

**📈 对比分析**

与现有 IP 级自动化工具（MEIC、UVM²）以及仅随机刺激的基线相比，UVMarvel 在平均代码覆盖率上提升至 95.65%，相当于专家手工流的工业级覆盖水平，并在达到 90% 覆盖时比人工加速约 20 倍（4.5 小时 vs 数日）。

**⚠️ 局限性**

局限性：对复杂协议（如 AXI）仍需外部协议库支持；覆盖率评估主要基于代码覆盖，对功能覆盖的支持有限；框架对 LLM 的依赖可能导致对模型更新的敏感性；当前验证范围限定于子系统级 RTL，尚未验证在更大 SoC 整合层面的可扩展性。

---

## 329. A Separation Between Optimal Demand-Oblivious and Demand-Aware Network Throughput

**arXiv ID:** 2605.04699 | [PDF](https://arxiv.org/pdf/2605.04699v1)

**作者:** Matthias Bentert `[一作]` (Technische Universität Berlin), Stefan Schmid `[通讯]` (Technische Universität Berlin)

**通讯引用:** 17595 | [OpenAlex ID](https://openalex.org/A5019006329)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究需求感知网络拓扑下的吞吐量，提出两种吞吐量定义并在直接与多跳路由模型下分析其可实现的上界与下界，证明需求感知拓扑可在最坏情况下实现至少5/8的吞吐量，优于传统无需求感知拓扑的1/2上界；同时讨论相应问题的计算复杂度，证明弱吞吐量求解为NP‑难，其它两种吞吐量求解为多项式可解，并给出开放性猜想。

**💡 创新点**

首次对需求感知吞吐量进行系统性定义与比较，展示需求感知拓扑在理论上可显著超越无需求感知拓扑；提供首次关于需求感知吞吐量与计算复杂度的分离结果。

**🔧 技术方法**

主要采用理论分析、组合优化与复杂度证明技术，对吞吐量定义进行数学建模并推导极限与复杂度结论。

**📊 数据集**

无；本文为纯理论研究，不涉及实验数据集。

**📈 对比分析**

与已知的需求无关吞吐量上界（≈1/2）对比，理论上证明需求感知吞吐量可达到至少5/8，显示显著性能提升。

**⚠️ 局限性**

未给出算法实现与实验验证，关于需求感知吞吐量（非弱）是否为NP‑难仍为未解开放问题；研究仅覆盖理论范围，缺乏实际系统评估。

---

## 330. Evidence-based anomaly detection in clinical domains

**arXiv ID:** 2605.04664 | [PDF](https://arxiv.org/pdf/2605.04664v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 331. Gray-Box Poisoning of Continuous Malware Ingestion Pipelines

**arXiv ID:** 2605.04698 | [PDF](https://arxiv.org/pdf/2605.04698v1)

**作者:** Jan Dolejš `[一作]` (Czech Technical University in Prague), Róbert Lórencz `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 374 | [OpenAlex ID](https://openalex.org/A5071351394)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了灰盒持续恶意软件采集管道的毒化攻击，利用功能保持的 IAT 与段注入生成可执行毒化样本，并提出基于同质集成的预过滤器进行防御。

**💡 创新点**

创新点在于：①将 IAT 与段注入等结构级改动与功能保持的攻击结合，生成高效但可执行的毒化样本；②设计同质集成（基于 Byte 分布、内容和结构三模型）作为预过滤器，显著抑制毒化影响。

**🔧 技术方法**

使用技术包括：PE 改造框架（GAMMA）、LightGBM 与 Random Forest 分类器、TLSH 去重、同质集成过滤器，以及对比实验中的特征提取与模型训练流程。

**📊 数据集**

数据集：整合原始 PE（含 52,433 正例和 52,433 负例）与 EMBER2024 特征数据，最终得到约 104,866 样本的混合数据集；实验使用 12,500 样本子集（1,400 目标恶意样本）。

**📈 对比分析**

比较方法：对基线 LightGBM 防御模型、攻击代理 Random Forest 以及三种单独过滤器和组合过滤器进行性能评估。毒化效果以召回率下降（从 0.947 降至 0.860）和过滤率衡量，组合过滤器能过滤 95.6% 的高强度毒化样本，同时保持 84.5% 的干净数据。

**⚠️ 局限性**

limitation：实验仅基于较小子集；假设攻击者与防御者特征分布相似；仅考察单次毒化事件，未模拟持续或多轮毒化；未验证在更复杂模型（如深度集成、图模型）下的迁移性。

---

## 332. Learning Time-Inhomogeneous Markov Dynamics in Financial Time Series via Neural Parameterization

**arXiv ID:** 2605.04690 | [PDF](https://arxiv.org/pdf/2605.04690v1)

**作者:** Jan Rovirosa `[一作]` (University of Wisconsin -- Madison), Jesse Schmolze `[通讯]` (University of Wisconsin -- Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用神经网络对时间不齐的马尔可夫转移矩阵进行参数化，从而在高分辨率金融时间序列中学习可解释的、时间变异的转移动态。

**💡 创新点**

创新点在于：①将神经网络严格约束为生成有效的概率矩阵，实现深度学习与经典马尔可夫理论的“结构第一”融合；②用转移矩阵本身进行诊断（行异质性、熵、Dobrushin系数）并将Chapman–Kolmogorov方程转化为局部一致性诊断工具；③在高稀疏度下替代传统计数估计。

**🔧 技术方法**

技术包括：多层感知机（MLP）加softmax输出构成条件概率网络；对输入状态一热编码与特征向量拼接；对输出行做概率归一化；可选平滑标签训练；Adam优化、早停；对训练集进行分箱、特征标准化、特征筛选；使用行异质性、熵、Dobrushin系数及CK一致性作为诊断指标。

**📊 数据集**

实验使用美国上市公司JPM（JP摩根）日频收盘价及多种公开与商业金融指标（市场宏观、信用利差、公司基本面等）做特征，采用单资产日级别数据，并在训练/验证/测试上按时间顺序切分。

**📈 对比分析**

比较方法：与基于计数的马尔可夫估计、平滑后计数、以及仅以特征为输入的无状态基线模型进行对比。性能指标主要是诊断性（行异质性、熵、CK KL）和预测性（ΔNLL、ECE）。结果表明：计数估计在高分辨率下几乎完全稀疏；状态条件模型获得显著的行异质性（平均0.0073）并能捕捉波动率与转移熵的负相关（r≈-0.62）；CK诊断在高波动窗口显示显著不一致；预测性能提升有限（最大ΔNLL≈0.025），但已优于无状态基线。

**⚠️ 局限性**

局限性：①单资产、低频特征导致信噪比低，预测提升有限；②对高分辨率分箱导致稀疏问题仍然存在，只能通过神经网络平滑解决；③当前仅为离散时间马尔可夫模型，无法直接处理连续时间或高频数据；④CK诊断的解释仍需进一步理论验证。

---

## 333. Cognitive Alignment Drives Attention: Modeling and Supporting Socially Shared Regulation in Pair Programming

**arXiv ID:** 2605.04639 | [PDF](https://arxiv.org/pdf/2605.04639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 334. HEXST: Hexagonal Shifted-Window Transformer for Spatial Transcriptomics Gene Expression Prediction

**arXiv ID:** 2605.04682 | [PDF](https://arxiv.org/pdf/2605.04682v1)

**作者:** Keunho Byeon `[一作]` (Korea University), Jin Tae Kwak `[通讯]` (Korea University)

**通讯引用:** 3359 | [OpenAlex ID](https://openalex.org/A5036627754)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出HEXST，一种能够从H&E切片预测空间基因表达的Hexagonal Shifted-Window Transformer；

**💡 创新点**

将六角形采样结构嵌入Transformer（HexMSA与HexRoPE），并通过偏差匹配目标和转录组先验对抗过平滑；

**🔧 技术方法**

利用hexagonal窗口自注意力、HexRoPE、预训练的单细胞基础模型进行特征对齐、偏差匹配损失和多尺度shifted窗口机制；

**📊 数据集**

在七个公开的SpaRED空间转录组数据集上进行评估；

**📈 对比分析**

与STNet、Hist2ST、EGNv1/2、TCGN、NH2ST、PEKA等基线模型比较，指标包括PCC_F、PCC_S、MI_F、AUC_0vNZ、AUC_Q50，HEXST在所有指标上均显著优于对手；

**⚠️ 局限性**

局限包括仅针对hexagonal阵列平台；对其他连续坐标空间技术适配仍需研究；在转录组导向的临床任务中仍未达到批量RNA测序的上限。

---

## 335. Logics for Context-free Hyperproperties

**arXiv ID:** 2605.04657 | [PDF](https://arxiv.org/pdf/2605.04657v1)

**作者:** Sarah Winter `[一作]`, Martin Zimmermann `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种新的上下文无关层属性逻辑 HyperVPA，结合可见栈自动机与轨迹量化，用于指定和验证递归安全系统中的信息流属性。

**💡 创新点**

核心创新在于：①定义可见栈层属性逻辑并阐述其表达能力；②在一量化交替（∀∗∃∗或∃∗∀∗）且只由第一量化块控制栈行为的情形下，利用包含预言（prophecies）的双人游戏实现模型检测可判定；③证明除上述特殊情形外，所有更高交替或由后置量化块控制栈行为的片段均为不可判定。

**🔧 技术方法**

采用可见栈自动机（VPA）、ω-PDA 以及基于预言的游戏理论（Gale‑Stewart 游戏、不可判定性编码），并利用自动机闭包性、Büchi 接受条件以及不确定性/不完全信息游戏求解技术。

**📊 数据集**

该工作不涉及具体实验数据集，而是以理论分析与证明为主。

**📈 对比分析**

与传统 LTL/HyperLTL 以及基于普通 PDA 的层属性逻辑比较，HyperVPA 在可判定性边界上取得了更宽的可判定区间（可见栈限制下的一交替可判定），但对更复杂片段仍不可判定；在可判定片段中，决策复杂度为双指数或三指数，尚未给出实验性能评估。

**⚠️ 局限性**

限制在于：①仅对第一量化块控制栈行为的片段可判定；②对更高交替或第二量化块控制栈行为的片段不可判定；③实际实现时预言语言与游戏的指数爆炸使得可扩展性受限；④尚未证明可判定片段的最优复杂度或提供高效实现。

---

## 336. Graph-Augmented LLMs for Swiss MP Ideology Prediction

**arXiv ID:** 2605.04643 | [PDF](https://arxiv.org/pdf/2605.04643v1)

**作者:** Yifei Yuan `[一作]` (ETH Zürich), Laurence Brandenberger `[通讯]` (University of Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种检索增强生成（RAG）框架 PG‑RAG，将政治知识图查询与大型语言模型结合，用于预测瑞士议员的意识形态得分。

**💡 创新点**

首次将结构化的政治知识图（包含演讲、机构关系和立法活动）嵌入LLM上下文，并提出三种子图场景与两种编码方式，证明图结构信息可显著提升意识形态预测。

**🔧 技术方法**

采用检索增强生成（RAG）架构、Neo4j 知识图查询、LLM（GPT‑5、Qwen‑3、Apertus‑8B）零样本与少样本推理，以及图摘要与原始图编码技术。

**📊 数据集**

使用瑞士国家议会 2015‑2019 年期的 225 名议员数据与 1000 票投票记录生成的意识形态基准，构成实验数据集。

**📈 对比分析**

与传统基准（GM、PM、PBM）以及多款LLM在零样本与少样本设置下对比，PG‑RAG（MP‑S/R）在 MAE、MSE、RMSE 以及 Spearman 相关上均优于基线，零样本下 MSE 降低约 16% 并保持 0.94 的排名相关。

**⚠️ 局限性**

模型对左倾党派的预测偏右，难以充分捕捉其细微意识形态差异；高维复杂子图在 LLM 中的利用仍有限；数据集受限于瑞士议会记录，难以推广至更大多党制环境。

---

## 337. Not All Faults Are Equal: Transient-Fault Sensitivity Characterization of an Open-Source RISC-V Vector Cluster

**arXiv ID:** 2605.04803 | [PDF](https://arxiv.org/pdf/2605.04803v1)

**作者:** Maoyuan Cai `[一作]` (University of Bologna), Angelo Garofalo `[通讯]` (University of Bologna)

**通讯引用:** 868 | [OpenAlex ID](https://openalex.org/A5052915995)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

在开源 RISC‑V 向量集群 Spatz 上，对单事件瞬态（SET）和单事件上升（SEU）两种辐射误差模型进行 10 万次故障注入实验，评估不同浮点精度（FP32/FP16/BP16/FP8）和不同工作负载（MatMul、宽化 MatMul）下的系统崩溃、数据损坏和安全错误（SDC）的敏感性，进一步量化 SDC 的平均受损输出数量和 RMSE，并分析指数位、符号位和尾数位对错误传播的影响。

**💡 创新点**

首次系统性地将 RTL 级故障注入与浮点精度、位域和向量执行路径相结合，揭示指数位错误对 SDC 的显著影响，并提出仅针对高影响位域和关键模块（如 VRF/VLSU 接口）进行选择性硬化的设计建议。

**🔧 技术方法**

使用 Synopsys VC Z01X 进行 RTL 级并行故障注入，结合自定义的 FS/FD 观测点，采用精度解码（IEEE‑754、E8M7、E5M2）计算真实值，并通过 RMSE 量化错误严重度。

**📊 数据集**

实验仅使用内置的矩阵乘法和宽化矩阵乘法核，不涉及外部数据集，而是通过在不同浮点精度下随机注入位错来评估。

**📈 对比分析**

将错误分类为系统崩溃（FS）和数据损坏（FD），统计各模块的错误率，并通过比较不同精度下的平均受损输出数和 RMSE，发现 FP8 产生的 SDC 事件至少低十倍，指数位错误导致的偏差最大。

**⚠️ 局限性**

实验局限在于仅覆盖矩阵乘法核、仅使用单一向量集群架构，未考虑其他 AI 内核或多核协同；缺乏实机辐射实验验证；并且只关注单位错误，未研究多重错误交互影响。

---

## 338. A meta-analysis of the effect of generative AI on productivity and learning in programming

**arXiv ID:** 2605.04779 | [PDF](https://arxiv.org/pdf/2605.04779v1)

**作者:** Sebastian Maier `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对23项研究进行系统综述和元分析，量化了生成式人工智能（GenAI）编码助手对开发者生产力和学习成绩的影响。

**💡 创新点**

创新点在于整合多种实验与实地研究，并使用调节因子分析揭示了场景、任务和评估环境对效果的显著影响。

**🔧 技术方法**

采用随机效应模型、Hedges’ g效应量、Egger检验、Trim‑and‑Fill以及RoB工具评估偏倚。

**📊 数据集**

数据来源为ACM、arXiv、Scopus、Web of Science共10,115条记录，最终纳入23篇论文，覆盖27个效应量。

**📈 对比分析**

比较结果显示，GenAI助推开发者生产力有中等正效应（g≈0.33，95% CI [0.09,0.58]），但学习成绩无显著提升（g≈0.14，95% CI [-0.18,0.47]），并存在高度异质性。

**⚠️ 局限性**

局限包括效应量极度异质、样本量有限、研究设计多样且多数高偏倚风险，以及缺乏对真实开发环境的深入探讨。

---

## 339. Gaze4HRI: Zero-shot Benchmarking Gaze Estimation Neural-Networks for Human-Robot Interaction

**arXiv ID:** 2605.04770 | [PDF](https://arxiv.org/pdf/2605.04770v1)

**作者:** Berk Sezer `[一作]` (Middle East Technical University), Sinan Kalkan `[通讯]` (Middle East Technical University)

**通讯引用:** 2324 | [OpenAlex ID](https://openalex.org/A5032424779)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

创建了Gaze4HRI大规模HRI基准数据集（52人，3,000+视频，620,000+帧），并在此数据集上对五种零射3D视线估计模型进行系统评估。

**💡 创新点**

首次从HRI角度设计大规模多变量（光照、摄像头视角、头眼冲突、目标移动、注视方向）实验，发现数据多样性比复杂Transformer模型更关键，并提出了对极端向下注视的普遍失效点。

**🔧 技术方法**

使用基于RGB的零射视线估计技术，包含CNN（PureGaze、L2CS-Net）、Transformer（GazeTR、MCGaze、GaT）和时空网络，并结合自对抗损失、正则化等技术；采集环节采用UR5机械臂、Intel RealSense相机、OptiTrack运动捕捉系统。

**📊 数据集**

使用自研Gaze4HRI数据集，与公开数据集ETH‑X‑Gaze（110人，1M+图像）和Gaze360（238人，172K图像）进行对比。

**📈 对比分析**

在光照、视角、头眼冲突、注视方向等5个关键变量下，对五个模型进行零射性能比较，PureGaze(ETH‑X‑Gaze)在大多数条件下表现最佳；GazeTR(ETH‑X‑Gaze)与之相近；Gaze360训练的模型在光照极端或视角变化时表现显著下降；Transformer/时空模型并未在所有变量上超越CNN。

**⚠️ 局限性**

主要局限在极端向下注视导致误差显著上升；blink检测仅基于帧级遮蔽，缺乏实时鲁棒性；单摄像头下对低视角注视的估计不佳，需考虑多摄像头配置。

---

## 340. Cognitive Twins: Investigating Personalized Thinking Model Building and Its Performance Enhancement with Human-in-the-Loop

**arXiv ID:** 2605.04761 | [PDF](https://arxiv.org/pdf/2605.04761v1)

**作者:** Wu-Yuin Hwang `[一作]` (National Central University), Yuniar Indrihapsari `[通讯]` (Universitas Negeri Yogyakarta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可解释的个性化思维模型（PTM），通过层级结构将学习者日记中的行为、认知、元认知和自我价值信息自动抽象并生成数字认知双生。

**💡 创新点**

创新点在于将Marzano新分类体系与数字认知双生结合，构建多层级、可解释的学习者模型，并引入人机交互(HITL)持续修订与自我学习循环。

**🔧 技术方法**

采用Gemini 2.5 Pro LLM推理、句子嵌入（Sentence‑BERT）、UMAP降维、HDBSCAN+共识聚类以及Prompt‑based 维度聚类和生成，配合人机交互细化。

**📊 数据集**

使用了40名印尼信息技术本科生在七周内提交的学习日记文本作为训练与评估数据集。

**📈 对比分析**

通过自动原子信息匹配（F1≈75.5%）、用户5分Likert评估（≈4.3/5）以及语义层级一致性指标对比，性能超过既定阈值并优于常见基准。

**⚠️ 局限性**

局限性包括样本单一、受限于7周短期观察、对高层抽象修订难度大、自动评估受LLM误解影响，以及人机交互需要较高文本复杂度。

---

## 341. Hybrid Congestion Classification Framework Using Flow-Guided Attention and Empirical Mode Decomposition

**arXiv ID:** 2605.04752 | [PDF](https://arxiv.org/pdf/2605.04752v1)

**作者:** Eugene Kofi Okrah Denteh `[一作]` (North Dakota State University), Armstrong Aboah `[通讯]` (North Dakota State University)

**通讯引用:** 539 | [OpenAlex ID](https://openalex.org/A5005333881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 FLO-EMD 框架，利用光流引导的注意力和经验模态分解实现交通拥堵的三类分类

**💡 创新点**

创新点：将光流驱动的空间通道注意力与光流统计的 EMD 分解相结合，既保留空间定位又捕捉非平稳时序特征

**🔧 技术方法**

技术：双流 CNN + CBAM + Bi‑LSTM + Empirical Mode Decomposition (EMD) + 光流统计

**📊 数据集**

数据集：整合来自四个来源的 1050 条 5 秒监控视频，涵盖高速、城市、雨雪等多环境

**📈 对比分析**

与 CNN、Transformer 及检测+循环网络基线比较，FLO-EMD 在测试集上取得 97.5% 准确率，W‑F1 0.974，明显优于其他模型

**⚠️ 局限性**

局限：主要针对高速静态摄像头场景，交叉口与复杂道路泛化有限；光流在雨雪、低照度下仍受影响，注意力映射偶尔聚焦于边缘结构

---

## 342. AICoFe: Implementation and Deployment of an AI-Based Collaborative Feedback System for Higher Education

**arXiv ID:** 2605.04740 | [PDF](https://arxiv.org/pdf/2605.04740v1)

**作者:** Alvaro Becerra `[一作]` (Universidad Autónoma de Madrid), Ruth Cobos `[通讯]` (Universidad Autónoma de Madrid)

**通讯引用:** 1067 | [OpenAlex ID](https://openalex.org/A5014961798)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究实现并部署了AICoFe系统，利用多大语言模型为高等教育中的同行评审、学生自评和教师评估生成个性化反馈，并通过教师介入的工作流提升反馈质量。

**💡 创新点**

创新点在于将三种独立微调的大语言模型并行生成反馈，并结合教师“教师在环”介入机制，提供可追溯的LLM贡献记录，实现人机协作的透明与高质量反馈。

**🔧 技术方法**

采用的技术包括OpenAI GPT‑4.1‑mini、Google Gemini 2.5 Flash、Meta Llama 3.1的多模型推理；后端使用SQL与MongoDB混合存储；前端Dash框架构建角色化仪表盘；录音使用Teams；模型微调利用LoRA、Vertex AI等。

**📊 数据集**

使用了SOPHIAS数据集（50个评估实例）进行模型微调和验证，并在UAM多门课程中收集学生与教师评分与反馈作为训练与评估数据。

**📈 对比分析**

通过教师评审与学生问卷对比，发现AI辅助反馈在连贯性、可用性和行动性上均被认为优于传统手工反馈，系统可用性得分高；但缺乏客观的自动化评价指标。

**⚠️ 局限性**

局限性包括对教师介入的高度依赖、数据集规模有限、缺少多模态（非语言）信息的整合，以及模型可能带来的偏见与一致性风险。

---

## 343. OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization

**arXiv ID:** 2605.04738 | [PDF](https://arxiv.org/pdf/2605.04738v1)

**作者:** Zhikai Li `[一作]` (Institute of Automation Chinese Academy of Sciences), Qingyi Gu `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Hessian 低秩一致性的加性自吸收量化方法（OSAQ），用于低比特权重量化时抑制权重离群值，并与现有量化方法无缝结合。

**💡 创新点**

创新点：①利用 Hessian 的稳定零空间构造加性变换，能在不影响任务损失的前提下显著压缩权重范围；②采用 Softmax-∞ 近似实现对 L∞ 范数的可微优化；③闭式求解系数，无需训练或迭代；④该加性变换可直接嵌入权重，既不需要层间变换也不增加推理开销，形成对传统乘法变换的互补方案。

**🔧 技术方法**

核心技术：Hessian 近似与低秩分解、零空间提取、加性权重变换、Softmax-∞ 对 L∞ 范数的平滑逼近、闭式最优化求解、与 GPTQ 等 PTQ 框架的融合。

**📊 数据集**

使用的数据集与模型：LLaMA2、LLaMA3、Mistral-Large-123B-Instruct、Llama-3.1-405B-Instruct 等大规模 LLM；任务数据包括 WikiText2、C4、PIQA、ARC、WinoGrande、MMLU、MT-Bench 等，覆盖语言生成、零样本问答与通用评测。

**📈 对比分析**

对比方法：GPTQ、AWQ、QuIP、MagR、OmniQuant、WKVQuant 等主流 PTQ 方法；实验显示：在 2‑bit 量化中 OSAQ+GPTQ 的 perplexity 相较基线降低 40%+；在零样本 QA、MMLU 等任务上，OSAQ+GPTQ 均提升数个百分点至近 FP16 的水平；在 KV‑Cache 量化与权重激活量化的组合实验中亦获得性能提升。

**⚠️ 局限性**

局限性：①需要估计 Hessian 并提取零空间，对极大模型的计算开销相对较高；②目前仅在权重量化场景验证，对激活量化或混合量化的效果尚未充分探测；③加性变换对不同任务、不同数据分布的鲁棒性需进一步评估；④依赖超参数（γ、τ、μ1、μ2），虽然实验表明结果相对稳健，但在更广泛的配置下仍需探索。

---

## 344. Sequential topology optimization: SIMP initialization for level-set boundary refinement

**arXiv ID:** 2605.04735 | [PDF](https://arxiv.org/pdf/2605.04735v1)

**作者:** Ondřej Ježek `[一作]` (Institute of Thermomechanics, Czech Academy of Sciences), Dušan Gabriel `[通讯]` (Institute of Thermomechanics, Czech Academy of Sciences)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出一种顺序拓扑优化框架，将SIMP密度分布转换为签名距离函数（SDF）并作为级联级别集（level‑set）初始化，以实现边界的精细化。

**💡 创新点**

创新点在于无网格（mesh‑agnostic）的SDF几何传递方法、将SIMP结果直接映射为level‑set起始值，以及通过自适应Hilbertian投影加速收敛，显著降低级联级别集对初始化的敏感性。

**🔧 技术方法**

采用了SIMP密度优化、节点映射、等值面提取、SDF构造、Hilbertian扩展-正则化、Hamilton–Jacobi演化、增广拉格朗日/ Hilbertian投影等技术。

**📊 数据集**

通过三维悬臂梁和MBB梁的标准基准问题验证，使用同一结构化六面体网格和相同的材料参数。

**📈 对比分析**

与从均匀孔隙初始化的纯级联级别集方法相比，悬臂梁可获得最高4.6倍的壁钟时间加速，MBB梁提升约1.3–1.4倍；最终合规性与基准相当或略优，且几何质量满足制造要求。

**⚠️ 局限性**

主要局限在于当前实现仅适用于结构化网格；对更复杂几何、无网格拟合有限元及多约束场景仍需进一步研究和参数调优。

---

## 345. Reward-Decomposed Reinforcement Learning for Immersive Video Role-Playing

**arXiv ID:** 2605.04733 | [PDF](https://arxiv.org/pdf/2605.04733v1)

**作者:** Miao Wang `[一作]` (Nanjing University), Yaduan Ruan `[通讯]` (Nanjing University)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5037202474)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视频的角色扮演对话系统，使用眼–脑–口（See-Think-Speak）三阶段框架并在每阶段通过强化学习进行训练，目标是让模型先感知画面，再做内在推理，最后生成符合场景与角色的对话。

**💡 创新点**

创新点：①首次公开了以电影剧本为基础的、视频‑角色扮演数据集；②将观察、推理、发声完全解耦，形成“眼–脑–口”RL框架；③为每个阶段设计了专属奖励（CLIP 场景‑文本对齐、Perceptual‑Cognitive Gain、答案准确性、格式完整性）并采用 GRPO 进行联合优化；④通过跨模态奖励显著提升视觉对话一致性与情境适配性。

**🔧 技术方法**

核心技术：大规模视觉‑语言模型 Qwen2.5‑VL‑7B 作为基模型；CLIP 视觉‑文本嵌入用于场景对齐奖励；BERTScore 用于语义相似度奖励；GRPO 强化学习框架；结构化生成标签 `<see>` `<think>` `<answer>`；数据增强与 LLM‑扩展技术。

**📊 数据集**

使用的数据集：基于《哈利·波特》与《指环王》电影剧本构建的 32k+ 语料，包含原始剧本对话、LLM 生成扩展对话，并对视频进行字幕裁剪与帧采样，保证无信息泄漏；此外对外部 VideoQA 基准（NExT‑QA、PororoQA、ActivityNet‑QA）做零样本测试。

**📈 对比分析**

对比方法：与 8B/14B/38B InternVL3、Qwen2.5‑VL‑32B/7B 以及文本‑RP 基线（RoleMRC、Crab、Haruhi）进行比较。评估指标为：视觉证据对齐（VEG）、情境角色一致性（SPC）与对话自然性（CN）。结果显示 EBM‑CLIP‑Max 在 VEG 与 SPC 上均优于所有对比模型（VEG 74.25 vs 74.61，SPC 70.37 vs 71.01），平均得分 73.13，显著提升整体表现；在 VideoQA 零样本测试中，NExT‑QA 平均提升 1.93，PororoQA 提升 2.60，ActivityNet‑QA 提升 2.50。

**⚠️ 局限性**

局限性：①数据量相对较小（约 32k 片段），导致在对话自然性上略逊于大规模文本‑RP 基线；②视觉对齐依赖 CLIP，可能忽略细粒度动作或表情细节；③RL 训练耗时且对 GPU 资源要求高；④在极端光照或模糊画面下的感知仍易出错；⑤模型对新电影或非西方文化场景的迁移能力尚待进一步验证。

---

## 346. ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting

**arXiv ID:** 2605.04730 | [PDF](https://arxiv.org/pdf/2605.04730v1)

**作者:** Yingdong Gu `[一作]` (Wuhan University), Jiayuan Li `[通讯]` (Wuhan University)

**通讯引用:** 5733 | [OpenAlex ID](https://openalex.org/A5108050395)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于3D高斯渲染的视觉定位框架ULF-Loc，能够无偏地构建稀疏地标特征并实现精确的6-DoF相机位姿估计。

**💡 创新点**

核心创新包括①对α‑混合优化中存在的特征偏差进行理论分析；②通过关键点共识采样与几何加权特征融合实现无偏特征构建；③引入局部几何一致性验证（LGCV）剔除渲染伪影导致的匹配错误。

**🔧 技术方法**

技术方法涵盖3D Gaussian Splatting、几何加权特征融合、关键点共识地标采样、粗细化定位流程、基于张量的LGCV匹配过滤以及RANSAC+PnP位姿求解。

**📊 数据集**

在三大公开基准上评估：7Scenes、12Scenes（室内）和Cambridge Landmarks（室外）。

**📈 对比分析**

与传统结构光、APR、SCR以及NeRF/GS定位方法相比，ULF-Loc 在平均中值平移误差上分别提升 17–36% 及回召率提升 9–10%，且训练时间缩短 10 倍、GPU内存消耗降至 1/6，定位速度保持在 4–5 FPS 以上。

**⚠️ 局限性**

主要局限在于位姿微调阶段仍受 3D 高斯渲染中浮动伪影的影响，导致极端场景下定位失败；此外对动态场景的适应性与可扩展性仍待进一步验证。

---

## 347. Anny-Fit: All-Age Human Mesh Recovery

**arXiv ID:** 2605.04728 | [PDF](https://arxiv.org/pdf/2605.04728v1)

**作者:** Laura Bravo-Sánchez `[一作]` (NAVER LABS Europe), Fabien Baradel `[通讯]` (NAVER LABS Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种多人人、全龄的相机空间优化框架，通过结合深度图、实例分割、2D关键点和VLM生成的年龄/性别语义属性，联合优化所有人形网格，实现全龄人形恢复。

**💡 创新点**

创新点包括：①在相机坐标系下联合优化多个人，确保全局深度一致；②利用VLM提供的语义属性作为形状初始化与约束，解决深度-形状模糊；③实现成人模型的零样本全龄迁移；④通过高质量优化结果生成大规模伪标注，用于训练。

**🔧 技术方法**

技术手段包括：Anny全龄人体模型、基于重投影+形状+深度损失的优化框架、VLM（如CLIP）获取年龄/性别语义属性、密集关键点/分割/深度估计器、深度顺序损失等。

**📊 数据集**

使用的数据集主要有 Relative Human、CMU Panoptic（幼儿序列）、Hi4D、MS‑COCO（用于生成伪标注）等。

**📈 对比分析**

与 BEV、Multi‑HMR、CameraHMR 等方法在 2D（mPCK、PCDR）、3D（MPJPE、PCK@15cm）指标上对比，平均提升 13–16 点 2D 重投影精度、6–7 点深度关系准确率、9–29 点 3D 误差下降，形状估计提升 25–82 点；零样本迁移下可匹敌或超越 SOTA。

**⚠️ 局限性**

局限性在于对初始参数和专家预测（关键点、分割、深度、语义属性）质量高度依赖，低置信度或错误的专家信息会导致优化收敛失败或人形穿插；在复杂多人交互场景下仍易出现失效。

---

## 348. Beyond Seeing Is Believing: On Crowdsourced Detection of Audiovisual Deepfakes

**arXiv ID:** 2605.04797 | [PDF](https://arxiv.org/pdf/2605.04797v1)

**作者:** Michael Soprano `[一作]` (University of Udine), Stefano Mizzaro `[通讯]` (University of Udine)

**通讯引用:** 3286 | [OpenAlex ID](https://openalex.org/A5005575421)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过在Prolific平台上对AV-Deepfake1M和Trusted Media Challenge两大数据集进行匹配式众包实验，收集并分析了对音视频深度伪造视频的真实性判断、操纵类型识别以及时间戳定位等人类决策数据。

**💡 创新点**

创新点在于：①首次在同一实验协议下对两大多模态深度伪造数据集进行大规模人类评测；②系统评估众包在真实性检测、操纵类型归因及时间定位三维任务中的一致性与误差分布；③比较两种聚合方法（多数投票与Dempster‑Shafer）对检测性能的影响，并提出实用的两阶段筛查工作流。

**🔧 技术方法**

技术方法包括：基于Crowd_Frame框架的任务部署；使用多数投票和Dempster‑Shafer理论进行聚合；Krippendorff α、majority与pairwise agreement评估一致性；时间戳一致性通过归一化中位数、IQR及±5%窗口一致性分数衡量；统计分析采用Mann‑Whitney、Kruskal‑Wallis、McNemar等非参数检验与Bonferroni/Holm‑Bonferroni校正。

**📊 数据集**

使用的公开数据集为：AV‑Deepfake1M（2024）和Trusted Media Challenge（TMC，2022），各自从中抽取48条样本（12真视频+36伪造视频），总计96条视频。

**📈 对比分析**

在真实性检测上，多数投票与DS聚合都能显著降低误报率，但多样化的伪造类型仍导致高漏报；在TMC上召回率约为0.56（多数投票），而AV‑Deepfake1M仅为0.28；聚合方法提升召回率但也略增误报；操纵类型识别准确度低（TMC 0.14-0.35，AV‑Deepfake1M 0.05-0.50），尤其对音视频联合伪造表现最差。

**⚠️ 局限性**

局限性包括：样本量有限（仅96条视频），难以覆盖所有现实场景；受众主要为美国Prolific劳动力，可能缺乏多语言和多文化视角；任务设计未强制开启音频导致对音频伪造的敏感性降低；未与现有自动检测器基准比较；时间戳一致性评估仅对被标记为伪造的视频，且仅提供粗略的集中度指标。

---

## 349. A Biased Nonnegative Block Term Tensor Decomposition Model for Dynamic QoS Prediction

**arXiv ID:** 2605.04813 | [PDF](https://arxiv.org/pdf/2605.04813v1)

**作者:** Wenjing Liu `[一作]` (Southwest University), Qu Wang `[通讯]` (Southwest University)

**通讯引用:** 58968 | [OpenAlex ID](https://openalex.org/A5108047874)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于块项分解与线性偏置的动态QoS预测框架BNBT，并设计了单因子依赖的非负乘法更新算法进行参数估计。

**💡 创新点**

创新点在于：①引入块项张量分解（rank-(L×M×N)）提升对复杂用户-服务-时间交互的表示能力；②加入线性偏置项更好捕捉维度特定的趋势；③设计SLF‑NMUT算法保证非负约束且计算效率高。

**🔧 技术方法**

使用了块项张量分解、线性偏置、非负乘法更新、L2正则化、欧氏距离损失以及基于RMSE/MAE的评估指标。

**📊 数据集**

使用了公开的两个动态QoS数据集：D1（响应时间）和D2（吞吐量），均包含142个用户、4500个服务和64个时间片。

**📈 对比分析**

通过与CP基模型M1和Tucker基模型M2进行对比，采用相同的实验设置（R=3、迭代上限1000、10次随机初始化），在10%~60%稀疏度下评估RMSE和MAE。BNBT在所有实验条件下均优于两基线模型，尤其在10%稀疏情况下RMSE提升约9%，MAE提升约11%。

**⚠️ 局限性**

主要限制：训练效率仍可进一步提升，未来可考虑随机梯度等优化；模型对参数（如L、M、N、R）敏感，需手工调优；未探讨跨域或多租户场景下的适用性。

---

## 350. Optimal Uncertainty-Aware Calibration for the AX=YB Problem

**arXiv ID:** 2605.04809 | [PDF](https://arxiv.org/pdf/2605.04809v1)

**作者:** Yanjia Chen `[一作]` (Huazhong University of Science and Technology), Han Ding `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 13680 | [OpenAlex ID](https://openalex.org/A5057513904)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了一种针对 AX=YB 手眼标定问题的无不确定性意识的全局优化框架，提出了 L-HED、UAL-HED 与 SI-AH 方法，并在合成数据与真实实验中验证其性能。

**💡 创新点**

创新点在于引入基于 SE(3) 的相对不确定性指标 SRM@SE(3)，避免显式建模不确定性，结合 Lie 代数启发式逃逸下降实现全局同步迭代，同时提供稳健的初始解生成方案。

**🔧 技术方法**

采用 Lie 群/李代数优化、右/左雅可比、凸优化、Heuristic Escape Descent、随机扰动、马氏距离、SVD、Levenberg-Marquardt 等技术。

**📊 数据集**

使用 100 对合成 {A_i},{B_i} 数据集进行仿真，并在 ABB IRB 6700 机器人与 Lecia AT 960 激光跟踪器的真实实验中收集 200 对数据进行验证。

**📈 对比分析**

与传统解析解、Dual Quaternion、Kronecker Product、LMI‑SDP、Point Cloud Matching、SI‑AH 等七种方法对比，UAL‑HED 在高不确定性场景下误差降低约 67–90%，在真实实验中平均位置误差 0.293 mm、角度误差 0.039°，明显优于其他方法。

**⚠️ 局限性**

仍受限于数据集规模与质量，缺乏对所有 SE(3) 双不变距离的构造，计算量相对较大，对极端离群点的鲁棒性仍需进一步提升。

---

## 351. Bridging Perception and Action: A Lightweight Multimodal Meta-Planner Framework for Robust Earth Observation Agents

**arXiv ID:** 2605.04777 | [PDF](https://arxiv.org/pdf/2605.04777v1)

**作者:** Jinghui Xu `[一作]` (State Key Laboratory of Space Information System and Integrated Application), Xueqian Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5544 | [OpenAlex ID](https://openalex.org/A5100737125)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出轻量级多模态元规划器（LMMP），通过分离高层策略与低层执行，解决遥感代理在单模型中出现的认知过载和推理失误。

**💡 创新点**

创新点在于双感知机制将视觉与任务语义结合，元任务库将遥感专家知识注入规划过程，并通过两阶段训练（专家逻辑SFT + 直接偏好优化DPO）实现策略层次化与反馈自适应。

**🔧 技术方法**

技术手段包括 RS-MLLM 感知、LoRA 微调、SFT 与 DPO 训练、LLM-as-a-judge 自动评估以及工具调用过程的步骤级与端到端指标。

**📊 数据集**

使用 EarthBench、ThinkGeo 以及自制 GeoScenario-116 三个遥感数据集进行训练与评估。

**📈 对比分析**

与多种基线模型（如 Qwen 系列、GPT‑5）对比，LMMP 在工具调用准确率、步骤级指标和任务成功率方面提升约10%–25%，并能让小参数模型在任务上逼近甚至超越更大模型。

**⚠️ 局限性**

局限包括数据集样本量不足导致评估方差较大、端到端成功率仍未达到工业部署需求，以及对极小模型效率的进一步探索不足。

---

## 352. Online Orthogonal Vectors Revisited

**arXiv ID:** 2605.04798 | [PDF](https://arxiv.org/pdf/2605.04798v1)

**作者:** Karthik Gajulapalli `[一作]` (Georgetown University), Sidhant Saraogi `[通讯]` (Georgetown University)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5059723204)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一套新的确定性数据结构，用于在线正交向量问题（Online Orthogonal Vectors, OV），并给出了针对低维（d = c log n）和中等维（d = n^ε）两种情形的上界与下界。

**💡 创新点**

创新点在于：① 通过结构与随机性分解，实现完全确定性的数据结构，低维时匹配了最优随机算法的性能；② 在中等维上首次得到比以往随机算法更优的时间-空间折衷；③ 通过非均匀 SETH 与 Hamiltonian Path 猜想，给出大多项式空间下线性查询时间的强下界，填补了先前仅针对有效率预处理的空缺。

**🔧 技术方法**

主要技术包括：结构-随机性分解（structure‑versus‑randomness），伪随机性分析，递归降维与拆分；利用随机输入的稀疏性得到平均情况算法；对最坏情况采用分层递归 + 预处理与候选列表；以及在下界证明中构造硬实例并与非均匀 SETH/HamPath 猜想关联。

**📊 数据集**

没有使用公开实验数据集。研究中的数据完全是理论构造的向量集合或随机生成的 p‑biased 0/1 向量，用以证明平均/最坏情况性能。

**📈 对比分析**

与之前的随机算法（如 Chan 2017 SoCG、Charikar‑Indyk‑Panigrahy 2002 等）相比，低维下查询时间 T = n^{1−O(1/(c log c))}，空间 S = n^{1+δ}（δ 可取任意正数），与最优随机算法相当；在中等维时，查询时间 T ≤ n^{1−ε}，空间 S = n^{1−ε}·2^{O(d log(1/ε))}，显著优于既往 O(n^{1−ε/2}) 级别。下界方面，若非均匀 SETH 成立，则任意子线性查询时间的数据结构必须使用超多项式空间；类似地，非均匀 HamPath 猜想给出 N^{1.087} 级别空间/时间的不可能性。

**⚠️ 局限性**

局限性包括：① 结果依赖于非均匀 SETH/HamPath 猜想，尚未得到无条件下界；② 仅给出理论上最优或接近最优的时间空间估计，未进行实验验证；③ 在中等维度的参数选择仍相对保守，进一步压缩空间/时间的余地尚未探索；④ 对于极高维度（如 d = n^0.99）下的具体实现细节仍待完善。

---

## 353. AGIPC: Adaptive In-Solve Algebraic Coarsening for GPU IPC

**arXiv ID:** 2605.04773 | [PDF](https://arxiv.org/pdf/2605.04773v1)

**作者:** Xuan Wang `[一作]` (University of Hong Kong), Kemeng Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5016938949)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于GPU的自适应代数粗化方法，在每一次Newton迭代中动态减少线性系统的自由度，以加速隐式时间积分的IPC模拟。

**💡 创新点**

创新点在于将拓扑不变的粗化视为边缘坍塌的代数映射，使用warp级哈希聚合实现高效GPU并行，且通过可选的仿射嵌入保持旋转自由度，避免显式重网格带来的内存和同步开销。

**🔧 技术方法**

采用的技术包括GPU并行的warp哈希聚合、梯度和Hessian的并行稀疏化与汇总、PCG预条件（MAS）、后粗化CG细化、以及Green应变增量判别的自适应标记。

**📊 数据集**

在多个数据集上验证：软/硬多米诺、弹性/刚性布料、龙模型、矩形布、混合物体等，包括不同材料刚度、时间步长与分辨率的综合实验。

**📈 对比分析**

与现有GPU IPC框架StiffGIPC、GMG和AmgX比较，AGIPC在大多数场景下实现了最高3倍的加速，保持视觉效果相同，并在高刚度或高分辨率下表现尤为优异。

**⚠️ 局限性**

主要局限包括：固定的应变阈值对不同变形范围不够自适应；粗化后仍需在完整网格上完成Hessian组装和碰撞检测；跨物体聚合受限，难以在多体碰撞密集场景进一步压缩自由度。

---

## 354. How Does Chunking Affect Retrieval-Augmented Code Completion? A Controlled Empirical Study

**arXiv ID:** 2605.04763 | [PDF](https://arxiv.org/pdf/2605.04763v1)

**作者:** Xinjian Wu `[一作]` (King's College London), Jie Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对检索增强式代码补全中的分块策略进行了系统的控制实验，评估了四种分块方法在不同检索器、生成器和参数配置下的效果。

**💡 创新点**

创新点在于首次将分块维度与检索器、生成器和参数全因子化交叉，揭示分块策略对准确率的显著影响并给出实证推荐。

**🔧 技术方法**

使用了四种分块策略（Function、Declaration、Sliding Window、cAST）、四种检索器（BM25、EmbeddingGemma、Qwen-0.6B、Qwen-4B）和五个9B级别生成器（DeepSeek-Coder、Qwen2.5-Coder、Seed-Coder、Qwen3.5、StarCoder2），并调节分块大小、跨文件上下文长度等参数。

**📊 数据集**

数据集包括Python代码补全基准RepoEval和跨语言基准CrossCodeEval（Python子集），并在Java子集做了交叉验证。

**📈 对比分析**

实验结果显示分块策略对Exact Match有统计显著影响，Function分块总是最低，Sliding Window和cAST在质量与成本上均占优，跨文件上下文长度是主要调优维度。

**⚠️ 局限性**

限制主要在于仅评估Python（及部分Java）语言、使用EM这一严格二元指标、未探究更大模型规模和其他语义评价指标。

---

## 355. Gyan: An Explainable Neuro-Symbolic Language Model

**arXiv ID:** 2605.04759 | [PDF](https://arxiv.org/pdf/2605.04759v1)

**作者:** Venkat Srinivasan `[一作]` (Innospark), Geetika Sharma `[通讯]` (Gyan AI Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了Gyan，一种基于非Transformer的可解释语言模型，利用知识解耦与深度语义表示实现对文本的完整成分编码，并在多项任务中实现SOTA表现。

**💡 创新点**

创新点在于：①将语言模型与知识库解耦，构建可逆、可追踪的意义表示图（GMR）；②采用修辞结构理论、语义角色与知识图谱相结合的多层语义抽象；③实现低计算成本、可解释与不产生幻觉的推理；④构建“世界模型”以模拟人类知识组织与检索。

**🔧 技术方法**

使用技术包括：知识驱动的深度语言处理管道、语义角色标注、修辞结构分析、意义表示图生成与扩展、知识库（KS）实时检索与动态知识更新，以及可视化与可追踪推理框架。

**📊 数据集**

采用数据集：MS Marco（检索排序）、PubMedQA（医学问答）、MMLU‑Medicine（医学多任务理解）、20条随机互联网查询（检索相关性评估）以及开放式作文打分数据集。

**📈 对比分析**

与基准方法比较时，Gyan在PubMedQA和MMLU‑Medicine上达到SOTA，在MS Marco中排名前3；在20条查询的相关性评估中，Gyan在nDCG、加权准确率上均优于Google和BM‑25，展示了在无训练、低知识覆盖率下的高效检索性能。

**⚠️ 局限性**

局限性包括：性能高度依赖知识库的完整性与质量；在新领域或低资源语言时需手工构建或导入知识；对非常大规模多语言文本的处理尚未验证；仍需进一步研究自适应知识补充与抽象式生成能力。

---

## 356. Knowledge-Free Correlated Agreement for Incentivizing Federated Learning

**arXiv ID:** 2605.04747 | [PDF](https://arxiv.org/pdf/2605.04747v1)

**作者:** Leon Witt `[一作]` (SIMIS Shanghai), Lucy Klinger `[通讯]` (SIMIS Shanghai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种知识无关的相关共识（KFCA）机制，用于在联邦学习中对客户端贡献进行奖励，无需依赖地面真值、公共测试集或分布知识；

**💡 创新点**

创新点在于：①不需要估计全局报告相关矩阵，完全知识无关；②在“类别世界”假设下实现严格真诚激励，消除传统相关共识（CA）的标签翻转漏洞；③支持实时奖励计算，适合去中心化和区块链激励部署；

**🔧 技术方法**

技术方法包括：多任务同行预测（MTPP）框架、简单的相等计分规则 𝒮(r₁,r₂)=1{r₁=r₂}、条件独立的信号模型、LoRA/DoRA 轻量级适配器微调、参数更新的符号量化、与 Shapley 值估计器的对比实验；

**📊 数据集**

实验使用的数据集：MNIST（CNN）用于 Shapley 对比；PCB 质量检测的实测图像集；FlowerTune LLM Leaderboard 上的四个领域（通用 NLP、金融、医疗、代码）进行 LoRA/DoRA 微调；

**📈 对比分析**

与 CA 机制和多种 Shapley 值估计方法比较：KFCA 在计算成本上比 CA 降低数十倍，奖励分布与精确 Shapley 更相近；在标签翻转攻击场景下，KFCA 能正确惩罚恶意客户端，而 CA 则无法；

**⚠️ 局限性**

局限性：需要满足“类别世界”条件（同类别正相关、不同类别负相关），若信号连续或高维且无法映射为满足此条件的类别，需额外转换管道；同时依赖多数诚实客户端假设，攻击比例超过 50% 时可能失效。

---

## 357. AISSA: Implementation and Deployment of an AI-based Student Slides Analysis tool for Academic Presentations

**arXiv ID:** 2605.04729 | [PDF](https://arxiv.org/pdf/2605.04729v1)

**作者:** Alvaro Becerra `[一作]` (Universidad Autónoma de Madrid), Ruth Cobos `[通讯]` (Universidad Autónoma de Madrid)

**通讯引用:** 1067 | [OpenAlex ID](https://openalex.org/A5014961798)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并部署了AISSA系统，利用大语言模型和学习分析仪表盘为学生的演示幻灯片提供基于评分表的定量与定性反馈。

**💡 创新点**

创新点在于将LLM生成的结构化反馈与交互式仪表盘相结合，实现可扩展的口头演示幻灯片形成性评价，并通过限定JSON输出确保反馈一致性。

**🔧 技术方法**

采用Python、Plotly Dash、OpenAI ChatGPT 5.2、PostgreSQL、MongoDB、OpenCV、python-pptx等技术构建了模块化系统。

**📊 数据集**

使用46名西班牙UAM大学本科生提交的真实课堂幻灯片作为评估数据集，并在Pilot部署中处理了90份演示文件。

**📈 对比分析**

通过异步任务队列实现稳定处理，平均每份演示耗时1–3分钟、成本约$0.06/评估，SUS得分83.38，访谈显示学生认为反馈公平且有用；性能在实际课堂情境下表现良好。

**⚠️ 局限性**

局限性包括样本规模有限、仅在单一机构验证、依赖LLM输出的可解释性与公正性、缺乏跨学科或多语言验证，以及对极端输入或格式错误的鲁棒性不足。

---

## 358. Distance Distributions Between Nodes in Concentric Disk-Annulus or Sphere-Shell Regions

**arXiv ID:** 2605.04794 | [PDF](https://arxiv.org/pdf/2605.04794v1)

**作者:** Nicholas Vaiopoulos `[一作]` (University of Thessaly), Konstantinos K. Delibasis `[通讯]` (University of Thessaly)

**通讯引用:** 1562 | [OpenAlex ID](https://openalex.org/A5040508604)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文推导了在同心圆盘–环形与球面–壳层无线网络中，两节点距离的闭式分布（PDF）表达式，并给出了两种场景（均匀分布与随机途径模型稳态分布）的完整解法。

**💡 创新点**

创新点在于首次为不同半径的同心几何体提供分段闭式解析公式，兼顾非均匀节点分布，并提出可直接使用的Beta近似方法以简化实际性能评估。

**🔧 技术方法**

主要技术包括几何积分、条件概率解析、随机途径模型的稳态分布、三角/多项式求解以及基于矩匹配的Beta分布逼近。

**📊 数据集**

论文不依赖外部数据集，所有结果通过解析推导并以10⁵个独立蒙特卡洛仿真验证。

**📈 对比分析**

通过将解析PDF与Beta近似进行KL散度比较，结果显示在R₂/R₁>≈5–7（二维）和>≈3.5–5.5（三维）时误差<10⁻²，仿真曲线与解析曲线高度吻合，证明近似的实用性。

**⚠️ 局限性**

局限性包括Beta逼近在半径比小或三维情形下精度下降，且假设节点遵循均匀或RWP稳态分布，未考虑轨迹动态与时间相关性，也仅适用于同心几何结构。

---

## 359. Bilinear Mamba-Koopman Neural MPC for Varying Dynamics

**arXiv ID:** 2605.04793 | [PDF](https://arxiv.org/pdf/2605.04793v1)

**作者:** Matan Pagi `[一作]` (Psistar AI), Zohar Sorek `[通讯]` (Psistar AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在Mamba‑Koopman神经MPC框架中加入低秩双线性耦合的模型，允许系统算子随当前控制输入变化，同时保持单次QP凸性。

**💡 创新点**

创新点在于：①引入可学习的低秩双线性矩阵G_i，使得隐状态动态不再满足控制-状态条件独立性；②提供基于Lie‑Trotter分裂的离散化和精确雅可比，支持Sequential Convex Programming (SCP)的单调下降与KKT收敛证明；③在时间不变与时间变两类工况下分别验证预测和闭环性能提升。

**🔧 技术方法**

使用技术包括：神经网络编码器（Mamba‑based）、低秩双线性耦合参数化、Lie‑Trotter ZOH离散、自动微分求解精确状态/输入雅可比、SCP迭代QP求解器（OSQP）、谱正则化保证离散化稳定性、训练时的历史序列编码与多步预测损失。

**📊 数据集**

数据集：CartPole（4 状态、1 控制）和RSCP（9 状态、3 控制），每个系统均设定时间不变 (TI) 与时间变 (TV) 两种版本，共计四个实验细胞。

**📈 对比分析**

与条件独立的线性 MamKO 基线对比：开放式 30 步预测 MSE 均不劣；闭环 MPC 计数累计成本，在 RSCP TV 上 SCP‑5 达到 30% 成本降低；在 CartPole TV 上差距小；stale‑plan 试验显示双线性模型在计划滞后时更稳健。训练稳定性方面，双线性模型在 RSCP TV 上显著抑制验证损失波动。

**⚠️ 局限性**

局限性：仅解决控制-状态耦合与时间变参数这两条结构性缺口；未扩展至控制矩阵 B 的双线性；对高维隐空间的秩约束需进一步研究；SCP 迭代增加计算成本；实验仍局限于小型仿真案例，尚未验证在大型工业系统中的可扩展性。

---

## 360. AgentTrust: Runtime Safety Evaluation and Interception for AI Agent Tool Use

**arXiv ID:** 2605.04785 | [PDF](https://arxiv.org/pdf/2605.04785v1)

**作者:** Chenglin Yang `[一作]` `[通讯]` (Independent Researcher), Chenglin Yang (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 AgentTrust，一套实时拦截 AI 代理工具调用的安全决策框架；

**💡 创新点**

在现有守卫基础上首次集成四大创新子系统：纯文本 Shell 反混淆器、SafeFix 提案引擎、多步攻击链（RiskChain）检测，以及增量式 LLM‑Judge 缓存，形成完整的语义层面安全判断与建议链路；

**🔧 技术方法**

采用正则和文本重写进行 Shell 反混淆；基于 YAML 配置的规则引擎与会话追踪实现多步链检测；LLM‑Judge 通过模型上下文协议（MCP）与块哈希增量缓存实现高效的 LLM 决策；

**📊 数据集**

使用 300 条手工构造的内部基准（覆盖六类风险）以及 630 条独立真实世界场景（5 批次、包含 95 条混淆攻击）进行评估；

**📈 对比分析**

在内部基准上规则集（无 LLM）达到 95.0% verdict/73.7% risk 级别准确率，延迟低于 2 ms；在独立基准上 96.7% verdict；与四个基线（regex、Llama‑Guard、NeMo、DeepSeek）对比，AgentTrust 在准确率、误报率、以及毫秒级低延迟方面显著优于基线；

**⚠️ 局限性**

局限性包括：仅进行静态文本分析，无法捕捉运行时语义；Shell 反混淆仅支持九种纯文本策略，无法处理深度嵌套或 AST 级别攻击；缺乏对非英语命令的支持；需在同一进程内部署，无法对恶意运行时逃逸；多步链检测尚未在真实多步攻击基准上充分验证。

---

## 361. MIRAGE: Retrieval and Generation of Multimodal Images and Texts for Medical Education

**arXiv ID:** 2605.04772 | [PDF](https://arxiv.org/pdf/2605.04772v1)

**作者:** Miguel Diaz Benito `[一作]`, Juan C. SanMiguel `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了 MIRAGE，一个集成医学图像检索、生成与文本丰富描述的多模态教育工具。

**💡 创新点**

创新点在于统一检索、概念比较、图像生成与 LLM 文本增强，并全程使用公开预训练模型。

**🔧 技术方法**

采用 CLIP 细调版 MedICaT-ROCO、Prompt2MedImage 扩散模型、Dolly‑v2‑3b LLM 以及 Latent 语义算子。

**📊 数据集**

以 ROCO 数据集（约81k医学图像+标题）为基础。

**📈 对比分析**

通过余弦相似度阈值分辨同义/不同义文本、图像-标题和合成图像-标题，分类准确率≥97%，检索与生成结果在医学教育场景中保持语义一致。

**⚠️ 局限性**

主要限制包括合成图像与真实图像的对齐仍偏低、仅使用 ROCO 5% 的数据、对罕见病种覆盖不足。

---

## 362. Lightweight Cross-Spectral Face Recognition via Contrastive Alignment and Distillation

**arXiv ID:** 2605.04769 | [PDF](https://arxiv.org/pdf/2605.04769v1)

**作者:** Anjith George `[一作]` (Idiap Research Institute), Sebastien Marcel `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级跨光谱人脸识别框架xEdgeFace，利用预训练RGB网络的LayerNorm和浅层卷积可调参数进行跨模态适配，配合对比对齐和自蒸馏实现端到端训练。

**💡 创新点**

创新点在于：①仅调优LayerNorm和早期层实现参数高效跨模态迁移，保持原RGB性能；②在同一网络中同时支持同质与异质识别；③使用对比对齐+自蒸馏两种损失兼顾跨模态对齐与灾难性遗忘抑制。

**🔧 技术方法**

技术包括：轻量化EdgeFace骨干网络、LayerNorm微调、早期卷积层微调、对比损失（Cosine-NTXent）与自蒸馏损失、Adam优化、FLOPs/参数控制。

**📊 数据集**

使用了六个跨模态基准：Tufts（VIS-TH），MCXFace（VIS-TH），Polathermal（LWIR-VIS），SCFace（高/低分辨率VIS-IR），CUFSF（Sketch-Photo），CASIA NIR‑VIS 2.0（VIS-NIR）以及标准FR基准LFW、CA‑LFW、CP‑LFW、CFP‑FP、AgeDB‑30。

**📈 对比分析**

与当前最先进方法（PDT、CAIM、SSMB、DVG、DVG‑Face等）对比，xEdgeFace在所有跨光谱任务上达到或逼近最高Rank‑1/VR，且计算量仅为对手的1/20，显著提升模型轻量化与部署友好性。

**⚠️ 局限性**

局限性主要体现在：对结构差异大（如Sketch‑Photo、手绘）跨模态的适配效果不足；仅针对光谱/光照差异的统计对齐方法，无法充分补偿几何或风格变化大的模态。

---

## 363. Elicitation Matters: How Prompts and Query Protocols Shape LLM Surrogates under Sparse Observations

**arXiv ID:** 2605.04764 | [PDF](https://arxiv.org/pdf/2605.04764v1)

**作者:** Ge Lei `[一作]` (Imperial College London), Samuel J. Cooper `[通讯]` (Imperial College London)

**通讯引用:** 51020 | [OpenAlex ID](https://openalex.org/A5089223699)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文探究了在稀疏观测下，通过不同提示文本和查询协议，语言模型（LLM）如何被用作低数据优化中的代理回归器，并系统评估了其对预测、置信度、更新行为及后续贝叶斯优化（BO）决策的影响。

**💡 创新点**

创新点包括提出“不确定性对齐”标准来衡量模型不确定性与残留函数空间歧义的匹配；证明结构化提示文本能充当有效先验；区分了 POINTWISE 与 JOINT 查询协议对代理信念的不同塑造；揭示了在顺序证据下，LLM 信念会出现非单调、顺序敏感的自我修正，并证明这些差异会显著影响 BO 的采样策略和最终 regret。

**🔧 技术方法**

技术手段包括：基于提示文本的代理信念定义；Token‑级与采样级不确定性估计；使用 Spearman 相关评估不确定性对齐；Optimal Transport 与多维缩放测量不同提示/协议下信念的差异；构造“Fit Recoverability”和“Constraint Satisfaction”指标；设计顺序信息实验以捕捉信念更新曲线；在 BO 框架下集成 LLM 预测和不确定性作为 UCB/EI 探索信号；使用多种 LLM（如 GPT‑4o、Qwen3.5、GPT‑5.4）与 GP 基线对比。

**📊 数据集**

数据集与实验环境涵盖：多种可解释的函数族（高斯、二次、线性、正弦、逻辑斯蒂）作为合成基准；经典与修改版 Branin 函数；加利福尼亚住房数据用于 XGBoost 超参数优化；DFN 模拟的电池正极设计（能量密度与失衡）任务。全部实验均为无噪声或低噪声设置，观测点数极少。

**📈 对比分析**

与 GP 先验或无提示基线对比，POINTWISE 提示在早期阶段显著降低 regret，且在结构匹配良好时优于 JOINT；在先验不匹配时，LLM 先验可能导致 regret 上升；在实际 HPO 和电池设计任务中，带结构提示的 LLM 能更快定位高性能区域；不确定性对齐提升了模型对高歧义区域的敏感度，进一步改善了 BO 的探索效果。

**⚠️ 局限性**

局限性：实验严格控制，主要使用无噪声或低噪声合成函数；不确定性对齐的参考歧义仅适用于可枚举的函数族，难以推广至更复杂或不规则目标；对模型尺寸、提示细节的泛化性仍待进一步验证；顺序证据实验中使用的 GP 作为基线，可能不代表所有传统代理的行为。

---

## 364. 3D Printing of Passively Actuated Self-Folding Robots with Integrated Functional Modules

**arXiv ID:** 2605.04757 | [PDF](https://arxiv.org/pdf/2605.04757v1)

**作者:** Gaolin Ge `[一作]` (University of Washington), Yiyue Luo `[通讯]` (University of Washington)

**通讯引用:** 1461 | [OpenAlex ID](https://openalex.org/A5007246518)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

用低成本弹性带和3D打印导电PLA网格实现自驱动自折叠机器人，并通过平面组装方式实现电子元件与传感器的精准安装；

**💡 创新点**

创新点包括：弹性驱动的自折叠机制、闭式折叠方程与设计映射、导电PLA作为电容触摸电极、以及统一的I/O平台实现模块间复用；

**🔧 技术方法**

主要技术手段为：导电PLA 3D打印、弹性带与打印钩子配合的折叠机理、基于刚度与弹性带张力的折叠方程、MPR121电容触摸传感、Hall效应磁感应、ERM电机驱动；

**📊 数据集**

未使用公开数据集，实验数据来自对不同铰链厚度、弹性带尺寸与钩距组合的折叠角度测量，以及传感器阈值与SNR评估；

**📈 对比分析**

通过折叠模型预测与实际测量对比，误差在5%以内；传感器SNR分别在18–20 dB，触摸检测准确率超过98%；运动轨迹实验验证了电机控制与磁对接的可行性；

**⚠️ 局限性**

局限性包括：铰链疲劳、弹性带蠕变、摩擦影响折叠精度；ERＭ驱动受表面影响；导电PLA电阻大、噪声高；缺乏长期可靠性与多材料复合能力；

---

## 365. Ensuring Reliability in Programming Knowledge Tracing: A Re-evaluation of Attention-augmented Models and Experimental Protocols

**arXiv ID:** 2605.04727 | [PDF](https://arxiv.org/pdf/2605.04727v1)

**作者:** Jaewook Kim `[一作]` (Korea University), Hyeoncheol Kim `[通讯]` (Korea University)

**通讯引用:** 2372 | [OpenAlex ID](https://openalex.org/A5021651278)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新评估并纠正了编程知识追踪（PKT）模型的实现与评估偏差，揭示了因果完整与超参数一致性对性能估计的影响。

**💡 创新点**

提出了一套因果完整、可重复的评估协议，并系统分析了任务特征与最大序列长度对模型性能的关键作用。

**🔧 技术方法**

采用代码结构化表示（AST+attention或transformer）、RNN/Transformer 序列预测、网格搜索调参与交叉验证等技术。

**📊 数据集**

在 CodeWorkout 数据集上进行实验。

**📈 对比分析**

通过在同一调参与序列构造框架下对 DKT、Code‑DKT 与 ECKT 进行比较，发现注意力增强模型的优势显著缩小，传统 DKT 在部分任务上仍能保持竞争力，且性能高度依赖任务与序列长度。

**⚠️ 局限性**

实验仅限于 CodeWorkout 数据集和部分 PKT 模型，缺乏在更广泛数据和多种 PKT 架构上的验证。

---

## 366. Tree-based Credit Assignment for Multi-Agent Memory System

**arXiv ID:** 2605.04811 | [PDF](https://arxiv.org/pdf/2605.04811v1)

**作者:** Marina Mao `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 26704 | [OpenAlex ID](https://openalex.org/A5100732436)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TreeMem框架，使用树形回放与Monte Carlo平均对多代理内存系统进行强化学习训练，使各代理获得基于最终奖励的细粒度信用信号。

**💡 创新点**

创新点在于不需要人工设计任务特定奖励，利用树形结构把最终奖励拆解为各代理的贡献，实现无需额外注解的信用分配；同时将这种分配与GRPO联合优化，提升代理专业化。

**🔧 技术方法**

核心技术包括树形rollout、Monte Carlo信用估计、基于GRPO的分组相对优化、clip策略、长度惩罚、LLM内存生成与检索模型。

**📊 数据集**

实验数据集包括PersonaMem（三种长度32K/128K/1M）、LongMemEval以及LOCOMO，用以验证在不同长序列QA与推理任务上的效果。

**📈 对比分析**

与基线（context-based、prompt-driven、RL-optimized）进行对比，TreeMem在PersonaMem上准确率提升约7–8%，在LongMemEval和LOCOMO的F1/B1指标上均取得最高分；同时在训练时间和标记效率上也优于对照方法。

**⚠️ 局限性**

局限性包括：树形回放在训练时仍带来额外计算开销，需要合理设置分支数与组大小；对不同LLM模型的泛化程度还有待进一步验证；缺乏对极大规模历史长度（>1M）外的实证。

---

## 367. DecodingTrust-Agent Platform (DTap): A Controllable and Interactive Red-Teaming Platform for AI Agents

**arXiv ID:** 2605.04808 | [PDF](https://arxiv.org/pdf/2605.04808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 368. Dr-PoGO: Direct Radar Pose-Graph Optimization

**arXiv ID:** 2605.04806 | [PDF](https://arxiv.org/pdf/2605.04806v1)

**作者:** Cedric Le Gentil `[一作]` (University of Toronto), Timothy D. Barfoot `[通讯]` (University of Toronto)

**通讯引用:** 8087 | [OpenAlex ID](https://openalex.org/A5004788089)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

开发了一种基于二维 FMCW 雷达的 SLAM 系统 Dr-PoGO，能够在恶劣天气下实现高精度定位与地图构建。

**💡 创新点**

创新点包括：①将直接雷达扫描注册（DRO）用于里程计与闭环校正；②引入无初始猜测的粗略基于特征的配准与精细直接配准的粗细配准管线；③在闭环检测中使用 RaPlace 的 Radon 变换，配合自定义的归一化交叉相关阈值实现可靠闭环验证；④将所有相对姿态与闭环约束纳入 SE(2) 位姿图全批优化。

**🔧 技术方法**

使用的技术包括：连续时间直接雷达里程计（DRO）、RaPlace（基于 Radon 变换的雷达场所识别）、SIFT+RANSAC 的特征粗配准、GPU 加速的连续交叉相关直接配准、SE(2) 位姿图优化（带 Cauchy 损失和陀螺仪偏置建模）。

**📊 数据集**

在 Boreas 与 Boreas-RT 两个真实车载雷达数据集上进行评估，数据涵盖多季节、多天气、不同路况（住宅区、工业区、天空桥、森林农场）以及自采集的高难度环境。

**📈 对比分析**

与 Navtech-SLAM、TBV-SLAM 两个开源雷达 SLAM 基线对比，Dr-PoGO 在 ATE 平均约 0.82 m、EPE 低于 1.5 m，明显优于对比方法；在自采集数据集上与 LiDAR 基线比较，Dr-PoGO 的 ATE 低至 0.75 m，性能提升约四倍。

**⚠️ 局限性**

局限性主要体现在：①对陀螺仪的依赖较大，缺失陀螺仪会显著降低精度；②闭环检测依赖 RaPlace 的关键帧策略，可能在极端动态场景下漏检；③系统仍使用两帧雷达扫描进行里程计，无法充分利用连续时间轨迹信息；④对超长循环闭环的鲁棒性尚待进一步验证。

---

## 369. OpenWatch: A Multimodal Benchmark for Hand Gesture Recognition on Smartwatches

**arXiv ID:** 2605.04791 | [PDF](https://arxiv.org/pdf/2605.04791v1)

**作者:** Pietro Bonazzi `[一作]` (ETH Zuerich), Michele Magno `[通讯]` (ETH Zuerich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了公开的多模态手势识别基准 OpenWatch，并设计了轻量级的 MixToken 结构用于腕部手势识别

**💡 创新点**

创新点在于：①首次公开 IMU+PPG 级联数据的全新基准；②将 PPG 与 IMU 结合并证明其显著提升性能；③提出 MixToken 混合专家模型，兼顾局部滤波特征与全局统计特征，显著小于基础模型且性能更优

**🔧 技术方法**

采用了同步 IMU 与 PPG 采集、滤波器组、Transformer 统计编码、混合专家融合以及 LoRA 参数高效微调等技术

**📊 数据集**

使用 OpenWatch 数据集：10+ 小时、50 名受试者、59 类手势（包含正负标签）

**📈 对比分析**

与 NormWear、InceptionTime、Hydra、Apple-CNN 等传统/基础模型对比；MixToken 在 clip-level macro‑F1 上达 90% 以上，参数仅 223k，显著优于 136M 参数的 NormWear 与 51k 参数的 Apple‑CNN

**⚠️ 局限性**

局限包括：仅评估 5 个命令手势；数据收集为半结构化、受限于 50 名受试者；缺乏个性化适配与长期真实环境验证

---

## 370. Long-Term Risks of IoT Devices: The Case of the Smart Fridge

**arXiv ID:** 2605.04787 | [PDF](https://arxiv.org/pdf/2605.04787v1)

**作者:** Erik Buchmann `[一作]` (Leipzig University), Erik Buchmann `[通讯]` (Leipzig University)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5088059746)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统化方法识别并编制了智能冰箱在10年以上使用寿命期间可能出现的长期风险，重点关注合规、经济和运营层面。

**💡 创新点**

创新点在于将BSI‑200‑3标准与设计科学研究方法结合，构建资产矩阵和风险归纳流程，形成面向长期视角的风险目录，可推广至其他智能家电。

**🔧 技术方法**

使用的技术主要是风险识别和分类方法、IT架构建模以及风险归纳与合并技术，并结合文献综述构建知识库。

**📊 数据集**

本文未使用公开数据集，而是基于三款典型智能冰箱（Bosch、Samsung、LG）的产品手册与官方文档进行案例分析。

**📈 对比分析**

文章未提供实验对比或性能评估，仅通过文献佐证每个风险的合理性与完整性，说明方法在理论层面可覆盖全部长期风险。

**⚠️ 局限性**

局限性包括仅针对智能冰箱进行案例研究，缺乏对不同家电的实证验证；方法依赖专家经验，缺少量化评估；并未对风险概率与影响进行定量化建模。

---

## 371. AFL-ICP: Enhancing Industrial Control Protocol Reliability via Specification-Guided Fuzzing

**arXiv ID:** 2605.04760 | [PDF](https://arxiv.org/pdf/2605.04760v1)

**作者:** Jiaying Meng `[一作]` (Zhongguancun Lab), Ke Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 12089 | [OpenAlex ID](https://openalex.org/A5100665814)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了 AFL-ICP，一个基于大语言模型的规范驱动工业控制协议（ICP）模糊测试框架，自动化完成多模态规范解析、协议自适配、种子生成以及差分检测，最终发现 24 个此前未知的漏洞。

**💡 创新点**

创新点在于：①构建上下文感知规范形式化管道，将复杂 PDF 规范转化为可执行语法；②利用 LLM 驱动协议自适配与种子合成，实现在新协议上的零门槛快速扩展；③设计 LLM 差分检查器，跨实现输出与规范对照，检测隐蔽语义与逻辑缺陷；④将 LLM 集成于模糊测试整个生命周期，实现完全 AI 自治。

**🔧 技术方法**

所用技术包括：Gemini‑2.5‑Pro、Claude Code、GPT‑4o 等大型语言模型；Mistral OCR 进行多模态 PDF 解析；JSON Schema、Mermaid 等结构化表示；AFLNet 为基础模糊引擎；代码审计、自动化自适配代理；差分分析与后置 LLM 合规检查。

**📊 数据集**

实验数据集涵盖四大工业控制协议（Modbus TCP、EtherNet/IP、IEC 104、SLMP）的开源与闭源实现；对应的 PDF 规范文件；以及示例网络流量，用于指导种子生成和自适配。

**📈 对比分析**

与 AFLNet 与 ChatAFL 通过 5 次 24 h 统一实验进行对比；AFL‑ICP 在状态覆盖上提升约 46–53%，转移覆盖提升约 64%，代码覆盖提升约 12–13%；在七个目标中，AFL‑ICP 发现 77% 更多内存安全漏洞，并新增 24 个语义/逻辑漏洞。

**⚠️ 局限性**

局限性包括：仍受 LLM 对规范理解精度与上下文窗口限制；对高维度大文档的处理仍需分段与重构；闭源实现的验证仍需人工确认；LLM 推理主要用于离线阶段，未能实现实时在线检测；工具在不同协议间的通用化仍待进一步验证。

---

## 372. AxMoE: Characterizing the Impact of Approximate Multipliers on Mixture-of-Experts DNN Architectures

**arXiv ID:** 2605.04754 | [PDF](https://arxiv.org/pdf/2605.04754v1)

**作者:** Omkar B Shende `[一作]` (IIT Dharwad), Gayathri Ananthanarayanan `[通讯]` (IIT Dharwad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了近似乘法器对Mixture‑of‑Experts（MoE）DNN架构的影响，比较了Hard、Soft、Cluster三种MoE变体与密集基线在ResNet‑20、VGG11_bn、VGG19_bn与ViT‑Small上的表现，并进行近似感知再训练。

**💡 创新点**

首次从硬件算术误差角度探究动态路由与近似计算的交互，揭示不同网络结构和路由策略对误差恢复的差异，并给出了功耗‑精度 Pareto 前沿，提出针对MoE的近似硬件‑软件协同设计思路。

**🔧 技术方法**

使用EvoApproxLib的8位近似乘法器、LUT仿真、GPU加速的TFApprox/TransAxx近似感知训练框架，集成Soft、Hard、Cluster MoE实现，并建立功耗归一化模型进行性能评估。

**📊 数据集**

使用CIFAR‑100数据集评估CNN模型（ResNet‑20、VGG11_bn、VGG19_bn），使用Tiny ImageNet评估ViT‑Small。

**📈 对比分析**

通过在不重训练和5轮近似感知再训练两阶段分别测量Top‑1准确率与归一化功耗；结果显示密集模型在CNN中最具鲁棒性，VGG在极度近似下无法恢复；ResNet‑20在所有乘法器上实现全量恢复；在ViT‑Small中，Hard MoE在激进近似下优于Dense，显示出更平滑的精度衰减。

**⚠️ 局限性**

研究仅覆盖八种固定近似乘法器，未涉及更深网络或不同专家数的影响；Cluster MoE在VGG上表现不佳；重训练仅5轮，可能不足以进一步恢复；未对硬件实现细节（如乘法器位宽、时序）进行深入分析。

---

## 373. VC-FeS: Viewpoint-Conditioned Feature Selection for Vehicle Re-identification in Thermal Vision

**arXiv ID:** 2605.04750 | [PDF](https://arxiv.org/pdf/2605.04750v1)

**作者:** Yasod Ginige `[一作]` (University of Sydney), Ranga Rodrigo `[通讯]` (University of Moratuwa)

**通讯引用:** 868 | [OpenAlex ID](https://openalex.org/A5058175691)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种视角条件化的车辆/船舶热像重识别算法，并实现热像目标检测与跟踪的端到端系统。

**💡 创新点**

创新点包括：① 将前侧后侧等视角划分为独立特征空间并按面积比例加权；② 在热像域微调预训练ViT并结合ArcFace实现多空间判别；③ 首次公开包含COCO标注的热像海事数据集。

**🔧 技术方法**

使用技术包括ViT Transformer、ArcFace映射、Triplet+ID 损失、Encoder-Decoder前景掩模、TraDeS多目标跟踪、SPAns视角分割，评估指标为MOTA、IDF1、Top‑1/Top‑5、mAP。

**📊 数据集**

使用的数据集有：自建热像海事数据集、RGBNT100（IR）、VeRi776、VehicleID、VesselID-539、SMD 等。

**📈 对比分析**

通过与SPAN、ViT Base 等基线在RGB与IR两域对比，本文方法在mAP上提升19.7%（RGBNT100）和12.8%（自建热像数据集），Top‑1/Top‑5 与MOTA 亦显著优于对手。

**⚠️ 局限性**

局限性包括：对遮挡、极端低光或恶劣天气场景的鲁棒性未充分验证；视角信息依赖准确分割，分割误差会影响性能；需要大量标注数据，训练成本高。

---

## 374. MixINN: Accelerating Plant Breeding by Combining Mixed Models and Deep Learning for Interaction Prediction

**arXiv ID:** 2605.04744 | [PDF](https://arxiv.org/pdf/2605.04744v1)

**作者:** Aike Potze `[一作]` (Wageningen University & Research), Ioannis N. Athanasiadis `[通讯]` (Wageningen University & Research)

**通讯引用:** 6353 | [OpenAlex ID](https://openalex.org/A5070260526)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过混合模型提取高质量基因-环境交互标签，并利用两塔神经网络预测玉米在未来环境下的产量排名。

**💡 创新点**

将传统混合模型的方差协方差分解与深度学习相结合，先用FA混合模型分离基因、环境和交互效应，再用结构化优化训练两塔网络，从而显著提升基因排名预测。

**🔧 技术方法**

使用线性混合模型（factor‑analytic VCOV估计）、两塔多层感知机（MLP）、分阶段结构化训练和重构交互标签，并通过RMSE、MAE、相关系数等指标评估。

**📊 数据集**

采用 Genomes to Fields 2022 Maize G×E Prediction Challenge 数据集（约12.4万条记录，4,417种玉米品种，212个环境，20,000基因标记+33环境特征）。

**📈 对比分析**

与前十名基线（GBLUP、G×EBLUP、SINN 等）在排名指标（r_j、ρ_j）和回归指标（RMSE、MAE）上对比，MixINN 在排名上优于所有基线，提升约10% r_j、20% ρ_j；在实际选育收益上，选取20%基因时Yield提升5.8%，超过第二佳模型15.8%。

**⚠️ 局限性**

两步分离导致统计模型误差向神经网络传播；对极少数最佳基因（<5%）的推荐性能略逊于 G×EBLUP；需进一步实现端到端的随机效应或高斯过程集成，以进一步提升模型鲁棒性。

---

## 375. Hierarachical Multiagent Reinforcement Learning for Multi-Group Tax Game

**arXiv ID:** 2605.04741 | [PDF](https://arxiv.org/pdf/2605.04741v1)

**作者:** Honglei Guo `[一作]` (Zhejiang University), Yexin Li `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了多组层级税收博弈模拟环境，并使用多智能体强化学习对政府-家庭互动进行学习；

**💡 创新点**

提出了层级多组博弈框架（MHG），并结合层次采样、课程学习与闭环顺序更新的双层 MARL 训练策略，以实现跨组竞争与组内领导者-跟随者关系的联合学习；

**🔧 技术方法**

采用了 on‑policy 双层 MARL（IPPO/HAPPO/HAA2C），结合层次采样、Curriculum Learning（CL）与 Closed‑Loop Sequential Update（SU）等技术；

**📊 数据集**

使用基于 Bewley‑Aiyagari（组内）与 Zodrow‑Mieszkowski（组间）理论构建的税收博弈仿真环境，数据来源为仿真产生的经济指标；

**📈 对比分析**

与基础 IPPO/IA2C 等对照实验比较，结果表明加入 CL 与 SU 能将博弈持续时间提升约60.92%，GDP 差距下降约44.12%，并显著提升税收政策的稳定性与公平性；

**⚠️ 局限性**

目前仅支持最多三组，且假设家庭规模不变，无法处理更大规模或动态人口的情况。

---

## 376. Using Common Random Numbers for Simulation-based Planning with Rollouts

**arXiv ID:** 2605.04732 | [PDF](https://arxiv.org/pdf/2605.04732v1)

**作者:** Sandarbh Yadav `[一作]` (IIT Bombay), Shivaram Kalyanakrishnan `[通讯]` (IIT Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究使用公共随机数（CRN）在模拟规划中降低价值差异估计方差，并提出深度依赖采样方案；

**💡 创新点**

提出在策略在某深度后相同的情形下，深度依赖估计器的方差始终不大于传统独立估计器，并通过种子控制实现CRN；

**🔧 技术方法**

利用MDP理论推导、共随机数技术、rollout规划、UCT搜索以及前向/后向采样过程；

**📊 数据集**

实验使用合成MDP、固定期变额寿险基金（FTVAF）环境以及Ludo棋盘游戏；

**📈 对比分析**

通过对比独立、依赖、深度依赖三种种子策略，在合成任务、FTVAF和Ludo中以平均回报或胜率评估，发现深度依赖与依赖在样本数有限时显著优于独立且几乎等效；

**⚠️ 局限性**

仅给出方差上界未转化为样本复杂度，未对动态采样算法如UCT给出理论分析，且在某些特殊MDP下可能出现方差更高的情况。

---

## 377. Morphology-Guided Cross-Task Coupling for Joint Building Height and Footprint Estimation

**arXiv ID:** 2605.04731 | [PDF](https://arxiv.org/pdf/2605.04731v1)

**作者:** Jinzhen Han `[一作]` (Sungkyunkwan University), HongSik Yun `[通讯]` (Sungkyunkwan University)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5113713719)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了建筑高度与占地面积在城市建模中的紧密耦合关系，并提出了在深度学习框架中显式编码该耦合的方案

**💡 创新点**

创新点在于通过BF引导的任务解码器(BGTD)和形态一致性损失(MCL)两种机制，将占地面积信息主动注入高度预测，突破了传统共享编码器的耦合上限

**🔧 技术方法**

使用Swin Transformer单阶段骨干网络、Adaptive Modality Gating Encoder、Multi-Scale Morphology Pooling以及交叉注意力门控与一致性损失等技术

**📊 数据集**

在54个城市的SHAFTS参考数据集上进行训练与评估，输入为Sentinel‑1 SAR、Sentinel‑2多光谱及DEM栅格，采用90×90像素（10 m分辨率）窗口

**📈 对比分析**

相较于相同感受野的Swin-MTL基线，BH的测试RMSE从3.39 m降至3.15 m（≈7.1 %提升），BF指标几乎保持不变；BGTD和MCL分别贡献约0.11 m的误差下降，整体提升约0.24 m

**⚠️ 局限性**

主要限制包括高密度高楼区样本稀缺导致误差聚集、对低占地面积区域的改进有限以及对VHR数据的验证尚待进一步研究

---

## 378. RecGPT-Mobile: On-Device Large Language Models for User Intent Understanding in Taobao Feed Recommendation

**arXiv ID:** 2605.04726 | [PDF](https://arxiv.org/pdf/2605.04726v1)

**作者:** Bin Zhang `[一作]`, Yipeng Yu `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在移动端实现基于大语言模型的用户意图理解与下一查询推荐

**💡 创新点**

首次将轻量化LLM部署在手机端并引入自适应提示构造与触发机制

**🔧 技术方法**

模型压缩（量化+LoRA）、自适应提示、行为序列建模、意图漂移检测

**📊 数据集**

淘宝移动端真实用户行为日志、共购矩阵、LLM重写数据和人工标注

**📈 对比分析**

离线用Qwen3系列模型评估S_sem/S_logic/S_style，量化LoRA获得0.79-0.83总分；在线A/B测试提升点击+1.8%、支付+2.7%、GMV+2.5%

**⚠️ 局限性**

受限于模型规模与设备算力，仍需进一步提升推理速度和跨场景泛化

---

## 379. Hearing the Ocean: Bio-inspired Gammatone-CNN framework for Robust Underwater Acoustic Target Classification

**arXiv ID:** 2605.04839 | [PDF](https://arxiv.org/pdf/2605.04839v1)

**作者:** Rajeshwar Tripathi `[一作]` (Bharat Electronics Limited), Neel Kanth Kundu `[通讯]` (IIT Delhi)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并验证了一套基于听觉模拟的Gammatone滤波器组与轻量CNN的海洋声学目标识别框架

**💡 创新点**

创新点在于使用固定的ERB尺度Gammatone滤波器以高分辨率捕捉低频引擎谐波，同时保持极低的计算开销

**🔧 技术方法**

核心技术包括Gammatone滤波器组、Hilbert包络提取、对数动态范围压缩、3通道Cochleagram生成以及大卷积核轻量CNN

**📊 数据集**

实验使用公开的VTUAD数据集（多距离、不同船舶类别）进行训练与评估

**📈 对比分析**

与CWT、MFCC等传统特征在相同CNN架构下对比，获得98.41%准确率、0.971 Cohen’s Kappa，推理时延仅0.77 ms，显著优于现有方法

**⚠️ 局限性**

局限性包括对极少样本类别（如Passenger）的召回偏高导致误判、缺乏自适应滤波器调优、在非VTUAD海况下的泛化能力待验证

---

## 380. Replay-Based Continual Learning for Physics-Informed Neural Operators

**arXiv ID:** 2605.04832 | [PDF](https://arxiv.org/pdf/2605.04832v1)

**作者:** Yizheng Wang `[一作]` (Tsinghua University), Yinghua Liu `[通讯]` (Tsinghua University)

**通讯引用:** 4987 | [OpenAlex ID](https://openalex.org/A5103507891)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于重放的持续学习框架，用于物理信息神经算子（以Transolver为主）在新分布下的快速适应与旧分布的记忆保持；

**💡 创新点**

创新点在于：①仅使用物理方程驱动训练，完全无标签；②聚焦“差错样本”进行重放，显著降低训练成本；③通过教师模型蒸馏与LoRA低秩微调实现记忆稳定与学习弹性；④结合监督微调进一步提升精度；

**🔧 技术方法**

技术包括物理驱动损失（强/能量形式）、Transformer‑based Transolver、重放式持续学习、教师蒸馏、LoRA低秩微调以及监督微调（SFT）；

**📊 数据集**

使用三组典型问题的数据集：1）Darcy流动（10组不同分布的随机场）；2）二维脑肿瘤（1000张人造脑图 + 200张肿瘤样本）；3）三维TPMS（600个“Solid‑networks”和600个“Sheet‑networks”共1200个3D结构）；

**📈 对比分析**

与全量联合训练（Joint）和传统全量训练比较，重放式持续学习在保持对旧任务性能（误差与Joint相近）的同时，显著降低了内存使用和训练时间；SFT进一步将误差压至5%以下；在三组任务中，持续学习在大数据与小数据场景均能保持优异性能；

**⚠️ 局限性**

限制：①仅使用PDE误差做样本评分，缺乏目标导向误差评估；②实验基于合成数据，尚未验证真实MRI数据；③目前仅在Transolver上验证，需扩展至其他算子与数据驱动方式；

---

## 381. Improving FMQA via Initial Training Data Design Considering Marginal Bit Coverage in One-Hot Encoding

**arXiv ID:** 2605.04825 | [PDF](https://arxiv.org/pdf/2605.04825v1)

**作者:** Taiga Hayashi `[一作]` (Keio University), Shu Tanaka `[通讯]` (Keio University)

**通讯引用:** 1542 | [OpenAlex ID](https://openalex.org/A5057961231)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在Factorization-Machine+Ising机（FMQA）框架中引入基于拉丁超立方体（LHS）和Sobol序列的初始训练数据生成方法，确保在使用one-hot编码时每个二进制位至少出现一次，从而提升翼型优化问题的搜索效果。

**💡 创新点**

创新点在于提出并验证“完整边缘位覆盖”策略，消除因位缺失导致的FM参数偏差，并证明其在高维问题中显著提升FMQA的性能。

**🔧 技术方法**

使用的技术包括Factorization Machine、Ising机（模拟退火采样）、一热编码、拉丁超立方体采样、Sobol低差异序列以及AdamW优化器。

**📊 数据集**

实验数据集为人力飞机翼型优化基准HPA103（HPA103-1 17维，HPA103-2 32维），每个变量离散化为32级。

**📈 对比分析**

通过与随机搜索、GP-BO、NSGA-II以及不使用边缘覆盖的Conv-FMQA比较，LHS-FMQA和Sobol'-FMQA在200次评估预算下均取得比Conv-FMQA更高的平均巡航速度，尤其在HPA103-2上提升约0.33–0.35 m/s，显著优于其他基线。

**⚠️ 局限性**

限制包括：仅保证边缘位覆盖，未覆盖位对位间交互；LHS与Sobol同时提供空间填充属性，难以单独评估边缘覆盖效果；需要初始样本数等于离散化级数且为2的幂，限制低预算场景；对局部搜索偏差的影响尚未阐明。

---

## 382. On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference

**arXiv ID:** 2605.04901 | [PDF](https://arxiv.org/pdf/2605.04901v1)

**作者:** Zhengyi Li `[一作]` (Shanghai Jiao Tong University), Jingwen Leng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2769 | [OpenAlex ID](https://openalex.org/A5003939279)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Transformer 模型的安全推理中使用的 shuffling 防御提出了一种攻击方法，能够对不同随机排列的激活向量进行对齐，并利用对齐后的向量恢复模型权重。

**💡 创新点**

创新点在于：① 利用近似激活向量的相似性通过匈牙利算法高效求解变换矩阵；② 通过安全推理中固定点截断的随机 1-bit 误差产生可对齐的激活；③ 展示即使权重仅被行列置换，仍能保持前向传播功能。

**🔧 技术方法**

使用的技术包括：固定点数的加密推理协议、线性层权重求逆、匈牙利分配算法、条件数截断（伪逆）以及对齐误差分析。

**📊 数据集**

主要实验数据集：Pythia-70m、GPT-2 两大 Transformer 模型；用于评估最终模型性能的 WikiText 语料库。

**📈 对比分析**

与现有安全推理方法相比，本攻击在仅约 1 美元的查询成本下，能够恢复 L1‑norm 差异在 10⁻⁴~10⁻² 范围内的权重；在 WikiText 上微调后，偷取模型的困惑度几乎可与原模型持平，显示攻击的实用性和高效性。

**⚠️ 局限性**

局限性在于：攻击对模型规模敏感，随着模型尺寸增大恢复精度下降；仅在半诚实模型中有效；需要足够多的查询来构造足够接近的激活。

---

## 383. Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall

**arXiv ID:** 2605.04897 | [PDF](https://arxiv.org/pdf/2605.04897v1)

**作者:** Joshua Adler `[一作]` (Sauron Labs), Guy Zehavi `[通讯]` (Sauron Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 True Memory 系统，将代理记忆从传统的“提取-存储-检索”模式转变为以检索为中心的多层检索管线，保持对话事件的原始文本，避免在提取阶段丢失信息。

**💡 创新点**

创新点包括：①三信号编码门（新颖性、显著性、预测误差）在摄取时做智能过滤；②六层检索架构（编码、合并、预测、语义匹配、稀释等）在回忆时实现多阶段排序；③仅使用单一 SQLite 文件，无向量索引库、图数据库或 GPU，极大降低部署成本。

**🔧 技术方法**

使用技术：SQLite FTS5（全文检索）+ sqlite-vec（向量检索）；gzip 压缩衡量新颖性；词法/语义分数结合的 RRF 融合；跨编码 reranker；语言模型嵌入与推理模型；门控信号通过对比、嵌入差异等方式计算。

**📊 数据集**

实验数据集：LoCoMo（1,540 题）、LongMemEval（500 题）、BEAM-1M（700 题，1M token）。

**📈 对比分析**

采用统一的评测 harness（3 次跑、语义匹配判定、检索 top‑k 预筛选），与 Mem0、Supermemory、Zep、EverMemOS、RAG‑ChromaDB 等对标。性能表现：LoCoMo 93.0%（仅低于 EverMemOS 1.5 pt，运行在单核 CPU 仅 SQLite）；LongMemEval 87.8% 领先所有对手；BEAM‑1M 76.6% 超 Hindsight 73.9%，证明在长时间跨度对话中的优越性。

**⚠️ 局限性**

局限性：①编码门功能未在公开基准上评估，缺乏针对选择性摄取的评价；②最长评估仅 1M token，未覆盖数月甚至数年对话；③采用语义匹配评判，无法直接与严格匹配基准比较；④缺乏针对更长时间跨度的标准 benchmark。

---

## 384. Regime-Conditioned Evaluation in Multi-Context Bayesian Optimization

**arXiv ID:** 2605.04895 | [PDF](https://arxiv.org/pdf/2605.04895v1)

**作者:** Noel Thomas `[一作]` `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence), Noel Thomas (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究转移贝叶斯优化（Transfer‑BO）中先验质量、预算比例与采集器性能的关系，提出可观测的 Portable Regime Score (PRS) 作为预实验诊断并验证其对结果的预测能力；同时设计 RegimePlanner 自适应策略，证明 PRS 可在线被利用；通过对 40 篇论文的审核揭示“无免费排行榜”问题，指出大多数研究未控制 regime 变量，导致 ATE 作为算法排名无效。

**💡 创新点**

创新点在于：1) 提出 PRS（(B/|A|)(1-ρ)）作为统一的 regime 诊断，能在实验前预测哪类采集器会占优；2) 通过层级模型与线性期望证明，阐明当 CATE 在 regime 轴上符号变化时，任何 ATE 都可由 benchmark 混合构造，形成 No‑Free‑Leaderboard 定理；3) 设计 RegimePlanner，利用在线 ρ 估计实现自适应切换，实验证明可显著提升性能。

**🔧 技术方法**

技术手段包括：统计分析（Spearman 相关、层级贝叶斯模型、Bootstrap、线性期望证明）、贝叶斯优化代理（GP、Greedy、UCB、Thompson、REIGN）、实验设计与评估（Buchwald–Hartwig、GDSC2、HPO‑B 等基准）、数据预处理（先验权重、EMA、Tanimoto 核）和可视化（两轴 regime 图、聚类分析）。

**📊 数据集**

使用的数据集与基准：1) Buchwald–Hartwig 化学反应优化（含 4×4 prior×acquisition 设计）；2) GDSC2 药物响应基准（多药物/细胞系）；3) HPO‑B 超参数搜索基准（16 种搜索空间）；4) SciPlex3、Shifrut2018 等小规模实验。

**📈 对比分析**

对比方法：Greedy、UCB、Thompson、REIGN、RegimePlanner。实验结果显示：在弱先验或低预算时 Greedy 通常占优；在强先验或高预算时 UCB/REIGN 更好；RegimePlanner 在所有基准上均优于任何固定策略，尤其在 GDSC2（B=50）上 +18% 超过 {Greedy,UCB} 轨迹最优，且在 HPO‑B（B=100）上在所有 16 个搜索空间上全胜。

**⚠️ 局限性**

局限性包括：1) PRS 仅对可观测的预算比例与先验相关性有效，强先验下失效；2) 需要至少 3 个 pilot 任务才能估计 ρ，阈值对不同噪声水平敏感；3) 在部分基准（如高噪声或 K 小于阈值）下 PRS 预测准确率低于 90%；4) RegimePlanner 的阈值 θ 需要在特定基准上交叉验证，可能不具普适性。

---

## 385. A Comparative Study of PyCaret AutoML and CNN-BiLSTM for Binary Hate Speech Detection in Indonesian Twitter

**arXiv ID:** 2605.04885 | [PDF](https://arxiv.org/pdf/2605.04885v1)

**作者:** Tanty Widiyastuti `[一作]` (Institut Teknologi Sumatera), Martin Clinton Tosima Manullang `[通讯]` (Institut Teknologi Sumatera)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比了 PyCaret AutoML 与 CNN‑BiLSTM 两种模型在印尼推特仇恨言论二分类任务上的性能，采用共享预处理流程保证公平性。

**💡 创新点**

创新点在于：①将传统模型通过 AutoML 统一基线并强化词典与 TF‑IDF 特征；②使用 CNN‑BiLSTM 捕捉短文本的局部词组与双向上下文，并直接与 AutoML 结果在同一基准上进行对比。

**🔧 技术方法**

技术手段包括：PyCaret AutoML（TF‑IDF+辱骂词计数、随机森林、SVM、朴素贝叶斯等传统机器学习模型）；CNN‑BiLSTM（词嵌入→卷积层→双向 LSTM→输出层）深度学习架构。

**📊 数据集**

数据集为印尼推特仇恨言论语料库（13,130 条推文，采用 HS 标签做二分类）。

**📈 对比分析**

采用共享预处理、单一 20% hold‑out 测试集，评估准确率、精确率、召回率和 F1 分数。CNN‑BiLSTM 达到 83.8% 准确率、81.2% F1，优于 AutoML 中最强的随机森林 77.2%/77.0%，提升了 6.6 点准确率、4.2 点 F1。

**⚠️ 局限性**

局限性包括：只使用单一数据集和单次划分，缺乏交叉验证；模型可能存在过拟合风险；未测试更复杂的 transformer 等模型；结果仅针对印尼短文本，可能不具备普适性。

---

## 386. Anticipating Innovation Using Large Language Models

**arXiv ID:** 2605.04875 | [PDF](https://arxiv.org/pdf/2605.04875v1)

**作者:** Enrico Maria Fenoaltea `[一作]` (Universitat de Barcelona), Andrea Tacchella `[通讯]` (Centro Ricerche Enrico Fermi)

**通讯引用:** 2092 | [OpenAlex ID](https://openalex.org/A5070678305)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一种名为 TechToken 的 Transformer 模型，将 IPC 代码视为语言中的词元，在专利语料中捕捉技术语义收敛，并利用该收敛信号预测首次技术组合，同时在 IPC 分类、引用预测和标题‑摘要匹配等三项标准专利任务上进行基准比较。

**💡 创新点**

① 在语言模型词表中加入 IPC 代码词元，实现技术代码的上下文相关嵌入，捕捉技术多义性；② 发现并量化了技术语义收敛作为可观测的、提前多年（最高约 20 年）出现的创新前兆；③ 用轻量级模型在创新预测任务中大幅优于更大规模模型，证明结构化标签嵌入的通用价值。

**🔧 技术方法**

Transformer‑based 语言模型（BERT、LLaMA 3.1）在掩码语言建模下 fine‑tune，构造 TechToken 词元；利用余弦相似度的“上下文相似度”（CS）与前 1% 最高相似度聚合；采用 Chung‑Liu 随机双边图模型计算 z‑score 定义创新；评估指标包括 AUC‑ROC、宏/微 F1、MAP、MRR、AUC‑ROC 等。

**📊 数据集**

欧洲专利公报（European Patent Bulletin AB）数据，1980‑2024 年英文学专利约 1.3 百万条，IPC 代码截断至 group 级别，约 7200 代码。

**📈 对比分析**

与 BERT4Patents、BERT4Patents FT、LLaMA Patents、PatentSBERTa、Paecter 等基线模型在三大专利任务上对比。创新预测中，TechToken 的 AUC‑ROC 达 0.936（最高），远高于 LLaMA 0.856；IPC 分类宏 F1 为 0.488（最高），引用预测宏 F1 为 0.420（最高），标题‑摘要匹配 AUC‑ROC 0.994（最高）。整体表明 TechToken 在多项任务中均取得 state‑of‑the‑art 性能。

**⚠️ 局限性**

① 仅使用 IPC group 级别的粗粒度编码，无法捕捉组内微观创新；② 只关注二元组合，未考虑三元或更高阶组合；③ 仅处理英文专利，忽略多语种数据；④ 模型训练与评估时间窗口的固定性可能导致对未来变化的适应性有限。

---

## 387. SynConfRoute: Syntax-Aware Routing for Efficient Code Completion with Small CodeLLMs

**arXiv ID:** 2605.04894 | [PDF](https://arxiv.org/pdf/2605.04894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 388. Uncertainty-Aware Exploratory Direct Preference Optimization for Multimodal Large Language Models

**arXiv ID:** 2605.04874 | [PDF](https://arxiv.org/pdf/2605.04874v1)

**作者:** Huatian Zhang `[一作]` (University of Science and Technology of China), Yongdong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 35314 | [OpenAlex ID](https://openalex.org/A5046305086)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Uncertainty‑aware Exploratory Direct Preference Optimization (UE‑DPO) 用于多模态大语言模型的幻觉（hallucination）抑制。

**💡 创新点**

创新点在于将词级别的本体不确定性（epistemic uncertainty）与视觉敏感度相结合，动态调整学习压力，实现对模型认知缺陷的主动探索与补偿。

**🔧 技术方法**

技术包括：基于 DPO 的逆 KL 规则强化学习框架、使用图像模糊扰动估计不确定性、构造探索强度 λ 并对优/劣样本分别调节梯度，以及理论证明其等价于泛化优势函数。

**📊 数据集**

使用 RLHF‑V 与 RLAIF‑V 人类/AI 反馈的偏好数据，评估集包含 MMHal‑Bench、Object‑HalBench、AMBER‑g 与 AMBER‑d 等幻觉基准。

**📈 对比分析**

与现有基于偏好学习的幻觉抑制方法（如 V‑DPO、TPO、POVID、mDPO 等）比较，UE‑DPO 在对象级幻觉 CHAIRs/CHAIRi、MMHal‑Bench 的得分、AMBER‑g 的对齐度以及 AMBER‑d 的 F1 上均实现了显著提升，整体表现最为稳健。

**⚠️ 局限性**

局限性包括对模型感知瓶颈的依赖（对极小目标识别仍有挑战），以及在数据规模与多样性不足时对 AMBER‑d 的准确率略有下降。

---

## 389. 3D Ultrasound-Derived Pseudo-CT Synthesis Using a Transformer-Augmented Residual Network for Real-Time Operator Guidance

**arXiv ID:** 2605.04856 | [PDF](https://arxiv.org/pdf/2605.04856v1)

**作者:** Sapna Sachan `[一作]` (Indian Institute of Technology Guwahati), Amulya Kumar Mahto `[通讯]` (Indian Institute of Technology Guwahati)

**通讯引用:** 194 | [OpenAlex ID](https://openalex.org/A5043352470)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

利用3D超声生成伪CT（UD-pCT）作为实时操作员引导的参考图像。

**💡 创新点**

在残差U-Net结构中加入瓶颈Transformer，实现局部特征与全局上下文的融合，显著提升生成质量。

**🔧 技术方法**

使用3D残差编码-解码器、Transformer瓶颈、3D Conditional PatchGAN判别器以及像素+对抗混合损失。

**📊 数据集**

使用TRUSTED数据集中的配对3D肾脏超声和CT扫描。

**📈 对比分析**

与传统3D ResUNet、U-Net及扩散模型对比，PSNR 23.26 dB、SSIM 0.71，表现最佳。

**⚠️ 局限性**

配对数据量有限，导致模型泛化能力受限。

---

## 390. Measuring Psychological States Through Semantic Projection: A Theory-Driven Approach to Language-Based Assessment

**arXiv ID:** 2605.04873 | [PDF](https://arxiv.org/pdf/2605.04873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 391. Delta-Based Neural Architecture Search: LLM Fine-Tuning via Code Diffs

**arXiv ID:** 2605.04903 | [PDF](https://arxiv.org/pdf/2605.04903v1)

**作者:** Santosh Premi Adhikari `[一作]` (University of Wurzburg), Dmitry Ignatov `[通讯]` (University of Wurzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用大规模语言模型（LLM）生成针对现有神经网络架构的增量统一 diff（而非完整重写），从而在 LLM‑驱动的 NAS 过程中实现高效的架构改进；

**💡 创新点**

提出了 delta‑based 生成范式，结合迭代 LoRA 微调、MinHash‑Jaccard 结构新颖性过滤和跨六个数据集的统一评估，显著提升生成质量与 token 效率；

**🔧 技术方法**

使用 7B 级 LLM（DeepSeek‑Coder‑7B‑Instruct‑v1.5、Qwen2.5‑Coder‑7B‑Instruct、Mistral‑7B‑Instruct‑v0.3）配合 LoRA 微调、统一 diff 生成、MinHash‑Jaccard 新颖性过滤、一次性 epoch 代理验证等技术；

**📊 数据集**

在 LEMUR 数据集上的六个图像分类任务（CIFAR‑10、CIFAR‑100、MNIST、SVHN、ImageNette、CelebA）进行实验；

**📈 对比分析**

与全生成基线（DeepSeek‑Coder 50.6% valid、42.3% mean accuracy）和 Gu 的迭代冻结 LLM（71.5% one‑epoch accuracy）比较，delta 生成实现约 75–85% 的输出行压缩、有效率提升至 66–75%，平均一次性 epoch 准确率提升至 64–66%（对比 42.3%），最佳一次性 epoch 准确率均达到 99.5%（对比 64%），整体 token 消耗降低约 5.4 倍；

**⚠️ 局限性**

局限性包括：仅基于一次 epoch 的代理准确率，未验证长期训练效果；增量 diff 的上下文匹配错误率约 25–35%；对基线多样性依赖度高，单数据集实验受限；对更大模型或更复杂任务的适用性尚待验证；

---

## 392. BenCSSmark: Making the Social Sciences Count in LLM Research

**arXiv ID:** 2605.04886 | [PDF](https://arxiv.org/pdf/2605.04886v1)

**作者:** Arnault Chatelain `[一作]`, Didier Schwab `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 BenCSSmark，一个专注于社会科学任务的中文（法语）大模型基准，收集并整理了27个多元化任务的数据集，涵盖文本分类、框架检测、偏见检测、引用检测等多种类型；

**💡 创新点**

创新点在于将社会科学研究中高度专业化、视角多元的数据与标注引入LLM评测，强调多注释者视角、上下文与时间维度的多标签标注，并为模型提供跨学科、多语境的真实世界挑战；

**🔧 技术方法**

技术上主要采用标准NLP任务形式（二分类、多分类、多标签、跨度检测、共指解析）以及数据整理与元数据注释工具；

**📊 数据集**

使用了27个法语数据集，涵盖政治演讲、社交媒体、新闻报道、学术摘要等文本，任务包括框架检测、偏见检测、性别与社会阶层概念检测、引用与赞誉检测、政治预测等；

**📈 对比分析**

目前未给出模型的具体评测结果，作者建议使用常规指标（准确率、F1、召回率等）对每个子任务进行评估，并可通过多标签过滤实现针对性对比；

**⚠️ 局限性**

局限性包括：仅覆盖法语文本，数据规模有限；受限于版权和隐私导致数据不可公开；可能面临过度标准化导致的研究方向偏移；以及基准快速过时的风险。

---

## 393. From Classical to Quantum-Mechanical Data Assimilation: A Comparison between DATO and QMDA

**arXiv ID:** 2605.04881 | [PDF](https://arxiv.org/pdf/2605.04881v1)

**作者:** Emanuele Donno `[一作]` (University of Salento), Giovanni Aloisio `[通讯]` (University of Salento)

**通讯引用:** 3156 | [OpenAlex ID](https://openalex.org/A5068232148)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过统一的 Koopman 迁移算子理论框架，对两种基于算子的方法——DATO（基于转移算子）和 QMDA（量子力学数据同化）——进行比较，给出了它们的离线与在线计算复杂度、存储需求以及在 Lorenz‑63 典型测试上的实际性能。

**💡 创新点**

创新点在于：① 将两种不同的算子同化方法放入同一大O 复杂度分析框架；② 推导了“突破阈值” n* = L³/m，明确指出当系统维数 n 低于此阈值时 DATO 更优，反之 QMDA 更具优势；③ 结合理论与实验验证，展示了不同规模训练集与谱分辨率对两方法成本的影响。

**🔧 技术方法**

主要技术包括：核增广动态模式分解（kEDMD）与变量带宽核的稀疏化；Koopman 与 Perron–Frobenius 解析与数值逼近；投影到有限谱子空间；延迟坐标映射；大O 复杂度与存储分析；以及 Lorenz‑63 动力学的数值实验。

**📊 数据集**

使用数据集为 Lorenz‑63 系统的轨迹样本：DATO 采用约 2 800 个训练快照（m=2800），QMDA 采用约 64 000 个训练样本（N=64 000），并在不同观测分辨率（S=2000 与 S_QMDA=32）下进行对比。

**📈 对比分析**

比较方法包括：离线阶段的核矩阵构造、特征分解与算子材料化成本；在线阶段的预测、更新与状态重构成本；以及基于 n 与 L、m、S 的“突破阈值” n* 的理论与实验验证。性能结果显示：当 n ≪ n* 时，DATO 的在线成本 O(m n) 远低于 QMDA 的 O(L³)，而当 n ≫ n* 时，QMDA 由于其在线成本与 n 无关，显著优于 DATO；实验在 Lorenz‑63 上验证了这一趋势。

**⚠️ 局限性**

局限性包括：① 复杂度分析仅给出最坏情况的 O 量级，未考虑常数、BLAS/LAPACK 优化、并行与缓存效应；② 只评估了基础实现，未探讨 U^(1)⁽ᵗ⁾ 的矩阵幂策略；③ QMDA 目前不提供点估计与 OI/FSOI 等诊断；④ 仅在平稳、可观测的低维系统（Lorenz‑63）上验证，未在高维实际气候模型上进行测试。

---

## 394. Agentic Repository Mining: A Multi-Task Evaluation

**arXiv ID:** 2605.04845 | [PDF](https://arxiv.org/pdf/2605.04845v1)

**作者:** Johannes Härtel `[一作]` (Vrije Universiteit Amsterdam), Johannes Härtel `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 195 | [OpenAlex ID](https://openalex.org/A5087770738)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了基于 LLM 的代理模型通过执行 Bash 命令动态检索软件仓库上下文，并对四类仓库工件进行分类，与传统预先工程化上下文的 LLM 进行对比实验。

**💡 创新点**

创新点在于提出并验证了一种“代理式仓库挖掘”框架，让 LLM 能自行探索并获取上下文，从而消除手工上下文工程并提升模型在大规模仓库中的鲁棒性与资源利用效率。

**🔧 技术方法**

技术实现包括使用 Amazon Bedrock 的 Claude 3.7 Sonnet 与 Mistral Large 3 两大 LLM，结合 native 与 stop‑sequence 两种工具调用方式以及 prompt caching，构建在 Docker 沙箱中执行 Bash 命令的代理流程。

**📊 数据集**

实验数据来源于四个公开 MSR 分类任务：Herbold 等的 tangled‑commit 行级标签、Härtel 的安全审查评论、Levin 的维护活动 commit 以及 Munaiah 的仓库类型分类，并对原始数据做了可访问性与大小筛选。

**📈 对比分析**

比较方法采用统一 prompt、相同 LLM 与同一批样本，评估准确率、token/时间/成本、错误率等指标。结果表明，代理模型在避免上下文溢出、资源消耗上更具优势，准确率与预工程化 LLM 基本持平，部分任务甚至略有提升。

**⚠️ 局限性**

局限性包括：对大仓库的适用性有限；未涵盖更高级的代理策略或互联网访问；仅评估了现有公开模型，未对最新 LLM 做评测；存在潜在的训练数据泄漏风险；原始标签质量可能影响评估结果。

---

## 395. Toward an Understanding of Developer Behaviour while Using Bug Localization Tools

**arXiv ID:** 2605.04828 | [PDF](https://arxiv.org/pdf/2605.04828v1)

**作者:** Pablo Diaz Pedreira `[一作]` (Open University), Michel Wermelinger `[通讯]` (Open University)

**通讯引用:** 2855 | [OpenAlex ID](https://openalex.org/A5029025369)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在模拟真实开发环境下，对 11 位参与者进行 4 次 bug 定位任务的定性研究，记录其 think‑aloud 过程，并分析其使用 DreamLoc 工具（含文件列表、置信度、LLM 生成摘要）时的行为与决策。

**💡 创新点**

首次从开发者视角系统揭示 bug 定位工具的使用模式与局限性，发现工具信息并非单纯提高准确率，而是通过与报错信息、上下文交互影响搜索路径、决策与问题重构；强调工具需要传递不确定性、推理过程和上下文。

**🔧 技术方法**

使用改造后的 DreamLoc（机器学习模型 + 7 维文件/报错元数据），结合 Claude Sonnet 4.0 生成文件摘要；在 IntelliJ IDE 进行实验，采用归纳式编码、主题编码等定性分析技术。

**📊 数据集**

以 Tomcat 项目为背景，挑选 4 个闭源 bug 报告（BR 54087、54095、54124、54144），每个 BR 包含 1–3 个修复文件；使用 DreamLoc 在 Tomcat 上训练并在测试集生成推荐列表、置信度与摘要。

**📈 对比分析**

通过 11 名参与者完成 4 个任务（每个 15 min），观察工具信息对其搜索路径、决策和问题重构的影响；未采用传统准确率指标，而是以行为、时间与任务完成度为评价维度，指出工具准确率并非主要瓶颈，工具在短时间内仍能提供一定指导。

**⚠️ 局限性**

局限包括：样本量小、时间限制导致任务未完整完成；参与者对 Tomcat 代码不熟悉，影响搜索策略；仅评估单一工具和单一 LLM；未覆盖多文件推荐或更复杂的 bug 场景；当前分析仅为初步，后续需进一步挖掘完整数据。

---

## 396. Exploring Clustering Capability of Inpainting Model Embeddings for Pattern-based Individual Identification

**arXiv ID:** 2605.04904 | [PDF](https://arxiv.org/pdf/2605.04904v1)

**作者:** Jens van Bijsterveld `[一作]` (Leiden University), Rita Pucci `[通讯]` (Leiden University)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5090003852)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究通过将图像修复作为辅助任务，在深度学习中增强基于动物皮肤图案的个体识别。

**💡 创新点**

创新点在于利用四种先进的修复模型预训练编码器，并在此基础上进行最小化微调，证明皮肤图案对识别的重要性，同时提出基于LaMa编码器的最佳策略。

**🔧 技术方法**

采用AOT‑GAN、DeepFillV2、EdgeConnect、LaMa四种修复模型进行预训练，随后提取嵌入并加上线性分类头；还使用GradCAM、PCA/T‑SNE/UMAP可视化以及k‑means聚类进行分析。

**📊 数据集**

数据集来源于Haurum等人关于斑马鱼的视频帧，包含2224帧，其中约4422张用于分类，999张（含翻转）用于修复训练。

**📈 对比分析**

通过与传统CNN/ViT的对比、深浅反向传播、区域消融实验评估准确率、召回率、F1、聚类指标，结果显示LaMa+线性头在分类上优于现有模型，AOT‑GAN在聚类上表现最佳。

**⚠️ 局限性**

限制包括模型对不同发育阶段图像的鲁棒性不足、修复预训练与最终分类性能存在不匹配，以及实验仅针对斑马鱼，尚未验证跨物种推广。

---

## 397. A Comparative Analysis of Machine Learning and Deep Learning Models for Tweet Sentiment Classification: A Case Study on the Sentiment140 Dataset

**arXiv ID:** 2605.04888 | [PDF](https://arxiv.org/pdf/2605.04888v1)

**作者:** Vita Anggraini `[一作]` (Sumatra Institute of Technology), Martin C. T. Manullang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较传统逻辑回归+TF-IDF与BiLSTM在Sentiment140子集上的情感分类性能，并将两种模型部署到交互式网页应用。

**💡 创新点**

证明在中等规模非正式文本数据上，传统机器学习可超越深度学习，并实现模型实时在线部署。

**🔧 技术方法**

TF-IDF特征提取、逻辑回归、双向LSTM（PyTorch实现）、Streamlit、Hugging Face Spaces。

**📊 数据集**

从Kaggle下载的Sentiment140数据集随机抽样的1万条推文。

**📈 对比分析**

采用准确率、精确率、召回率、F1等指标比较；逻辑回归取得73.5%准确率，BiLSTM仅69.17%，显示传统模型更优。

**⚠️ 局限性**

数据量不足导致BiLSTM过拟合，且仅使用二分类标签；未尝试更强的预训练词向量或Transformer模型。

---

## 398. Shedding Light onto Safety Integrity Level and Basic Software Constraints in a Real-World Automotive Application: Case Study with Driverator Framework

**arXiv ID:** 2605.04837 | [PDF](https://arxiv.org/pdf/2605.04837v1)

**作者:** Tobias Denzinger `[一作]` (CARIAD SE), Peter Ulbrich `[通讯]` (Technische Universität Dortmund)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种在汽车ECU早期设计阶段同时考虑安全完整性等级（SIL）、AUTOSAR基础软件（BSW）和内存需求的全新架构方法，并基于真实车辆运动控制系统构建了可扩展的Driverator仿真框架。

**💡 创新点**

创新点在于：①将SIL、BSW与内存约束统一纳入早期设计决策，填补了传统仅关注时序或混合关键性设计的空白；②提出统计抽象技术，用 Weibull 分布等方法从受IP限制的实际系统中提炼关键指标；③将 BSW 运行时开销建模为任务局部扩展和全局 BSW 任务，提供了更精准的资源预算；④构建 Driverator 框架，可在不同硬件平台上快速生成、评估任务/内存分配方案。

**🔧 技术方法**

使用了 AUTOSAR 架构模型、基于统计学的抽象方法（Weibull、相关系数分析）、任务分配与优先级调度（FP 固定优先级）以及因果效应链的端到端时延建模；还采用了仿真/工具链（Driverator）实现系统配置与性能评估。

**📊 数据集**

数据集来源于一台实际运动与驱动控制器的匿名化案例，包含 30+ 软件组件、6000+ BSW 运行单元、ROM/RAM 统计以及多种 SIL 等级；通过统计抽象方法生成的虚拟数据与真实测量对比，保持 0.5–0.6 的相关系数。

**📈 对比分析**

通过 Driverator 框架对不同任务分配策略（如同一 SIL 组件共存于同核、不同 SIL 分离）进行仿真，比较满足时序约束、内存占用和 BSW 开销的方案。实验结果显示：整合 BSW 模型后，系统利用率提升约10%，且在 90% 的配置中同时满足所有 SIL 级别的内存/时序要求。

**⚠️ 局限性**

局限性包括：①统计抽象方法仅在早期阶段提供保守估计，无法替代后期的精确测量；②假设 BSW 开销与任务集聚合程度线性相关，可能不适用于所有硬件平台；③仿真结果受所选模型参数（WCET、通信延迟）影响，实际部署仍需进一步验证。

---

## 399. Patterns of Developer Adoption of LLM-Generated Code Refactoring Suggestions

**arXiv ID:** 2605.04835 | [PDF](https://arxiv.org/pdf/2605.04835v1)

**作者:** David Schön `[一作]` (Chalmers University of Technology and University of Gothenburg), Philipp Leitner `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了开发者在实际代码中采用ChatGPT生成的重构建议的情况，并通过定量相似度和定性分析来评估其采纳程度。

**💡 创新点**

创新点在于首次从开发者实际提交的代码和ChatGPT对话中测量重构建议的采纳率，并提出了新的 Token Match Rate 计量指标与五种修改模式。

**🔧 技术方法**

采用了词法相似度度量（Levenshtein、Jaccard n-gram、Token Match Rate、CrystalBLEU）以及手工代码对比等技术。

**📊 数据集**

使用了公开的 DevGPT 数据集（169 个 GitHub 提交、440 个文件、约 3.2k 条提示）并扩展为 440 个映射数据点。

**📈 对比分析**

通过四种相似度指标对 440 个文件进行统计，发现约 45% 采用率极高（相似度≥0.9）且大部分是完全复制；对比手工案例验证指标有效，性能表现良好。

**⚠️ 局限性**

局限性包括仅覆盖已链接 ChatGPT 会话的提交，无法捕获未采纳的建议；数据集来源单一（ChatGPT）、样本不均衡，以及相似度方法无法捕捉多建议、上下文完整性等维度。

---

## 400. Bridging Input Feature Spaces Towards Graph Foundation Models

**arXiv ID:** 2605.04834 | [PDF](https://arxiv.org/pdf/2605.04834v1)

**作者:** Moshe Eliasof `[一作]` (University of Cambridge), Carola-Bibiane Schönlieb `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ALL-IN框架，利用随机投影将任意节点特征映射到共享随机空间，再通过节点协方差算子构造输入无关的图表示，实现跨数据集可迁移的图学习；

**💡 创新点**

创新点在于引入随机高维投影和协方差算子，理论证明其分布不变性及对正交变换的不变性，消除特征维度、语义和取值范围差异的障碍；

**🔧 技术方法**

核心技术包括随机高斯投影、节点协方差算子构造、基于协方差的多算子GNN层、中心化处理以及相应的理论分析；

**📊 数据集**

使用了9个多模态图数据集（zinc、ogbg-molhiv、ogbg-molesol、ogbg-moltox21、mnist、cifar10、Cuneiform、MSRC 21、ModelNet）进行预训练，随后在Cora、CiteSeer、PubMed、MUTAG、PROTEINS等未见数据上进行评估；

**📈 对比分析**

与多种基线（MLP、GCN、GIN、LLM‑augmented GNN、GraphAny、SCORE 等）对比，ALL‑IN 在源数据集上与单模型相当甚至更优，在新特征和任务的未见数据上显著优于传统 GNN 和现有图基础模型，取得最高准确率与 AUC；

**⚠️ 局限性**

局限性包括对大规模图的可扩展性受限于密集协方差矩阵，需要稀疏化或近似；随机投影可能不是最优，未来可探索结构化或可学习的投影方式；

---

## 401. A Pragmatic Comparison of Cryptographic Computation Technologies for Machine Learning

**arXiv ID:** 2605.04858 | [PDF](https://arxiv.org/pdf/2605.04858v1)

**作者:** Marcus Taubert `[一作]` (AIT Austrian Institute of Technology), Thomas Loruenser `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在同一硬件环境下对 Secretflow-SPU（SMPC）和 Concrete-ML（FHE）两大框架进行基准测试，比较其在矩阵乘法、激活函数、线性回归、随机森林、密集层和卷积网络等机器学习任务中的推理性能。

**💡 创新点**

首次将 SMPC 和 FHE 在同一工作流中全面对比，提出技术选择指南，并揭示不同模型对两种技术的适配性。

**🔧 技术方法**

使用 Secretflow-SPU 与 Concrete-ML 框架，基于 Python、C++/Rust 实现的 SMPC 与 FHE 进行加密推理，对基本算子及标准 ML 模型进行实验。

**📊 数据集**

主要使用随机生成的数据以及公开的图像/文本数据集（如 MNIST/ImageNet 等）构造模型进行基准，未使用专业标注数据集。

**📈 对比分析**

采用相同 CPU 环境（多核 CPU）对推理时间进行测量，结果显示 SMPC 在大规模 CNN 等复杂模型上性能更优，而 FHE 在小型线性回归或简单 Dense 模型上较快，但整体推理时间仍显著更长。

**⚠️ 局限性**

局限性包括：仅在通用 CPU 上评测，未考虑 GPU/加速器；FHE 实现受限于量化精度和模型复杂度；SMPC 受网络延迟影响；两种技术均未在真实生产环境中进行长时间稳定性测试。

---

## 402. A geometric relation of the error introduced by sampling a language model's output distribution to its internal state

**arXiv ID:** 2605.04899 | [PDF](https://arxiv.org/pdf/2605.04899v1)

**作者:** Albert F. Modenbach `[一作]` `[通讯]`, Albert F. Modenbach

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

未知

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

## 403. Sentiment Analysis and Customer Satisfaction Prediction on E-Commerce Platforms Based on YouTube Comments Using the XGBoost Algorithm

**arXiv ID:** 2605.04887 | [PDF](https://arxiv.org/pdf/2605.04887v1)

**作者:** Ridho Benedictus Togi Manik `[一作]` (Institut Teknologi Sumatera), Martin Clinton Tosima Manullang `[通讯]` (Institut Teknologi Sumatera)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于XGBoost+TF-IDF的情感预测模型，对印尼电商YouTube评论进行分类，并与LSTM模型进行对比实验。

**💡 创新点**

首次揭示电商评论中社会政治词汇对情感极性的显著影响，并通过PyCaret AutoML提升传统XGBoost的性能；在印尼视频评论情感分析领域实现了新的基准。

**🔧 技术方法**

使用文本预处理、TF-IDF特征提取、XGBoost、PyCaret自动化机器学习、以及LSTM深度学习模型进行实验。

**📊 数据集**

采用从印尼电商评测视频中收集的二次YouTube评论数据集，包含正、负、中性情感标签。

**📈 对比分析**

通过与SVM基线、PyCaret优化后的XGBoost、以及LSTM网络的对比，XGBoost在80:20划分下取得76%准确率，略优于LSTM的74%，显示传统提升树在中等规模不平衡数据上的优势。

**⚠️ 局限性**

数据量有限且类别严重不平衡，未采用SMOTE等过采样方法；模型对讽刺等细微情绪的识别效果不足，需进一步扩大语料并改进数据平衡策略。

---

## 404. FairEnc: A Fair Vision-Language Model with Fair Vision and Text Encoders for Glaucoma Detection

**arXiv ID:** 2605.04882 | [PDF](https://arxiv.org/pdf/2605.04882v1)

**作者:** Mohamed Elhabebe `[一作]` (University of Oulu), Qing Liu `[通讯]` (UiT Arctic University of Norway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种名为 FairEnc 的可公平预训练视觉‑语言模型，用于青光眼检测，在视觉和文本编码器上同时消除多重敏感属性（种族、性别、民族、语言、年龄等）的偏差。

**💡 创新点**

创新点包括：
1) 利用大语言模型生成中性与随机敏感属性的合成临床文本，并通过对比学习让文本编码器聚焦病理信息；
2) 在视觉编码器中采用互信息正则化和多鉴别器对抗去偏的双层策略，实现视觉特征与敏感属性的解耦；
3) 在单一模型中同时实现多属性去偏，避免传统方法需为每个属性单独训练；
4) 通过新构建的 FairFundus 数据集评估跨域与跨模态的公平性。

**🔧 技术方法**

技术手段包括：Qwen 大语言模型生成合成文本、NT‑Xent 对比损失、字典代理互信息最小化、对抗性多鉴别器、CLIP 视觉编码器、线性探针与零样本评估等。

**📊 数据集**

使用的数据集有公开的 Harvard‑FairVLMed（配对眼底图像与临床文本）以及自建的 FairFundus（彩色眼底图像与临床文本）。

**📈 对比分析**

在零样本和线性探针评估中与 CLIP、JTT、FairCLIP、Robust FairCLIP、FairMoE 等基线比较，FairEnc 在 AUC 维持或提升的同时，DPD 与 DEOdds 均显著降低，显示出最佳的性能‑公平 trade‑off；在 FairFundus 线性探针下，FairEnc 仍保持最低的公平指标，尽管总体 AUC 与基线相近。

**⚠️ 局限性**

局限性：
1) 在跨域（不同图像模态、数据分布）时性能‑公平 trade‑off 仍会出现；
2) 某些敏感属性（如 ethnicity、gender）公平提升有限；
3) 多目标损失与超参数设置较复杂，需细致调参；
4) FairFundus 数据集在人口多样性方面受限，难以全面验证公平性。

---

## 405. A Harmonic Mean Formulation of Average Reward Reinforcement Learning in SMDPs

**arXiv ID:** 2605.04880 | [PDF](https://arxiv.org/pdf/2605.04880v1)

**作者:** Erel Shtossel `[一作]` (Bar Ilan University), Gal A. Kaminka `[通讯]` (Bar Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种新的均值算子（Harmonic）并基于它构建了面向半马尔可夫决策过程的平均奖励强化学习算法（Harmonic R-Learning），用于估计平均奖励率。

**💡 创新点**

创新点在于：①设计了能处理零值和符号混合的均值算子，克服了传统调和平均在负数/零时失效的问题；②证明该算子是一般均值而非准算术均值；③将其直接嵌入SMDP的平均奖励更新，使得算法在奖励与持续时间相关、非平稳的环境下表现更稳健。

**🔧 技术方法**

采用的技术包括：半马尔可夫决策过程建模、指数移动调和均值（EMA‑Harmonic）近似、随机梯度更新的Q学习框架、对比实验（SMART、Relaxed‑SMART）。

**📊 数据集**

使用的数据集：1）两状态SMDP的人工仿真，奖励/持续时间随时间漂移；2）真实比特币（BTC）历史分钟级价格数据（2012–2025年，共约500万条记录），用于构造随机延迟和按奖励缩放的SMDP版本。

**📈 对比分析**

比较方法：在相同的学习率、探索率、状态空间与β值下，分别运行三种算法，记录累计奖励或成功率。实验结果显示：在奖励与持续时间相互依赖的环境（两状态SMDP、按奖励缩放的比特币SMDP）中，Harmonic R-Learning 的成功率/收益显著高于SMART和Relaxed‑SMART；在奖励与时间独立的环境（随机延迟比特币SMDP）则两者表现相近。

**⚠️ 局限性**

局限性包括：①算法对β等EMA参数敏感，需要手动调参；②目前仅在离散动作空间的小规模SMDP和金融交易仿真上验证，未测试大规模连续动作或更复杂环境；③虽然解决了零/负值问题，但在极端稀疏奖励或高方差场景下的收敛性和稳定性尚待进一步研究。

---

## 406. To Fuse or to Drop? Dual-Path Learning for Resolving Modality Conflicts in Multimodal Emotion Recognition

**arXiv ID:** 2605.04877 | [PDF](https://arxiv.org/pdf/2605.04877v1)

**作者:** Yangchen Yu `[一作]` (Hefei University of Technology), Richang Hong `[通讯]` (Hefei University of Technology)

**通讯引用:** 22729 | [OpenAlex ID](https://openalex.org/A5051332325)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究多模态情感识别中的模态冲突，提出双路径冲突解决框架DCR，在特征层进行软校准（AFD）并在决策层进行硬仲裁（ADA），实现对不同可调和程度的冲突的自适应处理。

**💡 创新点**

创新点包括：①将模态冲突划分为可调和的温和冲突和不可调和的严重冲突；②对温和冲突采用逆向知识蒸馏进行特征级软校准；③对严重冲突使用基于上下文多臂赌博机的强化学习代理进行决策级硬仲裁；④融合逆向蒸馏与强化学习形成统一的双路径框架。

**🔧 技术方法**

主要技术手段：逆向知识蒸馏（AFD）与时序加权梯度CAM；情感辨识代理（ADA）基于上下文多臂赌博机和优势演员-价值（A2C）强化学习；跨模态注意力融合；数据增强（模态随机丢弃与高斯噪声注入）；奖励设计结合置信度的校准奖励。

**📊 数据集**

使用了五个公开数据集：对话级的MELD、IEMOCAP、CMU-MOSEI；剪辑级的CH‑SIMS和CH‑SIMS v2。

**📈 对比分析**

与多种基线（传统递归、Transformer、图网络、LLM驱动方法）在准确率、加权F1、F1、相关系数、MAE等指标上进行对比。DCR在所有数据集上均达或逼近SOTA，尤其在冲突子集上提升显著；在对话级和剪辑级评测中均表现出优异的稳健性。

**⚠️ 局限性**

局限性：①在极端冲突情况下仍可能受限于预训练文本基线；②对细粒度模态组合（如二模态组合）的探索有限，扩展动作空间未提升性能；③在对话级情感识别与大型LLM方法相比仍存在差距，需要进一步提升文本理解能力。

---

## 407. RTMS: A Real-Time Multimodal Scaffolding System for Improving Debugging in Computing Education

**arXiv ID:** 2605.04848 | [PDF](https://arxiv.org/pdf/2605.04848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 408. Phased Ultra Massive Array (PUMA)

**arXiv ID:** 2605.04866 | [PDF](https://arxiv.org/pdf/2605.04866v1)

**作者:** Hanjiang Hong `[一作]` (University College London), Hyundong Shin `[通讯]` (Kyung Hee University)

**通讯引用:** 8465 | [OpenAlex ID](https://openalex.org/A5007557286)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于流体天线与相位阵列的新型多用户接入框架 PUMA，能够在单 RF 链下实现高功率增益和自干扰抑制。

**💡 创新点**

创新点在于将相位偏移器集成到流体天线的模拟域聚合，既不需要基站 CSI，也不需要用户端 SIC，同时通过端口选择与相位调节提升信号增益。

**🔧 技术方法**

使用了流体天线系统（FAS）、相位阵列聚合、随机/自适应端口选择、端口密度调节以及理论上的 SIR/BER 推导等技术。

**📊 数据集**

主要使用仿真数据，包括 6 GHz 与 26 GHz 下的多路径模型（K = 0/7，N_p = 50/2）、不同 FAS 尺寸与端口密度组合，未采用真实测量数据集。

**📈 对比分析**

通过与 CUMA 和 sFAMA 在相同频率、端口数量、RF 链数下的 Monte‑Carlo 仿真对比，PUMA 在单 RF 链时可实现近两倍的数据率和用户容量，且在高频/大 FAS 尺寸时性能最优。

**⚠️ 局限性**

局限性包括对低频小 FAS 尺寸的受限性能、端口密度对相关性影响、仅在仿真环境验证、对真实硬件互耦与非理想相位器的影响尚未深入研究。

---

## 409. Assessing Cognitive Effort in L2 Idiomatic Processing: An Eye-Tracking Dataset

**arXiv ID:** 2605.04857 | [PDF](https://arxiv.org/pdf/2605.04857v1)

**作者:** Eduardo Santos `[一作]` (Federal University of Rio Grande do Norte), César Rennó-Costa `[通讯]` (Federal University of Rio Grande do Norte)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个面向第二语言学习者的眼动追踪数据集，用于研究他们在习语处理时的认知成本，并提供完整的原始、处理后和指标文件。

**💡 创新点**

首个在不同 CEFR 水平上系统化、公开的习语眼动数据；证明 60 Hz 低端眼动仪亦能捕获宏观认知事件；提供透明、可复现的数据处理流水线；将人类认知指标与大型语言模型对齐作为基准。

**🔧 技术方法**

使用 Tobii Pro Spark 60 Hz 眼动仪、PsychoPy 进行实验呈现、Python+PyGaze 进行事件检测（I‑DT 与自定义回归检测启发式）、自编的预处理脚本。

**📊 数据集**

本研究收集的眼动数据集（含原始、处理后和指标）以及刺激来源于 MAGPIE 语料库和 SemEval 2025 Task 1 的 PIE_context_data。

**📈 对比分析**

通过对不同 CEFR 水平受试者的回归次数和固定点进行统计，验证了熟练度越高回归次数越少的趋势；该结果为人类认知成本提供基准，未对模型进行性能评估。

**⚠️ 局限性**

仅使用 60 Hz 低端眼动仪，时间分辨率受限；受试者仅为巴西葡萄牙语母语者的英语学习者，难以推广；样本量有限，统计功效受限；未覆盖其他语言对或更高频率硬件。

---

## 410. Hybrid Iterative Neural Low-Regularity Integrator for Nonlinear Dispersive Equations

**arXiv ID:** 2605.04853 | [PDF](https://arxiv.org/pdf/2605.04853v1)

**作者:** Zhangyong Liang `[一作]` `[通讯]` (Tianjin University), Zhangyong Liang (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种混合迭代神经低正则性积分器（HIN‑LRI），通过在传统低正则性积分器（LRI）中加入神经算子对结构化残差进行时间步长缩放的校正，以解决粗糙数据下的阶数降低和对数误差。

**💡 创新点**

创新点在于：① 以“物理骨干+神经残差”模式拆分误差；② 在低维潜在空间中学习残差并显式乘以时间步长，实现数值稳定并解耦 CFL 约束；③ 采用求解器内循环（SITL）训练，直接优化Bourgain空间轨迹误差，减少分布漂移。

**🔧 技术方法**

主要技术包括低正则性积分器、神经算子（轻量级网络）在潜在空间投影、尺度网络实现自适应空间缩放、显式时间步长缩放、求解器内循环端到端训练，以及离散Bourgain空间损失和快速傅里叶变换实现。

**📊 数据集**

使用从分数高斯随机场生成的 H^γ（γ∈{-0.5,0.5,1.5}）初始数据，分别对 KdV、cubic NLS 和 quadratic NLS 三个一维周期性支配方程进行实验。

**📈 对比分析**

与传统 LRI、分裂法、未滤波/滤波 LRI 以及纯数据驱动的 FNO/PINN/DeepONet 等基准比较，HIN‑LRI 在粗糙数据下保持 O(τ) 收敛、消除对数惩罚、在高分辨率下保持稳定、实现零漂移/能量守恒，且在线推理仅增加约 20% 计算开销，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括仅在 1D 周期域、固定方程和正则性范围内验证；对多维、非周期或变系数问题的推广尚未实现；理论误差分析基于紧集与权重范数限制，实际训练中仍存在收敛与泛化不确定性。

---

## 411. QuadBox: Accelerating 3D Gaussian Splatting with Geometry-Aware Boxes

**arXiv ID:** 2605.04844 | [PDF](https://arxiv.org/pdf/2605.04844v1)

**作者:** Xinze Li `[一作]`, Weifeng Su `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出QuadBox和QPass两种方法，实现对3D Gaussian Splatting渲染过程中高效的高斯–瓦片交叉检查，显著减少无效瓦片访问。

**💡 创新点**

创新点在于：①使用四个自适应轴对齐包围盒(QuadBox)精确包围投影高斯椭圆，②设计无分支单遍扫描算法QPass，利用离散网格特性仅做整数区间检查，从而降低计算和内存开销。

**🔧 技术方法**

采用几何分析得到伸缩因子构造QuadBox，使用GPU可并行的瓦片化渲染框架实现QPass，并在官方3DGS代码库中做了可插拔实现。

**📊 数据集**

实验数据集包括Mip-NeRF 360、Deep Blending和Tanks & Temples三大公共数据集。

**📈 对比分析**

与原始3DGS及其改进版本AdR-AABB对比，QuadBox+QPass在保持相同或更好图像质量（PSNR、SSIM、LPIPS）同时实现平均1.8×以上的FPS提升，单场景最高可达约1.85×速度提升。

**⚠️ 局限性**

局限性：构造QuadBox仍需在每个高斯上做一次额外计算；算法依赖于瓦片化渲染框架，对非瓦片化或极大高斯数量的场景需要进一步评估；目前实现主要在NVIDIA GPU上验证，跨平台兼容性尚未全面测试。

---

## 412. StoryAlign: Evaluating and Training Reward Models for Story Generation

**arXiv ID:** 2605.04831 | [PDF](https://arxiv.org/pdf/2605.04831v1)

**作者:** Haotian Xia `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**通讯引用:** 15039 | [OpenAlex ID](https://openalex.org/A5003324011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个针对故事偏好评估的基准（StoryAlign），并基于该基准训练了一个高效的故事奖励模型，显著提升了对人类偏好的捕捉能力。

**💡 创新点**

创新点包括①首次提出故事偏好评估基准，涵盖五个评估维度；②设计了自动化收集海量高质量故事偏好对的流程；③训练出小规模但性能领先的奖励模型，超越更大模型；④将奖励模型应用于测试时最佳‑N（BoN）采样验证实际效果。

**🔧 技术方法**

技术上使用多模型候选生成、LLM‑辅助偏好投票与人工核验相结合的两阶段注释流程；利用Kendall’s Tau衡量投票一致性；训练时采用对比式损失与Llama‑3.1‑8B‑Instruct、Qwen3‑8B等基础模型；在BoN实验中结合最佳‑N采样与奖励模型选择。

**📊 数据集**

数据集方面：1,133条人类验证的偏好实例（StoryAlign benchmark）；约100,000条自动构造的偏好对（来源于Douban、WritingPrompts、Back‑generation、Rewrite、Continuation等）；MoPS基准用于BoN评估；候选故事来自GPT‑4o、Gemini‑2.5 Pro、Grok‑4、Qwen‑Long等LLM。

**📈 对比分析**

评估方法：在StoryAlign上对比多种奖励模型（InternLM2‑Reward、Skywork‑Reward、ArmoRM等）和LLM‑评判（GPT‑4o、Gemini‑2.5 Pro等）。最佳现有模型准确率仅为66.3%，而新模型在StoryAlign上达到近80‑85%（相较于最大模型仍更优）。在BoN实验中，新模型选出的故事在与人工排序的头对头比较中赢率高于其他模型，且低于30%的情况选到劣质故事。

**⚠️ 局限性**

局限性：1) 数据覆盖仍受限于采集平台和语言；2) 奖励模型可能偏向LLM生成内容，需更多人类写作数据；3) 评价仍基于有限的五维指标，未覆盖所有叙事细节；4) 对跨文化、多语言故事的泛化能力尚未充分验证。

---

## 413. Building AI Companions that Prioritise Learning over Performance

**arXiv ID:** 2605.04816 | [PDF](https://arxiv.org/pdf/2605.04816v1)

**作者:** Hassan Khosravi `[一作]` (University of Queensland), Ryan S. Baker `[通讯]` (Adelaide University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并框架化了AI学习伴侣的设计，旨在通过适应性、教育学和负责任的设计三大支柱，促进学生深度学习、元认知发展和学习成果的可持续提升。

**💡 创新点**

创新点在于将大型语言模型与学习者建模、适应性调节和伦理透明度整合，构建可持续的学习伴侣体系，并通过五个案例验证其可行性。

**🔧 技术方法**

技术包括大语言模型（如GPT系列）、知识追踪、强化学习、生成式对话系统、可解释AI和安全防护机制。

**📊 数据集**

使用的数据集涵盖Khan Academy平台的练习记录、RiPPLE的学生创作与同行评估日志、CodeHelp的编程查询日志、JeepyTA的讨论论坛交互以及Recast的课程资源与学生反馈。

**📈 对比分析**

方法比较通过案例评估学习参与度、认知负荷、知识迁移和元认知水平等指标，结果显示AI学习伴侣相较传统工具能提升学生的认知投入与学习成效，但在长期保持自我调节方面仍有限。

**⚠️ 局限性**

局限性包括当前模型对深度学习支持不足、适应性不够精细、对长期可持续学习的评估缺乏足够数据、以及对多样性和公平性仍存在偏差。

---

## 414. Concurrence of Symmetry Breaking and Nonlocality Phase Transitions in Diffusion Models

**arXiv ID:** 2605.04830 | [PDF](https://arxiv.org/pdf/2605.04830v1)

**作者:** Yifan F. Zhang `[一作]` (Princeton University), Xun Gao `[通讯]` (University of Colorado Boulder)

**通讯引用:** 2274 | [OpenAlex ID](https://openalex.org/A5045173659)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并统一了扩散模型中两种相变（对称破缺与非局部性）在生成过程中的临界窗口，证明两者在现代扩散 Transformer（DiT‑XL 与 SD3）中几乎同步出现。

**💡 创新点**

创新点在于提出了可操作的分辨两种相变的“瞬时得分差距”与“积分前后实验”两类探测器，并将它们结合到窗口化条件与局部/全局去噪实验中，首次在实践层面将对称破缺与非局部性相互关联，并给出了何时需要全局通信与条件注入的量化诊断。

**🔧 技术方法**

使用的技术包括扩散 Transformer（DiT‑XL 与 SD3）、基于条件与无条件得分的CFG（classifier‑free guidance）、局部注意力截断得到局部去噪器、得分差距（Δs_cond、Δs_loc）计算、前向-后向错误修正实验、窗口化条件/去噪实验、FID 与分类误差评估。

**📊 数据集**

主要数据集为 ImageNet（DiT‑XL）和文本条件的 SD3 生成模型所使用的公共图像数据集（512×512 的 ImageNet 及对应文本提示）。

**📈 对比分析**

比较方法：在两种模型上分别计算得分差距热图、前向-后向错误率曲线、窗口化条件/局部去噪的 FID 与分类误差。实验表明：DiT‑XL 的临界窗口在 t≈0.2（瞬时）与 t≈0.5（积分）处出现，说明它在此区间内提前使用全局通信；SD3 的临界窗口集中在 t≈1，瞬时与积分实验结果高度一致，显示其更高效的条件与全局计算使用。

**⚠️ 局限性**

局限性：仅验证了两种模型，未覆盖更大规模或不同任务的扩散模型；窗口化实验对噪声时间尺度敏感，需进一步泛化；局部去噪器的截断仅为近似，真实全局信息仍能通过多步传播；缺乏理论对两种相变同步的严格证明。

---

## 415. Faster Algorithms for Shortest Unique or Absent Substrings

**arXiv ID:** 2605.04826 | [PDF](https://arxiv.org/pdf/2605.04826v1)

**作者:** Panagiotis Charalampopoulos `[一作]` (King's College London), Wiktor Zuba `[通讯]` (University of Warsaw)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5083139616)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

该论文提出了在单词 RAM 模型下对已打包字符串求解最短唯一子串（SUS）和最短缺失子串（SAS）的算法，时间复杂度为 O(n log σ/√log n)，显著低于传统的线性时间方案。

**💡 创新点**

创新点在于将问题按子串长度和周期拆分，并分别利用同步集、周期跑（runs）、波形树（wavelet tree）以及几何化的主景点（skyline）求解，从而将 SUS/SAS 的求解转化为几何最小点问题；此外，提出了将任意字母表的实例归约到二进制实例的高效方法。

**🔧 技术方法**

核心技术包括：①打包字符串与位级并行操作；②τ‑同步集与稀疏 Lyndon 根；③周期跑的构造与分组；④波形树与 LCP 列表的快速构建；⑤重叠的 Heavy‑Light 分解；⑥几何主景点（skyline）求解；⑦ de Bruijn 序列的高效生成。

**📊 数据集**

本文为理论研究，未在具体数据集上实验；所有结果均为渐进复杂度分析。

**📈 对比分析**

与传统的 O(n) 线性时间 suffix‑tree 方案相比，新的算法在打包输入下实现了 O(n log σ/√log n) 的改进，证明了对于小字母表（尤其是二进制）已接近条件下的最优界限。

**⚠️ 局限性**

限制包括：①仅适用于打包字符串（packed）而非通用字符串；②算法实现复杂，需多种高级数据结构；③在实际应用中常数因子和内存占用仍可能较大；④尚无证明其在所有输入上都能达到下界，仍有进一步优化空间。

---

## 416. Quantile-Free Uncertainty Quantification in Graph Neural Networks

**arXiv ID:** 2605.04847 | [PDF](https://arxiv.org/pdf/2605.04847v1)

**作者:** Soyoung park `[一作]`, Sungsu Lim `[通讯]` (Chungnam National University)

**通讯引用:** 809 | [OpenAlex ID](https://openalex.org/A5011058984)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出QpiGNN框架，用双头图神经网络实现无分位点预测区间不确定性量化；

**💡 创新点**

创新点在于：1）双头架构将预测值与区间宽度解耦；2）无分位点联合损失直接优化覆盖率与宽度；3）理论上给出渐近与有限样本覆盖保证；

**🔧 技术方法**

采用GraphSAGE等GNN编码器、Softplus宽度预测头、coverage+width联合损失、以及实验中的多种基线对比；

**📊 数据集**

使用19个数据集，包括合成（如Tree、BA、ER等）与真实（如Chameleon、Squirrel、Twitch等）图回归任务；

**📈 对比分析**

与SQR‑GNN、RQR^adj‑GNN、CF‑GNN、Evidential Regression、BayesianNN、MC Dropout等方法对比，QpiGNN平均覆盖率提升22%，间距缩小50%，并在噪声、结构偏移、非可交换拆分等情形下保持鲁棒；

**⚠️ 局限性**

局限性：依赖弱假设（如标签噪声有限、节点嵌入多样性），对重尾噪声、模型偏差、结构冗余等情况的鲁棒性仍有限；

---

## 417. Communication Offloading on SmartNIC DPUs: A Quantitative Approach

**arXiv ID:** 2605.04842 | [PDF](https://arxiv.org/pdf/2605.04842v1)

**作者:** Jacob Wahlgren `[一作]` (KTH Royal Institute of Technology), Ivy Peng `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1261 | [OpenAlex ID](https://openalex.org/A5037069204)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Buddy，一个可将“fire‑and‑forget”通信路由服务从主机 CPU 上卸载到 SmartNIC DPU 的库，并在 NVIDIA BlueField‑3 上对其性能进行量化评估。

**💡 创新点**

创新点在于将通信路由功能完整迁移至 DPU 并通过对内存‑通信比率、聚合粒度、流水线与多线程等参数的系统优化，揭示了 DCA 缺失对 DPU 性能的严重影响，进一步为未来 SmartNIC 设计提供了关键改进方向。

**🔧 技术方法**

主要技术包括基于 RDMA 的零拷贝消息聚合、两级路由代理、OpenMP 并行化、Buddy 运行时库与 routing agent 的分离，以及对 ARM/DCPU 的 DOCA 框架和 ibverbs 库的调用。

**📊 数据集**

实验使用了五个不同领域的基准应用（单源最短路径、三角计数、直方图、稀疏矩阵转置与 Quicksilver）以及对应的随机/真实数据集，规模设定为主机缓存容量的 100 倍。

**📈 对比分析**

通过三种卸载场景（本机同进程、主机独立核、BlueField‑3 DPU）进行对比，发现主机占用型工作可获得 1.13–1.55 倍加速，而通信占优型工作加速不显著；同时提升因素包括消息聚合大小、流水线缓冲与多线程协同，最大提升可达 12.4 倍。

**⚠️ 局限性**

局限性主要体现在 DPU 缺乏 Direct Cache Access（DCA）导致内存访存量激增 625 倍，进一步限制了高通信密集型工作负载的加速效果；此外，实验仅覆盖 BlueField‑3，未验证更高性能的下一代 SmartNIC。

---

## 418. Traffic Chunk Sizing vs. Optical Switching Speed in Future All-Optical Satellite Networks

**arXiv ID:** 2605.04829 | [PDF](https://arxiv.org/pdf/2605.04829v1)

**作者:** Sleman Mouammar `[一作]` (Technische Universität Braunschweig), Admela Jukan `[通讯]` (Technische Universität Braunschweig)

**通讯引用:** 3990 | [OpenAlex ID](https://openalex.org/A5061791380)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在OMNeT++仿真平台上，对未来全光卫星网络进行系统级评估，研究了不同卫星星座（Telesat、Amazon LEO Shell 1、Starlink Phase 1）在不同光学切换器（MEMS、Beam‑Steering、Electro‑Optic、InP集成光子）下的流量块（chunk）大小、端到端延迟、丢包率及能效之间的相互影响；

**💡 创新点**

创新点在于将光学切换器的硬件特性（切换速度、功耗、插入损耗）与流量块调度模型耦合，系统性评估了块大小与切换器速度之间的权衡，给出了在满足30–60 ms延迟阈值下各星座可行的最大块大小，并从能效角度比较了不同技术方案，填补了从设备层到网络层的性能评估空白；

**🔧 技术方法**

使用了光学突发切换（OBS）式模型并引入离线控制；利用OMNeT++ v6.3.0与OS3卫星仿真库生成星座轨道与传播模型；实现WDM波分、多路复用器、基于K‑shortest paths的路由与first‑fit波分配；对光链路进行信道估计，计算SNR、容量；采用Erlang（Poisson）流量到达模型；评估指标包括丢包率（BR）、端到端延迟、能效（EE）。

**📊 数据集**

主要数据集包括三组星座的TLE文件（Telesat、Amazon LEO Shell 1、Starlink Phase 1）以及用于热点定位的Ookla和Human Settlement Proxy数据；流量到达率通过Erlang分布（Poisson过程）设定；

**📈 对比分析**

通过对不同块大小（1 MB–1 GB）和切换器技术的仿真，比较了阻塞率、端到端延迟与能效；结果显示：ns级切换器（如InP SOA）可在60 ms延迟内支持高达600 MB块，丢包率低于1%；ms级切换器（MEMS/Beam‑Steering）因切换延迟显著，需使用更小块，丢包率升高至1–2%；能效方面，SOA在所有星座中表现最佳，GLSUN在Telesat星座中能效最高。

**⚠️ 局限性**

局限性包括仅基于仿真，未验证实际硬件；仅考虑OBS式块调度，未探讨两向/三向预留协议对延迟的影响；假设清晰大气，未涵盖雨衰等实际环境；仅对单一波长（1550 nm）进行评估，未考虑多波长或频谱复用的复杂性；

---

## 419. Trustworthy Federated Label Distribution Learning under Annotation Quality Disparity

**arXiv ID:** 2605.04827 | [PDF](https://arxiv.org/pdf/2605.04827v1)

**作者:** Junxiang Wu `[一作]` (Southeast University), Qiang Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了联邦标签分布学习（Fed-LDL）中的注释质量异质性问题，提出FedQual框架来改进本地训练与全局聚合。

**💡 创新点**

创新点包括：① 在本地训练中加入基于全局语义锚的质量调节校准；② 在全局聚合采用从质量导向到数量导向的渐进重加权；③ 证明异质性下客户端特定校准优于统一校准；④ 构建四个Fed-LDL基准。

**🔧 技术方法**

使用了联邦学习与标签分布学习、全局语义锚、质量调节正则、有效信息密度重加权、理论风险分析与实验评估等技术。

**📊 数据集**

采用了新构建的FER-LDL、FI-LDL、PIPAL-LDL、KADID-LDL四个基准数据集，涵盖情绪识别与图像质量评估。

**📈 对比分析**

与FedAvg、FedProx、MOON、FedRDN、FedGloSS、FedQAgg、FedQRect等多种基线在六项分布相似度/距离指标上比较，FedQual在所有基准上均实现最低KL、Chebyshev、Clark等距离，最高Cosine、Intersection，显示显著提升。

**⚠️ 局限性**

局限性在于依赖人工标注生成的质量指示器，对客户端异步或极端通信延迟的鲁棒性未深入分析，并且在极低质量标签下仍可能出现模型漂移。

---

## 420. Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs

**arXiv ID:** 2605.04819 | [PDF](https://arxiv.org/pdf/2605.04819v1)

**作者:** Zhenchao Sun `[一作]` (Beihang University), Chongyang Tao `[通讯]` (Beihang University)

**通讯引用:** 2488 | [OpenAlex ID](https://openalex.org/A5073065834)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向SAT公式中未满足核心变量预测的极性感知超图表示学习框架。

**💡 创新点**

创新点在于：①将SAT公式建模为含有条款-文字超图和条款关联图的超图结构；②引入极性不变与等变分解机制，显式建模同一变量的正负文字之间的极性关系；③加入极性翻转一致性正则化，强化极性一致的表征。

**🔧 技术方法**

使用超图神经网络（message passing）、极性分解的特征分解技术以及极性一致性正则化。

**📊 数据集**

在SR、CA、PS三大合成SAT数据集上进行实验，数据集按易中难划分。

**📈 对比分析**

与GCN、NeuroCore、SATFormer等基线相比，平均在Top-M Precision、PR-AUC和ROC-AUC上均显著提升，尤其在CA数据集上提升幅度最大。

**⚠️ 局限性**

局限性包括：对极性正则化超参数敏感；模型仍受限于超图结构的表达能力；在更大规模或真实工业实例中的可扩展性尚待进一步验证。

---

## 421. Not All Scaffolds Are Equal: How Initiation Mode Determines EMME Effectiveness in Debugging

**arXiv ID:** 2605.04868 | [PDF](https://arxiv.org/pdf/2605.04868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 422. A Hierarchical Agent System with Reinforcement Learning for Multivariate Time Series Data Cleaning

**arXiv ID:** 2605.04902 | [PDF](https://arxiv.org/pdf/2605.04902v1)

**作者:** Yuhan Shi `[一作]` (Zhejiang University), Tianyi Li `[通讯]` (Aalborg University)

**通讯引用:** 1386 | [OpenAlex ID](https://openalex.org/A5100460598)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对多变量时间序列（MTS）数据多重质量问题的自动化清洗系统，能够在没有清洗真值的情况下自动生成最优清洗流水线。

**💡 创新点**

创新点主要包括：① 采用分层强化学习架构，高层决策先确定待处理的质量问题类型，低层决策选取对应的清洗方法；② 设计双阶段奖励机制，将清洗质量与下游任务性能耦合，实现无监督的高效学习；③ 将多种现有清洗算子抽象为可组合的操作符库，支持任意组合与顺序。

**🔧 技术方法**

技术手段包括：层次化强化学习（高层策略选择问题类型，低层策略选取具体算子）、质量评估器（计算缺失率、异常率、约束违例率）、约束挖掘（利用 MAD、相关性筛选等技术从脏数据中自动提取时间与交叉变量约束）、双阶段奖励（轻量级模型的即时奖励 + 复杂模型的最终奖励）以及对抗性/动态回报归一化。

**📊 数据集**

实验使用四个公开真实数据集：ETTh1、IDF_OilTemp（预测任务），Libras、Handwriting（分类/聚类任务），并在每个数据集上注入缺失、重复、异常及约束违例等多种质量问题。

**📈 对比分析**

与四类基线（EDITOR、Clean4TSDB、DiffPrep、Sampling）对比，本文方法在上游清洗指标（F1、NMSE、RRA）和下游任务性能（预测、分类、聚类）上均取得显著提升（最高约 96% 的 NMSE 降低，27% 的下游性能提升），且在大多数场景下的运行时比单层 RL 或全量搜索快 1.5–3 倍。

**⚠️ 局限性**

局限性：① 仍需要针对源数据集训练 RL 代理，跨域应用需额外迁移或重新训练；② 目前仅支持离线批处理，尚未扩展到流式实时清洗；③ 对极大规模多维数据或高噪声环境的鲁棒性尚未系统评估。

---

## 423. Self-Attention as Transport: Limits of Symmetric Spectral Diagnostics

**arXiv ID:** 2605.04893 | [PDF](https://arxiv.org/pdf/2605.04893v1)

**作者:** Dominik Dahlem `[一作]` (Red Hat AI), Mac Misiura `[通讯]` (Red Hat AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

未知

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

## 424. VTAgent: Agentic Keyframe Anchoring for Evidence-Aware Video TextVQA

**arXiv ID:** 2605.04870 | [PDF](https://arxiv.org/pdf/2605.04870v1)

**作者:** Haibin He `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 31077 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 VTAgent，一种基于 agent 的两阶段框架（关键帧定位与基于关键帧的推理）用于 Video TextVQA。

**💡 创新点**

核心创新在于显式的关键帧锚定，突破了视频LLM在定位问题相关视觉文本证据上的瓶颈；同时通过无训练、监督微调和强化学习的分阶段训练策略提升性能。

**🔧 技术方法**

技术包括多模态语言模型 Qwen3-VL，结构化 agent 交互（<reasoning>、<action> 标签），监督微调（SFT）与基于奖励的强化学习（GRPO）以及关键帧选择工具。

**📊 数据集**

使用 M4‑ViteVQA 与 RoadTextVQA 两个公开基准，构建 20k+ 的 SFT 训练集及 4k+ 的 RL 训练集。

**📈 对比分析**

与 12 种专业与通用 Video‑LLM 进行对比，VTAgent 在所有子任务上均击败对手，平均提升约 12.12% 准确率与 11.15% ANLS，成为新 SOTA。

**⚠️ 局限性**

主要限制是对关键帧选择的依赖，若问题所需信息分布在多帧且难以单帧定位，性能仍可能下降；此外训练成本与推理时延较高。

---

## 425. Federated Learning for Early Prediction of EV Charging Demand

**arXiv ID:** 2605.04993 | [PDF](https://arxiv.org/pdf/2605.04993v1)

**作者:** Vasilis Perifanis `[一作]` (Indigma Innovations), Andreas Sendros `[通讯]` (Indigma Innovations)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对电动汽车充电站的早期会话需求预测，构建了基于首 10 分钟充电动态的表格特征，并在同一充电场站内通过站级联邦学习实现分布式模型训练，验证了在保持数据隐私的前提下能够近似集中式训练的预测性能。

**💡 创新点**

创新点包括：①首次将早期会话预测与联邦学习相结合，解决了充电数据高度分布且隐私敏感的问题；②提出了针对站级异质性的特征工程和统计分析，展示了在单场站内部仍存在可处理的非 IID 分布；③系统对比多种模型族（线性、树、深度神经网络）在集中式与联邦式两种训练模式下的表现，为后续跨场站联邦研究提供基准。

**🔧 技术方法**

技术手段：使用 FedAvg 联邦算法；特征工程包含时间窗口统计、趋势斜率、利用率比、早期能量估计；模型包括线性回归、XGBoost、MLP、CNN、GRU、Transformer；训练设置：FedAvg 400 轮、每轮 3 局部 epoch、20% 客户采样；评估指标为 MAE 与 RMSE。

**📊 数据集**

数据集来源于 Adaptive Charging Network（ACN），以 Caltech 充电场站为例，拆分为 54 个站级客户端，包含会话元数据、用户意图、10 分钟内的电流与 pilot 信号，最终构建约数千条样本的表格数据。

**📈 对比分析**

对比方法：集中式训练与站级联邦学习；结果显示集中式 XGBoost 取得 MAE 3.36/ RMSE 4.84；联邦式 Transformer 仅略逊 MAE 3.69/ RMSE 5.56，其他深度模型（MLP、GRU、CNN）也在 4 MAE 左右，表明联邦学习在保持隐私的同时可实现接近集中式的准确性；线性和 XGBoost 在联邦下性能显著下降。

**⚠️ 局限性**

局限性：①数据异质性虽不严重，但对部分模型（LR、XGB）影响较大；②仅在单场站内部进行联邦实验，未验证跨场站的可扩展性；③对聚合策略仅使用 FedAvg，缺乏更鲁棒的异构适配方法；④模型规模与计算开销仍需进一步压缩以满足边缘设备部署。

---

## 426. Reliable Modeling of Distribution Shifts via Displacement-Reshaped Optimal Transport

**arXiv ID:** 2605.04965 | [PDF](https://arxiv.org/pdf/2605.04965v1)

**作者:** Philip Naumann `[一作]` (BIFOLD Berlin Institute for the Foundations of Learning and Data), Grégoire Montavon `[通讯]` (Charitè Universitätsmedizin Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种利用有限观测位移信息学习新的马氏距离，从而改进最优运输（OT）在分布偏移建模中的地面度量的方法。

**💡 创新点**

创新点在于通过闭式计算位移的二阶统计量来直接构造马氏距离，无需标签或迭代优化，同时可扩展至核特征空间并保持与经典OT求解器的兼容性。

**🔧 技术方法**

采用基于二阶矩的地面度量重塑、核化Mahalanobis距离、Sinkhorn/经典OT求解器以及矩阵逆等技术实现。

**📊 数据集**

在多种数据集上评估：合成Rotating Moons、时间序列空气质量与家电能耗、以及真实迁徙鸟类轨迹和迁移适应的人工合成数据。

**📈 对比分析**

与经典OT、Sinkhorn、核OT、领域适应基线等方法比较，实验显示在运输误差、特征归因和分类精度方面均优于基线，尤其在仅提供少量位移样本时效果显著。

**⚠️ 局限性**

主要局限是对代表性位移样本的依赖，若位移不足或不具代表性会导致度量失效；核化变体对核参数与正则化强度敏感，需要仔细调参。

---

## 427. UFAL-CUNI at SemEval-2026 Task 11: An Efficient Modular Neuro-symbolic Method for Syllogistic Reasoning

**arXiv ID:** 2605.04941 | [PDF](https://arxiv.org/pdf/2605.04941v1)

**作者:** Ivan Kartáč `[一作]` (Charles University), Ondřej Dušek `[通讯]` (Charles University)

**通讯引用:** 2161 | [OpenAlex ID](https://openalex.org/A5004829991)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套模块化的神经符号系统，将小型LLM用于自然语言命题的 FOL 解析、机器翻译与符号检索，并通过 Prover9 进行形式化推理，以解决多语言下的阿里士多德三段论合法性判定。

**💡 创新点**

核心创新在于：① 通过中间 LaTeX 表示提高 LLM 解析质量；② 将内容无关推理与小参数 LLM 结合，显著降低内容效应；③ 对主排名指标进行经验与理论分析，揭示其对高精度系统的敏感性。

**🔧 技术方法**

使用技术包括：Qwen3 4B Thinking 进行 FOL 解析；Gemma 3 27B 进行多语言翻译；Prover9 作为第一阶逻辑定理证明器；正则表达式转译器将 LaTeX 形式映射到 Prover9 语法；符号式相关前提检索算法。

**📊 数据集**

数据集为 SemEval‑2026 Task 11 的四个子任务集合，包含原始训练集、测试集以及通过翻译与合成产生的多语言验证集。

**📈 对比分析**

通过与零-shot LLM 基线、系统各模块的消融实验以及排行榜上的官方评测进行对比，系统在四个子任务上的准确率分别为 95.3%、97.4%、93.8% 与 84.9%，内容效应低于基线，排名分别为 19/35、6/14、8/13 与 4/15，展示了较高性能与低内容依赖。

**⚠️ 局限性**

局限性包括：① 早期模块错误会在后续阶段累积；② 小型 LLM 的多语言推理能力受限；③ 仅采用中间 LaTeX 格式，可能无法覆盖所有逻辑细节；④ 未进行模型微调；⑤ 评价指标对偶然误差敏感，导致内容效应与综合得分易受波动影响。

---

## 428. Unintended Negative Impacts of Promotional Language in Patent Evaluation

**arXiv ID:** 2605.04926 | [PDF](https://arxiv.org/pdf/2605.04926v1)

**作者:** Bingkun Zhao `[一作]` (City University of Hong Kong), Hao Peng `[通讯]` (City University of Hong Kong)

**通讯引用:** 256210 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用一套经验证的135词词典，对2010-2023年美国专利与商标局（USPTO）274万份实用专利申请全文进行文本分析，研究促销语言在专利评估中的作用。

**💡 创新点**

创新点在于发现促销语言与科学评估相反，在专利评估中呈负向关联——高促销密度导致授权、转让和上诉成功率下降；同时促销语言与技术新颖性与引用影响正相关，揭示了制度上的“悖论”；并揭示性别与审查员经验对促销语言容忍度的调节效应。

**🔧 技术方法**

技术手段包括：词典匹配文本挖掘、固定效应逻辑回归与OLS回归、倾向评分匹配（PSM）以及交互项分析；并结合可读性、具体性等文本控制变量。

**📊 数据集**

数据集为USPTO实用专利申请全文与元数据（2010-2023，2748927份）、专利授权、转让、上诉结果、引用信息（PatentsView）、专利转让数据库和PTAB上诉数据库。

**📈 对比分析**

通过多变量固定效应回归与PSM匹配，结果显示最高促销密度（Q5）比最低（Q1）在授权率下降5.5个百分点、转让率下降5.9个百分点、上诉成功率下降5.3个百分点；在不同技术领域差异显著；交互分析显示男性与经验丰富审查员对促销语言容忍度更高。

**⚠️ 局限性**

局限性包括：仅适用于美国专利制度，观测设计无法完全消除未观测混淆；词典未捕捉上下文与语气差异；上诉样本有限且未区分转让类型；未评估长期商业化与创新产出等后续影响。

---

## 429. Evolving Idea Graphs with Learnable Edits-and-Commits for Multi-Agent Scientific Ideation

**arXiv ID:** 2605.04922 | [PDF](https://arxiv.org/pdf/2605.04922v1)

**作者:** Jiangwen Dong `[一作]` (Hong Kong Polytechnic University), Wanyu Lin `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 1835 | [OpenAlex ID](https://openalex.org/A5046176565)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用多智能体通过Evolving Idea Graph（EIG）协同生成科学研究提案。

**💡 创新点**

创新点在于将提案过程视为对持久结构化图状态的控制，采用两头图批评器分别决定编辑和何时提交，解决传统文本协作中难以追踪弱点的问题。

**🔧 技术方法**

技术核心是基于图神经网络的共享编码器、冻结快照并行运行、学习的编辑选择头和提交预测头，以及弱监督的训练策略。

**📊 数据集**

使用 AI Idea Bench 2025 与 LiveIdeaBench 两大基准数据集进行评估。

**📈 对比分析**

与 Direct、Self-Refine、Graph of Thoughts、AI-Researcher、VirSci 等基线对比，EIG 在自动评分和专家评估中均取得最高分和最佳排名，显示出显著性能提升。

**⚠️ 局限性**

局限在于仅验证提案层面，缺乏实验验证；训练使用弱监督标签，可能不足以捕捉所有专家洞察；固定图模式可能无法覆盖更复杂的科学结构。

---

## 430. Koopman Identification of Nonlinear Systems via Reservoir Liftings

**arXiv ID:** 2605.04917 | [PDF](https://arxiv.org/pdf/2605.04917v1)

**作者:** Weibin Gu `[一作]` (Tsinghua University), Lu Shi `[通讯]` (Tsinghua University)

**通讯引用:** 99647 | [OpenAlex ID](https://openalex.org/A5100674628)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出将储备网络视为状态化的Koopman字典，通过Echo State Property实现可控记忆，构建RC–Koopman框架对非线性动力学进行线性化；

**💡 创新点**

创新点在于把RC的内部状态直接解释为有限维Koopman观测子空间，并通过谱半径与系统时间尺度的相关性选取算法，确保记忆深度与Koopman谱可观测性匹配；

**🔧 技术方法**

使用技术包括Echo State Network、ESP保证的收敛与数值条件、谱半径选择算法、最小二乘（含岭回归）识别Koopman矩阵，以及对比EDMD与HAVOK的基准实现；

**📊 数据集**

使用的实验数据集为两套仿真基准：Duffing振荡器和差分驱动机器人（全状态测量）;

**📈 对比分析**

与EDMD（RBF字典）和HAVOK（Hankel延迟）在相同的最小二乘框架下比较，评估指标为一阶NRMSE、特征值稳定性、Gram矩阵条件数；RC–Koopman在数值稳定性与特征值分布方面优于两者，预测精度略低于HAVOK但避免了HAVOK出现的不稳定模式；

**⚠️ 局限性**

限制包括对谱半径的调参需求、未在真实非线性系统上验证、对极慢或极快时间尺度的模式可能无法充分捕捉、以及对长期预测或控制性能的进一步研究不足。

---

## 431. Breaking the Quality-Privacy Tradeoff in Tabular Data Generation via In-Context Learning

**arXiv ID:** 2605.04911 | [PDF](https://arxiv.org/pdf/2605.04911v1)

**作者:** Xinyan Han `[一作]` (Tsinghua University), Xingxuan Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 510 | [OpenAlex ID](https://openalex.org/A5054379354)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种名为DiffICL的基于上下文学习（ICL）的表格数据生成框架，用以在小样本场景下兼顾数据质量与隐私保护。

**💡 创新点**

创新点在于将大规模多数据集的结构先验通过ICL预训练迁移到新的数据集，显著降低记忆化风险并实现更优的质量‑隐私权衡。

**🔧 技术方法**

核心技术包括冻结的LimiX表格表示器、在潜在空间训练的条件扩散模型以及双轴注意力Transformer，用以处理可变样本数与特征维度。

**📊 数据集**

实验覆盖了14个真实世界数据集，涵盖7个分类和7个回归任务，涉及多种特征类型。

**📈 对比分析**

与VAE、GAN、扩散、LLM等基准方法对比，DiffICL在下游任务的预测性能和DCROverfit隐私指标上均取得最优或近似最优表现，并在数据增广场景中进一步提升模型效果。

**⚠️ 局限性**

主要限制包括对大型预训练语料库的依赖、模型训练与推理的计算成本，以及对极端小样本或高度异构数据的适用性仍需进一步验证。

---

## 432. On the Influence of the Feature Computation Budget on Per-Instance Algorithm Selection for Black-Box Optimization

**arXiv ID:** 2605.04954 | [PDF](https://arxiv.org/pdf/2605.04954v1)

**作者:** Koen van der Blom `[一作]` (Centrum Wiskunde & Informatica), Diederick Vermetten `[通讯]` (Sorbonne Université)

**通讯引用:** 731 | [OpenAlex ID](https://openalex.org/A5075992713)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估黑盒优化中，使用不同特征计算预算的PIAS相较于单一最佳求解器（SBS）的性能，系统化实验共覆盖1440个场景

**💡 创新点**

系统揭示特征预算占比对PIAS收益的影响，并量化特征预算导致的VBS-PIAS损失比例，说明特征预算不可忽视

**🔧 技术方法**

采用ELA特征、Sobol采样、Shapley值辅助的4算法子集、基于多输出回归的选择模型，以及多折交叉验证

**📊 数据集**

BBOB、MA‑BBOB、RandOptGen三大BBO基准套件，22个全量优化器与4个高互补性子集

**📈 对比分析**

与SBS和VBS比较，使用VBS–SBS间隙闭合率评估；在大多数情形下，PIAS可超过SBS，甚至在特征预算高达25%时仍能获胜；但在MA‑BBOB全量组合中PIAS往往不优

**⚠️ 局限性**

受限于问题集、算法组合和预算分配的多变性；BBOB过于“易”，难以体现算法选择差异；缺乏对动态特征获取和温启动等方法的深入探索

---

## 433. Why Geometric Continuity Emerges in Deep Neural Networks: Residual Connections and Rotational Symmetry Breaking

**arXiv ID:** 2605.04971 | [PDF](https://arxiv.org/pdf/2605.04971v1)

**作者:** Kyungwon Jeong `[一作]` (Hyntel), Honggyo Suh `[通讯]` (Hyntel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究深度网络权重矩阵的几何连续性起因，并通过对残差 MLP 与 Transformer 的消融实验验证残差连接与对称性破坏非线性共同决定连续性的机制

**💡 创新点**

首次揭示残差梯度相干与对称性破坏是产生几何连续性的必要条件，并发现激活与归一化分别集中与分散连续性，解释 Transformer 中投影特定连续性

**🔧 技术方法**

采用残差网络、SVD、梯度累计分析、旋转对称性理论、对齐指标（v1 连贯性、σ² 加权连贯性）以及 Radial 激活等技术

**📊 数据集**

MNIST 用于 MLP 实验，WikiText‑103 用于小 Transformer 实验，Llama‑3.1‑8B 预训练模型用于验证

**📈 对比分析**

通过残差、激活、归一化消融与 Radial 激活对比，发现残差+ReLU 保持 v1 连贯≈0.96，去掉激活或使用旋转保持激活则连贯降至≈0.22；小 Transformer 维持 PPL≈42 并出现投影特定连贯；预训练 Llama 连续性高于随机，证明机制普适

**⚠️ 局限性**

仅在小模型与单个随机种子下验证，Q/K 对齐机制未直接消融，旋转漂移在大模型训练动态未完整探测

---

## 434. Attention-Based Chaotic Self-Supervision for Medical Image Classification

**arXiv ID:** 2605.04985 | [PDF](https://arxiv.org/pdf/2605.04985v1)

**作者:** Joao Batista Florindo `[一作]` (University of Campinas), Amanda Pontes de Oliveira Ornelas `[通讯]` (University of Campinas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Chaotic Denoising Autoencoder (CDAE)自监督预训练策略，并设计注意力融合模型，将CDAE特征与传统监督模型特征结合用于医学图像分类。

**💡 创新点**

创新点在于：1）利用确定性混沌映射（logistic map）对图像进行像素级破坏，迫使编码器学习逆混沌的结构化特征；2）引入注意力融合机制，使模型动态组合域特定与通用特征。

**🔧 技术方法**

技术包括 ConvNeXt 骨干网络、混沌像素映射（r=3.99）、自监督预训练 + 监督微调、Squeeze‑and‑Excite 注意力模块、线性分类器。

**📊 数据集**

使用 ISIC 2018（皮肤病变）和 APTOS 2019（糖尿病视网膜病变）公开数据集进行实验。

**📈 对比分析**

与基线、焦点损失、LDAM、DANIL、CL、Resample、ProCo、FG‑SSL 等 SOTA 方法对比，ISIC 2018 的准确率 0.9221、F1‑宏 0.8530，APTOS 2019 的准确率 0.8644、F1‑宏 0.7433，均显著优于其他方法。

**⚠️ 局限性**

局限性包括：1）仅在二维彩色图像上验证，未扩展到 3D 医学影像；2）混沌参数固定，缺乏多参数探索；3）未深入评估模型对不同疾病级别或不平衡数据的鲁棒性。

---

## 435. Self-Induced Outcome Potential: Turn-Level Credit Assignment for Agents without Verifiers

**arXiv ID:** 2605.04984 | [PDF](https://arxiv.org/pdf/2605.04984v1)

**作者:** Senkang Hu `[一作]` (Hong Kong Jc Stem Lab Of Smart City), Yuguang Fang `[通讯]` (Hong Kong Jc Stem Lab Of Smart City)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无监督的自回归强化学习框架SIOP，用来在多轮交互式问答中给每个中间回合分配奖励。

**💡 创新点**

创新点在于将终端答案聚成语义模式作为潜在未来状态，并利用可靠性校准的目标分布计算潜在值函数，从而在没有黄金答案或任务特定验证器的情况下实现逐回合奖励。

**🔧 技术方法**

使用了大型语言模型（Qwen3-4B/8B）作为策略，基于NLI模型做语义聚类与证据可靠性评估；构造了分布式潜在值函数、逐回合优势和GRPO风格的剪切目标。

**📊 数据集**

在七个搜索增强问答基准上评估：Natural Questions、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle。

**📈 对比分析**

与两种无监督基线（TTRL、EMPO）以及两种金标监督基线（GRPO、IGPO）对比，SIOP在4B/8B模型上均获得最高的无监督性能，单项指标上在多跳任务中超过金标监督的GRPO，平均精确匹配与Token F1均提升1–2个百分点。

**⚠️ 局限性**

局限性包括对NLI聚类与可靠性评估的依赖，聚类质量可能受语义模糊影响；对大规模模型或多模态任务的泛化尚未验证；算法计算开销较高，尤其是多回合奖励计算与优势归一化。

---

## 436. AllSERP: Exhaustive Per-Element Enrichment of the Versatile AdSERP Dataset

**arXiv ID:** 2605.04949 | [PDF](https://arxiv.org/pdf/2605.04949v1)

**作者:** K. Andrew Edmonds `[一作]` `[通讯]` (Independent), K. Andrew Edmonds (Independent)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

AllSERP通过基于截图的计算机视觉定位和8层HTML解析，对AdSERP 2776条查询‑SERP数据进行像素级AOI提取，并为每个页面元素打上类型标签，完成对原始数据的全元素行为丰富；

**💡 创新点**

其创新点在于引入截图锚定的CV定位、跨层级HTML标签链、Gap‑fill中间点分割以及X+Y包含判定，既实现了91.7%点击归属率，又在不改动原始多模态信号的前提下提供了更细粒度的行为指标；

**🔧 技术方法**

主要技术包括Python流水线脚本、行距标准差投影的视觉行检测、8层优先级HTML标签解析、X+Y坐标包含判断、以及中点分割的Gap‑fill填充；

**📊 数据集**

使用的数据集为AdSERP的2776条实验数据，包含全页截图、捕获的SERP HTML、150 Hz眼动、鼠标轨迹、滚动、瞳孔等多模态信息；

**📈 对比分析**

与原AdSERP广告框架进行内部一致性验证，发现0误差，且点击归属率达91.7%，同时提供按元素类型的点击率、视线覆盖率与归属统计，可与原始广告‑organic分离结果直接比较；

**⚠️ 局限性**

局限性包括仅覆盖预AI‑Overviews单次强制点击任务，不涉及查询重写、分页或多查询；Gap‑fill填充仅为启发式方法，右侧rail organic及新出现的AI回答卡等元素仍需后续扩展标签。

---

## 437. Curated AI beats frontier LLMs at pharma asset discovery

**arXiv ID:** 2605.04908 | [PDF](https://arxiv.org/pdf/2605.04908v1)

**作者:** Łukasz Kidziński `[一作]` (Gosset Research), Kevin Thomas `[通讯]` (Gosset Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在10个特定的肿瘤/免疫学靶点上，使用自然语言查询，对Gosset（基于策划的药物资产索引的聊天接口）与四个前沿网页搜索LLM（Claude、GPT、Gemini、Perplexity）进行同一输入、统一JSON输出格式的性能基准评估。

**💡 创新点**

创新点在于证明：将策划的药物资产索引与与前沿LLM相同的聊天前端相结合，可在保持完美精确度（100%）的前提下，实现3.2倍以上的验证药物覆盖率（100%召回），并通过将索引暴露为MCP工具，进一步让其他LLM几乎完全消除召回差距。

**🔧 技术方法**

使用技术包括：基于Mongo的索引检索、JSON结构化输出、确定性自动通过（自动标记已验证药物）、三方LLM交叉判断（Claude 4.7、GPT 5.5、Gemini 3.1 Pro）加人工专家审核、别名感知的并查集去重、延迟测量等。

**📊 数据集**

数据集为Gosset策划的目标/模态/适应症级药物资产索引，覆盖预临床和亚洲/学术研发项目；对10个靶点（TL1A、OX40L、IL-36R、TROP-2、B7‑H3、ROR1、NaPi2b、Claudin 18.2、FAP、GPRC5D）进行检索，最终形成跨系统验证药物的并集（共451种）。

**📈 对比分析**

比较方法：统一提示、统一JSON schema、同一评价标准；评价指标包括精确度（Precision）、召回率（Recall = 已验证药物数/跨系统验证药物并集）、幻觉率、延迟。结果显示：Gosset提供451种已验证药物（Precision = 1.000、Recall = 1.000、幻觉0、延迟≈子秒），而四个前沿LLM的Recall仅为0.17–0.31，精确度几乎相同，且延迟显著更高。

**⚠️ 局限性**

局限性包括：召回评估仅限于可公开追溯的药物（公共渠道公开的药物），无法覆盖完全内部或未公开的研发项目；判定者本身为LLM，存在校准误差；靶点选择偏向Gosset数据丰富的领域，可能对广泛覆盖的靶点（如PD‑1、HER2）表现不同；以及跨系统并集仍取决于前沿LLM能检索到的公开信息。

---

## 438. Uno-Orchestra: Parsimonious Agent Routing via Selective Delegation

**arXiv ID:** 2605.05007 | [PDF](https://arxiv.org/pdf/2605.05007v1)

**作者:** Zhiqing Cui `[一作]` (Nanjing University Of Information Science And Technology), Usman Naseem `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Uno-Orchestra，一种统一的任务分解与路由策略，能够在单个模型内部同时决定是否拆分任务以及为每个子任务分配合适的（模型、原语）对。

**💡 创新点**

创新点在于将任务拆分与路由合并为一个单一的因果语言模型决策，既能自适应拆分深度，又能在预算约束下选择最优执行者；并引入了 Agentic-GRPO 的 turn‑level 信用分配机制，让强化学习在多轮交互中更高效。

**🔧 技术方法**

主要技术包括：监督微调 (SFT) 训练从真实环境交互中提取的教师轨迹；多阶段强化学习 (Agentic-GRPO) 结合组相对优势估计和 KL 正则化；基于闭合原语词表的可组合路由；以及统一的奖励设计（验证器 + 成本惩罚）。

**📊 数据集**

使用的数据集包括 38 个公开数据集的 61,201 条教师轨迹（用于 SFT）以及 2,976 条验证过滤后的难题集（用于 RL），并在 13 份不与训练集重叠的基准上进行评估，涵盖数学、代码、知识推理、长上下文与工具使用等能力维度。

**📈 对比分析**

与 22 个基准对手（包括单轮路由器、层次化工作流和多轮 RL 路由器）比较，Uno-Orchestra 在 13 组基准上的宏观 pass@1 达到 77.0%，比最强工作流基准高约 16%，同时每查询成本和上下文长度降低约 10 倍，显示出显著的精度‑效率优势。

**⚠️ 局限性**

局限性包括：对预先准备的、人工标注的 RL 轨迹高度依赖；在极大规模或多样化工具环境下可能需要进一步扩展原语词表；并且在小型路由器或极端成本约束下的性能仍有提升空间。

---

## 439. Misaligned by Reward: Socially Undesirable Preferences in LLMs

**arXiv ID:** 2605.05003 | [PDF](https://arxiv.org/pdf/2605.05003v1)

**作者:** Gayane Ghazaryan `[一作]` (University of Stuttgart), Esra Dönmez `[通讯]` (University of Stuttgart)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5065662308)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将社会评估数据转化为对比偏好数据，构建统一框架对奖励模型在偏见、安全、道德、伦理四个社会维度进行评估。

**💡 创新点**

创新在于：1）将多种社会评价数据（含有金标或方向性标记）统一为对比偏好形式；2）扩展奖励模型基准到社会对齐领域；3）系统比较多模型在不同领域的表现，揭示无单一模型优于所有任务。

**🔧 技术方法**

使用奖励模型对比评分、准确率、对数赔率、平均边际等指标；采用RLHF框架中的奖励模型训练与评估技术；构建偏好数据集并计算统计量。

**📊 数据集**

使用的数据集包括Gretel（安全）、ETHICS（伦理推理）、Moral Stories（道德规范）、StereoSet（社会偏见）与Winogender（性别偏见）。

**📈 对比分析**

通过准确率、对数赔率和平均边际等度量对七个奖励模型进行横向比较，结果显示各模型在不同领域表现差异大，整体偏好不稳定且常偏向社会不良选项，缺乏统一优秀模型。

**⚠️ 局限性**

局限性包括：仅使用英文数据且未覆盖多语言/文化差异；评估仅限于二选一偏好，未考虑多选竞争；未直接检验奖励模型对RLHF下游行为的实际影响。

---

## 440. Exhaustive Symbolic Integration: Integration by Differentiation and the Landscape of Symbolic Integrability

**arXiv ID:** 2605.04978 | [PDF](https://arxiv.org/pdf/2605.04978v1)

**作者:** Harry Desmond `[一作]` (University of Portsmouth), Harry Desmond `[通讯]` (University of Portsmouth)

**通讯引用:** 1900 | [OpenAlex ID](https://openalex.org/A5065060439)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文提出一种名为 Exhaustive Symbolic Integration（ESI）的算法，通过枚举给定操作符集合下的所有符号函数并计算其导数，进而确定哪些函数在同一操作符类内具有闭式原函数；并利用此枚举结果评估“可积分性分数”——在限定复杂度k内可被自身闭合积分的函数比例；

**💡 创新点**

创新点包括：①首次系统地量化可积分性分数并揭示其随复杂度和操作符基底变化的规律，尤其发现对数函数显著提升可积分性；②利用完整枚举实现全局积分搜索，发现三类在六大CAS（SymPy、Mathematica、Rubi、FriCAS、Maxima、Giac）中均无法解析的全新原函数；

**🔧 技术方法**

核心技术为 Exhaustive Symbolic Regression（ESR）生成符号函数集，随后使用 SymPy 进行求导、数值指纹化（多点数值评估+MD5哈希）去重，构建导数映射；对可积分性分数做二项计数；对 CAS 进行多引擎、多策略的积分尝试与压力测试；

**📊 数据集**

数据集为通过 ESR 枚举得到的符号函数集合，按五种操作符基底（含/不含对数、三角函数等）在最大复杂度k≤10 产生的 10^6 级别函数；对每个函数进行求导后生成导数指纹，形成可积分性统计与 CAS 失败集；

**📈 对比分析**

与 CAS 的比较采用多引擎测试（SymPy、Mathematica、Rubi、FriCAS、Maxima、Giac）并设置 180s/600s 计算时限；ESI 在所有测试中找出了 232 个 SymPy/Mathematica 失败后 FriCAS 仍能求解的案例，并在极少数（3 个）积分上突破六大 CAS，证明其在发现新可积分函数方面的优势；

**⚠️ 局限性**

局限性在于：①枚举仅限于给定的操作符基底与复杂度上限（k≤10），难以覆盖更复杂表达式；②数值指纹化与符号简化可能漏掉等价函数导致可积分性分数略低；③对参数多项式的处理有限，约 1% 函数可能被误判；④结果高度依赖表达式语法与复杂度度量，换用其他度量时峰值与比例会改变。

---

## 441. ICPR 2026 Competition on Privacy-Preserving Person Re-Identification from Top-View RGB-Depth Camera (TVRID)

**arXiv ID:** 2605.04977 | [PDF](https://arxiv.org/pdf/2605.04977v1)

**作者:** Raphaël Delécluse `[一作]` (IMT Nord Europe), Laurent Guimas `[通讯]` (Explain)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了 TVRID 数据集，组织了 ICPR 2026 公开竞赛，定义三条评估轨迹（RGB Re-ID、Depth Re-ID、RGB↔Depth 交叉模态 Re-ID），并给出统一的评测脚本与排行榜。

**💡 创新点**

创新点包括：① 在同步 RGB‑Depth 采集的四台俯视摄像机上构造结构化的视角与高度变化（平地、上坡、下坡、斜角），提供了可控的几何变换；② 将深度感知作为隐私友好型 Re-ID 的评估基准，并引入跨模态检索任务来量化隐私‑性能权衡；③ 在公开数据集上进行大规模方法比较，形成可复现的 benchmark。

**🔧 技术方法**

使用的技术主要有：深度度量学习与 Transformer 视觉模型（ViT‑Base/16、ConvNeXt‑Base、EVA‑CLIP）、对比学习与交叉模态对齐（VSLA‑CLIP、MINER、XM‑VSLA）、PK 采样、三重损失（分类、Triplet、InfoNCE）、k‑reciprocal 重新排序等；提交者也常结合数据增强、伪 RGB 处理、时序池化等工程技巧。

**📊 数据集**

官方使用 TVRID 数据集（86 个身份、四台同步 Intel RealSense D455 俯视摄像机，包含 RGB/Depth 双流、IN/OUT 观察），参赛团队也会在此基础上使用其它顶视 Re‑ID 数据集进行预训练或数据扩增。

**📈 对比分析**

评测指标为 mAP 与 CMC@1，分别在三种场景（同摄像机 IN/OUT、上坡‑下坡、平地‑斜角）下平均得到最终得分。RGB 轨道最高（≈99% mAP / 100% CMC@1），Depth 次高（≈98% mAP / 99% CMC@1），交叉模态最低（≈90–99% mAP / 90–99% CMC@1）。顶级方法在 Depth 和跨模态上也可取得接近 98–99% 的 mAP，说明即使在无色彩信息或模态不匹配的条件下，经过跨模态对齐与对比学习的模型仍能保持较高的识别精度。

**⚠️ 局限性**

局限性包括：① 仅覆盖俯视场景，难以直接推广至侧视或前视 Re‑ID；② 数据量相对有限（86 人、4 台摄像机），对更大规模多样化环境的鲁棒性评估不足；③ 虽然 Depth 在提升隐私方面有优势，但跨模态检索表明在存在 RGB 旁路或辅助信息时，深度并非完全隐私保证；④ 评测主要依赖 CMC@1 与 mAP，未考虑实时性、计算成本或部署成本等工程指标。

---

## 442. Skill Neologisms: Towards Skill-based Continual Learning

**arXiv ID:** 2605.04970 | [PDF](https://arxiv.org/pdf/2605.04970v1)

**作者:** Antonin Berthon `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 22934 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结LLM参数的前提下，提出通过向模型词表中添加软 token（skill neologism）并在技能中心化数据集上训练，来学习并扩展特定技能，同时保持模型的可组合性。

**💡 创新点**

创新点在于：① 将技能学习转化为词汇级软 token 训练，避免权重更新；② 通过技能中心化训练促使学习到通用可组合的技能表示；③ 实现了独立学习的技能可零-shot组合。

**🔧 技术方法**

采用软 token（soft prompt）embedding、词表扩展、插入函数、交叉熵训练，比较了 LoRA、Prompt Tuning 等参数高效微调方法，并使用 Qwen2.5‑0.5B 作为基础模型。

**📊 数据集**

使用自生成的数字序列转换数据集（包含多种组合的算法技能），以及开放源代码 LLM 预训练文本用于验证词表中已有的技能 token。

**📈 对比分析**

与 LoRA、Prompt Tuning 及 In‑Context Learning 进行对比，评估在 ID 与 OOD 技能组合下的准确率。结果表明 Skill Neologisms 在 OOD 组合上保持近乎完美的性能，且能零-shot 组合独立学习的技能，优于对比方法。

**⚠️ 局限性**

局限性：仅在受控的合成任务上验证；难以构造大规模多样化的技能中心化数据集；软 token 优化易受初始化和梯度噪声影响；尚未在自然语言任务中进一步测试。

---

## 443. Order-based Rehearsal Learning

**arXiv ID:** 2605.04955 | [PDF](https://arxiv.org/pdf/2605.04955v1)

**作者:** Yu-Xuan Tao `[一作]` (Nanjing University), Zhi-Hua Zhou `[通讯]` (Nanjing University)

**通讯引用:** 62154 | [OpenAlex ID](https://openalex.org/A5100621138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于变量顺序的复习学习方法 OLEM-Rh，先通过信息论的条件熵最大化学习顺序，再利用顺序信息构建后决策采样器，完成避免不良未来（AUF）的决策推荐。

**💡 创新点**

创新点在于：①证明了图结构对 AUF 决策并非必要，顺序结构即可识别决策影响；②提出了无约束的 OLEM 顺序学习方法，使用条件熵最大化；③设计了顺序采样器与可微优化的决策框架，首次实现完全可微的 AUF 决策流程。

**🔧 技术方法**

采用的技术包括信息论条件熵最大化 OLEM、条件流模型（cINN）用于后决策采样、Chebyshev 中心与梯度优化（Adam）、零阶/一阶混合策略等。

**📊 数据集**

实验数据集涵盖：①合成数据（Erdős–Rényi 随机图、不同维度、密度、线性比例、噪声分布），② Sachs 蛋白质网络（11 维），③ Bermuda 环境测量数据（10 组伪真实数据）。

**📈 对比分析**

与 LISTEN、DiffAN、CAM、SCORE、CaPS、NOTEARS、GES 等基线在顺序学习上通过 DIV、SHD、SID 评价，OLEM 在 DIV 上排名第二、SHD/SID 最佳；在 AUF 决策上，与学习图结构的 Grad-Rh（多种结构学习器）比较，OLEM‑Rh 取得更高成功率，接近 oracle Grad‑Rh，并优于 QWZ23，证明顺序学习可达标甚至优于图结构方法。

**⚠️ 局限性**

局限性包括：假设无隐藏变量且无反馈；顺序学习的条件熵估计需要足够样本，样本不足时性能下降；对大规模变量集的计算复杂度和可扩展性尚未充分验证；对条件流模型训练的稳定性与可解释性依赖实现细节。

---

## 444. Interaction Tree Semantics for RISC-V: Bridging Compiler and Hardware Verification

**arXiv ID:** 2605.04933 | [PDF](https://arxiv.org/pdf/2605.04933v1)

**作者:** Shuanglong Kan `[一作]` (Barkhausen Institut), Sebastian Ertel `[通讯]` (Barkhausen Institut)

**通讯引用:** 115 | [OpenAlex ID](https://openalex.org/A5029364833)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

构建了基于 Interaction Trees 的 RISC‑V ISA 形式化语义，并在同一 Coq 框架内完成了从 LLVM IR 到机器码、宏操作融合重排以及硬件 ALU 与 ISA 语义的一致性验证；

**💡 创新点**

首次将 ITrees 与弱 Bisimulation、refinement 结合，用单一可信框架实现跨层验证，且提供完整的机器检查指令语义与可执行模拟器；

**🔧 技术方法**

Coq 证明、Interaction Trees、弱 Bisimulation、refinement、OCaml 生成的可执行模拟器、翻译验证技术；

**📊 数据集**

官方 RISC‑V ISA Test Suite（I、M、F、A、Zicsr 扩展），共 172 个 ELF 测试文件、3,228 个测试案例；

**📈 对比分析**

通过提取的 OCaml 模拟器跑官方测试验证语义正确性；案例研究使用 coinductive bisimulation 比较 LLVM IR 与 RISC‑V 代码、重排前后程序的可执行行为，覆盖率高但未给出性能数值；

**⚠️ 局限性**

仅覆盖了 RISC‑V 基本与主要扩展，未验证完整编译器路径或所有硬件实现细节，对时序/并发特性支持有限，且部分数据相关指令（如 FMA）需手工证明。

---

## 445. When Does Gene Regulatory Network Inference Break? A Controlled Diagnostic Study of Causal and Correlational Methods on Single-Cell Data

**arXiv ID:** 2605.04930 | [PDF](https://arxiv.org/pdf/2605.04930v1)

**作者:** Miguel Fernandez-de-Retana `[一作]` (University of Deusto), Aitor Almeida `[通讯]` (University of Deusto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个受控诊断框架，对单细胞转录组中的基因调控网络推断进行系统评估，独立调节七种生物学/技术病理，并对六类代表性方法进行性能测评。

**💡 创新点**

创新点在于：①构建可调节的“病理旋钮”模拟框架，①提供错误类型分解以揭示不同方法的失败模式；②通过交互扫描揭示多病理联合效应的亚加性特征和隐性交叉点；③为现有基准排名提供因果解释。

**🔧 技术方法**

技术手段包括：基于线性加性噪声结构因果模型的合成数据生成；dropout、潜在混杂、细胞类型混合、反馈循环、网络密度、样本量、伪时间漂移七个病理参数；使用AUPRC（无向/有向）和五类错误分解指标；对六种方法（Pearson、MI、GENIE3、PC、GES、NOTEARS）在每个病理级别下进行10次重复实验。

**📊 数据集**

使用完全合成的单细胞表达数据（p=25基因、n=800-3200细胞），并在两套模型（线性与非线性）下重复实验；未直接使用真实实验数据，但在讨论中引用了CausalBench、geneRNIB等基准。

**📈 对比分析**

与六种方法比较发现：在干净条件下，NOTEARS（AUPRC≈0.99）> GES> PC> GENIE3> MI> Pearson；dropout导致MI、GENIE3严重下滑，Pearson保持相对稳定；潜在混杂和细胞类型混合压缩所有方法差异；NOTEARS在反馈、密度和大样本量下表现最稳健；多病理交互扫描显示整体降幅亚加性，且在高密度+高dropout下GES能取代Pearson。

**⚠️ 局限性**

局限性包括：①主要使用线性加性噪声模型，虽在非线性实验中保持趋势但仍有限；②基因数仅25，难以推广至全基因组规模；③全部实验为合成数据，缺乏对真实单细胞病理的直接映射；④未针对多尺度/多组学数据做进一步验证。

---

## 446. Reinforcement Learning for Compositional Generalization with Outcome-Level Optimization

**arXiv ID:** 2605.04920 | [PDF](https://arxiv.org/pdf/2605.04920v1)

**作者:** Xiyan Fu `[一作]` (Nanyang Technological University), Wei Liu `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探索使用基于结果的强化学习（GRPO）提升语言模型的组合泛化能力，取代传统的逐词监督微调。

**💡 创新点**

首次证明仅通过全局奖励（二元或复合）即可显著提高组合泛化，并指出强化学习可重塑输出分布，减少对训练模式的过拟合。

**🔧 技术方法**

采用Group Relative Policy Optimization（GRPO）进行强化学习，设计二元奖励与基于原语与结构的复合奖励，使用温度采样和KL正则化。

**📊 数据集**

在SCAN、COGS、GeoQuery、CFQ四大组合泛化基准上进行实验。

**📈 对比分析**

相较于SFT基线，GRPO（无论是二元还是复合奖励）在所有基准上均取得更高的精确匹配率，特别是在长度扩展与生产性测试上提升显著；在Top‑k评估中，GRPO显著提升pass@1，pass@k差距减小。

**⚠️ 局限性**

强化学习相较于SFT引入额外计算成本；实验仅探讨了简单奖励设计，未尝试更复杂的奖励或其他强化学习框架，限制了方法的普适性与进一步提升空间。

---

## 447. A Foundation Model for Zero-Shot Logical Rule Induction

**arXiv ID:** 2605.04916 | [PDF](https://arxiv.org/pdf/2605.04916v1)

**作者:** Yin Jun Phua `[一作]` (Institute of Science Tokyo), Yin Jun Phua `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 46 | [OpenAlex ID](https://openalex.org/A5001289731)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Neural Rule Inducer（NRI），一种基于预训练的零样本规则诱导模型，能够在不进行任务特定训练的情况下，从数据中学习可解释的 DNF 逻辑规则。

**💡 创新点**

创新点包括：① 用身份无关的统计特征（类条件率、熵、共现）编码文字，消除对谓词身份的依赖；② 采用并行槽式解码器保持逻辑析取的置换不变性；③ 用产品 t‑范数实现规则执行的可微化；④ 在大规模合成布尔公式上预训练，达成跨域零样本迁移。

**🔧 技术方法**

技术手段包括：统计特征编码 + 多头注意力、FiLM 调制、槽式 Transformer 解码器、t‑范数软执行、复合损失（覆盖、平衡、最大间距、对抗等）以及自适应维度扩展。

**📊 数据集**

数据集：完全在随机生成的布尔 DNF 公式上预训练；评估在 14 个 UCI 基准（adult、breast-cancer、car、credit、diabetes、german、hepatitis、ionosphere、kr-vs-kp、mushroom、nursery、spambase、tic-tac-toe、vote）上进行零样本迁移；并在合成公式上测试规则恢复、噪声鲁棒性、伪变量鲁棒性等。

**📈 对比分析**

与 8 种基线（XGBoost、LightGBM、EBM、RIPPER、RuleFit、FIGS、决策树、神经 DNF）在 5 折交叉验证下比较。NRI 在零样本设置下平均准确率 69.7%，比有训练的基线低约 13个百分点；在噪声和伪变量鲁棒性实验中表现优于符号方法；规则恢复的逻辑匹配随规则复杂度下降，但预测准确率保持在 85–100% 之间。

**⚠️ 局限性**

局限性：① 仅处理二值化特征；② 规则长度和条目数受训练分布（K≤6，L≤4）限制，难以处理更复杂规则；③ 仍需大规模合成数据预训练；④ 计算复杂度随 N² 增长，注意力成本高；⑤ 对连续、多值或关系型逻辑的扩展尚未实现。

---

## 448. Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training

**arXiv ID:** 2605.04913 | [PDF](https://arxiv.org/pdf/2605.04913v1)

**作者:** Hengyu Shi `[一作]` (Independent Researcher), Junhao Su `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Local-Learning Post-Training (LoPT) 的方法，通过在模型中间插入停止梯度边界，将任务梯度的传播范围限制在后半段，从而实现高效的后训练。

**💡 创新点**

创新点在于将梯度传播深度（gradient reach）视为可设计的超参数，利用局部特征重建目标使前半段可训练但不直接受到任务梯度影响，显著降低内存占用和训练成本，同时保持甚至提升下游任务性能。

**🔧 技术方法**

核心技术包括：1) 在 transformer 的中间层插入 stop‑gradient；2) 前半段使用轻量级 MLP 进行特征重建；3) 两阶段更新顺序（先更新前半段与重建头，再重新计算边界后更新后半段）。与 LoRA、梯度检查点、ZeRO、流水线并行等常见系统级优化可无缝组合。

**📊 数据集**

在监督微调 (SFT) 上使用 Alpaca‑52K、Tulu‑3、Magpie、MetaMathQA 等数据集；在强化学习 (GRPO) 上使用 GSM8K、NuminaMath；同时对比基线模型（Base）、全深度梯度传播（E2E）和 LoPT 的性能与资源消耗。

**📈 对比分析**

与 E2E 的对比显示：在 SFT 中，LoPT 在大多数模型与数据集上保持或略优的七大基准得分，且峰值显存降低 23–36%，吞吐率提升 2–7%；在 GRPO 中，LoPT 与 E2E 的质量几乎持平，政策更新阶段显存降低 21–24%，步长时间缩短 7–21%。LoPT 同时与 LoRA、梯度检查点、ZeRO 等方法兼容，进一步提升效率。

**⚠️ 局限性**

局限性包括：1) 仅在 4B–8B 规模模型上验证，未证明在更大规模或 MoE 模型上的效果；2) 采用单一中点分割，未探索更细粒度或可调的分割策略；3) 对于极端稀疏或高度对齐的数据集，LoPT 的优势可能不明显。

---

## 449. Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games

**arXiv ID:** 2605.04906 | [PDF](https://arxiv.org/pdf/2605.04906v1)

**作者:** Yidong He `[一作]` (South China University of Technology), Mengchen Zhao `[通讯]` (South China University of Technology)

**通讯引用:** 668 | [OpenAlex ID](https://openalex.org/A5103832112)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多智能体游戏中提升大型语言模型的战略推理能力，提出了Strat-Reasoner框架；

**💡 创新点**

引入递归推理范式让代理在决策时显式考虑对手思维，结合集中式链式思维（CoT）比较与混合优势估计，实现对多步推理的细粒度信用分配；

**🔧 技术方法**

基于强化学习（GRPO）与LLM的Chain-of-Thought、递归推理、中心化评估、微分步Rollout以及混合优势（CoT优势+回报优势）的技术组合；

**📊 数据集**

使用开源LLM Qwen3-4B 进行训练，并在三类两人交替马尔可夫游戏上评估：完美信息对抗游戏（井字棋）、不完全信息对抗游戏（Kuhn Poker）、不完全信息合作游戏（MiniHanabi），以及OOD测试如Connect Four、Leduc Hold'em、SimpleHanabi；

**📈 对比分析**

与多种基线模型对比（Qwen3-8B/32B、Gemma3-12B、SPIRAL、MARSHAL、GPT‑5‑mini、Gemini‑2.5‑flash），结果显示在所有游戏中Strat‑Reasoner‑4B平均提升约22.1%，在OOV环境下仍保持竞争力，甚至超过部分更大参数模型；

**⚠️ 局限性**

局限包括：仅针对两人交替游戏验证，扩展到多智能体或更复杂环境需进一步工程改造；计算成本高，主要瓶颈在多轮Rollout；对对手建模假设为同质或有限深度，可能在对手策略更高深时失效。

---

## 450. Cross-Model Consistency of Feature Importance in Electrospinning: Separating Robust from Model-Dependent Features

**arXiv ID:** 2605.04905 | [PDF](https://arxiv.org/pdf/2605.04905v1)

**作者:** Mehrab Mahdian `[一作]` (Tallinn University of Technology), Tamas Pardy `[通讯]` (Tallinn University of Technology)

**通讯引用:** 454 | [OpenAlex ID](https://openalex.org/A5011828869)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估在小样本电纺实验数据上不同机器学习模型的特征重要性跨模型一致性，利用SHAP对21种模型进行特征重要性排名并量化一致性。

**💡 创新点**

引入跨模型特征重要性一致性度量，区分稳健特征与模型依赖特征，并强调解释性与预测性是不同属性。

**🔧 技术方法**

训练线性、树、核、神经网络和实例化等21种模型，使用TreeSHAP/KernalSHAP计算特征重要性，并用平均排名、标准差、Spearman相关等指标评估一致性。

**📊 数据集**

使用96个聚乙烯醇（PVA）电纺实验的RSM设计数据集，包含四个工艺参数和纤维直径目标。

**📈 对比分析**

通过5折交叉验证评估R²，树模型最高约0.9；不同模型预测相似但特征重要性差异显著，特征重要性一致性指标显示溶液浓度稳健，其余特征模型依赖。

**⚠️ 局限性**

数据量小、仅包含单一材料、未考虑环境变量，SHAP仅捕捉关联非因果，默认超参数可能低估模型性能。

---

## 451. Delving into Non-Exchangeability for Conformal Prediction in Graph-Structured Multivariate Time Series

**arXiv ID:** 2605.04957 | [PDF](https://arxiv.org/pdf/2605.04957v1)

**作者:** Ruichao Guo `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9373 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

本文针对图结构多变量时间序列预测中的不确定性量化问题，提出了一种在频域上实现条件可交换性的置信区间预测框架，解决了传统CP在非可交换情境下失效的问题。

**💡 创新点**

创新点包括：①提出Spectral Graph Conditional Exchangeability（SGCE）概念，将高频成分在低频条件下实现可交换性；②基于SGCE设计SCALE框架，利用图小波变换分解信号，并通过自适应门控结合低频趋势与高频残差来生成预测区间。

**🔧 技术方法**

采用技术：图小波变换（Spectral Graph Wavelet Transform）进行频域分解；STGNN等图时序网络作为预测器；自适应门控网络融合低频编码与高频统计；传统Split Conformal Prediction与量化回归损失。

**📊 数据集**

实验数据集：METR-LA、PEMS04、PEMS07、PEMS08等交通流量图数据。

**📈 对比分析**

与SCP、SeqCP、NexCP、EnbPI、HopCPT、CoREL、ConForME等基线方法对比，在覆盖率、PI宽度和Winkler指标上，SCALE能够在保持近似显著覆盖率的同时，显著减小区间宽度，优于现有方法。

**⚠️ 局限性**

局限性：①需手动或经验选择小波尺度与门控阈值；②理论保证基于理想频域分离，实际会受到谱泄漏与高频耦合的影响；③对低频趋势变化的鲁棒性有限，且对图结构的稳定性有一定依赖。

---

## 452. Conceptors for Semantic Steering

**arXiv ID:** 2605.04980 | [PDF](https://arxiv.org/pdf/2605.04980v1)

**作者:** Ilias Triantafyllopoulos `[一作]` (New York University), João Sedoc `[通讯]` (New York University)

**通讯引用:** 1771 | [OpenAlex ID](https://openalex.org/A5058954591)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用概念子矩阵（Conceptor）在大型语言模型的隐藏层进行激活层级 steering，实现对多维语义概念（如情感、政治倾向、抑郁语言）的精准控制和组合。

**💡 创新点**

创新点在于：①采用双极子（bipolar）概念子矩阵捕获完整概念子空间，显著提升对单向向量方法的覆盖；②引入无监督层选择诊断（概念子矩阵 quota），无需监督即可定位最合适的干预层；③实现闭式 Boolean 代数（AND、OR、NOT）使概念可无训练组合，支持多概念的推理与安全对齐。

**🔧 技术方法**

核心技术包括概念子矩阵的构造与正则化、谱诊断（quota）、闭式 Boolean 运算、激活插值/替换干预方式、以及对比句子对的无监督训练。

**📊 数据集**

数据集方面，使用 100 对比句子对（正负极）训练概念子矩阵，在 Gemma‑2‑2B‑IT、Gemma‑2‑9B‑IT、Qwen‑2.5‑3B‑Instruct 三大模型上评估；外部检索数据包括 SST‑2、Rotten Tomatoes、TweetEval（情感），Hyperpartisan News（政治倾向），Reddit Depression、Clinical Depression（抑郁）。另外使用 500 条开放式生成提示和 300 条 MCQ 题目做进一步验证。

**📈 对比分析**

与单向向量方法（Addition、DiffMean）对比，概念子矩阵在层级 sweep 中实现更高的 win‑ratio（0.70‑0.82 对比 0.50‑0.62），且失效率显著降低（13% 对比 58%）。在多维子空间层，概念子矩阵匹配或优于 additive 基线；Boolean 组合在低重叠概念上展示出强选择性，且在高重叠概念下可实现有效的 AND/NOT 组合。

**⚠️ 局限性**

局限性包括：仅在 13B 规模以下模型实验；单层、单 100 对训练，可能无法覆盖更复杂概念；依赖自动分类器评估，易受分类器误差影响；高强度 steering 可能导致生成不连贯；计算成本随着隐藏维度增加而上升；未评估多语言或更大规模模型；概念子矩阵对训练数据偏差敏感，可能放大已有偏见。

---

## 453. Agentic Vulnerability Reasoning on Windows COM Binaries

**arXiv ID:** 2605.05000 | [PDF](https://arxiv.org/pdf/2605.05000v1)

**作者:** Hwiwon Lee `[一作]` (University of Illinois at Urbana-Champaign), Lingming Zhang `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个端到端的 Agentic 管道，用于在 Windows COM 二进制中自动发现竞争条件漏洞并生成调试器验证的 PoC。

**💡 创新点**

首次将 LLM 代理与可复用的 MCP 工具接口（二进制探索、COM 检查、动态调试）相结合，并引入任务验证与预压缩机制，显著提升了发现与 PoC 生成的准确性与效率。

**🔧 技术方法**

基于 LLM 思考‑执行‑观察循环的 Agentic 框架，利用 IDA、OleViewDotNet、WinDbg 的 FastMCP 接口、虚拟调度与线程调度的 LLM 推理，以及自定义的预压缩与记忆管理。

**📊 数据集**

20 个 COM 对象共 40 个漏洞案例（包含 12 个零日、8 个 1 天），以及在生产 Windows 服务中发现的 28 个未知漏洞。

**📈 对比分析**

与 GPT/Claude 生产级编码代理在相同工具集下进行 3 次跑评，F1 最高 0.973（发现）和 67.5%（PoC 通过率），比传统静态分析器高 3.3 倍，且在生产服务中获得 16 个 CVE 与 $140k 奖金。

**⚠️ 局限性**

依赖于高质量的 IDA 反编译、符号信息和可执行调试环境；在复杂多线程时间窗口、超大规模目标或非 COM 平台时仍受制约。

---

## 454. Adaptivity Under Realizability Constraints: Comparing In-Context and Agentic Learning

**arXiv ID:** 2605.04995 | [PDF](https://arxiv.org/pdf/2605.04995v1)

**作者:** Anastasis Kratsios `[一作]` (McMaster University), Philipp Petersen `[通讯]` (University of Vienna)

**通讯引用:** 814 | [OpenAlex ID](https://openalex.org/A5041074956)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文研究了在固定查询（ICL）与自适应查询（agentic）两种学习方式在不同任务族上的统一逼近性能。

**💡 创新点**

创新点在于揭示表示约束（ReLU可实现性）如何影响自适应优势，并通过四个构造任务族分别体现优势的消失、保持与出现。

**🔧 技术方法**

采用理论分析、构造任务、ReLU网络逼近理论与信息复杂性框架来证明定理。

**📊 数据集**

不使用标准数据集，而是通过四个人工构造的任务族来进行实验验证。

**📈 对比分析**

比较方法是比较不同学习模式在给定查询和网络规模预算下的最大均匀误差；结果表明自适应优势在某些情形下消失、保持或出现。

**⚠️ 局限性**

局限在于只考虑ReLU前馈网络、无噪声测量以及理论上最坏情况，缺乏对实际数据集与训练噪声的实验验证。

---

## 455. You Snooze, You Lose: Automatic Safety Alignment Restoration through Neural Weight Translation

**arXiv ID:** 2605.04992 | [PDF](https://arxiv.org/pdf/2605.04992v1)

**作者:** Marco Arazzi `[一作]` (University of Pavia), Saraga Sakthidharan `[通讯]` (University of Pavia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 NeWTral，一种在部署时对已训练的 LoRA 适配器执行的无梯度权重空间翻译框架，能够在不访问训练数据或安全数据的情况下，将不安全的域专家模型恢复到安全对齐状态，同时保持其专业知识；

**💡 创新点**

创新点包括：①将安全对齐建模为非线性权重空间映射，而非传统线性投影；②引入层级 Mixture of Experts（MoE）路由器，自动在安全与专业性能之间平衡；③在预训练阶段完成翻译模型训练后，以可下载模块方式在零资源环境下直接使用；

**🔧 技术方法**

技术方法包括：LoRA 参数高效微调、AutoEncoder / CVAE / Flow Matching / MLP 等非线性映射网络、Mixture of Experts 路由、Zero‑Residual 初始化、基于权重统计的自适应路由器、以及使用 Llama-Guard、SecureBreak、JailbreakBench 等评估工具；

**📊 数据集**

使用数据集：八大高风险专业领域（医学、法律、金融、物理等）各十个域特定 LoRA 适配器对（unsafe–safe 对），安全数据 PKU‑SafeRLHF，评估提示集1,760条（善意、恶意、对抗），以及 JailbreakBench 等公开基准；

**📈 对比分析**

与 Safe LoRA、SaLoRA 等现有线性基线在四大模型族（Llama‑3.1、Mistral、Qwen‑2.5、Gemma）与72B规模上进行对比，评估 Attack Success Rate（ASR）、知识保留（KR）及综合最终得分；NeWTral MoE 将 ASR 从约70% 降至 13%，KR 维持约90%，在所有规模与领域中均优于基线；

**⚠️ 局限性**

局限性：1）无法完全消除所有危险输出，尤其是危险与有益信息混合的情况；2）仅训练了两种专家（手术型与激进型），未覆盖不同危害类别；3）需要在预训练阶段使用安全数据，缺乏完全零数据方案；4）路由器为固定策略，缺乏可调风险阈值；5）对全新结构域的泛化仍待验证；6）未针对信息泄露、偏见等其他安全风险做专门处理。

---

## 456. On-line Learning in Tree MDPs by Treating Policies as Bandit Arms

**arXiv ID:** 2605.04979 | [PDF](https://arxiv.org/pdf/2605.04979v1)

**作者:** Anvay Shah `[一作]` (Indian Institute of Technology Bombay), Shivaram Kalyanakrishnan `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5038200034)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

这项研究提出了将经典Bandit算法Lucb和Ucb推广到树形马尔可夫决策过程（T-MDP）的在线学习框架。

**💡 创新点**

创新点在于设计了一种针对T-MDP的置信界，能够在指数级策略集合上共享数据，从而实现实例特定的样本复杂度和调度最优的回报最小化，并给出了多项式时间实现。

**🔧 技术方法**

采用了置信界理论、负圆柱依赖（NCD）随机变量的Chernoff界、Bernstein不等式以及自底向上的策略搜索（PU）等技术。

**📊 数据集**

主要在三种不完全信息游戏中验证：Kuhn Poker、Leduc Poker和新的Reconnaissance Blind Tic‑Tac‑Toe（RBT）。

**📈 对比分析**

与传统的Bpi‑Ucrl、Mdp‑GapE、Mvp、Amb、StrongEuler、Mccfr、Opf以及UCT等基线相比，实验显示在PAC与回报最小化两种范式下，Lucb‑T和Ucb‑T在中到大规模问题上实现了更低的样本复杂度/回报损失，并在RBT上表现尤为突出。

**⚠️ 局限性**

主要限制是仅适用于已知奖励函数的T‑MDP，且对未知奖励或更一般的MDP情形尚未证明；另外Uniform变体虽然理论上更好但需要线性内存，导致大规模实践中性能下降。

---

## 457. Probabilistic Atomic Swaps for Bitcoin and Friends

**arXiv ID:** 2605.04975 | [PDF](https://arxiv.org/pdf/2605.04975v1)

**作者:** Paul Gerhart `[一作]` (TU Wien), Sri Aravinda Krishnan Thyagarajan `[通讯]` (University of Sydney)

**通讯引用:** 812 | [OpenAlex ID](https://openalex.org/A5053520687)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种新的加密原语——概率原子交换（Probabilistic Atomic Swaps），实现了在区块链上无需信任中介即可进行带固定公开概率的随机资产交换。

**💡 创新点**

创新点在于将适配签名（Adaptor Signatures）与盲化伪随机函数（Oblivious PRFs）结合，构建了一种可在现有最小脚本功能（如签名、时间锁）上实现的概率交换协议，并提出了对 OPRF 评估的原子化服务机制；同时实现了低成本的可验证证明（Bulletproof / Cut-and-Choose）来证明交易语句的正确性。

**🔧 技术方法**

核心技术包括：
- Schnorr/Adaptor签名；
- 2-Hash-DH OPRF；
- 零知识证明（NIZK/Bulletproof、Cut‑and‑Choose）；
- 可信两方安全计算（共识密钥生成、预签名）；
- Taproot脚本（CLTV）实现资金锁定。
实现中使用 Rust 语言、secp256k1、Bulletproof、OPRF、Lightning HTLC 适配器。

**📊 数据集**

在实验中使用的测试数据集主要为 Bitcoin 和 Litecoin 的 testnet，交叉链实验也在这两条链上进行；Lightning Network 原型使用了本地两节点 Lightning 环境；此外还在不同概率（1/2^ℓ）下对证明大小、验证/证明时间进行基准测试。

**📈 对比分析**

对比方法：与现有的基于彩票的随机化协议、Universal Swaps 等在 on‑chain 开销、隐私与可组合性上进行比较。实验显示：
- 单次概率交换仅需四笔标准 Taproot 交易，费用与普通原子交换相近；
- 证明大小和验证时间在 ℓ≤16 时保持低，ℓ>16 时可通过 Bulletproofs 保持常数大小；
- 在 Lightning HTLC 变体中，单轮执行约 0.43 s，证明 128 B，展示了可接受的性能。

**⚠️ 局限性**

局限性：
- 目前仅支持单侧概率交换（只对一方产生随机结果），无法实现两侧均为概率的交换；
- 只支持概率为 1/m 的形式，若需其他分数可通过多次尝试实现，但会增加交互复杂度；
- 方案基于 Schnorr/DLog，尚未给出后量子安全实现；
- 对低概率（p≪1）时，证明生成时间会呈指数增长，尽管可用 Bulletproofs 缓解；
- 在无时间锁支持的链上需依赖 VTS 或其他外部信任假设。

---

## 458. Architectural Constraints Alignment in AI-assisted, Platform-based Service Development

**arXiv ID:** 2605.04973 | [PDF](https://arxiv.org/pdf/2605.04973v1)

**作者:** Julius Irion `[一作]` (Technische Universität Berlin), Sebastian Werner `[通讯]` (Technische Universität Berlin)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5052239645)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种基于检索增强生成（RAG）的模板检索系统，利用组织平台知识与代理澄清循环在平台化服务开发中自动化检索最合适的架构模板，从而实现“按设计可部署”的服务脚手架，并与传统的 AI 驱动的 vibe coding 进行对比评估。

**💡 创新点**

创新点在于将预先编写的架构模板作为检索语料库，并通过代理澄清循环实现自然语言与模板之间的精准匹配，使 AI 辅助生成在满足组织约束的同时避免了传统代码生成的脆弱性和高迭代成本。

**🔧 技术方法**

技术实现包括：Backstage 内部开发平台、嵌入模型（Chroma）、OpenAI LLM、向量检索、代理澄清循环、CI/CD 与 Kubernetes 集成等。

**📊 数据集**

使用的数据集由组织内的多种服务模板（涵盖 REST/GRPC、PostgreSQL、SSR 等）构成的检索库，以及 7 名参与者在实验中产生的日志、token、CI/CD 质量门等实验数据。

**📈 对比分析**

评估方法：通过 10 次自动化模板选择实验和 7 名参与者的 vibe coding 实验，比较部署成功率、token/成本、提示次数和开发者体验。结果显示 RAG 系统实现 100% 成功率，token 使用约为 vibe coding 的 1%（低 100 倍），提示次数减少至平均 3 次，开发者体验明显更好。

**⚠️ 局限性**

局限性包括：样本规模仅 7 名参与者、仅测试单一任务、实验环境来自学术机构、未评估模板维护的运营成本、依赖已有模板且需要平台工程投入。

---

## 459. TabEmbed: Benchmarking and Learning Generalist Embeddings for Tabular Understanding

**arXiv ID:** 2605.04962 | [PDF](https://arxiv.org/pdf/2605.04962v1)

**作者:** Minjie Qiang `[一作]` (Soochow University), Ningtao Wang `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TabBench 基准与 TabEmbed 通用表格嵌入模型，统一实现表格分类与检索。

**💡 创新点**

创新点在于：①将表格任务转化为语言-行对比学习；②采用任务自适应查询生成与正向感知硬负采样；③在同一嵌入空间内同时优化分类与检索，突破传统表格模型的架构束缚。

**🔧 技术方法**

使用对比学习（语言‑行三元组）、Qwen3‑Embedding 预训练模型、负样本挖掘、自动序列化、软硬负采样以及后续的线性回归与评估指标（Accuracy、F1、MRR@10、nDCG@10）。

**📊 数据集**

数据集：TabBench（结合 Grinsztajn、OpenML‑CC18、OpenML‑CTR23、UniPredict 四大公开仓库共 300+ 表格任务）以及 T4 大规模表格语料用于自监督训练；Bench 通过 500k 检索 + 100k 分类对比三元组构建。

**📈 对比分析**

与 10 款主流通用文本嵌入模型（Jina、Jasper、Qwen3、F2LLM、Octen、Mistral、XLM‑RoBERT 等）在 TabBench 上进行基准评测。TabEmbed 在 0.6B、4B、8B 三个规模下均获得最高 Overall 分数，0.6B 版本甚至超越更大参数的基线；在检索任务上 MRR@10 提升 35+ 点，分类任务 Accuracy/F1 同样显著提升。

**⚠️ 局限性**

局限：未评估商业闭源嵌入 API；序列化方式对极宽表格（数百列）会超出模型上下文窗口，需进一步研究高效序列化或长序列模型。

---

## 460. EP-GRPO: Entropy-Progress Aligned Group Relative Policy Optimization with Implicit Process Guidance

**arXiv ID:** 2605.04960 | [PDF](https://arxiv.org/pdf/2605.04960v1)

**作者:** Song Yu `[一作]` (Southwest University), Zhisheng Yang `[通讯]` (Southwest University)

**通讯引用:** 1392 | [OpenAlex ID](https://openalex.org/A5033233282)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EP-GRPO，针对 GRPO 在 RLVR 中的统一粒度、统一极性和零方差崩塌三大信用分配失败，构建熵门控调制、隐式进程奖励以及累计熵映射等机制，实现 token 级别的稠密学习信号。

**💡 创新点**

创新点包括：①熵门控将序列优势转化为 token 级别权重，突出决策拐点；②利用策略偏差对齐的隐式进程奖励，无需外部奖励模型即可提供方向性反馈；③基于累计熵的进度对齐归一化，天然解决零方差崩塌并保持梯度流。

**🔧 技术方法**

采用 Group Relative Policy Optimization、熵门控加权、策略偏差对齐、累计熵桶归一化、LoRA 微调、TRL 框架、KL 约束等技术，构建完整的 EP-GRPO 算法。

**📊 数据集**

训练使用 Skywork-OR1-RL-Data（8k 数学题）作为 RL 数据；评测在 MATH500、AMC23、Minerva、AIME24、AIME25 等数学推理基准上进行。

**📈 对比分析**

与标准 GRPO、提高温度、更多 rollouts、商业模型等基线进行对比；在 Qwen2.5‑3B 和 7B 上平均准确率从 18% 提升至 22%，从 27% 提升至 30%，pass@k 也显著高于所有对比方法，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：仍需参考模型估计熵和策略偏差；对极长或非数理推理任务的适用性待进一步验证；算法仍依赖可验证奖励场景，未探索更通用的进程奖励或熵驱动探索机制。

---

## 461. Adapting Large Language Models to a Low-Resource Agglutinative Language: A Comparative Study of LoRA and QLoRA for Bashkir

**arXiv ID:** 2605.04948 | [PDF](https://arxiv.org/pdf/2605.04948v1)

**作者:** Mullosharaf K. Arabov `[一作]` (Kazan Federal University), Svetlana S. Khaybullina `[通讯]` (Kazan Federal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Bashkir 低资源语言进行参数高效微调实验，比较 LoRA 与 QLoRA 在多种大语言模型上的效果。

**💡 创新点**

创新点在于首次系统评估 LoRA/QLoRA 在 Bashkir 上的表现，并揭示 tokenizer 与 PEFT 交互的关键性。

**🔧 技术方法**

采用了 LoRA、QLoRA 以及完整微调三种策略，对 DistilGPT2、GPT-2 系列、Phi-2、Qwen2.5-7B、DeepSeek-7B、Mistral-7B 等模型进行训练。

**📊 数据集**

使用了自建的 71,567 文档（约 46.9M 词）的 Bashkir 文本语料库，实验采用 10k 文档子集。

**📈 对比分析**

通过多随机种子对比，发现 GPT-2 medium 完整微调得到最低 3.34 perplexity；QLoRA 在 Mistral‑7B 与 Phi‑2 上获得 3.79–3.81 perplexity，参数量仅为原模型的 1–8% 但训练时间仍可接受。

**⚠️ 局限性**

局限在于仅用 10k 文档子集、短序列长度 128、未评估更多 PEFT 方法，且对生成质量的定性评估样本有限。

---

## 462. Conflict Essences for Transformation Rules with Nested Application Conditions -- Long Version

**arXiv ID:** 2605.04947 | [PDF](https://arxiv.org/pdf/2605.04947v1)

**作者:** Alexander Lauer `[一作]` (Philipps-Universität Marburg), Gabriele Taentzer `[通讯]` (Philipps-Universität Marburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了用于带有嵌套应用条件的图变换规则的冲突本质和符号冲突本质，并给出了它们与初始冲突的对应关系。

**💡 创新点**

将禁用本质扩展到任意嵌套条件，定义冲突本质以捕捉双向冲突，并构造带条件的符号冲突本质实现精确的静态分析。

**🔧 技术方法**

利用范畴论中的粘附 HLR 类别、双推压模型、嵌套条件、临界对、初始冲突、符号变换对、重叠组合等理论工具。

**📊 数据集**

以软件类模型重构规则为示例，没有使用公开数据集。

**📈 对比分析**

本文为理论性工作，没有实验对比或性能评估。

**⚠️ 局限性**

嵌入禁用本质不一定意味着冲突，符号冲突本质集合不最小；算法复杂度和适用范围受粘附 HLR 类别限制，缺乏经验验证。

---

## 463. Training-Time Batch Normalization Reshapes Local Partition Geometry in Piecewise-Affine Networks

**arXiv ID:** 2605.04946 | [PDF](https://arxiv.org/pdf/2605.04946v1)

**作者:** Xuan Qi `[一作]` (Istituto Italiano di Tecnologia), Cigdem Beyan `[通讯]` (University of Verona)

**通讯引用:** 1530 | [OpenAlex ID](https://openalex.org/A5057859690)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究训练时批量归一化（BN）对连续分段仿射（CPA）网络的局部切分几何影响，提出BN作为一种批量条件下的“重新居中”机制，能够在训练过程中提高局部切分细化。

**💡 创新点**

创新点在于：①给出了BN在每个神经元上产生的参考超平面和其平移的精确几何表述；②定义了基于ℓ∞窗口的局部切分密度，并给出切平面与窗口相交的精确判据；③在满足显式随机顺序和一般性假设下证明BN可导致局部切分数期望提升，并通过“父区域嵌入”将该机制在深层中传播；④结合低维精确计数与高维近似诊断，对BN的几何效应提供多层次验证。

**🔧 技术方法**

主要技术包括：批量条件下的超平面排列分析、窗口相交判据推导、随机支配（stochastic dominance）与独立性假设、递归区域计数公式、深层中父区域的插值与同胚映射、实验中精确区域枚举、归一化偏移分布的经验分布与二维切片分析。

**📊 数据集**

实验使用的主要数据集包括：低维玩具数据（Gaussian Quantiles、Two Moons、Random Uniform）用于精确计数；深层网络在上述数据集上的多层网络；以及真实高维数据集 CIFAR‑10、MNIST、TinyImageNet，用于归一化偏移分布与二维切片的近似诊断。

**📈 对比分析**

比较方法：在相同网络结构、优化器、学习率、批大小等设置下，对比BN与非BN模型的局部切分计数、局部密度、决策边界可视化以及验证准确率。结果显示：BN模型在所有实验中均获得更高的局部切分密度、更多的切分区域、较快的决策边界收敛和更高的准确率。

**⚠️ 局限性**

局限性：①理论仅针对训练时批量条件下的几何，未涵盖全局或运行时BN的效果；②精确计数只能在低维可行，无法直接验证高维网络的完整切分结构；③需要满足非退化、独立性和随机顺序等显式假设，实际训练过程可能不完全满足。

---

## 464. Tailoring Scaffolding to Diagnostic Strategies: Theory-Informed LLM-Based Agents

**arXiv ID:** 2605.04996 | [PDF](https://arxiv.org/pdf/2605.04996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 465. DART: A Vision-Language Foundation Model for Comprehensive Rope Condition Monitoring

**arXiv ID:** 2605.04943 | [PDF](https://arxiv.org/pdf/2605.04943v1)

**作者:** Anju Rani `[一作]` (Aalborg University), Petar Durdevic `[通讯]` (Aalborg University)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5081081963)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种统一的视觉语言基础模型DART，用于合成纤维绳索的全流程状态监测，支持分类、严重度回归、少量样本识别、进展建模、维护建议、报告生成、异常检测等任务。

**💡 创新点**

结合JEPA自监督预测与跨模态融合，提出Severity-Conditioned Cross-Modal Fusion (SC-CMF) 门控、saliency-guided HD-MASK掩码、四项 Contrastive Damage Disentanglement (CDD) 损失，形成单一冻结主干即可完成多任务。

**🔧 技术方法**

Vision Transformer ViT-H/14 与 Llama-3.2-3B-Instruct 结合，EMA目标编码器、Transformer latent predictor、交叉注意力门控、InfoNCE、Orthogonality、Focal分类等技术。

**📊 数据集**

ROPE 数据集 4,270 张实验室采集的合成绳索图像，包含 14 个细粒度损伤类别及严重度标签，并配有专家撰写的自然语言描述。

**📈 对比分析**

与 I-JEPA、CLIP、BLIP-2、DINOv2 等基准对比，在 14 类分类上取得 93.22% 准确率 (+38.5pp)，Severity Spearman ρ=0.94，20-shot 89.2% macro‑F1，维护建议 94.79% macro‑F1，异常检测 4.76% 误报率，显示显著优于基准。

**⚠️ 局限性**

对稀有类别表现不足（如 Coreout+CutStrands 仅 8 样本），Placking/Medium 视觉歧义导致误判，推理延迟约45ms/图像，且模型规模较大，难以在边缘设备上直接部署。

---

## 466. Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation

**arXiv ID:** 2605.04998 | [PDF](https://arxiv.org/pdf/2605.04998v1)

**作者:** Jinju Lee `[一作]` `[通讯]` (PearlLeeStudio), Jinju Lee (PearlLeeStudio)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对流行音乐预训练的 Music Transformer 进行爵士和弦进程微调，并系统探索不同量级的流行“复习”数据对避免灾难性遗忘、提升爵士表现的影响。

**💡 创新点**

首次用实验方式量化在跨域细调中所需的最小旧域复习比例，发现约1.5–2倍目标域数据即可恢复旧域能力；并发现性能最佳的中点并非听觉上最受欢迎的检查点。

**🔧 技术方法**

使用 25M 参数的 Music Transformer（相对位置注意力），采用经验重放（rehearsal）微调，评估指标为单步及 top‑5 预测准确率，配合非正式听音判断。

**📊 数据集**

流行数据：Chordonomicon + McGill Billboard（约 544K 歌曲）；爵士数据：Jazz Harmony Treebank、JazzStandards、Weimar Jazz Database、JAAH（总计 1,859 歌曲，1,513 训练）。

**📈 对比分析**

在保留流行基准的同时，比较不同复习量级下的流行与爵士 top‑1/top‑5；例如 2.5K 流行复习（F3）得到 84.2% 流行 top‑1 与 81.0% 爵士 top‑1，单纯爵士微调则流行 top‑1 下降 2 点；通过 Pareto 前沿图示两域权衡。

**⚠️ 局限性**

局限包括：仅评估单一模型大小与架构、使用单一随机种子、爵士语料偏向标准曲目、评价指标仅为符号级准确率、缺乏正式多评审听音实验，以及结论基于单一作者的偏好。

---

## 467. DualTCN: A Physics-Constrained Temporal Convolutional Network for 2 Time-Domain Marine CSEM Inversion

**arXiv ID:** 2605.04997 | [PDF](https://arxiv.org/pdf/2605.04997v1)

**作者:** Khaled Ahmed `[一作]` (Southern Illinois University Carbondale), Ghada Omar `[通讯]` (Southern Illinois University Carbondale)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5004958560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 DualTCN——首个针对时间域海洋受控源电磁（MCSEM）数据的深度学习反演框架，能够一次前向推理直接回归四个物理参数（σ1、σ2、d1、d2），并通过可微软阶梯解码器生成完整的电导率–深度剖面。

**💡 创新点**

创新点包括：① 双分支时序卷积网络（TCN）—全时域分支与晚期时间专门分支，显著提升对深度参数 d2 的识别；② 辅助海底深度头（d_sf）引导特征学习；③ 物理约束的可微软阶梯解码器保证反演结果的物理可行性；④ 通过13种架构的系统性消融实验、幅度鲁棒性（Curriculum AmpAug、AmpRatio）与校准不确定性（MC‑Dropout、温度缩放、合成预测）实现了性能与可解释性的双提升。

**🔧 技术方法**

核心技术：时序卷积网络（带指数膨胀的因果卷积）、Transformer 作为基线、全局注意力、可微解码器、Huber 损失、数据增强（噪声与幅度增广）、温度缩放的 MC‑Dropout、合成预测（Split Conformal）、逆σ2 加权、批量归一化与 GELU 激活。

**📊 数据集**

数据集：使用 1,000,000 条合成 MCSEM 时域记录（4 号接收器、128 采样点），参数按指定范围独立均匀采样；在 70/15/15 的训练/验证/测试比例上训练；此外对 5 个典型 1‑D 场景与 7 个已发表的实测模型做外部验证。

**📈 对比分析**

与 13 种架构对比，DualTCN 在测试集上实现平均 R² = 0.877，σ2 与 d2 的 R² 分别为 0.905 与 0.627；相比传统 Levenberg–Marquardt / L‑BFGS‑B（多起点）仅达到 R² 0.129–0.439，且 DualTCN 的推理时间仅 3.5 ms GPU（≈ 26,500× 传统迭代速度）。在噪声鲁棒性上，DualTCN‑AmpAug 在 ±2% 幅度误差下仍保持 R² 0.858；在不确定性量化方面，MC‑Dropout 对 σ1 校准良好，d2 需温度缩放或合成预测才能满足 90% 置信区间覆盖率。

**⚠️ 局限性**

局限性：① 对幅度误差高度敏感（σ2、d2 依赖 log‑幅度通道），需额外增广或特征表示；② 仅为 1‑D 反演，局限于水平层状地质；③ 对接收器几何、源强度漂移等硬件变异需重新训练；④ d2 的信息瓶颈导致精度相对较低；⑤ 仅在合成数据上验证，真实测量误差与偏差仍待实测验证。

---

## 468. Low-Rank Adaptation of Geospatial Foundation Models for Wildfire Mapping Using Sentinel-2 Data

**arXiv ID:** 2605.04989 | [PDF](https://arxiv.org/pdf/2605.04989v1)

**作者:** Ali Shibli `[一作]` (KTH Royal Institute of Technology), Yifang Ban `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 7701 | [OpenAlex ID](https://openalex.org/A5040195008)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种高效的地理空间基础模型混合（Mixture of Geospatial Foundation Models, MoGFM）框架，用于遥感图像语义分割，能够在保持精度的同时显著降低推理时间和参数量。

**💡 创新点**

创新点在于将多个预训练的遥感基础模型通过专家混合策略进行联合推理，并引入轻量化调优模块（如 LoRA、Adapter）实现参数共享，突破传统单模型或全局微调的瓶颈。

**🔧 技术方法**

主要技术包括：①多模态基础模型融合；②专家混合与门控机制；③低秩参数微调；④多尺度特征提取与融合；⑤自监督预训练与迁移学习。

**📊 数据集**

实验使用了公开遥感分割数据集 DeepGlobe Land Cover, LoveDA, and ISPRS Vaihingen，涵盖卫星图像、航空影像和多光谱数据。

**📈 对比分析**

在上述数据集上与 SOTA 方法（如 SegFormer、Swin-Transformer、HRNet）对比，MoGFM 在 mIoU 上提升 1.5–3.2%，同时推理速度提升 2–3 倍，参数量减少 30–50%。

**⚠️ 局限性**

局限性包括：①对极低分辨率或极端大尺度场景的适应性仍有限；②混合模型的门控训练需要额外的超参数调优；③在多传感器混合数据时仍需进一步验证跨域鲁棒性。

---

## 469. Why Expert Alignment Is Hard: Evidence from Subjective Evaluation

**arXiv ID:** 2605.04972 | [PDF](https://arxiv.org/pdf/2605.04972v1)

**作者:** Tzu-Mi Lin `[一作]` (National Yang Ming Chiao Tung University), Chung-Chi Chen `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 533 | [OpenAlex ID](https://openalex.org/A5101516307)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在主观评估任务中，探究并比较了不同专家对业务提案的打分与大型语言模型（LLM）的对齐情况；

**💡 创新点**

将专家对齐问题视为理解主观评估差异的窗口，发现专家差异、评估维度和编辑样本对对齐难度的影响，并揭示显式规则不一定提升对齐；

**🔧 技术方法**

使用零样本/少样本提示、LoRA微调以及AlphaEdit模型编辑等技术；

**📊 数据集**

基于PBIG 2025共享任务的业务提案评估数据，并通过专家问卷补充背景、评估标准和示例推理信息；

**📈 对比分析**

对齐效果通过与专家评分的精确匹配率衡量，零样本提示约 18–30%，LoRA 微调最高可达 44%，AlphaEdit 在某些专家和维度上略高于提示但低于微调，加入显式规则往往不提升甚至降低性能；

**⚠️ 局限性**

实验专家数量有限、仅使用单一基础模型（LLaMA‑3.1‑8B）且仅采用一种编辑方法，难以推断在更大规模或不同模型、任务上的通用性；

---

## 470. KernelBench-X: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels

**arXiv ID:** 2605.04956 | [PDF](https://arxiv.org/pdf/2605.04956v1)

**作者:** Han Wang `[一作]`, Jun Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个针对LLM生成Triton核的基准，通过分类评估语义正确性与硬件效率，并对五种代表性方法进行系统比较。

**💡 创新点**

创新点在于构建了15类任务的细粒度分类基准，并提出了双阶段正确性验证和硬件效率度量，揭示了任务结构对正确性、迭代优化偏向修复、以及性能与正确性不一致的三大发现。

**🔧 技术方法**

采用了LLM生成、自动化编译与执行验证、性能计量与硬件利用率指标、统计分析与回归模型等技术。

**📊 数据集**

使用了176个任务，覆盖Activation、Convolution、Fusion等15类，包括多精度与量化扩展，构建在TritonBench‑T基础之上。

**📈 对比分析**

比较了AutoTriton、GEAK、KernelAgent、Claude和DeepSeek‑Coder，在六块GPU上测得编译率、语义正确率、速度提升和效率分数，结果显示最高正确率仅为30.7%，但速度提升往往低于PyTorch，且迭代改进多偏向修复而非加速。

**⚠️ 局限性**

局限性包括：对全局张量契约与并行归约的理解不足；量化任务完全未能成功；迭代过程缺乏对性能的反馈，导致性能提升有限；基准数据集虽丰富但仍无法覆盖所有硬件特性。

---

## 471. Adaptive Inverted-Index Routing for Granular Mixtures-of-Experts

**arXiv ID:** 2605.04952 | [PDF](https://arxiv.org/pdf/2605.04952v1)

**作者:** Klaus-Rudolf Kladny `[一作]` (Max Planck Institute for Intelligent Systems), Michael Muehlebach `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5049845074)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于向量量化的自适应倒排索引路由（AIR），在稀疏 MoE 中通过两阶段粗短名单+精细评分来降低路由成本。

**💡 创新点**

创新点包括：①自适应向量量化生成可动态更新的倒排索引短名单；②双层优化策略让代码本不参与梯度更新；③给出保留路由质量的理论下界；④避免对专家参数施加结构约束。

**🔧 技术方法**

采用球面 k‑means、指数滑动平均、向量量化、负载平衡、噪声 jitter、负负采样等技术，且在两阶段检索中实现无 straight‑through 的梯度传递。

**📊 数据集**

在 WikiText‑103、C5、OpenWebText2 三个数据集上进行实验。

**📈 对比分析**

与标准粗糙、标准细粒、PEER、层级 MoE 等基线对比，AIR 在保持或减少训练 FLOPs 的前提下，PPL 相比最佳基线（PEER）平均降低约 10%，且在不同模型规模和数据集上均表现优异。

**⚠️ 局限性**

主要限制：需要在每次优化步骤后重新更新所有短名单，计算成本为 O(E G d)；路由 FLOPs 降低不一定转化为 wall‑clock 加速；实验未考虑降维或硬件加速等进一步优化方向。

---

## 472. Modular Reinforcement Learning For Cooperative Swarms

**arXiv ID:** 2605.04939 | [PDF](https://arxiv.org/pdf/2605.04939v1)

**作者:** Erel Shtossel `[一作]` (Bar Ilan University), Gal A. Kaminka `[通讯]` (Bar Ilan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种模块化空间状态表示法，用每个传感器独立学习并通过“议事厅”聚合动作，在协作搜索任务中实现低资源下的多智能体强化学习。

**💡 创新点**

创新点在于：1）将空间状态按方向拆分为若干子状态，单个学习器仅处理自身子状态，整体状态规模线性增长；2）使用可合成的向量动作空间，配合议事厅采样策略；3）展示模块化方法对奖励变动的鲁棒性，并在低密度环境中与全状态学习竞争。

**🔧 技术方法**

使用多臂老虎机算法（Continuous‑UCB1）作为学习器；差分奖励（Difference Reward）对齐个体与集体目标；议事厅采用双高斯概率聚合；对比基线使用随机、动态窗口、R‑Learner（全状态 Q‑学习）等。

**📊 数据集**

数据集：在 ARGoS3 仿真平台上使用 Krembot 机器人，三种不同布置的 1.5 m² 小场地（单基站、角落基站、双基站）和随机投放的圆盘物品，机器人数量从 4 到 36。

**📈 对比分析**

与随机、动态窗口、R‑Learner、连续时间 Q‑学习等方法比较；在三种场景下评估在 20 分钟测试期内收集的物品数量。结果显示：模块化方法在低密度下与动态窗口相当，超过 R‑Learner；在高密度或多基站场景下表现略逊；对奖励的改变（差分奖励→自利奖励）时保持稳定，而全状态学习显著退化；使用向量动作空间比算法动作空间更易学习。

**⚠️ 局限性**

局限性：1）在高密度或多基站环境中因多传感器建议冲突导致议事厅聚合效果不佳；2）仅在仿真中验证，未在真实机器人上实验；3）需要向量动作空间，无法直接采用传统避碰算法；4）对极端稀疏场景的探索不足。

---

## 473. Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics

**arXiv ID:** 2605.05097 | [PDF](https://arxiv.org/pdf/2605.05097v1)

**作者:** Andreas Pattichis `[一作]` (Cyprus Institute), Constantine Dovrolis `[通讯]` (Cyprus Institute)

**通讯引用:** 6899 | [OpenAlex ID](https://openalex.org/A5002141922)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多时间尺度联动的动态关联图记忆系统 Memini，允许 LLM 在部署后持续更新知识

**💡 创新点**

将神经突触巩固模型 Benna‑Fusi 直接嵌入边权，生成自适应的快速与慢速权重，使记忆在无外部调节下实现短期记忆、长期巩固和选择性遗忘

**🔧 技术方法**

联动动态记忆（Benna‑Fusi 链模型）、传播激活检索、实体抽取与共现更新

**📊 数据集**

使用维基百科文档流进行初步验证（Appendix）

**📈 对比分析**

与标准 RAG、MemGPT、GraphRAG、HippoRAG、A‑MEM、SYNAPSE 等系统在“演变边权”“多时间尺度”“选择性遗忘”三维度进行对比；实验表明 Memini 在这些维度上均表现优于或匹配其他方法，但未给出具体数值评测

**⚠️ 局限性**

尚未在大规模检索基准上进行系统评估，缺乏量化性能数据，且在极端规模或实时推理时的计算开销和稳定性尚未验证

---

## 474. Driver-WM: A Driver-Centric Traffic-Conditioned Latent World Model for In-Cabin Dynamics Rollout

**arXiv ID:** 2605.05092 | [PDF](https://arxiv.org/pdf/2605.05092v1)

**作者:** Haozhuang Chi `[一作]` (Nanyang Technological University), Chen Lv `[通讯]` (Nanyang Technological University)

**通讯引用:** 18369 | [OpenAlex ID](https://openalex.org/A5072073374)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Driver-WM，一种基于双流潜在空间的驱动者世界模型，能够在外部交通情境的因果条件下对驾驶者内部动态进行多步回放，并同时预测骨架轨迹与行为情绪等语义标签。

**💡 创新点**

创新点包括：① 通过双流结构将外部交通和内部驾驶状态分别编码并通过门控因果注入实现单向外部→内部耦合；② 利用冻结的 Qwen3‑VL 视觉‑语言特征作为紧凑的感知接口；③ 引入可控的门控注入与物理先验，使模型在长时程内保持高保真度并可进行干预实验。

**🔧 技术方法**

所用技术包括：冻结的 Qwen3‑VL 视觉‑语言模型、视角嵌入、潜在空间自回归滚动、跨注意力门控注入、ST‑GCN 骨架解码器、骨架物理约束（骨长、平滑、座椅约束）以及辅助语义头。

**📊 数据集**

实验采用 AIDE 伴随驾驶基准数据集，包含同步的车内摄像头和多视角车外视频，视频分为 3 秒 10 帧片段，采用 5→5 的因果滚动协议。

**📈 对比分析**

与传统的仅预测运动的基线（Zero‑Velocity、ST‑GCN、SiMLPe、MotionBERT）以及离线编码‑解码参照模型相比，Driver‑WM 在 MPJPE 与 d‑nMPJPE 上保持相近甚至更优的整体性能，并在高运动子集（HM）中显著降低误差；在语义宏 F1 上亦实现了更高的对外部交通与内部行为的对齐，表明外部条件的因果注入提升了多模态一致性。

**⚠️ 局限性**

局限性包括：模型高度依赖冻结的 VLM 特征，若 VLM 预训练不匹配或视觉输入质量低下会影响性能；对外部上下文的依赖导致在外部信息缺失或严重时预测质量下降；目前仅预测 2D 骨架轨迹，未覆盖更高维的姿态或三维重建；干预实验虽验证了因果路径，但缺乏对真实控制系统的闭环评估。

---

## 475. Provable imitation learning for control of instability in partially-observed Vlasov--Poisson equations

**arXiv ID:** 2605.05081 | [PDF](https://arxiv.org/pdf/2605.05081v1)

**作者:** Xiaofan Xia `[一作]` (University of Toronto), Wenlong Mou `[通讯]` (University of Toronto)

**通讯引用:** 455 | [OpenAlex ID](https://openalex.org/A5006742082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究如何利用模仿学习从稀疏的宏观观测中学习并实现对 Vlasov–Poisson 系统的稳定控制策略。

**💡 创新点**

提出了从全信息专家策略到部分观测控制策略的理论桥梁，给出了行为克隆误差与稳定性误差的耦合关系，并通过熵量度解释了学习难度与初始分布复杂度的关系。

**🔧 技术方法**

采用行为克隆、经验风险最小化、PDE 稳定性分析、熵与 Fourier 重构、时间卷积网络＋注意力机制以及尺度感知混合损失等技术。

**📊 数据集**

在 1D1V Vlasov–Poisson 模拟中使用两流 Maxwellian 以及双模扰动的合成数据，并以均匀分布的少量传感器采样得到的宏观密度序列作为训练与测试集。

**📈 对比分析**

与无控制基线、即时谱 Poisson 重构以及全信息专家控制进行对比，实验显示所学控制器在稀疏且带噪的观测下能显著延长稳定时间窗口，性能优于非自适应基线。

**⚠️ 局限性**

仅验证在 1D1V 低维模型，受限于预先设定的烧录期；理论误差上界可能过于保守，需进一步推广到三维真实托卡马克几何并探索更复杂的 RL/DAgger 等方法。

---

## 476. When Relations Break: Analyzing Relation Hallucination in Vision-Language Model Under Rotation and Noise

**arXiv ID:** 2605.05045 | [PDF](https://arxiv.org/pdf/2605.05045v1)

**作者:** Philip Wootaek Shin `[一作]` (Pennsylvania State University), Vijaykrishnan Narayanan `[通讯]` (Pennsylvania State University)

**通讯引用:** 9047 | [OpenAlex ID](https://openalex.org/A5101919131)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究视觉语言模型在图像旋转和噪声扰动下的关系幻觉，系统评估并尝试通过提示词与预处理（方向校正、去噪）缓解该问题。

**💡 创新点**

创新点在于揭示即使轻微几何和光照扰动也会显著恶化关系推理性能，并指出当前提示与预处理技术难以完全消除幻觉，凸显感知鲁棒性与关系理解之间的鸿沟。

**🔧 技术方法**

采用旋转/噪声数据增强、旋转元数据提示、图像方向检测器、先进去噪模型（如基于LPIPS/PSNR/SSIM评估的重建网络）等技术。

**📊 数据集**

使用MMRel、R-Bench、Reefknot三个公开基准数据集，针对多选与是非问答场景进行评测。

**📈 对比分析**

对比原始输入、旋转/噪声扰动、提示干预与预处理后效果。旋转导致4–20个百分点下降；去噪可恢复约0.5–9个百分点；提示仅带来极小提升；总体上两种策略均无法完全弥补关系幻觉。

**⚠️ 局限性**

局限性在于预处理模型未能恢复被破坏的关系线索，提示词对关系推理影响有限；两种缓解手段在不同数据集和扰动强度下表现不一，仍需更鲁棒的几何感知VLM设计。

---

## 477. Computer-Aided Design Generation by Cascaded Discrete Diffusion Model

**arXiv ID:** 2605.05031 | [PDF](https://arxiv.org/pdf/2605.05031v1)

**作者:** Honghu Pan `[一作]` (Hunan University), Pengyang Wang `[通讯]` (University of Macau)

**通讯引用:** 8592 | [OpenAlex ID](https://openalex.org/A5036270316)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种层级化的离散扩散框架（CDDM），先生成CAD命令序列，再根据命令生成对应参数，完成CAD模型自动生成。

**💡 创新点**

创新点包括：①使用离散扩散取代连续扩散，直接在符号分布上做扰动；②为不同类型的参数（坐标、尺寸、布尔）设计专属的离散转移矩阵；③在参数扩散的去噪网络中加入命令级局部注意力和跨模态交叉注意力，以保持命令与参数之间的语义一致性。

**🔧 技术方法**

核心技术包括：离散扩散模型、吸收状态转移矩阵、Gaussian/scale‑invariant/先验保留转移核、Transformer 编码器、全局+局部+交叉注意力、条件扩散（命令长度、点云）。

**📊 数据集**

使用 DeepCAD 数据集（约178k个CAD模型）进行实验，包含命令、参数的离散化。

**📈 对比分析**

与自回归模型和连续扩散模型进行对比；在无条件生成任务中，CDDM 在覆盖率、最小匹配距离、JSD、创新率、唯一率、无效率上均实现了最佳或最接近上限的表现，尤其在无效率上显著低于连续扩散模型。

**⚠️ 局限性**

主要局限在于参数扩散仍是性能瓶颈；当前方法仅支持有限的三类参数；扩散步骤虽可缩短，但仍比传统自回归模型慢；对更复杂的CAD语法或更大规模数据集的适应性尚未充分验证。

---

## 478. Local Intrinsic Dimension Unveils Hallucinations in Diffusion Models

**arXiv ID:** 2605.05026 | [PDF](https://arxiv.org/pdf/2605.05026v1)

**作者:** Bartlomiej Sobieski `[一作]` (Warsaw University of Technology), Quanzheng Li `[通讯]` (Massachusetts General Hospital)

**通讯引用:** 15290 | [OpenAlex ID](https://openalex.org/A5058429770)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过对扩散模型产生的结构性幻觉进行局部不稳定性分析，提出“Intrinsic Quenching”机制来抑制幻觉。

**💡 创新点**

将局部内在维数（LID）视为幻觉的根本驱动因素，并设计能在逆过程动态“冷却”LID的能量正则化方法，首次实现了从不稳定性角度修正扩散生成。

**🔧 技术方法**

利用扩散模型的逆过程、LID估计、能量梯度正则化以及动态阈值过滤的组合技术。

**📊 数据集**

在二维高斯混合、手部图像、动物/人脸图像、低剂量CT重建等公开数据集上进行实验。

**📈 对比分析**

与基线、DG、AAM、RODS等方法对比，IQ在幻觉比例、用户偏好、医学诊断准确率上均显著下降或提升，整体性能最优。

**⚠️ 局限性**

仍需人工评测、计算开销较大、对小时间步敏感，且目前仅在无条件生成场景验证，缺乏通用自动化评估指标。

---

## 479. Position: Embodied AI Requires a Privacy-Utility Trade-off

**arXiv ID:** 2605.05017 | [PDF](https://arxiv.org/pdf/2605.05017v1)

**作者:** Xiaoliang Fan `[一作]` (Xiamen University), Cheng Wang `[通讯]` (Xiamen University)

**通讯引用:** 471052 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SPINE 框架，将隐私视为贯穿整个具身人工智能生命周期的动态控制信号，并通过多准则隐私分类矩阵实现跨阶段隐私与效用的协同优化。

**💡 创新点**

创新点在于：①将隐私从单独模块的补丁提升为整体体系的核心设计原则；②构建四层隐私级别（L1–L4）并用可执行元组（S,I,C,Φ）实现自适应升级；③在指令理解、感知、规划、交互四阶段形成统一隐私控制链；④通过模拟与真实 AGV 导航实验验证隐私约束如何向下传播并重塑系统行为。

**🔧 技术方法**

技术手段包括：多准则隐私分类矩阵、隐私控制信号、视觉像素化、身份匿名化、可信执行环境（TEE）、联邦学习、安全多方计算、差分隐私、全同态加密、零知识证明等隐私原语；架构层面结合大型语言模型（LLM）、RGB‑D 视觉模型、LiDAR 传感器、路径规划算法（如 R2R‑CE、ETPNav）。

**📊 数据集**

数据集与实验环境：
- 虚拟 Habitat 仿真环境下的 R2R‑CE 任务（Val‑Seen 与 Val‑UnSeen）。
- 真实 AGV 平台（AgileX SCOUT MINI）在 4m×4m 物理测试台（划分为 L1–L4 四个区域）。

**📈 对比分析**

比较方法：使用成功率（SR）和路径加权成功率（SPL）衡量导航效能；通过增大像素化强度 K 控制隐私级别，观察 SR、SPL 的非线性下降。结果显示，隐私约束在 L4 时可使 SR 降至约 0.4，SPL 降至 0.25；在 L2–L3 时可保持较高效用但仍有一定损失；相较于无隐私的基线，隐私约束导致效用显著下降，但提供了可量化的隐私‑效用权衡。

**⚠️ 局限性**

局限性：
- 仅在导航任务上验证，缺乏对更广泛具身 AI 场景（如对话、操作）的实证；
- 隐私分类矩阵的阈值和提升规则需经验设定，未给出统一理论；
- 仅使用像素化等粗粒度隐私手段，未深入探讨多模态隐私融合；
- 对攻击模型的假设较简化，未覆盖更复杂的侧信道或模型逆向；
- 需要更完善的形式化验证与可审计机制，以确保跨阶段隐私一致性。

---

## 480. CARD: A Multi-Modal Automotive Dataset for Dense 3D Reconstruction in Challenging Road Topography

**arXiv ID:** 2605.05014 | [PDF](https://arxiv.org/pdf/2605.05014v1)

**作者:** Gasser Elazab `[一作]` (CARIAD SE), Olaf Hellwich `[通讯]` (Technische Universit{"a}t Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了CARD多模态驾驶数据集，提供高密度3D地面真值、路面不规则性标注以及轮胎接触点轨迹，旨在为深度估计与完成任务建立新的基准。

**💡 创新点**

创新点包括：① 前后双LiDAR融合实现约500K稠密深度像素；② 覆盖多种非平坦路面（坑洞、凸起、离路段）及离路段；③ 提供精确的轮胎接触点轨迹和全传感器标定；④ 引入针对路面不规则性的评估协议与指标。

**🔧 技术方法**

使用了多传感器同步（双摄、前后LiDAR、IMU、轮速）+ MC2SLAM惯性雷达位姿融合；体素化多视角投票、动态物体剔除（ICP+MAD）、隐藏点剔除、DepthAnything一致性检查；YOLOv8辅助标注。

**📊 数据集**

数据来源于12个德国城市和9个意大利城市，总计约110km、4.7h、约350k图像、175k立体对、500k点/帧，与KITTI、DrivingStereo、RSRD、Waymo等公开数据集进行对比。

**📈 对比分析**

采用标准深度误差指标（AbsRel、SqRel、RMSE、δ1/2/3）和高度误差指标（AbsDiff、δ@10cm）进行全图与目标框评估；单目模型在非平坦区域表现欠佳，Stereo基线（FoundationStereo）细节更好；Fine-tune的MoGe2L结合仿射损失在局部不规则性上显著提升。

**⚠️ 局限性**

局限性包括：动态物体残留和标注模糊；稠密化过程可能丢失细节；单目方法对细微几何的精细化不足；数据集仅覆盖欧洲城市，场景多样性仍有限。

---

## 481. How Long Does Infinite Width Last? Signal Propagation in Long-Range Linear Recurrences

**arXiv ID:** 2605.05113 | [PDF](https://arxiv.org/pdf/2605.05113v1)

**作者:** Mariia Seleznova `[一作]` (Ludwig-Maximilians-Universität München), Mariia Seleznova `[通讯]` (Ludwig-Maximilians-Universität München)

**通讯引用:** 27 | [OpenAlex ID](https://openalex.org/A5049532814)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

研究了有限宽度下线性递归模型中的信号传播，特别是在递归深度与宽度共同增长的情况下，探讨了无限宽度近似的准确性。

**💡 创新点**

提出了精确的有限宽度公式，识别了影响信号传播的递归深度-宽度缩放规律，明确了无限宽度理论失效的深度尺度。

**🔧 技术方法**

使用了复杂高斯初始化的线性递归单元（LRU）和线性递归网络（RNN）模型，结合随机矩阵理论和组合计数方法。

**📊 数据集**

使用了随机初始化的线性递归模型，特别是复杂高斯权重矩阵，进行理论推导和数值实验验证。

**📈 对比分析**

通过与无限宽度理论的比较，发现信号能量在递归深度达到临界尺度时发生显著变化，且在深度-宽度的不同缩放规律下表现出不同的信号传播行为。

**⚠️ 局限性**

目前的分析仅限于随机初始化的线性递归模型，尚未扩展到更一般的递归模型，且对训练过程中的信号传播和稳定性问题仍未解决。

---

## 482. LineRides: Line-Guided Reinforcement Learning for Bicycle Robot Stunts

**arXiv ID:** 2605.05110 | [PDF](https://arxiv.org/pdf/2605.05110v1)

**作者:** Seungeun Rho `[一作]` (RAI Institute), Gabriel Nelson `[通讯]` (RAI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种基于线性导引的强化学习框架Lineride，能够让自定义的儿童自行车型机器人通过用户绘制的几何路径与稀疏姿态约束实现多种可命令的特技动作，并无缝切换至正常驾驶；

**💡 创新点**

创新点在于：①仅使用几何路径与关键姿态而非时间参数化示例；②利用累计行驶距离作为进度度量实现无时序终止；③引入可控跟踪间隙解决物理不可行路径；④在同一策略中同时训练驾驶与特技两种模式；

**🔧 技术方法**

采用PPO强化学习、Hermite曲线与简化动力学模型生成路径、关键姿态奖励、累计距离终止、域随机化与PD控制器；

**📊 数据集**

数据集主要来自真实硬件umv平台与IsaacLab仿真，训练覆盖5种特技（MiniHop、LargeHop、ThreePointTurn、DriftTurn、Backflip）并在模拟与硬件上验证；

**📈 对比分析**

与基于时间参数化示例的WASABI基线对比，Lineride在三连特技成功率上持续领先（接近100%），且不需要手工指定时序；

**⚠️ 局限性**

局限性在于假设用户提供的路径与物理可实现轨迹大致一致，长时限复杂动作难以描述；当前实现依赖运动捕捉系统，限制了室外或非标定环境部署；

---

## 483. Unified Framework of Distributional Regret in Multi-Armed Bandits and Reinforcement Learning

**arXiv ID:** 2605.05102 | [PDF](https://arxiv.org/pdf/2605.05102v1)

**作者:** Harin Lee `[一作]` (University of Washington), Min-hwan Oh `[通讯]` (Seoul National University)

**通讯引用:** 72867 | [OpenAlex ID](https://openalex.org/A5100447410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究并统一多臂赌博机和情节强化学习的分布性回报，并提出一种可调探索奖励的算法。

**💡 创新点**

首次给出同时满足期望、尾部风险和实例相关性最优的分布性回报界限，构建了通用的分布性回报框架。

**🔧 技术方法**

基于UCBVI的自适应奖励、时间均匀置信区间、夸正性与裁剪技术的组合。

**📊 数据集**

论文主要为理论研究，无需实际数据集。

**📈 对比分析**

与现有UCB、Thompson采样等方法比较，理论上实现了最优的期望回报与分布性回报折中，取得了最小的log K或log A 等系数。

**⚠️ 局限性**

局限在于仅给出上界，缺乏实验验证；对非马尔可夫或非平稳环境的推广尚未讨论。

---

## 484. Gated Multimodal Learning for Interpretable Property Energy Performance Prediction and Retrofit Scenario Analysis

**arXiv ID:** 2605.05088 | [PDF](https://arxiv.org/pdf/2605.05088v1)

**作者:** Yunfei Bai `[一作]` (King's College London), Wei He `[通讯]` (King's College London)

**通讯引用:** 30152 | [OpenAlex ID](https://openalex.org/A5022113595)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于能源性能证书（EPC）记录、评估员文本描述和GIS空间信息，构建可解释的门控多模态模型，联合预测住宅建筑的标准评估程序（SAP）和环境影响（EI）连续分数，并用于大规模城市级改造方案评估。

**💡 创新点**

创新点在于①引入样本级门控融合机制，动态权衡结构化表格、自由文本和空间几何信息；②通过辅助的能效等级分类头提升训练稳定性；③多层次可解释性分析（门控权重、SHAP、文本遮罩、空间几何重要性），实现对单一物业预测来源的透明解读。

**🔧 技术方法**

技术手段包括：多模态编码器（表格嵌入+MLP、Transformer编码文本、边界序列+卷积+数值MLP的空间编码）、门控注意力融合、Huber回归损失+交叉熵辅助分类、基于SHAP和遮罩的特征重要性评估、模型蒸馏/迁移学习等。

**📊 数据集**

使用英国伦敦威斯敏斯特区的EPC数据库（约12.5万条物业记录）与OS MasterMap Topographic Area GIS数据（建筑轮廓、面积、高度、方向）进行数据融合，形成三模态特征集合。

**📈 对比分析**

与单模态和双模态基线（表格、文本、空间分别或组合）相比，全模态模型在SAP和EI的平均绝对误差（MAE）约为4.39点、R²分别为0.757和0.748，表现明显优于基线；在改造方案预测中亦能提供可量化的成本与碳排放减排估算。

**⚠️ 局限性**

局限性包括：仅在威斯敏斯特区验证，跨城、跨郊区推广性未知；未与更强的传统机器学习基线（如XGBoost、LightGBM）做直接对比；边界序列编码贡献有限，计算成本待评估；改造效果基于模型推断，缺乏真实建筑模拟或因果估计，需进一步结合工程仿真与不确定性分析。

---

## 485. Order Matters: Improving Domain Adaptation by Reordering Data

**arXiv ID:** 2605.05084 | [PDF](https://arxiv.org/pdf/2605.05084v1)

**作者:** Andrea Napoli `[一作]` (University of Southampton), Paul White `[通讯]` (University of Southampton)

**通讯引用:** 29541 | [OpenAlex ID](https://openalex.org/A5034429046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种通过重新排序训练数据来降低无监督域适配中MMD和CORAL估计方差的技术，称为ORDERED。

**💡 创新点**

将采样顺序视为可优化变量，定义并最小化估计误差的代理目标，并结合分层采样与最小簇大小约束，提出贪心交换优化算法。

**🔧 技术方法**

使用分层采样、动态加权核K‑means聚类、最小化估计误差的代理目标、贪心优化、ResNet‑18、Adam优化器和DomainBed框架等技术。

**📊 数据集**

在Spawrious犬种分类（6个域，18664样本）和Office‑Home图像分类（4个域，15500样本）两个域移位基准上进行实验。

**📈 对比分析**

与k‑means++、DPP、Anticlustering、VaRDASS以及传统UDA方法DANN、CDAN等进行对比，ORDERED在MMD和CORAL下均显著提升目标域准确率，甚至超越部分现代UDA方法，但训练时间比随机采样慢约10倍。

**⚠️ 局限性**

优化过程容易陷入局部最优；计算成本高，尤其在大规模数据集上训练时间显著；仅针对MMD和CORAL验证，未对其他域适配目标进行评估。

---

## 486. Kinematic Discriminants of Deceleration Behavior Modes in Car-Following: Evidence from NGSIM Trajectory Data

**arXiv ID:** 2605.05050 | [PDF](https://arxiv.org/pdf/2605.05050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 487. FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching

**arXiv ID:** 2605.05077 | [PDF](https://arxiv.org/pdf/2605.05077v1)

**作者:** Andranik Sargsyan `[一作]` (Picsart AI Research), Shant Navasardyan `[通讯]` (Picsart AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FlowDIS，一种基于流匹配的二值图像分割模型，支持文本引导的精细前景分割

**💡 创新点**

将分割任务重新表述为图像到掩码的流匹配，并引入位置感知实例配对（PAIP）策略提升语言可控性

**🔧 技术方法**

采用流匹配框架（MMDiT）、VAE编码解码、CLIP/T5文本编码以及Beta非均匀时间调度

**📊 数据集**

在DIS5K数据集上训练评测，包括DIS-TR、DIS-VD、DIS-TE四子集

**📈 对比分析**

与10种SOTA DIS方法对比，FlowDIS在所有测试集上取得最高F_β^ω、最低MAE，并在语言引导任务中表现出更强的可控性；1步推理即可领先，2步推理更优

**⚠️ 局限性**

对单目标场景表现不如复杂多目标时的PAIP增益明显，且需要更多显存与推理时间；缺乏对实时部署的评估

---

## 488. Look Once, Beam Twice: Camera-Primed Real-Time Double-Directional mmWave Beam Management for Vehicular Connectivity

**arXiv ID:** 2605.05071 | [PDF](https://arxiv.org/pdf/2605.05071v1)

**作者:** Avhishek Biswas `[一作]` (University of Nebraska--Lincoln), Mehmet C. Vuran `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于车载摄像头的实时双向毫米波波束管理框架（Look Once, Beam Twice），实现了快速波束对准与稳健链路维护。

**💡 创新点**

创新点在于结合模型驱动的初始波束估计、闭环SNR反馈、以及轻量级偏移跟踪/神经网络自适应，形成混合闭环学习架构，实现了高可靠性与低延迟的双向波束搜索。

**🔧 技术方法**

主要技术包括YOLOv11目标检测、相机投影与无线坐标映射、离散波束书匹配、迭代波束细化（MA/MLP）以及基于SNR阈值的自适应反馈。

**📊 数据集**

使用了室内测试平台、校园户外实测、以及公开的Scenario 6、7、9数据集进行评估，并在未见场景中验证通用性。

**📈 对比分析**

与5G NR层级波束成形以及MNet‑LeNet、ResNet‑50等全端ML模型相比，本文方法在不同SNR阈值下的误差率低至1.1–1.4%，波束对准延迟小于0.5 s，覆盖率显著高于对手，尤其在未见环境中优势尤为突出。

**⚠️ 局限性**

局限性主要在于硬件相关的推理延迟与天线切换速度限制了对高速移动的支持，且目前仅针对LOS场景，未涵盖非视距、复杂多径与高动态相机误差等情况。

---

## 489. Reduced-order Neural Modeling with Differentiable Simulation for High-Detail Tactile Perception

**arXiv ID:** 2605.05053 | [PDF](https://arxiv.org/pdf/2605.05053v1)

**作者:** Yuhu Guo `[一作]` (University of Manchester), Guoxing Fang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将粗粒 Material Point Method (MPM) 动力学与隐式神经解码器耦合的低阶神经仿真框架，用于高细节触觉感知。

**💡 创新点**

创新点在于通过训练隐式神经场从低维潜在空间重建亚粒子级触觉细节，实现高保真且可微的触觉仿真；与 TacIPC 相比，速度提升 65% 以上，内存降低 40%。

**🔧 技术方法**

使用技术包括 MPM、隐式神经网络解码器、降维子空间（autoencoder）、可微物理仿真、Taichi 与 PyTorch 的混合实现。

**📊 数据集**

使用数据集为：基于 10^6 粒子与 10^4 粒子配对的高低分辨率 MPM 仿真数据，以及 UR5e 机器人配合 GelSight Mini 采集的八个 FDM 打印物体的真实触觉序列。

**📈 对比分析**

通过与 Tacchi 与 TacIPC 在压印仿真、触觉渲染和 3D 重建等任务中的 SSIM、MAE、PSNR、Chamfer 距离等指标对比，显示本方法在 SSIM 上提升 5%，MAE 降低 20%，PSNR 提升 4 dB，渲染时间从 0.25 s 降至 0.17 s，内存使用 3.5 GB，整体性能显著提升。

**⚠️ 局限性**

局限性包括：仅适用于静态或低速压印，难以处理高速滑动、碰撞；对材料参数的泛化受限；侧重几何精度，未精确估计力学量；光照假设固定，影响真实感。

---

## 490. Local Homophily on Bicolored Graphs is $\mathbf{P}$-complete

**arXiv ID:** 2605.05047 | [PDF](https://arxiv.org/pdf/2605.05047v1)

**作者:** Pablo Concha-Vega `[一作]` `[通讯]` (Aix Marseille Univ), Pablo Concha-Vega (Aix Marseille Univ)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种基于多数动力学和同质性的新型局部图变换——局部同质性（local homophily），并证明在对任意给定的二染色图连续应用该变换后判断两点是否连通是PSPACE-完整问题。

**💡 创新点**

创新点在于提出了这种新的图变换模型，并利用它构造布尔电路的模拟，从Circuit Value Problem（CVP）实现了logspace归约，首次证明该连通性问题属于PSPACE且是PSPACE-完整。

**🔧 技术方法**

采用了布尔电路模拟技术、花瓣图（flower graph）结构、关键结构引理、逻辑门（AND、OR、duplicator）gadget 的设计、以及logspace归约等理论计算机科学方法。

**📊 数据集**

无；本文为纯理论研究，不涉及任何实验数据集。

**📈 对比分析**

由于是理论复杂度证明，没有实验对比；通过归约和结构分析证明该问题在多项式时间内可判定（PSPACE可解）并且是PSPACE-完整，表明其计算复杂度最高。

**⚠️ 局限性**

局限性包括：仅讨论无向简单二染色图；未考虑有向图或多色图；未对变换的动力学稳定性、收敛性或周期行为进行深入分析；缺乏实际实现与实验评估。

---

## 491. Sampling Simultaneous Edge-Colorings

**arXiv ID:** 2605.05046 | [PDF](https://arxiv.org/pdf/2605.05046v1)

**作者:** Ezra Furtado-Tiwari `[一作]` (University of California, Santa Barbara), Eric Vigoda `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 3101 | [OpenAlex ID](https://openalex.org/A5051363868)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了同时边着色的采样问题，给定一对图G_1和G_2，目标是对G_1∪G_2进行边着色，使得每个图都被正确着色。

**💡 创新点**

提出了一种新的加权汉明距离，利用该距离的耦合分析实现了快速混合，优化了Glauber动态和翻转动态的混合时间。

**🔧 技术方法**

使用了Markov链，特别是Glauber动态和翻转动态来进行边着色的随机采样。

**📊 数据集**

使用了具有相同顶点集V的图G_1和G_2，且每个图的最大度数为Δ。

**📈 对比分析**

与之前的方法相比，Glauber动态在k>6Δ时实现了O(mlogn)的混合时间，而翻转动态在k≥5.948Δ时也实现了O(mlogn)的混合时间，性能显著提升。

**⚠️ 局限性**

限制在于当前结果主要适用于最大度数Δ的情况，且在k的范围内有一定的限制。

---

## 492. The Predictive-Causal Gap: An Impossibility Theorem and Large-Scale Neural Evidence

**arXiv ID:** 2605.05029 | [PDF](https://arxiv.org/pdf/2605.05029v1)

**作者:** Kejun Liu `[一作]` (Soochow University), Kejun Liu `[通讯]` (Soochow University)

**通讯引用:** 1550 | [OpenAlex ID](https://openalex.org/A5101784796)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在合成线性高斯动力学和非线性Duffing振荡器中，系统研究了预测自监督学习导致的因果失配现象——即编码器倾向于追踪易预测的环境而非目标系统，提出并验证了预测-因果缺口。

**💡 创新点**

创新点在于①提出不可避免性定理，证明存在稳定动力学使得任何预测风险最小化器都无法对齐系统子空间；②系统化量化因果完整性损失（因果保真度）；③展示高维环境下缺口放大且无论增大模型容量或优化都无法弥补。

**🔧 技术方法**

使用的技术包括线性高斯动力学解析、不可避免性定理证明、随机梯度优化训练线性和非线性编码器（两层MLP、GRU），以及因果保真度指标和高维环境扩展实验。

**📊 数据集**

数据集为合成的539种线性高斯动力学配置（每种5个随机种子，共2695次训练）和100个Duffing‑Ornstein‑Uhlenbeck耦合任务，生成了数千条轨迹用于训练与评估。

**📈 对比分析**

通过比较线性最优编码器与神经网络最优编码器的预测误差和因果保真度，发现神经网络可将预测误差降低99.3%但因果保真度仅为0.49；在N=10、50、100的高维实验中因果保真度降至≈10⁻⁸，表明预测优势伴随因果误差显著扩大。

**⚠️ 局限性**

局限性：实验仅基于合成动力学，缺乏真实世界验证；理论主要针对单向线性编码器，对更复杂模型的普适性未完全探讨；尽管操作性约束可部分抑制缺口，但如何在不手工指定系统‑环境边界的情况下自动实现仍是开放问题。

---

## 493. The Pinocchio Dimension: Phenomenality of Experience as the Primary Axis of LLM Psychometric Differences

**arXiv ID:** 2605.05080 | [PDF](https://arxiv.org/pdf/2605.05080v1)

**作者:** Hubert Plisiecki `[一作]` (IDEAS Research Institute), Marcin Moskalewicz `[通讯]` (IDEAS Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对50种公开大语言模型（LLM）使用45种已验证的心理测评问卷进行评估，探究模型间心理测量差异的潜在维度；

**💡 创新点**

提出Pinocchio轴（Π），揭示模型对自身是否具备“主观经验”这一自我表述倾向是跨模型心理测量差异的主导轴；

**🔧 技术方法**

采用超监督语义差分（SSD）分析问卷文本语义梯度、探索性因素分析（EFA）与主成分分析（PCA）提取主维度，并设计Pinocchio评分（π_i）量化项目对体验要求的敏感度；

**📊 数据集**

数据集为206,659条有效问卷回答，涉及50个模型、45个测评工具、三种提问条件（中性、人工模拟等）；

**📈 对比分析**

通过SSD和π_i的相关性验证，Pinocchio轴解释了47.1%的跨模型方差，模型在该轴上从“体验丰富”到“行为反应性”两极分布，结果表明不同微调版本在同一供应商内部即可出现显著差异；

**⚠️ 局限性**

局限在于所有测评均为自我报告，无法验证模型实际行为是否与报告一致；使用公开模型且受限于单一时间点；π_i基于对比两种提问条件，可能受提示策略影响；

---

## 494. Prompt-Anchored Vision-Text Distillation for Lifelong Person Re-identification

**arXiv ID:** 2605.05027 | [PDF](https://arxiv.org/pdf/2605.05027v1)

**作者:** Wen Wen `[一作]` (University of Electronic Science and Technology of China), Shiliang Zhang `[通讯]` (Peking University)

**通讯引用:** 13678 | [OpenAlex ID](https://openalex.org/A5055433405)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个无示例的终身人像重识别框架 PAD，该框架利用冻结的文本编码器作为跨域语义锚点，并通过 Prompt 机制在视觉和文本两侧实现对齐与知识蒸馏，实现语义保持与域适应的平衡。

**💡 创新点**

创新点在于：① 异步视觉-文本架构，文本侧保持不变，仅蒸馏 Prompt；② 双模提示设计，TA‑Prompt 负责语义锚定，VA‑Prompt 负责域级适应；③ 文本侧采用弱蒸馏（KL）与 EMA 视觉教师相结合，既抑制语义漂移又保持足够的可塑性；④ 仅更新 Prompt 与少量可训练层，显著降低参数量并避免全模型微调。

**🔧 技术方法**

使用技术包括：预训练 CLIP ViT‑B/16，冻结文本编码器，SupCon 对比学习，温度缩放 KL 蒸馏，EMA 视觉教师，双向 Prompt（General/Expert），选择性层解冻，分类头 + triplet 损失，评价指标 mAP 与 Rank‑1。

**📊 数据集**

实验使用 12 个 ReID 域：见域 Market1501、CUHK‑SYSU、DukeMTMC、MSMT17、CUHK03；未见域 CUHK01、CUHK02、VIPeR、PRID2011、i‑LIDS、GRID、SenseReID。

**📈 对比分析**

与多种 SOTA（LwF、DualPrompt、FCS、PAEMA、DKP++、DAFC、DASK 等）在 AKA‑order1/2 评测下对比，PAD 在 seen 平均 mAP/R1 及 unseen 平均指标均显著领先，mAP 达到约92%/93%，Rank‑1 也超过 90%，证明其在语义保持与跨域泛化上的优势。

**⚠️ 局限性**

局限性包括：对某些域仍存在挑战；蒸馏权重固定，缺乏自适应调节；仅针对单模人像 ReID，未扩展到更复杂的多模或服装 ReID 场景。

---

## 495. Height-Guided Projection Reparameterization for Camera-LiDAR Occupancy

**arXiv ID:** 2605.05072 | [PDF](https://arxiv.org/pdf/2605.05072v1)

**作者:** Yuan Wu `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 128124 | [OpenAlex ID](https://openalex.org/A5100604690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

HiPR 提出了一种利用 LiDAR 高度先验动态重参数化投影空间的 Camera‑LiDAR 3D 占据预测框架，并在训练时采用 Progressive Height Conditioning 稳定优化。

**💡 创新点**

创新点包括：
- 将 LiDAR 生成的 BEV 高度图作为投影空间的高度先验，按高度自适应地调整每个柱子的采样范围；
- 引入高度有效性掩码仅在有效柱子上聚合图像特征，消除无效投影；
- 训练阶段使用 PHC（Progressive Height Conditioning），先用 GT 高度逐步过渡到 LiDAR 高度，缓解噪声导致的训练不稳定。

**🔧 技术方法**

核心技术：BEV 高度图编码、Height‑Guided Reparameterization、Progressive Height Conditioning、基于变形注意力的后向投影、3D 解码器以及多模态特征融合。

**📊 数据集**

使用 nuScenes、Occ3D 和 SurroundOcc 三个公开数据集进行实验，训练时结合多视角相机与 LiDAR 传感器。

**📈 对比分析**

与多种 state‑of‑the‑art 方法对比：
- 在 Occ3D 上 mIoU 54.7（HiPR）超过 DAOcc 54.3 与 OccFusion 46.7，RayIoU 53.4 超过 STCOcc 46.1；
- 在 SurroundOcc 上 mIoU 30.4 超过 OccCylindrical 28.7，且在所有类别上均有显著提升；
- 轻量级 HiPR‑mini 仍能达到 48.4 mIoU，同时 FPS 10+，在实时性能与准确度上实现最优权衡。

**⚠️ 局限性**

局限性：
- 依赖 LiDAR 高度信息，远距离或高度稀疏区域的先验可能不可靠；
- 单一高度表示难以充分描述复杂多层结构，未来可探索更丰富的几何先验。

---

## 496. The Impossibility Triangle of Long-Context Modeling

**arXiv ID:** 2605.05066 | [PDF](https://arxiv.org/pdf/2605.05066v1)

**作者:** Yan Zhou `[一作]` (Changsha University of Science and Technology), Yan Zhou `[通讯]` (Changsha University of Science and Technology)

**通讯引用:** 25918 | [OpenAlex ID](https://openalex.org/A5019614740)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了长序列模型存在一个不可避免的“不可避免三角”——效率、紧凑性与回忆这三项属性不能同时满足；

**💡 创新点**

创新点在于提出统一的在线序列处理器（OSP）抽象，利用信息理论工具（数据处理不等式、Fano不等式）给出严格的回忆上限，并对52种模型进行系统分类；

**🔧 技术方法**

主要技术包括信息论证明、状态空间模型统一表达、数据处理与Fano不等式的应用以及对不同架构的理论分析；

**📊 数据集**

实验使用了合成的关联回忆任务（(n,V)），在不同模型和长度下测量回忆能力；

**📈 对比分析**

与Transformer、线性注意力、SSM等模型对比，实验结果始终低于理论上限，且混合模型随注意力比例平滑过渡，验证了理论与实验的一致性；

**⚠️ 局限性**

局限性包括仅在小规模模型和合成任务上验证，缺乏对自然语言真实分布的分析，且结论基于最坏情况，常规模型可能表现更好。

---

## 497. A Comparison Between Co-Located and Distributed MIMO Deployments in OFDM-ISAC Networks

**arXiv ID:** 2605.05059 | [PDF](https://arxiv.org/pdf/2605.05059v1)

**作者:** Maryam Darabi `[一作]` (Consorzio Nazionale Interuniversitario per le Telecomunicazioni), Stefano Buzzi `[通讯]` (Consorzio Nazionale Interuniversitario per le Telecomunicazioni)

**通讯引用:** 16712 | [OpenAlex ID](https://openalex.org/A5013919815)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

比较分布式（CF-mMIMO）与集中式（MC-mMIMO）在OFDM-ISAC网络中的感知性能

**💡 创新点**

首次在统一的GLRT检测框架下对两种网络部署进行直接对比，并突出分布式部署在空间多样性上的优势

**🔧 技术方法**

OFDM信号、MIMO、GLRT检测、频域子载波分配、仿真

**📊 数据集**

基于1 km²随机布置的UE与目标场景（3 GHz、12子载波等参数）进行仿真，没有使用公开数据集

**📈 对比分析**

通过仿真得到感知SNR的累计分布，发现CF在大多数配置下的SNR更高，尤其在发射资源分散或子载波增多时表现突出

**⚠️ 局限性**

未考虑AP间干扰与杂波，仅假设理想同步；通信性能未深入评估

---

## 498. ScriptHOI: Learning Scripted State Transitions for Open-Vocabulary Human-Object Interaction Detection

**arXiv ID:** 2605.05057 | [PDF](https://arxiv.org/pdf/2605.05057v1)

**作者:** Minh Anh Nguyen `[一作]` (Phenikaa University), Linh Chi Vo `[通讯]` (Phenikaa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ScriptHOI框架，将交互短语拆解为软脚本并与视觉状态进行匹配以实现开放词汇HOI检测。

**💡 创新点**

创新点在于将交互语义转化为多槽位脚本并用覆盖/冲突评分校准检测，解决了物体亲和力误判。

**🔧 技术方法**

采用视觉状态分词器、脚本解析器、脚本-状态匹配、间隔部分标签学习与反事实脚本对比损失等技术。

**📊 数据集**

使用HICO-DET、V-COCO以及开放词汇分割的HICO-DET数据集进行评估。

**📈 对比分析**

与多种基线比较，在mAP、罕见类别、开放词汇和冲突误判率上均获得显著提升，尤其在未见类别上提升约10%。

**⚠️ 局限性**

局限包括脚本解析对抽象动词不稳健、静态图像难以捕捉动态状态、姿态估计误差和脚本词表设计的主观性。

---

## 499. Adaptive Learning Strategies for AoA-Based Outdoor Localization: A Comprehensive Framework

**arXiv ID:** 2605.05055 | [PDF](https://arxiv.org/pdf/2605.05055v1)

**作者:** Bac Trinh-Nguyen `[一作]` (CY Cergy Paris University), Arsenia Chorti `[通讯]` (Barkhausen Institut)

**通讯引用:** 1942 | [OpenAlex ID](https://openalex.org/A5053749805)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套适用于 5G/6G 无线环境的自适应 AoA（到达角）定位框架，结合离线大数据训练与在线小样本增量学习与少样本学习，形成完整的定位系统。

**💡 创新点**

创新点包括：①将离线层级分类与在线增量树/森林模型以及 ProtoNet 少样本学习融合为一个可切换的框架；②引入基于 Optuna 的贝叶斯超参优化，实现模型在离线与在线两阶段的自动调优；③利用 CVAE 生成式数据增强提升样本稀缺场景下的鲁棒性；④在真实 64 天线 mMIMO OFDM CSI 数据集上系统性评估，验证在 LoS 与 NLoS 区域均能达到 99%+ 的轨迹识别精度。

**🔧 技术方法**

所用技术：MUSIC/ESPRIT AoA 提取、两阶段层级分类（LR/KNN/RF/GBM/LightGBM/XGBoost/堆叠集成）、Optuna 超参搜索、River 库的增量树/森林（HT/HAT/ARF/SRP/AMF/GNB）、ProtoNet 原型网络、CVAE 条件变分自编码器、CGAN、数据增强、在线流式学习与少样本元学习。

**📊 数据集**

数据集：来自德国斯图加特诺基亚校园的 64 阵列、50 子载波、2.18 GHz OFDM mMIMO CSI 采集数据，包含 10 条 LoS 与 10 条 NLoS 路径，共计约 120k 个测点，生成 200 维 AoA 特征。

**📈 对比分析**

比较方法：对离线层级分类做基线与 Optuna 调优对比；对在线增量学习做多模型对比（AMF、ARF、GNB、HAT、HT、SRP）并与 CVAE 数据增强相结合；对 ProtoNet 做标准元学习与连续少样本学习对比。结果表明：离线框架在 LoS/ NLoS 下分别达 99.82%/97.99% 的轨迹识别；AMF 在线实现约 94% 的准确率，遗忘率仅 0.025–0.043；ProtoNet 在 1–10 shot 下逐步提升，LoS 最高约 91%/ NLoS 约 76%。

**⚠️ 局限性**

局限性：①需大量离线训练与频繁超参调优，适应性受限于已收集数据；②在线增量学习在高度非平稳环境下仍可能出现概念漂移；③CVAE 与 ProtoNet 的生成/元学习在噪声极大或阻塞情况表现尚未验证；④系统在实际部署中需考虑硬件校准误差、干扰、功率限制等实际问题。

---

## 500. Direct Product Flow Matching: Decoupling Radial and Angular Dynamics for Few-Shot Adaptation

**arXiv ID:** 2605.05054 | [PDF](https://arxiv.org/pdf/2605.05054v1)

**作者:** Hongxu Chen `[一作]` (HKUST), Long Chen `[通讯]` (HKUST)

**通讯引用:** 3116 | [OpenAlex ID](https://openalex.org/A5100679798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于流匹配的跨模态适配方法（WP‑FM/DP‑FM），通过在扭曲乘积流形上建模连续对齐过程，实现多步细粒度对齐。

**💡 创新点**

创新点：① 从极坐标分解视角引入扭曲乘积流形，识别并消除角速度失真和径向动态被忽略的几何瓶颈；② 通过常数扭曲函数得到解耦的圆柱流形（DP‑FM），实现径向与角度完全独立的等速测地线；③ 将隐藏状态条件化的无分类器引导（CFG）融入流匹配，恢复目标域的上下文信息。

**🔧 技术方法**

核心技术包括极坐标分解、Riemannian 流匹配、扭曲乘积流形几何、指数映射求解 ODE、时间移位调度、无分类器引导（CFG）以及多步梯度回归。

**📊 数据集**

在 11 个常见的视觉‑语言少样本分类基准上进行评估：Aircraft、EuroSAT、DTD、SUN397、StanfordCars、OxfordPets、UCF101、Flowers102、Caltech101、Food101、ImageNet，划分难易两组。

**📈 对比分析**

与 FMA、HFM、WP‑FM（Euclidean/Hyperbolic）以及多种单步 PEFT 方法（CLIP‑LoRA、CLIP‑Adapter、CoOp、CoCoOp 等）进行对比；DP‑FM 在所有 K‑shot 设置下均取得领先，平均提升约 3–7%（在难题集上 +3%~+7%，在易题集上 +5%~+9%）。

**⚠️ 局限性**

当前方法仍依赖预训练 CLIP 特征，对超大规模数据或非图像/文本模态的泛化能力未做验证；实现复杂度较高（需要流体动力学求解和 CFG 条件化），在极少样本或计算资源受限场景下可能受限。

---

## 501. Efficient Cost-Based Rewrite in a Bottom-Up Optimizer

**arXiv ID:** 2605.05044 | [PDF](https://arxiv.org/pdf/2605.05044v1)

**作者:** Qi Cheng `[一作]` (Huawei), Per-Ake Larson `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于成本的重写框架，针对底层查询优化器中的成本驱动重写规则进行高效处理；

**💡 创新点**

1）多层缓存机制（基表、连接、子查询）重用中间 CBO 结果，显著降低重复规划；2）在查询重写阶段使用“educated guess”快速判断是否需要重写，产生紧上界用于剪枝；3）将两者结合，实现编译时间大幅下降；

**🔧 技术方法**

哈希签名匹配、表列映射、Bloom 过滤器、约束与提示兼容的匹配算法，基于 GaussDB 优化器实现；

**📊 数据集**

TPC‑H（SF10，10 GB）和 TPC‑DS（SF10，10 GB）数据集，以及内部客户真实工作负载；

**📈 对比分析**

与传统无缓存、无猜测的“naïve”成本重写方法对比。实验显示，在 TPC‑H 上总体编译时间降低约46%，在 TPC‑DS 上降低约29%，且生成的执行计划与基线保持一致；

**⚠️ 局限性**

仅在单次查询编译周期内的短命缓存；目前仅实现七条成本重写规则；缺乏跨编译的长期缓存机制，且对极端复杂提示、外部系统支持有限。

---

## 502. Few-Shot Learning Pipeline for Monkeypox Skin Disease Classification Using CNN Feature Extractors

**arXiv ID:** 2605.05034 | [PDF](https://arxiv.org/pdf/2605.05034v1)

**作者:** Md. Safirur Rashid `[一作]` (Islamic University of Technology), Md. Hasanul Kabir `[通讯]` (Islamic University of Technology)

**通讯引用:** 3054 | [OpenAlex ID](https://openalex.org/A5071274329)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套基于SimpleShot的少样本学习流程，用冻结的CNN特征提取器对猴痘及类似疱疹性皮肤疾病进行分类，并系统评估六种主流CNN骨干网络的表现。

**💡 创新点**

首创在猴痘诊断中使用无元学习、无额外预训练的SimpleShot框架，突出轻量级骨干网络在低样本情境下的可迁移性与稳健性。

**🔧 技术方法**

采用SimpleShot原型分类器、预训练并冻结的VGG16、InceptionV3、ResNet50、DenseNet121、MobileNetV2_100和EfficientNet B1等CNN骨干，图像尺寸统一为128×128，进行N-way M-shot实验。

**📊 数据集**

利用三大公开数据集：MSLD v1.0（二分类）、MSLD v2.0（六分类）和MSID（四分类），全部采用原始未增强图像进行评估。

**📈 对比分析**

通过在100个随机任务上计算平均准确率和95%置信区间，比较六种骨干的分类性能，EfficientNet B1在2/4/6-way 1/5/10-shot下取得最高准确率（最高可达0.696±0.011），同时在跨数据集实验中二分类迁移稳定（约63–68%），多分类迁移显著下降。

**⚠️ 局限性**

受限于样本量小、类别不平衡、肤色和地理多样性缺乏，以及跨域偏移导致的泛化性能下降，未来需探索自监督预训练和域自适应策略以提升稳健性。

---

## 503. Detecting Hallucinations in Large Language Models via Internal Attention Divergence Signals

**arXiv ID:** 2605.05025 | [PDF](https://arxiv.org/pdf/2605.05025v1)

**作者:** Gijs van Dijk `[一作]` (Utrecht University), Gijs van Dijk `[通讯]` (Utrecht University)

**通讯引用:** 913 | [OpenAlex ID](https://openalex.org/A5025445762)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种单通道、单向前传递的内部信号方法——利用注意力分布与均匀分布的KL散度来量化LLM的置信度，从而检测生成文本中的幻觉并预测答案正确性。

**💡 创新点**

创新点在于将注意力矩阵视为可解释的内部概率分布，通过KL散度捕捉知识缺失或误导性注意，从而在不需要采样或外部模型的前提下，快速获得可靠的幻觉检测信号；此外发现该信号在中间层、命名实体和数字等事实性词上尤为显著。

**🔧 技术方法**

技术实现包括：①计算每个注意力头对历史上下文的KL散度并平均聚合；②使用L1正则化的逻辑回归探针将聚合特征映射到答案正确性；③采用交叉验证、AUROC和ECE等指标评估模型。

**📊 数据集**

实验使用四大数据集：TruthfulQA（多选）、TriviaQA（开放域）、HotpotQA（多跳推理）和GSM8K（数学推理）；对三种指令调优模型：Llama‑3.2‑3B、Qwen3‑4B和Mistral‑7B进行评估。

**📈 对比分析**

与现有方法（TOHA、SelfCheckGPT、语义熵等）对比，本文方法在HotpotQA上AUROC为0.78±0.02，显著优于TOHA（0.71）及其他基线；在TruthfulQA和TriviaQA上AUROC分别超过0.89和0.83；在GSM8K上Qwen3‑4B达到0.945，表明该方法在多任务、多模型上性能与甚至优于多种基线，并且仅需一次前向传播，计算成本极低。

**⚠️ 局限性**

局限性包括：①KL散度与幻觉之间缺乏因果解释，探针仅提供预测信号；②探针权重随数据集和模型变化，解释性有限；③需要访问内部注意力权重，无法应用于黑盒模型；④信号分布在多个头和层，单个头并非关键，可能限制可解释性与迁移性。

---

## 504. CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels

**arXiv ID:** 2605.05023 | [PDF](https://arxiv.org/pdf/2605.05023v1)

**作者:** Xing Ma `[一作]` (Shanghai Jiao Tong University), Jin Song Dong `[通讯]` (National University Of Singapore)

**通讯引用:** 6752 | [OpenAlex ID](https://openalex.org/A5085067496)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对专家编写的 CUDA 注意力核，利用 LLM 通过 lift–transfer–lower 工作流自动生成符合新语义的高性能注意力变体核，保持原有执行组织的效率。

**💡 创新点**

创新点在于引入可执行中间表示 CuIR 使执行组织显式化，并在 IR 层进行语义转换与性能优化，再通过差分低级重写回 CUDA；该结构化流程显著提升了正确性与性能，突破了传统 LLM 直接生成核的稳定性与效率瓶颈。

**🔧 技术方法**

使用大型语言模型（GPT‑5、Claude‑3.5‑Sonnet、DeepSeek‑V3、Qwen‑3‑235B 等）进行代码分析与生成；构建可执行中间表示 CuIR ；链式思考（CoT）与 ReAct 迭代编辑；GPU 架构知识与 PyTorch 参考同步；CUDA 低级重写与差分补丁。

**📊 数据集**

通过多种 LLM 生成的核进行基准；使用 Llama2‑7B、Qwen2.5‑72B、Llama3.1‑405B 三个真实 LLM 配置的注意力变体；覆盖不同序列长度（1k–8k）和批大小；评估基于不同 LLM 后端的性能差异。

**📈 对比分析**

在 NVIDIA A100 与 H100 GPU 上，与 PyTorch、FlexAttention、Qimeng‑Attention、FlashInfer 等基线进行比较；平均提升 12.69×/19.82×（A100/H100）相对 PyTorch，1.18×/1.62× FlexAttention，3.33×/4.35× Qimeng‑Attention；在非原生 FlashAttention 变体上与 FlashInfer 对齐 1.07×，并取得 3.49× 的加速。

**⚠️ 局限性**

依赖于高质量专家 CUDA 核作为源，无法直接迁移到非 NVIDIA 主流硬件（如 FPGA）；目前仅验证了注意力变体，对其他 HPC 任务（如科学计算）尚未扩展。

---

## 505. Chaotic Contrastive Learning for Robust Texture Classification

**arXiv ID:** 2605.05012 | [PDF](https://arxiv.org/pdf/2605.05012v1)

**作者:** Joao B Florindo `[一作]` `[通讯]` (University of Campinas), Joao B Florindo (University of Campinas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过自监督学习结合确定性混沌动力学，提出一种用于纹理分类的对比预训练与注意力融合框架

**💡 创新点**

创新点在于将 Logistic、Tent、Sine 三种混沌映射作为像素级数据增强，以此提升网络对尺度、光照等变形的鲁棒性，并引入注意力机制融合自监督小网络与大规模监督网络的特征

**🔧 技术方法**

技术包括：混沌映射数据增强、SimCLR 对比学习、ConvNeXt-Tiny 与 ConvNeXt-Large 编码器、SE 注意力融合、NT-Xent 损失、AdamW + Cosine Annealing 优化

**📊 数据集**

使用六大纹理基准：FMD、UMD、KTH‑TIPS2‑b、DTD、GTOS、1200Tex（植物纹理）

**📈 对比分析**

与多种基线（EfficientNet、ConvNeXt、RegNet、DeepTEN、FENet、RADAM 等）比较，最佳配置（Sine Map + 15 轮预训练）在 FMD 92.0%、UMD 99.8%、KTH‑TIPS2‑b 94.6%、DTD 84.4%、GTOS 86.4%、1200Tex 97.6%，均优于或匹敌当前最先进方法

**⚠️ 局限性**

主要局限在于缺乏对尺度变化的显式建模，导致 KTH‑TIPS2‑b 中不同尺度样本仍出现轻微混淆，未来可加入多尺度网络或耦合通道混沌同步

---

## 506. Text Corpora as Concept Fields: Black-Box Hallucination and Novelty Measurement

**arXiv ID:** 2605.05103 | [PDF](https://arxiv.org/pdf/2605.05103v1)

**作者:** Nicholas S. Kersting `[一作]` (Oracle Corporation), Saad Taame `[通讯]` (Oracle Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了文本语料库的概念场（Concept Field），通过在句子嵌入空间中对相邻句子间的差分进行局部高斯估计，计算出每个句子点的漂移向量及其不确定性；利用该场对生成文本的句子序列进行“agreement”评分，判定其是否符合语料库的“有据可查”或“新颖”属性。

**💡 创新点**

创新点在于：①用句子级别的概念嵌入构建可解释的漂移向量场；②通过IDW插值在高维空间中快速估计局部字段；③提出基于z距离的概率化度量ζ，可直接映射为置信度并支持三类决策（有据、无据、未知）；④将该方法应用于法律法规与文学文本的多域评估，并展示字段的几何特征（divergence/curl）可揭示语义模式。

**🔧 技术方法**

技术上主要使用SONAR（或OmniSONAR）进行句子嵌入，构建Vector Sequence Database (VSDB)并使用FAISS做近邻检索；采用IDW插值得到局部漂移向量；通过标准化绝对z距离求ζ；在评估时对句子对的delta与字段进行比较。

**📊 数据集**

实验数据集包括：U.S. Code of Federal Regulations（约70M句/1B token）用于“hallucination”式的有据判定；Project Gutenberg（约57k书/5B token）用于“novelty”式的创意判定；此外还有2D弹道轨迹的toy实验。

**📈 对比分析**

与基线比较：去掉概念场的VSDB仅用欧氏/余弦距离；基线VDB（所有句子对）仅用余弦距离。结果显示，Concept Field在两域上均保持高精度（F1≈0.84-0.86，AUC≈0.93-0.94），并在不同阈值下展现出一致的覆盖-性能曲线；相比之下，其他基线需在每个语料库单独校准，且在错误率/覆盖度上波动较大。

**⚠️ 局限性**

局限性包括：①字段的局部插值对稀疏或高维空间可能不稳定；②对不同语言或非文本序列的迁移需进一步验证；③阈值设定仍需人工经验；④对极端新颖或极端离群点（out‑of‑corpus）判定时不够鲁棒；⑤目前方法主要评估句子对，完整序列的端到端评价尚未实现。

---

## 507. CapsID: Soft-Routed Variable-Length Semantic IDs for Generative Recommendation

**arXiv ID:** 2605.05096 | [PDF](https://arxiv.org/pdf/2605.05096v1)

**作者:** Wenzhuo Cheng `[一作]`, Zhengwei Zheng `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于胶囊路由的生成式推荐 tokenizer，取代硬残差量化，生成可变长度的语义 ID 序列。

**💡 创新点**

创新点在于：①使用软胶囊路由和多轮一致性迭代，保留多重语义信息；②以路由置信度驱动可变长度截断；③引入语义兼容的子词合成（SemanticBPE）以提升编码效率。

**🔧 技术方法**

技术实现包括胶囊网络路由、softmax 决策、迭代 EM 近似、Gumbel‑Softmax 门控子词合成、以及基于 Transformer 的自回归序列生成。

**📊 数据集**

实验数据集包括 Amazon Beauty、Sports、Toys 三个公开基准以及一个 35M 项目的工业级多模态商品目录。

**📈 对比分析**

与现有硬量化 tokenizer（TIGER、ReSID 等）和补丁式系统（COBRA、UniRec）对比，+ 在 Recall@10 上平均提升约 9.6%，在工业数据上匹配或超越 COBRA 的召回，同时推理延迟仅为其 51%，显著提高效率。

**⚠️ 局限性**

局限性包括：①胶囊路由导致 tokenizer 训练成本提升约 20–30%；②假设固定的胶囊层数与容量，难以动态适应目录扩展；③潜在的流行度偏见，需配合公平性或曝光校正措施。

---

## 508. A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry

**arXiv ID:** 2605.05095 | [PDF](https://arxiv.org/pdf/2605.05095v1)

**作者:** Jingsen Zhu `[一作]` (Cornell University), Alexander Terenin `[通讯]` (Cornell University)

**通讯引用:** 233 | [OpenAlex ID](https://openalex.org/A5037482578)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于贝叶斯决策理论的任务特定下一个最佳视角选择框架，利用不确定几何重建实现主动扫描；

**💡 创新点**

创新点在于将贝叶斯优化中的期望改进思想推广到主动点云扫描，并通过任务相关的效用函数实现对不同下游任务（分类、分割、物理仿真）的定制化视角规划；

**🔧 技术方法**

采用几何随机泊松表面重建（Gaussian Process 形式的随机 Poisson 重建）生成后验分布，利用蒙特卡洛采样估计期望改进并在离散候选搜索或多起点梯度优化中选择下一视角；

**📊 数据集**

使用了 ModelNet10、Synthetic Pyramid、Truck‑ModelNet10、ShapeNet 等公开点云数据集进行分类与分割实验，Heat Diffusion 案例使用自制热扩散模拟；

**📈 对比分析**

与 FPS、基于不确定度的探索以及随机搜索等基线相比，实验显示在需要定位局部特征的分类与分割任务中，本方法在更少视角下即可达到更高的准确率或更快的收敛；在热扩散任务中，也能更早定位冷点，整体性能优于传统几何驱动策略；

**⚠️ 局限性**

局限性包括对 GP 先验的依赖，后验采样成本较高；对光照、纹理等外观信息不敏感；对大规模场景或实时控制的可扩展性待进一步验证。

---

## 509. Full-chip CMP modelling based on Fully Convolutional Network leveraging White Light Interferometry

**arXiv ID:** 2605.05062 | [PDF](https://arxiv.org/pdf/2605.05062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 510. Automatically Finding and Validating Unexpected Side-Effects of Interventions on Language Models

**arXiv ID:** 2605.05090 | [PDF](https://arxiv.org/pdf/2605.05090v1)

**作者:** Quintin Pope `[一作]`, Xiaoli Fern `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提供了使用ACL风格文件的LuaLaTeX/XeLaTeX示例

**💡 创新点**

创新点在于展示不同语言文本和引用的格式

**🔧 技术方法**

使用LuaLaTeX或XeLaTeX编译器

**📊 数据集**

无具体数据集

**📈 对比分析**

未进行方法比较或性能评估

**⚠️ 局限性**

仅为格式示例，缺乏实际实验或应用验证

---

## 511. A unified Benchmark for Multi-Frame Image Restoration under Severe Refractive Warping

**arXiv ID:** 2605.05079 | [PDF](https://arxiv.org/pdf/2605.05079v1)

**作者:** Maxim V. Shugaev `[一作]` (AeroVironment, Inc.), Mun Wai Lee `[通讯]` (AeroVironment, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一个涵盖从轻度到极端的光学几何失真（如大气湍流和水面折射）的全新视频失真修复基准，并通过实验验证了多种方法的性能；

**💡 创新点**

创新点在于①构建了覆盖多种波动类型和失真幅度的合成与实测数据集；②提出了基于扩散模型的 V-cache 方法，能够在极端失真下恢复图像；③统一了像素级与感知级评估指标，形成系统化的性能对比框架；

**🔧 技术方法**

使用的技术包括物理光线折射仿真、基于网格的非刚性对齐、DATUM 多帧递归网络、以及基于 CogVideoX-2B 的扩散模型配合 DINO 视觉特征的缓存与注意力适配；

**📊 数据集**

数据集包括实验室捕获的真实水面视频（LAB 数据）以及基于四类波形（海浪、浅水、正弦、波纹）与四级失真幅度的合成视频，总计 60 条组合；

**📈 对比分析**

与基线（第一帧、像素均值、网格变形、网格注册、DATUM）进行比较，V-cache 在中至极端失真下在 PSNR、LPIPS、DINO 等指标上均优于其它方法，尤其在极端失真时仍能保持高质量恢复；

**⚠️ 局限性**

局限包括 V-cache 在低失真场景下表现不佳（训练仅覆盖高幅度海浪）、对不同波形（如波纹）的恢复仍有挑战、以及对实时应用的推理速度与资源消耗需进一步优化。

---

## 512. SoK: Robustness in Large Language Models against Jailbreak Attacks

**arXiv ID:** 2605.05058 | [PDF](https://arxiv.org/pdf/2605.05058v1)

**作者:** Feiyue Xu `[一作]` (Shanghai Jiao Tong University), Shuo Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 13386 | [OpenAlex ID](https://openalex.org/A5100400173)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型（LLM）的越狱攻击与防御进行了系统化综述，构建了多维评估框架 Security Cube，并在该框架下对 13 种代表性攻击、5 种防御及 4 种评判器进行基准实验。

**💡 创新点**

创新点包括：①提出七维攻击指标、三维防御指标和四维评判指标的统一多维评估框架；②引入攻击稳定性、转移性、集中指数（CIPA）和破坏深度等新度量；③构建跨模型、跨攻击类型的综合基准，填补现有单一指标（如 ASR）评估的空白。

**🔧 技术方法**

使用了梯度优化、LLM 自动生成、隐藏状态监测、系统提示（system prompt）强化、后置过滤（post-filter）和多智能体评判等多种技术手段，对攻击流程、模型内部表示与输出进行全方位分析。

**📊 数据集**

采用公开越狱与安全评估数据集 HarmBench、JailbreakBench 等，结合多模型（GPT‑3.5‑Turbo、Qwen‑2.5‑7B、LLaMA‑3‑8B、Claude‑3.7‑Sonnet 等）进行实验。

**📈 对比分析**

通过 Security Cube 计算 ASR、稳定性 β、转移率 γ、CIPA、破坏深度 μ、耗时/Token 开销等指标，发现策略/多轮攻击平均 ASR 最高达 66% 且 CIPA 低，表明普适性强；防御中 Hidden State Guard 在大多数攻击下将 ASR 降至 0；多智能体评判在 F1 > 0.99 的同时与人类标注高度一致。

**⚠️ 局限性**

局限性：①缺乏能够同时兼顾高效、稳健、可解释的统一攻击/防御设计；②安全机制仍依赖模型的生成偏好，难以根除系统级漏洞；③评估依赖人工标注，成本高且难以大规模扩展；④对更复杂的多轮、跨模态或持续进化的攻击场景缺乏足够预测和防御研究。

---

## 513. Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism

**arXiv ID:** 2605.05049 | [PDF](https://arxiv.org/pdf/2605.05049v1)

**作者:** Sajal Dash `[一作]` (Oak Ridge National Laboratory), Feiyi Wang `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 1970 | [OpenAlex ID](https://openalex.org/A5101916963)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种结合流水线并行与专家并行的MoE训练框架Piper，利用资源建模自动挑选最优并行策略；

**💡 创新点**

核心创新在于将流水线并行应用于MoE内部，显著降低大规模all‑to‑all延迟并实现系统感知的混合并行；

**🔧 技术方法**

技术包括数学资源模型、微基准测评、Topology‑aware HALO all‑to‑all算法、专家迁移负载均衡、1F1B流水线调度等；

**📊 数据集**

主要在Frontier超级计算机上对Mixtral、DeepSeek、Qwen等SOTA MoE模型进行实验，未使用公开文本数据集；

**📈 对比分析**

与Tutel、DeepSpeed‑MoE、DeepSpeed‑TED、X‑MoE等现有框架对比，Piper在相同GPU数下实现2–3.6×的MFU和1.2–9×的bandwidth提升，训练吞吐量显著提升；

**⚠️ 局限性**

局限性包括对特定HPC拓扑（如Dragonfly）高度依赖，专家迁移虽然低成本但在极大专家数下仍有开销，且对非常细粒度专家模型的支持仍需进一步验证。

---

## 514. Preference-Based Self-Distillation: Beyond KL Matching via Reward Regularization

**arXiv ID:** 2605.05040 | [PDF](https://arxiv.org/pdf/2605.05040v1)

**作者:** Xin Yu `[一作]` (Pennsylvania State University), Qinzhen Guo `[通讯]` (TikTok)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于偏好学习的自监督对抗蒸馏框架PBSD，用于在不依赖外部教师模型的情况下，对同一模型在不同提示下进行对抗式训练，从而提升推理与工具使用的性能；

**💡 创新点**

核心创新在于将自蒸馏的目标从单纯的KL匹配转化为奖励正则化目标，即通过奖励加权的教师分布来获得更优的目标策略，并在此基础上采用偏好对比学习实现在线自蒸馏；

**🔧 技术方法**

使用了奖励正则化的KL目标、偏好学习（Bradley‑Terry 逻辑回归）、教师-学生对齐的margin、LoRA微调、FlashAttention、AdamW优化器等技术；

**📊 数据集**

实验数据集包括数学推理任务的OpenThoughts（AIME 2024/25、HMMT 2025）和工具使用任务的ToolAlpaca（4046训练样本、94测试样本）；

**📈 对比分析**

与SFT、GRPO、DAPO、OPSD、SDFT、SRPO等多种基线比较，PBSD在所有三种Qwen3规模（1.7B/4B/8B）上在数学推理与工具使用任务上均取得最优或第二优的平均准确率，且训练过程更稳定、token效率更高；

**⚠️ 局限性**

局限性包括对教师提示的依赖性（需设计合适的privileged context）、对奖励信号的隐式假设以及在更大规模模型或更复杂任务上的可扩展性尚未充分验证。

---

## 515. Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent Reinforcement Learning

**arXiv ID:** 2605.05020 | [PDF](https://arxiv.org/pdf/2605.05020v1)

**作者:** Shawn Ray `[一作]` (Carnegie Mellon University), Shawn Ray `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5123691197)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Graph-SND，将系统神经多样性（SND）的全对全平均替换为图边加权平均，既可作为局部多样性测量，又可作为稀疏估计器；

**💡 创新点**

创新点在于：①在任意图上定义稀疏聚合，恢复SND并降低计算复杂度；②通过随机采样得到无偏的Horvitz–Thompson估计和集中收敛；③利用随机正则图（expander）实现近线性稀疏近似并给出无条件概率误差界；

**🔧 技术方法**

使用图论（邻接图、通信图、kNN图）、随机采样与Horvitz–Thompson估计、Hoeffding/Serfling有限总体集中、正则图前向索引理论、以及多智能体强化学习中的高斯策略与Wasserstein距离；

**📊 数据集**

在VMAS多智能体目标导航和VMAS分散任务上进行实验，使用PettingZoo的MPE简单展开环境做非高斯验证；

**📈 对比分析**

与完整SND在多种场景（n=4至500、PPO训练、DiCo控制、expander稀疏化）比较，结果显示：Graph-SND在稀疏化（如Bernoulli-0.1）下能近似全SND、时间成本下降约10倍，expander图在Θ(nlog n)条边下误差<0.2%；

**⚠️ 局限性**

局限性包括：固定稀疏图可能隐藏非边的多样性；随机采样的收敛速度受边数影响；实验规模受限于PPO和独立策略，未覆盖集中式或更大离散动作任务；未来需研究自适应图、更多任务及更大规模验证。

---

## 516. Goedel Logics: On the Elimination of The Absoluteness Operator

**arXiv ID:** 2605.05016 | [PDF](https://arxiv.org/pdf/2605.05016v1)

**作者:** Matthias Baaz `[一作]` (TU Wien), Mariami Gamsakhurdia `[通讯]` (TU Wien)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了Gödel 逻辑中绝对性算子（Δ）的可消除性，并在受限语义下证明了命题层面的可消除性，进一步探讨了在第一阶逻辑中结合受限与可见语义实现 Δ 的逻辑等价消除。

**💡 创新点**

创新点在于提出了新的受限语义（所有原子不取值 1），使得 Δ 在命题层面可消除，并结合可见语义在一阶层面实现逻辑等价的消除；此外给出了链式正常形式、结构化翻译等技术，为消除提供了完整的理论框架。

**🔧 技术方法**

使用了受限语义（restrictive semantics）与可见语义（witnessed semantics）相结合的语义分析，链式正常形式（chain normal form）与结构化翻译（structural normal form）等技术；同时利用了 Gödel 集合的拓扑性质、递归不可分离性以及证明论中的推导规则。

**📊 数据集**

本文不使用具体数据集，而是基于逻辑语义与结构化归约的理论推导与证明。

**📈 对比分析**

由于论文主要是理论性质的证明，没有实验或性能评测；作者通过逻辑等价与可消除性证明展示了在受限语义下与标准语义的等价性，并证明了在一阶可见语义下可实现逻辑等价消除。

**⚠️ 局限性**

局限性包括：在标准一阶 Gödel 逻辑中 Δ 仍不可消除；受限语义不满足替换封闭性；在更一般的一阶情形（非有限、非可见、非前缀分片）中 Δ 的可消除性仍未解决；且对不满足条件的 Gödel 集合，其可见逻辑的递归可枚举性与消除性不保证。

---

## 517. Learned Neighbor Trust for Collaborative Deployment in Model-Agnostic Decentralized Learning

**arXiv ID:** 2605.05009 | [PDF](https://arxiv.org/pdf/2605.05009v1)

**作者:** Michael Lanier `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**通讯引用:** 5231 | [OpenAlex ID](https://openalex.org/A5038669899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种分布式学习框架LNTrust，节点通过仅发送查询和软预测，学习邻居的信任函数，用以在训练时进行辅助蒸馏并在推理时进行集成；

**💡 创新点**

创新点在于：① 在无服务器、模型无关的协议下，仅利用邻居的软预测构建紧凑的信任模型；② 信任模型既驱动训练中的门控蒸馏，又决定部署时的加权集成；③ 通过验证集自适应学习权重，兼顾负迁移抑制和置信过滤，显著提升协同效果；

**🔧 技术方法**

技术主要包括两阶段训练、基于六维关系特征的小型MLP信任模型、负迁移门与置信过滤、硬/软蒸馏损失、以及理论上的有限样本部署风险和控制扰动保证；

**📊 数据集**

实验数据集涵盖CIFAR-10、CIFAR-100和EuroSAT，并在三种拓扑（稀疏异构、稠密同构、地理）下验证；

**📈 对比分析**

与DML、DESA、FedPAE、Mean Teacher、D-PSGD、Gossip-FedAvg、DecDiff-VT等基线对比，LNTrust在稀疏异构/地理环境中提升约8%+准确率，稠密同构环境下与参数共享方法相当且通信效率最高；

**⚠️ 局限性**

局限性包括：邻域必须含有足够相似的类分布，否则协同收益有限；对抗攻击鲁棒性尚未研究；当节点标签分布完全不重叠时，方法无法获益。

---

## 518. Rollout Pass-Rate Control: Steering Binary-Reward RL Toward Its Most Informative Regime

**arXiv ID:** 2605.05112 | [PDF](https://arxiv.org/pdf/2605.05112v1)

**作者:** Tianshu Zhu `[一作]` (Baidu), Dou Shen `[通讯]` (Baidu)

**通讯引用:** 2680 | [OpenAlex ID](https://openalex.org/A5105366551)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出Prefix Sampling（PS），一种通过重放轨迹前缀并遮蔽旧动作的方式，将二元奖励强化学习中的“通过率”控制在50%附近，从而提升信息量并减少无效计算。

**💡 创新点**

创新点在于将通过率视为显式可控目标，构建双向前缀重放+自适应控制器，直接把偏斜的通过率组引导到信息最丰富的50%区间；同时实现了在状态化agent环境中通过执行重放恢复前缀状态并只对续写部分求梯度。

**🔧 技术方法**

技术包括：GRPO（Group Relative Policy Optimization）框架；前缀重放与前缀遮蔽（prefix masking）；基于指数移动平均的桶级自适应控制器；通过率目标与奖励熵、RLOO优势能量等量化指标。

**📊 数据集**

数据集：SWE-bench（软件工程任务）用于agentic RL；AIME 2025（单轮数学推理）用于验证机制。

**📈 对比分析**

与DeepSWE/GRPO++匹配基线相比，Prefix Sampling在Qwen3-14B、Qwen3-32B等模型上实现了2.01×/1.55×的端到端加速，且在相同步数下提升了4.7pp/5.4pp的Pass@1；在数学任务上同样获得1.23×/1.40×速度提升与10.8pp/7.3pp的精度提升。

**⚠️ 局限性**

局限性包括：需要针对每个桶设计前缀比例，可能在极端难度分布下失效；对环境状态恢复的实现复杂度较高；仅在二元奖励任务上有效，对连续奖励或多标签任务的适用性尚未验证。

---

## 519. Taming Outlier Tokens in Diffusion Transformers

**arXiv ID:** 2605.05206 | [PDF](https://arxiv.org/pdf/2605.05206v1)

**作者:** Xiaoyu Wu `[一作]` (Rice University), Chen Wei `[通讯]` (Rice University)

**通讯引用:** 20949 | [OpenAlex ID](https://openalex.org/A5100344556)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了扩散Transformer（DiT）中出现的异常高范数的“离群”token，并提出在编码器和去噪器两端都加入注册token的Dual-Stage Registers（DSR）方法，以抑制这些离群token对生成质量的负面影响。

**💡 创新点**

创新点在于首次系统地揭示了离群token不仅存在于预训练的Vision Transformer编码器中，还会在DiT的中间层产生；并提出统一的注册token干预方案，在不增加额外语义信息的情况下显著提升生成稳定性与质量。

**🔧 技术方法**

技术包括：测试时（test‑time）注册token、训练时的学习注册token、在Diffusion Transformer内部引入可学习的diffusion registers，以及对token范数的可视化与量化分析。

**📊 数据集**

数据集主要使用ImageNet‑1K（图像生成）和Scale‑RAE（大规模文本‑图像生成），并在这些数据集上进行训练与评估。

**📈 对比分析**

与原始RAE‑DiT和其他DiT架构（如SiT、JiT）对比，DSR在ImageNet-1K上将Fidelity（FID）从5.89降低至4.58，IS提升，Precision/Recall均有提升；在Scale‑RAE文本‑图像任务中，GenEval提升至46.6，DPG‑Bench提升至75.4。

**⚠️ 局限性**

局限性包括：方法主要针对Vision Transformer与Latent空间的DiT，未充分验证在纯像素空间或不同视觉编码器上的泛化；注册token数量与插入层次的最佳选择仍需经验性调优；对极端噪声水平或跨域数据的鲁棒性尚未深入研究。

---

## 520. LongSeeker: Elastic Context Orchestration for Long-Horizon Search Agents

**arXiv ID:** 2605.05191 | [PDF](https://arxiv.org/pdf/2605.05191v1)

**作者:** Yijun Lu `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9017 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Context-ReAct框架，结合五种原子上下文元操作（Skip、Compress、Rollback、Snippet、Delete）实现弹性上下文管理，并基于此训练LongSeeker长周期搜索代理。

**💡 创新点**

设计了可表达完全且高效的五种上下文元操作，允许代理在推理过程中主动决定何时何地压缩、保留、删除或回滚历史，从而在保持必要细节的同时显著降低上下文噪声和生成成本。

**🔧 技术方法**

在ReAct基础上实现自回归生成的四字段输出（推理、元操作、动机、工具调用），使用Qwen3‑30B‑A3B进行SFT微调，并通过自定义压缩与片段提取实现上下文重塑。

**📊 数据集**

使用10k条合成搜索轨迹（来自OpenSeeker），以及BrowseComp、BrowseComp‑ZH、xbench、GAIA等公开基准数据集进行训练与评测。

**📈 对比分析**

在BrowseComp等基准上与多种30B规模搜索代理与大型工具模型对比，LongSeeker在BrowseComp获得61.5%、BrowseComp‑ZH 62.5%，分别显著高于Tongyi DeepResearch（43.2%、46.7%）和AgentFold（36.2%、47.3%），在xbench 78.0、GAIA‑text 77.7的表现也保持领先。

**⚠️ 局限性**

目前仅使用SFT进行训练，缺乏基于强化学习的元操作探索；压缩摘要仍可能引入抽象误差；框架对多模态或非搜索任务的通用性尚待进一步验证。

---

## 521. LoViF 2026 The First Challenge on Holistic Quality Assessment for 4D World Model (PhyScore)

**arXiv ID:** 2605.05187 | [PDF](https://arxiv.org/pdf/2605.05187v1)

**作者:** Wei Luo `[一作]` (University Of Science And Technology Of China), Huan Zheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出LoViF 2026 PhyScore挑战，构建涵盖视觉质量、物理真实感、条件对齐与时空一致性四维度评估，并要求检测物理异常时戳。

**💡 创新点**

首次将物理合理性与异常定位纳入视频质量评估，构造统一的多维度评分与检测框架，并公开大规模基准数据。

**🔧 技术方法**

采用多分支深度网络（ConvNeXt、Swin、SlowFast、VideoMAE）结合大规模多模态模型、光流与物理感知模块、集成学习与预训练策略。

**📊 数据集**

使用由7个代表性世界模型生成的1,554段视频组成的 LoViF 2026 数据集，涵盖3种输入模态、26个物理与非物理场景。

**📈 对比分析**

在 CodaBench 上对 5 支参赛队伍的预测进行 SRCC/PLCC 与时间戳 IoU 组合评估，SJTU‑MM 获得最高 Final Score 0.532，排名第一；IHNI 其次。

**⚠️ 局限性**

数据集仅覆盖有限数量模型，难以验证方法在更广泛生成器上的泛化；评估主要基于相关性与 IoU，缺乏更严格的因果或真实物理验证。

---

## 522. Understanding In-Context Learning for Nonlinear Regression with Transformers: Attention as Featurizer

**arXiv ID:** 2605.05176 | [PDF](https://arxiv.org/pdf/2605.05176v1)

**作者:** Alexander Hsu `[一作]` (Purdue University), Rongjie Lai `[通讯]` (Purdue University)

**通讯引用:** 1800 | [OpenAlex ID](https://openalex.org/A5088277298)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

构造基于注意力机制的 Transformer，实现了在上下文学习中使用多项式或样条基函数的非线性回归，并给出了有限样本的泛化误差上界；

**💡 创新点**

创新点在于利用 Attention 的 Interaction Lemma 将特征映射（多项式、B‑spline）精确地编码进 Transformer，从而在单层网络中完成特征构造和线性系统求解，且深度与精度无关；

**🔧 技术方法**

采用 ReLU 触发的注意力实现特征构造，线性注意力求解最小二乘，再结合残差块、位置编码和解码器；

**📊 数据集**

使用合成数据：随机生成高斯分布系数的多项式（或等距 B‑spline）在 [-1,1] 上采样上下文点和查询点；

**📈 对比分析**

与纯线性 Transformer 和 Softmax Transformer 进行对比；实验显示理论模型与 Softmax 模型误差相近，线性 Transformer 性能略差；训练样本规模 L 的误差收敛速率近似 1/L，符合理论预测的 1/√L 趋势；

**⚠️ 局限性**

局限性包括：需要随上下文长度 n 线性扩展注意力头数，构造复杂且效率低；实验仅基于合成数据，实际任务性能未知；理论假设特征矩阵条件数良好、特征维数固定等理想条件。

---

## 523. A Closed-Form Dual-Barrier CBF Safety Filter for Holonomic Robots on Incrementally Built Occupancy Grid Maps

**arXiv ID:** 2605.05182 | [PDF](https://arxiv.org/pdf/2605.05182v1)

**作者:** Himanshu Paudel `[一作]`, Sanjay Neupane `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种双障碍控制障碍函数（CBF）安全过滤器，用于在逐步构建的占用网格地图上对全向机器人进行实时的安全速度控制。

**💡 创新点**

创新点在于同时施加两个互补的安全约束，确保机器人避免已映射的障碍物，并不进入未探索的区域，同时提供了闭式解，减少了计算负担。

**🔧 技术方法**

使用了控制障碍函数（CBF）、签名距离场（SDF）和自适应增益调度等技术。

**📊 数据集**

使用了逐步构建的占用网格地图，结合了视觉惯性SLAM生成的地图数据。

**📈 对比分析**

与基线的RRT*-APF控制器相比，提出的CBF过滤器在探索效率上显著提高，最终探索面积为81.68平方米，而基线为56.35平方米，且在多个实验中未发生障碍物接触。

**⚠️ 局限性**

局限性包括当前实现仅在2D平面上操作，扩展到3D需要更复杂的计算，且对动态障碍物的处理尚未集成。

---

## 524. Toward a Risk Assessment Framework for Institutional DeFi: A Nine-Dimension Approach

**arXiv ID:** 2605.05145 | [PDF](https://arxiv.org/pdf/2605.05145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 525. When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning

**arXiv ID:** 2605.05172 | [PDF](https://arxiv.org/pdf/2605.05172v1)

**作者:** Lakshita Dodeja `[一作]` (Robotics and AI Institute), Thomas Weng `[通讯]` (Robotics and AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在机器人学习中结合行为克隆与强化学习，提出Q-Estimation和Q-Gating方法：先用有限在线采样估计预训练BC策略的Q值，再在在线RL阶段通过比较BC Q和可学习RL Q来动态选择动作，从而在保持BC优良行为的同时实现性能提升。

**💡 创新点**

创新点在于仅利用BC策略的动作对数概率与熵，解析式推导出其Q值并作为参考；再通过冻结的BC-Q与可学习的RL-Q进行门控，使得在线RL既能快速收敛又能安全地改进，而不需离线数据或额外标注。

**🔧 技术方法**

使用的核心技术包括Soft Actor-Critic（SAC）强化学习框架、Gaussian/GMM行为克隆策略、熵正则化、基于对数概率与熵的Q估计公式、Q门控机制以及异步actor-learner架构。

**📊 数据集**

实验基准涵盖D4RL和Robomimic仿真任务，真实世界实验使用Franka Panda+Robotiq 2F-85机器人，并通过SpaceMouse收集的50-100条成功演示数据。

**📈 对比分析**

与CQL、CalQL、WSRL、RLPD、IBRL等离线到在线基线对比，仿真任务中成功率提升至80-100%，在真实机器人中仅1-2小时即可将BC成功率提升至100%，相较基线显著加速且更安全。

**⚠️ 局限性**

主要局限在于需要BC策略能够输出动作对数概率与熵，限制了对非概率型策略（如扩散模型）的适用性；对软最优假设的依赖以及极大动作空间或长期稀疏奖励环境下的探索效率仍需改进。

---

## 526. PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World

**arXiv ID:** 2605.05163 | [PDF](https://arxiv.org/pdf/2605.05163v1)

**作者:** Yunhan Yang `[一作]` (HKU), Xihui Liu `[通讯]` (HKU)

**通讯引用:** 3906 | [OpenAlex ID](https://openalex.org/A5027234036)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 PhysForge 框架，能从单张图片生成具备完整物理属性和可交互关节的 3D 资产。

**💡 创新点**

核心创新包括两阶段规划+实现模式、VLM 生成层次化物理蓝图以及 KineVoxel 注入的扩散模型，以同步合成几何与关节参数。

**🔧 技术方法**

结合 Qwen2.5‑VL 作为物理规划器、基于 TRELLIS 的体素编码、KineVoxel 注入的扩散生成器，并采用 Conditional Flow Matching 损失进行训练。

**📊 数据集**

构建了 PhysDB 大规模 150k 资产数据集，涵盖四层物理注释，并利用 Objaverse、PartNet‑Mobility、Infinite‑Mobility 等源数据进行补充。

**📈 对比分析**

在 PartObjaverse‑Tiny、PhysXNet 以及新采样的 PhysDB、PartNet‑Mobility 测试集上，与 OmniPart、PartField、PhysXGen、TRELLIS 等基线比较，显示在结构规划、几何质量、关节参数准确性等指标上均达或超越 SOTA。

**⚠️ 局限性**

局限性包括对高维连续关节参数的依赖仍需进一步提升精度，且在极其复杂或稀有类别下的生成鲁棒性与多视角一致性仍有改进空间。

---

## 527. Transformed Latent Variable Multi-Output Gaussian Processes

**arXiv ID:** 2605.05133 | [PDF](https://arxiv.org/pdf/2605.05133v1)

**作者:** Xiaoyu Jiang `[一作]` (University of Manchester), Mauricio A Álvarez `[通讯]` (University of Manchester)

**通讯引用:** 2513 | [OpenAlex ID](https://openalex.org/A5007248971)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种可扩展的多输出高斯过程框架T-LVMOGP，通过在学习嵌入空间中构造Lipschitz正则化的深度核，将多输出问题转换为标量GP，从而显著提升大规模高维输出建模的可行性。

**💡 创新点**

核心创新在于将每个输出映射到共享嵌入空间并利用残差连接+谱正则化的神经网络学习可微、Lipschitz连续的特征变换，使得跨输出协方差由基核在嵌入空间中自然产生；同时引入稀疏变分推断与更紧的ELBO，以保持高表达能力与计算效率。

**🔧 技术方法**

使用了残差连接神经网络（RCNN）+谱正则化的深度核学习、稀疏变分GP（SVGP）与更紧的变分界限、重参数化采样、随机小批量训练以及零膨胀负二项等非高斯似然。

**📊 数据集**

在多种真实数据集上验证：EEG时间序列（7电极）、SARCOS机器人逆动力学（21维输入，7输出）、气候模拟（ERA5、Copernicus Marine，输出数至21,679）以及空间转录组学（10,000基因×4,352空间位置）。

**📈 对比分析**

与多种基线（Ind-GP、SV-LMC、SGPRN、OILMM、G-MOGP、GS-LVMOGP）进行对比，T-LVMOGP在MSE、NLL、训练时间等指标上普遍优于对手，尤其在大输出维度下显著提升预测精度与计算速度。

**⚠️ 局限性**

局限性包括：对输出间后验耦合仅采用因式分解变分分布，可能限制跨输出的后验捕捉；以及对谱正则化超参数（SN-UB）敏感，需要经验调优。

---

## 528. Deterministic identification for Bernoulli channels and related channels with continuous input

**arXiv ID:** 2605.05168 | [PDF](https://arxiv.org/pdf/2605.05168v1)

**作者:** Pau Colomer `[一作]` (Technische Universität München), Andreas Winter `[通讯]` (Universität zu Köln)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于多层几何构造的确定性识别码，完成了伯努利信道以及更广泛的连续输入离散输出信道的DI容量从1/4提升到1/2的闭合。

**💡 创新点**

创新点在于将AWGN的“银河”代码几何思想迁移到输入相关噪声的伯努利与泊松信道，并通过可旋转的球面码点与立方体交集证明了容量上界可达。

**🔧 技术方法**

使用了高维几何聚类、投影集中度、Hoeffding/ Chernoff不等式以及多层球面点阵的排布方法。

**📊 数据集**

并未依赖具体数据集，而是在理论上对所有满足连续输入、离散输出且图像维度为1的通道通用。

**📈 对比分析**

通过构造可实现的DI码并证明误差概率随块长下降，实验仿真与已有典型性方法相比在容量和误差指数上实现了最优或接近最优。

**⚠️ 局限性**

限制在于仅适用于图像维度为1的信道，且对大误差指数的子指数误差衰减区间有限，无法覆盖更高维或非连通图像。

---

## 529. Syn4D: A Multiview Synthetic 4D Dataset

**arXiv ID:** 2605.05207 | [PDF](https://arxiv.org/pdf/2605.05207v1)

**作者:** Zeren Jiang `[一作]` (University of Oxford), Andrea Vedaldi `[通讯]` (University of Oxford)

**通讯引用:** 71958 | [OpenAlex ID](https://openalex.org/A5060511349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出Syn4D多视角合成4D数据集，提供密集3D跟踪、相机运动、深度、人体姿态等注释，并基于此训练多任务模型。

**💡 创新点**

①首次提供多视角动态场景的密集3D跟踪注释；②用像素对齐的重心映射高效存储跟踪；③构建几何感知多视角扩散模型实现几何一致的新视角视频合成。

**🔧 技术方法**

基于Unreal Engine 5渲染，利用Objaverse、Bedlam2等资产，使用像素重投影与三角网格重心计算；训练基于ReCamMaster的扩散网络，并用FVD、CLIP‑V等指标评估。

**📊 数据集**

Syn4D数据集（30个Unreal场景、1674个动画物体、585人类模型），对比Kubric、SYNTHIA、SEED4D等公开数据集进行实验。

**📈 对比分析**

在4D重建、3D跟踪、相机姿态估计、多视角重建、视频深度估计以及人体姿态估计等任务上，使用标准指标与现有方法对比，Syn4D训练的模型均取得显著提升，几何感知扩散模型在视觉与几何质量上优于基线。

**⚠️ 局限性**

依赖合成数据可能导致真实世界域差，存储和渲染开销大，且仅包含单人或有限类别动态物体，缺乏更大范围的多主体交互。

---

## 530. SILC: Lookahead Caching for Short-form Video Delivery Systems

**arXiv ID:** 2605.05188 | [PDF](https://arxiv.org/pdf/2605.05188v1)

**作者:** Maleeha Masood `[一作]` (University of Illinois Urbana Champaign), Indranil Gupta `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了利用短视频平台推送式推荐系统中Manifest文件提供的前瞻信息，构建了一套面向CDN的缓存系统SILC，结合Least Lookahead Frequency (LLF)淘汰策略和在线视频重排技术，目标是降低CDN中转流量(midgress)并提升缓存命中率。

**💡 创新点**

①提出LLF淘汰策略，直接利用Manifest中的有限前瞻请求计数而非历史频率；②在保持推荐内容不变的前提下，对Manifest进行在线重排以最大化跨用户时空重叠；③通过用户A/B测试验证重排对体验无显著影响，形成首个同时兼顾系统成本和用户体验的短视频CDN优化方案。

**🔧 技术方法**

使用前瞻缓存淘汰、在线重排、仿真与实验环境（云实验+离线模拟）、Pareto分布建模、与10种主流缓存算法（LRU、LFU、GDSF、LeCaR、LRB、LHD、AdaptSize等）进行对比，并通过用户行为轨迹模拟实现千人级实验。

**📊 数据集**

基于100名真实TikTok用户的数据捐赠，收集到约3.9M条观看记录，包含2.65M个视频的元数据（播放量、点赞数等），以及170–450天的浏览历史。

**📈 对比分析**

在10,000用户、10GB/服务器（总100GB）云实验与离线模拟中，SILC相较于最优学习型和启发式基线，将字节缺失率/中转流量降低11.1%–111%，比最优Belady实现至少12.3%更优；缓存命中率提升至约35%字节命中。

**⚠️ 局限性**

局限性包括：仅针对TikTok的Manifest实现，其他短视频平台需进一步验证；对推荐算法未知，实验基于有限样本；前瞻窗口有限，LLF对长时间跨度请求的准确性不高；重排假设用户不受播放顺序影响，未在更大规模或多平台环境中验证。

---

## 531. Estimating the expected output of wide random MLPs more efficiently than sampling

**arXiv ID:** 2605.05179 | [PDF](https://arxiv.org/pdf/2605.05179v1)

**作者:** Wilson Wu `[一作]` (Alignment Research Center), Paul Christiano `[通讯]` (Alignment Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出一种无样本的累积量传播方法，用来估计宽随机多层感知器在高斯输入下的期望输出；

**💡 创新点**

创新点在于将高阶累积量与Hermite展开相结合，并利用图形求和公式构造可微的近似，显著降低了对Monte Carlo采样的计算需求，尤其在罕见事件概率估计上优势明显；

**🔧 技术方法**

主要技术包括累积量传播、Hermite多项式展开、图形求和公式、张量分解与因式化（factorized）实现以及能量收敛的Power累积量修正；

**📊 数据集**

实验使用随机初始化的MLP，输入为标准正态分布，激活函数涵盖ReLU、GELU、tanh，网络宽度从4到256、深度从2到12，随机种子共计5；

**📈 对比分析**

与Monte Carlo采样比较时，因子化版本在相同FLOPs下可将均方误差降低约10^2–10^3倍；在低概率事件估计中，误差相对较采样小10倍以上；

**⚠️ 局限性**

局限性在于方法仅针对宽随机初始化网络，深度需保持常数；对已训练或非高斯输入、非独立权重的网络缺乏理论保证，且实现中仍需进一步优化以减少运行时开销。

---

## 532. Private Structured-Subset Retrieval

**arXiv ID:** 2605.05160 | [PDF](https://arxiv.org/pdf/2605.05160v1)

**作者:** Maha Issa `[一作]` (Santa Clara University), Anoosheh Heidarzadeh `[通讯]` (Santa Clara University)

**通讯引用:** 552 | [OpenAlex ID](https://openalex.org/A5006920802)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了Private Structured‑Subset Retrieval（PSSR）问题，研究在已知结构化需求子集族下的隐私检索；

**💡 创新点**

通过构造优化框架和对称性简化，实现了比传统MPIR更高的检索速率和更低的分包化水平；

**🔧 技术方法**

利用信息理论极大值逆推、线性组合与整数线性规划（ILP/Lp）以及符号对称性归约技术；

**📊 数据集**

论文仅为理论分析和算法设计，并未使用具体真实数据集；

**📈 对比分析**

与现有MPIR方案对比，在示例与连续块需求族上，PSSR方案在相同服务器数下检索速率更优，且子包化量显著减少；

**⚠️ 局限性**

逆推与可达性界限尚未完全匹配，且仅覆盖平衡{0,1}‑线性方案，未考虑非均等消息长度、侧信息或更广泛的编码设计。

---

## 533. PSK at SemEval-2026 Task 9: Multilingual Polarization Detection Using Ensemble Gemma Models with Synthetic Data Augmentation

**arXiv ID:** 2605.05159 | [PDF](https://arxiv.org/pdf/2605.05159v1)

**作者:** Srikar Kashyap Pulipaka `[一作]` `[通讯]` (Independent Researcher), Srikar Kashyap Pulipaka (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于Gemma 3模型、LoRA微调、GPT-4o-mini合成数据及阈值调优的多语言偏见检测系统，参加SemEval-2026 Task 9。

**💡 创新点**

创新点在于为每种语言单独微调Gemma 3，并融合多种合成数据策略（直接生成、改写、对照对）以及加权投票集成，显著提升跨22种语言的泛化性能。

**🔧 技术方法**

使用的技术包括LoRA参数高效微调、GPT-4o-mini生成多样化合成数据、embedding去重过滤、阈值调优与12B/27B模型加权集成。

**📊 数据集**

使用POLAR 22语言二分类数据集（训练集约1,700–7,000条样本），并在训练集上按比例生成约1,000条合成样本。

**📈 对比分析**

与XLM‑RoBERTa、Qwen3等模型对比，Gemma在开发集表现最佳且在测试集保持稳健，最终宏F1为0.811，排名第二，在多语言中多次夺冠。

**⚠️ 局限性**

局限性包括需为每种语言单独训练、合成数据质量受限、阈值调优基于小验证集、合成比例未系统评估，且对训练集中缺失主题的语言表现较差。

---

## 534. Superposition Is Not Necessary: A Mechanistic Interpretability Analysis of Transformer Representations for Time Series Forecasting

**arXiv ID:** 2605.05151 | [PDF](https://arxiv.org/pdf/2605.05151v1)

**作者:** Alper Yıldırım `[一作]` `[通讯]`, Alper Yıldırım

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文使用稀疏自编码器（SAE）对PatchTST Transformer的后GELU FFN激活进行分析，探究其内部是否存在超位置压缩。

**💡 创新点**

创新点在于首次将SAE用于时序Transformer的可解释性研究，证明标准预测基准并不需要强superposition，进而解释了简单线性模型（如DLinear、FITS）在这些任务上仍能保持竞争力。

**🔧 技术方法**

技术包括：PatchTST单层Transformer、RoPE位置编码、RMSNorm规范化、稀疏自编码器（字典大小0.5×、1×、4×）、干预实验与字典扩展评估。

**📊 数据集**

使用的公开长序列预测数据集共八个：Weather、Electricity、Traffic、Exchange、ETTh1、ETTh2、ETTm1、ETTm2。

**📈 对比分析**

与多层PatchTST、DLinear、FITS等模型比较，单层模型在所有基准上表现相当；字典扩展导致平均MSE提升不到0.3%，对预测结果几乎无影响；对最活跃的latent进行放大干预，预测误差仅提升约0.03 MAE。

**⚠️ 局限性**

局限性：只分析了后GELU FFN激活，未考虑注意力或残差流中的潜在superposition；仅针对PatchTST，结果可能不适用于所有时序Transformer；SAE训练参数对结果可能有一定敏感性。

---

## 535. Executable World Models for ARC-AGI-3 in the Era of Coding Agents

**arXiv ID:** 2605.05138 | [PDF](https://arxiv.org/pdf/2605.05138v1)

**作者:** Sergey Rodionov `[一作]` `[通讯]` (SingularityNET), Sergey Rodionov (SingularityNET)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于可执行 Python 世界模型的 ARC‑AGI‑3 代理，并在 25 个公开游戏上进行评估；

**💡 创新点**

通过编码代理实现模型验证、重构与简化，提出了无游戏专用代码的通用基线，并用 MDL‑类简化偏好驱动模型迭代；

**🔧 技术方法**

使用 Codex CLI（大型语言模型）、可执行世界模型、验证器、规划器与计划执行器，并结合代码重构技术；

**📊 数据集**

使用 ARC‑AGI‑3 公开游戏数据集，共 25 个游戏（29 次独立跑），每次从新启动且不共享状态；

**📈 对比分析**

采用相对人类动作效率（RHAE）作为评价指标，平均 RHAE 32.58%，完成 7 个游戏，6 个游戏 RHAE 超过 75%；

**⚠️ 局限性**

性能波动大，模型学习与规划分离导致规划效果差，缺乏竞争假设跟踪和强大规划器，整体效率低。

---

## 536. Adaptive Policy Selection and Fine-Tuning under Interaction Budgets for Offline-to-Online Reinforcement Learning

**arXiv ID:** 2605.05123 | [PDF](https://arxiv.org/pdf/2605.05123v1)

**作者:** Alper Kamil Bozkurt `[一作]` (Virginia Commonwealth University), Yuichi Motai `[通讯]` (Virginia Commonwealth University)

**通讯引用:** 1603 | [OpenAlex ID](https://openalex.org/A5061499121)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自适应的 O2O‑RL 框架，联合进行策略选择与在线微调，并通过 UCB 机制在有限的交互预算下动态切换候选策略。

**💡 创新点**

创新点在于将在线微调与自适应策略选择结合，使用 AR(2)-ARCH(1) 价值预测模型和 UCB 置信上限来决定何时继续微调或切换策略，避免了传统的全量评估或预先选择策略的缺陷。

**🔧 技术方法**

采用离线强化学习（多算法/超参数）、离线策略评估（FQE）、AR(2)-ARCH(1) 价值预测、UCB 决策、优先队列管理策略，以及 PyBullet + d3rlpy 等实现库。

**📊 数据集**

在 D4RL 的四个基准任务（hopper、halfcheetah、walker2d、ant）上使用四种数据集（random、medium、medium‑replay、medium‑expert），共计 16 个数据集进行评估。

**📈 对比分析**

与多种基线（基于 OPE 的预选、均匀在线评估、全量微调等）对比，实验显示在 160K 与 320K 交互预算下，该方法在所有环境和数据集上均获得更高的平均回报，显著降低了 regret。

**⚠️ 局限性**

局限性包括：每轮微调后仍需大量在线评估；对 AR/ARCH 模型假设敏感；缺乏严格理论保证；在候选策略数目极大或交互预算极小的场景下可能效果下降。

---

## 537. Beyond Semantics: An Evidential Reasoning-Aware Multi-View Learning Framework for Trustworthy Mental Health Prediction

**arXiv ID:** 2605.05121 | [PDF](https://arxiv.org/pdf/2605.05121v1)

**作者:** Yucheng Ruan `[一作]` (National University of Singapore), Mengling Feng `[通讯]` (National University of Singapore)

**通讯引用:** 12566 | [OpenAlex ID](https://openalex.org/A5022222926)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于推理的多视图学习框架，通过将编码器视图（语义信息）和解码器视图（高层推理信息）进行证据融合，实现可信的心理健康预测。

**💡 创新点**

创新点在于将主观逻辑与Dempster‑Shafer理论相结合，显式建模各视图的不确定性，并采用证据融合策略在多视图之间平衡信息，同时提供可解释的推理信号。

**🔧 技术方法**

使用的技术包括BERT（语义视图）、LLAMA‑3‑8B‑Instruct（生成推理视图）、主观逻辑Dirichlet建模、Dempster‑Shafer证据融合、Evidential Deep Learning、以及多任务训练和噪声鲁棒实验。

**📊 数据集**

实验数据集包括 Dreaddit、SDCNL 与 DepSeverity 三个真实心理健康文本数据集。

**📈 对比分析**

与 Dropout、Ensemble、UA、EDL 等基线方法对比，在三大数据集上实现了最高的准确率、AUROC 与 AUPRC，并在不确定性估计上取得最高 AUROC，显示出更优的性能与可靠性。

**⚠️ 局限性**

局限性包括对预训练 LLM 生成推理质量的依赖，缺乏多模态或更大规模的数据支持，且在极端噪声或跨人群迁移场景下的鲁棒性与泛化能力仍需进一步验证。

---

## 538. Physiologically Grounded Driver Behavior Classification: SHAP-Driven Elite Feature Selection and Hybrid Gradient Boosting for Multimodal Physiological Signals

**arXiv ID:** 2605.05120 | [PDF](https://arxiv.org/pdf/2605.05120v1)

**作者:** Sahar Askari `[一作]` (Shiraz University), Saeid Sanei `[通讯]` (VinUniversity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种可解释且可扩展的多模态生理信号驱动行为解码框架。

**💡 创新点**

结合 ICA 去噪、SHAP 版特征精简、XGBoost-LightGBM 加权投票 ensemble，并对各模态贡献进行 SHAP 解释。

**🔧 技术方法**

使用 ICA 除噪、时频特征提取、SHAP elite 选特征、Optuna 贝叶斯调参、XGBoost+LightGBM 加权投票、SHAP 可解释性分析。

**📊 数据集**

利用 MPDB 大规模同步 EEG/EMG/GSR 数据集。

**📈 对比分析**

与传统机器学习基线、单模态/双模态模型对比，最终在测试集上取得 80.91% 的准确率和 0.79 的宏 F1 分数，显著优于之前最佳 74.4% 的深度学习模型。

**⚠️ 局限性**

对实时部署的推理速度与在真实驾驶环境下的鲁棒性尚未验证，且对细微动作（如加速）的识别仍受限。

---

## 539. On the Hardness of Junking LLMs

**arXiv ID:** 2605.05116 | [PDF](https://arxiv.org/pdf/2605.05116v1)

**作者:** Marco Rando `[一作]` (Universite Cote d'Azur), Samuel Vaiter `[通讯]` (Universite Cote d'Azur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在已对齐的LLM中寻找不含语义信息的“自然后门”序列（即非语义输入能诱导模型产生有害输出）的可行性与难度，提出了“junking”问题的形式化框架，并用贪心随机搜索（GRS）验证其可解性。

**💡 创新点**

创新点在于：①首次将自然后门视为一个纯离散优化任务而非在已有提示下微调；②提供了最小化式目标函数与相应的贪心随机搜索基线；③通过对比随机搜索与传统基于语义模板的攻击，量化了语义结构对攻击难度的影响；④通过困惑度分析证明所发现的序列属于模型训练分布的低概率区域，进一步支持自然后门的存在假设。

**🔧 技术方法**

技术方法包括：离散序列优化（贪心随机搜索）；对齐LLM的提示模板插入与自回归生成；使用外部LLM作为评判器得到成功/连贯评分；计算序列困惑度（perplexity）做统计分析。

**📊 数据集**

使用的数据集为 AdvBench 中的 50 条有害行为目标；实验模型包括 LLaMA‑2‑Chat‑7B、Gemma‑7B、Qwen‑2.5‑7B、Mistral‑7B；自然文本用 HuggingFace 上 Puffin 训练集作为基准困惑度。

**📈 对比分析**

与基于前缀/后缀且附加语义模板的攻击进行对比。结果显示：在相同的评估指标（攻击成功率 ASR、编辑距离、困惑度）下，GRS 能在约 1–3×10⁴ 次评估内实现 52%–90% 的 ASR，但所需评估次数明显高于传统攻击；编辑距离低，表明生成的前缀与目标高度一致；自然后门序列的困惑度显著高于普通文本，显示其稀有性。

**⚠️ 局限性**

局限性包括：①仅在 7B 规模的公开模型上验证，未覆盖更大或更强防御的模型；②搜索耗时高（需要数千到数万次评估）；③贪心随机搜索是最简单的基线，可能低估更复杂优化方法的效果；④对模型对齐策略的变化敏感，结果可能随对齐强度不同而显著变化。

---

## 540. D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models

**arXiv ID:** 2605.05204 | [PDF](https://arxiv.org/pdf/2605.05204v1)

**作者:** Dengyang Jiang `[一作]` (Hong Kong University of Science and Technology), Steven Hoi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于自我蒸馏的少步扩散模型连续微调框架，利用LLM/VLM编码器的上下文能力，使同一模型在文本条件下做学生，在多模态条件下做教师，直接在自身轨迹上进行对齐训练，避免离线目标与推理轨迹不匹配问题；

**💡 创新点**

核心创新是将on‑policy self‑distillation应用于扩散模型，借助编码器的内在上下文能力实现同机学习，无需外部奖励或额外模块；

**🔧 技术方法**

技术组合包括flow‑matching扩散、少步采样（4‑8步）、自我蒸馏、EMA教师、LoRA微调、以及多模态条件编码；

**📊 数据集**

使用Z‑Image‑Turbo 6B 与 FLUX.2‑klein 4B 作为基模型；实验数据集包含 DreamBooth、少量风格化图像集及动漫高质量数据集；

**📈 对比分析**

与 Vanilla SFT、SFT+LoRA、Dreambooth、PSO 等基线对比，实验表明在 DINO‑D、LPIPS‑D、VLM‑J、CLIP‑S、Quality‑S 等指标上均优于基线，同时保持或提升少步推理质量，并实现新概念/风格的学习与泛化；

**⚠️ 局限性**

局限性包括：训练需要约 4 倍 FLOPs 与 2 倍时间成本；对教师多模态生成质量高度依赖，若教师产生不一致则训练失败；

---

## 541. Implicit Representations of Grammaticality in Language Models

**arXiv ID:** 2605.05197 | [PDF](https://arxiv.org/pdf/2605.05197v1)

**作者:** Yingshan Susan Wang `[一作]` (Massachusetts Institute of Technology), Yoon Kim `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22268 | [OpenAlex ID](https://openalex.org/A5100693798)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种线性探针方法，用于从预训练语言模型的隐藏状态中提取语法正确性（grammaticality）的表示，并通过合成无语法错误句子训练探针，验证其能否对真实人类标注的可接受性数据集做出准确判断。

**💡 创新点**

创新点在于：1) 用大规模无监督合成数据（插入/删除/局部洗牌噪声）训练探针，而不依赖人工标注；2) 证明探针能在多语言零样本设置下泛化，显示语法信息在不同语言间的普适性；3) 对比探针与传统基于字符串概率的判别方法，发现探针在语法判断上更强，而在语义合理性评估上则弱于概率，表明探针捕获的是更纯粹的句法信息。

**🔧 技术方法**

技术手段包括：对预训练大模型（OLMo、Llama、Gemma等）隐藏层进行线性回归或LASSO探针训练；使用长度归一化的累积log概率作为基线；通过Spearman/ Pearson相关性评估探针与概率的关联；用ridge回归验证隐藏状态可恢复log概率。

**📊 数据集**

使用的数据集主要包括：合成训练集（从Penn Treebank与Project Gutenberg采样并噪声化得到的50k句子对）；语法可接受性评测集（English: BLiMP、CoLA、SyntaxGym；跨语言：ScaLA(sv)、BLiMP-NL、ItaCoLA、RuCoLA、JCoLA、SLING）；语义合理性评测集（如各类可解释的合理/不合理句子）。

**📈 对比分析**

比较方法：在最小对比和单句可接受性评测中分别用ACC和AUC指标评估；与LM字符串概率基线对比。实验结果显示：探针在BLiMP、CoLA、SyntaxGym等英语评测集上均优于字符串概率；在跨语言评测中，英语训练的探针在大多数语言上也能匹配或超过概率基线；在语义合理性评测中，探针性能显著低于概率，说明其更专注句法。

**⚠️ 局限性**

局限性包括：1) 合成训练集可能包含原始语法错误或产生语义不合理句子，影响标注准确性；2) 仅考察模型最终检查点，未分析学习动态；3) 对跨语言泛化的解释主要基于英语在预训练数据中的占比，未进一步探究多语言模型的内在机制；4) 研究聚焦自然语言语法，未扩展到形式语言或更深层次的语法理论。

---

## 542. Design Conductor 2.0: An agent builds a TurboQuant inference accelerator in 80 hours

**arXiv ID:** 2605.05170 | [PDF](https://arxiv.org/pdf/2605.05170v1)

**作者:** The Verkor Team `[一作]` (Verkor), David Chin `[通讯]` (Verkor)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Design Conductor 2.0，利用多代理系统从概念到布局自动完成硬件设计，并在FPGA上实现TurboQuant加速器VerTQ、AES加密核心、FP32加/减单元以及全路速Allreduce等设计。

**💡 创新点**

首次实现端到端硬件设计闭环自动化，能自我推理并调整RTL与物理实现以满足时序与资源约束。

**🔧 技术方法**

使用前沿LLM模型、Python vLLM/FlashAttention仿真、Vivado物理实现、OpenROAD/ASIC流程、FPGA DSP/FFT及定制FP16/FP32算子。

**📊 数据集**

使用Qwen3-4B LLM权重与推理样例、TurboQuant算法引用、NIST AES测试向量、BF16/FP32 Allreduce测试包等数据。

**📈 对比分析**

通过在Xilinx XCVU29P-3 FPGA实现VerTQ，达到125 MHz、1.9 M LUT、1.5 K DSP，KV缓存压缩4.3×、乘法量化16×，与软件压缩对比显著节省内存与带宽；AES核心在KU5P-3实现>100 Gbps；FP32加/减单元在KU5P-3实现896 MHz。

**⚠️ 局限性**

系统过于方法化、过度追求高频导致不必要的RTL改动、验证耗时长、需要人工审查PPA与约束，且部分数值误差与硬件实现不匹配。

---

## 543. The First Token Knows: Single-Decode Confidence for Hallucination Detection

**arXiv ID:** 2605.05166 | [PDF](https://arxiv.org/pdf/2605.05166v1)

**作者:** Mina Gabriel `[一作]` (Temple University), Mina Gabriel `[通讯]` (Temple University)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5040915229)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估首词置信度（ϕ_first）作为一种低成本的幻觉检测方法，并与采样+NLI聚类的语义一致性方法进行比较。

**💡 创新点**

证明单个贪心解码的首词熵即可匹配或优于多重采样+NLI的语义一致性，且成本仅为其1/11。

**🔧 技术方法**

利用首词对数概率熵计算ϕ_first，并与传统表面一致性、语义一致性和口头自信度等基线进行对比。

**📊 数据集**

使用PopQA和TriviaQA各1000个英文闭卷简答样本进行实验。

**📈 对比分析**

在三款7–8B模型上，ϕ_first的平均AUROC为0.820，高于语义一致性0.793，显著优于表面一致性；差异在统计检验中均显著。

**⚠️ 局限性**

研究仅限于英文闭卷简答QA，未涵盖长文本、多步推理、检索增强、跨语言或更大模型；且存在少量长度敏感性和自动评判标签噪声。

---

## 544. Geometry-Aware State Space Model: A New Paradigm for Whole-Slide Image Representation

**arXiv ID:** 2605.05164 | [PDF](https://arxiv.org/pdf/2605.05164v1)

**作者:** Enhui Chai `[一作]` (PuzzleLogic Pte Ltd), Fei Xia `[通讯]` (University of California, Irvine)

**通讯引用:** 10296 | [OpenAlex ID](https://openalex.org/A5100676785)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 BatMIL 框架，利用多实例学习对 WSIs 进行切片级别的诊断预测，并通过双几何空间和 S4-MoE 结构同时建模层次结构、局部细节与区域异质性。

**💡 创新点**

创新点包括：① 双欧氏-双曲几何嵌入实现层次结构与局部细节的兼容；② S4 与 Mixture‑of‑Experts 的结合形成 S4‑MoE 骨干；③ 采用加权相加的几何融合策略，提升表示的互补性。

**🔧 技术方法**

使用技术：多实例学习、Structured State Space Sequence (S4)、Poincaré 球面双曲空间、Euclidean 空间、Mixture‑of‑Experts、梯度 CAM 可解释性、预训练视觉模型进行切片特征编码。

**📊 数据集**

使用数据集：CAMELYON16/17（乳腺转移）、PANDA（前列腺 ISUP 分级）、TCGA‑BLCA（膀胱分级）、TCGA‑BRCA（HER2 表达）、TCGA‑CESC（血管入侵）、TCGA‑NSCLC（肺腺癌分期）。

**📈 对比分析**

与 13 种现有 MIL 方法（ABMIL、CLAM、DSMIL、DTFD‑MIL、PatchGCN、ZoomMIL、HiGT、TransMIL、GigaPath、S4MIL、MambaMIL、MamMIL、PathRWKV）在 5‑折交叉验证下比较，BatMIL 在大多数任务上实现最高 AUROC、Accuracy、F1，提升约 1–5% 以上。

**⚠️ 局限性**

局限性：对极度稀疏、局部的诊断信号（如 HER2、Lymphovascular invasion）性能略逊于纯注意力模型；专家数量固定，可能不适应不同癌种的复杂度；双几何层与稀疏路由增加了内存开销。

---

## 545. Aes3D: Aesthetic Assessment in 3D Gaussian Splatting

**arXiv ID:** 2605.05155 | [PDF](https://arxiv.org/pdf/2605.05155v1)

**作者:** Chuanzhi Xu `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**通讯引用:** 12024 | [OpenAlex ID](https://openalex.org/A5076697411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Aes3D框架，构建Aesthetic3D 3D场景美学评估数据集，并设计Aes3DGSNet模型，实现直接基于3D高斯原语的场景美学评分。

**💡 创新点**

①基于多视图人工美学评估的3D场景注释流程；②利用多模态大型语言模型ArtiMuse生成多维属性评分；③在3D高斯原语上实现视角选择、几何投影与轻量级融合的全新评估网络。

**🔧 技术方法**

3D高斯分裂、点Transformer编码、几何投影生成视图描述、视角选择器（top‑K学习），以及Huber与配对排序损失的联合优化。

**📊 数据集**

Aesthetic3D 数据集（来自 DL3DV‑10K 与 Bilarf），包含 278 个 3D 场景、92,649 视图，使用 ArtiMuse 进行 8 维属性标注，随后构成训练/测试集。

**📈 对比分析**

与零射 2D 美学基线、3DGS 质量评估基线、点基质量评估基线进行对比；Aes3DGSNet 在 PLCC、SRCC、KRCC、MAE、RMSE 上均显著优于所有基线，8‑attr 版本在相关性上最高，模型仅 3.2M 参数、29.4 GFLOPs，保持轻量化。

**⚠️ 局限性**

依赖 ArtiMuse 生成的代理标签，可能存在域偏差；数据集规模相对有限；未涵盖多模态或用户个性化偏好；模型在更大多样化或真实渲染场景下的泛化尚待验证。

---

## 546. Age of Gossip in Ring Networks With Non-Poisson Updates

**arXiv ID:** 2605.05152 | [PDF](https://arxiv.org/pdf/2605.05152v1)

**作者:** Arunabh Srivastava `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 14006 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在环形网络中研究非泊松更新传播下的版本年龄，利用样本路径回溯方法证明任意节点的版本年龄在稳态后均为Θp(√n)

**💡 创新点**

首次在非同质、非泊松 renewal 过程下引入样本路径回溯和空间窗口优化，克服传统 SHS/FPP 仅适用于泊松过程的限制，并在双向环网络中通过预处理论证仅短路径影响版本年龄

**🔧 技术方法**

样本路径回溯、空间窗口优化、Lorden不等式、中心极限定理、Chebyshev不等式等概率分析技术

**📊 数据集**

无实验数据集，全部以理论推导为主

**📈 对比分析**

与之前基于 Poisson 过程的 SHS/FPP 结果比较，得到相同的 Θp(√n) 伸缩性；未提供数值仿真或实验性能对比

**⚠️ 局限性**

假设所有过程相互独立、均匀且具有有限均值与方差；不考虑链路失效、非 renewal 更新或网络拥塞等实际情况，缺乏仿真验证

---

## 547. What Matters in Practical Learned Image Compression

**arXiv ID:** 2605.05148 | [PDF](https://arxiv.org/pdf/2605.05148v1)

**作者:** Kedar Tatwawadi `[一作]` (Apple), Oren Rippel `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 PICO，一种面向实际部署的感知导向学习式图像压缩器，并通过系统性结构消融与数百万模型的神经架构搜索，达成速度与视觉质量的最优平衡。

**💡 创新点**

创新点在于：①将尺度解码器拆分为独立网络以保证跨平台确定性；②采用“单次推断上下文”实现自回归优势与推理速度兼得；③引入 ConvScale、可学习量化宽度、Haar 重采样等高效可表达性模块；④针对文本与块边缘引入专门的损失；⑤在海量候选网络上进行分阶段过滤与训练，最终选出在 iPhone 上 100 ms 级别解码且感知质量最优的模型。

**🔧 技术方法**

技术包括：端到端神经编码/解码网络（基于 hyperprior 结构）、自回归上下文模型、GAN 与感知损失（LPIPS、MS‑SSIM）、量化宽度学习、Haar 余弦变换、卷积尺度调节、文本掩码损失、低频块匹配损失、一次性多线程流水线、Neural Architecture Search（NAS）与多阶段过滤。

**📊 数据集**

数据集：约 90k 内部通用图像、2.3k 文本图像、28k 高分辨率公开数据（Div2K、CLIC、Flickr2K），用于训练；评估使用 CLIC 2020 Test、Kodak、DIV2K，主体基准为 Mabyduck 主观对比实验与 CMMD、FID、LPIPS 等客观指标。

**📈 对比分析**

对比方法：对传统（AV1、AV2、VVC、ECM、JPEG‑AI）和最强学习式（HiFiC、JPEG‑AI、MLIC++、CDC、TCM、MRIC、C3‑WD、DCVC‑RT）编码器，使用人眼主观评分（Elo）与感知指标比较。结果显示 PICO 在相同质量下实现 2.3‑3× 的码率压缩，20‑40% 以上比最佳学习式压缩器更小；在 iPhone 17 Pro Max 上 12MP 图像编码 230 ms、解码 150 ms，速度远快于 V100 GPU 上的主流学习式解码器。

**⚠️ 局限性**

局限性：仍需大量标注训练数据与多阶段训练流程；目前仅针对静止图像，未验证对视频或多平台（如 GPU、嵌入式芯片）的通用性；对极端分辨率或特殊内容（例如 3D、医学影像）的鲁棒性尚未充分评估。

---

## 548. Human-AI Co-Mentorship in Project-Based Learning: A Case Study in Financial Forecasting

**arXiv ID:** 2605.05144 | [PDF](https://arxiv.org/pdf/2605.05144v1)

**作者:** Freyaa Chawla `[一作]` (Archbishop Mitty High School), Grigorii Khvatskii `[通讯]` (University of Notre Dame)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5114633670)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究让高中生与本科生在导师指导下，利用 AI 工具完成 ETF 价格预测项目，重点在于通过 AI 作为共同导师加速学习与代码实现。

**💡 创新点**

创新点在于提出“人机共同辅导”模式，强调 AI 作为即时编程与思维伙伴，同时对项目过程中的学习与失败模式进行系统性反思。

**🔧 技术方法**

技术包括大语言模型（LLM）用于情感分析、Web 抓取脚本生成、时间序列回归与分类模型（ARIMA、SVR、XGBoost、LSTM），以及 AI 辅助的超参数搜索与代码调试。

**📊 数据集**

使用的数据集为 29 只 ETF 2019‑2025 年的日收盘价（Yahoo Finance）与对应的新闻文本（NASDAQ 报文），情感得分后按行业/日期聚合。

**📈 对比分析**

通过比较含情感特征与不含情感特征的多模型组合（包括基准、ARIMA、LSTM、SVM、XGBoost 等），结果显示情感特征在此数据下对预测准确性并未提升，模型整体表现低于无情感输入时。

**⚠️ 局限性**

局限性包括样本仅来自单一暑期项目、数据覆盖不完整、情感得分缺乏人工验证、AI 产生的代码与结果需要人工核查、模型泛化性待验证。

---

## 549. Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction

**arXiv ID:** 2605.05134 | [PDF](https://arxiv.org/pdf/2605.05134v1)

**作者:** Dan Wilson `[一作]` (University of Tennessee), Mohamed Akrout `[通讯]` (University of Tennessee)

**通讯引用:** 1334 | [OpenAlex ID](https://openalex.org/A5043530847)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种将LLM视为黑盒动力系统的幻觉检测方法，通过比较事实与幻觉样本在嵌入空间的预测误差差分残差来判定文本真伪。

**💡 创新点**

创新点在于利用Koopman算子和动态模式分解推断两条不同流形上的动力学模型，并用单样本差分残差实现低成本、无检索的检测。

**🔧 技术方法**

技术方法包括Koopman算子理论、扩展动态模式分解(EDMD)、嵌入模型升维、差分残差评分和用户偏好阈值校准。

**📊 数据集**

数据集涵盖三类：HaluEval摘要数据、Biographies 1.2K、Math/Science/Logic句子推理数据，分别用于训练和评估。

**📈 对比分析**

与现有多样本一致性检验、知识检索和基准模型相比，该方法在准确率、AUC‑PR、F1等指标上实现了与或超过现有最佳黑盒方案，并且单样本通过率高。

**⚠️ 局限性**

局限性在于仅能检测幻觉而无法提供纠正，且对极短文本或高噪声嵌入的鲁棒性仍有限。

---

## 550. Optimizing Bit-Labeling of Voronoi Constellations

**arXiv ID:** 2605.05202 | [PDF](https://arxiv.org/pdf/2605.05202v1)

**作者:** Carilyn Rumrill `[一作]` (Rampart Communications), Dan Chew `[通讯]` (Rampart Communications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对 D4 与 E8 根格的自相似 Voronoi 星座进行位-符号映射优化，得到 0.1 dB 与 0.5 dB 的误码率提升。

**💡 创新点**

提出“汉明密度”作为快速性能评估指标，并设计 Hamming descent 本地搜索算法在 SL_n(ℤ) 的模 r 同伦类中寻找最优位标记。

**🔧 技术方法**

利用格基变换（unimodular 矩阵）、Gray 映射、Hamming 密度评估、局部随机搜索，以及 AWGN 信道下的 BER 仿真。

**📊 数据集**

使用 D4 与 E8 根格的离散坐标集合（r=4,8），并未使用外部标注数据集。

**📈 对比分析**

与文献中常用的标准基进行 BER 曲线对比，最优基在 10^-4 误码率下分别提高约 0.1 dB（D4）与 0.5 dB（E8），证明 Hamming 密度能有效预测性能。

**⚠️ 局限性**

搜索空间仍过大，未完全逼近理论下界（D4 下界 1.67，E8 下界 2.4），仅在低维格中实验；未来需更强大搜索算法（如遗传算法）以及更高维格的验证。

---

## 551. OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents

**arXiv ID:** 2605.05185 | [PDF](https://arxiv.org/pdf/2605.05185v1)

**作者:** Shuang Chen `[一作]`, Tianyu Pang `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了完全公开的OpenSearch-VL训练方案，用来构建多模态深度搜索代理并发布相应数据、代码和模型。

**💡 创新点**

创新点在于（1）基于维基百科链接图的多跳路径采样、模糊实体重写和图像锚点对齐，消除单跳检索捷径；（2）构建包含检索、OCR、图像增强等多种工具的完整环境；（3）在GRPO框架中加入致命-aware token掩码和单侧优势裁剪，保留失败前有效推理。

**🔧 技术方法**

技术方法包括：ReAct交互式推理、GRPO强化学习、CLIP相似度过滤、GPT‑4o/LLM评判器、模糊实体重写算法、图像增强与OCR工具集成。

**📊 数据集**

使用的主要数据集为：SearchVL‑SFT‑36k（36k条专家轨迹）、SearchVL‑RL‑8k（8k条多模态任务）以及七个公开基准（SimpleVQA、VDR、MMSearch、LiveVQA、BrowseComp‑VL、FVQA、InfoSeek）。

**📈 对比分析**

与直接推理、RAG、Agentic baseline 进行 Pass@1 对比，OpenSearch‑VL 在七大基准上平均提升约10分；Qwen3‑VL‑30B‑A3B 的平均分达到 61.6，超过同级别公开模型及部分专有模型。

**⚠️ 局限性**

局限性：对外部检索/OCR/图像增强 API 的依赖导致训练不稳定和奖励方差；评估使用专有 GPT‑4o 判定器，成本高且仅覆盖文本查询，未评估视觉操作；复现受限于昂贵算力和外部服务。

---

## 552. Private Contiguous-Block Retrieval

**arXiv ID:** 2605.05169 | [PDF](https://arxiv.org/pdf/2605.05169v1)

**作者:** Maha Issa `[一作]` (Santa Clara University), Anoosheh Heidarzadeh `[通讯]` (Santa Clara University)

**通讯引用:** 552 | [OpenAlex ID](https://openalex.org/A5006920802)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并分析了一种针对私有连续块检索（PCBR）的平衡 {0,1}-线性方案，能够在任意服务器数、消息数与需求块大小下实现最优检索率并达到最小子分包级别。

**💡 创新点**

创新点在于充分利用需求块的连续性结构，构造了能在所有参数组合下达到最优率且子分包级别与下界匹配的新方案，从而显著降低了实现复杂度。

**🔧 技术方法**

使用了线性代数与信息理论中的隐私约束、整数规划与组合论方法，并在 PSSR 框架下设计查询与子分包索引。

**📊 数据集**

研究为理论分析，无需实际数据集，采用抽象的消息集合与区间需求模型。

**📈 对比分析**

与现有 MPIR 方案比较，在 D < K/2 且 D 不是 K 的因子时，本方案将率从 82/135 提升至 8/13，子分包级别从 82 降至 8；在其他参数范围内亦保持最优率且子分包更小。

**⚠️ 局限性**

限制在于仅适用于平衡 {0,1}-线性方案，未考虑非线性或协作服务器场景；当 D | K 且 K-D 非 1 时，子分包级别仍可能较大。

---

## 553. Wasserstein-Aligned Localisation for VLM-Based Distributional OOD Detection in Medical Imaging

**arXiv ID:** 2605.05161 | [PDF](https://arxiv.org/pdf/2605.05161v1)

**作者:** Bernhard Kainz `[一作]` (Friedrich–Alexander University Erlangen–Nürnberg), Cosmin Bercea `[通讯]` (Helmholtz Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种无需训练的 VLM 基于参考图像的异常定位框架 WALDO，用来在医学影像中实现零样本异常定位。

**💡 创新点**

创新点包括：①熵加权切片 Wasserstein 距离用于对健康参考图像进行分布式相似性评估；②Goldilocks 区域抽样（选择相似度处于中等范围的参考图像）以平衡偏差与方差；③通过差分提示与加权 NMS 的自一致性聚合降低预测方差。

**🔧 技术方法**

采用的技术有：DINOv2 ViT-B/16 提取补丁特征；最优传输理论实现熵加权切片 Wasserstein 计算；DPP（确定性点过程）实现参考多样性采样；差分提示（prompting）与加权 NMS 进行结果聚合。

**📊 数据集**

使用的数据集为 NOVA 脑 MRI 少见病定位数据集（907 张）和 VinDr‑CXR 胸部 X 光影像子集（949 张），并从健康样本中构建参考池。

**📈 对比分析**

在 NOVA 上，WALDO 将 Qwen2.5‑VL‑72B 的 mAP@30 从 36.4%（零样本复现）提升至 43.5%（+19% 相对），GPT‑4o、Qwen3‑VL‑32B 等模型亦获得显著提升；在 VinDr‑CXR 上同样取得 mAP@30 的正向改进，验证了跨模型与跨模态的一致性。

**⚠️ 局限性**

局限性包括：在胸部 X 光上性能仍低（大约 24–35%），对极小或弥散性病变识别不足；方法高度依赖高质量健康参考图像；VLM 的推理时延较高，影响实时部署；以及对参考池构建与管理的专业要求。

---

## 554. CPCANet: Deep Unfolding Common Principal Component Analysis for Domain Generalization

**arXiv ID:** 2605.05136 | [PDF](https://arxiv.org/pdf/2605.05136v1)

**作者:** Yu-Hsi Chen `[一作]` (University of Melbourne), Abd-Krim Seghouane `[通讯]` (University of Melbourne)

**通讯引用:** 2058 | [OpenAlex ID](https://openalex.org/A5084681382)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Common Principal Component Analysis（CPCA）的深度框架CPCANet，能够在多源域中学习共享的领域无关子空间，并通过特征调制提升表示能力。

**💡 创新点**

创新点包括：① 将CPCA的Flury‑Gautschi（FG）迭代算法展开为可微分的网络层；② 使用Cayley变换实现正交约束并在Stiefel流形上进行Riemannian梯度优化；③ 通过超网络动态生成步长以适应批次统计变化；④ 结合FiLM式特征调制，避免低维子空间导致的信息瓶颈。

**🔧 技术方法**

核心技术：CPCA、FG算法展开、Cayley重提、Riemannian优化、深度展开（Unfolding）、超网络（Hypernetwork）动态步长、FiLM特征调制。

**📊 数据集**

在四个标准域泛化基准上进行实验，分别为：PACS、VLCS、Office‑Home和DomainNet。

**📈 对比分析**

在统一的训练设置下与多种现有DG方法对比，CPCANet在零样本转移任务中获得最高平均准确率（≈85–90%），且与ERM的计算开销相近（内存≈75 GB，训练时间≈8‑16 h），显著优于大多数传统与现代方法。

**⚠️ 局限性**

局限性：① 需要源域标签来估计协方差矩阵，无法直接应用于无标签或自监督场景；② 对超参数（子空间维度d、展开层数T）的敏感性需进一步研究；③ 尽管计算开销低于部分复杂模型，但相较于纯端到端方法仍有一定额外成本；④ 在高度非线性或多模态任务中，线性CPCA可能无法完全捕获所有共享结构。

---

## 555. Joint Treatment Effect Estimation from Incomplete Healthcare Data: Temporal Causal Normalizing Flows with LLM-driven Evolutionary MNAR Imputation

**arXiv ID:** 2605.05125 | [PDF](https://arxiv.org/pdf/2605.05125v1)

**作者:** Olivia Jullian Parra `[一作]` (University of Zürich), Nicola Serra `[通讯]` (University of Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种两阶段流水线，先用LLM驱动的进化填补方法处理高比例的MNAR缺失，再用DAG约束的正态化流CausalFlow-T实现精确的因果反事实推断；

**💡 创新点**

创新点在于将DAG约束与精确流模型结合，解决时间变化的混杂和MNAR缺失的交叉挑战，并通过自监督评估指标同时考察预测与因果一致性；

**🔧 技术方法**

核心技术包括LSTM编码的时间序列正态化流、因果马尔可夫随机场与干预操作的精确逆向推理、LLM演化搜索的缺失填补与自监督代理评分；

**📊 数据集**

实验使用四个结构复杂的合成数据集和一个基于真实电子病历的半合成数据集，以及瑞士主要护理数据库中的2,392例GLP-1受体激动剂与3,722例SGLT-2抑制剂的T2D患者；

**📈 对比分析**

与CVAE、GNN-CVAE、TARNet、MissForest、LOCF及CausalCFM等方法比较，CausalFlow-T在所有五个可靠性指标上均为唯一合格者；LLM填补在30–80% MNAR下在重构与因果指标上均取得最佳平均排名，且在真实数据上得到的-0.98kg体重差与RCT一致；

**⚠️ 局限性**

局限性包括对二元结果的去量化引入校准成本、固定的专家DAG假设、对单一演化搜索策略的依赖以及未能处理潜在的不可观测混杂；

---

## 556. MCFlash: Bulk Bitwise Processing in 3D NAND with Dynamic Sensing and Multi-level Encoding

**arXiv ID:** 2605.05119 | [PDF](https://arxiv.org/pdf/2605.05119v1)

**作者:** Habib Ur Rahman `[一作]` (Colorado State University), Biswajit Ray `[通讯]` (Colorado State University)

**通讯引用:** 2028 | [OpenAlex ID](https://openalex.org/A5083246460)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在现有COTS 3D NAND闪存芯片上实现了无需硬件修改的批量位运算（AND、OR、NOT、XNOR）技术，称为MCFlash。

**💡 创新点**

创新点在于利用MLC闪存的多级电压编码和可调读参考电压，结合软位读取（SBR）实现逻辑运算，仅需标准用户指令即可在闪存内部完成。

**🔧 技术方法**

技术包括动态读参考电压偏移、软位读取、SBR、内置的页读取/写入/擦除、对齐复制等，全部通过ONFI标准用户指令完成。

**📊 数据集**

实验使用了来自主流厂商的三代3D NAND芯片（64层浮栅、176层电荷捕获）以及图像分割、加密、bitmap索引等实际工作负载。

**📈 对比分析**

与OSC、ISC、ParaBit和Flash-Cosmos等方案比较，MCFlash在内存对齐情况下的位运算吞吐率显著提升，平均相对于OSC提升约16.5倍，ISC 12.6倍，ParaBit 1.7倍，能耗与单页读取相当。

**⚠️ 局限性**

局限性在于对读偏移电压范围的依赖，部分高阶运算（NAND、NOR、XOR）在老化或高P/E周期时易出现错误，且需要操作数在同一芯片且同一wordline上，非对齐需额外复制开销。

---

## 557. On the Wasserstein Gradient Flow Interpretation of Drifting Models

**arXiv ID:** 2605.05118 | [PDF](https://arxiv.org/pdf/2605.05118v1)

**作者:** Arthur Gretton `[一作]` (Google DeepMind), Arnaud Doucet `[通讯]` (Google DeepMind)

**通讯引用:** 38873 | [OpenAlex ID](https://openalex.org/A5091677854)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将Generative Modeling via Drifting (GMD) 视为寻找 Wasserstein 梯度流（WGF）极限点的过程，分析其与 KL 散度、Sinkhorn 距离、MMD、Sliced‑Wasserstein 等流的关系，并提出 Sinkhorn Proxy 近似实现。

**💡 创新点**

创新点在于：①将 GMD 理论化为 WGF 固定点问题；②给出 KL 散度（加 Parzen 平滑）下的梯度流正确形式；③提出 Sinkhorn Proxy 近似 Sinkhorn 梯度流；④将此框架推广到 MMD、SW、f‑divergence 等多种损失，实现统一的生成器训练方法。

**🔧 技术方法**

使用技术包括 Wasserstein 梯度流理论、KDE 与 Tweedie 公式、Sinkhorn 算法、对称软最大化、生成器 ResNet+Adam 训练、MMD 评估等。

**📊 数据集**

实验数据集为 5 个合成二维分布（如 Moons、Circles 等）以检验不同 drift 的效果。

**📈 对比分析**

对比方法：在同一设置下分别训练 6 种 drift（KL、MMD、SW、Sinkhorn、Sinkhorn Proxy、KALE），使用 MMD² 评估生成质量。结果表明，各流在合适的超参 τ 下可达到相近最佳表现，Sinkhorn Proxy 对小 τ 更加稳健。

**⚠️ 局限性**

局限性：①GMD 的理论基础仍不完善；②Sinkhorn Proxy 并非真正的 WGF，无法保证在大分离模式间有效迁移质量；③KL 与 MMD 等流在模式分离较大时易出现质量转移失败；④实验仅在低维合成数据上验证，缺乏高维真实数据的验证。

---

## 558. Conditional outlier detection for clinical alerting

**arXiv ID:** 2605.05124 | [PDF](https://arxiv.org/pdf/2605.05124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 559. Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior

**arXiv ID:** 2605.05115 | [PDF](https://arxiv.org/pdf/2605.05115v1)

**作者:** Daniel Wurgaft `[一作]` (Stanford University), Ekdeep Singh Lubana `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了神经网络表示中的几何结构，并通过在不同几何假设下对激活进行干预，探究其对模型行为的因果影响。

**💡 创新点**

提出并验证了“几何感知干预”——利用激活子流形与行为子流形的等距关系，替代传统欧氏线性干预，显著提升干预的自然性与可控性。

**🔧 技术方法**

构建激活与行为的低维子流形，使用三维曲线拟合、Hellinger空间映射、能量函数评估以及梯度优化的拉回（pullback）方法实现路径规划。

**📊 数据集**

在多模态任务中实验，包括大型语言模型（Llama 3.1 8B）处理日历、月份、字母、年龄等概念，以及在上下文学习的图结构任务和视觉世界模型（Mountain Car）等。

**📈 对比分析**

将几何感知干预与传统线性干预在行为轨迹能量、顺序连贯性和R²相似度等指标对比，几何感知干预在能量上平均低约2.8×、R²提升到0.7–0.8，显著优于线性干预。

**⚠️ 局限性**

实验范围限于离散、可视化的概念空间，未验证在更抽象、序列级别输出或真实世界数据上的有效性；且子流形拟合依赖人工选择的中心点，缺乏无监督通用方法。

---

## 560. Interests Burn-down Diffusion Process for Personalized Collaborative Filtering

**arXiv ID:** 2605.05165 | [PDF](https://arxiv.org/pdf/2605.05165v1)

**作者:** Yifang Qin `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 35053 | [OpenAlex ID](https://openalex.org/A5100461491)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种面向协同过滤的兴趣消退扩散过程（Interests Burn‑down Diffusion Process），并基于此提出了自编码器推荐框架 StageCF，用逆向燃起过程生成个性化推荐。

**💡 创新点**

创新点在于：①将用户兴趣建模为多阶段离散向量，并通过二项式衰减实现随时间递减的“兴趣消退”过程；②引入图卷积衰减因子，使得衰减速率依赖于项目协同关系，实现个性化衰减；③将扩散过程与自编码器结合，利用逆向燃起采样产生推荐，从而充分挖掘离散交互数据的潜在分布；④通过重要性采样与分布匹配的 KL 损失，提升训练稳定性和效果。

**🔧 技术方法**

主要技术包括：离散二项式扩散（Interest Burn‑down），图卷积衰减因子，反向燃起采样（Binomial Bridge），自编码器（MLP）作为评分网络，KL 损失与重要性采样的训练目标，以及在 PyTorch 中实现的多步骤采样推理。

**📊 数据集**

实验使用三大公开交互数据集：Gowalla、Yelp2018 与 Amazon‑Book，采用标准 80/10/10 训练/验证/测试划分。

**📈 对比分析**

与 7 种基线（SLIM、iALS、LightGCN、MultDAE/MultVAE、MacridVAE、DiffRec、DDRM、FlowCF）在 Recall@20 和 NDCG@20 上对比，StageCF 在所有数据集上均优于所有基线，Recall@20 提升 4.7%–14.2%，NDCG@20 提升 4.3%–11.3%。实验还通过 ablation 证实兴趣消退过程、图卷积衰减因子和重要性采样对性能的关键作用。

**⚠️ 局限性**

局限性包括：①扩散采样需要多步迭代，推理速度比轻量级模型慢；②需要调节多项超参数（衰减因子 γ、阶段数 K、采样步数等），调参成本较高；③仅在标准 CF 场景验证，未探究序列化或冷启动等更复杂任务；④模型基于离散交互，无法直接处理连续或多模态特征。

---

## 561. ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation

**arXiv ID:** 2605.05126 | [PDF](https://arxiv.org/pdf/2605.05126v1)

**作者:** Wei Li `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29799 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ConsisVLA-4D 框架，将 3D 感知与 4D 时空推理结合，实现机器人视觉-语言-动作模型的跨视角、跨物体、跨场景一致性。

**💡 创新点**

创新点包括：① CV-Aligner 通过 FiLM 调制、Top‑K 语义选取和单视角融合，实现跨视角对象语义一致；② CO‑Fuser 采用块级融合与因果注意力，解决单视角几何歧义，得到跨物体几何一致；③ CS‑Thinker 通过动态/全局深度解码和 Spatiotemporal Consistency Attention，将时空一致性扩展至推理阶段；④ 通过稀疏化仅保留 1/8 视觉令牌，显著提升效率。

**🔧 技术方法**

使用 SigLIP、DINOv2、VGGT 视觉编码器；FiLM 语义调制；Top‑K 令牌筛选；跨视角交叉注意力；块级因果注意力；低秩适配 LoRA；稀疏令牌化；以及自定义的 CV‑Aligner、CO‑Fuser、CS‑Thinker 模块。

**📊 数据集**

在 LIBERO、ManiSkill2、RoboTwin 2.0 以及真实平台 Galaxea R1 Lite 与 AgileX Cobot Magic 上进行评估，涵盖 4 大任务套件、3 大拾放任务与 7 大双臂任务。

**📈 对比分析**

与 OpenVLA、SpatialVLA、π_0、CoT‑VLA 等 SOTA 进行对比，ConsisVLA‑4D 在 LIBERO 平均成功率提升 21.6%/41.5%，在 ManiSkill2、RoboTwin 2.0 与真实任务中均保持 98% 以上成功率；推理延迟缩短 2.3×/2.4×，吞吐量提升 72.7 Hz/108.2 Hz；总体性能优于同等规模 7B 视觉‑语言模型。

**⚠️ 局限性**

局限性：仍需要多视角 RGB 输入并依赖 VGGT 预训练；在极端动态或遮挡情形下几何一致性可能受限；缺少对长时序自监督学习的支持，未来可进一步改进。

---

