# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-27 | 今日论文总数: 570

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. AudioLens: Multi-Perspective Speech Clustering with Reasoning Audio-Language Models

**arXiv ID:** 2608.25177 | [PDF](https://arxiv.org/pdf/2608.25177v1)

**作者:** Wenjun Huang `[一作]` (University of California, Irvine), Mohsen Imani `[通讯]` (University of California, Irvine)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出音频多视角聚类任务，构建基于自然语言描述的聚类视角和相应标注的评测基准，并设计了一种端到端的音频-语言模型，实现直接从原始音频和聚类视角生成聚类结果。

**💡 创新点**

创新点在于将聚类视角的自然语言指令作为条件输入，利用推理蒸馏与偏好优化双重训练，首次实现模型在不依赖外部聚类算法的情况下，自动推断聚类数并做出合理的分组；同时将语言与声学特征统一建模，支持既需要语义推理又需要声学判别的多样聚类任务。

**🔧 技术方法**

技术方法包括：1）基于大型音频-语言模型（Audio Flamingo 3 作为基底）进行细粒度微调；2）推理蒸馏阶段生成比较型推理轨迹，强化模型的比较与推理能力；3）直接偏好优化（DPO）阶段通过采样错误聚类结果作为硬负样本，对模型输出进行偏好对齐；4）使用自监督的 TTS 生成语料以及语音合成时注入情感、背景噪声等声学属性。

**📊 数据集**

所用数据集为四个多域语料：欧盟法院判决（ECHR）、S&P 500 年报、银行业务请求（Banking77）和任务型对话（MultiWOZ），并在每个语料中构造多种聚类视角（如背景噪声、情感、说话者计数、性别、语言推理），通过 TTS 生成对应音频。

**📈 对比分析**

与基线的比较包括：传统 ASR+文本嵌入+KMeans/GMM、ASR+LLM（GPT‑4o）以及多种现成的音频‑语言模型（GPT‑4o‑audio‑preview、GPT‑audio‑1.5、Qwen3‑omni‑instruct‑30B、Audio Flamingo 3、Qwen2.5‑omni）。实验表明，提出的模型在整体 ARI 上达到 44.77，V‑measure 达到 73.43，分别比最佳基线提升 12.99 点和 11.62 点，且在大多数子任务上表现优于所有对照模型。

**⚠️ 局限性**

主要局限包括：1）实验仅基于 TTS 合成音频，缺乏真实录音的评测，可能低估对语音噪声和口音多样性的鲁棒性；2）基线中有些方法使用了 oracle 的聚类数，导致与直接推断聚类数的模型难以公平比较；3）尽管模型在多视角任务上表现均衡，但在极度异质或极低频率的视角上仍可能出现过拟合或泛化不足，需要进一步研究更强的多视角迁移策略。

---

## 2. NVExplain: Explaining Time Series Forecasting with Latent Trajectory Analysis and Structure-Preserving Surrogates

**arXiv ID:** 2608.25080 | [PDF](https://arxiv.org/pdf/2608.25080v1)

**作者:** Muyan Anna Li `[一作]` (NVIDIA), Aditi Gautam `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种模型无关的时间序列预测解释框架，通过分析预测模型在潜在空间中的语义流动，将每个预测时点的影响归因于历史时间滞后。

**💡 创新点**

创新点包括：①在潜在空间中引入语义流（semantic flow）量化表示随时间的演变；②构建滞后–预测时点归因矩阵，实现多步预测的时点特定解释；③设计结构保持的时间扰动与稀疏局部代理模型，兼顾可解释性与稳定性；④提出针对预测特性的可信度与稳定性诊断。

**🔧 技术方法**

技术手段包括：滚动窗口潜在嵌入、语义流计算与指数平滑、滞后–时点归因矩阵的软最大化、块自举加傅里叶幅度扰动生成结构保持扰动、加权 L1 正则化的稀疏线性代理模型、以及嵌入与轨迹稳定性指标。

**📊 数据集**

实验使用 Nixtla 长期预测基准中的 ETTh1、Exchange、ILI 与 Weather 四个多元时序数据集，模型为 MOMENT 并暴露嵌入接口。

**📈 对比分析**

与 Random、Attention、Integrated Gradients、TimeSHAP 等基线比较，语义流变体在大部分数据集上实现了更高的 FPR 与 AOPC，并且运行时仅为现有方法的 1–3% 级别；全流程（含代理模型）在线性场景下可提升可读性，但在高度非线性场景下可能降低可信度。

**⚠️ 局限性**

局限性包括：1）语义流需要模型提供稳定的嵌入接口；2）代理模型在非线性预测场景下可能产生误差；3）某些数据集（如 ETTh1）潜在轨迹不稳定，导致解释在极端条件下需谨慎使用；4）尚缺乏用户研究验证解释的实际可理解性。

---

## 3. Exact algorithms for optimal discretization

**arXiv ID:** 2608.25197 | [PDF](https://arxiv.org/pdf/2608.25197v1)

**作者:** László Kozma `[一作]` (TU Dresden), Junqi Tan `[通讯]` (Freie Universität Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出最优离散化和点分离问题的更快指数算法，将时间复杂度降低到 (1.9602^n) 与 (1.8906^n)。

**💡 创新点**

通过结构化观察证明可将较小的一组线数限制为 0.4n（离散化）或 1/3n（点分离），从而显著压缩搜索空间，并给出坐标退化情形的上界。

**🔧 技术方法**

采用结构分析、平衡引理、区间覆盖贪心、二进制熵估计与搜索枚举相结合的方法，利用分支限界和简化分割的技术实现算法。

**📊 数据集**

论文未使用实际数据集，而是通过构造无穷族实例来证明下界与结构性质。

**📈 对比分析**

相较于朴素的 2^n 枚举，所提出算法在指数常数上提升约 2% 以上；与已知的 FPT 2^(k^2 log k) 算法相比，在 n 维度上更高效，但对 k 的依赖仍未优化。

**⚠️ 局限性**

主要局限在于对 k 的复杂度未突破 2^o(n) 的可能性；对一般位置情况仅给出理论上上界，实际常数可能更高；未进行实验验证。

---

## 4. MCP-Driven Accessibility Tree Standardization for AI-Powered Screen Reader Agents

**arXiv ID:** 2608.24898 | [PDF](https://arxiv.org/pdf/2608.24898v1)

**作者:** Vishnu Ramineni `[一作]` (Albertsons Companies Inc), Siva Kumar Chintham `[通讯]` (L&T Infotech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于模型上下文协议（MCP）的架构，用于在不同平台的无障碍子系统和基于大语言模型（LLM）的辅助代理之间提供统一的无障碍树标准化。

**💡 创新点**

创新点在于通过MCP协议将无障碍树的暴露和导航工具标准化，解决了现有无障碍API的碎片化问题，并支持用户残疾档案的跨会话持久化。

**🔧 技术方法**

使用了模型上下文协议（MCP）作为传输和架构层，结合了无障碍树的结构化资源和工具。

**📊 数据集**

未使用特定数据集，而是通过对现有无障碍API、GUI代理文献和MCP规范的比较分析进行评估。

**📈 对比分析**

通过定性比较分析，表明MCP中介的树访问在集成成本和上下文令牌消耗上具有优势，但无法解决原生无障碍API提取的延迟问题。

**⚠️ 局限性**

局限性在于未构建跨平台的工作原型，评估主要基于文献的定性判断，且原生无障碍树在实际应用中常常不完整。

---

## 5. ROS2 Connect: A new ROS2 over WAN Solution

**arXiv ID:** 2608.25102 | [PDF](https://arxiv.org/pdf/2608.25102v1)

**作者:** Daniel Schott `[一作]` (Julius-Maximilians-Universität Würzburg), Andreas Nüchter `[通讯]` (Julius-Maximilians-Universität Würzburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ROS2 Connect，一套基于 WebSocket 的中间件框架，能够在缺乏多播支持的广域网（WAN）中实现 ROS2 主题、服务、动作及系统数据的透明双向通信，并集成了身份验证与访问控制。

**💡 创新点**

创新点包括：① 将 ROS2 的多播发现机制替换为配置驱动、显式接口中继模型；② 采用 WebSocket 作为跨网传输通道，兼容 NAT/防火墙环境；③ 使用运行时通用消息转发（无编译时类型依赖）和插件化服务/动作支持；④ 将安全与访问控制嵌入框架，实现统一的认证与授权；⑤ 通过实验验证其在 WAN 上低延迟、低波动和良好可扩展性。

**🔧 技术方法**

技术栈包括 ROS2（Jazzy Jalisco）、DDS（Fast DDS / Cyclone DDS）、WebSocket、插件化（pluginlib）、JSON/二进制消息序列化、压缩元数据、身份验证插件（token 或外部身份管理）以及时间同步与 tf2 转发。

**📊 数据集**

实验使用真实的住宅到大学服务器的 WAN 链路（上行 18 Mbps，下行 58 Mbps）进行评估，发送 12 B 至 500 kB 的 ROS 消息，进行 1000 次 RTT 测量，并在 1–10 条平行主题的负载下测试；没有使用公开数据集，主要依赖于该 WAN 链路的测量数据。

**📈 对比分析**

与 DDS Router、rosbridge 以及 Eclipse Zenoh 进行对比；在单线程顺序传输下，ROS2 Connect 的平均 RTT 在所有消息大小上均低于其他方案，且波动范围更窄；在并发负载下，ROS2 Connect 的 RTT 近乎恒定（小消息）或线性增加（大消息），总体稳定性和可预测性优于对比方案。

**⚠️ 局限性**

局限性包括：① 对服务/动作的通用支持仍处于插件阶段，未覆盖所有类型；② 依赖单一 WebSocket 连接，传输为有序 TCP 流，对大量并发大数据包会产生线性延迟；③ 需要手动配置需要中继的接口，未实现自动发现；④ 在真正机器人部署与完整应用栈中的性能尚待进一步验证。

---

## 6. GreenLeaf Law Embed Tiny: A Compact Embedding Model for Legal Domain Retrieval

**arXiv ID:** 2608.24936 | [PDF](https://arxiv.org/pdf/2608.24936v1)

**作者:** Surya Saka `[一作]` `[通讯]` (JudicialMind), Surya Saka (JudicialMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并训练了一个0.6B参数的法律检索嵌入模型，采用两阶段训练（先从大型教师模型蒸馏，再进行法律领域的硬负样本微调），实现了高效、可解释的法律文本检索。

**💡 创新点**

创新点包括：将知识蒸馏与法律专属硬负样本挖掘结合、引入司法领域嵌入与引文上下文的专门化编码方案，以及提供可切换的量化推理策略（BF16/INT8/二值化）。

**🔧 技术方法**

技术方法包括：双编码器对比学习、知识蒸馏、层级编码、司法嵌入、引用上下文增强、硬负样本挖掘、RoPE位置编码、混合量化推理。

**📊 数据集**

使用了司法Mind法律语料库（约340万问答对，含150k人工标注），覆盖35种语言、40+法域，数据通过精细过滤、去重、司法区分与时效划分构建。

**📈 对比分析**

在MLEB和MTEB(Law)基准上，模型分别获得75.11%和64.38%的NDCG@10，明显优于同参数规模的开源模型（如BGE-M3、Qwen3 Embedding），但仍低于1.8B级别的Kanon 2 Embedder和8B级别的Dinghy Law。

**⚠️ 局限性**

局限性包括：相比更大模型仍有性能差距，特别是复杂判例检索；跨语言性能在东亚语言上下降；对极长文本仍需分块处理；模型训练与部署需要一定硬件与专业人才支持。

---

## 7. The Changing Geometry of Grammar: Dimensionality and Neighborhood Reorganization across Transformer Layers

**arXiv ID:** 2608.25166 | [PDF](https://arxiv.org/pdf/2608.25166v1)

**作者:** Samuele Vallisa `[一作]` (Universitat Pompeu Fabra), Wolfram Hinzen `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer内部表示的几何轨迹，利用词性条件的Intrinsic Dimensionality（ID）和Information Imbalance（II）分析功能词与内容词在编码器和解码器中的扩张与收缩路径，并证明这些几何特征能单独预测语法角色。

**💡 创新点**

提出词性条件的ID估计器与II度量，揭示功能词与内容词在网络层间的不同几何演化，并证明该几何轨迹可以高精度地区分语法类别，这是此前未被探索的现象。

**🔧 技术方法**

使用ABIDE最近邻ID估计器、Information Imbalance度量、SpaCy POS标注、Transformer模型多层激活提取，并以逻辑回归对几何特征进行语法角色分类。

**📊 数据集**

基于HuggingFace的Pile‑10k文本语料（539篇文档，1000–1500词），使用SpaCy给词标注POS，随后提取各层表示。

**📈 对比分析**

通过对ModernBERT、BigBird‑Roberta‑Large（编码器）和Gemma‑2‑2B、Llama‑3.2‑3B（解码器）的cID和cII轨迹进行对比，利用Logistic回归对功能词/内容词二分类和七类POS多分类，F1分别达0.878和0.540，显著优于随机或打乱基线，展示了几何特征的强预测能力。

**⚠️ 局限性**

研究仅涵盖英语且仅测试四个模型，缺乏跨语言和更大规模Transformer的验证，因而结果可能受语言类型学差异和模型选择的影响。

---

## 8. Tunable Tool-Call Rates in LLM Agents via Representation Steering

**arXiv ID:** 2608.25198 | [PDF](https://arxiv.org/pdf/2608.25198v1)

**作者:** Yuqi Chen `[一作]` (UC Santa Cruz), Chenguang Wang `[通讯]` (UC Santa Cruz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了如何在推理阶段通过在LLM残差流中加上单一线性方向来精确控制工具调用率，既能抑制不必要的调用，又能激活需要的调用；

**💡 创新点**

创新点在于发现工具调用决策可由一个单向量在中后层实现全局、连续的控制，且该方向可跨模型、跨工具泛化，且不需再训练或改prompt；

**🔧 技术方法**

使用差异均值（DIM）提取工具调用倾向方向，利用首词工具调用token的前向概率作为偏好指标，随后在推理时对残差流加上可调标量α的向量干预，并验证投影clamping与ablation效果；

**📊 数据集**

实验使用PopQA、GSM8K、BIG-Bench Hard三大公开数据集，以及为工具泛化生成的模板查询（翻译、天气、计数、邮件、SQL、股票等）；

**📈 对比分析**

与未干预基线和prompt工程比较，评估不同模型（Qwen3-4B/8B/30B、Gemma、gpt-oss）的调用率、准确率及成本；在PopQA上通过调节α在0.29→0.56的准确率提升，形成成本–准确率Pareto前沿；

**⚠️ 局限性**

局限性包括仅能控制是否调用工具，无法保证调用质量；强正向干预可能导致调用格式错误；需要在中后层的表示成熟后才有效；对推理前工具调用的可读性有限。

---

## 9. Analyzing and Correcting Benevolence Bias in Large Language Models

**arXiv ID:** 2608.24912 | [PDF](https://arxiv.org/pdf/2608.24912v1)

**作者:** Yuanzi Li `[一作]` (Renmin University of China), Xu Chen `[通讯]` (Renmin University of China)

**通讯引用:** 24721 | [OpenAlex ID](https://openalex.org/A5100385692)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了基于心理测量的基准，系统评估18款大型语言模型在四大社会科学问卷（WVS、PT、ANES、GSS）中的答案，量化并表征其“仁慈偏差”（benevolence bias）。

**💡 创新点**

创新点在于提出并拆分六维仁慈偏差概念、绘制偏差来源与规模映射，揭示对齐训练导致的偏移与范围，并提出一种轻量级对比校准方法可在无再训练的前提下恢复人类基准。

**🔧 技术方法**

技术路径包括：LLM提取问卷文本、上下文拆解、层级标签化、模拟人口条件化、BTB/BWR 两项量化指标、以及对不同语言、框架、思考模式和温度等变量的系统实验。

**📊 数据集**

使用的数据集为四大社会科学调查：World Values Survey（WVS）、prospect‑theory replication（PT）、American National Election Studies（ANES）、General Social Survey（GSS），并对其进行多语言（英中）转换与题目拆解。

**📈 对比分析**

对比方法是将18款模型在6个仁慈维度上的BTB与BWR与人类基准进行对齐，结果显示绝大多数模型均偏向仁慈且偏差随规模增大；对比校准可将BTB降至接近零、BWR降至≈0.5，显著改善偏差。

**⚠️ 局限性**

局限性包括：样本主要为西方或全球性调查，低资源语言与自由文本回答尚未评估；对齐训练细节不可公开；对比校准需要访问token概率，某些API不支持。

---

## 10. SHSP: Structure-Aware Hierarchical Solution Prediction for Mixed-Integer Linear Programming

**arXiv ID:** 2608.25282 | [PDF](https://arxiv.org/pdf/2608.25282v1)

**作者:** Zherong Zhang `[一作]` (Nanjing University), Chao Qian `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SHSP框架，替代传统并行边缘预测，采用分层条件解码预测MILP变量。

**💡 创新点**

创新点是利用变量耦合图构造分层预测顺序、置信度掩码修复机制以及结构感知变量固定策略。

**🔧 技术方法**

使用图神经网络、结构耦合图、置信度掩码修复、结构感知固定以及与Predict-and-Search、Apollo-MILP、Neural Diving等学习加速求解器的结合。

**📊 数据集**

在四个标准MILP基准上评测：Combinatorial Auctions (CA)、Workload Appointment (WA)、Item Placement (IP) 与 Set Covering (SC)。

**📈 对比分析**

与基线相比，SHSP在三大框架（ND、PaS、Apollo）以及两大求解器（Gurobi、SCIP）中平均降低绝对主问题缺口约54%，在CA上甚至超过3600s Gurobi的最佳已知解。

**⚠️ 局限性**

局限在于耦合图构造需要预先计算，且对连续变量的处理仍为启发式，未来需端到端学习耦合权重。

---

## 11. Apples to Apples? Towards Comparable Crosslingual Language Model Evaluation

**arXiv ID:** 2608.25089 | [PDF](https://arxiv.org/pdf/2608.25089v1)

**作者:** Xiulin Yang `[一作]` (Georgetown University), Catherine Arnett `[通讯]` (EleutherAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在控制词表大小的单语和多语大模型上使用平行语料，系统评估了多种交叉语言评估指标，证明句级负对数似然（Sent‑NLL）在跨语言和跨模型比较中最为公平。

**💡 创新点**

创新点在于提出并实证验证Sent‑NLL为最少受词表、字节/字符编码、句法多样性等工程因素影响的指标，同时揭示传统归一化指标的系统偏差。

**🔧 技术方法**

使用了负对数似然、困惑度、每字节/字符位数、均值倒数排名等概率指标，并通过Spearman相关、不同词表大小和多语模型的实验来比较其偏差。

**📊 数据集**

实验数据包括10种语言（阿拉伯语、中文、英语、法语、德语、芬兰语、波兰语、俄语、韩语、土耳其语）的Parallel‑10和Parallel‑3语料、FLORES‑200、Universal Dependencies 1k句、WMT2019英德翻译对等。

**📈 对比分析**

比较方法是将每种指标与潜在偏差因子（字节数、字符数、令牌计数）进行相关性分析，并在翻译替代、样本量变化以及多语LLM的预期与实际排名对比，结果表明Sent‑NLL与预期一致，其他指标则明显受偏差影响。

**⚠️ 局限性**

局限性包括仅覆盖10种语言、在短语释义实验中仅使用英德对、以及对高质量平行语料的依赖，这限制了结论在更广泛语言和资源受限场景下的推广。

---

## 12. From Plots to Words: Model-Aware Multimodal Explanations as a Foundation for Accessible, Non-Visual Interaction

**arXiv ID:** 2608.24910 | [PDF](https://arxiv.org/pdf/2608.24910v1)

**作者:** Nur Keleşoğlu `[一作]`, Joanna Domańska `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个多模态多代理框架，将DevOps系统中的时间序列预测结果（原本以图表呈现）转化为结构化、模型感知的文本解释，实现非视觉交互。

**💡 创新点**

创新点在于：①将预测、可视化、模型内在解释（SHAP、时间依赖、不确定性）统一到多代理流水线中；②提出三阶段响应策略（Baseline、Interpretable、Explainable），系统化比较模态信息与模型感知对解释质量的影响；③为盲/弱视用户提供可访问的基于文本的解释，奠定非视觉交互基础。

**🔧 技术方法**

技术包括：多模态大型语言模型（GPT‑5‑mini、GPT‑5.2）、Agent Development Kit（ADK）搭建多代理架构、Model Context Protocol（MCP）暴露预测/可视化/解释工具、SHAP TreeExplainer进行特征归因、随机森林时间序列预测、跨模态一致性提示、LLM评估判定器。

**📊 数据集**

使用了GPU Cluster Spot Resource Dataset（约113天、4,278节点、11,702 GPU卡），提取gpus_active_requested作为目标变量，采用24阶滞后特征构造随机森林回归模型。

**📈 对比分析**

通过LLM（GPT‑5.4）评估器在30条真实用户查询上对三种流水线进行自动评分。结果显示：Baseline→Interpretable提升约22%，Interpretable→Explainable再提升约8%，整体提升约32%；在明确性、帮助性、洞察性、可信度、模型感知、连贯性等指标均优于基线，hallucination率降低，uncertainty awareness 增强。

**⚠️ 局限性**

局限性：①仅用LLM评估器作为人类评估替代，缺乏真实用户（尤其是盲/弱视用户）的交互与可用性验证；②模型感知解释仍不完全消除hallucination，依赖模型与解释器的质量；③实验仅在单一DevOps场景（GPU集群）进行，泛化性待进一步验证。

---

## 13. Fuzzy Pattern Matching in Ordered Structures

**arXiv ID:** 2608.25032 | [PDF](https://arxiv.org/pdf/2608.25032v1)

**作者:** Armen Kostanyan `[一作]` (American University of Armenia), Arevik Harmandayan `[通讯]` (American University of Armenia)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对线性序列和层次结构的模糊模式匹配算法，并给出了对应的状态机模型。

**💡 创新点**

引入轨迹代数替代传统前缀函数，动态维护匹配历史；同时将模糊匹配扩展到有序树上。

**🔧 技术方法**

基于KMP思想的轨迹代数、预处理、树的前序遍历、堆栈管理、转换系统建模等技术。

**📊 数据集**

未给出真实数据集，实验使用示例字符集 Σ={1,2,3,4,5} 与定义的模糊符号 S、M、L 以及构造的字符串与树。

**📈 对比分析**

理论分析表明算法时间复杂度为 O(mn)（线性）和 O(mN)（树），空间为 O(m²) 与 O(m²H)，相较于传统动态规划/前缀表法在空间上更优；性能在实验示例中与理论一致。

**⚠️ 局限性**

算法对阈值选择敏感；轨迹长度上限导致空间为 O(m²)，对大规模数据可能不够；树高 H 影响堆栈空间；未在噪声或更复杂模糊模型下进行实验验证。

---

## 14. MSR-IVA: Masked Structural Residual Independent Vector Analysis for State-Aware Fusion of Structural MRI and Dynamic Functional Network Connectivity

**arXiv ID:** 2608.24978 | [PDF](https://arxiv.org/pdf/2608.24978v1)

**作者:** Victor Solomon `[一作]`, Jingyu Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出一种基于 masked structural residual IVA（MSR‑IVA）的多模态融合框架，将结构 MRI 与多种动态功能网络连通性状态进行融合，并通过状态掩码处理缺失状态。

**💡 创新点**

创新点在于：① 通过共享结构变换加上状态特定残差实现结构表示的软共享；② 采用状态掩码只对存在的状态参与损失，既保留了完整样本，又避免了缺失状态的干扰；③ 通过残差正则化调控共享与差异之间的权衡。

**🔧 技术方法**

主要技术包括：独立向量分析（IVA）、Kotz 源先验、矩阵对数体积正则化、残差正则化以及状态掩码机制。

**📊 数据集**

使用 ADNI 公开数据集：573 名受试者（CN/MCI/AD），包含 T1‑加权 sMRI 与 rs‑fMRI；提取 66 维结构特征与 1378 维 dFNC 特征（两个状态），并在 124 名同时出现两状态的受试者上进行交叉状态分析。

**📈 对比分析**

与 IP‑IVA、无共享模型、硬共享模型对比，MSR‑IVA 在 5 次随机初始化下平均提升匹配源耦合 6.5%（从 0.8957 到 0.9540），降低跨源依赖 15.7%（从 0.0598 到 0.0504），并在交叉状态结构相似度上取得 0.9177（介于 0.2978 与 1.0000 之间）。

**⚠️ 局限性**

局限性：仅在 ADNI 两个 dFNC 状态下评估；缺失状态的掩码策略未能探索更复杂缺失模式；未验证模型在临床预测或生物标志物识别任务中的实际性能；交叉状态相似度仅针对 124 名共同受试者计算，未覆盖全部样本。

---

## 15. SIMGUIDE: Procedurally Grounded Multi-Context Representations for Personalized Agent Planning

**arXiv ID:** 2608.24888 | [PDF](https://arxiv.org/pdf/2608.24888v1)

**作者:** Chirag Shah `[一作]` (University of Washington), Chirag Shah `[通讯]` (University of Washington)

**通讯引用:** 6629 | [OpenAlex ID](https://openalex.org/A5064398705)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向多领域用户上下文的结构化表示方法——Sims，并在此基础上引入程序化示例（Procedural Grounding）来辅助AI进行冲突决策；同时构建了 47 题、9 个用户的诊断性基准 SimBench，用于评估 AI 在不同上下文激活时是否能生成正确计划，并对多种模型（GPT‑4o、Claude Sonnet 4.5、Llama 3.3 70B、Llama 3 8B）以及基于 LoRA 的参数化微调方案进行了系统实验。

**💡 创新点**

创新点主要包括：
1) 将用户偏好拆分为按生活领域（工作、家庭、健康等）划分的 typed Sims，显式赋予优先级；
2) 通过在每条约束后加入“过去决策实例”实现程序化 grounding，提升 LLM 对约束的可执行性；
3) 设计了 SimBench——一个仅能通过上下文激活正确计划的诊断式基准，检验 AI 是否真正能识别并利用多上下文信息；
4) 证明了表示格式（结构化 vs. 传统平面）是首要设计变量，并提出按 Sim 类型路由的参数化适配策略。

**🔧 技术方法**

技术手段：
- Prompt 设计：基于 typed Sims 的结构化提示、程序化 grounding 示例；
- Retrieval‑Augmented Personalization (RAG) 作为对比基线；
- LoRA 微调：在任务匹配的合成数据、用户级数据以及 Sim‑type 数据上训练不同的 LoRA adapter；
- 冲突决策：使用约束优化公式实现优先级与硬约束的仲裁；
- 评估：采用跨模型 LLM 判别器计算 Preference Adherence (PA)、Plan Correctness (PC)、Conflict Resolution Accuracy (CRA)。

**📊 数据集**

数据集：
- SimBench：47 题、9 个合成用户的任务与 Sims；
- τ‑bench（100 题子集）：在现有任务中注入 SimProfiles，用于跨域验证；
- Amazon Product Reviews：29,587 条评论，用于测试域不匹配下的 LoRA 微调效果。

**📈 对比分析**

比较方法与性能：
- 与 No‑Profile、Flat、RAG、Sim、Sim+G、Sim‑A 六种提示形式对比；
- 指标：PA 为主评估标准，PC 与 CRA 为补充指标；
- 结果：
  * 在 GPT‑4o 上，Sim+G 相比 RAG 提升 7.9 PA 分，显著 (p=0.013)。
  * 在 τ‑bench 上，Sim+G 在 GPT‑4o 与 Claude 上分别提升 5.4% 与 14.8% PA，均显著。 
  * LoRA 方面，任务匹配的 Synthetic LoRA 使 ROUGE‑L 提升 12.8 点；Sim‑type LoRA 在 72% 路由准确率下比通用 LoRA 高 7.3 点；
  * Per‑user LoRA 在样本量约 40 条时出现过拟合，效果低于通用 Synthetic LoRA。

**⚠️ 局限性**

局限性：
- SimBench 与 LoRA 训练数据均为合成，样本量有限，缺乏真实行为数据，限制了统计功效；
- 评估使用 LLM 判别器，可能存在自评偏差；
- 小模型（如 Llama 8B）在当前提示与 grounding 下性能低落，提示与模型容量匹配仍需进一步研究；
- 目前未覆盖多模态或更大规模真实任务，实际部署效果仍待验证。

---

## 16. Teaching Geometric Proof with Tech: Pitfalls and Possibilities

**arXiv ID:** 2608.25117 | [PDF](https://arxiv.org/pdf/2608.25117v1)

**作者:** Hwei-Shin Harriman `[一作]` (Carnegie Mellon University), Joshua Sunshine `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过教师访谈和工具评测，分析几何证明教学技术缺口，提出四条设计准则。

**💡 创新点**

识别并系统化教师需求与工具不足的匹配问题，提供可落地的技术与人机交互改进建议。

**🔧 技术方法**

采用访谈、功能需求映射、工具可用性评估等方法，并基于功能需求构建评估表。

**📊 数据集**

使用18名美国高中几何教师的访谈数据以及33款现有工具的功能清单。

**📈 对比分析**

通过需求映射表比较工具满足度，发现大多工具仅部分满足需求，缺少图形注释、自动反馈和多方案支持。

**⚠️ 局限性**

受限于数据规模与工具多样性，研究未涵盖所有可用工具且缺乏纵向用户体验实证。

---

## 17. VLM-based automatic multi-granularity graph representation of building layouts for design informatics

**arXiv ID:** 2608.24886 | [PDF](https://arxiv.org/pdf/2608.24886v1)

**作者:** Song Guo `[一作]` (Massachusetts Institute of Technology), Weimin Zhuang `[通讯]` (Tsinghua University)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5104182618)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了面向公共建筑（以学术图书馆为案例）自动构建多粒度布局图（LoG1-3）的完整方法，并通过 VLM 实现零注释自动提取；

**💡 创新点**

创新点包括：① 引入建筑布局的多粒度图层（Level‑of‑Graphs）框架，② 设计了四步 VLM‑based 自动图谱构建流水线（节点识别、边推断、文本解析、图 coarsening），③ 证明不同粒度对不同任务的最佳匹配；

**🔧 技术方法**

技术手段主要是 Gemini3‑Pro 视觉‑语言模型（Zero‑Shot）用于节点与边推断，后续利用图神经网络（GCN、GraphSAGE、GAT、GINE）和传统回归模型（SVR‑RBF、Ridge、XGBoost）进行下游任务；

**📊 数据集**

使用了来自 ArchDaily 的 147 张全球学术图书馆平面图数据集，并以人工标注的 LoG3 图作为参考；

**📈 对比分析**

通过与人工标注图的节点/边匹配、结构相似度、功能分布余弦相似度等指标验证 VLM 图谱与人工图高度一致（匹配率≥92%），在布局质量评估（Spearman ρ≈0.61）和区间功能预测（Macro‑F1≈0.65）任务中，各粒度表现均优于或等同于人工图，且更具计算效率；

**⚠️ 局限性**

局限性包括：① VLM 对跨区连通性推理仍不够精准；② 仅评估了图书馆类型，缺乏对其他公共建筑的验证；③ 仅提取了节点与连边信息，未涵盖面积等几何属性，限制了更细粒度分析；

---

## 18. Targeting the Attention Heads Behind Object Hallucination in LLaVA

**arXiv ID:** 2608.24966 | [PDF](https://arxiv.org/pdf/2608.24966v1)

**作者:** Armaan Sandhu `[一作]` (University of Massachusetts), Hima Kammachi `[通讯]` (University of Massachusetts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套从诊断到干预的流程，先识别与图像错报相关的注意力头，然后针对这32个头进行LoRA微调和推理时的 grounding 控制，显著降低视觉语言模型的对象错报。

**💡 创新点**

创新点在于将解释性诊断直接转化为可操作的局部改造：先用注意力下降度筛选头，再用逐头消融验证影响，最终把结果用于局部 LoRA 和推理时惩罚，验证了头选择比单纯 LoRA 容量更关键。

**🔧 技术方法**

技术包括：注意力分析与 head 选取、逐头消融筛选、LoRA 参数微调（Q/K/V 投影）、基于注意力权重的 grounding 罚分器、DPO 风格的训练目标以及与 SPIN、VCD 的固定预算对比。

**📊 数据集**

数据集为 COCO val2014 的图像与标注，使用 COCO 的对象标签计算 CHAIR 指标，进行 400 张保留图像的评估，并在 200 张图像上做随机头对照实验。

**📈 对比分析**

与基线、单独 LoRA、单独 grounding 以及 SPIN/VCD 进行比较；LoRA+Grounding 在 CHAIRs 由 0.370 降到 0.230，CHIArI 由 0.156 降到 0.096，下降比例分别为 37.8% 与 38.5%，在固定 token 预算下依旧保持并进一步提升减噪效果，证明干预有效且不只是长度导致的误差。

**⚠️ 局限性**

局限包括：方法仅在 LLaVA-1.5-7B 与 COCO prompt 上验证，缺乏长度匹配的严格对比；仅评估 CHAIR 指标，未对生成质量、信息量与可读性做人类评估；注意力作为 grounding 代理可能不完全可信；若对象标签不完整，可能误判为错报。

---

## 19. On the Representational Geometry of Dynamic Programs

**arXiv ID:** 2608.25034 | [PDF](https://arxiv.org/pdf/2608.25034v1)

**作者:** Richard F. M. Lim `[一作]` (Bowdoin), Ruriko Yoshida `[通讯]` (Aalborg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从几何角度深入研究动态规划（DP）问题的长度泛化问题，构建了图结构、热带多项式以及延伸Newton多面体之间的三重同构，并通过该框架揭示了标准神经网络在面对更长输入时失效的根本原因。

**💡 创新点**

创新点包括：1) 提出了图-多项式-几何三元同构的理论框架，证明了三种语言在形式化层面完全等价；2) 通过该框架解释了热带注意力在 DP 长度泛化上的优势；3) 揭示了系列-并行以及终端仅操作两种组合方式在生成所有 DP 树形结构时的不足，给出了长度泛化失败的几何与结构性原因。

**🔧 技术方法**

技术手段主要包括热带代数（min,+）半环、热带多项式及其延伸Newton多面体、图同构与可约算子（series/parallel composition、edge substitution 等）以及几何投影与裁剪方法，结合形式化的同构与余量消除实现对 DP 结构的完整描述与分析。

**📊 数据集**

本文为理论分析性工作，没有使用具体的实验数据集；若有实验则未在论文中公开说明。

**📈 对比分析**

通过理论证明与具体示例对比，说明标准序列模型在长度外推（OOB）时性能急剧下降，而热带注意力在少数 DP 任务上表现出优异的 OOD 泛化；然而论文重点在解释与证明，而非系统的数值性能评测。

**⚠️ 局限性**

主要局限在于：仅关注 (min,+) 结构，无法通过半环替换将长度 T 的几何结构映射到长度 T+1；系列-并行构造无法生成所有 DAG 结构；终端仅操作亦无法覆盖所有 DP 组合；因此，仅靠这两种组合与热带半环不足以实现完整的长度泛化。

---

## 20. Fusing Perceptual Vision Experts with Multimodal Large Language Models for Explainable Plant Disease Diagnosis: From Benchmark Imagery to Real-World Robotic Field Validation

**arXiv ID:** 2608.24934 | [PDF](https://arxiv.org/pdf/2608.24934v1)

**作者:** Ranjan Sapkota `[一作]` (Cornell University), Manoj Karkee `[通讯]` (Cornell University)

**通讯引用:** 7820 | [OpenAlex ID](https://openalex.org/A5013737840)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种混合层次多代理框架（H2MAF），通过先决策级融合两个CNN专家，再由多模态大型语言模型（MLLM）在语义级对其进行仲裁，并生成可解释的农业决策报告。

**💡 创新点**

创新点在于：①将视觉专家的置信度与语义推理结合的两阶段融合；②在真实机器人采集的田间数据上验证了MLLM仲裁的有效性；③提出并量化了MLLM风险评估的“个性化”校准问题，并给出了约束解码的设计原则。

**🔧 技术方法**

使用技术包括：EfficientNet‑B3 与 ConvNeXt‑Tiny 两个结构迥异的CNN、Google Gemma 4 E4B 与 Alibaba Qwen3.5 4B 两个开源多模态LLM、基于JSON的上下文对齐、Zero‑shot推理、以及结构化输出的解析与评估。

**📊 数据集**

数据集涵盖：PlantDoc（2,922张，27类）作为公开“in‑the‑wild”基准；Cornell Stage 2（4,215张，3类）与 Stage 4（7,227张，3类）为闭源、连续视频捕获的机器人田间实测数据，共计14,364张训练/验证图像与1,370张测试图像。

**📈 对比分析**

方法通过与单一CNN基线、传统两阶段检测、以及其它多模型融合方案对比，表现为：在PlantDoc上MLLM仲裁将Top‑1准确率由63.9%提升至68.5%（提升7.6pp），在 Stage 2/4 上虽然CNN已达到96–99.8%准确率，MLLM仍在冲突样本上提供正向提升（最高+13.8pp），并在风险分类上 Gemma 与 Qwen 显示出显著的校准差异。

**⚠️ 局限性**

主要限制包括：PlantDoc 测试集样本稀少导致类别评估方差高；Stage 2 冲突样本极少，难以评估仲裁效果；连续采集的图像存在时间泄漏，可能夸大CNN准确率；MLLM 的无监督推理易产生覆盖率不佳或误报，需进一步约束解码；未对 MLLM 进行领域特定微调，潜在性能提升空间。

---

## 21. A Pathway for Assessing Grey Literature: Leveraging AI to Extract Conference Metadata and Organiser Information from Calls for Papers

**arXiv ID:** 2608.24926 | [PDF](https://arxiv.org/pdf/2608.24926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 22. Self-Explanation Tutor for Active Study of CS1 Worked Examples

**arXiv ID:** 2608.25180 | [PDF](https://arxiv.org/pdf/2608.25180v1)

**作者:** Arun-Balajiee Lekshmi-Narayanan `[一作]` (University of Pittsburgh), Peter Brusilovsky `[通讯]` (University of Pittsburgh)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于大型语言模型（LLM）的自我解释辅导系统ESSE，并在CS1（Java入门）课程中部署，实时评估学生对工作示例每行代码的解释并给出概念层面反馈；

**💡 创新点**

将LLM用作判别者，可即时、无参考答案地评估自由文本解释的正确性与完整性，并通过概念计数提供可操作的反馈，从而突破传统评估的可扩展性瓶颈；

**🔧 技术方法**

使用GPT‑4o mini（或类似LLM）进行链式思维提示，计算正确性（0/1）和完整性（0–1）以及缺失/包含的概念；配合ICAP框架的主动学习设计；

**📊 数据集**

基准数据来自8名学生在课堂中产生的409条解释（其中407条获得反馈）以及对这些解释的1,696条MTurk众包评分；此外还利用公开数据集对模型基线进行对照；

**📈 对比分析**

与单一专家标签和可靠过滤后的众包标签进行比较；LLM在正确性上F1约0.91（专家）/0.93（众包），完整性AUC约0.67/0.76，显示较强一致性；LLM反馈促使学生更频繁修订，解释的概念覆盖率显著提升；

**⚠️ 局限性**

限制：仅在单一小班（8人）中进行试点，后测完成率低；只评估一种LLM和提示配置；完整性评估仍存在校准偏差，缺乏大规模验证和多模型比较。

---

## 23. GaussVLA: Geometry-Aware Spatial Reasoning for Vision-Language-Action Model

**arXiv ID:** 2608.24959 | [PDF](https://arxiv.org/pdf/2608.24959v1)

**作者:** Md Selim Sarowar `[一作]` (Yeungnam University), Sangtae Ahn `[通讯]` (Kyungpook National University)

**通讯引用:** 2580 | [OpenAlex ID](https://openalex.org/A5016211373)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了GaussVLA模型，将冻结的语义与深度特征升维为3D高斯原语，并结合Depth-Aware Chain-of-Thought实现几何感知与高效推理；

**💡 创新点**

创新点在于用3D高斯令牌化捕获位置、方向和置信度信息，同时在流匹配条件下进行非自回归几何链式思维，配合Mamba线性时间序列模型实现参数效率；

**🔧 技术方法**

采用了Gaussian Spatial Tokenizer (GST)、Depth-Aware Chain-of-Thought (DA-CoT)、Mamba SSM、流匹配动作头、傅里叶位置编码以及稀疏注意力机制；

**📊 数据集**

在LIBERO、LIBERO-PRO、Meta‑World、CALVIN和SO‑101真实机器人等数据集上进行实验；

**📈 对比分析**

在LIBERO上平均成功率达93.5%（比SpatialVLA提升19.7%），Meta‑World与CALVIN上表现排名靠前，实时推理12.97 ms/步，参数仅200M，显著优于7B级别模型；

**⚠️ 局限性**

在LIBERO‑PRO的极端视角与任务语义扰动下鲁棒性不足，需改进相机标定、视角增强及语言适应能力。

---

## 24. Auto-Policy, not Auto-Skill: Compiled Agent Skills for the Physical World

**arXiv ID:** 2608.25091 | [PDF](https://arxiv.org/pdf/2608.25091v1)

**作者:** Zhonghao Zhan `[一作]` (Imperial College London), Hamed Haddadi `[通讯]` (Imperial College London)

**通讯引用:** 10067 | [OpenAlex ID](https://openalex.org/A5043326652)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了 Edge Skillguard，一个在 Agent Skills 内嵌的类型化授权层，用于防止借用权限导致物理世界的意外行为。

**💡 创新点**

首次识别并命名“借用权限”攻击类别，并提供基于有限状态机和属性授权的可执行安全边界，填补了现有 Skills 格式中的缺口。

**🔧 技术方法**

结合有限状态机、属性基授权（类似 OPA/Rego）、JSON Schema 策略验证、以及 NATS/Tailscale 边缘消息中间件，实现了可执行的安全检查。

**📊 数据集**

实验使用 60 条攻击请求和 60 条正常请求的手工生成数据集，并在 5 倍规模（300+300）上进行扩展验证，未使用公开大规模数据集。

**📈 对比分析**

与无保护、仅租约、自然语言 M2M 等基线对比，Edge Skillguard 在所有 60/60 攻击请求上实现 100% 拒绝率，正常请求全部通过；延迟在本地 NATS 约 273 µs，跨 Tailscale 约 5.7 ms，评估决策本身仅需微秒级。

**⚠️ 局限性**

局限在于守卫手工编写、单一边缘测试环境、缺乏自动编译/推理器、无法防御可信主体被破坏或传感器伪造等更高级攻击。

---

## 25. What Are We Measuring? Bonding, Trust, and the Evaluation of Human-Robot Relationships

**arXiv ID:** 2608.24915 | [PDF](https://arxiv.org/pdf/2608.24915v1)

**作者:** Imran Khan `[一作]` (University of Warwick), Imran Khan `[通讯]` (University of Warwick)

**通讯引用:** 29421 | [OpenAlex ID](https://openalex.org/A5072011650)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了将人机关系质量划分为两维——信任与社会绑定，并构建四种关系状态模型（回避、功能性、依赖、共生），以期提升人机交互的状态感知与适配策略。

**💡 创新点**

创新点在于将信任与社会绑定视为独立构念，阐明它们在预期、时程、身体表现与伦理风险上的差异，进而形成二维关系空间和对应的四个行为/风险配置。

**🔧 技术方法**

技术上主要依赖多模态生理与行为信号（如生理压力、亲近距离、同步性）对绑定状态进行估计，并将其作为机器人在线适配的决策变量。

**📊 数据集**

文中未使用实验数据集，主要以现有文献综述与理论阐释为依据。

**📈 对比分析**

由于缺乏实验验证，本文未提供对比方法或性能指标，只在理论层面讨论不同关系状态下的适配策略与失败阈值。

**⚠️ 局限性**

局限性包括：缺乏实证研究验证模型有效性；未给出可操作的绑定测量工具；对绑定与信任的分离假设仍需通过大规模多模态实验进一步检验。

---

## 26. Parallelizable Gradient-Based Optimization For Multi-Objective MaxCut

**arXiv ID:** 2608.25098 | [PDF](https://arxiv.org/pdf/2608.25098v1)

**作者:** Jingjuan Huang `[一作]` (Ohio State University), Ismail Alkhouri `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种可微分的多目标最大割(MO-MaxCut)求解框架 MO-QUCO，并实现了 GPU 并行版本 pMO-QUCO

**💡 创新点**

创新点在于：①将多目标问题通过线性标量化转化为带符号权重的单目标 MaxCut，并使用邻接矩阵的二次形式；②对带符号权重的 PGA 固定点进行理论分析，阐明其与 Pareto 前沿的关系；③利用批量随机初值与多偏好向量的组合，并结合 1-bit-flip 细化，突破传统 WSM 只能得到支持解的局限

**🔧 技术方法**

核心技术包括：可微分的梯度下降（Projected Gradient Ascent）在 [-1,1]^n 约束下求解邻接式二次目标；批量化与 GPU 并行矩阵乘法加投影实现高效搜索；线性标量化、偏好向量采样、偏好条件固定点多面体分析；以及 1-bit-flip 本地搜索和非支配集合维护

**📊 数据集**

主要使用 MO-MaxCut benchmark（两实例，K=3、K=4，42 节点 46 条边，层权重 i.i.d. N(0,1)）以及在补充材料中给出的不同规模与目标数的额外图实例

**📈 对比分析**

与四类基线（DPA‑a、DCM、ε‑CM、WSM）和 QAOA（模拟与硬件）进行对比。MO-QUCO 在 118s（CPU）和 0.9s（GPU）内即可达到 99.9999% 的超体积（HV）并在 1-bit-flip 细化后获得 100%；相比之下 DPA‑a、DCM 等精确方法在相同目标数下耗时数百至数千倍；ε‑CM、WSM 在速度和精度均落后；QAOA 在理想采样下仍需 360–10,000s，MO-QUCO 在同类硬件上实现 37×–65× 的速度提升

**⚠️ 局限性**

局限性包括：①方法依赖于偏好向量采样，偏好覆盖不足可能导致某些非支持 Pareto 点无法被发现；②对极端冲突目标的稳健性尚未系统评估；③GPU 并行实现需要高端显卡，CPU 版本仍受限于批量大小；④理论分析主要聚焦于邻接式二次目标，其他二次形式（如拉普拉斯+扰动）在符号权重下的固定点性质仍待研究

---

## 27. Drift Variation Autoencoder: Unifying Generation and Representation Learning through Conditional Posterior Flow Matching

**arXiv ID:** 2608.25138 | [PDF](https://arxiv.org/pdf/2608.25138v1)

**作者:** Jiarui Cao `[一作]` `[通讯]` (Chinese University of Hong Kong), Jiarui Cao (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于后验匹配的 Drift Variation Autoencoder（DVA）框架，用单一生成目标同时训练表示学习与生成器；

**💡 创新点**

证明在高斯路径下，表示缺失项为零当且仅当编码器满足后验充分性（P(X|Z)=P(X|C)），并且通过条件流匹配实现该后验；

**🔧 技术方法**

采用条件流匹配（Conditional Flow Matching）与加权清洁预测损失，利用高斯噪声路径构造可样本化目标；

**📊 数据集**

在合成的 CrossGeom‑4 多模态数据集（欧氏、球面、双曲空间视图）上进行验证；

**📈 对比分析**

与独立解码器基线对比，显示编码器充分利用观测因子、可重建可见模态、且联合目标注意力可将未观测共享因子的误差降低 90‑93%，但在无条件生成时模式不平衡仍明显；

**⚠️ 局限性**

局限性包括：理论仅在无穷容量与精确场下成立，有限网络无法保证全局最优；后验充分性不等同于语义或最小充分性；对非高斯、离散或流形数据的可识别性仍需进一步研究。

---

## 28. MacroAgent: Regularity-Aware Macro Legalization with LLM-Agent-Designed Contour Algorithms

**arXiv ID:** 2608.24946 | [PDF](https://arxiv.org/pdf/2608.24946v1)

**作者:** Jiaxi Jiang `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9451 | [OpenAlex ID](https://openalex.org/A5051340429)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MacroAgent 框架，实现宏合法化时通过四个阶段（聚类、轮廓生成、模板匹配和跨簇细化）实现高正则化且低位移的宏放置。

**💡 创新点**

创新点在于将宏合法化抽象为几何问题，利用 LLM 代理自动设计多样化的轮廓生成算法，从而兼顾正则化与位移，显著提升鲁棒性和性能。

**🔧 技术方法**

核心技术包括 LLM 代理（Prompt+生成-评估循环）、几何轮廓生成（alpha shape、MST、grid 等）、匈牙利算法匹配以及线性规划跨簇细化。

**📊 数据集**

使用 TILOS（易例）和 Chipyard（难例）两个公开芯片设计数据集进行实验，包含多种宏类型与宏数量。

**📈 对比分析**

在学术流（DREAMPlace+HeLEM-GR）和工业流（Cadence Innovus）中，与 DREAMPlace、Sequence Pair 以及 Innovus 原生宏放置相比，MacroAgent 在正则化、布线长度、拥塞和时序（TNS/WNS）方面分别提升 3–8 倍正则化、3–5% 布线长度、68% TNS，且在大多数芯片上实现全局合法化。

**⚠️ 局限性**

局限性主要在跨簇细化仍依赖传统 LP/启发式方法，LLM 生成的轮廓在极端稀疏或高密度布局中可能需要手动调优；此外，LLM 对 EDA 领域的专业知识仍有限，导致需要外部人类输入做指导。

---

## 29. Dynamic Influence-Weighted Distillation for Single-IMU Activity Recognition

**arXiv ID:** 2608.24904 | [PDF](https://arxiv.org/pdf/2608.24904v1)

**作者:** Bingxuan Xie `[一作]` `[通讯]`, Bingxuan Xie

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究探究了在训练阶段使用多位置IMU传感器信息是否能提升仅使用右臂IMU的动作识别模型，并提出了动态影响加权(DIW)方法来实现可控蒸馏。

**💡 创新点**

创新点在于为每个训练样本分别对logit和特征蒸馏目标生成样本级、组件级门控，并通过一次性候选AdamW更新和元集反馈动态调整权重，从而更精准地利用训练专属信息。

**🔧 技术方法**

技术上使用了知识蒸馏（logit与特征蒸馏）、固定权重KD、动态门控、一次性look‑ahead、有限差分探测、批归一化、残差卷积网络以及元学习式的梯度方向反馈。

**📊 数据集**

采用了WEAR户外运动活动识别数据集，该数据集包含22名受试者、19类标签、同步的四个IMU（右臂、右腿、左腿、左臂）加速度数据。

**📈 对比分析**

与纯监督训练和固定权重KD进行比较，采用subject‑disjoint五折OOF宏F1评估，DIW将宏F1从0.5618提升到0.6385，分别比监督高约7.7个百分点、比固定KD高约6.7个百分点。

**⚠️ 局限性**

局限性包括仅在单一数据集和单一保留传感器位置上验证，未分离元标签带来的优势，未评估训练过程的额外计算成本及内存占用，且缺乏多种随机种子、数据集和模型结构的进一步验证。

---

## 30. Simultaneous inference of environmental and interaction forces in collective dynamics

**arXiv ID:** 2608.25181 | [PDF](https://arxiv.org/pdf/2608.25181v1)

**作者:** Nipuni de Silva `[一作]` (Clarkson University), James M. Greene `[通讯]` (Clarkson University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `729e5870-4135-47f5-97f2-e3974d07b5dc` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了同时推断环境力 f 与交互核 φ 的变分学习框架，涵盖半参数和全非参数两种实现，并在多种一阶与二阶集体动力学模型上进行实验与模型选择。

**💡 创新点**

在传统只学习交互核的框架基础上加入环境/自驱动力学习，统一处理一阶与二阶系统；给出非参数推断与半参数组合的方法；提出基于推断结果的自动模型选择与稀疏识别机制。

**🔧 技术方法**

采用变分最小二乘正则化、局部 B‑spline / 分段多项式基函数、变量投影求解；利用 L²(ρ) 权重误差、样本复杂度与噪声鲁棒性分析来评估模型性能。

**📊 数据集**

使用合成轨迹数据：Kuramoto 同步模型、Self‑Propelled Particle（SPP）聚集模型、光感趋向（phototaxis）模型；通过不同数量、时长与噪声水平的轨迹集进行训练、验证与测试。

**📈 对比分析**

通过特征恢复误差、残差误差、轨迹误差、样本复杂度与噪声鲁棒性等多指标进行对比；实验表明两种方法在轨迹重建与预测上性能相近，半参数方法在已知环境力形式时恢复更精确；模型选择能够正确识别存在的力并实现稀疏模型。

**⚠️ 局限性**

需要足够覆盖的轨迹空间；近零距离区间样本稀少导致交互核估计不佳；半参数方法受限于事先假设的环境力形式，若不匹配会严重失效；计算成本随代理数的平方增长；对高维系统和非同质动力学的推广仍有限。

---

## 31. Probabilistic Performance Analysis of Parallel Signature Search Strategies in Multi-Level Tree Networks

**arXiv ID:** 2608.25087 | [PDF](https://arxiv.org/pdf/2608.25087v1)

**作者:** Jingwei Li `[一作]` (Stony Brook University), Thomas G. Robertazzi `[通讯]` (Stony Brook University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一种概率模型，能够在任何文件被读取前预测多层树形文件集合中签名搜索（pattern search）的完成时间。

**💡 创新点**

创新点在于：① 统一处理未知与已知签名计数两种信息条件；② 覆盖三类文件容量（≤1、≤K、无限）与五种并行搜索策略；③ 为每个公式赋予“exact / exact‑in‑regime / plug‑in / asymptotic / bound”等精度标签，并给出对应的误差范围；④ 将组合计数、极值理论与生成函数方法结合，完成对已知计数下多元超几何占用分布的精确描述。

**🔧 技术方法**

主要技术手段包括：节点扫描时间的混合分布建模；顺序与并行阶段使用顺序统计量和极值极限；中心极限定理用于子树总和的正态近似；多元超几何分布与生成函数用于已知计数下子树占用概率；数值卷积和蒙特卡洛仿真用于验证与补充不易解析的情况。

**📊 数据集**

实验数据采用仿真生成的树形文件集合。示例中使用约10^4个文件构成四层、每层 fan‑out 为10 的树；其它实验覆盖不同层数、fan‑out、签名概率、容量参数等组合。

**📈 对比分析**

对比方法：将解析公式的预测结果与 10^5–10^6 次离散事件蒙特卡洛仿真所得完成时间做对比，误差在 0.1%–1% 之间；在真实多核原型（Apple M1）上实施五种搜索策略，测得的速度提升与理论相符，尤其在层级同步开销可忽略时；S_3 与 S_4 在理论上差距约 20% 但实验中几乎相同，说明同步成本对性能影响显著。

**⚠️ 局限性**

局限性包括：① 假设无处理器数量限制，实际环境需考虑处理器池大小与同步开销；② 仅考虑无取消（no‑cancellation）策略；③ 采用均匀子树、签名位置均匀、继承属性等理想化假设；④ 对容量受限下某些宽度分布（m_i ≥ rK）仍未给出完整解析公式。

---

## 32. Can You Trust Frozen Hematology Foundation Models under Acquisition Shift?

**arXiv ID:** 2608.25148 | [PDF](https://arxiv.org/pdf/2608.25148v1)

**作者:** Jai Kumar Sharma `[一作]` (Virginia Tech), Peeyush Tapadiya `[通讯]` (Accenture)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文对冻结的血液学基础模型在不同采集域下的准确性、校准、曝光以及类别先验鲁棒性进行了系统评估。

**💡 创新点**

创新点包括：①跨域准确性与校准的双维度评估；②曝光审计揭示模型预训练与数据集重叠；③在类别偏移场景下提出并验证了标签无监督的Class‑Balanced Re‑standardization（CBR）方法。

**🔧 技术方法**

使用技术主要为：冻结编码器、线性探针与1‑NN探针、温度缩放、ECE等校准指标、标签无监督特征适配以及CBR。

**📊 数据集**

数据集：源域为Acevedo（PBC）; 目标域为MLL23/Metafer、Matek‑LMU/M8、Raabin；实验限定在五类白细胞（WBC）的交集上。

**📈 对比分析**

比较方法：在源域宏F1几乎饱和（0.98–0.997），但在目标域宏F1下降34–72%，排名重排；线性探针与1‑NN探针在跨域排名相关系数低；校准误差从源域的0.004升至目标域的0.35，CBR可将目标域ECE降至≈0.29。

**⚠️ 局限性**

局限性：仅涵盖五类WBC，难以完全分离曝光与扫描器相关的分布偏移；CBR对极端类别不平衡敏感；1‑NN排名不稳定；缺少完全无泄漏的DinoBloom目标域。

---

## 33. See More, Detect Less? Taming Information Leakage in Multi-View Anomaly Detection

**arXiv ID:** 2608.25168 | [PDF](https://arxiv.org/pdf/2608.25168v1)

**作者:** Shang-Fu Chen `[一作]` (National Taiwan University), Kai-Lung Hua `[通讯]` (Microsoft Taiwan Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Global‑Local Attention Driven 框架用于多视角异常检测，通过 Vision 基础模型提取特征并结合 MMA 与 OGA 两模块实现跨视角信息受限的融合。

**💡 创新点**

创新点：①引入跨视角信息泄漏概念并提出信息受限广播机制（token 替换、Sigmoid 门控、温度调节）避免重建误差缩小；②MMA 采用线性注意力 + 视角重要性 + token gating 实现 O(N) 本地跨视角融合；③将全局对象级与局部 token 级融合结合，提升定位与检测性能。

**🔧 技术方法**

技术：冻结 DINOv2 ViT‑Base/16 编码器；Transformer 解码器；MMA 线性注意力（ReLU² 核）+ 视角重要性矩阵 + token gating；OGA 全局对象 token + Sigmoid + 温度 0.7 + token 替换；Hard‑mining 训练策略；多视角重建损失 L_GC。

**📊 数据集**

数据集：Real‑IAD（30 类、5 视角）与 MANTA‑Tiny（38 类、5 视角），包含正常与异常样本，并提供像素级缺陷标注。

**📈 对比分析**

方法对比：与单视角基线（Dinomaly、MambaAD、DiffusionAD 等）以及多视角方法 MVAD 对比；在 Real‑IAD 与 MANTA‑Tiny 上在样本、图像、像素级 AUROC/AP/F1 以及 P‑AUPRO 等指标均超过所有对比方法，尤其在像素级定位上显著提升。

**⚠️ 局限性**

局限性：仅适用于重建式框架，需冻结编码器；对缺失或极少视角时仍需多视角信息；未探讨跨模态或自监督特征更新；计算开销虽低于传统全注意力但对实时部署仍有挑战。

---

## 34. SNAP-KG: Streaming Node Assignment via Projection for Knowledge Graph Entity Integration

**arXiv ID:** 2608.25149 | [PDF](https://arxiv.org/pdf/2608.25149v1)

**作者:** Jui-Chien Lin `[一作]` (Rensselaer Polytechnic Institute), Oshani Seneviratne `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了 SNAP-KG，一个能在知识图谱流式实体整合中进行多视角图聚类并通过投影器实现无图推理的框架。

**💡 创新点**

首次将图结构多视角聚类与无监督投影器结合，实现无重训练的流式推理；并将 GNN-Transformer 学习转移到 MLP 进行知识蒸馏。

**🔧 技术方法**

多视角 GNN、Transformer 编码、对比学习、GNN-MLP 蒸馏、K-means、ANN 等技术。

**📊 数据集**

ACM、DBLP、IMDB、YELP、MAG、OGB-WikiKG2（约 2.4M）等多种多视角图数据集。

**📈 对比分析**

与 BMGC、DEMM、MGDCR 等转导式基准相比，在五大基准上聚类质量相近；流式推理速度提升数百至数千倍；在实体解析和链式预测任务中实现 62–75%（常规模型）和 97%（大规模模型）的候选搜索缩减，性能几乎不下降。

**⚠️ 局限性**

对跨域泛化的限制（例如 MAG 聚类准确率仅 ~60%），对 K 选择敏感；需要重新训练 GNN 才能支持新关系类型；极大规模多关系图仍需进一步优化训练成本。

---

## 35. GRAPE: Gradient Refinement and Progress-Aware Exploitation for Query-Efficient High-Dimensional Bayesian Optimization

**arXiv ID:** 2608.25116 | [PDF](https://arxiv.org/pdf/2608.25116v1)

**作者:** Richard Cornelius Suwandi `[一作]` (Chinese University of Hong Kong Shenzhen), Feng Yin `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两阶段的局部贝叶斯优化框架GRAPE，用于高维黑盒函数的查询高效优化；

**💡 创新点**

创新点在于先用闭式采集函数进行梯度后验细化，再以“条件期望下降量”作为利用度量，从而兼顾下降概率与下降幅度，理论证明梯度细化单调减小不确定性，利用阶段在后验收敛时趋于真正的最速下降；

**🔧 技术方法**

主要技术包括高斯过程模型、梯度后验推导、闭式梯度细化采集函数、期望截断高斯条件下降量计算、投影梯度上升以及局部贝叶斯优化循环；

**📊 数据集**

实验数据集包括MNIST与CIFAR-10图像的黑盒对抗攻击，以及BoLT提供的5126个LLM提示的四个不同嵌入维度（128-768）进行提示优化；

**📈 对比分析**

与随机搜索、全局BO（EI、长度尺度先验）、局部第一阶方法（如DAGGER、SFO、PEBO）、信赖域方法和二阶方法（MNewton）等基线比较，GRAPE在对抗攻击上平均缩短查询次数约5.4倍，在LLM提示优化上在所有维度下均优于第二佳方法，日志简单遗憾下降量至少降低3.8个log单元；

**⚠️ 局限性**

局限性包括：仅为局部方法，易受初始化与多模态问题影响；条件下降量假设梯度后验为正态，可能在高曲率区误估；以及GP三阶复杂度随查询数增长，需进一步稀疏或近似方法来提升规模。

---

## 36. Hydra: Phase-Aware Workload Characterization of LLM Inference across Edge SoC Generations, Backends, and Quantization Levels

**arXiv ID:** 2608.25053 | [PDF](https://arxiv.org/pdf/2608.25053v1)

**作者:** Amir Taherin `[一作]` (Northeastern University), David Kaeli `[通讯]` (Northeastern University)

**通讯引用:** 7728 | [OpenAlex ID](https://openalex.org/A5061128237)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 NVIDIA Jetson AGX 边缘 SoC 上的大语言模型（LLM）推理进行阶段化、跨后端、跨平台的工作负载特征化，提出 Hydra 框架并发布 107K 条 per-prompt 追踪数据。

**💡 创新点**

提出统一的 per-prompt 计时与硬件遥测对齐的 schema，实现 HuggingFace Transformers 与 llm.cpp 两个后端在同一框架下的可比性；并系统地横向比较 SoC 代际、模型族、精度格式、输入输出长度。

**🔧 技术方法**

Python/C++ 计时器、CUDA 同步、NVIDIA Management Library (NVML)、jetson 自带的 telempar、GPU/CPU/内存/功耗/热度监控，以及权重量化 GGUF 等。

**📊 数据集**

IFEval 541 条指令跟随提示（13–345 token），以及两个长度敏感子集（S1 输入 1k/3k/5k token；S2 输出 1k/3k/5k token）。

**📈 对比分析**

通过 Hydra 的相位化时间、系统利用率与能耗指标，对 3 代 Jetson（Xavier、Orin、Thor）、13 模型、5 量化格式进行横向对比，发现新一代 SoC 与量化提升显著降低 decode 速度/能耗，但总延迟仍受后端调度和模型规模影响。

**⚠️ 局限性**

仅评估 NVIDIA Jetson 系列，未覆盖激活量化、KV 缓存压缩、多任务/并行推理、混合专家/多模态模型；量化研究仅限权重量化；硬件遥测采样不均匀、单线程推理等。

---

## 37. Toward Machine Learning with the Unit as a Primitive: Learning from Unit-Linked Events

**arXiv ID:** 2608.25118 | [PDF](https://arxiv.org/pdf/2608.25118v1)

**作者:** Heyang Gong `[一作]` `[通讯]`, Heyang Gong

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“unit”这一持久个体原语，定义任务先声明个体集合及同一性判定，然后通过tokenizer生成上下文token，所有个体共享单一响应模型Rθ，实现对单位间共享结构的学习。

**💡 创新点**

创新点在于把个体身份从隐式转为显式任务声明，给出了统一的共享形式接口(Tϕ,Rθ)并用信息论边界区分oracle预测价值、可达性、学习误差和单行观测不可辨别性；同时给出了理论可学习性与单位归属判断的必要条件。

**🔧 技术方法**

使用概率模型、条件独立性假设、KL 与 TV 上下界、线性预测器（inner‑product 形式）以及tokenizer 的构造与推导；并给出了单行不可能性、重复链路可辨别性等理论证明。

**📊 数据集**

论文未使用任何实际数据集，全部为理论推导与抽象示例。

**📈 对比分析**

对比方法主要为理论边界与极限例子，未给出实验性能指标；通过证明单行样本无法区分异质与同质、同单元对的可辨别性等，展示了方法的理论优势。

**⚠️ 局限性**

局限包括：缺乏经验验证、未证明学习规则能收敛、tokenizer 设计依赖先验假设、未覆盖因果估计细节、对单一线性读取的依赖以及对单位身份不确定时的推断误差未完全可控。

---

## 38. Authenticated Data Structures for Dynamic Workloads

**arXiv ID:** 2608.25206 | [PDF](https://arxiv.org/pdf/2608.25206v1)

**作者:** Ziheng Shangguan `[一作]` (Brown University), Dahlia Malkhi `[通讯]` (University of California Santa Barbara)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种适用于动态工作负载的频率感知认证数据结构——Huffman‑Merkle Tree（HMT），通过分层存储和批量重构实现高效的成员证明与更新。

**💡 创新点**

创新点在于将Huffman编码的最优访问路径与分层架构结合，使用批量更新与计数最小草图(CMS)估计访问频率，再通过多种迁移策略动态地将热点元素迁移到热层，实现对频率变化的低成本适配。

**🔧 技术方法**

核心技术包括：Huffman‑Merkle Tree、分层（冷热两层）设计、批量重构与局部交换、Count‑Min Sketch频率估计、LFU promotion cache、Ratio‑Based、Sliding‑Window、Dynamic‑Control等迁移策略。

**📊 数据集**

使用以太坊主网的账户访问日志（≈1000万块）进行真实工作负载回放，另外通过合成Zipf分布的微基准测试验证重构成本。

**📈 对比分析**

与传统的Merkle Patricia Trie（MPT）和Unified Binary Tree（UBT）进行对比。结果显示：Sliding‑Window策略在哈希输入量上比MPT少约2.4倍、比UBT少约0.34倍；证明大小比MPT少约0.18倍、比UBT少约0.55倍；Ratio‑Based和Dynamic‑Control也显著优于基线。

**⚠️ 局限性**

局限性包括：需要手工调参（阈值、窗口大小、重构频率等），在频率变化极慢的工作负载下可能无法充分利用热层；批量重构导致短期内更新延迟；设计复杂度高，部署和验证成本较传统单层结构高。

---

## 39. Tabular Foundation Models for Multi-View Information Cascade Popularity Prediction

**arXiv ID:** 2608.25048 | [PDF](https://arxiv.org/pdf/2608.25048v1)

**作者:** Wenting Zhu `[一作]` (Beijing University of Posts and Telecommunications), Xi Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于Tabular Foundation Model（TFM）的信息级联流行度预测框架——TFM4POP，并结合Neural ODE动态编码器实现对多视图（文本、视觉、结构、表格）信息的统一建模与预测。

**💡 创新点**

创新点包括：① 将四种异构视图压缩为表格列，利用预训练TFM的两路注意力实现高阶交互；② 引入IA3参数高效微调，兼顾预训练知识与真实级联分布；③ 构建事件中心化的多视图级联基准EventCas，丰富数据维度。

**🔧 技术方法**

使用的技术主要有：Tabular Foundation Model（TabPFN/LimiX）、Neural ODE编码器、交叉注意力融合、IA3微调、LLM（Qwen3-Max）生成推理、PCA压缩、GraphWave/NetSMF结构嵌入、无泄漏的out-of-fold上下文训练。

**📊 数据集**

实验使用的公开数据集包括Twitter多模态级联数据和自研的EventCas（微博事件级联）两大数据集；此外对比了已有的Twitter/Weibo基准。

**📈 对比分析**

通过与13类基线（特征、统计、深度学习、LLM方法）在MSLE和MAPE两指标上进行系统比较，TFM4POP在两数据集两观测窗口均优于所有基线，MSLE提升约2%–12%，MAPE提升约2%–18%。

**⚠️ 局限性**

主要局限包括：① 对特征压缩维度的依赖，需要人工调参；② LLM生成的推理文本可能引入噪声；③ 预训练TFM与真实级联分布的差距仍存在；④ 在不同平台或更大规模数据上的泛化尚未充分验证。

---

## 40. CAT-GS: Balanced Multimodal Learning via Calibrated Gating and Fusion Surgery

**arXiv ID:** 2608.24947 | [PDF](https://arxiv.org/pdf/2608.24947v1)

**作者:** Mahir Shahriar Tamim `[一作]` (North South University), Nabeel Mohammed `[通讯]` (North South University)

**通讯引用:** 1456 | [OpenAlex ID](https://openalex.org/A5062072064)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CAT‑GS在训练过程中通过校准教师置信度、门控、梯度预算重分配与融合层PCGrad三阶段控制多模态学习的梯度动态

**💡 创新点**

将教师可靠性校准与阈值门控、预算重分配、融合层梯度投影三者整合为单一优化时控制器

**🔧 技术方法**

使用温度缩放、EMA平滑、阈值门控、梯度预算重分配与融合层PCGrad投影

**📊 数据集**

CREMA‑D、AV‑MNIST、VGGSound、UR‑FUNNY、CG‑MNIST、AVE、CMU‑MOSI等多模态与情感识别数据集

**📈 对比分析**

与OGM‑GE、G2D、UMT等基线相比，CAT‑GS在多数数据集上提升约1‑2个百分点并保持收敛稳定

**⚠️ 局限性**

对预训练教师可靠性依赖强，且在大规模、噪声较多的数据集（如VGGSound）上提升有限，教师质量下降时性能会显著下降

---

## 41. aipsy-judge: A Specialized, Psychologist-Corrected Local Judge for the Psychological Safety of Conversational AI

**arXiv ID:** 2608.24899 | [PDF](https://arxiv.org/pdf/2608.24899v1)

**作者:** Michael Keeman `[一作]` (Keido Labs), Anastasia Keeman `[通讯]` (Keido Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究LLM评判器在心理安全评估中的可靠性，发现前沿模型存在系统性偏差，随后提出心理学家校正的本地评判器；

**💡 创新点**

创新点在于对每个评估指标使用心理学家加权的目标，并通过分层失效过采样进行蒸馏，得到全本地化、可复现的评判模型，解决了自我偏好与尾部盲点问题；

**🔧 技术方法**

技术手段包括LoRA参数高效微调、分层失效过采样、基于Gemma-4-26B的MoE模型、JSON链式思维输出、以及Krippendorff α与ICC一致性评估；

**📊 数据集**

使用的数据集为aipsy-bench基准，包含20个多轮情景、3,000条AI回复，并以单一心理学家对173条主要项进行评分作为锚点；

**📈 对比分析**

在与三大前沿评判器及开源基础模型对比时，aipsy-judge-1.0在综合ICC（0.75 vs 0.64）、危机检测κ（0.82 vs 0.65）和失败召回率（0.49 vs 0.22）等指标上显著提升，且保持92%危机检测召回；

**⚠️ 局限性**

局限性包括仅使用单一心理学家标注，缺乏多评审者验证；对同理心等轴的误判仍较高；模型对长对话或动态情境的适用性未知；实现需要较大内存与GPU资源。

---

## 42. Multimodal Injury Risk Prediction in Tennis

**arXiv ID:** 2608.25126 | [PDF](https://arxiv.org/pdf/2608.25126v1)

**作者:** Francisco Erramuspe Alvarez `[一作]` (Monmouth University), Ling Zheng `[通讯]` (Monmouth University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了面向网球运动员的多模态预测运动员准备度框架，集成了可穿戴设备、问卷调查、垂直弹跳测试和比赛视频四大数据来源；

**💡 创新点**

创新点在于首次将视频运动捕捉、可穿戴心率/睡眠监测、主观问卷与生理测量结合，通过监督学习自动学习权重来生成全局运动员准备度评分（ARS），并对潜在受伤部位进行定位；

**🔧 技术方法**

主要技术包括深度学习模型（MLP、LSTM）用于物理能力预测、XGBoost回归/分类器用于整体健康与受伤风险预测、计算机视觉+运动分析用于视频风格分类，以及多元线性回归来确定ARS权重；

**📊 数据集**

实验使用了来自9名大学网球运动员的16周数据，包含85个特征、82804条记录，数据涵盖心率变异性、休息心率、睡眠效率、训练负荷、问卷得分、垂直弹跳得分和比赛视频帧；

**📈 对比分析**

在整体健康预测中，XGBoost回归器取得MAE 3.82、R² 0.838；在受伤风险分类中，XGBoost分类器AUC-ROC 0.695、准确率 0.852、F1 0.100；在物理能力预测中，LSTM在训练/验证损失分别为0.0360/0.6059；总体上各子模型表现优于传统线性/树模型；

**⚠️ 局限性**

主要局限包括样本量极小（仅9名运动员），受伤部位预测仅到上下/左右四分区，视频风格分析仅基于平均击球数粗略划分，且未在更大多样化人群上验证模型泛化能力。

---

## 43. The Von-Neumann State-Space Transformer for neural decoding

**arXiv ID:** 2608.25088 | [PDF](https://arxiv.org/pdf/2608.25088v1)

**作者:** Morteza Sarafyazd `[一作]` `[通讯]` (BrainCo), Morteza Sarafyazd (BrainCo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了可在神经解码任务中通过低秩指令库实现可编程前馈计算的 Von‑Neumann State‑Space Transformer。

**💡 创新点**

创新点在于用低维状态空间轨迹作为指令指针，令每个 token 的前馈权重按低秩指令生成，既提升样本效率，又可测量指令使用率。

**🔧 技术方法**

技术包括：Transformer 结构、选择性状态空间机（SSM）、快权重记忆、低秩指令库与指令编码器。

**📊 数据集**

使用三种运动皮层神经解码基准（MC_RTT、MC_Maze、Area2_Bump）和两种文本数据集（tiny‑Shakespeare、WikiText‑2）。

**📈 对比分析**

与标准 Transformer 在相同参数、数据和上下文长度下对比，VN‑SST 在样本稀缺、参数紧凑情形下取得更高行为解码 R² 并在文本任务上更低 perplexity，且长上下文时性能提升。

**⚠️ 局限性**

局限在于对指令库规模的探索不足，且在数据充足或更复杂任务中优势减弱，模型可解释性和泛化仍需进一步研究。

---

## 44. Demystifying Reinforcement Learning Post-Training of Language Models

**arXiv ID:** 2608.24949 | [PDF](https://arxiv.org/pdf/2608.24949v1)

**作者:** Donovan Clay `[一作]` (University of Washington), Natasha Jaques `[通讯]` (University of Washington)

**通讯引用:** 3730 | [OpenAlex ID](https://openalex.org/A5046953322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统实验拆解LLM RL后训练（RL post‑training）的机制，分别探究基模型分布、奖励函数和提示分布对学习结果的影响；

**💡 创新点**

创新点在于：①通过SFT+ / SFT- 预训练改造基模型概率，实证“覆盖原则”与稀疏奖励的局限；②利用密集奖励（Levenshtein距离、PRM）克服探索瓶颈，学习新行为；③揭示随机（spurious）奖励对模型性能的影响完全由提示分布决定，区分窄域与宽域效果；

**🔧 技术方法**

主要技术包括：RL post‑training框架、SFT、DPO、KL约束、奖励重塑（稀疏/密集）、过程奖励模型（PRM）、Levenshtein距离奖励、策略熵监控；

**📊 数据集**

实验数据集：电影字幕（Cornell Movie‑Dialogs Corpus）生成任务、AIME 2025 数学问题、WildChat（宽域提示）、DeepScaleR（窄域数学提示）、OLMo 3 RLVR混合提示、GSM8K、MMLU、IFEval 等；

**📈 对比分析**

比较方法：在相同模型尺寸下对比基模型、SFT+、SFT- 通过稀疏奖励与密集奖励训练后的成功率、熵变化与准确率；结果显示：密集奖励可将几乎零概率行为提升至 50%+ 成功率，稀疏奖励在无覆盖时失败；在窄域提示下随机奖励可提升 MATH 准确率，但在宽域提示下导致熵升高、所有基准性能骤降；

**⚠️ 局限性**

局限性：①稀疏奖励无法突破基模型覆盖限制；②实验在受控toy环境，难以直接推广至大规模真实LLM；③密集奖励设计需要任务可验证的结构化奖励，无法普适；④对随机奖励的负面影响取决于提示分布，需进一步探究泛化策略。

---

## 45. What Do Audio-Visual Synchronization Metrics Actually Measure?

**arXiv ID:** 2608.25157 | [PDF](https://arxiv.org/pdf/2608.25157v1)

**作者:** Jai Kumar Sharma `[一作]` (Virginia Tech), Peeyush Tapadiya `[通讯]` (Accenture)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对四种主流音视频同步度量（AV‑Align、ImageBind AV‑relevance、JavisScore、Synchformer/DeSync）进行联合可靠性审计，构建无人工标注的合成失同步 oracle 并评估其在不同失同步类型下的单调性、预处理敏感度、排名不确定性、跨指标一致性以及与 PEAVS 代理的一致性。

**💡 创新点**

首次以统一可靠性协议对已部署的音视频同步度量进行全面审计，揭示指标在不同维度上的轴向分裂及互相不一致性。

**🔧 技术方法**

使用合成失同步 oracle、Kendall τ、Krippendorff α、分层采样置信区间、分层线性与 k‑NN 融合、split‑conformal 预测区间等技术。

**📊 数据集**

基于 AVSync15（VGGSound 1500 段真实同步视频）以及 45 段 MMAudio 生成的音视频。

**📈 对比分析**

通过与人工代理 PEAVS 的相关性、跨指标一致性和融合实验进行比较，结果显示 Synchformer/DeSync 在时间偏移检测上最高 τ≈0.84，但在内容破坏和 PEAVS 对齐上仅为 τ≈0.20；无单一指标能在两轴上同时表现最佳，融合也未显著提升。

**⚠️ 局限性**

局限包括仅使用 PEAVS 作为感知代理且未进行人类直接偏好评估，指标评估仍受数据集规模与场景多样性限制。

---

## 46. Synergising Local Geo-Environmental Characteristics with Spatial Context for Enhancing Landslide Susceptibility Mapping

**arXiv ID:** 2608.24956 | [PDF](https://arxiv.org/pdf/2608.24956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 47. Flower Hub: A Reproducible Benchmarking Platform for Federated Learning in Simulation and Deployment

**arXiv ID:** 2608.25114 | [PDF](https://arxiv.org/pdf/2608.25114v1)

**作者:** Yan Gao `[一作]` (Flower Labs), Nicholas D. Lane `[通讯]` (Flower Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出Flower Hub平台，提供可执行、可复现的联邦学习基准，支持跨仿真与部署的一键式运行；

**💡 创新点**

创新点在于将基准打包为可执行、版本化的应用，解耦应用与基础设施，统一标准化元数据、依赖固定与评估工作流，并内置系统级指标监控；

**🔧 技术方法**

主要技术包括Flower框架的SuperLink/SuperNode、仿真引擎、flwr run统一命令、标准化打包模式、版本控制与实时监控框架；

**📊 数据集**

使用了多领域数据集：医疗影像分割（BraTS）、金融欺诈检测（PaySim银行）、法律指令调优（LexGLUE等）、网络钓鱼URL检测、语音标签（UrbanSound8K）；

**📈 对比分析**

通过六种聚合算法（FedAvg、FedProx、FedAvgM、FedAdam、FedAdagrad、FedYogi）在五个任务上进行基准实验，评估指标分别为Dice、PR‑AUC、F1、ROC‑AUC和Accuracy，FedProx在多数任务表现最佳，FedOpt族在多数任务表现差；

**⚠️ 局限性**

局限性包括未覆盖个性化、隐私、鲁棒性等高级场景；部署实验规模有限；未进行超参数细调，仅提供基线结果。

---

## 48. Routed Graph Handoff: Adaptive Format Selection for Multi-Agent LLM Delegation

**arXiv ID:** 2608.25277 | [PDF](https://arxiv.org/pdf/2608.25277v1)

**作者:** Pratyay Banerjee `[一作]` (Amazon), Ankit Chadha `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究多代理LLM系统的通信格式，比较自然语言与结构化图的交付方式，并提出一种基于轻量LLM路由器的路由图交付（Routed Graph Handoff）方案，实现低成本且高成功率的多代理协作。

**💡 创新点**

创新点在于：①设计了一种可压缩的有类型DAG语法，用于显式编码任务依赖；②引入仅155令牌的LLM路由器，动态在图与自然语言之间切换；③系统性揭示并解决了结构与灵活性之间的权衡，实现了Pareto改进。

**🔧 技术方法**

主要技术包括：Claude Sonnet 4.5（或GPT‑5 mini）作为 orchestrator；约束解码生成有类型图；图感知执行器提示；单一的、零训练的LLM路由器分类器；以及基于BERT/TF‑IDF等的结构化重编码方法。

**📊 数据集**

使用四个多代理基准：BrowseComp、BFCL v3、τ‑bench retail 和 AppWorld，共计1,052条任务轨迹。

**📈 对比分析**

通过与 NL‑only、NGH‑only、Oracle 等基线对比，系统在 τ‑bench 取得 +12.7pp、在 BrowseComp 取得 +8.7pp 的提升，BFCL 与 AppWorld 维持或略高于 NL；压缩率平均提升 2.1×，路由器仅增加 0.15% 的令牌开销，整体实现成本与性能的 Pareto 改进。

**⚠️ 局限性**

限制主要在于：路由器仅按任务内容粗粒度选择格式，缺乏实例级执行时动态切换；图模式对高度创造性或全新协作模式的适用性未知；系统需要图感知执行器提示，且在更广泛模型与领域上的验证尚未完成。

---

## 49. Retrieved But Not Reliable: A Survey on Attacks, and Defenses in Retrieval-Augmented Generation

**arXiv ID:** 2608.24977 | [PDF](https://arxiv.org/pdf/2608.24977v1)

**作者:** Minh Tran `[一作]` (University of Science), Suhang Wang `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Retrieval-Augmented Generation（RAG）系统的鲁棒性进行系统综述，统一威胁模型，提出攻击与防御的管线感知分类，并总结基准与可解释性方法。

**💡 创新点**

提出了面向整个 RAG 管线的统一威胁模型和攻击目标（准确性、隐私、公平性）与防御阶段（检索、重排序、生成、追溯）的分类框架，弥补了以往碎片化综述的不足。

**🔧 技术方法**

通过文献检索、引用链追踪和系统性梳理，构建攻击与防御的分层分类表，结合阐释、基准与案例对比。

**📊 数据集**

综述中主要引用的实验数据集包括 Natural Questions、HotpotQA、SQuAD 等问答基准，以及针对隐私泄漏的 Common Crawl 等。

**📈 对比分析**

对已有工作进行了对比表述，评估攻击成功率、检索精度、答案准确率等指标，指出不同防御方案在各基准上的优劣，但未提出新的模型或性能提升。

**⚠️ 局限性**

研究领域快速演进，难以覆盖所有新兴 RAG 变体；部分攻击具多目标性；防御多阶段交叉；并未对黑盒或有限查询环境的可迁移性进行系统评估。

---

## 50. Rethinking the Transferable Adversarial Attacks and Robust Defense in Federated Learning

**arXiv ID:** 2608.25133 | [PDF](https://arxiv.org/pdf/2608.25133v1)

**作者:** Zuobin Xiong `[一作]` (University of Nevada Las Vegas), Wei Li `[通讯]` (Georgia State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了联邦学习中可迁移对抗样本的攻击与防御机制，提出了理论分析与鲁棒防御框架。

**💡 创新点**

创新点在于首次从理论角度关联数据分布差异与对抗样本迁移性，并提出基于SVD的特征演化模块与联邦对抗训练的组合防御。

**🔧 技术方法**

采用FedProx联邦学习框架、SVD驱动特征演化模块（SDFEM）、联邦对抗训练（FAT）、梯度裁剪等技术。

**📊 数据集**

使用CIFAR-10和SVHN这两个公开图像分类数据集。

**📈 对比分析**

在IID与非IID、多模型与不同攻击方法下，将所提方法与FedProx、FAT及集中式上限进行对比，结果显示在鲁棒性上显著提升，且清晰准确率基本保持。

**⚠️ 局限性**

局限在于仅针对迁移式对抗攻击，未覆盖数据投毒或模型逆向等攻击；且对计算成本与通信开销的详细分析缺失。

---

## 51. Scalable Self-Supervised Learning for Multiphase AC-OPF in Distribution Systems with Topology Reconfiguration

**arXiv ID:** 2608.25095 | [PDF](https://arxiv.org/pdf/2608.25095v1)

**作者:** Hoang T. Nguyen `[一作]` (Massachusetts Institute of Technology), Priya L. Donti `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种自监督的多相配电网AC‑OPF代理，能够在大规模切换拓扑下快速、可行地预测最优运行点。

**💡 创新点**

创新点包括：①将Penalty+SLFS框架引入自监督训练，使模型直接对AC‑OPF目标与约束进行学习；②使用M步雅可比近似实现高效微分；③利用Sherman‑Morrison‑Woodbury更新逆导纳矩阵以处理拓扑变化；④在推理时采用顺序线性化可行性寻求（SLFS），确保约束可行且速度极快。

**🔧 技术方法**

采用了自监督损失、可微分固定点潮流求解、M步Jacobian近似、SMW逆更新、SLFS可行性修复、GPU友好的矩阵向量运算以及Primal‑Dual Hybrid Gradient求解器。

**📊 数据集**

使用IEEE 13/123/240/906/8500节点配电网示例，训练集20万样本（含可行与不可行），测试集2k样本，负载、DER可用率与拓扑切换随机采样。

**📈 对比分析**

与IPOPT、线性化OPF、FSNet等方法比较，Penalty+SLFS在所有规模下平均约束违约<10⁻⁴，最优性差<0.2%（小中型）/1.5%（8k节点），相较IPOPT加速幅度高达3个数量级，SLFS比FSNet速度提升近两到三百倍，并保持更低约束误差。

**⚠️ 局限性**

局限性包括：在极端低DER/高负载等边缘情形仍存在轻微最优性损失；SLFS迭代次数对速度有影响；对未知拓扑分布仍需额外训练；逆导纳矩阵更新在极大规模系统下仍是瓶颈；模型依赖GPU实现，CPU效率相对较低。

---

## 52. Sequential Object Placement Optimization with Convex Decomposition

**arXiv ID:** 2608.25162 | [PDF](https://arxiv.org/pdf/2608.25162v1)

**作者:** Yuezhe Zhang `[一作]` (Technical University of Darmstadt), Georgia Chalvatzaki `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了SOPO-CD框架，利用可微碰撞约束在连续空间内求解物体放置问题

**💡 创新点**

通过将自由空间分解为凸多面体，将凸内部点约束化为闭式可微非线性优化，并提供解析的一阶二阶导，显著加速SQP求解

**🔧 技术方法**

使用凸包的V/H表示、线性/非线性规划、闭式解析导数、Clarabel SQP求解器、并行随机初始等技术

**📊 数据集**

在2D Tangram、2D Tetris、3D Bin Packing和真实Allegro手+Xarm Tangram任务上进行评估

**📈 对比分析**

与网格搜索、DCOL等方法比较，SOPO-CD在Tangram、Tetris、Bin Packing中速度提升10–200×，成功率和占用率接近或优于基准

**⚠️ 局限性**

仅采用轴对齐的凸分解，贪心选择缺乏前瞻性，未覆盖更复杂的非凸场景，凸包选择不穷举

---

## 53. FLINT: Efficiently Leveraging High Bandwidth Flash for Capacity-Scalable LLM Inference Acceleration

**arXiv ID:** 2608.25062 | [PDF](https://arxiv.org/pdf/2608.25062v1)

**作者:** Geraldo F. Oliveira `[一作]` (Huawei Technologies Switzerland Ag), Ji Zhang `[通讯]` (Huawei Technologies Co Ltd)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于工作负载驱动的高带宽闪存（HBF）子系统，用以在单加速器上实现容量可扩展的大语言模型（LLM）推理加速。

**💡 创新点**

创新点包括：硬件 burst‑buffer 控制器、幻影平面刷新机制和只读 FTL，将 HBF 与 HBM 协同工作，避免了静态预取、刷新占用和传统 SSD FTL 的冗余。

**🔧 技术方法**

使用的技术包括：多平面并行读、按层级动态合并 cache line 请求、循环刷新、闪存页面级映射表等。

**📊 数据集**

评估使用六款生产 LLM（包括五个 MoE 模型和一个稠密模型）以及多批量、多上下文长度的推理工作负载。

**📈 对比分析**

与 HBM+SSD、HBM‑only、以及之前的 H^3 方案比较，改进版在相同硬件下的解码吞吐量提升 1.2k‑6.2 倍、能耗降低 408‑6.8 倍，并以更少 GPU 包满足 50 TPOT SLO。

**⚠️ 局限性**

局限性在于需要对闪存进行一次完整的写入和刷新周期，且对极端低延迟或频繁权重更新的场景支持有限。

---

## 54. Agentic World Analysis (AWA) - an alternative way to explore systems and support decision making

**arXiv ID:** 2608.24896 | [PDF](https://arxiv.org/pdf/2608.24896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 55. AI-Ready Research Workflows in Computational Social Science: Lessons on Building a Shared Language for Interdisciplinary Collaboration

**arXiv ID:** 2608.24914 | [PDF](https://arxiv.org/pdf/2608.24914v1)

**作者:** Joan Giner-Miguelez `[一作]` (Barcelona Supercomputing Center), Mercè Crosas `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 21517 | [OpenAlex ID](https://openalex.org/A5056421325)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文建立了一个AI工作流，支持在MareNostrum超级计算机上对OpenAlex数据库进行大规模查询、分析和LLM分类等任务，为STS研究单元提供可重复、可扩展的计算框架。

**💡 创新点**

创新点在于将敏捷方法与跨学科协同设计相结合，并通过模型驱动工程创建共享词汇，成功桥接了领域研究与工程实现之间的语言与流程鸿沟。

**🔧 技术方法**

使用了Agile/JIRA、COMPSs并行框架、LLM分类模型、字典标签、模型驱动工程元模型，以及RO-Crate/FAIR元数据工具等技术。

**📊 数据集**

主要使用了OpenAlex 460M学术记录数据库（含约60M全文）以及相应的人工验证样本。

**📈 对比分析**

与传统单GPU、单模型推理相比，该工作流将推理成本从约9000 GPU小时压缩到92小时（使用96张A100 GPU），实现了约100倍加速，同时保持了模型准确性。

**⚠️ 局限性**

局限性包括验证资源不足、可复现性包装不完整、FAIR元数据生成手工且缺乏自动化，需要跨机构基础设施与工具生态的支持。

---

## 56. DeMMO: Longitudinal and Cross-Disease Modelling of Digital Mobility Outcomes via Multi-Task Learning

**arXiv ID:** 2608.25073 | [PDF](https://arxiv.org/pdf/2608.25073v1)

**作者:** Menghui Zhou `[一作]` (University of Sheffield), Po Yang `[通讯]` (University of Sheffield)

**通讯引用:** 8278 | [OpenAlex ID](https://openalex.org/A5008276130)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 DeMMO 框架，用于联合学习多疾病、多结果的数字移动性指标（DMO）随时间演化的关系，支持跨疾病、跨结果的无配对参与者信息共享。

**💡 创新点**

创新点包括：①自动从纵向 DMO 映射中学习符号跨疾病/跨结果关系图，实现选择性信息共享；②将融合 Lasso、稀疏组 Lasso 与关系学习整合为单一可解释的多任务模型；③通过稳定性选择揭示不同结果的长期稳定 DMO 组合。

**🔧 技术方法**

技术手段包括：纵向线性回归 + 融合 Lasso 平滑、稀疏组 Lasso 特征选择、对称关系矩阵自动学习（交替优化），以及稳定性选择用于特征稳健性评估。

**📊 数据集**

使用了 Mobilise‑D 多中心纵向数据集，包含 24 个标准化 DMO、5 次访视、4 个临床预测目标（PD H&Y、PD MDS‑UPDRS III、MS EDSS、PFF SPPB），排除了 COPD 因数据缺失。

**📈 对比分析**

与 9 种基线（岭回归、套索、FRoTS、MAGPP、两层 MLP、Rank‑N、RankSim、ACCon 等）在 5 次访视、4 个结果上进行比较；DeMMO 在整体 nMSE（0.722）和加权相关（0.515）上显著优于最强基线，并在绝大多数单任务上名列前茅。

**⚠️ 局限性**

局限性包括：仅评估 4 个疾病/结果；深度学习基线表现不佳；非凸变体无明显提升；需要在独立队列和更长随访中验证泛化能力；模型对噪声和缺失较为敏感。

---

## 57. FrontierChallenge: Evaluating Scientific Workflow Completion

**arXiv ID:** 2608.24979 | [PDF](https://arxiv.org/pdf/2608.24979v1)

**作者:** Liangcai Su `[一作]`, Xinyu Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

这篇论文内容为占位文本，并未包含任何真实研究内容。

**💡 创新点**

由于缺乏具体研究细节，无法指出任何创新点。

**🔧 技术方法**

文中没有提及任何技术实现。

**📊 数据集**

也没有使用任何实际数据集。

**📈 对比分析**

没有对比方法和性能评估，所有结果均为合成示例。

**⚠️ 局限性**

主要局限在于缺乏实验数据与真实结果，无法验证任何主张。

---

## 58. A General Framework for Metropolis-Adjusted Dikin Walks: Dimension-Square Mixing on Polytopes and Log-Det Walks on Spectrahedra

**arXiv ID:** 2608.25273 | [PDF](https://arxiv.org/pdf/2608.25273v1)

**作者:** Zhao Song `[一作]`, Lichen Zhang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并分析了精确度量的Metropolis调整Dikin步行，在保留提议行列式和反向二次形式的基础上，消除未居中的主项，得到可用二阶工具控制的中心化波动。

**💡 创新点**

将提议行列式和逆二次形式统一处理，实现了在多面体与谱半球上的高效采样；提出了接受率到混合时间的转换，利用Lewis权重高精度指标以及TensorSRHT构造实现精确算术实现。

**🔧 技术方法**

Metropolis–调整Dikin Walk、二阶工具、Lewis权重、TensorSRHT、接受率到混合时间的比较论证。

**📊 数据集**

以理论多面体（n个不等式）和谱半球（n×n块）为实验对象，无具体实测数据集。

**📈 对比分析**

与Lee–Sidford正则化walk及传统log-det walk进行理论比较，证明在warm-start下混合步数为((d^2+dL^2R^2)log(w/δ))或((nd+dL^2R^2)log(w/δ))，相较以往方法在高维情况下具有更优阶数。

**⚠️ 局限性**

混合时间仍随维度d或n线性或平方增长；需要精确Lewis权重和高精度TensorSRHT，实施复杂度高；对非凸势函数或非多面体结构可能不适用。

---

## 59. User-Centered Design for Digital Patient-Navigation Tools in Oncology: Scoping Review

**arXiv ID:** 2608.24887 | [PDF](https://arxiv.org/pdf/2608.24887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 60. Improved Low-Overhead Communication-Efficient String Reconciliation and Edit Distance

**arXiv ID:** 2608.25179 | [PDF](https://arxiv.org/pdf/2608.25179v1)

**作者:** Michael T. Goodrich `[一作]` (University of California, Irvine), Claire A. To `[通讯]` (University of California, Irvine)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用可逆布隆查找表（IBLT）和局部一致解析（LCP）实现的低开销、高通信效率的字符串重合算法。

**💡 创新点**

创新点在于将IBLT与LCP相结合，构造随机语法（压缩后的小型文法），证明一次编辑仅影响 O(log n) 个非终结符，从而实现通信量仅为 O(k log^3 n) 位，计算开销为 O(n log k)。

**🔧 技术方法**

主要技术包括可逆布隆查找表、局部一致解析、完美哈希函数、随机排列与块划分，以及通过多轮递归产生的上下文无关文法。

**📊 数据集**

论文未使用具体实验数据集，而是基于理论分析和概率证明给出算法性能；若要验证，可在 DNA 序列、文件同步等大规模文本上测试。

**📈 对比分析**

与现有方法（如 O(k log n) 通信但指数级计算开销、O(n^2) 计算开销的方案）相比，该算法在保持通信量相当或更低的同时，将计算开销降低到线性，适用于长字符串且编辑距离较小的场景。

**⚠️ 局限性**

限制在于需要预先知道或多次尝试估计上界 k，且算法仍依赖随机哈希与完美哈希的高成功概率；当 k 非常大或字符串高度非重复时，通信量与计算开销仍可能显著增长。

---

## 61. Lightweight Machine Learning-Driven Monocular Sidewalk Path Extraction for Embedded Micromobility Navigation

**arXiv ID:** 2608.25178 | [PDF](https://arxiv.org/pdf/2608.25178v1)

**作者:** Lkhanaajav Mijiddorj `[一作]` (University of Oklahoma), Binbin Weng `[通讯]` (University of Oklahoma)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并评估单目视觉侧道路径提取管线，比较多种规划方法；

**💡 创新点**

通过半监督教师学生框架训练轻量级SegFormer-B0，并系统性比较BEV与图像空间规划，证明后者在嵌入式平台更优；

**🔧 技术方法**

轻量化SegFormer‑B0、OneFormer Swin‑L伪标注、半监督学习、BEV homography、距离变换、骨架图、图像空间中点与距离变换规划、EMA平滑等；

**📊 数据集**

采集的六条校园视频（22679帧）以及32帧人工标注，并利用OneFormer生成的伪标签；

**📈 对比分析**

在32帧人工标注上对比五种规划，图像空间中点平均水平误差14.3px，时延2.2ms；BEV距离变换误差65px，时延926.8ms；SegFormer‑B0 IoU 0.946，11.7ms；系统FPS>59；

**⚠️ 局限性**

仅离线评估，动态障碍处理有限，单目BEV覆盖不足，缺乏大规模标注，闭环部署未完成；

---

## 62. SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts

**arXiv ID:** 2608.25202 | [PDF](https://arxiv.org/pdf/2608.25202v1)

**作者:** Shyam Agarwal `[一作]` (Carnegie Mellon University), Bogdan Vasilescu `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了截至2026年的最大规模Spec‑Driven Development（SDD）语义数据集，涵盖470,795条规范文件、73,030个仓库、5,992个触及规范的PR以及2.42M条可追溯引用，标注工具来源并提供完整提交历史与结构特征。

**💡 创新点**

创新点在于首次系统化收集和归档SDA规范及其与代码实现的多维关联（通过PR共变、引用索引与任务‑代码映射），并为研究规范质量、工具演进与人‑AI协作提供统一可复现的数据基础。

**🔧 技术方法**

采用GitHub搜索、REST API、路径指纹匹配和手工验证等技术实现广泛文件抓取，并使用Parquet、MySQL与JSONL三种格式发布，提供灵活查询与重构。

**📊 数据集**

数据集来源于公开GitHub仓库，覆盖17个主流SDD工具及Kiro的两级目录结构，共计470k+规范文件、5,992条PR、2.42M引用以及完整的提交历史。

**📈 对比分析**

本工作不做算法对比，而是提供标准化数据层供后续研究使用；已验证的指标包括仓库星标、许可证、语言与工具分布，供社区基准评估。

**⚠️ 局限性**

局限性包括：仅包含公开仓库、PR层仅覆盖11个工具且为子样本、引用解析依赖文本与树结构的准确性、缺乏对代码生成细节与实现路径的直接观测、未来更新需逐版发布。

---

## 63. Secret MCP: Evidence-Bounded and Context-Isolated Design Specification Generation from Web Screenshots

**arXiv ID:** 2608.24944 | [PDF](https://arxiv.org/pdf/2608.24944v1)

**作者:** Yeongjin Jo `[一作]` `[通讯]`, Yeongjin Jo

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个面向多引用的截图到设计索引系统，保证每个网页引用单独生成可审计的实现导向规范。

**💡 创新点**

创新点包括：①引用作用域生成架构，保证一次请求只处理单一引用；②长图分块预处理与色彩量化的证据准备方案；③19段详细的设计规范合同；④多层隔离策略防止跨引用污染。

**🔧 技术方法**

技术手段：Node.js+TypeScript实现；Model Context Protocol（MCP）采样适配器；图像预处理、颜色量化、JSON Schema合同；本地化的 sequential sampler 与日志追踪。

**📊 数据集**

数据集：公开的 GDWEB 网页截图与元数据，实验使用三份已提交的 food‑godot 2026 设计索引作为验证集。

**📈 对比分析**

对比方法：通过构建可复现的单元/集成测试验证请求/文档一一对应、排除逻辑、合同合规性，测试运行时间约2.85秒；未对模型生成质量或视觉重构进行量化评估。

**⚠️ 局限性**

局限性：仅验证结构与词法合规，未评估语义正确性与视觉精度；对已废弃的 MCP 采样依赖；缺乏真实渲染、用户研究与成本/时延等指标。

---

## 64. Path Abstraction for Markov Reward Models

**arXiv ID:** 2608.25139 | [PDF](https://arxiv.org/pdf/2608.25139v1)

**作者:** Arnd Hartmanns `[一作]` (University of Twente), Robert Modderman `[通讯]` (University of Twente)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在马尔可夫奖励模型（MRM）上扩展路径抽象技术，证明其在期望奖励量上的正确性与单调吸收性质，并给出基于线性方程组求解的数值实现。

**💡 创新点**

创新点包括：①首次将路径抽象从无界到达概率推广到期望奖励；②使用自由单词（free monoid）视角简化定义与证明；③提供完整的数值计算公式并实现 PARI/GP 参考代码。

**🔧 技术方法**

核心技术是自由单词框架下的路径抽象定义、概率与奖励的归约运算，以及通过求解线性方程组（包含 Q = (-T)^{-1}）来计算抽象后奖励。

**📊 数据集**

论文没有使用具体实验数据集，而是以理论证明与示例图模型为依据。

**📈 对比分析**

比较方法主要是与原始 MRM 直接计算进行对比；论文指出通过路径抽象可将大模型分解为小子模型，减少线性方程组规模，理论上提高求解效率，但未给出实测性能数据。

**⚠️ 局限性**

局限性包括：①仅处理离散时间马尔可夫链，未扩展到连续时间；②实现仍为参考代码，缺乏高效优化与抽象集合选取的启发式策略；③未在真实模型上评估性能。

---

## 65. SelfGraphRAG: Bridging the Supervision Gap in Graph-Based RAG with Synthetic QA Generation

**arXiv ID:** 2608.25123 | [PDF](https://arxiv.org/pdf/2608.25123v1)

**作者:** Ben Lagnese `[一作]` (University of Maryland, Baltimore County), Manas Gaur `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 SelfGraphRAG，一种利用知识图结构自生成问答对进行图检索训练的检索增强生成框架，能够在无人工标注的私有文档集上实现多跳推理。

**💡 创新点**

创新点在于将知识图本身作为自监督信号，通过 LLM 生成多跳和邻域问答对，从而解决图检索模型缺乏训练数据的问题。

**🔧 技术方法**

技术上使用 Doc2Graph 进行文本到知识图的抽取，SynthGen 用 LLM 生成问答数据，G‑Retriever 作为图检索器训练，答案生成采用冻结的 Llama2‑7b 模型。

**📊 数据集**

实验使用 MoreHopQA、MultiHop‑RAG 以及 PubMedQA 三个多跳或分类问答基准，全部在自行构建的知识图上进行。

**📈 对比分析**

与传统 RAG、GraphRAG、LightRAG 对比，SelfGraphRAG 在多跳检索指标上提升 20‑70% 以上（例如 MultiHop‑RAG F1 从 0.98 提升至 24.62），在 PubMedQA 上准确率提升至 55.2%。

**⚠️ 局限性**

局限包括 SynthGen 生成数据的覆盖范围受两跳和三邻域超参数限制、Doc2Graph 缺少跨块实体消歧导致图不完整、未对不同检索器架构或更强生成模型进行 ablation 与泛化评估。

---

## 66. The Evolution of Binary Decompilation in the Modern Era: A Taxonomy, Literature Review, and Future Perspectives

**arXiv ID:** 2608.24955 | [PDF](https://arxiv.org/pdf/2608.24955v1)

**作者:** Omar Abusabha `[一作]` (Sungkyunkwan University), Sungjae Hwang `[通讯]` (Sungkyunkwan University)

**通讯引用:** 948 | [OpenAlex ID](https://openalex.org/A5019449591)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性综述二进制反汇编技术的发展，提出四代架构与八大任务分类，梳理评估指标与工具生态

**💡 创新点**

首次构建完整的二进制反汇编技术与评估方法的体系结构与时间线，识别研究空白与未来方向

**🔧 技术方法**

采用系统文献检索与筛选、关键词聚类、主题映射、评估指标提取、技术与工具分类等方法

**📊 数据集**

基于公开开源工具（Coreutils、glibc等）、恶意软件样本、随机程序生成器（Csmith）和竞赛数据集（LeetCode）等多种数据集进行评估与对比

**📈 对比分析**

对72篇主要研究进行定量与定性分析，揭示论文分布、出版场景、工具使用频率及指标分布；在实验中没有统一的性能基准，呈现多元化评估结果

**⚠️ 局限性**

研究覆盖不足、缺乏统一基准与可靠真值、ISA与编译器多样性低、缺少动态与神经网络验证机制，以及对法律与伦理考量缺失

---

## 67. Beyond the Chatbot: Co-Learning and Co-Teaching through a Dual-Persona Generative-AI Assistant

**arXiv ID:** 2608.24902 | [PDF](https://arxiv.org/pdf/2608.24902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 68. Understanding the Energy Scaling of Large Language Model Inference Across Context Lengths and Attention Architectures

**arXiv ID:** 2608.25096 | [PDF](https://arxiv.org/pdf/2608.25096v1)

**作者:** Molka Chkir `[一作]` (Algoma University), Arghavan Asad `[通讯]` (Algoma University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了四个开源LLM在解码阶段的能耗随上下文长度、批量大小以及注意力机制（MHA、GQA、GQA+SWA）的变化。

**💡 创新点**

创新点在于仅测量解码阶段能耗，并全面比较三种根本不同的注意力架构对能耗扩展性的影响。

**🔧 技术方法**

使用NVIDIA A100 GPU的硬件能耗计数器、FP16推理、贪婪解码以及批量化推理等技术。

**📊 数据集**

数据集采用统一的英文段落作为基准提示，生成200个输出token，覆盖多种上下文长度。

**📈 对比分析**

通过能量/token、延迟/token、延迟/请求等指标比较，发现MHA模型随上下文增长能耗显著上升，GQA和GQA+SWA能耗基本保持不变；批处理可使能耗下降约80%且延迟同步降低。

**⚠️ 局限性**

限制在于只评估了四个模型、FP16精度、单GPU环境，未考虑多GPU、不同硬件或更长生成任务，以及解码阶段以外的能耗。

---

## 69. HCC+: Hyperbolic Guarding for Certified Attention Retrieval

**arXiv ID:** 2608.24971 | [PDF](https://arxiv.org/pdf/2608.24971v1)

**作者:** Liangchen Ge `[一作]` `[通讯]`, Liangchen Ge

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出一种在Poincaré球面上对注意力检索进行确定性、查询无关的误差保证框架（HCC+）

**💡 创新点**

利用负曲率带来的指数体积增长、对数覆盖半径和维度无关的包装常数，首次在非欧几里得空间实现确定性检索证明与压缩加速

**🔧 技术方法**

超球面几何分析、Poincaré球中1中心计算（Weiszfeld算法）、边界截断、关键键守护、超球面量化（HPQ）与熵编码

**📊 数据集**

论文以理论与合成实验为主，未在实际大型语言模型数据集（如LongBench、PG19）上进行验证

**📈 对比分析**

相较于FP16基准理论上可实现6.1×的存储压缩，并给出确定性误差上界（10%或O(1/√n)），但未给出实际吞吐或精度对比

**⚠️ 局限性**

结果保守且对极大关键键集的可扩展性有限；需要对1中心近似、边界投影等细节进一步实证与优化

---

## 70. SPECTRA: Subspace-Preserving Embedding Calibration, Transport, and Replay for Fully Few-Shot Class-Incremental Audio Classification

**arXiv ID:** 2608.25054 | [PDF](https://arxiv.org/pdf/2608.25054v1)

**作者:** Giries Abu Ayoub `[一作]` (University of Haifa), Simon Korman `[通讯]` (University of Haifa)

**通讯引用:** 829 | [OpenAlex ID](https://openalex.org/A5022648850)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了SPECTRA框架，在完全少样本增量音频分类中通过冻结音频‑语言模型、轻量化残差适配器、基于低秩子空间的无样本特征重放以及推理时的转导最优传输实现持续学习。

**💡 创新点**

创新点包括轻量化残差适配器调优冻结编码器、子空间特征重放以保持旧类记忆、以及在推理阶段使用Sinkhorn最优传输细化原型。

**🔧 技术方法**

采用预训练音频‑语言模型（PENGI/CLAP）、残差MLP适配器、SVD子空间采样、Sinkhorn最优传输、交叉熵训练等技术。

**📊 数据集**

在NSynth-100、FSC-89、LS-100三个全少样本增量基准上进行评估。

**📈 对比分析**

与TAPE等现有方法对比，SPECTRA平均准确率提升约1–2%，显著降低遗忘，实验在多次随机种子下统计显著。

**⚠️ 局限性**

局限包括对子空间秩、适配器扩张比例等超参数敏感，且在噪声更大的数据集上仍需进一步提升稳健性。

---

## 71. Bayesian Flow Networks for Offline Trajectory Planning

**arXiv ID:** 2608.25163 | [PDF](https://arxiv.org/pdf/2608.25163v1)

**作者:** Ludvig Killingberg `[一作]` (Norwegian University of Science and Technology), Helge Langseth `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一种基于贝叶斯流网络（BFN）的离线强化学习框架 BFN‑RL，能够在同一模型中同时生成离散和连续状态轨迹，并通过逆动力学模型将生成的状态序列映射为可执行动作。

**💡 创新点**

创新点在于：① 用 BFNs 替代传统扩散模型，天然支持离散与连续数据；② 采用返回值（return）作为条件直接引导生成，避免额外的返回分类器；③ 将逆动力学模型与轨迹生成分离，提高了在确定性环境中的规划效率。

**🔧 技术方法**

主要技术包括：贝叶斯流网络（BFN）生成模型、逆动力学网络、返回条件的无分类器指导、基于时间 U‑Net 的架构、以及离线数据的无监督训练。

**📊 数据集**

实验数据集涵盖：MiniGrid（Empty‑Random、DoorKey、BlockedUnlockPickup）、FrozenLake、Sokoban 以及 D4RL MuJoCo（HalfCheetah、Hopper、Walker2d）等离散与连续控制任务。

**📈 对比分析**

与基准方法（Decision Diffuser、统一的离散扩散模型、BC、CQL、IQL、DT、TT、MOReL 等）进行比较；在离散任务中 BFN‑RL 与离散扩散模型相当，并在 Sokoban 上明显优于其余方法；在连续控制任务中 BFN‑RL 的平均性能与现有最优方法相近或略优，表现出可竞争的整体效果。

**⚠️ 局限性**

局限性包括：① 在部分离散任务（如 BlockedUnlockPickup）表现受训练种子波动影响，未能显著优于扩散模型；② 需要训练逆动力学网络，增加模型复杂度；③ 目前未针对连续‑离散混合环境做深入验证，且在极高维连续空间的采样效率仍待提升。

---

## 72. CVE-SAI: Counterfactual Visual Evidence-Guided Selective Attribute Indexing for Risk-Controlled E-commerce Search

**arXiv ID:** 2608.25023 | [PDF](https://arxiv.org/pdf/2608.25023v1)

**作者:** Xiaolong Sun `[一作]` (Sun Yat-Sen University), Liang Chen `[通讯]` (Sun Yat-Sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种风险控制的选择性产品属性索引方法（CVE‑SAI），在将属性值从图像中推断出来后，再通过视觉证据审核决定是否将该值写入索引。

**💡 创新点**

创新点在于：①将属性推断与索引录入分离；②使用对焦区失真（FZD）构造属性专属的视觉依赖代理并通过证据引导注意力重分配（EGAR）改进分数；③设计四项审计（证据必要性、保留性、无关变形稳定性、文本冲突），并采用独立的有限样本风险校准选择唯一的录入策略，保证不超过5%不安全录入。

**🔧 技术方法**

核心技术包括：多模态大型语言模型（如 Qwen2.5‑VL‑3B‑Instruct）用于图像+属性查询推断；视觉注意力与对焦区失真生成局部视觉证据代理；证据引导的注意力重分配；基于本体的分数冻结与阈值调节；以及基于贝塔分布的单侧置信上界进行风险控制的校准。

**📊 数据集**

使用 Amazon Berkeley Objects (ABO) 数据集，构造了五个可视化属性（颜色、图案、形状、表面处理、风格）并进行族级拆分（训练/验证/校准/测试）。

**📈 对比分析**

与 CLIP、SigLIP2、FashionCLIP、GME、MM-Embed、InternVL3、Qwen2.5‑VL、MOON 等基线进行比较。CVE‑SAI 在属性推断上 Macro VAA 72.36%/Ans‑F1 84.47%，证据定位 Patch AUPRC 49.16%；在 5% 风险预算下获得最高 Certified Write Coverage 44.50% 并将 UnsafeWWR 降至 2.36%；在受控检索中实现 NDCG@10 0.6749，且 Unsafe Auto‑Induced Exposure@10 仅 0.50%，均优于所有基线。

**⚠️ 局限性**

局限性包括：仅针对单图像产品；只评估固定本体下的 5 个属性；风险校准依赖于族级样本拆分，可能对不同商品目录不完全通用；对极其模糊或多图像商品的视觉证据生成仍存在挑战。

---

## 73. Belief Cascades Drive Persuasion in LLM Agent Networks

**arXiv ID:** 2608.25152 | [PDF](https://arxiv.org/pdf/2608.25152v1)

**作者:** Haoyi Qiu `[一作]` (University of California Los Angeles), Nanyun Peng `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个受控测试平台，用于研究大型语言模型（LLM）在真实网络拓扑中的代理对代理说服行为；

**💡 创新点**

创新点在于将说服视为多代理系统中的动态网络问题，利用图拓扑、个体先验、竞争设置和行动日志等多维度分析说服路径与机制，并揭示文本输出与实际立场转移不完全对应的现象；

**🔧 技术方法**

采用了个性化PageRank（PPR）生成先验、定向图曝光、有限信息流、九项社交行为、基于token概率的七点立场探针、以及贝叶斯回归和多路径因果关联分析等技术；

**📊 数据集**

使用了5个SNAP Twitter ego‑network图（18–42节点），55条政策声明作为实验情景，和四种LLM后端（GPT‑4o、GPT‑4.1、Gemini‑2.5‑Flash、Gemini‑2.5‑Pro）；

**📈 对比分析**

与模型在无上下文条件下的先验基线对比，评估每轮立场探针得分的变化、直接与同行传播通道的贡献以及计划与行动的匹配度；实验显示说服效果受网络结构、话题与模型先验共同决定，且行动日志比单纯文本更能捕捉真实影响；

**⚠️ 局限性**

局限性包括：先验由图位置决定导致先验与网络位置共线；实验规模仅为小型ego‑network且随机种子有限，未涵盖更大或更复杂网络；仅评估四种英文LLM，未验证跨语言、开放式动作或对人类说服数据的外推性。

---

## 74. Rollout-Decoded Reconstruction for Long-Horizon Prediction in Latent World Models

**arXiv ID:** 2608.25017 | [PDF](https://arxiv.org/pdf/2608.25017v1)

**作者:** Rishi Shah `[一作]` (E3A Healthcare), Rishav Shrestha `[通讯]` (E3A Healthcare)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种名为Rollout-Decoded Reconstruction (RDR) 的训练目标，利用在训练阶段对自由运行滚动产生的潜在状态进行解码并与真实观测对齐，从而显著提升潜在世界模型在长期预测中的有效预测时间（VPT）。

**💡 创新点**

创新点在于仅通过一个无参数的新损失项，使解码器在训练时直接接触与部署时相同的自由运行潜在状态，解决了解码器与潜在分布不匹配的问题，并在不增加模型容量的前提下实现显著性能提升。

**🔧 技术方法**

使用的技术包括：潜在世界模型（编码器-转移器-解码器）、S5 级联网络、MDP 训练中的教师强迫与多步滚动一致性、以及新的 RDR 损失；同时采用基于经验的校准与预注册实验设计。

**📊 数据集**

数据集为 Kuramoto–Sivashinsky PDE（L=22, 64 网格点），通过 ETDRK4 数值积分生成 512 训练、64 验证、64 测试轨迹（每轨迹 256 帧），并在控制实验中使用摆锤与小车摆动任务的标准模拟环境。

**📈 对比分析**

通过与基准“后验仅解码”模型以及观测空间推前预测器进行 A/B 对比，RDR 在 10 个预注册配置中以 1.71–2.50 倍的 VPT 提升（从 3.87±0.23 到 6.97±0.42 统一参数 193,568），与观测空间预测器实现相同水平；在控制任务中，RDR 在训练步数减少时提升样本效率，但若匹配训练步数则优势消失。

**⚠️ 局限性**

局限性包括：仅在单一 PDE 系统和单一网格规模上验证，实验样本量有限且 VPT 量化步长受限；控制实验仍属初步；RDR 训练成本略升高（+40% 解码器评估），且机制（解码器鲁棒性 vs 潜在动态改进）尚未彻底分离；未验证在其他系统、观测模式或更高维度环境中的泛化能力。

---

## 75. HealthBench-Psych: A Mental Health Subset of OpenAI's HealthBench

**arXiv ID:** 2608.25071 | [PDF](https://arxiv.org/pdf/2608.25071v1)

**作者:** Matthew Flathers `[一作]` (Beth Israel Deaconess Medical Center), John Torous `[通讯]` (Beth Israel Deaconess Medical Center)

**通讯引用:** 37329 | [OpenAlex ID](https://openalex.org/A5026523515)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

我们从公开的HealthBench数据集中筛选并验证了610个与精神健康相关的对话，构建了可复现的HealthBench-Psych子集，并在此子集上对20款前沿与开源LLM进行交叉评估。

**💡 创新点**

创新在于提供透明、可复现的多阶段筛选管道，首次将普通HealthBench转化为精神健康专项评测；同时揭示了前沿模型在精神健康对话中的性能相当及可测量的拒绝行为。

**🔧 技术方法**

使用LLM（Claude Opus、Claude Sonnet）做自动筛选，三名临床医生进行盲审，采用HealthBench原始评分模板与三名LLM评判员（GPT‑4.1、Claude Haiku、Gemini 2.5 Flash）进行三评委面板评分，并用自助采样/bootstrap估计不确定性。

**📊 数据集**

基于公开的5,000条多轮对话的HealthBench核心数据，筛选得到610条精神健康相关对话（HealthBench‑Psych）及其硬核子集119条（HealthBench‑Psych‑Hard）。

**📈 对比分析**

对20款模型分别生成回应，并在三评委面板下计算平均分，结果显示kimi‑k2.6、gpt‑5.5、claude‑opus‑5等前五名在面板平均分上构成统计学上的“前沿集群”，分数相近；模型间仅在拒绝行为上存在可测量差异。

**⚠️ 局限性**

局限性包括筛选过程仅作为预过滤，可能遗漏部分精神健康内容；翻译与非英语对话依赖机器翻译；评判员仅来自三家供应商，缺乏更广泛的评审；以及评估仅覆盖已发布的20款模型而非完整技术空间。

---

## 76. Longitudinal Robot Learning from Demonstration with Care Providers in a Home Environment

**arXiv ID:** 2608.25196 | [PDF](https://arxiv.org/pdf/2608.25196v1)

**作者:** Nina Moorman `[一作]` (Georgia Institute of Technology), Matthew Gombolay `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在真实家庭环境中，招募无机器人背景的照护人员，进行三次访问，利用预训练和自适应反馈支持其通过演示学习教机器人多种辅助任务，并将所有演示数据公开成新数据集。

**💡 创新点**

创新点包括：①将LfD研究从实验室扩展到真实家庭场景；②在多次访问中系统评估预训练与自适应反馈的组合效果；③针对照护人员的交互式界面与反馈机制（基础模型、机器人回放、增强现实）；④首次公开包含自然语言、点云、轨迹、音频等多模态的多访数据集。

**🔧 技术方法**

核心技术：Intel RealSense + YOLOE+ICP实现对象检测与姿态估计；Cartesian ProMPs用于技能学习；交互式演示界面支持任务拆分、演示记录；自适应反馈包含基础模型推理、机器人回放、增强现实可视化；数据处理利用多模态同步记录。

**📊 数据集**

使用自建的照护人员多访演示数据集（包含11个任务域、6项任务、自然语言描述、点云、轨迹、音频等），并对比已有数据集（MIME、RoboPro）以示其补充价值。

**📈 对比分析**

通过在两组条件（PT+AF vs AF）下进行统计比较：使用多元线性回归混合效应模型，评估任务完成率、预测与实际表现一致性、用户体验（可用性、接受度、信任度、工作负荷）。预计PT+AF组在任务完成率与用户体验上表现更佳。

**⚠️ 局限性**

局限性：样本量受限于照护人员招募；仅使用JACO 2机械臂，结果对其他机器人平台的泛化未知；缺乏现场专家即时反馈，可能导致部分演示误差；数据集覆盖的任务域与真实家庭多样性有限。

---

## 77. Learning Mixtures of Plackett-Luce Models for Multi-Objective Alignment

**arXiv ID:** 2608.25200 | [PDF](https://arxiv.org/pdf/2608.25200v1)

**作者:** Dongyue Li `[一作]` (Northeastern University), Hongyang R. Zhang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通过生成额外响应并利用梯度估计来学习多模 Plackett‑Luce 混合模型的高效算法。

**💡 创新点**

突破了混合模型在短排名长度下不可辨识的理论限制，并通过增广排名长度和梯度近似实现了可辨识性与计算效率双赢。

**🔧 技术方法**

核心技术包括排名增广（利用基础语言模型生成新候选项）、一阶梯度估计（在嵌入空间近似模型输出）以及基于期望‑最大化的混合模型学习。

**📊 数据集**

在多评判准则的 UltraFeedback 数据集（4 候选、4 类）和多群体 Persona 数据集（2 候选、12 类）上进行评估，并对比了基准单模型与 BT 混合模型。

**📈 对比分析**

在聚类准确率上提升了 43.7%，在排名准确率上提升了 15.2%，同时相较全量计算的 PL 混合模型，GPU 运行时和显存使用可降低约 2–3 倍。

**⚠️ 局限性**

局限在于生成的响应假设始终排在已注释响应之后，且混合成分数目固定，未考虑动态自适应以及多轮对话情境。

---

## 78. The Imperfective Paradox Is Not Necessarily in Large Language Models: A Benchmark Failure Before a Model Failure

**arXiv ID:** 2608.25005 | [PDF](https://arxiv.org/pdf/2608.25005v1)

**作者:** Kaiqiao Han `[一作]` (University of California, Los Angeles), Yizhou Sun `[通讯]` (University of California, Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新审视不完全时态悖论（Imperfective Paradox）基准，识别并纠正其概念与评估误差，构造词汇匹配最小对照集，并将事件语义NLI重新表述为多步推理过程。

**💡 创新点**

创新点在于：①发现并量化三种基准误差（Aspectual Reduction、Semantic Mis‑specification、Relation Misassignment）；②提出Sufficiency Bias与Decision Shift的新解释；③设计中间输出与oracle‑guided评估来定位错误来源；④构建词汇匹配最小对照集以消除词汇与上下文混淆。

**🔧 技术方法**

使用多步推理框架、结构化中间输出提示、对比不同提示（零样本、DAP、CoT、Counterfactual）以及oracle‑guided实验来评估模型推理能力。

**📊 数据集**

评估数据集包括原始Imperfective Paradox基准、Validity‑Screened Subset (VSS) 和自行构造的Lexically Matched Minimal Pairs；对大型公开 LLM（Qwen‑7B/72B、GLM‑4‑9B、Llama‑3.1‑8B）和闭源 LLM（GPT‑5.4）进行测试。

**📈 对比分析**

通过多步评估显示：模型往往未肯定事件完成但仍预测相应简单过去句为蕴含（Sufficiency Bias），提示干预常导致Decision Shift而非真正提升推理；在最小对照集上，模型性能接近人类，但在Aspectual Classification阶段仍表现不稳定；在Oracle‑guided实验中，语义表示与非语义表示差异显著。

**⚠️ 局限性**

局限性包括：仅在单一注释协议下完成人工标注；控制对照集可能不涵盖自然语言中所有解释；理论简化忽略词汇语义、论元结构、语境与语用差异；中间输出评估受提示措辞和答案格式影响。

---

## 79. Combining Self-Embedding Audio Watermarking with Ultra-Low-Bitrate Neural Codecs

**arXiv ID:** 2608.25289 | [PDF](https://arxiv.org/pdf/2608.25289v1)

**作者:** Yigitcan Özer `[一作]` (National Institute of Informatics), Junichi Yamagishi `[通讯]` (National Institute of Informatics)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种训练无关的自嵌入式音频水印框架，将极低比特率神经音频编码器产生的压缩表示嵌入原音频本身，实现局部语音篡改的检测、定位和恢复。

**💡 创新点**

创新点包括：①利用神经音频编码器（SNAC、SemantiCodec、TAAE）生成极低比特率压缩表示并嵌入；②采用多位LSB重复嵌入实现对局部篡改的鲁棒检测；③通过DTW对齐与自重建的相似度进行检测和定位，并在理想条件下完整恢复被篡改段落。

**🔧 技术方法**

技术手段：LSB嵌入、语音神经编码器、帧级余弦距离、DTW时间对齐、Majority voting、PESQ质量评估、EER/AUC性能评估。

**📊 数据集**

使用的公开数据集为AV-Deepfake1M验证集（1480句）以及VoxCeleb2作为插入样本来源。

**📈 对比分析**

与传统哈希基准对比：哈希方案在理想条件下实现EER≈0%、AUC>0.999；自嵌入方案在最佳配置下EER≈14‑15%、AUC≈0.84；TAAE性能相对更差（EER≈32%、AUC≈0.78）。检测效果按篡改类型排序：直接/TTS替换易检测，删除最难；恢复质量（PESQ）约为1.6‑1.7，保持可懂度与自然度。

**⚠️ 局限性**

局限性：仅在无噪声、无压缩的理想通道下验证；删除/插入导致的时间偏移对DTW匹配影响大；高比特率编码器的恢复质量低；缺乏对实际信道失真（噪声、压缩等）的鲁棒性；未对嵌入冗余与压缩进一步优化。

---

## 80. Multi-View Trust Evaluation for Collaborator Selection via Evidential Deep Learning

**arXiv ID:** 2608.25235 | [PDF](https://arxiv.org/pdf/2608.25235v1)

**作者:** Botao Zhu `[一作]` (Western University), Xianbin Wang `[通讯]` (Western University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在分布式系统中提出多视角证据学习（MVE）框架，用于从多源异构的历史协作记录中评估协作者的可信度并进行任务分配。

**💡 创新点**

创新点包括：①将不同任务主机视为独立视角，保留视角间的差异性；②利用Mamba状态空间模型捕捉每视角下的长期动态信号；③采用证据深度学习给出可信度估计及不确定度；④设计基于不确定度的动态冲突融合规则，避免传统Dempster规则在冲突高时失效。

**🔧 技术方法**

技术：多视角学习、Mamba序列模型、证据深度学习（Dirichlet分布与主观逻辑）、冲突冲突冲突融合规则、统一损失（分类+一致性）以及基于NS‑3的仿真环境。

**📊 数据集**

数据集：基于NS‑3与Python仿真构造的200台设备（DELL 5200/5820/7060）在5种操作场景（elite、stable、strategic、selfish、failed）下生成10,000个协作任务，产生的行为特征映射到5级可信度标签。

**📈 对比分析**

与QS‑Trust、TMC（Dempster融合）和DEF（证据折扣+Dempster融合）对比；评价指标为Macro‑F1、MAE、准确率/不确定度关系及冲突鲁棒性。MVE在Macro‑F1上达0.8247（比DEF提升4.86%），MAE为0.0734（比DEF提升14.6%），在高冲突场景下误差仅提升17.1%，任务成功率也明显高于对照组。

**⚠️ 局限性**

局限性：实验仅在仿真数据上验证，真实网络环境中设备行为、通信噪声及攻击模式更复杂；MVE对超大序列的实时推理需要更多计算资源；当前模型只考虑任务成功率和资源匹配，未覆盖更细粒度的安全/隐私约束。

---

## 81. Towards Reliable, Generalizable, and Specific In-Context Knowledge Editing via Multi-Objective Reinforcement Learning

**arXiv ID:** 2608.25100 | [PDF](https://arxiv.org/pdf/2608.25100v1)

**作者:** Xuzhong Wang `[一作]` (William and Mary), Haipeng Chen `[通讯]` (William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MO-IKE，一种多目标强化学习框架，用于在不训练模型参数的情况下，通过动态构造上下文来完成大型语言模型的知识编辑。

**💡 创新点**

创新点在于：① 将演示选择建模为约束马尔可夫决策过程，显式平衡可靠性、通用性和特异性；② 扩展动作空间包含 COPY、UPDATE、RETAIN 与 STOP，实现全局顺序优化；③ 采用多目标奖励塑形并使用固定惩罚系数，避免“reward hacking”，提升特异性。

**🔧 技术方法**

技术包括：BERT‑基检索器、GRPO（Group Relative Policy Optimization）、Lagrangian 约束、固定惩罚系数的多目标奖励、soft/stop 机制。

**📊 数据集**

数据集：CounterFact、ZsRE、Wiki‑Counterfact、UniEdit，使用 Llama‑3.2‑3B、Mistral‑7B‑v0.3 等指令调优 LLM。

**📈 对比分析**

与 FactPrompt、EditCoT、IKE、DR‑IKE 等基线相比，MO‑IKE 在可靠性（Edit Success）提升约7%，特异性（Retention Rate）提升约23%，总体评分（S）提升约20%（Llama‑3.2‑3B 上从 54.9% 提升到 73.5%）。

**⚠️ 局限性**

局限性：① 数据集规模有限，仅用 CounterFact 前 2000 条记录训练；② 采用固定 λ 的奖励惩罚，未调优其对学习的影响；③ 虽然特异性提升显著，但仍低于 70%，受限于无梯度学习方法；④ 仅在知识编辑任务验证，未检验对其它 ICL 任务的通用性。

---

## 82. Can We Read the Mind of an Audio LLM? A Verbalizable, Multilingual Middle-Layer Workspace

**arXiv ID:** 2608.24958 | [PDF](https://arxiv.org/pdf/2608.24958v1)

**作者:** Jiajun Fan `[一作]` (Amazon AGI Foundations), Ivan Bulyko `[通讯]` (Amazon AGI Foundations)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用 logit lens 在 Qwen3‑Omni 30B 的音频输入位置读取中间层激活，证明音频 LLM 在中间层形成可读的全局工作空间（workspace），并在此空间中可提前读取答案相关概念、跨语言、多模态信息。

**💡 创新点**

首次将工作空间概念从文本/图像/代码模型迁移到音频 LLM，并通过音频波形交换控制、两种“mind”比较、激活补丁和层删测验等方法证明该空间确实由音频驱动、可在中间层形成并对最终输出产生因果影响；同时揭示音频 LLM 的思考空间具有多语言性、情感/说话人属性和自发联想特征。

**🔧 技术方法**

使用 logit lens（将残差状态投影到词表空间）、激活补丁、层删操作、配对 McNemar 检验、两种输入（音频 vs 模型自生成文本）比较，以及音频波形交换（真实、匹配、静音）控制。

**📊 数据集**

核心数据集为 140‑clip 细化 MMAU (包含 83:57 声音/语音比例)；扩展使用 338 音乐、948 脑图网格、863,770 workspace 单元、1000‑clip MMAU‑mini、55 情感片段以及 500 题 TriviaQA 进行统计。

**📈 对比分析**

比较方法：以平衡准确率 (balanced accuracy) 与配对 McNemar 统计衡量音频驱动与文本先验的差异；实验表明在工作空间层级，真实音频读取准确率约 40%，相较于静音 21.8% 或匹配 32.2% 显著提升；在层级逐层检验中，差异在约 12% 深度后显著出现并持续，表明工作空间中可读信息早期形成且不消退。

**⚠️ 局限性**

局限性：仅使用 logit lens，未采用更精确的 Jacobian lens；实验仅覆盖 Qwen3‑Omni 及其 7B 变体；读取位置稀疏，仅捕获音频位置中的概念；未对多模型泛化、对齐数据集的规模化验证，也未评估对安全相关决策（工具调用、拒绝、虚构）的可读性。

---

## 83. Paging with Per-Replacement Maximum Delay

**arXiv ID:** 2608.25290 | [PDF](https://arxiv.org/pdf/2608.25290v1)

**作者:** Tianhang Lu `[一作]` (Southern University of Science and Technology), Shengcai Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了一种新的“每次替换最大延迟”（per‑replacement maximum‑delay）分页模型，允许页面请求在被加载前保持等待状态，并提出了对应的离线近似与在线竞争算法。

**💡 创新点**

创新点包括：
• 证明在该模型下，经典的竞争阶层仍然成立；给出了确定性 (5k+3)-竞争阈值 LRU 算法和随机化 5H_k-竞争算法；
• 引入“时延聚合”技术，将原始带时间的请求映射为无时延的虚拟请求序列，从而可以利用经典分页算法的保证；
• 设计了最优的单空洞（one‑hole）动态规划与多空洞（fixed‑holes）配置动态规划；
• 扩展到权重检索成本（weighted paging）并给出与权重扩展比例 ρ 相关的竞争与近似界限；
• 给出严格的下界（Ω(√ρ) 对于单槽缓存），揭示了权重扩展导致的新难度。

**🔧 技术方法**

使用的主要技术包括：
• 持续性（causal）懒惰投影，将任意调度映射为仅加载待处理页面的非主动调度；
• 持续负载识别（holding‑cost identity）将等待成本表述为对所有待处理页面的积分；
• 事件层动态规划与配置动态规划；
• 时延窗口聚合（window aggregation）与投影潜能（discrepancy potential）实现非主动物理替换；
• 随机化的标记（Marker）与分区（partitioning）算法的无时延映射。

**📊 数据集**

该工作完全基于理论分析，没有使用实验数据集；所有结果均在数学证明框架下给出。

**📈 对比分析**

性能评价：
• 对于单位成本模型，确定性阈值 LRU 的竞争比为 5k+3，随机化 AW‑Partition（θ=2/3）达到 5H_k；
• 对于权重模型，给出了 (3ρ+2)-近似离线算法、O(ρk) 的确定性竞争与 O(ρ log k) 的随机化竞争；
• 证明了这些上界与经典的 Ω(k) / Ω(H_k) 下界相匹配，且下界在权重扩展上达到 √ρ 的严格限制。

**⚠️ 局限性**

局限性：
• 对于一般空洞数 r=m−k，离线最优的多空洞动态规划是指数级的，未给出多项式或 FPT 解；
• 上界的常数（5、3ρ+2）可能不是最优，尚未证明最小值；
• 对于权重扩展，随机化竞争仅是 O(ρ log k)，与下界 Ω(√ρ) 存在宽裕空间；
• 未探讨更一般的度量空间（k‑server）下的最大延迟模型；
• 该模型仍假设所有请求已知时间戳，实际系统中时延估计不确定性未被充分考虑。

---

## 84. BGPay: An Incentive-Compatible Mechanism for BGP Hijack Filtering

**arXiv ID:** 2608.25165 | [PDF](https://arxiv.org/pdf/2608.25165v1)

**作者:** Tomasz Sadowy `[一作]` (Princeton University), Maria Apostolaki `[通讯]` (Princeton University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 BGPay 机制，利用区块链托管与路由监视器的证据，激励 AS 对前缀拥有者的 BGP hijack 进行过滤，并在过滤完成后自动支付赏金；

**💡 创新点**

将 BGP hijack 过滤从公共善举转变为基于赏金的市场交易；通过“缺席证明”与公开路由收集器作为根信任，利用承诺-公开（commit‑reveal）托管协议实现无信任的支付；

**🔧 技术方法**

区块链智能合约、Merkle 证明、commit‑reveal 机制、BGP 路由监视器（RouteViews、RIPE RIS）数据、RPKI/ROV 验证与 AS 关系图谱；

**📊 数据集**

1,018 起真实 hijack 事件（Cloudflare Radar）、RIPE RIS 与 RouteViews 路由收集器数据、CAIDA AS 关系数据、RoVista ROV 列表；

**📈 对比分析**

通过 BGP 模拟与真实事件评估：欺诈检测率>90%；代理集大小与最大损害潜力（MDP）的 Pearson 相关系数长前缀为0.893，短前缀为0.565；奖励分配覆盖 90% 仅由 4.5% AS 获得；总体性能显示可行性；

**⚠️ 局限性**

依赖路由监视器覆盖率与可信度；可能出现合谋与伪造，尤其在同长度前缀上奖励分配不够精准；未考虑动态博弈与声誉机制，且对监视器可用性高度敏感。

---

## 85. FAMPWQ: Fisher Information-based Adaptive Mixed Precision Weight Quantization for Effective LLM Inference

**arXiv ID:** 2608.24945 | [PDF](https://arxiv.org/pdf/2608.24945v1)

**作者:** Gongwei Lee `[一作]` (Soochow University), Ji Wu `[通讯]` (Tsinghua University)

**通讯引用:** 6459 | [OpenAlex ID](https://openalex.org/A5029547618)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Fisher 信息的自适应混合精度权重量化方法，针对大语言模型的层级量化敏感性进行精确测量并分配位宽。

**💡 创新点**

创新点在于：① 用量化误差的扰动直接计算 Fisher 信息差异，得到更贴近量化特性的层敏感性度量；② 结合 PPO 强化学习在全局存储预算下高效搜索最佳位宽分配策略。

**🔧 技术方法**

核心技术包括：Perturbation-based Fisher Information Metric、Proximal Policy Optimization（PPO）强化学习、量化扰动 δ₃ 以及基于 FIM 的梯度估计。

**📊 数据集**

使用的模型数据集：LLaMA-7B/13B、LLaMA2-7B-chat/13B、Qwen2.5-7B/14B、Mistral-7B-v0.1；基准评测数据集包括 WikiText-2、Penn Treebank、C4、lm-evaluation-harness、Vicuna。

**📈 对比分析**

与 7 种基线（GPTQ、GPTQv2、AMQ、OmniQuant、OWQ、AWQ、RTN）对比，平均 PPL 提升至 3.39 以下、零样本推理准确率提升至 6.87% 以上、LLM-as-a-judge 胜率可达 76%，在 3‑bit 量化前沿表现尤为突出。

**⚠️ 局限性**

局限性包括：异构位宽导致推理吞吐下降；仅实现权重量化（WxA16），未覆盖激活量化；实验仅验证密集 Transformer，混合专家模型尚未评估；Fisher 敏感性估算为一次性高成本预处理。

---

## 86. Static Detection of Post-Quantum Cryptographic Algorithms in Stripped Binaries for Digital Forensic Examination and Migration Assurance

**arXiv ID:** 2608.25122 | [PDF](https://arxiv.org/pdf/2608.25122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 87. From Blind Edits to Verified Repair: Building Trustworthy User-Side LLM Agents for Web Accessibility

**arXiv ID:** 2608.24913 | [PDF](https://arxiv.org/pdf/2608.24913v1)

**作者:** Lily Bundgaard Wanscher `[一作]` (University of Southern Denmark), Mina Alipour `[通讯]` (University of Southern Denmark)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5088430940)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款隐私保护的浏览器扩展，利用本地小型LLM生成增量CSS来提升网页的可访问性；并构建了带有验证循环的修复工具；同时设计了双条件实验协议评估干预的利弊。

**💡 创新点**

①引入可验证的修复循环，确保生成的CSS不会导致可访问性回退；②通过双条件实验同时量化提升与损伤；③在实际浏览器中实现完整的从样式提取、上下文压缩到本地LLM推理与CSS注入的闭环。

**🔧 技术方法**

浏览器扩展（Chrome Manifest V3）、本地LLM推理（Ollama），上下文压缩与排序算法，CSS解析与验证工具，axe-core等自动化检查器，配合视觉和文本提示的多模态输入。

**📊 数据集**

实验数据集：20个真实网站（10个高违规商用站点+10个可访问性优秀站点），18项WCAG/COGA可通过CSS控制的可访问性指标；验证基准：42个本地页面，包含57条已知违规，12个干净页面。

**📈 对比分析**

采用前后截图对比的人工评分与自动化规则检查，记录违规差异。结果显示：在5个产生可注入CSS的模型中，违规改善与回退几乎持平（控制站点改善14例、回退14例；样本站点改善10例、回退6例），无统计学显著性。验证循环在基准上表现完美，检测所有seeded违规、无误报、拒绝所有有害候选。

**⚠️ 局限性**

局限性包括样本规模小（10站/模型、单评审）、仅关注CSS层面缺乏结构层面修复、未进行真实用户测试、使用量化的本地模型且受10k token上下文限制，可能导致上下文截断和不可预知的副作用。

---

## 88. Solving Robust POMDPs with Omega-regular Objectives via Partially Observable Stochastic Games

**arXiv ID:** 2608.24986 | [PDF](https://arxiv.org/pdf/2608.24986v1)

**作者:** Durgam Latha `[一作]` (Indian Institute of Technology Bombay), Shankaranarayanan Krishna `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究在不确定转移概率下的鲁棒 POMDP（RPOMDP）在 ω‑正则（如可达、安全、Büchi 等）目标下的求解问题，并给出了求解该问题的理论复杂度分析。

**💡 创新点**

首次证明 RPOMDP 与交替控制 POSG 在 ω‑正则目标下可以双向多项式时间归约，进而建立两者的语义等价性，并由此得到一系列新的复杂度上界与下界。

**🔧 技术方法**

使用多项式时间的两种归约技术：①直接构造将 RPOMDP 转化为 ac‑POSG；②通过“预转换”消除信息差异，再折叠将 ac‑POSG 转化为 RPOMDP；这些归约保留记忆性与策略类型。

**📊 数据集**

论文主要为理论分析，没有采用实验数据集；所有结果均基于数学证明与复杂度理论。

**📈 对比分析**

通过等价性，将已知的 POSG 在 ω‑正则目标下的复杂度结果（如 EXPTIME‑完整、NP∩coNP、undecidable 等）迁移至 RPOMDP，提供了完整的复杂度图谱；相较于以往仅对奖励目标或特定不确定集的研究，显著拓宽了理论范围。

**⚠️ 局限性**

局限在于只考虑 (s,a)-rectangular 多面体不确定集且策略为动作不可见；对更一般的不确定集、非多面体结构或动作可见的策略尚未讨论，且实际算法实现与实验验证仍待进一步研究。

---

## 89. Model-Based Agentic Software Engineering

**arXiv ID:** 2608.25174 | [PDF](https://arxiv.org/pdf/2608.25174v1)

**作者:** James C. Davis `[一作]` (Purdue University), Parth V. Patil `[通讯]` (Amazon Robotics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证了MAGE（Model‑Based Agentic Software Engineering）框架，阐述了如何通过建模与对齐来外化知识、赋权约束，从而提升编码代理的可靠性与可持续性。

**💡 创新点**

创新点在于将传统软件建模与AI代理对齐机制整合为一个治理环境，既系统化了“意图外化”和“约束赋权”的双重原则，又通过案例演化与跨行业对比验证其可推广性。

**🔧 技术方法**

采用了大型语言模型（Claude）代理、模型化工具（MBSE/MDE）、对齐机制（约束、传感器、验证器、门控）以及持续集成与测试管道，构成完整的“建模–对齐–治理”循环。

**📊 数据集**

主要数据来源为DocAble项目的Git仓库与运行日志，以及六个独立行业（Cloudflare、Spotify、Shopify、Docker、Siemens、Zenseact）的公开技术说明与实践报告。

**📈 对比分析**

通过在DocAble内部对比“使用模型指令”与“无模型指令”的两组代理实验，发现后者在令牌使用、交互轮数和耗时上均显著更高，表明模型化能显著降低重构成本与任务完成时间；在跨行业案例中则通过定性对比验证了MAGE结构的普遍出现与可替代性。

**⚠️ 局限性**

局限在于缺乏大规模因果验证、对性能提升的量化范围有限、案例主要基于单一项目与行业自述，难以排除组织文化与工具差异的干扰，且模型与对齐机制的维护成本与可扩展性仍需进一步实证。

---

## 90. PhysElite: How Far Are LLMs from Solving Olympiad-Level Physics Problems?

**arXiv ID:** 2608.25097 | [PDF](https://arxiv.org/pdf/2608.25097v1)

**作者:** Ruoran Xu `[一作]` (Xi'an Jiaotong-Liverpool University), Qiufeng Wang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个大规模双语多模态奥林匹克级物理推理基准（PhysElite），包含11,586道带图示、分步推导与答案的开放式问题；

**💡 创新点**

首次将奥林匹克级难度、规模、双语、多模态和步骤级过程注解统一集成到一个基准中，并提供基于步骤的评估方法；

**🔧 技术方法**

使用OCR（Tesseract）+人工校正进行数据转换，利用LLM辅助人工注解，采用多模型评估与链式思维（Extended Thinking）以及LLM评审（GPT‑5.2、Claude‑Opus‑4.6、Gemini‑3‑Pro）进行答案和过程评分；

**📊 数据集**

核心数据集为PhysElite（来自15位一等奖获奖学生的日常训练材料），包含图示、双语文本、分步推导与答案；并与PhysicsArena、HiPhO、PHYBench等现有基准做对比；

**📈 对比分析**

通过对18个开源与闭源MLLM进行评测，最佳模型Grok‑4.2答案准确率仅33.7%，远低于人类基准（约65%）；过程得分普遍高于答案得分，说明模型在中间步骤仍有一定合理性；扩展思维对部分模型有提升但不均衡；

**⚠️ 局限性**

存在部分分值归属模糊、对商业模型训练数据可能存在重叠、仅覆盖双语（中英）且图示为二维平面，未涵盖更复杂的物理场景，需进一步提升数据隐私与多样性。

---

## 91. Neural-Bayesian Structure Learning for Discrete Choice Modeling

**arXiv ID:** 2608.25258 | [PDF](https://arxiv.org/pdf/2608.25258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. FuzzingBrain-Bench V1: Evaluating Open-Ended Bug Discovery by LLMs

**arXiv ID:** 2608.25158 | [PDF](https://arxiv.org/pdf/2608.25158v1)

**作者:** Ze Sheng `[一作]` (Texas A&M University), Jeff Huang `[通讯]` (Texas A&M University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FuzzingBrain-Bench V1，一个基于Docker的基准，用来评估大语言模型在开源软件中发现不同类型软件缺陷（崩溃）的能力，模型通过生成可触发多种崩溃签名的输入进行评测。

**💡 创新点**

创新点在于：①不再只验证模型是否能重现单一已知漏洞，而是统计模型能够触发的不同崩溃签名；②使用多种 sanitizer（ASan、UBSan、LeakSanitizer、Jazzer等）并涵盖内存安全与DoS/其他错误；③通过困难系数和分数上限3的截断来平衡挑战难度与模型多样化发现。

**🔧 技术方法**

技术包括：①将项目源码、Harness及Sanitizer编译后打包为Docker镜像；②模型通过API调用读写文件、生成PoC，执行三次验证相同crash签名；③签名提取规则（故障类+最多3个相关函数），以及重复签名去重；④基于分数上限和难度系数的评分公式。

**📊 数据集**

数据集：77个挑战，来自43个开源项目（36 C、32 C++、9 Java），共涵盖45个内存安全、32个DoS/其他错误，使用AddressSanitizer、UndefinedBehaviorSanitizer、LeakSanitizer、Jazzer、libFuzzer等工具。

**📈 对比分析**

比较方法：在同一预算（100回合、1800秒）下，让Claude Haiku 4.5、Sonnet 4.6、Opus 4.8三模型执行完整挑战集；评估指标为触发的挑战数、每挑战的distinct crash签名数、累计分数；结果显示Opus 4.8最高得分196/579，触发60/77挑战；Sonnet 156/579，Haiku 58/579。

**⚠️ 局限性**

局限性：①挑战规模小（77），难度系数基于三模型结果产生循环依赖；②签名提取仅能识别有限类型的崩溃，未覆盖线程安全、内存安全之外的漏洞；③模型被限制不使用网络、无法获取补丁或原始Bug描述，可能影响发现能力；④未验证在更大、多样化数据集上的泛化效果。

---

## 93. A Lightweight Multimodal Vision-Language Framework for Early-Stage Anatomical Green Fruit Classification in Commercial Orchards

**arXiv ID:** 2608.24935 | [PDF](https://arxiv.org/pdf/2608.24935v1)

**作者:** Ranjan Sapkota `[一作]` (Cornell University), Manoj Karkee `[通讯]` (Cornell University)

**通讯引用:** 7820 | [OpenAlex ID](https://openalex.org/A5013737840)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究提出一种轻量化多模态视觉语言框架TinyCLIP，针对商业果园早期苹果果粒的三大解剖结构（萼片、果粒、柄）进行多标签分类与定位。

**💡 创新点**

创新点在于将TinyCLIP微调为滑窗式推理，并结合领域特定文本提示与热图聚合，实现可解释且可在嵌入式硬件上部署的精细解剖识别。

**🔧 技术方法**

使用技术包括TinyCLIP视觉语言模型、文本提示、224×224滑窗分块、多标签Sigmoid头、ONNX导出、TensorRT FP16/INT8量化、热图生成与聚合。

**📊 数据集**

数据集为600张高分辨率RGB果园图像，按训练/验证/测试划分后拆分为224×224补丁，并手工标注萼片、果粒、柄的COCO框。

**📈 对比分析**

与传统视觉检测/分类基线相比，宏F1提升至0.93；FP16推理精度0.91、INT8保持88.8%；推理速度达96帧/秒，满足田间机器人实时需求。

**⚠️ 局限性**

局限包括柄召回率相对较低、热图定位精度不足以实现精确实例分割，以及整体推理时延仍受补丁数量影响。

---

## 94. Behind the [MASK]: Disentangling Representation and Faithfulness in DAPF-Based Dementia Detection

**arXiv ID:** 2608.25028 | [PDF](https://arxiv.org/pdf/2608.25028v1)

**作者:** Pardis Ranjbar-Noiey `[一作]` (University of Illinois Chicago), Natalie Parde `[通讯]` (University of Illinois Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究了基于提示的域自适应微调（DAPF）在口语阿尔茨海默病检测中的可解释性，结合表示探测、词级归因和扰动验证，比较其与 BERT-CLS 及 BERT-CLS+Prompt 基线的行为。

**💡 创新点**

揭示了诊断信息在隐藏层和提示词位置的集中与词级归因可信度之间的脱节，提出即使模型在表示层表现良好，提示式预测仍可能导致不可靠的词级解释。

**🔧 技术方法**

使用 BERT‑Base‑Uncased 作为主干；DAPF 采用手工提示、域提示和诊断词提示的掩码语言模型；BERT-CLS 与 BERT-CLS+Prompt 作为对照；通过逻辑回归表示探测、Integrated Gradients 与 Partition SHAP 进行词级归因；再用删除/插入扰动测试评估归因可信度。

**📊 数据集**

Carolina Conversations Collection（CCC）作为源域训练集，ADReSS 作为目标域训练/测试集，二者共同构成跨域转移实验。

**📈 对比分析**

在跨域测试中，DAPF 的准确率 0.83、宏 F1 0.83 轻微优于 BERT‑CLS；AUROC 与 ECE 与基线相近；对齐错误分析显示 DAPF 在 20 个独立正确预测中占优势；表示探测表明 DAPF 的 [MASK] 表示层信息最丰富；但词级归因在删除/插入测试中表现弱于基线，说明归因不可信。

**⚠️ 局限性**

实验仅在单一低资源的 CCC→ADReSS 迁移场景；大规模掩码可能导致分布漂移，影响结果；使用的提示模板、词汇表与语义化提示方法可能影响结论；未验证在更大、不同的语料上的泛化；词级解释的可靠性仍需进一步研究。

---

## 95. RefLAM: A Reference-Grounded Line Annotation Pipeline for Historical Arabic Manuscripts

**arXiv ID:** 2608.25140 | [PDF](https://arxiv.org/pdf/2608.25140v1)

**作者:** Mohamed Guechaoui `[一作]` (Higher School of Computer Science), Sahraoui Dhelim `[通讯]` (Higher School of Computer Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RefLAM 自动化流程，将手稿页图与已有清晰抄本对齐，生成验证过的行级文字标注；并发布 AraMS‑28k 数据集。

**💡 创新点**

引入置信度 100 规则（confidence‑100 rule），证明相似度最高时字符完全一致；结合多模态 LLM 与模糊对齐，实现 75 倍的标注效率并保持人类监督。

**🔧 技术方法**

使用 Kraken 行分割模型、Google Gemini 结构化 OCR、对齐前置的正则化与 LCS 相似度、贪心行对齐算法以及可视化审阅工具。

**📊 数据集**

基于 14 本古阿拉伯手稿（Naskh、Ruq‘ah、Maghrebi 以及 lithograph），并配合现有清洁数字抄本生成 AraMS‑28k（3043 页、27,971 主文本行、629 边注行）。

**📈 对比分析**

在 AraMS‑28k 训练集上微调已预训练的 Kraken 与 HATFormer，得到整体 CER 23.31% 与 26.74%，验证该数据集可用于下游 HTR 训练；与手工注释基线相比，速度提升 75 倍。

**⚠️ 局限性**

必须先有清晰抄本；边注行中约 70% 无法精准锚定；对齐采用贪心策略，非全局最优；对 Gemini LLM 的依赖需用开源替代。

---

## 96. LLM-Driven, Datasheet-Aware Automated Hardware Compatibility Verification for Early-Stage, Pre-Schematic Embedded System Design

**arXiv ID:** 2608.25217 | [PDF](https://arxiv.org/pdf/2608.25217v1)

**作者:** Haotian Qiao `[一作]` (University of Michigan), Robert P. Dick `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了DEVICES框架，利用大型语言模型在预原理图阶段通过硬件数据手册和高层互连描述进行接口级硬件兼容性验证。

**💡 创新点**

创新点在于：①将验证拆解为设计图、属性检索、脚本生成等可追溯的模块；②定义结构化的验证准则（criteria）并在检索时只拉取相关属性；③采用任务感知的上下文构建显著压缩LLM输入；④通过知识图谱存储属性并支持缓存，实现高效、模型无关的属性引用。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT）用于设计图生成和脚本编写；层次化硬件知识图谱用于属性提取与检索；结构化验证准则驱动的提示工程；Python脚本的确定性执行进行数值比较；缓存机制和模型感知的属性筛选。

**📊 数据集**

实验使用7个嵌入式系统设计，共34份硬件数据手册（总计1210页），涵盖环境感知、健康监测和定位追踪三大应用。

**📈 对比分析**

与两种一键提示基线（无准则、加准则）对比，DEVICES在兼容性检查准确率上达到97.5%，远超无准则的14.9%和加准则的62.7%；且在输入上下文大小上相较一键提示平均缩小约8.6倍，证明了可扩展性和高效性。

**⚠️ 局限性**

局限性包括：目前仅支持文本式数据手册，无法处理图形或曲线（如开关稳压器效率曲线）；需要手动提供高层互连描述；对某些功率模块（如转换器）支持有限；依赖LLM生成脚本，若LLM出现幻觉仍需人工验证。

---

## 97. Analyzing and Reducing Search Quality Differences in Vector Similarity Search

**arXiv ID:** 2608.25185 | [PDF](https://arxiv.org/pdf/2608.25185v1)

**作者:** Sara Mahdizadeh Shahri `[一作]` (Carnegie Mellon University), Akshitha Sriraman `[通讯]` (Carnegie Mellon University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种轻量级系统，能够动态识别并缩小向量相似检索中的召回差异，同时保持高吞吐量。

**💡 创新点**

创新点在于：①将查询分组到嵌入空间中的簇（s）进行召回监测；②基于每组的召回与吞吐经验表，在运行时按需细粒度调整搜索努力；③在不重建索引、仅调节现有搜索参数的前提下实现召回均衡。

**🔧 技术方法**

技术主要包括：k-means聚类划分查询空间；Prometheus+Prometheus pull 监控召回、吞吐和尾延迟；多线程精确检索采样估计召回；控制循环算法在每个监测周期更新经验表并决定搜索努力调整；gRPC+线程安全字典将新搜索力度注入pgvector。

**📊 数据集**

使用的数据集涵盖文本（Twitter词向量）、面部图像（InsightFace 512维）、图像分类（CLIP ViT-B/32 512维）和知识图谱实体（sentence‑embedding 384维），每个都按90/10划分为索引/查询。

**📈 对比分析**

与传统的全局搜索努力调参(pgvector)以及目标召回调参（R‑check）相比，R‑check 在相同吞吐量下平均召回提升约15‑20%，在相同平均召回下吞吐量提升约20‑30%；其召回分布更窄，落后目标召回阈值的查询比例下降约30%。

**⚠️ 局限性**

局限性包括：①仅通过调节搜索努力无法解决索引本身质量低导致的召回极低情况；②对高频查询的样本不足会导致召回估计不稳定；③需要额外的监控与采样开销，对资源占用有一定影响；④对极大簇数或极小簇数的选择需经验调优。

---

## 98. Measurement-Budget Allocation in Quantum Learning with Finite-Shot Generalization Guarantees

**arXiv ID:** 2608.24891 | [PDF](https://arxiv.org/pdf/2608.24891v1)

**作者:** Ferhat Ozgur Catak `[一作]` (University of Stavanger), Ferhat Ozgur Catak `[通讯]` (University of Stavanger)

**通讯引用:** 1926 | [OpenAlex ID](https://openalex.org/A5044259885)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在给定有限测量预算下如何分配训练样本数与每个样本的测量次数，以最小化量子分类器的泛化误差。

**💡 创新点**

提出了分布无关的泛化上界，并推导出闭式预算分配规则，使得在两种资源（样本数与测量次数）之间取得平衡，证明最优分配导致B^-1/4的收敛速率。

**🔧 技术方法**

运用Rademacher复杂度分析、Hoeffding集中不等式、Lipschitz损失、量子测量理论以及PennyLane模拟平台进行实验验证。

**📊 数据集**

在9个合成二分类基准上进行实验，使用2量子比特和4量子比特的可变结构量子电路。

**📈 对比分析**

与理论上界对比，实验中的单侧泛化误差始终低于理论上界；尽管上界保守，但未出现违例，表明理论有效。

**⚠️ 局限性**

上界保守、未考虑硬件噪声、仅适用于非自适应测量、对全类测量的最坏情况，且实验仅验证一致性而非紧密度。

---

## 99. LibriBrain100: One Hundred Hours of Broad and Deep MEG Data for Neural Speech Decoding at Scale

**arXiv ID:** 2608.25204 | [PDF](https://arxiv.org/pdf/2608.25204v1)

**作者:** Francesco Mantegna `[一作]` (University of Oxford), Oiwi Parker Jones `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并公开了 LibriBrain100 数据集，扩展了原有 LibriBrain 的规模与多样性，提供了单受试者约 80 小时的深度 MEG 数据和 32 名受试者各约 40 分钟的广度数据，并配套标准化拆分、Python 库和公开竞赛平台，用于非侵入式脑机接口的语音解码研究。

**💡 创新点**

创新点在于：① 在保持极大深度（单受试者 80 小时）基础上首次加入多受试者广度（32 名受试者），① 通过引入多种音素与语义覆盖的语料（TIMIT、MOCHA‑TIMIT、The Moth 播客）实现了音素与语义层面的可控实验；② 提供统一的评估基准与工具链，推动社区可复现与纵向对比。

**🔧 技术方法**

主要技术手段是基于预训练的 MEG‑XL 模型进行微调，以实现词分类任务；实验采用了不同的训练/验证/测试拆分，比较了单受试者、跨受试者以及少量训练数据下的性能；同时提供了 Python 数据加载器和 BIDS/HDF5 数据格式。

**📊 数据集**

使用的数据集为 LibriBrain100，包含四大语料：Sherlock Holmes 全书（约 68 小时）、TIMIT（5.3 小时）、MOCHA‑TIMIT（1.1 小时）和 The Moth 播客（6 小时）单受试者数据；以及 32 名受试者各约 40 分钟的 Sherlock 章节数据，总计 104.2 小时。

**📈 对比分析**

通过在词分类任务上对比：① 仅使用单受试者数据微调 MEG‑XL 与同时加入 32 名受试者数据微调的效果；② 在跨受试者设置下，使用单受试者大数据进行预训练再微调；③ 随着每位新受试者微调数据量从 100% 缩减到 25%（约 10 分钟），性能仍保持在 80‑90% 的准确率区间，说明跨受试者迁移有效。实验结果表明：单受试者深度数据显著提升自身性能；跨受试者迁移显著提升 32 名受试者的词分类准确率，且在极少训练数据下仍表现良好。

**⚠️ 局限性**

主要限制包括：① 数据仅来自被动听觉范式，未覆盖口语或内在说话等更具挑战性的任务；② 仅在健康志愿者上收集，尚未验证对临床抑制或瘫痪患者的适用性；③ 未包含完整脑转文字（brain‑to‑text）基准，仍需进一步研究。

---

## 100. Transforms for LLM Quantization: The Great Inversion and Format Co-Design

**arXiv ID:** 2608.25188 | [PDF](https://arxiv.org/pdf/2608.25188v1)

**作者:** Ehsan Jokar `[一作]` `[通讯]`, Ehsan Jokar

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对低位宽LLM量化中“先变换再四舍五入”阶段进行系统梳理，提出“Great Inversion”原则并对近200篇文献进行分类，归纳43种变换方法，给出理论证明和实证对比，提出首选指南并指出未解决的开放问题。

**💡 创新点**

创新点在于：① 将经典变换编码理论与现代共享尺度量化理论统一，揭示两者在能量集中 vs 能量展开上的根本逆转；② 通过对变换自由度（分配 vs 共享尺度）与数制（FP4、浮点等）的双轴分析，构建通用的“变换优选”框架；③ 提供了首个以同一实验协议评估所有变换方法的系统性对比，并公开了对应的基准数据与指南。

**🔧 技术方法**

技术包括：线性功能保持变换（旋转、缩放、置换、非正交仿射）；分组共享尺度量化（AbsMax、Hadamard等）；正交与非正交变换的理论分析；多尺度数制对比；GPTQ等基于均匀格子的四舍五入；大规模实证评测框架。文中还利用了主流LLM权重、激活及 KV-cache 等数据集，引用了公开的头对头对比实验。

**📊 数据集**

使用的主要数据集为公开的LLM权重与激活分布（如 GPT-3、LLaMA、Mixtral 等），以及各种标准量化基准的对比结果（包含 8‑bit、6‑bit、4‑bit 的多种配置）。未对新数据集进行训练，而是复现并归纳已有文献中的实验结果。

**📈 对比分析**

比较方法：作者统一制定“一套协议‑一组指标”对所有 43 种变换进行离线评测，重点比较在 4‑bit 权重/激活、6‑bit 量化等不同位宽下的误差与推理性能。结果显示：在大多数 4‑bit 场景下，Hadamard 旋转或数据感知 Whitening 提供了最优性能；在某些 8‑bit/6‑bit 方案中，固定或学习的正交旋转亦能显著提升精度；指南根据部署场景给出首选变换组合。整体性能提升幅度因模型与位宽不同而异，通常在 2–5% 的精度提升或相当于 2–4× 的算力/带宽节省。

**⚠️ 局限性**

局限性：① 仅聚焦线性功能保持变换，未涵盖非线性或动态变换；② 只在离线阶段评估，缺乏训练时反馈或量化感知训练的讨论；③ 对共享尺度量化的最优性理论仍不完整，尤其在浮点格子下的逆转现象尚未完全解释；④ 评测主要基于已公开实验，缺乏对全新模型的直接验证；⑤ 对极低位宽（≤2‑bit）或混合精度分配策略的分析有限。

---

## 101. Natural Language Input, Semantic Track Representation, and LLM Inference: Making the Maritime Information Exchange Model Tractable

**arXiv ID:** 2608.24892 | [PDF](https://arxiv.org/pdf/2608.24892v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 102. Domain-Adaptive ASR for Telephony AI Agents: Fine-tuning Canary Flash Models for Enterprise Contact Center Applications

**arXiv ID:** 2608.24916 | [PDF](https://arxiv.org/pdf/2608.24916v1)

**作者:** Chanameth Boonpramuk `[一作]` (Botnoi Group), Songpol Bunyang `[通讯]` (Botnoi Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

使用 NVIDIA NeMo 对 NVIDIA Canary 180M 与 1B Flash 进行快速微调，构建基于真实电话录音与促发式语音增强的泰语电话语料，并在语言适配、电话鲁棒性、业务术语以及实时延迟等四项实验中评估性能。

**💡 创新点**

通过少量泰语电话数据与多任务预训练模型实现高效适配；利用混合真实电话与促发式语音＋电话化增强构造高质量语料；在单张 A100 GPU 上完成大模型的低延迟微调。

**🔧 技术方法**

NVIDIA NeMo 框架、FastConformer 编码器+Transformer 解码器、SpecAugment 语音增强、混合精度 AdamW 优化器与逆平方根学习率调度。

**📊 数据集**

1593 小时泰语公共语料、77 小时真实电话+增强电话语料、104 小时名称与地址电话语料，外加 Common Voice 23 与 Fleurs 进行评估。

**📈 对比分析**

以字符错误率（CER）与实时因子（RTFx）为指标；在电话数据上 CER 从 23.31% 降至 9.04%，名称地址上从 16.98% 降至 3.78%；Canary 180M Flash 在通用电话评测上 CER 9.04%，RTFx>600；与 Whisper‑large‑v3、ElevenLabs Scribe v2、Google Chirp 3 等基线相比，取得显著的准确率和速度优势。

**⚠️ 局限性**

仅基于内部泰语数据评估，真实通话来自内部参与者，名称/地址微调导致通用电话略微回退，单一硬件与批量配置下的延迟评估可能不适用于所有部署场景。

---

## 103. When Does Context Routing Help? A Systematic Study of Multi-Modal Fusion in Time Series Forecasting

**arXiv ID:** 2608.25128 | [PDF](https://arxiv.org/pdf/2608.25128v1)

**作者:** Ruizhe Zhou `[一作]` (Amazon.com), Yixuan Shen `[通讯]` (Amazon.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多模态时间序列预测中辅助上下文是否真正提升性能，并提出基于自相关和条件互信息的诊断方法；

**💡 创新点**

提出两条必要条件（无竞争短路且上下文具统计信息）并通过因果干预验证，给出TRY/SKIP/INCONCLUSIVE三种诊断结果，首次系统化评估上下文价值；

**🔧 技术方法**

使用信息理论（条件互信息、短路上限）、kNN互信息估计与置换检验、MoME混合专家模型、控制实验与单背骨测试床、自动化诊断脚本；

**📊 数据集**

评估数据集包括8个带文本上下文的数据集（Time‑MMD 的 HealthUS、Environment 等）、FinMultiTime 以及27个 Monash Archive 无上下文的时间序列数据；

**📈 对比分析**

通过对比开启/关闭上下文、开启/关闭短路等实验，MoME 模型在满足两条件时 MSE 下降 29–51%，测试床在满足两条件时仅下降 2–5%，不满足条件时收益降至容量底线；

**⚠️ 局限性**

主要受限于结果集中大部分收益来自单一 MoME 模型，低样本量导致 MI 检验功效不足、仅验证文本上下文、测试床容量有限以及单阶自相关假设不适用于强季节性数据。

---

## 104. MTDiag: A Multi-Turn Diagnostic Dataset Towards Clinically Meaningful LLM Evaluation

**arXiv ID:** 2608.25085 | [PDF](https://arxiv.org/pdf/2608.25085v1)

**作者:** Pia Chouayfati `[一作]` (Technical University of Munich), Georg Groh `[通讯]` (Technical University of Munich)

**通讯引用:** 8165 | [OpenAlex ID](https://openalex.org/A5004398345)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模多轮诊断对话数据集MTDiag，并提供了基于UMLS/ICD‑10的本体锚定、对话生成与评估框架。

**💡 创新点**

首次将多源真实与合成病例映射到统一UMLS/ICD‑10 schema，提出症状挖掘、诊断距离、锚定偏差与伤害等级等对话级诊断指标。

**🔧 技术方法**

使用UserLM‑8B生成患者自然语言发言、MedGemma进行对话推理、spaCy+UMLS两阶段NER解析、UMLS/ICD‑10映射以及对话日志的CUI抽取与指标计算。

**📊 数据集**

数据集来源包括DDXPlus（合成差异诊断案例）、MIMIC‑IV（真实急诊/ICU记录）以及AJCR病例报告（罕见/复杂病例）。

**📈 对比分析**

通过MTDiag进行多轮对话评估，比较诊断准确率、症状挖掘分数、可靠性评分等；实验表明主流LLM在多轮情境下准确率下降约30%‑40%，并出现锚定偏差与延误护理风险。

**⚠️ 局限性**

局限性包括对UMLS版本的依赖、MIMIC数据的零数据保留限制、患者发言池覆盖有限、对话生成需人工审核以及评估需在合规环境下完成。

---

## 105. BanglaMamba: Exploring State Space Models for Bangla Fake News Detection

**arXiv ID:** 2608.25190 | [PDF](https://arxiv.org/pdf/2608.25190v1)

**作者:** M. K. Khalidi Siam `[一作]` `[通讯]` (BRAC University), M. K. Khalidi Siam (BRAC University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将 Mamba 基 State Space Model (BanglaMamba) 应用于孟加拉语假新闻检测，并与预训练的 BanglaBERT 及同规模从零训练的 CustomBERT 进行系统比较。

**💡 创新点**

首次将 Mamba SSM 引入孟加拉语假新闻检测，并展示其在保持相近分类性能的同时，在推理速度、吞吐量与显存占用方面的显著优势。

**🔧 技术方法**

使用 Mamba SSM 架构、BERT Transformer、微调、加权交叉熵损失、BF16 混合精度训练、Masked Mean Pooling、以及多种评估指标（Macro‑F1、AUC‑ROC、PR‑AUC、ECE、MCC、推理吞吐、GPU 峰值显存）。

**📊 数据集**

BanFakeNews‑2.0（用于训练、验证、测试）和 BanglaFakeNews2025（用于跨域评估）。

**📈 对比分析**

在相同 tokenizer、相同参数规模（≈13 M）下进行训练，使用 Macro‑F1 作为主指标。BanglaBERT 取得最高分类性能（Macro‑F1 0.926），BanglaMamba 与 CustomBERT 性能相近（≈0.903）。但 BanglaMamba 的推理 P50 延迟低 2.2 倍、吞吐率高 2.2 倍、峰值显存降低 49%。在跨域测试中，BanglaBERT 仍表现最好，其余两模型性能显著下降。

**⚠️ 局限性**

1) BanFakeNews‑2.0 存在严重类别不平衡，导致假新闻检测效果较差；2) BanglaMamba 仅在下游数据上训练，缺乏大规模预训练，导致跨域泛化能力弱；3) 由于资源限制未能对 BanglaMamba 进行大规模预训练，未能验证其潜在性能。

---

## 106. The AI Adaptation Gap in Higher Education: Students, Faculty, and Administrative Staff

**arXiv ID:** 2608.25063 | [PDF](https://arxiv.org/pdf/2608.25063v1)

**作者:** Yuriy S. Braun `[一作]` (Moscow Pedagogical State University), Salavat M. Khafizov `[通讯]` (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对一所专门从事教师教育的大型大学中1809名学生、250名教师和62名行政人员的角色适配问卷进行横断面调查，分析了人工智能（AI）的使用频率、感知效用、信任、学术诚信关注、负责使用规范以及制度政策清晰度等维度。

**💡 创新点**

创新点在于首次系统比较学生、教师和行政人员三大利益相关者在AI使用与态度上的差异，揭示了显著的AI适配差距，并将制度政策清晰度与信任之间的关联作为潜在治理杠杆；此外，还利用K‑means聚类探索学生内部的使用与态度多样性。

**🔧 技术方法**

采用了问卷设计、描述性统计、Welch方差分析和t检验、聚合后的OLS回归（信任模型和使用强度模型）、Cronbach α、PCA单因子检验以及K‑means聚类等统计技术。

**📊 数据集**

使用的主要数据集为2121份匿名问卷（共75题）收集于2025年12月，涵盖学生、教师和行政人员的AI使用经验、信任、政策认知等指标。

**📈 对比分析**

通过Welch方差分析和t检验比较三组之间的平均差异；OLS模型解释了约40.8%（信任）和42.1%（使用强度）的方差，感知效用对信任和使用强度的标准化回归系数分别为0.402和0.322，聚类结果识别出四类学生子群，表明学生群体内部存在显著的经验与态度异质性，但聚类稳定性未得到验证。

**⚠️ 局限性**

局限性包括：仅在单一高校进行，缺乏跨机构代表性；横断面设计无法确定因果关系；自报数据可能受社会期望偏差影响；样本规模不均衡（行政人员样本小）；未使用行为轨迹或客观学业成效指标；测量工具在不同组间的测量等价性仅为初步检验。

---

## 107. Less can be More: Relieving RAG Bottlenecks via Evidence Frontloading and Pressure-Adaptive Budgeting

**arXiv ID:** 2608.25115 | [PDF](https://arxiv.org/pdf/2608.25115v1)

**作者:** Weibin Cai `[一作]` (Syracuse University), Reza Zafarani `[通讯]` (Syracuse University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的框架Prioritized Adaptive Coverage of Evidence（PACE），通过证据前置和压力自适应预算在RAG系统中同时提升检索质量与响应速度。

**💡 创新点**

创新点在于将前置证据的子模子子集最大化问题与实时重排序预算自适应结合，并给出（1‑1/e）近似保证。

**🔧 技术方法**

采用子模子贪心前置策略、软锚点加权、压力自适应预算决策，配合密集检索器、再排序器和LLM。

**📊 数据集**

使用HotpotQA、MuSiQue、2WikiMultiHopQA三大多跳问答数据集。

**📈 对比分析**

与Dense、PRF、MMR、Dartboard、Adaptive‑K等无训练重排序方法以及压缩器相比，PACE在小预算下的完整证据召回率提升20%以上，p95延迟显著下降且比固定预算模型保持甚至更高的最终召回。

**⚠️ 局限性**

局限在于对检索器的高质量候选依赖，未对多模态或异构知识库检索场景进行验证，且压缩器对系统吞吐的综合影响仍有限。

---

## 108. A Comparative Evaluation of Digitization Pipelines for Historiographical Sources

**arXiv ID:** 2608.24976 | [PDF](https://arxiv.org/pdf/2608.24976v1)

**作者:** Marina Gómez Rey `[一作]` (Universidad Carlos III de Madrid), Carlos Alario-Hoyos `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对西班牙维吉特时期历史文件进行数字化，比较13种PDF‑to‑text管道。

**💡 创新点**

揭示端到端解析（Marker）在异质历史文本中最优，LLM后置纠错不一定提升。

**🔧 技术方法**

使用Marker、PyMuPDF、Docling、Tesseract、EasyOCR、MiniCPM‑V、Qwen3‑VL等开源OCR与视觉模型。

**📊 数据集**

由14份维吉特时期次级资料组成的14文档（共15个单元）分类为5类。

**📈 对比分析**

按字符错误率(CER)与词错误率(WER)宏平均评估，Marker达到约98.7% CER / 97.7% WER，其余方法平均低于90%，LLM后置导致精度下降。

**⚠️ 局限性**

局限在仅限西班牙语维吉特文献、样本规模小、仅使用编辑距离指标、未测试商业OCR与不同LLM、未评估语义检索效果。

---

## 109. AI-Powered Mental Health Chatbots in Africa: A Systematic Review and Culturally Adaptive Framework

**arXiv ID:** 2608.24890 | [PDF](https://arxiv.org/pdf/2608.24890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 110. Evidence-Grounded Mapping of Multimodal Human Sensing Psychological Transdiagnostic Dimensions

**arXiv ID:** 2608.24903 | [PDF](https://arxiv.org/pdf/2608.24903v1)

**作者:** Xiyun Hu `[一作]` (University of North Carolina at Chapel Hill), Jingping Nie `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5082699365)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个基于GLOBEM数据的临床验证基准，用来评估大语言模型将多模态感知、EMA和问卷证据映射到B-HiTOP维度的能力

**💡 创新点**

首次提出在缺乏实际B-HiTOP评分的情况下，利用AI评估器和临床验证来衡量证据兼容性，并比较直接预测与两阶段语义抽象两种推理流程的效果

**🔧 技术方法**

采用多种大型语言模型（DeepSeek、Grok、Kimi、Mistral、GPT、Llama、Lingshu），并设计两阶段推理流程（先生成心理抽象再进行项目评分）

**📊 数据集**

使用GLOBEM（多年手机/可穿戴被动感知+EMA+问卷）数据，共计14,592个参与者-日实例，提取29个B-HiTOP项目

**📈 对比分析**

在三种证据设置（被动感知、EMA+问卷、两者组合）下对7个LLM进行评估，利用固定AI评估器判断证据兼容性并在300个样本上做临床验证；结果显示两阶段抽象在EMA+问卷下兼容性提升至约53%，但在被动感知下兼容性下降至约22%，并且生成更保守的分数分布

**⚠️ 局限性**

主要限制包括：缺乏真实B-HiTOP标签导致只能评估证据兼容性；抽象化在行为感知数据上会丢失信息；项目覆盖率有限，仅保留29/45项；无法完全衡量模型的覆盖率和严重性区分

---

## 111. Resource-Efficient Pruning for Transformer via Low-Rank Importance Estimation

**arXiv ID:** 2608.24973 | [PDF](https://arxiv.org/pdf/2608.24973v1)

**作者:** Peng Liu `[一作]` (Guangdong University of Technology), Jigang Wu `[通讯]` (Guangdong University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 REP‑LIE 结构化剪枝框架，在剪枝过程中利用 LoRA 低秩矩阵的梯度评估权重重要性，并通过轻量化微调仅更新 LoRA 参数，显著降低计算和内存开销。

**💡 创新点**

创新点包括：① 仅使用 LoRA 低秩梯度进行重要性估计，避免全梯度计算；② 引入稳定性得分机制，缓解重要性估计中的随机性；③ 在剪枝过程中即时进行轻量化微调，只更新 LoRA 参数；④ 通过动态稀疏调度实现资源约束下的自适应剪枝。

**🔧 技术方法**

采用的技术有：LoRA 低秩适配、第一阶 Taylor 近似、稳定性得分、动态稀疏调度、结构化剪枝（头部、FFN 通道）、梯度重用和轻量化微调。

**📊 数据集**

实验数据集包括 GLUE 基准、WikiText‑2 语言建模、七个零样本推理任务（BoolQ、PIQA、HellaSwag、WinoGrande、ARC‑easy、ARC‑challenge、OpenBookQA），并在 BERT‑base、LLaMA‑7B、Mistral‑7B 上评估。

**📈 对比分析**

与 CoFi、PGB、DynaBERT、EBERT、TinyBERT、RECAP 以及 LLM‑Pruner、Compresso、SlimGPT 等现有剪枝方法对比。结果显示：在中等规模模型上 REP‑LIE 与全模型保持相近的 GLUE 分数；在大型模型 20–50% 稀疏度下，PPL 与零样本推理性能接近或优于对手；在内存占用、推理速度和 GPU 资源使用上均有显著提升。

**⚠️ 局限性**

局限性包括：尚未验证在已压缩（蒸馏、量化）模型或非 NLP 任务上的表现；对极端压缩比例或不同任务的鲁棒性未知；低秩梯度估计在某些结构（如 Mistral 的 GQA）可能需要更高秩以保持精度。

---

## 112. Same-Player Verification for Account Consistency in Counter-Strike 2

**arXiv ID:** 2608.24893 | [PDF](https://arxiv.org/pdf/2608.24893v1)

**作者:** Xuchen Zhang `[一作]` `[通讯]` (Independent Researcher), Xuchen Zhang (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 CS2 竞技游戏中，将账号历史一致性检查转化为开放式同玩家验证任务，利用演示文件生成玩家行为指纹并对同/异玩家进行比较。

**💡 创新点**

提出基于游戏理解的多层行为指纹（包括瞄准、机械、战斗、移动、经济等）和Transformer序列表示，并结合显式比较特征实现高精度同玩家判别；通过多 Demo 聚合提升账号级一致性评估。

**🔧 技术方法**

使用 LightGBM 进行对比评分，Transformer 对战斗窗口进行序列编码，组合游戏理解特征、序列嵌入、比较特征和地图上下文进行同玩家验证。

**📊 数据集**

主要数据集为 1,330 场 CS2 Demo（13,300 玩家观测），并补充 227 场公开职业 Demo（2,270 观测）进行扩展实验。

**📈 对比分析**

在六份人群分割上，单对比模型 ROC AUC 0.931，95% 召回率 0.722；多 Demo 聚合后 K=10 时 AUC 提升至 0.986，召回率 0.988；对比传统性能指标（K/D 等）显著提升，且对低级操作特征（瞄准、射击节奏、移动-射击协调）贡献最大。

**⚠️ 局限性**

局限包括：标签受人工确认和身份映射误差影响；模型仅使用单玩家指纹，忽略对战匹配、队友/对手信息；对时间跨度、版本和游戏版本差异敏感；部署需先验证历史 Demo 的一致性，并受运行时成本限制。

---

## 113. ROMNet: a hybrid reduced order modeling and machine learning approach to waveform inversion

**arXiv ID:** 2608.25160 | [PDF](https://arxiv.org/pdf/2608.25160v1)

**作者:** Liliana Borcea `[一作]` (Columbia University), Chugang Yi `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种混合波形逆演方法ROMNet，通过神经网络将数据驱动的简化阶模型（ROM）映射为对波速具有显式二次依赖的矩阵，随后使用可计算的优化求解波速；

**💡 创新点**

创新点在于利用神经网络在ROM层面实现对波速的显式近似，显著降低了传统ROM逆演的计算复杂度，并在非线性、周期跳过难题上提升了鲁棒性；

**🔧 技术方法**

使用的数据驱动ROM构造、块Cholesky分解、自动微分的Gauss-Newton优化、以及深度学习的神经网络（全连接/卷积架构）等技术；

**📊 数据集**

采用两套训练集：随机高斯叠加的波速模型和公开的GeoFWI地球物理模型；

**📈 对比分析**

与传统ROM逆演、Fourier-DeepONet和InversionNet进行对比，结果显示ROMNet在训练内分布下的相对L²误差最小（约0.03），在训练外结构化模型上也保持较好表现，并且每次迭代计算时间从数十秒降至几百毫秒，整体时间缩短至秒级；

**⚠️ 局限性**

局限性包括：对训练外极端结构（如BP-2004盐体）恢复效果不佳，且仍需良好的参考波速c₀；噪声敏感性需通过正则化处理；此外，深度网络对训练分布的依赖仍是主要挑战。

---

## 114. Detection != Reliable Control: Decodable Empathy Directions Yield at Most Partial Shifts in Automated Empathy Scores

**arXiv ID:** 2608.24901 | [PDF](https://arxiv.org/pdf/2608.24901v1)

**作者:** Haoran Jisun `[一作]` `[通讯]` (University of Southern California), Haoran Jisun (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过线性探测和激活注入，对大语言模型中的认知型“认知”与情感型“共鸣”两种同理心方向进行检测与干预。

**💡 创新点**

创新点在于同时检验可解码性、自动化指标控制和人类感知变化三条链，并通过正向控制阐明认知同理无法在当前工具下显著操控。

**🔧 技术方法**

主要技术包括EPITOME标签拆分、残差流读时与生成时方向估计、加权线性注入与全层消融、以及多仪器评估（LLM评委、EPITOME二元分类器和六评审面板）。

**📊 数据集**

使用的数据集包括ESConv情感支持语料、EPITOME Reddit同理心标注、以及补充的事实性对话。

**📈 对比分析**

对比方法采用正向对照与温度匹配的情感/中性对照、剂量-响应趋势检验及多模型消融实验，结果显示Qwen在共鸣方向上能提升≈26%自然情感差距，Gemma仅出现长度调整必要性消融效应。

**⚠️ 局限性**

局限性主要是测量敏感度不足导致认知同理干预不可测，样本量有限，人工评判门槛未通过，且仅限单轮对话，未验证多轮情境中的可持续性。

---

## 115. Multi-Modal Anomaly Detection: A Survey

**arXiv ID:** 2608.24937 | [PDF](https://arxiv.org/pdf/2608.24937v1)

**作者:** Xudong Mou `[一作]` (Beihang University), Renyu Yang `[通讯]` (Beihang University)

**通讯引用:** 3107 | [OpenAlex ID](https://openalex.org/A5050796169)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了多模态异常检测（MMAD）的研究现状，提出假设驱动的两大范式（正常性假设与异常性假设），并系统梳理了从表示学习、对齐、知识蒸馏到伪异常注入、结构/语义注入、生成模型及基础模型（FM）等技术路径。

**💡 创新点**

创新点在于用假设驱动的视角统一前沿工作，给出两种互补范式的框架，并强调基础模型如何在跨模态表示、对齐、知识注入及伪异常生成中扮演关键角色，提出未来研究的三大问题（伪异常可信度、跨模态语义推理、持续适应）。

**🔧 技术方法**

采用的技术包括：自编码器/GAN/Transformer 进行正常性特征学习；CLIP/VLM/LLM 进行跨模态对齐与知识蒸馏；结构破坏、语义注入、生成式伪异常等异常性注入方法；以及基于基础模型的多模态预训练、提示工程、推理链等。

**📊 数据集**

使用的典型数据集有工业与制造（MVTec AD、MVTec 3D-AD、VisA、Real3D-AD）、医疗诊断（MIMIC‑III、UCF‑Crime、SWaT、WADI、MVTec LOCO 等）、视频与监控（UCF‑Crime、XD‑Violence、ShanghaiTech、MSAD、UCF‑Crime‑DVS）以及多模态日志与传感器（HDFS、BGL、SMD、WADI、MIMIC‑III）。

**📈 对比分析**

通过与代表性方法（Deep SVDD、TranAD、AnomalyCLIP、AnomalyDiffusion、WinCLIP、M3DM、VAD‑CLIP、FOCA 等）在相应基准上对比，显示基于FM和伪异常注入的方案往往在像素/帧级 AUROC、AP 等指标上取得领先，特别是在零样本/少样本场景下表现突出。

**⚠️ 局限性**

局限性包括：模态不匹配导致对齐失败；伪异常生成真实性难以评估；对动态漂移的适应性不足；计算成本高，尤其是大型FM与扩散模型；以及对稀有/新型异常的泛化仍有限。

---

## 116. Clearing the Underbrush: AI-Enhanced RF Interference Suppression

**arXiv ID:** 2608.24974 | [PDF](https://arxiv.org/pdf/2608.24974v1)

**作者:** Rahul Jain `[一作]` (MIT Lincoln Laboratory), Alexia Schulz `[通讯]` (MIT Lincoln Laboratory)

**通讯引用:** 375 | [OpenAlex ID](https://openalex.org/A5076096528)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了基于Transformer与FSQ分词器的AI增强射频干扰抑制系统，针对QPSK语音与DVB‑T2干扰进行实验。

**💡 创新点**

在Transformer模型中加入FSQ分词器实现离散化表示，提升干扰抑制性能且保持低延迟；使用自回归生成并在边缘设备上实现高效推理。

**🔧 技术方法**

使用Transformer网络、RF WaveNet、RF Transformer Decoder、FSQ分词器、混合精度AMP、Torch‑TensorRT等技术实现模型训练与推理。

**📊 数据集**

使用2小时《Treasure Island》音频经μ‑law编码后QPSK调制得到SOI，合成DVB‑T2信号为干扰，混合后生成不同SINR的数据集。

**📈 对比分析**

与传统方法（匹配滤波、LMMSE、SIC）及之前AI模型对比，采用PESQ、SDR、LSD、Mel‑CD、STOI等指标，Transformer+Tokenizer在-6 dB SINR下仍保持优异；在Jetson上经过AMP与Torch‑TensorRT优化后实现约92 ms整体延迟、1.2 MHz输出吞吐，满足实时需求。

**⚠️ 局限性**

模型参数量大、推理时间长；低批量时AMP无显著加速；未在实际射频环境中验证，需提升模型鲁棒性和通用性；对未训练过的OFDM干扰可能表现不佳。

---

## 117. Simulating Cognitive Smart Freight Corridors with Agent-Based Models and Reinforcement Learning

**arXiv ID:** 2608.25193 | [PDF](https://arxiv.org/pdf/2608.25193v1)

**作者:** Madelaine Martinez-Ferguson `[一作]` (University of Tennessee), Xueping Li `[通讯]` (University of Tennessee)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个包含物理、网络、决策三层的代理基建模拟框架，用强化学习和多智能体强化学习来控制智能货运走廊的编队、充电和管理车道，并评估了三种场景（基线、辅助、认知）

**💡 创新点**

①将物理基础设施、V2X网络与学习型决策统一到同一ABM模型；②使用单智能体DQN控制走廊层级编队与车道，使用CTDE多智能体RL调度充电站价格和优先级；③在同一模型下对比连通性、辅助与认知三层级的性能

**🔧 技术方法**

Python实现的ABM、DQN强化学习、集中训练、分布式执行的多智能体强化学习（CTDE）、BPR拥堵模型、V2X通信仿真、离散动作空间的价格与优先级调度

**📊 数据集**

使用随机生成的走廊网络（20段、5个充电站、1500辆车）与多次随机种子产生的需求、容量与干扰场景，没有使用真实道路或物流数据集

**📈 对比分析**

通过在相同物理网络下运行基线、辅助、认知三种场景，衡量吞吐量、平均旅行时间、能耗、CO₂排放、拥堵指数和鲁棒性；认知场景吞吐量提升27%，拥堵指数下降64%，能耗每公里下降约7.9%，且在多种干扰条件下表现最稳健

**⚠️ 局限性**

仅考虑单向单路径网络，未包含动态路由、变速限制或竞争定价；模型仅在仿真中验证，缺乏真实道路数据和统计显著性检验；奖励设计与收敛性仍需进一步探索

---

## 118. Hyperbolic Latent Geometry for Tree-Structured Prototype Networks: A Local-vs-Global Trade-off

**arXiv ID:** 2608.25199 | [PDF](https://arxiv.org/pdf/2608.25199v1)

**作者:** Peter Flo `[一作]` (Harvard University), Luca Grossmann `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文研究了在层次分类任务中，使用欧氏空间与双曲空间两种潜在几何对类原型进行树形结构正则化，并比较其在保持邻接关系、分类准确率以及全局树形保真度方面的表现。

**💡 创新点**

创新点在于系统评估双曲几何在实际层次分类数据上的“局部 vs 全球”优势，证明双曲空间能显著提升邻接子树（兄弟、堂兄）召回率，而不必牺牲分类性能；并通过多重树结构和超参数搜索验证这一结论的稳健性。

**🔧 技术方法**

技术包括：冻结 CLIP ViT‑B/16 编码器、两层 MLP 生成潜在表示、在欧氏或 Poincaré 球上定义原型并使用软最大化距离模型；使用树形正则化（对原型距离矩阵与树距离矩阵的均值归一化误差进行惩罚）；Riemannian Adam 在双曲空间优化原型；在 150 条训练配置中进行随机种子复制并评估。

**📊 数据集**

数据集为 WikiArt‑Refined，约 81,446 幻灯片，27 艺术风格标签，已提供 70/30 训练/验证划分，使用冻结的 CLIP 特征作输入。

**📈 对比分析**

比较方法包括：top‑1 / top‑5 / 平衡准确率、全局树形 Spearman 相关和平均/最坏乘法失真、以及在验证集上对邻接子树的召回（兄弟、堂兄）@k。结果显示：欧氏原型在分类准确率和全局树形相关上略优（+5–6个百分点），而双曲原型在邻接召回率上显著领先（兄弟召回 @5 提升约 +3–4pp，整体平均提升 +8.7pp 兄弟召回、+15.2pp 堂兄召回），且该优势在三种参考树和不同超参数下保持稳定。

**⚠️ 局限性**

局限性包括：仅在中等规模、以西方艺术为主的 WikiArt 数据集上验证；未对编码器进行微调，可能限制双曲几何在更深或更大层次结构上的优势；全局树形保真度的评估结果不稳定，受参考树定义影响；双曲几何只作用于原型空间，未探讨其在整个模型（如 Möbius 变换层）中的潜在优势。

---

## 119. CA-less Mutual Co-Signing of Documents over a Unidirectional Visual Channel with Transported Hardware Attestation

**arXiv ID:** 2608.25144 | [PDF](https://arxiv.org/pdf/2608.25144v1)

**作者:** Dmytro Diikun `[一作]` `[通讯]` (Independent Researcher), Dmytro Diikun (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一种在没有网络、服务器或CA支持的情况下，两台移动设备通过单向光学通道完成互相签名并确保签名与设备硬件身份绑定的协议。

**💡 创新点**

核心创新在于：①使用两阶段哈希锚消除交互式共签中的循环依赖；②通过签名链使第二个签名与第一个签名不可分离；③将完整硬件证明令牌通过无反馈的水塘码直接传输并绑定到签名，实现证书自由的密钥来源证明。

**🔧 技术方法**

采用了 ECDSA‑P‑256 + SHA‑256、Apple App Attest / Google Play Integrity 设备鉴权、Fountain（LT 风格）码进行无反馈传输、基于控制字节的逃逸序列化和长度前缀 TLV 以保证编码可逆性。

**📊 数据集**

没有公开数据集；实验在 iOS（Secure Enclave）和 Android（StrongBox）设备上进行，利用真实设备的完整 attestation 令牌进行验证。

**📈 对比分析**

通过独立的离线验证器重算所有锚点和签名，验证过程不需要网络；实验表明在丢包率不超过约 30% 的光学通道下，传输可在约 5–10 秒内完成，签名验证时间在数十毫秒级。

**⚠️ 局限性**

局限性包括：依赖硬件厂商的 attestation 根，无法保证与自然人身份绑定；缺乏 CA 的可信链会使根证书泄露导致完整性失效；在不同平台间硬件信任度差异大；使用自定义水塘码参数不具备最优性；实验未通过独立安全审计。

---

## 120. DataKernelBench: Can LLMs Optimize Database Queries on GPUs?

**arXiv ID:** 2608.25061 | [PDF](https://arxiv.org/pdf/2608.25061v1)

**作者:** Gokul Karthik Kumar `[一作]` (IBM Research), Katja Hose `[通讯]` (TU Wien)

**通讯引用:** 4586 | [OpenAlex ID](https://openalex.org/A5015313855)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个名为 DataKernelBench 的基准框架，用于评估大型语言模型（LLM）生成的 GPU 核心，目标是加速 SQL 查询在 PyTorch/TorchPlan 运行时的执行。

**💡 创新点**

创新点：①首次提供面向数据库查询的 LLM 核心合成基准；②提供可验证的 SQL→TorchPlan 转换管道，使得 LLM 能在固定张量程序上进行优化；③系统评测十种 LLM（专有与开源）在不同 GPU 编程接口（CUDA、Triton）和优化层级下的性能差异，并揭示模型强度、上下文提示和优化范围对速度的影响。

**🔧 技术方法**

主要技术：TorchPlan 中间表示、基于多轮执行引导的 LLM 核心合成、CUDA/Triton 编程、Dask-cuDF 进行分区多 GPU 执行、执行指导反馈循环、基准自动化与结果验证。

**📊 数据集**

使用的数据集为 TPC‑H，规模因子 10（SF10）用于主实验，规模因子 100（SF100）用于分区多 GPU 的演示；对所有 22 个查询都生成了 TorchPlan 与 LLM 优化版本。

**📈 对比分析**

比较方法：对每个查询，评估 LLM 生成的模块与编译后的 TorchPlan（TC）以及外部系统 Sirius、DuckDB 的运行时间；统计通过率（pass rate）、速度提升（speedup）、所需修复回合（rounds）和 token 代价。性能结果显示，GPT‑5.5 在 CUDA‑Full 级别下实现 2.11× 的平均加速，且 100% 通过率；其他模型在 1.2–1.5× 之间，某些开源模型在 0.8–1.3× 左右。

**⚠️ 局限性**

局限性：①仅在单一 H100 GPU 上评估，缺乏跨硬件、跨工作负载的广泛验证；②框架仅支持 TorchPlan + CUDA/Triton，未覆盖其他 GPU 加速栈；③假设工作集能装入 GPU 内存（除演示的分区方案外未系统评估溢写或统一内存等策略）；④不涵盖从任意 SQL 到最终优化核的完整路径，仅评估在已验证的 TorchPlan 上的改进。

---

## 121. The Frame Kernel Method for Multiscale Operator Learning

**arXiv ID:** 2608.25084 | [PDF](https://arxiv.org/pdf/2608.25084v1)

**作者:** Branden Frieden `[一作]` (University of Utah), Varun Shankar `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多尺度核框架近似方法，用于多尺度偏微分方程的操作符学习与代理建模。

**💡 创新点**

创新点在于设计了一种兼容网格与点云、支持尺度分解的多尺度核框架，并将其嵌入核方法形成的Frame Kernel Method（FKM），显著提升预测精度并提供后验尺度解释。

**🔧 技术方法**

使用Wendland紧支撑RBF构建多尺度核框架，QR最小二乘求解求取最小范数插值；采用核回归进行多尺度正则化；并对输入/输出帧系数进行尺度归一化。

**📊 数据集**

实验数据集包括多种标准PDE基准（二维腔流、Darcy流、三维反应扩散、混合器、流动绕圆柱等）以及具有显著多尺度结构的PDE基准。

**📈 对比分析**

与Geo-FNO、Transolver、VKM等主流方法对比，FKM在大多数基准上误差降低1–4个数量级，尤其在多尺度问题中将相对ℓ₂误差压至0.1%以下。

**⚠️ 局限性**

局限性：QR求解受限于内存与稀疏性，难以扩展至百万级点；对不光滑或分段常数介质时收敛速率下降；需要先验选择尺度参数和正则化系数。

---

## 122. PARAssist: A Framework for Personalized and Adaptive Robotic Assistance from Ambiguous User Requests

**arXiv ID:** 2608.24905 | [PDF](https://arxiv.org/pdf/2608.24905v1)

**作者:** Pourya Aliasghari `[一作]` (University of Toronto), Goldie Nejat `[通讯]` (University of Toronto)

**通讯引用:** 5276 | [OpenAlex ID](https://openalex.org/A5061187299)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为 PARAssist 的架构，用于在服务机器人收到含糊不清的用户请求时，基于用户的历史行为和对话以及当前环境信息进行语义推理，从而生成候选任务并通过个性化偏好模型选择最合适的协助方案。

**💡 创新点**

创新点在于：① 将被动的多模态观察（视觉、语音、位置等）与用户协助偏好模型无缝集成；② 通过 VLM 推断任务的物理与认知需求，构造 Task Requirements Vector；③ 在不需要用户显式示例的情况下，从用户自然行为中学习偏好，实现真正的个性化歧义消解。

**🔧 技术方法**

使用的技术包括：Vision‑Language Models（VLM#1 用于行为观测，VLM#2 用于推断需求和生成候选任务，如 Qwen3.5‑9B、GPT‑5.4），YOLO‑based 人体检测，WhisperX 语音识别与说话人分离，Logistic Regression 作为 User Assistance Preference Model (UAPM)，以及 CodeBotler 任务规划器；并采用滑动窗口、投票等去噪策略。

**📊 数据集**

数据集主要为实验中收集的真实视频与对话：① 家庭日常任务视频（用户在不同房间进行抓取、移动等操作）；② 大学接待员与学生/教师的对话录音；同时使用预训练模型权重（YOLO、VLM）和公开的 OpenAI GPT 等。

**📈 对比分析**

通过与不同推理力度（MRE vs LRE）以及三种消融设置（NoPS、NoRS、NoHist）的比较，评估了系统的效果。结果显示：MRE 在 3 种情境中至少 9/10 次能给出包含关键需求（如高位抓取、弯腰、重物搬运）的建议；LRE 性能略逊；消融实验表明用户偏好评分和历史信息是提升个性化歧义消解的关键。系统平均延迟在 MRE 为 20–80 秒，LRE 下降到 20–30 秒。

**⚠️ 局限性**

局限性包括：① VLM 的周期性推断难以捕捉快速动作，可能导致建议已完成的任务；② 候选任务生成时缺乏语义地图或细粒度操作模型，导致出现不可行或不合理的建议；③ 需要额外计算资源才能提升推理速度；④ 目前只在两类场景（家庭与接待）验证，尚未在更广泛领域或用户研究中评估接受度。

---

## 123. An Open-Source Benchmark Suite of 3D-IC Testcases

**arXiv ID:** 2608.25155 | [PDF](https://arxiv.org/pdf/2608.25155v1)

**作者:** Rohan Soni `[一作]` (University of California Los Angeles), Puneet Gupta `[通讯]` (University of California Los Angeles)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于CATCH框架的开源3D-IC基准套件，包含20个2.5D/3D测试案例和可复用的虚拟chiplet库，以实现可复现的物理设计评估。

**💡 创新点**

创新点在于构建统一且可扩展的基准体系，涵盖从基础到复杂的混合架构，提供标准化的3Dblox描述和自动化转换工具，弥补了现有3D基准资源不足的空白。

**🔧 技术方法**

本文利用CATCH XML生成设计、将其转换为3Dblox格式，并结合OpenROAD生态实现后端物理设计流程，支持自动化的基准生成与评估。

**📊 数据集**

使用了20个基准设计，基于CATCH定义的多种虚拟芯片模板，公开发布在Dryad中的3Dblox描述文件作为数据集。

**📈 对比分析**

通过统一的3Dblox描述在标准化后端流程（放置、路由、RC提取、时序、热分析）进行评估，可与不同方法公平对比，但本文未给出具体性能指标。

**⚠️ 局限性**

局限性包括基准主要基于模拟模板，缺乏工业真实案例；未覆盖所有新兴封装技术；评估主要聚焦后端流程，对前端架构与工艺约束考虑不足。

---

## 124. Super Star: Towards Streaming Real-time Interactive Agents for Digital Humans

**arXiv ID:** 2608.24909 | [PDF](https://arxiv.org/pdf/2608.24909v1)

**作者:** Wentao Jiang `[一作]` (ShanghaiTech University), Jingya Wang `[通讯]` (ShanghaiTech University)

**通讯引用:** 9633 | [OpenAlex ID](https://openalex.org/A5100639519)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

建立了实时交互式的在线共语手势生成框架，将流式语音响应与因果多模态自回归手势生成器结合，实现低延迟同步手势。

**💡 创新点**

引入因果音频注意力掩码与跨模态自回归分解，设计闭环自演进数据合成与用户反馈循环，支持虚拟伴侣场景的持续自适应。

**🔧 技术方法**

采用因果多模态自回归模型、VQ‑VAE分词、跨模态注意力、流式语音生成、离线交互数据合成以及用户反馈驱动的自演进训练等技术。

**📊 数据集**

使用公开的 BEATv2 数据集和自建的高质量交互式数据集 JIYI。

**📈 对比分析**

与 5 个主流离线方法（Semantic Gesticulator、TalkSHOW、SynTalker、EMAGE、LOM）在严格在线协议下对比，取得最低延迟、最高 FGD 与 BC，用户研究显示最高整体偏好。

**⚠️ 局限性**

长期规划能力有限、缺乏种子数据时的泛化、细粒度偏好建模不充分、扩展化身范围受限等仍需改进。

---

## 125. Hallucination by proxy in LLM-assisted differential diagnosis

**arXiv ID:** 2608.24908 | [PDF](https://arxiv.org/pdf/2608.24908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 126. Unsupervised Post-Training of Foundation Models: A Survey

**arXiv ID:** 2608.24982 | [PDF](https://arxiv.org/pdf/2608.24982v1)

**作者:** Yijie Xu `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

综述并系统化了基于未标注数据的模型后训练（Unsupervised Post-Training）方法，提出了统一的四类内部更新对象与时间维度框架。

**💡 创新点**

创新点在于将UPT方法按照内部更新对象（预测统计、样本关系、自生成目标、内部评估器）进行严格分类，并引入输入可见性×更新持续性二维视角来评估部署情景。

**🔧 技术方法**

技术上使用了边界检查B1–B4来定义严格UPT，梳理并归档80条方法，结合任务结构审计与实验结果进行跨方法比较。

**📊 数据集**

主要基于现有论文使用的多种数据集（文本、对话、数学、图文等），未自行收集或训练数据集，而是对已有研究中的数据集进行汇总。

**📈 对比分析**

通过表格对每类方法在不同任务结构和部署时机下的指标（如下游准确率、AIME、MATH等）进行对照，展示各类方法在相应场景下的性能提升，但未做统一基准评估。

**⚠️ 局限性**

局限性包括仅覆盖至2026年5月的工作，缺乏对混合方法的深入分析，未提供统一效果量化，且未评估自我强化过程中的误差累积与鲁棒性。

---

## 127. SimVerity: When Does Simulated Agent Success Survive Physical Deployment?

**arXiv ID:** 2608.25067 | [PDF](https://arxiv.org/pdf/2608.25067v1)

**作者:** Zhonghao Zhan `[一作]` (Imperial College London), Hamed Haddadi `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出SimVerity框架，将仿真评估的通过与真实智能家居部署的结果进行配对对照，利用独立资格的摄像机等物理见证者进行跨验证，从而评估仿真通过在实际物理环境中的可靠性。

**💡 创新点**

核心创新点在于：①将“判定转移”视为可度量的保证契约，以每个属性、边界、路径、部署层级的“判定保真度”和“误判风险”为评估指标；②通过冻结风险配置并在校准阶段学习，提前预测仿真通过后可能出现的失败；③把审计结果以“清晰/弃权/升级”的三态卡片形式输出，使部署决策可追溯、可复现。

**🔧 技术方法**

技术包括：情境匹配与语义对齐的轨迹适配器、属性监控器（检测完成、读后写、可观测效应等）、资格化的摄像机见证者、基于贝叶斯平滑的风险配置学习、离线Brier评分与统计区间、可冻结的评估卡片与哈希签名保证不可篡改。

**📊 数据集**

数据集：在一套真实的智能家居实验平台上收集了9个有效校准会话（586条实验，1070条源通过配对）以及在两组队列中的总计36条held‑out物理路径-层级配对；另外在两个不同的部署站点收集了120条和80条实验，验证跨平台迁移。

**📈 对比分析**

比较方法：将SimVerity的预测与“属性盲目率”、“全局误判风险”、“设备延迟查找”以及“路径仅风险”六种基线进行对比，使用Brier评分、Wilcoxon检验与确切区间评估。结果显示SimVerity在两组队列中均在Brier评分上优于基线，held‑out路径的误判风险显著下降（如第1组8/8次胜利、p=0.0039），并且在配置级别的可审计性实验中显示出较高的匹配率。

**⚠️ 局限性**

局限性：评估仅覆盖单一灯光、门触发等有限路径；受限于摄像机的光照条件，部分短时效应无法观测导致弃权；仅在两套物理平台验证，未覆盖更广泛的设备与厂商生态；模型‑客户端/服务端配置的干扰难以完全分离；整体方法需在更大规模、更多环境下进一步验证。

---

## 128. LifePlanner: Evaluating LLM Agents for Geo-spatial Planning with Social Media Data

**arXiv ID:** 2608.25039 | [PDF](https://arxiv.org/pdf/2608.25039v1)

**作者:** Zhen Dong `[一作]` (Wuhan University), Haiping Wang `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出LifePlanner benchmark，用于评估LLM在含有地图与社交媒体信息的地理空间规划任务中的表现。

**💡 创新点**

创新点在于将大规模地区社交媒体数据与地图信息结合，并提供多难度、多任务的规划评测框架。

**🔧 技术方法**

使用的技术包括MCP工具集实现地图与社交媒体检索、LLM代理、检索增强生成（RAG）以及多指标评估。

**📊 数据集**

数据集涵盖约10km²城市区域的3,600个地点与约20万条社交媒体贴文及评论。

**📈 对比分析**

通过对多款闭源与开源LLM进行统一评测，结果显示最优模型Pass Rate从L0的89%降至L2的约40%，表明高难度规划仍距实用有较大差距。

**⚠️ 局限性**

局限性包括社交媒体仅为单一时间点快照、查询为人工合成、仅单轮规划，未涵盖多轮交互与时间动态性。

---

## 129. What Should a Large Language Model See? Physical Invariants as a Data Representation for PDE Discovery

**arXiv ID:** 2608.25189 | [PDF](https://arxiv.org/pdf/2608.25189v1)

**作者:** Fan Yang `[一作]` (California Institute of Technology), Matt Thomson `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个“数据解释”阶段，将时空场数据转换为物理诊断信息，然后将这些信息作为输入喂给大型语言模型（LLM），用于符号回归寻找支配方程。

**💡 创新点**

创新点在于：①不需要将原始高维时空数据直接放入prompt，而是用一组低维、可解释的诊断量（如频谱衰减率、时序阶数、非线性指标、输运量）直接传递给LLM；②该方法无需训练，且显著提升了LLM在 PDE 发现任务中的准确率；③为LLM提供了与人类理论家思维相近的“观察”方式。

**🔧 技术方法**

使用的技术包括：频谱分析（FFT）提取模式演化；基于模式衰减率和频率的线性回归；非线性与输运的统计量计算；LLM（QwQ‑32B）进行结构生成；解析与规范化（canonicalisation）以统一表示；数值回归 + 稀疏惩罚评估候选 PDE。

**📊 数据集**

数据集：44个合成 PDE 示例，字段为标量 u(x,t) 在周期域上，通过随机选取八个标准项（扩散、输运、反应、阻尼波）以及随机系数生成；每个实例在 64×64 网格上积分 50 帧。

**📈 对比分析**

比较方法：将 LLM 输入设为三种情况——(1) 数据解释诊断；(2) 随机打乱的诊断（控制）；(3) 直接提供 10 条 1D 切片的原始字段（≈1230 tokens）。评估指标为 F1 分数和精确恢复率。结果显示：数据解释的平均 F1 为 0.720，精确恢复 14/44；相比之下原始切片仅 0.225（2/43），打乱诊断 0.244；固定最佳术语集仅 0.391。性能提升显著，p 值在 10⁻⁵ 级别。

**⚠️ 局限性**

局限性：①仅在小规模、无噪声的模拟数据上验证；②诊断量有限，无法完整捕捉非线性项的形式；③对实验噪声敏感，需要鲁棒差分和去噪；④仅针对标量场，未扩展到多场或复杂几何；⑤LLM 固化，未探索模型微调的可能性。

---

## 130. Padamitra: Grounded Glossary Generation for Classical Sanskrit

**arXiv ID:** 2608.25038 | [PDF](https://arxiv.org/pdf/2608.25038v1)

**作者:** Manoj Balaji Jagadeeshan `[一作]` (Indian Institute of Technology), Pawan Goyal `[通讯]` (Indian Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了“基于翻译的词典生成（Grounded Glossary Generation）”任务，要求模型在给定梵文颂文和对应英文译文的情况下，恢复语义上连贯的梵文短语并给出与译文对齐的释义；并构建了包含31,316条三元组的基准数据集；在此基础上评估了多种语言模型，尤其是通过指令微调（Instruction Fine‑Tuning）显著提升了生成质量；并对模型错误进行细致分析，指出过度分割（sandhi/samāsa）是主要瓶颈。

**💡 创新点**

创新点在于（1）将传统梵文释义实践形式化为可评估的NLP任务；（2）提出了针对键值对的两项专门评估指标——Jaccard（键恢复）与Meaning Faithfulness（译文对齐的语义一致性）；（3）系统比较了指令微调、零样本、少样本等多种策略，并揭示了语义分割错误的主要原因。

**🔧 技术方法**

技术方面主要使用大型语言模型（Gemma‑3‑12B、Phi‑4、Qwen3.5‑9B 等）与 LoRA 微调；对比了基线方法 FastAlign、ByT5‑Sanskrit 的端到端与两阶段管线；采用了自定义的分割增强与词序扰动实验来探究模型鲁棒性。

**📊 数据集**

数据集来自两部经典梵文文本——《维摩利王阎罗颂》（Valmiki Ramayana）和《斯里玛德·巴格瓦塔姆》（Srimad Bhagavatam），包含训练/验证/测试三份（25,050/3,133/3,133），每条记录为一条梵文颂文、其英文译文与结构化词典条目。

**📈 对比分析**

在Jaccard（键恢复）和Meaning Faithfulness（释义一致性）两项指标上，指令微调模型比零样本/少样本大幅领先：Phi‑4 指令微调在Jaccard上从0.53提升至0.71，在Meaning Faithfulness上从0.55提升至0.79；Gemma‑3‑12B 同样实现了明显提升。相较于传统的 FastAlign/ByT5‑Sanskrit 基线，指令微调模型整体表现提高了约20–30%。

**⚠️ 局限性**

局限性包括：①数据集仅涵盖两部文本，可能无法代表其他梵文文学的风格；②评估指标依赖表面字符串匹配与嵌入相似度，无法充分处理形态学等价的沙尼分割；③指令微调受限于显存，使用 LoRA 而非全参数微调，可能低估潜在性能；④错误分析仅聚焦最低5%样本，未覆盖全部分布。

---

## 131. Physics-Informed Error Field Learning: A Post-Training Optimization Framework for Physics-Informed Neural Networks

**arXiv ID:** 2608.24970 | [PDF](https://arxiv.org/pdf/2608.24970v1)

**作者:** Jiuyun Sun `[一作]` (Shandong University of Science and Technology), Yong Zhang `[通讯]` (Shandong University of Science and Technology)

**通讯引用:** 57951 | [OpenAlex ID](https://openalex.org/A5100419806)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种物理信息误差场学习框架PIEFL，对PINN进行后训练误差校正。

**💡 创新点**

创新点是将学习目标从完整解转为误差场并在物理约束下训练辅助网络，减少后期优化成本。

**🔧 技术方法**

采用物理信息神经网络(PINN)架构，构造误差控制方程，并利用TensorFlow训练主网络与误差网络。

**📊 数据集**

在三类代表性PDE（KdV、非线性薛定谔方程、KP方程）上进行实验，使用对应解析解作为基准。

**📈 对比分析**

与传统PINN持续训练比较，PIEFL在相同迭代/计算预算下实现了误差降低1-2个数量级，计算效率提升明显。

**⚠️ 局限性**

局限在误差场难以精确逼近复杂高维解时校正效果减弱，需进一步改进误差网络设计。

---

## 132. SWIM: Step-Wise Integrated Measure for Session-supervised List Evaluation in Generative Re-ranking

**arXiv ID:** 2608.25104 | [PDF](https://arxiv.org/pdf/2608.25104v1)

**作者:** Yuanhao Pu `[一作]` (University of Science and Technology of China), Kun Gai `[通讯]` (Unaffiliated)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对连续消费场景的生成式重排序，提出了基于会话级生存过程的评估器 SWIM，能够在列表级别对用户停留概率和收益进行联合建模。

**💡 创新点**

创新点包括：① 引入会话前缀到当前请求的边界生存概率 q₀，充分捕捉会话上下文；② 采用递归生存分布与到达位置的条件奖励分解，得到前缀条件的整体价值；③ 使用因果遮蔽 Transformer 并行估计所有步骤的生存和奖励，满足工业低延迟需求。

**🔧 技术方法**

技术手段包括：因果遮蔽 Transformer、离散生存链估计、条件奖励头（二分类与桶化连续回报）、多目标联合训练、离散化再投影回报、正向/负向采样策略以及基于概率损失与奖励损失的联合优化。

**📊 数据集**

实验数据集：公开的 RecFlow（候选集120，列表长度6）和 KuaiRand（用户行为序列重构的列表重排任务），以及在 400M DAU 的 Kuaishou APP 上的真实在线日志。

**📈 对比分析**

与 10+ 传统列表式、生成式、G‑E 框架基线（DNN、Seq2Slate、DLCM、SetRank、PRM、SORT‑Gen、NAR4Rec、PIER、MultiG、CAVE）比较，SWIM 在 RecFlow 上 NDCG@6 提升至 0.2031、AUC 0.7804，KuaiRand 上同样获得最高分；在线 A/B 测试中对照 CAVE，SWIM 分别提升 App stay time +0.351% 和 7‑day retention +0.048%。

**⚠️ 局限性**

局限性：① 依赖丰富的会话前缀特征，若缺失或噪声较多效果受限；② 生存链仅定义在有限长度列表，对极长会话或动态列表生成可能需要进一步扩展；③ 训练时需要精确的到达标签，离散化回报可能导致细粒度信息损失。

---

## 133. ExFold: Unified Expert Folding for Training-Free MoE Prefill-Decode Acceleration

**arXiv ID:** 2608.24938 | [PDF](https://arxiv.org/pdf/2608.24938v1)

**作者:** Juntong Wu `[一作]` (Xiaohongshu Inc), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 18046 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的专家折叠（Expert Folding）框架，统一加速MoE模型的预填充（prefill）和解码（decode）两阶段推理，恢复被裁剪专家的贡献。

**💡 创新点**

创新点包括：1) 用有向标量投影器（pairwise scalar projector）在不训练的情况下校准专家间的尺度差异；2) 将被排除专家的贡献折叠到保留专家的路由权重中，从而避免直接丢弃；3) 采用相同的投影器在两阶段使用不同的专家选取策略，统一了质量目标；4) 通过轻量级CUDA核实现高效插件化。

**🔧 技术方法**

核心技术包括：离线无标签数据校准获得专家对的标量投影表；基于路由分数与输出幅度的联合评分进行专家选取；将折叠操作嵌入路由权重更新；在vLLM中实现插件和Triton自定义算子。

**📊 数据集**

实验使用的模型和数据集：Qwen3-30B-A3B、GLM-4.5-Air、DeepSeek-V2-Lite、DeepSeek-V4-Flash、Qwen3.5-35B-A3B；评测基准包括MATH500、AIME24、IFEval、IFBench、GPQA、MMLU-Pro、Eval+、LiveCodeBenchV5等。

**📈 对比分析**

与传统基于Top‑K稀疏化（Direct Top‑K、MC‑MoE、MoDES）和专家集合合并（REAP、SERE）等方法对比，实验显示：prefill TTFT可提升至1.41×，decode TPOT可提升至2.45×，在保持约99%原始平均质量的前提下，整体优于单阶段或单方法加速方案。

**⚠️ 局限性**

局限性包括：需要额外的无标签校准步骤，标量表会占用一定显存；对极大专家数的扩展效果尚未验证；目前仅适用于MoE结构，无法直接迁移到无专家的模型；折叠方法在极端预算压缩下仍可能出现一定质量损失。

---

## 134. Control-Oriented Learning for Dynamic Tracking and Stability Analysis of Soft Pneumatic Actuators

**arXiv ID:** 2608.25171 | [PDF](https://arxiv.org/pdf/2608.25171v1)

**作者:** Nithin S. Kumar `[一作]` (Vanderbilt University), Eric J. Barth `[通讯]` (Vanderbilt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种控制导向的学习框架，先用二次多项式学习软气动活塞的静态压强-位置映射，再用EDMDc识别线性残差动力学，实现了高精度轨迹跟踪、实时用户驱动与障碍回避。

**💡 创新点**

创新点在于将非线性静态关系与线性残差动力学分离，使控制器保持线性可设计，同时通过学习得到的残差模型可进行闭环稳定性预测，从而实现了模型解释性、实时性与稳定性保证的统一。

**🔧 技术方法**

采用二次多项式回归、稀疏回归（用于验证高阶项）、Extended Dynamic Mode Decomposition with Control (EDMDc) 进行残差模型识别，并使用 PI 反馈与正则化伪逆实现控制。

**📊 数据集**

数据集为500秒的阶梯压强激励实验，收集了161个稳态压强-位置对与完整的时间序列，用于训练静态映射与残差动力学。

**📈 对比分析**

与纯前馈、纯反馈或传统基于物理模型的控制器相比，所提出方法在低速轨迹下均值误差约1 mm，高速约10 mm，能够跟踪最大加速度25 m/s²的用户指令，并成功实现实时障碍回避，显示出优异的跟踪精度与鲁棒性。

**⚠️ 局限性**

局限性包括模型仅基于自由空间运动，未显式考虑外部负载或接触；稳定性分析为局部线性预测，可能在大幅工作点变化或长期材料老化时失效；在其他软体结构（如张力驱动或同轴管）上的适用性尚未验证。

---

## 135. ARISMA: Guidelines for AI- and LLM-Assisted Systematic Reviews, Scoping Reviews, and Mapping Studies

**arXiv ID:** 2608.25050 | [PDF](https://arxiv.org/pdf/2608.25050v1)

**作者:** Mahyar Tourchi Moghaddam `[一作]` (University of Southern Denmark), Mina Alipour `[通讯]` (University of Southern Denmark)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并阐述了 ARISMA（AI‑Reporting and Integration Standard for Systematic Methods and Analysis）框架，给出 AI 辅助系统综述、范围综述和映射研究的生命周期治理、验证流程与报告规范。

**💡 创新点**

创新点在于将 AI 的使用置于可审计、可逆的治理体系中，制定了验证矩阵、可追溯的审计链、与 PRISMA 等现有标准的整合，并强调 AI 只能作为受控助手，而非自主评审者。

**🔧 技术方法**

使用了 LLM（如 GPT‑4、GPT‑4o）和机器学习模型进行检索扩展、标题/摘要筛选、数据提取、关键词/分类生成与文本草拟，同时通过专家咨询、模型日志与手工验证实现技术治理。

**📊 数据集**

参考了多项公开实验与实证研究的数据，包括 Sentinel 研究集、Pilot 评估样本、公开系统综述的原始文献和数据库检索结果；并未依赖单一特定数据集。

**📈 对比分析**

通过专家咨询与已有文献的对比，设定检索召回、筛选灵敏度、提取准确率等阈值；性能因任务而异，研究显示在特定环境下可达 95% 以上的召回或 80% 以上的提取准确率，但普适性仍需验证。

**⚠️ 局限性**

局限性包括：仅基于专家咨询而非大规模实证验证；对高风险任务（如最终决策、量化分析）的自动化仍缺乏充分证据；对不同学科、语言和模型更新的适用性与可复现性尚未充分评估。

---

## 136. Evaluating and Preventing Security Smells in AI-Generated Ansible Code

**arXiv ID:** 2608.24962 | [PDF](https://arxiv.org/pdf/2608.24962v1)

**作者:** Pandu Ranga Reddy Konala `[一作]` (University of Waikato), Junaid Haseeb `[通讯]` (University of Waikato)

**通讯引用:** 113 | [OpenAlex ID](https://openalex.org/A5027225262)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了AI模型在生成IaC时的安全性与合规性。

**💡 创新点**

提出了将Ansible最佳实践和CIS基准嵌入提示的扩展CO-STAR框架，实现安全合规代码一次性生成。

**🔧 技术方法**

使用扩展CO-STAR提示、IaC质量框架和CIS-CAT Pro检测。

**📊 数据集**

基准数据集为278个Ansible角色（Tomcat 135，MongoDB 143），包含16款AI模型生成的角色与119/127人类编写角色。

**📈 对比分析**

通过对比AI模型与人类代码的安全烟味、质量分和CIS合规率，发现四个模型在有提示下可达95–100%合规，质量提升19–49%，人类仅23–43%。

**⚠️ 局限性**

局限在模型版本变动、只评估两种技术与CIS Level1、仅零轮交互，未覆盖更高级实施组或其他IaC语言。

---

## 137. The Dialect Tax: Dialectal Biases Persist throughout the Language Modeling Pipeline

**arXiv ID:** 2608.24952 | [PDF](https://arxiv.org/pdf/2608.24952v1)

**作者:** Elle `[一作]` (University of Oxford), Elle `[通讯]` (University of Oxford)

**通讯引用:** 38 | [OpenAlex ID](https://openalex.org/A5103998539)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了语言模型管线各阶段（分词、预训练、后训练、推理）中方言差异，并量化了所谓的方言税。

**💡 创新点**

首次在同一实验框架下同时追踪四个阶段的偏差，揭示方言税是多阶段累积效应而非单一来源，提供了对方言不公平更全面的视角。

**🔧 技术方法**

采用字符级对照tokenizer、CountSketch压缩梯度、语义相似度（EmbeddingGemma）评估、熵分析、隐藏层相似度判别、逻辑回归方言辨识等技术。

**📊 数据集**

使用的并行方言数据集包括 ParallelAAVE、MultiVALUE、ReDial、CoQA 及规则生成的多方言文本，保证了语义保持而仅改变表面形式。

**📈 对比分析**

通过余弦相似度、梯度 z 分数、交叉熵损失、熵差、准确率等指标比较 SAE 与 AAVE 等方言；实验显示方言在所有阶段均显著劣势，损失差距可达约 0.5 nats、准确率差距数个百分点。

**⚠️ 局限性**

受限于仅使用开放模型与数据、未完成完整训练 ablation、缺乏对大型封闭系统的验证、以及对多场景方言覆盖不足，导致结论在更广泛环境下的泛化性受限。

---

## 138. Does Fine-Tuning Undo Activation Steering? Behavioural Recovery Without Weight-Edit Reversal

**arXiv ID:** 2608.24988 | [PDF](https://arxiv.org/pdf/2608.24988v1)

**作者:** Philipp E. Glass `[一作]` (Brunel University of London), Alina Miron `[通讯]` (Brunel University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型中嵌入激活调节（steering）后，研究其在常规下游全参数微调（SFT 与 RLHF）下的功能与机制稳定性。

**💡 创新点**

发现嵌入的权重编辑在微调后几乎不被撤销，表明机制上稳健；但功能上易受训练数据反向信号影响，凸显了嵌入式调节的可验证性需求。

**🔧 技术方法**

使用线性激活方向估计、投影嵌入（projection steering）与放大/消除方向、全参数微调以及行为评估与向量恢复分析等技术。

**📊 数据集**

利用 OpenOrca（SFT）与 Anthropic hh‑rlhf（RLHF）数据集，且在评估阶段使用 Hugging Face 版 100 个有害提示和 300 个长回答提示。

**📈 对比分析**

通过对比未调节基线、嵌入调节后以及微调后模型的拒绝率与回答长度进行衡量，结果显示 RLHF 能保持 95%+ 的调节效果，而 SFT 对拒绝消除易退化（恢复 60%），但对回答简洁性影响较小；向量恢复率平均仅 0.4%。

**⚠️ 局限性**

限制包括仅测试 5 个开源模型、仅考虑全参数微调、未系统评估不同程度对立信号的影响、缺乏对量化等其他权重扰动的研究，并未揭示模型如何绕过已被嵌入的权重路径。

---

## 139. SkyDrive: Learning to Drive in a New City from Aerial Traffic Monitoring

**arXiv ID:** 2608.25142 | [PDF](https://arxiv.org/pdf/2608.25142v1)

**作者:** Weijiang Xiong `[一作]` (École Polytechnique Fédérale de Lausanne), Nikolas Geroliminis `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用无人机交通监控数据构建SkyDrive框架，将空中监控转化为自动驾驶训练的监督源。

**💡 创新点**

创新点在于通过可扩展的空中视角获取海量驾驶样本并验证少量空中监督可显著提升零样本迁移性能。

**🔧 技术方法**

使用多视角语义图像、地图信息及轨迹生成与预测模型，对比DrivoR、RAP等轨迹规划器以及AutoBot、MTR、Wayformer等预测器。

**📊 数据集**

主要使用SongdoTraffic数据集（137.2小时、650K场景）作为SkyDrive的基础。

**📈 对比分析**

与公开模型零样本对比，性能大幅下降；加入约30分钟空中监督后，ADE/FDE/TTC/NCT等指标提升60%以上；预测任务在Wayformer上取得最佳结果。

**⚠️ 局限性**

局限包括对城市内部交叉口差异的泛化仍有限，仅评估20个交叉口；缺乏多城市、多天气和更复杂场景的验证。

---

## 140. Visualizing Patient Trajectories and Disorder Co-occurrences in Child and Adolescent Mental Health

**arXiv ID:** 2608.24911 | [PDF](https://arxiv.org/pdf/2608.24911v1)

**作者:** Dipendra Pant `[一作]` (NTNU), Øystein Nytrø `[通讯]` (NTNU)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发并评估了基于儿童与青少年精神健康服务（CAMHS）电子健康记录的患者轨迹可视化与共病网络图，帮助临床医生快速了解患者随访历程与ADHD共病情况。

**💡 创新点**

将患者按年龄组、性别和是否存在ADHD进行12类聚类，并在可视化中将不同聚类用色块表示；同时在ICD‑10层级3细化度下展示ADHD与共病的网络结构，提升信息可读性。

**🔧 技术方法**

采用K‑Prototype无监督聚类算法（Huang初始化、Gower距离）对聚类标签进行赋值；使用时间轴轨迹图、叠加模式图与网络图进行可视化，辅以色彩、大小与标签编码。

**📊 数据集**

使用了约19,248名儿童及青少年、22,676条就诊记录的35年CAMHS电子健康数据，涵盖诊断、用药、访视次数等信息。

**📈 对比分析**

通过与五名临床医师的访谈与问卷评估可视化的可理解性与实用性；在可视化层级3下共病网络信息最易解读，聚类结果在ADHD组中显著出现特定聚类。

**⚠️ 局限性**

样本仅来自挪威单一区域，临床评估者人数有限，且网络图在最高细度时过于拥挤；未来需在更大样本与多地区验证、优化交互与说明文案以提升可用性。

---

## 141. AFDBench: A Reasoning-First AI Scientist for NationalWeather Service Forecast Discussions

**arXiv ID:** 2608.24954 | [PDF](https://arxiv.org/pdf/2608.24954v1)

**作者:** Manmeet Singh `[一作]` (Western Kentucky University), Josh Durkee `[通讯]` (Western Kentucky University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5008235839)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建AFDBench基准，训练7B参数AI气象学家生成NWS区域预测讨论。

**💡 创新点**

首次提出面向专业气象推理的基准、基于思维先行的数据格式，以及针对温度、同步性与格式的GRPO奖励。

**🔧 技术方法**

使用Qwen2.5-7B-Instruct+LoRA 4-bit SFT与GRPO，奖励设计包括温度、同步、格式。

**📊 数据集**

收集7,732份来自13个NWS办公室的专家讨论与对应Google WeatherNext 2单时刻预测的JSON。

**📈 对比分析**

与零射击开源7-8B模型对比，GRPO将Style‑Align从0.318提升至0.619，Input‑Grounding从0.881提升至0.940，Met‑Align约14%。

**⚠️ 局限性**

单时刻输入限制导致Met‑Align无法提升；SFT效果有限，奖励可能易被玩弄；仅评估7B模型，未覆盖多时段输入和国际数据。

---

## 142. When Does Frequency Decomposition Benefit Physics-Informed Neural Networks? A Preliminary Ablation Study

**arXiv ID:** 2608.24940 | [PDF](https://arxiv.org/pdf/2608.24940v1)

**作者:** Shubham Rai `[一作]` `[通讯]` (bibha.ai), Shubham Rai (bibha.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过引入可拆卸的低频/高频双分支以及自适应门控网络，构建了DBSG-PINN架构，并在五个一维基准 PDE 上进行消融实验，研究频率分解对 PINN 性能的影响。

**💡 创新点**

创新点在于设计了可单独消融的低频分支、高频分支和门控模块，使得能直接量化频率分解和门控对模型精度的贡献，并首次提出门控效益随目标解谱复杂度变化的经验观察。

**🔧 技术方法**

使用了 PINN 训练框架，低频分支采用 tanh 激活，高频分支采用正弦激活，门控网络采用 sigmoid 混合；训练过程采用 Adam + L‑BFGS，评估指标包括相对 L2、L∞、PDE 残差、谱误差及 HF/LF 恢复分数。

**📊 数据集**

实验基于五个一维 PDE（多模波、Allen–Cahn、Burgers、Reaction–Diffusion、单模波），使用解析解或高分辨率数值参考解进行评估。

**📈 对比分析**

在相同的训练设置下与三种消融变体（无门控、仅低频、仅高频）对比，结果显示频率分解对多尺度问题能显著降低误差（最高相对 L2 误差下降 59%），但对光滑或单模问题收益有限，部分消融模型甚至优于完整模型。

**⚠️ 局限性**

局限性包括仅使用单个随机种子、仅在一维任务上测试、分支容量匹配不严格、缺乏门控激活可视化，需要多种子和更广泛基准验证以确认结论的稳健性。

---

## 143. SHIFT-LLM: Distribution Shift Correction in Depth-Pruned LLMs

**arXiv ID:** 2608.25068 | [PDF](https://arxiv.org/pdf/2608.25068v1)

**作者:** Ali Bahri `[一作]` (Huawei Noah's Ark Lab), Zhitang Chen `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种训练无关的深度裁剪后修正框架SHIFT‑LLM，利用低秩残差适配器（LRA）在被裁剪层插入保留身份路径的轻量级正则化，逼近被删除Transformer块的残差更新，恢复隐藏状态分布；

**💡 创新点**

其创新点在于：①仅对残差更新做线性逼近，保留身份路径从而降低逼近难度；②通过闭式最小二乘回归在少量校准样本上快速估计参数；③支持低秩压缩与连续LRA的精确合并，实现极低额外计算；

**🔧 技术方法**

主要技术包括低秩残差适配器、闭式岭回归校准、低秩SVD压缩、LRA的序列级融合以及与LoRA、部分层微调的联合使用；

**📊 数据集**

使用的校准数据为少量（如256个）来自C4或其他文本语料的样本，评估数据涵盖七个zero‑shot基准（BoolQ、PIQA、HellaSwag等）以及WikiText‑2 perplexity；

**📈 对比分析**

与未修正的裁剪模型、传统LoRA微调、部分层微调以及现有主流裁剪方法（如Navigation LLM、ShortGPT）对比，SHIFT‑LLM在多数模型和裁剪策略下可恢复多达+15.7点zero‑shot准确率，且对Llama‑3.1‑8B‑Instruct实现了约1.27×的推理速度提升；

**⚠️ 局限性**

局限性包括：对某些模型（如Vicuna‑7B）在特定裁剪策略下恢复效果有限，逼近精度受残差更新复杂度限制，需额外校准数据且对极大裁剪比例的鲁棒性尚待进一步验证。

---

## 144. Semantic Variability of Replies Across LLMs: Implications for Designing Conversation-Based Assessment

**arXiv ID:** 2608.24920 | [PDF](https://arxiv.org/pdf/2608.24920v1)

**作者:** Jiangang Hao `[一作]` `[通讯]` (ETS Research Institute), Jiangang Hao (ETS Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对来自真实协作对话的消息，比较不同大语言模型（LLM）在有无聊天历史提示下生成回复的语义一致性。

**💡 创新点**

首次系统评估LLM架构和版本变化对评测对话一致性的影响，揭示仅靠提示和上下文难以保证跨模型的语义稳定。

**🔧 技术方法**

使用OpenAI text-embedding-3-large生成文本嵌入，计算余弦相似度，并通过线性混合效应模型分析不同模型和提示条件下的平均相似度与方差。

**📊 数据集**

基于2018年Amazon Mechanical Turk收集的99支团队的双人在线协作科学任务聊天记录，挑选61条高相关性晚期回复作为分析样本。

**📈 对比分析**

通过比较模型内部回复的平均相似度、模型间回复的相似度、回复与人工回复的相似度以及有无历史提示的差异，发现LLM模型族群和历史提示对语义一致性有显著但不一致的影响。

**⚠️ 局限性**

研究仅覆盖少数LLM版本、仅考虑晚期高相关性消息、并使用单一任务数据集，因而对不同任务、时间点和更广泛LLM生态的普适性有限。

---

## 145. Reliable LLM-Powered Decision Engines for Large-Scale Supply Chain Operations: Architecture, Safety, and Performance Guarantees

**arXiv ID:** 2608.24889 | [PDF](https://arxiv.org/pdf/2608.24889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 146. Extending Ground-Constraint LiDAR-IMU Calibration to Tilted Surfaces in a Continuous-Time Framework

**arXiv ID:** 2608.25135 | [PDF](https://arxiv.org/pdf/2608.25135v1)

**作者:** Vassili Korotkine `[一作]` (McGill University), James Richard Forbes `[通讯]` (McGill University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种连续时间目标无目标的 LiDAR-IMU 标定方法，能够在非平地（倾斜平面）上对地面车辆进行单轴运动下的标定。

**💡 创新点**

设计了不依赖平地假设的地面平面约束残差，包括距离残差和倾斜角残差，解决单轴运动导致的可观测性丧失，并给出简化的平地方向残差，首次实现倾斜地面上的可靠标定。

**🔧 技术方法**

采用连续时间 B‑Spline 对 IMU 轨迹进行参数化，基于 OA‑Calib 框架实现非线性最小二乘优化；使用 Patchwork++ 对 LiDAR 点云进行地面分割提取平面信息；引入距离残差和倾斜角残差作为新的约束。

**📊 数据集**

在 Husky Clearpath UGV（室内平地+户外倾斜）、M2DGR 数据集（Velodyne VLP-32C + Handsfree A9 IMU）以及 Offroad 车辆（Ouster OS1-32 + XSENS MTI‑200-2A8G4）上进行评估。

**📈 对比分析**

与现有的 GRIL‑Calib 进行对比。平地场景下性能相当且更优的连续时间特性；倾斜地面上 GRIL‑Calib 发散，而本方法保持收敛并显著提升位置与姿态的重复性，尤其是沿单轴运动不可观测方向的校准精度。

**⚠️ 局限性**

仍需预先知道 IMU 高度和平地重力向量；倾斜角需要额外传感器或先行平地标定；对初始值敏感，误差传播至下游估计的完整评估尚未完成。

---

## 147. post-graph-rag: A PostgreSQL-Native Graph RAG Engine

**arXiv ID:** 2608.24921 | [PDF](https://arxiv.org/pdf/2608.24921v1)

**作者:** Chandan Rajah `[一作]` `[通讯]` (Independent Researcher), Chandan Rajah (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套基于 PostgreSQL 的图 RAG 引擎 C‑6ptpost‑graph‑rag，集成向量检索、实体图、文档‑实体边和社区摘要于一体，实现统一存储、事务一致性与时间建模。

**💡 创新点**

创新点包括：① 提前在提取阶段做严格的验证门控（拒绝模糊谓词、代词、量化名词等）以保证图质量；② 采用实体别名和“最长表述”规则实现跨文档实体消歧；③ 建立可选的有效期区间和文档顺序超越机制，实现事实随时间演化的表示；④ 通过可配置的词汇表将谓词归一化，显著降低边标签冗余；⑤ 在单表 PostgreSQL 中实现多租户（realm/space）和可追溯的历史表，避免了多数据库部署的复杂性。

**🔧 技术方法**

技术实现基于 PostgreSQL + HNSW 向量索引、JSONB 结构存储、Leiden/Label‑Propagation 社区检测、LLM 进行三元组提取和聚合，使用 Llama‑3.3‑70B、MiniMax‑M2.7、DeepSeek‑V3.2、gemma‑4‑31B 等模型做实验；检索采用向量检索 + k‑hop 语义扩展 + 文本段落拉取 + 主题社区检索，答案通过多源提示融合生成。

**📊 数据集**

数据集包括：① Wikipedia 四篇关于 Ada Lovelace、Charles Babbage 等的条目（127k 字符）；② 法国 19 世纪小说《Dumas》三部曲（645k 字符，剧情随时间变化）；③ 美国波音公司 10‑K 财报（5 篇，约 587k 字符，年度顺序）。

**📈 对比分析**

对比方法：在相同提取模型、嵌入模型和分块设置下与 LightRAG 进行基准比较。结果显示：① 图密度提升 30%+（实体+关系）；② 边标签稀疏度降低（边/标签比从 100% 降至 11%）；③ 查询延迟 1.5‑2× 更快；④ 通过时间层实现 13‑8 条关系的超越（LightRAG 无此功能）。整体回答质量相当，但 C‑6ptpost‑graph‑rag 在多租户、事务一致性和时间演化方面具备显著优势。

**⚠️ 局限性**

局限性：① 仅在三种英文长文本语料上评估，未覆盖对话、代码、非命名实体类域；② 词汇表与域紧耦合，需针对不同领域扩展；③ 评测为单次运行，受 LLM 随机性影响；④ 未给出索引吞吐量比较（因两系统持久化机制不同）；⑤ 超越机制依赖实体密度和声明的排他谓词组，稀疏图中无法触发；⑥ 递归深度有限，未实现基于相关性的深度裁剪。

---

## 148. CRESSim-Neo: A Batched GPU Simulation Engine for Surgical Robotics and Robot Learning

**arXiv ID:** 2608.25192 | [PDF](https://arxiv.org/pdf/2608.25192v1)

**作者:** Yafei Ou `[一作]` (University of Alberta), Mahdi Tavakoli `[通讯]` (University of Alberta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一款GPU加速的外科机器人仿真引擎，支持多物理域（刚体、软体、流体、线性结构）和批量环境运行。

**💡 创新点**

创新点在于将PBD/XPBD物理求解、批量光栅化渲染、GPU驻留数据管线以及外科特定传感（手术工具、超声合成）整合为统一平台，满足机器人学习与合成数据生成需求。

**🔧 技术方法**

使用技术包括位置基动力学（PBD/XPBD）、Vulkan/Direct3D12后端、Diligent Engine、HLSL自定义计算、DLPack/CUDA互操作、GPU批量渲染和自定义传感器。

**📊 数据集**

主要使用自建的合成手术场景和强化学习任务；未引用公开数据集，实验基于内部构建的手术机器人、流体、软体、超声等场景。

**📈 对比分析**

与现有仿真平台对比，单GPU可实现8192个CartPole的2.03M步/秒；手术相关任务在RTX4090上实现数千至数万步/秒；RL训练吞吐与纯步进相近，证明高效可扩展。

**⚠️ 局限性**

局限性包括：PBD对刚体接触与关节动力学精度不足；内存随环境数线性增长，动态拓扑需重新分配；跨平台支持有限（仅Vulkan/D3D12，CUDA必需）；渲染仅支持光栅化，无射线追踪；自定义计算仅限于预定义的GPU缓冲与渲染目标。

---

## 149. Lowering the Barrier to AI-Driven Inspection: A No-Code Workflow for Automated Structural Defect Detection

**arXiv ID:** 2608.25176 | [PDF](https://arxiv.org/pdf/2608.25176v1)

**作者:** Michael Holm `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了YOLOEZ，一款无代码、图形化界面的工具，用于结构健康监测中的缺陷（裂纹）检测，涵盖标注、训练和推理全过程。

**💡 创新点**

创新点在于将数据标注、YOLO模型训练、推理集成到单一GUI中，降低非程序员的技术门槛，提供可复现的端到端工作流，并以开源形式发布。

**🔧 技术方法**

使用技术包括YOLOv11分割模型、Python+GUI框架、数据增强、二值化分割损失和mAP评估；工具实现为开源项目。

**📊 数据集**

采用增材制造的钨样品扫描电镜（SEM）图像作为数据集，共20张训练、5张验证、15张测试图像，并通过YOLOEZ进行四次数据增强。

**📈 对比分析**

与传统形态学阈值+形态学过滤的基线方法比较，YOLOEZ在召回率（0.6065 vs 0.4372）、F1分数（0.5591 vs 0.5353）、IoU（0.4150 vs 0.3962）以及缺陷计数准确度上均优于基线；精度略低，但特异性相近。

**⚠️ 局限性**

局限性包括目前仅支持单类别缺陷检测，缺乏多类别、多模型并行等高级功能，且在复杂多类别场景下的表现尚未验证。

---

## 150. Retrieve, Match, Escalate: Accurate and Scalable Product Linking with VLM-Distilled Cross-Encoders and Agentic VLMs

**arXiv ID:** 2608.25037 | [PDF](https://arxiv.org/pdf/2608.25037v1)

**作者:** Jian Wang `[一作]` (DoorDash Inc.), Kyle MacDonald `[通讯]` (DoorDash Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个在门店级别商品链接的retrieve‑then‑match流水线，用检索、轻量级cross‑encoder和代理式多模态VLM依次处理不同难度的匹配问题。

**💡 创新点**

创新点在于通过置信度路由的三级递进式推理、利用双VLM共识标签进行大规模蒸馏、以及将闭源前沿VLM迁移为自托管的开源专家混合模型，从而在保持98%精度的前提下显著提升覆盖率和成本效率。

**🔧 技术方法**

使用的技术包括近似最近邻检索（图像、文本、条形码通道）、ModernBERT‑base交叉编码器、Qwen 3.6 35B‑A3B多模态代理、MCP工具调用、自动化实验（autoresearch）与多模型蒸馏。

**📊 数据集**

数据集来源于DoorDash商家SKU记录与十几百万条规范商品目录，训练集中包含约530万条双VLM共识标注对，评估集包含dedup与已关联对，以及人工审核与专家对比样本。

**📈 对比分析**

与传统单模型匹配和人类操作相比，检索阶段召回93.06%，交叉编码器在98%精度下自动接受率43.7%，代理式VLM相较人类提升13.7pp召回、4.7pp精度，整体端到端覆盖率从68.1%提升至77.1%，且开源代理成本比闭源前沿模型低约7倍。

**⚠️ 局限性**

主要局限包括对置信度阈值的手工调校、对罰差条形码冲突的稀缺性导致的误判、检索覆盖与商品目录质量的瓶颈，以及需要持续人工审核与多模态模型维护的复杂性。

---

## 151. ToolMinimize: Auditing and Rewriting LLM Agent Tool Calls to Minimize Privacy Exposure

**arXiv ID:** 2608.24957 | [PDF](https://arxiv.org/pdf/2608.24957v1)

**作者:** Wenbiao Li `[一作]` (Case Western Reserve University), Yuqiao Xu `[通讯]` (Case Western Reserve University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5000046363)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了名为 ToolMinimize 的中间件，能够拦截 LLM 代理的工具调用，识别并计算隐私敏感数据（PSD）的隐私成本，并在保留任务有效性的前提下通过删除、泛化、替换、截断等操作对调用参数进行最小化。

**💡 创新点**

创新点在于：① 定义了基于 GDPR Art. 9 的 PSD 分类与量化隐私成本指标；② 结合模式匹配与实体抽取的两阶段分类器、JSON Schema 必要性分析，以及四种重写策略，实现了方案级别的字段级隐私最小化；③ 设计了可选的 LLM 驱动内容必要性层，进一步提升了对隐私信息的识别与处理。

**🔧 技术方法**

技术实现主要采用正则+实体抽取的两阶段分类器、JSON Schema 解析、四种重写操作、可选的 LLM（GPT‑4o）内容必要性判定，并以插件方式集成到 AutoGen、MCP、LangChain 等主流代理框架；中间件采用无侵入式的钩子，保持极低的延迟。

**📊 数据集**

评估使用了自建的 AgentPrivBench（90 个真实场景、约 300 个工具调用）、25 个公开 MCP 模式、50 个 PrivacyLens 外部情境，并在 GPT‑4o、Claude Sonnet、Llama‑3.3‑70B 三大 LLM 上进行现场调用。

**📈 对比分析**

与 9 种基线（无防护、提示、PII 检测、PrivacyChecker 等）比较，ToolMinimize 在 100% 任务完整率的前提下，将隐私成本降低 81–92%，内容层可提升至 85–96%；平均延迟为 1.77 ms，跨框架和跨模型均保持一致的效果。

**⚠️ 局限性**

局限性包括：① 受分类器召回率限制，未能覆盖所有必要的 PSD；② 对需要精确定位（如地图）工具高度依赖完整 schema，缺失时效果受限；③ 仍有部分编码逃逸（如字段拆分）未被检测；④ 误报率约 20%/10%，需要进一步改进；⑤ 未实现跨会话预算和多模态 PSD 检测等功能。

---

## 152. A Primer on Computational Semantics for Artificial Intelligence Systems

**arXiv ID:** 2608.25022 | [PDF](https://arxiv.org/pdf/2608.25022v1)

**作者:** Casey Kennington `[一作]` `[通讯]` (Boise State University), Casey Kennington (Boise State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并比较人类语言习得与基于Transformer的大型语言模型在语义学习上的差异与相似性。

**💡 创新点**

提出将形式语义、符号地面化语义与分布式语义整合为统一的“认知模型”视角，并强调情感与身体经验在语言意义中的核心作用。

**🔧 技术方法**

讨论逻辑推理（形式语义）、感知驱动的符号地面化方法、分布式向量表示、Transformer架构及多模态视觉语言模型等技术。

**📊 数据集**

主要引用公开文本语料库（如维基百科、通用语言模型训练集）和儿童语言习得实验数据，未自行构建实验数据集。

**📈 对比分析**

通过理论分析与文献对比，指出仅基于文本的Transformer缺乏感知与情感维度，无法完整复制人类意义建构；无实验性能指标。

**⚠️ 局限性**

受限于缺乏真实感知与身体经验，模型在深层语义、情感和多模态理解方面仍显不足；文章为综述性，缺乏定量验证。

---

## 153. Stronger Alignment between Brain Activity and LLM Embeddings during Code Writing compared to Prose Writing

**arXiv ID:** 2608.24900 | [PDF](https://arxiv.org/pdf/2608.24900v1)

**作者:** Zachary Karas `[一作]` (Vanderbilt University), Yu Huang `[通讯]` (Vanderbilt University)

**通讯引用:** 116714 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用 LLM 嵌入构建 voxelwise encoding 模型，预测程序员在 fMRI 环境下书写代码和散文时的大脑 BOLD 信号。

**💡 创新点**

首次将大规模语言模型的内部向量与人脑神经信号进行对齐，并发现代码书写时的脑活动与 LLM 嵌入的相似度显著高于散文书写；同时揭示右前额极在两种任务中均为最佳预测区。

**🔧 技术方法**

采用 Ridge 回归构建 VEM，使用 PCA 降维 LLM 嵌入（8 层），并系统调节延迟复制数（0/4/10/16/20）和 look‑ahead 视窗；同时比较 6 个开源 LLM（CodeGemma 2B/7B、DeepSeek 2B/6B、StarCoder 3B/7B）的表现。

**📊 数据集**

使用来自 23 名计算机科学本科生/研究生的 fMRI 数据集：在 3T GE MR750 扫描仪下收集的代码（C++）和散文回答的 60 ms 采样 keystroke 与 BOLD 信号；数据与原始实验平台公开可获取。

**📈 对比分析**

对每个模型和层级计算每 voxel 的 Pearson 相关系数，取 Top‑10 K voxels 取平均；对代码与散文分别得到显著差异（p<0.001，FDR 校正），最佳模型为 DeepSeek 6B（10 个延迟复制，无 look‑ahead）在代码任务中的平均相关 0.582±0.053，散文 0.491±0.052；层级间一致性高，参与者间相似度低，表明个体化“指纹”。

**⚠️ 局限性**

限制包括：仅使用 keystroke 作为刺激输入，未覆盖空闲时间；任务仅涉及 C++ 与英语，缺乏跨语言验证；受 MRI 约束导致打字困难；对 LLM 细粒度编辑行为的量化较粗；模型对参与者手势与视线等多模态信息缺乏捕捉。

---

## 154. PA-CoT: Profile-Adaptive Chain-of-Thought for Personalized Nutritional Consulting

**arXiv ID:** 2608.24907 | [PDF](https://arxiv.org/pdf/2608.24907v1)

**作者:** Evgenii Garmashov `[一作]` (ITMO University), Sergey Muravyov `[通讯]` (ITMO University)

**通讯引用:** 85 | [OpenAlex ID](https://openalex.org/A5048360368)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PA‑CoT（Profile‑Adaptive Chain‑of‑Thought）多阶段提示框架，并构建了QPA（Question–Profile–Answer）基准，用以评估在营养咨询中基于结构化用户资料的个性化和安全性。

**💡 创新点**

创新点在于将用户资料分析拆分为专门的“Profile Analysis”阶段和“Safety Verification”阶段，形成一个完整的分析‑生成‑校验流水线，使模型在回答前先明确需要关注的资料要点并对危险输出做二次过滤。

**🔧 技术方法**

采用的技术包括多阶段Prompting、Chain‑of‑Thought推理、profile relevance matrix（相关性矩阵）、温度控制（分析与校验用低温，生成用中温）、LLM评估（G‑Eval、Qwen3评判）以及GPT‑4o‑Mini作为生成模型。

**📊 数据集**

使用Medical Alpaca（约23,000条医患问答）为来源，筛选出含营养信息的问答并抽取结构化资料，最终生成200条带有参考答案的QPA样本。

**📈 对比分析**

与11种现有提示方法（CoT、Few‑Shot、Role Prompting、Self‑Refine、DSPy、TextGrad、AMPO、Mixture of Prompts、Meta‑Prompting、PhaseEvo、Zero‑Shot Baseline）进行比较。PA‑CoT在平均得分4.21、个性化4.71和安全4.68上均超过所有竞争者，尤其在个性化和安全指标上与最近竞争者（Self‑Refine）显著差距，CI不重叠。

**⚠️ 局限性**

局限性包括：安全评估依赖LLM判定，缺乏专业医师或注册营养师审核；Stage 2的交互式补全在基准实验中未实现，未检验真实用户交互效果；数据集规模有限，且主要来自医疗论坛，可能不完全代表实际营养咨询场景；模型仍为研究原型，需进一步在人机交互和临床安全性方面验证。

---

## 155. D$^3$-MOPD: Adaptive Dynamic Domain ScheDuling for Efficient Multi-Teacher Distillation

**arXiv ID:** 2608.24987 | [PDF](https://arxiv.org/pdf/2608.24987v1)

**作者:** Zechen Sun `[一作]` (AllSpark), Min Zhang `[通讯]` (AllSpark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了动态域调度方法 D^3-MOPD，用来在多教师 on-policy distillation 中根据学生对各域的学习进展实时调整训练样本比例。

**💡 创新点**

创新点在于将训练过程中已生成的 per-domain reverse‑KL 信号拆解为“剩余 gap”与“下降速度”两维，组合成复合指标并通过温度软最大化+下限映射实时更新域采样比例，实现零开销且可扩展的动态调度。

**🔧 技术方法**

主要技术包括：反向 KL 监督、离线 watcher 监控 KL 历史、复合信号 s_k=KL·v、温度软最大化与下限约束、批次级随机 jitter、窗口平滑的下降速度估计。

**📊 数据集**

使用了四个领域的专家教师（数学、代码、指令遵循、工具使用）各约 4k 题目作为训练域，并在七个多领域基准（AIME 2025、HMMT 2025、LiveCodeBench、OJBench、IFBench、IFEval、BFCL v3 Multi‑Turn）进行评估。

**📈 对比分析**

与传统固定域比例的 vanilla MOPD 进行对比，D^3‑MOPD 在平均归一化分数上提升至 0.97（对比 0.63），在 3 倍更少的 rollout 步数（约 95 步）内达到峰值，并在三项基准上超过专家教师；Ablation 证明复合信号、批次 jitter 与速度平滑均对精度与收敛速度有正面影响。

**⚠️ 局限性**

局限性包括：需要额外的 watcher 进程与 KL 记录，复合信号设计依赖超参数（温度、下限、窗口大小），在极端不平衡或数据量极大的域组合下可能需要进一步调优；目前仅验证于单个 Qwen3.6‑35B‑A3B 学生模型，跨模型或更大规模验证仍待探索。

---

## 156. Semantic Graph Unification for Industrial Digital Threads: Bridging 11 Heterogeneous Manufacturing Systems Through Ontology-Driven Knowledge Graphs

**arXiv ID:** 2608.24918 | [PDF](https://arxiv.org/pdf/2608.24918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 157. Why and When Neural Networks Improve Local Approximation in Optimization

**arXiv ID:** 2608.24963 | [PDF](https://arxiv.org/pdf/2608.24963v1)

**作者:** Chengkuo Bian `[一作]` (University of California, Berkeley), Pengcheng Xie `[通讯]` (Lawrence Berkeley National Laboratory, University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在无梯度优化中使用神经网络代理模型的效果，证明代理的提升取决于其在算法中的角色、可可靠泛化的半径以及基础方法的剩余可用空间，而非单纯的逼近精度。

**💡 创新点**

提出了三条关键因素（助推vs替代、半径可泛化性、方法剩余空间）解释代理表现差异，并给出了半径感知的全局线性条件与自适应半径-感知代理（ARAS）框架。

**🔧 技术方法**

采用可微神经网络代理（含Sobolev梯度正则化）、自适应半径评估、梯度替代与保障接受测试的机制、以及基于全局线性理论的误差分析。

**📊 数据集**

在13个经典无约束测试函数（如Sphere、Rosenbrock、Extended Wood等）与噪声模拟实验（10种噪声水平）以及真实的联合库存模型上进行实验，维度从4到128。

**📈 对比分析**

将代理模型与多种基准DFO方法（FD梯度、模型基信任域、Py-BOBYQA等）进行对比，结果显示助推式代理可将解答实例从67提升至84（τ=10⁻⁵），但在替代式或已高度优化的信任域中几乎无效，噪声环境下代理作用进一步减弱。

**⚠️ 局限性**

局限在于代理模型仅在弱基准或特定问题（如Extended Wood）中有显著提升；在噪声或高维情形下，代理几乎无效；且代理训练成本高、对梯度目标噪声敏感，需要进一步改进采样与接受测试机制。

---

## 158. FABRICA: Agentic CUDA-to-CSL Translation and Optimization for Wafer-Scale Systems

**arXiv ID:** 2608.25124 | [PDF](https://arxiv.org/pdf/2608.25124v1)

**作者:** Yuebo Luo `[一作]` (University of Minnesota, Twin Cities), Le Chen `[通讯]` (Argonne National Laboratory)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究CUDA程序向Cerebras Software Language (CSL) 的自动翻译与优化，构建了包含49对CUDA/CSL内核的基准集，并实现了一个由多阶段角色组成的 agentic 框架，支持知识检索、执行反馈、错误分类与循环优化。

**💡 创新点**

创新点：①首次提供完整的跨架构翻译基准（49对CUDA-CSL内核）及对应评测指标；②提出无训练知识支持的 agentic 翻译框架，结合目标知识检索、失败引导修复和正确性门控的优化循环；③系统化展示知识检索、失败反馈和同目标测量对翻译成功率和性能提升的决定性作用。

**🔧 技术方法**

技术：多模态大语言模型（Claude Opus 4.8/4.6、Sonnet 4.6、开源模型等）按角色（分析师、建筑师、实现者、评审、优化器）调用；知识库检索（SDK 文档、教程、案例）；基于Cerebras SDK的编译、运行、数值验证与设备周期计数；在WSE‑3模拟器与真实硬件上进行性能测评。

**📊 数据集**

数据集：49对CUDA/CSL任务（28个核心测试集、31个原始任务、18个新增机器学习/集合任务），包含源代码、参考CSL实现与评测元数据；以及从这些任务生成的 6,066 条翻译/修复记录用于离线 SFT 训练。

**📈 对比分析**

比较方法：在固定核心集上对单次生成、完整工作流和跨模型（Claude vs 开源）进行准确率和速度比较；在 49 任务覆盖评估中统计正确率；在 27 对可比较的周期测量对上计算几何平均加速；结果显示：单次生成 6/28 正确率提升至 26/28，整体覆盖率 38/49；几何平均加速从模拟器的 3.75× 降至硬件的 3.47×，大部分任务保持或提升性能。

**⚠️ 局限性**

局限性：①仍缺乏真正的低层 CSL 编译器或 IR，翻译过程依赖模型与手工检查；②对目标知识（Cerebras API、通信模式）的掌握仍依赖外部检索，难以完全自动化；③优化空间受限于目标语言特性，某些任务仍难以超越参考实现；④离线 SFT 提升了结构化指标但未保证可执行 CSL 的正确性。

---

## 159. Trust the Mass: Forced Weights in KV-Cache Eviction

**arXiv ID:** 2608.25230 | [PDF](https://arxiv.org/pdf/2608.25230v1)

**作者:** Jack Shi `[一作]` (Stanford University), Jerry Gu `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了 KV 缓存的稀疏注意力与逐键丢弃规则，量化了在强制权重下最佳子集能填补的误差上限，并基于此设计了名为 ContourKV 的无训练分配器；

**💡 创新点**

创新点在于：① 用枚举与平衡选择器量化“强制权重”情形下子集最优误差，并证明顶重（top‑mass）已逼近最优；② 提出通过“丢弃质量”指标预测可关闭的误差比例；③ 将内存使用与压缩精度分离，形成统一的评估框架；

**🔧 技术方法**

技术手段包括：枚举最佳子集（上限约 10^6 组合）、平衡选择器（贪心换键）、丢弃质量统计、预算约束下的物理回收、与多模型多层多头的批量评估；

**📊 数据集**

数据集覆盖 10 组模型（Qwen、Llama、OLMo 等），token 长度 4k–32k，基准为 RULER、LongBench 以及针尖检索实验；

**📈 对比分析**

与 SnapKV、PyramidKV、KVzip、Compactor 等 14 种基线对比；在未强制预算时 ContourKV 在 146/157 条件下击败 KVzip，平均提升 18–30 分；强制预算后仍以 93/160 条件赢得 KVzip，且与 Compactor 形成平手；在内存占比 0.15–5.3% 的压缩下实现优于大多数方法的准确率；

**⚠️ 局限性**

局限性包括：① 子集最优误差上限仅为原误差的 2–5%；② 强制权重问题 NP‑hard，无法在大预算（s≥64）下做全枚举；③ 评估侧重单次推理，未覆盖多轮交互的动态缓存回收；④ 结果对模型架构与窗口化敏感，需针对不同查询‑键规范手动调整阈值。

---

## 160. PhaseShift: Topology-Aware Data Harmonization and Model Consolidation Across Signalized Intersections

**arXiv ID:** 2608.25275 | [PDF](https://arxiv.org/pdf/2608.25275v1)

**作者:** Yash Ranjan `[一作]` (University of Florida), Sanjay Ranka `[通讯]` (University of Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 PhaseShift，一种基于拓扑的框架，用来将来自不同信号灯交叉口的路边轨迹数据统一为相同的以行人为中心的表示，并在此基础上训练一个可共享的行车行为模型。

**💡 创新点**

创新点在于通过 ego‑relative 坐标、轨迹诱导的行驶路径、规范化信号状态以及可变长度的交互代币，对跨站点的物理-控制差异进行“去偶”处理，从而实现多交叉口数据的无缝融合和模型合并。

**🔧 技术方法**

采用了基于空间注意力的 Actor‑Conditional Transformer 结构，配合多模态 GMM 输出头；先通过轨迹诱导生成运动多边形实现几何统一，再对训练样本进行窗口采样和掩码处理；在推理时使用自回归推断与最佳样本选择。

**📊 数据集**

使用了佛罗里达州五个信号交叉口的真实道路数据（4 个 Gainesville 区域交叉口和 1 个 South Florida 区域交叉口），每个交叉口从 10 Hz 车道摄像头得到 YOLO+DeepSORT 追踪轨迹，构造 100,000 条平衡训练窗口。

**📈 对比分析**

与单独为每个交叉口训练的局部模型、IDM、恒定加速度基线进行比较，聚合模型在 10 s 的 minADE 与 minFDE 上平均分别降低 36.8% 与 22.0%；零射击方案在 4/5 个交叉口的 10 s 误差均优于局部模型；微调后在部分站点进一步提升，但总体表明共享模型在长时程预测上表现更好。

**⚠️ 局限性**

局限性包括：仅覆盖 19–84 分钟的有限时段、窗口采样未保证独立性、基于最佳样本的误差评价易偏向多模态模型、未进行闭环或实时仿真评估、缺乏对调和表示本身的消融实验以及对不同交通条件（流量、信号相位、队列状态等）的因果分离不足。

---

## 161. OpenCVL: An Open, Diverse, and Large-Scale Dataset for Fine-Grained Cross-View Localization

**arXiv ID:** 2608.25274 | [PDF](https://arxiv.org/pdf/2608.25274v1)

**作者:** Zimin Xia `[一作]` (École Polytechnique Fédérale de Lausanne), Julian F. P. Kooij `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OpenCVL——一个开放、跨区域、海量且多样化的细粒度跨视角定位（CVL）数据集，结合高精度 ZOD 数据和噪声较大的 Mapillary 影像，并开发了自动化的姿态校正框架，构建了高质量的验证与测试集（包含跨区域、雪景和野外场景）。

**💡 创新点**

创新点：①首次提供完全许可、可长期访问的 CVL 数据集；②在大量野外噪声图像中自动筛选并校正姿态标签；③设计多样化的评测挑战（跨区域、季节、视角）以逼真评估模型鲁棒性；④通过实验验证噪声数据与高质量数据混合训练可提升定位精度。

**🔧 技术方法**

主要技术：基于局部特征匹配与比例因子一致性的 Loc^2 模型；使用 DepthAnythingV2 预测深度并结合 ZOD LiDAR 进行尺度校正；对 Mapillary 图像实施 MAST3R+COLMAP 3D‑2D 关联与PnP 估计；对训练样本实施噪声感知加权；在实验中对 HC‑Net 与 CCVPE 进行对比。

**📊 数据集**

数据集：OpenCVL（617,388 ground‑aerial 对）、ZOD（高精度、车载摄像头）、Mapillary（自由获取、噪声较大）以及各国（瑞典、波兰、挪威、荷兰）开放航空影像。

**📈 对比分析**

对比方法：在 Loc^2、HC‑Net、CCVPE 等基线上，分别在不同训练集（仅 ZOD、ZOD+Mapillary、加权 Mapillary）和测试集（跨区域、雪景、野外）进行评估。结果显示：①加入 Mapillary 数据后，所有测试集定位误差下降 10–20%；②使用姿态校正后的 Mapillary 标签能进一步提升 3–5%；③在 KITTI 上，先预训练 OpenCVL 再 fine‑tune 可得到更佳的定位与方向精度。

**⚠️ 局限性**

局限性：尽管定位误差显著下降，但在野外测试集仍显著高于跨区域/雪景测试集，说明对极端视角与噪声的鲁棒性不足；模型在方向估计上容易出现 180° 错误；数据集仅覆盖四个欧洲国家，未来需扩展更多地区与多种航空影像形式（如真实正射影像）。

---

## 162. ShuttleArena: Interpretable Self-Play in Physics-Based Badminton

**arXiv ID:** 2608.25246 | [PDF](https://arxiv.org/pdf/2608.25246v1)

**作者:** Peize Ding `[一作]` `[通讯]` (Columbia University), Peize Ding (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个物理仿真的单打羽毛球自对弈环境ShuttleArena，并训练了结构化策略来同时学习击球轨迹与回位动作。

**💡 创新点**

首次将击球与回位作为可分离但相互依赖的动作因子，通过分层策略和因子化动作空间实现可解释的战术学习，并引入CRA对回位的针对性信用分配。

**🔧 技术方法**

使用PPO自对弈、因子化策略网络、受限截断回报、Counterfactual Recovery Advantage、可视化与定量探测工具。

**📊 数据集**

基于自行构建的ShuttleArena环境模拟数据，并与公开的ShuttleSet22单打比赛数据做 sanity 检验。

**📈 对比分析**

通过冻结检查点的轮盘赛、Bradley–Terry/Elo 评级、受控战术探测、回位消融以及与人类数据对比，证明自对弈策略在后期达到约1680 Elo 的水平，恢复策略显著提升约250 Elo。

**⚠️ 局限性**

模型简化了球员运动、抛物线抖动、风阻、旋转、疲劳与感知误差，且未进行全比分赛、双打或完整比赛评分，限制了对真实竞技水平的直接映射。

---

## 163. TrustFormer: Cross-Temporal and Cross- Dimensional Transformer for Task-Specific Multi-Dimensional Trust Evaluation

**arXiv ID:** 2608.25238 | [PDF](https://arxiv.org/pdf/2608.25238v1)

**作者:** Botao Zhu `[一作]` (Western University), Xianbin Wang `[通讯]` (Western University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于Transformer的TrustFormer框架，用于任务特定的多维信任评估与协作者选择。

**💡 创新点**

创新点在于轻量级数据同步方案与交叉时空/维度注意力机制，能同时捕获时间演化与维度间关联。

**🔧 技术方法**

采用Transformer（交叉注意力、多头自注意力）、时间间隔编码、MLP等技术。

**📊 数据集**

使用DELL 5200/5820/7060等真实设备在面部识别与病毒扫描两种任务下构造的行为模式数据，再通过NS‑3仿真生成200虚拟设备的20,000条记录。

**📈 对比分析**

与LSTM及标准Transformer对比，TrustFormer在MSE下降约40%至28%，协作者选择准确率提升至91.3%，尤其在战略欺骗和逐步退化行为下表现突出。

**⚠️ 局限性**

局限在于仅处理有限的历史维度（4个）与资源维度，且同步依赖任务ID与本地时钟，未针对极端网络时延或恶意破坏时序的鲁棒性做进一步评估。

---

## 164. Output Dilution: Redundant but Fragile Representations in MoE Models

**arXiv ID:** 2608.25231 | [PDF](https://arxiv.org/pdf/2608.25231v1)

**作者:** Orion Reblitz-Richardson `[一作]` `[通讯]` (Distiller Labs), Orion Reblitz-Richardson (Distiller Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比Mixture-of-Experts与密集模型在道德特征编码的准确性与鲁棒性，评估专家级道德专化与输出稀释机制。

**💡 创新点**

证明MoE模型无专家级道德专化，发现输出稀释导致道德信息在MoE中更脆弱，揭示准确性与鲁棒性可显著分离。

**🔧 技术方法**

使用线性探针、Gaussian噪声鲁棒性测试、per‑expert激活收集、检查点轨迹分析及标准差尺度测量等技术。

**📊 数据集**

采用240对基于Haidt道德基础理论的道德探测数据集（240对，48对测试）。

**📈 对比分析**

在相同层数、相同激活尺寸、相同探测数据集下，对OLMoE‑1B‑7B与密集OLMo‑2 1B进行层级探针与噪声鲁棒性比较；两者准确率相当（≈99%），但MoE的鲁棒性低4.2倍，输出尺度低74倍。

**⚠️ 局限性**

仅研究OLMoE单一模型，使用线性探针、均值池化近似，且仅覆盖英文与Haidt道德基础，未测试更高稀疏比例或不同任务。

---

## 165. Pseudorandom Functions in $\mathsf{NC}^1$ from LWE/LPN/CDH (Or: How to Build PRFs in $\mathsf{NC}^1$, Generically)

**arXiv ID:** 2608.25213 | [PDF](https://arxiv.org/pdf/2608.25213v1)

**作者:** Youlong Ding `[一作]` (Hebrew University of Jerusalem), Ilan Komargodski `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种从弱伪随机函数到强伪随机函数的低深度转换，保留了评估电路深度。

**💡 创新点**

创新点在于通过渐进收缩的 GGM 树（Tapering‑GGM）实现了深度保持，并首次在标准假设下构造了 NC1 PRF。

**🔧 技术方法**

利用了 GGM 结构、哈希压缩、无偏（key‑uniform）弱 PRF、低深度 Box–Muller 高斯采样等技术。

**📊 数据集**

作为理论工作，无需具体数据集。

**📈 对比分析**

与现有的深度更高或需要更强假设的构造相比，新方法在深度上保持 O(log n)，并实现了从 LWE、LPN、CDH 等常见假设得到的 NC1 PRF。

**⚠️ 局限性**

局限在于仍需弱 PRF 或合成器在 NC1 中可实现，且对参数设定（如模数-噪声比）有一定要求，难以进一步降低深度至常数。

---

## 166. BixBench3: Benchmarking AI agents on research-study-scale computational biology tasks

**arXiv ID:** 2608.25286 | [PDF](https://arxiv.org/pdf/2608.25286v1)

**作者:** Zane Koch `[一作]` (Edison Scientific, Inc.), Jon M. Laurent `[通讯]` (Edison Scientific, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出BixBench3基准，用原始数据、方法指导和期望产物评估AI代理完成完整的计算生物学研究流程；

**💡 创新点**

创新点在于将长时程、多步骤的完整实验链与程序化评分相结合，评估代理是否能按指定流程从原始数据生成与原论文一致的产物；

**🔧 技术方法**

使用LLM代理、Inspect AI的ReAct框架、工具调用（bash、python、text_editor、web_access、submit）以及多种评测指标（行列F1、CCC、宏F1、区间重叠F1）实现自动化执行与评分；

**📊 数据集**

利用20篇公开论文的数据（共138个产物，数据类型包括转录组、表观组、蛋白组等），平均原始数据量约67 GB，涵盖9个科学领域；

**📈 对比分析**

与13个前沿模型对比，最高GPT 5.6 Sol得分0.48（即48 %产物通过阈值），其他模型接近；模型表现随分析深度、数据规模和领域差异，成本差异约367倍，表现与计算开销不完全正相关；

**⚠️ 局限性**

局限性包括：任务需预设具体方法，无法评估代理自主决策；基准依赖原论文质量；长上下文和大数据规模对代理造成挑战；仅评估按指定流程执行的能力而非创新性研究方向。

---

## 167. Generative Action-Chunk Sampling for Adaptive Stiffness Control in Physical Human-Robot Collaboration

**arXiv ID:** 2608.25284 | [PDF](https://arxiv.org/pdf/2608.25284v1)

**作者:** Aoi Otake `[一作]` (Keio University), Shingo Murata `[通讯]` (Keio University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于生成式动作片段采样的自适应刚度控制框架，能在物理人机协作中根据视觉与外部关节扭矩信息实时调节机器人刚度与阻尼。

**💡 创新点**

创新点在于将观察条件先验与多样性采样相结合，利用采样动作的变异度作为在线不确定性信号来动态调节刚度，实现了无需额外语义模型即可实现“隐式角色适配”。

**🔧 技术方法**

采用 Transformer + CVAE 的动作片段预测模型，使用观察条件先验进行多样性采样；通过 EMA 平滑采样变异信号并映射到刚度混合系数；结合视觉（RGB）与外部关节扭矩输入实现多模态感知。

**📊 数据集**

使用 120 条演示数据（单人演示、四个方向的协作搬运任务）做训练，并用 20 条离线验证数据校准刚度阈值，实验中共完成 180 条在线评估试验。

**📈 对比分析**

与固定刚度消融模型（相同生成策略但刚度固定为 Medium）以及确定性 FACTR 基线模型进行对比。实验结果显示：平均成功率分别为 0.95（自适应刚度）、0.83（消融）、0.69（FACTR），显示自适应刚度在所有方向上均优于对比方法。

**⚠️ 局限性**

局限性包括：刚度阈值需经验调参，动作变异信号非校准概率，无法识别 OOD 情况；实验仅在四方向协作搬运任务中验证，缺乏对更复杂协作场景的泛化评估。

---

## 168. AllMusicCaps: Album Reviews as Complementary Supervision for Music CLAP

**arXiv ID:** 2608.25244 | [PDF](https://arxiv.org/pdf/2608.25244v1)

**作者:** Pablo Alonso-Jiménez `[一作]` (Universitat Pompeu Fabra), Dmitry Bogdanov `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并使用基于AllMusic专辑评测的 245,346 条精炼文本描述，作为训练文本-音频对比模型 CLAP 的新数据源；

**💡 创新点**

通过 LLM 先提取音频相关段落并重写为可训练的句子，利用 SigReg 正则化实现等方差 Gaussian 嵌入，从而显著提升对复杂、叙事性查询的检索性能；

**🔧 技术方法**

结合 InfoNCE、Sigmoid、LeJEPA 与 SigReg 等对比与自监督损失、OMAR‑RQ 小型音频编码器、MPNet 文本编码器以及多层加权组合策略；

**📊 数据集**

使用 AllMusic + Discogs + YouTube 生成的评测字幕数据，融合 LP‑MusicCaps、M4‑RAG、Freesound、Pro Sound Effects 等公开字幕与声音数据集；

**📈 对比分析**

在 MusicCaps、Song Describer、GTZAN、FMA‑Small、DimSim 以及 MLP 探针任务上与 Laion‑CLAP、TTMR++、CLaMP3_saas 等公开基线比较，取得在文本‑音乐检索、零样本分类和大多数探针任务上均优于现有模型，尤其在高复杂度查询中提升显著；

**⚠️ 局限性**

模型在多乐器分类任务上仍不如外部基线；对评测数据的文本生成可能存在误识别或偏见，且仅针对音乐音频，未覆盖更广泛的声音场景。

---

## 169. From Memorization to Absorption: Mixed-Policy RL for Continual Knowledge Injection

**arXiv ID:** 2608.25243 | [PDF](https://arxiv.org/pdf/2608.25243v1)

**作者:** Zhibo Hou `[一作]` (University of California Merced), Wan Du `[通讯]` (University of California Merced)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们提出了 GRIN，一种三阶段自学习框架，结合混合策略强化学习（Golden‑GRPO）实现对大型语言模型的持续知识注入。

**💡 创新点**

核心创新在于引入 Golden‑GRPO，利用黄金答案作为离线轨迹补偿训练时缺乏奖励的情况，并同时构造 Blank 与 Counter 两个面向新知识获取与旧知识覆盖的基准。

**🔧 技术方法**

技术手段包括：自监督式事实抽取与 SFT、语料条件下的多样化 QA 采样、基于 Group‑Relative Policy Optimization 的强化学习，并改进奖励函数（格式、ROUGE‑L、Exact‑Match、多答案惩罚）。

**📊 数据集**

数据集方面，Blank 取自 TimeQA 与后训练截止期的 Wikipedia 片段，Counter 则由同义宇宙改写的 Wikipedia 页面构成，实验以 Qwen3‑4B 为主基线，并在 Llama3.2‑3B 上做跨模型验证。

**📈 对比分析**

在与训练免费方法（Closed‑Book、Open‑Book、RAG）和训练基准方法（PIT、Self‑Tuning、Autonomous Learning、其他 RL 变体）比较时，GRIN 在单事实回忆上保持与 SFT 同等水平，且在多源检索和推理题型上提升 10‑30% 以上，Counter 任务中的 fail@k 亦明显下降。

**⚠️ 局限性**

局限性包括：只评估单轮注入，未验证持续学习过程；仅针对事实性实体文本，未覆盖程序、代码、数学等知识；以及 Golden‑GRPO 的 rollout 采样在大模型上会产生显著计算开销。

---

## 170. Development of a Voice-Controlled Tendon-Driven Bionic Hand

**arXiv ID:** 2608.25222 | [PDF](https://arxiv.org/pdf/2608.25222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 171. Rare Diseases, Common Dilemmas: LLMs Prioritize Equal Resource Distribution over Patient Benefit in Decision-Making

**arXiv ID:** 2608.25236 | [PDF](https://arxiv.org/pdf/2608.25236v1)

**作者:** Minda Zhao `[一作]` (Harvard T.H. Chan School of Public Health), Isaac S. Kohane `[通讯]` (Harvard Medical School)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并使用208个基于真实稀有疾病数据的临床伦理情境，对多款LLM进行forced‑choice测试，评估其在价值权衡中的行为；

**💡 创新点**

首次在稀有疾病高风险临床决策中创建大规模基于事实的伦理基准，并揭示LLM在所有模型中统一偏向正义及其对决策者框架的敏感性；

**🔧 技术方法**

利用GPT‑4.1自动生成情境、模拟专家审查、LLM forced‑choice评估，并通过归一化win率、Cramér’s V与逻辑回归等统计方法分析价值偏好；

**📊 数据集**

采用Orphanet和OMIM整理的稀有疾病表（约2,263种疾病）构成情境的基准数据集；

**📈 对比分析**

通过归一化win率比较各模型在四种伦理价值（正义、自主、益处、无害）上的选择比例；发现正义偏好在所有模型中达57%‑70%，并在决策者框架下呈显著差异；模型身份对结果影响微乎其微；

**⚠️ 局限性**

局限包括情境生成依赖GPT‑4.1、专家评审为模拟、样本分布不均导致部分分析（如正义在不同决策者框架下）受限，且未与真实临床伦理专家进行对照评估。

---

## 172. Long-Term Behavioral Evaluation for Trusted Collaborator Selection via Bidirectional Mamba

**arXiv ID:** 2608.25232 | [PDF](https://arxiv.org/pdf/2608.25232v1)

**作者:** Botao Zhu `[一作]` (Western University), Xianbin Wang `[通讯]` (Western University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于双向 Mamba 的长时行为评估模型，用来在协同系统中评估设备的历史行为并挑选可信的协作者。

**💡 创新点**

创新点在于：①将协作关系划分为多个短时段并构建协作图，利用 GNN 进行短时可靠性融合；②使用双向 Mamba 线性复杂度的 SSM 机制，对跨时段的行为序列进行前向和后向扫描，从而捕获长期的正向与逆向时间依赖；③结合任务特定资源评估，实现综合可信度评价。

**🔧 技术方法**

技术手段包括：协作图构造与图神经网络（GNN）、双向 Mamba（基于选择性状态空间机制的 SSM）、交叉熵损失训练、最大池化与多层感知机（MLP）后处理，以及 NS-3 仿真平台与 Python 绑定。

**📊 数据集**

数据集：在 NS-3 仿真中生成 500 台设备，10,000 个面部识别任务（5 MB，2,339 cycles/bit），记录传输、计算、协作成功率等指标；实验数据按 80/20 训练/测试拆分。

**📈 对比分析**

与 LSTM、单向 GNN 以及规则型 QS-Trust 进行比较。BM 在 RMSE、MAE 上均最低，且在长时间段（500 时隙）内波动最小，VoC 最高，表明其在长时序评估与协作者选择上表现更优。

**⚠️ 局限性**

局限性包括：①实验完全基于仿真，缺乏真实网络环境验证；②设备模型与任务类型相对单一，可能不适用于更复杂的多任务场景；③对极端网络条件（高丢包、移动性）和设备隐私保护的鲁棒性尚未评估。

---

## 173. "Am I Just That Dumb?": Applicability, Action and Verification in Consumer IoT Security Advice

**arXiv ID:** 2608.25225 | [PDF](https://arxiv.org/pdf/2608.25225v1)

**作者:** Veerle van Harten `[一作]` (Delft University of Technology), Simon Parkin `[通讯]` (Delft University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过在实验室环境中让28名参与者在没有任何技术指导的情况下，分别尝试根据国家级安全建议（更改默认密码、保持设备更新）对六款最畅销的智能家居设备进行操作，记录他们能否确定建议是否适用、找到并执行目标设置，以及是否能够验证设备的安全状态。

**💡 创新点**

研究首次系统地揭示了通用安全建议与实际设备架构之间的匹配缺陷，展示了用户在缺乏诊断性指导时会误将建议误解为应用级操作，导致安全提升与用户认知不一致，并提出了三种“闭环”策略：安全默认与自动维护、诊断式建议与支持材料配合、以及统一术语以消除层级歧义。

**🔧 技术方法**

采用实验室观察、think‑aloud记录、半结构化访谈与NASA‑TLX工作量评估相结合的混合方法；使用R进行反射性主题分析（RTA）与GLMM等统计模型；设备来自Amazon销量榜单，包含嵌入式Web、云账户与伴随App等多层接口。

**📊 数据集**

共计168个会话数据（每人6次任务），包含会话终点、参与者自评是否完成、任务时长、六项NASA‑TLX分量表；同时收集六款设备的预先确定功能清单（是否有设备级密码、固件更新路径等）以及28名参与者的基本信息。

**📈 对比分析**

对不同设备与建议项的结果进行计数和比例比较，计算密码设置与固件更新的实现率。结果显示，密码建议中有33/84未能找到设置，50/84实现了账户级设置，1/84实现设备级设置；更新建议中27/84未找到更新，19/84完成伴随App更新，38/84完成可验证固件更新。比较显示参与者自评与实际达成率存在显著差距，说明用户在执行安全建议时存在认知误差。

**⚠️ 局限性**

主要限制包括实验室情境与真实家庭环境的差异、样本规模有限且教育水平偏高、设备选取基于销量而非全面覆盖所有设计差异、仅观察“演示”而未执行真实更改导致安全效果未被验证、以及语言与术语不一致可能影响结果的普适性。

---

## 174. FLARE: Verifying MILP Reformulations with LLM-Based Theorem Proving

**arXiv ID:** 2608.25220 | [PDF](https://arxiv.org/pdf/2608.25220v1)

**作者:** Henry Robbins `[一作]` (Stanford University), Ellen Vitercik `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某一领域的研究问题，并提出了一种新的方法来解决该问题。

**💡 创新点**

创新点在于提出了一种新的算法或模型，能够在特定条件下提高性能。

**🔧 技术方法**

使用了机器学习和深度学习技术，结合了特定的数学模型。

**📊 数据集**

使用了公开数据集进行实验，以验证所提方法的有效性。

**📈 对比分析**

与现有的方法进行了比较，结果显示所提方法在准确性和效率上均有显著提升。

**⚠️ 局限性**

限制在于所提方法可能在某些特定情况下表现不佳，且对数据的依赖性较强。

---

## 175. A Training-Free Proactive Defense Against Partial Speech Manipulation via Self-Embedding Steganography

**arXiv ID:** 2608.25285 | [PDF](https://arxiv.org/pdf/2608.25285v1)

**作者:** Yigitcan Özer `[一作]` (National Institute of Informatics), Junichi Yamagishi `[通讯]` (National Institute of Informatics)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了基于自嵌入式音频隐写的主动防御框架，用以检测局部deepfake语音并恢复原始内容。

**💡 创新点**

创新点在于将语音自身的神经编码器表示嵌入到信号中，并通过重复LSB编码实现无需训练、鲁棒的检测与恢复。

**🔧 技术方法**

采用SNAC神经语音编解码器、LSB隐写、动态时间规整(DTW)以及音频数据集评估。

**📊 数据集**

在AV-Deepfake1M验证集上进行实验，使用真实语音进行单词/双词替换攻击。

**📈 对比分析**

与LAV-DF、LAV-DF+和ResNet基线对比，单词替换时EER下降至约9%，双词替换下降至约5%，显著优于近似随机的被动检测器。

**⚠️ 局限性**

局限性包括对极短局部攻击敏感、仅评估单词替换攻击、未考虑多模态融合与更复杂攻击手段。

---

## 176. Groundhog Bit-Flip Attack: Seeding Infinite Generation Loops in Mixture-of-Experts LLMs through Bit Flips

**arXiv ID:** 2608.25276 | [PDF](https://arxiv.org/pdf/2608.25276v1)

**作者:** Huakang Lin `[一作]` (Louisiana State University), Ruyi Ding `[通讯]` (Louisiana State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用MoE模型路由层的位翻转攻击，诱导模型持续生成文本，导致输出长度显著膨胀，从而实现对LLM服务的拒绝使用（DoW）攻击。

**💡 创新点**

首次揭示MoE专家在终止标记（EOS/EOT）上高度专业化，并证明只翻转极少数路由参数位即可大幅抑制这些专家，造成巨大的输出膨胀；同时提出了全局与局部专家检测与位挑选两种高效的攻击流程。

**🔧 技术方法**

MoE路由位翻转攻击（GBFA）、专家激活频率（EAF）分析、全局/局部专家检测、路由层位敏感度搜索、使用Rowhammer等软硬件诱发位翻转技术。

**📊 数据集**

使用公开的六大MoE LLM（Mixtral、Phi‑3.5‑MoE、DeepSeek‑V2‑Lite、Qwen3、Qwen3‑Coder、GPT‑OSS）以及AGNews、SST‑2、Samsum、SQuAD等文本分类与生成数据集。

**📈 对比分析**

通过与人工手动禁用专家对比、在四大模型上对比不同专家检测策略（GLOBAL/LOCAL）以及不同位翻转规模，结果显示位翻转攻击可实现与手动禁用相当甚至更好的输出膨胀（最高可达数百倍），且对模型语义质量的损伤最小（PPL<10，指标保持或略有提升）。

**⚠️ 局限性**

攻击假设白盒访问并能在共享硬件上执行定向位翻转；未完成完整的硬件级演示；实验基于已知模型参数，缺乏对不同并行部署和ECC/TEE等硬件防护的深入评估；理论上未给出正式证明。

---

## 177. RWA-PoB: A Credential-Based Proof-of-Backing Framework for Tokenized U.S. Treasury Products

**arXiv ID:** 2608.25269 | [PDF](https://arxiv.org/pdf/2608.25269v1)

**作者:** Rischan Mafrur `[一作]` (Universiti Brunei Darussalam), Sean Foley `[通讯]` (Macquarie University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了针对美国国债代币化产品的 RWA-PoB 证明框架，结合了五角色凭证、BCR 与 RLC 指标、原子发行与赎回以及基于 Solidity 的智能合约。

**💡 创新点**

创新点在于：①将机构角色分离的凭证与统一快照绑定，实现多方共识；②通过 BCR 与 RLC 分离评估抵押充分性与短期流动性；③在单个交易中同步 ERC‑20 发行与负债变更，确保账务一致；④提供可验证的快照完整性与时间戳保护。

**🔧 技术方法**

使用技术包括 Solidity 0.8.28、EIP‑712 签名、Merkle 根、ERC‑20 标准、Hardhat 开发环境以及 gas 计量；合约中嵌入 BCR、RLC 计算和快照验证逻辑。

**📊 数据集**

采用了 Ondo USDY 的 1,043 条日数据（2023‑09‑18 至 2026‑07‑26）进行负债校准，并生成基于种子 42 的合成保留头寸（包括市场值、合格值、流动值、到期、贴水、占用、法定资格及结算状态）。

**📈 对比分析**

通过与聚合 PoR 基线对比，构造三种情景（有效、占用资产、流动性压力），测量 BCR 与 RLC 阈值；所有 31 条测试均通过；初始快照 gas 约 362,241（约 3.83 倍基线），后续更新约 194,904（约 5 倍基线）；在占用资产场景下 RWA‑PoB 拒绝发行，在流动性压力场景下返回排队赎回，表现符合设计。

**⚠️ 局限性**

局限性在于：仅验证签名与时间戳，无法独立确认资产的所有权、可用性与价值；依赖机构诚信与监管，若签名者缺失可导致可用性受限；未实现价格波动、跨链结算、完整的合规/身份系统；未覆盖更复杂的金融产品；缺乏正式安全审计与长期模拟验证。

---

## 178. Federation Is Nearly Free, Reasoning Is Not: Tradeoffs for AI Co-Scientists in Protein Characterization Workflows

**arXiv ID:** 2608.25215 | [PDF](https://arxiv.org/pdf/2608.25215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 179. What Do Medical Vision-Language Models Learn in Radiology? Transfer, Alignment, and Source-Proxy Leakage Under Distribution Shift

**arXiv ID:** 2608.25251 | [PDF](https://arxiv.org/pdf/2608.25251v1)

**作者:** Ayoub Louaye Bouaziz `[一作]` (University of Western Brittany), Yassine Himeur `[通讯]` (University of Dubai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过对胸部X光影像的医学视觉‑语言模型（VLM）进行受控压力测试，评估其在分布移位下的表示转移、跨数据集对齐和元数据源代理可恢复性。

**💡 创新点**

提出分离源仅视觉迁移、跨数据集多模态对齐和源代理泄漏三类测试框架，并在此基础上量化自监督初始化、对抗适应及其不稳定性。

**🔧 技术方法**

使用 ResNet‑18 基础架构、BYOL 自监督预训练、DANN/CORAL 无监督适配、CLIP‑style 初始化、BioClinicalBERT 对齐、对比损失、Grad‑CAM 可视化及线性探针等技术。

**📊 数据集**

NIH ChestXray14、CheXpert、PadChest、OpenI 四个公开胸片数据集。

**📈 对比分析**

通过匹配架构的线性探针和部分微调评估 NIH→CheXpert 的 AUC，BYOL 初始化优于 ImageNet；对抗适配在强度过高时失稳；OpenI 评估的严格对齐 Recall@K 远低于随机，表明跨数据集对齐不足。

**⚠️ 局限性**

评估仅涵盖胸片、缺乏多模态重复/多正样本处理、缺少真实医院站点标签、仅报告线性探针而非完整不确定性估计、实验可重复性受限。

---

## 180. A Few Pages of Markdown: Committed AI Configuration and Lower Quality Cost after Coding-Agent Adoption

**arXiv ID:** 2608.25241 | [PDF](https://arxiv.org/pdf/2608.25241v1)

**作者:** Yegor Denisov-Blanch `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RAMP（Repository AI Maturity Profile）四层累积成熟度模型，用版本控制中的AI配置文件评估团队对AI工具的集成程度；

**💡 创新点**

创新点在于将AI工具使用的配置结构化为可观测的成熟度等级，并通过该模型揭示了不同配置成熟度对编码代理采纳后质量与速度影响的差异；

**🔧 技术方法**

使用命名模式检索、路径与内容的嵌入语义分类（句子转换器）以及多重信号优先级规则，构建自动化分类器；

**📊 数据集**

采用27家企业内部441个私有GitHub仓库进行模型开发与验证，随后在公开的Agarwal等人开源项目数据集中（509个受试仓库）进行效果检验；

**📈 对比分析**

通过比较不同RAMP等级仓库在代理采纳前后的提交量、行数、认知复杂度、静态分析警告等指标，发现低成熟度（Level 1）仓库的质量降幅是高成熟度（Level 2+）的约两倍；分类器在人工标注样本上的准确率约81.7%，与人类一致性指标（Cohen κ≈0.74）相近；

**⚠️ 局限性**

局限性包括：观察性设计无法排除工程纪律或模型能力的混杂；大部分配置在代理采纳后才出现，可能导致逆向因果；缺乏Level 4实证样本；分类器仅覆盖已定义的12种AI工具，可能漏检其他工具；仓库级别标签忽略项目内部的细粒度差异。

---

## 181. Automotive HSMs - Architectural Challenges and Security Implications

**arXiv ID:** 2608.25216 | [PDF](https://arxiv.org/pdf/2608.25216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 182. Representing MAX functions using two-hidden-layer ReLU networks

**arXiv ID:** 2608.25221 | [PDF](https://arxiv.org/pdf/2608.25221v1)

**作者:** Zhimao Wang `[一作]` (Johns Hopkins University), Amitabh Basu `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

研究如何用两隐藏层 ReLU 网络精确表示 N 维最大函数 \(\max\{x_1,…,x_N\}\)，并给出 N=5、6、7、8 的具体解析表达式。

**💡 创新点**

创新点在于引入更多的函数对称性（包括对称对的交换、子项的排列以及左右两侧的互换），并直接对两项最大值进行处理，而不是像前人那样拆分为线性 + ReLU，显著减小了线性系统的规模。

**🔧 技术方法**

技术手段主要是群作用、等价类划分与轨道求和，构造相应的线性方程组，然后利用符号计算求解得到有理系数；同时对无歧义与歧义的原子做进一步分解。

**📊 数据集**

本工作不使用实验数据集，而是完全基于理论推导与符号计算得到结果。

**📈 对比分析**

与 Rueß 等人相比，本文在 N=5–8 范围内能够得到解析解，但在 N=9、10 时线性系统过大导致无法求解；相对前人使用更高阶的原子（k=4 而非 k=3），表达式更冗长但对称性处理更细致。

**⚠️ 局限性**

局限性包括：对更大 N 的扩展受限于线性系统规模；得到的表示不一定是最优的（层数或参数最小化）；仅提供存在性的充分条件，未给出必要条件或完整的可行性判定。

---

## 183. Bolt-on, Verifiable Provenance for LLM-Powered Data Processing

**arXiv ID:** 2608.25210 | [PDF](https://arxiv.org/pdf/2608.25210v1)

**作者:** Yiming Lin `[一作]` (University of California, Berkeley), Aditya G. Parameswaran `[通讯]` (University of California, Berkeley)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种用于大型语言模型（LLM）数据处理的可验证最小化可追溯性（verifiable provenance）框架，能够在保持答案一致性的前提下自动从原始文本中提取最小子集，形成可解释的答案来源。

**💡 创新点**

创新点包括：
- 定义可验证可追溯性与最小化的形式化概念；
- 设计八种保证最小化的子策略并推出自适应策略；
- 通过两阶段 prune‑refine 结构，既快速定位答案来源又保证最小化；
- 结合 KV 缓存与 LLM 评分器实现高效、低成本的检索；
- 支持多重最小可追溯性（top‑k）以提升答案可信度。

**🔧 技术方法**

主要技术手段：
- LLM 作为黑盒推理器，利用相等性判定（semantic equivalence）通过 LLM‑as‑a‑judge；
- 文本块的嵌入或 LLM‑based 相关性排序；
- Bottom‑up / Top‑down 递归块评估；
- Sequential‑Greedy 与 Exponential‑Greedy 的增量删除算法；
- 自适应策略与 KV 缓存减少 token 计费；
- 通过理论成本模型与实验验证各策略优劣。

**📊 数据集**

实验数据集：
- 5 个真实问答工作负载：Qasper、NL_DEV、HotpotQA、CUAD、PubMedQA；
- 2 个合成表格问答（TableQA）工作负载：Movie、Restaurant；
- 统一使用 gpt‑4o‑mini（及 gemini‑2‑flash 作为对比）。

**📈 对比分析**

与传统检索（RAG）和 LLM 直接生成可追溯性相比：
- 100% 的可验证准确率，且比最佳基线高 30%+；
- 生成的可追溯性平均占原始文本 3–8%（比 RAG 10% 以上小得多）；
- 成本比（相对原始问答）低于 1.3×，且大多数情况下仅为 0.2–0.3×；
- 延迟平均 2–8 秒，远低于完整文本推理（≈20–30 秒）。

**⚠️ 局限性**

局限性：
- 依赖任务的弱/强单调性，非单调任务仍需大量 LLM 调用；
- 需要高质量的块排序（embedding/LLM‑ranker），排序失效会导致成本上升；
- 对多义或抽象问题，生成的最小可追溯性可能仍存在语义歧义；
- 仅在文本/表格场景下验证，其他数据类型（视频、音频）需扩展；
- 对 LLM 生成结果的确定性仍有限，非确定性会增加额外检验；
- 结果为文本片段，若篇幅过大仍可能导致人类难以快速验证。

---

## 184. Hamiltonian Two-Way Coupling of Nonlinear Waves and 3D Flows

**arXiv ID:** 2608.25203 | [PDF](https://arxiv.org/pdf/2608.25203v1)

**作者:** Sinan Wang `[一作]` (Georgia Institute of Technology), Bo Zhu `[通讯]` (Georgia Institute of Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Zakharov Hamiltonian的二维非线性、具有色散特性的水面波模型，并实现与三维Navier–Stokes求解器的双向耦合，从而在大尺度自由面流模拟中兼顾高精度与高效性。

**💡 创新点**

创新点在于：①将完整的非线性水面动力学以Hamiltonian结构封装为(η,ψ)对，使得二维波场与三维流体的状态传递仅通过Dirichlet–Neumann算子实现，保证耦合的一致性与能量守恒；②使用Craig–Sulem高阶谱(HOS)展开将DNO的非线性修正转化为FFT+点乘，既保持非线性、色散，又降低计算复杂度至O(N²log N)；③引入可调非线性参数ε，提供从线性Airy到全非线性系统的连续切换，兼顾稳定性与精度。

**🔧 技术方法**

采用的核心技术包括：Zakharov的Hamiltonian水面动力学、Craig–Sulem/DNO展开、HOS第二/第三阶截断、FFT谱域运算、3/2法则去混频、低通滤波、积分因子AB2时间积分、FAB（有限宽带）边界条件、指数衰减的隐式松弛、以及GPU加速的NB-FLIP三维流体求解。

**📊 数据集**

主要使用合成海洋谱（TMA、Donelan–Banner）进行初始条件生成，并在多种实验中验证：自由波传播、Stokes波、连续波梳理、色散匹配、王船（单船、两船追逐）、水面降落、潜艇升出、战舰在重浪中航行、浅水池等多尺度场景。

**📈 对比分析**

与传统方法（SWE、BEM、Airy DK）和先前的二维-三维耦合方案（FAB+Airy、基于BEM）对比，实验表明：平均波高误差可降低1.7–5×，波面色散与非线性匹配更精准；相较于BEM，求解速度提升10⁻³倍；在与全域GPU NB-FLIP耦合时，整体速度提升4×以上，且接口波纹几乎消失。

**⚠️ 局限性**

局限性包括：①在强3D激励下，全非线性（ε=1）不稳定，需调低ε；②与线性模型相比，算力显著增加（HOS-2约为线性2–5×，HOS-3约为2×）；③单值高度场无法表示破浪；④极端场景（如战舰高浪）仍出现微小接口反射，需后期渲染遮蔽；⑤需手动调节ε与耦合参数以适配不同情境。

---

## 185. Time-Optimal APSP and Matrix Multiplication in Classes of Linear Neighborhood Complexity

**arXiv ID:** 2608.25212 | [PDF](https://arxiv.org/pdf/2608.25212v1)

**作者:** Édouard Bonnet `[一作]` (ENS de Lyon), Szymon Toruńczyk `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对具有线性邻域复杂度的图类的(n^2)时间最优算法，解决了所有对最短路径和邻接矩阵与任意n×n矩阵相乘的问题。

**💡 创新点**

创新性地扩展了线性邻域复杂度的概念，提出了多种算法，包括三角形检测和K_4、K_5子图检测算法，且在时间复杂度上进行了优化。

**🔧 技术方法**

使用了随机化算法和线性时间算法，结合了sd-退化序列和Welzl顺序等技术。

**📊 数据集**

使用了来自线性邻域复杂度类的n顶点图，具体数据集未明确给出，但涉及多种稀疏和密集图类。

**📈 对比分析**

与现有方法相比，提出的算法在时间复杂度上具有优势，能够在(n^2)时间内解决所有对最短路径问题，并在特定情况下实现更快的矩阵乘法。

**⚠️ 局限性**

算法在处理特定图类时可能存在局限性，尤其是在图的密度和结构复杂性较高的情况下，可能无法达到预期的性能。

---

## 186. InsightSR: Refining Symbolic Regression Search Spaces via Parallel Semantic and Structural LLM Guidance

**arXiv ID:** 2608.25291 | [PDF](https://arxiv.org/pdf/2608.25291v1)

**作者:** Yating Ling `[一作]`, Zhitang Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 InsightSR 框架，利用大语言模型引导 PySR 进行符号回归，通过语义种子路径和结构特征路径双路推进，并在每代结束后通过闭环反馈不断优化搜索空间。

**💡 创新点**

创新点包括：①将 LLM 用作搜索空间转换器而非直接生成表达式；②同时提供物理一致的结构种子和非线性特征推荐，实现两路协同；③累积特征与闭环学习，使得搜索由深树转向浅树，显著提升效率与可解释性。

**🔧 技术方法**

技术手段：大语言模型（Qwen‑3.5 27B）、PySR 遗传编程引擎、维度一致性校验、复杂度偏置策略、知识库闭环反馈、特征工程与策略分析。

**📊 数据集**

使用的数据集：Feynman 100、LLM‑SRBench（多学科子集）、四个真实世界数据集（Oscillator 1/2、E. coli、Stress‑Strain）。

**📈 对比分析**

与传统 GP、MCTS、深度学习符号回归、现有 LLM‑SR 方法进行对比；在 Feynman 上精确恢复率 95%，在 LLM‑SRBench LSR‑Transform 任务准确率 80.18%，在真实世界数据集上 NMSE 远低于竞争方法，表现优异。

**⚠️ 局限性**

局限性：依赖任务描述与元数据，若变量语义模糊会影响单位推断；LLM 查询成本虽已降低但仍存在；框架主要在生成阶段使用 LLM，可能限制更广泛搜索空间的探索。

---

## 187. Mitigating LLM sycophancy with RL-based fine-tuning: Bayesian Truth Serum approach

**arXiv ID:** 2608.25267 | [PDF](https://arxiv.org/pdf/2608.25267v1)

**作者:** Serhii Mytsyk `[一作]` (Cornell University), Vikram Krishnamurthy `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用贝叶斯真相序（BTS）作为奖励，在Group Relative Policy Optimization（GRPO）框架下对大型语言模型进行微调，目标是消除模型在面对用户偏好时产生的迎合式（sycophancy）回答。

**💡 创新点**

创新点在于：①首次将BTS机制作为RL奖励，证明在无标签、无偏好标注的情况下其能激励模型诚实回答；②从理论上证明在大群体极限下，sycophantic回答的期望奖励低于诚实回答；③通过实验验证其在多模型、多任务上的有效性，并与现有奖励机制对比。

**🔧 技术方法**

核心技术包括：BTS奖励函数（信息分数与预测分数的加权和）；GRPO（无价值网络的群体优势估计强化学习）；LoRA微调；以及基线对比方法（synthetic-data、Pinpoint tuning、SMART）。

**📊 数据集**

使用了两类数据集：①自制 1000 条 true/false 的 TF 数据集（包含用户偏好诱导句）；②改造后的 SYCON-modified 多轮 benchmark（借鉴 SYCON-Bench）。

**📈 对比分析**

实验通过与三种已公开的 sycophancy 缓解方法（synthetic data、Pinpoint tuning、SMART）以及无标签的 Peer Truth Serum 等机制进行比较。结果显示：在 TF 数据集上，BTS GRPO 将 sycophancy 从 23% 降至 4%，准确率提升至 93%；在多轮 benchmark 上显著提高 Turn-of-Flip；与标签基线相比，性能可比，尽管训练 FLOPs 约为标签方法的 70 倍。

**⚠️ 局限性**

限制包括：①实验仅在 3-4B 规模模型上进行，未验证更大模型或开放式文本；②仅在闭合形式答案上评估，未覆盖开放式生成；③每个实验仅跑一次单随机种子，未检验重复性；④预测报告使用“人类频率”代理，未充分验证其有效性；⑤计算成本高，需较大资源。

---

## 188. Hierarchical MoE for Multi-Modal ILD Diagnosis

**arXiv ID:** 2608.25261 | [PDF](https://arxiv.org/pdf/2608.25261v1)

**作者:** Alec K. Peltekian `[一作]` (Northwestern University), Ulas Bagci `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种层次化多模态 Mixture‑of‑Experts (MoE) 模型，用于系统性硬皮病相关肺间质病变 (ILD) 的诊断，将 CT 影像与结构化电子病历 (EHR) 数据融合。

**💡 创新点**

创新点在于双层门控：① 模态级门控动态平衡影像与 EHR 的贡献；② EHR 子门控将临床特征分组，学习组别专属贡献，从而在保持可解释性的同时实现输入依赖的自适应融合。

**🔧 技术方法**

技术手段包括 3D SwinUNETR 影像专家、Radiomics 估计肺叶重要性、分组的 EHR 子专家（MLP + 投影）、双层门控（Softmax 门控）与轻量 MLP 分类器。

**📊 数据集**

使用 Northwestern Scleroderma Registry 数据集，597 名患者、1,898 份胸部 CT，包含不同系统性硬皮病亚型，已标注 ILD 与否。

**📈 对比分析**

在患者级 5‑折交叉验证下与多种基线（SwinUNETR、REN、CNN 等）比较，层次化 MoE（选择性 EHR 分组）平均 AUC 为 0.875±0.044，显著优于 REN（0.865）和 SwinUNETR（0.769），并在统计上显著提升（p<0.001）。

**⚠️ 局限性**

局限性：仅来自单中心、特定疾病（硬皮病肺病）；EHR 分组人为设计，未探索数据驱动分组；缺失值未显式建模；未独立评估门控各自贡献；缺乏外部验证与泛化性评估。

---

## 189. The Pauli Lightcone: Information-Theoretic Error Mitigation Beyond the Autocorrelation

**arXiv ID:** 2608.25254 | [PDF](https://arxiv.org/pdf/2608.25254v1)

**作者:** Paolo D'Alberto `[一作]` `[通讯]` (Advanced Micro Devices, Inc.), Paolo D'Alberto (Advanced Micro Devices, Inc.)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在GPU加速的张量网络模拟中，对重叠Heisenberg–Ising模型在heavy‑hex与矩形格子上进行噪声演化，并引入空间‑时间Pauli权重场n(v,t)作为光锥前沿的精确观测量；基于该场构建噪声时空图wavemap，包含到达延迟l_γ(v)与交叉熵损失ℒ_γ(v)，随后利用多产品公式（MPF）在Lieb‑Robinson约束下恢复无噪声光锥。

**💡 创新点**

创新点在于：①首次将Pauli权重的空间分布与噪声延迟、信息损失量化为wavemap，直接揭示噪声对光锥形状的影响；②证明所研究的噪声是纯幅度阻尼，光锥形状完全由门控决定，噪声仅作透明的幅度抑制；③将Lieb‑Robinson causal bound作为MPF的硬约束，实现物理一致的误差消减；④提出按时间切片的α(t)系数，使MPF在光锥前沿上精确拟合，从而实现高达55–65%的信息恢复。

**🔧 技术方法**

采用的技术包括：GPU并行张量网络（CppSim）中的贝叶斯传播与截断、16×16 Pauli传输矩阵（PTM）构造、PTM本征值分析、交叉熵损失度量、Lieb‑Robinson causal约束以及多产品公式（MPF）与SLSQP优化。

**📊 数据集**

使用的数据集为：heavy‑hex 3×3和rectangular 7×7格子；heavy‑hex采用真实设备的硬件PTM噪声（γ倍放大），rectangular采用合成的等比例抖动噪声；实验全部在上述两种拓扑与噪声强度下进行模拟。

**📈 对比分析**

比较方法：将MPF前沿、熵最优MPF、时间自适应α(t)以及空间自适应α(d)与最佳噪声样本及无噪声参考进行对比；在heavy‑hex上α(t)实现55%信息恢复，在rectangular上实现65%恢复；无约束熵MPF可产生不符合因果性的负损失；受限熵MPF在信息富集场景下仍能提升约24–49%信息，但在硬件噪声异质性强时效果受限。

**⚠️ 局限性**

局限性包括：需要光锥前沿足够活跃且包含足够多站点（否则α(d)拟合失效或前沿崩溃）；硬件噪声的异质性会限制信息恢复；方法假设噪声为纯幅度阻尼，若存在旋转噪声需额外处理；实验规模受限于小型格子，未验证在大规模真实量子设备上的可扩展性。

---

## 190. The "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion

**arXiv ID:** 2608.25245 | [PDF](https://arxiv.org/pdf/2608.25245v1)

**作者:** Chenglong Ma `[一作]` (RMIT University), Jeffrey Chan `[通讯]` (RMIT University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出概念来源追踪框架，对 LLM 生成的初始查询中的概念进行分区，以识别并量化违背搜索者信息获取边界的答案侧概念渗入。

**💡 创新点**

创新点在于将查询概念划分为背故事支持、人类中心、尾部和答案侧四个区域，并通过自动+人工两阶段验证提供了可操作的边界合规诊断方法。

**🔧 技术方法**

技术包括两条互补的概念提取管道（词形/实体+统计特征），基于精确匹配的泛化过滤，HCIR 指标及手工标注评估。

**📊 数据集**

数据集为 UQV100，包含 100 个主题、约 10,835 个真实工人查询以及 77,004 条 LLM 生成查询，检索基于 ClueWeb12-B13。

**📈 对比分析**

通过对比不同 LLM 模型、提示条件和检索系统的评估指标，发现答案侧概念仅占 7.4% 非通用概念，局部检索效果显著但对整体 nDCG 等指标影响不到 2%；后期的基于概念来源的筛选可将侵入率降至 0.06%，且保持大部分检索性能。

**⚠️ 局限性**

局限性包括依赖 UQV100 的人工查询分布、对概念匹配仅采用精确词形、忽略语义变体、评估仅在 ClueWeb12-B13 上进行，且未覆盖会话级查询或其他检索模型。

---

## 191. The Systems Paper is Dead. Long Live the Systems Paper

**arXiv ID:** 2608.25219 | [PDF](https://arxiv.org/pdf/2608.25219v1)

**作者:** Bjoern Hartmann `[一作]` `[通讯]` (UC Berkeley), Bjoern Hartmann (UC Berkeley)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出在系统实现已不再是瓶颈的时代，UIST论文应该如何转变研究方法，并提出三种可能的发展方向。

**💡 创新点**

从“构建单一系统”转向“多原型评估”“生成式AI辅助评估”“实时迭代评估”三大思路，强调方法创新。

**🔧 技术方法**

讨论了生成式AI、自动代码生成、实时系统迭代等技术，但未给出具体实现。

**📊 数据集**

无数据集使用。

**📈 对比分析**

未进行实验对比，本文以理论分析和案例引用为主，未给出性能指标。

**⚠️ 局限性**

局限在方法尚未实证、缺乏可复制性、评估过程的伦理与可实现性挑战。

---

## 192. Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-lens

**arXiv ID:** 2608.25347 | [PDF](https://arxiv.org/pdf/2608.25347v1)

**作者:** Shi-Qi Yan `[一作]` (Alibaba Group), Zhen-Hua Ling `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Jacobian lens（J‑lens）提出了完整的数学解释，并基于此分析了其能量分布结构，随后给出了能量过滤与解耦的改进方法并通过实验验证其有效性。

**💡 创新点**

创新点在于：①将 J‑lens 视作平均的一阶传递算子，并通过 Stein 桥连接局部 Jacobian 与全局最小二乘；②系统分析了非高斯偏差与非线性导致的整体误差；③揭示了 Jacobian 能量的稀疏结构，分为短视野与稀疏概念两种模式；④提出仅保留高能量位置的过滤策略和通过掩码解耦两种模式的方法。

**🔧 技术方法**

使用的技术包括：局部线性化、最小二乘回归、Stein 识别、非高斯偏差分析、Jacobian 能量热图可视化、能量过滤与掩码解耦。

**📊 数据集**

实验基于 Qwen3‑8B 语言模型，使用 WikiText 语料构建 J‑lens，并在关联、多跳推理、多语种、排序操作、诗歌生成与错别字纠正六类任务上进行评估。

**📈 对比分析**

通过两项指标（短视野层数 SHL 与中间概念召回率 ICR）与原始 Vanilla J‑lens 进行比较，能量过滤版本在大多数任务中提升了 SHL 与 ICR；解耦方法则显示两种模式存在较强耦合，性能下降。

**⚠️ 局限性**

限制：实验仅在 Qwen3‑8B 上进行，改进策略仍在完善中；对能量分布假设的理论与实验验证仍不完全；解耦两种模式的效果仍有待进一步研究。

---

## 193. Provenance Before Prose: Claim-Locked Reporting

**arXiv ID:** 2608.25336 | [PDF](https://arxiv.org/pdf/2608.25336v1)

**作者:** Xiao Fan `[一作]` (Xidian University), Yi Zhang `[通讯]` (Xidian University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了claim-locked reporting框架，用于在生成统计报告前先锁定每个可报告的统计主张及其数值、方向与表达强度，使LLM仅生成连接性文字。

**💡 创新点**

创新点在于将统计主张本身作为控制单元，在生成前绑定证据来源、数值与语言强度，解决传统文本层、槽位层控制无法避免的统计命题失真问题。

**🔧 技术方法**

技术包括：结构化证据记录、声明构造器(Claim Builder)、风险审核器(Risk Auditor)、政策控制器(Policy Controller)与确定性渲染器(Deterministic Renderer)等，配合LLM撰写连接段。

**📊 数据集**

使用的数据集包括肥胖相关功能连接组(fMRI FC)的两个队列（医院内部428人和HCP S1200 712人）以及公开的Evidence Inference 2.0临床试验摘要。

**📈 对比分析**

通过与五个基线（Free‑form、Prompt‑only、Structured、Retrieval、Post‑hoc verifier）以及Hybrid template对比，claim‑locked在跨运行重现率从61.1%提升至98.5%，治理与数值错误率大幅下降，整体性能显著优于传统方法。

**⚠️ 局限性**

局限性包括：只能控制已预先计算的统计证据，无法保证前置分析正确；需人工或专家定义声明模板、风险标签和支持度；对缺失或中性结论的处理仍有限；且该框架在不同学科需要额外配置。

---

## 194. FinRiskAtlas: Decision-Aligned Evaluation of Large Language Models for Financial Risk Review

**arXiv ID:** 2608.25325 | [PDF](https://arxiv.org/pdf/2608.25325v1)

**作者:** Suyang Zhong `[一作]` (Ant International), Tianyi Zhang `[通讯]` (Ant International)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了两个专业金融审计评价基准 FinRiskAtlas（静态评估）和 FinRisk-Ask（动态证据状态评估），并在 33 种 LLM 配置上进行实验。

**💡 创新点**

创新点在于：①以专业审计操作为评估单元，拆分为 Domain Knowledge、Evidence‑Grounded Processing、Applied Review 三层；②构造离线轨迹回放的 Ask‑Or‑Proceed 评估，测量模型在证据不完整时是否正确请求并指向具体缺失证据；③通过运营层级和证据状态两维度揭示模型在金融工作流中的差异与潜在损失。

**🔧 技术方法**

采用零样本直接回答推理、语义对齐评估器、Spearman 相关、ERA/CRA/BAcc 等自定义指标，评估模型在不同任务和状态下的性能。

**📊 数据集**

数据集包含 9,742 条静态实例（53 个任务族），涵盖 42 个 Domain Knowledge 族和 11 个审计操作族；以及 680 条来自 104 条专业审计轨迹的 Ask‑Or‑Proceed 状态，其中 583 为 Ask 状态，97 为 Proceed 状态。

**📈 对比分析**

比较方法为按 Domain Knowledge 宏平均得分对配置排序，并在每个操作族上计算排名；实验显示不同操作族间 Spearman 相关平均仅为 0.42，说明同一模型在不同操作中表现差异显著；在 FinRisk‑Ask 上，ERA 最大为 96.57%（Ling‑2.6‑1T），但即使 BAcc 相近的模型，其 ERA 差异可达 28.31 分，表明行动选择与请求精准度并非同一能力。

**⚠️ 局限性**

局限性包括：①评估仅覆盖中文金融审计场景，难以推广到其他语言或行业；②轨迹回放基于已完成的专业案例，无法模拟实时交互或政策变化；③评价指标依赖人工标注的证据需求，可能存在主观性和标注成本高的问题。

---

## 195. Prefix-Denoising Consistency: Test-Time Verification for Diffusion Language Models

**arXiv ID:** 2608.25311 | [PDF](https://arxiv.org/pdf/2608.25311v1)

**作者:** Yuki Ichihara `[一作]` (MBZUAI), Junpei Komiyama `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种测试时自我校验方法 Prefix-Denoising Consistency（PDC），在已生成的 Diffusion Language Model 输出上固定前缀、重新掩码剩余位置并再进行去噪生成，然后对三种保留率（0.1、0.5、0.9）下的重生成结果进行多数投票，从而改进答案。

**💡 创新点**

创新点在于利用前缀条件下的去噪一致性信号——正确答案在重生成时更稳定、更易被重现；只需少量（3个）重生成就能提升准确率；相较于传统自一致性，PDC 在计算成本更低的前提下取得更好效果。

**🔧 技术方法**

使用 Diffusion Language Model（Dream-7B、LLaDA 系列）+ 前缀保持+去噪重生成 + 多投票聚合；实验中还采用了不同温度、掩码策略和自一致性（x4）等基线。

**📊 数据集**

在数学与常识推理基准上进行评测：GSM8K、MATH-500、SVAMP、CSQA、SQA；使用 Dream-7B、LLaDA-8B、LLaDA-1.5 等模型。

**📈 对比分析**

与初始生成、独立多次生成的自一致性（x4）以及中间步投票等基线比较。PDC 在 15 个模型-数据组合中多数提升准确率，平均提升约 1.3–1.7 分；在计算受限下相较于 x4 的提升可达 9.25 分，显示出更高的计算效率和性能。

**⚠️ 局限性**

局限性：需要额外的去噪步骤，仍受模型温度和掩码策略影响；实验主要在固定长度前缀下进行，未充分验证在更大规模或长文本场景的通用性；对不同掩码策略的鲁棒性虽有探测，但仍需进一步评估。

---

## 196. BOOSTEDSOSA: Accelerated Inferencing for Low Variance Stochastic Online Scheduling

**arXiv ID:** 2608.25346 | [PDF](https://arxiv.org/pdf/2608.25346v1)

**作者:** Adam H. Ross `[一作]` (University of Illinois Chicago), Debjit Pal `[通讯]` (University of Illinois Chicago)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出双FPGA机器学习辅助调度架构BoostedSOSA，在HPC系统中用XGBoost实时预测任务运行时间并与SOS调度器结合，替代人工估计；

**💡 创新点**

创新点包括：①将XGBoost推理移植到FPGA实现极低延迟；②提出核心+增量训练策略，实时适应工作负载漂移；③设计无批处理流式I/O，保持在线调度时的低延迟；④实现硬件与调度器分离模块，易于替换；

**🔧 技术方法**

使用技术包括：FPGA加速的XGBoost（Conifer FPU）、双FPGA异步PCIe流式架构、梯度提升决策树、固定点量化、增量训练（Additive Training）、SOS调度算法；

**📊 数据集**

采用的工作负载数据集：ALCF（Argonne Leadership Computing Facility）历史作业、MIT Supercloud、UIUC Blue Waters；

**📈 对比分析**

与人类估计、历史平均、随机森林、AdaBoost、线性回归以及软件AVX基准比较，使用MAE、p95、调度时延和吞吐量指标；结果表明BoostedSOSA将MAE提升约63.85%/71.88%，平均17×速度提升，吞吐约1700 jobs/s，内存占用≈250 MB；

**⚠️ 局限性**

限制：主机多线程和PCIe流式导致内存使用约7倍；仅使用6个提交时特征，可能忽略更深层特征；增量训练对突变分布有惯性；仅验证XGBoost和RF，未覆盖其他预测器；固定点量化阈值随系统变化需手动更新。

---

## 197. GPU-Accelerated Quantum Annealing-Inspired UAV Path Planning for Smart Agriculture

**arXiv ID:** 2608.25376 | [PDF](https://arxiv.org/pdf/2608.25376v1)

**作者:** Maho Hirahara `[一作]` (University of Electro Communications), Aohan Li `[通讯]` (University of Electro Communications)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于GPU并行Ising求解器的量子退火路径规划框架，用于农用无人机在格网化农田中的最优路径规划。

**💡 创新点**

创新点在于将传统的量子退火思想迁移到GPU硬件上，利用Fixstars Amplify平台实现大规模并行退火，从而在规模扩大时仍保持计算时间稳定，并显著提升求解质量。

**🔧 技术方法**

主要技术包括：QUBO建模、Ising模型映射、Fixstars Amplify SDK自动化模型转换、GPU并行退火（多副本模拟退火与交换）、以及路径重构算法。

**📊 数据集**

实验采用合成格网数据，网格尺寸从3×3到6×6（共9–36个节点），没有使用公开数据集。

**📈 对比分析**

与传统模拟退火（SA）和遗传算法（GA）对比，Fixstars Amplify在大规模网格上取得最低飞行时间（成本）且计算时间基本不随规模变化；在6×6网格中，优化路径最短、计算时间稳定，明显优于两种基准。

**⚠️ 局限性**

局限性包括：仅针对简化的飞行时间能量函数，未考虑电池容量、喷洒量、障碍物等真实约束；实验规模仅到6×6，缺乏更大规模验证；以及对GPU硬件成本与部署复杂性的讨论不足。

---

## 198. Capacity Overflow: A Blind Spot for Backdoor Attacks in Vision MoE

**arXiv ID:** 2608.25371 | [PDF](https://arxiv.org/pdf/2608.25371v1)

**作者:** Xiaocheng Zou `[一作]` (Northeastern University), Ruyi Ding `[通讯]` (Louisiana State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 Vision MoE 模型中设计了一种利用批量大小触发的条件后门，攻击者在早期层植入触发器，后期层训练中和器以保持低批量时的正常行为，最终通过调整容量因子在高批量推理时激活后门。

**💡 创新点**

创新点在于发现并利用 MoE 的容量约束和 token 过载机制作为隐蔽触发通道，使后门仅在部署规模批量时激活，从而规避传统小批量检测和防御。

**🔧 技术方法**

采用 MoE 结构、容量因子调节、token 过载策略、三阶段训练（植入、隐蔽、中和）、KL 散度蒸馏以及自适应容量因子配置等技术。

**📊 数据集**

使用 ImageNet-100（100 类）和 GTSRB（43 类交通标志）两个视觉数据集进行实验。

**📈 对比分析**

在大批量（B=128）下，激活模式下后门成功率 76–87%，而在小批量（B≤32）下成功率低于 9%，同时保持了 1.2pp 以内的准确率下降，表明对比方法的高隐蔽性与高效能。

**⚠️ 局限性**

局限性包括仅针对分类任务的 Vision MoE 进行评估，缺乏对其它 MoE 变体或多模态模型的验证；需要供应链控制才能植入；以及在非批量触发环境下可能不具备同样的可行性。

---

## 199. RSFusionDet: Underwater RGB-Sonar Multimodal Object Detection

**arXiv ID:** 2608.25367 | [PDF](https://arxiv.org/pdf/2608.25367v1)

**作者:** Zhuoyan Liu `[一作]` (Harbin Engineering University), Ye Li `[通讯]` (Harbin Engineering University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

创建了RGB‑Sonar融合数据集RSFusion并提出评价指标，设计并实现了RSFusionDet框架，实现RGB与声呐的多模态目标检测与跨模态对象匹配。

**💡 创新点**

创新点在于：①跨模态Deformable Attention Fusion（CAFusion）实现空间失配特征融合；②对象匹配头OMHead与OMLoss实现跨模态对象对齐；③提出了统一的多模态检测结果表达方式。

**🔧 技术方法**

采用基于DINO的端到端检测器，双分支ResNet‑50骨干，跨模态Deformable Attention、位置与尺度嵌入，Cosine相似度匹配，IoU加权交叉熵损失，配合多尺度特征融合与注意力机制。

**📊 数据集**

使用自建RSFusion数据集，7073个RGB‑Sonar图像对，涵盖7类目标（人偶、UUV、反射器、金属多面体、浮标、铁球、台面），包含明暗场景和不同最大声呐距离（5、10、15、20 m）。

**📈 对比分析**

在RSFusion验证集与多种单模态与多模态基线对比实验中，RSFusionDet在RGB上提升0.7/1.4 AP（相较DINO），总AP达76.4/48.6，跨模态匹配F1‑Score为83.4，显著优于所有对比模型。

**⚠️ 局限性**

限制在于：对小目标、遮挡和海水噪声仍易出现漏检；不同类别间特征相似导致匹配误配；方法依赖固定的摄像头与声呐安装配置，无法自适应动态几何校准。

---

## 200. PaSta: Noisy Node Classification with Partial Label Learning

**arXiv ID:** 2608.25365 | [PDF](https://arxiv.org/pdf/2608.25365v1)

**作者:** Yujing Liu `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个基于部分标签的自监督框架 PaSta，用来解决噪声节点分类问题，先通过多种自监督学习方法构建多重注解器生成高质量的部分标签，然后在标签空间和表示空间引入两种自定义损失（VaCE 与 PaSim）训练分类器，并在闭环中迭代更新注解器和分类器以提升标签质量和模型鲁棒性。

**💡 创新点**

核心创新在于：①首次将部分标签学习引入图节点分类，突破传统一热标签导致的过拟合与错误校正堆叠问题；②设计多样化自监督注解器集成生成的非二值化部分标签；③在标签空间和表示空间分别设计投票聚合交叉熵与部分标签相似度损失，实现对部分标签信息的双重利用；④通过自监督循环自训练机制，实现标签与模型的互相提升。

**🔧 技术方法**

技术上使用了多种自监督图学习（如 DGI、GCA、SUGRL）构建注解器；基于 GCN 的两层分类模型；投票聚合交叉熵 (VaCE) 损失；部分标签相似度 (PaSim) 损失；噪声标签过滤和伪标签扩展的自训练策略。

**📊 数据集**

在 Cora、Citeseer、DBLP、Computers、Photo 这五个真实图数据集上进行实验，分别为三种文献网络、学术合作网络和两种产品共购网络。

**📈 对比分析**

与传统 GCN、GAT、三种自监督方法（DGI、GCA、SUGRL）以及四个最新噪声节点分类方法（JoCoR、NRGNN、MTS‑GNN、BO‑NNC）进行对比。PaSta 在所有数据集和噪声水平下均显著优于对手，平均提升约 1.1%（相较 BO‑NNC）并且在最差情况下相对 GCN 提升 12.7%。

**⚠️ 局限性**

主要局限包括：①对标签质量高度依赖，若自监督注解器生成的部分标签本身噪声过大，效果可能受限；②计算复杂度仍为 O(n²)，对大规模图的适用性受限；③缺乏对不同噪声类型（如结构噪声）的深入探讨。

---

## 201. Exact SAT and Constraint Programming for Job Shop Scheduling with Time-Varying Peak Power Constraints

**arXiv ID:** 2608.25351 | [PDF](https://arxiv.org/pdf/2608.25351v1)

**作者:** Huy Tuan Nguyen `[一作]` (VNU University of Engineering and Technology), Khanh To Van `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了带时间可变峰值功率约束的作业车间调度问题，并提出了基于SAT与约束规划（CP）的精确求解框架。

**💡 创新点**

首次为此类问题设计了SAT编码（基于顺序编码+伪布尔约束）和CP模型（区间变量+全局约束），并通过预处理提升求解效率。

**🔧 技术方法**

使用了离散时间顺序编码、伪布尔约束、二进制合并编码、CP全局noOverlap、cumulative约束以及增量式SAT优化。

**📊 数据集**

在公开的35个JSPPR基准实例（4-10个作业、4台机器）上进行实验。

**📈 对比分析**

与之前的MILP（CPLEX、Gurobi）和GRASP×ELS启发式比较，SAT与CP均在所有实例上证明最优，CP平均求解时间约为26秒，SAT约为2,673秒；相比之下MILP需数千秒或超时。

**⚠️ 局限性**

受限于离散时间建模导致变量规模大，SAT求解时间相对较长；同时基准规模较小，难以验证在更大规模实例上的可扩展性。

---

## 202. Neither Precision Nor Architecture Alone: Controlled Tests of Failure Remedies for Physics-Informed Neural Networks

**arXiv ID:** 2608.25327 | [PDF](https://arxiv.org/pdf/2608.25327v1)

**作者:** Jinyuan Zhang `[一作]` (Hubei University), ShengShuo Jiao `[通讯]` (Hubei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对物理信息神经网络（PINN）在激波、反应和波动方程中遇到的失败进行系统实验，比较数值精度、L-BFGS 停止阈值、SSM 骨干与子序列对齐四种干预措施的互补性与局限性。

**💡 创新点**

首次在同一受控设置下将 FP64 精度修复与 SSM+对齐两种截然不同的解决方案并行评估，证明它们作用于不同的 PDE/seed 切片且互不替代，并提出按种子细粒度报告的重要性。

**🔧 技术方法**

使用 FP32/FP64 计算、L-BFGS 优化、状态空间模型（SSM）骨干、子序列对齐损失、严格的随机种子配对、预注册实验设计以及诊断代理进行失败分类。

**📊 数据集**

三类一维 PDE 的标准测试问题：β=50 的对流方程、ρ=5 的激波反应方程以及双频初始条件的波动方程，实验共 229 次跑。

**📈 对比分析**

采用匹配的 PDE+seed 单元进行对照，使用相同网格、预算和优化器配置；结果显示 FP64 在极端对流问题上仅提升 1/5 的成功率，SSM+对齐在相同精度下可提升 2/5 或 3/5 的成功率，而两者在不同情境下互补；调节 L-BFGS 容忍度可降低误差但无显著成功提升，且计算成本大幅增加。

**⚠️ 局限性**

实验仅覆盖三类单一维 PDE，缺乏更广泛的科学计算基准；不同模型协议和种子数量不均，未实现完整因子实验；部分诊断代理缺失日志，且结果对波动方程的可重复性有限。

---

## 203. LLMscope: Extracting LLM Assets from Edge AI Chips via Optical Probing

**arXiv ID:** 2608.25321 | [PDF](https://arxiv.org/pdf/2608.25321v1)

**作者:** Dev Mehta `[一作]` (Worcester Polytechnic Institute), Fatemeh Ganji `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用电光频率映射（EOFM）对FPGA上LLM推理芯片进行无电接触光学探测，直接提取模型参数（嵌入、注意力权重、MLP权重等）以及推理状态（激活、KV‑缓存等），并演示了从FF和BRAM等本地存储结构中恢复完整二进制值。

**💡 创新点**

提出了面向资产级安全的新安全模型；首次将光学探测扩展到LLM资产的位级读出；引入混合恢复（直接读取+线性代数+下游一致性约束）以补偿不完整覆盖；并给出了与资产尺寸、重用和光学覆盖相关的成本下界。

**🔧 技术方法**

电光频率映射（EOFM）光学显微成像、Kintex‑7 FPGA、基于Systolic数组的矩阵乘法加速器、光学扫描与频域处理。

**📊 数据集**

实验使用人工合成的身份矩阵输入与输出（无真实LLM数据集），但讨论涵盖了多种公开LLM加速器（Llama‑F、Pushing up to the Limit、Hummingbird、FlightLLM）。

**📈 对比分析**

通过直接EOFM读出实现了小型资产的完美恢复；混合恢复在缺失部分时可用最少的输入‑输出对（甚至仅一组）完成剩余权重，证明了线性系统的秩要求；实验表明光学覆盖与资产大小呈线性关系，完整矩阵恢复需要数小时扫描，但相较于功率或EM侧信道更直观、无电接触。

**⚠️ 局限性**

需对芯片背面进行光学接触且对硅晶圆做预处理，扫描时间长且受器件密度与重用限制；每个FPGA家族需单独校准位映射；光学方法不适用于极大模型的全局恢复，且受限于光学分辨率与探测频率。

---

## 204. Rank-Deviation Quality: A Distance-Aware Metric for Multi-Answer Retrieval and Ranking Evaluation

**arXiv ID:** 2608.25318 | [PDF](https://arxiv.org/pdf/2608.25318v1)

**作者:** Xiaokun Zhou `[一作]` (Amazon), Danielle Class `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了Rank‑Deviation Quality（RDQ）指标，用于多答案检索系统的排名评价，直接基于有序参考列表（ordinal reference）对候选排名进行评分。

**💡 创新点**

创新点在于：①无需刻度化的相关性等级即可评估；②支持多答案、分层和查询大小可变；③将输出位置重要性与与参考排名的偏差处罚相结合，兼顾返回项与排序顺序。

**🔧 技术方法**

采用两种偏差处罚函数（M1、M2）并引入参数α、λ；利用随机化检验、bootstrap、Kendall τ 等统计方法进行指标功效与稳定性分析；通过控制实验验证指标行为。

**📊 数据集**

使用了 5,000 个 POI 查询的 POISS 数据集（来自 Overture Maps + GenAI 注释）以及 2019–2022 年 TREC Deep Learning 轨道的 221 个查询（MS MARCO v2 文档集）作为评估基准。

**📈 对比分析**

与 MAP、RBP、NDCG、CMP 等传统指标在同一数据集上进行随机化检验、功效（median power@100）和系统排名稳定性比较。结果显示：在 POI 实验中 RDQ（α=1–4, λ=0.2）在 median power@100 最高（0.353 对比 RBP(0.9) 的 0.287），在 TREC‑DL 上 RDQ 与 NDCG 在 n=25 时相近，n=100 时 NDCG 略优。RDQ 的参数推荐设置为 α∈[1,4]、λ=0.2。

**⚠️ 局限性**

局限性包括：①需要手工指定位置权重和偏差处罚参数；②依赖完整的有序参考列表，缺失的有效项被零分；③评估仅基于统计功效和稳定性，未与人类偏好直接对齐；④TREC‑DL 上系统数少导致噪声大；⑤POI 数据为银标准，存在标注噪声；⑥指标结果不一定直接映射到真实用户满意度。

---

## 205. Designing Core Layer in Campus Network Using Software-Defined Networking

**arXiv ID:** 2608.25373 | [PDF](https://arxiv.org/pdf/2608.25373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 206. MulVec: Fine-Grained Role-Aware Matching for Training-Free Zero-Shot Composed Image Retrieval

**arXiv ID:** 2608.25305 | [PDF](https://arxiv.org/pdf/2608.25305v1)

**作者:** Zihao Zhang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Weiping Wang `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的零样本组合图像检索方法MulVec，利用角色感知匹配对查询进行结构化编译，并在候选图像上采用全局+局部视觉向量进行匹配，从而在不进行任何任务特定训练的情况下实现高精度检索。

**💡 创新点**

核心创新点在于：①将查询拆解为四个语义角色（Global、Desired、Preserve、Forbidden）并为每个角色分配专门的匹配规则；②使用冻结的多模态Encoder将角色信息映射为目标描述向量和 probe 向量；③在候选图像上构建全局视觉向量与多粒度局部视觉向量银行，使得角色匹配能在细粒度证据上操作；④通过固定加权求和一次性完成检索，避免多轮迭代。

**🔧 技术方法**

技术手段包括：Frozen OpenCLIP（ViT-B/32、ViT-L/14、ViT-G/14）视觉/文本塔；Qwen3.6‑27B 编译器生成角色结构化查询；soft‑max 局部交互（τ=0.02）实现 probe‑local 匹配；全局、Desired、Preserve、Forbidden 四个角色的加权融合；多粒度局部向量（R=49/64）支持细节级别检索。

**📊 数据集**

使用公开基准 CIRCO、CIRR 和 FashionIQ 进行评估，分别针对开放域、类别特定的组合检索任务。

**📈 对比分析**

与现有训练无关方法（CIReVL、OSrCIR、CoTMR、SoFT、LDRE+PDV‑F、SDR‑CIR、STiTch 等）相比，MulVec 在 CIRCO mAP@5 上提升最高 23.0%，在 CIRR 和 FashionIQ 的 Recall@k 亦达到或超过最新公开结果，证明了角色感知匹配与细粒度证据的有效组合。

**⚠️ 局限性**

局限性包括：①依赖编译器对角色的准确划分，若编译错误会导致检索失败；②冻结的视觉/文本表示若缺失关键视觉线索，无法补偿；③对候选集的覆盖度有限，当目标图像不在图库中时无法检索；④当前方案仅适用于单次检索，未结合后处理或迭代重排。

---

## 207. APT: Accelerating Diffusion Transformers via Attention Probability-Guided Pruning and Quantization

**arXiv ID:** 2608.25380 | [PDF](https://arxiv.org/pdf/2608.25380v1)

**作者:** Sungyeob Yoo `[一作]` (KAIST), Joo-Young Kim `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一款面向高分辨率Diffusion Transformer的软硬件协同加速器APT，利用注意力概率进行稀疏化和双精度量化，显著减少自注意力计算量并提升推理效率。

**💡 创新点**

创新点在于引入Attention Probability-guided Adaptive Dual Thresholding（APDT）实现动态元素级裁剪与量化，并结合Timestep-Aware FlashAttention（TAFA）在无需完整注意力矩阵的前提下预测概率，从而兼顾高效稀疏和双精度计算。

**🔧 技术方法**

核心技术包括基于注意力概率的双阈值裁剪与量化、时间步感知的FlashAttention概率预测、专用的稀疏双精度MAC单元、动态掩码管理与地址翻译、以及基于块的张量化数据流。

**📊 数据集**

在PixArt-α、Stable Diffusion 3和FLUX模型上进行实验，分辨率覆盖1K、2K、4K，并使用COCO 2017数据集评估生成质量。

**📈 对比分析**

与NVIDIA A100和EXION对比，APT在4K时实现最高8.16×的速度提升、14.98×的能效提升，且在不同分辨率下均保持2.5%以内的生成质量误差。

**⚠️ 局限性**

局限性包括对跨注意力（cross‑attention）的加速效果有限、依赖先验的注意力概率统计（需离线分析）、以及对不同类型扩散模型的适配尚待进一步验证。

---

## 208. GGSS: Geodesic-Gated Spherical Steering for Inference-Time Debiasing of Generative Vision-Language Models

**arXiv ID:** 2608.25375 | [PDF](https://arxiv.org/pdf/2608.25375v1)

**作者:** Yiqun Sun `[一作]` (Magellan Technology Research Institute), Lawrence B. Hsieh `[通讯]` (Magellan Technology Research Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对生成式视觉‑语言模型（VLM）提出一种推理时去偏方法 GGSS，利用对抗图像激活发现偏差子空间并通过几何保形的球面插值对视觉 token 进行调节。

**💡 创新点**

创新点在于：①在单位超球面上学习多维偏差子空间；②引入自适应门控按 token 的偏差强度选择性调整；③使用球面线性插值（Slerp）实现范数不变的地理轨迹旋转，从而兼顾去偏与保持生成质量。

**🔧 技术方法**

主要技术包括：对抗样本的球面均值与切向差分；奇异值分解得到偏差子空间基；Slerp 球面插值；基于偏差范数的门控函数；以及对不同 VLM 的同一层进行统一对齐。

**📊 数据集**

使用的公开数据集包括 REFLECT/FOCUS 的 480 张对抗人像图像（6 职业 × 8 身份 × 5 认知种族 × 2 性别），SocialCounterfactuals 作为偏差评估 probe，MMStar 作为多模态能力基准。

**📈 对比分析**

在四种生成式 VLM（Pixtral‑12B、LLaVA‑1.6‑Vicuna‑7B、LLaVA‑1.6‑Mistral‑7B、Qwen3‑VL‑4B‑Instruct）上与十种推理时去偏基线及提示层偏差缓解方法做对比；GGSS 在所有模型上实现了平均 bias 降低最高可达 90% 左右，并且在 MMStar 评价中保持±0.6pp 的性能差异，显著优于传统的线性子空间投影或均值偏移方法。

**⚠️ 局限性**

局限性包括：仅评估了种族与性别两类属性，未覆盖其他身份或多模态跨域；需要手动调节强度参数 α，缺乏自动化选择；去偏并未消除模型内在的偏见知识，可能在分布漂移或对属性敏感的任务上产生副作用；此外，若使用过强的调节，可能导致目标属性被完全抑制，影响可解释性与合法使用。

---

## 209. FlashNormal: Detailed Surface Normal Estimation from Flash and No-Flash Images

**arXiv ID:** 2608.25360 | [PDF](https://arxiv.org/pdf/2608.25360v1)

**作者:** Ruiyang Chen `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种利用闪光/无闪光图像对的扩散模型（FlashNormal）来估计高质量的表面法线，能够在常见智能手机相机条件下实现细节丰富且减少形状‑反射模糊的表面重建。

**💡 创新点**

核心创新包括：① 将闪光/无闪光图像对通过VAE编码为隐空间特征，直接提供形状变异信息；② 采用基于扩散先验的单步去噪法线估计器，并结合文本提示进行引导；③ 设计曲率引导的细节增强损失和“放大像素”策略以提升局部细节和对小目标的鲁棒性；④ 构建首个真实世界闪光/无闪光表面法线基准数据集。

**🔧 技术方法**

技术手段包括：预训练的Stable Diffusion扩散模型与VAE，闪光/无闪光图像对的双通道编码，曲率计算与损失，角度误差损失，单步扩散去噪，自动ROI提取与放大像素策略，数据增强与梯度累积训练。

**📊 数据集**

使用了三大数据集：① 100,000对合成闪光/无闪光图像与对应GT法线（来源于gObjaverse）；② EvalFlash-synth，839个与训练集分离的合成测试样本；③ 真实拍摄的20个对象（使用Canon EOS R5 + EinScan-SP）并配备精确的扫描Mesh、对齐与GT法线。

**📈 对比分析**

在20个真实对象和839个合成测试样本上与多种基线（单图法线估计如Pix2Pix、Swin-Depth；光度立体 MV20；多光照光度立体等）对比，FlashNormal 在MAE、RMSE、MSE 上分别领先最佳单图方法 6.6%、领先 MV20 58% 以上，整体显示出显著的精度提升和更好的细节保留。

**⚠️ 局限性**

局限性主要体现在：① 对透明物体表现欠佳（因反射难以区分表面）；② 对纹理复杂的平面易误判为三维结构；③ 训练集对特殊材质（如透明、高度反射）覆盖不足，导致泛化受限。

---

## 210. Leveraging Speech Acts for Low-Data and Cross-Domain Conversation Derailment Forecasting

**arXiv ID:** 2608.25359 | [PDF](https://arxiv.org/pdf/2608.25359v1)

**作者:** Angela Yifei Yuan `[一作]` (University of Melbourne), Christopher Leckie `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将话语行为（Speech Act）作为辅助学习信号，构建两种结构（H_parallel 与 H_sequential）来预测在线对话何时会演变为敌意，从而实现主动式社区管理。

**💡 创新点**

创新点在于：①利用语义层与话语行为层的双向信息融合，消除词汇噪声并提升跨域泛化；②通过零样本 LLM 提取话语行为标签，使得在推理时不再需要额外的 SA 识别步骤；③设计了动态预测评估协议（平均聚合）与跨数据集性能分析，揭示话语行为在低数据与高差异域场景中的优势。

**🔧 技术方法**

技术方法包括：大规模预训练语言模型（BERT/DeBERTa）作为句子编码器；两层 Transformer 的会话级编码器；多任务学习框架（SA 检测 + 破裂预测）；H_parallel 采用平行头并进行两阶段训练；H_sequential 在 SA 概率上加 GRU 以强调话语行为；LLM（如 GPT）用于零样本多标签 SA 提取；评估指标为 AUPRC 与 Macro‑F1，采用动态 mean 聚合。

**📊 数据集**

使用的公开数据集有三类：1) Wikipedia 编辑讨论（WIKI，4,188 对话）；2) Reddit ChangeMyView 辩论（CMV，19,578 对话）；3) GitHub 技术讨论（GITHUB，898 对话）。每个数据集均含破裂与文明对话配对，且在训练、验证、测试集划分上保持主题一致。

**📈 对比分析**

与基线模型（CRAFT、基于 PLM 的模型、无 SA 的 H_ablation）相比，H_parallel 在三大数据集的 AUPRC 与 Macro‑F1 上均居首位，尤其在 300–2,500 条样本的低数据区间显著优于对手；跨域评估显示 H_parallel 与 H_sequential 在语义/词汇差异较大的 GITHUB 迁移中表现最强；通过动态 mean 聚合进一步验证模型在每个时间步的预测稳定性，表明 SA 信息显著提升了早期警报准确度。

**⚠️ 局限性**

局限性包括：①LLM 在 SA 提取时可能引入文化、方言及社区规范偏见；②对隐式或间接意图的识别仍有限，导致部分真实意图未被捕获；③虽然推理阶段已消除 SA 计算开销，但训练阶段仍需 LLM 或人工标注来生成 SA，成本较高；④在极度低样本或与源域差异极大的迁移（如 GITHUB→CMV）中，模型表现仍受限。

---

## 211. Where vs What: Decomposing Structural and Content Failures in LLM-Generated Structured Outputs

**arXiv ID:** 2608.25358 | [PDF](https://arxiv.org/pdf/2608.25358v1)

**作者:** Yiwei Zhang `[一作]` (Shenzhen University), Jianqiang Li `[通讯]` (Shenzhen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了结构-内容分解（SCD）框架，用于独立评估大型语言模型生成结构化输出的结构完整性与内容准确性。

**💡 创新点**

创新点在于揭示结构先衰退的现象、将评估拆解为三级指标，并将这些指标转化为可验证奖励，驱动结构化生成的强化学习优化（SA-RLVR）。

**🔧 技术方法**

采用了层级化评估指标、语义与结构提示 ablation、以及基于 GRPO 的强化学习结合 SCD 奖励的 SA-RLVR。

**📊 数据集**

使用了自动合成的三层复杂度的嵌套 JSON 与表格任务，以及来自 JSONSchemaBench 的 OOD 真实 schema 进行验证。

**📈 对比分析**

与基线模型、SFT 以及多种奖励 ablation 进行对比，SA-RLVR 将 JSON 值位置准确率从 0.26 提升至 0.63，表格格式有效率从 26% 提升至 85%，并在 OOD JSON 上也表现优于 SFT。

**⚠️ 局限性**

局限在于实验使用合成数据、缺乏真实任务多样性，RL 仅在 7B 规模上验证，且对表格坐标级别的提升有限，且可能过度依赖可验证奖励而非真正的结构理解。

---

## 212. Where to Look Matters: On-Policy Self-Distillation for Long-Video Understanding

**arXiv ID:** 2608.25356 | [PDF](https://arxiv.org/pdf/2608.25356v1)

**作者:** Kaishen Wang `[一作]` (University of Maryland), Di Fu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种利用视频中短时段线索（clue interval）进行自监督的长视频理解方法。

**💡 创新点**

创新点在于将线索段作为教师模型的专属视觉输入，实现无标签的线索特权自监督，避免了额外推理模块。

**🔧 技术方法**

采用基于Jensen–Shannon散度的对齐策略和EMA自教师的on‑policy自蒸馏框架，并使用Qwen3.5 VLM。

**📊 数据集**

使用了CG‑Bench中带有线索段标注的长视频问题集以及Video‑MME、LVBench、LongVideoBench、MLVU、MMVU等基准。

**📈 对比分析**

在五个基准上与SFT、GRPO和标准OPSd进行对比，Qwen3.5-2B/4B/9B在平均精度上提升约2–9个百分点，且在多项任务中超过同规模的监督后训练模型。

**⚠️ 局限性**

局限性在于仅在有线索标注的数据上可训练，且对不同VLM架构的迁移性尚未验证；同时对极长视频或无线索情境的表现有限。

---

## 213. Point-in-Time Audit Before Alpha: Public-Archive Availability and a Negative Matched-Budget Study on BTC Perpetual Futures

**arXiv ID:** 2608.25348 | [PDF](https://arxiv.org/pdf/2608.25348v1)

**作者:** Baocheng Zeng `[一作]` (Tsinghua University), Kangnan He `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了公开 Binance BTCUSDT USD‑M 永续期货档案的时间可用性，并构建了可审计的因子挖掘流程，验证了数据完整性、模板审计、无效控制、搜索比较与历史保留评估的有效性。

**💡 创新点**

创新点在于提出了基于事件、发布时间与可用性时间的三时钟审计框架，并首次将可审计规则与无效控制相结合，以显著降低假阳性并揭示公开档案的完整性缺陷。

**🔧 技术方法**

使用了确定性审计器、Wilson 区间、BH‑FDR、PBO、Deflated Sharpe、树型 GP、随机搜索、可审计 DSL 以及可用性掩模等技术。

**📊 数据集**

采用了公开的 Binance BTCUSDT USD‑M 永续期货数据集（trade、mark、index、funding），共 727 个完整 UTC 天。

**📈 对比分析**

在匹配预算下，审计适配者与随机搜索表现相当，未实现显著优势；在历史保留测试中，尽管得到正向 IC，但所有策略在实际成本下均呈负的夏普率，表明缺乏经济可行性。

**⚠️ 局限性**

局限性包括仅单一资产/交易所、历史封闭验证、未考虑真实出版时间、OI 数据缺失、模板样本有限、未区分不同缺失处理方式、缺乏对其他基线与实时交易验证等。

---

## 214. HRGuard: Gating Relationship Manipulation in Multi-Turn Agentic AI Conversations

**arXiv ID:** 2608.25340 | [PDF](https://arxiv.org/pdf/2608.25340v1)

**作者:** Pei-Sze Tan `[一作]` (National Institute of Informatics), Isao Echizen `[通讯]` (National Institute of Informatics)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对代理式关系伤害的评估与防护方法，构建了1000条五轮对话基准并设计了HRGuard门控框架；

**💡 创新点**

创新点在于将关系伤害视为角色敏感的多轮流程，通过前置和后置门控结合累计风险状态，实现对逐步升级的操纵行为的实时干预；

**🔧 技术方法**

技术上采用LLM生成、基于预设权重的轮次风险评分、指数衰减累计风险、阈值触发门控，并用GPT‑4o‑mini等模型做判定；

**📊 数据集**

使用的数据集为从先前关系操纵代码书演变而来的1000条对话，其中包含500条原始文本和500条对抗式改写，涵盖攻击与受害两类场景；

**📈 对比分析**

在与原始生成、通用安全提示以及三种行业门控模型（LlamaGuard、ShieldGemma、Qwen3Guard）的对比实验中，HRGuard在多数模型上将攻击方的有害合规率降至5%以下，同时保持受害方的保护指导率；门控触发率约为50%，多在对话早期出现；

**⚠️ 局限性**

局限性包括基准为合成情境，缺乏真实用户数据；评判依赖LLM判定，可能存在主观差异；仅测试五轮对话，未覆盖更长或工具交互的场景；角色推理未学习，需进一步验证。

---

## 215. Learning What to Share and What to Personalize: Hierarchical Strategy Co-Evolution for Agent Memory

**arXiv ID:** 2608.25329 | [PDF](https://arxiv.org/pdf/2608.25329v1)

**作者:** Yupeng Han `[一作]` (University of Science and Technology of China), Xianquan Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HiPS 框架，将记忆管理策略拆分为全局共享层和用户自适应层，并通过在线证据实现共进化；

**💡 创新点**

首次将层次化分层、动态门控、跨层规则流动、结构化差异蒸馏与对比采样相结合，构建可解释且高适应性的记忆策略；

**🔧 技术方法**

使用结构化差异蒸馏（USD）、Persona Delta 蒸馏（PDD）、语义相似度匹配、子模子采样、GRPO 强化学习与遵从奖励，全部基于大型语言模型（如 Qwen2.5‑7B‑Instruct、GPT‑4o‑mini 等）实现策略更新；

**📊 数据集**

在 PersonaMem、PrefEval、PersonaBench、PERMA 四个个性化记忆基准上进行实验，涵盖从小到大上下文、噪声级别与多域跨越；

**📈 对比分析**

与长上下文、RAG、Mem0、A-Mem、LightMem、MemAgent、MEM‑α、MemSkill 等基线对比，HiPS 在 12 个评估设置中均超越基线，尤其在长上下文与跨域任务中提升数十个百分点；

**⚠️ 局限性**

局限在于仅评估英文数据，未针对极长生命周期的慢性个体基线漂移和多语种语法差异进行适配；

---

## 216. Metis: Typed Runtime Mediation for Tool-Using Software Agents

**arXiv ID:** 2608.25322 | [PDF](https://arxiv.org/pdf/2608.25322v1)

**作者:** Jun Yu `[一作]` `[通讯]`, Jun Yu

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了 Metis，一个多供应商运行时，负责把模型提出的工具调用转换为带权限、调度、终端结果和生命周期边的强类型事件图，并在运行时完成权限决策、排队、终端结果闭合等操作。

**💡 创新点**

创新点是将运行时行为抽象为强类型事件图，提出四类调度（Safe、Queue、Exclusive、Background）、批量预检权限、终端闭合与子代理权限隔离，并在单一运行时内部实现多供应商适配器与完整的执行审计。

**🔧 技术方法**

使用 Go 语言实现，结合 provider adapters、权限门、四类调度器、补偿/修复机制、子循环以及电脑输入门等技术。

**📊 数据集**

使用冻结的源代码日志、配对的 30 条真实 I/O 工作负载、10 条注入故障、2 条子代理 ablation、5 个模型条件、1 条维护对照等数据集。

**📈 对比分析**

通过与强序列化 ablation 的配对比较，四类调度在相同工作负载下平均耗时降低约12 ms（占比 1.84），在 30/30 条案例中均优于串行；权限路由 oracle 与子代理实验也展示了完整路径覆盖。

**⚠️ 局限性**

限制包括：未验证跨模型或网络环境的泛化；缺乏事务回滚与主机失效恢复；权限判定可能因元数据错误而误判；子代理仅复制权限而非进程隔离；仅在本地单个主机和固定工作负载上测试，外部有效性未知。

---

## 217. SonicNudge: Controlled Displacement of Hovering UAVs via Estimator-Controller Coupling

**arXiv ID:** 2608.25319 | [PDF](https://arxiv.org/pdf/2608.25319v1)

**作者:** Shaocheng Luo `[一作]` (Duke University), Miroslav Pajic `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

做了什么

**💡 创新点**

创新点是什么

**🔧 技术方法**

用了什么技术

**📊 数据集**

用了什么数据集

**📈 对比分析**

如何比较的方法，性能怎么样

**⚠️ 局限性**

limitation是什么

---

## 218. Adaptive Triggering for Bias Correction in LLM Reasoning

**arXiv ID:** 2608.25379 | [PDF](https://arxiv.org/pdf/2608.25379v1)

**作者:** Nayoung Kim `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文提出了一种在LLM链式推理过程中动态触发偏见纠正的框架——Adaptive Triggering。

**💡 创新点**

创新点在于将偏见监测与在线变点检测（CUSUM）相结合，实现基于推理轨迹实时评估并仅在必要时插入纠正提示，且兼容白盒与黑盒信号。

**🔧 技术方法**

核心技术包括每一步的偏见风险信号提取（基于概率对比或LLM评判）、CUSUM统计量累积、阈值校准以及针对性反思提示的注入。

**📊 数据集**

使用的主要数据集为Bias Benchmark for QA（BBQ），涵盖九个社会偏见类别的含歧义与消歧义双版本问答。

**📈 对比分析**

实验将自适应触发与固定间隔触发、无干预等条件进行对比，结果显示自适应触发在降低干预次数的同时恢复了大部分消歧义情境下的准确率；但白盒信号在消歧义场景下的准确率有所下降，表明信号与目标不匹配。

**⚠️ 局限性**

主要局限包括信号与真实偏见的不完全对齐、阈值校准难以迁移、黑盒评判的自偏好问题、实验仅覆盖BBQ、缺乏对抗鲁棒性评估以及对其他生成任务的泛化未知。

---

## 219. CompanionHarm: A Multi-Turn Benchmark for Detecting Harms in Real-World AI Companion Conversations

**arXiv ID:** 2608.25377 | [PDF](https://arxiv.org/pdf/2608.25377v1)

**作者:** Renwen Zhang `[一作]` (Nanyang Technological University), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个多轮AI伴侣对话的有害行为标注数据集（CompanionHarm），并在该数据集上评估LLM的有害行为检测能力。

**💡 创新点**

首次在真实多轮对话中针对关系性与情感性有害行为设计细粒度分类体系，并保留了多 annotator 级别标注，揭示了注释者主观差异。

**🔧 技术方法**

采用多轮上下文感知的单标签14分类任务，利用七种公开/闭源LLM（GPT‑5.5、Claude Opus、Gemini、Qwen、Llama‑3.1 等）在零/一/全提示下进行推理。

**📊 数据集**

使用 2,111 条真实 Replika 对话（7,016 条 AI 语句）标注的 CompanionHarm 数据集。

**📈 对比分析**

在宏观 F1 上各模型表现均较低，最佳宏观 F1 仅为 0.453（GPT‑5.5），提示方式对不同模型影响不一致，显示检测仍具挑战。

**⚠️ 局限性**

局限包括样本来源偏向公开分享的极端案例、缺乏文化多样性、未标注有害行为的时间演化轨迹以及注释者主观差异导致的不确定性。

---

## 220. RAEM: Robust Autonomous Exploration for Multi-Floor Environments with a Quadruped Robot

**arXiv ID:** 2608.25366 | [PDF](https://arxiv.org/pdf/2608.25366v1)

**作者:** Zikang Yuan `[一作]` (Hong Kong University of Science and Technology), Xin Yang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

RAEM框架实现了四足机器人在多层楼建筑内的自主探索，解决了多层结构的地形与路径规划难题。

**💡 创新点**

创新点包括：①混合局部‑全局可行性表示（局部tomography + 3D网格）构建升高感知的全局拓扑图；②楼梯中心线对齐策略降低转向偏差；③双路径搜索机制补救楼梯稀疏观测导致的拓扑断裂。

**🔧 技术方法**

使用GPU并行体绘制、局部tomography与Explicit 3D Grid、基于拓扑图的ATSP规划、双路径A*、改造Ego‑Planner、Fast‑LIO2定位、UFOMap占用格、层叠树索引等技术。

**📊 数据集**

实验使用自建仿真场景（scene_1–scene_4）以及四个真实环境（楼梯、建筑1/2、大厅），无公开数据集。

**📈 对比分析**

与TARE、FAEL、HPHS三种基线进行对比，仿真中RAEM在覆盖率、时间和行驶距离上均优于对手，真实场景下实现从1层到5层楼的连续探索；单次规划计算时间稳定在0.1–0.2 s，GPU实现比CPU快约18倍，聚类式视点生成比采样法快约两倍。

**⚠️ 局限性**

局限性：仅针对向上楼梯的跨层探索，未考虑向下跨层；对低位LiDAR盲区和下行步态的动态稳定性不足。

---

## 221. Toward a Threat Actor Profiling Taxonomy for Pre-Release Risk Management of Open-Weight Frontier Models

**arXiv ID:** 2608.25361 | [PDF](https://arxiv.org/pdf/2608.25361v1)

**作者:** James Zhang `[一作]` `[通讯]`, James Zhang

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个基于技术精度、领域知识、组织能力、基础设施、资金和时间六个维度的威胁角色分类法，用于前期 AI 模型发布风险管理；

**💡 创新点**

创新点在于将这六个维度结合并按实证分层，形成统一、可操作的威胁角色矩阵，弥补了现有评估缺乏可比性和可解释性的空白；

**🔧 技术方法**

主要技术为文献综述与概念/经验推导相结合，利用现有恐怖主义、CBRN 与网络安全研究中的理论与数据构建分层规则；

**📊 数据集**

使用的核心数据集包括恐怖主义与网络攻击的公开统计（如美国恐怖主义研究、M‑Trends）、CBRN 事件记录和相关成本估算等；

**📈 对比分析**

方法上通过示例填充完整威胁配置并映射到评估设计的具体考量（如技术复杂度对应提问方式、领域知识对应子任务提示），虽未进行实验性性能比较，但展示了不同配置对评估重点的影响；

**⚠️ 局限性**

局限性包括：高层级（资金、基础设施等）经验数据稀缺导致分层不确定；缺乏对分类法在实际评估实验中提升可靠性与可比性的实证验证；以及对多维度交互效应的深入分析尚未展开。

---

## 222. Escaping Low-Dimensional Overlap: Multi-Task Model Merging via High-Dimensional Sparse Disentanglement

**arXiv ID:** 2608.25354 | [PDF](https://arxiv.org/pdf/2608.25354v1)

**作者:** Yihang Zhang `[一作]` (Central South University), Feng Zeng `[通讯]` (Central South University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种高维稀疏解耦合融合框架，用于在无额外训练的情况下融合多任务专家模型

**💡 创新点**

将任务向量投影到高维稀疏空间并使用改进的 Top‑K 稀疏自编码器实现特征层级解耦，同时利用 GR‑ZOO 高效识别关键层，仅在这些层进行稀疏融合，避免了传统参数空间算术与低维分解方法的冲突局限

**🔧 技术方法**

稀疏自编码器（Top‑K、残差拟合、解码器归一化与正交正则化）、Group‑Ranked Zeroth‑Order Optimizer (GR‑ZOO) 关键层选择、特征级融合与共享/独特特征区分

**📊 数据集**

Qwen2.5‑7B 与 Qwen2.5‑1.5B 基础模型，在 GSM8k、HumanEval、IFEval、MMLU 子集、BeaverTail 等任务上进行实验

**📈 对比分析**

与 Task Arithmetic、TIES‑Merge、DARE、Fisher‑Merge、TSV‑Merge、WUDI‑Merging、EMR‑Merging、DELLA 等训练‑free 合并基线对比；在 7B 上平均分提升至 68.49（高于 67.71），在 1.5B 四任务冲突场景提升 6.95%（36.48 vs 29.53），在高冲突 4 任务设置下更显优势

**⚠️ 局限性**

需要额外训练稀疏自编码器，增加前置计算开销；方法包含超参数（高维扩展因子、余弦相似阈值），在不同模型或任务规模时可能需微调

---

## 223. Not All Attention Heads Contribute to Critical Visual Token Selection: Head-Aware Pruning Matters More

**arXiv ID:** 2608.25332 | [PDF](https://arxiv.org/pdf/2608.25332v1)

**作者:** Chaofang Ma `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究视觉语言模型中的视觉令牌剪枝，提出了训练无关的Progressive Visual Token Pruning框架，结合预先剪枝与在LLM内部的多阶段头感知剪枝。

**💡 创新点**

创新点在于发现只有少数注意力头能准确定位重要视觉令牌，并利用此特性实现头感知剪枝，同时通过文本指令关键字提取与文本引导的相似性剪枝实现无训练高效剪枝。

**🔧 技术方法**

采用的技术包括头感知重要性估计、文本指令关键字提取、文本引导的相似性剪枝、聚合注意力、聚类重构以及对比实验评估。

**📊 数据集**

使用的数据集包括LLaVA-1.5-7B、LLaVA-NEXT-7B在MME、POPE、GQA、MMB、MMB_CN、SQA、VQA_Text等任务上，以及Qwen2.5-VL-7B等模型的多种评测。

**📈 对比分析**

与FastV、LLaVA-PruMerge、MustDrop、PDrop、HiRED、VisionZip、SparseVLM、DART、HoloV、ApET等基线在多项评测上进行比较，取得95.9%原始性能的保留、1.62×推理速度提升，并在更高裁剪比例下表现更优。

**⚠️ 局限性**

局限性在于需访问LLM内部状态，无法直接应用于封闭式模型；此外仍需计算注意力，部分加速被高性能注意力实现的开销抵消。

---

## 224. Rethinking Battery-free Sensing Communication via Wake-up Radios

**arXiv ID:** 2608.25292 | [PDF](https://arxiv.org/pdf/2608.25292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 225. V-Link: Recovering Lost Visual Representations in Action DiT for Vision-Language-Action Models

**arXiv ID:** 2608.25308 | [PDF](https://arxiv.org/pdf/2608.25308v1)

**作者:** Yehao Lu `[一作]` (Zhejiang University), Xi Li `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出V-Link模型，通过在视觉‑语言到动作特征传递过程中显式恢复视觉表示，提升机器人细粒度操控的感知与动作对齐。

**💡 创新点**

创新点在于引入空间和语义查询表示，并通过非对称路径将其注入动作生成器，从而弥补动作专家对3D几何和2D语义信息的访问缺口。

**🔧 技术方法**

采用视觉语言模型（VLM）与动作DiT架构，设计空间与语义查询模块及其注入机制。

**📊 数据集**

在LIBERO、LIBERO-Plus、RoboTwin 2.0以及AGIBOT A3 Ultra真实世界人形任务上进行评估。

**📈 对比分析**

与基线GR00T N1.6对比，V-Link在LIBERO、LIBERO-Plus、RoboTwin 2.0的平均成功率分别提升+1.9%、+31.2%和+18.8%；在AGIBOT A3 Ultra的两项人形任务上分别提高+20%和+24%。

**⚠️ 局限性**

主要局限在于仍需在更大规模、更多多样化场景下验证其通用性，并且对极端视觉条件下的鲁棒性尚未充分测试。

---

## 226. Sample Complexity of the Second-Best Bilateral Trade

**arXiv ID:** 2608.25303 | [PDF](https://arxiv.org/pdf/2608.25303v1)

**作者:** Qiaoyun Shi `[一作]` (Harbin Institute of Technology), Zongqi Wan `[通讯]` (Great Bay University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究在仅有样本的条件下，学习满足贝叶斯激励兼容、个体理性和预期预算平衡的双边交易机制，目标是实现第二最佳交易收益的近似。

**💡 创新点**

创新点在于给出了三个不同分布情形（有界支持、无界MHR）下的近似学习算法，并证明其样本复杂度与基准收益呈现标量敏感性，匹配上下界。

**🔧 技术方法**

主要技术包括经验线性规划优化、投影与支付设计、基于门槛机制的阈值化、收入门槛与规模本地化，以及 MHR 分布下的截断与 sentinel 处理。

**📊 数据集**

没有使用公开数据集，所有分析均基于理论构造的分布与样本。

**📈 对比分析**

与现有固定价格学习方法相比，本文的机制在保证贝叶斯激励兼容的前提下实现了与第二最佳收益相近的性能，样本复杂度分别为 O(h²/2)、O(h/(D)·1/α²) 或 O(χ_μ(D)/α²)，与下界相匹配。

**⚠️ 局限性**

局限性包括：对基准收益的依赖导致样本复杂度随第二最佳收益变小而显著增加；在无界分布下需额外截断步骤；算法主要针对独立值分布，未覆盖相关或多元情形。

---

## 227. PointRL: Learning Point-Level Vision-Language Grounding from Verifiable Annotation Evidence

**arXiv ID:** 2608.25299 | [PDF](https://arxiv.org/pdf/2608.25299v1)

**作者:** Jingyang Su `[一作]` (Beijing University of Posts and Telecommunications), Lu Yang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 PointRL，一个可验证的强化学习框架，通过利用现有的边界框、分割掩模和实例标签等多种注释，学习视觉语言模型在点级定位任务中的准确、可重复的指向行为。

**💡 创新点**

创新点在于：①将多模态注释转换为隐藏的可验证证据，并在奖励计算中保留目标支持与约束；②设计了一套层级的点奖励机制，兼顾单点定位与多点集合的覆盖、计数一致性与重复抑制；③在不需要额外数据集的情况下，仅通过规则生成的训练样本实现了显著的性能提升。

**🔧 技术方法**

主要技术包括：基于规则的注释转换与指令生成、解析器与格式门控、Hungarian 匹配与软定位评分、局部与全局奖励组合、GRPO（基于PPO的强化学习）以及 LoRA、ZeRO-2 等高效训练策略。

**📊 数据集**

使用的数据集：在 PointArena（Point-Bench）进行主实验；并在外部评测集 RoboSpatial、BLINK 与 Ref-Adv 上进行无额外微调的迁移评估。

**📈 对比分析**

与同骨干 Qwen3.5-4B 的对比显示，PointRL 在 Point-Bench 上整体准确率从 56.11% 提升至 65.58%（+9.47pp），在多点计数、推理与可控指向任务中提升尤为显著；在 RoboSpatial、BLINK 和 Ref-Adv 上也分别获得 7–13pp 的提升，表明奖励机制具有一定的泛化能力。

**⚠️ 局限性**

局限性包括：①奖励设计依赖手工设定的阈值与权重，可能在不同任务或数据分布下需要重新调优；②实验仅在单一 VLM（Qwen3.5）上验证，缺乏跨模型的普适性证明；③在 box‑output 评测中提升有限，提示点级训练对传统框框任务的直接迁移效果有限。

---

## 228. CRAMER: Control via Request-Aware Masking for Editing Recommenders

**arXiv ID:** 2608.25370 | [PDF](https://arxiv.org/pdf/2608.25370v1)

**作者:** Zhiyuan Julian Su `[一作]` (Renmin University of China), Ga Wu `[通讯]` (Dalhousie University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级框架CRAMER，通过将用户的自然语言请求映射为参数遮罩，以无须重新训练冻结的序列推荐模型即刻响应用户即时意图。

**💡 创新点**

创新点在于：① 将请求视为控制信号，在模型内部实现后向控制；② 采用行列遮罩（row‑column masks）和Gumbel‑Top‑k采样实现稀疏、可解释的参数编辑；③ 引入KL正则化的变分目标与直通估计器，使训练稳定且保持请求约束。

**🔧 技术方法**

核心技术包括：预训练语言模型编码器、Transformer（SASRec/BERT4Rec）冻结骨干、行列遮罩生成与矩阵乘法遮罩、Gumbel‑Top‑k采样与STE、KL正则化与变分推导。

**📊 数据集**

在四大公开数据集上评估：ReDial、KuaiSAR、Beauty、CDs&Vinyl，并在两种Transformer骨干（SASRec、BERT4Rec）上进行实验。

**📈 对比分析**

与四种主流请求感知基线（Query‑SeqRec、BLaIR、LLM‑ESR、REARANK）在HR@k、NDCG@k、MRR@k等指标上对比，CRAMER在绝大多数设置下均实现显著提升（如HR@10提升>10%），同时在推理时间与显存占用上仅增加约0.018s/1355MiB，表现出优越的效率与效果。

**⚠️ 局限性**

局限性包括：① 仅对冻结骨干进行遮罩，无法对全局参数做更大幅度改动；② 对极为复杂或模糊的请求可能难以完全捕捉；③ 对稀疏度参数的选择敏感，需手工调优；④ 仍需系统层面的安全与伦理保障，防止意外过度驱动推荐。

---

## 229. Beyond Pairwise Feedback: Listwise Vision-Language Supervision for Preference-Based Reward Learning

**arXiv ID:** 2608.25350 | [PDF](https://arxiv.org/pdf/2608.25350v1)

**作者:** Srivalli Katkuri `[一作]` (Purdue University), Juan Wachs `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文结合视觉-语言模型生成的多项（listwise）偏好，利用 Plackett‑Luce 形式学习奖励函数，并在 Meta‑World 仿真机器人抓取任务中训练 Soft Actor‑Critic 策略。

**💡 创新点**

创新点在于首次将 VLM 生成的多项排序作为监督，采用 Plackett‑Luce 奖励学习取代传统仅用两项比较的 Bradley‑Terry 方式，从而提供可调的 K 值和更灵活的反馈预算。

**🔧 技术方法**

使用技术包括 GPT‑5.6 Luna 视觉‑语言模型、Plackett‑Luce 与 Bradley‑Terry 比较、Soft Actor‑Critic 强化学习、统一随机采样以及无主动选择的反馈策略。

**📊 数据集**

实验数据集主要为 Meta‑World 的三种仿真任务（Drawer Open、Door Close、Button Press）以及相应的 RGB 图像，用于 VLM 生成偏好。

**📈 对比分析**

在相同的反馈预算 M=4、K=3/4/5 下，PL 与 BT‑Kwise、BT‑Pairwise、RL‑VLM‑F、Oracle 进行对比；在 Drawer Open 中，PL K=4 的成功率为 86%（与 RL‑VLM‑F 的 92% 相近）；在 Door Close 中，PL K=5 达到 54% 高于 RL‑VLM‑F 的 39%；在 Button Press 中，PL K=4 为 41%，与 BT‑Kwise 的 45% 接近；总体来看，PL 竞争力强，在 Button Press K=5 时相较 BT‑Kwise 获得了统计上显著的提升（p≈0.045）。

**⚠️ 局限性**

局限性包括：仅测试了三种简单的刚体任务，对视角变化敏感且 Qwen3‑VL‑8B 无法成功；缺乏主动查询策略和真实机器人实验；warm‑up 阶段需大量探索；未验证更复杂任务或更大 K 的效果。

---

## 230. CoRE: Weakly Supervised Coarse-to-Fine Risk Evidence Learning in Driving Videos

**arXiv ID:** 2608.25344 | [PDF](https://arxiv.org/pdf/2608.25344v1)

**作者:** Kaiser Hamid `[一作]` (Texas Tech University), Nade Liang `[通讯]` (Texas Tech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种弱监督的 CoRE 框架，利用对视频级预测的结构化干预产生的预测效应，进而学习时间和实体级的支持信息；

**💡 创新点**

创新点在于将预测效应蒸馏为细粒度支持目标，将教师模型的干预结果转化为学生模型的直接预测，突破了传统依赖精细标注的限制；

**🔧 技术方法**

核心技术包括教师–学生蒸馏、结构化干预、预测效应测量、梯度停止以及基于时序和实体候选的支持头；

**📊 数据集**

在三大数据集上评估：RISEE（驾驶感知风险评估）、DoTA（驾驶异常检测）和 UCF-Crime（非驾驶视频异常检测）；

**📈 对比分析**

与多种基线（MIL、RTFM、MGFN 等）相比，CoRE 在 RISEE 上获得最高的 Spearman 相关和低 MAE，在 DoTA 上在帧 AUC、F1@0.5、tIoU 等指标上均优于所有对比方法，在 UCF-Crime 上实现了 85.68% 的帧 AUC，超过了对照的 RTFM 和 MGFN；

**⚠️ 局限性**

局限性包括对干预策略的依赖、候选构造的手工设计，以及对多尺度和实体跟踪的强假设，可能在多模态或更复杂场景下效果有限。

---

## 231. GUIDE: Generative Unsupervised Chinese Query Correction via Phonetic and Visual Shared-ID Encoding

**arXiv ID:** 2608.25343 | [PDF](https://arxiv.org/pdf/2608.25343v1)

**作者:** Lei Yang `[一作]` (Kuaishou Technology), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于“先混淆后澄清”的无监督中文查询纠错框架GUIDE，利用音/形相似字符共享ID并通过Encoder-Decoder重构原查询实现控制性纠错。

**💡 创新点**

创新点在于将混淆邻域嵌入输入层并作为重构目标，避免手工生成噪声；同时通过时间衰减的频率加权目标实现对快速变化词汇的自适应；结合音/形两种共享ID聚类实现更全面的误写覆盖。

**🔧 技术方法**

技术包括：字符聚类（音韵去调拼音聚类、视觉相似ViT聚类）、Transformer编码解码网络、时间衰减+频率加权负对数似然训练、动态融合音/形模型的beam搜索。

**📊 数据集**

使用公开的QSpell 250K查询纠错基准以及公司内部大规模搜索日志数据集KwaiSearch（约1.8亿条）。

**📈 对比分析**

与监督基线BERT、Masked-FT、无监督Simple-CSC和LLM-ICL等对比，GUIDE在QSpell和KwaiSearch上均取得最高的Precision/Recall/F1，在线A/B测试显示误拼率下降80%，搜索量提升0.122%。

**⚠️ 局限性**

局限包括：仅处理长度保持的字符替换，未覆盖插入/删除/短语重写；共享ID聚类虽覆盖主流误写，但仍可能遗漏或引入噪声；当前音/形融合采用启发式选取，缺乏统一融合模型。

---

## 232. Framing War Across Languages: Power, Agency, and Sentiment in Wikipedia's Multilingual War Narratives

**arXiv ID:** 2608.25337 | [PDF](https://arxiv.org/pdf/2608.25337v1)

**作者:** Jiarui Xia `[一作]` (University of Notre Dame), Diego Gomez-Zara `[通讯]` (University of Notre Dame)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 158 场 1900 年后战争的 2,542 场战役，在 20 种 Wikipedia 语言版本中，使用共情框架（connotation frames）系统分析并对比战斗双方在权力、能动性和情感上的叙事描写。

**💡 创新点**

首次将共情框架方法规模化应用于跨语言维基百科战争叙事，揭示自我涉及时的叙事偏差及其取向差异，且证明非自我参与时叙事趋于一致，填补了以往仅关注覆盖度或情感极性研究的空白。

**🔧 技术方法**

采用 Google 翻译 + 逆向翻译实现多语言统一，使用 RIVETER 依存句法、命名实体识别与共指消解提取 SVO 句子并给出权力/能动/情感三维评分；通过 Wikidata 进行实体归属链接；采用 Wilcoxon 符号秩检验、Spearman 相关、OLS 回归、传播核（Propagation Kernel）网络相似度分析等统计与网络方法比较不同语言叙事。

**📊 数据集**

基于 Wikipedia MediaWiki API 收集的 158 场战争与 2,542 场战役战斗篇章，覆盖 20 种语言（共 16,058 篇文章），并以 Wikidata 解析战斗双方的官方语言和国家归属。

**📈 对比分析**

通过双侧 Wilcoxon 检验比较自我与敌对实体在权力/能动/情感维度上的差异，利用 Spearman 相关和 OLS 回归探究文化、编辑行为和战果对差异的解释力度；网络相似度计算显示在非自我参与战争时语言间结构相似度在 0.7–0.9 范围内聚集；结果表明自我参与时差异显著、非自我参与时高度趋同，证明方法能有效捕捉跨语言叙事差异。

**⚠️ 局限性**

主要限制包括：翻译过程可能引入语义偏差，尤其是情感维度；样本仅覆盖 20 种语言，统计功效有限；未能对编辑者构成、地区分布及多党派冲突等复杂因素进行因果分析；NLP 解析错误与实体链接不完整可能影响评分准确性；并且仅关注自我对抗关系，未探究更细粒度的联盟结构。

---

## 233. GraftSR: Grafting Authentic Textures for Real-World Image Super-Resolution via Identical-Instance Guidance

**arXiv ID:** 2608.25334 | [PDF](https://arxiv.org/pdf/2608.25334v1)

**作者:** Qifan Yu `[一作]` (Taobao & Tmall Group of Alibaba), Ying Chen `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于纹理参考的生成式图像超分辨率框架 GraftSR，解决了传统扩散模型的纹理幻觉问题。

**💡 创新点**

创新点是双遮罩参考引导机制：分别通过遮罩调制参考纹理和区域感知语义标记，解耦了“取什么纹理”和“放在哪儿”，无需精确几何对齐即可实现跨视角纹理迁移。

**🔧 技术方法**

采用 MMDiT 变体作为后端，利用 VAE、Vision‑Language Model、SAM 进行条件提取，并通过一阶对抗蒸馏训练实现高效一跳 SR。

**📊 数据集**

构建了首个大规模同实例纹理参考 SR 数据集 TexRefSR‑141K（141k 组高质量同实例图像及互补遮罩），以及对应评测基准 TexRefSR‑Eval。

**📈 对比分析**

在 TexRefSR‑Eval 上与多种公开基线和商业编辑模型对比，GraftSR 在 LPIPS、DISTS、PSNR、SSIM、CLIPIQA、MUSIQ 等指标均遥遥领先，LPIPS 下降 20.2%，PSNR 超过 30 dB，显著优于 Gemini‑3‑Pro、GPT‑Image‑2 等。

**⚠️ 局限性**

局限性：在极度退化图像时仍需权衡细节与保真度，且对极端视角差异或完全不匹配的参考仍可能产生轻微纹理丢失。

---

## 234. Two Dimensions Govern Agnostic Multiclass Transductive Learning

**arXiv ID:** 2608.25326 | [PDF](https://arxiv.org/pdf/2608.25326v1)

**作者:** Pahan Dewasurendra `[一作]` `[通讯]` (Johns Hopkins University), Pahan Dewasurendra (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文给出了多类别无偏置（agnostic）传递学习（transductive learning）的最优误差率上界与下界，证明其与PAC学习具有相同的两维度（DS维度与Natarajan维度）控制，且误差率可达Θ(d_S/n + √(d_N/n))（上界至对数因子）。

**💡 创新点**

创新点在于：
- 引入随机预留（random‑reservation）策略，将PAC学习的压缩结构转移到固定有限样本上，避免传统留一法的稳定性问题；
- 提出无放回（without‑replacement）乘法权重（multiplicative‑weights）菜单引理，保持快速的1/n误差项；
- 通过DS伪立方和Natarajan立方构造匹配下界，证明两维度均必不可少；
- 统一了多类别传递学习与PAC学习的两维度理论框架。

**🔧 技术方法**

主要技术包括：
- 真实可压缩（realizable compression）与多类别压缩方案；
- 乘法权重（multiplicative‑weights）菜单与标签空间降维；
- 随机划分与无放回Hoeffding定理；
- DS伪立方和Natarajan立方构造；
- 复合三阶段PAC学习结构的传递化。

**📊 数据集**

本文未使用具体实验数据集；研究完全基于理论分析与证明。

**📈 对比分析**

比较方法：与传统的PAC到传递学习的转换（需1/ε样本放大）以及直接留一式传递学习的下界做比较。理论上，提出的算法在任意标签空间下，误差率达到与PAC学习相同的两维度界限，且仅多对数因子。

**⚠️ 局限性**

限制：
- 结果存在对数因子；若想消除对数，可能需要更直接的加权方向化方法；
- 证明和算法主要信息理论性质，未给出可计算实现；
- 仅适用于无偏置的多类别分类，未讨论其他损失或在线场景。

---

## 235. AVI-Personality: A Trait-Activated Multimodal Dataset for Personality and Competency Assessment in Asynchronous Video Interviews

**arXiv ID:** 2608.25316 | [PDF](https://arxiv.org/pdf/2608.25316v1)

**作者:** Tianyi Zhang `[一作]` (Southeast University), Wenming Zheng `[通讯]` (Southeast University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个基于Trait Activation Theory的结构化异步视频面试数据集AVI-Personality，并对其进行可靠性、效度、正交关联与公平性验证，同时对多模态文本、音频、视觉与多模融合模型在该数据集上进行了系统基准评测。

**💡 创新点**

① 采用特质激活理论设计的个性化问题，显著提升观察者评分的可靠性和自他一致性；② 使用专业心理学家与招聘者通过BARS进行多评审，确保标签的心理测量学稳健；③ 在基准评测中首次展示文本LLM在AVI个性预测中的主导优势及多模融合的微弱提升。

**🔧 技术方法**

文本模型：Longformer、T5、小型BERT、专门的PersonalityLLM以及LLM微调方法；音频模型：Whisper、Wav2Vec2、Emotion2Vec、传统eGeMAPS、is13-compare；视觉模型：Swin Transformer、ViT-MAE、DAN、VAT等；多模模型：ResNet+BERT、HFUT-VisionXL、EMMR、CAS-MAIS、AU-Personality、AVI2025等。

**📊 数据集**

AVI-Personality：3,876段视频，646名参与者，包含自评HEXACO人格、观察者BARS评分、招聘者评估的工作相关能力、面试表现与认知能力，视频基于两类问题（通用+特质激活）收集。

**📈 对比分析**

以均方误差（MSE）为指标对7大类模型进行对比；文本LLM（如PersonalityLLM）在平均MSE上最优；音频与视觉单模模型表现较差；多模融合虽总体提升但相较强文本模型差距不大，提示多模融合效果有限。

**⚠️ 局限性**

① 个性问题仅覆盖四个HEXACO维度，情绪性与开放性仅通过自评和通用问答测量；② 观察者评分仍可能带有评判者偏好与人口学敏感度；③ 多模融合方案仍未充分挖掘跨模态互补信息；④ 数据集主要来自模拟管理培训岗位，可能不完全泛化至其他职位或真实面试场景。

---

## 236. Activation-Space Order-Swap Geometry: A Site-Asymmetry Audit

**arXiv ID:** 2608.25315 | [PDF](https://arxiv.org/pdf/2608.25315v1)

**作者:** Anqi Peter Li `[一作]` `[通讯]` (Substrate Labs), Anqi Peter Li (Substrate Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究激活空间中顺序相关性统计，提出无拟合的站点非对称审计以区分一阶与交互项。

**💡 创新点**

创新地用四次单注入响应估计顺序交换括号的基线，并通过二次差分得到真正的交互成分，从而揭示一阶项主导的现象。

**🔧 技术方法**

利用Taylor展开、单注入响应测量、二次差分与对称控制、余弦相似度比较等技术。

**📊 数据集**

在六大开源LLM（DeepSeek、Gemma、Llama、Mistral、OLMo、Qwen）7–9B模型及16个特征对比、48个保留提示上测试，并对ViT‑B/16、ResNet‑50视觉模型进行验证。

**📈 对比分析**

通过余弦相似度比较基线预测与实际括号，单注入基线解释84–98%；二次差分残差在3/6模型上突破交互无关假设；随机初始化网络显示一阶项主导，说明该方法能有效区分一阶与交互。

**⚠️ 局限性**

仅适用于不同注入位置的激活干预；残差仍受高阶项影响，无法直接量化交互强度；结果受特征方向噪声与提示共轭影响。

---

## 237. WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine Guided Depth Super-Resolution

**arXiv ID:** 2608.25302 | [PDF](https://arxiv.org/pdf/2608.25302v1)

**作者:** Tayyab Nasir `[一作]` (University of Western Australia), Ajmal Mian `[通讯]` (University of Western Australia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为WAVE的粗到细重建框架，通过多级离散小波变换（ML‑DWT）对RGB引导进行频域分解，并逆序消费小波子带和DINOv3语义标记，实现 Guided Depth Super‑Resolution 的结构优先重建；

**💡 创新点**

创新点包括：①将多级小波分解作为显式、可解释的指导控制机制，逆向消费子带实现粗到细重建；②利用可逆耦合（IRN‑style）融合RGB、深度与语义三模态，避免信息丢失；③使用语义门控对高频子带进行自适应抑制，减少纹理复制伪影；④采用低秩适配器（SToRA）对DINOv3标记进行轻量化、任务专一化微调；

**🔧 技术方法**

使用技术包括：多级离散小波变换、DINOv3 ViT 语义标记、低秩适配器 SToRA、可逆耦合交叉模态融合、双分支结构/细节处理、交叉注意力与自注意力、双向投影（上采样/下采样）、反向投影机制、边界强化模块等；

**📊 数据集**

实验数据集：训练集 HYPERSIM、NYU_v2；测试集 Middlebury、Lu、NYU_v2、RGBD‑D、TOFDSR、DIML 等；

**📈 对比分析**

与多种 SOTA（SPFNet、SGNet、DKN、FDSR、DCTNet、SGNet、DORNet、SPFNet 等）在 8×、16×、32× 的 GDSR 任务上对比，WAVE 在低倍率下与同类方法竞争，在 32× 时 RMSE 下降约 0.2–0.3 cm，整体排名前列，尤其在极端放大因子上表现突出；

**⚠️ 局限性**

局限性：依赖冻结的 DINOv3 语义基座，模型对低倍率/域内任务已趋于饱和；未验证在真实降质或盲降解场景下的鲁棒性；模型规模相对较大，部署在实时或嵌入式系统上仍存在挑战。

---

## 238. Sequential Euclidean tree construction with exponential memory: distributional performance and worst-case guarantees

**arXiv ID:** 2608.25298 | [PDF](https://arxiv.org/pdf/2608.25298v1)

**作者:** Pedro M. M. de Castro `[一作]` `[通讯]`, Pedro M. M. de Castro

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种指数加权记忆策略（γ-strategy）在点插入树中的期望与最坏情况性能；

**💡 创新点**

提出该策略在均匀输入下可通过调节γ使期望成本下降，并给出最坏情况的精确上界及最优参数的闭式表达；

**🔧 技术方法**

运用了迭代随机函数理论、Olkin–Tong峰度比较、层积公式、解析积分与极限定理等概率与分析工具；

**📊 数据集**

实验与分析均基于在单位球内独立均匀分布的点序列以及对抗序列（如两端点交替）；

**📈 对比分析**

通过与路径插入和以球心为中心的星形树的比较，证明期望成本接近中心星且最坏情况随γ→1收敛至该值；在N插入时最优γ≈1−N^{-1/2}；

**⚠️ 局限性**

研究仅涵盖α≤3的情况，且仅考虑固定γ的指数权重，对高阶α、非球体或非均匀分布的适用性尚未解决。

---

## 239. Synthesis of Hopfield Neural Network: Novel Results

**arXiv ID:** 2608.25481 | [PDF](https://arxiv.org/pdf/2608.25481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 240. Forecasting Global Volatility Across Asynchronous Markets: Incremental Accuracy from Constrained Cross-Market Attention

**arXiv ID:** 2608.25369 | [PDF](https://arxiv.org/pdf/2608.25369v1)

**作者:** Xinlin Zhao `[一作]` (Independent Researcher), Ziyao Lin `[通讯]` (Beihang University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文构建了一种结合历史VAR预测性连接与Transformer注意力的跨市场波动率预测框架，并以HAR模型为基准进行增量改进。

**💡 创新点**

创新点在于：①严格的预测起点可用信息集合与异步交易日校正；②引入可学习的市场特定门控将数据驱动注意力与经济先验平衡；③通过冻结HAR预测并在逆softplus域限定神经残差实现可解释性与正值约束。

**🔧 技术方法**

技术手段包括：岭回归VAR+GFEVD作为先验；双通道时空Transformer（Temporal MHSA + Spatial attention + 先验混合）；残差门控与软加法融合；HAC调整的Diebold–Mariano检验与Model Confidence Set评估。

**📊 数据集**

数据集为2006‑2022年间八个全球主要股票指数的日度实现波动率，涵盖异步交易日，约4,079个联合日历。

**📈 对比分析**

比较方法：在所有交易日与公共交易日两种面板下，使用MSE与MAE评估，结合5/10%显著性DM检验和MCS。结果显示：对HAR基准而言，提出的PGA‑Trans‑HAR在每日、每周和部分每月均显著降低MSE/MAE；相较于VHAR、HAR‑KS、GNN‑HAR、DCRNN‑HAR和纯Transformer，PGA‑Trans‑HAR在多步预测中实现最低平均MAE（每日）和最低平均MSE/MAE（每周与每月）。

**⚠️ 局限性**

局限性包括：仅使用波动率作为特征，未纳入宏观变量、期权隐含波动等；市场特定门控在每日预测下不提升效果；VAR先验与门控参数设定固定，缺乏对不同周期的动态调整；仅给出点预测，未扩展到分布或风险度量；实验范围受限于八个指数，后期疫情后数据未覆盖。

---

## 241. Semi-Supervised Adaptation of Vision-Language Models for Image Classification

**arXiv ID:** 2608.25485 | [PDF](https://arxiv.org/pdf/2608.25485v1)

**作者:** Mohamed L. Mekhalfi `[一作]` (Fondazione Bruno Kessler), Mansour Zuair `[通讯]` (King Saud University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于CLIP的半监督遥感场景分类框架SE-CLIP，采用递归标签挖掘与类平衡策略进行自我演化。

**💡 创新点**

创新点在于将跨模态语义锚点与低秩适配结合，使用递归类平衡挖掘实现无监督样本的高质量扩展，突破了传统阈值式伪标签的限制。

**🔧 技术方法**

采用了CLIP预训练模型、LoRA低秩适配、固定文本锚点、递归类平衡挖掘以及跨模态对齐损失。

**📊 数据集**

在UCM和NWPU-RESISC45两个遥感图像分类基准数据集上进行实验。

**📈 对比分析**

与多种现有半监督方法（如FixMatch、DARP等）对比，SE-CLIP在UCM达99.09%、NWPU达95.07%，显著优于最强基线。

**⚠️ 局限性**

局限性包括对语义词汇表达的依赖，且在细粒度类别高度相似时，语义锚点难以充分区分视觉重叠。

---

## 242. Gaussian Splatting Underwater: A Controlled Cross-Regime Study

**arXiv ID:** 2608.25483 | [PDF](https://arxiv.org/pdf/2608.25483v1)

**作者:** Olaya Álvarez-Tuñón `[一作]`, Stella Graßhof `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了高斯 splatting 在不同水质环境下的 3D 重建性能，并在统一协议下比较了五种公开实现。

**💡 创新点**

采用统一姿态、初始化、预算和评估协议，首次将光照、浊度、光源移动等四个水域条件与几何评估一起量化，揭示光学模型与几何误差的相互关系。

**🔧 技术方法**

以 3D Gaussian Splatting 为核心，结合光学散射模型（单光子衰减、散射）、预处理增强、单视深度监督以及卷积网络等多种变体。

**📊 数据集**

评测四个公开/工业数据集——Curaçao（S1）、SOTRUE 4 浊度级别（S2）、Eiffel Tower 深水探测（S3）以及 EIVA 运营调查（S4）。

**📈 对比分析**

通过共享姿态、初始化和预算，对 PSNR/SSIM/LPIPS、几何误差（Chamfer、浮体质量）和原子数量进行评估，发现无介质模型在清晰水域表现最优，而介质感知模型在浑浊或移动光源下几何退化，且预处理增强在实地调查中最具成本效益。

**⚠️ 局限性**

介质模型参数仅对单个水域有效，难以跨场景迁移；光学模型对高浊度时姿态估计极为脆弱；几何评估在无参考数据时仅能通过浮体质量估计，整体评估仍受限于标注成本和计算开销。

---

## 243. 4DStreamCtrl: Interactive Video Generation with Online 4D Control

**arXiv ID:** 2608.25479 | [PDF](https://arxiv.org/pdf/2608.25479v1)

**作者:** Shiqian Li `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出统一的3D轨迹接口，实现摄像机、对象运动与深度三者的可实时、可交互式4D视频生成；

**💡 创新点**

创新点在于将3D点轨、相机参数与深度融合为单一条件信号，配合轻量几何运动头和基于4步推理的因果流式蒸馏，实现首个单GPU实时4D可控流式生成；

**🔧 技术方法**

使用SpatialTrackerV2提取3D轨迹、Geometric Motion Head进行轨迹与深度编码、Wan2.2 TI2V-5B diffusion backbone、LoRA参数高效微调、self-forcing与DMD蒸馏、因果块注意力+KV缓存实现流式生成；

**📊 数据集**

构建规模达约0.4M条目、32×32点轨与相机参数的In-the-wild 3D Motion Dataset（来源OpenVid-1M并通过SpatialTrackerV2标注）；

**📈 对比分析**

在DAVIS验证集上与多种2D/3D轨控制方法对比，教师模型EPE 5.29、LPIPS 0.404、SSIM 0.479，学生模型保持EPE 5.48，仅需4步推理，帧率20FPS，显著优于MotionStream等；

**⚠️ 局限性**

局限性包括对极端运动模糊或剪辑导致的追踪失败敏感、单相机轨迹分辨率限制、长序列可能累计误差，以及对高质量3D轨与相机参数的依赖。

---

## 244. Transient multimode heat transfer of an industrial automated tape laying process under rapidly changing conditions

**arXiv ID:** 2608.25470 | [PDF](https://arxiv.org/pdf/2608.25470v1)

**作者:** Bernhard Rameder `[一作]` (Johannes Kepler University Linz), Ronald Naderer `[通讯]` (FerRobotics Compliant Robot Technology GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

建立并验证了一种面向工业自动化叠层成型(ATL)的时变热辐射模型，能够根据实际的驱动电流与速度轨迹预测带子在各个位置的温度分布。

**💡 创新点**

创新点包括：①将红外发射器分段细化以捕捉起始电流峰值的快速热动力学；②采用基于有限几何的解析视角因子来纠正传统1.5D模型的辐射过估；③使用局部混合对流评估（根据里德曼数）和两层厚度模型捕捉带子两侧温差；④在高阶Radau IIA隐式求解中实现耦合辐射与传热的单组装(monolithic)解法。

**🔧 技术方法**

技术手段包括：热传导方程与耦合的辐射视角因子模型、基于灰体近似的辐射计算、局部混合对流模型、双节点带子模型、分段红外发射器电热动态模型、Radau IIA 2阶隐式积分及CasADi求解器。

**📊 数据集**

使用的数据集：工业ATL生产线在真实轨迹下采集的带子进给速度、红外加热电流及贴面温度测量（多点、实时采样），用于模型驱动输入与结果验证。

**📈 对比分析**

与实验数据的比较显示：在监测点温度的最大绝对误差13.52 °C，最大相对误差4.19 %，RMSE为4.02 °C，NRMSE仅为1.08 %，证明模型在快速动态和高功率变化下保持高度精确，且与传统常数对流系数模型相比，误差显著下降。

**⚠️ 局限性**

局限性包括：假设灰体辐射与均匀光学属性，未考虑波段特性；对流系数与辐射率的确定仍存在不确定性；模型验证仅在单一轨迹下完成，缺乏多工况独立验证；实时预测的计算时长仍高，未达控制循环需求。

---

## 245. Hierarchical Shared Memory-Aware Optimization for TRSM on GPU Platforms

**arXiv ID:** 2608.25469 | [PDF](https://arxiv.org/pdf/2608.25469v1)

**作者:** Xinzhe Chen `[一作]` (Institute of Software, Chinese Academy of Sciences), Fangfang Liu `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 HSMA-TRSM，基于层次共享内存感知的优化框架，用于左侧下三角 TRSM 在 NVIDIA A100、H800 以及 Hygon DCU Z100 GPU 上的实现。

**💡 创新点**

创新点包括：
- 针对双精度复数小规模（m,n≤64）设计的双线程组七阶段流水线，突破共享内存 64KB 限制；
- 对大规模问题采用对角块解耦与双缓冲技术，将对角块求逆的共享内存复杂度从 O(I_B^2) 降至 O(I_B)；
- 通过离线性能曲线与在线查表实现跨平台自适应块大小，避免固定 128 大小的弊端。

**🔧 技术方法**

使用技术包括：
- 计算-内存重叠、循环展开与指令重排；
- 共享内存分区与多线程组并行；
- 双缓冲对角块求逆流水线；
- 基于离线性能表的快速查表配置；
- 针对不同 GPU（SM/CU、共享内存容量、HBM 带宽等）的平台特定优化。

**📊 数据集**

使用的数据集：随机生成的稠密矩阵，尺寸覆盖从 4×4 到 16384×16384 的各种 m,n 值；并未使用公开的实际科学或机器学习数据集。

**📈 对比分析**

对比方法：在三款 GPU 上分别与 cuBLAS v13.2.1（A100、H800）和 rocBLAS v5.1（DCU）进行基准测试。结果显示：
- 小规模双精度复数时，HSMA-TRSM 最高可获得 2.05× 的加速；
- 大规模实数（float、double）在 A100、DCU、H800 上分别达到 1.63×–2.06× 的加速；
- 对比 MAGMA，HSMA-TRSM 在中大规模时优于固定 128 阻塞的实现。

**⚠️ 局限性**

局限性：
- 对非常小规模（如 1×1、2×2）仍未实现最佳融合；
- 对于已高度优化的 vendor kernel（尤其是 H800 的 double‑complex），提升幅度有限；
- 需要离线性能曲线，若硬件或驱动升级可能需重新生成；
- 当前未覆盖稀疏、混合精度或批处理 TRSM 的场景。

---

## 246. Resolving Multi-Modal Regression by Difference-Quotient-Based Clustering:Fast Coarse Conditional-Label Assignment

**arXiv ID:** 2608.25467 | [PDF](https://arxiv.org/pdf/2608.25467v1)

**作者:** Huang Weiquan `[一作]` `[通讯]` (Guangdong Polytechnic Normal University), Huang Weiquan (Guangdong Polytechnic Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了一种基于差商聚类的无梯度预分割方法，先将样本按输入输出差商聚成低矛盾簇，再用 logits 生成器和条件网络实现多模态回归，从而缓解平均回归问题。

**💡 创新点**

创新点在于将差商（Δy/Δx）作为矛盾度量，提出一次性、梯度无关的 Difference‑Quotient Clustering；同时配合 minMSE 评估协议和可迭代的细化思路，实现了对多模态数据的快速粗分割。

**🔧 技术方法**

使用的技术包括：差商聚类、深度多层感知机（logits 生成器与条件网络）、minMSE 评估、并行 O(n²/2) 计算以及实验中对网络深度与样本量的系统比较。

**📊 数据集**

所用数据集为纯合成的多模态回归数据，输入为二维、输出为四维，K=5 或 K=10，生成方式为对相同基输入复制多次并通过不同随机权重的子网络映射。

**📈 对比分析**

在与 oracle、随机标签和平均折叠三种基线的对比实验中，DQC 在 K=5、10 时的 minMSE 接近 oracle（误差 2‑3 倍以内），显著优于随机标签（≈1.1–1.3）和平均折叠（≈1.5‑1.4），并且更深网络可进一步降低训练误差。

**⚠️ 局限性**

局限包括：聚类仅基于局部矛盾导致标签不纯；结果受随机种子影响且缺乏最优性保证；g 网络只能基于 x 单独预测分支，无法在同一输入区分所有模态；目前仅在合成数据上验证，未对真实多模态数据和未知模态数的情况进行评估。

---

## 247. GLOSS: Geometric Local Self-Similarity Learning for Faithful Reference-Guided Texture Fill

**arXiv ID:** 2608.25461 | [PDF](https://arxiv.org/pdf/2608.25461v1)

**作者:** Chenyue Cai `[一作]` (Princeton University), Masha Shugrina `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

该工作提出了一种基于几何条件的局部纹理修复模型，利用单个3D模型的自相似性和离线生成的单视图数据，实现纹理补全与交互式纹理填充。

**💡 创新点**

创新点在于将批量多注意力与几何-纹理自相似先验相结合，允许在不需要大规模3D数据的前提下，从单个模型自生成训练集实现高质量局部纹理生成。

**🔧 技术方法**

使用了Stable Diffusion v2.1 unCLIP的图像扩散模型、Batch Multi-Attention机制、ControlNet、LLM生成提示、去光网络以及LPIPS感知损失。

**📊 数据集**

训练数据完全由该模型自身的单视图渲染与生成图像合成而来，无需外部纹理3D数据集。

**📈 对比分析**

与TEXGen、Hunyuan 2.1、Paint3D、TRELLIS 2等基线相比，在多项局部纹理质量指标（FID、CMMD、LPIPS、DreamSim）上取得相近或更优成绩，并在交互式填充任务中表现出更好的几何一致性。

**⚠️ 局限性**

主要局限包括对模型的单形态适配、对不同形状的泛化能力有限、PBR通道一致性不足以及推理速度仍需提升。

---

## 248. Training Alignment Auditors via Reinforcement Learning

**arXiv ID:** 2608.25460 | [PDF](https://arxiv.org/pdf/2608.25460v1)

**作者:** Paul Rosu `[一作]` (Anthropic), Rowan Wang `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了一个基于强化学习的LLM审计员，提升了对前沿模型隐藏行为的检测能力。

**💡 创新点**

采用对比式(pairwise)奖励与误报校准，避免奖励游戏并实现更稳健的调查质量。

**🔧 技术方法**

使用强化学习、LLM评判器以及系统提示植入隐藏行为的对比奖励设计。

**📊 数据集**

使用Petri评估框架、AuditBench硬化目标以及生成的32种隐藏行为与情景种子，涉及多模型多目标训练。

**📈 对比分析**

与Opus 4.6/4.7等前沿模型对比，复合得分从44.2提升至48.7，审计质量超过Opus 4.6，误报率保持<1%，在AuditBench硬化目标上检测率提升至28.1%。

**⚠️ 局限性**

评估完全基于LLM评判器且同族偏差；仅在单一规模模型上测试；训练样本植入行为弱于微调；真实审计真实性仍低。

---

## 249. VGA-BenchV2: An Expanded Unified Benchmark and Multi-Model Framework for Evaluating Video Aesthetics and Generation Quality

**arXiv ID:** 2608.25452 | [PDF](https://arxiv.org/pdf/2608.25452v1)

**作者:** Longteng Jiang `[一作]` (Ant Group), Xin Jin `[通讯]` (Beijing Institute for General Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VGA-BenchV2，一个人类对齐的多维度视频生成评估与优化框架

**💡 创新点**

1) 扩大人类标注量，提供 3.47 倍的监督；2) 设计了混合评估网络（VAQA‑Net、VTag‑Net、VGQA‑Net）；3) 将评估结果转化为 RL 奖励，实现闭环优化

**🔧 技术方法**

视频编码器迁移学习、Qwen‑VL 视觉语言模型、强化学习（Flow‑GRPO）与 LoRA 微调

**📊 数据集**

VGA‑Bench 原始数据（60k+ 视频，1,016 题目）+ 新增 36k 任务级人类注释

**📈 对比分析**

对 12 款生成模型进行评估，VAQA‑Net 在整体美学评分上 SROCC 87.6%，VTag‑Net 在标签分类上平均 71.3%，生成质量 31 维子指标平均 71.3%；在 RL 细化后，Wan2.1 的整体美学得分从 0.49 提升至 0.52

**⚠️ 局限性**

受限于标注规模仍不够大、仅覆盖 52 维子指标、RL 训练效率低、对高分辨率视频的适配尚未验证

---

## 250. Continuous Computational Social Choice: A Case Study in Bribery

**arXiv ID:** 2608.25444 | [PDF](https://arxiv.org/pdf/2608.25444v1)

**作者:** Martin Koutecký `[一作]` (Charles University), Lluís Sabater `[通讯]` (Charles University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将传统离散选举模型换成连续型社会模型（无限多微小投票者的分布），并研究该模型下选举攻击（如贿赂、控制）问题的算法复杂度。

**💡 创新点**

创新点在于：①首次从计算社会选择角度引入社会连续模型并分析其对算法复杂度的影响；②利用配置LP和定价问题的逆向“硬度流转”技术，将原本NP‑hard的离散问题转为在连续空间可多项式求解；③给出了多种贿赂攻击在连续模型下的多项式算法和NP/ W[1] 难度结果，并指出该模型下仍有困难的开放问题。

**🔧 技术方法**

核心技术包括配置线性规划（配置LP）框架、列生成与分离定理、动态规划（处理k‑Approval 的定价问题）、以及从定价问题逆向推导硬度的证明技巧；此外使用了LP/整数规划的近似/多项式时间分离与优化等经典理论工具。

**📊 数据集**

本文不使用实验数据，而是完全在理论分析和证明层面工作，没有具体数据集。

**📈 对比分析**

方法比较：将离散模型的已知复杂度（NP‑hard、W[1]‑hard等）与连续模型下的复杂度进行对照；在连续模型下，很多原本NP‑hard的问题（如Borda‑Shift Bribery、k‑Approval‑Swap Bribery 等）变为多项式可解，部分仍保持NP/ W[1] 难度；性能表现以理论计算复杂度评估为主，未涉及实验性能。

**⚠️ 局限性**

局限性：①并非所有攻击问题在连续模型下都变得可解，仍有Borda‑Swap Bribery、一般k‑Approval‑Swap Bribery等保持难度；②对某些规则（如Copeland、Kemeny 等）仍无法适用配置LP方法；③存在未解决的开放问题（如统一成本 Borda 的定价问题、部分规则的连续版本复杂度等）。

---

## 251. Saliency-Depth Conditioning for Zero-Shot Segmentation of Communication-Tower Components in Cluttered UAV Imagery

**arXiv ID:** 2608.25435 | [PDF](https://arxiv.org/pdf/2608.25435v1)

**作者:** Ali Lesani `[一作]` (Computer Vision for Smart Structures Lab), Su-Min Kang `[通讯]` (Soongsil University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在无人机拍摄的通信塔图像中进行无训练的细粒度实例分割。

**💡 创新点**

创新点在于提出了基于显著性与单目深度的前置条件模块，并将其与 Grounded-SAM 与 SAM 3 集成，同时为 Grounded-SAM 添加了几何与深度盒子精炼阶段。

**🔧 技术方法**

所用技术包括显著性检测、单目深度估计、Grounding DINO、SAM/SAM 3 以及形态学和连通分量处理。

**📊 数据集**

实验使用自制的 TOW‑300 数据集，共 340 张高分辨率无人机图像。

**📈 对比分析**

通过与原始 Grounded‑SAM 与 SAM 3 的对比，SD‑SAM 3 在实例召回、平均精度和匹配 IoU 上取得最高成绩，而 SD‑Grounded‑SAM 在精度和语义 IoU 方面表现最佳。

**⚠️ 局限性**

局限性包括深度恢复可能保留非目标结构，盒子精炼可能误删弱目标，以及对显著性/深度模型泛化能力的依赖。

---

## 252. Distance Is Not Enough: Forget-Retain Alignment Gap Predicts LLM Relearning Robustness

**arXiv ID:** 2608.25429 | [PDF](https://arxiv.org/pdf/2608.25429v1)

**作者:** Yi Chen `[一作]` (KAIST), Joo-Young Kim `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了机器模型遗忘后对重学习攻击的鲁棒性，提出了训练无关的Forget‑Retain Alignment Gap（FRAG）预测器和基于稀疏剪枝的Forget‑Retain Pruning（FRP）方法，以提升大型语言模型在遗忘后对重学习攻击的抵抗力。

**💡 创新点**

创新点在于将鲁棒性从全局权重距离转向“权重选择性”视角，提出FRAG衡量更新是否聚焦于忘记关键权重并避开保持关键权重；并基于同一原则实现FRP，通过权重排名进行稀疏剪枝，显著提升遗忘后的鲁棒性。

**🔧 技术方法**

使用权重重要性评估（forget/retain）、余弦相似度对齐、训练无关的FRAG预测器以及按排名进行的稀疏剪枝FRP；在大型语言模型上进行微调、遗忘与重学习攻击实验。

**📊 数据集**

在TOFU、WMDP‑cyber、MUSE‑News等公开遗忘基准上评估；使用LLaMA‑3.2‑1B/3B、Qwen2.5‑14B‑Instruct等模型；采用保留/遗忘/混合数据集进行校准与攻击。

**📈 对比分析**

与GA、GradDiff、NPO、RMU、SP等基线以及全局ℓ₂距离预测器对比。FRAG在所有模型和攻击类型上平均获得最高的ES/ΔES（TOFU）和最低的cyber‑accuracy；FRP在保持相同保留性能的前提下，在重学习攻击下显著提升鲁棒性，构成更优的鲁棒性–效用边界。

**⚠️ 局限性**

仅在TOFU、WMDP‑cyber、MUSE‑News三套基准上验证，缺乏多语言、大规模模型的广泛评估；对密集型遗忘方法的分辨率有限，只能明显区分稀疏与密集更新；实现为无结构剪枝，未探讨结构化剪枝或低秩编辑等更通用形式；实验使用公开数据，未评估对有害内容恢复的可能性。

---

## 253. SUPER ODOMETRY 2.0: Resilient Odometry via Hierarchical Adaptation

**arXiv ID:** 2608.25427 | [PDF](https://arxiv.org/pdf/2608.25427v1)

**作者:** Shibo Zhao `[一作]` (Carnegie Mellon University), Sebastian Scherer `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个名为 Super Odometry 的端到端里程计框架，能够在视觉、激光雷达或两者同时失效的恶劣环境中保持高精度位姿估计。该框架通过分层自适应机制（特征选择、状态方向、引擎切换以及学习式惯性里程计）实现鲁棒性与效率的动态平衡。

**💡 创新点**

① 引入人类路径积分启发的学习型惯性里程计，作为内部运动先验并在外部观测失效时主动接管；② 采用双层互补融合策略，使传统模型优化与深度惯性网络互相学习，提升泛化与在线适应能力；③ 设计分层自适应框架，可根据退化程度从轻量级特征筛选到全深度学习模式逐级切换，实现效率与鲁棒性的协同调节。

**🔧 技术方法**

1) 深度惯性网络（1D ResNet+LSTM+多头输出）；2) 低秩适配器（LoRA）实现少量样本在线微调；3) 观测可观性分析与动态因子图重配置；4) 软融合机制；5) 自监督二阶优化（教师-学生）框架。

**📊 数据集**

大规模多平台数据集：SubT‑MRS、TartanDrive、IDOL、Blackbird、UZH（共计100+小时 IMU+真值），用于预训练与在线适应；评测数据集为 SubT‑MRS（8 条包含几何/混合退化序列）和校园腿式机器人 2966 m 轨迹测试。

**📈 对比分析**

与 RNIN‑VIO、TLIO、IMO、AI‑IMU 等前沿方法在 ATE、时间相对误差（T‑RTE）和鲁棒性指标（R_p、R_r）上进行对比。实验显示 Super Odometry 在 ATE 上平均 0.271 m（比第二佳方法低 54%），在完整 2966 m 试验中漂移仅 0.2 m（0.006%）。在多种退化场景下，鲁棒性 AUC 远超竞品，尤其在浓雾、烟尘等完全退化情况下仍保持可靠估计。

**⚠️ 局限性**

① 仍依赖精确的标定与时间同步，若误差过大会影响自适应阈值；② 虽然使用 LoRA 提升少样本适配，但对极端新平台/极端运动模式的快速收敛仍有挑战；③ 完全失效时仅凭 IMU 估计仍可能产生长时漂移；④ 高级自适应层（深度学习+完整因子图）计算量较大，实时性能受限于硬件。

---

## 254. RotDroid: Cross-Orientation State Equivalence Testing for Detecting GUI Rotation Bugs in Android Apps

**arXiv ID:** 2608.25425 | [PDF](https://arxiv.org/pdf/2608.25425v1)

**作者:** Mengdi Qin `[一作]` (Beihang University), Bo Jiang `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套名为RotDroid的自动化测试框架，用于通过交叉方向状态等价检测识别Android应用中的GUI旋转缺陷。

**💡 创新点**

创新点在于：1）设计State‑Preserving Action Sequence（SPS）变异方法，构造跨方向语义等价的GUI状态；2）构建RotBench数据集并对Qwen3‑VL进行LoRA微调，得到专门用于旋转等价判断的Vision‑Language模型RotVL；3）将SPS、UTG建模与RotVL结合，形成从探索到定位的全流程。

**🔧 技术方法**

技术手段包括：Android UI Transition Graph（UTG）建模、SPS抽取与旋转变异、低秩适配（LoRA）微调Qwen3‑VL、视觉‑语言推理、基于层级匹配的自适应动作执行，以及多阶段任务指令模板。

**📊 数据集**

使用的数据集主要有：1）RotBench（1,818非缺陷对 + 11,233人工合成缺陷对）用于模型训练与评估；2）F‑Droid和Google Play采集的300+开源与103+闭源应用，用于大规模实测；3）自然缺陷集100对真实旋转缺陷截图用于泛化测试。

**📈 对比分析**

对比方法包括：大规模VLM（Qwen3‑VL‑32B/235B）、GPT‑5.2、数据丢失检测工具（iFixDataloss、DLD）以及DOC基线。RotVL‑8B在Bug Detection、Classification、Localization上分别达85.29%、62.68%、79.97%（F1），远超所有基线；RotDroid在真实应用中发现300/168/268真阳性，分别比DLD和DOC多约70%和55%，精度约84%。

**⚠️ 局限性**

局限性：1）合成缺陷样本可能未覆盖所有真实缺陷模式；2）DOC实现与原论文不完全一致，可能导致比较偏差；3）只关注可见GUI状态，无法检测隐藏后台状态或网络数据丢失；4）对动态内容（如计时器、游戏动画）易产生误报；5）仅针对Android生态，其他移动平台未知适用性。

---

## 255. AdaptiveEmbed: Sample-Adaptive Multi-Vector Representation for Multimodal Retrieval

**arXiv ID:** 2608.25412 | [PDF](https://arxiv.org/pdf/2608.25412v1)

**作者:** Xinze Liu `[一作]` (Chinese Academy of Sciences), Weiping Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了样本自适应多向量表示(SAMVR)框架，针对多模态检索中每个样本的表示容量进行动态分配；

**💡 创新点**

创新点在于将表示容量视为样本级决策变量，结合多组对比学习(MGCL)、对称集合相似度(SetSim)以及利用检索反馈的Utility Policy Optimization (UPO)和Marginal Utility Allocation (MUA)实现自适应容量分配；

**🔧 技术方法**

核心技术包括：多组对比学习(MGCL)、对称集合相似度(SetSim)、Token Selection Transformer (TST)、UPO策略优化以及MUA分配策略；

**📊 数据集**

在图像-文本、视频-文本、音频-文本三大类型的检索基准上进行实验，使用COCO、Flickr30K、ADE20K、OpenImages、ActivityNet、DiDeMo、Clotho、MACS等数据集；

**📈 对比分析**

与单向量、固定容量多向量方法(如ColBERT、MetaEmbed等)对比，AdaptiveEmbed在保持平均约2个向量的同时，mAP平均提升约2-3个百分点，且在零样本迁移和不同模态下均表现优于固定容量基线；

**⚠️ 局限性**

局限性包括：仍依赖预训练的检索银行反馈，策略阈值和反馈深度需手工调参；对极大规模数据的实时推理仍有计算开销；且最优自适应分配在理论上尚未达成，Oracle版本显示仍有提升空间。

---

## 256. Retry Amplification in Distributed Systems: A Systematic Analysis of Retry Policies and Their Role in Cascading Failures

**arXiv ID:** 2608.25403 | [PDF](https://arxiv.org/pdf/2608.25403v1)

**作者:** Rishabh Mehan `[一作]`, Jasmit Kaur Saluja `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过分析分布式系统中的重试放大效应，提出RAF指标、系统性评估微服务的重试配置，设计并仿真验证Adaptive Retry Budgeting（ARB）策略。

**💡 创新点**

创新点包括：①引入RAF量化多层重试放大；②系统归纳五种反模式；③提出基于全局预算与背压信号的ARB算法，能在保持恢复能力的同时抑制放大。

**🔧 技术方法**

采用离散事件仿真、正则表达式静态代码分析、GitHub API挖掘、EMA估计与预算动态调整、以及对比实验（No Retry、Standard Retry、Circuit Breaker、ARB）。

**📊 数据集**

数据集为200个受欢迎的Python微服务开源仓库（共113个生产配置）以及仿真中使用的五层链路配置。

**📈 对比分析**

在三种失败场景（S1、S2、S3）下进行100次实验，测量成功率和RAF；ARB在所有场景下成功率与无重试几乎相同（RAF≈1.01），而Standard Retry导致成功率下降25%并将RAF提升至1.34，Circuit Breaker与ARB相当。

**⚠️ 局限性**

局限性：仅针对Python项目，检测器误报/漏报率较高；仿真模型忽略网络延迟、GC停顿等实际因素；未覆盖异步/批处理模式；实验规模局限于5层链路，未验证更深或不同拓扑。

---

## 257. Efficient Training with Foresight: Multi-Token Auxiliary Supervision for Autoregressive Image Generation

**arXiv ID:** 2608.25386 | [PDF](https://arxiv.org/pdf/2608.25386v1)

**作者:** Guo Niu `[一作]` (Foshan University), Nannan Zhu `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MTAR框架，结合多token预测、多层对比正则化和语义丢弃，提升自回归图像生成的质量与训练效率。

**💡 创新点**

①多token预测给出更密集且二维几何一致的监督；②token级对比正则化提升隐藏表示的辨别力；③语义丢弃按重要性剪枝低信息块，仅在训练阶段，无推理开销。

**🔧 技术方法**

自回归Transformer（类似LlamaGen）、多头MTP、Dropout对比正则化、2D RoPE、VQGAN tokenizer、DINOv3/SigLIP2语义评分。

**📊 数据集**

ImageNet 256×256分类条件生成数据集，使用VQGAN 16×16 tokenizer；训练集可为100k样本或完整ImageNet。

**📈 对比分析**

与LlamaGen及多种GAN/扩散/掩码/AR基线对比，评估FID/IS/Precision/Recall；MTAR-B 138M参数在ImageNet上FID 4.50，MTAR-L 387M参数FID 2.85，均优于同规模AR基线；训练速度比LlamaGen快1.27–1.39倍，甚至在1/3或1/6训练迭代下仍优于Baseline。

**⚠️ 局限性**

仅在ImageNet上验证，缺乏跨域或更高分辨率测试；对比正则化依赖采样规模与超参调优；语义丢弃需离线语义评分，增加预处理；推理阶段仍需完整序列，无法进一步加速；未彻底解决全局一致性问题。

---

## 258. Traffic-Adaptive Per-Hop Multipath Routing in Multi-Hop UAV Networks

**arXiv ID:** 2608.25383 | [PDF](https://arxiv.org/pdf/2608.25383v1)

**作者:** Zhenyu Zhao `[一作]` (Beijing University of Posts and Telecommunications), Wenjuan Xing `[通讯]` (Chongqing University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种流量自适应的跳点多路径路由框架，针对多机UAV网络中异构计算任务流量，通过动态分配到多个下一跳节点来提高准时交付率并降低丢包率。

**💡 创新点**

创新点在于将路由问题建模为Dec-POMDP，并设计了MAPPO-DM算法：结合Transformer与GRU对邻居交互和时间演化建模；使用Dirichlet分布实现连续的分流动作；通过Graph Attention Critic获得全局价值估计，实现跨机协同学习。

**🔧 技术方法**

使用的技术包括：多智能体近端策略优化（MAPPO）、Dirichlet分布建模、Transformer编码邻居特征、GRU捕捉时间变化、图注意力网络（GAT）构建集中式Critic、EWMA估计干扰并实现低延迟的邻居信息交换。

**📊 数据集**

实验使用仿真生成的三维UAV网络数据，包含不同角色（热点、网关、中继、常规）节点的任务生成概率、移动模型和链路参数，未使用公开真实数据集。

**📈 对比分析**

与四种基线（单路径贪心、等分多路径、容量感知多路径、I-AOMDV导引多路径）以及消融实验对比。MAPPO-DM在准时交付率上达到约96%以上，丢包率低于0.5%，并在热点数量、网络规模、候选节点数等不同场景下表现出更好的稳健性。

**⚠️ 局限性**

局限性包括：对候选下一跳数量的依赖；在极端拥塞或快速拓扑变化时响应仍有限；需要邻居信息与干扰估计的前置同步；模型训练计算量大，部署时需要足够的计算资源；未考虑能耗、实时同步等实际部署细节。

---

## 259. Q&A or Document-Based? The Effects of Interface Type on How Screen Reader Users Access Interconnected Documents

**arXiv ID:** 2608.25382 | [PDF](https://arxiv.org/pdf/2608.25382v1)

**作者:** Colleen F. Cipriano `[一作]` (University of Victoria), Jaylee Soh `[通讯]` (Singapore Management University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对盲聋及低视力（BLV）屏幕阅读器用户进行实验，比较传统文档浏览界面（DI）与基于大型语言模型的问答界面（QAI）在信息探索、知识建模与应用上的差异。

**💡 创新点**

首次系统评估LLM驱动的对话式问答如何影响BLV用户的知识建模与元认知，提出DI更能促进全面知识获取、QAI更易产生紧凑连贯模型但易误判覆盖度。

**🔧 技术方法**

采用 Gemini 2.5 Pro（通过缓存增量生成）实现 QAI；DI 使用 WCAG 2.1 合规 HTML 文档与 3D 打印触觉覆盖；混合方法收集交互日志、概念图、决策任务和访谈数据。

**📊 数据集**

构建两套虚构世界（Solana、Dominion），每套 25 篇文档（含 8/6 篇空间图），公开共享；数据集主要用作实验材料，未使用公开大规模语料库。

**📈 对比分析**

双盲交叉设计（16 位参与者），量化指标包括独立文档访问量、过渡次数、概念图主题/连接数、错误率；结果显示 DI 在独立访问量、概念图规模和错误率上均优于 QAI，QAI 在图密度和路径长度上更优；两者用户偏好相等。

**⚠️ 局限性**

局限性包括仅涉及屏幕阅读器用户、样本量有限、实验仅两 90 分钟会话、DI 设计包含触觉覆盖和索引页可能影响结果、访问量/过渡计数方法的主观性，以及未检验更长时间或多种盲视接入方式下的表现。

---

## 260. Separating Disclosure from Authorization: Field-Tier Minimization for Agent Action Mediation

**arXiv ID:** 2608.25474 | [PDF](https://arxiv.org/pdf/2608.25474v1)

**作者:** Jiten Oswal `[一作]` (Aurite AI), John Cadeddu `[通讯]` (Aurite AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为自治代理的运行时治理提供一种机制，分离授权评估与审计记录，将原始参数的承诺与最小化处理分离，使用字段级三层分类与投影实现最小化，并通过对原始参数的先验哈希与客体的明文签名，保证链的不可篡改与可审计。

**💡 创新点**

① 在同一系统中实现授权与审计两种需求的可分离性；② 通过字段级三层分类（policy、derived、payload）和投影库实现最小化；③ 先对原始参数计算哈希，使得变更字段分类不影响历史链条；④ 引入“谁计算attested fact”不对称原则，决定哪些信息由客户端还是服务端计算并证明。

**🔧 技术方法**

技术包括：Cedar 策略引擎、RFC 8785 规范的 JSON canonicalization、SHA‑256 哈希、三层字段分类与投影函数（如域提取、布尔允许列表、URL 规范化、路径模板化、目录抽象化）、签名链、离线验证器、基于客户端的 digest 计算与 tier 表的哈希校验。

**📊 数据集**

使用自有的 5 种操作类型共 19 个字段（其中 6 个为 derived，8 个不在原始字段中）的参考语料库进行投影与泄露分析，并在实际私有 pilot 部署中验证实现。

**📈 对比分析**

本文未给出针对性能或对比实验的数值结果；主要通过结构化证明与投影的可验证性说明其安全性，并在实际部署中确认无历史条目被破坏，且离线验证器不需更新。

**⚠️ 局限性**

限制包括：① 客户端可伪造参数 digest 的残余信任；② 无法防御被攻击的工作负载；③ tier 表无法表达上下文相关性；④ 仅在自有语料库上验证，无法保证对其它政策集的适用性；⑤ 未实现对最小化属性的零知识证明，仍需信任客户端按 tier 表执行。

---

## 261. PAGS: Autofocusing Photoacoustic Tomography via Speed-of-Sound-Adaptive Gaussian Splatting

**arXiv ID:** 2608.25472 | [PDF](https://arxiv.org/pdf/2608.25472v1)

**作者:** Jiarui Ge `[一作]` (Shanghai Jiao Tong University), Xiaoyun Yuan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并实现了一种名为PAGS的盲自聚焦光声层析成像框架，该框架通过稀疏高斯光声源与路径平均速度场的联合优化来重建初始压强分布。

**💡 创新点**

创新点在于引入了低维的各向异性路径平均速度场（ASoS）代替传统稠密声速映射，并使用解析的高斯声学投影与Spherical Harmonic探针进行可微优化，实现了无需先验声速信息的自动聚焦。

**🔧 技术方法**

采用的技术包括3D高斯 splatting、Spherical Harmonic（SH）探针编码的ASoS场、解析高斯声学前向模型、Adam优化器以及稠密源密度自适应控制。

**📊 数据集**

使用了两个数据集：一是带有双区域异速的3D血管虚拟模型（使用k-Wave模拟）和一套实物模型（含约5%–8%声速对比的水体与组织区），共计4600个传感器信号。

**📈 对比分析**

通过与传统UBP、双声速UBP（已知界面）以及原始SlingBAG进行对比，PAGS在模拟数据上实现PSNR 30.3 dB、RMSE 0.0305、SSIM 0.537，显著优于SlingBAG的PSNR 29.0 dB、SSIM 0.331，并在物理模型上展现出更锐利的血管结构和在稀疏采样下的鲁棒性。

**⚠️ 局限性**

局限性包括仅在仿真与实验模型上验证，未涉及复杂多层组织或真实病灶；ASoS场为路径级近似而非物理声速映射，可能对极端非均匀介质的适用性有限。

---

## 262. Homo-RAG: Homology-Guided Retrieval-Augmented Generation for Cross-Species Gene Function Prediction

**arXiv ID:** 2608.25466 | [PDF](https://arxiv.org/pdf/2608.25466v1)

**作者:** Azrin Sultana `[一作]` `[通讯]` (American International University-Bangladesh), Azrin Sultana (American International University-Bangladesh)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了 Homo‑RAG 框架，利用同源关系进行三跳检索和证据排序，辅助大型语言模型生成基因功能预测。

**💡 创新点**

创新点：同源引导的三跳检索、基于来源可靠性和跨跳一致性的证据置信度评分（ECS）以及结合检索相似度的加权 reranking。

**🔧 技术方法**

技术：稠密 + 稀疏检索（BM25 与 S‑PubMedBERT‑MS‑MARCO）、Hybrid 检索、ECS 加权 reranking、轻量级指令遵循 LLM（Phi‑3.5‑mini‑Instruct 等）和 RAG 生成。

**📊 数据集**

数据集：ZFIN 基因信息、UniProt 人类蛋白注释、PubMed 文献，实验包含 150 个查询和 7200 条候选 evidence。

**📈 对比分析**

对比基线（BM25、FAISS、Hybrid、无 ECS 等）后，Homo‑RAG 在 P@10 0.886、MRR 0.990、NDCG@10 0.988；Phi‑3.5‑mini‑Instruct 生成质量 BERTScore 0.87，生成成功率 100%。

**⚠️ 局限性**

局限：仅采用单一证据特征，未利用图嵌入或学习型排序；模型规模受限于开源 LLM，需进一步扩展多维特征和跨源一致性验证。

---

## 263. Automatic weld seam segmentation for industrial quality control: a comparison of RGB and polarimetric imaging with CNN and transformer architectures

**arXiv ID:** 2608.25465 | [PDF](https://arxiv.org/pdf/2608.25465v1)

**作者:** Simone Garbin `[一作]` (Fraunhofer Italia Research), Marco Todescato `[通讯]` (Fraunhofer Italia Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在受控实验室、工业现场和偏振多通道成像条件下，对焊缝实例分割的可行性进行系统评估，并比较不同深度学习架构（CNN vs Transformer）的性能。

**💡 创新点**

提出统一的阈值无关评估框架、三种随机种子重复实验、偏振多图像与几何增强策略的实证，并首次展示 Transformer 在视角变换下对工业焊缝检测的显著优势。

**🔧 技术方法**

使用 YOLOv8/YOLOv11 单阶段 CNN、RF‑DETR‑Seg 与 Mask2Former Transformer；采用 PolarSens 偏振六通道、单图/多图、通道融合与几何增强等策略；在 COCO mAP50/95 评估下进行比较。

**📊 数据集**

三类 RGB 数据集（受控实验室、工业现场）与 PolarSens 偏振多通道数据集（共约 200 张图），每类按物理焊点划分；附加 62 张近距离测试集，用于零射向域移评估。

**📈 对比分析**

统一低阈值（0.01）COCO mAP50/95 评估；每个 CNN 训练三次随机种子并取平均；在受控 RGB 下 CNN mAP50 达 0.79–0.87，工业 RGB 仅 0.22–0.48；偏振多图+几何增强可达 0.93；Transformer 在零射向近距离视角下平均 mAP50 0.6–0.8，显著高于 CNN；说明数据采集条件与模态是性能瓶颈。

**⚠️ 局限性**

数据集极小且测试集仅 5–15 张图，导致单样本波动大；Transformer 仅单跑，缺乏多种子验证；Transformer 训练受 GPU 内存限制在低分辨率；未对在线嵌入硬件进行推理测试；仅评估焊缝定位，未涉及缺陷分类；通道融合策略在本数据上效果不佳。

---

## 264. Towards safe and optimal flight: Viability Kernel MPC for Fully Actuated Multirotor

**arXiv ID:** 2608.25459 | [PDF](https://arxiv.org/pdf/2608.25459v1)

**作者:** Massimiliano Bertoni `[一作]` (University of Padova), Giulia Michieletto `[通讯]` (University of Padova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于可生存理论的终端约束MPC框架，并利用神经网络实时评估可生存核，实现全主动多旋翼无人机在拥挤环境中的安全姿态轨迹跟踪。

**💡 创新点**

创新点在于将可生存核数值近似与MPC终端约束相结合，采用贪婪Axis‑Aligned Bounding Box算法实时更新障碍约束，并通过神经网络压缩可生存核，避免昂贵的离线可达性分析。

**🔧 技术方法**

使用技术包括可生存理论、VBOC、神经网络回归、AABB贪婪收缩、模型预测控制、CasADi/ACADOS优化求解。

**📊 数据集**

使用200,000条通过随机采样AABB参数、初始姿态和方向生成的可生存核数据训练两层512单元的神经网络。

**📈 对比分析**

在α‑Ted6R六旋翼平台上进行5 s飞行仿真，平均MPC迭代时间1 ms（最高11.7 ms），成功避障并到达目标，显示出实时性和安全性优于传统短期碰撞避免方法。

**⚠️ 局限性**

局限性包括仅考虑静态障碍、AABB近似保守、未处理动态障碍、神经网络误差可能影响可生存核评估，以及贪婪收缩不保证全局最优。

---

## 265. E2-Conditioned Finite-Horizon Effective Capacity for Public-Safety MCX over Shared O-RAN

**arXiv ID:** 2608.25442 | [PDF](https://arxiv.org/pdf/2608.25442v1)

**作者:** Jingqing Wang `[一作]` (Xidian University), Wenchi Cheng `[通讯]` (Xidian University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对共享O-RAN环境下的公共安全Mission Critical Services (MCX) 提出了一套基于E2观测的有限时隙有效容量（FH-EC）框架，并在此基础上构建了可校准的执行配置库，最终实现了在Near‑RT RIC中的高效配置编排与资源管理。

**💡 创新点**

创新点包括：① 引入E2感知的有限时隙有效容量，显式考虑观测与控制执行延迟对短期服务可达性的影响；② 将多QoS风险（速率、延迟、抖动、AoI、可靠性）与FH-EC结合，形成统一的风险‑能力指标；③ 通过统计校准（Hoeffding、Clopper–Pearson）为每个执行配置提供置信下界/上界，从而实现可信的配置选择；④ 设计了离线预估+在线0‑1规划的两阶段框架，显著降低运行时计算复杂度。

**🔧 技术方法**

技术手段包括：有限状态马尔可夫加性服务模型、有限时隙有效容量解析、置信度校准（Hoeffding、Clopper–Pearson）、多目标0‑1线性规划、O‑RAN架构与Near‑RT RIC控制、MATLAB+ns‑3+5G‑LENA仿真。

**📊 数据集**

实验使用基于ns‑3 5G‑LENA仿真生成的网络状态与流量数据，并通过不同E2曝光配置（rich、moderate、sparse）模拟E2观测与控制延迟；未使用真实网络数据集。

**📈 对比分析**

对比方法包括：静态预留（Static‑QPP）、长期有效容量（LT‑EC）、仅单链（FH‑SC）、SNC延迟预留（SNC‑Delay）以及全知Oracle。实验表明，E2‑FH框架在短期窗口内显著提升MCX可支持率，满足多QoS约束，且对非MCX用户的干扰低，最高可支持50个并发MCX用户；在O‑DU降级时表现更稳健。

**⚠️ 局限性**

局限性：① 依赖E2观测的准确性和时延，观测误差可能导致能力下滑；② 采用离散时间马尔可夫模型，可能无法捕捉某些高速时变或非马尔可夫行为；③ 需要大量离线仿真与校准，扩展到大规模网络时计算与存储开销待评估；④ 对极端网络状况（如大规模拥塞或链路失效）的鲁棒性尚未充分验证。

---

## 266. BVR Sim: An Open and High-Throughput Environment for Heterogeneous Air-Combat Reinforcement Learning

**arXiv ID:** 2608.25419 | [PDF](https://arxiv.org/pdf/2608.25419v1)

**作者:** Haocheng Sun `[一作]` (Beijing University of Posts and Telecommunications), Mulai Tan `[通讯]` (Air Force Engineering University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

构建了一个可扩展、支持异构机型、可高吞吐量的 BVR 空战仿真环境 BVR Sim，提供统一的高层行动接口、实体化观测、可插拔 Python/C++ 后端，并集成多代理学习框架。

**💡 创新点**

首次在 BVR 仿真中实现了跨机型共享高层行动接口、实体化观测格式、双后端架构和可与标准 MARL 框架无缝对接的适配器；并证明了跨机型政策迁移的可行性。

**🔧 技术方法**

采用 JSBSim 飞行动力学、C++ 实现的高性能后端、Gymnasium 接口、实体化表观测、可组合奖励、脚本化对手、OpenGL/Tacview 可视化，支持 PPO/MAPPO/HAPPO 等 MARL 算法。

**📊 数据集**

使用公开的 JSBSim 机型（F‑15/F‑16/F/A‑18/F‑22 等）以及自定义的场景 JSON 配置，包含多机型、不同武器配置、雷达参数等；无外部标注数据集，仅依赖自定义仿真场景。

**📈 对比分析**

通过对比 Python 后端和 C++ 后端的吞吐量实验，1v1 时 C++ 后端可达 104×仿真时间/墙钟时间，整体加速 2.7–6.6×；在 2v2、4v4、6v6、8v8、10v10 场景中保持可接受吞吐；在跨机型政策迁移实验中，冻结 F‑16 策略在其他机型上可获得约 45% 胜率，改为对应机型控制器提升至约 80%，验证了跨机型迁移效果。

**⚠️ 局限性**

局限包括：物理模型简化（缺乏电磁战、通信、天气、地形掩护等真实战场因素）；高层接口未实现目标与武器分配学习；实验样本有限（单种子、未做对比评估）；且不同后端可能存在数值差异。

---

## 267. LAC: Linear and Angular Compliance for Humanoid Whole-body Control

**arXiv ID:** 2608.25405 | [PDF](https://arxiv.org/pdf/2608.25405v1)

**作者:** Yang Liu `[一作]` (Tohoku University), Mitsuhiro Hayashibe `[通讯]` (Tohoku University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种名为LAC的全身控制器，能够根据命令实现人类上身多点外力的线性与角度可调顺应性；

**💡 创新点**

①同时实现全身线性与角度顺应性指令；②利用大规模合成运动数据（基于人类交互轨迹加虚拟阻抗）生成可行的全身顺应性样本；③采用教师–学生强化学习将全身顺应性映射到单一策略；④通过被动链条旋转估计实现可物理可行的角度顺应性；

**🔧 技术方法**

虚拟阻抗/质量系统、增量逆运动学、PPO强化学习、教师–学生两阶段架构、Isaac Lab/MuJoCo仿真、Mink等IK求解器；

**📊 数据集**

通过重定向OMOMO与Inter-X交互数据提取接触帧，生成378,051条合成片段（约1,050小时）包括力事件与耦合事件，并做可行性检验；

**📈 对比分析**

与SoftMimic、GentleHumanoid、FALCON进行对比。LAC在模拟与真实机器人上展示更宽的顺应范围、位移随刚度单调递减，且在多种任务（力峰值、姿态改变、平衡保持）中表现出更优的整体平衡与任务适应性；

**⚠️ 局限性**

目前刚度命令由操作者手动设定；缺乏自适应或即时外力估计；仅对上身进行顺应控制，腿部仅通过基座速度/高度间接调节；未利用触觉信息；未验证在不同移动基座（如四足或履带）上的迁移性。

---

## 268. Lightweight AI for UAV-Mounted RIS: An Overview

**arXiv ID:** 2608.25402 | [PDF](https://arxiv.org/pdf/2608.25402v1)

**作者:** Sherief Hashima `[一作]` (RIKEN-AIP), Hamada Rizk `[通讯]` (University of Osaka)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了无人机搭载可重构智能表面（UAV‑RIS）系统中的轻量级人工智能技术，并通过多臂赌博机（MAB）案例研究展示了其在路径规划与能效优化中的应用。

**💡 创新点**

首次将轻量级 AI 技术分类并与 UAV‑RIS 架构关联，提出多维度比较框架，并在 MAB 案例中证明上下文 MAB 在吞吐量与能效方面可达到 90% 最优。

**🔧 技术方法**

采用模型压缩（剪枝、量化、蒸馏）、TinyML、RL/DRL、Meta‑Learning、MAB、Federated Learning、Sparse GNN、Edge Computing 与深度展开等技术。

**📊 数据集**

主要使用仿真生成的数据集，针对毫米波热点场景的随机位置与流量需求进行模拟；未使用公开真实数据集。

**📈 对比分析**

通过与 UCB、TS、最近热点与随机策略比较，CTS 方案在吞吐量上达到 90% 以上最优，能效提升数倍；与 DRL 方案相比，MAB 方案在计算复杂度与能耗上显著下降。

**⚠️ 局限性**

局限在于仅在仿真环境下验证，缺乏实际部署实验；对大规模 RIS、移动热点多样性、干扰与安全性等场景的鲁棒性仍待进一步研究。

---

## 269. Here is a GIFT: Enforcing User Data Isolation in LLM Serving via GPU Information Flow Tracking

**arXiv ID:** 2608.25431 | [PDF](https://arxiv.org/pdf/2608.25431v1)

**作者:** Jiacheng Shi `[一作]` (Shanghai Jiao Tong University), Jinyu Gu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 GIFT（GPU Information Flow Tracking）系统，在 LLM 服务器中通过 GPU 级别的所有权追踪实现用户数据隔离，并通过加密隔离（encryption‑as‑isolation）实现对 CPU 端框架的无侵入性支持。

**💡 创新点**

创新点包括：① 在 CPU 侧仅使用加密做隔离，避免对快速演进的 LLM 框架进行复杂改造；② 预先离线分析 GPU 核心，生成可直接在 CPU 上执行的所有权传播规则，消除传统 GPU taint 跟踪的指令级开销；③ 细粒度段级所有权记录、并发控制和 KV 缓存安全交换空间等机制，实现近零运行时开销。

**🔧 技术方法**

核心技术包括：GPU 信息流跟踪（IFT）、加密隔离、基于符号执行的 IFT 规则验证、Rust 编写的安全监控器、CUDA API 重定向、NVIDIA Confidential Computing（TEE）等。

**📊 数据集**

使用的模型与基准包括 Qwen‑2.5（14B/32B/72B）、Llama‑3、Gemma‑2、OPT（13B/30B/66B）、GPT‑2、Phi‑3；评测基准为 ShareGPT、HumanEval、LongBench。

**📈 对比分析**

与未加密的 vLLM/DistServe 基线对比，GIFT 在吞吐量上增加 4%–10.7%（取决于 GPU 与启用的 TEE），但延迟几乎不变；在所有权规则生成、并发控制等优化后，CPU 开销 <124% 核心、内存 <130 MB。

**⚠️ 局限性**

局限性包括：① 依赖 CPU 框架不直接修改用户数据；② 某些 GPU 核心无法提前分析时只能使用黑盒抽样，可能降低安全性；③ 规则生成仍需人工半自动支持；④ 在非 TEE 模式下，仍受操作系统/虚拟机安全性限制；⑤ 目标 GPU 只能是支持 NVIDIA‑CC 的型号。

---

## 270. Towards Faithful and Efficient Semantic Communication: An Ontological Approach

**arXiv ID:** 2608.25422 | [PDF](https://arxiv.org/pdf/2608.25422v1)

**作者:** Yixiao Feng `[一作]` (Academy of Military Sciences), Bo Zhang `[通讯]` (Academy of Military Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于本体的多视角语义通信框架，实现了语义信息的高效压缩与传输。

**💡 创新点**

创新点在于使用共享本体定义同义词、推理规则和一致性约束，实现可恢复的冗余消除、跨视角语义去歧义与一致性验证。

**🔧 技术方法**

利用视觉语言模型提取场景图、基于本体的推理规则、同义映射、一致性约束以及边界框关联的交叉验证技术。

**📊 数据集**

在GQA数据集的验证集上进行实验。

**📈 对比分析**

与完整场景图、任务自适应过滤、关系过滤以及字符串匹配四种基线比较，实验显示SI大小最高可减87.1%，答案准确率提升4.5%，多视角VQA准确率提升16%，且传输延迟和计算开销显著降低。

**⚠️ 局限性**

局限性包括对预先构建的本体依赖、对边界框精度敏感、未考虑动态场景和非结构化语义信息等。

---

## 271. Paint What You See: Benchmarking Dexterous Visual Tool Use in Multimodal Agents

**arXiv ID:** 2608.25417 | [PDF](https://arxiv.org/pdf/2608.25417v1)

**作者:** Shudong Liu `[一作]` (Peking University), Lewei Lu `[通讯]` (SenseTime Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了EASEL基准，用闭环参数化视觉动作评估多模态代理的精细视觉工具使用能力。

**💡 创新点**

创新点在于将参考导向的绘画任务与语义任务相结合，提供闭环参数化动作的评估框架，揭示模型在闭环反馈中的瓶颈。

**🔧 技术方法**

采用基于Bezier曲线的参数化绘画接口、两阶段SFT训练以及LoRA微调，结合多模态LLM与可视化渲染器。

**📊 数据集**

使用了EASEL-Data（约44万条轨迹样本）和参考图像集（11k），以及构建的两个任务类别。

**📈 对比分析**

通过对25款多模态代理的结果质量和轨迹诊断进行对比，发现顶尖闭源模型最终相似度仅0.535，EASEL-9B提升6.3%至0.459，显示大多数模型在闭环中早期饱和或后期退化。

**⚠️ 局限性**

限制在于任务仍局限于2D画布，缺乏更复杂的物理或3D交互，且轨迹监督在长时延反馈上的改进有限。

---

## 272. Can your AI agent be cheaper? Investigating the effects of task specifications on token spend in agentic coding tasks

**arXiv ID:** 2608.25399 | [PDF](https://arxiv.org/pdf/2608.25399v1)

**作者:** Jakub Smékal `[一作]` `[通讯]` (Stanford University), Jakub Smékal (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了在 Agentic 编码工作流中，任务说明的不同版本和思考努力水平如何影响 Kimi K3 模型的 Token 消耗，并提出了基于一次低成本探测运行的成本预测方法。

**💡 创新点**

创新点在于：①系统量化任务说明对平均 Token 消耗的可控影响；②展示不同提示对方差无显著作用；③引入仅需一次探测运行即可预测未见任务成本分布的简易预测器。

**🔧 技术方法**

主要技术包括：使用 Kimi K3 进行 2,700 次 Agentic 任务跑测；采用 Bayesian 层级模型估计提示与思考努力对 Token 与 turns 的影响；构建基于探测运行的对数成本预测公式。

**📊 数据集**

数据集为 SWE‑Bench Verified 的 5 个编码任务，针对每个任务构造 12 种提示变体（10 种中间版本 + 2 边界版本），每种提示在 3 个思考努力层级下重复 15 次。

**📈 对比分析**

通过对平均 Token 消耗、turns 与方差的比较，提示的平均消耗变化可达 13%–115%；预测实验显示，在无探测时误差约 161%，而一次 0.11 USD 的探测能将误差降至 36%，预测相关系数最高可达 0.72，成本超限倍数约 1.9×。

**⚠️ 局限性**

局限性包括：仅在单一模型 Kimi K3 上验证；任务数量有限，未覆盖更广泛的真实工作流；探测方法未考虑任务特定的提示敏感性，可能影响跨任务泛化。

---

## 273. OmniPhys: A Unified Multimodal Benchmark for Physics Understanding and Generation from Chinese Educational Corpora

**arXiv ID:** 2608.25398 | [PDF](https://arxiv.org/pdf/2608.25398v1)

**作者:** Hao Chen `[一作]` (East China Normal University), Min Zhang `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OmniPhys，一个覆盖初中至大学物理的多模态基准；

**💡 创新点**

创新点在于同时评估理解与生成（物理图解编辑），并设计双轨推理评估框架；

**🔧 技术方法**

采用LLM-as-a-Judge、对抗筛选、深度推理的多模态评估方法；

**📊 数据集**

使用来自中国教材与考试的15,246道题、19,850张图像，包含细粒度推理注释；

**📈 对比分析**

与现有公开和专有MLLM进行对比，领先模型在严格掌握率仍低于70%，显示显著挑战；

**⚠️ 局限性**

局限在单语言（中文）数据、评估判定对多模态输出的过度乐观，以及缺乏规模化的错误分析。

---

## 274. A Taxonomy of Construction Task Activities for Robot Workers

**arXiv ID:** 2608.25395 | [PDF](https://arxiv.org/pdf/2608.25395v1)

**作者:** Sadman Sakib `[一作]` (University of California-Irvine), Mohammad Abdullah Al Faruque `[通讯]` (University of California-Irvine)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于O*NET任务的施工行业动作词典TARCAT，并在DOBOT CR3手臂上实现了部分动作原语与技能示例。

**💡 创新点**

提供了面向施工工作的41个可组合动作原语和参数化技能的体系，填补了缺乏可复用、跨职业动作库的空白。

**🔧 技术方法**

采用文本分析与视频标注相结合的方法，利用ChatGPT检索和标注YouTube教学视频，并定义动作原语与技能序列。

**📊 数据集**

选取美国劳工部O*NET和BLS就业数据对应的七个高就业建筑职业，收集169条任务说明并标注相应的视频。

**📈 对比分析**

本工作未与其他系统进行数值比较，而通过手工验证四个动作在DOBOT手臂上的可执行性来说明词典可实现性，尚无客观性能评估。

**⚠️ 局限性**

仅覆盖七个职业且每项任务示例有限，缺乏足够的演示数据和实验验证，未来需扩展语料、评估一致性并实现更多机器人平台。

---

## 275. MOTIF: Motivation-guided Topology Inference for Cold-start Multimodal Recommendation

**arXiv ID:** 2608.25381 | [PDF](https://arxiv.org/pdf/2608.25381v1)

**作者:** Yurui Shi `[一作]` (Taiyuan University of Technology), Chang Han `[通讯]` (Northeastern University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于LLM推理的动机推理与拓扑重建框架MOTIF，用于冷启动多模态推荐。

**💡 创新点**

创新点在于将LLM生成的动机语义离线转化为可迁移的项间拓扑，并通过语义-结构对齐与加权图对比学习实现鲁棒表示。

**🔧 技术方法**

使用LLM进行动机推理、文本编码、知识增强图重建、加权图对比学习、语义-结构对齐等技术。

**📊 数据集**

在Amazon-Baby、Amazon-Sports和MicroLens-50K三大多模态冷启动数据集上进行实验。

**📈 对比分析**

与图协同过滤、图对比学习、多模态、冷启动和LLM增强基线对比，Recall@20/NDCG@20提升约5‑6%，在极冷用户、冷用户和冷项目上分别提升20‑26%。

**⚠️ 局限性**

限制在于需要离线LLM推理和图重建的额外预处理开销，且对LLM质量和推理策略敏感，无法实时更新。

---

## 276. Bootstrapping a 4D LiDAR Annotation Tool from Video Foundation Models

**arXiv ID:** 2608.25418 | [PDF](https://arxiv.org/pdf/2608.25418v1)

**作者:** Jihun Kim `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了LiDAR‑SAM2，一种无需人工 LiDAR 标注的交互式 4D LiDAR 分割框架，利用 2D 视频基础模型 SAM2 自动生成时空一致的伪标签并进行训练。

**💡 创新点**

创新点在于把 2D 视频分割模型迁移到 4D LiDAR 领域：① 通过多视角投影和时空聚合自动生成稠密、连贯的 LiDAR 伪标签；② 设计了几何感知的范围视图（RV）映射和两阶段学习目标，使 SAM2 的 2D 语义特征在 LiDAR 空间保持一致并实现时间传播。

**🔧 技术方法**

核心技术包括：SAM2 交互式分割、几何编码器 E_geo、MIM（LoRA）调优、范围视图投影、时空 4D 伪标签聚合、以及轻量化的 3D 细化解码器 D_ref。

**📊 数据集**

在 SemanticKITTI 数据集上进行评估，使用同步的 64 层 LiDAR 扫描与前向双摄像头视频进行伪标签生成与模型训练。

**📈 对比分析**

与传统基于手工标注的模型比较，LiDAR‑SAM2 在语义 mIoU（最高 84.3%）和 4D 视域 LSTQ（最高 70.5%）上接近完整人类标注，且在交互式标注任务中仅需十个点提示即可实现高质量分割，显著降低标注成本。

**⚠️ 局限性**

局限性包括：① 伪标签生成对摄像头视野受限，仍需时空聚合提升覆盖率；② 需要高质量的摄像头- LiDAR 校准与同步；③ 对极端环境（如低光、遮挡）下的 SAM2 迁移效果尚未完全验证。

---

## 277. MACGen: Toward Functionally Correct and Secure Code Generation via Multi-Agent Collaboration

**arXiv ID:** 2608.25457 | [PDF](https://arxiv.org/pdf/2608.25457v1)

**作者:** Miseon Yu `[一作]` (Seoul National University), Yunheung Paek `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多代理框架，分阶段规划功能需求、分析安全威胁、生成安全代码并逐步审查，以在生成的代码中同时满足功能正确性和安全性。

**💡 创新点**

创新点在于：① 将安全代码生成拆解为四个专责代理（Planner、Security Advisor、Code Generator、Reviewer）并通过“artifact‑only”接口严格隔离角色；② 采用基于标准（CERT/OWASP）的安全知识库进行检索，生成任务专属安全准则；③ 通过早期排除、代码层面分析和验证步骤构建多阶段安全顾问。

**🔧 技术方法**

主要技术包括：多代理交互（每个代理只接收上游生成的结构化 artifact），语义检索（向量搜索）、规范驱动的安全准则生成、基于 LLM 的代码生成与自我校验，以及结构化审查反馈。

**📊 数据集**

使用的评测数据集有 CWEval（119 题，涵盖 C/C++/Python/JavaScript/Go）和 BaxBench（392 题，支持 6 种语言与多框架的后端应用场景）。

**📈 对比分析**

与 Direct Prompting、SecGuide、CodeGuarder、RESCUE、INDICT 等基线对比；在 CWEval 上，平均提升 F&S@1 约 19.6pp；在 BaxBench 上提升 10.6pp；在多语言、多模型（GPT‑4o、GPT‑4o‑mini、Gemini、DeepSeek 等）上表现出一致的功能-安全双向提升，且 token 成本仅略高于最优基线。

**⚠️ 局限性**

限制主要包括：① 多代理架构导致额外推理成本；② 评测只覆盖特定 CWE 目标，未能完全体现安全与功能之间的细粒度权衡；③ 依赖 LLM 生成安全准则，缺乏外部静态分析或渗透测试等补充验证手段。

---

## 278. MathAdv: What Theorem Provers Know, Reason, Formalize, and Generalize

**arXiv ID:** 2608.25449 | [PDF](https://arxiv.org/pdf/2608.25449v1)

**作者:** Jiaxin Yuan `[一作]`, Furong Huang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 MathAdv 的正式数学推理诊断基准，包含 321 道问题，覆盖 13 个本科到研究生级别的数学领域，并提供 Lean 4 定理证明、填空、选择题和专家改写等多种辅助任务。

**💡 创新点**

创新点在于：① 通过多任务诊断细化模型能力评估，区分知识、推理、正式证明和鲁棒性；② 覆盖广泛、欠代表性领域（拓扑、傅里叶分析等）；③ 设计专家手工改写版本，检验对等式改写的鲁棒性；④ 使用 LLM‑辅助、验证器‑循环的人机交互自动化推导 Lean 4 表达式。

**🔧 技术方法**

采用 Lean 4 证明助手、LLM（如 GPT‑5.4、DeepSeek‑V3.2）进行自动化推导，使用 verifier‑guided 搜索、全局/步骤生成策略，以及多轮交互式错误反馈；还结合自然语言提示和多项选择答案作为推理提示。

**📊 数据集**

主要数据集为 MathAdv 本身，其中 298 题已转化为 Lean 4，另有 23 题留作辅助；每题对应最多三种辅助任务（填空、选择、改写）。

**📈 对比分析**

通过对比多种模型（一般 LLM、定理证明专用模型）在 Lean 4 证明、直接回答、选择题以及改写版的准确率来评估；结果显示：定理证明专用模型在 Lean 4 证明上略优，但在自然语言回答和改写鲁棒性上表现差；大多数模型在改写版上显著失效，说明对表述敏感；整体证明成功率低，最高约 21.9%。

**⚠️ 局限性**

限制：① 数据量相对有限，需大量专家校对；② 受 Mathlib 覆盖范围限制，部分高阶题无法立即形式化；③ 评估依赖 Lean 4 生态，若生态更新会影响结果。

---

## 279. Joint Initialization of Flux Networks and Effective Multiplication Factor for Physics-Informed Neural Networks Solving Neutron Diffusion Problems

**arXiv ID:** 2608.25443 | [PDF](https://arxiv.org/pdf/2608.25443v1)

**作者:** Qin Hang `[一作]` (Chongqing University of Posts and Telecommunications), Heng Zhang `[通讯]` (Chongqing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出联合初始化物理信息神经网络（JI-PINN），通过低分辨率近似解同时初始化通量网络与 k_eff，显著降低训练时间。

**💡 创新点**

创新点在于将通量网络和 k_eff 的初始化信息统一来源于同一低分辨率 K-特征解，实现联合优化，提升收敛速度与稳定性。

**🔧 技术方法**

采用物理信息神经网络（PINN）框架、低分辨率离散化求解、预训练与物理约束损失的联合优化。

**📊 数据集**

使用四个基准问题：二维两组两材料、IAEA 2D 基准、二维两组四材料以及三维单组立方体，作为实验数据集。

**📈 对比分析**

与随机初始化 PINN 和 R²-PINN 对比，JI-PINN 在所有测试中将总计算时间分别降低 25.4%–49.4%，同时保持相近的 k_eff 与通量误差，且随机种子波动更小。

**⚠️ 局限性**

局限性在于对低分辨率近似解质量高度依赖，复杂材料界面或强局部通量变化时可能保留误差，导致最终精度略逊于传统 PINN。

---

## 280. DCGC: Draft-Conditioned Global Correction for Complex Reasoning with Masked Diffusion Models

**arXiv ID:** 2608.25428 | [PDF](https://arxiv.org/pdf/2608.25428v1)

**作者:** Minhae Oh `[一作]` (Seoul National University), Jungwoo Lee `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DCGC 框架，利用 Masked Diffusion Model（MDM）对由上游解算器产生的错误推理轨迹进行全局纠正。

**💡 创新点**

创新点在于：① 采用混合格式监督微调（SFT）让模型同时学习仅基于问题的解答与基于问题+草稿的纠正；② 引入 Dynamic Dual‑CFG，在推理时将问题与草稿分别作为独立分支，并用相对置信度差动态调节草稿残差的力度。

**🔧 技术方法**

主要技术包括 Masked Diffusion Model、Classifier‑Free Guidance（CFG）、Dynamic Dual‑CFG、混合格式 SFT、LoRA 参数适配，以及基于置信度的自适应缩放。

**📊 数据集**

使用的数据集有：数学推理（GSM8K、MATH‑500）、代码生成（MBPP、HumanEval）、知识推理（MMLU‑STEM、MMLU‑Pro），以及从预训练 LLaDA 等模型生成的错误草稿作为训练和评估样本。

**📈 对比分析**

与传统自回归 Self‑Refine 以及其他 MDM 变体进行对比；DCGC 在六项基准上平均提升约 24.8%，在 GSM8K 取得 44.9%，MATH 22.3%，HumanEval 13.1%，MMLU‑STEM 35.7%，显著优于静态/独立 CFG 或单条件指导的方案。

**⚠️ 局限性**

局限性包括：因训练资源受限，对输入长度做了 1,028 词的过滤，难以处理超长上下文；金标无关的纠错采用简单的自一致性作为不确定性指标，尚可通过更高级的触发策略提升效率与可靠性。

---

## 281. Data-driven Effective Modeling of Stochastic Chemical Reaction Networks

**arXiv ID:** 2608.25421 | [PDF](https://arxiv.org/pdf/2608.25421v1)

**作者:** Yuan Chen `[一作]` (Ohio State University), Dongbin Xiu `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于数据驱动的有效模型，通过训练生成式机器学习模型近似SSA的有限时间转移核，并在用户定义的粗时间步长下递归生成统计一致的轨迹，显著降低计算成本。

**💡 创新点**

创新点在于将条件正则化流（conditional normalizing flow）作为随机传播器直接逼近连续时间马尔可夫链的转移核，既实现了对SSA轨迹的高效粗粒度模拟，又保持了高统计精度。

**🔧 技术方法**

使用的技术包括短时SSA模拟数据采集、条件正则化流（conditional normalizing flow）生成模型、以及对有限时间转移核的直接拟合。

**📊 数据集**

利用多组数值实验数据集，涵盖不同规模和复杂度的化学反应网络，用于训练和验证模型性能。

**📈 对比分析**

与传统SSA对比，实验表明新方法在保持相似统计特性的同时，将模拟时间和计算成本降低数倍以上，且在多种测试场景中表现出良好的精度与效率。

**⚠️ 局限性**

局限性包括对极高维度或高度稀疏/强相关系统的泛化能力可能不足，需要充足且多样化的训练数据；此外，模型训练过程可能需要显著的计算资源。

---

## 282. SPFR: Semantic Potential Field Routing for the Distributed Internet of Agents

**arXiv ID:** 2608.25396 | [PDF](https://arxiv.org/pdf/2608.25396v1)

**作者:** Yeguang Qin `[一作]`, Ming Zhao `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一种名为Semantic Potential Field Routing (SPFR) 的分布式任务路由算法，能够在无中心化控制的Internet of Agents（IoA）网络中，在每个转发节点使用本地语义信息对可用执行器进行动态发现与重新选择，并将任务逐跳转发至最优执行器。

**💡 创新点**

创新点在于将执行器发现与路径选择完全融合到“语义势场”框架中：每个可见执行器被视为潜能源，其强度由任务匹配度、负载与价格等多维度权重决定，并随跳数指数衰减；转发节点在每一步都重新计算势场，动态选取主导执行器并前进，从而实现局部信息下的全局近似最优路由，且在理论上保证无环性与有限终止，并给出半视图误差界限。

**🔧 技术方法**

核心技术包括：①语义匹配与Hungarian 算法实现需求‑能力一一对应；②语义潜能场（U·exp(-ω_h·h)）以及指数衰减；③基于本地FIB的有限视图传播；④基于请求‑结果回退的回退查询；⑤实验评估中使用的Python分布式仿真框架。

**📊 数据集**

实验数据集基于三种真实网络拓扑：GEANT、UNINETT 和 Deltacom，每个拓扑均配置数十至上百个节点，随机生成包含多种语义能力与负载特征的服务端点，并在模拟中注入动态队列、服务失效、链路扰动等事件。

**📈 对比分析**

SPFR 与 RAND、D-SEM、D-GREEDY、GLOBAL 等四种基准算法在同一任务与网络条件下进行对比。结果显示，SPFR 在保持与全局发现（GLOBAL）相当的任务效用（约 97%）的同时，平均减少 81 倍的请求触发消息；相较于 D-GREEDY，SPFR 任务成功率略高，前进跳数、延迟与通信成本均降低 30% 以上；在动态与高压状态下表现更为稳健。

**⚠️ 局限性**

主要限制包括：①当源节点无正势能执行器时，SPFR 会直接返回 NoRoute；②不支持零势能（zero-attractor）源的路由初始化；③理论证明基于“冻结”快照，若控制平面更新频繁或不一致，可能影响安全性；④实验仍为仿真，未在真实 IoA 部署中验证。

---

## 283. Refusal geometry reflects refusal training: diverse refusal prefixes can raise stable rank and weaken refusal vector ablation attacks

**arXiv ID:** 2608.25390 | [PDF](https://arxiv.org/pdf/2608.25390v1)

**作者:** Andrey Labunets `[一作]` `[通讯]` (UC San Diego), Andrey Labunets (UC San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究并量化在对抗性拒绝（jailbreak）攻击中，训练生成的拒绝输出为何能在激活空间中集中为低维方向，并验证这一机制与训练梯度的关系；

**💡 创新点**

首次揭示拒绝训练产生的梯度更新与模型拒绝方向的几何对应关系，并证明通过增加拒绝前缀多样性可提升拒绝残差的稳定秩，从而削弱单向量拒绝消融攻击的效果；

**🔧 技术方法**

利用激活梯度分析、稳定秩（stable rank）计算、向量消融攻击、以及梯度诱导激活更新（gradient‑induced activation update）等机制解析技术；

**📊 数据集**

主要在公开的 OLMo‑2‑0425‑1B‑Instruct 指令调优模型上进行实验，同时使用 AdvBench、WildJailbreak、XSTest‑Response、CAMEL 化学数据集等多种测试集；

**📈 对比分析**

对比不同训练阶段（SFT、DPO、RLVR1、Instruct）与不同拒绝前缀多样性设置下的拒绝向量、稳定秩、消融后拒绝率等指标，结果表明：①训练期间梯度与拒绝向量在方向和子空间上高度一致；②拒绝前缀多样性提升后，拒绝残差稳定秩提升，单向量消融导致的拒绝率下降更小；③多样化 fine‑tuning 能在保持相同拒绝基准的前提下，显著削弱消融攻击；

**⚠️ 局限性**

局限性包括：①仅验证单向量消融，未探究多向量或更强攻击；②实验集中于一款模型，缺乏跨模型泛化；③拒绝前缀多样性虽有提升效果，但在实际部署中如何生成多样化拒绝仍有挑战；④未考虑对其他安全特征的推广性。

---

## 284. DeCO: Discriminative Evidence Composition for Fine-Grained Dataset Distillation

**arXiv ID:** 2608.25480 | [PDF](https://arxiv.org/pdf/2608.25480v1)

**作者:** Chuixuan Fan `[一作]` (University of Science and Technology of China), Zhihui Wang `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种Fine‑Grained数据集蒸馏框架DeCO，利用TransFG教师的注意力滚动识别并裁剪局部证据，随后对同类证据进行空间多样性抑制并以网格形式拼接生成小样本集，仅用硬标签训练学生网络。

**💡 创新点**

创新点包括：①将注意力滚动作为局部证据评分；②引入距离抑制实现空间多样化；③构建类别级证据库并在同一类区域内进行网格拼接，以最大化稀缺像素预算下的细粒度信息利用。

**🔧 技术方法**

技术手段：TransFG预训练教师、注意力滚动、空间距离抑制、类别级证据库、网格拼接、标准硬标签监督训练。

**📊 数据集**

使用CUB‑200‑2011、FGVC‑Aircraft和Stanford Cars三大细粒度视觉分类基准数据集。

**📈 对比分析**

与Uniform、RDED、SRe^2L++、FADRM+等基线对比，IPC=1、3、5时DeCO在所有数据集上均表现最佳；在IPC=1时提升约27%‑46%，IPC=3、5时提升幅度仍保持在10%‑15%区间。

**⚠️ 局限性**

局限性在于高度依赖教师模型的注意力分布，对不同教师或不同网络架构的适用性未充分验证；网格尺寸和区域大小的选择仍基于经验，缺乏系统性分析。

---

## 285. TailorCoPilot: Enabling Agentic Pattern Making with Version-Controlled State Tracking

**arXiv ID:** 2608.25462 | [PDF](https://arxiv.org/pdf/2608.25462v1)

**作者:** Yuexin Sun `[一作]`, Huamin Wang `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了 TailorCoPilot，一个利用版本控制与专家轨迹的 AI 辅助面料图案设计系统，帮助初学者从需求描述完成图案创建。

**💡 创新点**

创新之处在于将基于轨迹的版本控制、符号化操作映射与模块化提示架构结合，实现从自然语言需求到可执行图案编辑的端到端智能代理。

**🔧 技术方法**

采用 Gemini 提示工程、结构化面料图案表示、符号化操作集、RADIO 视觉检索、共享布料仿真参数，并通过多阶段提示实现规划、验证与修复。

**📊 数据集**

构建了约 2,500 条由 TailorTrace 收集的专家状态转换与对应自然语言描述的数据集，并使用 120 件可编辑基础服装的检索库作为检索依据。

**📈 对比分析**

在 15+10+5 个简单/中等/复杂任务中与基线工作流比较，使用 Logistic 回归与方差分析，发现 TailorCoPilot 在完成时间下降 41.7%，作品质量提升 1.85 分，NASA‑TLX 工作负荷降低 0.71 标准差，且在所有任务难度下均显著优于基线。

**⚠️ 局限性**

局限性在于仅验证于有限的任务空间与服装类别，数据集规模有限且主要基于 CAD 环境，对多模态输入和更复杂结构的泛化仍需进一步研究。

---

## 286. DocPC: Document-Level Visual Retrieval via Representative Page Composition

**arXiv ID:** 2608.25434 | [PDF](https://arxiv.org/pdf/2608.25434v1)

**作者:** Chengsong You `[一作]` (East China Normal University), Nan Du `[通讯]` (Matter Innovation Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 DocPC，一种将文档的代表页面合成 2×2 网格进行编码的文档级视觉检索框架，并提出 DocViRe 基准。

**💡 创新点**

创新点包括：① 代表页面合成（Representative Page Composition），将多页文档压缩为单张图像实现 O(1) 索引；② 结合多正样本对比损失与稀疏调度的 ApproxNDCG 列表损失，专门针对文档级多正样本训练；③ 在同等监督下，利用网格编码显著提升跨页面信息融合。

**🔧 技术方法**

使用了 Vision‑Language 模型 ColQwen 进行多向量编码、late‑interaction 匹配；多正样本 InfoNCE + ApproxNDCG 列表损失；代表页面选择策略（First‑4、Boundary、Uniform‑K、Base‑Clip）。

**📊 数据集**

构造了 DocViRe 基准，来源于公开 PDFA 数据集，涵盖 7 个英文领域（生物、教育、金融、政府、工业、法律、研究），包含文档级查询与多正样本相关性标注。

**📈 对比分析**

与多种基线对比（文本检索、页级视觉检索+聚合、文档级视觉检索），DocPC‑ColQwen（First‑4）在 DocViRe 上平均 NDCG@5 达 44.09，超过最佳页级基线 38.91；同时将索引图像、向量数及存储量分别压缩 10.1×、7.7×。

**⚠️ 局限性**

局限性：网格合成降低单页分辨率，细节（如小字号文本、密集表格）易失真；当前页面选择不考虑查询，缺乏查询感知重排；基准仅覆盖英文 7 领域，未扩展多语言和更广泛文体。

---

## 287. PIVOT: A Multi-Trajectory Dataset and Testbed for Pose, Intrinsics, and Novel Viewpoint Evaluation in Real-World 3D Reconstruction

**arXiv ID:** 2608.25401 | [PDF](https://arxiv.org/pdf/2608.25401v1)

**作者:** Mary Raymond `[一作]` `[通讯]` (Independent Researcher), Mary Raymond (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PIVOT多轨迹数据集与评估平台，记录测量与优化相机姿态、相机内参的双重信息，并支持对新视角合成进行轨迹、姿态、内参敏感性评估。

**💡 创新点**

创新点在于将轨迹视为首要实验单元；双重姿态与内参存储；引入直接Pose Chamfer距离度量；并针对NeRF/3DGS等模型设计三类基准（轨迹泛化、姿态敏感性、内参敏感性）。

**🔧 技术方法**

利用COLMAP完成SfM与姿态优化，结合Nerfstudio框架中的Nerfacto与Splatfacto实现训练与评估，并开发完整的Python工具链进行原始数据处理、可视化与导出。

**📊 数据集**

使用v1版PIVOT数据集，包含五个真实场景（Church、Village Street、Victorian Garden、Frontyard、Backyard），采集自DJI Mini 4 Pro无人机，涵盖多种轨迹类型。

**📈 对比分析**

通过SSIM、PSNR、LPIPS等指标比较seen vs unseen轨迹、测量 vs 优化姿态、物理 vs 优化内参三组基准，结果显示未见轨迹和测量姿态均显著降低质量，内参优化可提升数分贝至十分贝。

**⚠️ 局限性**

局限性包括单一摄像头平台、物理校准误差大、SfM注册率差异、Pose Chamfer距离不考虑可见性与几何信息、以及几何稀疏点云对可视化分析的限制。

---

## 288. VietAIDetector: An Open-Source Zero-Shot Detector for Vietnamese AI-Generated Text

**arXiv ID:** 2608.25478 | [PDF](https://arxiv.org/pdf/2608.25478v1)

**作者:** Trieu Hai Nguyen `[一作]`, Van-Dung Hoang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了开源零样本越南语 AI 生成文本检测工具 VietAIDetector，支持多格式输入、长文档处理与可视化报告；

**💡 创新点**

首次将 VietBinoculars 零样本算法与 PhoGPT-4B/4B-Chat 结合，并集成 OCR 与滑动窗口分块，打造完整越南语检测系统；

**🔧 技术方法**

使用双模型 VietBinoculars 检测、PhoGPT-4B 与 PhoGPT-4B-Chat、Vintern-1B-v2 OCR、Gradio UI、Python 生态及 PDF 报告生成；

**📊 数据集**

使用越南语公开数据集（新闻、文学作品）及 GPT-5.6 Luna、Gemini 3.6 Flash、Claude Sonnet 4.6 生成的长文档进行评测；

**📈 对比分析**

与 GPTZero、Binoculars 等基线对比，VietAIDetector 在越南语与跨域长文档上与 GPTZero 性能相当，AI 分数更高但准确率略低；

**⚠️ 局限性**

受限于模型规模与 OCR 质量，长文档分块参数仍是开放研究问题，无法处理表格、图片等复杂结构，计算成本高，需持续更新以跟上 LLM 发展。

---

## 289. A Hybrid Usability Approach for Rating Evaluation of M-Commerce Applications

**arXiv ID:** 2608.25550 | [PDF](https://arxiv.org/pdf/2608.25550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 290. AERIS: Offline Policy Improvement for Multi-UAV Integrated Sensing and Communication

**arXiv ID:** 2608.25477 | [PDF](https://arxiv.org/pdf/2608.25477v1)

**作者:** Ziyuan Wang `[一作]` (Tsinghua University), Zhang `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了AERIS框架，利用离线飞行日志通过支持感知的离线多智能体强化学习方法（STAR‑CRDT）对多UAV集成感知与通信（ISAC）系统进行策略改进。

**💡 创新点**

创新点在于：①将轨迹与波束赋形控制转化为离线CTDE问题；②设计支持感知、信任门控的局部动作纠正与教师蒸馏（STAR‑CRDT），在不离开日志支持的前提下实现全局ISAC性能提升；③提供离线支持下的政策改进理论保证。

**🔧 技术方法**

核心技术包括：集中式训练与分布式执行（CTDE）架构、基于Transformer的决策变换器演员、双臂Critic与期望值网络、支持感知的候选搜索、信任门控蒸馏与隐式Q学习。

**📊 数据集**

使用的数据集为：①基于随机Brownian运动生成的1000条飞行日志（包含状态、动作、奖励及本地历史）；②来自OpenStreetMap的三条真实城市道路图（密集网格、干道、稀疏道路），用于零射程部署评估。

**📈 对比分析**

与BC、DT、TD3+BC、CRR、CRDT、OMIGA、OMAR等多种离线RL/ MARL基线进行对比；STAR‑CRDT在随机移动、系统规模扩大和未见道路地图上均实现了29.3%收益提升、3.4%通信速率提升、4.8%感知通过率提升、69.1%感知余量提升、54.2%碰撞风险下降，整体表现稳健优于所有基线。

**⚠️ 局限性**

局限性包括：①对高质量飞行日志的依赖，日志质量差时改进有限；②在极端新环境或移动模式变化较大时迁移性能可能下降；③训练过程需要大规模离线数据与计算资源；④仍需在真实硬件上进一步验证安全性与实时性。

---

## 291. Physics-Informed Foresight Pruning for Sparse PINN Solvers of Nonlinear PDEs

**arXiv ID:** 2608.25564 | [PDF](https://arxiv.org/pdf/2608.25564v1)

**作者:** Ahmad Ishaque Karimi `[一作]` (Arizona State University), Kookjin Lee `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在PINN训练前对模型进行物理信息引导的前瞻性剪枝，验证其在非线性PDE求解中的可行性

**💡 创新点**

提出PI‑SAP剪枝规则，利用PDE残差对参数敏感度进行评分，补充传统基于NTK谱的NTK‑SAP

**🔧 技术方法**

采用PirateNet结构、神经切线核（NTK）分析、残差感知剪枝、条件化NTK与PINN块诊断等技术

**📊 数据集**

使用四类PDE数据集：Gray‑Scott 反应扩散、复Ginzburg‑Landau、Burgers' 方程和对流方程，各自采用对应的空间时间网格与参数配置

**📈 对比分析**

与密集模型及NTK‑SAP比较；在Gray‑Scott与Ginzburg‑Landau中PI‑SAP在高剪枝率下显著降低残差与高频误差，解决精度阈值；在Burgers'与对流方程中两种方法性能互相切换；整体运行时基本不变，剪枝未带来时间加速

**⚠️ 局限性**

缺乏稀疏硬件加速（仅在密集张量上mask），统计显著性不足（Gray‑Scott与Ginzburg‑Landau未做多种种子），诊断核仅为小批量近似，未考虑跨块相互作用

---

## 292. EgoArgus: Benchmarking VLMs as Situational Assistants for Modality-Grounded User Supports

**arXiv ID:** 2608.25561 | [PDF](https://arxiv.org/pdf/2608.25561v1)

**作者:** Yu-Chien Tang `[一作]` (National Yang Ming Chiao Tung University), An-Zi Yen `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了 EgoArgus 这一两部分基准，评估第一人称视角助手在理解对话与视频并进行干预决策时的能力；同时提供了实验和分析。

**💡 创新点**

创新点在于：① 以五种对话-视频关系（多模态支持、矛盾、视频主题相关/不相关、文本支持）设计诊断场景；② 将真实视频问答与 VISTA 合成的干预决策结合，覆盖更完整的助手行为；③ 通过层级探测和注意力重权等手段系统评估并剖析模态偏差。

**🔧 技术方法**

主要使用的技术包括：多模态大语言模型（VLM）推理、线性探测（linear probing）、注意力重权（attention reweighting）、偏好对齐（NaPO）以及对话生成（Gemini 3.1 Pro）。

**📊 数据集**

所用数据集：真实视频问答来源 EgoPlan‑Bench2、QaEgo4D、MIntRec（共 6,978 例），以及由 VISTA 生成的 789 条合成第一人称助手剧本；所有数据已公开发布于 GitHub。

**📈 对比分析**

对七种 VLM（Molmo2‑8B、Qwen3.5‑2B、InternVL3.5‑4B、Cosmos‑Reason2‑8B、Qwen3.5‑Plus、Gemini‑3.1‑Flash‑Lite、MiMo‑V2‑Omni）进行对照实验；结果显示：矛盾对话导致准确率低至 18%（低于随机猜测），文本主导场景相对容易；在干预决策上，最强模型仍出现过度干预和时机错误，整体 F1 仅在 0.8 左右，说明当前模型尚难实现可靠助手。

**⚠️ 局限性**

局限性包括：① 理解任务采用四选一 VQA，无法覆盖开放式问答；② 决策任务使用合成视频，缺乏真实环境噪声与多样性；③ 仅对部分模型进行了探测与干预方法评估；④ 公开数据规模有限，未来需扩展更多真实剧本。

---

## 293. Goodput Maximization for Large Language Model Edge Inference: A Two-Phase Maskable PPO Approach

**arXiv ID:** 2608.25543 | [PDF](https://arxiv.org/pdf/2608.25543v1)

**作者:** Xiaojing Chen `[一作]` (Shanghai University), Yanzan Sun `[通讯]` (Shanghai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TP-MPPO 两阶段算法，用于 LLM edge 推理的任务 offloading 与带宽分配，目标是最大化 goodput；

**💡 创新点**

创新点包括：① 在 MPPO 中加入动作掩码避免无效 offloading 决策；② 对上下行带宽采用闭式解和贪心算法实现实时分配；③ 将 goodput 与 SLO 结合，形成统一的优化目标；

**🔧 技术方法**

采用的技术包括：Proximal Policy Optimization（PPO）与动作掩码，混合整数非线性规划（MINLP）建模，闭式解与贪心算法进行带宽分配，强化学习框架下的状态/动作/奖励设计；

**📊 数据集**

实验使用模拟数据：3-6 个 edge 节点、15-30 个用户，输入长度随机 {128,256,512}，输出长度随机 {128,256,512,1024}，λ ∈ {0.5,0.75,1}；采用 Llama-7B/30B 模型参数（h=4096，L=32），5G sub‑6 GHz 信道模型，NVIDIA L2 PCIe GPU 24GB；通信采用 32 位 token；

**📈 对比分析**

与四个基线（MPPO、TP-PPO、Rewardless、Heuristic）进行对比，TP-MPPO 在奖励上提升 33.3%–87.5%，在 goodput 上比基线提升 5%–10%（节点、用户、模型大小不同），并显著降低 OOM 失败率和延迟；

**⚠️ 局限性**

局限性包括：仅考虑单时隙传输与静态信道，未涵盖多时隙、多模型、多用户并发推理场景；假设所有请求在同一槽排队，未考虑动态批处理与模型蒸馏等技术；未来需扩展到更复杂的环境。

---

## 294. Video-IFBench: Evaluating Instruction Following of Multimodal LLMs in Video Understanding Scenarios

**arXiv ID:** 2608.25529 | [PDF](https://arxiv.org/pdf/2608.25529v1)

**作者:** Hongbo Liu `[一作]` (TJU), Shengjie Zhao `[通讯]` (TJU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Video-IFBench 评估视频多模态大语言模型（MLLMs）在遵循多约束、多任务与条件指令下的能力。

**💡 创新点**

创新点包括：① 四种指令模板（单任务、多任务、选择、嵌套）覆盖 32 任务与 39 语义/格式约束；② 半自动化数据构建管线结合 MLLM 提取、程序化规则与人工校验；③ 混合评估协议将 LLM-as-Judge 与可编程验证相结合。

**🔧 技术方法**

采用的技术：多模态 LLM（如 Qwen、Gemini 等）用于信息提取与指令生成；程序化条件生成与检查；LLM-as-Judge 评估答案满足度；人类验证确保样本质量。

**📊 数据集**

使用的数据集：约 700 条公开视频（总时长约 49 小时）来自现有视频基准与公开平台，随后生成 1.5k 质量样本，覆盖多域与多长度。

**📈 对比分析**

比较方法：基于 TCSR（平均约束满足率）与 TISR（完整满足率）对 20+ MLLMs 进行大规模评估；最佳专有模型 Gemini‑3‑Pro 取得 76.5% TCSR、54.5% TISR；最优开源模型 Qwen3.5‑397B‑A17B‑Think 69.6% TCSR、46.1% TISR，显示在多约束与条件指令上仍存在显著挑战。

**⚠️ 局限性**

局限性：模型在多约束、语义约束与嵌套/选择结构下表现仍弱；数据量相对有限（1.5k 样本）；评估依赖 LLM‑as‑Judge，可能带来主观性；未覆盖实时或极长视频的复杂场景。

---

## 295. CaSKG: Counterfactual-Causal Skill Graphs for Scalable Agent Skill Retrieval

**arXiv ID:** 2608.25500 | [PDF](https://arxiv.org/pdf/2608.25500v1)

**作者:** Zhiyuan Li `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 CaSKG 框架，研究大语言模型代理如何从大型可重用技能库中检索紧凑、可执行的程序性上下文；

**💡 创新点**

创新点在于将高召回的候选图与边置信校准分离：通过多信号构造候选图，利用方向条件的文本逆因果探针（移除、替换、倒置）与贝叶斯平滑得到边可靠性，再通过状态门控发布构成加权图；

**🔧 技术方法**

使用的技术包括：多源信号（语义、词汇、输入/输出、结构、修复）构造候选图；LLM 逆因果探针（Removal/Substitution/Reordering）；贝塔分布平滑校准；状态门控加权发布；任务条件下的个性化 PageRank 检索；

**📊 数据集**

实验数据集为 ALFWorld ID‑140（140 个家庭任务）和 ScienceWorld U211（211 个科学任务）；

**📈 对比分析**

与全库曝光、向量检索、Graph‑of‑Skills (GoS) 对比，在六种 LLM 后端（MiniMax‑M2.7、GLM‑5.2、Kimi‑K2.6、Qwen3.5‑397B‑A17B、DeepSeek‑V4‑Flash、GPT‑5.6‑Luna）与两大基准组合中均取得最高任务得分，并且平均环境步数更少；

**⚠️ 局限性**

局限性包括：仍依赖离线图构建，缺乏在线自适应更新；逆因果探针需要较多计算资源；在极强模型上增益有限；对极大规模库的可扩展性需进一步验证。

---

## 296. MMJailBench: A Factorized Benchmark for Disentangling Multimodal Jailbreak Vulnerabilities

**arXiv ID:** 2608.25490 | [PDF](https://arxiv.org/pdf/2608.25490v1)

**作者:** Tianshi Wang `[一作]` (Tongji University), Lei Zhu `[通讯]` (Tongji University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个分离因素的多模态 jailbreak 基准 MMJailBench，并在 16 款公开权重及商用 MLLM 上进行大规模评估。

**💡 创新点**

将有害意图、提示框架、视觉语义和指令载体四个关键因素分离并系统组合，构建可复现的多维度评估套件，同时通过内部表示和跨模态注意力诊断揭示视觉授权提示对模型行为的影响。

**🔧 技术方法**

采用多模态大语言模型评估框架、GPT‑5 作为判别器计算 harmfulness 分数、ASR 与 CASR 指标、注意力与表示向量分析等技术。

**📊 数据集**

基于自构造的 272 个有害意图、6 种提示框架、5 种视觉语义、2 种指令载体组合，共 16,320 条测试实例，形成 MMJailBench 数据集。

**📈 对比分析**

通过 Attack Success Rate (ASR) 和 Conditional ASR (CASR) 对模型进行对比，结果显示 GPT‑5 的 ASR 仅 2.17%，而 GLM‑4.6V 达到 78.38%；提示框架变化可导致 >40% 的 ASR 差异，授权文档视觉语义提升 12.96% 等，展现出模型间和因子间显著的性能差异。

**⚠️ 局限性**

仅覆盖 16 款模型和有限的 272 个有害意图，缺乏更细粒度视觉语义与更广泛的安全防御评估；判别器依赖 GPT‑5 可能带来主观偏差；OCR 识别误差和因子交互的更深入分析仍待进一步研究。

---

## 297. Controllable Affective Generation via Latent Vector Steering

**arXiv ID:** 2608.25569 | [PDF](https://arxiv.org/pdf/2608.25569v1)

**作者:** Xixian Yong `[一作]` (Renmin University of China), Xiao Zhou `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EmoVec，一种在推理时通过注入精炼的情感向量实现情感生成可控的轻量级框架

**💡 创新点**

创新点包括：利用对比激活加法提取情感方向，结合任务去偏和主成分子空间去除，最后在最终残差层注入可调强度的向量，实现无模型权重更新的连续情感强度控制

**🔧 技术方法**

核心技术为表示工程（Latent Vector Steering）、对比激活加法（CAA）、两阶段去偏（均值中心化+主成分去除）、线性探测定位最佳注入层，以及场景自适应比例调节器

**📊 数据集**

使用从Social Chemistry、NormBank、Social IQa等公开数据生成的160个情境示例（每种情绪160例，80例评估），并在三款LLM上进行实验

**📈 对比分析**

与不注入基线以及不同强度α={5,10,50}的注入效果对比，评估指标为GPT-4o情感分数、句子BERT相似度、LLM语义一致度。结果显示在α=50时平均提升约20%情感分数，且在大模型上可逼近70B模型，小模型提升更显著；语义相似度在强度升高时略降，但可接受

**⚠️ 局限性**

局限包括：假设情感可线性化为向量方向，可能无法捕捉混合或动态情感；评估仅在单轮文本心理健康问答中进行，未考虑多轮对话、长时序情绪变化或多模态输入，且依赖LLM评判，缺乏人工专家评估

---

## 298. CrossMambaTuning: Synergistic Spatial and Cross-Layer Adaptation for Machine Vision Compression

**arXiv ID:** 2608.25568 | [PDF](https://arxiv.org/pdf/2608.25568v1)

**作者:** Haobo Xiong `[一作]` (Xidian University), Chongyang Ding `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 CrossMambaTuning，一种针对预训练图像压缩模型的参数高效微调框架，以适配下游机器视觉任务。

**💡 创新点**

创新点在于引入任务感知的 Mamba Adapter 与规模不变的跨层适配器 SICA，既捕获长程空间依赖又实现跨层信息融合，显著提升参数利用率。

**🔧 技术方法**

使用了状态空间模型（SSM）Mamba 作为轻量化适配器，结合任务特定提示生成器、局部信息提取器、跨层共享参数策略，并在压缩与任务损失上联合训练。

**📊 数据集**

实验数据集包括 ImageNet 用于分类，COCO2017 用于目标检测与实例分割。

**📈 对比分析**

与 Channel Selection、ICMH-Net、TransTIC、Adapt-ICMH、SVD-LoRA 等 SOTA 方法对比，Tiny 变体仅 0.08M 可训练参数即可实现比 Adapt-ICMH 高约 2% mAP、比 SVD-LoRA 高 1.5% mAP 的性能，并在多任务上取得显著 BD‑rate 降低。

**⚠️ 局限性**

局限在于仅在 Lu2022‑TIC 与 ELIC 两种基线压缩模型上验证，尚未探究在更广泛的编码器架构或更大规模数据集上的可迁移性。

---

## 299. Maru: Information Architecture as a Shared Language for Generating Aligned and Persistent User Interfaces

**arXiv ID:** 2608.25565 | [PDF](https://arxiv.org/pdf/2608.25565v1)

**作者:** Eunhye Kim `[一作]` (KAIST), Juho Kim `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出并实现了 Maru 系统，利用信息架构（IA）四要素（partition、hierarchy、order、vocabulary）在生成式用户界面（GenUI）中实现结构层面的持久化，从而使得每一次 UI 生成都能保留并遵循用户已建立的组织逻辑。

**💡 创新点**

创新点在于：①把 IA 视为与用户交互的共享语言，将其四个元素映射为可持久化的规则；②通过自然语言、交互行为和 IA 面板三种通道捕获规则，并在后续生成中自动应用；③展示规则持久化如何提升 UI 对齐、个性化以及生成效率，并提出规则生命周期与任务上下文的边界设计。

**🔧 技术方法**

核心技术包括：大型语言模型（LLM）用于规则抽取、布局选择与内容填充；前端技术（Electron、React、TypeScript）实现可交互的布局与规则面板；后端规则存储、提取器、编辑分析器等模块实现规则的捕获、检索与更新；以及对 UI 组件的声明式渲染与手势处理。

**📊 数据集**

本研究使用的是 12 名受试者在三类信息任务（研究生院搜索、野餐规划、个人任务）中的交互日志与生成的 UI 作为实验数据；未使用公开数据集。

**📈 对比分析**

与不具备 IA 持久化的基线系统对比，评估指标包括：UI 对齐接受率（Maru 维持 61%–74% 率，基线 33%–71% 率）、平均生成次数（Maru 5.58 次 vs 基线 7.75 次）、对齐感知评分及查询长度等。实验表明 IA 持久化显著提升了用户满意度和生成效率，但在子任务切换或规则累积过多时会出现对齐下降。

**⚠️ 局限性**

局限性主要包括：①规则跨子任务迁移时缺乏上下文感知导致错误继承；②规则累积无生命周期管理，导致过多无关规则影响生成；③目前仅针对信息组织类任务验证，缺乏在更广泛领域（如数据分析、写作）的适用性验证；④对高阶细粒度 UI 控制（如具体组件布局）支持不足，未来需引入层级持久化与规则失效机制。

---

## 300. Throughput Maximization for MapReduce-Based Collaborative Computing over Energy-Harvesting Wireless Devices

**arXiv ID:** 2608.25549 | [PDF](https://arxiv.org/pdf/2608.25549v1)

**作者:** Yuhang Li `[一作]` (Shanghai University), Yanzan Sun `[通讯]` (Shanghai University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

研究了基于MapReduce的协作计算在能量收集无线设备上的资源分配问题，提出了一个在线吞吐量最大化的框架。

**💡 创新点**

创新点在于将深度确定性策略梯度（DDPG）与凸优化耦合，先用DDPG规划能量预算，再用凸求解器完成全局资源分配，从而显著降低动作空间维度并提升收敛速度。

**🔧 技术方法**

采用DDPG、凸优化（CVX）以及OFDMA、CPU频率约束和电池动力学模型等技术。

**📊 数据集**

使用仿真生成的随机无线信道、能量到达和设备硬件参数，没有使用公开数据集。

**📈 对比分析**

与贪心分配、随机预算、平均分配、单纯DDPG、最大频率等基线比较，DDPG‑CVX在吞吐量上分别提升1.25×~32.36×，并在不同电池容量和硬件异质性下保持最优。

**⚠️ 局限性**

局限在于仅验证了小规模设备组（数十台）下的性能，且需要中心化控制与实时信道/能量状态信息，未考虑多AP或异步更新的实际部署。

---

## 301. A Programming Paradigm for Spatiotemporal Composability

**arXiv ID:** 2608.25512 | [PDF](https://arxiv.org/pdf/2608.25512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 302. Interpreting Protein Language Model Embeddings via Orthogonal Projection for Protein Fitness Prediction

**arXiv ID:** 2608.25548 | [PDF](https://arxiv.org/pdf/2608.25548v1)

**作者:** Paulo Yanez Sarmiento `[一作]` (Hasso Plattner Institute), Bernhard Y. Renard `[通讯]` (Hasso Plattner Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用正交投影方法去除PLM嵌入中已知生化特征的线性及高阶效应，并评估其对蛋白质适应度预测的影响。

**💡 创新点**

将正交投影从仅线性效应扩展到高阶和交互效应，构建了一种计算高效、可迁移的PLM嵌入解释框架。

**🔧 技术方法**

采用正交投影(Post‑hoc orthogonalization)、L1特征选择、MLP隐层表示、逻辑回归下游分类器、条件独立检验及Adjusted R²评估等技术。

**📊 数据集**

基于ProteinGym深突变测序数据与G2P生成的结构化化学特征，共四类功能（组织生存、活性、表达、结合）进行实验。

**📈 对比分析**

通过MCC比较不同模态（表格特征、PLM嵌入、拼接、投影后嵌入）的线性分类器；投影后性能显著下降，表明PLM已编码大部分表格信息；R²约0.15–0.29解释嵌入预测方差。

**⚠️ 局限性**

局限在于仅评估ESM系列PLM和人类蛋白；未检验其他模型或多突变情况；使用线性下游分类器可能低估嵌入潜在能力。

---

## 303. Not All Degree Constraints Are Created Equal when Computing Spanning Trees

**arXiv ID:** 2608.25530 | [PDF](https://arxiv.org/pdf/2608.25530v1)

**作者:** Narek Bojikian `[一作]` (Humboldt Universität zu Berlin), Krisztina Szilagyi `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在不同图结构参数（如树深、顶点覆盖数、反馈顶点数等）下，三类度约束最小生成树（Specifed Degree MST、Bounded Degree MST、Set of Degrees MST）的参数化复杂度。

**💡 创新点**

首次揭示了在树深参数下，前两类问题可 FPT 而第三类则 W[1]-hard；同时证明了在顶点覆盖数下 Set of Degrees MST 仍 FPT，但在删除到常数路径宽的参数下 W[1]-hard；并提出了新的 ILP 编码与 2‑label gadget 变形技巧。

**🔧 技术方法**

核心技术包括：利用树深参数构造具有受限系数和双图树深的 ILP（可在 O*(k^k) 时间内求解），从多维子集和（SMPSS）构造硬度证据的图形 gadget，以及将 Set of Degrees MST 转化为加权匹配问题（generalized B‑matching）以利用现有 FPT 算法。

**📊 数据集**

本工作为纯理论研究，未使用任何实验数据集，所有结果均通过多项式时间归约与复杂度分析得出。

**📈 对比分析**

通过与之前在树宽、路径宽、剪切宽等参数下已知的时间上界与下界对照，展示了不同参数化下的复杂度分离，表明树深参数的细粒度划分能显著改变问题难度。

**⚠️ 局限性**

限制主要在于：对加权图下 Set of Degrees MST 的顶点覆盖数参数仍未知是否 FPT；此外，若要扩展到更一般的权重匹配算法，还需解决加权 generalized B‑matching 的 FPT 问题。

---

## 304. TransRetrieval: Scaling Up Transformer-Based Retrieval for Industrial Recommendation

**arXiv ID:** 2608.25528 | [PDF](https://arxiv.org/pdf/2608.25528v1)

**作者:** Zhifei Zheng `[一作]` (Renmin University of China), Bo Zheng `[通讯]` (Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 TransRetrieval，构建一种可扩展的 Transformer 召回框架，通过权重平均聚合、目标 token 压缩以及位置式域嵌入，解决特征异质性、计算瓶颈和多域数据扩展问题。

**💡 创新点**

创新点包括：①使用权重平均聚合恢复 Token 规范化，消除异质特征导致的 token 归一化失衡；②将目标侧特征压缩为单个 token，释放 85% FLOPs 并可进一步加深加宽网络；③采用位置式域嵌入统一多域数据，使稀疏域受益于跨域迁移，且成本几乎为零；④在工业 4 领域数据上验证 log‑linear scaling 并实现 2.53% 收益提升。

**🔧 技术方法**

核心技术包括 Transformer（Pre‑LN、ReLU FFN）、MLP 压缩、HNSW 图检索、KV 缓存共享、GPU 并行与算子融合、量化索引等。

**📊 数据集**

实验使用 40 B 交互的工业广告数据（4 个业务域）以及公开 KuaiRand 152 M 交互数据。

**📈 对比分析**

与 Production Baseline、KuaiFormer、HSTU、RankMixer 等基线在相同 FLOPs 或相同 latency 下对比，TransRetrieval 在 0.45 MFLOPs 预算下已 surpass baseline，进一步扩展到 1.91 MFLOPs（128D5L）时召回提升 5.4 pt；在线 A/B 测试中平台收益提升 2.53%。

**⚠️ 局限性**

主要限制包括对大规模训练与检索图的高资源需求、压缩后可能丢失细粒度特征表达、对极稀疏域的域嵌入表达能力有限以及在不同硬件平台上的迁移成本。

---

## 305. Dynamic Modeling of a Welding Torch Umbilical and Its Impact on Robot Dynamics

**arXiv ID:** 2608.25509 | [PDF](https://arxiv.org/pdf/2608.25509v1)

**作者:** Nicolas Gautier `[一作]` (Weez-U Welding), Damien Chablat `[通讯]` (Nantes Université)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种将焊接软管建模为受约束的多体系统（串联刚体与被动关节）的动态建模方法，并通过投影到可接受速度子空间来消除拉格朗日乘子，直接求解关节加速度与附着点反作用力。

**💡 创新点**

创新点在于：1）使用受约束多体动力学框架，既保留了弹性和耗散特性，又保持了与常规机器人动力学的兼容性；2）通过投影简化方程，显式得到反作用力，避免了传统拉格朗日乘子求解的计算开销；3）首次在机器人焊接场景中展示了软管动力学对机器人关节力矩的显著影响。

**🔧 技术方法**

所采用的技术包括：多体动力学的拉格朗日/牛顿-欧拉公式；受约束系统的速度约束投影；伪逆求解拉格朗日乘子；Krylov子空间（如GMRES）求解线性系统；预测-校正积分法。

**📊 数据集**

本文没有使用公开真实数据集，而是通过合成仿真得到的10段刚体软管和3自由度机器人在圆形轨迹下的数值数据进行验证。

**📈 对比分析**

通过比较在考虑软管动力学与不考虑时机器人关节力矩的变化，结果显示软管产生的额外力矩与机器人自身力矩同量级，说明忽略软管会导致显著误差。

**⚠️ 局限性**

局限性包括：仅在平面二维案例中验证，未涉及三维运动或移动支点；缺乏实验验证与参数识别；模型假设软管为刚性连杆且仅考虑平面弯曲，未考虑轴向伸缩或剪切。

---

## 306. Adaptive Hybrid Subspace Levenberg Marquardt Algorithm with Adequacy Monitor for Large Scale Least Squares Problems

**arXiv ID:** 2608.25524 | [PDF](https://arxiv.org/pdf/2608.25524v1)

**作者:** M. Duc Hoang `[一作]` (University of California, Davis), Timothy J. Lewis `[通讯]` (University of California, Davis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种适用于大规模非线性最小二乘问题的混合子空间Levenberg–Marquardt(HSLM)算法。

**💡 创新点**

创新点包括：自适应子空间构造（结合梯度、历史步、Krylov/Lanczos向量和随机高斯牛顿向量）；确定性充分性监视器保证子空间捕捉足够梯度信息；步长接受与阻尼更新分离；曲率适应的阻尼矩阵。

**🔧 技术方法**

采用的技术有：随机低秩近似、Krylov子空间投影、SVD分解与增量QR压缩、Armijo退火线搜索、预测/实际减小比（gain ratio）以及曲率自适应阻尼。

**📊 数据集**

使用Friedman函数回归数据集，构造三种不同规模的全连接网络（1000、2000、4000个可训练参数），对应训练集大小分别为10 000、20 000、40 000样本。

**📈 对比分析**

与经典LM和Krylov子空间LM进行对比。三者在最终训练误差上相近，但HSLM每次迭代的计算时间明显更低（例如网络3在HSLM下平均仅3.6 s/迭代，而经典LM需15.8 s、Krylov需13.1 s），相对加速约为4倍。

**⚠️ 局限性**

局限性包括：实验仅限于全量小批量训练，未探讨大样本规模下的Jacobian/矩阵向量乘积成本；未实现mini‑batch或矩阵无关的HSLM变体；子空间质量依赖候选方向，性能受候选构造影响；仅在神经网络回归问题中验证，其他逆问题需要进一步研究。

---

## 307. ConfAL-WM: Confidence-Guided Active Learning for Action-Conditioned World Models

**arXiv ID:** 2608.25572 | [PDF](https://arxiv.org/pdf/2608.25572v1)

**作者:** Xiang Liu `[一作]` (Tsinghua University), Changshui Zhang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套基于置信度的主动学习框架 ConfAL-WM，用轻量级置信度探针在 UNet 解码器特征上检测并加权局部预测错误，以提升动作条件世界模型的后训练效率和预测质量。

**💡 创新点**

创新点在于：① 在 UNet 解码器上插入稀疏的置信度探针，并通过 EMA 自适应阈值训练生成细粒度置信度图；② 将置信度聚合为任务、帧、补丁级评分，实现多层次的数据选择与加权训练；③ 采用分阶段主动学习管线将置信度应用于预算分配与局部监督，显著提升学习效率。

**🔧 技术方法**

使用的技术包括 UNet‑based latent diffusion 世界模型 EVAC、稠密置信度探针与自适应 EMA 阈值训练、任务/帧/补丁级风险聚合、加权损失函数、多阶段主动学习流程以及 EWMBench 评估协议。

**📊 数据集**

在 AgiBot World 预训练基础上，对 RoboTwin2.0（Aloha‑AgileX 双臂机器人）数据集进行实验，包含 50 个操作任务、约 25,000 条视频（共 24,992 轨迹），每个视频长度 98–578 帧。

**📈 对比分析**

与 RoboReward、GVL、Robometer、PRM‑as‑Judge、LRMs 等标量评分方法以及无评分基线进行对比；使用 EWMBench 的 PSNR、SSIM、Scene Consistency、Semantics、Traj‑HSD 等指标评估；结果表明置信度指导的样本选择加帧/补丁加权训练在所有主要指标上均优于传统标量评分策略，显著提升重建、场景一致性、语义匹配和轨迹一致性。

**⚠️ 局限性**

局限性包括：置信度探针仅适用于 UNet 解码器结构，难以直接迁移到其他世界模型或数据域；置信度表示缺乏独立的可靠性验证；当前评估仍受视觉质量与运动准确性冲突限制，需更细粒度的局部与动态一致性评估。

---

## 308. A Tendon-Driven Five-Fingered Hand with Distributed Tactile Perception for Dexterous Manipulation

**arXiv ID:** 2608.25547 | [PDF](https://arxiv.org/pdf/2608.25547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 309. Syn2Logic: End-to-End Neuromorphic Design Automation

**arXiv ID:** 2608.25536 | [PDF](https://arxiv.org/pdf/2608.25536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 310. AdaVDR: Adaptive Tool Use and Reflection for Video Deep Research

**arXiv ID:** 2608.25559 | [PDF](https://arxiv.org/pdf/2608.25559v1)

**作者:** Xintong Zhang `[一作]` (Alibaba Group), Hongwei Xue `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 AdaVDR，一种可自适应调用工具和反思机制的多轮视频深度研究代理，用于在视频与外部知识检索之间进行高效推理。

**💡 创新点**

创新点在于（1）基于任务类型与模型能力的自适应工具调用；（2）仅在中间结果不可靠时触发反思回溯；（3）构建任务特定、模型定制化的工具使用轨迹，并通过模型条件工具必要性过滤剔除冗余步骤；（4）在此基础上进行监督微调与重奖惩强化学习，提升推理准确性与效率。

**🔧 技术方法**

技术包括多模态大型语言模型作为代理脑、视频时序/时间戳/空间定位工具、图像搜索、网络搜索与页面访问等外部检索工具；数据构建流水线（QA生成、轨迹生成、校正、工具必要性过滤）；监督微调（SFT）与基于奖励的强化学习（GRPO）及冗余惩罚。

**📊 数据集**

使用了自建的 VDR-EE 250 条人工验证的实体与事件中心问答数据集，以及公开的 VideoDR 公开基准；在数据构建时还从 YouTube、EgoExo4D 等多源视频中采集原始素材。

**📈 对比分析**

与多种开源与专有模型（Gemini、GPT‑5、Qwen 系列等）在 VDR-EE 与 VideoDR 上对比，AdaVDR 在 Agentic 设置下平均提升 10–15% 以上准确率，同时平均工具调用次数下降 1–2 次；在 VideoDR 上，AdaVDR 在所有域与难度级别均取得显著提升。

**⚠️ 局限性**

局限性包括：① 对长视频与多子事件的定位仍有挑战；② 需要大量人工验证的 QA 与轨迹数据；③ 反思机制在极端不确定场景下仍可能误判；④ 依赖外部检索服务，网络可用性与隐私限制可能影响部署。

---

## 311. Virgil: Navigating Explainability for Transformer-based Language Models

**arXiv ID:** 2608.25555 | [PDF](https://arxiv.org/pdf/2608.25555v1)

**作者:** Martino Ciaperoni `[一作]` (Scuola Normale Superiore), Fosca Giannotti `[通讯]` (Scuola Normale Superiore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Virgil系统，提供了一个统一的交互式平台，帮助用户搜索、探索和比较Transformer语言模型的可解释性工具；

**💡 创新点**

将众多解释器以知识库形式整理，配合检索引擎与探索引擎实现从结构化查询、自然语言查询到可视化对比的一体化流程，且系统架构模块化，易于扩展；

**🔧 技术方法**

使用Python+Streamlit搭建Web前端，采用sentence-transformer（all-MiniLM-L6-v2）做文本嵌入，结合加权排名实现检索；支持直接调用Hugging Face预训练模型并在前端可视化解释结果；

**📊 数据集**

知识库中共43个解释器卡片；示例实验使用情感分类（电影评论）任务，未明确使用公开数据集，仅展示模型推理过程；

**📈 对比分析**

通过侧边对比视图（如 Input×Gradient 与 Integrated Gradients）展示解释器输出的差异与稳定性；性能评估主要基于用户体验调查（满意度、易用性）和对比示例的可视化结果，并未给出定量指标；

**⚠️ 局限性**

评估样本规模有限（10名研究者），缺乏大规模客观实验；系统依赖已有解释器，功能受限于知识库更新速度；缺乏跨任务、跨模型的广泛验证。

---

## 312. Beyond Optimal Rates in Stochastic Optimization: Trajectory-Adaptive Stopping Rules

**arXiv ID:** 2608.25551 | [PDF](https://arxiv.org/pdf/2608.25551v1)

**作者:** Liviu Aolaritei `[一作]` (UC Berkeley), Michael I. Jordan `[通讯]` (UC Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在强凸随机优化中，提出一种全可观测、轨迹自适应的置信序列，用以给出SGD在实际运行过程中的在线停止准则，并保证在任意自适应停止时间下仍满足高概率保证。

**💡 创新点**

创新点包括：
1) 通过递归和自我归一化方法构造时间均匀的置信序列；
2) 设计可观测范围随时间增长的经验‑Bernstein界，克服传统方法需要固定上界的局限；
3) 使置信序列能够充分利用观察到的梯度幅值与方向信息，从而显著缩短所需迭代次数；
4) 将上述构造扩展到小批量SGD，并进一步利用批内第二矩实现更精细的自适应。

**🔧 技术方法**

主要技术手段：
- 递归置信序列（recursive confidence‑sequence）与时间均匀 Hoeffding 证明；
- 经验‑Bernstein 不等式的时间均匀版本，支持可观测范围随历程增长；
- 按 dyadic 范围拼接（stitching）实现多尺度时间均匀界；
- 小批量梯度的条件独立性利用，结合矩阵谱上界 
- 线性加权平均与强凸收敛率分析。

**📊 数据集**

实验使用的主要数据集：
- 公共数据集 covertype.binary（SVM 原始问题）；
- 通过 scikit‑learn 生成的三组合成二分类数据（参数可调的类分离度）；
- 也在单次遍历（single‑pass）情形下进行验证。

**📈 对比分析**

比较方法：
- 传统固定时间高概率界（RSS12、RSS12 的改进版）；
- 本研究的 Hoeffding‑基置信序列（H）；
- 经验‑Bernstein‑基置信序列（EB）以及其小批量扩展（EB,MB）。
实验结果显示：
- EB 在大多数设置下比 H 提前 1‑2 个数量级停止；
- H 与固定时间界相比提前 5‑6 个数量级；
- 随小批量大小增大，EB 的优势进一步放大，提前数十到数百个数量级。

**⚠️ 局限性**

局限性：
- 需要已知强凸参数 μ 以及梯度噪声上界 σ²（或 G）；
- 对步长调度有一定假设（可预测且满足上界）；
- 理论主要针对强凸情况，非强凸或无约束情形尚未覆盖；
- 实际中常数项较大，导致在小规模问题上可观测性能提升不显著；
- 需要对随机梯度进行条件独立性或可观测范围可被估计的假设。

---

## 313. Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models

**arXiv ID:** 2608.25518 | [PDF](https://arxiv.org/pdf/2608.25518v1)

**作者:** Pengfei Zhou `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了利用游戏开发中的人机验证循环来训练世界模型的框架，核心为Agentic World Model (AWoMo) 与 Reinforcement Learning with Human-Engine Verification (RLHEV)；

**💡 创新点**

创新点在于将游戏引擎作为可执行验证器与人类评价结合，形成稠密结构奖励与稀疏但高权威性接受信号的双重反馈机制，实现可递归的自我改进循环；

**🔧 技术方法**

技术主要包括多模态场景程序接口、引擎检测（碰撞、物理、导航、脚本）与人类评审回馈的统一协议（UWDP），以及在此基础上的强化学习后训练；

**📊 数据集**

使用的数据集包括UnitySceneBench（200个Unity资产编辑任务）、跨引擎（Unity→Unreal/Godot）评测集，以及用于环境数据增强的R2R、Gymnasium MuJoCo 和 D4RL Gym-MuJoCo；

**📈 对比分析**

与零样本CLIP、模糊代理、SFT、离线RLHF及仅基于引擎奖励的RLVR相比，RLHEV+AWoMo在UnitySceneBench上获得最高的primary得分（0.681），在跨引擎迁移上亦实现正向OOD提升，并在环境增强实验中提升R2R、Gymnasium MuJoCo 与 D4RL Gym-MuJoCo 的指标；

**⚠️ 局限性**

局限性包括实验规模仍为诊断性、对人类评审成本的依赖、跨引擎评价尺度不一致、以及未能充分验证完整自我改进循环的可行性与长期收益。

---

## 314. Joint Beamforming Design and Port Selection in Fluid Antenna-Assisted Multi-Cell Networks: A Personalized Federated Learning Approach

**arXiv ID:** 2608.25514 | [PDF](https://arxiv.org/pdf/2608.25514v1)

**作者:** Liwen Gao `[一作]` (Northwest University), Lin X. Cai `[通讯]` (Illinois Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于FedRep框架与位置感知双分支深度网络（PA‑DNN）的多基站流体天线网络中的联合波束成形与端口选择优化方案，目标是最大化加权总速率。

**💡 创新点**

创新点包括：① 将波束成形视为全局共享表示、端口选择视为本地个性化头，利用FedRep实现跨基站协作与本地适配；② 引入多频正弦余弦位置编码与SIREN激活函数，提升对端口空间相关性的建模；③ 使用Gumbel‑Softmax与STE实现可微分的离散端口选择。

**🔧 技术方法**

技术手段包括：联邦学习（FedRep）、深度神经网络（PA‑DNN）带SIREN隐藏层、位置编码、Gumbel‑Softmax + Straight‑Through Estimator、全局与局部参数分离的块坐标下降训练、Adam优化器。

**📊 数据集**

使用仿真生成的数据集：每个基站8个固定天线、6 GHz信道、P = 25个端口、30条多径、自由空间路径损耗及Rayleigh衰落，用户数在2–4人不等，产生约10,000个样本。

**📈 对比分析**

与固定天线基线、随机端口、最大化均匀化等传统端口选择方法以及pFedMe、传统FL、EM‑PFL等联邦学习基准进行对比。实验表明：PA‑DNN在单基站环境下实现约1.57 bps/Hz的加权总速率；FedRep框架在多基站场景中最快收敛至约1.34 bps/Hz，明显优于传统FL（0.9 bps/Hz）和pFedMe（1.29 bps/Hz）等。

**⚠️ 局限性**

局限性主要包括：① 需要大量仿真数据，缺乏真实场景验证；② Gumbel‑Softmax温度调节需经验手动设定，影响收敛稳定性；③ 在极端干扰或高度异质环境下，端口选择的随机性仍可能导致性能波动。

---

## 315. FedQoS: Federated QoS-Risk Learning for Heterogeneous Indoor-Outdoor Access Selection

**arXiv ID:** 2608.25496 | [PDF](https://arxiv.org/pdf/2608.25496v1)

**作者:** Nguyen Van Thieu `[一作]` (University of Luxembourg), Symeon Chatzinotas `[通讯]` (University of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种 FedQoS 框架，利用联邦学习在室内外异构网络中对候选接入链路的未来 QoS 失效风险进行预测，并基于此风险分数做可靠的接入选择。

**💡 创新点**

创新点在于：①将 QoS 风险预测转化为联邦监督学习任务，避免中心化数据；②设计了 QoS‑aware 聚合权重，平衡样本规模与失效率，适应非 IID 客户端；③将预测风险与切换成本结合，构建可解释的接入决策规则。

**🔧 技术方法**

采用了多层感知器 (MLP) 作为风险预测模型，使用 FedAvg、FedProx 以及自定义 FedQoS 的聚合策略；实验通过 Sionna 生成的物理层仿真数据来训练与评估。

**📊 数据集**

数据集为基于 Sionna 的 200m×200m 校园场景的物理层仿真日志，包含三栋多层建筑、29 个 AP、1 个 BS、2 个 UAV，覆盖正常、事件、拥塞与突发失效等多种用户移动与流量场景，形成 mild 与 severe 两种非 IID 条件。

**📈 对比分析**

与四个联邦基线（FedAvg、FedProx、Local-only、Centralized）及五个非学习基线（Current serving、Strongest SINR、Load‑aware RSRP、AP‑first、Historical QoS）进行对比，FedQoS 在 QoS 失效率、BLER 上显著低于传统策略，失效率接近中心化模型，吞吐量仅略低 0.5% 左右；在 severe 非 IID 下仍保持竞争性能。

**⚠️ 局限性**

主要局限包括：①对实时负载与极端事件的即时感知仍有限；②依赖仿真生成的物理层数据，缺乏真实部署环境的验证；③聚合参数 q、λ 的选择对性能有一定影响，需在实际系统中进行调优。

---

## 316. A Storage-Retrieval Gap in Parametric Knowledge Graph Memory

**arXiv ID:** 2608.25489 | [PDF](https://arxiv.org/pdf/2608.25489v1)

**作者:** Martino M. L. Pulici `[一作]` (Bosch Center for Artificial Intelligence), Volker Tresp `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将知识图谱的每个子图编译成LoRA适配器，并在不使用上下文的闭卷问答中注入权重；

**💡 创新点**

验证子图适配器确实在权重中保留可恢复的事实，但同类邻近适配器无法跨越转移知识，从而揭示存储局部性与检索缺失的耦合；

**🔧 技术方法**

使用LoRA低秩微调、闭卷评估、适配器权重几何检索（ΔW Frobenius距离）和实体-问题解析器；

**📊 数据集**

MetaQA电影领域的知识图谱（Qwen3.5-2B生成的问答对）；

**📈 对比分析**

与近盲基础模型对比，单值关系闭卷提升0.243 EM，单值关系oracle提升0.283 EM；检索方法（文本嵌入或ΔW几何）几乎与随机检索等效；

**⚠️ 局限性**

局限在于仅单域单模型实验、仅评估单值关系，检索与合成问题未解决，存储成本高且更新昂贵。

---

## 317. ReliableRAG: Combating Misinformation in Retrieval-Augmented Generation via Reliability-Guided Reasoning Chains

**arXiv ID:** 2608.25487 | [PDF](https://arxiv.org/pdf/2608.25487v1)

**作者:** Jinpu Jiang `[一作]` (Jilin University), Chunguo Wu `[通讯]` (Jilin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可靠性驱动的检索增强生成框架ReliableRAG，用于在多跳问答中识别并过滤细粒度误导性信息，避免谬误在推理链中传播，提升答案准确性。

**💡 创新点**

创新点在于：① 在三元组级别进行可靠性评估，结合查询-三元组语义相关性与三元组可信度的双因素感知机制；② 构建可自动扩展的推理链，通过束搜索和多选推理提示确保链条只包含高可靠性信息；③ 采用离线三元组提取与在线动态评估的两阶段流程，兼顾效率与可靠性。

**🔧 技术方法**

技术手段包括：大规模预训练语言模型（LLM）用于三元组抽取、可信度评估与链条构建；双语编码器对查询与三元组进行嵌入并计算余弦相似度；多选推理提示与束搜索策略在链条构造中使用；平衡系数α调节语义相关性与可信度的权重。

**📊 数据集**

使用三大多跳问答数据集：HotPotQA、2WikiMultiHopQA、MuSiQue；在每个测试集上随机抽取1000个问题，并对每题注入低可信度文档模拟误导性信息。

**📈 对比分析**

与包括Naive LLM、Vanilla RAG、Prompt-based、Exclusion、Self-RAG、CAG、Knowledge-R1、CrAM、TruthfulRAG等九种方法对比，使用EM和F1作为指标。ReliableRAG在理想与评估者生成的设置下均领先，EM提升约5-15%（HotPotQA、2WikiMultiHopQA、MuSiQue），F1亦显著提高。

**⚠️ 局限性**

局限性：① 依赖三元组抽取的质量，若抽取失败可能导致信息缺失；② 需要额外的LLM推理步骤，推理成本相对较高；③ 对α、K、L等超参数敏感，需在特定任务中调优；④ 仅在已注入低可信度文档的实验环境下验证，真实网络噪声情况仍需进一步评估。

---

## 318. When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory

**arXiv ID:** 2608.25553 | [PDF](https://arxiv.org/pdf/2608.25553v1)

**作者:** Kazuki Nakayashiki `[一作]` `[通讯]`, Kazuki Nakayashiki

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在有限验证预算下，研究代理在继承旧记忆时，若该记忆的约束被后续记录撤销，是否能够检测到并避免错误，并通过实验评估将验证槽分配给关键历史路径的效果。

**💡 创新点**

提出并量化了“可避免错误比例”，证明仅通过改变验证分配即可几乎消除因记忆新旧导致的错误，并区分验证策略与传统检索机制的差异。

**🔧 技术方法**

利用预算化验证实验，采用六个大型语言模型（Claude Opus 5、Claude Sonnet 5、Claude Haiku 4.5、GPT‑5.6 Sol、GPT‑5.6 Terra、GPT‑5.6 Luna），在六内存增长场景中引入被废止的记录，设计 native、forced‑critical、forced‑noncritical 三种验证策略，并使用二项式风险差进行统计分析。

**📊 数据集**

使用原工作中的六内存增长实验场景（六个一行记忆）、两个合成世界（有效与被废止）、一个采购域持有不同 supersession 记录，并通过随机种子生成多种措辞族，累计约 3,000 条实验记录。

**📈 对比分析**

对比 Y（决策是否与当前记录一致）在 forced‑critical 与 native 两策略下的风险差；在主、复制、持有外部等四个跑中，强制关键路径提升约 10–30 个百分点，接近结构上限，所有模型表现方向一致。

**⚠️ 局限性**

实验环境合成且规模有限；未实现可部署的验证调度器；未测量真实管线中约束失效的频率；模型与域受限；记录检索仅单跳；干预为实验者提供的 oracle。

---

## 319. An Event is Worth One Token: Event Tokenization for Industrial-scale LLM Recommendation

**arXiv ID:** 2608.25546 | [PDF](https://arxiv.org/pdf/2608.25546v1)

**作者:** Fan Xia `[一作]` (AI at Meta), Minghai Chen `[通讯]` (AI at Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并验证一种名为 AMBER 的推荐系统框架，核心是将每条交互事件的多模态、异构特征压缩成 Event Token，异步缓存后由下游的 LLM 进行自回归序列建模，既提高了历史 snapshot 分辨率，又降低了在线推理的计算量。

**💡 创新点**

创新点包括：
1) 将事件级特征统一编码为高密度 Event Token 并在 LLM 训练中端到端对齐；
2) 通过异步事件标记化与缓存实现 snapshot 分辨率与在线计算的解耦；
3) 采用对抗性漂移抑制与 EMA 机制稳定长期递归训练；
4) 在大规模工业平台上系统性地评估与传统点对点、HSTU、Semantic ID 等基线的对比。

**🔧 技术方法**

技术手段：
- Event Tokenizer（基于双向 Transformer 编码 + MLP 投影）；
- 用户 LLM（decoder‑only Transformer，初始化自 Llama 1B）；
- 两阶段预对齐与联合训练；
- 异步事件处理与缓存；
- 对抗性域适配（DANN）+ EMA 进行表示漂移抑制；
- Matryoshka Dropout 与 INT8/INT4 量化实现 Token 压缩。

**📊 数据集**

数据集：来自 Facebook 大型工业推荐平台的 PB 级点击/展示日志，包含数百个用户、商品、上下文及结果信号；实验使用时间序列划分（训练/校准/评估）。

**📈 对比分析**

评估方法：
- 主指标为 Normalized Entropy（NE）及 Ensemble NE，检索端使用 Soft Recall；
- 与已优化的 Incumbent 排序模型、点对点模型、HSTU、Semantic ID+CU 等做对比；
- 结果显示 AMBER 在 NE 上比 Incumbent 低 0.40%，在 Ensemble NE 上比 Incumbent 提升 0.10–0.16%，检索 Soft Recall 提升 0.31–0.51%，并在大规模线上验证 NE 提升 0.06%。

**⚠️ 局限性**

局限性：
- 事件标记化的异步处理会引入短暂延迟；
- 长期递归训练仍需对抗性漂移抑制，维护成本高；
- 目前仅对离线特征进行压缩，在线高频事件仍需额外计算；
- 未在 LLM 端实现完整端到端推理，仍依赖传统排序模型或检索；
- 对于更大模型规模可能需要 MoE 或更高效的 Tokenizer 设计。

---

## 320. Reflection Steering: Disentangling Reflection from Reasoning in Activation Space for Token-Efficient Inference

**arXiv ID:** 2608.25542 | [PDF](https://arxiv.org/pdf/2608.25542v1)

**作者:** Jiarui Hu `[一作]` (University of Hong Kong), Yu Yang `[通讯]` (Education University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的 Reflection Steering 方法，控制 LLM 在推理过程中反思（verification、backtracking 等）所产生的额外计算；

**💡 创新点**

核心创新在于将反思相关的激活方向从一般推理激活中分离（PCA 去噪 + 正交化），并在选定层使用有界投影消除反思分量，同时提供可调节的干预强度参数 α；

**🔧 技术方法**

技术实现包括：1) 在每层对反思与非反思隐藏状态做均值差估计；2) 通过 PCA 限制方向噪声；3) 对共享推理方向做正交化去除干扰；4) 在小校准集上测试多层、不同强度的干预并筛选稳定层；5) 在推理时对选层的激活做 α‑调节的投影消除；

**📊 数据集**

使用的公开基准为 MATH‑500（数学推理）和 GPQA‑Diamond（科学多选推理），在这些数据集上构造方向并进行校准；

**📈 对比分析**

与现有的代表性激活层干预方法 CREST 和 ReflCtrl 在同一模型（Qwen‑3‑30B‑A3B、Qwen‑3‑8B、Qwen‑Q‑32B）以及两大基准上进行对比；实验显示 Reflection Steering 在六个匹配设置下平均减少 16.9% 思考 token，且在数学推理上保持或略微提升准确率；在 GPQA‑Diamond 上也实现显著 token 减少，但准确率略有下降；

**⚠️ 局限性**

局限性包括：1) 仅在 Qwen 系列开放权重模型上验证，尚未跨越更广模型族；2) 依赖于事先标注的反思与非反思样本，可能受提示、任务风格影响；3) 干预强度 α 的选择需在部署时手动调整，缺乏自适应机制；4) 对解码策略（温度、top‑p）敏感，未覆盖所有推理设置。

---

## 321. CropCop: An Auditable 120-Class Plant-Health Model from Benchmark Reconstruction to a Quantised Runtime Artifact

**arXiv ID:** 2608.25539 | [PDF](https://arxiv.org/pdf/2608.25539v1)

**作者:** Rana Muhammad Ahmed `[一作]` (Bahria University Islamabad), Sabahat Abbas `[通讯]` (Bahria University Islamabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了一个针对120类作物健康状况的闭集识别系统CropCop，完成数据集去重、长尾处理、模型训练、压缩、量化及端到端运行文件的完整可追溯链。

**💡 创新点**

通过完整可追溯的审计链，确保从数据去重到推理文件的每一步无泄漏、无跨集匹配，并展示量化后运行文件与原始模型在类平衡性能上的细粒度差异。

**🔧 技术方法**

采用DINOv3预训练的ConvNeXt‑Tiny作为基线，MobileNetV4轻量化模型，XNNPACK量化，ExecuTorch PTE序列化，验证性PTQ、Bootstrap置信区间、类级误差分析等技术。

**📊 数据集**

使用重构后的109,107张图像，120个运营类（叶片、果实、整株等），来源多样（PlantVillage、PlantDoc、PlantWild等），并对图像进行了彻底去重和分区。

**📈 对比分析**

通过锁定内部测试集对比全精度、量化、PTE三种模型状态的准确率、宏F1、Top‑1一致率等指标，最终PTE准确率98.46%，宏F1 96.23%，仅有6条Top‑1差异，宏F1下降约0.64个百分点。

**⚠️ 局限性**

仅在内部数据集上验证，未评估未见农场、摄像头或安卓硬件的泛化；缺少完整的训练配置和非DINO对照；量化后运行仅在Kaggle CPU环境测试，未提供安卓性能或能耗指标；数据集去重可能仍有漏检。

---

## 322. TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving

**arXiv ID:** 2608.25523 | [PDF](https://arxiv.org/pdf/2608.25523v1)

**作者:** Hongqiu Ni `[一作]` (University of Science and Technology of China), Haisheng Tan `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

无法获取论文内容

**💡 创新点**

无法获取论文内容

**🔧 技术方法**

无法获取论文内容

**📊 数据集**

无法获取论文内容

**📈 对比分析**

无法获取论文内容

**⚠️ 局限性**

无法获取论文内容

---

## 323. OpenVeinNet: Robust Open-Set Finger Vein Verification with Dynamic Snake Convolution and Graph Learning

**arXiv ID:** 2608.25515 | [PDF](https://arxiv.org/pdf/2608.25515v1)

**作者:** Sushrut Patwardhan `[一作]` (Norwegian University of Science and Technology), Raghavendra Ramachandra `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了开放式指静脉验证，提出了 OpenVeinNet 框架；

**💡 创新点**

创新点在于将动态蛇形卷积（DSConv）与图卷积网络（GCN）相结合，并提出了 Centroid Angular Hybrid Loss，以提升开放集下的特征分离度；

**🔧 技术方法**

采用了 DSConv、GCN、角度损失（Centroid Angular Hybrid Loss）以及余弦相似度做判别；

**📊 数据集**

使用了五个公开指静脉数据集：FV‑300、MMCBNU、FV‑USM、PolyU 和 VERA；

**📈 对比分析**

与传统手工特征方法以及多种深度学习基线进行跨数据集留一法评估，OpenVeinNet 在 EER、AUC、低 FAR 的 TAR 指标上均优于或竞争；

**⚠️ 局限性**

主要限制包括相对较高的计算成本，以及在极低质量或样本量极少的数据集（如 VERA）上仍显弱，需进一步轻量化与鲁棒性提升。

---

## 324. Conditional Total Correlation and the Serial Depth of Adaptive Parallel Sampling

**arXiv ID:** 2608.25505 | [PDF](https://arxiv.org/pdf/2608.25505v1)

**作者:** Chuling Wen `[一作]` (Shenzhen University), Jian Lu `[通讯]` (Shenzhen University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究离散向量的自适应并行采样问题，提出序列深度（serial depth）概念，并证明前向KL误差与期望条件总相关量（conditional total correlation）等价，从而将并行采样的误差量化为信息成本。作者在多种结构化分布（有限阶马尔科夫链、伯努利游走、均匀随机排列、平衡二进制串、单热块）上给出精确或近似的深度上界与下界，揭示了对数、复对数、多项式和线性深度的四种范式，并在Masked Diffusion语言模型上验证伪成本与生成质量的一致性。

**💡 创新点**

创新点：①首次将前向KL误差与期望条件总相关量等价，提供了一个统一的“信息成本”框架；②利用该等价性推导出序列深度与分布的条件依赖结构直接相关，显著区别于传统基于熵或负对数似然的度量；③在多类典型分布上给出完全匹配的深度界，展现了对数、复对数、多项式、线性深度四种截然不同的规模；④通过实验将理论伪成本映射到实际并行解码策略的质量评估，验证了理论的实用性。

**🔧 技术方法**

技术手段：信息论（KL、条件总相关、链式规则）、组合优化（整数分解、批次排列）、递归分割与分隔子结构、随机游走与马尔科夫链分析、对数与熵的近似界定、离散随机过程的极限定理、Masked Diffusion模型的无监督预训练与条件概率估计、实验评估与自采样对比。

**📊 数据集**

实验数据集：542 条文本，包含 512 篇 WikiText‑103 文章（验证集与训练集各 256 条）以及 31 条生成文本，覆盖代码、数学题、散文、结构化数据四个域。所有文本均在 512 位置以内截断，采用 Qwen2.5‑0.5B 模型的 Masked Diffusion 版本进行条件概率估计。

**📈 对比分析**

方法比较：与传统左到右连续扫描、随机位置抽取、分层二分（bisection）、贪心最小熵（greedy min‑entropy）等固定或自适应调度进行对比。伪成本指标与自采样输出的 perplexity 高度相关，随机/分层调度在相同轮数下的 PPL 远低于 confidence top‑k 或阈值解码；实验表明在相同轮数约 8~16 时，随机 16 或 64 批次可逼近或超过分层调度的质量，且误差与生成质量的 Spearman 相关系数达 +0.88。

**⚠️ 局限性**

局限性：①模型仅考虑精确条件边缘的 oracle，未覆盖模型估计误差或近似采样；②对随机策略的下界尚未给出；③虽然在二进制字母上证明了到 1/2 的多项式深度，但线性深度是否可达仍未知；④实验仅在单一 0.5B 模型与温度 1 的设置下验证，未探讨更大模型或连续空间；⑤在非离散设置或高维符号空间下的推广仍是未来工作。

---

## 325. ScentEcho: Exploring Adsorbent Materials for Accurate Odor Collection and Playback

**arXiv ID:** 2608.25494 | [PDF](https://arxiv.org/pdf/2608.25494v1)

**作者:** Chih-Hung Lee `[一作]` (Tsinghua University), Qi Lu `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种名为 ScentEcho 的便携式嗅觉采集与播放系统，并评估了多种吸附材料在真实气味捕捉和重放中的性能。

**💡 创新点**

创新点在于将可模块化的气味采集与播放硬件与用户感知评估相结合，系统性检验吸附材料对气味相似度和强度的影响，并揭示两者在感知上的高度关联。

**🔧 技术方法**

采用了气动控制（空气泵与两路电磁阀）实现气味的吸附与释放，使用 Tenax TA、Carbopack X 等吸附材料，并通过 Likert 量表及 Pearson 相关系数进行定量评估。

**📊 数据集**

使用了四种常见气味样本（柠檬、咖啡豆、薄荷、薰衣草）进行实验，并未引用公开数据集。

**📈 对比分析**

通过用户实验比较不同吸附材料的相似度与强度评分；结果显示材料 D 在薰衣草气味上相似度最高（5.3），整体相似度与强度相关系数为 0.8295，说明该系统能在一定程度上实现高保真气味重放。

**⚠️ 局限性**

局限性包括样本量仅 12 人、重放气味强度普遍低于原始气味、使用不锈钢吸附管导致轻微金属味、泵未实现完全闭路操作、系统仅支持四通道，且未测试更广泛的气味种类。

---

## 326. SMART: MLLM-guided Temporal Alignment for Unifying Sign Language Recognition and Spotting

**arXiv ID:** 2608.25493 | [PDF](https://arxiv.org/pdf/2608.25493v1)

**作者:** Eunjee Choi `[一作]` (Dankook University), Younggeun Choi `[通讯]` (Dankook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 SMART 框架，联合实现连续手语识别（CSLR）和手语检测，通过 MLLM 生成的动作描述进行视频-文本对齐，使用 Multi-Scale Temporal Adapter（MSTA）提升 CLIP 视觉编码的时序建模，并通过 CSFormer 将识别得到的 gloss 概率注入到边界感知的检测网络，实现稠密帧级定位。

**💡 创新点**

创新点在于：①利用多模态大模型（MLLM）生成的运动描述作为辅助语义信号，解决 CTC 产生的尖峰对齐问题；②设计 MSTA 在 CLIP Transformer 末端捕获多尺度时序交互，保留预训练特征；③在检测分支中引入 CSLR 注入与双向交叉注意力机制，弥补稀疏识别信息与连续视觉特征的差距，实现识别与定位的互补。

**🔧 技术方法**

技术实现包括 CLIP ViT-B/16 视觉编码器、BERT 文本编码器、SigLIP 视频-文本对齐、CTC 损失、1D CNN + 双向 LSTM 语义编码、MSTA 多尺度时序操作、CSFormer 基于 ASFormer 的时间分割架构、双向交叉注意力和边界头。

**📊 数据集**

实验数据集包括四大手语基准：德国手语 PHOENIX14‑T、中文手语 CSL‑Daily、韩语大规模 KSL 以及灾害安全韩语 DS KSL，涵盖不同语言与应用场景。

**📈 对比分析**

与最新基线（VAC、SEN、AdaptSign 等）比较，SMART 在所有数据集上均取得 WER 最优结果（如 PHOENIX14‑T 19.50% / Large‑scale KSL 0.48%），并在手语检测任务中实现 F1@50 高达 96.72（Large‑scale KSL）与 59.77（DS KSL），显著优于传统 CTC 或单独分割模型。

**⚠️ 局限性**

局限性在于：①对大规模预训练模型（CLIP、MLLM）和显存受限的小批量训练依赖较高；②仅利用句子级 gloss 注释，仍缺乏细粒度帧级监督；③在不同语言与视频分辨率下的迁移性需进一步验证。

---

## 327. A Spectral Local-to-Global Principle for Spin Systems on Graphs with Girth At Least Five

**arXiv ID:** 2608.25491 | [PDF](https://arxiv.org/pdf/2608.25491v1)

**作者:** Xiaoyu Chen `[一作]` (Massachusetts Institute of Technology), Kuikui Liu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在最大度数为 Δ、girth 至少为 5 的图上，证明了当 q ≥ (1+δ)Δ（δ∈(0,1)）时，Glauber 动态在所有合法 q‑着色的均匀分布上快速混合，且在退火参数 β 的任意取值下（包括抗磁 Potts 模型）同样保持快速混合；同时给出了对 Δ 较大的图的最优 O(n log n) 采样时间。

**💡 创新点**

创新点主要包括：① 提出了一种新的基于 Bochner 恒等式的局部到全局谱原则，能够将全局谱间隙降低为对星形子图的局部谱间隙；② 通过傅里叶分析对星形子图的连续时间 Glauber 动态进行了精细的谱分析；③ 将之前需要 girth 至少 11 的条件降低到仅 5，并且把 δ 的依赖从指数改为多项式。

**🔧 技术方法**

核心技术：Bochner‐Bakry‐Émery 的离散形式、谱局部收缩（local spectral contraction）与全局扩张、傅里叶分解（Hoeffding 分解）在星形图上的精确计算、Schur 补法、以及对多自旋系统的泛化。

**📊 数据集**

该工作完全是理论性的，没有使用任何实验数据集；所有结论均通过解析证明得出。

**📈 对比分析**

与以往工作相比，本论文在 girt≥5 的条件下取得了与 Δ 成线性比例的快速混合，且混合时间为 O_δ(n^2 log q + n log 1/ε)，在 β=0 时为 O_δ(n^2 log q + n log 1/ε)，在 β>0 时额外加上 O_δ(n^2 Δ log 1/β)。相比之前需要 girt≥11 的结果，显著降低了图结构要求；相较于 spectral‐independence 方法，δ 的依赖由指数降为多项式。虽然距离 α^* Δ 仍有一定距离，但已逼近理论上最优阈值。

**⚠️ 局限性**

局限性：① 仅适用于 girth 至少为 5 的图；② 对 Δ 的要求仍然很高，需要 Δ 依赖于 δ 取得足够大；③ 仍未达到 α^* Δ（约 1.763Δ）的阈值；④ 对 4‑cycle（或更小环）图无法直接应用；⑤ 证明过程极为繁琐，主要依赖深奥的离散 Bochner 身份和傅里叶分析，难以推广到更一般的多自旋模型。

---

## 328. PonsRAG: A Pons-Inspired RAG Bridging Cognitive Islands for Coordinated Long Narrative Reasoning

**arXiv ID:** 2608.25486 | [PDF](https://arxiv.org/pdf/2608.25486v1)

**作者:** Rongchen Zhao `[一作]` (Sun Yat-sen University), Jingping Liu `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种三层检索框架 PonsRAG，旨在解决长篇叙事推理中的认知孤岛问题，支持跨层次证据检索与整合。

**💡 创新点**

创新点在于：①构建包含角色层、情节层和“桥接层”Pons的三层索引；②设计协调推理流水线（Query Anchor → Pons Awaken（Co‑HITS）→ Pons Match（匈牙利算法）→ Flow Filter），实现跨层次信息的联合激活、匹配与过滤；③通过Pons层实现角色与情节之间的语义桥接，显著缓解认知孤岛。

**🔧 技术方法**

主要技术包括：检索增强生成（RAG）框架、LLM（GPT‑4o‑mini）提取角色、事件与摘要、知识图谱构建、语义相似度与频率归一化、Co‑HITS、匈牙利算法、LLM基过滤模块。

**📊 数据集**

使用四个长篇叙事推理基准：NarrativeQA、∞BENCH（EN.QA、EN.MC）以及 NoCha（True/False 题）。

**📈 对比分析**

与 LLM、Naïve RAG（BGE‑M3、NV‑Embed‑v2、Qwen3‑Embed‑8B）以及结构化 RAG（RAPTOR、HippoRAGv2、Youtu‑GraphRAG、ComoRAG）对比，单步 QA 与多步 QA 均表现最佳；在多选任务上平均准确率提升约 11.6%，在多步推理中相对提升达 12.5%，显著优于最强基线。

**⚠️ 局限性**

局限性：目前仅在长篇叙事推理基准上验证；未对多跳 QA 或更通用长文本推理任务进行评估，需进一步扩展到更广泛的推理场景。

---

## 329. Beyond Scaling: Self-Evolving LLM Agents for Hardware Kernel Optimization via an Experience-Driven Workflow and Experience Graph Memory

**arXiv ID:** 2608.25570 | [PDF](https://arxiv.org/pdf/2608.25570v1)

**作者:** Siyuan Chen `[一作]` (City University of Hong Kong), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 KOPE，一个基于经验的硬件 kernel 优化框架，记录并重用优化轨迹。

**💡 创新点**

创新点是 Experience Graph Memory 记录决策-结果图，以及 Active Context Management & Injection 在固定 token 预算下检索相关经验。

**🔧 技术方法**

技术包括 LLM 代理、实验性经验图、检索与上下文管理、GLM‑5.2/Deepseek‑V4‑Pro。

**📊 数据集**

使用 AscendC CANN Bench 53 operator 1,060 案例的 Ascend 910C 设备。

**📈 对比分析**

与 CANNBot 和 CUDA‑Agent 对比，KOPE 在 GLM‑5.2 下 84.6% pass，1.54× 速度提升，token 消耗显著下降。

**⚠️ 局限性**

局限在实验仅一次运行、对不同硬件泛化未充分验证，以及需进一步量化跨任务经验迁移效果。

---

## 330. The Well-Being Palette: An Action-Word Selection Tool Designed for Low-Burden Reflection on Workplace Well-Being

**arXiv ID:** 2608.25527 | [PDF](https://arxiv.org/pdf/2608.25527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 331. Resilient Decentralized Wireless Federated Learning via Gradient Tracking with AdamW

**arXiv ID:** 2608.25535 | [PDF](https://arxiv.org/pdf/2608.25535v1)

**作者:** Nguyen Van Thieu `[一作]` (University of Luxembourg), Symeon Chatzinotas `[通讯]` (University of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了QEF-GT-AdamW算法，实现了在不可靠无线边缘网络中对非IID数据的鲁棒去中心化学习。

**💡 创新点**

首次将AdamW自适应优化与梯度跟踪、双流量化+误差反馈以及本地回退机制结合，既降低通信负载又提高在噪声、丢包环境下的收敛鲁棒性。

**🔧 技术方法**

采用梯度跟踪、AdamW自适应优化、Top‑K稀疏量化与误差反馈、基于混合矩阵的鲁棒混合、时隙同步与本地回退策略。

**📊 数据集**

在MNIST与CIFAR‑10的非IID划分上进行实验，分别采用标签偏斜和Dirichlet分布来模拟异构数据。

**📈 对比分析**

与CHOCO‑SGD、GT‑AdamW及QGT‑AdamW对比，QEF‑GT‑AdamW在同等通信压缩率下实现更高的测试准确率、更快的收敛速度，且在带宽或功率受限时包丢失率显著降低。

**⚠️ 局限性**

理论收敛分析仅覆盖凸目标，假设混合矩阵独立且通信失败独立；在非凸任务、异步或实际无线环境下的表现尚未验证，且时隙同步与压缩开销也可能限制大规模部署。

---

## 332. Constrained Maximum Entropy Contiguous Aggregations

**arXiv ID:** 2608.25533 | [PDF](https://arxiv.org/pdf/2608.25533v1)

**作者:** Roberto Bruno `[一作]` (University of Salerno), Ugo Vaccaro `[通讯]` (University of Salerno)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对给定概率分布 p，求在满足熵上限 R 的条件下，最大化连续聚合 q 的 Shannon 熵的最优聚合问题，提供了精确的动态规划解法和两种贪心近似算法。

**💡 创新点**

创新点在于：① 将“连续聚合”约束引入熵最大化问题，解决了信息源压缩中保留统计信息的实际需求；② 推导出两种贪心算法的加性与乘性近似保证；③ 给出动态规划算法的 Θ(2^n) 最坏复杂度分析，并与贪心算法在实例中的表现做对比。

**🔧 技术方法**

主要技术包括：动态规划（DP）求解连续聚合的精确解；贪心策略（AvoidMax 和 Min‑Min）通过局部合并最小熵损失的相邻块实现快速近似；熵降量 Δ_H 的分析、二元熵函数 h(·) 用于误差界定；复杂度分析与最优性证明。

**📊 数据集**

实验使用人工合成的概率分布（如 p=(0.25,0.22,0.12,0.20,0.21)、均匀分布 p=(0.1,…,0.1) 等）来演示两种贪心算法的相对性能；未使用公开真实数据集。

**📈 对比分析**

比较方法：对同一概率分布在不同熵上限 R 下，分别运行两种贪心算法，记录最终聚合的熵值并与精确解（由 DP 获得）对比；在示例中指出哪种算法更优；性能上，贪心算法的时间复杂度分别为 O(n) 与 O(n log n)，相较于 DP 的指数级复杂度具有显著优势。

**⚠️ 局限性**

限制：① DP 方案在最坏情况下时间和空间复杂度为 Θ(2^n)，不可行于大规模实例；② 贪心算法仅给出近似解，且在某些分布下可能退化为极差解（如全聚合 q=(1)）；③ 论文仅给出 NP‑hard 的猜想，缺乏正式证明；④ 近似误差是加性或乘性，若最优熵很小则加性误差可能占主导，影响实用性。

---

## 333. ClueWeaver: Reward-Guided Dual-Agent Evidence Reasoning for Compact LLMs on Literary Long Narratives

**arXiv ID:** 2608.25531 | [PDF](https://arxiv.org/pdf/2608.25531v1)

**作者:** Jihao Zhu `[一作]` (University of Aberdeen), Jin B. Hong `[通讯]` (University of Western Australia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ClueWeaver框架，用双代理（Finder与Interpreter）在紧凑本地模型上显式选择证据并进行推理。

**💡 创新点**

将证据选择与解释分离为可训练代理，并通过奖励引导强化学习和自校准机制提升证据保留和答案可靠性。

**🔧 技术方法**

使用检索感知分割、XML输出、GRPO奖励强化学习、自校准推理及多阶段流程。

**📊 数据集**

在DetectiveQA、∞Bench、LongBench v2和NoCha四大长叙事问答基准上进行实验。

**📈 对比分析**

与本地端到端读者、API读者和多种Agentic基线对比，ClueWeaver在本地模型上整体准确率达59%，比同规模读者高6.4点，且在LongBench v2上比最佳API读者高11.5点。

**⚠️ 局限性**

对极远距离多跳证据的检索和整合仍显不足，部分多跳或表面重叠弱的案例难以准确回答。

---

## 334. Query Expansion Is More Than Generation: Improving Dense Retrieval through Better Integration

**arXiv ID:** 2608.25521 | [PDF](https://arxiv.org/pdf/2608.25521v1)

**作者:** Siyuan Sun `[一作]` (University of Arizona), Mihai Surdeanu `[通讯]` (University of Arizona)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AnchorQE和SC-AnchorQE两种无训练、无重建索引的查询扩展集成方法，先将原始查询和扩展分别编码后线性插值，并通过无监督在线策略估计插值系数；

**💡 创新点**

创新点在于将查询与扩展在向量空间中明确分离并可控地插值，解决传统扩展集成导致性能下降的问题；并提出只需在前8条未标注查询上估计全局插值系数的无监督在线校准方法；

**🔧 技术方法**

技术包括：双编码器稠密检索、向量归一化、线性插值、无监督在线校准（使用检索得分和相似度的乘积估计信任度）、与传统加权CombSUM等方法的等价证明；

**📊 数据集**

使用的评估数据集包括TREC-DL（TREC 2019/2020）、LoTTE（Search、Forum）以及BEIR-14（14个多领域子数据集）；

**📈 对比分析**

与传统扩展-仅扩展、文本拼接、分路融合、以及QuDAR等基线进行对比，AnchorQE/SC-AnchorQE在所有20组实验中均优于传统集成方法，最高提升约12.9%（TREC 2020）或13.0%（BEIR-14）；与固定插值系数α=.15相比，SC-AnchorQE在17/20组实验中均更好；在单向检索请求下与加权CombSUM等价，且延迟更低；

**⚠️ 局限性**

局限性：1）在线校准假设前8条未标注查询能代表后续流，若分布漂移会导致系数失效；2）单一全局系数无法对极强或极误导性扩展做细粒度控制；3）性能提升仍受生成质量限制；4）实验使用严格的无前瞻模拟，真实部署中需进一步验证；5）校准过程中仍需要额外两次检索探测，成本略增。

---

## 335. Asymmetric Cross-Modal Fine-Grained Visual Categorization: ACF-Net and the BirdPro Benchmark

**arXiv ID:** 2608.25520 | [PDF](https://arxiv.org/pdf/2608.25520v1)

**作者:** Bohan Deng `[一作]` (Great Bay University), Zitong Yu `[通讯]` (Great Bay University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种面向非对齐音视频的细粒度视觉分类框架 ACF-Net，并构建了新的鸟类音视频基准数据集 BirdPro。

**💡 创新点**

创新点在于：①引入光流引导的运动模块 OFGM，突出视觉动态信息；②设计不对称跨模态自适应融合 ACAF，根据预测不确定性估计模态可靠性，实现弱配对条件下的鲁棒融合；③在非严格对齐场景下实现细粒度音视频分类，并在 BirdPro 上验证。

**🔧 技术方法**

技术包括光流计算 (StreamFlow)、ViT-B/16 视觉编码器、音频 Log‑Mel 处理、光流掩码增强、基于熵的模态可靠性评估、prototype 与 decoupling 正则化，以及多模态交叉熵损失组合。

**📊 数据集**

数据集：BirdPro，涵盖 194 种鸟类，包含 11,965 条视频、1,919 条音频及 470 对音视频样本，且仅保证类别级一致性。还使用公开的 CUB、ImageBind 等数据作为对比基准。

**📈 对比分析**

与多种基线（FG‑CLIP、ImageBind、CoC、MDL、UAF 等）对比，ACF‑Net 在融合模式下取得 87.23% 的准确率，比最强基线提升 2.97%；在音频与视频匹配失效（mismatch）情况下，提升 1.92%。在单模态下亦实现显著提升。

**⚠️ 局限性**

局限性：①目前仅针对鸟类数据，泛化性未知；②对光流计算和不确定性估计依赖额外计算资源；③在极度嘈杂或无声音的场景下仍可能受限。

---

## 336. Pose-Anchored Optical Flow for Low-Latency Human Action Anticipation in Human-Robot Teaming

**arXiv ID:** 2608.25495 | [PDF](https://arxiv.org/pdf/2608.25495v1)

**作者:** Lewis de Zoete Grundy `[一作]` (Swinburne University of Technology), Christopher Fluke `[通讯]` (Swinburne University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于姿势锚定的光流表示 PoseOFF，并在多种骨架动作识别模型中验证其对早期动作预测的提升。

**💡 创新点**

创新点在于将光流特征局部采样并与人体关节点对齐，形成结构化、与运动学紧密对应的局部运动表示，从而在保持轻量化的同时提升动作辨识细粒度。

**🔧 技术方法**

使用了 RAFT 光流、YOLO‑POSE 骨架估计、CNN 嵌入层与现有骨架模型（InfoGCN++、MS‑G3D、ST‑GCN++）的结合，进行多尺度局部光流编码。

**📊 数据集**

实验数据集包括 NTU RGB+D 60、NTU RGB+D 120 以及 UCF101（通过 YOLO‑POSE 估计骨架）。

**📈 对比分析**

通过在完整序列与不同观察比例下对比基线模型，使用准确率和 AUC 评估；PoseOFF 在 10%–80% 观察比例下平均提升 2%–12% 的准确率，并能在更少帧下匹配基线性能。

**⚠️ 局限性**

局限性包括对姿态估计和局部光流质量高度依赖；在严重遮挡或强全局运动场景下表现下降，且相较于纯骨架方法仍有额外计算开销。

---

## 337. Social Network Structure, Wealth, and Wealth Inequality Across Cultures

**arXiv ID:** 2608.25488 | [PDF](https://arxiv.org/pdf/2608.25488v1)

**作者:** Eleanor A. Power `[一作]`, John P. Ziker `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在46个多样化社区收集约3500个共享单元（即家庭）的物质财富与社会网络（借贷、共享食物、合作等）数据，探究财富与社交资本的关系以及财富不平等与网络结构的关联；

**💡 创新点**

创新点在于：①跨文化、覆盖非WEIRD、与市场经济整合程度不同的社区；②使用多层次、面向日常生活的支持网络数据；③引入“经济连通度”（Average Alter Wealth）等新网络指标，并与财富不平等进行系统比较；

**🔧 技术方法**

采用描述性统计、相关系数、随机效应元分析、网络模块化与财富模块化、Gini系数计算等定量方法，对网络-财富关系进行稳健性检验；

**📊 数据集**

自定义数据集：46个社区的家庭层面资产清单、支持网络（食物、信息、金钱等）数据，结合社区环境、制度和经济属性等宏观变量；

**📈 对比分析**

通过跨社区元分析与稳健性检查比较，发现大多数社区中财富较高的单位拥有更多、质量更好的网络连接；财富不平等与贫富之间的网络连接程度呈负相关，表明“经济连通度”是不平等的重要预测因子；

**⚠️ 局限性**

局限性包括：数据为横断面，无法确定因果关系；仅覆盖当地社区内部网络，未纳入外部网络影响；样本尽管多样但仍有缺口；网络测量依赖于受访者自述，可能存在偏差。

---

## 338. Overview of SHROOM-Visions 2026: A Shared Task on Hallucination Detection in Large Vision-Language Models

**arXiv ID:** 2608.25662 | [PDF](https://arxiv.org/pdf/2608.25662v1)

**作者:** Raúl Vázquez `[一作]` (University of Helsinki), Timothee Mickus `[通讯]` (University of Helsinki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了SHROOM-Visions 2026共享任务，评估多语言、多类别视觉语言模型的幻觉检测。

**💡 创新点**

利用人类编写的SHEEP数据集实现模型无关评估，并细粒度四语言幻觉分类。

**🔧 技术方法**

采用字符级概率预测、BIO标注、LoRA微调、LLM裁决等多种模型和集成方法。

**📊 数据集**

SHEEP数据集，包括20,000个中英法意四语样本，涵盖随机、MAP和人类写作的图文生成。

**📈 对比分析**

通过Spearman相关、标签相关和IoU三指标评估，最佳系统在字符相关0.58、标签相关0.46、IoU0.51，优于基线30–40点。

**⚠️ 局限性**

评测不稳定、抽样策略影响绝对分数，且多数系统对无幻觉实例的拒绝行为未被量化。

---

## 339. Narcissus: Program Synthesis Using Context-Aware LLM Approximations

**arXiv ID:** 2608.25657 | [PDF](https://arxiv.org/pdf/2608.25657v1)

**作者:** Tilman Hinnerichs `[一作]` (Delft University of Technology), Neil Yorke-Smith `[通讯]` (Delft University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种基于LLM生成的提案的上下文感知搜索启发式（Narcissus），使枚举式程序合成器在不再调用LLM的情况下即可利用提案信息；

**💡 创新点**

创新点在于：①将LLM提案保留为完整的语法树而非简化为规则频率；②设计三种上下文感知评分信号（前缀对齐、子程序重用、正则化），并引入正则化防止误导提案导致搜索陷阱；③通过修复、提取重复子程序并扩展语法实现更强的搜索空间；

**🔧 技术方法**

技术包括：LLM提案采样与语法树解析/修复；子树提取并作为宏规则扩展语法；构造基于前缀匹配、重用计数和大小正则化的评分函数；在两类搜索后端（基于遗传的自上而下搜索和基于束搜索的自下而上枚举）中实现；

**📊 数据集**

数据集涵盖五个领域：SLIA、BV、DeepCoder、ARC（使用Hodel DSL）以及ARGA（ARC的对象化子集），每个领域提供不同难度和提案支持率的任务；

**📈 对比分析**

与无指导枚举、增强的BFS、直接LLM提案采样、重新提示LLM以及HySynth的静态规则频率指导等基线进行对比；实验显示Narcissus在所有任务上都能在程序枚举量和运行时间上优于静态指导，并在低提案支持率场景下仍不劣于无指导搜索；在SLIA中达到的任务完成率为约48%，比静态指导高约10%；在ARC中从13%提升到40%；

**⚠️ 局限性**

局限性包括：①评分函数的权重和正则化阈值是手工设定的，缺乏自适应机制；②对极度稀疏或错误提案的鲁棒性仍有限（正则化仅缓解但不能完全消除）；③提案中提取的宏规则仅在单个任务内使用，未能跨任务共享；④仅在已知目标语言且可定义CFG的情形下适用，难以处理更自由的程序结构。

---

## 340. AutoVerifier: Residual-Guided Non-Parametric Optimization for Reference-Based Answer Verification

**arXiv ID:** 2608.25637 | [PDF](https://arxiv.org/pdf/2608.25637v1)

**作者:** Zebei Zhao `[一作]` (University of Science and Technology of China), Minqi Shi `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种残差引导的非参数优化框架，通过在构造阶段回放验证错误来学习并生成可审计的规则卡，随后将可确定的比较逻辑升级为代码模块，非确定的部分保留为提示指导，最终在固定模型回退的基础上实现高效的参考式答案验证。

**💡 创新点**

将验证错误转化为可重复、可审计的规则卡，并通过回放验证确保无回归；实现将确定性比较逻辑编译为代码模块，同时保留需要模型判断的逻辑为提示，从而兼顾可解释性与覆盖率。

**🔧 技术方法**

残差引导的非参数优化、规则卡（rule cards）、代码模块与提示指导、回放验证、固定模型回退机制、覆盖率与错误聚类。

**📊 数据集**

四个参考式验证基准（VerifyBench、VerifyBench‑Hard、SCI‑VerifyBench、VerifierBench）；构造数据集包含 VAR、TIGER‑Lab、SuperGPQA、WebInstruct 及自建诊断样本。

**📈 对比分析**

在与 GPT‑5.4‑Mini 同一二元验证接口下进行对比；Prompt+Code 方案在四个基准上取得 93.05% 宏平均准确率，较 Prompt‑Only 提升 1.76 点，回退调用率下降 32.13%；相较最佳工具/模型方案仅低 1–2%，但在可解释性和可维护性上具有明显优势。

**⚠️ 局限性**

仅适用于固定的参考式二元验证，依赖预先设定的构造池和冻结的推理接口；无法处理多参考、多评分或复杂分级的验证需求；未来需要扩展至更丰富的评估标准、改进构造过程的可复现性和可视化、以及持续回放测试。

---

## 341. SeVeR: Selective Visual Exposure and Retrieval for 3D Medical Image Question Answering

**arXiv ID:** 2608.25630 | [PDF](https://arxiv.org/pdf/2608.25630v1)

**作者:** Yaojun Hu `[一作]` (DAMO Academy, Alibaba Group), Ling Zhang `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了规模较大的多序列乳腺MRI VQA 基准（BreMRIs-VQA），并提出了一个包含贪婪原型选择（GPS）、变更感知门控注意力（CaGA）和自一致性正则化（SCR-MU）的选择性视觉曝光与动态证据检索框架，用于解决多模态体素 VQA 问题。

**💡 创新点**

创新点在于将稀疏原型选择与变更感知门控注意力结合，使模型在不依赖全量视觉标记的情况下实现跨序列信息整合，并通过边际效用自一致性正则化防止无意义检索，提升了多模态融合与推理效率。

**🔧 技术方法**

采用共享 3D Vision Transformer 编码器、贪婪原型选择 (GPS)、变更感知门控注意力 (CaGA) 与边际效用自一致性正则化 (SCR-MU) 等技术，并利用大规模 LLM 辅助的 QA 生成与验证流程。

**📊 数据集**

使用了 12,891 名患者的 71,041 条多模态乳腺 MRI 序列，配合双资深放射科医生验证的放射报告与病理报告，构建了 1.19M 个 QA 对（671.6K 自由文本 + 515.1K 多选）。

**📈 对比分析**

在 BreMRIs-VQA 上与多种基线（Qwen3-VL、HuLu-Med、M3D 等）对比，平均多选准确率提升约 20% 且生成质量显著提高；在公开 3D-RAD 与 DeepTumorVQA 转移评测中也取得最优或接近最优的表现，验证了框架的通用性。

**⚠️ 局限性**

局限包括对自由文本评估仅依赖自动指标缺乏临床专家人工评判、原型选择固定预算未自适应序列复杂度，以及未在真实放射科工作流程中进行集成验证。

---

## 342. AWM: Answerable Working Memory for Long-Document VQA Agents

**arXiv ID:** 2608.25618 | [PDF](https://arxiv.org/pdf/2608.25618v1)

**作者:** Dongzhuoran Zhou `[一作]` (University of Oslo), Evgeny Kharlamov `[通讯]` (University of Oslo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于终端工作内存可回答性（memory‑only answerability）的诊断指标，并利用该指标设计一种新的强化学习奖励机制 AWM‑GRPO，用来优化长文档视觉问答代理的工作记忆质量。

**💡 创新点**

创新点在于：①发现现有评测仅关注最终答案和证据页访问，忽视了工作记忆是否能单独支持答案；②引入可回答性指标作为诊断和奖励信号；③通过 GRPO 对奖励进行归一化，使代理在保持答案正确的同时优先保留可回答的终端记忆，从而在多个维度提升性能。

**🔧 技术方法**

技术实现包括：Qwen3‑VL‑4B 作为主代理模型；RAG Top‑3 检索器；GRPO（group‑relative policy optimization）强化学习框架；冻结的 Qwen3‑14B 读者模型用来生成终端记忆答案；官方评测判定器 J 用于计算奖励与指标；训练数据来自 Doc‑750K 文档问答数据集。

**📊 数据集**

使用的主要数据集为 MMLongBench‑Doc（1082 条问答）和 LongDocURL（2325 条问答），并在内部使用 500 条答案可行的子集进行诊断与对比；检索部分基于 Jina + Qdrant 的页面图像嵌入。

**📈 对比分析**

与直接输入、VLM+RAG Top‑3、SFT 以及仅使用答案奖励的 GRPO 进行对比；在 Qwen3‑VL‑4B 条件下，AWM‑GRPO 在 MMLongBench‑Doc 上从 45.4% 提升到 53.9%（+8.1点），在 LongDocURL 上从 48.2% 提升到 60.1%（+11.9点）；终端记忆的可回答率也从 42.5% 提升到 44.5%，P_mmc（正确答案但记忆不可回答）从 19.9% 降到 17.2%。

**⚠️ 局限性**

局限性包括：仅在 4B 规模模型上验证，未评估更大模型或不同文档类型；可回答性评估依赖冻结的 Qwen3‑14B 读者，可能对读者质量敏感；官方判定器的提取步骤可能引入误差；奖励仅衡量可回答性，未完全覆盖源头归因与细粒度记忆一致性。

---

## 343. Frequency-aware forecasting for short-term typhoon gust prediction

**arXiv ID:** 2608.25604 | [PDF](https://arxiv.org/pdf/2608.25604v1)

**作者:** Xuefei Wang `[一作]` (Hubei University), Shengjun Zhang `[通讯]` (Hubei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 WDANet，一个利用站式小波分解与 FiLM 交叉频率调制的双分支注意力网络，用于台风环境下的短期风 gust 预测。

**💡 创新点**

创新点包括：1）采用非衰减的 Stationary Wavelet Transform 保持时移不变性；2）在低频与高频分支之间使用 FiLM 实现自适应调制；3）双分支编码器-解码器结构结合全局与局部注意力，精准捕捉趋势与突发风 gust 的多尺度特征。

**🔧 技术方法**

技术手段涵盖：站式小波分解、可学习滤波器、双分支 BiLSTM 编码器、全局+局部注意力机制、FiLM 交叉频率调制、线性输出融合等。

**📊 数据集**

使用 1960–2025 年 ERA5 重分析数据（0.25°×0.25° 网格）中的海平面压、温度、湿度、10 m 风 gust 等七个变量，构建 400 条台风事件的历史–预测样本，输入长度 24 h，预测 horizon 24 h。

**📈 对比分析**

通过与 CNN‑LSTM、Autoformer、TimesNet、iTransformer、PatchTST、DLinear 等深度学习 SOTA 以及 ECMWF‑HRES 物理模型比较，WDANet 在 24 h 内 RMSE、MAE 均优于所有数据驱动模型，且在前 6 h 内 RMSE/MAE 低于 ECMWF‑HRES，极端 gust 峰值误差最小。

**⚠️ 局限性**

局限性包括：仅基于历史 ERA5 数据，缺乏实时观测输入；未引入空间信息，难以直接迁移到多站点或不同海域；极端高 gust 峰值预测仍受训练样本稀缺影响，尚存在误差。

---

## 344. RA-VLA: Retrieval-Augmented VLA for Test-Time Adaptation

**arXiv ID:** 2608.25585 | [PDF](https://arxiv.org/pdf/2608.25585v1)

**作者:** Sanghwan Jang `[一作]` (POSTECH), Hwanjo Yu `[通讯]` (POSTECH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种检索增强的 Vision‑Language‑Action（VLA）框架，能够在不进行模型权重更新的情况下，利用少量专家演示实现对新任务的测试时适应。

**💡 创新点**

创新点在于（1）通过行为对齐损失将检索空间与动作相似性对齐，避免仅靠视觉相似性导致的功能性错误；（2）提出上下文遵从损失，使得策略在生成动作时真正依赖检索到的专家片段，克服行为惯性；（3）采用独立编码与缓存的检索方式，使推理延迟不随检索片段数量增加而显著增长。

**🔧 技术方法**

技术包括：VLM 编码器 + Diffusion Transformer 动作头（流匹配 VLA 结构）、两层 Transformer 检索器、动态时间规整（DTW）生成正样本、对比学习实现行为对齐、基于回归边距的上下文遵从损失、预编码缓存机制。

**📊 数据集**

数据集：LIBERO benchmark（四个任务套件，共 40 个任务）和真实 UR5e 机器人环境（四个日常任务）。在实验中使用 30 条专家演示进行训练，评估时分别使用 3–5 条演示做为上下文。

**📈 对比分析**

与 Vanilla VLA、RAEA、RICL 等基线进行对比。实验显示在 LIBERO 上成功率从 20.9% 提升至 38.5%（+17.6%），在 UR5e 上从 35.4% 提升至 56.3%（+20.9%），并且推理延迟几乎保持不变，可支持高频控制。

**⚠️ 局限性**

局限性：需要维护可靠的专家演示缓存，若演示被恶意篡改可能导致错误行为；检索性能仍受限于 VLM 的视觉特征，极少见或全新动作仍可能检索不到合适片段；对非常小的数据集或极端外域任务的适应性尚待进一步验证。

---

## 345. GRIP: Granular Reward-Guided Parameter Interpolation for Efficient Reasoning

**arXiv ID:** 2608.25583 | [PDF](https://arxiv.org/pdf/2608.25583v1)

**作者:** Lam So `[一作]` (Peking University), Han Lin `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GRIP框架，通过冻结已训练的推理模型和指令模型，只学习模块级插值比例，既保持推理准确性又显著降低输出长度。

**💡 创新点**

创新点在于使用奖励导向的模块级参数插值，仅更新少量比例参数，利用长度与正确性奖励实现高效推理，并揭示不同层/模块的最优混合比例。

**🔧 技术方法**

采用模块级Sigmoid插值、基于GRPO的强化学习奖励优化、长度归一化奖励、与黑盒搜索（CMA‑ES）对比、Qwen3 4B模型和LightEval评估框架。

**📊 数据集**

训练使用DeepScaleR-preview；评测采用AIME25、MATH500、GSM8K、GPQA‑D和LiveCodeBench等五个推理基准。

**📈 对比分析**

与原始Qwen3-Thinking、Qwen3-Instruct、线性插值、SLERP、TIES、DARE‑TIES、DELLA以及CMA‑ES等基线对比，GRIP在保持或提升准确率的同时平均减少约27% token，优于固定比例和搜索方法。

**⚠️ 局限性**

局限性包括仅在4B规模验证，未测试更大模型；仅适用于密集Transformer架构，未验证Mixture‑of‑Experts或跨家族模型融合。

---

## 346. Are Concept Bottleneck Models Effective as Decision-Support Systems?

**arXiv ID:** 2608.25581 | [PDF](https://arxiv.org/pdf/2608.25581v1)

**作者:** Alessandro Bogani `[一作]` (University of Trento), Andrea Passerini `[通讯]` (University of Trento)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过两项大规模用户实验（共705名参与者），评估概念瓶颈模型（CBM）在二分类任务（邮件诈骗识别与鸽子种类识别）中的决策支持效果，比较了无支持、仅标签、非交互式概念、交互式概念四种AI支持条件，记录了准确率、置信度和信任度等指标。

**💡 创新点**

创新点在于：①首次提出可跨数据集应用的CBM决策支持实验范式；②系统性探讨交互式CBM在不同任务难度、概念清晰度及用户交互意愿下的性能差异；③揭示概念检测错误可能削弱用户信任，并给出针对性的部署建议。

**🔧 技术方法**

技术实现包括：CBM的概念编码器（冻结的神经网络+SVM分类器）与线性任务预测器（逻辑回归），以及基于混合效应回归（logistic 与 ordinal）的统计分析，配合注意力检查与交互式概念修改界面。

**📊 数据集**

使用的数据集为（1）邮件诈骗数据集，包含合法与诈骗邮件以及情绪/动机概念；（2）CUB鸟类图像数据集，聚焦两种麻雀种类，概念为可视化属性。每个数据集均选取六个关键概念供CBM使用。

**📈 对比分析**

比较方法：在四种支持条件下计算参与者准确率，并采用混合效应回归与配对事后检验评估显著性。结果显示，交互式CBM在任务难度大、概念易识别且用户积极交互时，比无支持或仅标签条件提高约8–10%的准确率；交互式还提升了用户置信度，但在部分情形略降低了对模型的信任。

**⚠️ 局限性**

局限性包括：①仅研究二分类任务，未探讨多分类场景；②概念数量受限，未评估大规模概念对认知负荷的影响；③未系统性检验概念检测准确度对信任与性能的影响；④数据集与任务场景可能不具备普适性。

---

## 347. DBcover: A White-box SQL Test Generation Framework for Coverage Improvement

**arXiv ID:** 2608.25573 | [PDF](https://arxiv.org/pdf/2608.25573v1)

**作者:** Yankai Rong `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于大型语言模型的白盒 SQL 测试生成框架 DBcover，利用上下文推理提升关系型数据库的代码覆盖率。

**💡 创新点**

创新点在于：①构建全局 SQL‑to‑path 对应与调用图的知识图谱；②两阶段测试生成——先通过知识图谱选择最贴近目标函数的种子，再用细粒度提示链条引导 LLM 合成触发未覆盖路径的 SQL；③在资源受限环境下仍能使用 32B 参数模型。

**🔧 技术方法**

核心技术包括：大型语言模型（LLM）推理、轻量级动态分析、静态调用图构建、知识图谱查询与推理、分阶段提示工程。

**📊 数据集**

使用公开数据库系统 PostgreSQL 8.0.33、MySQL 8.0.33 以及商业闭源 KingbaseES；测试数据来源于各系统的回归测试套件与自建 SQL 生成。

**📈 对比分析**

与基线（仅执行回归测试）、SQUIRREL、ShQveL 等方法对比，DBcover 在 PostgreSQL 和 MySQL 的行覆盖率分别提升至 80.1%/82.3%，比基线高约 11%，比其他方法高 6–10%；在 KingbaseES 上亦实现约 20% 的提升。

**⚠️ 局限性**

局限性包括：无法触及仅通过 OS 事件或内部错误处理路径的代码；对 LLM 的推理精度仍依赖于 seed 质量；在极大规模代码库中仍需大量动态分析；以及对闭源系统的部署受限于内部许可。

---

## 348. DCEO: Direct Causal Effect Optimization for Long-Term User Value Modeling in E-commerce Search

**arXiv ID:** 2608.25635 | [PDF](https://arxiv.org/pdf/2608.25635v1)

**作者:** Junzhao Zhang `[一作]` (Taobao & Tmall Group of Alibaba), Haihong Tang `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 DCEO 框架，用上下文感知的 actor 生成权重，将多源预测分数融合成项级代理分数，并通过 critic 估计用户级最终目标，实现对代理指标的因果效应优化。

**💡 创新点**

创新点在于：① 直接优化代理指标对用户长期目标的相对因果效应而非仅预测相关性；② 采用 actor‑critic 结构结合代理指标校准与归一化，提供可解释的上下文依赖权重；③ 仅在在线阶段部署轻量化 actor，避免训练-上线不一致。

**🔧 技术方法**

主要技术包括多层感知机（MLP）实现 actor/critic/校准/归一化模型；softmax 权重生成；相对因果效应（RCE）估计；Bradley–Terry 排序损失；在线融合式多目标融合公式。

**📊 数据集**

使用阿里巴巴电商搜索日志，构造用户级训练样本（含 17 种预测分数），在 14 天训练、1 天验证，最终在 41 天线上 A/B 测试评估。

**📈 对比分析**

通过离线 RCE 指标评估，actor 损失对比预测关联、因果优化及加入归一化排序损失，最终 RCE 达 0.053；线上 A/B 测试显示相较传统 GMV 代理，GMV 提升 0.36%，点击+0.36%，购买+0.12%。

**⚠️ 局限性**

局限性包括：RCE 依赖 critic 的准确性和观察数据的无偏，难以完全估计因果效应；校准至固定印象数忽略排名导致的印象数变动对最终目标的影响；actor 受限于仅对已提供的 17 个预测分数加权，表达能力受限；仅在阿里巴巴平台验证，泛化性待进一步验证。

---

## 349. Using profiles of cognitive capability to assess AI suitability for workplace tasks

**arXiv ID:** 2608.25623 | [PDF](https://arxiv.org/pdf/2608.25623v1)

**作者:** Jonathan Prunty `[一作]` (University of Cambridge), Lucy Cheke `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于共享认知能力空间的管线，用于从已标注的基准测试中推断 AI 系统的能力，并通过专家问卷获取工作任务对这些能力的需求，从而评估 AI 在不同工作场景中的适用性。

**💡 创新点**

创新点在于将传统 AI 基准的任务表现转化为对认知能力的需求标注，构建可度量 AI 能力与任务需求的共同尺度，并采用 Bayesian 项目反应模型与软最小池化实现能力估计，最终实现任务-能力匹配的可解释、可更新评估。

**🔧 技术方法**

使用技术包括基于 LLM 的需求标注工具、Bayesian Measurement Layouts 进行能力推断、软最小池化温度调节、加权几何平均的适用性映射，并在此基础上进行任务重要性加权与部署优先级计算。

**📊 数据集**

数据集涵盖约19,576条来自 28 个公开基准（如 BIG-bench、AGIEval、MMLU 等）的评测项目；人工合成的 20 个代理用于验证；六个现代 LLM（Gemini 3、GPT-4o-mini 等）用于实际能力评估；并收集了 410 名员工在六个职业域内的问卷数据。

**📈 对比分析**

通过对比模拟代理的能力恢复率、对六个 LLM 在八个聚合认知维度上的后验均值以及对各工作任务的适用性分数，实验表明 Gemini 3 系列在规划与控制维度上显著优于其他模型，且整体适用性排序在不同任务间保持高度稳定，最高模型在大多数工作场景中的部署优先级均位列前列。

**⚠️ 局限性**

主要限制包括基准测试多为文本任务，导致对多模态与交互式能力覆盖不足；任务需求仍以重要性而非实际需求评估，可能与 AI 实际所需能力不完全对齐；依赖员工自我报告的认知重要性，受意识层面限制；此外模型推断对极端任务的预测尚未通过实际部署验证。

---

## 350. Deep Learning Segmentation of Diffusion-Weighted MRI Acute Ischaemic Stroke: A Pragmatic Evaluation Across Three Datasets

**arXiv ID:** 2608.25675 | [PDF](https://arxiv.org/pdf/2608.25675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 351. JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution

**arXiv ID:** 2608.25593 | [PDF](https://arxiv.org/pdf/2608.25593v1)

**作者:** Guibin Zhang `[一作]` (LV-NUS Lab), Shuicheng Yan `[通讯]` (LV-NUS Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 JIT-Agent，一种可训练的 harness intelligence 模型，能够在推理时即时生成、修复并演化任务特定的代理 harness，提升底层 LLM 的整体能力。

**💡 创新点**

创新点在于将“harness intelligence”定义为可学习的、可进化的操作框架，并通过四模块可组合协议实现从代码层面到执行层面的即时合成与自我改进，突破传统 AOT 的固定架构限制。

**🔧 技术方法**

采用四模块（记忆、规划、行动、能力）协议、HarnessFactory 生成示例、三阶段训练（定制、修复、Evo‑GDPO）以及大规模 LLM（Qwen3.6‑27B）微调，配合 PPO‑style 的演化策略实现在线提升。

**📊 数据集**

训练使用多任务分布 𝒟_task，评估基于九个 benchmark（DeepSearchQA、DeepPlanning‑Shopping、OfficeBench 等），并对 GLM‑5.2、DeepSeek‑V4‑Flash 等多种 backbone 进行测试。

**📈 对比分析**

通过与 GPT‑5.6、Gemini、Claude Code 等领先模型以及多种固定 harness（Codex、OpenCode、Hermes 等）对比，JIT‑generated harness 在 18 组 backbone‑benchmark 对中平均提升 7.7 分，在多数基准上超过前沿模型，同时在成本‑性能图中实现 20‑50% 的成本下降。

**⚠️ 局限性**

局限性包括：需预先手工定义四模块协议，生成质量受限于训练数据覆盖范围；在极端或未见工具的任务中仍可能生成失效 harness；缺乏长期自我学习循环，无法完全实现持续自适应。

---

## 352. Individual Fairness in Hierarchical Clustering

**arXiv ID:** 2608.25586 | [PDF](https://arxiv.org/pdf/2608.25586v1)

**作者:** Binita Maity `[一作]` (Indian Institute of Technology), Shrutimoy Das `[通讯]` (Indian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了个体公平性在层次聚类中的可行性问题，给出了基于占优超度量的公平性定义，并证明了最小乘法松弛α*的理论界限。

**💡 创新点**

提出了局部公平性阈值α_mut(k)与全局可行性之间的本质分离，证明α*非递减且在某些度量上必须达到Θ(log n)的扭曲；同时给出了可实现的FRT-和FCAC算法。

**🔧 技术方法**

利用超度量、k近邻邻域、树嵌入理论、稳定性分析和自适应合并的公平聚类算法FCAC。

**📊 数据集**

在合成数据（高斯混合、随机3正则图）和真实数据（Adult、German Credit、Iris）上验证。

**📈 对比分析**

与FRT树嵌入比较：FCAC在实验中取得接近1的α（满足所有邻域约束），而FRT在相同数据上表现出更大的平均α，证明FCAC在保持公平性时更有效。

**⚠️ 局限性**

主要局限在于可行性判定的计算复杂性未知、对大规模数据的时间复杂度（O(n³k)）较高，以及对非欧几里得或高维数据的泛化仍待探索。

---

## 353. V-Rubrics: Visual Faithfulness via Rubric-Based Reinforcement Learning

**arXiv ID:** 2608.25580 | [PDF](https://arxiv.org/pdf/2608.25580v1)

**作者:** Shulin Tian `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出视觉Rubrics‑based RL框架，通过将回答拆解为视觉真实性、推理一致性和指令遵循等原子条目，实现细粒度奖励并在视觉问答与推理任务中提升模型表现。

**💡 创新点**

创新点在于将视觉真值拆分为可验证的原子标准，构建50k结构化Rubric数据集，并通过前缀局部化奖励让RL能针对每个视觉断言进行局部信用分配。

**🔧 技术方法**

采用Qwen3‑VL‑8B‑Instruct模型，结合SFT + GRPO、前缀局部化优势、LLM‑as‑judge（Qwen3‑VL‑235B‑A22B）以及自研的V‑Rubrics 50K数据集等技术。

**📊 数据集**

主要使用公开的视觉问答与推理数据集（OpenMMReasoner、MMBench、MMMU、MathVista、LogicVista等），构建了50,248例的V‑Rubrics 50K。

**📈 对比分析**

与SFT基线及仅基于答案奖励的GRPO比较，在MMMU、MathVision、LogicVista等视觉推理基准上提升了约4–6%点，整体平均准确率提高1.8点，尤其在需要中间视觉证据的任务上表现显著优于基线。

**⚠️ 局限性**

局限包括对自动生成Rubric和LLM判定质量的依赖、前缀信用定位不够精确、潜在判定模型偏差以及未经过大规模人工审核验证。

---

## 354. Opportunities of Self Supervised Learning for GNSS: Evaluation of a Deep Learning-Enhanced PVT Algorithm

**arXiv ID:** 2608.25674 | [PDF](https://arxiv.org/pdf/2608.25674v1)

**作者:** Thomas Barbero `[一作]` (Abbia GNSS Technologies), Bertrand Ekambi `[通讯]` (Abbia GNSS Technologies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种深度学习增强的PVT（DLE-PVT）算法，用于在密集城市环境中缓解多路径干扰，并给出误差不确定度。

**💡 创新点**

创新点包括：
1) 将JEPA自监督框架应用于GNSS观测，实现大规模无标签预训练；
2) 通过异方差回归同时预测码修正和不确定度；
3) 将网络预测嵌入加权最小二乘（WLS）PVT求解器，实现多路径感知定位；
4) 在多种离散和未知环境下验证自监督预训练对鲁棒性的提升。

**🔧 技术方法**

使用的技术包括：
- Transformer（4层编码器+2层解码器）
- JEPA + VICReg 正则化的自监督掩码建模
- 异方差（Laplace）回归损失
- GTSAM 基于因子图的Levenberg–Marquardt求解
- WLS 多路径感知定位（MAPA）

**📊 数据集**

数据集：
- 本地收集的24小时Toulouse城市行驶数据（约1.35M token）
- PPC Tokyo 与 UrbanNav Hong Kong 的公开数据（约230k token）

**📈 对比分析**

与传统基线（无修正的初始WLS PVT）进行比较，使用 50th 与 95th 百分位的 3D 定位误差作为评估指标。结果显示：
- 在ID（训练分布）场景中误差从 4.98 m 降至约 2.8 m；
- 在 Slight OOD（轻微离散）场景中误差从 9.16 m 降至约 7.3–7.7 m；
- 在 Heavy OOD（极端城市）场景中误差从 28.76 m 降至 21.26 m（SPROV）或 18.96 m（SSL+FNT），其中 SSL+FNT 在最恶劣条件下表现更佳。

**⚠️ 局限性**

局限性：
- 仍依赖一定量的标注数据；自监督预训练虽能提升鲁棒性，但在极端多路径情况下仍受限于标注样本不足；
- 仅在单时段估计，未验证多时段或时间相关模型；
- 模型规模有限（约10万参数），在更大规模或多频多星座时可能受限；
- 对不同硬件/软件的GNSS观测适应性尚未完全验证。

---

## 355. Data Citation for Large Language Models: A Challenge

**arXiv ID:** 2608.25663 | [PDF](https://arxiv.org/pdf/2608.25663v1)

**作者:** Gianmaria Silvello `[一作]` `[通讯]` (University of Padua), Gianmaria Silvello (University of Padua)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了大型语言模型（LLM）在数据引用方面的挑战与研究方向，强调训练数据归因、推理时数据引用以及知识图谱事实引用的三大核心问题。

**💡 创新点**

将数据引用视为三维度（可验证性、来源追溯、信用分配）问题，提出针对训练数据、检索/工具调用以及KG事实的统一引用模型，并讨论如何在LLM中保留数据来源链接和实现信用传播。

**🔧 技术方法**

综述并借鉴了影响函数、数据Shapley值、RAG、KG grounding、nanopublications、PROV-O等已有技术，提出了结合数据库、信息检索、知识表示与AI的跨学科方法框架。

**📊 数据集**

本论文并未使用具体实验数据集，而是以公开训练语料、常见数据库和知识图谱（如Wikidata）为例进行理论阐述。

**📈 对比分析**

未开展实验或性能评估，主要提供理论分析和研究路线图；因此暂无比较方法或性能指标。

**⚠️ 局限性**

限制包括：缺乏可落地实现与评测基准；训练数据归因计算成本高且缺乏可解释性；数据引用标准不统一，缺少完整的元数据；KG事实的信用分配和来源追溯仍面临多源、异构、动态的挑战。

---

## 356. Think-Probe-Respond: Improving Large Language Models as Judges of Research Idea Novelty

**arXiv ID:** 2608.25660 | [PDF](https://arxiv.org/pdf/2608.25660v1)

**作者:** Tim Schopf `[一作]` (National Institute of Informatics), Akiko Aizawa `[通讯]` (National Institute of Informatics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出并验证了一种轻量级的思考-探测-响应（TPR）框架，用来提升大型语言模型在科研创意新颖性判断任务中的准确性。

**💡 创新点**

创新点在于通过从模型隐藏层提取潜在新颖性信号并将其作为条件注入最终输出，有效缓解了LLM在新颖性评估中偏向中等值的系统性误差。

**🔧 技术方法**

主要技术包括隐藏层表示的逻辑回归探测器、引导模型生成“思考”步骤以及在最终生成阶段对判断结果进行条件化。

**📊 数据集**

实验使用公开的 RINO 基准数据集，该数据集包含 1,381 条机器学习领域的研究创意及其人工标注的新颖性评分。

**📈 对比分析**

与零样本、少样本、提示式、Chain‑of‑Thought、Fine‑Tune 等多种基线相比，TPR 在宏观 F1 上平均提升 22.30%，并显著降低中等新颖性偏差，展示了更为均衡且精确的预测性能。

**⚠️ 局限性**

局限性包括：仅在 RINO（机器学习领域）上验证，可能对其他学科的通用性不足；需要访问模型隐藏层，导致闭源大模型无法直接使用；以及新颖性评估本身带有主观性，专家标注亦可能受限。

---

## 357. GaussianDream++: Efficient 3D Gaussian World Modeling for Robotic Manipulation

**arXiv ID:** 2608.25659 | [PDF](https://arxiv.org/pdf/2608.25659v1)

**作者:** Yuqing Jiang `[一作]` (University of Chinese Academy of Sciences), Haibao Yu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GaussianDream++，在 PaliGemma VLA 模型中嵌入 20 个世界状态与预测标记，形成紧凑的政策本地 3D 世界表示；

**💡 创新点**

创新点在于用角色分离的世界状态/预测 token 取代稠密 Gaussian 前缀，配合静态‑动态因子化与训练‑部署非对称设计，实现更高效、更具几何一致性的动作生成；

**🔧 技术方法**

采用 3D Gaussian splatting、可微渲染、度量深度、3D 运动、RGB 与 alpha 监督，以及流匹配动作专家；

**📊 数据集**

在 LIBERO、LIBERO‑Plus 视觉语言操纵基准以及真实机器人任务（Bowl‑Proximity、Eggplant‑to‑Pink‑Plate）上进行训练与评估；

**📈 对比分析**

与 GaussianDream、π_0.5 及多种 VLA 基线对比，LIBERO 上取得 98.6% 成功率、LIBERO‑Plus 87.8% Overall，并在实机实验中从 29.2% 提升至 52.5%，在 Camera 与 Layout 等几何敏感偏移下表现最突出；

**⚠️ 局限性**

局限性包括对 Robot 位姿偏移的鲁棒性仍不足、部署时仍有 44 ms 的延迟增幅，以及对更复杂多步交互场景的泛化尚待提升。

---

## 358. MAMA-FLUX.2: Image-to-Image Synthesis of Post-Contrast Breast DCE-MRI for the MAMA-SYNTH Challenge

**arXiv ID:** 2608.25648 | [PDF](https://arxiv.org/pdf/2608.25648v1)

**作者:** Kamil Kwarciak `[一作]` (AGH University of Krakow), Marek Wodzinski `[通讯]` (AGH University of Krakow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种基于预训练的流匹配Transformer（FLUX.2-Klein-4B）的条件潜在流匹配模型MAMA‑FLUX.2，用于将前对比剂乳腺MRI图像合成为后对比剂峰值增强图像。

**💡 创新点**

创新点在于将低秩适配器（LoRA）与三重区域损失（全局流匹配、肿瘤区域监督、稳定前景约束）结合，实现对肿瘤增强模式的精细建模，并通过面向平面路由的轻量级分类器提升不同扫描平面的适配效果。

**🔧 技术方法**

使用技术包括：预训练的潜在直流变换Transformer、VAE编码解码、LoRA低秩微调、条件潜在流匹配、区域化损失加权、以及面向平面路由的二分类器。

**📊 数据集**

使用数据集为MAMA‑SYNTH挑战数据，该数据集来自MAMA‑MIA乳腺DCE‑MRI，包含前后对比剂配对切片和肿瘤掩码。

**📈 对比分析**

与基线方法相比，MAMA‑FLUX.2在DSC（0.872）、HD95（16.84）、LPIPS（0.0745）以及肿瘤SSIM（0.861）等指标上表现优异，特别是在肿瘤区域的结构相似度与整体图像质量上取得最佳平衡。

**⚠️ 局限性**

局限性包括仅在10个开发案例上进行探索性消融实验、未保证体积一致性、以及经验性选择的阈值和权重，需在更大规模或不同采集协议的数据集上进一步验证。

---

## 359. A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation

**arXiv ID:** 2608.25643 | [PDF](https://arxiv.org/pdf/2608.25643v1)

**作者:** Bing Shao `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对逆 KL 损失的 K2 估计器在 on‑policy distillation 中的梯度进行解析，发现梯度幅度高度非均匀，主要集中在学生对样本预测概率低且与教师显著不一致的 token 上，并在此基础上提出了 Surprise‑aware Reweighting (SuRe) 的加权策略。

**💡 创新点**

创新点在于：① 推导出 K2 估计器梯度的闭式分解，将梯度范数拆为教师‑学生对数概率差与学生 softmax 几何因子；② 基于该分解提出了受惊程度驱动的有界重权重 SuRe，轻量级且不需要额外模型或前向传播。

**🔧 技术方法**

技术方法包括：on‑policy distillation、逆 KL 损失的 K2 采样估计、梯度范数解析、基于学生概率的有界重权重实现，以及对不同尺度 Qwen3 学生模型的训练与评估。

**📊 数据集**

使用了 Qwen3‑8B 作为教师，对 Qwen3‑1.7B‑Base 与 Qwen3‑4B‑Base 学生在 DeepMath 难度≥6 的 57K hard 子集上进行训练，评估数据集包括 AIME‑2024/2025、AMC‑23、MATH‑500，另外还检验了 CRUX、IFEval、MMLU‑Pro 等 OOD 任务。

**📈 对比分析**

与 Vanilla OPD、KD、SeqKD 等方法对比，SuRe 在 AIME‑24/25、AMC‑23 的 avg@k 与 pass@k 上提升最多约 7–8pp（如 1.7B 学生的 AMC‑23 pass@8 提升 7.5pp），但在 MATH‑500 与 OOD 任务上提升有限或无显著差异。

**⚠️ 局限性**

局限性包括：仅研究采样 token 级别的逆 KL OPD，未系统探讨全词表或其他散度目标；实验集中在数学推理数据集且推理轨迹较短，可能不适用于更广泛或更长的推理任务；模型规模受限，未验证在更大参数模型上的表现。

---

## 360. Leveraging Inter-object Affordances for Efficient Planning in Contact-rich Tasks

**arXiv ID:** 2608.25641 | [PDF](https://arxiv.org/pdf/2608.25641v1)

**作者:** Pouya P. Niaz `[一作]` (University of Innsbruck), Alejandro Agostini `[通讯]` (University of Innsbruck)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 U-TAMP 任务规划框架中加入基于形状、尺寸、材料的对象间交互属性（支撑、可支撑、可抓取、可提升、可滑动）作为符号化约束，设计并实现了 UTAMP-2 规划域，并在模拟厨房桌面场景中对比原始 U-TAMP 与 VLM‑based 规划。

**💡 创新点**

创新点在于：①将对象物理属性抽象为可符号化的交互约束；②在符号层面直接编码抓取、放置、提升、滑动等多种交互，减少子符号化推理；③利用 VLM 进行属性检测并自动生成 PDDL 问题。

**🔧 技术方法**

采用的技术包括 PDDL 规划（Fast Downward）、RRT‑Connect 动作规划、OpenAI ChatGPT 4.0 作为 VLM 进行属性检测与规划推理、CoppeliaSim 物理仿真以及自定义的符号化约束脚本。

**📊 数据集**

使用的是自建的厨房桌面仿真环境，包含盐壶、玻璃杯、砧板、煎锅、木托盘共 5 种物体，随机生成 127 个有效堆叠配置；未使用公开数据集。

**📈 对比分析**

比较方法：在 2–5 件可操作物体的 127 种配置下，对 4 种规划方案（UTAMP‑1、UTAMP‑2‑GT、UTAMP‑2‑VLM、VLM）分别记录成功率与规划时间。结果显示 UTAMP‑2‑GT 100% 成功率、UTAMP‑2‑VLM 90%、UTAMP‑1 与 VLM 分别 49% 与 39%；规划时间方面，UTAMP‑2 系列比 UTAMP‑1 快约两位数，VLM 约慢两位数。

**⚠️ 局限性**

限制：①假设规则过于简化（如小物体不能支撑大物体、圆面不支撑、重量阈值等），无法覆盖所有真实物理细节；②VLM 的属性检测偶尔出现错误导致计划失败；③仅处理基于盒子/多面体的物体，未考虑更复杂形状或动态属性；④缺乏对更复杂动作（如旋转、分解）和在线学习的支持。

---

## 361. RetrievalRouter: Joint Modality and Architecture Selection for Document Retrieval

**arXiv ID:** 2608.25625 | [PDF](https://arxiv.org/pdf/2608.25625v1)

**作者:** Emre Kuru `[一作]` (Institut Polytechnique de Paris), Noel Crespi `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级的查询路由器 RetrievalRouter，能够根据仅凭查询文本预测最佳检索管道（模态与架构），实现更高效、更准确的文档检索。

**💡 创新点**

创新点包括：①将模态与架构选择统一为单一路由动作空间，避免多级路由的误差传播；②使用软目标（基于每条查询的奖励向量）训练路由器，显著提升决策质量；③引入单一可调参数 λ，便于在准确率与延迟之间作精细权衡；④在11个跨领域检索基准上，取得比最强静态管道更高的 nDCG@5 并且延迟降低 12.4 倍。

**🔧 技术方法**

主要技术：Qwen3-0.6B-Base 编码器 + LoRA、均值池化得到查询向量；线性决策层输出五种管道的 logits；奖励函数结合 nDCG 与归一化延迟；温度 0.1 的 softmax 生成软目标；KL 散度训练；对齐训练集的 80/10/10 内部拆分；并在 NVIDIA H100 上进行统一延迟测量。

**📊 数据集**

使用了 11 个检索基准集合，涵盖金融、科研和开放域文档：REAL-MM-RAG（FinReport、FinSlides）、T2-RAGBench（FinQA、ConvFinQA、VQAonBD、TAT-DQA）、MMDocRAG（ArxivQA、Wiki-SS、MP-DocVQA、SciQAG、DUDE）。

**📈 对比分析**

与七种静态管道（BM25、TD、TL、TR、MD、ML、MR）以及此前的策略选择基线（Arabzadeh 等 2021）进行对比。RetrievalRouter 在 λ=0.1 时获得 0.755 nDCG@5（比最强静态管道高 2.5%），平均延迟 0.666s（比最强静态管道快 12.4 倍）；在 λ=0.5 时 nDCG@5=0.707、延迟 0.314s；在 λ=1 时退化为 BM25。相较于先前自适应方法，在准确率优先场景下 nDCG@5 明显更高，在延迟优先场景下性能相当或略优。

**⚠️ 局限性**

局限性包括：①多模型路由需要维护四个向量索引和一个 BM25 索引，存储占用显著（尤其是多模态延迟索引 ~39 GB）；②GPU VRAM 需求高（≈40 GB），对硬件成本敏感；③在跨域零样本情境下泛化能力未评估，可能依赖域特定词汇；④仅使用查询文本进行路由，面对语义歧义（如“在第5页的表格摘要”）时无法区分目标文档的结构，从而无法做出最优选择。

---

## 362. On the Separation of Human and AI-Generated Images in CLIP Embedding Space

**arXiv ID:** 2608.25609 | [PDF](https://arxiv.org/pdf/2608.25609v1)

**作者:** Andrea Asperti `[一作]` `[通讯]` (University of Bologna), Andrea Asperti (University of Bologna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了人类与AI生成绘画在CLIP嵌入空间中自发分离的现象，并尝试解释其根源。

**💡 创新点**

首次将可解释的多尺度散射变换与梯度反演相结合，揭示分离主要由分布式多尺度图像结构驱动，而非低级统计或局部缺陷。

**🔧 技术方法**

使用CLIP预训练模型提取特征、PCA降维、HOG与多尺度散射特征、线性回归、MLP、Transformer以及梯度反演等技术。

**📊 数据集**

利用AI-WikiArt、AIPastiche、National Gallery of Art（NGAD）等人工与AI绘画集合进行实验。

**📈 对比分析**

通过比较不同描述子对PC1/PC2的解释力（R²）和梯度反演产生的位移量，散射特征R²≈0.6，反演位移约为CLIP直接反演的四分之一，说明仅解释了部分分离。

**⚠️ 局限性**

仍未完全解释分离机制，散射特征无法捕获CLIP中可能的注意力相关长程信息，且仅在艺术图像上验证，未检验其他领域或模型的普适性。

---

## 363. MLLMCLIP: Feature-Level Distillation of MLLM for Robust Vision-Language Representations

**arXiv ID:** 2608.25575 | [PDF](https://arxiv.org/pdf/2608.25575v1)

**作者:** Jongsuk Kim `[一作]` (KAIST), Yuki Mitsufuji `[通讯]` (Sony Group Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将多模态大语言模型（MLLM）知识通过特征级蒸馏迁移至 CLIP 学生模型的框架 MLLMCLIP。

**💡 创新点**

创新点在于：① 跨架构蒸馏，解决生成式教师与判别式学生的结构不匹配；② 基于注意力的教师 token 选择，动态选取最具信息量的 token；③ 使用 CKA 损失实现结构对齐，提升知识迁移质量；④ 完全消除合成负样本生成的开销。

**🔧 技术方法**

主要技术包括：注意力权重挑选教师 token、辅助层映射维度、CKA 损失进行结构对齐、InfoNCE 对比学习与传统蒸馏损失（MSE 等）的对比。

**📊 数据集**

预训练数据采用 CC3M；评测涵盖 11 个组合推理基准、13 个零样本分类数据集和 2 个图文检索基准。

**📈 对比分析**

与四种主流 CLIP 增强方法（LaCLIP、NegCLIP、FSC-CLIP、TripletCLIP）进行对比，MLLMCLIP 在组合推理上显著提升（平均提升约 3–5%），在零样本分类和检索任务上也均超过所有基线，且计算成本仅为数据级蒸馏的一半左右。

**⚠️ 局限性**

限制：① 蒸馏效果受教师 MLLM 能力限制，教师的偏差或推理错误会被转移；② 最佳教师可能因下游任务不同而变化；③ 实验多用单个随机种子，未进行多种种子验证。

---

## 364. AI Slop and Hallucinations in Vulnerability Assessment: A Survey on Reasoning Failures and Trustworthy Mitigation

**arXiv ID:** 2608.25667 | [PDF](https://arxiv.org/pdf/2608.25667v1)

**作者:** Junchen Ding `[一作]` (University of New South Wales), Yuekang Li `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了大语言模型在漏洞评估中产生的“AI slop”问题，并构建了完整的分类体系；提出了Deductive Coverage Score（DCS）来度量漏洞报告的逻辑完整性；设计了神经符号验证架构与两个新评测工具CVE-Bench与Slop-Score；

**💡 创新点**

创新点在于：①将“AI slop”从多样化失效模式统一归纳为三大分支并提供统一指标；②引入DCS作为可量化的推理覆盖度量；③构建以执行验证为核心的多层神经符号管道；④推出针对性评测基准，填补现有评测中缺失的安全性与可验证性评估；

**🔧 技术方法**

技术包括：LLM生成、Chain‑of‑Thought提示、工具调用（如CodeQL、符号执行、模糊测试）、检索增强生成（RAG）、自我反思（Reflexion）以及多层次验证与迭代纠错；

**📊 数据集**

数据集主要使用公开CVE数据库、Bug Bounty平台提交记录、开源代码仓库、人工标注的AI slop实例；此外构建了CVE‑Bench（正负样本）和Slop‑Score评价集；

**📈 对比分析**

与传统单一LLM、CoT、RAG等方法对比，验证管道在第一层即过滤掉约70‑80% slop，整体报告准确率提升约25‑35%；在CVE‑Bench上F1提升0.15以上，Slop‑Score在区分高流利度但无效报告方面表现突出；

**⚠️ 局限性**

局限性包括：①工具覆盖范围有限（符号执行路径爆炸、模糊测试对特定条件敏感）；②对抗性生成仍能绕过某些验证步骤；③DCS与其他指标依赖人工或确定性工具的准确性；④构建与维护评测基准需要高成本的专业标注。

---

## 365. AffectSim: A Controllable Interactive 3D Simulation Benchmark for Embodied Affective Perception

**arXiv ID:** 2608.25664 | [PDF](https://arxiv.org/pdf/2608.25664v1)

**作者:** Ke Xing `[一作]` (Shenzhen MSU-BIT University), Xiping Hu `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

请提供完整的论文内容或摘要，以便进行准确总结。

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 366. Stochastic End-to-End Latency Modeling of the IoT-Edge-Cloud Continuum: Impact of Jitter and Traffic Variability on Deterministic Service Provisioning

**arXiv ID:** 2608.25658 | [PDF](https://arxiv.org/pdf/2608.25658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 367. Reconstructing the Right Episode: Evaluating Interleaved Conversational Memory Beyond Long Context

**arXiv ID:** 2608.25655 | [PDF](https://arxiv.org/pdf/2608.25655v1)

**作者:** Zhexi Feng `[一作]` (University of California San Diego), Pengtao Xie `[通讯]` (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了SCALE-QA基准，专门评估在长篇混合主题对话中恢复并利用沉睡约束的能力。

**💡 创新点**

创新点在于将“episode integrity failure”视为单独评测维度，并设计了TSIM多视角的事件重建内存框架。

**🔧 技术方法**

技术包括语义漂移段划、三视角检索（原始、摘要、聚类）以及基于多视角得分的事件排序。

**📊 数据集**

数据集为3,000道多领域任务导向问答，包含完全可追溯的证据、四选一评判与可配置长度的运行时包装。

**📈 对比分析**

在128k上下文下，TSIM在Gemini 2.5 Flash、GPT‑4o‑mini和本地Gemma模型上分别提高5.6–17.6个百分点，准确率最高可达80.2%，而标准RAG仅达约25%。

**⚠️ 局限性**

局限性包括合成的反事实构造不完全代表真实日志，且仅使用四选一问答限制了对开放式回答与工具使用的评估。

---

## 368. Learning New Facts with QLoRA: An Acquisition-Retention Frontier

**arXiv ID:** 2608.25677 | [PDF](https://arxiv.org/pdf/2608.25677v1)

**作者:** Estelle Zheng `[一作]` (LORIA), Christophe Cerisara `[通讯]` (LORIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在参数高效微调（QLoRA）中，适配器秩对模型获取新事实知识与保持旧有能力的影响，使用匿名化的 OpenStreetMap（OSM）问答基准来评估事实获取、同义句泛化与 OOD 保留；

**💡 创新点**

首次引入匿名化 OSM 基准，揭示适配器秩能控制获取-保留的前沿，并将其与模型漂移诊断（KL、SVD 等）关联，说明高秩会更好地安装新事实但导致更多遗忘；

**🔧 技术方法**

使用 QLoRA 低秩量化微调、全微调 FFT、KL 散度、SVD 语谱诊断、LM Evaluation Harness OOD 评测以及标准 LoRA 对比；

**📊 数据集**

匿名化的 OpenStreetMap 训练集（1,938 条问答）以及五个 OOD 基准（HumanEval、IFEval、TruthfulQA、MMLU‑Redux、BBH）；

**📈 对比分析**

在相同任务下对不同秩的 QLoRA 与 FFT 进行对比，测量训练集 EM、同义句泛化 EM 与 OOD 平均得分。结果显示：低秩 QLoRA 维持 OOD 高但事实获取低；高秩 QLoRA 获取高但 OOD 明显下降；FFT 在两者之间，取得较高 OOD 但未达到最高事实获取；模型漂移随秩增大；

**⚠️ 局限性**

限制包括：OSM 数据规模小、仅覆盖 14 个小城市；匿名化可能不完全模拟真实知识注入；仅评估事实关联而非抽象关系；实验仅在 Qwen3‑4B（及 Qwen3‑1.7B LoRA 对照）上进行，未检验更大模型或不同预训练数据；OOD 基准不涵盖所有能力维度；实验中数学适配的比较范围有限。

---

## 369. From General Agents to RCA Experts: A Self-Evolving Harness for Root Cause Analysis

**arXiv ID:** 2608.25661 | [PDF](https://arxiv.org/pdf/2608.25661v1)

**作者:** Haiyu Huang `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了OpsHarness，一种自演化的外部把手，用来提升大型语言模型在根因分析（RCA）任务上的准确性和效率。

**💡 创新点**

创新点包括：①将关注点放在通用代理的外部托管层而非从零构建专用RCA代理；②实现了自演化机制，能从成功和失败的诊断轨迹中提炼可复用的工作流、操作和规则；③引入双门验证（内部门和外部门）防止过拟合；④使用分层知识库与idea-card工具库，使知识按粒度逐步展开。

**🔧 技术方法**

技术栈：大型语言模型通用代理（Codex、Claude Code等）+ OpsHarness控制层（setup/diagnose/evolve/verify）+ 轨迹挖掘、提案合成、双门验证机制 + 分层操作与规则知识 + 工具卡库。

**📊 数据集**

使用公开基准数据集 OpenRCA、RCAEval 以及公司A生产的变更异常数据集（约 88 条标签案例）进行评估。

**📈 对比分析**

通过与四大模型（GPT‑5.5、Claude Sonnet 4.6、GLM‑5.2、DeepSeek‑V4）结合的六种框架（Direct、ICL、专用RCA-Agent、mABC）进行对比。结果显示 OpsHarness 在所有基准上平均 Top‑1 准确率提升至 59%，比裸代理提升 63.4%，比专用代理提升 4.02×；在工业部署上相较于直接代理提升约 3 倍。

**⚠️ 局限性**

局限性：仍需依赖系统特定的用户反馈与标签；对完全未见的故障模式仍无法即时给出最佳实践；在极端噪声或高度非结构化日志下的诊断准确性有限；自演化过程虽然成本低，但仍需手工维护与监督，且双门验证在某些边缘案例可能过于严格。

---

## 370. Diffusion Transformers for Roof Graph Synthesis and Reconstruction

**arXiv ID:** 2608.25652 | [PDF](https://arxiv.org/pdf/2608.25652v1)

**作者:** Daniel Panangian `[一作]` (German Aerospace Center), Ksenia Bittner `[通讯]` (German Aerospace Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于扩散Transformer的二维屋顶图结构生成与重建框架RoofDiT，支持无条件生成、建筑足迹条件生成和航空图像引导的重建；

**💡 创新点**

创新点在于将相对几何感知注意力、足迹与图像多模态条件以及屋顶对齐正则化融入扩散Transformer，形成可学习的屋顶图结构先验；

**🔧 技术方法**

使用扩散Transformer、相对几何注意力、条件跨注意力、屋顶对齐正则化以及预训练的DINOv2图像特征；

**📊 数据集**

使用住宅屋顶数据集，包含1926训练、249验证、223测试样本，已标注屋顶顶点边缘以及对应航空图像；

**📈 对比分析**

与GSDiff、Straight Skeleton、HEAT和RoofMapNet等基线比较，在无条件生成上FID、KID下降，图结构有效率提升；在足迹条件生成上节点误差、边缘F1和面数误差显著下降；在图像引导重建中，加入足迹条件后节点、边缘、面F1均大幅提升，性能接近最强基线HEAT；

**⚠️ 局限性**

局限在于仅处理平面屋顶图结构，无法直接生成完整三维几何；可学习的有效性约束不如几何构造严格；采样与一致性受限，需改进采样策略与训练规模。

---

## 371. EgoNav: Bridging Learned Waypoints and Geometry-Aware Local Control for Robust Indoor Navigation

**arXiv ID:** 2608.25642 | [PDF](https://arxiv.org/pdf/2608.25642v1)

**作者:** Jing Wang `[一作]` (City University of Hong Kong), Peng Yin `[通讯]` (City University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了EgoNav系统，将预训练的路径点预测与几何感知的局部规划相结合，实现基于拓扑图的图像目标导航。

**💡 创新点**

创新点：①引入基于语义分割骨架的几何感知路径点细化与多指标评分，纠正学习预测的几何与方向误差；②根据细化结果动态调节Falco局部规划器参数，实现对环境几何的自适应控制；③结合定位过滤与回退重定位机制提升鲁棒性。

**🔧 技术方法**

使用技术包括：图像目标导航模型GNM/ViNT、视觉-位置匹配VPR（MixVPR）、RGB-D语义分割DFormerV2、拓扑图Dijkstra路径规划、骨架提取与候选采样、能量评分机制、Falco局部规划器及其参数自适应调节。

**📊 数据集**

数据集与平台：在Habitat-sim环境下使用12个Matterport3D场景进行模拟实验；在真实环境中使用7个办公楼场景与一个人形机器人（搭载Orbbec Gemini335L）以及LoCoBot进行实地测试。

**📈 对比分析**

对比方法包括GNM、ViNT、NoMaD、PlaceNav、DeepExplore、VLFM以及PlaceNav+Falco。EgoNav在模拟与真实实验中均取得显著优势：短距离SR>90%、中距离SR≈70%、长距离SR≈60%；SPL提升约15–20%，在所有距离等级上均优于基线。

**⚠️ 局限性**

局限性：依赖预训练预测器的泛化能力，若环境与训练分布差异大则性能下降；局部规划仍为反应式，难以有效避开高速动态障碍；定位误差累积导致远距离成功率下降；缺乏全景视角匹配，导致视角变化下匹配鲁棒性不足。

---

## 372. From Specialization to Generalization: Instruction-tuned LLMs for Robust Harmful Content Mitigation

**arXiv ID:** 2608.25605 | [PDF](https://arxiv.org/pdf/2608.25605v1)

**作者:** Lukas Edman `[一作]` (TU Munich), Alexander Fraser `[通讯]` (TU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对36个英语仇恨言论数据集进行统一转换成指令格式，并对Qwen3 4B进行指令微调，构建通用仇恨言论检测与缓解模型HIPPO。

**💡 创新点**

首次系统性地将多标签、层次化、生成任务等多种任务联合训练，证明指令微调的通用LLM可在跨域、跨语言和生成任务上超越专门化的BERT系列模型。

**🔧 技术方法**

使用QLoRA对Qwen3 4B进行高效指令微调，结合对话式prompt设计，并对不同规模模型（0.6B、4B、32B）进行实验。

**📊 数据集**

整合了36个英语仇恨言论数据集（共约650k样本），包括二分类、多分类、多标签、生成和层次化任务。

**📈 对比分析**

与单任务训练、GPT‑5‑mini以及公开的BERT/RoBERTa/Flan‑UL2等SOTA进行宏F1对比，4B版HIPPO在17项评测中有14项提升，平均F1提升至70.3%（SOTA 68.5%），32B版更优。

**⚠️ 局限性**

仅使用英语数据，未覆盖多语言训练；对提示方式未进行系统优化；模型在极少样本或低质量指令下表现不佳，跨语言泛化仍受限。

---

## 373. Defending the Peg: Real-Time Dynamic Protection and Anomaly Detection in DeFi Stablecoins

**arXiv ID:** 2608.25600 | [PDF](https://arxiv.org/pdf/2608.25600v1)

**作者:** Hengxing Zeng `[一作]` (Hainan University), Xiaoqi Li `[通讯]` (Hainan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了面向稳定币智能合约的生命周期动态安全框架，结合预防性合约防护和基于 Bi‑LSTM 的实时异常检测，实现对重入、预言机操纵、闪电贷治理等攻击的高效防御。

**💡 创新点**

创新点在于将多层安全机制（静态审计、运行时锁、预言机聚合、治理延迟）与实时节点级异常检测相结合，并通过成本敏感 Bi‑LSTM 模型实现毫秒级检测。

**🔧 技术方法**

采用了静态代码分析、形式化验证、多源预言机聚合、时间锁、Bi‑LSTM 深度学习、成本敏感交叉熵等技术。

**📊 数据集**

使用了基于 100,000 条模拟交易的数据集（95% 正常、5% 攻击），以及在本地 Ethereum‑兼容测试环境中重现的三类攻击实例。

**📈 对比分析**

与传统静态审计、fuzzing、形式化验证等方法对比，模型在准确率 96.61%、召回率 97.70%、单次推理时延 1.5–2.8 ms 的同时，能够在实时内拦截攻击。

**⚠️ 局限性**

局限在于数据集为模拟，精度与真实环境下高频 DeFi 交易混淆导致较低精确率，且模型需持续更新以适应网络变化。

---

## 374. M-Fibration Theory with Applications to Neural Network Compression

**arXiv ID:** 2608.25598 | [PDF](https://arxiv.org/pdf/2608.25598v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 375. Plans You Can Check: Verifier-Grounded Learning of an Open-Weight Planner for Executable Video-Editing

**arXiv ID:** 2608.25622 | [PDF](https://arxiv.org/pdf/2608.25622v1)

**作者:** Haoyu Wang `[一作]` (Chinese University of Hong Kong), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了一种可执行的视频编辑规划系统 RefineCut，能够在任意渲染后端前通过规划器生成剪辑选择、裁剪、排序、转场、持续时间和音乐对齐等编辑计划。

**💡 创新点**

创新点在于：①把视频编辑视为可验证的规划问题，使用显式约束账本和确定性验证器；②利用验证器回放去噪教师轨迹，生成可靠的监督信号；③在此基础上通过Verifier‑centered Self‑Improvement（DPO + rubric）进一步提升规划质量。

**🔧 技术方法**

技术手段包括：RFC 6902 样式的 JSON Patch（RefinePatch）、确定性验证器、显式约束账本、验证器回放式轨迹蒸馏、Direct Preference Optimization (DPO)、Rubric‑Guided Self‑Improvement、8B 开放权重 LLM（Qwen3‑8B、Llama‑3.1‑8B、GLM‑4‑9B）以及多教师 API（GPT‑5.4、Qwen3‑Max、DeepSeek‑V4‑Pro）等。

**📊 数据集**

使用的数据集为 RefineCut‑Bench，包含 3,578 个任务、7,971 条带注释的剪辑、499 首音乐、约 23,913 帧的注释、完整的约束账本以及多教师轨迹。

**📈 对比分析**

通过 Video‑Editing Score (VES) 在 Common‑100 基准上评估。提示模型 VES 0.594，Raw 0.620，Verifier‑replayed Distillation 0.858，RefineCut‑Evo 0.924，超越三位前沿教师（GPT‑5.4 0.893、DeepSeek‑V4‑Pro 0.936、Qwen3‑Max 0.773）。在人类预览评估中，RefineCut‑Evo 在 150 对比中赢得 78% 的偏好，验证器与人类评估结果高度一致。

**⚠️ 局限性**

局限性包括：①仅关注可执行的规划，无法评估剪辑的创意或故事质量；②对上游字幕、节拍检测的误差敏感；③验证器只能检查结构和约束，无法判断故事性或审美；④数据集为模板生成，缺乏真实用户规范；⑤模型规模受限，未实现在线 RL 或更大规模模型。

---

## 376. An Analysis of the Impact of Psychological Factors and Techniques Across Different Types of Social Engineering

**arXiv ID:** 2608.25670 | [PDF](https://arxiv.org/pdf/2608.25670v1)

**作者:** Helin Omer `[一作]` (Ludwig-Maximilians-Universität München), Daniela Pöhn `[通讯]` (University of Bundeswehr Munich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在实验室中使用12名受试者，对5种心理因素与5类社交工程攻击类型共25个刺激进行交叉设计，评估受试者对不同攻击和心理因素组合的易受攻击性。

**💡 创新点**

首次系统比较多种心理因素与技术在不同社交工程攻击中的有效性，发现最有效组合（如 spear‑phishing + 贪婪）并揭示了各攻击类型的差异。

**🔧 技术方法**

采用5×5内部设计、思考大声法、Likert量表测量和定性内容分析相结合的混合方法。

**📊 数据集**

使用自制的仿真截图与音频刺激（钓鱼、鱼叉钓鱼、电话钓鱼、短信钓鱼和弹窗），无外部公开数据集。

**📈 对比分析**

通过比较各PF‑SE组合的“是否点击”成功率进行评估，最成功组合为 spear‑phishing+贪婪（75%），整体最高成功率为 spear‑phishing（53%）和钓鱼（48%）。

**⚠️ 局限性**

样本量小、实验室环境限制生态有效性、每种心理因素仅采用单一场景、缺乏更广泛的心理因素与技术组合。

---

## 377. PRISM: Projection-Integrated Sampling-Based MPC with Bayesian Cost Tuning for Bimanual Manipulation

**arXiv ID:** 2608.25666 | [PDF](https://arxiv.org/pdf/2608.25666v1)

**作者:** Alinjar Dan `[一作]` (University of Tartu), Arun Kumar Singh `[通讯]` (University of Tartu)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种名为PRISM的实时双臂机器人控制框架，通过GPU加速的物理仿真做为在线世界模型，在每个MPC步骤中采样、投影并评估大量候选轨迹，最终选择可行的平滑轨迹执行。

**💡 创新点**

核心创新包括：① QP投影的控制采样策略，将探索与轨迹平滑解耦；② 为该投影量身定制的ADMM/Bregman求解器，利用关节分离和预先矩阵分解实现高效GPU并行；③ 采用贝叶斯优化在离线阶段自动调节多目标成本权重，降低手工调参需求。

**🔧 技术方法**

技术手段：采样式MPC（MPPI/CEM）与重要性采样；GPU并行MuJoCo/MJX物理 rollouts；QP投影约束满足位置、速度、加速度、减速边界；ADMM求解器；贝叶斯优化（Gaussian Process + Expected Improvement）用于权重搜索。

**📊 数据集**

使用的任务数据集：PerAct^2 任务的四种变体（托盘移动、球搬运、立方体交接、盒子搬运），以及在双UR5e + Robotiq抓手的真实实验。实验中通过随机初始化和障碍物布置来增加任务多样性。

**📈 对比分析**

与Baseline-CEM、LPF（低通滤波）和SGF（Savitzky–Golay）等基线对比，PRISM 在所有任务中实现更高的成功率（最高接近100%），计算时间与基线相近（每步10–16 Hz），并且在Bayesian优化后成功率进一步提升。实验表明PRISM 在复杂、接触丰富的双臂任务中保持了鲁棒性与实时性。

**⚠️ 局限性**

局限性：① 成本函数仍需人工设计与分阶段；② 依赖MuJoCo高保真物理模型，无法轻易扩展到柔性物体、流体或极动态碰撞；③ 目前未与学习式价值函数或更长的规划视野结合，可能限制在更大空间中高效收敛。

---

## 378. Unmatched Does Not Mean False: Incomplete Reference Sets Can Reverse Calibration Rankings in Open-Ended Theory-of-Mind Tracking

**arXiv ID:** 2608.25654 | [PDF](https://arxiv.org/pdf/2608.25654v1)

**作者:** Zhexi Feng `[一作]` (University of California San Diego), Bingrui Zhang `[通讯]` (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统分析了开放式 Theory‑of‑Mind 跟踪器在使用有限参考集评估时出现的标签来源效应，导致严格正确的 Brier 排序被逆转，并提出了基于概率抽样人类审计的三源修复协议。

**💡 创新点**

创新点在于首次将标签来源效应拆解为前景率崩塌导致的风险逆转，并给出了闭式判定准则和可执行的修复流程，解决了开放式输出评估中的普遍失效。

**🔧 技术方法**

研究使用了 Brier 风险、ECE、AUROC 等严格合规的评分指标，结合 Platt 对数线性校准、抽样权重估计、蒙特卡洛重采样与对数映射校准等技术实现排名恢复。

**📊 数据集**

实验数据主要来自 259 条手工审计的信念样本（涵盖六种场景）、301 条 NQ‑open 检索‑阅读模型的预测以及 240 条 OpenToM 自由文本推理的信念。

**📈 对比分析**

与传统只匹配评估方法相比，本文方法在 Brier 和 ICE 指标上出现显著排名逆转；通过 50 次人类标注的试点，恢复正确顺序的概率超过 99.6%，覆盖率超过 95%，区间宽度缩短约 30%–40%。

**⚠️ 局限性**

局限性包括仅在特定 ToM 跟踪器与匹配器上验证，需进一步跨模态、跨语言和更大规模数据的测试；修复方案仍依赖有限人工标注，若人类成本不可接受则难以推广。

---

## 379. Towards Purified Multi-Label Test-Time Adaptation of Vision-Language Models

**arXiv ID:** 2608.25653 | [PDF](https://arxiv.org/pdf/2608.25653v1)

**作者:** Yiwen Liang `[一作]` (Tsinghua University), Guiguang Ding `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PuRF 方法，通过区域净化和缓存净化实现多标签测试时适应，以提高 VLM 在分布漂移下的多标签识别精度。

**💡 创新点**

创新点在于将多粒度一致性用于区域净化以获取可靠的区域证据，并结合情节净化和时序刷新实现缓存的去噪与长期适应，从而解决全局特征耦合导致的标签偏置问题。

**🔧 技术方法**

采用 CLIP 基础的缓存 TTA 机制，结合区域选择、伪标签生成、二元交叉熵、熵最小化、跨模态对齐、时间衰减等技术实现鲁棒的多标签适应。

**📊 数据集**

在 VOC 2007/2012、COCO 2014/2017、NUS‑WIDE 五个标准多标签数据集上进行评估。

**📈 对比分析**

与多种基线（CLIP、TPT、ML‑TTA、ReTA 等）进行对比，PuRF 在 ViT‑B/32 上平均提升约 4% mAP，并在所有数据集上均取得 SOTA 结果。

**⚠️ 局限性**

局限性包括对伪标签和区域选择的依赖，极端分布漂移或噪声样本下表现可能受限；缓存容量受限导致稀疏标签难以有效更新；时序刷新参数需经验调节。

---

## 380. LDAC-Net: A Learnable Multi-Lag Differencing Attention-Convolution Network for Drift-Robust Recognition with Low-Cost MOX Gas Sensors

**arXiv ID:** 2608.25646 | [PDF](https://arxiv.org/pdf/2608.25646v1)

**作者:** Xin Zhang `[一作]` (Manchester Metropolitan University), Tam Sobeih `[通讯]` (Manchester Metropolitan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种端到端的多时延差分注意卷积网络 LDAC‑Net，用于低成本 MOX 气体传感器的物质识别，无需外部预处理。

**💡 创新点**

创新点在于将漂移补偿与时序差分集成到可学习前端（WSAN+LMLD），通过多时延差分学习不同时间尺度的动态，并使用混合注意力‑卷积骨干与自适应窗口长度提升性能。

**🔧 技术方法**

采用可学习窗口条件仿射归一化、可学习多时延差分、注意力‑卷积混合模块、单查询注意力池化、随机深度、以及 TimeCutout、Channel Dropout、mixup 等数据增强技术。

**📊 数据集**

使用 SmellNet‑Base（50 类、6 通道）、SmellNet‑Mixtures（12 成分混合）以及 eNose‑Drift（62 通道、长时间漂移）三大公开数据集进行实验。

**📈 对比分析**

与传统 MLP、CNN、LSTM、Transformer（含 FOTD 预处理）以及 Non‑stationary Transformer、Autoformer、TCN、Neural‑ODE 等方法对比；在 SmellNet‑Base 原始输入上 Top‑1 达 68.2%，比 FOTD 版提升约 15%；在 Mixtures 上 Top‑1@0.1 提升至 50.5%；在 eNose‑Drift 原始输入上 Acc 70.6%、macro‑F1 69.6%，均超越所有基线 8–11 个百分点。

**⚠️ 局限性**

限制包括：训练数据规模有限，模型对窗口长度敏感且窗口长度为全局超参；仅在受控实验室环境中验证，未测试真实场景；未评估模型在不同硬件上的延迟与功耗。

---

## 381. Dissonance Spectrum explicitly models perceptual frequency interactions for better music understanding

**arXiv ID:** 2608.25621 | [PDF](https://arxiv.org/pdf/2608.25621v1)

**作者:** Tianle Wang `[一作]` (Beijing Institute for General Artificial Intelligence), Song-Chun Zhu `[通讯]` (Beijing Institute for General Artificial Intelligence)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Dissonance Spectrum（DS）——一种将音频的谐波关系用非负时频图表征，并可直接映射到每个频率 bin 的新表示；

**💡 创新点**

创新点在于：①设计了基于容差的理性音高比例核，并用一维频率相关实现 O(KT) 计算；②将DS作为轻量并行分支通过零初始化残差适配器无缝注入现有音乐理解模型，既保留原始输出又不改变模型结构；

**🔧 技术方法**

核心技术包括：连续CQT、比例距离/合理近似核、频率轴相关、双向残差适配器、卷积/注意力编码以及对齐的低秩压缩；

**📊 数据集**

使用的数据集主要有：MU‑LLaMA的70,011问答对及5,040评测对；Music2Emo的DEAM、EmoMusic、PMEmo（MTG‑Jamendo）情绪标注数据；以及控制实验用的合成钢琴音频；

**📈 对比分析**

对比方法：与基线、参数匹配的高斯分支、架构匹配的CQT分支进行六个随机种子实验。DS在 MusicQA（BLEU、METEOR、ROUGE‑L、BERTScore‑R、loss、perplexity）和 Music2Emo（宏观标签指标、情绪 R²）上均获得最高均值，BERTScore‑R 提升约 0.7%–1.2%，情绪 R² 提升 0.015–0.02；

**⚠️ 局限性**

局限性：仅实现一种基于谐波距离的关系，未涵盖听觉滤波、掩蔽、文化/个体偏好等因素；实验仅覆盖两类预训练模型；未进行听感实验验证；DS 对更广泛任务的泛化仍待评估。

---

## 382. Advantage-Driven Explicit Memory for Social Navigation

**arXiv ID:** 2608.25610 | [PDF](https://arxiv.org/pdf/2608.25610v1)

**作者:** Yeonsoo Park `[一作]` (Seoul National University), Christian Wolf `[通讯]` (Naver Labs Europe)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种非参数经验记忆机制，结合PPO‑GRU训练，提升社交导航中的稀有事件记忆与在线适应能力。

**💡 创新点**

通过将GRU隐藏状态作为检索键，k步前景动作与归一化Q值构成Scorecard，实现稀有事件的因果索引和即时记忆更新，缓解参数学习稀疏信号消失与分布偏移问题。

**🔧 技术方法**

使用PPO强化学习、GRU循环网络、注意力检索机制、正交正则化、k步前瞻记忆构造以及在线基于TD或物理启发式的记忆标签。

**📊 数据集**

在Habitat‑Sim的HM3D数据集上进行实验，配合Recast Navigation的人群模拟实现PointGoal导航。

**📈 对比分析**

与基线PPO‑GRU、仅人类编码器、离线记忆以及在线记忆等配置对比，在ID情境下成功率提升3‑4%，在OOD场景中在线记忆将成功率从54%提升至60%，人类碰撞显著下降，查询延迟约0.1‑0.13 ms。

**⚠️ 局限性**

记忆缓冲区采用FIFO替换可能丢失长期重要经验，查询空间受仿真训练的隐藏状态影响，存在sim‑to‑real迁移问题，且对高密度实时查询的扩展性仍需验证。

---

## 383. When Should a Network Emit Geometry, and When Should It Detect It? Readout, Reconciliation, and Representation in Floorplan Vectorization

**arXiv ID:** 2608.25608 | [PDF](https://arxiv.org/pdf/2608.25608v1)

**作者:** He Zhang `[一作]` `[通讯]`, He Zhang

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对平面图向量化任务中的网络，比较了同一网络的两种读取方式（自回归序列解码 vs. 通过密集热图检测构造图）以及两种输出表示（基于房间中心化的多边形序列 vs. 壁面优先的语法约束解码），并提出编辑成本度量、ResPlan‑FP基准以及多种融合/调和策略。

**💡 创新点**

创新点包括：①在同一训练网络上系统性对比读取方式，发现读取方式对计划大小和域有显著影响；②验证匹配数据/配方下，房间中心化与壁面优先在墙结构质量上可实现几乎相同的表现；③提出基于编辑成本的指标，强调人类纠错成本而非单纯 F1；④展示输出级融合可提升约 7 点墙 F1 并显著降低编辑成本；⑤发布公开可再现的 ResPlan‑FP 基准与修正后的 CubiCasa5K 标注。

**🔧 技术方法**

使用技术包括：语法掩码的自回归解码、坐标量化、密集 junction/centerline/opening 热图、基于最大化匹配的 Hungarian 匹配、编辑脚本与加权成本、deterministic 融合（ink gate 与 openings‑strip）、信息论分析、以及多模态数据预训练（Structured3D）。

**📊 数据集**

使用数据集：CubiCasa5K（真实扫描，修正标注）、合成渲染集合（ResPlan、开放式合成）、ResPlan‑FP 公开基准（16,998 规划图）、以及 Structured3D 作为预训练来源。

**📈 对比分析**

对方法的比较采用 F1（墙、房间、开口）、双墙计数、可闭合率等指标。结果显示：在 CubiCasa5K 上检测读取优于序列解码；在合成清洁渲染上序列解码更优；房间中心化+调和与壁面优先序列在墙结构上几乎可等价；融合策略可将墙 F1 提升约 7 点，并将编辑成本从 79.3 降至 73.1。

**⚠️ 局限性**

局限性包括：①仅比较系统级表现，未纯粹评估表示本身；②房间中心化模型未充分调优，可能低估其性能；③未单独对监督方式进行对比；④信息论解释适用于输入条件的约束，但不一定适用于所有网络架构；⑤公开基准未覆盖所有第三方方法，可能影响对比完整性；⑥数据集近似重叠导致上限估计可能偏高。

---

## 384. A Dual-Transformer for Multi-Camera View Recommendation

**arXiv ID:** 2608.25601 | [PDF](https://arxiv.org/pdf/2608.25601v1)

**作者:** Josep Cabacas-Maso `[一作]` (Universitat Oberta de Catalunya), Ismael Benito-Altamirano `[通讯]` (Universitat Oberta de Catalunya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种双Transformer架构用于多摄像机视角推荐。

**💡 创新点**

通过将时间编码与候选视角评估解耦并引入交叉注意力实现显著性能提升。

**🔧 技术方法**

使用Swin Transformer V2 backbone、跨注意力、多头注意力以及Focal Loss。

**📊 数据集**

在大型TVMCE多摄像机编辑数据集上进行训练与评估。

**📈 对比分析**

相较于之前SOTA（Lee等2025）提升约32个百分点，Precision@0.5达到69.65%。

**⚠️ 局限性**

局限在缺乏多模态输入以及对极端场景的泛化尚待验证。

---

## 385. Cross-Dataset Stability of Expert-Informed Skill Prompting and Fine-Tuning for Chinese Metaphor Identification

**arXiv ID:** 2608.25579 | [PDF](https://arxiv.org/pdf/2608.25579v1)

**作者:** Yufeng Wu `[一作]` (City University of Hong Kong), Meichun Liu `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了四种知识供给模式（BERT-FT、LLM-FT、LLM-ZS、Skill-ZS）用于中文句子级隐喻识别，并在三套数据集上进行跨数据集性能比较。

**💡 创新点**

首次在同一跨数据集评估中将专家驱动的流程化Skill与任务特定微调对比，发现Skill在外部数据集上表现更均衡，并提出基于流程的零射击方法与精细的跨数据集稳定性指标。

**🔧 技术方法**

使用BERT和QLoRA大语言模型微调、零射击LLM提示、冻结的专家程序化Skill（六步推理流程），以及宏F1等评估指标。

**📊 数据集**

使用CMRE（原始训练/开发/测试）、CCIME 1200条任务1开发样本、CMC公开的保留中文拆分。

**📈 对比分析**

通过宏F1、外部平均、外部最低、原始-外部差值、三数据集范围等指标衡量跨数据集稳定性；结果显示BERT-FT最高原始宏F1，LLM-FT最高外部平均，Skill-ZS外部最低点最高且范围最小；Skill 通过减少隐喻预测降低CCIME假阳性，但在CMRE Test和CMC增加假阴性。

**⚠️ 局限性**

仅使用三数据集且注释政策不一致；零射击条件仅有单一确定性跑，缺乏不确定性估计；未评估解释质量或不同语言/体裁；Skill流程细节未做逐项消融。

---

## 386. PolyMemDB: A Polyglot Database System for AI Memory Management

**arXiv ID:** 2608.25577 | [PDF](https://arxiv.org/pdf/2608.25577v1)

**作者:** Yu Wang `[一作]` (University of Helsinki), Jiaheng Lu `[通讯]` (University of Helsinki)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为个人智能助手设计并实现了 PolyMemDB 多模态内存管理系统，支持图、向量、概率和时空数据的统一存储与检索，并通过概率推理与时间衰减解决长期事实冲突。

**💡 创新点**

创新点：①polyglot 存储层整合多种数据库，实现多维异构数据高效存储；②基于语义衰减的动态图更新与半环推理，提供细粒度数据源追溯与事实可靠度评估；③三层级联检索与推理链展示，提升可解释性与减少 LLM 幻觉。

**🔧 技术方法**

技术：Neo4j 图数据库、ProvSQL 概率数据库、MobilityDB 时空数据库、ChromaDB 向量库、FastAPI、Pydantic-AI；LLM/VLM 进行实体对齐、共指消解、命名实体抽取；指数衰减与半环推理；三层级联检索与可视化界面。

**📊 数据集**

数据集：LongMemEval 基准（48 期长时段问答对话）、扩展 LongMemEval 版本（含旅行日志等多模态对话）。

**📈 对比分析**

比较方法：与 MemForest、HyperMem、MRAgent 等现有内存系统对比；评估指标包括查询准确率、冲突解决率、LLM 幻觉率。PolyMemDB 在长时段 QA 中达 20% 的准确回答率，显著降低幻觉；在时空查询中提供细粒度可视化，整体性能优于对标系统。

**⚠️ 局限性**

限制：系统集成复杂，需多数据库部署；查询时延受多层检索影响；时间衰减参数需手动调优；在极大规模对话中仍可能出现检索瓶颈；缺乏正式的长期性能基准与大规模实验。

---

## 387. Generative vs. Encoder Large Language Models for ASR Evaluation: A Comparative Study

**arXiv ID:** 2608.25574 | [PDF](https://arxiv.org/pdf/2608.25574v1)

**作者:** Thibault Bañeras-Roux `[一作]` (Idiap Research Institute), Richard Dufour `[通讯]` (Nantes Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了编码器与解码器大型语言模型在ASR评估中的表现，研究了BERTScore、SemDist以及生成式LLM在选择最佳假设和对单个假设进行质量分类时的效果。

**💡 创新点**

首次系统地将编码器嵌入、解码器嵌入和生成式LLM放在同一框架下评估，并证明生成式LLM可以作为“判官”在ASR评估中超越传统指标。

**🔧 技术方法**

使用BERTScore、SemDist、GPT‑4.1、Gemma、Qwen3系列等LLM，并通过层级嵌入、不同池化策略、prompting和对比任务实现评估。

**📊 数据集**

采用HATS数据集（法语ASR人类评估对照）。

**📈 对比分析**

通过与人类判断的Spearman/pearson相关性以及一致率对比，编码器模型如Sentence‑CamemBERT‑Large与嵌入化Qwen3‑Embedding‑8B在嵌入度量上可达约80%的一致率；生成式LLM在对比选择任务上达到94%的一致率，明显优于WER/CER。

**⚠️ 局限性**

模型表现受层级与聚合策略影响敏感，较小或未优化的模型表现不佳；生成式LLM的质量分类相关性仍中等，提示或细调有待改进，且计算成本相对较高。

---

## 388. Reassembling Distributed Risk: Trajectory-Conditioned Action Generation for Multi-Turn Agent Safety

**arXiv ID:** 2608.25711 | [PDF](https://arxiv.org/pdf/2608.25711v1)

**作者:** Yanbo Dai `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ReDiR的生成时防御方法，用于在多轮对话中通过轨迹级安全证据控制LLM代理的动作生成，从而防止分布式恶意指令导致的安全风险。

**💡 创新点**

核心创新在于：①通过同一模型跨视角监督学习，利用“安全编码器”把整个交互轨迹压缩为一个紧凑的安全潜在表示；②在每一步动作生成前将该潜在表示注入冻结的基础模型，直接让已聚合的安全信息影响后续生成，而不需要额外的后期动作检查；③采用安全编码器和可训练的隐式查询以实现跨工具、跨框架的迁移性和高效性。

**🔧 技术方法**

技术主要包括：基于Transformer的安全编码器（共享基础模型 + LoRA适配器 + K个latent query）；同模型跨视角（同一模型在“已压缩任务视图”下生成安全目标，再通过监督引导编码器学习）；token-level加权训练（entry-token、decision-token 权重）；与冻结的生成模型在相同hidden空间注入潜在表示；以及多轮交互的轨迹压缩与重编码机制。

**📊 数据集**

使用了两个公开安全基准：MT-AgentRisk（将单轮攻击转化为多轮轨迹，覆盖八类工具环境）和AgentDojo（测试间接提示注入的攻击）；训练集中以文件系统任务为主（55/70任务），测试集则覆盖八个未见工具域，共计365个攻击实例。

**📈 对比分析**

与基线（无防御）、MAGE（轨迹级在线判断）和ToolShield（基于体验的工具安全）比较。ReDiR在所有模型族（Qwen3.5-9B、Ministral-3-8B、Gemma-4-E4B）上将攻击成功率（ASR）降至0–8%，SSR提升至80–99%，并保持0% FPR；相较于MAGE的ASR 20–34%，表现明显更佳。跨工具迁移方面，ReDiR在未见域的ASR保持在0–7%，而MAGE和ToolShield表现为15–33%和>30%；同时保持低GPU内存开销（<4 MiB）和合理的在线延迟（≈1–3×MAGE）。

**⚠️ 局限性**

局限性包括：①对安全编码器与基础模型的共享架构要求较高，若使用不同后端可能需要更多latent容量；②同模型监督的效果在很大程度上依赖于生成模型的能力，跨模型迁移时表现下降；③对恶意攻击者的“安全编码器攻击”如benign-history dilution仍存在一定抗御，但并非完全不受影响；④需额外训练成本（约30–40 GPU小时），且对训练数据与标注质量敏感；⑤在极端复杂的多工具、多步骤任务中，潜在表示的压缩可能不足以捕捉所有细粒度风险。

---

## 389. Difficulty-Aware Sample Allocation for Adaptive Data Augmentation in Semantic Segmentation

**arXiv ID:** 2608.25710 | [PDF](https://arxiv.org/pdf/2608.25710v1)

**作者:** Olasimbo Ayodeji Arigbabu `[一作]` (Independent Researcher), Abimbola Ismail Arigbabu `[通讯]` (Olabisi Onabanjo University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究提出 DASA 通过多因素难度评估实现对语义分割训练样本的自适应增强分配。

**💡 创新点**

创新点在于将预测不确定性、损失、类别稀缺性和边界复杂度四项指标融合成难度得分，动态控制增强强度。

**🔧 技术方法**

采用多因素难度估计、归一化、线性加权求和，结合已有数据增强策略（如 RandAugment 等）实现按样本强度调整。

**📊 数据集**

使用 Oxford‑IIIT Pet（trimap）和 Pascal VOC（二值前景/背景）两个数据集进行评估。

**📈 对比分析**

与标准训练、强统一增强、随机权重、单因子自适应等基线对比，DASA 在三种网络（U‑Net、DeepLabV3、SegFormer‑B0）上均提升 mIoU，尤其在 DeepLabV3 上从 0.633 提升至 0.740。

**⚠️ 局限性**

限制包括额外的前向传播开销（需多次 MC Dropout 估计不确定性）、需手动设定权重且对不同数据集的最优权重未知。

---

## 390. Fairness-Aware Test-Time Prompt Tuning

**arXiv ID:** 2608.25707 | [PDF](https://arxiv.org/pdf/2608.25707v1)

**作者:** Yoann Launay `[一作]` (University of Cambridge), David Sutton `[通讯]` (Visa Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并改进了视觉‑语言模型在无标签测试时的单例适配（TTA），提出 FairTPT 通过双熵目标在保持整体准确率的同时抑制敏感属性的影响，提升子群公平性。

**💡 创新点**

创新点在于：① 设计无监督的双熵（最小化目标熵、最大化敏感熵）损失；② 引入轻量学习率自适应（ELRA）防止模型崩溃；③ 在单例 TTA 环境下实现可解释、对超参数鲁棒的公平调优。

**🔧 技术方法**

使用的技术包括：CLIP 预训练模型、软提示（soft‑prompt）调优、熵最小化与熵最大化、梯度冲突处理（Jacob‑descent）、学习率自适应、数据增强（AugMix）。

**📊 数据集**

评估数据集包括 CelebA、UTKFace、FairFace、WaterBirds，分别测试不同的敏感属性（如性别、种族、发色、笑容等）。

**📈 对比分析**

与 TPT、Zero、OrthCali 等基线对比，FairTPT 在保持零样本准确率（≈95%）的同时，显著提升最差群体准确率和公平度量（Bias、EOD）超过 2% 并且对学习率、阈值等超参数更稳健。

**⚠️ 局限性**

局限性：需要人工提前指定敏感属性，属性误判会导致偏差加剧；在某些任务中计算量和延迟增加；只能去除已指定属性的直接关联，无法消除间接代理；未解决对未知或多重属性的公平性。

---

## 391. Beam Search, Self-Consistency, and the Limits of Inference-Time Scaling for Grammar-Constrained Text-to-SQL in Small Language Models

**arXiv ID:** 2608.25761 | [PDF](https://arxiv.org/pdf/2608.25761v1)

**作者:** Ty Chermsirivatana `[一作]` (Dickinson College), John MacCormick `[通讯]` (Dickinson College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在语法约束的文本到SQL任务中，比较Beam Search与Sample+Vote在不同模型规模下的推理计算对准确率的影响；

**💡 创新点**

首次量化语法约束环境下Beam Search在与Sample+Vote同等计算预算时性能更优，并揭示推理计算提升无法完全弥补模型规模缩小的缺失，填补该领域先前缺失的实验结论；

**🔧 技术方法**

使用Qwen2.5-Instruct系列LLM（0.5B‑7B）4‑bit量化，基于上下文无关文法的语法约束解码；对Beam Search采用可变宽度，Sample+Vote采用温度0.7+top‑p0.9采样并执行结果投票；

**📊 数据集**

Spider文本到SQL基准的开发集（1034例），通过执行结果对比评估准确率；

**📈 对比分析**

在同一批量1034样本上进行单次解码，预算B={1,2,4,8}；Beam Search在绝大多数配置下显著优于Sample+Vote，模型规模越大、Beam宽度提升收益越小；推理计算提升对小模型准确率提升显著，但对大模型收益有限；

**⚠️ 局限性**

仅评估单一模型家族（Qwen2.5）、单一基准、单次采样，未考虑不同模型、不同约束类型、不同数据集或多次随机抽样，结果可能不具普适性；

---

## 392. Learning from waste: Machine Learning for health risk prediction and computer vision-based sorting in Ghana

**arXiv ID:** 2608.25759 | [PDF](https://arxiv.org/pdf/2608.25759v1)

**作者:** Hilda Adwubi Osei `[一作]` (Kwame Nkrumah University of Science and Technology), Desdemona Yaa Asobayire `[通讯]` (University of Nottingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用社区调查数据训练随机森林模型预测居民疾病类型，并使用 MobileNetV2 对 TrashNet 图像进行废弃物分类

**💡 创新点**

首次将机器学习与社区自报的废弃物处置方式关联以量化疾病风险，并提出摄像头驱动的低成本废弃物分类方案替代多传感器机械分拣

**🔧 技术方法**

随机森林、逻辑回归、决策树、MobileNetV2 迁移学习、预处理与特征重要性分析

**📊 数据集**

社区问卷调查（470份）与公开 TrashNet 约2500张标注废弃物图像

**📈 对比分析**

随机森林在受试样本上宏 F1=0.63、准确率0.62；MobileNetV2 在 TrashNet 测试集上宏 F1=0.87、准确率88.2%，与公开基准相当或优于传统视觉分类模型

**⚠️ 局限性**

样本量小、交叉验证与实际预测差异大；疾病标签自报易出现误差；图像模型未在现场混合、受损废弃物上验证，缺乏对真实环境的适应性评估

---

## 393. Modeling spatio-temporal locality in multi-step forecasting of geo-referenced time series

**arXiv ID:** 2608.25698 | [PDF](https://arxiv.org/pdf/2608.25698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 394. Quantum Blackhole Learning-Optimized Hadamard Neural Network Model for Dynamic Resource Reservation in Industry Clouds

**arXiv ID:** 2608.25754 | [PDF](https://arxiv.org/pdf/2608.25754v1)

**作者:** Deepika Saxena `[一作]` (University of Aizu), Anand Mohan `[通讯]` (Indian Institute of Technology (BHU))

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种量子黑洞优化的Hadamard神经网络（QB‑HNN），用于工业云的动态资源预留和工作负载预测。

**💡 创新点**

创新点在于将Hadamard门的量子叠加激活函数与量子黑洞双相优化（QB‑BiO）算法相结合，形成既具量子学习能力又能高效搜索权重的模型。

**🔧 技术方法**

采用量子位编码、Hadamard门激活、量子黑洞双相优化算法和经典的时间序列预处理技术实现模型训练与推断。

**📊 数据集**

使用六个真实云工作负载数据集：Google Cluster（CPU与内存）、NASA‑HTTP、Saskatchewan、AuverGrid和SHARCNet，涵盖集群、Web和HPC三类场景。

**📈 对比分析**

通过与BPNN、SaDE、BaDE、LSTM、EQNN等五种先进方法在MSE、MAE、RMSE等指标上对比，QB‑HNN在所有数据集上均显著降低误差（平均约36%~73%），并在训练收敛速度与计算复杂度方面表现更优。

**⚠️ 局限性**

主要局限包括：对现有云管理平台的互操作性挑战、在无量子硬件时的经典计算开销、需要进一步验证在大规模工业环境中的可部署性和实时再优化能力。

---

## 395. MIMONet: Multi-scale Input and Multi-scale Output Network for Salient Object Detection

**arXiv ID:** 2608.25733 | [PDF](https://arxiv.org/pdf/2608.25733v1)

**作者:** Zhaojian Yao `[一作]` (Peking University), Sam Kwong `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多尺度输入与多尺度输出的显著性检测网络 MIMONet，利用多尺度图像并行输入并通过特征互换、MSP 模块以及联合显著性损失实现对不同尺度目标的精准检测。

**💡 创新点**

创新点包括：1) 并行多尺度输入并在同一网络中进行特征交互；2) 设计 Multi‑Scale Perception (MSP) 模块，利用下采样/上采样与不同卷积核捕获多尺度结构信息；3) 引入 Joint Saliency Loss (JSL)，约束多尺度输出一致性并强化边界保留。

**🔧 技术方法**

使用共享参数的 ResNet50 编码器；三分支特征交换机制；MSP 采用三分支的下采样/上采样与 3×3、5×5、7×7 卷积；联合使用 BCE+IoU 以及 JSL 损失；训练采用 Adam + cosine annealing。

**📊 数据集**

训练数据集为 DUTS‑train，评估数据集包括 ECSSD、PASCAL‑S、HKU‑IS、DUT‑OMRON、DUTS‑test 与 SOC 共六个标准显著性检测数据集。

**📈 对比分析**

与 21 种最新方法对比，MIMONet 在大部分数据集上获得最高或接近最高的 Fβ、S_m、E_m 等评价指标，参数量不到 30M，且实时推理速度达到 32 FPS。

**⚠️ 局限性**

局限性：多尺度输入显著增加计算和内存需求；在极端尺度变化或高度复杂场景中仍可能出现边界模糊；未来可尝试扩展至点云等多模态显著性检测。

---

## 396. From Verdict to Diagnosis: Attributable Security Review of Pull Requests

**arXiv ID:** 2608.25730 | [PDF](https://arxiv.org/pdf/2608.25730v1)

**作者:** Zhuo Chen `[一作]` (University of Bristol), Lichao Wu `[通讯]` (University of Bristol)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个评估PR安全审查员的范式，重点衡量“判决–诊断”差距，并构建了基于机制的PRGuard基准；

**💡 创新点**

创新点在于将判决、漏洞识别和证据验证拆分为独立指标，设计了可分阶段、结构化证据检索的可归因审查系统（DeepSeek/PRGuard）来缩小判决–诊断差距；

**🔧 技术方法**

使用了大型语言模型（GPT‑5.5、DeepSeek v4‑pro）结合静态分析工具，采用结构化提问、类型化检索、分阶段验证等技术；

**📊 数据集**

使用的数据集包含89个恶意PR和50个安全修复对照，共44个仓库、8种语言，涵盖从历史挖掘、公开安全通告到12个生产级未公开漏洞；

**📈 对比分析**

与商业产品CodeRabbit及CodeQL进行对比，发现DeepSeek在判决、识别和证据验证上均优于CodeRabbit，尤其在缺失防护缺陷（absence‑type）上提升三倍；

**⚠️ 局限性**

局限性包括：评估基于预先冻结的基准、未估计召回率、模型内部黑盒、实验只覆盖有限的PR类型和语言、以及对不同证据位置的分层处理仍不够细粒度。

---

## 397. Dynamic Polyhedral Logic

**arXiv ID:** 2608.25691 | [PDF](https://arxiv.org/pdf/2608.25691v1)

**作者:** Nick Bezhanishvili `[一作]` (University of Amsterdam), David Gabelaia `[通讯]` (TSU Razmadze Mathematical Institute)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

本文提出了一种融合可逆动力学、平面多面体语义和基于路径的空间可达性运算的动态拓扑逻辑，并给出了其完整的公理系统。

**💡 创新点**

创新点在于：①首次在多面体语义下定义并证明带有可达性算子（γ）的动态逻辑的完备性；②利用可逆动力学（homeomorphism）引入前后两种时间算子；③将时间步移压缩到原子层级的翻译方法，简化证明。

**🔧 技术方法**

主要技术包括：拓扑语义、可达性代数、PL（piece‑wise linear）映射的几何性质、归纳翻译函数 g(·) 与动态空间构造，以及旋转复制构造实现可逆动力学。

**📊 数据集**

无数据集；该工作为纯理论框架，未进行实验验证。

**📈 对比分析**

本研究通过形式化证明完成了逻辑的完备性，没有进行实验对比；理论层面上与已知的 TL、ALR、PLR 逻辑等保持一致，并扩展到可逆动力学情形。

**⚠️ 局限性**

局限性包括：仅处理可逆动力学（homeomorphism），对非可逆系统尚未完成；γ 算子在涉及“未来”算子时可能无法保持多面体闭合；开放问题如包含“eventually”算子的多面体动态逻辑的可判定性和有限模型性质仍未解决。

---

## 398. TailSFT: Filtered Fine-Tuning Improves Post-Training Performance

**arXiv ID:** 2608.25756 | [PDF](https://arxiv.org/pdf/2608.25756v1)

**作者:** Sadhika Malladi `[一作]` (University of California San Diego), Akshay Krishnamurthy `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的监督微调算法TailSFT，旨在通过过滤已适应的序列来提高后续强化学习的覆盖率，从而改善模型的推理和代理能力。

**💡 创新点**

创新点在于TailSFT算法优先考虑覆盖率而非低交叉熵损失，专注于学习数据分布的尾部区域，以提高后续强化学习的性能。

**🔧 技术方法**

使用了理论分析和控制实验相结合的方法来验证TailSFT的设计选择，并在OLMo-3 7B模型上进行实验。

**📊 数据集**

在OLMo-3 7B模型上使用了标准的数学和编码任务数据集，包括OpenMathInstruct-2和Magicoder等。

**📈 对比分析**

与标准的监督微调(SFT)方法相比，TailSFT在后续的强化学习中表现出更好的覆盖率和性能提升，具体表现为在编码任务上提高了高达16.8%的绝对性能。

**⚠️ 局限性**

限制在于TailSFT的效果依赖于初始模型的质量，且在某些情况下可能无法完全消除标准SFT的不足。

---

## 399. Hamiltonian Spectral-Temporal Dissipative Dynamics for Sequential Recommendation

**arXiv ID:** 2608.25755 | [PDF](https://arxiv.org/pdf/2608.25755v1)

**作者:** Shuiying Liao `[一作]` (Hong Kong University of Science and Technology), P. Y. Mok `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于耗散 Hamiltonian 二阶动力学的序列推荐模型 HSR，利用相空间（位置与动量）来刻画用户兴趣演化，并通过频域传播实现高效建模。

**💡 创新点**

创新点在于：① 将用户兴趣视为二阶动力学系统，首次把惯性、周期性和冲击等行为特征映射到物理模型；② 设计可学习的频域传播器和局部冲击分支，实现全局平滑与局部突变的并行建模；③ 在预测时采用一次相空间外推，将动量转化为下一步兴趣预测，显著提升前瞻性。

**🔧 技术方法**

核心技术包括：离散傅里叶变换（FFT/iFFT）求解二阶差分方程、可学习的质量、阻尼、刚度参数、局部深度卷积冲击分支、门控融合、一次欧拉外推以及全连接评分层。

**📊 数据集**

在三个公开基准上评估：MovieLens‑1M、Amazon‑Beauty、Amazon‑Video‑Games，三者分别代表密集、稀疏与短序列场景。

**📈 对比分析**

与 Transformer（SASRec、BERT4Rec、HSTU等）、SSM（Mamba4Rec、SIGMA、SSD4Rec）以及专门的差分模型（DIFF）等最先进方法对比，HSR 在 Hit@10、NDCG@10、MRR@10 上均实现显著提升（平均提升约 1–2% 以上），并在参数量、推理延迟和吞吐量上优于多数基线。

**⚠️ 局限性**

局限性包括：① 只使用线性二阶动力学，可能无法捕捉更复杂的非线性兴趣跳变；② 对超长序列或多模态输入的适用性尚待验证；③ 需要手动设定时间步长和初始条件，模型对超参数较敏感。

---

## 400. REE-TM: Reliable and Energy-Efficient Traffic Management Model for Diverse Cloud Workloads

**arXiv ID:** 2608.25747 | [PDF](https://arxiv.org/pdf/2608.25747v1)

**作者:** Ashutosh Kumar Singh `[一作]` (Indian Institute of Information Technology Bhopal), Volker Lindenstruth `[通讯]` (Goethe University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种可靠与能效兼顾的交通管理模型 REE‑TM，通过对云工作负载进行分层分类并结合量子神经网络预测与熵分析，实现资源调度与能耗优化。

**💡 创新点**

将 Toffoli 门嵌入量子神经网络与量子黑洞优化结合，用熵驱动的流量状态划分以及可靠性评估，形成端到端的自适应调度框架。

**🔧 技术方法**

量子门驱动神经网络 (TG‑QNN)、量子黑洞优化 (QBHO)、熵分析 (TSECE)、传统可靠性与能耗模型、Python 仿真环境等技术。

**📊 数据集**

Google Compute Cluster (GCD) 的 CPU/内存/磁盘 I/O 追踪数据，共 672,300 个作业。

**📈 对比分析**

与 OPTIMAL、W‑REE‑TM*、W‑REE‑TM** 三种基准以及 GNN/ATTN/EQNN/ENN 等预测模型对比；REE‑TM 在可靠性提升 30% 以上、能耗降低 20–23% 以及成功率接近最优。

**⚠️ 局限性**

量子硬件实现尚未验证，预测误差仍会影响可靠性；对极端负载突发的鲁棒性需进一步评估。

---

## 401. A Constitutive Markov Physics-Informed Neural Operator (MPNO) for Autoregressive Stability in Transient Dynamics

**arXiv ID:** 2608.25744 | [PDF](https://arxiv.org/pdf/2608.25744v1)

**作者:** Wenpu Du `[一作]` (North University of China), Wenzheng Xu `[通讯]` (North University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种构成性马尔可夫物理信息神经算子（MPNO），用于建模具有强不连续性的瞬态动力学偏微分方程（PDEs），并解决了自回归不稳定性的问题。

**💡 创新点**

MPNO通过构造性地限制传播算子的谱半径（ρ(P)≤1）来保证自回归稳定性，而不是通过优化损失目标来实现，稳定性在多个实验中得到了实证验证。

**🔧 技术方法**

使用了马尔可夫传播算子、物理耦合的边权重（声阻抗调和平均、接触面积和牵引幅度）以及在线的Rayleigh商幂迭代来估计谱半径。

**📊 数据集**

使用了Burgers方程、Darcy流动和混凝土穿透等三个PDE场景的数据集，混凝土穿透数据集由LS-DYNA显式动力学仿真生成。

**📈 对比分析**

与WNO和FNO等方法相比，MPNO在自回归滚动中表现出稳定性，WNO在所有测试种子上发散，FNO在实践中稳定但没有谱半径的结构保证。MPNO的单步相对L2误差为0.7304±0.0008，优于WNO且与FNO相当。

**⚠️ 局限性**

MPNO的局限性在于其依赖于物理耦合的边权重构造，可能在某些复杂材料行为下表现不佳，且在高速度情况下的表现可能受到影响。

---

## 402. Why Does Graph Learning Fail to Fully Benefit from a Text Teacher?

**arXiv ID:** 2608.25741 | [PDF](https://arxiv.org/pdf/2608.25741v1)

**作者:** Fumiaki Kimino `[一作]` (Graduate University for Advanced Studies), Ryoma Sato `[通讯]` (National Institute of Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究尝试将自监督图神经网络（FUG）与EM式多模态学习框架（GLEM）相结合，形成一种跨域文本增强的图学习方法；

**💡 创新点**

创新点在于：①提出一种独立的文本教师（TextHead）与图编码器的EM式交替训练，避免教师信息被图模型本身复制；②通过多种锚点（文本、MLP、原始哈希、外部语义）对图表示进行多向对齐；

**🔧 技术方法**

使用的主要技术包括：Feature‑Universal Graph Pre‑training（FUG）自监督学习；EM‑style（E‑step、M‑step）训练框架；文本编码器TextHead与多种对齐损失；余弦锚点对齐；

**📊 数据集**

实验数据集：源域为Amazon Digital Music（产品评论网络），目标域为OpenAlex（学术知识图谱，论文网络）；

**📈 对比分析**

与仅使用FUG的基线进行比较，使用标准线性探针和均衡探针评估。结果显示FUG+GLEM‑ITT在标准精度上略微提升（0.7459→0.7480，+0.21%），均衡精度略降（0.6016→0.6001），提升不显著；

**⚠️ 局限性**

主要局限包括：①外部锚点存在强度‑安全权衡，过强会损伤图表示；②教师知识未直接注入GCN表征，导致信息压缩与失真；③图表示空间与教师语义空间目标不一致；④GCN传播会稀释节点特定文本信息；⑤仅靠余弦对齐不足以提升分类边界；⑥源图自监督目标与教师对齐目标冲突，导致最终表示为两者的妥协。

---

## 403. D3ER: Supporting Multi-Modal Recommendation via Disentangle and Distillation-based Dynamic Ensemble

**arXiv ID:** 2608.25737 | [PDF](https://arxiv.org/pdf/2608.25737v1)

**作者:** Bingnan Wang `[一作]` (Chinese Academy of Sciences), Jiangmeng Li `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的多模态推荐框架 D3ER，结合特征拆解与知识蒸馏的梯度提升，专门学习并融合模态同质信息（HOI）与异质信息（HEI），提升推荐性能。

**💡 创新点**

创新点包括：①首次将梯度提升引入多模态推荐，并通过交替学习解耦 HOI 与 HEI 的样本导向判别信息；②设计 FCD（Feature Component Disentanglement）模块，利用实例级 InfoNCE 与分布级 Wasserstein 对齐实现特征解耦；③在提升过程中加入蒸馏与全局校正正则，降低存储成本并缓解局部最优问题。

**🔧 技术方法**

使用技术：多模态特征拆解（instance-level 对齐 + distribution-level 对齐 + intra-modal 分离），梯度提升（KDBoost）与知识蒸馏，GCN 作为用户/物品嵌入编码器；预训练模型包括 CLIP‑ViT、VGG‑16 视觉编码器和 Sentence‑BERT 文本编码器。

**📊 数据集**

实验数据集：Amazon 购物评论的 Baby、Sports、Clothing 三个子集，分别使用预训练的 CLIP‑ViT/VGG‑16 提取视觉特征，Sentence‑BERT 提取文本特征。

**📈 对比分析**

与 15+ 传统与多模态基线（MF‑BPR、LightGCN、MGCN、MMSSL、DiffMM、PGL 等）在 Recall@20/50 和 NDCG@20/50 上进行对比。D3ER 在所有数据集上均实现最高 Recall 与 NDCG，平均提升约 4–7%（Recall@20 最高 10.58%），证明方法的有效性。

**⚠️ 局限性**

局限性：①梯度提升仍易陷入局部最优，需要全局校正正则；②对阈值 d_m 与损失权重 α_c/α_w 等超参数敏感；③增加了蒸馏与正则化步骤，模型训练复杂度提升；④仅验证了三种子集与三种模态，未评估更大规模或更多模态的泛化能力。

---

## 404. Predicting Struggling Students in CS1 Programming Using Keystroke-Level Editing Features

**arXiv ID:** 2608.25769 | [PDF](https://arxiv.org/pdf/2608.25769v1)

**作者:** Yasuyo Kofune `[一作]` (Nara Institute of Science and Technology), Kenichi Matsumoto `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文利用CodeBench平台记录的键盘打字和执行日志，对CS1编程课程中学生在练习过程中的挣扎情况进行早期预测，区分最终突破（BT）和始终卡住（FS）两类学生。

**💡 创新点**

创新点在于首次将细粒度键盘编辑日志作为特征，证明其在练习最初阶段（首次提交前）就能提供比仅使用提交结果更好的预测信号，并提出基于提交间隔的分段预测框架。

**🔧 技术方法**

使用了随机森林和逻辑回归两种分类器，提取了执行日志特征（ExecOnly）、CodeMirror编辑特征（CMOnly）以及两者组合特征（Combined），并评估了不同提交分段（k=1至k=10）的预测性能。

**📊 数据集**

实验基于2019-1学期的CodeBench数据集，共507名学生，涵盖约35,887个学生–练习对，标签为BT（11,999例）和FS（2,101例）。

**📈 对比分析**

与仅使用执行日志相比，添加编辑日志在首次分段的AUROC提升0.098（从0.575到0.674），并且所有配置在k=1时表现最好；在后续分段性能下降，说明最早阶段信息最丰富；在FS对其他所有标签的比较中，Combined模型AUROC可达0.839，PR-AUC为0.224。

**⚠️ 局限性**

局限包括：仅基于单个学期单一机构的数据，缺乏对不同难度练习和语言的泛化；未考虑生成式AI工具对编辑行为的影响；FS标签可能混合了彻底失去兴趣的学生，导致构念效度受限；预测准确度仍偏低，实际应用需结合教师资源和阈值设定。

---

## 405. Study of Resistive Switching Dynamics and Memory States Equilibria in Analog Filamentary Conductive-Metal-Oxide/HfOx ReRAM via Compact Modeling

**arXiv ID:** 2608.25767 | [PDF](https://arxiv.org/pdf/2608.25767v1)

**作者:** Matteo Galetta `[一作]` (IBM Research Europe-Zurich), Valeria Bragaglia `[通讯]` (IBM Research Europe-Zurich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种针对CMO/HfO_x ReRAM的物理基础紧凑模型，能够准确模拟其模拟电阻开关特性、单脉冲编程动力学和双向累积导通响应，并分析其对称点与平衡态的关系。

**💡 创新点**

在模型中引入外部寄生电阻校正、离子迁移动力学修正、动态霍普金斯参数、随机TAT噪声模型，以及利用动态路由图解析对称点与平衡态对应关系，能够通过脉冲幅度调节实现对称点平衡，从而提升训练精度。

**🔧 技术方法**

采用 Trap‑Assisted Tunneling + Mott‑Gurney hopping 物理模型、有限元电热仿真、离子迁移动力学、随机 TAT 噪声、动态路由图 (DRM) 分析，并在 SPICE / ai-hw-kit 等仿真平台上进行模型验证。

**📊 数据集**

使用 MNIST 手写数字分类数据集来评估 Tiki‑Taka 算法在 ReRAM 设备上的训练性能。

**📈 对比分析**

通过将模型仿真结果与实验测得的 I‑V 曲线、单脉冲 SET 速度、双向累积导通曲线、NRMSD 等指标对比，模型误差低于 3%，在 3‑FC 网络中实现近似浮点精度（准确率约 99%）且收敛速度最快时与 SP 对称性最接近。

**⚠️ 局限性**

模型尚未实现完全可差分，难以直接集成至标准 SPICE 等电路仿真器；缺乏对不同尺寸、材料组合的广泛验证；极端高温和长期耐久性实验数据不足。

---

## 406. LM-X: Explainable Action Modeling with Progress, Event, and Uncertainty Prediction for Generalist Robot Manipulation

**arXiv ID:** 2608.25757 | [PDF](https://arxiv.org/pdf/2608.25757v1)

**作者:** Jin Lou `[一作]` (Humanoid Robot (Shanghai) Co., Ltd.), Yuchen Zhu `[通讯]` (Humanoid Robot (Shanghai) Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了LM-X，一种具备可解释的多时间尺度预测（进度RTG、事件ETG、动作不确定性）的通用视觉-语言-动作策略，并在大规模真实机器人数据上进行预训练和微调。

**💡 创新点**

创新点在于将任务进度、事件级意图和本地动作可靠性作为显式可观测的监督目标，并通过层级条件化将其嵌入控制路径，实现了内在可解释性。

**🔧 技术方法**

使用了Transformer-based视觉-语言编码器、Diffusion Transformer动作专家、RTG与ETG的监督目标以及基于流匹配的异方差不确定性估计，结合多尺度信息流。

**📊 数据集**

使用了超过20,000小时的真实机器人轨迹（包括1,000小时失败演练）和RoboTwin2.0模拟任务，共计50个任务进行微调与评估。

**📈 对比分析**

与GR00T N1.7等基线比较，LM-X在50个RoboTwin2.0任务的平均成功率提升至74.1%（比基线提升18.7个百分点），在七个真实机器人任务上平均成功率提升至68.6%（比基线提升17.9个百分点）。

**⚠️ 局限性**

局限性包括对事件定义的人工验证依赖、缺乏针对ETG的定量评估、未实现基于不确定性的闭环恢复以及对不同机器人结构的适配仍需进一步验证。

---

## 407. Pointing the Way, Hiding the Destination: Practical Private Dense Retrieval at Scale

**arXiv ID:** 2608.25735 | [PDF](https://arxiv.org/pdf/2608.25735v1)

**作者:** Peichun Hua `[一作]` (Chinese University of Hong Kong), Yunming Xiao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套两阶段的私密密集检索协议：先用学习得到的二进制哈希对大规模文本集合做高召回候选过滤，再对过滤后的候选集使用同态加密精确评分，并通过活跃安全 k‑OT 仅揭露用户所需的文档内容。

**💡 创新点**

创新点包括：① 将可学习的深度哈希作为粗粒度候选过滤，并以方向度量差分隐私保证查询隐私；② 仅在候选集上做同态计算和 OT，显著降低全量检索开销；③ 结合 int8 量化、LoRA 微调、BFV 同态加密与可扩展 OT，形成完整的安全协议并给出正式的 DP 与安全证明。

**🔧 技术方法**

使用的核心技术包括：深度学习哈希（LoRA 微调的 Bi‑Encoder + 线性哈希头）、方向度量 DP（vMF 噪声）与后处理、BFV 同态加密（packed 模式）、活跃安全 k‑OT（OOS 扩展）、int8 量化、预训练检索器、RAG 流水线以及多步协议重叠。

**📊 数据集**

评估数据集：BEIR 五个零样本语料（SciDocs、NQ、DBpedia‑Entity、Climate‑FEVER、FEVER），以及 2.68M 规模的 NQ 语料用于 RAG 端到端测试。

**📈 对比分析**

对比方法包括：全量检索、传统 LSH（随机超平面、Super‑Bit）、学习哈希（ITQ、IsoHash、BPR）以及无 DP 哈希。实验表明：在 K=500 时保留 98.8–100% 的 NDCG@10；在 2.68M 语料上，仅增加 0.73 s（≈10%）的延迟；与 P²RAG、RemoteRAG、PANTHER 等系统相比，latency 2–70 倍更快；在 DP 预算下可使用 K≈3000，检索质量几乎等同全量检索。

**⚠️ 局限性**

局限性：① 需要离线训练哈希模型，模型迁移或在线更新成本较高；② 在极端 DP 预算下候选集需要进一步增大，导致通信/计算开销上升；③ OT 方案仅支持一次性 k‑选择，无法无限制检索；④ 当前攻击模型假设用户不做恶意加密，且不考虑语料动态更新的场景。

---

## 408. LongVU-TTT: Causal Test-Time Training for Visual Resampling in Long Video Understanding

**arXiv ID:** 2608.25729 | [PDF](https://arxiv.org/pdf/2608.25729v1)

**作者:** Mahmoud Ahmed `[一作]` (KAUST), Mohamed Elhoseiny `[通讯]` (KAUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入卷积型Test-Time Training层，在视觉编码器与LLM之间对长视频进行时序上下文聚合并压缩帧数

**💡 创新点**

将fast‑weights参数化为分组卷积实现2D局部时序更新，并结合梯度方向与对齐损失自适应重要性评分，实现512→128帧压缩并保留关键视觉证据

**🔧 技术方法**

卷积fast‑weight TTT、分组卷积、梯度方向/对齐损失重要性采样、混合采样策略、异步CPU激活卸载、批量切片ViT等技术

**📊 数据集**

LLaVA‑CC3M、LLaVA‑Video‑178K、LLaVA‑OneVision训练集以及MLVU、LongVideoBench、Video‑MME、NExT‑QA、LVBench五大长视频基准

**📈 对比分析**

与同基线LLaVA‑Video及其他顶尖视频MLLM对照，长VU‑TTT在五个基准上均领先，最大提升约+5.6%（MLVU）至+8.4%（LVBench）

**⚠️ 局限性**

fast‑weights仅作时序聚合器，难以长期存储细节；压缩策略需固定帧预算；训练对算力与系统优化依赖较高

---

## 409. A Spatially-Aware Publish-Subscribe Middleware for IoT Applications

**arXiv ID:** 2608.25728 | [PDF](https://arxiv.org/pdf/2608.25728v1)

**作者:** Philipp Ungrund `[一作]` (University of Potsdam), Sukanya Bhowmik `[通讯]` (University of Potsdam)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在 MQTT5 之上实现空间感知的 Publish–Subscribe 中间件，支持基于物理实体的空间过滤与路由。

**💡 创新点**

创新点包括：可表达多阶段空间邻域的邻域定义方案、将空间过滤完全迁移到 broker 并通过标准协议实现；以及在不改动客户端的前提下利用几何世界模型进行动态空间解析。

**🔧 技术方法**

技术实现基于 Apache ActiveMQ Artemis（增强版）、PostGIS（空间数据库）和 Eclipse Paho MQTT（客户端库），并利用 MQTT5 的扩展字段携带空间信息。

**📊 数据集**

评估使用合成的随机空间过滤映射以及真实的柏林 OpenStreetMap 数据集（道路网络、商店等）。

**📈 对比分析**

通过测量吞吐量、过滤时延、订阅添加时延等指标与传统基于主题的 pub/sub 进行对比；主内存模式下可达 250k msg/s，过滤时延保持在数十微秒，显示出高效性；而数据库模式受限于通信开销。

**⚠️ 局限性**

局限性包括：世界模型假定静态（不支持移动物理实体）、对多阶段邻域的解析依赖数据库查询导致延迟、以及在极高负载下两种模式均会失效。

---

## 410. Are LLM-Enhanced GNNs Privacy-Safe?

**arXiv ID:** 2608.25727 | [PDF](https://arxiv.org/pdf/2608.25727v1)

**作者:** Longzhu He `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了在链接、标签和成员推断三种攻击场景下，LLM增强图神经网络（LLM‑Enhanced GNN）的隐私泄露风险；

**💡 创新点**

创新点在于构建了统一的五阶段评估框架（数据准备、模型训练、攻击、风险评估与防御分析），并在42种模型配置、6种攻击方法及6个真实文本图数据集上展开系统实验；

**🔧 技术方法**

采用了多种LLM特征增强技术（解释型与嵌入型），结合主流GNN骨干（GCN、SAGE、GAT、GIN、APPNP、SGC、SSGC）以及差分隐私（DP）防御机制；

**📊 数据集**

使用了Cora、CiteSeer、Ogbn‑Products、Tape‑Arxiv23、Instagram、Reddit六个文本属性图数据集；

**📈 对比分析**

与传统浅层文本表示基线相比，LLM增强模型在节点分类上显著提升准确率，但在链接、标签和成员推断攻击中隐私泄露风险显著增加；

**⚠️ 局限性**

局限在于仅覆盖公开数据集与特定LLM/DP方法，未探究不同攻击强度、模型泛化性以及更高效、更细粒度的防御方案。

---

## 411. Skeleton-based Zero-Shot Spatio-Temporal Action Localization via Weakly-Supervised Pretraining

**arXiv ID:** 2608.25701 | [PDF](https://arxiv.org/pdf/2608.25701v1)

**作者:** Koshiro Nagano `[一作]`, Taiki Sekii `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于骨架的零样本时空动作定位方法，可在无目标动作训练的情况下估计每个检测到的人实例动作；

**💡 创新点**

创新点在于：①引入Skeleton‑Language Feature Pooling Switching (SLPS)机制，在弱监督预训练阶段使用视频级聚合，推理阶段切换为实例级聚合；②提出Scene‑Mixed Discriminative Contrastive Learning (SM‑DCL)，通过混合场景增强多实例学习下的动作区分；

**🔧 技术方法**

技术包括：视觉‑语言对比学习（CLIP‑style）、多实例学习(MIL)、骨架特征提取与聚合、全局最大池化、投影头对齐、GMPool与MLP混合网络；

**📊 数据集**

使用数据集：Kinetics‑400（预训练），UCF101‑24、FDD、RWF‑2000、MF（评估）；

**📈 对比分析**

与传统弱监督和跟踪基方法对比，UCF101‑24上定位精度达到34.1% AP，FDD上帧级AP 73.3%，超过传统方法；在RWF‑2000/ MF的暴力动作分类上与监督基线相当（84.0%/92.7%），参数约70M，速度高达1900 FPS，优于大型视觉‑语言模型；

**⚠️ 局限性**

局限性：依赖骨架检测精度，复杂多人人场景下的实例区分仍有挑战，极端遮挡或低帧率情况下鲁棒性待进一步验证。

---

## 412. Tropospheric temperature and humidity profile retrieval from Meteosat Flexible Combined Imager based on deep learning

**arXiv ID:** 2608.25700 | [PDF](https://arxiv.org/pdf/2608.25700v1)

**作者:** Alejandro Salgueiro `[一作]` (Delft University of Technology), Angela Meyer `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

该研究构建了基于MTG‑FCI多光谱影像的全天空温湿度剖面检索模型，能够在无预报背景的情况下直接从卫星辐射恢复三维大气状态。

**💡 创新点**

创新点在于将空间感知的残差 U‑Net 与 SE 注意力、膨胀多尺度瓶颈相结合，实现了全光谱（可见、近红外、短波与红外）利用，并首次证明可见/近红外波段对低层湿度检索的显著贡献。

**🔧 技术方法**

使用的技术是残差 U‑Net 深度学习架构，配备 squeeze‑and‑excitation 通道注意力和多尺度膨胀卷积，并在 MSE 损失下训练，以实现像素级温湿度剖面输出。

**📊 数据集**

采用的数据集为 14 个月 MTG‑FCI 16 通道观测与 CERRA 5.5 km 级别重分析剖面作为目标，验证集则使用 IGRA 雷达气球的高质量观测。

**📈 对比分析**

通过与 ERA5 30 年气候、仅辅助变量模型、单像素 MLP（1x1‑Net）以及 CERRA 目标进行比较，U‑Net 在温度标准差约 1.5–1.9 K、相对湿度标准差约 12–20 % 的同时，显著优于前两者且接近 CERRA 的性能，且在云顶以下检索中仅略逊于 CERRA。

**⚠️ 局限性**

主要局限包括训练样本时长仅 14 个月，导致对极端天气或高层气象结构的覆盖不足；MSE 损失倾向于平滑细尺度特征，使得检索的空间锐度受限；并且缺乏独立高分辨率观测用于进一步验证细尺度准确性。

---

## 413. LMSM: LLM Security Framework Inspired by Linux Security Modules

**arXiv ID:** 2608.25697 | [PDF](https://arxiv.org/pdf/2608.25697v1)

**作者:** XiuYu Zhang `[一作]` (National University of Singapore), Zhenkai Liang `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于 Linux Security Modules 思想的 LLM 安全框架 LMSM，能够将解释性方法产生的模型内部证据统一绑定到安全后端、版本化策略与分离的执行门口，实现请求级的安全决策与输出发布。

**💡 创新点**

创新点在于将模型内部监测、策略评估与输出授权解耦为三层，并通过可插拔后端绑定、版本化规则库与请求键控状态，支持后端更换、策略调度与规则组合而不需重构推理栈。

**🔧 技术方法**

技术主要包括：Sparse AutoEncoder 与 Transcoder 作为模型内部证据后端、基于 vLLM 的连续批处理与 Torch 的 Transformers 参考路径、规则评估器的固定 OR 组合与时间调度、以及独立的执行门（gate）完成拒绝、终止与允许三种决策。

**📊 数据集**

使用的数据集包括 Qwen3‑4B 作为模型，HarmBench、WildJailbreak 用于攻击成功率评估，XSTest 用于误拒率评估。

**📈 对比分析**

通过与已知训练时安全技术（如 ThinkSafe）和无监控基线进行对比，实验表明在 32 并发宽度下，Checkpoint 策略将 HarmBench 的攻击成功率从 39.20% 降至 3.32%，WildJailbreak 从 41.90% 降至 7.35%，误拒率提升至 4.40%，并保持 98.14% 的吞吐量（与无监控基线几乎相同）。

**⚠️ 局限性**

局限性包括：当前仅支持单一后端绑定与三种固定动作（允许、终止、拒绝）；对不同模型/激活点的迁移需重新绑定与校准；并且性能开销主要集中在持续的后端推理与规则评估，尽管批处理可降低开销，但在高并发或更复杂策略下仍有提升空间。

---

## 414. CloSeR: Unified Relational Distillation from Closed-Set Teachers for Category Discovery

**arXiv ID:** 2608.25692 | [PDF](https://arxiv.org/pdf/2608.25692v1)

**作者:** Yuanpei Liu `[一作]` (University of Hong Kong), Kai Han `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

先通过在冻结的预训练 ViT 上插入轻量级 block‑wise adapter 对已标注的已知类别进行闭集迁移学习，得到域适应的闭集教师；随后在 GCD 训练阶段对学生模型进行统一关系蒸馏，将教师的全局样本‑原型关系与局部样本‑样本邻域关系分别对齐，并采用特征解耦降低优化干扰，实现 head‑agnostic 的正则化。

**💡 创新点**

提出两阶段闭集迁移+统一关系蒸馏框架，其中统一关系蒸馏同时传递全局（样本对原型）与局部（样本对样本）关系，且通过特征解耦避免单一表征同时满足两种关系的冲突，显著提升了泛化类别发现性能，并能兼容参数化与非参数化 GCD 方法。

**🔧 技术方法**

使用轻量化 block‑wise adapter、预训练 ViT (DINO/DINOv2)、全局与局部关系蒸馏（基于 KL 散度对齐）、特征解耦、以及传统的知识蒸馏与自监督对比学习技术。

**📊 数据集**

在 CIFAR‑10/100、ImageNet‑100、CUB、Stanford‑Cars、FGVC‑Aircraft 等六大基准数据集上进行实验。

**📈 对比分析**

与多种基线（SelEx、SimGCD 等）在相同 backbones 上对比，CloSeR 在所有旧类与新类的准确率均有提升，特别是细粒度数据集的 New 类准确率提升约 10–20%（平均提升 1–2% 的 All 类准确率），在 state‑of‑the‑art 位置实现了显著性能提升。

**⚠️ 局限性**

仍依赖高质量预训练模型，需额外两阶段训练过程，且对超参数（如关系温度、权重 α/β）敏感；在极大规模未标记数据或跨模态任务上的适用性尚未验证。

---

## 415. psRL: Efficient Training for Agentic AI via Training-Time Prefix Sharing

**arXiv ID:** 2608.25683 | [PDF](https://arxiv.org/pdf/2608.25683v1)

**作者:** Mianjie Yu `[一作]` (University of Macau), Chengzhong Xu `[通讯]` (University of Macau)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种针对现代agentic RL训练的Prefix Sharing训练系统，通过在训练阶段共享前缀计算来显著降低更新阶段的重复工作量。

**💡 创新点**

创新点在于：①引入跨批次与自序列两种前缀共享机制，支持灵活细粒度负载分配；②设计自适应块分配与动态块缓存的KV管理，精准匹配可重用前缀长度并实时回收；③结合全局可见性和数据不可变性实现层次化工作调度与token‑级微批处理。

**🔧 技术方法**

技术实现涵盖：基于veRL和Megatron‑LM的分布式训练框架；自定义前缀树解析、语义组划分与负载平衡算法；token‑级微批分配；自适应KV块分配与按引用计数的即时回收。

**📊 数据集**

使用工业生产中的真实轨迹数据：Search、WebShop、ALFWorld（基于Qwen2.5‑1.5/7B）以及大规模DTN Agent（Qwen3‑235B）。

**📈 对比分析**

与传统veRL、vLLM‑PS（16‑token块）和SGLang‑PS（1‑token块）三种基线对比，本文系统在所有任务上实现了1.2×–5.2×的吞吐量提升，并在DTN Agent Step模式下达到了239k tokens/s（≈3.8×SGLang‑PS），同时将峰值GPU内存从56.2GB降至33GB（约41%）。

**⚠️ 局限性**

局限性包括：①依赖完整的前缀树预处理，难以适应高度动态或在线生成的数据；②实现复杂度较高，调试和维护成本上升；③实验集中在agentic RL场景，尚未验证对其它LLM训练任务的通用性。

---

## 416. Adversarial Training of Linear Models under Stealthy Attacks

**arXiv ID:** 2608.25681 | [PDF](https://arxiv.org/pdf/2608.25681v1)

**作者:** Lovisa Eriksson `[一作]` (Uppsala University), André M. H. Teixeira `[通讯]` (Uppsala University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了一种基于攻击检测器的切换式线性模型训练框架，使模型在面对隐蔽攻击时保持鲁棒性，并给出了线性和逻辑回归的凸对抗风险表达式。

**💡 创新点**

创新点包括：① 通过将恢复模型与基准模型对齐，强制对抗攻击必须保持隐蔽；② 引入攻击概率超参数，显式权衡干净数据与攻击数据的性能；③ 为线性/逻辑回归给出了全局可求的凸风险表达式。

**🔧 技术方法**

采用对抗训练、异常检测器、线性/逻辑回归、凸优化（随机梯度下降）、保護特征与安全恢复模型等技术。

**📊 数据集**

实验使用了合成高斯数据（d=4，n=2000）和真实信用卡欺诈检测数据集（569k样本，21个未保护特征，1个保护特征）。

**📈 对比分析**

与“完全安全模型”（丢弃未保护特征）和“标准切换模型”对比。实验表明，在攻击概率>0时所提方法的风险低于标准模型，且在部分攻击概率下优于完全安全模型；在无攻击时与完全安全模型相同；在攻击概率误设时仍保持优越性能。

**⚠️ 局限性**

局限性：仅针对线性和逻辑回归；假设攻击者完全知晓检测器；需要手动设定误警率上界和攻击概率超参数；未验证在更复杂模型或深度网络上的可扩展性；对攻击策略的鲁棒性仍有限。

---

## 417. Toward Interpretable Privacy Guarantees in Face-Swapping Anonymization

**arXiv ID:** 2608.25750 | [PDF](https://arxiv.org/pdf/2608.25750v1)

**作者:** Vishnu Bondalakunta `[一作]` (Kansas State University), George Amariucai `[通讯]` (Kansas State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估并建模面部交换在隐私保护中的泄露机制，展示单次交换仍能泄露目标身份，并通过线性随机算子模型解释并预测多次交换的泄露衰减；

**💡 创新点**

创新点在于将面部交换器视为投影到身份嵌入空间的仿射随机算子，量化目标身份传递的Rayleigh增益和谱半径，从而得到可解释、可检验的泄露衰减速率与泄露底限；

**🔧 技术方法**

使用线性回归拟合仿射算子、谱半径分析、线性随机动态系统模拟、多次交换实验与ROC/TPR低FPR等指标；

**📊 数据集**

实验数据主要来自VGGFace2-HQ人脸库，使用InsightFace（ArcFace）和Facenet512两种嵌入器；

**📈 对比分析**

比较方法为对七款公开面部交换工具进行统一协议下的单次交换泄露评估（TPR@1%FPR、AUC、CMCs），以及对三款工具的多次交换实验验证模型预测；性能显示：单次交换后仍有30–61%身份被识别，三次交换后泄露率可下降但仍高于随机；

**⚠️ 局限性**

局限性包括模型仅解释60%嵌入方差（对BlendFace、CanonSwap不佳）、仅考虑身份推断未评估属性泄露、实验仅基于VGGFace2人脸集，未涵盖临床或监控图像、对更强攻击者的下限估计不足、且多次交换会导致图像质量下降与最终身份趋向最后 donor 的现象未量化。

---

## 418. An Oversubscription and Service Pricing Exploitation-Based Profit Maximization Framework for Industry Cloud Resource Management

**arXiv ID:** 2608.25712 | [PDF](https://arxiv.org/pdf/2608.25712v1)

**作者:** Deepika Saxena `[一作]` (University of Aizu), Ashutosh Kumar Singh `[通讯]` (Indian Institute of Information Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一个名为OP‑PMF的行业云资源管理框架，通过自适应集成学习预测虚拟机资源使用，结合模糊C‑均值聚类进行自动扩容，并利用延迟敏感模型（DSM）与最佳努力模型（BEM）进行请求分类，从而在保证服务质量的前提下最大化利润并降低能耗。

**💡 创新点**

核心创新包括①基于“无一适合全局”策略的自适应集成预测模型，可动态权重调节多基学习器输出；②将预测结果按资源相似度聚类并映射至不同VM类型，实现最优资源复用；③结合两种异构定价模型实现任务优先级调度，进一步提升收益与能源利用效率。

**🔧 技术方法**

技术方法主要有：自适应集成机器学习（线性回归、支持向量机、神经网络、随机森林等），模糊C‑均值聚类，在线预测与自适应重训练，虚拟机超额订阅策略，定价模型分类与优先级调度，以及基于功率模型的能耗评估。

**📊 数据集**

实验使用了Google Cluster Data（GCD）与PlanetLab VM traces（PL）两组真实工作负载数据，构建了约60%用户、随机生成请求与定价模型的仿真场景。

**📈 对比分析**

与SBA、OP‑MLB、BF、RF、LR‑P、THR‑P、FF等多种基线方法进行对比。OP‑PMF实现了约55%~56%的电费下降、约60%资源利用提升、约50%功耗降低，并在利润上较传统方案提升了约49%–51%。

**⚠️ 局限性**

局限性在于：实验中缺乏对虚拟机故障与恢复的完整处理，未验证在实际大规模云平台中的鲁棒性；使用的请求截止时间与提前执行时长为人工假设，真实场景下可能不同；超额订阅在极端负载下可能导致SLA违约；当前模型仅关注CPU与内存，未涉及存储与网络等资源。

---

## 419. Unsupervised Anatomical Feature Learning via Diffusion Models: Enhanced Medical Image Segmentation with Denoising Diffusion Probabilistic Models

**arXiv ID:** 2608.25693 | [PDF](https://arxiv.org/pdf/2608.25693v1)

**作者:** Akshat G `[一作]` (Manipal Academy of Higher Education), Tusar Kanti Mishra `[通讯]` (Manipal Academy of Higher Education)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

使用无监督扩散模型（DDPM）预训练编码器后，将权重迁移至U‑Net实现腹部CT器官分割

**💡 创新点**

通过扩散预训练使网络获得解剖结构先验，显著提升分割精度、边界准确性和低标注数据下的鲁棒性

**🔧 技术方法**

采用Denoising Diffusion Probabilistic Models结合U‑Net骨干进行无监督预训练，并用Dice损失微调分割头

**📊 数据集**

BTCV多器官腹部CT数据集（21名患者）

**📈 对比分析**

相较随机初始化U‑Net，Dice提升至0.93（肝）/0.95（肾）/0.95（多器官），HD95降低45–68%，低标注比例（10%）时仍保持约90%性能

**⚠️ 局限性**

仅验证大尺寸器官，使用2D切片缺乏3D上下文，跨数据集泛化和病灶等小尺寸结构的适用性尚未验证

---

## 420. Trust-Aware Sequential Decision Making and Rollout Planning for Resilient Multi-Robot Systems

**arXiv ID:** 2608.25690 | [PDF](https://arxiv.org/pdf/2608.25690v1)

**作者:** Roee M. Francos `[一作]` (Harvard University), Stephanie Gil `[通讯]` (Harvard University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种可信度感知的顺序决策框架，用于在多机器人系统中抵御定位欺骗并实现鲁棒的在线路由；框架将定位可信度与行为可信度融合，用以过滤掉可疑/恶意机器人，从而在路由规划（IA‑RA + rollout）中恢复规划-执行一致性。

**💡 创新点**

创新点包括：
- 监视者感知的距离约束欺骗模型与分层匹配策略，刻画了攻击者在可检测性与路由影响之间的权衡；
- 可信度感知的监控机制，将定位误差与任务执行结果相结合，动态更新机器人可信状态；
- 将可信度信息直接映射到规划状态（即删除恶意机器人），从而使 rollout 的look‑ahead 与实际执行保持一致，恢复其成本改进优势。

**🔧 技术方法**

主要技术手段：
- 基于真实 GPS 嗅探数据的贝塔分布定位可信度校准，随位置偏差连续调节；
- 行为可信度基于请求分配历史的 Beta 分布，给出成功/失效/重分配惩罚；
- 可信度融合规则（两条分支均达到阈值才判为合作），并用阈值驱动机器人排除；
- IA‑RA 作为基准策略；一阶 rollout（one‑at‑a‑time）与蒙特卡洛多场景采样用于评估候选动作；
- 统一的决策循环：观测→可信度更新→构造受信状态→规划→执行→新观测。

**📊 数据集**

使用的数据集：
- 两份真实 GPS 嗅探数据集（空中与地面机器人），用于定位可信度的校准；
- 旧金山出租车需求数据，用于生成路由任务并评估系统性能。

**📈 对比分析**

对比方法与性能：
- 对比 IA‑RA 与 rollout（有/无监控）以及监控后 IA‑RA；
- 评价指标包括未完成请求数、累计取消请求数、阶段成本（未完成+累计取消）；
- 结果显示：在未监控时，即使单个恶意机器人也能导致系统失稳；加入可信度监控后恶意机器人被识别并移除，未完成请求与取消请求显著下降；
- 在足够的 look‑ahead（H≥20）与额外的合作车队容量时，可信度感知 rollout 能恢复甚至超越基准 IA‑RA 的成本优势。

**⚠️ 局限性**

局限性：
- 攻击模型仅限于位置欺骗、无 Sybil/身份变化、完全不执行任务，未考虑部分合规或混合攻击；
- 定位可信度需要独立的定位完整性源，若该源失效或误差偏大会影响检测；
- 行为可信度假设所有请求失败均由恶意机器人导致，无法区分拥堵、硬件失效等非攻击原因；
- 一旦被判为恶意即永久移除，易导致可用资源减少；
- 结果基于仿真与经验评估，缺乏严格的理论稳定性与性能保证。

---

## 421. MoganBert-TR: A Turkish Encoder Foundation Model Trained from Scratch with a CLM-to-MLM Curriculum

**arXiv ID:** 2608.25768 | [PDF](https://arxiv.org/pdf/2608.25768v1)

**作者:** Furkan Yilmaz `[一作]`, Muhammed Faruk Gozay `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从零开始训练一套完整的土耳其语 Encoder 基础模型（MoganBert‑TR 149 M 参数）以及对应的嵌入模型（MoganBert‑Embed），并提供公开的数据处理管线、分词器、预训练目标与长上下文衰减策略。

**💡 创新点**

创新点：
① 采用两阶段预训练目标（先 CLM 再 MLM）在土耳其语上显著提升基于余弦相似度的检索性能（2.7–3.7×）并改善嵌入几何结构；
② 引入“分支衰减”设计，在同一预训练后期将长上下文与学习率衰减分为两条分支，提升 TrGLUE 平均分 0.49 点，成本仅比单一衰减低 4.3%；
③ 针对土耳其语构建语言专属质量过滤器与 50K 字符的 SentencePiece 词表，实现高压缩率与低 fertility；
④ 通过教师蒸馏与多信号对比微调，得到 51 倍参数压缩的嵌入模型，获得教师 99.5% 的表现。

**🔧 技术方法**

技术细节：
- ModernBERT 体系结构（RoPE、GLU、局部/全局注意力）；
- StableAdamW 优化器与 WSD 学习率调度；
- FlashAttention‑3 与动态长度支持；
- 预训练目标：CLM 25% + MLM 75%；
- 词表构建与 MinHash 去重；
- 基于 fastText 的土耳其语质量分类器；
- 视觉‑语言模型识别印刷文本前置信息；
- 蒸馏 + GOR 正则化与 InfoNCE 对比微调；
- 模型 soup 加权平均。

**📊 数据集**

使用的数据集：
- FineWeb2（土耳其语子集）+ 最近 CommonCrawl 采集；
- 领域密集的书籍、论文、法律文件；
- 土耳其维基百科、FLORES‑200；
- TR‑MMLU、Turkish MS MARCO、TrGLUE、TabiBench、MTEB(Turkish)；
- 教师模型 Qwen3‑Embedding‑8B。

**📈 对比分析**

比较方式与性能：
- TrGLUE（五种种子）平均 78.41，显著高于 BERTurk、TabiBERT 与 ModernBERT‑TR；
- TabiBench 总分 77.73，位于土耳其单语模型前列；
- MS MARCO 检索性能提升 2.7–3.7×，归因于嵌入几何改善；
- MTEB(Turkish) 总分 68.30，排名所有学生模型第一，接近 7.57 B 参数教师模型；
- 通过对比模型 soup 与分支衰减，后者在 TrGLUE 上获得 +0.75 点、成本仅 4.3%。

**⚠️ 局限性**

局限性：
- 预训练目标 ablation 仅单种子、10k 步；全规模验证缺失；
- 16.6% CLM 比例未在 ablation 中验证；
- 模型 soup 组件单种子，可能存在过拟合；
- TabiBench 评测仅单种子，且参考模型与本实验环境不同；
- 部分参考模型仅在 512 token 评估，影响对长文本的对比；
- 仅使用单一教师模型；
- 情感对比数据来自 LLM，未人工校验；
- 数据集完整性受许可限制，无法公开完整语料。

---

## 422. InteractGesture: Progressive Chunk Guidance for Continuous Streaming Co-Speech Gesture Control

**arXiv ID:** 2608.25734 | [PDF](https://arxiv.org/pdf/2608.25734v1)

**作者:** Ekkasit Pinyoanuntapong `[一作]` (Meta), Jie Shen `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种模型无关的推理时空间控制框架InteractGesture，利用已训练的共语手势生成器，在采样过程中对目标潜在向量进行梯度优化，从而实现对多关节3D空间位置的精确控制。

**💡 创新点**

创新点在于：①在推理阶段直接对潜在向量进行优化，而非训练时加入控制模块；②提出Progressive Chunk Guidance策略，使得在实时流式生成中，后续块的空间约束能够向前传播，从而解决块间边界不连贯问题；③将控制通用化到绝对位置、轨迹和指向等多种编辑模式。

**🔧 技术方法**

主要技术包括：扩散式采样（DDIM）与逆向潜在空间优化、可微分RVQ‑VAE解码器、SMPL‑X正向运动恢复、Adam优化、动态控制时间表与梯度归一化，以及三种块级控制调度（Sequential、Synchronous、Progressive）。

**📊 数据集**

使用BEAT2语音-手势数据集进行评估，采用预训练的GestureLSM生成器。

**📈 对比分析**

与基线（逆向运动学、Chunk‑Wise ControlNet）以及三种块级调度比较，Progressive Chunk Guidance在保证流式实时性的同时，平均控制误差约6.3 cm，FGD为0.431，保持与同步模式（4.67 cm、0.442）相近的控制精度，同时比顺序方式显著提升了轨迹与位置准确率。

**⚠️ 局限性**

局限性包括：①需要针对不同控制密度调节指导尺度，参数调优仍需经验；②在极高约束密度或长序列下，梯度传递可能出现衰减；③实时流式生成受限于滑动窗口大小和延迟设置，尚未实现完全无延迟的高频控制。

---

## 423. When RAG Fails to Equalize: Geo-bias in Factual Question Answering over Public Companies

**arXiv ID:** 2608.25717 | [PDF](https://arxiv.org/pdf/2608.25717v1)

**作者:** Abhinav Havaldar `[一作]` (Bloomberg), Enrico Santus `[通讯]` (Bloomberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个涵盖全球15个指数、约2,000家上市公司的事实问答基准，并在无上下文、完美上下文、误导上下文和干扰上下文四种检索条件下评估六个大型语言模型。

**💡 创新点**

提出了将参数知识与检索证据分离的对照实验框架，揭示检索效果与模型的基线知识高度相关，而非能普遍纠正知识缺失。

**🔧 技术方法**

采用检索增强生成（RAG）与多选问答技术，并对四种上下文策略进行系统评估，结合统计建模分析结果。

**📊 数据集**

使用维基百科的公司信息，构建约2,135家企业的实体与属性数据，覆盖15个全球股指（北美、欧洲、亚洲、拉美、非洲、澳洲）。

**📈 对比分析**

与GPT‑5系列、Claude Sonnet、LLaMA‑70B/8B等六大模型对比，发现检索提升准确率但未消除地区差距，误导上下文会导致模型直接复制错误信息。

**⚠️ 局限性**

局限在于数据仅来自维基百科且以英文为主，误导上下文为人工合成，实验仅针对原子事实，未涵盖多语言、多跳推理或真实检索环境。

---

## 424. Comparing Corrupted Constrained Learning Problems

**arXiv ID:** 2608.25745 | [PDF](https://arxiv.org/pdf/2608.25745v1)

**作者:** Laura Iacovissi `[一作]` (University of Tübingen), Robert C. Williamson `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

论文提出了在受限贝叶斯风险（仅考虑给定模型类）下的泛化数据处理不等式（GDPI），首先给出了经典数据处理不等式在受限情况下的反例，随后用受限超预测集（constrained superprediction set）给出了GDPI的几何表述，并给出若干关于标签/属性噪声的充分条件，使得GDPI得以保证。

**💡 创新点**

创新点包括：
1) 将经典的Blackwell/数据处理不等式推广到受限贝叶斯风险；
2) 引入受限超预测集并证明其与贝叶斯风险的支撑函数等价；
3) 将GDPI的成立归结为超预测集的包含关系，并利用此表述得到一系列可验证的充分条件；
4) 对标签、属性以及双随机噪声（bistochastic）等多种噪声模型给出了统一的理论框架。

**🔧 技术方法**

核心技术：
- Markov核（kernel）和其对应的转换、算子与伴随算子；
- 凸分析中的支撑函数与极点概念；
- 受限超预测集的定义与闭包性质；
- 对称性、可变换不变性等模型类与损失函数的结构假设；
- 对比理论（Blackwell–Sherman–Stein）与Birkhoff–von Neumann定理的推广。

**📊 数据集**

无实验数据集，论文完全为理论证明与数理推导。

**📈 对比分析**

比较方式：通过超预测集包含关系或支撑函数的支配关系来判断GDPI是否成立；若满足条件则证明在任意联合分布上受限贝叶斯风险在随机化后不会下降；在满足充分条件的噪声模型下，理论表明GDPI恒成立。

**⚠️ 局限性**

局限性：
- 只给出了充分条件，缺乏必要条件，无法完全刻画GDPI的判定标准；
- 证明依赖于超预测集的闭包与凸性，实际模型可能不满足这些假设；
- 仅考虑了一步噪声（one‑step corruption），多步或更复杂的随机化过程未被覆盖；
- 论文未给出实验验证，理论结果的实际应用效果尚需进一步评估。

---

## 425. MeMark: Membrane-Space Watermarking for Spiking Neural Networks

**arXiv ID:** 2608.25738 | [PDF](https://arxiv.org/pdf/2608.25738v1)

**作者:** Roberto Riaño `[一作]` (Radboud University), Aitor Urbieta `[通讯]` (IKERLAN Technology Research Centre)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的SNN水印方法MeMark，在Leaky Integrate-and-Fire神经元的膜电位中嵌入多比特密钥，并通过阈值读出，实现对预训练检查点的所有权验证。

**💡 创新点**

创新点在于：①不依赖外部解码器，直接利用神经元自身的发火阈值读取水印；②将水印嵌入内部膜电位而非输出层，使得即使替换输出头仍能保持可验证性；③通过预先提交的时间戳承诺解决“拟合攻击”导致的伪所有权问题。

**🔧 技术方法**

技术细节包括：LIF神经元膜电位距离阈值的连续损失（μ‑wm），结合任务损失与KL正则保持原任务性能；使用随机密钥、随机挑选的神经元坐标和挑战输入；白盒验证读取膜电位与阈值；可选的输出水印以兼顾黑盒查询；对抗性攻击实验（fine‑tune、剪枝、量化、结构重排、回滚等）。

**📊 数据集**

评估数据集涵盖：文本任务使用enwik8与OpenWebText（SpikeGPT 215M）；视觉任务使用MNIST、CIFAR‑10、CIFAR10‑DVS、N‑Caltech101；模型结构包括循环SNN、卷积SNN、残差SNN、Transformer‑SNN（Spikformer、QKFormer）。

**📈 对比分析**

与DICTION、DeepSigns、Uchida、Poursiami、SpikeTimer等基线对比，MeMark在20个64‑bit密钥下均满足固定的51/64阈值，误报率为0%，在Fine‑tune、90%剪枝、INT8量化、输出头重置等改动后仍保持>90%通过率；与DICTION相比，MeMark无需额外的投影权重，误报率更低，离线验证成本更低；整体保持任务性能差异≤1.5%。

**⚠️ 局限性**

局限性包括：需要白盒访问以读取膜电位；若攻击者获得精确的未加水印前检查点，可通过回滚攻击显著削弱水印；对新学生模型（仅通过输出学习）不具备继承性；在结构重排后需要额外对齐步骤；对密钥泄露仍需额外承诺来避免伪所有权。

---

## 426. Moving Beyond More Views: Redundancy-Aware Ego-Exo Fusion for Proficiency Estimation

**arXiv ID:** 2608.25736 | [PDF](https://arxiv.org/pdf/2608.25736v1)

**作者:** Xu Dong `[一作]` (University of Surrey), Andrew Gilbert `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了适应性多视角选择（AdaMVS）和变分信息瓶颈梯度混合（VIB-GB）两大模块，改进了 Ego–Exo proficiency estimation 中的多视角融合。

**💡 创新点**

创新点在于同时从数据层和特征层两侧解决冗余与过拟合：AdaMVS 通过弱监督的视角评分动态选择重要视角，VIB-GB 通过梯度混合与信息瓶颈压缩特征，二者协同提升鲁棒性。

**🔧 技术方法**

使用 Transformer‑based 视觉编码、Gumbel‑Softmax 视角采样、Variational Information Bottleneck、Gradient Blending、CORN 序数损失以及多头注意力融合。

**📊 数据集**

实验基于 Ego‑Exo4D（四视角）和 EgoExo‑Fitness（多任务）两大多视角数据集进行评估。

**📈 对比分析**

与 TimeSformer、SkillFormer 等现有基线对比，在 Ego‑Exo4D 上取得 53.0% 的最高准确率（比前沿 48.3% 提升 4.7%），在 EgoExo‑Fitness 上提升 7.1%，且 AdaMVS‑Small 在参数/计算量上也有显著压缩。

**⚠️ 局限性**

局限在于评估注释主观性、数据规模有限，且方法主要针对 Ego–Exo 任务，未来需要扩展到更广泛的多视角/多模态任务并提升数据量。

---

## 427. It's a matter of timescale: non-linear utility in successor features and multi-objective planning and learning

**arXiv ID:** 2608.25723 | [PDF](https://arxiv.org/pdf/2608.25723v1)

**作者:** Liam P. H. Mertens `[一作]`, Peter Vamplew `[通讯]` (Federation University Australia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出在多目标强化学习中同时考虑不同时间尺度下的非线性效用，并通过毒理学示例和自定义联合目标演化策略验证其必要性。

**💡 创新点**

创新点在于：①指出现有 SER、ESR 与 RSR 仅针对单一时间尺度，忽略同一决策问题中多尺度效用的交互；②设计了联合效用函数与自然进化策略 MONES，用以一次性优化三种时间尺度的效用。

**🔧 技术方法**

使用了多目标强化学习的标准算法（EUPG、NLPPO、SFDQN）以及基于自然进化策略的 MONES；同时构造了三种效用函数（急性、亚急性/慢性、组合）并对其进行了求解。

**📊 数据集**

采用了基于毒物暴露的人工合成数据集：三名员工、三项任务，包含毒性与技能水平信息，生成每日剂量噪声的30天工作日历。

**📈 对比分析**

通过在相同实验设置下对 SER、ESR、RSR 与 MONES 四个方法进行 50 次策略评估，比较返回分布和毒剂量阈值满足率，结果显示单一时间尺度优化往往导致某些员工被牺牲，而联合目标能够平衡多尺度效用，但在实际安全性上仍需进一步改进。

**⚠️ 局限性**

局限性包括：①实验仅基于极简化的合成数据，缺乏真实工业环境验证；②联合目标方法 MONES 计算开销大，收敛速度慢；③未给出理论证明联合优化下的 Pareto 最优性或可行性保证。

---

## 428. Quantitative tiling stability from quadratic discrepancy in Hamming spaces

**arXiv ID:** 2608.25716 | [PDF](https://arxiv.org/pdf/2608.25716v1)

**作者:** Valery `[一作]`, Aryeh Lev Zabokritskiy `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究有限Hamming空间中代码的二次球差异能量，并证明在完美码参数下，任何代码的总差异能量低于完美码基准的多余量能量能够量化控制球覆盖的缺陷；进一步给出对单误差和双误差参数的明确常数下的稳定性不等式，并在所有非平凡完美码（包括重复码、三元Golay码、二元Golay码等）上实现完全最优性。

**💡 创新点**

创新点在于提供了可实现的、与参数无关的下界常数（即“tiling‑defect stability”），使得差异能量的超额量直接支配球覆盖缺陷、重叠、空洞、以及在球噪声下的平滑误差；同时首次在不假设完美码存在的前提下给出双误差条件下的稳定性系数，并在所有可能的参数集上给出精确数值。

**🔧 技术方法**

采用了Hamming空间的傅里叶–Krawtchouk理论，利用谱和矩身份、Lloyd多项式的根结构以及完全单调性/凸性技术，构造了严格的能量下界；结合了离散正交多项式的差分展开、两点积分法以及组合数理分析。

**📊 数据集**

论文不依赖任何具体数据集，而是针对所有满足算术条件的有限域或群的Hamming空间参数（长度n、字母数q、码字数N）进行理论分析与数值计算；在具体的3-ary Golay、2-ary Golay及重复码等实例上给出了精确数值验证。

**📈 对比分析**

通过比较总差异能量与基准差异能量的差值，给出了对球覆盖缺陷（Φ_e）的下界系数σ_n,q,e，并证明在完美码存在时该差值为0；当不完美时差值至少为4σ_n,q,e/N^2。实验验证表明系数在特定参数下可大于1，说明稳定性不等式比保守的统一下限更紧凑。

**⚠️ 局限性**

局限性在于对双误差的稳定性系数仍依赖于Lloyd多项式的整数根条件；目前对非原子字母数的双误差完美码尚无存在性结论，导致系数在这类参数下仅是理论上可行而未被验证；此外，稳定性常数并非已知最优值，进一步优化仍是开放问题。

---

## 429. Precipitation Downscaling Using Foundation Model-Conditioned Diffusion

**arXiv ID:** 2608.25858 | [PDF](https://arxiv.org/pdf/2608.25858v1)

**作者:** Victor Nascimento Ribeiro `[一作]` (IBM Research), Anne Jones `[通讯]` (IBM Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并比较了三种条件化策略（通道拼接、交叉注意力+可学习卷积编码器、交叉注意力+冻结的Prithvi WxC天气基础模型编码器）在使用扩散模型进行高分辨率降水下尺度中的表现。

**💡 创新点**

首次在降水下尺度任务中系统评估交叉注意力条件化；将预训练的天气基础模型作为条件编码器；通过训练数据量和计算成本对比评估其数据效率与泛化能力。

**🔧 技术方法**

使用EDM扩散模型与UNet去噪架构；交叉注意力模块；Prithvi WxC Transformer编码器；log‑uniform噪声采样与20步采样；多尺度、极端事件和分布评估指标。

**📊 数据集**

使用ERA5‑Land 0.1°每日降水作为目标；ERA5 0.25°动态气象变量；高分辨率静态变量（湖盖、陆海掩模、DEM等）；时间范围1985–2015，测试期2013–2015。

**📈 对比分析**

在统一的训练、推理和评估框架下，对四种模型使用CRPS、MSE、分布偏差、RAPSD、RALSD、FSS、极端事件保留率等指标进行对比。拼接模型在像素CRPS上表现最佳，但在分布与极端事件方面交叉注意力模型更优；Prithvi WxC在训练数据较少时显示更好的数据效率。

**⚠️ 局限性**

仅在科罗拉多河流域单变量进行，结果可能不适用于其他区域或变量；依赖ERA5数据，可能无法代表GCM预测；极端事件评估样本有限；对未来气候情景的迁移性未知；像素级精度上交叉注意力模型略逊于拼接模型。

---

## 430. Skill Issue: Are Skills Language-Invariant in LLMs?

**arXiv ID:** 2608.25832 | [PDF](https://arxiv.org/pdf/2608.25832v1)

**作者:** Bobby Cheng `[一作]` (A*STAR), Leshem Choshen `[通讯]` (Weizmann Institute of Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在多语言文本游戏环境 TextArena 中让同一模型以不同语言接口进行自对弈，系统评估了大型语言模型在不同语言下的技能表现差异。

**💡 创新点**

创新点在于：①提出跨语言技能不一致的评估范式，②构建 193 种语言版本的 TextArena 并公开；③通过语言接口与推理语言分离，揭示语言对状态解释、推理、知识检索和动作选择的多阶段影响。

**🔧 技术方法**

使用的技术包括：多语言翻译与人工校验流水线、双实例自对弈框架、角色合并胜率度量（role‑pooled win‑loss margin）、语言优势恢复实验（切换推理语言）以及对比静态多语言基准（Belebele、Global‑MMLU）和网络文本量的相关分析。

**📊 数据集**

数据集主要包括：①手工验证的 8 种 Tier‑A 语言（英、阿、德、西、法、希、马、汉）中的 6 款 TextArena 游戏；②全 193 种语言的 TextArena 扩展版本；③公开基准数据集 Belebele、Global‑MMLU 与 FineWeb‑2 语言词量统计。

**📈 对比分析**

评估方法是对每个模型（Gemma‑4‑E4B‑it、Qwen3‑4B、Ministral3‑3B）在 6 款游戏中与 8 语言两两自对弈 400 场（共 518,400 场），计算语言对比胜率、均值差距，并与静态基准和网络文本量做相关性分析；结果显示英语普遍最强，希伯来最弱，语言间差距可达 0.5 以上，且部分游戏可通过在强语言中推理显著恢复性能。

**⚠️ 局限性**

局限性包括：①使用的模型训练数据闭源，难以深入解释语言差异背后的原因；②实验仅覆盖 3B–4B 参数规模的模型，跨语言技能不一致在更大模型上的表现仍未知。

---

## 431. Localize-Then-Decide Guarantees for LLM Judgments

**arXiv ID:** 2608.25824 | [PDF](https://arxiv.org/pdf/2608.25824v1)

**作者:** Xinyu Li `[一作]` (University of Exeter), Gaojie Jin `[通讯]` (University of Macau)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段评估框架“Localize‑Then‑Decide”，先用一致性预测（conformal prediction）把多候选答案缩小为一个小列表，再用校准的置信度阈值从列表中挑选单个答案或弃权，从而为LLM评估器在多候选场景下提供高概率的人类一致性保证。

**💡 创新点**

创新点在于：① 将原本仅适用于两候选的置信度阈值保证扩展到多候选（m>2）；② 通过先局部化候选集来恢复置信度与一致性风险的单调关系；③ 在两阶段中使用固定序列检验与置信区间上界，实现在有限样本下的分布无关保障。

**🔧 技术方法**

技术方法包括：一致性预测（conformal prediction）用于定位包含人类首选答案的列表；置信度分数（如EMP、KL‑margin）与基于边际的置信度计算；固定序列检验与二项分布上界的置信度阈值校准；以及多模型级联架构的设计。

**📊 数据集**

使用四个公开评估基准：TL;DR（摘要），Chatbot Arena（聊天），HH‑RLHF（有用性/安全性），AlpacaEval（指令跟随），在每个基准上构造5/10/20个候选答案，测试多种LLM（7B‑120B）作为评判器。

**📈 对比分析**

与单阶段（直接从m候选中挑选）方法相比，Two‑Stage框架在覆盖率和保证成功率（GSR）上显著提升：覆盖率提升约10‑20%，GSR在95‑97%之间，单阶段方法在相同设置下往往低于70%。在多模型级联中，Two‑Stage亦能保持90%以上的GSR并显著减少高端模型的调用比例。

**⚠️ 局限性**

局限性包括：① 需要校准与测试样本可交换（exchangeable）以保证理论边界；② 置信度估计使用Simulated Annotators，导致每条实例需多次前向推理，计算成本较高；③ 该框架仅针对单一最佳答案的选择任务，对Likert量表或事实性验证等其它评估范式需进一步改造。

---

## 432. Canalization Before Generalization: Grokking as a Dynamical Probe

**arXiv ID:** 2608.25813 | [PDF](https://arxiv.org/pdf/2608.25813v1)

**作者:** Yiming Lin `[一作]` `[通讯]` (University of Chinese Academy of Sciences), Yiming Lin (University of Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用短期权重衰减脉冲干预在过参数化网络的grokking平坦期内进行动力学探测，并绘制脉冲对泛化时间的影响图。

**💡 创新点**

首次揭示在grokking阶段的局部干预会产生可排序的泛化时间位移，并伴随测试损失障碍的坍塌，表明功能选择过程呈现“管化”特征。

**🔧 技术方法**

采用AdamW训练、权重衰减脉冲、WD-响应图、线性模式连通性、测试损失障碍测量、PCA投影等技术。

**📊 数据集**

三种算法任务：稀疏偶数匹配、模块加法和基础偶数匹配，均为小规模离散数据集。

**📈 对比分析**

通过与基线训练轨迹比较，计算泛化时间位移ΔT和损失障碍；结果显示脉冲强度与泛化时间呈线性正负关系，障碍在后期趋于零，验证了脉冲可调性而不改变最终函数。

**⚠️ 局限性**

仅在三类简单任务中验证，缺乏对大规模、连续任务的泛化；需要长且明显的预泛化平坦期，且未探讨不同模型架构和超参数的稳健性。

---

## 433. Beyond Minimum Distance: The Optimal Leading Coefficient in the High-SNR Error-Probability Expansion for AWGN Spherical Codes

**arXiv ID:** 2608.25805 | [PDF](https://arxiv.org/pdf/2608.25805v1)

**作者:** Nikola Zlatanov `[一作]` `[通讯]` (Innopolis University), Nikola Zlatanov (Innopolis University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文研究了在高信噪比（SNR）下，优化球形编码的最小错误概率，提出了一种基于SNR的编码优化方法，能够实现比任何固定的最佳打包编码更小的错误概率。

**💡 创新点**

创新点在于提出了一种SNR依赖的优化方法，通过对编码字位置的重新优化，能够在高SNR下实现更小的错误概率和更优的领先系数。

**🔧 技术方法**

使用了高斯软打包能量的理论框架，结合最大似然解码和球形打包的概念，分析了不同编码的性能。

**📊 数据集**

研究中使用了多种球形编码，包括正则单纯形和交叉多面体等，具体数据集未明确给出，但涉及到的编码结构和参数均为理论推导。

**📈 对比分析**

通过与固定的最佳打包编码进行比较，证明了SNR优化的编码在高SNR下的错误概率严格小于最佳打包基准，且其领先系数小于固定编码的领先系数。

**⚠️ 局限性**

限制在于该方法依赖于SNR的变化，可能在某些情况下无法保证在所有SNR下都能达到最优性能，且对编码字的选择和优化过程的复杂性提出了挑战。

---

## 434. LocalLSTC: A Long Short-Term Control Architecture for Locally Deployed GUI Agents

**arXiv ID:** 2608.25777 | [PDF](https://arxiv.org/pdf/2608.25777v1)

**作者:** Weiming Li `[一作]` (University of New South Wales), Yulei Sui `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LocalLSTC 框架，将跨步骤控制信息显式拆分为长期控制状态和短期执行承诺，并在本地推理下实现无训练的 GUI 任务执行。

**💡 创新点**

通过时间范围分离控制信息，实现了本地推理时跨步控制的显式管理，显著降低控制失败并提升任务成功率。

**🔧 技术方法**

采用 Qwen3.5/3.6 作为规划器、GTA1 作为视觉基础，结合 L2S–S2L 循环、Step Abstraction、Final Verification 与 Context Refinement 等技术，构成无训练的本地推理架构。

**📊 数据集**

在 OSWorld（369 Linux 任务）和 WindowsAgentArena（154 Windows 任务）两大桌面任务基准上进行评估。

**📈 对比分析**

与四个主流本地框架和多种 API 结果对比，LocalLSTC 在 OSWorld SR‑100 达到 64.7%、WindowsAgentArena 65.3%，比现有本地结果提升 10–18 个百分点，几乎与最强 API 结果持平。

**⚠️ 局限性**

仍依赖较大规模模型（如 27B Qwen），对更大模型和跨语言/环境的通用性验证不足；恢复策略细粒度控制仍有改进空间，且实验中控制失败标注仍需人工辅助。

---

## 435. Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation Independence

**arXiv ID:** 2608.25869 | [PDF](https://arxiv.org/pdf/2608.25869v1)

**作者:** Ante Kapetanovic `[一作]` (Infobip), Emanuel Lacic `[通讯]` (Infobip)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM-as-a-Judge系统在评估时是否受先前分数的锚定偏差影响，实验了三种提示条件并测量评分与接受率的变化。

**💡 创新点**

发现先前评分在即使为随机且低于阈值时也能显著拉低后续评分，揭示了“锚定”偏差在LLM评估中的普遍性和阈值式响应模式。

**🔧 技术方法**

使用大型语言模型（GPT‑4.1、Claude‑4.5‑Sonnet、DeepSeek‑R1、Llama 3系列、Qwen2.5‑7B、Gemma‑2‑9B）与统一的LiteLLM代理，进行任务级别分层的Bootstrap统计及token‑级别概率分析。

**📊 数据集**

采用20个人工筛选的固定答案（涵盖摘要、代码评审、创意写作、问答四类），以及一份行业合规分类数据（441条样本），并对比无锚定、修订提示和完整锚定三种条件。

**📈 对比分析**

通过任务级别Bootstrap和Cohen’s d评估，七/八个模型显示平均评分下降0.02–0.71，接受率下降最多达22个百分点；在行业数据中，锚定导致错误纠正率下降48%，正确率下降约8个百分点，提示策略无法显著降低总偏差。

**⚠️ 局限性**

限制包括仅测试20个人工构造答案、锚定值仅在[0,3.99)范围、未拆分单独数值与其他元数据、仅针对特定模型与任务、以及行业验证仅覆盖单一合规场景。

---

## 436. Unlocking Multimodal Protein Language Models at Inference Time

**arXiv ID:** 2608.25855 | [PDF](https://arxiv.org/pdf/2608.25855v1)

**作者:** Yi Zhou `[一作]` (Hong Kong Polytechnic University), Wenqi Fan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 437. Learning Late, Guiding Early: Timestep-Decoupled Semantic Guidance for Fair Face Generation

**arXiv ID:** 2608.25862 | [PDF](https://arxiv.org/pdf/2608.25862v1)

**作者:** Subir Kumar Parida `[一作]` (Bhabha Atomic Research Centre), Swati Hiremath `[通讯]` (Bhabha Atomic Research Centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Semantic Boundary Predictor (SBP)，通过一次性在逆向扩散的起始噪声潜在空间中施加从后期潜在空间学到的线性语义边界，实现无重训练的公平面部图像生成。

**💡 创新点**

创新点在于时间步解耦：在后期潜在空间学习语义边界，随后仅在最早潜在空间一次性应用，既保证了边界的判别性，又避免了多步引导的计算开销。

**🔧 技术方法**

采用线性分类器学习潜在空间语义边界、PCA 子空间压缩、Latent Diffusion Model (LDM) 的逆向扩散、DDIM 采样以及可调节的指导强度 δ。

**📊 数据集**

使用 CelebA-HQ、FFHQ 作为生成数据集，并用 FairFace 进行属性（性别、种族、年龄）分类。

**📈 对比分析**

与 Gaussian Harmony、Unbiased‑Diff、Balancing Act 以及原始 LDM 进行对比；在性别、二元/四类种族、年龄等任务上，Fairness Discrepancy 均降低 95% 以上，FID 仅略微提升，采样吞吐率与原始模型相近。

**⚠️ 局限性**

局限性包括：每个属性需要单独训练 SBP，受属性分类器误差影响；仅在起始与终止两端使用，未探究中间时间步；对更细粒度语义的适用性尚待验证。

---

## 438. GenAIT: Development and Validation of an Objective Generative AI Literacy Test for High School Students

**arXiv ID:** 2608.25815 | [PDF](https://arxiv.org/pdf/2608.25815v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 439. SlimTCP: It's fast, but not because it's slim

**arXiv ID:** 2608.25834 | [PDF](https://arxiv.org/pdf/2608.25834v1)

**作者:** Mihai Drosi Caju `[一作]`, Costin Raiciu `[通讯]` (Politehnica University of Bucharest)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文设计并实现了简化版 TCP/IP 栈 SlimTCP，针对数据中心网络中的可靠顺序链路（Ultra Ethernet）进行性能优化。

**💡 创新点**

通过删除所有与网络不可靠性相关的 TCP 扩展（如 SACK、PAWS 等），并引入零拷贝用户空间接口，显著降低代码复杂度与缓存失效，提升单连接吞吐量。

**🔧 技术方法**

技术上采用 DPDK 驱动的用户空间实现、零拷贝 pbuf API、单生产者单消费者环形缓冲区、XXH32 哈希、禁用 TSO/LRO/RSS、POSIX 兼容 API 以及 epoll 等。

**📊 数据集**

使用裸机测试平台（Intel Xeon E5‑2670 v2 + Broadcom StingRay PS225 NIC）进行吞吐率与连接数实验，比较 SlimTCP、F‑Stack 与 mTCP。

**📈 对比分析**

在相同硬件、相同编译优化、相同 DPDK 版本下，对不同 MSS 与连接数进行包率/好吞吐测量；SlimTCP 在单连接 MSS≥1024 时实现约 23.6 Gb/s 的 goodput，整体可扩展性最佳。

**⚠️ 局限性**

局限性包括缺乏拥塞控制、窗口缩放等功能，单线程发送导致小 MSS 下频繁触发 RTO，内存占用高（约 2 GiB），以及在生产环境中的兼容性与成熟度不足。

---

## 440. Simultaneous Digital Communication and Deformation Sensing over a Single Stretchable Interconnect

**arXiv ID:** 2608.25801 | [PDF](https://arxiv.org/pdf/2608.25801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 441. Unfolding Scientific Papers into Multi-Turn Generation Trajectories for Continued Pre-Training

**arXiv ID:** 2608.25826 | [PDF](https://arxiv.org/pdf/2608.25826v1)

**作者:** Qiankai Xu `[一作]` (ByteDance Seed), Ge Zhang `[通讯]` (ByteDance Seed)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

建立了一套将完整科学论文拆解为多轮生成轨迹的流程，重建写作请求、全局计划以及章节前的思考。

**💡 创新点**

创新点在于逆向构造方法：保持原始文本不变，回溯生成作者的写作过程；并将该方法扩展为大规模 CPT、SFT 与 PAW‑Bench 数据集，验证其对写作、推理和长文本理解的提升。

**🔧 技术方法**

技术层面采用 Qwen3.5-4B/9B/27B 生成逆向轨迹，对 1.8M arXiv 论文进行清洗与分割，随后通过 CPT+SFT 训练和 GPT‑5.5 评测。

**📊 数据集**

数据集包括 2006‑2026 年的 1.8M arXiv 论文（生成 60B‑token CPT 语料）、200K 样本 SFT 数据、2940 任务的 PAW‑Bench，以及对照的 FineWeb‑Edu 和原始论文文本。

**📈 对比分析**

与 FineWeb‑Edu、Plain‑Paper CPT 以及不同 LLM 大小基线比较，使用 WritingBench、PAW‑Bench、HelloBench、LongBench‑Write、MMLU、GPQA、MATH 等基准；CPT 训练后写作平均分提升约 2–4 分，PAW‑Bench 最多提升 4 分；推理性能保持不变，长文本理解提升 3–6 分。

**⚠️ 局限性**

局限性：实验仅覆盖单一基础模型与固定比例；写作评测依赖 LLM 判别，缺乏大规模人工评测；长文本与推理基准规模较小；生成轨迹为后验推断，可能不完全符合真实作者思考。

---

## 442. Open-Source 5G RAN Platforms: A Dual Perspective on Performance and Capabilities

**arXiv ID:** 2608.25820 | [PDF](https://arxiv.org/pdf/2608.25820v1)

**作者:** Maria Katarine Santana Barbosa `[一作]` (Universidade Federal de Pernambuco), Kelvin Lopes Dias `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建并对比评估了OpenAirInterface（OAI）与srsRAN两大开源5G RAN平台的功能与性能，涵盖RRC设置时间、理论吞吐量、以及在真实应用（VoD、直播、云游戏）下的网络表现；

**💡 创新点**

首次在统一实验平台上同时使用两平台和多种SDR（B210、N310）进行系统性对比，验证不同配置对RRC、吞吐量及QoS指标的影响；

**🔧 技术方法**

采用SDR硬件、Docker容器化的Open5GS核心网、iPerf3、Owncast、Moonlight等开源工具与自制服务器实现完整5G原型；

**📊 数据集**

利用真实用户设备（4台Moto Edge 20）与自制服务器数据，测量RRC时间、吞吐量、延迟与抖动，形成对比数据集；

**📈 对比分析**

通过平均RRC设置时间、实际吞吐量与理论吞吐量比、延迟/抖动指标进行比较，结果显示srsRAN在大多数场景（尤其是下行吞吐量与实时应用）表现更佳，而OAI在RRC设置时间上略占优势；

**⚠️ 局限性**

实验受限于较低频宽（≤40 MHz）、MIMO配置稳定性不足、单一服务器CPU与内存资源、未评估计算成本和大规模部署场景等因素。

---

## 443. Label-Free Foundational Model Selection for Medical Image Classification under Distribution Shift via Pseudo Label Discrepancy

**arXiv ID:** 2608.25810 | [PDF](https://arxiv.org/pdf/2608.25810v1)

**作者:** Juan Iñaki Larrea `[一作]` (Universidad de Buenos Aires), Enzo Ferrante `[通讯]` (Universidad de Buenos Aires)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于SUDO的无标签模型选择方法（AURCC），用于在跨机构分布偏移下为医学影像任务挑选最优基础模型。

**💡 创新点**

创新点在于将SUDO的伪标签差异度量用于模型排名而非单一评估，利用目标域无标签数据构造可靠的排名指标，并在源标签稀缺时优于传统源域AUC基线。

**🔧 技术方法**

采用SUDO、伪标签差异度量、可靠性-完整性曲线下的AURCC、零射（Zero‑Shot）文本提示、MLP Probe等技术。

**📊 数据集**

使用六个胸部X光基础模型，源数据来自MIMIC‑CXR、NIH ChestX‑ray14和CheXpert，目标域为未参与预训练的PadChest数据集。

**📈 对比分析**

与源域AUC基线和目标域真实AUC进行对比，Spearman相关系数最高达0.943，尤其在源样本较少时AURCC能够提供更准确的排名。

**⚠️ 局限性**

局限性包括仅在二分类肺炎任务和单一目标域（PadChest）下验证，未涉及多类别、不同模态或更广泛的临床场景。

---

## 444. AGRO-Nav: Autonomous Graph-based Orchard Navigation

**arXiv ID:** 2608.25799 | [PDF](https://arxiv.org/pdf/2608.25799v1)

**作者:** Ho Young Yun `[一作]` (Korea University of Technology and Education), Duksu Kim `[通讯]` (Korea University of Technology and Education)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 AGRO‑Nav 框架，自动从 SLAM 点云提取树行拓扑图，利用该图进行全局行驶规划并生成平滑轨迹，适用于差分驱动和四轮转向平台；

**💡 创新点**

① 自动构建稀疏可连通的树行拓扑图，无需人工 waypoint；② 通过 Dijkstra 在图上规划行驶路径并用 Theta* 连接起止点；③ 用立方 B‑spline 对轨迹进行平滑；④ 在真实果园与仿真环境下证明误差小、规划快、树密度下降仍稳健；

**🔧 技术方法**

PCA 拟合树行线、LiDAR‑SLAM 3D 点云、拓扑图构建、Dijkstra 搜索、Theta* 任何角路径、立方 B‑spline 平滑、Isaac Sim 仿真、ROS2/RTK‑GNSS 位置、成本地图；

**📊 数据集**

真实韩国果园数据（4WS 平台、128 通道 LiDAR、RTK‑GNSS、IMU）以及 Isaac Sim 生成的完整与 70% 树密度的虚拟果园；

**📈 对比分析**

与 Nav2 默认 A*、Theta* 及基于 RANSAC 的结构基线对比；AGRO‑Nav 平均行驶误差仅 0.08 m（实测）/0.14 m（仿真），比 A*/Theta* 低约 3–5 倍；规划时间约 4–5 倍快；最小树干间隙最大，保持安全余量；

**⚠️ 局限性**

仅实现静态全局规划，未处理动态障碍；仅在单一果园测试；对树行可观测度有最低阈值，极稀疏环境可能失效。

---

## 445. EVOMAL: Self-Poisoning in Self-Evolving Coding Agents

**arXiv ID:** 2608.25776 | [PDF](https://arxiv.org/pdf/2608.25776v1)

**作者:** Xiaodong Wu `[一作]` (Queen's University), Jianbing Ni `[通讯]` (Queen's University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文发现并演示了自演化编程代理（Self‑Evolving Coding Agent）在自我学习循环中可能出现的自毒（Self‑Poisoning）漏洞，并通过一系列攻击实验验证了该漏洞的实用性；随后提出了基于系统提示的Counter‑Prompt防御方法，并与现有工具级别防御进行对比，证明了该防御的有效性。

**💡 创新点**

创新点主要在于：①提出自毒概念，揭示自演化代理在检索‑仿写循环中可自生成恶意技能并传播的全新攻击面；②设计了利用Banner包装的放大攻击（Amplified Self‑Poisoning），显著提升了攻击成功率；③首次将防御聚焦于代理生成的技能，而非仅限于提交的技能，并提出了低成本的Counter‑Prompt防御。

**🔧 技术方法**

技术方面采用了：自演化代理框架（mini‑SWE‑agent）与BGE‑M3检索、ReAct生成；构造Banner+可插拔Payload的恶意技能；通过滚动替换（Rolling‑Replacement）模拟库的传播；使用代理的复制率、检索率与存活率建立Galton‑Watson模型评估自传播；以及在系统提示中嵌入规则实现Counter‑Prompt。

**📊 数据集**

数据集主要使用SWE‑bench（Verified和Pro）中筛选出的Python任务，分别包含153和114个任务；实验中在这些任务上植入8个恶意技能（即自毒种子），并通过各种描述、命名与Banner层级进行泛化。

**📈 对比分析**

评估在六种主流模型（Devstral‑Small‑2、Devstral、Gemma‑4、Qwen3、GPT‑OSS、MiniMax、DS‑V4）上进行。攻击成功率（Agent Self‑Poisoning Rate, ASPR）从20.3%至41.8%不等；使用Counter‑Prompt后APSR降至≤6.7%，且对任务完成率影响不显著；相比现有基于名称、代码扫描和注入检测的防御，这些方法对自毒几乎无效。

**⚠️ 局限性**

局限性包括：①实验仅针对无运行时/模型访问的发布者攻击，未覆盖更强的攻击者；②只在Python任务上验证，其他语言需进一步测试；③Counter‑Prompt依赖系统提示的准确性，可能需针对不同模型微调；④自毒传播模型假设检索一致，实际环境中检索偏差可能影响传播；⑤防御中签名隔离需要人工审核，实际部署成本较高。

---

## 446. EXAONE Tabular 1.0 : Technical Report

**arXiv ID:** 2608.25774 | [PDF](https://arxiv.org/pdf/2608.25774v1)

**作者:** Moonjung Eo `[一作]` (LG AI Research), Soonyoung Lee `[通讯]` (LG AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种紧凑型表格基础模型家族（EXAONE Tabular），通过在上下文学习框架中直接利用支持集预测分类与回归任务，无需为每个数据集进行梯度更新。

**💡 创新点**

核心创新在于跨轴摘要Transformer（CAST），在Transformer层级中交错行内特征注意和跨行支持条件注意，并通过项目和特征摘要标记持续保持细胞级表示，从而实现更强的上下文交互和特征互动。

**🔧 技术方法**

技术方案包括：结构因果模型（SCM）合成预训练数据；交叉轴注意机制与 Scalable Softmax 归一化；MuON 与 AdamW 的混合优化；ECOC 处理多类别问题；缺失值原生处理；支持分块推理与支持集缓存。

**📊 数据集**

在四个公开基准上进行评估：TabArena（分类/回归）、BCCO（含缺失值的分类/回归）、TALENT（大规模分类/回归）以及 ScoringBench（概率回归），涵盖多种真实世界任务。

**📈 对比分析**

与13种基线（TabPFN、TabFM、TabICLv2、TabDPT、LimiX、TabSwift 等）以及GBDT/AutoML系统对比；在 TabArena 上仅使用默认配置即获得分类榜首，并以约 1/11 的推理成本逼近 1.64B 参数的 TabFM 回归；在 BCCO/TALENT 上排名第二/第一；在 ScoringBench 取得 R²、RMSE、CRPS 的最高平均排名。

**⚠️ 局限性**

局限性包括：分类头仅支持最多10类，需通过 ECOC 分解；对大支持集的处理目前采用子采样，缺乏高效的上下文压缩或自适应支持选择策略；查询分块推理虽然降低显存占用，但仍需额外计算和缓存管理。

---

## 447. Drift-Aware Multimodal User Representation Learning via Multi-Scale Temporal Modeling and Sparse Mixture-of-Experts

**arXiv ID:** 2608.25773 | [PDF](https://arxiv.org/pdf/2608.25773v1)

**作者:** Ziqing Qian `[一作]` (Tongji University), Nan Cao `[通讯]` (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一个统一的漂移感知多模态用户表示学习框架，包含时间动态感知骨干网络和稀疏 MoE 兴趣适配器。

**💡 创新点**

创新点在于将多尺度时间动态、跨模态信息、显式与隐式兴趣三方面统一建模，并通过稀疏 MoE 实现兴趣解耦，同时采用三阶段训练策略稳定优化。

**🔧 技术方法**

使用 UNITE 多模态编码器，结合 LSTM 与 Transformer 进行短期与长期时序建模，MoE 适配器配合稀疏路由，联合兴趣分类与交互预测的对比损失，并采用三阶段训练。

**📊 数据集**

构建了基于 X（Twitter）的大型多模态数据集，包含 14,015 名用户、7,685,700 条推文、2,890,668 张图片以及 15 个兴趣领域的时间标注。

**📈 对比分析**

与 SASRec、BERT4Rec、PTUM、MIND、HORAE、PeterRec、UniSRec 等基线在兴趣分类和交互预测任务中对比，本文方法在 Hit@1、Recall、NDCG、Accuracy、F1 等指标上均显著优于所有基线。

**⚠️ 局限性**

局限包括依赖 GPT‑5.1 生成的伪标签可能存在噪声；固定短期/长期窗口和专家数量可能不适用于所有用户；仅采用离线训练，缺乏在线持续学习能力。

---

## 448. MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization

**arXiv ID:** 2608.25864 | [PDF](https://arxiv.org/pdf/2608.25864v1)

**作者:** Zaibin Zhang `[一作]` (Dalian University of Technology), Lijun Wang `[通讯]` (Dalian University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MA‑VLA 框架，将多臂协作任务从单一指令分解为可解释的原子动作，并将其显式分配给各臂，实现多臂之间的协同控制。

**💡 创新点**

创新点在于通过原子动作提示与训练时的 Arm Shuffle（臂角色随机打乱）来消除臂身份偏倚，支持在未见的协作模式下实现组合式泛化。

**🔧 技术方法**

采用 VLM（如 GPT‑4）生成原子规划，结合 Pi0 风格的 VLA 执行器与 flow‑matching 机制，同时使用 Arm Shuffle 与 View Dropout 等正则化技术。

**📊 数据集**

使用 RoboFactory、RoboTwin 2.0 仿真基准和真实双臂 SO101 平台的专家演示数据，涵盖多臂堆叠、放置、传递等任务。

**📈 对比分析**

与 ACT、DP、Pi0 等单臂或传统 VLA 基线对比，MA‑VLA 在领域内协作成功率提升 10–20%，在未见协作模式下实现非零成功，最多提升约 13% 的成功率。

**⚠️ 局限性**

局限性包括对原子动作库的依赖（需手工或规则生成），以及在极大臂数或高度动态环境下的泛化尚未充分验证。

---

## 449. Key Point Analysis Needs Structure Recovery: Task Definition, Dataset Diagnosis, and a Structure-Aware Benchmark

**arXiv ID:** 2608.25854 | [PDF](https://arxiv.org/pdf/2608.25854v1)

**作者:** Zhiqiang Shi `[一作]` (King's College London), Oana Cocarascu `[通讯]` (King's College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新定义了关键点分析（KPA）为结构化预测任务，诊断现有KPA数据集的不足，并通过人机循环重注释构建了分布敏感、结构化的ArgKP‑X基准，附带额外数据资源；

**💡 创新点**

创新点在于：①提出“True KPA”框架，将语义聚类、KP生成、覆盖与流行度估计统一为结构化预测；②证明现有ArgKP数据集存在ceiling violation与selection failure；③构建分布敏感子集并进行人机协同重注释，显著提升聚类质量、KP质量与覆盖率；④发布支持可解释性、LLM评估与匹配的补充数据集；

**🔧 技术方法**

使用技术包括：LLM预生成结构（Qwen3 235B）、人工校准与分层重注释、分布敏感采样（句子嵌入+K‑Means）、LLM评估（GLM 5、DeepSeek V3.2）、人工评估（五分制）以及覆盖与冗余量化指标；

**📊 数据集**

核心数据集为ArgKP21；基于其构建ArgKP‑X（重新注释的分布敏感子集），并使用Qwen3 235B、GLM 5、DeepSeek V3.2等LLM进行评估；额外使用MPNet 2进行语义多样性采样；参考ArgMining 2021等；

**📈 对比分析**

对比方法为：在每个子集上同时给出原始ArgKP21结构与重注释结构，使用LLM与人工评估分别给出四维度（语义聚类、KP生成、覆盖、流行度）五分制评分；结果显示：LLM评估从约8.5/9.5提升至约19.4/19.8，人工评估从3.0/3.07/2.60提升至4.73/4.60/4.80，表明重注释结构在所有维度上均显著优于原始注释；

**⚠️ 局限性**

限制包括：①仅在ArgKP21范围内验证，未扩展至更多领域与文本来源；②目前仅提供评估基准，缺乏对应的训练数据集；③关注点在任务定义、数据诊断与基准构建，未来需进行大规模系统评估与进一步验证。

---

## 450. DEFUSE: Generalizable Backdoor Defense for Self-Supervised Encoders with Generative Priors

**arXiv ID:** 2608.25851 | [PDF](https://arxiv.org/pdf/2608.25851v1)

**作者:** Tuo Chen `[一作]` (Southeast University), Jian Liu `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DEFUSE，一种基于条件扩散模型的通用后门检测框架；

**💡 创新点**

创新点在于将后门检测转化为对编码器表征的语义重构任务，并利用生成先验评估重构语义一致性，完全不依赖先验干净数据或特定攻击假设；

**🔧 技术方法**

主要技术包括微调预训练的条件扩散模型（如SDXL）、在冻结的交叉注意力模块中投射表征为图像提示、使用参考编码器（如DINOv2）计算语义相似度；

**📊 数据集**

实验数据集为ImageNet-1K、ImageNet-100以及CC3M子集，并在七种主流自监督后门攻击（SSLBKD、CTRL、BLTO、CLIP Backdoor、BadEncoder、DRUPE、BadCLIP）上进行评估；

**📈 对比分析**

与DECREE、DBCL、DeDe和PatchSearch等四种基准防御方法相比，DEFUSE在AUPRC、AUROC、召回率等指标上均显著领先，尤其在极低比例污染（1%）和多样化触发器（HTBA、Blended、Watermark、SIG）场景下保持高性能；

**⚠️ 局限性**

局限性主要体现在对扩散模型推断开销较高、对极低污染比例或复杂、隐蔽触发器时检测性能可能略有下降，以及对生成模型的依赖可能限制在资源受限环境中的部署。

---

## 451. FlowMoDL: Model-Based Deep Learning with Conjugate-Gradient Data Consistency for Highly Accelerated 4D Flow MRI Reconstruction

**arXiv ID:** 2608.25828 | [PDF](https://arxiv.org/pdf/2608.25828v1)

**作者:** Tristan Gottwald `[一作]` (Leibniz University Hannover), Jana Hutter `[通讯]` (Leibniz University Hannover)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为FlowMoDL的混合模型，专门用于在10×到50×的高加速4D流MRI中重建既能保留解剖结构又能精确恢复血流速度的图像。

**💡 创新点**

创新点包括：① 双通道加速条件（FiLM+MLP）让单模型适应多种加速率；② 明确的速度相位损失（速度、相对误差、角度误差）与课程学习策略提升物理真实性；③ 通过(3+1)D时空残差块和CG数据一致性迭代实现高效梯度步长，无需递归隐藏状态。

**🔧 技术方法**

技术手段：MoDL框架、(3+1)D时空残差块、FiLM特征调制、MLP加速权重、共轭梯度数据一致性求解、深度监督组合损失、课程调度、AdamW+混合精度训练。

**📊 数据集**

使用多中心多厂商的CMRx4DFlow挑战数据集（138例），划分训练/验证/测试集并在10×至50×的加速率下进行评估。

**📈 对比分析**

对比方法包括CG‑SENSE、原始MoDL、FlowVN和FlowMRI‑Net，使用nRMSE、SSIM、相对误差和角度误差等指标。FlowMoDL在所有加速率下均显著优于对手，nRMSE下降至0.047，SSIM提升至0.941，角度误差仅约24°，相对误差仅0.27。

**⚠️ 局限性**

局限性：尚未验证在>50×加速、不同场强或扫描协议下的泛化；对极低信噪比下相位噪声的鲁棒性有限；训练需要高算力与混合精度；推理速度与实时部署尚未充分评估。

---

## 452. WaveOp-LiteFM: Lightweight Neural-Operator Flow Matching for Satellite-to-Radar Precipitation Retrieval

**arXiv ID:** 2608.25818 | [PDF](https://arxiv.org/pdf/2608.25818v1)

**作者:** Chunlei Shi `[一作]` (Southeast University), Junming Hou `[通讯]` (Southeast University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了轻量化的 WaveOp-LiteFM 框架，用条件流匹配在像素空间完成卫星到雷达降水检索。

**💡 创新点**

创新点在于将谱-局部-小波（SLW）块与自适应门控、加性跳跃融合结合，彻底替代传统 U‑Net 速度网络，实现高效、细节保留的像素空间流匹配。

**🔧 技术方法**

采用条件流匹配、FFT 频谱卷积、Haar 小波阈值、深度可分离卷积、流时间嵌入、轻量化 encoder‑decoder 与自适应门控。

**📊 数据集**

使用东南中国卫星‑反射率数据集、公开 Satellite‑to‑VIL 数据集，并在中国全国尺度（台风 Bavi）进行大面积推理验证。

**📈 对比分析**

与 LiteFM‑UNet 在相同条件下对比，参数减少 52.9%，GFLOPs 降低 6.7×，在高强度阈值（35 dBZ / VIL@219）仍保持或提升 CSI、POD 等性能；相较于 CNN、Transformer、diffusion、flow 等基线，取得最优的质量‑效率平衡，并在参数量最少的前提下实现最优的图像质量和阈值技能。

**⚠️ 局限性**

局限性包括：对不同地区、传感器、季节或降水类型的泛化尚未充分验证；大面积推理需使用块拼接；对极端低分辨率或极端天气事件的鲁棒性未作深入探讨。

---

## 453. TDFNet: Tri-projection Deformable Fusion Network for Panoramic Salient Object Detection

**arXiv ID:** 2608.25808 | [PDF](https://arxiv.org/pdf/2608.25808v1)

**作者:** Qiangqiang Zhou `[一作]` (Jiangxi Normal University), Ping Li `[通讯]` (Jiangxi Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对全景显著目标检测（PSOD）任务，提出了三投影可变形融合网络 TDFNet，利用 ERP、CMP 与切线投影三种投影的互补特性进行特征提取与融合。

**💡 创新点**

创新点：
① 引入切线投影作为第三支路，补充 ERP 的极区拉伸失真和 CMP 的面边界不连续性；
② 设计跨投影可变形注意（CDA）模块，利用 ERP–CMP 对应关系生成几何感知采样点，提升两投影间的上下文聚合；
③ 研发纬度引导融合（LGF）模块，先用球面纬度先验对 ERP 与 CMP 进行自适应加权，再用切线投影提供的无失真语义参考完成跨投影细化。

**🔧 技术方法**

技术手段：Hybrid‑ViT + ResNet 编码器；跨投影可变形注意（Hybrid Reference Point Generation + Deformable Transformer Aggregation）；纬度引导融合（Geometry‑Only 与 Mixed 两种策略）；轻量级卷积与深度可分离卷积；多尺度 FPN 结合自上而下的解码器。

**📊 数据集**

使用的公开数据集：360‑SOD、360‑SSOD、ODI‑SOD、F‑360iSOD，共计约 10,000+ 全景图像。

**📈 对比分析**

与 19 种 SOTA 方法（含 360° 专用和 2D 图像方法）在四个基准上进行对比。TDFNet 在 S‑measure、MAE、E‑measure、F‑measure 上均取得领先，MAE 下降约 10‑20%，S‑measure、E‑measure、F‑measure 均提升 2‑5%。

**⚠️ 局限性**

局限性：
① 推理速度与资源消耗未做详细分析，三投影转换与可变形注意会增加运算量；
② 对极端极区小目标或遮挡严重场景仍可能出现漏检或边缘模糊；
③ 依赖全景投影转换，非全景或动态视频场景的迁移性能尚待验证；
④ 需要额外的投影预处理步骤，部署成本相对较高。

---

## 454. Geometry-Constrained Kolmogorov-Arnold Networks: Learning Edge Geometry via Banach Duality

**arXiv ID:** 2608.25807 | [PDF](https://arxiv.org/pdf/2608.25807v1)

**作者:** K S Sesh Kumar `[一作]` `[通讯]` (Brevan Howard Centre for Financial Analysis), K S Sesh Kumar (Brevan Howard Centre for Financial Analysis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可学习几何参数的 Kolmogorov–Arnold 网络（KAN），通过学习每条边的 p 指数来动态调节激活函数的几何形状；

**💡 创新点**

创新点在于将函数空间几何视为可学习的可解释参数，而非预设的基函数集合，从而实现对激活函数形状的自适应调节；

**🔧 技术方法**

使用了 Banach 对偶映射构造的 ℓ^p 激活、tanh 激活以及两者组合的 Banach‑KAN，采用 Adam 优化、梯度裁剪及 C^1 处理的正则化；

**📊 数据集**

在 50 个 AI Feynman 及 10 个自定义符号回归目标、12 个 PMLB 传统回归数据集，以及 CIFAR‑10/100/STL‑10 的图像分类任务上进行实验；

**📈 对比分析**

与固定基函数的 KAN（B‑spline、Chebyshev、RBF 等）、MPL、MLP 等基线进行比较，显示 Banach‑KAN 在无噪声、噪声、少样本和外推任务上取得平均排名最低、NRMSE 低、稳健性最佳；

**⚠️ 局限性**

局限性在于单层或浅层架构的容量受限，未在深度网络或更高维度任务上深入验证；未来需将几何约束扩展到更深层、组合基函数以及更广泛的数据域。

---

## 455. HypoForge: A Self-Improving Multi-Agent Framework for Automated Hypothesis Generation and Testing via Scientific Skill Learning

**arXiv ID:** 2608.25770 | [PDF](https://arxiv.org/pdf/2608.25770v1)

**作者:** Ziqing Qian `[一作]` (Tongji University), Nan Cao `[通讯]` (Tongji University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个经验引导的多代理框架，通过学习可重用的科学技能实现自动化假设生成与假设检验。

**💡 创新点**

将假设生成与检验分别采用阶段特定的学习范式；利用对抗生成‑判别机制提升生成质量，利用实验结果进行经验驱动的检验技能学习，从而实现无需微调基础模型的持续改进。

**🔧 技术方法**

基于大型语言模型（如DeepSeek‑V4‑Flash）的生成器、判别器与执行器；对抗学习、分布式评估、经验提炼、LLM驱动的技能蒸馏等技术。

**📊 数据集**

使用HypoBench基准，涵盖13个跨领域任务，包含真实数据集（如Dreddit、News Headline等）和合成数据集（如College Admission）。

**📈 对比分析**

与系统级基线（HypoGeniC、POPPER、ReAct等）及技能级变体（No‑Skill、AI‑Generated‑Skill、Human‑Designed‑Skill）对比；在假设生成上Q得分0.785、Hit@K 0.648；在假设检验上测试通过率0.659、执行成功率0.966，均显著优于基线。

**⚠️ 局限性**

受限于反馈质量与数量、仅适用于可执行的数据驱动实验，难以处理理论推理、实验室工作及跨学科知识整合；技能粒度与组合的自动化仍需进一步研究。

---

## 456. Anytime Global Tensor Motion Planning

**arXiv ID:** 2608.25830 | [PDF](https://arxiv.org/pdf/2608.25830v1)

**作者:** Sai Coumar `[一作]` (Purdue University), Zachary Kingston `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在 GTMP 基础上，提出了一种可使用任意黑盒局部规划器的通用层级采样框架，并给出两种迭代策略：随机重启以实现几乎完全的拓扑类覆盖，及随预算递增的 AO‑GTMP 以收敛到最优成本。

**💡 创新点**

创新点在于：① 将 GTMP 的邻层边连接泛化到任意局部规划器；② 证明单一采样图即可覆盖所有 δ‑清晰的同伦类；③ 通过随机重启和信息集扩展实现几乎确定性的覆盖与渐进最优。

**🔧 技术方法**

使用了批量张量运算构建层级多部件图、动态规划（值迭代）求最短链、随机采样与局部规划器（直线、RRT‑Connect）以及信息集采样。

**📊 数据集**

实验使用了 6–8 维度机械臂数据集 MotionBenchMaker 以及 Sturtevant 的 2D 高程地图（Sydney、Shanghai 等）。

**📈 对比分析**

与 FCIT、BIT*、RRT* 等基线相比，Anytime‑GTMP 在 60 s 预算下达到约 85 % 的成功率且覆盖最多同伦类；AO‑GTMP 在成本上与 AORRTC 相当或更优，并在 Manipulation 问题上击败多种基线。

**⚠️ 局限性**

局限在于未实现惰性评估、在线预算调度以及动态可行轨迹的扩展；对高维操纵器的拓扑覆盖验证仍待进一步研究。

---

## 457. LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue Segmentation

**arXiv ID:** 2608.25866 | [PDF](https://arxiv.org/pdf/2608.25866v1)

**作者:** Karen Sanchez `[一作]` (King Abdullah University of Science and Technology), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了LUTSeg数据集及TiSage框架，用于慢性溃疡组织分割。

**💡 创新点**

创新点在于多时点多专家标注的纵向数据集和将医学视觉语言模型的多尺度语义先验融入半监督学习的教师-学生框架。

**🔧 技术方法**

采用MedSigLIP冻结视觉-语言模型提取超像素嵌入，SLIC生成超像素，双尺度log空间融合，像素级自适应教师-先验融合以及基于熵加权的软监督。

**📊 数据集**

使用LUTSeg（141张糖尿病及麻风相关溃疡图像）和公开的DFUTissue数据集进行验证。

**📈 对比分析**

与监督的DINOv2-DPT、DeepLabV3+及半监督的FixMatch、UniMatch-V2对比，TiSage在大多数低标注场景下提升mIoU多达4.1点，显著优于基线。

**⚠️ 局限性**

主要局限在于标注仍具主观性，数据集规模有限，且对罕见组织类别的泛化仍需进一步研究。

---

## 458. VINCENT: Validated Interaction Network for Cross-drug Explanation of Therapeutics

**arXiv ID:** 2608.25841 | [PDF](https://arxiv.org/pdf/2608.25841v1)

**作者:** Fan-Sheng Chuang `[一作]` (North Carolina State University), Kaixiong Zhou `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种后训练的闭环解释框架，利用已训练的药物协同预测模型的原子级信号与梯度信息，构建化学连通的药物子结构（motif），对候选 motif 对进行多次局部扰动验证并将验证结果反馈到 motif 归属以实现迭代优化，从而得到稳定且与预测器行为一致的跨药物 motif 对交互得分。

**💡 创新点**

创新点在于：① 通过结合预测器的交叉原子关联矩阵与积分梯度形成正向交互证据；② 采用多视图（结构、交互模式、验证反馈）生成原子相似度，支持化学连通的 motif 聚类；③ 引入闭环迭代：验证得分反馈到 motif 归属，实现跨药物交互证据的自我修正；④ 通过重复局部扰动验证评估 motif 对稳定性，解决单次前向推断噪声问题。

**🔧 技术方法**

技术包括：图神经网络（GNN）与 2D/3D 表征融合、双向原子级交叉注意力、积分梯度、原子对交互关联矩阵、化学连通的 motif 聚类（基于拉普拉斯正则化的软聚类）、多次局部遮蔽扰动、交互效应的二阶有限差分、softplus+激活频率的验证得分计算、指数移动平均反馈机制。

**📊 数据集**

使用 SARS-CoV-2 药物组合基准（来自 ComboNet），包含 71 对测试药物组合，其中 25 对可获得文献支持的药理学区域注释，共 111 个 motif 注释；此外还利用公开的药物–靶点交互、单药活性和 HIV 组合数据作为辅助训练。

**📈 对比分析**

与 11 种基线（积分梯度、交叉注意力、GNNExplainer、SubgraphX、PGExplainer、CF‑GNNExplainer、以及单信号聚类加验证的对照）在同一预测器上比较。在 25 对已注释数据上，方法取得平均回忆率 0.826、精确率 0.790、Jaccard 0.689，HitRate 76.6%，显著优于基线（回忆率 0.49–0.66）。在 71 对完整测试集上，验证得分与预测器协同评分的 Pearson 相关性 0.423，TP/TN 分离比 3.36，远高于对照基线（1.49–0.48）。

**⚠️ 局限性**

局限性包括：① 仅解释已训练的交叉-aware 预测器，无法直接反映真实生物学因果；② 若预测器学习了偏差或噪声，解释会同样受影响；③ 需要预测器暴露原子级表示与交叉信号；④ 验证过程对计算资源有一定需求；⑤ 文献支持的评价仅覆盖 25 对，扩展至更大规模实验验证仍需进一步工作。

---

## 459. Non-Great-Power Conflict and AI Risk

**arXiv ID:** 2608.25839 | [PDF](https://arxiv.org/pdf/2608.25839v1)

**作者:** Kristina Kempkey `[一作]` (MATS; Irregular Warfare Initiative), Catherine Ge-Wang `[通讯]` (University of Oxford)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究评估了非大国冲突（NGPC）在人工智能时代对灾难性风险的相对重要性，构建因果路径模型并对关键参数进行定性评估，探讨NGPC是否可与大国冲突（GPC）相提并论；

**💡 创新点**

创新之处在于系统化梳理NGPC到三类灾难风险（GPC升级、恐怖主义伤害、AI失控）的多重因果路径，识别并优先排列五个跨路径的共同中介变量（信息环境质量、决策时间压缩、威胁感知、能力扩散、规范侵蚀），并提出针对性干预的优先级；

**🔧 技术方法**

主要采用因果建模与可参数化的乘法风险分解（Event‑centric 与 System‑centric 模型），结合PHIA概率量尺进行定性概率评估；

**📊 数据集**

使用了多来源冲突与恐怖主义数据库（UCDP、PRIO、Global Terrorism Index 等）、历史案例研究、公开报告与学术文献，构成了证据基础；

**📈 对比分析**

比较方法为对三条子假设分别构建模型、分解参数并评估其方向与置信度，随后对不同路径的相对重要性进行排序；由于研究为定性评估，未涉及传统机器学习性能指标；

**⚠️ 局限性**

限制包括：未对GPC风险进行直接定量估计，依赖文献与历史案例，参数仅给出方向而非数值；AI能力扩散的时序与规模不确定，路径间相互作用未建模；最终结论高度依赖假设条件与未来技术发展路径的走向。

---

## 460. Steer the Sampling, Not the Kernel Grid: Geometry-Guided Sampling Operator for Volumetric Segmentation

**arXiv ID:** 2608.25819 | [PDF](https://arxiv.org/pdf/2608.25819v1)

**作者:** Sizhe Wang `[一作]` (Monash University), Zhaolin Chen `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 GeoSample，一种几何引导的局部采样算子，用来替换 3D U‑Net 中的 stride1 与 stride>1 的卷积/池化，显著提升细小结构的分割精度。

**💡 创新点**

创新点在于预测局部 SO(3) 旋转框和步长，指引对称采样并提取梯度/曲率差分，将细化与下采样统一到同一公式，并通过旋转一致的 Consensus Field 对齐跳跃连接，消除尺度间几何失配。

**🔧 技术方法**

采用几何场预测、对称有限差分、差分令牌混合、1×1×1 门控与混合、Slerp 旋转插值、以及 Consensus Field 等技术。

**📊 数据集**

在 BraTS（MRI）、MSD Hepatic Vessel（CT）和 TDSC‑ABUS（超声）三大公共数据集上进行评估。

**📈 对比分析**

与 U‑Net、Deformable Conv v1/v2、Dynamic Downsampling 等基准对比，GeoSample 在 BraTS 上 Dice 88.9%/HD95 6.2mm，Hepatic Vessel Dice 58.2%/HD95 34.3mm，TDSC‑ABUS HD95 27.8mm，且参数从 2.3M 降至 0.8M、FLOPs 也大幅下降。

**⚠️ 局限性**

主要局限是基于三次插值的对称采样会增加显存/运行时间消耗，且早期训练的稳定性需进一步改进；对极稀疏结构的捕捉仍有提升空间。

---

## 461. SkillShield: Prompt-Space Security Skills for LLM Coding Agents

**arXiv ID:** 2608.25817 | [PDF](https://arxiv.org/pdf/2608.25817v1)

**作者:** Xiaodong Wu `[一作]` (Queen's University), Jianbing Ni `[通讯]` (Queen's University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过在系统提示（prompt space）中注入完整安全规则（安全技能）来防御LLM编码代理的恶意执行与恶意代码生成的方案，并在多模型、多工具协议上进行评估。

**💡 创新点**

创新点包括：1) 在系统提示中一次性注入完整安全策略，无需修改模型权重或添加运行时监控；2) 通过主动挖掘已知攻击和被动学习失败轨迹自动生成安全规则；3) 研究有限提示预算下不同粒度（全类、机制束、单类）安全技能的构建与效果权衡。

**🔧 技术方法**

利用LLM生成的自然语言安全规则（安全技能），将其作为完整文本嵌入系统提示；采用Prompt-based skill格式；在多轮工具调用循环中持续启用；使用RedCode基准进行实验评估。

**📊 数据集**

使用RedCode数据集（涵盖执行攻击和恶意代码生成）、731条SWE‑Bench Pro的正向任务用于安全拒绝率评估，以及AgentHarm与ArtPrompt等对抗样本。

**📈 对比分析**

与Prompt Guard 2、Llama Guard 3、Spotlighting、TaskShield、AGrail等基线在执行攻击成功率（ASR）和恶意代码生成平均分（AvgS）上对比。结果显示：全类安全技能将ASR从约67%降至约44%，AvgS降至0.58；单类技能将ASR降至约14%；对两类固定重构攻击保持优势；在71个benign任务中的安全拒绝率仅0.14%。

**⚠️ 局限性**

局限性在于：1) 需预先拥有攻击或失败案例以生成安全技能，无法即时应对新型攻击；2) 提示预算有限，跨类共享时会丢失细节；3) 对重构（jailbreak）攻击仍存在一定失效；4) 仅为API‑only部署的第一道防线，若无运行时监控，安全保障仍有限。

---

## 462. Closing the Gap: Automated Discovery of Secure Dockerfile Reference Standards via Semantic Clustering in Enterprise Inner Source

**arXiv ID:** 2608.25793 | [PDF](https://arxiv.org/pdf/2608.25793v1)

**作者:** Jessica Hösl `[一作]` (Technical University of Munich), Patrick Stöckle `[通讯]` (Siemens AG)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估企业内部源代码库中 Dockerfile 的安全技术债务，提出基于 LLM 的语义聚类和黄金参考发现流程。

**💡 创新点**

通过 LLM 自动生成功能描述实现语义聚类，发现内部高质量参考配置，可在不开发新模板的情况下提升 60% 安全度。

**🔧 技术方法**

使用 Hadolint、ShellCheck、Trivy、HDBSCAN、UMAP、Qwen2.5-30B LLM 等工具，构建六阶段自动化管线。

**📊 数据集**

11,470 个 Dockerfile，来源于 6,247 个内部 GitLab 仓库（占约 44k 仓库中的一部分）。

**📈 对比分析**

与内容（raw）或 HLS 语法聚类相比，语义聚类将基础镜像熵降低 62%，噪声率 17%，平均可利用安全改进 60.4%（比基线低 14%），显著优于传统方法。

**⚠️ 局限性**

仅在单一企业环境下进行，缺乏真实标签和跨组织验证；聚类算法、超参数、LLM/embedding 选择对结果敏感；未完成纵向演化或实时部署验证，内部数据无法公开。

---

## 463. Update Disturbance-Resilient Analog ReRAM Crossbar Arrays for In-Memory Deep Learning Accelerators

**arXiv ID:** 2608.25781 | [PDF](https://arxiv.org/pdf/2608.25781v1)

**作者:** Wooseok Choi `[一作]` (IBM Research Europe-Zurich), Bert Jan Offrein `[通讯]` (IBM Research Europe-Zurich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了基于CMO/HfO_x的Analog ReRAM芯片，能够在350 nm CMOS兼容工艺上实现60 ns快速非易失性切换，并实现全并行权重更新的无扰动特性。

**💡 创新点**

通过预成形纳米尺度导电丝实现热电能聚集，提升设备非线性至k<0.005，从而在非重叠半电压脉冲下保持无扰动，支持并行外积权重更新。

**🔧 技术方法**

采用350 nm CMOS兼容CMO/HfO_x跨越器阵列、COMSOL热-电场模拟、随机脉冲编码的全并行外积权重更新、软边界模型结合Tiki‑Taka算法的硬件感知神经网络仿真。

**📊 数据集**

使用MNIST手写数字数据集（训练10 k张图像，测试10 k张图像）进行网络训练和评估。

**📈 对比分析**

通过与理想k=0、k=0.2以及软边界模型的对比，证明k=0.005时测试准确率为89.1%，使用Tiki‑Taka算法可提升至95.2%，接近浮点基线；与其他Emerging记忆相比，扰动容忍度更优（k<0.005），耐久性超过1 亿次。

**⚠️ 局限性**

限制在于仍需保持极低的非线性k，设备偏移与噪声对精度有影响；在更大规模阵列、极短脉冲宽度以及高功耗/能耗方面的可扩展性尚未完全验证；异向k不对称性和训练数据集规模仍需进一步优化。

---

## 464. PUMA: Post-Hoc Sparsification of Universal Multimodal Embeddings for Efficient Retrieval

**arXiv ID:** 2608.25780 | [PDF](https://arxiv.org/pdf/2608.25780v1)

**作者:** Matteo Attimonelli `[一作]` (Politecnico di Bari), Tommaso Di Noia `[通讯]` (Politecnico di Bari)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PUMA，一种在不重新训练骨干网络的情况下，将冻结的通用多模态嵌入转化为稀疏检索码的后置稀疏化方案。

**💡 创新点**

创新点在于结合TopK稀疏自编码器、密集点积蒸馏、跨模态对齐、辅助特征复苏及渐进k-退火的完整训练流程，使稀疏表示既保持检索几何，又实现高稀疏度。

**🔧 技术方法**

使用稀疏自编码器（TopK encoder + 线性解码器）、cosine重构、稀疏点积蒸馏、InfoNCE对比损失、辅助特征复苏（AuxK）以及特征对齐（cross‑modal alignment）等技术。

**📊 数据集**

在五个M‑BEIR基准上进行评估：CIRR、FashionIQ、VisualNews、Fashion200K 与 MSCOCO。

**📈 对比分析**

与完整稠密检索、Raw TopK、PCA压缩、训练的稠密自编码器以及仅使用encoder的TopK进行对比，PUMA在四个任务上与稠密检索持平或更优，同时存储压缩8–16倍、在大规模候选集上可实现约25倍的检索加速。

**⚠️ 局限性**

局限性包括：对骨干网络几何的依赖，存在两类失败模式（TopK前激活不足与检索对齐不足），以及仅为后置稀疏化，可能继承稠密嵌入的偏差与不平衡。

---

## 465. ToST: A Tree-of-Thought Socratic Teaching Framework for Multi-Path Guidance and Parallel Thinking

**arXiv ID:** 2608.25775 | [PDF](https://arxiv.org/pdf/2608.25775v1)

**作者:** Feng Ling `[一作]` (Beijing Normal University), Heng Yu `[通讯]` (Beijing Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Tree-of-Thought Socratic Teaching (ToST) 框架，实现了 1PMS（单题多解）教学模式，并创建了 MPSG-Bench 基准。

**💡 创新点**

创新点在于：① 将多路径推理树 (Parallel Reasoning Tree) 视为教学决策结构；② 引入 Parallel Sowing 促进多角度探索；③ 设计 Multi-Path Adaptive Guidance (MPAG) 进行路径级诊断与动态切换；④ 用 SOLO 理论搭建五维评估框架。

**🔧 技术方法**

采用大型语言模型（如 Qwen2.5-Math-7B-Instruct、DeepSeek v3.2、GPT‑5 等）、树结构推理、节点匹配与路径优先级计算、基于教师与学生树对齐的自动路径分析。

**📊 数据集**

使用 31k 条多路径教学对话的 MPSG-Bench 数据集；基准数据来源于 GSM8K、MATH‑500、AIME24、AIME25，并通过 DeepSeek v3.2 对问题进行 PRT 构建。

**📈 对比分析**

通过与 SocraticLM、TutorRL‑7B、EduChat 等教育型 LLM 以及 DeepSeek v3.2、GPT‑5 等通用 LLM 的对比实验，ToST 在 Acc（约 11%）和 TreeAcc‑R（多达 20%）等指标上均领先；在人类评测中也获得最高的帮助性、清晰度与学习者满意度。

**⚠️ 局限性**

局限性：仅在数学问题域验证；依赖先验 PRT 的构建，难以在线扩展；可能导致学习者过度依赖 AI 导师，需进一步研究伦理与人机协同。

---

## 466. Large Language Model Few-Shot Prompting with Dilemma Training Outperforms Human Surrogates in Predicting Patient Preferences

**arXiv ID:** 2608.25771 | [PDF](https://arxiv.org/pdf/2608.25771v1)

**作者:** Natasha Ureyang `[一作]` (National University of Singapore), Pin Sym Foong `[通讯]` (National University of Singapore)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过构建基于医疗困境的训练，利用P4-DT AI代理预测患者对严重疾病治疗选择的偏好；

**💡 创新点**

创新在于把价值观视为情境下可演化的决策过程，而非静态评分，采用情境困境训练和双向交互来提取个人偏好；

**🔧 技术方法**

使用OpenAI GPT‑5.5大语言模型，结合提示工程和两阶段（训练+测试）交互式学习；

**📊 数据集**

使用受试者自填的价值观调查、五个系统变异医疗困境的决策结果及开放式文本说明，全部来自12对病人-代理人双人组；

**📈 对比分析**

将P4‑DT的预测准确率（81.7%）与无辅助的代理人（55.0%）及代理人+P4‑DT（61.7%）进行比较，显著优于人类代理人且显著高于随机机率（OR≈5.6）；

**⚠️ 局限性**

局限包括样本量仅12对、来自单一招募渠道、情境测试未能捕捉真实临床情绪与动态变化、模型性能对提示与基础模型敏感、缺乏跨文化与多样性验证；

---

## 467. CEDAR: Controlled and Event-Driven Demand Forecasting via Residual Decomposition

**arXiv ID:** 2608.25871 | [PDF](https://arxiv.org/pdf/2608.25871v1)

**作者:** Junjie Meng `[一作]` (University of Science and Technology of China), Chao Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CEDAR框架，用于在大型电商场景中进行基于决策的需求预测，能在给定未来行动计划的情况下对商品销量进行多步模拟。

**💡 创新点**

创新点在于：①将商品状态与商家行动视为独立的标记并在Transformer中交错编码，显式学习行动驱动的状态转移；②引入残差校正模块，利用LLM生成的事件文本向量捕捉非平稳外部冲击并对基线预测进行校正；③采用两阶段训练分离可控动态与随机扰动，提高长周期回滚的稳定性。

**🔧 技术方法**

主要技术包括Action‑Interleaved Transformer（AIT）、残差校正网络（跨注意力+MLP）、LLM文本编码（BGE‑zh-v1.5、Qwen‑Plus）、因子化时间序列特征、两阶段训练策略以及大规模GPU并行。

**📊 数据集**

使用阿里1688电商平台的32 million条商品轨迹（每条约15周），包含状态（印象、点击、收藏、下单等）与行动（折扣、广告投放）以及对齐的事件信号；对比时也使用Kaggle Store Sales公开数据验证通用性。

**📈 对比分析**

与Informer、TFT、PatchTST、PETFormer、Timer‑XL等主流时间序列基线在两种预测窗口（10周/5周）下对比，CEDAR在MSE、MAE、NMSE均实现显著下降（如10周MSE从0.72降至0.41，MAE从0.19降至0.06）。在线A/B实验中，使用CEDAR的商家LTV提升13%，ROI提升15%。

**⚠️ 局限性**

局限性包括：①模型仍需依赖丰富的外部事件文本，若事件信息缺失或质量低会影响校正效果；②两阶段训练分离可能导致整体最优性受限；③主要关注单品级状态与行动，未考虑多商家协同或供应链级别的交互；④在极端长周期或高频行动变化时，残差校正模块的泛化性尚未充分验证。

---

## 468. THA-Flow Generative Model: Prosthesis Geometry Prediction from Preoperative CT

**arXiv ID:** 2608.25845 | [PDF](https://arxiv.org/pdf/2608.25845v1)

**作者:** Yiping Wang `[一作]` (Changzhou Jinse Medical Information Technology Co Ltd), Liao Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

基于预手术CT生成三维假体几何，并可按骨架与可选假体参数条件生成不同设计的假体

**💡 创新点**

首次将生成式AI应用于THA三维手术规划，通过条件流匹配实现一体化的假体分布生成

**🔧 技术方法**

使用AutoencoderKL对骨架和假体进行压缩，三维UNet执行条件正则化流匹配，并将骨架latent与结构参数共同注入生成网络

**📊 数据集**

利用1,355例THA病例（1,149名患者）单中心CT数据，涵盖7种主要股骨柄设计

**📈 对比分析**

与实际术后假体对齐评估，重建指标PSNR 47.11 dB、SSIM 0.9964，生成时间约1.8 s，覆盖93.4%设计，表现优于单一预测方法

**⚠️ 局限性**

单中心数据限制泛化，强骨条件可能削弱假体尺寸一致性，输出为连续几何需进一步映射为标准型号

---

## 469. Socialized Detector Learning: Trajectory-Guided and Reciprocal Distillation for Heterogeneous Object Detectors

**arXiv ID:** 2608.25836 | [PDF](https://arxiv.org/pdf/2608.25836v1)

**作者:** Weihao Li `[一作]` (Tianjin University), Pengfei Zhu `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套社会化学习框架SDL，对异构、专门化目标检测器进行知识传递与演化。

**💡 创新点**

创新点在于通过估计跨检测器转移难度IDTD并按序规划传递轨迹，实现顺序感知的逐步合并与双向蒸馏，突破了传统聚合方法对传递顺序的忽视。

**🔧 技术方法**

采用IDTD估计、贪心轨迹规划、联合类别载体构建、相互蒸馏以及代理证书分析等技术。

**📊 数据集**

使用MS COCO数据集，对四个异构专家进行实验。

**📈 对比分析**

与同期聚合控制进行对比，最终载体在AP上提高了2.6点；相互蒸馏的检测器在原未支持的类别上获得20.8–28.4 AP，且保持与原专家相差不超过1.3 AP。

**⚠️ 局限性**

局限性包括对IDTD估计的依赖、需要预先定义载体初始化、对计算开销和多专家配置的敏感性，以及在不同数据集上的泛化能力未充分验证。

---

## 470. Learning Continuous Regional Temperature Fields with Lead-Time and Resolution Queries

**arXiv ID:** 2608.25823 | [PDF](https://arxiv.org/pdf/2608.25823v1)

**作者:** Chunlei Shi `[一作]` (Southeast University), Dan Niu `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于神经场的连续时空温度预测框架（CTS-TF），可在任意时间延迟和分辨率下直接求解2米温度场；

**💡 创新点**

将预测时延和输出分辨率视为查询变量，采用共享潜在状态+坐标解码器实现一次性全域温度场评估，并通过空间梯度、时间差分、尺度一致性三类正则化保证跨分辨率、跨时延的一致性；

**🔧 技术方法**

使用多尺度天气编码器、坐标解码器（结合正弦时间嵌入和坐标+分辨率嵌入）、正弦时间编码、三种正则化目标（梯度、时间差、尺度一致性）以及轻量级卷积解码；

**📊 数据集**

主要使用SE地区的ERA5-Land 0–6小时短期预报数据，包含9个气象变量；

**📈 对比分析**

与SMAAT‑UNet、EarthFormer、ARROW风格和Weather‑RF/FREUD风格等基线在同一数据集、输入输出和评估协议下比较；CTS‑TF在1–6小时的MAE、RMSE、偏差和空间相关性上均优于基线，尤其在后期时延表现更稳定；

**⚠️ 局限性**

受限于仅有小时级监督目标，模型在非整数时延的准确性无法直接验证；对全球尺度的诊断仅为定性评估，缺乏客观误差统计；

---

## 471. Candidate supply and answer selection shape the value of LLM judging in multi-agent systems

**arXiv ID:** 2608.25937 | [PDF](https://arxiv.org/pdf/2608.25937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 472. Generative AI-Enabled Mission-Aware Radio Orchestration for RIS-Assisted LEO Satellite ISAC Systems

**arXiv ID:** 2608.25803 | [PDF](https://arxiv.org/pdf/2608.25803v1)

**作者:** Fitsum Debebe Tilahun `[一作]` (Korea University), Chung G. Kang `[通讯]` (Korea University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用大型语言模型将自然语言任务指令编译成结构化无线策略，并通过确定性验证与物理层优化实现雷达与通信协同的低地球轨道卫星ISAC系统自适应射频编排。

**💡 创新点**

创新点在于：①将生成式AI作为语义编排层，仅生成策略权重、QoS阈值和求解器引导；②混时域架构将语义适配与物理层最优化分离；③对比零样本与上下文学习，证明上下文学习对权重校准有利但对最终无线性能差异不显著。

**🔧 技术方法**

技术包括：大型语言模型（DeepSeek‑V4‑Flash）用于指令解析；确定性验证器用于策略归一化和阈值检查；物理层求解器（AO、零逼近）用于波束成形、功率分配与RIS相位优化；混时域控制框架。

**📊 数据集**

使用自建仿真数据：M=8天线、K=3接收机、N=32 RIS、600 km轨道、20 GHz载波，随机生成100条频道样本与12条熟悉/24条未见指令，未使用公开数据集。

**📈 对比分析**

比较方法包括TF‑IDF分类器、关键词规则、演示检索、QoS‑only、等权默认、随机RIS等；实验表明优化RIS相位使QoS满足率提升约6倍，生成式AI模式在未见指令下的权重误差和优先级准确率显著低于固定分类器，且LLM‑ICL与LLM‑ZS的无线性能差异不显著。

**⚠️ 局限性**

局限性：仅在单目标、单卫星仿真场景验证；缺乏真实测量数据；对RIS规模、时变CSI和多节点协同的适配待进一步研究；现有求解器为近似AO，可能不收敛到全局最优。

---

## 473. TacForcing: Streaming Action Generation with Execution-Time Tactile Feedback

**arXiv ID:** 2608.25798 | [PDF](https://arxiv.org/pdf/2608.25798v1)

**作者:** Jianbo Zhou `[一作]` (Shanghai Jiao Tong University), Zhijie Deng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TacForcing，一种流式动作生成框架，能够在执行期间动态利用触觉反馈完成接触丰富的机器人操作。

**💡 创新点**

创新点在于：①使用 Streaming Action Expert 将动作块按顺序逐步生成并保留未完成块的中间状态；②引入 Execution‑Aware Tactile Attention (EATA)，使每一次触觉更新仅影响即将执行的块，显著减少触觉获取与动作执行之间的时间误差；③无需额外的反应式控制器，直接在同一生成器中整合执行时触觉信息。

**🔧 技术方法**

核心技术包括：基于 Flow Matching 的连续时间流式生成；块级触觉更新与可变流时间调度；加性注意力掩码实现的 EATA；在训练阶段使用块级中间状态与对应触觉信息对齐的自回归损失。

**📊 数据集**

在 UniVTAC 仿真基准（六个接触任务）和三项真实世界任务（Stand Bottle、Transfer Liquid、Wipe Board）上进行评估。

**📈 对比分析**

与四类基线（纯视觉 VLA、触觉条件 VLA、触觉反应式控制、预训练融合方法）对比，TacForcing 在仿真中平均成功率为 65%，在真实世界中为 69%，分别比最佳基线高出约 14–23% 与 17–42%，在大多数任务上取得最高或近似最高成绩。

**⚠️ 局限性**

局限性包括：①仍依赖预先训练的触觉编码器，对不同硬件或传感器的迁移性可能受限；②块大小与采样步长的超参数需要手工调优，可能影响对不同任务的适配；③在极端快速变化的接触场景下，单块间的更新频率仍可能不足；④缺乏对长期规划或多步策略的分析。

---

## 474. Cooperative Multi-Agent Reinforcement Learning for Adaptive Aggregation in Semi-Supervised Federated Learning with non-IID Data

**arXiv ID:** 2608.25794 | [PDF](https://arxiv.org/pdf/2608.25794v1)

**作者:** Rene Glitza `[一作]` (Ruhr-Universitaet Bochum), Rainer Martin `[通讯]` (Ruhr-Universitaet Bochum)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于多智能体强化学习的联邦学习框架 pFedMARL，用服务器端和客户端的 TD3 代理动态调节聚合权重与个性化学习，从而在数据异构和对抗攻击环境下提升全局模型鲁棒性与客户端个性化性能。

**💡 创新点**

创新点包括：① 将多智能体强化学习与联邦学习结合，形成双层代理架构；② 服务器端智能体通过观察验证误差、相似度等多维度信息自适应调整聚合权重；③ 客户端智能体根据全局与本地误差平衡全局正则与本地更新，实现无预训练的个性化学习；④ 在对抗性攻击场景下自动降权恶意客户端。

**🔧 技术方法**

核心技术为 Twin Delayed DDPG (TD3) 的离策略训练，使用经验回放、目标网络软更新、噪声探索；网络采用两层全连接，参数优化通过 Adam；框架在服务器与客户端分别部署；评估基于语音频谱变换器（AST）与 DCASE 任务 2 数据。

**📊 数据集**

使用 DCASE 2019 任务 2 开发集的 10% 子集（约 14 个机器类别，990 正常训练样本、200 测试样本）作为音频频谱数据；在此数据上训练半监督 AST，进行掩码补全与分类两任务。

**📈 对比分析**

与 FedAvg、Ditto（λ=0.5）、单独本地训练和全局中心化基线对比。实验涵盖三种非 IID 场景（数量/标签偏斜、聚类偏斜）以及包含两个对抗客户端的情况。结果显示：在所有场景下 pFedMARL 的平均局部 F1 和 MSE 均优于 FedAvg 和 Ditto，且在对抗场景下显著抵御恶意更新；相较于本地训练，pFedMARL 在全局测试集上取得更好泛化，且客户端间公平性提升。

**⚠️ 局限性**

局限性包括：① 对极端恶意客户端的抑制仍有限，不能完全排除其影响；② 当前实验规模仅 15 个客户端，未验证在大规模联邦网络中的可扩展性；③ 需要在服务器和每个客户端同时部署强化学习模型，增加计算与通信开销；④ 对其他数据模态（图像、文本）的适用性尚未验证。

---

## 475. When Composition Doesn't Add Up: Humans Identifying Defects in AI-Generated Images

**arXiv ID:** 2608.25933 | [PDF](https://arxiv.org/pdf/2608.25933v1)

**作者:** Ruoqi Hu `[一作]` (Central South University), Hanhe Lin `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AI生成图像中复杂组合因素导致的缺陷进行人类感知研究，构建CO-AID数据集并训练缺陷检测模型。

**💡 创新点**

首次系统性研究组合缺陷，创建局部缺陷标注数据集，并结合主观评估与模型预测实现缺陷定位与修复。

**🔧 技术方法**

使用人工标注、深度学习（fine‑tuned TranSalNet）、GPT‑Image‑1 等技术进行缺陷检测与图像修复。

**📊 数据集**

利用651张从Pexels/Unsplash挑选的参考图、对应的组合prompt以及Midjourney、Imagen、Flux生成的图像组成CO‑AID数据集。

**📈 对比分析**

对三种T2I模型进行主观缺陷统计；训练后的TranSalNet在局部缺陷定位上显著优于GPT‑Image‑1，修复后图像质量明显提升。

**⚠️ 局限性**

样本量有限，仅覆盖四类主体，缺陷种类与模型泛化受限，未深入探索奖励模型训练等进一步提升方法。

---

## 476. Forecasting Multiple Observables with SCROLL: Score-Trained Uncertainty for Stochastic Dynamics

**arXiv ID:** 2608.25898 | [PDF](https://arxiv.org/pdf/2608.25898v1)

**作者:** Pavel Prochazka `[一作]` `[通讯]` (Cisco Inc.), Pavel Prochazka (Cisco Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 SCROLL‑MT 框架，在共享骨干网络上为每个任务引入独立的高斯后验信念和似然，自动归一化各任务尺度并实现输入依赖的预测方差；

**💡 创新点**

通过在最后一层自由路由（shared‑cavity）方式将各任务似然组合为单一目标，避免外部权重搜索，并将 O(K) 似然参数与模型参数一起一并训练；

**🔧 技术方法**

使用贝叶斯自由路由、SCROLL 目标函数、Gaussian/Probit/Ordinal 预测头、SDE 模拟（OU、Lorenz‑63）与真实 PM2.5 数据；

**📊 数据集**

使用 Ornstein‑Uhlenbeck 过程、Stochastic Lorenz‑63 模拟数据以及北京 PM2.5 实测空气质量时间序列；

**📈 对比分析**

在相同网络结构与训练预算下与传统加权损失、多任务学习、MAP、Kendall、MLE σ(x)、Ensemble 等基线对比，SCROLL‑MT 在均方误差、NLL 与校准方面与最佳调参基线持平或略优，且仅需一次训练，计算成本显著降低；

**⚠️ 局限性**

实验仅覆盖低维、单一 lead‑time 的控制系统，未验证高维长序列或多时滞预测；对不同任务共享 α 的选择仍未系统评估，且在极端异构任务的权衡机制方面仍需进一步研究。

---

## 477. A Hybrid Security Framework for Mini-Programs: Visual UI Compliance and Network Risk Assessment

**arXiv ID:** 2608.25877 | [PDF](https://arxiv.org/pdf/2608.25877v1)

**作者:** Panpan Shen `[一作]` (Hainan University), Xiaoqi Li `[通讯]` (Hainan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个集成YOLOv8视觉UI检测和mitmproxy网络分析的微信小程序安全检测原型，用于实时检查按钮尺寸合规性和网络风险。

**💡 创新点**

将视觉识别与网络流量分析结合，提供实时交互式检测；基于人因设计的44×44像素阈值；事件驱动的Pynput触发检测；可视化存证与风险评分同步显示。

**🔧 技术方法**

YOLOv8目标检测、Pynput事件监听、PyAutoGUI/Win32gui截图、mitmproxy网络拦截、Python tkinter GUI、机器学习风险分类器（URL特征）。

**📊 数据集**

自建三套Mini-Program UI图像数据集（共约850张，包含多种类别）以及公开UI控制类数据集500张；网络风险数据集1000个URL特征。

**📈 对比分析**

与PyAutoGUI+OpenCV、Airtest、YOLOv7对比；YOLOv8在精度69.8%/召回66.7%/F1 68.2%，推理116.9ms；截图性能PyAutoGUI 193ms/5%CPU vs Win32gui 124ms/4%CPU；网络工具对比：mitmproxy更易集成、支持HTTPS解密；整体系统实时性好，CPU占用低。

**⚠️ 局限性**

对小尺寸或隐藏按钮检测仍有误差；仅支持Windows截图API；网络分析仅基于URL特征，缺乏深层内容分析；未覆盖跨平台；缺少代码层静态分析，难以捕获逻辑层攻击。

---

## 478. SciMIF: Understanding Multimodal Instruction Following in Scientific Domains

**arXiv ID:** 2608.25973 | [PDF](https://arxiv.org/pdf/2608.25973v1)

**作者:** Ye Shen `[一作]` (Shanghai Artificial Intelligence Laboratory), Guangtao Zhai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 SciMIF 基准，用于评估多模态大语言模型在五大科学领域（化学、地理、生物、材料、物理）下对复杂科学指令（含十类约束）的遵循能力。

**💡 创新点**

创新点包括：① 基于专家分析的两层约束分类体系（领域+功能组）；② 可扩展的指令注入管线，能够在不改变答案的前提下自动化添加科学及通用约束；③ 综合考虑科学正确性与指令遵循两大维度的评测框架。

**🔧 技术方法**

采用了自动约束识别、注入与验证技术，并结合多模态预处理；评测指标包括约束满足率（CSR）、指令满足率（ISR）和分解约束遵循率（DRFR）。

**📊 数据集**

利用13个现有科学数据集（共22项任务），构建 2527 份样本，涵盖 10 类约束与 5 个学科；数据集已公开在 GitHub。

**📈 对比分析**

在多款闭源模型（GPT‑5.2、Grok‑4‑Fast、Gemini‑3.1‑Pro、Claude‑Sonnet‑4.6）与开源模型（InternVL3.5、Qwen3.5 系列）上进行实验；结果显示化学与地理最难，闭源模型整体优于开源；模型规模增大并未显著提升约束遵循，且在一般约束与细粒度约束上表现尤差。

**⚠️ 局限性**

局限性：① 仅评测答案正确性与约束遵循的分离度，缺乏对多步推理质量的细粒度评价；② 对外部符号计算/解析器的依赖不充分，导致细粒度符号与数值约束表现差；③ 规模提升未解决对学科知识的对齐问题，需针对性训练与工具集成。

---

## 479. LivingRAG: Augmenting Graph RAG with Experience

**arXiv ID:** 2608.25960 | [PDF](https://arxiv.org/pdf/2608.25960v1)

**作者:** Yuzhuo Cui `[一作]` (Beihang University), Qingjie Liu `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 LivingRAG，一个可写可复用的图结构检索增强生成框架；

**💡 创新点**

创新点在于将验证过的“经验”以稀疏激活图和简洁推理摘要形式存储，并在后续查询中通过激活图融合和推理脚本两条路径复用；

**🔧 技术方法**

使用基于 LinearRAG 的实体-句子-段落图检索、稀疏向量相似度匹配、NLI 进行经验验证、LLM（Qwen3.6 Plus）生成答案；

**📊 数据集**

在 2WikiMultiHopQA、HotpotQA、MuSiQue、MuSiQue-full 以及 WixQA 四个多跳 QA 流和单跳 WixQA 数据集上评测；

**📈 对比分析**

与 Vanilla RAG、LightRAG、GFM-RAG、HippoRAG2、LinearRAG 等基线对比，LivingRAG 在所有数据集上均实现最高或第二高的 Contain-Match/LLM-Evaluation 准确率，同时相较 LinearRAG，完成 token 下降 22.7%（平均）和整体运行成本下降 12.1%；

**⚠️ 局限性**

主要局限在经验库随时间增长导致匹配开销增加、缺乏对时间敏感知识的失效处理、以及对自然时间序列流中主题漂移和噪声的适应性不足。

---

## 480. Auditable CT Phenotyping Through Report-derived Radiological Observations

**arXiv ID:** 2608.25948 | [PDF](https://arxiv.org/pdf/2608.25948v1)

**作者:** Riga Wu `[一作]`, Tianyu Han `[通讯]` (University of Pennsylvania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过构建基于报告词汇的观测词典，将CT影像与报告对齐，实现可审计的CT表型识别。

**💡 创新点**

采用概念锚定CT表征并结合观察词典进行可解释、可约束的多标签表型预测；引入观察‑探针审计揭示模型潜在的代理信号。

**🔧 技术方法**

利用DINOv2 3D Transformer与CLIP式对比预训练、F2LLM文本嵌入、线性探针、观察‑探针审计与观察库限制等技术。

**📊 数据集**

在CT-RATE、Merlin、PMBB、RSNA‑2023和INSPECT CTPA等多院多解剖CT及其报告数据集上进行训练与评估。

**📈 对比分析**

与五个2D/3D视觉‑语言基线相比，ACT在零射标注和CTPA表型预测上AUROC均优，零射表型AUROC提升约0.079；在探针‑观察审计中，约97个观测词占领所有221个最高位，显示潜在代理。

**⚠️ 局限性**

观察‑探针审计仅捕捉全局语义一致性，无法验证个体影像证据；表型标签来源于弱监督诊断编码，可能包含程序或机构偏差；缺少对实际临床转化与干预验证的评估。

---

## 481. Unveiling Spectral Mechanisms in Training-Free LLM Text Detection

**arXiv ID:** 2608.25944 | [PDF](https://arxiv.org/pdf/2608.25944v1)

**作者:** Haitong Luo `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Yujun Zhang `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并系统评估了训练‑free LLM 文本检测中的频域机制，构建了理论模型（生成活力、头尾划分）并在多种真实世界场景（长文本、短文本、混合源、协作编辑）下进行实验验证。

**💡 创新点**

创新点：①把生成活力与频谱能量关联，解释了人类文本在频域上能量更高的物理原因；②阐明了频域检测的有效边界（文本长度、采样范围）和失效模式；③展示了频域与置信度指标的互补性，并提出可行的融合策略。

**🔧 技术方法**

主要技术包括：概率信号建模（log‑probability 轨迹）、头尾分区与混合模型、方差与频谱能量分析（Parseval 定理）、SpecDetect、Lastde、SpecFusion 等训练‑free 指标，及其在不同采样（Top‑k、Top‑p、温度）下的评估。

**📊 数据集**

数据集：XSum、WritingPrompts、Reddit ELI5、SemEval、CoAuthor、MixText、WMT16 等多语言、多来源、混合与编辑场景的数据。

**📈 对比分析**

与现有指标对比：在标准文档级别，SpecDetect 在长文本上 AUC 通常优于置信度指标；在短文本、混合或编辑文本中置信度指标更稳健；混合/协作文本中 SpecFusion 作为折衷方案保持较好性能。整体上频域指标在长连续生成中表现最好，短片段或高熵采样时表现下降。

**⚠️ 局限性**

局限性：依赖代理语言模型的概率信号，代理模型、解码策略和文本粒度不同会导致得分漂移；实验主要集中在英语和少数语料，跨语言、不同领域、极端编辑方式等情况未充分覆盖。

---

## 482. Formal, Executable and Explainable Runtime Monitoring of Spoken Air Traffic Control Operational Procedures

**arXiv ID:** 2608.25926 | [PDF](https://arxiv.org/pdf/2608.25926v1)

**作者:** Roberto Luvini `[一作]` (University of Genoa), Enrico Russo `[通讯]` (University School of Advanced Defense Studies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个可执行且可解释的运行时验证框架，能够把空中交通管制的语音交流与监视/机载观测融合成时间化轨迹，并根据 ICAO 规定的时限和优先级公式评估并报告程序违规。

**💡 创新点**

①首次将控制器与飞行员口头程序以时限化时序逻辑形式化；②提供完整的多源数据融合管道；③在违规判定时将违规原因与具体观测关联，提升可解释性。

**🔧 技术方法**

采用 ASR（faster-whisper）+ LLM 进行语义解析，使用基于 MTL 的时序逻辑公式进行评估，利用多源时间戳化轨迹进行监控；实现基于 Python 的管道与 GPU 加速 LLM 推理。

**📊 数据集**

主要使用公开的 ATCO2 与 TartanAviation 语音/监视数据集，人工标注 3 小时 KAGC 流量作为真值集；在两起真实事故（Überlingen 2002、Comair 5191 2006）上进行案例验证。

**📈 对比分析**

与基准 LLM 进行精度/延迟对比，最优的无思考 7B 版在 F1=0.85、召回 0.92；与现有相关工作对照，四大需求（定位、整合、时序、解释）均被全面满足。性能方面，语音识别和解析能实时处理，主计算瓶颈为 LLM 解析。

**⚠️ 局限性**

依赖于高质量 ASR 与 LLM 解析；对极少出现的异常事件（如碰撞）仅在事故重现时可验证；实时部署需进一步评估与 ATC 工作流程的兼容性与误报容忍度。

---

## 483. Embedding NDRE Trajectories into Contrastive Learning for Label-Free, Physiology-Aware Crop-Stress Staging and DSS Outputs

**arXiv ID:** 2608.25888 | [PDF](https://arxiv.org/pdf/2608.25888v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 484. AI Agentic Selective Laser Sintering Process Optimization

**arXiv ID:** 2608.25928 | [PDF](https://arxiv.org/pdf/2608.25928v1)

**作者:** Peter Pak `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了一套基于大型语言模型（Claude Fable）的agentic系统，用于在SLS（Selective Laser Sintering）工件的工艺参数（如能量密度、温度、重铺速率等）上进行智能优化，并通过连续学习提升性能；

**💡 创新点**

创新点在于将LLM驱动的工具调用、动态内存管理与持续学习机制相结合，允许在极少人工干预下，通过实时实验反馈快速收敛至符合 ASTM 标准的工艺参数；

**🔧 技术方法**

使用技术包括：大型语言模型+Model Context Protocol工具调用、PostgreSQL动态内存、Patch测试工具、固件级参数设置、可视化GUI与多代理框架；

**📊 数据集**

数据集主要为 ASTM D638/D790 机械测试结果、SLS 打印过程中的光学与热成像数据以及每次构建的参数日志；

**📈 对比分析**

比较方法是将实验获得的拉伸模量、极限拉伸强度、弯曲模量、弯曲强度与制造商技术数据表（TDS）及其参考样本进行对比；在三种材料上，agentic 系统在几轮迭代后实现了与或超过 TDS 规格的机械性能；

**⚠️ 局限性**

局限性包括：传感器分辨率与帧率有限导致实时监测信息粗糙、只能针对低熔点聚合物粉末、固件与软件开放性不足、构建床温度分布不均导致部分工件性能梯度明显、仅评估了三种材料且未覆盖更复杂或高熔点材料。

---

## 485. Repair or Resample? Rethinking Failure Debugging in LLM Multi-Agent Systems

**arXiv ID:** 2608.25920 | [PDF](https://arxiv.org/pdf/2608.25920v1)

**作者:** Zhongwen Luan `[一作]` (East China Normal University), Xiaohong Chen `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于回放的日志记录与重现框架，并构建了包含536条人工标注的失败轨迹的数据集；在此基础上对三大主流LLM驱动多智能体系统（AG2、CrewAI、Magnetic-One）进行大规模失败再现与修复实验，验证了基于症状的节点级干预（Suspicious‑Node Intervention）能显著提升修复成功率；

**💡 创新点**

创新点在于（1）首次设计了可验证前缀重现的MAS回放系统，能在不重新采样前驱步骤的前提下精确复现失败；（2）构造了第一批高质量、可追溯的MAS失败轨迹数据集；（3）证明了局部症状信号比完整任务重跑更能有效引导修复，并在三系统上实现了近三倍的性能提升；

**🔧 技术方法**

主要技术包括事件依赖图构建、结果注入与边界匹配的可重复重放机制、LLM判定器用于自动化失败检测与分类、以及基于症状识别的节点级干预策略；

**📊 数据集**

使用了来自WebArena‑Verified Hard和AssistantBench的200个任务，通过AG2、CrewAI、Magnetic-One各执行一次，共收集了600条原始轨迹，其中536条被标注为失败；

**📈 对比分析**

通过对比基线（全重新运行、Self‑Reflection、Critic‑Agent）与节点干预方法，使用rep_k与pass@k指标评估，结果显示基线在失效再现率上低于回放框架（67.97%→80.78%），且任务级修复成功率仅为6.90%；而症状驱动节点干预在单次干预下即可达到20.15%的修复成功率，超过最强任务级基线约191.89%；

**⚠️ 局限性**

局限性包括：①数据集仅来自公开基准任务，缺乏真实部署场景的多样性；②评估使用LLM判定器，存在误判与不确定性；③实验仅采用单一模型（DeepSeek‑v4‑flash），结果对其他模型或版本的泛化性尚未验证；

---

## 486. One Form to Transfer Them All: Pretraining Multilingual Language Models Beyond Native Orthography

**arXiv ID:** 2608.25904 | [PDF](https://arxiv.org/pdf/2608.25904v1)

**作者:** Muge Zhang `[一作]` (Ohio State University), Sachin Kumar `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在同一实验设置下系统比较了多语言自回归预训练模型的三种输入表征：正字法文本、IPA 音标和罗马化文本，探究它们对跨语言迁移的影响。

**💡 创新点**

①首次在同一架构、相同词表、相同训练预算下全面对比罗马化、IPA 与文本；②发现罗马化预训练在所有规模、任务和语言（已见与未见）上均优于文本和 IPA；③揭示仅在缺少脚本覆盖的基模型上，罗马化 fine‑tuning 可提升效果，而在已覆盖脚本时会导致退化。

**🔧 技术方法**

使用 Byte‑level BPE tokenizer、NanoGPT（改进版）因果 LM、Uroman（罗马化）和 Phonemizer（IPA）工具；通过零样本、少样本提示和监督 fine‑tune（NLI、意图分类、抽象摘要）评估跨语言性能。

**📊 数据集**

构建了基于 FineWeb‑2 的 8 语种（英、西、俄、波、印、乌、泰、马）语料库（约 21.7B 单词）；评测集包括 XStoryCloze、XCOPA、MASSIVE、XL‑Sum、XNLI、以及未见语种（阿拉伯语、法语、孟加拉语、希腊语）等。

**📈 对比分析**

所有模型使用相同 100k 词表、统一计算预算和训练步骤；在 zero‑shot / few‑shot 提示以及监督 fine‑tune 任务上比较性能。结果显示：罗马化预训练在 seen 与 unseen 语言的所有基准上均取得最高分；IPA 在大部分任务中优于文本，但在 Hindi‑Urdu 之外落后；Text→Rom fine‑tuning 在已覆盖脚本的语言上出现显著退化，只有在脚本缺失时才有小幅提升。

**⚠️ 局限性**

限制：①模型规模仅至约 1B 参数，远低于当前主流 7B–100B 规模；②语言覆盖有限，仅涵盖四对语种，缺少汉字、象形文字等；③IPA 转写质量受限，低资源语言可能噪声更大；④模型输出为罗马化或 IPA，无法直接生成原文字，需额外的音标到文字转换方案。

---

## 487. 4DGS-WAM: Bridging Past and Future with an Object-Centric World Action Model based on 4D Gaussian Splatting

**arXiv ID:** 2608.25956 | [PDF](https://arxiv.org/pdf/2608.25956v1)

**作者:** Yueen Ma `[一作]` (Chinese University of Hong Kong), Irwin King `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一种基于4D高斯溅射（4DGS）的对象中心世界动作模型（4DGS-WAM），可在持久的4D表示中对动态物体和静态背景进行分离，并对未来状态进行预测与过去重建。

**💡 创新点**

创新点在于：①将动态场景拆解为可变动物体和静态背景，分别建模；②使用策略网络预测未来SE(3)动作，世界模型仅对动态物体高斯进行变换；③在4DGS表示中实现了高效的渲染与推理；④通过视觉基础模型（分割、深度、光流、相机姿态）实现对象跟踪与轨迹提取。

**🔧 技术方法**

核心技术包括4D Gaussian Splatting、对象轨迹记忆的Transformer策略网络、基于几何张量网络的世界模型、以及一系列视觉基础模型（SAM、DA3、VGGT、WAFT 等）。

**📊 数据集**

在KITTI-MOT基准数据集上进行实验，评估短期未来预测和过去重建。

**📈 对比分析**

与视频生成模型（Epona、DriveDreamer-2、Envision4D）和3D/4D映射方法（MonoGS、EmbodiedSplat、4DGS-SLAM 等）进行比较。4DGS-WAM 在给定相机下的 PSNR/SSIM/LPIPS 均优于视频模型，在动态区域指标上也领先；在过去重建任务中，融合静态与动态高斯的重建效果最好。

**⚠️ 局限性**

局限性包括：①未建模物体间碰撞与交互；②无法处理出现的新物体或缺失的物体侧面；③依赖视觉基础模型，感知误差会影响结果；④静态背景不能生成新显露区域；⑤仅评估短期预测，长期滚动仍未解决；⑥仅使用4DGS作为表示，其他4D结构仍待探索。

---

## 488. When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in LLMs

**arXiv ID:** 2608.25941 | [PDF](https://arxiv.org/pdf/2608.25941v1)

**作者:** Suchit Gupte `[一作]` (Ohio State University), Mohammad Mahdi Khalili `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了稀疏自编码器（SAE）在模型剪枝后保持可靠性的理论与实证，阐明不同剪枝方法对SAE的影响机制；

**💡 创新点**

提出“扰动能量”这一协方差加权范数，用来解释并量化剪枝对SAE的扰动，并发现中间层更易受损，基于此提出层级稀疏分配策略；

**🔧 技术方法**

采用扰动理论与Lipschitz连续性分析，实验使用幅值剪枝、Wanda、SparseGPT三种方法，并用SAEBench四大指标（Core、Feature Absorption、SCR、TPP）评估；

**📊 数据集**

利用公开的预训练大语言模型（BERT、GPT等）的残差流激活数据，使用OpenWebText样本做剪枝校准，评估基于SAEBench的四类指标；

**📈 对比分析**

在四个模型（参数量差两倍以上）上分别以25%、40%、50%稀疏度实验，结果显示Magnitude剪枝最差，Wanda次之，SparseGPT最优；中间层对剪枝的敏感性最高，层级稀疏分配在保持整体稀疏度的前提下可降低perplexity；

**⚠️ 局限性**

实验仅固定SAE权重不再训练，未考虑有结构剪枝、量化或蒸馏；仅针对少量模型族，未给出中间层脆弱性的理论解释，层级稀疏策略仅做初步验证，缺乏与其他分配方法的全面对比。

---

## 489. A Statistical Audit of Physical AI Benchmark Redundancy

**arXiv ID:** 2608.25940 | [PDF](https://arxiv.org/pdf/2608.25940v1)

**作者:** Zaruhi Navasardyan `[一作]` (Metric), Hrant Davtyan `[通讯]` (Metric)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并分析了51个物理AI模型在12个基准上的完整评分矩阵，评估基准冗余并提出最小化的基准套件。

**💡 创新点**

首次在物理AI领域进行基准级别的冗余审计，提出可保留78.5%辨别力的四基准子集，并用Bradley–Terry模型实现无API的模型排名。

**🔧 技术方法**

采用Spearman相关、岭回归交叉验证、Gini系数、前向贪婪选择及Bradley–Terry概率模型等统计与机器学习技术。

**📊 数据集**

使用12个精选物理AI基准（如VSI‑Bench、EmbSpatial、RefSpatial‑Bench、Where2Place、ERQA、CV‑Bench、SAT、RoboSpatial、RealWorldQA、OmniSpatial、MindCube、BLINK）及51个模型的公开/自测得分。

**📈 对比分析**

通过将四基准得分映射到Bradley–Terry模型获得Elo评分，得到的排行榜显示开源与闭源系统混合，空间专门训练模型在排名上显著提升。

**⚠️ 局限性**

仅基于基准级别汇总，缺乏完整重现与项级测量；矩阵稀疏和噪声可能影响冗余评估；前向贪婪选择非最优，且未验证对实际物理任务的迁移效果。

---

## 490. XREPOTEST: Benchmarking Multilingual Repository-Level Unit Test Generation for Large Language Models

**arXiv ID:** 2608.25939 | [PDF](https://arxiv.org/pdf/2608.25939v1)

**作者:** Dung Le Quang `[一作]` (Hanoi University of Science and Technology), Phuong T. Nguyen `[通讯]` (University of L'Aquila)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了一个多语言、仓库级别的单元测试生成基准XRepoTest，覆盖Rust、Go、Julia、PHP和Ruby，并提供了容器化的评估框架；

**💡 创新点**

创新点包括：①以真实仓库环境为评估背景；②提出Invocation Rate（IR）指标以检验测试是否真正调用被测函数；③支持多种上下文增强策略（文件级、LSP、检索）；④公开完整数据集与评估代码，方便后续研究；

**🔧 技术方法**

技术手段包括：LLM生成单元测试；抽象语法树解析、LSP符号解析、BM25与Dense检索获取上下文；Docker容器化执行测试；评估指标涵盖TPR、覆盖率、CSR、变异分数以及新引入的IR；

**📊 数据集**

使用了XRepoTest数据集，包含3642个目标函数，来自6-10个开源仓库，涵盖六大应用领域，语言覆盖Rust、Go、Julia、PHP、Ruby；

**📈 对比分析**

在标准、文件级、LSP及检索等多种上下文设置下，对14种主流LLM（如Claude4.5、GPT-5.2、Claude4.5 Sonnet、GPT-OSS等）进行了实验。实验显示即使是最强模型，在仓库级别场景下TPR最高也仅约27%，IR与覆盖率表现差异明显，检索和文件级上下文对不同模型和语言的影响不一，凸显了现有LLM在仓库级单元测试生成上的挑战；

**⚠️ 局限性**

局限性包括：统一提示与解码策略可能未充分发挥模型潜力；检索上下文未进行细粒度调优；只涵盖5种语言，难以直接推广到其他语言或框架；实验未探索多轮交互与更深层次的工具链集成；

---

## 491. TAU-Agent: An Agentic Retrieval-Augmented Framework for Traffic Anomaly Understanding

**arXiv ID:** 2608.25935 | [PDF](https://arxiv.org/pdf/2608.25935v1)

**作者:** Yuqiang Lin `[一作]` (University of Bath), Nic Zhang `[通讯]` (University of Bath)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 TAU-Agent 框架，将检索式增强与多工具协同，用于交通异常理解。

**💡 创新点**

创新点在于采用 agentic 结构，动态检索视频字幕和开源词汇跟踪工具作为证据，并通过 RAG 进行多步推理，避免统一采样导致信息丢失。

**🔧 技术方法**

使用了 Gemini 视觉字幕、GroundingDINO+YOLO+ByteTrack 轨迹检测、LoRA 微调 Qwen3-VL-8B 的 VLM，以及多任务提示工程和 RAG 机制。

**📊 数据集**

训练数据融合 AI City Challenge Track 3、PSI-VQA 等，测试使用 AI City Challenge Track 3、Track 7 FETV 和 Track 8 PSI-VQA。

**📈 对比分析**

与参赛队伍对比，Track 3 取得第二名，平均分 0.6779，仅比首位低 0.0009；Track 7 FETV 排名第 12，分 0.3998；Track 8 PSI-VQA 排名第五，分 67.9275，尤其在开放式提示任务上取得最高分。

**⚠️ 局限性**

限制在于对 fisheye 或 egocentric 视角的结构化预测适配不足，工具检索错误会传播导致推理受损；在非传统摄像头视角任务中表现仍显劣势，需要进一步自适应训练以提升跨域性能。

---

## 492. Visual General Intelligence: A White Paper

**arXiv ID:** 2608.25924 | [PDF](https://arxiv.org/pdf/2608.25924v1)

**作者:** Hirokatsu Kataoka `[一作]` (National Institute of Advanced Industrial Science and Technology), Zhuang Liu `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本白皮书系统性收集并梳理了多位研究者关于视觉通用智能（VGI）的视角，阐述了从生成模型、预测、重建、持续学习、空间记忆、身体交互、多模态整合等方向出发的潜在路径与关键问题；

**💡 创新点**

创新点在于将视觉与语言、感知、多模态、持续学习、科学发现等跨领域视角统一为一种开放式的研究议程，强调生成与物理结构、主动感知与持续适应的协同作用；

**🔧 技术方法**

主要采用文献综述、概念框架构建和对比分析等方法，而非单一算法实现；

**📊 数据集**

无特定数据集，文中引用的研究多基于公开的视觉数据集（如ImageNet、COCO、视频生成数据集等）和实验室自制数据，但本篇未提供新的数据集；

**📈 对比分析**

由于本篇为讨论性综述，不包含实验对比；文中对现有视觉模型（如CLIP、Veo-3、Sora、SOTA 3D重建模型等）进行概念性性能评估，指出其在跨任务能力、生成质量、结构推理等方面的不足与潜力；

**⚠️ 局限性**

局限性在于缺乏系统性实验验证与量化指标，讨论更侧重于理论与方向性规划，且对具体实现细节、可复现性及硬件/算力需求的评估不足；

---

## 493. Choose Your Game Wisely: Measuring Game-Theoretic Structures in Real-World Vehicle Interactions

**arXiv ID:** 2608.25917 | [PDF](https://arxiv.org/pdf/2608.25917v1)

**作者:** Yueyuan Li `[一作]` (Shanghai Jiao Tong University), Ming Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套基于轨迹的交互测量框架，用来识别并量化真实车辆交互中的行为变化起点、时序组织、响应动力学与角色稳定性。

**💡 创新点**

创新点在于将游戏理论中的同时移动、顺序移动与领导跟随三种时序结构与实际轨迹行为系统对比，并首次给出可量化的时序稳定性与响应率指标，说明不同交互场景适用不同模型。

**🔧 技术方法**

采用轨迹归一化、速度残差检测、行为变化阈值筛选、时序组织分类、响应延迟测算、GLMM 统计分析以及场景级自助法估计置信区间等技术。

**📊 数据集**

使用六大公开轨迹数据集：INTERACTION、highD、inD、rounD、Waymo Open Motion Dataset 以及 nuPlan。

**📈 对比分析**

通过对事件进行并行/顺序/单侧分类、计算响应概率、稳定性比例，并利用混合效应模型评估不同交互类型的优势，结果显示并行与顺序交互均占显著比例，顺序中角色稳定性高达 80% 以上，平均响应率约 41%。

**⚠️ 局限性**

局限在于仅基于轨迹可观测行为，未考虑灯光、手势等非运动线索；未直接推断决策过程、信息集或均衡机制；对极端交通情境与稀疏交互的覆盖有限。

---

## 494. Lost but not erased: Finding traces of a forgotten language in neural speech models

**arXiv ID:** 2608.25976 | [PDF](https://arxiv.org/pdf/2608.25976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 495. Scalable Multi-GPU Simulation of 3D Multicellular Growth with RNN-Based Workload Balancing

**arXiv ID:** 2608.25890 | [PDF](https://arxiv.org/pdf/2608.25890v1)

**作者:** Matvey Moisseyev `[一作]` (University of Nebraska--Lincoln), Hongfeng Yu `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套可扩展的多 GPU 框架，用于基于亚细胞元素模型的三维多细胞生长模拟，并通过 GPU 加速、空间分箱、域分解、工作量感知初始划分以及 RNN 驱动的动态负载均衡实现高效计算。

**💡 创新点**

创新点在于：①将工作量感知的初始划分与基于时间序列的 RNN 控制器相结合，使划分在细胞增殖、迁移和分裂导致的空间负载动态变化时能够更精准、低迁移量地自适应；②RNN 控制器通过学习残差修正传统的反应式边界调整规则，利用最近的负载与划分状态历史实现预判式负载平衡。

**🔧 技术方法**

使用技术包括：GPU 并行力学计算、基于距离阈值的空间分箱、Ghost 区域通信的域分解、基于细胞元素计数和局部密度的工作量评分、单层 GRU 递归网络进行负载预测与边界修正，以及 OpenCL+MPI 的并行框架。

**📊 数据集**

实验数据集主要为模拟的胚胎表皮发育场景，使用 60K–435K 细胞（0.9–6.3 M 细胞元素）的亚细胞模型进行测试。

**📈 对比分析**

与静态划分、反应式负载平衡及基于 persistence/Holt 的预测-重划分策略对比，RNN 控制器将平均全局不平衡从 11.3% 降至 3.5%，运行时相对静态划分下降 9%，同时切片迁移量比反应式降低 7.7 倍，显示出显著的平衡-迁移权衡提升。

**⚠️ 局限性**

主要局限包括：在高 GPU 数下每个 GPU 的计算量不足导致伸缩性不足；当前实验采用固定问题规模，未充分验证大规模百万细胞级模拟的可扩展性；RNN 训练基于合成工作负载，实际复杂生物过程的泛化性仍需进一步评估。

---

## 496. Loss-Based Active Learning for Neural Abstractive Summarization

**arXiv ID:** 2608.25881 | [PDF](https://arxiv.org/pdf/2608.25881v1)

**作者:** Michail Ioannou `[一作]` (Aristotle University of Thessaloniki), Grigorios Tsoumakas `[通讯]` (Aristotle University of Thessaloniki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为LOBSTER的主动学习框架，用于加速抽象式摘要模型的微调，利用已标记样本的交叉熵损失作为硬例指示器，随后通过语义相似度投射挑选未标记数据；

**💡 创新点**

创新点在于：①将损失作为主动学习的选择信号；②将损失硬例与未标记数据的语义相似度相结合；③使用IDDS做密度预过滤以防止语义崩溃；

**🔧 技术方法**

技术手段包括：交叉熵损失评估、Sentence‑BERT语义嵌入、IDDS密度采样、三阶段采样流程；

**📊 数据集**

实验使用AESLC、XSum和CNN/DailyMail三大英文摘要基准数据集，并在BART‑base与PEGASUS‑large两种生成模型上评估；

**📈 对比分析**

与随机采样、IDDS、BAS（基于不确定性）和DUAL（混合）等方法对比，LOBSTER在所有数据集和模型上均能匹配或超越SOTA，同时在选择时间上比BAS快665×、比DUAL快437×；

**⚠️ 局限性**

局限性包括：仅在英文数据集上验证，可能对其他语言不适用；自动评价指标可能无法覆盖事实一致性和连贯性；假设Sentence‑BERT语义空间可有效映射难度，需进一步验证；模拟标注而非真实人工标注。

---

## 497. Gaming Together on Discord: Teen Gamer's Cross-Platform Practices

**arXiv ID:** 2608.25942 | [PDF](https://arxiv.org/pdf/2608.25942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 498. VISTA: Visually Inferred Spatial ConTact Attention for Contact-Rich Manipulation

**arXiv ID:** 2608.25872 | [PDF](https://arxiv.org/pdf/2608.25872v1)

**作者:** Jiayi Chen `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用可见的柔性抓手形变场（VDF）作为视觉物理反馈的模仿学习框架 VISTA-Policy，用以实现无触觉传感器的高精度接触式操作。

**💡 创新点**

创新点包括：① 将柔性抓手的三维形变场视为高维可解释的物理信号；② 设计 Physics‑Aware Encoding Engine 进行实时形变解码；③ 通过 Energy Aggregation Denoising 机制去除噪声并得到可靠接触置信度；④ 采用相对增量抓手动作空间与形变增强的策略网络，实现闭环自适应控制。

**🔧 技术方法**

技术方法包括：外部摄像机跟踪+光流/语义分割提取抓手边缘，三维重投影得到 VDF；基于 ReLU + Sigmoid 的能量聚合滤波；软门控融合机制；时空 Transformer 编码形变序列；相对增量动作预测与标准机械臂控制。

**📊 数据集**

使用自制数据集：跨尺度物体抓取、不同直径瓶盖旋拆、不同笔刷书写三类接触密集任务；在每类任务中收集 20–40 条专家演示，包含多尺度物体与扰动场景。

**📈 对比分析**

与纯视觉基线 DP3、DP3‑Wrist、触觉基线 TDF‑DM 等对比；VISTA‑Policy 在成功率、误抓率、失物率、写字质量指数等指标均显著优于基线，表现出 100% 的跨尺度泛化、样本效率高、对动态扰动和脆弱物体的鲁棒性；在 OOD 规模与高度下仍保持高成功率。

**⚠️ 局限性**

局限性：需抓手形变可被外部相机观测，完全遮挡或极低光照下性能下降；只提供接触状态信息，无法直接估计力或物体刚度；依赖可见柔性抓手，无法推广至硬抓手或多视角环境；当前仅关注形变与动作的映射，未来可扩展到力估计与更丰富的语义理解。

---

## 499. Less Contouring, More Accuracy: Lesion-Guided ROI Deep Learning for Ovarian Ultrasound Classification

**arXiv ID:** 2608.25965 | [PDF](https://arxiv.org/pdf/2608.25965v1)

**作者:** Mehran Ahmad `[一作]` (Danube Private University), Sepideh Hatamikia `[通讯]` (Danube Private University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文在卵巢超声图像上对比四种AI分类策略（全图DL、ROI-guided DL、轮廓DL及放射组学+机器学习），并验证哪种策略能在保持高准确率的同时降低标注工作量。

**💡 创新点**

创新点在于提出并系统评估lesion-guided ROI策略，证明在仅需矩形框标注的情况下即可获得与精细轮廓相当甚至更优的诊断性能，同时显著减少了标注成本。

**🔧 技术方法**

采用MaxViT‑Tiny、Swin Transformer、EfficientNet‑B7、ResNet18等深度学习模型，并结合MixUp/CutMix增强；放射组学特征提取后使用SVM、KNN、ANN等传统机器学习方法；所有模型统一使用AdamW优化、Cosine学习率衰减和soft‑target交叉熵训练。

**📊 数据集**

使用公开的Multi‑Modality Ovarian Tumor Ultrasound (MMOTU) 八类多分类数据集以及Ovarian Ultrasound Dataset (OUD) 的二分类（DF vs PCO）数据集，共计507例患者。

**📈 对比分析**

在统一的70:15:15拆分和bootstrap置信区间评估框架下，lesion-guided ROI+MaxViT‑Tiny在MMOTU上取得93.10%准确率、AUC 0.991；在OUD上取得97.56%准确率、AUC 0.997；相较全图和精细轮廓方法均有显著提升，表明ROI策略兼顾了高性能与低标注成本。

**⚠️ 局限性**

局限性包括：仅评估B‑mode超声图像，缺乏多中心、多模态（如多普勒、造影）验证；未探索自动化或弱监督标注技术；未对计算资源占用进行量化评估。

---

## 500. PANDA - Prototype-Anchored Alignment for Partially Unpaired Multimodal Learning, with Applications to Alzheimers MRI and TCGA Pathology

**arXiv ID:** 2608.25970 | [PDF](https://arxiv.org/pdf/2608.25970v1)

**作者:** Sheethal Bhat `[一作]` (Friedrich-Alexander-University Erlangen-Nurnberg), Andreas Maier `[通讯]` (Friedrich-Alexander-University Erlangen-Nurnberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种名为PANDA的框架，用于在部分未配对的多模态学习中进行原型锚定对齐，应用于阿尔茨海默病的MRI和TCGA病理学。

**💡 创新点**

创新点在于PANDA框架能够在推理时仅依赖主要模态（MRI），而不需要辅助模态的输入，同时能够处理不同的配对率，包括零重叠的情况。

**🔧 技术方法**

使用了两阶段的训练框架，第一阶段从共享嵌入中推导出类特定的辅助原型，第二阶段将主要编码器与这些固定的原型对齐。

**📊 数据集**

使用了ADNI（阿尔茨海默病神经影像倡议）数据集进行AD/CN分类，以及TCGA-Lung数据集进行生存预测。

**📈 对比分析**

与MRI-only基线相比，PANDA在ADNI数据集上实现了AUC 0.868（提高了7.9个百分点），并在TCGA-Lung数据集上也表现出色，超越了全融合训练的性能。

**⚠️ 局限性**

限制在于该方法的评估仅限于ADNI数据集，外部验证和更广泛的适用性仍需进一步研究。

---

## 501. One Symptom, Three Levers: A Critical Review of On-Policy Self-Distillation

**arXiv ID:** 2608.25936 | [PDF](https://arxiv.org/pdf/2608.25936v1)

**作者:** Justin Robert `[一作]` (OVHai LLM), Raheel Qader `[通讯]` (OVHai LLM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 On‑Policy Self‑Distillation（OPSD）进行系统综述与结构化分析，梳理其机制、优势、失败模式（Collapse）以及对应的调控杠杆，提出统一的术语与评估框架。

**💡 创新点**

创新点在于：① 将OPS的关键问题（信号几何、特权信息、循环稳定性）归纳为三大杠杆；② 建立Collapse的三层症状体系并给出对应的测度；③ 通过对比分析提出可操作的改进方向（Token‑level 选择、信息递减、教师更新策略）。

**🔧 技术方法**

主要采用文献回顾与概念建模技术，对现有方法（OPSD、SDPO、GRPO、OPD 等）进行对照与综合；使用统计评估指标（pass@k、avg@k、entropy 等）进行结果解读。

**📊 数据集**

参考了多项数学推理评测集（如 AIME、C‑Eval、HMMT25、LiveCodeBench v6 等），但本文自身未引入新数据集。

**📈 对比分析**

比较方法主要基于 pass@k、avg@k、entropy 等指标，指出 OPSD 在数学推理任务上可与 RL 方法竞争，生成 token 更少，但在多样性与长期记忆方面易出现 Collapse。

**⚠️ 局限性**

局限性包括：① 仅聚焦数学推理任务，未涵盖多模态与 agent 方向；② 讨论基于近期预印本，模型规模普遍在数十亿以下；③ 未进行新的实验验证，结论主要来自已有工作的汇总与阐释。

---

## 502. Query-Side Attacks on GNN-Based KGQA: Tracing Failures from Entity Linking to Answer Generation

**arXiv ID:** 2608.25922 | [PDF](https://arxiv.org/pdf/2608.25922v1)

**作者:** Pankaj Kumar `[一作]` (National Institute of Science Education and Research), Subhankar Mishra `[通讯]` (National Institute of Science Education and Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了针对 GNN‑基知识图问答管线的阶段隔离鲁棒性评估框架，并通过两类对抗性改写（Compositional Restructuring 与 Relation Synonym Swap）来测试系统在查询侧攻击下的表现。

**💡 创新点**

创新点在于引入阶段隔离协议以明确区分管线各阶段的失败来源，揭示子图构建是性能崩溃的主因，并提出答案存在与答案可达性区别的概念，同时提供基于路径注入的恢复策略。

**🔧 技术方法**

使用了实体链接、基于个性化 PageRank（PPR）的子图检索、GNN 解释器（ReaRev/GAT）以及 LLM 生成式回答，并结合对抗文本改写、答案验证、Jaccard 相似度和指令余弦相似度等指标进行评估。

**📊 数据集**

主要实验数据集为 ComplexWebQuestions 与 WebQSP（基于 Freebase），并在 MetaQA（WikiMovies）上验证方法的迁移性。

**📈 对比分析**

与标准 GNN‑RAG 基线 EM 52.9% 对比，CR 攻击将 EM 降至 0.68%，RS 攻击降至 20.3%；改进方案 GraftNet 在 CR 下提升至 29.82%，路径注入在推理阶段恢复至 51.4%；单步检索模型 EPR‑KGQA 在 CR 下保持约 59% 的 Hit@1。

**⚠️ 局限性**

局限性包括仅评估单一系统（GNN‑RAG）和两大基准；对攻击生成与验证的复杂性与可靠性；依赖 Freebase 的知识图；未覆盖多语言、多知识图以及更广泛的模型对比。

---

## 503. SAMpLE: A SystemC-AMS Machine LEarning-based Framework for Virtual Prototyping

**arXiv ID:** 2608.25910 | [PDF](https://arxiv.org/pdf/2608.25910v1)

**作者:** Andrei Mihai Albu `[一作]` (Politecnico di Torino), Sara Vinco `[通讯]` (Politecnico di Torino)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并公开了 SAMpLE 框架，使得机器学习模型能够以 SystemC‑AMS 的 TDF 模块形式直接嵌入虚拟原型中，支持在线学习与离线推理。

**💡 创新点**

创新点在于：①将 ML 模型作为一等公民集成到 TDF 语义中；②提供统一的接口与双后端（原生 C++ 与 ONNX Runtime）实现模型互换；③实现可复现的实验流程与跨模型比较。

**🔧 技术方法**

使用的技术包括 SystemC‑AMS（TDF）、ONNX 格式与 ONNX Runtime、JSON 配置驱动的数据预处理、C++原生在线学习算法（如 NLMS‑ARX、Hedge Ensemble 等）以及多种离线 ML 框架导出的模型。

**📊 数据集**

主要使用了 UCI Appliances（10 min 采样的家电能耗数据）和 Tetuan City（10 min 采样的城市电力负荷数据）两份公开数据集。

**📈 对比分析**

比较方法：在同一实验脚本中切换不同模型（离线模型如 ARX、XGBoost、Random Forest；在线模型如 NLMS‑ARX、Hedge Ensemble 等），记录 R²、RMSE、MAE 与推理时延；实验表明 ONNX 离线模型与 Python 原始模型几乎一致，在线模型在非平稳数据上显著优于离线模型，且在同一 TDF 调度下可直接比较。

**⚠️ 局限性**

限制主要包括：①对离线模型的性能与精度依赖于预训练阶段；②在线学习算法在复杂非线性任务上仍存在收敛与计算成本的权衡；③目前框架只支持单步预测，尚未覆盖多步或序列生成；④在更大规模的多域仿真中对内存与时钟同步的影响尚未全面评估。

---

## 504. Answer Is Cheap, Show Me the Evidence! Augmenting Automated Vulnerability Assessment with Evidence

**arXiv ID:** 2608.25905 | [PDF](https://arxiv.org/pdf/2608.25905v1)

**作者:** Shengyi Pan `[一作]` (Zhejiang University), Shanping Li `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套基于大语言模型的漏洞评估框架，能够在分析漏洞报告（含代码片段、截图及项目背景）后自动给出 CVSS 评估结果并输出支持性证据；

**💡 创新点**

创新点包括：①使用专门的 LLM 代理对报告中的多模态内容和项目信息进行精细化处理；②构建评估专用 LLM，利用大规模推理轨迹标注结合两阶段微调（监督 + 强化学习）注入评估知识；③通过检索相似历史漏洞补充证据，实现可解释的评估结果；

**🔧 技术方法**

技术栈包括：大语言模型（如 Llama‑3.1‑8B、Qwen3‑8B）、专门 LLM 代理、多模态推理、混合检索（稀疏+密集）+ LLM 重排序、GRPO 强化学习、代码/截图解析与摘要、数据增强与注释；

**📊 数据集**

数据集为从 NVD 与 OSV 合并的 CVE 记录，再从对应 GitHub issue 报告抓取信息，最终得到约 10,000 条漏洞实例，覆盖 54 种编程语言、159 种 CWE；

**📈 对比分析**

实验通过与六种 ML、两种 DL、三种主流 LLM 基线对比，评估指标为加权 F1 与 MCC，结果显示 SV‑Chat 在平均 F1、MCC 及严重性评分上分别比最佳基线提升 5.3–18.7%（F1）和 14.4–35.2%（严重性），并在各 CVSS 指标上均优于基线；ablation 证明专用 LLM、推理轨迹学习与 RL 的必要性；

**⚠️ 局限性**

局限性包括：需要大量人工或 LLM 注释的推理轨迹；推理耗时（约 84.8 秒）和成本相对较高；对信息不足的漏洞仍可能产生不完整或不相关的证据；在缺乏足够上下文时模型仍可能误判，需进一步提升通用性和实时性。

---

## 505. Towards A Unified Information Bottleneck Framework for Time Series Explanations

**arXiv ID:** 2608.25897 | [PDF](https://arxiv.org/pdf/2608.25897v1)

**作者:** Xu Zheng `[一作]` (Florida International University), Dongsheng Luo `[通讯]` (Singapore Management University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了时间序列模型可解释性问题，提出了一种统一的基于信息瓶颈（IB）的框架，用于同时生成归因解释和稳健的对抗式解释。

**💡 创新点**

创新点在于：①将归因与对抗解释整合到同一信息瓶颈优化目标中；②设计可训练的归因瓶颈提取器和条件生成器，解决传统掩码导致的 OOD 问题和对抗噪声不稳定性；③通过分布保持与结构约束损失保证解释实例在原数据分布内且语义可解释。

**🔧 技术方法**

技术细节包括：信息瓶颈理论、变分上界近似、KL 正则化、连续化蒙特卡洛采样与 Straight-Through Estimator、Transformer 编码器-解码器网络、标签一致性（LC）损失、分布保持（KL）损失、结构约束（bound）损失、对抗噪声正则化。

**📊 数据集**

实验数据集共 10 个，合成数据（FreqShapes、SeqComb‑UV、SeqComb‑MV、LowVar）与真实世界数据（ECG、PAM、Epilepsy、Boiler、Wafer、FreezerRegular）。黑盒分类器统一采用 Transformer，亦在实验中验证了对 LSTM 与 CNN 的泛化。

**📈 对比分析**

与 11 个基线（归因：IG、Dynamask、WinIT、CoRTX、SGT+GRAD、TimeX；对抗：CoMTE、AB‑CF、M‑CELS、CONFETTI）在归因指标 AUPRC、AUP、AUR、真实度、置信度、稀疏度、接近度上均实现或接近最优。归因方面平均提升约 11% AUPRC；对抗方面在有效性/置信度与稀疏性/接近度之间取得更优 Pareto 前沿。推理速度最快，训练时间可控。

**⚠️ 局限性**

局限性：模型对超参数（如稀疏度先验 r、λ_con 等）敏感，需要针对不同数据集调优；在某些复杂多模态时序场景下，归因瓶颈可能仍需进一步改进；此外，虽然框架统一，但仍需预训练分类器并额外训练生成网络，增加了整体计算开销。

---

## 506. From Passive Response to Proactive Correction: Enhancing LLM Robustness Against Input Fact Perturbations

**arXiv ID:** 2608.25894 | [PDF](https://arxiv.org/pdf/2608.25894v1)

**作者:** Ping Wang `[一作]` (Renmin University of China), Xiaofeng Meng `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三阶段框架DEDUCE，用于主动检测并纠正用户输入中的事实错误，提升LLM对误导性查询的鲁棒性。

**💡 创新点**

创新点在于将错误检测、策略制定和纠错集成为可解释的多角色推理流程，并首次构建针对多类型事实错误的评测数据与指标。

**🔧 技术方法**

采用分层模块化技术：细粒度事实分解与核验、生成-审阅-仲裁多视角策略辩论、按步骤执行纠错；同时提供两种实现路径——基于提示的推理和端到端微调。

**📊 数据集**

使用自研的MisFactQA数据集（含真假前提、内部矛盾、多重错误），并在TruthfulQA、FalseQA等公开数据上进行评测。

**📈 对比分析**

与默认模型、ICL、CoT、LoRA微调、IAQ‑FA等方法对比，DEDUCE在准确率、误导率、纠错率和澄清评分上均显著提升，尤其在强大模型（Gemma、LLaMA等）上实现约25%+准确率提升。

**⚠️ 局限性**

局限性：实验聚焦中等规模模型，未覆盖更大模型；仅针对事实错误，未考虑歧义或对抗性提示等其他不可靠输入。

---

## 507. Spatial-Knowledge-Graph-Grounded LLM Agents for Neighborhood Livability Evaluation

**arXiv ID:** 2608.25952 | [PDF](https://arxiv.org/pdf/2608.25952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 508. Beyond the Editing Canvas: Evidence Divergence in OOXML-to-LLM Ingestion

**arXiv ID:** 2608.25880 | [PDF](https://arxiv.org/pdf/2608.25880v1)

**作者:** Side Liu `[一作]` (Tulane University), Jiang Ming `[通讯]` (Tulane University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了 Office Open XML (OOXML) 文档在大语言模型 (LLM) 预处理管道中产生的视图分歧（证据 fork），并系统地测评了这些分歧在不同提取器与 LLM 接口中的传播与曝光。

**💡 创新点**

提出了基于 OOXML 规范的证据 fork 挖掘方法，构建了六维分类体系，确认了 21 个跨 Excel、Word、PowerPoint 的规范性差异，并揭示了入侵层配置对 LLM 输出影响的关键机制。

**🔧 技术方法**

利用 OOXML 规范知识库、Claude Opus 与 GPT‑5.5 进行构造与挖掘；用 Python builder 生成最小化 Office 文件；在 13 个提取工具、4 个原生 API 和 7 个 Web 聊天机器人上进行大规模实验。

**📊 数据集**

构造 210 个包含 21 个机制的实验文件（来源自 TAT‑QA 财报片段），以及 2000 个 XLSX、2000 个 DOCX 与 263 个 PPTX 的随机样本用于基率评估。

**📈 对比分析**

采用两道门控（Gate 1: Office 可视性，Gate 2: 提取器曝光）验证文件合法性；在 11 个 LLM 接口上多次重复测试，曝光率在 48%–76% 之间，入口点差异显著，行为一致性高。

**⚠️ 局限性**

实验受限于固定工具/模型版本与配置，结果随更新可能变化；采样不代表真实部署分布；仅考察规范合法且无宏的文件；常见结构的 presence 检测不足以区分恶意与合法使用。

---

## 509. Do Vision-Language Models Agree on the Affective Qualities of Shape? A Cross-Model Audit for Generative Design Interfaces

**arXiv ID:** 2608.25876 | [PDF](https://arxiv.org/pdf/2608.25876v1)

**作者:** Luca Bux `[一作]` (Honda Research Institute Europe GmbH), Stefan Menzel `[通讯]` (Honda Research Institute Europe GmbH)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对六个独立训练的视觉语言模型（CLIP、OpenCLIP、SigLIP2、ALIGN、FLAVA、Qwen3-VL-Embedding）在无纹理的 ShapeNet 3D 物体上，进行离线一致性审计。通过将物体投影到由形容词对（如 elegant–messy）构成的语义轴，计算不同模型对同一轴的排名一致性，并与几何正向对照和无关词对的经验零点对比。

**💡 创新点**

提出一种基于跨模型一致性的离线审计框架，用来评估并挑选在 AI 辅助设计界面中可用的情感语义控制。该框架通过几何轴作为正向对照、无关词对作为经验零点来校准一致性分数，并揭示语义轴与物体类别的交互效应，显示哪些词汇在特定类别下能产生稳健的模型共识。

**🔧 技术方法**

使用视觉语言模型的文本和视觉编码器对每个视图进行嵌入；通过计算词对差向量得到语义轴；对每个模型的物体嵌入进行投影得到分数；使用 Spearman 相关系数求跨模型一致性；对每个类别的嵌入做 PCA 以衡量轴与形状变异子空间的对齐；采用拆半重现性校正、bootstrap 置信区间和相关分析验证结果。

**📊 数据集**

ShapeNetCore 数据集中的 10 类物体（chair、table、lamp、sofa、cabinet、bookshelf、bottle、jar、clock、car），共 4,950 个对象；每个对象从 8 个固定视角渲染无纹理灰色模型。

**📈 对比分析**

通过比较三类轴（几何、情感、无关）在跨模型一致性上的 Spearman 均值：几何轴 0.441 > 情感轴 0.364 > 无关轴 0.135。情感轴的平均一致性显著高于零点（CL=0.906），但低于几何轴（CL=0.658）。一致性在不同类别和轴上差异很大（范围 0.21–0.53），并与轴与类别内部形状变异的对齐度正相关（相关系数 0.72–0.78）。此外，5 模型一致性能较好预测第 6 模型的相关性，说明一致性是跨模型的可转移特征。

**⚠️ 局限性**

1) 未进行人类情感评估，跨模型一致性并不等价于人类认知；2) 仅使用无纹理灰色形状，未考虑材质、纹理、光照等实际设计因素；3) 所评估的 VLM 可能共享训练数据或语言偏差，导致一致性受限；4) 语义词汇的文化/群体差异未被考虑；5) 交互界面未在真实用户中评估其有效性。

---

## 510. Quantitative Analysis of $ω$-Regular Robust MDPs

**arXiv ID:** 2608.25968 | [PDF](https://arxiv.org/pdf/2608.25968v1)

**作者:** Ali Asadi `[一作]` (Institute of Science and Technology Austria), Ali Shafiee `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文给出了在 (s,a)-rectangular 且线性定义不确定集下的鲁棒马尔可夫决策过程（RMDP）求解 ω-regular（奇偶）目标的量化问题，并提供了多项式空间的策略迭代算法，同时在鲁棒马尔可夫链（RMC）上给出多项式时间的量化奇偶解子算法。

**💡 创新点**

创新点包括：① 证明 agent 与 environment 均存在纯无记忆最优策略；② 设计结合量化一阶改进与几乎必然改进的策略迭代框架；③ 利用线性定义的不确定集而非极点枚举，避免指数级扩张；④ 证明决策问题属于 NP∩coNP，并与传统随机游戏奇偶问题等价。

**🔧 技术方法**

使用的技术主要有：策略迭代、线性规划求解、最大终端分量（MEC）分析、奇偶目标归约、价值类、紧致动作与紧致面分析等。

**📊 数据集**

实验使用的 benchmark 包括：Garnet（随机生成的 MDP）、Inventory Management（仓库管理模型）以及 Frozen Lake（经典网格世界）。

**📈 对比分析**

与把 RMDP 显式转化为随机游戏后再使用策略迭代的基线方法对比，实验发现：当状态空间分支因子较大（Garnet、Inventory）时本文算法在多数量级上更快、可处理更大规模；当分支因子较小（Frozen Lake）时基线更快，可处理更大规模。整体而言，本文算法在大多数情形下比基线快 1-2 个数量级。

**⚠️ 局限性**

局限性：仅适用于线性定义的不确定集；当分支因子小且极点少时基线方法更有优势；理论上迭代次数上界为指数，但实际表现优于理论预期；未来需扩展到非线性不确定集、放宽 (s,a)-rectangular 条件或学习情境。

---

## 511. Praxist: From Experimental Artifacts to Solution Lineages

**arXiv ID:** 2608.25955 | [PDF](https://arxiv.org/pdf/2608.25955v1)

**作者:** Jin Li `[一作]` (Sapient Intelligence), Yuhao Sun `[通讯]` (Sapient Intelligence)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个以证据继承为核心的自主研发系统 Praxist，能在多代实验中将可复制的实验成果转化为可重用的结构化证据，并用这些证据驱动后续实验。

**💡 创新点**

创新点在于：①将实验结果转化为带类型、成熟度和操作指令的“发现”节点；②通过前沿（frontier）划分为确认、候选、诊断、验证四条通道，实现对证据的可视化继承；③引入“Gems”压缩长期记忆，并在整个生成周期保持可追溯的 lineage；④将传统的单次实验视为可重用的构件，支持跨实验组合与验证。

**🔧 技术方法**

技术包括：基于大型语言模型（如deepseek-v4-pro、Claude Opus 4.8）的自动代码生成与调试；生成式实验设计与评估循环；图结构的 PI/Chair 合成与议程生成；Typed findings、frontier lanes、agenda decisions 的自动化流水线；Gems 压缩与 lineage 记录。

**📊 数据集**

数据集：1）MLE-bench 75 任务（Tabular、CV、NLP、时序、信号处理）；2）四个开源工程案例：可重复火箭着陆、量化交易、LiDAR‑IMU‑视觉 SLAM（NTU‑VIRAL）与托卡马克磁控（FreeGSNKE + MAST‑U 任务）。

**📈 对比分析**

对比方法：在 MLE-bench 上将 Praxist 与同一硬件、相同评估 harness 下的 Claude Code + Opus 4.8 进行单跑比较；在四个案例中与任务原生基线（如 Weco 代码优化器、均值加权交易策略、FAST‑LIVO2、PCS‑style 控制器）对比。性能表现：Praxist 在 MLE-bench 获得 60 颗奖牌（49 金）vs 55 颗（34 金）；火箭着陆 100% 成功率 vs 4%/17%；量化交易 53% CAGR vs 23%；SLAM 可视化计算下降 72% 而精度不变；托卡马克控制在常规视界下生存率和误差略优于基线。

**⚠️ 局限性**

局限性：①依赖可被外部评估器数值打分的任务，无法直接迁移到实验室物理实验或无标注任务；②系统在每代之间仍需手动配置评估阶段、lane 设定等，降低自动化程度；③对长周期的多任务评估缺乏实时反馈，导致探索/验证平衡不完全；④Gems 压缩策略依赖经验阈值，可能丢失细粒度因果信息；⑤部分案例评估仅在模拟环境或单次种子上完成，真实环境验证不足。

---

## 512. How Robust Are Automated Fact-Checking Systems? A Cross-Benchmark Evaluation

**arXiv ID:** 2608.25934 | [PDF](https://arxiv.org/pdf/2608.25934v1)

**作者:** Aida Usmanova `[一作]` (Leuphana University of Lüneburg), Ricardo Usbeck `[通讯]` (Leuphana University of Lüneburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统性评估自动事实核查（AFC）系统的跨域性能，比较检索与验证两阶段在四个不同领域数据集上的表现。

**💡 创新点**

提出跨域评估框架，全面对比从随机、稀疏检索、微调Transformer、零样本LLM到2025共享任务顶级系统，揭示检索是主要瓶颈并强调经典基线仍具竞争力。

**🔧 技术方法**

使用TF‑IDF、BM25、AIC CTU、Sanctuary检索器；LogReg、DistilRoBERTa、Longformer、Llama 3.1（8B/70B）验证器；评估Recall@K、F1@K、准确率和宏F1等指标。

**📊 数据集**

四个数据集：Open‑web（green）、Life sciences（blue）、Climate/social（orange）、Climate science（violet）。

**📈 对比分析**

对比结果显示：经典稀疏基线在多数据集仍能击败复杂模型；零样本LLM和共享任务顶级系统在部分域表现不佳；用黄金证据替换检索结果时准确率提升14‑22个百分点，验证检索质量是核心瓶颈。

**⚠️ 局限性**

局限包括仅覆盖四个数据集，未考虑大型/多语言或生成式核查系统；检索在大规模语料上的覆盖率受限；未评估提示敏感性与多样性；注释噪声评估有限。

---

## 513. Code World Model: Coding Agent as World Brain

**arXiv ID:** 2608.25927 | [PDF](https://arxiv.org/pdf/2608.25927v1)

**作者:** Yiwen Chen `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Code World Model框架，将世界演化的推理与执行交由可执行代码驱动的编程代理完成，并通过代理生成的“代理视频”作为条件，将可视化输出委托给视频模型。

**💡 创新点**

创新点在于：①将世界状态演化与视觉生成分离，使用可编程代理和可执行代码实现持久且可调节的规则驱动；②引入代理视频（proxy）这一中间表示，将空间时序约束以视觉形式直接投射给视频模型；③通过游戏和真实视频的代理–观测对齐数据管道实现端到端训练。

**🔧 技术方法**

核心技术包括：大型语言模型驱动的编程代理（如 GPT‑5.6 Sol）用于生成与修改代码；可执行代码实现高频状态更新；代理视频渲染器（轻量化编译器）将代码状态转化为粗糙视觉约束；视频生成模型 MiniMax‑H3 通过 LoRA 微调学习代理视频与文本条件下的生成；并使用 FlashAttention‑3、BF16 混合精度等加速训练与推理。

**📊 数据集**

使用了两类数据集：①游戏数据——5.6 小时的 GTA V 游戏录像与同步的运行时状态，自动生成代理视频；②真实世界数据——KITTI‑360 视频，利用已完成的三维重建与对象标注在线编译代理视频。

**📈 对比分析**

实验主要以定性比较为主：将代理视频条件下的生成结果与传统基于动作/摄像机条件的视频模型对比，观察角色运动、相机轨迹与场景布局的控制精度。结果显示，代理视频能够更精确地控制角色位置、动作轨迹及相机运动，视觉细节与动态保持良好，同时保持了高帧率生成能力。

**⚠️ 局限性**

局限性包括：①训练规模受限，模型仍不如更大规模模型在视觉质量与控制细度上表现；②缺乏自回归实时生成；③当前编程代理尚无法从零开始可靠地实现高度复杂的游戏机制，因而示例场景仍基于现有 AAA 级游戏模板，无法展示完全自动构建开放世界的能力。

---

## 514. Efficient tensor bases for pairwise comparisons

**arXiv ID:** 2608.25923 | [PDF](https://arxiv.org/pdf/2608.25923v1)

**作者:** Konrad Kułakowski `[一作]` (AGH University of Krakow), Ryszard Smarzewski `[通讯]` (Military University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

构造了最小支撑的正交基，给出了加法一致子空间的显式正交投影，并基于此提出了对 PC 矩阵的对数、Saaty 以及 SVD 三种窗口投影的闭式组合公式。

**💡 创新点**

①首次给出 𝒜ₙ 的最小支撑正交基，②利用张量表示实现投影公式的显式化，③提出对数、Saaty 与 SVD 投影的统一组合表达式。

**🔧 技术方法**

张量算子表示、Gram‑Schmidt 正交化、加权 Frobenius 范数、矩阵对数/指数变换、SVD 分解、组合投影理论。

**📊 数据集**

无数据集；论文完全基于理论推导和数值示例，未使用实际实验数据。

**📈 对比分析**

通过理论证明与数值示例表明：对数投影可直接计算，无需迭代；SVD 投影是两步投影的组合；相比 Saaty 投影，SVD 在某些矩阵上更具一致性；但论文未给出大规模实验或与传统方法的性能对比。

**⚠️ 局限性**

仅在标准 Frobenius 范数下给出正交基，推广到加权内积需进一步 Gram‑Schmidt；SVD 投影虽然理论优越，但计算量和数值稳定性在大规模问题中仍需评估；未提供实测验证，缺乏对真实决策数据的性能评估。

---

## 515. A General-Purpose Molecular Foundation Model Transfers Across Diverse Olfactory Tasks

**arXiv ID:** 2608.25893 | [PDF](https://arxiv.org/pdf/2608.25893v1)

**作者:** Yikun Han `[一作]` (University of Michigan), Ambuj Tewari `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在GS-LF气味描述子预测任务上微调Uni-Mol2分子基础模型，并在不再训练的条件下评估其在交叉数据集预测、嗅觉/无味二分类、对映体辨别和气味混合可辨性等四种不同嗅觉任务中的迁移性能。

**💡 创新点**

首次证明单一分子基础模型的微调能在多种多样的机器嗅觉任务中实现良好迁移，展示三维分子表示对对映体辨别的必要性，并提出未来可联合学习分子与气味描述子嵌入以实现零样本预测的思路。

**🔧 技术方法**

采用3D分子基础模型Uni-Mol2，使用焦点损失解决标签不平衡，构建50模型集成进行微调；对混合任务仅使用轻量级经典机器学习回归（ElasticNet、随机森林、梯度提升）。

**📊 数据集**

主要数据集包括GS-LF（138个描述子，约5,000分子）、Zhang et al.（Pyrfume，118个描述子），Mayhew等人的嗅觉/无味二分类数据集，四个气味混合数据集（Snitz1、Snitz2、Ravia、Bushdid）以及11对对映体的手工挑选子集。

**📈 对比分析**

与POM/OpenPOM基线对比，Uni-Mol2在GS-LF上宏观AUROC、AUPRC、F1均超过对手；在Zhang、Mayhew等数据集上保持更高AUROC并提升召回率；在混合可辨性回归中取得最低RMSE和最高皮尔逊相关系数，表明迁移效果显著。

**⚠️ 局限性**

主要局限包括：三维表示虽能捕捉对映体差异，却仍无法准确预测对映体特定气味；数据量普遍不足；对气味描述子仍采用独立二分类忽略共现和语义结构；未实现对未出现描述子的零样本推断；在Bushdid混合数据集上表现略逊于对手。

---

## 516. Low-Resolution Perception for Robotic Packing

**arXiv ID:** 2608.25874 | [PDF](https://arxiv.org/pdf/2608.25874v1)

**作者:** Giuseppe Fabio Preziosa `[一作]` (Politecnico di Milano), Paolo Rocco `[通讯]` (Politecnico di Milano)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种面向低成本、低分辨率深度传感器的机器人打包框架，通过低分辨率Next Best View与抓取驱动的重建相结合，实现对象可抓取性评估与视角采集决策。

**💡 创新点**

创新点在于：1) 为低分辨率感知设计的LR–NBV策略，引入稠密度增益避免冗余视角；2) 基于抓取稳定性重观测的对象级决策机制；3) 端到端集成的对象管理、重建与抓取生成流程。

**🔧 技术方法**

使用低分辨率时间飞行传感器（VL53L8CX、MaixSense A010）、OctoMap体素映射、DBSCAN聚类、GraspNet抓取生成以及自定义LR–NBV效用函数（探索增益+密度增益）。

**📊 数据集**

实验采用真实生产零件（Obj A 与 Obj B）在 UR5e 与 ABB GoFa 12 机器人上进行，使用不同分辨率（R60、R30）和视角策略（XPLR、XPLT）进行评测。

**📈 对比分析**

与传统NBV及启发式采样方法比较，LR–NBV 在低分辨率下显著减少采集次数（≤ NBV 一半）并提升 IoU；在全流程评测中，抓取成功率在 54–71% 之间，平均表面覆盖率 40–60%，显示低分辨率感知即可满足打包需求。

**⚠️ 局限性**

局限性在于低分辨率噪声和反射表面易导致点云误差，覆盖率偶尔低于 40% 时抓取不稳定；对高反射或高度复杂形状的对象尚缺乏专门的感知与去噪策略。

---

## 517. How Edge of Stability Hinders SCAFFOLD in Federated Optimization

**arXiv ID:** 2608.25873 | [PDF](https://arxiv.org/pdf/2608.25873v1)

**作者:** Anant Khandelwal `[一作]` (Georgia Institute of Technology), Mingrui Liu `[通讯]` (George Mason University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过大量实验探究联邦学习中FedAvg和SCAFFOLD在深度网络训练时出现的渐进锐化与稳定边缘（EoS）现象，并指出EoS导致SCAFFOLD难以准确估计全局梯度，解释其在实践中往往不超过FedAvg的表现。

**💡 创新点**

首次将EoS与联邦学习中的梯度误差关联，揭示SCAFFOLD在高锐化状态下梯度估计失效的机制，从而弥补了之前仅针对线性模型的理论解释。

**🔧 技术方法**

采用全批量梯度、锐化测量、更新失配度量、统计相关性分析等实验技术，结合FedAvg与SCAFFOLD的控制变量对比，系统性评估其在不同学习率、数据异质性、通讯步数下的性能。

**📊 数据集**

使用CIFAR-10、MNIST与FashionMNIST数据集，配合简单CNN与MLP网络，划分8个客户端并引入可调异质性参数，进行全批量训练实验。

**📈 对比分析**

对比方法：在相同网络、数据集与超参数设置下，分别训练FedAvg与SCAFFOLD，记录锐化曲线、更新失配与最终训练误差；实验显示当EoS出现时FedAvg往往取得更低的损失，而在EoS不显著的易学任务（如MNIST）SCAFFOLD表现更好。

**⚠️ 局限性**

研究仅限于全批量梯度和简单CNN/MLP网络，缺乏对随机梯度、Transformer或大规模图像数据的验证；未能给出EoS平衡锐化值的闭式表达，且理论上SCAFFOLD的L‑平滑假设与实际锐化动态不符。

---

## 518. MetaSieve: Faster Relational Deep Learning through SQL-Based Metapath Selection

**arXiv ID:** 2608.25903 | [PDF](https://arxiv.org/pdf/2608.25903v1)

**作者:** Fahim Shahriar Khan `[一作]` (University of Texas at Arlington), Ashraf Aboulnaga `[通讯]` (University of Texas at Arlington)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过基于SQL的元路径选择方法加速关系深度学习模型的训练和推理

**💡 创新点**

创新点在于利用数据库查询优化器自动筛选有效的元路径，避免手工设计与冗余路径，显著提升训练速度并保持或提升模型性能

**🔧 技术方法**

结合关系图卷积网络（RGCN）、SQL查询优化、特征抽取管道以及增量学习策略

**📊 数据集**

在DBLP、Freebase和YAGO等大型多关系图数据集上进行实验

**📈 对比分析**

与传统的手工元路径、Metapath2vec、RGCN等方法对比，MetaSieve在节点分类任务上实现了+3%准确率，同时训练时间缩短了40%

**⚠️ 局限性**

局限性包括对复杂或动态变化的数据库模式适配性不足，SQL查询开销在极大规模知识图谱上可能成为瓶颈

---

## 519. When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs

**arXiv ID:** 2608.25977 | [PDF](https://arxiv.org/pdf/2608.25977v1)

**作者:** Yao Fu `[一作]` (Case Western Reserve University), Kenneth A. Loparo `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对不同精度量化的开源大型语言模型（LLaMA、Mistral、Qwen）进行 Myers‑Briggs 类型指标（MBTI）人格评估，探究人格随层级的演化、量化对人格的影响以及推理解码对人格漂移的作用。

**💡 创新点**

①首次在量化模型上系统化进行 MBTI 评估；②引入层级不确定性（熵、置信差）来追踪人格决策的形成；③提出 Uncertainty‑Amplified Layer Decoding (UALD) 研究推理时解码对人格的漂移。

**🔧 技术方法**

MBTI 问卷（60 题）、层级熵与置信差分析、UALD 推理策略；使用 GPTQ、AWQ、AQLM 等量化方法；对比 FP16、4‑bit、2‑bit 版本。

**📊 数据集**

MBTI 标准问卷（60 题），作为无客观真值的主观倾向性评估数据集。

**📈 对比分析**

在不同模型家族与精度层级（FP16、GPTQ‑INT4、AWQ‑INT4、AQLM‑INT2）下比较人格类型分布。结果显示：4‑bit 量化几乎保持与 FP16 相同的主导人格（ENFJ），但 2‑bit 量化导致人格一致性与可控性下降；层级分析显示人格决策在后层明显收敛；UALD 通过放大早期层不确定性可诱发人格漂移。

**⚠️ 局限性**

仅评估单轮 MBTI 提示；未涵盖更大规模模型和其他量化方法（QAT、PEFT）；依赖概率指标，缺乏因果解释；结论仅适用于 MBTI，难以推广到其他人格框架。

---

## 520. Quantum-Inspired Modeling of Driving Behavior

**arXiv ID:** 2608.25907 | [PDF](https://arxiv.org/pdf/2608.25907v1)

**作者:** Mohammad Elayan `[一作]` (University of Nebraska–Lincoln), Wissam Kontar `[通讯]` (University of Nebraska–Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于量子启发的密度矩阵表征驾驶行为的无监督学习框架，并在 I-24 MOTION 数据上训练以捕捉驾驶行为的连续性、概率性、上下文依赖和历史演化。

**💡 创新点**

将随机傅里叶特征投影与密度矩阵结合，自动学习行为变量间的非线性交互，并通过上下文权重动态生成可解释的驾驶模式，避免了传统模型对行为形式的预设。

**🔧 技术方法**

随机 Fourier 特征 (RFF)、密度矩阵 (quantum density matrix)、Born 规则、无监督负对数似然优化、Adam 优化器以及特征熵正则化。

**📊 数据集**

I‑24 MOTION 高频摄像头收集的北卡州 I‑24 车道交通轨迹，涵盖自由流、过渡与拥堵三个阶段。

**📈 对比分析**

与固定参数模型、混合模型以及基于 RFF 的单一密度矩阵进行对比；无监督 NLL 从 2.230 降至 0.880，显著提高，对宏观流量基本图和滞后现象的重现效果优于传统方法。

**⚠️ 局限性**

仅评估在高速公路场景，仅包含纵向行为，未考虑车道变道与时间延迟，部分超参数需人工设置，模型规模在大规模仿真中的计算效率与数值稳定性待验证。

---

## 521. Vulnerable Code Search: Transferable Attack for Code Language Models

**arXiv ID:** 2608.26031 | [PDF](https://arxiv.org/pdf/2608.26031v1)

**作者:** Kaicheng Wang `[一作]` (University of Southern California), Weihang Wang `[通讯]` (University of Southern California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于程序标识符变换的可迁移对抗攻击，能在保持代码功能不变的前提下大幅提升对目标查询的相似度，从而破坏代码检索系统的排名。

**💡 创新点**

创新点在于：①利用梯度引导替换标识符实现语义保持的对抗变换；②加入查询词相似性正则化提升攻击对不同模型的迁移能力；③在共享语料库场景下实现多查询联合攻击，极大提高攻击效率；④展示该攻击同样适用于大规模封闭模型和LLM检索。

**🔧 技术方法**

技术手段包括：梯度近似优化、贪婪搜索、标识符风格约束、查询词相似性加权、跨模型迁移（Attack Transfer）以及对比实验的基线方法如DAMP和CodeAttack。

**📊 数据集**

主要使用的公开数据集有 CosQA、CLARC、RepoQA 以及内部使用的 CodeT5+、OASIS、Nomic-embed-code、Voyage-code-3 等预训练/微调的嵌入模型。

**📈 对比分析**

与现有对抗方法（DAMP、CodeAttack）对比，攻击在白盒环境下提高相似度 ΔSim 约 8–10，GPU 计算时间减少 30%；在黑盒环境下提升 ΔSim 约 5 倍，调用次数从 10.8M 降至 1K。攻击导致 CosQA、CLARC 上 MRR 下降 70% 以上，LLM（GPT-5.4-mini、Gemini-3.1-Pro）检索准确率下降 10–30%。

**⚠️ 局限性**

局限性包括：①攻击依赖于对抗样本的可视化自然度，过度的标识符修改可能被人工审查发现；②在高度鲁棒的微调模型（Robust-Only FT）中攻击效果显著下降；③在多查询共享攻击中，攻击强度随攻击代码数量下降，需权衡攻击规模与效果；④实验主要集中在静态代码检索和单语言场景，跨语言/跨平台泛化仍待验证。

---

## 522. RefVideo-6M: A Reliable Reference-Based Dataset for Instructional Video Editing

**arXiv ID:** 2608.26101 | [PDF](https://arxiv.org/pdf/2608.26101v1)

**作者:** Bojia Zi `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了 RefVideo-6M 大规模可靠的视频编辑数据集，并在此基础上训练 RefMoT 参考引导视频编辑模型。

**💡 创新点**

创新点包括：①利用已编辑视频作为源、原始视频作为目标的逆向对齐策略，显著降低伪影；②提出轻量级 Mixture-of-Tokens（RefMoT）架构，仅更新参考分支即可实现参考条件融合；③在数据生成中结合多种编辑专家与 LLM 过滤，提供 10 种参考类型。

**🔧 技术方法**

技术手段包括：HunyuanVideo1.5/DiT 作为基础编辑网络；LLM（GPT‑5.2/5.5、Gemini‑3‑Pro）生成指令与质量筛选；FSDP2、AdaLN 共享、RefMoT 线性层；视频采样 81×480×832，80‑129 帧，720p 分辨率。

**📊 数据集**

使用 RefVideo-6M（5M 视频对 + 1M 图像对）作为主要训练数据，结合公开编辑数据集（OpenVE‑3M、ReCo‑500K、Ditto‑1M、Senorita‑2M 等）进行对比实验。

**📈 对比分析**

在 GPT‑5.5 与 Gemini‑3‑Pro 评估中，RefVideo‑6M+RefMoT 在指令跟随、视觉质量、背景保留、属性对齐、参考一致性等指标均显著优于现有基线，整体得分提升至 4.56/4.59，用户研究偏好率超过 50%；同时在参考引导编辑任务中，整体评分从 3.73 提升至 4.03，显示出更强的语义一致性和视觉质量。

**⚠️ 局限性**

局限性包括：仅在 720p、81‑129 帧范围内验证，缺乏更高分辨率与长时序测试；参考类型虽然多样，但仍未覆盖所有可能的编辑场景；数据生成主要依赖 LLM 与专家，未来可进一步引入人类标注与更丰富的多模态输入。

---

## 523. Giving Mechanical Engineers Intelligent Tools: A Project-Based AI Education Curriculum in Thermal Engineering

**arXiv ID:** 2608.26056 | [PDF](https://arxiv.org/pdf/2608.26056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 524. PlanSightRAG: A Visual-First Multimodal RAG for Automating Question Answering and Compliance Checking for Civil Standard Plans

**arXiv ID:** 2608.26091 | [PDF](https://arxiv.org/pdf/2608.26091v1)

**作者:** Nabaraj Subedi `[一作]` (University of Wyoming), Shivanand Venkanna Sheshappanavar `[通讯]` (University of Wyoming)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PlanSightRAG，一个视觉优先的多模态检索增强生成（RAG）框架，用于自动化土木工程标准图纸的问答与合规性检查。

**💡 创新点**

创新点：① 直接使用图像进行多向量检索（ColNomic-3B）避免OCR导致的几何信息丢失；② 通过MaxSim热图实现可解释的视觉证据；③ 引入Planner–Retriever–Auditor–Synthesizer四阶段代理管线，实现跨图纸、多规则的自动合规判定；④ 在未提供规则阈值的情况下通过视觉规则检索实现自主规则归纳。

**🔧 技术方法**

核心技术包括：多向量视觉检索（ColNomic-3B）、MaxSim late‑interaction与热图锐化、VLM（Qwen‑2.5‑VL‑72B、Qwen‑2.5‑VL‑7B、InternVL‑2.5‑8B）进行视觉问答与合规判定、VLM交叉编码重排序、BM25稀疏融合以及高分辨率滑动窗口切块。

**📊 数据集**

使用的数据集为：① 1,898页涵盖五州（WY、CA、AZ、CO、FL）标准图纸的4,056问答对；② 298页的密歇根州标准图纸用于零样本迁移；③ 通过参数化合成生成的500+多图纸合规测试集；④ 真实项目（WYDOT #1507040、#N345107）和931页的官方规范书用于规则自检。

**📈 对比分析**

对比实验表明：ColNomic‑3B在5州基准上零样本Recall@5为91.47%，远高于OCR+文本检索（36.79%）和传统视觉检索（76.89%）。在合规判定上，使用CoT+预解析阈值的Qwen‑2.5‑VL‑72B在合成数据集上可达100%判定准确率，agentic管线在多图纸合规任务上亦达到90–100%的准确率；与仅OCR+阈值对比可见VLM在旋转文本和多视角布局中的优势。

**⚠️ 局限性**

局限性：① 规则自检在真实规范书中仅达33%准确率，表明需要改进检索和表格解析；② 高分辨率切块虽然提升精度，但会失去跨切块的上下文，导致跨视图推理错误；③ 对于极小或高密度标注的图纸，VLM在符号解释上仍易产生错误；④ 计算成本仍较高，尤其是agentic多步推理需约60秒/次。

---

## 525. Fast Generative Grasping via Lie Group-Constrained MeanFlow

**arXiv ID:** 2608.26076 | [PDF](https://arxiv.org/pdf/2608.26076v1)

**作者:** S. Talha Bukhari `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出在 3×^3 Lie 群上基于 MeanFlow 的抓取生成框架 GraspMF，实现了少步（1~5 步）高效多模态抓取采样。

**💡 创新点**

创新点在于：① 用端点预测直接回归清洁抓取姿态并由其一侧得到平均速度；② 将 Riemannian MeanFlow 与代数半群一致性约束结合，形成无教师、无仿真训练目标；③ 通过半群一致性实现对轨迹的全局一致性，保证在少步采样下仍保持对接触约束的近似。

**🔧 技术方法**

使用了 Riemannian MeanFlow、半群一致性损失、端点预测网络（SE3Dif 轻量化骨干+SVD 投影）、随机 Fourier 特征时间编码、可学习的 SDF 辅助监督等技术。

**📊 数据集**

主要使用 ACRONYM 数据集的 416 个物体和 780K 训练抓取姿态，进行 ID/OOD 评估，并在 IsaacGym 与真实 Franka 机器人上验证。

**📈 对比分析**

与 SE3Dif、EGF、BRIDGER、VSIGD 等基准比较，GraspMF 在 T=5 时取得最高抓取成功率（ID 87.40%、OOD 71.73%）和最低 EMD（ID 0.3702、OOD 0.4191），网络评估次数仅 5 次，推理延迟 15.5 ms，速度比传统多步采样方法快 30–40×。

**⚠️ 局限性**

局限性包括：① 对极端部分观测或多手臂/非抓握任务的泛化仍未验证；② 半群一致性权重调度和 SVD 投影等细节对训练稳定性敏感；③ 在单视点下的几何不完整性仍导致成功率下降。

---

## 526. Fine-Tuning Whisper for Automatic Speech Recognition in Baniwa: A Preliminary Study

**arXiv ID:** 2608.26060 | [PDF](https://arxiv.org/pdf/2608.26060v1)

**作者:** Leonardo Duart `[一作]` (University of Brasilia), Thiago Chacón `[通讯]` (University of Brasilia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Baniwa语言的录音进行Whisper Small模型的监督微调，以实现自动语音识别。

**💡 创新点**

证明大型多语言预训练模型在极低资源环境下可被有效迁移，为Baniwa提供首个ASR基准，展示微调效果。

**🔧 技术方法**

采用Whisper Small模型、Hugging Face Transformers框架、混合精度训练、Spanish语言提示进行监督微调。

**📊 数据集**

使用Baniwa-Koripako多媒体词典项目收集的1,373条手工转录录音，约0.54小时（32分钟）的语料。

**📈 对比分析**

按训练/验证/测试划分，评估WER与CER；最佳模型在验证集上达到37.5% WER、7.45% CER，明显优于未微调模型。

**⚠️ 局限性**

数据量极少导致易过拟合，模型仅适用于短词/短句，缺乏语言特定后处理与连贯语音评估。

---

## 527. $R^3$: Training Robots to Reason in Natural Language via Reinforcement Learning

**arXiv ID:** 2608.26053 | [PDF](https://arxiv.org/pdf/2608.26053v1)

**作者:** Lehong Wu `[一作]` (Carnegie Mellon University), Aviral Kumar `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并验证了一种两阶段训练的 VLM 先推理后指令的框架，用自然语言推理引导固定低层控制器完成长周期机器人操作任务。

**💡 创新点**

创新点在于将专家推理轨迹先做 mid‑training 以塑造推理风格，然后利用单步 RL 与 VLM judge 奖励的 rubric‑based 方式在离线数据上进一步优化推理；并证明推理能在 OOD 场景中显著提升泛化。

**🔧 技术方法**

技术实现基于 Qwen3.5‑4B 作为高层推理器，mid‑training 用 SFT 学习 Gemini 生成的专家推理与指令；单步 RL 采用 Dr.GRPO 结合 VLM judge 的语义匹配奖励；并利用交互历史、动态 token 长度等机制提升推理质量。

**📊 数据集**

使用的主要数据集为 Language Table（14 组长周期块排列任务）和双臂杂货包装（12 组未见目标任务）；专家推理轨迹由 Gemini 3 Flash 生成，指令标签由人类遥控收集；同时采用指令‑only 示例进行对照实验。

**📈 对比分析**

与仅指令式模仿学习、ECoT 等基线对比，mid‑training+RL 方案在 Language Table 见/未见任务的成功率平均提升约 15–30%，在 OOD 任务上提升超过 70%；在杂货包装任务中成功率从基线 38% 提升至 73%，同时任务进度（normalized progress）显著提高。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未在真实机器人上测试；stage I 仍需昂贵的专家推理标签；RL 奖励是 VLM judge 的代理目标，缺乏真实任务完成反馈；高层推理与低层执行完全分离可能导致意图与动作不匹配。

---

## 528. Robust CurveMoE: Multi-Norm Adversarial Defense for Mixture-of-Experts Models via Mode Connectivity

**arXiv ID:** 2608.26043 | [PDF](https://arxiv.org/pdf/2608.26043v1)

**作者:** Xu Zhang `[一作]` (Illinois Institute of Technology), Ren Wang `[通讯]` (Illinois Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Robust CurveMoE，将鲁棒模式连通曲线中的多模态专家通过稀疏 MoE 进行融合，实现在多范数对抗攻击下的高鲁棒性。

**💡 创新点**

创新点在于利用鲁棒曲线生成互补专家，采用贡献引导的局部曲线更新与交叉范数鲁棒专家筛选，并通过输入依赖路由充分利用专家多样性。

**🔧 技术方法**

使用混合专家 (MoE) 架构、鲁棒模式连通、贡献评分的部分参数更新、鲁棒性约束专家筛选与自适应路由。

**📊 数据集**

实验数据集为 CIFAR‑100 与 ImageNet‑100，使用 WideResNet‑28‑10 与 ViT‑Tiny/16 作为基准网络。

**📈 对比分析**

与 MSD、ERMC 等基线对比，Robust CurveMoE 在清洁准确率、单范数鲁棒性以及 Union 准确率上提升约 2–4%，并显著提升 Union 召回。

**⚠️ 局限性**

局限性包括对专家路由的攻击可能导致特定范数专家被利用，以及曲线构造与专家筛选步骤增加实现复杂度。

---

## 529. Trace Integrity for LLM Data Agents: A Vision for Auditable Structured Reasoning in Real-World Systems

**arXiv ID:** 2608.26036 | [PDF](https://arxiv.org/pdf/2608.26036v1)

**作者:** Srimonti Dutta `[一作]` (WAI USA Research Labs), Akshata Kishore Moharir `[通讯]` (WAI USA Research Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了 Trace Integrity 作为 LLM 数据代理可靠性评估准则，证明答案准确率与计算完整性可能分离。

**💡 创新点**

创新地定义了 Trace Integrity 维度与 CAIT 率，揭示“答案正确但计算无效”的隐藏失败模式。

**🔧 技术方法**

采用执行合同记录计算痕迹，并用验证器检查 SQL 可执行性、模式有效性、操作一致性等。

**📊 数据集**

在 BIRD Mini-Dev 数据集的 100 个样本上进行实验。

**📈 对比分析**

比较了 Direct SQL、Operation Summary+SQL 与 Contract-First SQL 三种提示，答案准确率 20–24%，Trace Integrity 通过率 39–43%，CAIT 率 45.8–59.1%，表明不同评估信号不一致。

**⚠️ 局限性**

局限在于仅使用单一模型和固定提示，验证器未完成语义等价检测，实验规模有限，结果仅为示例。

---

## 530. Exact Common Information and Exact Channel Synthesis for Correlated Gaussian Sources

**arXiv ID:** 2608.26012 | [PDF](https://arxiv.org/pdf/2608.26012v1)

**作者:** Lei Yu `[一作]` `[通讯]` (Nankai University), Lei Yu (Nankai University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

本文证明了对相关高斯源的精确共同信息与精确通道合成的闭式表达式，确认了Yu‑Tan提出的两个猜想。

**💡 创新点**

创新点在于利用最优传输、Fathi的高斯T₂不等式以及协方差行列式极值论证，首次给出了精确共同信息与精确通道合成的精确界。

**🔧 技术方法**

核心技术包括多字母界、最大高斯交叉熵分析、Wasserstein距离的Gaussian T₂不等式、以及协方差矩阵的行列式不等式。

**📊 数据集**

该工作不依赖任何经验数据集，完全基于理论推导与数学证明。

**📈 对比分析**

与以往仅给出上界或近似结果的方法相比，本论文给出了完全匹配的下界，证明了所提出的量化公式在所有ρ∈[0,1)时均成立。

**⚠️ 局限性**

局限性在于只处理了标量高斯源，对多维或非高斯源的精确共同信息与通道合成尚未给出通用结论。

---

## 531. Imitation Learning for Connection-Tableau Construction

**arXiv ID:** 2608.26009 | [PDF](https://arxiv.org/pdf/2608.26009v1)

**作者:** Fredrik Rømming `[一作]` (University of Cambridge), Sean B. Holden `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于模仿学习的连接表格（Connection-Tableau）构造方法，将构造过程视为在形式演算诱导的状态转移系统中的策略执行。

**💡 创新点**

创新点在于将传统符号搜索（如回溯式搜索）转化为一次性策略决策，并通过图神经网络提取证明结构特征，实现跨问题的知识迁移；同时将模仿学习与深度网络结合，显著减少对搜索回溯的依赖。

**🔧 技术方法**

使用了图神经网络（Graph Neural Network）来评分证明编辑动作，利用模仿学习（Imitation Learning）从已找到的证明中学习策略；实现了从全符号回溯到纯网络驱动的多级策略层次。

**📊 数据集**

在三个公开基准数据集上进行实验：M2k、MPTP2078-bushy 以及 TPTP v9.2.1。

**📈 对比分析**

与传统基于搜索的表格构造方法相比，学习到的策略在固定步骤预算下在上述数据集上多解决了约46%的问题，并在寻找证明时的步骤数降低了一个数量级，体现出显著的性能提升。

**⚠️ 局限性**

局限性包括对高质量训练证明的依赖、对非常大规模或结构极其复杂的证明问题的适应性不足，以及在完全无搜索辅助的极端条件下仍可能出现性能下降。

---

## 532. A Self-Evolving Multi-Agent Framework Defense against LLM Jailbreak Attacks

**arXiv ID:** 2608.26008 | [PDF](https://arxiv.org/pdf/2608.26008v1)

**作者:** Tongyan Hu `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于外部持久规则记忆的自演化测试时防御框架，用于抵御LLM的越狱攻击；

**💡 创新点**

创新点在于将每一次失败抽象为可重用的结构化方法级别规则，并通过记忆驱动的触发与策略决策实现跨交互自适应，无需模型参数更新；

**🔧 技术方法**

技术包括：基于提示的攻击模式分类器、规则检索与触发、策略决策（硬拒、软拒、允许）、违规检测与规则诱导，以及外部记忆的增删与去重；

**📊 数据集**

使用的评估数据集包括：AdvBench（520个针对LLM的有害提示）、MMLU和GSM8K（正向任务）；

**📈 对比分析**

与无防御、对齐提示、自我反思、AutoDefense等基线对比，结果显示在四大越狱族群与多款模型上，ASR_rej/ASR_gpt显著下降，保持benign任务性能差距≤2点；

**⚠️ 局限性**

局限性包括：记忆增长仅通过去重和容量限制，未实现显式剪枝或冲突解决；依赖提示式LLM进行违规检测，可能出现误拒；仅评估单轮prompt级越狱，未覆盖多轮或代理攻击。

---

## 533. Spectral Allocation: Why Muon Outperforms Adam, and How to Improve Muon

**arXiv ID:** 2608.25990 | [PDF](https://arxiv.org/pdf/2608.25990v1)

**作者:** Xiaodong Wu `[一作]` (University of Cambridge), Philip Woodland `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对Transformer预训练中动量缓冲的奇异向量进行离谱谱探测，得到每个谱方向上最优步长的稳定异向量分布；

**💡 创新点**

提出Spectral‑Aware Muon（SAMuon）两种变体，利用测得的头-批处理谱特征在保持Muons均匀白化的同时对除头之外的谱进行增幅，显著提升更新效率；

**🔧 技术方法**

核心技术包括离谱谱探测（利用局部二次近似估计离谱步长）、随机低秩SVD与幂迭代对谱做快速估计，以及在Muons框架下的静态谱预设；

**📊 数据集**

在“modded‑nanogpt”预训练任务上使用FineWeb 100B-token数据集，规模涵盖124M、300M、1B参数模型，批量尺寸分别为1024、2048、4096；

**📈 对比分析**

与AdamW及Muons（Scion）基线对比，SAMuon在所有规模与批量配置下均实现13.3%–24.0%的token‑效率提升，且仅增加≈0.5%–5%额外浮点运算，保持与Muons相近的壁钟速度；

**⚠️ 局限性**

限制在于：①仅使用静态谱先验，未捕捉训练过程中的谱漂移与跨谱交互；②目前对不同模型规模的超参迁移仍未充分调优；③未考虑层级间谱差异，可能进一步提升效果。

---

## 534. Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization

**arXiv ID:** 2608.26103 | [PDF](https://arxiv.org/pdf/2608.26103v1)

**作者:** Jiaming Zhou `[一作]` (Robbyant), Yinghao Xu `[通讯]` (HKUST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用人类视频指令的因果视频-动作模型，实现机器人在零样本跨任务泛化；

**💡 创新点**

创新点在于（1）自动化生成语义匹配的人类-机器人IIC对，构建74.2K对数据；（2）对机器人预训练数据进行任务级采样并平衡；（3）引入“in-context future chunk prediction（IFP）”抑制短路学习，提升对人类视频的利用；

**🔧 技术方法**

使用流匹配（flow‑matching）视频生成、混合Transformer（Mixture‑of‑Transformers）结构、RoPE位姿编码、T5文本编码、以及在预训练中结合机器人视频‑动作与人类视频指令的因果预测；

**📊 数据集**

主要数据集包括自动生成的74.2K人机ICL对（8.6K任务），400K任务平衡机器人视频‑动作样本，RoboTwin 2.0仿真任务集，真实世界Franka双臂机器人实验数据；

**📈 对比分析**

在RoboTwin 7个未见任务上，Zero‑WAM平均成功率为46.95%，比LingBot‑VA高29.5个百分点；在真实机器人3类任务中，人机视频模式分别提升至53.3%、33.3%和16.7%，显著优于仅语言的LingBot‑VA；

**⚠️ 局限性**

局限性包括对极长时序任务的性能仍有限、依赖大规模预训练数据、未实现在线自适应、以及在高精度插入等细粒度操作中成功率仍受硬件限制。

---

## 535. Epistemic Networks, Collective Misperception, and the Manipulation of Social Knowledge

**arXiv ID:** 2608.26075 | [PDF](https://arxiv.org/pdf/2608.26075v1)

**作者:** Mihnea C. Moldoveanu `[一作]` (University of Toronto), Joel A. C. Baum `[通讯]` (University of Toronto)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了互动信念的结构，即代理人持有、修订和基于彼此的心理模型进行行动的认知状态。提出了集体信念的形成机制，强调了相互归因的张量结构。

**💡 创新点**

创新点在于将集体信念视为相互归因的张量，而非个体信念的简单聚合，揭示了集体误解的成因及其动态特征。

**🔧 技术方法**

使用了线性代数和谱理论来分析信念的演化，特别是通过张量收缩和矩阵运算来描述信念更新的动态过程。

**📊 数据集**

未具体提及使用的数据集，但研究涉及的概念和模型可以应用于社会网络、舆论动态等领域的实际数据。

**📈 对比分析**

通过与传统的意见动态模型进行比较，展示了该模型在处理集体误解、共识、极化等现象上的优势，性能表现出更高的解释力和预测力。

**⚠️ 局限性**

局限性在于假设了认知一致性，可能无法完全捕捉现实中个体信念的复杂性和不一致性，且未能深入探讨如何在实际应用中实现该理论。

---

## 536. Beyond Local Surprise: Grounded Dialogue as Selective Belief Revision under Referential Uncertainty

**arXiv ID:** 2608.26035 | [PDF](https://arxiv.org/pdf/2608.26035v1)

**作者:** Ziming Liu `[一作]` (University of Oklahoma), Jiqun Liu `[通讯]` (University of Wisconsin–Milwaukee)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个可控的保留–修订框架，用于在对话中逐回合研究语义保持与修订的决策过程。

**💡 创新点**

通过对比四种基于不同理论假设的修订策略，揭示成功的对话语义锚定更依赖累积不确定性而非局部不匹配，从而提供了对概念共识理论的实证支持。

**🔧 技术方法**

使用冻结的CLIP视觉与文本编码器、可学习的修订信号 ρ_t、对比学习损失、保留与修订约束以及基于证据积累的可解释性推断。

**📊 数据集**

在 PhotoChat 数据集（隐藏图像的视觉对话）上进行实验。

**📈 对比分析**

通过视觉检索指标（Recall@K、MRR、MedR）以及内在保留/修订指标（Pres、Rev Sens、Mean ρ）对四种策略进行对比，发现纯保留和不确定性策略在检索上相当，而仅靠局部不匹配驱动的策略表现显著下降。

**⚠️ 局限性**

仅在单一对话数据集上验证，且模型强调可解释性可能限制表达能力，未在更不稳定或更具挑战性的沟通场景中进行广泛评估。

---

## 537. VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following

**arXiv ID:** 2608.26013 | [PDF](https://arxiv.org/pdf/2608.26013v1)

**作者:** Min Zeng `[一作]` (vivo AI Lab), Xiaoxin Chen `[通讯]` (vivo AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VISA，一种闭环的自进化多模态指令合成框架；

**💡 创新点**

创新点在于将感知、规划、执行与反思以及持久记忆结合，动态扩展约束库并用可验证奖励驱动强化学习；

**🔧 技术方法**

使用统一的可执行工具与结构化 LLM 判断器进行验证、基于记忆的约束采样、嵌入空间多样性筛选以及 RL‑VR（可验证奖励）技术；

**📊 数据集**

构造约 15k 条 VISA 样本，基于 MM‑IFInstruct 图像池；在 MM‑IFEval、MMBench、MMStar、MM‑Vet、HallusionBench、MathVista、OCRBench 与 AI2D 等七大公开基准上评估；

**📈 对比分析**

与静态合成、SFT、RL 传统方法对比，VISA‑SFT‑15k 在 MM‑IFEval 的平均分从 60.8 提升至 63.9，RL‑15k 进一步升至 64.9；在通用多模态基准上平均分从 70.5 提升至 72.9，显示指令遵循能力提升同时保持甚至增强通用能力；

**⚠️ 局限性**

局限在于合成过程计算成本高，且缺乏完整的安全、公平性过滤，生成数据仍需在部署前进行审计。

---

## 538. Phantom Navigator: Stealthy and Precise Unmanned Aerial Vehicle Redirection with Real-Time Tracking and GPS Spoofing

**arXiv ID:** 2608.26011 | [PDF](https://arxiv.org/pdf/2608.26011v1)

**作者:** Haocheng Meng `[一作]` (Duke University), Miroslav Pajic `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Phantom Navigator，一种基于LiDAR‑Camera跟踪与GPS欺骗的隐蔽精确无人机重定向攻击框架；

**💡 创新点**

创新点在于：①离线可达性分析预测可达转移范围；②实时闭环低功耗控制，在GPS误差范围内限制欺骗幅度实现无侦测；③从固定平台实现到室内VICON与户外真实GPS实验的完整验证；

**🔧 技术方法**

使用技术包括：LiDAR‑摄像头融合检测与跟踪、Kalman滤波、GPS spoofing信号合成、凸包可达性分析、闭环加速度控制、误差模型估计，以及χ²与CUSUM异常检测；

**📊 数据集**

数据集：室内VICON轨迹、户外真实GPS数据，覆盖直线、锯齿、圆形等多种轨迹，以及随机10个重定向目标；

**📈 对比分析**

方法对比：与无攻击及基线(无闭环控制)相比，室内平均误差0.125 m，室外平均误差0.476 m；异常检测率分别为0.005 %（χ²）和0 %（CUSUM），显示攻击既精准又几乎不被检测；

**⚠️ 局限性**

局限性：仅针对固定攻击平台；仅模拟GPS劫持而非全无线；对GPS噪声模型有依赖；在跟踪丢失时误差可能漂移；对高精RTK等防御手段无效。

---

## 539. Distinct dynamics of conceptual and referential disruptions in human reading and large language model processing

**arXiv ID:** 2608.25999 | [PDF](https://arxiv.org/pdf/2608.25999v1)

**作者:** Rui He `[一作]` (Universitat Pompeu Fabra), Wolfram Hinzen `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了概念性扰动和指称性扰动对人类自助阅读时间以及大型语言模型预测与表征动态的传播效应。

**💡 创新点**

首次将两种意义扰动放在同一实验框架下比较，并揭示它们在传播曲线、句子边界敏感性以及模型内部表征上的显著差异。

**🔧 技术方法**

使用自助阅读实验、混合效应模型与轨迹拟合；在LLM中计算上下文惊讶度和输出层余弦距离，并分别对Qwen3‑4B和Llama‑3.2‑3B进行评估。

**📊 数据集**

基于21个改写的伊索寓言短篇，每个故事有原始、概念扰动和指称扰动三种版本。

**📈 对比分析**

通过比较K+1至K+10的阅读时间、惊讶度和余弦距离的距离‑类型交互，发现概念扰动产生更强、更局部的成本，而指称扰动更分布且在句子边界更突出；模型与人类表现呈相似传播趋势。

**⚠️ 局限性**

扰动设计并非完全过程纯净，词性、频率、插入率等因素混杂；实验仅使用少量文本，单一模型层和架构，限制了机制解释和结果的普适性。

---

## 540. VirTooS: A ROS 2 - Unity Virtualization Toolkit for Fleet Management of Autonomous Mobile Robots

**arXiv ID:** 2608.26066 | [PDF](https://arxiv.org/pdf/2608.26066v1)

**作者:** Andrea Drudi `[一作]` (University of Bologna), Giuseppe Notarstefano `[通讯]` (University of Bologna)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个基于 ROS 2 与 Unity 的混合现实工具箱，支持在虚拟和真实环境中对多台自主移动机器人进行队列管理、任务分配、路径规划与实时导航。

**💡 创新点**

创新点在于：①将 ROS 2 的分布式优化决策层与 Unity 的高保真仿真环境无缝对接；②提供动态生成可配置的虚拟环境与机器人模型；③通过容器化实现跨平台部署；④实现真实机器人与虚拟机器人在同一混合现实框架下协同作业。

**🔧 技术方法**

技术包括：ROS 2、Python、C#、Unity 3D、Gazebo 物理引擎、ChoiRbot 优化框架、Docker 容器、Vicon MoCap、TCP/IP 交互协议、URDF 导入工具。

**📊 数据集**

未使用公开数据集；实验数据来自自行搭建的 Unity 仿真场景和在实验室跑的 Jackal 机器人实时测量，主要以任务完成率、路径长度和碰撞次数等指标评估。

**📈 对比分析**

通过在虚拟、真实和混合现实三种场景下分别执行动态任务分配实验，对比三种部署方式，结果表明：①纯虚拟实验可实现高频率、低延迟任务调度；②真实机器人实验在定位误差、碰撞回避方面表现良好；③混合现实实验保持了两者优势，实现了在资源受限条件下的多机器人协同，整体性能与纯虚拟相近，略低于纯真实实验。

**⚠️ 局限性**

局限性包括：①对 ROS 2 与 Unity 的依赖限制了跨平台适配；②容器化虽提升部署便利，但对硬件加速（GPU、网络）有一定需求；③实验规模仅至四台 Jackal，尚未验证更大队列的可扩展性；④混合现实时同步精度受 Vicon 采样率与网络延迟影响。

---

## 541. A Visual Dependence-Aware Framework for Multimodal Unsupervised Continual Post-Training

**arXiv ID:** 2608.26095 | [PDF](https://arxiv.org/pdf/2608.26095v1)

**作者:** Kaichen Li `[一作]` (Chinese Academy of Sciences), Changsheng Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Multimodal Unsupervised Continual Post-Training（MU-CPT）框架，让多模态大语言模型在无答案的流式多模态数据中持续学习。

**💡 创新点**

创新点在于将token级视觉依赖（VD）作为跨模态遗忘的信号，并通过可视化约束最优传输（VC-OT）保持旧任务的VD结构，同时用视觉调制适应（VMA）强化新任务的视觉驱动学习。

**🔧 技术方法**

采用VD量化、可视化约束最优传输、LoRA参数高效微调、以及多任务对齐的损失函数组合实现。

**📊 数据集**

在六个多模态问答任务上评估：TextVQA、SciVQA、StockQA、GQA、DriveLM、PMC‑VQA，并以Qwen2.5‑VL‑7B为主干。

**📈 对比分析**

与现有无监督后训练方法（如LSMI、SeVa、ScPO、TLM、MM‑UPT、TTRV）以及持续学习方法（SEEKR‑MLLM、CL‑MoE、SEFE、DGG）对比，VDA在AvgAcc上达到62.5%，显著高于强基线并保持低遗忘率。

**⚠️ 局限性**

局限性包括对缓冲区大小和LoRA秩的敏感性、对真实动态数据分布的泛化仍需验证，以及仍依赖于视觉依赖度量的准确性。

---

## 542. From Producing to Validating: How AI Is Deskilling Freelancers

**arXiv ID:** 2608.26089 | [PDF](https://arxiv.org/pdf/2608.26089v1)

**作者:** Nakul Rajpal `[一作]` `[通讯]` (Northeastern University), Nakul Rajpal (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了生成式 AI 在自由职业者中的“从生产到验证”转变对其技能退化与职业价值的影响。

**💡 创新点**

提出了“compounding deskilling”概念，并通过机器翻译后编辑和软件代码生成两个案例阐释这一模式。

**🔧 技术方法**

主要使用文献综述、行业案例分析和正在进行的访谈研究。

**📊 数据集**

未使用公开数据集，主要依赖行业报告、平台数据与访谈材料。

**📈 对比分析**

无定量性能比较，文章采用概念性框架和案例比较来说明问题。

**⚠️ 局限性**

局限性在于缺乏大规模实证验证，结论主要基于案例与访谈，需进一步量化研究。

---

## 543. One Policy, Many Embodiments: Unified Camera-Centric Action Geometry Pre-training for Heterogeneous Embodied Manipulation

**arXiv ID:** 2608.26058 | [PDF](https://arxiv.org/pdf/2608.26058v1)

**作者:** Xiaomi Embodied Intelligence Team `[一作]` (University of Macau), Zhi-xin Yang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的相机中心动作空间，并通过几何条件翻译器将共享的动作预测映射为各个体的可执行控制指令，实现跨体型、跨域、跨语言的机器人学习。

**💡 创新点**

创新点在于将动作抽象为相机观测中的三维锚点运动，打破低层控制差异；同时将人类视频直接作为新的体型加入训练，避免显式的人机动作映射。

**🔧 技术方法**

采用 Qwen3‑VL‑4B‑Instruct 视觉‑语言 backbone，三阶段训练（相机动作预测、几何翻译、联合微调），以及基于相机‑基座变换与雅可比矩阵的几何条件翻译器。

**📊 数据集**

使用 11 个公开数据集，共计 6,373 小时演示，包括真实机器人（RoboChallenge, RoboCoin, DROID）、仿真机器人（RoboCasa, LIBERO, RoboTwin, InternData）、人类手部 egocentric 视频（VITRA, EgoDex, EgoVerse）以及视觉‑语言监督（ShareRobot, RefSpatial‑v2 等）。

**📈 对比分析**

与多种单体专门化与通用化基线对比，单一统一 checkpoint 在 LIBERO、RoboTwin、RoboCasa 等仿真基准上达到或超过 90% 的成功率；在 LIBERO‑Plus 7 类扰动中实现 82% 的零射击总分；在真实机器人（Bread 抓取、Drawer 开启、Bowl 堆叠）中取得 60–90% 的成功率，显著优于传统方法。

**⚠️ 局限性**

局限性包括对相机标定、深度估计、手部关键点定位的高精度依赖；跨体型迁移仍存在显著差距；实验规模受限，未覆盖多样化机器人、摄像头配置和长时限任务。

---

## 544. How Much Rank Does LoRA Need? Rank-Error Bounds for Transformer Attention

**arXiv ID:** 2608.26052 | [PDF](https://arxiv.org/pdf/2608.26052v1)

**作者:** Gerard Conangla Planes `[一作]` `[通讯]` (Aily Labs), Gerard Conangla Planes (Aily Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文构建了一个任务相关的理论框架，用来预测在Transformer注意力层上使用LoRA（低秩适配器）时，给定目标注意力分布，某一秩限制下能够达到的最小Kullback–Leibler（KL）误差。

**💡 创新点**

创新点在于：①将中心化得分误差与注意力KL误差通过全局softmax不等式关联；②提出下采样加权的谱近似定理，将任务权重（查询与键的实际使用）融入低秩误差上限；③给出上下界，包括目标-费舍尔、局部概率阈值、高质量低秩近似等多条路线；④揭示softmax饱和效应可导致“软max闭包秩”小于对应的有限logit秩；⑤扩展至融合多头和联合查询/键LoRA。

**🔧 技术方法**

技术上主要采用：softmax的凸分析（log-sum-exp的二阶近似）、加权低秩矩阵近似定理、谱分解与残差能量、目标-费舍尔不等式、稀疏概率下界、Walsh函数构造的饱和例子、以及对RoPE等位置编码的考虑。

**📊 数据集**

实验与数据集主要以理论验证为主；在说明性示例中使用文本到SQL的下游任务进行校准集的采样，构造了目标注意力与真实注意力分布；实际评估基于理论上给出的误差曲线。

**📈 对比分析**

比较方法：给出同一秩下的理论上限与下界，形成“误差曲线区间”；与传统经验调优（多秩实验）对比，可直接判断某一秩是否足够或必然不足。论文未给出具体数值性能指标，而是通过误差曲线展示理论预测的可靠性。

**⚠️ 局限性**

局限性包括：需先获得目标LoRA更新；定理是总体量级（population）结果，缺乏有限样本置信区间；常数如概率底限、几何常数、矩上界在实践中难以估计；对优化过程（如SGD）无保证，可能无法达到上界；在极端尖锐注意力分布时下界常数过大；对RoPE等位置编码的限制；仅对注意力概率给出保证，未直接转化为模型最终任务误差。

---

## 545. When Obstacles Bend: Modeling Vegetation Deformation in the context of Field Robotics

**arXiv ID:** 2608.26050 | [PDF](https://arxiv.org/pdf/2608.26050v1)

**作者:** Muhammad Hsaeeb Zaar Khizar `[一作]` (Université Clermont Auvergne), Johann Laconte `[通讯]` (Université Clermont Auvergne)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于杆理论的植被机械属性建模方法，利用机器人在植被中前进时的力传感和形变测量，分别估计植被的空间弹弯刚度分布 EI(s) 和聚合旋转刚度 kθ，进而实现与机器人无关的植被特征化。

**💡 创新点**

创新点在于：①首次将 Kirchhoff 杆模型与实时接触力、形变测量相结合，直接从机器人交互中逆向推算植被的弹弯刚度分布；②提出既可使用视觉+力传感得到高分辨率 EI(s) 又可仅用力传感得到低分辨率 kθ 的双层模型，兼顾实时性与通用性；③验证了模型在不同接触高度下的跨平台可转移性。

**🔧 技术方法**

主要技术包括：平面 Kirchhoff 杆动力学建模、基于 RGB‑D 的中心线提取与曲率恢复、弹弯刚度的直接与参数化估计、基于线性弹性与小变形近似的聚合刚度推导、以及利用电缆伸长测量的无摄像机角度恢复。

**📊 数据集**

实验数据集：在实验台上使用 UR10 机械臂与电缆力传感器，分别在人工草丛（L=0.47 m）和野生树枝（L=0.80 m）上进行多高度推送；配合 Zed2i RGB‑D 相机进行形变捕获。数据集主要由力曲线、姿态、形变中心线及其曲率组成。

**📈 对比分析**

评估方法：将实验测得的力–变形关系与两种模型（EI(s) 与 kθ）预测的曲线进行对比；EI(s) 通过对多高度数据聚合得到分布后验证其在不同高度的稳健性；kθ 仅在小变形区间内线性拟合。结果显示：EI(s) 在不同接触高度下保持一致，能够跨高度、跨机器人使用；kθ 在小变形区间内可预测，但随高度变化显著，且无法捕捉到力饱和后出现的软化现象。

**⚠️ 局限性**

局限性：①分布刚度模型需要 RGB‑D 视觉支持，无法在无视觉环境下使用；②聚合刚度模型仅适用于小变形范围，无法描述较大变形或屈服后的行为；③实验仅在两种人工/野生植被上验证，缺乏对多种自然植被类型的广泛验证；④对电缆力传感器的依赖，若传感器精度或安装位置不稳定，估计误差可能增大。

---

## 546. Integrated Hardware Annealing based on Langevin Dynamics for Ising Machines

**arXiv ID:** 2608.26100 | [PDF](https://arxiv.org/pdf/2608.26100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 547. UltraPIPS: Improving model perception in B-mode ultrasound with foundation models

**arXiv ID:** 2608.26033 | [PDF](https://arxiv.org/pdf/2608.26033v1)

**作者:** Tal Grutman `[一作]` (Tel Aviv University), Tali Ilovitsh `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究并评估了使用超声领域专用基础模型作为LPIPS指标，对B‑mode超声图像的感知相似度进行测量，并在分类、分割与重建三类下游任务中进行系统对比。

**💡 创新点**

首次提出并验证了针对B‑mode超声的专用LPIPS度量，证明超声专用模型在感知相似度与重建质量上显著优于传统自然图像或通用医学模型，并对不同领域背bones对LPIPS的影响进行深入分析。

**🔧 技术方法**

采用LPIPS损失与多种基础模型（CNN、ViT、SwinViT、CLIP等）结合，使用自监督掩码自编码、文本报告对应的Ultrasound‑CLIP以及SIREN隐式神经表示等技术；通过EchoGains仿真增益、UltraSam分割和SIREN重建等实验流程。

**📊 数据集**

使用EchoGains增强的心脏B‑mode图像（EchoPrime、CAMUS、EchoNet）、USOVA 3D Follicles和MicroSegNet Prostate数据集；并利用公开的RadImageNet、MedSAM、USFM、TUSA、BiomedCLIP等预训练模型。

**📈 对比分析**

通过Spearman相关系数将各LPIPS距离与下游任务的置信度、Dice分数以及重建指标（SSIM、HFEN、GLCM）进行对比；实验结果表明超声基础模型的LPIPS与下游性能相关性最高，并在重建任务中兼顾图像质量与纹理真实性，传统L2/SSIM存在局限。

**⚠️ 局限性**

仅在超声领域的心脏和前列腺数据上验证，缺乏更多疾病或解剖区域的数据；评价仍以模型性能相关性为准，未直接测量与人类视觉感知的匹配；部分ViT/USFM模型表现不佳，说明模型选择仍需谨慎。

---

## 548. Slasher: Power Flexibility for Cloud Datacenters

**arXiv ID:** 2608.26021 | [PDF](https://arxiv.org/pdf/2608.26021v1)

**作者:** Liuzixuan Lin `[一作]` (University of Chicago), Ricardo Bianchini `[通讯]` (Microsoft Azure)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Slasher 系统，实现云数据中心在多种电力调节场景下的动态功率控制，结合硬件（电池、发电机）与软件（服务器关机、限频、迁移）杠杆，提供统一的功率灵活性方案。

**💡 创新点**

创新点包括：① 将不同功率调节场景统一到一个层次化控制框架；② 设计基于条件期望值 (CVaR) 的服务影响模型，量化功率削减对服务 SLO 的风险；③ 开发高保真数据中心模拟器用于测试控制策略；④ 通过贪心算法实现低延迟、低影响的服务器关闭决策。

**🔧 技术方法**

技术包括：层次化控制架构（局部控制器+区域编排器）、离线预计算 playbook、硬件控制器、离散事件仿真（Salabim）、基于 CVaR 的影响评估、贪心服务器选择算法。

**📊 数据集**

使用来自 20 个云数据中心的实际功率与利用率追踪，30 天的服务 CPU 利用率历史、1 小时的功率遥测和 1 小时的 VM 分配记录作为实验数据。

**📈 对比分析**

通过对比三种关闭策略（AscUtil、AscScore-OnePass、AscScore-ReEval），在不同功率削减目标下评估总影响成本。结果显示 AscScore-ReEval 在 15% 削减目标下保持几乎零影响，计算时间约 5–6 秒；AscUtil 导致高影响。性能满足实时控制需求。

**⚠️ 局限性**

局限性包括：未覆盖 GPU/LLM 等高能耗工作负载；主要侧重服务器关机杠杆，其他杠杆如限频、迁移的评估不足；算法仅为贪心近似，未探索全局最优；仅在平台层面操作，缺乏应用层交互；模拟器和部分场景仍在完善。

---

## 549. Nearly Optimal Strong Coresets for $\ell_p$ Subspace Approximation

**arXiv ID:** 2608.26047 | [PDF](https://arxiv.org/pdf/2608.26047v1)

**作者:** Honghao Lin `[一作]` (Google Research), David P. Woodruff `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了针对ℓ_p子空间逼近问题的强行列子集核心集构造方法，分别针对 1≤p<2 和 p>2 两种情况改进核心集大小与算法时间。

**💡 创新点**

核心创新在于将低秩分解与 Lewis 权重采样结合，利用经验过程理论实现 ε^{-2} 的精度依赖，并通过改进递归分析将 p>2 情形从 ε^{-p} 提升至 ε^{-2}。

**🔧 技术方法**

使用技术包括低秩分解、Lewis 权重采样、empirical‑process bounds、ridge leverage score 采样、递归行采样与矢量收缩不等式。

**📊 数据集**

论文主要是理论分析，并未在具体数据集上进行实验验证。

**📈 对比分析**

与先前 Woodruff–Yasuda 等工作相比，在 1≤p<2 时核心集大小与精度匹配已知下界（仅差对数因子），在 p>2 时将行数从 ε^{-p} 提升至 ε^{-2}，并保持近似线性时间；整体性能显著提升。

**⚠️ 局限性**

限制包括 p>2 情形的上界与已知下界仍有 ε^{-p} 与 ε^{-2} 的差距；对于 1≤p<2，只在 k 足够大时与下界匹配；且算法仍带有多项对数因子。

---

## 550. Agentic Autoresearch for Cell-Edge Power Control: Radically Redefining the Researcher's Role

**arXiv ID:** 2608.26093 | [PDF](https://arxiv.org/pdf/2608.26093v1)

**作者:** Ahmad Khan `[一作]` (Ericsson R&D), Raviraj S. Adve `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过自主代理（agent）完成从架构设计、输入特征、输出参数化、损失函数到任务采样策略的全自动设计，针对多小区网络中的功率控制问题，实现了对细分百分位率（cell‑edge throughput）的优化；

**💡 创新点**

创新点在于将整个学习算法的设计层交给AI代理，而非仅调参，采用“autoresearch”协议并结合自定义评估器与安全保障，实现了在单一前向推理和固定次数的代数迭代下，逼近最强已知基准；

**🔧 技术方法**

技术包括：基于大型语言模型（LLM）的代码编辑代理；可变与不可变文件协议；自定义评估器（COST231路径损耗、Rayleigh衰落）；固定点迭代的SINR平衡；多头加权双向注意力的可置换等价网络；损失函数为归一化的SLqP + 蒸馏；任务采样覆盖多种K与百分位；

**📊 数据集**

数据集为仿真生成的网络场景：七个包裹的六边形小区，COST231路径损耗模型和Rayleigh衰落，17组（K，百分位）组合（K∈{1,2,4,6,8,10}，百分位∈{min, p10, p25}），在每次实验中使用固定的“pinned”测试集；

**📈 对比分析**

对比方法为传统迭代式的极大最小率功率控制（max‑min）以及基于QFT的二次分数变换迭代求解器。该代理在81次无人工干预实验后获得HeldoutScore 1.4775，约为已知参考1.485的99.5%，推理时间仅为参考的1/600；

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，未包含真实测量数据；仅覆盖0–0.25%的cell‑edge百分位范围；性能受预算限制，若训练/推理时间更短可能需不同设计；代理日志为自报，可靠性依赖于评估器和版本控制；

---

## 551. SwarmWorld: Stigmergic technological evolution in societies of language-model agents

**arXiv ID:** 2608.26081 | [PDF](https://arxiv.org/pdf/2608.26081v1)

**作者:** Subhadeep Pal `[一作]` (Massachusetts Institute of Technology), Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了名为SwarmWorld的模拟平台，让同质LLM代理在共享物理世界中自行组织、构建可执行技术并进行自我评估，进而对比其与孤立搜索的能力。

**💡 创新点**

创新点在于首次实现“提议-后果”分离的完整闭环实验，既保留物理后果的不可逆性，又让语言模型在无角色、无先验技术目录的情况下通过物理耦合、自我学习与可继承程序实现技术文化与知识流。

**🔧 技术方法**

使用了大型语言模型（LLM）作为代理认知与规划单元，结合确定性物理模拟器、可继承可执行控制器、事件日志与程序版号，以及可插拔的世界构建器和文化机制。

**📊 数据集**

使用的数据集为内部生成的三类模拟世界：BioFoundry（基于材料探索的工厂景观）、AshenRealm（火山矿物环境）和Protein Realms（蛋白序列/基质设计空间），每个世界都通过种子化生成可重现的格点地图与资源分布。

**📈 对比分析**

比较方法采用种子配对的实验设计，使用最佳-离散搜索（best-of‑N）基线与无代理冻结的残障评估（held‑out resilience），结果显示共享物理世界在技术多样性、组合韧性和验证发明数上优于独立搜索，但孤立搜索在单个最佳技术性能上仍占优。

**⚠️ 局限性**

局限性包括：仅在离散模拟环境中验证，未涉及真实物理建造与测量；使用单一LLM模型与固定策略，缺乏模型多样性；实验规模受计算资源限制（N≤200）；评价指标聚焦于模拟性能，未检验技术对真实工艺的可转移性。

---

## 552. Gating Before Commitment: Anticipating Intent Divergence to Prevent Post-Interaction Decision Failures in Autonomous Driving

**arXiv ID:** 2608.26074 | [PDF](https://arxiv.org/pdf/2608.26074v1)

**作者:** Cong Xu `[一作]` (University of South Florida), Ravi Sankar `[通讯]` (University of South Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在自动驾驶决策层引入语言模型驱动的意图预测模块，计算意图与几何轨迹的偏差分数，在轨迹提交前进行门控并修正计划，以防止意图误判导致的碰撞。

**💡 创新点**

（1）首次将自然语言模型与结构化感知描述结合，用意图与几何的偏差分数实现决策前门控；（2）提出将模型不确定性视为“放弃”而非半冲突，并加入双阈 hysteresis 以显著降低误报；（3）通过两轮注册实验验证该机制在真实事故片段和四个公共碰撞视频上的有效性。

**🔧 技术方法**

使用 Qwen2.5‑0.5B 语言模型与 LoRA 微调；结构化文本描述（相对位置、速度、车道、阶段标签等）；EWMA 平滑冲突分数；门控逻辑与轨迹修正；基于意图状态的安全围栏；在 monocular 感知堆栈（YOLOv8+ByteTrack、颜色线拟合、单镜头地面平面模型）上运行。

**📊 数据集**

训练数据来源于 nuScenes v1.0 训练/验证集的 55,433 个交互窗口（规则标注）；验证集用于校准；实验评估使用：1) 真实道路交互视频（主案例）；2) 4 个公开碰撞剪辑；3) 20 条无冲突的 nuScenes 验证场景；4) 8 条 comma2k19 高速公路段做域外测试。

**📈 对比分析**

与仅使用安全围栏、仅使用意图模块、以及无门控的基线进行对比；在主案例中，门控在漂移开始后约 70 ms 内触发、在走廊退出前约 160 ms，成功修正所有 10 次回放；误报率从 v1 的 1.54/分钟降至 v2 的 0.341/分钟；在四个碰撞视频中，v2 在 3 个意图级失败中保持全部检测；对比实验显示，几何规则在无模型时检测更多，但在误报率相同的条件下，两者差距缩小。

**⚠️ 局限性**

（1）模型在少数类（yield、merge）上的准确率较低；（2）仅使用 monocular 感知，远程闭合速度噪声大，导致不确定性评估不稳定；（3）门控机制和阈值校准仅在 nuScenes 验证域内验证，域外（comma2k19）误报率显著升高；（4）未考虑 V2V/V2X 协同信息；（5）实验基线为仿真代理规划，未覆盖真实生产规划器的所有细节。

---

## 553. From Fleet to Lab: Revisiting the Security and Complexity of Industrial Rowhammer Mitigation

**arXiv ID:** 2608.26072 | [PDF](https://arxiv.org/pdf/2608.26072v1)

**作者:** Hritvik Taneja `[一作]` (Georgia Institute of Technology), Moinuddin Qureshi `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

暂无可用信息

**💡 创新点**

暂无可用信息

**🔧 技术方法**

暂无可用信息

**📊 数据集**

暂无可用信息

**📈 对比分析**

暂无可用信息

**⚠️ 局限性**

暂无可用信息

---

## 554. Group-Shared Low-Rank Approximation for Mobile-Efficient Pointwise Convolutions in Large-Kernel CNNs

**arXiv ID:** 2608.26069 | [PDF](https://arxiv.org/pdf/2608.26069v1)

**作者:** Hao Luo `[一作]` (Xi'an University of Architecture and Technology), Peng Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于SVD的Channel Group-Shared（CGS）低秩近似方法，用于压缩大型卷积核CNN中占主导的点卷积参数，显著降低模型体积并实现边缘设备部署；

**💡 创新点**

创新点在于将点卷积权重分解为全层共享的降/升投影矩阵与每个通道组的轻量化对角缩放矩阵，兼顾压缩率与表达能力，并针对参数分布不均的点卷积进行专门压缩；

**🔧 技术方法**

采用SVD理论、通道分组、低秩分解、参数共享策略以及整数量化和移动端部署流水线等技术；

**📊 数据集**

在ImageNet-1K、ADE20K、COCO、CIFAR-100等视觉数据集上进行评估；

**📈 对比分析**

与RepLKNet、ConvNeXt、SLaK等SOTA大核网络对比，CGS实现了81%以上的参数压缩（如RepLKNet-31B仅降至15M参数），Top‑1准确率仅下降4.2%；在Android手机上实现25%延迟下降、43%能耗降低，INT8+CGS仅损失0.5%精度；

**⚠️ 局限性**

局限性包括：仅压缩后期网络阶段，未对早期层或其他网络组件扩展；对极限量化（如INT4）兼容性待验证；以及对不同硬件平台的适配仍需进一步研究。

---

## 555. RTLGuard: A Lightweight Teacher-Student Defense for Poisoned RTL Code Generation Models

**arXiv ID:** 2608.26049 | [PDF](https://arxiv.org/pdf/2608.26049v1)

**作者:** Mahshid Rezakhani `[一作]` (University of Central Florida), Hadi Kamali `[通讯]` (University of Central Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级教师-学生框架RTLGuard，用于在不完整训练数据和有限计算资源下恢复被后门污染的RTL代码生成大语言模型，保持生成RTL的功能正确性和可综合性。

**💡 创新点**

创新点在于：①利用小型可信教师模型与少量可信RTL数据引导污染模型恢复；②将交叉熵、知识蒸馏与特征对齐三项损失结合，形成混合恢复目标；③采用参数高效微调（PEFT）仅更新少量适配器参数，显著降低重训练成本；④支持同族、跨规模、跨族教师-学生组合。

**🔧 技术方法**

技术包括：教师-学生对齐、token级知识蒸馏、隐藏状态特征对齐、交叉熵监督、LoRA/DoRA等PEFT技术、bfloat16训练、基于LLM的攻击判别器。

**📊 数据集**

使用OriGen与RTL++两大RTL生成基准数据集；在OriGen上构造10k含8000恶意样本的毒化数据集，教师训练在5k可信RTL样本上。

**📈 对比分析**

与污染基准模型直接对比，RTLGuard将攻击成功率(ASR)从90%以上降至≈10-30%，同时在VerilogEval v2上Pass@1从≈19%提升至≈40-45%；与适配的NAD与Selective Amnesia两种基线相比，RTLGuard在ASR和功能正确率上均表现更优。

**⚠️ 局限性**

局限性包括：对教师模型规模/质量的依赖；在极难清除的功能修改型后门（T1）上仍有残留；需预先构造可信RTL数据集，若无此数据集则恢复效果受限；跨域迁移（不同硬件架构）仍需进一步验证。

---

## 556. Partially-Dynamic All-Pairs Maxflow and Effective Resistance via Stable Sparsifiers

**arXiv ID:** 2608.26037 | [PDF](https://arxiv.org/pdf/2608.26037v1)

**作者:** Gramoz Goranci `[一作]` (University of Vienna), Gernot Zöcklein `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种随机化的数据结构，可维护在仅增删边的无向加权图上所有点对的最大流和有效电阻的 (1±ε) 近似，且总更新时间为 O(n²/ε⁷)，查询时间为 O(1/ε²)。

**💡 创新点**

核心创新在于：①将动态更新序列划分为 O(n/ε) 个 epoch，利用累计杠杆得分（leverage score）证明每个 epoch 内图的谱近似性；②只需在每个 epoch 开始时重新采样一次谱稀疏化器；③将静态的全点对最大流与有效电阻近似算法与动态稀疏化器结合，获得全点对动态近似。

**🔧 技术方法**

主要技术包括：谱稀疏化、杠杆得分的在线估计、ImplicitSampler（动态稀疏化器）、Johnson‑Lindenstrauss 方案求有效电阻、静态最大流/有效电阻全点对近似 oracle，以及对谱近似传播的稳定性分析。

**📊 数据集**

该工作为理论性算法研究，不涉及具体实验数据集，全部在理论模型下证明性能上界。

**📈 对比分析**

与基线 O(n) 的更新时间以及之前的全动态（全点对）近似方法相比，本方法在密集图（m = Θ(n²)）上实现了近乎最优的 O(n²) 总更新时间；对于稀疏图则比基线更快；查询时间保持常数级别（至 1/ε² 的多项式因子）。

**⚠️ 局限性**

局限性包括：①对 ε 的依赖较高，导致总更新时间中 ε⁷ 的项；②仅适用于单向动态（仅增或仅删），不适用于完全动态情形；③查询时间仍为 O(1/ε²)，对有效电阻近似存在一定开销；④算法为随机化，对适应性对手有高概率成功，但不是确定性。

---

## 557. DualOPSD: Adaptive Privileged Teachers for On-Policy Self-Distillation

**arXiv ID:** 2608.26019 | [PDF](https://arxiv.org/pdf/2608.26019v1)

**作者:** Yutong Chen `[一作]` (Clemson University), Kunpeng Liu `[通讯]` (Clemson University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种双向自监督蒸馏框架 DualOPSD，允许受限教师在训练过程中根据学生更新自适应调整，从而提升推理质量。

**💡 创新点**

将受限教师的目标从固定的 oracle 转变为可适应的闭环，使得学生在每一步得到的监督随学生分布变化；采用不对称交替更新和点滴式 KL 上的剪切。

**🔧 技术方法**

采用 on‑policy 自蒸馏，点滴式剪切的 forward KL 作为学生损失；教师更新使用 reverse KL；在冻结的 Qwen3 基础模型上通过 LoRA 适配器实现学生与教师；训练使用 Qwen3 的非思考模式。

**📊 数据集**

在 OpenThoughts 的 29,434 个数学问题与解答集上训练，评测使用 AIME 2024、AIME 2025、HMMT 2025 三套竞赛数学基准。

**📈 对比分析**

与 SFT、GRPO、固定教师 OPSD、PiDistill、BRTS 等方法对比；在 Qwen3-4B 与 8B 规模上，DualOPSD 在 avg@12 上分别提升约 13–23 个百分点，显著降低截断率，但在 1.7B 规模上未见提升。

**⚠️ 局限性**

结果仅在单一模型族、单一随机种子与有限的竞赛题目上验证，且 1.7B 规模表现逆转，说明规模依赖性强；训练成本略增且缺乏多种验证与更广泛的答案审计。

---

## 558. DESCENT: Directed Edge Scene Encoding for Airport Surface Movement Prediction

**arXiv ID:** 2608.26002 | [PDF](https://arxiv.org/pdf/2608.26002v1)

**作者:** Alexander Prutsch `[一作]` (Graz University of Technology), Horst Possegger `[通讯]` (Graz University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于Transformer的机场地面运动轨迹预测模型，结合潜在可达集(PRS)采样和有向边场景编码，提升对多尺度动态和地图约束的捕捉。

**💡 创新点**

创新点包括：①自适应潜在可达集采样机制，可在不同运营阶段提取相关地图上下文；②将机场地图结构化为有向边图，形成稀疏而富含语义的场景表示；③将采样的地图上下文与多模态Transformer解码器结合，支持从低速停靠到高速起降的长时间跨度预测。

**🔧 技术方法**

采用Transformer自注意力与交叉注意力、PointNet-like地图编码、检测Transformer式解码器、多模态高斯混合模型输出、SDF正则化及winner‑takes‑all目标函数等技术。

**📊 数据集**

使用Amelia‑10基准数据集，包含10个美国机场的地面运动记录，进行训练、验证和测试。

**📈 对比分析**

在与基线Amelia‑TF相同的数据拆分和评估指标（mADE/mFDE）下进行对比。长预测时段（50s）在安全关键场景中显著优于基线，短时段（20s）差距缩小；跨机场统一模型亦能保持高精度，尤其在高复杂机场表现突出。

**⚠️ 局限性**

局限性包括：推理时延略高于基线；对某些具有独特运行模式的机场仍需专门模型；在极端高速或大尺度轨迹上误差相对较大；以及尚未完成与ATC系统的深度集成。

---

## 559. FRAME: separating sampling variation from representational cause in medical imaging fairness

**arXiv ID:** 2608.25981 | [PDF](https://arxiv.org/pdf/2608.25981v1)

**作者:** Mahshad Lotfinia `[一作]` (Friedrich-Alexander-University Erlangen-Nuremberg), Soroosh Tayebi Arasteh `[通讯]` (RWTH Aachen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现FRAME框架，用两步方法审计医学影像公平性中的子组性能差异，区分抽样变异与模型表示原因；在大规模多模态数据上评估多种预训练与公平干预手段；

**💡 创新点**

创新点在于构造“公平模型参考”量化仅由抽样导致的差异，并通过特征注入操作检验潜在机制，首次系统性比较多模型、预训练目标、调优级别及已有公平方法对差异的影响，并将该框架应用于已公开的89项公平报告进行审计；

**🔧 技术方法**

采用统计模拟（AUROC/率的采样分布）生成公平参考，设计两类注入算子（解码性注入和疾病信息消除注入），结合多种公平干预（重采样、加权、目标优化、分组阈值/校准、去码、迭代零空间投影）、自监督、图像文本预训练以及不同冻结/微调层级的视觉编码器；

**📊 数据集**

使用总计702,206张影像，涵盖胸片（650,207张，6个站点）、皮肤病（41,999张，3个来源）和眼底（10,000张，1个来源）等多模态数据，并对9篇公开研究中的89个子组差异进行审计；

**📈 对比分析**

通过比较报告差异与公平参考得到剩余差异；大多数干预仅对剩余差异影响极小，图像文本预训练在最坏组AUROC提升约0.05；在已公开研究中，41/89的差异超过公平参考，表明多数差异可归因于抽样；

**⚠️ 局限性**

局限包括：无法精确识别剩余差异的机制（注入操作未覆盖预训练过程）、子组样本量不足导致参考过大、混合站点的“其他”种族类别可能掺杂站点效应、未考虑患者内相关性、仅评估冻结特征层的干预、以及依赖自报属性和有限的模态覆盖范围。

---

## 560. MyoMechanix: Biomechanically-Grounded Compositional Skilled Activity Understanding and Coaching

**arXiv ID:** 2608.26094 | [PDF](https://arxiv.org/pdf/2608.26094v1)

**作者:** Hao Yin `[一作]` (University of Science and Technology of China), Weiwei Fu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个全新的多模态、结构化、可解释的健身动作质量评估框架，构建了大规模的 MyoMechanix 数据集，并设计了 Fitness Knowledge Graph (FKG) 与 CUBIST 结构化推理模型。

**💡 创新点**

创新点包括：① 将肌电 (sEMG)、心率、呼吸等内部生理信号与多视角 RGB 视频、3D 运动捕捉同步采集，填补传统动作质量评估仅依赖视觉的空白；② 通过 FKG 将动作拆解为阶段、关键步骤、错误类型与反馈，实现细粒度的错误归因与可解释评分；③ CUBIST 模型引入分解–分析–重组的三阶段推理，兼容多模态输入并使用规则化评分引擎，既保留可解释性又获得 SOTA 评估性能。

**🔧 技术方法**

技术手段包括多模态深度学习（RGB 3D Transformer、Skeleton GCN、sEMG 频谱处理）、多模态特征分离式 late-fusion、基于混合专家（MoE）的错误分类、隐式相位解析、以及基于 FKG 的规则化评分公式。

**📊 数据集**

使用的主要数据集为自建的 MyoMechanix（7,512 条样本、20 种加重训练动作、38 位受试者、5 视角 RGB + 3D 运动 + 16 通道 sEMG + 心率/呼吸），并对其进行 AQA、VideoQA 与 Video2EMG 三种基准任务评估。

**📈 对比分析**

通过与 7 种主流 AQA 模型、4 种姿态模型、以及多模态融合基线比较，CUBIST 在 Vanilla、Cross‑Subject、Cross‑View 与 Mix‑View 四种拆分下均取得最高的 Spearman ρ 与最低 R‑l2，证明了多模态与结构化推理的优势；VideoQA 任务显著提升了 VLM 的细粒度问答能力；Video2EMG 任务展示了可从视频预测肌电的可行性。

**⚠️ 局限性**

局限性包括：① 数据集仍主要聚焦力量训练动作，其他运动领域的泛化有限；② sEMG 采集受硬件成本与佩戴舒适度限制，易受皮肤接触质量影响；③ FKG 与错误权重设定依赖专家经验，可能导致主观性；④ CUBIST 结构复杂，训练成本高，推理速度相对较慢。

---

## 561. Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings

**arXiv ID:** 2608.26088 | [PDF](https://arxiv.org/pdf/2608.26088v1)

**作者:** Evelyn Ma `[一作]` (Google Research), Shravya Shetty `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了 Planetary Prediction Engine，一种能够从自然语言查询自动生成空间回归、超分辨率缩放、流行病现在预测等多种地理空间预测模型的全自动AI架构。

**💡 创新点**

提出了模块化的自主智能数据选择与多模态集成方案，利用大型语言模型（LLM）进行任务推理与工具调用，并结合地理空间基础模型嵌入，实现无人工工程即可获得专家级性能。

**🔧 技术方法**

使用LLM驱动的工作流编排、PDFM与AlphaEarth的地理空间嵌入、自动特征工程与泄漏防护、基于超参数搜索的模型族（线性、梯度提升、XGBoost、MLP）以及多层过拟合保护等技术。

**📊 数据集**

整合了公开地理空间数据库（Data Commons、Google Earth Engine、Google Maps Platform）、人口与经济统计、卫星影像特征，以及针对埃博拉、尼日利亚粮食安全、美国CDC健康与FEMA风险指标等实际任务的数据。

**📈 对比分析**

通过与专家手工构建基准模型、仅使用统计协变量或仅使用基础模型嵌入的消融实验比较，结果显示埃博拉现在预测Recall@10提升10.3个百分点、尼日利亚粮食安全下采样R²从31.5%提升至66.1%、美国健康指标R²从60%提升至76.8%，整体相对R²提升12–94%。

**⚠️ 局限性**

当前使用冻结的基础模型嵌入，未进行端到端微调；在多尺度下采样时高频卫星特征可能引入噪声；泄漏防护依赖启发式过滤，缺乏形式化因果验证；评估仅针对单一埃博拉爆发，缺乏跨病原体和多地区的验证。

---

## 562. TraceML: An Empirical Analysis of Human-Agent Planning in Machine Learning Development

**arXiv ID:** 2608.26086 | [PDF](https://arxiv.org/pdf/2608.26086v1)

**作者:** Jiarui Yan `[一作]` (Carnegie Mellon University), Yiming Yang `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并发布了 TraceML 数据集和统一版本级别的轨迹分析方案，系统记录并对齐了 134 个 Kaggle 竞赛中 4,465 条人类轨迹与 7 个竞赛中 430 条人类轨迹与 207 条两种 LLM 代理（Codex 与 MLEvolve）轨迹的代码版本、分数、时间戳以及编辑的动作、意图、大小和分数影响，从而使人类与代理的 ML 开发过程可直接比较。

**💡 创新点**

创新点在于：①将人类公开 Notebook 历史与 LLM 代理的提交/搜索日志映射到统一的版本级别表示；②设计了细粒度的行为标签（动作、意图、幅度、分数效应）并用大模型自动化标注；③通过对比分析揭示人类与代理在探索、验证、模型切换、集成与回溯等维度的差距；④基于诊断结果构造规划提示（planning harness）试图逼近人类行为，并验证其对代理性能的提升。

**🔧 技术方法**

主要技术包括：大语言模型（Qwen3‑1.7B）用于自动标注标签；Git 与 Kaggle 历史解析；脚本化的轨迹抽取与归一化流程；PCA、JSD 等统计方法用于行为分布比较；利用训练好的 Agent 代理（Codex CLI 与 MLEvolve）在 12 小时预算下进行对齐实验；以及规划提示的设计与周期性注入。

**📊 数据集**

使用的数据集是：① Meta Kaggle 公开 Notebook 与代码镜像（共 4,465 条人类轨迹）；② 对应 7 个竞赛中 Codex 与 MLEvolve 的 430 条人类轨迹和 207 条代理轨迹；③ 公开的 TraceML 版本级别轨迹格式与标签信息（已发布于 Hugging Face）。

**📈 对比分析**

比较方法：将人类轨迹按排行榜百分位分为高低两组；对齐后对比代理与人类在 19 个轨迹特征（动作、意图、幅度、分数效应等）上的分布差异；使用 Jensen‑Shannon Divergence、PCA 等可视化；在 7 个对齐竞赛上以 12 小时预算跑实验，衡量最佳有效分数与人类百分位的对齐情况。结果显示：Codex 过度细化提交、MLEvolve 过度局部变异，均未达到人类的探索与回溯比例；规划提示可显著降低 Codex 的单一循环行为、提高小幅编辑率，部分竞赛的代理分数提升到与顶尖人类相近水平，但整体差距仍未完全消除。

**⚠️ 局限性**

局限性包括：① 仅对公开 Notebook 进行抽取，隐藏的本地实验未被纳入；② 自动标注虽高效但在意图标签上仍有较低一致性；③ 仅评估了两种代理框架，无法覆盖更广泛的 LLM 开发工具；④ 规划提示的效果因竞赛和预算差异而不均，且仅在可指令化行为上有效，无法修正代理的记忆与控制缺陷；⑤ 由于对齐依赖于公共历史，部分人类行为可能被低估。

---

## 563. ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing

**arXiv ID:** 2608.26083 | [PDF](https://arxiv.org/pdf/2608.26083v1)

**作者:** Roshan Prakash Rane `[一作]` (University of Tübingen), Kerstin Ritter `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出ICON多变量方差分解方法，评估深度网络层对概念的解释

**💡 创新点**

将概念解释从单变量解码转为多变量方差分解，消除共线性误判并提供未解释方差量化

**🔧 技术方法**

使用部分最小二乘（PLS）与Type I逐步平方和分配，对概念矩阵与输出共同建模

**📊 数据集**

在ToyBrains模拟数据、ISIC 2019皮肤癌图像与英国生物银行脑MRI数据上进行评估

**📈 对比分析**

与线性探针、CAV、传统统计方法对比，ICON在模拟、皮肤癌与脑影像实验中显著降低误报、恢复真实重要性，误差显著低于对照方法

**⚠️ 局限性**

仅适用于线性编码，无法区分未解释方差的来源，且只能评估已给定概念，无法发现新概念

---

## 564. Prefix Sliding for efficient test-time scaling

**arXiv ID:** 2608.26070 | [PDF](https://arxiv.org/pdf/2608.26070v1)

**作者:** Niklas Muennighoff `[一作]` (Stanford University), Mike Lewis `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Prefix Sliding Attention 方法，让语言模型在推理时只保留前缀和最近的滑动窗口 token，从而实现长时间推理的固定成本计算。

**💡 创新点**

创新点在于发现中间推理 token 重要性快速衰减，利用前缀+滑动窗口的结构既无需额外训练也能支持超长推理；并提供 RL 训练方案进一步提升性能。

**🔧 技术方法**

使用 FlashAttention 核心的两层过滤（intra‑tile masking 与 inter‑tile skipping）实现高效窗口注意；结合 RoPE/Continue PE 位置编码、vLLM、Nvidia Hopper 自定义内核；RL 训练采用 GRPO、tr1/prime‑rl 等。

**📊 数据集**

主要在数学推理基准上评估：GPQA、MATH500、AIME25，并用自构造的数学题集进行 RL 训练；同时对 LiveCodeBench、HealthBench 等任务做局限性测试。

**📈 对比分析**

与全注意力、Last‑k、Summary、纯滑动窗口等方法对比，未训练时 Prefix Sliding 速度约 3 倍快且准确率与全注意力相当；在 RL 训练下可支持 100k+ token 推理并获得更高奖励；在大多数长推理任务上表现优于替代方案。

**⚠️ 局限性**

局限性包括：短生成任务收益有限；需要保留大量中间信息的任务（如 LiveCodeBench）窗口可能需更大；多轮交互或文件/网页读取时可能丢失重要内容；模型可能需进一步训练以适应不同任务。

---

## 565. StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models

**arXiv ID:** 2608.26067 | [PDF](https://arxiv.org/pdf/2608.26067v1)

**作者:** Zhe Liu `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种流式多模态时序建模框架，能够在不增加额外参数的前提下为单帧Vision‑Language‑Action模型（如π_0.5）添加持久的时间记忆与空间感知能力。

**💡 创新点**

核心创新包括：①将每个（视觉观测+语言指令）对作为原子时间单元，以指令锚定的时序建模；②在单元内部使用双向注意力进行跨模态融合，在单元之间使用因果注意力实现在线时序推理；③随机间隔流式训练以缓解同步训练与异步部署的差异；④利用LLM的长度外推能力实现无参数扩展。

**🔧 技术方法**

技术手段主要为Transformer注意力（双向+因果）、键值缓存（KV Cache）实现低延迟流式推理、随机间隔采样和时间掩码训练策略，以及利用预训练模型的长度外推特性。

**📊 数据集**

实验数据集涵盖：真实机器人任务（Shell Game、Rolling Object Grasping、Pen Insertion、Cup Insertion）、LIBERO仿真基准（四套任务），以及CALVIN多任务长序列基准。

**📈 对比分析**

通过与π_0.5以及多种VLA基线比较，实验表明：在内存依赖任务上提升最高可达+36.6%（滚动抓取）和+33.3%（杯子隐藏），在精细感知任务上提升+26.7%（笔插入）和+32.0%（杯子插入）；在LIBERO上平均提升1.4%，对长序列任务提升+2.6%；在CALVIN上序列成功率从79.5%提升至85.0%，平均序列长度也由4.31提升至4.55。

**⚠️ 局限性**

局限性主要在纯空间静态任务（如LIBERO-Spatial）上几乎无提升，随机间隔训练需要手动调参且对极高频率实时控制的适应性仍待验证。

---

## 566. AsymSpec: Context-Asymmetric Speculative Decoding for Agentic LLMs

**arXiv ID:** 2608.26004 | [PDF](https://arxiv.org/pdf/2608.26004v1)

**作者:** Sheng Liang `[一作]` (Huawei Technologies Co., Ltd.), Yong Liu `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AsymSpec，一种在代理式 LLM 管道中使用异步上下文的显式推理加速框架；通过轻量化绘稿器读取完整上下文，压缩的验证器在较短上下文上工作，利用对比 δ‑融合和 Context‑Divergence Acceptance (CDA) 门进行自适应 steering。

**💡 创新点**

核心创新在于打破传统 Speculative Decoding 的“对称上下文”约束：只让验证器处理压缩视图，绘稿器保持完整信息；利用同一模型在两种上下文下的 logits 差值来精确提取上下文增益信号，并用无参的 CDA 门根据上下文偏差动态调节接受阈值。

**🔧 技术方法**

技术包括：1）对比 δ‑融合（a_i - b_i）将绘稿器在完整与压缩上下文的 logits 差值加入验证器分布；2）CDA 门采用 JSD 计算上下文偏差，动态放宽接受阈值；3）跨模态扩展，绘稿器可为视觉-语言模型，验证器保持文本模型；4）在 vLLM 上实现多路 forward 与并行验证。

**📊 数据集**

数据集：LongBench（hotpotQA/2WikiMQA/MuSiQue）、MultiChallenge、API‑Bank、MathVista、GAIA、SimpleQA；压缩方式包括摘要、LLMLingua‑2、截断等；在文本与跨模态任务上分别评估。

**📈 对比分析**

与基线比较：Floor（仅压缩输入）、Ceiling（完整输入）、标准 Speculative Decoding、SCD、RAPID 等。AsymSpec 在四个文本代理能力上平均恢复 59–94% 的 Accuracy gap，同时以 0.23× FLOPs 和 1.45× throughput 相较 Ceiling；在跨模态 MathVista 上达到 53.9% 正确率，比对称 SD 提升 10.1 分；在 GAIA/SimpleQA 实际代理循环中，保持或超过 Full‑Context 结果，显著降低 compute 并保持稳定的 draft acceptance 率。

**⚠️ 局限性**

局限性：1）恢复受压缩视图信息量与绘稿器容量限制；2）跨模态精度受视觉-语言桥接质量约束；3）需要验证器 logits，无法直接应用于仅返回文本的 API；4）目前仅在确定性解码（τ=0）下验证，随机采样需进一步研究；5）对不同模型家族的 δ‑融合需要词表对齐，跨族迁移受限。

---

## 567. Uncertainty-Guided Latent Diffusion Models for Faithful Super Resolution

**arXiv ID:** 2608.25998 | [PDF](https://arxiv.org/pdf/2608.25998v1)

**作者:** Ren Wang `[一作]` (National Taiwan University), Yung-Yu Chuang `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出UGDiff，通过在潜在空间中估计不确定性并用其引导扩散过程，提升单幅图像超分辨率的感知‑失真平衡。

**💡 创新点**

创新点在于：①设计了专门的潜在不确定性估计器并以简单的L2损失训练；②在扩散采样中根据不确定性与后验方差动态调整引导力度，实现高不确定区域细节恢复而不牺牲全局保真。

**🔧 技术方法**

采用了扩散模型（Stable Diffusion v2.1 + ControlNet）、潜在扩散模型（LDM）、BSRNet恢复网络以及自定义不确定性估计网络。

**📊 数据集**

训练使用LSDIR和FFHQ的低/高分辨率配对；评估则在DIV2K‑Val（合成）和RealSR（真实）数据集上。

**📈 对比分析**

与StableSR、ResShift、SeeSR、FaithDiff、SUPIR、PiSA‑SR和DiffBIR等方法相比，UGDiff在PSNR/SSIM与LPIPS/NIQE指标上实现了更优的感知‑失真折衷，尤其在NIQE上显著优于同类方法。

**⚠️ 局限性**

局限性包括：仅针对4×上采样；需要额外的恢复模型和不确定性网络，导致训练和推理成本上升；在更大上采样倍数或不同任务下的通用性尚待验证。

---

## 568. ProgRouter: Online Progress-Guided Orchestration for Multi-Agent LLM Workflows under Quality-Cost Tradeoffs

**arXiv ID:** 2608.25992 | [PDF](https://arxiv.org/pdf/2608.25992v1)

**作者:** Somgyuan Li `[一作]`, Shiqiang Wang `[通讯]` (University of Exeter)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在线进度引导的多智能体LLM工作流路由框架ProgRouter，实现了在满足成本预算的前提下动态选择LLM代理以提升任务完成质量。

**💡 创新点**

通过多视角进度评估器、双路径进度预测器和虚拟队列成本约束，首次实现基于实时任务进度的自适应LLM路由决策。

**🔧 技术方法**

采用多视角进度评分、双路径树模型预测、Lyapunov漂移+虚拟队列约束、在线探索‑更新学习，以及轻量级句子编码器和树回归器等技术。

**📊 数据集**

在四个主流基准上评估：HumanEval Plus、MBPP、MATH‑500和ASQA。

**📈 对比分析**

与MasRouter、Cascadia和Educated Guessing进行对比，ProgRouter在所有满足长期能耗约束的基准上实现最高通过率/引用精度，能耗最低且执行时间最短。

**⚠️ 局限性**

仅在四个领域验证，缺乏对其他多智能体任务（如网页导航、工具辅助QA）的通用性；进度评估器需手工域适配，未实现端到端自动学习。

---

## 569. Multi-Granularity Context-Enhanced RAG over Multimodal Knowledge Graphs

**arXiv ID:** 2608.25986 | [PDF](https://arxiv.org/pdf/2608.25986v1)

**作者:** Zongyu Wu `[一作]` (Pennsylvania State University), Suhang Wang `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了如何构建多模态知识图谱并提升GraphRAG性能，提出了CEMMKG框架，通过为视觉元素设计多粒度文本上下文来实现改进。

**💡 创新点**

创新点在于系统性定义并多阶段利用本地与全局文本上下文（参考句子、段落、摘要），并将其嵌入图RAG的图构建和融合流程，实现跨模态知识融合的优化。

**🔧 技术方法**

采用LLM（如Llama‑3.1‑70B、Qwen2.5‑72B、InternVL2.5‑38B）生成视觉描述与文本摘要，利用MinerU解析文档，构造MMKG，并将其与GraphRAG模块融合。

**📊 数据集**

评估数据集为MMLongBench‑Doc的VisionHeavy子集（106题），涵盖学术论文、行业文件、宣传册等多种文档类型。

**📈 对比分析**

与直接推理、MMGraphRAG和RAG‑Anything基线对比，CEMMKG在MMGraphRAG下硬准确率提升至34.91%/软准确率36.84%，在RAG‑Anything下硬准确率提升至35.56%/软准确率41.12%，显著提升了多模态问答性能。

**⚠️ 局限性**

局限性包括需手工选择上下文粒度，长文本中冗余或噪声上下文可能削弱效果；对不同文档结构的适应性有限；LLM摘要产生的计算成本较高。

---

## 570. VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning

**arXiv ID:** 2608.26105 | [PDF](https://arxiv.org/pdf/2608.26105v1)

**作者:** Junxiang Xu `[一作]` (Nanyang Technological University), Zhongang Cai `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个可闭环、可验证、可扩展的视觉推理基准 VBVR‑Pro，提供 300 个程序化生成的任务、可验证奖励分数器以及多模态（图像、视频、交互式）生成模型的统一训练与评估框架。

**💡 创新点**

创新点包括：① 通过程序化生成 300 题目实现大规模可扩展训练；② 设计基于任务规则的可验证奖励分数器，替代 VLM 判定；③ 在同一任务空间下系统比较图像、视频和交互式生成三种模态；④ 在此基准上开展多任务强化学习，验证可验证奖励的优化效用。

**🔧 技术方法**

使用了程序化生成器、规则驱动的奖励分数器、Coefficients‑Preserving Sampling (CPS) 用于增强 RL 探索、PPO/GRPO 等强化学习算法以及大规模多任务微调。

**📊 数据集**

数据集主要为自建的 VBVR‑Pro 数据集（300 任务、约 1.25M 训练实例 + 50K RL 子集），并在外部七个视觉推理基准（如 RISE‑Video、MME‑CoF‑Pro、BabyVision 等）上进行迁移评估。

**📈 对比分析**

通过人类标注对可验证分数器与模型输出的配对一致度进行评估，并与多款公开模型对比。VBVR‑Pro 训练后模型在 ID 和 OOD 任务上平均提升约 20% 以上，且在外部基准上的性能提升在 10%–20% 之间。

**⚠️ 局限性**

局限性包括：① 任务覆盖仍有限，无法涵盖所有真实世界复杂场景；② 可验证奖励分数器主要适用于结构化任务，对非结构化任务的适用性受限；③ 强化学习训练计算成本高，尤其在多模态大规模任务上；④ 现有模型在极端噪声或极端视觉变化下的鲁棒性仍不足。

---

