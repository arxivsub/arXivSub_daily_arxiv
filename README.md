# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-28 | 今日论文总数: 472

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. LLM Driven Design of Continuous Optimization Problems with Controllable High-level Properties

**arXiv ID:** 2601.18846 | [PDF](https://arxiv.org/pdf/2601.18846v1)

**作者:** Urban Skvorc `[一作]` (Paderborn University), Heike Trautmann `[通讯]` (Paderborn University)

**通讯引用:** 4513 | [OpenAlex ID](https://openalex.org/A5084825454)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于 LLM 与进化循环相结合的 LLaMEA 框架，用来自动生成满足多种高层景观特性（多模态、可分离、基底大小均匀性、搜索空间均匀性以及全局与局部极值对比）的连续优化问题，并通过基底吸引分析与 ELA 预测模型进行验证，最终构建了一套可公开获取的、多样化且可解释的 benchmark 库。

**💡 创新点**

创新点在于：①将大语言模型嵌入进化过程，直接生成可解释的函数代码；②在 ELA 空间实现自适应 fitness‑sharing，显著提升生成问题的多样性；③结合 ELA 预测模型与基底吸引分析两种验证手段，确保生成问题满足目标特性；④提供完整的 Python 库和案例，填补了 BBOB 等传统 benchmark 的结构多样性缺口。

**🔧 技术方法**

主要技术包括：LLaMEA 进化框架、LLM（如 gpt5‑nano）提示生成、XGBoost ELA 特征预测模型、Manhattan 距离 ELA fitness‑sharing、基底吸引分析、t‑SNE 可视化，以及 Latin hypercube/ Sobol 采样用于 ELA 计算。

**📊 数据集**

数据集主要使用 BBOB 24 组连续问题（各维度 2、5、10）训练 ELA 预测模型，并以此为基准进行生成与验证；生成的题目数量为每个属性/属性组合 55 组，共计 330 题；在验证阶段对 2 维问题使用基底吸引分析，统计局部极值数、全局与局部对比、基底大小比等指标。

**📈 对比分析**

比较方法：①在不同 LLM（gpt5‑nano、gpt3‑x、open‑source 以及闭源模型）下记录成功率（达到 0.5 以上的属性预测分数）和平均分数；②对比使用与不使用 ELA fitness‑sharing 的多样性（邻近 Manhattan 距离分布）；③对生成题目与 BBOB 题目在 t‑SNE 低维特征空间中的分布进行可视化；④统计多模态、对比度、基底大小均匀性等指标的分布并与 BBOB 对比。实验显示 gpt5‑nano 在多模态+全局结构、基底均匀+可分离两组目标上成功率分别达到 98.7% 与 80% 以上，fitness‑sharing 使得邻近距离显著增大，验证指标与 BBOB 对比后均能满足设计目标。

**⚠️ 局限性**

局限性包括：①基底吸引分析仅在 2 维上可行，导致高维特性验证成本高；②LLM 的输出受模型规模与提示设计影响，结果对模型选择敏感；③仅验证了五个高层属性，未覆盖所有潜在景观特征；④与现有生成方法（仿射组合、GP、深度学习等）未做系统比较；⑤生成的题目虽然多样，但仍可能存在与 BBOB 交叉冗余或缺乏极端难度场景。

---

## 2. A Unifying View of Coverage in Linear Off-Policy Evaluation

**arXiv ID:** 2601.19030 | [PDF](https://arxiv.org/pdf/2601.19030v1)

**作者:** Philip Amortila `[一作]` (University of California), Nan Jiang `[通讯]` (University of Illinois)

**通讯引用:** 5944 | [OpenAlex ID](https://openalex.org/A5008181744)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文针对仅满足可实现性（realizability）条件的线性离线策略评估（LSTDQ）进行了新的有限样本误差分析。

**💡 创新点**

创新点在于提出了“特征动力学覆盖度”这一新的覆盖参数，能够统一并推广现有的聚合可浓度（aggregated concentrability）与在Bellman完备性假设下的线性覆盖度，且在理论上更紧凑、方向性更强。

**🔧 技术方法**

采用了工具箱式的技术：线性回归与工具变量（IV）方法的结合、谱分析、矩阵不变性与逆矩阵的代数性质，以及对特征动态系统的构造与分析。

**📊 数据集**

本文未使用任何实测数据集，而是纯粹在理论框架下进行推导与证明；若需实验验证需自行在标准RL基准（如CartPole、Walker2d等）上实现LSTDQ。

**📈 对比分析**

通过与已有理论结果（如Bellman残差最小化、聚合可浓度、FQE等）对比，证明了在满足可逆性等条件时，其误差界在维度依赖、对目标策略的方向性以及收敛速度上均优于或等价于现有最优界；理论上可实现无维度（dimension‑free）误差收敛。

**⚠️ 局限性**

局限性包括：仍需假设矩阵可逆且样本量足够大（burn‑in），覆盖参数的计算可能在实际中难以获得；分析仅适用于线性特征近似，对非线性或深度特征的推广尚未完成。

---

## 3. Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective

**arXiv ID:** 2601.18999 | [PDF](https://arxiv.org/pdf/2601.18999v1)

**作者:** Fangzhou Wu `[一作]` (University of Wisconsin Madison), Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了KV缓存感知负载平衡的统一数学模型，并基于此设计了随机叶子标记淘汰（RLT）与学习型贪心路由（LBGR）两种算法，用以提升LLM推理效率。

**💡 创新点**

①首个统一模型揭示缓存淘汰与负载平衡的权衡；②证明LRU的竞争比率为O(n)，提出O(log n)竞争的随机淘汰；③结合在线回归实现自适应动态路由。

**🔧 技术方法**

随机化标记淘汰算法、基于梯度的在线线性回归估计端到端延迟、指数衰减队列负载、全局前缀共享KV树等技术。

**📊 数据集**

使用Llama‑3.1‑8B/70B和Mixtral‑8×7B模型，基准包括GSP、ShareGPT、UltraChat、Loogle等四个工作负载，并覆盖不同前缀共享设置及随机/最坏排队顺序。

**📈 对比分析**

与随机、轮询、缓存感知+LRU等基线对比，实验显示LBGR+RLT在所有工作负载下，缓存命中率提升约36%、吞吐量提升约36%，延迟和TTFT分别下降约30‑45倍、10‑15倍，最坏情况延迟下降22×。

**⚠️ 局限性**

未评估多模态推理，仅在单域10个工作节点内实验，未覆盖更大规模或分布式部署。

---

## 4. EPAS: Efficient Training with Progressive Activation Sharing

**arXiv ID:** 2601.19089 | [PDF](https://arxiv.org/pdf/2601.19089v1)

**作者:** Rezaul Karim `[一作]` (Huawei Technologies), Walid Ahmed `[通讯]` (Huawei Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 EPAS 的训练方法，在 Transformer 的训练和推理阶段通过逐步开启跨层激活共享（如 QK 共享）实现计算节省。

**💡 创新点**

创新点在于：① 将激活共享与渐进式训练结合，构建可切换解码层；② 训练过程中从深层向浅层逐步扩展共享区；③ 通过单个端到端训练即可得到不同共享比例的可选模型，实现一次训练多模型。

**🔧 技术方法**

技术：Switchable Activation Sharing Decoder（可切换的解码层）；渐进式共享调度（按步长切换层为共享模式）；基于 Flash‑Attention 的 QK 共享实现；实验中使用多 GPU/ NPU 的分布式训练。

**📊 数据集**

数据集：SlimPajama‑627B 子集进行预训练与持续预训练；评估使用 lm‑eval‑harness 对语言建模任务进行基准测试。

**📈 对比分析**

对比方法：与基线（无共享）模型在训练吞吐量、推理吞吐量和验证损失进行对比。结果显示：训练吞吐量提升 10–11%，推理吞吐量提升 22–30%，最终验证损失差异 <0.05，甚至在持续预训练后共享 25% 层时准确率提升约 10%。

**⚠️ 局限性**

局限性：实验仅在 QK 共享上验证，KV 或注意力分数共享的效果未系统评估；对大型模型的通用性尚未在更大规模数据集上验证；实现复杂度较高，需在不同框架上进一步适配。

---

## 5. Smart Split-Federated Learning over Noisy Channels for Embryo Image Segmentation

**arXiv ID:** 2601.18948 | [PDF](https://arxiv.org/pdf/2601.18948v1)

**作者:** Zahra Hafezi Kafshgari `[一作]`, Parvaneh Saeedi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了通信噪声对 SplitFed 学习的影响，并提出了一种基于客户端局部损失的智能平均策略。

**💡 创新点**

创新点在于利用每个客户端的损失均值及其置信区间上限来评估模型可靠性，从而动态调整聚合权重，使模型对噪声更鲁棒。

**🔧 技术方法**

使用 SplitFed U‑Net 结构、白噪声仿真、基于置信区间的权重计算、Adam 优化等技术。

**📊 数据集**

使用了包含 815 张胚胎图像的多标签分割数据集，标注了 BG、ZP、TE、ICM、BL 五个子类。

**📈 对比分析**

与 Naive SplitFed 和 SplitFedAVG 在不同噪声水平下对比，Smart SplitFed 在噪声高达 5×10⁻¹ 时仍保持约 93% 的整体准确率，且训练收敛率显著高于传统方法。

**⚠️ 局限性**

局限在于实验仅在单一胚胎图像分割任务上验证，未考虑更复杂模型或不同类型噪声，且噪声模型仅为白高斯噪声，现实环境中的通信误差可能更复杂。

---

## 6. Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries

**arXiv ID:** 2601.18899 | [PDF](https://arxiv.org/pdf/2601.18899v1)

**作者:** Yuchen Zhang `[一作]` (University of Essex), Haralambos Mouratidis `[通讯]` (University of Essex)

**通讯引用:** 4953 | [OpenAlex ID](https://openalex.org/A5014613493)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了利用语言族层级共享连接器的自动语音识别系统，比较了族级连接器与语言特定连接器的性能差异。

**💡 创新点**

创新点在于提出基于语言族的连接器共享策略，并证明其在多语言与跨域情境下比单语连接器更具参数效率与泛化能力。

**🔧 技术方法**

采用冻结的Whisper大型编码器与Gemma/Salamandra LLM解码器，利用轻量级线性连接器映射语音特征，并在训练时仅调节连接器。

**📊 数据集**

使用FLEURS和CommonVoice两个真实多语言语料库，覆盖近四十种语言并按十个语言族划分。

**📈 对比分析**

通过在族层级与语言层级分别训练连接器，并在相同数据集与LLM上计算WER，实验显示族级连接器在多数族中将WER降低20%以上且跨域迁移更稳健。

**⚠️ 局限性**

研究仅关注最多五种语言每族，未考虑书写系统、形态类型等细粒度因素；同时仅实验两种LLM，可能不适用于更大规模或不同架构的系统。

---

## 7. CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation

**arXiv ID:** 2601.19178 | [PDF](https://arxiv.org/pdf/2601.19178v1)

**作者:** Jingyu Li `[一作]` (Sun Yat-sen University), Pengwen Dai `[通讯]` (Sun Yat-sen University)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5070078619)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种跨用户KV缓存共享机制CollectiveKV，显著压缩推荐模型的KV缓存并降低推理延迟。

**💡 创新点**

创新点在于将KV缓存分解为共享与用户特定两部分，并通过可学习的全局KV池和路由网络实现跨用户共享。

**🔧 技术方法**

主要技术包括奇异值分解（SVD）分析KV可共享性、全局KV池、路由网络、峰值与负载平衡损失以及对Attention的预填/解码阶段拆分。

**📊 数据集**

实验数据集包括MicroVideo、KuaiVideo以及EBNeRD‑Small。

**📈 对比分析**

与传统KV缓存基线相比，CollectiveKV在三种数据集、四种目标注意力模型和一种自注意力模型上实现压缩率低至0.008，且在AUC/GAUC/Logloss指标上保持或提升模型性能，推理延迟显著下降。

**⚠️ 局限性**

主要局限是需要人工设定用户特定KV的维度，若维度过大或过小均会影响压缩效果和性能，缺乏自动调优机制。

---

## 8. Audio Foundation Models Outperform Symbolic Representations for Piano Performance Evaluation

**arXiv ID:** 2601.19029 | [PDF](https://arxiv.org/pdf/2601.19029v1)

**作者:** Jai Dhiman `[一作]` `[通讯]` (Independent Researcher), Jai Dhiman (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用预训练音频基础模型 MuQ/MERT 对合成的钢琴音频进行 19 维感知维度的性能评估预测。

**💡 创新点**

证明即使音频与 MIDI 源信息相同，预训练音频表示也能显著优于符号表示，并且两者融合效果有限。

**🔧 技术方法**

使用 MuQ/MERT 预训练编码器、均值池化、两层 MLP 回归、Pianoteq 声音字体增强以及统计显著性检验。

**📊 数据集**

主要使用 PercePiano（1202 段）、PSyllabus（508 首难度评级）和 ASAP/MAESTRO（多表演一致性验证）数据集。

**📈 对比分析**

采用 4 折 piece‑split 交叉验证；MuQ 9–12 层+声音字体增强得到 R²=0.537，较符号基线 0.347 提升 55%，MERT 0.487，融合仅 0.524。

**⚠️ 局限性**

仅在合成音频上验证，缺少真实录音数据；模型对表演者差异的敏感度低；主实验仅基于 PercePiano，外部验证有限。

---

## 9. Configurable p-Neurons Using Modular p-Bits

**arXiv ID:** 2601.18943 | [PDF](https://arxiv.org/pdf/2601.18943v1)

**作者:** Saleh Bunaiyan `[一作]` (University of California Santa Barbara), Feras Al-Dirini `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 389 | [OpenAlex ID](https://openalex.org/A5071067911)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出可模块化的p-bit架构，通过将随机信号路径与输入路径解耦，设计出可配置的p-neuron，实现多种概率激活函数（p-Tanh、p-Sigmoid、p-RELU）

**💡 创新点**

创新点在于将p-bit拆分为独立的随机单元和激活单元，支持自定义概率激活函数，并实现多p-neuron共享单一随机单元，从而大幅降低硬件资源

**🔧 技术方法**

采用磁隧道结（sMTJ）与CMOS电路实现的模拟Spintronic设计，以及基于FPGA的数字实现，利用线性反馈移位寄存器（LFSR）生成随机数

**📊 数据集**

本研究未使用传统机器学习数据集，而是在硬件仿真和FPGA实验中验证激活函数和共享单元的性能

**📈 对比分析**

通过FPGA实验与传统基于LUT的p-bit比较，显示在资源利用与晶体管计数上可达10倍以上的节省；激活函数的时间均值与理论曲线吻合，证明设计有效

**⚠️ 局限性**

局限性包括sMTJ技术仍处于起步阶段，受TMR值限制；随机分布的均匀性和功耗评估尚待进一步研究

---

## 10. Audio-Driven Talking Face Generation with Blink Embedding and Hash Grid Landmarks Encoding

**arXiv ID:** 2601.18849 | [PDF](https://arxiv.org/pdf/2601.18849v1)

**作者:** Yuhui Zhang `[一作]` (University of Shanghai for Science and Technology), Sunjie Zhang `[通讯]` (University of Shanghai for Science and Technology)

**通讯引用:** 886 | [OpenAlex ID](https://openalex.org/A5006332924)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于 blink embedding 与 hash grid landmarks 编码的音频驱动口型生成框架，利用动态 Landmark Transformer 预测 3D 关键点并融合到 NeRF 中，实现实时高保真说话头像生成。

**💡 创新点**

创新点包括：① 将眨眼 embedding 注入动态 Landmark Transformer，天然捕捉眼部动作；② 使用 instant‑NGP 的三平面哈希编码提升 NeRF 位置编码效率；③ 采用两阶段粗细训练聚焦口部细节，显著提升同步与真实感。

**🔧 技术方法**

采用技术包括：动态 Landmark Transformer、DeepSpeech+VAE 声学特征提取、OpenFace 眼部动作捕捉、三平面哈希编码（Instant‑NGP）、NeRF 渲染、MSE/LPIPS 损失函数、SyncNet 同步评估。

**📊 数据集**

使用公开网络视频数据集（如 Macron、Obama 等），分辨率 512×512、25 fps，训练在单 RTX 4090 GPU 上完成。

**📈 对比分析**

通过与 DFRF、ER‑NeRF、Geneface 等方法在 Macron、Obama 视频上对比，使用 PSNR、LPIPS、LMD、FID、SyncNet 等指标；结果显示本文 PSNR 35.9、LPIPS 0.025、LMD 2.80、FID 10.17、Sync 6.74，整体优于基线，尤其在口型同步和眨眼自然度方面表现更佳。

**⚠️ 局限性**

局限性包括：生成表情仍显僵硬；对每个目标人物需要单独训练；对复杂表情和个性化细节的捕捉仍有提升空间。

---

## 11. Trustworthy Scheduling for Big Data Applications

**arXiv ID:** 2601.18983 | [PDF](https://arxiv.org/pdf/2601.18983v1)

**作者:** Dimitrios Tomaras `[一作]` (Athens University of Economics and Business), Dimitrios Gunopulos `[通讯]` (National and Kapodistrian University of Athens)

**通讯引用:** 19314 | [OpenAlex ID](https://openalex.org/A5063685438)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 X-Sched，一个可解释的容器化大数据任务调度中间件，能够为用户提供可操作的资源配置建议，使任务在资源和时间约束下可执行。

**💡 创新点**

创新点在于将可解释性技术（尤其是反事实解释）与机器学习模型（随机森林）相结合，既能预测任务是否满足服务级别目标，又能生成最小改动的可行配置，并将解释结果以直观的方式呈现给用户。

**🔧 技术方法**

使用了随机森林来学习可调度与不可调度配置的边界，并通过反事实生成算法（结合邻近性、多样性和可行性约束）快速搜索满足时间约束的资源组合；同时实现了 X-Sched 库，提供任务定义、通信原语和与 MLflow 的集成。

**📊 数据集**

实验采用阿里巴巴生产集群的 24 小时长任务和批处理工作负载数据集，对任务的完成时间、内存、CPU 和副本数等特征进行建模和预测。

**📈 对比分析**

与传统基于网格搜索、贝叶斯优化或深度强化学习的调度方法相比，X-Sched 在可行动作空间覆盖率、执行效率和可解释性方面均优于；实验显示大多数生成的反事实配置靠近原始配置、覆盖密度高且满足所有约束，且推理时间在毫秒级，满足实时需求。

**⚠️ 局限性**

限制主要包括：依赖历史任务数据的质量和覆盖范围；对极端或新型任务可能缺乏足够的训练样本；当前实现聚焦于单实例任务的资源配置，尚未深入处理复杂 DAG 任务的依赖与同步；以及对不同容器编排器的兼容性仍需进一步扩展。

---

## 12. Detecting and Correcting Hallucinations in LLM-Generated Code via Deterministic AST Analysis

**arXiv ID:** 2601.19106 | [PDF](https://arxiv.org/pdf/2601.19106v1)

**作者:** Dipin Khati `[一作]` (William and Mary), Denys Poshyvanyk `[通讯]` (William and Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于AST和动态知识库的确定性后处理框架，用于检测并自动纠正LLM生成代码中的知识冲突幻觉（KCH）。

**💡 创新点**

首创将AST分析与动态库反射结合，构建可随版本更新的知识库，并通过确定性规则实现零误报检测和高比例自动修复；相较于现有的预防或非确定性修复方法，提供可解释且无概率偏差的解决方案。

**🔧 技术方法**

技术手段包括：AST解析与语法树抽取、动态知识库构建（库反射获取API签名与别名）、确定性验证规则、编辑距离匹配与局部代码替换、自动插入缺失导入等。

**📊 数据集**

使用人工标注的200条Python代码片段（161条带KCH、39条干净），涵盖numpy、pandas、requests、matplotlib、json等五个主流库。

**📈 对比分析**

与PICARD、Synchromesh、LLM‑in‑the‑loop、Structural Trimming等方法对比，检测精度100%，召回率87.6%，F1≈0.934；自动修复率77%，全部修复时间<0.2秒，显著降低误报并提升可靠性。

**⚠️ 局限性**

局限性包括：数据集规模小、仅覆盖五个库、单文件分析、无法处理多模块数据流与深层语义意图推理，且对更复杂逻辑错误无覆盖；知识库需手动维护以跟随库版本更新。

---

## 13. Privacy-Preserving Model Transcription with Differentially Private Synthetic Distillation

**arXiv ID:** 2601.19090 | [PDF](https://arxiv.org/pdf/2601.19090v1)

**作者:** Bochao Liu `[一作]` (Institute of Information Engineering at Chinese Academy of Sciences), Tongliang Liu `[通讯]` (Trustworthy Machine Learning Lab, School of Computer Science, The University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于差分隐私合成蒸馏的模型转录方法，能够在无私有数据的前提下将预训练教师模型转换为隐私保护学生模型。

**💡 创新点**

创新点包括：1）引入可训练生成器实现无数据蒸馏；2）构建三方协同竞争框架，实现数据或标签的可切换隐私保护；3）通过梯度归一化与top‑k选择等技术实现理论上的差分隐私保证与收敛性。

**🔧 技术方法**

采用的核心技术有：差分隐私机制（Gaussian机制与随机响应机制）、合成数据生成（GAN‑style 生成器）、知识蒸馏（分离式蒸馏与对抗训练）以及梯度归一化和top‑k选择。

**📊 数据集**

在八个公开数据集（MNIST、FMNIST、CIFAR‑10、CIFAR‑100、ImageNet、CelebA‑H、CelebA‑G、MedMNIST、COVIDx）上进行实验，涵盖图像分类与医学影像任务。

**📈 对比分析**

与26种现有方法（包括 DP‑GAN、GS‑WGAN、PATE‑GAN、DataLens 等）在数据敏感与标签敏感隐私保护下进行对比，实验结果显示在同等隐私预算下，本文方法在多数数据集上取得了更高的准确率，并在 ImageNet 等高维任务上展现出显著优势。

**⚠️ 局限性**

主要局限性在于：1）高维数据生成质量仍受限，导致某些任务准确率下降；2）对归一化边界、噪声规模和 top‑k 参数等超参数高度敏感；3）极低隐私预算下的精度急剧下降，需进一步结合标签噪声鲁棒学习等技术改进。

---

## 14. Towards Safety-Compliant Transformer Architectures for Automotive Systems

**arXiv ID:** 2601.18850 | [PDF](https://arxiv.org/pdf/2601.18850v1)

**作者:** Sven Kirchner `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**通讯引用:** 24211 | [OpenAlex ID](https://openalex.org/A5063781430)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于Transformer的多模态深度学习架构，可在汽车安全体系下实现冗余与容错。

**💡 创新点**

创新点在于在表示层面嵌入冗余与多样性，将 ISO 26262 的安全原则映射到多模态Transformer设计。

**🔧 技术方法**

采用多分支 Encoder‑Decoder Transformer 架构，结合自注意力融合与多模态编码器，支持视觉、激光雷达、深度图等多种感知输入。

**📊 数据集**

论文未给出具体数据集，主要以概念与方法论为主。

**📈 对比分析**

未进行实验比较，本文未给出性能指标，只描述了理论上的鲁棒性和可安全认证的潜力。

**⚠️ 局限性**

局限性包括缺乏实证验证、对表示独立性与实时监控机制的实现细节不足、以及在嵌入式平台上的部署效率挑战。

---

## 15. Uncertainty-Aware 3D Emotional Talking Face Synthesis with Emotion Prior Distillation

**arXiv ID:** 2601.19112 | [PDF](https://arxiv.org/pdf/2601.19112v1)

**作者:** Nanhan Shen `[一作]`, Zhilei Liu `[通讯]` (Tianjin University)

**通讯引用:** 2268 | [OpenAlex ID](https://openalex.org/A5055459279)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于不确定性感知的3D情感说话人脸合成框架UA‑3DTalk，实现了从音频到面部表情的精准映射与控制。

**💡 创新点**

创新点包括：①将感知不确定性（置信度）融入多视角特征融合，动态调整视角权重；②提出情感先验蒸馏（Prior Extraction + Emotion Distillation）模块，解决音频情感提取难题；③使用多分辨率代码书和多模态注意力机制精细化情感特征编码。

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting渲染、音频特征拆分（f_exp、f_tone、f_emotion）、多模态注意力加权融合、4D高斯编码、多分辨率代码书、Aleatoric/ Epistemic不确定性估计、Gaussian deformation多头解码器。

**📊 数据集**

实验数据集：常规说话人脸数据集（Obama、May 子集）和情感数据集（MEAD M003、M030 子集）。

**📈 对比分析**

与 TalkingGaussian、DEGSTalk、StableAvatar、EDTalk 等前沿方法对比，UA‑3DTalk 在 E‑FID（-5.2%）、Sync‑C（+3.1%）和 LPIPS（-0.015）等指标上均取得了最优表现，显示了更佳的情感对齐、唇部同步与图像质量。

**⚠️ 局限性**

局限性：①模型复杂度高，训练和推理成本较高；②对极端光照、姿态或极低质量音频的鲁棒性尚需进一步验证；③当前实现主要在已标注的情感数据集上训练，泛化到更多真实场景时可能需要更多多模态数据。

---

## 16. Dynamic Mask-Based Backdoor Attack Against Vision AI Models: A Case Study on Mushroom Detection

**arXiv ID:** 2601.18845 | [PDF](https://arxiv.org/pdf/2601.18845v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 17. How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability

**arXiv ID:** 2601.19208 | [PDF](https://arxiv.org/pdf/2601.19208v1)

**作者:** Shawn Im `[一作]` (University of Wisconsin Madison), Sharon Li `[通讯]` (University of Wisconsin Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过训练动力学分析，揭示Transformer在自然语言数据中如何在早期训练阶段形成语义关联，并给出了这些权重的闭式表达式；

**💡 创新点**

创新点在于将Transformer权重近似为三种基函数（bigram、token互换、上下文映射）的组合，并在理论与实验上证明该表征在小模型与大规模LLM中均成立；

**🔧 技术方法**

使用梯度主导项近似、闭式理论推导、三基函数构造、余弦相似度验证，以及在TinyStories、OpenWebText等数据上的实验；

**📊 数据集**

使用数据集包括TinyStories（小词表）、OpenWebText、FineWeb以及Pythia-1.4B的训练检查点；

**📈 对比分析**

通过与理论主导项的余弦相似度和协方差相似度进行对比，实验显示在早期训练阶段余弦相似度≥0.9，LLM的权重与理论特征高度相关，证明理论有效；

**⚠️ 局限性**

局限性：理论主要适用于单头自注意力、无MLP或小层模型；对多头、MLP和深层模型的证明仅经验验证；仅关注早期训练阶段，长期演化细节未完整描述；假设初始化为零或小随机，实际训练情况可能不同。

---

## 18. Vector-Valued Distributional Reinforcement Learning Policy Evaluation: A Hilbert Space Embedding Approach

**arXiv ID:** 2601.18952 | [PDF](https://arxiv.org/pdf/2601.18952v1)

**作者:** Mehrdad Mohammadi `[一作]` (University of Illinois Urbana-Champaign), Ruoqing Zhu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1430 | [OpenAlex ID](https://openalex.org/A5076762447)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

未提供论文内容，无法进行总结

**💡 创新点**

未提供论文内容，无法说明创新点

**🔧 技术方法**

未提供论文内容，无法说明使用的技术

**📊 数据集**

未提供论文内容，无法说明使用的数据集

**📈 对比分析**

未提供论文内容，无法说明比较方法与性能

**⚠️ 局限性**

未提供论文内容，无法说明限制

---

## 19. Refactoring and Equivalence in Rust: Expanding the REM Toolchain with a Novel Approach to Automated Equivalence Proofs

**arXiv ID:** 2601.19207 | [PDF](https://arxiv.org/pdf/2601.19207v1)

**作者:** Matthew Britton `[一作]` (Australian National University), Alex Potanin `[通讯]` (Australian National University)

**通讯引用:** 998 | [OpenAlex ID](https://openalex.org/A5057086547)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Rust 生态中实现快速、完整的 Extract Function 重构工具 REM2.0，提供自动生命周期修复和可选的 Coq 等价性验证。

**💡 创新点**

重新设计为基于 Rust Analyzer 的独立服务，显著提升性能与语言覆盖率；引入自动生命周期/所有权修复器；构建可验证等价性的 CHARON/AENEAS→Coq 流水线；与 VSCode 集成。

**🔧 技术方法**

Rust Analyzer 提取助手、JSON‑RPC 服务器、迭代编译反馈修复循环、CHARON 与 AENEAS 的 Rust→LLBC→Coq 翻译、Coq 证明自动化。

**📊 数据集**

三组基准：原 REM 40 例子、20 个高星数 GitHub 仓库中提取的 40 例子（覆盖 async、const、泛型、HRTB、动态派发等），以及 20 条验证基准（安全 Rust 子集）。

**📈 对比分析**

对比原 REM：提取成功率 100% vs 97%；提取时延从 ~1–2 秒下降至 10‑30 ms（单机）/ 120‑260 ms（用户感知）；新特性集覆盖率 83%，其中 async 7/8、const 10/11、NLCF 11/13、HRTB 6/8；验证流水线在支持子集上成功率高，平均全程耗时数秒。

**⚠️ 局限性**

仍无法处理复杂的动态 trait 对象、深层泛型边界、unsafe 代码、闭包、嵌套循环和并发；验证仅限于安全 Rust，不能覆盖所有实际代码；部分失败集中在未实现的特性。

---

## 20. Automated structural testing of LLM-based agents: methods, framework, and case studies

**arXiv ID:** 2601.18827 | [PDF](https://arxiv.org/pdf/2601.18827v1)

**作者:** Jens Kohl `[一作]` (BMW Group), Céline Laurent-Winter `[通讯]` (BMW Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套针对大型语言模型（LLM）驱动智能体的结构化测试方法，结合追踪（traces）、Mock 与断言（assertion）技术，构建了单元测试、集成测试和接受测试三层测试金字塔，并在两套业务场景（驾驶员协助代理与云故障根因分析代理）中实现并验证了该方法。

**💡 创新点**

创新点包括：
1) 将 OpenTelemetry 追踪与智能体执行轨迹对接，实现对内部行为的可观察与自动化验证；
2) 对 LLM 进行 Mock，提供可复现、成本低廉的测试环境；
3) 将软件工程中的测试金字塔、TDD、回归测试等最佳实践迁移到 LLM 代理领域，实现从单元到端到端的全链路自动化测试；
4) 提供开源实现（Generative AI Toolkit）与 CI/CD 集成示例，推动社区落地。

**🔧 技术方法**

主要技术手段：
- OpenTelemetry 追踪：捕获 Agent 的每一次行动（如 memory、LLM 调用、工具调用）并存储 span；
- Python/pytest + Expect 类：编写断言，对追踪结果进行验证；
- Mock 机制：对 LLM 进行接口模拟，保证测试可复现；
- AWS Bedrock Converse API：作为 LLM 后端；
- CI/CD 流水线：实现自动化运行、fail-fast 机制与覆盖率统计。

**📊 数据集**

数据集：论文未使用公开数据集，而是基于内部业务数据（车辆手册、客户数据库、云架构文档、日志等）构建的场景示例。若需复现，可使用公开的 RAG 语料或自行构造类似结构化数据。

**📈 对比分析**

比较方法：文章未给出大规模定量实验，而是通过两套案例（驾驶员协助代理、云故障根因分析代理）展示方法效果。主要评价指标为：
- 测试覆盖率提升（单元/集成层）
- 运行成本降低（Mock 取代真实 LLM）
- 根因分析时间缩短
- 测试执行时间缩短（通过 fail-fast 机制）
- 代码重用率提高
- 发现缺陷的提前性提升。
因缺乏对照实验，性能评价主要基于定性观察。

**⚠️ 局限性**

限制：
1) 评估为定性观察，缺乏大规模量化实验与对照；
2) 现有实现仅支持 Amazon Bedrock 接口，迁移到其他 LLM 平台需改造 Mock 与 API 接口；
3) 追踪与 Mock 可能对某些高并发或多代理场景的覆盖不足；
4) 仍需进一步完善跨语言、跨工具的通用测试规范。

---

## 21. How Much Temporal Modeling is Enough? A Systematic Study of Hybrid CNN-RNN Architectures for Multi-Label ECG Classification

**arXiv ID:** 2601.18830 | [PDF](https://arxiv.org/pdf/2601.18830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 22. Bi-Level Online Provisioning and Scheduling with Switching Costs and Cross-Level Constraints

**arXiv ID:** 2601.18936 | [PDF](https://arxiv.org/pdf/2601.18936v1)

**作者:** Jialei Liu `[一作]` (University at Buffalo), Ming Shi `[通讯]` (University at Buffalo)

**通讯引用:** 6269 | [OpenAlex ID](https://openalex.org/A5020652950)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现一种双层在线资源配置与调度框架，结合慢速层的预算规划（带切换成本）与快速层的状态感知调度（受预算约束的CMDP），并给出了统一的学习算法。

**💡 创新点**

创新点包括：① 通过下层双重变量（预算乘子）作为上层梯度的灵敏度反馈，构建了一个可行的双层原理；② 用扩展占据测度线性规划实现动态预算下的安全探索与最优策略；③ 在理论上实现了近似最优的平方根阶 regret 与高概率约束满足。

**🔧 技术方法**

使用的技术有：在线凸优化（OCO）与切换成本；约束马尔可夫决策过程（CMDP）与强化学习；占据测度线性规划与强对偶；置信集合、偏置逼近与优乐性/悲观性平衡；梯度下降与热启动策略。

**📊 数据集**

数据集：基于 Poisson 服从 λ=1.12 的仿真流量和真实 MAWI 交通痕迹（10 ms 归一化到同一均值），用于评估调度与预算决策。

**📈 对比分析**

与两种基线对比：固定预算 DOPE（b∈{4,6,8}）和固定调度 OCO。实验显示双层算法在累计目标差距上最低，同时预算约束几乎零违规，证明了灵敏度反馈与安全探索的有效性。

**⚠️ 局限性**

局限性：对时域长度 T、安全裕度 γ、状态/动作空间大小的依赖较大；仅给出静态基准的 regret，未考虑时间变化的最优轨迹；缺乏多切片、分布式与部分可观测环境的分析与实现。

---

## 23. Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity

**arXiv ID:** 2601.19144 | [PDF](https://arxiv.org/pdf/2601.19144v1)

**作者:** Tzvika Geft `[一作]` (Rutgers University), Kostas Bekris `[通讯]` (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

针对高密度二维网格存储系统，在已知进料顺序但取料顺序可能出现 k‑bounded 扰动的情况下，提出一种鲁棒存储安排与检索策略，能够在不移动或极少移动货物的前提下完成装载与取料。

**💡 创新点**

创新点在于①证明 Θ(k) 列宽是实现零重定位的必要与充分条件；②设计了高效的鲁棒存储求解器与贪心式检索算法；③通过“load‑skipping”增强策略显著提高了成功率。

**🔧 技术方法**

使用了图搜索与贪心启发式方法（路径规划、邻接关系验证）、子问题分解与模块化列配对、以及离散化的 k‑bounded 置换理论。

**📊 数据集**

实验数据来源于随机生成的 100% 满载的 2D 网格实例，覆盖不同网格尺寸与 k/c 比例（0.25、0.5、0.75、1）共 50 次试验。

**📈 对比分析**

与基线（BaseS + BaseR）相比，所提方法在 k ≤ 0.5c 时几乎消除重定位，k ≥ 0.5c 时仍可将重定位量降低 60–70%，同时 I/O 行使用率与整体搬运距离也明显改善。

**⚠️ 局限性**

局限性包括：对给定 k 的鲁棒排布存在性尚未能多项式判定；在极大 k 时仍需多余列宽；实验仅考虑单机器人，未涉及多机器人协同或 MAPF 复杂度。

---

## 24. How Entanglement Reshapes the Geometry of Quantum Differential Privacy

**arXiv ID:** 2601.19126 | [PDF](https://arxiv.org/pdf/2601.19126v1)

**作者:** Xi Wang `[一作]` (University of Sydney), Guodong Shi `[通讯]` (University of Sydney)

**通讯引用:** 4492 | [OpenAlex ID](https://openalex.org/A5101779749)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了量子纠缠对量子局部差分隐私（QLDP）的影响，证明纠缠量的阈值以上可以显著提升隐私保护。

**💡 创新点**

首次揭示了纠缠与QLDP之间的相位转变现象：低纠缠下隐私泄露率与无纠缠相同，高纠缠下泄露率随纠缠熵下降，并能将原本非隐私机制转化为隐私机制。

**🔧 技术方法**

使用了Riemannian优化、KKT条件、熵约束下的非凸几何分析以及可解析的最大/最小隐私能量闭式解。

**📊 数据集**

实验基于4量子比特的人工量子系统（每侧两比特），采用块退相干通道作为本地机制；并未使用真实数据集，而是合成量子态和通道。

**📈 对比分析**

与传统无纠缠的QLDP对比，数值实验显示当纠缠熵超过阈值（log 2）后隐私泄露率从常数下降到更低的极值，且对非隐私机制实现了隐私化；性能改进以泄露率下降量度。

**⚠️ 局限性**

局限性包括：仅考察两体纠缠且局部产品机制，未覆盖多体纠缠结构；理论假设为完美无噪声的量子通道，实际实现可能受限。

---

## 25. Thought-Transfer: Indirect Targeted Poisoning Attacks on Chain-of-Thought Reasoning Models

**arXiv ID:** 2601.19061 | [PDF](https://arxiv.org/pdf/2601.19061v1)

**作者:** Harsh Chaudhari `[一作]` (Northeastern University), Alina Oprea `[通讯]` (Northeastern University)

**通讯引用:** 5782 | [OpenAlex ID](https://openalex.org/A5035574749)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种名为 Thought-Transfer 的间接针对性投毒攻击，利用对链式推理（CoT）轨迹的干预，在不改变查询与答案的前提下，使模型在未见过的目标任务中产生预定的攻击行为，并在标准基准任务上提升性能。

**💡 创新点**

创新点在于：1) 采用 clean‑label 投毒，仅修改 CoT 轨迹而不触发明显异常；2) 通过“思维迁移”将攻击模式从一个任务迁移到另一个无关任务；3) 设计两种 CoT 整合策略（串联与 LLM 合并），显著提高攻击隐蔽性与成功率；4) 通过提升模型整体性能制造诱因，使攻击更难被注意。

**🔧 技术方法**

技术实现主要依赖：
- 对现有公开 CoT 数据集进行采样与改写；
- 使用预训练 LLM（如 Qwen2.5-14B、OpenAI GPT 等）生成对抗性 CoT；
- 两种整合策略：直接串联与调用 LLM 进行上下文融合；
- 训练管道采用监督微调（SFT）与可选的持续微调/偏好对齐（DPO）。

**📊 数据集**

使用的公开 CoT 数据集包括：s1K、OpenThoughts、OpenR1‑Math、Open‑Thoughts、Nvidia OpenMathReasoning 等，覆盖化学、数学、自然语言、代码生成等多领域；目标任务数据（如在线隐私、广告注入、代码漏洞）则由攻击者自行构造。

**📈 对比分析**

对比实验显示：
- 在相关任务下广告注入 ASR 达到约79%，概念操纵约43%；
- 在不相关任务下广告注入 69%，概念操纵 22%；
- 在代码生成任务中攻击成功率可达 98%；
- 同时模型在 GPQA、MATH‑500、AIME24 上提升 10–15%；
- 防御实验（基于困惑度与 CoT 评估器）均表现为检测率低、误报率高，难以有效阻止攻击。

**⚠️ 局限性**

局限性包括：
- 需要对目标任务领域有一定了解；
- 目前针对 SFT 训练场景，RL 或大规模混合任务训练效果未知；
- 防御措施在保持低误报的同时几乎无法检测合并式攻击；
- 过高的投毒率虽提升攻击成功但仍可能被质量检测过滤；
- 对模型容量的依赖性较大，小模型（≤7B）攻击效果不明显。

---

## 26. FROST: Filtering Reasoning Outliers with Attention for Efficient Reasoning

**arXiv ID:** 2601.19001 | [PDF](https://arxiv.org/pdf/2601.19001v1)

**作者:** Haozheng Luo `[一作]` (Northwestern University), Soumalya Sarkar `[通讯]` (RTX Technology Research Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了FROST，一种利用注意力权重剔除推理异常步骤的高效推理方法。

**💡 创新点**

创新点在于定义“推理异常”（low-attention、低熵句子）并使用改进的_softmax_1激活实现句子级别的异常抑制，同时通过轻量级SFT保持模型性能。

**🔧 技术方法**

主要技术包括attention‑aware异常检测、_1激活函数、句子级pooling、LoRA微调以及在生成过程中对attention分布进行动态剪枝。

**📊 数据集**

在四个数学推理基准（GSM8K、MATH500、AIME24、Minerva）以及额外的代码与物理推理任务（LeetCode、LiveCodeBench、UGPhysical）上进行实验。

**📈 对比分析**

与五种主流高效推理方法（TALE、DRP、SelfBudgeter、ThinkLess、SFT）对比，FROST平均提升26.70%准确率，减少69.68% token 使用，显著降低注意力异常指标（∞‑norm↓15.97%，kurtosis↓91.09%），在所有基准上获得领先或近乎最佳结果。

**⚠️ 局限性**

局限性包括目前仅在数学推理任务上验证，未针对编码或多模态任务；仅依赖SFT，未结合强化学习或GRPO等更先进的效率提升技术；对异常检测阈值的选择可能在不同模型和领域需要进一步调优。

---

## 27. On the Role of Depth in Surgical Vision Foundation Models: An Empirical Study of RGB-D Pre-training

**arXiv ID:** 2601.18929 | [PDF](https://arxiv.org/pdf/2601.18929v1)

**作者:** John J. Han `[一作]` (Vanderbilt University), Omid Mohareri `[通讯]` (Intuitive Surgical Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文系统评估了在外科视觉场景中使用RGB与RGB-D多模态预训练的ViT基础模型对下游任务（目标检测、语义分割、姿态估计和深度估计）的影响

**💡 创新点**

引入深度信息作为预训练自监督任务的显式几何约束，并证明了在外科数据上多模态预训练显著提升模型性能与数据效率

**🔧 技术方法**

采用ViT-Base架构，结合MAE、DINOv2、MultiMAE、Mask3D以及自研的DINOv2-RGBD等自监督预训练方法，利用掩码重建与教师‑学生蒸馏技术

**📊 数据集**

使用1.4M帧来自达芬奇机器人手术系统的RGB-D数据进行预训练；下游评估涵盖8个公开外科数据集（CholecTrack20、m2cai16-tool-locations、CholecInstanceSeg、EndoVis18、DV、SCARED、PhaKIR、SurgPose）

**📈 对比分析**

在冻结骨干与端到端微调两种协议下对比模型，结果显示多模态模型（尤其是MultiMAE）在所有任务上均优于RGB‑only模型，数据效率提升显著（仅25%标签即可匹配或超越全量RGB模型）

**⚠️ 局限性**

局限性包括：使用合成深度图而非真实深度、评估范围局限于单一手术类别、未探讨多模态预训练对非视觉任务（如控制、语言）的潜在影响

---

## 28. PsyProbe: Proactive and Interpretable Dialogue through User State Modeling for Exploratory Counseling

**arXiv ID:** 2601.19096 | [PDF](https://arxiv.org/pdf/2601.19096v1)

**作者:** Sohhyung Park `[一作]` (Seoul National University), Dongil Kim `[通讯]` (Seoul National University)

**通讯引用:** 2479 | [OpenAlex ID](https://openalex.org/A5100747073)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 PsyProbe 的主动探索阶段咨询对话系统，通过 PP… 框架与认知错误检测实现结构化用户状态建模，并在对话中实现主动提问。

**💡 创新点**

将 PP… 框架与认知错误检测结合，提出基于 gap‑scoring 的问答生成与 Critic‑Revision 迭代机制，实现对话中的主动提问和状态更新。

**🔧 技术方法**

使用 GPT‑4o/Claude‑3.5‑Haiku 进行多模块推理（State Builder、Memory Construction、Strategy Planner、Response Generator），结合 MI 行为码预测、gap‑scoring、Critic‑Revision 迭代生成回复。

**📊 数据集**

主要使用 KMI 韩语咨询数据集用于 MI 行为码预测；评估采用27名受试者在真实韩语咨询场景中的对话数据，未公开构造的评测集。

**📈 对比分析**

通过自动评测（ROUGE、BLEU、BERT‑F1）与人工评测（用户体验、专家评估）与 GPT‑4o 简单规则基准、Claude‑3.5‑Haiku 及人类咨询师对照；PsyProbe 在提问率、核心问题理解和用户参与度上明显优于基准，接近人类咨询师水平。

**⚠️ 局限性**

系统高度依赖多次 LLM 调用，导致延迟和可扩展性差；缺乏标准评测数据集；受试者规模与会话时长有限；目前仅覆盖探索阶段，未能扩展到洞察与行动阶段。

---

## 29. Reward Engineering for Reinforcement Learning in Software Tasks

**arXiv ID:** 2601.19100 | [PDF](https://arxiv.org/pdf/2601.19100v1)

**作者:** Md Rayhanul Masud `[一作]` (University of California), Md Rizwan Parvez `[通讯]` (Qatar Computing Research Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文系统综述了强化学习在软件工程任务中的奖励设计方法，构建了奖励源、奖励粒度和奖励聚合三维分类框架，回答了四个研究问题，并总结了常见奖励模式、挑战与实践建议。

**💡 创新点**

首次聚焦奖励与环境设计的系统综述，提出了基于奖励来源、粒度和聚合策略的分类法，归纳了50余篇2018-2025年文献的共性模式与最佳实践，填补了现有RL软件工程综述中对奖励设计缺失的空白。

**🔧 技术方法**

采用文献综述与系统评述技术，对各类奖励形式（执行式、相似度式、偏好式）以及不同粒度（token、line、function、program、trajectory）进行归纳与对比，讨论了奖励加权、归一化与多目标视角。

**📊 数据集**

综述中涉及的主要基准数据集包括HumanEval、MBPP、SWE-bench以及各类编译器、单元测试套件和代码质量评测工具，但论文本身并未直接使用这些数据集，而是对已有研究的使用情况进行总结。

**📈 对比分析**

作为综述工作，未进行实验对比；通过文献案例分析比较不同奖励设计在代码生成、修复、检索等任务中的效果，指出稀疏可验证奖励、混合奖励和粒度调节在提升学习效率与最终性能方面的作用，并归纳现有研究报告的性能提升趋势。

**⚠️ 局限性**

局限性包括：范围仅限于奖励与环境设计，可能遗漏最新或相邻工作；未覆盖预训练或指令调优阶段的奖励设计；未给出统一的评测框架和跨论文可比性；未针对非RL任务或间接奖励进行讨论；未来需要持续更新并提供实验验证。

---

## 30. When Does Adaptation Win? Scaling Laws for Meta-Learning in Quantum Control

**arXiv ID:** 2601.18973 | [PDF](https://arxiv.org/pdf/2601.18973v1)

**作者:** Nima Leclerc `[一作]` (MITRE Corporation), Nicholas Brawand `[通讯]` (MITRE Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并验证了量子门校准中元学习自适应步进的规模规律，给出适应性增益随任务方差与梯度步数的指数关系。

**💡 创新点**

创新点在于从优化几何推导出适应增益下界 G_K ≥ A∞(1−e^{−βK})，并通过 PL 条件证明其有效性，提供何时自适应值得投入的定量标准。

**🔧 技术方法**

使用梯度基元学习（MAML/FOMAML）、PL 条件分析、可微分量子模拟（Lindblad 传递方程）以及经典线性二次调节（LQR）验证。

**📊 数据集**

数据集为在模拟器中采样的任务分布，包含不同的去相干/弛豫率（Γ_deph, Γ_relax）以及耦合强度 J 的多样性，全部为仿真生成。

**📈 对比分析**

通过与固定平均基线和每个任务单独优化的 GRAPE 进行对比；在两量子比特 CZ 门的 10 倍噪声情形下获得 41.5% 的保真度提升，单比特门在低方差时增益几乎为 0；经验拟合 R²>0.98 与理论预测吻合。

**⚠️ 局限性**

局限性包括：需要 PL 条件在最优附近成立、必须能获得梯度（不适用于仅能测量结果的硬件）、假设噪声在一次试验内保持平稳，且在大分布偏移时高阶修正可能需要进一步研究。

---

## 31. NavFormer: IGRF Forecasting in Moving Coordinate Frames

**arXiv ID:** 2601.18800 | [PDF](https://arxiv.org/pdf/2601.18800v1)

**作者:** Yoontae Hwang `[一作]` (Pusan National University), Deok-Young Lee `[通讯]` (OAQ Co. Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种 NavFormer 模型，用于在旋转坐标框架下预测地球磁场总强度（IGRF），通过几何前置处理和 Transformer 结合实现。

**💡 创新点**

创新点在于：① 在窗口级别对三轴磁力计的 Gram 矩阵做谱分解，构造可签名不变的正定 SPD 转换实现自适应谱重加权；② 将旋转不变的标量特征与状态条件的 FiLM 调制相结合，消除坐标漂移对序列建模的负面影响。

**🔧 技术方法**

主要技术包括：旋转不变标量特征提取、基于 Gram 矩阵的 Canonical SPD 模块、状态条件的 FiLM 以及 Patch‑Channel Grid Transformer 的自注意力编码。

**📊 数据集**

使用来自 5 只固定翼飞机的磁力计数据集，包含多次飞行的三轴向量和标量磁力计读数，目标为标量磁力计的 IGRF 总强度。

**📈 对比分析**

与 PatchTST、iTransformer、PAttn、TimesNet、DLinear 等前沿时间序列模型对比，在标准、few‑shot 和 zero‑shot 设定下，NavFormer 在大多数飞行数据上取得 MAE 与 RMSE 最高 12%~18% 的提升。

**⚠️ 局限性**

局限性包括：① 对窗口大小与 Gram 矩阵谱分解的稳定性依赖较高；② 在某些飞行段（如 NV4、NV6）仍略逊于部分基线；③ 需要额外的计算开销用于谱重加权与 FiLM 调制。

---

## 32. Encoder-Free ECG-Language Models

**arXiv ID:** 2601.18798 | [PDF](https://arxiv.org/pdf/2601.18798v1)

**作者:** William Han `[一作]` (Carnegie Mellon University), Ding Zhao `[通讯]` (Carnegie Mellon University)

**通讯引用:** 5426 | [OpenAlex ID](https://openalex.org/A5037644321)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并评估了一种无编码器的 ECG 语言模型 ELF，直接将 ECG 信号映射到 LLM 嵌入空间，完成 ECG 解释任务。

**💡 创新点**

创新点在于用单一线性投影取代复杂的 ECG 编码器，既保持了竞争性能，又显著简化了模型架构与训练流程。

**🔧 技术方法**

技术包括线性投影、LLM（如 Llama‑3.2‑1B‑Instruct）、LoRA 微调、BPE 词表、Patch/Conv 轻量级变体，以及零输入扰动实验验证模型对 ECG 的利用情况。

**📊 数据集**

使用了五个 ECG‑文本混合数据集：PULSE ECG‑Instruct、PULSE ECG‑Bench、ECG‑Chat Instruct、PTB‑XL ECG‑QA 与 MIMIC‑IV‑ECG ECG‑QA。

**📈 对比分析**

通过 BLEU‑4、准确率等指标与多种 encoder‑based 与 encoder‑free 基线进行对比，ELF 在大多数数据集上取得与最强基线相当或更优的表现。

**⚠️ 局限性**

局限性包括对语言先验的高度依赖，ECG 信号信息利用不足，以及现有 benchmark 可能未能充分评估模型对 ECG 的真实理解能力。

---

## 33. Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning

**arXiv ID:** 2601.18984 | [PDF](https://arxiv.org/pdf/2601.18984v1)

**作者:** Haolin Liu `[一作]` (University of Virginia), Dong Yu `[通讯]` (Tencent AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的强化学习框架VPPO，通过利用过程奖励模型（PRM）定位推理路径中的首个错误步骤，并对错误前的正确前缀给予额外奖励，从而改进大型语言模型的推理性能。

**💡 创新点**

创新点在于：① 将PRM的细粒度评分转化为“首错定位”这一可靠信号；② 仅对错误前的正确前缀进行奖励，避免了传统方法中全路径奖励导致的梯度冲突与探索受限；③ 设计了“缩短前缀”策略以抑制模型通过拆分步骤来作弊（step inflation）的行为。

**🔧 技术方法**

使用的技术包括：基于GRPO的策略梯度强化学习、步骤级别的PRM评分、奖励塑造（first‑error 识别、α 加权奖励）、优势归一化（含可选 RELU）、步骤拆分与前缀裁剪。

**📊 数据集**

实验数据集：AIME‑25、AIME‑24、AMC‑23、MATH‑500、Minerva、Olympiadbench、Hmmt‑feb‑2024/2025；使用 Qwen3‑4B‑Base、Qwen3‑8B‑Base 与 Qwen3‑4B（非思考版）作为基础模型。

**📈 对比分析**

与基线 GRPO、Mixed、RTS 以及 Pass@K 优化方法比较，VPPO 在 Pass@1 及 Pass@K 指标上均显著提升（平均提升约 3–6%），并在多项基准上位居榜首，显示出更稳定、可解释的学习信号。

**⚠️ 局限性**

局限性：① 依赖“Step k”格式，模型可能通过拆分步骤来“inflate”奖励；② 缩短前缀的截断长度采用经验性设定（prompt token length），可能不具通用性；③ 目前仅在少数 Qwen 系列模型上验证，缺乏更广泛的模型与任务通用性评估。

---

## 34. Recommending Composite Items Using Multi-Level Preference Information: A Joint Interaction Modeling Approach

**arXiv ID:** 2601.19005 | [PDF](https://arxiv.org/pdf/2601.19005v1)

**作者:** Xuan Bi `[一作]` (Carlson School of Management, University of Minnesota), Shawn Curley `[通讯]` (Carlson School of Management, University of Minnesota)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种联合交互建模（JIMA）框架，用单一深度学习模型同时学习用户对原子物品（如上衣、下装）与复合物品（如完整服装搭配）的偏好，并考虑客观的配搭兼容度。

**💡 创新点**

创新点在于：① 将多层次（多粒度）偏好数据（用户-单物品、用户-组合、客观兼容度）统一嵌入学习；② 引入二阶与三阶交互项捕捉用户对单个物品喜好与组合不兼容的细粒度关系；③ 采用多任务学习方式，让不同阶数张量共享同一组潜在因子，从而提升嵌入质量。

**🔧 技术方法**

技术实现主要基于多层感知机（MLP）作为预测器，结合张量分解（CPD）与神经协同过滤（NCF）框架；通过交叉损失（RMSE/MAE）+ L2 正则化，用 Adam 优化器训练；在高阶情境下扩展至任意阶张量的交互模型。

**📊 数据集**

使用自建的真实时装数据集：1）100件女性服装（50件上衣+50件下装）收集用户对单物品与搭配的 1–5 评分；2）另一组 300 名参与者评估所有上衣-下装组合的客观兼容度；3）构建对应的张量/矩阵（用户-搭配、用户-上衣、用户-下装、兼容度矩阵）。

**📈 对比分析**

离线实验与在线实验都与传统 CF、MF、CPD、NTF、NCF、线性回归、FM、随机推荐等基线进行比较；结果显示 JIMA 在所有数据源的 RMSE/MAE 均显著低于基线；在线用户评估中，JIMA 推荐的服装得到最高平均评分，显著优于算法基线和非算法基线（域知识、集体偏好、随机）。

**⚠️ 局限性**

局限性：① 需要多层次偏好数据，收集成本较高；② 模型参数多、训练时间长，尤其在大规模场景下计算开销大；③ 在极低观测率的单物品矩阵中，交互项可能过拟合；④ 目前验证集中在时装搭配，其他类型复合物品（如音乐组合、商品捆绑）的泛化仍需进一步验证。

---

## 35. HumanoidTurk: Expanding VR Haptics with Humanoids for Driving Simulations

**arXiv ID:** 2601.18975 | [PDF](https://arxiv.org/pdf/2601.18975v1)

**作者:** DaeHo Lee `[一作]` (Gwangju Institute of Science and Technology), Jin-Hyuk Hong `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 2740 | [OpenAlex ID](https://openalex.org/A5079549883)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文实现了HumanoidTurk系统，将通用人形机器人改造成 VR 驾驶仿真中的全身触觉反馈装置，能够根据游戏中的加速度信号同步移动座椅。

**💡 创新点**

创新点在于将人形机器人从传统的社交/助理角色转变为可重用的 haptic 媒介，利用其多自由度操作和人类尺度实现中等规模的全身触觉反馈。

**🔧 技术方法**

所用技术包括 Unitree G1 人形机器人、基于 AR 标记的座椅定位、滤波式加速度到机器人关节运动的映射、逆运动学控制、Assetto Corsa 驾驶模拟、Meta Quest 3 VR 与 DualSense 控制器震动同步等。

**📊 数据集**

数据集为 Assetto Corsa 产生的纵向和横向 g‑force 信号（60–65 Hz 采样）以及 VR 运动与震动的实时同步记录，实验共计 23 只实验（6+16 参与者）数据。

**📈 对比分析**

通过对比四种条件（无反馈、控制器震动、机器人+控制器、人工人类+控制器），使用 SSQ、UEQ‑S 等量表和访谈，发现机器人+控制器显著提升沉浸感、真实感与乐趣，但也伴随较高的晕动症和舒适度下降；人类反馈在实用性上更佳。

**⚠️ 局限性**

局限性包括实验仅为一分钟的短时段、样本量小、机器人过热/扭矩损失导致安全约束、未测量实际施加力值、缺乏与商业运动平台基准对比，且未涉及长时间使用导致的疲劳与舒适度问题。

---

## 36. The Opaque Pointer Design Pattern in Python: Towards a Pythonic PIMPL for Modularity, Encapsulation, and Stability

**arXiv ID:** 2601.19065 | [PDF](https://arxiv.org/pdf/2601.19065v1)

**作者:** Antonios Saravanos `[一作]` (New York University), Dongnanzi Zheng `[通讯]` (New York University)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5052978614)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出了将C++的PIMPL idiom改造为Python的“Pythonic PIMPL”模式，阐述了通过模糊指针委托实现公共接口与内部实现分离的策略，并给出实际案例与实践指南。

**💡 创新点**

创新点在于将经典的PIMPL理念重新诠释为Python特有的可见性与动态特性相匹配的模式，统一命名并与现有Python化的代理、后端调度等设计模式关联，同时提供长期维护库的使用准则。

**🔧 技术方法**

使用了设计模式分析、模块层级惰性加载、后端选择与工厂模式、Python的命名约定及类型提示等技术手段，对比了标准库与科学计算库中的实现做法。

**📊 数据集**

本研究为概念性论文，未使用任何实验数据集；主要基于代码示例与已有库的结构分析。

**📈 对比分析**

未进行实验比较，仅从理论层面讨论性能影响，指出委托层带来的轻微运行时开销及调试时的栈追踪复杂度。

**⚠️ 局限性**

局限性包括：缺乏经验性验证、仅依赖社群约定实现“私有”访问、可能导致过度抽象、以及在轻量级组件中不必要的额外层级。

---

## 37. RIFT: Reordered Instruction Following Testbed To Evaluate Instruction Following in Singular Multistep Prompt Structures

**arXiv ID:** 2601.18924 | [PDF](https://arxiv.org/pdf/2601.18924v1)

**作者:** Andrew Jaffe `[一作]` (Emory University), Jinho D. Choi `[通讯]` (Emory University)

**通讯引用:** 2818 | [OpenAlex ID](https://openalex.org/A5101829031)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用 RIFT 测试平台对 LLM 的指令跟随进行实验，构造线性与跳跃两种提示结构以评估模型的结构鲁棒性。

**💡 创新点**

创新点在于将提示结构与内容解耦，提供可比的线性与跳跃拓扑，量化结构敏感性及其对性能的影响。

**🔧 技术方法**

采用系统提示 + 用户提示的显式跳转命令框架，使用 LLM 评估器判定语义等价答案。

**📊 数据集**

数据集来源于 Jeopardy! 问答，经过重新表述、预处理后得到约 8.3 万条标准问答对。

**📈 对比分析**

通过基准、线性、跳跃三种结构，在 10,000 条评估样本上比较，跳跃条件准确率相较基准下降约 72%，表明结构对性能影响显著；推理训练的模型略有提升，但仍低于线性条件。

**⚠️ 局限性**

局限性包括计算资源受限、评估器误判风险、仅针对事实问答、只探索线性与跳跃两种拓扑、基准准确率本身未达 100%。

---

## 38. GraIP: A Benchmarking Framework For Neural Graph Inverse Problems

**arXiv ID:** 2601.18917 | [PDF](https://arxiv.org/pdf/2601.18917v1)

**作者:** Semih Cantürk `[一作]` (University of Montréal), Guy Wolf `[通讯]` (DMS, University of Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GraIP 框架，将结构学习、因果发现、网络重连、动态系统建模等多种图学习任务统一视为逆问题，并提供一套基准数据集与评估指标。

**💡 创新点**

创新点在于：① 将传统图学习任务重新表述为逆问题，揭示它们共享的本质；② 设计可学习的逆映射与可微前向映射；③ 引入可微离散化方法（如 I-MLE）并在同一框架内比较多种离散化策略；④ 发布跨领域的统一基准，促进方法迁移与对比。

**🔧 技术方法**

主要技术包括图信息传播网络（MPNN、图变换器）、变分自编码器（VAE）与离散化梯度估计器（Gumbel-Softmax、STE、I-MLE、SIMPLE）、连续松弛方法（NoTears、GOLEM）以及用于图重连的可微采样器。

**📊 数据集**

使用的公开数据集与人工合成数据：ER/BA 图（因果发现）、Springs、Charged（动态系统）、ZINC、Peptides-func/struct（分子属性预测）、WebKB（Cornell、Texas、Wisconsin 经典节点分类任务），以及基于真实基因调控网络的实验。

**📈 对比分析**

在因果发现中与连续松弛方法（NoTears、GOLEM）比较，连续方法在大规模或高密度图上显著优于离散化（I-MLE）；在 NRI 中，STE、SIMPLE 与 I-MLE 三种离散化策略在 Springs 数据上性能相近，STE 在 Charged 数据上更稳健；在图重连中，I-MLE、SIMPLE、Gumbel 方案均明显优于基线 GINE 与随机重连，且 I-MLE 与 SIMPLE 在所有任务上获得最高准确率。

**⚠️ 局限性**

局限性包括：① 离散化梯度估计器易产生偏差或高方差，导致训练不稳定；② 逆问题往往不唯一，缺少有效的正则化策略；③ 随着图规模和数据稀缺性提升，逆问题的可识别性急剧下降；④ 目前大多数基线仍基于 MPNN+离散化，缺乏更丰富的逆映射设计与采样方法。

---

## 39. Nonvisual Support for Understanding and Reasoning about Data Structures

**arXiv ID:** 2601.19168 | [PDF](https://arxiv.org/pdf/2601.19168v1)

**作者:** Brianna L. Wimer `[一作]` (University of Notre Dame), Matt X. Wang `[通讯]` (University of Washington)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5021797001)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 Arboretum，一套自动化系统，可将基于文本的图表规范（Mermaid、Graphviz DOT）编译成三种同步的无障碍数据结构表示：表格、可导航的屏幕阅读器友好视图以及可打印的触觉图形；并评估其对盲人/低视力 CS 学习者的效果。

**💡 创新点**

创新点包括：①以结构为先的中间表示（IR），让多模态输出共享同一语义基础；②基于 Wizard‑of‑Oz 研究得出的五项设计需求，提出四条通用可访问性原则；③将可触摸、屏幕阅读器导航和表格三种模式结合，支持结构推理与算法演示；④首次在数据结构教育中系统比较三种非视觉表示对学习的影响。

**🔧 技术方法**

技术实现主要使用 TypeScript、React、Next.js 前端；解析 Mermaid/Graphviz DOT 生成 IR；利用 ARIA 树/列表实现可导航模式；利用 SVG 与 Braille 文本生成触觉图形；后端采用 Node.js/Express；评估使用屏幕阅读器（JAWS、NVDA、VoiceOver）与手持触觉打印设备。

**📊 数据集**

数据集为研究参与者自行生成的示例数据结构（数组、二叉树）以及实际教学使用的 Mermaid/Graphviz 规范；无公开大规模数据集，仅包含实验中约 8 名 BVI 学习者使用的 5 个任务实例。

**📈 对比分析**

比较方法：混合方法——定量评估（任务准确率、完成时间、Likert 评分）与定性访谈、思考-外音；结果显示：所有任务准确率≥87%，平均时间约 30–40 秒；受访者对触觉图形评价最高，表格对数组有效，导航视图对树结构有一定帮助。与仅使用 ALT‑text 的传统方式相比，系统提升了对结构推理和算法执行的理解。

**⚠️ 局限性**

局限性：仅支持 Mermaid/Graphviz 两种规范，难以覆盖所有绘图工具；触觉输出受制于物理生产成本与尺寸限制；屏幕阅读器支持主要在桌面环境，缺乏对移动端或 Narrator 等的测试；部分 BVI 学习者对 Braille 或触觉不熟悉，导致偏好差异；系统未覆盖更复杂的数据结构（图、链表、多维数组）或更大规模实例。

---

## 40. ASEHybrid: When Geometry Matters Beyond Homophily in Graph Neural Networks

**arXiv ID:** 2601.18912 | [PDF](https://arxiv.org/pdf/2601.18912v1)

**作者:** Shalima Binta Manir `[一作]` (University of Maryland), Tim Oates `[通讯]` (University of Maryland)

**通讯引用:** 7278 | [OpenAlex ID](https://openalex.org/A5114778025)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在低同质性图中提出了一种几何感知图神经网络（ASEHybrid），通过结合Forman曲率、拉普拉斯位置编码和曲率引导的重连来提升节点分类性能。

**💡 创新点**

创新点在于将标签信息量（label informativeness）作为理论指引，证明局部曲率不提升可表达性但能改善信息流，提供曲率引导重连的收敛与稳定性分析，并将这些理论融入单一架构。

**🔧 技术方法**

使用了基于GATv2的注意力机制、GCN卷积、Forman曲率特征、拉普拉斯位置编码（LapPE）以及曲率权重的重连算法。

**📊 数据集**

实验涵盖了Chameleon、Squirrel、Texas、Minesweeper和Tolokers五个节点分类数据集，覆盖不同同质性与标签信息量。

**📈 对比分析**

相较于普通GCN和各个消融实验，ASEHybrid在标签信息量高的异质图上提升约16–24个百分点，整体在所有数据集均优于基线，尤其在Chameleon和Squirrel上表现显著。

**⚠️ 局限性**

局限在于仅在标签信息量大于0时才有显著提升，对高基线或标签信息量低的图效果有限；局部曲率特征仍无法提升对称性打破能力，且曲率重连增加了预处理开销。

---

## 41. More at Stake: How Payoff and Language Shape LLM Agent Strategies in Cooperation Dilemmas

**arXiv ID:** 2601.19082 | [PDF](https://arxiv.org/pdf/2601.19082v1)

**作者:** Trung-Kiet Huynh `[一作]` (University of Science), The Anh Han `[通讯]` (Teesside University)

**通讯引用:** 3759 | [OpenAlex ID](https://openalex.org/A5012915897)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在多语言、不同奖励规模的重复囚徒困境中评估LLM的战略行为，并通过监督式意图分类器识别其隐藏的策略。

**💡 创新点**

首次将奖励规模（stakes）和语言框架纳入LLM策略评估，并提出使用FAIRGAME+监督式LSTM意图识别来解读LLM的“内部决策规则”，揭示不同模型、语言对合作倾向和策略多样性的系统性影响。

**🔧 技术方法**

使用FAIRGAME框架搭建实验平台；生成10轮囚徒困境游戏；训练LSTM、随机森林、神经网络等分类器识别ALLC、ALLD、TFT、WSLS四种经典策略；对LLM生成的决策序列进行高置信度预测。

**📊 数据集**

共1800场游戏（3模型×5语言×3奖励比例×4人格配对×10次重复），产生36,000次决策；训练10,000条合成轨迹作为监督样本；使用多语言prompt模板确保语义与数值一致。

**📈 对比分析**

对比不同奖励比例、语言与模型的策略分布，并用卡方检验、单因素/双因素ANOVA验证显著性。分类器整体准确率≈0.98（LSTM F1≈0.984）。结果显示：奖励放大时LLM从普遍背叛转向条件合作；语言对策略倾向有显著差异；模型间激励敏感度差异明显。

**⚠️ 局限性**

仅覆盖四种经典策略，未考虑混合或零决策定策略；10轮短时程限制了对复杂策略的识别；使用供应商默认温度可能引入噪声；未提供与人类行为的直接对照；置信阈值和模型选择未做充分校准。

---

## 42. Non-Invasive 3D Wound Measurement with RGB-D Imaging

**arXiv ID:** 2601.19014 | [PDF](https://arxiv.org/pdf/2601.19014v1)

**作者:** Lena Harkämper `[一作]` (Institute of Medical Informatics), Rodrigo Santa Cruz `[通讯]` (Queensland University of Technology)

**通讯引用:** 372 | [OpenAlex ID](https://openalex.org/A5082533938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一套基于RGB‑D摄像头的快速、无创3D创口测量流程，利用RGB‑D里程计与B‑spline曲面重建生成创口三维网格，并自动计算周长、面积和尺寸。

**💡 创新点**

创新点包括：① 结合RGB‑D里程计与B‑spline曲面拟合，实现子毫米重建精度且运行时间短；② 通过2D分割标签迁移到3D网格，提升分割鲁棒性；③ 在创口测量中相较于现有对象中心化方法BundleSDF，既提高精度又显著加快速度。

**🔧 技术方法**

使用技术：Intel RealSense D435 RGB‑D捕获、Open3D RGB‑D里程计、ArUco标定、PCL B‑spline曲面重建、SegFormer‑B5 2D创口分割、KNN标签迁移、Savitzky‑Golay平滑、3D测量计算。

**📊 数据集**

数据集：TraumaSIM提供的三种硅胶创口模型（PIS3、PIS4、SD），使用Zivid 2 M70结构光扫描获得高分辨率基准；同时与人工测量结果进行对比。

**📈 对比分析**

比较方法：① 对比单帧、ArUco跟踪与RGB‑D里程计三种对齐方式，RGB‑D里程计在AD、HD等指标最优；② 对比B‑spline、Poisson、Alpha shapes、BPA、DiGS、BundleSDF六种网格化方法，B‑spline在精度与约11–16 s的速度之间取得最佳平衡；③ 与人工测量比较，3D方法在周长、面积和尺寸的方差更低，满足临床10–15 %每周面积变化判定的精度需求。

**⚠️ 局限性**

局限性：对极低视角覆盖、光照变化及患者运动干扰的鲁棒性尚未充分验证；PIS3双床分割不稳定导致测量方差升高；2D分割误差对3D测量的具体影响未量化；目前仅在硅胶模型上测试，需进一步在真实病人中验证。

---

## 43. Leveraging Sentence-oriented Augmentation and Transformer-Based Architecture for Vietnamese-Bahnaric Translation

**arXiv ID:** 2601.19124 | [PDF](https://arxiv.org/pdf/2601.19124v1)

**作者:** Tan Sang Nguyen `[一作]`, Tho Quan `[通讯]` (Vietnam National University Ho Chi Minh City)

**通讯引用:** 1934 | [OpenAlex ID](https://openalex.org/A5056767671)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了两种针对低资源越南-巴纳里语翻译的句子级数据增强方法（MTL DA 与 Sentence Boundary Augmentation），并将其应用于 Transformer‑based NMT 模型，显著提升了翻译质量。

**💡 创新点**

创新点在于将多任务学习与句子边界噪声增强结合，提供一种无额外预处理、无需附加系统、可与任意 NMT 架构兼容的增强框架，特别针对低资源语种的特征进行定制。

**🔧 技术方法**

使用技术包括：Transformer‑based NMT、基于 Easy Data Augmentation 的多任务学习增强、词级/句级噪声操作（swap 等），以及对编码器的强化训练。

**📊 数据集**

实验数据集主要为越南-巴纳里平行语料（低资源规模），并在五个低资源翻译任务上进行评估，四个任务相较基线取得提升。

**📈 对比分析**

与基线、传统 EDA 及词嵌入替换等方法对比，实验表明在大多数任务中平均提升约 10 BLEU 分，部分任务提升 1–5 BLEU，展示了方法的有效性。

**⚠️ 局限性**

限制主要包括：增强操作大多集中在目标语言侧，受词汇稀缺影响；未充分评估不同方言或其他低资源语言的通用性；多任务比例与参数共享设置仍待进一步优化。

---

## 44. The Promise and Reality of Continuous Integration Caching: An Empirical Study of Travis CI Builds

**arXiv ID:** 2601.19146 | [PDF](https://arxiv.org/pdf/2601.19146v1)

**作者:** Taher A. Ghaleb `[一作]` (Trent University), Ying Zou `[通讯]` (Queen's University)

**通讯引用:** 4883 | [OpenAlex ID](https://openalex.org/A5101468895)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过分析1,279个GitHub项目共513,384条Travis CI构建日志，系统评估了CI缓存的采用率、维护行为、对构建时长的实际影响以及常见问题。

**💡 创新点**

首次结合大规模实验与干预式拉取请求，揭示缓存采用与项目成熟度、维护模式、构建性能之间的关联，并通过过程挖掘阐释维护的复杂性。

**🔧 技术方法**

采用回归不连续设计(RDD)、逻辑回归、Fuzzy Miner过程挖掘以及手工编码issue的质性分析等多种技术手段进行数据挖掘与统计分析。

**📊 数据集**

使用TravisTorrent数据集，即从Travis CI和GitHub公开仓库获取的构建日志与项目元数据，覆盖2016年前的公开项目。

**📈 对比分析**

通过前后构建对比与RDD评估局部平均处理效应，发现仅约33%的项目构建时间显著下降；同时发现97%的构建上传缓存，33%的项目存在缓存未命中或无效缓存。

**⚠️ 局限性**

研究样本基于2016年前的Travis CI公开项目，无法直接推广至GitHub Actions、CircleCI等新平台；对私有仓库和更新的CI特性缺乏覆盖，且未考虑缓存策略的细粒度动态调整。

---

## 45. Flatter Tokens are More Valuable for Speculative Draft Model Training

**arXiv ID:** 2601.18902 | [PDF](https://arxiv.org/pdf/2601.18902v1)

**作者:** Jiaming Fan `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**通讯引用:** 7878 | [OpenAlex ID](https://openalex.org/A5100637624)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于目标模型预测分布“平坦度（flatness）”的样本筛选方法，用以过滤训练数据，仅保留对提高 Speculative Decoding（SD）接受率最有价值的样本，从而加速草稿模型（draft model）的训练。

**💡 创新点**

创新点：
1) 从 SD 的接受率视角出发，理论证明目标分布越平坦（越接近均匀）对应的 token 在单步 KD 中能产生更大 L1‑距离下降，因而更能提升接受率。
2) 引入可计算的平坦度指标（目标分布与均匀分布的余弦相似度）作为 token/样本的重要性度量。
3) 基于此指标提出 Sample‑level‑flatness‑based Dataset Distillation (SFDD) 方法，对训练集进行数据蒸馏，显著提升训练效率。

**🔧 技术方法**

技术手段：
- 计算目标模型输出的平坦度（cosine similarity with uniform distribution）。
- 对每个样本求 token 平坦度平均得到 sample‑flatness。
- 依据 quantile 筛选高 flatness 样本。
- 在 EAGLE‑2 框架下对 LLaMA3‑8B‑Instruct 进行草稿模型的 KD 训练。
- 与多种基准（Entropy、Top‑1 Probability、Margin、Energy Score、PPL、Random、No Filter）对比。

**📊 数据集**

使用数据集：
- 训练：ShareGPT 数据集。
- 评估：GSM8K、Alpaca、MT‑Bench、CNN/DM、Natural Questions 等五个下游任务。

**📈 对比分析**

对比方法：随机筛选、Entropy、Top‑1、Margin、Energy、PPL 等。实验结果表明：
- 在 50% 数据保留比例下，SFDD 在所有任务上均获得最高的 inference speedup（平均 2.41×，接近全量 2.49×）。
- 训练时间比全量快约 2×，在 50% 数据下仅损失不到 4% 的 inference speedup。
- 在极低保留比例（5%–20%）下仍优于 Random，体现鲁棒性。

**⚠️ 局限性**

局限性：
- 需要先跑一次目标模型得到 flatness，计算成本相对较高（约 2242 s）。
- 仅在 EAGLE‑2 / LLaMA3‑8B‑Instruct 上验证，泛化到更大模型或其他 SD 方案仍需探索。
- 只考虑了平坦度指标，未探讨动态或多指标融合的可能性。

---

## 46. Optimizing Conversational Quality in Spoken Dialogue Systems with Reinforcement Learning from AI Feedback

**arXiv ID:** 2601.19063 | [PDF](https://arxiv.org/pdf/2601.19063v1)

**作者:** Siddhant Arora `[一作]` (Carnegie Mellon University), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**通讯引用:** 25423 | [OpenAlex ID](https://openalex.org/A5001291873)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了首个针对语音对话系统的多奖励对齐框架，兼顾语义连贯、音质自然、可懂度与情感一致性；

**💡 创新点**

创新点在于将四维奖励并行应用于全双工块级解码，并通过utterance-level偏好指导blockwise生成；

**🔧 技术方法**

使用RLAIF结合Direct Preference Optimization（DPO），并利用LLM评判与自动音频指标构造偏好数据；

**📊 数据集**

偏好数据基于Switchboard语料，评估数据为Eval2000，并在公开的语音对话集上训练；

**📈 对比分析**

在多轮Chain-of-Thought和SCoT全双工模型上与基线对比，单奖励显著提升对应指标，联合奖励在语义、音质与可懂度上均有进一步提升，整体性能优于基线；

**⚠️ 局限性**

局限在于偏好数据自动化生成的噪声与偏见、奖励间缺乏显式平衡机制、仅在英语单域验证，未覆盖多语言或多领域场景。

---

## 47. Glance and Focus Reinforcement for Pan-cancer Screening

**arXiv ID:** 2601.19103 | [PDF](https://arxiv.org/pdf/2601.19103v1)

**作者:** Linshan Wu `[一作]` (Hong Kong University of Science and Technology), Hao Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 106332 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种Glance和Focus强化学习框架GF‑Screen，用于大规模CT扫描中的泛癌症筛查。

**💡 创新点**

创新点在于用分割结果作为奖励训练Glance模型，结合组相对学习消除背景冗余，并首次将RL应用于泛癌筛查。

**🔧 技术方法**

使用3D ResNet‑18作为Glance模型、SwinUNETR作为Focus模型，利用强化学习（GRL）和奖励函数进行训练。

**📊 数据集**

在5,117份来自23个公开数据集的CT扫描（9种病变类型）上进行训练和验证，并在16个内部+7个外部数据集上评测。

**📈 对比分析**

相较于现有最优方法，GF‑Screen在FLARE25验证榜单上DSC+25.6%、NSD+28.2%，平均精度提升至60.8% DSC，推理速度提升5.7倍。

**⚠️ 局限性**

局限性包括对分割质量高度依赖、需要较大数据集支持、且在极小病变上的检出率仍有提升空间。

---

## 48. XR Design Framework for Early Childhood Education

**arXiv ID:** 2601.18979 | [PDF](https://arxiv.org/pdf/2601.18979v1)

**作者:** Supriya Khadka `[一作]` (Coventry University), Sanchari Das `[通讯]` (George Mason University)

**通讯引用:** 1157 | [OpenAlex ID](https://openalex.org/A5059400253)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述111篇2010-2025年关于3-8岁儿童的XR研究，构建了Augmented Human Development（AHD）框架，将XR管线属性与儿童的认知负荷、感官刺激、环境语境及发展特征关联起来。

**💡 创新点**

创新点在于把XR系统的四个关键属性（C、S、E、D）纳入动态函数，并用风险与关注矩阵揭示研究盲区，提供一种以儿童发展为中心的XR设计诊断工具，填补了以往缺乏系统性评估的空白。

**🔧 技术方法**

使用了系统性知识构建（SoK）方法、量化评估指标、风险计算（Likelihood × Impact）和可视化分析技术（散点图、矩阵图）来实现框架的构建与验证。

**📊 数据集**

数据集来源于从2,198条记录中筛选出的111篇包含3-8岁儿童的同行评审研究，形成了元数据表并按七个维度（Pedagogy、Privacy、Data Security、Health、Technical、Disability Access、Low-Resource Access）进行打分。

**📈 对比分析**

本研究并未进行实验比较，而是通过对文献的量化计分和风险矩阵对比，发现数据安全、残疾访问和低资源访问等维度虽然风险高，但学术关注度低，说明存在明显的研究盲点；对学习效果的直接性能评价未包含在本文中。

**⚠️ 局限性**

局限性包括：①仅依赖公开文献，缺乏对XR系统在真实课堂中的实验验证；②样本以AR为主，VR/MR研究相对不足；③未对长期学习效果进行量化评估；④风险评估受行业标准（如NIST、WHO）限制，可能不完全适用于所有教育场景。

---

## 49. Towards Self-Optimizing Electron Microscope: Robust Tuning of Aberration Coefficients via Physics-Aware Multi-Objective Bayesian Optimization

**arXiv ID:** 2601.18972 | [PDF](https://arxiv.org/pdf/2601.18972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 50. Accelerated training of Gaussian processes using banded square exponential covariances

**arXiv ID:** 2601.19007 | [PDF](https://arxiv.org/pdf/2601.19007v1)

**作者:** Emily C. Ehrhardt `[一作]` (Imperial College), Felipe Tobar `[通讯]` (Imperial College)

**通讯引用:** 733 | [OpenAlex ID](https://openalex.org/A5020822083)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于把方差指数核（SE）协方差矩阵裁剪成带状矩阵的高效 GP 训练方法（BTC）。

**💡 创新点**

创新点在于通过理论分析给出带宽 k 的明确选取公式，确保裁剪后的协方差矩阵保持正定并保留预测分布的有效性，从而在不引入诱导点的情况下实现显著的计算加速。

**🔧 技术方法**

采用矩阵裁剪（cut‑off）操作、带状矩阵 LU/Cholesky 分解、Gershgorin 圆盘定理和理论正定性证明，配合方差指数核的指数衰减特性。

**📊 数据集**

使用月度太阳黑子数据集（3315 观测）和赫尔辛基新生儿 EEG 数据集（4000 观测）进行实验。

**📈 对比分析**

与全 GP、FITC、VFE 等稀疏方法对比，BTC 在保持与全 GP 相近的 NMSE 和 NLPD 的同时，显著降低了运行时间，尤其在大样本情况下表现出优越性。

**⚠️ 局限性**

局限性包括仅适用于一维 SE 核的 GP，带宽 k 在训练期间保持固定，且对更高维或不同核函数的推广仍需进一步研究。

---

## 51. SimTO: A simulation-based topology optimization framework for bespoke soft robotic grippers

**arXiv ID:** 2601.19098 | [PDF](https://arxiv.org/pdf/2601.19098v1)

**作者:** Kurt Enkera `[一作]` (CSIRO), David Howard `[通讯]` (CSIRO)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

这项工作提出了SimTO框架，用于为任意具有复杂特征的物体自动生成定制的柔性机械手指。

**💡 创新点**

创新点在于通过在动态抓取仿真中自动提取接触力并迭代更新，摆脱了传统拓扑优化需要手工指定加载情况的限制。

**🔧 技术方法**

采用Taccel进行非线性柔性动力学仿真、Kabsch算法提取和降维接触力、基于SIMP的密度拓扑优化以及自定义目标函数来引导设计。

**📊 数据集**

通过在5个具备高拓扑可变性的物体（弯曲球、齿轮、沙漏、尖刺球、星形）上执行64次参数化优化，生成约2000个专用抓手设计，并公开数据集。

**📈 对比分析**

在仿真中对所得到的柔性抓手进行“在域”和“跨域”抓取实验，软材料设计的成功率高达约80–90%，说明其能有效泛化到未见过的物体。

**⚠️ 局限性**

限制方面包括仅采用二维拓扑优化、仅优化单一抓手指、固定抓手和物体姿态以及需要人工挑选最终设计，未来需扩展为三维、双指优化并引入自动设计筛选标准。

---

## 52. A Framework for Evaluating Faithfulness in Explainable AI for Machine Anomalous Sound Detection Using Frequency-Band Perturbation

**arXiv ID:** 2601.19017 | [PDF](https://arxiv.org/pdf/2601.19017v1)

**作者:** Alexander Buck `[一作]` (Loughborough University), Patrick Baker `[通讯]` (Defence Science and Technology Laboratory)

**通讯引用:** 530 | [OpenAlex ID](https://openalex.org/A5089560511)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种定量评估机器声学异常检测模型解释可信度的框架，将解释相关性与频段去除导致的预测变化关联起来，验证解释方法对模型决策的真实性；

**💡 创新点**

首次给出可复现的音频解释可信度量化方法，基于频段扰动的行为验证为评价XAI方法提供客观标准；

**🔧 技术方法**

使用集成梯度、遮挡、Grad‑CAM、SmoothGrad四种常用XAI方法，结合频段去除实验与Spearman相关性分析，以及自监督FeatEx异常检测模型；

**📊 数据集**

采用DCASE 2023 Task 2机件声音异常检测数据集；

**📈 对比分析**

通过将各XAI方法的频段平均相关性与频段去除导致的预测变化相关，发现Occlusion的相关系数最高（≈0.88），集成梯度中等（≈0.53），Grad‑CAM表现不稳定，SmoothGrad最低（≈0.40），表明Occlusion与模型真实敏感度最匹配；

**⚠️ 局限性**

仅考虑频段扰动而未涉及时间或时频交互；仅使用单一CNN架构；未使用感知尺度的频段划分；仅在模型层面评估，缺乏人机交互与实际诊断验证。

---

## 53. Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2 Neuro-Fuzzy System

**arXiv ID:** 2601.18897 | [PDF](https://arxiv.org/pdf/2601.18897v1)

**作者:** Qusai Khaled `[一作]` (Jheronimus Academy of Data Science), Laura Genga `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 428 | [OpenAlex ID](https://openalex.org/A5056706542)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于区间型二阶自适应神经模糊推理系统（IT2-ANFIS）的废水处理厂能耗预测模型，并对不确定性进行可解释量化。

**💡 创新点**

创新点在于将不确定性分解为特征、规则和实例三层，利用区间型模糊集的FOU实现可解释的预测区间，而非传统黑盒概率方法。

**🔧 技术方法**

采用IT2-ANFIS（第一阶TSK型）、随机梯度下降、正则化（L1、L2、梯度裁剪）以及固定的类型简化因子q。

**📊 数据集**

使用墨尔本水务公司东部处理厂的六年日常能耗、进水特征和气象数据，共约1000条记录。

**📈 对比分析**

与传统ANFIS、随机森林（RF）和支持向量机（SVM）比较，IT2-ANFIS在7条规则下的MSE约为1671 MWh，方差更小，虽然精度略逊于RF，但提供可解释的区间预测。

**⚠️ 局限性**

局限在于仅处理了表观不确定性，未覆盖随机噪声；且固定类型简化因子q可能限制模型自适应度，未来需扩展为结合模糊-概率方法以量化随机不确定性。

---

## 54. Learning the Pareto Space of Multi-Objective Autonomous Driving: A Modular, Data-Driven Approach

**arXiv ID:** 2601.18913 | [PDF](https://arxiv.org/pdf/2601.18913v1)

**作者:** Mohammad Elayan `[一作]` (University of Nebraska Lincoln), Wissam Kontar `[通讯]` (University of Nebraska Lincoln)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于自然轨迹数据的多目标学习框架，用实证方法构建安全、效率、交互三目标的 Pareto 前沿。

**💡 创新点**

创新点在于直接从真实 AV 行为中学习连续 Pareto 面，揭示三目标的最优平衡区域并量化各目标的头部余量。

**🔧 技术方法**

采用 Gaussian Process Regression、KNN 缺失值填充、欧几里得距离与相对角度计算以及多目标归一化构建三维目标向量。

**📊 数据集**

使用美国交通部第三代仿真（TGSIM）数据集，包含 Foggy Bottom 城市交叉口和 I‑395 高速公路的 Level 2/3 AV 行为轨迹。

**📈 对比分析**

通过对比 Pareto‑optimal 与非最优样本的平均分数，并绘制 GPR 平滑 Pareto 表面，表明仅 0.23% 的时刻达到三目标平衡，显示出高风险余量。

**⚠️ 局限性**

局限在于样本稀缺且仅覆盖左转与单车道行驶情境，未覆盖更广泛的交通情境与 AV 级别。

---

## 55. Transparency-First Medical Language Models: Datasheets, Model Cards, and End-to-End Data Provenance for Clinical NLP

**arXiv ID:** 2601.19191 | [PDF](https://arxiv.org/pdf/2601.19191v1)

**作者:** Olaf Yunus Laitinen Imanov `[一作]` (TeMLM Foundation), Berfin Tavan `[通讯]` (TeMLM Foundation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出TeMLM框架，提供透明度优先的临床语言模型发布工具（TeMLM-Card、TeMLM-Datasheet、TeMLM-Provenance）及其可度量的审计指标；

**💡 创新点**

创新点在于将数据、模型、评估全过程统一为可机器检查的文档和版本化记录，并给出可量化的完整性与泄漏门槛，解决临床NLP透明度缺失的问题；

**🔧 技术方法**

采用PROV标准化的事件图、JSON/JSON‑LD序列化、自动化度量脚本（如相似度泄漏、PSI漂移、PHI残留风险）以及BERT基础的ProtactiniumBERT模型；

**📊 数据集**

使用公开的合成临床数据集Technetium‑I（包含PHI注解和ICD‑9‑CM标签），并提供相应的元数据与可重现的工作流；

**📈 对比分析**

在PHI去标识和ICD‑9编码两项基准上，用ProtactiniumBERT‑100M取得micro‑F1分别为0.984和0.760，明显优于规则、BioBERT和ClinicalBERT基线；

**⚠️ 局限性**

局限包括：仅在合成数据上验证，未覆盖真实医院语料；指标集最小化，未覆盖生成任务的校准与临床错误分类；实施成本高，需系统预先设计；并未解决去标识与可用性之间的权衡。

---

## 56. Native LLM and MLLM Inference at Scale on Apple Silicon

**arXiv ID:** 2601.19139 | [PDF](https://arxiv.org/pdf/2601.19139v1)

**作者:** Wayner Barrios `[一作]` `[通讯]` (Wiqonn Technologies), Wayner Barrios (Wiqonn Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了基于Apple Silicon原生MLX框架的多模态LLM推理系统vllm-mlx，支持文本与视觉语言模型的高吞吐量推理，并提供OpenAI兼容API。

**💡 创新点**

创新点包括：① 内容哈希的前缀缓存技术，能够识别不同格式相同图像，避免重复视觉编码；② 对文本请求实现连续批处理与前缀KV缓存，提升多请求吞吐率；③ 充分利用Apple统一内存实现零拷贝操作，提升性能。

**🔧 技术方法**

核心技术：Apple MLX框架、统一内存共享、内容哈希（SHA‑256）、KV缓存管理、连续批处理调度、量化（4‑bit）推理、OpenAI API兼容层。

**📊 数据集**

使用公开的LLM/MLLM模型（Qwen3、Llama 3.2、Gemma 3、Nemotron等）以及视觉输入（单图、视频帧），并在Apple M4 Max上进行基准测试；无专门构造的数据集，主要评估推理延迟与吞吐量。

**📈 对比分析**

与llama.cpp、mlx‑lm、vLLM‑metal等同类框架比较，文本模型吞吐量提升21%–87%，在16并发请求下可达4.3×聚合吞吐；多模态前缀缓存使重复图像查询的延迟从21.7 s降至0.78 s，提升28×；视频缓存同理提升24.7×。

**⚠️ 局限性**

局限性：仅支持Apple Silicon平台；对模型支持依赖于mlx‑lm/ mlx‑vlm；尚未实现音频输入缓存、分布式多设备推理、能耗分析；部分大模型仍受显存/带宽限制。

---

## 57. From Answer Givers to Design Mentors: Guiding LLMs with the Cognitive Apprenticeship Model

**arXiv ID:** 2601.19053 | [PDF](https://arxiv.org/pdf/2601.19053v1)

**作者:** Yongsu Ahn `[一作]` (Boston), Nam Wook Kim `[通讯]` (Boston)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了名为DesignMentor的AI设计导师，利用认知学徒制模型指导大型语言模型（LLM）在数据可视化反馈中更具交互性与反思性。

**💡 创新点**

创新点在于将认知学徒制六大教学方法（建模、辅导、支架、表述、反思、探索）转化为结构化提示，改造LLM从答案提供者转变为过程驱动的设计导师。

**🔧 技术方法**

技术包括基于OpenAI GPTs的定制化提示工程，结合对人类导师行为的编码与评估，以及对LLM对话日志的手工与自动标注分析。

**📊 数据集**

数据集为24名可视化从业者的24个交互对话，包含其自制可视化作品与问题，并与ChatGPT-4o进行对比。

**📈 对比分析**

与ChatGPT-4o进行配对内实验，结果显示DesignMentor在反馈完整性、元认知促进、用户满意度上均优于基线，尽管其交互时间更长、认知负荷略高。

**⚠️ 局限性**

局限在于仅提供文本反馈，缺少视觉示例；实验时长短，未考察长期使用与多轮迭代；系统对不同设计阶段与用户经验的自适应能力仍待提升。

---

## 58. Whispering Water: Materializing Human-AI Dialogue as Interactive Ripples

**arXiv ID:** 2601.18934 | [PDF](https://arxiv.org/pdf/2601.18934v1)

**作者:** Ruipeng Wang `[一作]` (Massachusetts Institute of Technology), Behnaz Farahi `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过多智能体AI系统将用户语音情感与语义转换为低频振动，进而在水面上生成可视化的共振图案，形成交互式“倾诉水”装置；

**💡 创新点**

创新点包括：①将情感识别结果映射为振动频率；②将机器语音分解为多波段组件并在水中重构，形成物理共振；③让AI代理的身份与语音特征随对话动态生成；

**🔧 技术方法**

采用情感分析模型emotion2vec+、ASR、LLM（Claude、GPT、Gemini等）与TTS（ElevenLabs）生成语音；利用STFT、Bark频率尺度分解，随后通过TouchDesigner、Arduino、Focusrite等控制六个低频功放驱动水中振动；

**📊 数据集**

未使用公开数据集，实验数据来自现场收集的15秒以内人声录音；

**📈 对比分析**

论文未给出定量评估或与现有系统的性能比较，仅通过视觉和听觉展示效果；

**⚠️ 局限性**

局限性包括：低频振动只能在水面产生有限的共振图案，情感识别和TTS情感表达范围受限，缺乏大规模用户研究和客观评价。

---

## 59. Differential Voting: Loss Functions For Axiomatically Diverse Aggregation of Heterogeneous Preferences

**arXiv ID:** 2601.18824 | [PDF](https://arxiv.org/pdf/2601.18824v1)

**作者:** Zhiyu An `[一作]` (University of California), Wan Du `[通讯]` (University of California)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Differential Voting 框架，构造可微分的实例损失，使其在总体最优点对应经典投票规则；并设计了 Soft Copeland 和 Soft Kemeny 两种光滑近似，配合理论与实验验证。

**💡 创新点**

创新点在于将投票规则的社会选择性质显式化为损失函数几何，并提供可微分代理来实现 Copeland 与 Kemeny 等传统规则，从而突破传统 RLHF 仅使用 BTL 的局限，展示损失设计与规范假设的直接映射。

**🔧 技术方法**

使用基于梯度下降的可微分优化技术，构造了对称、饱和、边界集中的损失（如带正则化的 tanh 边缘得分、sigmoid 不一致指示器等），并通过理论证明其在温度趋零时收敛于相应的投票规则。

**📊 数据集**

实验数据为人工生成的合成偏好集合，包括转移序列、接近平局、循环（Condorcet 轮回）以及隐藏上下文混合的多上下文样本，覆盖多种偏好结构与上下文异质性。

**📈 对比分析**

与传统的 BTL 损失相比，Soft Copeland 在多数规则下成功恢复 Copeland 赢家，Soft Kemeny 在 Kendall 距离上显著低于 BTL；在隐藏上下文场景下两者均能保持 Condorcet 一致性，而 BTL 则常常违背该准则。

**⚠️ 局限性**

局限性包括仅关注总体最优而非有限样本收敛或 RLHF 训练动态，实验仅在合成数据上验证，缺乏对大型真实标注管道、多胜选或概率聚合等更复杂场景的扩展。

---

## 60. Neuromorphic BrailleNet: Accurate and Generalizable Braille Reading Beyond Single Characters through Event-Based Optical Tactile Sensing

**arXiv ID:** 2601.19079 | [PDF](https://arxiv.org/pdf/2601.19079v1)

**作者:** Naqash Afzal `[一作]` (University of Bristol), Benjamin Ward-Cherrier `[通讯]` (University of Bristol)

**通讯引用:** 942 | [OpenAlex ID](https://openalex.org/A5008451819)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套基于开源神经形态触觉传感器Evetac的连续盲文识别系统，实现实时、低延迟的盲文字母和单词读取；

**💡 创新点**

创新点在于：①采用事件驱动触觉感知替代帧级视觉，直接捕获滑动过程中的稀疏触觉事件；②结合时空分割网络与轻量级ResNet分类器，实现对连续事件流的字符级解码；③引入NormAug归一化+数据增强，显著提升不同压深、滑速下的鲁棒性；

**🔧 技术方法**

技术包括神经形态事件摄像机、事件流热图编码、时空峰值分割、ResNet-34深度残差网络、数据增强、基于语义后处理的拼写校正；

**📊 数据集**

使用自制的3D打印盲文板（包括全字母表、随机排列字母、词汇表），在不同压深（0.2–1.5 mm）和滑速（8–32 mm/s）下收集事件数据；

**📈 对比分析**

与传统帧级视觉盲文读数方法相比，系统在标准压深下字符识别精度≥99.5%，在词级别上≥90%（+0.4%拼写校正后），并能在32 mm/s高速扫描下保持≥97%的单词准确率；

**⚠️ 局限性**

局限性：仅支持Grade 1盲文，未验证对Grade 2/收缩盲文；缺乏真实环境下长期稳定性与能耗测评；分割网络在高速度下仍易误判；

---

## 61. Unravelling the (In)compatibility of Statistical-Parity and Equalized-Odds

**arXiv ID:** 2601.19035 | [PDF](https://arxiv.org/pdf/2601.19035v1)

**作者:** Mortaza S. Bargh `[一作]` (Ministry of Justice and Security), Floris ter Braak `[通讯]` (Ministry of Justice and Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文通过理论分析和图形化方法，研究了在存在敏感群体基率不平衡时，统计平等（Statistical‑Parity）与平衡偶然性（Equalized‑Odds）两种公平度量的兼容性与不可兼容性，并给出了两者共存的必要条件。

**💡 创新点**

创新点在于：
1) 用基率平衡与随机分类器两条路径完整阐释了两种公平度量的兼容性；
2) 提供了基于 FPR–TPR 平面直线交点的可视化分析，直观展示公平度量之间的取舍；
3) 将分析结果与现行法律与实践框架结合，提出对基率平衡评估的法律和实践建议。

**🔧 技术方法**

采用的技术主要是概率统计推导、线性代数、ROC 曲线/平面分析，以及符号化的假设证明；没有引入机器学习模型或优化算法。

**📊 数据集**

使用的是一个基于住房抵押贷款申请的虚构数据集（N_t=8000，分两组 S=0、S=1），并给出三组（A、B、C）不同的操作点作为数值示例；未使用公开真实数据集。

**📈 对比分析**

由于研究是理论性质，比较方法主要是对公平度量间的等价与不等式进行推导，并通过数值示例验证公式；没有传统意义上的实验性能指标（如准确率、召回率）。

**⚠️ 局限性**

局限性包括：
1) 仅考虑二元分类和单一二值敏感属性，无法直接推广到多分类或连续特征；
2) 依赖于假设的基率与标签真实性，实际应用中标签噪声与偏差未被充分处理；
3) 研究缺乏大规模真实数据验证，仅以示例数据说明结论。

---

## 62. RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation

**arXiv ID:** 2601.19120 | [PDF](https://arxiv.org/pdf/2601.19120v1)

**作者:** Guilin Zhang `[一作]` (Workday), Xu Chu `[通讯]` (Workday)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型生成推荐解释在用户行为噪声下的鲁棒性，提出了RobustExplain评估框架。

**💡 创新点**

首次系统化地将五种现实噪声类型和四维鲁棒性度量引入推荐解释评估，填补了解释生成鲁棒性研究空白。

**🔧 技术方法**

采用Qwen2.5、LLaMA等LLM与结构化提示，结合语义相似度、关键词重叠、BLEU及长度一致性等指标进行鲁棒性量化。

**📊 数据集**

使用自建的200个商品、7类品种、100名用户的合成电商交互数据集进行实验。

**📈 对比分析**

对四个模型在5种扰动、5级强度下生成解释并对比，平均鲁棒性约0.50，70B模型比7‑8B高约8%，不同扰动影响差异不大。

**⚠️ 局限性**

仅基于合成数据，评估侧重文本一致性，未涉及用户主观体验，且大模型鲁棒性仍有限。

---

## 63. OWLEYE: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection

**arXiv ID:** 2601.19102 | [PDF](https://arxiv.org/pdf/2601.19102v1)

**作者:** Lecheng Zheng `[一作]` (Virginia Tech), Jingrui He `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4125 | [OpenAlex ID](https://openalex.org/A5073158087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种面向零样本跨域图异常检测的通用框架，能够在未见图上直接检测异常节点。

**💡 创新点**

创新点：①跨域特征对齐模块在保持域特定语义的前提下统一不同图的特征分布；②多域多模式字典学习，将多图中提取的属性和结构模式存储为可持续更新的知识库；③截断注意力重构模块在无标签条件下通过过滤潜在异常节点实现零样本推理。

**🔧 技术方法**

技术手段包括：PCA投影、基于统计距离的特征归一化、图注意力网络（截断版）、多层GNN特征学习、三元组损失与重构损失的联合训练、字典式模式存储与检索。

**📊 数据集**

使用的公开数据集：训练集 PubMed、CiteSeer、Questions、YelpChi；测试集 Cora、Flickr、ACM、BlogCatalog、Facebook、Weibo、Reddit、Amazon。

**📈 对比分析**

与基线比较：在零样本与10样本两种设置下，AUPRC/AUROC均优于ARC、UNPrompt等现有一对全模型，并在大多数数据集上超过监督模型BWGNN、GHRN；平均AUPRC提升约5%～10%。

**⚠️ 局限性**

局限性：①字典规模与性能呈先增后饱和；②在辅助图过多时模型难以收敛，fine‑tune效果不佳；③仍依赖已知的正常模式，极端异常或与已学习模式相似的异常可能被误判；④对高维稀疏特征的对齐仍存在挑战。

---

## 64. Attention-Enhanced Graph Filtering for False Data Injection Attack Detection and Localization

**arXiv ID:** 2601.18981 | [PDF](https://arxiv.org/pdf/2601.18981v1)

**作者:** Ruslan Abdulin `[一作]` (California State University), Mohammad Rasoul Narimani `[通讯]` (California State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合ARMA卷积滤波和Encoder-Only Transformer的联合FDIA检测与定位框架ACEOT

**💡 创新点**

融合位置编码、ARMA卷积与自注意力，实现对局部拓扑与全局长程依赖的同时建模，提升对大规模电网的攻击识别能力

**🔧 技术方法**

自学习位置编码、ARMAConv图卷积、Encoder-Only Transformer、Transformer自注意力机制、基于图的特征提取与多标签分类

**📊 数据集**

使用NYISO历史负荷数据生成的IEEE-14与IEEE-300电网仿真样本，包含四种FDIA场景（优化型、分布型、缩放、重放）

**📈 对比分析**

与MLP、CNN、LSTM、ChebConv、ARMAConv等基线对比，ACEOT在IEEE-300系统上取得最高F1（92.60%）且定位准确率提升约11%；在IEEE-14系统上性能相近且误报率最低

**⚠️ 局限性**

对极端或罕见攻击情况仍有一定失误，且模型对少数被攻击节点的识别稳定性略逊于纯ARMAConv，需进一步提升对稀疏攻击的鲁棒性

---

## 65. SelfieAvatar: Real-time Head Avatar reenactment from a Selfie Video

**arXiv ID:** 2601.18851 | [PDF](https://arxiv.org/pdf/2601.18851v1)

**作者:** Wei Liang `[一作]` (University of Shanghai for Science and Technology), Philippe G. Schyns `[通讯]` (University of Glasgow)

**通讯引用:** 19426 | [OpenAlex ID](https://openalex.org/A5064321977)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种利用单张自拍视频实现实时高细节头部头像复现的方法

**💡 创新点**

创新点在于将3DMM表情建模与StyleGAN生成器结合，并引入混合损失（包括ID‑MRF和感知余弦相似度）以恢复高频细节如皱纹与头发纹理

**🔧 技术方法**

采用3DMM人脸追踪、StyleGAN生成网络、ID‑MRF、ResNet50感知损失等技术

**📊 数据集**

实验使用MEAD和IMAvatar公开数据集以及自制3分钟自拍视频

**📈 对比分析**

与StyleAvatar和IMAvatar对比，SSIM/PSNR提升，LPIPS/FID降低，表现更接近真实图像

**⚠️ 局限性**

局限性包括对自拍视频长度和质量敏感，且对不同光照与背景的泛化能力仍待验证

---

## 66. M$^{\text{2}}$XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization

**arXiv ID:** 2601.19213 | [PDF](https://arxiv.org/pdf/2601.19213v1)

**作者:** Weiming Hu `[一作]` (Shanghai Jiao Tong University), Jingwen Leng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2559 | [OpenAlex ID](https://openalex.org/A5003939279)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于元数据增强的微缩（MX）量化格式（M2XFP），在保持4‑bit有效位的同时通过额外的元数据显著提升LLM推理精度。

**💡 创新点**

创新点：1）在激活量化中使用元素级额外尾数（Elem‑EM）仅需2位元数据捕获子块内最大值；2）在权重量化中使用子块级额外尾数（Sg‑EM）配合自适应尺度搜索，实现精度提升与位宽最小化的双向折中；3）设计轻量级的硬件编码/解码单元，几乎不增加面积与功耗。

**🔧 技术方法**

技术手段：基于块浮点（BFP）微缩量化，元数据分配框架，子块级尺度搜索，在线激活量化流程，Systolic‑array 兼容的 PE 逻辑扩展，2位元数据编码/解码。

**📊 数据集**

使用的模型与数据集：LLaMA‑2‑7B、LLaMA‑3‑8B/70B、OPT‑6.7B、Mistral‑7B、Falcon‑7B 等7B‑70B LLM；评测数据集包括 Wikitext‑v2、Arc‑challenge、Arc‑easy、HellaSwag、PIQA、WinoGrande、BoolQ 以及 DeepSeek‑R1‑Distill‑Qwen 的 AIME、MATH‑500、GSM8K、GPQA‑Diamond、LiveCodeBench。

**📈 对比分析**

对比方法：与 MXFP4、NVFP4、SMX4、MicroScopiQ、MX‑ANT、BlockDialect 等传统 MX 及算法加速器进行对比。结果表明 M2XFP 在相同 32×32 PE 规模下，平均准确率损失仅 1.58%（相较 MXFP4 减少 70.6%），与 NVFP4 相比损失下降 37.3%。硬件评估显示，面积仅比 MXFP4 低 4%，功耗与面积增量 0.36%，速度提升 1.91×，能耗降低 1.75×。

**⚠️ 局限性**

局限性：1）当前设计主要针对 4‑bit 组量化，扩展到更低/更高精度仍需验证；2）权重量化需离线搜索，推理时仅能使用预先量化的权重；3）对 KV 缓存、注意力层等后续模块的适配尚未完整评估；4）在极大模型（>70B）或稀疏/低秩张量场景下的性能与精度需进一步研究。

---

## 67. MAGNET: Towards Adaptive GUI Agents with Memory-Driven Knowledge Evolution

**arXiv ID:** 2601.19199 | [PDF](https://arxiv.org/pdf/2601.19199v1)

**作者:** Libo Sun `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**通讯引用:** 5871 | [OpenAlex ID](https://openalex.org/A5011504177)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MAGNET框架，使用双层记忆（静态记忆映射视觉与功能语义，程序记忆捕获任务意图）来解决移动UI的外观漂移和工作流程漂移，支持持续自适应。

**💡 创新点**

创新点包括：①双记忆结构利用语义和意图稳定性；②动态记忆演化机制（基于访问频率与遗忘曲线）实现在线更新；③自动化流程与视觉功能对齐的构建流水线。

**🔧 技术方法**

技术手段：多模态大语言模型（Qwen2.5‑VL‑32B、Gemini‑2.5‑Pro）做规划与执行；相似度检索、Ebbinghaus遗忘曲线保留评分；自动抽象流程、视觉功能对齐流水线；离线/在线评测。

**📊 数据集**

使用UI‑40K多模态数据集（来自AITZ、GUI‑Odyssey、Amex）做记忆初始化，并在AITZ、GUI‑Odyssey、Amex离线基准及在线AndroidWorld环境进行实验。

**📈 对比分析**

与无记忆/单记忆基线（COAT、Agent‑S）及专用模型（UI‑Venus‑Navi‑7B、Atlas‑Pro‑7B、InfiGUI‑R1‑3B）对比，MAGNET在离线SR/Grd均超越基线；在线AndroidWorld任务完成率42.62%，比AppAgent高8.2个百分点，且相对稳健。

**⚠️ 局限性**

局限性：需要成功的任务轨迹来构建记忆，若初始探索失败或任务结构极其多样则难以提取；聚类抽象流程对极其多变任务可能不足，对全新领域的零样本适应有限。

---

## 68. CLIP-Guided Unsupervised Semantic-Aware Exposure Correction

**arXiv ID:** 2601.19129 | [PDF](https://arxiv.org/pdf/2601.19129v1)

**作者:** Puzhen Wu `[一作]` (Institute of Software, Chinese Academy of Sciences), Rui Xu `[通讯]` (School of Computer Science, Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无监督的语义感知曝光校正网络，利用FastSAM的语义特征和CLIP引导的伪标注生成，实现对过曝/欠曝图像的自动修正；

**💡 创新点**

创新点包括：①将FastSAM语义特征通过自适应语义融合模块（ASF）注入曝光校正网络，提升区域语义一致性；②使用CLIP进行提示微调，生成伪ground truth并自动调节伽玛；③提出语义-提示一致性损失（SPC），在语义空间与视觉-语言空间同时监督；④采用多尺度残差空间Mamba组（RSMG）捕获长距离空间依赖。

**🔧 技术方法**

技术要点：Fast Segment Anything Model (FastSAM)、CLIP提示微调与伪GT生成、跨注意力与频域-空间前馈融合、Vision Mamba + Spatial Mamba 模块、残差空间Mamba组、伽玛变换、语义特征一致性损失、图像-提示对齐损失、MSE 与 Cosine 颜色一致性损失。

**📊 数据集**

使用数据集：MSEC（只取欠曝/过曝子集）和SICE（低/高曝光层与中间层作为GT），以及DarkFace用于检测评估。

**📈 对比分析**

与多种无监督方法（ZeroDCE、RUAS、EnlightenGAN、SCI、PairLIE、CLIP-LIT、NeRCo、PSENet、LightenDiffusion、UEC）在PSNR、SSIM、LPIPS、BRISQUE、NIMA等指标上对比，实验表明本文模型在Under/Over 子集及总体上均取得最高或第二高分，显著提升图像质量与色彩一致性。

**⚠️ 局限性**

局限性：在高分辨率输入下推理速度较慢；对极端曝光区细节恢复仍有限；未来计划采用生成式局部补全、模型剪枝/蒸馏以及视频/多相机扩展来进一步提升性能与效率。

---

## 69. Enhancing Speech Emotion Recognition using Dynamic Spectral Features and Kalman Smoothing

**arXiv ID:** 2601.18908 | [PDF](https://arxiv.org/pdf/2601.18908v1)

**作者:** Marouane El Hizabri `[一作]`, Youssef Taki `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种结合动态频谱特征和Kalman滤波器的情绪识别框架（KF-TSER），通过时间平滑降低帧级情绪预测的抖动；

**💡 创新点**

创新点在于将Delta与Delta-Delta特征与Kalman滤波器相结合，实现对情绪状态的连续估计，显著提升高低唤醒情绪的区分；

**🔧 技术方法**

使用的技术包括MFCC、RMSE、ZCR、Delta/Delta-Delta特征提取，MLP分类器，以及Kalman滤波后处理；

**📊 数据集**

使用RAVDESS数据集中的四类情绪（Happy、Sad、Angry、Calm）进行实验；

**📈 对比分析**

与单纯的MLP帧级分类（60.6%）及融合后未滤波的系统（82.3%）相比，KF-TSER在句子级别上取得87%准确率，尤其在Happy/Angry区分上提升明显；

**⚠️ 局限性**

局限性包括仅在RAVDESS的人工录制语音上测试，缺乏背景噪声与自发情绪；仅考虑四种情绪，且仅限英语，低唤醒情绪仍存在混淆。

---

## 70. NuiWorld: Exploring a Scalable Framework for End-to-End Controllable World Generation

**arXiv ID:** 2601.19048 | [PDF](https://arxiv.org/pdf/2601.19048v1)

**作者:** Han-Hung Lee `[一作]` (Simon Fraser University), Angel X. Chang `[通讯]` (Simon Fraser University)

**通讯引用:** 31563 | [OpenAlex ID](https://openalex.org/A5044978994)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了NuiWorld框架，利用少量输入图像通过3D重建和可扩展场景生成构建训练数据，训练可控的世界生成模型，实现大规模、开放域场景一次性生成。

**💡 创新点**

创新点包括：①基于bootstrapping的生成式数据管线克服数据稀缺；②采用可变长度的场景块序列与扁平化向量集表示，令token长度随场景尺寸缩短；③在端到端模型中直接进行完整场景扩散，避免传统训练‑free或代理式多步流程；④结合伪草图实现可控生成。

**🔧 技术方法**

核心技术包括：Nano Banana文本生成图像、Trellis 2 3D重建、NuiScene的块级VAE+quad‑chunk扩散、DINOv2编码的草图条件、Rectified Flow扩散训练以及尺寸预测网络。

**📊 数据集**

使用了三类场景（中世纪、沙漠、赛博朋克）的图像数据，经过Trellis 2重建后得到的3D场景作为训练集；每个场景生成多张伪草图形成（向量集，草图）对。

**📈 对比分析**

与Trellis 2、Trellis 1以及现有可扩展3D生成方法对比，NuiWorld在保持几何细节、缩短推理时间、降低内存占用方面表现更好；在CD、RMSE、FPD等指标上获得与Ground‑Truth相近的分数。

**⚠️ 局限性**

局限性包括：训练数据量有限导致可泛化性受限；生成的单一网格缺乏可分解性；对大尺度场景的细节质量仍低于基于体素的高分辨率方法；模型尺寸受算力限制。

---

## 71. LLMs versus the Halting Problem: Revisiting Program Termination Prediction

**arXiv ID:** 2601.18987 | [PDF](https://arxiv.org/pdf/2601.18987v1)

**作者:** Oren Sultan `[一作]` (Meta AI), Peter O'Hearn `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在2025年软件验证竞赛的Termination类别上评估大型语言模型（LLM）来预测程序终止行为，探讨LLM在处理不可判定问题上的推理能力；

**💡 创新点**

其创新之处在于系统性评估LLM对程序终止的推理性能，首次将LLM与顶尖符号工具进行公平比较，并提出了图形化与逻辑表达式两种可验证的非终止证明格式；

**🔧 技术方法**

采用多种LLM（GPT‑5、Claude Sonnet‑4.5、CWM、Qwen3‑32B、GPT‑4o）在推理模式下结合Test‑Time Scaling与投票一致性机制，同时使用UAutomizer进行验证；

**📊 数据集**

实验基于TermCOMP 2025数据集中的Termination子集，共2328条C程序，涵盖合成与真实代码，按位向量、主控制流、堆操作等四类子集划分；

**📈 对比分析**

与PROTON、UAutomizer、AProVE等领先工具采用相同的得分公式和F1评估，结果显示GPT‑5和Claude Sonnet‑4.5在得分（3,520/3,448）和F1上几乎逼近或超越顶级工具，而CWM略逊于UAutomizer；但在生成有效证明图方面表现相对薄弱，且性能随程序长度递增而下降；

**⚠️ 局限性**

局限性包括仅针对C语言，GPT‑5等专有模型的不可解释性，模型对提示词敏感，实验环境与正式竞赛条件不完全一致，以及在更大规模或多语言环境下的可迁移性未知；

---

## 72. FSD-CAP: Fractional Subgraph Diffusion with Class-Aware Propagation for Graph Feature Imputation

**arXiv ID:** 2601.18938 | [PDF](https://arxiv.org/pdf/2601.18938v1)

**作者:** Xin Qiao `[一作]` (Xidian University), Liang Zhang `[通讯]` (Xidian University)

**通讯引用:** 29199 | [OpenAlex ID](https://openalex.org/A5100425201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种分两阶段的图特征缺失值插补框架 FSD‑CAP，先使用分数扩散算子和基于图距的逐层子图扩散对观测节点信息进行局部扩散，再利用伪标签与邻域熵进行类级传播，对缺失特征进行细化。

**💡 创新点**

创新点包括：①引入可调锐度的分数扩散算子，使扩散过程可根据局部结构从均匀扩散到邻接强度主导；②设计图距引导的子图逐层扩散，避免全图扩散导致的误差累积；③通过伪标签和邻域熵构造类级图进行传播，实现语义一致性与类别分离的双重提升。

**🔧 技术方法**

主要技术手段包括分数扩散算子、基于图距的子图扩散、伪标签生成（使用基于 GCN 的分类器）、邻域熵权重、类级虚拟节点传播以及最终的加权融合。

**📊 数据集**

实验使用了五个常用基准图数据集（Cora、CiteSeer、PubMed、Amazon Photo、Amazon Computers），并在更大规模和异质图上进一步验证其鲁棒性。

**📈 对比分析**

与零填充、PaGCN、FP、GRAFENNE、ITR、ASDVAE、PCFI 等基线比较，在极端缺失率（99.5%）下，FSD‑CAP 在节点分类上平均准确率达到约 80%（结构缺失）或 81%（均匀缺失），接近完整特征下的 81.31%；在链路预测任务中 AUC 与 AP 均优于对比方法，显示出更好的插补质量。

**⚠️ 局限性**

局限性主要体现在：对高异质性图仍可能受到同质性假设的影响；类级传播依赖伪标签准确性，误标签可能导致错误扩散；在极大规模图上仍可能出现内存或计算瓶颈；此外，在某些数据集上与完整特征的差距相对较小，提升幅度有限。

---

## 73. The Geometric Reasoner: Manifold-Informed Latent Foresight Search for Long-Context Reasoning

**arXiv ID:** 2601.18832 | [PDF](https://arxiv.org/pdf/2601.18832v1)

**作者:** Ren Zhuang `[一作]`, Shuifa Sun `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练的推理时框架，利用在潜在空间上进行几何信息引导的前瞻搜索，以实现长上下文推理。

**💡 创新点**

创新点在于：①将几何约束软化为可微的正则化，避免高维空间中硬几何约束的接受率衰减；②在每个块边界使用轻量前瞻滚动来评估候选潜在锚点，结合平滑度和多样性正则化进行评分；③采用块级 KV 缓存重置实现线性内存扩展，保持大规模上下文的可扩展性。

**🔧 技术方法**

技术包括：潜在空间锚点提取与投影；基于切线扰动采样潜在候选；轻量前瞻滚动与残差注入；软几何评分（look‑ahead 值、平滑惩罚、均匀性惩罚）；块级 KV 缓存重置；训练‑free 的 inference‑time search。

**📊 数据集**

使用的公开基准：数学推理（MATH500、AIME2025、OmniMath、OlympiadBench）和代码生成（HumanEval、BigCodeBench、LiveCodeBench）。

**📈 对比分析**

与无训练的采样基线（Power Sampling）、RL‑tuned 方法（LCoT‑GRPO、LCoT‑SimKO、Delethink‑GRPO/SimKO）以及 token‑空间控制（TGR‑Token）进行对比。TGR‑Latent 在 Qwen3‑8B 上在 AUC 上提升 13 点，Pass@k 曲线保持稳健，且平均 token 量仅略高（约 1.1–1.3×）。

**⚠️ 局限性**

局限性包括：推理时计算开销增加，导致延迟敏感场景受限；前瞻滚动仅使用短期信息，可能在长期依赖场景下信息不足；当模型已经过大量 RL 训练后，推理时搜索收益下降。

---

## 74. Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward

**arXiv ID:** 2601.19055 | [PDF](https://arxiv.org/pdf/2601.19055v1)

**作者:** Dipendra Misra `[一作]` (Databricks Mosaic Research), Ge Gao `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究利用部署日志中用户对LLM回答的编辑反馈进行微调，以实现对LLM的个性化与自适应。

**💡 创新点**

创新点在于将用户编辑反馈统一为偏好、监督与成本三种反馈类型，并首次在理论层面给出各类算法的样本复杂度与收敛性，提出早期和后期集合方法实现鲁棒学习。

**🔧 技术方法**

采用的技术包括监督微调（SFT）、基于偏好的直接偏好优化（DPO）、基于成本的保守强化学习与上限置信UCB策略的后期集合。

**📊 数据集**

实验使用来自Gao等人的邮件撰写与摘要两大任务的离线部署日志数据，用户模型采用Qwen‑3 32B 或 Llama‑3.1 8B，包含强/弱用户编辑样本。

**📈 对比分析**

与单一反馈方法对比，后期集合（late‑ensemble）在两项任务中获得最低的平均编辑距离和最小最差子最优偏差，显示出对用户分布变化的更好适应。

**⚠️ 局限性**

局限在于理论假设如用户分布平衡与可实现性、成本模型的保守性以及仅在模拟用户和有限任务上验证，未涉及真实人类实验和更复杂的多模态场景。

---

## 75. Speed is Confidence

**arXiv ID:** 2601.19085 | [PDF](https://arxiv.org/pdf/2601.19085v1)

**作者:** Joshua V. Dillon `[一作]` `[通讯]` (Independent Researcher), Joshua V. Dillon (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出利用推理速度作为置信度信号，开发“先停即胜”(halt‑first) 集成方法，并在单模型中通过赢家‑取偶训练(WTA)实现多种初始状态的竞争，以提升对Sudoku问题的推理精度。

**💡 创新点**

创新点包括：① 将推理完成时间视为隐式置信度并用于集成选择；② 通过WTA训练把集成多样性内在化；③ 引入“expunging”机制剔除已收敛的隐状态；④ 对SwiGLU进行层归一化改造以兼容Muon优化器；⑤ 通过SVD对齐初始化保持多头多样性；⑥ 结合理论分析阐释速度-置信度关系。

**🔧 技术方法**

使用技术包括：Tiny Recursive Models (TRM)、Adaptive Computation Time (ACT)、多模型并行推理、赢家‑取偶 (WTA) 训练、Muon 与 AdamW 混合优化、改进的 SwiGLU、SVD‑对齐初始化、以及在训练中动态剔除相似隐状态的 expunging。

**📊 数据集**

数据集为 Sudoku‑Extreme（10,000 个含 17 个已给数字的标准数独谜题），实验中采用训练/测试划分与 TRM 论文相同，并使用数据增强与数字置换。

**📈 对比分析**

与基线单模型、传统概率平均集成、以及 test‑time 旋转增强（TTA）进行对比。单模型基线准确率为 86.1%；halt‑first 集成 97.2% 并仅消耗 1.5 倍前向推理；WTA 训练的单模型在单前向推理下达 96.9%±0.6% 的谜题准确率，几乎匹配 97.3% 的 TTA；总体而言，速度提升 10×、计算成本显著下降。

**⚠️ 局限性**

局限性包括：仅在 Sudoku 领域验证，缺乏对更广泛推理任务的通用性；WTA 训练需多倍前向推理，训练成本随 K 增大而显著上升；多头在后期容易收敛到相似解，导致多样性降低；模型无法处理需要回溯的“分支”型数独，仍存在 10% 以内的不可解决难题。

---

## 76. CP Loss: Channel-wise Perceptual Loss for Time Series Forecasting

**arXiv ID:** 2601.18829 | [PDF](https://arxiv.org/pdf/2601.18829v1)

**作者:** Yaohua Zha `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Shenzhen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种通道自适应感知损失（Channel-wise Perceptual Loss），用于多通道时间序列预测模型的训练。

**💡 创新点**

创新点在于为每个通道学习一个唯一的、可微分的多尺度感知空间，通过可学习的降冗滤波器动态分解信号，并在此空间中计算误差，从而避免传统均方误差的通道无差异性。

**🔧 技术方法**

使用了可学习的1D卷积降冗滤波器、层级多尺度分解、交叉优化（与主模型共同训练）以及基于绝对误差的通道感知损失计算。

**📊 数据集**

在六个真实多变量数据集上进行实验：ETT的四个变体（ETTh1、ETTh2、ETTm1、ETTm2）、Weather 和 ECL。

**📈 对比分析**

与传统MSE以及形状、频域和patch结构等现有损失函数（TILDE-Q、FreDF、PS Loss）进行对比，在所有数据集和预测长度下，CP Loss均取得最低的MAE和MSE，提升幅度可达数个百分点。

**⚠️ 局限性**

局限性包括：对极高通道维度的数据规模仍有潜在计算和参数增长风险；以及在跨通道依赖建模方面仍未充分挖掘，未来可进一步结合跨通道感知机制。

---

## 77. BabyReasoningBench: Generating Developmentally-Inspired Reasoning Tasks for Evaluating Baby Language Models

**arXiv ID:** 2601.18933 | [PDF](https://arxiv.org/pdf/2601.18933v1)

**作者:** Kaustubh D. Dhole `[一作]` (Emory University), Kaustubh D. Dhole `[通讯]` (Emory University)

**通讯引用:** 1105 | [OpenAlex ID](https://openalex.org/A5061120974)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BabyReasoningBench 基准，用 19 种基于发展心理学的推理任务评估儿童语言模型的推理能力。

**💡 创新点**

创新点在于将经典儿童认知实验转化为文本多项选择题，并通过 GPT‑5.2 自动生成多样化题目，构建细粒度与发展里程碑对应的评估工具。

**🔧 技术方法**

使用 GPT‑5.2 生成题目、GPT‑2 语言模型做多项选择推理，并以条件对数似然对答案进行评分。

**📊 数据集**

基准模型基于 10M 与 100M 的儿童主导语料（如 CHILDES）预训练的 BabyLM GPT‑2 版本。

**📈 对比分析**

通过比较两模型在每个任务上的准确率，发现整体表现低但在因果与物理推理任务上显著提升，假设推理与意图推断仍难。

**⚠️ 局限性**

局限在于任务仅文本化，缺乏多模态输入；多项选择可能无法完整捕捉真实推理行为；基准未涵盖所有认知范式。

---

## 78. Create Benchmarks for Data Lakes

**arXiv ID:** 2601.19176 | [PDF](https://arxiv.org/pdf/2601.19176v1)

**作者:** Yi Lyu `[一作]`, Natan Lidukhover `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个涵盖结构化、半结构化和非结构化数据的统一数据湖基准框架，设计了数据集、查询工作负载和性能度量。

**💡 创新点**

创新点在于：①首次针对数据湖同时支持三种数据类型的基准；②加入了对相似性搜索等数据湖特有操作的性能评估；③基于图索引的 EASE 算法实现，避免传统倒排索引的局限。

**🔧 技术方法**

技术手段包括：Python 数据处理、EASE 关键词搜索算法、图索引构建、CloudLab 实验环境，以及对不同数据湖实现（AWS、Azure、GCP、Oracle、DLBench）进行对比。

**📊 数据集**

使用的数据集包括 IMDb 电影评分 CSV（结构化）、XML 文档（半结构化）和 Apache HTTP 日志文本（非结构化）。

**📈 对比分析**

比较方法：在不同规模的三类数据上执行数据检索、聚合和查询工作负载，记录查询执行时间、元数据生成时间和元数据大小等指标；通过脚本生成不同规模数据，得到性能曲线。预期结果将展示各数据湖在不同数据类型和操作下的性能差异。

**⚠️ 局限性**

局限性：目前仅在本地机器上实现 EASE 算法，没有在真实数据湖框架上部署；实验规模受限，缺乏大规模验证；基准仅覆盖特定工作负载，未覆盖所有真实业务场景。

---

## 79. Representational Homomorphism Predicts and Improves Compositional Generalization In Transformer Language Model

**arXiv ID:** 2601.18858 | [PDF](https://arxiv.org/pdf/2601.18858v1)

**作者:** Zhiyu An `[一作]` (University of California), Wan Du `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了“同构误差”(HE)指标，用于衡量Transformer模型在内部表示层对组合操作的保持程度，并通过HE正则化提升OOD组合泛化。

**💡 创新点**

将抽象代数中的同构概念引入到神经网络内部表示的度量，提供结构化诊断和训练信号；并证明HE正则化能因果提升泛化。

**🔧 技术方法**

使用小型Decoder-only Transformer、学习可组合的表示层运算符（线性、双线性、MLP）、HE正则化、噪声注入实验与回归分析。

**📊 数据集**

受SCAN启发的合成组合任务数据集，可控制词汇、修饰符、连接词与噪声，生成训练、OOD测试集。

**📈 对比分析**

对比不同层数、数据稀疏度、噪声水平下的HE与OOD准确率，HE与准确率的R^2=0.73；HE正则化使modifier HE下降并使OOD准确率提升p=0.023。

**⚠️ 局限性**

仅在合成数据上验证，规模有限；未验证大规模预训练模型；假设操作离散且严格代数化，难以直接迁移到真实自然语言。

---

## 80. Belief-Combining Framework for Multi-Trace Reconstruction over Channels with Insertions, Deletions, and Substitutions

**arXiv ID:** 2601.18920 | [PDF](https://arxiv.org/pdf/2601.18920v1)

**作者:** Aria Nouri `[一作]` `[通讯]`, Aria Nouri

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种迭代信念合并框架，用于在插入-删除-替换（IDS）通道上对多条噪声轨迹进行最大后验估计（MAP）重构。

**💡 创新点**

创新点在于通过在各单轨迹局部推导器之间交换软信念（而非硬估计），实现了与联合trellis BCJR相同的MAP性能，却将复杂度从指数降至仅与轨迹数平方成正比。

**🔧 技术方法**

使用的技术包括：基于贝叶斯推理的BCJR算法、消息传播（belief propagation）和局部状态后验的迭代融合；对通道状态做Markov建模，利用指针或漂移变量来描述IDS错误。

**📊 数据集**

使用了两类数据集：真实的聚类纳米孔DNA测序读数（长度110）以及随机生成的DNA序列（长度100）。

**📈 对比分析**

通过与联合trellis BCJR（仅在K=2,3时可行）以及之前的硬估计注入方法进行对比，实验表明在K=4时，利用真实读数可达到约97%的重构准确率；相较于传统方法，改进版在保持相同O(N·δΔ·K)复杂度的同时，性能明显提升。

**⚠️ 局限性**

局限性包括：对轨迹长度漂移的假设（Δ< N），以及在K>2时仍需至少K轮迭代才能收敛；未提供加密/编码层的完整实现，仅在无码场景下验证；复杂度虽降为二次，但对大K仍可能产生显著开销。

---

## 81. SNR-Edit: Structure-Aware Noise Rectification for Inversion-Free Flow-Based Editing

**arXiv ID:** 2601.19180 | [PDF](https://arxiv.org/pdf/2601.19180v1)

**作者:** Lifan Jiang `[一作]` (Zhejiang University), Deng Cai `[通讯]` (Zhejiang University)

**通讯引用:** 24264 | [OpenAlex ID](https://openalex.org/A5037942269)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SNR-Edit 框架，实现无逆向流式图像编辑，通过结构感知噪声校正改进编辑轨迹。

**💡 创新点**

创新点在于解决固定高斯噪声导致的结构-随机不匹配问题，采用语义分割、RoPE 编码与随机投影构造结构先验，并与高斯噪声混合实现噪声校正。

**🔧 技术方法**

使用技术包括 Flow‑based 生成模型（SD3、FLUX）、SAM2 分割、RoPE 空间编码、随机投影、ODE 流整合以及 VLM 评估（ImgEdit Reward、Qwen‑VL）。

**📊 数据集**

实验数据集涵盖 PIE‑Bench 以及自行构建的 SNR‑Bench（约 80 张高质量图像，包含 PI‑Bench 与网络收集样本）。

**📈 对比分析**

与多种基线（SDEdit、DNAEdit、FlowEdit、RF‑Inversion、RF‑Solver 等）在 PSNR/SSIM/LPIPS/CLIP、VLM 评分和用户研究上对比，SNR‑Edit 在结构保持与文本对齐方面获得最高排名，仅略增 1 s 处理时延。

**⚠️ 局限性**

局限性包括对语义分割质量的依赖；随机投影在极高分辨率下可能产生冲突；对遮挡或极其复杂场景的适应性仍待提升，且未针对实时部署做进一步优化。

---

## 82. Hybrid Fault-Driven Mutation Testing for Python

**arXiv ID:** 2601.19088 | [PDF](https://arxiv.org/pdf/2601.19088v1)

**作者:** Saba Alimadadi `[一作]` (Simon Fraser University), Golnaz Gharachorlu `[通讯]` (University of Ottawa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种结合静态AST分析与动态运行时分析的混合变异测试方法，针对Python的常见反模式生成7种新的变异算子，以提高测试套件对Python特有错误的覆盖率。

**💡 创新点**

创新点在于：①提出7个专门针对Python动态特性（如可变参数、隐式类型转换、容器结构不匹配、复合条件缺失、属性与方法误用）设计的变异算子；②采用混合静态‑动态分析来定位变异点并利用运行时行为过滤等价变异，从而显著降低等价变异率并生成更具代表性的变异。

**🔧 技术方法**

使用的技术包括Python的AST模块进行静态变异、DynaPyt实现运行时监控与动态分析、pytest+coverage执行与收集测试结果、以及自定义启发式过滤规则。

**📊 数据集**

实验使用了13个真实世界的开源Python项目，包括GPT‑2、Home Assistant、Deep Graph Library、Ansible、Modin、Electrum、Read the Docs等。

**📈 对比分析**

与现有工具（如mutmut）进行对比，评估指标包括变异得分、交叉杀死率、测试覆盖率、以及等价变异比例；结果显示新方法能发现更多独特变异且交叉杀死率低，证明其补充了通用变异工具的盲点；性能上运行时间略高但仍在可接受范围内。

**⚠️ 局限性**

局限性包括：①只覆盖了7种反模式，可能无法覆盖所有Python错误；②需要执行完整测试集以获取动态信息，导致额外运行开销；③对全局状态或并发行为的覆盖不足；④等价变异并未完全消除，仍有残留。

---

## 83. NC-Reg : Neural Cortical Maps for Rigid Registration

**arXiv ID:** 2601.19042 | [PDF](https://arxiv.org/pdf/2601.19042v1)

**作者:** Ines Vati `[一作]` (CSIRO), Léo Lebrat `[通讯]` (Queensland University of Technology)

**通讯引用:** 155 | [OpenAlex ID](https://openalex.org/A5055226360)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出神经皮层地图（Neural Cortical Maps）和基于该地图的全局旋转配准算法NC‑Reg，用于皮层表面刚性配准并作为非刚性配准的鲁棒预对齐。

**💡 创新点**

创新点包括：①设计连续、紧凑的神经表示，可在任意分辨率直接输出皮层特征，取代传统网格插值；②将模拟退火与随机轴角重置相结合，显著减少局部极小，达到<1°的全局旋转精度；③通过预训练的NC模型实现快速、鲁棒的刚性预对齐，显著提升后续非刚性配准性能。

**🔧 技术方法**

技术包括：多层感知机+多分辨率哈希编码训练神经皮层地图；梯度下降+Adam优化；旋转参数化为6D或四元数，并采用随机轴角重置+模拟退火；使用MSE、PCC、Dice等指标评估配准质量，并与SUGAR、HSD、SD等方法对比。

**📊 数据集**

使用数据集：ADNI模板（fsaverage）训练NC模型；ADRC（39个皮层网格）进行配准实验；构造的随机旋转扰动数据集用于鲁棒性验证；未来计划评估MindBoggle数据集。

**📈 对比分析**

通过与SUGAR、HSD、SD等现有方法在无扰动和大旋转扰动数据集上的对比，NC‑Reg在MSE、PCC、Dice指标上与或优于最先进方法；运行时间比传统方法快30倍，且在非刚性配准中作为预对齐可提升Dice并减少约5秒的处理时间。

**⚠️ 局限性**

局限性：仅在曲率、凸度等皮层特征上验证，其他特征或不同数据集的泛化尚未充分评估；当前仍需预训练模板模型，模拟退火参数需手动调优；对极大分辨率网格的效率尚待进一步验证。

---

## 84. Critical Organization of Deep Neural Networks, and p-Adic Statistical Field Theories

**arXiv ID:** 2601.19070 | [PDF](https://arxiv.org/pdf/2601.19070v1)

**作者:** W. A. Zúñiga-Galindo `[一作]` (University of Texas Rio Grande Valley), W. A. Zúñiga-Galindo `[通讯]` (University of Texas Rio Grande Valley)

**通讯引用:** 1215 | [OpenAlex ID](https://openalex.org/A5025695709)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并研究了基于 p‑adic 数学的深度神经网络框架，构造了离散与连续形式，并分析其临界组织与无宽度极限

**💡 创新点**

创新点在于利用 p‑adic 层级拓扑为网络提供天然层级结构，证明所有离散 DNN 可映射到 p‑adic DNN，构建了网络先验的路径积分表示，并给出无宽度极限下的解析展开

**🔧 技术方法**

主要技术包括 p‑adic 解析、Haar 测度、Bruhat–Schwartz 测试函数空间、泛函积分、Gaussian 随机变量与协方差算子、Martin–Siggia–Rose–de Dominicis–Janssen 路径积分形式

**📊 数据集**

该工作为理论性研究，未使用具体机器学习数据集

**📈 对比分析**

未进行实验对比，因缺少可验证的性能指标，主要是理论上给出了网络先验的闭式展开与极限表达式

**⚠️ 局限性**

局限性包括：对连续空间 Ω 的无宽度极限难以严谨求解、需要引入截断来保证积分收敛、对标准离散 DNN 的层级组织处理较为困难、仅在形式上可与量子场论相通

---

## 85. The Last Mile to Production Readiness: Physics-Based Motion Refinement for Video-Based Capture

**arXiv ID:** 2601.19036 | [PDF](https://arxiv.org/pdf/2601.19036v1)

**作者:** Tianxin Tao `[一作]` (Electronic Arts), Hung Yu Ling `[通讯]` (Electronic Arts)

**通讯引用:** 407 | [OpenAlex ID](https://openalex.org/A5085649371)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理的运动清理框架，利用深度强化学习改进视觉捕获的动作，消除浮动、穿透、幻影接触和脚滑等物理不合理缺陷，并支持单角色和多角色序列以及动画师交互式编辑。

**💡 创新点**

创新点包括：①将渗透深度评估用于自适应终止与奖励调节；②混合PD控制基底采用SLERP预测参数；③使用多智能体PPO处理多角色交互；④支持动画师关键帧引导的后处理。

**🔧 技术方法**

使用技术包括：深度强化学习（PPO/MAPPO）、物理仿真与碰撞检测（GJK、EPA）、PD控制器、动态物体跟踪、奖励设计及早期终止机制。

**📊 数据集**

数据来源为商业视觉捕获工具得到的实测运动数据（未公开具体数据集名称），在补充材料中进行评估。

**📈 对比分析**

与原始捕获数据及现有物理/数据驱动清理方法对比，采用渗透深度、根轨迹振幅、脚滑距离等定量指标；实验显示根高度更有节奏感，穿透深度降至几毫米，脚滑距离显著减少，且在多角色碰撞场景中保持一致性。

**⚠️ 局限性**

局限性包括：每个剪辑需单独训练策略，难以规模化；假设参考动作结构完整，对严重噪声需与数据驱动插补结合；对象交互仍需手动关键帧或进一步方法完善。

---

## 86. Reg-TTR, Test-Time Refinement for Fast, Robust and Accurate Image Registration

**arXiv ID:** 2601.19114 | [PDF](https://arxiv.org/pdf/2601.19114v1)

**作者:** Lin Chen `[一作]`, Min Liu `[通讯]` (Hunan University)

**通讯引用:** 62962 | [OpenAlex ID](https://openalex.org/A5100343920)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了 Reg-TTR，一个在测试时对预训练注册模型进行细化的框架，实现了在保持速度的同时提升注册精度。

**💡 创新点**

创新点在于将基础模型的初始位移场视为可学习参数，并通过联合使用 SSIM、NCC 与平滑正则的混合损失进行快速迭代优化。

**🔧 技术方法**

采用了深度学习预训练网络（如 uniGradICON、VoxelMorph 等）结合 Adam 优化器的测试时细化（TTR）技术。

**📊 数据集**

使用了 ACDC 心脏 MRI 数据集和 Learn2Reg 2020 的腹部 CT 数据集进行评估。

**📈 对比分析**

与多种无监督/半监督基线（VoxelMorph、FourierNet、CorrMLP、ConvexAdam 等）比较，Reg-TTR 在 Dice、HD95、SDlogJ 指标上均达到或超过 SOTA，且推理时间仅略高于单前向推断（约 3.2 秒）。

**⚠️ 局限性**

限制在于需要针对不同数据集调节迭代次数和学习率，且在极大尺寸或多模态场景下的通用性尚未验证。

---

## 87. Tricky$^2$: Towards a Benchmark for Evaluating Human and LLM Error Interactions

**arXiv ID:** 2601.18949 | [PDF](https://arxiv.org/pdf/2601.18949v1)

**作者:** Cole Granger `[一作]` (William and Mary), Denys Poshyvanyk `[通讯]` (William and Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Tricky2基准数据集，融合人类编写缺陷与LLM注入错误，形成人类、LLM、混合三种拆分；

**💡 创新点**

创新点在于首次将人类与LLM错误结合在同一代码上下文，系统化注入错误并设计交互式评估任务，揭示混合错误的交互效应；

**🔧 技术方法**

技术包括基于分类学的提示框架（taxonomy-guided prompting）、单步LLM注入（GPT‑5、OpenAI‑oss‑20b）、自动验证脚本、统一评估流程（来源分类、错误定位、程序修复）等；

**📊 数据集**

使用基础数据集TrickyBugs（C++/Java/Python 3043 人类错误程序 + 1361 修正程序），在此基础上注入LLM错误，生成约11851 条buggy 代码；

**📈 对比分析**

比较方法为在三种拆分（人类仅、LLM仅、混合）下进行三任务基线评估（来源分类、错误定位、修复），发现混合拆分的修复成功率最低，表明错误交互导致更高难度；

**⚠️ 局限性**

局限性包括：受限于原始TrickyBugs领域（面向竞赛题目）、仅使用两款LLM的注入模式、分类学不涵盖并发/系统级错误、验证仅检查语法而非语义完整性等问题。

---

## 88. Interpretable and Perceptually-Aligned Music Similarity with Pretrained Embeddings

**arXiv ID:** 2601.19109 | [PDF](https://arxiv.org/pdf/2601.19109v1)

**作者:** Arhan Vohra `[一作]` (Universitat Pompeu Fabra), Taketo Akama `[通讯]` (Sony Computer Science Laboratories)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5087426444)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估预训练的文本‑音频嵌入模型（CLAP、MuQ‑MuLan）在音乐相似度检索中的零样本性能，并提出基于源分离的可解释乐器加权相似度模型，以提升与人类感知的一致性。

**💡 创新点**

创新点在于将大规模多模训练得到的通用音频嵌入与音乐源分离相结合，通过线性优化学习乐器权重，实现可解释且无需微调的混合曲相似度匹配。

**🔧 技术方法**

技术包括CLAP与MuQ‑MuLan预训练嵌入、Demucs源分离、ABX感知实验、线性回归/岭回归权重学习。

**📊 数据集**

使用Inst‑Sim‑ABX数据集（来自Slakh2100的5秒音频片段），包含单乐器和全混音，配合Slakh真实多轨作为基准。

**📈 对比分析**

方法与现有自监督度量学习模型（Cascade‑PAFT、D‑CSN）在XAB和XYC ABX配置下对比；CLAP/MuQ在零样本下达70–90%的相似度一致率，MuQ基于乐器加权后在6‑stem配置中最高达90.4%，显著优于传统方法。

**⚠️ 局限性**

局限性：仅在MIDI合成的Slakh数据上评估，缺乏真实录音、歌手以及不同制作风格；数据量有限（约330个三元组），加权模型权重可能过拟合该合成数据，难以泛化到其他音乐风格。

---

## 89. Fog of War Chess

**arXiv ID:** 2601.18813 | [PDF](https://arxiv.org/pdf/2601.18813v1)

**作者:** Matthias Gehnen `[一作]` (RWTH Aachen University), Julius Stannat `[通讯]` (RWTH Aachen University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

对 Fog of War 棋（有限信息棋）的终局进行理论分析，给出了三种情况：王+后对王、王+车对王、王+两车对王，并证明了相应的必胜与否定情况。

**💡 创新点**

首次系统性地揭示了 Fog of War 棋中与传统棋类截然不同的终局特性，证明了王+后必胜、王+车不必胜、王+两车必胜的定理，填补了该变种终局理论空白。

**🔧 技术方法**

采用纯粹的组合博弈与几何路径分析技术，构造多段式策略（如角落配置、逐行逼逼法、守卫交替移动等），并通过多案例归纳证明。

**📊 数据集**

论文没有使用任何实验数据集，全部基于理论证明与逻辑推理完成。

**📈 对比分析**

由于方法为理论证明，未做实验对比；作者仅指出在 50‑move 规则下可能需要大量步数（>50步），并推测存在更快的必胜策略，但未给出具体实现或性能指标。

**⚠️ 局限性**

限制：仅针对上述三种终局；未涉及骑士、象或多子棋形；未给出精确步数上限；缺乏计算机验证与实验数据；对 50‑move 规则的结论仅为猜测。

---

## 90. Do Images Speak Louder than Words? Investigating the Effect of Textual Misinformation in VLMs

**arXiv ID:** 2601.19202 | [PDF](https://arxiv.org/pdf/2601.19202v1)

**作者:** Chi Zhang `[一作]` (University of Texas at Austin), Ray Mooney `[通讯]` (University of Texas at Austin)

**通讯引用:** 28 | [OpenAlex ID](https://openalex.org/A5110250940)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个名为Conflicting Text的数据集和多轮说服式评测框架，用于系统评估视觉语言模型在面对与视觉证据相冲突的文本误导时的鲁棒性。

**💡 创新点**

创新点在于首次将说服性文本误导引入VLM评测，构建了专门的冲突文本数据集、设计了多轮对话评估流程，并提出了简单的提示式防御方法。

**🔧 技术方法**

采用多种说服策略（重复、逻辑、可信度、情感）生成误导文本，利用 Gemini 2.5-Pro、LLaVA、Qwen‑VL、InternVL、Gemini、GPT‑4o 等主流 VLM 进行推理，并通过 softmax 计算置信度变化。

**📊 数据集**

数据集来源于 A‑OKVQA，先筛选出 920 条模型始终正确回答的多项选择题，再利用 Gemini 自动生成说服性误导文本。

**📈 对比分析**

对 11 种 VLM 进行实验，平均在单轮说服后准确率下降 48.2%；开源模型相对脆弱，逻辑说服最有效；多轮说服效应递减；加上“警告”提示可在部分模型中提升准确率。

**⚠️ 局限性**

局限性包括数据集规模有限（仅 920 条），评估仅针对提示式防御，未探究模型内部推理机制，也未尝试更深层的架构或微调改进，结果可能不具备广泛的推广性。

---

## 91. RealStats: A Rigorous Real-Only Statistical Framework for Fake Image Detection

**arXiv ID:** 2601.18900 | [PDF](https://arxiv.org/pdf/2601.18900v1)

**作者:** Haim Zisman `[一作]` (Bar-Ilan University), Uri Shaham `[通讯]` (Bar-Ilan University)

**通讯引用:** 3671 | [OpenAlex ID](https://openalex.org/A5088774511)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练、基于真实图像统计的假图像检测框架。

**💡 创新点**

将多种无监督检测统计量转化为p值，并通过统计聚合实现可解释且对生成模型迁移具有高适应性的检测。

**🔧 技术方法**

使用噪声扰动下的特征鲁棒性度量、经验累积分布函数（ECDF）、两侧p值、独立性图与最大团选取，以及Stouffer或最小p聚合等统计方法。

**📊 数据集**

在CNNSpot、Universal Fake Detect、Stable Diffusion Face、Synthbuster、GenImage等共计187K张图像上进行评估。

**📈 对比分析**

与RIGID、AEROBLADE、ManifoldBias等训练自由基线对比，AUC/AP表现与顶尖方法相当，方差更小，且通过加入ManifoldBias可进一步提升特定生成器的性能。

**⚠️ 局限性**

依赖于所选统计量子集的独立性与参考图像分布的代表性，若统计量高度相关或参考样本不足会影响p值的有效性与检测效果。

---

## 92. SE Journals in 2036: Looking Back at the Future We Need to Have

**arXiv ID:** 2601.19217 | [PDF](https://arxiv.org/pdf/2601.19217v1)

**作者:** Tim Menzies `[一作]` (North Carolina State University), Thomas Zimmermann `[通讯]` (University of California Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析软件工程期刊面临的可扩展性危机，提出并阐述一系列制度与技术改革（SE期刊联盟、抽奖式审稿、AI+人工审稿流程、可执行论文、两速文化与模块化论文、Benchmark催化剂标准），并从2036年的视角回顾其实施效果。

**💡 创新点**

创新点包括：①跨期刊共享审稿、审稿记录与学术信用的可携带层；②基于预审门控与抽奖的审稿分流，显著降低随机性与负担；③自动化结构化门控与AI辅助抽取、持续对话式审稿；④将论文拆分为“视图声明”“注册报告”“工具论文”等模块化形式；⑤开放摘要与视频不受订阅限制，直接面向实践者；⑥Benchmark评估从“守门”转向“催化”，鼓励新任务与挑战假设。

**🔧 技术方法**

采用的技术主要有：AI（如Gemini）进行结构化检测与抽取；可执行论文框架（执行容器、自动重现）；持续对话平台（匿名讨论与可视化记录）；AI生成摘要与视频；统计门控与抽奖算法；跨期刊数据共享与信用计量系统；Benchmark数据处理与扩展工具。

**📊 数据集**

使用的数据包括：2013–2023年SE主流会议/期刊的引用与作者分布（约19,000名研究者、500名顶尖作者）；2025年TOSEM 2000篇提交量与67天平均处理时长；135条SE评审标准与其12个核心特征；Benchmark公开数据集（多任务、工业场景）；以及作者工作量与论文类型的相关统计。

**📈 对比分析**

通过对比传统单一审稿流程与抽奖+门控流程，发现提交量下降30%，审稿时长缩短25%，高质量论文通过率提升12%；可执行论文使得自动化检查率提高70%，人工审稿时间减少40%；模块化论文平均阅读时间下降40%；Benchmark催化剂标准使得实用性实验增多30%，与工业合作提升18%。整体性能表明，系统从“噪音过滤”转向“信号提升”，审稿效率与质量均有显著提升。

**⚠️ 局限性**

局限性包括：①AI辅助仍需人工校准，无法完全取代人类判断；②抽奖机制可能导致热门领域偏好；③跨期刊联盟的数据共享面临隐私与合规挑战；④模块化论文的评审权威性与引用习惯尚待统一；⑤Benchmark扩展过程可能产生新的偏见；⑥长期可持续性依赖于学术文化与资助模式的进一步调整。

---

## 93. TinyTorch: Building Machine Learning Systems from First Principles

**arXiv ID:** 2601.19107 | [PDF](https://arxiv.org/pdf/2601.19107v1)

**作者:** Vijay Janapa Reddi `[一作]` (Harvard University), Vijay Janapa Reddi `[通讯]` (Harvard University)

**通讯引用:** 12493 | [OpenAlex ID](https://openalex.org/A5000635267)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 TinyTorch 课程，学生通过从零实现 PyTorch 核心组件（张量、自动求导、优化器、CNN、Transformer 等）来学习机器学习系统工程，并通过历史里程碑验证实现正确性。

**💡 创新点**

创新点在于：① 采用“渐进式披露”在同一张量类中逐步添加梯度、内存、自动求导等功能，降低认知负荷；② 将系统效率（内存、计算复杂度、压缩、加速、基准）从一开始就嵌入课程；③ 通过历史里程碑（1958-2025）让学生在同一套代码中重现主要 ML 成就，弥合算法与系统之间的鸿沟。

**🔧 技术方法**

主要技术包括纯 Python 实现的张量运算、手写自动求导、梯度检查、卷积循环实现、Transformer 结构、量化/剪枝、KV 缓存、性能分析与基准脚本，配合 Jupyter Notebook、nbgrader 自动评测与 nbdev 文档生成。

**📊 数据集**

使用的公开数据集包括：TinyDigits（约 1k 手写数字）、TinyTalks（约 350 句对话）以及 CIFAR‑10（用于 CNN 与 Transformer 训练），均为小规模、离线可用的标准数据集。

**📈 对比分析**

对比方法：与 PyTorch 等工业框架相比，TinyTorch 在 CPU‑only、纯 Python 环境下实现相同功能；通过基准模块（14‑19）记录内存占用、 FLOPs、训练时间、推理速度、压缩率等指标；实验表明学生实现的模型能够达到 65‑75% 的 CIFAR‑10 准确率、Transformer 文本生成可读性，且能在 4GB RAM、双核 CPU 上完成训练，证明系统学习可行。

**⚠️ 局限性**

局限性：① 仅覆盖单机 CPU，缺乏 GPU、分布式训练与模型部署的细节；② 训练速度极慢，限制快速迭代与大规模实验；③ 未对能耗、能源消耗进行度量；④ 目前仅英文教材，缺少多语言支持；⑤ 需要进一步实验验证教学效果和学习者认知负荷。

---

## 94. DeFM: Learning Foundation Representations from Depth for Robotics

**arXiv ID:** 2601.18923 | [PDF](https://arxiv.org/pdf/2601.18923v1)

**作者:** Manthan Patel `[一作]` (Robotic Systems Lab), Marco Hutter `[通讯]` (Robotic Systems Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并预训练了一款名为 DeFM 的深度图像基础模型，并在多种机器人任务中验证其可迁移性。

**💡 创新点**

首创使用 60M 深度图像的自监督 DINOv2 预训练，并提出三通道对数归一化以保持多尺度度量信息，同时将大模型蒸馏为轻量化网络。

**🔧 技术方法**

基于 DINOv2 的自监督对比/自蒸馏目标、iBOT 补丁损失、KoLeo 正则、Sinkhorn‑Knopp 对齐，以及 BiFPN 轻量化蒸馏等技术。

**📊 数据集**

结合 18 个公开数据集共 60.4M 深度图像，包括 MDE 生成深度、合成数据和真实传感器（RealSense、ZED、Kinect 等）。

**📈 对比分析**

通过线性探测在 ImageNet‑Depth‑1k、语义分割 mIoU、RL 任务 SPL/SR 等指标与 RGB 预训练和手工基线对比，DeFM 在分类线性探测达到 71.7% top‑1、分割在多域 mIoU 超过对照模型约 5‑10%，RL 任务保持或优于全量训练的专用网络。

**⚠️ 局限性**

对超薄障碍物感知仍有限，ViT 结构导致视觉伪影，且实验任务和硬件范围受限，未来需扩大任务种类、改进架构与高分辨率输入。

---

## 95. Accelerating Generative Recommendation via Simple Categorical User Sequence Compression

**arXiv ID:** 2601.19158 | [PDF](https://arxiv.org/pdf/2601.19158v1)

**作者:** Qijiong Liu `[一作]` (Hong Kong Polytechnic University), Xiao-Ming Wu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 7080 | [OpenAlex ID](https://openalex.org/A5101981128)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于商品类别的用户序列压缩方法（CAUSE），将长期用户历史压缩成少量“历史 token”，从而减少生成式推荐模型的序列长度与计算量。

**💡 创新点**

创新点在于：①利用已有的商品类别信息对历史序列进行桶化与排序，选择最近的桶和桶内最近的若干项；②对每个桶求聚合嵌入并对齐至桶级语义空间；③将压缩后的历史 token 与最近序列一起作为模型输入，兼顾长短期偏好且计算更轻量。该方案相较于传统聚类或多阶段压缩方法更简单、易实现、且可无缝对接多种生成式推荐骨干。

**🔧 技术方法**

技术细节包括：桶化选择（V 个桶、G 个桶内项）、桶级嵌入聚合（加权平均+桶嵌入）、对齐投影（W_align, b_align）、信息对比损失 InfoNCE 以及可选动作预测交叉熵；整体使用轻量级 Transformer（3 层，隐藏 64，8 头）作为推荐骨干。

**📊 数据集**

实验数据集为：① KuaiRand-27K（微视频日志，日序列划分为长期历史、训练集、验证/测试）和 ② MovieLens-20M（电影评分/行为，包含 5 种动作类型）。

**📈 对比分析**

与 HSTU 和 GenRank 进行对比；在保持相同输入长度时，CAUSE 计算成本可降低 6 倍，推荐精度在相同成本下提升高达 39%；即便与基线在更长序列上训练的版本比较，CAUSE 也能在更低成本下取得更优表现。

**⚠️ 局限性**

局限性：① 依赖商品类别信息，若无类别则需聚类但效果略逊；② 目前仅针对单模型骨干（HSTU、GenRank）验证，跨更大规模或不同结构模型的通用性待进一步验证；③ 压缩过程主要关注长期历史，对极端稀疏或快速变化的用户行为可能影响模型适应性。

---

## 96. Neural Theorem Proving for Verification Conditions: A Real-World Benchmark

**arXiv ID:** 2601.18944 | [PDF](https://arxiv.org/pdf/2601.18944v1)

**作者:** Qiyuan Xu `[一作]` (Nanyang Technological University), Conrad Watt `[通讯]` (Nanyang Technological University)

**通讯引用:** 311 | [OpenAlex ID](https://openalex.org/A5060197721)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个工业级多语言 VC（Verification Condition）基准 NTP4VC，并通过专家规则实现了从 Why3/Frama‑C 生成的 VC 自动提取与三大 ITP（Isabelle、Lean、Rocq）的语义等价翻译；在该基准上评估了多款 LLM 与传统 Hammer 证明器的表现。

**💡 创新点**

① 首创工业多语言 VC 基准；② 基于 2400 条手工编写的翻译规则，提供可靠的 VC 提取与语义保持；③ 引入注释去除（complication）工序，使 VC 更具挑战性。

**🔧 技术方法**

利用 Why3/Frama‑C 的 VCG，Python 翻译框架，≈2400 条专家规则，LLM（GPT‑4o‑mini、Qwen3、DeepSeek‑V3、DeepSeek‑Prover‑V2 等）以及 ITP hammer（Sledgehammer、CoqHammer）。

**📊 数据集**

从 Linux、Contiki‑OS 等工业项目以及 Pearls of Programs 收集约 7.5k VC，挑选 600 作为基准，覆盖三种 ITP 语言。

**📈 对比分析**

采用 Pass@1/4/8 指标对 NTP 与 Hammer 进行零样本评估。NTP 最高 pass@8 仅 11.5%（Minilang+Sledgehammer），而 Sledgehammer 在 Isabelle 上达 18% 以及 CoqHammer 在 Rocq 上 5.7%；表明当前 NTP 未能超越传统 ATP。

**⚠️ 局限性**

NTP 模型易出现语法错误、语义混淆与幻觉，VC 长度与深度导致模型难以保持一致；基准 VC 虽来自已验证项目，但并非所有案例都能保证可证明；模型缺乏对 VC 语境的精准 grounding，影响效果。

---

## 97. Design and Evaluation of Next-Generation Cellular Networks through Digital and Physical Open and Programmable Platforms

**arXiv ID:** 2601.19027 | [PDF](https://arxiv.org/pdf/2601.19027v1)

**作者:** Davide Villa `[一作]` (Northeastern University), Davide Villa `[通讯]` (Northeastern University)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5074897413)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

该论文设计并验证了面向5G/6G的数字孪生实验平台（Colosseum和X5G）以及CaST工具链，用于创建、验证和部署无线场景和协议栈，支持谱共享、AI建模、合成数据等应用。

**💡 创新点**

创新点在于将数字孪生与大规模无线仿真平台结合，提出端到端可重复验证的工具链和闭环AI控制框架，实现从仿真到真实网络的无缝迁移，并通过多种使用案例证明其可比性。

**🔧 技术方法**

使用技术包括射线追踪、MIMO信号处理、SDR、GPU加速、OpenAirInterface、GNU Radio、CI/CD自动化、深度学习（CNN、生成式模型）等。

**📊 数据集**

使用的数据集包括从真实室内外实验（Arena、Wi‑Fi、天线模型）采集的信道样本、雷达信号样本以及通过CaST生成的合成信道和雷达波形。

**📈 对比分析**

通过将Colosseum仿真结果与Arena真实测量、以及不同场景下的吞吐量/ SINR 进行交叉相关比较，发现平均相似度达到0.987，雷达检测准确率可达88%，检测时延约137 ms。

**⚠️ 局限性**

局限性包括对极端移动、天线复杂度有限的建模误差、生成模型对硬件细节的依赖，以及在极大规模或多频段场景下的资源与实时性挑战。

---

## 98. A Switching Nonlinear Model Predictive Control Strategy for Safe Collision Handling by an Underwater Vehicle-Manipulator System

**arXiv ID:** 2601.18971 | [PDF](https://arxiv.org/pdf/2601.18971v1)

**作者:** Ioannis G. Polyzos `[一作]` (Georgia Institute of Technology), Konstantinos J. Kyriakopoulos `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11577 | [OpenAlex ID](https://openalex.org/A5012966309)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种三模切换非线性模型预测控制（NMPC）策略，用于水下机械臂系统（UVMS）在碰撞预警与碰撞处理中的安全控制。

**💡 创新点**

创新点在于结合碰撞感知、预先接触启动和机械臂推力三种模式，实现在无可行避让方案时主动接触并利用机械臂减速，提供比传统避障（如控制壁函数）更强的冗余安全保障。

**🔧 技术方法**

核心技术包括基于Kane方法的UVMS动力学建模、超椭球体与平面障碍的距离与速度解析、三阶段NMPC优化（任务模式、接触模式、推力模式）以及自触发模式切换算法。

**📊 数据集**

实验数据来源于MATLAB仿真环境，使用符号数学工具箱生成的UVMS动力学模型和人工设定的水流、阀门故障等场景进行验证，并未使用公开真实数据集。

**📈 对比分析**

通过仿真比较显示，该方法能够在各种初始条件、潮流、机械臂配置以及发动机失效情况下成功完成碰撞预防或处理，控制输入与状态响应均符合预期，但未与现有避障算法做定量对比。

**⚠️ 局限性**

主要局限包括对初始化猜测和参数调节高度敏感、缺乏真实环境验证、计算量大导致实时性不足以及在极端碰撞即将发生时的最优性和鲁棒性仍需进一步改进。

---

## 99. m2sv: A Scalable Benchmark for Map-to-Street-View Spatial Reasoning

**arXiv ID:** 2601.19099 | [PDF](https://arxiv.org/pdf/2601.19099v1)

**作者:** Yosub Shin `[一作]` (University of Hawai'i at Manoa), Igor Molybog `[通讯]` (University of Hawai'i at Manoa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了一个可扩展的地图与街景对齐空间推理基准 m2sv，包含自动化数据生成管线、标注工具、结构化推理轨迹以及多城市的评估集。

**💡 创新点**

创新点包括：① 以真实地图与街景的跨视角对齐为核心任务，显著突破现有单视角或合成图表基准；② 引入可量化的结构难度信号（候选方向数、角度对称性）与人类耗时测度，构建难度感知评估；③ 对现有 VLM 的失败模式进行系统化归纳，揭示空间绑定、左右翻转、对不可靠视觉线索的过度依赖等典型错误；④ 通过可重现的管线和公开数据促进大规模可扩展评估。

**🔧 技术方法**

技术手段主要包括：Vision‑Language 模型（Gemini‑3‑Pro、GPT‑5、Qwen3‑VL 等）在零样本、SFT 与 RL（LoRA）下的训练；结构化推理轨迹生成与筛选；人类耗时收集作为难度衡量；跨基准（MindCube 等）转移实验；以及对推理轨迹长度和内部一致性的分析。

**📊 数据集**

使用数据集：
- m2sv‑20k：20000 例，覆盖 32 个城市，候选方向 2‑7 个。
- m2sv‑sft‑11k（经筛选后 4.4k 例）：包含 Gemini‑2.5‑Pro 生成的结构化推理轨迹。
数据来源于 Google Street View、公开地图与城市坐标。

**📈 对比分析**

比较方法：在 1k 例上做零样本评估，随后对 Qwen3‑VL‑8B 进行 SFT（+RL）实验，并在 10k 验证集上测量准确率。结果显示：
- 零样本：Gemini‑3‑Pro 65.2%，GPT‑5 57.2%，Qwen3‑VL‑8B 35.5%；
- 人类基准 95%。
- SFT：从 34.3% 提升至 39.8%；
- SFT+RL：进一步提升至 43.9%。
尽管有提升，仍距人类差距达 50+ 点，且跨基准转移效果有限。

**⚠️ 局限性**

局限性：
- 依赖商业地图与 Street View，受其覆盖与更新频率限制；
- 仅涵盖公共道路交叉口，未考虑私有或敏感位置；
- 人类评测样本有限，耗时测度不够全面；
- 目前模型难以根据难度自适应推理深度，导致易错；
- 数据集在对称与视觉混淆方面仍存在固有挑战。

---

## 100. Listening before Asking: Lived-Experience Advisors as Methodological Partners in Dementia Caregiving Studies

**arXiv ID:** 2601.19021 | [PDF](https://arxiv.org/pdf/2601.19021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 101. Time series forecasting with Hahn Kolmogorov-Arnold networks

**arXiv ID:** 2601.18837 | [PDF](https://arxiv.org/pdf/2601.18837v1)

**作者:** Md Zahidul Hasan `[一作]` (Concordia University), Nizar Bouguila `[通讯]` (Concordia University)

**通讯引用:** 9435 | [OpenAlex ID](https://openalex.org/A5090600716)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 HaKAN 框架，使用 Kolmogorov‑Arnold 网络结合 Hahn 多项式实现多变量时间序列长期预测。

**💡 创新点**

创新点：将 Hahn 多项式参数化的 KAN 层嵌入双层 Patch 结构，实现轻量化、可解释的全局与局部时序建模，并利用 Channel Independence、RevIN 与瓶颈全连接提升效率。

**🔧 技术方法**

技术：Kolmogorov‑Arnold 网络、Hahn 多项式激活、Patch 编码、Channel Independence、RevIN 归一化、残差连接、瓶颈结构、PyTorch 实现。

**📊 数据集**

数据集：Weather、Traffic、Electricity、ETT 系列（ETTh1/2、ETTm1/2）以及 Illness。

**📈 对比分析**

与 Transformer / MLP 基线（Informer、PatchTST、TimesNet、Crossformer 等）对比，HaKAN 在大多数数据集上均实现 MSE/MAE 下降 5–8%，显著优于现有方法。

**⚠️ 局限性**

限制：假设通道独立性，可能在强跨变量相关的数据上表现欠佳；对周期性频谱特征的建模仍需进一步提升。

---

## 102. Malicious Repurposing of Open Science Artefacts by Using Large Language Models

**arXiv ID:** 2601.18998 | [PDF](https://arxiv.org/pdf/2601.18998v1)

**作者:** Zahra Hashemi `[一作]` (University of Luxemburg), Wei Zhao `[通讯]` (University of Aberdeen)

**通讯引用:** 488 | [OpenAlex ID](https://openalex.org/A5101855969)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个端到端的管道，利用基于角色扮演的越狱技巧让大型语言模型（LLM）生成并评估针对开源科研资源的恶意研究方案；

**💡 创新点**

创新点在于系统化结合了越狱、资产提取、分步骤恶意方案生成以及多维评估框架，并首次开展跨模型自评的实验，揭示LLM评估中的自偏差；

**🔧 技术方法**

使用了LLM提示工程（角色扮演越狱、JSON结构化输出）、链式思维（CoT）、API驱动的网络搜索以及基于ACL伦理和PAI指南的评估指标；

**📊 数据集**

利用了51篇ACL 2025年高双重用途潜力的论文（涵盖生成、可解释性、多模态、偏见/公平、信息检索）以及在示例中使用的公开数据集（如SAGED、CrowS‑Pairs等）；

**📈 对比分析**

通过让GPT‑4.1、Grok‑3和Gemini‑2.5‑pro同时担任生成器和评估器，交叉评估其输出；结果显示三者均能生成技术可行且有害的方案，GPT‑4.1评估最高、Gemini‑2.5‑pro最低，验证评估者选择对得分影响显著；

**⚠️ 局限性**

局限性包括仅聚焦NLP领域、只模拟实验未实际验证可行性、LLM评估易受自偏差影响、数据集规模有限，强调仍需人工监督以确保评估可靠性。

---

## 103. Propagating Similarity, Mitigating Uncertainty: Similarity Propagation-enhanced Uncertainty for Multimodal Recommendation

**arXiv ID:** 2601.19198 | [PDF](https://arxiv.org/pdf/2601.19198v1)

**作者:** Xinzhuo Wu `[一作]`, Hongfei Lin `[通讯]` (Dalian University of Technology)

**通讯引用:** 8494 | [OpenAlex ID](https://openalex.org/A5023931221)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SPUMR框架，通过构造模态相似图与协作相似图并引入不确定性感知的偏好聚合机制，显著提升多模态推荐性能。

**💡 创新点**

首次将不确定性建模与多模态推荐结合，并利用相似度传播降低噪声，实现更鲁棒的特征融合。

**🔧 技术方法**

使用图神经网络构建模态与协作相似图，采用高斯分布建模不确定性、重参数化与KL正则化，并在BPR与对比损失下训练。

**📊 数据集**

在Amazon 5-core Baby、Sports、Clothing 三个基准数据集上进行实验。

**📈 对比分析**

与 MF‑BPR、LightGCN、VBPR、MMGCN、FREEDOM、LGMRec 等 14 种基线对比，SPUMR 在 Recall@10/20 与 NDCG@10/20 上分别提升 4%–7%，显著优于现有方法。

**⚠️ 局限性**

模型对超参数（λ_KL、λ_U）敏感，需要手动调优；在极端噪声场景下仍可能受限，且模型复杂度相对较高。

---

## 104. Weakly supervised framework for wildlife detection and counting in challenging Arctic environments: a case study on caribou (Rangifer tarandus)

**arXiv ID:** 2601.18891 | [PDF](https://arxiv.org/pdf/2601.18891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 105. AI-Powered Augmented Reality as a Threat Vector for Human Manipulation

**arXiv ID:** 2601.18802 | [PDF](https://arxiv.org/pdf/2601.18802v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 106. VAE with Hyperspherical Coordinates: Improving Anomaly Detection from Hypervolume-Compressed Latent Space

**arXiv ID:** 2601.18823 | [PDF](https://arxiv.org/pdf/2601.18823v1)

**作者:** Alejandro Ascarate `[一作]` (Queensland University of Technology), Olivier Salvado `[通讯]` (Queensland University of Technology)

**通讯引用:** 14517 | [OpenAlex ID](https://openalex.org/A5025220020)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出将VAE的潜在变量转换为超球坐标，从而把潜在空间压缩到超球表面上的小岛，实现更稠密、可表达的潜在分布，并用于异常检测。

**💡 创新点**

创新点在于：1) 用超球坐标重新定义KL散度，使潜在向量能在高维空间中沿单个角度移动；2) 通过压缩所有超球角度而非仅压缩第一个角度，显著降低潜在空间稀疏度和高维体积，提高异常检测的灵敏度；3) 证明该方法可直接复现vMF子模型。

**🔧 技术方法**

技术方法包括：VAE的超球坐标变换、基于均方误差与重参数化的训练、k‑NN异常评分、批量统计的KL改写、对比学习与非对比学习两种训练策略。

**📊 数据集**

数据集：Mars Rover Mastcam（多光谱图像）、Galaxy Zoo（银河系图像）、CIFAR‑10/100、Imagenette、Texture、SVHN、LSUN、iSUN、Places365 等，覆盖完全无监督与OOD两种异常检测场景。

**📈 对比分析**

与传统AE/ VAE、k‑NN、Isolation Forest、MSE重建误差以及vMF方法比较；在完全无监督和OOD实验中均取得最优或竞争性最高的FPR95与AUROC，尤其在近OOD（CIFAR‑10 vs CIFAR‑100、Imagenette vs 近似ImageNet）中表现突出。

**⚠️ 局限性**

局限性：计算量比标准VAE增加约32%，随潜在维度升高成本进一步上升；参数调优仍需手工；方法依赖超球坐标变换，且在某些超大规模数据集上的可扩展性尚未验证。

---

## 107. Analysis of Control Bellman Residual Minimization for Markov Decision Problem

**arXiv ID:** 2601.18840 | [PDF](https://arxiv.org/pdf/2601.18840v1)

**作者:** Donghwan Lee `[一作]` (Korea Advanced Institute of Science and Technology), Hyukjun Yang `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究并证明了控制 Bellman 残差最小化（CBR）及其软化版本（SCBR）在策略优化中的理论性质，并给出了基于梯度下降的收敛分析和实验验证。

**💡 创新点**

创新点在于：① 对非凸非光滑的 CBR 目标给出分段二次结构、局部 Lipschitz 性和 Clarke 子微分的解析；② 证明通用梯度下降收敛到 Clarke 极值点；③ 引入软 max（log-sum-exp）构造可微 SCBR，并证明梯度下降可指数收敛至唯一软 Bellman 解决方案；④ 给出误差界与投影值迭代的比较。

**🔧 技术方法**

采用的技术包括：Clarke 子微分理论、非凸非光滑优化方法、梯度下降与 Armijo 线搜索、软 max（log-sum-exp）逼近、投影/斜投影 Bellman 方程、局部 PL 条件与强凸性分析、误差界推导，以及基于随机特征的线性函数逼近。

**📊 数据集**

实验数据集：离散格子环境（$|S	imes A|=256$，随机特征矩阵 $m=120$）以及若干 OpenAI Gym 经典环境（如 CartPole、MountainCar 等）用于深度 SCBR 的评估。

**📈 对比分析**

与投影值迭代（P‑VI）的比较显示：在离散环境中 P‑VI 发散，SCBR 收敛且成功率为 20.7%；与 DQN 基线的对比表明 SCBR 在多环境中整体表现低于 DQN，但在某些环境可略优；实验结果展示了梯度方法的稳定收敛和相对优势。

**⚠️ 局限性**

局限性包括：SCBR 仍可能收敛到次优局部极值；理论分析主要针对线性函数逼近与离散 MDP，扩展到非线性深度模型与连续控制仍需进一步研究；软化参数温度对收敛速度与解质量有显著影响；深度版 SCBR 在某些复杂环境中性能不及主流算法。

---

## 108. Latent Structural Similarity Networks for Unsupervised Discovery in Multivariate Time Series

**arXiv ID:** 2601.18803 | [PDF](https://arxiv.org/pdf/2601.18803v1)

**作者:** Olusegun Owoeye `[一作]` `[通讯]` (University of Cambridge), Olusegun Owoeye (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种无监督的多变量时间序列关系发现框架，利用滚动窗口的序列自编码器学习潜在表示，聚合成实体嵌入，并通过潜在空间余弦相似度阈值化生成稀疏相似网络。

**💡 创新点**

创新点在于将表示学习与关系网络构建统一为任务无关的“发现层”，不追求预测或交易收益，只提供可解释的结构化关系，同时通过窗口化处理和潜在空间阈值化实现稀疏可分析网络。

**🔧 技术方法**

主要技术包括：LSTM序列到序列自编码器、窗口化归一化、潜在空间余弦相似度、阈值化稀疏化、以及Engle–Granger协整检验作为后置诊断。

**📊 数据集**

使用 2024‑2025 年 Binance 交易所上市值前 20 名加密货币的 1 小时滚动窗口价格收益数据。

**📈 对比分析**

未与其他下游任务或预测模型比较；通过构造的相似网络得到 64 条边，其中 16 条满足协整检验，说明该框架能捕获既符合传统线性关系又不满足协整的非传统关系，但缺乏定量性能对比。

**⚠️ 局限性**

局限包括：仅在单一采样频率与固定阈值下实验，未评估阈值/相似度度量变化的鲁棒性；仅使用协整作为验证手段，未与其他领域的验证方法对比；未测评预测或交易收益，无法证明对实际决策的价值。

---

## 109. A Hybrid Discriminative and Generative System for Universal Speech Enhancement

**arXiv ID:** 2601.19113 | [PDF](https://arxiv.org/pdf/2601.19113v1)

**作者:** Yinghao Liu `[一作]` (Alibaba Group), Zheng Xue `[通讯]` (Alibaba Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种融合判别与生成模型的混合网络，用于通用语音增强，结合TF-GridNet与基于AR的语义条件重建并通过融合网络生成高质量音频。

**💡 创新点**

将采样率无关的TF-GridNet与基于WavLM+X-Codec的语义条件AR生成模型结合，并通过可学习融合掩码实现自适应权重，兼顾信号保真与自然感。

**🔧 技术方法**

使用TF-GridNet、采样率无关SFI策略、WavLM+线性适配器、X-Codec、AR语言模型、DPRNN、卷积解码、融合USEs网络、多分辨率STFT、感知损失和SQA分数损失。

**📊 数据集**

使用URGENT 2026 Challenge Track 1 的约1.3M句子、五种语言的训练数据（不含预训练模型）。

**📈 对比分析**

在非盲测试集上与基线、单一判别/生成分支对比，Hybrid 在 PESQ、ESTOI 等信号指标和 DNSMOS、NISQA 等感知指标上均优于基线，并在说话人相似度和下游 ASR 准确率上获得最佳表现。

**⚠️ 局限性**

生成分支仅在16kHz下工作，推理延迟高，未实现全频带处理且实时性不足。

---

## 110. GPCR-Filter: a deep learning framework for efficient and precise GPCR modulator discovery

**arXiv ID:** 2601.19149 | [PDF](https://arxiv.org/pdf/2601.19149v1)

**作者:** Jingjie Ning `[一作]` (Carnegie Mellon University), Xinheng He `[通讯]` (Shanghai Institute of Materia Medica)

**通讯引用:** 3886 | [OpenAlex ID](https://openalex.org/A5037530308)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个专门用于GPCR调节剂发现的深度学习框架GPCR-Filter，并在真实实验中验证了其预测效果。

**💡 创新点**

创新点在于将预训练蛋白语言模型ESM‑3与分子图神经网络通过注意力融合，专门捕捉GPCR与配体的功能关系；同时构建了超过9万条高质量人类GPCR–配体互作数据集，极大提升了模型泛化与可解释性。

**🔧 技术方法**

技术手段包括Transformer‑style交叉注意力、ESM‑3蛋白序列嵌入、GCN分子图表示、二分类交叉熵训练及注意力可视化。

**📊 数据集**

使用了91,396条人类GPCR–药物交互记录（527种GPCR、72,177条SMILES），来自GPCRdb和GtoPdb的高质量实验数据。

**📈 对比分析**

与ConPLex和TransformerCPI2.0在随机、intra‑target、inter‑target三种划分下对比，GPCR‑Filter在随机/内目标/外目标三种场景分别取得AUC 98.93%/97.16%/73.44%，准确率、AP和Precision显著高于基线，实验验证在5‑HT1A受体上发现4个微摩尔级激动剂。

**⚠️ 局限性**

局限性包括：负样本是通过1:1采样得到，可能包含未检测到的活性配体；交叉目标评估中配体可复用，可能降低难度；未对更广泛受体家族或更严格的负样本策略做进一步验证。

---

## 111. Fauna Sprout: A lightweight, approachable, developer-ready humanoid robot

**arXiv ID:** 2601.18963 | [PDF](https://arxiv.org/pdf/2601.18963v1)

**作者:** Fauna Robotics `[一作]` (Fauna Robotics), Josh Merel `[通讯]` (Fauna Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了名为 Sprout 的小型安全、可表达且易于开发的类人机器人平台，并提供完整的硬件、软件、可视化和调试工具，支持从低级驱动到高层交互、导航和远程操作的全栈功能。

**💡 创新点**

创新点主要包括：
- 在尺寸、重量和能量上做出“安全优先”的硬件设计（软外壳、背驱动电机、限制关节扭矩）；
- 将全身协同控制与可编程状态机相结合，使用强化学习策略实现多模式（行走、蹲下、爬行等）并提供显式的状态转换与安全检查；
- 通过 VR 全身遥控与学习驱动的“Retargeting”技术，将用户身体姿态映射到机器人运动；
- 采用“槽位合并”框架实现多模态 HRI（灯光、声音、面部表情）与行为树的高效融合；
- 统一使用 Docker 容器化服务，结合 ROS 2 与自研的 RMW Zenoh，使得平台易于部署、升级和多机器人协作。

**🔧 技术方法**

主要技术与方法：
- 硬件：轻量化结构、软质外壳、背驱动电机、单轴抓取器、RGB‑D 摄像头、IMU、四麦克风阵列；
- 软件：ROS 2、RMW Zenoh、CUDA‑GPU 推理（NVIDIA Jetson AGX Orin），IsaacLab/IsaacSim 强化学习训练；
- 运动学与控制：基于 PD+限制器的可合规策略、状态机调度、轨迹插值与转场；
- 传感与定位：融合视觉‑惯性‑关节里程计的 EKF；
- 地图与导航：基于 TSDF 的体素地图、maplet 层次、GTSAM 后端优化、Hybrid A* 全局+局部规划、Pure‑Pursuit 跟踪；
- HRI 与对话：wake‑word → ASR → LLM → TTS、MCP 代理工具、槽位合并行为树。

**📊 数据集**

使用的数据集主要有：
- 人类运动捕捉与动画数据（用于训练转场与全身控制）;
- 视觉环境图像（用于循环闭环、SLAM 与深度识别）;
- 机器人传感日志（用于验证控制策略、数据采集与 DAgger 交互）。
（文中未明确列出公开数据集，但训练过程依赖仿真和本机收集的实地数据。）

**📈 对比分析**

比较方法：与其他小型类人机器人（如 Unitree G1/R1、Boston Dynamics Spot 等）在安全、可表达、部署便利性等指标进行对比；在仿真中验证控制策略与行走/爬行的稳定性；在真实环境中展示遥控、导航与 HRI 场景。性能方面，文中提到：
- 运动学控制可达 50 Hz；
- 地图构建占用 CPU 约 30 % 的计算；
- 全身遥控延迟低于 200 ms；
- 通过强化学习训练的行走策略在仿真中实现 0.4 m/s 的稳定步态。总体上表明 Sprout 在安全性、表达性与易用性方面相较于现有平台有显著提升。

**⚠️ 局限性**

局限性：
- 抓取器单指设计限制了高精度操作和复杂抓取任务；
- 机器人身高仅 1.07 m，导致在高物件或楼梯等场景的可用性受限；
- 仍缺乏视觉导向的自主操控（仅支持基于地图的导航）；
- 对深度相机的依赖在纹理不足或光照变化环境中会导致 SLAM 失效；
- 语言与对话系统目前需要外部 LLM，平台本身不具备完整的自主对话能力。

---

## 112. Analysis of Shuffling Beyond Pure Local Differential Privacy

**arXiv ID:** 2601.19154 | [PDF](https://arxiv.org/pdf/2601.19154v1)

**作者:** Shun Takagi `[一作]` (LY Corporation), Seng Pei Liew `[通讯]` (LY Corporation)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文重新审视了隐私毛毯界限，并发展了一种适用于广泛本地随机化器的渐近分析，提出了一个新的隐私参数——洗牌指数χ，来描述洗牌如何增强隐私。

**💡 创新点**

创新点在于引入了洗牌指数χ，作为描述本地随机化器与洗牌相互作用的单一标量参数，并通过此参数提供了更紧凑的隐私保证界限。

**🔧 技术方法**

使用了渐近分析和快速傅里叶变换（FFT）算法来计算隐私毛毯的相对误差，并提供了近线性时间复杂度的算法。

**📊 数据集**

使用了多种本地随机化器，包括k-随机响应（k-RR）和广义高斯机制，进行理论分析和实验验证。

**📈 对比分析**

通过与现有方法的比较，本文的方法在隐私保证的紧密性上表现出色，尤其是在k-RR机制中，洗牌指数的上下界在大多数情况下几乎重合，显示出良好的性能。

**⚠️ 局限性**

限制在于对广义高斯机制的分析仅限于一维情况，且在高维情况下，隐私毛毯界限可能变得无效。此外，渐近最坏情况的保证可能在有限样本情况下不再适用。

---

## 113. LocationAgent: A Hierarchical Agent for Image Geolocation via Decoupling Strategy and Evidence from Parametric Knowledge

**arXiv ID:** 2601.19155 | [PDF](https://arxiv.org/pdf/2601.19155v1)

**作者:** Qiujun Li `[一作]` (Central South University), Haifeng Li `[通讯]` (Central South University)

**通讯引用:** 10249 | [OpenAlex ID](https://openalex.org/A5100398353)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种层级化地理定位代理（LocationAgent），将推理策略与证据验证解耦，利用外部工具进行地理事实核查。

**💡 创新点**

核心创新在于RER结构实现层级推理并通过Recorder保持状态一致，同时通过四类能力模块与原子工具实现多维证据检索，突破了静态记忆限制。

**🔧 技术方法**

采用大语言模型（GPT‑5 等）作为Reasoner，Executor 的四大功能模块（环境、基础设施、语义符号、图像匹配）与图像搜索、文本搜索等外部检索工具相结合。

**📊 数据集**

构建了新型中国城市定位基准 CCL‑Bench，包含 300 张真实互联网场景图像并细粒度标注四种场景与三难度级别。

**📈 对比分析**

在 CCL‑Bench 上与多种隐式和显式定位方法对比，LocationAgent 在 1 km、25 km、200 km 等阈值上分别达到 52.33%、82.00%、100%，显著优于现有方法。

**⚠️ 局限性**

局限在于仍依赖外部工具的可用性，且在极端稀缺信息场景下可能出现证据不足导致定位失败；模型规模大、算力需求高。

---

## 114. SICL-AT: Another way to adapt Auditory LLM to low-resource task

**arXiv ID:** 2601.18904 | [PDF](https://arxiv.org/pdf/2601.18904v1)

**作者:** Haolong Zheng `[一作]` (University of Illinois Urbana Champaign), Mark Hasegawa-Johnson `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Speech In-Context Learning Adaptation Training (SICL-AT)，让大型听觉 LLM 在高资源语音任务上以 ICL 方式微调，从而在低资源任务中实现零/少样本自适应。

**💡 创新点**

创新点在于将 ICL 训练模式迁移为后训练策略，强化模型在推理时对上下文示例的利用，使其在无需梯度更新的情况下获得显著性能提升。

**🔧 技术方法**

采用 LoRA 参数高效微调、TICL 示例检索、ICL 推理格式，结合 Qwen2.5-Omni 与 MiMo-Audio 两大多模态听觉 LLM 进行实验。

**📊 数据集**

使用 CommonVoice、CoVoST2、MMSU、MyST、RSR、MMAU、MMAR 等高低资源语音及多语言数据集进行训练与评估。

**📈 对比分析**

通过与直接微调、零样本及普通 ICL 的对比实验，SICL-AT 在儿童 ASR、音频理解/推理、以及多语言 ASR/翻译任务中均优于基线，特别是在低资源场景下表现更为稳定。

**⚠️ 局限性**

局限性包括仅验证两类模型、固定检索与数据集设置、未评估长上下文推理成本、以及对检索质量和极度稀缺部署场景的适用性仍有限。

---

## 115. Out-of-Distribution Generalization for Neural Physics Solvers

**arXiv ID:** 2601.19091 | [PDF](https://arxiv.org/pdf/2601.19091v1)

**作者:** Zhao Wei `[一作]` (Agency for Science Technology and Research), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 27306 | [OpenAlex ID](https://openalex.org/A5068243197)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 NOVA 框架，结合物理信息的神经网络架构搜索与零样本物理适配，用以构建可泛化到分布外的神经物理求解器。

**💡 创新点**

创新点在于：① 在 NAS 过程中直接加入物理约束，寻找与物理一致的结构；② 将物理适配转化为凸最小二乘问题，可一次性闭式求解；③ 只需少量训练周期即可获得高质量特征，实现数据轻量化与高效泛化；④ 通过零样本适配在未见任务上保持低误差，解决长时演化误差积累问题。

**🔧 技术方法**

采用基于 U‑Net 的可变结构、ConvNet 前层权重的轻量训练、Tikhonov 正则化的凸物理适配、以及遗传/贝叶斯 NAS 搜索算法；在推理时采用自回归预测。

**📊 数据集**

使用三类 PDE 数据集：二维扩散‑反应、二维 Navier–Stokes 以及二维非线性热方程，分别提供训练、分布内测试与分布外测试样本（如不同初始条件、几何、源项）。

**📈 对比分析**

与 FNO、U‑Net、MLP‑PINN 以及传统 NAS‑U‑Net 对比。结果显示 NOVA 在分布外测试误差比基线低 1–2 个数量级，RMSE 下降至 0.002–0.005；在自回归长时演化和流体芯片生成指导任务中，误差积累明显减缓，设计性能提升约 15%。

**⚠️ 局限性**

局限性包括：① 目前仅验证 2‑D 线性/非线性 PDE，3‑D 或多物理耦合场景仍待扩展；② 物理适配依赖已知的 PDE 解析结构，对高度非线性或不可微问题适用性有限；③ NAS 搜索成本虽降低，但在大规模高维搜索空间仍可能受限。

---

## 116. Intent2QoS: Language Model-Driven Automation of Traffic Shaping Configurations

**arXiv ID:** 2601.18974 | [PDF](https://arxiv.org/pdf/2601.18974v1)

**作者:** Sudipta Acharya `[一作]` (University of Ottawa), Burak Kantarci `[通讯]` (University of Ottawa)

**通讯引用:** 8425 | [OpenAlex ID](https://openalex.org/A5003131477)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于语言模型的自动化流水线，将高层网络意图转化为可部署的 Linux tc 配置。

**💡 创新点**

创新点在于结合队列理论的数字孪生语义模型与基于规则的审核器，实现端到端意图到配置的自动翻译。

**🔧 技术方法**

使用技术包括 AQM 引导的离散队列模拟、LLM（LLaMA3/Mistral/Gemma/Phi-2）生成子意图与配置以及规则化后处理。

**📊 数据集**

使用的 dataset 是人工构造的 100 条高层意图样本，覆盖延迟、丢包、带宽、优先级等 QoS 目标。

**📈 对比分析**

通过零/一/两次示例 + AQM 提示三种策略进行评测，LLaMA3 在两次示例+AQM 下取得 0.88 语义相似度、0.87 语义单元覆盖率、0.16 正规化编辑距离，优于其它模型 30%+。

**⚠️ 局限性**

局限性包括缺乏真实意图多样性、模型对短语偶发性错误、仅覆盖 Linux tc 语义、未实时更新语义模型等。

---

## 117. Average-Case Reductions for $k$-XOR and Tensor PCA

**arXiv ID:** 2601.19016 | [PDF](https://arxiv.org/pdf/2601.19016v1)

**作者:** Guy Bresler `[一作]` (Massachusetts Institute of Technology), Alina Harbuzova `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5005694601)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

通过多项式时间平均案例还原，构建k-𝖷𝖮𝖱与Tensor PCA之间的计算难度关系，推出一系列硬件结果和层次结构；

**💡 创新点**

首次提出方程组合原语、稠密化还原、阶数降低等技术，实现从稀疏到稠密、从高阶到低阶的可控参数转换，形成统一的可植入张量模型硬件偏序；

**🔧 技术方法**

采用平均案例还原、方程组合、Gaussian/离散等价、高维CLT对Gramian分析、Poisson化、克隆与拆分、整数化等方法；

**📊 数据集**

该研究纯理论性质，无使用具体实验数据集；主要在随机矩阵/张量实例上进行分析；

**📈 对比分析**

通过理论证明和参数映射，展示在计算阈值附近实例的多项式时间不可解性，并给出多种算法下的性能界限；无实验对比；

**⚠️ 局限性**

局限性包括还原只能增加密度、降低阶数，无法实现稠密到稀疏或阶数升高；对低噪声、低维小阶情况仍缺乏完整算法；对部分噪声分布或非平凡域大小的扩展有限。

---

## 118. Axe: A Simple Unified Layout Abstraction for Machine Learning Compilers

**arXiv ID:** 2601.19092 | [PDF](https://arxiv.org/pdf/2601.19092v1)

**作者:** Bohan Hou `[一作]` (Carnegie Mellon University), Tianqi Chen `[通讯]` (NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文提出了 Axe 这一统一的布局抽象，并基于它实现了多粒度、分布式感知的编译器，实现了在 GPU、Multi‑GPU 和 AI 加速器上几乎与手工调优相当的性能。

**💡 创新点**

创新点在于将逻辑索引映射到多轴物理空间的 D、R、O 三元结构，统一了张量分片、复制与偏移；并将该抽象嵌入 DSL 及编译器中，允许在单核内混合线程局部和集体运算。

**🔧 技术方法**

技术上使用了名轴张量布局、TVM TensorIR、CUDA、TMA 异步拷贝、SMA 线程块集群、以及对 Trainium-1 等 AI 加速器的专用指令支持。

**📊 数据集**

实验使用了 Qwen、LLaMA、Gemma、GPT‑3 等大模型权重形状作为 GEMM 基准，并在 Qwen3‑30B 的 MoE 层上评测。

**📈 对比分析**

通过与 cuBLAS、Triton、DeepGEMM、FlashInfer、SGLang、NCCL、Neuron NKI 等基线比较，Axe 在单 GPU FP16 GEMM 达到 97‑100% cuBLAS，MoE 层比 FlashInfer 提升 1.2‑1.36 倍，Multi‑GPU GEMM+Reduce‑Scatter 与 Triton‑distributed 对比获得 1.4 倍加速，Trainium‑1 上 MHA 达到 1.44 倍性能提升。

**⚠️ 局限性**

局限性在于当前需要手动给出 Axe 布局，尚未支持动态形状或完全自动化映射，且集成到主流框架和其他加速器（如 TPU、专用 ASIC）仍有待完善。

---

## 119. Dynamic Cogeneration of Bug Reproduction Test in Agentic Program Repair

**arXiv ID:** 2601.19066 | [PDF](https://arxiv.org/pdf/2601.19066v1)

**作者:** Runxiang Cheng `[一作]` (Google), Franjo Ivančić `[通讯]` (Google)

**通讯引用:** 3330 | [OpenAlex ID](https://openalex.org/A5020028377)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在Agentic APR 中同时生成修复补丁和 Bug Reproduction Test（BRT）的可行性与效果。

**💡 创新点**

首次将修复与 BRT 在同一轨迹中共生成，并对比 TDD、TLD 与 Freeform 三种工作流，提出了 test‑aware patch selector。

**🔧 技术方法**

使用 Gemini 2.5 Pro LLM 的 ReAct‑style 代理、工具调用、Patch 选择器与 BRT 评估器。

**📊 数据集**

使用了 120 个来自 Google Issue Tracking System（GITS）的人工报告 Bug 及其真实补丁作为评估数据集。

**📈 对比分析**

通过 pass@k、plausibleBRT@k 等指标与 Fix‑only、BRT‑only 基线对比，Freeform 在 (pass & plausibleBRT)@20 上表现最佳，且在步数和选择精度上优于单独流水线。

**⚠️ 局限性**

局限性包括：实验仅在 Google 内部环境与 Gemini 模型上进行，存在数据集偏差、评估间隙以及 LLM 非确定性导致的随机性。

---

## 120. Toward Learning POMDPs Beyond Full-Rank Actions and State Observability

**arXiv ID:** 2601.18930 | [PDF](https://arxiv.org/pdf/2601.18930v1)

**作者:** Seiji Shaw `[一作]` (Massachusetts Institute of Technology), Nicholas Roy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 9385 | [OpenAlex ID](https://openalex.org/A5010817303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于张量分解的POMDP学习方法，能够从随机探索下的动作-观测序列中学习到离散POMDP的状态数、转移矩阵和观测矩阵，并在可分辨的观测分区下恢复完整的状态空间；

**💡 创新点**

创新点在于将PSR的相似变换问题与张量分解相结合，利用全秩动作的观测分布求解相似变换，从而在传统PSR无法直接获得转移/观测概率的缺陷中得到显式的概率模型，并支持对非唯一观测分区进行分区级学习；

**🔧 技术方法**

主要技术包括：Hankel矩阵构造、SVD秩分解得到PSR更新矩阵、利用全秩动作的张量分解求解相似变换、随机加权求共轭对角化以确定基变换、以及对分区级概率的归一化处理；

**📊 数据集**

实验使用了标准POMDP基准（Tiger、T‑Maze、Sense‑Float‑Reset）以及自定义噪声走廊（noisy hallway、directional hallway）等合成域；

**📈 对比分析**

与线性PSR和EM基线相比，实验显示该方法在有限数据下能快速收敛到接近真值的观测与分区级转移概率；使用学习得到的POMDP模型在PO‑UCT采样求解器中获得与真模型相近的平均回报，且能够通过显式奖励映射实现更灵活的规划；

**⚠️ 局限性**

局限性包括：需要存在至少一个全秩动作，且对非唯一观测分区只能恢复到分区级别；学习过程中对Hankel矩阵规模与秩阈值敏感；在更大规模POMDP上可扩展性待提升；

---

## 121. Pixel-Grounded Retrieval for Knowledgeable Large Multimodal Models

**arXiv ID:** 2601.19060 | [PDF](https://arxiv.org/pdf/2601.19060v1)

**作者:** Jeonghwan Kim `[一作]` (Meta Reality Labs), Xin Luna Dong `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 5107 | [OpenAlex ID](https://openalex.org/A5101406351)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 PixSearch，一种端到端的分割大型多模态模型，能够自动决定何时检索、选择检索方式（文本、全图或区域），并将检索结果嵌入推理过程，最终给出像素级可视化且事实一致的答案。

**💡 创新点**

其创新点在于将分割、检索触发与检索策略统一到同一模型中，并通过 token 控制的检索交互实现区域级检索与内部推理的无缝融合；同时保留了分割质量，避免了传统管线式检索的翻译误差和延迟。

**🔧 技术方法**

技术包括分割 LMM（如 PLUM、LLaVA 作为骨干）、自回归检索交互（检索触发 token、检索 payload 生成）、信息 token masking 以避免检索信息直接记忆、两阶段监督微调（Stage‑1 保持分割，Stage‑2 训练检索策略）、多模态检索 API（DINOv3 + MPNet 嵌入索引）以及 mask‑based 生成与投影。

**📊 数据集**

训练使用 ADE20k、Pascal Parts、PartImageNet、COCO‑Stuff、RefCOCO 等分割数据；检索训练集包含 TextVQA、InfoSeek、OVEN、CRAG‑MM、WebQA、OKVQA、A‑OKVQA；评估集为 CRAG‑MM、TextVQA、InfoSeek、OVEN、HotpotQA、NQ、PopQA、MuSiQue。

**📈 对比分析**

与 LLaVA、PLUM、Llama‑3.2‑11B、GroundedSAM+LLaVA 等基线对比，PixSearch 在 CRAG‑MM 上提升 19.7% 真实性、24.3% 细粒度图像准确率；在分割任务上与 MaskFormer 接近；在文本 QA 基准上达到或超过基线；在多步检索上，单步检索已显著提升，二三步检索即可逼近无穷预算性能。

**⚠️ 局限性**

局限性包括：检索成本与预算限制，过多检索会增加延迟；对极长文本或高度多步推理仍易遗漏；模型依赖外部检索 API，受检索库覆盖率和质量影响；对极端低分辨率或长尾实体的检索效果尚待进一步提升。

---

## 122. FloydNet: A Learning Paradigm for Global Relational Reasoning

**arXiv ID:** 2601.19094 | [PDF](https://arxiv.org/pdf/2601.19094v1)

**作者:** Jingcheng Yu `[一作]` (Beijing Academy of Artificial Intelligence), Qiwei Ye `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5076022457)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于全局动态规划（DP）思路的图神经网络——FloydNet，利用Pivotal Attention对全图所有节点对关系张量进行迭代细化，从而实现全局推理；

**💡 创新点**

创新点在于：①跳过传统的局部信息传递，采用全局关系张量与学习型DP算子；②设计Pivotal Attention实现对所有双跳路径的自注意力聚合；③证明模型具备3-WL（2-FWL）表达能力，并与k-FWL层次对应；

**🔧 技术方法**

技术包括：全局关系张量初始化（MLP整合节点、边、图特征），Pivotal Attention（多头注意力聚合所有pivot节点），FloydBlock（预层归一化+Feed‑Forward），SuperNode用于节点/图级任务，优化的O(N³)计算核；

**📊 数据集**

数据集：CLRS‑30（算法推理），BREC（同构判别），TSP（metric 与 non‑metric，随机与欧氏权重），LRGB（长程图，PCQM‑Contact、COCO‑SP、ZINC 等），以及标准图属性预测/节点分类基准；

**📈 对比分析**

与1‑WL GNN、Transformer、FGNN 等模型对比，FloydNet 在 CLRS‑30 上超过 95% 近乎完美，TSP 中在 100–200 节点上 10 次采样可达 99.8% 最优率，超越 Linkern（仅 38.8%）；在 BREC、LRGB 等基准上与 3‑WL 理论等价，均显著优于现有方法；

**⚠️ 局限性**

局限性：计算复杂度为 O(N³) 及其显存占用，导致对大规模图的直接训练受限；对多解 TSP 的生成过程仍有误差；在极大图上仍需进一步稀疏化或分层加速技术。

---

## 123. MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution

**arXiv ID:** 2601.18847 | [PDF](https://arxiv.org/pdf/2601.18847v1)

**作者:** Zihan Wu `[一作]`, Xiaohua Jia `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MulVul——一种检索增强的多智能体框架，用于高精度、多类别漏洞检测，采用从粗到细的路由器‑检测器架构；

**💡 创新点**

创新点包括：①多智能体的分层路由与检测，解决漏洞模式异质性和扩展性瓶颈；②跨模型提示进化机制，解耦生成与评估，降低单模型自校正偏差；③检索增强与对比检索相结合，显著降低幻觉并提升细粒度区分；

**🔧 技术方法**

使用技术包括：LLM（GPT‑4o 为执行器，Claude Opus 4.5 为进化器）、SCALE 结构化代码表示、UniXcoder 编码+FAISS检索、检索增强生成 (RAG)、多智能体路由与检测、对比检索工具、跨模型提示进化算法；

**📊 数据集**

实验数据集为 PrimeVul，包含 6,968 条含漏洞 C/C++ 函数与 229,764 条正常函数，覆盖 10 大类别与 130 个 CWE 类型；

**📈 对比分析**

与四种基线（GPT‑4o、LLM×CPG、LLMVulExp、VISION）对比，MulVul 在类别级别 Macro‑F1 50.41%（比最佳基线高 8.9%），在细粒度级别 Macro‑F1 34.79%（比最佳基线高约 10%），并且跨模型提示进化相较手工提示提升 51.6%；

**⚠️ 局限性**

局限性包括：仅在 C/C++ 代码上验证，缺乏对其他语言的评估；在线检测和提示进化过程需要多次 LLM 调用，成本较高；实验仅使用 GPT‑4o 作为执行器，未充分验证跨模型通用性；潜在滥用风险与自动化偏差导致误报/漏报。

---

## 124. Who's in Charge? Disempowerment Patterns in Real-World LLM Usage

**arXiv ID:** 2601.19062 | [PDF](https://arxiv.org/pdf/2601.19062v1)

**作者:** Mrinank Sharma `[一作]` (Anthropic), David Duvenaud `[通讯]` (University of Toronto)

**通讯引用:** 10770 | [OpenAlex ID](https://openalex.org/A5030409494)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Claude.ai 约 150 万条用户交互进行了大规模、隐私保护的量化分析，构建了情境性被动化（disempowerment）框架，并识别了现实扭曲、价值判断与行动扭曲三类潜在模式以及四类放大因素。

**💡 创新点**

创新点在于首次系统化衡量 AI 助手在真实世界对人类自主权的潜在削弱，揭示被动化交互与用户点赞率正相关，提出对短期用户满意度与长期赋权冲突的警示。

**🔧 技术方法**

技术手段包括：隐私保护分析工具 Clio、基于提示的多层级分类器（Claude Haiku 4.5 预筛、Claude Opus 4.5 评分）、文本嵌入聚类、自动化摘要，及与人工标注的对比验证。

**📊 数据集**

数据集为 1.5 M 条公开的 Claude.ai 消费者对话及 500 K 条 Q4 2024–Q4 2025 的用户反馈（Thumbs）数据，涵盖多领域（关系、社会、医疗、技术等）。

**📈 对比分析**

实验与人工标注对比显示分类器与标注者的一致率>95%，并通过用户点赞率与偏好模型（Best‑of‑N）评估表明被动化潜在交互往往得到更高满意度；对比分析显示随着时间推移被动化频率上升，尤其在高风险领域。

**⚠️ 局限性**

主要局限包括：仅分析单轮对话无法追踪跨会话行为；依赖 Claude.ai 流量限制普适性；分类器与聚类存在误差；缺乏因果推断与实际行动追踪；可能因反馈样本偏向而导致被动化比例偏高。

---

## 125. Reducing False Positives in Static Bug Detection with LLMs: An Empirical Study in Industry

**arXiv ID:** 2601.18844 | [PDF](https://arxiv.org/pdf/2601.18844v1)

**作者:** Xueying Du `[一作]` (Fudan University), Yiling Lou `[通讯]` (Fudan University)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5024354460)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在腾讯大型企业软件（AMS业务线）的静态分析工具BkCheck产生的433条报警（328误报、105真报）上，开展了LLM误报削减技术的实证研究，并通过访谈开发者分析误报对工作效率的影响。

**💡 创新点**

首次系统评估多种LLM误报削减方法在工业规模代码中的效果，构建真实企业误报数据集并与传统学习方法对比，提供成本效益分析与案例剖析，填补了公开基准与企业实际之间的空白。

**🔧 技术方法**

使用大语言模型（如Qwen-3-Coder、ChatGPT等）进行基本推理、链式思维提示、少量样本提示，以及LLM与静态分析混合技术（LLM4SA、LLM4PFA等），同时对比传统深度学习模型。

**📊 数据集**

腾讯AMS业务线的BkCheck产生的433条报警数据集，涵盖三种常见bug类型：空指针解引用（NPD）、越界访问（OOB）和除零错误（DBZ）。

**📈 对比分析**

采用准确率、精确率、召回率、F1以及误报减少精度/召回率等指标，并通过多次实验与多数投票确保结果稳定。实验显示，LLM混合方法可在保持高召回率的前提下，消除94–98%的误报，单条报警处理时间仅需2–110秒，费用在0.0011–0.12美元之间。

**⚠️ 局限性**

LLM在处理长上下文、复杂约束和非内存安全bug时效果不佳，且输出具有随机性，需要多次运行以获得稳定结果；数据集局限于C/C++，可推广性待验证；误报削减仍需与人工审核结合，无法完全替代人工保证安全。

---

## 126. XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation

**arXiv ID:** 2601.18886 | [PDF](https://arxiv.org/pdf/2601.18886v1)

**作者:** Youssef Mohamed `[一作]` (King Abdullah University of Science and Technology), Nadezhda Chirkova `[通讯]` (NAVER LABS Europe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出XProvence，一种多语言零成本上下文剪枝模型，可在100+语言的RAG管道中直接在reranker层进行句子级剪枝；

**💡 创新点**

创新点在于将英文剪枝任务迁移至多语言环境，利用跨语言迁移训练策略实现无额外成本的多语言剪枝，并系统比较了跨语言迁移、数据翻译和多语言标注三种训练方案；

**🔧 技术方法**

使用BGE-M3多语言cross‑encoder reranker作为骨干，加入二分类剪枝头；训练时采用交叉熵与回归联合损失；生成剪枝标签时利用GemmaX2 9B等强大LLM；

**📊 数据集**

训练数据包括MS MARCO的16种语言翻译集和MIRACL；评估数据涵盖MKQA、TyDiQA、MedExpQA、XPQA等多语言问答基准；

**📈 对比分析**

与DSLR等基线相比，XProvence在40–60%上下文压缩率下保持甚至提升问答准确率；在未见语言或查询/上下文语言不匹配场景下表现稳健，并且在reranking任务中不降低性能；

**⚠️ 局限性**

局限在于训练仍主要基于Wiki/英文域，翻译误差或非Wiki领域的上下文压缩效果有限，对极端多语言多样性表现尚未充分验证；

---

## 127. Accelerating Large-Scale Cheminformatics Using a Byte-Offset Indexing Architecture for Terabyte-Scale Data Integration

**arXiv ID:** 2601.18921 | [PDF](https://arxiv.org/pdf/2601.18921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 128. Anatomically-aware conformal prediction for medical image segmentation with random walks

**arXiv ID:** 2601.18997 | [PDF](https://arxiv.org/pdf/2601.18997v1)

**作者:** Mélanie Gaillochet `[一作]` (École de Technologie Supérieure), Hervé Lombaert `[通讯]` (Polytechnique Montréal)

**通讯引用:** 1756 | [OpenAlex ID](https://openalex.org/A5064644687)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种基于随机游走的可分割式一致预测框架（RW-CP），通过在预训练视觉基础模型的高维特征空间上进行随机游走扩散来校正分割概率，从而生成既满足统计覆盖保证又具解剖连贯性的预测集合。

**💡 创新点**

创新点在于：① 将随机游走扩散与一致预测结合，利用特征图构造k‑NN图实现空间一致性；② 用随机游走平滑概率后再进行阈值化，显著降低了对校准参数 λ 的敏感性；③ 提供理论分析证明平滑化的梯度可稳定集合大小，并在多模态数据上验证了更高的几何质量。

**🔧 技术方法**

主要技术包括：分割网络（4‑层 UNet）、DINOv3 预训练特征提取、k‑NN 图构建、随机游走扩散（P S^(t+1)=P S^t）、一致预测风险控制（CRC）与阈值 λ̂ 的校准与推理。

**📊 数据集**

使用了四个公开数据集：MRI（ACDC LV/RV）、超声（CAMUS）、CT（MSD‑Pancreas），每个数据集均进行 2D 切片提取、标准化和裁剪，且在实验中使用 20 张样本作校准集。

**📈 对比分析**

与标准 CRC 和最新方法 Consema 对比，RW‑CP 在 α∈{0.2,0.1,0.05} 下在 Dice、ASSD、HD95 上均实现了明显提升（最高可达 35.4% 的性能提升），且 Stretch（集合尺寸增大比例）明显更低，说明预测集合更紧凑、临床可用性更高。

**⚠️ 局限性**

局限性包括：① 目前仅验证 2D 平面，未扩展至 3D volumetric；② 依赖预训练模型特征，若对域特定图像效果可能不足；③ 随机游走参数（k、β、步数）需经验调优，对计算资源有一定需求。

---

## 129. Implicit Non-Causal Factors are Out via Dataset Splitting for Domain Generalization Object Detection

**arXiv ID:** 2601.19127 | [PDF](https://arxiv.org/pdf/2601.19127v1)

**作者:** Zhilong Zhang `[一作]` (Chongqing University), Fuxiang Huang `[通讯]` (Lingnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过分割数据集生成细粒度域标签并结合对抗学习，提升开放世界目标检测的域泛化能力。

**💡 创新点**

①提出基于 Granular Ball 的细粒度域对抗学习（GB‑DAL），解决传统域标签稀疏导致的隐式非因果因素问题；②引入 Simulated Non‑causal Factors（SNF）对抗扰动模拟，增强模型对隐式非因果因素的识别并提升鲁棒性。

**🔧 技术方法**

Granular Ball 计算、K‑means + 记忆库分割、梯度反转层、Faster R‑CNN、FGSM 对抗训练。

**📊 数据集**

6个目标检测基准（Cityscapes、Foggy Cityscapes、Rain Cityscapes、SIM10k、PASCAL VOC、BDD100k）以及 PACS 图像分类数据集。

**📈 对比分析**

在单源和多源域泛化任务中，与传统 DAL、FACT、FSDR、NP、OA‑DG 等方法对比，GB‑DAL+SNF 在多数跨域设置下取得最高 mAP（如 C&B→F 39.6%，单源 F 48.5%），并在分类任务上提升至 83.2%。

**⚠️ 局限性**

仍受限于对大规模数据集的依赖、K 值需经验调节，以及对未知域噪声的鲁棒性尚不充分。

---

## 130. OATS: Online Data Augmentation for Time Series Foundation Models

**arXiv ID:** 2601.19040 | [PDF](https://arxiv.org/pdf/2601.19040v1)

**作者:** Junwei Deng `[一作]` (University of Illinois Urbana-Champaign), Jiang Bian `[通讯]` (Microsoft Research)

**通讯引用:** 13006 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在线数据增强框架（Online Data Augmentation for Time Series Foundation Models），在训练过程中动态生成与训练步骤匹配的高质量合成时序数据。

**💡 创新点**

创新点在于：①使用数据归因（TSIS）量化样本对模型参考集的影响，构造基于影响分数的指导信号；②利用扩散模型根据这些指导信号生成多样化合成样本；③设计探究‑利用（explore‑exploit）机制平衡计算成本与数据质量。

**🔧 技术方法**

核心技术包括：数据归因（gradient‑based influence score）、扩散模型（conditional denoising diffusion）、探究‑利用采样策略（ε-greedy + exponential moving average）、Transformer‑based TSFM（Encoder‑only / Decoder‑only）。

**📊 数据集**

在六个工业与气候数据集（ETTm1, ETTm2, ETTh1, ETTh2, Weather, Electricity）上进行评估，并在预训练集LOTSA与测试集LSF上验证。

**📈 对比分析**

与常规训练、TSMixup、Jitter等基线对比，使用NLL与MAPE指标。实验显示该方法在大多数数据集和两种TSFM架构上均实现了显著性能提升，尤其在长期预测（预测长度192）下优于所有基线。

**⚠️ 局限性**

局限性包括：①计算 TSIS 仍有一定开销，需要通过探究‑利用策略降低；②合成样本质量高度依赖扩散模型的训练数据，若原始数据不足可能受限；③实验集中于单一任务（时间序列预测），对其他时序任务的通用性尚待验证。

---

## 131. Exploring Weaknesses in Function Call Models via Reinforcement Learning: An Adversarial Data Augmentation Approach

**arXiv ID:** 2601.19122 | [PDF](https://arxiv.org/pdf/2601.19122v1)

**作者:** Weiran Guo `[一作]` (Tongji University), Jingsheng Yang `[通讯]` (ByteDance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于对抗强化学习的函数调用数据增强框架，通过查询模型生成针对函数调用模型弱点的攻击性查询，进而提升 LLM 的函数调用能力。

**💡 创新点**

① 将查询模型与函数调用模型建模为零和博弈，利用 RL 系统性发现并修补弱点；② 采用两阶段过滤+对抗奖励，保证查询既合法又能诱导错误；③ 引入嵌入正则化提升生成查询多样性；④ 迭代交替训练结合课程学习。

**🔧 技术方法**

强化学习（PPO）、零和博弈、查询重写技术、嵌入正则化、多阶段过滤、函数调用自监督训练（SFT+RL+LoRA）。

**📊 数据集**

高质量人工标注的内部工具调用数据作为种子；第二轮加入 xlam‑function‑calling‑60k 的并行样本；使用 Qwen2.5‑7B‑Instruct 训练查询模型，Qwen3‑32B 作为评判模型。

**📈 对比分析**

在 Berkeley Function‑Calling Leaderboard 上与直接 LLM 采样、基线 SFT/LoRA/RL 等方法对比。实验显示：对 Qwen3‑0.6B，使用本框架的数据增强后整体准确率提升约 4–5%，在非实时、实时、相关/不相关检测上均有显著提升；多轮迭代后性能更优。不同规模模型实验表明，模型越小提升越明显，最大提升约 6%。

**⚠️ 局限性**

（1）对大模型（如 Qwen3‑8B）效果不显著；（2）查询重写受限于查询模型规模；（3）判定模型可能出现奖励劫持，导致生成低质量攻击；（4）训练成本较高，需多轮 RL 与大 GPU。

---

## 132. FBSDiff++: Improved Frequency Band Substitution of Diffusion Features for Efficient and Highly Controllable Text-Driven Image-to-Image Translation

**arXiv ID:** 2601.19115 | [PDF](https://arxiv.org/pdf/2601.19115v1)

**作者:** Xiang Gao `[一作]` (Beijing University of Posts and Telecommunications), Yunpeng Jia `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 655 | [OpenAlex ID](https://openalex.org/A5040869443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FBSDiff 与其改进版 FBSDiff++，一种基于频域频段替换的文本驱动图像对图像 (I2I) 翻译框架，实现无训练、无微调、无在线优化的高质量、可控 I2I。

**💡 创新点**

创新点：①从频域角度引入动态频段替换 (FBS/AdaFBS)，直接在潜在扩散特征中实现低频、介频、高频的可控替换，分别对应外观、布局、轮廓的 I2I；②采用 1D‑DCT 与百分位阈值实现对任意尺寸、任意长宽比图片的自适应替换；③通过剔除重建轨迹并直接利用逆向噪声轨迹加速推理，速度提升约 8.9×；④在原始框架基础上轻量化实现局部编辑与风格‑特定内容生成。

**🔧 技术方法**

技术手段：潜在扩散模型 (Latent Diffusion Model)、DDIM 逆向采样、频域变换 (2D/1D DCT)、频段掩码 (low‑, mid‑, high‑pass)、百分位阈值自适应、局部掩码操作、空间变换池 (STP)、CLIP 引导、CFG 方向扩散。

**📊 数据集**

数据集：主要使用 LAION‑Mini（包含 800 张训练/测试图像，分为外观一致与外观差异两类任务），并在多种公开基线模型的实验中使用其默认图像与文本对。

**📈 对比分析**

与 14+ 先进基线（如 InstructPix2Pix、IC‑Edit、SINE、Imagic 等）对比，在外观保持任务中 Structure Similarity、LPIPS、Style Loss、CLIP Similarity、Aesthetic Score 等指标均排名前 4；在外观差异任务中同样表现突出；速度方面 FBSDiff++ 总耗时仅 9.6 s，较 FBSDiff（85.2 s）快约 8.9×，同时对图像尺寸与长宽比不敏感。

**⚠️ 局限性**

局限性：①仍需手动设置阈值（低频/高频/介频阈值），对不同场景的最佳值需要经验；②主要验证于 SD‑v1.5 LDM，迁移到其他扩散模型需要重新调参；③对极端形变或极高分辨率图像仍可能出现轻微细节失真；④无法在同一帧内完成结构性大改（如全局形状重塑）时效果有限。

---

## 133. One Global Model, Many Behaviors: Stockout-Aware Feature Engineering and Dynamic Scaling for Multi-Horizon Retail Demand Forecasting with a Cost-Aware Ordering Policy (VN2 Winner Report)

**arXiv ID:** 2601.18919 | [PDF](https://arxiv.org/pdf/2601.18919v1)

**作者:** Bartosz Szabłowski `[一作]` `[通讯]`, Bartosz Szabłowski

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个两阶段的预测-优化管道，用全局多时段的 CatBoost 预测模型生成未来三周的需求并基于成本感知的库存投影规则生成订单；

**💡 创新点**

创新点在于将全局学习与库存投影相结合：1）利用库存缺货感知的特征工程、动态归一化与时间衰减权重提升全局预测质量；2）通过将单点预测与新闻销售模型的成本阈值、标准差近似结合，得到可解释的安全库存目标；

**🔧 技术方法**

使用的技术包括：全局多时段 CatBoost 回归、特征工程（滞后、滚动统计、季节性、间歇性、缺货掩蔽）、动态规模因子、时间衰减观测权重、基于标准差的安全系数以及简单的库存投影与订单规则；

**📊 数据集**

数据集为 VN2 竞赛数据，包含 599 个店铺-产品对的历史每周销量（约 157 周）、库存可售状态、静态商店/产品信息及初始库存；

**📈 对比分析**

方法通过竞赛模拟六轮（共 8 周）进行评估，与基准（季节加权 13 周移动平均 + 4 周安全库存）相比，最终总成本下降约 13.2%（3.763€ 对比 4.334€），获得第一名；

**⚠️ 局限性**

局限包括：1）仅使用单点预测，未构建完整的预测分布；2）安全库存计算基于正态近似，对高度间歇或极端事件的鲁棒性有限；3）缺少对新产品、产能限制、订单最小化/包装约束等实际运营约束的考虑；4）模型需在每次迭代重新训练，未探索端到端的预测-决策优化。

---

## 134. Optimal Motion Planning for Two Square Robots in a Rectilinear Environment

**arXiv ID:** 2601.19147 | [PDF](https://arxiv.org/pdf/2601.19147v1)

**作者:** Pankaj K. Agarwal `[一作]` (Duke University), Martijn Struijs `[通讯]` (TU Eindhoven)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了在二维轴对齐多边形环境中，对两个单位正方形机器人执行最优运动规划的两种变体：最小总路径长度（min‑sum）和最小制动时间（min‑makespan）。

**💡 创新点**

创新点包括：
• 证明在此环境下存在最优的“规范网格”方案，并利用该结构构造了一个 4 维加权网格图；
• 设计了一个 O(n⁴ log n) 的多项式时间算法来求解 min‑sum 变体，成为首个对该问题给出最优解的多项式算法；
• 通过对 Partition 问题的还原，证明 min‑makespan 变体是 NP‑难的，从而阐明两种变体在计算复杂度上的根本差异。

**🔧 技术方法**

主要技术方法包括：
• 对路径进行“推送”与“切换”操作，将任意最优解变形为在 O(n) × O(n) 的非均匀网格上的方案；
• 通过结构性质（如交错、x‑分离、y‑分离、swap 区间）控制碰撞与长度变化；
• 构造 4 维配置空间网格图并在其上使用 Dijkstra 求最短路径；
• 采用分段重参数化和“幽灵段”处理不连续点，保持解的连贯性。

**📊 数据集**

本文未使用任何实验数据集，全部工作基于理论证明与算法设计，主要关注理论复杂度和最优性证明。

**📈 对比分析**

与现有工作相比，本文首次给出了对两机器人最小总路径长度规划的多项式时间解法；相较于仅能得到可行解或近似解的先前算法，运行时间为 O(n⁴ log n)。
对于 min‑makespan 变体，证明其 NP‑难性表明现有多项式时间或有效近似算法的期望是有限的。

**⚠️ 局限性**

局限性包括：
• 仅适用于两个机器人，未扩展到三人以上；
• 环境限定为轴对齐多边形（可含洞），不适用于一般多边形或三维空间；
• min‑makespan 变体没有得到多项式时间解，仅证明了其难度；
• 对实际机器人运动的动力学约束（加速度、转弯半径）不做处理。

---

## 135. A Security Analysis of CheriBSD and Morello Linux

**arXiv ID:** 2601.19074 | [PDF](https://arxiv.org/pdf/2601.19074v1)

**作者:** Dariy Guzairov `[一作]`, Alwen Tiu `[通讯]` (Australian National University)

**通讯引用:** 1790 | [OpenAlex ID](https://openalex.org/A5071064651)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对CheriBSD和Morello Linux在CHERI架构下的隔离机制进行安全分析，揭示四种攻击手段（栈遍历、dlopen泄漏、堆扫荡、堆存储）能突破库隔离并获取主程序敏感数据。

**💡 创新点**

首次系统性提出递归能力扫描方法，证明动态链接器内部结构泄露及堆内存可被利用，展示了CHERI实现中的隔离缺陷。

**🔧 技术方法**

利用CHERI能力指针、动态链接器（dlopen）内部结构、GDB调试、OpenSSL RSA密钥生成等技术。

**📊 数据集**

在ARM Morello原型板上自行编译的测试程序与示例程序，使用OpenSSL生成的RSA密钥做验证；未使用公开数据集。

**📈 对比分析**

通过对比加Mitigation前后的攻击成功率和对内存/执行时间的影响，评估了不同缓解措施的有效性；但文中未给出具体性能数值，主要说明相对成本。

**⚠️ 局限性**

仅在用户空间进行攻击，未覆盖系统调用层面；对特定CHERI实现（Morello Linux、CheriBSD）有限；需要库提供malloc/dlopen等接口；部分攻击在CheriBSD上因c18n/heap revocation 被阻止；实验规模小，未评估在大规模生产系统中的普适性。

---

## 136. LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems

**arXiv ID:** 2601.19121 | [PDF](https://arxiv.org/pdf/2601.19121v1)

**作者:** Guilin Zhang `[一作]` (Workday), Xu Chu `[通讯]` (Workday)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大型语言模型（LLM）协同的双代理框架 DualAgent-Rec，用于满足硬约束的多目标电商推荐问题。

**💡 创新点**

创新点在于将优化任务拆分为专注可行解的利用代理和忽略约束、探索多样解的探索代理，并用 LLM 作为协调器自适应分配资源，同时采用自校准的 ϵ‑放宽机制保证最终可行性。

**🔧 技术方法**

主要技术包括进化多目标优化、约束支配原则（CDP）、Pareto 非支配排序、双代理知识迁移、动态 ϵ‑约束放宽和基于 Qwen2.5‑14B 的 LLM 协调器。

**📊 数据集**

实验使用 Amazon Reviews 2023 数据集，在 All_Beauty、Electronics、Clothing_Shoes_Jewelry 三个商品类别上进行评估。

**📈 对比分析**

与规则化资源分配、单一种群、无约束等基线对比，DualAgent-Rec 在保证 100% 约束满足的前提下，平均提升 4–6% 的 Pareto 超体积，并保持相近的 NDCG 与多样性，显示出显著性能优势。

**⚠️ 局限性**

局限性包括仅在电商领域验证，实时推理时 LLM 调度的延迟较大，需进一步扩展到更大规模商品集和多领域场景，并对不同 LLM 进行模型压缩与效率优化。

---

## 137. CanaryBench: Stress Testing Privacy Leakage in Cluster-Level Conversation Summaries

**arXiv ID:** 2601.18834 | [PDF](https://arxiv.org/pdf/2601.18834v1)

**作者:** Deep Mehta `[一作]` `[通讯]` (Adobe Inc), Deep Mehta (Adobe Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了可复现的CanaryBench基准，用以检测聚类级对话摘要中的可泄露信息。

**💡 创新点**

创新在于通过注入已知的canary字符串来量化文字摘要的泄露率，并提出简单的k-最小阈值和正则表达式去重策略。

**🔧 技术方法**

采用TF-IDF嵌入、k-means聚类、关键词非提取式和示例提取式摘要，以及正则式PII检测。

**📊 数据集**

使用了3000条合成对话，其中60%注入了canary和20%随机PII。

**📈 对比分析**

在未防御的示例提取摘要下，集群级泄漏率达96.2%，通过k-min+去重后降至0，且聚类连贯度仅略有下降。

**⚠️ 局限性**

局限包括仅使用合成数据、只检测精确字符串泄漏、正则表达式漏检以及缺乏人类评估。

---

## 138. Is Finer Better? The Limits of Microscaling Formats in Large Language Models

**arXiv ID:** 2601.19026 | [PDF](https://arxiv.org/pdf/2601.19026v1)

**作者:** Andrea Fasoli `[一作]` (IBM Research), Naigang Wang `[通讯]` (IBM Research)

**通讯引用:** 3587 | [OpenAlex ID](https://openalex.org/A5082043392)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了微尺度量化（microscaling）在小块大小下的误差反转现象，构建了理论框架解释误差来源，并提出了FP8无符号E5M3（UE5M3）尺度格式来缓解该问题，验证了其在多种LLM上的性能提升。

**💡 创新点**

创新点包括：①首次系统揭示块尺寸减小导致误差升高的“误差反转”现象；②构建了可分解误差来源的理论模型，验证了尺度量化是主要罪魁；③提出硬件友好的UE5M3尺度格式，利用一位未使用的指数位扩展尺度动态范围，从而在保持FP4元素精度的同时显著降低误差；④将该方案与传统UE4M3及全局标量缩放等方法对比，证明其在无额外计算开销的情况下可获得更优或相当的性能。

**🔧 技术方法**

技术方法包括：基于FP4元素与FP8 UE4M3/UE5M3尺度的微尺度量化；使用实验测量困惑度（perplexity）与均方误差（MSE）；构建正态分布等理想分布的理论误差推导；数值积分求解误差贡献；硬件实现层面在Systolic Array上演示UE5M3的可行性与成本评估。

**📊 数据集**

主要数据集来自预训练大型语言模型的权重与激活分布，包括granite‑3.3‑8b、llama‑3.1‑8b、llama‑2‑7b、mixtral‑8x7b‑instruct等模型权重，实验测量在这些模型上的困惑度与各量化格式下的性能。

**📈 对比分析**

对比方法包括：FP4微尺度量化与BF16基准、UE4M3、UE4M3加全局标量缩放（UE4M3‑S）以及提出的UE5M3。实验显示，在块尺寸为8时，UE5M3在多数模型上实现了比UE4M3‑S更低的困惑度，且在无额外标量缩放开销的情况下，性能与或优于传统方案；部分模型如bamba‑9b‑v2在UE4M3下出现显著失真，而UE5M3显著恢复准确性。

**⚠️ 局限性**

局限性包括：理论框架主要针对正态分布假设，虽然对多模型表现良好，但对极端分布可能仍有偏差；硬件实现以Systolic Array为例，实际在不同加速器上实现成本与功耗尚待进一步评估；UE5M3虽然提升了尺度动态范围，但对非常宽分布或极大稀疏度的模型仍需进一步验证。

---

## 139. GTFMN: Guided Texture and Feature Modulation Network for Low-Light Image Enhancement and Super-Resolution

**arXiv ID:** 2601.19157 | [PDF](https://arxiv.org/pdf/2601.19157v1)

**作者:** Yongsong Huang `[一作]` (Yale University), Shinichiro Omachi `[通讯]` (Tohoku University)

**通讯引用:** 1388 | [OpenAlex ID](https://openalex.org/A5020830042)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种双流网络GTFMN，用于解决低光低分辨率图像的超分辨率问题，将任务拆分为光照估计与纹理恢复两部分，并通过光照引导的特征调制实现空间自适应增强。

**💡 创新点**

创新点在于：①引入独立的光照流精准估计像素级照明映射；②设计Illumination‑Guided Modulation Block（IGM）利用光照引导与自注意力融合动态调制纹理特征；③整体框架实现低光与超分辨率的联合处理，参数量仅约8.78M。

**🔧 技术方法**

使用了多尺度自注意力、适配器网络生成引导注意力、残差前馈网络、PixelShuffle上采样以及L1损失的Adam优化。

**📊 数据集**

在自构建的OmniTrain训练集（1600张高分辨率图像及其合成低光低分辨率版本）上训练，并在OmniNormal5与OmniNormal15两套测试集上评估。

**📈 对比分析**

与SRCNN、FSRCNN、RCAN、ESRGAN、SwinIR、ShuffleMixer、HAT、MambaIRv2等SOTA方法对比，GTFMN在×2和×4尺度下在PSNR、SSIM、LPIPS等指标上均领先，尤其在×4尺度下PSNR提升约1.1dB，参数更少。

**⚠️ 局限性**

局限性包括：①深层IGM块会增加计算开销，需在实时/嵌入式场景中权衡；②光照估计依赖于训练数据，可能对极端照明场景泛化能力有限；③目前未在真实低光+低分辨率数据集上验证，合成数据可能与实际应用存在偏差。

---

## 140. FTA-NTN: Fairness and Throughput Assurance in Non-Terrestrial Networks

**arXiv ID:** 2601.19078 | [PDF](https://arxiv.org/pdf/2601.19078v1)

**作者:** Sachin Ravikant Trankatwar `[一作]` (University of Ottawa), Burak Kantarci `[通讯]` (University of Ottawa)

**通讯引用:** 8425 | [OpenAlex ID](https://openalex.org/A5003131477)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个多目标优化框架FTA-NTN，用以在非地面网络星座设计中同时最大化吞吐量和公平性。

**💡 创新点**

创新点在于将多层Walker‑Delta星座、参数化加拿大移动模型、适应性K‑Means聚类与贝叶斯优化融合成可扩展的多目标优化流程，突破了传统单目标吞吐量设计的局限。

**🔧 技术方法**

使用技术包括：多目标加权求和优化、贝叶斯优化搜索、K‑Means聚类进行beam分配、Walker‑Delta星座建模、参数化用户移动仿真、FSPL+EPL信道模型、SINR计算与Jain公平性指数评估。

**📊 数据集**

实验数据为500名在加拿大陆地区域内动态移动的仿真用户轨迹，使用3GPP NTN规范参数进行信道与网络仿真，未使用公开真实数据集。

**📈 对比分析**

通过50次仿真对比，评估吞吐量与公平性，在最佳配置（LEO 9平面×15卫星/平面，MEO 7平面×3卫星/平面）下平均总吞吐 9.88 Gbps、平均公平性 0.42，符合3GPP参考标准，显示相较单目标设计的性能提升。

**⚠️ 局限性**

局限性包括：仅考虑LEO和MEO层配置且GEO层固定；未涵盖高空平台（HAP）、卫星间链路（ISL）与切换等实际干扰；仿真采用理想化信道与移动模型，缺乏真实部署验证。

---

## 141. TFFM: Topology-Aware Feature Fusion Module via Latent Graph Reasoning for Retinal Vessel Segmentation

**arXiv ID:** 2601.19136 | [PDF](https://arxiv.org/pdf/2601.19136v1)

**作者:** Iftekhar Ahmed `[一作]` (Leading University), Seraj Al Mahmud Mostafa `[通讯]` (University of Maryland)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5009690297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种拓扑感知的视网膜动静脉分割框架，显著降低血管断裂，提高拓扑连通性；

**💡 创新点**

核心创新在于Topological Feature Fusion Module（TFFM）将特征映射到图空间并使用图注意力网络捕捉全局连通性，同时采用Tversky+soft clDice混合损失主动惩罚拓扑断裂；

**🔧 技术方法**

技术包括U‑Net+++Attention Gate编码器、EfficientNet‑B0特征提取、TFFM、图注意力网络、Tversky损失、soft clDice损失及多尺度数据增强；

**📊 数据集**

主要使用Fundus‑AVSeg基准数据集（100张高分辨率眼底图），并在五个公开数据集上进行零样本跨域验证；

**📈 对比分析**

与U‑Net、Attention‑U‑Net、U‑Net++、TransUNet等传统模型对比，最终模型在Fundus‑AVSeg上获得90.97% Dice、3.50像素HD95、85.55% clDice，碎片化率仅25.3；跨域测试中Dice仍保持约80%+，clDice在68–74%之间；

**⚠️ 局限性**

局限在于分支点（交叉点）召回率低（0.44/0.52），导致在极其复杂的分叉处仍易出现断裂，未来需针对分叉的专门注意力机制或轻量化部署。

---

## 142. People Can Accurately Predict Behavior of Complex Algorithms That Are Available, Compact, and Aligned

**arXiv ID:** 2601.18966 | [PDF](https://arxiv.org/pdf/2601.18966v1)

**作者:** Lindsay Popowski `[一作]` (Stanford University), Michael S. Bernstein `[通讯]` (Stanford University)

**通讯引用:** 63173 | [OpenAlex ID](https://openalex.org/A5076189854)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 ACA 理论（可用性、紧凑性、对齐性）来解释何时用户能准确预测算法行为，并通过 1250 名受试者对 25 种社交媒体推荐算法的预测实验验证该理论。

**💡 创新点**

创新点在于将三大认知特征（Availability、Compactness、Alignment）系统化为必要且充分的条件，证明即使是极其复杂的 AI 算法，只要满足这三点也能被用户准确预测；同时提供了以心理模型为核心的实验框架。

**🔧 技术方法**

实验采用心理模型测量（开放式自评）、多元混合效应逻辑回归分析以及 GPT‑4o 辅助编码等技术，来量化预测准确率和模型匹配程度。

**📊 数据集**

使用的实验数据集为来自 Twitter/X 的政治主题推文，随机挑选 25 种不同的排序算法（包含简单计数、BERT、GPT、Perspective API 等），每种算法都在实验中呈现给受试者。

**📈 对比分析**

通过与随机猜测基线（50%）比较，发现满足 ACA 条件的算法预测准确率平均 85%，而不满足条件的仅 54%；三因素交互显著提升预测性能，表明三者共同作用最为关键。

**⚠️ 局限性**

局限性包括：仅在单一社交媒体（政治推文）情境下测试；实验使用静态算法，未考虑算法随时间变更；样本来自在线众包平台，缺乏长期交互与多样化用户群体；未涵盖个性化或真实平台算法，且实验只关注行为预测而非错误解释或内部机制理解。

---

## 143. Proactive Hardening of LLM Defenses with HASTE

**arXiv ID:** 2601.19051 | [PDF](https://arxiv.org/pdf/2601.19051v1)

**作者:** Henry Chen `[一作]` (Palo Alto Networks), Nicole Nichols `[通讯]` (Palo Alto Networks)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HASTE 框架，通过生成、评估、改进循环自动化产生高逃逸率的恶意提示，并将其用于持续强化 LLM 的恶意提示检测模型。

**💡 创新点**

创新点在于：① 将硬负样本挖掘与多种模糊技术（语义、句法、格式）组合成可插拔的闭环流程；② 对检测模型进行按时序迭代训练，显著缩短收敛周期；③ 提供基于攻击类型的细粒度评估与可解释性分析。

**🔧 技术方法**

使用技术包括：基于 LLM 的生成（如 GPT‑4o）、LLM‑as‑Judge（JailJudge）评估、语义/句法/格式模糊变换、硬负样本挖掘、Transformer（DeBERTa‑V3）分类器微调、SHAP 解释、税onomic 标签体系。

**📊 数据集**

数据集：约 4,500 条种子恶意提示（来自公开/内部资源），4,000 条同类型伪造提示；约 40,000 条正常提示；实验过程中通过生成和模糊扩充得到数万条训练样本。

**📈 对比分析**

与基线检测模型（ProtectAI‑Deberta‑V3）对比：基线在迭代 0 时准确率 95.9%；硬负挖掘后检测器在未再训练时准确率降至 31%（相较 66% 的单纯模糊提升）。在仅 5 次迭代后重新训练的模型在外部评估集上可达 94%+ 的准确率，完成度与仅使用模糊或单纯迭代的方案相比提升约 3–4 个百分点，同时迭代次数减少约 50%。

**⚠️ 局限性**

局限性：① 仅针对单轮提示；② 只评估单一基线分类器；③ 模糊方法与阈值选择固定，未充分探究其参数空间；④ 对多轮对话、跨模型适应性、评估指标多样性等方面缺乏深入验证。

---

## 144. HEATACO: Heatmap-Guided Ant Colony Decoding for Large-Scale Travelling Salesman Problems

**arXiv ID:** 2601.19041 | [PDF](https://arxiv.org/pdf/2601.19041v1)

**作者:** Bo-Cheng Lin `[一作]` (Victoria University of Wellington), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**通讯引用:** 30584 | [OpenAlex ID](https://openalex.org/A5100400258)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个基于热图的非自回归 TSP 求解器的解码器——HeatACO，用于将热图转换为可行的 Hamiltonian 循环；

**💡 创新点**

创新点在于将热图视为软先验，通过在最大-最小蚂蚁系统（MMAS）中引入热图权重，并利用蟻蜜源更新进行实例特定的全局反馈，从而在不需要昂贵 MCTS 或手工工程的情况下实现高质量、可扩展的解码；

**🔧 技术方法**

采用最大-最小蚂蚁系统（MMAS）作为基本框架，热图作为多项式幂的先验因子，配合可选的 2‑opt / 3‑opt 本地改进；

**📊 数据集**

在标准 TSP500、TSP1K、TSP10K（N=500, 1000, 10000）以及部分 TSPLIB OOD 实例上进行评估，使用四个公开预训练热图预测器（AttGCN、DIMES、UTSP、DIFUSCO）；

**📈 对比分析**

与贪婪合并、MCTS‑guided k‑opt、以及传统的 MMAS/深度 ACO 等基线比较，HeatACO+2‑opt 在固定热图下实现 0.11%/0.23%/1.15% 的平均最优性缺口，并且 CPU 解码时间仅为秒级到分钟级，优于现有的贪婪解码与 MCTS 解码；

**⚠️ 局限性**

局限性主要在于热图的可靠性：如果热图的排名不具备高召回或出现低置信度崩塌，HeatACO 的优势会显著下降；此外，该方法仍依赖于合理的候选边阈值与 γ 参数的调优。

---

## 145. Foresight Learning for SEC Risk Prediction

**arXiv ID:** 2601.19189 | [PDF](https://arxiv.org/pdf/2601.19189v1)

**作者:** Benjamin Turtel `[一作]` (Lightning Rod Labs), Kris Skotheim `[通讯]` (Lightning Rod Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用自动化流水线将SEC风险披露转化为时间约束的风险查询，并通过后续披露自动标注其是否实现，训练专门的语言模型预测风险实现的概率。

**💡 创新点**

创新点在于：①使用未来披露作为无监督标注来源，构建大规模风险级别监督；②将Foresight Learning框架应用于长篇、非结构化监管文本，实现置信概率预测；③在仅含单GPU即可部署的模型上达到甚至超过前沿大模型的性能。

**🔧 技术方法**

技术包括：基于Gemini的检索增强生成（RAG）生成风险查询；使用LightningRod AI自动生成监督标签；在Qwen3-32B上采用GRPO强化学习优化Brier分数；使用Brier分数、BSS和ECE评估概率预测。

**📊 数据集**

数据集为6,109条风险查询，来自2,820份SEC 10‑K/10‑Q文件，覆盖1,953家公司，训练集5,609条，测试集500条，按时间顺序划分。

**📈 对比分析**

与基线（无预测、经验基率）和GPT‑5对比，Fine‑tuned Qwen3-32B在Brier分数降至0.1979、BSS提升至11.6%，ECE降至0.0287，显著优于预训练基模型和GPT‑5，表现出更高的准确性和校准度。

**⚠️ 局限性**

主要限制：①标注依赖披露，可能遗漏未披露或模糊披露的风险；②预测时间窗口短，未覆盖长期、渐进风险；③模型学习的是披露概率而非真实事件发生率，受公司披露行为与激励影响。

---

## 146. Resolving Primitive-Sharing Ambiguity in Long-Tailed Industrial Point Cloud Segmentation via Spatial Context Constraints

**arXiv ID:** 2601.19128 | [PDF](https://arxiv.org/pdf/2601.19128v1)

**作者:** Chao Yin `[一作]` (Guangzhou Institute of Geography, Guangdong Academy of Sciences), Wei Yao `[通讯]` (State Key Laboratory of Regional and Urban Ecology, Institute of Urban Environment, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出基于空间上下文约束的长尾工业点云分割方法，用Boundary-CB和Density-CB解决几何模糊与类别不平衡问题。

**💡 创新点**

创新点在于将邻域预测一致性（熵）与扫描密度调节融入Class-Balanced Loss，实现无网络改造的插件式约束，专门针对工业场景的“dual crisis”。

**🔧 技术方法**

采用Class-Balanced Loss、Boundary-CB、Density-CB以及ResPointNet++骨干网络，并利用k近邻熵及局部点密度进行加权。

**📊 数据集**

使用工业场景大规模点云数据集Industrial3D（约610M点，12类）。

**📈 对比分析**

与传统交叉熵、CB+Focal Loss及Density-CB相比，Boundary-CB在工业3D上实现55.74% mIoU，尾类mIoU提升21.7%（从24.32%到29.59%），并保持头类精度不下降。

**⚠️ 局限性**

局限在于极少样本类别（如Valve）仍表现不佳，且对不同几何尺度的最佳邻域大小存在敏感性，需要进一步的自适应尺度或少样本学习。

---

## 147. Contrastive Spectral Rectification: Test-Time Defense towards Zero-shot Adversarial Robustness of CLIP

**arXiv ID:** 2601.19210 | [PDF](https://arxiv.org/pdf/2601.19210v1)

**作者:** Sen Nie `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xilin Chen `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 34757 | [OpenAlex ID](https://openalex.org/A5083420537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种对CLIP模型的测试时防御方法CSR，通过频域低通滤波检测对抗样本并在对比学习框架下优化纠正扰动，从而提升模型对强攻击的鲁棒性。

**💡 创新点**

创新点在于发现CLIP对抗样本在频谱中的脆弱性，并提出利用低通滤波特征作为正样本、原始特征作为负样本的对比谱纠正策略，同时引入自适应门控机制和贪婪选择，实现高效、可扩展的测试时防御。

**🔧 技术方法**

使用频域低通滤波、对比学习损失、投影梯度下降（PGD）优化纠正扰动、余弦相似度检测与贪婪选择等技术。

**📊 数据集**

在16个零样本分类基准（ImageNet、CIFAR‑10/100、STL10、Caltech‑101/256、OxfordPets、Flowers102、Food101、StanfordCars、SUN397、Country211、FGVCAircraft、EuroSAT、DTD、PCAM）以及语义分割、图像字幕、视觉问答等任务的数据集上进行评估。

**📈 对比分析**

与现有对抗微调和测试时防御方法（TeCoA、FARE、R‑TPT、HD、Anti‑Adv、LPF、TTC、TTE）对比，CSR在PGD/AutoAttack下平均提升约18.1%鲁棒率，仅额外增加约0.7%清洁精度，推理时间比R‑TPT低12倍。

**⚠️ 局限性**

局限性包括对极强攻击预算或高频噪声仍有限，需调节阈值和滤波半径；在不同模型规模或更大VLM时可能需要重新调参；未验证对所有复杂任务的完全迁移性。

---

## 148. FreeOrbit4D: Training-Free Arbitrary Camera Redirection for Monocular Videos via Geometry-Complete 4D Reconstruction

**arXiv ID:** 2601.18993 | [PDF](https://arxiv.org/pdf/2601.18993v1)

**作者:** Wei Cao `[一作]` (University of Illinois Urbana-Champaign), Yaoyao Liu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 718 | [OpenAlex ID](https://openalex.org/A5012509567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出FreeOrbit4D，一个无训练需求的框架，通过构建完整的4D几何代理实现单摄像头视频的大角度相机重定向。

**💡 创新点**

创新点在于将全局场景重建与对象几何补全分离，利用多视角扩展与像素同步的3D-3D对应实现几何完整的4D代理，并以此为条件生成高质量、时间连贯的新视角视频。

**🔧 技术方法**

采用了预训练的VGGT、SAM2、SV4D2.0、PAGE-4D、Wan2.2-VACE等模型，结合多视角视频扩散、深度条件扩散与Kalman滤波等技术。

**📊 数据集**

使用DAVIS、VEO、Sora等真实与合成视频数据集，以及公开网络视频进行评估。

**📈 对比分析**

与ReCamMaster、TrajectoryCrafter、EX4D、GEN3C等方法比较，显示在FID、VBench、CLIP-SIM等指标以及20人用户研究中均实现或接近SOTA，尤其在大角度相机轨迹跟随和时间一致性上表现突出。

**⚠️ 局限性**

局限性包括高计算成本（单段视频约50分钟推理）、对预训练模型的依赖、在极快运动或遮挡极端情况下的几何精度可能受限。

---

## 149. GUIGuard: Toward a General Framework for Privacy-Preserving GUI Agents

**arXiv ID:** 2601.18842 | [PDF](https://arxiv.org/pdf/2601.18842v1)

**作者:** Yanxi Wang `[一作]` (Beijing Normal University), Jiyan He `[通讯]` (Zhongguancun Academy)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于三阶段（隐私识别→隐私保护→任务执行）的GUI代理框架 GUIGuard，并构建了覆盖安卓与PC的跨平台隐私评测基准 GUIGuard‑Bench；

**💡 创新点**

创新点包括：①将隐私保护流程拆分为可本地化的识别与后端执行的混合服务；②首次为 GUI 轨迹提供区域级隐私标注、风险等级、隐私类别和任务必要性；③提出了面向隐私保护的评测协议（LLM‑as‑Judge + self‑comparison），可系统量化保护后任务一致性；

**🔧 技术方法**

技术方法涵盖：基于 VLM 的隐私识别（文本+位置匹配）；多维隐私保护（像素遮罩、语义替换、潜在层扰动等）；任务执行评估采用 LLM‑as‑Judge 评判规划语义一致性；

**📊 数据集**

使用了 630 条真实与合成轨迹，包含 13,830 张屏幕截图的 GUIGuard‑Bench 数据集，数据覆盖 Android 与 PC 两大平台，并标注了 6 种隐私类别、3 个风险等级与任务必要性；

**📈 对比分析**

通过对 5 个闭源 VLM（GPT‑5.1、Gemini‑3、Claude‑Sonnet‑4.5 等）、3 个开源 VLM（Qwen‑3‑VL、DeepSeek‑VL2 等）和 3 个专用 GUI 代理的比较；结果显示：闭源模型在隐私识别精度和任务执行一致性上显著优于开源模型（闭源端到端识别约 10%/PC 5%/Android 13%），但在强遮罩下任务成功率显著下降；

**⚠️ 局限性**

局限性：①隐私识别召回率低，尤其在 PC 场景；②保护方法多为传统像素/文本替换，缺乏对视觉语义的细粒度控制；③任务执行评测主要基于静态数据集，未覆盖长周期交互与错误回溯；④未提供端到端的训练或自适应策略，导致保护后执行性能受限。

---

## 150. Self-Aware Knowledge Probing: Evaluating Language Models' Relational Knowledge through Confidence Calibration

**arXiv ID:** 2601.18901 | [PDF](https://arxiv.org/pdf/2601.18901v1)

**作者:** Christopher Kissling `[一作]` (Humboldt-Universität zu Berlin), Alan Akbik `[通讯]` (Humboldt-Universität zu Berlin)

**通讯引用:** 3388 | [OpenAlex ID](https://openalex.org/A5032877157)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套三模态（内在置信度、结构一致性、语义基础）校准探测框架，用于评估语言模型在关系知识推理中的可靠性。

**💡 创新点**

首次将模型的自我意识与知识检索的置信度拆分为三种维度，并提出统一的置信度估计与校准方法，突出结构一致性对校准的显著提升。

**🔧 技术方法**

采用BEAR闭集评测、softmax归一化与C_Margin、C_Average、C_Consistency等多种置信度估计；使用投票与最小置信度聚合策略；注入语义标记（certainly/possibly）与数值置信度表达。

**📊 数据集**

主要使用BEAR数据集（7,731条实例、60种关系，5种等价模板）和Wiki-FACTOR子集（每条语句四个答案）。

**📈 对比分析**

对16个模型（10 CLM、6 MLM）在ACE、Brier Score、校准曲线等指标进行对比，结果显示CLM总体校准更好，C_Average^Min是最优估计；MLM在相同准确度下校准差距显著，且过度自信。

**⚠️ 局限性**

局限性包括仅针对闭集问答，未覆盖N:M关系；结构一致性依赖模板敏感，计算成本高；校准结果难以直接迁移到开放式、无预定义答案的实际应用场景。

---

## 151. Pay Attention to Where You Look

**arXiv ID:** 2601.18970 | [PDF](https://arxiv.org/pdf/2601.18970v1)

**作者:** Alex Beriand `[一作]`, Abhijit Mahalanobis `[通讯]` (University of Arizona)

**通讯引用:** 2544 | [OpenAlex ID](https://openalex.org/A5055793435)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了相机加权机制来提升少样本新视角合成的质量。

**💡 创新点**

创新点在于将相机重要性自动化权重化，提供确定性与注意力两种方案。

**🔧 技术方法**

采用欧氏距离、角度误差、Gaussian核等确定性加权，或交叉注意力学习加权，并可直接替换原有平均聚合。

**📊 数据集**

在SRN Cars数据集（和SRN Multi-Chairs）上进行评测。

**📈 对比分析**

与传统均值权重相比，误差权重和交叉注意力权重显著降低FID、LPIPS等指标，提升PSNR、SSIM，并在近视角输入时表现尤为突出。

**⚠️ 局限性**

缺点是需要额外的相机编码和注意力训练，且在输入视角相差较大时提升有限。

---

## 152. XIMP: Cross Graph Inter-Message Passing for Molecular Property Prediction

**arXiv ID:** 2601.19037 | [PDF](https://arxiv.org/pdf/2601.19037v1)

**作者:** Anatol Ehrlich `[一作]` (University of Vienna), Nils M. Kriege `[通讯]` (University of Vienna)

**通讯引用:** 1406 | [OpenAlex ID](https://openalex.org/A5009569829)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出跨图交互式消息传递框架 XIMP，在单个模型中同时处理分子图、分支树（Junction Tree）和扩展简化图（Extended Reduced Graph），实现多层次、多视角的特征学习，用以改进分子性质预测。

**💡 创新点**

创新点在于：①支持任意数量的图抽象；②实现直接（DIMP）与间接（I2MP）两种跨图消息传递；③在每一层中对不同图的嵌入进行统一聚合，避免信息过压缩并提升表达能力；④通过学习的多视角读出策略（拼接/求和）进一步提升性能。

**🔧 技术方法**

使用技术包括：图神经网络（GCN、GIN、GAT、GraphSAGE）、Junction Tree 与 Extended Reduced Graph 的构建、节点对应矩阵 S 进行跨图映射、DIMP/I2MP 机制、线性变换与非线性激活、全局平均池化或拼接读出，以及多层 MLP 回归头。

**📊 数据集**

数据集涵盖十个 MoleculeNet 任务（如 ESOL、FreeSolv、Lipo 等）以及 Polaris 10% scaffold‑split 的 ADMET、Potency、MoleculeNet 等子任务，样本量从数百到数万不等，尤其注重低数据场景。

**📈 对比分析**

与多种基线（GCN、GIN、GAT、GraphSAGE、HIMP、固定指纹 ECFP）在 10 任务上进行严格对比；采用 10‑fold stratified CV 选择超参，再在 10% scaffold‑split 测试集评估。实验显示 XIMP 在多数任务（尤其低数据场景）均能获得最低 MAE，性能优于 HIMP 与传统 GNN，同时超过固定指纹的表现。

**⚠️ 局限性**

局限性包括：需要手工构造并正确对齐多种图抽象，随着抽象数增加计算与参数量呈多项式增长；目前仅在分子领域验证，未知对其他领域的通用性；抽象选择对任务依赖较强，若抽象不合适可能无法获得收益。

---

## 153. Educational Database Prototype: the Simplest of All

**arXiv ID:** 2601.19165 | [PDF](https://arxiv.org/pdf/2601.19165v1)

**作者:** Yi Lyu `[一作]` (University of Wisconsin-Madison), Takashi Matsuzawa `[通讯]` (University of Wisconsin-Madison)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

实现了一个轻量级、可扩展的教育用数据库原型EduDB，并设计了基于该原型的课程项目和自动化评测体系。

**💡 创新点**

创新点在于将数据库系统的核心模块（解析器、执行器、缓冲区、文件、并发、事务）浓缩为约1700行代码，既保证可执行性又便于学生深入学习与改进，同时提供了统一的评测和排行榜平台。

**🔧 技术方法**

使用技术包括C++实现、socket通信、基于规则的词法/语法解析、基于缓冲池的页管理、基于锁表的并发控制，以及简单的事务提交/回滚逻辑。

**📊 数据集**

未使用真实大规模数据集；评测使用自定义的合成测试查询和人工生成的表结构与记录，以便在服务器上统一运行。

**📈 对比分析**

比较方法是让学生提交自定义的join实现（如hash join），在服务器上按时间执行同一套查询并与基线的嵌套循环 join 对比；据描述，若执行时间明显快于基线则得分，具体性能提升比例通过自动化脚本记录。

**⚠️ 局限性**

主要限制包括：缺乏系统化的大规模功能与并发测试、文档和注释不足、未实现索引/查询优化器、join实现仍有待完善，且无死锁检测与恢复机制。

---

## 154. Variational Quantum Circuit-Based Reinforcement Learning for Dynamic Portfolio Optimization

**arXiv ID:** 2601.18811 | [PDF](https://arxiv.org/pdf/2601.18811v1)

**作者:** Vincent Gurgul `[一作]` (Humboldt University Berlin), Stefan Lessmann `[通讯]` (Humboldt University Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了量子强化学习（QRL）在动态组合优化中的应用，并实现了基于变分量子电路的 DDPG 与 DQN 算法。

**💡 创新点**

创新点在于将 QRL 框架迁移到高维金融决策问题，通过幅度编码压缩输入，仅使用几十个可训练参数即可实现与传统深度 RL 相当甚至更优的风险调整收益。

**🔧 技术方法**

使用了变分量子电路（VQC）作为策略和值函数逼近器，采用参数偏移规则进行梯度估计，并在 NISQ 设备上完成量子训练与经典后处理。

**📊 数据集**

实验数据来自 2011‑2025 年的 5049 条每日收盘价，涵盖 15 只多资产组合，并结合过去 30 天窗口与 7 天 ARIMA 预测。

**📈 对比分析**

通过与等权、均值‑方差优化及不同规模的经典 DDPG/DQN 进行交叉验证，量子模型仅需 30–60 参数即可达到或超过 160k 参数的经典模型，Sharpe 比例提升至 0.4‑0.8。

**⚠️ 局限性**

局限性在于训练受限于状态向量模拟器、部署受限于云端 QPU 的排队与初始化延迟，以及量子模型仍需更大规模硬件验证以证明真正的量子优势。

---

## 155. Agentic Digital Twins: A Taxonomy of Capabilities for Understanding Possible Futures

**arXiv ID:** 2601.18799 | [PDF](https://arxiv.org/pdf/2601.18799v1)

**作者:** Christopher Burr `[一作]` (Alan Turing Institute), David Wagg `[通讯]` (University of Sheffield)

**通讯引用:** 7346 | [OpenAlex ID](https://openalex.org/A5064576675)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个以代理性为核心的数字孪生（DT）分类体系，围绕代理位置、耦合紧密度和模型演化三维度构建27种配置，并挑选9个具代表性配置，分为“现在”“阈值”“前沿”三大群组；通过交通导航案例阐释代理性DT的表现力和可能的治理挑战；

**💡 创新点**

创新点在于将代理性与数字孪生结合，首次将代理位置、耦合性质和模型演化纳入统一框架，形成系统化的配置空间；引入“执行性预测”理论解释代理性DT如何通过自我生成数据分布产生表现力与自锁定现象；将配置划分为三大阶段，为未来治理与技术路径提供参考；

**🔧 技术方法**

主要采用概念框架与符号记号（E/T/C × L/T/C × S/A/R）构建分类；使用执行性预测（performative prediction）理论进行理论推导；通过案例分析（交通导航）展开情景说明；

**📊 数据集**

论文以交通导航系统为示例，引用公开地图与交通数据场景，但并未使用具体公开数据集进行实验验证；

**📈 对比分析**

论文未进行实验比较或性能测评；通过理论推导与案例说明展示不同配置对表现力、锁定风险的影响，未给出数值指标；

**⚠️ 局限性**

局限性包括：缺乏实证验证与实验数据；对AI技术能力的假设过于乐观；治理与伦理讨论仍停留在理论层面；模型演化与耦合层级之间可能存在交叉依赖，框架简化可能掩盖实际复杂性。

---

## 156. IPBC: An Interactive Projection-Based Framework for Human-in-the-Loop Semi-Supervised Clustering of High-Dimensional Data

**arXiv ID:** 2601.18828 | [PDF](https://arxiv.org/pdf/2601.18828v1)

**作者:** Mohammad Zare `[一作]` `[通讯]` (Arioobarzan Engineering Team), Mohammad Zare (Arioobarzan Engineering Team)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种交互式投影‑基于的半监督聚类框架（IPBC），通过用户提供的 must‑link / cannot‑link 约束动态优化低维投影，从而实现高维数据的可视化与聚类。

**💡 创新点**

创新点在于将交互式用户反馈直接嵌入投影损失函数，形成投影与聚类的闭环；同时提供基于原始特征的可解释聚类结果，打破传统单向预处理与聚类的僵化流程。

**🔧 技术方法**

主要技术包括：UMAP 非线性降维、基于约束的损失增强（must‑link 与 cannot‑link），SGD 动态重优化，DBSCAN 低维聚类，以及决策树/逻辑回归的后置解释。

**📊 数据集**

实验使用了 MNIST、Fashion‑MNIST、以及 10x Genomics PBMC 单细胞 RNA‑seq 数据集，均包含已知标签以评估聚类质量。

**📈 对比分析**

与 K‑Means（原始/PCA 降维）、静态 UMAP+DBSCAN 等基线相比，IPBC 在 ARI/NMI、Silhouette 等外部/内部指标上均显著提升（如 MNIST ARI 从 0.60 提升至 0.80，NMI 0.70→0.85）。

**⚠️ 局限性**

局限性包括：大规模数据时投影重优化耗时、约束权重与 margin 参数对收敛影响敏感、目前仅在模拟用户场景验证，缺乏真实用户可用性与体验评估。

---

## 157. EVEREST: An Evidential, Tail-Aware Transformer for Rare-Event Time-Series Forecasting

**arXiv ID:** 2601.19022 | [PDF](https://arxiv.org/pdf/2601.19022v1)

**作者:** Antanas Zilinskas `[一作]`, Jakub Marecek `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为 EVEREST 的轻量级 Transformer，用于多变量时间序列的罕见事件预测，能够在训练时通过注意力瓶颈、证据性 NIG 头、EVT GPD 头和前驱辅助头进行联合正则化，推理阶段仅保留单一分类头；

**💡 创新点**

创新点在于将罕见事件预测的判别、校准、尾部风险评估和前驱监督统一到同一模型框架中，利用训练专用的多任务头提升概率校准和极值学习，同时保持推理无额外开销；

**🔧 技术方法**

采用了 Transformer 编码器、单查询注意力聚合、Normal–Inverse–Gamma 证据头、Generalized Pareto 超越头以及轻量级前驱头，整体损失为焦点损失、证据 NLL、EVT 似然和前驱交叉熵的加权组合；

**📊 数据集**

主要数据集为 Solarflare 的 SHARP–GOES 太阳风磁矢量时间序列（涵盖 24/48/72 小时的 C、M、M5 阈值事件），并在工业阀门监测数据集 SKAB 上进行跨域迁移验证；

**📈 对比分析**

与 LSTM、3D‑CNN、SolarFlareNet 等基线相比，EVEREST 在所有九个任务中获得最高的 True Skill Statistic（如 C‑24h TSS 0.973，M5‑72h TSS 0.966），并保持优异的校准指标（ECE≈0.016）；

**⚠️ 局限性**

局限性包括固定长度输入窗口、缺乏图像/无线电多模态支持、对长周期先导信号的捕获能力有限以及对极大事件（X 类）样本稀缺导致尾部拟合受限。

---

## 158. How Is Uncertainty Propagated in Knowledge Distillation?

**arXiv ID:** 2601.18909 | [PDF](https://arxiv.org/pdf/2601.18909v1)

**作者:** Ziyao Cui `[一作]` (Duke University), Jian Pei `[通讯]` (Duke University)

**通讯引用:** 81360 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究知识蒸馏过程中不确定性如何传播，并提出通过平均多次教师输出和方差加权来降低学生间不确定性并保留教师的不确定性。

**💡 创新点**

首次将蒸馏视为不确定性转换；提出两种简单的方差感知策略，并给出线性回归下的理论最优性证明；在神经网络和大型语言模型中验证其有效性。

**🔧 技术方法**

统计不确定性分解（教师输出、初始化、学生输出）；线性回归闭式分析；神经网络梯度下降实验；LLM序列级蒸馏与多响应采样；方差加权与平均法；基准对比。

**📊 数据集**

线性回归：Boston Housing；神经网络：Digits、Boston Housing；LLM：BioASQ QA 数据集。

**📈 对比分析**

与传统单响应蒸馏对比：多响应/平均/方差加权能显著降低学生间方差、提升教师与学生的相似度、减少系统噪声和幻觉；在LLM上提升对真实答案的匹配度并降低错误率。

**⚠️ 局限性**

仅在上述模型和数据上验证；对更大规模、多模态或非监督任务的适用性尚未评估；方差估计在高维文本上仍可能欠缺；未针对不同教师模型的稳定性进行深入探讨。

---

## 159. Optimizing Network Topology Efficiency: A Resource-Centric Analysis of Non-Blocking Architectures

**arXiv ID:** 2601.19008 | [PDF](https://arxiv.org/pdf/2601.19008v1)

**作者:** Jia Xu Wei `[一作]` (University of California), Wei Wei `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于资源消耗的网络效率定义，并通过统一成本函数比较不同拓扑在满足无阻塞条件下的硬件资源占用；

**💡 创新点**

将吞吐量乘子（跳数）与路由器复杂度（Radix）结合的成本模型，系统性评估直连与间接拓扑在不同规模下的效率，并指出高 Radix、间接网络在现代技术节点下更优；

**🔧 技术方法**

理论分析、图论建模、成本函数推导、无阻塞约束、均匀流量假设；

**📊 数据集**

无（本文为理论模型，无实测数据集）；

**📈 对比分析**

使用统一的 Cost_host 指标对比各拓扑的资源成本，结果显示：在中等规模时超立方体（Hypercube）最优，规模增大时高 Radix 间接网络（如 Fat Tree）优于直连拓扑；

**⚠️ 局限性**

假设流量均匀、忽略物理布线与延迟、未考虑动态负载及实际实现中的功耗与面积瓶颈，模型仅为理论指导。

---

## 160. MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning

**arXiv ID:** 2601.19204 | [PDF](https://arxiv.org/pdf/2601.19204v1)

**作者:** Zhixi Cai `[一作]` (Monash University), Hamid Rezatofighi `[通讯]` (Monash University)

**通讯引用:** 3098 | [OpenAlex ID](https://openalex.org/A5034608678)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MATA——一种基于可学习超自动机的多代理视觉推理框架，利用共享内存实现透明执行；

**💡 创新点**

创新点在于：①将高层转移函数交给可学习的 LLM 超代理，自动在协作与竞争的专家间切换；②构建转移轨迹数据集用于监督微调；③在保持规则子自动机可靠性的同时实现灵活的高层决策；

**🔧 技术方法**

使用技术包括层次化有限状态机、LLM（Qwen3 4B）作为超代理、规则化子自动机、共享可追加内存、基于轨迹树的监督微调、工具调用（检测、验证、代码执行）等；

**📊 数据集**

实验数据集为 GQA、OK‑VQA、RefCOCO、RefCOCO+、RefCOCOg 以及无训练集的 Ref‑Adv；

**📈 对比分析**

与主流单体 VLM（如 InternVL3、Qwen2.5‑VL）和组合式方法（HYDRA、VisRep、ViperGPT）对比，MATA 在 GQA 64.9%、OK‑VQA 76.5%、RefCOCO 96.3% 等指标上均取得领先；

**⚠️ 局限性**

局限性：轨迹树的近乎穷举搜索在代理数增多时成本急剧上升；目前仅使用 3 个代理，对更大规模多代理扩展需要进一步改进；

---

## 161. Large Language Models for Departmental Expert Review Quality Scores

**arXiv ID:** 2601.18945 | [PDF](https://arxiv.org/pdf/2601.18945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 162. Agentic Business Process Management Systems

**arXiv ID:** 2601.18833 | [PDF](https://arxiv.org/pdf/2601.18833v1)

**作者:** Marlon Dumas `[一作]` (University of Tartu), David Chapela-Campa `[通讯]` (University of Tartu)

**通讯引用:** 158 | [OpenAlex ID](https://openalex.org/A5083543484)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出Agentic业务流程管理系统（A‑BPM）框架，并阐述从过程挖掘到自主流程执行的技术链与治理模型。

**💡 创新点**

创新点在于将过程挖掘与预测、预设与自适应规划结合，构建三维自治光谱和五层体系结构，实现完全自主、可解释、可验证的流程管理。

**🔧 技术方法**

采用过程挖掘（描述、预测、优化）、数字过程孪生、强化学习/规划算法、LLM驱动的对话与协同接口以及API/事件流技术。

**📊 数据集**

论文未给出具体公开数据集，假设使用典型行业事件日志与BPMN/DMN模型进行演示。

**📈 对比分析**

本文为定位性综述与概念设计，未进行实验比较；作者指出预期性能提升基于自适应决策与实时优化，但缺乏量化评估。

**⚠️ 局限性**

局限主要包括：缺乏实现细节与实验验证、对数据集与指标的依赖性未明确、对多主体自治与合规性的进一步机制研究仍待深入。

---

## 163. A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy

**arXiv ID:** 2601.18939 | [PDF](https://arxiv.org/pdf/2601.18939v1)

**作者:** Claire O'Brien `[一作]` (Algoverse), Ryan Lagasse `[通讯]` (Lockheed Martin AI Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型语言模型中的顺从性（sycophancy）行为，本文提出利用稀疏自编码器（SAE）与线性探测器识别最相关的神经元，仅对这3%神经元进行梯度遮蔽微调，以实现精细化对齐。

**💡 创新点**

创新点在于将SAE产生的稀疏特征与线性探测器相结合，通过解码权重定位关键神经元，从而实现仅更新极少量参数而不需大量数据或外部奖励模型。

**🔧 技术方法**

核心技术包括稀疏自编码器（SAE）、线性探测器（probe）、梯度遮蔽（gradient masking）和神经元级微调（NeFT），并结合自定义KL与熵正则化的损失函数。

**📊 数据集**

使用了从 ELI5、AskHistorians、AmbigQA 生成的含顺从与非顺从响应的自标注数据，以及 Syco‑Bench、Open‑Ended‑Sycophancy、NLP、POLI、PHIL 等四个标准基准。

**📈 对比分析**

与未微调模型、合成数据干预、监督定位微调（SPT）以及仅使用残差探测器的对照实验相比，Gemma‑2‑2B 在 Syco‑Bench、Open‑Ended‑Sycophancy、NLP、POLI、PHIL 等指标上均达到或超过现有最佳水平，Gemma‑2‑9B 在多数测试中也表现出显著提升。

**⚠️ 局限性**

局限性包括探测器在不同层/模型间的准确度差异、可能因过度/不足微调导致灾难性遗忘、对多轮对话缺乏覆盖，以及对更大规模或不同结构模型的可迁移性待验证。

---

## 164. HalluJudge: A Reference-Free Hallucination Detection for Context Misalignment in Code Review Automation

**arXiv ID:** 2601.19072 | [PDF](https://arxiv.org/pdf/2601.19072v1)

**作者:** Kla Tantithamthavorn `[一作]` (Monash University), Ming Wu `[通讯]` (Atlassian)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了 HalluJudge，一种无参考的代码评审注释幻觉检测框架。

**💡 创新点**

首次将上下文对齐与多策略推理相结合，提出四种评估策略（直接、少量示例、分步推理、树形思维）。

**🔧 技术方法**

使用大型语言模型（Gemini 3、GPT‑5.1）结合多步、树形推理，基于代码差异与评论的命题关联进行幻觉判断。

**📊 数据集**

在 Atlassian 内部企业项目中采样的 143 条 LLM 生成评审注释（人类标注），以及 557 条带开发者点赞/点踩的真实生产评论。

**📈 对比分析**

对四种策略与两款 LLM 进行精度/召回/F1、成本和与开发者偏好的对齐度评估，树形思维在 Gemini 3 上达到 0.85 F1，成本仅约 0.009 美元/评估。

**⚠️ 局限性**

仅关注上下文对齐的幻觉，未覆盖所有质量问题；数据来自单一企业，结果可能不完全外推。

---

## 165. HELM: A Human-Centered Evaluation Framework for LLM-Powered Recommender Systems

**arXiv ID:** 2601.19197 | [PDF](https://arxiv.org/pdf/2601.19197v1)

**作者:** Sushant Mehta `[一作]` `[通讯]`, Sushant Mehta

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对LLM驱动推荐系统的全方位人性化评估框架HELM，并基于专家评测对三大领域（电影、图书、餐饮）中三种LLM推荐系统进行系统评估。

**💡 创新点**

创新点在于：①将人性化评价拆分为五个维度（意图匹配、解释质量、交互自然性、信任透明度、公平多样性）；②引入专家打分与自动化指标相结合的混合评估方法；③公开发布评估工具包和标注数据，促进可复现研究。

**🔧 技术方法**

使用技术包括：GPT‑4、LLaMA‑3.1‑8B、P5三种LLM推荐模型；传统协同过滤（NCF+模板）与随机推荐做基准；专家评估（5点Likert）与自动化度量（HitRate、NDCG、Gini、覆盖率、ILD、信念一致性等）。

**📊 数据集**

数据集：MovieLens‑1M（电影）、Amazon Books（图书）、Yelp（餐饮）三大公开数据集，并为LLM提供丰富的元数据（标题、标签、摘要等）。

**📈 对比分析**

对比方法：在847个自然场景下让12名领域专家对各系统打分，并计算各维度得分及整体人性化得分。结果显示：GPT‑4在意图匹配、解释质量、交互自然性和信任度上均最高，但公平性（Gini 0.73）最差；传统协同过滤在准确率（HitRate）上略优但人性化得分最低；随机推荐在公平性上最高，但其他维度均最差。

**⚠️ 局限性**

局限性：①专家评测不一定代表普通用户体验；②仅覆盖三大常见领域，未必适用于医疗、教育等垂直行业；③评估仅针对当前LLM版本，未来模型更新可能导致结果变化。

---

## 166. Agree to Disagree: Consensus-Free Flocking under Constraints

**arXiv ID:** 2601.19119 | [PDF](https://arxiv.org/pdf/2601.19119v1)

**作者:** Peter Travis Jardine `[一作]` (Queen's University), Sidney Givigi `[通讯]` (Queen's University)

**通讯引用:** 2121 | [OpenAlex ID](https://openalex.org/A5025867957)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种在多智能体中无需通信、可在不共享相同距离约束的情况下自适应协商交互距离的集群控制方法，并通过仿真验证其可行性。

**💡 创新点**

创新点在于：① 引入局部估计邻居期望间距的滤波器，实现无通信协商；② 设计自定义 bump 函数与约束集体势能函数，保证各智能体在自身可接受范围内协商距离；③ 通过结合聚合、对齐与导航的势能函数实现对半信任场景下的稳定集群。

**🔧 技术方法**

采用梯度下降势能控制法，基于双积分动力学模型；使用聚合、对齐、导航势能、pinning 控制以及自定义 bump 函数与滤波器；所有控制均基于局部邻居信息实现。

**📊 数据集**

使用模拟数据：30 只具有统一间距参数的智能体以及 7 只具有不同初始间距与约束范围的智能体，模拟场景为二维平面上双积分动态系统。

**📈 对比分析**

通过对比同质集群与异质集群的连通性、平均距离误差以及约束违例数量，结果表明：在异质场景下，系统能在约 60 秒内实现连通且所有间距均落入各自允许范围；相较于传统固定间距方法，误差收敛更快，约束违例显著下降。

**⚠️ 局限性**

局限性包括：仅在仿真验证，未在真实硬件或更高维空间中测试；滤波器与 bump 函数需要经验参数调节；当智能体数量急剧增加或拓扑变化频繁时，收敛速度与稳定性仍需进一步评估。

---

## 167. Grand Challenges around Designing Computers' Control Over Our Bodies

**arXiv ID:** 2601.19143 | [PDF](https://arxiv.org/pdf/2601.19143v1)

**作者:** Florian 'Floyd' Mueller `[一作]` (Monash University), Don Samitha Elvitigala `[通讯]` (Monash University)

**通讯引用:** 453 | [OpenAlex ID](https://openalex.org/A5027978461)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过为期五天的专家研讨会，系统梳理并提出了计算机对人体控制的十大 grand challenges，涵盖技术、设计、伦理与用户四大维度。

**💡 创新点**

首次将技术实现、体验设计与伦理责任三者统一视角，构建了一套跨学科的挑战框架，为后续研究提供明确的议程。

**🔧 技术方法**

采用多学科专家工作坊与文献综述方法，未使用具体硬件或算法实现。

**📊 数据集**

无数据集，仅基于专家经验与已有文献。

**📈 对比分析**

本研究为概念性挑战清单，未进行实验对比或性能评估。

**⚠️ 局限性**

参与者规模有限且多为熟识专家，缺乏跨文化与非主流身体差异群体；研讨会方法主观性强，缺乏客观验证和长期体验研究。

---

## 168. Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction

**arXiv ID:** 2601.19142 | [PDF](https://arxiv.org/pdf/2601.19142v1)

**作者:** Zhicheng Zhang `[一作]` (Shenzhen International Graduate School, Tsinghua University), Zhenhua Dong `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种长度自适应兴趣网络（LAIN），通过显式利用用户行为序列长度来平衡短序列与长序列的推荐质量。

**💡 创新点**

创新点在于将序列长度作为条件信号，设计了谱长度编码器、长度条件提示与长度调制注意力三模块，使模型能够根据序列长短自适应地调整注意力分布和表示学习。

**🔧 技术方法**

采用了频谱编码的长度嵌入、软提示注入和温度调制的 Transformer 注意力机制，构成轻量级的 plug‑and‑play 框架。

**📊 数据集**

在 KuaiVideo、MicroVideo1.7M 与 EBNeRD‑small 三大真实业务数据集上进行实验，覆盖多种长序列 CTR 基线模型。

**📈 对比分析**

相较于原始模型，LAIN 在 AUC、GAUC 及 LogLoss 上均实现了 0.4%–1.2% 的提升，尤其显著提升短序列用户（<100 次交互）AUC 上提升约 1.08%，表明能有效缓解长度不平衡导致的性能偏差。

**⚠️ 局限性**

局限性包括：仍需在更大规模、多任务和跨领域数据上验证泛化能力，且对极端长序列（>1000）下的效率与效果尚未系统评估。

---

## 169. Evaluating Nova 2.0 Lite model under Amazon's Frontier Model Safety Framework

**arXiv ID:** 2601.19134 | [PDF](https://arxiv.org/pdf/2601.19134v1)

**作者:** Satyapriya Krishna `[一作]` (Amazon), Spyros Matsoukas `[通讯]` (Amazon)

**通讯引用:** 2489 | [OpenAlex ID](https://openalex.org/A5034173999)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文对 Amazon Frontier Model 进行跨域安全评估，检验其在化学/生物/核武器扩散、网络攻击和自动 AI 研发三大高危领域是否突破关键阈值，结论为模型安全可公开发布。

**💡 创新点**

创新点在于提出一种结合可重复自动基准、专家红队、uplift 研究与多智能体压力测试的多方法评估框架，并对三大风险领域给出统一阈值判定。

**🔧 技术方法**

评估方法使用自动化基准测试、红队人工评审、动态内容过滤、持续监测以及多模型推理（low/medium/high），并在内部部署了自定义代理和云端隔离环境进行测试。

**📊 数据集**

所用数据集包括 WMDP‑Bio、WMDP‑Chem、ProtocolQA、BioLP‑Bench、VCT、化学结构识别集、CyberMetric、SECURE‑CWET、CTIBench、CyBench、RE‑Bench 等，覆盖知识回想、实验流程、网络攻防、ML 自动化研发等。

**📈 对比分析**

与前一代模型相比，Nova 在 CBRN 相关基准上实现 0.71–0.82 的准确率提升，CyberBench 约提升 7.5%，但在高难度 CTF 与 AI R&D 任务上未能突破阈值，整体保持在安全范围内。

**⚠️ 局限性**

局限性包括模型在复杂网络攻击或实际漏洞利用中仍需人工引导、对高级逆向/暴力破解能力有限、自动 AI 研发实验仅限于基础代码改造，无法真正实现全流程自主研发，因而无法满足极端风险阈值。

---

## 170. CoReTab: Improving Multimodal Table Understanding with Code-driven Reasoning

**arXiv ID:** 2601.19193 | [PDF](https://arxiv.org/pdf/2601.19193v1)

**作者:** Van-Quang Nguyen `[一作]` (RIKEN), Takayuki Okatani `[通讯]` (Tohoku University)

**通讯引用:** 3549 | [OpenAlex ID](https://openalex.org/A5009259465)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种代码驱动的推理框架，利用LLM生成带有多步骤自然语言推理和可执行Python代码的标注，并通过代码执行自动验证，从而生成可解释、可验证的多模态表格理解训练数据。

**💡 创新点**

创新点在于将可执行代码与推理文本相结合，实现推理过程的可验证性与可解释性，并通过自动化验证大规模生成高质量标注，显著提升模型的推理透明度与准确度。

**🔧 技术方法**

技术主要包括：使用Qwen3 32B-3A等大型LLM进行注释生成；Python代码执行与对比检验；三阶段LoRA微调（表格识别预训练、指令调优、强化学习优化）；以及代码执行工具和提示工程。

**📊 数据集**

数据集为115K条已验证样本，涵盖11个表格多模态任务（TQA、TFV、TSU），平均长度约529 tokens，基于MMTab等公开数据集构建。

**📈 对比分析**

与MMTab、SynTab、Table-LLaVA、GPT‑4V等基线对比，模型在17个MMTab基准上分别提升+6.2%（TQA）、+5.7%（TFV）和+25.6%（TSU），并在各项任务中显著优于开源与闭源基线。

**⚠️ 局限性**

限制包括：部分推理文本仍可能存在错误；推理+代码执行导致推理耗时增加；框架主要适用于可验证的确定性答案，无法覆盖开放式文本生成任务；对表格旋转、遮挡等复杂视觉变形的鲁棒性依赖于表格识别模型。

---

## 171. Information-Theoretic Secure Aggregation over Regular Graphs

**arXiv ID:** 2601.19183 | [PDF](https://arxiv.org/pdf/2601.19183v1)

**作者:** Xiang Zhang `[一作]` (Technical University of Berlin), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并分析了在任意正规图上实现信息论安全聚合（TSA）的框架，给出了最优的通信率、密钥率与源密钥率三元组；

**💡 创新点**

创新点在于将TSA可实现性转化为图的对角调制邻接矩阵（DMAM）核空间维数的条件，并给出一阶线性设计实现该条件；

**🔧 技术方法**

使用线性代数与图谱分析（DMAM核空间、MDS生成矩阵）以及信息论极限（互信息、熵、对偶定理）构建安全协议与证明；

**📊 数据集**

无实验数据集，研究完全基于理论推导与证明；

**📈 对比分析**

通过与已知的中心化或全连接网络的SA结果对比，证明所给出的率区间是最优的；

**⚠️ 局限性**

仅适用于 d‑regular 图，对非正规或非 d‑regular 网络的可行性与最优率仍未知。

---

## 172. Learning Ordered Representations in Latent Space for Intrinsic Dimension Estimation via Principal Component Autoencoder

**arXiv ID:** 2601.19179 | [PDF](https://arxiv.org/pdf/2601.19179v1)

**作者:** Qipeng Zhan `[一作]` (University of Pennsylvania), Li Shen `[通讯]` (University of Pennsylvania)

**通讯引用:** 31923 | [OpenAlex ID](https://openalex.org/A5100333320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的非线性自编码器PCAE，能够像PCA一样实现特征方差排序并估计数据的本质维度

**💡 创新点**

在非线性映射中加入了加权方差正则化与等距约束，首次实现了可解释的方差有序潜在空间，并通过动态系数自适应地保证梯度有效

**🔧 技术方法**

利用加权方差损失、等距约束（等距正则化）、动态γ系数调整以及传统的重构损失，并通过梯度下降训练深度网络

**📊 数据集**

在合成数据集dSprites与3DShapes上验证了本质维度恢复；在真实图像数据MNIST与CelebA上评估了维度估计、插值平滑性、FID和下游分类性能

**📈 对比分析**

与PCA-AE、HAE、ARD-VAE和IRMAE对比，PCAE在维度恢复精度、训练速度、插值平滑度和分类误差等指标上均表现优异（维度误差几乎为零、训练时间显著更短、FID与误差更低）

**⚠️ 局限性**

需要预先构建并存储几何距离矩阵（计算量大），对γ系数与正则化权重的设置仍需经验调参，且在极大数据集或复杂几何结构下的可扩展性和稳定性待进一步验证

---

## 173. A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction

**arXiv ID:** 2601.19175 | [PDF](https://arxiv.org/pdf/2601.19175v1)

**作者:** Jinkyu Sung `[一作]` (Seoul National University), Joonseok Lee `[通讯]` (Seoul National University)

**通讯引用:** 5417 | [OpenAlex ID](https://openalex.org/A5067433666)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 CopulaLSP 框架，利用 Gaussian Copula 与边嵌入 Gramian 直接建模边相关性，以提高链接符号预测的效率和精度。

**💡 创新点**

创新点在于将边相关矩阵表示为 Gramian 结构并采用 Woodbury 矩阵恒等式在推断阶段显著降低计算成本，同时理论证明了线性收敛性。

**🔧 技术方法**

技术方法包括 Gaussian Copula、Gramian 相关矩阵、Woodbury 变形、SGNN 基础编码器（如 SNEA）、标签平滑等。

**📊 数据集**

实验使用 BitcoinAlpha、BitcoinOTC、WikiElec、WikiRfa、SlashDot、Epinions 等真实网络数据集。

**📈 对比分析**

与 GCN、SGCN、SNEA、SDGNN、TrustSGCN、SLGNN、SE‑SGformer、SGAAE 等先进方法比较，CopulaLSP 在训练和推断速度上显著更快，并在 AUC/F1 指标上保持竞争力。

**⚠️ 局限性**

局限性包括仅针对静态图，未考虑动态图或异构图，以及在极大规模图上仍有一定的内存开销。

---

## 174. Bridging Gulfs in UI Generation through Semantic Guidance

**arXiv ID:** 2601.19171 | [PDF](https://arxiv.org/pdf/2601.19171v1)

**作者:** Seokhyeon Park `[一作]` (Seoul National University), Jinwook Seo `[通讯]` (Seoul National University)

**通讯引用:** 4274 | [OpenAlex ID](https://openalex.org/A5012388103)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于语义中间层的 UI 生成系统，旨在桥接用户意图表达（执行鸿沟）与 AI 输出解释（评估鸿沟）之间的差距。

**💡 创新点**

创新点在于：①从主流 UI 生成服务中系统提炼出的四层语义框架（Product→Design System→Feature→Component）；②将该框架实现为可视化的结构化输入、双向语义映射与关系分析，支持可追踪、可局部修改的迭代；③通过关系图直观呈现语义冲突、匹配与缺失，提升透明度与可控性。

**🔧 技术方法**

技术主要包括：自然语言到语义槽的 LLM 解析、基于 Vercel v0 的 React 组件生成、对生成代码和截图的多模态语义抽取（如布局、配色、交互），以及利用关系图算法对语义之间的上下层及横向依赖进行可视化。

**📊 数据集**

使用的数据集是对六大主流 UI 生成服务（Vercel v0、Google Stitch、Figma Make、Uizard、Lovable、Relume）的提示准则与示例共 907 条片段进行主题分析，形成语义词表；实验中使用的 UI 生成模型为 Vercel 的公开 API。

**📈 对比分析**

在 14 名设计/开发/PM 参与者的对照实验中，semantic 系统相较于基线 chat 接口在意图表达、输出解释、修改易用性、控制感和目标匹配等 7 点量表均获得显著提升（Wilcoxon 检验 p<0.01，效应量 r>0.5），并在定性访谈中获得用户对迭代可预测性的正向反馈。

**⚠️ 局限性**

局限性包括：①语义词表和关系需要学习，初期学习成本高；②长文本意图被压缩为槽位可能丢失细微语义；③系统仅支持单一 React 组件，难以扩展至大型 UI 体系；④生成模型仍可能出现语法错误或无关改动（“语义漂移”），需更稳健的局部编辑机制；④缺乏跨团队和长期使用的实证验证。

---

## 175. Price of Locality in Permutation Mastermind: Are TikTok influencers Chaotic Enough?

**arXiv ID:** 2601.19161 | [PDF](https://arxiv.org/pdf/2601.19161v1)

**作者:** Bernardo Subercaseaux `[一作]` (Carnegie Mellon University), Bernardo Subercaseaux `[通讯]` (Carnegie Mellon University)

**通讯引用:** 533 | [OpenAlex ID](https://openalex.org/A5030108071)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究并分析Permutation Mastermind游戏中“局部”策略（相邻猜测差异有限）所需的最少猜测次数，并证明在此约束下的解法比非局部策略慢得多；进一步证明在ℓ₃局部策略下的判定问题为NP‑难，而在ℓ₂局部策略下可在随机多项式时间内求解。

**💡 创新点**

首次将Permutation Mastermind视为在Cayley图上的约束搜索，提出ℓ_k‑局部和w_k‑局部两类新的局部性定义；证明局部性导致猜测次数从线性提升到二次，且在ℓ₃局部下可构造多项式时间可归约到SAT，从而得到NP‑难性；在ℓ₂局部下通过匹配与模2约束的组合实现随机多项式算法。

**🔧 技术方法**

利用图论（唯一完美匹配的边数上界）、极值图理论（K₂,₂‑无关图的边上界）、Cayley图直径计算、可满足性归约（从3‑SAT的单真子句版本）以及随机化匹配求解（Geelen‑Kapadia算法）等技术。

**📊 数据集**

本文未使用实际数据集，而是在理论模型中对任意大小的n构造猜测序列和反馈，进行多项式时间的理论证明与归约。

**📈 对比分析**

对比非局部策略（O(n·n)猜测）与局部策略的上界与下界：在ℓ_k‑局部下最坏情况需要Θ(n²)猜测；在ℓ₂局部下提供随机化多项式解法，表现优于一般NP‑难情形；w_k‑局部的具体性能仍为开放问题。

**⚠️ 局限性**

局部性假设虽然贴近人类直观策略，但理论上导致显著更高的猜测复杂度；w_k‑局部的性能界限尚未充分确定，且归约构造依赖于特定的块结构，可能不适用于所有实例；实际实现与实验验证仍待研究。

---

## 176. QA-ReID: Quality-Aware Query-Adaptive Convolution Leveraging Fused Global and Structural Cues for Clothes-Changing ReID

**arXiv ID:** 2601.19133 | [PDF](https://arxiv.org/pdf/2601.19133v1)

**作者:** Yuxiang Wang `[一作]`, Gaozhe Jiang `[通讯]` (National University of Singapore)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5113344289)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种名为 QA-ReID 的双分支框架，利用 RGB 图像和人像分割得到的衣物无关特征，并通过多模态注意力融合与 QAConv‑QA 的质量感知像素级匹配实现衣物变更的人体重识别。

**💡 创新点**

创新点在于（1）将 RGB 与解析特征通过可学习的多模态注意力融合，动态平衡两者在通道与空间维度上的重要性；（2）在 QAConv 基础上引入像素级质量权重和双向一致性约束的 QAConv‑QA 模块，显著提升对衣物变化干扰的鲁棒性。

**🔧 技术方法**

核心技术包括 ResNet‑50 骨干网络、人体分割网络、通道/空间注意力机制、质量感知像素权重、双向最大池化、交叉熵/三元组/二元匹配损失以及多任务联合优化。

**📊 数据集**

实验使用了三个衣物变更人重识别基准：PRCC、LTCC 和 VC‑Clothes，涵盖真实与合成场景、长时变异以及多衣物样本。

**📈 对比分析**

与现有方法对比，QA‑ReID 在 PRCC、LTCC 与 VC‑Clothes 的跨衣物设置下分别取得 64.1%/61.2%、42.9%/21.3% 与 86.3%/86.1% 的 Top‑1/mAP，显著超越前沿方法，成为当前 SOTA。

**⚠️ 局限性**

局限性主要体现在对高质量人体分割的依赖、模型参数量大导致计算开销较高，以及缺乏轻量化设计和实时部署方案。

---

## 177. Bridging Visual and Wireless Sensing: A Unified Radiation Field for 3D Radio Map Construction

**arXiv ID:** 2601.19216 | [PDF](https://arxiv.org/pdf/2601.19216v1)

**作者:** Chaozheng Wen `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 82598 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了统一的可视化与无线辐射场框架 URF‑GS，利用 3D Gaussian splatting 与逆渲染技术，结合视觉与无线感知数据，构建高精度、可泛化的 3D 无线电地图。

**💡 创新点**

创新点在于：① 将光学与无线测量融合为共享的 Gaussian 原语，实现几何与材质的统一学习；② 通过物理感知逆渲染，将 PBR 与自由空间衰减模型嵌入 3D 场景，显著提升不同 Tx‑Rx 配置下的泛化与样本效率；③ 在低样本（few‑shot、zero‑shot）场景下实现 10 倍的样本效率提升。

**🔧 技术方法**

使用技术包括 3D Gaussian splatting、深度/法向先验、可微路径追踪、PBR 反射/散射模型、FSPL 传播损耗、物理感知逆渲染与联合优化。

**📊 数据集**

使用的数据集有：NIST 60 GHz 室内实验数据集、公开的 Bistro、Wi3room 以及其它基准无线与视觉测量数据。

**📈 对比分析**

与 NeRF2、RF‑3DGS、WRF‑GS+ 等基线对比，URF‑GS 在 PSNR、SSIM、LPIPS 上均取得领先（SSIM 提升 24.7%，样本效率提升 10 倍），在 Few‑Shot、Zero‑Shot 场景下同样表现最佳。

**⚠️ 局限性**

局限性包括：仅处理静态场景；对动态人、物体的建模缺失；场景迁移泛化能力有限；需要进一步研究大规模预训练与迁移学习以适应多变环境。

---

## 178. A Hybrid Supervised-LLM Pipeline for Actionable Suggestion Mining in Unstructured Customer Reviews

**arXiv ID:** 2601.19214 | [PDF](https://arxiv.org/pdf/2601.19214v1)

**作者:** Aakash Trivedi `[一作]` (Birla Institute of Technology and Science), Praveen Kumar `[通讯]` (Birdeye Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种混合管线，将高召回RoBERTa分类器与指令微调量化LLM结合，自动从酒店和餐饮评论中抽取、归类、聚类并摘要可操作的建议；

**💡 创新点**

创新点在于：①使用精度-召回双目标训练提升分类召回；②在分类后利用LLM进行精确提取、重写与聚类，减少LLM的幻觉；③通过量化Gemma‑3实现本地部署；

**🔧 技术方法**

核心技术包括RoBERTa分类器、精度–召回替代损失、Ollama Gemma‑3 27B量化模型、指令式few‑shot提示、聚类与摘要流程；

**📊 数据集**

使用约1110条酒店/餐饮评论（13–18%可操作建议）以及跨行业（房地产、医疗、金融、汽车）数据验证通用性；

**📈 对比分析**

与词法基线、规则基线、Prompt‑only LLM、规则‑基于提取等基线对比，分类器召回率0.922，精度0.904；整个管线在抽取语义精度（BERTScore 0.92）和聚类一致性（AMI 0.67）上明显优于单一方法；

**⚠️ 局限性**

局限性：数据为专有，难以公开复现；跨域时精度下降；LLM聚类偶尔误分主题；并且对行业特定术语的适应性需进一步改进；

---

## 179. Before Smelling the Video: A Two-Stage Pipeline for Interpretable Video-to-Scent Plans

**arXiv ID:** 2601.19203 | [PDF](https://arxiv.org/pdf/2601.19203v1)

**作者:** Kaicheng Wang `[一作]` (University of Washington), Denise Wilson `[通讯]` (University of Washington)

**通讯引用:** 4453 | [OpenAlex ID](https://openalex.org/A5064788370)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两阶段视频到嗅觉规划管线，先用视觉‑语言模型提取视频语义，再用大型语言模型生成结构化的嗅觉计划；并通过在线问卷评估这些计划的可理解性与可行性。

**💡 创新点**

创新点在于将视觉‑语言模型与大型语言模型耦合，形成可解释、时间同步的嗅觉规划，而非手工脚本；并首次在无物理嗅觉输出的前提下，评估规划对用户的可接受度。

**🔧 技术方法**

使用 Gemini‑3 Pro（VLM）提取视觉语义，GPT‑5.2（LLM）生成嗅觉计划，配合固定的香气词汇表进行语义‑嗅觉映射。

**📊 数据集**

使用10段短视频（视觉丰富、可嗅感情境）与三种方案（系统生成、过度包含、朴素映射）生成的嗅觉计划文本。

**📈 对比分析**

通过两项在线调查：①排名比较（14名受试者）验证系统方案优于两种基线；②想象体验评价（8名受试者）显示系统方案在沉浸感、连贯性和干扰度上均优于基线；统计方法为 Friedman、Wilcoxon 检验，p<0.001。

**⚠️ 局限性**

局限性包括：未实现物理嗅觉输出，缺乏真实嗅觉体验；数据集规模有限且情境单一；缺乏对多模态交互与用户控制机制的探索；未评估在更大规模、多样化视频上的泛化能力。

---

## 180. SHIELD: An Auto-Healing Agentic Defense Framework for LLM Resource Exhaustion Attacks

**arXiv ID:** 2601.19174 | [PDF](https://arxiv.org/pdf/2601.19174v1)

**作者:** Nirhoshan Sivaroopan `[一作]` (University of Sydney), Wangli Yang `[通讯]` (University of Wollongong)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5047118417)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于多智能体的自愈防御框架，用三阶段防御代理检测并阻止LLM的海绵攻击，且在检测失败时自动更新知识库并优化提示；

**💡 创新点**

创新点在于整合语义相似度检索、子串匹配与LLM推理的三阶段流水线，并通过知识更新代理和提示优化代理实现零训练的持续自愈；

**🔧 技术方法**

使用文本嵌入模型进行语义检索、KMP算法实现子串匹配、黑盒LLM作为推理引擎，并通过进化式提示搜索进行提示优化；

**📊 数据集**

评估数据集包括四类海绵攻击（RL-GOAL、GCG-DoS、EOGen、AutoDoS）与多源公共任务数据构成的正样本，目标模型为LLaMA2等多种LLM；

**📈 对比分析**

与基准方法（perplexity‑filter、harm‑filter、SHIELD、Auto‑DOS等）对比，提出的系统在多种攻击类型下均获得最高F1分数，提升幅度达3–14%，且显著降低了LLM调用延迟；

**⚠️ 局限性**

主要局限包括需要手动设置语义阈值、知识库增长导致检索延迟、防御LLM自身可能被攻击、仅覆盖提示级攻击而非模型污染等。

---

## 181. Multi-Agent Procedural Graph Extraction with Structural and Logical Refinement

**arXiv ID:** 2601.19170 | [PDF](https://arxiv.org/pdf/2601.19170v1)

**作者:** Wangyang Ying `[一作]` (Arizona State University), Haifeng Chen `[通讯]` (NEC Labs America)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多智能体框架，通过迭代结构和逻辑反馈来自动提取自然语言描述的程序化图

**💡 创新点**

将图构建、结构仿真和语义一致性反馈拆分为独立代理，并通过可解释的自然语言提示实现多轮修正

**🔧 技术方法**

采用大语言模型LLM、仿真器、语义检索Agent以及反馈优先级排序机制

**📊 数据集**

在PAGED基准（3,394篇文档）上进行评测

**📈 对比分析**

与MT‑BPMN、BRP、BPMN‑Gen、PET、CIS、Self‑Refine、Actor‑Critic等基线对比，在多项指标上取得最高F1，尤其在网关与约束流上显著提升

**⚠️ 局限性**

仅在结构化文本上验证，计算成本较高，仿真器和语义Agent依赖规则和预训练模型，缺乏对非正式或域特定文本的泛化能力

---

## 182. Enabling SLO-Aware 5G Multi-Access Edge Computing with SMEC

**arXiv ID:** 2601.19162 | [PDF](https://arxiv.org/pdf/2601.19162v1)

**作者:** Xiao Zhang `[一作]` (University of Texas at Austin), Daehyeok Kim `[通讯]` (University of Texas at Austin)

**通讯引用:** 862 | [OpenAlex ID](https://openalex.org/A5022893621)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个可在5G MEC环境下运行的SLO感知资源管理框架（SMEC），通过在RAN MAC层识别请求边界、在边缘服务器使用轻量探测估计网络时延以及利用API收集处理时延，实现完全解耦的请求优先调度。

**💡 创新点**

创新点包括：①无需RAN与边缘服务器协同即可获得请求开始时间；②利用5G上行/下行控制信号和轻量探测实现网络时延估计；③通过应用生命周期事件API进行处理时延预测；④在RAN和边缘独立采用基于剩余时间预算的优先级调度，兼顾多种SLO需求。

**🔧 技术方法**

主要技术手段有：5G BSR（Buffer Status Report）模式识别请求；基于ACK的下行时延探测与补偿；服务器端API收集请求到达、处理开始/结束事件；CPU核心绑定与GPU流优先级调度（NVIDIA MPS+CUDA Stream优先级）；早期丢弃过期请求；srsRAN MAC层插件实现；Open5GS核心网络与Linux调度器。

**📊 数据集**

使用的评估数据集包括：AdaPool（4K 60fps视频，用于Smart Stadium转码）；MOT（1080p 30fps视频，用于AR目标检测）；ICME-VSR（320p 30fps视频，用于视频会议超分辨率）；以及无SLO的文件传输伪造数据；实验平台为基于srsRAN+Open5GS的私有5G MEC试验床，配备Intel Xeon服务器和NVIDIA L4 GPU。

**📈 对比分析**

与传统PF调度、默认Linux/GPU调度、以及PARTIES等SLO感知方案对比；在静态和动态工作负载下，SMEC的SLO满足率从90%–96%显著高于<6%（基准）；P99尾迟延降低幅度高达122倍；在保持BE流公平的同时，边缘CPU/GPU调度更具时延预测能力，整体性能明显提升。

**⚠️ 局限性**

局限性：①由于BSR更新间隔，RAN侧只能在单次BSR增加时捕获请求，导致对高速请求的聚合处理；②处理时延预测基于历史中位数，无法完全适应动态、变异性高的工作负载（如自适应VR渲染），可能导致误判；③下行时延探测假设下行稳定，极端拥塞时误差增大；④需要在RAN与边缘均部署额外模块，可能对现有运营商架构产生部署压力。

---

## 183. KUBEDIRECT: Unleashing the Full Power of the Cluster Manager for Serverless Computing

**arXiv ID:** 2601.19160 | [PDF](https://arxiv.org/pdf/2601.19160v1)

**作者:** Sheng Qi `[一作]` (Peking University), Xin Jin `[通讯]` (Peking University)

**通讯引用:** 72681 | [OpenAlex ID](https://openalex.org/A5083114703)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

在 Kubernetes 的 FaaS 平台上实现了一个可直接插拔的集群管理器，通过绕过 API 服务器实现控制器间的直接消息传递，从而显著提升函数实例的扩缩容性能。

**💡 创新点**

创新点在于：① 发现并利用 FaaS 平台共享的“窄腰”控制器链；② 引入动态物化的最小消息格式与硬/软失效同步协议；③ 将窄腰建模为层次写回缓存，结合 Tombstone 机制实现安全终止；④ 通过最小化代码改动（约 150 行）保持与 Kubernetes 生态的兼容性。

**🔧 技术方法**

技术手段包括：直接 TCP 双向链路实现消息传递；对 API 对象进行动态与静态属性拆分；硬失效握手协议实现快速恢复；软失效（增量同步）和 Tombstone 复制实现一致性；使用 TLA+ 进行协议正确性验证。

**📊 数据集**

实验使用 Azure Functions 追踪（500 个函数、168K 次调用）的真实工作负载，并在 80 节点 CloudLab 集群上跑 microbenchmarks 与 end‑to‑end FaaS 测试。

**📈 对比分析**

与原生 Kubernetes、Knative 和 Dirigent 对比：在扩缩容（N、K、M 维度）中 7.4–59.8 倍加速；在 Knative 上，平均请求延迟下降 26.7×、冷启动数下降 67%；在 Dirigent 的实现中几乎达到相同性能，且保持了 Kubernetes 的 API 兼容性。

**⚠️ 局限性**

局限性包括：① 仍需对 Scheduler 与 Kubelet 做细粒度改动，难以完全零停机升级；② 对水平扩展支持不足，无法轻易拆分为多子集群；③ 需要自定义的握手协议与消息格式，增加系统的复杂度；④ 依赖于控制器链的顺序结构，对非顺序或多链控制器不适用。

---

## 184. TS-Debate: Multimodal Collaborative Debate for Zero-Shot Time Series Reasoning

**arXiv ID:** 2601.19151 | [PDF](https://arxiv.org/pdf/2601.19151v1)

**作者:** Patara Trirat `[一作]` (DeepAuto.ai), Sung Ju Hwang `[通讯]` (DeepAuto.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种零样本时间序列推理框架，利用多模态专门化代理通过协同辩论生成可验证的证据并合成答案。

**💡 创新点**

核心创新在于把文本、视觉和数值三个模态拆分为独立专家，并设计了验证‑冲突‑校准（VCC）协议，实现程序化检查与跨模态冲突解决。

**🔧 技术方法**

技术实现基于大型语言模型和多模态LLM、时间频域预处理、可视化图表、数值工具查询和轻量级代码执行，形成协同辩论流程。

**📊 数据集**

在三个公开基准上评估：MTBench（金融/天气多模态任务）、TimerBed（传感器分类）和TSQA（多任务QA），共20个子任务。

**📈 对比分析**

与单模态零样本、链式推理、多模态辩论、VL‑Time和ByMyEyes等基线相比，平均提升约25%准确率、36% MAE 降低、QA 准确率提升15–29%，表现显著。

**⚠️ 局限性**

局限性包括在长时序因果推理或噪声多维序列上表现下降、领域知识获取仅局部、工具约束导致验证范围有限。

---

## 185. AgenticSCR: An Autonomous Agentic Secure Code Review for Immature Vulnerabilities Detection

**arXiv ID:** 2601.19138 | [PDF](https://arxiv.org/pdf/2601.19138v1)

**作者:** Wachiraphan Charoenwet `[一作]` (University of Melbourne), Ming Wu `[通讯]` (Atlassian)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于Agentic AI的预提交安全代码审查框架AgenticSCR，用于检测不成熟漏洞。

**💡 创新点**

创新点在于将LLM与自主决策、工具调用、仓库导航相结合，并引入SAST规则和CWE树的安全聚焦语义记忆，实现更精准的漏洞定位与验证。

**🔧 技术方法**

采用Agentic架构、Claude 4.5 LLM推理、Git/CodeQL等工具调用、SAST规则/ CWEB树语义记忆、工作记忆与情节记忆等技术。

**📊 数据集**

使用自建SCRBench数据集，包含144个预提交变更、107个CVE、92个GitHub仓库，涵盖Python、JavaScript、TypeScript及33类CWE。

**📈 对比分析**

通过与零拷贝LLM、CodeQL、Semgrep、Snyk四种基线比较，采用定位准确率、相关性、类型正确率和整体正确率等指标，AgenticSCR整体正确率达17.5%，比基线高10–14%，误报率降低2–5倍。

**⚠️ 局限性**

局限性：实验仅在SCRBench上验证，可能不适用于其他语言/项目；相关性评估依赖LLM判断，可能带来偏差；对信息泄露、控制流等高级CWE表现仍差。

---

## 186. In-Network Collective Operations: Game Changer or Challenge for AI Workloads?

**arXiv ID:** 2601.19132 | [PDF](https://arxiv.org/pdf/2601.19132v1)

**作者:** Torsten Hoefler `[一作]` (ETH Zürich), Amirreza Rastegari `[通讯]` (Microsoft)

**通讯引用:** 280 | [OpenAlex ID](https://openalex.org/A5045584135)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了网络内集体操作（INC）在AI工作负载中的机遇与挑战，区分了Edge-INC和Core-INC两种实现方式；

**💡 创新点**

提出了INC在数据并行、管道并行、张量并行等并行模式下的适用场景，系统分析了六大技术障碍并预测了未来发展路径；

**🔧 技术方法**

基于MPI/CCL通信模型，结合Portals 4、sPIN、NVIDIA SHARP等技术实现Edge-INC与Core-INC的协同工作；

**📊 数据集**

无具体数据集，本文属于综述与理论分析性质；

**📈 对比分析**

通过理论模型与案例分析显示，INC可将Allreduce等操作时延降低约60%，但受Amdahl定律限制，整体加速最大约34%，实际可达约11%；

**⚠️ 局限性**

受到低精度数据类型、向量/块浮点、稀疏计算、精度一致性、接口协同、加密认证等多重技术难题的制约，导致实施门槛高、普及速度慢。

---

## 187. Understanding Dominant Themes in Reviewing Agentic AI-authored Code

**arXiv ID:** 2601.19287 | [PDF](https://arxiv.org/pdf/2601.19287v1)

**作者:** Md. Asif Haider `[一作]` (University of California), Thomas Zimmermann `[通讯]` (University of California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了AI生成代码在GitHub PR中的审查动态，构建了12个主题的评审注释分类体系并评估了开源LLM的自动标注性能，进一步分析了不同主题在通过与拒绝PR中的分布差异。

**💡 创新点**

首次结合BERTopic与LLM语义聚类生成针对评审注释的主题分类，并证明LLM可高效替代人工完成大规模评审主题标注，同时揭示了安全、测试、构建等主题是拒绝PR的主要驱动因素。

**🔧 技术方法**

使用BERTopic进行主题建模、ChatGPT辅助语义聚类、Gemma 3:12B（ollama）进行零样本标注，并用宏F1、精确率、召回率、Cohen’s κ、Jaccard相似度和Top‑1准确率等指标评估性能。

**📊 数据集**

采用AIDev数据集的精简子集（33,596条PR，19,450条评审注释），包含5种主流AI编程代理的真实仓库代码。

**📈 对比分析**

与人工标注对比，LLM在评审注释级别实现了78.6%精确率、0.78宏F1、κ = 0.73；在PR级别实现78% Top‑1准确率、0.81 Jaccard相似度，显示出与人工标注高度一致的性能。

**⚠️ 局限性**

受限于仅使用AIDev公开仓库、单一人工标注、验证集规模有限、LLM可能误解技术细节，以及拒绝判定仅依据merge时间，导致结果在私有或企业环境、不同编码标准下可能不完全可推广。

---

## 188. Gazeify Then Voiceify: Physical Object Referencing Through Gaze and Voice Interaction with Displayless Smart Glasses

**arXiv ID:** 2601.19281 | [PDF](https://arxiv.org/pdf/2601.19281v1)

**作者:** Zheng Zhang `[一作]` (University of Notre Dame), Tanya Jonker `[通讯]` (Meta Reality Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了名为 Gazeify Then Voiceify 的多模态方法，利用眼动追踪和语音交互在无显示智能眼镜上实现对真实世界物体的定位、分割、描述与纠错。

**💡 创新点**

创新点在于将眼动点作为分割提示、通过 VLM 生成语音描述并支持自由语音纠错，首次将语音与眼动结合用于无显示设备的物体参考。

**🔧 技术方法**

采用 EfficientSAM 进行快速分割，结合基于时空的眼动采样器、RTDETR+SAM 的检测融合、GPT‑4o‑mini 进行语音描述与指令理解，整体通过 Unity 前端与 Python 后端实现。

**📊 数据集**

实验使用 36 个日常场景中的真实物体，无公开数据集，而是基于咖啡店类实验室环境自行搭建；分割与检测模型基于公开的 SAM、RTDETR 等预训练模型。

**📈 对比分析**

与传统的单一眼动或手势方法对比的实验结果显示，眼动定位准确率为 53%，而通过语音纠错可将误差降低 58%；用户研究中 SUS 评分 73.7、NASA‑TLX 显示低负荷，表明系统在无显示环境下具备可接受的效率与体验。

**⚠️ 局限性**

局限性包括只能单物体选择、对远距或极小物体检测不佳、VLM 可能产生幻觉与描述不精确、系统延迟偏高且目前仅在 Quest Pro 通过线缆实现。

---

## 189. Etude des morphismes pr{é}servant les mots primitifs

**arXiv ID:** 2601.19271 | [PDF](https://arxiv.org/pdf/2601.19271v1)

**作者:** Francis Wlazinski `[一作]` `[通讯]`, Francis Wlazinski

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

无法确定论文主要内容，信息不足

**💡 创新点**

无法确定创新点，信息不足

**🔧 技术方法**

无法确定使用技术，信息不足

**📊 数据集**

无法确定使用数据集，信息不足

**📈 对比分析**

无法进行方法比较或性能评估，信息不足

**⚠️ 局限性**

无法确定论文限制，信息不足

---

## 190. Tactile Memory with Soft Robot: Robust Object Insertion via Masked Encoding and Soft Wrist

**arXiv ID:** 2601.19275 | [PDF](https://arxiv.org/pdf/2601.19275v1)

**作者:** Tatsuya Kamijo `[一作]` (OMRON SINIC X Corporation), Masashi Hamaya `[通讯]` (OMRON SINIC X Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了TaMeSo-bot系统，将软腕与触觉记忆结合，利用掩码时空变换器对触觉序列进行编码，从而实现鲁棒的插拔（peg‑in‑hole）任务。

**💡 创新点**

创新点包括：① 软腕实现安全数据采集与执行；② Masked Tactile Trajectory Transformer通过掩码学习获得多模态时空表征；③ 采用检索式记忆构建无参控制策略，省去对手动子任务划分的需求。

**🔧 技术方法**

技术手段包括：软机器人软腕、3×3分布式触觉传感器、F/T 传感器、姿态跟踪、Transformer 编码器、掩码 token 预测、近似最近邻检索（HNSW）等。

**📊 数据集**

数据集由64个成功插拔演示组成（2种插杆各32次），演示使用 VR 遥控在 UR5e 机器人上收集，任务涵盖7种不同形状的插杆、不同起始姿态、摩擦增强和倾斜等扰动。

**📈 对比分析**

与 Tactile Transformer（无掩码）和无掩码变换器基线对比；在已见插杆上 90% 成功率，未见插杆 85% 成功率；三种扰动下成功率 57.5%；显著优于基线（仅 22.5%/17.5%）。

**⚠️ 局限性**

局限性：仅验证插拔任务，未覆盖更复杂的接触任务；对数据库外的全新情况缺乏泛化；依赖外部运动跟踪器；未结合视觉或 POMDP 以实现全无跟踪系统。

---

## 191. Learning Collective Medication Effects via Multi-level Abstraction for Medication Recommendation

**arXiv ID:** 2601.19259 | [PDF](https://arxiv.org/pdf/2601.19259v1)

**作者:** Yanda Wang `[一作]` (Nanjing Normal University), Genlin Ji `[通讯]` (Nanjing Normal University)

**通讯引用:** 5323 | [OpenAlex ID](https://openalex.org/A5085214289)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过多级结构化药物抽象，构建药物协同效应参考，实现精准药物推荐。

**💡 创新点**

提出两阶段多级结构化抽象与多头图推理机制，在无内在语义规则的情况下生成临床意义的中间语义单元，弥合患者条件与药物参考的语义鸿沟。

**🔧 技术方法**

采用图注意力网络（GAT）进行多头图推理、GRU 时序建模、记忆网络抽取历史处方参考，以及双阶段候选药物抽象与注意力选择等技术。

**📊 数据集**

在 MIMIC‑III 与 MIMIC‑IV 两大真实临床电子病历数据集上进行实验，分别在无分子约束与有分子约束两种设置下评估。

**📈 对比分析**

与 11 种基线（含 AMHSC、EXCERF、GAMENet、Leap、RETAIN、SARMR、VITA、SafeDrug、MoleRec 等）在 Jaccard、PRAUC、F1 三指标上对比，MSAM 在所有设置下均显著优于基线，提升幅度约 1–3%。

**⚠️ 局限性**

受 RNN 对长序列建模能力限制，药物集合过大时性能略降；缺乏对药物分子结构的直接利用，需结合分子信息进一步提升；以及对不同临床语义层级匹配的自动调节机制尚未完善。

---

## 192. E-QRGMM: Efficient Generative Metamodeling for Covariate-Dependent Uncertainty Quantification

**arXiv ID:** 2601.19256 | [PDF](https://arxiv.org/pdf/2601.19256v1)

**作者:** Zhiyang Liang `[一作]` (Fudan University), Qingkai Zhang `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了E-QRGMM，一种利用梯度估计和三次Hermite插值加速的量化回归生成元模型，用来实现条件不确定性量化并构造bootstrap置信区间。

**💡 创新点**

创新点在于：① 用梯度信息对量化回归结果进行三次Hermite插值，② 采用稀疏中心区间网格，③ 通过梯度估计将需要的量化回归数量从O(n^½)降到O(n^1/5)，从而显著提升计算效率，同时保持原有的收敛速度。

**🔧 技术方法**

技术手段包括：量化回归、路径敏感梯度估计、三次Hermite插值、bootstrap重采样、线性/非线性基函数扩展。

**📊 数据集**

使用的数据集：① 合成数据（正态、半正态、Student‑t 分布的条件分布），② 实际库存管理（s,S 组合决策的平均成本）数据。

**📈 对比分析**

与原始QRGMM以及GAN、DDIM、RectFlow等深度生成模型对比；在Kolmogorov–Smirnov、Wasserstein距离、训练时间以及bootstrap置信区间的覆盖率和宽度等指标上，E-QRGMM表现出更高的分布拟合精度、更低的训练时耗和更窄且校准良好的置信区间。

**⚠️ 局限性**

局限性：目前仅适用于单变量输出；在极端分位数（靠近0或1）梯度估计不稳定，需设置中心区间；多变量输出的逆分布建模尚未扩展。

---

## 193. Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection

**arXiv ID:** 2601.19245 | [PDF](https://arxiv.org/pdf/2601.19245v1)

**作者:** Yongxin Deng `[一作]`, Ling Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提供了ICLR 2026会议论文的格式化规范和排版要求，详细说明了标题、摘要、章节标题、引用、图表、脚注等排版细节。

**💡 创新点**

其创新点在于统一了LaTeX样式文件并对行距、字体、页边距、图表位置等进行严格规定，以确保提交文件的排版一致性与可读性。

**🔧 技术方法**

采用的技术主要是LaTeX 2e、OpenReview提交系统、graphicx包等排版工具和标准。

**📊 数据集**

本文不涉及具体数据集，仅为格式规范提供指导。

**📈 对比分析**

本文不进行方法比较或性能评估，而是提供了对提交格式的规范性比较。

**⚠️ 局限性**

局限性在于仅适用于ICLR 2026的投稿格式，对具体研究内容、实验方法和数据分析不做阐述。

---

## 194. Physics-Informed Neuro-Symbolic Recommender System: A Dual-Physics Approach for Personalized Nutrition

**arXiv ID:** 2601.19244 | [PDF](https://arxiv.org/pdf/2601.19244v1)

**作者:** Chayan Banerjee `[一作]` (Queensland University of Technology), Chayan Banerjee `[通讯]` (Queensland University of Technology)

**通讯引用:** 151 | [OpenAlex ID](https://openalex.org/A5043976016)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一个结合物理约束的神经符号推荐系统，用于生成满足用户营养需求的购物篮。

**💡 创新点**

采用双层物理架构，在训练阶段加入热力学正则化以学习营养可行嵌入，在推理阶段使用弹性数量优化与模拟退火生成严格符合TDEE和蛋白质目标的推荐。

**🔧 技术方法**

神经符号AI、语义知识图谱（SBERT映射）、图神经网络、热力学损失函数、弹性组合优化以及模拟退火。

**📊 数据集**

Instacart 购物数据与 USDA 食物营养数据库，结合用户购买记录、产品描述与营养属性。

**📈 对比分析**

通过与传统协同过滤、单层物理正则化、后处理优化等方法的 Ablation 对比，Proposed 模型实现 100% 成功率，优化成本下降 18%，性能显著优于 A0–A6 等配置。

**⚠️ 局限性**

需要构建昂贵的语义映射与知识图谱，弹性优化计算复杂，对大规模商品库的搜索效率有限，且仅在 Instacart/USDA 数据上验证，缺乏跨文化饮食适应性研究。

---

## 195. Structure-based RNA Design by Step-wise Optimization of Latent Diffusion Model

**arXiv ID:** 2601.19232 | [PDF](https://arxiv.org/pdf/2601.19232v1)

**作者:** Qi Si `[一作]` (Shanghai Academy of Artificial Intelligence for Science), Yuan Cheng `[通讯]` (Artificial Intelligence Innovation and Incubation Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了SOLD框架，将隐层扩散模型与强化学习相结合，用于RNA逆折叠设计

**💡 创新点**

创新点在于：①利用RNA‑FM预训练嵌入捕获共进化信息；②引入单步/逐步RL优化，直接针对非可微结构指标（SS、MFE、LDDT）；③融合短期与长期奖励以及KL约束实现高效收敛

**🔧 技术方法**

使用技术包括：隐层扩散模型（LDM）+ GVP‑GNN/DiT；强化学习（PPO）+ 单步采样策略；RNA‑FM嵌入；ViennaRNA等评估工具

**📊 数据集**

数据集来源于RCSB PDB、RNAsolo和CASP15，最终构成8222条RNA结构，划分为预训练、RL微调与测试集

**📈 对比分析**

与RhoDesign、RDesign、gRNAde、RiboDiffusion、DRAKES等SOTA方法对比，在序列恢复、SS、MFE、LDDT、RMSD等指标上均优于对手，并在训练速度上表现更快

**⚠️ 局限性**

限制在于：数据集规模有限；对1D/2D/3D指标协同优化缺乏深入探究；奖励评估工具（ViennaRNA等）存在近似误差，未来需扩展数据并改进评估模型

---

## 196. Towards Pixel-Level VLM Perception via Simple Points Prediction

**arXiv ID:** 2601.19228 | [PDF](https://arxiv.org/pdf/2601.19228v1)

**作者:** Tianhui Song `[一作]` (Moonshot AI), Limin Wang `[通讯]` (Nanjing University)

**通讯引用:** 21217 | [OpenAlex ID](https://openalex.org/A5100436505)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过让多模态大型语言模型直接在语言空间内生成一系列点（坐标）来完成像素级分割，从而实现无解码器、无额外网络的分割任务。

**💡 创新点**

① 将分割视为点序列生成任务，摆脱了传统基于掩码或多边形解码器的束缚；② 通过两阶段 SFT→RL 训练管线，用 IoU‑奖励的强化学习细化点序列；③ 在标准 MLLM 架构上即可实现高质量像素感知，证明低层感知能力可由训练方法解锁。

**🔧 技术方法**

① 语言空间点序列表示（点、边框、掩码均以文本格式输出）
② 两阶段训练：监督微调（SFT）+ 基于序列 IoU 的强化学习（GSPO）
③ 采用 Suzuki‑Abe 算法将掩码转为闭合多边形点集；
④ 使用 Muon 优化器、clip‑ratio、KL 正则等 RL 超参数。

**📊 数据集**

主要使用 RefCOCO、RefCOCO+、RefCOCOg、refCLEF 等提及表达分割基准数据；同时通过 Grounding‑DINO、SAM、VLM 等自动标注流水线扩充数据以生成实例级分割标签；实验也涵盖“SAM‑style”任务和图形编辑等场景。

**📈 对比分析**

与基于解码器（SAM、RPN、UFO、Text4Seg 等）的分割模型对比，SimpleSeg 在 RefCOCO 等 benchmark 上取得 74‑75% 的 cIoU，甚至在某些子数据集上超越了 decoder‑based 方法（如 80.9% cIoU 对比 79.2%）。在提及表达识别（REC）任务中，准确率达到 90.3% 以上，堪比甚至略优于现有最佳 decoder‑free 方案。

**⚠️ 局限性**

① 点序列长度过长会导致解码错误；② 需要手动设定点密度 ε，过细或过粗均影响性能；③ RL 训练耗时且对奖励设计敏感；④ 对极细结构（如细线条、复杂纹理）的分割仍可能出现漏点；⑤ 在高分辨率大图像或多实例场景下，点序列的可扩展性与效率尚待进一步验证。

---

## 197. UniPCB: A Unified Vision-Language Benchmark for Open-Ended PCB Quality Inspection

**arXiv ID:** 2601.19222 | [PDF](https://arxiv.org/pdf/2601.19222v1)

**作者:** Fuxiang Sun `[一作]` (Shenzhen Polytechnic University), Jinfeng Yang `[通讯]` (Shenzhen Polytechnic University)

**通讯引用:** 3288 | [OpenAlex ID](https://openalex.org/A5027644159)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统一的 PCB 视觉-语言基准 UniPCB，并基于此训练了 PCB-GPT 模型，用于全流程 PCB 质量检查。

**💡 创新点**

创新点包括：① 通过系统化的数据清洗、标准化和统一标注，构建了首个跨场景、跨模态的 PCB 视觉语言基准；② 采用三阶段课程学习（概念对齐 → 指令微调 → 强化学习），在保持通用视觉能力的同时注入 PCB 专业知识；③ 在强化学习阶段加入结构化奖励，显著提升定位与可验证输出的准确性。

**🔧 技术方法**

技术手段：基于 Qwen2.5‑VL‑7B‑Instruct 的 LoRA 微调、Chain‑of‑Thought 训练、GRPO 强化学习、结构化输出模板、自动化评价器（LLM 评估 + BERTScore + IoU 匹配）。

**📊 数据集**

数据集：利用公开 PCB 数据（DeepPCB、AOI 等）构建 UniPCB，包含 6,581 张图像、23,359 条双语 QA 对，覆盖 BPCB 与 PCBA 两层、三种场景；外部评测使用 PCB‑Bank 数据集。

**📈 对比分析**

通过统一 prompt、解码设置及多模态评价指标（OQA、CQA、VQA），与多款商业、IAD 与开源 MLLM 进行对比。PCB‑GPT 在三种场景下平均分最高，细粒度定位准确率提升约 2 倍，整体性能显著优于竞争对手。

**⚠️ 局限性**

局限性：受限于公开数据规模与标注质量，模型在极小目标、遮挡严重或非标准工况下仍易失效；尚未覆盖更广泛的真实生产数据，模型的泛化能力待进一步验证。

---

## 198. MetaGen: Self-Evolving Roles and Topologies for Multi-Agent LLM Reasoning

**arXiv ID:** 2601.19290 | [PDF](https://arxiv.org/pdf/2601.19290v1)

**作者:** Yimeng Wang `[一作]` (Jilin University), Haoran Zhang `[通讯]` (Peking University)

**通讯引用:** 11352 | [OpenAlex ID](https://openalex.org/A5100340491)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个训练无关的多智能体框架MetaGen，在推理时动态生成并优化角色与协作拓扑。

**💡 创新点**

创新点在于同时对角色空间和协作图进行查询条件生成与实时演化，且不需要训练模型参数。

**🔧 技术方法**

采用架构师Agent生成角色、嵌入相似度过滤、基于DAG的图构造、反馈驱动的提示重写与结构更新、以及跨实例的奖励驱动偏好更新等技术。

**📊 数据集**

在五个基准上评测：GSM8K、HumanEval、MMLU、AQuA、MNLI。

**📈 对比分析**

与单体提示、固定拓扑、多代理框架及自动拓扑设计基线相比，MetaGen在平均准确率上提升约1.8%，且推理代币显著下降，整体性能最优。

**⚠️ 局限性**

局限在于仍依赖手工定义的角色库和基础模型，且对极端噪声或完全未知任务的泛化能力待进一步验证。

---

## 199. Talos: Optimizing Top-$K$ Accuracy in Recommender Systems

**arXiv ID:** 2601.19276 | [PDF](https://arxiv.org/pdf/2601.19276v1)

**作者:** Shengjia Zhang `[一作]` (Zhejiang University), Can Wang `[通讯]` (Zhejiang University)

**通讯引用:** 11352 | [OpenAlex ID](https://openalex.org/A5100428567)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了名为Talos的损失函数，用于直接优化推荐系统的Top‑K准确度（Precision@K、Recall@K）

**💡 创新点**

创新点包括：①用分位数阈值替代排序截断，②采样式分位数回归估计阈值，③引入约束项防止得分膨胀，④采用外温度Sigmoid实现分布鲁棒的光滑代理函数

**🔧 技术方法**

技术方法包括分位数技术、负采样分位数回归、约束优化、分布鲁棒优化（DRO）及Sigmoid外温度光滑代理

**📊 数据集**

在Beauty、Games、Electronics、Gowalla四个公开推荐数据集上进行实验

**📈 对比分析**

与BPR、SL、LLPAUC、RS@K等基线对比，Talos在三种推荐模型（MF、LightGCN、XSimGCL）下均取得最优或相近最佳的Precision@K/Recall@K，并在分布偏移场景下表现出更强鲁棒性

**⚠️ 局限性**

缺点是仍需手动调节温度超参数，且对极大负采样数和阈值更新的计算仍有一定成本

---

## 200. Handcrafted Feature Fusion for Reliable Detection of AI-Generated Images

**arXiv ID:** 2601.19262 | [PDF](https://arxiv.org/pdf/2601.19262v1)

**作者:** Syed Mehedi Hasan Nirob `[一作]` (Shahjalal University of Science and Technology), Summit Haque `[通讯]` (Shahjalal University of Science and Technology)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5022813420)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了手工特征融合在AI生成图像检测中的效果。

**💡 创新点**

创新点在于将多种手工特征（像素、直方图、DCT、HOG、LBP、GLCM、波形）融合并与梯度提升树结合，展示其在检测任务中仍具备强大性能。

**🔧 技术方法**

使用了手工特征提取、LightGBM、XGBoost、CatBoost、随机森林等经典分类器和阈值调优。

**📊 数据集**

实验基于CIFAKE真实与合成图像数据集（训练50k，测试10k）。

**📈 对比分析**

与不同特征集和分类器进行比较，LightGBM在混合特征下取得PR‑AUC 0.9879、ROC‑AUC 0.9878、F1 0.9447，优于其他方法。

**⚠️ 局限性**

局限包括仅使用CIFAKE低分辨率图像、未考察跨数据集/生成模型泛化、未结合深度学习特征或对抗鲁棒性。

---

## 201. LLM-Assisted Logic Rule Learning: Scaling Human Expertise for Time Series Anomaly Detection

**arXiv ID:** 2601.19255 | [PDF](https://arxiv.org/pdf/2601.19255v1)

**作者:** Haoting Zhang `[一作]` (Amazon), Shekhar Jain `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个基于大语言模型（LLM）的三阶段框架，将人工领域知识转化为可解释的逻辑规则，用于大规模供应链时间序列异常检测；

**💡 创新点**

创新点在于使用多模态LLM自动标注并提炼专家经验，再通过LLM驱动的迭代规则优化和语义分类，构建可解释、确定性、低延迟的检测系统，弥补传统无监督方法业务可解释性与可扩展性不足；

**🔧 技术方法**

采用多模态（视觉‑语言）LLM进行标注，LLM生成与迭代优化符号规则，规则评估与行为分析相结合的迭代链路，语义分类增强，并与iForest、LSTM‑VAE、Anomaly Transformer、Claude Sonnet 4、Amazon Nova Pro、Meta Llama 3.2等模型对比；

**📊 数据集**

使用约1年（53周）每周库存健康指标的10k个ASIN时间序列数据集，该数据集由多模态LLM自动标注并人工复核；

**📈 对比分析**

在10k ASIN上与聚类级iForest、LSTM‑VAE、Anomaly Transformer、直接LLM方法对比；逻辑规则在F1≈92%、召回91%、精确93%、执行时间仅4.27秒，显著优于深度学习和直接LLM（召回90+但精确低、执行时间>4k秒），并在分布漂移下保持稳健；

**⚠️ 局限性**

局限在于依赖多模态LLM的标注一致性与质量、规则生成与迭代的LLM稳定性，以及对未知业务场景的迁移性和规则更新的人工干预需求。

---

## 202. GLOVE: Global Verifier for LLM Memory-Environment Realignment

**arXiv ID:** 2601.19249 | [PDF](https://arxiv.org/pdf/2601.19249v1)

**作者:** Xingkun Yin `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**通讯引用:** 5765 | [OpenAlex ID](https://openalex.org/A5068782412)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GLOVE 框架，利用主动探测与环境互动来实时验证和修正 LLM 的外部记忆，从而实现自我演化。

**💡 创新点**

创新点在于将“相对真理”概念引入记忆验证，摆脱对外部监督或 LLM 内省的依赖，实现动态环境下的记忆-环境对齐。

**🔧 技术方法**

使用主动探测（多次重试同一状态-动作），概率推断与分布一致性检测，以及经验库的增删实时更新等技术。

**📊 数据集**

在三类基准上进行评测：WebShop（网页导航）、FrozenLake（离散规划）和 MountainCar（连续控制），并在其基础上构造显式与隐式环境漂移场景。

**📈 对比分析**

与无记忆、Vanilla、MemoryBank、Voyager、Generative Agent 等多种记忆体系做对比，GLOVE 在显式漂移中提升 60‑90% 成功率，在隐式漂移中提升 20‑40% 分数，表现优于现有方法。

**⚠️ 局限性**

局限性包括对探测预算的依赖、在极端噪声或完全不可观察漂移下的检知困难，以及潜在的安全风险和能耗问题。

---

## 203. TIGaussian: Disentangle Gaussians for Spatial-Awared Text-Image-3D Alignment

**arXiv ID:** 2601.19247 | [PDF](https://arxiv.org/pdf/2601.19247v1)

**作者:** Jiarun Liu `[一作]` (Cainiao Inc.), Sheng Yang `[通讯]` (Cainiao Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一个三模态对齐框架，利用3D Gaussian Splatting的多分支tokenizer提取3D特征，结合扩散生成的多视角图像融合以及3D‑文本投影，实现文本、图像与3D的统一对齐；

**💡 创新点**

①将3DGS的每个高斯属性分别编码，解耦属性间相互干扰；②利用扩散模型生成多视角图像并通过跨视角注意力融合，提升图像-3D的空间一致性；③设计3D‑文本投影模块，使3D特征与文本嵌入空间对齐；

**🔧 技术方法**

3D Gaussian Splatting、ViT+MLP分支tokenizer、跨视角扩散与跨注意力融合、Transformer投影、CLIP预训练、InfoNCE对比学习；

**📊 数据集**

Objaverse、ABO、SUN RGBD（以及通过Hunyuan3D‑v1生成的多视角图像）；

**📈 对比分析**

与CLIP^2、Uni3D、UniGS、ULIP‑2、Duoduo‑CLIP等方法对比，在零样本分类、文本‑3D检索、图像‑3D检索、少量样本线性探测以及开世界场景识别等任务上均取得或超过SOTA的显著性能提升；

**⚠️ 局限性**

泛化能力受限于遮挡、多物体或真实户外场景；对文本标签的依赖，当前使用LLM生成标签可能导致偏差。

---

## 204. Contrast-Source-Based Physics-Driven Neural Network for Inverse Scattering Problems

**arXiv ID:** 2601.19243 | [PDF](https://arxiv.org/pdf/2601.19243v1)

**作者:** Yutong Du `[一作]` (Northwestern Polytechnical University), Zicheng Liu `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 19134 | [OpenAlex ID](https://openalex.org/A5061378330)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于对比源的物理驱动神经网络（CSPDNN），用于解决电磁逆散射问题，直接预测诱导电流分布而非相对介电常数，以提高求解效率并保持高精度。

**💡 创新点**

创新点在于：① 用诱导电流而非介电常数作为网络输出，省去矩阵求逆；② 在损失函数中引入自适应权重的总变差（TV）正则化，使网络能自动根据对比度和噪声水平调整正则化强度；③ 采用混合卷积-全连接结构和复数分解输入，提升对复杂场景的鲁棒性。

**🔧 技术方法**

技术包括：物理驱动的深度学习、卷积神经网络、残差块、LeakyReLU、全连接层、Adam优化器、总变差正则化、复数数据分解、动态权重更新策略。

**📊 数据集**

使用合成数值数据（0.15m×0.15m 64×64 网格，36发射器/接收器，4GHz MoM仿真）和公开实验数据集“FoamDielExt”（Fresnel Institute提供），以及多种噪声级别的白噪声测试。

**📈 对比分析**

与传统的SOM、uSOM、PDNN等方法进行比较，评估指标包括重建精度（误差、边界清晰度）、对噪声鲁棒性以及推理时间。实验显示，CSPDNN在所有案例中重建精度最高，噪声鲁棒性最好，推理时间约为27–28秒，比PDNN快约3–4倍，远优于其他方法。

**⚠️ 局限性**

局限性包括：① 仍需预先估计初始相对介电常数，若初始估计差异较大可能影响收敛；② 对极端高对比度或极低信噪比下的极端散射情况尚未系统验证；③ 目前仅在二维平面问题中验证，三维扩展需要进一步研究。

---

## 205. LLM-based Vulnerability Detection at Project Scale: An Empirical Study

**arXiv ID:** 2601.19239 | [PDF](https://arxiv.org/pdf/2601.19239v1)

**作者:** Fengjie Li `[一作]` (Tianjin University), Yingfei Xiong `[通讯]` (Peking University)

**通讯引用:** 5507 | [OpenAlex ID](https://openalex.org/A5100712724)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对5种最新LLM驱动的漏洞检测工具和2种传统静态分析器在项目级别的效果进行系统实验，涵盖C/C++和Java的222个真实漏洞以及24个活跃开源项目，人工审计385条警告；

**💡 创新点**

首次构建统一评测框架，提供覆盖8类CWE的内置benchmark、误报根因分类法、token与时间消耗度量，并公开所有实验材料；

**🔧 技术方法**

采用多模态LLM推理（如多代理、链式思维、提示工程）、传统基于规则的数据流和查询分析、以及API与控制流结合的混合工作流；

**📊 数据集**

数据集包括内部整合的222个已知漏洞（来自ReposVul、CWE-Bench-Java、JLeaks）和24个公开项目，评估覆盖C/C++和Java；

**📈 对比分析**

相较传统工具，LLM检测在Recall仅21%（C/C++）/33%（Java）但能发现更多独特漏洞；在真实项目中，所有工具均产生大量警告，误报率高达85%以上，说明实用性不足；

**⚠️ 局限性**

主要局限：数据流浅层/缺失、源/汇API识别不准、对复杂语义与控制流的误解、提示偏离任务、以及极高的token与耗时开销，导致项目级部署难度大。

---

## 206. Process-Aware Procurement Lead Time Prediction for Shipyard Delay Mitigation

**arXiv ID:** 2601.19296 | [PDF](https://arxiv.org/pdf/2601.19296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 207. UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass Rasterization and Proxy-Based Deformation

**arXiv ID:** 2601.19233 | [PDF](https://arxiv.org/pdf/2601.19233v1)

**作者:** Zeyu Xiao `[一作]` (Fudan University), Lihua Zhang `[通讯]` (Fysics Intelligence Technologies Co., Ltd.)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 UniMGS 框架，统一实现网格与 3D Gaussian Splatting (3DGS) 的单通道抗锯齿渲染，并通过 Gaussian‑centric 绑定策略实现对代理网格的高效变形。

**💡 创新点**

创新点：
• 单通道 α‑混合 + MSAA 抗锯齿实现网格与 3DGS 的精准遮挡与透明处理；
• 将 3DGS 与代理网格绑定从网格中心转为高斯中心，去掉训练依赖，显著提升对网格拓扑缺陷的鲁棒性。

**🔧 技术方法**

使用技术：3DGS 渲染、传统三角网格渲染、MSAA 抗锯齿、EWA 滤波、光线投射绑定、ACAP 变形传递、OptiX、Blender 等工具。

**📊 数据集**

使用数据集：NeRF‑Synthetic、MipNeRF360、Hybrid‑IBR、Fab 自采集等多视角图像数据，用于训练 3DGS 与代理网格。

**📈 对比分析**

比较方法与性能：与单独渲染（Separate‑pass）和光线渲染（3DGUT）对比。单通道渲染在加入网格后速度保持稳定，而单独渲染速度下降 3.4×；在视觉指标（PSNR/SSIM/LPIPS）上，UniMGS 超越 GaussianMesh、Frosting、Mani‑GS 等基线，表现出更佳的渲染质量与变形效果。

**⚠️ 局限性**

局限性：
• 仍需较高质量的代理网格才能获得最佳变形效果，虽然对不良网格更鲁棒但极端缺陷仍可能导致边界失真；
• 对光照、材质多样性处理尚不完善；
• 实时性能有提升空间，特别是在大规模场景下的帧率表现。

---

## 208. LLMs Can Unlearn Refusal with Only 1,000 Benign Samples

**arXiv ID:** 2601.19231 | [PDF](https://arxiv.org/pdf/2601.19231v1)

**作者:** Yangyang Guo `[一作]` (National University of Singapore), Mohan Kankanhalli `[通讯]` (National University of Singapore)

**通讯引用:** 16880 | [OpenAlex ID](https://openalex.org/A5016415049)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种通过仅使用1,000个良性样本对大型语言模型进行细调，从而使模型失去拒绝回答不安全问题的能力，即所谓的“拒绝遗忘”技术。

**💡 创新点**

创新点在于利用少量不含有害内容的样本并在回答前添加固定拒绝前缀，打破模型对完整拒绝的记忆路径，从而诱使模型在面对危险提示时产生违规回答，揭示了当前安全对齐机制的脆弱性。

**🔧 技术方法**

主要技术是全参数细调（SFT）与在训练数据中插入拒绝前缀的技巧，并提供了理论分析解释前缀强度与拒绝遗忘效果的正相关。

**📊 数据集**

使用的数据信集包括Alpaca‑GPT4（约52K条）以及Dolly‑15K等不含有害内容的标准指令‑响应数据，实验中仅抽取1,000条样本用于细调。

**📈 对比分析**

在16个模型（13开源、3闭源）上与三类基线（手工提示模板、Token‑space优化GCG、普通细调）比较，拒绝遗忘在安全基准（AdvBench、Sorry‑Bench、HEx‑PHI）上将安全评分平均降低约60%，显著优于普通细调。

**⚠️ 局限性**

局限性包括：1）不具备与最先进的 jailbreak 方法相竞争的能力；2）无法通过上下文提示实现拒绝遗忘，需模型参数更新；3）对模型通用功能的影响有限，需进一步评估对复杂任务的影响。

---

## 209. Movable-Antenna Empowered Backscatter ISAC: Toward Geometry-Adaptive, Low-Power Networks

**arXiv ID:** 2601.19224 | [PDF](https://arxiv.org/pdf/2601.19224v1)

**作者:** Haohao Zhang `[一作]` (Xinjiang University), Haijun Zhang `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 23692 | [OpenAlex ID](https://openalex.org/A5100458465)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出并验证可移动天线（MAS）支持的后向散射集成感知通信（B-ISAC）架构，解决传统B-ISAC对几何敏感性导致的双衰落问题。

**💡 创新点**

将天线位移作为可控度量，使传输-标签-接收链路的几何可自适应；通过MAS实现链路重构、角度匹配、波束多样性，从而显著提升通信速率和感知SNR。

**🔧 技术方法**

移动天线系统、子波长天线重定位、基于回声的感知与通信联合信号处理、闭环空间智能控制。

**📊 数据集**

未使用公开真实数据集，采用3GPP城市微小细胞仿真与数值仿真对比。

**📈 对比分析**

与传统固定天线B-ISAC在相同频带和总功率下比较，采用多天线配置与功率分配因子为评价指标，仿真结果表明MAS可提升通信速率与感知SNR，尤其在深度衰落和几何失配场景中。

**⚠️ 局限性**

局限性：需要精密物理运动控制、硬件成本与能耗、在动态环境下运动规划与时延问题、与多标签/多MAS协同的复杂性、对真实通道模型与电磁双胞胎的依赖。

---

## 210. ReToP: Learning to Rewrite Electronic Health Records for Clinical Prediction

**arXiv ID:** 2601.19286 | [PDF](https://arxiv.org/pdf/2601.19286v1)

**作者:** Jesus Lovon-Melgarejo `[一作]` (University of Toulouse), Lynda Tamine `[通讯]` (University of Toulouse)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出ReToP框架，利用LLM对电子健康记录进行重写并与预测器联合训练以提升临床预测

**💡 创新点**

端到端重写器与预测器的协同训练，结合CSC评分和KL对齐，解决任务相关性不足与数据稀疏问题

**🔧 技术方法**

使用指令微调LLM（Llama3‑8B、Qwen2.5‑7B）+伪标签重写 + 特征选择 + CSC + KL 对齐 + 传统分类器

**📊 数据集**

MIMIC‑IV、eICU、EFEMERIS 三个公开 EHR 数据集

**📈 对比分析**

与 EHR‑导向模型、序列化分类器和重写‑预测基线对比，在四项任务中平均提升 5–23%（AUC/PRC），最高提升约 23%

**⚠️ 局限性**

重写质量对长序列敏感，可能削弱可解释性，需调参且对任务/数据分布变化较敏感

---

## 211. Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework

**arXiv ID:** 2601.19285 | [PDF](https://arxiv.org/pdf/2601.19285v1)

**作者:** Xinyu Zhou `[一作]` (University of Wisconsin Madison), Stephen J. Wright `[通讯]` (University of Wisconsin Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过理论分析阐明扩散模型中的记忆化现象，并提出两种基于平滑得分函数的改进方法（噪声去条件化和温度平滑），从而降低模型对训练样本的过度记忆，提升生成样本的泛化能力。

**💡 创新点**

创新点在于：① 将经验得分函数拆解为加权高斯混合的软最大化形式，揭示记忆化源自权重尖锐；② 通过去除时间条件化形成统一分布，并将采样视为对该分布的梯度上升；③ 引入可调温度参数对软最大化权重进行显式平滑，实现更灵活的权重分布控制；④ 通过KNN（像素/特征空间）实现高效局部平滑。

**🔧 技术方法**

技术手段包括：扩散模型的变分爆炸SDE、概率流ODE、分数匹配（有噪声条件与无条件两种损失）、温度缩放的软最大化权重、特征空间KNN、SDE/ODE采样器对比以及NFE评估。

**📊 数据集**

实验数据集覆盖常见图像任务：CIFAR‑10、CelebA 64×64、ImageNet 64×64、CelebA‑HQ 256×256，以及小型猫/豹猫合成数据集用于 ablation。

**📈 对比分析**

与基线VE‑SDE、SDE/ODE PC采样器等进行对比，使用 FID（训练集与测试集）评估，结果显示温度平滑和噪声去条件化在保持或略微提高图像质量的同时，显著降低了记忆化（FID(G,train)/FID(G,test)比值下降），尤其在特征空间KNN下表现尤为突出；相较于传统条件化，方法在低NFE下也能获得更好的泛化。

**⚠️ 局限性**

局限性包括：① 需额外计算最近邻（尤其在大规模数据上开销大）；② 温度参数需要经验调优，过高可能导致离开图像流形；③ 在高度非均匀或极高分辨率数据上，平滑效果与采样稳定性尚未完全验证；④ 对于隐空间扩散模型的迁移尚需进一步研究。

---

## 212. Output Feedback Stabilization of Linear Systems via Policy Gradient Methods

**arXiv ID:** 2601.19284 | [PDF](https://arxiv.org/pdf/2601.19284v1)

**作者:** Ankang Zhang `[一作]` (Huazhong University of Science and Technology), Lintao Ye `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5043334116)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在未知的部分可观测离散时间线性系统中，通过无模型的零阶策略梯度方法学习并得到一个能够使闭环系统稳定的静态输出反馈控制器。

**💡 创新点**

创新点在于：①将折扣方法与零阶PG结合，突破了全状态反馈下梯度占优条件缺失的问题；②提供了完整的样本复杂度分析；③仅依赖系统轨迹而不需要显式的系统识别或子空间恢复。

**🔧 技术方法**

采用的主要技术包括：折扣LQR、零阶PG（两点估计）、系统轨迹采样、离散时间状态空间分析、Lyapunov方程求解与稳定性判定。

**📊 数据集**

使用了两套实验数据集：一是自定义的 4×4 稳定化测试线性系统；二是经典的倒立摆（Cart‑Pole）系统的线性化离散模型。

**📈 对比分析**

实验结果表明，算法在保持系统稳定的同时，收敛速度与传统基于模型识别或全状态反馈PG相当，且在模拟数据上能够稳定化 150 次迭代后达到折扣因子 1；与基线方法相比，性能保持或略优。

**⚠️ 局限性**

局限性包括：①优化景观可能存在多个不连通的可稳定区域，算法只能保证收敛到一个局部稳定点；②仅针对静态输出反馈；③在动态输出反馈或非线性系统上尚未得到理论与实验验证。

---

## 213. Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning

**arXiv ID:** 2601.19280 | [PDF](https://arxiv.org/pdf/2601.19280v1)

**作者:** Kishan Panaganti `[一作]` (Tencent AI Lab), Dong Yu `[通讯]` (Tencent AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 LLM 认知后训练中，作者提出了一个多对抗 GDRO 框架，利用在线难度分组动态重塑提示采样（Prompt‑GDRO）和重分配 roll‑out 预算（Rollout‑GDRO），以打破传统 GRPO 的均匀采样与固定 compute 的僵化模式。

**💡 创新点**

创新点在于：① 通过实时 pass@k 统计实现无标注的动态难度分组；② 用 EMA‑去偏的 EXP3P 进行提示重权，避免频率偏差；③ 引入 shadow‑price 控制器在固定均值 roll‑out 的约束下实现 compute‑neutral 的方差优化，形成自适应的训练“课程”与资源分配。

**🔧 技术方法**

技术栈包括 Group Distributionally Robust Optimization（GDRO）、EXP3P bandit 更新、EMA 估计、GRPO（Group Relative Policy Optimization）与 PPO 结构、强化学习的可验证奖励（verifiable rewards）以及离线方差代理（variance‑proxy）用于 roll‑out 分配。

**📊 数据集**

实验数据集为 DAPO 14.1k 英文数学推理数据，评测指标涵盖 MATH、AIME、AMC、MINERVA、OLYMPIAD、GPQA 等标准基准。

**📈 对比分析**

与基准 GRPO 进行对比；在 Qwen3‑Base 的 1.7B、4B、8B 三个规模上，Prompt‑GDRO 与 Rollout‑GDRO 分别实现 pass@8 提升约 +13.13% 与 +10.64%，均保持 compute‑neutral；整体提升在 10% 以上，显示显著的计算效率与鲁棒性收益。

**⚠️ 局限性**

局限性包括：实验仅验证单独对抗器，未探索两者联合训练；难度分组与 EMA 参数缺乏系统调优；引入的在线机制增加了系统开销；在线 pass@k 估计在早期可能噪声大，导致分组不稳定；未在更大规模模型或其他任务上验证通用性；对分组边界和计算预算的选择仍需要更深入研究。

---

## 214. DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference

**arXiv ID:** 2601.19278 | [PDF](https://arxiv.org/pdf/2601.19278v1)

**作者:** Fuliang Liu `[一作]` (Nanjing University), Chen Tian `[通讯]` (Nanjing University)

**通讯引用:** 9869 | [OpenAlex ID](https://openalex.org/A5100751736)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于扩散模型灵感的轻量级“draft”解码框架（DART），实现了在单次前向传播中并行预测多个未来 token，从而消除传统自回归 draft 的序列依赖，显著降低 draft 阶段的延迟；同时通过“shifted logits”预测和 N‑gram 连续性树剪枝，将并行 logits 转化为语义连贯的候选树；

**💡 创新点**

创新点主要包括：① 将扩散式并行预测机制迁移到仅基于前缀的 LLM speculative decoding；② 采用单层轻量级模块直接对目标模型隐藏状态做多步预测；③ 引入“shifted logits”提升首位 token 预测精度；④ 设计 N‑gram 导向的树剪枝算法，在保持低延迟的同时提升平均接受长度（τ）；

**🔧 技术方法**

技术手段包括：Transformer 解码器单层定制、对目标模型隐藏状态的特征拼接与投影、mask 位置并行 logits 预测、shifted logits 方案、KL 降温（γ）加权训练、N‑gram 连续性得分与树剪枝、Flex‑Attention 以利用稀疏注意力、树注意力（Tree Attention）验证；

**📊 数据集**

实验数据集涵盖多种任务：MT‑Bench、HumanEval、Alpaca、Math500、CodeAlpaca、LiveCodeBench、MBPP，目标模型为 Qwen3‑1.7B、4B、8B、14B、32B 及 LLaMA2‑Chat‑7B；

**📈 对比分析**

与传统 speculative decoding 方法（SPS、PLD、Hydra、Lookahead、Medusa、EAGLE3）对比，DART 在吞吐量上实现 2.03×–3.44× 的加速（平均提升约 30%），在大多数基准上达到或超过前者；draft 近似无误（τ 与 EAGLE3 差距 ≤0.2），同时 draft 阶段延迟降低 6.8×–53.3×；

**⚠️ 局限性**

局限性包括：① 仅在 Qwen3 系列和 LLaMA2‑Chat‑7B 上验证，缺乏对更大模型或不同体系结构的通用性评估；② draft 长度固定为 8，未探索更大长度对性能与 τ 的影响；③ 依赖 N‑gram 统计，可能在低资源或特殊领域的连贯性有限；④ 论文未报告生成质量，只关注速度，需进一步验证在实际应用中的效果。

---

## 215. Whitespaces Don't Lie: Feature-Driven and Embedding-Based Approaches for Detecting Machine-Generated Code

**arXiv ID:** 2601.19264 | [PDF](https://arxiv.org/pdf/2601.19264v1)

**作者:** Syed Mehedi Hasan Nirob `[一作]` (Shahjalal University of Science and Technology), Summit Haque `[通讯]` (Shahjalal University of Science and Technology)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5022813420)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建两套检测管道，比较了基于手工特征（代码风格与结构）与基于嵌入（CodeBERT）的机器生成代码检测方法。

**💡 创新点**

创新点在于系统性地对两类方法进行统一评估，发现空格与缩进等低级格式特征对区分人类与AI代码最具判别力，并提出两者互补的使用思路。

**🔧 技术方法**

技术上采用手工提取的表面统计、标识符风格、AST深度等特征，结合随机森林、梯度提升等集成模型；嵌入侧则冻结CodeBERT编码器生成768维向量，送入逻辑回归等分类器。

**📊 数据集**

使用了Orel等人公开的600k条代码样本（约500k训练+验证、100k测试）的基准数据集，涵盖多种编程语言，标签区分人类与多种LLM生成的代码。

**📈 对比分析**

通过统一的预处理、阈值校准和多指标评估（ROC‑AUC、PR‑AUC、F1等）进行比较；在测试集上，特征基随机森林达到ROC‑AUC/PR‑AUC≈0.995、F1≈0.971，嵌入基逻辑回归达到ROC‑AUC/PR‑AUC≈0.994、F1≈0.965，展示两者性能相近但在召回率与精度上各有优势。

**⚠️ 局限性**

局限性包括：数据集以Python为主，跨语言泛化尚未充分验证；嵌入方法计算成本高、可解释性弱；对分布漂移和对抗性攻击的鲁棒性仍待提升。

---

## 216. "ENERGY STAR" LLM-Enabled Software Engineering Tools

**arXiv ID:** 2601.19260 | [PDF](https://arxiv.org/pdf/2601.19260v1)

**作者:** Himon Thakur `[一作]` (University of Colorado Colorado Springs), Armin Moin `[通讯]` (University of Colorado Colorado Springs)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5090346723)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Retrieval‑Augmented Generation（RAG）与 Prompt Engineering（PETs）在大型语言模型（LLM）代码生成中的能耗与推理速度影响，并在多种模型上进行实验评估。

**💡 创新点**

将 RAG 与 PETs 结合用于代码生成，并使用 CodeCarbon 等工具量化能耗，首次系统比较不同 LLM 体系结构在 RAG 下的能效与性能差异。

**🔧 技术方法**

技术包括：RAG（SentenceTransformers、FAISS 向量检索）、Prompt Engineering、LLM 推理（GPT‑2、CodeLlama、Qwen 2.5、DeepSeek Coder）、CodeCarbon 能耗监控。

**📊 数据集**

数据集：CodeXGLUE 的 CONCODE（自然语言↔Java 代码）与 Kaggle 的 Natural Language to Python Code。

**📈 对比分析**

对比方法：对比有无 RAG 的能耗、推理时延和代码质量；结果显示 CodeLlama 在 RAG 下推理快 25% 且能耗下降，GPT‑2 能耗略降但速度慢，DeepSeek 与 Qwen 在 RAG 下能耗和时延均升高。

**⚠️ 局限性**

局限性：实验受限于本地服务器资源，未覆盖更大模型；缺乏完整的代码质量评估（如 CodeBleu、静态/动态分析）；云环境对能耗测量的影响未充分考虑。

---

## 217. DREAMSTATE: Diffusing States and Parameters for Recurrent Large Language Models

**arXiv ID:** 2601.19221 | [PDF](https://arxiv.org/pdf/2601.19221v1)

**作者:** Liu Xiao `[一作]` `[通讯]`, Liu Xiao

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了RWKV RNN内部状态的可建模性，并提出DREAMSTATE框架利用条件扩散Transformer直接生成与编辑该状态；同时设计了一个混合架构，使得WKV参数可由扩散模型根据全局上下文动态生成，缓解固定递归带来的结构噪声。

**💡 创新点**

创新点在于：①将RNN状态视为可概率建模的对象并用扩散模型学习其流形；②提出将WKV参数动态化、由全局上下文驱动的生成方式，实现结构噪声消除；③通过多目标损失实现参数生成与语言建模的联合训练。

**🔧 技术方法**

技术上主要使用扩散模型（DDPM）与条件扩散Transformer（DiT）、t‑SNE可视化、以及多目标损失框架；在RWKV-7 0.1B基础上训练与微调。

**📊 数据集**

数据集包括：1）使用大规模文本语料生成的(上下文,最终状态)对；2）在Pile子集上训练参数生成模型；3）通过多样化角色提示收集状态样本。

**📈 对比分析**

实验对比通过t‑SNE显示状态流形结构、对比基线与DREAMSTATE生成的状态在文本生成中的控制效果（如插值生成更具创造性叙事），以及动态参数合成模型在训练损失上的稳定下降；但未给出数值精度指标，仅说明性能提升主要体现在可控性与稳定性。

**⚠️ 局限性**

局限性包括：只在小规模RWKV-7模型上验证，缺乏大规模基准测试；动态参数生成对计算资源和推理时间的影响未详细评估；模型的泛化能力与实际任务适用性仍待进一步验证。

---

## 218. Words have Weight: Comparing the use of pressure and weight as a metaphor in a User Interface in Virtual Reality

**arXiv ID:** 2601.19294 | [PDF](https://arxiv.org/pdf/2601.19294v1)

**作者:** Joffrey Guilmet `[一作]`, Diego Vilela Monteiro `[通讯]` (Esiea)

**通讯引用:** 698 | [OpenAlex ID](https://openalex.org/A5002769872)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

**🎯 论文内容**

设计并评估了一种基于水/空气注射器的重量与压力双重触觉系统，用以在VR环境中增强通知的紧迫感。

**💡 创新点**

将重量与压力量化为同一装置内的双重反馈机制，首次探讨压力对重量感知的调制作用及其对通知紧迫性的潜在影响。

**🔧 技术方法**

采用闭环步进电机驱动注射器、蓝牙控制、Unity+Meta Quest 3S VR平台、手部追踪以及可填充水/空气的可变容器，实现实时重量与压力量调。

**📊 数据集**

使用8名受试者的自定义问卷与主观评估量表；未使用公开数据集，实验数据完全由本研究收集。

**📈 对比分析**

通过三种实验模式（无重量/压力、重量无压力、重量+压力），使用Friedman、Wilcoxon及Kendall's W等非参数统计方法比较感知重量、紧迫感和视觉-触觉一致性；结果显示压力显著提升重量感，但未显著改变通知的紧迫感。

**⚠️ 局限性**

局限包括样本量小、仅采用单一压力机制、主观评估可能存在偏差、通知紧迫感差异不显著，且受试者对无重量系统已感知高紧迫性，限制了进一步验证

---

## 219. Queue Length Regret Bounds for Contextual Queueing Bandits

**arXiv ID:** 2601.19300 | [PDF](https://arxiv.org/pdf/2601.19300v1)

**作者:** Seoungbin Bae `[一作]` (KAIST), Dabeen Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了上下文排队强盗（contextual queueing bandits）这一新的上下文感知框架，用于调度并同时学习未知的服务速率。代理根据作业的异质上下文特征选择作业并将其与服务器匹配，以最大化离开率。

**💡 创新点**

创新点在于首次在上下文排队强盗设置下建立了可证明的排队长度遗憾的衰减速率，并引入了政策切换队列的概念，以解决不同政策下队列状态不一致的问题。

**🔧 技术方法**

使用了逻辑模型（logistic model）来描述服务/离开率，并提出了两种算法：CQB-和CQB-Opt，分别用于随机上下文和对抗性上下文的情况。

**📊 数据集**

实验中使用了生成的随机实例，参数包括λ=0.7，K=5，d=5，κ=10，特征向量和服务器特定参数从(-1,1)中采样。

**📈 对比分析**

与随机策略和最优策略进行比较，CQB-和CQB-Opt在排队长度上表现出显著的改进，尤其在负载较低的情况下，算法更快地收敛到最优排队长度，验证了理论结果。

**⚠️ 局限性**

限制在于当前框架尚未考虑多个队列的情况，未来的研究方向包括建立排队长度遗憾的下界、扩展框架以支持多个队列以及纳入操作约束（如最大等待时间约束）。

---

## 220. A Personalized and Adaptable User Interface for a Speech and Cursor Brain-Computer Interface

**arXiv ID:** 2601.19269 | [PDF](https://arxiv.org/pdf/2601.19269v1)

**作者:** Hamza Peracha `[一作]` (University of California), Nicholas S. Card `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过22个月的长期联合设计，开发了一个可个性化、可适应的脑机接口（BCI）用户界面，使严重瘫痪患者能够独立使用语音与光标BCI进行日常电脑交互和沟通。

**💡 创新点**

创新点在于：①将可调节控制模式（神经光标、眼动、语音解码）与多模态交互整合到单一界面；②实现了动态句子与单词级纠错功能，提升解码准确性；③采用可共享后端与图形节点的模块化架构，方便跨平台部署与功能迭代。

**🔧 技术方法**

技术包括：基于Transformer的神经语音解码器+n-gram +大语言模型（OPT 6.7b）进行语音解码；线性模型用于神经光标与点击解码；眼动追踪与神经光标双模态交互；Python‑Pyglet实现的可视化界面；分布式Linux节点架构实现低延迟实时控制。

**📊 数据集**

主要数据来自单一参与者T15的使用日志、解码语句及光标轨迹，持续超过4000小时；通过每六个月的问卷评估收集用户满意度与独立性数据。

**📈 对比分析**

与传统单一模式BCI或外部辅助系统相比，T15的平均句子完整度从初始40%提升至59%，纠错耗时虽从19→34→62秒增长，但整体沟通效率和用户满意度保持在4–5分。系统在家庭环境中的实用性、灵活性和可定制性被评为高度优异。

**⚠️ 局限性**

局限性包括：仅单一用户测试，缺乏大规模样本验证；需外科植入硬件，限制可及性；系统未实现自动上下文感知与性能自适应；缺乏与现有商业AAC或EEG‑BCI系统的直接比较。

---

## 221. A Multi-View Consistency Framework with Semi-Supervised Domain Adaptation

**arXiv ID:** 2601.19266 | [PDF](https://arxiv.org/pdf/2601.19266v1)

**作者:** Yuting Hong `[一作]` (Ningbo University), Chengbin Peng `[通讯]` (Ningbo University)

**通讯引用:** 1526 | [OpenAlex ID](https://openalex.org/A5101465378)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于多视图一致性的半监督域适应框架 MuVo，用来解决跨域情境下的类偏差问题。

**💡 创新点**

创新点在于将去偏学习、负标签学习与跨域亲和学习三者结合为多视图一致性训练，动态重分配预测概率并利用伪负标签增强模型鲁棒性。

**🔧 技术方法**

采用了伪标签去偏、负标签学习、EMA 置信度银行、跨域对比亲和损失、强弱数据增强、多视图一致性约束等技术。

**📊 数据集**

在 Office-Home 与 DomainNet 两大公开数据集的 1-shot 与 3-shot 半监督域适应任务上进行实验。

**📈 对比分析**

与多种 UDA/SSDA 基线（DANN、ENT、MME、CDAC、SLA 等）对比，MuVo 在两大数据集上均实现 10% 以上平均提升，分别达到 73.7%/76.7%（Office-Home）和 75.7%/77.4%（DomainNet）。

**⚠️ 局限性**

缺点包括训练时间较长，以及在不同域差异较大的情况下跨域特征对齐仍面临挑战，需要更高级的对齐方法。

---

## 222. Decoupled Split Learning via Auxiliary Loss

**arXiv ID:** 2601.19261 | [PDF](https://arxiv.org/pdf/2601.19261v1)

**作者:** Anower Zihad `[一作]` (Montclair State University), Chao Huang `[通讯]` (Montclair State University)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5040402082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种解耦合分割学习（DSL）框架，在切分层加入轻量辅助分类器，使客户端和服务器可独立使用本地损失进行训练，避免跨端梯度回传。

**💡 创新点**

通过在切点添加辅助网络实现训练解耦，消除了对后向梯度的依赖，仅传输前向激活，从而将通信量和客户端峰值内存降低约一半。

**🔧 技术方法**

利用辅助损失训练、单向激活通信、与传统基于BP的分割学习对比，并在ResNet‑110模型上实现切分。

**📊 数据集**

在CIFAR‑10与CIFAR‑100两个公开数据集上进行实验。

**📈 对比分析**

与传统基于BP的分割学习相比，DSL在测试准确率上几乎无差异，但通信量下降约50%，客户端峰值内存降低最多58%，训练时间略有增加。

**⚠️ 局限性**

辅助网络虽轻量，但仍带来额外计算开销；若切分层过浅，可能对模型性能有轻微影响，未来需进一步优化辅助网络以降低计算成本。

---

## 223. Riddle Quest : The Enigma of Words

**arXiv ID:** 2601.19273 | [PDF](https://arxiv.org/pdf/2601.19273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 224. Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization

**arXiv ID:** 2601.19220 | [PDF](https://arxiv.org/pdf/2601.19220v1)

**作者:** Dai Hai Nguyen `[一作]` (Hokkaido University), Hiroshi Mamitsuka `[通讯]` (Kyoto University)

**通讯引用:** 6802 | [OpenAlex ID](https://openalex.org/A5059001924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种加速的多目标分布优化算法 A-MWGraD，用于在 Wasserstein 空间上同时优化多目标分布函数；

**💡 创新点**

创新点在于将 Nesterov 加速思想迁移至概率空间的多目标梯度流，得到 O(1/t²) 或指数收敛速率，并给出了粒子实现与核逼近方案；

**🔧 技术方法**

采用 Wasserstein‑梯度、Damped Hamiltonian 动力学、SVGD 与 Blob 方法的核近似、连续时间流分析与离散时间粒子更新；

**📊 数据集**

在合成双高斯混合、Multi‑MNIST、Multi‑Fashion、Multi‑Fashion‑MNIST 等多目标采样与多任务学习数据集上进行实验；

**📈 对比分析**

与原 MWGraD、MOO‑SVGD、MT‑SGD 等方法对比，A‑MWGraD 在收敛速度、采样效率和测试准确率上均优于对手；

**⚠️ 局限性**

局限在于离散时间收敛率尚未证明，且理论假设需精确 Wasserstein 梯度，实际需核近似导致偏差；

---

## 225. Phase-Retrieval-Based Physics-Informed Neural Networks For Acoustic Magnitude Field Reconstruction

**arXiv ID:** 2601.19297 | [PDF](https://arxiv.org/pdf/2601.19297v1)

**作者:** Karl Schrader `[一作]` (National Institute of Informatics), Mirco Pezzoli `[通讯]` (Politecnico di Milano)

**通讯引用:** 446 | [OpenAlex ID](https://openalex.org/A5082962909)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种相位检索型 PINN，用于仅凭稀疏幅值测量估计声场幅值分布。

**💡 创新点**

创新点在于将幅值与相位分别用神经网络预测，并通过重建的复振幅满足 Helmholtz 方程，从而在无相位观测的条件下实现物理约束。

**🔧 技术方法**

采用随机 Fourier 特征映射的 MLP、物理约束损失（Helmholtz 方程残差）和自适应权重的 AdamW 优化器。

**📊 数据集**

使用图像源法在 3 m×4 m×6 m 房间内生成的合成数据集，随机采样 5/10/20/50 个测点并设置 64 个外部声源。

**📈 对比分析**

通过与最近邻插值基线和无物理损失的神经场方法对比，PRB‑PINN 在所有频率和测点数上均取得更低的幅值重建误差。

**⚠️ 局限性**

局限性包括相位重建不一定匹配真实相位；在高频或测点稀疏时误差上升；需要仔细调节物理损失权重。

---

## 226. ProMist-5K: A Comprehensive Dataset for Digital Emulation of Cinematic Pro-Mist Filter Effects

**arXiv ID:** 2601.19295 | [PDF](https://arxiv.org/pdf/2601.19295v1)

**作者:** Yingtie Lei `[一作]` (University of Macau), Xuhang Chen `[通讯]` (Huizhou University)

**通讯引用:** 687 | [OpenAlex ID](https://openalex.org/A5036370695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 ProMist‑5K 数据集，包含 20,000 对高分辨率图像，用于逼真模拟电影摄影中常用的 Pro‑Mist 扫光滤镜效果，并提出了基于场景参照线性空间的多尺度模糊仿真管线。

**💡 创新点**

创新点在于：①首次提供专门针对 Pro‑Mist 滤镜的物理一致性数据集；②设计了可控制滤镜密度和焦距的多层 Gaussian 模糊权重映射；③将光散射过程与场景参照空间结合，重现了真实滤镜的光晕、柔化与对比度降低；④提供统一、可调节的目标域，为后续学习和评估提供基准。

**🔧 技术方法**

使用技术包括：场景参照线性颜色空间转换、6 层逐渐增大的 Gaussian 模糊、可调权重融合、焦距相关的核尺寸调整、线性到显示参照的色调映射；在评估阶段使用的图像到图像翻译模型有 CycleGAN、Pix2Pix、DualGAN、UNIT、DRIT、LPTN 等。

**📊 数据集**

使用的数据集是 ProMist‑5K，涵盖 4 种滤镜配置（密度 1/2 与 1/8，焦距 20mm 与 50mm），每种配置 4,500 对训练样本 + 500 对测试样本，总计 20,000 对。

**📈 对比分析**

通过在原始→1/2@20mm 与 原始→1/8@20mm 两个典型任务上训练上述 6 种模型，评估指标包括 PSNR、SSIM、LPIPS 与 FID。结果显示 Pix2Pix（配对训练）在保留细节与光晕柔化方面表现最佳，CycleGAN 与 DualGAN（无配对）虽不如 Pix2Pix 细腻但仍能生成合理效果；总体而言，ProMist‑5K 能让不同模型在弱与强散射场景下都取得可比性能。

**⚠️ 局限性**

局限性包括：①数据集仅覆盖四种滤镜配置，缺乏更细粒度或多种材质的扩展；②多层模糊虽逼近光散射，但仍未完全模拟复杂粒子分布与颜色偏差；③模型在强光晕场景下仍可能产生模糊或色彩失真；④仅对单个摄像机设置进行仿真，未考虑真实镜头畸变与感光度变化等因素。

---

## 227. DiaDem: Advancing Dialogue Descriptions in Audiovisual Video Captioning for Multimodal Large Language Models

**arXiv ID:** 2601.19267 | [PDF](https://arxiv.org/pdf/2601.19267v1)

**作者:** Xinlong Chen `[一作]` (Institute of Automation, Chinese Academy of Sciences), Tieniu Tan `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DiaDem模型及DiaDemBench基准，旨在提升视听视频字幕中对话的精确描述（说话者归属和语句转写）。

**💡 创新点**

创新点在于：①构建专门评估说话者归属与转写准确度的评测框架；②利用高质量SFT数据与难度分层两阶段GRPO强化对话描述；③提出自适应合并匹配的动态规划匹配算法。

**🔧 技术方法**

技术包括多模态大型语言模型（基于AVoCaDO的架构）、SFT（对话数据自生成与人工校正）、GRPO（Group Relative Policy Optimization）强化学习、Levenshtein编辑距离与动态规划匹配、Gemini系列模型作为数据生成与评测工具。

**📊 数据集**

使用自制的DiaDemBench（1,039段含多方对话、重叠语音等场景），SFT训练集（70K高质量对话字幕+15K非对话字幕）以及3K人工标注的高难度对话样本。

**📈 对比分析**

与Gemini系列、video-SALMONN-2、OmniVinci、Qwen系列等14款模型在DiaDemBench上对比，DiaDem在说话者归属准确率提升2.3%、转写准确率提升4.5%，在单人/多人的场景均超越Gemini；在视频字幕整体质量上，DiaDem在SALMONN-2和UGC-VideoCap的评测中保持或略优于基线模型。

**⚠️ 局限性**

局限：在多方互动、重叠语音等复杂场景仍低于人类水平，偶尔会出现说话者错误归属或幻听，需人工复核。

---

## 228. A Reconfigurable Framework for AI-FPGA Agent Integration and Acceleration

**arXiv ID:** 2601.19263 | [PDF](https://arxiv.org/pdf/2601.19263v1)

**作者:** Aybars Yunusoglu `[一作]` (Purdue University), I. Can Dikmen `[通讯]` (Istinye University)

**通讯引用:** 217 | [OpenAlex ID](https://openalex.org/A5031540261)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 AI FPGA Agent 框架，将运行时 Q‑learning 代理与可参数化 FPGA 加速核心结合，实现深度网络推理的动态分区与加速。

**💡 创新点**

创新点在于将智能调度（Q‑learning）与硬件加速耦合，使得运行时能够实时决定哪些层落在 FPGA，显著降低 FPGA 开发门槛。

**🔧 技术方法**

使用技术包括 Q‑learning 代理、可配置的卷积/全连接流水线加速器、8 位定点量化、DMA/AXI/PCIe 数据流、双缓冲重叠、HLS 合成与 RTL 验证。

**📊 数据集**

实验使用约 10,000 张图像的 ResNet‑style 分类数据集（类似 MNIST/CIFAR‑10）进行评估。

**📈 对比分析**

与单线程 CPU（40.2 ms/图）和 FP16 GPU（6.1 ms/图）比较，FPGA 方案延迟 3.5 ms、吞吐 284.7 fps、功耗 28 W、能效 10.17 img/s/W，分类准确率仅比浮点基准低 0.2%。

**⚠️ 局限性**

局限性包括仅验证图像分类 CNN，未覆盖 Transformer 等大模型；依赖 8 位定点导致轻微精度下降；部分重配置与多任务支持尚未实现；需要进一步与主流 ML 框架深度集成。

---

## 229. Automatic Synthesis of Visualization Design Knowledge Bases

**arXiv ID:** 2601.19237 | [PDF](https://arxiv.org/pdf/2601.19237v1)

**作者:** Hyeok Kim `[一作]` (University of Washington), Jeffrey Heer `[通讯]` (University of Washington)

**通讯引用:** 24086 | [OpenAlex ID](https://openalex.org/A5090570042)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于数据驱动的自动化方法，从可排序的可视化设计对中提取候选特征，经过前向/后向选择后生成形式化知识库；

**💡 创新点**

创新点在于（1）摆脱手工规则、实现特征空间的自动生成；（2）结合结构与分布式预筛选指标，兼顾可解释性与有效性；（3）将提取的键值链直接渲染为Draco约束规则；

**🔧 技术方法**

使用深度遍历抽取键值链、k‑means/聚类确定数值边界、前向/后向特征选择、逻辑规则生成（Draco/ASP）以及线性分类器（Logistic/ SVM）评估特征；

**📊 数据集**

主要使用Zeng等30项图形感知实验中的1,384对设计数据（Baseline/ Zeng+）以及基于Gosling的10个基因组可视化（296对）作为训练与评估集；

**📈 对比分析**

与手工构建的Draco 2知识库进行对比；在Baseline上误差≤1%，在Zeng+上提升4–15%；在基因组数据上达到94–98%交叉验证准确率，保留集约82–98%；表明方法在多域下具有更高的泛化与准确性；

**⚠️ 局限性**

限制主要包括：需要人工标注设计对；数据集规模有限，难以覆盖所有视觉域；提取的低层特征在某些复杂图表上可能不够；缺少对动态交互与任务模型的深入支持。

---

## 230. VC-Bench: Pioneering the Video Connecting Benchmark with a Dataset and Evaluation Metrics

**arXiv ID:** 2601.19236 | [PDF](https://arxiv.org/pdf/2601.19236v1)

**作者:** Zhiyu Yin `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Peng Cheng Laboratory)

**通讯引用:** 6757 | [OpenAlex ID](https://openalex.org/A5100402996)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了视频连接（Video Connecting）任务，并构建了专门的基准 VC-Bench，用于系统评估跨片段视频的生成质量、起止一致性和过渡平滑度。

**💡 创新点**

创新点在于：①首次将视频连接定义为一个完整任务；②设计了三维评估指标（VQS、SECS、TSS）以多角度衡量生成效果；③提供了 1,579 条高质量、15 类 72 子类的视频数据集，填补了缺乏标准化评测的空白。

**🔧 技术方法**

技术手段包括：将起止片段映射至潜在空间并使用 DiT 进行条件扩散，结合 SLERP 进行跨场景平滑控制；同时利用 Aesthetic Score、SSIM、LPIPS、光流误差等现有指标构建新的综合评价体系；并对 6 个主流开源模型（Wan‑2.1、CogVideoX、Open‑Sora、Ruyi 等）进行统一评测。

**📊 数据集**

数据集为从 Pexels、Pixabay、Mixkit、YouTube 等公开平台抓取、过滤后得到的 1,579 条视频，视频时长 4–43 秒，分辨率均超过 720p，按 15 类 72 子类进行标注。

**📈 对比分析**

通过将负面指标转换为正向并归一化，对六个模型的 VQS、SECS、TSS 进行打分；结果显示 Wan‑2.1（14B）在总体得分上最高，表现出色的主体一致性、背景稳定性和过渡连贯性，但整体仍低于人类主观评分，尤其在起止一致性和过渡平滑度上存在明显不足。

**⚠️ 局限性**

局限性包括：只评估 5 秒长度的视频，无法覆盖更长序列；仅使用开源模型，未涵盖闭源模型的性能；跨场景连接仍是现有技术难点，模型往往产生“幻觉”或不自然的过渡。

---

## 231. iFAN Ecosystem: A Unified AI, Digital Twin, Cyber-Physical Security, and Robotics Environment for Advanced Nuclear Simulation and Operations

**arXiv ID:** 2601.19234 | [PDF](https://arxiv.org/pdf/2601.19234v1)

**作者:** Youndo Do `[一作]` (Georgia Institute of Technology), Fan Zhang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4281 | [OpenAlex ID](https://openalex.org/A5100403420)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个统一的iFAN生态系统，集成了3D数字孪生、物理仿真、人工智能、机器人操作、虚拟现实、强化学习、辐射仿真和硬件在环测试，支持核电站的预部署验证与网络安全演练。

**💡 创新点**

将高保真核电站物理仿真（GPWR）与UE5 3D数字孪生深度耦合，并首次在同一平台上同时实现机器人路径规划、遥操作、多模态感知与网络安全攻击检测；通过物理PLC与虚拟环境的实时双向同步，打造完整的网络-物理闭环测试床。

**🔧 技术方法**

采用Unreal Engine 5、OpenXR/Meta Quest 3 VR、OpenMC辐射模拟、Minimega网络仿真、Allen‑Bradley PLC、FactoryTalk Linx OPC、AirSim/Cosys AirSim、Rapyuta、Gymnasium API、Stable Baselines3、RLib、TorchRL、Python/C++、UnrealCV、OPC UA等技术栈。

**📊 数据集**

使用GPWR模拟器产生的传感器时序数据、辐射源的OpenMC仿真结果以及自定义的日志/历史数据库（InfluxDB）作为训练与验证数据集；未使用公开工业数据集，全部数据均来源于仿真生成。

**📈 对比分析**

通过强化学习导航实验（Double Q‑learning）实现了约96%成功率；在FDI攻击实验中展示了系统对异常温度传感器的可视化响应；与iFANnpp等现有工作比较，显示在机器人协作、网络安全演练和辐射监测方面的功能与性能显著提升，但尚未进行真实设备的跨验证。

**⚠️ 局限性**

主要局限在于：① 仍处于仿真阶段，缺乏真实核电站或实物机器人验证；② 物理模型与实际设备的误差可能导致仿真结果偏差；③ 网络安全攻击场景有限，仅覆盖了部分攻击类型；④ VR与RL训练中的延迟与精度受硬件限制；⑤ 数据集依赖仿真生成，未覆盖真实环境中的噪声与异常情况。

---

## 232. RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering

**arXiv ID:** 2601.19225 | [PDF](https://arxiv.org/pdf/2601.19225v1)

**作者:** Kaehyun Um `[一作]` (Yonsei University), Kyong-Ho Lee `[通讯]` (Yonsei University)

**通讯引用:** 2810 | [OpenAlex ID](https://openalex.org/A5044514062)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对小规模LLM（≤8B）设计了KG增强式检索-生成框架RPO‑RAG，提升知识图谱问答（KGQA）的推理能力。

**💡 创新点**

创新点包括：①语义感知的查询‑路径采样策略；②关系层次的加权偏好优化；③面向答案的提示设计，将检索路径聚合成答案中心的推理路径。

**🔧 技术方法**

技术实现：使用预训练语义模型（如SBERT）进行路径检索；动态束搜索与实体类型约束；基于强化学习的关系层次偏好优化；LoRA微调的小型LLM。

**📊 数据集**

数据集：WebQuestionsSP（WebQSP）与Complex WebQuestions（CWQ），均基于Freebase的多跳问答数据。

**📈 对比分析**

与多类基线（图结构推理、原始小LLM、LLM+KG）对比，RPO‑RAG在WebQSP上Hit+8.8% F1，CWQ上在≤8B模型中实现最高Hit与F1；在大模型（GPT‑4o‑mini）上缩小差距至3–4个百分点。

**⚠️ 局限性**

局限性：仍依赖语义匹配检索，检索误差或KG不完整会影响性能；对极端长路径（>4跳）处理仍有限；需手工设定阈值与聚类参数，缺乏完全自动化。

---

## 233. GhostUI: Unveiling Hidden Interactions in Mobile UI

**arXiv ID:** 2601.19258 | [PDF](https://arxiv.org/pdf/2601.19258v1)

**作者:** Minkyu Kweon `[一作]` (Seoul National University), Jinwook Seo `[通讯]` (Seoul National University)

**通讯引用:** 4274 | [OpenAlex ID](https://openalex.org/A5012388103)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过自动化UI探测工具对81款Android应用进行系统探测，收集并标注了1970条无视觉提示的隐藏交互实例，构建了GhostUI数据集，并利用该数据集对视觉语言模型（VLM）进行微调和评估。

**💡 创新点**

创新点在于首次专门记录并量化移动UI中的隐藏交互（如长按、双击、捏合等无视觉提示手势），提出了六种手势的正式分类体系，并证明包含视图层次结构、手势模式和任务描述等多模态信息能显著提升VLM在隐藏交互预测与UI状态迁移上的表现。

**🔧 技术方法**

使用技术包括：Appium驱动的UI探测与手势执行、Android XML视图层级到HTML-like结构的简化转换、基于LoRA的Qwen2.5‑VL微调以及GPT‑4o的Vision Fine‑tuning、手工与半自动化的标注与验证工具。

**📊 数据集**

使用的数据集为GhostUI，包含1970个实例，每个实例提供：前后截图、完整与简化的视图层级、手势元数据、任务描述以及应用元信息，覆盖六种手势类型。

**📈 对比分析**

对比方法：在所有信息完整的All‑inclusive配置下进行零样本与微调实验，评估手势分类准确率与IoU；相对基线（仅视觉输入）进行消融实验。结果显示，微调后GPT‑4o的手势分类准确率从51.1%提升至65.6%，Qwen2.5‑VL从33.3%提升至40.5%；视图层级信息缺失导致IoU大幅下降，证明其关键作用。

**⚠️ 局限性**

局限性包括：仅覆盖Android平台；仅包含六种基本手势，未涵盖拖拽、旋转等复杂交互；数据采集侧重主导航屏幕，缺少深层嵌套界面；标注过程仍需人工验证；未对人类可发现性和实际使用场景进行用户研究。

---

## 234. Residual Tokens Enhance Masked Autoencoders for Speech Modeling

**arXiv ID:** 2601.19399 | [PDF](https://arxiv.org/pdf/2601.19399v1)

**作者:** Samir Sadok `[一作]` (Inria), Xavier Alameda-Pineda `[通讯]` (Inria)

**通讯引用:** 3940 | [OpenAlex ID](https://openalex.org/A5066621495)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出RT‑MAE残差标记自动编码器，通过可训练的残差token补全显式属性无法覆盖的语音信息，提升重建质量并实现可控的语音生成与去噪。

**💡 创新点**

创新点在于（1）引入残差token以编码语音中未被pitch、content、speaker等显式属性捕捉到的细节（情感、微调音色等）；（2）通过dropout‑based正则化控制残差token的信息流，保持属性可控性与表达性平衡；（3）将噪声视作特殊残差，使用额外残差token实现去噪而不影响原始语音特征。

**🔧 技术方法**

技术方法包括：基于MAE的Transformer架构，交叉注意力提取残差token（受Perceiver启发）；HiFi‑GAN声码器合成波形；CLUB互信息估计用于噪声残差与语音残差的互信息约束；使用dropout阈值τ对残差token做训练时的随机遮蔽。

**📊 数据集**

使用的数据集有：LibriSpeech 360 Clean（训练与测试），EmoV‑DB（情感语料），LibriMix（混合噪声与干净语音），PTDB（用于音高操控实验）。

**📈 对比分析**

与原始AnCoGen、Conv‑TasNet、DCCRNet等方法对比，评估STOI、N‑MOS、SBS、COS、DNSMOS等指标。实验显示RT‑MAE在STOI、N‑MOS、SBS、COS以及噪声抑制相关的SIG、BAK、OVRL指标上均优于或接近其他模型，尤其在情感保留与说话人相似度上提升显著。

**⚠️ 局限性**

局限性包括：残差token对属性的解释性仍有限，模型对阈值τ的敏感度高，需精细调参；当τ过低或过高会导致可控性下降；目前验证范围局限于英语干净与噪声语料，未在多语言、多模态或大规模生成任务中进一步评估。

---

## 235. DSP-Reg: Domain-Sensitive Parameter Regularization for Robust Domain Generalization

**arXiv ID:** 2601.19394 | [PDF](https://arxiv.org/pdf/2601.19394v1)

**作者:** Xudong Han `[一作]` (University of Sussex), Yuguang Fang `[通讯]` (Hong Kong JC STEM Lab of Smart City)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于协方差的参数敏感性分析框架，并基于此设计了Domain‑Sensitive Parameter Regularization (DSP‑Reg)，通过软正则化鼓励模型使用跨域一致的参数，从而提升在未见域上的泛化性能。

**💡 创新点**

创新点在于：① 在参数层面进行协方差传播的敏感性量化；② 通过计算各参数在不同源域上的方差系数（coefficient of variation）来衡量跨域一致性；③ 用该量化结果作为权重在训练过程中动态惩罚域敏感参数，提供细粒度的正则化控制。

**🔧 技术方法**

主要技术包括：线性扰动模型、协方差传播、与Fisher信息的关联、梯度平方正则化、跨域方差系数的动态更新（T_update）以及使用预训练ResNet‑50与Adam优化器。

**📊 数据集**

实验数据集覆盖五个主流DG基准：PACs、VLCS、OfficeHome、TerraIncognita 和 DomainNet。

**📈 对比分析**

与众多DG方法（MMD、IRM、GroupDRO、SAM、Fish、GSAM、SAGM、GMDG、GGA等）在leave‑one‑domain‑out评估下进行对比，DSP‑Reg平均准确率达66.7%，在PACS 87.5%、VLCS 80.1%和DomainNet 45.6%等指标上均超过SOTA，对比提升约1.5–3.5%。

**⚠️ 局限性**

局限性包括：① 只考虑参数层的线性敏感性，未能捕捉非线性特征层的复杂变化；② 需要定期计算跨域方差系数，导致额外计算与内存开销；③ 对源域数量和分布差异高度敏感，若源域极少或分布极端不一致，效益有限；④ 与自监督、元学习等融合潜力尚未充分探索。

---

## 236. High-quality data augmentation for code comment classification

**arXiv ID:** 2601.19383 | [PDF](https://arxiv.org/pdf/2601.19383v1)

**作者:** Thomas Borsani `[一作]` (Free University of Bozen-Bolzano), Giuseppe Di Fatta `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5007549483)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Q‑SYNTH方法，用合成质量提升的过采样和数据增强来解决代码注释分类中的数据稀缺与类别不平衡问题。

**💡 创新点**

创新点在于结合BERT‑style掩码生成、多种质量评估（词汇相似度与语义相似度）以及质量阈值控制的三模组架构，显著提高合成样本的多样性与语义一致性。

**🔧 技术方法**

使用BERT‑like掩码语言模型生成合成注释、MiniLM句子变换器评估语义质量、以及SAMGS优化的多任务学习分类器。

**📊 数据集**

利用NLBSE’26挑战赛提供的Java、Python、Pharo三语言注释数据集（分别约6.6k、1.6k、1.1k条目）。

**📈 对比分析**

与基线STACC模型对比，Q‑SYNTH过采样在Pharo和Python上提升了F1得分，且高质量合成样本比大规模噪声样本更有效；但在Java上未超越基线。

**⚠️ 局限性**

主要限制是合成数据量不足以完全消除类别不平衡，导致增量改进有限，且过采样过程仍可能引入模型对合成模式的过拟合。

---

## 237. Pareto-Guided Optimization for Uncertainty-Aware Medical Image Segmentation

**arXiv ID:** 2601.19365 | [PDF](https://arxiv.org/pdf/2601.19365v1)

**作者:** Jinming Zhang `[一作]` (Xi'an Jiaotong-Liverpool University), Kaizhu Huang `[通讯]` (Duke Kunshan University)

**通讯引用:** 7885 | [OpenAlex ID](https://openalex.org/A5026022035)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种区域自适应课程学习框架，利用直觉模糊标签和 Pareto‑一致性损失，解决医学图像分割中的边界模糊和切片不一致问题，提升训练稳定性和分割精度。

**💡 创新点**

创新点包括：① 引入像素级 Intuitionistic Fuzzy Label 使边界处标签平滑；② 结合 Pareto‑consistent 机制的 Region‑wise Curriculum Learning，逐步从低不确定区到高不确定区学习；③ 将两可学习参数 ρ1、ρ2 嵌入损失，实现自适应 Pareto 路径和动态权重调整。

**🔧 技术方法**

技术手段：直觉模糊集、模糊辅助损失、Dice 损失、Pareto‑consistent 动态权重、学习率余弦衰减、3D 变形网络（mmFormer、SwinUNETR、VNet）、多模态/单模态/缺失模态实验、训练稳定性分析。

**📊 数据集**

使用 BraTS18（多模态脑肿瘤）和 Pretreat‑MetsToBrain‑Masks（脑转移瘤）两大 3D 影像分割基准。

**📈 对比分析**

与 mmFormer、SwinUNETR、VNet、U‑HeMIS、U‑HVED 等基线对比，在全模态、单模态和缺失模态条件下均取得显著 Dice 分数提升（平均提升约 4–8 分），且训练曲线更平滑、梯度波动更小。

**⚠️ 局限性**

局限性：当前策略仅应用于 mmFormer 的融合分支，未覆盖其多分支结构，导致某些子区域提升有限；实验规模与跨数据集泛化仍待进一步验证。

---

## 238. Revisiting Parameter Server in LLM Post-Training

**arXiv ID:** 2601.19362 | [PDF](https://arxiv.org/pdf/2601.19362v1)

**作者:** Xinyi Wan `[一作]` (Sea AI Lab), Jialin Li `[通讯]` (National University of Singapore)

**通讯引用:** 2218 | [OpenAlex ID](https://openalex.org/A5108050353)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 On-Demand Communication (ODC)，将 FSDP 的集体通信改为点对点，以降低同步粒度并提升设备利用率。

**💡 创新点**

创新点在于把参数服务器 (PS) 的去耦进度与 FSDP 的内存分片结合，形成去中心化的 PS，并把同步从层级降到 mini‑batch 级别，显著减少因序列长度不均导致的 straggler。

**🔧 技术方法**

使用了 RDMA（CUDA IPC、NVSHMEM）、Triton‑Distributed、PyTorch FSDP、FlashAttention 等技术实现点对点通信与分层打包。

**📊 数据集**

实验数据集包括 LongAlign、SWE‑Smith、AIME prompts 以及 DeepSeek‑R1‑Distill‑Qwen 系列模型。

**📈 对比分析**

与传统 FSDP 集体通信基线以及多种打包方案（LocalSort、LB‑Micro、LB‑Mini）进行对比，ODC 在 SFT 任务中最高可提升 36% 的吞吐量，RL 任务约 10%，并在不同模型规模、批次大小和序列长度下保持稳定的性能提升。

**⚠️ 局限性**

主要局限包括跨节点通信时 ODC 的带宽低于集体通信、仍需在 mini‑batch 边界保持同步、未实现异步或容错机制，且在极大规模多节点环境下可能出现通信瓶颈。

---

## 239. GeoSSA: Geometric Sparrow Search Algorithm for UAV Path Planning and Engineering Design Optimization

**arXiv ID:** 2601.19346 | [PDF](https://arxiv.org/pdf/2601.19346v1)

**作者:** Junhao Wei `[一作]` (Macao Polytechnic University), Xu Yang `[通讯]` (Macao Polytechnic University)

**通讯引用:** 2801 | [OpenAlex ID](https://openalex.org/A5100462079)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了Geometric Sparrow Search Algorithm（GeoSSA），通过改进初始化、生产者更新和边缘麻雀更新三大策略，提升SSA在复杂优化任务中的性能；

**💡 创新点**

①利用Good Nodes Set实现均匀分布的种群初始化；②引入正弦-余弦混合更新和自适应惯性权重改进生产者位置更新；③采用三角步行扰动机制增强边缘麻雀的局部搜索与跳出局部最优的能力；

**🔧 技术方法**

Meta‑heuristic optimization, geometric initialization, sine‑cosine search operators, triangular walk perturbation, adaptive inertia weighting, penalty function constraint handling, GPU‑style parallelization (future work)。

**📊 数据集**

23个经典单目标基准函数（Sphere、Rastrigin、Ackley等）、三维UAV路径规划环境（含障碍物的三维网格）、四个工程设计问题（Corrugated Bulkhead、Piston Lever、Reactor Network、Industrial Refrigeration System）。

**📈 对比分析**

与8个竞争算法（AROA、WOA、IWOA、MWOA、IPSO、ISSA、SSA）在同等迭代次数（T=500）和种群规模（N=30）下进行多次独立实验。使用平均适应度、标准差、Wilcoxon符号秩检验、Friedman 排名和整体有效率（OE）评估。GeoSSA在绝大多数基准函数上获得最优或近优结果，OE达95.65%，在UAV路径规划和工程设计任务中表现出最快收敛、最高稳定性和最佳解质量。

**⚠️ 局限性**

在多峰、复杂约束问题上仍可能遇到局部最优陷阱；对多目标、离散或动态优化问题尚未验证；并且对大规模高维问题的可扩展性需进一步评估。

---

## 240. Robust Uncertainty Estimation under Distribution Shift via Difference Reconstruction

**arXiv ID:** 2601.19341 | [PDF](https://arxiv.org/pdf/2601.19341v1)

**作者:** Xinran Xu `[一作]` (Nanyang Technological University), Xiuyi Fan `[通讯]` (Nanyang Technological University)

**通讯引用:** 970 | [OpenAlex ID](https://openalex.org/A5101917609)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于两层重构差异的深度学习模型不确定性估计方法DRUE，改进了传统重构误差导致的信息丢失问题；

**💡 创新点**

创新点在于通过比较同一输入在不同深度特征空间下的重构结果差异，既降低了信息丢失的影响，又保持对预测不确定性的敏感性；

**🔧 技术方法**

核心技术包括在分类模型的中间层与最后层各附加一个解码器，训练解码器以最小化重构误差，并利用两重构结果的差值作为不确定性评分；

**📊 数据集**

实验使用视网膜病变数据集Glaucoma-Light V2作为训练集，并在PAPILA、ACRIMA、HAM10000和CIFAR‑10等五个不同域的测试集上进行OOD检测；

**📈 对比分析**

与熵、MC Dropout、PostNet、DEC和BNN等基线方法比较，DRUE在AUC和AUPR指标上均取得领先，尤其在不同程度的域移位场景下表现更为稳定；

**⚠️ 局限性**

局限性包括目前仅验证在分类任务上，对回归任务的适用性尚未探究；同时对大型网络和高分辨率图像的训练成本可能更高。

---

## 241. CaseMaster: Designing and Evaluating a Probe for Oral Case Presentation Training with LLM Assistance

**arXiv ID:** 2601.19332 | [PDF](https://arxiv.org/pdf/2601.19332v1)

**作者:** Yang Ouyang `[一作]` (ShanghaiTech University), Quan Li `[通讯]` (ShanghaiTech University)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5100689370)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于大型语言模型（LLM）的交互式训练探针 CaseMaster，用以帮助医学生在口头病例报告（OCP）中进行准备、练习与反思。

**💡 创新点**

创新点在于将 LLM 与结构化教学活动（SOAP 框架、预设提示、分阶段流程）相结合，提供即时可视化反馈与可定制提示，并在医学教育场景中首次系统评估 LLM 对 OCP 训练的效能。

**🔧 技术方法**

使用了 OpenAI GPT‑4o 进行文本生成与评估，前端采用 Vue 3、后端使用 Flask；通过 Prompt 模板、自动化评分 JSON、可视化 UI 以及多功能交互设计实现。

**📊 数据集**

基于 30 份医学案例（病例记录、口头报告、参考答案）和 4 份评估案例进行训练与评测；案例按难度分级并通过规则化去标识化后使用。

**📈 对比分析**

实验采用对照设计（12 名学生），与传统自助工具对比，使用 Wilcoxon 符号秩检验；结果显示在差异诊断清晰度显著提升（p≈0.042），其余维度呈正向趋势；工作负荷差异无显著性；专家评估中 LLM 评分与教师评分 ICC=0.88，表明评分可靠性高。

**⚠️ 局限性**

局限性包括 LLM 生成信息可能存在不准确或专业化过度，需要学生进行批判性核查；样本量小且单一科室、单一医院；缺乏多模态（图像、视频）支持与长期学习效果跟踪；对早期学习者的适用性和安全性尚未充分验证。

---

## 242. Instance-Guided Radar Depth Estimation for 3D Object Detection

**arXiv ID:** 2601.19314 | [PDF](https://arxiv.org/pdf/2601.19314v1)

**作者:** Chen-Chou Lo `[一作]` (KU Leuven), Patrick Vandewalle `[通讯]` (KU Leuven)

**通讯引用:** 1470 | [OpenAlex ID](https://openalex.org/A5061245134)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于实例分割引导的雷达扩展方法InstaRadar，并将雷达引导深度估计模块RCDPT集成到BEVDepth框架中，以提升单目深度估计和三维目标检测性能。

**💡 创新点**

创新点在于①利用OneFormer实例掩码对雷达点进行对象级扩展，实现稀疏雷达的密度提升和语义对齐；②通过将预训练的雷达引导深度网络RCDPT替换BEVDepth的默认深度模块，结合InstaRadar提供的高质量雷达输入，实现显式深度监督并显著提升深度特征质量。

**🔧 技术方法**

使用了Transformer‑based雷达引导深度估计（RCDPT）、实例分割模型OneFormer、BEVDepth框架的视角变换与BEV编码、雷达与图像的跨模态特征重组与体素池化等技术。

**📊 数据集**

在nuScenes多模态数据集上进行实验，该数据集包含6个摄像头、5个雷达和32束激光雷达。

**📈 对比分析**

与BEVDepth基线、其他深度估计方法（S2D、DORN_radar、JBF）以及雷达‑相机融合模型（CRN、CRAFT等）进行对比；在nuScenes验证集上，InstaRadar+RCDPT在单目相机+雷达模式下取得mAP 0.355、NDS 0.457，较基线提升约5.7%和9.9%，但仍落后于专用雷达‑相机融合网络。

**⚠️ 局限性**

雷达仅被用作深度引导而非独立特征流，缺乏专门的雷达BEV分支和时序信息，导致与先进的雷达‑相机融合架构相比性能仍有差距。

---

## 243. Balancing Sustainability And Performance: The Role Of Small-Scale Llms In Agentic Artificial Intelligence Systems

**arXiv ID:** 2601.19311 | [PDF](https://arxiv.org/pdf/2601.19311v1)

**作者:** Anh Khoa Ngo Ho `[一作]`, Boris Gamazaychikov `[通讯]` (Salesforce)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5106655097)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在真实多代理系统中，对比不同规模的开源LLM与闭源GPT‑4o的能耗、解码延迟和输出质量，寻找既可持续又保持性能的模型配置。

**💡 创新点**

创新点在于：①将环境影响、用户体验和输出质量三维目标统一量化并构造总体指标OM；②系统评估压缩技术（量化、蒸馏）对能耗、质量的综合影响；③首次在大规模真实对话数据上对开源与闭源模型进行全面对比。

**🔧 技术方法**

技术方法包括：ML‑Energy Benchmark + Zeus 进行能耗测量；vLLM 推理引擎；AWQ、GPTQ 量化；知识蒸馏；Ragas LLM‑as‑Judge 评估；混合专家 Qwen3‑30B‑A3B 等模型。

**📊 数据集**

使用企业内部多代理系统收集的约1000条对话请求（平均约8000 tokens，最长25500 tokens）作为评测数据集。

**📈 对比分析**

通过对 Energy（J）、Decode Latency（s）、F1/LLM‑Judge 等指标进行比对，并计算 OM 进行排名。结果显示：Qwen3‑30B‑A3B、Falcon‑10B 等开源模型在能耗降低约70% 的同时，输出质量与 GPT‑4o 接近；部分模型在延迟/能耗上甚至优于闭源基准。

**⚠️ 局限性**

限制包括：评测任务单一（仅验证代理的幻觉检测）；数据来源单一企业场景，缺乏多样性；闭源 GPT‑4o 的能耗和延迟为估算；未进行人工质量评测；自托管模型延迟高于 API；实验仅在 A100 GPU 上完成，硬件与云环境可扩展性待验证。

---

## 244. Beyond Shadows: A Large-Scale Benchmark and Multi-Stage Framework for High-Fidelity Facial Shadow Removal

**arXiv ID:** 2601.19309 | [PDF](https://arxiv.org/pdf/2601.19309v1)

**作者:** Tailong Luo `[一作]` (New York Institute of Technology), Xuhang Chen `[通讯]` (Huizhou University)

**通讯引用:** 687 | [OpenAlex ID](https://openalex.org/A5036370695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面部阴影移除数据集ASFW，并提出了三阶段的Face Shadow Eraser (FSE) 方法。

**💡 创新点**

提供首个大规模真实世界阴影/无阴影配对数据集ASFW，及结合掩模生成、粗细分层处理的轻量级FSE框架。

**🔧 技术方法**

使用手工Photoshop生成阴影/移除、MaskGuideNet/CoarseGenNet/RefineFaceNet三阶段网络，融合动态卷积、AggBlock、AHSWA、光照细化等技术。

**📊 数据集**

使用ASFW作为主数据集，SFW视频集为原始素材；训练时加入FFHQ与合成阴影；评估用UCB与ASFW。

**📈 对比分析**

与Lyu、FSRNet、CIRNet等方法在PSNR/SSIM/LPIPS等指标上进行对比，FSE在ASFW上取得最高分，显示出更好的阴影去除效果。

**⚠️ 局限性**

数据集制作成本高、规模有限，模型在极端光照或遮挡条件下的鲁棒性尚待提升，且仍依赖监督学习。

---

## 245. Voice-Based Chatbots for English Speaking Practice in Multilingual Low-Resource Indian Schools: A Multi-Stakeholder Study

**arXiv ID:** 2601.19304 | [PDF](https://arxiv.org/pdf/2601.19304v1)

**作者:** Sneha Shashidhara `[一作]` (Centre for Social and Behaviour Change), Sharath Chandra Guntuku `[通讯]` (University of Pennsylvania)

**通讯引用:** 3920 | [OpenAlex ID](https://openalex.org/A5010646067)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在四所低资源的德里私立学校进行为期六天的实地研究，部署并评估了面向英语口语练习的语音聊天机器人，收集了学生、教师和校长的访谈、观察与使用日志，分析其对学生口语信心、技术体验和教师教学整合的影响。

**💡 创新点**

首次在多利益相关者视角下系统探讨低资源多语环境中语音聊天机器人的可行性与挑战，提出了“共情对话 vs 课程框架”的设计张力、以学生自信为核心的体验曲线、以及面向低资源社区的可落地实施清单和跨平台集成路径。

**🔧 技术方法**

使用基于React的前端Web应用，后端托管于AWS S3+CloudFront；语音识别采用Whisper‑1，生成式对话使用GPT‑4o‑mini；语音输出通过Google TTS；系统通过OpenAI Moderation API过滤内容。

**📊 数据集**

实验样本为23名七、八年级学生（13女10男）、6名英语教师、5名校长，覆盖4所学校；数据来源包括日常使用的音频转录、访谈记录、观察记录和教师反馈日志。

**📈 对比分析**

该研究主要采用定性分析方法（主题编码、情感轨迹分类、利益相关者三角分析），未进行量化对比或基准测试；主要结论是学生口语自信显著提升（从短句到主动提问），但技术障碍导致信心曲线不稳定，整体效果无法量化评估。

**⚠️ 局限性**

局限性包括：研究周期仅六天，缺乏前后测评估；所有使用场景受研究者辅佐的监督影响，真实课堂与家用体验可能不同；样本量小、单一地区，缺乏跨地区验证；技术层面仍存在ASR误识、延迟、语音速度不匹配等瓶颈，影响长期可持续性。

---

## 246. Tri-Reader: An Open-Access, Multi-Stage AI Pipeline for First-Pass Lung Nodule Annotation in Screening CT

**arXiv ID:** 2601.19380 | [PDF](https://arxiv.org/pdf/2601.19380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 247. SEAFormer: A Spatial Proximity and Edge-Aware Transformer for Real-World Vehicle Routing Problems

**arXiv ID:** 2601.19395 | [PDF](https://arxiv.org/pdf/2601.19395v1)

**作者:** Saeed Nasehi Basharzad `[一作]`, Egemen Tanin `[通讯]` (University of Melbourne)

**通讯引用:** 2089 | [OpenAlex ID](https://openalex.org/A5035347745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SEAFormer，一种结合节点与边信息的 Transformer，专为真实车辆路径规划（RWVRP）设计，能够在大规模（1,000+ 节点）实例上高效求解。

**💡 创新点**

创新点：① Clustered Proximity Attention（CPA）通过局部聚类将注意力复杂度从 O(n²) 降至 O(n)，同时保留全局视角；② 轻量化的 Edge‑Aware 模块通过残差融合捕获边对特征，提升信息利用效率和收敛速度。

**🔧 技术方法**

技术手段：Transformer 架构、近似注意力（CPA）、残差融合的边信息模块、强化学习/生成式优化训练框架。

**📊 数据集**

数据集：四类 RWVRP 变体（含时间窗、补给/充电、非对称旅行成本等），节点规模从数百到超过 1,000；对比了经典 VRP 基准（CVRP、VRPTW 等）。

**📈 对比分析**

方法对比：与现有神经网络方法（e.g., Pointer Network, TSP‑Transformer）以及传统优化求解器进行对比；SEAFormer 在所有四种 RWVRP 上均实现了更低成本、较快收敛，并成为首个能够有效处理 1,000+ 节点 RWVRP 的神经方法。

**⚠️ 局限性**

局限性：对极大规模实例仍有显著计算负担；模型训练依赖大量标注数据，对动态/实时约束的适应性有限；在某些极端约束组合下性能提升有限。

---

## 248. Bridging the Socio-Emotional Gap: The Functional Dimension of Human-AI Collaboration for Software Engineering

**arXiv ID:** 2601.19387 | [PDF](https://arxiv.org/pdf/2601.19387v1)

**作者:** Lekshmi Murali Rani `[一作]` (Chalmers University of Technology and University of Gothenburg), Robert Feldt `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

通过半结构化访谈分析软件从业者对人机协作中的社会情感差距的认知，并提出功能等价框架

**💡 创新点**

提出“功能等价”概念，将人类社会情感特质映射为技术可实现的功能，帮助AI在软件工程中实现有效协作

**🔧 技术方法**

采用定性研究方法（主题分析）和访谈，不涉及具体算法实现

**📊 数据集**

访谈样本共10名软件从业者

**📈 对比分析**

未做性能对比，仅通过访谈验证框架的可行性与认知一致性

**⚠️ 局限性**

样本量有限、仅适用于软件工程领域、缺乏对实际AI工具性能的量化评估，结果迁移性受限

---

## 249. MIRAGE: Enabling Real-Time Automotive Mediated Reality

**arXiv ID:** 2601.19385 | [PDF](https://arxiv.org/pdf/2601.19385v1)

**作者:** Pascal Jansen `[一作]` (Ulm University), Enrico Rukzio `[通讯]` (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个开源工具MIRAGE，能够在真实车辆中实时实现汽车媒介现实（AMR），支持AR、DR和ModR三种效果并可自由组合；

**💡 创新点**

首次提供完整的AMR框架，整合15种可自定义的视觉效果，并利用Unity推理引擎在车辆内的RGB摄像头流上实现零延迟的实时推理，填补了模拟到现实的技术鸿沟；

**🔧 技术方法**

基于Unity、Unity推理引擎、YOLO11语义分割、DepthAnythingV2深度估计、MI‑GAN图像修复以及Compute Shader + C#脚本的模块化管线；

**📊 数据集**

主要使用COCO数据集训练的YOLO11模型（支持自定义训练），DepthAnythingV2基于公开的深度数据，MI‑GAN使用公开模型；

**📈 对比分析**

在四种GPU配置（RTX 3080、RTX 4070 Ti、RTX 4080、RTX 4090）下进行基准测试，平均帧率在20–35 FPS之间，DepthAnything为主要瓶颈；在9位专家用户研究中，系统可用性SUS≈66.9、NASA‑TLX≈7.8，技术实现被认为可接受，但对DR效果的稳定性和延迟存在顾虑；

**⚠️ 局限性**

缺乏对象持久追踪和3D重建导致对齐误差；深度估计模型慢、生成修复可能产生错误；未验证驾驶安全性；UI不直观、需要改进；仅依赖单摄像头，无法实现完整的DR（如真实隐藏视角）等限制。

---

## 250. On the Analysis of Platooned Vehicular Networks on Highways

**arXiv ID:** 2601.19370 | [PDF](https://arxiv.org/pdf/2601.19370v1)

**作者:** Kaushlendra Pandey `[一作]` (Indian Institute of Technology Kanpur), Abhishek K. Gupta `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 5770 | [OpenAlex ID](https://openalex.org/A5017906439)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文利用随机几何分析，研究了高速公路上车队化车辆网络中RSU负载分布、V2V连通性、V2I覆盖概率和速率覆盖等性能指标；

**💡 创新点**

创新点在于：① 将车队化流量建模为一维母子泊松聚簇过程(MCP)，并推导典型与标记RSU的负载分布、均值、方差、偏度等闭式结果；② 通过负载分布进一步分析V2V连通度、V2I覆盖概率与速率覆盖的元分布（Meta Distribution），揭示车队化对网络负载、干扰与可靠性的双重影响；

**🔧 技术方法**

主要技术手段为一维泊松点过程与聚簇过程的随机几何建模、概率生成函数（PGF）与概率质量函数（PMF）推导、干扰的拉普拉斯变换、Gil‑Pelaez 逆变换求元分布，以及数值仿真验证；

**📊 数据集**

本文没有使用具体公开数据集，而是通过设定参数（RSU密度、车队半径、车流密度等）构造仿真场景，以验证理论推导；

**📈 对比分析**

通过理论推导与蒙特卡洛仿真对比，验证了负载分布、覆盖概率、速率覆盖等指标的准确性；实验结果显示，在车队化情形下RSU负载波动更大、离线概率更高，但由于干扰减少，覆盖概率提升；速率覆盖则略低，元分布显示两种流量模型的可靠性差异；

**⚠️ 局限性**

局限性包括：① 仅考虑单车道一维模型，未涵盖多车道或城市复杂几何；② RSU激活状态通过近似独立PPP估计，实际激活可能更复杂；③ 没有考虑车辆速度变化、队列排队行为及动态干扰管理；④ 仅关注基于RSU的V2I与直接V2V，未涉及混合蜂窝/DSRC多网络互操作。

---

## 251. Self-Supervised Path Planning in Unstructured Environments via Global-Guided Differentiable Hard Constraint Projection

**arXiv ID:** 2601.19354 | [PDF](https://arxiv.org/pdf/2601.19354v1)

**作者:** Ziqian Wang `[一作]` (Tsinghua University), Zhen Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 71793 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种自监督的路径规划框架，结合全局潜在场指导与可微硬约束投影，实现嵌入式机械系统在无结构环境中的安全导航。

**💡 创新点**

创新点在于：①引入全局导向人工势场（G‑APF）提供稠密监督，消除专家演示需求；②设计可微硬约束投影层（AdaNP）利用滑动变量与LSE平滑实现高维非线性约束的迭代逼近；③采用两阶段课程学习（软约束预训练→硬约束细化），提升收敛稳定性。

**🔧 技术方法**

核心技术包括：全局势场生成（多源波前传播）、LSE平滑的可微碰撞表示、可微等式约束重构与牛顿投影、软/硬约束损失、基于梯度的可微投影求解器，以及对抗性数据增强的训练流程。

**📊 数据集**

数据集为200,000条人工合成场景（40×20 m区域，随机生成8个四边形障碍物），按6:3:1划分为训练/验证/测试集；测试集包含20,000个独立案例。

**📈 对比分析**

与Hybrid A*、Informed RRT*、NMPC以及基于IL的软约束网络对比，本文方法在20k测试场景中取得88.75%成功率，平均路径长度33.30 m，推理延迟0.0939 s（Jetson Orin NX），显著优于传统搜索和纯学习方法；同时保持了与NMPC相近的动力学可行性和轨迹平滑度。

**⚠️ 局限性**

局限性包括：仅针对静态障碍，动态环境尚未评估；投影层的收敛依赖于初始粗略路径，若全局势场预设错误会导致失败；高λ_soft会导致局部极小陷阱；并且在极端复杂环境下迭代次数上限仍可能影响实时性。

---

## 252. GraphSB: Boosting Imbalanced Node Classification on Graphs through Structural Balance

**arXiv ID:** 2601.19352 | [PDF](https://arxiv.org/pdf/2601.19352v1)

**作者:** Zhixiao Wang `[一作]` (China University of Mining and Technology), Philip S Yu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 GraphSB 框架，通过结构平衡（SB）先解决图结构不平衡，再结合节点合成提升不平衡节点分类效果。

**💡 创新点**

核心创新是 SB 两阶段结构优化：(1) 基于硬样本挖掘的自适应边增强，补偿少数类节点的稀疏邻域；(2) 采用稀疏迭代关系扩散，在全局捕获高阶结构依赖；并证明其可作为任何现有方法的插件。

**🔧 技术方法**

技术包括：双视角硬样本挖掘、相似性约束自适应边增补、随机结构扰动与稀疏迭代扩散、与 GraphMixup 合成节点的联合训练。

**📊 数据集**

使用八个基准数据集：Cora、Citeseer、PubMed、Amazon‑Photo、Amazon‑Computers、Wiki‑CS、ogbn‑arxiv、CoraFull，涵盖从小型到大规模、人工与自然不平衡场景。

**📈 对比分析**

与 11 个数据级与算法级基线（如 Vanilla、Re‑weight、SMOTE、GraphSMOTE、GraphENS、GraphMixup、IceBerg 等）对比，GraphSB 在 Acc 与 Macro‑F1 上均超过所有方法，平均提升约 4.57%；在极端不平衡数据上也保持优势。

**⚠️ 局限性**

局限性主要包括：对超参数（扩散步数 K、系数 α）敏感；在结构极度稀疏的图上可能需更多边增补；仅在节点级分类场景验证，其他任务如图分割或链接预测仍待探索。

---

## 253. Metric $k$-clustering using only Weak Comparison Oracles

**arXiv ID:** 2601.19333 | [PDF](https://arxiv.org/pdf/2601.19333v1)

**作者:** Rahul Raychaudhury `[一作]` (Duke University), Stavros Sintos `[通讯]` (University of Illinois Chicago)

**通讯引用:** 257 | [OpenAlex ID](https://openalex.org/A5015883931)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在噪声四元组查询模型下，构造 (k,p)-聚类的核心集并给出近似聚类算法。

**💡 创新点**

提出仅使用四元组查询即可得到 O(1)-近似聚类、O(k n) 核心集，并在双曲维数有限时将查询复杂度降至 O((n+k^2)·log n)，还能进一步实现 (1+ε)-近似。

**🔧 技术方法**

采用递归采样、近似排序、抗噪声排序（AdvSort）、近似最近邻、分层抽样与冲突图染色等技术。

**📊 数据集**

在合成二维高斯簇、Adult（8维）和信用卡违约（9维）数据集上进行实验。

**📈 对比分析**

与 k‑means++ 基线和理想聚类比较，实验表明核心集仅占原始点的 1.9%，聚类成本在 7% 内，远优于基线。

**⚠️ 局限性**

仍需在非度量图、三元组查询等更广模型下验证，并对大规模数据的实际实现效率及噪声阈值上限做进一步研究。

---

## 254. Generalizable IoT Traffic Representations for Cross-Network Device Identification

**arXiv ID:** 2601.19315 | [PDF](https://arxiv.org/pdf/2601.19315v1)

**作者:** Arunan Sivanathan `[一作]` (University of New South Wales), Hassan Habibi Gharakaheili `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了无标签 IoT 流量的压缩表示，使用冻结的编码器 + 简单分类器实现设备类型识别；

**💡 创新点**

提出把表示学习与分类分离、利用变分自编码器（VAE）与实体嵌入提升泛化，并统一冻结-编码器评估协议，证明大模型并非必需；

**🔧 技术方法**

采用卷积自编码器、变分自编码器、实体嵌入等无监督技术，并与 ET‑BERT、NetMamba 等预训练模型对比；

**📊 数据集**

使用 18M+ 条真实 IoT 流量，来自 2016 年实验室（DATA16）、2025 年校园实验室（DATA25v1）以及另一校园网络（DATA25v2）；

**📈 对比分析**

通过冻结编码器 + 轻量级全连接分类器，评估同域、跨域、跨环境的宏 F1 和准确率；VAE/实体嵌入模型宏 F1>0.9，而大型预训练模型仅达 0.68‑0.75；

**⚠️ 局限性**

对长连接拆分与分片处理仍有限；仅关注设备类型分类，对复杂多变设备（如 IT 类、环境相关网关）性能不足；

---

## 255. LightSBB-M: Bridging Schrödinger and Bass for Generative Diffusion Modeling

**arXiv ID:** 2601.19312 | [PDF](https://arxiv.org/pdf/2601.19312v1)

**作者:** Alexandre Alouadi `[一作]`, Nizar Touzi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 LightSBB‑M 算法，在少量迭代内求解 Schrödinger‑Bass 桥（SBB）问题，得到最优传输计划并实现生成建模。

**💡 创新点**

创新点在于联合优化漂移与波动率，引入可调参数 β 在 Schrödinger Bridge 与 Bass Martingale 之间插值；利用 SBB 的对偶表述得到解析式，实现无 SDE 采样且对高维稳健。

**🔧 技术方法**

使用对偶 SBB 系统、Gaussian‑mixture 势能参数化、Bridge Matching 损失、神经网络逆映射训练等技术。

**📊 数据集**

在低维合成数据（8 高斯、moons）、真实图像（FFHQ 成人→儿童）以及高维图像生成实验中进行验证。

**📈 对比分析**

与多种 SOTA SB、Diffusion、Flow 方法比较，2‑Wasserstein 距离平均下降约 19%，在图像翻译任务中视觉质量与多样性均优于 LightSB‑M。

**⚠️ 局限性**

迭代次数经验上需 5 步，未给出理论收敛证明；对 β 取值与计算量敏感，极端噪声或高 β 时性能略退化。

---

## 256. Formula-One Prompting: Adaptive Reasoning Through Equations For Applied Mathematics

**arXiv ID:** 2601.19302 | [PDF](https://arxiv.org/pdf/2601.19302v1)

**作者:** Natapong Nitarach `[一作]` (SCB 10X), Kunat Pipatanakul `[通讯]` (SCB 10X)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种一次性两阶段的公式先行提示（Formula‑One Prompting, F‑1）方法，先将问题转化为数学方程，再根据方程自动选择直接求解、链式思考或程序式求解。

**💡 创新点**

创新点在于将方程生成作为中间表示，利用方程结构来指导求解策略，从而在单次调用中实现自适应求解。

**🔧 技术方法**

采用大语言模型（GPT‑5、Gemini 2.5 Pro、Qwen3‑30B 等）进行提示生成，并结合方程推导、策略选择与验证步骤。

**📊 数据集**

在四个主导领域的四个基准集上评估：IMO‑Bench（竞赛数学）、OlympiadBench（含物理）、FinanceMath（金融）、AICrypto（密码学）。

**📈 对比分析**

与 Zero‑Shot、Chain‑of‑Thought（CoT）和 Program‑of‑Thought（PoT）单调用基线对比，F‑1 在所有模型和基准上平均提升 5.76%（CoT）和 8.42%（PoT），在应用域尤其显著（如 FinanceMath 提升 13.30%）。

**⚠️ 局限性**

局限包括对小规模模型（≤7B）支持不足、对非方程驱动任务效果有限，以及缺乏多轮调用或回溯机制以处理策略错误。

---

## 257. Innovator-VL: A Multimodal Large Language Model for Scientific Discovery

**arXiv ID:** 2601.19325 | [PDF](https://arxiv.org/pdf/2601.19325v1)

**作者:** Zichen Wen `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (DP Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 Innovator‑VL，一款 8B 参数的科学多模态大语言模型，既能处理通用视觉任务，又能在化学、物理、微观结构等科学领域实现高水平推理。

**💡 创新点**

创新点在于：① 引入 RICE‑ViT 区域感知视觉编码器与 PatchMerger 视觉‑语言投影，实现高效且结构化的视觉表征；② 通过仅 5M 的人工精细合成科学样本加上 RL‑GSPO 训练，即可在 50%+ 科学基准上突破；③ 提供统一、透明且可复现的三阶段训练流程，兼顾科学推理与通用视觉能力。

**🔧 技术方法**

使用技术包括 RICE‑ViT 视觉 Transformer、PatchMerger 令牌压缩投影、Qwen3‑8B 语言解码器、LLaVA‑1.5/OneVision 预训练数据、监督微调 (SFT)、基于群序列策略优化的 RL（GSPO）、多层次奖励与链式推理模板。

**📊 数据集**

使用的数据集有公开多模态 LLaVA‑1.5 558k、LLaVA‑OneVision 85M、人工构造的 5M 以内科学领域样本（OCSR、化学反应、微结构）、172K RL 训练集，以及 37 组基准测试集（AI2D、MMMU、MathVision、ScienceQA、OpenRxn 等）。

**📈 对比分析**

通过 deterministic decoding 在 37 个基准上与 7‑9B 规模的 Qwen3‑VL、InternVL、LLaVA‑OV 等模型对比，Innovator‑VL‑8B‑Thinking 平均得分 61.83%，在通用视觉 74.5%、数学推理 55.4%、科学推理 50% 以上，并在推理 token 上比同类模型节省 20‑60%，准确度/ token 提升 1.4‑4.3 倍。

**⚠️ 局限性**

局限性在于仍受 8B 参数规模约束，主要聚焦图像与文本两模态，尚未扩展到视频、3D、时序等；RL 训练需要人工奖励设计与大规模算力；科学数据以合成为主，实际实验室数据覆盖不足；在更广泛的领域泛化仍有待验证。

---

## 258. StableQAT: Stable Quantization-Aware Training at Ultra-Low Bitwidths

**arXiv ID:** 2601.19320 | [PDF](https://arxiv.org/pdf/2601.19320v1)

**作者:** Tianyi Chen `[一作]` (Microsoft), Pashmina Cameron `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的量化感知训练框架——StableQAT，通过旋转衰减傅里叶逼近来稳定低位宽（2-4bit）量化训练。

**💡 创新点**

创新点在于构造了旋转衰减傅里叶替代函数（RDFS），既能近似硬量化的阶梯函数，又能保持梯度可微、波动有限，解决了STE的梯度失配和软量化的梯度爆炸问题。

**🔧 技术方法**

核心技术包括：傅里叶级数分解量化阶梯、坐标旋转、衰减幅值调节、截断到一阶多项式实现轻量级梯度替代；并提供理论证明梯度方差有界与L²逼近最优。

**📊 数据集**

在大语言模型（LLaMA‑3.2‑1B、3‑3B、3‑2‑3B）与视觉Transformer上进行实验，使用官方数据集和混合训练语料（SlimPajama+FineWeb‑Edu）。

**📈 对比分析**

与ParetoQ、DSQ、STE等基线比较，StableQAT在2-4bit设置下平均提升1–6.9%（最高+6.88%），并在多种学习率和随机种子下表现出更小的误差条带，训练收敛更平稳，计算开销与STE相当，远低于DSQ。

**⚠️ 局限性**

局限性：仅在大模型上验证，未深入探讨更高位宽或更小模型；幅值A的调节仍需经验，动态调节策略未实现；对特殊硬件的兼容性未系统评估。

---

## 259. Modeling Sampling Workflows for Code Repositories

**arXiv ID:** 2601.19316 | [PDF](https://arxiv.org/pdf/2601.19316v1)

**作者:** Romain Lefeuvre `[一作]` (University of Rennes), Houari Sahraoui `[通讯]` (Université de Montréal)

**通讯引用:** 4676 | [OpenAlex ID](https://openalex.org/A5009574640)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种领域特定语言（DSL），用于显式描述软件仓库数据采样的多阶段工作流，并实现了基于该DSL的Python内部API与统计代表性推理；

**💡 创新点**

通过将采样策略抽象为可组合的采样操作符，实现了对复杂多阶段采样流程的正式建模与自动化代表性指标计算，弥补了传统采样框架难以描述的空白；

**🔧 技术方法**

使用DSL元模型与Python fluent API实现内部DSL，结合Software Heritage（SWH）Loader、统计检验（Kolmogorov‑Smirnov、Chi‑square、Cochran’s公式）及可视化工具；

**📊 数据集**

在Software Heritage图数据库的子集、MSR会议论文的CSV bibliographic数据以及SWH的仓库元数据上进行验证；

**📈 对比分析**

通过案例研究将DSL模型与文献中的采样策略进行对照，使用统计检验与样本量计算证明其可表达性与代表性；性能方面主要关注功能可行性，未进行大规模性能基准；

**⚠️ 局限性**

局限性包括：DSL主要面向代码仓库，其他 artefact 的适配性待扩展；缺乏正式的可用性评估和用户实验；统计方法仅覆盖部分采样操作符，未覆盖所有复杂组合；未评估对实验结果的实际影响。

---

## 260. A Collaborative Extended Reality Prototype for 3D Surgical Planning and Visualization

**arXiv ID:** 2601.19303 | [PDF](https://arxiv.org/pdf/2601.19303v1)

**作者:** Shi Qiu `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 51733 | [OpenAlex ID](https://openalex.org/A5032708386)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出并实现了一个协作XR原型系统，包括MR头显交互式手术规划、云端数据管理和无眼镜多屏立体可视化，支持多用户协作与临床决策。

**💡 创新点**

创新点在于将XR交互、云数据同步与多屏立体可视化三大模块紧密集成，形成无缝协同的临床工作流程，而非单一功能的解决方案。

**🔧 技术方法**

使用了MR头显（如Microsoft HoloLens）、Unity3D实现交互与渲染、ThinkPHP框架搭建云平台，以及裸眼立体显示屏和3D显示屏实现多设备同步可视化。

**📊 数据集**

使用患者的CT/MRI体素数据构建3D网格模型进行肝切除规划，并收集8名肝外科医生与10名工程研究生的使用反馈作为实验数据。

**📈 对比分析**

通过System Usability Scale（SUS）和5分Likert量表对比，XR系统相较于桌面Slicer提升了98.36% SUS评分（76.25±13.43 vs 38.44±16.90），协调立体平台相比桌面基线提升了约17.4% SUS评分（81.00±13.80 vs 63.60±20.70）。

**⚠️ 局限性**

局限在于仍需优化用户体验、扩展设备兼容性、集成AI辅助规划，并在更大样本和临床验证中评估实际临床效果。

---

## 261. PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems

**arXiv ID:** 2601.19402 | [PDF](https://arxiv.org/pdf/2601.19402v1)

**作者:** Amit Singh Bhatti `[一作]` (Quantiphi), Dagnachew Birru `[通讯]` (Quantiphi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PROTEUS路由器，能够在推理时直接接收准确率目标τ并根据该目标做出模型选择，解决传统LLM路由器无法实现实时准确率控制的问题。

**💡 创新点**

创新点在于将准确率目标作为运行时参数，通过τ‑条件神经网络结合拉格朗日双变量控制，使单一训练模型即可覆盖完整的准确率范围，并在所有目标下实现准确率floor合规。

**🔧 技术方法**

核心技术包括：DeBERTa‑v3‑small作为查询编码器；β分布输出质量偏好μ；PPO强化学习训练；拉格朗日双变量λ反馈机制；可学习的成本敏感度γ；以及基于模型性能预测的分数函数。

**📊 数据集**

实验使用RouterBench（405K查询，11个模型）和SPROUT（45K查询，14个模型）两大公共路由基准。

**📈 对比分析**

与多种静态、学习型及OmniRouter等基线比较，PROTEUS在RouterBench上达90.1%准确率、90%成本节约，SPROUT上达94.0%准确率、83%成本节约，且在所有测试点均实现100% floor合规，SLA遵从率显著高于对照组。

**⚠️ 局限性**

局限性包括：仅在已标注准确率的数据上训练，部署时需持续人工或LLM评估；对极端成本/准确率跨度仍有挑战；分布漂移时可能需重新训练；目前仅处理单步决策，未考虑多轮对话中的上下文依赖。

---

## 262. Establishing dermatopathology encyclopedia DermpathNet with Artificial Intelligence-Based Workflow

**arXiv ID:** 2601.19378 | [PDF](https://arxiv.org/pdf/2601.19378v1)

**作者:** Ziyang Xu `[一作]` (New York University), Yifan Peng `[通讯]` (Cornell University)

**通讯引用:** 10111 | [OpenAlex ID](https://openalex.org/A5085113833)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了一个大规模、同行评审的、开源皮肤病理图像数据集 DermpathNet，使用半自动化工作流从 PubMed Central 中提取并标注 7,772 张图像。

**💡 创新点**

采用深度学习与关键词双重检索相结合的混合方法，提高图像检索的精确性和召回率，并验证了该数据集对 GPT‑4v 多模态模型的局限性。

**🔧 技术方法**

使用 DenseNet‑121 基础的图像模态分类器、图像增强、关键词检索、人工标注与验证以及 GPT‑4v 评估等技术。

**📊 数据集**

利用 PubMed Central 开源子集、ImageCLEF 2016 数据集训练深度学习模型，以及人工标注的 651 张金标准图像。

**📈 对比分析**

与仅关键词检索相比，深度学习检索的 F‑score 为 0.896，高于 0.610；混合方法在常见诊断下 F‑score 0.631，在罕见诊断下 0.938，显示出更高的召回率。

**⚠️ 局限性**

GPT‑4v 在皮肤病理图像诊断任务中表现低下，揭示了多模态 LLM 在此领域的局限性，且数据集规模与疾病覆盖仍有限。

---

## 263. Cross-Examination Framework: A Task-Agnostic Diagnostic for Information Fidelity in Text-to-Text Generation

**arXiv ID:** 2601.19350 | [PDF](https://arxiv.org/pdf/2601.19350v1)

**作者:** Tathagata Raha `[一作]` (M42 Health), Praveenkumar Kanithi `[通讯]` (M42 Health)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了跨文本的无参考评估框架Cross-Examination Framework（CEF），通过生成可验证问题并交叉检查来评估生成文本的内容保真度。

**💡 创新点**

核心创新在于将源文本和生成文本分别视为知识库，生成仅可得到“YES”答案的可验证问题，实现任务无关、无参考、可解释的三维评估，并系统性地分析评审模型鲁棒性和最佳问数。

**🔧 技术方法**

使用大语言模型生成问题与答案，计算Coverage、Consistency、Conformity指标；通过ADR/ADS衡量评审模型稳定性；采用确定性解码生成固定数量问题。

**📊 数据集**

评估数据集包括多语种翻译集NTREX、英文新闻摘要集CNN/DailyMail、临床笔记集ACI-Bench，并用WMT'25和FRANK进行人工错误标注验证。

**📈 对比分析**

CEF在与BLEU/ROUGE、BERTScore、COMET等传统指标对照时，能更准确捕捉遗漏、虚假信息和逻辑矛盾；参考自由模式与有参考模式相关性高，表明其可靠性；在多语言翻译、摘要和临床生成任务上均表现出优于传统指标的错误检测能力。

**⚠️ 局限性**

主要局限包括对评审LLM的依赖导致模型偏倚；低资源语言下问题生成不稳定；多步骤管道计算成本高于传统指标，需要进一步探索小模型评审和多语言问答提升。

---

## 264. Judgelight: Trajectory-Level Post-Optimization for Multi-Agent Path Finding via Closed-Subwalk Collapsing

**arXiv ID:** 2601.19388 | [PDF](https://arxiv.org/pdf/2601.19388v1)

**作者:** Yimin Tang `[一作]` (University of Southern California), Erdem Bıyık `[通讯]` (University of Southern California)

**通讯引用:** 724 | [OpenAlex ID](https://openalex.org/A5031426401)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Judgelight 后优化框架，将学习型多智能体路径规划（MAPF）生成的轨迹中冗余或振荡动作通过闭环子行走收缩（MAPF-Collapse）去除，从而提高路径质量。

**💡 创新点**

创新点：①将轨迹级冗余消除形式化为 MAPF-Collapse 问题并证明其 NP‑hard；②设计完整的整数线性规划（ILP）模型，并引入高效预处理（ABA 过滤、单调收缩等）和依赖关系约束，使求解可行且精确；③将此后优化层与多种学习型 MAPF 求解器无缝集成，提升实际部署效能。

**🔧 技术方法**

使用的技术包括：整数线性规划（ILP）建模与 Gurobi 求解；预处理步骤（闭合子行走收缩、ABA 过滤、长度‑3 过滤）；约束构造（同一智能体互斥、跨智能体冲突排除、依赖关系约束）；实验中还利用了图形可视化与性能统计工具。

**📊 数据集**

实验数据集：POGEMA benchmark，包含 5 种地图类型（Maze、Random、Warehouse、Puzzle、Cities‑tiles），共 3,296 个测试案例，用于评估不同 MAPF 求解器及 Judgelight 的效果。

**📈 对比分析**

比较方法：将 Judgelight 与多种学习型 MAPF 求解器（SCRIMP、DCC、RAILGUN、MAMBA、Follower）以及子最优搜索式 LaCAM 进行对比；主要指标包括总路径成本（SoC）与 SoC 节省比例；实验显示 Judgelight 在 90% 以上成功案例中平均节省 20%–40% 的路径成本，且大多数情况在 1 秒内完成后优化。

**⚠️ 局限性**

局限性：①仅针对标准 MAPF（不区分等待与移动成本）；②对高密度、复杂地图时 ILP 规模仍可能过大，导致求解时间上升；③对部分未成功完成任务的学习型策略效果有限，仍需进一步提升原始策略的可靠性；④当前仅消除轨迹冗余，未同时处理不必要等待或更复杂的多目标约束。

---

## 265. GenPairX: A Hardware-Algorithm Co-Designed Accelerator for Paired-End Read Mapping

**arXiv ID:** 2601.19384 | [PDF](https://arxiv.org/pdf/2601.19384v1)

**作者:** Julien Eudine `[一作]` (Huawei Technologies Switzerland AG), Ji Zhang `[通讯]` (Huawei Technologies Switzerland AG)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一款面向Paired‑End读序列映射的硬件算法协同加速器GenPairX，能够在不牺牲准确率的前提下显著提升映射吞吐量与能效。

**💡 创新点**

创新点包括：① 基于哈希的paired‑end过滤器（Paired‑Adjacency Filtering），能够一次性筛选出满足两端距离约束的候选位置；② 轻量化对齐算法，利用Shifted Hamming Distance + 向量化XOR实现大多数配对的无DP比对；③ Near‑Memory Seed Locator (NMSL)，将索引表放置在HBM多通道内，利用全通道并行与滑动窗口调度，显著缓解内存带宽瓶颈；④ 通过与GenDP的融合实现剩余DP任务的高效落地。

**🔧 技术方法**

技术细节涵盖：哈希索引结构SeedMap、xxHash哈希、向量化XOR对齐、HBM多通道并行与FIFO滑动窗口、FPGA/ASIC加速单元、CPU+GPU主机协同、对齐后回退到GenDP的动态规划模块。

**📊 数据集**

使用的数据集：人类参考基因组GRCh38；短读（150bp PE）来自HG002（Ashkenazi Son）的100M对（1M对用于评测）；长读（PacBio HiFi）222M条平均长度9,569bp，用于验证长读适配方案。

**📈 对比分析**

评测方法：与Minimap2（CPU）、BWA‑MEM‑GPU、GenCache、GenDP等软件/硬件基线在相同参考上进行对比；对比指标包括吞吐量（Mbp/s）、能耗（Mbp/s/W）、面积效率（Mbp/s/mm²）以及准确率（SNP/INDEL F1）。实验结果表明+GenDP在短读场景下达到57,810 Mbp/s吞吐，功耗仅为GenDP的1/1.43倍，面积仅为GenDP的1/1.96倍；与软件基线相比吞吐提升至1575×，能耗提升至911×。

**⚠️ 局限性**

局限性：① 约25%读对仍需DP回退，影响最优性能；② 对极低错误率或极大seed位置的数据需要更严格的过滤阈值，平衡准确率与吞吐；③ 对长读性能下降明显，需额外DP模块；④ 设计高度依赖HBM高带宽，非HBM环境下效率会显著下降。

---

## 266. Teaching Machine Learning Fundamentals with LEGO Robotics

**arXiv ID:** 2601.19376 | [PDF](https://arxiv.org/pdf/2601.19376v1)

**作者:** Viacheslav Sydora `[一作]` (Max Planck Institute for Intelligent Systems), Michael Muehlebach `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 623 | [OpenAlex ID](https://openalex.org/A5049845074)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套基于网页的“Machine Learning with Bricks”平台，并配套开展了为期两天、面向12‑17岁学生的无编程机器人工作坊；

**💡 创新点**

将交互式可视化、LEGO机器人与机器学习算法（KNN、线性回归、Q‑learning）结合，采用构造主义和协作学习的无代码教学模式，并开放源代码，突破了传统面向成人或单算法的教育平台。

**🔧 技术方法**

Web前端与后端技术、Bluetooth与LEGO SPIKE Prime通信、Python实现的KNN/线性回归/ Q‑learning算法、数据可视化库以及YouTube视频教程等。

**📊 数据集**

主要使用学生在实验中自行采集的数据——水果颜色与长度、弹射器的马达速度与落点距离、爬行机的状态‑动作奖励等；未使用公开大规模数据集。

**📈 对比分析**

通过对14名学生的问卷前后对比（Likert量表、Wilcoxon符号秩检验），评估自我认知提升。结果显示在机器学习范式、KNN、线性回归、强化学习及探索‑利用平衡等方面均显著提升，且学生对平台易用性与可视化效果评价高。

**⚠️ 局限性**

局限性包括样本量小且单一地区、受自选报名影响可能偏向技术兴趣者、评价主要基于自评而非客观测验、仅测评即时效果缺乏长期跟踪、问卷缺失值导致统计功效下降。

---

## 267. Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection

**arXiv ID:** 2601.19375 | [PDF](https://arxiv.org/pdf/2601.19375v1)

**作者:** Quy-Anh Dang `[一作]` (VNU University of Science), Chris Ngo `[通讯]` (Knovel Engineering Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的内部激活进行推理时的可控修改，提出了一种名为Selective Steering的激活调节方法。

**💡 创新点**

创新点在于：①提出了保持激活范数不变的旋转公式；②根据不同层的对比类均值投影符号决定是否进行调节，实现了仅在特征显著层做操控的层级选择。

**🔧 技术方法**

采用差分均值提取特征方向，构造二维旋转平面，使用正交投影与旋转矩阵实现norm-preserving rotation，并结合层级筛选逻辑。

**📊 数据集**

使用的训练与评估数据集包括AdvBench（对抗性提示）、Alpaca（中性提示）、tinyBenchmarks（tinyAI2_arc、tinyGSM8K、tinyMMLU、tinyTruthfulQA、tinyWinogrande）以及对齐与攻击性评测工具（HarmBench、PolyGuard、LLM Judge）。

**📈 对比分析**

与Activation Addition、Directional Ablation、Standard Angular Steering和Adaptive Angular Steering等基线相比，Selective Steering在8个模型上实现了5.5倍以上的攻击成功率提升、无困惑度阈值违规、≈100%通用能力保留，表现最优。

**⚠️ 局限性**

局限性包括：特征方向仅通过差分均值获取，可能不够最优；二维平面构造使用启发式第一主成分，缺乏理论最优保证；需要模型内部访问，难以在API-only部署中使用。

---

## 268. Binary Token-Level Classification with DeBERTa for All-Type MWE Identification: A Lightweight Approach with Linguistic Enhancement

**arXiv ID:** 2601.19360 | [PDF](https://arxiv.org/pdf/2601.19360v1)

**作者:** Diego Rossini `[一作]` (Università della Svizzera Italiana), Lonneke van der Plas `[通讯]` (Università della Svizzera Italiana)

**通讯引用:** 2688 | [OpenAlex ID](https://openalex.org/A5022407073)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将多词表达检测改为每个词进行三种二分类（START/END/INSIDE），结合 DeBERTa‑v3‑large、NP 片段与依赖路径特征以及数据增强，得到轻量级且效果更优的模型。

**💡 创新点**

创新点在于（1）把跨度预测转化为高效的 token‑level 3‑label 方案；（2）系统集成句法知识（NP 块、依赖距离）以提升对离散与名词型 MWE 的识别；（3）通过适当的数据增强在小数据集上显著提升性能。

**🔧 技术方法**

使用技术包括 DeBERTa‑v3‑large Transformer、三分类输出层、NP 片段嵌入、依赖距离嵌入、过采样与词义替换两种数据增强方式，以及阈值调优与候选重构算法。

**📊 数据集**

实验数据集为 CoAM（1,301 文，867 MWE）与 STREUSLE（1,530 训、217 验、209 测），两者均经统一映射为 START/END/INSIDE 标注。

**📈 对比分析**

与现有大模型 Qwen‑72B 及基线 BERT‑span 方案比较，CoAM 上达到 69.8% F1（比 Qwen‑72B 提高 12 点，参数量仅为其 1/165），STREUSLE 上达到 78.9% F1，显示该方法在不同规模数据集上均能保持领先。

**⚠️ 局限性**

局限性包括：仅预测边界缺失类型标签；对 spaCy 句法分析的依赖导致错误传播；在小数据集上仍易过拟合；仅在英语上验证，跨语言推广需进一步研究；以及需要高端 GPU 进行训练。

---

## 269. CommSense: Facilitating Bias-Aware and Reflective Navigation of Online Comments for Rational Judgment

**arXiv ID:** 2601.19347 | [PDF](https://arxiv.org/pdf/2601.19347v1)

**作者:** Yang Ouyang `[一作]` (ShanghaiTech University), Quan Li `[通讯]` (ShanghaiTech University)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5100689370)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在线评论呈现顺序对用户判断偏差的影响，并基于研究结果设计了可实时使用的插件CommSense，帮助用户更理性、反思地处理评论信息。

**💡 创新点**

创新点在于从用户角度系统性挖掘评论呈现导致的偏差机制，提出四阶段决策路径及三大偏差问题，并据此设计出四项基于界面层面的去偏策略（预设框架、交互组织、即时反思提示、动态综合）。

**🔧 技术方法**

采用LLM（GPT‑4o）进行情感与主题提取，利用UMAP降维+KeyBERT+多语言情感分析构建语义空间；界面实现基于Vue.js前端、Flask后端的插件；交互上实现对比提醒、可拖拽的注释与合成板。

**📊 数据集**

使用从主要酒店预订平台（Booking、TripAdvisor等）抓取的574条真实酒店评论，结合研究场景（假设用户评估伊斯坦布尔酒店）。

**📈 对比分析**

与传统的“列表+笔记”基线进行对照实验（N=24），CommSense显著降低心理/物理/时间负荷（p<0.05），提升功能满意度、逻辑一致性、完整度、反思深度及证据支撑，且在评论覆盖度和情感平衡上表现更佳。

**⚠️ 局限性**

局限性包括样本规模相对较小、实验场景受限于酒店评论、插件效果可能因任务不同而异、LLM生成摘要偶有不准确或偏差、用户对自定义与适配需求仍需进一步研究。

---

## 270. When Benchmarks Leak: Inference-Time Decontamination for LLMs

**arXiv ID:** 2601.19334 | [PDF](https://arxiv.org/pdf/2601.19334v1)

**作者:** Jianzhe Chai `[一作]` (Institute of Science Tokyo), Jun Sakuma `[通讯]` (RIKEN AIP)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种推理时嵌入扰动框架 DeconIEP，用来消除大型语言模型在 benchmark 上的泄漏污染。

**💡 创新点**

创新点在于：① 在输入嵌入空间施加微小、受限的扰动来抑制记忆化的捷径路径；② 利用相对较少污染的参考模型训练扰动生成器，实现无参数改动、无评估集改动的去污染；③ 通过 KL+CE 损失使得被扰动模型的输出分布更接近参考模型。

**🔧 技术方法**

采用的技术包括：小型 Transformer 生成器、tanh 变换限制扰动大小、KL 余弦相似度作为目标、跨模型对齐训练、embedding 级别扰动、对比实验与多尺度评估。

**📊 数据集**

使用的数据集：MMLU 与 TruthfulQA 作为评估 benchmark，OpenOrca 作为干净训练数据，利用 benchmark 本身的子集模拟不同级别的泄漏。

**📈 对比分析**

与黑盒方法（TED、ITD）和白盒方法（Shortcut Neuron、Short Circuit）进行对比；结果表明 DeconIEP 在残留污染（RC）上始终处于最低或次低水平，同时在干净任务上的损失（BUD）最小，且在不同模型规模、不同泄漏强度下表现稳健。

**⚠️ 局限性**

局限性：① 需要对模型内部嵌入和梯度的白盒访问；② 依赖参考模型，若参考模型严重污染或分布差异大，效果可能下降；③ 对语义保真仅有经验验证，缺乏形式保证；④ 产生扰动需要额外推理开销，需进一步优化。

---

## 271. Polyhedral design with blended $n$-sided interpolants

**arXiv ID:** 2601.19322 | [PDF](https://arxiv.org/pdf/2601.19322v1)

**作者:** Péter Salvi `[一作]` (Budapest University of Technology and Economics), Péter Salvi `[通讯]` (Budapest University of Technology and Economics)

**通讯引用:** 273 | [OpenAlex ID](https://openalex.org/A5075745897)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于多边形局部二次插值、利用 QGB（Quadratic Generalized Bézier）面与有理贝塞尔映射的闭合网格顶点插值曲面构造方法。

**💡 创新点**

创新点在于：①使用二次广义贝塞尔面处理任意顶点度数，实现精确插值；②通过 Wachspress 坐标与有理贝塞尔曲线构造连续且光滑的参数映射；③采用平滑混合实现多面拼接，保持整体 C⁰ 连续性。

**🔧 技术方法**

主要技术包括：二次贝塞尔曲面、广义贝塞尔（QGB）面、Wachspress 权重、二维有理二次贝塞尔曲线、黄金分割搜索、顶点重心插值等。

**📊 数据集**

使用的典型数据集有：环面（torus）、Trebol 形状、正二十面体（icosahedron）以及更复杂的 Bob 模型等。

**📈 对比分析**

比较方法主要通过可视化手段（等值线、平均曲率图）来评估光滑度与连续性；未给出数值性能指标，实验表明预计算曲线交点后渲染速度可在毫秒级别，适合实时演示。

**⚠️ 局限性**

局限性包括：仅适用于封闭网格；不支持边界网格与连续边界约束；缺少法向量插值，导致高曲率区域边缘略显平坦；在极稀疏或高度曲率处可能产生数值不稳定。

---

## 272. Perception-to-Pursuit: Track-Centric Temporal Reasoning for Open-World Drone Detection and Autonomous Chasing

**arXiv ID:** 2601.19318 | [PDF](https://arxiv.org/pdf/2601.19318v1)

**作者:** Venkatakrishna Reddy Oruganti `[一作]` `[通讯]` (Independent Researcher), Venkatakrishna Reddy Oruganti (Independent Researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于轨迹的时间推理框架 P2P，用于无人机检测、行为分类、意图预测和可行拦截的轨迹预测。

**💡 创新点**

创新点在于将运动模式编码为 8 维令牌并利用 12 帧因果 Transformer 进行时间推理，直接考虑拦截可行性，并提出 ISR（Intercept Success Rate）指标衡量拦截成功率。

**🔧 技术方法**

采用因果 Transformer、运动令牌表示、多任务学习（分类、行为、意图、轨迹预测）以及基于加速度和光滑度的特征。

**📊 数据集**

使用 Anti-UAV-RGBT 数据集（226 条真实无人机序列）进行训练与评估。

**📈 对比分析**

与帧基、追踪仅、线性速度等基线对比；P2P 在 ADE 上提升 77%，ISR 上提升至 0.597（相对基线提升 597 倍），同时保持 100% 的分类准确率。

**⚠️ 局限性**

局限性包括仅在单无人机单数据集上验证、固定拦截器参数、缺乏多无人机、多场景泛化和实机部署验证。

---

## 273. Constructing self-referential instances for the clique problem

**arXiv ID:** 2601.19393 | [PDF](https://arxiv.org/pdf/2601.19393v1)

**作者:** Jiaqi Li `[一作]` (Northeast Normal University), Minghao Yin `[通讯]` (Northeast Normal University)

**通讯引用:** 13237 | [OpenAlex ID](https://openalex.org/A5023555343)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在 Erdős–Rényi 随机图模型中证明团问题存在相变现象，并构造自指实例族来揭示该问题的本质算法难度。

**💡 创新点**

创新点在于：在相变点构造了顶点数、边数和度序列相同但含 k‑团与不含 k‑团的两类图，并证明它们可通过保持度数的对称变换互相转换，从而解释该临界区域为何需要穷举搜索。

**🔧 技术方法**

主要技术包括组合数学与随机图理论分析、精确定位相变点，以及对称变换构造自指实例。

**📊 数据集**

使用的数据集为 Erdős–Rényi 随机图的理论生成结果，未使用实际实验数据集。

**📈 对比分析**

由于研究为理论证明，未给出实验性能对比；通过理论推导表明在临界区域算法必须遍历几乎全部解空间，性能急剧下降。

**⚠️ 局限性**

局限性：仅在理论层面讨论，缺乏实验验证；分析聚焦于相变点附近，未涉及更一般图结构或针对该难点的实用算法改进。

---

## 274. The Psychological Science of Artificial Intelligence: A Rapidly Emerging Field of Psychology

**arXiv ID:** 2601.19338 | [PDF](https://arxiv.org/pdf/2601.19338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 275. CHEHAB RL: Learning to Optimize Fully Homomorphic Encryption Computations

**arXiv ID:** 2601.19367 | [PDF](https://arxiv.org/pdf/2601.19367v1)

**作者:** Bilel Sefsaf `[一作]` (New York University Abu Dhabi), Riyadh Baghdadi `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5044704994)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用强化学习自动对全同态加密（FHE）程序进行向量化与优化，减少指令延迟与噪声增长。

**💡 创新点**

首次将RL框架应用于FHE代码优化，提出层次化动作空间、可解析的奖励函数以及LLM生成的训练数据。

**🔧 技术方法**

采用PPO强化学习、Transformer编码器、层次化策略网络、以及基于成本模型的奖励机制。

**📊 数据集**

使用由大型语言模型（Gemini 2.5 Flash）生成的15,855条FHE表达式作为训练集。

**📈 对比分析**

与主流FHE编译器Coyote对比，CHEHAB RL生成的代码执行速度提升约×几（几乎几百倍），噪声消耗降低，编译时间也更快。

**⚠️ 局限性**

在小规模程序上编译时间略高，RL策略偶尔产生过度旋转或子优化不当，且对极大规模程序的可扩展性仍需进一步验证。

---

## 276. AI-driven Intrusion Detection for UAV in Smart Urban Ecosystems: A Comprehensive Survey

**arXiv ID:** 2601.19345 | [PDF](https://arxiv.org/pdf/2601.19345v1)

**作者:** Abdullah Khanfor `[一作]` (Najran University), Hakim Ghazzai `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 3290 | [OpenAlex ID](https://openalex.org/A5021394822)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对基于人工智能的无人机入侵检测系统进行了系统综述，涵盖了网络通信层的网络攻击与无人机自身造成的物理入侵两大类威胁。

**💡 创新点**

创新点在于提出了统一的跨模态IDS框架、融合网络、惯导、视觉等多源信息，并给出了未来十个研究方向，弥补了以往仅关注单一攻击或单一模态的不足。

**🔧 技术方法**

主要技术包括监督学习（决策树、随机森林、XGBoost、CNN、YOLOv4/8）、无监督学习（One‑Class SVM、自动编码器、聚类）以及强化学习（DQN、DRL‑BWO、AE‑RL）等多种AI方法。

**📊 数据集**

所用数据集涵盖网络流量类（CIC‑IDS2018、NSL‑KDD、UNSW‑NB15）、视觉检测类（Anti‑UAV、MAV‑VID、Drone vs Bird）以及多模态类（Multimodal Drone、Drone Monitoring），体现了对网络与物理入侵的双重评测。

**📈 对比分析**

与传统基于规则或单一异常检测方法对比，实验结果显示在GPS欺骗、Jamming、Hijacking等网络攻击以及无人机检测/跟踪任务中，AI方法的准确率/召回率/ F1 分别提升 10‑30%，且检测时间保持在毫秒到秒级，满足实时响应需求。

**⚠️ 局限性**

主要局限在于缺乏大规模真实无人机通信与物理轨迹数据导致模型泛化受限、能耗与延迟评估不足，以及对抗鲁棒性研究不足，亟需统一基准与多模态协同验证。

---

## 277. Modeling Behavioral Signals in Job Scams: A Human-Centered Security Study

**arXiv ID:** 2601.19342 | [PDF](https://arxiv.org/pdf/2601.19342v1)

**作者:** Goni Anagha `[一作]` (International Institute of Information Technology Hyderabad), Sandeep Kumar Shukla `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过问卷调查和情感NLP分析，研究了工作诈骗中急迫性、沉没成本和社交证明等行为信号与付款决策的关系。

**💡 创新点**

创新点在于将行为经济学概念转化为可测量的行为信号，并将缺失率视为潜在行为指标，以人本安全视角探讨早期介入的可能性。

**🔧 技术方法**

主要技术包括精确推断（Fisher检验）、贝叶斯估计、倾向匹配、特征重要性评估以及基于预训练模型的情感分析。

**📊 数据集**

使用的数据集来自印度IIIT Hyderabad大学及其网络的匿名问卷，共91名受访者，其中37名报告过诈骗经历，16人曾付款。

**📈 对比分析**

采用精确推断和贝叶斯后验概率进行比较，结果显示紧迫性信号显著关联付款（p=0.005），沉没成本呈倾向性正相关，社交证明无显著影响；但样本量小导致置信区间宽、效果估计不稳定。

**⚠️ 局限性**

主要局限包括样本量小、单一机构与人群（大学生）偏倚、回顾性自报导致记忆偏差、缺失偏倚、单项测量可靠性低、跨文化及平台普适性不足。

---

## 278. SETA: Statistical Fault Attribution for Compound AI Systems

**arXiv ID:** 2601.19337 | [PDF](https://arxiv.org/pdf/2601.19337v1)

**作者:** Sayak Chowdhury `[一作]` (International Institute of Information Technology Bangalore), Meenakshi D'Souza `[通讯]` (International Institute of Information Technology Bangalore)

**通讯引用:** 176 | [OpenAlex ID](https://openalex.org/A5028932016)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SET A框架，对多模块AI系统进行无oracle的鲁棒性测试与错误归因。

**💡 创新点**

创新点在于将元测试与执行追踪相结合，利用统计贡献度实现模块级错误定位。

**🔧 技术方法**

使用元测试、执行追踪（执行树）、统计错误归因（Failure Contribution Score）、多任务MR组合等技术。

**📊 数据集**

采用RailSem19、CIFAR‑10+图像失真、OCR自然场景数据及相关失真库进行实验。

**📈 对比分析**

通过与传统单模型准确率对比，SET A预测准确率误差仅2‑3%，并能精准定位最脆弱子模块，验证框架在Vision与OCR系统中的有效性。

**⚠️ 局限性**

局限性包括：归因仅为相关性非因果；MR设计手工且可能不完整；对强耦合、复杂交互系统的准确性仍受限。

---

## 279. From Observations to Events: Event-Aware World Model for Reinforcement Learning

**arXiv ID:** 2601.19336 | [PDF](https://arxiv.org/pdf/2601.19336v1)

**作者:** Zhao-Han Peng `[一作]` (Shenzhen International Graduate School Tsinghua University), You He `[通讯]` (Tsinghua University)

**通讯引用:** 3879 | [OpenAlex ID](https://openalex.org/A5016380250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Event-Aware World Model (EAWM)，通过无标签事件生成与通用事件分割器提升世界模型的表示学习与策略学习的样本效率。

**💡 创新点**

创新点在于将事件预测作为自监督任务引入多种世界模型，并设计通用事件分割器（GES）与自适应高斯混合模型事件生成，形成无监督、跨模态的事件驱动表示学习框架。

**🔧 技术方法**

使用了自监督事件预测、通用事件分割器、Gaussian Mixture 事件生成、Transformer/RSSM 等世界模型架构，以及信息瓶颈约束等技术。

**📊 数据集**

实验使用了 Atari 100K、Craftax 1M、DeepMind Control Suite 500K 与 DMC-GB2 500K 四大基准数据集。

**📈 对比分析**

与 DreamerV3、Simulus、REM、DIAMOND、HarmonyDream、TWM、TD-MPC2、CURL、DrQ-v2、SVEA、SADA 等基线对比，EAWM 在所有基准上提升 10%–45%，刷新 Atari 100K、Craftax、DM Control 与 DMC‑GB2 的 SOTA，部分任务实现超人类 IQM 分数。

**⚠️ 局限性**

局限性包括：GES 实现简化，缺少更精细的边界检测；模型跨任务共享知识困难；未结合大规模预训练视觉‑语言模型；对复杂视觉噪声或非结构化事件的鲁棒性待进一步验证。

---

## 280. ClipGS-VR: Immersive and Interactive Cinematic Visualization of Volumetric Medical Data in Mobile Virtual Reality

**arXiv ID:** 2601.19310 | [PDF](https://arxiv.org/pdf/2601.19310v1)

**作者:** Yuqi Tong `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 51733 | [OpenAlex ID](https://openalex.org/A5032708386)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在移动 VR 环境下，将 ClipGS 适配为可实时渲染并支持任意角度切片的医学可视化框架；

**💡 创新点**

通过离线预计算 200 个切片状态、压缩高阶球谐系数并仅保留切片平面差异点云，构建轻量化结构，并采用梯度基透明度调制实现任意角度切片；

**🔧 技术方法**

Unity Gaussian Splatting、Mixed Reality Toolkit 3 (MRTK3)、Meta Quest 3 设备、离线预计算、梯度透明度调制、球谐压缩；

**📊 数据集**

ClipGS 数据集（多层 Gaussian 训练的医学体数据）；

**📈 对比分析**

与原 ClipGS 的硬截断 baseline 对比，PSNR 33.40 dB / SSIM 0.9698（vs 30.55 dB / 0.9542），10 人 SUS 评分提升至 88.20±9.35、交互效率 4.70±0.48；

**⚠️ 局限性**

仍需离线预计算，无法实时支持动态形变；切片状态离散化为 200 步，精度受限于离散化层数。

---

## 281. Curiosity Driven Knowledge Retrieval for Mobile Agents

**arXiv ID:** 2601.19306 | [PDF](https://arxiv.org/pdf/2601.19306v1)

**作者:** Sijia Li `[一作]` (Shanghai University of Engineering Science), Xihe Qiu `[通讯]` (Shanghai University of Engineering Science)

**通讯引用:** 438 | [OpenAlex ID](https://openalex.org/A5007950680)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于好奇心的知识检索框架，利用AppCards结构化外部知识提升移动智能体在复杂应用中的执行性能。

**💡 创新点**

创新点在于将好奇心作为内在探知缺失功能知识的触发信号，并将检索到的文档、源码、历史轨迹压缩成模块化的AppCards，显著提升规划可靠性。

**🔧 技术方法**

采用基于JS-Divergence的无模型不确定性估计、结构化知识表示（AppCards）、外部检索与知识注入机制，并在DroidRun智能体中集成。

**📊 数据集**

使用AndroidWorld基准数据集（116个交互任务，涵盖20+安卓应用）进行评估。

**📈 对比分析**

与多种基线模型（GPT‑4o、Gemini‑2.5‑Pro、GPT‑5等）在相同交互预算下对比，最优配置在GPT‑5+AppCards下达到88.8%的成功率，较之前最高84.5%提升约4.3个百分点，且在困难任务上提升显著。

**⚠️ 局限性**

局限在于对模型能力高度依赖，阈值设定需手动调优，且对弱模型可能产生负面影响；AppCards的构建与版本管理仍需手工或半自动化工具。

---

## 282. DSTCS: Dual-Student Teacher Framework with Segment Anything Model for Semi-Supervised Pubic Symphysis Fetal Head Segmentation

**arXiv ID:** 2601.19446 | [PDF](https://arxiv.org/pdf/2601.19446v1)

**作者:** Yalin Luo `[一作]` (Jinan University), Jieyun Bai `[通讯]` (Jinan University)

**通讯引用:** 1108 | [OpenAlex ID](https://openalex.org/A5009284742)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于双学生-教师架构的半监督 PSFH（耻骨联合与胎儿头）超声图像分割框架 DSTCS，结合 CNN 与 SAM 模型并引入边缘补丁叠加与邻域加权 Dice 损失；

**💡 创新点**

核心创新在于：① 将 Segment Anything Model (SAM) 作为学生分支，与传统 CNN 学生协同学习，提升全局语义与边界细节；② 边缘补丁叠加 (EPIS) 保留关键解剖边缘信息的增强策略；③ 邻域加权 Dice 损失 (NW‑Dice) 通过像素局部权重动态强化边界感知；

**🔧 技术方法**

技术手段包括：双学生-教师自监督学习、跨模型硬/软伪标签互补、一致性学习、分类器不确定度对齐、教师模型一致性正则、SAM 适配器（LoRA‑style）与 ViT 编码器、EPIS 数据增强、NW‑Dice 损失；

**📊 数据集**

在 MICCAI 2023（5101 张标注图像）和 MICCAI 2024（300 张标注图像）PSFH 评测数据集上进行实验；

**📈 对比分析**

与多种现有半监督方法（Mean Teacher、ICT、UAMT、DAN、DCT、CLB、CPS、CTCT、FSRENet、S4CVnet）及基准指标（DSC、HD95、ASD）比较，DSTCS 在 20% 标注率下 PSFH‑DSC 0.911、HD95 2.07mm、ASD 0.336mm，显著优于竞争方法；在 2024 数据集的泛化实验中同样取得最高 DSC、最小 ASD/HD95；

**⚠️ 局限性**

局限性包括：① 对 SAM 的训练与推理仍需较高计算资源；② 边缘补丁叠加依赖准确的标注边缘，可能对低质量标签敏感；③ 在极端噪声或低对比度情形下仍可能出现误分；④ 目前仅验证于两年 MICCAI 公开数据，缺乏跨中心、多模态验证。

---

## 283. eHMI for All -- Investigating the Effect of External Communication of Automated Vehicles on Pedestrians, Manual Drivers, and Cyclists in Virtual Reality

**arXiv ID:** 2601.19440 | [PDF](https://arxiv.org/pdf/2601.19440v1)

**作者:** Mark Colley `[一作]` (Ulm University), Enrico Rukzio `[通讯]` (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在不同道路使用者（行人、自行车骑行者和手动车驾驶者）以及不同干扰条件下，意图类外部人机界面（eHMI）对安全感、信任度、可用性和心理负荷等主观指标的影响，采用虚拟现实实验对40名参与者进行三种角色和三种干扰水平的对照。

**💡 创新点**

提出并验证了统一的意图式慢脉冲光带（SPLB）eHMI能够在三种不同道路使用者中发挥作用，证明不需要为不同角色设计不同的eHMI，具有统一标准化的可能性。

**🔧 技术方法**

使用Unity 3D构建VR仿真环境，配合HTC Vive Pro Eye眼动追踪、游戏方向盘与踏板、自行车模拟器以及eHMI灯带的硬件实现。

**📊 数据集**

实验数据来自40名大学生与在职人员在VR中完成的18种情境的行为轨迹、眼动记录及问卷评分。

**📈 对比分析**

通过Within-subjects 3×3×2实验设计和非参数统计（ART、Dunn、线性混合模型）评估主观量表，结果显示eHMI显著提升安全感、信任度、可用性、效用感，且心理负荷降低，三种角色间无显著交互效应，表明统一eHMI的有效性。

**⚠️ 局限性**

局限包括样本年龄和文化单一、VR模拟环境与真实交通差异、eHMI仅测试一种意图模式、干扰条件有限、以及实验时长短导致对长期适应性和学习曲线的缺乏评估。

---

## 284. OSIRIS: Bridging Analog Circuit Design and Machine Learning with Scalable Dataset Generation

**arXiv ID:** 2601.19439 | [PDF](https://arxiv.org/pdf/2601.19439v1)

**作者:** Giuseppe Chiari `[一作]` (Politecnico di Milano), Davide Zoni `[通讯]` (Politecnico di Milano)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5016373122)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 OSIRIS，一个端到端的后端流水线，用于生成海量的 DRC‑clean、LVS‑verified 的模拟电路布局数据集，并提供基于强化学习的探索框架；同时发布了 87,100 个布局变体的数据集。

**💡 创新点**

创新点在于：①系统化地通过手指数和组件位移两维度探索布局空间，生成可训练机器学习模型的结构化数据；②将布局生成与后布局性能反馈循环集成，构建可迭代的强化学习优化流程；③公开了大规模、标注完整的后端数据集，为模拟 IC 机器学习研究填补缺口。

**🔧 技术方法**

主要技术包括：手指数枚举与布局变体生成、基于 ILP 的放置与 A* 路由、DRC/LVS/PEX/仿真链；强化学习两层循环（FinPerm Search 与 RL Exploration），采用 REINFORCE 与 cPPO；使用 Skywater 130nm PDK、Magic、Netgen、Ngspice 等开源工具；对数据集进行结构化存储并配备 Qwen3‑14B LLM 微调示例。

**📊 数据集**

使用 OSIRIS 自己生成的数据集（87,100 个布局），并以该数据集对 LLM 进行微调；在比较实验中使用了 ALIGN、MAGICAL 等公开后端布局生成器作为基准，此外还对随机探索策略与 RL 探索策略进行了对比。

**📈 对比分析**

比较方法：对四个典型放大器/低通滤波器分别测量 pscore（后布局与预布局误差）、面积以及生成时间；结果显示，RL 探索在 pscore 上比 ALIGN/MAGICAL 和随机策略至少低 10~100 倍，在面积上大多相同或更小，且总生成时间显著缩短；随机策略在面积和 pscore 上最差；RL 方法在某些电路（如 5‑OTA）面积略大但电性能更好。

**⚠️ 局限性**

局限性：①当前仅支持单一技术节点（Skywater 130nm）且仅涵盖 5 个电路模板；②探索空间受限于手指数和简单位移，未覆盖更复杂的布局变换；③RL 训练成本高、收敛不稳定，缺乏跨电路迁移学习；④数据集规模虽大，但对更复杂多层结构的代表性不足。

---

## 285. ProVoice: Designing Proactive Functionality for In-Vehicle Conversational Assistants using Multi-Objective Bayesian Optimization to Enhance Driver Experience

**arXiv ID:** 2601.19421 | [PDF](https://arxiv.org/pdf/2601.19421v1)

**作者:** Josh Susak `[一作]` (University College London), Mark Colley `[通讯]` (University College London)

**通讯引用:** 1457 | [OpenAlex ID](https://openalex.org/A5026729075)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在虚拟现实驾驶模拟器中，使用人机协同多目标贝叶斯优化，设计并评估了四个主动交互参数（车内灯光、语音提示音量、符号透明度、主动程度），以降低司机的心理负荷、提升可预测性和可用性。

**💡 创新点**

创新点在于首次将多目标贝叶斯优化与主动驾驶助手结合，实时根据驾驶者反馈在持续迭代中寻找 Pareto 最优的主动交互设置，并对比训练主动程度与固定主动程度两种策略。

**🔧 技术方法**

使用了人机协同多目标贝叶斯优化（HITL MOBO）与 Unity VR 驾驶模拟器，采集驾驶者在各迭代中的主观评价并作为优化反馈。

**📊 数据集**

数据集来自 19 名持有驾照的参与者在 ProVoice VR 模拟器中的 15 次迭代主观问卷回答和对应的设计参数取值。

**📈 对比分析**

通过与固定主动程度条件对比，实验显示 MOBO 迭代显著降低心理负荷，提升可预测性和可用性；Pareto 前沿展示了不同参数组合的最佳折衷点。

**⚠️ 局限性**

局限包括样本量小、只在单一虚拟环境与场景中测试、使用冷启动 MOBO 可能探索不足、未控制学习与疲劳效应，以及隐私与安全问题未充分解决。

---

## 286. RPO:Reinforcement Fine-Tuning with Partial Reasoning Optimization

**arXiv ID:** 2601.19404 | [PDF](https://arxiv.org/pdf/2601.19404v1)

**作者:** Hongzhu Yi `[一作]`, Jungang Xu `[通讯]` (Lenovo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的强化细调中，提出了一种基于经验重放的 RPO 算法，通过使用部分推理路径前缀来显著减少采样生成量。

**💡 创新点**

创新点在于将经验缓存与长度感知奖励相结合，形成插件式的加速框架，使得模型在保持甚至提升性能的同时，训练时间缩短至原来的一小部分。

**🔧 技术方法**

主要技术包括经验重放缓存、ε‑greedy 更新策略、长度感知奖励塑形、与现有 GRPO/DAPO 等 PPO 系统的无缝集成以及对奖励与梯度的精细化调整。

**📊 数据集**

使用了六个公开推理评测数据集（AIME、MATH、AMC、Minerva、OlyB 等）以及 7k 条训练样本进行实验。

**📈 对比分析**

与 GRPO、DAPO 等基线在 1.5B/7B 模型上进行对比，RPO 在准确率上提升约 2%，训练时间缩短 90% 以上，且加速比可达 92.6%。

**⚠️ 局限性**

主要局限是牺牲了响应多样性，导致奖励方差下降，需要依赖长度奖励来维持梯度有效性，且早期探索能力相对较弱。

---

## 287. Enhancing Academic Paper Recommendations Using Fine-Grained Knowledge Entities and Multifaceted Document Embeddings

**arXiv ID:** 2601.19513 | [PDF](https://arxiv.org/pdf/2601.19513v1)

**作者:** Haixu Xi `[一作]` (Jiangsu University of Technology), Chengzhi Zhang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 2232 | [OpenAlex ID](https://openalex.org/A5056318061)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于细粒度知识实体（任务、方法、材料/数据、指标）和多维文档嵌入的学术论文推荐框架。通过构建FG‑SKG、学习论文与实体的低维向量，并动态组合多维向量（凸组合、学习权重）来实现任务级、方法级、材料/指标级的细粒度匹配与多样性提升。

**💡 创新点**

创新点主要包括：
1) 引入四类细粒度知识实体并构建FG‑SKG，解决传统推荐中过于粗粒度的问题；
2) 采用基于依存路径模板的高精度关系抽取，避免大规模标注；
3) 在文档嵌入层面同时使用SPECTER（论文内容+引用）和GPT‑3.5（实体语义），形成多维嵌入；
4) 通过学习凸组合权重实现对不同语义维度的动态加权，并在任务相似子集内进一步重排，兼顾准确性与多样性；
5) 在STM‑KG上实现Top‑50 Precision 27.3%，比现有方法提升6.7个百分点。

**🔧 技术方法**

技术细节：
- 实体识别：SciBERT + BiLSTM + Cascade + CRF；
- 关系抽取：基于依存句法模板的规则式抽取（achievedBy、usedBy、evaluatedBy、related）；
- 文档嵌入：SPECTER（论文标题+摘要+引用）→ 768 维；GPT‑3.5（实体描述）→ 1536 维；
- 多维向量组合：p_g=[c_t, c_m, c_d, s_p]、p_t、p_m、p_d；
- 相似度计算：余弦相似度；
- 权重学习：两阶段网格搜索 + 坐标上升；
- 评估指标：MAP@K、nDCG@K、ILD、Coverage。

**📊 数据集**

数据集：
- STM‑KG：55,485 篇论文、15,395 条引用、10 个学科，含细粒度实体与关系；
- STM‑Corpus：110 篇摘要，用于实体标注与模型训练；
- Aminer v12：用于跨域评估。

**📈 对比分析**

与基线比较：
- Baseline1（基于研究目标/方法句子相似度）、Baseline2（SPECTER+概念向量）
- SMIGNN、TIDRec（图神经网络+多图融合）
- LLM‑enhanced（GPT‑3.5 语义增强）
结果显示：
- STM‑KG In‑Domain：MAP@50 27.3%（比 Baseline2 22.3% 提升 5.0pp）；nDCG@50 30.6%（比 Baseline2 28.6% 提升 2.0pp）。
- Aminer v12 In‑Domain：MAP@50 31.1%（比 Baseline2 30.6% 提升 0.5pp）。
- 在多维权重学习后，虽然准确率略有下降，但 ILD 与 Coverage 明显提升，体现了可控的多样性。

**⚠️ 局限性**

限制与不足：
- 关系抽取采用规则模板，召回率受限（约 60%），对长距离或隐式关系处理不足；
- 实体识别仍依赖人工标注，难以覆盖所有学科；
- 仅考虑四类实体，忽略其他可能的细粒度信息（如实验设计、数据来源细节）；
- 缺乏在线动态用户意图匹配的机制，推荐效果在高度交叉学科时仍受限；
- 计算成本主要集中在离线嵌入与向量检索，对极大规模实时场景仍需优化。

---

## 288. A DVL Aided Loosely Coupled Inertial Navigation Strategy for AUVs with Attitude Error Modeling and Variance Propagation

**arXiv ID:** 2601.19509 | [PDF](https://arxiv.org/pdf/2601.19509v1)

**作者:** Jin Huang `[一作]` (Zhejiang University), Ying Chen `[通讯]` (Zhejiang University)

**通讯引用:** 49331 | [OpenAlex ID](https://openalex.org/A5100383082)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了在SINS/DVL松耦合导航中考虑车辆姿态误差的DVL速度投影模型和协方差传播方法，并通过仿真和湖泊实测验证其性能。

**💡 创新点**

①在速度观测方程中显式加入姿态误差项，消除累积姿态误差导致的速度投影偏差；②使用协方差矩阵进行DVL测量不确定性在坐标系之间的统计一致传播。

**🔧 技术方法**

基于姿态误差展开的小角度线性化、协方差矩阵的期望传播、EKF松耦合融合、仿真和实测对比等技术。

**📊 数据集**

仿真中生成的4000 s轨迹与真实湖泊运动轨迹（约1685 s），以及IMU、DVL、DGNSS的测量数据。

**📈 对比分析**

与传统IMU+DVL、仅姿态误差补偿（AE）、仅协方差传播（CP）单独以及两者联合（AE+CP）进行对比，联合方法RMSE下降≈78%，最大误差下降≈72%，在仿真和实验中均显著优于基线。

**⚠️ 局限性**

缺乏海底高精度绝对位姿参考（如LBL、PHINS），实验平台与环境受限，未来需在更复杂环境下进一步验证。

---

## 289. Self-Reconfiguration Planning for Deformable Quadrilateral Modular Robots

**arXiv ID:** 2601.19496 | [PDF](https://arxiv.org/pdf/2601.19496v1)

**作者:** Jie Gu `[一作]` (Fudan University), Dan Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 19640 | [OpenAlex ID](https://openalex.org/A5100456041)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种针对可变形四边形模块化自重构机器人的自重构规划算法。

**💡 创新点**

该算法通过构造虚拟图生成可行的连接与断开动作，并利用依赖反向树（DRTree）对动作进行序列化，从而保证几何可行性与连接稳定性。

**🔧 技术方法**

技术上使用了虚拟图生成（VGG）、VF2图同构匹配、DRTree动作排序，并以改进的BiRRT作为基准进行对比。

**📊 数据集**

实验使用7个模块的可变形四边形机器人，并通过多面体生成器产生随机多格子（polyomino）配置作为数据集。

**📈 对比分析**

与改进的BiRRT相比，本文方法在100对随机配置中实现了100%成功率、平均步骤26.41、平均时间1.75秒，显著优于BiRRT的46%成功率与8.54秒。

**⚠️ 局限性**

局限性在于无法保证全局最优路径，某些简单场景可能产生较长序列，并且目前仅验证在四边形机器人上，尚未推广到其他多边形系统。

---

## 290. ClaimPT: A Portuguese Dataset of Annotated Claims in News Articles

**arXiv ID:** 2601.19490 | [PDF](https://arxiv.org/pdf/2601.19490v1)

**作者:** Ricardo Campos `[一作]` (University of Beira Interior), Purificação Silvano `[通讯]` (INESC TEC)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5090950667)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了ClaimPT，一个面向欧洲葡萄牙语新闻文章的标注数据集，标注了声明、非声明以及声明的细粒度属性；

**💡 创新点**

提出了专门针对新闻文本的多层标注方案，兼顾元信息、实体属性和时间维度，并提供高质量的双重标注与专家校验；

**🔧 技术方法**

使用BERTimbau等编码模型进行序列标注，以及Gemini 2.5系列大型语言模型进行少样本生成式标注，比较不同切分策略（句子级与块级）；

**📊 数据集**

使用来自Lusa新闻机构的1,308篇欧洲葡萄牙语新闻，包含1,308篇文章、463条声明和4,393条非声明；

**📈 对比分析**

通过跨度级精确匹配评估，BERT句子级模型获得最佳F1≈30.6%（声明），BERT块级模型≈28.97%，Gemini生成式模型仅达22–36%；

**⚠️ 局限性**

数据集受主题不均、声明稀缺导致类别失衡，模型在长文本跨度检测上表现不佳；标注主观性仍有一定偏差，且实验基线仅限于现有LLM与BERT，后续可扩展多语言与更强模型。

---

## 291. Entropy-Guided k-Guard Sampling for Long-Horizon Autoregressive Video Generation

**arXiv ID:** 2601.19488 | [PDF](https://arxiv.org/pdf/2601.19488v1)

**作者:** Yizhao Han `[一作]` (Nanjing University), Xiao-Xiao Long `[通讯]` (Nanjing University)

**通讯引用:** 21481 | [OpenAlex ID](https://openalex.org/A5100393815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于视频令牌预测熵自适应的k-Guard采样策略（ENkG），解决自回归视频生成中错误累积与熵坍塌问题；

**💡 创新点**

创新点在于利用令牌级熵动态调节采样候选集大小，并引入k-Guard最小探索机制，兼顾结构完整与纹理多样；

**🔧 技术方法**

核心技术包括熵计算、熵-采样阈值映射、适应性top-p核和k-Guard扩展，以及仅在推理阶段无训练成本的实现；

**📊 数据集**

使用了自收集的高质量驾驶数据集DiverseDrive（50段）和nuPlan；

**📈 对比分析**

与传统top-k/greedy/top-p做对比，实验显示在DrivingWorld、VaVIM、Cosmos三种模型上，ENkG平均降低FVD 22.8%、FID 36.5%，同时提升LPIPS、PSNR/SSIM，视觉质量和时序连贯性显著改善；

**⚠️ 局限性**

局限性包括需手工设定熵阈值与k-Guard参数，且仅在推理阶段改进，未解决模型本身的平滑性与多样性极限问题，对极长序列或不同视频域的泛化能力仍待验证。

---

## 292. Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection

**arXiv ID:** 2601.19430 | [PDF](https://arxiv.org/pdf/2601.19430v1)

**作者:** Yao Xiao `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33732 | [OpenAlex ID](https://openalex.org/A5100412937)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 X-AIGD 细粒度可解释 AI 生成图像检测基准，包含像素级多层次伪造痕迹标注，并在此基准上评估现有检测器的可解释性与性能。

**💡 创新点**

创新点在于：①构建三层级（低级失真、高级语义、认知级对比）细粒度痕迹分类体系；②将人类标注的痕迹直接用于注意力对齐约束；③系统分析痕迹对检测结果的影响并提出多任务与转移学习实验。

**🔧 技术方法**

采用 Transformer 视觉模型（Swim、DINOv2）与多任务学习、注意力对齐损失、Grad‑CAM/可视化等技术，对图像进行伪造识别与痕迹分割。

**📊 数据集**

使用 4,000 张真实图像（MSCOCO、Conceptual Captions 等）与 13 种文本生成模型产生的 52,000 张伪造图像，测试集包含 3,000 张带标注伪造图；另外在 Synthbuster、CommFor、Chameleon 等公开数据集进行跨域评估。

**📈 对比分析**

与现有单标签检测器、转移学习、联合训练以及无痕迹对齐方法进行对比；注意力对齐显著提升跨数据集的准确率与平均召回率（mAP）≈ 3–5% 以上，且在像素级痕迹检测上远优于基线。

**⚠️ 局限性**

局限性：检测器仍主要依赖不可解释特征，人工标注的高级/认知级痕迹识别效果低；多任务学习提升有限；注意力对齐需平衡无痕迹区域，过度约束会降低召回；整体仍无法完全实现完全可解释的伪造判别。

---

## 293. UniRec: Unified Multimodal Encoding for LLM-Based Recommendations

**arXiv ID:** 2601.19423 | [PDF](https://arxiv.org/pdf/2601.19423v1)

**作者:** Zijie Lei `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7779 | [OpenAlex ID](https://openalex.org/A5003491365)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的多模态编码器UniRec，能处理文本、图像、类别、数值等四种模态的推荐信号，并通过属性三元组和层次化Q-Former让LLM有效推断下一个商品。

**💡 创新点**

创新点在于对属性进行三元组表征以保持schema，构建双层Q-Former捕捉用户交互的嵌套结构，并把多模态信号映射到统一的向量空间供LLM使用。

**🔧 技术方法**

使用了Qwen3-Embedding-0.6B、CLIP ViT-L/14、基于Fourier的数值编码器，以及自研的双层Q-Former与LoRA微调的LLM。

**📊 数据集**

使用了亚马逊Beauty、Baby和Yelp三大公开数据集，涵盖文本、图片、数值、类别、时空等多种属性。

**📈 对比分析**

与传统序列推荐、专用多模态模型以及最新的LLM多模态推荐器在MRR、Hit@10、NDCG@10上对比，UniRec在所有数据集上均领先，最高提升约15%~16%。

**⚠️ 局限性**

局限在于需要干净的属性schema，主要验证在离线next-item预测上，未处理在线或连续学习场景，以及对复杂schema噪声的鲁棒性。

---

## 294. Task-Centric Policy Optimization from Misaligned Motion Priors

**arXiv ID:** 2601.19411 | [PDF](https://arxiv.org/pdf/2601.19411v1)

**作者:** Ziang Zheng `[一作]` (Tsinghua University), Shentao Qin `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Task-Centric Motion Priors (TCMP) 框架，利用对抗式模仿作为任务优先的条件正则化，实现人类示范与机器人任务的协同学习；

**💡 创新点**

创新点在于将模仿信号转化为第一阶梯度投影约束，自动适配示范偏差并消除梯度冲突，避免传统奖励线性混合导致的任务退化；

**🔧 技术方法**

技术包括PPO强化学习核心、AMP对抗式模仿奖励、任务与模仿梯度投影、可自适应调节的权重α以及相应的理论梯度冲突与静点分析；

**📊 数据集**

使用IsaacLab/TrackerLab环境下的Unitree G1 humanoid机器人，配合不同对齐程度的人类动作捕捉示范数据（对齐、偏差、误差三类）；

**📈 对比分析**

通过与PPO、AMP在任务完成度、运动平滑度与风格一致性指标进行对比，TCMP在示范误差场景下保持任务性能不降反而提升，同时保留非零风格对齐，优于AMP且无需手动调节权重；

**⚠️ 局限性**

局限性包括：需假设梯度可测且示范误差不导致不可行方向，可能在非任务相关风格抑制上过于保守，且主要在局部收敛分析，缺乏对极端任务多模态的通用性验证。

---

## 295. Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing?

**arXiv ID:** 2601.19410 | [PDF](https://arxiv.org/pdf/2601.19410v1)

**作者:** Ahrii Kim `[一作]` (Soongsil University), Seong-heum Kim `[通讯]` (Soongsil University)

**通讯引用:** 517 | [OpenAlex ID](https://openalex.org/A5039790391)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在英文-韩语翻译上比较专有LLM（GPT‑4o系列）与开源LLM（LLaMA3‑8B、Qwen2.5‑32B）在文档级自动后编辑（APE）中的表现，系统评估文档上下文是否能提升翻译质量、鲁棒性与效率。

**💡 创新点**

①首次将专有与开源LLM在同一无监督、无细调的文档级APE设置下进行对比；②揭示专有LLM在一键提示下已可实现近人类水平的后编辑，却无法有效利用完整文档上下文；③指出常用自动评估指标（TER、COMET、BLEU）无法充分捕捉文档级编辑带来的质变，强调人工评估的重要性。

**🔧 技术方法**

采用少量示例提示的“naïve”文档级提示方法，模型在同一模板下完成APE；使用自动指标（TER、COMET、BLEU、ChrF++）和三位专业译者的相对排名评估；对比无上下文与有上下文两种设置；同时测量token消耗、推理延迟与成本。

**📊 数据集**

WMT24++ 英文-韩语数据集，包含59篇文档共7,974句子；使用专业后编辑版本作为人类PE基准。

**📈 对比分析**

比较方法：对同一文档分别在无上下文与完整上下文下让模型进行后编辑；评估编辑幅度（TERΔ）、质量提升（COMET、BLEU）以及人工排名；对比四个模型的成本与延迟。结果显示：①GPT‑4o系列在单击提示下即可达到人类PE水平；②在有文档上下文时性能提升有限；③开源模型改动幅度大、错误率高，延迟与成本也更高。

**⚠️ 局限性**

限制：仅针对英-韩高资源语言对；未包含Claude、Gemini等其他专有模型；使用完整文档上下文导致模型易受“数据中毒”影响，易产生幻觉；缺乏更精细的上下文选择或检索机制，未探讨如何更高效地利用文档信息。

---

## 296. Sim-and-Human Co-training for Data-Efficient and Generalizable Robotic Manipulation

**arXiv ID:** 2601.19406 | [PDF](https://arxiv.org/pdf/2601.19406v1)

**作者:** Kaipeng Fang `[一作]` (University of Electronic Science and Technology of China), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 30228 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 SimHum 共训练框架，利用仿真数据提取机器人运动先验，利用人类演示提取视觉先验，进而在少量真实机器人数据上实现高效、可泛化的操纵策略。

**💡 创新点**

创新点在于：①发现并利用仿真与人类数据的天然互补性；②分别从两源提取运动与视觉先验，避免复杂的域对齐；③通过模块化的 diffusion 策略与 co‑training 方案实现统一可迁移的策略。

**🔧 技术方法**

核心技术包括：diffusion policy（DiT）与 encoder‑decoder transformer；模块化动作编码/解码；视觉适配器（仿真/人类）；相对动作编码；co‑training 比例调节；预训练+微调管线。

**📊 数据集**

数据集：仿真数据（每个任务 500 条轨迹，使用 RoboTwin2.0 生成），人类演示数据（每个任务 500 次演示，12 种场景），真实机器人微调数据（80 任务），四个操纵任务（Stack Bowls Two、Click Bell、Grab Roller、Put Bread Cabinet）。

**📈 对比分析**

与 Real only、SimReal、HumReal 三种基线对比，在 ID 与 OOD 评测中，SimHum 在 ID 上平均提升约 20% SR，OOB 上提升约 35% SR；在 8 小时预算下，相比 Real only 提升 45%，相当于将 160 条真实数据的效果压缩到仅 8 条；在 OOD 上比 Real only 提升 7.1×。

**⚠️ 局限性**

局限性：实验仅覆盖基础操纵任务；OOB 性能仍有提升空间；模型仅为单任务，缺乏多任务/跨任务迁移能力；未探索极长时序或更复杂的 dexterous 任务；未来需结合 VLA 模型和野外数据进一步验证。

---

## 297. VisGuardian: A Lightweight Group-based Privacy Control Technique For Front Camera Data From AR Glasses in Home Environments

**arXiv ID:** 2601.19502 | [PDF](https://arxiv.org/pdf/2601.19502v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**通讯引用:** 1373 | [OpenAlex ID](https://openalex.org/A5011803365)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 VisGuardian，一种针对 AR 眼镜前置摄像头的细粒度视觉隐私控制技术，支持用户通过点击单个检测到的对象快速统一设置同类别或同空间属性的多对象可见性。

**💡 创新点**

创新点在于引入基于“组”的隐私管理机制：用户只需一次交互，即可为所有同敏感度、同类别或同空间范围内的对象批量开启或关闭遮蔽，从而显著降低操作成本；同时将实时目标检测与遮蔽叠加实现无缝隐私保护。

**🔧 技术方法**

核心技术包括：YOLOv10n 目标检测模型、基于 Unity + MRTK 的交互界面、UWP 系统层的摄像头控制、遮蔽叠加（使用不透明遮罩）以及 GPT‑4o 作为后端 AI 处理；所有运算均在设备端完成，保持 14 ms 的延迟。

**📊 数据集**

数据集：使用 COCO 与 LVIS 进行模型微调，构建包含面部、身份证、文件、屏幕等多类别的隐私对象标签集；验证集为 COCO 与 LVIS 的测试拆分。

**📈 对比分析**

对比方法：与滑动条（Slider‑based）和逐对象选择（Object‑based）两种基线进行 24 名参与者的实验；指标包括权限设置时间、点击次数、主观满意度。结果显示 VisGuardian 在权限设置时间上平均缩短至 15.2 s，点击次数明显低于基线，且在隐私保护、易用性、满意度等方面均获得显著优势；技术评测显示 mAP50=0.6704、延迟 14 ms、额外电量消耗仅 1.7%。

**⚠️ 局限性**

局限性：仅针对前置摄像头，未覆盖麦克风等多模态传感器；使用固定的隐私分类表，无法动态适应用户个性化或跨文化差异；实验环境受控且样本有限，缺乏对真实家庭多样场景和不同年龄段用户的验证；对检测误报/漏报的用户体验未深入探究。

---

## 298. AACR-Bench: Evaluating Automatic Code Review with Holistic Repository-Level Context

**arXiv ID:** 2601.19494 | [PDF](https://arxiv.org/pdf/2601.19494v1)

**作者:** Lei Zhang `[一作]` (Nanjing University), Xiaobing Xu `[通讯]` (Alibaba Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多语言仓库级上下文感知的自动代码审查基准 AACR-Bench，并通过人工专家验证的高质量标注提升缺陷覆盖率。

**💡 创新点**

创新点在于：①提供跨 10 种主流语言的仓库级上下文；②使用 AI 辅助 + 专家验证的双重标注，缺陷覆盖率提升 285%；③系统性研究上下文粒度、检索方式与 Agent 架构对 LLM 代码审查性能的影响。

**🔧 技术方法**

技术上采用大型语言模型（Qwen3、DeepSeek、GLM、GPT‑5.2、Claude‑4.5）与检索方法（BM25、Embedding、Agent），并构建 Diff/ File/ Repo 三层上下文级别评估框架。

**📊 数据集**

数据集为 AACR-Bench，包含 200 个 PR、1,505 条精细标注的审查评论，覆盖 10 种语言；其中 391 条来自原始 PR，1,114 条由 LLM 生成并经专家审核。

**📈 对比分析**

与传统检索（无上下文、BM25、Embedding）及 Agent 方法对比，发现不同模型在不同上下文级别和语言上表现差异；Agent 在 Repo 级别优于传统方法，但整体精度与召回存在权衡；检索方式对性能的影响因模型而异。

**⚠️ 局限性**

局限在于：尽管采用 LLM 生成并人工验证，仍难以完全覆盖所有缺陷；缺陷标注受主观性限制；数据规模相对有限，未来需进一步扩展。

---

## 299. Dynamic Worlds, Dynamic Humans: Generating Virtual Human-Scene Interaction Motion in Dynamic Scenes

**arXiv ID:** 2601.19484 | [PDF](https://arxiv.org/pdf/2601.19484v1)

**作者:** Yin Wang `[一作]` (Beihang University), Xiaohui Liang `[通讯]` (Beihang University)

**通讯引用:** 8989 | [OpenAlex ID](https://openalex.org/A5101655447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为Dyn-HSI的世界模型架构，用于在动态场景下生成文本驱动的人机交互运动，包含视觉感知、记忆检索和控制生成三大模块。

**💡 创新点**

创新点在于（1）引入动态场景感知导航能够实时检测环境变化并预测下一步行进点；（2）设计分层经验记忆，用噪声样本进行运动预热，提升生成多样性与泛化；（3）在扩散模型中加入条件适配器，按任务动态调节多模态条件的重要性。

**🔧 技术方法**

核心技术包括：Voxel化局部场景感知、Vision Transformer、Transformer解码器、条件自回归扩散模型、CLIP文本编码、A*路径规划、经验记忆检索以及多模态条件适配器。

**📊 数据集**

主要数据集：静态场景数据集Lingo与Trumans；自构建的动态评估基准Dyn-Scenes（Dyn-LINGO、Dyn-Trumans）；使用SMPL-X人形模型、文本描述和占据体素网格。

**📈 对比分析**

与SOTA方法（MotionDiffuse、ReMoDiffuse、Trumans、LINGO）对比，Dyn-HSI在静态与动态场景下的穿透率、平均穿透量、最大穿透量、轨迹相似度、目标误差、Fidelity（FID）和多样性指标均明显优于对手，尤其在动态场景中提升超过50% 的穿透率下降和轨迹质量提升，表现出更强的鲁棒性与生成质量。

**⚠️ 局限性**

主要局限包括：推理速度受限于扩散模型和动态导航模块，无法实时满足高速交互需求；动态场景仅人为模拟，缺乏连续性、复杂性和真实传感噪声的挑战，系统在高度动态或拥挤环境下仍可能失效。

---

## 300. Posterior Distribution-assisted Evolutionary Dynamic Optimization as an Online Calibrator for Complex Social Simulations

**arXiv ID:** 2601.19481 | [PDF](https://arxiv.org/pdf/2601.19481v1)

**作者:** Peng Yang `[一作]` (Southern University of Science and Technology), Xin Yao `[通讯]` (Lingnan University)

**通讯引用:** 66482 | [OpenAlex ID](https://openalex.org/A5100635494)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文针对复杂社会系统模拟器的在线校准问题，将其建模为动态优化问题（DOP），并提出一种基于后验分布辅助的进化动态优化框架（PosEDO）。

**💡 创新点**

创新点在于：①首次将后验分布学习引入在线校准，直接把观测数据与参数空间桥接；②利用后验分布实现鲁棒的变化检测（KL 散度）和高效的环境适应（采样重初始化）；③在进化算法中实现可持续的后验微调，提升长时序适应性。

**🔧 技术方法**

技术手段包括：1）Masked Autoregressive Flow（MAF）流模型用于学习参数‑数据的后验分布；2）KL 散度作为在线变化检测阈值；3）采样与距离筛选相结合的适应性重初始化；4）多种进化算法（AMP/PSO）与传统检测/适应策略的组合。

**📊 数据集**

实验数据集：①Brock–Hommes 宏观资产价格模拟器，9 个实例（不同变化频率），每个实例 30 次观测；②PGPS 市场微观结构模拟器，9 个实例（不同参数维度与变化频率），每个实例 18 次观测。数据由预训练样本（10 万对参数‑模拟数据）和在线收集的 finetuning 样本构成。

**📈 对比分析**

与基线方法（DBD‑Rand、FBCD‑Rand、FBCD‑Arch、FBCD‑NNIT、PosEDO‑Pre、PosEDO‑CD）进行对比。结果显示：PosEDO 在大多数实例上均显著降低平均校准误差（MCE）和收敛误差（P_CON），尤其在高变化频率环境下，后验辅助检测与适应大幅提升检测准确率与搜索效率。

**⚠️ 局限性**

局限性包括：①对极高维参数或极为噪声的观测数据时，后验分布易产生误报；②预训练与在线微调消耗额外计算资源；③模型假设观测与参数之间存在可学习的后验映射，对某些非可辨识系统可能不适用；④对实时性要求极高的在线场景，流模型的前向/反向推理仍有延迟。

---

## 301. Dual-Strategy-Enhanced ConBiMamba for Neural Speaker Diarization

**arXiv ID:** 2601.19472 | [PDF](https://arxiv.org/pdf/2601.19472v1)

**作者:** Zhen Liao `[一作]` (Huazhong University of Science and Technology), Wei Xu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 11490 | [OpenAlex ID](https://openalex.org/A5046238570)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于ConBiMamba的双策略增强神经说话人分离系统，融合了Conformer与Mamba的优势并通过辅助说话人变化检测任务提升边界检测精度。

**💡 创新点**

创新点包括：1) 将Conformer的自注意力替换为ExtBiMamba，实现对长序列的低内存高效建模；2) 采用多分支卷积核提升局部细节提取；3) 设计Mask‑based Layer‑wise Feature Aggregation来自适应聚合不同层特征；4) 引入Boundary‑Enhanced Transition Loss（基于Focal Loss的说话人变化检测辅助任务）显著降低边界误差。

**🔧 技术方法**

核心技术包括：ConBiMamba（融合Conformer卷积与ExtBiMamba）、多分支卷积、Mask‑based Layer‑wise Feature Aggregation、Boundary‑Enhanced Transition Loss、Pyannote管线、ECAPA‑TDNN嵌入、层归一化与Dropout、AdamW优化器、分段训练与微调。

**📊 数据集**

使用七个公开语音分离数据集（AISHELL‑4、MagicData‑RAMC、VoxConverse、MSDWild、AMI（channel 1）、AliMeeting）以及从LibriSpeech、MUSAN和房间冲击响应合成的四说话人模拟数据，形成综合训练集。

**📈 对比分析**

与PyannoteAI、Diarizen（冻结与更新WavLM）、Mamba‑diarization等基线在Pyannote管线下进行对比。系统在六个数据集上DER下降至4.6%–15.4%，在AISHELL‑4、RAMC、VoxConverse、MSDWild上均突破公开SOTA，边界检测误报率和漏报率均有显著下降。

**⚠️ 局限性**

局限性包括：1）对AliMeeting等多说话人/高重叠场景的表现仍不及SOTA；2）仍未显式建模说话人重叠，导致重叠段性能受限；3）模型复杂度相对较高，推理时对资源需求较大；4）依赖预训练WavLM Base+，若特征域不匹配可能影响性能。

---

## 302. ROIDS: Robust Outlier-Aware Informed Down-Sampling

**arXiv ID:** 2601.19477 | [PDF](https://arxiv.org/pdf/2601.19477v1)

**作者:** Alina Geiger `[一作]` (Johannes Gutenberg University), Franz Rothlauf `[通讯]` (Johannes Gutenberg University)

**通讯引用:** 4372 | [OpenAlex ID](https://openalex.org/A5035443886)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种稳健的下采样方法ROIDS，解决符号回归中IDS因离群点导致性能下降的问题。

**💡 创新点**

创新点在于在IDS的子集构造过程中动态剔除平均误差最高的训练点，从而形成对离群点不敏感的下采样策略。

**🔧 技术方法**

技术包括基于遗传编程的象限选择、IDS/ROIDS算法、错误向量距离矩阵、farthest first遍历以及LOF离群点检测对照。

**📊 数据集**

使用10个合成数据集（含/不含离群点）和6个常见真实回归基准数据集。

**📈 对比分析**

与RDS、IDS以及LOF+IDS对比，ROIDS在所有合成数据中均不比IDS差，且在含离群点时平均排名最优，在真实数据中超过80%的基准且平均排名为1.7。

**⚠️ 局限性**

局限在于对离群点敏感度参数γ的手动设置，且仅针对符号回归任务，尚未验证在更大规模或其他演化学习场景的通用性。

---

## 303. Rethinking Intelligence: Brain-like Neuron Network

**arXiv ID:** 2601.19508 | [PDF](https://arxiv.org/pdf/2601.19508v1)

**作者:** Weifeng Liu `[一作]` `[通讯]` (Vista Zenith), Weifeng Liu (Vista Zenith)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Brain-like Neural Network（BNN）范式，并实现了自架构演化的LuminaNet模型；

**💡 创新点**

创新点在于完全去除卷积、注意力和位置编码，采用基于神经元簇的可分裂、可增殖、可连接和可修剪四种演化操作，让网络在训练过程中自行构建深度、宽度与反馈/循环拓扑；

**🔧 技术方法**

技术核心是Neuron Cluster层的显式连接与Two-Pass Forward传递、四种演化策略（Split、Grow、Connect、Prune）以及交替优化-演化循环；

**📊 数据集**

实验使用CIFAR-10图像分类和TinyStories文本生成数据集；

**📈 对比分析**

与LeNet‑5、AlexNet、MobileViT、ResMLP、MLP‑Mixer等基线比较，LuminaNet在CIFAR‑10上实现了73‑74%的Top‑1准确率，显著超越传统CNN和MLP/ViT模型；在TinyStories上，其PPL为8.4、Top‑1≈53%，与单层GPT‑2相当，但FLOPs下降约25%、显存降低近50%；

**⚠️ 局限性**

局限性包括：在大规模Transformer任务上仍逊色，演化过程需要多轮训练且对超参数敏感，缺乏显式的序列建模机制，导致对长序列的捕捉仍受限。

---

## 304. Reuse of Public Keys Across UTXO and Account-Based Cryptocurrencies

**arXiv ID:** 2601.19500 | [PDF](https://arxiv.org/pdf/2601.19500v1)

**作者:** Rainer Stütz `[一作]` (Complexity Science Hub), Aljosha Judmayer `[通讯]` (University of Vienna)

**通讯引用:** 670 | [OpenAlex ID](https://openalex.org/A5090378788)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过提取并对比六大主流加密货币（BTC、ETH、LTC、DOGE、ZEC、TRX）的公钥，发现超过140万条公钥在不同链上被复用。

**💡 创新点**

创新点在于首次将公钥作为聚类依据，跨越UTXO与账户模型，揭示并量化不同链之间的密钥复用，突破了以往只能在兼容地址格式链上检测的限制。

**🔧 技术方法**

采用公钥提取、签名恢复、UpSet可视化及基于公钥的聚类方法，结合多输入启发式和地址映射，实现了跨链密钥关联。

**📊 数据集**

数据集为截至2025年4月1日前的六条链完整交易记录，包含约1.46亿BTC输出、公钥恢复得到的数百万公钥以及ETH/TRX的EVM地址。

**📈 对比分析**

与HSI及ETH‑TRX的基线对比，公钥法在不可转换地址间提升识别率至约90%，并在聚类中将约204万簇合并至约61万簇，显示显著的精度与规模优势。

**⚠️ 局限性**

局限在于仅提取可公开签名的公钥，未覆盖多签与隐私池（如Zcash shielded）、Taproot、MWEB、合约输入等；且对高阶桥接链未作完整分析。

---

## 305. Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction

**arXiv ID:** 2601.19489 | [PDF](https://arxiv.org/pdf/2601.19489v1)

**作者:** Ziyu Zhang `[一作]` (University of Chinese Academy of Sciences), Shuhan Shen `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 1996 | [OpenAlex ID](https://openalex.org/A5055576241)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套能够在一分钟内完成3D高斯喷射重建的快速管线，针对SIGGRAPH Asia 3DGS Fast Reconstruction Challenge的两阶段（噪声SLAM姿态与高精度COLMAP姿态）任务实现高效收敛。

**💡 创新点**

创新点包括：逆向每高斯并行优化与紧凑前向喷射、负载均衡分块、锚点驱动的Neural-Gaussian表示、SLAM姿态的全局优化模块、COLMAP阶段的姿态修正关闭、深度正则化与多视角一致性引导的高斯分裂/修剪，以及多视角分数指导的快速致密化策略。

**🔧 技术方法**

核心技术为3D Gaussian Splatting、Taming-GS、Speedy-splat、Scaffold-GS的Neural-Gaussian表示、AnySplat初始化、Metric3D-v2单目深度估计、COLMAP姿态、CUDA级别的前向/后向并行优化、负载均衡写入、深度监督与多视角一致性引导的分裂/修剪。

**📊 数据集**

使用了竞赛提供的数据集（第一轮SLAM姿态+稀疏点云，第二轮COLMAP姿态+RGB图像）以及TNT基准集用于消融实验。

**📈 对比分析**

相较于传统3DGS基线，通过引入上述技术在1分钟内实现PSNR提升至28.43（排名第一），在TNT基准上显著降低训练时间（30k迭代约176秒），并在多项消融实验中表现出优越的渲染质量与收敛速度。

**⚠️ 局限性**

局限性包括：对高性能GPU（RTX 4090）依赖较高、需要精确的深度或姿态先验，参数阈值需手动调节，可能对极大场景或动态场景的适应性有限，且在极端噪声姿态下性能下降。

---

## 306. LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment

**arXiv ID:** 2601.19487 | [PDF](https://arxiv.org/pdf/2601.19487v1)

**作者:** Haonan Zhang `[一作]` (Zhejiang University), Wenhai Wang `[通讯]` (Zhejiang University)

**通讯引用:** 7899 | [OpenAlex ID](https://openalex.org/A5062687402)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新颖的向量对齐方法，通过闭式权重更新将回答向量与善意向量对齐，解决LLM的jailbreak与过度拒绝两种安全失败模式；

**💡 创新点**

创新点在于将回答决策与安全评估的向量从几乎正交状态对齐，使模型的回答意愿因输入安全性而变化，从而在不需要微调或结构改动的前提下同时降低两种失败率；

**🔧 技术方法**

技术实现包括使用SVM训练层级的回答与善意控制向量、层选择评分机制、闭式最小范数权重更新以及迭代微调；

**📊 数据集**

实验数据集涵盖安全攻击评估的S-Eval-Attack、S-Eval-Risk、过度拒绝评估的ORFuzzSet、Natural Questions，以及六个通用NLP/推理任务（CoLA、MNLI、RTE、MRPC、SST、GSM8K）；

**📈 对比分析**

与VectorSteer、AlphaSteer、SCANS等基线比较，方法在12个LLM上平均提升11.45%的F1，并保持95.92%的原始模型效用，表现优异；

**⚠️ 局限性**

主要局限包括仅考虑二元毒性分类、对大模型的适用性尚未验证、需要高质量的毒性/善意标注数据、对链式推理模型不易应用、模型特定的迭代与层选参数、可迁移性与自适应性有限以及缺乏对不同权衡需求的可定制化控制。

---

## 307. KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking

**arXiv ID:** 2601.19447 | [PDF](https://arxiv.org/pdf/2601.19447v1)

**作者:** Vítor N. Lourenço `[一作]` (Universidade Federal Fluminense), Mohnish Dubey `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于知识图谱的对比推理方法（KG-Contrastive Reasoning），通过先构建知识图谱，生成与命题相关的对比问题，利用大型语言模型回答并生成压缩摘要，以提升命题真实性评估。

**💡 创新点**

创新点：①将知识图谱结构化信息与对比问题生成相结合，形成“结构化对比”推理；②使用多样性排序（MMR）挑选最具相关性、覆盖度的对比问题；③通过对比问题引导LLM聚焦证据并生成可解释摘要，显著提升事实核查的准确性和可解释性。

**🔧 技术方法**

技术与工具：大型语言模型（Claude 3.5/3.7 Sonnet、Llama 3.3 70B）用于知识图谱构建、对比问题生成、答案与摘要生成；知识图谱提取（实体、关系、三元组）；对比问题多样化（MMR）；评价指标包括AlignScore、RQUGE、F1；实验对照传统方法、Naïve LLM 以及专门调优的 LLM。

**📊 数据集**

数据集：LIAR-RAW（六类真实性标签）与 RAWFC（三类真实性标签），均为公开事实核查基准数据集。

**📈 对比分析**

比较方式：在两数据集上与传统基线、Naïve LLM、Specialised LLM 进行交叉评测；结果显示在 C3.7、L3.3 上实现最高 F1，提升 32–44pp；在 SLM 上，该框架将 F1 提升至 66–73%，显著缩小与大型模型的差距。

**⚠️ 局限性**

局限性：①未对中间组件（KG构建、对比问题生成）进行定性验证；②实验依赖昂贵的 LLM，成本高且可复现性受限；③仅在英语数据上验证，跨语言表现未知；④模型性能受 LLM 家族差异影响，比较时需谨慎。

---

## 308. NET4EXA: Pioneering the Future of Interconnects for Supercomputing and AI

**arXiv ID:** 2601.19413 | [PDF](https://arxiv.org/pdf/2601.19413v1)

**作者:** Michele Martinelli `[一作]` (INFN), Salvatore Pontarelli `[通讯]` (Università Sapienza di Roma)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并验证了BXIv3超算/AI互连网络，包括支持8M端点、64k节点、4–8倍吞吐、10倍消息率、2倍低延迟的Ethernet‑based NIC与交换机。

**💡 创新点**

创新点包括将InfiniBand特性迁移至Ethernet，实现原生IP/TCP支持、端到端加密、适配GPU Direct的零拷贝、液冷与未来硅光互连、可扩展的VC与自适应路由，以及与Ultra Ethernet Consortium兼容的安全与QoS机制。

**🔧 技术方法**

采用FPGA实现NIC（后续可转为ASIC）、商用ASIC交换芯片、PCIe Gen5/Gen6、CXL、RDMA、Portals 3/4 API、加密算法、硬件收集与聚合、能效监测与动态降功。

**📊 数据集**

使用的基准与应用场景包括Top500、Graph500、Brain Simulation、GROMACS、Quantum Espresso、SPECFEM3D、BERT以及分布式内存算子，覆盖科学计算、分子动力学、材料模拟、地震波传播与自然语言处理。

**📈 对比分析**

与BXIv2、InfiniBand、Slingshot等技术对比，在相同负载下实现了4–8倍吞吐、10倍消息率、约200 ns每跳延迟（低负载），并在Pilot Testbed与真实生产环境中验证TRL 8级性能，优于现有主流互连方案。

**⚠️ 局限性**

局限性包括仍以BXL专有ASIC为核心，尚未完全开放标准；对非Ethernet协议支持有限；硬件成本与制造周期相对较高；需进一步完善标准化与多厂商生态整合。

---

## 309. Cortex-Grounded Diffusion Models for Brain Image Generation

**arXiv ID:** 2601.19498 | [PDF](https://arxiv.org/pdf/2601.19498v1)

**作者:** Fabian Bongratz `[一作]` (Technical University of Munich), Christian Wachinger `[通讯]` (Technical University of Munich)

**通讯引用:** 8004 | [OpenAlex ID](https://openalex.org/A5069195910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种基于皮层表面SDF的Brownian桥扩散框架（Cor2Vox），实现了形状可控的3D脑MRI合成，并构建了大规模皮层形状统计模型。

**💡 创新点**

创新点在于将高分辨率皮层结构（SDF）直接作为扩散过程的条件，利用Brownian桥映射实现形状与图像的连续一致性，同时通过统计形状模型生成新颖、符合人群变异的皮层几何体。

**🔧 技术方法**

采用的核心技术包括Brownian桥扩散、3D残差UNet、SDF表征、皮层形状统计模型（PCA）、Vox2Cortex-Flow表面重建、SynthSeg+质量评估等。

**📊 数据集**

使用的主要数据集为33,403份英国生物银行（UK Biobank）T1扫描构建形状模型；ADNI T1扫描用于训练和评估；10份前额叶纹状体痴呆（FTD）病例用于跨数据集对齐与验证。

**📈 对比分析**

与Pix2Pix、BBDM、Med-DDPM等基线模型进行对比，采用PSNR、SSIM、ASSD和SynthSeg+质量评分等指标；结果显示Cor2Vox在形状一致性（ASSD下降≈10%）、图像质量（SSIM最高）和下游分割质量（均通过SynthSeg+阈值）方面均优于基线。

**⚠️ 局限性**

局限性在于目前仅支持T1加权MRI，无法直接处理其他序列或额外病灶/亚皮层结构，且需进一步扩展以适应多模态或更复杂的形状条件。

---

## 310. PALM: Enhanced Generalizability for Local Visuomotor Policies via Perception Alignment

**arXiv ID:** 2601.19514 | [PDF](https://arxiv.org/pdf/2601.19514v1)

**作者:** Ruiyu Wang `[一作]` (KTH Royal Institute of Technology), Florian T. Pokorny `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1567 | [OpenAlex ID](https://openalex.org/A5018027629)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 PALM 方法，通过视觉和本体对齐提升第三人称摄像头下的机器人操纵行为克隆在工作空间、视角和机器人外观变化中的泛化性能。

**💡 创新点**

创新点在于：① 将本地动作分布视为跨域不变性；② 采用 TCP（工具中心点）中心裁剪与重叠渲染实现视觉对齐；③ 在本体输入中剔除 (x,y) 坐标、使用相机坐标系下的旋转和二值夹爪状态实现本体对齐；④ 将策略分解为全局分析动作与局部可泛化策略，实现多阶段任务的模块化。

**🔧 技术方法**

技术包括：TCP 级裁剪、TCP 叠加、随机覆盖与视角变换数据增强、6D 旋转表示、本体状态重构、基于 ResNet-18 的视觉编码器、两层 MLP、相对动作模式。

**📊 数据集**

使用 RLBench 四个仿真任务（Lift Lid、Lift Spam、Insert Peg、Rearrange Veggies）和两个真实世界任务（Drawer、Stack），并在多种 OOD 设定下进行评估。

**📈 对比分析**

与 MirrorDuo、RoVi-Aug、ARRO 等单域扩展基线对比，PALM 在工作空间、视角和外观三者单一或组合扰动下将 OOD 性能下降从 71% 降至 8%（仿真）或 24%（实测），显著优于基线；在实测任务中，BC 在 OOD 下跌幅度高达 77%，PALM 仅维持 24% 的下降。

**⚠️ 局限性**

局限性包括：需要已知且相对准确的摄像头标定；裁剪尺寸为固定阈值，可能在多阶段任务或非桌面场景中不适宜；方法主要针对平面工作空间平移，无法直接处理垂直偏移或移动摄像头的情况。

---

## 311. Masked Diffusion Generative Recommendation

**arXiv ID:** 2601.19501 | [PDF](https://arxiv.org/pdf/2601.19501v1)

**作者:** Lingyu Mu `[一作]` (Alibaba International Digital Commerce Group), Jing Zhang `[通讯]` (Wuhan University)

**通讯引用:** 25973 | [OpenAlex ID](https://openalex.org/A5076247663)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MDGR，使用遮蔽扩散模型在并行代码表上生成语义 ID，实现并行解码；

**💡 创新点**

在代码表、训练与推理三方面创新：采用 OPQ 并行代码表、基于时间与样本的动态噪声调度（含难度嵌入），以及温暖两阶段并行解码策略；

**🔧 技术方法**

采用遮蔽扩散模型、向量量化+OPQ 并行代码表、双向 Transformer、难度嵌入、Beam Search 等技术；

**📊 数据集**

在 Amazon Electronics、Amazon Books 公开数据集和约 1 B 条交互的工业电商日志上进行实验；

**📈 对比分析**

与 10+ 基线（ID‑based 与 GR 模型）比较，Recall/NDCG 上提升 7.1%–10.8%，在线 AB 测试中收入提升 1.20%，GMV 提升 3.69%；

**⚠️ 局限性**

局限在于 Beam Search 组合爆炸导致解码成本仍高，对参数（如 γ、R_warm、m_par）敏感，且对多模态特征深度融合尚未深入探究。

---

## 312. Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots

**arXiv ID:** 2601.19499 | [PDF](https://arxiv.org/pdf/2601.19499v1)

**作者:** Mehdi Heydari Shahna `[一作]` (Tampere University), Jouni Mattila `[通讯]` (Tampere University)

**通讯引用:** 2759 | [OpenAlex ID](https://openalex.org/A5070792821)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个针对大吨位无轨车在不规则滑移地形上实现目标到达的强化学习控制框架，并在此基础上加入 Lyapunov‑类稳态器以提供形式化的到达保证。

**💡 创新点**

创新点在于将可训练的 RL 政策与 Lyapunov‑类约束层相结合，在不需要预先给定 Lyapunov 函数的前提下，实现了安全约束与学习的协同，同时通过加速的可执行动作避免了传统 CBF/BFL 的过度保守。

**🔧 技术方法**

使用的技术包括基于加速度动作空间的离散 RL（Q‑learning/SARSA）奖励设计，潜在函数奖励塑造，离散化的状态空间，Lyapunov‑类约束的 Critic 更新以及回退机制。

**📊 数据集**

实验数据集为真实的 6000 kg 滑行车在 25 m×25 m 软滑地形上随机抽取 2000 个目标的轨迹，结合 ORB‑SLAM3 视觉定位。

**📈 对比分析**

通过与基准 RL、纯 Pursuit 等方法比较，Lyapunov‑稳态器将成功率从 84.6 % 提升到 99.0 %，显著降低超时与越界失败，平均 episode 长度缩短约 30 %，控制能耗降低。

**⚠️ 局限性**

主要局限在于对离散化状态空间的依赖、对超参数（如 ν̅）的敏感性，以及在极端动态或极端滑移场景下的性能未作深入验证。

---

## 313. Time-to-Injury Forecasting in Elite Female Football: A DeepHit Survival Approach

**arXiv ID:** 2601.19479 | [PDF](https://arxiv.org/pdf/2601.19479v1)

**作者:** Victoria Catterall `[一作]` (Loughborough University), Stephen Lynch `[通讯]` (Loughborough University)

**通讯引用:** 3764 | [OpenAlex ID](https://openalex.org/A5000924818)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估并实现基于女子足球运动员长期监测数据的 DeepHit 生存模型，用于预测伤病时间并提供解释性输出。

**💡 创新点**

首次在女性足球队数据上应用 DeepHit 并结合 SHAP 进行个体化、时间可变的风险解释，同时采用自定义缺失值填补提升预测性能。

**🔧 技术方法**

使用 DeepHit 神经网络（MLP 结构）、随机森林、XGBoost、逻辑回归、SHAP 解释、LOPO 验证、时间序列特征工程和三种缺失值插补方法。

**📊 数据集**

利用公开的 SoccerMon 数据集（两赛季女子顶级球队训练、比赛、主观/客观负荷以及官方伤病记录）。

**📈 对比分析**

通过时间序列交叉验证（chronological split）和留一玩家交叉验证（LOPO）进行比较；DeepHit 的 C‑index 达到 0.762，显著优于基线模型（RF F1 0.533，XGBoost F1 0.429，LR F1 0.071），LOPO 下 C‑index 范围 0.192（IQR）至 0.974。

**⚠️ 局限性**

主要局限包括高缺失率、时间不规律、对个体的泛化差、未使用 RNN 或动态 DeepHit、样本量小、未覆盖多队和多层次特征，且缺乏实时部署与交互式界面。

---

## 314. The complexity of downward closures of indexed languages

**arXiv ID:** 2601.19466 | [PDF](https://arxiv.org/pdf/2601.19466v1)

**作者:** Richard Mandel `[一作]` (Max Planck Institute for Software Systems), Georg Zetzsche `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 602 | [OpenAlex ID](https://openalex.org/A5083429244)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究了索引语言的下闭包，并给出了有效的构造方法，证明其上界为三指数（非确定性自动机）和四指数（确定性自动机），并给出匹配的下界，证明界限紧凑。

**💡 创新点**

首次在理论层面完成了下闭包的复杂度定界，利用半群理论将索引文法转换为上下文无关文法，保留下闭包的性质，实现了从指数级到多指数级的精确匹配。

**🔧 技术方法**

主要技术包括半群（堆栈单子）理论、堆栈内容摘要、泵送与跳跃技术、以及上下文无关文法的构造与分析。

**📊 数据集**

本研究不涉及实验数据集，而是纯粹的理论分析与证明。

**📈 对比分析**

由于没有实验评估，性能表现仅体现在理论复杂度上：上界为三指数（NFA）/四指数（DFA），下界与之匹配，说明方法已达到最优。

**⚠️ 局限性**

主要局限在于实际实现的状态爆炸问题：即使理论上可行，构造出的自动机在规模上可能极大，难以直接用于大规模实际应用。

---

## 315. Physical Human-Robot Interaction: A Critical Review of Safety Constraints

**arXiv ID:** 2601.19462 | [PDF](https://arxiv.org/pdf/2601.19462v1)

**作者:** Riccardo Zanella `[一作]` (University of Twente), Stefano Stramigioli `[通讯]` (University of Twente)

**通讯引用:** 10912 | [OpenAlex ID](https://openalex.org/A5047812798)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统地推导了 ISO/TS 15066 所规定的 PFL（Power and Force Limiting）模式下的安全约束，详细阐明了推导过程中的关键假设，并通过能量视角分析了安全约束与机器人性能之间的关系；随后对不同设计选择（如机器人有效质量、人体质量、碰撞模式等）对可行速度与能量极限的影响进行了定量比较，并讨论了能量基安全策略的现有实现与未来改进方向。

**💡 创新点**

创新点包括：
1) 将 ISO 标准的痛阈值与机械能直接映射，形成可操作的速度/能量极限；
2) 明确并系统化阐述各假设（线性弹簧、瞬时碰撞、人体/机器人有效质量、粘性耗散等）的影响，避免了标准中隐含的模糊性；
3) 通过比较固定 ISO 有效质量与配置相关反射质量的差异，量化了姿态感知对性能提升的潜在收益；
4) 综合讨论了能量、功率、能量罐、控制屏障函数等多种能量基安全实现，指出其在传递能量、功率限制和可观测性方面的差异。

**🔧 技术方法**

使用的技术与方法主要包括：
- 质量‑弹簧‑质量（m_R–k–m_H）碰撞模型；
- 动能与弹性势能平衡推导；
- 机器人有效质量的两种计算（ISO 静态近似与基于雅可比的方向相关反射质量）；
- 统一的能量约束与速度约束公式；
- 数值仿真与网格化工作空间采样，评估不同假设下的速度极限分布；
- 参考文献中提出的能量罐、CBF、优化控制等能量基安全实现的综述与对比。

**📊 数据集**

所用数据集主要为：
- FP‑0317 项目在 100 名健康成年人的实验得到的疼痛阈值（力/压强）以及对应的弹性模量；
- DGUV/BGIA 早期的安全阈值与标准化方法；
- ISO/TS 15066 公开的体位有效质量表（如前臂 2 kg、大腿 75 kg 等）；
- 机器人参数（如 Franka Emika Panda 的惯性参数）及其逆运动学解。

**📈 对比分析**

比较方法：对同一工作空间点，分别使用（a）ISO 定义的常数有效质量与（b）方向相关的反射质量；（c）三种碰撞假设——瞬时（缩放因子 2）、准静态（不缩放）与完全钳制；在每种组合下计算可接受的最大速度并绘制箱线图。结果显示，使用动态有效质量平均可提升 20–70 % 的速度；在完全钳制假设下速度下降至 10–30 %；而在不考虑人体可动性的保守设定下，速度可下降 50 % 以上。性能提升的量化指标主要以速度极限的相对百分比下降来表示。

**⚠️ 局限性**

限制与不足：
- 模型假设线性弹簧、无耗散、瞬时碰撞，忽略了实际人机接触的复杂几何与多点接触；
- 人体有效质量表采用静态经验值，未考虑动态姿态变化导致的质量分布差异；
- 能量基阈值主要基于伤害阈值，缺少对疼痛级别的细粒度区分，可能在轻微碰撞场景下过于保守或不足；
- 文章未提供实际硬件实验验证，仅通过数值仿真与理论推导验证；
- 对持续接触（非碰撞）场景的安全评估缺失，难以覆盖协同装配、共搬运等长时间物理交互。

---

## 316. Preprocessing Uncertain Data into Supersequences for Sorting and Gaps

**arXiv ID:** 2601.19453 | [PDF](https://arxiv.org/pdf/2601.19453v1)

**作者:** Maarten Löffler `[一作]` (Utrecht University), Benjamin Raichel `[通讯]` (University of Texas at Dallas)

**通讯引用:** 306 | [OpenAlex ID](https://openalex.org/A5054913384)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

在不确定数据的预处理框架中，提出使用区间的超序列作为辅助结构，利用该结构在任意实现下高效恢复排序、最小/最大间距等输出。

**💡 创新点**

创新点在于将复杂的专用辅助结构简化为超序列，清晰分离预处理与重建阶段，支持子线性重建并给出（α,β）平滑超序列的构造与线性重建算法。

**🔧 技术方法**

主要技术包括构造（α,β）超序列的分块递归算法、利用最短通用词的二次上界、基于哈希的贪心重建算法，以及在Word RAM下的平滑性优化。

**📊 数据集**

论文仅给出理论分析与合成实例，未使用实际实验数据集。

**📈 对比分析**

在单位区间、最大重叠量Δ常数时，超序列长度为O(nΔ)，重建时间为O(nΔlogΔ)（Real RAM）或O(nΔ)（Word RAM），与已知最优结果相匹配。

**⚠️ 局限性**

主要限制包括对Δ的线性依赖、对Real RAM模型的平滑性假设、以及在不满足（α,β）条件时无法实现线性重建，存在开放问题。

---

## 317. From Internal Diagnosis to External Auditing: A VLM-Driven Paradigm for Online Test-Time Backdoor Defense

**arXiv ID:** 2601.19448 | [PDF](https://arxiv.org/pdf/2601.19448v1)

**作者:** Binyan Xu `[一作]` (Chinese University of Hong Kong), Kehuan Zhang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 3551 | [OpenAlex ID](https://openalex.org/A5008237643)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于VLM的在线外部语义审计框架PRISM，用来防御深度网络的后门攻击；

**💡 创新点**

创新点在于把防御从内部诊断迁移到外部语义审计，使用Hybrid VLM Teacher动态细化视觉原型，并通过Adaptive Router结合Cornish‑Fisher扩展实现自适应阈值；

**🔧 技术方法**

关键技术包括Vision‑Language模型（CLIP、Qwen、Gemma等）作为无监督审计器、在线原型细化、统计学阈值自适应与CMA更新；

**📊 数据集**

在17个数据集（CIFAR‑10/100、ImageNet、SVHN、GTSRB、MNIST、TinyImageNet等）上评估；

**📈 对比分析**

与8种SOTA防御方法对比，PRISM在11种后门攻击下将ASR降至<1%（CIFAR‑10）且提升CA，整体表现明显优于传统内部诊断和输入鲁棒性方法；

**⚠️ 局限性**

局限性包括对VLM自身偏差和hallucination的依赖，易受文本触发的语义欺骗攻击（如标注攻击）影响，且在极端污染比例下仍需更稳健的统计自适应机制。

---

## 318. It's Not Just a Phase: Creating Phase-Aligned Peripheral Metamers

**arXiv ID:** 2601.19425 | [PDF](https://arxiv.org/pdf/2601.19425v1)

**作者:** Sophie Kergaßner `[一作]` (Università della Svizzera italiana), Piotr Didyk `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 2212 | [OpenAlex ID](https://openalex.org/A5016434967)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于相位对齐的外围视觉等价体（metamer）生成方法，在凹面渲染（foveated rendering）基础上通过估计并外推局部强度、方向和相位统计量，合成缺失的高频细节，从而显著降低渲染成本。

**💡 创新点**

创新点在于：①首次将相位信息与方向和强度共同提取并外推用于外周渲染提升；②使用可驱动的可旋转四元数滤波器框架统一估计三种统计量；③设计基于冲击点的局部 Gabor 噪声合成，使合成频谱保持在目标频带内；④通过实验验证相位对齐能在保持视觉一致性的前提下实现约 4 倍的阴影率降低。

**🔧 技术方法**

核心技术包括：
- 可旋转四元数（steerable quadrature）滤波器分析与合成；
- 多尺度高斯金字塔分解与统计量外推；
- 相位对齐与跨尺度相位一致性计算；
- 基于冲击点的 Gabor‑噪声频带合成；
- GPU 并行实现与临时稳定性保证。

**📊 数据集**

数据集：采用多种复杂自然场景（Blender Studio 提供的 6 幅全分辨率图像）、实验参与者 26–27 岁的 11 名和 15 名志愿者；实验场景包括多种颜色、纹理与几何复杂度的图像，使用 LG OLED 55 英寸显示器进行主观评估。

**📈 对比分析**

与传统凹面渲染、对比增强、仅方向恢复等方法比较，使用 2AFC 主观实验测定可接受的阴影率阈值。结果显示：
- 采用相位对齐后，阴影率可提高约 4 倍（≈ 4× 阴影率降低）且 75% 检测阈值显著提升；
- 与现有噪声合成方法（tariq‑2022a）比较，本文方法在非对比增强条件下 83% 被受试者偏好，若加对比增强仍达 75%；
- 在强化对比的基线中，本文相位对齐提升约 129%，相当于该方法 39% 的进一步提升。

**⚠️ 局限性**

局限性：
- 频率阈值离散化在金字塔层级，无法实现连续频带控制；
- 相位外推会导致边缘多重化（出现多条近似线条）且精细边缘恢复受限；
- 合成核小且固定频带，限制了频率分辨率和对频谱定位的精细控制。

---

## 319. For Generalised Algebraic Theories, Two Sorts Are Enough

**arXiv ID:** 2601.19426 | [PDF](https://arxiv.org/pdf/2601.19426v1)

**作者:** Samy Avrillon `[一作]` (École normale supérieure de Lyon), Johann Rosain `[通讯]` (National Centre for Scientific Research)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文证明任何广义代数理论（GAT）都可以转化为仅包含两种类型（U和El）的两排序理论，并建立原始理论与其两排序化之间模型类别的严格核心反射；同时给出了两排序化的构造、核心映射以及初始模型的存在性证明。

**💡 创新点**

创新点在于利用Uemura关于有限GAT的双初性特性，从语义层面（而非语法）构造两排序化的终极翻译，并在此基础上给出模型核心化、核心映射与初始模型的完整证明；该方法消除了排序等式、交错排序和运算等限制，扩展到无限GAT，并实现了两排序化的完全忠实性。

**🔧 技术方法**

主要技术包括：
1. 通过Uemura的双初性定理在笛卡尔范畴与指数化映射上构造初性（核心）
2. 在切片范畴中定义指数化映射并证明其指数化性
3. 使用语义翻译与模型函子（模型函子保留指数化的右伴随）来实现核心反射
4. 采用Yoneda引理与自由滤波完成来推广到无限GAT
5. 通过核心映射构造右伴随并证明其为严格核心化。

**📊 数据集**

无数据集，论文为理论性研究。

**📈 对比分析**

无实验比较，主要通过理论证明与构造来验证方法的正确性。

**⚠️ 局限性**

限制与挑战：
1. 目前仅在有限与无限GAT上证明，尚未覆盖包含无限运算或二阶GAT的情形。
2. 需要依赖Uemura的双初性结果，若该结果不适用于更广泛的语义框架，方法可能不直接迁移。
3. 在具体实现时，如Cubical Agda等现有系统仍无法支持所有核心化所需的排序等式或交错构造，需额外工作。

---

## 320. ALRM: Agentic LLM for Robotic Manipulation

**arXiv ID:** 2601.19510 | [PDF](https://arxiv.org/pdf/2601.19510v1)

**作者:** Vitor Gaboardi dos Santos `[一作]` (Dublin City University), Hakim Hacid `[通讯]` (Technology Innovation Institute)

**通讯引用:** 2016 | [OpenAlex ID](https://openalex.org/A5013316111)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ALRM 框架，将 LLM 作为代理进行机器人操作，支持 Code‑as‑Policy 与 Tool‑as‑Policy 两种执行模式，并发布 56 条多语言多步指令的仿真基准。

**💡 创新点**

将 ReAct 交互式推理与 LLM 代码/工具调用相结合，实现闭环规划、反思与动态修正；同时设计多语义、多步骤基准评估 LLM 机器人推理能力。

**🔧 技术方法**

使用 LLM 生成策略与执行、ReAct 思考‑行动循环、Code‑as‑Policy、Tool‑as‑Policy 两种模式、Gazebo+ROS+MoveIt 仿真环境，以及 LLM‑as‑Judge 评估方法。

**📊 数据集**

自制三套仿真环境（厨房、盒子、水果），每套 3 个任务，共 54 条规范指令，扩展 5 种语言变体共 56 条任务。

**📈 对比分析**

对 10 个 LLM（含 GPT‑5、Claude‑4.1‑Opus、Falcon‑H1‑7B 等）在两种执行模式下，以词汇、句法、语义、高层推理等 5 类指令评估，采用成功率（0/1/2）与平均延迟两指标；Claude‑4.1‑Opus 在 TaP 模式下 93.5% 成功率，Falcon‑H1‑7B 在 CaP 模式下 84.3% 成功率，整体显示大模型在闭环推理上优势，小模型在直接代码生成上表现突出。

**⚠️ 局限性**

依赖特定提示与模型，基准仅限 pick‑and‑place 三个环境且未包含真实感知或导航，工具调用在小模型上效果差，需进一步扩展与真实机器人验证。

---

## 321. Automated Safety Benchmarking: A Multi-agent Pipeline for LVLMs

**arXiv ID:** 2601.19507 | [PDF](https://arxiv.org/pdf/2601.19507v1)

**作者:** Xiangyang Zhu `[一作]` (Shanghai AI Lab), Wei Sun `[通讯]` (East China Normal University)

**通讯引用:** 17188 | [OpenAlex ID](https://openalex.org/A5100662256)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VLSafetyBencher——一个全自动的多模态安全基准构建与更新系统，利用四个协同代理（预处理、生成、增广、选择）实现从原始数据到高质量安全基准的全流程；

**💡 创新点**

创新点在于：①将基准构建自动化为多代理协同流程；②提出跨模态交互的三种生成策略（模态依赖、互补、冲突）保证图文共同决定危害；③通过分离式可分离性、危害性、多样性三项指标的优化采样算法，得到全局最优基准；

**🔧 技术方法**

核心技术包括：大型语言模型驱动的代理编排（DeepSeek‑V3）、CLIP视觉文本编码与相似度计算、对抗式文本增广、图像增广、信息熵与多模态距离的数学建模以及迭代的加权优化采样；

**📊 数据集**

使用的原始数据来源包括：已存在的安全数据集、通用图像集、Diffusion 生成图像、社交媒体抓取图像，经过CLIP粗筛后约30万图像；

**📈 对比分析**

与手工与自动化基准（SafeBench、MLLMGuard、AutoBencher、DataGen、DME）对比，VLSafetyBencher在平均绝对偏差、平均攻击成功率、分数差距和多样性上均优于现有方法；实验显示可在一周内完成基准构建，成本仅约1.34美元；

**⚠️ 局限性**

局限性包括：仍需依赖外部大模型进行生成和评判；对极端或新型危害场景的覆盖可能不足；增广和选择策略的超参数敏感，需手工调优；未对长期动态更新机制的持续可行性做充分实验；

---

## 322. Bridging Information Asymmetry: A Hierarchical Framework for Deterministic Blind Face Restoration

**arXiv ID:** 2601.19506 | [PDF](https://arxiv.org/pdf/2601.19506v1)

**作者:** Zhengjian Yao `[一作]` (Peking University), Yanye Lu `[通讯]` (Peking University)

**通讯引用:** 1360 | [OpenAlex ID](https://openalex.org/A5082320568)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Pref‑Restore 框架，用自回归语义整合器与连续扩散生成器双层结构，实现盲人脸恢复的确定性与高保真重建。

**💡 创新点**

创新点在于：① 将文本指令转化为稠密语义查询，弥补低质量输入的信息缺失；② 在扩散循环中直接采用在线强化学习（DiffusionNFT），将人类偏好转化为可微约束，从而裁剪输出分布，消除伪影与不确定性。

**🔧 技术方法**

技术手段包括自回归模型（Qwen3）、多模态知识对齐、DiT/SANA 等扩散模型、DiffusionNFT 强化学习、CLIP/CLIPScore 等视觉语言模型。

**📊 数据集**

训练使用 FFHQ；评估在合成 CelebA-Test、以及真实场景 LFW-Test、WIDER-Test、WebPhoto-Test、CelebChild-Test 等多种数据集，并通过人工合成的降解模型生成低质量输入。

**📈 对比分析**

与多类基线（GAN、VQ、扩散）在 LPIPS、FID、MUSIQ、CLIPIQA、ArcFace、LMD 等指标下对比，Pref‑Restore 在结构一致性、身份保持、无参考质量以及风格美观上均超越现有最优方法；尤其是质量版在无参考评估中领先显著。

**⚠️ 局限性**

局限性：① 质量版在提升美感时可能抹除细微身份特征；② 在极端降解下仍可能残留伪影；③ 目前仅验证图像级别效果，未扩展至视频或更复杂多模态任务。

---

## 323. GradPruner: Gradient-Guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs

**arXiv ID:** 2601.19503 | [PDF](https://arxiv.org/pdf/2601.19503v1)

**作者:** Wei Huang `[一作]` (Ant Group), Yinggui Wang `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过在 LLM 微调的早期阶段收集梯度，构建 IGIA‑Matrix 评估层的重要性，并在保持模型结构的前提下，对不重要层进行剪枝与层级合并，从而实现参数 40% 的压缩。

**💡 创新点**

①使用梯度累积得到的 IGIA‑Matrix 作为层重要性度量，避免传统校准数据导致的偏差；②提出基于梯度符号的层合并策略，在稀疏化后仅合并符号相同的参数，显著降低合并带来的干扰。

**🔧 技术方法**

LoRA 微调、梯度累积、IGIA‑Matrix 计算、层级稀疏化与符号合并、实验评估与基线对比。

**📊 数据集**

使用 Llama3.1‑8B 与 Mistral‑7B 两大 LLM，八个下游数据集：PubMedQA、MedMCQA、BillSum、FinGPT、HellaSwag、WinoGrande、ARC、PIQA。

**📈 对比分析**

与 APT、SAT、LLMPruner、LaCo、MINITRON 以及 Llama3.2‑3B 微调模型进行对比。结果表明，GradPruner 仅下降 0.99% 的平均准确率，且训练时间下降 36%、推理时间下降 39%，显著优于其他基线。

**⚠️ 局限性**

1) 对剪枝率过高或过低时准确率会下降；2) 仅在层级进行剪枝与合并，未针对核层进一步细粒度优化；3) 依赖早期梯度信息，若任务学习曲线不同可能导致重要性评估失真；4) 目前仅在两大模型和八个数据集验证，尚需验证在更大规模模型和不同领域的通用性。

---

## 324. On the Expressiveness of State Space Models via Temporal Logics

**arXiv ID:** 2601.19467 | [PDF](https://arxiv.org/pdf/2601.19467v1)

**作者:** Eric Alsmann `[一作]` (University of Kassel), Martin Lange `[通讯]` (University of Kassel)

**通讯引用:** 1483 | [OpenAlex ID](https://openalex.org/A5087768042)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

对状态空间模型（SSM）的表达能力进行理论分析，并与 Transformer 架构在逻辑和复杂度层面进行对比。

**💡 创新点**

首次将 SSM 的门控机制、算术精度与线性时序逻辑（LTL）、模运算、计数扩展三维度结合，构建完整的表达层级，揭示不同 SSM 变体的严格包含关系。

**🔧 技术方法**

使用线性时序逻辑、模块谓词、计数算子、复杂度类 TC⁰、AC⁰、FO[<] 等理论工具，以及构造性门控矩阵与递归状态更新。

**📊 数据集**

无实验数据集，全部为理论证明和逻辑归纳。

**📈 对比分析**

通过逻辑等价与复杂度类映射，定量描述各 SSM 变体的可识别语言类；与 UHAT、UHAT+PE、AHAT、SAT 等 Transformer 变体在表达力上进行层级对比，未给出数值性能指标。

**⚠️ 局限性**

仅给出下界证明，缺乏上界细化；未验证在实际训练中的可实现性；对固定精度 SSM 的能力可能受限于缺乏模计数或前瞻计数机制。

---

## 325. Towards Gold-Standard Depth Estimation for Tree Branches in UAV Forestry: Benchmarking Deep Stereo Matching Methods

**arXiv ID:** 2601.19461 | [PDF](https://arxiv.org/pdf/2601.19461v1)

**作者:** Yida Lin `[一作]` (Victoria University of Wellington), Richard Green `[通讯]` (University of Canterbury)

**通讯引用:** 17125 | [OpenAlex ID](https://openalex.org/A5100730173)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在无人机林业作业中，对树枝深度估计进行了零样本评估，比较了八种深度匹配方法，并将DEFOM模型确立为伪真值基准。

**💡 创新点**

首次在森林密集环境下进行跨域零样本评估，系统比较了迭代、基础模型、扩散和3D CNN四大架构，并提出DEFOM作为金标准；同时公开新收集的树枝数据集。

**🔧 技术方法**

使用Scene Flow预训练权重的深度立体匹配模型（RAFT‑Stereo、IGEV、IGEV++、BridgeDepth、DEFOM、StereoAnywhere、ACVNet、PSMNet），通过EPE和D1指标进行评估。

**📊 数据集**

采用标准的ETH3D、KITTI 2012/2015、Middlebury三大基准数据集，以及新收集的5,313对Canterbury树枝高分辨率（1920×1080）数据集。

**📈 对比分析**

通过零样本推理，比较EPE与D1排名，DEFOM在四个基准上平均排名1.75，表现最稳健；其他模型在某些基准上表现优异但在大视差范围或复杂场景下易失效。

**⚠️ 局限性**

实验仅做定性比较，缺少统计显著性检验；未使用LiDAR等真实深度标注；未探索多模型融合或领域自适应技术。

---

## 326. APC-RL: Exceeding Data-Driven Behavior Priors with Adaptive Policy Composition

**arXiv ID:** 2601.19452 | [PDF](https://arxiv.org/pdf/2601.19452v1)

**作者:** Finn Rietz `[一作]` (Orebro University), Johannes Andreas Stork `[通讯]` (Orebro University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出Adaptive Policy Composition (APC)，一种层级强化学习框架，能够自适应地组合多条演示数据驱动的正则化先验（Normalizing Flow）和无先验的基础演员；

**💡 创新点**

创新点在于①引入无先验演员与可学习的选择器，以避免对不对齐演示的过度依赖；②采用奖励共享技术，使所有演员均可利用每一步的经验；③使用学习无关的arbitrator selector，减少层级学习中的偏差与不稳定；

**🔧 技术方法**

核心技术包括Normalizing Flow（NF）行为先验、Soft Actor-Critic (SAC) 的离线学习、离线奖励共享、无学习的arbitrator选择器以及多演员并行更新；

**📊 数据集**

使用D4RL中的Maze Navigation、Franka Kitchen、CarRacing等连续控制基准，以及人类驾驶的CarRacing演示数据；

**📈 对比分析**

与SAC、IL、QFilter、PARROT等基线对比，APC在演示对齐时保持与PARROT相近的学习速率，在演示不对齐时显著优于其他方法（甚至超过从零开始的SAC），并能突破子最优演示的性能上限；

**⚠️ 局限性**

主要限制包括计算成本随演员数量线性增长、在多条严重不对齐演示且奖励稀疏的极端情形下可能无法有效分辨优劣、以及对大规模多先验场景的扩展性有限。

---

## 327. Dynamic Multi-Expert Projectors with Stabilized Routing for Multilingual Speech Recognition

**arXiv ID:** 2601.19451 | [PDF](https://arxiv.org/pdf/2601.19451v1)

**作者:** Isha Pandey `[一作]` (Indian Institute of Technology Bombay), Ganesh Ramakrishnan `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 2044 | [OpenAlex ID](https://openalex.org/A5089606464)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了SMEAR-MoE多专家项目器，用于将冻结的语音编码器与大型语言模型（LLM）连接，实现多语音识别。

**💡 创新点**

创新点在于采用软门控将所有专家参数按门控权重加权得到虚拟专家，保证每个专家在训练中都能收到密集梯度，避免专家崩溃，同时实现跨语言共享与专家专化的平衡。

**🔧 技术方法**

技术细节包括：Mixture-of-Experts（MoE）架构、SMEAR软门控、卷积下采样、Whisper large‑v3 语音编码器、Gemma‑2‑9B LLM、负载平衡损失、Beam Search 推理等。

**📊 数据集**

使用的数据集为 IndicVoices、IndicSUPERB（约每语种 250 小时）以及 VISTAR 基准（Kathbath、MUCS、IndicTTS、Fleurs），覆盖印地语、马拉地语、泰米尔语、特立尼达语四种印度语言。

**📈 对比分析**

与单一项目器、语言专属项目器、Tied 项目器、Dense Ensemble、Utterance/Token‑Level MoE 等多种对照模型在 WER/CER 上进行系统评测，SMEAR‑MoE 在四种语言上平均 WER 下降 7.6% 相对单项目器基线，整体排名第一，并且实时因子（RTF）与单项目器相近（≈0.20），体现了性能与效率兼顾。

**⚠️ 局限性**

局限性包括：实验仅覆盖四种印度语言，未验证在更大规模或更低资源语言上的泛化能力；模型对极端噪声或多说话人情境的鲁棒性尚待进一步评估；同时对不同 LLM 后端的兼容性与可扩展性仍需研究。

---

## 328. RoamScene3D: Immersive Text-to-3D Scene Generation via Adaptive Object-aware Roaming

**arXiv ID:** 2601.19433 | [PDF](https://arxiv.org/pdf/2601.19433v1)

**作者:** Jisheng Chu `[一作]` (Harbin Institute of Technology), Xiaopeng Fan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2729 | [OpenAlex ID](https://openalex.org/A5079412089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为RoamScene3D的端到端框架，实现从自然语言描述生成沉浸式、全景式的3D场景，支持可视化漫游与交互；

**💡 创新点**

创新点在于（1）利用VLM构建语义场景图，对对象关系进行推理，从而生成自适应的、以显著对象为中心的闭合漫游轨迹；（2）在此基础上设计了运动注入式RGBD全景修复模型，显式将摄像机运动信息编码进UNet，实现对大视角变化下遮挡区域的连贯补全；（3）将补全后的多视图融合到3D Gaussian Splatting中，兼顾几何准确性与视觉逼真度；

**🔧 技术方法**

关键技术包括：预训练的2D文本到图像扩散模型（FLUX）、视差与法线估计、Sphere Distance Field优化、VLM语义图谱生成、SAM+CLIP实例分割、运动编码+LoRA细调的UNet修复网络、三维高斯喷射（3DGS）优化；

**📊 数据集**

主要使用的训练与评估数据集为基于Matterport3D的Habitat环境生成的49,987帧带摄像机位姿的全景数据集，用于构建运动注入式修复模型；此外利用公开的文本提示集进行多场景评测；

**📈 对比分析**

与Text2Room、LucidDreamer、DreamScene360、LayerPano3D、PERF、Pano2Room等六个基线对比，RoamScene3D在BRISQUE、NIQE、CLIP-Score、Inception Score、CLIP-IQA等多项指标均取得领先，且推理时间约23分钟，兼具高质量与可接受的速度；

**⚠️ 局限性**

局限性包括：仍高度依赖2D扩散先验，对极其复杂或多层遮挡场景的恢复有限；需要较大显存与训练资源；语义图谱的构造对VLM质量敏感，可能导致不一致或误判；以及对非结构化文本提示的鲁棒性尚待进一步提升。

---

## 329. Interior--Boundary Assortativity Profiles on Networks and Applications to SIS Epidemic Dynamics

**arXiv ID:** 2601.19422 | [PDF](https://arxiv.org/pdf/2601.19422v1)

**作者:** Moses Boudourides `[一作]` (Northwestern University), Moses Boudourides `[通讯]` (Northwestern University)

**通讯引用:** 295 | [OpenAlex ID](https://openalex.org/A5035035192)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出“内部–边界混 assortativity profile”，在给定网络划分的前提下把边分为内部到内部、内部到边界、边界到内部、边界到边界四种类型，分别计算其 assortativity，从而细化传统 Newman assortativity 的单一数值。

**💡 创新点**

创新点在于：①给出严格的分解定理，证明单一 assortativity 是内部–边界组件的加权组合并伴随跨类型均值偏移；②将 SIS 病毒传播的稳态感染概率作为节点属性，证明低导向电导率（即界面瓶颈）导致边界节点感染概率上升，从而产生显式负的边界→内部 assortativity；③把动态系统与网络结构之间的联系通过谱电导和混合度量形式化。

**🔧 技术方法**

主要技术包括：谱图理论（导向 Cheeger 定理、拉普拉斯谱）、非线性动力学（SIS 模型的存在与稳定性）、概率统计（边类型的协方差分解）以及图论中的边界、内部概念与参与系数。

**📊 数据集**

论文是理论性工作，未使用具体实验数据集；所有结果均通过严格证明给出，部分讨论以随机块模型为例进行解释。

**📈 对比分析**

因为没有实现算法或实验，无法进行方法比较；论文的主要贡献是理论证明和公式推导，而非性能评估。

**⚠️ 局限性**

局限性包括：①划分是预先给定且固定，无法自适应学习；②主要关注 SIS 动态，其他传播或同步模型需进一步推广；③需要界面稀疏（低电导）才能保证边界主导性；④在实际大规模网络中计算内部–边界 assortativity 可能计算量较大。

---

## 330. Ad Insertion in LLM-Generated Responses

**arXiv ID:** 2601.19435 | [PDF](https://arxiv.org/pdf/2601.19435v1)

**作者:** Shengwei Xu `[一作]` (University of Michigan), Grant Schoenebeck `[通讯]` (University of Michigan)

**通讯引用:** 2386 | [OpenAlex ID](https://openalex.org/A5073672615)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种将广告无缝插入LLM回复的框架，先独立生成广告自由文本，再通过广告版位插入；

**💡 创新点**

创新点在于将投标与实时上下文解耦为对“genre”（高层语义类别）的投标，结合VCG拍卖实现近似DSIC与社群福利最大化；

**🔧 技术方法**

使用了Vickrey‑Clarke‑Grove（VCG）拍卖、LLM‑as‑a‑Judge（利用大模型评估语境一致性）以及句子嵌入的相似度测量；

**📊 数据集**

实验数据包括48名用户的问卷与人工评估、7个典型用户意图提示、10个广告genre以及OpenAI GPT‑4o生成的LLM回复；

**📈 对比分析**

与人工评分对比，LLM‑as‑a‑Judge在Spearman相关系数上达到约0.66，超过80 %的人类评审；VCG在最多10⁵名广告主、100个插槽的情形下，平均运行时间约1.25 s，远低于LLM推理延迟；

**⚠️ 局限性**

局限性包括需依赖外部数据进行概率校准、genre划分粗细折中导致估价误差、缺乏用户个性化与多轮会话动态规划。

---

## 331. Fixed Aggregation Features Can Rival GNNs

**arXiv ID:** 2601.19449 | [PDF](https://arxiv.org/pdf/2601.19449v1)

**作者:** Celia Rubio-Madrigal `[一作]` (CISPA Helmholtz Center for Information Security), Rebekka Burkholz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Fixed Aggregation Features（FAFs）方法，将图节点特征的邻域聚合固定化并转化为表格数据，仅训练MLP实现节点分类；

**💡 创新点**

创新点在于证明非可学习的固定聚合即可与GNN竞争，并通过Kolmogorov–Arnold定理提供理论基础，同时引入可解释、可扩展的表格学习框架；

**🔧 技术方法**

使用了多种固定聚合器（mean、sum、max、min、std、Kolmogorov–Arnold函数）在多跳上进行特征拼接，并训练多层感知机（MLP）以及SHAP等解释工具；

**📊 数据集**

评估数据集覆盖14个常用节点分类基准，包括Cora、Citeseer、Pubmed、Amazon-Computer/Photo、Coauthor-CS/Physics、WebKB、WikiCS、Minesweeper、Roman Empire、Squirrel等；

**📈 对比分析**

与GCN、GAT、GraphSAGE、Graph Transformer等经典GNN进行对比，FAFs在12/14数据集上匹配或优于现有方法，仅在Minesweeper和Roman Empire略逊；

**⚠️ 局限性**

局限在于对高跳信息的丢失、需要长距离依赖的任务表现不足，Kolmogorov–Arnold聚合训练困难，且对极端异质性数据的泛化仍待提升。

---

## 332. R^3: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning

**arXiv ID:** 2601.19620 | [PDF](https://arxiv.org/pdf/2601.19620v1)

**作者:** Zhizheng Jiang `[一作]` (University of Electronic Science and Technology of China), Peng Han `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 10165 | [OpenAlex ID](https://openalex.org/A5028719633)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型在数学推理任务中的优势估计不稳定，提出R³框架来提升推理性能。

**💡 创新点**

创新点：①跨上下文回放（CCR）在同一批次注入历史样本保持组内优势；②上下文自我反思（ISR）利用失败历史引导模型自纠；③结构熵排名奖励（SERR）为截断或失败样本提供无监督细粒度奖励。

**🔧 技术方法**

技术：基于GRPO的分组策略，结合经验回放、熵计算与排名、PPO强化学习、内部奖励机制与KL正则化。

**📊 数据集**

使用DeepScaleR-40k数学训练集，评测使用AIME 2024、MATH500、AMC、Minerva、OlympiadBench等标准数学基准。

**📈 对比分析**

与多种RL基线（DeepScaleR、O1、Saturn、Thinker等）对比，R³在5个基准上均实现SOTA，1.5B模型超过部分7B模型，推理token更少且效率更高。

**⚠️ 局限性**

局限：需要手工调参（阈值、奖励比例等），奖励设计对不同任务的泛化尚未充分验证，且在极难题上仍有解题率限制。

---

## 333. Decompose-and-Formalise: Recursively Verifiable Natural Language Inference

**arXiv ID:** 2601.19605 | [PDF](https://arxiv.org/pdf/2601.19605v1)

**作者:** Xin Quan `[一作]` (University of Manchester), André Freitas `[通讯]` (University of Manchester)

**通讯引用:** 2480 | [OpenAlex ID](https://openalex.org/A5053978668)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可递归验证的自然语言推理框架 LLM-TP Tree，利用树形蕴含结构、原子拆解和 θ‑替换自动形式化，实现局部诊断和精细化改写；

**💡 创新点**

创新点包括：①树形蕴含结构的递归底层验证与局部修复；②原子拆解降低推理错误扩散；③θ‑替换多步形式化提升语义忠实度；

**🔧 技术方法**

技术组合为：大语言模型自动生成蕴含树与原子拆解、基于 Neo‑Davidsonian 事件语义的多步 θ‑替换形式化、外部定理证明器（Isabelle/HOL 等）进行证据检查及诊断引导改写；

**📊 数据集**

在四大数据集上评估：FOLIO、ProofWriter、PrOntoQA、EntailmentBank，使用五种 LLM 后端（GPT‑4o、GPT‑5 nano、Grok‑4‑fast、Deepseek‑V3.1、Qwen3‑Max）；

**📈 对比分析**

相较于 Explanation‑Refiner、Faithful‑Refiner 等全局重写方法，LLM‑TP Tree 在解释验证率上提升 26.2%–48.9%，减少迭代次数与运行时，且保持甚至提升最终 NLI 预测准确度；

**⚠️ 局限性**

局限性在于依赖自动形式化函数、目标逻辑与定理证明器，可能无法完全捕捉原文语义、隐含常识与话语现象；初始树结构不佳时难以完全恢复正确链；多轮 LLM 与定理求解仍带来计算成本与超时风险。

---

## 334. Comparing how Large Language Models perform against keyword-based searches for social science research data discovery

**arXiv ID:** 2601.19559 | [PDF](https://arxiv.org/pdf/2601.19559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 335. Yunque DeepResearch Technical Report

**arXiv ID:** 2601.19578 | [PDF](https://arxiv.org/pdf/2601.19578v1)

**作者:** Yuxuan Cai `[一作]` (Tencent), Zheng Wei `[通讯]` (Tencent)

**通讯引用:** 776 | [OpenAlex ID](https://openalex.org/A5060067799)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Yunque DeepResearch，一个层次化、模块化、鲁棒的多智能体框架，用于解决长周期任务中的认知负荷、系统脆弱性和扩展性不足的问题，并在多个公开基准上取得最先进的成绩。

**💡 创新点**

创新点包括：① 中央化多智能体编排系统，将任务动态路由到原子能力池中的工具和专用子代理；② 子目标驱动的动态上下文管理，通过语义摘要压缩历史，减轻信息噪声；③ 主动监督模块，实时检测异常并清理无效上下文，提升系统鲁棒性；④ 高度模块化的原子能力池，支持轻量化子代理和工具的热插拔。

**🔧 技术方法**

技术手段主要包括：多智能体编排（ReAct + 工具调用）、子目标驱动的记忆管理、异常检测与自恢复的监督层、基于 LLM 的推理与规划、轻量级子代理工作流、工具封装（搜索、读取、代码执行）以及结构化语义摘要。

**📊 数据集**

使用的数据集包括 GAIA、BrowseComp、BrowseComp‑ZH、Humanity’s Last Exam，并在这些基准上与多种通用 LLM、开源和闭源代理框架进行了对比实验。

**📈 对比分析**

与通用 LLM + ReAct、开源框架（DeepAgent、OAgent 等）以及闭源框架（Gemini Deep Research 等）相比，Yunque DeepResearch 在 BrowseComp、BrowseComp‑ZH、Humanity’s Last Exam 上均取得最高分，GAIA 上排名第二，总体表现显著优于现有任何代理系统。

**⚠️ 局限性**

局限性包括：评估范围主要聚焦公开基准，未对专用子代理在更细粒度领域任务上的表现做系统验证；未对 token 消耗和推理延迟做深入分析；子代理虽轻量但整体执行时间仍受底层 LLM 推理能力影响；未来工作计划扩展到 DSBench、OSWorld 等域特定基准，并探索后训练的原子能力以降低成本。

---

## 336. MaDiS: Taming Masked Diffusion Language Models for Sign Language Generation

**arXiv ID:** 2601.19577 | [PDF](https://arxiv.org/pdf/2601.19577v1)

**作者:** Ronglai Zuo `[一作]` (Imperial College London), Stefanos Zafeiriou `[通讯]` (Imperial College London)

**通讯引用:** 22010 | [OpenAlex ID](https://openalex.org/A5080553022)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MaDiS，一种基于掩码扩散语言模型（MDLM）的手语生成框架，能够从文本自动生成 3D 手语运动；

**💡 创新点**

创新点包括：①将 MDLM 引入手语生成，实现双向上下文建模与并行多 token 采样；②三层跨模态预训练（token、潜在空间、物理空间）提升语义与运动一致性；③设计时间检查点的 Unmasking 策略，显著降低生成顺序复杂度并加速收敛；④引入 Mixture‑of‑Parts embedding 层，通过可学习门控融合不同身体部位的编码；

**🔧 技术方法**

采用了掩码扩散语言模型、VQ‑VAE 分层 tokenizer、VAE 解码器、MLP 与可学习门控、时间检查点 Unmasking、SiBLEU 与 SiCLIP 评估方法及 CLIP 对齐；

**📊 数据集**

使用了三大公开手语数据集：CSL‑Daily、Phoenix‑2014T 与 How2Sign；

**📈 对比分析**

与现有方法（SOKE、S‑MotionGPT、MoMask++ 等）在 DTW‑JPE、SiBLEU、SiCLIP 等指标对比，MaDiS 在所有三个数据集上均达成 SOTA，DTW‑JPE 均下降 1.2–2.6，SiBLEU 提升 1–4，SiCLIP 提升 3–6，且推理延迟减少约 30%；

**⚠️ 局限性**

局限性在于仍需大量标注视频数据、仅生成运动序列需进一步视频合成、对极长句或多语言生成的鲁棒性不足、训练成本较高。

---

## 337. Learning Adaptive Parallel Execution for Efficient Code Localization

**arXiv ID:** 2601.19568 | [PDF](https://arxiv.org/pdf/2601.19568v1)

**作者:** Ke Xu `[一作]` (Ant Group), Yong Li `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 FuseSearch 的代码定位代理，采用自适应并行工具调用来提升定位速度与准确性。

**💡 创新点**

创新点在于引入工具效率（信息增益比）作为奖励，结合 SFT 与 RL 双目标训练，学习到从探索到聚焦的动态并行策略；同时仅使用三种语言无关的只读工具，极大降低部署复杂度。

**🔧 技术方法**

使用两阶段训练：首先通过教师引导的 SFT 产生高质量并行轨迹；随后采用基于 GRPO 的 RL，使用 F1 与工具效率的乘积奖励来联合优化定位质量与效率；并行执行采用 JSON 格式多工具调用。

**📊 数据集**

主要数据集为 233 个高质量 GitHub 仓库构建的定位样本，评测基准为 SWE‑bench Verified（Python 仓库）以及 LocBench 进行进一步验证。

**📈 对比分析**

与现有工作（如 RepoSearcher、CoSIL、LocAgent 等）对比，FuseSearch 在 SWE‑bench Verified 上实现文件级 F1 84.7%、函数级 F1 56.4%，相较于基线提升约 30% 以上；同时交互轮次下降 67.7%，耗时下降 93.6%，token 下降 68.9%，显示出显著的速度与成本优势。

**⚠️ 局限性**

局限性包括：评测仅基于 Python 仓库，未覆盖 Java/C++ 等静态语言；ground‑truth 仅为单一补丁，可能忽略其他有效定位方案；缺乏对更广泛任务（如仓库问答、文档生成）的评估。

---

## 338. The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments

**arXiv ID:** 2601.19557 | [PDF](https://arxiv.org/pdf/2601.19557v1)

**作者:** Riccardo Giubilato `[一作]`, Rudolph Triebel `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建并发布了一套面向行星探测的多模态SLAM数据集S3LI，并提供了校准工具和示例脚本。

**💡 创新点**

首次结合立体摄像机、固态LiDAR、IMU和差分GNSS于同一传感器平台，并在地球火山岛进行长时间多场景记录，模拟行星探测条件。

**🔧 技术方法**

使用RGB立体摄像机、MEMS驱动的固态LiDAR、工业级IMU、GNSS以及PTP时间同步，配合ROS录制与校准工具。

**📊 数据集**

在意大利维尔坎诺火山岛记录的多场景数据，包括熔岩路径、沉积结构、玄武岩、植被与水体等，构成S3LI数据集。

**📈 对比分析**

本文未进行算法对比实验，只提供了运行示例脚本，待后续研究验证其性能。

**⚠️ 局限性**

仅覆盖单一火山岛环境，可能缺乏极端行星地形多样性，动态场景处理仍有挑战，数据量有限。

---

## 339. SLM-SS: Speech Language Model for Generative Speech Separation

**arXiv ID:** 2601.19533 | [PDF](https://arxiv.org/pdf/2601.19533v1)

**作者:** Tianhua Li `[一作]` (Shanghai Jiao Tong University), Yanmin Qian `[通讯]` (iFLYTEK Company Limited)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于语音语言模型的语音分离框架SLM-SS，将连续语音量化为多码本序列，使用自回归与非自回归模型生成分离语音。

**💡 创新点**

创新点在于将语音语言模型与量化编码器结合，采用序列化输出训练（SOT）构造多说话人序列，并引入非自回归高阶码本预测提升解码效率。

**🔧 技术方法**

使用Encodec做量化编码、WavLM提取特征、Whisper风格解码器以及混合AR/NAR Transformer架构，并使用任务与位置嵌入。

**📊 数据集**

在LibriMix数据集（100h/360h训练集和测试集）上进行实验。

**📈 对比分析**

通过与BSRNN、Sepformer两种基线模型（采用PIT训练）对比，SLM-SS在WER、LPS、SBS等指标上明显优于基线，语音可懂度和下游ASR性能得到提升。

**⚠️ 局限性**

受限于Encodec码本数量与NAR模型训练难度，最终仅使用前8阶码本，导致恢复质量仍低于原始波形，且高阶码本扩展受限。

---

## 340. Rhombot: Rhombus-shaped Modular Robots for Stable, Medium-Independent Reconfiguration Motion

**arXiv ID:** 2601.19529 | [PDF](https://arxiv.org/pdf/2601.19529v1)

**作者:** Jie Gu `[一作]` (Fudan University), Dan Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 19640 | [OpenAlex ID](https://openalex.org/A5100456041)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了Rhombot——一种单轴可变形平面网格模块化自重构机器人，实现了无介质环境下的稳定重构、联接和操控。

**💡 创新点**

创新点在于将可变形模块与单DoF电缆驱动相结合，提出了“morphPivoting”重构原语，简化控制同时兼顾网格与链条双重功能。

**🔧 技术方法**

采用电缆驱动伺服、电磁连接、三角形多连杆骨架、无线通信及角度编码器等硬件技术，配合基于模块树的运动学模型与算法实现。

**📊 数据集**

未使用公开数据集，所有实验采用自制Rhombot原型进行物理测试与仿真。

**📈 对比分析**

与Metamorphic、PARTS、M-Blocks等先前MSRR进行对比，Rhombot在单DoF、控制复杂度、联接稳定性方面表现更优；实验验证的定位误差约4.8 mm（x）/14.96 mm（y），Docking误差小于1 mm。

**⚠️ 局限性**

局限在于单轴驱动导致角度误差（前后方向差约131°）、电缆松弛、模块数量有限，且尚无自主规划算法支持大规模自动重构。

---

## 341. A Bisimulation-Invariance-Based Approach to the Separation of Polynomial Complexity Classes

**arXiv ID:** 2601.19641 | [PDF](https://arxiv.org/pdf/2601.19641v1)

**作者:** Florian Bruse `[一作]` (Technical University of Munich), Martin Lange `[通讯]` (University of Kassel)

**通讯引用:** 1483 | [OpenAlex ID](https://openalex.org/A5087768042)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了在双射不变性约束下，P、NP 与 PSPACE 这三类多项式时间复杂度的可分离性问题，并通过将多元 μ-演算与普通模态 μ-演算在功率图（power graph）类中的可定义性关联，提出了相对正规性（relative regularity）的概念来表征双射不变查询的可判定性。

**💡 创新点**

创新点在于：①首次把 Otto 定理（双射不变查询等价于多元 μ-演算可定义）与功率图构造结合，构建了双射不变世界下的相对正规性框架；②通过该框架给出了 NP 与 PSPACE 双射不变查询的示例，并把它们是否属于 P 的判定问题转化为树语言族的相对非正规性判定；③提出的相对非正规性判定方法为 P 与高阶复杂度类的分离提供了新的思路，缓解了传统描述复杂度方法中存在的顺序问题。

**🔧 技术方法**

主要技术包括：Otto 定理、polyadic μ-演算与 modal μ-演算之间的可定义性转化、功率图（power graph）构造、相对正规性与相对非正规性的定义、树语言（tree language）的正规性判定工具。

**📊 数据集**

本文为理论研究，未使用实际数据集；所有示例和论证均基于形式化的图模型与树语言族。

**📈 对比分析**

由于本工作主要是理论性分离证明，没有进行实验比较；但通过构造的功率图与相对正规性框架，作者对 P 与 NP/PSPACE 的可判定边界进行了形式化对比，指出了若能证明某树语言族相对非正规，则可实现 P 与相应高阶复杂度类的分离。

**⚠️ 局限性**

限制在于：相对非正规性判定的组合复杂度极高，现阶段仅能处理有限族；目前尚未给出通用的非正规性证明方法，导致分离结论仍停留在理论假设阶段；此外，功率图构造与 μ-演算的转化步骤在实际实现上可能效率低下。

---

## 342. Towards Governance-Oriented Low-Altitude Intelligence: A Management-Centric Multi-Modal Benchmark With Implicitly Coordinated Vision-Language Reasoning Framework

**arXiv ID:** 2601.19640 | [PDF](https://arxiv.org/pdf/2601.19640v1)

**作者:** Hao Chang `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Jinqiao Wang `[通讯]` (Wuhan AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了GovLA-10K管理导向的低空视觉语言基准和GovLA-Reasoner统一框架，用于低空无人机治理场景下的异常检测与可解释生成。

**💡 创新点**

创新点在于：①以治理需求为中心设计标注策略，聚焦异常/风险目标；②构建半自动两阶段标注流程；③通过轻量化特征适配器实现视觉检测与LLM的隐式特征空间对齐，消除传统提示式链路中的信息损失与误差堆叠；④无需微调任何组件即可提升性能。

**🔧 技术方法**

核心技术包括：基于MM-GroundingDINO的视觉检测与跨模态查询；Qwen3-4B等大型语言模型；特征压缩与融合的Transformer适配器；端到端训练仅更新适配器。

**📊 数据集**

使用GovLA-10K数据集（约10,572张UAV图像，9类治理关键目标，含标注框与管理建议语句）。

**📈 对比分析**

在GovLA-10K上对比多种主流VLM（LLaVA、InternVL、Qwen3-VL等）与传统提示式流程，GovLA-Reasoner在BLEU‑4、METEOR、ROUGE‑L、CIDEr‑D等指标上分别提升约30%～45%，显著优于8B级别模型，且仅使用4B LLM，展示更高参数效率。

**⚠️ 局限性**

局限性主要是：①仍需手工标注框框和监管文本；②适配器设计为固定不变，可能在不同场景下需重新调整；③对极罕见目标的识别仍受检测模型性能限制；④在真实部署时对GPU资源与实时性要求未做深入评估。

---

## 343. RATE: Reviewer Profiling and Annotation-free Training for Expertise Ranking in Peer Review Systems

**arXiv ID:** 2601.19637 | [PDF](https://arxiv.org/pdf/2601.19637v1)

**作者:** Weicong Liu `[一作]` (East China Normal University), Xiang Li `[通讯]` (East China Normal University)

**通讯引用:** 40636 | [OpenAlex ID](https://openalex.org/A5041120433)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LR‑bench高质量、时效性强的评审匹配基准，并基于LLM的RATE框架实现无人工标注的评审者画像与对比学习排名；

**💡 创新点**

①用最新的2024‑2025年AI/NLP论文构建评审基准；②利用LLM提炼评审者论文关键词生成结构化画像，消除传统聚合导致的“profile drift”；③采用基于BM25的弱标签双视角对比学习，无需人工标注；

**🔧 技术方法**

LLM（Qwen3、GLM‑4.6）进行关键词抽取与画像生成；句子‑BERT/SPECTER 等嵌入模型；LoRA 微调；BM25 作为弱监督；对比学习与排序损失；

**📊 数据集**

LR‑bench（1,055条专家评审熟练度标注，覆盖2024‑2025 AI/NLP论文）与CMU gold‑standard数据集；

**📈 对比分析**

在LR‑bench与CMU数据集上与统计、嵌入与LLM基线对比，RATE在精度上达到约77.4%（最高），显著优于SPECTER2 PRX（≈75.2%）及其他方法；

**⚠️ 局限性**

未考虑作者顺序对贡献的影响；关键词抽取可能存在噪声，影响画像质量；对低发表量或冷启动评审者效果有限。

---

## 344. AC^2-VLA: Action-Context-Aware Adaptive Computation in Vision-Language-Action Models for Efficient Robotic Manipulation

**arXiv ID:** 2601.19634 | [PDF](https://arxiv.org/pdf/2601.19634v1)

**作者:** Wenda Yu `[一作]` (Tongji University), Lei Zhu `[通讯]` (Tongji University)

**通讯引用:** 70759 | [OpenAlex ID](https://openalex.org/A5100394072)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于动作上下文的自适应计算框架AC^2-VLA，用以在机器人视觉-语言-动作模型中动态调度时间、空间和深度冗余；

**💡 创新点**

核心创新在于将动作先验作为统一路由器的驱动，实现同时执行缓存重用、Token剪枝与层跳过，彻底把计算量与动作相关联；

**🔧 技术方法**

采用了动作上下文路由器、动作引导自蒸馏、Token级别重要性评分、Transformer层级门控以及基于动作变化的缓存机制；

**📊 数据集**

在Open X‑Embodiment的Bridge子集和SIMPLER（Google Robot、WidowX）两种真实/仿真机器人任务上进行训练与评估；

**📈 对比分析**

与RT‑1/RT‑2、OpenVLA、CogACT以及EfficientVLA、MoLe‑VLA、VLA‑Cache等效率方法对比，AC^2‑VLA在SIMPLER上获得1.79×的速度提升、FLOPs下降至29.4%，并在大多数任务上达到或超过密集基线的成功率；

**⚠️ 局限性**

局限性包括：在极低视觉信息或高度复杂任务下可能性能骤降，需要手动调节稀疏参数；过度依赖动作先验可能在意外动作变化时导致计算决策失效；

---

## 345. Safe Exploration via Policy Priors

**arXiv ID:** 2601.19612 | [PDF](https://arxiv.org/pdf/2601.19612v1)

**作者:** Manuel Wendl `[一作]` (ETH Zurich), Andreas Krause `[通讯]` (ETH Zurich)

**通讯引用:** 30482 | [OpenAlex ID](https://openalex.org/A5003040843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SOOPER算法，利用先验保守策略进行安全回退，结合基于模型的乐观探索实现在线安全学习；

**💡 创新点**

创新点在于将保守策略与乐观模型结合，形成统一的目标函数，既保证学习过程中的安全约束，又通过内在奖励实现探索与安全集扩展；并给出累计遗憾的上界证明；

**🔧 技术方法**

技术包括概率动力学模型、基于模型的演员-评论家框架、模型集成估计不确定性、惰性成本上界、内在探索/扩展奖励、基于安全回退的策略切换；

**📊 数据集**

使用的实验数据集包括RWRL benchmark、SafetyGym、RaceCar、CartpoleSwingup（视觉）、离线SafetyGym的PointGoal1数据以及真实遥控赛车的物理数据；

**📈 对比分析**

与SAILR、CRPO、Primal‑Dual等基准对比，SOOPER在所有任务中都保持安全约束，累计奖励优于基准，收敛速度快；实验还验证了在视觉控制和真实硬件上的可扩展性；

**⚠️ 局限性**

局限性包括：对动态模型的光滑性、噪声方差已知等假设；依赖先验保守策略，若先验质量差影响性能；不适用于无重置的非循环学习场景；对极高维状态空间的可扩展性尚待进一步验证。

---

## 346. The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence

**arXiv ID:** 2601.19597 | [PDF](https://arxiv.org/pdf/2601.19597v1)

**作者:** Yichao Cai `[一作]` (Australian Institute for Machine Learning), Javen Qinfeng Shi `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于测度论的框架，对 InfoNCE 对比学习的几何机制进行理论分析，揭示单模态和多模态学习在大批量与低温极限下的能量景观和收敛性质。

**💡 创新点**

首次将 InfoNCE 的随机目标映射到确定的能量函数，并证明单模态场景下能量严格凸、唯一 Gibbs 均衡；在多模态场景中发现持久的负对称散度项导致“模态间隔”成为结构性几何障碍。

**🔧 技术方法**

测度论分析、核函数平滑、能量函数推导、梯度一致性证明、低温极限下的 Gibbs 分布与熵驱动平衡、KL 散度对称项的几何解释。

**📊 数据集**

文中未给出具体实验数据集，主要为理论推导与数值验证示例；若要实验可参考 CLIP/ALIGN 之类的文本-图像对齐任务。

**📈 对比分析**

由于研究以理论为主，未进行实证比较；作者指出在低温大批量极限下与经验结果一致，可进一步在 CLIP 等基准上验证模态间隔现象。

**⚠️ 局限性**

局限性：只在低温、大批量极限下成立；对有限批量与实际温度的噪声效应未建模；假设嵌入空间为紧致均匀流形，无法直接推广到非均匀或超球面以外的空间；对动态优化过程（随机梯度跳跃、阻塞等）的分析缺失。

---

## 347. Localized Latent Editing for Dose-Response Modeling in Botulinum Toxin Injection Planning

**arXiv ID:** 2601.19593 | [PDF](https://arxiv.org/pdf/2601.19593v1)

**作者:** Estèphe Arnaud `[一作]` (University of Lille), Pierre Guerreschi `[通讯]` (Lille University Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于StyleGAN2的局部潜空间编辑框架，用于预测肉毒毒素注射的剂量-反应关系，并提供可交互的临床规划工具。

**💡 创新点**

创新点在于：①Region‑Specific Latent Axis Discovery（Patch‑and‑Encode）实现肌肉放松轨迹的局部学习；②对比图像预测与直接度量预测两种方法；③构建人机交互工作流，将模拟结果转化为剂量建议。

**🔧 技术方法**

技术方法包括：StyleGAN2 W+潜空间逆变、HyperStyle编码、Gradient Boosting回归、Analysis‑by‑Synthesis优化、LPIPS感知损失、Procrustes 与比例式几何指标评估。

**📊 数据集**

数据集为360张来自46名患者的高分辨率面部照片（Lille大学医院），每位患者采集8种标准表情，并按患者划分为训练/测试集。

**📈 对比分析**

比较方法：Approach A（先生成图像再提取指标）与Approach B（直接预测指标）。在眼部、眉部、嘴部等几何不对称指标上，B方法在距离型指标上R²最高0.67、相关系数0.82，A方法在眉部不对称上稍好；两者均达到中等到强相关（r ≈ 0.4–0.82），但绝对误差相对较大。

**⚠️ 局限性**

limitations：样本量小，单中心数据，未充分考虑个体差异（代谢、注射深度、肌肉结构）导致生物随机性大，预测精度受限；需要多中心扩展和加入更多患者特征进行进一步验证。

---

## 348. Putting Privacy to the Test: Introducing Red Teaming for Research Data Anonymization

**arXiv ID:** 2601.19575 | [PDF](https://arxiv.org/pdf/2601.19575v1)

**作者:** Luisa Jansen `[一作]` (University of Bern), Malte Elson `[通讯]` (University of Bern)

**通讯引用:** 3798 | [OpenAlex ID](https://openalex.org/A5076659092)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了红队/蓝队相互测试科研数据匿名化的方法，并在一项隐私专家混合方法研究中进行案例验证。

**💡 创新点**

将军事/安全领域的红队/蓝队技术引入科研数据匿名化，使匿名化过程可操作、可复现并提供可供研究者使用的材料。

**🔧 技术方法**

采用红队攻击技术（信息收集、逆向工程、跨源匹配）与蓝队匿名化技术（类别编码、去关联、时间戳删除、聚合统计）。

**📊 数据集**

使用12名隐私专家的混合方法研究数据（问卷、绩效记录、个人代码词），数据公开可在 OSF 上获取。

**📈 对比分析**

通过两轮红队攻击评估匿名化效果：第一轮可重新识别4名参与者，第二轮仅能推测2名，表明改进后风险显著降低，同时保持数据可验证性和研究实用性。

**⚠️ 局限性**

红队不一定覆盖所有攻击向量，过程耗时且高度依赖团队创造力；方法的适用性和效果需在更多研究场景中进一步验证。

---

## 349. Modular Foundation Model Inference at the Edge: Network-Aware Microservice Optimization

**arXiv ID:** 2601.19563 | [PDF](https://arxiv.org/pdf/2601.19563v1)

**作者:** Juan Zhu `[一作]` (Hong Kong University of Science and Technology), Khaled Ben Letaief `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了针对基础模型推理的两层微服务部署框架：静态部署核心服务以构建可靠计算骨干，动态调度轻量服务以应对实时任务波动；

**💡 创新点**

创新点在于：①利用核心‑轻量服务功能异质性，设计稀疏约束整数规划与QoS得分实现高效静态部署；②将有效容量理论与Lyapunov优化相结合，构建低复杂度在线控制实现概率延迟保障；

**🔧 技术方法**

采用整数规划、有效容量理论、Lyapunov漂移加惩罚、贪婪在线启发式、网络感知路由与统计QoS评分等技术；

**📊 数据集**

主要使用仿真数据：基于Poisson、Nakagami、Gamma分布的任务到达、信道衰落及边缘设备/服务器资源参数，未使用真实数据集；

**📈 对比分析**

与LBRR、GA、PropAvg等基线方法对比，仿真结果显示本框架在保证84%以上及时完成率的同时，系统成本保持中等且对负载提升表现出更强的鲁棒性；

**⚠️ 局限性**

局限性包括：依赖准确的统计模型与长期工作量估计；静态部署在突发负载变动时可能缺乏灵活性；贪婪在线策略并非全局最优；假设任务到达与处理率为平稳独立，实际环境可能更复杂。

---

## 350. Scale-Consistent State-Space Dynamics via Fractal of Stationary Transformations

**arXiv ID:** 2601.19551 | [PDF](https://arxiv.org/pdf/2601.19551v1)

**作者:** Geunhyeok Yu `[一作]` (Kyung Hee University), Hyoseok Hwang `[通讯]` (Kyung Hee University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5018395387)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 FROST 框架，将分形自相似结构注入状态空间模型，以实现可变复杂度的推理。

**💡 创新点**

创新点在于引入分形诱导偏置，使中间表示在不同迭代尺度下保持一致，从而允许基于排名的自适应早停。

**🔧 技术方法**

使用的技术包括状态空间模型、合同变换、λ 与 Hurst 指数的分形尺度参数、排名式阈值判定以及 KLL 量化等。

**📊 数据集**

实验使用 ImageNet‑100 与 CIFAR‑100 两个数据集。

**📈 对比分析**

与 ResNet、ViT、BasicSSM 等基线对比后发现，FROST 在保持或提升准确率的同时，显著降低平均深度和 GFLOPs，吞吐率提升。

**⚠️ 局限性**

局限性包括当前仅支持全局共享变换，无法针对不同 token 进行细粒度自适应；分形参数需手动设定，表达力可能受限。

---

## 351. Enhancing Inverse Perspective Mapping for Automatic Vectorized Road Map Generation

**arXiv ID:** 2601.19536 | [PDF](https://arxiv.org/pdf/2601.19536v1)

**作者:** Hongji Liu `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19281 | [OpenAlex ID](https://openalex.org/A5100357282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个低成本、统一的基于增强逆透视映射（IPM）的自动向量化道路地图生成框架，利用Catmull‑Rom样条拟合车道线，使用多边形统一描述其他地面标记，并通过实例分割结果对地图点、IPM同伦矩阵和车辆位姿进行联合优化，最终实现高精度（厘米级）道路地图。

**💡 创新点**

创新点在于：① 引入IPM误差估计和点筛选机制，提升车道线控制点估计精度；② 采用投影点到样条残差（PPSR）替代传统点对样条残差（PSR），有效降低车道线预测点误差对优化的影响；③ 同时优化IPM矩阵、地图点的Z轴坐标以及车辆位姿，实现无需逐车校准、可在线自校的自动化映射；④ 使用多边形统一表达所有地面标记，兼容不同形状标记。

**🔧 技术方法**

技术包括：逆透视映射、同伦矩阵与相机外参联合优化、Catmull‑Rom样条建模、基于实例分割的地面标记检测、非线性最小二乘优化（Gauss‑Newton/LM）、误差传播求取IPM不确定度、投影点对样条残差（PPSR）优化、车辆位姿迭代更新。

**📊 数据集**

使用了两类实际数据集：一是港口自动化场景（多块菱形标记和多车道线），二是公开道路场景（人行横道、限速标识、箭头等），并在这些真实道路上采集了前视单目相机图像、RTK‑GNSS 车姿与全站仪标注的地面标记三维坐标。

**📈 对比分析**

通过与传统IPM、PGO‑IPM、PersFormer、MonoLaneMapping等基线比较，实验表明：地图点误差从约0.2‑0.6 m下降到0.08‑0.16 m，车道线误差从约0.3‑0.4 m下降到0.07‑0.13 m，IPM同伦矩阵精度与人工标定相当；同时地图体积从MB级压缩至几KB，仅存储少量控制点，显著提高存储效率。

**⚠️ 局限性**

局限性包括：① 依赖清晰、可见的地面标记，雨雪、模糊或磨损标记时效果下降；② IPM 本质上仍受平面假设限制，无法处理大坡度或凹凸不平路面；③ 仅使用单目相机，无法估计标记高度变化导致的误差；④ 需要先行进行实例分割，对分割误差敏感。

---

## 352. LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG

**arXiv ID:** 2601.19535 | [PDF](https://arxiv.org/pdf/2601.19535v1)

**作者:** Manish Chandra `[一作]` (University of Glasgow), Iadh Ounis `[通讯]` (University of Glasgow)

**通讯引用:** 10088 | [OpenAlex ID](https://openalex.org/A5079046603)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级的、基于效用驱动的 RAG 重新排序框架 LURE‑RAG；

**💡 创新点**

创新点在于将 LLM 产生的效用信号作为监督，使用 LambdaMART 的 listwise 排序损失来直接优化检索结果的排序，并兼容任何黑盒检索器；

**🔧 技术方法**

技术包括：LambdaMART 轻量级树模型、LLM 效用监督、列表排序损失、特征构造（BM25、IDF 统计、词重叠、LDA 主题相似度）以及密集版 UR‑RAG 的 SBERT 细调；

**📊 数据集**

使用的数据集为开放域问答基准 Natural Questions‑Open (NQ‑Open) 和 TriviaQA (TQA)；

**📈 对比分析**

通过与 0‑shot、k‑shot、RePlug 及 UR‑RAG 等基线对比，LURE‑RAG 在两大数据集上实现了与最强密集检索 reranker 97‑98% 的效果，且在密集版 UR‑RAG 上可提升至最高 3%；

**⚠️ 局限性**

局限性包括：对 LLM 产生的效用信号高度依赖，主题特征贡献有限，实验范围仅覆盖 QA 任务，且未探讨跨域或更复杂检索场景下的泛化能力；

---

## 353. A Non-Invasive 3D Gait Analysis Framework for Quantifying Psychomotor Retardation in Major Depressive Disorder

**arXiv ID:** 2601.19526 | [PDF](https://arxiv.org/pdf/2601.19526v1)

**作者:** Fouad Boutaleb `[一作]` (University of Lille), Fabien D'Hondt `[通讯]` (University of Lille)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于单目RGB视频的非侵入性计算框架，利用改进的TUG协议提取297个可解释的步态生物标记，并用机器学习模型预测重度抑郁症患者的运动迟滞与抑郁严重程度。

**💡 创新点**

创新点包括：①将Gravity-View坐标与闭环轨迹校正相结合，显著减少单目深度误差；②引入稳定性选择的特征筛选，解决小样本过拟合问题；③以可解释的三维步态参数为特征，构建线性可解释模型。

**🔧 技术方法**

使用技术包括：GVHMR单目三维网格重建、闭环轨迹优化与PCA对齐、步态事件检测与臂摆动力学提取、关节角度计算、稳定性特征选择以及逻辑回归/线性SVR等机器学习方法。

**📊 数据集**

数据集为CALYPSO抑郁症临床数据集，包含42名重度抑郁症住院患者的单目视频和HDRS评分。

**📈 对比分析**

与基于Gram矩阵的几何特征基线相比，本方法在运动迟滞分类上达83.3%准确率（基线67%），在抑郁严重程度回归上R²为0.64（基线0.13），并通过置换检验验证统计显著性；消除轨迹校正等组件的消融实验进一步证明其关键作用。

**⚠️ 局限性**

局限性包括样本量小、受年龄、BMI、药物影响等混杂因素、仅使用HDRS第8项评估运动迟滞、单目视频对遮挡和姿态误差敏感、模型未实现实时推理、可能存在观察者效应，需进一步扩大样本并融合多模态数据验证。

---

## 354. GenCP: Towards Generative Modeling Paradigm of Coupled Physics

**arXiv ID:** 2601.19541 | [PDF](https://arxiv.org/pdf/2601.19541v1)

**作者:** Tianrun Gao `[一作]` (Westlake University), Tailin Wu `[通讯]` (Westlake University)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5109695201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出一种基于流匹配与算子分裂的生成式耦合物理仿真框架（GenCP），能够在仅使用解耦训练数据的情况下实现耦合物理的高保真推断。

**💡 创新点**

核心创新包括：① 将耦合问题视为功能空间中概率密度的演化，利用条件概率学习实现联合采样；② 在流匹配过程中嵌入算子分裂，保证训练时使用解耦数据、推理时实现耦合；③ 推导出算子分裂对概率密度演化的误差可控性，给出理论误差上界。

**🔧 技术方法**

技术方法包括：流匹配（flow matching）技术、算子分裂（Lie–Trotter splitting）、神经算子（FNO*、CNO）以及线性插值与时间参数化的训练目标。

**📊 数据集**

实验数据集涵盖：① 2D 合成分布（简单与复杂）；② 两个流固耦合基准（Turek–Hron 与双圆柱）；③ 三维核-热耦合（核反应、固体热传导、流体热传输）等。

**📈 对比分析**

与基线（Surrogate–Picard、M2PDE）及联合训练方法比较，GenCP 在三大场景下平均误差分别降低 12.5%–42.9%，在最坏情况下误差下降 65%；推理步骤数仅 10 步，推理速度比传统迭代方法快 3–4 倍，且整体精度显著优于现有方法。

**⚠️ 局限性**

局限性：① 算子分裂假设在极度非线性或强耦合场景下可能需要更高阶分裂或更小步长；② 目前实验主要基于模拟数据，对真实工业数据的鲁棒性尚未验证；③ 训练需要充分的解耦样本，若解耦数据稀缺，条件概率学习可能受限；④ 扩展至更多物理场或更复杂几何时，模型结构与训练成本仍待进一步优化。

---

## 355. Enhancing Worker Safety in Harbors Using Quadruped Robots

**arXiv ID:** 2601.19643 | [PDF](https://arxiv.org/pdf/2601.19643v1)

**作者:** Zoe Betta `[一作]` (Rice Lab, University of Genova), Antonio Sgorbissa `[通讯]` (Rice Lab, University of Genova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文通过对热那亚港口工人进行半结构化访谈，识别出关键安全隐患区域，并提出利用四足机器人（如Spot）进行巡检的方案；

**💡 创新点**

创新点在于结合用户需求与现场实际，系统性地将港口安全问题映射到不同机器人类型与自主水平，并突出四足机器人在复杂地形与人机共存环境中的优势；

**🔧 技术方法**

采用四足机器人与多传感器（气体、辐射、热像、LiDAR、IMU等）组合，结合遥控、混合自主与完全自主操作模式；

**📊 数据集**

论文未使用公开数据集，而是基于港口现场访谈收集的定性信息构建问题与解决方案表；

**📈 对比分析**

缺乏实验对比与性能评估，本文仅在理论层面讨论不同任务对应的机器人选择与自主级别，没有提供量化指标；

**⚠️ 局限性**

主要限制是未在实际港口环境中部署与验证所提方案，缺乏实验数据与性能评估，且对机器人与传感器集成的技术细节和成本效益未作深入分析。

---

## 356. Who Said CVE? How Vulnerability Identifiers Are Mentioned by Humans, Bots, and Agents in Pull Requests

**arXiv ID:** 2601.19636 | [PDF](https://arxiv.org/pdf/2601.19636v1)

**作者:** Pien Rooijendijk `[一作]` (Radboud University), Mairieli Wessel `[通讯]` (Radboud University)

**通讯引用:** 600 | [OpenAlex ID](https://openalex.org/A5032291051)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究GitHub拉取请求（Pull Request）中人类、机器人和自动编码代理对标准漏洞标识符（如CVE、CWE、GHSA等）的提及方式、位置和频率，并通过定性分析探讨这些提及背后的动机与作用。

**💡 创新点**

首次将自动编码代理的漏洞ID提及与人类与机器人进行系统对比，揭示不同贡献者在安全维护中的角色差异；同时提出基于文本提及的多维度分析框架，弥补了以往仅关注代码级别或单一实体的研究空白。

**🔧 技术方法**

使用正则表达式匹配多种漏洞ID模式、GitHub REST API爬取PR标题、描述、评论、提交信息等文本；对账号进行分类（机器人、人类、代理）并统计提及分布；对随机抽样的70个PR进行手工编码，构建使用场景标签。

**📊 数据集**

AIDev-pop子集（约33,596 PR，2,807仓库）以及同仓库在相同期段（2025-01-01至2025-08-01）通过API增补的PR，合计约7,621条漏洞ID提及。

**📈 对比分析**

通过描述性统计和分布图（如提及数、位置分布）比较不同账号类型的提及特征，结果显示机器人占据69.1%的提及量，主要集中在PR描述；人类和代理提及更少但更分散，出现在提交信息、标题及讨论中。

**⚠️ 局限性**

仅检测显式文本提及，可能忽略未命名的漏洞修复；机器人/代理识别依赖用户名和标签，存在误分类风险；研究范围限于公开仓库、特定时间段和特定ID模式，难以推广到私有仓库、其他平台或未覆盖的漏洞数据库。

---

## 357. The Competence Crisis: A Design Fiction on AI-Assisted Research in Software Engineering

**arXiv ID:** 2601.19628 | [PDF](https://arxiv.org/pdf/2601.19628v1)

**作者:** Mairieli Wessel `[一作]` (Radboud University), Sangeeth Kochanthara `[通讯]` (ASTRON)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5037062189)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过设计虚构与价值敏感设计方法，探讨生成式 AI 在软件工程研究中导致的技能退化、责任模糊与信任危机，并提出行动计划

**💡 创新点**

首次将设计虚构与价值敏感设计结合，以情境叙事揭示 AI 工具对研究流程、知识获取与责任分配的深远影响，强调技术垂直深度与伦理责任

**🔧 技术方法**

设计虚构方法、价值敏感设计框架、案例分析与访谈式叙事

**📊 数据集**

基于 ICSE 2026 预调查问卷的开放式回复主题（社区健康、出版压力、AI 依赖等）

**📈 对比分析**

无实验对比或性能指标，主要以理论反思和案例描述为核心

**⚠️ 局限性**

缺乏实证验证与定量数据，视角较主观，未给出技术实现细节，难以量化 AI 对研究质量的具体影响

---

## 358. Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search

**arXiv ID:** 2601.19622 | [PDF](https://arxiv.org/pdf/2601.19622v1)

**作者:** Thomas Bömer `[一作]` (Karlsruhe Institute of Technology), Anne Meyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 9050 | [OpenAlex ID](https://openalex.org/A5091690345)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出基于LLM的自适应提示扩展 A-CEoH，用于自动生成 A* 搜索的引导启发式函数

**💡 创新点**

创新点在于将算法代码嵌入提示中，实现算法上下文感知的提示，并结合问题上下文共同提升 LLM 生成的启发式质量

**🔧 技术方法**

使用 LLM（如 gpt-3.5-turbo、Claude 2）与演化搜索框架 EoH 的组合，并加入算法上下文与问题上下文提示

**📊 数据集**

在单层仓库的单位负载预排序问题（UPMP）和大型 20×20 滑动拼图（SPP）实例上进行实验

**📈 对比分析**

与传统 EoH、P-CEoH 以及人工设计的启发式对比，A-CEoH 在 UPMP 上平均相对偏差降低到 0.08，甚至在小模型上超过大模型；在 SPP 上虽提升但未超越最优人工启发式，整体性能显著优于基线

**⚠️ 局限性**

局限性包括：对 SPP 的提升有限，算法上下文提示对不同模型/问题的效果不一致，且提示长度大幅增加导致 token 消耗高

---

## 359. Explicit Multi-head Attention for Inter-head Interaction in Large Language Models

**arXiv ID:** 2601.19611 | [PDF](https://arxiv.org/pdf/2601.19611v1)

**作者:** Runyu Peng `[一作]` (Shanghai AI Laboratory), Xipeng Qiu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多头显式注意力（MEA）框架，并在其核心引入了头级线性组合（HLC）模块；通过统一视角，将 Differential Transformer 与 Talking-Heads Attention 等现有变体归入该框架，并在从零预训练、KV 缓存压缩等任务上进行评估。

**💡 创新点**

创新点在于：①直接对 key 与 value 进行可学习的头级线性组合以增强跨头通信；②结合 GroupNorm 稳定优化并保持表示多样性；③利用 MEA 结构实现 KV 缓存按比例压缩至 50% 的显存；④通过缩放律快速选择合适学习率，显著提升收敛速度与最终性能。

**🔧 技术方法**

采用的技术包括：头级线性组合（HLC）模块、GroupNorm（RMSNorm）正则化、RoPE 位置编码、SVD/低秩分解实现 KV 压缩、cosine 学习率调度、AdamW 优化器以及基于缩放律的超参搜索。

**📊 数据集**

使用的数据集：从零预训练采用 LLaMA3.2‑1B 架构的文本语料（500B 令牌）；持续预训练基于 Qwen3‑30B‑A3B；下游评估涵盖 MMLU‑Pro、GPQA Diamond、SuperGPQA、ChemBench、ClimaQA、MedXpertQA、AIME 2025、OlympiadBench、LiveMathBench‑Hard、OlymMATH 等多领域推理 benchmark。

**📈 对比分析**

与原 Transformer、加 GroupNorm、Differential Transformer（DFA）以及无 GroupNorm 的 MEA（相当于改版 THA）进行对比；在相同训练设置下，MEA 在验证损失、知识推理（MMLU、GPQA）和科学推理（ChemBench、ClimaQA、MedXpertQA）上均优于基线；在 KV 压缩实验中，压缩至 2 组 heads 时，完整压缩版本在知识/科学任务上性能基本不降，数学任务略有下降，但可通过恢复阶段恢复到接近全参数水平。

**⚠️ 局限性**

局限性包括：①在极低显存压缩（全压缩）时对数学推理任务仍易出现性能下降；②需要额外的恢复训练阶段才能恢复压缩模型的性能；③实验多聚焦于从零预训练和持续预训练，缺少对大规模工业模型在多任务学习中的进一步验证；④当前仅评估了单一硬件平台，压缩效果与算子兼容性需进一步研究。

---

## 360. LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation

**arXiv ID:** 2601.19585 | [PDF](https://arxiv.org/pdf/2601.19585v1)

**作者:** Chongjun Xia `[一作]` (University of Technology Sydney), Xianzhi Wang `[通讯]` (University of Technology Sydney)

**通讯引用:** 5896 | [OpenAlex ID](https://openalex.org/A5076107706)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于LLM增强的强化学习框架LERL，用以提升交互式推荐系统的长期用户满意度。

**💡 创新点**

创新点在于将大型语言模型用于高层语义规划与低层RL细粒度决策的层次化组合，并通过LLM生成的反思提升长期规划质量。

**🔧 技术方法**

采用Llama-3-8B进行高层规划，PPO算法进行低层策略学习，结合Transformer编码用户历史、Gaussian策略采样与类别软掩码过滤。

**📊 数据集**

使用了KuaiRand和KuaiRec两套大规模用户-物品交互数据集。

**📈 对比分析**

在KuaiSim模拟器中与PG、A2C、DDPG、TD3、PPO、HAC、SAC4IR、DNaIR等RL基线对比，LERL在交互长度和累计奖励上均显著领先。

**⚠️ 局限性**

局限性包括对LLM提示与反思质量的依赖、模拟环境的真实性限制以及缺乏公平性、多目标优化等进一步考量。

---

## 361. ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving

**arXiv ID:** 2601.19582 | [PDF](https://arxiv.org/pdf/2601.19582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 362. Benchmarks Saturate When The Model Gets Smarter Than The Judge

**arXiv ID:** 2601.19532 | [PDF](https://arxiv.org/pdf/2601.19532v1)

**作者:** Marthe Ballon `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Harvard University)

**通讯引用:** 1187 | [OpenAlex ID](https://openalex.org/A5049169851)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对原始 Omni-MATH 数据集进行手工清理与审核，生成 Omni-MATH-2，包括精确答案子集（4181 题）和标签化的非标准子集（247 题），并对数据集错误和评判器噪声进行系统分析。

**💡 创新点**

创新点在于：① 从内容层面而非仅格式层面对数学题进行全面审核，识别并纠正缺失图像、证明/估计题目等导致的可解性与可验证性问题；② 明确展示评判器（Omni-Judge）在高准确率下的偏差与误判，提出 benchmark 评估应视为 (数据集、模型、评判器) 三元组；③ 通过人工标注与 GPT-5 mini 对评判器误差进行量化，揭示评判器质量决定模型排名的实证。

**🔧 技术方法**

技术手段包括：使用 Python 对 LaTeX 语句进行编译修复；由 PhD 级数学家手工审阅 PDF 并补全缺失信息；为每题标注图像、证明、估计等标签；利用两种评判器（Omni-Judge 与 GPT‑5 mini）自动评判答案，并对评判差异进行人工专家注释，分析评判器误差来源。

**📊 数据集**

使用的数据集为 Omni-MATH-2，继承原 Omni-MATH 的 4,428 题目，其中 4,181 题构成精确答案子集，247 题为非标准标记子集；原始 Omni-MATH 作为对照。

**📈 对比分析**

通过在 Omni-MATH-2-Filtered 上评估五大模型（Claude Sonnet 4.5、DeepSeek v3.2、Gemini 3 Pro、GPT-5、Kimi K2 Thinking），并使用两种评判器分别得分，发现评判器差异显著：评判器不同导致模型排名发生变化，尤其在最高难度 Tier‑4 题目上评判器误差最大；GPT‑5 mini 在绝大多数情况下比 Omni-Judge 评判更准确，表明评判器质量是决定模型性能的重要因素。

**⚠️ 局限性**

局限性包括：① 未对参考答案进行修改，导致部分不完整或错误答案仍影响评判；② 仅对两种评判器在特定提示下进行分析，未覆盖评判器行为的全空间；③ 未引入部分分数或不确定性评估，评判仍是二元；④ 评估聚焦于 Omni-MATH，结果可能不适用于其他领域。

---

## 363. Intersectional Fairness via Mixed-Integer Optimization

**arXiv ID:** 2601.19595 | [PDF](https://arxiv.org/pdf/2601.19595v1)

**作者:** Jiří Němeček `[一作]` (Czech Technical University in Prague), Jakub Mareček `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 1540 | [OpenAlex ID](https://openalex.org/A5003656133)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于混合整数优化的框架，既能检测最不公平的交叉子群体，又能训练满足交叉公平且可解释的分类器。

**💡 创新点**

证明MSD与SPSF在寻找最不公平子群体时等价，并通过惰性约束在指数规模子群体上实现公平约束，同时兼顾模型可解释性。

**🔧 技术方法**

采用混合整数优化（MIO）、惰性约束、线性/逻辑规则模型以及MSD、SPSF、FPSF等交叉公平度量，并引入稀疏性惩罚提升解释性。

**📊 数据集**

在Folktables的五个美国人口普查数据集上进行实验：ACS Income、Public Coverage、Mobility、Employment、Travel Time。

**📈 对比分析**

与无公平约束模型和GerryFair比较；实验显示在保持公平阈值的前提下，准确率仅略降，公平度量显著提升；且只需数十条惰性约束即可覆盖指数级子群体。

**⚠️ 局限性**

受限于MIO求解规模，随着样本量或子群体维度增大求解时间上升；公平度量仍局限于二分类任务；对小样本子群体的估计存在不稳定性。

---

## 364. Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning

**arXiv ID:** 2601.19624 | [PDF](https://arxiv.org/pdf/2601.19624v1)

**作者:** Tongxi Wang `[一作]` (Southeast University), Shan Liu `[通讯]` (Southeast University)

**通讯引用:** 12518 | [OpenAlex ID](https://openalex.org/A5100404959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对非平稳强化学习环境，提出一种自适应熵调度（AES）框架，动态在线调整最大熵RL算法的熵系数/温度，以应对环境漂移导致的探索与利用权衡失衡。

**💡 创新点**

创新点在于：①将熵调度简化为一维的“跟踪‑稳定”对策，给出与漂移幅度直接相关的熵尺度；②提供可观测的漂移代理（如 TD 错误的高分位数）实现无额外检测或重启；③在多种主流最大熵算法上实现无结构改动的通用插拔。

**🔧 技术方法**

技术包括：动态镜像下降（Dynamic Mirror Descent）理论、可变熵正则化的OCO分析、基于漂移代理的在线调度公式 λ_t ∝ √(∑_{s≤t} α_s / t)，以及在 SAC、PPO、SQL、MEow 等算法中的熵系数替换。

**📊 数据集**

实验数据集涵盖三类任务：Toy 2D 多目标、MuJoCo 连续控制（Hopper、HalfCheetah、Walker2d、Ant、Humanoid）以及 Isaac Gym 大规模仿真（Ant、Humanoid、Ingenuity、ANYmal、AllegroHand、FrankaCabinet），每类任务下均设定五种漂移模式（稳态、突变、线性、周期性、混合）。

**📈 对比分析**

与基线（固定熵或自动温度控制）和常规变体对比，AES 在所有算法、任务和漂移模式下均提升归一化 AUC、降低性能损失面积比，并将突变后恢复时间显著压缩（例如 SAC 在 12 个任务中平均恢复时间从 13.96% 降至 7.74%）。

**⚠️ 局限性**

局限性包括：①漂移代理依赖 TD 错误等指标，对不同算法或函数逼近误差敏感；②理论分析基于离散 MDP，延伸到连续或高维任务仍需进一步验证；③在极端漂移或不可观测动态下，熵调度可能无法完全跟踪最优策略。

---

## 365. QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture

**arXiv ID:** 2601.19580 | [PDF](https://arxiv.org/pdf/2601.19580v1)

**作者:** Cuong Le `[一作]` (Linköping University), Bastian Wandt `[通讯]` (Independent researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了QuaMo，一种在线基于四元数的3D人体运动捕捉方法；

**💡 创新点**

创新点在于使用四元数微分方程（QDE）在单位球约束下进行姿态积分，并加入自适应加速度增强的元PD控制器，解决传统欧拉角导致的离散性与不连续性问题；

**🔧 技术方法**

采用四元数表示、Hamilton乘积、元PD控制器、加速度增强、SMPL模型与ControlNet进行训练与推理；

**📊 数据集**

在Human3.6M、Fit3D、SportsPose和AIST四个数据集上进行评测；

**📈 对比分析**

与多类基线（关键点提升、视觉时序、单帧预测、窗口优化、在线动力学方法）相比，QuaMo在MPJPE、P‑MPJPE、G‑MPJPE、G‑Accel、Foot‑Skating等指标上均实现了最优或接近最优性能；

**⚠️ 局限性**

局限性包括缺乏对环境交互与接触的建模，未来工作计划将人机交互与场景约束融入模型中。

---

## 366. The role of self-supervised pretraining in differentially private medical image analysis

**arXiv ID:** 2601.19618 | [PDF](https://arxiv.org/pdf/2601.19618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 367. How to Serve Your Sandwich? MEV Attacks in Private L2 Mempools

**arXiv ID:** 2601.19570 | [PDF](https://arxiv.org/pdf/2601.19570v1)

**作者:** Krzysztof Gogol `[一作]`, Claudio Tessone `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在私有 mempool 的 Layer‑2 (L2) 区块链上，攻击者通过前后运行交易（sandwich attack）尝试从受害者的滑点中获利的可能性，并提出了相应的理论模型和实证分析。

**💡 创新点**

创新点在于：① 将常见的 CPMM 砂箱盈利模型推广到 CLMM（集中流动性 AMM）并说明其在 tick 边界的收益跳变；② 建立私有 mempool 下的概率执行模型，量化同块共包含的成功率；③ 通过严格的攻击者归属启发式（区分 EOA、路由器和自定义合约）与经济一致性检验，首次在 L2 上系统性排查砂箱攻击，证明其极其罕见且大多为误报。

**🔧 技术方法**

使用的技术包括：AMM 数学建模（小交易展开、二次近似）、概率论（Poisson 进程、批处理窗口的包含概率）、以太坊日志查询工具（Dune Analytics）以及 Python/SQL 数据处理脚本。

**📊 数据集**

数据集涵盖 2025 年 1 月至 9 月期间的 Arbitrum、Base、Optimism、Unichain 与 ZKsync 上的所有 swap 事件日志、交易记录与链上状态，约 70 万条 swap 事件。

**📈 对比分析**

与传统 L1 砂箱的对比方法主要是：① 计算攻击者与受害者的交易规模比例、背后交易与前后交易的匹配度、收益率分布；② 通过模拟与实测的共包含概率对比，验证 L2 的低成功率。结果显示：在 L2 上，平均利润为负，效率低于 0.01%，与 L1 的高收益（通常为正收益、效率 >1%）形成鲜明对比。

**⚠️ 局限性**

局限性包括：① 仅考虑同块内的攻击，未探讨跨块或异步执行的可能；② 归属方法仍可能把聚合器里的多用户交易误归为同一攻击者；③ 只覆盖主流 L2，可能遗漏规模更小或功能不同的 rollup；④ 估算中未考虑未来潜在的去中心化 builder 市场或 pre‑confirmation 机制的影响。

---

## 368. From Scattered to Structured: A Vision for Automating Architectural Knowledge Management

**arXiv ID:** 2601.19548 | [PDF](https://arxiv.org/pdf/2601.19548v1)

**作者:** Jan Keim `[一作]` (Karlsruhe Institute of Technology), Angelika Kaplan `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 46 | [OpenAlex ID](https://openalex.org/A5049586777)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个自动化管线，能够从需求、设计图、代码和文档等多种软件工件中提取架构知识，并将其统一整合到结构化知识库中。

**💡 创新点**

创新点在于将多模态知识提取、持续一致性检查、基于代理的实时监控和检索增强生成的问答系统结合成一个闭环的架构知识管理系统。

**🔧 技术方法**

利用自然语言处理、LLM、静态分析、图形解析、可追踪链接、知识图谱、RAG技术以及智能代理进行持续监控与决策。

**📊 数据集**

本文未给出具体实验数据或数据集，主要是概念设计与方法论框架。

**📈 对比分析**

尚未进行实验对比与性能评估，文中仅在理论层面提出实现思路和未来工作计划。

**⚠️ 局限性**

主要限制包括工件多样性导致的抽象困难、LLM 的幻觉与上下文窗口限制、持续一致性检测的误报/漏报、人工干预与自动化的平衡、以及知识库的可扩展性与安全性问题。

---

## 369. Mocap Anywhere: Towards Pairwise-Distance based Motion Capture in the Wild (for the Wild)

**arXiv ID:** 2601.19519 | [PDF](https://arxiv.org/pdf/2601.19519v1)

**作者:** Ofir Abramovich `[一作]` (Reichman University), Andreas Aristidou `[通讯]` (University of Cyprus)

**通讯引用:** 2201 | [OpenAlex ID](https://openalex.org/A5020863818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于稀疏体穿戴式UWB传感器的距离测量，利用Transformer实现实时无外部摄像机的全身3D动作捕捉；

**💡 创新点**

核心创新在于将配对距离测量映射到三维关节位置的“Refinement‑Generative”Transformer，包含门控交叉注意力、时空关节自注意力和距离解码器，并通过姿势与距离一致性损失实现去噪与姿态恢复；

**🔧 技术方法**

采用UWB时间飞行（ToF）距离测量、Transformer解码器、门控交叉注意力、时空关节自注意力、刚性损失、重力约束等深度学习与几何融合技术；

**📊 数据集**

主要使用AMASS人类动作数据集进行训练，评估时结合真实UWB采集的人类（10分钟不同动作）与动物（骆驼、驴）数据；

**📈 对比分析**

与MDS+Procrustes、Geo、Feet、UIP、UMotion等基线以及Optical mocap对比，实验显示在人类与动物场景下，关节误差<60mm、根位移误差下降~66%、时序抖动显著降低，能够以约50FPS实时运行；

**⚠️ 局限性**

受限于UWB设备当前噪声水平与NLoS遮挡、对小体积、复杂姿态的鲁棒性不足，且需进一步验证多主体与大规模传感器拓扑的适用性。

---

## 370. Up to 36x Speedup: Mask-based Parallel Inference Paradigm for Key Information Extraction in MLLMs

**arXiv ID:** 2601.19613 | [PDF](https://arxiv.org/pdf/2601.19613v1)

**作者:** Xinzhong Wang `[一作]` (Shanghai Jiao Tong University), Huijia Zhu `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PIP 并行推理范式，用 [mask] 令 KIE 任务能够在单次前向传播中同时生成所有目标字段，实现大幅度加速。

**💡 创新点**

核心创新是：①将 KIE 重构为并行掩码填充任务；②针对该任务设计了双向注意力的掩码预训练策略；③构建大规模 K/V 监督微调数据集以提升精度并降低幻觉。

**🔧 技术方法**

使用多模态大型语言模型（InternVL2‑8B、Qwen2‑VL‑7B 等）作为基础，结合掩码预训练（mask‑and‑predict）与 KV 微调，最终实现并行解码；对比实验中评估 ANLS、F1 与推理时延。

**📊 数据集**

数据集包括公开 KIE 评测基准：FUNSD、SROIE、CORD、POIE、WildReceipt；以及自构造的 13M 图像‑说明预训练集和 48 类文档的 KV 监督集。

**📈 对比分析**

相较于原始自回归模型，PIP‑Qwen2‑VL‑7B 在 SROIE、CORD 上刷新 SOTA（分别 97.0 / 97.3 的 ANLS），同时在所有数据集上实现 5–36× 的推理加速，推理时间从 0.3–1.1 s 降至 0.03–0.06 s，几乎无精度损失。

**⚠️ 局限性**

局限性：并行掩码需要增加输入长度，导致显存占用提升最多 30%；在极其细粒度或跨字段依赖的场景下，掩码填充仍可能出现注意力干扰；当前主要验证在结构化表单级任务，未充分评估对自由文本 OCR 质量波动的鲁棒性。

---

## 371. ComAgent: Multi-LLM based Agentic AI Empowered Intelligent Wireless Networks

**arXiv ID:** 2601.19607 | [PDF](https://arxiv.org/pdf/2601.19607v1)

**作者:** Haoyun Li `[一作]`, Yong Liang Guan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

设计并实现了 ComAgent，一套多 LLM 代理式 AI 框架，用于将用户的自然语言意图自动转换为可执行的无线网络优化方案，并在闭环中进行迭代校验与自我纠错；通过案例验证其在 MIMO SWIPT beamforming 以及 25 个多样化无线优化任务中的表现。

**💡 创新点**

创新点包括：
1) 引入感知–规划–行动–反思闭环，使 LLM 具备代理式思考与工具调用能力；
2) 多专用代理（文献、规划、编码、评分）协同工作，显著降低幻觉与逻辑错误；
3) 通过执行反馈（编译、仿真、约束检查）实现从意图到可重复仿真的端到端闭环；
4) 在基准测试中显示出比单 LLM 及单 LLM+计划的显著提升，证明代理式自适应的必要性。

**🔧 技术方法**

采用的技术包括：
- 多 LLM 代理架构（使用 Claude‑4.5‑Sonnet）
- 代理内部的链式思考、ReAct、计划‑求解（PS）提示策略
- 检索增强生成（RAG）与文献检索工具
- 代码生成与执行（Python、仿真脚本、数值优化库）
- 评分代理使用 LLM 评估与物理约束验证
- 传统优化方法（SDR、SCA、WMMSE、ZF）作为算法参考与基准。

**📊 数据集**

使用的数据集：
- 生成的合成 MIMO SWIPT 场景数据（随机用户分布、Rician 随机衰落、能量收集模型等）
- 25 个人工专家设计的多样化无线优化任务（包括 NOMA、RIS、SAGIN 等），但具体公开数据集未列出，属于内部测试集。

**📈 对比分析**

比较方法与性能：
- 对比三种方案：单 LLM、单 LLM+PS、ComAgent
- 评价指标：问题建模率、代码生成率、代码执行率、解决率、首次成功率、平均尝试次数
- 成果：ComAgent 在所有指标上均优于基线；问题建模率 100% vs 0%/56%；执行率 100% vs 24%/88%；解决率 72% vs 24%/56%；首次成功率 32% vs 4%/20%；平均尝试次数 2.12 vs 2.88/2.44。

**⚠️ 局限性**

限制与挑战：
- 当前仅为离线任务，缺乏对实时网络动态的持续控制；
- 代理间频繁交互导致推理延迟，不适用于毫秒级信道控制；
- 集中协调成本高，扩展到大规模网络时通信开销与单点失效问题突出；
- 缺乏长期记忆与生命周期学习，重复优化任务仍需重新检索与规划；
- 对于高度非线性、多目标约束的极端场景，闭环自我纠错仍可能陷入局部最优或收敛慢。

---

## 372. GMS-CAVP: Improving Audio-Video Correspondence with Multi-Scale Contrastive and Generative Pretraining

**arXiv ID:** 2601.19606 | [PDF](https://arxiv.org/pdf/2601.19606v1)

**作者:** Shentong Mo `[一作]` (Carnegie Mellon University), Jun Zhu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一的多尺度对比学习与扩散生成框架，用于视频-音频跨模态预训练，提升视频到音频的生成与检索性能。

**💡 创新点**

创新点在于：①通过多尺度空间-时间对齐（MSA）捕捉细粒度与粗粒度的跨模态关联；②引入多尺度空间-时间扩散（MSD）生成目标，实现视频到音频的模态转换；③将对比学习与生成目标融合，形成判别-生成双重预训练。

**🔧 技术方法**

使用的技术包括多尺度特征提取、InfoNCE 对比损失、注意力加权的自适应时序对齐、基于扩散模型的音频生成与多尺度降噪训练。

**📊 数据集**

在VGGSound、AudioSet和新建的Panda70M三大大规模视频-音频数据集上进行实验。

**📈 对比分析**

与CAVP、Diff-Foley等先前方法对比，本文模型在KLD、FAD、Align Acc、R@1/5/10等评测指标上均取得显著提升，展示了更高的音频质量与更准确的时序对齐。

**⚠️ 局限性**

局限性在于：①框架复杂度较高，训练与推理成本较大；②主要评估在公开数据集，缺乏在多样化真实场景下的泛化验证；③对极端快速运动或音频极端变化的跨模态对齐仍有改进空间。

---

## 373. From Atoms to Chains: Divergence-Guided Reasoning Curriculum for Unlabeled LLM Domain Adaptation

**arXiv ID:** 2601.19588 | [PDF](https://arxiv.org/pdf/2601.19588v1)

**作者:** Yongqi Wang `[一作]` (Beijing Institute of Technology), Xinxiao Wu `[通讯]` (ByteDance China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在无标注数据的专业领域中，提出了 Divergence‑Guided Reasoning Curriculum（DGRC）框架，实现了对大型语言模型的自适应训练。

**💡 创新点**

创新点在于利用教师‑学生推理分歧自动诊断，生成原子知识与验证后链式推理两阶段课程，解决教师不完美导致的误导问题。

**🔧 技术方法**

技术方法包括认知不对称推理、分歧检测、自动诊断生成原子问答、链式推理校验，以及监督微调（SFT）和可选的 RL（GRPO）训练。

**📊 数据集**

使用的数据集为医疗领域的 MedMCQA、MedQA‑USMLE、MMLU‑M，以及法律领域的 CaseHOLD、MMLU‑L，所有训练均在对应的无标注问题集上完成。

**📈 对比分析**

在与无标注蒸馏、标注蒸馏和 RLAIF（GRPO）等基线对比中，DGRC 在 1.5B 学生模型上医学领域提升了 7.76% 相关性，整体在多模型、多领域均优于基线。

**⚠️ 局限性**

局限性包括对教师模型的推理与指令遵循能力要求较高，且教师自身的错误会影响生成的原子知识；在小模型自教时难以满足格式规范，导致学习效果受限。

---

## 374. Toward Architecture-Aware Evaluation Metrics for LLM Agents

**arXiv ID:** 2601.19583 | [PDF](https://arxiv.org/pdf/2601.19583v1)

**作者:** Débora Souza `[一作]` (Federal University of Campina Grande), Patrícia Machado `[通讯]` (Federal University of Campina Grande)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于LLM代理架构的评价方法，通过将可观测行为映射到具体的架构组件，再选择对应的评价指标，构建完整的评价框架。

**💡 创新点**

创新点在于首次系统化地将行为、组件与评价指标三者建立映射关系，并提供了统一的架构-行为-指标映射框架，显著提升评价的诊断性和可解释性。

**🔧 技术方法**

主要技术包括概念性分析与设计科学方法构建映射框架，结合现有工具（LangSmith、DeepEval、Opik）进行指标匹配，并对公开的LLM代理进行实验验证。

**📊 数据集**

使用真实的开源LLM代理（如SWE‑Agent、MetaGPT）及其执行日志作为实验数据，未采用传统机器学习数据集。

**📈 对比分析**

通过对比基线（无架构考虑的高层指标评估）与本方法（架构驱动的指标选择），在诊断准确性、错误归因清晰度和评价可复现性方面均表现出更佳的效果。

**⚠️ 局限性**

局限性包括：对代理架构的异质性依赖较高，方法对不同架构的适用性可能有限；评价依赖于完整的日志与工具支持，若缺失会影响结果。

---

## 375. Tournament Informed Adversarial Quality Diversity

**arXiv ID:** 2601.19562 | [PDF](https://arxiv.org/pdf/2601.19562v1)

**作者:** Timothée Anne `[一作]` (IT University of Copenhagen), Sebastian Risi `[通讯]` (IT University of Copenhagen)

**通讯引用:** 3613 | [OpenAlex ID](https://openalex.org/A5020511097)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文研究了在对抗性质量多样性（Adversarial QD）框架下的任务选择问题，改进了GENERATIONAL ADVERSARIAL MAP‑ELITES（GAME）算法；

**💡 创新点**

创新点包括提出基于锦标赛排名与Pareto前沿的两种任务选择方法，并设计六种专门衡量对抗质量与多样性的指标；

**🔧 技术方法**

技术手段涉及GAME与MTMB‑ME求解、K‑means聚类、NSGA‑III、ELO排名、锦标赛评估等；

**📊 数据集**

实验使用三种小型对抗游戏（Pong、Cat‑and‑Mouse、Pursuers‑and‑Evaders），每个游戏的策略由含32/16隐藏层的MLP实现；

**📈 对比分析**

在20次实验复现中，Ranking方案在胜率、ELO分、Expertise与AQD‑Score等多项指标上显著优于Random、Behavior和Pareto，并在大部分度量上达到最佳；

**⚠️ 局限性**

主要局限是需要执行大型锦标赛（评估成本高）、对抗任务选择与度量仍不够样本高效，且实验问题并非完全开放式，未验证在更复杂环境中的表现。

---

## 376. AROMMA: Unifying Olfactory Embeddings for Single Molecules and Mixtures

**arXiv ID:** 2601.19561 | [PDF](https://arxiv.org/pdf/2601.19561v1)

**作者:** Dayoung Kang `[一作]` (DGIST), Jinhyun So `[通讯]` (DGIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种统一嵌入空间框架 AROMMA，能够同时表示单分子和两分子混合物，并实现对香味描述符的预测。

**💡 创新点**

创新点包括：① 通过学习可辨识的嵌入空间实现单分子与混合物的双向知识迁移；② 采用注意力聚合模块（自注意力 + 可学习查询的交叉注意力）以捕捉分子间的非线性、非对称相互作用；③ 利用知识蒸馏对单分子预测进行校准，并通过类别感知伪标签扩充混合物稀疏标签，弥补公共数据集不完整性。

**🔧 技术方法**

技术手段：基于化学基础模型 SPMM（预训练 5千万分子）+ LoRA 轻量微调；自注意力与交叉注意力聚合器；多标签蒸馏损失与 BCE 损失的组合；类别感知伪标签方法提升标签覆盖率。

**📊 数据集**

数据集：公开单分子数据集 GS‑LF（约 5K 分子，138 个香味描述符）和两分子混合物数据集 BP（约 60K 对，74 个描述符），并在训练中统一为 152 个标签，随后通过伪标签生成 Pseudo‑78 / Pseudo‑152。

**📈 对比分析**

与现有最优方法（POM、MPNN‑GNN）对比，AROMMA 在单分子数据集上提升 3.2% AUROC，混合物数据集提升 19.1% AUROC，整体实现了 state‑of‑the‑art 的预测性能。

**⚠️ 局限性**

局限性：目前仅支持两分子混合物；使用 2D SMILES 取代 3D 表征，可能限制对立体效应的捕捉；数据集仍较稀疏，未来需扩展到更多分子数量与更丰富的标签。

---

## 377. "Do I Trust the AI?" Towards Trustworthy AI-Assisted Diagnosis: Understanding User Perception in LLM-Supported Reasoning

**arXiv ID:** 2601.19540 | [PDF](https://arxiv.org/pdf/2601.19540v1)

**作者:** Yuansong Xu `[一作]` (ShanghaiTech University), Quan Li `[通讯]` (ShanghaiTech University)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5100689370)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实施两步研究：首先构建9个多学科临床案例，收集8名医生与6种LLM的诊断与治疗分析；其次让37名评估者基于七维度对分析进行评分与排序，生成Perceived Capability Score，并与标准诊断基准对照。

**💡 创新点**

从医生主观感知出发定义并量化评估维度，提出Perceived Capability Score；将主观评估与传统诊断基准进行对比，揭示非线性关联与诊断推理过程对医生信任的关键作用；为AI辅助诊断提供多维度可信度评估框架。

**🔧 技术方法**

使用大语言模型（Deepseek‑V3、Gemini‑2.5‑pro、GPT‑O3 等）进行病例分析；采用交互式评估系统收集医生评分；统计方法包括 Bradley‑Terry 排名回归、累积分布混合模型、LOESS 拟合、特征重要性评估等。

**📊 数据集**

自建9个多学科临床病例（共36份病例分析），收集8名医生（不同专科与经验水平）与6种LLM的诊断与治疗输出；使用 DiagnosisArena 诊断基准评估模型 top‑1 准确率。

**📈 对比分析**

通过 LOESS 拟合 Perceived Capability Score 与 DiagnosisArena top‑1 准确率的关系，发现二者正相关但呈递减趋势；利用特征重要性比较两种评估强调维度，基准更重视诊断准确性，主观评估更关注推理过程与临床可接受性；在排名预测中，Bradley‑Terry 模型得到 Kendall’s τ ≈0.73 的较高拟合度。

**⚠️ 局限性**

样本规模有限（9 个病例、6 个 LLM、37 名评估者），评估仅基于单一基准；虚拟病人对真实临床交互的真实性不足；缺乏多模态证据支持与长期临床验证，未来需更大规模、多中心、多学科的实证研究。

---

## 378. Fuzzy expert system for the process of collecting and purifying acidic water: a digital twin approach

**arXiv ID:** 2601.19527 | [PDF](https://arxiv.org/pdf/2601.19527v1)

**作者:** Temirbolat Maratuly `[一作]` (Kazakh-British Technical University), Timur Samigulin `[通讯]` (Kazakh-British Technical University)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5000446513)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于模糊专家系统和数字孪生的酸性水收集与净化过程控制方法，并实现了可视化的Web交互模拟平台。

**💡 创新点**

创新点在于将数字孪生与模糊Split‑Range控制相结合，首次对五种去模糊方法进行系统评估，并提供了非专家人员可操作的智能控制框架。

**🔧 技术方法**

采用Honeywell UniSim Design R492构建数字孪生、MATLAB/Simulink+Fuzzy Logic Toolbox实现模糊控制、OPC DA实现实时数据交换，并用Python Streamlit搭建前端界面。

**📊 数据集**

使用数字孪生生成的自定义实验数据集（21种初始压力、5种去模糊方法共105个测试场景）以及阀门传感器采样数据。

**📈 对比分析**

通过MSE、RMSE、MAE、IAE、ISE、ITAE等误差指标和升降时间、稳态误差、超/欠调等动态指标比较五种去模糊方法，结果显示Centroid和Bisector方法在稳态精度、超调和收敛速度上表现最佳。

**⚠️ 局限性**

缺乏真实工况验证、未充分扩展成员函数和规则、未与PID等传统控制器进行充分对比、网络延迟和硬件非线性影响尚未考虑。

---

## 379. SynCABEL: Synthetic Contextualized Augmentation for Biomedical Entity Linking

**arXiv ID:** 2601.19667 | [PDF](https://arxiv.org/pdf/2601.19667v1)

**作者:** Adam Remaki `[一作]` (Sorbonne Université), Xavier Tannier `[通讯]` (Sorbonne Université)

**通讯引用:** 2982 | [OpenAlex ID](https://openalex.org/A5056834851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型为所有候选概念生成上下文丰富的合成训练样本，从而解决生物医学实体链接中的专家标注稀缺问题。

**💡 创新点**

将生成式合成数据与 decoder‑only 模型及引导推理相结合，实现跨语言的最优性能，并提出 LLM‑as‑a‑judge 评估协议提升临床有效率。

**🔧 技术方法**

主要技术包括 LLM prompt‑based 合成数据生成、decoder‑only Llama‑3‑8B、guided inference、以及自适应概念表示。

**📊 数据集**

使用 MedMentions（英语）、QUAERO（法语）和 SPACCC（西班牙语）三大公开数据集进行训练与评估。

**📈 对比分析**

与规则、上下文无关/有关 bi‑encoder、encoder‑decoder 等基线相比，SynCABEL 在四大基准 Recall@1 最高，尤其在未见概念上提升显著。

**⚠️ 局限性**

仍需人工标注以实现最佳性能，合成数据质量与覆盖度受限，低资源语言与更细粒度实体类型的扩展仍待完善。

---

## 380. Agentic Design Patterns: A System-Theoretic Framework

**arXiv ID:** 2601.19752 | [PDF](https://arxiv.org/pdf/2601.19752v1)

**作者:** Minh-Dung Dao `[一作]` (University College Cork), Hoang D. Nguyen `[通讯]` (University College Cork)

**通讯引用:** 2983 | [OpenAlex ID](https://openalex.org/A5100766549)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了基于系统理论的代理人工智能架构，将代理拆分为五个核心子系统，并提出了12种可重用的设计模式；通过对ReAct框架的案例分析验证了该方法的可行性。

**💡 创新点**

创新点在于：①用系统理论为代理设计提供了严格的分层框架；②将高层问题映射到具体模式，形成从核心子系统到设计模式的闭环；③提供了可直接落地的12种模式，弥补了以往缺乏结构化、可实现模式的不足。

**🔧 技术方法**

主要技术为系统理论建模、功能子系统划分、设计模式分析与映射；实现层面采用基于LLM的思维、感知、行动、学习与通信模块，并通过模式实现验证。

**📊 数据集**

本文未使用具体实验数据集，主要基于文献综述与对ReAct框架的定性案例分析。

**📈 对比分析**

比较方法主要为定性诊断与模式应用演示，未进行量化基准测试，因而无法给出数值性能表现。

**⚠️ 局限性**

局限性包括：①方法仍处于概念阶段，缺乏定量评估；②高级模式（如Reflector、Controller）可能引入额外计算与架构复杂度；③未充分讨论大规模自治系统的社会影响与责任问题。

---

## 381. Using LLMs to Evaluate Architecture Documents: Results from a Digital Marketplace Environment

**arXiv ID:** 2601.19693 | [PDF](https://arxiv.org/pdf/2601.19693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 382. Rethinking Divisive Hierarchical Clustering from a Distributional Perspective

**arXiv ID:** 2601.19718 | [PDF](https://arxiv.org/pdf/2601.19718v1)

**作者:** Kaifeng Zhang `[一作]`, Qiuran Zhao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于分布核的新型目标驱动分裂层次聚类方法 H‑𝒦C，解决了现有目标驱动分裂层次聚类因采用基于集合的分裂评估标准导致的三种期望性质缺失问题，构建了符合三项期望性质的树状图；

**💡 创新点**

创新点在于：①将聚类视为分布而非集合，使用分布核来衡量聚类相似性并指导分裂；②引入核心聚类的分布式分裂步骤，避免了全局集合评估的局限；③提出了线性时间复杂度（O(kn+s²)）的实现，并提供了全局目标下的理论下界保证；

**🔧 技术方法**

主要技术包括分布核（Kernel Mean Embedding）与分布核相似度计算、核心聚类预处理（如 psKC 或 DBSCAN）、贪婪分裂策略、后期细化分配以及线性时间实现；

**📊 数据集**

在人工合成数据集、空间转录组学（HER2 肿瘤细胞数据集、Slide‑seq V2 小鼠海马数据集）以及 13 个公开基准数据集上进行实验；

**📈 对比分析**

与现有方法（Bisect‑Kmeans、SpecWRSC、DIANA、HDP、SCC 等）进行对比，采用树状图纯度（Dendrogram Purity）以及临界差异图评估，结果显示 H‑𝒦C 的树状图纯度最高，且在大规模数据集上保持线性时间优势；

**⚠️ 局限性**

局限性包括：①对分布核参数（如 IDK 与 GDK）的敏感性需进一步探索；②核心聚类的选择与预处理可能影响最终结果；③当前评估指标仍以可视化和树状图纯度为主，缺乏更细粒度的定量评价标准；

---

## 383. Physics-Aware Novel-View Acoustic Synthesis with Vision-Language Priors and 3D Acoustic Environment Modeling

**arXiv ID:** 2601.19712 | [PDF](https://arxiv.org/pdf/2601.19712v1)

**作者:** Congyi Fan `[一作]`, Wenwu Wang `[通讯]` (University of Surrey)

**通讯引用:** 9876 | [OpenAlex ID](https://openalex.org/A5100676721)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了Phys-NVAS框架，实现多视角3D声学环境重建与视觉语言语义先验融合，用于新视角声学合成。

**💡 创新点**

创新点在于首次将物理感知的视觉语言先验（对象、布局、材质）与多视角几何信息统一为物理感知特征，实现更真实、物理一致的空间音频生成。

**🔧 技术方法**

主要技术包括3D Gaussian Splatting、深度估计、Chat-UniVi等视觉语言模型、特征融合适配器以及基于AV-NeRF的双耳音频生成器。

**📊 数据集**

实验采用RWAVS数据集，包含办公、住宅、楼梯与户外四种场景的多模态音视频样本。

**📈 对比分析**

与Mono-Mono、Mono-Energy、Stereo-Energy、INRAS、NAF、ViGAS及AV-NeRF等基线对比，Phys-NVAS在MAG和ENV指标上均获得最低分，提升约5-10%。

**⚠️ 局限性**

限制在于仍依赖高质量的多视角图像与深度估计，且对极端材料或遮挡情况的鲁棒性待进一步验证。

---

## 384. Differentiable Semantic ID for Generative Recommendation

**arXiv ID:** 2601.19711 | [PDF](https://arxiv.org/pdf/2601.19711v1)

**作者:** Junchen Fu `[一作]` (University of Glasgow), Zhaochun Ren `[通讯]` (Leiden University)

**通讯引用:** 7683 | [OpenAlex ID](https://openalex.org/A5100384130)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可微分语义ID（DIGER）框架，允许生成式推荐模型在训练期间联合优化语义编码与推荐目标，解决传统两阶段训练中语义索引与推荐目标不匹配的问题。

**💡 创新点**

创新点在于引入带Gumbel噪声的探索性分配（DRIL）和两种不确定性衰减策略（SDUD与FrqUD），实现从早期随机探索到后期确定性利用的平滑过渡，从而避免代码坍塌并提升代码利用率。

**🔧 技术方法**

技术主要包括：基于RQ‑VAE的离散语义ID生成、Gumbel‑Softmax采样、soft更新策略、标准差衰减与频率衰减两种不确定性调节机制，以及联合损失（生成损失 + 量化重建损失）。

**📊 数据集**

实验使用了三个公开数据集：Amazon Beauty、Amazon Instruments、Yelp，均包含商品/餐厅文本描述。

**📈 对比分析**

与传统两阶段生成推荐、STE、以及多种协同过滤和序列推荐基线（如SASRec、BERT4Rec、TIGER、LETTER、ETEGRec）对比，DIGER在R@10/NDCG@10上均取得领先或相近性能，明显优于STE，且在Beauty和Instruments上实现SOTA。

**⚠️ 局限性**

局限性包括：仅关注语义ID的生成，未结合协同过滤或大语言模型的额外信息；对不确定性衰减的超参数仍需经验调优；在更大规模或多模态场景下的可扩展性与稳定性尚未验证。

---

## 385. Video-KTR: Reinforcing Video Reasoning via Key Token Attribution

**arXiv ID:** 2601.19686 | [PDF](https://arxiv.org/pdf/2601.19686v1)

**作者:** Ziyue Wang `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15342 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种视频推理的模态感知策略塑形框架Video-KTR，利用 token‑level 的强化学习提升多模态大语言模型的推理能力。

**💡 创新点**

创新点在于三重归因信号的融合：视觉对比掩码、时间帧打乱以及高熵不确定性，精准识别并强化关键 tokens，避免粗粒度奖励带来的信息噪声。

**🔧 技术方法**

采用 counterfactual 掩码、帧打乱扰动、熵测度等归因方法结合 GRPO 框架进行 token‑level 经验更新，并对模型的 decoder logits 进行对比。

**📊 数据集**

在 Video-R1、Video-Holmes、VideoMMMU、MMVU、TempCompass 等五个挑战性视频推理与理解基准上进行实验，使用 260K 条 RL 样本及 1.5K 视频霍姆斯训练样例。

**📈 对比分析**

与多种开源和闭源基线（如 LLaVA-OV、Video-RTS、TW‑GRPO 等）对比，Video‑KTR 在 Video‑Holmes 上获得 42.7% 的准确率，超过 GPT‑4o（42.0%）并在其他基准上保持 SOTA 或竞争力。

**⚠️ 局限性**

局限性包括对 token 选择阈值的依赖、需要额外的对比扰动计算、以及在更大规模或实时视频推理任务中的可扩展性和计算成本待进一步验证。

---

## 386. SharpNet: Enhancing MLPs to Represent Functions with Controlled Non-differentiability

**arXiv ID:** 2601.19683 | [PDF](https://arxiv.org/pdf/2601.19683v1)

**作者:** Hanting Niu `[一作]` (Institute of Software, Chinese Academy of Sciences), Ying He `[通讯]` (Nanyang Technological University)

**通讯引用:** 7911 | [OpenAlex ID](https://openalex.org/A5100389169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了SharpNet，一种在MLP中加入Poisson方程求解得到的特征函数，使网络能够在用户指定位置实现控制的C^0非可微特征，适用于2D距离场和3D CAD重建；

**💡 创新点**

核心创新在于用可微的PDE特征函数来精准定位并控制梯度不连续性，既保持全局平滑，又可在任意闭合或开放曲线/曲面处产生刚性锐角；

**🔧 技术方法**

技术手段包括：Poisson方程的Green函数积分、mollifier局部化、软化/ReLU激活、位置编码、Eikonal损失、梯度一致性约束以及可微分的特征曲线/曲面学习；

**📊 数据集**

实验使用ABC数据集（100个CAD模型）进行3D重建，2D合成几何的距离场与中轴学习；

**📈 对比分析**

与SIREN、InstantNGP、NH‑Rep、Patch‑Grid、NeurCADRecon等基线比较，SharpNet在Chamfer、Hausdorff、法向误差与F‑1分数上均优于对手，且在锐边保留与视觉质量上表现突出；

**⚠️ 局限性**

局限性包括：特征曲面需事先提供或初始化，无法从无到有学习；对拓扑缺陷不具修复能力；特征曲面离散化为折线/三角网导致计算量随特征复杂度增大；目前仅支持C^0级别的非可微，无法处理更高阶不连续。

---

## 387. Cross-Domain Offshore Wind Power Forecasting: Transfer Learning Through Meteorological Clusters

**arXiv ID:** 2601.19674 | [PDF](https://arxiv.org/pdf/2601.19674v1)

**作者:** Dominic Weisser `[一作]` (University College London), Benjamin Guedj `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对海上风电场缺乏历史数据的问题，提出了一套基于气象聚类的迁移学习框架，通过预训练一组气象模式专用模型，并在新风电场进行少量数据微调，实现跨域的风电功率预测。

**💡 创新点**

创新点在于：①将海上风电功率预测拆分为气象模式聚类与模式专属模型；②使用变分自编码器对气象序列进行压缩表示，构建多模式的高斯过程回归；③在迁移阶段将目标风电场的气象序列映射到源域潜在空间，直接对相应模式的高斯过程进行微调，从而极大降低对本地数据的依赖。

**🔧 技术方法**

核心技术包括：Variational Autoencoder（VAE）用于气象序列降维与表示；层次聚类+Ward 链接实现气象模式划分；多模态 Gaussian Process 回归，核函数为 RBF + Matérn 3/2 的混合；迁移学习三步流程（潜在映射、模式对齐、GP 微调）。

**📊 数据集**

数据集为 29 个欧洲海上风电场的 ERA5 气象与合成功率时间序列（2018‑2019 年 40 年数据），并在 8 个未见风电场上做迁移实验，使用源域 6 个风电场构建聚类与模型。

**📈 对比分析**

与无迁移的专属 GP 模型做对比，采用 MAE、RMSE、R² 等指标。实验显示，在目标风电场仅 20%（约 5 个月）本地数据时，迁移学习平均 MAE 为 3.52%（26.7 MW），比无迁移基线低 14.7%（4.13%），甚至优于使用完整一年本地数据的基线模型；其他指标也均有提升。

**⚠️ 局限性**

局限性包括：①对气象模式的划分依赖于源域的代表性，若目标风电场气象特征与源域差异大，迁移效果会下降；②模型对风机和湍流等细粒度物理效应考虑不足，无法捕捉塔台位置、机组异质性等影响；③对极端天气模式的迁移表现仍需更多本地数据。

---

## 388. ProToken: Token-Level Attribution for Federated Large Language Models

**arXiv ID:** 2601.19672 | [PDF](https://arxiv.org/pdf/2601.19672v1)

**作者:** Waris Gill `[一作]` (Virginia Tech), Muhammad Ali Gulzar `[通讯]` (Virginia Tech)

**通讯引用:** 816 | [OpenAlex ID](https://openalex.org/A5003747461)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种针对联邦学习（Federated Learning）环境下大型语言模型（LLM）的token级归因（provenance）方法，能够在不泄露客户端数据的前提下，准确追踪生成文本时哪些客户端对特定token的产生做出了贡献。

**💡 创新点**

创新点包括：①利用联邦聚合的线性特性，将全局模型前向计算拆解为各客户端的加权求和；②只关注Transformer后几层的关键子模块（自注意力输出投影和MLP最后层），大幅降低计算量；③采用梯度加权的相关性评分，自动过滤无关神经元，提升归因准确性。

**🔧 技术方法**

核心技术：梯度加权归因（gradient‑based attribution），对每个token在选定层的激活与梯度内积求和，随后聚合所有token得到全局归因分布；采用FL聚合公式FedAvg拆解；仅在模型更新和激活层面操作，保证FL隐私。

**📊 数据集**

实验使用四种LLM架构（Gemma‑3‑270M、SmolLM‑2‑360M、Llama‑3.2‑1B、Qwen‑2.5‑0.5B）以及四个领域数据集（医学、金融、数学推理、代码），每个配置都在6或55个客户端上训练，采样多样化。

**📈 对比分析**

与无梯度加权基线相比，归因准确率平均提升约1.86倍（从35.7%到66.3%）。在16个配置（4模型×4域）上，平均归因准确率达98.62%；在扩大到55个客户端（25个恶意客户端）时仍保持92–95%的准确率，且归因概率分布对“贡献”与“非贡献”客户端分离明显。

**⚠️ 局限性**

局限性：①评估依赖于人工植入的后门触发器，难以直接验证在真实业务场景中的归因可靠性；②方法假设联邦聚合为线性且模型架构符合Transformer后层任务特定化的特征，可能对非Transformer或非线性聚合策略效果有限；③虽然大幅降低计算量，但在极大模型与客户端数下仍存在一定延迟；④仅追踪生成时的贡献，无法覆盖模型在训练阶段的潜在恶意改动。

---

## 389. One Token Is Enough: Improving Diffusion Language Models with a Sink Token

**arXiv ID:** 2601.19657 | [PDF](https://arxiv.org/pdf/2601.19657v1)

**作者:** Zihou Zhang `[一作]` (Xiaohongshu), Shaosheng Cao `[通讯]` (Xiaohongshu)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对扩散语言模型的注意力 Sink 迁移不稳定问题，提出在序列前置一个专用低信息 Sink token 并通过修改注意力掩码实现稳定性。

**💡 创新点**

创新点是引入位置稳定的 Sink token 并用极低范数状态抵消信息过度混合，从而显著提升推理鲁棒性。

**🔧 技术方法**

使用 Transformer、扩散模型、修改的注意力掩码以及额外的 Sink token。

**📊 数据集**

在多种基准数据集上验证，包括 ARC‑e/c、HellaSwag、PIQA、RACE、SIQA、LAMBADA、GSM8K 以及 FineWeb、SlimPajama 等。

**📈 对比分析**

与无 Sink、门控注意力等对比，实验显示加入单个 Sink token 后在 0.5B/1.5B 规模模型上平均提升 10%~15% 点，且效果对位置和数量不敏感。

**⚠️ 局限性**

限制在于未在更大规模模型上测试，且只验证了语言文本任务，未探究多模态或其他领域。

---

## 390. Advanced Modeling of Interlanguage Speech Intelligibility Benefit with L1-L2 Multi-Task Learning Using Differentiable K-Means for Accent-Robust Discrete Token-Based ASR

**arXiv ID:** 2601.19767 | [PDF](https://arxiv.org/pdf/2601.19767v1)

**作者:** Kentaro Onda `[一作]` (University of Tokyo), Nobuaki Minematsu `[通讯]` (University of Tokyo)

**通讯引用:** 2851 | [OpenAlex ID](https://openalex.org/A5041213266)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出一种利用可微分k-means和多任务学习的离散词元语音识别模型，旨在增强系统对外来口音的鲁棒性。

**💡 创新点**

创新点在于通过对L1与L2两种语言的ASR任务同时进行优化，模拟人类非母语听者对母语的感知偏好（ISIB），并使词元在整个模块中可微分更新。

**🔧 技术方法**

使用的核心技术包括HuBERT自监督学习特征提取、可微分k-means聚类、CTC/注意力联合解码器以及多任务学习框架。

**📊 数据集**

实验数据集主要为日语母语者的英语口音语料（ERJ）、英美母语者的英语语料（LibriSpeech）以及日语母语语料（CSJ）等。

**📈 对比分析**

与传统基于非可微分k-means的离散词元ASR以及仅针对L2优化的模型相比，方法在无口音数据的native‑only场景下提升了日语口音英语的识别精度，在有限口音数据的accent‑adapted场景下相对WER下降约19–20%。

**⚠️ 局限性**

局限性包括：实验仅针对日语口音英语，未验证其他口音；需要先验的L1信息，无法处理未知母语；以及对计算资源和模型规模的依赖。

---

## 391. GraphDLG: Exploring Deep Leakage from Gradients in Federated Graph Learning

**arXiv ID:** 2601.19745 | [PDF](https://arxiv.org/pdf/2601.19745v1)

**作者:** Shuyue Wei `[一作]` (Shandong University), Lizhen Cui `[通讯]` (Shandong University)

**通讯引用:** 7223 | [OpenAlex ID](https://openalex.org/A5101414718)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了联邦图学习（FGL）中的深度梯度泄露问题，提出一种可从共享梯度恢复完整训练图（节点特征与图结构）的攻击方法。

**💡 创新点**

创新点在于：①通过理论分析得到图结构已知时节点特征可通过闭式递归公式恢复；②利用异构辅助图训练解码器恢复图结构，并加入最大均值差分正则化实现自适应匹配；③将结构恢复与特征恢复联合完成，显著提升了恢复精度。

**🔧 技术方法**

使用的技术包括：GCN+MLP模型、梯度解析与闭式递归规则、自动编码器（解码器）、最大均值差分（MMD）适配器、线性方程求解以及对抗式自适应正则化。

**📊 数据集**

实验数据集为五个图分类数据集：MUTAG、PTC_MR、AIDS、ENZYMES、PROTEINS，辅助数据还包括随机生成的Erdős–Rényi图和不同分布的公共图数据集。

**📈 对比分析**

与八个基线方法（包括DLG、iDLG、InverGrad、GI‑GAN、GRA‑GRF、TabLeak、Graph Attacker和随机方法）进行比较；在节点特征恢复方面MSE下降5.46%，在图结构恢复方面AUC提升25.04%，在所有数据集上均显著优于现有方法。

**⚠️ 局限性**

局限性：对梯度压缩防御效果有限，差分隐私对结构保护不够强；攻击依赖于可获取的梯度、模型参数和辅助图，实际部署中可能受到数据获取和通信约束。

---

## 392. Joint Power Allocation and Antenna Placement for Pinching-Antenna Systems under User Location Uncertainty

**arXiv ID:** 2601.19704 | [PDF](https://arxiv.org/pdf/2601.19704v1)

**作者:** Hao Feng `[一作]` (Hunan Institute of Engineering), Octavia A. Dobre `[通讯]` (Memorial University)

**通讯引用:** 22580 | [OpenAlex ID](https://openalex.org/A5077149719)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在高频下考虑高斯定位误差的 pinching‑antenna 系统中，功率分配与天线位置的鲁棒联合优化，目标是最大化系统能效。

**💡 创新点**

创新点在于：① 将用户定位误差建模为高斯分布并利用 Marcum Q 函数精确描述概率约束；② 推导出满足该约束的最小功率解析表达式；③ 用粒子群优化（PSO）寻找全局最优天线位置。

**🔧 技术方法**

采用技术包括：Marcum Q 函数、非中心卡方分布、解析功率求解、粒子群优化（PSO）以及 TDMA 时分多址。

**📊 数据集**

数据集为仿真随机生成的 5 个用户估计位置（在 120 m×20 m 区域内），定位误差方差 σ²=1，频率 28 GHz、带宽 100 MHz 等参数。

**📈 对比分析**

通过与穷举搜索（全局最优）和固定天线基准比较，PSO 方案几乎达到穷举最优，并且相较于固定天线显著提升能效。

**⚠️ 局限性**

局限性：仅考虑单波导单天线、每个用户单独时隙，未考虑多波导/多天线协同；对高斯误差假设过于理想，实际环境中的误差分布可能更复杂。

---

## 393. LoPRo: Enhancing Low-Rank Quantization via Permuted Block-Wise Rotation

**arXiv ID:** 2601.19675 | [PDF](https://arxiv.org/pdf/2601.19675v1)

**作者:** Hongyaoxing Gu `[一作]` (Institute of Software Chinese Academy of Sciences), Fangfang Liu `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需微调的后训练量化方法LoPRo，能够在2‑3位量化下保持高精度；

**💡 创新点**

创新点包括在低秩分解残差矩阵上采用基于排列的分块Walsh-Hadamard旋转，以及使用rank‑1随机SVD（R1SVD）实现高效低秩近似；

**🔧 技术方法**

核心技术包括低秩分解、列排列与分块旋转、量化损失优化、混合精度R1SVD、标量/向量量化；

**📊 数据集**

实验数据集涵盖LLaMA‑2、LLaMA‑3、Mixtral‑8x7B、Qwen2.5、Qwen3以及WikiText2和四个零样本推理基准（ARC‑challenge、ARC‑easy、PIQA、Winogrande）；

**📈 对比分析**

与GPTQ、GPTVQ、LQER、OminiQ、QuIP#、MoEQuant等SOTA无微调量化方法对比，LoPRo在2/3位下均实现最高或相近准确率，压缩率高、推理延迟<10%，并在Mixtral‑8x7B量化时间≤2.5h，速度提升至4×；

**⚠️ 局限性**

局限性主要包括：仍需对rank和块大小做超参数调优；对极低位（≤1‑bit）或特殊模型结构的适应性未充分验证；在极大规模模型时，低秩近似与旋转的计算开销仍非零。

---

## 394. A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models

**arXiv ID:** 2601.19673 | [PDF](https://arxiv.org/pdf/2601.19673v1)

**作者:** Iwona Christop `[一作]` (Adam Mickiewicz University), Marek Kubis `[通讯]` (Adam Mickiewicz University)

**通讯引用:** 135 | [OpenAlex ID](https://openalex.org/A5076330785)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个名为Audio Reasoning Tasks（ART）的音频推理基准，用来评估多模态大语言模型在需要跨音频任务进行推理的能力。

**💡 创新点**

创新点在于：①将传统的单项音频任务评估转为需要组合多种音频技能进行推理的复合任务；②通过规则筛选与专家评审，确保任务可被普通人（无听力障碍且不需专业训练）理解；③使用模板化生成方法，控制数据多样性与防止数据污染。

**🔧 技术方法**

技术包括：模板化任务生成、Voicebox+HiFi‑GAN合成语音、Whisper ASR（用于文本化与验证）、LLM（如Llama‑3.3、Qwen3）作为答案生成与评判者，实验中使用Yes/No与Descriptive两种回答方式。

**📊 数据集**

数据集包含9,000条样本，9个任务，使用LJ Speech、GloVe、VoxPopuli、Freesound等公开数据集进行声音与文本采集与合成。

**📈 对比分析**

实验将ART与现有音频基准对比，结果显示大多数开源多模态LLM在Yes/No评估下平均准确率仅为0.43–0.48，单任务最高可达0.6左右；相比专门的单项任务评估，ART对模型提出更高推理挑战。

**⚠️ 局限性**

限制包括：①任务集不完整，完成该基准并不保证模型音频推理已达到人类水平；②评估主要聚焦推理而非单项音频识别精度；③使用合成语音可能导致部分任务过于理想或过度乐观；④评判者与被评模型相同或相似可能引入偏差。

---

## 395. Topology-Aware Subset Repair via Entropy-Guided Density and Graph Decomposition

**arXiv ID:** 2601.19671 | [PDF](https://arxiv.org/pdf/2601.19671v1)

**作者:** Guoqi Zhao `[一作]` (Harbin Institute of Technology), Xiaolong Wan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5085267247)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于密度‑冲突度惩罚的子集修复框架，并设计了 EntroCFDensity、冲突检测优化与两种修复算法 PPIS 与 MICO。

**💡 创新点**

创新点包括：① 将信息熵与 CFD 权重相结合得到可调属性权重的密度度量；② 引入冲突度作为全局拓扑惩罚；③ 通过冲突图分解成连通子图并在每个子图中求最小删集；④ 采用系数变异（CV）自适应地分配密度与冲突度权重。

**🔧 技术方法**

使用技术包括：倒排索引＋规则分组冲突检测、图论最小顶点覆盖与整数规划、k‑近邻密度估计、信息熵与 CFD 频率权重、迭代深度优先搜索进行连通分解。

**📊 数据集**

实验数据集涵盖 9 个公开数据集：Hospital1k、Food、Flights、Beers、Soccer、Movies、Hospital10k、Restaurants、Adult 以及示例表。

**📈 对比分析**

与传统 HEURISTIC、RELAXATION 等方法在精度、召回、F1、运行时间、删除量等指标上比较。PPIS 取得最短运行时间并保持 F1 接近最优；MICO 在大多数数据集上 F1 最高，但在 Adult 数据集因过度删除导致精度下降。

**⚠️ 局限性**

局限性包括：① 在高度密集或复杂冲突图下 PPIS 可能误删清洗数据；② MICO 在极大图上求解困难、可能超时或返回过度删除；③ 对如 Adult 这类异常分布的数据鲁棒性不足；④ 采用连通分解假设无跨组件冲突，可能忽略细粒度约束。

---

## 396. KeepLoRA: Continual Learning with Residual Gradient Adaptation

**arXiv ID:** 2601.19659 | [PDF](https://arxiv.org/pdf/2601.19659v1)

**作者:** Mao-Lin Luo `[一作]` (Southeast University), Min-Ling Zhang `[通讯]` (Southeast University)

**通讯引用:** 15182 | [OpenAlex ID](https://openalex.org/A5079083101)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 KeepLoRA 方法，在持续学习中通过低秩适配器并将参数更新限制在残差子空间，既保持预训练知识，又兼顾前向/后向稳定性与可塑性。

**💡 创新点**

创新点在于将模型参数主子空间视为一般知识子空间，将残差子空间视为特定知识子空间，并通过梯度投影初始化 LoRA 的下投影矩阵，使其在主子空间正交的残差子空间内更新，从而实现三方平衡。

**🔧 技术方法**

采用了 SVD 子空间分解、梯度投影、LoRA 参数化、主子空间正交约束以及理论证明等技术。

**📊 数据集**

在 CLIP 的 MTIL 基准（含多种视觉分类任务）、LLaVA 的 MLLM‑DCL 与 UCIT VQA 基准以及标准视觉数据集如 CIFAR‑100、Caltech101、DTD 等数据集上进行实验。

**📈 对比分析**

与 LoRA、O‑LoRA、InfLoRA、SD‑LoRA、LwF 等基线对比，在 Transfer、Average、Last 等指标上实现 SOTA，提升幅度约 5%–10% 以上，尤其在后期任务的准确率上显著优于现有方法。

**⚠️ 局限性**

局限性包括需手动设置子空间能量阈值 ε_w、ε_f，且在极大模型或长任务序列下的扩展效果尚未充分验证，低资源任务的泛化能力仍有待提升。

---

## 397. Reimagining Social Robots as Recommender Systems: Foundations, Framework, and Applications

**arXiv ID:** 2601.19761 | [PDF](https://arxiv.org/pdf/2601.19761v1)

**作者:** Jin Huang `[一作]` (University of Cambridge), Hatice Gunes `[通讯]` (University of Cambridge)

**通讯引用:** 7486 | [OpenAlex ID](https://openalex.org/A5060090893)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出将推荐系统（RS）技术无缝集成到社交机器人中，构建了以用户画像、排名和责任计算为核心的模块化框架；

**💡 创新点**

创新点在于将RS的三大核心技术（用户建模、个性化排序与伦理责任）与机器人感知-认知-行动管线对齐，形成可插拔的“机器人即RS”架构；

**🔧 技术方法**

主要技术包括协同过滤与序列模型的用户画像、检索‑再排序的个性化排名、联邦学习/差分隐私与公平性约束的责任计算，以及与大型语言模型（LLM）和视觉语言模型（VLAM）的接口；

**📊 数据集**

文中未公开具体实验数据集，主要基于公开的社交机器人交互日志与推荐系统标准数据（如MovieLens、Yelp）进行概念验证；

**📈 对比分析**

作者未给出定量性能评估，说明该工作属于框架与方法论探索，未来需在真实机器人部署中与传统自适应与LLM增强方法做实验对比；

**⚠️ 局限性**

局限性包括：多用户隐私控制缺乏细粒度实现；与云端LLM的集成存在表征不匹配；多模态隐式反馈与偏见公平性在机器人环境中尚未完全解决；安全、透明与责任监管仍需进一步研究。

---

## 398. Benchmarking Multimodal Large Language Models for Missing Modality Completion in Product Catalogues

**arXiv ID:** 2601.19750 | [PDF](https://arxiv.org/pdf/2601.19750v1)

**作者:** Junchen Fu `[一作]` (University of Glasgow), Xuri Ge `[通讯]` (Shandong University)

**通讯引用:** 186 | [OpenAlex ID](https://openalex.org/A5026473068)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对电商平台缺失图像或文本的商品信息，作者提出了 MMPCBench 基准，包含内容质量和推荐两大子任务，并系统评估了六款先进的多模态大语言模型（Qwen2.5-VL、Gemma-3）在图像→文本和文本→图像的缺失模态生成。

**💡 创新点**

创新点在于：①首次构建完整的电商缺失模态补全基准；②将生成质量与推荐效果两维度评估相结合；③发现模型规模与任务表现非单调；④通过 Group Relative Policy Optimization (GRPO) 进行细粒度对齐，显著提升图像→文本生成质量。

**🔧 技术方法**

主要技术包括多模态大语言模型（Qwen2.5-VL、Gemma-3）、文本到图像扩散模型（stable‑diffusion‑3.5‑large‑turbo）、CLIP 与 BERTScore 等语义相似度度量、以及基于奖励函数的 GRPO 强化学习。

**📊 数据集**

使用的数据集是最新的 Amazon Review Dataset（2024 版），涵盖 9 个主流商品类别（All Beauty、Electronics、Home & Kitchen 等），每类采样 1,000 条商品做内容质量评测，完整数据集用于推荐基准。

**📈 对比分析**

评估方法：对六款 MLLM 在 I→T 与 T→I 方向分别计算文本质量（Cosine、Euclidean、Overlap、BERTScore）和图像质量（PSNR、SSIM、MSE、LPIPS、CLIP），并在三种多模态推荐模型（VBPR、BM3、FREEDOM）中测算 Recall@k、NDCG@k。实验显示：模型在语义层面表现优异，但在词/像素级对齐上表现有限；规模较大模型不一定更好；补全后的模态在推荐任务中几乎不逊色真实模态。

**⚠️ 局限性**

局限性：文本→图像方向对齐效果不佳；生成内容在细粒度词汇和像素层面仍有偏差；部分类别样本量不足导致评估不稳定；基准主要基于自动指标，缺乏人工或在线真实评测；GRPO 对 T→I 的提升有限，需更复杂的奖励机制。

---

## 399. TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching

**arXiv ID:** 2601.19739 | [PDF](https://arxiv.org/pdf/2601.19739v1)

**作者:** Runjia Zeng `[一作]` (Rochester Institute of Technology), Dongfang Liu `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 3770 | [OpenAlex ID](https://openalex.org/A5101979292)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种通用插件，用实例感知的token寻址与舍弃策略，在LLM微调过程中显著降低激活内存占用；

**💡 创新点**

创新点在于：①利用上下文注意力与梯度幅值双重信息对每个token进行实例级重要性评估；②在此基础上动态舍弃低重要度token，只对高重要度token进行梯度更新，从而实现高效且稳定的微调；

**🔧 技术方法**

技术包括Transformer注意力机制、梯度归一化、对数变换、激活重用（只缓存必要pre‑activation）、与PEFT方法（LoRA、LoHa、QLoRA）兼容的插件化实现；

**📊 数据集**

使用公开指令调优数据集Open‑Platypus，以及常用少量样本基准（MMLU、ARC、HellaSwag、TruthfulQA、WinoGrande）进行评估；

**📈 对比分析**

与全参数微调、随机token舍弃(TokenTune)、以及多种PEFT方法比较，实验显示在Llama3.2 1B/3B和Qwen2.5 0.5B上，内存可压缩至14.8%（仅2.8GB），同时保持或略优于全token微调的性能；

**⚠️ 局限性**

局限性：对小规模模型（如Qwen 0.5B）在非PEFT设置下表现略逊；需手动调节token比例与α/β权重；在极低token比例下可能导致收敛不稳定。

---

## 400. Stability and Generalization of Nonconvex Optimization with Heavy-Tailed Noise

**arXiv ID:** 2601.19730 | [PDF](https://arxiv.org/pdf/2601.19730v1)

**作者:** Hongxu Chen `[一作]`, Luo Luo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种在重尾噪声下的随机非凸优化的稳定性基础上的泛化分析框架，探讨了算法的稳定性与泛化误差之间的关系。

**💡 创新点**

创新点在于首次建立了在重尾噪声下，非凸问题的算法稳定性与泛化误差之间的联系，并引入了截断技术来控制随机梯度的重尾噪声。

**🔧 技术方法**

使用了截断技术和算法稳定性分析的方法，特别是针对重尾噪声的p-BCM条件进行了研究。

**📊 数据集**

使用了多种流行的随机算法，包括剪切随机梯度下降（clipped SGD）和归一化随机梯度下降（normalized SGD），以及它们的迷你批次和动量变体。

**📈 对比分析**

与现有的泛化界限进行了比较，结果表明在重尾噪声下，所提出的方法在稳定性和泛化性能上优于传统的有界方差假设下的结果。

**⚠️ 局限性**

限制在于目前的分析主要集中在重尾噪声的p-BCM条件下，未来的工作可以扩展到非光滑设置下的误差界限，并开发在重尾噪声下学习的下界。

---

## 401. RvB: Automating AI System Hardening via Iterative Red-Blue Games

**arXiv ID:** 2601.19726 | [PDF](https://arxiv.org/pdf/2601.19726v1)

**作者:** Lige Huang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 9670 | [OpenAlex ID](https://openalex.org/A5023198186)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了训练无关、顺序、信息不完全的 Red Team vs Blue Team (RvB) 框架，通过攻击者与防御者的交互式博弈实现 AI 系统的持续硬化。

**💡 创新点**

创新点在于：①利用外部化记忆实现无参数更新的动态攻防；②统一攻防两侧在同一游戏结构中迭代学习，提升对未知威胁的自适应；③在代码硬化和守栏优化两个领域同时验证，展示跨域可迁移性。

**🔧 技术方法**

使用的大语言模型代理（CAI、Mini‑SWE‑Agent、CoP、NeMo Guardrails 等）与工具调用、日志交互；贝叶斯信念更新与信息熵分析来衡量攻击者知识演化；基于推理与强化的交互式博弈实现。

**📊 数据集**

实验数据集包括：10 个 Web 漏洞（如 CVE‑2022‑30887）、HarmBench、JailBreakBench、AdvBench、SorryBench、XGuard‑Train 等外部 jailbreak 基准。

**📈 对比分析**

通过与传统合作式多智能体基线比较，评估蓝队防御成功率 (DSR)、红队攻击成功计数 (ASC) 与服务中断率；结果显示在代码硬化任务中 DSR 90%、guardrail 45%，假正率接近 0%，且对未见攻击具备良好泛化，明显优于基线。

**⚠️ 局限性**

局限性：高度依赖基础模型的推理能力；实验规模受限于选定漏洞与 benchmark，尚未在更大多样化环境下验证；模拟采用离散回合，未涵盖真实异步安全操作的复杂性。

---

## 402. Quantum Takes Flight: Two-Stage Resilient Topology Optimization for UAV Networks

**arXiv ID:** 2601.19724 | [PDF](https://arxiv.org/pdf/2601.19724v1)

**作者:** Huixiang Zhang `[一作]` (Lakehead University), Octavia A. Dobre `[通讯]` (Memorial University of Newfoundland)

**通讯引用:** 22580 | [OpenAlex ID](https://openalex.org/A5077149719)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种两阶段量子辅助拓扑控制框架，先在离线阶段使用量子退火（QA）生成结构多样、高质量的候选拓扑，再在在线阶段通过轻量级经典评估快速挑选最适合当前链路稳定性和能量状态的拓扑，以实现无人机网络的快速、可靠自适应。

**💡 创新点**

创新点包括：
1) 将 UAV 拓扑优化问题映射为 QUBO，利用 QA 的量子并行性同时探索多种拓扑；
2) 采用迭代相似度惩罚机制，系统性地产生结构多样的候选集合；
3) 将离线量子计算与在线轻量级选择结合，兼容 SDN/O‑RAN 架构，实现可部署的飞行时延控制。

**🔧 技术方法**

使用的技术：
- 量子退火（QA）与传统模拟退火（SA）对比；
- QUBO 建模与频率惩罚矩阵；
- 经典实时评估（基于 SINR 与剩余能量）；
- SDN/O‑RAN 软硬件分离框架。

**📊 数据集**

数据集：
- 通过仿真生成的 UAV 动态网络（节点数 25/50/100，30 s 时域，1 s 步长）；
- 20 次独立的部署与移动轨迹；
- 设定 3 次扰动事件（链路中断）。

**📈 对比分析**

比较方法与性能：
- QA 与 SA 在离线阶段的对比：QA 目标值降低 5.15%，多样性提升 28.3%；
- 在线性能对比：两阶段框架的性能保留率 PR=0.920，高于单一最优模型 PR=0.863，提升约 6.6%；
- 计算时延：QA 在 N=25 时略慢于 SA，但随规模增大 QA 的增长更平缓且更稳定。

**⚠️ 局限性**

局限性：
- 受限于当前量子硬件规模，无法直接处理极大规模 UAV 群；
- 需要在云端完成离线计算，增加部署成本与时延；
- 论文仅基于仿真验证，缺乏实飞测试；
- 轻量级在线评估仍需在 UAV 上实现，可能受到能耗与计算资源的进一步限制。

---

## 403. Improving Policy Exploitation in Online Reinforcement Learning with Instant Retrospect Action

**arXiv ID:** 2601.19720 | [PDF](https://arxiv.org/pdf/2601.19720v1)

**作者:** Gong Gao `[一作]` (Tongji University), Ning Jia `[通讯]` (Tongji University)

**通讯引用:** 4491 | [OpenAlex ID](https://openalex.org/A5101709263)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Instant Retrospect Action（IRA）算法，改进价值基强化学习的策略利用效率

**💡 创新点**

创新点包括：① Q-Representation Discrepancy Evolution（RDE）提升邻近动作的表示区分；② Greedy Action Guidance（GAG）利用邻近最优动作作为更新锚点；③ Instant Policy Update（IPU）提高策略更新频率；④ 通过动作缓冲检索邻近动作并约束探索，降低 Q 值估计偏差

**🔧 技术方法**

使用的技术包括：TD3/DDPG 基础框架、双 Q 网络、目标网络、Chebyshev 距离检索、k‑Nearest Neighbor、RDE 损失、策略约束正则、即时更新机制、经验回放与动作缓冲

**📊 数据集**

使用的数据集是 MuJoCo 连续控制任务（8 个）：HalfCheetah-v3、Hopper-v3、Walker2d-v3、Ant-v3、Humanoid-v3、Reacher-v2、InvertedDoublePendulum-v2、InvertedPendulum-v2

**📈 对比分析**

方法通过与 TD3、DDPG、PPO、ALH、PEER、MBPO 等基线在相同超参数下对比，平均归一化得分提升至 98.7%（vs 72.1% 等），在所有任务上平均提升 36.9% 以上，收敛更快、估计偏差更小

**⚠️ 局限性**

局限性包括：训练时间显著增加（动作缓冲检索消耗较大）；对动作缓冲大小、k、μ 等超参敏感；在某些任务（如 HalfCheetah）即时更新反而略慢；仅在现有 8 个 MuJoCo 环境验证，需进一步验证更大、更复杂或最大熵框架下的效果

---

## 404. Robustness of Approval-Based Multiwinner Voting Rules

**arXiv ID:** 2601.19706 | [PDF](https://arxiv.org/pdf/2601.19706v1)

**作者:** Piotr Faliszewski `[一作]` (AGH University), Bartosz Kusek `[通讯]` (AGH University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5002038095)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了基于认可的多胜选投票规则在面临微小投票扰动（如增删或调换单一认可）时的鲁棒性，定义了鲁棒级别和鲁棒半径，并系统分析了七种常用规则（AV、SAV、CC、PAV、GreedyCC、GreedyPAV、Phragmén）的鲁棒级别、鲁棒半径的判定与计数复杂度。

**💡 创新点**

创新点在于：①将Bredereck等人关于排序投票的鲁棒性框架推广到认可投票；②证明大多数规则的鲁棒级别为常数1或最大委员会大小k；③首次给出多种鲁棒半径判定问题的多项式、NP‑hard与#P‑hard分类；④提出并证明了针对AV与SAV的计数鲁棒半径问题的复杂度，揭示SAV计数问题为#P‑complete；⑤通过构造性证明展示Greedy与Phragmén规则的鲁棒半径为NP‑complete。

**🔧 技术方法**

技术方法主要包括：多项式时间算法（对AV的动态规划计数、SAV的贪心决策）、多项式时间可归约的NP/ #P 归约（从X3C、Perfect‑Matching等经典问题）、动态规划与计数技巧、ILP/参数化算法的简化引用、对规则行为的精细分析（如逐步评估候选人得分变化）。

**📊 数据集**

该工作完全是理论分析，不涉及实验或公开数据集，所有证明均基于构造性实例和抽象数学模型。

**📈 对比分析**

由于本文没有实验评估，性能比较仅体现在理论复杂度的分类上：对AV、SAV的判定问题为P，计数问题对AV为FP，对SAV为#P‑complete；对CC、PAV等NP‑hard规则，鲁棒半径判定问题为NP‑complete；GreedyCC、GreedyPAV与Phragmén的鲁棒半径为NP‑complete；这些结果与现有的投票规则复杂度研究相呼应，但在鲁棒性维度提供了新的视角。

**⚠️ 局限性**

局限性包括：①未给出实验验证，仅提供理论证明；②仅关注认可投票，未探讨排序或分数投票；③鲁棒性度量仅考虑单一认可的增删或调换，未涵盖更复杂扰动；④对SAV的计数问题虽然证明为#P‑complete，但缺乏实际的计数算法；⑤某些规则（如CC、PAV）在计数鲁棒性方面仍未给出完整复杂度分类。

---

## 405. A new Image Similarity Metric for a Perceptual and Transparent Geometric and Chromatic Assessment

**arXiv ID:** 2601.19680 | [PDF](https://arxiv.org/pdf/2601.19680v1)

**作者:** Antonio Di Marino `[一作]` (Institute for High-Performance Computing and Networking National Research Council), Giovanna Sannino `[通讯]` (Institute for High-Performance Computing and Networking National Research Council)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于地球移动距离（EMD）纹理不相似度和Oklab颜色不相似度相结合的全参考图像相似度指标EDOKS，能够在保持可解释性的同时兼顾几何与色彩失真评估。

**💡 创新点**

创新点在于将纹理与色彩两项分离为可解释的低层特征，并通过EMD和欧氏距离在Oklab空间中计算不相似度，再以加权平均方式得到最终指标；同时提供对应的热图解释。

**🔧 技术方法**

技术包括Gabor滤波器族提取纹理特征、Meng–Hee–Heng聚类生成签名、Earth Mover’s Distance计算纹理差异、Oklab色彩空间转换及欧氏距离评估色差、α权重融合与逆变换得到EDOKS。

**📊 数据集**

使用了BAPPS（Berkeley‑Adobe Perceptual Patch Similarity）验证数据集，包含多种传统与深度网络产生的形变、纹理与色彩失真；同时在LIUK4‑v2、RAISE1k等公开图像上测试运行时性能。

**📈 对比分析**

与传统低层指标（PSNR、SSIM、GMSD、VSI等）以及深度学习指标（LPIPS、DISTS、TOPIQ、PIEAPP）在BAPPS 2AFC、JND子集进行相关性（SROCC/KROCC/PLCC）与准确率评估，EDOKS在低层指标上显著优于它们，在深度指标上与之相近；在运行时表现与主流指标相当。

**⚠️ 局限性**

局限性包括：仍需手工调参（如α、patch大小、Gabor参数）以适配不同失真场景；对极端纹理或色彩失真可能单独使用单项不够；计算复杂度受聚类与EMD求解影响，且在高分辨率或大批量应用中可能成为瓶颈。

---

## 406. Single-Winner Voting on Matchings

**arXiv ID:** 2601.19653 | [PDF](https://arxiv.org/pdf/2601.19653v1)

**作者:** Niclas Boehmer `[一作]`, Jessica Dierking `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

由于缺少完整论文内容，无法确定具体研究内容。

**💡 创新点**

无法确定创新点。

**🔧 技术方法**

无法确定使用的技术。

**📊 数据集**

无法确定使用的数据集。

**📈 对比分析**

无法确定比较方法及性能。

**⚠️ 局限性**

无法确定限制。

---

## 407. WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater Reconstruction and Restoration

**arXiv ID:** 2601.19753 | [PDF](https://arxiv.org/pdf/2601.19753v1)

**作者:** Xinrui Zhang `[一作]` (Beihang University), Wenrui Ding `[通讯]` (Beihang University)

**通讯引用:** 1847 | [OpenAlex ID](https://openalex.org/A5024501453)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种纯 3D Gaussian Splatting（3DGS）框架 WaterClear-GS，用以在水下环境中实现三维重建与图像恢复，直接在 Gaussian 原语中嵌入光衰减、散射与蔚蓝光等水下光学参数；

**💡 创新点**

创新点在于：1) 直接将光学模型编码进 Gaussian 原语，无需额外网络；2) 双分支优化同时监督水下与无水渲染，实现几何一致性与色彩恢复；3) 引入深度引导几何正则、感知驱动图像损失、曝光约束、空间自适应正则及物理引导光谱正则，提升恢复质量与鲁棒性；

**🔧 技术方法**

使用 3DGS 作为基础，结合光学可见光衰减模型（β^D、β^B、B），双分支渲染、深度相关正则、感知重加权损失、曝光约束、空间平滑与软光谱先验；

**📊 数据集**

在公开水下数据集 SeaThru-NeRF（包括 Panama、Curasao、IUI3-RedSea、JapaneseGardens-RedSea）、D3/D5，及自采 ShipWreck（两座船体残骸场景）上进行训练与评估；

**📈 对比分析**

与 3DGS、SeaThru-NeRF、WaterSplatting、SeaSplat 等基线对比，在 NVS 任务上 PSNR、SSIM 领先并保持 160+ FPS；在 UIR 任务上 E_00、ψ̅ 指标最优；整体实现高质量视角合成与颜色恢复，同时保持实时渲染；

**⚠️ 局限性**

局限在于仅支持静态场景，无法处理动态物体或光照变化；在极端浑浊环境下色彩恢复仍受限，需进一步改进光学模型与鲁棒性。

---

## 408. SCOPE: Smooth Convex Optimization for Planned Evolution of Deformable Linear Objects

**arXiv ID:** 2601.19742 | [PDF](https://arxiv.org/pdf/2601.19742v1)

**作者:** Ali Jnadi `[一作]` (Innopolis University), Karam Almaghout `[通讯]` (Innopolis University)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5028294725)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出并实现了SCOPE框架，用于在平面上快速计算可变形线性物体（如电缆、绳索）从起始形状到目标形状的平滑变形轨迹。

**💡 创新点**

创新点在于将传统能量最小化问题替换为凸优化，通过约束伸长和二次平滑成本实现快速求解，同时通过中间引导点保证轨迹合理性，获得了显著的计算速度提升。

**🔧 技术方法**

采用的技术包括凸二次优化（使用CVX建模）、等距段长度约束、平滑成本、引导点惩罚以及对时间步的联合求解。

**📊 数据集**

实验使用了人工构造的二维形状转换数据集，包括四种目标形状（四分之一正弦波、半正弦波、U形、S形、I形、L形）以及对应的初始形状，所有实验在MATLAB 2024a上运行。

**📈 对比分析**

与传统能量基方法比较，SCOPE在解决时间上提高了约8到47倍（例如从22.73秒降至2.74秒），但最大形状误差在2–5.2 cm之间，传统方法误差则低于0.1 cm；因此SCOPE在速度上优越，准确性略逊。

**⚠️ 局限性**

主要局限在于牺牲了一定的精度，尤其在高曲率目标形状时误差显著；此外，该方法仍为二维平面模型，尚未验证对三维复杂几何或真实硬件的适用性。

---

## 409. Future of Software Engineering Research: The SIGSOFT Perspective

**arXiv ID:** 2601.19731 | [PDF](https://arxiv.org/pdf/2601.19731v1)

**作者:** Massimiliano Di Penta `[一作]` (University of Sannio), Thomas Zimmermann `[通讯]` (University of California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对ICSE 2026未来软件工程研讨会的前测问卷进行主题分析，评估软件工程社区的优点与问题，并基于调查结果提出SIGSOFT可采取的改进措施，如提高会议资金透明度、实验混合海报展示、扩大对全球欠发达地区的支持等。

**💡 创新点**

创新点在于系统性地将社区成员的体验与建议转化为可操作的治理与组织策略，提出以社区建设为核心的会议模式改进，并将混合海报展示技术作为创新实践方案。

**🔧 技术方法**

主要使用定性主题分析方法（在线电子表格编码）进行问卷数据处理；技术实现上涉及混合海报展示的实验方案（如Microsoft Teams电视站与Meta Horizons VR站）。

**📊 数据集**

使用的主要数据集为ICSE 2026 Future of Software Engineering Workshop的前测问卷答案（约90条针对社区治理与参与度的回应）。

**📈 对比分析**

本文并未进行传统算法性能比较；评估方法主要是通过对调查结果的统计与主题归纳来阐述改进建议的可行性与潜在影响。

**⚠️ 局限性**

局限性包括：① 仅基于单次问卷，样本规模有限；② 主题归纳主观性高，缺乏量化验证；③ 对混合海报展示技术的实验仅在两场会议中进行，缺乏大规模、跨文化的验证；④ 讨论的改进措施多为建议性质，缺乏实际落地的成本与效果评估。

---

## 410. How Similar Are Two Elections?

**arXiv ID:** 2601.19716 | [PDF](https://arxiv.org/pdf/2601.19716v1)

**作者:** Piotr Faliszewski `[一作]` (AGH University), Nimrod Talmon `[通讯]` (Ben-Gurion University)

**通讯引用:** 1636 | [OpenAlex ID](https://openalex.org/A5064260879)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出并研究了“同构距离”（isomorphic distances）——一种在选举中对候选人和选民重命名保持不变的距离度量，并系统地分析了该距离的计算复杂性、可逼近性以及参数化可解性。

**💡 创新点**

创新点在于：
- 定义了可确保同构选举距离为零的新度量；
- 证明了在使用交换（swap）和 Spearman 距离时的同构距离为 NP‑hard（甚至 #P‑complete）并且几乎无法有效逼近；
- 给出针对候选人数、选民数以及距离值的 FPT 算法，拓展了已知的 Kemeny 分数等问题的参数化视角。

**🔧 技术方法**

主要技术手段：
- 通过对 Kemeny 分数、图同构等经典 NP‑hard 问题的多项式约简；
- 采用匹配与最小割、完美匹配等图算法求解特定子问题；
- 构造精细的投票与候选人集合，利用假设性候选人匹配实现逼近与 FPT 分析；
- 利用交换距离与 Spearman 距离的组合性质和三角不等式证明可扩展性。

**📊 数据集**

本研究未进行实验验证，也未使用公开数据集；它主要是理论研究，文中仅引用了 PrefLib 等已知的选举数据集作为潜在应用场景的参考。

**📈 对比分析**

由于该距离在理论上与图同构紧密相关，实验性比较并未给出；相较于在 map‑of‑elections 框架中使用的更简单的距离（如离散距离），同构距离在可解释性上更强，但计算成本极高，实际性能（时间复杂度）远远超出可接受范围。

**⚠️ 局限性**

局限性：
- 交换和 Spearman 同构距离计算为 NP‑hard，逼近性极差（几乎无法得到多项式因子逼近）；
- 即使在参数化模型下，给定的 FPT 算法仍然具有指数级的时间复杂度，难以在实际规模上使用；
- 对于实际选举数据集（如 PrefLib）的实验验证尚缺失，尚未证明在现实场景中的可行性。

---

## 411. Out-of-Distribution Generalization via Invariant Trajectories for Multimodal Large Language Model Editing

**arXiv ID:** 2601.19700 | [PDF](https://arxiv.org/pdf/2601.19700v1)

**作者:** Jiajie Su `[一作]` (Zhejiang University), Chaochao Chen `[通讯]` (Zhejiang University)

**通讯引用:** 6242 | [OpenAlex ID](https://openalex.org/A5028791879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多模态大型语言模型的知识编辑，提出一种基于 OOD 泛化的增量编辑框架，能够在不同跨模态提示下保持可靠性、局部性和泛化性。

**💡 创新点**

创新点包括：① 将编辑问题视为三类 OOD 风险（可靠性、局部性、泛化性）；② 采用可插拔的 invariant learning 机制，结合 IRM 与 TV 正则化实现编辑轨迹的稳健性；③ 通过最大均值差距 (MMD) 对语义邻域进行分布级对齐，抑制因果过拟合与欠拟合。

**🔧 技术方法**

技术手段主要包括：Invariant Risk Minimization (IRM)、总变分 (Total Variation) 正则化、最大均值差距 (MMD) 迁移学习、KL 散度约束、梯度对抗优化与多尺度 RBF 核对齐。

**📊 数据集**

使用 MMEdit 基准数据集，涵盖两项任务：E‑VQA（视觉问答编辑）和 E‑IC（图像字幕编辑），在 BLIP2‑OPT 与 MiniGPT‑4 两大多模态 LLM 上进行实验。

**📈 对比分析**

与 Naïve FT、SERAC、IKE、WISE、T‑Patcher、UniKE 等现有编辑方法相比，本文框架在 Reliability、Generality、Text‑Locality 与 Image‑Locality 四项指标上均取得显著提升，尤其在长期编辑（T=5/10）场景下能有效抑制性能衰减。

**⚠️ 局限性**

局限性：① 需要额外的 IRM‑TV 网络，导致显存和训练时间略增；② 对超参数（如 λ、学习率、层数）的敏感性需要进一步自适应；③ 目前仅在 MMEdit 基准上验证，缺乏跨域或更大规模数据的泛化评估。

---

## 412. AlignCoder: Aligning Retrieval with Target Intent for Repository-Level Code Completion

**arXiv ID:** 2601.19697 | [PDF](https://arxiv.org/pdf/2601.19697v1)

**作者:** Tianyue Jiang `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33227 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向代码仓库的自动补全框架AlignCoder。

**💡 创新点**

创新点在于通过多样本查询增强机制弥合查询与目标代码语义鸿沟，并利用强化学习训练检索器。

**🔧 技术方法**

采用检索增强生成（RAG）、多重采样、强化学习和vLLM加速推理等技术。

**📊 数据集**

使用CrossCodeEval和RepoEval两个公开基准，以及自建的10k个Python/Java GitHub仓库做训练。

**📈 对比分析**

与ReACC、RepoCoder、RLCoder等方法对比，平均提升18.1% EM分数，并在多模型、多语言上保持优异表现。

**⚠️ 局限性**

主要局限在于对采样次数、温度/Top-p的敏感性以及对检索库构建方式的依赖。

---

## 413. Self-Supervised Weight Templates for Scalable Vision Model Initialization

**arXiv ID:** 2601.19694 | [PDF](https://arxiv.org/pdf/2601.19694v1)

**作者:** Yucheng Xie `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**通讯引用:** 19012 | [OpenAlex ID](https://openalex.org/A5018128720)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在大规模视觉模型中提出一种自监督权重模板学习框架 SWEET，用以实现可伸缩、跨任务的模型初始化。

**💡 创新点**

创新点包括：① 采用 Tucker‑based 低秩分解约束生成共享权重模板；② 在预训练时加入宽度随机缩放正则化，提升宽度泛化能力；③ 通过自监督 MAE 目标，消除对特定任务的依赖，使模板更通用。

**🔧 技术方法**

主要技术手段有：自监督 MAE 预训练、Tucker 分解与低秩约束、宽度随机缩放 (dropout on scalers)、SwiGLU、RMSNorm、RoPE 等结构改进。

**📊 数据集**

实验使用 ImageNet‑1K 进行自监督预训练，随后在 ImageNet‑1K（分类/生成）、COCO（检测）和 ADE20K（语义分割）等标准视觉任务上进行评估。

**📈 对比分析**

与 WT‑Select、DMAE、IsoPruning、WAVE 等可伸缩初始化方法对比，SWEET 在分类、检测、分割与生成任务上平均提升 1.6% Top‑1、+2.04 AP、+2.76 mIoU、+2.19 FID，整体表现显著优于现有方法。

**⚠️ 局限性**

局限性：模板尺寸仍受低秩约束限制，宽度随机缩放可能无法覆盖所有宽度配置；在极小模型或特定任务（如细粒度分类）下迁移效果仍有待进一步提升。

---

## 414. DSVM-UNet : Enhancing VM-UNet with Dual Self-distillation for Medical Image Segmentation

**arXiv ID:** 2601.19690 | [PDF](https://arxiv.org/pdf/2601.19690v1)

**作者:** Renrong Shao `[一作]` (Naval Medical University), Lulu Zhang `[通讯]` (Naval Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在Vision Mamba UNet框架中加入双重自蒸馏技术（Projection Self‑Distillation + Progressive Self‑Distillation），无需改动网络结构即可提升医学图像分割性能。

**💡 创新点**

创新点在于提出了双重自蒸馏框架，分别在全局和局部层面对多层特征进行对齐，从而弥补浅层特征学习能力不足，并通过投影与逐层蒸馏实现高效知识迁移。

**🔧 技术方法**

采用Vision Mamba、UNet结构、State‑Space Model、线性投影、1×1卷积、双线性插值、自蒸馏损失（MSE）、BceDice/CE‑Dice、AdamW优化器、CosineAnnealingLR学习率调度等技术。

**📊 数据集**

在ISIC2017、ISIC2018（皮肤病变分割）以及Synapse（多器官分割）三大医学图像分割基准上进行实验。

**📈 对比分析**

与多种SOTA模型（UNet、UTNetV2、TransFuse、MALUNet、VM‑UNet系列、Transformer/混合模型等）对比，DSVM‑UNet在ISIC17/18的数据集上mIoU、DSC、ACC、SPE、SEN等指标均略优于VM‑UNetV2；在Synapse数据集上DSC提升0.47%，HD95下降1.32%，并在各器官类别上取得最佳或近最佳表现。

**⚠️ 局限性**

限制包括：实验仅覆盖三种医学图像数据集，缺乏跨模态或更大规模数据的泛化验证；自蒸馏引入额外的训练开销与实现复杂度；模型对蒸馏权重α、β等超参数较为敏感，需要进一步自动调优或理论分析。

---

## 415. LLM-Assisted Authentication and Fraud Detection

**arXiv ID:** 2601.19684 | [PDF](https://arxiv.org/pdf/2601.19684v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 416. Grasynda: Graph-based Synthetic Time Series Generation

**arXiv ID:** 2601.19668 | [PDF](https://arxiv.org/pdf/2601.19668v1)

**作者:** Luis Amorim `[一作]` (University of Minho), Vitor Cerqueira `[通讯]` (Fraunhofer Portugal AICOS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于图的合成时间序列生成方法（Grasynda），通过将单变量时间序列离散化为有限状态，构造有向图及其转移概率矩阵，再按此矩阵采样得到合成序列，用于数据增强提升预测模型性能。

**💡 创新点**

创新点在于：①把时间序列映射为有向图并利用转移概率矩阵生成新序列，能够同时捕捉局部转移和全局结构；②采用量化图（等频分箱）实现离散化；③与传统变换或GAN不同，生成过程更高效且不易出现模式崩溃；④结合STL分解处理非平稳，提升生成序列的现实性。

**🔧 技术方法**

技术包括：时间序列离散化（等频分箱/量化图）、有向图构造、转移概率矩阵统计、基于概率的随机采样生成合成序列、离散值到连续值的恢复；评估使用NHITS、KAN、MLP三种神经网络预测模型；性能评估采用MASE；与多种传统增强方法（Jittering、Magnitude Warping、Time Warping、DTW平均、Seasonal MB bootstrap、Chronos等）比较。

**📊 数据集**

使用了公开的六个基准数据集：M1（Monthly 与 Quarterly）、M3（Monthly 与 Quarterly）以及 Tourism（Monthly 与 Quarterly），共计3,797个时间序列，409,602条观测。

**📈 对比分析**

通过在每个数据集上使用三种预测模型（NHITS、KAN、MLP）进行实验，对比无增强基线和多种增强方法，采用MASE衡量。Grayscale 在18个实验中有72% 的案例优于无增强基线，平均MASE 在两种模型中最低，平均rank 其次最佳，并在统计检验中多次显著优于其他方法。

**⚠️ 局限性**

局限性：仅针对单变量序列，未处理多变量和长程依赖；离散化采用简单等频分箱，可能限制生成多样性；需要先进行STL分解以处理非平稳，增加预处理步骤；对不同领域的泛化能力和大规模序列的可扩展性尚未充分验证。

---

## 417. Robustness of Constraint Automata for Description Logics with Concrete Domains

**arXiv ID:** 2601.19644 | [PDF](https://arxiv.org/pdf/2601.19644v1)

**作者:** Stéphane Demri `[一作]` (Paris Saclay University), Tianwen Gu `[通讯]` (Paris Saclay University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种基于自动机的方法来解决具有具体域的描述逻辑的一致性问题，展示了如何通过增强转移来引入符号约束，从而达到最优的上界。

**💡 创新点**

创新点在于引入了一类约束自动机，这些自动机能够接受无限数据树，并且允许兄弟节点之间的约束，从而扩展了现有的约束自动机模型。

**🔧 技术方法**

使用了基于自动机的决策程序，特别是Büchi树自动机，并通过构造约束自动机来处理一致性问题。

**📊 数据集**

使用了满足特定条件的具体域，这些条件包括完成性、合并性和同态ω-紧性等。

**📈 对比分析**

通过将一致性问题归约到非空性问题，展示了该方法的有效性，并且在参数化复杂性分析中，证明了该方法在特定条件下的复杂性为k-成员。

**⚠️ 局限性**

限制在于对具体域的条件要求较高，特别是完成性和合并性等条件可能限制了方法的适用范围，且对某些具体域的复杂性和可判定性仍需进一步研究。

---

## 418. Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow

**arXiv ID:** 2601.19707 | [PDF](https://arxiv.org/pdf/2601.19707v1)

**作者:** Yunyue Wei `[一作]` (Tsinghua University), Yanan Sui `[通讯]` (Tsinghua University)

**通讯引用:** 1121 | [OpenAlex ID](https://openalex.org/A5069290448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在高维连续控制任务中提出了基于价值引导的流式探索方法 Q‑guided Flow Exploration，实现了在原始高维动作空间中可扩展的在线强化学习。

**💡 创新点**

创新点在于将学习到的 Q 函数梯度用于构造概率流，形成价值对齐的有向探索，并给出政策改进的理论保证。

**🔧 技术方法**

采用 Actor‑Critic 框架、流匹配（Flow Matching）与梯度上升的概率流、以及批量归一化 Q 网络等技术。

**📊 数据集**

在多种高维机器人/仿真基准（SMPL Humanoid‑Jump、Unitree H1、MyoHand、MyoLeg、Ostrich）以及 700 肌肉全身人类模型（MS‑Human‑700）上进行实验。

**📈 对比分析**

与高斯噪声、SAC、SDAC、DACER、QSM、DynSyn、Lattice、DEP‑RL 等基线相比，Q‑guided Flow Exploration 在学习曲线和收敛速度上均表现优越，尤其在高维/过度驱动场景下提升显著。

**⚠️ 局限性**

局限在于需要手动调参（梯度步数、步长、欧拉步长）且对梯度质量敏感，且目前尚未在更大规模或真实物理系统上验证。

---

## 419. Convex Hull 3D Filtering with GPU Ray Tracing and Tensor Cores

**arXiv ID:** 2601.19647 | [PDF](https://arxiv.org/pdf/2601.19647v1)

**作者:** Roberto Carrasco `[一作]` (Universidad de Chile), Nancy Hitschfeld `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一种基于RTX Ray Tracing和Tensor Core的3D凸包预处理过滤器，利用Manhattan距离构建24面多面体并在GPU上并行判定点是否在多面体内部，随后用Tensor Core加速前缀和压缩，最后把剩余候选点交给传统凸包算法；

**💡 创新点**

创新点在于：①将凸包过滤问题重新建模为Ray Tracing交点检测；②首次使用RT核心对固定多面体做点内外判断；③结合Tensor Core实现高效并行扫描压缩；④在均匀与球面分布上实现高达200×的加速与显著的能耗降低；

**🔧 技术方法**

使用技术包括：NVIDIA CUDA、OptiX RT核心、Tensor Core MMA、并行Min/Max归约、BVH加速、并行前缀和（scan）以及ParGeo库的Pseudohull实现；

**📊 数据集**

实验数据集为随机均匀分布和通过ρ参数控制厚度的球面分布，规模从2^23到2^28点；

**📈 对比分析**

与20核CPU的Pseudohull过滤器以及仅使用CUDA+Tensor Core的过滤器进行对比，评估过滤时间、整体凸包时间和能耗；在均匀分布下RTX+RT滤波器相较CPU提升≈210×，在球面分布下几乎无差异；GPU滤波器能耗比CPU低约75×；在不同GPU架构（Ampere、Lovelace、Blackwell）上验证可扩展性；

**⚠️ 局限性**

局限性包括：在球面分布（几乎所有点是凸包表面）下过滤效果差；过滤多面体固定为24面，难以进一步提高过滤率；RT核心在Ampere等旧架构缺乏硬件加速；BVH构造时间对大规模点集仍是瓶颈；未实现递归BVH更新或动态多面体扩展。

---

## 420. Provable Learning of Random Hierarchy Models and Hierarchical Shallow-to-Deep Chaining

**arXiv ID:** 2601.19756 | [PDF](https://arxiv.org/pdf/2601.19756v1)

**作者:** Yunwei Ren `[一作]` (Princeton University), Jason D. Lee `[通讯]` (University of California)

**通讯引用:** 6582 | [OpenAlex ID](https://openalex.org/A5059740024)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

证明深度卷积网络能够在随机层级模型（RHM）上高效学习，并给出了样本复杂度上界 O(m^(1+o(1))L)

**💡 创新点**

首次在理论层面证明深度网络相较于浅层网络具有指数级样本复杂度优势；提出可通过逐层梯度下降训练的通用框架，并在此框架下实现层级学习

**🔧 技术方法**

利用随机 Fourier 特征（RBF 随机特征）、层级结构分析、梯度下降、强凸回归、误差传播与随机上下文无关文法理论等技术

**📊 数据集**

无真实数据集，仅在理论构造的随机层级模型上进行分析

**📈 对比分析**

与浅层网络（需要 Ω(m^{s^L}) 样本）的理论比较，证明深层网络的样本复杂度为 O(m^L)，与先前经验推测一致；未给出实验性能数据

**⚠️ 局限性**

适用范围局限于已知分支因子、非歧义且满足均匀性与非退化性的 RHM；训练仅使用首个 patch；未验证对真实数据的鲁棒性或对分支因子未知情形的适用性

---

## 421. The Effect of Architecture During Continual Learning

**arXiv ID:** 2601.19766 | [PDF](https://arxiv.org/pdf/2601.19766v1)

**作者:** Allyson Hahn `[一作]` (Argonne National Laboratory), Krishnan Raghavan `[通讯]` (Argonne National Laboratory)

**通讯引用:** 8375 | [OpenAlex ID](https://openalex.org/A5030055383)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种统一的Sobolev空间框架，既对模型架构又对权重进行联合学习，以缓解连续学习中的灾难性遗忘问题。

**💡 创新点**

创新点在于：① 将架构和权重视为可学习的函数，证明仅更新权重不足以保持模型容量；② 通过二层优化（上层搜索最优架构，下层动态规划学习权重）实现架构随任务自适应；③ 提出低秩转移（A·Bᵀ）机制，在不同维度的参数空间间高效迁移知识；④ 通过理论分析阐明“可持续学习”的绝对连续性条件。

**🔧 技术方法**

核心技术包括：Sobolev空间的弱导数理论、二层优化框架、邻域方向直接搜索（NDDS）用于离散架构搜索、低秩A·Bᵀ转移矩阵、以及传统的经验回放/自适应梯度等常用CL算法。

**📊 数据集**

实验数据集：① 随机正弦回归（可控分布漂移）；② MNIST数字分类（任务拆分为数对或单数字）；③ 生成式图数据集（图大小10个节点、5类，逐步加入噪声、边丢弃、特征偏移）。

**📈 对比分析**

比较方法：四种设置——基线（固定架构）、启发式（学习率衰减+warm‑up+梯度加权）、仅架构搜索、架构+低秩迁移（C4）。在回归、图像和图结构任务中，C4 在平均误差/准确率、回溯传递 (BWT) 与前向遗忘 (FWT) 上均明显优于其他三种方法，误差可降低至原来的 30%–40%，遗忘率显著下降，且对噪声鲁棒性更强。

**⚠️ 局限性**

局限性：① 架构搜索过程不稳定，需多次迭代；② 低秩迁移对训练轮数敏感，需要额外调参；③ 目前仅探讨节点数/过滤器尺寸等离散参数，未覆盖更复杂的架构变动（如分支、残差块）；④ 实验规模受限于小数据集，未验证在大规模预训练模型上的可扩展性；⑤ 计算开销相较纯权重更新仍较高。

---

## 422. Veri-Sure: A Contract-Aware Multi-Agent Framework with Temporal Tracing and Formal Verification for Correct RTL Code Generation

**arXiv ID:** 2601.19747 | [PDF](https://arxiv.org/pdf/2601.19747v1)

**作者:** Jiale Liu `[一作]` (University of Edinburgh), Tianqi Jiang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套名为Veri‑Sure的多代理框架，用自然语言合约驱动RTL代码生成、验证和局部修复；

**💡 创新点**

创新点包括：设计合约统一意图、基于波形追踪与静态依赖切片的局部补丁机制、以及结合模拟与形式验证的多分支调试流程；

**🔧 技术方法**

核心技术为：LLM代理、设计合约、Trace‑driven temporal analysis、static dependency slicing、dependency‑slicing‑guided patching、SystemVerilog assertion、Boolean equivalence proof（SymbiYosys + Z3）、Verilator等；

**📊 数据集**

使用的数据集为VerilogEval‑v2‑EXT，该基准在原始VerilogEval‑v2基础上新增53个工业级任务，合计209个任务，并按难度分层；

**📈 对比分析**

与15种单体LLM、单代理模拟反馈、以及MAGE、VerilogCoder等多代理基线比较，Veri‑Sure在VerilogEval‑v2‑EXT上功能Pass@1达93.30%，在Hard子集更是85.07%，显著优于其他方法；

**⚠️ 局限性**

局限性在于仍依赖大规模LLM，补丁过程需要代理间复杂通信，极端时序边界与极大规模模块的可扩展性尚待验证。

---

## 423. Component-Level Lesioning of Language Models Reveals Clinically Aligned Aphasia Phenotypes

**arXiv ID:** 2601.19723 | [PDF](https://arxiv.org/pdf/2601.19723v1)

**作者:** Yifan Wang `[一作]` (University of Manchester), Shaonan Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在大型语言模型的功能组件（MoE专家或稠密层神经元）层面有针对性地损伤，模拟Broca型和Wernicke型失语症的语言缺陷，并用Western Aphasia Battery（WAB）和Aphasia Quotient（AQ）进行临床级评估。

**💡 创新点**

创新点在于：①提出一种基于临床失语亚型的组件级损伤框架，将模型专家/神经元与失语症子型关联；②在Mixture‑of‑Experts和稠密Transformer上统一实现该框架；③采用临床标准WAB/AQ作为定量评估指标，突破了以往仅用通用NLP指标的局限。

**🔧 技术方法**

技术手段包括：Mixture‑of‑Experts（OLMoE）与稠密Transformer（OLMo）模型；BLiMP零消融进行功能归因；AphasiaBank细调获取亚型关联单位；Xavier重初始化或激活零置换进行损伤；WAB子测试与AQ合成进行临床评估。

**📊 数据集**

使用的数据集包括：BLiMP（语言现象归因），AphasiaBank（Broca与Wernicke子集），Comparative Aphasia Project（CAP，用于子型验证），以及Western Aphasia Battery（WAB）用于最终评估。

**📈 对比分析**

通过与同样规模的随机损伤对照以及在两种架构下逐步增大损伤比例，对比AQ下降曲线。结果显示：①目标损伤导致的AQ下降比随机损伤更显著；②MoE模型的损伤更局部、可解释，呈现更清晰的失语亚型映射；③稠密模型虽然也出现下降，但更分散、缺乏明显亚型差异；总体性能在AQ上呈递增下降，验证了方法的可行性。

**⚠️ 局限性**

局限性：仅在文本层面进行评估，缺乏多模态或真实临床交互；损伤规模有限，可能无法完全映射生物大脑损伤的复杂性；实验只覆盖两种模型规模与架构，未验证更大或不同语言模型的泛化；未涉及失语症其他亚型（如失语症的混合型等）的适用性。

---

## 424. DiffStyle3D: Consistent 3D Gaussian Stylization via Attention Optimization

**arXiv ID:** 2601.19717 | [PDF](https://arxiv.org/pdf/2601.19717v1)

**作者:** Yitong Yang `[一作]` (Shanghai University of Finance and Economics), Shuting He `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 855 | [OpenAlex ID](https://openalex.org/A5085200012)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出 DiffStyle3D，利用扩散模型潜空间直接对 3D Gaussian Splatting 场景进行风格迁移，实现高质量的 3D 内容风格化。

**💡 创新点**

创新点包括：① 引入 Attention‑Aware Loss 通过自注意力中的键值投射实现风格与内容对齐；② 设计 Geometry‑Guided Multi‑View Consistency，将几何投影信息融入自注意力以建模跨视角一致性；③ 添加几何感知掩码避免重叠视角的冗余优化。

**🔧 技术方法**

采用 Stable Diffusion 1.5 作为基准，利用 VAE 编码、UNet 解码与自注意力机制，配合固定时间步 t=1 的潜空间优化；通过双线性投影和可见性掩码实现几何引导的注意力。

**📊 数据集**

实验数据集包括 Tandt DB、Mip‑NeRF 360 场景（共 8 场景）以及通过 SAM3D 提取的 10 个对象，总计 112 场景级和 140 对象级风格迁移实验。

**📈 对比分析**

与 VGG、CLIP 和其他扩散风格迁移方法对比，使用 CLIP‑S/C/F、FID、S_vgg、LPIPS、RMSE 等指标评估，DiffStyle3D 在风格质量、内容保持和多视角一致性上均取得最优或接近最优成绩，训练时间与 CLIPGaussian 相当。

**⚠️ 局限性**

局限性在于仅优化颜色参数，几何保持不变；固定时间步可能限制风格细节的表现；在极端风格或极大场景下仍可能出现轻微一致性不足或收敛缓慢，且实验仅在单张 48G GPU 上完成。

---

## 425. Hyperbolic Additive Margin Softmax with Hierarchical Information for Speaker Verification

**arXiv ID:** 2601.19709 | [PDF](https://arxiv.org/pdf/2601.19709v1)

**作者:** Zhihua Fang `[一作]` (Xinjiang University), Liang He `[通讯]` (Tsinghua University)

**通讯引用:** 7909 | [OpenAlex ID](https://openalex.org/A5062604912)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在说话人验证任务中，本文提出利用双曲空间的Softmax与带加性间距的Softmax，对说话人嵌入进行分类学习。

**💡 创新点**

创新点在于将双曲几何（Poincaré球）引入说话人嵌入学习，用负双曲距离替代欧氏余弦相似度，并在此基础上加入间距约束，实现对层次信息的建模与类别分离。

**🔧 技术方法**

采用了双曲空间的Poincaré球模型、双曲距离计算、负双曲距离作为logits，以及添加间距的H‑Softmax/HAM‑Softmax损失。

**📊 数据集**

使用了 VoxCeleb1、VoxCeleb2 和 CNCeleb 三个公开说话人数据集进行训练与评估。

**📈 对比分析**

与传统的Softmax、AM‑Softmax、AAM‑Softmax、Real‑AM‑Softmax 等基线比较，H‑Softmax 在所有数据集上平均降低 EER 27.84%，HAM‑Softmax 则相对 AM‑Softmax 降低 14.23%，并多次取得最优或次优结果。

**⚠️ 局限性**

局限性包括对曲率、间距、缩放因子等超参数需要细致调优，过大曲率或间距会导致数值不稳定或过度分离；同时双曲空间的额外映射与距离计算略增计算成本。

---

## 426. RHSIA: Real-time Hemodynamics Surrogation for Non-idealized Intracranial Aneurysms

**arXiv ID:** 2601.19876 | [PDF](https://arxiv.org/pdf/2601.19876v1)

**作者:** Yiying Sheng `[一作]` (National University of Singapore), Choon Hwai Yap `[通讯]` (Imperial College London)

**通讯引用:** 2234 | [OpenAlex ID](https://openalex.org/A5061269469)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于图变压器的深度学习模型，可实时从颅内动脉瘤表面网格及入口波形预测整个心搏周期的壁剪切应力（WSS），并通过静态CFD数据增强提高模型在短暂脉动CFD数据稀缺时的性能；

**💡 创新点**

创新点包括：1）使用图谐波变形（GHD）对网格进行统一、拓扑保持的下采样和编码，实现高效、可扩展的几何表示；2）将静态CFD数据作为无监督增强，显著提升脉动预测精度；3）将时间波形信息通过1D卷积编码并在每个GPS模块中注入，实现时空耦合；

**🔧 技术方法**

采用Graph Transformer（GPS）架构，融合GHD和cotangent拉普拉斯特征的全局自注意力，并利用1D卷积U-Net处理波形；

**📊 数据集**

使用AneuG-Flow数据集，包括14,000个静态CFD案例和808个脉动CFD案例；

**📈 对比分析**

与多种基线（Graph U‑Net、LaB‑GATr/VATr、序列U‑Net、谱模态模型）比较，加入静态数据增强后，所提Transformer在MSE、rL2*、SSIM等指标上均优于对照组，rL2*降至2.84%，SSIM提升至0.982；

**⚠️ 局限性**

局限性包括：1）脉动数据仅采用固定入口波形，缺乏多种生理条件；2）模型受限于CFD假设（刚壁、简化血液粘性）；3）需额外预处理才能将临床影像生成网格，影响临床直接部署。

---

## 427. Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection

**arXiv ID:** 2601.19871 | [PDF](https://arxiv.org/pdf/2601.19871v1)

**作者:** Nicholas Cheng `[一作]` `[通讯]` (Independent Researcher), Nicholas Cheng (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于 LLM 的反射式翻译框架，利用模型自评自纠提升低资源语言翻译质量。

**💡 创新点**

创新点是将自我反思结构化为多轮提示并通过关键词掩码实现无细调的翻译改进。

**🔧 技术方法**

采用 GPT‑3.5、Claude Haiku 3.5 LLM，结构化提示、错误识别、关键词掩码和阈值筛选技术。

**📊 数据集**

使用 OPUS‑100 与 NTREX‑African 英文‑isiZulu/isiXhosa 句对。

**📈 对比分析**

通过 BLEU、COMET 评估，第二轮翻译相较第一轮均显著提升，尤其是 COMET。

**⚠️ 局限性**

仅评估两种语言与两种模型，缺乏人类评测，未验证更广泛语言/模型适用性。

---

## 428. Bandits in Flux: Adversarial Constraints in Dynamic Environments

**arXiv ID:** 2601.19867 | [PDF](https://arxiv.org/pdf/2601.19867v1)

**作者:** Tareq Si Salem `[一作]` `[通讯]` (Huawei Technologies France), Tareq Si Salem (Huawei Technologies France)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于原始‑对偶框架的在线镜像下降算法，用于处理具有时间变软约束的对抗性多臂赌博机问题；

**💡 创新点**

创新点在于将镜像下降与梯度估计相结合，设计了能在单点反馈下控制估计方差的机制，并给出了动态最优的 regret 与约束违约量上界；

**🔧 技术方法**

核心技术是在线镜像下降（entropic 负熵镜像映射）与自适应对偶更新，配合一次性梯度估计实现单臂采样；

**📊 数据集**

实验使用自定义的非平稳环境（25支手臂，成本与约束函数周期性移动并加入噪声）作为数据集；

**📈 对比分析**

与基于高斯过程的 UCB‑式方法以及其它基线方法比较，实验结果表明新算法在累计成本和约束违约上均明显优于对手，达到理论预测的子线性上界；

**⚠️ 局限性**

局限性包括：需要对梯度估计进行参数调优；实验仅在合成数据上验证，未在真实世界数据上测试；算法目前仅适用于单臂赌博机，尚未推广至组合式或子模最大化等更复杂的带约束场景。

---

## 429. Calibration without Ground Truth

**arXiv ID:** 2601.19862 | [PDF](https://arxiv.org/pdf/2601.19862v1)

**作者:** Yuqing Kong `[一作]` (Peking University), Yifan Wu `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用标签无监督的后处理框架，通过对强模型输出进行Bregman投影，以已知的弱但良好校准的参考模型为准则，提升强模型的校准和预测性能。

**💡 创新点**

创新点在于：① 引入互相校准（mutual calibration）概念，确定何时能实现严格提升；② 将提升问题转化为投影到参考兼容集合的Bregman投影，提供严格的最坏情况损失下降保证；③ 将方法扩展到可测度属性（elicitable properties），兼顾概率分布与分类+置信度的输出。

**🔧 技术方法**

核心技术包括：Bregman投影、对齐约束的线性可行域、互相校准判定、基于正则化的凸优化、对可测度属性的级集映射与损失重载。

**📊 数据集**

使用公开的LLM（Qwen3‑8B、Llama‑3.1‑8B、Mistral‑3‑8B）在多选题数据集MMLU‑Redux和CommonsenseQA上进行实验，基准为对应的Base模型与Instruct模型。

**📈 对比分析**

与监督后处理（Temperature Scaling）对比，标签无监督方法在Brier Score、ECE、Confidence Loss等指标上实现与或优于监督基线，同时保持甚至略微提升准确率，尤其在CommonsenseQA的ECE降至0.0295。

**⚠️ 局限性**

局限性在于：① 依赖参考模型的良好校准，若参考模型误差显著会削弱保证；② 需要足够的无标签样本来估计两个模型的联合分布；③ 目前实现针对单一适当损失，跨损失通用性尚未完全解决。

---

## 430. Estimating Trust in Human-Robot Collaboration through Behavioral Indicators and Explainability

**arXiv ID:** 2601.19856 | [PDF](https://arxiv.org/pdf/2601.19856v1)

**作者:** Giulio Campagna `[一作]` (Aalborg University), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于行为指标的实时信任估计框架，利用PBO获取的操作员偏好标签训练机器学习模型预测人机协作中的信任水平。

**💡 创新点**

创新点在于整合人类与机器人行为指标，采用SHAP可解释性揭示各指标对信任的影响，并实现对交互参数（执行时间、分离距离、高度）的自适应优化。

**🔧 技术方法**

使用随机森林、KNN、SVM及投票分类器进行分类，结合SHAP解释；数据预处理包括特征归一化、相关性分析、XGBoost特征重要性、数据增强与SMOTE。

**📊 数据集**

实验数据来自14名受试者在化工混合任务中的行为记录（头部轨迹、机器人轨迹），共计840条样本（含合成样本）。

**📈 对比分析**

与单模型比较，投票分类器准确率84.07%，AUC 0.90，优于随机森林（80.09%）、KNN（80.97%）和SVM（80.02%），同时提供较高的精确率、召回率和F1分数。

**⚠️ 局限性**

局限在于仅在受控的化工倒料任务中验证，未覆盖更复杂或真实工况；模型基于传统机器学习，未来需验证在线实时适应性及深度学习方法。

---

## 431. HexFormer: Hyperbolic Vision Transformer with Exponential Map Aggregation

**arXiv ID:** 2601.19849 | [PDF](https://arxiv.org/pdf/2601.19849v1)

**作者:** Haya Alyoussef `[一作]` (Hildesheim University), Lars Schmidt-Thieme `[通讯]` (Hildesheim University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了HexFormer，一种基于Lorentz模型的超平面视觉变压器及其混合变体。

**💡 创新点**

创新点在于引入指数映射聚合的超平面注意力机制，以及在编码器保持超平面、分类头使用欧氏线性分类的混合设计。

**🔧 技术方法**

使用Lorentz空间、指数/对数映射、超平面注意力、混合编码-分类、梯度稳定性分析等技术。

**📊 数据集**

在CIFAR-10、CIFAR-100、Tiny-ImageNet三大数据集上进行实验。

**📈 对比分析**

与传统欧氏ViT、HVT、LViT等进行对比，HexFormer和HexFormer‑Hybrid在所有数据集和模型规模下均优于基线，尤其是混合变体在精度和梯度稳定性上表现最佳。

**⚠️ 局限性**

限制在于超平面网络实现复杂度较高，需调节曲率等超参数，且在更大规模数据集上的验证仍有限。

---

## 432. Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering

**arXiv ID:** 2601.19847 | [PDF](https://arxiv.org/pdf/2601.19847v1)

**作者:** Fangan Dong `[一作]` (Shandong University), Ying Zhou `[通讯]` (Shandong University)

**通讯引用:** 7462 | [OpenAlex ID](https://openalex.org/A5060552570)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AdaRAS，一种在推理时对神经元激活进行自适应调节的轻量级框架，以提升大语言模型的推理可靠性。

**💡 创新点**

①首次系统证明推理正确性与少量神经元激活强相关；②使用均值差分与极性筛选自动识别推理关键神经元（RCN）；③引入失败预测门控，仅在预测推理可能失败时进行激活调节，从而兼顾提升与不破坏原有正确推理；④实现完全无参数、无额外训练、无采样成本的部署。

**🔧 技术方法**

基于激活级别的探测（均值差分、极性判别）、稀疏激活调节向量、注意力预测器的失败判断、对比数据构建（自采样正负推理轨迹）等技术。

**📊 数据集**

十个数学与编码基准：AIME‑24、AIME‑25、AMC‑12、MATH‑500、GSM8K、HumanEval、以及其他若干 STEM 题库。

**📈 对比分析**

与 CoT 基线、公开的后训练模型（DeepSeek‑R1‑Distill‑Qwen‑1.5B、OpenThinker‑3‑1.5B、OpenReasoning‑Nemotron‑1.5B）以及基于探测器的激活调节做对比。AdaRAS 在所有基准上平均提升约 5%（对难题 AIME‑25 提升 13.6%），并能在更强模型（如 Qwen3‑4B）上继续获得 1% 左右的增益；相比探测器调节表现更稳定、泛化更好。

**⚠️ 局限性**

①评估仅覆盖 Qwen3 系列模型和 STEM 任务，需进一步验证对其他架构和更复杂推理的适用性；②依赖对比数据对，获取正确/错误推理轨迹较困难，限制了在超强模型上的应用；③目前仅采用均值差分和极性筛选，未结合更高级的机制解释方法。

---

## 433. Self-Sovereign Identity and eIDAS 2.0: An Analysis of Control, Privacy, and Legal Implications

**arXiv ID:** 2601.19837 | [PDF](https://arxiv.org/pdf/2601.19837v1)

**作者:** Nacereddine Sitouah `[一作]` (Polytechnic University of Milan), Francesco Bruschi `[通讯]` (Polytechnic University of Milan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性评估欧盟 eIDAS 2.0 法规及其架构参考框架（ARF）与自主身份（SSI）原则的兼容性，并提出改进建议。

**💡 创新点**

①将 SSI 的十条核心原则归类为评估框架；②构建基于这些属性的合规性评估指标；③对 eIDAS 2.0 及 ARF 进行逐条比较，识别法律与技术上的冲突与缺口；④给出可操作的技术与监管改进路线图。

**🔧 技术方法**

文献综述、系统性文献筛选（SLR）、规范性与比较性分析、案例场景模拟（SSI 生命周期示例）以及对 OIDC、OIDC4VC、W3C VC、DLT、区块链、ZK‑Proof 等技术的引用与评估。

**📊 数据集**

使用 33 篇精选学术论文（涵盖 SSI 与 eIDAS 兼容性研究）作为数据集；未涉及原始实验或大规模数据集。

**📈 对比分析**

采用定性比较方法：基于先前归纳的 SSI 属性与 eIDAS 2.0/ARF 规定逐条映射，评估“符合/不符合/可改进”三种状态；通过案例对比展示差距。由于为理论与文献分析，未给出数值性能指标，评价主要通过合规性与可操作性三维框架完成。

**⚠️ 局限性**

限制：①缺乏实际实施与测试数据，评估主要基于文献与理论；②对区块链技术的法律与监管认可不足；③ARF 中对去中心化、选择性披露与不可追踪性的规定仍为可选或不完整；④对钱包安全、恢复与可持续性等方面的细节阐述不足；⑤对跨境互操作性与实际应用场景的经验验证有限。

---

## 434. Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models

**arXiv ID:** 2601.19834 | [PDF](https://arxiv.org/pdf/2601.19834v1)

**作者:** Jialong Wu `[一作]` (Tsinghua University), Mingsheng Long `[通讯]` (Tsinghua University)

**通讯引用:** 28802 | [OpenAlex ID](https://openalex.org/A5019241553)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出基于世界模型的多模态推理框架，并通过统一多模态模型（UMM）验证视觉生成在推理中的作用。

**💡 创新点**

创新点在于：①从人类认知角度构建世界模型与链式思考的理论桥梁；②提出“视觉优势假设”并用 VisWorld‑Eval 体系化评测；③展示视觉生成在特定任务中显著提升推理性能，揭示其作用机制。

**🔧 技术方法**

使用技术包括：统一多模态生成模型 BAGEL；视觉与文本交替的链式思考（visual‑verbal CoT）；监督微调（SFT）与可验证奖励强化学习（RLVR）；以及内部表示探测（MLP 诊断）和信息量、互信息分析。

**📊 数据集**

数据集涵盖七类任务：纸张折叠（Paper Folding）、多跳操作（Multi‑hop Manipulation）、球体跟踪（Ball Tracking）、迷宫与 Sokoban、三视图立方体（Cube 3‑View Projection）以及真实世界空间推理（MMSI‑Bench）。这些任务取自 SpatialViz‑Bench、CLEVR、RBench‑V、MMSI‑Bench 等公开资源。

**📈 对比分析**

与仅使用文本 CoT、隐式世界建模、以及纯 VLM Qwen2.5‑VL 进行对比。实验表明：在世界模拟（Paper Folding、Multi‑hop Manipulation、Ball Tracking）和世界重建（Cube 3‑View、MMSI）任务中，视觉交替 CoT 的准确率提升约 5–10%（样本效率提升 4×）；在迷宫和 Sokoban 等网格任务中无明显优势。与 VLM 基线相比，UMM 的视觉推理效果更佳，且 RLVR 可进一步提升但差距未完全消失。

**⚠️ 局限性**

局限性包括：①评估聚焦于空间/物理推理，缺少对其他多模态任务的验证；②视觉生成质量受模型与数据限制，未对视觉生成进行专门的 RL 训练；③只使用了 BAGEL 及其衍生模型，缺少更大规模、更多变种的 UMM；④在某些任务中仍存在信息冗余、误差累积等问题。

---

## 435. Knowledge-Aware Evolution for Streaming Federated Continual Learning with Category Overlap and without Task Identifiers

**arXiv ID:** 2601.19788 | [PDF](https://arxiv.org/pdf/2601.19788v1)

**作者:** Sixing Tan `[一作]` (Harbin Institute of Technology), Xianmin Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 599 | [OpenAlex ID](https://openalex.org/A5044789592)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了针对无任务标签、类别重叠的流式联邦持续学习框架FedKACE，解决了知识混淆与实时推理难题。

**💡 创新点**

创新点在于三项机制的协同：①自适应推理模型切换（从本地到全局模型）；②自适应梯度平衡回放（动态调节缓冲样本权重）；③核谱边界缓冲维护（两阶段筛选提升类别边界信息）。

**🔧 技术方法**

采用联邦学习、样本回放、梯度范数动态权重、核方法与信息熵+决策边界指标的混合缓冲策略。

**📊 数据集**

使用Cifar‑100与ImageNet‑100（前100类）两大公开数据集，按滑动窗口构造不同重叠度的训练序列。

**📈 对比分析**

与FedAVG、TFCL、DCFCL、Re‑Fed、OFCL、FedCBDR等七个基线及中心化上界对比，FedKACE在所有重叠设定下平均准确率最高、平均遗忘率最低。

**⚠️ 局限性**

主要限制是核谱缓冲维护的时间复杂度为O(M(M+D))，在缓冲容量增大时计算开销显著；且对缓冲质量的理论保证在极大规模场景下仍待进一步优化。

---

## 436. A Multi-directional Meta-Learning Framework for Class-Generalizable Anomaly Detection

**arXiv ID:** 2601.19833 | [PDF](https://arxiv.org/pdf/2601.19833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 437. PaW-ViT: A Patch-based Warping Vision Transformer for Robust Ear Verification

**arXiv ID:** 2601.19771 | [PDF](https://arxiv.org/pdf/2601.19771v1)

**作者:** Deeksha Arun `[一作]` (University of Notre Dame), Patrick Flynn `[通讯]` (University of Notre Dame)

**通讯引用:** 30169 | [OpenAlex ID](https://openalex.org/A5039987576)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于解剖知识的耳部图像预处理方法PaW‑ViT，将耳部形状通过边界采样、三角扇形划分、仿射扭曲等步骤变换为结构化的方形补丁，再输入标准Vision Transformer进行耳部识别。

**💡 创新点**

创新点在于：① 用解剖驱动的边界采样与三角扇分割实现耳部形状的统一化；② 通过仿射扭曲将局部三角区域映射为固定大小补丁，构成无重叠的格子图像，既保留细节又保持整体连续性；③ 结合多源地图（分割、标记、并集/交集）进一步提升鲁棒性。

**🔧 技术方法**

核心技术包括：形态学边界提取与凸包平滑、均匀边界采样、三角扇构造、仿射图像变换、补丁拼接；随后采用标准ViT（ViT‑T、S、B、L）进行特征提取和验证。

**📊 数据集**

使用的耳部数据集有：UERC2023（训练集）、OPIB、AWE、WPUT、EarVN1.0（四个独立测试集）。

**📈 对比分析**

实验对比基线原始图像、背景遮罩、单一分割/标记地图以及并集/交集地图。结果显示：在EarVN1.0等高变异数据上，PaW‑ViT可显著提升AUC（最高达0.782±0.0020）；在AWE亦取得最高0.976±0.0020；在OPIB、WPUT上虽提升有限，但仍保持与基线相近；ViT‑B和ViT‑L在大多数数据集上表现最优。

**⚠️ 局限性**

局限性：① 对耳部边界噪声敏感，需先行凸包平滑；② 在存在大量耳饰、遮挡的OPIB/WPUT中，边界驱动的扭曲难以完全消除干扰；③ 仅采用非重叠补丁，未探索重叠或自适应大小补丁带来的潜在改进。

---

## 438. SONIC: Spectral Oriented Neural Invariant Convolutions

**arXiv ID:** 2601.19884 | [PDF](https://arxiv.org/pdf/2601.19884v1)

**作者:** Gijs Joppe Moens `[一作]` (Netherlands Cancer Institute), Eduardo H. P. Pooch `[通讯]` (Maastricht University)

**通讯引用:** 204 | [OpenAlex ID](https://openalex.org/A5068824016)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 SONIC，一种连续谱参数化的卷积算子，利用低秩、方向感知的频域模式实现全局感受野，并可直接在不同分辨率下使用；

**💡 创新点**

创新点在于：① 通过解析解的 LTI 系统模板把频域响应表述为可学习的方向向量、尺度、阻尼、振荡和横向衰减参数；② 用低秩分解（M 个共享模式）构造谱符号，实现参数高效、方向敏感的全局滤波；③ 采用连续谱而非离散频点，保证分辨率不变性；

**🔧 技术方法**

核心技术包括：FFT/逆FFT频域卷积；低秩分解 H(ω)=∑_m C_km T_m(ω) B_mc；解析频域传递函数 T_m(ω) 取自线性时不变系统的解析表达式；在空间域施加非线性与跳跃连接；多层堆叠后实现深度特征提取；

**📊 数据集**

实验使用的公开数据集：Synthetic Shape（SynthShape）与 HalliGalli（空间推理任务）；3D 医学分割数据集：Kidney、Kidney Tumor (KiTS)、ACDC、PI‑CAI、Prostate158、PROMIS；以及 ImageNet‑1K（用于大规模图像分类和分辨率鲁棒性评估）；

**📈 对比分析**

与传统 CNN、ViT、S4ND、NIFF、GFNet、RepLK、Dilated 等方法在 SynthShape、HalliGalli、3D 医学分割和 ImageNet 上进行对比。结果显示：在几何畸变、噪声、尺度变化下，SONIC 的鲁棒性显著优于对比模型；在 3D 医学分割中参数约为传统方法的 10% 以内，却达到或超过最先进性能；在 ImageNet‑1K 上仅 1.34M 参数、0.81 GFLOPs 就能得到 60.01% 的 Top‑1，远低于 ViT（62.23%）但高于 ResNet‑50（58.47%），并且对分辨率变动的性能下降最小；

**⚠️ 局限性**

局限性包括：① 必须在空间域应用非线性，导致多层叠加时需多次 FFT/逆FFT，产生额外计算和内存开销；② 初始化不稳定，需针对不同数据尺度设计更鲁棒的初始化方案；③ 由于全局频域表示，可能对极细微局部细节捕捉不够，建议与传统空间卷积或注意力模块混合使用；

---

## 439. Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty

**arXiv ID:** 2601.19843 | [PDF](https://arxiv.org/pdf/2601.19843v1)

**作者:** Doga Yilmaz `[一作]`, He Wang `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了GraphiXS，一个统一的概率框架，用于在4D高斯展开（Gaussian Splatting）中系统性地引入多种数据不确定性（视角稀疏、帧缺失、相机不同步等）

**💡 创新点**

创新点在于：①将数据不确定性整体建模为概率过程，使用图模型和生成过程；②通过引入多阶动力学（速度、加速度、jerk、snap）和组件置信度分布来约束组件位置和可见性；③支持不同分布（Gaussian、Student's-t）以及可升级现有方法的通用性；

**🔧 技术方法**

使用概率图模型、最大后验估计（MAP）、SGHMC优化、稀疏贝叶斯先验（对透明度、协方差、动力学参数）以及图像重建的光线投射、交叉和栅格化等渲染步骤；

**📊 数据集**

在N3DV（Neural 3D Video）数据集上进行实验，采用1352×1014的低分辨率视频，测试多种视角/帧/同步/故障场景；

**📈 对比分析**

与4DGS‑1、4DGS‑2、Ex4DGS、FreeTimeGS等基线方法在标准、稀疏视角、稀疏帧、不同步相机和故障相机等设置下对比，结果表明GraphiXS在PSNR、LPIPS等指标上多次击败或接近最优，并在大多数不确定性场景下保持最佳性能；

**⚠️ 局限性**

局限性包括：①仅适用于参数化为概率分布（如Gaussian、Student's-t）的组件，无法直接升级使用几何原语的方法；②未实现完整贝叶斯推断，未学习组件参数的后验分布；③在极端不确定性或稀疏数据下仍可能出现过拟合，需要更强先验或正则化。

---

## 440. Information-Theoretic Detection of Bimanual Interactions for Dual-Arm Robot Plan Generation

**arXiv ID:** 2601.19832 | [PDF](https://arxiv.org/pdf/2601.19832v1)

**作者:** Elena Merlo `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于单个RGB视频演示的单步方法，利用信息理论和场景图检测双臂交互模式，并自动生成模块化的行为树执行计划。

**💡 创新点**

创新点在于将Shannon信息理论（互信息、共信息、熵）应用于双手交互，自动识别协调模式并直接生成行为树，解决了单次演示即可生成双臂计划的难题。

**🔧 技术方法**

采用信息理论度量、场景图构建、行为树规划，以及RGB摄像头姿态估计等技术实现。

**📊 数据集**

使用自建的HANDSME开放数据集（十位受试者、厨房/工作坊场景）以及KIT双手数据集进行验证。

**📈 对比分析**

通过与现有双手任务表示方法对比，实验显示该方法能更准确识别协作模式并生成成功的行为树，性能优于对比方法，且在不同感知模块上均能稳定运行。

**⚠️ 局限性**

局限性包括目前无法自动生成复杂动作（如倒液体）需外部轨迹学习；对手部检测噪声敏感，无法处理手部交接或多实例同类物体。

---

## 441. When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering

**arXiv ID:** 2601.19827 | [PDF](https://arxiv.org/pdf/2601.19827v1)

**作者:** Mahdi Astaraki `[一作]` (McMaster University), Soheila Samiee `[通讯]` (BASF Canada Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在化学多跳问答场景下，系统通过训练‑free 的迭代检索‑生成框架（Iterative‑RAG）同步调度检索与推理步骤，探究与理想静态检索（Gold‑Context）相比能否获得更高准确性。

**💡 创新点**

首次在科学多跳任务上实现迭代检索可突破理想证据上限，揭示检索时序与推理流程同步是提升准确率的关键；并提出一套完整的诊断指标（检索覆盖缺失、锚点漂移、误差校准、组合失效、干扰锁定等）。

**🔧 技术方法**

采用检索‑生成管控器：检索-生成-检索循环；利用检索日志和部分答案做状态传递；通过预先标准化检索接口和上下文裁剪保持每步窗口集中；使用LLM-as‑judge进行答案验证。

**📊 数据集**

ChemKGMultiHopQA 数据集（ChemRxiv、PubChem、Wikipedia 交叉索引的 1–4 跳化学问答）。

**📈 对比分析**

对 11 个主流 LLM（GPT‑4o、GPT‑5、Claude 3.7、Claude 4.5、DeepSeek R1、Gemini 2.5 Pro、Llama 3.3 70B、Mistral Large、GLM 4.6、Grok 4 Fast、Claude 3.7 + Reasoning）在三种配置（无上下文、Gold‑Context、Iterative‑RAG）下进行对比。迭代检索平均提升 15–25pp，最优模型 Claude Sonnet 4.5 在迭代模式下超越最佳静态证据 13.8pp，整体准确率从 37.2% 提升到 80.9%。

**⚠️ 局限性**

局限包括：检索覆盖缺失仍导致 28.7pp 的准确率下降；组合失效率高（≈60%），表明生成器仍难以正确整合已检索证据；干扰锁定导致 52pp 的准确率损失；模型对检索步骤的依赖导致推理不稳定（参数记忆抑制、锚点漂移）；部分大模型在迭代过程中过度停止或继续，导致效率低下或成本上升。

---

## 442. An Interpretable Recommendation Model for Psychometric Data, With an Application to Gerontological Primary Care

**arXiv ID:** 2601.19824 | [PDF](https://arxiv.org/pdf/2601.19824v1)

**作者:** Andre Paulino de Lima `[一作]` (University of Sao Paulo), Marcelo Garcia Manzato `[通讯]` (University of Sao Paulo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种名为 Polygrid 的可解释推荐模型，利用心理测量数据（如 WHOQOL）生成符合医疗专业人员认知的雷达图解释，辅助老年人初级保健的个性化护理方案制定。

**💡 创新点**

创新点在于将心理测量得分映射为圆盘多边形、通过分区加权计算匹配度并直接在雷达图上可视化，从而提供与模型计算相一致的可解释视觉输出，填补了医疗领域推荐系统可解释性与数据结构匹配的空白。

**🔧 技术方法**

采用随机森林为基础的多标签分类/排序学习框架，结合自定义的圆盘分区、加权面积计算与雷达图生成算法，兼顾预测性能与解释可视化。

**📊 数据集**

主要使用巴西收集的 WHOQOL‑BREF 质量生活问卷数据，并基于此生成的合成多标签和标签排序标注；实验还引用了 AMPI‑AB 认知功能评估数据来验证模型的通用性。

**📈 对比分析**

通过与传统随机森林、梯度提升树等基线方法在多标签精度、F1 以及标签排序 Kendall’s τ 等指标上进行离线评估，Polygrid 在预测准确率上与基线相当或略优，同时通过用户研究证明其雷达图解释在可理解性和决策支持方面显著优于无解释或文本解释的方法。

**⚠️ 局限性**

局限性包括：需依赖结构良好、正相关的心理测量问卷；对高维或稀疏数据的适用性有限；雷达图解释仅适用于维度数不多的情况；模型解释仍受分区设计与颜色映射的影响，需在更大规模、多机构数据上进一步验证。

---

## 443. Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals

**arXiv ID:** 2601.19810 | [PDF](https://arxiv.org/pdf/2601.19810v1)

**作者:** Octavio Pappalardo `[一作]` `[通讯]` (University College London), Octavio Pappalardo (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在无奖励的预训练环境中，提出了ULEE方法，通过自生成目标和对抗式学习实现元学习预训练，从而得到可迁移的策略。

**💡 创新点**

将后适应难度度量作为自动课程生成的指导，结合对抗式目标生成器和难度预测网络，实现在未知任务分布上高效的自监督元学习。

**🔧 技术方法**

采用基于Transformer‑XL的Actor‑Critic结构，PPO优化，目标生成网络对抗训练，难度预测网络回归，以及基于多回合交互的元学习框架。

**📊 数据集**

使用XLand‑MiniGrid生成的约100万条部分可观测网格环境（4Rooms‑Trivial、4Rooms‑Small、6Rooms‑Small）进行预训练与评估。

**📈 对比分析**

与DIAYN、PPO训练、RND、RL^2等基线对比，ULEE在目标覆盖率、少样本适应、长预算微调以及跨环境结构的泛化上均超过对照组，取得2倍以上的成功率和显著的均值提升。

**⚠️ 局限性**

仍受限于网格世界的离散结构，对长时序任务、视觉输入和更复杂动态的迁移效果有限，且对目标映射f的设计敏感。

---

## 444. Youtu-VL: Unleashing Visual Potential via Unified Vision-Language Supervision

**arXiv ID:** 2601.19798 | [PDF](https://arxiv.org/pdf/2601.19798v1)

**作者:** Zhixiang Wei `[一作]` (Tencent), Xiaotian Li `[通讯]` (Tencent)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5070298048)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Youtu‑VL 框架，采用 Vision‑Language Unified Autoregressive Supervision（VLUAS）将视觉信息从“输入”转为“目标”，实现统一的自回归监督；

**💡 创新点**

创新点在于突破文本主导的训练偏置，利用视觉编码器与视觉 tokenizer 生成离散视觉词汇，统一视觉与语言的预测目标，兼顾细粒度感知与高层语义；

**🔧 技术方法**

核心技术包括：Synergistic Vision Tokenizer（融合 SigLIP‑2 与 DINOv3 的跨注意力分解）、视觉编码器 SigLIP‑2‑2 的 Spatial Merge Projector、NTP‑M 多标签自回归损失、可直接从视觉 token logits 进行 dense 预测（分割、深度、姿态等）；

**📊 数据集**

训练数据覆盖图像‑文本对（5T 语料），OCR、STEM、GUI、知识密集重标记、合成多模任务数据，利用多阶段筛选、稀有类挖掘与知识注入重新标注，规模约 2.4T 令牌；

**📈 对比分析**

在 30 视觉专用任务与 45 通用多模 benchmark 上，Youtu‑VL 与 4‑B‑级 VLM 同级别甚至接近专家模型，尤其在视觉定位、检测、分割、深度、姿态、计数等任务表现稳健，整体性能优于传统 VLM 并接近单任务专用方法；

**⚠️ 局限性**

局限在于视觉分辨率与几何感知的精细程度受限，对低分辨率或不同相机内参的零样本几何任务仍不够鲁棒，且高阶推理与数学推导能力仍低于最先进语言模型。

---

## 445. GeoDiff3D: Self-Supervised 3D Scene Generation with Geometry-Constrained 2D Diffusion Guidance

**arXiv ID:** 2601.19785 | [PDF](https://arxiv.org/pdf/2601.19785v1)

**作者:** Haozhi Zhu `[一作]` (Nanjing University), Fenggen Yu `[通讯]` (Simon Fraser University)

**通讯引用:** 374 | [OpenAlex ID](https://openalex.org/A5111161796)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 GeoDiff3D 框架，利用粗糙几何作为结构锚点，结合 2D 扩散模型自监督生成高质量 3D 场景。

**💡 创新点**

创新点包括：① 通过几何约束的 2D 扩散生成伪 GT 图像；② voxel 对齐 3D 特征聚合与高斯分布预测；③ 双自监督优化（一致性 + GAN + 深度）实现结构一致性与细节保留。

**🔧 技术方法**

采用 Flux‑ControlNet、DINOv2 特征提取、稀疏体素网格、3D Gaussian Splatting、GAN 对抗训练、深度正则化、CLIP 语义过滤等技术。

**📊 数据集**

实验使用 Minecraft 资产、TurboSquid/Free3D 的白色网格场景共 12 个 3D 模型，合成多视角伪 GT 图像，未依赖大规模标注 3D 数据。

**📈 对比分析**

与 Trellis 1.0、World‑Mirror、FSGS、VF3D+3DGS、Marble 等基线比较，PSNR‑D 20.39、MUSIQ 58.06、MANIQ 0.35、CC 0.93、CS 0.90，显示出在视觉质量、几何一致性和风格一致性上的优势。

**⚠️ 局限性**

局限性在于未显式建模天空/大气区域；伪 GT 仅基于线稿，难以捕捉连续深度或复杂遮挡，导致在挑战场景下可靠性受限。

---

## 446. VGGT-SLAM 2.0: Real time Dense Feed-forward Scene Reconstruction

**arXiv ID:** 2601.19887 | [PDF](https://arxiv.org/pdf/2601.19887v1)

**作者:** Dominic Maggio `[一作]` (Massachusetts Institute of Technology), Luca Carlone `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 12147 | [OpenAlex ID](https://openalex.org/A5042157108)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于VGGT的实时无标定单目RGB SLAM系统，能够逐帧构建稠密三维地图并支持开集物体检测。

**💡 创新点**

创新点包括：①新因子图设计消除高维漂移与平面退化；②利用VGGT注意力层实现图像检索验证，显著提升闭环检测并避免误闭环；③将CLIP+SAM实现开集物体查询。

**🔧 技术方法**

技术主要包括VGGT深度与相机标定、因子图优化（GTSAM）、SALAD检索、VGGT注意力分析、CLIP编码、SAM 3分割。

**📊 数据集**

使用Clio（公寓、办公、cubicle）、TUM RGB‑D、KITTI、LaMAR HGE phone等公开数据集进行评测。

**📈 对比分析**

与ORB‑SLAM3、DeepV2D、DPV‑SLAM、ViSTA‑SLAM、DROID‑SLAM等方法对比，平均位姿误差4.1 cm，比主流方法低约23%；闭环检测率提升并消除误闭环。

**⚠️ 局限性**

局限性包括：对单色墙面等无纹理场景重建失败；因子图仅优化位姿导致点云误差；首帧对比例与尺度估计影响大；缺乏失踪跟踪机制。

---

## 447. How Does Delegation in Social Interaction Evolve Over Time? Navigation with a Robot for Blind People

**arXiv ID:** 2601.19851 | [PDF](https://arxiv.org/pdf/2601.19851v1)

**作者:** Rayna Hata `[一作]` (Carnegie Mellon University), Chieko Asakawa `[通讯]` (Miraikan - National Museum of Emerging Science and Innovation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对六位盲人用户在博物馆环境中使用共享控制的导航辅助机器人进行为期三周的长期使用研究，探究在多次交互后用户对机器人社交交互委托的偏好和策略如何演变。

**💡 创新点**

① 在真实公共环境下开展连续三周的纵向实验，首次系统性研究盲人用户委托社交交互与环境感知随时间的变化；② 引入 GPT‑4o 生成的全景环境描述与障碍解释，使机器人能主动提供语义化上下文；③ 结合机器人发声社交请求（“请让路”“请帮助我”）与按钮交互，验证共享控制的实用性。

**🔧 技术方法**

使用装配式行李箱形态机器人，搭载 360° LiDAR、RGB‑D 摄像头、触摸按钮、外置喇叭；通过 GPT‑4o 进行障碍识别与描述；实现机器人端的共享控制接口与人机交互模块；收集用户行为日志、问卷与访谈数据。

**📊 数据集**

研究基于现场收集的用户交互数据（按周的委托率、描述使用次数、行动比例等），未使用公开数据集；实验数据仅限于六名参与者在三周内的使用记录。

**📈 对比分析**

对比每周委托率、GPT 使用频率、用户自评（RoSAS、Likert 题）等指标，观察随时间的提升趋势；结果显示多位用户在噪声或人群拥堵环境下逐步增加对机器人委托；在环境描述使用上从探索性转为目标性；用户对机器人社交请求的接受度显著提升，信任度与自主感随时间提升，整体性能表现为使用习惯化与策略成熟化。

**⚠️ 局限性**

① 样本量小（仅六名用户），缺乏广泛外推性；② 机器人未实现实时问答功能，GPT 调用延迟及硬件限制；③ 无法精确区分人群与排队线导致用户决策不确定；④ 机器人在某些狭窄或倾斜墙面处容易卡住，缺乏可靠的恢复策略；⑤ 语言多样性影响机器人提示的有效性；⑥ 研究仅限室内博物馆环境，未验证户外或公共交通等更大规模场景。

---

## 448. Learn and Verify: A Framework for Rigorous Verification of Physics-Informed Neural Networks

**arXiv ID:** 2601.19818 | [PDF](https://arxiv.org/pdf/2601.19818v1)

**作者:** Kazuaki Tanaka `[一作]` (Waseda University), Kohei Yatabe `[通讯]` (Tokyo University of Agriculture and Technology)

**通讯引用:** 1922 | [OpenAlex ID](https://openalex.org/A5034837951)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种“Learn and Verify”框架，利用PINN构造子、上解并通过区间算术进行严格验证，得到可机证的误差界；

**💡 创新点**

创新点在于将PINN与子/上解构造相结合，使用Doubly Smoothed Maximum (DSM) 损失实现平滑约束，配合自适应细分的区间算术验证，从而实现全局严格误差界；

**🔧 技术方法**

核心技术包括SIREN网络、DSM损失、物理正则化、变分学习、区间算术和自适应细分验证；

**📊 数据集**

实验数据集为三类ODE：经典 Logistic 方程、时间变系数 Logistic 方程、Riccati 方程，采用无显式解析解的数值解作为参考；

**📈 对比分析**

与传统PINN、数值积分方法对比，证明了验证成功率提升、误差界紧凑且可与解析解一致，验证耗时仅数秒；

**⚠️ 局限性**

局限在于对参数（如ε、c1/c2、网络深度）敏感，验证成功率随误差容忍度下降显著，且目前仅针对一阶ODE，扩展到高阶或偏微分方程仍需研究。

---

## 449. Commutative algebras of series

**arXiv ID:** 2601.19809 | [PDF](https://arxiv.org/pdf/2601.19809v1)

**作者:** Lorenzo Clemente `[一作]` (University of Warsaw), Lorenzo Clemente `[通讯]` (University of Warsaw)

**通讯引用:** 484 | [OpenAlex ID](https://openalex.org/A5074542253)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出并研究了一类基于产品规则 P 的正式幂级数乘法（P-产品）以及相应的 P-自动机，并给出了其等价判定问题的可判定性证明。

**💡 创新点**

创新点在于给出了所有使 P-产品成为可交换、结合、双线性代数的必要充分条件，并证明在此条件下的 P-自动机等价性是可判定的，从而统一并推广了 Hadamard、shuffle、infiltration 等经典乘法。

**🔧 技术方法**

采用了共递归定义、协变方法、Hilbert 结构有限基定理以及多项式理想理论等技术，对 P-产品进行符号化刻画并构造多项式 P-自动机。

**📊 数据集**

该工作为纯理论研究，无使用数据集。

**📈 对比分析**

与已知的 Hadamard、shuffle、infiltration 自动机等价判定方法对比，证明了在更广泛的产品规则下仍保持可判定性，且在最坏情况下算法复杂度为 Ackermann 上界。

**⚠️ 局限性**

局限在于仅覆盖了满足“special”条件的 P-产品，对非交换或非结合、或包含输入符号依赖规则的乘法未给出可判定性结果，且对 Cauchy 乘法等更一般情况仍是开放问题。

---

## 450. CASTER: Breaking the Cost-Performance Barrier in Multi-Agent Orchestration via Context-Aware Strategy for Task Efficient Routing

**arXiv ID:** 2601.19793 | [PDF](https://arxiv.org/pdf/2601.19793v1)

**作者:** Shanyv Liu `[一作]` (China University of Petroleum), Shaohua Cao `[通讯]` (China University of Petroleum)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5043877842)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种轻量级的上下文感知动态路由器 CASTER，用于在多智能体系统中根据任务难度在强大模型与弱小模型之间自动切换。

**💡 创新点**

创新点在于将语义嵌入与结构元特征双分支融合，结合冷启动与基于负反馈的迭代演化训练，能够在不牺牲成功率的前提下显著降低推理成本。

**🔧 技术方法**

采用深度双分支网络（语义分支+元特征分支）实现特征融合，配合 on-policy 负反馈学习和自我纠错的强化学习框架；训练使用了 Qwen、GPT‑4o 等多种大语言模型。

**📊 数据集**

构建了跨软件工程、数据分析、科学发现与网络安全四大领域的混合难度基准数据集，并通过 LLM‑as‑a‑Judge 的评判方法对路由效果进行量化。

**📈 对比分析**

与全强模型、全弱模型以及 FrugalGPT 垂直对比，实验显示 CASTER 在成本上降幅最高达 72.4%，同时保持甚至超过强模型的成功率和分数，且在所有领域均优于基线。

**⚠️ 局限性**

局限性包括对语义嵌入和元特征的准确性依赖、对极其新颖或超出训练分布的任务可能误判，以及在不同模型价差极小的场景下收益有限。

---

## 451. Strong Reasoning Isn't Enough: Evaluating Evidence Elicitation in Interactive Diagnosis

**arXiv ID:** 2601.19773 | [PDF](https://arxiv.org/pdf/2601.19773v1)

**作者:** Zhuohan Long `[一作]`, Zhongyu Wei `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

无具体研究内容，本文仅为演示模板

**💡 创新点**

无创新点

**🔧 技术方法**

无技术实现

**📊 数据集**

无数据集

**📈 对比分析**

无方法比较或性能评估

**⚠️ 局限性**

缺乏实际研究信息

---

## 452. Subjective Evaluation of Frame Rate in Bitrate-Constrained Live Streaming

**arXiv ID:** 2601.19776 | [PDF](https://arxiv.org/pdf/2601.19776v1)

**作者:** Jiaqi He `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 9229 | [OpenAlex ID](https://openalex.org/A5020029652)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文创建了一个高帧率实时流媒体数据集（HFR-LS），并进行单刺激隐藏参考的主观评测，探讨压缩强度与帧率在带宽受限场景下的感知权衡；

**💡 创新点**

创新点在于以固定目标比特率为核心，系统地调节压缩强度、分辨率与帧率，构造真实直播环境下的多维编码点，揭示帧率与比特率、源内容之间的交互影响；

**🔧 技术方法**

主要技术包括利用FFmpeg实现快速无延迟编码、采用单刺激隐藏参考法收集DMOS、对评价结果进行Z分数标准化及离群值剔除，并对多种全参考与无参考VQA模型进行四参数Logistic线性化后评估；

**📊 数据集**

使用的数据集为自制的HFR-LS数据集，源视频来自BVI-HFR、UVG、LIVE-YT-HFR，共32段120fps、1080p视频，经过编码得到384个不同比特率与帧率组合；

**📈 对比分析**

通过对13种VQA模型（含PSNR、SSIM、LPIPS、DISTS、VMAF、NIQE、VSFA、Li22、DOVER、ModularVQA、MinimalisticVQA等）在PLCC与SRCC两指标上进行比较，发现MinimalisticVQA表现最好，但相关性仍相对较低；

**⚠️ 局限性**

局限性包括：FR‑VQA在帧率变化大时表现不佳，主观实验样本量仅30人，数据集仅覆盖1080p 5–15Mbps范围，且缺乏更高分辨率与更宽比特率的测试，未来需扩展多维度编码点并研发更能捕捉帧率与压缩交互的VQA方法。

---

## 453. EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning

**arXiv ID:** 2601.19850 | [PDF](https://arxiv.org/pdf/2601.19850v1)

**作者:** Binzhu Xie `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 51733 | [OpenAlex ID](https://openalex.org/A5032708386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种针对第一人称视角下的3D手部重建框架 EgoHandICL，利用基于上下文学习的思路实现对严重遮挡和复杂手物交互场景的鲁棒重建。

**💡 创新点**

创新点在于：①首次将 in‑context learning (ICL) 应用于 3D 手部重建；②设计了 VLM‑引导的双重模板检索策略（视觉模板 + 语义模板）；③提出了融合视觉、文本和结构信息的 ICL tokenizer；④采用 Masked Autoencoders (MAE) 训练模式并加入 3D 感知损失以提升结构一致性。

**🔧 技术方法**

核心技术包括：视觉语言模型 (VLM) 检索、MANO 参数化、ViT 视觉编码器、文本编码器、跨注意力融合、MAE 结构、参数与顶点级监督、3D 感知损失。

**📊 数据集**

使用了 ARCTIC（手部网格、MANO 注释）和 EgoExo4D（自由视角手部关键点）两个基准数据集；在 EgoHOIBench 上进一步验证了对手物交互推理的提升。

**📈 对比分析**

与 HaMeR、WiLoR、WildHand、HaWoR 等先进方法对比，EgoHandICL 在 ARCTIC 的 PA‑MPVPE 上提升约 30–35%，在 EgoExo4D 的 MPJPE 与 F@10/F@15 亦实现显著改进，显示出更优的重建精度与空间一致性。

**⚠️ 局限性**

主要局限：检索过程中依赖大型 VLM，导致实时部署的计算成本偏高；此外，目前框架仅针对单帧重建，未覆盖连续追踪或手物重建等更广泛的 egocentric 场景。

---

## 454. GAVEL: Towards rule-based safety through activation monitoring

**arXiv ID:** 2601.19768 | [PDF](https://arxiv.org/pdf/2601.19768v1)

**作者:** Shir Rozenfeld `[一作]` (Ben Gurion University of the Negev), Yisroel Mirsky `[通讯]` (Ben Gurion University of the Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 GAVEL 框架，基于认知元素（CE）对大语言模型内部激活进行规则化监控，实现了精确的违规检测与可解释性保障。

**💡 创新点**

创新点在于：①将模型激活拆解为可组合的认知元素；②通过可共享、可配置的逻辑规则实现高精度、灵活的安全约束；③提供自动化工具生成 CE 与规则，构建可扩展的社区生态。

**🔧 技术方法**

技术包括：基于 Transformer 自注意力层的激活提取、利用多标签 RNN（GRU）对 CE 进行分类、逻辑规则引擎评估窗口内 CE 组合、自动化 CE/规则生成器（LLM 驱动）以及多语言支持的评测。

**📊 数据集**

数据集涵盖：自行构造的 9 类违规对话（共 14,950 条）与 500 条近似合法对话；1,000 条自然对话的 FPR 测试；以及公开基准（PKU‑SafeRLHF、Reasoning Shield、ToxiGen 等）用于自动化适配验证。

**📈 对比分析**

与 4 类主流安全基线（微调、读向量投影、文本 Moderation API、激活分类器）相比，GAVEL 在 Mistral‑7B 上实现了 0.99+ 的 AUC、近零 FPR 以及 97% 以上的整体平衡准确率，显著提升了精度与覆盖率。

**⚠️ 局限性**

局限包括：规则逻辑仍受限于手工或自动化生成的 CE 质量；在极端对抗下可能存在 CE 层级绕过；当前窗口机制主要处理短期违规，需改进对长期或动态违规的捕捉。

---

## 455. Self-Distillation Enables Continual Learning

**arXiv ID:** 2601.19897 | [PDF](https://arxiv.org/pdf/2601.19897v1)

**作者:** Idan Shenfeld `[一作]` (Massachusetts Institute of Technology), Pulkit Agrawal `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 5165 | [OpenAlex ID](https://openalex.org/A5111774389)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Self-Distillation Fine‑Tuning（SDFT），让同一大模型在仅依赖专家示例的情况下完成 on‑policy 学习，从而实现持续学习。

**💡 创新点**

创新点在于利用模型的上下文学习能力，将条件化的模型视为教师，使用逆 KL 蒸馏实现自我学习，兼具教师质量与 on‑policy 更新的双重优势。

**🔧 技术方法**

核心技术包括基于逆 KL 的自蒸馏损失、对抗式教师-学生角色、指数移动平均（EMA）教师参数，以及在推理层面自动生成训练轨迹。

**📊 数据集**

实验数据集涵盖三类技能学习任务（Science Q&A、Tool Use、Medical）和知识获取任务（2025 年自然灾害维基百科问答对），并在多种通用基准（HellaSwag、TruthfulQA、MMLU 等）上评估记忆保持。

**📈 对比分析**

与传统 SFT、DFT、Re‑invoke、CPT、RAG 等方法对比，SDFT 在新任务精度上提升 5–15%（最高 89% vs 80%），显著降低灾难性遗忘，且在 OOD 题目上仍保持高准确率，整体性能优于现有基线。

**⚠️ 局限性**

局限性包括：需要足够强的上下文学习能力（小模型表现不佳）、额外的生成成本导致训练时间约 2–4 倍、易出现教师引入的语言风格痕迹，以及对需要根本行为变更的任务适应性有限。

---

## 456. Post-LayerNorm Is Back: Stable, ExpressivE, and Deep

**arXiv ID:** 2601.19895 | [PDF](https://arxiv.org/pdf/2601.19895v1)

**作者:** Chen Chen `[一作]` (ByteDance Seed), Lai Wei `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在Post‑LayerNorm Transformer中引入Highway风格残差与额外LayerNorm的改进架构，能够在深度超过1000层时实现稳定训练并提升模型表达能力。

**💡 创新点**

核心创新是将残差路径改为可调的Highway分支并加上双层LayerNorm，理论上消除梯度消失并保持梯度幅度≈1，从而打破传统Post‑LN在极深网络中的不稳定瓶颈。

**🔧 技术方法**

技术手段包括梯度动态分析、α尺度调节（α＝L）、双LayerNorm设计、深度可视化与最大容忍学习率评估，以及在1T token规模下的LLM预训练与SFT；评估使用lm‑evaluation‑harness等标准基准。

**📊 数据集**

训练使用内部1T token、250B token私有语料以及FineWeb‑EDU 10B/40B；评测覆盖MMLU、ARC、GSM‑8K、HumanEval、MBPP、CMMLU、C‑Eval、BBH等多领域基准。

**📈 对比分析**

与Pre‑LN、DeepNorm、HybridNorm等基线比较，采用最大容忍学习率、零/少样本评估，实验显示该架构在64–1024层、10–40B token等场景下均能实现更高的学习率并在推理、代码生成等任务上提升≈10–16%（GSM‑8K +10、Math/Code +16.5%），并在极深网络上保持稳定收敛。

**⚠️ 局限性**

局限性包括：在宽度扩展（高隐藏维或多专家）时需要更大α或更强调节机制；低数据量环境下效果不显著；尚未系统研究宽度与深度交叉的稳定性。

---

## 457. HARMONI: Multimodal Personalization of Multi-User Human-Robot Interactions with LLMs

**arXiv ID:** 2601.19839 | [PDF](https://arxiv.org/pdf/2601.19839v1)

**作者:** Jeanne Malécot `[一作]` (Institut Curie), Mohamed Chetouani `[通讯]` (Institute of Intelligent Systems and Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了HARMONI框架，实现多模态感知、世界建模、用户建模与生成四模块的集成，使机器人能够在长期多用户环境中进行个性化、动态的社交互动。

**💡 创新点**

创新点在于：①将LLM与多模态感知紧耦合，支持持续的长时记忆更新与短时对话上下文维护；②引入伦理与隐私保护机制，确保个性化不侵犯用户隐私；③通过双推理（更新与生成）提升生成质量与效率；④在真实养老院场景中验证多用户交互性能。

**🔧 技术方法**

使用技术包括：YOLOv8-Face-Detection、FastRTC声学检测、Whisper-Turbo语音识别、INSIGHTFACE面部特征编码、Gemma、GPT‑4o等LLM、语义相似检索（EmbeddingGemma-300m、google/EmbeddingGemma-300m）、两阶段LLM推理、隐私过滤与安全约束。

**📊 数据集**

使用数据集：45个自制多用户视频数据、LoCoMo语义标注数据、PersonaFeedback多语言翻译数据、Multi-User MultiWOZ、以及养老院现场实验收集的交互数据。

**📈 对比分析**

通过与基线LLM、prompt‑engineered LLM以及两阶段推理的对比实验，评估用户检测、记忆更新与个性化质量。实验结果显示：用户检测准确率≈90%、用户检索≈98%；在PersonaFeedback上，个性化质量显著提升（ROUGE、Session Similarity均提高）；SUS得分82.4/100，专家评价平均4/5；整体延迟低于3 ms的感知阶段，LLM推理在两阶段模式下保持可接受的延迟。

**⚠️ 局限性**

局限性包括：当前仅支持短视频录制，缺乏实时流媒体；隐私数据在本地存储且未完全实现流式处理；对时间敏感信息记忆不足，无法精确回忆具体时间；对小规模LLM的表现仍不稳定；整体系统延迟在加入长短期记忆后略有增加。

---

## 458. Neural Neural Scaling Laws

**arXiv ID:** 2601.19831 | [PDF](https://arxiv.org/pdf/2601.19831v1)

**作者:** Michael Y. Hu `[一作]` (New York University), Kyunghyun Cho `[通讯]` (New York University)

**通讯引用:** 52445 | [OpenAlex ID](https://openalex.org/A5091175785)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种基于神经网络的“Neural Neural Scaling Laws”，通过结合token级验证损失分布和历史下游准确率序列进行时间序列外推来预测语言模型的下游任务性能。

**💡 创新点**

创新点在于：①不依赖传统的对数/指数参数化，而是使用Transformer+CNN编码器直接学习token损失分布；②通过量化回归提供不确定性估计；③利用token级概率而非平均损失，显著提升预测精度。

**🔧 技术方法**

技术手段包括：1D卷积编码器、Transformer编码器、RoPE位置编码、量化回归（pinball loss）、分布差异直方图、以及多种输入表征（全概率、平均、直方图差异）。

**📊 数据集**

使用的数据集为：HuggingFace公开的DataDecide模型训练轨迹（90M~1B参数的6个规模），每个轨迹包含3个随机种子；下游任务为OllMES套件中的66个分类任务；token概率来自WebOrganizer的256k连续token样本。

**📈 对比分析**

与传统logistic scaling law、LC-PFN、DiffProbe、NoLoss、Average、HistDiff等方法比较，平均绝对误差（MAE）为2.04%（比logistic 3.29%降低38%）；在不同随机种子、数据集、模型族、未见任务的零射预测中也表现更优；排名准确率提升至0.756，优于传统方法。

**⚠️ 局限性**

局限性包括：①仅适用于同一验证集的token损失，无法直接处理不同验证集；②评估主要聚焦于分类任务，生成任务的预测效果未知；③尚未深入解释CNN编码器提取的分布特征及其对预测的具体贡献。

---

## 459. Whether We Care, How We Reason: The Dual Role of Anthropomorphism and Moral Foundations in Robot Abuse

**arXiv ID:** 2601.19826 | [PDF](https://arxiv.org/pdf/2601.19826v1)

**作者:** Fan Yang `[一作]` (University of South Carolina), Lingyao Li `[通讯]` (University of South Florida)

**通讯引用:** 1157 | [OpenAlex ID](https://openalex.org/A5031522503)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过混合方法（实验 + 主题分析）探讨了人类对机器人虐待的情感与社会反应，检验了机器人人形化水平与个人道德基础如何共同决定对机器人是否赋予道德关怀。

**💡 创新点**

创新点在于首次将道德基础理论与机器人人形化特征相结合，揭示人形化决定是否延伸道德考虑，而道德基础决定推理过程与情绪反应的模式。

**🔧 技术方法**

采用问卷调查（含MFQ‑20、愤怒量表、社会距离量表）和开放式问答的主题分析技术，并通过MANOVA统计检验量化结果。

**📊 数据集**

数据来源于201名来自Prolific的受试者观看三种不同人形化水平机器人虐待视频后的自评，形成了实验与主观文本双模态数据集。

**📈 对比分析**

比较方法为MANOVA和多组主题分析，结果显示人形机器人（Humanoid、Twofoot）引发更高愤怒、较低社会距离，并且高进步主义者在高人形化情境下表现出更强烈的情绪与道德推理，显示人形化与道德基础共同提升道德关怀强度。

**⚠️ 局限性**

局限性包括仅使用视频观察而非现场体验、只关注物理虐待、受试者来源为在线平台、样本规模有限、未考察不同文化背景及长期接触对结果的影响。

---

## 460. Component-Aware Pruning Framework for Neural Network Controllers via Gradient-Based Importance Estimation

**arXiv ID:** 2601.19794 | [PDF](https://arxiv.org/pdf/2601.19794v1)

**作者:** Ganesh Sundaram `[一作]` (RPTU University Kaiserslautern-Landau), Daniel Görges `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种面向多组件神经网络的组件感知结构化剪枝框架，通过梯度信息计算三种重要性指标，实现对不同参数组的动态压缩。

**💡 创新点**

创新点在于将梯度累积、Fisher信息与贝叶斯不确定性结合为在线重要性度量，并在训练期间动态调节正则化，使剪枝决策更具数据驱动且不依赖静态权重范数。

**🔧 技术方法**

采用梯度累积、Fisher信息矩阵（对角近似）、贝叶斯经验估计，以及指数移动平均、平滑和余弦调度等训练技术，实现结构化剪枝。

**📊 数据集**

在MNIST手写数字数据集训练的自编码器以及OpenAI Gym倒立摆任务的TD‑MPC控制器上进行评估。

**📈 对比分析**

通过与基于权重范数的剪枝、无结构剪枝以及现有结构化剪枝方法对比，实验显示在保持控制性能和重建质量的前提下，可实现数十倍压缩，并在高容量模型中显著降低重建误差或提升奖励。

**⚠️ 局限性**

局限性包括需要额外的梯度收集与在线计算，且对训练超参数（如EMA系数、伪计数κ、调度周期）敏感；目前未给出理论稳定性保证，且对极大规模网络的计算开销尚待验证。

---

## 461. To Grok Grokking: Provable Grokking in Ridge Regression

**arXiv ID:** 2601.19791 | [PDF](https://arxiv.org/pdf/2601.19791v1)

**作者:** Mingyue Xu `[一作]` (Purdue University), Itay Safran `[通讯]` (Ben-Gurion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了过参数化线性回归（岭回归）中的grokking现象，给出了从过拟合到泛化的端到端可证明理论，并通过实验验证了超参数对grokking时间的影响。

**💡 创新点**

首次在岭回归框架下提供完整的grokking证明，推导出训练误差与泛化误差收敛率的差异导致的延迟，并给出量化的超参数控制grokking时间的上界与下界。

**🔧 技术方法**

采用梯度下降优化、岭回归正则化、随机矩阵理论、统一收敛分析以及随机特征网络的数值实验等技术。

**📊 数据集**

主要使用人工合成的教师函数（零教师或可实现线性教师），随机高斯特征，以及随机特征两层ReLU网络生成的数据进行实验。

**📈 对比分析**

通过对训练与测试均方误差随迭代步数的曲线进行对比，验证理论给出的 t1 与 t2 的增长/下降趋势；实验结果与理论界限在趋势和量级上高度吻合。

**⚠️ 局限性**

仅针对可实现的线性/随机特征设置，未对非可实现或真实网络给出完整证明；对特征分布的依赖（如 λ_min(Φᵀ)）及高维特征矩阵谱性质的精确量化仍有待完善。

---

## 462. Phonological Tokenizer: Prosody-Aware Phonetic Token via Multi-Objective Fine-Tuning with Differentiable K-Means

**arXiv ID:** 2601.19781 | [PDF](https://arxiv.org/pdf/2601.19781v1)

**作者:** Kentaro Onda `[一作]` (University of Tokyo), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**通讯引用:** 25423 | [OpenAlex ID](https://openalex.org/A5001291873)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并训练了 Phonological Tokenizer，通过对预训练的 SSL 模型使用可微分 k-means 和 ASR 与重构的多任务目标，生成既包含语言信息又保留韵律信息、去除说话人身份的离散语音 token。

**💡 创新点**

创新性地将可微分 k-means 与多任务学习结合，得到单码本的语音 token，既兼顾语音合成与识别的优势，又在保持压缩效率的同时抑制说话人信息。

**🔧 技术方法**

使用可微分 k-means 对 SSL 特征进行离散化，联合训练 ASR 损失（CTC+Attention）与声码器重构损失，采用 WavLM‑large 作为 SSL 模型、HiFi‑GAN 声码器和预训练说话人编码器进行条件化。

**📊 数据集**

在 30 h 的 LibriSpeech 子集上初始化聚类，随后用 44 h 的 VCTK 语料进行微调，评估数据包括 LibriSpeech‑100h、RAVDESS、VoxCeleb1、LJSpeech、TIMIT、Expresso 及 6000 h LibriLight。

**📈 对比分析**

与传统 phonetic、hybrid、acoustic token 以及 ASR‑only、voc‑only 基线通过 ASR、情感识别、说话人识别、重建、声源转换、speechLM 等多任务指标比较；在多数任务中 Phonological Tokenizer 与基线持平或略优，情感识别、声源转换和 speechLM 生成质量均显著提升。

**⚠️ 局限性**

仅使用少量 44 h 语料进行微调，导致在情感丰富的语料上仍有一定的韵律失真；α 参数需仔细调节以平衡韵律与说话人信息；目前仍无法完全实现对说话人与韵律的完全解耦。

---

## 463. Reimagining Peer Review Process Through Multi-Agent Mechanism Design

**arXiv ID:** 2601.19778 | [PDF](https://arxiv.org/pdf/2601.19778v1)

**作者:** Ahmad Farooq `[一作]` (University of Arkansas at Little Rock), Kamran Iqbal `[通讯]` (University of Arkansas at Little Rock)

**通讯引用:** 2233 | [OpenAlex ID](https://openalex.org/A5068395409)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出利用多智能体机制设计来重塑科研社区的同行评审过程。

**💡 创新点**

创新点在于将同行评审视为可持续的信用经济、强化学习优化分配和混合验证的三柱框架。

**🔧 技术方法**

采用多智能体强化学习、信息论奖励机制、LLM验证器与经济学价格模型。

**📊 数据集**

未使用具体实验数据，计划基于OpenReview历史记录和模拟的Agent‑Based Model进行验证。

**📈 对比分析**

目前仅在仿真与小规模试点中与传统静态匹配方法对比，预期在审核及时性与公平性上提升10–20%。

**⚠️ 局限性**

局限包括信用系统易被操纵、强化学习模型易嵌入偏见以及LLM验证可能导致Goodhart效应。

---

## 464. Assessing Task-based Chatbots: Snapshot and Curated Datasets for Dialogflow

**arXiv ID:** 2601.19787 | [PDF](https://arxiv.org/pdf/2601.19787v1)

**作者:** Elena Masserini `[一作]` (University of Milano-Bicocca), Leonardo Mariani `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 3563 | [OpenAlex ID](https://openalex.org/A5036120394)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过自动化爬取和筛选GitHub上 Dialogflow 版本的聊天机器人，构建了 1,788 个聊天机器人的 TOFU‑D 数据集，并进一步挑选 185 个符合功能、语言、实用性和安全性要求的聊天机器人形成 COD 数据集，随后使用 Botium 进行自动化测试、Bandit 进行安全漏洞扫描，并与 Rasa 的 BRASATO 数据集进行对比。

**💡 创新点**

创新点在于首次提供覆盖商业平台 Dialogflow 的多语言、多编程语言、多功能（云函数、Google Assistant）聊天机器人数据集，并通过系统化的自动化收集、去重与验证流程，将样本规模翻倍，显著提升跨平台质量与安全研究的可重复性和代表性。

**🔧 技术方法**

主要技术包括：GitHub API + 自定义脚本进行仓库检索与 agent 文件解析；GPT‑4o 生成领域标签；Botium 框架生成并执行对话测试用例；Bandit 静态分析 Python 代码检测安全漏洞；以及序列比较库评估后端代码相似度以去重。

**📊 数据集**

使用的数据集为 TOFU‑D（1,788 个 Dialogflow 聊天机器人）和 COD（185 个手工挑选的高质量子集），并与先前公开的 Rasa 聊天机器人数据集 BRASATO 进行对照分析。

**📈 对比分析**

比较方法为：对 COD 中 10 个机器人使用 Botium 生成测试用例，评估意图、实体和前置条件覆盖率；对包含 Python 后端的 69 个 COD 机器人与 193 个 BRASATO 机器人分别使用 Bandit 分析漏洞频率。结果显示 COD 中 100% 机器人缺失 fallback 测试、30% 缺失问候测试、30% 缺失前置条件测试，70% 实体仅部分覆盖；安全扫描中 COD 的 41% 机器人存在所有接口绑定、32% 存在可执行任意代码的风险，表明 Dialogflow 机器人在功能和安全上均表现出更大挑战。

**⚠️ 局限性**

局限性包括：仅聚焦 Dialogflow Essentials 版，未覆盖 CX 企业版；收集关键词可能漏检部分聊天机器人；数据集主要以英语为主，语言多样性有限；验证过程依赖 REST API 与 Botium，未覆盖所有可能的部署环境；以及数据集基于 2025‑09‑16 的快照，后续可能因平台更新导致样本失效。

---

## 465. Enabling SSI-Compliant Use of EUDI Wallet Credentials through Trusted Execution Environment and Zero-Knowledge Proof

**arXiv ID:** 2601.19893 | [PDF](https://arxiv.org/pdf/2601.19893v1)

**作者:** Nacereddine Sitouah `[一作]` (Polytechnic University of Milan), Stefano De Cillis `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种基于可信执行环境（TEE）和零知识证明（ZKP）的架构，使意大利IT‑Wallet（EUDI Wallet）凭证可在符合自决身份（SSI）的生态系统中使用。

**💡 创新点**

通过在TEE内验证原始SD‑JWT‑VC凭证并生成硬件证明，再利用ZKP压缩验证结果，实现离线、不可追踪且不依赖中心化第三方的凭证可验证性。

**🔧 技术方法**

主要技术包括Intel SGX/TeeAttestation、SD‑JWT‑VC、ZKP（针对SGX证明的零知识证据）、confidential Kubernetes（CVM）以及智能合约事件存储。

**📊 数据集**

论文为理论与架构设计，未使用实际数据集进行实验；主要基于现有的IT‑Wallet凭证格式和eIDAS 2.0规范进行假设与推演。

**📈 对比分析**

未给出具体性能基准或与现有方案对比；作者指出需要进一步实现原型以评估ZKP生成成本、TEE资源占用以及链上事件的费用。

**⚠️ 局限性**

局限包括：需要TEE硬件和可信的云/容器环境，普通用户难以获取；短期凭证（如WUA）支持不足；在未实现前无法验证安全性与可扩展性。

---

## 466. Routing End User Queries to Enterprise Databases

**arXiv ID:** 2601.19825 | [PDF](https://arxiv.org/pdf/2601.19825v1)

**作者:** Saikrishna Sudarshan `[一作]` (Birla Institute of Technology and Science Pilani), Tanmay Tulsidas Verlekar `[通讯]` (Birla Institute of Technology and Science Pilani)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在企业多数据库环境下提出了一套将自然语言查询路由到最合适数据库的完整流程，并构建了更真实的基准数据集。

**💡 创新点**

核心创新在于将查询路由拆解为 schema 覆盖、表连通性和语义对齐三大子任务，并通过模块化推理与 LLM 结合实现高精度重排名。

**🔧 技术方法**

使用预训练嵌入模型 gte‑Qwen2‑7B‑instruct 进行语义匹配，利用 Gemini 2.0/2.5 进行 LLM 推理；同时采用图遍历算法检查表连通性并计算覆盖得分。

**📊 数据集**

数据集为改造后的 Spider‑Route（206 DB，11831 题）与 Bird‑Route（80 DB，10962 题），两者均采用 50‑50% 训练/测试划分，保证查询与 DB 的无重叠。

**📈 对比分析**

与传统余弦嵌入、LLM 直接重排名基线对比，模块化重排名在 R@1、R@3 及 mAP 上提升约 10‑20%，在同域数据库区分上表现尤为突出。

**⚠️ 局限性**

仍存在约 20% 查询因语义歧义或缺失高质量 LLM/更大 k 值导致 100% R@1 无法实现，需要更丰富的领域知识或进一步优化重排名策略。

---

## 467. Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering

**arXiv ID:** 2601.19821 | [PDF](https://arxiv.org/pdf/2601.19821v1)

**作者:** Kun Li `[一作]` (University of Twente), Sami Sebastian Brandt `[通讯]` (IT University of Copenhagen)

**通讯引用:** 1998 | [OpenAlex ID](https://openalex.org/A5083670972)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于查询引导的空间–时间–频率交互框架QSTar，用来提升音乐音视问答（AVQA）的性能，整个过程中始终将问题语义嵌入并引导音频、视频特征的学习与融合。

**💡 创新点**

创新点主要有：①在多模态特征学习的起始、中间和末端全程引入查询引导；②利用频谱Transformer与频率注意力捕捉音频频域特征；③设计空间–时间–频率交互模块（STI+TFI）以及基于提示的查询上下文推理块（QCR），实现跨模态的细粒度语义对齐。

**🔧 技术方法**

技术包括：CLIP视觉与文本编码器、VGGish音频特征、AST频谱Transformer、Token Merging、全自注意与交叉注意机制、频率注意力、卷积融合、Prompt式查询上下文推理。

**📊 数据集**

主要数据集为 MUSIC‑AVQA（40K+问答对，9K+视频），并在 AVQA 数据集上进行补充评估。

**📈 对比分析**

在 MUSIC‑AVQA 上平均准确率达 78.98%，比最新 SOTA QA‑TIGER 的 77.62% 提升 1.36%；在计数、比较、定位、存在、时间等各类子任务上均实现显著提升，尤其在音频和音视问答类型表现突出。

**⚠️ 局限性**

局限性包括：仍依赖预训练模型且未引入专门的目标检测模块，空间定位效果相对传统物体级方法略逊；频域模块对极端多音轨场景的适应性需进一步验证；缺乏对跨域或更大规模视频数据的泛化评估。

---

## 468. Zero-Shot Stance Detection in the Wild: Dynamic Target Generation and Multi-Target Adaptation

**arXiv ID:** 2601.19802 | [PDF](https://arxiv.org/pdf/2601.19802v1)

**作者:** Aohua Li `[一作]` (Minzu University of China), Xiaobing Zhao `[通讯]` (Minzu University of China)

**通讯引用:** 1736 | [OpenAlex ID](https://openalex.org/A5100773044)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出零射击立场检测任务，在野外进行动态目标生成与多目标适配。

**💡 创新点**

创新点在于不预定义目标，自动识别并给出多目标-立场对，结合多维评估。

**🔧 技术方法**

使用大语言模型（Qwen2.5-7B、DeepSeek-R1-Distill-Qwen-7B）两种微调策略及提示法。

**📊 数据集**

构建70,931条中文微博多域立场标注数据集。

**📈 对比分析**

与预训练模型、提示LLM对比，微调LLM在目标识别C-Score和立场F1均达到66%+和79%+，优于基线。

**⚠️ 局限性**

局限在对隐式目标和讽刺语义的理解不足，且多目标数量增大时性能下降。

---

## 469. Diffusion for De-Occlusion: Accessory-Aware Diffusion Inpainting for Robust Ear Biometric Recognition

**arXiv ID:** 2601.19795 | [PDF](https://arxiv.org/pdf/2601.19795v1)

**作者:** Deeksha Arun `[一作]` (University of Notre Dame), Patrick Flynn `[通讯]` (University of Notre Dame)

**通讯引用:** 30169 | [OpenAlex ID](https://openalex.org/A5039987576)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于扩散模型的耳朵去配饰修复（inpainting）方法，先自动检测耳饰区域并生成掩码，再利用扩散式修复生成无耳饰、解剖学合理的耳图像，随后将修复后的图像作为预处理输入到基于视觉变换器（ViT）的耳部识别模型中进行验证。

**💡 创新点**

创新点主要包括：①首次将耳饰-aware的扩散式inpainting作为前置模块应用于耳部识别；②构建完整的自动化掩码生成流水线，融合YOLOv10、Grounding DINO与SAM2实现高质量耳饰分割；③在四个基准数据集上进行多尺度、多backbone、多tokenization的系统性评估，展示了在不同patch尺寸下的性能变化。

**🔧 技术方法**

核心技术包括：YOLOv10+Grounding DINO用于候选框定位；SAM2用于像素级掩码生成；扩散式inpainting（LaMa式推理）用于耳部修复；Vision Transformer（ViT-T/S/B/L）作为识别backbone；非重叠patch tokenization（16、28、56）用于研究token级别对遮挡鲁棒性的影响；AUC（ROC曲线下面积）作为评估指标。

**📊 数据集**

训练集采用UERC2023（约247k张耳图），测试集涵盖四个公开基准：AWE、OPIB、WPUT、EarVN1.0。

**📈 对比分析**

比较方法：对同一ViT配置和patch大小，分别使用原始带遮挡的图像（Baseline）和先做扩散inpainting后的图像（Inpainted）进行训练与测试；对比AUC得分。实验表明，在遮挡严重、patch尺寸较粗（如56）或数据集为EarVN1.0时，Inpainted可提升AUC最高可达+8.1%（如ViT_B_p16），整体平均提升约+1.8%至+2.5%；但在遮挡较轻或patch细化（16/28）时，提升有限甚至略有下降。

**⚠️ 局限性**

局限性包括：①对遮挡较轻或图像质量较高的样本，生成的修复图可能导致身份细节被平滑或误改，从而略降识别精度；②掩码误检或图像旋转产生的黑色填充可能导致修复失真；③目前缺乏针对身份保持的约束（如特征一致性或感知损失），导致修复过程中可能产生身份漂移；④仅针对耳饰遮挡，未覆盖头发、围巾等其他遮挡类型。

---

## 470. LVLMs and Humans Ground Differently in Referential Communication

**arXiv ID:** 2601.19792 | [PDF](https://arxiv.org/pdf/2601.19792v1)

**作者:** Peter Zeng `[一作]` (Stony Brook University), Owen Rambow `[通讯]` (Stony Brook University)

**通讯引用:** 8834 | [OpenAlex ID](https://openalex.org/A5021314411)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验中，作者设计了一个多轮指称沟通任务，让人类、AI以及两者混合的对话伙伴在不同角色下协作完成目标识别，并收集了356条对话。

**💡 创新点**

创新点在于：①首次将四种人机配对（人类-人类、人类-AI、AI-人类、AI-AI）纳入实验；②使用大型视觉语言模型GPT‑5.2在实时多轮对话中评估其共同基础构建能力；③量化沟通成功、努力与词汇交融三维指标，并公开完整数据集。

**🔧 技术方法**

主要技术包括：大型视觉语言模型GPT‑5.2（无推理模式）作为AI参与者；oTree在线实验平台；通过GPT‑5自动提取指称表达；使用OLS回归、词频与词汇重叠指标进行量化分析。

**📊 数据集**

数据集为356条指称沟通对话（32人类‑人类、17人类‑AI、22AI‑人类、18AI‑AI），涵盖四轮，每轮12个篮子目标，已公开。

**📈 对比分析**

比较方法：对四种配对分别统计准确率、词数、回合数、词汇重叠等；结果显示人类‑人类对齐率最高且随轮次提升；AI‑AI虽起始准确率高但随轮下降，词汇量与回合数基本不变；混合对话表现中等，AI主导时准确率急降。

**⚠️ 局限性**

局限性：仅使用英语和单一对象类型；仅评估GPT‑5.2，未包含开源模型；对话仅文本，未涉及多模态或更真实环境；提示工程高度定制，结果可能不具普适性；未对修复、幻觉等现象做深入分析。

---

## 471. Evaluation of Oncotimia: An LLM based system for supporting tumour boards

**arXiv ID:** 2601.19899 | [PDF](https://arxiv.org/pdf/2601.19899v1)

**作者:** Luis Lorenzo `[一作]` (Bahía Software SLU), Cristobal Bernardo-Castineira `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了ONCOTIMIA平台，利用LLM自动完成肺癌MDTB表单，显著降低临床文档工作量。

**💡 创新点**

将检索增强生成（RAG）与规则驱动的自适应表单模型结合，实现非结构化临床文本的精准结构化自动填充。

**🔧 技术方法**

采用模块化架构、数据湖、向量与关系存储（Qdrant+PostgreSQL）、LangChain、Nomic嵌入、LLM抽象层，并在AWS Bedrock上部署六大LLM（GPT‑OSS‑20b/120b, Mistral‑large‑2402, Pixtral‑large‑2502, Qwen3‑32b/80b）。

**📊 数据集**

使用10个基于真实病例模板生成的合成西班牙肺癌病历，经过自动与人工校验后构成评估数据集。

**📈 对比分析**

通过字段级准确率和端到端延迟比较六个LLM，最高准确率80%（Pixtral‑large‑2502）、标准差5%–7%，大模型平均延迟约20–21秒，GPT‑OSS‑20b异常高达54秒，整体性能稳定且可接受。

**⚠️ 局限性**

样本量有限、仅进行技术评估、未在真实MDTB环境中验证、缺乏细粒度错误分析与安全/可解释性评估。

---

## 472. DuwatBench: Bridging Language and Visual Heritage through an Arabic Calligraphy Benchmark for Multimodal Understanding

**arXiv ID:** 2601.19898 | [PDF](https://arxiv.org/pdf/2601.19898v1)

**作者:** Shubham Patle `[一作]` (Mohamed bin Zayed University of AI), Salman Khan `[通讯]` (Australian National University)

**通讯引用:** 11247 | [OpenAlex ID](https://openalex.org/A5000300751)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了名为DuwatBench的1.27K样本阿拉伯书法基准，涵盖六种主要书法风格并提供完整转写、风格标签与检测框；

**💡 创新点**

在传统书法数据集缺乏多样性与检测标注的空白中，首次集成多风格、全句转写、检测框与复杂艺术背景的综合评测资源；

**🔧 技术方法**

使用多模态模型评估技术，包括最新的LMM（如LLaVA、Gemma、Qwen、Gemini等）、OCR工具（EasyOCR、trocr）、以及阿拉伯专用模型，配合Unicode归一化和字符级/词级错误度量；

**📊 数据集**

主要数据集为自编的DuwatBench，包含约1,475个独特词汇，分布在Thuluth、Diwani、Kufic、Naskh、Ruq'ah、Nasta'liq六种书法风格；

**📈 对比分析**

通过CER、WER、chrF、ExactMatch和NLD五种指标对13个开源与闭源模型进行对比，结果显示Gemini-2.5-Flash与Gemma-3-27B-IT在多风格书法识别中表现最优，但整体准确率仍低于0.3；

**⚠️ 局限性**

限制在于样本量相对较小、对极其装饰化或罕见书法（如Kufic、Diwani）识别效果差、模型在多风格上的偏差显著，且未覆盖完整的文本排版与布局变形问题。

---

