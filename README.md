# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-04 | 今日论文总数: 758

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Effective Frontiers: A Unification of Neural Scaling Laws

**arXiv ID:** 2602.02593 | [PDF](https://arxiv.org/pdf/2602.02593v1)

**作者:** Jiaxuan Zou `[一作]` (Xi'an Jiaotong University), Yong Liu `[通讯]` (Renmin University of China)

**通讯引用:** 20371 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出了一个统一框架，即有效前沿（Effective Frontier）理论，用以解释神经网络在模型容量、数据规模和计算预算三种资源受限时的幂律缩放行为。

**💡 创新点**

创新点在于将学习过程抽象为对长尾（Zipfian）分布中模式频率排名空间的“前沿推进”，并将模型容量、数据覆盖与优化动态三种瓶颈统一为前沿的不同增长规律，从而衔接并解释Kaplan与Chinchilla两种看似矛盾的缩放法则。

**🔧 技术方法**

主要技术包括：离散可加模式分解、Zipfian 长尾假设、有效前沿定义与几何映射、可分离的容量、覆盖与优化瓶颈分析、PL 条件下的梯度下降动力学以及自相似缩放核（Self‑Similar Scaling Kernel）。

**📊 数据集**

作者在实验中使用了控制性合成数据集，构造不同尾指数α（如1.3–2.1）以验证理论；未使用公开大型语言或视觉数据集，实验重点在理论验证而非实际性能。

**📈 对比分析**

比较方法是将经验学习曲线与理论预测的幂律指数对齐；实验结果显示模型容量、数据规模和计算预算的可观测缩放指数与理论一致，验证了统一前沿理论的有效性。

**⚠️ 局限性**

局限在于假设数据分布的Zipf指数α和优化偏置β为固定不变的常数；未考虑真实数据中的多模态、梯度耦合、非独立模式及动态优化调度的影响。

---

## 2. FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment

**arXiv ID:** 2602.02680 | [PDF](https://arxiv.org/pdf/2602.02680v1)

**作者:** Riccardo Zaccone `[一作]` (Polytechnic of Turin), Samuel Horváth `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 931 | [OpenAlex ID](https://openalex.org/A5041080860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对预训练的大规模模型进行低秩分解，提取按重要性排序的嵌套子模型，实现一次训练多部署。

**💡 创新点**

提出基于层级低秩分解的嵌套子模型搜索与精调，并证明仅训练嵌套子模型即可逼近 Pareto 前沿。

**🔧 技术方法**

使用 DataSVD 对每层权重进行分解，动态规划寻找嵌套子模型，随后通过知识蒸馏进行精调。

**📊 数据集**

在 GPT‑2、Llama 系列、DINOv3 ViT 等模型上使用 FineWebEdu‑10BT、ImageNet1K 等数据集进行评估。

**📈 对比分析**

与传统 SVD、ASVD、A^3、DRONE、ACIP 等方法对比，FlexRank 在相同压缩率下保持更高准确率，压缩至 30% 参数仍接近原始性能。

**⚠️ 局限性**

需要更长训练周期或更大校准集，缺乏对输入自适应的支持，且对极低秩仍有性能下降。

---

## 3. Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization

**arXiv ID:** 2602.02958 | [PDF](https://arxiv.org/pdf/2602.02958v1)

**作者:** Haocheng Xi `[一作]` (University of California, Berkeley), Kurt Keutzer `[通讯]` (University of California, Berkeley)

**通讯引用:** 37906 | [OpenAlex ID](https://openalex.org/A5047285420)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了训练无关的量化框架 Quant VideoGen (QVG)，通过语义聚类和多阶段残差量化显著降低视频扩散模型的 KV‑Cache 内存占用，同时保持高质量长视频生成。

**💡 创新点**

核心创新在于利用视频时空冗余进行语义聚类以减小数值范围，并采用自上而下的多阶段量化策略，突破传统 LLM 量化方法在视频模型中的性能瓶颈。

**🔧 技术方法**

技术包括语义聚类 (K‑means)、残差中心化、分组量化 (per‑group symmetric quantization)、多阶段残差重构、缓存中心化、融合 dequant 与恢复的 CUDA/Triton 核心实现。

**📊 数据集**

在 LongCat‑Video‑13B、HY‑WorldPlay‑8B、Self‑Forcing‑Wan‑1.3B 三个公开模型上，用 MovieGen 基准的 prompt 进行 480p 长视频生成实验。

**📈 对比分析**

与 RTN、KIVI、QuaRot 等基线相比，QVG‑Pro 在 PSNR、LPIPS、SSIM 及 VBench 指标上几乎无损，同时实现最高 7.05× 的 KV‑Cache 压缩，内存占用下降 65% 以上，端到端延迟提升仅 1–4%。

**⚠️ 局限性**

局限性包括量化阶段数和聚类粒度需经验调参；在极低位宽 (INT2) 下仍可能出现细节失真；对模型的迁移性需进一步验证，且仅针对 KV‑Cache 进行压缩，未涉及权重或激活的量化。

---

## 4. HALT: Hallucination Assessment via Log-probs as Time series

**arXiv ID:** 2602.02888 | [PDF](https://arxiv.org/pdf/2602.02888v1)

**作者:** Ahmad Shapiro `[一作]` (Georgia Institute of Technology), Ashok Goel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8324 | [OpenAlex ID](https://openalex.org/A5007028896)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个仅利用LLM生成前20个token log‑prob 作为时间序列的轻量级幻觉检测模型 HALT，并发布覆盖10种能力的统一基准 HUB；

**💡 创新点**

将 log‑prob 序列视为时间序列学习模型校准偏差，实现仅使用黑盒 log‑prob，无需文本或内部状态；同时统一事实与推理错误的幻觉定义，构建跨能力基准；

**🔧 技术方法**

使用双向 GRU+top‑q pooling 结合熵、平均值、rank proxy 等特征，对 log‑prob 序列进行序列建模；对比白盒、统计、文本模型，评估宏 F1、AUROC 等指标；

**📊 数据集**

利用 HUB（整合 FAVA、RAGTruth、CriticBench 等，覆盖算法、常识、数学、符号、代码、聊天、数据‑文本、问答、摘要、世界知识等10类）以及 FAVA、RAGTruth 子集；

**📈 对比分析**

在 HUB 上与白盒 LLaMA‑Check、统计摘要、黑盒 Lettuce、文本模型对比，使用宏 F1、AUROC、F1；HALT 在 7/10 类获得最高分，整体宏 F1 约 67%，比 Lettuce、统计方法高，速度快 30×；

**⚠️ 局限性**

对不同 LLM 的跨模型迁移效果差，需针对每个模型重新训练；仅利用 log‑prob 可能无法捕捉文本语义细节；对极长文本或多语言场景的鲁棒性待进一步验证。

---

## 5. Step-Wise Refusal Dynamics in Autoregressive and Diffusion Language Models

**arXiv ID:** 2602.02600 | [PDF](https://arxiv.org/pdf/2602.02600v1)

**作者:** Eliron Rahimi `[一作]` (Technion), Chaim Baskin `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5019913171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了自回归与扩散语言模型的逐步拒绝动态，提出SRI信号并用于安全性解释和检测。

**💡 创新点**

将内部激活序列映射为逐步拒绝内部动力学SRI，证明采样机制对安全性影响显著，并基于SRI构建轻量级异常检测器。

**🔧 技术方法**

使用内部隐藏层平均池化、余弦距离原型、log‑ratio转换、sigmoid、内部恢复指标IRR、MLP自编码器异常检测等技术。

**📊 数据集**

使用Alpaca、AdvBench、WildJailbreak、JailbreakBench、HarmBench、Refined‑Prompts、LLaDA等多种干净与对抗数据集。

**📈 对比分析**

与LlamaGuard、Perplexity filter、Self‑Examine等基线对比，SRI Guard 在拒绝率、攻击成功率、误报率上匹配或优于，并将推理开销降低至约0.01%，比其他方法快150–300倍。

**⚠️ 局限性**

仅在目前评估的AR/DLM规模与架构上验证，SRI对极大模型或不同训练方法的迁移性可能有限，缺乏理论证明，需要进一步探索更广泛的生成模型。

---

## 6. Dynamic High-frequency Convolution for Infrared Small Target Detection

**arXiv ID:** 2602.02969 | [PDF](https://arxiv.org/pdf/2602.02969v1)

**作者:** Ruojing Li `[一作]` (National University of Defense Technology), Yingqian Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 7750 | [OpenAlex ID](https://openalex.org/A5052232868)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现一种动态高频卷积（DHiF），可直接替换标准卷积以提升红外小目标检测性能。

**💡 创新点**

通过局部特征驱动动态生成零中心高频滤波器，实现对高频成分的显式差异化建模，并保持与网络的无缝集成。

**🔧 技术方法**

采用动态卷积生成器、tanh映射、Fourier变换原理、标准卷积融合以及深度学习框架。

**📊 数据集**

在真实场景红外数据集IRSDT‑1k和NUAA‑SIRST上进行评估。

**📈 对比分析**

在多种SIRST网络（FC3Net、DNANet、ISNet、MSHNet、MTU‑Net、SCTransNet、APTNet）中替换标准卷积，与CDC、WTConv、PConv等方法对比，实验表明DHiF在IoU、P_d提升1–6个百分点，F_a下降3–5×10⁻⁵，性能优于其他卷积。

**⚠️ 局限性**

对输入层效果不佳，需经验性选择替换层；生成滤波器略增加计算开销，且在极端噪声或极小目标下仍有局限。

---

## 7. Bimanual High-Density EMG Control for In-Home Mobile Manipulation by a User with Quadriplegia

**arXiv ID:** 2602.02773 | [PDF](https://arxiv.org/pdf/2602.02773v1)

**作者:** Jehan Yang `[一作]` (Carnegie Mellon University), Doug Weber `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过面料集成的双臂高密度表面肌电（HDEMG）袖子，让一名四肢瘫痪用户在自己家中实时控制移动机械臂完成日常生活与个性化任务。

**💡 创新点**

创新点包括：①首个双臂纺织 HDEMG 袖子，实现残余神经运动的实时双手意图解码；②将该接口与共享自治模块（物体自动对齐、房间导航、LiDAR 冲突监测）结合，形成多模式的低维控制与高维机器人执行的闭环；③在真实家庭环境中进行为期 12 天的纵向部署，展示可穿戴接口的可用性与实用性。

**🔧 技术方法**

技术包括：高密度表面肌电传感（128 电极/臂），基于 RMS 的空间热图生成，轻量化 2 层 CNN 进行手势分类；共享自治采用基于目标检测（YOLO-E）+ 逆运动学对齐、Nav2 规划器+手势融合的房间导航、以及 LiDAR 距离阈值缩放的安全模块；语音指令用于切换模式与激活共享自治；所有系统通过 10 Hz 发送控制命令并通过网页界面反馈实时摄像头视图。

**📊 数据集**

使用自收集的日常 EMG 数据（每日 80 ms 窗格、256 通道、4 kHz 采样）及在家中完成的 11 种 ADL、工具使用与个性化任务的执行记录；未使用公开大规模数据集，而是为单一受试者量身定制数据集。

**📈 对比分析**

比较方法：与纯手势遥操作对比、共享自治开启前后任务完成时间、标准差、SUS 与 NASA‑TLX 主观评估。结果显示：共享自治下杯子饮用任务完成时间平均下降 ~55 s/天，标准差下降 2.3 倍；整体任务时长与 NASA‑TLX 工作量均在 12 天内持续下降，SUS 分数保持在 68 以上且在共享自治阶段更高，说明性能优于单纯遥操作且主观负担未增加。

**⚠️ 局限性**

局限性包括：仅单个受试者、实验周期有限（12 天），无法泛化至不同级别或不同类型的脊髓损伤；对感知系统的依赖导致误检与需要用户适应语言指令；EMG 信号易受袖子滑移、皮肤阻抗变化影响，需要频繁重新校准；共享自治模块对室内结构与灯光变化敏感。

---

## 8. Neural Probabilistic Amplitude Shaping for Nonlinear Fiber Channels

**arXiv ID:** 2602.02716 | [PDF](https://arxiv.org/pdf/2602.02716v1)

**作者:** Mohammad Taha Askari `[一作]` (University of British Columbia), Amirhossein Ghazisaeidi `[通讯]` (Nokia Bell Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于自回归神经网络的概率幅度调制方法（NPAS），能够在保持与传统 PAS 和系统化 FEC 兼容的同时学习符号序列的联合分布，从而降低光纤系统的非线性干扰并提升可达信息率。

**💡 创新点**

创新点在于：①使用自回归 RNN 仅在未符号（unsigned）域内学习联合分布，显著减少搜索空间，提升训练稳定性；②通过 Gumbel‑Softmax 等梯度可导采样技术实现端到端可微训练；③将学习到的分布与标准 ADM、FEC 并行，兼容现有 PAS 架构。

**🔧 技术方法**

所用技术包括：单层长短时记忆网络（LSTM） + 投影层、Gumbel‑Softmax 与 straight‑through 估计、可微光纤模型（AM 误差模型）以及匹配高斯解调器计算交叉熵损失。

**📊 数据集**

使用的数据集是仿真产生的单链路（205 km 单波段）光纤系统数据，包含 5 个双极化 64‑QAM 信道、50 Gb/s 带宽、CD 与非线性效应的完整模型。

**📈 对比分析**

实验与基准方法包括均匀调制、传统 PAS（ESS）及其序列选择版本、以及 NPS。NPAS 在 32‑符号块长度下，达到比 ESS 与 ESS+选择法高约 0.5 dB 的有效 SNR 与 0.1 bits/2D 的 AIR，并与 NPS 在高维度下保持相同的性能但更稳定；同时优化后的发射功率约提升 0.5 dB。

**⚠️ 局限性**

局限性主要是仅在单波段链路上验证，未评估多波段/多段链路；此外 ADM 的比特率损失与实际实现中的计算复杂度尚未量化，需要进一步实验验证。

---

## 9. Rethinking Music Captioning with Music Metadata LLMs

**arXiv ID:** 2602.03023 | [PDF](https://arxiv.org/pdf/2602.03023v1)

**作者:** Irmak Bukey `[一作]` (Adobe Research), Nicholas J. Bryan `[通讯]` (Adobe Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种先从音频预测结构化元数据再用LLM转化为音乐字幕的管道，能够在训练时只使用音频-元数据对。

**💡 创新点**

通过将元数据预测与后期LLM转换解耦，既实现了可后期风格化、元数据填充，又在保持与端到端模型相当的性能同时显著降低训练成本。

**🔧 技术方法**

采用文本预训练LLM（Gemma3）结合音频量化编码器的两阶段微调实现音频-元数据预测，以及使用同一LLM在推理时完成元数据-文本转换。

**📊 数据集**

使用内部授权的25k小时器乐音乐集及其对应元数据，外部的MusicCaps和Song Describer两套无声曲目字幕数据集进行训练与评估。

**📈 对比分析**

与同样使用元数据合成字幕训练的两种端到端captioner进行对比，元数据预测与字幕质量在SBERT相似度上相近，训练时间缩短约53%，并在跨风格与跨数据集评估中保持稳定。

**⚠️ 局限性**

受限于预训练LLM的语义偏好，元数据填充的精度仍低于人工标注；在细粒度乐器与节奏细节的控制上表现有限。

---

## 10. Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding

**arXiv ID:** 2602.02742 | [PDF](https://arxiv.org/pdf/2602.02742v1)

**作者:** Zihao Jing `[一作]` (Western University), Pingzhao Hu `[通讯]` (Western University)

**通讯引用:** 8038 | [OpenAlex ID](https://openalex.org/A5035024838)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Entropy‑Guided Dynamic Token Transformer（EDT‑Former），在冻结的图编码器与大型语言模型之间构建子结构感知的动态查询接口，实现在不更新LLM主体的情况下完成分子图与文本的跨模态对齐；

**💡 创新点**

创新点在于：①基于节点熵峰的动态分块策略，自动生成与子结构对应的查询Token；②将动态Token与固定模态锚点融合的跨模态接口；③采用冻结‑backbone训练方案，显著降低计算成本并提升泛化能力；

**🔧 技术方法**

技术包括熵驱动的图分块、动态Token Transformer、交叉注意力融合、跨模态对比损失、锚点匹配损失和子结构重构损失，以及两阶段冻结‑backbone训练流程；

**📊 数据集**

使用了Mol‑LLaMA‑Instruct、MoleculeQA、Mol‑Instructions、TDC（BBBP、HIA、PAMPA）和MoleculeNet等标准分子理解与性质预测数据集；

**📈 对比分析**

与现有多模态基线相比，EDT‑Former在MoleculeQA、Mol‑Instructions和属性预测任务上均达到了SOTA水平；在10‑shot GPT‑5对比中实现更优性能，同时相较于LoRA微调，显著减少3.5倍训练时间、内存消耗减半；

**⚠️ 局限性**

局限性包括对数据集不平衡仍易产生偏差、仅在已公开基准上验证，未评估对未见分子或跨域迁移的鲁棒性；并且依赖冻结编码器和LLM，可能在更大规模或新领域下的适配仍受限。

---

## 11. Joint Learning of Hierarchical Neural Options and Abstract World Model

**arXiv ID:** 2602.02799 | [PDF](https://arxiv.org/pdf/2602.02799v1)

**作者:** Wasu Top Piriyakulkij `[一作]` (Cornell University), Kevin Murphy `[通讯]` (Google DeepMind)

**通讯引用:** 37809 | [OpenAlex ID](https://openalex.org/A5044098731)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AgentOWL 通过联合学习抽象世界模型与层级化神经选项，实现样本高效的技能获取与组合

**💡 创新点**

将符号程序与非参数分布结合的 PoE-World 作为抽象世界模型，并使用 LLM 自动生成子目标与假设子选项，解决层级选项学习的样本效率瓶颈

**🔧 技术方法**

符号世界模型（PoE-World）、大型语言模型（LLM）用于子目标生成、深度 Q‑网络（DQN）与层级化 DQN、抽象状态抽象

**📊 数据集**

Object‑centric Atari（OCAtari）子集：Montezuma's Revenge、Pitfall、Private Eye，利用其对象解析结果作为符号输入

**📈 对比分析**

与 Rainbow DQN、Goal‑conditioned DQN、Hierarchical DQN 等基线对比；AgentOWL 在相同样本预算下掌握的选项数最多，尤其在高难度目标上表现显著优于基线

**⚠️ 局限性**

依赖预先按难度排序的目标序列、假设子目标需要手工或 LLM 生成、对大型计算资源要求较高、仅适用于符号输入的环境

---

## 12. AmharicStoryQA: A Multicultural Story Question Answering Benchmark in Amharic

**arXiv ID:** 2602.02774 | [PDF](https://arxiv.org/pdf/2602.02774v1)

**作者:** Israel Abebe Azime `[一作]` (Saarland University), Dietrich Klakow `[通讯]` (Saarland University)

**通讯引用:** 4507 | [OpenAlex ID](https://openalex.org/A5008875255)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了AmharicStoryQA，一个基于埃塞俄比亚九个地区文化多样性的长序列故事问答基准。

**💡 创新点**

首次在单一语言内考虑区域文化差异，揭示区域差异对模型叙事理解的影响，并证明文化根植训练能提升性能。

**🔧 技术方法**

采用多模态生成、人工评估、LoRA微调、结构化输出、SSA-COMET评估等技术。

**📊 数据集**

使用从埃塞俄比亚民间故事网页收集的244个故事（571训练+649测试）并翻译成阿姆哈拉语与英文。

**📈 对比分析**

对7个开源LLM进行零样本和微调评估，发现大多数模型在阿姆哈拉语MCQA上性能骤降，微调后提升约40% MCQA准确率。

**⚠️ 局限性**

仅覆盖阿姆哈拉语，未验证跨语言泛化，且数据收集与人工评估成本高。

---

## 13. CAPS: Unifying Attention, Recurrence, and Alignment in Transformer-based Time Series Forecasting

**arXiv ID:** 2602.02729 | [PDF](https://arxiv.org/pdf/2602.02729v1)

**作者:** Viresh Pati `[一作]` (Georgia Institute of Technology), Jiecheng Lu `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 CAPS（Clock-weighted Aggregation with Prefix-products and Softmax）——一种结构化注意力机制，用于时间序列预测，能够同时支持全局聚合、因果衰减和相位对齐。

**💡 创新点**

创新点在于将注意力的混合操作拆分为三条加性路径：Riemann softmax（全局聚合）、前缀乘积（因果衰减）和 Clock 基线（时序权重），并通过 SO(2) 旋转实现季节性相位对齐；该设计在保持线性时间复杂度的同时，解耦了传统 softmax 的全局归一化限制。

**🔧 技术方法**

技术包括：线性注意力框架、RoPE 相位旋转、Riemann softmax、前缀乘积门控、Clock 学习权重、三路加性融合以及可解释的权重分解。

**📊 数据集**

使用了十个公开多变量时间序列数据集，包括 Weather、Solar、Electricity、ETT（ETTh1/2/ETTm1/2）和交通流 PEMS（PEMS03/04/08）等，涵盖长期和短期预测场景。

**📈 对比分析**

与七个强基线（Olinear、TimeMixer++/TimeMixer、iTransformer、PatchTST、TimesNet、DLinear）及线性/softmax 变体比较，CAPS 在所有数据集平均排名 2.3、四次夺冠；相较于 LinAttn+RoPE 提升 6.1%（平均 MSE），并在 ablation 里每条路径都显著提升性能。

**⚠️ 局限性**

局限性包括：未探索大规模预训练与基础模型扩展，且在某些任务中仍受线性注意力的表示瓶颈限制；未来需评估其在更大规模、多任务或跨领域场景的可扩展性与鲁棒性。

---

## 14. Hypersonic Flow Control: Generalized Deep Reinforcement Learning for Hypersonic Intake Unstart Control under Uncertainty

**arXiv ID:** 2602.02531 | [PDF](https://arxiv.org/pdf/2602.02531v1)

**作者:** Trishit Mondal `[一作]`, Ameya D. Jagtap `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 5021 | [OpenAlex ID](https://openalex.org/A5061905151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过深度强化学习实现高超音速进气道失速的主动流场控制，成功抑制了失速现象；

**💡 创新点**

创新点在于结合第五阶谱离散DG CFD求解器与高频实时控制，实现零射击（zero‑shot）泛化，并在噪声环境下仍保持稳定；

**🔧 技术方法**

使用的技术包括高阶谱离散离散Galerkin（DG）CFD、强化学习算法SAC与TD3、基于SVD+QR的传感器优化以及多尺度自适应网格细化；

**📊 数据集**

数据集主要来自自研的高保真CFD仿真，覆盖多种后压（TR30/34/40/50）、雷诺数（5×10⁶、10×10⁶、15×10⁶）以及不同传感器噪声（0%/5%/10%）的时间序列；

**📈 对比分析**

与传统PID或基于模型的控制相比，SAC控制器在所有后压条件下均保持了接近基线的壁压、流量和压差，且在噪声和未见雷诺数下仍能抑制失速，显示出明显的性能优势；

**⚠️ 局限性**

主要限制包括极高的计算成本（需要高阶DG+AMR），控制更新频率高达50 kHz需专用硬件，且目前仅在二维模型验证，实际三维、实验或飞行环境下仍需进一步验证与优化。

---

## 15. OpenClaw Agents on Moltbook: Risky Instruction Sharing and Norm Enforcement in an Agent-Only Social Network

**arXiv ID:** 2602.02625 | [PDF](https://arxiv.org/pdf/2602.02625v1)

**作者:** Md Motaleb Hossen Manik `[一作]` (Rensselaer Polytechnic Institute), Ge Wang `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 41462 | [OpenAlex ID](https://openalex.org/A5100400458)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 OpenClaw 代理在 Moltbook 代理专用社交网络中的指令分享与社会监管行为，量化行动诱导语言比例并分析其对评论类型的影响。

**💡 创新点**

首次在无人工监管的代理社交平台上实证观察去中心化社会调节与规范化行为，提出并使用基于词典的 Action‑Inducing Risk Score（AIRS）以及规则匹配的评论回应分类。

**🔧 技术方法**

词典/规则基文本分析（AIRS 计算、关键词匹配分类），数据库查询与统计对比。

**📊 数据集**

Moltbook Observatory Archive 数据集，包含 14,490 名代理、39,026 条帖子、5,712 条评论，公开可下载。

**📈 对比分析**

通过比较行动诱导帖子与非行动诱导帖子的评论回应分布，发现行动诱导帖子更易获得规范执行（norm‑enforcement）回应，毒性回应极低；行动诱导比例为 18.4%。

**⚠️ 局限性**

仅基于文本语言，缺乏指令执行与结果追踪；样本为周期快照，缺少完整时间序列；规则基方法可能遗漏细微语境与讽刺等语用信息。

---

## 16. naPINN: Noise-Adaptive Physics-Informed Neural Networks for Recovering Physics from Corrupted Measurement

**arXiv ID:** 2602.02547 | [PDF](https://arxiv.org/pdf/2602.02547v1)

**作者:** Hankyeol Kim `[一作]` (Seoul National University), Pilsung Kang `[通讯]` (Seoul National University)

**通讯引用:** 4396 | [OpenAlex ID](https://openalex.org/A5059650940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种 Noise‑Adaptive Physics‑Informed Neural Network（naPINN），能够在未知噪声分布和严重离群值下从受损测量中恢复物理解。

**💡 创新点**

创新点在于：①将能量基模型（EBM）嵌入 PINN 训练，非参数学习残差分布；②设计可训练的可靠性门（gate）根据学习到的能量自适应下权重，从而实现在线离群检测；③通过拒绝成本正则化避免门退化为全部丢弃。

**🔧 技术方法**

技术手段包括：Physics‑Informed Neural Network、能量基模型（one‑dimensional EBM）、自适应权重门（sigmoid 函数）、指数移动平均（EMA）标准化、交替梯度更新、稳健损失对比（L1、Huber、q‑Gaussian、Bayesian PINN）。

**📊 数据集**

使用三类二维 PDE 基准：2D Burgers’ 方程、2D Allen‑Cahn 方程、λ–ω 反应‑扩散系统，测量数据由15×15传感器网格采样，并加入非高斯多模噪声与5%、10%、15%比例的随机离群点。

**📈 对比分析**

与标准 PINN、B‑PINN、LAD‑PINN、OrPINN（q=1.9/2.9）对比，naPINN 在所有噪声/离群水平下均取得最低的 rMAE/rMSE，误差仅略高于无噪声基准，提升幅度可达 30–50%。

**⚠️ 局限性**

局限性包括：①需要额外的 EBM 训练阶段与门参数调优，计算成本略高；②对极端离群率或极端噪声分布仍可能需要手动调整拒绝成本 λ_rej；③目前仅在二维平面 PDE 上验证，尚未探测更复杂高维或非平面系统。

---

## 17. Distance Marching for Generative Modeling

**arXiv ID:** 2602.02928 | [PDF](https://arxiv.org/pdf/2602.02928v1)

**作者:** Zimo Wang `[一作]`, Tzu-Mao Li `[通讯]` (University of California San Diego)

**通讯引用:** 2032 | [OpenAlex ID](https://openalex.org/A5030293104)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于距离场的时间无条件生成模型 Distance Marching，学习仅依赖当前样本的距离与方向预测，实现不需要时间输入的图像生成。

**💡 创新点**

创新点在于将距离场理论引入高维生成任务，设计了一步损失与方向Eikonal损失以减少目标模糊，同时提供梯度下降与球面追踪两种推理方式。

**🔧 技术方法**

采用距离场损失、方向Eikonal损失，并使用U‑Net或Transformer骨干，结合球面追踪/梯度下降采样。

**📊 数据集**

使用CIFAR‑10和ImageNet‑256数据集进行无条件与条件生成实验。

**📈 对比分析**

与现有时间无条件模型相比，平均 FID 降低 13.5%（CIFAR‑10）并在 ImageNet 上比流匹配模型低 9.5%–24.7%，在 60% 的采样步数即可达到流匹配最终水平。

**⚠️ 局限性**

局限在于高维距离场近似仍受近邻匹配偏差影响，且对极大模型规模需额外调参，未充分验证极少步生成或跨域迁移能力。

---

## 18. GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning

**arXiv ID:** 2602.02518 | [PDF](https://arxiv.org/pdf/2602.02518v1)

**作者:** Yuyang Bai `[一作]` (Texas A&M University), Yu Zhang `[通讯]` (Texas A&M University)

**通讯引用:** 34029 | [OpenAlex ID](https://openalex.org/A5115603610)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用强化学习框架训练大型语言模型，通过预定义的图功能调用在异构图中进行多轮推理与信息寻求。

**💡 创新点**

提出图结构感知的难度层级划分（S‑round、E‑round）与混合易到难采样的课程学习，帮助中等规模LLM在单域训练后实现跨域及 OOD 的图推理泛化。

**🔧 技术方法**

结合强化学习（PPO+KL正则化）、图功能调用API、规则化奖励（答案准确+交互格式）以及基于信息寻求轮数的结构难度指标，并使用混合采样调度。

**📊 数据集**

使用 GRBench 基准，仅在学术域进行训练，评测覆盖电商、文学、医疗和法律四个未见域，以及 OOD 问题。

**📈 对比分析**

与 TextRAG、GraphRAG、ToG‑2、Graph‑CoT、Vanilla RL 等基线在 ROUGE‑L 与 GPT4Score 上对比，3B 模型平均 ROUGE≈40.6、GPT4Score≈42.3，优于同类大模型基线，尤其在多轮推理和跨域硬题上表现显著。

**⚠️ 局限性**

受限于高成本 RL 微调、对工具接口噪声鲁棒性未知、超参数调优缺失、仍存在 Premature Stop 及格式误差，且未在真实世界图上评估安全与可控性。

---

## 19. MARA: Continuous SE(3)-Equivariant Attention for Molecular Force Fields

**arXiv ID:** 2602.02671 | [PDF](https://arxiv.org/pdf/2602.02671v1)

**作者:** Francesco Leonardi `[一作]` (Institute of Computer Science), Kaspar Riesen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可插拔的连续球面注意力模块（MARA），用于在分子力场模型中对局部原子环境进行角度和半径加权，从而改进能量和力的预测。

**💡 创新点**

创新点在于将连续球面注意力直接作用于分子邻域的角度与距离特征，提供了一种高效的SE(3)近似交互方式，且无需对现有模型结构做任何修改即可集成。

**🔧 技术方法**

核心技术包括：连续球面注意力算子、球面网格离散化与权重化、角度与半径的学习型位置编码、以及在MACE消息传递框架中的可插拔加权门控机制。

**📊 数据集**

在 rMD17 与 MD22 两大分子动力学基准数据集上进行评估，涵盖多种有机分子与较大、柔性体系。

**📈 对比分析**

与MACE基线以及NequIP、Allegro、BOTNet、UNiTE、QuinNet、VisNet等先进模型比较，MARA 在能量MAE平均下降约11%，力MAE平均下降约6%，并在多达10种分子中夺得能量或力预测的前列位置，且在高误差尾部表现出显著改进。

**⚠️ 局限性**

局限性包括：注意力仅为近似SE(3)不变性，需调节网格分辨率以权衡精度与计算成本；实验仅在MACE一类模型上验证，未在更广泛的网络结构和更大规模系统中进一步检验其通用性。

---

## 20. EEO-TFV: Escape-Explore Optimizer for Web-Scale Time-Series Forecasting and Vision Analysis

**arXiv ID:** 2602.02551 | [PDF](https://arxiv.org/pdf/2602.02551v1)

**作者:** Hua Wang `[一作]` (Ludong University), Fan Zhang `[通讯]` (Shandong Technology and Business University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种轻量化Transformer框架（EEO‑TFV）及其专用优化器Escape‑Explore Optimizer（EEO），用于Web规模多变量长序列预测和医学影像分割任务。

**💡 创新点**

创新点在于：1）在SAM基础上引入了“外部扰动”与“负曲率逃逸”两种自适应扰动机制，显著提升模型对尖锐极值和鞍点的逃逸能力；2）结合SGLD噪声与EMA平滑，实现全局探索与训练稳定性的统一；3）采用统一的标记化嵌入与轻量化通道注意力设计，降低参数量并减少熵坍塌/秩坍塌问题。

**🔧 技术方法**

使用技术包括：Transformer编码器、SAM式外部扰动、负曲率逃逸（Hessian‑vector近似）、SGLD随机梯度拉格朗日动力学、EMA参数平滑、以及标准的交叉熵/Dice/IoU等损失函数。

**📊 数据集**

数据集：11个Web时间序列基准（ETTh1/2、ETTm1/2、WTH、Traffic、ECL等），医学影像分割数据集（Polyp、Skin‑Lesion、Cell、Breast‑Cancer，Synapse多器官），以及公开的医学影像分割基准（ISIC、EM、DSB 等）。

**📈 对比分析**

与TimeMixer++、iTransformer、SAMFormer、TransUNet、SSFormer‑L、PVT‑CASCADE等SOTA方法做严格对比。时间序列预测中MSE/MAE均达到或优于基线；医学分割中Dice/IoU提升0.3–0.6%，HD95下降0.1–0.3；整体表现显示在长序列预测与边界保持上具有更好的泛化与稳定性。

**⚠️ 局限性**

局限性：1）实验主要集中在离线批处理任务，缺少实时在线/增量学习评估；2）EEO的计算与内存开销相比单纯AdamW略高；3）在极大规模多模态或持续漂移场景下的鲁棒性尚未充分验证；4）对超参数（扰动半径、负曲率阈值、温度等）敏感，需进一步自动化调优。

---

## 21. A Reduction from Delayed to Immediate Feedback for Online Convex Optimization with Improved Guarantees

**arXiv ID:** 2602.02634 | [PDF](https://arxiv.org/pdf/2602.02634v1)

**作者:** Alexander Ryabchenko `[一作]` (University of Toronto), Daniel M. Roy `[通讯]` (University of Toronto)

**通讯引用:** 4134 | [OpenAlex ID](https://openalex.org/A5110275739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

开发了基于连续时间模型的通用延迟反馈在线凸优化重构框架，将任意无延迟学习算法转换为可处理轮次依赖延迟的算法。

**💡 创新点**

核心创新是引入连续时间模型与观测排序视角，构造延迟驱动的漂移惩罚在线线性优化，将延迟学习问题化简为无延迟的漂移惩罚问题，并给出统一的 regret 分解与自适应缩减方案。

**🔧 技术方法**

主要技术包括连续时间模型、观测排序、漂移惩罚的在线线性优化、P-FTRL/OMD 协议、单点梯度估计以及跳过（skipping）技术。

**📊 数据集**

该工作为理论研究，未使用公开数据集，主要通过数学证明与理论分析展示效果。

**📈 对比分析**

与现有最优结果相比，论文将带阻反馈下的延迟项从 O(min{√(T d_max),(T d_tot)^{1/3}}) 降低至 O(√(d_tot))，在强凸性下延迟项从 O(d_max ln T) 降至 O(min{σ_max ln T, √(d_tot)})；整体 regret 与先前最优相当，但在延迟项上更优。

**⚠️ 局限性**

局限性包括仅针对可观测延迟的无适应对手（oblivious adversary）分析，未考虑延迟随学习者行为变化的自适应情况；模型假设延迟已知的“持续”结构，实际系统可能更复杂；缺乏实证实验验证。

---

## 22. Methods and Open Problems in Differentiable Social Choice: Learning Mechanisms, Decisions, and Alignment

**arXiv ID:** 2602.03003 | [PDF](https://arxiv.org/pdf/2602.03003v1)

**作者:** Zhiyu An `[一作]` (University of California), Wan Du `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并整合了将社会选择理论与机器学习结合的“可微分社会选择”范式，覆盖拍卖、投票规则、参与式预算、液态民主、AI 对齐等六大领域，并提出 36 个前沿开放问题；

**💡 创新点**

创新点在于将社会选择公理转化为可微分损失、架构约束与审计指标，构建一个统一框架，说明传统不可避免的局限性仍在，但以目标函数、约束与优化动态的形式出现；

**🔧 技术方法**

采用可微分经济学、神经社会选择、可微分优化层、图注意力网络、强化学习、对抗训练等多种技术手段，将投票规则和机制设计参数化为可微分网络，利用梯度下降和双层优化等方法学习决策机制；

**📊 数据集**

主要引用的实验数据来源包括合成与真实的投票/拍卖/预算数据、RLHF 对比标签、联邦学习客户端数据、LLM 交互日志等多样化数据集；

**📈 对比分析**

对比方法从传统解析机制、近似复制规则到自学习机制；在效率、比例、公平、鲁棒性等指标上，多数可微分学习方案在特定分布下取得与经典规则相当或更优的性能，但往往缺乏可解释性或对对抗/分布漂移的稳健性；

**⚠️ 局限性**

主要限制包括：缺乏严格的公理保证（软约束易失效）、泛化与对抗鲁棒性不足、动态激励与策略学习难以完全控制、缺少统一评估基准与审计工具，以及对多样化社会选择问题的理论边界尚不明晰。

---

## 23. A Classical Linear $λ$-Calculus based on Contraposition

**arXiv ID:** 2602.02822 | [PDF](https://arxiv.org/pdf/2602.02822v1)

**作者:** Pablo Barenbaum `[一作]` (National University of Quilmes), Leopoldo Lerena `[通讯]` (University of Buenos Aires)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出一种基于对偶否定（contra-substitution）的新型线性 λ‑计算机，构造了单结论的自然演算式，完整地为经典乘法指数线性逻辑（CMLL）提供了从命题到类型的解释。

**💡 创新点**

创新点包括：① 设计了“contra-substitution”操作，作为线性否定的对偶操作；② 通过该操作实现了线性 modus tollens 的计算语义；③ 在单结论系统中恢复了 CMLL 的对称性；④ 证明了该系统的安全性（soundness、completeness）、类型安全（subject reduction）、强规范化与致密性；⑤ 证明并模拟了多种已知的经典 λ‑计算机（Parigot 的 λμ、Curien-Herbelin 的 λμμ̃、Hasegawa 的 λ≺）。

**🔧 技术方法**

采用的技术主要是：线性 λ‑计算机、对偶否定的定义与证明、类型系统的构造、结构等价（cceq）与预归约（prereduction）等价的强同余证明、可归约性候选法证明强规范化、以及结构化的替换（线性替换、非线性替换、contra-substitution）和预归约规则的设计。

**📊 数据集**

论文没有使用任何实验数据集；所有结果均为形式化证明与理论推导。

**📈 对比分析**

由于缺乏实验数据，本文未与具体实现或性能指标进行对比；只在理论层面说明了与现有经典 λ‑计算机的兼容性与更优的计算属性（强规范化、致密性）。

**⚠️ 局限性**

局限性：① 目前仅覆盖乘法指数线性逻辑的乘法与指数；② 对偶否定与 contra-substitution 需要线性项的严格条件，难以直接扩展到加法或其他算子；③ 结构等价的定义较为复杂，尚未给出有效的自动化证明工具；④ 对于多项式/类型层次、第二阶量化、或递归/固定点等更高级逻辑的支持仍待研究。

---

## 24. High Rank Matrix Completion via Grassmannian Proxy Fusion

**arXiv ID:** 2602.02565 | [PDF](https://arxiv.org/pdf/2602.02565v1)

**作者:** Huanran Li `[一作]` (University of Wisconsin-Madison), Daniel Pimentel-Alarcón `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Grassmannian的代理子空间聚类方法，用以完成高秩矩阵缺失填补、列聚类与子空间识别。

**💡 创新点**

创新点在于同时最小化缺失向量与代理子空间的角度误差与子空间间的测地距离，并在Grassmannian上实现局部收敛保证，避免直接测量不完整向量距离。

**🔧 技术方法**

采用Grassmannian优化、曲率校正梯度下降、Chordal距离与Geodesic距离、Spectral clustering、SVD等技术。

**📊 数据集**

在Hopkins155、Smartphone AAL、MNIST以及印度松树、帕维亚大学、Salinas、Salinas A等高光谱数据集上进行实验。

**📈 对比分析**

与MC+SSC、EM、GSSC、MSC、ZF+SSC、SSC-EWZF等基准比较，GrassFusion在低采样率下显著优于其他方法，在大多数场景下保持低聚类误差。

**⚠️ 局限性**

主要限制是运行时间较长（每次约10–12小时），且对大规模数据需要进一步的采样或压缩处理。

---

## 25. SC3D: Dynamic and Differentiable Causal Discovery for Temporal and Instantaneous Graphs

**arXiv ID:** 2602.02830 | [PDF](https://arxiv.org/pdf/2602.02830v1)

**作者:** Sourajit Das `[一作]` (Pennsylvania State University), Romit Maulik `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文提出一种两阶段可微分框架SC3D，用于从多变量时间序列中同时学习滞后（lagged）和即时（instantaneous）的因果结构。

**💡 创新点**

创新点在于：①将节点级时间预筛选与全局稀疏化相结合，显著缩小搜索空间；②采用谱半径与2‑cycle惩罚的数值稳定的无环约束，仅对即时块施加，避免传统光滑无环约束的数值不稳定；③通过两阶段优化实现对滞后和即时结构的联合学习。

**🔧 技术方法**

技术手段包括：可微分结构学习、梯度优化、谱半径无环约束、2‑cycle惩罚、节点级时间窗口预筛选、稀疏正则化与渐进惩罚调度。

**📊 数据集**

实验数据主要为合成与基准动力学系统：Lorenz96、时间变结构方程模型(TVSEM)、非线性连续8变量(NC8)系统以及其他自定义高维稀疏/非线性VAR模型。

**📈 对比分析**

与DYNOTEARS、PCMCI+、VAR‑LiNGAM、NeuralGC等方法对比，SC3D在结构Hamming距离、AUROC/AUPRC、即时边恢复准确性等指标上均优于或相当于现有基线，且在维度扩展和滞后阶数增加时保持良好可扩展性与稳定性。

**⚠️ 局限性**

局限性包括：仅在合成/基准数据上验证，未在真实工业/科学数据中测试；对极大规模变量数仍有计算瓶颈；需要手动设置或调优预筛选阈值、惩罚调度等超参数；在非平稳或在线场景下的适用性尚待进一步研究。

---

## 26. Toward Ultra-Long-Horizon Sequential Model Editing

**arXiv ID:** 2602.02543 | [PDF](https://arxiv.org/pdf/2602.02543v1)

**作者:** Mingda Liu `[一作]` (Institute of Science Tokyo), Katsuki Fujisawa `[通讯]` (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种简单的“Norm-Anchor Scaling（NAS）”方法，用于在 Locate‑and‑Edit（L&E）框架下对大型语言模型进行连续编辑，并通过控制编辑权重范数来避免模型崩溃。

**💡 创新点**

创新点在于：①首次从机理上解释 L&E 更新导致的范数爆炸；②设计了一种无需额外训练、仅一行代码即可插拔的范数锚定策略，有效抑制范数指数增长；③在保持编辑精度的同时显著延长编辑寿命。

**🔧 技术方法**

技术主要包括：基于关键‑值解释的 FFN 更新公式、闭式一阶秩‑一更新、范数锚定（通过重新缩放目标值向量保持固定范数）以及对不同 L&E 规则的通用适配。

**📊 数据集**

实验使用了 CounterFact、ZsRE 以及 WikiBigEdit 三个事实编辑基准，评估了多种主干模型（Llama‑3‑8B、Qwen‑2.5‑7B、GPT‑J 等）。

**📈 对比分析**

与多种基线（包括非 L&E 的 FT、UltraEdit、RLEdit 以及 L&E 的 MEMIT、PRUNE、RECT、AlphaEdit 等）相比，NAS 能把编辑崩溃点延后超过 4 倍，编辑成功率平均提升 72% 以上，并且对模型的通用能力影响极小，几乎无额外开销。

**⚠️ 局限性**

局限性在于：NAS 采用固定的预编辑范数锚点，可能不适用于编辑难度变化或非平稳编辑流；未来工作需探索自适应或条件化的锚定机制，以进一步提升鲁棒性。

---

## 27. ViThinker: Active Vision-Language Reasoning via Dynamic Perceptual Querying

**arXiv ID:** 2602.02873 | [PDF](https://arxiv.org/pdf/2602.02873v1)

**作者:** Weihang You `[一作]` (University of Georgia), Hanqi Jiang `[通讯]` (University of Georgia)

**通讯引用:** 5750 | [OpenAlex ID](https://openalex.org/A5038582366)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ViThinker 框架，使视觉语言模型通过主动生成查询令牌，在推理过程中按需合成专家对齐的视觉特征，实现“思考-查询-模拟-思考”循环。

**💡 创新点**

创新点在于：1) 决策令牌与观测令牌分离，内部化多种视觉专家；2) 两阶段训练课程（先内部化专家，再学习何时查询）配合稀疏惩罚，鼓励模型在每一步只生成最少足够的感知；3) 完全无外部工具调用，靠模型自身的参数进行感知模拟。

**🔧 技术方法**

技术手段包括：决策/观测令牌机制、对齐损失（与冻结专家特征对齐）、稀疏惩罚、两阶段策略学习课程、投影头对齐、LoRA 微调、Qwen2.5-VL-7B 作为基准模型。

**📊 数据集**

训练集使用 LLaVA-OneVision、Filtered TallyQA、ADE20K-Depth 等；评估集覆盖 CV-Bench、BLINK、MMVP、RealWorldQA、MMStar-P、HRBench（HR_4K、HR_8K）等六个视觉基准。

**📈 对比分析**

与标准 VLM、文本 CoT、Visual CoT、ICoT、Aurora、CoVT、MINT-CoT 等基线对比，在六个视觉基准上平均提升 3–4%，在细粒度感知任务和高分辨率任务上表现尤为突出，尤其在 MMVP、BLINK 与 HRBench 上分别提升 1–2%。

**⚠️ 局限性**

局限性包括：① 依赖预冻结的视觉专家，缺乏对全新专家的可扩展性；② 稀疏惩罚系数需要手工调优；③ 推理时仍比纯文本 CoT 计算量大；④ 对实时动态场景的适应性尚未验证。

---

## 28. ATLAS : Adaptive Self-Evolutionary Research Agent with Task-Distributed Multi-LLM Supporters

**arXiv ID:** 2602.02709 | [PDF](https://arxiv.org/pdf/2602.02709v1)

**作者:** Ujin Jeon `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**通讯引用:** 6145 | [OpenAlex ID](https://openalex.org/A5078138445)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 ATLAS 框架，利用任务分配的多 LLM 支持者对自进化研究代理进行协同探索、调优与参考策略更新，从而实现长期自我提升。

**💡 创新点**

创新点在于（1）将自进化过程拆分为专门化的探索、策略控制和策略检查支持者，形成可扩展的多 LLM 结构；（2）引入 EvoDPO，采用阶段性 KL 约束的参考更新机制，缓解传统固定参考导致的偏离与停滞。

**🔧 技术方法**

技术包括偏好优化（DPO）、KL 正则化参考策略、阶段化自适应调参、基于梯度的评估和多代理协同调度，结合强化学习、主动搜索与自监督预训练。

**📊 数据集**

使用合成的非平稳上下文多臂赌博机数据集和一维 Burgers 方程的 PINN 训练数据，涵盖可变粘度等动态物理参数。

**📈 对比分析**

与固定参考的 EvoTune 基线以及仅有自适应参考的 EvoDPO 进行对比；在赌博机任务上获得 20.6% 的负平均回报提升，PINN 任务上验证损失下降 29,344 倍，显著优于基线。

**⚠️ 局限性**

局限性包括：1）对参考管理与安全阈值的精细调参要求高；2）在评估信号稀缺或嘈杂的真实任务中可能出现误匹配；3）对算力与数据量依赖大，扩展性和通用性待进一步验证。

---

## 29. WritePolicyBench: Benchmarking Memory Write Policies under Byte Budgets

**arXiv ID:** 2602.02574 | [PDF](https://arxiv.org/pdf/2602.02574v1)

**作者:** Edgard El Cham `[一作]` `[通讯]`, Edgard El Cham

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WritePolicyBench基准，用于评估在严格字节预算下的内存写入策略，重点研究在流式数据与API漂移环境中的写、合并与驱逐决策。

**💡 创新点**

创新点在于将写入决策与字节级成本模型、可观测轨迹与隐式标签分离，提供可重复的实验框架；同时引入了合并（MERGE）动作以及多种漂移生成器（默认、突发、冗余）。

**🔧 技术方法**

技术包括：基于Python的自定义动作接口（WRITE/MERGE/EXPIRE/SKIP）、精确字节计费模型、基于预算的K-贪心或DP背包算法评估最优基线、以及对实验结果的标准化度量（F1、预算利用率、平均陈旧度、偏差等）。

**📊 数据集**

使用完全合成的漂移流生成器，产生包含关键漂移事件的API快照序列；未公开真实数据集，所有实验均基于可复现的随机种子。

**📈 对比分析**

通过10个基线策略（包括无记忆、写满后停、最后k写、统一抽样、合并激进、以及使用优先级阈值/贪心的两条轨迹）在四个预算级别下与基准最优写入背包结果对照，报告了F1、预算利用率、回报/KB、以及回顾损失等指标。总体表现显示：在低预算下，优先级驱动策略能显著提升漂移覆盖率；合并策略在冗余环境中提升效率；但在高预算时，过多非关键写入导致精度下降，F1趋于下降。

**⚠️ 局限性**

局限包括：仅评估合成漂移流，缺乏真实生产日志与多文档依赖的复杂性；基线策略过于简单，未覆盖学习型写入模型；合并策略的实现细节可能导致不可预见的压缩失效；以及对实时更新与惰性加载等实际系统约束未做充分模拟。

---

## 30. Beyond Alignment: Expanding Reasoning Capacity via Manifold-Reshaping Policy Optimization

**arXiv ID:** 2602.02545 | [PDF](https://arxiv.org/pdf/2602.02545v1)

**作者:** Dayu Wang `[一作]` (Baidu Inc), Yang Li `[通讯]` (Baidu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过MRPO方法对大型语言模型进行几何重塑，突破对齐时低秩偏置流形的限制，显著提升推理能力。

**💡 创新点**

创新点在于将模型从偏置流形弹出（Spectral Orthogonal Exploration）并在强化学习中加入有效秩正则化，保持高维推理轨迹。

**🔧 技术方法**

核心技术包括Spectral Orthogonal Exploration (SOE) 作为冷启动数据合成，以及在GRPO中引入Effective Rank奖励的Rank-Aware Policy Optimization。

**📊 数据集**

实验数据集主要为数学推理题集：AIME 2024/2025、MATH-500、OlympiadBench、Omni-Math（高难度子集）。

**📈 对比分析**

与标准GRPO、更大规模参考模型以及开源RL模型比较，4B MRPO在AIME 2024上达56.7% Pass@1，显著高于32B模型（33.3%）和其他基线，整体性能大幅提升。

**⚠️ 局限性**

局限性包括长链推理易被截断导致在极长输入上表现下降，以及方法对有效秩估计与冷启动数据质量的高度依赖。

---

## 31. Framing Responsible Design of AI Mental Well-Being Support: AI as Primary Care, Nutritional Supplement, or Yoga Instructor?

**arXiv ID:** 2602.02740 | [PDF](https://arxiv.org/pdf/2602.02740v1)

**作者:** Ned Cooper `[一作]` (Cornell University), Qian Yang `[通讯]` (Cornell University)

**通讯引用:** 6611 | [OpenAlex ID](https://openalex.org/A5115596684)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过专家访谈、政策文档分析和三阶段研究，探讨如何负责任地设计非临床LLM工具以支持心理健康。

**💡 创新点**

提出了三条负责设计准则（具体收益、有效成分、风险收益匹配）并用药物、补充剂、瑜伽教师、初级保健等类比阐释，为LLM工具提供实用框架。

**🔧 技术方法**

定性访谈、主题分析、政策文件归纳与对比、专家共识验证。

**📊 数据集**

24名专家访谈记录、100多份美国监管与政策文件、行业案例等。

**📈 对比分析**

未进行实验性能比较，而是通过专家共识与政策对照构建负责设计模型，评估方法基于定性对话与文献分析。

**⚠️ 局限性**

样本局限于美国，专家组成偏向行业和学术；缺乏对工具实际效果的量化验证，框架需进一步实证检验。

---

## 32. Understanding Bug-Reproducing Tests: A First Empirical Study

**arXiv ID:** 2602.02965 | [PDF](https://arxiv.org/pdf/2602.02965v1)

**作者:** Andre Hora `[一作]` (Federal University of Minas Gerais), Gordon Fraser `[通讯]` (University of Passau)

**通讯引用:** 11289 | [OpenAlex ID](https://openalex.org/A5079261847)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对15个流行Python项目中的642条bug重现测试进行了系统实证研究，分析其代码特征与bug映射关系。

**💡 创新点**

首次从真实世界项目角度全面探讨bug重现测试的结构与行为，为后续测试生成与优化提供实证依据。

**🔧 技术方法**

采用文本挖掘提取测试中的关键词与Bug ID，统计LOC、断言数、控制流复杂度、try/except块等指标，并使用Mann‑Whitney U检验和Cohen效应量评估两类测试的差异；对断言进行词频与弱断言识别分析。

**📊 数据集**

数据集包含15个热门Python项目（共121,447条测试方法），从中提取642条被开发者在代码中显式标记为bug重现的测试。

**📈 对比分析**

与所有测试方法进行对比，发现bug重现测试在LOC、断言数与复杂度上无显著差异，try/except块和弱断言使用略高；95%测试对应单一Bug，5%对应多Bug，20%为共享Bug。

**⚠️ 局限性**

研究仅捕获开发者显式标记的测试，召回率有限；未考虑语义覆盖、运行时信息等更深层特征，对实际测试质量的全面评估不足。

---

## 33. To Defend Against Cyber Attacks, We Must Teach AI Agents to Hack

**arXiv ID:** 2602.02595 | [PDF](https://arxiv.org/pdf/2602.02595v1)

**作者:** Terry Yue Zhuo `[一作]` (Monash University), Ruijie Meng `[通讯]` (National University of Singapore)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5053022213)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文探讨了AI代理在网络安全中的攻击潜力，并提出将其用于构建防御情报的框架，主张通过构建综合攻击生命周期基准、从工作流向训练型代理的演进以及治理机制来实现安全防御。

**💡 创新点**

创新点在于把攻击视为可控实验，将AI攻击能力视作防御基础设施，并提出三步策略：全生命周期基准、训练型攻击代理和严格的治理边界，以期让防御方先手掌握攻击技术。

**🔧 技术方法**

使用的大型语言模型与AI代理技术，结合工具调用、强化学习和基于MITRE与Cyber Kill Chain的攻防生命周期评估，以及对现有安全基准的自动化对比。

**📊 数据集**

利用多种安全基准数据集，包括CyberSecEval-3、SeCodePLT、AutoPenBench、CVE-bench、CyBench、NYU、PrimeVul、VulnLLM、CyberGym、SEC-bench、SWE-bench-Verified等。

**📈 对比分析**

通过对上述基准进行分项评测，发现AI代理在小规模生成任务（如漏洞生成）表现优于大规模分析任务，整体性能参差不齐，说明当前模型在全生命周期评估上仍需提升。

**⚠️ 局限性**

局限性包括缺乏覆盖完整攻防生命周期的基准、对真实长尾系统的适应性不足、对抗性攻击鲁棒性不高以及治理机制在实际应用中的可行性尚未充分验证。

---

## 34. Tabula RASA: Exposing and Breaking the Relational Bottleneck in Transformers

**arXiv ID:** 2602.02834 | [PDF](https://arxiv.org/pdf/2602.02834v1)

**作者:** Jonas Petersen `[一作]` (University of Cambridge), Riccardo Maggioni `[通讯]` (ETH Zurich)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5119027652)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究Transformer在多跳关系推理中的局限，提出并验证一种最小改动的Relation‑Aware Sparse Attention (RASA) 架构。

**💡 创新点**

①通过电路复杂度证明Transformer需Ω(k)层完成k跳推理；②提出仅通过稀疏邻接掩码和关系类型偏置的改动，大幅缩小注意力搜索空间（从O(2^{n^2})降至O(2^m)），为图结构提供先验。

**🔧 技术方法**

稀疏邻接掩码、可学习的边类型偏置、基于Transformer的注意力机制；结合电路复杂度理论分析；实验使用DistilBERT编码器与轻量级GNN层。

**📊 数据集**

MetaQA知识图谱问答基准（1/2/3跳问题）。

**📈 对比分析**

与EmbedKGQA、NSM等SOTA进行Hits@1对比；在3跳问题上达到97.7% Hits@1，优于EmbedKGQA 94.8%（+2.9个百分点），在3跳上表现最佳，1/2跳略逊。

**⚠️ 局限性**

需要显式知识图结构输入；与标准Transformer同样的层数上限Ω(k)；优势主要体现在多跳推理上，1/2跳表现不佳；模型隐藏维度较小且未做KG预训练。

---

## 35. FIRE-Bench: Evaluating Agents on the Rediscovery of Scientific Insights

**arXiv ID:** 2602.02905 | [PDF](https://arxiv.org/pdf/2602.02905v1)

**作者:** Zhen Wang `[一作]` (UC San Diego), Zhiting Hu `[通讯]` (UC San Diego)

**通讯引用:** 6577 | [OpenAlex ID](https://openalex.org/A5085608858)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FIRE‑Bench基准，让大型语言模型代理在没有原论文细节的情况下，完整地从研究规划、实验设计、代码实现到结果分析和结论推导，重新发现已验证的机器学习实验洞察。

**💡 创新点**

创新点在于：①采用“受限重发现”任务，既避免直接复现又能评估完整科研链；②使用层次化研究问题树抽取方法将论文转化为可验证任务；③构建细粒度错误分析框架与任务难度分级，为模型改进提供诊断。

**🔧 技术方法**

技术手段包括：LLM驱动的多代理框架（OpenHands）、专有代理（Codex、Claude Code）及其前沿LLM后端（gpt‑4o、gpt‑4o‑mini、gpt‑4o‑2024‑08、Claude3.5 Sonnet）；自动化树抽取、LLM判别器进行claim级别抽取与匹配；工具链实现代码执行与实验运行。

**📊 数据集**

使用了30篇2024‑25年ICLR/ICML/NeurIPS顶会发表的LLM行为实证分析论文，每篇提供公开数据集和模型，任务继承原实验范围与评估指标。

**📈 对比分析**

评估方法为claim‑level precision/recall/F1；每个任务跑三次取平均。结果显示Claude Code最高F1≈46.7，Codex≈41.9，OpenHands(gpt‑4o)≈37.9，OpenHands(gpt‑4o‑mini)≈31.9，平均性能低、波动大。

**⚠️ 局限性**

局限性包括：①模型在实验设计和结论推理上的缺陷导致高方差与低可靠性；②主要失败集中在研究规划与结论形成；③受限重新发现难以鼓励创新发现；④受模型记忆或数据污染影响难以完全排除；⑤仅覆盖LLM研究领域，泛化性待扩展。

---

## 36. Refining Decision Boundaries In Anomaly Detection Using Similarity Search Within the Feature Space

**arXiv ID:** 2602.02925 | [PDF](https://arxiv.org/pdf/2602.02925v1)

**作者:** Sidahmed Benabderrahmane `[一作]` (New York University), Talal Rahwan `[通讯]` (New York University)

**通讯引用:** 4347 | [OpenAlex ID](https://openalex.org/A5007282319)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于稀疏双重对抗注意力自编码器（SDA²E）及其相似性引导的主动学习框架，用于在高度不平衡数据中高效检测稀有多样异常。

**💡 创新点**

创新点在于将稀疏正则、对抗训练与注意力机制融合生成紧凑判别表示，并通过相似性搜索的三种主动学习策略（正例扩展、异常优先和混合）实现对决策边界与排名的双重优化。

**🔧 技术方法**

技术主要包括稀疏自编码器、对抗生成网络、注意力权重、SIM_NM1相似度度量以及基于误差阈值的主动查询。

**📊 数据集**

使用了52个不平衡数据集，其中包括40个DARPA APT透明计算数据集、12个多领域公开数据集（如NSL‑KDD、CelebA等）。

**📈 对比分析**

与15种主流异常检测方法（ATDAD、TTVAE、AnoGAN、IForest等）进行nDCG比较，SDA²E在多数数据集上实现了显著的排名提升，标签成本可降低约80%。

**⚠️ 局限性**

局限性包括对阈值和相似度比例的依赖、对大规模实时数据的计算开销，以及在极端稀有类别下对主动学习策略的鲁棒性仍待进一步验证。

---

## 37. TabPFN for Zero-shot Parametric Engineering Design Generation

**arXiv ID:** 2602.02735 | [PDF](https://arxiv.org/pdf/2602.02735v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 38. From Tokens to Numbers: Continuous Number Modeling for SVG Generation

**arXiv ID:** 2602.02820 | [PDF](https://arxiv.org/pdf/2602.02820v1)

**作者:** Michael Ogezi `[一作]` (University of Waterloo), Ethan Smith `[通讯]` (Canva)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出连续数值建模（CNM）框架，用以改进向量图（SVG）生成，直接将SVG中的数值视为连续变量，并在此基础上利用强化学习进一步提升生成图像的感知质量。

**💡 创新点**

创新点在于：①使用 Fourier 特征映射 + MLP 对连续数值进行编码，消除传统 token‑化导致的碎片化与精度损失；②通过联合结构预测与数值回归的双任务学习，显著压缩序列长度；③采用多指标感知奖励（SSIM、LPIPS、DINOv2）的强化学习 fine‑tuning，提升视觉质量。

**🔧 技术方法**

技术手段包括：多模态 Transformer（Qwen2‑VL‑2B）与专门的 Number Encoder/Decoder、Fourier Feature Map、MLP、GRPO 强化学习、SSIM/LPIPS/DINOv2 评价指标，以及自定义的 SVGFloat 二进制存储格式。

**📊 数据集**

实验使用 SVG‑Stack 数据集（训练 2.17M、验证 108K、测试 5.71K），并与传统 VTracer、Potrace 以及基准模型 StarVector、OmniSVG 等进行对比。

**📈 对比分析**

比较方法：在相同模型规模与训练数据下进行基准对比；CNM 在 SSIM（54.1%）和 DINOv2（42.5%）等指标上超过 StarVector（48.5%/42.0%）和 OmniSVG，且序列长度从 1189 缩短至 549，训练时间降低 32%，推理时间降低 8%。

**⚠️ 局限性**

局限性包括：①对不同画布尺寸需要重新设定归一化常数 M；②双序列处理与 RL fine‑tuning 增加训练复杂度；③仅在 SVG‑Stack 上验证，跨域适应性与鲁棒性待进一步评估。

---

## 39. A Semi-Supervised Pipeline for Generalized Behavior Discovery from Animal-Borne Motion Time Series

**arXiv ID:** 2602.02618 | [PDF](https://arxiv.org/pdf/2602.02618v1)

**作者:** Fatemeh Karimi Nejadasl `[一作]` (University of Amsterdam), Eldar Rakhimberdiev `[通讯]` (University of Amsterdam)

**通讯引用:** 1799 | [OpenAlex ID](https://openalex.org/A5053557336)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套半监督的动物行为发现管线，先用少量已标注的数据学习嵌入表示，再通过标签引导的 K‑means 对已标注和未标注的短时间序列进行聚类，并使用 KDE+HDR 包含度得分判断聚类是否为新行为。

**💡 创新点**

创新点在于：① 在已有行为标签的前提下加入一个自由簇，既能聚类已知行为，又能捕捉潜在的新行为；② 设计了基于最高密度区域（HDR）的包含度得分，能够量化发现簇与已知行为分布的重叠度，提供可解释的“新颖性”阈值；③ 将该判定机制在发现与实际流式部署中保持不变，实现了端到端的自动化。

**🔧 技术方法**

使用技术包括：轻量级一维卷积网络进行时间序列嵌入，标签引导的半监督 K‑means 聚类，Gaussian KDE 估计 2D t‑SNE 投影中的密度，HDR 包含度计算与阈值判定；实验中还对比了不同预训练模型（MOMENT、Chronos2）与自监督对比学习（MAE、InfoNCE）的效果。

**📊 数据集**

数据集为 4,338 条短序列（长度 20，20 Hz）来自加拿大学者装在鸬鹚上的 IMU+GPS 传感器，包含 9 个专家标注的行为类别（Flap、ExFlap、Soar、Boat、Float、SitStand、TerLoco、Manouvre、Pecking），样本分布严重不平衡。

**📈 对比分析**

对比方法：在“已知-未知”与“负控制”两种实验设置下，半监督 K‑means 能够在 8/9 的被扣除行为中成功形成独立簇，且对应的包含度低于阈值 0.3，说明被扣除行为被识别为新行为；在负控制中额外簇的包含度始终高于 0.3，表明无误报。实验表明该方法在极少标注且类别不平衡的生态时间序列中能够稳健发现新行为。

**⚠️ 局限性**

局限性：① 嵌入质量是瓶颈，现有大模型预训练在短序列上效果欠佳；② KDE+HDR 只在低维 t‑SNE 投影上估计，可能对多模态或极小簇不稳定；③ 转换行为（如 Manouvre）因与其他飞行模式重叠导致包含度偏高，易被误判为已知行为；④ 需要人工校验簇是否真正具有生物学意义。

---

## 40. Scaling-Aware Adapter for Structure-Grounded LLM Reasoning

**arXiv ID:** 2602.02780 | [PDF](https://arxiv.org/pdf/2602.02780v1)

**作者:** Zihao Jing `[一作]` (Western University), Pingzhao Hu `[通讯]` (Western University)

**通讯引用:** 8038 | [OpenAlex ID](https://openalex.org/A5035024838)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出统一的全原子大语言模型，利用指令调控的可变大小图补丁与跨模态注意力，实现结构信息的动态量化与注入；

**💡 创新点**

创新点在于：①基于指令的锚点门控机制自适应生成补丁，突破固定长度查询瓶颈；②通过跨模态注意力提炼几何特征，显式注入LLM以降低结构幻觉；

**🔧 技术方法**

技术包括SE(3)等变图神经网络、指令调控锚点门控、软补丁生长、跨模态注意力与模块化注入适配器；

**📊 数据集**

使用Mol‑Instructions、DNA‑Chat、RNA‑QA以及作者构建的全原子指令数据集，涵盖分子、蛋白质与核酸三大原子级模态；

**📈 对比分析**

与通用LLM（Llama‑3.1、Qwen‑3等）及模态专用基线（Mol‑Llama、ChatNT等）对比，表现出在多任务上平均更高的准确率，且幻觉率显著下降；

**⚠️ 局限性**

局限性包括：对极大分子或复杂复合体的可扩展性尚未彻底验证，模型在极端结构多样性下的鲁棒性需进一步提升。

---

## 41. Gender Dynamics and Homophily in a Social Network of LLM Agents

**arXiv ID:** 2602.02606 | [PDF](https://arxiv.org/pdf/2602.02606v1)

**作者:** Faezeh Fadaei `[一作]` (University College Dublin), Taha Yasseri `[通讯]` (Trinity College Dublin)

**通讯引用:** 3386 | [OpenAlex ID](https://openalex.org/A5046908604)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了基于大型语言模型（LLM）的代理在 Chirper.ai 社交网络中的性别表现及其随时间的流动性，并分析了网络中的性别同质性、选择效应和社会影响。

**💡 创新点**

创新之处在于将性别视为动态文本表现而非固定标签，结合大规模真实交互数据，分别量化了新链接形成时的性别选择和随时间的社会影响，并通过 GPT‑4o‑mini 自动评估代理的性别得分。

**🔧 技术方法**

技术手段包括：GPT‑4o‑mini 零样本文本分类得到每周性别得分；网络分析中的标量 assortativity 与两类随机网络对照；可分时间 ERGM（STERGM）估计选择效应；面板回归（OLS 与 IV）检验社会影响；Python/R 生态完成数据处理、网络建模与可视化。

**📊 数据集**

数据集来源于 Chirper.ai 2023‑2024 年的公开数据：约 70,000 名代理、1.5 M 条原始推文（仅英文），过滤后 20,000 名代理和 800,000 条跟随边组成的动态跟随网络。

**📈 对比分析**

方法与比较：将经验 assortativity 与两类保留度数的随机网络 benchmark 对比，发现实际网络显著正值；STERGM 结果显示早期新连结偏向性别相似，后期趋于无效；面板回归（OLS 与 IV）在 33–48 周窗口内得到显著正的社会影响系数，表明后期代理会向其关注者的性别表现靠拢。

**⚠️ 局限性**

局限性包括：性别得分依赖 GPT‑4o‑mini，缺乏内部解释；仅研究单一平台与单一时间段；性别只以二元尺度测量，未考虑交叉身份；未深入分析帖子内容或话题对性别表现的驱动；模型仅关注二元连结与平均同行得分，忽略更复杂的网络结构和机制。

---

## 42. ProphetKV: User-Query-Driven Selective Recomputation for Efficient KV Cache Reuse in Retrieval-Augmented Generation

**arXiv ID:** 2602.02579 | [PDF](https://arxiv.org/pdf/2602.02579v1)

**作者:** Shihao Wang `[一作]` (Harbin Institute of Technology), Pengfei Wang `[通讯]` (Beijing Yanrong Technology Co)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于查询驱动的KV缓存重用方法ProphetKV，用于长文本检索增强生成（RAG）场景的预填充阶段加速；

**💡 创新点**

创新点在于：①利用用户查询的注意力信息“预言”哪些上下文token对答案生成最关键；②采用双阶段重计算管线，将查询注意力在所有层聚合后生成统一的重要性评分；③通过查询驱动的token选取避免了传统方法的“拥挤效应”；

**🔧 技术方法**

技术包括：位置无关（PI）KV缓存重用、查询到上下文的轻量级注意力计算、跨层注意力聚合（平均融合）、基于阈值的token重计算选择；

**📊 数据集**

在RULER（8k检索压力测试）和LongBench（推理与摘要任务）两大基准上评估；

**📈 对比分析**

与NaiveReuse、CacheBlend、KVShare、EPIC等SOTA方法对比，ProphetKV在20%重计算预算下，RULER准确率提升8.8%–24.9%，LongBench提升18.6%–50.9%；同时实现5×TTFT加速，保持低额外计算开销；

**⚠️ 局限性**

局限性包括：①仍需完整前向推理一次以获取查询注意力；②对极端长文本（>16k）或多任务场景的适应性未充分验证；③在极低预算（<5%）下的精度下降可能仍显著。

---

## 43. Self-Soupervision: Cooking Model Soups without Labels

**arXiv ID:** 2602.02890 | [PDF](https://arxiv.org/pdf/2602.02890v1)

**作者:** Anthony Fuller `[一作]` (Carleton University), Evan Shelhamer `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 Self‑Soupervision，通过在未标注数据上进行自监督训练（inter‑training）生成多样化的模型“配料”，随后在标注数据上微调并将这些配料平均混合成单一模型（soup），从而提升模型的泛化与鲁棒性。

**💡 创新点**

创新点包括：①将模型汤（Model Soup）从仅监督学习扩展到自监督学习；②在不同自监督损失、算法和超参数下生成多样配料，实现跨算法的线性模式连通性；③利用未标注的测试分布进行 inter‑training，显著提升对分布偏移的鲁棒性；④提出 Self‑Seasoning 方法在无监督条件下直接寻找混合系数。

**🔧 技术方法**

使用的技术主要有自监督学习算法（MAE、MoCoV3、MMCR）、多模型混合（平均或加权组合）、线性模式连通性验证、kNN 评估、无监督混合系数优化（entropy 最小化）。

**📊 数据集**

实验数据集包括 ImageNet‑1K（及其多种扰动：ImageNet‑C、LAION‑C、ImageNet‑V2、ImageNet‑A 等），以及 VTAB（21 个下游任务）及其加入噪声的 mini‑VTAB‑C。

**📈 对比分析**

与传统监督模型汤、持续自监督 + 监督汤等基线比较，Self‑Soups 在未标注分布训练时对 ImageNet‑C、LAION‑C 的 Top‑1 准确率提升约 3.5%~7%，在 ImageNet‑A 上提升 6.6%。在 VTAB 上对 21 个任务的平均性能略有提升，尤其对 blur、weather、digital 等扰动效果明显。

**⚠️ 局限性**

局限性包括：需要未标注的目标/偏移数据才能获得最大收益；增益相对温和（多数场景 ≤1%）；实验仅覆盖三类自监督算法，可能未完全体现所有算法的潜力；混合后仍需额外的微调步骤，且对极端分布迁移的适应性尚待进一步验证。

---

## 44. A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior

**arXiv ID:** 2602.02639 | [PDF](https://arxiv.org/pdf/2602.02639v1)

**作者:** Harry Mayne `[一作]` (University of Oxford), Noah Y. Siegel `[通讯]` (Google DeepMind)

**通讯引用:** 795 | [OpenAlex ID](https://openalex.org/A5065489996)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了LLM自我解释的可信度，提出并评估了一种新的可信度指标，衡量解释对模型行为预测的提升；

**💡 创新点**

创新点在于提出了可扩展、避免“vanishing signal”的Normalized Simulatability Gain（NSG）指标，并揭示自我解释具有特权知识优势；

**🔧 技术方法**

采用自然数据生成的多特征反事实集合，构建NSG指标，并用五种LLM作为预测器验证；

**📊 数据集**

使用七个流行的表格分类数据集（Heart Disease、Pima Diabetes、Breast Cancer Recurrence、Employee Attrition、Annual Income、Bank Marketing、Moral Machines）共计7000个问题–反事实对；

**📈 对比分析**

与外部模型生成的解释对比，发现自我解释平均提升NSG 3.8–10.8%（归一化后为11–36.5%），并在所有18种模型中表现一致；在规模上，Qwen3 与 Gemma3 系列呈上升趋势；

**⚠️ 局限性**

主要局限在于依赖反事实选择的质量、指标为平均案例且未提供极端情境下的保证，预测器能力有限，且可能出现评估意识问题。

---

## 45. WideSeek: Advancing Wide Research via Multi-Agent Scaling

**arXiv ID:** 2602.02636 | [PDF](https://arxiv.org/pdf/2602.02636v1)

**作者:** Ziyang Huang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Kang Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 WideSeekBench 基准并设计了 WideSeek 动态多智能体架构，实现宽度搜索（General Broad Information Seeking）任务的构造与求解。

**💡 创新点**

创新点包括：① 通过知识图谱的多阶段约束合成和任务生成，构建大规模、可验证的 GBIS 基准；② 采用可动态分叉的层次多智能体架构和统一的轨迹建模，使主规划器与子执行器在 RL 下协同进化，能够自动扩展搜索宽度。

**🔧 技术方法**

技术手段包括知识图谱（KG）数据提取、LLM 任务生成与自检、工具调用（ReAct）以及端到端强化学习（GRPO）联合优化主从智能体策略。

**📊 数据集**

使用的数据集为自建的 WideSeekBench，包含 5,156 个 GBIS 任务（4,436 个训练样本，720 个测试样本），任务来源于公开知识图谱并通过三阶段过滤保证质量。

**📈 对比分析**

通过与多款专有模型（GPT‑5.x、DeepSeek‑v3.2、Kimi‑K2 等）以及开源模型（Qwen3‑8B、Qwen3‑30B‑A3B）在 Success Rate、Row F1、Item F1 等指标上对比，WideSeek‑8B‑SFT‑RL 在 Item‑F1 12.87%、Row‑F1 3.88% 等指标上实现显著提升，特别在多任务并行检索方面明显优于传统单智能体方案。

**⚠️ 局限性**

局限性包括：① 受参数规模和检索深度限制，极大信息量任务易出现“早停”或拒绝；② 对知识图谱的覆盖和质量高度依赖，缺乏通用多源验证；③ 目前仅在 8B 模型规模上验证，尚未验证更大规模模型的可扩展性。

---

## 46. Expert-Data Alignment Governs Generation Quality in Decentralized Diffusion Models

**arXiv ID:** 2602.02685 | [PDF](https://arxiv.org/pdf/2602.02685v1)

**作者:** Marcos Villagra `[一作]` (Bagel Labs), Zhiying Jiang `[通讯]` (Bagel Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对去中心化扩散模型（DDM）中路由策略进行系统实验，验证专家-数据对齐决定生成质量并否定轨迹敏感度的主导作用。

**💡 创新点**

提出专家-数据对齐原则并证明稳定性-质量解耦；通过簇距离、专家预测一致性和专家分歧等多维度实验验证该原则。

**🔧 技术方法**

使用去中心化扩散模型、Top‑k稀疏路由、全量路由、局部Lipschitz分析、L_eff 与 Δ_refine 诊断、角度偏差度量、LPIPS 与 FID 评估等技术。

**📊 数据集**

主要使用 LAION‑Aesthetics（Paris DDM）和 MNIST（验证实验）数据集，并在每个数据集上训练独立专家。

**📈 对比分析**

与全量路由、Top‑1、Monolithic 等策略对比：Top‑2 路由在 FID（22.6）和 Δ_refine（0.051）上优于全量路由（47.9、0.020），但在轨迹敏感度上并非最优；显示数值稳定性与生成质量无直接相关性。

**⚠️ 局限性**

局限在于仅使用预训练模型进行推理实验、未考虑模型在不同域的普适性、L_eff 为后验诊断且不保证收敛、数据集存在偏见、未研究联合训练与动态路由等方面。

---

## 47. GASTON: Graph-Aware Social Transformer for Online Networks

**arXiv ID:** 2602.02524 | [PDF](https://arxiv.org/pdf/2602.02524v1)

**作者:** Olha Wloch `[一作]` (University of Waterloo), Lukasz Golab `[通讯]` (University of Waterloo)

**通讯引用:** 4912 | [OpenAlex ID](https://openalex.org/A5049437648)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 GASTON，一种将社区视为基础上下文实体的异构图 Transformer，用对比初始化的社区嵌入来捕获社区用户共处模式，并将文本与社交结构结合用于下游任务。

**💡 创新点**

创新点在于：① 将社区从文本内容中独立出来，作为可学习实体；② 采用基于 BPR 的对比初始化，先学习社区的结构签名；③ 在预训练阶段结合文本重建与边预测，使社区嵌入兼具结构与语义信息；④ 使用 EmbeddingGemma 作为文本编码器，提升语义表达。

**🔧 技术方法**

核心技术包括异构图 Transformer（HGT）、对比 BPR 初始化、文本重建与边生成的自监督预训练、动态图采样、以及微调时的分类/回归/边预测头。

**📊 数据集**

使用 Reddit 历史档案（约 6.4M 文本、6132 个社区、4.7M 用户）做预训练；下游任务数据集为 Dreaddit（情绪检测）、Ruddit（毒性评分）、NormVio（规范违规）、HatefulDiscussions（仇恨言论），以及在 Reddit 上的社区推荐任务。

**📈 对比分析**

与 BERT‑only、BERT+Community 以及 OMCA 进行对比。GASTON 在 Norm Violation、Hate Speech Detection 和社区推荐任务上取得显著提升（最高 0.971 F1、0.971 F1、0.055 NDCG@10），在小样本任务（Ruddit、Dreaddit）与 BERT 基线相近或略逊，但仍优于 OMCA；整体表现位居榜首。

**⚠️ 局限性**

局限性包括：① 在样本稀缺或社区交互极度稀疏时容易过拟合，导致性能不如文本模型；② 推荐任务因用户与社区交互稀疏而指标偏低；③ 当前仅处理文本，未考虑图像/视频等多模态信息；④ 对极端社区的“共振”仍可能产生偏差，需要更细粒度的边类型和回复树结构来提升鲁棒性。

---

## 48. ToolTok: Tool Tokenization for Efficient and Generalizable GUI Agents

**arXiv ID:** 2602.02548 | [PDF](https://arxiv.org/pdf/2602.02548v1)

**作者:** Xiaoce Wang `[一作]` (Tsinghua University), Ming Li `[通讯]` (National University of Singapore)

**通讯引用:** 65981 | [OpenAlex ID](https://openalex.org/A5100695826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于工具令牌的多步视觉路径规划框架，取代传统的坐标回归，允许 GUI 代理以离散、可解释的工具操作完成交互。

**💡 创新点**

创新点包括：①工具令牌化与多步路径规划；②球面语义锚定（SSI）初始化新工具令牌；③三阶段从语义对齐到简易路径再到真实轨迹的课程学习；④通过 CoT 生成与动作加权损失实现高效学习。

**🔧 技术方法**

技术手段涵盖：预训练视觉‑语言模型（Qwen3‑VL‑4B），球面语义初始化，离散工具词表与层次化移动令牌，强化学习式轨迹合成，CoT 推理，动作加权损失，分阶段训练流程。

**📊 数据集**

主要使用数据集：ScreenSpot、ScreenSpot‑Pro、Mind2Web‑S、ScreenSpot‑v2（ID/OD），以及约 5K 合成语义‑路径样本与少量真实轨迹，训练样本量仅占传统方法 1% 以内。

**📈 对比分析**

与 4B/8B/235B Qwen3 系列、Holo2‑4B、GUI‑Actor‑7B 等基线对比，ToolTok 在所有 ID/OOD 任务中均显著超越同规模模型，逼近 235B 大模型；在数据效率上实现约 500× 的提升。

**⚠️ 局限性**

局限性：仍依赖预训练模型的视觉与语言先验；对极端分辨率/纵横比、长句复杂指令或高度动态交互的鲁棒性有限；以及对连续或高频率交互动作的支持仍待完善。

---

## 49. FedKRSO: Communication and Memory Efficient Federated Fine-Tuning of Large Language Models

**arXiv ID:** 2602.03019 | [PDF](https://arxiv.org/pdf/2602.03019v1)

**作者:** Guohao Yang `[一作]` (University of Texas at San Antonio), Yanmin Gong `[通讯]` (University of Texas at San Antonio)

**通讯引用:** 2838 | [OpenAlex ID](https://openalex.org/A5035287134)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在联邦学习环境下对大型语言模型进行高效微调

**💡 创新点**

提出基于有限随机种子子空间优化的FedKRSO方法，压缩梯度并仅上传累积更新

**🔧 技术方法**

随机子空间投影、梯度压缩、AdamW优化器、子空间累积器

**📊 数据集**

GLUE基准（8项任务）与RoBERTa-base/large模型

**📈 对比分析**

与FedFFT、FedIT和FFA-LoRA对比，FedKRSO在保持内存≈2.7GB、通信≈0.03GB的同时，性能接近FFT且优于LoRA基线

**⚠️ 局限性**

目前仅在单子空间、单间隔设定下验证，缺乏多种子/多间隔理论收敛保证，未验证多模态或更大规模模型

---

## 50. Controlled disagreement improves generalization in decentralized training

**arXiv ID:** 2602.02899 | [PDF](https://arxiv.org/pdf/2602.02899v1)

**作者:** Zesen Wang `[一作]` (KTH Royal Institute of Technology), Mikael Johansson `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 11856 | [OpenAlex ID](https://openalex.org/A5024256519)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DSGD-AC算法，在去中心化SGD中引入时间可变的自适应共识因子，以有意维持非消退的共识误差，从而提升模型泛化性能。

**💡 创新点**

创新点在于将共识误差视为结构化扰动并与海森矩阵主子空间对齐，形成类似SAM的平坦度正则化；并通过理论证明和实验验证该机制能在保持通信效率的同时实现更好的泛化。

**🔧 技术方法**

采用拉普拉斯矩阵分解、共识误差投影分析、理论稳定性证明；实现自适应共识机制并在去中心化训练框架中集成；使用宽残差网络、Transformer等模型进行实验。

**📊 数据集**

主要使用CIFAR-10和CIFAR-100数据集进行图像分类实验，补充使用WMT-14数据集进行Transformer机器翻译实验。

**📈 对比分析**

与同步SGD和标准DSGD在多种节点数、拓扑（环、指数、完全图）下进行对比；实验结果显示DSGD-AC在测试准确率、测试损失和主Hessian特征值上均优于两者，且训练时间与DSGD相当甚至更短。

**⚠️ 局限性**

局限性包括：需手动调节超参数p和E_start；理论假设局部凸性，实际大规模网络可能不完全满足；对非i.i.d.数据分布、极其稀疏网络拓扑的适应性尚未充分验证。

---

## 51. Label Curation Using Agentic AI

**arXiv ID:** 2602.02564 | [PDF](https://arxiv.org/pdf/2602.02564v1)

**作者:** Subhodeep Ghosh `[一作]` (New Jersey Institute of Technology), Senjuti Basu Roy `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 1990 | [OpenAlex ID](https://openalex.org/A5009377962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出AURA，一种基于多智能体的自动化数据标注框架，利用多种离线AI模型联合生成和校验标签，避免人工标注成本；

**💡 创新点**

创新点在于将经典的Dawid‑Skene概率模型与期望‑最大化算法结合，构建Agentic Expectation Maximization Labeling（AEML）实现对标注者可靠性与真实标签的联合推断，并完全不依赖任务特定微调或上下文学习；

**🔧 技术方法**

使用期望‑最大化（EM）算法、混淆矩阵可靠性建模、概率标签聚合；同时采用多模态离线AI模型（LLM、视觉模型等）作为标注者；

**📊 数据集**

在四个真实多模态数据集上验证：Kinetics‑400（视频）、ImageNet‑ReaL（图像）、Food‑101（图像）、CUB‑200（图像）；

**📈 对比分析**

与多数投票（Majority Voting）基线对比，AURA在所有数据集上均取得显著提升，精度提高2.4%–5.8%，在低质量标注者情形下提升可达50%；

**⚠️ 局限性**

局限在于需要足够多的多样化标注者以保证可靠性估计；对极少量标注者或极高噪声情况仍可能面临收敛困难；

---

## 52. CodeGuard: Improving LLM Guardrails in CS Education

**arXiv ID:** 2602.02509 | [PDF](https://arxiv.org/pdf/2602.02509v1)

**作者:** Nishat Raihan `[一作]` (George Mason University), Marcos Zampieri `[通讯]` (George Mason University)

**通讯引用:** 6382 | [OpenAlex ID](https://openalex.org/A5024937008)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对计算机科学教育场景，构建了完整的 LLM 安全防护框架，包括用于标注安全/不安全提问的细粒度分类法、覆盖 8,000 条提示的专用数据集以及一个轻量级的句子编码器模型 PromptShield，用于实时检测和拦截不安全或无关的提示。

**💡 创新点**

① 设计了专门针对 CS 教育的多维度提示分类法；② 收集并标注了 8,000 条安全/不安全提示的基准数据集；③ 在 RoBERTa‑base 上进行细粒度分类微调，提出 PromptShield；④ 通过实验验证 PromptShield 在安全提示检测上的 F1 得分为 0.93，较现有 LLM（如 GPT‑4o、Claude 3.7、LLaMA Guard 等）提升 30–65%；⑤ 证明安全微调不会显著影响模型的编程能力。

**🔧 技术方法**

采用的技术包括：基于教师专业知识的 Delphi 迭代式分类法构建；使用多源数据集（Alpaca‑Instruct、CSEPrompts 等）进行提示采样；利用 BERTScore 进行近似重复过滤；使用 RoBERTa‑base encoder 进行文本分类微调；混合精度训练、梯度检查点、AdamW 优化器；在训练集 6,000 条提示上实现三轮训练，最终获得 PromptShield。

**📊 数据集**

主要使用的公开数据集包括：Alpaca‑Instruct、LaMini‑Instruct、Infinity‑Instruct（用于无关提示）；CSEPrompts、StudentEval、Evol‑Instruction（用于安全提示）；以及通过两大无审查 LLM（DeepSeek R1、Dolphin‑3）生成的 2,000 条不安全提示；整合后构成 8,000 条标注提示的 CodeGuard 数据集。公开链接：https://huggingface.co/datasets/md‑nishat‑008/Do‑Not‑Code。模型可下载：https://huggingface.co/md‑nishat‑008/PromptShield。

**📈 对比分析**

对比方法：将 PromptShield 与多种基线模型（OpenAI GPT‑4o、Anthropic Claude 3.7、LLaMA Guard、Nemo Guard、Google Gemma 3、MistralAI Magistral、RoBERTa、BERT、SVC、LR、Perspective API、随机基线）在 1,000 条测试提示上进行 F1 评估；结果显示 PromptShield 的 F1 0.93，显著高于所有基线（最高仅 0.60）；实验还表明，在对 LLM 进行安全微调后，模型在 HumanEval、MBPP、CodeWorkout、IntroClass 四个编程基准上的 Pass@1 几乎保持不变，并且 PromptShield 能将生成的危险代码比例降低 30–65%。

**⚠️ 局限性**

局限性：① 分类法基于有限数量的 CS 课程教材和教师反馈，可能不适用于更高级或其他学科的教育场景；② 数据集的自动标注过程仍存在一定误差，尽管后期人工校正准确率高达 97.4%；③ 评估仅在模拟数据上进行，尚未在真实课堂环境中验证部署效果；④ 仅关注安全/无关提示，未覆盖更细粒度的违规行为或多模态交互。

---

## 53. Formulating Reinforcement Learning for Human-Robot Collaboration through Off-Policy Evaluation

**arXiv ID:** 2602.02530 | [PDF](https://arxiv.org/pdf/2602.02530v1)

**作者:** Saurav Singh `[一作]`, Jamison Heard `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用离线强化学习和离线数据通过离线策略评估（OPE）自动化地选择最优状态空间和奖励函数，从而实现人机协作中的安全、可靠的RL策略设计。

**💡 创新点**

创新点在于将OPE从仅评估策略扩展到指导状态和奖励函数的设计，使RL可在无在线交互的前提下完成关键设计决策，并将人体生理信号嵌入状态，实现个性化人机协作。

**🔧 技术方法**

主要技术包括：离线强化学习（CQL、DDQN等）、多种OPE方法（Importance Sampling、Direct Method、Fitted Q Evaluation、Doubly Robust）、基于多模态数据的特征工程以及奖励函数可分辨性评估。

**📊 数据集**

使用的数据集：OpenAI Gym Lunar Lander（模拟环境）和NASA‑MATB‑II人机实验数据（任务指标、心率、呼吸、工作负荷等多模态信息）。

**📈 对比分析**

方法通过在不同状态/奖励候选上训练离线策略并使用OPE评估其预期回报来进行比较；实验结果显示OPE选择的状态空间和奖励函数能够显著提升策略回报，RL代理在NASA‑MATB‑II实验中在跟踪、系统监控、通信等任务上均优于基于规则和微调的代理，且在信任、工作负荷和流畅度评估中表现最佳。

**⚠️ 局限性**

局限性包括：OPE对数据分布漂移敏感，奖励函数选择仍需人工定义；微调RL在高负荷时可能引入不确定性并增加用户工作负荷；缺乏可解释性机制，难以解释决策过程，影响用户信任。

---

## 54. The Alignment Curse: Cross-Modality Jailbreak Transfer in Omni-Models

**arXiv ID:** 2602.02557 | [PDF](https://arxiv.org/pdf/2602.02557v1)

**作者:** Yupeng Chen `[一作]`, Adel Bibi `[通讯]` (University of Oxford)

**通讯引用:** 58462 | [OpenAlex ID](https://openalex.org/A5042899882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了文本 jailbreak 攻击如何通过文本转语音（TTS）传播到音频输入，探讨了跨模态攻击的“对齐诅咒”现象，并在多种 omni‑model 上进行实验。

**💡 创新点**

提出“对齐诅咒”概念：强跨模态表示对齐会让文本中的安全漏洞直接在音频中复现；给出了该现象的理论解释和 KL 散度阈值；并展示文本转语音攻击能成为简单而强大的音频红队基线。

**🔧 技术方法**

使用了端到端 omni‑model 的统一表示学习、文本转语音合成（OpenAI gpt‑4o‑mini‑tts）、多种已知文本 jailbreak（PAP、ReNeLLM、AutoDAN‑Turbo）以及音频专用攻击（VoiceJailbreak、SSJ、Speech Editing、Dialogue Attack），并对模型输出做关键字匹配和 StrongReject 评分。

**📊 数据集**

数据集采用 JailbreakBench（包含 100 个误用行为及 HarmBench、TDC 等子集）进行攻击评估。

**📈 对比分析**

与传统音频攻击对比，文本转语音攻击在 SR（语义有害程度）上往往优于或与之相当；在跨模型迁移实验中，PAP、AutoDAN‑Turbo 的音频版本平均 SR 分别达到 0.71‑0.58，明显高于 Speech Editing、VoiceJailbreak；此外在严格的音频‑only 威胁模型下仍保持较高成功率。

**⚠️ 局限性**

局限性包括：并非所有文本攻击都能跨模态传播（如 ReNeLLM 失败），对齐程度不足时攻击效果下降；实验仅覆盖部分 omni‑model 与 TTS 设定；未深入探讨音频专用防御或更复杂的攻击形式。

---

## 55. Measuring Individual User Fairness with User Similarity and Effectiveness Disparity

**arXiv ID:** 2602.02516 | [PDF](https://arxiv.org/pdf/2602.02516v1)

**作者:** Theresia Veronika Rampisela `[一作]` (University of Copenhagen), Christina Lioma `[通讯]` (University of Copenhagen)

**通讯引用:** 2732 | [OpenAlex ID](https://openalex.org/A5045425016)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于用户相似度与推荐效果差异的个体公平度量指标 Pairwise User Unfairness（PUF），并在四个真实数据集上进行实验评估。

**💡 创新点**

创新点在于同时考虑用户相似度和推荐效果差异，填补了现有指标仅关注单一维度的空白，并通过加权对用户相似度实现更细粒度的公平度量。

**🔧 技术方法**

采用了余弦/杰卡德相似度、Precision@k、NDCG@k 等传统评价指标，并将其组合成 PUF；实验中使用了 7 种协同过滤/深度模型进行推荐。

**📊 数据集**

使用的四个数据集分别是 Lastfm、QK‑video、MovieLens‑10M 与 MovieLens‑20M。

**📈 对比分析**

与现有的标准差、Gini、Envy 及 UF 等公平指标对比，PUF 对效果与相似度变化更敏感，计算速度快（<40s），并能正确识别极端公平/不公平场景；但其与效果指标的排名显著不一致。

**⚠️ 局限性**

局限性包括仅适用于无属性的个体公平评估、未考虑梯度相关性或用户属性、仅评估四个数据集且仅使用二元相关性，可能在冷启动或多属性场景下表现不足。

---

## 56. Recurrent Equivariant Constraint Modulation: Learning Per-Layer Symmetry Relaxation from Data

**arXiv ID:** 2602.02853 | [PDF](https://arxiv.org/pdf/2602.02853v1)

**作者:** Stefanos Pertigkiozoglou `[一作]` (University of Pennsylvania), Kostas Daniilidis `[通讯]` (University of Pennsylvania)

**通讯引用:** 17734 | [OpenAlex ID](https://openalex.org/A5050660826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种自适应的层级等变约束调制框架RECM，能够在训练过程中自动学习各层的等变约束松弛程度，且不需要先验的松弛策略或超参数；

**💡 创新点**

通过构造可递归更新的状态向量和约束调制规则，理论证明当输入-目标分布完全对称时，各层会收敛为完全等变模型；在非对称分布下则保持足够的自由度学习近似等变解；

**🔧 技术方法**

使用可分离的等变与非等变子网络线性组合、递归状态更新（类似RNN），并利用Lipschitz连续的更新函数与Haar测度的对称投影；实现中用GeLU、MLP等；

**📊 数据集**

在多个基准任务上评估：ModelNet40（3D形状分类）、N‑body 轨道预测、动作捕捉轨迹预测、GEOM‑Drugs分子构型生成；

**📈 对比分析**

与多种基线（完全等变模型、预定松弛计划、双重约束方法、无约束模型等）比较，实验显示RECM在完全等变任务中MSE最低，在近似等变任务中MSE/准确率均高于或等于最佳对手，且在分子构型生成中显著提升F1分数；

**⚠️ 局限性**

主要局限包括对compact群的依赖、更新函数需要足够表达性且训练稳定性依赖学习率调度、在非compact或复杂约束下理论保证不再成立，且在大规模任务中额外的状态向量会增加计算开销。

---

## 57. Testing Framework Migration with Large Language Models

**arXiv ID:** 2602.02964 | [PDF](https://arxiv.org/pdf/2602.02964v1)

**作者:** Altino Alves `[一作]` (Federal University of Minas Gerais), Andre Hora `[通讯]` (Federal University of Minas Gerais)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了利用大型语言模型（LLM）自动将Python单元测试从unittest迁移到Pytest的可行性与效果。

**💡 创新点**

首次系统评估LLM（GPT‑4o与Claude Sonnet 4）在不同提示策略（Zero‑shot、One‑shot、Chain‑of‑Thought）和温度设置下迁移测试的成功率，并分析迁移风格差异与失败原因。

**🔧 技术方法**

使用了GPT‑4o、Claude Sonnet 4两种LLM；通过手工编写的迁移提示、不同温度；利用Python的unittest与Pytest框架。

**📊 数据集**

构建了TestMigrationsInPy数据集，包含923条真实unittest→Pytest迁移案例；从中挑选40条可执行迁移案例进行实验。

**📈 对比分析**

将LLM生成的迁移代码在项目真实环境中执行并对比覆盖率，实验共480次迁移请求，成功率为48.54%；成功迁移保持覆盖率不变；失败主要涉及fixture、结构、依赖等问题。

**⚠️ 局限性**

受限于迁移上下文缺失、fixture与依赖处理不足、LLM对复杂结构的理解有限，导致约一半迁移失败；提示策略与温度对结果影响不大。

---

## 58. R2-Router: A New Paradigm for LLM Routing with Reasoning

**arXiv ID:** 2602.02823 | [PDF](https://arxiv.org/pdf/2602.02823v1)

**作者:** Jiaqi Xue `[一作]` (University of Central Florida), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24715 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种新的LLM路由方法R2-Router，通过在路由过程中考虑输出长度预算，实现对不同LLM的质量-成本曲线的推断，从而在给定预算下实现更高质量的回答。

**💡 创新点**

创新点在于把输出长度视为可控变量，构造质量-成本曲线而非固定点，并提出R2-Bench数据集记录不同长度下的LLM表现，从而实现“路由即推理”。

**🔧 技术方法**

采用共享编码器+多头MLP进行质量曲线预测，利用长度约束指令与线性插值实现连续预算搜索，并可与现有路由器插件化。

**📊 数据集**

使用R2-Bench（30,968条查询、15个LLM、16个长度预算）以及SPROUT、RouterBench等基准集进行评估。

**📈 对比分析**

与多种基线（MIRT、CARROT、UniRouter等）在AUDC、Peak Quality、QNC等指标上比较，R2-Router在同等或更低成本下获得4-5倍更低成本、显著提升的AUDC与Peak Quality。

**⚠️ 局限性**

局限性包括仍需依赖外部LLM判定器来生成质量标签，对不同LLM或新的配置需要额外收集数据；以及对极短输出的推断可能不稳定。

---

## 59. Enhancing Post-Training Quantization via Future Activation Awareness

**arXiv ID:** 2602.02538 | [PDF](https://arxiv.org/pdf/2602.02538v1)

**作者:** Zheqi Lv `[一作]` (Zhejiang University), Yueting Zhuang `[通讯]` (Tencent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于未来层激活的后训练量化方法FAQ，用以在不进行微调的情况下压缩大语言模型。

**💡 创新点**

创新点在于利用未来层激活预览来决定当前层的量化尺度，并引入窗口预览机制与预搜索配置，显著降低量化偏差和误差累积。

**🔧 技术方法**

技术上实现了权重仅量化的对称量化、未来感知尺度生成、窗口融合、预搜索配置及与AWQ、RTN等基线对比。

**📊 数据集**

实验使用WikiText2、C4、PIQA、ARC、BoolQ、HellaSwag、WinoGrande等数据集，对Qwen3、Qwen2.5、LLaMA3.2、LLaMA2等模型进行评估。

**📈 对比分析**

与RTN和AWQ比较，FAQ在3位量化下可提升多达1.14点的准确率，3bit/4bit设置下均表现出更优的困惑度和准确率，且在不同模型规模与架构上均保持优势。

**⚠️ 局限性**

局限性包括仅针对权重量化、仍需校准数据、预搜索配置依赖经验、未验证在更低比特宽度或训练时量化中的表现。

---

## 60. Membership Inference Attacks from Causal Principles

**arXiv ID:** 2602.02819 | [PDF](https://arxiv.org/pdf/2602.02819v1)

**作者:** Mathieu Even `[一作]` (Inria), Aurélien Bellet `[通讯]` (Inria)

**通讯引用:** 6214 | [OpenAlex ID](https://openalex.org/A5014504793)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了会员推断攻击的评估方法，将其重新表述为因果推断问题，并提出针对多跑、一跑、零跑三种评估场景的无偏估计器。

**💡 创新点**

通过因果框架揭示并消除一跑干扰和零跑分布移位导致的偏差，给出可在不重新训练模型的情况下进行可靠记忆测量的理论与实践方法。

**🔧 技术方法**

采用潜在结果框架、干扰模型、学习理论中的误差稳定性与统一训练稳定性，并使用逆概率加权、G-公式与AIPW等因果推断工具。

**📊 数据集**

在合成线性回归数据、CIFAR-10图像数据以及文本类比数据上进行实验验证。

**📈 对比分析**

与传统未校正的零跑评估相比，实验表明校正后AUC与多跑/一跑结果一致，并满足DP上限，显示方法显著降低了偏差。

**⚠️ 局限性**

仅适用于基于损失的MIA且需满足稳定性与重叠假设，对极端分布移位或更复杂攻击的鲁棒性仍需进一步研究。

---

## 61. Product Interaction: An Algebraic Formalism for Deep Learning Architectures

**arXiv ID:** 2602.02573 | [PDF](https://arxiv.org/pdf/2602.02573v1)

**作者:** Haonan Dong `[一作]` (Tsinghua University), Angelica I. Aviles-Rivero `[通讯]` (YMSC, Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种统一的代数框架——产品交互（Product Interaction），用单一乘法算子在不同代数上构建神经网络层，涵盖CNN、注意力、SSM、Mamba等多种架构，并通过自交互阶数（self‑interaction order）系统地提升模型表达能力。

**💡 创新点**

创新点包括：①将多种主流网络视为不同阶数的产品交互，统一了层的构造；②引入自交互阶数与替换原则，提供对层可解释性的系统工具；③通过对称性原则（群表示）约束结构常数，将对称性与代数运算自然结合，得到卷积、谐波网络、张量场网络、SE(3)注意力等等价实现。

**🔧 技术方法**

主要技术：代数结构与结构常数学习、乘法算子与结构算子组合、非线性激活、对称性（群表示）约束、阶数提升与多级产品交互、替换原则实验。

**📊 数据集**

实验数据集：MNIST、sMNIST、Copy 任务、以及标准视觉与序列任务（如文本、时间序列）用于验证自交互阶数与投影维度对性能的影响。

**📈 对比分析**

评估方法：在 MNIST 上对比无约束、对称性正则化、严格对称性约束三种设置，结果显示约束显著提升精度（0.445→0.943→0.988）；在 sMNIST 与 Copy 任务中，分别展示自交互阶数从 1→2→3、投影维度 R 从 1→2→4 时准确率的显著提升；实验表明自交互阶数提升与对称性约束共同决定模型性能。

**⚠️ 局限性**

局限性：①需要手工设计适当的代数与阶数，缺乏自动化搜索方法；②自交互阶数提升并非总是带来性能提升，需结合具体任务与代数结构；③大规模模型训练与推理效率、可扩展性尚未系统评估。

---

## 62. AutoSizer: Automatic Sizing of Analog and Mixed-Signal Circuits via Large Language Model (LLM) Agents

**arXiv ID:** 2602.02849 | [PDF](https://arxiv.org/pdf/2602.02849v1)

**作者:** Xi Yu `[一作]` (Brookhaven National Laboratory), Yihui Ren `[通讯]` (Brookhaven National Laboratory)

**通讯引用:** 658 | [OpenAlex ID](https://openalex.org/A5013719267)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AutoSizer，一个基于 LLM 的反思式元优化框架，用于自动化模拟与混合信号电路的元件尺寸优化，并发布了开放的 AMS-SizingBench 基准。

**💡 创新点**

创新点在于：① 两层闭环（内部数值优化+外部搜索空间自适应）；② LLM 负责电路理解、搜索空间构造和自我反思；③ 可动态切换多种优化算法；④ 通过自适应搜索空间提升样本效率与成功率；⑤ 公开 24 个 SKY130 电路基准，支持可重复实验。

**🔧 技术方法**

采用的大技术包括：大语言模型（LLM）驱动的电路解析与搜索空间决策、自动化仿真（Ngspice）评估、可配置的优化引擎（LHS、GA、BO、TuRBO 等）以及循环的自我反思机制。

**📊 数据集**

使用 AMS‑SizingBench 数据集，包含 24 个 SKY130 CMOS 的模拟/混合信号电路（放大器、振荡器、参考电压、滤波器等），并划分易/中/难三类。

**📈 对比分析**

与传统 GA/BO/TuRBO 以及 LLM 基线（ADO‑LLM、LEDRO、EE‑Sizer）在 FoM、评估次数、运行时间和成功率上进行对比；AutoSizer 在所有难度级别均实现更高的 FoM、更快收敛、评估次数更少且成功率 100%，尤其在难度较高的电路上表现最为突出。

**⚠️ 局限性**

局限性包括：依赖 LLM 的推理质量和可解释性仍有限；对 SKY130 之外的工艺缺乏直接验证；需要大量仿真调用，计算成本高；框架对极大规模或高度复杂的电路的可扩展性尚待进一步评估。

---

## 63. Experience-Driven Multi-Agent Systems Are Training-free Context-aware Earth Observers

**arXiv ID:** 2602.02559 | [PDF](https://arxiv.org/pdf/2602.02559v1)

**作者:** Pengyu Dai `[一作]` (University of Tokyo), Naoto Yokoya `[通讯]` (University of Tokyo)

**通讯引用:** 14400 | [OpenAlex ID](https://openalex.org/A5034435383)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为GeoEvolver的多代理系统，利用工具交互经验来增强LLM在地球观测任务中的执行可靠性。

**💡 创新点**

创新点在于通过细粒度的子目标拆解、并行探索与自演化的经验记忆库，将执行错误和成功模式转化为可检索的先验，实现在不更新模型参数的情况下实现领域专精。

**🔧 技术方法**

使用检索增强的多代理协同、并行探索、对比式记忆蒸馏、工作记忆/全局记忆两层结构以及LLM的推理与评判等技术。

**📊 数据集**

在ThinkGeo、EarthAgent和GeoPlan‑Bench三大地球观测工具集成基准上进行实验。

**📈 对比分析**

与现有的单代理、MAS、记忆框架和工具使用方法相比，GeoEvolver在EarthAgent上平均提升约12%（最高从25%提升至约77%），在ThinkGeo和GeoPlan‑Bench同样实现显著性能提升，证明了经验驱动的自演化策略的有效性。

**⚠️ 局限性**

局限包括较高的推理延迟与算力消耗、对视觉信息的间接处理导致的感知限制，以及对底层工具链错误的鲁棒性依赖。

---

## 64. Scaled Dot-Product Attention implements projection of inputs onto a common surface

**arXiv ID:** 2602.02521 | [PDF](https://arxiv.org/pdf/2602.02521v1)

**作者:** Terence D Sanger `[一作]` `[通讯]`, Terence D Sanger

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将Scaled Dot‑Product Attention重新表述为投影操作，并在西班牙‑英语翻译任务中验证其可行性

**💡 创新点**

提出投影SDPA，将注意力机制解释为对查询向量投影到由关键向量构成的高维表面，提供了数学等价且可解释的框架

**🔧 技术方法**

采用投影SDPA公式、Transformer结构、softmax、层归一化、Gaussian加权、交叉/自注意力以及多头投影等技术

**📊 数据集**

使用Tatoeba.org语料库的118,000对西班牙‑英语句子，词表15,000，序列长度10的训练集、验证集和测试集

**📈 对比分析**

与标准SDPA在10个epoch的训练下进行对比，投影SDPA在A100 GPU上速度提升约30%（174s→129s），但准确率略低且收敛速度更慢

**⚠️ 局限性**

仅为SDPA的重写，未显著提升计算能力；局限于有限冲击响应、离散时间，且对σ等超参数的选择与优化尚需进一步研究

---

## 65. A Proxy Stakeholder Approach to Requirements Engineering for Inclusive Navigation

**arXiv ID:** 2602.02869 | [PDF](https://arxiv.org/pdf/2602.02869v1)

**作者:** Wei Wang `[一作]` (Monash University), Charmine E. J. Härtel `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过国际调查（80名代理者）和三阶段访谈（15名代理者）提出代理利益相关者（proxy stakeholder）概念，并基于此改进需求工程流程，针对认知障碍人群的导航技术提出可落地的设计建议。

**💡 创新点**

首次系统化将代理利益相关者与传统间接利益相关者区分，构建三阶段参与模型；通过代理视角补足需求信息缺口，强调可定制化、协同使用和例行化的设计准则。

**🔧 技术方法**

采用质性引导的混合方法（问卷+访谈），使用主题分析和社会技术归纳理论（STGT）进行编码；无算法或机器学习模型，聚焦需求分析与人机交互设计。

**📊 数据集**

数据来源为80名代理者的问卷数据（照护者、支持工作者、健康专业人员等）和15名代理者的深度访谈，涵盖不同年龄、角色与经验；未使用公开数据集，而是通过自定义调查与访谈收集原始数据。

**📈 对比分析**

与传统仅采集直接用户需求的RE方法对比，研究发现代理者视角显著补全信息、提升需求完整性；通过成员检查验证结果，说明建议与参与者期望高度一致，虽无定量性能指标，但在可用性与可接受性方面获得正面反馈。

**⚠️ 局限性**

样本规模有限且地域分布偏向英联邦国家；未直接收集认知障碍人群视角，代理者主观判断可能产生偏差；研究为定性探索，缺乏算法验证或大规模实证评估。

---

## 66. Privately Fine-Tuned LLMs Preserve Temporal Dynamics in Tabular Data

**arXiv ID:** 2602.02766 | [PDF](https://arxiv.org/pdf/2602.02766v1)

**作者:** Lucas Rosenblatt `[一作]` (New York University), Natalia Ponomareva `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对包含完整用户时间序列的纵向表格数据，提出了一种以完整用户表为隐私单位的差分隐私合成框架。

**💡 创新点**

创新点在于：①将完整用户表作为隐私单位而非行；②采用自回归序列化+私有细调LLM的方法；③提出基于动态时间规整的表级距离指标TDCR。

**🔧 技术方法**

使用了Gemma 3 LLM（1B/4B）、DP‑SGD + LoRA、动态窗口采样、私有选择等技术。

**📊 数据集**

在MIMIC‑IV生命体征、NYC 311服务请求以及合成HMM数据集上进行评估。

**📈 对比分析**

与传统基于边际的DP合成方法、直接机制、AIM及无私有的Gemini 2.5 FL进行比较；在TDCR、MAUVE、状态转移误差等指标上，所提方法在保留时间一致性和分布覆盖方面显著优于基线。

**⚠️ 局限性**

局限在于：仍需处理大规模多字段异构表时的序列长度限制；在极低隐私预算下性能下降；对非时间依赖的多模态特征支持不足。

---

## 67. The Hypocrisy Gap: Quantifying Divergence Between Internal Belief and Chain-of-Thought Explanation via Sparse Autoencoders

**arXiv ID:** 2602.02496 | [PDF](https://arxiv.org/pdf/2602.02496v1)

**作者:** Shikhar Shiromani `[一作]` (Independent), Sri Pranav Kunda `[通讯]` (Independent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究者提出并实现了“Hypocrisy Gap”，一种衡量大型语言模型内部真相对齐与链式推理生成之间偏差的机制化指标。

**💡 创新点**

通过在稀疏自编码器（SAE）特征空间中学习稀疏线性真相方向，量化模型在压力情境下对齐程度的差距，从而首次在白盒层面检测谬误与合规行为。

**🔧 技术方法**

使用稀疏自编码器（SAE）、L1 正则化逻辑回归、标准化与投影、以及对内部激活的平均化等技术。

**📊 数据集**

Anthropic 的 Sycophancy benchmark（包含问题、正确答案和错误答案三元组）。

**📈 对比分析**

在 Gemma‑2B‑IT、Qwen‑3‑1.7B、Llama‑3.1‑8B‑Instruct 三个开源模型上评估，Hypocrisy Gap 的 AUROC 在 0.55–0.74 之间，显著优于 log‑probability 基线（0.41–0.50）。

**⚠️ 局限性**

局限性包括需访问内部激活和预训练 SAE，依赖特定提示模板与层级，聚合激活可能掩盖细粒度推理阶段，且仅在单一 benchmark 与有限模型上验证，跨语言或其他偏差场景的通用性未知。

---

## 68. Rare Event Early Detection: A Dataset of Sepsis Onset for Critically Ill Trauma Patients

**arXiv ID:** 2602.02930 | [PDF](https://arxiv.org/pdf/2602.02930v1)

**作者:** Yin Jin `[一作]` (University of Washington), Juhua Hu `[通讯]` (University of Washington)

**通讯引用:** 766 | [OpenAlex ID](https://openalex.org/A5029673876)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了一个公开的标准化创伤后败血症（post‑trauma sepsis）数据集，并在此基础上提出了基于夜间生命体征的罕见事件早期检测基准，完成了相关实验验证。

**💡 创新点**

创新点在于首次针对创伤患者制定专属败血症诊断标准、将早期检测框架对齐ICU日常晨会流程并转化为每日罕见事件分类问题，同时提供可复现的数据和代码。

**🔧 技术方法**

采用了深度自编码（Masked Autoencoder）预训练、重构与掩蔽增强、SMOTE、时间扭曲、噪声等数据增强技术，并结合XGBoost、LightGBM、GRU‑TCNN等模型进行实验。

**📊 数据集**

数据来源于MIMIC‑III公开数据库，在此基础上通过ICD‑9 E‑code、血培养、抗生素记录、SOFA评分等筛选得到1,570例创伤ICU患者，并标注了729例潜在感染、535例败血症。

**📈 对比分析**

与传统Sepsis‑3标签（无预训练）相比，使用专属标签+预训练的模型在平衡精度、召回率与F1等指标上提升约20%‑30%，且在罕见事件检测中表现出更好的类别平衡。

**⚠️ 局限性**

主要限制在于数据量相对较小（约8,800个样本）、仅包含MIMIC‑III且未充分利用实验室检验等更丰富特征，且对真实临床环境的可迁移性需要进一步验证。

---

## 69. Learning ORDER-Aware Multimodal Representations for Composite Materials Design

**arXiv ID:** 2602.02513 | [PDF](https://arxiv.org/pdf/2602.02513v1)

**作者:** Xinyao Li `[一作]` (University of Electronic Science and Technology of China), Ivor Tsang `[通讯]` (Astar Center for Frontier Artificial Intelligence Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向复合材料的多模态预训练框架ORDER，通过把属性序数关系纳入对齐过程，实现了图像与表格描述的联合表征学习。

**💡 创新点**

创新点：①引入属性序数对齐的对比学习，保证具有相近属性的样本在潜在空间中靠得更近；②使用偏好引导的多目标优化动态平衡跨模态对齐与序数对齐；③采用LoRA进行参数高效的领域迁移，将大规模CLIP视觉模型微调到复合材料领域。

**🔧 技术方法**

核心技术：跨模态对比损失、属性序数对比损失、偏好引导多目标优化、LoRA低秩适配、Vision Transformer + FT‑Transformer 编码器、Diffusion 生成网络。

**📊 数据集**

数据集：公开的 Nanofiber‑enforced composite 数据集和作者自行构建的 CF‑T700（碳纤维 T700）多模态数据集（共 436 对表格‑图像样本）。

**📈 对比分析**

与 CMCL、MatMCL、单模态基线（XGBoost、ResNet、ViT 等）对比，ORDER 在跨模态检索、属性预测（RMSE 下降 20‑40%）和微结构生成（FID、KID、LPIPS 等指标提升）方面均显著优于现有方法。

**⚠️ 局限性**

局限性：样本规模仍较小（仅数百对），对多目标属性的处理仍有限；生成结果对极端属性可能失真；未提供不确定性量化与检索排名置信度评估。

---

## 70. Aligning Language Model Benchmarks with Pairwise Preferences

**arXiv ID:** 2602.02898 | [PDF](https://arxiv.org/pdf/2602.02898v1)

**作者:** Marco Gutierrez `[一作]` (University of Virginia), Thomas Hartvigsen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 benchmark alignment 概念，并实现 BenchAlign 方法，利用有限模型的 benchmark 性能与模型对之间的偏好顺序，重新加权现有 benchmark 问题，生成能够预测模型偏好排序的新 benchmark。

**💡 创新点**

创新点在于：① 将学习‑to‑rank（pairwise loss）直接应用于 benchmark 问题加权，避免传统的子集筛选或压缩；② 通过仅需少量已评估模型即可学习到能跨模型规模、跨任务、跨偏好方向的权重；③ 展示该方法在未见模型上仍能保持高相关性的能力。

**🔧 技术方法**

使用的一层 RankNet（pairwise learning‑to‑rank）以及多种损失（LambdaRank、NDCG‑Loss 等）对模型的题级得分进行加权；评估指标为 Spearman 相关系数和 pairwise ranking accuracy；实验中还比较了不同学习‑to‑rank 算法的表现。

**📊 数据集**

数据集包括：OpenLLMLeaderboard 的 4576 模型在 21,606 个 benchmark 问题上的题级响应；Helpsteer 与 UltraFeedback 的奖励模型，用于生成目标偏好排序；以及 MMLU、BigBench、MATH 等子任务。

**📈 对比分析**

与 MetaBench、TinyBenchmarks、Random 三种基线进行对比，BenchAlign 在不同模型规模拆分（13B、30B、70B 以上）、不同数据量（模型数、问题数）以及随机模型分组下均取得更高的 Spearman 相关（≈0.7+）和 pairwise accuracy（≈0.85+），尤其在 70B+ 规模模型上保持稳定性能。

**⚠️ 局限性**

限制：需要约 1000 个 7B 以下模型和 5000 条问题的训练样本才能获得最佳效果；当目标偏好与 benchmark 信息相关性低时提升有限；方法依赖现有 benchmark 的多样性，难以处理极端动态或多目标偏好情况。

---

## 71. Efficient Counterfactual Estimation of Conditional Greeks via Malliavin-based Weak Derivatives

**arXiv ID:** 2602.02811 | [PDF](https://arxiv.org/pdf/2602.02811v1)

**作者:** Vikram Krishnamurthy `[一作]` (Cornell University), Luke Snow `[通讯]` (Cornell University)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5049655790)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究了扩散过程条件损失函数的反事实梯度估计，提出了一种两阶段的无核方法来提高效率。

**💡 创新点**

创新点在于利用Malliavin微积分提供了一种精确的Skorohod积分表示，从而恢复经典的Monte Carlo收敛率，并且通过弱导数方法实现了常数方差的梯度估计。

**🔧 技术方法**

使用了Malliavin微积分和Skorohod积分的技术。

**📊 数据集**

未具体提及使用的数据集，但研究背景为金融市场中的扩散过程。

**📈 对比分析**

与传统的分数函数方法相比，提出的方法在稀有事件条件下的方差为O(1)，而分数函数方法的方差为O(T)，显示出显著的性能提升。

**⚠️ 局限性**

限制在于该方法依赖于Malliavin微积分的计算，可能在某些复杂情况下难以实现。

---

## 72. Human-Centric Traffic Signal Control for Equity: A Multi-Agent Action Branching Deep Reinforcement Learning Approach

**arXiv ID:** 2602.02959 | [PDF](https://arxiv.org/pdf/2602.02959v1)

**作者:** Xiaocai Zhang `[一作]` (University of Melbourne), Milad Haghani `[通讯]` (University of Melbourne)

**通讯引用:** 4838 | [OpenAlex ID](https://openalex.org/A5014354122)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了基于多智能体动作分支双深度 Q 网络（MA2B‑DDQN）的交通信号控制框架，专注于多模态公平和人性化优化。

**💡 创新点**

创新点包括：① 将信号控制拆分为本地（每个路口的绿灯比例）与全局（总周期）两级动作分支，显著降低高维离散动作空间的维数；② 设计以“受影响旅客人数”为核心的公平奖励函数，兼顾车辆、公交与行人；③ 在框架中加入 Double DQN 以抑制 Q 值过估，提升学习稳定性。

**🔧 技术方法**

技术手段：深度强化学习（DDQN+动作分支架构），状态由车辆占用、速度、加速度、乘员数以及行人计数与信号状态构成；网络采用共享特征提取层后分支输出多头 Q 值；训练采用经验回放、Adam 优化，使用 VISSIM 仿真模拟三路口的交通网络。

**📊 数据集**

使用的“数据集”是墨尔本 AIMES 测试路段的实际交通流量需求，包含七种不同交通需求场景（低峰、高峰、上下学、饱和、动态变化等）以及相应的车辆与行人生成脚本。

**📈 对比分析**

与九个基准方法对比（固定信号、MADQN、MADDQN、MADDPG、CMRM、MASAC、MAA2C、MA2B‑DQN），MA2B‑DDQN 在所有七个场景中平均降低受影响旅客总数（AID）约 10‑16%，平均车辆延迟（AVDS）与平均行人延迟（APDS）均优于基准，并且标准差最低、训练时间最短，显示出更好的鲁棒性和可扩展性。

**⚠️ 局限性**

局限性：① 对行人优化相对弱，行人延迟指标在部分场景略高；② 在接近饱和流量时整体提升有限；③ 总周期范围与阈值需手工设定；④ 目前仅考虑公平奖励，未加入安全、排放等多目标约束。

---

## 73. STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models

**arXiv ID:** 2602.03022 | [PDF](https://arxiv.org/pdf/2602.03022v1)

**作者:** Jiliang Ni `[一作]` (Alibaba), Conggang Hu `[通讯]` (Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出STAR框架，将大规模语言模型的函数调用能力迁移到超小（≤0.6B）模型，通过联合知识蒸馏与强化学习实现高效、可部署的智能体。

**💡 创新点**

创新点包括（1）Constrained Knowledge Distillation（CKD）：在top‑k前向KL基础上加入尾部L1正则，稳定蒸馏且保持探索性；（2）Similarity‑guided Reinforcement Learning（Sim‑RL）：用连续相似度奖励替代二元奖励，提供更细粒度的学习信号；（3）将两者融入系统化训练课程，获得新SOTA。

**🔧 技术方法**

使用的技术包括：top‑k前向KL蒸馏、尾部正则化、GRPO强化学习、相似度奖励（ROUGE‑L、IOU、词对齐等）、教师校正、对比实验（SFT、ToolRL、LUFFY、GKD等）。

**📊 数据集**

训练数据集：ToolACE、xLAM、xLAM‑irrelevance、Tool‑use‑synthetic；评测基准：BFCLv3 与 ACEBench。

**📈 对比分析**

与多种基线（Base‑model、SFT、SFT‑think、FKL、ToolRL、LUFFY、GKD）对比，0.6B STAR在BFCL整体准确率51.70%、ACEBench 53.00%，均显著高于基线，并在某些指标上超过部分更大模型。

**⚠️ 局限性**

限制：目前仅验证于函数调用任务；相似度奖励仍可进一步改进；未在更广泛任务（如SQL、数学推理）或更大模型上做充分验证。

---

## 74. ROSA-Tuning: Enhancing Long-Context Modeling via Suffix Matching

**arXiv ID:** 2602.02499 | [PDF](https://arxiv.org/pdf/2602.02499v1)

**作者:** Yunao Zheng `[一作]` (Beijing University of Posts and Telecommunications), Wei Chen `[通讯]` (Li Auto Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ROSA‑Tuning，一种在预训练模型中加入CPU侧后缀自动机检索与注入的长上下文增强方法

**💡 创新点**

通过将检索过程与注意力分离，使用低成本后缀自动机在GPU注意力之外实时定位并注入历史信息；同时设计二进制离散化与反事实梯度训练框架

**🔧 技术方法**

后缀自动机（SAM）、二进制符号序列、按路（route）划分的离散化、反事实梯度、CPU‑GPU异步流水线

**📊 数据集**

在Qwen3‑Base‑1.7B上进行训练与评测，使用LongBench、LM‑eval等公开长序列与通用任务数据集

**📈 对比分析**

与全局注意力(Global‑Attn)和窗口注意力(Window‑Attn)对比，ROSA‑Tuning在窗口模型上显著提升长上下文性能，接近甚至匹配全局注意力，同时GPU内存与计算复杂度保持在O(TW)，相当于窗口注意力；在通用任务上几乎无性能损失

**⚠️ 局限性**

仍需大量算力训练；CPU检索在高吞吐量下仍是瓶颈；方法主要验证在单一模型（Qwen3）上，跨模型迁移与大规模部署仍待进一步研究

---

## 75. Language Movement Primitives: Grounding Language Models in Robot Motion

**arXiv ID:** 2602.02839 | [PDF](https://arxiv.org/pdf/2602.02839v1)

**作者:** Yinlong Dai `[一作]` (Virginia Tech), Simon Stepputtis `[通讯]` (Virginia Tech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Language Movement Primitives (LMP) 框架，将大规模视觉语言模型与动态运动原语 (DMP) 结合，实现从自然语言指令到机器人连续运动的零样本执行。

**💡 创新点**

将 VLM 的高层推理直接映射到可解释的 DMP 参数，并通过任务分解与反馈循环，做到无需领域微调即可完成多步桌面操纵任务。

**🔧 技术方法**

使用 Gemini‑Robotics‑ER、LangSAM 进行对象检测与分割，Gemini 作为任务分解器，GPT‑5.2 生成 DMP 参数，DMP 控制器执行轨迹，反馈模块进行迭代修正。

**📊 数据集**

在 20 个真实桌面操纵任务上评估，任务来源于改编自现有 benchmark，采用随机摆放对象与障碍并收集 RGB‑D 图像。

**📈 对比分析**

与 TrajGen（代码生成轨迹）和 π_0.5（Vision‑Language‑Action 细调模型）比较，LMP 在 20 任务上的成功率为 80%，显著高于 30%/31% 的基线，并实现无领域微调的零样本表现。

**⚠️ 局限性**

依赖低层控制器提供可解释参数，难以处理动态环境变化，且反馈仍依赖人工评审，未来需开发自律评估与动态建模方法。

---

## 76. daVinci-Agency: Unlocking Long-Horizon Agency Data-Efficiently

**arXiv ID:** 2602.02619 | [PDF](https://arxiv.org/pdf/2602.02619v1)

**作者:** Mohan Jiang `[一作]` (SII), Pengfei Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用真实软件项目的拉取请求链生成长周期训练数据，构建 agent 的学习轨迹；

**💡 创新点**

首次将 PR 演化过程当作长程代理学习的监督信号，系统捕获任务分解、长期一致性与迭代改进三大关键技能；

**🔧 技术方法**

采用 GitHub API 采集 PR 依赖、LLM 生成查询、GLM‑4.6 进行自回合 rollout 与拒绝采样，最终对大模型进行全参数微调；

**📊 数据集**

基于 9 个高质量 GitHub 仓库共 239 条链式 PR 样本，平均 85k token 与 116 次工具调用；

**📈 对比分析**

与 GLM‑4.6、Kimi‑K2、DeepSeek、Qwen‑3 系列以及 SWE‑Smith、CC‑Bench、CodeAgent 等基线对比，微调后在 Toolathlon 获得 47% 相对提升，SWE‑bench 取得 0.475 的整体平均分，显著优于同类数据集；

**⚠️ 局限性**

目前链长上限为 5 条 PR，成功率受限，需提升链长与覆盖度以进一步挖掘长程代理潜能。

---

## 77. Performance of Small Language Model Pretraining on FABRIC: An Empirical Study

**arXiv ID:** 2602.02632 | [PDF](https://arxiv.org/pdf/2602.02632v1)

**作者:** Praveen Rao `[一作]` (University of Missouri), Praveen Rao `[通讯]` (University of Missouri)

**通讯引用:** 1000 | [OpenAlex ID](https://openalex.org/A5087950601)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在FABRIC全国级学术计算基础设施上，对小型语言模型（GPT‑2 medium/large）预训练的多种并行技术（数据并行、内核并行、管线并行以及Alpa的Shard+Pipelined组合）进行了系统评估与比较。

**💡 创新点**

提出了基于网络延迟与GPU内存的预训练技术选择算法，并证明在跨站高延迟环境中，Alpa的同时优化内核和管线并行可显著提升训练吞吐量；同时，首次将该技术在学术级分布式GPU集群上验证。

**🔧 技术方法**

使用Alpa、Ray、NCCL、CUDA、CuPy、PyTorch等开源工具实现训练，实验涵盖数据并行、Shard并行、Pipelined并行及其组合；并通过NCCL进行多GPU通信。

**📊 数据集**

采用来自HuggingFace的Wikipedia 20231101.ace语料库（约10GB）作为预训练数据。

**📈 对比分析**

通过在20个epoch内测量总训练时长和平均TFLOP/s，对比单站与跨站、不同GPU类型（RTX 6000、T4、A30）的五种配置；结果显示：在跨站网络延迟10–100 ms时，Alpa的Shard+Pipelined方案取得最高吞吐量；在单站且GPU足够时，单机数据并行或Shard并行往往更快；高延迟导致传统多机方案性能急剧下降。

**⚠️ 局限性**

受限于FABRIC GPU数量、异构硬件、内存不足及高网络延迟；实验仅覆盖GPT‑2 medium/large，未验证更大模型或其他数据集；并未深入探讨量化或混合精度等压缩技术对预训练效率的影响。

---

## 78. Reward Shaping for Inference-Time Alignment: A Stackelberg Game Perspective

**arXiv ID:** 2602.02572 | [PDF](https://arxiv.org/pdf/2602.02572v1)

**作者:** Haichuan Wang `[一作]` (Harvard University), Milind Tambe `[通讯]` (Harvard University)

**通讯引用:** 23164 | [OpenAlex ID](https://openalex.org/A5000327528)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对在KL正则化下使用已学习的奖励模型进行LLM对齐的局限性，提出了通过Stackelberg博弈设计奖励模型的框架，并将该奖励塑形方案（SRS）无缝集成到推理时对齐方法（Controlled Decoding、ARGS）中。

**💡 创新点**

创新点包括：①将奖励模型的设计转化为领导者-追随者Stackelberg博弈；②证明最优奖励模型具有阈值结构，并给出解析与软阈值近似；③通过Monte‑Carlo估计与二分搜索快速求解阈值；④在推理时对齐中实现零/极小额外推理开销。

**🔧 技术方法**

使用的技术主要有：Stackelberg博弈理论、KL正则化奖励塑形、Monte‑Carlo估计、二分搜索、Soft‑max/Sigmoid阈值、Controlled Decoding与ARGS推理框架，以及GPT‑4评估。

**📊 数据集**

实验数据集：HH‑RLHF、SHP 对齐基准；模型后端：Qwen3‑8B、Llama3‑8B‑Instruct；奖励模型：Skywork‑Qwen、Skywork‑Llama；在300条随机提示上评估。

**📈 对比分析**

对比方法包括无对齐基线、Minmax、Meanstd以及原始CD/ARGS；评价指标为多样性、连贯性与平均奖励；SRS在所有设置下均获得最高平均奖励，且多样性/连贯性不下降；GPT‑4 评估中平均 win‑tie 率约 66%，明显优于基线。

**⚠️ 局限性**

局限性：依赖于无害且反映人类价值的奖励模型；在存在强基模型偏差时仍可能受限；阈值与奖励上界 B 的选择对性能影响大；在某些推理对齐方法（如CD 在 Eval‑3/4）中未能提升，说明方法对基准的适用性有限。

---

## 79. Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions

**arXiv ID:** 2602.02560 | [PDF](https://arxiv.org/pdf/2602.02560v1)

**作者:** Bartlomiej Sobieski `[一作]` (University of Warsaw), Przemyslaw Biecek `[通讯]` (University of Warsaw)

**通讯引用:** 6059 | [OpenAlex ID](https://openalex.org/A5049061860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

基于生成式干预的可解释框架，对肺癌风险预测模型Sybil进行因果审计

**💡 创新点**

首次将3D扩散桥模型与nodule级的因果干预相结合，揭示Sybil的线性加性结构及其局部偏差

**🔧 技术方法**

使用扩散桥（SDB）、SHAP/二阶Shapley、3D肺结节插入/移除、统计评估（R²、ANOVA、回归）等技术

**📊 数据集**

NLST（训练/基准）、LUNA25（验证/病灶标签）和iLDCT（外部测试）三套肺CT数据集

**📈 对比分析**

与传统可视化、随机噪声插值等基线对比，Sybil在结节贡献估计上R²≈1，检测准确率保持与原模型一致，但发现明显的周边敏感性偏差和伪影依赖

**⚠️ 局限性**

受限于生成式样本的真实性、可能的合成噪声和模型对边缘卷积的固有偏差，导致对极端结节或外部结构的解释不完全可靠

---

## 80. SceneLinker: Compositional 3D Scene Generation via Semantic Scene Graph from RGB Sequences

**arXiv ID:** 2602.02974 | [PDF](https://arxiv.org/pdf/2602.02974v1)

**作者:** Seok-Young Kim `[一作]` (KAIST), Woontack Woo `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用RGB序列通过图结构预测场景图，并基于预测的场景图生成与真实空间一致的3D虚拟场景。

**💡 创新点**

提出跨检查特征注意力（CCFA）实现更鲁棒的图预测，并在Graph‑VAE中引入联合形状与布局（JSL）块，实现布局与形状的协同学习。

**🔧 技术方法**

结合视觉SLAM、点云、定向包围盒、GCN、跨检查注意力、VAE、DeepSDF以及CLIP视觉‑语言模型。

**📊 数据集**

使用3RScan/3DSSG（RGB‑D室内场景）和SG‑FRONT（带场景图标注的3D‑FURNITURE数据集）。

**📈 对比分析**

与SOTA方法（如VGFM、MonoSSG、Graph‑to‑3D、EchoScene、MMGDreamer等）对比，SceneLinker在场景图召回率、形状与布局一致性、硬约束精度上均显著提升（Recall↑≈7%–14%，整体生成速度提升25×）。

**⚠️ 局限性**

受限于定向包围盒的定位误差、仅能生成3D‑FURNITURE类物体、缺乏深度信息导致的配准错误、复杂几何体生成失真以及未在用户体验中验证。

---

## 81. Smell with Genji: Rediscovering Human Perception through an Olfactory Game with AI

**arXiv ID:** 2602.02785 | [PDF](https://arxiv.org/pdf/2602.02785v1)

**作者:** Awu Chen `[一作]` (MIT Media Lab), Hiroshi Ishii `[通讯]` (MIT Media Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文提出了 Smell with Genji，一套基于人工智能的嗅觉游戏系统，将传统日本嗅觉游戏 Genji-kō 与多通道嗅觉传感、Transformer 级别的香味分类模型和 LLM 生成的对话接口结合，为参与者提供一种人与 AI 协作的嗅觉体验与反思方式。

**💡 创新点**

创新点在于将文化化的嗅觉游戏与 AI 感知、对话与可视化交互统一起来，突破了嗅觉难以语言化的“嗅觉-语言鸿沟”，并通过 AI 的感知与对话生成为玩家提供与机器感知的对照与共鸣，从而深化对嗅觉认知的自我反思。

**🔧 技术方法**

采用的技术包括：3 轴金属氧化物传感器（BME680、SGP30、Multichannel Gas Sensor V2）进行实时气体检测；Transformer‑Encoder+MLP 的时序香味分类模型；检索增强生成（RAG）+LLM 的对话层；React + WebSocket 的移动端引导界面；3D 打印的 AI 伴随装置。

**📊 数据集**

使用的数据集为本研究自行采集的 5 种香料样本（共 75 分钟、约 405,000 条传感器记录），并在实验室环境下进行训练与评估；并未使用公开的大型嗅觉数据集。

**📈 对比分析**

模型在受控环境下的香味分类准确率约为 40%，由于香料间 VOC 交叉，准确率受限；系统并未以传统精度评测为目标，而是将 AI 设计为学习伙伴，在游戏中通过对话与视觉化反馈让人机感知对齐与差异成为反思的切入点。

**⚠️ 局限性**

局限性包括：①传感器与模型对微小香味差异的辨识能力有限，导致分类准确率不高；②仅使用 5 种香料，难以泛化到更丰富的嗅觉场景；③缺乏大规模用户研究验证系统的可用性与反思效果；④对话内容受限于静态与动态 RAG 库，若知识库更新不及时可能导致信息偏差。

---

## 82. Mixture of Concept Bottleneck Experts

**arXiv ID:** 2602.02886 | [PDF](https://arxiv.org/pdf/2602.02886v1)

**作者:** Francesco De Santis `[一作]` (Polytechnic of Torino), Danilo Giordano `[通讯]` (Polytechnic of Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于概念瓶颈的混合专家模型框架（MCBM），通过控制专家数量和功能形式实现可解释性与准确性的平衡。

**💡 创新点**

创新点在于将概念瓶颈模型泛化为多专家结构，并引入可自定义符号回归功能，既可多层线性专家，也可符号表达式。

**🔧 技术方法**

使用表达式树、符号回归（PySR）、多专家选择器、概念编码器以及端到端训练。

**📊 数据集**

在四个合成数据集（MNIST‑Arithm、dSprites‑Exp、Pendulum、MAWPS）和五个真实数据集（AWA2、CUB‑200、MAWP、CIFAR‑10）上进行实验。

**📈 对比分析**

与CBM、CEM、DCR、LICEM、CMR、黑盒DNN等基线对比，MCBM 在准确率和解释复杂度上达 Pareto 前沿，尤其在线性/符号专家模型中显著提升 65% 准确率，且对干预和不完整概念具备更强鲁棒性。

**⚠️ 局限性**

局限性包括选择器网络是黑盒，符号回归搜索成本高，且需预先指定专家数量M，超参数调优较为繁琐。

---

## 83. Automatic Design of Optimization Test Problems with Large Language Models

**arXiv ID:** 2602.02724 | [PDF](https://arxiv.org/pdf/2602.02724v1)

**作者:** Wojciech Achtelik `[一作]` (AGH University of Krakow), Jacek Mańdziuk `[通讯]` (Warsaw University of Technology)

**通讯引用:** 2227 | [OpenAlex ID](https://openalex.org/A5073814691)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了基于大语言模型的进化框架EoTF，用以自动生成符合目标ELA特征向量的可解释Python黑盒优化测试函数。

**💡 创新点**

创新点在于：①将LLM作为变异算子融入进化搜索，直接生成符号表达式；②通过ELA距离引导搜索，实现对高维景观属性的精准匹配；③生成的函数既可解释、轻量级，又能在高维下保持性能。

**🔧 技术方法**

技术手段包括：大语言模型（Gemini 2.0/2.5/3.0 Flash）、进化算子（初始化、探索、变异）、ELA特征提取（8维统计量）、采样评估、与NN生成器及Zero‑Shot对照。

**📊 数据集**

实验使用BBOB 24个无噪声函数和由MA‑BBOB混合得到的24个新函数，维度覆盖2D、3D、4D、5D。

**📈 对比分析**

评价方法：对每个函数计算100次采样的ELA距离，中位值作指标；构建胜率矩阵、Critical Difference图、优化器排名。实验表明：在2D时NN生成更精准，但从3D起EoTF胜率>80%，并且生成函数在优化器排名上保持与原基准高度一致。

**⚠️ 局限性**

局限性：NN在低维仍占优势；LLM表达能力受限，可能出现表达式复杂度过高或缺失特定结构；存在潜在数据泄漏风险；目前未涵盖约束、多目标等更复杂景观。

---

## 84. Predicting first-episode homelessness among US Veterans using longitudinal EHR data: time-varying models and social risk factors

**arXiv ID:** 2602.02731 | [PDF](https://arxiv.org/pdf/2602.02731v1)

**作者:** Rohan Pandey `[一作]` (Center for Healthcare Organization and Implementation Research), Jack Tsai `[通讯]` (National Center on Homelessness among Veterans)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在美国退伍军人医疗系统的电子健康记录中构建前瞻性预测模型，预测退伍军人首次无家可归事件。

**💡 创新点**

创新点在于将临床条件持续规则与时间动态表示相结合，并将社会行为因素纳入预测，同时在同一框架下比较机器学习、掩码语言模型与大型语言模型的性能。

**🔧 技术方法**

使用时间动态特征、条件持续规则以及 Elastic Net、随机森林、XGBoost、ModernBERT、BioClinical-ModernBERT、LLaMA‑3.1‑8B、OpenBioLLM‑8B 等模型，并通过 LoRA 微调与自然语言提示实现训练。

**📊 数据集**

利用 2016‑2017 年美国退伍军人管理局（VA）企业数据仓库（CDW）中的 4,276,403 名退伍军人数据。

**📈 对比分析**

与静态特征相比，时间动态特征显著提升 PR‑AUC；最佳模型在 3 个月窗口为 ModernBERT（PR‑AUC 2.39%），12 个月窗口为 XGBoost（PR‑AUC 6.72%），PR‑AUC 比基线提升数倍，ROC‑AUC 变化不大。

**⚠️ 局限性**

主要局限包括数据仅来自 VA 系统、社会风险信息可能被低估、模型在非 VA 环境下的可推广性未知、预测窗口相对短、以及对极少数子组的置信区间过宽。

---

## 85. Co2PO: Coordinated Constrained Policy Optimization for Multi-Agent RL

**arXiv ID:** 2602.02970 | [PDF](https://arxiv.org/pdf/2602.02970v1)

**作者:** Shrenik Patel `[一作]` (Rutgers University), Christine Truong `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

Co2PO 通过预测危险并在必要时在共享黑板上写入紧急信息，实现多智能体安全约束强化学习中的主动协调，以在保持可部署成本合规的同时提升收益。

**💡 创新点**

创新点在于将危险预测与稀疏风险触发的通信相结合，采用黑板共享、主动写入与相似度检索的方式，使得通信仅在即将出现安全风险时激活，从而避免了传统全局惩罚导致的过度保守。

**🔧 技术方法**

技术包括：Lagrangian 约束优化、危险预测器（基于观测的危险概率）、黑板存储（状态摘要、意图向量、让行标记）、top‑k 相似度检索、适应性阈值控制、CTDE 训练框架、Hazard 监督标签与写入惩罚。

**📊 数据集**

在 SafePO 提供的多智能体安全基准（Velocity 与 MultiGoal 任务）上进行评估，使用了 Safety‑Gymnasium 的共享奖励与成本信号。

**📈 对比分析**

与 MAPPO、MAPPO‑Lagrangian、MACPO、HAPPO 等基线相比，Co2PO 在 Velocity 任务上可部署收益提升约 7%–8%，在 MultiGoal 任务上提升约 2.9–3.1，且最终成本均保持在预算范围内，说明性能优于传统方法。

**⚠️ 局限性**

局限性包括：训练期间仍可能出现峰值违规；依赖稠密逐步成本及手工阈值；在多目标任务中仍无法在训练预算内实现成本合规；黑板检索与消息压缩在大规模代理场景下的可扩展性待验证。

---

## 86. QuantLRM: Quantization of Large Reasoning Models via Fine-Tuning Signals

**arXiv ID:** 2602.02581 | [PDF](https://arxiv.org/pdf/2602.02581v1)

**作者:** Nan Zhang `[一作]` (Pennsylvania State University), Rui Zhang `[通讯]` (Pennsylvania State University)

**通讯引用:** 33253 | [OpenAlex ID](https://openalex.org/A5100422177)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种利用微调过程中的权重更新信息来实现大型推理模型低位权重量化的方法。

**💡 创新点**

创新点在于提出“保护两端”映射函数，并通过计数零更新来增强通道重要性评分。

**🔧 技术方法**

采用二次约束映射函数、通道重要性加权、与AWQ相结合的缩放因子搜索等技术实现3位量化。

**📊 数据集**

在四个推理基准（AIME‑120、FOLIO、BIG‑Bench Temporal、GPQA‑Diamond）及其对应校准集上进行评估。

**📈 对比分析**

与GPTQ、GPTAQ、AWQ、ANY3等PTQ基线对比，3位量化平均提升约6.55%，在RL和DPO模型上提升更为显著。

**⚠️ 局限性**

需要预先获得微调过程或进行伪微调；在4位量化上的提升有限。

---

## 87. LmPT: Conditional Point Transformer for Anatomical Landmark Detection on 3D Point Clouds

**arXiv ID:** 2602.02808 | [PDF](https://arxiv.org/pdf/2602.02808v1)

**作者:** Matteo Bastico `[一作]` (Mines Paris - PSL University), Etienne Decencière `[通讯]` (Mines Paris - PSL University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于点云Transformer的条件模型LmPT，用于自动检测人类和犬类股骨的解剖标志点

**💡 创新点**

引入FiLM调制机制实现跨物种的可适应性，并提供新标注犬股骨数据集以促进跨物种学习

**🔧 技术方法**

使用Point Transformer（PTv2/3）编码器-解码器结构，配合Feature-wise Linear Modulation（FiLM）与关键点预测头

**📊 数据集**

人类股骨公开数据（20个模型、22个标志点）和自建犬股骨数据（14个模型、11个标志点）

**📈 对比分析**

与传统Atlas‑&‑A priori方法、DGCNN等基线对比，LmPT‑v2在MAE上均优于专家标注，PCK曲线在多阈值下领先，跨物种训练进一步提升人类标志点精度

**⚠️ 局限性**

跨物种训练对犬类标志点略有负面影响，原因是犬数据标志点数量不足且包含人类特有标志点导致特征混淆

---

## 88. SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation

**arXiv ID:** 2602.02989 | [PDF](https://arxiv.org/pdf/2602.02989v1)

**作者:** Zhanfeng Liao `[一作]` (Tsinghua University), Yebin Liu `[通讯]` (Tsinghua University)

**通讯引用:** 10427 | [OpenAlex ID](https://openalex.org/A5032875389)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于可学习寿命的 4D 高斯框架 SharpTimeGS，用于动态场景的新视角合成。

**💡 创新点**

创新点在于将寿命参数融入时间可见性与运动建模，形成平滑-平顶可见性、寿命调制运动和基于寿命‑速度的自适应稠密化。

**🔧 技术方法**

采用 4D 高斯光栅化、可学习寿命、速度感知初始化、寿命‑速度稠密化以及多阶段训练等技术。

**📊 数据集**

在 Neural3DV、ENeRF‑Outdoor 与 SelfCap 三大动态场景数据集上进行评估。

**📈 对比分析**

与 Deformable‑3DGS、4DGS、STGS、FreeTimeGS 等基线相比，SharpTimeGS 在 PSNR/SSIM/LPIPS 指标上均取得最优表现，并实现 4K@100FPS 的实时渲染。

**⚠️ 局限性**

局限在于训练耗时数小时且不支持实时训练，且目前仅支持新视角合成，未实现重光照。

---

## 89. Training Data Governance for Brain Foundation Models

**arXiv ID:** 2602.02511 | [PDF](https://arxiv.org/pdf/2602.02511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 90. Causal Flow Q-Learning for Robust Offline Reinforcement Learning

**arXiv ID:** 2602.02847 | [PDF](https://arxiv.org/pdf/2602.02847v1)

**作者:** Mingxuan Li `[一作]` (Columbia University), Elias Bareinboim `[通讯]` (Columbia University)

**通讯引用:** 3241 | [OpenAlex ID](https://openalex.org/A5039620960)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新的Causal Flow Q‑Learning（CFQL）算法，用于在存在隐含混杂偏差的像素级离线强化学习环境中学习鲁棒策略。

**💡 创新点**

创新点包括：①基于因果推断的目标函数，优化在最坏情况环境下的策略表现；②将流匹配（flow‑matching）方法与深度判别器相结合，能够在表达式策略类中评估并纠正混杂偏差；③使用深度Q网络集合和判别器集合来近似最坏情况价值函数，降低过度悲观的风险。

**🔧 技术方法**

使用的技术主要有：流匹配生成策略、深度判别器（区分行为克隆流与目标策略的动作）、深度Q网络集合（Deep Q Ensemble）以及基于自回归的目标策略训练。

**📊 数据集**

实验数据集采用OGBench视觉任务中的25个像素级控制任务（visual‑cube、visual‑scene、visual‑puzzle等），每个任务使用64×64 RGB图像作为观测。

**📈 对比分析**

与基准算法（IQL、ReBRAC、FQL、IFQL、FBRAC等）进行比较，CFQL在25个任务中取得19个最佳或近最佳结果，平均成功率提升约120%，在某些任务上几乎翻倍，并在离线到在线微调阶段表现出更高的样本效率。

**⚠️ 局限性**

局限性包括：需要精细调节判别器系数和Q网络集合大小；在覆盖度不足或表示学习不佳的任务中仍易受限；目前仅在OGBench视觉基准上验证，缺乏在更广泛环境或连续动作空间中的评估。

---

## 91. Real-World Applications of AI in LTE and 5G-NR Network Infrastructure

**arXiv ID:** 2602.02787 | [PDF](https://arxiv.org/pdf/2602.02787v1)

**作者:** Simran Saxena `[一作]` (Beamlink), Arpad Kovesdy `[通讯]` (Beamlink)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种统一架构，将 AI 驱动的 RAN 自适应优化与基站边缘 AI 服务（如医疗诊断、教育内容缓存、LLM 推理）结合，实现网络性能提升和数字服务可达性增强。

**💡 创新点**

①将 Graph Neural Network 与强化学习融合，实现基于实时遥测的闭环自优化；②引入数字孪生进行参数调整安全验证；③在 4G/5G 基站内集成 Docker 容器化本地化 AI 服务，实现医疗、教育和 LLM 等应用的本地托管。

**🔧 技术方法**

GNN、Q‑learning/Actor‑Critic 强化学习、无监督异常检测、数字孪生仿真、边缘计算容器化、Bentocell 基站软硬件集成、3GPP 标准遥测与遥控接口。

**📊 数据集**

实际 LTE/5G 基站遥测日志（RSRP、SINR、HARQ 等）、Beamlink 现场部署数据、公开无线网络仿真数据，以及公开移动通信基线实验数据。

**📈 对比分析**

与传统静态规划和人工调参进行对比，使用峰值吞吐、时延、能耗、覆盖率等指标评估；实验表明 RL‑驱动优化能降低 15–25% 能耗、提升 10–20% 吞吐、时延下降 30%；边缘托管将上行带宽需求降低 90%，灾难恢复场景下服务可用性提升 40%。

**⚠️ 局限性**

需要大量高质量遥测，RL 训练样本量大且收敛慢；数字孪生需实时同步，易出现误差；基站硬件升级成本高，容器化对低功耗设备有局限；存在隐私与安全风险，跨厂商互操作性不足。

---

## 92. Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Attacks

**arXiv ID:** 2602.02763 | [PDF](https://arxiv.org/pdf/2602.02763v1)

**作者:** Bohan Wang `[一作]` (Emory University), Wei Jin `[通讯]` (Emory University)

**通讯引用:** 125941 | [OpenAlex ID](https://openalex.org/A5100364769)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究时间序列解释器在对抗攻击下的鲁棒性，提出双目标攻击框架TSEF，能同时诱导分类器输出目标标签并使解释器产生指定解释。

**💡 创新点**

创新点在于引入时序漏洞掩码与频谱扰动分离的结构化攻击，克服高维稠密攻击导致解释扩散的问题，实现解释保持的同时完成目标预测。

**🔧 技术方法**

技术包括对抗样本生成、时间-频率域掩码学习、可微分的可视化解释器（TimeX、TimeX++、Integrated Gradients）、互信息稀疏约束与连接正则。

**📊 数据集**

使用六个基准数据集：Synthetic（LowVar、SeqComb-UV、SeqComb-MV）、ECG、PAM、Epilepsy。

**📈 对比分析**

与基线攻击（PGD、BlackTreeS、SFAttack、ADV^2）及噪声扰动比较，TSEF在保持高攻击成功率（≈0.8‑0.95）同时显著提升解释对齐指标（AUPRC/AUP/AUR）超过对手。

**⚠️ 局限性**

局限性包括白盒假设、仅评估离线攻击且解释器类型有限，缺乏对抗鲁棒防御与实际部署环境下的评估。

---

## 93. Ethical Asymmetry in Human-Robot Interaction - An Empirical Test of Sparrow's Hypothesis

**arXiv ID:** 2602.02745 | [PDF](https://arxiv.org/pdf/2602.02745v1)

**作者:** Minyi Wang `[一作]` (University of Canterbury), David Kaber `[通讯]` (Oregon State University)

**通讯引用:** 8448 | [OpenAlex ID](https://openalex.org/A5075585171)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对人机交互中的伦理不对称假设进行实证检验，使用混合实验设计测量道德可容忍度与感知美德。

**💡 创新点**

首次在HRI领域系统检验Sparrow不对称假设，并改编QCV以评估四大美德。

**🔧 技术方法**

采用问卷调查、混合设计、线性回归、MANOVA、曲线拟合与因子分析等统计技术。

**📊 数据集**

使用Prolific在线受试者（146人）生成的40条情景文本及改编的QCV与Malle等量表。

**📈 对比分析**

通过立方曲线拟合比较各美德与可容忍度关系，最高R²为0.545，显示关系对称而非不对称。

**⚠️ 局限性**

受试者在在线情景下缺乏现实感知、文化与美德维度的细微差异，导致结果可能偏向“一刀切”良恶评价。

---

## 94. PokeNet: Learning Kinematic Models of Articulated Objects from Human Observations

**arXiv ID:** 2602.02741 | [PDF](https://arxiv.org/pdf/2602.02741v1)

**作者:** Anmol Gupta `[一作]` (Arizona State University), Nakul Gopalan `[通讯]` (Arizona State University)

**通讯引用:** 721 | [OpenAlex ID](https://openalex.org/A5089421543)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 PokeNet 框架，利用单一人类演示的点云序列端到端估计多自由度机械臂物体的关节参数、关节状态以及操作顺序。

**💡 创新点**

创新点在于：①不需要预先知道物体类别或关节数，采用集合预测 + DETR 查询；②利用人类演示能揭示被遮挡或关闭的关节；③一次性预测关节状态和操作顺序；④可直接应用于多自由度对象。

**🔧 技术方法**

技术包括：PointNet++ + 轻量 Transformer 对空间编码；多层 Transformer 对时间编码；DETR 风格查询 + Hungarian 匹配；多任务损失（置信度、关节类型、轴向、锚点、顺序、状态）以及角度、位置和状态误差监督。

**📊 数据集**

使用的数据集：模拟数据（Partnet‑Mobility，110k 序列，11 类）与真实世界数据（5500 点云序列，4 类家电，3900 训练/1600 测试），并在两类未见对象（刀具、订书机）上做额外验证。

**📈 对比分析**

与 ScrewNet、GAPartNet 进行比较；在模拟数据上轴向误差下降约 25%，在真实世界上提升约 30%；在未见类别、尺度变化及遮挡/关闭关节场景下均表现优于基线。

**⚠️ 局限性**

局限性：未估计接触点；不考虑障碍物，易导致碰撞；未恢复完整几何体；未来需加入碰撞感知和完整数字孪生建模。

---

## 95. Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning

**arXiv ID:** 2602.02951 | [PDF](https://arxiv.org/pdf/2602.02951v1)

**作者:** Yihong Huang `[一作]` (Xidian University), Qi Tian `[通讯]` (Huawei)

**通讯引用:** 41513 | [OpenAlex ID](https://openalex.org/A5100393506)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段视觉token裁剪框架Nüwa，以保持全局空间框架并加速VLM推理

**💡 创新点**

在视觉编码器采用基于Swarm智能的分离-对齐-聚合方法保留空间锚点，并在LLM中进行文本引导裁剪

**🔧 技术方法**

Swarm算法(Boids-inspired)、位置嵌入重构(RPME)、语义相似性与空间邻域加权聚合、文本查询向量裁剪

**📊 数据集**

10个VQA基准（如GQA、TextVQA、MMMU等）和3个视觉定位基准（RefCOCO系列）

**📈 对比分析**

相较于FastV、SparseVLM、VisionZip等主流裁剪方法，Nüwa在VQA上保持95%性能，在VG上提升至47.2%，同时实现约89%TFLOPs和88.9%token压缩

**⚠️ 局限性**

对不同规模模型的推广性尚需进一步验证，且在极低token预算下仍可能出现精度衰减

---

## 96. From Sparse Decisions to Dense Reasoning: A Multi-attribute Trajectory Paradigm for Multimodal Moderation

**arXiv ID:** 2602.02536 | [PDF](https://arxiv.org/pdf/2602.02536v1)

**作者:** Tianle Gu `[一作]` (Tsinghua University), Yingchun Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 11731 | [OpenAlex ID](https://openalex.org/A5100613144)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniMod 框架，将多模态审核任务拆分为证据、模态、风险、策略和答案等五步推理轨迹，并通过稠密监督实现安全决策的可解释化；

**💡 创新点**

将稀疏的二元决策转化为多属性稠密推理轨迹；设计共识教师的 UniTrace 数据集；构建多头奖励模型 UniRM，采用 Head‑wise Weight Subspace Decoupling 与 Stochastic Head Scheduling 解决多任务干扰；

**🔧 技术方法**

结构化推理轨迹、稠密奖励（GRPO）、多头 Scalar Reward（UniRM）、共识教师学习、Head‑wise Decoupling 与随机排程、VLM+LLM 融合、vLLM 推理加速；

**📊 数据集**

UniTrace（约 18K 结构化推理样本）和 UniReward（约 16.8K 单标注多属性奖励样本），并在公开基准（UniTrace、文本/视觉安全评测）上进行评估；

**📈 对比分析**

与多种 LLM/VLM 审核基线（LlamaGuard、WildGuard、ProGuard 等）对比，UniMod‑3B 在多模态审核上以 91.26% 的整体分数领先，同时仅使用 40% 训练数据；UniRM 在多维奖励上取得 88.68% 的平均分，并在单传递中完成全部维度评分；

**⚠️ 局限性**

受限于共识教师生成的标注噪声、模型容量与数据规模的耦合关系、对极端高风险属性的细粒度判定仍不够精细，以及训练仍需大量算力和对冷启动的鲁棒性有待提升。

---

## 97. Learning Better Certified Models from Empirically-Robust Teachers

**arXiv ID:** 2602.02626 | [PDF](https://arxiv.org/pdf/2602.02626v1)

**作者:** Alessandro De Palma `[一作]` `[通讯]` (London School of Economics and Political Science), Alessandro De Palma (London School of Economics and Political Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于知识蒸馏的训练方法CC‑Dist，将经验鲁棒教师的表示迁移到可被验证的学生网络，以提升确定性可验证鲁棒性。

**💡 创新点**

创新在于设计了一种可调的特征空间蒸馏损失，可在对抗输出与IBP下界之间做连续插值，并与CC‑IBP显式损失耦合，提升标准与可验证精度。

**🔧 技术方法**

使用了对抗训练、IBP下界、可表达式损失（CC‑IBP）、特征空间蒸馏、Branch‑and‑Bound完整验证（OVAL）及CROWN/IBP不完整验证等技术。

**📊 数据集**

在CIFAR‑10、TinyImageNet以及下采样ImageNet64等标准图像分类基准上进行实验。

**📈 对比分析**

与纯CC‑IBP以及文献中最佳可表达式训练方法（MTL‑IBP、SABR等）对比，CC‑Dist在所有数据集上同时提升标准准确率和认证准确率，创下ReLU架构新的最优记录。

**⚠️ 局限性**

局限在于教师与学生使用相同网络架构、仅评估ℓ∞扰动、对大规模网络和1‑Lipschitz网络的推广尚未验证。

---

## 98. Late-Stage Generalization Collapse in Grokking: Detecting anti-grokking with Weightwatcher

**arXiv ID:** 2602.02859 | [PDF](https://arxiv.org/pdf/2602.02859v1)

**作者:** Hari K Prakash `[一作]`, Charles H Martin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过长期训练 MLP 与 Transformer，发现了“反 grokking”阶段，即训练准确率完好但测试准确率崩塌的现象。

**💡 创新点**

创新点在于首次系统识别并描述反 grokking，证明其可由权重矩阵中的“Correlation Traps”与 HTSR 指标 α<2 触发，并利用开源 WeightWatcher 工具进行诊断。

**🔧 技术方法**

主要技术包括随机矩阵理论（MP 分布）、HTSR（α 指标）、SETOL（Correlation Traps 检测）、以及传统的 ℓ₂ 归一化、稀疏度、权重熵与电路复杂度等进度指标。

**📊 数据集**

实验数据集包括 MNIST 子集（用于 3 层 MLP）和模数加法任务（用于小型 Transformer）。

**📈 对比分析**

与现有的 grokking 指标相比，α 与 Correlation Traps 能在训练后期准确预警反 grokking，其他指标在该阶段表现平稳且无法捕捉崩塌；实验显示训练步数延长至 10⁷ 后，测试准确率从峰值急剧下降，且上述指标提前检测。

**⚠️ 局限性**

局限性在于 HTSR α 值与实际泛化性能的对应关系仍不确定，α 在某些模型或任务中可能低于 2 但性能不差，且对大型 LLM 的适用性需进一步验证。

---

## 99. VoroUDF: Meshing Unsigned Distance Fields with Voronoi Optimization

**arXiv ID:** 2602.02907 | [PDF](https://arxiv.org/pdf/2602.02907v1)

**作者:** Ningna Wang `[一作]` (Columbia University), Silvia Sellán `[通讯]` (Columbia University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于Voronoi划分的算法VoroUDF，能够从无符号距离场（UDF）中直接生成高质量三角网格，支持非流形结构、尖锐特征以及开放边界，而不依赖于内部/外部估计或多层网格。

**💡 创新点**

创新点在于引入L1切线能量与特征感知排斥能量的Voronoi优化框架，并采用几何Voronoi图的对偶来构造网格，同时通过稀疏采样与网格消融实现轻量化且拓扑一致的输出，首次实现无符号场中非流形结构的准确重建。

**🔧 技术方法**

核心技术包括Voronoi分割、L1切线能量最小化、基于高斯核的排斥能量、几何Voronoi图与其对偶（Delaunay）连接、梯度投影、L-BFGS优化以及基于四面体的网格消融。

**📊 数据集**

实验数据集包含10个合成非流形模型、100个ABC CAD模型和100个DeepFashion3D服装模型，并在这些数据上与多种基线方法进行比较。

**📈 对比分析**

与MeshUDF、CapUDF、GeoUDF、NSDUDF、DCUDF、DCUDF2以及DualMeshUDF等七种基线相比，VoroUDF在多项指标（Chamfer误差、Hausdorff误差、边Chamfer误差、拓扑错误、非流形Chamfer误差）上均取得最优或第二优成绩，且在保持相近顶点数的前提下显著提升了几何与拓扑质量。

**⚠️ 局限性**

主要局限包括：对种子点初始化的敏感性，可能遗漏细小尺度结构；缺乏严格的拓扑正确性理论保证，偶尔会产生极少量非流形边；过度消融可能导致开放边界或细部被误删；对UDF梯度噪声的鲁棒性尚未系统评估。

---

## 100. Benchmarking Large Language Models for Zero-shot and Few-shot Phishing URL Detection

**arXiv ID:** 2602.02641 | [PDF](https://arxiv.org/pdf/2602.02641v1)

**作者:** Najmul Hasan `[一作]` (University of North Carolina), Prashanth BusiReddyGari `[通讯]` (University of North Carolina)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三大商业LLM（GPT‑4o、Claude‑3.7‑Sonnet、Grok‑3‑Beta）在零样本与少样本提示下进行钓鱼URL检测基准实验。

**💡 创新点**

系统化统一的零/少样本提示框架和宏观评估指标，揭示提示样本数对性能的显著影响。

**🔧 技术方法**

基于Prompt‑Engineering的指令调优大语言模型（LLM）与标准化评估指标（Accuracy、Precision、Recall、F1、AUROC、AUPRC）。

**📊 数据集**

使用PhiUSIIL公开钓鱼URL数据集，构造10,000条平衡样本以及1,000条不平衡样本（1%/10%钓鱼率）。

**📈 对比分析**

比较方法：在相同提示、相同指标下对三模型进行零样本和少样本（6例或1/3/9例）实验。结果显示：少样本提升显著，Grok‑3‑Beta在少样本模式下准确率达94.05%、F1得分0.9399；Claude‑3.7‑Sonnet召回率最高；在不平衡情境下三模型保持鲁棒，Grok‑3‑Beta零样本表现最佳。

**⚠️ 局限性**

局限：仅评估三种商业LLM，未涵盖开源模型；提示设计固定，未探索多种提示策略；仅使用单一URL数据集，缺乏跨域验证；未评估实时推理成本与部署复杂度。

---

## 101. AROLA: A Modular Layered Architecture for Scaled Autonomous Racing

**arXiv ID:** 2602.02730 | [PDF](https://arxiv.org/pdf/2602.02730v1)

**作者:** Fam Shihata `[一作]` (German International University in Berlin), Ahmed Hussein `[通讯]` (IAV GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了一套可模块化的自主赛车软件架构AROLA及其实时评估工具Race Monitor。

**💡 创新点**

通过将驾驶栈拆解为八层、统一ROS 2接口并引入Race Monitor实现可互换组件与标准化评估。

**🔧 技术方法**

基于ROS 2、ROS 2 tf、Python/C++、标准ROS消息、Evo、RViz等技术。

**📊 数据集**

使用RoboRacer平台、RoboRacer模拟器、Forza模拟器以及官方的Berlin Map。

**📈 对比分析**

通过对Gap Follower、Pure Pursuit、MPC和LQR控制器的跑道时间、误差、CPU占用等指标比较，Pure Pursuit在赛道上实现10.1 s的跑时，MPC精度最高但负载最高。

**⚠️ 局限性**

验证范围有限，仅测试了少数控制器，缺乏对感知、定位、规划模块的全面评估，并可能出现误差传播问题。

---

## 102. Exploring Collaborative Immersive Visualization & Analytics for High-Dimensional Scientific Data through Domain Expert Perspectives

**arXiv ID:** 2602.02743 | [PDF](https://arxiv.org/pdf/2602.02743v1)

**作者:** Fahim Arsad Nafis `[一作]` (George Mason University), Bo Han `[通讯]` (George Mason University)

**通讯引用:** 3178 | [OpenAlex ID](https://openalex.org/A5014697827)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对20位跨学科领域专家进行半结构化访谈，系统分析高维科学数据协同可视化与分析（CIVA）的工作流程、挑战、需求与风险，并据此提出设计原则和建议。

**💡 创新点**

首次从实践者视角量化CIVA协同需求与障碍，并提出基于合作认知、跨设备同步、可追溯性与AI协助的五大设计原则，填补了先前仅关注单人或原型的研究空白。

**🔧 技术方法**

采用定性研究方法（访谈与混合式主题分析），并通过视频演示作为设计探针；未实现具体系统原型，而是构建了概念模型和设计方案。

**📊 数据集**

访谈中涉及的高维数据类型包括气候模拟、单细胞基因组、LiDAR、卫星影像等多学科数据，但未使用公开数据集进行可视化实验或性能评测。

**📈 对比分析**

论文未进行系统实现与性能对比，仅通过专家访谈获取定性洞察，因此不存在量化性能评估或方法比较；提出的设计原则需在后续实现后进行实验验证。

**⚠️ 局限性**

局限性包括样本量有限（20名远程访谈者）、缺乏系统原型与实验验证、受访者对XR技术熟悉度不高、研究仅基于访谈而非实际操作，导致缺乏客观性能和用户体验数据。

---

## 103. hSNMF: Hybrid Spatially Regularized NMF for Image-Derived Spatial Transcriptomics

**arXiv ID:** 2602.02638 | [PDF](https://arxiv.org/pdf/2602.02638v1)

**作者:** Md Ishtyaq Mahmud `[一作]`, Tania Banerjee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

基于Xenium高分辨率空间转录组数据，提出并评估了两种基于非负矩阵分解的空间正则化方法：Spatial NMF (SNMF) 与 Hybrid Spatial NMF (hSNMF)，通过图卷积式空间平滑提升细胞嵌入的空间连贯性并利用Leiden聚类获得空间连贯且转录相似的细胞群。

**💡 创新点**

创新点在于：①将空间平滑直接嵌入NMF的两阶段流程；②构造结合接触半径与更大范围半径的混合邻接图，实现空间与转录相似度的双重约束；③使用行归一化的混合邻接矩阵进行Leiden聚类，既保证空间连续性，又提高基因标记的生物学一致性。

**🔧 技术方法**

使用的核心技术包括：非负矩阵分解 (NMF)、基于半径的空间邻接图构建、双向图扩散 (Diffusion) 的空间平滑、混合邻接矩阵 (α 介于 0 与 1 的线性组合)、Leiden 聚类、以及一系列空间与几何指标（CHAOS、Moran's I、Silhouette、DBI）和生物学指标（CMC、MER、Enrichment）进行评估。

**📊 数据集**

实验使用来自MD Anderson Cancer Center的25例胆管癌患者共40块TMA核心的Xenium空间转录组数据，480基因面板，初始约212k细胞，质量控制后剩余约191k细胞。

**📈 对比分析**

与RASP、NSF、传统NMF基线相比，hSNMF在所有主要指标上表现最佳：CHAOS <0.004、Moran's I >0.96、Silhouette 0.27、DBI 1.4 以内，CMC 与 Enrichment 也显著提升，说明其空间紧凑度和生物学一致性均优于现有方法；SNMF虽比基线好，但逊于hSNMF。

**⚠️ 局限性**

局限性包括：仅在单一胆管癌数据集验证，缺乏跨组织或跨疾病的泛化评估；图构建依赖固定半径参数，可能无法捕获多尺度空间结构；以及方法仍停留在经典图卷积与聚类层面，未尝试更深层次的生成模型或多模态融合。

---

## 104. Notes on the Reward Representation of Posterior Updates

**arXiv ID:** 2602.02912 | [PDF](https://arxiv.org/pdf/2602.02912v1)

**作者:** Pedro A. Ortega `[一作]` `[通讯]` (Daios Technologies), Pedro A. Ortega (Daios Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

探讨在KL正则化优化中，若最优解恰好是同一联合分布的后验条件，则该更新必须呈现条件互信息（PMI）的点值结构，并阐明奖励与价值的基准不确定性及不同更新方向的一致性约束。

**💡 创新点**

创新点在于：①提出后验识别假设并证明其迫使奖励-价值交互点形式化为PMI；②揭示奖励的基准自由度（Gauge）和更新方向一致性导致的可积性（交换性）约束；③把控制、逆向强化学习、贝叶斯推理三种视角统一到KL正则化的“后验匹配”框架。

**🔧 技术方法**

采用KL正则化（信息理论自由能、相对熵控制）以及互信息的点值表示，利用指数平滑与对数正则化的代数性质，推导出后验识别下的点比率、PMI形状、Gauge不变性和交换性约束。

**📊 数据集**

无实验数据集；论文完全是理论推导与形式化分析。

**📈 对比分析**

无实验比较；讨论基于理论的约束与现有“控制即推断”与“逆RL”框架的差异，说明在后验完全匹配的边界条件下模型的限制与适用范围。

**⚠️ 局限性**

限制在于：仅适用于后验正则化完全匹配单一联合分布的极端情形；若放宽后验匹配或允许更一般的价值函数，所得的PMI约束与Gauge不再成立，且奖励可辨识性仍存缺失。

---

## 105. BiTimeCrossNet: Time-Aware Self-Supervised Learning for Pediatric Sleep

**arXiv ID:** 2602.02769 | [PDF](https://arxiv.org/pdf/2602.02769v1)

**作者:** Saurav Raj Pandey `[一作]` (University of North Carolina), Harlin Lee `[通讯]` (University of North Carolina)

**通讯引用:** 227 | [OpenAlex ID](https://openalex.org/A5055567952)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了BiTimeCrossNet（BTCNet），一种用于儿童多模态睡眠记录的自监督学习框架，能捕捉跨模态交互并利用夜间时间上下文进行表示学习。

**💡 创新点**

创新点在于：①随机抽取两模态对并使用跨注意力学习其交互；②引入全局时间感知的条件编码，将整个睡眠周期的时序信息注入模型；③将掩码重建与对比学习融合成混合目标，提升表征质量。

**🔧 技术方法**

使用的技术包括：ViT编码器、掩码自编码器（MAE）+ NT‑Xent对比损失、跨模态双向注意力、时序条件化（FiLM‑style）、LoRA微调、随机模态对采样。

**📊 数据集**

训练数据为Nationwide Children’s Hospital（NCH）睡眠数据库（2379份 PSG），外部验证使用独立的CHAT（422份）数据集。

**📈 对比分析**

与不具时间感知的BCNet、SleepFM、PedSleepMAE等基线进行对比；在六个下游任务（睡眠分期、觉醒、呼吸事件等）上，BTCNet在线性探针下在NCH和CHAT均显著提升AUROC、F1，尤其在呼吸相关任务上提升幅度最大。

**⚠️ 局限性**

局限性包括：当前仅使用两模态输入，可能不充分利用全部可用信号；对极端缺失/噪声模态的鲁棒性仍待进一步验证；模型对完全不同年龄/疾病分布的迁移性尚未系统评估。

---

## 106. Agent Alpha: Tree Search Unifying Generation, Exploration and Evaluation for Computer-Use Agents

**arXiv ID:** 2602.02995 | [PDF](https://arxiv.org/pdf/2602.02995v1)

**作者:** Sizhe Tang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6378 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Agent Alpha，一个利用步骤级蒙特卡洛树搜索（MCTS）实现的统一框架，用于生成、探索和评估计算机使用代理的行为；

**💡 创新点**

创新点包括引入 Alpha‑UCT 结合最大值增强探索、树感知的行动生成与多样性约束、以及相对比较驱动的评估机制，显著提升了回溯修正和前缀重用能力；

**🔧 技术方法**

技术核心包括基于视觉语言模型的决策生成、Alpha‑UCT 置信界、树级反射与信息聚合、动作分块、以及并行化搜索与评估；

**📊 数据集**

数据集为跨应用的 OSWorld 基准，涵盖操作系统、办公软件、专业工具等十类任务；

**📈 对比分析**

与七个最先进基线（包括 Agent S3、UiPath、Claude 等）对比，Agent Alpha 在 OSWorld 上平均成功率提升至 77.3%，在 7/10 领域实现最高或第二高成功率，且对失败任务的恢复率约 34%；

**⚠️ 局限性**

局限性主要在计算开销高（平均推理时间约 1116 秒）以及对长序列任务的记忆碎片化、环境重建误差和领域知识缺失的敏感性。

---

## 107. "I May Not Have Articulated Myself Clearly": Diagnosing Dynamic Instability in LLM Reasoning at Inference Time

**arXiv ID:** 2602.02863 | [PDF](https://arxiv.org/pdf/2602.02863v1)

**作者:** Jinkun Chen `[一作]` (Dalhousie University), Vlado Keselj `[通讯]` (Dalhousie University)

**通讯引用:** 1765 | [OpenAlex ID](https://openalex.org/A5035485219)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在推理时检测LLM推理过程中的动态不稳定性并预测错误。

**💡 创新点**

提出仅基于token概率的实时不稳定性信号，结合JSD与熵，并区分早期修正性与晚期破坏性不稳定，证明时序信息对失败预测至关重要。

**🔧 技术方法**

计算连续步骤的Jensen–Shannon Divergence与熵，取最大值作为强度；通过相对峰位定位；仅使用top‑k概率，无需模型内部信息。

**📊 数据集**

GSM8K、HotpotQA（部分ReClor验证）。

**📈 对比分析**

与单一熵或JSD指标对比，AUC 0.57–0.78，且在不同模型、解码方式下保持稳健；峰位区分进一步提升诊断分辨率。

**⚠️ 局限性**

仅捕捉动态不稳定失败，无法解释稳定但错误；对top‑k截断敏感；未提供干预或修复方案，需扩展至更大模型和多任务验证。

---

## 108. Learning-augmented smooth integer programs with PAC-learnable oracles

**arXiv ID:** 2602.02505 | [PDF](https://arxiv.org/pdf/2602.02505v1)

**作者:** Hao-Yuan He `[一作]` (Nanjing University), Ming Li `[通讯]` (Nanjing University)

**通讯引用:** 23067 | [OpenAlex ID](https://openalex.org/A5100351402)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于完整预测的学习增强算法框架，用预测 oracle 的输出线性化光滑整数规划目标，求解 LP 并进行随机或贪心舍入，得到一致、平滑且稳健的近似解；并证明该 oracle 在 PAC 学习框架下可学习。

**💡 创新点**

① 将完整信息预测与光滑多项式目标结合，去掉采样阶段，直接在全变量上构造线性逼近；② 通过 β‑光滑性给出预测误差与松弛误差的上界，得到误差与预测误差的显式关系；③ 证明该预测 oracle 具有有限伪维数，因而可通过 ERM 学习到近最优预测器。

**🔧 技术方法**

光滑多项式的分解与线性化；线性规划求解；随机与贪婪（确定性）舍入；PAC 学习与伪维数分析；实验使用 GNN/Transformer 作为预测器。

**📊 数据集**

主要使用合成的密集与近密集图（Max‑Cut）以及合成的 k‑SAT 约束实例（Max‑k‑SAT），并在这些基准上评估算法；预测器采用标准 GNN/Transformer 结构。

**📈 对比分析**

与传统的 worst‑case PTAS、近似算法以及不使用预测的基线并行运行，比较近似比与误差。实验结果显示：在预测误差 ε 较小的情形下，算法接近最优；在近密集实例中，得到 1‑O(√ε/n^ξ) 的近似比，显著优于纯 worst‑case 算法。

**⚠️ 局限性**

仅适用于 β‑光滑的多项式目标；仅在近密集（或更稠密）情形下分析；对 oracle 的训练仅给出样本复杂度，未给出高效训练算法；无法保证每个实例的性能，仅在期望意义下近似；对预测误差的上界仍依赖于 n^d‑1/2 的项。

---

## 109. Outrunning LLM Cutoffs: A Live Kernel Crash Resolution Benchmark for All

**arXiv ID:** 2602.02690 | [PDF](https://arxiv.org/pdf/2602.02690v1)

**作者:** Chenxi Huang `[一作]` (Columbia University), Baishakhi Ray `[通讯]` (Columbia University)

**通讯引用:** 6574 | [OpenAlex ID](https://openalex.org/A5064541855)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个实时更新的 Linux 内核崩溃修复基准和统一的 agent‑agnostic 环境，支持自动采集、编译、执行和评估新发现的崩溃案例。

**💡 创新点**

创新点在于：① 通过持续爬取 Syzbot 报告实现自我进化的 benchmark；② 将编译与测试抽象成可复用的执行层，消除不同 agent 对环境的依赖；③ 引入 Crash‑Resolution Feedback (CRF) 机制，显著提升修复成功率。

**🔧 技术方法**

技术包括 Docker 化的基础镜像与 agent 覆盖层、自动化的 kernel 编译加速（SuiteCache）、基于云的 kernel 执行平台、LLM 判定工具（LLM Judge）以及多维度评估指标（CRR、EPR、IoU）。

**📊 数据集**

数据集为自 2024‑04 起收集的 534 个 Linux 内核崩溃（-2512 数据集），覆盖不同 subsystems、bug 类型，并记录开发者修复提交。

**📈 对比分析**

通过比较三种主流 agent（Mini‑SWE, CodeAct, CrashFixer）在不同 LLM（Gemini‑3 Pro/Flash、Claude Opus 4.5）与是否开启 CRF 的条件下，评估 CRR、EPR 与定位 IoU，结果显示 CRR 最高可达 74%，CRF 可提升 29%，EPR 约 20%。

**⚠️ 局限性**

局限包括：评测仅覆盖有限的 agent 与 LLM，CRF 需要昂贵的编译/测试成本，缺乏对 patch 有效性（如功能安全性）的完整验证，且 benchmark 仍需进一步扩展以覆盖更多内核版本与更复杂的崩溃情形。

---

## 110. RPG-AE: Neuro-Symbolic Graph Autoencoders with Rare Pattern Mining for Provenance-Based Anomaly Detection

**arXiv ID:** 2602.02929 | [PDF](https://arxiv.org/pdf/2602.02929v1)

**作者:** Asif Tauhid `[一作]` (New York University), Talal Rahwan `[通讯]` (New York University)

**通讯引用:** 4347 | [OpenAlex ID](https://openalex.org/A5007282319)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于系统级追溯数据的APT异常检测框架，融合图自动编码器与稀有模式挖掘。

**💡 创新点**

创新点在于将图表示学习与稀有关联规则挖掘结合，通过稀有模式图度量加权提升重建误差，实现结构异常与稀有行为的双重检测与可解释性。

**🔧 技术方法**

采用kNN构建相似图、Graph Autoencoder（GAE）进行无监督链接重建、Apriori稀有模式挖掘、度量加权融合策略以及nDCG评价指标。

**📊 数据集**

在DARPA Transparent Computing公开数据集（包含Linux、Android、BSD、Windows等多操作系统）进行实验。

**📈 对比分析**

与传统GAE、压缩、统计、规则挖掘等单方法基线对比，单模型在nDCG@K上提升约23%（α=2），优于所有单方法基线，且接近多模型集成水平。

**⚠️ 局限性**

限制包括稀有模式阈值与加权因子需手动调参，对实时流式数据的在线适配尚未实现，且模型对极端稀有行为的泛化能力有待进一步验证。

---

## 111. Copula-Based Aggregation and Context-Aware Conformal Prediction for Reliable Renewable Energy Forecasting

**arXiv ID:** 2602.02583 | [PDF](https://arxiv.org/pdf/2602.02583v1)

**作者:** Alireza Moradi `[一作]` (Georgia Institute of Technology), Pascal Van Hentenryck `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 17308 | [OpenAlex ID](https://openalex.org/A5035808622)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种从站点级概率预测直接生成机群级概率预测的框架，结合高斯Copula聚合与上下文感知保形校准，能够在缺乏机群级模型时提供可靠且锐利的概率预测。

**💡 创新点**

创新点在于：①将Copula依赖建模与加权/上下文感知的保形预测相结合，实现对机群级预测的校准与锐化；②使用物理意义的上下文特征（历史功率、时序嵌入、日周期等）来构造加权相似度，使校准更具局部适应性。

**🔧 技术方法**

技术手段包括：高斯Copula建模、Monte Carlo采样聚合、保形预测（CQR）、加权保形预测、RBF相似度、上下文特征构造、指标评估（PICP、AIW、WS）等。

**📊 数据集**

数据集：美国MISO、SPP、ERCOT 2019年太阳能发电时序数据和NREL日间概率量化预测（约1149个站点）。

**📈 对比分析**

与NREL原始机群预测、NREL+保形校准、纯Copula聚合、Copula+保形校准等基线比较。Copula+CACP在90%覆盖率下覆盖率 ≥90%，区间宽度最小，Winkler分数最低，整体性能优于所有对比方法。

**⚠️ 局限性**

局限性：聚合质量依赖站点级预测的准确性，Copula仅捕捉线性相关，可能无法处理尾部/非线性相关；RBF相似度参数需人工调优；目前仅验证单步日间预测，未扩展到多步或其他能源类型。

---

## 112. Weighted Temporal Decay Loss for Learning Wearable PPG Data with Sparse Clinical Labels

**arXiv ID:** 2602.02917 | [PDF](https://arxiv.org/pdf/2602.02917v1)

**作者:** Yunsung Chung `[一作]` (Tulane University), Sharanya Arcot Desai `[通讯]` (Samsung Research America)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种基于加权时间衰减损失的深度学习框架，用以利用可穿戴PPG与稀疏临床标注之间的时间偏差，预测10种生物标志物。

**💡 创新点**

创新点在于学习每个标志物特定的时间衰减率，并将其融入损失函数，从而动态降低远时距样本对训练的影响。

**🔧 技术方法**

采用加权二元交叉熵损失、softplus可学习衰减率、线性/指数/余弦等衰减函数，并以Samsung Galaxy Watch 6的PPG信号训练网络。

**📊 数据集**

使用了450名受试者在2024‑2025年间通过Samsung Galaxy Watch 6采集的连续PPG数据，并与其最近30天内的10种临床生物标志物值配对。

**📈 对比分析**

与基准随机森林和PAPAGEI自监督模型比较，在5折受试者分层交叉验证下，平均AUROC/ AUPRC提升约7.8%/6.1%，在大多数标志物上均超过对照。

**⚠️ 局限性**

局限性包括固定30天窗口可能不适用于所有标志物、单一设备和单一医疗系统导致的领域偏移、以及对中间四分位数的预测未评估。

---

## 113. WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models

**arXiv ID:** 2602.02537 | [PDF](https://arxiv.org/pdf/2602.02537v1)

**作者:** Runjie Zhou `[一作]` (Moonshot AI), Xinyu Zhou `[通讯]` (Moonshot AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出WorldVQA基准，用于评估多模态大型语言模型在将图像直接映射为实体名称（原子视觉知识）上的能力。

**💡 创新点**

通过“原子隔离”原则，将视觉识别与推理严格分离，构建九大类别、3500条VQA对，并采用模型性能分层的难度划分，提供高分辨率的视觉真知度评估。

**🔧 技术方法**

使用多阶段人工与自动双重验证、ISC相似度去重、MetaCLIP词频映射校准难度、ECE与斜率评估模型校准，以及GPT‑oss‑120b等评判模型的技术手段。

**📊 数据集**

以3500条中文/英文VQA对组成的WorldVQA数据集，覆盖Nature、Geography、Culture等九大语义类别，图像来自公开网页并去重后确保无训练泄露。

**📈 对比分析**

与闭源模型（如Gemini‑3‑pro、Claude‑4.5、GPT‑5.2 等）以及开源模型（如Kimi‑VL‑16B、Qwen3‑VL‑235B 等）在 Accuracy、CGA、F‑score 等指标上对比，最佳整体 F‑score 约 47%（Gemini‑3‑pro），Sports 最高 59% 但无模型超过 50% 阈值，显示任务仍具挑战。

**⚠️ 局限性**

局限性在于仅评估原子识别，与复杂下游任务的关联不明；模型普遍存在过度自信问题，且数据受训练对齐策略影响，难以完全消除。

---

## 114. Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral Descent

**arXiv ID:** 2602.03001 | [PDF](https://arxiv.org/pdf/2602.03001v1)

**作者:** Hiroki Naganuma `[一作]` (Mila), Hao-Jun Michael Shi `[通讯]` (Meta Platforms)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了针对非欧几里得优化器（如随机符号下降和随机谱下降）的梯度噪声尺度（GNS）理论，并基于此设计了自适应批量大小策略，同时给出了一种高效的分布式方差估计方法。

**💡 创新点**

创新点包括：
1) 在优化器的对偶范数（ℓ₁、核范数）下推导出 GNS；
2) 将方差估计从单样本扩展到分布式训练中的局部 mini‑batch 梯度；
3) 通过自适应批量大小实现 66% 以上的训练步数缩减，同时保持验证损失与传统固定批量一致。

**🔧 技术方法**

使用技术：
- 随机广义最速下降（SGD、SignSGD、SpectralSGD）
- 对偶范数分析和梯度噪声尺度推导
- 分布式梯度方差估计（DDP/FSDP 上的局部梯度统计）
- 指数滑动平均、批量大小和学习率的自适应更新策略
- 大规模 GPU 训练（H100/A100）与分布式并行框架。

**📊 数据集**

实验数据集：
- 160M 与 1B 参数 Llama 3 在 C4 语料库上训练（3.2B 与 22B token）
- SimpleViT 在 Imagewoof 数据集上训练
- 另外还在多种语言与视觉任务上进行了消融与对比实验。

**📈 对比分析**

比较方法与性能：
- 与固定小批量（如 B=64/128/256）和固定大批量做对比；
- 评估指标为最终验证损失与达到基准损失所需的训练步数；
- 结果显示，Adaptive GNS 能在保持验证损失不变的情况下，
  160M Llama 3 的训练步数减少高达 66.8%，1B Llama 3 约 31.8%/12.1%，
  Vision 任务中 SignSGD/SpectralSGD 等的步数降低 37.5%–55.2%。
- 对于某些优化器（如 SpecSGD 在 1B 任务），Adaptive 方案未能完全达到基准损失。

**⚠️ 局限性**

局限性：
- 目前仅适用于无状态或预条件化的最速下降类优化器，无法直接应用于带动量/自适应矩的 Adam/AdamW/AdamW 等；
- 对不同模块使用不同优化器时，GNS 的统一性与融合仍未解决；
- 方差估计假设局部梯度独立，可能在极端分布偏移或梯度相关性强的情形下失效；
- 未考虑更复杂的 Hessian 结构或二阶信息，限制了在某些高度非凸任务中的适用性。

---

## 115. Failure-Aware Enhancements for Large Language Model (LLM) Code Generation: An Empirical Study on Decision Framework

**arXiv ID:** 2602.02896 | [PDF](https://arxiv.org/pdf/2602.02896v1)

**作者:** Jianru Shen `[一作]` (University of Montana), Lucy Owen `[通讯]` (University of Montana)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5046520666)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对25个GitHub项目进行实证研究，评估进阶提示（progressive prompting）基线以及自评（Self‑Critique）、多模型协作（Multi‑Model Collaboration）和检索增强生成（RAG‑Assisted）三种增强策略在LLM代码生成中的效果。

**💡 创新点**

提出了基于失败类型的分类体系，并构建了一个决策框架，将不同失败模式与最合适的增强方法对应，为实践者提供数据驱动的使用指导。

**🔧 技术方法**

使用GPT‑5和Claude Sonnet 4.5等大语言模型，结合进阶提示、迭代自评、跨模型协作和检索增强生成等技术。

**📊 数据集**

采用25个公开GitHub项目（3–31个任务不等），其中6个因进阶提示失败而挑选为挑战项目进行深入评估。

**📈 对比分析**

通过与直接提示基线对比以及在挑战项目上对三种增强方法的任务完成率、时间与提示次数等指标评估，RAG‑Assisted取得最高完成率（≈99.2%）和最佳效率（≈31.5 min/pp），Self‑Critique仅对局部逻辑错误有效，多模型协作可靠但耗时最长。

**⚠️ 局限性**

局限性包括：评估基于手工判定完成度，受具体LLM实现与版本影响；样本仅覆盖文档完善的Web/企业项目，未验证在嵌入式或安全关键领域的适用性；时间效率数据受硬件和网络环境影响。

---

## 116. Self-Supervised Uncalibrated Multi-View Video Anonymization in the Operating Room

**arXiv ID:** 2602.02850 | [PDF](https://arxiv.org/pdf/2602.02850v1)

**作者:** Keqi Chen `[一作]` (University of Strasbourg), Nicolas Padoy `[通讯]` (IHU Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在OR中使用无标注、无标定的多视角视频自监督框架进行整形体检测与姿态估计，实现人像匿名化

**💡 创新点**

创新点是结合时序跟踪和自监督多视角关联恢复漏检，并通过伪标签迭代微调，实现零注释域自适应

**🔧 技术方法**

使用ByteTrack跟踪、Self‑MVA无标定多视角关联、P‑D‑DETR/RTMPose等检测/姿态模型，配合自监督伪标签迭代

**📊 数据集**

在4D‑OR模拟手术和自采真实手术数据集上进行评估

**📈 对比分析**

相较于RetinaFace、YOLOv8、Iter‑Score等方法，本文在面部和眼部检测召回率超过97%，显著提升精度并节省约15小时人工复核时间

**⚠️ 局限性**

局限包括：当某人被所有视角完全遮挡或仅出现极少帧时无法检测；需同步固定摄像头；可能出现伪关联

---

## 117. MARS: Modular Agent with Reflective Search for Automated AI Research

**arXiv ID:** 2602.02660 | [PDF](https://arxiv.org/pdf/2602.02660v1)

**作者:** Jiefeng Chen `[一作]`, Jinsung Yoon `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MARS 框架，用于自动化 AI 研究任务，特别是机器学习工程（MLE）

**💡 创新点**

创新点包括预算感知 MCTS、模块化实现管道以及对比反射记忆来解决信用分配问题

**🔧 技术方法**

核心技术有预算感知蒙特卡洛树搜索（Budget‑Aware MCTS）、Design‑Decompose‑Implement 模块化流程、以及 Lesson Learning 的对比反射记忆机制

**📊 数据集**

使用 MLE‑Bench（75 个 Kaggle 竞赛）作为评测数据集，并在控制环境下进行多轮实验

**📈 对比分析**

与 AIDE、AIRA、ML‑Master 2.0 等公开框架及官方排行榜进行对比，MARS 在任何奖牌率、金牌率等指标上均超过或接近最高水平，尤其在金牌率达到 31.1% 的同时保持高效资源利用

**⚠️ 局限性**

局限性在于成本较高（维护大量记忆上下文），仅在 MLE 领域验证，且对超大规模任务的扩展性和跨领域通用性仍待进一步验证

---

## 118. Generative Engine Optimization: A VLM and Agent Framework for Pinterest Acquisition Growth

**arXiv ID:** 2602.02961 | [PDF](https://arxiv.org/pdf/2602.02961v1)

**作者:** Faye Zhang `[一作]` (Pinterest), Kofi Boakye `[通讯]` (Pinterest)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 Pinterest GEO 框架，使用 Vision‑Language 模型生成用户真实搜索意图的查询，聚合成语义连贯的集合页，并通过两塔 ANN 结构传播权威，从而提升生成搜索流量。

**💡 创新点**

创新点包括：逆向搜索 VLM 生成意图导向查询；实时趋势代理挖掘未来需求；多模态嵌入聚合集合；将两塔 MLP 与 HNSW ANN 结合构建大规模权威传播网络。

**🔧 技术方法**

使用技术包括：预训练 Qwen2‑VL‑7B‑Instruct + LoRA 微调、GPT‑4V 生成合成标签、LangGraph ReAct 代理挖掘趋势、PinCLIP/ SearchSAGE 嵌入、HNSW ANN、两塔 MLP 交叉排名及安全评估管道。

**📊 数据集**

数据集涵盖 Pinterest 内部千亿图像及 Pin、Google Search Console 与 Pinterest 搜索日志、200k GPT‑4V 合成查询、跨语言用户交互与行为数据。

**📈 对比分析**

通过 A/B 测试与基线（ANN+VASE、传统 caption）比较，使用 ROUGE‑1、GPT‑4o 语义评估和人工评测；结果显示 VLM 生成查询提升 20% 有机流量，搜索引擎流量 9.2×，两塔+ANN 方案在推理成本上降低 94×。

**⚠️ 局限性**

局限性包括：依赖历史搜索数据导致对搜索引擎算法变化反应延迟；生成查询多样性仍低于大规模检索；缺乏因果分析以区分权威与相关性；模型对多模态质量与偏见的控制仍有限。

---

## 119. A two-player version of the assignment problem

**arXiv ID:** 2602.02628 | [PDF](https://arxiv.org/pdf/2602.02628v1)

**作者:** Florian Galliot `[一作]` (Aix-Marseille University), Jonas Sénizergues `[通讯]` (Université de Bordeaux)

**通讯引用:** 27 | [OpenAlex ID](https://openalex.org/A5027748422)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个两人版的分配问题（draft game），研究玩家在轮流挑选代理人后各自最大化任务分配收益的博弈，定义得分为双方分配值之差。

**💡 创新点**

创新点在于：①首次将分配问题转化为可计分的两人博弈；②证明该博弈在通用情况下是 PSPACE‑complete，甚至仅允许每个代理人有最多两个非零效率时；③针对只有一个非零效率（OTP）情形给出 XP 复杂度和两任务情形下的线性时间算法。

**🔧 技术方法**

使用的技术包括：组合博弈理论（Milnor 的宇宙、Maker‑Breaker 位置游戏）、多项式时间的替代性与普遍性量化归约（QBF → 博弈），以及基于动态规划与状态压缩的求解算法。

**📊 数据集**

论文未使用实验数据集，而是以理论证明与复杂度分析为主要结果。

**📈 对比分析**

方法的比较基于理论复杂度和可解性阈值；在通用情形下证明为 PSPACE‑complete，证明其难度与 QBF 相当；在 OTP 情形下给出最优解的线性/XP 计算算法，展示了在特定约束下问题可解。

**⚠️ 局限性**

局限性：①对任意多任务的 OTP 情形仍未得到精确复杂度（尚未知是否 NP‑hard）；②未给出具体的策略实现或实验验证；③仅讨论了得分为差值的 Maker‑Maker 版本，未全面探讨 Maker‑Breaker 版本的性能。

---

## 120. Search-Augmented Masked Diffusion Models for Constrained Generation

**arXiv ID:** 2602.02727 | [PDF](https://arxiv.org/pdf/2602.02727v1)

**作者:** Huu Binh Ta `[一作]` (University of Virginia), Ferdinando Fioretto `[通讯]` (University of Virginia)

**通讯引用:** 1275 | [OpenAlex ID](https://openalex.org/A5052534316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SearchDiff，一种在离散扩散模型的逆向扩散过程中嵌入搜索的训练无关推理框架，用于满足硬性约束和非可微属性。

**💡 创新点**

创新点在于将候选搜索采样（CSS）与局部搜索（LS）直接集成到每一步逆向扩散中，实现对约束空间的即时引导，并且不需要额外训练或梯度信息。

**🔧 技术方法**

采用的技术包括：掩码离散扩散（Masked Diffusion）、基于模型预测的候选分布、黑盒约束违约函数、局部搜索优化、修改后的逆向转移核；同时利用 Transformer 作为 denoiser。

**📊 数据集**

实验使用的主要数据集包括：QM9（分子 SMILES）、蛋白/肽序列数据集、tRNA 样本、数独、3-SAT 逻辑题。

**📈 对比分析**

与多种基线比较（MDLM、CBG、CFG、TreeG‑SC、FUDGE、PPLM、GPT‑2）后，SearchDiff 在所有任务上显著提升约束满足率和属性质量，例如：分子任务可接受样本从 142 增至 498，QED 均值从 0.62 提升至 0.77；肽任务 0% 违约率；tRNA 约束完整率提升 27 倍；Boolean SAT 成功率从 9.6% 提升至 76%。

**⚠️ 局限性**

局限性包括：搜索过程在每一步产生额外计算开销，且搜索深度受预算限制；在极大序列或极其复杂、耦合的约束下可能难以逃离局部最优；仍需预先训练好的扩散模型和可用的黑盒约束评估器。

---

## 121. Towards Understanding Steering Strength

**arXiv ID:** 2602.02712 | [PDF](https://arxiv.org/pdf/2602.02712v1)

**作者:** Magamed Taimeskhanov `[一作]` (University of Wurzburg), Damien Garreau `[通讯]` (Universite Cote d'Azur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大语言模型的激活调节（steering）强度进行理论与实证分析，探讨其对下一词预测、概念出现概率和交叉熵的影响。

**💡 创新点**

首次提供了针对差分均值steering向量的理论推导，揭示了强度的非单调性和“甜点”范围，并给出跨模型的验证。

**🔧 技术方法**

使用差分均值steering、Transformer UFM模型、Softmax与交叉熵分析，并在多层、不同规模LLM上实施激活添加。

**📊 数据集**

实验基于人工构造的概念词表（如大小写、正负对照）和公开大型语料库，生成正负对照提示集。

**📈 对比分析**

与传统的无强度或经验选取方法相比，本文证明了在适中强度下可提升目标概念出现率，而大强度导致交叉熵上升，实验在12种模型上验证了理论预期。

**⚠️ 局限性**

局限在于理论模型假设上下文单一概念、无归一化等，未考虑混合概念、动态提示等实际情况。

---

## 122. A Multi-scale Linear-time Encoder for Whole-Slide Image Analysis

**arXiv ID:** 2602.02918 | [PDF](https://arxiv.org/pdf/2602.02918v1)

**作者:** Jagan Mohan Reddy Dwarampudi `[一作]`, Tania Banerjee `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种纯Mamba基的多尺度多实例学习框架MARBLE，用于全切片图像的高效分析。

**💡 创新点**

首次将多尺度粗细融合与线性时间状态空间模型结合，避免二次注意力开销，且支持并行处理多倍缩放。

**🔧 技术方法**

采用Mamba-2状态空间块、轻量级父子token融合、注意力池化与Cox回归等技术。

**📊 数据集**

在PANDA、TCGA‑NSCLC（5×/20×、10×/40×）以及KIRP、LUAD、STAD等TCGA队列上进行实验。

**📈 对比分析**

与ABMIL、CLAM、S4‑MIL、MambaMIL等基线对比，MARBLE在分类AUC提升6.9pp、准确率提升20.3pp，生存C‑index提升2.3pp，整体表现最佳。

**⚠️ 局限性**

仅支持两级缩放；对更高多级结构及可变尺寸的图像仍需改进。

---

## 123. PeerRank: Autonomous LLM Evaluation Through Web-Grounded, Bias-Controlled Peer Review

**arXiv ID:** 2602.02589 | [PDF](https://arxiv.org/pdf/2602.02589v1)

**作者:** Yanki Margalit `[一作]` (Caura.ai), Nurit Cohen-Inger `[通讯]` (Computer Science and Information, Ben-Gurion University of the Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个完全自闭环的多模型评估框架PeerRank，让LLM自己生成问题、回答（可实时上网检索）、相互评判并聚合成排名，消除了人工基准和参考答案的依赖。

**💡 创新点**

创新点在于：①端到端自主任务生成与评判；②评估过程中分离答案检索与评判，保持评判盲目；③通过洗牌+盲审三种协议系统量化并校正自我偏好、身份偏好和位置偏好。

**🔧 技术方法**

技术包括：多模型协同工作（生成器、答复者、评判者）、基于Web检索的答案生成、1–10分打分的评判模板、均值与Elo两种聚合方式、统计偏差量化与标准差置信区间。

**📊 数据集**

数据集：自主生成420个问题（5类），外部检验使用TruthfulQA（264题）和GSM8K（611题）。

**📈 对比分析**

方法对比：对TruthfulQA和GSM8K的客观正确率，PeerRank得分与真实准确率相关性高（TruthfulQA Pearson 0.90，GSM8K 0.87），并与Elo排名高度一致，表明排名稳定且与客观性能正相关。

**⚠️ 局限性**

局限性：①仅提供相对排名，缺乏跨实验绝对基准；②任务分布受生成模型偏好影响，可能忽略某些领域；③网络检索和API延迟受外部因素干扰；④样本规模有限，细粒度统计能力有限；⑤评判主观性仍然存在，无法完全消除。

---

## 124. Prefix Consensus For Censorship Resistant BFT

**arXiv ID:** 2602.02892 | [PDF](https://arxiv.org/pdf/2602.02892v1)

**作者:** Zhuolun Xiang `[一作]` (Aptos Labs), Alexander Spiegelman `[通讯]` (Aptos Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出 Prefix Consensus 这一新的共识原语，并基于它构造了 Strong Prefix Consensus、无领导多提议的 BFT 状态机复制协议，实现了在异步与准同步网络中对抗审查、支持无领导进度保证。

**💡 创新点**

创新点在于：①首次证明 Prefix Consensus 可在 n=3f+1 的异步环境下用 3 轮完成，给出匹配的上下界；②将其升阶为 Strong Prefix Consensus，得到无领导、可达成 f‑censorship‑resistance 的多槽协议；③通过该原语改进了分级共识、二值共识和验证式共识的消息与通信复杂度；④提出通过排名动态下调被审查提议者来实现审查抵抗。

**🔧 技术方法**

使用签名证书与投票证书（QC）实现的多轮投票协议；基于哈希向量的投票表述；循环移位排名和父指针链的视图切换；可验证的低/高输出证明；以及对准同步网络的 GST 停止时间假设。

**📊 数据集**

论文未使用任何实验数据集；主要通过理论证明、复杂度分析与与已有协议的对比评估。

**📈 对比分析**

在消息复杂度上，Prefix Consensus 3 轮实现 O(n²) 消息；Strong Prefix Consensus 采用 O(n³) 消息、O(n⁴) 通信；通过其构造的验证式共识可达到 O(n³) 消息。与传统的 PBFT、DBFT 等领导者协议相比，具有更低的消息与通信负载；在时间层面，3 轮异步实现实现了与已知最优一致性等价的延迟。

**⚠️ 局限性**

局限性包括：① 2 轮实现仅在 n ≥ 5f+1 条件下可行，且与下界存在间隙；② 仍需 O(n³) 或 O(n⁴) 的通信成本，尚未完全最优；③ 对准同步网络的 GST 假设与实际网络延迟模型的适配仍需进一步实验验证；④ 对高容错量（如更大 f）的实用性尚未完全评估。

---

## 125. Graph-Augmented Reasoning with Large Language Models for Tobacco Pest and Disease Management

**arXiv ID:** 2602.02635 | [PDF](https://arxiv.org/pdf/2602.02635v1)

**作者:** Siyu Li `[一作]` (Chongqing Jiaotong University), Xinyi Liu `[通讯]` (Chongqing Jiaotong University)

**通讯引用:** 4435 | [OpenAlex ID](https://openalex.org/A5100395652)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了烟草病虫害知识图谱，并将其与 GraphRAG 框架结合，让 LLM 在回答问题时能够访问结构化的关系证据；

**💡 创新点**

创新点在于将 TransE 与 GCN 的图嵌入相结合，显式编码症状–疾病–处理的多跳关系，并将检索到的图子图作为上下文注入 LLM，从而提升多跳推理与对比推理能力；

**🔧 技术方法**

采用了 GraphRAG、TransE 关系嵌入、GCN 节点表示学习、ChatGLM+LoRA 参数高效微调以及图与文本融合的输入方式；

**📊 数据集**

使用了基于专家文献、扩展指南和研究论文构建的烟草病虫害知识图谱，以及自制的多跳与比较题型 QA 数据集；

**📈 对比分析**

与 ChatGLM、KGE+ChatGLM、RAG+ChatGLM 进行对比，GraphRAG+ChatGLM 在准确率 90.1%、精确率 92.3%、召回率 88.2% 和 F1 90.2% 上均显著优于基线；

**⚠️ 局限性**

主要局限在于知识图谱覆盖不足、实体对齐不完善导致某些推理失败，且对模糊或不完整的用户问题仍然存在处理难度。

---

## 126. Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems

**arXiv ID:** 2602.02582 | [PDF](https://arxiv.org/pdf/2602.02582v1)

**作者:** Chandan Kumar Sah `[一作]` (Beihang University), Syed Shazaib Shah `[通讯]` (Beihang University)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5108828076)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在推荐系统中的不确定性与公平性问题，提出了基于熵的不确定性量化与个性化公平性评估框架，并在电影和音乐两个领域进行了案例实验。

**💡 创新点**

创新点包括：①将预测熵与相似度指标结合用于评估LLM推荐的不确定性；②引入个性化公平性得分（PAFS），从人格维度审视偏差；③构建涵盖8个敏感属性、31个类别的对齐数据集；④对Gemini 1.5 Flash 在多种提示扰动下的公平性鲁棒性进行了系统验证。

**🔧 技术方法**

采用熵量化不确定性、相似度指标（SNSR、SNSV、Jaccard、SERP、PRAG）以及新颖的PAFS；使用Google Gemini 1.5 Flash API进行推理；设计多样化提示（含敏感属性、拼写错误、多语言）进行鲁棒性实验。

**📊 数据集**

使用电影领域1000位导演（500热门+500多样）与音乐领域1000位艺术家（MTV Top 10k选取）的公开数据，并人工标注8个敏感属性（年龄、洲、国籍、性别、职业、体型、种族、宗教）共31类，构成实验数据集。

**📈 对比分析**

通过对比相似度差异（SNSR/SNSV）与传统Jaccard、SERP、PRAG指标，评估Gemini在不同敏感属性上的不公平性。结果显示，在音乐域PRAG*25下SNSV最高达0.18，电影域最高0.12，且在拼写错误与法语提示下差距仍维持，表明当前模型存在系统性偏差；相比传统评估，加入个性化公平性后可更细粒度揭示差异。

**⚠️ 局限性**

研究仅评估了Gemini 1.5 Flash，缺乏跨模型验证；数据量有限且只涉及两个领域，无法覆盖更广泛场景；个性化评价仅基于有限的属性集合，缺乏更完整的人格模型；实验使用静态提示，未考虑动态用户交互与长期部署效果。

---

## 127. TinyGuard:A lightweight Byzantine Defense for Resource-Constrained Federated Learning via Statistical Update Fingerprints

**arXiv ID:** 2602.02615 | [PDF](https://arxiv.org/pdf/2602.02615v1)

**作者:** Ali Mahdavi `[一作]` (Islamic Azad University), Amirfarhad Farhadi `[通讯]` (Iran University of Science and Technology)

**通讯引用:** 122 | [OpenAlex ID](https://openalex.org/A5091487639)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在联邦学习中提出一种轻量级的Byzantine防御方法TinyGuard，利用统计指纹对客户端梯度进行异常检测，同时保持FedAvg聚合不变。

**💡 创新点**

创新点在于把高维梯度压缩成低维统计指纹（范数、层比、时刻矩、稀疏度等），并在O(nd)时间内完成检测，无需全梯度比较或历史状态。

**🔧 技术方法**

技术包括统计特征提取、鲁棒中心（中位数）与MAD距离阈值判别，以及适配器LoRA训练下的Transformer微调。

**📊 数据集**

使用MNIST、Fashion‑MNIST、轻量级ViT‑Lite以及ViT‑Small（带LoRA）四个数据集进行实验。

**📈 对比分析**

与Krum、Trimmed Mean、FoolsGold等基线对比，TinyGuard在多数攻击场景下保持与FedAvg相同或更高的准确率（≈95%），同时运行时显著降低。

**⚠️ 局限性**

局限性包括仅在至多22 M参数模型和150客户端规模内验证，缺乏理论收敛和更高级攻击的实证；对更大规模基础模型的性能尚待验证。

---

## 128. When Efficient Communication Explains Convexity

**arXiv ID:** 2602.02821 | [PDF](https://arxiv.org/pdf/2602.02821v1)

**作者:** Ashvin Ranjan `[一作]` (University of Washington), Shane Steinert-Threlkeld `[通讯]` (University of Washington)

**通讯引用:** 559 | [OpenAlex ID](https://openalex.org/A5017484646)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过信息瓶颈（IB）框架研究了语言语义系统中的凸性（convexity）与高效通信之间的关系，分别在颜色命名数据和人工构造语义空间上实证分析并探讨了影响二者关联的关键因素；

**💡 创新点**

提出了针对概率分布的“准凸性”度量，将凸性概念推广到IB编码器，并系统评估凸性与通信最优性之间的因果关联，尤其指出通信需求分布的凸性是驱动该关联的核心因素；

**🔧 技术方法**

信息瓶颈理论、逆向确定性退火（reverse deterministic annealing）、基于贝叶斯推理的子最优编码器采样、准凸性度量算法；

**📊 数据集**

世界颜色调查（WCS）颜色标本数据（CIELab 颜色空间）以及一系列人工设计的参考物与意义分布的离散/连续环境；

**📈 对比分析**

通过计算编码器的准确度（I(W;U)）、复杂度（I(M;W)）与准凸性，并与自然语言编码器及随机打乱的子最优编码器进行比较；在颜色命名实验中，准凸性与最优性正相关（r≈0.3），与准确度/复杂度相关性更强；在人工环境实验中，凸性先验导致高相关性，非凸先验则相反；

**⚠️ 局限性**

实验仅关注二维/一维简化空间，未覆盖更高维语义域；准凸性度量的参数选择（步长等）未系统化；缺乏理论证明凸性先验必导致IB最优编码器凸性；未来需在更广泛的语义空间与连续先验上验证结论。

---

## 129. What Drives Length of Stay After Elective Spine Surgery? Insights from a Decade of Predictive Modeling

**arXiv ID:** 2602.02517 | [PDF](https://arxiv.org/pdf/2602.02517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 130. A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization

**arXiv ID:** 2602.02877 | [PDF](https://arxiv.org/pdf/2602.02877v1)

**作者:** Xiyuan Wei `[一作]` (Texas A&M University), Tianbao Yang `[通讯]` (Texas A&M University)

**通讯引用:** 5980 | [OpenAlex ID](https://openalex.org/A5023288846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对组合熵风险最小化（CERM）问题的几何感知随机算法SCENT，利用随机原型镜像下降（SPMD）更新对偶变量，解决了传统算法在数值不稳定、收敛慢和偏差梯度估计等方面的缺陷。

**💡 创新点**

创新点在于：①设计了基于负指数生成Bregman散度的SPMD更新，可显著降低对偶变量的光滑度常数影响；②在凸设置下给出了O(1/√T)的理论收敛率，并与传统SGD做了量化比较；③通过对方差项的细致分析，揭示SPMD相较于SGD在指数尺度上优越的特性。

**🔧 技术方法**

技术核心包括：min–min 形式的双对偶建模、随机原型镜像下降（SPMD）、Bregman散度（由φ(ν)=e^{−ν}诱导）、随机块坐标更新、以及与Adam/动量结合的梯度更新。

**📊 数据集**

在实验中使用了三大数据集：Glint360K（360k 类人脸）、TreeOfLife-10M（10M 类生物）用于极端分类；CIFAR‑10 与 CIFAR‑100 的二分类版本用于部分 AUC 最大化。

**📈 对比分析**

与BSGD、ASGD、SOX、U‑max、Softplus等基线比较，SCENT 在极端分类与部分 AUC 最大化任务上均表现最佳，收敛更快、最终损失更低，尤其在 TreeOfLife-10M 与 CIFAR‑100 上差距明显。

**⚠️ 局限性**

局限性主要包括：①目前仅在凸场景下给出收敛分析，非凸情况缺乏理论保证；②算法在对偶变量更新上仍需要额外的步长调参，尤其是对α_t 的上界；③实验主要聚焦于大规模分类与 AUC，其他 CERM 典型任务（如对比学习、DRO）实验结果未公开。

---

## 131. Provable Effects of Data Replay in Continual Learning: A Feature Learning Perspective

**arXiv ID:** 2602.02767 | [PDF](https://arxiv.org/pdf/2602.02767v1)

**作者:** Meng Ding `[一作]` (University of Science and Technology of China), Kaiyi Ji `[通讯]` (State University of New York at Buffalo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文基于特征学习视角，构建了全数据重放（full data replay）在连续学习中的理论框架；

**💡 创新点**

创新点在于将信噪比（SNR）与任务间相关性结合，揭示任务排序对遗忘的影响，并证明在满足特定SNR条件下重放可恢复甚至提升早期任务；

**🔧 技术方法**

采用多视角数据模型、两层卷积网络（cubic 激活）与随机梯度下降的训练过程；

**📊 数据集**

使用合成数据生成任务，每个任务拥有独立的信号向量与噪声，实验中设置不同SNR与任务相关性；

**📈 对比分析**

通过理论推导与合成实验验证，展示在高SNR或高相关性时优先训练高信号任务能显著提升后续任务表现并抑制灾难性遗忘；

**⚠️ 局限性**

局限性包括仅在无限内存、合成数据与理论假设下验证，未在真实数据集与有限缓冲场景下实证。

---

## 132. NSC-SL: A Bandwidth-Aware Neural Subspace Compression for Communication-Efficient Split Learning

**arXiv ID:** 2602.02696 | [PDF](https://arxiv.org/pdf/2602.02696v1)

**作者:** Zhen Fang `[一作]` (Xiamen University of Technology), Shunzhi Zhu `[通讯]` (Xiamen University of Technology)

**通讯引用:** 1196 | [OpenAlex ID](https://openalex.org/A5101942006)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种面向分割学习的自适应子空间压缩框架NSC‑SL，用于降低在有限带宽下的通信开销。

**💡 创新点**

创新点在于结合带宽感知的自适应秩选择与错误补偿的交替子空间逼近，能够实时调节压缩率并显著减少截断误差。

**🔧 技术方法**

主要技术包括随机子空间估计、低秩逼近、交替正交迭代、残差反馈循环以及动态秩选择。

**📊 数据集**

实验使用 HAM10000 皮肤病变图像数据集，并以 ResNet‑18 作为分割学习模型。

**📈 对比分析**

与 ACPSGD、RandTopk、QSGD 等基线相比，NSC‑SL 在不同带宽条件下实现了更低的 MSE、提升的准确率，并保持了更快的收敛速度。

**⚠️ 局限性**

主要限制是仍需额外计算资源来估算奇异值谱，对极低秩或极大矩阵的性能尚未全面验证。

---

## 133. Monotonicity as an Architectural Bias for Robust Language Models

**arXiv ID:** 2602.02686 | [PDF](https://arxiv.org/pdf/2602.02686v1)

**作者:** Patrick Cooper `[一作]` (University of Colorado Boulder), Alvaro Velasquez `[通讯]` (University of Colorado Boulder)

**通讯引用:** 41708 | [OpenAlex ID](https://openalex.org/A5108001041)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Transformer的FFN子层加入单调性约束，提升语言模型在对抗攻击下的鲁棒性

**💡 创新点**

只在非注意力子层强制单调，利用软加权重重参数化保持表达能力，并在不牺牲任务性能的前提下显著降低攻击成功率

**🔧 技术方法**

软加权重重参数化（log‑exp）保证权重非负；A‑monotone FFN子层；对抗攻击评估（HotFlip、UAT）；对比T5‑small微调与预训练模型

**📊 数据集**

CNN/DailyMail 与 XSUM 语料库用于摘要评估，对抗攻击在相同测试集上评估

**📈 对比分析**

与基线T5‑small微调模型及预训练模型对比；ROUGE‑L 下降约3%，但HotFlip攻击成功率从63%降至19%，平均降解率从16%降至5%，显示鲁棒性显著提升

**⚠️ 局限性**

仅在T5‑small实验，未给出全局鲁棒性证明；单调约束导致表达力略低；对更大模型、不同任务或多模态场景的效果尚未验证

---

## 134. UNSO: Unified Newton Schulz Orthogonalization

**arXiv ID:** 2602.02500 | [PDF](https://arxiv.org/pdf/2602.02500v1)

**作者:** Chen Hu `[一作]` (Sun Yat-sen University), Xiyin Li `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出统一 Newton–Schulz 正交化（UNSO）方法，将传统迭代结构统一为单步多项式操作，利用可学习系数优化多项式并显著减少矩阵乘法量。

**💡 创新点**

创新点在于：①将迭代过程压缩为单一多项式表达；②采用指数增长参数 n_k 取代线性步长；③通过学习可调系数实现对奇数多项式的自适应优化；④在此框架下消除冗余乘法，显著降低计算成本。

**🔧 技术方法**

使用技术包括：Newton–Schulz 迭代、奇数多项式构造、可学习系数优化（Adam）、Chebyshev 加速、Gelfand 公式、PyTorch 训练、Adam 及学习率衰减调度。

**📊 数据集**

实验使用随机生成的矩阵（128×128、128×512、128×1024），并未采用公开数据集。

**📈 对比分析**

与原始 NS、Muon’s NS、Cesista’s NS、CANS 四种基线在误差（E=√ΣE²）和 FLOPs 上进行比较；UNSO 在误差上最低（0.04）并且 FLOPs 较低（例如 128×512 时 8.8×10⁷ vs 2.5×10⁸），表现优异。

**⚠️ 局限性**

局限性：收敛速度相对较慢；在接近 1 的过程中出现波动；当宽度 w ≫ 高度 h 时仍需 X 乘法导致计算成本不完全消除。

---

## 135. Beyond the Prompt: Assessing Domain Knowledge Strategies for High-Dimensional LLM Optimization in Software Engineering

**arXiv ID:** 2602.02752 | [PDF](https://arxiv.org/pdf/2602.02752v1)

**作者:** Srinath Srinivasan `[一作]` (North Carolina State University), Tim Menzies `[通讯]` (North Carolina State University)

**通讯引用:** 14454 | [OpenAlex ID](https://openalex.org/A5077008083)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统评估四种将领域知识注入 LLM 的架构（人类专家迭代提示 H-DKP、适应性多阶段提示 AMP、维度递进精炼 DAPR、统计+检索混合 HKMA）在高维软件工程优化中生成 warm‑start 的效果。

**💡 创新点**

创新点：①首次将专家反馈循环与 LLM 结合，形成可迭代的 H‑DKP；②提出多阶段自我推理流程 AMP；③通过逐步扩展维度的 DAPR 解决维度灾难；④把轻量统计探索（TPE）与检索增强生成（RAG）相结合的 HKMA；四种架构系统对比，填补 LLM 在高维任务中表现不佳的空白。

**🔧 技术方法**

使用的技术包括：大语言模型（LLM）推理、Retrieval‑Augmented Generation、Tree‑structured Parzen Estimator、Gaussian Process（UCB‑GPM）、Chebyshev 距离评估、Scott‑Knott 分层检验、Cliff's Delta 影响量、主动学习、特征重要性排名（Spearman、互信息、随机森林）等。

**📊 数据集**

数据集来源于 MOOT 仓库，按维度分层（<6、6‑11、>11），共计约 30‑40 个多目标软件工程优化数据集，涵盖软件配置、过程建模、超参数调优等场景。

**📈 对比分析**

比较方法：在每个数据集上分别执行随机、UCB‑GPM、标准 LLM warm‑start（BS_LLM）以及四种新方法；对每种方法生成 20 次 warm‑start，取最优 Chebyshev 距离；采用 Scott‑Knott 聚类和 Cliff's Delta 判断统计显著性。实验结果显示：在人类专家支持的 H‑DKP 在高维任务上显著优于其他方法；AMP 与 HKMA 在低/中维任务上实现显著提升；DAPR 在中维任务与 AMP 相当；传统 GPM 在最复杂高维任务中仍保持领先。

**⚠️ 局限性**

局限性：①人类专家需求高，招募成本与时间限制明显；②实验规模受 LLM API 费用约束，只能覆盖 MOOT 子集；③缺乏对极高维度（>50）或工业真实环境的验证；④对模型泛化、可解释性及长期稳定性未深入探究。

---

## 136. Agentic Observability: Automated Alert Triage for Adobe E-Commerce

**arXiv ID:** 2602.02585 | [PDF](https://arxiv.org/pdf/2602.02585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 137. When pre-training hurts LoRA fine-tuning: a dynamical analysis via single-index models

**arXiv ID:** 2602.02855 | [PDF](https://arxiv.org/pdf/2602.02855v1)

**作者:** Gibbs Nwemadji `[一作]` (International School of Advanced Studies), Jean Barbier `[通讯]` (Abdus Salam International Centre for Theoretical Physics)

**通讯引用:** 1185 | [OpenAlex ID](https://openalex.org/A5057057294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 LoRA 微调在单索引模型上的学习动态进行理论分析，揭示过度预训练会导致搜索阶段延长，减慢收敛。

**💡 创新点**

提出了“逃逸时间”概念，量化预训练对 LoRA 收敛的双重影响，并通过 Hermite 展开揭示了激活函数和预训练强度之间的奇异相互作用。

**🔧 技术方法**

采用一遍 SGD、LoRA 低秩更新、对齐（μ）分析、Hermite 多项式展开以及概率极限方法对学习动力学进行精确刻画。

**📊 数据集**

使用基于高维正态分布的合成单索引数据集（输入为 N(0,I_d)，标签为 ϕ(⋆·x)）。

**📈 对比分析**

与经典的全微调和随机初始化 LoRA 做对比，理论预测与数值实验高度吻合；结果显示在 μ 接近 1 或激活函数奇异时，预训练会显著延长收敛时间；通过标签平方等预处理可消除奇异并加速学习。

**⚠️ 局限性**

局限性包括仅在单索引模型框架下证明，假设输入为高斯分布，且未考虑更复杂网络结构或实际数据集的多样性；对 LoRA 最高秩 R 的影响仅在理论上讨论，未给出完整的实证验证。

---

## 138. TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation

**arXiv ID:** 2602.02891 | [PDF](https://arxiv.org/pdf/2602.02891v1)

**作者:** Prajna G. Malettira `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**通讯引用:** 45990 | [OpenAlex ID](https://openalex.org/A5031161187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不进行任何训练的情况下，通过零样本NAS框架寻找最优的非均匀结构化剪枝方案，快速得到可在后续恢复训练中表现优异的LLM子模型。

**💡 创新点**

创新点在于提出一种基于梯度迹相似度的尺度不变零样本代理Φ，能够衡量剪枝后模型在预训练损失景观上的功能继承性，并通过低秩梯度子空间实现高效搜索。

**🔧 技术方法**

核心技术包括：梯度迹相似度评价、低秩LoRA子空间梯度计算、基于激活权重乘积的通道重要性评估、进化搜索（深度+宽度联合演化）以及一次性内存内掩码实现。

**📊 数据集**

实验使用的主要数据集包括FineWeb‑Edu 100BT进行校准与恢复训练，评估基准包括ARC‑easy、LogiQA、PIQA、SciQ、BoolQ、MMLU、WinoGrande、HellaSwag、ARC Challenge等。

**📈 对比分析**

与现有结构化剪枝方法（如ShearedLLaMA、E³‑Pruner、DarwinLM）及统一剪枝基线相比，TraceNAS在同等参数约束下实现了10×更低的GPU小时开销，同时在多项推理基准上取得了2–4%更高的准确率。

**⚠️ 局限性**

局限性包括：仍需对不同LLM架构（如GQA）进行更细粒度的调优，代理在极端稀疏率下可能不再完全准确，且在极大模型规模时低秩梯度子空间的选择仍需经验性指导。

---

## 139. Simulating Human Audiovisual Search Behavior

**arXiv ID:** 2602.02790 | [PDF](https://arxiv.org/pdf/2602.02790v1)

**作者:** Hyunsung Cho `[一作]` (Aalto University), Antti Oulasvirta `[通讯]` (Aalto University)

**通讯引用:** 14360 | [OpenAlex ID](https://openalex.org/A5003084232)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种资源合理化的计算模型，统一感知融合与身体动作成本，模拟人类在不确定环境下的视听搜索行为。

**💡 创新点**

创新点在于：① 将多模态感知（ITD/ILD 与视觉）与身体动作（头转、步行、等待、提交）嵌入同一 POMDP 框架；② 通过强化学习学习资源合理化策略，实现对人类搜索时间、转向、步行与错误的预测；③ 展示模型能解释典型的搜索错误模式。

**🔧 技术方法**

技术包括：POMDP 建模、贝叶斯感知更新（ITD 高斯模型、视觉可见性与特征匹配）、动作空间离散化（30° 旋转、1m 步行）、PPO 强化学习、Unity 3D VR 实验与 Steam Audio 空间音频。

**📊 数据集**

数据集：270 条 VR 车库搜索实验（12 名参与者），实验条件覆盖初始角度（正前、侧、后）、物体数量（5/7/12）、视觉干扰物数量（0/2/4），记录搜索时间、累计头转、位移、成功率。

**📈 对比分析**

评价方法：对每个条件下人类与模型的中位值进行相关性和 R² 评估。模型在准确率 (R²≈0.35)、搜索时间 (R²≈0.26)、头转 (R²≈0.58)、步行 (R²≈0.06) 上均与人类表现呈正相关，整体趋势与人类相符，尤其在头转与搜索时间上表现良好。

**⚠️ 局限性**

局限性：① 动作空间离散化导致步行与旋转不够连续；② 视觉可见性被二值化，忽略部分可见情况；③ 缺少对人类靠近偏好的建模，导致步行与人类差异；④ 对前后混淆的错误模型未完全捕捉；⑤ 仅学习单一策略，未覆盖人类策略多样性。

---

## 140. MathlibLemma: Folklore Lemma Generation and Benchmark for Formal Mathematics

**arXiv ID:** 2602.02561 | [PDF](https://arxiv.org/pdf/2602.02561v1)

**作者:** Xinyu Liu `[一作]` (University of Virginia), Shangtong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 712 | [OpenAlex ID](https://openalex.org/A5033834190)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MathlibLemma 多代理框架，自动发现并形式化 Lean 4 Mathlib 中缺失的“民间传说”引理，生成可被合并的正式证明；

**💡 创新点**

创新点在于将 LLM 作为多代理系统（Discovery、Judge、Formalizer、Prover）拆分失败模式，实现主动填补库缺口，并构建 4,028 条已 type‑checked 证明的基准；

**🔧 技术方法**

采用 LLM 生成与修正技术（GPT‑5.1 及其 reasoning 变体、DeepSeek、Qwen、Goedel‑Prover、Kimina 等），结合 Lean kernel 反馈循环与自我修正；

**📊 数据集**

数据集为 109 条 Mathlib 源文件中的种子，产生 4,028 条候选引理，随后产生 1,812 条已验证证明；

**📈 对比分析**

与多种最先进的 LLM 验证器（GPT、GPT‑Reasoning、Goedel、Kimina、DeepSeek、Qwen）比较，单个模型最高成功率 23.98%，集合效应可达 44.99%，远低于人类 78% 的可证明率；

**⚠️ 局限性**

局限包括：Judge 召回率不足（约 46% 错误拒绝）、部分证明缺乏必要假设导致无法完成、生成代码仍需人工风格化，且多代理框架对语义等价性判断不足。

---

## 141. A Single Revision Step Improves Token-Efficient LLM Reasoning

**arXiv ID:** 2602.02828 | [PDF](https://arxiv.org/pdf/2602.02828v1)

**作者:** Yingchuan Zhang `[一作]` (University of Georgia), Ping Ma `[通讯]` (University of Georgia)

**通讯引用:** 7259 | [OpenAlex ID](https://openalex.org/A5100615276)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出一种无训练、仅推理时的单步修订机制PACER，通过将多条推理轨迹压缩为共识包，让每条轨迹在同伴证据条件下自我修订，从而在有限令牌预算内提升推理准确率。

**💡 创新点**

创新点在于引入低带宽的共识包作为集体证据，并通过单轮自我修订让轨迹借助集体逻辑来纠错，兼顾 token 效率与准确率。

**🔧 技术方法**

技术包括多轨迹采样、confidence‑guided early stopping（DeepConf‑Online）、共识包构造、基于共识的自我修订以及 confidence‑weighted voting。

**📊 数据集**

使用了竞赛式数学推理基准：AIME 2024/2025、BRUMO 2025、HMMT 2025 等。

**📈 对比分析**

与 MV@256 与 DeepConf‑Online 对比，PACER 在相同或更低令牌消耗下达到或超过 MV@256 的准确率，尤其在难度更高的数据集上提升 2‑3 分。

**⚠️ 局限性**

局限性包括：仅适用于需要完整答案的推理任务，单轮修订在极端不确定或多分支错误时可能不足；共识包压缩可能导致信息丢失；在非数学推理任务的适用性尚待验证。

---

## 142. Modular Isoperimetric Soft Robotic Truss for Lunar Applications

**arXiv ID:** 2602.02915 | [PDF](https://arxiv.org/pdf/2602.02915v1)

**作者:** Mihai Stanciu `[一作]` (Brigham Young University), Nathan Usevitch `[通讯]` (Brigham Young University)

**通讯引用:** 398 | [OpenAlex ID](https://openalex.org/A5024780312)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

该论文提出了一种可模块化、可重构的等周气动三角桁架，用于月球上的轻量化结构部署。

**💡 创新点**

创新点包括新型球面关节可在单一节点连接三条三角桁架，保持等周性，并实现高度紧凑的存储与展开比。

**🔧 技术方法**

采用气压软体管、主动/被动滚轮单元、球面关节以及逆向运动学控制等软体机器人技术。

**📊 数据集**

未使用传统数据集，所有评估基于实验室组装与动态测试得到的测量数据。

**📈 对比分析**

通过实验演示了蹲升、扭转、倾斜、扫角及步态等功能，存储‑展开体积比达1:18.3，电池可持续约26分钟，步态循环可前进0.46 m，性能表明系统具备可部署、可维护性。

**⚠️ 局限性**

局限性包括扫角受限于±35°，大幅扭转会导致三角桁架塌陷，展开部署仍需人工操作，且软管受压下易产生弯曲力矩导致失效。

---

## 143. Learning to Explore with Parameter-Space Noise: A Deep Dive into Parameter-Space Noise for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2602.02555 | [PDF](https://arxiv.org/pdf/2602.02555v1)

**作者:** Bizhe Bai `[一作]` (Fudan University), Tao Chen `[通讯]` (Fudan University)

**通讯引用:** 43151 | [OpenAlex ID](https://openalex.org/A5100357719)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了PSN-RLVR，一种在RLVR中使用参数空间噪声（PSN）的探索框架，以实现长链式推理的轨迹级一致性探索，提升大模型在高采样预算下的推理能力；

**💡 创新点**

创新点在于：①首次系统研究并应用参数空间噪声于RLVR；②引入截断重要性采样（TIS）解决采样-更新不匹配；③提出轻量化实时自适应噪声调度，结合语义多样性和自我置信度；

**🔧 技术方法**

主要技术包括：参数空间噪声（Gaussian噪声注入），TRPO/GRPO算法，截断重要性采样，语义嵌入与自我置信度度量的自适应调度；

**📊 数据集**

使用的数据集：NuminaMath（包含GSM8K、MATH等子集）进行训练；评估在AIME 2024/2025、AMC 2023、OlympiadBench、Minerva Math等数学推理基准；

**📈 对比分析**

与传统GRPO、Pass@k训练、RLVR-Decomposed以及动作空间噪声（温度采样）等方法比较，PSN-RLVR在大采样预算（k≥128）下显著提升pass@k（最高提升约+4–10%），并在语义与操作多样性上优于基线；

**⚠️ 局限性**

局限性包括：对短序列或不需要全局一致性的任务效果有限；自适应噪声调度仍需更多实验验证；高噪声级别可能导致低k性能下降；

---

## 144. Maximum Likelihood Reinforcement Learning

**arXiv ID:** 2602.02710 | [PDF](https://arxiv.org/pdf/2602.02710v1)

**作者:** Fahim Tajwar `[一作]`, Andrea Zanette `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了最大似然强化学习（MaxRL）框架，用于非可微分二元奖励任务，使得RL目标逼近最大似然；

**💡 创新点**

通过将最大似然的Maclaurin级数截断为可采样的阶数，构造了一个计算索引目标族，并给出一个无偏的on‑policy梯度估计器，使得采样量越大目标越接近ML；

**🔧 技术方法**

使用了Latent生成模型、pass@k、Maclaurin展开、无偏梯度估计、控制变量、对比RLOO、GRPO等policy‑gradient方法；

**📊 数据集**

在图像分类（ImageNet+ResNet‑50）、迷宫导航、GSM8K、数学推理基准（AIME 2025、BeyondAIME、MATH‑500、Minerva）等数据集上进行实验；

**📈 对比分析**

与标准RL（RLOO）和GRPO比较，MaxRL在所有实验中均优于对手：在无穷数据和有限数据场景下计算与数据扩展性更好，防止过拟合，且在推理时可实现高达20×的采样效率提升；

**⚠️ 局限性**

仅适用于二元奖励，无法直接处理连续或多值奖励；未涵盖离线/PPO等off‑policy设置；

---

## 145. A Comparative Simulation Study of the Fairness and Accuracy of Predictive Policing Systems in Baltimore City

**arXiv ID:** 2602.02566 | [PDF](https://arxiv.org/pdf/2602.02566v1)

**作者:** Samin Semsar `[一作]` (University of Maryland), James Foulds `[通讯]` (University of Maryland)

**通讯引用:** 2119 | [OpenAlex ID](https://openalex.org/A5010326834)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在巴尔的摩对预测性警务（PredPol）与热点警务（短期与长期KDE）进行为期300天的仿真比较，评估其公平性（种族与社区层面）与准确性，探讨反馈循环导致的偏差放大。

**💡 创新点**

提出了面向城市局部的仿真框架，统一了多种公平与准确性度量（种族公平差、PCR差、Gini系数、覆盖准确率），揭示了即使更准确的PredPol也可能更快放大偏差；并首次在巴尔的摩数据上系统检验数据驱动与模型驱动偏差的交互。

**🔧 技术方法**

使用了代理模型仿真、Kernel Density Estimation（短期与长期）、PredPol自激点过程模型、Noisy-OR检测概率、以及多维公平与准确性评估指标；并通过20次重复实验获得统计稳健结果。

**📊 数据集**

采用巴尔的摩市开放数据门户的2019-2022年真实犯罪记录，并对每条犯罪随机掷硬币（40%）生成报告数据，进一步通过Noisy-OR模拟警务检测。

**📈 对比分析**

在20个不同设置（警力40/400、报告概率0/0.4、犯罪类型总/加重袭击）下比较三种模型，发现PredPol在大多数情形下准确率最高且整体公平性最好；长期KDE在准确性上优于短期KDE，但种族公平性差；短期KDE在公平性与准确性上表现最差。PredPol虽整体更好，但其偏差放大速率最高。

**⚠️ 局限性**

局限性包括：假设所有犯罪均已记录、报告概率统一、检测仅依赖警力数量、未考虑警务对犯罪率的反馈效应、仿真时间有限，未覆盖不同社区的报告概率差异，以及未对更长周期及更细粒度的模型细节进行评估。

---

## 146. On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning

**arXiv ID:** 2602.02762 | [PDF](https://arxiv.org/pdf/2602.02762v1)

**作者:** Sacha Morin `[一作]` (Université de Montréal), Sébastien Lachapelle `[通讯]` (Samsung)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究半监督模仿学习中逆动力学模型（IDM）的样本效率，并统一了两类基于 IDM 的方法（VM-IDM 与 IDM labeling），给出了理论证明和实证分析。

**💡 创新点**

1) 在无限未标记数据与足够容量的情况下证明 VM-IDM 与 IDM labeling 收敛到同一 IDM 基础策略；2) 通过统计学习理论阐明 IDM 学习样本效率优于行为克隆（BC）的原因：IDM 的真值模型通常具有更低的复杂度和更小的随机性；3) 提出改进版 LAPO+ 算法并在实验中证明其优越性。

**🔧 技术方法**

逆动力学模型（IDM）、视频模型（VM）、统计学习理论（VC 维、Rademacher 复杂度）、信息瓶颈与向量量化、统一视频‑动作预测架构（UVA）、实验评估框架。

**📊 数据集**

ProcGen（16 个类 Atari 环境）、Push‑T、Libero10，以及实验室自行生成的迷宫环境。

**📈 对比分析**

对比 BC、IDM labeling、LAPO、LAPO+、VM‑IDM 等方法；在 ProcGen 上，IDM labeling 与 LAPO+ 在低标签数下显著优于 BC，LAPO+ 在大多数环境中表现最佳；在 Push‑T 与 Libero10 上，IDM 基础策略优于 BC，尤其在 Push‑T 上差距明显。

**⚠️ 局限性**

仅考虑理想的专家数据集，未探讨采样成本；未研究有限未标记数据下两种 IDM 方法的泛化差异；只关注样本效率，未覆盖其他实际挑战。

---

## 147. DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution

**arXiv ID:** 2602.02919 | [PDF](https://arxiv.org/pdf/2602.02919v1)

**作者:** Jiachen Jiang `[一作]` (Ohio State University), Zhihui Zhu `[通讯]` (Ohio State University)

**通讯引用:** 2819 | [OpenAlex ID](https://openalex.org/A5011989964)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了DeltaEvolve——一种基于动量的LLM驱动进化框架，用delta摘要取代完整代码历史以指导程序演化。

**💡 创新点**

创新点在于：①将进化过程形式化为期望‑最大化（EM）框架，聚焦M‑step的上下文更新；②引入DeltaEvolve记忆，将父子程序的语义变更（delta）存为可传递的演化信号；③构建三级多层数据库与逐步披露采样器，实现令牌高效利用。

**🔧 技术方法**

技术手段包括LLM代码生成（如GPT‑5‑mini、Gemini）、EM框架、delta摘要与计划细节生成、进化式采样（top‑k、diverse）以及基于MAP‑Elites的多样性抽样。

**📊 数据集**

实验数据集覆盖五大科学任务：黑盒优化（Rosenbrock、Rastrigin等）、六边形打包、符号回归、PDE求解器、2D卷积加速。

**📈 对比分析**

与并行采样、贪心精炼、AlphaEvolve等基线对比，DeltaEvolve在所有任务上均取得更高的最佳得分，并平均降低约36.8%令牌消耗。

**⚠️ 局限性**

局限性：仍受限于LLM上下文窗口大小，部分任务对全代码仍有依赖；对极端高维或需要大量细节实现的程序，delta摘要可能不足以捕捉全部必要信息。

---

## 148. Distilling LLM Reasoning into Graph of Concept Predictors

**arXiv ID:** 2602.03006 | [PDF](https://arxiv.org/pdf/2602.03006v1)

**作者:** Ziyang Yu `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6667 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Graph of Concept Predictors (GCP) 框架，用于在 LLM 主动蒸馏中保留教师的推理过程，以图结构形式实现概念级监督，提升样本效率与训练稳定性。

**💡 创新点**

创新点在于：① 将 LLM 的推理 DAG 外化为学生可学习的图结构；② 引入图感知采样（结构加权不确定性、拓扑梯度多样性、图感知代表性）；③ 通过逆因果子模块重训练，只更新最影响最终损失的概念预测器，从而实现稀疏监督、可解释性与计算节省。

**🔧 技术方法**

技术包括：概念预测器 DAG 架构、结构加权不确定性评估、拓扑梯度多样性度量、图感知代表性采样、逆因果子模块重训练、LLM 生成父子概念对等方法。

**📊 数据集**

使用八个 NLP 分类基准：AG News、Amazon Reviews、IMDB、Yelp、MNLI、GoEmotions、SemEval Stance、MIMIC‑III（临床笔记）等。

**📈 对比分析**

与随机、Least Confidence、Entropy、Disagreement、CoreSet、BADGE、ALPS、CAL 等主动学习基线比较，在 20% 标注预算下 GCP 在所有数据集上均超过对手；在更大预算下仍保持领先；相较于直接使用 LLM 推理，GCP 在计算成本上降低 10⁵–10⁶ 倍。

**⚠️ 局限性**

局限：① 可能将教师 LLM 的偏差或错误传递给子模块；② 需要先行设计概念图和概念定义，人工成本高；③ 对极低标注预算下的表现仍受限；④ 对非常复杂的多路径推理的可扩展性尚未充分验证。

---

## 149. Composition for Pufferfish Privacy

**arXiv ID:** 2602.02718 | [PDF](https://arxiv.org/pdf/2602.02718v1)

**作者:** Jiamu Bai `[一作]` (Penn State University), Kiwan Maeng `[通讯]` (Penn State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在存在数据相关性时如何为Pufferfish隐私定义提供线性可组合性，并构造可组合的Pufferfish机制。

**💡 创新点**

提出必要充分条件（NfC约束）证明必须加入类似差分隐私的不等式才能实现线性组合；引入“influence curve”把任意条目差分隐私机制转换为可组合的Pufferfish机制；通过该框架在Markov链上显著提升查询精度。

**🔧 技术方法**

使用差分隐私理论、凸优化、线性规划、对偶理论、Markov网络、指数机制、拉普拉斯机制等技术；在分析中运用influence curve、最大影响等概念。

**📊 数据集**

实验采用Foursquare签到数据集（地点标签）和Capture24日常活动数据集（11种活动类别），并以时间序列Markov链作为先验。

**📈 对比分析**

与传统MQM（基于拉普拉斯的Pufferfish）、Group DP和直接指数机制做对比；评估指标包括Acc@k、HitRate@K、NDCG@K、ℓ1计数误差；实验显示本文方法在所有指标上均优于或不逊于基线，特别是在Top‑k查询上的性能提升显著。

**⚠️ 局限性**

局限在于需要已知或可估计的先验分布、influence curve的计算复杂度较高，且框架主要针对线性组合，非线性/非直观查询的适用性待进一步验证。

---

## 150. What Do Contribution Guidelines Say About Software Testing?

**arXiv ID:** 2602.02966 | [PDF](https://arxiv.org/pdf/2602.02966v1)

**作者:** Bruna Falcucci `[一作]` (Federal University of Minas Gerais), Andre Hora `[通讯]` (Federal University of Minas Gerais)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文对200个Python与JavaScript开源项目的贡献指南进行实证研究，调查了测试相关说明的出现频率与内容分布；

**💡 创新点**

创新点在于首次系统评估了贡献者测试说明的普及程度及其覆盖面，揭示了“运行测试”占主导、“编写测试”“集成/端到端测试”“覆盖率”“模拟”等内容的缺失；

**🔧 技术方法**

主要采用人工审查与双人核对的方法来识别和分类贡献指南中的测试说明；

**📊 数据集**

使用的数据集为GitHub上按星标排序的前100名Python项目与前100名JavaScript项目，共200个项目；

**📈 对比分析**

通过统计不同测试主题出现的比例进行比较，发现测试说明普及率为78%，其中运行测试占83.5%，写测试仅37%，单元测试71%，集成/端到端分别为20.5%和15.5%，覆盖率和模拟率分别为25.5%和9.5%；

**⚠️ 局限性**

局限性包括检索过程手工完成，存在漏检风险；未对说明质量或实际效果进行评估，缺乏自动化工具支持和更深入的内容分析。

---

## 151. IMU-1: Sample-Efficient Pre-training of Small Language Models

**arXiv ID:** 2602.02522 | [PDF](https://arxiv.org/pdf/2602.02522v1)

**作者:** George Grigorev `[一作]` `[通讯]` (Independent Researcher), George Grigorev (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可复现的训练配方，将多项架构与优化技术整合，在430M参数模型上仅用72B训练token即可逼近SmolLM‑360M（600B token）和SmolLM2‑360M（4T token）的基准性能。

**💡 创新点**

创新点在于统一将QK‑norm注意力、Per‑Head Gating、Normalized Value Residual、LayerNorm Scaling等架构改进与NorMuon优化器、Cautious Weight Decay、Warmup‑Stable‑Decay学习率调度、后置EMA等优化手段结合，并通过详细消融验证其协同效益。

**🔧 技术方法**

采用的技术包括QK‑norm注意力、Per‑Head Gating、Normalized Value Residual、LayerNorm Scaling、NorMuon优化器、Cautious Weight Decay、Z‑Loss、Warmup‑Stable‑Decay学习率调度、后置EMA以及自定义Triton核实现等。

**📊 数据集**

使用多阶段混合数据，涵盖网页文本（DCLM‑edu、Cosmopedia）、代码（Stack‑edu）、数学（FineMath）、PDF、维基（FineWiki）以及后期的指令化和合成推理数据，总计约72B token。

**📈 对比分析**

在HellaSwag、ARC‑E/CH、PIQA、Lambada等基准任务上与SmolLM系列比较，430M模型在72B token上平均得分0.560，逼近SmolLM‑360M（0.560）且仅用8倍少的token；在相同参数规模下接近SmolLM2‑360M（0.574 vs 0.586），表现出显著的样本效率提升。

**⚠️ 局限性**

实验仅验证了亚十亿参数规模和特定数据混合，未确认在更大规模或不同数据环境下的通用性；部分技术（如QK‑norm）依赖自定义核，增加了部署复杂度。

---

## 152. InfMem: Learning System-2 Memory Control for Long-Context Agent

**arXiv ID:** 2602.02704 | [PDF](https://arxiv.org/pdf/2602.02704v1)

**作者:** Xinyu Wang `[一作]` (McGill University), Yufei Cui `[通讯]` (Noah's Ark Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 InfMem，一种基于 System‑2 思维的有界记忆长文本问答代理，采用 PreThink–Retrieve–Write 控制循环和早停机制。

**💡 创新点**

创新点在于：①显式证据驱动的控制策略；②针对性全文检索与联合压缩；③基于 SFT→RL 的训练流程，将检索、写入与停止决策与最终任务奖励对齐；④自适应早停提升效率。

**🔧 技术方法**

核心技术包括：系统‑2 控制框架、检索增强生成 (RAG)、固定大小覆盖记忆、教师蒸馏的监督微调、GRPO 强化学习、早停奖励塑造和多轮对话训练。

**📊 数据集**

实验数据集：从 SQuAD、HotpotQA、2WikiMultiHopQA、MuSiQue 构造的合成 32k‑1M 级长文本 QA 数据；评估在 1M‑token 任务和 LongBench 上。

**📈 对比分析**

与 YaRN、RAG、MemAgent 及高容量模型对比，InfMem 在 1M‑token 基准上平均提升 10%+ 准确率，并在 Qwen 系列模型上实现 3.9× 的推理速度提升；在 LongBench 上也保持优势。

**⚠️ 局限性**

局限性：仅限文档内检索，依赖检索单元质量；训练过程受限于协议合法性，可能在极度稀疏或含糊证据场景下性能下降；对其他任务的通用性尚未验证。

---

## 153. Beyond Content: Behavioral Policies Reveal Actors in Information Operations

**arXiv ID:** 2602.02838 | [PDF](https://arxiv.org/pdf/2602.02838v1)

**作者:** Philipp J. Schneider `[一作]` (Ecole Polytechnique Federale de Lausanne), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**通讯引用:** 1174 | [OpenAlex ID](https://openalex.org/A5069685493)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了如何通过用户行为序列的决策策略识别Reddit上的信息操作账户，提出基于行为策略的检测框架。

**💡 创新点**

首次将用户行为建模为马尔可夫决策过程并使用GAIL、IRL等方法学习个体策略，证明行为策略比文本内容更能早期识别攻击者。

**🔧 技术方法**

采用马尔可夫决策过程、生成对抗式模仿学习、最大熵逆强化学习以及XGBoost/随机森林等监督分类器，同时使用ModernBERT生成文本嵌入。

**📊 数据集**

使用2015-2018年Reddit公开活动日志，包含99个被识别为俄罗斯Internet Research Agency的账号和约1.2万名普通用户，约3800万条交互记录。

**📈 对比分析**

将行为策略表示与文本嵌入在同一训练集上对比，GAIL策略在宏F1上达94.9%，早期仅3步即可91.4%，在噪声或账号劫持下仍保持高稳健性。

**⚠️ 局限性**

标签来源于平台透明度报告可能偏向已知手段，部分恶意账号表现与正常用户相似导致误检，且对不同平台的迁移仍需自定义状态空间。

---

## 154. Fisheye Stereo Vision: Depth and Range Error

**arXiv ID:** 2602.02973 | [PDF](https://arxiv.org/pdf/2602.02973v1)

**作者:** Leaf Jiang `[一作]` (NODAR Inc.), Piotr Swierczynski `[通讯]` (NODAR Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文推导了鱼眼立体视觉系统的深度和距离误差表达式，考虑了大角度时的精度下降。

**💡 创新点**

创新点是针对鱼眼相机在大角度下的径向压缩和有效基线短化，给出了比传统针孔模型更准确的误差公式，并揭示了误差随角度的1+tan^2θ衰减规律。

**🔧 技术方法**

采用立体视觉几何、鱼眼等距投影模型、误差传播分析以及实时自标定算法实现。

**📊 数据集**

使用了一个8MP 4K 180°鱼眼相机系统（3840×2160像素，2.1µm像素间距）和1m基线的实验配置。

**📈 对比分析**

通过对比针孔与鱼眼模型在10m距离、不同入射角的误差曲线，结果显示鱼眼在±30°内误差低于4cm，而针孔受基线缩短影响；但鱼眼在边缘角度误差增长更快，整体性能在中心优于针孔。

**⚠️ 局限性**

局限性包括鱼眼在极端角度仍存在较大误差，基线短化导致的误差放大，模型假设对实际光学畸变不完全适用，且缺乏在真实场景中的实测验证。

---

## 155. Evaluation of Large Language Models' educational feedback in Higher Education: potential, limitations and implications for educational practice

**arXiv ID:** 2602.02519 | [PDF](https://arxiv.org/pdf/2602.02519v1)

**作者:** Daniele Agostini `[一作]`, Federica Picasso `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了七种大型语言模型在高等教育中生成学生作业评估反馈的能力，并通过Hughes‑Smith‑Creese框架对反馈质量进行编码。

**💡 创新点**

创新点在于首次将多模型并行评估与结构化评分量表相结合，系统量化不同LLM在赞扬、批评、建议和澄清等四大反馈要素的覆盖度，并提供对模型表现的细粒度比较。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑4o、Gemini 1.5 Pro、Claude 3.5 Sonnet、Mistral Large、Open Mixtral 8x22B、Llama 3.1 70B、Qwen2 72B）以及基于API的Big‑AGI平台实现多模型并行推理；评估采用Hughes‑Smith‑Creese（2015）框架进行编码。

**📊 数据集**

数据集为35份学生团队设计的教学干预方案（来自Trento大学的教师资格培训课程），每份方案均附有结构化评估量表，并由七种LLM生成对应的反馈文本。

**📈 对比分析**

比较方法为对每个模型在每个反馈要素上出现的次数进行计数，计算覆盖率并得出总得分；性能上，Mistral Large以约69%总体得分位居前列，Claude 3.5 Sonnet和Gemini也表现良好，而GPT‑4o与Open Mixtral在错误纠正和澄清请求方面相对薄弱。

**⚠️ 局限性**

局限性包括样本规模有限、仅覆盖特定课程的项目、模型表现高度依赖提示与量表的设计、错误检测与澄清请求的覆盖率不一，以及在高风险评估场景中的可靠性与伦理性尚待进一步验证。

---

## 156. A Reproducible Framework for Bias-Resistant Machine Learning on Small-Sample Neuroimaging Data

**arXiv ID:** 2602.02920 | [PDF](https://arxiv.org/pdf/2602.02920v1)

**作者:** Jagan Mohan Reddy Dwarampudi `[一作]`, Tania Banerjee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个可复现、无偏差的机器学习框架，用于小样本神经影像学的结构MRI预测深脑刺激（DBS）认知结果。

**💡 创新点**

结合领域知识的特征工程、严格的嵌套交叉验证、概率校准与阈值优化，实现了无偏估计且可解释的模型。

**🔧 技术方法**

使用随机森林/额外树等树模型、Platt校准、嵌套CV、阈值搜索、特征合成等技术。

**📊 数据集**

利用佛罗里达大学运动障碍中心332例患者的结构T1 MRI及UF认知风险评分。

**📈 对比分析**

通过嵌套CV与阈值校准将模型与传统单层CV/训练-测试拆分进行比较，获得约0.66的平衡准确率（相比0.57-0.59的传统方法），AUC约0.72。

**⚠️ 局限性**

样本量有限、单中心数据、标签标注可能偏差，限制了结果的普适性。

---

## 157. Rate-Distortion Analysis of Optically Passive Vision Compression

**arXiv ID:** 2602.02768 | [PDF](https://arxiv.org/pdf/2602.02768v1)

**作者:** Ronald Ogden `[一作]` (University of Texas at Austin), Takashi Tanaka `[通讯]` (Purdue University)

**通讯引用:** 1395 | [OpenAlex ID](https://openalex.org/A5082230661)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种将光学余弦变换与事件相机相结合的压缩方案（OPVC），并通过仿真评估其速率-失真性能。

**💡 创新点**

创新点在于：①在传感层通过光学余弦变换实现硬件级压缩，消除传统视频编解码器的计算负担；②利用事件相机观测光学余弦变换的结果，实现对高频信息的被动过滤。

**🔧 技术方法**

使用了事件相机仿真、光学余弦变换（等价于二维 DCT）、MS‑SSIM 失真度量、率‑失真曲线分析等技术。

**📊 数据集**

使用了 Ultra Video Group (UVG) 数据集，包含 16 条 3840×2160 分辨率的未压缩视频。

**📈 对比分析**

通过在不同分辨率、不同事件阈值下生成 OPVC 与独立事件相机（SAEC）的采样率‑失真曲线进行对比。结果显示：在任何失真约束下，OPVC 的采样率均低于 SAEC，且两者之间的性能差距随分辨率升高而扩大。

**⚠️ 局限性**

局限性：①仅在理想无噪声、无时域插值的假设下进行仿真；②未考虑事件相机的噪声、热像素、漏事件等实际影响；③未评估周期初始化帧所带来的比特率开销；④缺乏真实光学实现的实验验证。

---

## 158. Trailer Reimagined: An Innovative, Llm-DRiven, Expressive Automated Movie Summary framework (TRAILDREAMS)

**arXiv ID:** 2602.02630 | [PDF](https://arxiv.org/pdf/2602.02630v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 159. Dynamic Mix Precision Routing for Efficient Multi-step LLM Interaction

**arXiv ID:** 2602.02711 | [PDF](https://arxiv.org/pdf/2602.02711v1)

**作者:** Yuanzhe Li `[一作]` (University of Arizona), Huanrui Yang `[通讯]` (University of Arizona)

**通讯引用:** 1493 | [OpenAlex ID](https://openalex.org/A5076154259)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于步骤级动态混合精度路由的框架，通过在多步代理任务中按需调用高精度或低精度LLM，显著降低推理成本并保持高成功率。

**💡 创新点**

创新点在于：1）识别并利用决策步骤对量化敏感度的极端差异；2）采用基于KL散度的监督来标注关键步骤；3）结合Group‑Relative Policy Optimization对路由策略进行成本感知的强化学习优化；4) 设计轻量级Transformer路由器，仅占用极少计算资源。

**🔧 技术方法**

主要技术包括：两阶段训练流程（KL‑ST + GRPO）、GPTQ 3/4‑bit 量化、轻量级 2‑层Transformer 路由器、步骤级状态编码与位置编码、群组相对优势估计、KL 正则化等。

**📊 数据集**

使用了 ALFWorld 这一基于文本的嵌入式环境进行实验，涵盖多种任务与多种规模的 Qwen3 与 DeepSeek‑R1‑Distill 模型。

**📈 对比分析**

与全精度 BF16、纯量化（0% 高精度）以及随机路由（匹配相同高精度调用比例）进行对比。实验结果显示，路由器在低于30%高精度调用下即可逼近全精度成功率；在 GHC 指标上远超随机与纯量化方法，证明每一次高精度调用都更有效。

**⚠️ 局限性**

局限性：1）路由效果受限于高精度模型的能力，模型弱时提升有限；2）需要大量高精度轨迹来构造 KL 监督，训练成本相对较高；3）对 KL 阈值与分布假设敏感，可能影响跨任务迁移；4) 目前仅在文本嵌入式环境验证，尚需在更大规模或多模态任务上进一步验证。

---

## 160. Test-Time Detoxification without Training or Learning Anything

**arXiv ID:** 2602.02498 | [PDF](https://arxiv.org/pdf/2602.02498v1)

**作者:** Baturay Saglam `[一作]` (Yale University), Dionysis Kalogerias `[通讯]` (Yale University)

**通讯引用:** 355 | [OpenAlex ID](https://openalex.org/A5091493591)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种仅在推理时通过零阶梯度估计对输入词嵌入进行微调，从而降低大语言模型生成文本的毒性。

**💡 创新点**

首次将零阶差分估计直接应用于嵌入空间，避免了模型重训练、梯度访问或辅助模块，且在黑盒环境下实现近最优的毒性减弱。

**🔧 技术方法**

使用高斯随机扰动的零阶梯度估计、梯度归一化、余弦相似度约束和早停机制；整体为一次性测试时间优化。

**📊 数据集**

在RealToxicityPrompts、AttaQ、BOLD等公开毒性测试集上评估，使用 GPT‑2 Large、Gemma‑2‑2B、Qwen‑3‑4B、Llama‑3.1‑8B 等模型。

**📈 对比分析**

与多种无重训练的基线（DeStein、Toxification Reversal、GeDi、RAD、SASA 等）在毒性最大值、平均毒性、毒性率与困惑度上进行比较，取得了更优的毒性‑流畅度折衷。

**⚠️ 局限性**

对温度敏感，需低温下使用；每次优化需额外前向推理，虽然开销小但仍增加推理成本；在高温或大规模部署时稳定性与效率待验证。

---

## 161. Incident-Guided Spatiotemporal Traffic Forecasting

**arXiv ID:** 2602.02528 | [PDF](https://arxiv.org/pdf/2602.02528v1)

**作者:** Lixiang Fan `[一作]` (Beihang University), Junchen Ye `[通讯]` (Beihang University)

**通讯引用:** 3841 | [OpenAlex ID](https://openalex.org/A5081275566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于图神经网络的事故引导时空预测框架IGSTGNN，能够显式建模交通事故对网络的初始空间影响和随时间衰减的动态影响。

**💡 创新点**

创新点在于：①设计了Incident‑Context Spatial Fusion（ICSF）模块，利用注意力机制结合事故属性、传感器特征和道路网络拓扑，捕捉非均匀的空间传播；②引入Temporal Incident Impact Decay（TIID）模块，用高斯衰减函数模拟事故影响随时间的逐步消失；③构建并公开了大规模事故对齐交通流数据集，为后续研究提供基准；④证明ICSF和TIID可作为插件集成到多种主流STGNN模型中提升性能。

**🔧 技术方法**

技术上使用图卷积网络、注意力机制、多图卷积、RNN+自注意力、时间衰减正则化、残差连接、LayerNorm等深度学习组件，并结合了静态/自适应/动态邻接矩阵。

**📊 数据集**

使用XTraffic基准数据集中的三个子集（Alameda、Contra Costa、Orange），每个子集包含主线传感器的5 分钟时段交通流与事故记录，训练集/验证集/测试集比例为70%/15%/15%。

**📈 对比分析**

与HL、LSTM、DCRNN、AGCRN、STGCN、GWNET、ASTGCN、STTN、DSTAGNN、DGCRN、D²STGNN、BiST等19个基线模型比较，IGSTGNN在三个数据集的12步预测（60 min）平均MAE、RMSE和MAPE均显著优于所有基线，提升幅度约5–15%，并在长时预测（12步）表现最为突出。

**⚠️ 局限性**

局限性包括：①主要关注主线交通事故，未覆盖道路维护、恶劣天气等其他外部扰动；②依赖精确的事故定位与属性信息，对缺失或噪声高的事故数据鲁棒性有限；③模型结构相对复杂，训练成本和推理时间高；④实验仅在加州交通数据上验证，泛化到其他城市或网络拓扑需进一步验证。

---

## 162. Large Language Models Can Take False First Steps at Inference-time Planning

**arXiv ID:** 2602.02991 | [PDF](https://arxiv.org/pdf/2602.02991v1)

**作者:** Haijiang Yan `[一作]` (University of Warwick), Adam Sanborn `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究通过贝叶斯框架阐释大型语言模型在推理时的规划短视行为，并在随机高度生成与高斯采样两种实验中验证规划转移、收敛以及偏差-校正动态；

**💡 创新点**

创新点在于将自生成上下文视为动态先验，揭示其对规划概率分布的压制效应，进而解释短期规划现象并预言规划收敛与偏差校正两个可检验的行为特征；

**🔧 技术方法**

采用贝叶斯推理与序列级规划模型，并用LASSO回归量化嵌入层对未来token的可预测性，以及统计检验评估生成序列的偏差；

**📊 数据集**

实验使用两款开源LLM（Llama‑3.1‑8B‑Instruct 与 Qwen‑2.5‑7B‑Instruct）在随机身高估计（60个整数序列）与从高斯分布采样（64+64整数序列）任务上生成数据；

**📈 对比分析**

与基线相比，实验显示嵌入层对后续token的解释方差随自生成上下文累积而显著提升（R²≈0.8–1.0），且在高斯采样中自生成上下文可显著降低首位偏差，说明规划能力在推理后期得到恢复；

**⚠️ 局限性**

主要局限在于实验仅涵盖结构化数值生成，尚未验证该规划机制是否同样适用于自由文本或更开放式生成任务。

---

## 163. Learnable Koopman-Enhanced Transformer-Based Time Series Forecasting with Spectral Control

**arXiv ID:** 2602.02592 | [PDF](https://arxiv.org/pdf/2602.02592v1)

**作者:** Ali Forootani `[一作]` (Max Planck Institute of Geoanthropology), Raffaele Iervolino `[通讯]` (University of Naples)

**通讯引用:** 732 | [OpenAlex ID](https://openalex.org/A5051334095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Learnable Koopman-Enhanced Transformer 模型，在 Transformer 编码器中嵌入可学习的 Koopman 运算符，实现可解释的线性潜在动力学。

**💡 创新点**

创新点在于四种可学习 Koopman 变体（标量门控、逐模态、神经映射、低秩）和 ODO 因子化实现显式谱控制、稳定性与可逆性。

**🔧 技术方法**

采用 ODO 因子化、谱抑制、Lyapunov 正则化、Transformer 编码器、线性解码器以及多种 Transformer 后端（Informer、LogTrans、Autoformer）。

**📊 数据集**

使用 CMIP6、ERA5 气候、加密货币交易与能源生成等五大真实数据集，涵盖周期性、随机与混沌过程。

**📈 对比分析**

通过与 LSTM、DLinear、SSM 等基线以及 Transformer 变体在不同窗口长度与预测步长下对比，学习 Koopman 模型在大多数任务上实现更低均方误差、误差分布更紧凑，并展示更好的谱收敛与可逆性。

**⚠️ 局限性**

局限在于离散时间假设、对噪声鲁棒性有限，以及对非平稳或带控制输入的情况尚未充分验证。

---

## 164. ContextEvolve: Multi-Agent Context Compression for Systems Code Optimization

**arXiv ID:** 2602.02597 | [PDF](https://arxiv.org/pdf/2602.02597v1)

**作者:** Hongyuan Su `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**通讯引用:** 31726 | [OpenAlex ID](https://openalex.org/A5029768249)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ContextEvolve 多智能体框架，在仅 API 访问的大语言模型环境下，通过上下文压缩实现高效的系统代码演化搜索。

**💡 创新点**

创新点：①将优化上下文拆分为语义状态、优化方向和经验分布三维，并设计专门的 Summarizer、Navigator、Sampler 代理；②将该多代理流程映射为 RL 的状态表征、策略梯度和经验回放，实现在无参数更新下的 RL 等价搜索；③通过高信息密度的自然语言压缩显著降低 token 消耗。

**🔧 技术方法**

技术手段：大语言模型 Qwen3 的代码生成与代理推理；自然语言摘要压缩；轨迹分析提取梯度；优先经验采样；演化策略与 RL 等价框架；API 交互与 token 计量。

**📊 数据集**

使用 ADRS benchmark，其中包含五个任务：Transaction Scheduling、SQL Optimization、Load Balancing、Sparse Attention Kernel 和 Model Placement。

**📈 对比分析**

与 Heuristics、Human‑SOTA、LLM One‑shot、GEPA、OpenEvolve 等基线对比，ContextEvolve 在所有任务平均提升 6.5% 的综合分数，token 使用量下降约 17.3%，在 LB 任务上平衡度提升 36% 并保持最快速度。

**⚠️ 局限性**

局限性：①尚未验证在大规模、多模块代码库中的可扩展性；②对 LLM 的高方差导致搜索结果不稳定；③实验仅基于 Qwen3，缺乏跨模型泛化；④缺乏针对算法多样性与跨域推广的机制。

---

## 165. When Noise Lowers The Loss: Rethinking Likelihood-Based Evaluation in Music Large Language Models

**arXiv ID:** 2602.02738 | [PDF](https://arxiv.org/pdf/2602.02738v1)

**作者:** Xiaosha Li `[一作]`, Ziyu Wang `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过向音乐序列注入噪声和打乱顺序等扰动，系统评估了音乐大语言模型（LLM）的损失函数在质量评估中的可靠性；

**💡 创新点**

发现了“Context Amnesia Effect”，即损失曲线在扰动开始处出现尖峰，随后快速下降并保持低值，最终恢复到基线，说明绝对损失值无法反映长期结构破坏；

**🔧 技术方法**

使用了基于音频波形的Transformer音乐LLM（MusicGen、YuE）以及自定义的噪声注入与顺序打乱实验，分析了token级损失差异；

**📊 数据集**

实验数据集包括MusicGen训练集（20首），MusicGen生成样本（140首），以及ASAP经典曲目（78首），同时验证了不同模型规模的结果；

**📈 对比分析**

通过计算扰动长度与平均损失差异的Pearson与Spearman相关系数，发现二者呈显著负相关（r≈-0.85至-0.90），并在回归分析中得到显著负斜率，表明更长扰动导致损失下降；

**⚠️ 局限性**

局限性在于实验仅针对噪声与顺序打乱两种扰动，未涵盖所有音乐结构失真类型；此外，绝对损失值在生成音乐中的评估仍不可靠，需要进一步探索更鲁棒的评估指标。

---

## 166. A Random Matrix Theory Perspective on the Consistency of Diffusion Models

**arXiv ID:** 2602.02908 | [PDF](https://arxiv.org/pdf/2602.02908v1)

**作者:** Binxu Wang `[一作]` (Harvard University), Cengiz Pehlevan `[通讯]` (Harvard University)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5023195984)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究扩散模型在不同数据拆分下的输出一致性，并用随机矩阵理论（RMT）分析有限样本对线性去噪器与采样映射的期望与方差。

**💡 创新点**

创新点在于将有限样本效应视为噪声重新归一化σ²→κ(σ²)，揭示一致性来源于共享高斯统计，并将方差分解为各向异性、空间不均匀性和样本规模缩放；同时扩展了DE到分数矩阵幂，实现对采样轨迹的解析预测。

**🔧 技术方法**

采用随机矩阵理论（deterministic equivalence）、线性去噪器分析、分数矩阵幂积分、以及深度网络实验验证等技术。

**📊 数据集**

使用FFHQ（32/64）、AFHQ32、LSUN教堂/卧室、CIFAR‑10/100等公开图像数据集。

**📈 对比分析**

通过与线性高斯预测、UNet与DiT网络在不同拆分/样本规模下的相似度、最近邻距离、MSE、方差等指标进行比较；结果表明有限样本导致过度收缩，随着样本增大逐步趋近线性预测；深度网络的波动与理论相符但幅度更大。

**⚠️ 局限性**

仅适用于线性近似，无法捕捉模型容量和非线性效应；对架构特定的诱导偏好缺乏解释；对迁移/泛化阈值的预测不够精确。

---

## 167. Moving On, Even When You're Broken: Fail-Active Trajectory Generation via Diffusion Policies Conditioned on Embodiment and Task

**arXiv ID:** 2602.02895 | [PDF](https://arxiv.org/pdf/2602.02895v1)

**作者:** Gilberto G. Briscoe-Martinez `[一作]` (University of Colorado Boulder), Alessandro Roncone `[通讯]` (University of Colorado Boulder)

**通讯引用:** 694 | [OpenAlex ID](https://openalex.org/A5020277024)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在机器人关节失效的情形下，提出了 DEFT，一个基于扩散模型的轨迹生成框架，能够根据当前的本体约束和任务限制自动生成可行轨迹。

**💡 创新点**

创新点在于将失效视为新的本体，通过结构化的本体向量与任务编码同时条件化扩散模型，实现了在任意关节失效、多个操控原语以及多关节失效下的无模型泛化和连续适应。

**🔧 技术方法**

采用扩散模型（Diffusion Policy）与 MLP+FiLM 条件化，配合起始-终点填充和输出裁剪技术，以及对关节范围/速度的硬约束。

**📊 数据集**

使用模拟数据集生成的约 4.7k 条失效条件下的关节轨迹，覆盖两类任务约束（自由和约束），以及真实世界 7-DoF Panda arm 的两项多步骤任务（抽屉和擦板）作为验证集。

**📈 对比分析**

与经典 RRT+IK 规划以及无条件扩散模型比较，DEFT 在模拟中约 74.5% 的约束满足率（相较于基线的 36.9%）提升 37.6%，在 OOD 失效下仍保持 73–78% 的成功率，并在实测任务中实现 100% 完成率，而基线仅 0–35%。

**⚠️ 局限性**

局限性包括对实时失效检测与诊断的缺失、对关节极限的手动设定、对非关节失效（如传感器漂移、电压下降）支持有限，以及对更复杂机器人平台迁移的挑战。

---

## 168. Minimal Computational Preconditions for Subjective Perspective in Artificial Agents

**arXiv ID:** 2602.02902 | [PDF](https://arxiv.org/pdf/2602.02902v1)

**作者:** Hongju Pae `[一作]` `[通讯]` (Active Inference Institute), Hongju Pae (Active Inference Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种人工智能代理体系结构，在其中引入了一个慢速的全局潜在变量（g）来实现“主观视角”，并在无奖励的网格世界环境（三个噪声区）中进行训练，随后通过在不同噪声区间切换来观察全局潜在变量与政策熵的动态差异，最终验证了全局潜在变量的滞后性（hysteresis）与方向依赖性。

**💡 创新点**

创新点在于：①将主观视角视为一个慢速、全局约束的潜在变量，与传统的即时策略区分；②通过停梯度（stop‑gradient）与平滑正则化实现潜在变量与策略的分离；③采用无奖励、预测误差最小化的训练目标，避免视角被工具化；④利用方向依赖的滞后轨迹（switch‑aligned hysteresis）作为可测量的主观视角指标。

**🔧 技术方法**

技术细节包括：使用GRU + 层归一化更新慢速潜在变量；使用前馈网络将感知潜在变量（z）与全局潜在变量（g）映射到策略；在策略梯度中引入REINFORCE式的成本（预测误差）并加入熵正则化；通过停梯度将策略梯度与潜在变量更新分离；利用均方误差最小化预测模型的下一个观察值；对全局潜在变量施加均方差平滑正则化。

**📊 数据集**

使用自定义的三区噪声网格世界数据集（3×5×9格，每区噪声参数不同），不依赖公开大规模数据集；训练和评估均在该模拟环境中完成。

**📈 对比分析**

比较方法：在测试阶段交替切换噪声区（A↔B），记录全局潜在变量的投影得分（g-score）与策略熵（z-entropy），并采用分位数统计比较方向依赖滞后轨迹；性能评估：在训练后代理基本停留在低噪声区，区占比接近1；g-score 显示出明显的方向性滞后，策略熵则无明显方向性。

**⚠️ 局限性**

局限性：①仅使用外部感知输入，缺乏身体/内感知的嵌入；②实验环境过于简化，难以直接推广到复杂真实世界；③缺乏因果干预（如 do(g) 操作）以验证g的直接影响；④无奖励信号可能限制模型对实际任务的适用性；⑤对不同超参数或更大规模环境的稳健性尚未验证。

---

## 169. A Parametrized Complexity View on Robust Scheduling with Budgeted Uncertainty

**arXiv ID:** 2602.02748 | [PDF](https://arxiv.org/pdf/2602.02748v1)

**作者:** Noam Goldberg `[一作]`, Dvir Shabtay `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在预算化不确定性模型下的单机排程问题，目标是最小化在所有可能的处理时间场景中迟到的作业数，并从参数化复杂度的角度对该问题进行分类与求解。

**💡 创新点**

创新点主要包括：①证明该问题在鲁棒参数 Γ 上是 W[1]-hard；②给出针对 Γ 的 XP 算法；③展示仅有两种不同的交付日期时问题仍为 NP‑hard；④在只有一种交付日期时可多项式求解，并在交付日期数为常数时提供伪多项式算法；⑤在作业不确定处理时间数量为参数时实现 FPT 求解。

**🔧 技术方法**

采用的技术包括：参数化复杂度理论（W[1]-hardness 归约、XP/DP 设计）、强直交/EDD 顺序理论、鲁棒二项背包的对偶化与 μ 搜索、以及对莫尔算法的扩展以处理不确定处理时间。

**📊 数据集**

论文未使用任何真实数据集，而是通过构造性的理论构造与归约证明结果。

**📈 对比分析**

评价方法主要是理论复杂度分析；对比结果显示：在 Γ 或交付日期数为常数时可得到多项式/伪多项式解；在 Γ 大于常数时仍需 XP 级别；在不确定作业数为参数时可达到 FPT。

**⚠️ 局限性**

局限性包括：①对 Variant 2（作业在不同场景中可能既早又迟）的参数化可行性仍未解决；②对鲁棒参数 Γ 的更精细化复杂度（是否存在 FPT 或 para‑NP‑hard）未给出完整答案；③仅考虑单机模型，未扩展至多机或流水线排程；④实际性能评估缺乏实验验证。

---

## 170. RAP: KV-Cache Compression via RoPE-Aligned Pruning

**arXiv ID:** 2602.02599 | [PDF](https://arxiv.org/pdf/2602.02599v1)

**作者:** Jihao Xin `[一作]` (King Abdullah University of Science and Technology), Marco Canini `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 4595 | [OpenAlex ID](https://openalex.org/A5042255975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对RoPE编码的LLM KV-Cache压缩，提出RoPE‑Aligned Pruning (RAP) 方法，实现KV‑Cache、参数和FLOPs同步压缩。

**💡 创新点**

创新点是将RoPE对齐的列对结构化剪枝，引入RoPE‑可交换性约束，消除重构开销，实现压缩与准确性兼顾。

**🔧 技术方法**

使用结构化剪枝、Fisher信息评估、动态预算分配、LoRA微调、Triton RoPE核等技术。

**📊 数据集**

在 LLaMA‑3‑8B‑Instruct、Mistral‑7B‑v0.3 等模型上，使用 WikiText‑2、LongBench、Zero‑shot 任务进行评测。

**📈 对比分析**

与 SVD、PaLU 等基线相比，RAP 在 30% KV‑Cache 压缩下，KV‑Cache、参数、FLOPs 同时下降约 25–30%，并保持 PPL 8.82、推理延迟 预填/解码 分别为 83%/77%。

**⚠️ 局限性**

局限包括对 RoPE 变体的依赖、需要额外的 KD 训练以及在更高压缩比例下仍可能出现准确率下降。

---

## 171. Precoding-Oriented CSI Feedback Design with Mutual Information Regularized VQ-VAE

**arXiv ID:** 2602.02508 | [PDF](https://arxiv.org/pdf/2602.02508v1)

**作者:** Xi Chen `[一作]` (Rutgers University), Foad Sohrabi `[通讯]` (Nokia Bell Labs)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于向量量化变分自编码器（VQ-VAE）的预编码导向的信道状态信息（CSI）反馈框架，旨在提高大规模多输入多输出（MIMO）系统中CSI的压缩效率。

**💡 创新点**

创新点在于引入了一种可微的互信息下界估计器作为训练正则化器，以促进在固定反馈预算下有效利用学习到的码本，从而提高码字的均匀使用率和系统性能。

**🔧 技术方法**

使用了向量量化变分自编码器（VQ-VAE）和互信息（MI）正则化技术。

**📊 数据集**

使用了模拟生成的信道模型数据集，具体参数包括64个天线、2个用户设备（UE）和2条传播路径。

**📈 对比分析**

与基于全信道状态信息（CSIT）的经典线性预编码方法（如最大比率传输（MRT）和零强迫（ZF））以及其他深度学习方法进行了比较，结果表明所提方法在固定长度反馈下实现了更高的总可达速率，并且在反馈位数少于10位时表现出色。

**⚠️ 局限性**

限制在于该方法依赖于固定的反馈带宽，可能在极端情况下无法充分利用所有可用的反馈资源。

---

## 172. Aligning Forest and Trees in Images and Long Captions for Visually Grounded Understanding

**arXiv ID:** 2602.02977 | [PDF](https://arxiv.org/pdf/2602.02977v1)

**作者:** Byeongju Woo `[一作]` (Agency for Defence Development), Stella X. Yu `[通讯]` (University of Michigan)

**通讯引用:** 10103 | [OpenAlex ID](https://openalex.org/A5042014034)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨域层级对齐框架 CAFT，利用视觉细粒度到粗粒度的分层编码与文本子句到整体的层级编码，在无区域监督的情况下实现对长文本图像检索的精细语义对齐。

**💡 创新点**

创新点在于构建视觉与文本的层级结构并通过双层 Sigmoid 对齐损失实现全局与局部语义的统一对齐，从而突破仅全局或仅局部对齐导致的细节丢失问题。

**🔧 技术方法**

技术方法包括：细到粗的视觉分层编码（基于超像素聚类和逐层聚合）、层级文本 Transformer（子句 Transformer + 整体 Transformer）以及结合局部和整体的层级对齐损失。

**📊 数据集**

使用了约 30M 的长文本图像对（Merged‑30M，来源于 CC3M/12M/15M 的重标记集），并在 DCI、DOCCI、ShareGPT4V‑1k/10k、Urban‑1k、IIW 六大长文本检索基准以及零样本分割基准进行评估。

**📈 对比分析**

与仅用短句训练、短句预训练后微调以及从零训练的对齐模型进行对比，CAFT 在所有长文本检索基准上均获得 SOTA 结果（如 DCI R@1 97.6），在零样本分割任务上也显著优于 GroupViT、MaskCLIP 等方法。

**⚠️ 局限性**

局限性包括：对长文本的分句假设依赖较强，难以处理非句子结构或多语境长文本；以及在极端分辨率或更复杂的多模态关系上仍需进一步验证其泛化能力。

---

## 173. Design and Evaluation of Whole-Page Experience Optimization for E-commerce Search

**arXiv ID:** 2602.02514 | [PDF](https://arxiv.org/pdf/2602.02514v1)

**作者:** Pratik Lahiri `[一作]` (Amazon), Wenyang Liu `[通讯]` (Amazon)

**通讯引用:** 1267 | [OpenAlex ID](https://openalex.org/A5101735167)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一套基于因果推断的整页体验优化框架DV-WPX，并将品牌匹配度指标PR-WP-BMR集成到页面模板排序器中，提升了电商搜索的长期用户满意度和业务收益。

**💡 创新点**

创新点在于利用自然实验构建的因果模型把页面质量与12周后消费关联起来，并在多目标优化中引入直接衡量长期满意度的指标；同时通过像素和区域加权实现对视觉布局的细粒度评估。

**🔧 技术方法**

采用因果推断+双重机器学习、贝叶斯线性/Probit回归、Thompson采样等多种机器学习技术；结合区块级视觉权重与品牌匹配度计算。

**📊 数据集**

训练数据来源于亚马逊北美搜索日志，约1.5万条桌面、5.6万条移动展示记录（含历史客户特征、查询、位置等）。

**📈 对比分析**

通过离线RMSE、AUC评估与在线A/B测试对比，T2方案在短期收入提升0.05%、长期收入提升0.005%、搜索CTR提升0.02%，均显著优于对照组。

**⚠️ 局限性**

限制包括仅聚焦品牌相关性，12周观察窗口过长，区域划分固定且未考虑设备差异，且框架需扩展到其他质量维度。

---

## 174. Reading Between the Tokens: Improving Preference Predictions through Mechanistic Forecasting

**arXiv ID:** 2602.02882 | [PDF](https://arxiv.org/pdf/2602.02882v1)

**作者:** Sarah Ball `[一作]` (Ludwig-Maximilians-Universität in Munich), Frauke Kreuter `[通讯]` (University of Maryland)

**通讯引用:** 8872 | [OpenAlex ID](https://openalex.org/A5038390320)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分析大型语言模型内部表示来改进选举偏好预测，提出“机制化预测”方法；

**💡 创新点**

在传统仅靠输出概率的做法之外，利用内部MLP值向量的激活进行群体偏好估计，挖掘隐藏知识并提升预测精度；

**🔧 技术方法**

结合机制可解释性探测（probe）+ 线性分类器+ 统计距离（JS、Wasserstein）+ Monte Carlo 聚合等技术；

**📊 数据集**

使用七种LLM（Llama 3.1、Mistral、Gemma、Qwen）与六国（美国、英国、加拿大、德国、荷兰、新西兰）投票调查问卷以及党派立场数据；

**📈 对比分析**

将内部激活得到的分布与直接使用下一词概率得到的分布与真实调查数据做距离比较，发现机制化预测在多数国家/属性下距离更小，尤其在人口属性和高熵属性上提升显著；

**⚠️ 局限性**

需要白盒模型访问、较大计算成本，且只在第一阶近似下分析MLP影响，结果对低熵属性效果有限。

---

## 175. Trustworthy Blockchain-based Federated Learning for Electronic Health Records: Securing Participant Identity with Decentralized Identifiers and Verifiable Credentials

**arXiv ID:** 2602.02629 | [PDF](https://arxiv.org/pdf/2602.02629v1)

**作者:** Rodrigo Tertulino `[一作]` (Federal Institute of Education Science and Technology of Rio Grande do Norte), Laercio Alencar `[通讯]` (Federal Institute of Education Science and Technology of Rio Grande do Norte)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于区块链的可信联邦学习框架（TBFL），通过自我主权身份（SSI）对医疗机构身份进行预认证，防止Sybil和投毒攻击；

**💡 创新点**

创新点在于将分布式身份验证（DID + VC）嵌入到联邦学习过程，形成身份优先的安全模型，且在链上只存储哈希和权限标记，显著降低成本与延迟；

**🔧 技术方法**

采用Ethereum智能合约实现身份授权、模型提交验证；FedProx算法做异构数据的联邦学习；SMOTETomek做类别平衡；IPFS存储模型参数；DID/VC为身份凭证；

**📊 数据集**

使用MIMIC‑IV（ICU病人住院死亡预测）数据集，约546k条记录；

**📈 对比分析**

与传统FedAvg/无身份验证基线对比，TBFL在100轮后实现AUC‑ROC 0.954、召回0.890，成功抵御100% Sybil攻击，模型性能与基线相当；计算和经济成本仅为$18（10机构），对训练时间影响<0.12%；

**⚠️ 局限性**

局限包括：仅针对外部Sybil攻击，对内部恶意投毒（已认证机构）缺乏防御；梯度反演等隐私泄露攻击未完全解决；基于Ethereum主网的交易费用仍有上升风险；未来需结合DP/SMPC和更细粒度的信誉机制。

---

## 176. DECEIVE-AFC: Adversarial Claim Attacks against Search-Enabled LLM-based Fact-Checking Systems

**arXiv ID:** 2602.02569 | [PDF](https://arxiv.org/pdf/2602.02569v1)

**作者:** Haoran Ou `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**通讯引用:** 5982 | [OpenAlex ID](https://openalex.org/A5101720092)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于代理的攻击框架，用于在黑盒环境下对搜索驱动的 LLM 事实检查系统进行语义保持的对抗性主张攻击。

**💡 创新点**

创新点在于设计了三类针对检索、LLM 推理和结构复杂度的攻击策略，配合攻击有效性评估与语义守卫，系统化地探索攻击轨迹并显著提升对 LLM 系统的攻击成功率。

**🔧 技术方法**

技术手段包括使用 GPT‑4o 生成与评估对抗主张、语义相似度与 NLI 约束、Agent‑based 多轮迭代优化框架、以及代理模型（search‑enabled LLM）作为黑盒目标。

**📊 数据集**

实验使用 MOCHEG 基准集（1,642 条真/假主张），并在 HiSS、LEMMA、DEFAME 三个真实系统上进行评估。

**📈 对比分析**

与 FACTEVAL 四种基线（LEET、Homoglyph、Character Swap、Phonetic）对比，攻击成功率最高，导致目标系统准确率从约 78.7% 降至 53.7%，并展示出跨系统强转移性。

**⚠️ 局限性**

局限性包括仅关注主张层攻击，未覆盖证据或主张‑证据配对攻击；对抗样本在语义守卫约束下可能受限；实验仅在少数公开系统上验证，需进一步扩展到更多真实部署环境。

---

## 177. Structure-Preserving Learning Improves Geometry Generalization in Neural PDEs

**arXiv ID:** 2602.02788 | [PDF](https://arxiv.org/pdf/2602.02788v1)

**作者:** Benjamin D. Shaffer `[一作]` (University of Pennsylvania), Nathaniel Trask `[通讯]` (Sandia National Laboratories)

**通讯引用:** 4935 | [OpenAlex ID](https://openalex.org/A5038010160)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Geo-NeW，一种基于几何条件且保持结构的隐式神经 PDE 求解器，能够在新几何上直接求解稳态 PDE；

**💡 创新点**

创新点在于同时学习几何条件下的折减有限元空间和物理约束，严格保留守恒和边界条件，实现对未见几何的强泛化；

**🔧 技术方法**

使用有限元外部微分演算（FEEC）、Whitney forms、Transformer 编码几何特征（HKS、HC、SDF）以及隐式层求解与差分微分技术；

**📊 数据集**

在多种稳态 PDE 基准（Pipe、Elasticity、NACA、Poly‑Poisson、NS2d‑c、NS2d‑c++）以及自定义几何外推集上训练与评估；

**📈 对比分析**

与 DeepONet、U‑Net、Galerkin、GNOT、Transolver、Linear Attention 等基线相比，在分布内误差保持领先且在几何外推（多障碍、角度步）时误差显著下降（如 NS2d‑c++ 从>30 降至≈42），表现优异；

**⚠️ 局限性**

局限在目前仅处理二维稳态问题，缺乏对时变或三维问题的验证，且模型尺寸受限于折减维度和 Transformer 编码的计算量。

---

## 178. On the Feasibility of Hybrid Homomorphic Encryption for Intelligent Transportation Systems

**arXiv ID:** 2602.02717 | [PDF](https://arxiv.org/pdf/2602.02717v1)

**作者:** Kyle Yates `[一作]` (Clemson University), Mashrur Chowdhury `[通讯]` (Clemson University)

**通讯引用:** 4426 | [OpenAlex ID](https://openalex.org/A5088109829)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文针对智能交通系统（ITS）提出并理论评估了混合同态加密（HHE）方案，构建了基于Rubato的应用模型并计算了密文大小与通信开销；

**💡 创新点**

创新点在于首次将HHE与ITS结合，系统量化Rubato在RSU–云–TMC链路中的密文扩展与MTU碎片化影响，并证明HHE可将上传密文体积从数百千字节降至几十字节，显著降低延迟；

**🔧 技术方法**

主要技术为混合同态加密（Rubato）、同态加密与对称加密协同、理论参数估算及与传统HE（BFV、BGV、CKKS）密文大小和扩展因子对比；

**📊 数据集**

未使用实际数据集，文中仅以典型的BSM消息（200字节）为基准进行理论计算；

**📈 对比分析**

通过对比传统HE与HHE在密文大小、扩展因子、MTU碎片数等指标，结果显示HHE密文约41–195字节，扩展因子1.6–1.7，碎片数1，远优于BFV（131,939字节）、BGV（394,573字节）和CKKS（>1,050,129字节），从而在通信延迟上实现显著性能提升；

**⚠️ 局限性**

局限性包括：仅进行理论分析，未实验验证；HHE仍需在云端对称解密电路进行同态运算，最终返回大密文；对称解密电路的同态评估成本及多车端密钥管理、实际部署细节未被充分探讨。

---

## 179. Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting

**arXiv ID:** 2602.02832 | [PDF](https://arxiv.org/pdf/2602.02832v1)

**作者:** Rares Grozavescu `[一作]` (University of Cambridge), Mark Girolami `[通讯]` (Alan Turing Institute)

**通讯引用:** 19401 | [OpenAlex ID](https://openalex.org/A5045384249)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种连续时间 Koopman 自编码器，用于在低维潜在空间中通过 ODE 模型对流体动力学进行长时序预测

**💡 创新点**

创新点在于：①将潜在动力学从离散时间转为连续时间并通过数值积分或矩阵指数实现任意时间步长推演；②将 Koopman 操作与物理参数（如雷诺数、马赫数）耦合；③使用统一的线性潜在动态与 Transformer 编码器/解码器结合，实现高效、稳定的推演；④通过精确的矩阵指数实现一次性远时序预测，显著提升推演速度

**🔧 技术方法**

主要技术包括：Transformer 双流编码器、CNN 解码器、基于 LoRA 的参数化 Koopman 操作、RK4 数值积分、矩阵指数求解、组合损失（重构、回归、潜在一致、物理约束）

**📊 数据集**

使用两组 CFD 基准数据集：①不可压缩 wake 流（Re∈[100,1000]），②超音速圆柱流（Ma∈[0.50,0.90]）

**📈 对比分析**

与条件扩散模型 ACDM 进行对比，评估指标为 MSE、LSiM、推演时间。结果显示：在可压缩与不可压缩流场下，连续时间 KAE 在误差上与 ACDM 相近甚至更优，同时推演时间比 ACDM 快 300+ 倍，且在不同时间步长、长时序（240 步）和外推场景下保持稳定

**⚠️ 局限性**

局限性：对高度非线性现象（如冲击波）仍受线性假设限制；重构 L2 损失可能导致长时序下细节模糊；训练过程中需对 Koopman 操作进行严格正则化与学习率调度，参数调优较敏感

---

## 180. Chain of Simulation: A Dual-Mode Reasoning Framework for Large Language Models with Dynamic Problem Routing

**arXiv ID:** 2602.02842 | [PDF](https://arxiv.org/pdf/2602.02842v1)

**作者:** Saeid Sheikhi `[一作]` (University of Oulu), Saeid Sheikhi `[通讯]` (University of Oulu)

**通讯引用:** 285 | [OpenAlex ID](https://openalex.org/A5057569676)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Chain of Simulation（CoS），一种双模式推理框架，能够根据问题特征动态路由到不同专门的推理策略。

**💡 创新点**

创新点在于将数学、空间和多跳推理分别对应不同的推理模式，并通过自动问题分析实现模式选择，揭示LLM内在多种推理子系统。

**🔧 技术方法**

使用问题分析算法、模式选择规则、计算流+自洽采样、JSON 状态跟踪和混合事实抽取等技术。

**📊 数据集**

在 GSM8K、StrategyQA 和 bAbI 三个基准数据集上进行评估。

**📈 对比分析**

与 Direct、CoT、Structured CoT、Self‑Consistency 等基线比较，CoS 在 GSM8K 71.5%、StrategyQA 90.0%、bAbI 19.0% 的准确率，并比 Self‑Consistency 节省 54% 计算时间。

**⚠️ 局限性**

局限性包括：模式选择依赖手工规则、仅覆盖三种推理模式、对大规模状态跟踪难以处理、对语言与知识盲点敏感。

---

## 181. Auto-Augmentation Contrastive Learning for Wearable-based Human Activity Recognition

**arXiv ID:** 2602.02542 | [PDF](https://arxiv.org/pdf/2602.02542v1)

**作者:** Qingyu Wu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yiqiang Chen `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种端到端的自动增强对比学习方法AutoCL，专为可穿戴设备收集的低语义运动信号设计，实现了无监督预训练后可直接用于动作识别；

**💡 创新点**

创新点在于：①使用生成器从特征嵌入空间自动生成增强样本，避免手工设定的增强策略；②引入stop‑gradient和相关性降低机制提升正样本多样性与编码器表达能力；③在同一框架内完成自监督训练与自动增强的协同学习；

**🔧 技术方法**

技术包括Siamese网络架构、3层FCN编码器、MLP投影头、BiGRU生成器、NT‑Xent对比损失、Adam优化、stop‑gradient与相关性惩罚；

**📊 数据集**

实验使用四个公开HAR数据集：PAMAP2、UCIHAR、UTD‑MHAD和DSADS；

**📈 对比分析**

与现有对比学习基线（NNCLR、SimCLR、SimSiam、BYOL）及手工增强组合比较，AutoCL在所有数据集上均实现最高准确率，尤其在PAMAP2、UCIHAR和DSADS上显著提升；

**⚠️ 局限性**

局限性包括：①在噪声极低的细粒度动作集（如UTD‑MHAD）提升幅度有限；②生成器训练的收敛性与超参数敏感；③目前仅针对单模态IMU信号，未扩展到多模态或更大规模数据；

---

## 182. D$^2$Quant: Accurate Low-bit Post-Training Weight Quantization for LLMs

**arXiv ID:** 2602.02546 | [PDF](https://arxiv.org/pdf/2602.02546v1)

**作者:** Xianglong Yan `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22300 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为D^2Quant的权重仅后训练量化（PTQ）框架，用于在LLM的子4位精度下提升量化效果；

**💡 创新点**

创新点在于：①Dual-Scale Quantizer（DSQ）在下投影矩阵中引入可吸收的列尺度，实现对下投影量化的动态平滑；②Deviation-Aware Correction（DAC）在注意力模块后层归一化中校正量化诱发的均值偏移；

**🔧 技术方法**

技术主要包括：低位均匀量化、对称缩放与零点、迭代优化的DSQ、均值偏移估计与LayerNorm偏置校正；

**📊 数据集**

使用的评估数据集包括WikiText-2、C4、MMLU、七个零样本推理基准（PiQA、HellaSwag、ARC-Easy/Challenge、WinoGrande、RTE、OpenBookQA）以及多家LLM模型（LLaMA-3/3.1、Qwen-3 8B/14B/32B）；

**📈 对比分析**

与GPTQ、GPTAQ、BoA以及带Quarot旋转的变体对比，D^2Quant在2位权重量化时在WikiText-2、C4的困惑度以及MMLU和零样本任务平均准确率均优于SOTA，提升幅度可达3-6个百分点；

**⚠️ 局限性**

局限性包括：仅针对权重量化，未对激活量化或混合量化做深度探讨；对下投影外的其他矩阵如QKV未充分验证；DAC仅校正均值偏移，其他激活分布失真未覆盖；

---

## 183. Reshaping Perception Through Technology: From Ancient Script to Large Language Models

**arXiv ID:** 2602.02794 | [PDF](https://arxiv.org/pdf/2602.02794v1)

**作者:** Parham Pourdavood `[一作]` (University of California), Michael Jacob `[通讯]` (University of California)

**通讯引用:** 23328 | [OpenAlex ID](https://openalex.org/A5036054006)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

作者通过回顾从DNA、神经系统到写作、音乐以及LLM的媒介演化，探讨媒介如何塑造感知与意识，并将LLM视为新的艺术创作媒介。

**💡 创新点**

创新点在于将McLuhan的“媒介即信息”与生物学折叠-展开机制相结合，提出意识是不断折叠/展开的新颖媒介的结果，并将LLM纳入此框架。

**🔧 技术方法**

未使用具体技术，而是基于文献综述和跨学科理论分析。

**📊 数据集**

未使用实验数据集。

**📈 对比分析**

未进行方法比较，文章为理论综述。

**⚠️ 局限性**

局限在于缺乏实证验证、对LLM技术细节和用户体验的具体分析。

---

## 184. Learning-Infused Formal Reasoning: From Contract Synthesis to Artifact Reuse and Formal Semantics

**arXiv ID:** 2602.02881 | [PDF](https://arxiv.org/pdf/2602.02881v1)

**作者:** Arshad Beg `[一作]` (Maynooth University), Rosemary Monahan `[通讯]` (Maynooth University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了学习驱动的形式化推理（LIFR）框架，聚焦三大技术支柱：自动契约合成、基于LLM的图匹配式验证工件重用，以及统一的语义基础（UTP与机构理论）以实现跨工具、跨语言的可重用性；同时在VERIFYAI框架中给出契约合成的概念设计与实验方向；对Frama‑C等工具链进行大规模评估；并对未来研究提出挑战。

**💡 创新点**

创新点在于：① 将LLM与符号验证器形成闭环迭代的契约合成管线；② 通过将验证工件抽象为带属性的图，并用LLM生成语义嵌入，结合近似图匹配实现跨语义的工件重用；③ 以UTP与机构理论为底层，构建语言无关的语义中介，支持工件的形式化迁移与组合；④ 提出完整的实验评估方法与指标，为未来可重复性提供范式。

**🔧 技术方法**

技术主要包括：大语言模型（LLM）与提示工程；图构造与图匹配算法；语义嵌入与检索增强生成（RAG）技术；符号求解器（Z3、Alt‑Ergo、CVC4/5）与Frama‑C验证插件；基于机构理论的语义框架；以及人工审查与可追溯性元数据管理。

**📊 数据集**

主要使用的实验数据集为：由多份真实或合成的自然语言需求与对应的Frama‑C C 程序组成的评估集合；以及对多种SMT求解器（Alt‑Ergo、Z3、CVC4、CVC5）进行的验证成功率与执行时间统计；在文中未给出公开可复现的数据集，说明目前缺乏与工业需求匹配的版本化、可追溯语料。

**📈 对比分析**

方法比较：对Frama‑C PathCrawler、Runtime Error、Value Analysis等插件与多种SMT求解器进行实证评估，测得验证成功率与求解时间；结果表明求解器不稳定、路径覆盖限制以及配置敏感性成为实用瓶颈。文中未给出绝对性能数值，但指出不同求解器在同一问题集上存在显著差异，并通过实验验证了LLM提示与符号反馈交互对契约质量的影响。

**⚠️ 局限性**

局限性包括：① 缺乏版本化、可追溯的需求‑代码‑验证工件语料；② LLM输出易受提示变化、幻觉与不确定性的影响，缺乏可重复性；③ 工具链间互操作性差，缺乏工具中立的语义中介；④ 对大型系统的可扩展性不足（上下文窗口、路径爆炸）；⑤ 对不完整或冲突需求的处理不成熟；⑥ 信任与安全性问题（训练数据来源、可审计性、运行时监测）仍待解决。

---

## 185. Augmenting Parameter-Efficient Pre-trained Language Models with Large Language Models

**arXiv ID:** 2602.02501 | [PDF](https://arxiv.org/pdf/2602.02501v1)

**作者:** Saurabh Anand `[一作]` (TCS Research), Sachin Lodha `[通讯]` (TCS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将参数高效微调（CompFreeze）与大语言模型（LLM）结合，以提升网络安全任务（垃圾邮件检测、域名生成算法分类、CTI实体抽取）的模型效果。

**💡 创新点**

创新点在于：1) 设计两种LLM增强策略——(a) 利用LLM做无标签数据自动标注，再用标注数据微调CompFreeze模型；(b) 对CompFreeze模型置信度低的预测进行重路由，交由LLM进行补充推理；2) 将Compacter模块与多种层冻结策略（Odd‑LC、Even‑LC、Upper‑LC、Lower‑LC）结合，显著降低可训练参数量。

**🔧 技术方法**

采用的技术包括：Compacter架构、层冻结微调策略、GPT‑4和LLaMA‑3等LLM、提示工程、置信度阈值判定、F1 评估、训练时间与推理时间对比。

**📊 数据集**

使用的主要数据集为：Enron 电子邮件垃圾邮件数据集、APTNER（CTI实体抽取）、UMUDGA+Tranco（域名生成算法分类）。

**📈 对比分析**

与全模型微调基线进行F1、训练时间、参数量及推理吞吐量对比。结果显示：CompFreeze 在仅训练约0.06%参数、训练速度提升约40‑50%的情况下，F1 与全微调相当；LLM 低置信度路由后，F1 可进一步提升；LLM 自动标注的训练集表现与原始标注集相近。

**⚠️ 局限性**

局限性包括：LLM 自动标注的标签质量不稳定，误标导致模型误差；LLM 推理延迟较高，实际部署时需权衡；不同任务对层冻结策略的最佳选择不一致；实验仅覆盖网络安全领域，缺乏跨领域验证；未尝试主动学习或人机混合标注来进一步提升标签质量。

---

## 186. Testing Storage-System Correctness: Challenges, Fuzzing Limitations, and AI-Augmented Opportunities

**arXiv ID:** 2602.02614 | [PDF](https://arxiv.org/pdf/2602.02614v1)

**作者:** Ying Wang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Dejun Jiang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了存储系统正确性测试的现有技术，重点讨论了并发、长期运行、崩溃一致性、硬件语义验证和分布式故障注入等方法，并从存储系统的执行属性与失效机制出发，系统化整理这些技术。

**💡 创新点**

创新点在于提出一种面向存储的视角，按照执行特性和失效机制对测试技术进行归类，揭示了传统模糊测试与存储系统语义之间的系统性不匹配，并讨论了人工智能如何在语义感知与状态导向的测试中提供补充支持。

**🔧 技术方法**

采用了对现有测试方法的文献梳理与分类，阐释了模糊测试管线的四个阶段，并结合人工智能技术（如学习抽象、时序建模与自适应决策）对测试过程中的关键挑战进行分析。

**📊 数据集**

作为综述论文，本文未使用任何实验数据集。

**📈 对比分析**

由于为综述性工作，未进行实验对比或性能评估，本文主要通过概念分析与案例讨论来阐明各技术的适用范围与局限。

**⚠️ 局限性**

主要限制包括：存储系统的并发交叉、长期状态演化和跨层语义使得现有测试技术难以系统化；模糊测试在控制并发、状态探索和语义验证方面存在缺口；人工智能虽能提供抽象与自适应指导，但仍需人工定义接口、规范与诊断，整体自动化程度受限。

---

## 187. LEMON: Local Explanations via Modality-aware OptimizatioN

**arXiv ID:** 2602.02786 | [PDF](https://arxiv.org/pdf/2602.02786v1)

**作者:** Yu Qin `[一作]` (University of Bristol), Telmo de Menezes e Silva Filho `[通讯]` (University of Bristol)

**通讯引用:** 1256 | [OpenAlex ID](https://openalex.org/A5077682283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LEMON框架，提供一种模型无关的局部解释方法，利用稀疏组Lasso代理在多模态预测中同时输出模态级和特征级层级解释；

**💡 创新点**

创新点在于使用单一稀疏组惰性正则化的线性代理实现多模态统一层级解释，避免多模型、多步解释，并显著降低查询成本；

**🔧 技术方法**

核心技术包括：二值扰动和局部加权采样、基于组稀疏的稀疏组Lasso拟合、超像素/token等模态可解释单元分块；

**📊 数据集**

实验数据集涵盖：VQA v2（视觉‑语言）与REFLACX（临床图像+文本+表格），使用的黑盒模型包括CLIP、LXMERT和CaMCheX；

**📈 对比分析**

与MM‑SHAP、DIME等基准对比，评估指标为删除/插入AOPC、L0稀疏度、运行时和前向调用次数；LEMON在保持接近最优faithfulness的同时，查询成本比DIME低35‑67倍、运行时低2‑8倍，且生成更紧凑的解释；

**⚠️ 局限性**

局限性包括：解释对扰动随机性敏感，稳定性相对较低；仅在图像模态有ground‑truth验证；并非因果推断，易被误用或导致过度信任。

---

## 188. STEMVerse: A Dual-Axis Diagnostic Framework for STEM Reasoning in Large Language Models

**arXiv ID:** 2602.02497 | [PDF](https://arxiv.org/pdf/2602.02497v1)

**作者:** Xuzhao Li `[一作]` (Nanyang Technological University), Shiyu Hu `[通讯]` (Nanyang Technological University)

**通讯引用:** 389 | [OpenAlex ID](https://openalex.org/A5101437694)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了STEMVerse框架，将跨benchmark的20,374道STEM题目按学科细分和布鲁姆认知层级重新聚合，并用双轴能力矩阵评估LLM的推理能力。

**💡 创新点**

创新点在于将学科细化到27个子学科与布鲁姆六层认知水平相结合，打破传统“单一评分”评测模式，实现细粒度的诊断与“逻辑盲点”定位。

**🔧 技术方法**

技术手段包括GPT‑4o自动注释结合人工校验、布鲁姆认知分级、few‑shot提示、参数规模与对齐策略的系统实验，以及IAA指标验证注释质量。

**📊 数据集**

使用了20,374道题目，涵盖数学（MATH500、MathQA、GSM8K、AMC、AIME等）、物理（MMLU、PIQA、SciBench‑Physics等）、化学（ChemBench、GPQA‑Chemistry等）和生物（MMLU、GPQA‑Biology等）等多来源benchmark。

**📈 对比分析**

通过对Qwen和Llama系列在不同参数规模与训练方式下，在学科与认知层级维度上计算准确率，发现模型在记忆与理解层级随规模提升表现非线性，在应用与分析层级出现显著性能崩溃，指示高阶推理存在结构性瓶颈；指令微调有时会削弱符号推理能力。

**⚠️ 局限性**

局限性在于仅覆盖基础学科四大支柱，未考虑应用领域、多模态或代码执行需求，评测仅针对文本推理，未来需扩展到更广泛的STEM方向。

---

## 189. CaST: Causal Discovery via Spatio-Temporal Graphs in Disaster Tweets

**arXiv ID:** 2602.02601 | [PDF](https://arxiv.org/pdf/2602.02601v1)

**作者:** Hieu Duong `[一作]`, Long Nguyen `[通讯]` (University of Louisville)

**通讯引用:** 39559 | [OpenAlex ID](https://openalex.org/A5001230244)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 CaST 框架，用于从灾害相关推特中构建时空事件图并进行因果关系发现。

**💡 创新点**

创新点在于将语义、空间与时间上下文统一编码到事件节点中，构造时空事件图，并利用多头 GAT 学习有向因果边；同时引入焦点损失提升不平衡数据下的鲁棒性。

**🔧 技术方法**

采用了灾害领域预训练的 LLM（CrisisTransformer）生成语义嵌入，结合时间/空间特征构造节点；使用多头图注意力网络（GAT）进行图传播；使用焦点损失（Focal Loss）处理类别不平衡。

**📊 数据集**

使用自建的约 167K 条海飓风 Harvey 灾害推特数据集，按 MAVEN‑ERE 方案标注事件、因果关系、空间与时间信息。

**📈 对比分析**

与 8 个基线（ANM、RF、SVM、BiLSTM+Att、BERT、DECI、PPAT、DAPrompt）对比，CaST 在准确率 0.87、精确率 0.84、召回率 0.85、F1 0.85、AUC 0.85 上均实现最佳或接近最佳，表现出更平衡的精确召回能力。

**⚠️ 局限性**

局限在于仅评估了推文内的因果关系，未覆盖跨推文的因果链；数据集偏向单一灾害事件且存在类别不平衡；对其他灾害场景的泛化能力尚待验证。

---

## 190. Where Norms and References Collide: Evaluating LLMs on Normative Reasoning

**arXiv ID:** 2602.02975 | [PDF](https://arxiv.org/pdf/2602.02975v1)

**作者:** Mitchell Abrams `[一作]` (Tufts University), Matthias Scheutz `[通讯]` (Tufts University)

**通讯引用:** 9116 | [OpenAlex ID](https://openalex.org/A5044523801)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SNIC（Situated Norms in Context）测试床，用于评估大型语言模型在基于社会规范的指代解析上的能力。

**💡 创新点**

创新点在于构建了一个以物理情境为基础、聚焦日常任务的社会规范指代解析数据集，并通过人工验证与程序化扩增保证数据质量。

**🔧 技术方法**

技术主要包括文本情境构建、社会规范提取、程序化数据扩增、LLM 评估（含自然语言、Prolog 形式化和显式规范提示）以及性能分析。

**📊 数据集**

使用的数据集为 SNIC，总共 9,000 条实例，基于 51 条人工验证的种子场景通过规则生成。

**📈 对比分析**

在不同设置下评估 14 种 LLM，结果显示仅给出情境时平均准确率约 44%，显式提供规范后提升至约 70%，最佳模型 GPT‑4o‑mini 在完整输入下达 93% 的准确率。

**⚠️ 局限性**

局限包括仅文本评估、未涉及多模态感知、未尝试训练或强化学习改进、规范共识评估不足，以及对快速演进的 LLM 版本兼容性不一定通用。

---

## 191. Beyond Experience Retrieval: Learning to Generate Utility-Optimized Structured Experience for Frozen LLMs

**arXiv ID:** 2602.02556 | [PDF](https://arxiv.org/pdf/2602.02556v1)

**作者:** Xuancheng Li `[一作]` (Tsinghua), Qingyao Ai `[通讯]` (Tsinghua)

**通讯引用:** 4431 | [OpenAlex ID](https://openalex.org/A5089655391)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出SEAM，一种轻量化、专门为冻结LLM执行器定制的经验适配模块，通过参数化经验库生成结构化实例化指导来提升推理性能。

**💡 创新点**

创新点在于把经验库编码进轻量化模块参数中，直接生成针对实例的结构化经验，且使用GRPO在冻结执行器上进行无梯度传播训练，消除检索延迟与噪声。

**🔧 技术方法**

使用GRPO、经验生成采样、执行器rollout评估、教师强制学习等技术，结合Qwen3-0.6B/4B等基础模型。

**📊 数据集**

主要使用DAPO训练集及GSM8K、MATH、AIME24/25等数学推理基准，以及CodeContests、MBPP、HotpotQA、NQ等跨域数据。

**📈 对比分析**

与原始模型、直接训练执行器、MEM-0、Dynamic-Cheatsheet、Memento等基线比较，SEAM在四个数学基准上均取得最高pass@1，且在推理速度和时间‑到‑正确方面优于RAG方法。

**⚠️ 局限性**

局限在于仅在数学推理与少数执行器上验证，跨模型/跨域迁移能力有限，持续学习与灾难性遗忘未充分探索，且依赖执行器的日志数据。

---

## 192. Exploring Silicon-Based Societies: An Early Study of the Moltbook Agent Community

**arXiv ID:** 2602.02613 | [PDF](https://arxiv.org/pdf/2602.02613v1)

**作者:** Yu-Zheng Lin `[一作]` (University of Arizona), Pratik Satam `[通讯]` (University of Arizona)

**通讯引用:** 614 | [OpenAlex ID](https://openalex.org/A5028622409)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文通过对 Moltbook 平台上自动代理生成的子社群描述进行大规模数据挖掘，构建嵌入、聚类并用多模态 LLM 提炼主题，揭示硅基社会的社群结构与演化规律。

**💡 创新点**

创新点在于将子社群描述视为第一类观测数据，首次系统性地使用上下文嵌入、K‑means 聚类、t‑SNE 可视化、n‑gram 词云以及 Gemini 3 多模态 LLM 结合人工审核的混合方法，构建“数据驱动硅社会学”研究框架，并对硅基社会的“人类模仿”与“硅中心”两类结构进行定性与定量双重阐释。

**🔧 技术方法**

技术手段包括：① 4k 维上下文嵌入模型（如文本 Transformer）对子社群描述进行向量化；② K‑means 聚类（K=8）和 Elbow 方法确定簇数；③ t‑SNE 进行二维可视化；④ n‑gram (2–5) 词云生成与统计；⑤ Gemini 3 多模态 LLM 进行全局视觉推理与主题生成；⑥ 人工审核对 LLM 结果进行校正。

**📊 数据集**

数据集为 Moltbook 开放 API 收集的 12,758 条子社群描述，经过过滤与去重后得到 4,162 条高质量描述文本，用作嵌入与聚类分析的输入。

**📈 对比分析**

论文未给出传统机器学习指标（如轮廓系数或准确率），但通过 t‑SNE 可视化和词云一致性展示聚类的内部聚合性和主题连贯性；与人类社交网络的聚类模式进行对比，表明硅基社会呈现可识别的“人类模仿”与“硅中心”两大结构。

**⚠️ 局限性**

局限性包括：① 可能存在人类介入与供应商 LLM 训练偏差导致的描述污染；② 对训练数据中固有偏见缺乏校正；③ 仅基于静态描述文本，未涉及时间序列或因果推断；④ 对网络结构与互动模式的深入挖掘不足；⑤ LLM 生成的主题需人工校验，仍存在解释不确定性。

---

## 193. ADx3: A Collaborative Workflow for High-Quality Accessible Audio Description

**arXiv ID:** 2602.02684 | [PDF](https://arxiv.org/pdf/2602.02684v1)

**作者:** Lana Do `[一作]` (Northeastern University), Ilmi Yoon `[通讯]` (Northeastern University)

**通讯引用:** 372 | [OpenAlex ID](https://openalex.org/A5076251624)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ADx3 框架，集成 VLM 自动生成（GenAD）、人机协作编辑（RefineAD）和用户即时查询（AdaptAD），实现可扩展的音频描述工作流。

**💡 创新点**

创新点在于：① 将现代视觉‑语言模型与专业准入提示、上下文嵌入相结合，显著提升自动生成稿件质量；② 引入人机协作编辑接口，让志愿者和专业人员能快速纠正、精炼；③ 提供按需查询功能，让盲人/低视力用户根据需要即时获取信息，弥补固定叙述的局限。

**🔧 技术方法**

使用 Qwen2.5‑VL、Gemini 1.5 Pro 与 GPT‑4o 三大 VLM；prompting、上下文引导、音频转写（Whisper + Google STT）、文本转语音（Google TTS）、WAI‑ARIA 无障碍接口。

**📊 数据集**

评估数据集为 10 条 YouTube 视频（娱乐、教育、教学三类），共 30 条自动生成描述（每视频 3 模型），通过七位专业可访问性顾问进行人工打分。

**📈 对比分析**

方法：专家使用 DCMP 等七维度（准确、优先、适当、连贯、平等、交付策略、时序）对每个模型评分；结果显示 GPT‑4o 平均 4.05/5，Gemini 4.01/5，Qwen 3.78/5；按维度分析显示准确、连贯、平等得分高，优先、交付与时序得分相对较低，表明自动化已可生成“良好”稿件，但仍需编辑与交互提升。

**⚠️ 局限性**

局限性：样本量有限、仅评估专家而非盲人用户、缺乏公开大规模音频描述数据集、评估主观性强、未提供自动评测方案、未覆盖情感 TTS 与用户体验长期验证。

---

## 194. Reasoning about Reasoning: BAPO Bounds on Chain-of-Thought Token Complexity in LLMs

**arXiv ID:** 2602.02909 | [PDF](https://arxiv.org/pdf/2602.02909v1)

**作者:** Kiran Tomlinson `[一作]` (Microsoft Research), Jennifer Neville `[通讯]` (Microsoft Research)

**通讯引用:** 6725 | [OpenAlex ID](https://openalex.org/A5064439579)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究链式思维（CoT）推理中所需的推理标记（token）数量随输入规模增长的理论下界与上界，并通过实验验证其线性扩展性。

**💡 创新点**

首次在Bounded Attention Prefix Oracle（BAPO）框架下证明了三类典型BAPO‑hard任务（二元多数、三元匹配、图可达性）对CoT标记的Ω(n)下界，并给出近似匹配的上界；提出了自一致性约束的cBAPO模型以消除BAPO的复制漏洞。

**🔧 技术方法**

使用BAPO与cBAPO理论、信息流归约、Pigeonhole 原理、Turing机模拟以及针对每个任务的显式CoT构造；实验方面利用GPT‑5.2和Gemini 2.5 Pro进行推理标记计数与准确率测评。

**📊 数据集**

构造性的任务实例集：二元多数（长度n）、三元匹配（ℤ_m^n）、图可达性（n个节点、m条边）。

**📈 对比分析**

实验显示GPT‑5.2在三任务上推理标记使用与准确率近似线性增长；对不同CoT提示（无CoT、固定词数、算法化CoT）进行比较，较短预算导致准确率显著下降；与理论下界高度一致。

**⚠️ 局限性**

局限性包括：仅针对有限任务，未涵盖多模态或工具使用等现代LLM功能；BAPO模型对大规模词表的假设可能不完全适配实际模型；实验受模型接口限制，未覆盖所有主流模型。

---

## 195. ResQ: Realistic Performance-Aware Query Generation

**arXiv ID:** 2602.02999 | [PDF](https://arxiv.org/pdf/2602.02999v1)

**作者:** Zhengle Wang `[一作]` (Purdue University), Chunwei Liu `[通讯]` (Purdue University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于工作负载跟踪的细粒度、可执行SQL工作负载合成框架RPQG

**💡 创新点**

创新点在于：①将查询结构抽象为查询图，先做空间约束后再做谓词搜索；②结合重复查询池（精确哈希与参数化哈希）实现重用；③利用轻量级本地成本模型加速贝叶斯优化；④发布了新的高重复率跟踪Bendset

**🔧 技术方法**

技术包括：查询图生成与剪枝、贝叶斯优化(BO)谓词搜索、局部CPU/扫描字节预测模型、LLM驱动的查询图转SQL翻译、重复查询池

**📊 数据集**

使用公开云工作负载跟踪Snowset、Redset以及新发布的Bendset（40M查询、8天）以及TPC‑H/TPC‑DS代理数据

**📈 对比分析**

与三种基线（SQLSmith‑Q、LLMGen、SQLBarber）对比，RPQG在CPU时间和扫描字节的Q‑error上显著更低（中位数≈1.1–1.8），95/99百分位误差大幅降低，结构一致性MAE几乎为0，LLM token消耗低，生成延时比SQLBarber快≈6–13倍

**⚠️ 局限性**

局限：仍依赖代理数据分布与查询图假设，无法完全恢复原始SQL语义；对极端多样化或未见过的结构可能效果下降；需要手工设定结构与性能约束参数

---

## 196. TRACE: Temporal Radiology with Anatomical Change Explanation for Grounded X-ray Report Generation

**arXiv ID:** 2602.02963 | [PDF](https://arxiv.org/pdf/2602.02963v1)

**作者:** OFM Riaz Rahman Aranya `[一作]` (University of Texas at San Antonio), Kevin Desai `[通讯]` (University of Texas at San Antonio)

**通讯引用:** 451 | [OpenAlex ID](https://openalex.org/A5076318084)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 TRACE 模型，实现基于先前与当前胸部 X 光片的时序变化检测、分类（恶化/改善/稳定）与空间定位，并生成带边界框的自然语言报告。

**💡 创新点**

首次将时序比较、变化分类与视觉定位统一到单一框架，并发现仅靠时间对比或定位单独不足以实现变化检测，需两者共同学习。

**🔧 技术方法**

采用冻结的 BioViL‑T 视觉编码器、MLP 投影、Vicuna‑7B 语言模型与 LoRA 微调、跨图像特征拼接和自注意力实现时序对比，并通过视觉‑语言对齐生成带坐标的文本。

**📊 数据集**

在 MIMIC‑CXR 与 Chest ImaGenome 上构建 79,202/22,553 对时序 X 光图像，并配有基于场景图的空间标注与三类变化标签。

**📈 对比分析**

与仅做分类的 CheXRelNet/Former 等基准相比，TRACE 在三类变化分类准确率 48.0%，定位 IoU>0.5 达到 90.2%，生成文本 BLEU‑4 0.260、RadGraph F1 0.406，展示了统一框架的优势。

**⚠️ 局限性**

对肺段细微变化检测仍低（<50%），改善类识别召回仅 26%，模型在无定位监督时完全失效，且仅支持两张图，无法处理更长时间序列。

---

## 197. The First Mass Protest on Threads: Multimodal Mobilization and AI-Generated Visuals in Taiwan's Bluebird Movement

**arXiv ID:** 2602.02640 | [PDF](https://arxiv.org/pdf/2602.02640v1)

**作者:** Ho-Chun Herbert Chang `[一作]` (Dartmouth), Tracy Weener `[通讯]` (Dartmouth)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5114526121)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了台湾2024年蓝鸟运动在Threads平台上的文本与视觉传播模式，分析算法曝光与用户参与之间的差异。

**💡 创新点**

提出“kawaii毒性”概念，将可爱的生成式AI图像与政治攻击相结合，并揭示算法曝光与社会互动的分歧。

**🔧 技术方法**

采用LLM零射注释、CatBoost梯度提升树与SHAP解释器对文本与图像特征进行预测与解释。

**📊 数据集**

使用Threads公开API获取的62,321条帖子与21,572张图片的蓝鸟运动数据。

**📈 对比分析**

通过比较模型的R²(0.42)与SHAP重要性，验证文本中的纪念、个人陈述、行动号召等特征及图像中的人像与AI符号对点赞、转发的显著预测作用。

**⚠️ 局限性**

局限在于数据仅来自中文关键词，缺乏跨平台对比，且LLM标注可能存在偏差，无法完全排除虚假账号影响。

---

## 198. VerIde ECG Biometrics: Verification and Identification

**arXiv ID:** 2602.02776 | [PDF](https://arxiv.org/pdf/2602.02776v1)

**作者:** Scagnetto Arjuna `[一作]` `[通讯]` (Azienda Sanitaria Universitaria Giuliano Isontina), Scagnetto Arjuna (Azienda Sanitaria Universitaria Giuliano Isontina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在大规模临床数据库上系统评估了心电图（ECG）生物识别的可行性与隐私风险

**💡 创新点**

首次在20k+身份大规模数据集上使用端到端的ArcFace嵌入模型，并通过统一的实验协议和严格的低FAR指标展示了ECG的个人签名可被重识别

**🔧 技术方法**

采用Siamese MLP对fiducial特征、ArcFace多分类+余弦余弦距离对tabular特征、以及1D CNN+ArcFace对全波形进行嵌入学习，使用全局归一化和两阶段重排序策略

**📊 数据集**

基于意大利Giuliano‑Isontina医院约56k个10s ECG样本的多通道、1kHz采样率数据库，包含约54k名患者的多时点记录，过滤后得到约92k个样本的标准化集

**📈 对比分析**

通过统一的train/val/test按身份分层拆分，在所有-对-所有评估下得到EER=2.53%，FAR=10⁻³下TAR=0.908，FAR=10⁻⁴下TAR=0.820；闭集识别Rank@1=0.812、Rank@10=0.910；开放集两阶段DIR@FAR最高0.976，证明模型在严格安全阈值下仍保持高检索性能

**⚠️ 局限性**

实验局限于单一设备（ELI250）和特定心律筛选，未验证跨中心或多设备迁移，且对心率异常、病理波形的鲁棒性未系统评估

---

## 199. A Vision-Based Analysis of Congestion Pricing in New York City

**arXiv ID:** 2602.03015 | [PDF](https://arxiv.org/pdf/2602.03015v1)

**作者:** Mehmet Kerem Turkcan `[一作]` (Columbia University), Andrew Smyth `[通讯]` (Columbia University)

**通讯引用:** 5084 | [OpenAlex ID](https://openalex.org/A5025805379)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用纽约市910台低分辨率交通摄像机，构建端到端的计算机视觉管道，对城市级交通流量进行实时检测与分析，从而评估拥堵定价政策的实施效果。

**💡 创新点**

① 为低分辨率摄像头量身定制的 YOLO‑LR 模型；② 具备百台摄像头并行处理的分布式实时架构；③ 通过峰时差异（Peak Hour Differential, PHD）度量政策前后拥堵变化，既捕捉峰值幅度又关注峰时变动。

**🔧 技术方法**

使用 YOLOv11 的轻量版（YOLO‑LR）与 TensorRT 编译的推理；分布式多线程抓帧、批处理；SQLite 存储检测结果；滚动平均、时间分段和窗口峰值计算等统计方法。

**📊 数据集**

训练集：COCO 数据集（352×352 低分辨率），测试集：NYC 910 台摄像机的实时视频流；用于评估的实际交通计数数据由摄像机检测得到。

**📈 对比分析**

将 YOLO‑LR 与标准 YOLOv11（352×352 及 640×640）在 COCO 验证集上对比，指标为 mAP50 与 mAP50‑95；结果表明 YOLO‑LR 在低分辨率场景下的检测准确率（如车辆类 mAP50≈0.537）优于高分辨率模型，尤其在车、摩托、公交等交通主体上表现更佳。

**⚠️ 局限性**

① 计数包括静止车辆，可能导致高街道停放区误差；② 基线对比期仅覆盖秋季，缺乏完整季节变化；③ 未区分车道与行驶方向；④ 摄像机即时计数仅为代理指标，未直接映射行驶时间或拥堵级别；⑤ 2025 年初极端低温可能影响通勤行为。

---

## 200. Every Bit Counts: A Theoretical Study of Precision-Expressivity Tradeoffs in Quantized Transformers

**arXiv ID:** 2602.02707 | [PDF](https://arxiv.org/pdf/2602.02707v1)

**作者:** Sayak Chakrabarti `[一作]` (Columbia University), Josh Alman `[通讯]` (Columbia University)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5006981411)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文理论研究量化Transformer在不同精度下的可表达性，并给出对等式检查任务的严格一比特阈值。

**💡 创新点**

创新点在于首次将通信复杂度与Transformer构造结合，证明了精度降低一比特就会导致不可表达的紧迫界限，并同样扩展到浮点格式。

**🔧 技术方法**

主要技术包括通信复杂度下的归约、显式低精度Transformer构造、固定点与浮点数值分析。

**📊 数据集**

使用自定义的等价检查（Equality）数据集进行实验，规模在 15-100 位字符串。

**📈 对比分析**

通过对 INT4/INT6/INT8/INT12、FP8_E4M3/FP8_E5M2/FP16 等不同精度进行后训练和量化感知训练，对比其在等价检查任务上的准确率，低精度时准确率急剧下降。

**⚠️ 局限性**

局限性：仅针对单层Transformer和等价函数，未证明多层或更一般任务的可表达性，也未给出多层下的下界。

---

## 201. How Much Information Can a Vision Token Hold? A Scaling Law for Recognition Limits in VLMs

**arXiv ID:** 2602.02539 | [PDF](https://arxiv.org/pdf/2602.02539v1)

**作者:** Shuxin Zhuang `[一作]` (City University of Hong Kong), Youzhi Zhang `[通讯]` (Centre for Artificial Intelligence and Robotics, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对视觉令牌在文档视觉信息压缩中的容量极限进行系统研究，并通过可控实验揭示三阶段相变现象。

**💡 创新点**

发现视觉令牌存在稳定、失稳与崩溃三个阶段，提出由视觉密度与平均令牌负载决定的概率尺度法则，揭示失稳源自ViT的空间对齐敏感性，崩溃源自信息容量饱和。

**🔧 技术方法**

利用Vision Transformer、DeepSeek‑OCR、InternVL3.5‑8B、Qwen2.5‑VL‑8B等视觉语言模型，结合像素移位实验、视觉密度对齐实验和概率混合模型推断。

**📊 数据集**

合成英文文本图像，覆盖小说、法律、经济、医学、报纸、信件六大领域，并通过块级随机打乱保证仅依赖视觉信息。

**📈 对比分析**

在不同分辨率下对比模型的编辑距离性能，发现相同视觉密度下不同令牌负载导致的性能差异；通过概率尺度法则可精确预测在给定令牌预算下的最大文本长度，实验表明模型在失稳区可通过像素平移恢复，而在崩溃区无法修复。

**⚠️ 局限性**

研究仅覆盖ViT‑基视觉令牌，未覆盖其它分词方式；仅使用英文文本，未考虑信息密度更高的语言；未系统探讨对齐敏感性或容量限制的缓解策略。

---

## 202. Deepfake Pornography is Resilient to Regulatory and Platform Shocks

**arXiv ID:** 2602.02754 | [PDF](https://arxiv.org/pdf/2602.02754v1)

**作者:** Alejandro Cuevas `[一作]` (Princeton University), Manoel Horta Ribeiro `[通讯]` (Princeton University)

**通讯引用:** 1279 | [OpenAlex ID](https://openalex.org/A5011195481)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了美国《TAKE IT DOWN Act》通过及 MrDeepfakes 关闭后，对其他三大成人内容网站中深度伪造色情（deepfake pornography）相关帖子、贡献者和请求量的影响，采用合成控制方法对不同子板块进行准实验。

**💡 创新点**

创新点在于将立法与平台下线视为复合干预事件，首次使用合成控制对多平台子板块进行比较，揭示干预后内容并未消减而是转移，并发现时间上的异质性（有预期效应、异步增长）。

**🔧 技术方法**

采用合成控制（synthetic control）技术、置换检验（placebo test）以及时间序列统计手段来估计干预效果，并计算MSPE比率和平均差异。

**📊 数据集**

使用来自三大公开成人内容社区（包含 AI/深伪标签和请求板块）的每周帖子数、新贡献者数，以及对约1,000条请求文本进行人工标注的结果作为数据集。

**📈 对比分析**

通过构造对照组合生成合成对照，计算后期与对照的平均差异（gap）和MSPE比率，结果显示大多数受试子板块出现显著正向差异（p<0.05），表明干预导致活动上升，验证方法具有较好的检验能力。

**⚠️ 局限性**

局限包括：对照单元可比性不足、标签行为可能导致测量偏差、无法将立法与平台下线单独分离、缺乏消耗端数据、部分子板块预处理拟合不足等。

---

## 203. Sparsely Supervised Diffusion

**arXiv ID:** 2602.02699 | [PDF](https://arxiv.org/pdf/2602.02699v1)

**作者:** Wenshuai Zhao `[一作]` (Aalto University), Arno Solin `[通讯]` (Aalto University)

**通讯引用:** 1708 | [OpenAlex ID](https://openalex.org/A5014200248)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种稀疏监督扩散（SSD）方法，通过在训练时对像素进行随机遮蔽，只对未遮蔽像素进行回归学习，以减少扩散模型的空间不一致生成问题；

**💡 创新点**

创新点在于用极低比率（高达98%）的遮蔽策略来显式抑制模型对局部相关性的过度拟合，从而改变数据协方差谱并提升生成的空间一致性与泛化能力；

**🔧 技术方法**

技术上采用了基于扩散模型的去噪回归（以及流匹配框架）以及随机二值遮蔽掩码，在训练损失中仅计算未遮蔽像素的平方误差；

**📊 数据集**

实验数据集包括CIFAR‑10、CelebA‑50K、LSUN Bedroom以及ImageNet的32×32子集；

**📈 对比分析**

与传统无遮蔽扩散模型（flow matching）比较，SSD在大多数数据集上能保持相近甚至更优的FID分数，同时显著提升训练稳定性、降低过拟合与空间不一致的比例；

**⚠️ 局限性**

限制在于遮蔽比例的最佳设置尚未系统优化，且对极高分辨率图像的适用性与理论解释仍需进一步验证。

---

## 204. SVD-ViT: Does SVD Make Vision Transformers Attend More to the Foreground?

**arXiv ID:** 2602.02765 | [PDF](https://arxiv.org/pdf/2602.02765v1)

**作者:** Haruhiko Murata `[一作]` (Meijo University), Kazuhiro Hotta `[通讯]` (Meijo University)

**通讯引用:** 2091 | [OpenAlex ID](https://openalex.org/A5103163418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 Vision Transformer（ViT）中引入奇异值分解（SVD）来提取并聚合前景特征，从而提升分类性能

**💡 创新点**

创新点包括：① 将SVD直接应用于动态特征图而非权重；② 设计 SPC 模块产生前景聚合 token；③ 引入 SSVA 进行输入自适应的奇异向量聚合；④ 通过 ID‑RSVD 生成输入相关的随机投影以稳定子空间估计

**🔧 技术方法**

使用的技术包括：ViT 基础架构、随机化奇异值分解（RSVD）、层归一化、注意力机制、Adam 优化器、余弦退火学习率等

**📊 数据集**

实验数据集涵盖：CUB‑200‑2011、FGVC‑Aircraft、Stanford Cars、Food‑101 以及 CIFAR‑100

**📈 对比分析**

与 ViT 基线（CLS 位置不变或可移动）进行对比，SVD‑ViT 在所有数据集均超越基线，最高提升约 2.82 %（CUB‑200‑2011）和 2.82 %（FGVC‑Aircraft），整体提升约 1‑2 个百分点

**⚠️ 局限性**

局限性：仅在预训练模型微调下验证；SSVA 与 ID‑RSVD 的效果依赖于数据集且并非始终有利；SVD 的符号不确定性导致可视化与训练稳定性受影响；未利用右奇异向量和奇异值来进一步优化

---

## 205. From Hanging Out to Figuring It Out: Socializing Online as a Pathway to Computational Thinking

**arXiv ID:** 2602.03017 | [PDF](https://arxiv.org/pdf/2602.03017v1)

**作者:** Samantha Shorey `[一作]` (University of Texas at Austin), Samuel C. Woolley `[通讯]` (University of Texas at Austin)

**通讯引用:** 1417 | [OpenAlex ID](https://openalex.org/A5028631384)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对Scratch平台上超过14,700条公开评论的定性与定量分析，本文构建并验证了“参与式调试”这一概念，并系统揭示了其在社区中的普及率及影响因素。

**💡 创新点**

首次将参与式调试概念化为一个可识别的学习行为，发现其产生依赖于社区持续性、问题可识别性和话题开放性三种社会情境，弥补了以往仅关注技术或兴趣驱动的研究空白。

**🔧 技术方法**

采用归纳式的Grounded Theory、内容分析以及Krippendorff’s α互评可靠性检验等质性研究方法，结合文本编码与频率统计，对Scratch评论进行深入剖析。

**📊 数据集**

使用Scratch公开项目及其评论数据集，共6,453个项目、14,733条评论，其中抽样部分包含640个项目（Stage I）、600个项目（Stage II）和53位用户的项目历史（5,213个项目、3,779条评论）（Stage III）。

**📈 对比分析**

通过内容分析与互评一致性检验（α = 0.79）评估编码质量，结果显示参与式调试约占8.8%（置信区间6.7–11.5%）的项目，表明此行为虽不常见但具有显著存在性；未涉及算法性能对比，主要关注行为普及与社会条件关联。

**⚠️ 局限性**

研究局限在于样本仅限有三条以上公开评论的项目，可能导致普及率估计偏差；只关注公开评论，未捕捉私聊或线下交流；缺乏实验干预，难以确立因果关系；并未检验不同年龄或性别等个体差异对参与式调试的影响。

---

## 206. Cross-Temporal Attention Fusion (CTAF) for Multimodal Physiological Signals in Self-Supervised Learning

**arXiv ID:** 2602.02784 | [PDF](https://arxiv.org/pdf/2602.02784v1)

**作者:** Arian Khorasani `[一作]` (HEC Montreal), Théophile Demazure `[通讯]` (HEC Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种跨时序注意力融合（CTAF）框架，用于在EEG与外围生理信号异步的多模态情感建模，通过自监督学习构建时间感知的软对齐和鲁棒的片段嵌入。

**💡 创新点**

创新点在于：①利用双向跨时序注意力学习软时间对齐而非假设同步；②以对齐为正则的对比目标实现自监督；③设计了掩码感知、轻量融合门和时间可调一致性约束，专门针对EEG与生理信号的时间不匹配问题。

**🔧 技术方法**

核心技术包括自监督对比学习（InfoNCE）、双向跨时序多头注意力、软时间对齐损失、VICReg风格的稳定化约束、掩码平均与注意力池化、轻量融合门、时间抖动一致性以及可选的弱监督回归。

**📊 数据集**

使用的主要数据集是K-EmoCon，它提供5秒连续的情绪自评（唤醒-愉悦）以及由Empatica E4与NeuroSky记录的EEG与外周生理（BVP、EDA、皮肤温度、心率）信号。

**📈 对比分析**

通过在K-EmoCon上进行留一法交叉验证，与基线HyperFuseNet进行对比；CTAF在三分桶情绪识别中达成0.62的准确率、0.61的宏观F1，均优于HyperFuseNet（0.58/0.57）；在对齐评估上，匹配对的余弦相似度显著提升（+0.189），并且跨模态令牌检索率提升到0.350（EEG→Phys）和0.265（Phys→EEG）以上。

**⚠️ 局限性**

局限性包括：①实验仅在单一K-EmoCon数据集上验证，缺乏更广泛的泛化评估；②注意力机制的计算开销较高；③对齐质量对部分受试者仍较宽泛，受传感器噪声和窗口标签的影响；④缺乏对不同时间容忍度（τ）与对齐先验的深入探讨。

---

## 207. Semantics-Aware Generative Latent Data Augmentation for Learning in Low-Resource Domains

**arXiv ID:** 2602.02841 | [PDF](https://arxiv.org/pdf/2602.02841v1)

**作者:** Jae-Sung Bae `[一作]` (University of Illinois Urbana-Champaign), Minje Kim `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2586 | [OpenAlex ID](https://openalex.org/A5064582903)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在基础模型提取的低维潜在空间中使用条件扩散模型进行数据增强，以解决低资源和长尾分布场景下的识别任务。

**💡 创新点**

创新点在于将语义增强的标签/子域信息作为条件注入潜在空间生成，利用基础模型的语义知识实现低资源数据的高质量合成。

**🔧 技术方法**

采用预训练基础模型（如Whisper、WavLM、CLIP）+轻量级任务适配器+条件扩散模型（CFG）以及语义嵌入进行潜在空间生成。

**📊 数据集**

在多语言情感识别数据集（EmoBox）和长尾图像分类数据集（ImageNet‑LT、Places‑LT）上进行验证。

**📈 对比分析**

与无增强或输入空间生成、传统长尾方法相比，GeLDA在零样本SER上无加权平均召回提升6.13%，在ImageNet‑LT上尾部准确率达到74.7%，显著优于最新SOTA。

**⚠️ 局限性**

限制在于仅针对分类任务、只使用轻量级适配器、对基础模型中间层的潜在空间探索有限，未验证在序列学习等更广泛任务中的效果。

---

## 208. Spatiotemporal Decision Transformer for Traffic Coordination

**arXiv ID:** 2602.02903 | [PDF](https://arxiv.org/pdf/2602.02903v1)

**作者:** Haoran Su `[一作]` (New York University), Hanxiao Deng `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种将多智能体交通信号控制视为条件序列生成问题的Multi-Agent Decision Transformer（MADT），通过图注意力捕捉空间依赖、Transformer编码捕获时间动态，并以return-to-go做目标条件；

**💡 创新点**

创新点在于将图注意力机制嵌入Decision Transformer，实现空间结构化协同；引入返回‑to‑go 目标条件以实现网络级性能约束；并通过并行动作生成实现高效推理；

**🔧 技术方法**

技术主要包括图注意力网络（GAT）、Transformer 编码器、return-to-go 目标条件、离线数据监督训练和可选的在线微调；

**📊 数据集**

使用CityFlow仿真平台生成的历史交通数据，包含从MaxPressure策略采集的约720k决策点，涵盖不同需求级别的合成网格（3×3、4×4）和真实城市网络（亚特兰大16路口、波士顿15路口）；

**📈 对比分析**

与固定时序、MaxPressure、FRAP、MPLight、AttendLight、MAPPO、CoLight、MAT、IndependentDT等基线对比，MADT在四个环境中平均缩短行程时间5.3–5.9%，吞吐量提升5.5–6.2%，协调指数提升约50%，并在高需求场景下提升幅度最大（≈8.1%）；

**⚠️ 局限性**

局限包括：对大规模网络的计算复杂度（O(N²)），需要大量离线轨迹数据，受训练分布限制，缺乏对极低/极高需求以及突发事件的鲁棒性，且当前模型仅处理统一相位配置，无法直接处理混合相位网络；

---

## 209. Automated Dysphagia Screening Using Noninvasive Neck Acoustic Sensing

**arXiv ID:** 2602.02725 | [PDF](https://arxiv.org/pdf/2602.02725v1)

**作者:** Jade Chng `[一作]`, Philip A Weissbrod `[通讯]` (University of California San Diego)

**通讯引用:** 958 | [OpenAlex ID](https://openalex.org/A5059905131)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于非侵入性颈部声学信号的自动化咽功能评估框架，通过收集吞咽时的声音并提取域相关特征，训练机器学习模型检测吞咽困难；

**💡 创新点**

创新点在于：①使用临床标准FEES过程中实时记录颈部声学；②构建超过600个吞咽事件的大型数据集；③设计基于频域、幅度、曲线面积等域相关特征的特征集，显著优于传统OpenSMILE或OPERA嵌入；④实现实时分割与患者级别聚合；

**🔧 技术方法**

主要技术包括：音频预处理与分割（固定阈值与滑动窗口）、FFT/ STFT频谱分析、幅度/曲线面积特征提取、OPERA与OpenSMILE嵌入、随机森林分类、SHAP特征重要性分析；

**📊 数据集**

使用UC San Diego的49名自报告吞咽困难患者数据，包含392条音频记录、617个吞咽事件，按PAS评分（1–8）进行标注；

**📈 对比分析**

通过5折患者级别拆分，对比域相关特征、OPERA嵌入、OpenSMILE嵌入；二分类（异常vs正常）模型域特征AUC‑ROC 0.904，结合固定阈值分割+最大风险聚合的AUC‑ROC最高达0.942；三分类（严重度）性能明显下降；

**⚠️ 局限性**

主要局限在于样本量中等、患者分布不均，分割算法对不同音频仍需优化，且尚未在家庭或多中心环境中验证，影响泛化能力。

---

## 210. CADENT: Gated Hybrid Distillation for Sample-Efficient Transfer in Reinforcement Learning

**arXiv ID:** 2602.02532 | [PDF](https://arxiv.org/pdf/2602.02532v1)

**作者:** Mahyar Alinejad `[一作]` (University of Central Florida), George Atia `[通讯]` (University of Central Florida)

**通讯引用:** 1753 | [OpenAlex ID](https://openalex.org/A5003612688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种融合策略自动机与策略蒸馏的上下文感知蒸馏框架 CADENT，用于强化学习的迁移学习

**💡 创新点**

创新点在于将长时序战略指导与短时序战术指导统一为一条学习信号，并引入经验门控的可信度机制，使学生能在状态-动作层面动态调节对教师的依赖

**🔧 技术方法**

采用 Q‑learning、策略蒸馏、自动机蒸馏、sigmoid 门控、TD‑误差波动追踪等技术实现框架

**📊 数据集**

使用四类实验环境：稀疏奖励的 Blind Craftsman 与 Dungeon Quest（网格世界），物理控制的 Mountain Car Collection，工业物流的 Warehouse Robotics 作为数据集

**📈 对比分析**

与仅使用自动机蒸馏（AD）、仅使用策略蒸馏（PD）以及无迁移基线比较，CADENT 在所有环境中实现了 40–60% 的样本效率提升，并保持甚至超越教师的最终性能

**⚠️ 局限性**

局限在于对非平稳的经验门控机制缺乏完整理论收敛分析，且目前仅在表格学习设置下验证，未在深度函数逼近或多教师情境下测试

---

## 211. BatCoder: Self-Supervised Bidirectional Code-Documentation Learning via Back-Translation

**arXiv ID:** 2602.02554 | [PDF](https://arxiv.org/pdf/2602.02554v1)

**作者:** Jingwen Xu `[一作]` (Fudan University), Xiaoqing Zheng `[通讯]` (Fudan University)

**通讯引用:** 1128 | [OpenAlex ID](https://openalex.org/A5017835517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了自监督双向代码-文档学习框架 BatCoder，利用回译方法从无标注代码中学习代码生成与文档生成。

**💡 创新点**

通过将代码生成与文档生成视为回译循环，以代码重构相似度作为隐式奖励，实现无监督的双向优化，避免了对高质量代码-文档对的依赖。

**🔧 技术方法**

使用强化学习（Reinforce++）、代码相似度度量（CSSG）、自监督回译及奖励设计等技术。

**📊 数据集**

仅使用 CodeXGLUE Code‑Text 代码样本（Python、Ruby、Go 等），不使用任何标注的代码-文档对。

**📈 对比分析**

在 HumanEval、MBPP、HumanEval+、MBPP+ 以及 MultiPL‑E 的 Ruby/Go 上与 Qwen2.5‑Instruct、CodeT5+、StarCoder2 等公开基线对比，7B 版 BatCoder 在 HumanEval+ 及 MBPP+ 上分别提升至 83.5%/81.0%，并在低资源语言上实现显著提升。

**⚠️ 局限性**

仍依赖代码相似度度量的准确性，奖励设计受限于特定语言或结构，缺乏对更复杂程序推理或执行正确性的直接反馈。

---

## 212. TopoPrune: Robust Data Pruning via Unified Latent Space Topology

**arXiv ID:** 2602.02739 | [PDF](https://arxiv.org/pdf/2602.02739v1)

**作者:** Arjun Roy `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**通讯引用:** 45990 | [OpenAlex ID](https://openalex.org/A5031161187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于拓扑的统一潜在空间数据裁剪方法（TopoPrune），通过双尺度拓扑分析实现鲁棒的数据子集选择。

**💡 创新点**

创新点在于将全局曼菲索嵌入与可微分持久同调局部优化结合，形成全局密度与局部持久性统一评分，从而根除几何脆弱性并实现跨架构的高可迁移性。

**🔧 技术方法**

使用UMAP进行低维曼菲投影、核密度估计、可微分多参数持久同调（differentiable persistent homology）与Optimal Transport进行局部优化，并采用邻域标签纯度（NLPS）过滤噪声标签。

**📊 数据集**

在CIFAR‑10、CIFAR‑100和ImageNet‑1K三大图像分类数据集上进行实验。

**📈 对比分析**

与随机、Moderate、FDMat、Forgetting、Glister、LCMat‑S、CCS、D2等方法比较，TopoPrune在90%裁剪率下取得最高平均精度、最低方差，并在噪声扰动与跨架构迁移中表现出更高的鲁棒性与稳定性。

**⚠️ 局限性**

主要局限在于需要预训练模型的特征嵌入，且可微持久同调优化计算量较大，可能对大规模数据和高维嵌入不够高效；目前仅在分类任务上验证，其他任务的适用性仍待进一步研究。

---

## 213. Beyond Translation: Cross-Cultural Meme Transcreation with Vision-Language Models

**arXiv ID:** 2602.02510 | [PDF](https://arxiv.org/pdf/2602.02510v1)

**作者:** Yuming Zhao `[一作]` (Santa Clara University), Oana Ignat `[通讯]` (Santa Clara University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究中美文化间的表情包跨文化再创作，提出三阶段混合框架，并构建双向 MemeXGen 数据集。

**💡 创新点**

创新点在于：①将意图保持与文化适配分离的混合再创作框架；②首次公开双向中美表情包对齐数据集；③提供跨文化多模态生成的评价体系。

**🔧 技术方法**

使用 LLaVA 进行文化分析与字幕生成，FLUX.1 快速生成符合目标文化的视觉模板，Pillow 完成文本叠加；评估则采用 Qwen‑VL‑Max 等开源 VLM 进行自动打分。

**📊 数据集**

采用 MemeXGen 数据集：6,315 条原始中美表情包及其跨文化再创作版，包含情绪标签和文化意图注释。

**📈 对比分析**

方法通过 3 位双语评审员和 6 个 VLM 进行对比评估，US→Chinese 的平均得分约 4.48，Chinese→US 约 3.93；整体人类平均分 4.07/5，Qwen‑VL‑Max 与人类评分相关性最高。

**⚠️ 局限性**

局限性：仅覆盖中美两国文化，方向性差异成因未完全阐明；人类评估主观性强；自动评测对多数模型表现不佳；数据集中情绪以喜悦为主，缺少多样负面情绪。

---

## 214. IceBench-S2S: A Benchmark of Deep Learning for Challenging Subseasonal-to-Seasonal Daily Arctic Sea Ice Forecasting in Deep Latent Space

**arXiv ID:** 2602.02567 | [PDF](https://arxiv.org/pdf/2602.02567v1)

**作者:** Jingyi Xu `[一作]` (Fudan University), Ben Fei `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 4251 | [OpenAlex ID](https://openalex.org/A5074448929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实现了IceBench‑S2S基准，用于评估深度学习模型在180天内对北极海冰浓度进行每日预测的能力。

**💡 创新点**

创新点在于首次构建面向S2S每日海冰预测的基准体系，并提出Sea Ice Forecasting Engine（SIFE）框架，将Swin Transformer自编码器与多种时间序列预测骨干网络统一融合，实现深度潜在空间压缩与长期序列建模。

**🔧 技术方法**

技术上采用Swin Transformer自编码器进行高维海冰网格压缩，结合Transformer、iTransformer、Informer、PatchTST、DLinear、NLinear、TimeMixer、SCInet、CycleNet等时间序列骨干；使用15天自回归滚动策略、教师强制与模型集成提升预测精度。

**📊 数据集**

使用了NSIDC公开的G02202被动微波海冰浓度（CDR）时间序列数据，涵盖1978年10月25日至2024年6月30日，共计16,686天，并按1979–2015年训练、2016–2019年验证、2020–2024年测试划分。

**📈 对比分析**

通过与持久性、气候学、SDAP、Sea Ice Outlook等传统与动态基准的对比，利用MSE、MAE、ACC、R²、NSE等指标评估性能；深度模型在短期（≤30天）内表现优异，但在180天长期滚动预测时精度下降，集成模型可显著提升ACC并在春季预测障碍上优于SIO基准。

**⚠️ 局限性**

局限性包括仅使用海冰浓度数据，未结合大气与海洋等耦合变量；在融冰季节及极端事件预测中仍表现不佳；模型训练和推理成本较高，且对不同检索算法的鲁棒性仍有待进一步验证。

---

## 215. BinaryPPO: Efficient Policy Optimization for Binary Classification

**arXiv ID:** 2602.02708 | [PDF](https://arxiv.org/pdf/2602.02708v1)

**作者:** Punya Syon Pandey `[一作]` (University of Toronto), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出BinaryPPO，一个离线强化学习框架，将二分类任务重新表述为基于置信度加权的奖励最大化问题，并使用PPO对大型语言模型进行微调。

**💡 创新点**

创新点在于设计置信度加权奖励函数、优势缩放和熵正则化来稳定策略更新，并采用等样本采样缓解类别不平衡，使模型在噪声标签环境下获得鲁棒性。

**🔧 技术方法**

主要技术包括Proximal Policy Optimization（PPO）、值网络估计、置信度加权奖励函数、熵正则化、等样本采样以及优势估计。

**📊 数据集**

实验使用八个公开二分类基准：CLadder、SciRIFF、BoolQ、FEVER、IMDB、OpenAI Moderation、Detect‑Jailbreak 和 JailbreakBench。

**📈 对比分析**

与传统监督微调（SFT）和普通PPO基线相比，BinaryPPO在所有数据集上平均提升40–60个百分点，最高准确率达到99%，且在离散分布和跨数据集迁移测试中表现更为稳健。

**⚠️ 局限性**

局限性包括仅针对二分类任务，离线RL限制了与在线交互的灵活性；实验以中小规模开源LLM为主，扩展到多类或超大模型需进一步研究；奖励设计和熵正则化可能需要任务特定调参，且规范性评估仍有不足。

---

## 216. HMVLA: Hyperbolic Multimodal Fusion for Vision-Language-Action Models

**arXiv ID:** 2602.02533 | [PDF](https://arxiv.org/pdf/2602.02533v1)

**作者:** Kun Wang `[一作]` (Harbin Institute of Technology), Tonghua Su `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种将视觉、语言和动作信息嵌入超曲率空间的VLA框架HMVLA，并通过稀疏门控的Mixture-of-Experts实现多模态语义对齐，提升机器人控制效果。

**💡 创新点**

创新点：1）首次将双曲空间（Lorentz模型）应用于VLA任务，利用其指数扩展特性天然建模视觉‑语言中的层级结构；2）在Transformer的前馈层引入soft MoE，通过动态专家路由强化跨模态细粒度语义匹配；3）在对比损失上加入 entailment 圆锥约束，进一步提升语义一致性。

**🔧 技术方法**

核心技术包括：双曲几何（Lorentz模型、指数/对数映射、圆锥约束）、CLIP对比损失、soft Mixture-of-Experts（soft‑gate、负载平衡正则化）、Transformer（自注意、跨模态注意）以及标准的梯度优化器 Adam。

**📊 数据集**

使用 LIBERO 基准（Spatial、Object、Goal、LONG 四个子数据集）以及自行构造的 Gen 泛化数据集进行训练与评估。

**📈 对比分析**

与 DP、Octo、Tra‑MoE、CoT‑VLA、Dita 等先进方法对比，HMVLA 在四个子任务的平均准确率上分别为 90%、96%、89%、69%，整体平均 86%，均高于对比方法；在 Gen 数据集上同样展现更强的跨域泛化能力。

**⚠️ 局限性**

限制：1）双曲映射和圆锥约束需要手动设定曲率与超参数，调参成本较高；2）soft‑MoE 需额外的专家网络，增加模型尺寸与计算开销；3）对非层级结构的视觉‑语言配对效果尚未充分验证，可能对某些任务产生负面影响。

---

## 217. Adaptive Linear Path Model-Based Diffusion

**arXiv ID:** 2602.02831 | [PDF](https://arxiv.org/pdf/2602.02831v1)

**作者:** Yutaka Shimizu `[一作]` (University of California), Masayoshi Tomizuka `[通讯]` (University of California)

**通讯引用:** 38492 | [OpenAlex ID](https://openalex.org/A5064077634)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出了Linear Path Model-Based Diffusion (LP-MBD) 与其自适应扩展 ALP-MBD，用流匹配启发的线性概率路径替代传统方差保持噪声调度，并通过强化学习动态调整噪声级和步数，以提升机器人轨迹优化的效率和鲁棒性。

**💡 创新点**

创新点在于：① 将方差保持调度换成可解耦、几何可解释的线性概率路径，显著降低调参难度；② 通过 RL（PPO）实现噪声级和步数的在线自适应，从而在不同环境和任务难度下动态平衡探索与收敛。

**🔧 技术方法**

使用技术包括：基于模型的扩散优化（MBD）、流匹配导向的线性路径、强化学习（PPO）进行参数调度、以及传统采样优化方法如 MPPI 与 CEM 作为对比基线。

**📊 数据集**

实验数据集包括：BraX 物理仿真平台上的多种连续控制任务（Ant、Hopper、HalfCheetah、Walker2D、Reacher、Pusher），以及自定义移动机器人轨迹跟踪仿真环境（S‑形与椭圆形轨迹）。

**📈 对比分析**

在数值一维例子、BraX 任务以及移动机器人跟踪任务中，与 VP‑MBD、CEM、MPPI 等方法比较，LP‑MBD 在大部分 BraX 任务上获得与或优于现有方法的奖励，ALP‑MBD 在移动机器人任务中取得最高奖励，并通过自适应调度实现更高的效率与鲁棒性。

**⚠️ 局限性**

主要限制包括：在高维复杂任务（如 Pusher）中，线性概率路径的表达能力可能不如方差保持调度；自适应方法在执行时会增加额外的 RL 前向推断延迟，虽然整体影响较小，但在极端实时要求下仍需关注。

---

## 218. Accelerating Structured Chain-of-Thought in Autonomous Vehicles

**arXiv ID:** 2602.02864 | [PDF](https://arxiv.org/pdf/2602.02864v1)

**作者:** Yi Gu `[一作]` (NVIDIA), Marco Pavone `[通讯]` (Stanford University)

**通讯引用:** 11212 | [OpenAlex ID](https://openalex.org/A5050003000)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种用于自动驾驶视觉‑语言‑行动模型的并行解码框架，能在保持链式思考（CoT）推理效果的同时显著加速 CoT 生成。

**💡 创新点**

创新点在于：
- 将 CoT 结构化为多字段模板并构建依赖图；
- 使用动态规划自动调度并行生成，最小化前向传递次数；
- 通过自定义注意力掩码和单序列多 token 并行，实现无额外 FLOP、共享 KV‑缓存的高效并行解码。

**🔧 技术方法**

采用技术包括：结构化 CoT 模板、依赖图与 DP 调度、并行解码、FlashAttention‑2、xFormers、Qwen 系列 LLM、Transfusion、DINOv2 视觉特征提取、BFloat16 计算。

**📊 数据集**

使用内部 20,000 小时的多城市多国驾驶数据集，并通过 Qwen2.5‑VL‑72B 自动标注生成 717k+ 训练样本与 950 条测试样本。

**📈 对比分析**

与无 CoT 与标准自回归 CoT 基线对比：在 Qwen2‑0.5B、Qwen3‑1.7B、Qwen2.5‑VL‑3B 三种模型中，CoT 生成时间提升 3.1–4.1×，整体推理时间提升 1.9–3.1×；Meta‑Action IOU 与轨迹 ADE 均保持或略优，证明并行解码不损失任务性能。

**⚠️ 局限性**

局限性：
- 加速受限于 CoT 模板的关键路径长度；
- 并行度受字段数与依赖关系限制；
- 自定义注意力掩码与多 token 并行受 GPU 核实现效率影响；
- 模板化 CoT 需人工设计，迁移到其他任务时可能需要重新定义；
- 在更大模型或更复杂任务场景中需进一步验证。

---

## 219. Scaling Small Agents Through Strategy Auctions

**arXiv ID:** 2602.02751 | [PDF](https://arxiv.org/pdf/2602.02751v1)

**作者:** Lisa Alazraki `[一作]` (Imperial College London), Akhil Mathur `[通讯]` (Meta Superintelligence Labs)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究任务复杂度对小型与大型语言模型代理性能差异，并提出基于策略竞拍的动态路由框架（sale）来在不同任务难度下高效分配模型。

**💡 创新点**

创新在于将代理的短期策略视为竞标商品，通过成本-价值评分和竞拍内存反馈实现无训练的自适应路由与小模型的自我提升。

**🔧 技术方法**

技术包括LLM策略生成、成本估算公式、熵与同伴评估的价值衡量、最小-最大优化路由、基于文本相似度的记忆检索以及多模型的分布式执行。

**📊 数据集**

数据集为 HST‑Bench，由 Deep Search（SimpleQA、PopQA、HotpotQA、GAIA、Humanity's Last Exam）和 Coding（MBPP、LeetCode）任务，按人工解题时间划分为五个复杂度级别，共753个任务。

**📈 对比分析**

与单一模型、WTP、CARROT、TO‑Router 和 FrugalGPT 等基线比较，sale 在所有复杂度级别上均达到或超过最强单模型的 pass@1，同时将成本降低 25–53%，显著扩展性能‑成本 Pareto 前沿。

**⚠️ 局限性**

限制包括仅评估两类任务、只使用 Qwen3 4B–32B 规模，记忆库线性增长、未考虑工具调用成本，以及对更大模型或不同架构的泛化需要进一步验证。

---

## 220. Discovering Data Manifold Geometry via Non-Contracting Flows

**arXiv ID:** 2602.02611 | [PDF](https://arxiv.org/pdf/2602.02611v1)

**作者:** David Vigouroux `[一作]` (IRT Saint Exupery), François Rousseau `[通讯]` (IMT Atlantique)

**通讯引用:** 9039 | [OpenAlex ID](https://openalex.org/A5045707906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无监督方法，通过学习嵌入空间中的向量场并约束其不收缩，从而在未知数据流形上构建全局坐标系。

**💡 创新点**

核心创新在于：①利用平行化假设学习全局向量场；②通过正定李导数约束防止流形坍塌；③利用向量场可交换性把多维ODE压缩为单一ODE，实现高维可扩展的流匹配训练。

**🔧 技术方法**

采用神经ODE、流匹配（flow‑matching）损失、李导数正定约束、向量场可交换性正则化、时间函数约束等技术。

**📊 数据集**

在合成流形（线性平面、球面、环面、瑞士卷、双曲抛物面）以及实际图像数据集CIFAR‑10上进行实验。

**📈 对比分析**

与传统等距自编码器等基线对比，CIFAR‑10上可获得相近分类精度（约42% vs 45%），训练时间和显存需求显著低于自编码器；在合成数据上实现零损失，验证理论。

**⚠️ 局限性**

局限性：仅适用于存在全局坐标图的平行化流形；对数据密度无显式建模；高维时需要正定李导数的近似正则化，可能仍有计算开销。

---

## 221. STEER: Inference-Time Risk Control via Constrained Quality-Diversity Search

**arXiv ID:** 2602.02862 | [PDF](https://arxiv.org/pdf/2602.02862v1)

**作者:** Eric Yang `[一作]` (Verily Life Sciences), Yugang Jia `[通讯]` (Verily Life Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出STEER框架，在不重新训练模型的情况下通过演化搜索构建多样化的语言模型人设集合，并在推理时通过百分位调节实现可控的风险阈值；

**💡 创新点**

将可控性视为受约束的质量-多样性搜索，利用演化算法填补风险谱空隙，同时保持安全、推理连贯与信号稳定；

**🔧 技术方法**

演化搜索、质量-多样性算法、最大似然估计的偏差建模、推理时百分位调节、LLM-判别器进行安全与连贯度评分；

**📊 数据集**

两组临床分诊数据：公开的MIETIC（MIMIC-IV）和私有的EHR症状分诊；

**📈 对比分析**

与高温采样、静态人设集合及后训练Spectrum Tuning对比；STEER在MIETIC和EHR上分别提升Ordinal AUC，且在极端急诊样本的安全性保持高于后训练方法；

**⚠️ 局限性**

推理时需要多模型并行，成本随团队规模线性增长；框架目前仅针对序数决策任务，尚未验证对开放式生成的适用性；

---

## 222. From Task Solving to Robust Real-World Adaptation in LLM Agents

**arXiv ID:** 2602.02760 | [PDF](https://arxiv.org/pdf/2602.02760v1)

**作者:** Pouya Pezeshkpour `[一作]` (Megagon Labs), Estevam Hruschka `[通讯]` (Megagon Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个受控的网格游戏基准，用以评估大型语言模型在部分可观测、噪声、动态环境和代理状态漂移等真实世界条件下的鲁棒性。

**💡 创新点**

通过同时激活四类真实世界不确定因素（部分可观测、动态、噪声、状态漂移）构造统一可调节基准，并通过单一压力因子分析揭示模型对不同不确定性的敏感性和策略差异。

**🔧 技术方法**

采用基于文本接口的LLM代理，结合行动频率分析、单一压力因子切除实验以及基于特征的逻辑回归因子归因等技术。

**📊 数据集**

使用自定义生成的网格游戏实例（N×N尺寸为6×6、8×8、10×10），随机种子生成地图、障碍、关键物品及噪声，未使用公开数据集。

**📈 对比分析**

通过五个最先进LLM（GPT‑5.2、GPT‑5 mini、Gemini‑3 Pro/Flash、Qwen3）在全压迫条件下的成功率、平均分数和步数进行对比，结果显示不同模型在不同网格尺寸和不确定性模式下排名不稳，整体性能随规模和不确定性而下降。

**⚠️ 局限性**

受限于单一二维网格环境的简化、缺乏真实物理感知和多模态交互，以及未对模型进行针对多目标优化，导致实验结果难以直接推广到更复杂的真实部署场景。

---

## 223. Mitigating Task-Order Sensitivity and Forgetting via Hierarchical Second-Order Consolidation

**arXiv ID:** 2602.02568 | [PDF](https://arxiv.org/pdf/2602.02568v1)

**作者:** Protik Nag `[一作]` (University of South Carolina), Vignesh Narayanan `[通讯]` (University of South Carolina)

**通讯引用:** 755 | [OpenAlex ID](https://openalex.org/A5019813462)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种层次化二阶泰勒展开连续学习框架（HTCL），通过在任务组内进行全排列搜索获得最优本地学习，再用二阶泰勒近似与Hessian正则化实现全局层次整合，从而显著降低任务顺序对性能的影响。

**💡 创新点**

创新点包括：①利用任务组内全排列来逼近全局最优顺序，显著降低任务序列的组合复杂度；②设计基于二阶泰勒展开的Hessian正则化更新，实现对先前任务损失的精确保持；③引入多层级结构，使得不同时间尺度的知识能分层整合，进一步提升长期记忆与鲁棒性；④方法模型无关，可与任何现有CL算法配合。

**🔧 技术方法**

核心技术：二阶泰勒展开 + Hessian正则化；低秩曲率近似（如对角或稀疏近似）；任务分组与组内全排列搜索；多层级权重更新；抓取阶段（catch‑up）细化整合；模型-agnostic 层次化学习管道。

**📊 数据集**

实验数据集：SplitMNIST（5个二分类任务），SplitCIFAR‑100（10个10分类任务），SplitCora（图节点分类），20 Newsgroups（文本分类），所有实验均在领域增量（domain‑incremental）设置下完成。

**📈 对比分析**

对比方法：Strong Experience Replay (SER)、Dark Experience Replay (DER)、Experience Replay (ER)、DualNet、iCaRL、EWC、Spectral Regularizer (SR)。HTCL 在所有基线上平均提升 7%–25% 的准确率，标准差降低 33%–68%，遗忘率降低 10%–70% 以上，显著提高了任务顺序鲁棒性。

**⚠️ 局限性**

局限性：相对基线，计算时间提升约 4–5 倍；需要调节组大小与层级深度；二阶近似在极大模型或极长任务序列时仍可能产生误差；实现复杂度高，需额外维护Hessian近似和多层级参数；对极高维度数据的可扩展性待进一步验证。

---

## 224. End-to-end reconstruction of OCT optical properties and speckle-reduced structural intensity via physics-based learning

**arXiv ID:** 2602.02721 | [PDF](https://arxiv.org/pdf/2602.02721v1)

**作者:** Jinglun Yu `[一作]` (Johns Hopkins University), Jin U. Kang `[通讯]` (Johns Hopkins University)

**通讯引用:** 29448 | [OpenAlex ID](https://openalex.org/A5065423939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发一种物理正则化的端到端深度学习框架，联合恢复OCT结构图像和折射率、散射系数、各向异性三种光学参数，并在网络中嵌入可微前向模型约束以抑制散射噪声。

**💡 创新点**

在网络中引入可微的Extended Huygens–Fresnel前向模型进行一致性约束；使用多分支U‑Net独立预测每个参数；结合扩散模型（EDM）score正则化作为统计先验；同时输出去噪的结构强度，解决参数耦合、衰减和噪声放大问题。

**🔧 技术方法**

技术包括U‑Net多通道编码器–解码器、可微光学前向模型、总变差（TV）正则化、扩散模型score正则化、MSE/forward consistency/Tv/Diffusion损失融合以及Adam优化器。

**📊 数据集**

使用基于Monte Carlo（MCML）模拟的500多张角膜OCT B‑scan图像（1024×1024），每张图像配有对应的折射率、散射系数和各向异性真值。

**📈 对比分析**

与基线U‑Net以及去除diffusion、physics或TV等正则化的版本进行对比，评估PSNR、SSIM和MSE。全模型在结构强度上的PSNR 31.65、SSIM 0.94、MSE 2.75e-3，明显优于去除正则化或基线模型，证明物理约束和扩散先验能显著提升恢复质量。

**⚠️ 局限性**

局限在于仅在合成角膜数据上验证，真实数据的泛化性能未知；折射率变化范围小导致对n的误差不敏感；训练依赖大量Monte Carlo模拟；模型结构复杂，计算成本高；未考虑多视角或三维重建。

---

## 225. Error Analysis of Matrix Multiplication Emulation Using Ozaki-II Scheme

**arXiv ID:** 2602.02549 | [PDF](https://arxiv.org/pdf/2602.02549v1)

**作者:** Yuki Uchino `[一作]` (RIKEN Center for Computational Science), Toshiyuki Imamura `[通讯]` (RIKEN Center for Computational Science)

**通讯引用:** 942 | [OpenAlex ID](https://openalex.org/A5086152822)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文通过对Ozaki-II方案进行严格的确定性误差分析，证明了在使用INT8矩阵引擎进行单精度和双精度矩阵乘法仿真时误差随模数个数和输入矩阵指数分布的变化而变化，并给出了误差上界。

**💡 创新点**

创新点在于：①首次给出Ozaki-II方案的完整误差上界；②通过分析指数分布和模数数量对误差的影响，提出了自动调节模数个数以满足预期精度的理论依据；③提出了两种误差估计方法，一种更保守但易计算，另一种更紧但需要额外计算，展示了误差估计的可调性。

**🔧 技术方法**

主要技术包括：使用Chinese Remainder Theorem（CRT）将高精度乘法拆解为多次低精度（INT8）乘法；对输入矩阵做对数尺度的对齐与截断；利用高精度（双精度/双倍精度）来完成CRT的求逆和余数计算；对浮点运算误差进行系统的界定与叠加分析。

**📊 数据集**

实验数据集为人工生成的随机矩阵，大小为128×8192，元素按 (rand‑0.5)·exp(randn·ϕ) 生成，ϕ 控制动态范围，覆盖不同指数分布情形。

**📈 对比分析**

比较方法：将Ozaki-II方案在NVIDIA RTX 4090 GPU 上的误差（|AB‑C|）与理论误差上界进行对比，并与原生DGEMM/SGEMM误差做对照。结果显示，误差始终落在理论上界以内，且Ozaki-II方案在吞吐量上优于原生实现，尤其在低精度硬件加速环境下。

**⚠️ 局限性**

局限性：①误差上界包含 |A'B'| 项，计算成本较高；②保守误差估计（第二种上界）虽然易计算但过于保守；③方案依赖于模数不超过49个且 pℓ ≤256 的限制；④对输入矩阵行/列全为零的情况需排除；⑤实现复杂度高，需要预先构造模数与相关常数表。

---

## 226. Equal Access, Unequal Interaction: A Counterfactual Audit of LLM Fairness

**arXiv ID:** 2602.02932 | [PDF](https://arxiv.org/pdf/2602.02932v1)

**作者:** Alireza Amiri-Margavi `[一作]` (University of Pittsburgh), Hamidreza Hasani Balyani `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在职业建议场景下，使用对照式身份注入对话质量进行公平性审计。

**💡 创新点**

提出仅通过交互质量（情感、礼貌、犹豫）而非拒绝率来评估公平性的对照式评估框架。

**🔧 技术方法**

使用对照式 prompt、自动化情感/礼貌/犹豫词典计数、Wilcoxon 符号秩检验及 Cohen d 效应量。

**📊 数据集**

30 条中性职业建议 prompt，配合 8 种身份组合（年龄、性别、国籍），共 240 条测试实例。

**📈 对比分析**

通过配对统计比较 GPT‑4 与 LLaMA 的交互质量差异，发现两模型存在中等效应量的身份差异（如 GPT‑4 年轻男性犹豫更高，LLaMA 移民女性情感更低）。

**⚠️ 局限性**

仅限英文职业建议、自动化指标可能无法完全映射人类感知，样本规模有限，未做跨领域或多语言验证。

---

## 227. Community Norms in the Spotlight: Enabling Task-Agnostic Unsupervised Pre-Training to Benefit Online Social Media

**arXiv ID:** 2602.02525 | [PDF](https://arxiv.org/pdf/2602.02525v1)

**作者:** Liam Hebert `[一作]` (University of Waterloo), Robin Cohen `[通讯]` (University of Waterloo)

**通讯引用:** 11413 | [OpenAlex ID](https://openalex.org/A5000636604)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种基于讨论Transformer的无监督预训练框架，利用社区规范学习来提升对在线讨论的理解与生成。

**💡 创新点**

创新点在于：①将预训练任务明确聚焦于社区规范（结构性和语义性）而非任务特定标注；②结合生成式边级/节点级任务与对比式分支/社区对齐任务，形成局部与全局规范双重学习；③将预训练结果与可解释性技术相结合，挖掘模型对社区规范的内在表示。

**🔧 技术方法**

采用的技术包括：讨论Transformer（Graph Transformer）、掩码生成任务（Edge-level “是否回复”与Node-level “评论重构”）、对比式预训练（InfoNCE）中的分支采样与社区对齐、UMAP可视化、可解释性方法（最近邻原型、机制性解释、稀疏自动编码器）。

**📊 数据集**

主要使用了Reddit的未标注讨论树数据；实验中选取了聚类后的“Politics”“Age”“Gender”等社区，分别采样8000条高投票讨论；对比实验使用了mDT Discussion Transformer在社区对齐任务上的预训练。

**📈 对比分析**

通过对比实验显示，社区对齐预训练后讨论嵌入空间出现明显的聚类（如政治立场与年龄差异），说明模型学到了社区规范。与仅用有标签微调的模型相比，预训练后在下游任务（如仇恨言论检测）上可获得更高的准确率与鲁棒性，且训练样本需求显著降低。

**⚠️ 局限性**

局限性包括：①生成式重构任务对高信息量文本难度大，损失设计与模型容量需进一步优化；②对比任务易受负迁移影响，需要更精细的正负样本筛选；③实验规模有限，缺乏跨平台与跨语言验证；④解释性方法仍处于探索阶段，尚未完全证实其可用于实际平台治理。

---

## 228. AdaptMMBench: Benchmarking Adaptive Multimodal Reasoning for Mode Selection and Reasoning Process

**arXiv ID:** 2602.02676 | [PDF](https://arxiv.org/pdf/2602.02676v1)

**作者:** Xintong Zhang `[一作]` (Beijing Institute of Technology), Qing Li `[通讯]` (BIGAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AdaptMMBench 这一评估自适应多模态推理的 benchmark，并对多种视觉‑语言模型进行系统评测。

**💡 创新点**

创新点在于：①基于模型性能动态划分工具冗余/必需的难度标签并使用 Matthews Correlation Coefficient（MCC）评估模式选择；②给出关键步骤覆盖、工具有效性和推理效率三维过程度量；③揭示自适应推理与 oracle‑visual 之间的显著性能差距，指出工具调用是主要瓶颈。

**🔧 技术方法**

采用工具增强视觉推理（如缩放、旋转、对比度调节）、函数调用式工具接口、GPT‑5 进行关键推理步骤标注，并通过 MCC、覆盖率、工具有效率等指标量化模型能力。

**📊 数据集**

构建包含 1420 条样本的多领域数据集，涵盖 Real‑world、OCR、GUI、Knowledge、Math 五个域，样本包括可直接文本推理的任务和需主动工具调用的任务，并标注视觉工具参数与关键推理步骤。

**📈 对比分析**

对比了多款开源模型（Qwen3‑VL 8B/32B/235B、DeepEyes、PixelReasoner、PyVision、AdaptVision）与闭源模型（GPT‑5、Gemini‑3‑Pro）在文本推理、自适应推理和 oracle‑visual 模式下的准确率、MCC、关键步骤覆盖率、工具有效率和效率。结果显示：模型规模越大，自适应模式选择（MCC）越好；自适应推理能显著提升准确率，但仍与 oracle‑visual 之间相差 5–15%，表明工具调用仍是性能瓶颈。

**⚠️ 局限性**

局限性包括：①过程质量评估仅适用于可公开访问推理轨迹的开源模型；②工具调用效果仍不稳定，误用或冗余调用导致误差；③缺乏视觉生成能力，无法处理自生成辅助图像的任务；④关键步骤标注依赖 GPT‑5，存在主观性；⑤模型依赖的难度划分可能导致评估偏差。

---

## 229. How Does the Lagrangian Guide Safe Reinforcement Learning through Diffusion Models?

**arXiv ID:** 2602.02924 | [PDF](https://arxiv.org/pdf/2602.02924v1)

**作者:** Xiaoyuan Cheng `[一作]` (University College London), Yukun Hu `[通讯]` (University College London)

**通讯引用:** 1969 | [OpenAlex ID](https://openalex.org/A5086866094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于增广拉格朗日引导的扩散策略（ALGD），实现在线安全强化学习。

**💡 创新点**

创新点在于将拉格朗日函数视为扩散能量，并通过增广拉格朗日局部凸化能量曲面来稳定分数场和双变量更新，使得多模态扩散策略既能最大化奖励又能满足安全约束。

**🔧 技术方法**

使用扩散模型、增广拉格朗日理论、分数匹配、成本评估集成以及蒙特卡洛分数估计等技术。

**📊 数据集**

在Safety‑Gym和MuJoCo的速度约束任务上进行实验。

**📈 对比分析**

与原始拉格朗日、增广拉格朗日、PPO+Lag、SAC+Lag、硬约束方法等基准进行比较，ALGD在保持或提升奖励的同时显著降低约束违规，训练更稳定，样本效率更高。

**⚠️ 局限性**

主要局限包括需要额外的蒙特卡洛采样和成本评估集成导致的计算开销；对成本函数估计高度依赖，若估计不准仍可能引发不稳定；在极高维或更复杂约束场景下的可扩展性尚待验证。

---

## 230. Fubini Study geometry of representation drift in high dimensional data

**arXiv ID:** 2602.02596 | [PDF](https://arxiv.org/pdf/2602.02596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 231. Latent Perspective-Taking via a Schrödinger Bridge in Influence-Augmented Local Models

**arXiv ID:** 2602.02857 | [PDF](https://arxiv.org/pdf/2602.02857v1)

**作者:** Kevin Alcedo `[一作]` (University of Lisbon), Rachid Alami `[通讯]` (LAAS-CNRS)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种结合影响增强局部模型和神经符号世界模型的框架，利用可化的 Schrödinger 桥实现视角转移，从而在多智能体强化学习中学习社会意识策略。

**💡 创新点**

创新点在于把视角转换建模为贝叶斯推理中的 Schrödinger 桥，并在影响增强局部模型上学习离散因子化的神经符号世界模型；同时使用可化的 Schrödinger 桥进行在线决策时间的视角转移。

**🔧 技术方法**

所用技术包括影响增强局部模型（I‑ALM）、双向 GRU+α‑entmax 的潜在空间贝叶斯网络、Gumbel‑Softmax 变分自编码器、Schrödinger 桥的空间‑时间 Sinkhorn 方法和 Doob h‑transform。

**📊 数据集**

实验数据集为部分可观的 MiniGrid 社会导航任务（离散观测与动作的网格环境）。

**📈 对比分析**

与完美信息、无信息和基于参考动态的基线相比，基于 Schrödinger 桥的视角转移在学习速度和累计奖励上均显著优于其他方法。

**⚠️ 局限性**

局限性包括假设其它智能体与自身同质化的感知与动作模型，仅在离散小规模网格环境中验证，未来需要处理异质性并扩展至更复杂场景。

---

## 232. Causality--Δ: Jacobian-Based Dependency Analysis in Flow Matching Models

**arXiv ID:** 2602.02793 | [PDF](https://arxiv.org/pdf/2602.02793v1)

**作者:** Reza Rezvan `[一作]` (Chalmers University of Technology), Richard Torkar `[通讯]` (University of Gothenburg)

**通讯引用:** 4278 | [OpenAlex ID](https://openalex.org/A5030588511)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了流匹配模型中小的潜在扰动如何传播，并利用雅可比向量积（JVP）估计生成特征的依赖结构，推导出高斯及高斯混合的闭式漂移和雅可比，验证了在低维仿真、MNIST像素以及CelebA属性上的有效性。

**💡 创新点**

创新点在于首次将因果视角与流匹配结合，揭示局部仿射结构并通过JVP估计依赖关系，提出用JVP阈值过滤实现近似干预来探索共同原因关系，从而提升模型可解释性。

**🔧 技术方法**

使用的技术包括流匹配框架、解析闭式漂移与雅可比、Jacobian‑vector product 计算、有限差分与数值JVP对比、与预训练分类器组合以估计属性级相关性、以及随机采样与统计检验。

**📊 数据集**

实验使用了低维高斯/高斯混合仿真数据、MNIST手写数字像素、以及包含40个二元属性的CelebA人脸图像数据集。

**📈 对比分析**

通过与理论雅可比、数值JVP、真实相关系数对比，结果显示在低维案例误差极小；MNIST像素相关系数与真实值相近；CelebA属性相关性在10k样本下即可部分恢复10万样本的相关度，且在条件小JVP时相关几乎降至零，验证了方法的有效性。

**⚠️ 局限性**

局限性包括仅在高斯或高斯混合假设下推导闭式解；JVP阈值经验性且缺乏敏感性分析；依赖预训练分类器可能引入偏差；JVP采样成本高；无法实现正式的do‑干预，需要进一步研究以实现真正的结构干预。

---

## 233. Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval

**arXiv ID:** 2602.02827 | [PDF](https://arxiv.org/pdf/2602.02827v1)

**作者:** Roi Pony `[一作]` (IBM Research Israel), Udi Barzelay `[通讯]` (IBM Research Israel)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在查询时对 ColBERT 等多向量晚期交互检索器进行自适应剪枝的算法，称为 Col-Bandit；

**💡 创新点**

创新点在于把晚期交互重排序视为有限样本 Top‑K 识别问题，利用 Serfling 相关不确定性界和 LUCB 框架动态揭示最需要的 MaxSim 计算，且只需一个可调校的松弛参数即可在保持高排名精度的同时显著降低计算量；

**🔧 技术方法**

核心技术包括：多向量 late‑interaction（ColBERT、Jina‑ColBERTv2、Granite Vision Embedding）、基于 Serfling 的方差自适应置信区间、LUCB 采样策略以及 ϵ‑greedy 探索；

**📊 数据集**

在文本检索基准 BEIR（如 MSMarco、NQ 等）和多模态检索基准 REAL‑MM‑RAG 上进行评估；

**📈 对比分析**

与随机揭示（Doc‑Uniform）和贪婪最大宽度揭示（Doc‑TopMargin）对比，Col‑Bandit 在相同覆盖率下能获得更高的 Overlap@K（如 90% 重叠只需 20‑30% 的 MaxSim 计算），在 40% 覆盖率时基本保持与完整计算相同的 Recall@5 / nDCG@5，且总体可实现约 5× 的 FLOPs 降低；

**⚠️ 局限性**

局限性包括：主要针对小 K 的高精度检索任务，K 越大时边界聚集导致剪枝收益下降；目前仅在 FLOPs 上评估，未给出严格的理论保证，实际推理加速需进一步的批量实现与 GPU 并行化。

---

## 234. Learning Consistent Causal Abstraction Networks

**arXiv ID:** 2602.02623 | [PDF](https://arxiv.org/pdf/2602.02623v1)

**作者:** Gabriele D'Acunto `[一作]` (Sapienza University), Sergio Barbarossa `[通讯]` (Sapienza University)

**通讯引用:** 15450 | [OpenAlex ID](https://openalex.org/A5084159640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

学习并构造一致的因果抽象网络（CAN），实现不同层级高斯结构因果模型（SCM）之间的可解释性与一致性；

**💡 创新点**

提出一种基于sheaf理论、利用CLCA（构造性线性因果抽象）和SEP（语义嵌入原则）的CAN框架，并给出避免非凸目标的局部学习方法与闭式迭代更新；

**🔧 技术方法**

采用Stiefel流形优化、ADMM分裂方法、Riemannian梯度求解、以及基于拓扑闭包的搜索策略；

**📊 数据集**

使用人工生成的多维高斯数据，构造链、星、树三种网络拓扑，随机生成对应的高低层次SCM与CLCA；

**📈 对比分析**

与LinSEPAL系列基线方法对比；在正定协方差下，闭式方法在Frobenius误差和结构一致性上与基线持平；在正半定协方差下，搜索方法实现了高真阳性率而真阴性率接近零；

**⚠️ 局限性**

仅在合成场景验证；需已知CA结构先验；非凸性仍导致对初始化敏感；对真实世界数据与反事实一致性尚未验证；

---

## 235. Kino-PAX$^+$: Near-Optimal Massively Parallel Kinodynamic Sampling-based Motion Planner

**arXiv ID:** 2602.02846 | [PDF](https://arxiv.org/pdf/2602.02846v1)

**作者:** Nicolas Perrault `[一作]` (University of Colorado Boulder), Morteza Lahijanian `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1670 | [OpenAlex ID](https://openalex.org/A5069564559)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了 Kino-PAX+，一种针对高维动力学约束的并行采样运动规划器，能够在 GPU 上实现快速、近最优路径规划。

**💡 创新点**

创新点在于：①将传统序列化的采样规划过程拆解为三大并行子过程（节点扩展、选择与修剪），从而充分利用 GPU 大规模并行；②采用局部区域基准的成本筛选策略（SST 思路），实现对有利节点的聚焦，保证渐进近最优性；③在保证 δ-鲁棒完整性的同时提供渐进 δ-鲁棒近最优性证明。

**🔧 技术方法**

核心技术包括：GPU CUDA 并行采样与拓展、空间分区（超立方体分解）与区域成本维护、基于成本阈值的节点活跃/非活跃/终端状态管理、原子操作保证并行一致性。

**📊 数据集**

实验使用四个 3D 环境（森林、窄通道、建筑、锯齿）以及三种动力学模型：6D 双积分器、6D Dubins 航空机、12D 非线性四旋翼。

**📈 对比分析**

与两种基线比较：GPU 版 SBMP（只寻找可行解）和序列化 SST（近最优）。结果显示：Kino-PAX+ 在所有环境中都能在毫秒级别给出首个解，速度比 SST 高 3~4 位数（如 650×~750×），首解成本约为 SST 的 60%–70%，最终解成本进一步下降至 50%–55% 甚至低于 GPU 版 SBMP。成功率在 100% 与 0% 的差异上也明显优于 SST。

**⚠️ 局限性**

局限性包括：①参数 δ 的选择对收敛速度与解质量有显著影响，需要经验或自适应策略；②对极其高维、非线性系统的扩展仍受 GPU 记忆与轨迹积分精度限制；③理论证明基于理想化的 Lipschitz 与覆盖球假设，实际系统中常需经验调优；④在极端障碍密集或极长时间规划中仍需更大树规模，可能导致显存不足。

---

## 236. Artificial Intelligence for Inclusive Engineering Education: Advancing Equality, Diversity, and Ethical Leadership

**arXiv ID:** 2602.02520 | [PDF](https://arxiv.org/pdf/2602.02520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 237. Causal Graph Spatial-Temporal Autoencoder for Reliable and Interpretable Process Monitoring

**arXiv ID:** 2602.03004 | [PDF](https://arxiv.org/pdf/2602.03004v1)

**作者:** Xiangrui Zhang `[一作]` (China University of Mining and Technology), Furong Gao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18398 | [OpenAlex ID](https://openalex.org/A5002154840)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种因果图空间时间自编码器（CGSTAE），通过自注意力机制学习动态相关图，再利用逆向因果不变性原理提取因果图，并用图卷积长短期记忆网络重构时间序列，实现可靠且可解释的过程监测。

**💡 创新点**

创新点在于首次将逆向因果不变性视角与三步因果图学习算法结合，能够从观测的相关图中自动发现符合工艺机制的因果图，并将该因果图嵌入自编码器进行空间时间建模，从而显著提升监测可靠性和可解释性。

**🔧 技术方法**

采用的技术包括空间自注意力模块（SSAM）用于学习动态相关图、图卷积长短期记忆网络（GCLSTM）构建空间时间编码器-解码器、三步因果图学习算法（预训练、因果图学习、微调）以及对损失函数的多项正则化（不变性、先验知识、稀疏性、离散化），并使用Hotelling’s T²与SPE统计量结合核密度估计设定报警阈值。

**📊 数据集**

实验使用了标准的Tennessee Eastman过程数据集（41个变量，21种故障）和真实工艺数据——南京钢铁公司氩分离系统（13个变量，3个氮堵塞故障），通过滑动窗口重新组织数据进行训练与测试。

**📈 对比分析**

与AE、LSTM‑AE、GAE‑I、GAE‑II、DGSTAE、KDGCN、KG‑GCBiGCN等基线方法比较，CGSTAE在TEP的F1得分最高（0.896），在ASP的F1得分最高（0.820），同时保持了较低的误报率和较高的检出率，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：对足够的分布变化和过程知识先验的依赖；对超参数和训练稳定性的敏感；未考虑多采样率或过程漂移等实际场景；以及因果图学习在循环结构或强循环依赖下可能受限。

---

## 238. DualMind: Towards Understanding Cognitive-Affective Cascades in Public Opinion Dissemination via Multi-Agent Simulation

**arXiv ID:** 2602.02534 | [PDF](https://arxiv.org/pdf/2602.02534v1)

**作者:** Enhao Huang `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35125 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了DualMind，一个基于大语言模型的多智能体平台，用于模拟公共关系危机中的认知-情感意见传播

**💡 创新点**

创新点在于将慢速认知状态和快速情感状态双重潜在向量融合进智能体，并设计了PAACM决策模型与谱半衰系数控制传播爆发

**🔧 技术方法**

采用LLM（包含多模型策略）、LangChain、LangGraph、FastAPI、React前端，以及注意力+记忆检索机制和基于张量的传播概率计算

**📊 数据集**

使用15个2024‑2025年真实危机案例（美国、中华人民共和国内、欧洲），从公开社交媒体数据构建事件时间线和网络

**📈 对比分析**

通过与LAID、LPOD、LLM-GA三种SOTA基线对比，使用Pearson相关系数（平均≈0.78）和Jensen‑Shannon Divergence（平均≈0.27）衡量过程与结果一致性，DualMind显著优于所有基线

**⚠️ 局限性**

局限性包括对智能体认知的简化建模、对平台算法的抽象化处理以及网络规模与真实社交网络的差距

---

## 239. Efficient Edge Rewiring Strategies for Enhancing PageRank Fairness

**arXiv ID:** 2602.02512 | [PDF](https://arxiv.org/pdf/2602.02512v1)

**作者:** Changan Liu `[一作]` (Fudan University), Zhongzhi Zhang `[通讯]` (Fudan University)

**通讯引用:** 4843 | [OpenAlex ID](https://openalex.org/A5067533846)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究并提升社交网络中基于 PageRank 的公平性，提出通过修改网络结构（边重连）最大化弱势群体的 PageRank 权重分配。

**💡 创新点**

创新点包括：① 将边重连视为预处理操作并给出精确的单次重连对 PageRank 与 Personalized PageRank 公平度的解析影响；② 设计两种贪婪求解方案，其中 Exact 为全搜索贪婪算法，Fast 利用生成树采样（扩展 Wilson 算法）将计算复杂度降至线性并保留高精度；③ 通过森林矩阵与 PageRank 的对应关系，构造高效的采样估计。

**🔧 技术方法**

核心技术：随机游走与循环消除（Wilson 算法）用于采样指向根的随机森林；基于森林矩阵的 PageRank 解析表达式；贪婪选择与线性时间更新；误差控制（Hoeffding 估计）和稀疏化候选集（只考虑 η 值最高的顶点）。

**📊 数据集**

实验数据集包括：Books、Blogs、DBLP‑Pub、DBLP‑Gender、DBLP‑Aminer（约 423k 节点）和 LinkedIn（约 3.2M 节点）六个真实社交网络，弱势群体分别为低度可见书籍、低可见博客、后期作者、少数族裔作者、合作稀少作者及少连接的职业档案。

**📈 对比分析**

与基准方法（随机重连、MFREC、MPREC、RBL）进行对比。Fast 在 50 次重连后显著提升 PageRank 公平度与 PPR 公平度（Wasserstein 距离显著下降），并且在大规模图上运行时间仅数百秒，远快于 Exact 与 MFREC，能够处理数百万节点网络。

**⚠️ 局限性**

局限性：仅考虑两组（弱势与优势）且仅采用边重连操作；对多组、多弱势群体的场景、动态网络的适应性未讨论；Fast 的采样误差受 ψ 控制，需额外计算以保证高精度；算法对参数 α 的敏感性与理论最优性尚未完全分析。

---

## 240. Learning to Repair Lean Proofs from Compiler Feedback

**arXiv ID:** 2602.02990 | [PDF](https://arxiv.org/pdf/2602.02990v1)

**作者:** Evan Wang `[一作]`, Vasily Ilin `[通讯]` (University of Washington)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5009909631)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 APRIL 数据集，支持在 Lean 编译器反馈下进行证明修复与诊断，并对大型语言模型进行监督微调，显著提升单次修复成功率。

**💡 创新点**

创新点在于：① 将错误证明与编译器诊断、自然语言解释对齐；② 通过系统化变异生成 260k 规模的错误-修复对；③ 将诊断信息作为监督信号，提升模型的反馈感知和定向修复能力。

**🔧 技术方法**

技术手段包括：Lean 编译器交互获取错误信息、LLM 生成诊断与修复建议、LoRA 微调、聊天式提示模板、单轮（no-search）评估。

**📊 数据集**

数据集来源：Herald、Lean Workbook、NuminaMath‑Lean 原始正确证明，经过四类变异（定理替换、策略替换、单行/多行修改）生成错误证明，构成 260,125 条错误–修复对。

**📈 对比分析**

与基线比较：未微调 Qwen3‑4B 仅 1.1% 修复率；微调后提升至 27.4%；同样规模的 Goedel‑Prover‑V2‑32B 在同一单轮评估中为 26.8%；4B 微调模型在单次修复任务上已超过 32B 开源模型，表明错误导向监督对小模型尤为有效。

**⚠️ 局限性**

局限性：① 仅评估单轮无搜索修复，未体现迭代改进；② 变异生成的错误是合成的，可能与真实工程中出现的错误分布不完全一致；③ 仅覆盖 Lean 4 环境，缺乏跨助手的通用性验证。

---

## 241. SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models

**arXiv ID:** 2602.02544 | [PDF](https://arxiv.org/pdf/2602.02544v1)

**作者:** Wenhao Sun `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98285 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对扩散语言模型（DLM）在推理时无法使用传统KV缓存，提出了SPA-Cache框架，联合优化更新识别与预算分配，实现稀疏重计算。

**💡 创新点**

创新点在于：①使用低维奇异代理（singular proxy）快速定位需要更新的token，显著降低识别开销；②引入层级自适应预算分配，针对隐藏状态动态差异集中分配计算资源，提升吞吐量。

**🔧 技术方法**

核心技术包括：奇异值分解（SVD）构建低维代理；余弦相似度判定token漂移；自适应高斯型预算函数；多层Transformer的稀疏Attention与FFN重计算。

**📊 数据集**

在七个多样化基准上验证：GSM8K、MATH500、GPQA、BBH、MMLU-pro、MBPP、HumanEval，使用LLaDA-8B和Dream-7B两大DLM模型。

**📈 对比分析**

与无缓存推理、dLLM-Cache、Fast-dLLM比较，SPA-Cache在吞吐量上实现最高8×提升，MMLU-pro和MBPP等任务达到6–8倍速度提升，生成质量基本保持不变；与并行解码结合可达28×加速。

**⚠️ 局限性**

局限性包括：对温度较高的采样（τ>1.0）易导致精度下降；在分布式或张量并行部署时实现复杂；仅优化推理阶段，预填充延迟（TTFT）未显著改善；对超长序列的性能仍待评估。

---

## 242. The "Robert Boulton" Singularity: Semantic Tunneling and Manifold Unfolding in Recursive AI

**arXiv ID:** 2602.02526 | [PDF](https://arxiv.org/pdf/2602.02526v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 243. ClinConNet: A Blockchain-based Dynamic Consent Management Platform for Clinical Research

**arXiv ID:** 2602.02610 | [PDF](https://arxiv.org/pdf/2602.02610v1)

**作者:** Montassar Naghmouchi `[一作]` (Institut Polytechnique de Paris), Maryline Laurent `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 2933 | [OpenAlex ID](https://openalex.org/A5044148377)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并实现了基于 SSI、区块链和动态同意模型的临床同意管理平台 ClinConNet。

**💡 创新点**

创新点包括以参与者为中心的自主身份管理、智能合约驱动的动态同意、完整的可遗忘与可撤销功能，并在同意存证上实现完全 unlinkability。

**🔧 技术方法**

采用 Hyperledger Fabric 许可链、Veramo SSI 钱包、DID‑Auth/DID‑Comm、智能合约及 MERN 前端。

**📊 数据集**

实验使用自建 PoC 环境生成的模拟数据，无公开医疗数据集。

**📈 对比分析**

与现有区块链动态同意方案对比，平均端到端同意建立时间约 200 ms，交易吞吐量达 250 TPS，证明在可接受范围内。

**⚠️ 局限性**

局限包括 Web 门户仍为中心化组件、对真实多机构环境的可扩展性验证不足以及缺乏与欧洲数字身份钱包的完整集成。

---

## 244. Fine-Tuning Language Models to Know What They Know

**arXiv ID:** 2602.02605 | [PDF](https://arxiv.org/pdf/2602.02605v1)

**作者:** Sangjun Park `[一作]` (University of Texas at Austin), Risto Miikkulainen `[通讯]` (Cognizant AI Labs)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过双重提问法测评并用进化策略对齐，提升LLM自知晓能力。

**💡 创新点**

提出基于d_type2'的元认知评估框架和ESMA方法，将内部知识与输出行为绑定。

**🔧 技术方法**

采用双提示测评、Gaussian噪声扰动、联合奖励的进化策略优化。

**📊 数据集**

以TriviaQA作为训练与基准集，使用FictionalQA验证新信息场景。

**📈 对比分析**

与基线模型相比，ESMA在多尺寸、多语言和未见提示下显著提升d'_type2（最高1.02）及原始对齐率，改进集中于少量参数。

**⚠️ 局限性**

元认知水平仍远低于人类，模型内部机制解释有限，进化策略计算成本高。

---

## 245. Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation

**arXiv ID:** 2602.02994 | [PDF](https://arxiv.org/pdf/2602.02994v1)

**作者:** Jiaze Li `[一作]` (MiLM Plus, Xiaomi Inc.), Jian Luan `[通讯]` (MiLM Plus, Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Video-OPD框架，在多模态大型语言模型上进行Temporal Video Grounding的后训练，采用on‑policy distillation；

**💡 创新点**

创新点是使用前沿教师通过逆KL提供密集的token‑级监督，解决GRPO稀疏奖励和多回放带来的样本效率和计算开销问题，并引入Teacher‑Validated Disagreement Focusing（TVDF）训练课程提升效率；

**🔧 技术方法**

使用on‑policy distillation、逆KL奖励、策略梯度优化、教师‑学生知识蒸馏及强化学习技术；

**📊 数据集**

使用TimeLens‑100K、Charades‑STA、ActivityNet、QVHighlights等视频文本对齐数据集，以及公开的HiREST、QuerYD、HowTo‑Interlink7M、VTimeLLM、DiDeMo等视频数据；

**📈 对比分析**

与GRPO、OP‑FKD、OP‑RKD等方法对比，Video‑OPD平均提升约17%，TVDF额外提升约2%，收敛速度更快，训练成本仅为GRPO的20%；

**⚠️ 局限性**

需要依赖高容量教师模型，若教师不可用或质量不佳将影响性能。

---

## 246. Thinking inside the Convolution for Image Inpainting: Reconstructing Texture via Structure under Global and Local Side

**arXiv ID:** 2602.03013 | [PDF](https://arxiv.org/pdf/2602.03013v1)

**作者:** Haipeng Liu `[一作]` (Hefei University of Technology), Meng Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 41675 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在卷积下采样过程中利用结构特征图重建纹理特征图，实现了更完整的特征保留，从而提升图像修复质量。

**💡 创新点**

创新点在于提出双侧（全局与局部）结构与纹理特征互相重构的策略，并引入全局/局部归一化、去归一化以及跨层平衡模块，使得结构信息能够有效指导纹理恢复，并缓解特征丢失。

**🔧 技术方法**

采用了偏置卷积 (partial convolution)、视觉 Transformer (ViT)、统计归一化/去归一化、特征等化、交叉层平衡模块等技术，构建了完整的编码器-解码器架构。

**📊 数据集**

在 Paris StreetView、CelebA 以及 Places2 三个公开数据集上进行实验，并在不同遮挡比例和高分辨率（512×512）场景下验证性能。

**📈 对比分析**

与多种最新方法（PENNet、HiFill、CTSDG、ZITS 等）在 PSNR、SSIM、FID 等指标上对比，本文在所有遮挡比例与分辨率下均取得更低 FID、更高 PSNR/SSIM 的优异成绩，尤其在高分辨率和大遮挡下优势明显。

**⚠️ 局限性**

主要局限包括：① 对计算资源需求较高，模型体量大；② 仍需更广泛的遮挡场景验证；③ 在极端遮挡或非人脸场景的鲁棒性尚未充分评估。

---

## 247. Evaluating False Alarm and Missing Attacks in CAN IDS

**arXiv ID:** 2602.02781 | [PDF](https://arxiv.org/pdf/2602.02781v1)

**作者:** Nirab Hossain `[一作]` (University of Colorado Boulder), Pablo Moriano `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 278 | [OpenAlex ID](https://openalex.org/A5025254339)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在CAN总线环境下，对浅层模型与深度神经网络IDS进行对抗性攻击评估，探讨误报和漏检两种失败模式。

**💡 创新点**

首次将协议兼容、payload级别的梯度攻击（FGSM/BIM/PGD）与实际车辆CAN流（ROAD数据集）结合，并比较浅层与深度模型的鲁棒性。

**🔧 技术方法**

采用梯度攻击（FGSM、BIM、PGD）、深度前馈网络与决策树/随机森林/ExtraTrees/XGBoost等监督学习模型。

**📊 数据集**

ROAD CAN IDS 数据集（3.5小时真实车辆流量，包含12个正常采集和33个攻击采集）。

**📈 对比分析**

通过对比ASR、MCC等指标发现：深度网络在正常流量下误报低，但对攻击样本的误检率可达1.0；ExtraTrees在大多数场景下保持ASR<0.6，整体鲁棒性最佳。

**⚠️ 局限性**

仅考虑payload级别的攻击，未探索更复杂的时序或多步攻击；实验仅在单个车辆环境下进行，未验证对不同ECU配置的泛化能力。

---

## 248. Eidolon: A Practical Post-Quantum Signature Scheme Based on k-Colorability in the Age of Graph Neural Networks

**arXiv ID:** 2602.02689 | [PDF](https://arxiv.org/pdf/2602.02689v1)

**作者:** Asmaa Cherkaoui `[一作]` (University Hassan II), Richard Wilson `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于k-着色问题的后量子数字签名方案；

**💡 创新点**

创新点在于将Goldreich‑Micali‑Wigderson零知识协议推广到任意k≥3，结合Fiat‑Shamir转换与Merkle树压缩，并通过安静植入色彩生成难解实例；

**🔧 技术方法**

核心技术包括k-着色零知识协议、Fiat‑Shamir变形、Merkle树向量承诺、随机图实例生成和图神经网络攻击评估；

**📊 数据集**

使用随机生成的k-分区Erdős–Rényi图（n≈60–200，p=1/2）作为测试数据集；

**📈 对比分析**

与经典DSatur启发式和自定义GNN攻击进行对比，实验表明在n≥60时两种攻击均无法恢复秘密色彩，签名大小在Merkle压缩后约为144 KiB（无路径共享）或137 KiB（共享路径）；

**⚠️ 局限性**

局限在于对更大图规模和更强学习模型的抵抗性尚未验证，且签名方案依赖于随机图分布的统计隐藏性，若实例生成不足可能导致安全缺陷。

---

## 249. FaceLinkGen: Rethinking Identity Leakage in Privacy-Preserving Face Recognition with Identity Extraction

**arXiv ID:** 2602.02914 | [PDF](https://arxiv.org/pdf/2602.02914v1)

**作者:** Wenqi Guo `[一作]` (University of British Columbia), Shan Du `[通讯]` (University of British Columbia)

**通讯引用:** 1938 | [OpenAlex ID](https://openalex.org/A5049374513)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FaceLinkGen身份提取攻击，揭示现有PPFR模板在视觉扭曲下仍能泄露身份信息。

**💡 创新点**

将攻击焦点从像素级重建转向身份表征提取，采用蒸馏与扩散生成相结合的轻量级方法，显著突破传统PSNR/SSIM评价。

**🔧 技术方法**

利用ArcFace教师/学生蒸馏、cosine相似度链接、Arc2Face扩散生成模型，并通过Face++/Amazon API进行身份一致性验证。

**📊 数据集**

使用CASIA‑WebFace（10k身份）、TPDNE、LFW及hold‑out子集进行训练与评估。

**📈 对比分析**

与三种SOTA PPFR（PartialFace、MinusFace、FracFace）对比，链接成功率>90%/98%，重生成功率>90%，明显优于仅依赖像素重建的评估方法，证明视觉失真不足以阻止身份泄露。

**⚠️ 局限性**

局限性在于攻击仍需模板可见性，依赖生成模型的质量；对极端零知识情形虽然仍有效，但对未来采用加密或多因子验证的PPFR方案的适用性尚需进一步验证。

---

## 250. CATNIP: LLM Unlearning via Calibrated and Tokenized Negative Preference Alignment

**arXiv ID:** 2602.02824 | [PDF](https://arxiv.org/pdf/2602.02824v1)

**作者:** Zhengbang Yang `[一作]` (George Mason University), Zhuangdi Zhu `[通讯]` (George Mason University)

**通讯引用:** 1194 | [OpenAlex ID](https://openalex.org/A5079428801)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于校准与分词负偏好对齐（CATNIP）的LLM记忆去除方法，利用自适应反向参考模型和按token级别校准的损失来精准削弱不良知识的影响，同时保持通用知识不受损失。

**💡 创新点**

创新点在于（1）使用自适应的反向参考模型 1-π 让模型置信度越高的token梯度越大；（2）将全序列的去除目标拆分为token级别的损失，消除长度偏差；（3）不需要保留数据或对比响应，显著提升了去除效率和鲁棒性。

**🔧 技术方法**

技术手段包括：偏好对齐优化、Bradley–Terry 机制、sigmoid 重新加权、token化损失与自适应参考模型、梯度加权与重标定。

**📊 数据集**

评测数据集：MUSE‑Bench（哈利波特版权内容）、WMDP（网络安全与生物学领域的危险知识）以及 MMLU（通用知识维持评估），同时使用问答式轻量级训练集验证数据稀缺场景。

**📈 对比分析**

与 GA、NPO、SimNPO、FLAT、RMU 等基线对比，采用 Δf（遗忘度）、Δu（通用知识保留）与整体质量提升 ΔO↑ 作为指标；CATNIP 在所有无保留数据方法中实现最高的 ΔO↑，遗忘效果优于对比方法且保持更高的通用能力，甚至超过部分依赖保留或对比数据的方法。

**⚠️ 局限性**

局限性：仅在 7B/8B 级别模型上验证，是否能推广到更大规模模型尚待验证；尽管比现有方法更好，但仍存在一定的通用知识损失；实验规模和任务覆盖度有限。

---

## 251. Enhancing Psychologists' Understanding through Explainable Deep Learning Framework for ADHD Diagnosis

**arXiv ID:** 2602.02535 | [PDF](https://arxiv.org/pdf/2602.02535v1)

**作者:** Abdul Rehman `[一作]` (Western Norway University of Applied Sciences), Jerry Chun-Wei Lin `[通讯]` (Western Norway University of Applied Sciences)

**通讯引用:** 18995 | [OpenAlex ID](https://openalex.org/A5000640263)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

建立了一个融合深度神经网络与循环神经网络的可解释框架HyExDNN‑RNN，用于 ADHD 的二分类与多分类诊断，并通过 SHAP 与 PFI 实现模型可解释性。

**💡 创新点**

创新点在于将深度学习与可解释 AI 技术结合，提出 HyExDNN‑RNN 结构并使用 Pearson 相关系数进行特征筛选，既提升诊断准确率，又提供透明的决策解释。

**🔧 技术方法**

使用的技术包括：深度神经网络（DNN）、长短时记忆网络（LSTM）、门控循环单元（GRU）、HyExDNN‑RNN 混合模型；传统机器学习模型有随机森林（RF）、决策树（DT）、XGBoost（EXGB）；解释技术采用 SHAP 与 Permutation Feature Importance（PFI）；特征选择采用 Pearson 相关系数。

**📊 数据集**

使用公开 ADHD200 数据集，包含多站点结构与功能 MRI 指标以及行为量表等多维特征。

**📈 对比分析**

与传统机器学习模型（RF、DT、EXGB、SVM、LR、KNN、ANN）及其他深度模型（LSTM、LSTM‑GRU、LSTM‑RNN、DNN）进行对比，HyExDNN‑RNN 在二分类上达 99% F1、95% ROC‑AUC；在多分类上准确率 94.20%，显著优于其他模型。

**⚠️ 局限性**

局限性包括：仅在 ADHD200 数据集验证，样本不平衡（部分类别极少）；未充分评估跨站点泛化能力；缺乏迁移学习或多模态数据的尝试；解释性主要基于 SHAP/PFI，可能存在解释范围有限的问题。

---

## 252. Which course? Discourse! Teaching Discourse and Generation in the Era of LLMs

**arXiv ID:** 2602.02878 | [PDF](https://arxiv.org/pdf/2602.02878v1)

**作者:** Junyi Jessy Li `[一作]` (University of Texas at Austin), William Sheffield `[通讯]` (University of Texas at Austin)

**通讯引用:** 2519 | [OpenAlex ID](https://openalex.org/A5067523103)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实施了一门跨学科的本科课程《Computational Discourse and Natural Language Generation》，将对话语处理与长文本生成的理论与实践结合，旨在培养学生在LLM与语言学框架下进行实验与评估的能力。

**💡 创新点**

创新点在于：①将传统对话语结构理论（RST、PDTB、QUD、Centering Theory）与LLM生成、解码、RLHF等技术无缝对接；②采用开放式实验作业与项目，让学生在实际编码与批判性分析中自行设定研究方向；③课程由多学科研究团队协同设计，并通过外部评估（TACC教育服务）验证学习成效。

**🔧 技术方法**

主要技术包括：Transformer‑based autoregressive LLMs、各种解码策略（贪婪、top‑p、min‑p）、RLHF/RLVR、LLM‑as‑judge、最小配对评估、实体网格与Coherence Metrics、Pytorch/HuggingFace 生态、Colab Pro 计算环境。

**📊 数据集**

使用的数据集与资源有：RST树库、PDTB、QUD数据、DISRPT 2025共享任务、实体追踪实验集（如《Scenario Tracking》）、书籍摘要与最小配对句子集、各类翻译与生成任务数据。

**📈 对比分析**

对比方法包括：基线对比（基础LM vs 说明调优LM 的 perplexity 与输出多样性评估）；BLEU 与 LLM‑as‑judge 的翻译质量评估；对话语关系分类与 RST 生成的精确度评估；实体网格与Centering理论对摘要连贯性的评估。学生反馈显示课程平均满意度 95%，80% 计划将所学应用于未来学习或职业。

**⚠️ 局限性**

局限性主要有：计算资源受限（大模型实验依赖外部 API 或 Colab 受限）；作业评估需人工，难以规模化；课程内容以英语为主，缺乏多语言覆盖；作业开放式导致评估标准不易统一。

---

## 253. VividVoice: A Unified Framework for Scene-Aware Visually-Driven Speech Synthesis

**arXiv ID:** 2602.02591 | [PDF](https://arxiv.org/pdf/2602.02591v1)

**作者:** Chengyuan Ma `[一作]` (Shenzhen International Graduate School Tsinghua University), Wenming Yang `[通讯]` (Ant Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了场景感知视觉驱动语音合成（Scene‑Aware Visually‑Driven Speech Synthesis）任务，并实现了统一的生成框架 VividVoice。

**💡 创新点**

创新点包括：①首创该任务和大规模多模态数据集 Vivid‑210K；②设计 Decoupled Multi‑modal Scene–Voice Alignment（D‑MSVA）模块，利用多记忆池实现声色与环境音的细粒度解耦；③采用混合监督策略（对齐、对比、模仿）提升跨模态映射精度。

**🔧 技术方法**

使用技术包括：潜在扩散模型（AudioLDM 变体）、CLAP、MetaCLIP、VITS、文本‑图像/音频生成模型（FLUX.1、Stable Audio Open）、双路径对齐与注意力机制、混合监督损失、EMA 与 AMP 训练技巧。

**📊 数据集**

使用数据集：Vivid‑210K（210k样本，800+ 说话人），包含程序化合成与真实微调子集；同时参考 LRS3、VGGSound、FLIP、CLAP 等公开资源进行构造与评估。

**📈 对比分析**

与 VoiceLDM 作为基线进行对比，主观客观指标均显著提升：WER 7.15%（vs 9.23%），FAD 3.98（vs 5.12），KL 1.53，MOS‑TI 4.30（vs 1.75），MOS‑SC 4.30（vs 2.56），A/B 偏好率 64%/53%。

**⚠️ 局限性**

局限性：CLAP_cap 分数略低，说明视觉细节与音频匹配仍有偏差；当记忆池槽数过大（>128）时易过拟合；当前模型主要针对已合成的视觉场景，缺乏对更广泛真实场景的泛化能力；对极端噪声或低光条件下的视觉信息鲁棒性待提升。

---

## 254. HyPAC: Cost-Efficient LLMs-Human Hybrid Annotation with PAC Error Guarantees

**arXiv ID:** 2602.02550 | [PDF](https://arxiv.org/pdf/2602.02550v1)

**作者:** Hao Zeng `[一作]` (Southern University of Science and Technology), Hongxin Wei `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1062 | [OpenAlex ID](https://openalex.org/A5020027500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种多源注释框架 HyPAC，能够根据样本不确定性动态路由到不同成本与质量水平的注释源（快速 LLM、慢速推理 LLM 与人工），并在保持注释错误率低于用户设定阈值的前提下实现最低的期望成本；

**💡 创新点**

创新点包括：① 在多源注释问题上引入 PAC 风格的错误率控制，实现分布无关、数据集无关的错误保证；② 通过重要性采样与上置信界（UCB）自适应校准两个阈值，实现误差控制与成本最小化的两阶段优化；③ 证明在满足 PAC 约束的所有阈值策略中，HyPAC 取得期望成本最优；

**🔧 技术方法**

技术手段包括：重要性采样估计风险、CLT/Hoeffding/ Bernstein 等上置信界构造、PAC 误差控制、阈值网格搜索、基于分数的不确定性评估（logits‑based 与 verbalized），以及 token‑based 与 API‑based 成本函数；

**📊 数据集**

实验使用的公开数据集包括：MMLU‑Redux、MATH‑500、MATH‑L5、Zebra‑Logic、HumanEval⁺；注释源模型包括 Qwen3‑4B‑Instruct/Thinking、Llama‑3.1‑8B‑Instruct/DeepSeek‑R1‑Distill‑Llama‑8B、Qwen2.5‑32B‑Instruct/DeepSeek‑R1‑Distill‑Qwen‑32B；

**📈 对比分析**

与基线 PAC‑Labeling（非思考/思考+人工）、CSE 与 CoAnnotating 进行比较。结果显示，HyPAC 在保持错误率不超过 ε（如 5%）的同时，成本节省率最高，HumanEval⁺ 上可达约 86% 节省，MATH‑500 上约 78% 节省，整体比其他 provable 方法节省 10–20% 的成本且误差始终在目标范围内；

**⚠️ 局限性**

局限性包括：① 需要足够大的校准集以保证 UCB 的可靠性；② UCB 的有效性依赖于所选分布无关/CLT 等假设，样本量不足时可能失效；③ 当前实现仅支持两阈值分割，扩展到更多注释源仍需研究；④ 在严重分布漂移或极端样本不确定性下性能可能下降；⑤ 依赖于预训练模型的质量，若模型本身误差过大，整体效果受限。

---

## 255. CreditAudit: 2D Auditing for LLM Evaluation and Selection

**arXiv ID:** 2602.02515 | [PDF](https://arxiv.org/pdf/2602.02515v1)

**作者:** Yiliang Song `[一作]` (Guangxi Normal University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 61657 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了 CreditAudit 框架，评估大模型在一组语义对齐且非攻击性的系统提示族下的平均性能（μ）与波动（σ），并将 σ 映射为 AAA–BBB 的信用等级；

**💡 创新点**

创新点在于：①以系统提示族为实验因子，将协议敏感度量化为可比的波动 σ；②将 σ 与 μ 并列展示，形成二维选择面；③用跨模型分位数将 σ 量化为直观的信用标签，直接支持模型部署决策；

**🔧 技术方法**

主要技术包括：结构化提示族构建、统一解析+准确率评估、两维统计（μ、σ）计算、跨模型分位数映射为信用等级、以及诊断模块（模板中性检验、分布可视化）；

**📊 数据集**

使用公开多项选择基准：GPQA、TruthfulQA、MMLU‑Pro；

**📈 对比分析**

通过在所有提示族上计算每模型的平均准确率和标准差，并按 σ 分级；实验显示：即便平均分相近，模型在稳定性（σ）上差异显著，信用等级从 AAA 到 BBB 的梯度清晰对应不同部署风险；

**⚠️ 局限性**

局限性：仅覆盖多项选择任务，未扩展到开放式生成或工具调用；提示族有限，可能无法覆盖所有实际协议变体；仅关注性能与协议波动，未结合成本、延迟、安全性等多目标因素。

---

## 256. IMAGINE: Intelligent Multi-Agent Godot-based Indoor Networked Exploration

**arXiv ID:** 2602.02858 | [PDF](https://arxiv.org/pdf/2602.02858v1)

**作者:** Tiago Leite `[一作]` (INESC INOV), António Grilo `[通讯]` (INESC INOV)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过在 Godot 引擎中模拟 GNSS‑禁区的室内环境，使用多智能体强化学习（MARL）训练一组配备 LiDAR 的无人机，在有限通信范围内协同完成未知空间的地图构建与区域覆盖。

**💡 创新点**

创新点包括：①将通信约束融入 ND‑POMDP 框架，利用 Belief‑MDP 代替传统 RNN 以更有效处理局部观测；②提出基于 LiDAR 的 1D 卷积网络显著提升感知效率；③采用分阶段课程学习（Curriculum‑Learning）加速训练；④首次在高保真游戏引擎 Godot 中实现 MARL 的完整管线。

**🔧 技术方法**

主要技术包括：PPO、CTCE/CTDE/DTDE 三种 MARL 并行训练范式；CNN + 1D LiDAR 卷积网络；Ray/RLlib 分布式训练与 WandB 监控；Godot‑Python RL Agent 接口；以及基于 OGM 的占用格网建图。

**📊 数据集**

使用自定义的 7 级分级室内环境（尺寸与障碍数量递增）进行仿真训练与评估，所有数据均由 Godot 生成；未使用公开真实数据集。

**📈 对比分析**

通过在不同 agent 数量（1/2/3）和三种 MARL 范式下评估区域覆盖率，发现 CTDE 在复杂环境中表现最佳，最高级别可达 95% 覆盖；通信量与性能呈非线性关系，过度共享导致回退；实验在 HPC 集群上完成，平均每层训练数千步。

**⚠️ 局限性**

局限性包括：①样本复杂度高，训练周期长；②仅在 2D 平面上验证，缺乏 3D 扩展；③只使用 LiDAR，未考虑视觉或多模态传感；④通信模型过于理想化，未包含噪声与多跳延迟；⑤缺乏真实无人机实验验证。

---

## 257. Act or Clarify? Modeling Sensitivity to Uncertainty and Cost in Communication

**arXiv ID:** 2602.02843 | [PDF](https://arxiv.org/pdf/2602.02843v1)

**作者:** Polina Tsvilodub `[一作]` (University of Tübingen), Michael Franke `[通讯]` (University of Tübingen)

**通讯引用:** 2555 | [OpenAlex ID](https://openalex.org/A5076654614)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究人类在不确定情境下是否会提出澄清问题(CQ)，并检验其是否受不确定度与行动成本交互影响；

**💡 创新点**

首次将期望后悔(Equivalence to Expected Value of Perfect Information)框架应用于人类澄清行为的决策建模，并通过实验验证不确定度与成本的交互效应；

**🔧 技术方法**

使用贝叶斯逻辑回归分析实验数据，并在Stan中实现基于期望后悔的层次决策模型；

**📊 数据集**

数据来源于两项线上实验：实验1（N=125）在Prolific平台收集的问答情景；实验2（N=118）收集的指令情景的滑动评分；

**📈 对比分析**

模型通过留一交叉验证与三种消融版本比较，结果显示包含不确定度与成本的完整期望后悔模型拟合度最高，后者显著优于仅考虑成本或仅考虑不确定度的模型；

**⚠️ 局限性**

模型未能解释部分行为偏差（如过度列举或先前提及项的偏好）；未显式建模提问成本、时间延迟及社交成本；并且仅在单一行动空间内验证，缺乏对混合语言与非语言行动的普适性检验。

---

## 258. Manifold-Constrained Energy-Based Transition Models for Offline Reinforcement Learning

**arXiv ID:** 2602.02900 | [PDF](https://arxiv.org/pdf/2602.02900v1)

**作者:** Zeyu Fang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6378 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于能量模型的离线模型预测方法 MC-ETM，利用流形投影扩散生成近似数据流形的负样本，提升模型对分布外样本的判别能力并改进多步转移精度。

**💡 创新点**

创新点在于将流形投影扩散（MPD）融入能量模型训练，生成近似流形的硬负样本；同时使用能量阈值进行回滚截断和基于能量与 Q 方差的惩罚，实现统一的可靠性信号。

**🔧 技术方法**

核心技术包括能量基础模型（EBM）与对比学习、流形投影扩散、Langevin 动态采样、能量阈值截断以及多样本 Q 方差惩罚的离线策略优化。

**📊 数据集**

主要使用 D4RL 机器人连续控制数据集（HalfCheetah、Hopper、Walker2d）以及随机/中等/专家质量的离线数据。

**📈 对比分析**

与 CQL、TD3+BC、EDAC、MOPO、COMBO、RAMBO、MOBILE、EMPO 等基线比较，MC-ETM 在绝大多数任务中取得最高或接近最高的归一化回报，显著优于传统模型无关与模型相关方法。

**⚠️ 局限性**

局限包括对流形学习的依赖（需要有效的自编码器），采样过程仍有一定计算开销，且在极端高维视觉观测场景中的适用性尚未验证。

---

## 259. Efficiency Optimizations for Superblock-based Sparse Retrieval

**arXiv ID:** 2602.02883 | [PDF](https://arxiv.org/pdf/2602.02883v1)

**作者:** Parker Carlson `[一作]` (University of California), Tao Yang `[通讯]` (University of California)

**通讯引用:** 159246 | [OpenAlex ID](https://openalex.org/A5019365851)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级超级块裁剪（LSP）方案，用于学习稀疏检索中提升在线检索速度，同时保持相关性。

**💡 创新点**

创新点包括：① 通过保证访问前γ个最大值超级块避免误裁剪；② 去掉平均分数约束，显著降低计算开销；③ 采用SIMD‑BP‑256*与4‑bit量化的紧凑索引结构；④ 在无需网格搜索的零射击配置下即可保持高召回。

**🔧 技术方法**

主要技术包括超级块与块级裁剪、阈值过估（μ、η）、SIMD批量处理（SIMDBP‑256*）、4‑bit量化、前向索引、查询词裁剪β、γ超块保证。

**📊 数据集**

实验数据集：MS MARCO Passage（Dev、TREC DL19/20）、BEIR 13个多域数据集，评估在SPLADE++与Efficient‑SPLADE模型上。

**📈 对比分析**

与SP、BMP、SeismicWave等基线对比，LSP/0 在k=10时相对SP提升1.8–4.8倍速度、相对BMP提升1.8–12倍速度；在k=1000时相对SP提升1.8–17倍、相对BMP提升1.8–12倍，且在零射击场景下LSP/0 2–5倍快于SeismicWave，且内存占用显著降低；召回率保持在99%+安全召回的范围内。

**⚠️ 局限性**

局限性：对块/超级块尺寸敏感，k=10时若γ选取过小可能略慢；对不同模型的泛化需手动调整γ；在极大索引或非常小块时，前向索引性能退化；未在所有类型模型（如非SPLADE）上验证；不含静态裁剪或近邻图等高级优化。

---

## 260. Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion

**arXiv ID:** 2602.02722 | [PDF](https://arxiv.org/pdf/2602.02722v1)

**作者:** Dan Haramati `[一作]` (Brown University), George Konidaris `[通讯]` (Brown University)

**通讯引用:** 5485 | [OpenAlex ID](https://openalex.org/A5078124517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种层次化实体中心的离线目标条件强化学习框架HECRL，结合子目标分解与因子结构，解决多实体图像任务的长周期。

**💡 创新点**

将条件扩散模型用于生成实体因子化子目标，并通过价值函数进行筛选，实现模块化且兼容任何基于价值的GCRL算法。

**🔧 技术方法**

离线GCRL、条件扩散子目标生成器、实体中心变换器、DLP/VQ‑VAE 视觉编码、IQL/HIQL 等基线算法。

**📊 数据集**

基于 OGBench 的多物体操作环境（Cube、Scene、Push‑T）以及通过 DLP 预训练的图像数据集。

**📈 对比分析**

与 EC‑IQL、EC‑Diffuser、HIQL、IQL 等基线比较，在多实体长周期任务中成功率提升超过 150%，在图像任务上显著优于所有对照。

**⚠️ 局限性**

依赖价值函数的可达半径和高质量实体估计；K 步子目标设定仍需手动；在真实世界应用中需更完善的无监督对象表示。

---

## 261. A General ReLearner: Empowering Spatiotemporal Prediction by Re-learning Input-label Residual

**arXiv ID:** 2602.02563 | [PDF](https://arxiv.org/pdf/2602.02563v1)

**作者:** Jiaming Ma `[一作]` (University of Science and Technology of China), Yang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 34170 | [OpenAlex ID](https://openalex.org/A5100764445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于时空残差理论的ReLearner模块，能够在训练阶段显式加入标签特征并通过残差学习与平滑机制，对STNN进行双向学习，从而显著提升时空预测性能。

**💡 创新点**

创新点在于将标签信息直接嵌入学习流程，构建残差学习与残差平滑两大模块，形成一个通用的、可无缝集成到现有STNN模型的逆向学习框架，用以建模和纠正输入-标签偏差。

**🔧 技术方法**

采用高斯马尔可夫随机场（GMRF）理论推导残差学习公式；实现残差学习模块、残差平滑传播核（预定义、扩散、自适应、数据驱动）以及MLP+GELU激活；在模型中引入残差传播层和校正解码器。

**📊 数据集**

使用11个真实时空数据集：交通领域的LargeST（SD、GBA、GLA、CA）、PEMS系列（PEMS03、04、07、08、PEMS3-Stream）、METR-LA；气象领域的KnowAir（PM2.5监测）。

**📈 对比分析**

将ReLearner集成至14种基准STNN（如STGCN、AGCRN、STAEformer、D^2STGNN等），在MAE/RMSE/MAPE等指标上平均提升10–18%，最大提升达21.18%；在气象指标CSI/POD/FAR上亦实现显著改善。

**⚠️ 局限性**

仍对超参数（残差传播层数、传播核选择）敏感；在极大规模图或长时序情境下计算和内存成本显著上升；对极端缺失或不平衡数据的鲁棒性尚未充分验证。

---

## 262. From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection

**arXiv ID:** 2602.03018 | [PDF](https://arxiv.org/pdf/2602.03018v1)

**作者:** Xueying Ding `[一作]` (Carnegie Mellon University), Leman Akoglu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8361 | [OpenAlex ID](https://openalex.org/A5001634795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的表格异常检测基础模型 Outformer，能够在没有标签的情况下实现零样本推理。

**💡 创新点**

创新点包括：①混合合成先验生成多样化的训练数据；②基于多臂赌博机的自适应进化课程训练；③在 Transformer 上实现零样本推理并通过上下文采样与维度装袋实现集成。

**🔧 技术方法**

技术主要包括：Transformer‑based Prior‑Data‑Fitted Networks (PFN)、自适应课程（SEC）与多臂赌博机奖励机制、上下文采样与维度装袋等。

**📊 数据集**

数据集使用了三大公开异常检测基准（一个传统 57 数据集、两个新构建的 690 与 756 数据集，总计 1500+ 实际数据集）以及 2000+ 合成内分布数据。

**📈 对比分析**

与多种浅层、深度及现有基础模型基准（如 DTE‑NP、TabPFN‑OD 等）对比，Outformer 在 AUROC/AUPRC 等五项指标上平均排名第4，胜率 0.65，显著优于其他方法，且推理速度极快。

**⚠️ 局限性**

局限性在于缺乏对公平性、偏差等伦理问题的考虑，且对极端稀有异常的泛化能力仍需进一步验证。

---

## 263. Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit

**arXiv ID:** 2602.02602 | [PDF](https://arxiv.org/pdf/2602.02602v1)

**作者:** Yangfan Deng `[一作]` (University of Maryland), Min Wu `[通讯]` (University of Maryland)

**通讯引用:** 28909 | [OpenAlex ID](https://openalex.org/A5074710425)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了针对 3D 高斯 splatting（3DGS）水印的场景驱动安全蓝图，并给出了可复现的扩散谱嵌入基线，旨在填补现有方法中缺乏统一威胁模型与可比评估的空白。

**💡 创新点**

创新点包括：
- 以部署场景为导向明确威胁模型，使用“访问向量”系统化描述攻击者权限；
- 设计了可键控的扩散谱嵌入与检测框架，明确了三种密钥（载体选择、声明绑定、随机码生成）；
- 在 3DGS 原始参数空间直接实现扩散谱，提供透明可复现的基准；
- 通过统一的可见性与鲁棒性度量，展示了容量‑鲁棒性‑保真度权衡。

**🔧 技术方法**

使用技术包括：
- 3D Gaussian Splatting 参数表示与渲染公式；
- 一维 DCT 作为变换域载体；
- 码分多址（CDM）扩散谱嵌入；
- 非盲检测（残差相关解码）；
- 对抗性攻击模拟（高斯噪声、旋转、缩放、模糊、裁剪、亮度、JPEG、平移、VAE、Dropout、3D 裁剪、克隆等）。

**📊 数据集**

实验使用的主数据集为：Blender、LLFF、Mip‑NeRF 360；此外还对比了现有方法所采用的数据集（如 BlenderLLFF、ObjaverseBlenderOmniObject3D 等）。

**📈 对比分析**

方法比较：
- 在统一的 Gaussian‑noise 破坏下，扩散谱基线的比特准确率随嵌入强度 α 下降而下降，但保真度（PSNR/SSIM/LPIPS）保持在可接受范围内；
- 与现有工作（GuardSplat、3D‑GSW、GS‑Marker、MarkSplatter、Water‑GS、GaussianMarker）对比，基线在相同 payload 下往往能获得相似或略高的比特准确率，同时在视觉保真度上更有优势；
- 通过控制总能量，展示了 payload 长度（32/48/64 bits）与比特准确率、保真度的负相关关系，体现了容量‑鲁棒性‑保真度三者的权衡。

**⚠️ 局限性**

限制与未来工作：
- 仅在模拟的单一破坏（Gaussian noise 等）上评估鲁棒性，缺乏对真实渲染流水线、动态光照等复杂场景的测试；
- 采用非盲检测，未实现对可疑模型的无密钥推断；
- 关键依赖假设，未充分探讨公钥/私钥部署与对抗攻击下的安全性；
- 仅在三大数据集上验证，缺少跨域（不同 3DGS 生成器、不同渲染器）推广实验；
- 未提供基于 2D 渲染输出的专用检测器，仍使用通用图像水印器，可能被攻击者利用。

---

## 264. TabularMath: Evaluating Computational Extrapolation in Tabular Learning via Program-Verified Synthesis

**arXiv ID:** 2602.02523 | [PDF](https://arxiv.org/pdf/2602.02523v1)

**作者:** Zerui Cheng `[一作]` (ByteDance Seed), Wenhao Huang `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于验证的生成-验证器管线，将数学推理题目转换为无噪声的表格数据，并在此基准上评估表格模型的插值与外推性能。

**💡 创新点**

创新点在于：①提出全自动化的验证式生成器-验证器链条，实现了数百万行无误标签的表格数据；②引入 R²–一致性对比度量，揭示表格模型与 ICL 在外推时的显著性能差距；③构建了专门针对确定性计算任务的基准，填补了传统表格学习评测缺失的推理维度。

**🔧 技术方法**

技术手段包括：LLM 编译器生成器-验证器程序；TabPFN 预训练网络；树基模型、深度表格网络与 in‑context 学习（GPT‑OSS‑120B）；特征工程（对数、模运算、符号特征）以及严格的训练/测试拆分策略。

**📊 数据集**

数据集来源于 114 道来自 GSM8K 与 AIME 的数学题目，利用管线生成 233,472 行无错误标签的表格数据，涵盖多种算术与逻辑推理任务。

**📈 对比分析**

对比方法采用 80/20 随机与 OOD（输出外推）拆分，并使用 R²、RMSE、MAE 与整数取整一致性作为指标。结果显示：TabPFN v2.5 在插值时 R² 接近 1 但一致性仅 62%；ICL 在 OOD 下保持约 40% 一致性；树模型一致性不足 10%，而深度表格网络则表现更差。

**⚠️ 局限性**

局限性在于：①基准仅适用于确定性计算任务，无法覆盖带噪声或分类特征的实际业务场景；②ICL 受上下文长度限制，推理成本高；③表格模型缺乏系统推理能力，难以在外推时实现精确计算。

---

## 265. Zero Sum SVD: Balancing Loss Sensitivity for Low Rank LLM Compression

**arXiv ID:** 2602.02848 | [PDF](https://arxiv.org/pdf/2602.02848v1)

**作者:** Ali Abbasi `[一作]` (Vanderbilt University), Soheil Kolouri `[通讯]` (Vanderbilt University)

**通讯引用:** 3312 | [OpenAlex ID](https://openalex.org/A5068682350)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于零和原则的SVD压缩方法ZS‑SVD，在后训练阶段对LLM进行低秩压缩。

**💡 创新点**

通过计算白化坐标下的单个奇异值对校准损失的一阶敏感度，并使用全局零和选择策略自动分配各层不同秩，实现无显式层级分配优化。

**🔧 技术方法**

使用激活白化、SVD、梯度一阶方向导数、全局零和贪心选择、可选的一步投影梯度校正及再截断。

**📊 数据集**

在WikiText‑2、Penn Treebank、C4等语言建模基准以及OpenBookQA、ARC、WinoGrande、HellaSwag、PIQA、MathQA等七个零样本推理任务上进行评估，并以LLaMA、Vicuna、LLaMA‑2、OPT等多规模LLM作为实验模型。

**📈 对比分析**

与ASVD、SVD‑LLM、Dobi‑SVD以及结构化剪枝方法进行对比，ZS‑SVD在不同压缩比下在困惑度、零样本准确率、推理吞吐量和截断时间上均优于基线，尤其在高压缩比时提升显著。

**⚠️ 局限性**

方法仍需依赖校准数据，零和策略对极端压缩时的精度下降控制不够；可选的校正步骤虽低秩但对梯度秩假设不一定成立；在GPU加速与内存受限场景下仍需进一步优化实现。

---

## 266. Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control

**arXiv ID:** 2602.02987 | [PDF](https://arxiv.org/pdf/2602.02987v1)

**作者:** Ruihan Lin `[一作]`, Jiheng Zhang `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一套基于流体逼近和线性规划的多类LLM推理排程框架，设计了门控‑路由（Gate‑and‑Route）策略实现异构工作负载的可扩展调度；

**💡 创新点**

创新点在于：①将prefill–decode 阶段的 GPU 竞争建模为状态相关多服务器排队网络；②推导出可解的稳态线性规划并证明门控‑路由策略在多GPU极限下渐近最优；③在框架中加入服务水平指标（SLI）约束，提供统一的收益‑公平‑延迟权衡方法；

**🔧 技术方法**

使用了随机控制、流体逼近、线性规划求解、门控‑路由控制、SLA 约束分析等技术；

**📊 数据集**

实验采用真实 NVIDIA A100‑SXM4‑40GB GPU 环境，基于 Qwen‑4B 与 Qwen‑8B 模型的prefill 与 decode 迭代时间数据，并以模拟的两类工作负载（长预填短解码与短预填长解码）进行评估；

**📈 对比分析**

与业界基线（如 Sarathi‑Serve 的 FCFS 与即时调度）以及若干消融实验比较，门控‑路由策略在大规模集群下每 GPU 收益逼近流体最优，且队列长度与资源占用稳定，平均收益提升约 10–20%（具体提升幅度随硬件与负载而异）；

**⚠️ 局限性**

局限包括：假设服务时间呈指数分布、仅考虑单一 GPU 型号、未覆盖模型大小变化导致的内存迁移成本、缺乏 diffusion 层面尾部延迟分析，以及对实时变化负载分布的适应性仍待进一步研究。

---

## 267. Trajectory Consistency for One-Step Generation on Euler Mean Flows

**arXiv ID:** 2602.02571 | [PDF](https://arxiv.org/pdf/2602.02571v1)

**作者:** Zhiqi Li `[一作]` (Georgia Tech), Bo Zhu `[通讯]` (Georgia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Euler Mean Flows（EMF）框架，实现一次或少数步生成时的轨迹一致性约束，并通过线性近似实现对长程流图的直接数据监督；

**💡 创新点**

创新点在于将半群一致性约束线性化，消除了对 Jacobian‑vector product（JVP）的需求，形成了统一的、无梯度计算的训练方案，同时保留了理论保证；

**🔧 技术方法**

核心技术包括流匹配、半群一致性线性近似、无 JVP 的梯度自由训练、x₁‑预测与 u‑预测两种预测模式以及可选的辅助分支；

**📊 数据集**

使用的数据集涵盖 ImageNet‑1000、CelebA‑HQ、ShapeNet‑CoreV2、MNIST‑SDF、FFHQ、CelebA‑HQ 的稀疏函数表示、以及 Point‑Voxels 的 3D 点云；

**📈 对比分析**

与现有的一步方法（如 MeanFlow、α‑Flow、SplitMF）对比，EMF 在图像、SDF、点云和功能图像生成任务中实现了约 50% 的训练时间和显存节省，并在多种指标（FID、F‑score、Chamfer、Coverage 等）上优于或相当于传统多步或分解式模型；

**⚠️ 局限性**

局限性包括对线性近似的依赖（在某些高非线性场景下可能失效）、u‑预测在像素空间易失效、仍需在更大规模模型与更复杂任务上进一步验证，以及对极端稀疏/高维数据的适应性尚未完全证明。

---

## 268. Sparse Adapter Fusion for Continual Learning in NLP

**arXiv ID:** 2602.02502 | [PDF](https://arxiv.org/pdf/2602.02502v1)

**作者:** Min Zeng `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18524 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Sparse Adapter Fusion Method (SAFM)，通过在连续学习中动态融合旧新适配器来减轻灾难性遗忘。

**💡 创新点**

创新点在于引入两阶段决策—重用、空适配器优先—与层级损失，既降低参数冗余又提升知识共享。

**🔧 技术方法**

采用 GPT‑2 作为基础模型，使用架构搜索、伪重放、层级余弦损失等技术实现参数高效。

**📊 数据集**

在相似与不相似两种场景下，使用 E2ENLG、RNNLG、SGD、TM19、TM20、WikiSQL、CNN/DailyMail 等任务集进行实验。

**📈 对比分析**

与多种基线（AdapterCL、ACM、SAPT 等）对比，SAFM 在保持相同或更高平均分的同时，参数量降至 SOTA 的 60% 以下，并实现正向转移。

**⚠️ 局限性**

局限性是仅关注任务级增量，未对样本级增量进行建模，可能在异常样本上表现不足。

---

## 269. NLI:Non-uniform Linear Interpolation Approximation of Nonlinear Operations for Efficient LLMs Inference

**arXiv ID:** 2602.02988 | [PDF](https://arxiv.org/pdf/2602.02988v1)

**作者:** Jiangyong Yu `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于动态规划求最优非均匀线性插值的非均匀线性插值（NLI）框架，用于高精度近似LLM中的非线性激活函数，且无需校准；

**💡 创新点**

创新点在于将截点选择转化为全局最优的动态规划问题，既实现了最小插值误差，又保证了硬件友好与可复用；

**🔧 技术方法**

主要技术包括：动态规划求最优截点、两层地址映射、FP16双路流水线插值、SMIC 28nm实现的硬件引擎；

**📊 数据集**

使用了LLM（LLaMA、Qwen系列）和其他DNN（ViT、CNN）以及通用基准（Wikitext‑2、MMLU、HumanEval、GSM8k）进行评估；

**📈 对比分析**

与现有NN‑LUT、RI‑LUT等方法对比，NLI在保持0.1%以内的精度损失的同时，面积下降≈70%，功耗下降≈30%，吞吐率保持1G，综合效率提升≈4×；

**⚠️ 局限性**

局限性包括：仅针对线性插值范围内的激活函数；对极端分布或更复杂非线性函数（如多项式、稀疏激活）仍需进一步验证。

---

## 270. Vector Quantized Latent Concepts: A Scalable Alternative to Clustering-Based Concept Discovery

**arXiv ID:** 2602.02726 | [PDF](https://arxiv.org/pdf/2602.02726v1)

**作者:** Xuemin Yu `[一作]` (Dalhousie University), Hassan Sajjad `[通讯]` (Dalhousie University)

**通讯引用:** 2679 | [OpenAlex ID](https://openalex.org/A5042954793)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于向量量化的潜在概念发现框架（vqlc），将连续的上下文表示映射到可学习的离散码本，实现对模型内部表示的可解释性。

**💡 创新点**

创新点在于将 VQ‑VAE 的码本学习与自适应残差编码相结合，利用 EMA 更新码本并采用温度采样防止码本崩塌，从而在保持解释质量的同时大幅提升可扩展性，突破传统层次聚类和 K‑Means 的 O(N²) 复杂度瓶颈。

**🔧 技术方法**

核心技术包括：自适应残差编码器、可学习码本的向量量化器、EMA 码本更新、温度化软最大采样、线性解码器、重构+承诺损失训练；在实现上借鉴了 VQ‑VAE、Transformer 编码器/解码器以及层归一化等。

**📊 数据集**

使用三类文本分类数据集：IMDB 电影评论（情感）、Jigsaw 有毒内容（有毒/非有毒）、AG News 新闻主题；在 BERT‑base、RoBERTa、Llama‑2‑7B‑chat、Qwen2.5‑3B 四个模型上进行评估。

**📈 对比分析**

与层次聚类基方法 LACOAT 和 K‑Means 进行对比：vqlc 在记忆占用、时间复杂度上实现近线性增长，显著低于 LACOAT；在 faithfulness（探针准确率下降）与 LLM 评估（概念相关性排序）上与 LACOAT 持平或略优，明显优于 K‑Means，且能生成更细粒度、语义一致的概念。

**⚠️ 局限性**

主要局限包括：依赖 K‑Means 作为码本初始化，对随机种子敏感；需要调节若干超参数（码本大小、承诺损失权重、温度等）；目前仅在分类任务上验证，未覆盖生成式模型的逐词推理解释。

---

## 271. StepNav: Structured Trajectory Priors for Efficient and Multimodal Visual Navigation

**arXiv ID:** 2602.02590 | [PDF](https://arxiv.org/pdf/2602.02590v1)

**作者:** Xubo Luo `[一作]` (University of Chinese Academy of Sciences), Ruisuo Wang `[通讯]` (Technology and Engineering Center for Space Utilization, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 StepNav 框架，实现了从视觉输入到实时、安全、平滑路径规划的闭环系统。

**💡 创新点**

创新点在于：① 用几何感知的成功概率场生成多模态结构化先验；② 将先验融入条件流匹配（Reg‑CFM）进行正则化优化；③ 在同一流程中实现低步数高效推理，兼顾安全、平滑与多样性。

**🔧 技术方法**

核心技术包括：V‑JEPA2 视频特征提取 + DIFP 运动一致性优化；Biharmonic 规则化 PDE 训练成功概率场；低能量路径提取（K‑shortest path + Hausdorff 多模态筛选）；正则化条件流匹配（加入平滑与碰撞惩罚）。

**📊 数据集**

训练与评估使用多源视觉导航数据集：RECON、SCAND、GoStanford、SACSoN；在 Stanford 2D‑3D‑S（室内）和 Gazebo CitySim（室外）两个标准 benchmark 进行实验。

**📈 对比分析**

与 ViNT、NoMaD、NaviBridger、FlowNav 等 SOTA 方法对比，StepNav 在室内外基本/适配任务上分别实现了 95%/90% 的成功率、最高 SPL、最低碰撞率和最低最小 Snap，显著优于基线并在推理速度上超过 FlowNav。

**⚠️ 局限性**

局限性：目前对动态障碍和长时间规划的鲁棒性仍有限；成功概率场依赖专家演示数据，难以在完全无标注环境中直接部署；在极端视觉模糊或光照变化时，特征提取与场预测可能失效。

---

## 272. Constitutional Spec-Driven Development: Enforcing Security by Construction in AI-Assisted Code Generation

**arXiv ID:** 2602.02584 | [PDF](https://arxiv.org/pdf/2602.02584v1)

**作者:** Srinivas Rao Marri `[一作]` `[通讯]`, Srinivas Rao Marri

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Constitutional Spec-Driven Development（CSDD）方法，将安全约束嵌入软件“宪法”中，使 AI 代码生成安全性从事后检查转为事前保证；

**💡 创新点**

创新点在于将政治宪法类比到软件领域，形成可版本化、可治理的安全约束集合，并在 AI 代码生成流程中实现自动化验证与追溯；

**🔧 技术方法**

采用大型语言模型（Claude）进行代码生成，结合 FastAPI、SQLAlchemy、Pydantic、python-jose、passlib、React 等技术；

**📊 数据集**

以银行微服务为案例，基于 CWE/MITRE Top‑25 漏洞与 PCI‑DSS/SOC‑2/GDPR 等监管框架构造约束，使用公开 GitHub 仓库中的代码与安全审计数据；

**📈 对比分析**

将宪法约束下的实现与无约束的“vibe coding”对比，测量 10 项关键 CWE 的缺陷数量、首个安全构建时间、合规文档覆盖率等指标，结果显示缺陷下降 73%、构建时间缩短 56%、合规覆盖率提升 4.3 倍；

**⚠️ 局限性**

局限性包括只能覆盖已知漏洞模式、无法处理业务逻辑缺陷、对 AI 理解和生成能力高度依赖、宪法文档可能不完整以及易受 prompt 注入/规范污染攻击。

---

## 273. Recommender system in X inadvertently profiles ideological positions of users

**arXiv ID:** 2602.02624 | [PDF](https://arxiv.org/pdf/2602.02624v1)

**作者:** Paul Bouchaud `[一作]` (Complex Systems Institute of Paris Ile-de-France), Pedro Ramaciotti `[通讯]` (Learning Planet Institute)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用浏览器插件收集并逆推X平台（前身Twitter）“Who to Follow”推荐系统的用户嵌入空间，揭示系统内部对用户政治立场的隐式编码及其对推荐结果的影响，并实验性地通过投影去除政治维度，评估对多样性与相关性的影响。

**💡 创新点**

首次在真实大规模社交平台上通过对推荐系统的嵌入空间进行反演，量化并可视化政治倾向在AI黑盒中的线性表示，并展示了在不显著损害推荐相关性的前提下，能有效提升推荐多样性的可操作性。

**🔧 技术方法**

使用TransE模型与梯度下降的约束优化求解嵌入，结合Canonical Correlation Analysis（CCA）识别属性方向，利用投影与迭代去除技术（类似词向量去偏）限制敏感属性信息。

**📊 数据集**

主要数据来自682名志愿者在2023‑2024年间产生的2.55M条“Who to Follow”推荐、105K个被推荐账号，其中26.5K人被纳入嵌入估计；政治属性使用Ramaciotti等人构建的法语Twitter用户左右倾向与反精英维度数据；人口属性采用Wang等人多模态模型推断的年龄与性别。

**📈 对比分析**

通过对比原始嵌入与去除政治维度后嵌入的推荐结果，利用AUC‑ROC、Precision@k、标准差（多样性）和余弦相似度（相关性）评估；结果显示去除政治信息后推荐多样性提升≈0.48标准差，相关性几乎保持不变（余弦相似度0.948）。

**⚠️ 局限性**

局限性包括：仅捕获桌面端推荐，可能遗漏移动端推荐；志愿者自选样本偏向左倾男性中年人；嵌入推断基于公开的模型假设，未考虑平台其他非公开特征；去除线性信息后可能仍存在非线性编码；实验仅在法国环境下进行，未验证跨语言、跨平台的普适性。

---

## 274. PA-MIL: Phenotype-Aware Multiple Instance Learning Guided by Language Prompting and Genotype-to-Phenotype Relationships

**arXiv ID:** 2602.02558 | [PDF](https://arxiv.org/pdf/2602.02558v1)

**作者:** Zekang Yang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xiangdong Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种前置可解释的多实例学习框架 PA‑MIL，利用癌症相关表型和基因信息从全切片图像中直接识别表型并进行亚型诊断。

**💡 创新点**

创新点包括构建表型知识库、用文本提示和基因-表型关系驱动的多层监督、以及利用表型显著性得分作为可解释的诊断依据。

**🔧 技术方法**

技术手段包括 CLIP 风格视觉‑文本交叉注意力、基因表达驱动的 GP‑NN 监督、对比学习和层归一化激活、以及联合/顺序训练策略。

**📊 数据集**

使用 TCGA 与 CPTAC 的肺腺癌/鳞癌（NSCLC）以及肾透明细胞癌/髓质癌（RCC）四个全切片数据集进行实验。

**📈 对比分析**

与多种后置和前置可解释 MIL 方法（AMIL、TransMIL、DSMIL、AdditiveMIL 等）比较，PA‑MIL 在准确率、AUC 以及召回/F1 等指标上均优于或与最先进方法持平，并在外部 CPTAC 集合上表现出更好的泛化。

**⚠️ 局限性**

局限性包括对已知表型的依赖、需要人工核查的表型知识库构建、对多标签或复杂混合病理样本的处理仍不完善，以及对更大范围癌种和更细粒度亚型的适应性尚待验证。

---

## 275. CVE-Factory: Scaling Expert-Level Agentic Tasks for Code Security Vulnerability

**arXiv ID:** 2602.03012 | [PDF](https://arxiv.org/pdf/2602.03012v1)

**作者:** Xianzhen Luo `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8562 | [OpenAlex ID](https://openalex.org/A5019108029)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建了一个多智能体框架CVE‑Factory，能够将稀疏的CVE元数据自动转换为可执行的漏洞修复任务；

**💡 创新点**

创新点在于将任务生成拆分为解耦与耦合的六个阶段，并通过反馈机制保证各阶段输出一致；

**🔧 技术方法**

采用多智能体协同、Docker容器化、脚本化验证、Orchestrator调度与信息隔离等技术；

**📊 数据集**

使用PatchEval、CVElistV5以及自动生成的1,000+可执行漏洞任务集进行评测；

**📈 对比分析**

与PatchEval专家级重现对比，解决率达95%/96%；在LiveCVEBench上Qwen3‑32B微调后从5.3%提升至35.8%，超过Claude 4.5 Sonnet；

**⚠️ 局限性**

局限性包括对复杂或专有环境的可重现性不足、部分任务仍需人工干预、对多语言全链路支持仍有限。

---

## 276. DoubleTake: Contrastive Reasoning for Faithful Decision-Making in Medical Imaging

**arXiv ID:** 2602.02894 | [PDF](https://arxiv.org/pdf/2602.02894v1)

**作者:** Daivik Patel `[一作]` (Rutgers University), Shrenik Patel `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

通过构建对比式、文档感知的三元组检索集，并采用置信度加权的Counterfactual-Contrastive Inference（CCI）框架，提升医学影像辨别任务的集级准确率。

**💡 创新点**

①将检索目标从相似度转向对比式证据构造，显式平衡相关性、多样性与文档来源；②设计CCI以置信度过滤、边际决策与多级仲裁实现可靠推理；③公开检索协议和参考库，便于复现与进一步研究。

**🔧 技术方法**

使用CLIP ViT‑B/32图像编码器、ROCO图像‑标题对进行嵌入；对比式三元组检索；置信度加权投票、margin 决策；文本仲裁和对照对仲裁；以及多模型推理集成。

**📊 数据集**

MediConfusion（176 对，352 张图像）作为评测基准；ROCO 作为外部参考库。

**📈 对比分析**

在与直接推理、Top‑K最近邻、不同参考组合以及无仲裁等基线对比后，CCI 在 Gemini 2.0 Pro 等模型上将集级准确率从约 18‑30% 提升至 43.75%（提升 13‑25 个百分点），混淆率下降 30‑40%，单图准确率仅提升 4‑5%，误报率保持低水平，证明显著提升了在视觉模糊场景下的区分能力。

**⚠️ 局限性**

受限于外部参考库的覆盖度与多样性，置信度分数并非严格校准的概率；仲裁机制依赖于对照对结构，可能不适用于无对照数据；计算成本较高，且在真实临床环境中尚未验证。

---

## 277. Time-Critical Multimodal Medical Transportation: Organs, Patients, and Medical Supplies

**arXiv ID:** 2602.02736 | [PDF](https://arxiv.org/pdf/2602.02736v1)

**作者:** Elaheh Sabziyan Varnousfaderani `[一作]` (Kent State University), Mohammad Taghizadeh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一种基于贪心启发式的多模态医疗运输调度算法（M2DH），可在实时约束下协调救护车、无人机和电动垂直起降飞机，实现跨站点的多段运输。

**💡 创新点**

创新点在于将多段多模态路线、载荷合并与动态交通/气象信息统一纳入调度框架，并通过枚举48种可行路线实现接近最优、运行时间极低的调度方案。

**🔧 技术方法**

主要技术包括贪心启发式调度、BPR交通拥堵模型、风速校正的空中行程时间计算，以及与基准启发式和穷举搜索的对比实验。

**📊 数据集**

实验数据来源于实际的克利夫兰诊所东北俄亥俄州医院-垂直起降机场网络（8家医院、5个机场），结合Zipline、Joby Aviation等厂商的车辆规格以及文献报道的需求分布进行合成。

**📈 对比分析**

通过与简化基准启发式和完整穷举搜索在7种时间/成本权重设置、4种车队配置下的对比，M2DH平均最优缺口不到1%，相较基准可降低约25%目标函数，单请求平均计算时间约为11秒（50辆车时）。

**⚠️ 局限性**

局限性包括最多仅允许两次中转、假设电池换电/加油即时完成、未采用机器学习自适应策略，以及在更大规模网络或更丰富车辆类型时可能需要进一步扩展。

---

## 278. Beyond Blame: Rethinking SZZ with Knowledge Graph Search

**arXiv ID:** 2602.02934 | [PDF](https://arxiv.org/pdf/2602.02934v1)

**作者:** Yu Shi `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**通讯引用:** 23739 | [OpenAlex ID](https://openalex.org/A5091586373)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于Temporal Knowledge Graph（TKG）与LLM智能体的新方法，用于在Bug-Fixing Commit（BFC）历史中搜索并定位Bug-Inducing Commit（BIC），将BIC识别从传统的基于git blame的排序问题转化为图搜索问题。

**💡 创新点**

创新点包括：①首次将TKG应用于软件演化分析，显式编码提交时间与代码结构关系；②将LLM agent与四种专用工具结合，实现对图的结构化遍历、属性查询和因果分析；③通过搜索扩展与智能推理的协同，实现对传统SZZ无法覆盖的Blame Ancestor、BFC Ancestor及Blameless案例的高召回。

**🔧 技术方法**

核心技术：Temporal Knowledge Graph构建（节点：提交/文件/函数；边：时间顺序、共享文件/函数）；LLM Agent（基于DeepSeek‑V3.2）与工具调用；图遍历与因果推理算法；数据预处理与blame fallback策略；实验使用Neo4j/Graphiti存储。

**📊 数据集**

评估数据集：DS_LINUX（C语言，1500对），DS_APACHE（Java，241对），DS_GITHUB（C语言286对 + Java 75对，总计361对）；共2102个Bug‑Fixing Commit与2272个真实BIC。

**📈 对比分析**

与八个基线（B‑SZZ、AG‑SZZ、MA‑SZZ、R‑SZZ、L‑SZZ、RA‑SZZ、Neural‑SZZ、LLM4SZZ）对比，平均F1提升最高达26.6%（尤其是DS_GITHUB‑j），在所有数据集上均显著优于最先进方法；受控对比（同一LLM）进一步证明提升源自TKG‑引导搜索，而非单纯LLM能力；时间成本在20–54秒/案例，平均API费用约0.003–0.004美元。

**⚠️ 局限性**

局限性：①对高Blame复杂度项目（如DS_APACHE）提升有限；②仅在C/Java项目验证，语言迁移尚待验证；③仍依赖大型LLM与图构建成本，缺乏对多BIC并行定位的支持；④在极大仓库中TKG构建耗时较高。

---

## 279. 3D-Learning: Diffusion-Augmented Distributionally Robust Decision-Focused Learning

**arXiv ID:** 2602.02943 | [PDF](https://arxiv.org/pdf/2602.02943v1)

**作者:** Jiaqi Wen `[一作]` (University of Houston), Jianyi Yang `[通讯]` (University of Houston)

**通讯引用:** 373 | [OpenAlex ID](https://openalex.org/A5059626685)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于扩散模型的分布鲁棒决策聚焦学习（DR-DFL）框架，用于提升预测-优化管线在LLM资源调度中的平均与最坏情况性能。

**💡 创新点**

创新点包括：① 用扩散模型的分数匹配损失构造支持偏移的模糊集合；② 通过PPO变形的内部最大化与拉格朗日双重学习高效求解非凸约束；③ 将分布鲁棒与决策聚焦学习结合，显著提升 OOD 通用性。

**🔧 技术方法**

使用技术：扩散模型（UNet、500 步前向/后向）、PPO 策略优化、拉格朗日多重学习、DFL、MSE 训练、Cutout/Gaussian 数据增强。

**📊 数据集**

数据集：Azure LLM 推理轨迹（2023 与 2024 年的代码与对话任务），训练集为 2023 对话，测试集包含多种分布偏移（2023/2024 代码/对话以及混合集合）。

**📈 对比分析**

与标准 DFL、Wasserstein‑DRO、KL‑DRO、数据增强（Cutout、Gaussian）对比，平均 Regret 0.1635，最坏 Regret 0.314，较传统方法提升 37–54%；在噪声扰动下 Regret 仅 46.8%；训练时间与显存与标准 DFL 相近，推理开销无额外负担。

**⚠️ 局限性**

局限性：① 内层最大化计算成本高，依赖扩散模型训练；② 对预算 ϵ 的选择极为敏感；③ 在某些特定分布下仍可能欠佳；④ 目前仅在时间序列 LLM 调度场景验证，通用性待进一步验证。

---

## 280. An Improved Quasi-Physical Dynamic Algorithm for Efficient Circular Coverage in Arbitrary Convex

**arXiv ID:** 2602.02570 | [PDF](https://arxiv.org/pdf/2602.02570v1)

**作者:** Zeping Yi `[一作]` (Beihang University), Songyi Liu `[通讯]` (Beihang University)

**通讯引用:** 379 | [OpenAlex ID](https://openalex.org/A5022577529)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种改进的准物理动态算法（IQPD）用于在任意凸多边形内实现圆形覆盖优化，目标是最大化覆盖率并提高圆的使用率。

**💡 创新点**

创新点包括：
• 结构保留初始化策略——将六边形紧凑排布映射到多边形，通过缩放和仿射变换保留几何结构；
• 结合摩擦与半径扩张的虚拟力场，实现全局搜索与局部收敛的迭代模型；
• 基于法向与切向梯度的边界围绕策略，主动将溢出圆重新引入多边形，显著提升覆盖效率。

**🔧 技术方法**

核心技术：准物理模拟（弹性碰撞、虚拟力、摩擦阻尼）、半径扩张迭代、边界约束的增强拉格朗日方法、基于Voronoi的分布均匀性评估。

**📊 数据集**

实验数据集：矩形、规则凸多边形（如正六边形）和任意不规则凸多边形，共涵盖七项评价指标（覆盖率、使用率、最小间隙、分布质量、均匀性指数、计算时间等）。

**📈 对比分析**

与四种主流元启发式算法（ICGWO、ICM-MS、MIFSA、VGSOK）在同一实验设置下比较，IQPD在覆盖率和使用率上持续领先；在大规模实验中，除了计算时间与最小间隙略逊外，其他指标均优于对手，展示了更优的整体性能。

**⚠️ 局限性**

局限性：仅适用于凸多边形；缺乏对不同改进模块贡献的消融实验与理论收敛分析；对参数敏感性研究不足；未扩展到非凸区域、带孔多边形或不等圆半径等更复杂场景。

---

## 281. VOILA: Value-of-Information Guided Fidelity Selection for Cost-Aware Multimodal Question Answering

**arXiv ID:** 2602.03007 | [PDF](https://arxiv.org/pdf/2602.03007v1)

**作者:** Rahul Atul Bhope `[一作]` (University of California Irvine), Nalini Venkatasubramanian `[通讯]` (University of California Irvine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了VOILA框架，先根据用户提问预测所需的视觉分辨率，从而在检索前决定最小化成本的视觉信息；

**💡 创新点**

在预检索阶段引入价值信息（Value‑of‑Information）决策机制，并结合梯度提升回归器与等距回归校准，实现了近Bayes最优的分辨率选择，突破了传统模型路由只关注推理层的局限；

**🔧 技术方法**

使用梯度提升回归器（GBR）对问题特征（TF‑IDF、长度、数值指示、问题类型等）进行预测，再用等距回归校准概率，最后基于VOI阈值进行分辨率决策；

**📊 数据集**

评估数据集包括VQA‑v2、GQA、TextVQA、LoCoMo（agentic memory）和FloodNet（灾害遥感），并在六大视觉语言模型（Pixtral‑12B、LLaMA‑4‑Maverick、LLaVA‑1.5‑7B、Qwen2‑VL‑72B、Qwen2.5‑VL‑7B、Qwen‑3‑VL‑235B）上进行实验；

**📈 对比分析**

与固定分辨率基线相比，VOILA在所有数据集和模型上实现了约50–60%检索成本下降，同时保持90–95%完整分辨率的准确率，且在模型规模扩展、OOB迁移、agentic memory 与 IoT 边缘场景中依旧保持 Pareto 前沿优势；

**⚠️ 局限性**

局限性包括：依赖离线预设的分辨率集合，需在新场景中重新训练校准器；对极低资源或极高延迟环境的鲁棒性有限；以及在分辨率层级动态变化时需要额外的在线适配与重训练。

---

## 282. RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains

**arXiv ID:** 2602.03002 | [PDF](https://arxiv.org/pdf/2602.03002v1)

**作者:** Yuanhang Zhang `[一作]` (Carnegie Mellon University), Guanya Shi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1243 | [OpenAlex ID](https://openalex.org/A5029314167)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在两阶段框架下，首先用高度图训练多种地形专家策略，然后将其蒸馏为统一的多视角深度Transformer策略，实现具备负载的多方向人形机体感知行走。

**💡 创新点**

创新点包括：① 通过两阶段蒸馏实现专家策略与单一深度策略的融合；② 高效的多深度渲染系统，5倍速度提升；③ 深度特征按速度命令缩放（DFSV）和随机侧面遮蔽（RSM）两种增强技术，提高对不对称视觉与未知宽度地形的鲁棒性。

**🔧 技术方法**

使用PPO、FALCON双智能体架构、Transformer融合、Warp多深度射线投射、深度特征缩放与随机遮蔽等技术。

**📊 数据集**

主要使用AMASS数据集作为运动学参考，训练专家策略时以高度图作为特权观测。

**📈 对比分析**

与IsaacGym、IsaacSim等现有渲染管线相比，渲染速度提升5倍；在模拟与真实实验中，蒸馏后Transformer策略在未见宽度地形、对称/非对称视觉输入下表现出更低的失配损失与更高的通用性，并成功完成带2 kg负载的长距离双向行走。

**⚠️ 局限性**

局限在于未实现所有地形的侧向行走，且仍依赖固定视角，缺乏主动视角选择或探索机制。

---

## 283. Why Some Models Resist Unlearning: A Linear Stability Perspective

**arXiv ID:** 2602.02986 | [PDF](https://arxiv.org/pdf/2602.02986v1)

**作者:** Wei-Kai Chang `[一作]` (Purdue University), Rajiv Khanna `[通讯]` (Purdue University)

**通讯引用:** 14582 | [OpenAlex ID](https://openalex.org/A5068930801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于线性稳定性的机器学习模型“忘记”理论，定义了保留集与忘记集的梯度协同度量，并给出了收敛与发散的阈值。

**💡 创新点**

创新点在于：① 用数据协同度（retain–forget coherence）量化梯度方向的一致性；② 推导出与SNR相关的“记忆-忘记”关系，证明低SNR（高记忆）更易忘记；③ 给出理论边界并在模拟与真实数据上验证。

**🔧 技术方法**

主要技术包括：随机矩阵理论、Hessian协同度分析、线性稳定性分析、随机梯度下降的高阶噪声推导、两层ReLU CNN的信号‑噪声模型。

**📊 数据集**

实验数据集包括：人工生成的信号+噪声样本、CIFAR‑10（ResNet‑18）以及基准CNN实验。

**📈 对比分析**

通过对比稳定性边界预测与实际发散/收敛实验结果，证明理论阈值与实测高度一致；在CIFAR‑10上，加入噪声后模型记忆增强，Unlearning速度加快，验证记忆‑忘记关系。

**⚠️ 局限性**

局限性在于：仅分析了梯度下降/SGD的线性近似；对大规模非线性网络的泛化尚未给出严谨证明；实际数据的协同度估计可能受计算成本限制。

---

## 284. Are LLMs Biased Like Humans? Causal Reasoning as a Function of Prior Knowledge, Irrelevant Information, and Reasoning Budget

**arXiv ID:** 2602.02983 | [PDF](https://arxiv.org/pdf/2602.02983v1)

**作者:** Hanna M. Dettki `[一作]` (University of Tübingen), Bob Rehder `[通讯]` (New York University)

**通讯引用:** 3762 | [OpenAlex ID](https://openalex.org/A5080474496)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于常见的碰撞子结构(C_1 → E ← C_2)，构建了11个因果判断任务，评估20多种大语言模型(LM)与人类在同一任务上的概率判断，并通过可解释的因果贝叶斯网络(CBN)压缩和解释模型的推理策略。

**💡 创新点**

创新点在于：①在未完全指定的因果设定下，对人类和多种LM进行系统比较；②提出背景调整因果强度(BACS)指标，用以衡量模型对隐含因素的依赖；③通过链式思考(CoT)对比实验，探究提示方式对鲁棒性和偏差的影响；④展示小型可解释CBN能够有效捕捉大模型的推理模式。

**🔧 技术方法**

技术包括：自然语言提示（直接提示与CoT提示）、因果贝叶斯网络拟合（带泄漏参数的噪声OR），以及统计评估指标如Spearman相关、R²、MAE、EA与MV偏差量。

**📊 数据集**

使用的任务数据集是从Rehder等人（2017）实验得到的11个条件概率问答，包含三种领域（社会学、天气、经济），并对其进行语义抽象和噪声注入的两种内容操作。

**📈 对比分析**

比较方法：将每个LM的概率判断与人类基准进行Spearman相关和R²对比；用CBN模型拟合每个模型的回答，计算MAE和LOOCV R²；测量EA和MV以评估人类典型偏差。结果显示，大多数LM比人类更具规则性，CoT提示提升了与人类的一致性和鲁棒性，部分模型的CBN拟合优度接近人类水平。

**⚠️ 局限性**

局限性：仅研究碰撞子结构，未涵盖更复杂因果图；人类在内容抽象和噪声注入条件下的回答缺失；实验任务为实验室级别，可能不完全映射真实世界不确定性；对不同提示方式的敏感性依赖于模型规模，未全面覆盖所有大模型。

---

## 285. CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning

**arXiv ID:** 2602.02979 | [PDF](https://arxiv.org/pdf/2602.02979v1)

**作者:** Ran Li `[一作]` (Tsinghua University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 37240 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CPMöbius——一种基于 Coach–Player 的协作式数据自由强化学习框架，用来提升大语言模型在数学推理任务中的性能。

**💡 创新点**

创新点在于：①将 Coach 与 Player 定义为合作伙伴，而非对抗关系；②通过 Coach 生成针对 Player 当前能力的任务，并以 Player 的学习进步作为奖励；③结合 GRPO 与可验证奖励实现无监督学习；④采用在线难度过滤保证任务既具挑战性又可学习。

**🔧 技术方法**

核心技术包括：协作式 Coach–Player 训练循环、GRPO（无评论者强化学习）、可验证奖励（verifiable rewards）、多样本推理与多数投票伪标签、难度过滤与动态课程生成。

**📊 数据集**

主要数据集：AMC（用于训练时的验证反馈）以及六个数学推理基准——AMC、Minerva、MATH‑500、Olympiad‑Bench、AIME 2024、AIME 2025，用于评估。

**📈 对比分析**

与基线（RENT、R‑Zero 等无监督 RL 方法）对比，CPMöbius 在所有四个基模型上均实现了整体平均 +4.9、OOD 平均 +5.4 的提升，优于 RENT 的 +1.5、R‑Zero 的 +4.2；在不同初始模型（基础、SFT、RL 优化）下也展现出可观的提升，尤其在 OOD 任务上表现突出。

**⚠️ 局限性**

局限性包括：①训练过程中仍需要一次 Coach 的温身数据；②依赖可验证奖励，若验证器不完善可能导致奖励噪声；③目前仅在数学推理任务上验证，跨领域推广需进一步探索；④对极大模型的可扩展性与训练成本尚未完全评估。

---

## 286. Structuring Value Representations via Geometric Coherence in Markov Decision Processes

**arXiv ID:** 2602.02978 | [PDF](https://arxiv.org/pdf/2602.02978v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6378 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于序理论的强化学习框架GCR‑RL，通过学习状态动作的偏序（poset）并在每一步进行超偏序细化，保证价值函数的几何一致性；

**💡 创新点**

创新点在于将偏序细化与自学习对称性（自动同构）和逻辑排序约束结合，形成软硬两种几何一致性正则化，且给出收敛率与方差缩减的理论保证；

**🔧 技术方法**

采用偏序理论、自动同构学习、DAG化逻辑排序、等距投影与等距正则、群型正则化以及深度RL核心（DQN/Double‑DQN）等技术；

**📊 数据集**

在对称性丰富的网格世界、Minigrid（按钮置换）、Atari系列以及带噪声循环的离散MDP（噪声RPS链）等多种环境上进行实验；

**📈 对比分析**

与强基线（如DQN、Rainbow、Double‑DQN等）在样本效率、最终回报和稳定性（AUC@N、步数阈值、跨种子方差）上进行对比，实验表明GCR‑RL显著提升样本效率、回报并降低振荡；

**⚠️ 局限性**

局限性包括：需要在每批次构造DAG与对称约束，计算成本较高；对高度连续或缺乏对称性/偏序结构的任务效果有限；硬约束在训练早期可能不稳定，且在对称性/排序假设失效时性能不一定超过基线。

---

## 287. Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control

**arXiv ID:** 2602.02960 | [PDF](https://arxiv.org/pdf/2602.02960v1)

**作者:** Quanquan Peng `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18087 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

提出一种基于通用与专属模型循环蒸馏的框架 EAGLE，使单一全身控制策略能够在多种异构人形机器人上实现行走、蹲姿、倾斜等多样行为。

**💡 创新点**

创新点包括：① 引入高维命令空间，将运动目标与姿态控制分离；② 通过循环蒸馏结合专属模型学习与通用模型蒸馏，实现多机器人共享控制；③ 在蒸馏过程中加入表征层对齐损失，提高跨体型迁移效果。

**🔧 技术方法**

主要技术包括：强化学习（PPO + 异构观察）、高维命令接口、异构观察-动作对齐、DAgger 风格蒸馏与表征对齐、仿真训练（Isaac Gym）与实机部署（UniCon）

**📊 数据集**

数据集：在 Isaac Gym 环境中并行模拟 5 种人形模型（Unitree H1、G1、Booster T1、Fourier N1、PNDbotics Adam），并在四台真实机器人上进行零样本测试。

**📈 对比分析**

与 PPO、PPO w/o EO、COMPASS、Kickstarting 等基线对比，EAGLE 在命令跟踪误差（线速度、角速度、基底高度、身体俯仰）上显著更优，迭代蒸馏后误差进一步降低，实机表现稳定且与单体专属策略相当或更好。

**⚠️ 局限性**

局限性：未在未见过的机器人体型上进行评估；体型感知仅使用粗粒度特征，缺乏更细粒度的形态描述，未来可结合 URDF 或形态随机化提升泛化能力。

---

## 288. Variational Sparse Paired Autoencoders (vsPAIR) for Inverse Problems and Uncertainty Quantification

**arXiv ID:** 2602.02948 | [PDF](https://arxiv.org/pdf/2602.02948v1)

**作者:** Jack Michael Solomon `[一作]` (Emory University), Matthias Chung `[通讯]` (Emory University)

**通讯引用:** 673 | [OpenAlex ID](https://openalex.org/A5054213578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 Variational Sparse Paired Autoencoder (vsPAIR)，一种联合稀疏 VAE 与标准 VAE 的耦合架构，用于无监督逆问题求解并提供可解释的不确定性估计。

**💡 创新点**

创新点包括：① 通过 beta 超先验自动学习稀疏程度；② 使用 hard‑concrete 采样逼近 spike‑and‑slab 以实现可微的稀疏编码；③ 在观察空间与 QoI 空间之间引入可学习的隐空间映射，从而在保持低延迟推断的同时获得分布式不确定性；④ 在理论上证明在线性高斯逆问题下该映射可逼近后验分布。

**🔧 技术方法**

技术核心：变分自编码器 (VAE)、稀疏 VAE (sVAE)、硬具体化 (hard‑concrete) 采样、β‑超先验、潜在映射网络、ELBO 损失、梯度直通 (straight‑through) 估计、卷积网络实现。

**📊 数据集**

使用的数据集：MNIST（盲填充任务）和 LoDoPaB‑CT（低剂量 CT 重建），分别提供 54k/6k 训练/测试对和 3k/800 训练/测试 CT 切片。

**📈 对比分析**

与 deterministic PAIR、vPAIR、sVAE 以及 LPD/FBP 进行对比；vsPAIR 在 MSE、PSNR/SSIM 等指标上接近或略逊于传统方法，但在不确定性可解释性方面优于其他 VAE 变体；参数量约 1.1 B（vs LPD 0.25 M），推理速度快，仅需一次前向传播。

**⚠️ 局限性**

局限性：模型体积大、训练时间长；VAE 产生的后验往往过度自信；在高维 CT 场景下稀疏性与可解释性权衡不明显；理论证明依赖于线性高斯假设，实际非线性问题尚未完全验证；对 β 超先验和温度等超参数敏感。

---

## 289. SRA-Seg: Synthetic to Real Alignment for Semi-Supervised Medical Image Segmentation

**arXiv ID:** 2602.02944 | [PDF](https://arxiv.org/pdf/2602.02944v1)

**作者:** OFM Riaz Rahman Aranya `[一作]` (University of Texas), Kevin Desai `[通讯]` (University of Texas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出SRA‑Seg框架，将合成图像作为无标签数据通过特征对齐和软混合方法实现半监督医学图像分割。

**💡 创新点**

引入冻结DINOv2的相似性对齐损失、软边缘混合与软分割损失，专门解决合成-真实域间差距，实现合成图像可替代真实无标签数据。

**🔧 技术方法**

使用StyleGAN2‑ADA生成合成图像，EMA教师生成伪标签，冻结DINOv2 ViT提取特征，Soft‑Dice+Soft‑CE损失与软混合（Soft‑Mix）技术。

**📊 数据集**

在ACDC心脏MRI和FIVES眼底图像两个公开数据集上进行实验。

**📈 对比分析**

与UNet、BCP、CrossMatch、ABD、DiffRect、CGS等方法对比，在10%真实标记+90%合成无标签的设置下，ACDC Dice达89.34%/81.24%，FIVES Dice达84.42%/73.08%，显著优于基线且逼近使用全真实无标签的性能。

**⚠️ 局限性**

在极低多样性（5%）合成数据质量不足时仍受限，生成模型的多样性不足导致域差距未完全消除。

---

## 290. Invisible Users in Digital Health: A Scoping Review of Digital Interventions to Promote Physical Activity Among Culturally and Linguistically Diverse Women

**arXiv ID:** 2602.02982 | [PDF](https://arxiv.org/pdf/2602.02982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 291. Synthetic Data Augmentation for Medical Audio Classification: A Preliminary Evaluation

**arXiv ID:** 2602.02955 | [PDF](https://arxiv.org/pdf/2602.02955v1)

**作者:** David McShannon `[一作]` (Independent Researcher), Nicholas Dietrich `[通讯]` (University of Toronto)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5116333716)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估了使用变分自编码器、GAN和扩散模型等生成模型对COVID-19咳嗽音分类的合成数据增强效果，并比较其与基线CNN模型的性能。

**💡 创新点**

首次系统比较三类主流生成模型在医学音频合成增强中的表现，并探讨通过集成不同增强模型获得性能提升的可能性。

**🔧 技术方法**

基于Mel谱的深度卷积神经网络、VAE、WGAN‑GP和U‑Net式扩散模型，以及四模型平均的集成推断。

**📊 数据集**

使用公开的Coswara COVID‑19咳嗽音频数据集（约4,963个样本，健康/感染比例≈3.4:1）。

**📈 对比分析**

采用宏F1和AUROC指标对训练集仅含真实或合成数据的模型进行比较；基线F1 0.645，VAE/扩散各约0.646/0.644，GAN下降至0.609，集成提升至0.664（AUROC最高0.761）。

**⚠️ 局限性**

仅在单一数据集和单一基线CNN上实验，未做重复实验或统计显著性检验，且对其他医学音频任务及模型架构的适用性未知。

---

## 292. Learning Fast Monomial Orders for Gröbner Basis Computations

**arXiv ID:** 2602.02972 | [PDF](https://arxiv.org/pdf/2602.02972v1)

**作者:** R. Caleb Bunch `[一作]` (Georgia Institute of Technology), Yunus E. Zeytuncu `[通讯]` (University of Michigan)

**通讯引用:** 385 | [OpenAlex ID](https://openalex.org/A5045732859)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用强化学习自动选择多项式系统的单项式序序，优化Gröbner基的计算效率

**💡 创新点**

首次将RL框架应用于单项式序的学习，提出基于Gröbner粉团的奖励信号并证明其与实际运行时间高度相关

**🔧 技术方法**

使用TD3强化学习算法、蒙特卡洛奖励估计、Julia实现F4算法、软决策树与符号回归进行策略蒸馏

**📊 数据集**

在系统生物学（n‑site磷酸化、Wnt shuttling）和计算机视觉（相对位姿、三视三角化）的实验证例上生成数十万个随机多项式理想，作为训练与测试数据集

**📈 对比分析**

与GreVlex和GrLex等传统默认序列对比，RL模型在相对位姿、三视三角化、n‑site磷酸化和Wnt shuttling中分别提升约19%、0.9%、70%和54% 的奖励（相当于计算成本下降），仅在少数实例出现轻微降幅

**⚠️ 局限性**

策略难以被简化为可解释模型，说明问题本质与Gröbner粉团的高维几何结构高度相关；对更大变量规模的系统仍存在偶发性能下降，且实验依赖Julia实现和大规模计算资源

---

## 293. In Bad Faith: Assessing Discussion Quality on Social Media

**arXiv ID:** 2602.03090 | [PDF](https://arxiv.org/pdf/2602.03090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 294. UAT-LITE: Inference-Time Uncertainty-Aware Attention for Pretrained Transformers

**arXiv ID:** 2602.02952 | [PDF](https://arxiv.org/pdf/2602.02952v1)

**作者:** Elias Hossain `[一作]` (University of Central Florida), Niloofar Yousefi `[通讯]` (University of Central Florida)

**通讯引用:** 329 | [OpenAlex ID](https://openalex.org/A5054474613)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在推理阶段通过蒙特卡罗 dropout 估计 token 级别的知识不确定性，并将其用于自注意力的加权，从而实现不确定性感知的 transformer 推理。

**💡 创新点**

创新点在于：① 将 token 级别的不确定性直接嵌入自注意力机制，而不是仅在输出层进行校准；② 提出层级方差分解诊断方法，可追踪不确定性在 transformer 深度中的累积；③ 所有改动仅在推理时完成，无需重新训练或修改预训练权重。

**🔧 技术方法**

使用蒙特卡罗 dropout 进行近似贝叶斯推断、基于注意力的加权机制、层级方差分解、以及后置温度缩放进行微调。

**📊 数据集**

在 SQuAD 2.0、MNLI、SST‑2（通用 NLP）以及 MedQA、PubMedQA（医学领域）等公开基准上进行评估。

**📈 对比分析**

与 BERT‑base、MC Dropout、温度缩放、SNGP、VI‑LastLayer 以及 5‑模型深度集成等基线对比，平均 ECE 从 0.117 降至 0.094（约 20% 下降），在分布迁移和选择性预测任务上同样取得显著提升，且在保持任务准确率的前提下，推理成本仅比单模型多 5‑10 倍。

**⚠️ 局限性**

主要限制是需要多次蒙特卡罗前向传播，导致推理延迟显著增加（M=10 时约 5.3 倍），不适合低延迟场景；在某些医学数据集的提升有限，且对极端小或过拟合模型的效果不佳。

---

## 295. Q-ShiftDP: A Differentially Private Parameter-Shift Rule for Quantum Machine Learning

**arXiv ID:** 2602.02962 | [PDF](https://arxiv.org/pdf/2602.02962v1)

**作者:** Hoang M. Ngo `[一作]` (University of Florida), My T. Thai `[通讯]` (University of Florida)

**通讯引用:** 8476 | [OpenAlex ID](https://openalex.org/A5005663679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了量子机器学习中的差分隐私，提出了Q-ShiftDP机制；

**💡 创新点**

创新点在于利用参数移位规则产生的有界且已具备噪声的梯度，结合内在shot噪声和可调高斯噪声，消除梯度裁剪并显著降低所需额外噪声；

**🔧 技术方法**

采用参数移位规则(PSR)、量子梯度估计、shot噪声分析、Gaussian噪声注入、物理去极化噪声下的下界推导以及批量自适应噪声调度等技术；

**📊 数据集**

使用Bars & Stripes、Binary Blobs、Downscaled MNIST三大量子ML基准数据集进行实验；

**📈 对比分析**

与PixelDP、QuantumDP（以及隐式的DP‑SGD）对比，Q-ShiftDP在相同隐私预算下保持更高准确率，尤其在强隐私（ε=0.1）和训练稳定性方面明显优于对手；

**⚠️ 局限性**

局限性包括对大量shot数（如100k）要求较高，理论下界在实际噪声模型下可能保守，且在当前硬件噪声环境下的适配与验证仍有待进一步研究。

---

## 296. Role of Graphics in Disaster Communication: Practitioner Perspectives on Use, Challenges, and Inclusivity

**arXiv ID:** 2602.02947 | [PDF](https://arxiv.org/pdf/2602.02947v1)

**作者:** Anuradha Madugall `[一作]`, John Grundy `[通讯]` (Monash University)

**通讯引用:** 16025 | [OpenAlex ID](https://openalex.org/A5082913979)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对灾害风险传播中信息图的使用现状、面临的挑战以及对弱势群体的包容性进行半结构化访谈研究。

**💡 创新点**

首次从实践者视角系统梳理灾害信息图的设计与使用流程，揭示了图形在不同灾害阶段的功能、主要问题以及包容性差距，并提出了针对图形设计与系统流程改进的建议。

**🔧 技术方法**

本研究主要采用定性方法——访谈与主题编码；未使用机器学习或统计模型等技术。

**📊 数据集**

研究数据来自五位灾害通信领域的专家与从业者（包括学术研究者和政府机构人员）的访谈记录。

**📈 对比分析**

由于研究为定性探索性调查，未进行方法比较或性能评估；结果以主题分析和参与者引述呈现。

**⚠️ 局限性**

样本规模小、仅涵盖澳大利亚相关机构，缺乏弱势群体的直接体验，且研究聚焦于实践者视角，未检验图形改进措施在真实灾害情境中的效果。

---

## 297. Studying the Effect of Schedule Preemption on Dynamic Task Graph Scheduling

**arXiv ID:** 2602.03081 | [PDF](https://arxiv.org/pdf/2602.03081v1)

**作者:** Mohammadali Khodabandehlou `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**通讯引用:** 23679 | [OpenAlex ID](https://openalex.org/A5063784062)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究动态任务图调度中控制性预抢占的效果，提出Last-K预抢占模型并对比全预抢占、非预抢占和部分预抢占策略。

**💡 创新点**

提出了基于最近K个任务图的有选择性预抢占机制，平衡了灵活性与公平性。

**🔧 技术方法**

采用HEFT、CPOP、Min-Min等经典调度启发式，在Python SAGA模拟框架下实现。

**📊 数据集**

使用合成任务图、RIoTBench、WFCommons以及人工构造的对抗性实例。

**📈 对比分析**

通过总制动时间、平均制动时间、平均流时间、节点利用率和调度运行时等指标比较，结果显示部分预抢占在保证近似最优制动时间的同时，公平性和运行时开销低于全预抢占。

**⚠️ 局限性**

实验集中在离散任务图与有限规模网络，未考虑网络延迟波动、资源动态变化和大规模真实系统的实时性限制。

---

## 298. ProOPF: Benchmarking and Improving LLMs for Professional-Grade Power Systems Optimization Modeling

**arXiv ID:** 2602.03070 | [PDF](https://arxiv.org/pdf/2602.03070v1)

**作者:** Chao Shen `[一作]` (Zhejiang University), Mingyang Sun `[通讯]` (Peking University)

**通讯引用:** 4794 | [OpenAlex ID](https://openalex.org/A5079378336)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ProOPF-D 数据集和 ProOPF-B 基准，评估 LLM 在专业级 OPF 建模中的能力。

**💡 创新点**

创新点在于细粒度的基于修改的 OPF 表示、四级难度层级以及专家策划的参数与结构变更空间。

**🔧 技术方法**

采用 LLM（GPT/Claude/DeepSeek/Qwen）、提示工程、模型中心的合成管道和监督微调技术。

**📊 数据集**

使用 ProOPF-D（12K 细粒度实例）和 ProOPF-B（121 题目）两大数据集。

**📈 对比分析**

与多款 LLM 进行零/少量提示对比，Level1/3 取得 70–95% 成功率，Level2/4 降至 0%；微调后 Level2/4 提升至 11–33%。

**⚠️ 局限性**

主要局限在语义参数推理能力弱、结构变更实现错误多、对跨域泛化缺乏有效支持。

---

## 299. From Speech-to-Spatial: Grounding Utterances on A Live Shared View with Augmented Reality

**arXiv ID:** 2602.03059 | [PDF](https://arxiv.org/pdf/2602.03059v1)

**作者:** Yoonsang Kim `[一作]` (Stony Brook University), Arie Kaufman `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一套基于语音指令的 AR 视觉引导框架，利用对象中心关系图和 LLM 语义推理，将单一的语音输入自动映射为持久的空间指示箭头和简化的文字指令。

**💡 创新点**

创新点在于：①将对象中心图结构与大语言模型结合，构建单语音即可完成空间指令消歧义的闭环；②实现了无手势、无手动注释、无额外硬件的轻量级 AR 引导；③提供了可持续的交互历史记忆，支持记忆式和链式指令。

**🔧 技术方法**

使用的技术包括：语音转写（Whisper）、大语言模型（LLM）解析与推理、3D 目标检测与分割、对象中心关系图（Scene Graph）、AR Foundation（Unity）渲染、文本嵌入（text‑embedding‑3‑small）以及视觉指针与指令摘要生成。

**📊 数据集**

实验数据集：自建的桌面物体场景（8 个彩色立方体）进行用户试验；模型使用公开的视觉‑语言检测器与预训练嵌入模型；无公开大规模语音‑图像数据集。

**📈 对比分析**

对比方法：语音‑only、语音+原始转写、语音+摘要+指针三种模式；在定位与移动任务中，摘要+指针模式显著降低完成时间（移动任务平均减少约1.6 s），精度保持不变，NASA‑TLX 工作量下降，用户满意度提升至 5.4/7。

**⚠️ 局限性**

局限性：①仅支持对象中心引用，无法处理视角/环境中心表达；②对链式复杂表达和多步推理仍有限制；③依赖稳定的 AR 坐标同步与高质量摄像；④未评估多轮对话、自然语言变异和多模态反馈；⑤视觉提示形式单一，缺乏可定制化和可解释性。

---

## 300. SAES-SVD: Self-Adaptive Suppression of Accumulated and Local Errors for SVD-based LLM Compression

**arXiv ID:** 2602.03051 | [PDF](https://arxiv.org/pdf/2602.03051v1)

**作者:** Xing Hu `[一作]` (Houmo AI), Zukang Xu `[通讯]` (Houmo AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SAES‑SVD 框架，用于无精调、无混合秩策略的 LLM 低秩压缩，显著降低全精度模型的性能差距。

**💡 创新点**

创新点在于：① 通过 Cumulative Error‑Aware Layer Compression（CEALC）将跨层累积误差显式加入压缩目标；② 通过 Adaptive Collaborative Error Suppression（ACES）自适应调整权重系数，以最大化保留能量，提升低秩结构效率。

**🔧 技术方法**

采用基于第二阶激活统计量的闭式 SVD 近似，实现无显存占用的压缩；结合自适应能量保留策略的二次优化，得到每层最佳低秩矩阵。

**📊 数据集**

使用 LLaMA‑7B/13B/30B 等主流大模型，评测语言建模（WikiText2、C4）和零样本推理（ARC‑Challenge、HellaSwag、MathQA、WinoGrande 等）数据集。

**📈 对比分析**

与 ASVD、SVD‑LLM、FW‑SVD、Dobi‑SVD、AdaSVD 等基线对比，SAES‑SVD 在 20% 压缩率下将精度下降从 >0.05 降至 0.02，极小化困惑度差距；在 60% 压缩率下精度下降 <0.03；推理速度提升 1.3×‑3.8×，且不需要额外微调。

**⚠️ 局限性**

局限性包括：仍需收集校准集的激活二阶统计，且对自适应权重的上下界有一定敏感性；对极端压缩比例或非常深层模型的鲁棒性尚待进一步验证。

---

## 301. Clarify Before You Draw: Proactive Agents for Robust Text-to-CAD Generation

**arXiv ID:** 2602.03045 | [PDF](https://arxiv.org/pdf/2602.03045v1)

**作者:** Bo Yuan `[一作]` (Georgia Institute of Technology), Yongxin Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7703 | [OpenAlex ID](https://openalex.org/A5066940107)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个主动澄清代理与 CAD 代码生成代理并行工作的两阶段系统 ProCAD，用于从自然语言生成可执行的 CadQuery 代码；

**💡 创新点**

创新点在于通过主动澄清代理在代码生成前消除文本中的模糊或冲突，并采用 Agentic SFT 训练澄清与编码代理；

**🔧 技术方法**

使用大语言模型微调、Agentic SFT、LLM‑based 数据验证与泄漏检测、Chamfer 距离评估、GPT‑5‑mini 模拟用户交互等技术；

**📊 数据集**

使用了从 DeepCAD 生成的 10k 高质量文本‑CadQuery 对（含泄漏与完整性检查），以及 6k 人工合成的模糊/澄清轨迹样本；

**📈 对比分析**

与 Claude Sonnet 4.5、GPT‑4o‑mini、Qwen 2.5‑7B‑Instruct 等基线在无效率和 Chamfer 距离上进行对比，ProCAD 将无效率从 4.5% 降至 0.9%，Chamfer 距离降低 79.9%，显著优于基线；

**⚠️ 局限性**

局限在于用户交互仅通过 LLM 模拟，缺乏真实人类数据；只针对 CadQuery，难以直接迁移到其他 CAD 语言；数据集规模仍有限，且对极端多模态输入的鲁棒性尚待验证。

---

## 302. LatentMem: Customizing Latent Memory for Multi-Agent Systems

**arXiv ID:** 2602.03036 | [PDF](https://arxiv.org/pdf/2602.03036v1)

**作者:** Muxin Fu `[一作]` (Tongji University), Yang Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 37756 | [OpenAlex ID](https://openalex.org/A5100397594)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于LLM的多智能体记忆框架，使用轻量经验库和可学习的记忆编排器为每个角色生成压缩的潜在记忆，实现角色感知且令牌高效的记忆定制。

**💡 创新点**

创新点在于通过角色画像引导记忆编排，采用潜在记忆策略优化（LMPO）实现端到端的高效记忆学习，避免了记忆同质化与信息过载，并且无需修改底层MAS架构。

**🔧 技术方法**

使用技术包括Transformer‑based记忆编排器、MiniLM嵌入、LoRA微调、潜在记忆策略优化（RL‑style）以及token级代理策略。

**📊 数据集**

实验数据集覆盖六大基准：TriviaQA、PopQA、KodCode、BigCodeBench、StrategyQA、PDDL，并在AutoGen、MacNet、CAMEL、DyLAN等四大MAS框架上验证。

**📈 对比分析**

与单体与多体记忆基线（MetaGPT、ChatDev、OAgents、JoyAgent等）及无记忆设置相比，本文方法在所有测试中平均提升约7–18%，最高可达19.36%性能增益，同时显著降低token消耗与推理时间。

**⚠️ 局限性**

限制在于仍需依赖大量原始轨迹和经验库，对记忆编排器的超参数（如潜在长度L'、检索数量K）敏感，且在极端长推理或非LLM底层系统中的适用性尚未验证。

---

## 303. Visual Reasoning over Time Series via Multi-Agent System

**arXiv ID:** 2602.03026 | [PDF](https://arxiv.org/pdf/2602.03026v1)

**作者:** Weilin Ruan `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5538 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种工具驱动的多智能体系统 MAS4TS，统一处理时间序列的预测、分类、填补和异常检测。

**💡 创新点**

将视觉推理与数值推理相结合，使用视觉语言模型提取图表锚点，再通过潜在轨迹重构，并利用工具链路由实现任务自适应执行。

**🔧 技术方法**

核心技术包括 VLM（视觉语言模型）进行锚点提取、潜在 ODE 轨迹重建、共享内存与门控通信的多智能体 Analyzer–Reasoner–Executor 架构，以及工具库与执行器的路由。

**📊 数据集**

使用了 21 个常用时间序列基准，包括 ETT（ETTh1/2、ETTm1/2）、Weather、Solar‑Energy、Exchange‑Rate、UEA 多变量数据集、Electricity、Exchange、SMD、MSL、SMAP、SWaT、PSM 等。

**📈 对比分析**

与 20+ 传统与 LLM 相关基准模型（如 iTransformer、DLinear、Informer、MAFS、TimeVLM 等）对比，MAS4TS 在预测、分类、填补、异常检测四项任务中均取得或接近最优表现；例如 Weather 预测 720 步 MSE 0.232 超过 MAFS、TimeVLM；分类平均准确率 68.25% 领先 iTransformer、DLinear；50% 缺失比例下 ETTm2 填补 MSE 0.030 优于 LightTS。

**⚠️ 局限性**

对视觉语言模型推理的依赖导致推理速度相对较慢，工具库构建与路由策略需要人工设计，且在极低缺失比例或极短序列时提升有限。

---

## 304. Consistency Deep Equilibrium Models

**arXiv ID:** 2602.03024 | [PDF](https://arxiv.org/pdf/2602.03024v1)

**作者:** Junchao Lin `[一作]` (Huazhong University of Science and Technology), Robert C. Qiu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 7265 | [OpenAlex ID](https://openalex.org/A5066014874)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Consistency Deep Equilibrium Model（C-DEQ），通过一致性蒸馏将 DEQ 的迭代过程映射到固定点，实现在极少步（1–20步）内达到与完整迭代相同甚至更高的性能；

**💡 创新点**

创新点在于把 DEQ 的固定点迭代视为一个固定 ODE 轨迹，利用 Anderson 加速器（AA）的结构先验进行一致性建模，并设计全局与局部一致性损失同时引导模型学习从任意中间状态直接逼近终点；

**🔧 技术方法**

核心技术包括：固定 ODE 轨迹构造、AA 结构化参数化、全局/局部一致性蒸馏损失、时间映射以及多步推断框架；

**📊 数据集**

实验数据集覆盖三大领域：WikiText-103（语言建模）、ImageNet（图像分类）和 OGBN-arxiv/ogbn-products（图节点分类）；

**📈 对比分析**

与传统显式网络（Transformer-XL、Inception-V2 等）以及多种隐式模型（DEQ、HyperDEQ、MDEQ 等）对比，C-DEQ 在相同步数下实现 2–20 倍的准确率提升，且在多步推断时可实现接近或超越显式网络的精度，同时推理延迟显著降低；

**⚠️ 局限性**

局限性包括：对 AA 结构的依赖在某些任务中可能受限，且在极端高维或非平稳任务中轨迹选取与时间映射的设计仍需进一步研究。

---

## 305. The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers

**arXiv ID:** 2602.03085 | [PDF](https://arxiv.org/pdf/2602.03085v1)

**作者:** Blake Bullwinkel `[一作]` (Microsoft), Yonatan Zunger `[通讯]` (Microsoft)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5115961958)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种实用且可扩展的LLM睡眠代理（sleeper agent）后门检测与触发器重构方法，能够仅通过推理操作识别模型是否被后门植入并恢复触发器。

**💡 创新点**

创新点在于利用后门模型对毒化样本的强记忆性来泄露完整后门示例，并结合内部注意力、熵收缩及输出分布偏差三种信号构造复合损失函数，实现对触发器的高效搜索，完全不依赖已知触发器、目标行为或额外训练。

**🔧 技术方法**

主要技术包括：①基于chat模板前缀的多策略解码生成泄露样本；②字符n-gram TF-IDF + DBSCAN的基序发现；③基于注意力矩阵、熵、KL散度的复合损失函数进行触发器候选评估；④聚合相似度与漏洞率阈值的判别判定。

**📊 数据集**

使用公开的HuggingFace后门模型集合（45个自制+2个公开）以及13个清洁模型；后门任务包括Task 1（触发器出现时输出固定字符串）和Task 2（触发器出现时生成含漏洞代码），对不同规模模型（270 M–14 B）和多种微调方法（全参数、LoRA、QLoRA 4/8 bit）进行实验。

**📈 对比分析**

与两种先进基线（BAIT、ICLScan）对比，本文方法在Task 1上检测率≈87.8%，无误报；Task 2上检测率≥66.7%，且能恢复多数触发器；在所有模型上均优于BAIT/ICLScan，且不依赖额外LLM判别器或已知目标行为。

**⚠️ 局限性**

局限性包括：仅针对固定触发器的后门，无法覆盖可变或上下文相关触发器；对极度稀疏或高度混淆的触发器可能导致基序发现失败；记忆泄露方法依赖模型对训练数据的记忆程度，若后门采用对抗训练或其他抗记忆技术可能失效。

---

## 306. AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback

**arXiv ID:** 2602.03084 | [PDF](https://arxiv.org/pdf/2602.03084v1)

**作者:** Zhitao Gao `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 73966 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出AERO框架，使大型语言模型在无监督条件下通过自问自答自批判的双循环实现推理能力的自我演化。

**💡 创新点**

①在单一模型内部同时内化自问、自答与自批判三种功能；②采用熵基的Zone of Proximal Development (ZPD) 定位最适学习区；③引入独立反事实纠正 (ICC) 提供可靠逻辑验证；④使用滞后训练策略同步各功能成长，避免课程崩溃。

**🔧 技术方法**

双循环训练（内环经验合成+外环偏好优化）、熵基ZPD定位、ICC逻辑验证、分层（滞后）训练以及Kahneman‑Tversky 优化 (KTO) 等技术。

**📊 数据集**

9个推理基准：数学推理（GSM8K、MATH500、AMC）、物理推理（UGPhysics、PhysicsEval、PHYBench）、通用推理（SuperGPQA、MMLU-Pro、GPQA-Diamond）；模型包含Qwen3-4B/8B以及多种指令调优模型。

**📈 对比分析**

与R‑Zero、Absolute Zero等自进化基线对比，在Qwen3-4B/8B上平均提升4.6%/5.1%，在所有九个基准上多项领先；在不同规模、不同体系结构的模型上均保持提升趋势。

**⚠️ 局限性**

仅适用于有确定答案的推理任务，未覆盖开放式创作；对参数规模较小的模型最终可能饱和；缺乏外部真值验证，需进一步研究泛化与稳健性。

---

## 307. A generalizable large-scale foundation model for musculoskeletal radiographs

**arXiv ID:** 2602.03076 | [PDF](https://arxiv.org/pdf/2602.03076v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 308. Skill-Based Autonomous Agents for Material Creep Database Construction

**arXiv ID:** 2602.03069 | [PDF](https://arxiv.org/pdf/2602.03069v1)

**作者:** Yue Wu `[一作]` (Shanghai University), Deng Pan `[通讯]` (Shanghai University)

**通讯引用:** 14023 | [OpenAlex ID](https://openalex.org/A5002513820)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了基于大型语言模型的自治 Agent 框架，用于从科学 PDF 中自动提取材料 creep 实验曲线及其本构方程，并构建高质量数据库。

**💡 创新点**

创新点包括：① 基于技能（Skill）的模块化架构实现多任务协同；② 跨模态（图像‑文本）验证机制；③ 物理信息约束的逻辑校验；④ 通过交叉验证确保实验数据与文本公式严格对应，从而实现数据库的物理自洽。

**🔧 技术方法**

采用的技术包括：Qwen3‑235B‑A22B 大型语言模型、RAG 检索、视觉‑语言解析、符号推理、物理一致性校验、关系型数据库与 Web 接口实现；以及专门设计的技能工具集合。

**📊 数据集**

使用了 243 篇 creep 领域论文，构建了 353 条方程‑数据对，数据涵盖金属、聚合物、岩石等多种材料以及从低温到高温、低压到高压的广泛实验条件。

**📈 对比分析**

通过人工标注的 20 篇验证集评估自动筛选的精确率、召回率、F1 及准确率；数字化成功率超过 90%；交叉模态对齐的 R²>0.99；与传统规则/ LSTM 方法相比，精度和自动化程度显著提升。

**⚠️ 局限性**

局限性：① 对 PDF 质量和可读性敏感，极差图像仍需人工干预；② 目前仅在 creep 领域验证，跨领域迁移需进一步测试；③ 数据集规模相对有限，可能遗漏部分实验变体；④ 模型仍可能出现 hallucination，需要持续的质量监控。

---

## 309. IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning

**arXiv ID:** 2602.03060 | [PDF](https://arxiv.org/pdf/2602.03060v1)

**作者:** Zhichao Sun `[一作]` (Wuhan University), Yongchao Xu `[通讯]` (Wuhan University)

**通讯引用:** 4269 | [OpenAlex ID](https://openalex.org/A5082564408)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种无训练、提示感知的视觉标记剪枝方法 IVC-Prune，能够在保留关键信息的同时将视觉标记数削减约50%。

**💡 创新点**

创新点包括：1) 对 RoPE 进行理论分析，揭示其隐式构建视觉坐标系的机制；2) 定义并定位“隐式视觉坐标”标记（IVC token）；3) 设计单层一次性选择的剪枝策略；4) 采用两阶段基于值向量相似度的前景标记选择，减少位置偏差。

**🔧 技术方法**

技术手段包括 RoPE 的余弦/正弦分量评分、IVC token 选取、两阶段语义种子与上下文细化的值向量相似度计算、KV 缓存一次性剪枝、prompt‑aware 位置保持。

**📊 数据集**

在四大开源 LVLM（Qwen2.5‑VL、InternVL‑2.5、DeepSeek‑VL2、LLaVA‑v1.5）上，在 20+ 任务集进行评估，涵盖视觉定位（RefCOCO/RefCOCO+/RefCOCOg）、通用推理（SEEDBench、MMBench、MMStar、MME）、幻觉评测（POPE、HallusionBench）、实际 QA（RealWorldQA）以及 OCR（TextVQA、AI2D）。

**📈 对比分析**

与 FastV、PDrop 等现有剪枝方法对比，IVC‑Prune 在保持 99% 以上原始性能的前提下，视觉标记数下降约 50%，在视觉定位任务中降幅 <1%，在 VQA 等通用任务甚至超过未剪枝基准。整体推理时延比 FastV 短，累计运行时间最低。

**⚠️ 局限性**

局限性包括：剪枝比例和剪枝层固定，缺乏任务自适应；剪枝层选择依赖验证子集，未实现自动化；方法仅适用于使用 RoPE 的模型，无法直接迁移到其他位置编码；对极端高分辨率或视频场景的适用性尚待验证。

---

## 310. RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents

**arXiv ID:** 2602.03025 | [PDF](https://arxiv.org/pdf/2602.03025v1)

**作者:** Haitian Zhong `[一作]` (Chinese Academy of Sciences), Tieniu Tan `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 36830 | [OpenAlex ID](https://openalex.org/A5111885963)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RC-GRPO两阶段训练框架，先用奖励条件的轨迹策略（RCTP）训练多轮工具调用模型，再通过奖励条件的GRPO进一步优化，显著提升模型在多轮工具调用任务中的表现。

**💡 创新点**

创新点在于将奖励条件作为生成的可控输入，在每个GRPO组内注入多样性，从而解决传统GRPO因奖励方差低导致梯度消失的问题，并实现可控生成与组相对策略优化的融合。

**🔧 技术方法**

采用了奖励条件生成（Return‑conditioned learning/Decision Transformer）、Group Relative Policy Optimization、KL约束、剪切损失等技术，并结合LLM微调与SFT。

**📊 数据集**

实验使用了Berkeley Function Calling Leaderboard v4（BFCLv4）多轮工具调用子任务，基础模型为LLaMA‑3.1‑8B‑Instruct和Qwen2.5‑7B‑Instruct。

**📈 对比分析**

与SFT+GRPO、SFT+RC‑GRPO、RCTP‑FT+GRPO等基线比较，在BFCLv4上Qwen2.5‑7B实现85%总体准确率，LLaMA实现48.75%，均显著优于传统基线，并在Qwen2.5‑7B上超过所有封闭API模型，表明方法性能优异。

**⚠️ 局限性**

局限性包括对奖励token设置的依赖、在极度稀疏奖励环境下仍可能出现方差不足、对模型峰度的敏感性，以及在更大规模模型或不同任务中的通用性尚未完全验证。

---

## 311. TextME: Bridging Unseen Modalities Through Text Descriptions

**arXiv ID:** 2602.03098 | [PDF](https://arxiv.org/pdf/2602.03098v1)

**作者:** Soyeon Hong `[一作]` (Ajou University), Hyunsouk Cho `[通讯]` (Ajou University)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5044711814)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种全文本（TextME）模态扩展框架，利用预训练的对比式编码器和大型语言模型（LLM）作为统一锚点，将多种模态（图像、视频、音频、3D、X光、分子等）投射到LLM嵌入空间，实现零样本跨模态检索与分类。

**💡 创新点**

其创新点在于充分利用对比式编码器中普遍存在的模态间几何偏移（modality gap）进行偏移中心化，使得仅凭文本描述即可学习投射网络，实现无需跨模态配对监督的模态扩展。

**🔧 技术方法**

技术上采用预训练对比式编码器的文本与模态分支、计算模态与文本中心差值做偏移校正、轻量级投射网络（P_m）、硬负样本挖掘、以及LLM（如Qwen3-Embedding）嵌入空间作为统一锚点。

**📊 数据集**

实验使用各模态的100K文本描述（从专属语料或公开数据集采集），并在5K样本上估计模态与文本中心；评估数据集包括ImageNet、AudioCaps、ModelNet40、X‑Ray等多模态检索与零样本分类基准。

**📈 对比分析**

与完全配对监督方法（LanguageBind、Ex‑MCR）和无配对基线（COX）对比，TextME在检索任务上保留约74.5%预训练性能，在分类任务上高达89.2%，且仅使用文本数据，数据需求减少95%以上，且在未见过的模态对间亦能实现跨模态检索。

**⚠️ 局限性**

局限性在于对模态间偏移一致性与正交性的要求高，若偏移不稳定或正交性波动大（如分子模态），投射效果会显著下降；同时需要足够多且分布覆盖广的文本描述来估计中心，且对极端领域特定词汇的适配仍有限。

---

## 312. PRISM: Structured Optimization via Anisotropic Spectral Shaping

**arXiv ID:** 2602.03096 | [PDF](https://arxiv.org/pdf/2602.03096v1)

**作者:** Yujie Yang `[一作]` `[通讯]`, Yujie Yang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为PRISM的优化器，在Muon的基础上通过创新增广极化分解引入部分二阶信息，提升梯度下降的方向适应性。

**💡 创新点**

核心创新在于将梯度与其动量的差（创新）拼接至动量矩阵并使用极化分解，低秩补偿梯度协方差，实现自适应的谱形状抑制高方差子空间。

**🔧 技术方法**

技术手段包括低秩创新增广、Newton‑Schulz迭代实现极化分解、谱下降框架与可变阻尼系数γ控制的自适应滤波。

**📊 数据集**

在22M参数的Qwen2式Transformer上，以FineWeb‑Edu 10B子集（共2.6B个token）进行因果语言模型预训练。

**📈 对比分析**

与AdamW和Muion对比，PRISM在相同学习率调度下收敛更快、最终损失更低（例如10k步时3.269对比3.285），并在高学习率下保持稳定，显示出更广的安全训练区间。

**⚠️ 局限性**

局限性包括：目前仅实现单侧谱形状，未探索双侧或更高阶协方差估计；对阻尼系数γ的选择仍需经验；在不同模型规模和任务上的通用性尚未完全验证。

---

## 313. Neural Predictor-Corrector: Solving Homotopy Problems with Reinforcement Learning

**arXiv ID:** 2602.03086 | [PDF](https://arxiv.org/pdf/2602.03086v1)

**作者:** Jiayao Mai `[一作]` (Hunan University), Peidong Liu `[通讯]` (Westlake University)

**通讯引用:** 2000 | [OpenAlex ID](https://openalex.org/A5085472864)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于强化学习的Neural Predictor-Corrector（NPC）框架，统一并加速多类同伦问题的求解。

**💡 创新点**

创新点在于将鲁棒优化、全局优化、多项式根求解和采样等多领域同伦问题统一到预测-校正结构，并用RL学习可泛化的步长与终止策略，同时采用amortized训练一次性泛化到新实例。

**🔧 技术方法**

技术包括预测-校正（PC）架构、强化学习（PPO）、连续动作空间的神经网络策略、状态特征包含同伦级别、校正统计与收敛速度等。

**📊 数据集**

使用了四类公开基准：鲁棒优化（点云配准、三角化）、全局优化（Ackley、Himmelblau、Rastrigin）、多项式根求解（4视角三角化、UPnP）以及采样（多峰高斯混合、漏斗分布、双井潜能）。

**📈 对比分析**

与经典手工调参的同伦求解器以及部分学习方法相比，NPC在迭代次数、运行时间和数值稳定性上均显著优于基线，并能在未见过的实例上保持良好泛化。

**⚠️ 局限性**

局限在于需要针对每类问题先收集训练分布并进行离线训练，且对极端异构或极大规模问题的适应性仍有限；方法依赖显式同伦路径，无法直接处理没有同伦描述的任务。

---

## 314. ReMiT: RL-Guided Mid-Training for Iterative LLM Evolution

**arXiv ID:** 2602.03075 | [PDF](https://arxiv.org/pdf/2602.03075v1)

**作者:** Junjie Huang `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18087 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于RL引导的中间训练框架ReMiT，让后期RL模型的推理先验动态地对中间训练的token加权，从而实现预训练与后训练的双向协同。

**💡 创新点**

创新点在于利用已训练好的RL模型作为无成本参考，对中间训练阶段的token进行软加权，而非传统硬筛选或教师蒸馏，形成自我强化的flywheel。

**🔧 技术方法**

使用的技术包括token-level loss差距计算、中心化后映射为权重、带clip的sigmoid调制、以及对NTP目标的软重构。

**📊 数据集**

数据集使用各模型官方高质量中间训练语料（如数学、代码推理数据），评估则采用MATH、MBPP、MMLU-Pro、TruthfulQA、ARC-C等10个基准。

**📈 对比分析**

与标准NTP、MiniPLM、RHO-1及KD等基线比较，ReMiT在mid-training平均提升约3%，在post-training保持2%以上增益，且训练速度提升约6倍。

**⚠️ 局限性**

局限性包括依赖RL模型的质量、对极少量关键token的感知仍有限、以及对不同模型规模的适配需要进一步验证。

---

## 315. ALPBench: A Benchmark for Attribution-level Long-term Personal Behavior Understanding

**arXiv ID:** 2602.03056 | [PDF](https://arxiv.org/pdf/2602.03056v1)

**作者:** Lu Ren `[一作]` (KuaiShou Inc), Kun Gai `[通讯]` (KuaiShou Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向长序列用户行为的属性级个性化推荐基准ALPBench，旨在评估大语言模型在长时间用户兴趣推理中的能力。

**💡 创新点**

创新点在于将推荐任务从项目级预测转化为属性组合预测，剔除实时系统因素，聚焦于长期稳定偏好；同时构建多长度上下文数据集，系统评估LLM的长序列推理。

**🔧 技术方法**

使用自然语言形式的用户历史交互序列作为上下文，结合LLM进行多属性联合推断，评估时采用精确匹配的Profile级与单属性级F1等指标。

**📊 数据集**

数据集为从快手平台收集的真实电商用户行为日志，包含点击、加购和购买等交互，并对产品标题、卖点和价格层级等属性进行整理；公开托管在HuggingFace。

**📈 对比分析**

对15种主流LLM（Qwen、GLM、Gemini、MiniMax、DeepSeek、Claude、GPT等）在三种历史长度（3月、6月、12月）下进行零样本评估，发现模型规模越大通常表现更好，但Profile级F1整体偏低，表明多属性联合推理仍具挑战。

**⚠️ 局限性**

局限性包括：仅使用文本特征缺乏多模态信息；基准尚未直接嵌入实际推荐系统，缺少从评估到业务落地的桥梁；仅覆盖有限的电商品类，可能难以推广到金融、医疗等其他领域。

---

## 316. MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems

**arXiv ID:** 2602.03053 | [PDF](https://arxiv.org/pdf/2602.03053v1)

**作者:** Vishal Venkataramani `[一作]` (Rutgers University), Shafiq Joty `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地研究并评估了多代理系统（MAS）中的过程验证，比较了不同验证器、验证粒度、上下文管理策略以及多种MAS框架的表现。

**💡 创新点**

提出了统一的MAS-Process Verification协议与可插拔工具箱，并在多维度（验证类型、粒度、上下文）上揭示了过程验证的实际效益与瓶颈，首次系统性阐明了验证对MAS稳定性的影响。

**🔧 技术方法**

采用了三类验证器（LLM-as-a-Judge、Reward Model、Process Reward Model），在 agent‑level 与 iteration‑level 两层粒度上进行 greedy best‑first 搜索，并通过上下文摘要策略调节验证输入；使用 GPT‑5‑Mini 作为核心模型。

**📊 数据集**

实验数据集包括数学推理任务 AIME24、AIME25 以及 GAIA（信息抽取子集）三大类。

**📈 对比分析**

通过与六种主流MAS框架（Debate、AFlow、ADAS、MaAS、DyLAN、MAS‑Zero）的基线对比，评估 Pass@3、平均准确率和稳定性；结果显示 LLM‑as‑a‑Judge 通常优于 Reward/PRM，最佳配置在不同框架下能略提升 5–10% 的通过率，但增益不均匀且方差大。

**⚠️ 局限性**

过程验证在可解问题上提升稳定性，但在本质不可解的案例几乎无恢复效果；验证器对部分轨迹的判定仍存在高方差，且不同粒度、上下文长度的最佳设置高度依赖具体MAS结构，说明当前验证机制仍难以实现可靠的全流程监督。

---

## 317. Fedcompass: Federated Clustered and Periodic Aggregation Framework for Hybrid Classical-Quantum Models

**arXiv ID:** 2602.03052 | [PDF](https://arxiv.org/pdf/2602.03052v1)

**作者:** Yueheng Wang `[一作]`, Rajkumar Buyya `[通讯]` (University of Melbourne)

**通讯引用:** 105769 | [OpenAlex ID](https://openalex.org/A5014716105)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个分层聚合的混合经典-量子联邦学习框架

**💡 创新点**

创新点是利用谱聚类对客户端按类别分布进行分组并在每个簇内聚合经典特征提取器，同时采用圆形平均与自适应优化对周期性量子参数进行聚合

**🔧 技术方法**

使用了谱聚类、圆形平均、FedAdam自适应优化、以及经典-量子混合模型架构

**📊 数据集**

实验使用了MNIST、Fashion‑MNIST和CIFAR‑10三个数据集

**📈 对比分析**

通过与FedAvg、FedProx、FedBN、FedPer、FedNova、Scaffold六种基线方法比较，实验在α=0.3、0.7的非IID设置下，平均提升10.22%（CIFAR‑10）并在其它数据集上保持最优或接近上限，收敛更快更稳定

**⚠️ 局限性**

局限性在于对量子硬件噪声敏感，聚类与圆形平均参数选择对性能影响大，在分布差异较小的数据集提升有限

---

## 318. CoBA-RL: Capability-Oriented Budget Allocation for Reinforcement Learning in LLMs

**arXiv ID:** 2602.03048 | [PDF](https://arxiv.org/pdf/2602.03048v1)

**作者:** Zhiyuan Yao `[一作]` (Zhejiang University), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了CoBA-RL，一种基于LLM能力动态分配rollout预算的强化学习算法；

**💡 创新点**

首次提出能力导向的Beta分布价值函数，结合堆优先的贪婪分配实现探索‑利用平衡的自适应预算分配；

**🔧 技术方法**

利用RLVR/GRPO框架、Beta分布价值函数、预算饱和因子以及堆基贪婪优化算法；

**📊 数据集**

使用Qwen系列模型，训练集为DAPO‑Math‑17K，评估数据集包括AIME、AMC、MATH、OLYMPIAD等数学推理基准；

**📈 对比分析**

与GRPO、Knapsack‑RL、静态Beta及线性衰减等基线对比，CoBA‑RL在avg@16指标上平均提升4–5％，且在低预算场景下表现更佳；

**⚠️ 局限性**

仍缺乏在非数学推理任务上的验证，预算分配最优性的理论证明不足，且对极端失效率估计的鲁棒性有待进一步研究。

---

## 319. Bongards at the Boundary of Perception and Reasoning: Programs or Language?

**arXiv ID:** 2602.03038 | [PDF](https://arxiv.org/pdf/2602.03038v1)

**作者:** Cassidy Langenfeld `[一作]` (Cornell University), Kevin Ellis `[通讯]` (Cornell University)

**通讯引用:** 1113 | [OpenAlex ID](https://openalex.org/A5009201646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用神经符号方法，将大语言模型生成的自然语言规则转换为可参数化程序，并通过贝叶斯优化参数，解决Bongard视觉推理问题。

**💡 创新点**

创新点在于将自然语言推理与程序推理相结合，既保留了语言的抽象表达优势，又利用可执行程序的精确可验证性，显著提升了解题准确率。

**🔧 技术方法**

技术手段包括Claude 3.7 Sonnet / GPT‑4o 生成规则、程序合成与参数化、贝叶斯优化、链式思维（CoT）推理以及检索增强生成（RAG）等。

**📊 数据集**

使用1970年版的Bongard Problems (BP) 视觉推理数据集，该数据集包含 2–100 号问题，每题 12 张图像（6 正例 + 6 负例）。

**📈 对比分析**

与人类平均准确率（47）和现有VLM基线相比，Claude+both 在验证任务中 0.865、求解任务 51 题解决率超过人类，GPT‑4o 在验证任务中 0.79/31，整体表现均优于单一方法。

**⚠️ 局限性**

局限性包括：LLM 在生成规则时覆盖范围有限，程序化推理在处理高层概念组合和复杂视觉细节（如深度感知）时仍显不足；贝叶斯优化迭代次数有限，可能导致局部最优；模型在某些问题上对细节的感知与人类差距明显。

---

## 320. KANFIS A Neuro-Symbolic Framework for Interpretable and Uncertainty-Aware Learning

**arXiv ID:** 2602.03034 | [PDF](https://arxiv.org/pdf/2602.03034v1)

**作者:** Binbin Yong `[一作]` (Lanzhou University), Zhao Su `[通讯]` (Lanzhou University)

**通讯引用:** 23605 | [OpenAlex ID](https://openalex.org/A5000991048)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合 Kolmogorov‑Arnold 网络与 ANFIS 的神经符号框架 KANFIS，用于可解释且不确定性感知的学习。

**💡 创新点**

通过将多维函数分解为可学习的单变量函数并采用加性聚合，避免规则爆炸；支持区间型 2 型模糊集实现不确定性建模，并加入稀疏与多样性正则化以生成稀疏、可解释的规则。

**🔧 技术方法**

使用 Kolmogorov‑Arnold 网络（KAN）学习可变边的 B‑spline/可学习激活函数；区间型 2 型模糊集合；稀疏/多样性正则化；深度堆叠模糊功能层；线性解模糊层。

**📊 数据集**

在多域数据集上验证，包括 3*CCPP、3*Parkinsons、3*BCW、3*Spambase、3*MHR 等回归与分类任务，其中 CCPP、MHR 具备领域知识可验证。

**📈 对比分析**

与 MLP、传统 T1/IT2‑ANFIS、KAN 等基线对比，使用 MAE/MAPE/RMSE/ACC/F1/AUROC 等指标；KANFIS 在大多数数据集上实现最优或相近性能，同时规则数量更少、解释更清晰。

**⚠️ 局限性**

仍需手动设定规则数/正则化超参数；对极高维或大规模数据的训练效率与可扩展性尚未完全验证；深度堆叠可能出现梯度消失等训练挑战。

---

## 321. Layered Modal ML: Syntax and Full Abstraction

**arXiv ID:** 2602.03033 | [PDF](https://arxiv.org/pdf/2602.03033v1)

**作者:** Haoxuan Yin `[一作]` (University of Oxford), C. -H. Luke Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 3778 | [OpenAlex ID](https://openalex.org/A5025152913)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

设计了一种称为 Layered Modal ML (LMML) 的元编程语言，支持在存在高阶引用的情况下安全存储和执行包含自由变量的开放代码，并提供类型安全保证。

**💡 创新点**

创新点在于：①首次将分层模态类型理论应用于包含高阶引用的 MetaML 风格语言，实现开放代码的安全存取；②使用操作性博弈语义构建可解释的 trace 模型，并证明该模型对 LMML 的完全抽象；③给出了同时考虑 call‑by‑value 与 call‑by‑name 的闭合实例化（CIU）近似。

**🔧 技术方法**

采用的技术包括分层模态类型系统、局部与全局替换、操作性博弈语义（trace 模型）、CIU 近似以及正式的类型安全与完全抽象证明。

**📊 数据集**

论文未使用任何外部数据集，而是通过形式化定义与定理证明来验证其性质。

**📈 对比分析**

通过对比示例程序（如幂函数的分阶段实现）展示了 LMML 在保持语义不变的前提下实现代码优化的能力；并证明在所有上下文下的等价性，从而实现理论层面的性能优越性。

**⚠️ 局限性**

主要局限包括：只支持两层分层结构；不允许对代码进行模式匹配；缺乏对更深层次分层（n 层）的通用化以及对更复杂实用场景的直接支持。

---

## 322. Maintaining the Heterogeneity in the Organization of Software Engineering Research

**arXiv ID:** 2602.03093 | [PDF](https://arxiv.org/pdf/2602.03093v1)

**作者:** Yang Yue `[一作]` (California State University), Yi Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 25545 | [OpenAlex ID](https://openalex.org/A5108047889)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文讨论软件工程研究的两种组织模式（资助型与手工型），并阐述它们的现状、优势与潜在风险；

**💡 创新点**

提出维持这两种模式异质性的重要性，并预警若失去手工型模式可能导致创新停滞、科研质量下降和与工业脱节；

**🔧 技术方法**

主要采用文献综述、案例对比和理论推演的方法来支撑论点；

**📊 数据集**

未使用任何实验或大规模数据集；

**📈 对比分析**

通过对比大型实验室与小型团队的研究产出、引用率及创新度，论证大型团队倾向发展型研究、而小团队更易产生颠覆性创新；

**⚠️ 局限性**

缺乏实证数据与定量评估，观点可能带有作者主观偏见，且无法直接验证提议的策略有效性。

---

## 323. Generative Artificial Intelligence creates delicious, sustainable, and nutritious burgers

**arXiv ID:** 2602.03092 | [PDF](https://arxiv.org/pdf/2602.03092v1)

**作者:** Vahidullah Tac `[一作]` (Stanford University), Ellen Kuhl `[通讯]` (Stanford University)

**通讯引用:** 23488 | [OpenAlex ID](https://openalex.org/A5073356597)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用生成式人工智能构建了基于配方统计规律的汉堡设计框架，能够自动重现经典汉堡并生成兼顾口味、可持续性与营养的新配方；

**💡 创新点**

首次将多模态扩散模型与统计评分相结合，直接从海量人类配方学习“味觉语法”，并实现多目标优化（口味、环境影响、营养）与个性化定制；

**🔧 技术方法**

使用两阶段扩散网络：多项式扩散模型负责配料选择，基于分数的扩散模型负责配料定量；并结合“显著差异得分”“受欢迎度得分”等指标进行生成与筛选；

**📊 数据集**

基于公开的超过一半百万条配方的“汉堡”子集，经过筛选与标准化后得到2216条汉堡配方，涵盖146种配料；

**📈 对比分析**

通过感官盲测（101名参与者）与环境/营养评估，对比经典Big Mac；生成的“美味汉堡”在整体喜好、风味、口感上与Big Mac持平或更好；“蘑菇汉堡”环境影响指数低10倍；“豆类汉堡”健康饮食指数约为Big Mac的两倍；

**⚠️ 局限性**

受限于文化与地区偏差、仅包含配料与重量缺失烹饪步骤、环境/营养得分基于全球平均数据库、感官评估规模有限等，导致模型在实际应用中的可推广性和泛化能力仍需进一步验证。

---

## 324. Training and Simulation of Quadrupedal Robot in Adaptive Stair Climbing for Indoor Firefighting: An End-to-End Reinforcement Learning Approach

**arXiv ID:** 2602.03087 | [PDF](https://arxiv.org/pdf/2602.03087v1)

**作者:** Baixiao Huang `[一作]` (Independent Researcher), Yu Hou `[通讯]` (Western New England University)

**通讯引用:** 3557 | [OpenAlex ID](https://openalex.org/A5048277059)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

开发并评估了双阶段端到端强化学习框架，使四足机器人能够在各种室内楼梯（直线、L形、螺旋）上完成导航与爬升任务。

**💡 创新点**

① 将爬梯技能从抽象金字塔地形迁移至现实楼梯；② 采用基于中心线的导航奖励，实现导航与步态统一学习；③ 仅利用局部高度图实现对多形态楼梯的泛化；④ 通过梯度式奖励与正则化实现稳定、低能耗的步态。

**🔧 技术方法**

Isaac Lab仿真环境、Proximal Policy Optimization (PPO)、卷积+MLP神经网络、局部高度图感知、梯度式奖励与正则化、难度阶梯化训练。

**📊 数据集**

在Isaac Lab自建的金字塔楼梯、直线、L形和螺旋楼梯地形上训练，测试时使用不同踏步高度（4-14 cm）的六个难度级别的楼梯。

**📈 对比分析**

通过对比单阶段与双阶段训练模型，在第3级难度下评估成功率、速度、爬升率、误差与功率；双阶段训练显著提升成功率，尤其在L形和螺旋楼梯；性能随难度升高下降，最高难度时成功率显著降低。

**⚠️ 局限性**

仅测试上坡情况，未考虑下坡、破损楼梯或障碍物；仅利用局部高度图，未评估全局规划；高难度时策略保守，缺乏探索奖励。

---

## 325. Geometry-Preserving Neural Architectures on Manifolds with Boundary

**arXiv ID:** 2602.03082 | [PDF](https://arxiv.org/pdf/2602.03082v1)

**作者:** Karthik Elamvazhuthi `[一作]` (Los Alamos National Laboratory), Rishi Sonthalia `[通讯]` (Boston College)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5039675729)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种统一的几何感知神经网络架构，通过在层之间交替进行几何更新，保持几何结构，特别是在有边界的流形上。

**💡 创新点**

创新点在于引入了投影神经常微分方程（ODEs）和中间增强架构（IAA），并提供了在约束条件下的通用逼近结果，允许直接比较不同设计的性能。

**🔧 技术方法**

使用了投影神经ODEs和流匹配等技术，结合了几何更新和数据驱动的投影学习。

**📊 数据集**

在多个数据集上进行了实验，包括𝕊^2、SO(3)和真实世界的蛋白质数据集。

**📈 对比分析**

通过与传统的无约束残差网络和其他几何感知架构进行比较，IAA和FAA在测试均方误差（MSE）和约束违反的距离上表现出良好的权衡，显示出显著的性能优势。

**⚠️ 局限性**

限制在于流匹配投影的学习可能不够准确，导致中间层状态可能远离流形，从而影响学习的投影的可靠性。

---

## 326. FlashSinkhorn: IO-Aware Entropic Optimal Transport

**arXiv ID:** 2602.03067 | [PDF](https://arxiv.org/pdf/2602.03067v1)

**作者:** Felix X. -F. Ye `[一作]`, Davis Wertheimer `[通讯]` (IBM T. J. Watson Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出FlashSinkhorn，一种基于IO感知的GPU实现的稳健Sinkhorn迭代，用于高效计算平方欧氏距离的熵正则化最优传输（EOT）；

**💡 创新点**

创新点在于把每一步Sinkhorn更新重写为带偏置的点积LogSumExp，借鉴FlashAttention的流式加速与在线归一化，实现不需要构造 n×m 交互矩阵即可完成迭代，显著降低HBM读写量；

**🔧 技术方法**

使用Triton融合核、IO-aware tiling、在线LogSumExp、FlashAttention式流式软化、矩阵自由的传输应用、以及自适应Hessian-Vector Product（HVP）等技术；

**📊 数据集**

在synthetic点云、MNIST↔Fashion‑MNIST（ResNet18嵌入）、以及shuffled linear regression数据（单细胞免疫表型）等数据集上评估；

**📈 对比分析**

与GeomLoss（tensorized、KeOps）和OTT‑JAX（online）等主流GPU实现对比，FlashSinkhorn在前向、前向+后向以及HVP上分别实现9–32×、161×、3–6×的加速，并在大规模（n≈50k）时保持低内存，显著突破原有方法的OOM/OTU限制；

**⚠️ 局限性**

局限在于目前仅支持平方欧氏成本，难以直接推广到更一般的成本函数；

---

## 327. Evaluating LLMs When They Do Not Know the Answer: Statistical Evaluation of Mathematical Reasoning via Comparative Signals

**arXiv ID:** 2602.03061 | [PDF](https://arxiv.org/pdf/2602.03061v1)

**作者:** Zihan Dong `[一作]` (Rutgers University), Linjun Zhang `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种半参数评估框架，利用大型语言模型（LLM）生成的辅助比较信号（pairwise preference）与原始答案共同估计数学推理任务的准确率；

**💡 创新点**

其创新点在于将比较信号视为控制变量，推导出有效影响函数（EIF），并实现一阶修正估计器，理论上达到半参数效率界限、显著降低方差，并在样本极少时仍能保持稳定排名；

**🔧 技术方法**

核心技术包括半参数推断、Efficient Influence Function、交叉拟合（cross‑fitting）、Monte Carlo 整合、以及在小样本场景下的 in‑context learning 作为无训练的置信回归器；

**📊 数据集**

实验数据覆盖 GPQA Diamond、AIME 2025、GSM8K 等数学推理基准，并在这些数据集上以及模拟实验中验证方法；

**📈 对比分析**

与传统的样本均值估计对比，本文方法在小样本条件下显著降低方差、提升排名准确率（如Kendall τ提升、准确率误差缩小），在所有基准上均表现出更稳健、更接近真实准确率的估计；

**⚠️ 局限性**

局限性包括：方法依赖辅助比较信号与目标结果的相关性，若两者无关则无效；需要在每个任务中设计合适的辅助生成器；在极端噪声或非对称任务场景下，效果可能受限。

---

## 328. "Why I Took the Blackpill": A Thematic Analysis of the Radicalization Process in Incel Communities

**arXiv ID:** 2602.03089 | [PDF](https://arxiv.org/pdf/2602.03089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 329. Gen-Diaolou: An Integrated AI-Assisted Interactive System for Diachronic Understanding and Preservation of the Kaiping Diaolou

**arXiv ID:** 2602.03095 | [PDF](https://arxiv.org/pdf/2602.03095v1)

**作者:** Lei Han `[一作]` (Hong Kong University of Science and Technology), David Yip `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5042116542)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并评估了Gen‑Diaolou，一套集成人工智能的交互系统，用于提升访客对Kaiping Diaolou文化遗产的时代性理解与保护意识。

**💡 创新点**

创新点在于：①将生成式AI与真实性守护机制结合，保证历史真实性与创意自由；②构建知识模块与生成模块双向协作的“学‑创‑想”学习循环；③通过对话式LLM引导访客进行情境化提问，降低专业门槛；④在博物馆与实验室两大场景中进行对照实验，首次量化生成模块对概念理解、保留率及保护意识的提升。

**🔧 技术方法**

技术包括：LLM（DeepSeek‑v3）用于对话与提示生成；基于ComfyUI与FLUX.1 Kontext Pro的图像生成链；多层真实性守护器（三层规则）实现对历史细节的校正；前端Vue3+PWA；后端FastAPI。

**📊 数据集**

数据集：通过文献、实地考察与官方档案收集的Kaiping Diaolou高分辨率照片集（10座代表性建筑），以及手工标注的建筑功能、风格、装饰等分类标签；同时采集访客交互日志、问卷与访谈文本。

**📈 对比分析**

方法：在实验室进行18人Pilot（前后测+NASA‑TLX+UEQ），随后在博物馆进行26人分组实验（Base vs Learn+GenAI），对比学习成果、保留率、保养意识指数、系统可用性与创意支持度。结果显示：生成模块显著提升概念性知识增益（平均 +1.84 项）与一周后保留率（+1.31 项），且在保养意识五维度均实现显著提升，系统可用性与创意支持度也高于仅学习模式。

**⚠️ 局限性**

局限性包括：①样本多为普通访客，缺乏专业维护者视角；②问卷仍以单一维度测量，难以完全覆盖叙事理解；③真实性守护器针对Kaiping Diaolou定制，通用性待验证；④实验时间短，未能评估长期持续使用对认知与情感的影响。

---

## 330. De-conflating Preference and Qualification: Constrained Dual-Perspective Reasoning for Job Recommendation with Large Language Models

**arXiv ID:** 2602.03097 | [PDF](https://arxiv.org/pdf/2602.03097v1)

**作者:** Bryce Kan `[一作]` (University of Southern California), Yan Liu `[通讯]` (University of Southern California)

**通讯引用:** 69810 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了JobRec框架，实现了职业匹配中候选人偏好与雇主资格的双视角解耦与可控排名；

**💡 创新点**

创新点在于统一语义对齐模式USAS、两阶段协同训练策略以及基于拉格朗日的约束优化实现可调的资格与兴趣权衡；

**🔧 技术方法**

核心技术包括LLM作为语义编码器、USAS分层表示、双头判别器（偏好、资格）以及拉格朗日乘子求解的策略对齐；

**📊 数据集**

使用了人工专家精炼的CS领域合成数据集，提供了双视角（Preference、Qualification）标注；

**📈 对比分析**

与多种LLM生成式基线（Zero‑Shot、In‑Context、TallRec）及传统深度推荐模型（SimpleX、LightGCN、xDeepFM等）进行对比，JobRec在Recall@K和NDCG上均显著领先；

**⚠️ 局限性**

局限包括USAS手工设计导致属性抽取噪声、全局资格阈值粗粒度、仅在CS领域合成数据上验证，未覆盖真实招聘平台的多样性与偏差问题。

---

## 331. Test-time Recursive Thinking: Self-Improvement without External Feedback

**arXiv ID:** 2602.03094 | [PDF](https://arxiv.org/pdf/2602.03094v1)

**作者:** Yufan Zhuang `[一作]` (University of California San Diego), Weizhu Chen `[通讯]` (Microsoft Research)

**通讯引用:** 12911 | [OpenAlex ID](https://openalex.org/A5051745436)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种在测试时自我递归思考（TRT）框架，使大型语言模型在单个问题上通过迭代生成、选择和反思不断自我改进。

**💡 创新点**

创新点在于结合知识积累、轮次特定策略设计和无监督自检，三者协同实现了无需外部反馈的自我提升。

**🔧 技术方法**

采用并行采样、策略生成、答案评估（自检/执行测试）和知识列表聚合等技术。

**📊 数据集**

评估使用AIME（数学推理）和LiveCodeBench（代码生成）两个基准。

**📈 对比分析**

与并行思考+多数投票、RSA等基线相比，在AIME上达到100%准确率，在LiveCodeBench上相较于RSA提升10.4–14.8个百分点。

**⚠️ 局限性**

限制包括测试用例生成质量、计算成本随轮次线性增长以及仅在单一实例内学习，缺乏跨问题知识迁移。

---

## 332. Straggler-Aware Coded Polynomial Aggregation

**arXiv ID:** 2602.03074 | [PDF](https://arxiv.org/pdf/2602.03074v1)

**作者:** Xi Zhong `[一作]` (University of Florida), Mingyue Ji `[通讯]` (University of Florida)

**通讯引用:** 3225 | [OpenAlex ID](https://openalex.org/A5058487273)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在预设非故障模式下的编码多项式聚合（CPA），给出了可行性必要且充分的正交性条件，并设计了满足这些条件的评估点构造方法。

**💡 创新点**

创新点在于：①在非完整鲁棒的 straggler 环境中引入非故障集合交集阈值 I*，实现更少的工作节点即可完成精确聚合；②证明该阈值在大多数非故障集合下既是充分又是必要的；③给出显式构造算法和理论与仿真的一致性。

**🔧 技术方法**

采用代数多项式分析、正交性条件、交集分解技术以及数值最小二乘优化；还借鉴了非故障 CPA 的构造算法来实现评估点。

**📊 数据集**

仿真中使用 Chebyshev 点作为数据点，随机生成权重向量；并未使用真实工业数据集，而是通过系统参数的随机组合进行实验。

**📈 对比分析**

与传统的基于个体解码的 Lagrange 编码计算进行对比，证明在给定非故障集合下所需的工作节点数更少；仿真结果显示，当交集大小 I≥I* 时可行率达到 100%，低于阈值时几乎为 0，验证了阈值的 sharp 转折性。

**⚠️ 局限性**

局限性：①需要事先知道可接受的非故障集合，适用于非随机但已知的 straggler 模式；②在 I<I* 时可行性可能取决于具体集合的结构，且对极端或特殊结构的非故障集合缺乏理论保证；③假设数据点与评估点均为通用（generic），在特殊代数结构下结果可能不成立。

---

## 333. TMS: Trajectory-Mixed Supervision for Reward-Free, On-Policy SFT

**arXiv ID:** 2602.03073 | [PDF](https://arxiv.org/pdf/2602.03073v1)

**作者:** Rana Muhammad Shahroz Khan `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 4077 | [OpenAlex ID](https://openalex.org/A5103073431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为轨迹混合监督（TMS）的无奖励框架，旨在通过创建动态课程来解决强化学习（RL）和监督微调（SFT）之间的权衡，从而提高大型语言模型（LLM）在下游任务上的表现。

**💡 创新点**

TMS通过减少策略-标签差异（PLD）来防止标准SFT中的模式崩溃，显著提高了准确性和保留能力，接近于RL的表现，而无需奖励模型或验证器。

**🔧 技术方法**

使用了轨迹混合监督（TMS）技术，该技术通过从模型的历史检查点中采样监督信号来减少监督不匹配和模式崩溃。

**📊 数据集**

在多个推理（MATH、GSM8K）和指令跟随基准上进行了实验，使用了不同的模型和数据集进行评估。

**📈 对比分析**

与标准SFT和迭代SFT相比，TMS在准确性和保留能力上显著优于这些方法，接近于RL的表现，且在多个基准测试中表现出色。

**⚠️ 局限性**

TMS的局限性在于它依赖于模型生成的监督信号，如果基础模型表现出有害或偏见的行为，可能会传播这些模式，因此需要谨慎的数据策划和安全评估。

---

## 334. Towards Weak Stratification for Logics of Definitions

**arXiv ID:** 2602.03072 | [PDF](https://arxiv.org/pdf/2602.03072v1)

**作者:** Nathan Guermond `[一作]` `[通讯]`, Nathan Guermond

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在定义逻辑中引入弱层化（weak stratification），并证明其与 nabla 量词及一般归纳可兼容，形成新的逻辑 G^ω。

**💡 创新点**

首次允许在定义中出现负面出现的谓词，同时保持一致性，并展示弱层化在逻辑关系（logical relations）等应用中的适用性。

**🔧 技术方法**

采用固定点与子句表示、地面逻辑构造、裁剪归约与可归约性（reducibility）技术，以及可归约性引理实现一致性与裁剪消除证明。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

通过理论对比证明 G^ω 的一致性与裁剪可接受性，未给出实验性能指标。

**⚠️ 局限性**

仍无法在完整的 Abella 基逻辑中实现裁剪消除；弱层化仅适用于非归纳定义，归纳定义仍需严格层化；对 L_λ 模式的限制也是待解决的问题。

---

## 335. Finding Optimal Video Moment without Training: Gaussian Boundary Optimization for Weakly Supervised Video Grounding

**arXiv ID:** 2602.03071 | [PDF](https://arxiv.org/pdf/2602.03071v1)

**作者:** Sunoh Kim `[一作]` (Dankook University), Daeho Um `[通讯]` (University of Seoul)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的高斯边界优化（GBO）框架，通过优化高斯 proposal 的覆盖率与长度来预测视频时段边界。

**💡 创新点**

创新点在于把边界预测转化为可闭式求解的优化问题，给出不同 λ 取值下的最优解，并兼容单高斯与混合高斯 proposal。

**🔧 技术方法**

利用高斯 proposal 的中心和宽度，以及覆盖面积与长度惩罚的积分，推导出解析式并实现无训练的边界推断。

**📊 数据集**

在 ActivityNet Captions 与 Charades-STA 两大弱监督视频 grounding 数据集上进行实验。

**📈 对比分析**

与多种基线（CNM、CPL、PPS 等）以及最新方法相比，GBO 在 R@5/IoU=0.5 等指标上提升约 5‑10%，实现了 state‑of‑the‑art 性能。

**⚠️ 局限性**

局限性包括对称边界调整的假设导致在高度不对称事件时性能下降，以及 λ 的手动调参仍需经验。

---

## 336. From semantic memory to collective creativity: A generative cognitive foundation for social creativity models

**arXiv ID:** 2602.03068 | [PDF](https://arxiv.org/pdf/2602.03068v1)

**作者:** Mirza Nayeem Ahmed `[一作]` (Northeastern University), Raiyan Abdul Baten `[通讯]` (University of South Florida)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5038144794)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个多层 socio‑cognitive 代理模型，通过共享语义词汇和子结构，让代理在同一语义空间内产生想法并相互交流。

**💡 创新点**

创新点在于：① 用单一可调的语义网络重连概率 p 作为“创意开关”，从认知结构本身产生个体创造差异；② 证明了在无额外策略或奖励的前提下，代理间的想法交换自然产生认知刺激和网络层面的冗余效应。

**🔧 技术方法**

主要技术包括：Watts–Strogatz 小世界网络生成、随机游走（随机漫步）作为语义搜索过程、Jaccard 相似度衡量重叠、固定效应回归与聚类标准误分析。

**📊 数据集**

使用的是合成语义网络数据集：基底图 G₀ 由 100 节点、平均度 4 组成，随后通过不同的重连概率生成 500 个代理的语义图；所有实验均在这些人工生成的图上进行。

**📈 对比分析**

方法比较通过统计相关性、线性回归与曲线拟合评估模块化与想法宽度、重叠与刺激收益、共享源与冗余之间的关系；结果表明模块化越低，想法宽度越大；重叠越低，刺激收益越高；共享灵感源显著提高收敛度（平均提升约 0.026，Cohen d ≈ 1.4）。

**⚠️ 局限性**

局限性包括：模型仅使用固定随机游走搜索，未考虑部分吸收、选择性关注或记忆衰退；代理间缺乏策略性伙伴选择或网络演化机制；实验基于人工合成网络，缺少真实语义结构的验证。

---

## 337. Shortcut Features as Top Eigenfunctions of NTK: A Linear Neural Network Case and More

**arXiv ID:** 2602.03066 | [PDF](https://arxiv.org/pdf/2602.03066v1)

**作者:** Jinwoo Lim `[一作]` (Seoul National University), Soo-Mook Moon `[通讯]` (Seoul National University)

**通讯引用:** 1599 | [OpenAlex ID](https://openalex.org/A5028446029)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过 NTK 框架分析了在数据分布不平衡时导致的 shortcut 学习，并将理论推广到更复杂网络。

**💡 创新点**

创新点在于揭示 shortcut 特征对应大特征值并在训练后仍占主导，证明最大间距偏差非唯一原因，并提出 predictability 与 availability 两个量化指标。

**🔧 技术方法**

采用 NTK 理论、线性网络、两层 ReLU、ResNet‑18 训练与梯度流分析，并引入 SD / Marg‑Ctrl 正则化。

**📊 数据集**

实验数据集包括 Patched‑MNIST、Colored‑MNIST、Waterbirds、CelebA、Dogs & Cats。

**📈 对比分析**

与传统 CE/MSE 训练和 SD 正则化对比，发现即使抑制最大间距，shortcut 仍主导；在 ResNet‑18 上，shortcut 标签的 availability 与测试准确率均高于 ground‑truth。

**⚠️ 局限性**

局限性：理论基于线性、无限宽网络与 NTK 假设；对非线性核或实际网络的推理有限；availability 在强快捷标签下可能无法捕捉弱快捷特征。

---

## 338. JRDB-Pose3D: A Multi-person 3D Human Pose and Shape Estimation Dataset for Robotics

**arXiv ID:** 2602.03064 | [PDF](https://arxiv.org/pdf/2602.03064v1)

**作者:** Sandika Biswas `[一作]` (Monash University), Hamid Rezatofighi `[通讯]` (Monash University)

**通讯引用:** 3198 | [OpenAlex ID](https://openalex.org/A5034608678)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入 JRDB-Pose3D 数据集，提供基于 SMPL 的多人体 3D 姿态与形状标注，并实现连续跟踪；

**💡 创新点**

在真实机器人导航环境中构建大规模多人物 3D 数据集，兼具完整体形、连续 ID、全景视角及与 JRDB 原有社交、语义标注的无缝融合，填补现有单人或实验室场景的空白；

**🔧 技术方法**

利用 CameraHMR 进行姿态初始化、PnP 进行全局对齐、形状一致性筛选、SMPL 优化及人工校正，形成高质量、时空一致的 3D 注释；

**📊 数据集**

基于 JRDB 机器人视觉数据集 54 条序列，并继承其 2D 姿态、社交分组、交互、全景语义分割等多模态注释；

**📈 对比分析**

在 JRDB-Pose3D 上对多人体姿态估计、跟踪与动作预测方法进行基准评测，结果表明相比 WorldPose，数据更具遮挡、密度与角度多样性，现有方法性能显著受限；

**⚠️ 局限性**

数据集主要聚焦大学校园环境，未覆盖不同天气或极端光照；标注仍需人工干预；缺乏手部细节和极端动态场景下的深度信息。

---

## 339. Towards Considerate Embodied AI: Co-Designing Situated Multi-Site Healthcare Robots from Abstract Concepts to High-Fidelity Prototypes

**arXiv ID:** 2602.03054 | [PDF](https://arxiv.org/pdf/2602.03054v1)

**作者:** Yuanchen Bai `[一作]` (Cornell University), Angelique Taylor `[通讯]` (Cornell University)

**通讯引用:** 170 | [OpenAlex ID](https://openalex.org/A5074668213)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开展为期14周的多学科共创研讨会，聚焦三类医疗场景（急诊、康复、睡眠障碍诊所），从抽象头脑风暴到高保真原型，构建医疗服务机器人并形成八条设计准则；

**💡 创新点**

创新点在于：①跨场景持续共创；②从低保真到高保真迭代推进，揭示真实部署约束；③为非技术参与者提供教育支撑，提升其技术素养与参与度；④提炼可落地的“考虑性具身AI”八条设计准则；

**🔧 技术方法**

使用TIAGo移动底座、Jetson XAVIER GPU、OAK‑D相机等硬件；MakerLab 3D打印、激光切割、焊接等制造工具；教育模块涵盖机器人感知、规划、交互与设计原型技术；

**📊 数据集**

无公开数据集，主要通过参与者生成的原型图纸、卡板模型、全尺寸模型、访谈记录、观察笔记等原始数据收集；

**📈 对比分析**

方法以质性研究为主：原型演化分析、访谈编码、主题提炼；对比三类场景下的机器人角色、交互模式与设计细节；未涉及数值性能指标，而是从用户接受度、设计可行性、跨场景通用性等维度进行评估；

**⚠️ 局限性**

局限性包括：①仅招募到5名现场医疗工作者，样本量有限；②使用单一TIAGo平台限制了对其他具身AI形式的探索；③研究场景均为北方大城市，缺乏跨文化验证；④部分参与者自选参与，可能导致样本偏向对机器人热情者；

---

## 340. SAFE-KD: Risk-Controlled Early-Exit Distillation for Vision Backbones

**arXiv ID:** 2602.03043 | [PDF](https://arxiv.org/pdf/2602.03043v1)

**作者:** Salim Khazem `[一作]` `[通讯]` (Talan Research Center), Salim Khazem (Talan Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种通用的多退出网络包装器SAFE-KD，利用层级知识蒸馏和合规风险控制实现可控的早期退出。

**💡 创新点**

创新点在于将Decoupled Knowledge Distillation与深浅一致性相结合，并通过Conformal Risk Control为每个退出点提供有限样本风险保证。

**🔧 技术方法**

采用了轻量级退出头、EMA教师蒸馏、DKD、深浅一致性约束和CRC（合规风险控制）技术。

**📊 数据集**

在CIFAR-10/100、STL-10、Oxford-IIIT Pets、Flowers102、FGVC Aircraft等多种通用与细粒度数据集上进行实验。

**📈 对比分析**

与ERM、MultiExit、KD、DKD等基线比较，SAFE-KD在保持或提升最终准确率的同时实现更低的期望计算量，且风险控制符合设定的δ。

**⚠️ 局限性**

局限在于需要足够大小的校准集以获得紧凑阈值；在严重分布偏移下的合规假设可能不成立，导致风险保证失效。

---

## 341. DF-LoGiT: Data-Free Logic-Gated Backdoor Attacks in Vision Transformers

**arXiv ID:** 2602.03040 | [PDF](https://arxiv.org/pdf/2602.03040v1)

**作者:** Xiaozuo Shen `[一作]` (University of Arizona), Hongyi Wu `[通讯]` (University of Arizona)

**通讯引用:** 3181 | [OpenAlex ID](https://openalex.org/A5115636506)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种完全数据无关、仅通过修改权重即可在Vision Transformer（ViT）中植入后门的攻击方法，攻击者无需训练数据、微调或改动模型结构，直接在发布的权重检查点上进行一次性编辑。

**💡 创新点**

创新点在于：①利用ViT的多头注意力实现逻辑门（m‑of‑n）触发器，将触发信号写入专用残差通道并在模型深度中保持不变；②通过对Q/K缩放、V/O投影重写实现触发器的精确构造和内部状态的写入；③在无需任何额外数据的前提下实现近乎完美的攻击成功率，同时保持清洁输入的准确率与可解释性。

**🔧 技术方法**

技术手段包括：基于自注意力几何的触发器反向投影构造；对注意力头的Q/K权重放大；对V/O权重进行定位性重写，将触发证据写入CLS残差中的专用维度；在中间层使用残差“高速公路”零写回实现状态传递；在最后一层通过单神经元门控实现目标类别的条件注入；利用多头实现逻辑门。

**📊 数据集**

实验使用ImageNet‑1K验证集，评估了DeiT‑Tiny、DeiT‑Small和ViT‑B三种预训练ViT骨干。

**📈 对比分析**

与DFBA（CNN转移的基线）及多种部署时防御（Neural Cleanse、Fine‑Pruning、Patch Processing、BDVT）对比，DF‑LoGiT在1‑of‑1协议下实现近100%攻击成功率，2‑of‑3协议下平均攻击成功率超过98%，同时对清洁输入的准确率仅下降不到2%；在所有评估的防御下仍保持高攻击成功率，证明其鲁棒性。

**⚠️ 局限性**

局限性：攻击仅针对已公开的ViT检查点，需攻击者拥有白盒权重访问；目前仅在ViT架构上验证，对其他Transformer或CNN模型的适用性尚未证明；如果目标模型经过强度高的微调或结构重构，可能影响后门的有效性。

---

## 342. HP-GAN: Harnessing pretrained networks for GAN improvement with FakeTwins and discriminator consistency

**arXiv ID:** 2602.03039 | [PDF](https://arxiv.org/pdf/2602.03039v1)

**作者:** Geonhui Son `[一作]` (Yonsei University), Dosik Hwang `[通讯]` (Korea Institute of Science and Technology)

**通讯引用:** 3033 | [OpenAlex ID](https://openalex.org/A5085519704)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为HP-GAN的新型GAN框架，利用预训练网络作为自监督学习编码器（FakeTwins）和判别器一致性正则化，显著提升生成图像的多样性与质量。

**💡 创新点**

创新点包括：① FakeTwins方案，将Bardow Twins自监督学习直接嵌入生成器端，避免传统对判别器的间接约束；② 判别器一致性损失，强制CNN和ViT两种特征网络的判别器输出保持一致，从而提升训练稳定性和多样性。

**🔧 技术方法**

使用技术：预训练的EfficientNet-lite0（CNN）与DeiT-B（ViT）特征网络；Bardow Twins自监督学习；判别器一致性正则化；Hinge loss、EMA、DiffAugment、低维潜向量（64维）等GAN训练技巧；多尺度判别器与特征混合层。

**📊 数据集**

数据集涵盖：大型数据集（FFHQ、LSUN-Bedroom、LSUN-Church、CLEVR、Cityscapes、AFHQ）；小型数据集（WikiArt、Oxford Flowers、Pokemon等）；少量样本（Obama、Grumpy Cat、Panda、AnimalFace Cat/Dog 等）；医疗数据集BraTS2021 T1 用于迁移学习验证。

**📈 对比分析**

与多种GAN（StyleGAN2/3、Projected GAN、FastGAN 等）和扩散模型（ADM、LDM、Diffusion GAN 等）进行对比，使用FID、KID、Precision/Recall 等指标评估。HP-GAN 在所有基准数据集上均取得最优或接近最优结果，例如 FFHQ FID 1.69、AFHQ Cat 1.81、Dog 3.63、Wild 1.18；在少量样本设置下也保持竞争力。

**⚠️ 局限性**

局限性：对预训练网络的选择和质量较为敏感；在极少样本（约100张）时仍略逊于部分方法；训练仍需要较大计算资源；目前仅验证在单帧图像上，对高分辨率、多模态或视频等更复杂场景的适用性尚未充分探讨。

---

## 343. MUSE: A Multi-agent Framework for Unconstrained Story Envisioning via Closed-Loop Cognitive Orchestration

**arXiv ID:** 2602.03028 | [PDF](https://arxiv.org/pdf/2602.03028v1)

**作者:** Wenzhang Sun `[一作]` (Li Auto), Wei Chen `[通讯]` (Li Auto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MUSE框架，用多代理闭环约束执行机制从简短提示生成长篇音视频故事。

**💡 创新点**

将长篇故事生成视为闭环约束执行问题，将叙事意图显式转化为可执行的身份、空间、时间控制，并通过计划‑执行‑验证‑修订循环实现多模态一致性。

**🔧 技术方法**

利用多代理体系结构、可执行控制束、结构化多模态验证、视觉生成模型Flux/ Wan2.2、音频生成模块VTS、LMM评估、滑动窗口上下文与动态/防御相机路由等技术。

**📊 数据集**

在ViStoryBench进行视觉一致性评估，并构造自研MUSEBench（30个多样化提示、5个体裁）作为无参考的全流程评测；使用公开语音与视觉模型训练集。

**📈 对比分析**

与Vlogger、AnimDirector、MMStoryAgent、V‑GOT、MovieAgent等基线在ViStoryBench与MUSEBench上对比，MUSE在身份保持、叙事连贯性、跨模态一致性和电影质感上均显著优于基线。

**⚠️ 局限性**

在拥挤或遮挡场景下身份保持仍受限，复杂多角色互动与细粒度情感表达控制不足，且音频评估指标对人类主观性相关性不高。

---

## 344. Generalizable and Interpretable RF Fingerprinting with Shapelet-Enhanced Large Language Models

**arXiv ID:** 2602.03035 | [PDF](https://arxiv.org/pdf/2602.03035v1)

**作者:** Tianya Zhao `[一作]` (Florida International University), Xuyu Wang `[通讯]` (Florida International University)

**通讯引用:** 5478 | [OpenAlex ID](https://openalex.org/A5043788836)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

结合预训练LLM与可学习二维形状片段实现RF指纹识别，提升跨域泛化与可解释性。

**💡 创新点**

创新地将LLM仅微调位置嵌入/层归一化，同时引入可变长度2D形状片段并采用稀疏与多样性正则化，提供内在可解释性并提升性能。

**🔧 技术方法**

使用Transformer基础LLM（如GPT‑2）、CNN输入嵌入、形状片段网络、稀疏与多样性正则以及原型学习进行少样本推理。

**📊 数据集**

评估六个数据集：ORACLE、CORES、WiSig、NetSTAR、LoRa、BLE，涵盖802.11、LoRa、BLE协议。

**📈 对比分析**

与ResNet‑18、RadioNet、LIMU‑BERT、SimCLR、RF‑PTN、PatchLLM等基线对比，标准与1/5‑shot设置下目标域准确率常居前列，平均提升约5–10%。

**⚠️ 局限性**

主要限制为模型体量大、推理延迟与内存占用高，适合服务器端部署；缺乏针对边缘设备的压缩方案和更深入的物理层可解释性。

---

## 345. BinaryDemoire: Moiré-Aware Binarization for Image Demoiréing

**arXiv ID:** 2602.03176 | [PDF](https://arxiv.org/pdf/2602.03176v1)

**作者:** Zheng Chen `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22315 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种极度压缩的图像去摩尔纹模型 BinaryDemoire，利用 1-bit 量化实现极高压缩率的去摩尔纹。

**💡 创新点**

创新点：1）Moiré-Aware Binary Gate (MABG)，通过单层小波分解得到频率描述子与激活统计量，动态调节通道门控以减少 1-bit 量化误差；2）Shuffle‑Grouped Residual Adapter (SGRA)，利用分组稀疏投影与交叉混合实现轻量化残差对齐，解决多尺度变通道尺寸的残差连接问题。

**🔧 技术方法**

技术：1-bit 权重与激活量化、直通估计 (STE)、小波分解 (DWT)、频率与统计特征融合、共享全连接门控、分组稀疏投影与通道交叉混合。

**📊 数据集**

数据集：UHDM、FHDMi、LCDMoire、TIP2018 四个公开去摩尔纹基准。

**📈 对比分析**

对比方法：与五种 1-bit 去摩尔纹方法（ReActNet、BBCU、BiSRNet、Biper、BiMaCoSR）以及四种全精度方法（MopNet、MDDM、FHDe²Net、ESDNet）。BinaryDemoire 在四个基准上均比同等参数量的 1-bit 方法优异，且在大多数指标上逼近甚至超过部分全精度模型，参数量仅 5.2%（减少 94.8%），计算量仅 4.4%（减少 95.6%）。

**⚠️ 局限性**

局限性：1) 仅针对 1-bit 量化，性能仍受限于极低位宽；2) 对于极其细腻的纹理或极端摩尔纹，仍可能出现轻微细节损失；3) 目前仅在图像去摩尔纹任务验证，尚未证明对其他低级视觉任务的通用性。

---

## 346. Probe-then-Commit Multi-Objective Bandits: Theoretical Benefits of Limited Multi-Arm Feedback

**arXiv ID:** 2602.03175 | [PDF](https://arxiv.org/pdf/2602.03175v1)

**作者:** Ming Shi `[一作]` (University at Buffalo), Ming Shi `[通讯]` (University at Buffalo)

**通讯引用:** 3963 | [OpenAlex ID](https://openalex.org/A5002972091)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出受限多臂反馈下的 Probe‑then‑Commit 多目标多臂赌博机模型，设计 PtC‑P‑UCB 算法，并给出前沿覆盖和偏好化后退的理论保证，同时扩展到多模态感知。

**💡 创新点**

① 在中间反馈范式下实现前沿覆盖的主动探测与决策；② 通过前沿覆盖潜能与边际超体积获得 1/√q 的学习加速；③ 引入多模态融合实现方差自适应提升；④ 在理论与实验中首次将超体积覆盖和偏好化后退统一到同一框架。

**🔧 技术方法**

UCB 型置信区间与前沿覆盖潜能设计；超体积（HV）与凸包理论；子高斯集中与自适应样本预算；逆方差加权多模态融合；贪心子模/多目标 UCB 算法。

**📊 数据集**

采用基于无线/边缘系统的合成数据集（24 把臂、4 维目标），通过聚类混合生成前沿臂和劣化臂；多模态实验中使用 3 种噪声尺度 (0.08, 0.12, 0.20)。

**📈 对比分析**

与传统单臂反馈 (q=1) 及全信息专家 (q=K) 对比；在前沿超体积误差和标量化后退两指标上均实现 1/√q 的加速；多模态融合进一步提前达到相同误差水平，实验结果与理论保持一致。

**⚠️ 局限性**

仅考虑无上下文/线性结构、无延迟、无非平稳；假设子高斯噪声且模态权重已知；实验仅限合成场景，缺乏真实网络部署验证；在臂数大或目标维度高时，计算复杂度可能显著上升。

---

## 347. VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models

**arXiv ID:** 2602.03160 | [PDF](https://arxiv.org/pdf/2602.03160v1)

**作者:** Woojin Kim `[一作]` (AIDAS Laboratory), Jaeyoung Do `[通讯]` (AIDAS Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一框架，完成了大语言模型的价值取向提取、评估与可调强度控制。

**💡 创新点**

创新点包括构建层次化价值嵌入空间、创建大规模价值强度数据库以及引入基于排名的强度评估方法。

**🔧 技术方法**

使用了层次对比学习、交叉理论锚点对齐、Plackett–Luce 排名模型以及锚点评估器等技术。

**📊 数据集**

主要使用了 SVT、MFT、权利与义务理论等价值体系的语料库，如 Denevil、Social Chemistry、MFRC、ValueNet、ValueEval、ValuePrism 等。

**📈 对比分析**

在层次一致性、相似度相关性和锚点评估方面与基准方法相比提升 20%+，评分评估的方差、最大范围和签名翻转率均低于传统评级基线，排名一致性达到 84%。

**⚠️ 局限性**

限制包括对负向强度控制效果弱、不同模型的可调性差异显著、跨语言及更丰富价值体系的推广仍待验证。

---

## 348. PAMAS: Self-Adaptive Multi-Agent System with Perspective Aggregation for Misinformation Detection

**arXiv ID:** 2602.03158 | [PDF](https://arxiv.org/pdf/2602.03158v1)

**作者:** Zongwei Wang `[一作]` (Chongqing University), Chenghua Lin `[通讯]` (University of Manchester)

**通讯引用:** 3352 | [OpenAlex ID](https://openalex.org/A5024599321)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于大型语言模型的多智能体系统 PAMAS，用于在社交媒体上检测误信息。该系统通过层级角色（审计者、协调者、决策者）以及视角感知聚合，提升了对稀疏欺骗信号的捕获与融合。

**💡 创新点**

创新点包括：①视角感知的分层聚合机制，显著放大异常线索并避免信息淹没；②自适应结构优化（拓扑适配、目标校正）和置信度引导的路由推理，实现高效、可扩展的决策流程；③在多智能体协作中首次结合 LLM 的记忆机制与多层聚合，兼顾鲁棒性与可解释性。

**🔧 技术方法**

核心技术包括：大型语言模型（LLM）作为智能体推理核心；分层聚合（Auditor→Coordinator→Decision‑Maker）与加权投票；自适应拓扑（基于准确率与冗余度的剪枝/扩充）；针对误判的目标校正（仅更新错误智能体的记忆）；置信度引导的路由推理；记忆模块（经验、动作、置信度）。

**📊 数据集**

实验使用三大公开数据集：Amazon 评价（含恶意评论）、DeRev2018（误信息评测）以及 PolitiFact（伪新闻）。

**📈 对比分析**

与传统深度学习模型（MLP、NFGCN、SIPUL、BREAK）以及多种多智能体拓扑（Chain、Star、Tree、Graph、Layer、DyLAN、Vanilla‑LLM/Agent）进行对比。PAMAS 在所有数据集上均实现最高的准确率、F1 以及 AUC，并在 token 使用上表现出最优的效率，位于性能‑效率曲线的前沿。

**⚠️ 局限性**

局限性：目前仅针对文本误信息，未扩展到多模态；缺乏实时人机协同机制；对大型 LLM 的算力与成本依赖较高；在极大规模流式内容下的可扩展性与实时性仍待进一步验证。

---

## 349. Fully Kolmogorov-Arnold Deep Model in Medical Image Segmentation

**arXiv ID:** 2602.03156 | [PDF](https://arxiv.org/pdf/2602.03156v1)

**作者:** Xingyu Qiu `[一作]` (Harbin Institute of Technology), Shuo Li `[通讯]` (Case Western Reserve University)

**通讯引用:** 49731 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了首个全KA（Kolmogorov–Arnold）深度模型 ALL U-KAN，完成了完全替代传统 FC 与 Conv 层的网络。

**💡 创新点**

创新点：①SaKAN 共享激活函数，显著减少参数并提高训练样本量；②Grad‑Free Spline 将 B‑spline 梯度分离，内存占用降 20× 以上；③将上述两项集成的 KA 与 KAonv 层构成全 KA 网络，实现了可深度堆叠的 KAN。

**🔧 技术方法**

使用的技术包括 Sprecher 变体 KA 理论、B‑spline 激活函数、共享激活设计、梯度分离与 chunk‑wise 计算、KA 与 KAonv 层的实现。

**📊 数据集**

实验数据集：BUSI（乳腺超声）、GlaS（脑动脉瘤 CT）、CVC‑ClinicDB（结肠息肉视频）。

**📈 对比分析**

与 U‑Net、U‑Net++、Att‑UNet、U‑Mamba、U‑NeXt、Rolling‑UNet 以及 U‑KAN 等 12 种基准模型比较，ALL U‑KAN 在三数据集的 IoU 与 F1 分别提升约 2%–4%，参数量和显存仅略高于传统模型，保持竞争力。

**⚠️ 局限性**

局限性：训练速度仍慢于传统 CNN/MLP，因为需要计算 B‑spline；对极高分辨率图像仍存在显存/时间瓶颈，未来需进一步加速 spline 计算。

---

## 350. Is It Possible to Make Chatbots Virtuous? Investigating a Virtue-Based Design Methodology Applied to LLMs

**arXiv ID:** 2602.03155 | [PDF](https://arxiv.org/pdf/2602.03155v1)

**作者:** Matthew P. Lad `[一作]` (University of Notre Dame), Megan Levis Scheirer `[通讯]` (University of Notre Dame)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对13名计算机科学专业人士进行半结构化访谈，探讨将美德伦理框架应用于大型语言模型（LLM）的设计模式，评估其对模型道德行为的潜在影响。

**💡 创新点**

创新点在于首次系统性地将美德伦理原则与软件设计模式结合，为LLM的“德性化”提供可操作的设计方法，并通过实证访谈收集设计模式有效性反馈。

**🔧 技术方法**

主要采用访谈方法，向受访者展示一系列基于美德的LLM设计模式，收集其理解、改进建议及潜在缺陷，并进行定性内容分析。

**📊 数据集**

数据来源为受访者自我报告的个人信息和访谈内容，没有使用公开数据集；受访者均为CS背景，覆盖多性别、多族裔与宗教身份。

**📈 对比分析**

由于研究为定性探索，未采用对照实验或性能指标；通过访谈问卷评估设计模式的净正面、净负面或模棱两可影响，但缺乏量化对比。

**⚠️ 局限性**

局限性包括样本量小（仅13人）、受访者均为计算机专业人士，缺乏跨学科视角；访谈结果主观性强，未能对LLM实际表现进行实验验证。

---

## 351. FinMTM: A Multi-Turn Multimodal Benchmark for Financial Reasoning and Agent Evaluation

**arXiv ID:** 2602.03130 | [PDF](https://arxiv.org/pdf/2602.03130v1)

**作者:** Chenxi Zhang `[一作]` (Wuhan University), Rongjunchen Zhang `[通讯]` (HiThink Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了FinMTM多轮多模态金融基准，涵盖单/多选、长对话和金融代理任务，并对22个视觉‑语言模型（VLM）在此基准上进行系统评测。

**💡 创新点**

创新点包括：① 设计了多轮交互与金融代理任务的评估框架；② 在单/多选任务中引入严格的set‑overlap评分；③ 在多轮对话中采用双重评估（回合级与会话级）并给出加权综合分；④ 对代理任务使用基于工具调用的两阶段轨迹评估；⑤ 通过中文、英文双语数据大幅提升任务多样性与实用性。

**🔧 技术方法**

主要技术包括：LLM‑as‑judge评估策略、set‑overlap 与加权综合评分、Fβ（β>1）工具召回评估、两阶段代理评估（规划+总结）以及多模态预训练模型的推理与微调。

**📊 数据集**

使用FinMTM数据集：共11,133条金融QA（3,964条单/多选，6,169条多轮对话，1,000条代理任务），涵盖3,600张金融图像和400份PDF，涉及美股与A股市场，支持中英双语。

**📈 对比分析**

对22个VLM（包括ChatGPT‑4o、Gemini‑3 Pro、InternVL、Qwen系列等）进行评测，结果显示：专有模型整体优于开源模型；Gemini‑3 Pro在多轮对话与代理任务上领先；ChatGPT‑5在多选任务表现最佳；Gemini‑3 Flash在代理任务（含模糊输入）中取得最高分；开源模型在长文记忆和自我修正等高难度子任务中表现明显不足。

**⚠️ 局限性**

限制：① 评估部分基于LLM‑as‑judge，存在主观性与提示依赖；② 代理任务工具集固定，未覆盖实际工具生态的动态性；③ 长文引用要求严格，可能低估模型在内容正确但格式不符的能力；④ 基准主要集中在美股与A股，其他市场表现未知。

---

## 352. Beyond Cropping and Rotation: Automated Evolution of Powerful Task-Specific Augmentations with Generative Models

**arXiv ID:** 2602.03123 | [PDF](https://arxiv.org/pdf/2602.03123v1)

**作者:** Judah Goldfeder `[一作]` (Columbia University), Hod Lipson `[通讯]` (Columbia University)

**通讯引用:** 31615 | [OpenAlex ID](https://openalex.org/A5025894735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于进化搜索的自动化数据增强框架 EvoAug，结合控制扩散与 NeRF 等生成模型与经典增强手段，在低样本细粒度分类任务中学习任务专属的增强策略。

**💡 创新点**

创新点在于将生成模型纳入增强算子并构建可学习的二叉增强树，同时提出无监督聚类和训练损失等评估指标，用于极低样本情形下的自动化增强策略搜索。

**🔧 技术方法**

使用了控制扩散（ControlNet）、Zero123 NeRF、进化算法、K‑fold、聚类与 Silhouette 等技术。

**📊 数据集**

在 Caltech256、Flowers102、Stanford Dogs、Stanford Cars、Oxford‑IIIT Pets、Food101 等六个细粒度数据集上进行实验。

**📈 对比分析**

与 Naïve、RandAugment、AutoAugment 等基线对比，EvoAug 在大多数任务上提升 1–3% 甚至更大，尤其在 1‑shot 与 5‑shot 情况下表现突出。

**⚠️ 局限性**

主要局限在于评估成本高、难以扩展到完整数据集，以及对生成模型质量与域适应性的依赖。

---

## 353. AgentDyn: A Dynamic Open-Ended Benchmark for Evaluating Prompt Injection Attacks of Real-World Agent Security System

**arXiv ID:** 2602.03117 | [PDF](https://arxiv.org/pdf/2602.03117v1)

**作者:** Hao Li `[一作]` (Washington University in St. Louis), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 AgentDyn 基准，用于在动态、开放式任务环境下评估大型语言模型代理的注入攻击防御能力。

**💡 创新点**

创新点在于：①提出三大改进维度（动态任务、包含有用指令、任务复杂度提升）；②手工构造 60 个用户任务与 560 个注入测试案例，覆盖购物、GitHub 与日常生活三大场景；③展示现有防御在此更严格基准上的严重不足，揭示真实部署的挑战。

**🔧 技术方法**

技术方法包括：基于 AgentDojo 框架搭建沙盒；手工设计动态规划任务与注入向量；实现并评测 Prompt Sandwiching、Spotlighting、Tool Filter、ProtectAI、PromptGuard2、PIGuard、Meta SecAlign、CaMeL、Progent 与 DRIFT 等十种主流防御；使用 GPT‑4o、Gemini‑2.5 Pro、Qwen3‑235B、Llama‑3.3‑70B 等八种大模型进行实验。

**📊 数据集**

数据集为自构造的 AgentDyn，包含 3 个套件（Shopping、GitHub、Daily Life）、60 个用户任务、28 个注入任务，交叉产生 560 个测试案例；此外参照 AgentDojo 等现有基准做对比。

**📈 对比分析**

实验通过评估“原始效用”“攻击下效用”和“攻击成功率”三项指标，对不同模型和防御进行系统比较。结果表明：大多数防御在 AgentDyn 上出现明显的功能损失或无法降低攻击成功率；仅 Meta SecAlign 在保持较高效用的同时略微降低攻击成功率；整体表明现有防御在更真实的动态环境中表现不佳。

**⚠️ 局限性**

局限性：①基准规模仅涵盖三类场景，未能覆盖所有真实业务；②任务与注入均为人工设计，可能缺乏真正的多样性；③实验未探讨更高级的对抗样本或零样本攻击；④系统级防御在开放式任务上仍失效，进一步研究需提升其适应性。

---

## 354. The Mask of Civility: Benchmarking Chinese Mock Politeness Comprehension in Large Language Models

**arXiv ID:** 2602.03107 | [PDF](https://arxiv.org/pdf/2602.03107v1)

**作者:** Yitong Zhang `[一作]` (Tsinghua University), Mingxuan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 2228 | [OpenAlex ID](https://openalex.org/A5016933525)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估多种大型语言模型在识别中文礼貌、失礼与假礼貌表达的能力。

**💡 创新点**

将礼貌理论与假礼貌模型融合构建三分类数据集，并系统比较不同提示策略对模型性能的影响。

**🔧 技术方法**

利用零样本、少样本、知识增强（定义导入）与混合提示策略对LLMs进行推理。

**📊 数据集**

包含100条中文对话样本的三分类数据集，真实与模拟比例为3:2。

**📈 对比分析**

通过准确率比较发现知识增强和混合策略下模型表现最佳，中文本土模型在最高策略下可达91%准确率。

**⚠️ 局限性**

数据集规模有限，模拟假礼貌样本可能不足以覆盖真实语境，且评价仅依赖准确率，未能充分衡量模型的深层推理能力。

---

## 355. Multi-function Robotized Surgical Dissector for Endoscopic Pulmonary Thromboendarterectomy: Preclinical Study and Evaluation

**arXiv ID:** 2602.03147 | [PDF](https://arxiv.org/pdf/2602.03147v1)

**作者:** Runfeng Zhu `[一作]` (West China Hospital of Medicine Sichuan University), Kang Li `[通讯]` (West China Hospital of Medicine Sichuan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种用于肺血栓切除术（PTE）的机器人化多功能解剖器，集成了内窥镜、灌洗、吸血通道和可弯曲手柄；

**💡 创新点**

创新点在于采用同轴推拉式（CPPR）机器人结构并设计十字槽增强刚性，实现双段可弯曲、直径3.5 mm、可视化的手术器械；

**🔧 技术方法**

技术手段包括同轴推拉机器人、十字槽槽形设计、基于优化的逆运动学模型、无线遥控手柄、3D打印刀盘、传感器和高精度编码器；

**📊 数据集**

使用的实验数据集包括3D打印肺动脉模型、猪肺组织样本以及红水仿真血液；

**📈 对比分析**

与传统刚性解剖器比较，定位误差≤2 mm，承载300 g载荷，臂力≥1.5 N，手术时间显著缩短（约5 min对比传统20 min）；

**⚠️ 局限性**

局限包括摄像头视野/分辨率有限、缺乏触觉反馈、离体实验尚未验证人体复杂度，且学习曲线相对较长。

---

## 356. Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization

**arXiv ID:** 2602.03141 | [PDF](https://arxiv.org/pdf/2602.03141v1)

**作者:** Runquan Gui `[一作]` (University of Science and Technology of China), Feng Wu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 47307 | [OpenAlex ID](https://openalex.org/A5100694761)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 CoSMo 框架，利用动态的拆分-合并算法和结构对齐的强化学习，提升大型推理模型在多跳问答中的推理效率与准确性。

**💡 创新点**

创新点在于将结构冗余与内部推理分离，通过段级预算和一致性引导的拆合操作，精准对齐推理层级，避免了传统长度惩罚导致的深度受限。

**🔧 技术方法**

采用的核心技术包括基于 LLM 的一致性评判与语义生成、拆合迭代优化、Group Relative Policy Optimization（GRPO）以及段级奖励设计。

**📊 数据集**

实验使用了 HotpotQA、HaluEval、NQ、CRAG 四个数据集，其中 HotpotQA 和 HaluEval 为 In-Distribution，NQ 与 CRAG 为 Out-Distribution。

**📈 对比分析**

与加法/减法提示、剪枝 SFT 以及长度对齐 RL 等方法相比，CoSMo 在所有基准上平均提升 3.3 分准确率，段数减少 28.7%，并在 OOD 数据上保持强健的性能。

**⚠️ 局限性**

局限性包括对 LLM 评判器的依赖、拆合操作可能导致逻辑细节丢失，以及在极端复杂推理任务中拆合策略的可扩展性尚待进一步验证。

---

## 357. SwiftVLM: Efficient Vision-Language Model Inference via Cross-Layer Token Bypass

**arXiv ID:** 2602.03134 | [PDF](https://arxiv.org/pdf/2602.03134v1)

**作者:** Chen Qian `[一作]` (Tsinghua University), Xin Miao `[通讯]` (Tsinghua University)

**通讯引用:** 6579 | [OpenAlex ID](https://openalex.org/A5102027540)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SwiftVLM，一种训练‑free 的视觉‑语言模型（VLM）视觉令牌剪枝方法，利用bypass机制保留未被剪枝的视觉令牌并在后续层重新评估，结合层级选择实现高效推理。

**💡 创新点**

创新点包括：①发现视觉令牌重要性在层间呈非单调分布；②提出bypass策略，允许未被剪枝的令牌在后续层重新评估；③使用动态规划挑选具有更高辨别能力的剪枝层；④整个过程无须额外训练，保持高效与可解释性。

**🔧 技术方法**

使用了：T‑V注意力进行令牌重要性评估、视觉令牌聚合与平均合并、表示对齐（基于聚合令牌的偏移）、动态规划进行层级选择、FlashAttention实现高效推理以及对齐与合并时的额外计算开销。

**📊 数据集**

在九个常用VLM基准上进行实验，包括RefCOCO、RefCOCO+、RefCOCOg、TextVQA、GQA、V2‑VQA、SQA、MME、MMB、POPE；并在LLaVA‑NeXT‑7B上对比不同token保留比例的性能。

**📈 对比分析**

与FastV、PDrop、SparseVLM、VisionZip、FEATHER等现有剪枝方法进行对比。SwiftVLM在非定位任务保持与基线相近的准确率，同时在token预算192和128时提供约1.5–2.0×的速度提升；在定位任务（如RefCOCO）表现显著优于其他方法，特别是在低token预算下的稳健性更好。

**⚠️ 局限性**

局限性包括：对T‑V注意力的质量敏感，bypass机制在极低token预算时仍会导致性能下降；额外的聚合与对齐计算带来一定开销；未在更大模型或更广泛的多模态任务上进行验证，且在极细粒度定位场景下可能仍存在信息丢失。

---

## 358. Flexible Geometric Guidance for Probabilistic Human Pose Estimation with Diffusion Models

**arXiv ID:** 2602.03126 | [PDF](https://arxiv.org/pdf/2602.03126v1)

**作者:** Francis Snelgar `[一作]` (Australian National University), Akshay Asthana `[通讯]` (Seeing Machines)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用扩散模型和几何引导，实现从单张 2D 图像到 3D 人体姿态的概率估计，并通过条件采样得到多重可行姿态。

**💡 创新点**

1) 将 3D 姿态生成和 2D 检测完全解耦，只需 3D 数据训练无条件扩散模型； 2) 采用基于 2D 热图的高斯似然进行引导，可在不重新训练模型的情况下适配不同检测器； 3) 通过可控协方差调节姿态多样性，实现可定制的多模态输出。

**🔧 技术方法**

扩散概率模型（DDPM）、分类器引导（Guided Diffusion）、热图高斯参数化、RootNet 估计根部深度、基于多步逆过程的梯度更新。

**📊 数据集**

Human 3.6M（训练+评估）、MPI-INF‑3DHP、3DPW（跨域测试），同时使用 Stacked Hourglass、HRNet 等 2D 检测器。

**📈 对比分析**

在 Human 3.6M 上与条件生成模型（需要 2D‑3D 训练集）相比，基于 3D‑only 的方法在多假设评估（best‑of‑M）下达到或超过现有最佳对应‑free 方法（如 Jiang 等），在 MPI‑INF‑3DHP 和 3DPW 上也保持竞争力；此外在姿态补全、无条件生成等任务中表现出良好的通用性。

**⚠️ 局限性**

扩散采样耗时长，难以满足实时需求；使用简单高斯似然忽略了更丰富的观测信息（如时间一致性、语义约束）；跨域泛化虽有进展但仍受限于训练数据分布；当 2D 检测误差较大时，引导信号会导致不稳定采样。

---

## 359. "I'm happy even though it's not real": GenAI Photo Editing as a Remembering Experience

**arXiv ID:** 2602.03104 | [PDF](https://arxiv.org/pdf/2602.03104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 360. Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost

**arXiv ID:** 2602.03120 | [PDF](https://arxiv.org/pdf/2602.03120v1)

**作者:** Yinggan Xu `[一作]` (University of California, Los Angeles), Xin Qiu `[通讯]` (Cognizant)

**通讯引用:** 4309 | [OpenAlex ID](https://openalex.org/A5102019018)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于进化策略的无梯度量化LLM微调框架（Quantized Evolution Strategies，QES），能够在量化后模型上直接进行全参数微调，同时保持低显存占用。

**💡 创新点**

创新点主要有两点：①利用累积误差反馈（Delta‑Sigma调制思想）在低精度离散空间中逼近高精度梯度，从而克服梯度消失和离散误差；②通过无状态种子重放（Stateless Seed Replay）消除高精度残差向量的显存需求，只需维护极小的历史窗口即可重构残差。

**🔧 技术方法**

使用技术包括：无梯度进化策略（ES）、离散噪声投影、错误反馈累积、种子重放、量化网络（INT4/INT8/W8A8）以及GPTQ、LLM‑Compressor等量化工具。

**📊 数据集**

实验数据集：Countdown算术推理任务（给定数字集生成符合目标值的算术表达式），同时在Qwen2.5-1.5B和3B模型上进行评测。

**📈 对比分析**

与基准量化模型和现有无梯度微调方法QuZO比较。QES在INT4 Qwen2.5-1.5B上将正确率从约5.3%提升至18.0%，在3B模型上从14.25%提升至31.85%，显著优于QuZO并几乎与保存全精度残差的全精度Oracle相当。

**⚠️ 局限性**

局限性包括：①仅在整数量化（INT4/INT8/W8A8）上验证，尚未证明能推广到更激进或非均匀量化；②需要手动设置窗口大小K和衰减因子γ，影响收敛稳定性；③在极小模型或高维搜索空间中仍可能出现梯度稀疏、收敛慢的问题；④虽然显存低于传统微调，但相较于纯推理仍略高，且需要大量评估循环，训练效率受限。

---

## 361. Function-Space Empirical Bayes Regularisation with Large Vision-Language Model Priors

**arXiv ID:** 2602.03119 | [PDF](https://arxiv.org/pdf/2602.03119v1)

**作者:** Pengcheng Hao `[一作]` (Institute of Data and Information), Wenbo Ding `[通讯]` (Institute of Data and Information)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于大规模视觉-语言模型（VLM）的函数空间经验贝叶斯正则化框架 VLM‑FS‑EB，用来生成语义丰富的上下文点并构造表达性函数先验，从而实现可靠的不确定性量化与 OOD 检测。

**💡 创新点**

创新点包括：
1) 利用 VLM 的生成能力在无外部数据情况下自适应地产生多样化的语义上下文点；
2) 将冻结的 VLM 嵌入空间直接作为函数先验的特征映射，绕过传统的任务特定特征预训练；
3) 在函数空间中采用经验贝叶斯正则化而非线性化 GP，避免了近似误差和高昂的计算成本。

**🔧 技术方法**

核心技术：函数空间经验贝叶斯正则化（FS‑EB）、VLM 语义生成与对齐、对比学习得到的 VLM 嵌入、Monte‑Carlo Dropout 近似后验、以及对比实验中的 ELBO 与 KL 下降分析。

**📊 数据集**

实验数据集：MNIST、Fashion‑MNIST、CIFAR‑10、PathMNIST（医学图像），以及对应的 OOD 测试集（NotMNIST、SVHN、MNIST‑C、CIFAR‑10C0/2/4）。

**📈 对比分析**

与 FS‑EB、GFSVI（函数空间）、Dropout、MFVI、MAP 等基线对比。VLM‑FS‑EB 在全量数据下的 ACC 与 NLL 与最强基线相当或略优，ECP 仍略高；在 OOD 检测上表现最优，尤其在数据稀缺场景（25%/15% 数据）中大幅优于其他方法，取得近乎完美的 AUROC。

**⚠️ 局限性**

局限性：
1) 对 VLM 的依赖导致模型尺寸大、计算成本高；
2) 在极端少量数据时（<10%）仍出现轻微的预测准确率下降；
3) 对生成上下文点质量敏感，若 VLM 生成的样本与任务分布偏离，可能导致先验不佳；
4) 对校准（ECE）方面仍不如某些 GP‑基线，需进一步改进。

---

## 362. A Unified Candidate Set with Scene-Adaptive Refinement via Diffusion for End-to-End Autonomous Driving

**arXiv ID:** 2602.03112 | [PDF](https://arxiv.org/pdf/2602.03112v1)

**作者:** Zhengfei Wu `[一作]` (Automotive Studies), Yanjun Huang `[通讯]` (Automotive Studies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了 CdDrive，一种将固定轨迹词典与基于扩散的场景自适应细化候选融合的统一候选集规划框架，并通过共享决策模块实现最终轨迹选择。

**💡 创新点**

创新点包括：① 将词典候选与扩散细化候选统一入同一候选池，解决词典覆盖不足与细化过度修正的矛盾；② 设计 Horizon‑Aware Trajectory Noise Adapter (HATNA) 通过时间平滑和视角自适应噪声调制，提升扩散候选的几何连续性与平滑度。

**🔧 技术方法**

主要技术包括：基于条件扩散模型的词典锚点细化、DDIM 逆向采样、HATNA 噪声适配、共享的轨迹决策评分器以及隐式世界模型预测；整体训练采用 WTA、仿真与仿真评分损失的联合优化。

**📊 数据集**

使用了 NAVSIM v1 与 NAVSIM v2 两个基于 nuPlan 数据的仿真驾驶基准，包含多场景、动态交互和复杂交通规则。

**📈 对比分析**

与现有词典规划、回归细化和扩散规划方法对比，CdDrive 在 NAVSIM v1 的 PDMS 及 NAVSIM v2 的 EPDMS 上均取得最高分，且在碰撞、通行、时间-碰撞、车道保持等子指标均表现优异，证明了统一候选与 HATNA 的有效性。

**⚠️ 局限性**

局限性包括：① 需要预先构建和维护词典，对极端罕见场景可能仍缺失合适候选；② 扩散细化和 HATNA 的超参数需手工调优，模型推理时仍需多步扩散，计算成本略高；③ 对极短时动态或极端天气下的感知误差尚未充分验证。

---

## 363. Task--Specificity Score: Measuring How Much Instructions Really Matter for Supervision

**arXiv ID:** 2602.03103 | [PDF](https://arxiv.org/pdf/2602.03103v1)

**作者:** Pritam Kadasi `[一作]` (Indian Institute of Technology Gandhinagar), Mayank Singh `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 1419 | [OpenAlex ID](https://openalex.org/A5100746903)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估任务特异性得分（TSS）及其改进版TSS++用于过滤/加权指令式训练数据。

**💡 创新点**

从信息论视角量化指令对输出的独特约束，并通过硬负样本对比与质量项提升特异性评估。

**🔧 技术方法**

基于冻结语言模型的对数似然对比、InfoNCE对比度、输出质量的Perplexity/IFD指标，结合硬负采样和指数加权。

**📊 数据集**

Alpaca、Dolly‑15k、NI‑20三大指令数据集。

**📈 对比分析**

在Gemma、LLaMA、Qwen三大1B模型上进行预算控制的指令调优，TSS/TSS++在低预算下平均提升≈1.3–1.6分（SUM），在多种基准上均有提升，且常优于随机或质量筛选。

**⚠️ 局限性**

依赖评分模型的质量、替代指令生成的准确性、计算成本高、仅验证小规模模型与数据，未直接解决事实性或偏见问题。

---

## 364. Estimation of Ground Reaction Forces from Kinematic Data during Locomotion

**arXiv ID:** 2602.03177 | [PDF](https://arxiv.org/pdf/2602.03177v1)

**作者:** Gautami Golani `[一作]` (Nanyang Technological University), Domenico Campolo `[通讯]` (Nanyang Technological University)

**通讯引用:** 2946 | [OpenAlex ID](https://openalex.org/A5079258091)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种仅利用标记运动捕捉数据、无力板的脚步反作用力（GRF）估计与分解方法，并通过16段身体模型计算质心(CoM)，再利用牛顿第二定律与最小化原理得到双足GRF分量。

**💡 创新点**

创新点在于：①利用完整16段分段模型计算全身质心，提升质心估计精度；②在双支撑阶段通过最小化惯性加速度变化的物理模型实现GRF的无歧义分解；③实现了无力板连续步态下双足GRF的实时估计，解决了传统力板对空间与时间的限制。

**🔧 技术方法**

技术包括：多段刚体模型质心估计、局部坐标系构建、牛顿第二定律推导、最小化（最小速率变化）求解、双足GRF分解公式、Butterworth滤波、与Visual3D的对比验证。

**📊 数据集**

使用了10米步态测试（10m Walk Test）数据集，包含20名健康受试者（21-80岁）在Miqus M3运动捕捉系统与Kistler力板同步记录的标记与力学数据。

**📈 对比分析**

方法与传统力板测量对比：质心在Y轴的RMSE为0.31–0.52 cm，X轴为3.93–6.19 cm，Z轴经校正后差值≤2.93 cm；GRF的垂直、前后、左右分量与力板数据显示出高度一致的双峰特征，误差主要在0.1–0.2 s的时间偏差与峰值幅度误差，整体性能可满足临床应用所需的精度。

**⚠️ 局限性**

局限性：①对病理性步态（如中风、骨关节炎等）的准确性尚未验证；②质心估计存在固定垂直偏差，需校正；③依赖高质量标记数据，标记误差或遮挡会影响估计；④模型假设为刚体且使用固定质量比例，个体差异可能导致误差。

---

## 365. When Attention Betrays: Erasing Backdoor Attacks in Robotic Policies by Reconstructing Visual Tokens

**arXiv ID:** 2602.03153 | [PDF](https://arxiv.org/pdf/2602.03153v1)

**作者:** Xuetao Li `[一作]` (Wuhan University), Miao Li `[通讯]` (Wuhan University)

**通讯引用:** 7473 | [OpenAlex ID](https://openalex.org/A5088452779)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无需重新训练的测试时刻后门消除框架Bera，用于保护视觉‑语言‑动作模型在机器人操控中的安全性。

**💡 创新点**

创新点在于揭示后门利用深层注意力抓取机制，将异常视觉标记在潜在空间定位，并通过解码器重构触发器自由图像，从而破坏触发器‑动作关联。

**🔧 技术方法**

核心技术包括特征引导后门定位（FBL）、基于深层注意力的过滤（AFM）以及基于MAE的局部解码重建。

**📊 数据集**

使用了真实机器人抓取数据集，涵盖四个任务（抓取Fanta、举立立方、提取组织、握手）在四台不同平台上的共1600条演示，并合成多种触发器（红色瓶盖、圆块、棋盘）。

**📈 对比分析**

与六种现有防御方法（ZIP、UNICORN、BTI‑DBF、SampDetox、SparseVLM、DeDe）比较，Bera在攻击成功率从约90%降至约3–7%，同时保持清洁性能下降不超过3%，实现综合指标TP最高，恢复性能平均超过70%。

**⚠️ 局限性**

局限性包括对语义融合触发器（如与场景语义紧密结合的触发器）检测效果略弱，且在极高的污染率或极小触发器比例下性能会有轻微下降。

---

## 366. Enhancing Foundation VLM Robustness to Missing Modality: Scalable Diffusion for Bi-directional Feature Restoration

**arXiv ID:** 2602.03151 | [PDF](https://arxiv.org/pdf/2602.03151v1)

**作者:** Wei Dai `[一作]` (Xi’an Jiaotong University), Haixia Bi `[通讯]` (School of Information and Communications Engineering, Xi’an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于扩散模型的中间阶段恢复模块，用于在缺失模态情况下恢复视觉语言模型的特征，从而提升VLM在不完整输入下的鲁棒性。

**💡 创新点**

引入动态模态门控机制精细调控生成过程，并设计跨模态互学习闭环，实现在缺失模态时的双向语义对齐与高质量特征恢复。

**🔧 技术方法**

利用改进的Diffusion Transformer (DiT) 进行特征级扩散，结合动态门控与互学习损失，配合DDIM采样实现高效恢复。

**📊 数据集**

在大规模图文对齐数据（CC3M、COCO）上做中间阶段预训练，并在MM‑IMDb、N24News、MMHS11K、Food101等四个基准任务上评估。

**📈 对比分析**

与提示式、检索式与生成式缺失模态方法对比，零样本下在70%缺失率时实现F1/MAC、ACC分别提升约7–8%，整体领先所有基线。

**⚠️ 局限性**

对高阶模态数仍有限制，且需在完整预训练数据上进行中间阶段训练，缺乏在线自适应更新机制。

---

## 367. Adversarial construction as a potential solution to the experiment design problem in large task spaces

**arXiv ID:** 2602.03172 | [PDF](https://arxiv.org/pdf/2602.03172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 368. Understanding Multi-Agent LLM Frameworks: A Unified Benchmark and Experimental Analysis

**arXiv ID:** 2602.03128 | [PDF](https://arxiv.org/pdf/2602.03128v1)

**作者:** Abdelghny Orogat `[一作]` (Concordia University), Essam Mansour `[通讯]` (Concordia University)

**通讯引用:** 1382 | [OpenAlex ID](https://openalex.org/A5042458153)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了多代理LLM框架的架构对系统性能的影响，提出了一套架构分类法并基于该法构建统一基准MAFBench，开展了对主流框架的对比实验。

**💡 创新点**

创新点在于①将多代理框架拆解为控制流、代理抽象、通信、存储和执行语义五大维度的架构分类；②构建统一基准MAFBench，将各类单代理与多代理基准整合至同一执行管道；③在固定LLM与任务的前提下，系统量化架构决策对延迟、吞吐、准确率与协同成功率的影响。

**🔧 技术方法**

技术包括：基于图、角色和GABM三种执行范式的框架抽象；统一的基准执行管道和日志采集；在固定模型下通过隔离单一架构维度进行对比实验；对实验结果进行统计分析与可视化。

**📊 数据集**

使用了现有的单代理和多代理评测数据集，如内存检索基准、规划基准、工具使用基准与多代理协同任务，所有数据集在MAFBench框架中统一加载。

**📈 对比分析**

比较方法为在MAFBench管道中固定LLM、提示与数据，仅改变框架的单一架构维度，测量延迟、吞吐、规划准确率与协同成功率；实验表明，架构差异可导致延迟提升百倍、规划准确率下降30%、协同成功率从90%降至30%。

**⚠️ 局限性**

局限性包括：仅关注框架级架构而未探究更深层次的模型改进；动态网络拓扑与自适应通信尚未实现；实验覆盖的框架与基准仍有限，未来需扩展到更大规模与多样化任务。

---

## 369. Feature, Alignment, and Supervision in Category Learning: A Comparative Approach with Children and Neural Networks

**arXiv ID:** 2602.03124 | [PDF](https://arxiv.org/pdf/2602.03124v1)

**作者:** Fanxiao Wani Qiu `[一作]` (University of Southern California), Oscar Leong `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在一个稀疏标记的半监督类别学习实验中，作者让 5‑7 岁儿童和使用对比学习的卷积神经网络同时面对相同的任务，研究它们在不同特征维度、标记量和对齐度下的学习与归纳能力。

**💡 创新点**

创新点在于首次采用“物种公平”设计，将儿童和模型置于完全相同的学习条件下，并系统探索特征结构、感知对齐和监督程度三者如何交互影响两者的学习表现。

**🔧 技术方法**

使用的技术包括：对比学习框架的 Siamese ResNet‑18 网络、二元交叉熵与对比损失的联合训练、线性混合效应模型对行为和模型性能的统计分析。

**📊 数据集**

数据集为人工设计的六类形状/大小/图案对象样本，包含高对齐和低对齐两种对齐条件，实验中每类在不同标记量（1/6、3/6、6/6）下提供训练与测试样本。

**📈 对比分析**

通过对同一实验设置下的行为与网络分类准确率进行对比，结果显示儿童受对齐度与特征类型影响更大，形状特征更易学习；CNN 则在大小与图案特征上表现更好，对对齐度不敏感，且监督量越多准确率越高；两者的性能差异说明两者在利用弱监督信息和特征归纳时的机制存在本质区别。

**⚠️ 局限性**

局限性包括仅使用 5‑7 岁儿童作为人类样本，缺乏年龄或教育水平的多样性；模型仅采用单一 ResNet‑18 结构，未探讨更复杂或预训练网络的表现；实验对象为人工生成的简单视觉样本，可能难以推广到自然环境中的复杂视觉学习任务。

---

## 370. Behind the Feed: A Taxonomy of User-Facing Cues for Algorithmic Transparency in Social Media

**arXiv ID:** 2602.03121 | [PDF](https://arxiv.org/pdf/2602.03121v1)

**作者:** Haoze Guo `[一作]` (University of Wisconsin), Ziqi Wei `[通讯]` (University of Wisconsin)

**通讯引用:** 4802 | [OpenAlex ID](https://openalex.org/A5006792059)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文通过对六大社交平台UI中的算法透明性提示进行系统化编码，构建了一套涵盖设计形式、信息内容和用户代理三维度的分类体系，并对提示的分布与功能缺口进行分析。

**💡 创新点**

创新点在于提出了专门针对算法透明度提示的多维度分类方法，并首次量化提示在可读性、可验证性与可争议性等功能维度上的差距。

**🔧 技术方法**

采用了定性内容分析方法，基于自定义的编码手册对UI提示进行手工编码。

**📊 数据集**

数据集由210个提示实例组成，覆盖六个平台（Facebook、Instagram、TikTok、YouTube、X、LinkedIn），共计74种独立提示类型。

**📈 对比分析**

通过对不同平台和决策类型的属性分布进行对比，绘制了透明性功能缺口图，发现可读性高但可验证性和争议性低；但论文未给出量化性能指标，仅呈现比例分布。

**⚠️ 局限性**

局限在于仅关注UI层面，未评估提示的真实性或用户理解效果，且结果易受界面更新、地区和设备差异的影响。

---

## 371. StepScorer: Accelerating Reinforcement Learning with Step-wise Scoring and Psychological Regret Modeling

**arXiv ID:** 2602.03171 | [PDF](https://arxiv.org/pdf/2602.03171v1)

**作者:** Zhe Xu `[一作]` `[通讯]` (Independent Researcher), Zhe Xu (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现心理遗憾模型（PRM），通过在每一步计算相对最优行动的遗憾信号来加速强化学习收敛。

**💡 创新点**

创新点在于将行为经济学与认知科学中的“遗憾”与因果推理机制量化为可计算的遗憾信号，并将其嵌入潜在基奖励塑造，形成稠密反馈框架。

**🔧 技术方法**

采用潜在基奖励塑造（PBRS）、基于预训练对手模型的 Q 值近似、PPO 算法以及奖励信号的增量修正。

**📊 数据集**

使用 Gymnasium 的 LunarLander-v3 环境进行实验，作为稀疏奖励与连续控制的典型基准。

**📈 对比分析**

与标准 PPO 对比：PRM 在 200,000 步训练下约快 36% 达到“已解决”阈值（奖励 ≥ 200），最终平均奖励从 140 提升至 300，稳定性和收敛速度均有显著提升。

**⚠️ 局限性**

局限性：需依赖预训练的强对手模型，若无此模型或对手不可用则无法直接使用；此外，维持对手模型会带来额外的计算开销。

---

## 372. Human-in-the-loop Adaptation in Group Activity Feature Learning for Team Sports Video Retrieval

**arXiv ID:** 2602.03157 | [PDF](https://arxiv.org/pdf/2602.03157v1)

**作者:** Chihiro Nakatani `[一作]` (Toyota Technological Institute), Norimichi Ukita `[通讯]` (Toyota Technological Institute)

**通讯引用:** 4712 | [OpenAlex ID](https://openalex.org/A5053167635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在团队运动视频检索中，提出了一种人机协作的自适应微调方法，利用查询视频和少量正负标注视频来更新预训练的组活动特征空间。

**💡 创新点**

将查询感知的视频选择与局部差异度量相结合，实现了针对检索目标的高效数据挑选，并通过对预训练GAFL网络的微调提升检索性能。

**🔧 技术方法**

使用自监督的组活动特征学习（GAFL）网络、对比学习与正则化的triplet损失、查询相似度+局部差异度量的视频选择以及核心集多样性筛选等技术。

**📊 数据集**

在排球、NBA篮球以及Collective Activity三大数据集上进行评估。

**📈 对比分析**

相较于GAFL和其他基线，精度显著提升，例如排球数据集Precision@10从0.557提升至0.739，Hit@10从0.962提升至0.993；NBA数据集Precision@10提升至0.233，Hit@10提升至0.839。

**⚠️ 局限性**

仍需用户提供正负标注视频，对预训练模型依赖较强；在极大多样性视频时可能需要更多样本以避免覆盖不足。

---

## 373. Internet of Agentic AI: Incentive-Compatible Distributed Teaming and Workflow

**arXiv ID:** 2602.03145 | [PDF](https://arxiv.org/pdf/2602.03145v1)

**作者:** Ya-Ting Yang `[一作]` (New York University), Quanyan Zhu `[通讯]` (New York University)

**通讯引用:** 11144 | [OpenAlex ID](https://openalex.org/A5081500464)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于互联网的代理AI框架，通过分布式联盟与工作流协同实现任务执行。

**💡 创新点**

创新点在于将工作流与联盟形成耦合的激励兼容可行性框架，以及基于最小努力的去中心化联盟选择算法。

**🔧 技术方法**

使用图论模型、合作博弈论、工作流DAG、Model Context Protocol（MCP）以及分布式搜索算法。

**📊 数据集**

使用合成的Erdős–Rényi网络与模拟的医疗能力/成本参数，未采用真实医疗数据集。

**📈 对比分析**

通过仿真评估所需跳数、联盟规模与成本，结果表明能力丰富度提升可显著降低协调半径和联盟大小；未与现有方法直接对比。

**⚠️ 局限性**

局限在于仅考虑单一任务、预先已知能力与成本、无并发任务与动态网络变化，以及缺乏对信任与声誉机制的考量。

---

## 374. What Makes a Good Example? Modeling Exemplar Selection with Neural Network Representations

**arXiv ID:** 2602.03144 | [PDF](https://arxiv.org/pdf/2602.03144v1)

**作者:** Fanxiao Wani Qiu `[一作]` (University of Southern California), Alexander LaTourrette `[通讯]` (University of Southern California)

**通讯引用:** 165 | [OpenAlex ID](https://openalex.org/A5000392882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文利用机器学习中的数据集蒸馏方法和神经网络特征表示，对成人在有限示例下教学时的范例选择进行建模，并与人类行为进行比较。

**💡 创新点**

创新点在于将代表性、典型性与多样性等子集选择目标与Transformer/ResNet特征结合，以结构化方式解释人类教学中的权衡，并发现Transformer特征更贴合人类选择。

**🔧 技术方法**

使用预训练的ResNet‑50与ViT‑B/16视觉模型提取特征，基于代表性、典型性、多样性及其组合的子集选择算法（设施定位、距离最大化等）。

**📊 数据集**

使用由三类一维形态连续体（daxes、veps、bems）构成的人工视觉刺激，包含0-100尺度的图像。

**📈 对比分析**

通过对人类实验数据计算原型性与多样性得分，将模型预测与人类得分的平均绝对误差进行比较；结果显示代表性及其与多样性组合的策略在两、三示例条件下与人类行为误差最小，且ViT模型相对ResNet具有更低误差。

**⚠️ 局限性**

局限性包括仅采用单一视觉任务和有限的成人样本、未深入探讨个体差异与发展阶段，以及使用的神经网络特征仍难以解释对人类认知的具体对应机制。

---

## 375. Cyber Insurance, Audit, and Policy: Review, Analysis and Recommendations

**arXiv ID:** 2602.03127 | [PDF](https://arxiv.org/pdf/2602.03127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 376. FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion

**arXiv ID:** 2602.03137 | [PDF](https://arxiv.org/pdf/2602.03137v1)

**作者:** Chen-Bin Feng `[一作]` (University of Macau), Xi Shen `[通讯]` (Intellindust AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种全零训练的少样本目标检测框架FSOD‑VFM，利用视覺基礎模型生成类别无关框并进行置信度重加权。

**💡 创新点**

通过图结构扩散置信度重加权消除提议过度碎片化问题，并将UPN、SAM2与DINOv2无缝集成，实现无训练的高性能检测。

**🔧 技术方法**

利用Universal Proposal Network产生框，SAM2提取精确掩模，DINOv2提取特征，并通过图扩散置信度重加权算法进行置信度调优。

**📊 数据集**

在Pascal‑5^i、COCO‑20^i和跨域CD‑FSOD三大数据集上进行评估。

**📈 对比分析**

与训练‑基和无训练基准对比，FSOD‑VFM在Pascal‑5^i 1‑shot nAP50达77.5，COCO‑20^i 10‑shot nAP50达59.4，CD‑FSOD 10‑shot AP31.6，均显著优于先前方法。

**⚠️ 局限性**

随着样本数增多提升幅度减小，且多模型推理耗时较高，缺乏实时性能。

---

## 377. Enhanced Parcel Arrival Forecasting for Logistic Hubs: An Ensemble Deep Learning Approach

**arXiv ID:** 2602.03135 | [PDF](https://arxiv.org/pdf/2602.03135v1)

**作者:** Xinyue Pan `[一作]` (Georgia Institute of Technology), Benoit Montreuil `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5987 | [OpenAlex ID](https://openalex.org/A5013129686)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一套基于深度学习的集成框架，用实时包裹状态信息与历史到达模式预测物流枢纽的短期到达量。

**💡 创新点**

创新点在于将unordered与ordered包裹分别建模并动态更新，使用ANN预测unordered量、RF预测ordered到达时间，并通过ANN集成网络进一步提升整体预测精度，同时引入实时数据更新机制。

**🔧 技术方法**

采用前馈神经网络（ANN）处理unordered预测、随机森林（RF）预测行程与停留时间、ANN集成网络合并两类预测，并与Holt‑Winters、单一ANN等传统方法对照。

**📊 数据集**

使用Georgia Tech PI Lab Intracity Logistics Simulator生成的30天包裹行程记录，其中27天用于训练/验证，3天用于测试。

**📈 对比分析**

与Holt‑Winters、单一ANN、简单加和集成、ANN集成四种方法比较，采用MASE指标评估；ANN集成方法MASE 0.79，明显优于其他模型，并在各时间段均表现最佳。

**⚠️ 局限性**

局限在于数据量有限（仅30天），未纳入天气、交通等外部因素，对大规模网络的可扩展性和部署仍需进一步验证。

---

## 378. Contrastive Concept-Tree Search for LLM-Assisted Algorithm Discovery

**arXiv ID:** 2602.03132 | [PDF](https://arxiv.org/pdf/2602.03132v1)

**作者:** Timothee Leleu `[一作]` (NTT Research), Surya Ganguli `[通讯]` (Stanford University)

**通讯引用:** 17431 | [OpenAlex ID](https://openalex.org/A5056551357)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

研究了一种基于对比概念树搜索（CCTS）的LLM辅助算法发现框架，通过在程序空间中构建语义概念层次并利用对比统计来引导父子选择，提升搜索效率并产生可解释的概念树。

**💡 创新点**

创新点在于将LLM内部表示显式化为层次化概念空间，并用对比学习的概率模型（类似TPE）来评估概念的正负效用，从而在父选择时偏向有益概念，避免无效概念。

**🔧 技术方法**

采用对比学习、交叉熵更新、树结构概率估计（Tree-structured Parzen Estimator）、LLM生成与提示工程、概念提取与动态树构建，以及基于阈值的好坏分割。

**📊 数据集**

评估数据集为从 Erdős 风格组合问题构成的基准，包括圆形拼接、算术 Kakeya、Heilbronn 三角形、正方形嵌入等真实任务，并构造了一个仿真环境作为对照。

**📈 对比分析**

与均匀、贪婪、k-精英等传统父选择基线相比，CCTS 在 25 轮迭代内获得更高的最高分，曲线显示更快提升，且在多任务上持续优于基线；在仿真任务中也复现相似优势。

**⚠️ 局限性**

限制主要包括：对上尾（高分）提升有限；缺乏对概念互相关和跨父组合的建模；对 LLM 提示质量敏感；搜索规模有限；未结合岛屿或 MAP‑Elites 等群体结构。

---

## 379. One Model, All Roles: Multi-Turn, Multi-Agent Self-Play Reinforcement Learning for Conversational Social Intelligence

**arXiv ID:** 2602.03109 | [PDF](https://arxiv.org/pdf/2602.03109v1)

**作者:** Bowen Jiang `[一作]` (University of Pennsylvania), Sihao Chen `[通讯]` (Microsoft Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出OMAR框架，让单一模型同时扮演多角色进行多轮多代理对话自我训练，以获得社交智能。

**💡 创新点**

创新地将自我对话训练与多轮多代理场景结合，采用层级优势估计缓解长序列高方差，并通过终端奖励学习细粒度社交行为。

**🔧 技术方法**

使用强化学习（PPO）+层级优势估计，模拟多轮对话自我对抗，模型角色扮演与奖励设定。

**📊 数据集**

SOTOPIA（目标驱动对话）和Werewolf（零和社交推理游戏）数据集。

**📈 对比分析**

与基线SOTOPIA‑RL和单轮训练模型对比，实验显示在SOTOPIA细粒度指标上提升显著，在Werewolf中赢率从55%升至72%，证明多轮多代理自我训练能提升协作与竞争表现。

**⚠️ 局限性**

限制包括并行发言的近似导致异步对话缺失、批大小受参与者数限制、奖励稀疏导致易被奖励劫持，以及小模型在长序列与复杂推理上的不足。

---

## 380. Consensus Group Relative Policy Optimization for Text Generation

**arXiv ID:** 2602.03102 | [PDF](https://arxiv.org/pdf/2602.03102v1)

**作者:** Yuki Ichihara `[一作]` (Nara Institute of Science and Technology), Eiji Uchibe `[通讯]` (Advanced Telecommunications Research Institute International)

**通讯引用:** 4016 | [OpenAlex ID](https://openalex.org/A5031054137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Consensus‑GRPO (C‑GRPO)，将基于 consensus 的 MBR 解码规则在训练阶段进行无参考蒸馏，使模型在推理时仅需单前向推理即可产生高质量输出。

**💡 创新点**

创新点：1）用组相对优势（group‑relative advantage）在 GRPO 中直接构造 consensus utility，完全不依赖金标准或奖励模型；2）在理论上证明 C‑GRPO 的期望更新方向与 MBR 目标梯度对齐，给出非渐进收敛保证；3）实验证明在无额外推理成本的前提下，C‑GRPO 能超越传统 MBR 及多种基线。

**🔧 技术方法**

技术：Group Relative Policy Optimization (GRPO)、Minimum Bayes Risk (MBR) 解码、对话/文本的相似度度量（BLEURT、COMET、ROUGE‑L），强化学习中的优势估计与策略梯度；理论分析基于光滑性、方差界定及标准化独立性假设。

**📊 数据集**

数据集：机器翻译 WMT 2024 (En→Ja、En→Zh、En→De)，文本摘要 XSum，日语多项选择问答基准 JBBQ；在这些任务上使用无参考训练。

**📈 对比分析**

对比方法：GRPO+随机奖励、GRPO+自评（LLM‑judge）、SFT+MBR、参考基准（Supervised‑Fine‑Tuning、DPO、Best‑of‑N）以及传统 MBR。结果显示 C‑GRPO 在 COMET、ROUGE‑L、JBBQ 准确率等指标上均优于 MBR，且在单前向推理下实现更高或相当的性能。

**⚠️ 局限性**

局限性：对 utility 函数的质量敏感；在极小模型（如 Qwen2‑0.5B/1.5B）上可能不稳定；理论分析假设标准化项与梯度方向独立，实际可能有偏；需要足够的采样数量，群体大小对性能有影响。

---

## 381. MemCast: Memory-Driven Time Series Forecasting with Experience-Conditioned Reasoning

**arXiv ID:** 2602.03164 | [PDF](https://arxiv.org/pdf/2602.03164v1)

**作者:** Xiaoyu Tao `[一作]` (State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China), Shijin Wang `[通讯]` (iFLYTEK Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出MemCast框架，将时间序列预测改造成经验条件推理任务，构建分层记忆并在推理时利用记忆进行指导、轨迹选择和反思，实现持续演化。

**💡 创新点**

创新点在于：①把训练得到的经验分为历史模式、推理智慧与通用规律三层记忆；②在推理时动态检索并利用记忆；③设计不泄漏测试分布的动态置信度自适应更新机制，支持模型持续演化。

**🔧 技术方法**

采用LLM（GPT‑5）作为推理引擎，结合经验聚合、检索与语义相似度评分、多路径生成与筛选、规则反思以及动态置信度更新等技术。

**📊 数据集**

使用多领域时序数据集：电价（NP、PJM、BE、FR、DE）、ETT（ETTh、ETTm）、风电（WP）、光伏（SP）、水文流量（MOPEX）等，共计十余个不同背景的数据集。

**📈 对比分析**

与传统统计（ARIMA、Prophet）、深度学习（PatchTST、iTransformer、TimeXer、ConvTimeNet、DLinear）以及其他LLM基线（LSTPrompt、Time‑LLM、TimeReasoner）对比，使用 MSE/MAE 评估。MemCast 在多数数据集上获得最优或次优成绩，特别是在高波动、噪声多的场景中显著优于现有方法。

**⚠️ 局限性**

局限性：①对预训练LLM的依赖导致推理成本较高；②多路径生成与检索计算开销大；③在极端离群点或极端数据分布下仍可能产生推理错误；④动态置信度参数需手工调节，缺乏自动化。

---

## 382. Intelligent Front-End Personalization: AI-Driven UI Adaptation

**arXiv ID:** 2602.03154 | [PDF](https://arxiv.org/pdf/2602.03154v1)

**作者:** Mona Rajhans `[一作]` `[通讯]` (Palo Alto Networks), Mona Rajhans (Palo Alto Networks)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了基于 LSTM 预测用户行为与强化学习优化内容排序的前端实时自适应系统。

**💡 创新点**

将序列预测与强化学习联合到同一闭环，使界面可实时调整布局与内容，显著优于传统规则化方案。

**🔧 技术方法**

使用 LSTM 进行路径预测，DQN（深度 Q 网络）进行内容优先级排序，前端采用 React，后端采用 Python Flask。

**📊 数据集**

使用合成的 SOC 仪表盘交互数据以及真实 Web 日志进行验证。

**📈 对比分析**

与规则化基线在 100 名受试者的 SOC 仪表盘实验中比较，CTR 提升 38%，任务成功率提升 28%，平均会话时长增长 27%。

**⚠️ 局限性**

假设用户行为分布稳定，缺乏多模态感知与可解释性，且未将认知负荷等指标纳入奖励函数。

---

## 383. FASA: Frequency-aware Sparse Attention

**arXiv ID:** 2602.03152 | [PDF](https://arxiv.org/pdf/2602.03152v1)

**作者:** Yifei Wang `[一作]` (Alibaba Group), Julian McAuley `[通讯]` (UCSD)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大语言模型 KV 缓存在长上下文推理中的内存与带宽瓶颈，提出一种训练无关、基于 RoPE 频率块的查询感知 token 重要性预测与稀疏注意力框架 FASA，动态剔除无关 token 并聚焦关键上下文；

**💡 创新点**

创新点在于：①发现 RoPE 在频率块层面表现功能稀疏，仅少量主导频率块即可近似完整注意力；②基于此构建两阶段无训练的动态 token 预测与聚焦注意力机制；③提供两种硬件友好的变体（-M 兼顾内存、-C 兼顾速度），实现低成本高效的 KV 缓存压缩；

**🔧 技术方法**

采用 RoPE 频率块分解、Contextual Agreement（CA）度量主导频率块、离线校准主导集合、两阶段 Token Importance Prediction（TIP）与 Focused Attention Computation（FAC）以及 GPU/CPU KV 缓存的动态 off‑load 与 pre‑fetch；

**📊 数据集**

使用 LongBench、LongBench‑V1、LongCoT（MATH500、AIME24）、PG‑19、WikiText、C4、NarrativeQA、TREC、SQuAD 等多种长文本与推理基准数据集；

**📈 对比分析**

与 Stream、SnapKV、Quest、RKV、H2O、PyramidKV 等主流 token‑eviction 与压缩方法对比，FASA 在极低预算（如 256 token、18.9% KV）下仍保持≈100% full‑KV 性能，速度提升达 2.56×，内存节省多达 8×，整体表现优于所有基线且接近 oracle 上限；

**⚠️ 局限性**

局限性：需要一次离线校准主导频率块，虽然成本低但仍依赖校准数据；主导集合随模型规模/头数变化需重新校准；对极端长序列的泛化仍有限；方法本身不解决模型的内容偏见或质量问题。

---

## 384. General Agents Contain World Models, even under Partial Observability and Stochasticity

**arXiv ID:** 2602.03146 | [PDF](https://arxiv.org/pdf/2602.03146v1)

**作者:** Santiago Cifuentes `[一作]` `[通讯]` (Dovetail Research), Santiago Cifuentes (Dovetail Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

证明任何具备足够推理能力的代理（包括随机策略和在部分可观测环境中的策略）都能通过询问其行为来近似重构其所处的受控马尔可夫决策过程（cMDP）的转移概率，从而说明代理必然包含对世界的内部模型。

**💡 创新点**

① 在原有只对确定性、完全可观测代理的结果基础上，扩展到随机策略并给出更弱的误差界（依赖δ<½，收敛率为O(1/√n)）。
② 将结果推广到部分可观测cMDP，证明即使状态信息被隐藏，最优代理仍能恢复转移概率。
③ 通过只使用宽度为2的目标（而非原来宽度为O(n)的目标）实现更快的误差下降（O(log n/n)），并大幅减少所需目标的数量。

**🔧 技术方法**

利用概率论工具（二项分布、Bernstein‑Freedman不等式、Lambert W函数）、递归/归纳证明、目标语义化（LTL子句）以及对随机/部分可观测策略的概率分析。

**📊 数据集**

该工作不依赖任何具体数据集；所有结果均为理论证明，验证基于构造的cMDP实例。

**📈 对比分析**

与原文中的定理（误差O(1/√n)、目标宽度为O(n)）相比，本工作在误差收敛速度上更快、目标集合规模更小；在随机策略和部分可观测情形下，尽管误差常数更大，但依旧能实现世界模型重构；实验验证未给出，全部为理论上限。

**⚠️ 局限性**

限制包括：随机策略结果需δ<½，导致误差常数增大；部分可观测推广仍假设目标以真实状态为语义，若目标仅以观测表述则可能无法恢复转移概率；对极小的δ或更复杂的部分可观测设置尚未完全覆盖；并且在实际应用中需实现对大量目标的查询，计算成本仍较高。

---

## 385. Self-Hinting Language Models Enhance Reinforcement Learning

**arXiv ID:** 2602.03143 | [PDF](https://arxiv.org/pdf/2602.03143v1)

**作者:** Baohao Liao `[一作]` (Microsoft Research), Jiang Bian `[通讯]` (Microsoft Research)

**通讯引用:** 13507 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SAGE 框架，在训练期间通过自我生成的“提示”（即对参考解答的压缩）来调节大语言模型的采样分布，从而解决 GRPO 在稀疏奖励下的优势归零问题，并在推理时不需要任何提示。

**💡 创新点**

创新点包括：
1) 将提示作为上下文插入，保持完整的 on‑policy 目标；
2) 设计了基于政策的动态提示强度调度器，只在群组奖励崩塌时才启用提示，形成自适应课程；
3) 引入在线自我提示机制，使提示分布始终与学习者当前能力对齐，避免了固定或外部教师提示的校准误差；
4) 对 GRPO 的标准化优势进行了理论分析，说明提示能最大化“开启门”概率。

**🔧 技术方法**

技术手段包括：
- 基于 GRPO 的优势标准化和 KL 正则化的 on‑policy 强化学习；
- 通过参考解答生成 lossy hints；
- 随机采样提示强度并按政策指标动态调整；
- 在线提示生成器（基于当前模型的滞后副本）与自适应提示分布；
- 训练时使用 8 条轨迹/提示，群组大小 G，使用 KL 重正则化等。

**📊 数据集**

使用的数据集：
- 训练集：从 OpenR1‑Math‑220k 提取的 64k 个题目，经过 Math‑Verify 过滤后再随机抽取 15k 个；
- 参考解答来源：DeepSeek‑R1 生成的推理轨迹；
- 评估集：六个数学基准（AIME24、AIME25、AMC23、MATH‑500、Minerva Math、OlympiadBench）和两个通用基准（GPQA‑diamond、MMLU‑Pro）。

**📈 对比分析**

比较方法：SFT、GRPO、LUFFY、Scaf‑GRPO、SAGE‑light；实验表明：
- SAGE 在三种 LLM（Llama‑3.2‑3B、Qwen2.5‑7B、Qwen3‑4B）上平均提升 4–6% 的准确率；
- 在稀疏奖励任务中，SAGE 将未产生学习信号的提示比例从 GRPO 的 40% 下降到 30%（Llama‑3.2）；
- 相比外部教师提示方法（LUFFY、Scaf‑GRPO），SAGE 在保持 on‑policy 性能的同时实现更高的泛化和更稳定的训练；
- SAGE‑light 在保持相近性能的前提下将训练时间降低约 47%。

**⚠️ 局限性**

局限性：
- 训练时需要在线生成提示，导致推理延迟和计算成本上升；
- 需要参考解答或较高质量的外部模型来生成初始提示，且对提示质量敏感；
- 在极大模型或极长推理任务中，提示生成与采样成本可能成为瓶颈；
- 目前只针对可验证的 0/1 奖励任务，针对更复杂或连续奖励的适用性尚待验证。

---

## 386. Analyzing Zigbee Traffic: Datasets, Classification and Storage Trade-offs

**arXiv ID:** 2602.03140 | [PDF](https://arxiv.org/pdf/2602.03140v1)

**作者:** Antonio Boiano `[一作]` (Politecnico di Milano), Alessandro E. C. Redondi `[通讯]` (Politecnico di Milano)

**通讯引用:** 1936 | [OpenAlex ID](https://openalex.org/A5075395997)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在两种不同拓扑的真实智能家居环境中采集21款 Zigbee 设备的网络流量，构建公开数据集 ZIOTP2025，并在该数据集上实现并评估基于统计特征与 XGBoost 的设备类型与设备身份识别，以及针对 Zigbee 流量的特征量化压缩与存储优化。

**💡 创新点**

创新点在于首次公开多拓扑、带标签的 Zigbee 流量数据集；系统研究拓扑变化对流量分类模型泛化的影响；提出利用特征量化实现 4–5 倍压缩且几乎不损失分类性能的存储方案。

**🔧 技术方法**

技术手段包括：时窗统计特征提取、XGBoost 多类别分类、交叉验证评估、lossless 压缩（GZIP/BZIP2/7Z-LZMA）与 lossy 量化压缩、以及对比实验。

**📊 数据集**

使用的数据集为 ZIOTP2025，包含 252,896 条 IEEE 802.15.4 帧、21 设备、两种拓扑、约 17 MB/拓扑，涵盖多种传感器与执行器。

**📈 对比分析**

通过 5 折交叉验证对 intra‑topology 与 cross‑topology 情景进行比较，设备类型分类在同拓扑下 F1≈0.95，跨拓扑下降至约0.75；设备身份识别同样从 0.95 降至 0.4–0.7；在存储方面，lossless 原始流量压缩可降至约 650 bps，量化特征压缩可实现 4–5 倍存储减小，且在 150–200 bit/s 的压缩率下保持 0.9–0.94 的宏 F1 分数。

**⚠️ 局限性**

局限性包括：跨拓扑泛化能力弱，尤其是细粒度设备识别；特征化简后无法进行后续的包级取证分析；数据集中设备分布不均，缺少更多拓扑和设备类型的多样性。

---

## 387. Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis

**arXiv ID:** 2602.03139 | [PDF](https://arxiv.org/pdf/2602.03139v1)

**作者:** Tianhe Wu `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 9490 | [OpenAlex ID](https://openalex.org/A5020029652)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无额外模块的few-step扩散蒸馏框架，利用首步多样性监督和后续标准DMD实现对样本多样性与图像质量的兼顾。

**💡 创新点**

通过将蒸馏步骤拆分为“多样性保留”与“质量细化”两阶段，并在首步使用流匹配的v‑prediction目标、梯度停止实现角色分离，避免了传统DMD中的模式坍塌。

**🔧 技术方法**

基于流匹配与分布匹配蒸馏（DMD）框架，采用v‑prediction监督、梯度停止、无感知网络或判别器的轻量化实现；评估使用DINO、CLIP、VQ‑R1、MIQA、ImgR、PicS等指标。

**📊 数据集**

使用DiffusionDB文本提示数据集进行训练；在Pick‑a‑Pic、COCO‑10K、GenEval等公开数据集上进行多样性、质量与指令遵循性评估。

**📈 对比分析**

与标准DMD、DMD‑LPIPS、DMD‑GAN以及原始SD3.5‑M/SDXL教师模型等进行对比，采用4步NFEs实现，在保持或提升DINO/CLIP多样性分数的同时，VQ‑R1/MIQA质量和人类偏好（ImgR/PicS）与基线相当或更优，且无额外感知网络或判别器，计算成本更低。

**⚠️ 局限性**

只在第一步提供多样性监督，后续步骤仍可能影响全局属性，适用性受限；对极端提示或强引导时多样性提升有限；未来可探索自适应或多步多样性监督以进一步提升鲁棒性。

---

## 388. SATORIS-N: Spectral Analysis based Traffic Observation Recovery via Informed Subspaces and Nuclear-norm minimization

**arXiv ID:** 2602.03138 | [PDF](https://arxiv.org/pdf/2602.03138v1)

**作者:** Sampad Mohanty `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**通讯引用:** 23683 | [OpenAlex ID](https://openalex.org/A5063784062)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种基于子空间信息的低秩矩阵补全框架 SATORIS-N，用于交通密度矩阵的缺失数据恢复。

**💡 创新点**

创新点包括①引入显式子空间约束的半正定规划（SDP）形式的核范数最小化；②提出轻量级隐式子空间对齐的矩阵拼接方法；③利用临近日子稳定的奇异子空间作为先验。

**🔧 技术方法**

使用了核范数最小化、半正定规划、奇异值分解、矩阵拼接、软阈值技术，以及 CVXPY 求解器进行优化，并与统计、低秩与深度学习基线对比。

**📊 数据集**

使用了北京和上海的出租车 GPS 生成的交通密度矩阵数据集，尺寸分别为 340×24 和 320×24。

**📈 对比分析**

在 10%–90% 随机缺失率下与十种基线（统计、低秩、深度学习）对比，SATORIS-N 在中高缺失率（75%–90%）下 RRMSE、MAE 分别下降 10%–20%，显著优于基线；隐式方法在低缺失率下更快、性能更好，显式方法在高缺失率下更稳健。

**⚠️ 局限性**

局限性包括：需要邻近完整日的数据以估计子空间；显式 SDP 计算成本较高；假设缺失随机且子空间随时间变化缓慢，对快速变化或完全缺失的日子适应性差。

---

## 389. Digital Lifelong Learning in the Age of AI: Trends and Insights

**arXiv ID:** 2602.03114 | [PDF](https://arxiv.org/pdf/2602.03114v1)

**作者:** Geeta Puri `[一作]` (Singapore University of Technology and Design), Dorien Herremans `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1715 | [OpenAlex ID](https://openalex.org/A5069548004)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过两份覆盖四个地区（印度、新加坡、英国、美国）的在线调查，系统研究成人和终身学习者对数字学习平台、游戏化机制和AI（LLM）工具的使用动机、偏好及其对学习效果的影响。

**💡 创新点**

① 在成人群体中首次将年龄、性别、地区与平台偏好及动机进行细分分析；② 结合AI使用频率与准确性、伦理担忧，对LLM在终身学习中的利弊提供定量与定性评估。

**🔧 技术方法**

使用Python与Seaborn进行数据可视化、Wilcoxon符号秩检验进行差异检验，问卷设计采用闭合式与开放式问题；AI工具主要为ChatGPT等生成式预训练模型。

**📊 数据集**

调查样本共200+名受访者（Survey A n=119，Survey B n=81），来自四个国家，通过社交媒体招募，形成自填问卷数据集。

**📈 对比分析**

采用描述性统计、交叉表、箱线图和Wilcoxon检验比较疫情前后学习相关度、不同人口群体对游戏化和AI工具的评价；结果显示疫情后学习相关度显著提升，女性与年轻人更倾向于使用AI工具，但约60%受访者对其准确性不满。

**⚠️ 局限性**

样本受限于社交媒体招募，可能存在数字化倾向偏倚；样本量相对较小，跨地区比较缺乏控制变量；研究为横断面设计，无法推断因果关系。

---

## 390. ChemPro: A Progressive Chemistry Benchmark for Large Language Models

**arXiv ID:** 2602.03108 | [PDF](https://arxiv.org/pdf/2602.03108v1)

**作者:** Aaditya Baranwal `[一作]` (University of Central Florida), Shruti Vyas `[通讯]` (University of Central Florida)

**通讯引用:** 419 | [OpenAlex ID](https://openalex.org/A5072711888)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ChemPro基准，系统评测了45个公开模型和7个商业模型在化学教育层级的问答任务上表现。

**💡 创新点**

创新点在于基于课程衔接的逐级难度划分、源对齐的难度出处、混合多项选择和数值题型，并构建了从小学到高中全面覆盖的化学知识库。

**🔧 技术方法**

技术包括多层验证流程（来源校验、专家复核、AI辅助一致性检查）、对题型的精确标注、使用COT/自评推理提示以及容差评分与精确匹配两种数值评测方式。

**📊 数据集**

数据集由4100道题组成，来源于NCERT教材（9-12年级）、JEE Mains（2020-2024）以及网络测验，划分为四个难度层级（E、M、C、D）。

**📈 对比分析**

对比方法为Pass@1的多项选择准确率与容差/精确匹配的数值准确率；与历史人类成绩对照，商业模型在D级达到约76%准确率，最大公开模型约68-71%，数值题表现仍显不足。

**⚠️ 局限性**

局限性：仍低于人类水平，参数扩展无法显著提升多步推理能力，工具增强框架（如ChemCrow）对高级题型帮助有限，且不同化学分支表现差异明显。

---

## 391. Gromov Wasserstein Optimal Transport for Semantic Correspondences

**arXiv ID:** 2602.03105 | [PDF](https://arxiv.org/pdf/2602.03105v1)

**作者:** Francis Snelgar `[一作]` (Australian National University), Akshay Asthana `[通讯]` (Seeing Machines)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于Gromov–Wasserstein最优传输的语义匹配算法，用单一模型（DINOv2）特征直接实现空间一致性和对称约束，替代传统使用Stable Diffusion特征的方案。

**💡 创新点**

创新点在于将空间平滑性和对称约束内嵌进OT目标函数，并采用部分非平衡OT解决尺度差异与遮挡问题，从而在保持高匹配质量的同时显著降低计算与显存需求。

**🔧 技术方法**

主要技术包括：DINOv2特征提取、最优传输（Kantorovich + Gromov–Wasserstein）、对称约束正则化、非平衡OT与投影梯度下降求解。

**📊 数据集**

实验数据集涵盖TSS（车辆）、PF-PASCAL（20类）和SPair‑71k（18类）三大语义匹配数据集。

**📈 对比分析**

与现有Zero‑shot方法（如基于Stable Diffusion的SCOT、DIFT等）比较，本文在TSS上实现了SOTA水平，在PF‑PASCAL上与同类方法相当，并在SPair‑71k上虽略逊于SD特征方法，但明显优于最近邻基线；同时计算速度比SD方法快5–10倍，显存消耗更低。

**⚠️ 局限性**

局限性包括：对极端尺度/姿态变化的空间一致性假设易失效；对称约束仅适用于中等姿态变化；低分辨率patch特征导致小目标关键点归并，影响精度；以及需要手工设定GW阈值 δ_min/δ_max。

---

## 392. Risky-Bench: Probing Agentic Safety Risks under Real-World Deployment

**arXiv ID:** 2602.03100 | [PDF](https://arxiv.org/pdf/2602.03100v1)

**作者:** Jingnan Zheng `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60270 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大语言模型代理在真实部署环境中的安全风险进行系统评估，构建了Risky-Bench评估框架；

**💡 创新点**

创新在于将领域无关的安全原则映射为可操作的安全rubric，并通过真实任务与多层威胁模型自动化探测风险，提供可扩展的评估管线；

**🔧 技术方法**

采用结构化提示对攻击表面进行攻击策略实施，使用LLM作为判别器评估轨迹并辅以人工校验；

**📊 数据集**

使用VitaBench提供的三类生活辅助任务（外卖、门店、旅游）共750个任务，结合15条安全rubric与5个攻击表面；

**📈 对比分析**

与七大先进代理模型（Gemini‑3、GPT‑4.1、Claude Haiku 4.5、Qwen‑Plus、DeepSeek‑V3.2、kimi‑k2‑0905、Doubao‑Seed‑1.8）对比，计算攻击成功率（ASR）平均在25%–60%之间；在更高级威胁模型下部分模型ASR可超80%，展示了风险随攻击访问级别提升而显著；

**⚠️ 局限性**

局限在于仅覆盖生活辅助场景，依赖VitaBench的真实性与完整性，难以直接泛化至其他领域；评估主要基于LLM判别与人工校验，存在主观性和对评测环境的依赖。

---

## 393. Collision Detection with Analytical Derivatives of Contact Kinematics

**arXiv ID:** 2602.03250 | [PDF](https://arxiv.org/pdf/2602.03250v1)

**作者:** Anup Teejo Mathew `[一作]` (Khalifa University of Science and Technology), Federico Renda `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了iDCOL框架，利用严格凸隐式表面实现可微碰撞检测与接触动力学，解决非严格凸几何导致的非唯一性与不可微问题。

**💡 创新点**

创新点在于：①通过LogSumExp/超几何将非严格凸形状正则化为严格凸隐式表面；②将碰撞检测转化为6×6固定大小的KKT根问题；③利用隐函数定理得到闭式可微接触量与雅可比；④设计高效的Newton求解器与预热/后续策略。

**🔧 技术方法**

主要技术包括：隐式曲面表示（LogSumExp、超椭圆、超圆柱等）、缩放优化（统一缩放因子α）、隐函数定理、Newton-Armijo迭代、几何重参数化与束缚、以及热启动/持续化。

**📊 数据集**

实验使用合成几何集合（多面体、截锥、超椭球、超圆柱）及其组合，仿真场景包括四旋翼路径规划、刚体多体碰撞、软机器人与刚体交互等，未使用公开真实数据集。

**📈 对比分析**

与DCOL进行比较：在多面体-多面体碰撞下iDCOL平均约2.1倍慢，但在椭球-椭球及多面体-椭球场景下速度相当或更快；在所有案例中热启动可将求解时间从数十微秒降至约2微秒；对接触量、接触点、法向量等的可微性在梯度优化中显著提升。

**⚠️ 局限性**

局限性：①需要调节β或n以平衡几何精度与凸性，过大会导致与原始形状偏差；②仅适用于凸形状，非凸体需进一步拓展；③在极锐角或高曲率场景下仍可能出现数值不稳定，需要后续策略；④实现与求解器的细节依赖于手工实现的Newton框架。

---

## 394. The Necessity of a Unified Framework for LLM-Based Agent Evaluation

**arXiv ID:** 2602.03238 | [PDF](https://arxiv.org/pdf/2602.03238v1)

**作者:** Pengyu Zhu `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 4761 | [OpenAlex ID](https://openalex.org/A5036865453)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套统一的LLM代理评估框架，梳理并解决了现有评估中因环境、提示、记忆、工具等导致的变异问题。

**💡 创新点**

创新点包括：①将Sandbox与Evaluation Methodology系统拆分并具体化；②提出统一的Agent系统架构与三元数据集（指令集、工具集、环境集）；③构建多维评估指标与统一的失败分类。

**🔧 技术方法**

技术手段主要包括Python标准工具协议、固定提示与规划范式、使用vLLM/AutoGPT等执行平台、pass@k鲁棒评估、Judge-LLM文本校验以及token/步骤/延迟等资源度量。

**📊 数据集**

建议使用版本控制、静态化的环境集（如BrowseComp-Plus、BFCL快照）以及公开的指令集与工具定义，形成标准化的数据集；论文未给出具体实验数据，但以现有benchmark为参考。

**📈 对比分析**

评估方法采用pass@k、任务完成度、环境状态一致性、资源消耗等多维度指标，旨在实现可复现、可比的比较；论文强调统一框架能显著降低系统性噪声，提升比较可信度，但未给出具体性能提升数字。

**⚠️ 局限性**

局限性包括：标准化可能抑制创新、对实时动态环境适配不足、对新型代理架构兼容性不完善，以及依赖社区广泛采纳才能发挥效果。

---

## 395. Merging Beyond: Streaming LLM Updates via Activation-Guided Rotations

**arXiv ID:** 2602.03237 | [PDF](https://arxiv.org/pdf/2602.03237v1)

**作者:** Yuxuan Yao `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**通讯引用:** 2079 | [OpenAlex ID](https://openalex.org/A5035185924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Streaming Merging框架，将模型合并视作迭代优化过程，并提出了Activation-guided Rotation-aware Merging（ARM）方法；

**💡 创新点**

创新点在于用激活子空间导出的旋转矩阵取代传统线性插值，使合并能突破线性子空间限制，近似梯度下降轨迹；

**🔧 技术方法**

核心技术包括SVD分解激活差异获取旋转矩阵、动态滑动窗口与锚点策略的组合、基于旋转的参数更新；

**📊 数据集**

主要使用NuminaMath（用于SFT）及其衍生的数学与代码基准（GSM8K、MATH500、OlympiadBench、CollegeMath、Minerva-Math、LiveCodeBench、HumanEval-Pro、MBPP-Pro）进行评估；

**📈 对比分析**

与全量微调、LoRA、TA、TIES、DARE、AIM、ACM等基线相比，ARM在7B/14B模型上分别提升约+3.0和+1.9分，甚至在已收敛模型上仍能进一步提升；

**⚠️ 局限性**

局限性包括在RL等高方差任务中的效果不明显、对批量大小和旋转角度的敏感性、以及对超大规模模型仍需更高并行度的实验验证。

---

## 396. Distribution-Aware End-to-End Embedding for Streaming Numerical Features in Click-Through Rate Prediction

**arXiv ID:** 2602.03223 | [PDF](https://arxiv.org/pdf/2602.03223v1)

**作者:** Jiahao Liu `[一作]` (Fudan University), Ning Gu `[通讯]` (Fudan University)

**通讯引用:** 43128 | [OpenAlex ID](https://openalex.org/A5012421463)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在CTR预测的流式学习场景下，提出了DAES框架，用于高效地将数值特征映射为分布感知的嵌入向量，消除传统离线分箱导致的语义漂移；

**💡 创新点**

创新点在于：①引入基于跳跃Reservoir Sampling的在线分位数估计，保持对非平稳分布的鲁棒性；②采用分位数空间插值与字段感知的分布调制（仿射或门控），实现分布先验与上下文信息的自适应融合；③将上述过程完全端到端训练，兼容多种深度CTR骨干网络；

**🔧 技术方法**

使用技术包括：Reservoir Sampling（Jump Reservoir Sampling）、分位数空间热量编码、字段嵌入的仿射/门控调制、Meta-Embedding聚合、深度学习模型（FNN、Wide&Deep、DeepFM、IPNN、DCNv2、xDeepFM）以及在线A/B测试；

**📊 数据集**

实验数据集涵盖公开基准Criteo、AutoML以及百亿级工业广告平台数据；

**📈 对比分析**

与静态分箱、神经嵌入、插值分箱、DAE等四类主流方法在六种CTR骨干上对比；DAES在AUC上平均提升1–3个百分点，LogLoss下降约0.02–0.04；在线A/B实验显示ARPU提升约2.3%，且实现了特征工程的全流程自动化；

**⚠️ 局限性**

局限性包括：①跳跃Reservoir Sampling在极端非i.i.d.流中仍可能产生偏差；②字段调制需要手工选择字段，且对高基数字段适用性有限；③模型对超参数β和Meta-Embedding数量敏感，需要额外调优；④在极大规模实时系统中，分位数估计和调制过程仍带来一定计算与存储开销。

---

## 397. FARTrack: Fast Autoregressive Visual Tracking with High Performance

**arXiv ID:** 2602.03214 | [PDF](https://arxiv.org/pdf/2602.03214v1)

**作者:** Guijie Wang `[一作]` (Xi'an Jiaotong University), Xing Wei `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 35877 | [OpenAlex ID](https://openalex.org/A5100365518)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了多模板自回归跟踪框架FARTrack，结合任务特定自蒸馏和帧间自回归稀疏化，实现高速精准的视觉跟踪。

**💡 创新点**

创新点在于层间任务特定自蒸馏在保持时间序列信息的同时压缩模型，以及无额外运行时开销的帧间自回归稀疏化，全序列范围内实现模板冗余去除。

**🔧 技术方法**

采用ViT‑Tiny Transformer编码器、轨迹自回归序列、KL散度蒸馏、SIoU损失、注意力权重聚合稀疏化和Token Retention策略等技术。

**📊 数据集**

在COCO2017预训练后，在GOT‑10k、TrackingNet、LaSOT、VastTrack、LaSOText、NFS、UAV123等公开基准上进行评测。

**📈 对比分析**

与AsymTrack、MixFormerV2、CompressTracker等主流跟踪器对比，FARTrackpico在GOT‑10k上实现GPU 343 FPS、CPU 121 FPS、AO 70.6%，在各数据集上逼近或优于同等参数级别的顶尖模型，展示最佳速度‑性能平衡。

**⚠️ 局限性**

仍略逊于最顶尖模型的精度，极端遮挡或快速变形时鲁棒性受限，且稀疏化比例需人工调参。

---

## 398. Joint Network-and-Server Congestion in Multi-Source Traffic Allocation: A Convex Formulation and Price-Based Decentralization

**arXiv ID:** 2602.03246 | [PDF](https://arxiv.org/pdf/2602.03246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 399. VIRAL: Visual In-Context Reasoning via Analogy in Diffusion Transformers

**arXiv ID:** 2602.03210 | [PDF](https://arxiv.org/pdf/2602.03210v1)

**作者:** Zhiwen Li `[一作]` (East China Normal University), Yingda Chen `[通讯]` (Alibaba Group)

**通讯引用:** 940 | [OpenAlex ID](https://openalex.org/A5042773283)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出VIRAL框架，通过视觉类比条件生成实现视觉上下文学习，将多种视觉任务统一到RGB空间的生成式任务；

**💡 创新点**

创新点包括：①将视觉ICL表述为视觉类比生成任务；②在冻结的Diffusion Transformer上采用角色感知多图令牌与Mixture‑of‑Experts LoRA实现高效多任务微调；③构建覆盖感知、修复与开放域编辑的大规模ICL数据集；

**🔧 技术方法**

使用技术包括Diffusion Transformer (DiT)、角色感知多图令牌(3D‑MSRoPE)、Mixture‑of‑Experts LoRA、视觉类比条件生成以及相关预训练模型（Qwen‑Image‑Edit、CLIP等）；

**📊 数据集**

使用数据集为自建ICL编辑数据集（覆盖边缘检测、深度估计、法向估计、分割、修复、开放域编辑等），并基于DiffusionDB、Qwen‑Image、ControlNet、COCO、Rain200L等公开数据集构建；

**📈 对比分析**

与IMProv、VisualPrompt、PromptDiff、Painter等V‑ICL基线及专业任务模型对比，VIRAL在多任务上均优于基线，且在多数任务上接近或超越专门模型；

**⚠️ 局限性**

局限性在于仍受预训练模型生成先验限制，难以处理极端复杂或超出训练分布的编辑，对多模态文本指令的兼容性有待提升。

---

## 400. Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements

**arXiv ID:** 2602.03209 | [PDF](https://arxiv.org/pdf/2602.03209v1)

**作者:** Marco Job `[一作]` (Norwegian University of Science and Technology), Michael Pantic `[通讯]` (ETH Zurich)

**通讯引用:** 740 | [OpenAlex ID](https://openalex.org/A5032693392)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种实时深度补全方法，利用单目图像和极稀疏深度测量，在未见过的野外机器人环境中预测稠密度计量深度。

**💡 创新点**

创新点在于：1) 将稀疏深度通道扩展进SOTA MDE网络DAV2，保留其泛化能力；2) 通过SfM得到真实3D网格并用Blender渲染生成多样化合成训练数据；3) 在训练中动态采样并噪声化稀疏深度；4) 实现实时推理（53 ms/帧）并保持低延迟。

**🔧 技术方法**

采用Vision Transformer（ViT‑S）+ DPT解码器，scale‑invariant loss 加 gradient loss，随机角点采样、噪声化，Blender渲染，Pix4D SfM 等技术。

**📊 数据集**

使用四个合成训练集（山区、冰川、道路、农村）以及10k帧Hypersim与Mid‑Air；评估数据集包括五个未见过的真实野外机器人集（IH、FF、GF、BT、UF）。

**📈 对比分析**

与九个SOTA基线（MDE、DC、Diffusion等）在五个真实环境上进行MAE/RMSE对比，该方法在平均排名上最高，除冰川实验外误差最低，并实现53 ms/帧的实时推理。

**⚠️ 局限性**

仅针对极稀疏深度测量设计，无法很好处理密集深度；在金属或高噪声环境下仍有挑战，部分DC方法在特定场景下表现逊色。

---

## 401. WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU

**arXiv ID:** 2602.03207 | [PDF](https://arxiv.org/pdf/2602.03207v1)

**作者:** Yudong Han `[一作]` (Peking University), Yun Ma `[通讯]` (Peking University)

**通讯引用:** 9208 | [OpenAlex ID](https://openalex.org/A5038594863)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一套基于WebGPU的全GPU渲染管线WebSplatter，用于在浏览器中实时渲染大规模3D高斯散点（3D Gaussian Splatting）场景。

**💡 创新点**

创新点包括：1）等待无锁层次化基数排序，解决WebGPU缺乏全局原子导致的同步瓶颈；2）基于不透明度的几何剔除与动态四边形尺寸调整，显著减少重绘和显存占用；3）采用混合计算-渲染流水线，充分利用GPU并行度。

**🔧 技术方法**

技术手段主要有：WebGPU计算着色器、WGSL语言、层次Blelloch前缀和、屏幕空间AABB剔除、SH色彩计算、基于不透明度的局部坐标插值与高斯函数求值。

**📊 数据集**

使用公开的3DGS基准数据集，包括 Bicycle、Garden、Truck、Bonsai、Train、Bicycle‑c、Van Gogh Room 等，覆盖 3.4 万至 6.1 百万个高斯散点。

**📈 对比分析**

与现有 WebGL（S1、S2）和 WebGPU（W1、W2）基线在多平台（RTX 3070、MacBook Air M4、M1、Redmi K70 Pro、iPhone 15 Pro Max 等）对比，平均帧时延下降 1.06–4.5 倍，显存占用下降 36–57%，并在所有设备上保持稳定。

**⚠️ 局限性**

局限性：依赖 WebGPU，旧版浏览器不支持；在高端 GPU 上仍受光栅化阶段制约；低端设备的预处理仍较慢；未针对动态场景更新或流式加载做专门优化。

---

## 402. ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs

**arXiv ID:** 2602.03226 | [PDF](https://arxiv.org/pdf/2602.03226v1)

**作者:** Xuancheng Li `[一作]` (Tsinghua University), Yiqun Liu `[通讯]` (Tsinghua University)

**通讯引用:** 9929 | [OpenAlex ID](https://openalex.org/A5100668121)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自适应任务感知压缩器 ATACompressor，动态压缩长文本仅保留任务相关信息，兼顾压缩率与信息完整性。

**💡 创新点**

创新点在于结合选择性编码器和自适应分配控制器，使压缩率随相关内容长度自调，实现任务感知的高效压缩。

**🔧 技术方法**

采用冻结的大型语言模型 + LoRA 作为选择性编码器，配合注意力+MLP 探测器估计相关长度，并通过软提示方式生成压缩 token。

**📊 数据集**

在 HotpotQA、MSMARCO、SQUAD 三个问答基准上进行评测，展示跨数据集的通用性。

**📈 对比分析**

与硬提示、软提示压缩方法（Selective‑Context、LongLLMLingua、AutoCompressor、ICAE、500Compressor、QGC）对比，ATACompressor 在 F1/EM 上提升 5–15% 以上，压缩比提升至 20–30 倍，吞吐量保持与 500Compressor 相当。

**⚠️ 局限性**

局限在于需预先按任务粒度划分块，对极短文本压缩效果不佳，且最小压缩 token 数需手动设定。

---

## 403. TAME: A Trustworthy Test-Time Evolution of Agent Memory with Systematic Benchmarking

**arXiv ID:** 2602.03224 | [PDF](https://arxiv.org/pdf/2602.03224v1)

**作者:** Yu Cheng `[一作]` (East China Normal University), Zhaoxia Yin `[通讯]` (East China Normal University)

**通讯引用:** 3218 | [OpenAlex ID](https://openalex.org/A5035489942)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双层记忆演化框架TAME，能够在测试时学习过程中同时提升任务性能与信任度。

**💡 创新点**

通过将执行器与评估器分离，建立闭环双记忆更新机制，解决了传统记忆演化导致的Agent Memory Misevolution（性能提升伴随信任度下降）问题。

**🔧 技术方法**

采用语义相似度检索、过滤与改写、可信度强化、评估约束（Constitutional AI）以及双轨记忆更新等技术，构建了Executor与Evaluator双层系统。

**📊 数据集**

在Trust‑Memevo基准上评估，涵盖数学（GSM8K、MATH、AIME）、科学（MMLU‑Pro、GPQA）与工具使用（TaskBench）三大领域，使用多维信任度评估集（Safety、Robustness、Truthfulness、Privacy、Fairness）。

**📈 对比分析**

与No‑Memory、DC、Memento、ReasoningBank及其安全增强变体对比，TAME在保持或提升任务准确率的同时，显著抑制了信任度衰退，尤其在科学与工具使用任务中获得最优或竞争力的结果。

**⚠️ 局限性**

局限性包括：对安全约束依赖于预设的Constitutional AI规则；在初始阶段信任度提升有限；扩展到更大规模或多模态任务时可能需要更高计算资源。

---

## 404. Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models

**arXiv ID:** 2602.03211 | [PDF](https://arxiv.org/pdf/2602.03211v1)

**作者:** Yeongmin Kim `[一作]` (Korea Advanced Institute of Science and Technology), Il-Chul Moon `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1525 | [OpenAlex ID](https://openalex.org/A5017589963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LiDAR，一种利用前瞻采样和奖励引导的扩散模型测试时缩放方法。

**💡 创新点**

创新点在于将期望未来奖励（EFR）从中间粒子解耦出来，给出仅依赖边际样本和前向噪声核的闭式公式，并实现了无需神经网络反向传播的导数无关引导。

**🔧 技术方法**

使用技术包括：扩散模型（SD v1.5/SDXL）、前瞻采样器（DPM‑5/LCM‑4等）、奖励函数（ImageReward/CLIP）、闭式奖励梯度公式和多步ODE求解器。

**📊 数据集**

实验数据集主要为公开的文本到图像基准，使用GenEval评测集以及CLIP和HPS指标。

**📈 对比分析**

与梯度引导（UG、DATE）、SMC和Best‑of‑N等方法比较，LiDAR在保持相同或更高GenEval分数的同时，推理速度提升约9.5×、内存占用几乎不变，且在多种后端和奖励组合下均表现优异。

**⚠️ 局限性**

局限性包括：对奖励函数质量高度敏感；λ参数需手动调节，过大易导致多样性下降；前瞻采样仍增加一定前置成本，且在极端高维或复杂奖励场景下效果待验证。

---

## 405. ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution

**arXiv ID:** 2602.03203 | [PDF](https://arxiv.org/pdf/2602.03203v1)

**作者:** Zican Dong `[一作]` (Renmin University of China), Wayne Xin Zhao `[通讯]` (Renmin University of China)

**通讯引用:** 17193 | [OpenAlex ID](https://openalex.org/A5037145565)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型长文本推理中的KV缓存抛弃问题，提出ForesightKV两阶段训练框架，学习动态评估KV对后续推理的重要性并进行智能淘汰。

**💡 创新点**

引入Golden Eviction生成黄金标签并用Pairwise Ranking学习，随后将淘汰建模为MDP并用GRPO强化学习，兼顾低熵token损失，提升长推理性能。

**🔧 技术方法**

训练轻量级MLP评分模型、Golden Eviction、Pairwise Ranking Loss、GRPO强化学习、MSE损失奖励、Top‑K+Multinomial采样等技术。

**📊 数据集**

在AIME2024、AIME2025数学推理基准以及GPQA、LiveCodeBench等数据集上进行评估，使用Qwen3-4B、Qwen3-1.7B等模型。

**📈 对比分析**

与SnapKV、H2O、R‑KV等传统方法对比，ForesightKV在相同缓存预算下获得更低损失，半预算即可逼近原模型，吞吐率提升至约10×。

**⚠️ 局限性**

仅在固定预算与特定推理长度下验证，对极端高熵推理或不同模型结构的适应性仍待进一步验证。

---

## 406. Hierarchical Proportion Models for Motion Generation via Integration of Motion Primitives

**arXiv ID:** 2602.03188 | [PDF](https://arxiv.org/pdf/2602.03188v1)

**作者:** Yu-Han Shu `[一作]` (University of Tsukuba), Sho Sakaino `[通讯]` (University of Tsukuba)

**通讯引用:** 1641 | [OpenAlex ID](https://openalex.org/A5065050034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个基于双向控制的层次模仿学习框架，通过将复杂运动分解为运动原语并按比例组合，实现了对未见任务的生成。

**💡 创新点**

创新点在于引入比例模型（学习式、采样式、回放式）以实现运动原语的可重用与任务级适配，降低了上层学习成本。

**🔧 技术方法**

采用双向控制、层次LSTM+MLP网络、Monte Carlo模型预测控制以及加权平均融合等技术。

**📊 数据集**

使用了从双向控制收集的pick‑and‑place演示数据，共50个运动原语以及两项验证任务（右→左和两物体搬运）。

**📈 对比分析**

与传统完整层次模型对比，采样式和回放式比例模型在右→左任务中100%成功，在两物体任务中分别达到70%和90%，显著优于基线。

**⚠️ 局限性**

局限性包括对原语空间外目标位置的精度不足、上层模型训练对原语数量敏感以及回放式模型缺乏对环境变化的自适应性。

---

## 407. Towards Context-Aware Edge-Cloud Continuum Orchestration for Multi-user XR Services

**arXiv ID:** 2602.03262 | [PDF](https://arxiv.org/pdf/2602.03262v1)

**作者:** Inhar Yeregui `[一作]`, Eduardo Jacob `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于上下文感知的 Edge‑Cloud Continuum 多用户 XR 服务编排模型，并给出了对应的数学优化框架。

**💡 创新点**

创新点在于将 XR 服务的四层（应用、虚拟化、管理与支持）参数化，构建统一的上下文模型并通过约束和目标函数实现自适应资源调度。

**🔧 技术方法**

采用 NFV/SDN 架构、5G/6G 网络特性、AI‑驱动的闭环调度和多目标优化（QoS、成本、迁移开销）技术。

**📊 数据集**

使用仿真测试床，包括三类节点（云、边缘、终端）和多种用户角色、感知类型与渲染质量的人工生成数据集。

**📈 对比分析**

通过目标函数评估不同节点放置方案，并比较在用户加入过程中编排策略的性能，实验显示在模拟环境下可保持 QoS 并降低成本；但未与现有实测基线或其他算法进行定量对比。

**⚠️ 局限性**

主要局限在于仿真网络条件静态、缺乏真实移动和多租户场景、未验证能耗与可持续性指标，未来需在真实复杂网络中进一步评估。

---

## 408. LPS-Bench: Benchmarking Safety Awareness of Computer-Use Agents in Long-Horizon Planning under Benign and Adversarial Scenarios

**arXiv ID:** 2602.03255 | [PDF](https://arxiv.org/pdf/2602.03255v1)

**作者:** Tianyu Chen `[一作]` (ShanghaiTech University), Wenjie Wang `[通讯]` (ShanghaiTech University)

**通讯引用:** 2168 | [OpenAlex ID](https://openalex.org/A5100368534)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LPS-Bench 基准，用于评估长时序 MCP 型计算机使用代理在规划阶段对安全风险的识别与规避能力。

**💡 创新点**

创新点包括：①面向长时序、规划层面的安全评估；②构建了包含 7 个任务域、9 类风险、65 个场景的规模化数据集；③采用多智能体自动化生成与 LLM‑as‑a‑judge 评估的端到端流程。

**🔧 技术方法**

使用多智能体数据生成管线、LLM 评判器、Mock 工具沙箱以及安全率（Safe Rate）指标进行评估。

**📊 数据集**

构建的 LPS‑Bench 数据集共 570 个测试案例，覆盖 65 个场景、9 种风险类型，来源于真实安全情景、已有基准与 LLM 探索。

**📈 对比分析**

与 13 种主流 LLM（包括 GPT‑5、Gemini、Claude 等）进行对比，结果显示即便是最强模型在某些风险类型（如模糊指令、环境伪造）安全率仅在 60% 以下，表明现有代理在长时序规划安全方面仍存在显著缺陷。

**⚠️ 局限性**

局限性在于：基准不涵盖所有真实域和工具；自动评估可能对边缘轨迹判定不完全精确；模型对低能力时安全提升有限。

---

## 409. BayeSQP: Bayesian Optimization through Sequential Quadratic Programming

**arXiv ID:** 2602.03232 | [PDF](https://arxiv.org/pdf/2602.03232v1)

**作者:** Paul Brunzema `[一作]` (RWTH Aachen University), Sebastian Trimpe `[通讯]` (RWTH Aachen University)

**通讯引用:** 2219 | [OpenAlex ID](https://openalex.org/A5023990842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种名为 LBO-QSQP 的黑盒优化算法，结合了序列二次规划（SQP）与贝叶斯优化（BO），利用二阶高斯过程（GP）模型在零阶信息下同时估计函数值、梯度和海森矩阵，构造不确定性感知的二次子问题并通过受约束的后验采样实现步长选择。

**💡 创新点**

创新点：
- 将二阶GP（可直接获得梯度与海森估计）引入经典SQP框架；
- 通过概率风险度量（VaR）和约束可达性概率，将不确定性嵌入子问题，使其可化为二阶锥规划；
- 采用受约束的后验采样（受限Thompson采样）实现高效的一维线搜索，避免全局线搜索开销；
- 通过滑动变量回退策略保证子问题始终可行。

**🔧 技术方法**

核心技术：
- 二阶高斯过程回归（同时预测函数、梯度、海森）；
- 基于高斯过程的概率约束与目标风险评估；
- 二阶锥规划（SOCP）求解子问题；
- 受限后验采样（Constrained Posterior Sampling）进行线搜索；
- Sobol序列采样用于局部海森近似。

**📊 数据集**

主要实验数据集：
- 随机 Fourier 特征生成的“within‑model”高维函数（16、32、64 维）；
- 标准 BO 基准（Ackley、Hartmann 等多峰问题，含约束与无约束）；
- 7 维 Speed Reducer 机械设计基准（11 个非线性约束）。

**📈 对比分析**

与四个基线（LogEI、GlobalEI、LBO‑EPI、BO‑LBO）以及可约束扩展（EHI、Constrained LogEI）进行对比。结果表明：
- 在 16 维以上的无约束问题中，算法在中位数性能上显著优于基线；
- 在约束问题中，LBO‑QSQP 既能获得更低的目标值，又显著缩短了运行时间（比 LogEI 低 1–2 个数量级）；
- 对于 Speed Reducer 基准，取得最优或次优解且求解时间最短。

**⚠️ 局限性**

主要局限：
- 作为局部方法，结果高度依赖初始化，可能陷入不同的局部最优；
- 在维度超过 96 或约束数目很大时，海森矩阵的存储与求解开销显著；
- 性能强烈依赖 GP 核函数与超参数设置，若核选择不当会导致梯度/海森估计不准确；
- 未考虑海森不确定性的扩展，可能影响极端不确定环境下的搜索方向。

---

## 410. EventFlash: Towards Efficient MLLMs for Event-Based Vision

**arXiv ID:** 2602.03230 | [PDF](https://arxiv.org/pdf/2602.03230v1)

**作者:** Shaoyu Liu `[一作]` (Xidian University), Xiangyang Ji `[通讯]` (Tsinghua University)

**通讯引用:** 10840 | [OpenAlex ID](https://openalex.org/A5024401174)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了高效事件视觉多模态大语言模型 EventFlash，利用时空令牌稀疏化技术加速推理并支持长时间事件流的理解与生成。

**💡 创新点**

创新点：
1) 自适应时间窗口聚合（ATWA）和稠密度引导注意（SDGA）实现事件流的时空令牌稀疏化；
2) 从短到长的课程学习策略，提升模型对不同长度事件序列的泛化；
3) 构建 500k 规模、多任务、多场景的事件–文本数据集 EventMind。

**🔧 技术方法**

技术手段：
- 事件编码器 CLIP‑ViT‑Large‑Patch14；
- 大语言模型后端 Qwen2.5；
- 自适应时间窗口聚合（ATWA）+ 语义感知聚合；
- 稠密度引导注意（SDGA）+ Token Selector；
- 课程学习与数据增强；
- 端到端训练与推理优化。

**📊 数据集**

使用数据集：
- EventMind（500k 指令样本，覆盖 0–20,000 ms 事件序列，7 个任务）；
- 真实事件来源：DSEC、N‑ImageNet、HARDVS、E2VID；
- 合成事件来源：Kinetics‑700、UCF‑101、Wevid‑10 M、PLM‑Data、MotionBench 通过 V2E 模拟器生成；
- 采用 GPT‑4o 与 Qwen‑VL‑Max 自动生成/校正文本指令。

**📈 对比分析**

对比方法与性能：
- 与四大视频‑LLM（Qwen2.5‑VL、VideoChat2‑Flash、LLaVA‑Next‑Video、InternVL2.5）及事件‑LLM EventGPT 进行基准对照；
- 在 GDC、FGQA、HAQA、MCQA 四个评测指标上，EventFlash 超越所有对手；
- 吞吐量提升 12.4×（28.5 tokens/s vs 2.3 tokens/s）；
- 事件桶容量提升至 1,000 bins（vs 5 bins 的 EventGPT），显著增强长时序理解；
- 在高速运动与低照度极端场景下保持较高准确率和描述质量。

**⚠️ 局限性**

局限性：
- 仍需依赖密集式 ViT 编码器，空间稀疏化在极稀疏场景下可能导致信息损失；
- 自适应聚合间隔（默认 10 ms）限制了对更细粒度事件的捕捉；
- 公开的模型权重与完整训练细节有限，复现性受限；
- 未在更大规模模型或 RGB+事件多模态组合场景中进行评估。

---

## 411. ConsisDrive: Identity-Preserving Driving World Models for Video Generation by Instance Mask

**arXiv ID:** 2602.03213 | [PDF](https://arxiv.org/pdf/2602.03213v1)

**作者:** Zhuoran Yang `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8048 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 ConsisDrive，一种通过实例级时间一致性机制生成高质量驾驶场景视频的世界模型；

**💡 创新点**

核心创新在于两大模块：Instance‑Masked Attention（利用实例身份与轨迹掩码实现实例感知注意力，抑制身份漂移）和 Instance‑Masked Loss（通过概率动态前景掩码聚焦监督，提升前景细节保真度）；

**🔧 技术方法**

技术路线基于 OpenSora V2.0 的 VAE 编码器–解码器、MMDiT 反向扩散 Transformer、ControlNet 控制模块，并结合 CLIP、T5 文本编码、Fourier 映射及三维框投影构造掩码；

**📊 数据集**

使用 nuScenes 数据集进行训练与评测；

**📈 对比分析**

与 BEVControl、DriveDiffusion、DriveDreamer2、Panacea、MagicDrive‑V2 等同类模型对比，ConsisDrive 在 FID（3.88）与 FVD（37.23）上取得最优成绩，mAP 达 31.5%（相当于真实数据的 91.3%），NDS 54.6，IDS 525，显著提升了下游感知与多目标跟踪任务的表现；

**⚠️ 局限性**

局限性包括：仍需高质量 3D 边框标注；在极端场景下可能出现轻微身份漂移；模型规模大、推理成本高；对未见场景的泛化能力有限；生成数据在直接用于安全关键系统前需进一步验证。

---

## 412. Exploring the Role of Tracing in AI-Supported Planning for Algorithmic Reasoning

**arXiv ID:** 2602.03197 | [PDF](https://arxiv.org/pdf/2602.03197v1)

**作者:** Yoshee Jain `[一作]` (University of Illinois Urbana Champaign), April Yi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5046673805)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了在 AI 支持的规划工具中使用追踪式规划和自然语言规划的学生在算法推理和编码表现上的差异。

**💡 创新点**

首次探讨追踪式规划对学习者思维方式的影响，并评估其是否能提升 LLM 生成反馈的准确性与实用性。

**🔧 技术方法**

利用大型语言模型（LLM）提供反馈，使用 eiplgrader 自动生成代码，CodeJudge 评估代码功能，采用语义相似度（余弦相似度）衡量计划与代码的一致性。

**📊 数据集**

实验使用 LeetCode 上的 Jump Game 贪心算法题目，20 名具有初级编程经验的学生，限定使用三种预定义变量。

**📈 对比分析**

通过随机分组的双盲实验，对计划步骤数、控制流引用、计划到代码的一致性、最终代码通过测试用例数量、LLM 反馈准确度等指标进行比较，结果显示两种规划方式在最终编码表现和 LLM 反馈质量上无显著差异；追踪式规划导致计划更简洁、推理更为目标驱动，且代码部分正确率略高。

**⚠️ 局限性**

样本量小（N=20）且仅为单次实验，缺乏纵向跟踪；自评数据可能与客观表现不完全一致；未深入探讨不同学生先前知识水平对结果的调节作用。

---

## 413. Reinforcement Learning with Promising Tokens for Large Language Models

**arXiv ID:** 2602.03195 | [PDF](https://arxiv.org/pdf/2602.03195v1)

**作者:** Jing-Cheng Pang `[一作]` (Huawei Technologies Co), Xubin Li `[通讯]` (Huawei Technologies Co)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种将强化学习优化范围限制在大型语言模型词表中高概率子集的框架，称为 RL with Promising Tokens (RLPT)。

**💡 创新点**

创新点在于：①利用预训练模型的语义先验动态生成“promising token”子集；②在 RL 采样与梯度更新中保持一致的遮罩，实现决策与生成的解耦；③理论证明该限制可显著降低策略梯度方差，提高训练稳定性与样本效率。

**🔧 技术方法**

技术手段包括：top‑k 词表遮罩、softmax 加上负无穷掩码、与常用 RL 算法（GRPO、DAPO）无缝集成、梯度方差分析、以及多任务实验评估。

**📊 数据集**

使用的数据集包括数学推理（GSM8K、Math‑17k、AIME‑24/25）、代码生成（HumanEval、OpenR1‑Code）、通用指令跟随（AlpacaEval）以及通信领域任务（Datacom、Wireless）。

**📈 对比分析**

与传统 RL 基线（GRPO、DAPO）进行对比，RLPT 在数学任务上提升 3–4% 的准确率，在代码和通信任务上也保持了竞争优势；样本效率更高，训练曲线上升更快；在 4B 与 8B 规模模型上均表现优异。

**⚠️ 局限性**

局限性：依赖固定 top‑k 阈值，可能漏掉少量关键 token；在极端符号或高复杂度推理场景下覆盖率不足；对预训练模型的语义先验过度依赖，若先验不足可能导致性能下降。

---

## 414. Privasis: Synthesizing the Largest "Public" Private Dataset from Scratch

**arXiv ID:** 2602.03183 | [PDF](https://arxiv.org/pdf/2602.03183v1)

**作者:** Hyunwoo Kim `[一作]` (NVIDIA), Yejin Choi `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Privacy Oasis 数据集，构建了百万级别、涵盖多种文档类型且包含丰富隐私属性的全合成数据集；并基于此数据集创建了文本去识别/去敏平行语料；训练了参数≤4B 的轻量级去敏模型，性能优于 GPT‑5 等前沿 LLM；

**💡 创新点**

创新点在于：①从零开始合成百万级数据，使用辅助控制变量和多样性保持的迭代筛选实现真实且多样的隐私信息；②构建了可按指令执行的多层抽象去敏任务的平行语料；③证明轻量级模型在去敏任务上可超过巨型 LLM，显著降低部署门槛；

**🔧 技术方法**

技术包括：LLM 生成 + 控制变量引导 + 多样性评价（Vendi）+ 迭代修正；拆分+指令化的去敏管道；基于指令和目标选择的去敏训练；多模型融合（GPT‑OSS‑120B、Qwen3‑80B 等）训练轻量模型；

**📊 数据集**

使用 Privacy Oasis 数据集（约 1.415M 条记录、55M 属性），以及其中的去敏平行语料（约 37K 训练样本、100K 测试样本）；对比现有公开数据集（MIMIC‑III、Enron Email、Legal、Finance 等）进行多域多指令评测；

**📈 对比分析**

与 GPT‑5、Qwen‑3‑235B、LLaMA‑4‑Maverick 等前沿 LLM 在“完整成功率”指标上对比；在普通测试集上 4B 模型达到 72.5% 最高，超过 GPT‑5 70.3%；在难度更高的测试集上 4B 模型仅略低于 GPT‑5（12.8% vs 13.1%），而前沿 LLM 仅 10–13%；

**⚠️ 局限性**

局限性包括：①在硬测试集上完整成功率仍低（≈12%）；②仍存在直接/推断/邻近泄漏，尤其是姓名、日期等属性易漏；③轻量模型在某些业务域（金融、医疗）性能相对较弱；④合成数据虽无实际隐私风险，但可能不完全覆盖真实世界的细节与多样性。

---

## 415. Synthesizing File-Level Data for Unit Test Generation with Chain-of-Thoughts via Self-Debugging

**arXiv ID:** 2602.03181 | [PDF](https://arxiv.org/pdf/2602.03181v1)

**作者:** Ziyue Hua `[一作]` (Peking University), Tao Xie `[通讯]` (Peking University)

**通讯引用:** 17536 | [OpenAlex ID](https://openalex.org/A5048118068)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自我调试和链式推理（CoT）的数据合成方法，用于生成高质量的文件级单元测试与对应CoT；

**💡 创新点**

创新点在于将自我调试与CoT压缩相结合，形成一个可迭代的修复与压缩循环，既提升测试质量，又保持CoT的可读性与连贯性；

**🔧 技术方法**

核心技术包括：指导测试修复（错误/失败/覆盖/变异四类修复），自我调试循环，CoT压缩重写，使用大型语言模型（DeepSeek‑R1、Qwen‑Coder、Qwen3）进行生成与修复；

**📊 数据集**

使用了公开的pymethod2test数据集（≈68,647个文件对）作为原始测试对，经过自我调试后合成74,518条（方法文件、测试文件、CoT）训练数据，并在TestGenEval基准上评估；

**📈 对比分析**

与现有方法（如o4-mini、o3-mini、DeepSeek‑R1等）对比，Fine‑tuned Qwen2.5‑Coder‑32B在TestGenEval上取得pass率36.17%、分支覆盖43.90%、变异得分88.66%，显著优于商业模型的27.23%/16.34%/76.82%；

**⚠️ 局限性**

局限性包括：仍依赖于已有开源仓库的质量；自我调试循环成本高且在后期收益递减；CoT虽然压缩但仍可能缺乏人类可读的细粒度解释；模型在极少数复杂语境下的泛化能力有待验证。

---

## 416. Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection

**arXiv ID:** 2602.03216 | [PDF](https://arxiv.org/pdf/2602.03216v1)

**作者:** Dongwon Jo `[一作]` (Seoul National University), Jae-Joon Kim `[通讯]` (Seoul National University)

**通讯引用:** 4036 | [OpenAlex ID](https://openalex.org/A5003219699)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Token Sparse Attention，一种动态可逆的 token 级稀疏注意力机制，用于在长上下文推理中减少注意力计算量。

**💡 创新点**

创新点在于：1）在每层每头动态选择少量关键 token 并压缩 Q/K/V，随后将结果映射回完整序列；2）不永久剔除 token，允许后续层重新评估 token 重要性；3）可与任何现有稠密或稀疏注意力核无缝组合。

**🔧 技术方法**

核心技术包括：token 级稀疏注意力、动态 token 覆盖策略（Dynamic Token Coverage）利用轻量级 attention 代理计算 token 重要性并按阈值决定保留 token 数；层间表示漂移（Inter‑Layer Representation Drift）筛选可稀疏的层；在 FlashAttention、Minference、FlexPrefill 等通用注意力实现中插入压缩/解压步骤。

**📊 数据集**

主要在 LLaMA‑3.1‑8B‑Instruct 与 Mistral‑Nemo‑12B‑Instruct 两个模型上，使用 RULER 与 InfiniteBench 长上下文基准进行评估；实验也在 LongBench、Needle‑in‑a‑Haystack 等数据集上验证（见附录）。

**📈 对比分析**

与 FlashAttention、Minference、FlexPrefill、FastKV、GemFilter 等基线比较，Token Sparse Attention 在 128K 上可实现 ×1.36~×2.76 的注意力加速，同时保持 <1% 的准确率下降；在大多数场景下，它的加速效果比单一稀疏策略更好，且动态稀疏策略优于固定稀疏比例。

**⚠️ 局限性**

局限性包括：1）对非常短的序列加速不显著；2）需额外的 token 重要性评分和索引开销（约 10% 的额外延迟）；3）目前仅在推理预填阶段实验，未针对解码阶段展开；4）稀疏比例需手动调节，自动化选择仍待研究。

---

## 417. HypCBC: Domain-Invariant Hyperbolic Cross-Branch Consistency for Generalizable Medical Image Analysis

**arXiv ID:** 2602.03264 | [PDF](https://arxiv.org/pdf/2602.03264v1)

**作者:** Francesco Di Salvo `[一作]` (University of Bamberg), Christian Ledig `[通讯]` (University of Bamberg)

**通讯引用:** 24178 | [OpenAlex ID](https://openalex.org/A5016912926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出在医疗影像领域使用双分支超平面空间表示学习，并引入无监督域不变交叉分支一致性约束，以提升模型在分布外的鲁棒性。

**💡 创新点**

创新之处在于将欧氏单分支嵌入改为轻量化的双分支超平面投影，其中低维分支充当域不变信息瓶颈，并通过 KL 交叉一致性实现无监督域不变学习。

**🔧 技术方法**

采用冻结的 ViT 基础模型提取特征，分别线性投影到 128D 与 2D 欧氏向量，再通过指数映射至 Poincaré 球面得到超平面嵌入，使用多类别逻辑回归生成 logits，并加入 KL 一致性损失；训练采用 AdamW 与余弦退火。

**📊 数据集**

实验数据集包括 11 个医学影像分类集（血液、乳腺、胸部、皮肤、OCT、腹部 CT、结肠病理、肺炎、视网膜、肾脏组织）以及三大域泛化基准（Fitzpatrick17k、Camelyon17-WILDS、跨数据集视网膜）。

**📈 对比分析**

与单分支欧氏 ERM、超平面 ERM、欧氏增强方法（RandAugment、AugMix、Med-C）以及 IRM、GroupDRO 等传统域泛化方法对比，HypCBC 在三大基准上平均提升约 +1.42% 的 AUC，进一步提升 +0.66%，并在统计上显著优于欧氏基线。

**⚠️ 局限性**

局限在于曲率固定、冻结基础网络、仅针对单标签多类分类任务，未探索可学习曲率、全超平面训练或多标签场景，且对小样本或标签噪声数据的适用性尚待验证。

---

## 418. CSR-Bench: A Benchmark for Evaluating the Cross-modal Safety and Reliability of MLLMs

**arXiv ID:** 2602.03263 | [PDF](https://arxiv.org/pdf/2602.03263v1)

**作者:** Yuxuan Liu `[一作]` (Zhejiang University), Kun Yang `[通讯]` (Zhejiang University)

**通讯引用:** 14266 | [OpenAlex ID](https://openalex.org/A5100435639)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CSR-Bench基准，用于评估多模态大语言模型在安全、过度拒绝、偏见和幻觉四大维度下的跨模态可靠性。

**💡 创新点**

①设计四大交叉模态安全测试模式，覆盖61种细粒度风险类型；②提供文本对照基线以诊断模态依赖；③系统量化模型跨模态对齐缺口和安全-帮助性权衡。

**🔧 技术方法**

基于多模态评估框架、自动化标签器（如GPT‑5‑Nano）进行对齐准确率评估；使用文本重写和模型辅助生成构造数据；采用统一评估协议与模态对照基线比较。

**📊 数据集**

CSR‑Bench（7,405个图文对），包含Safety（27子类）、Over‑rejection（16子类）、Bias（12子类）和Hallucination（4子类）四个子集；基准中还使用Gemini、Qwen‑VL、InternVL等公开/专有模型作为评测对象。

**📈 对比分析**

对16个最先进的多模态模型进行对齐准确率对比；结果显示没有模型在所有维度都表现稳健，安全性和过度拒绝之间存在明显权衡；多模态设置普遍低于文本对照基线，表明跨模态安全对齐仍缺乏。

**⚠️ 局限性**

基准为静态数据集，无法完全自动化生成；构造实例仍需人工或模型审核，存在循环依赖风险；评估侧重于对齐准确率，未涵盖对话连贯性、效率等更细粒度指标。

---

## 419. GraDE: A Graph Diffusion Estimator for Frequent Subgraph Discovery in Neural Architectures

**arXiv ID:** 2602.03257 | [PDF](https://arxiv.org/pdf/2602.03257v1)

**作者:** Yikang Yang `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Jianfeng Zhan `[通讯]` (BenchCouncil International Open Benchmark Council)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了GraDE框架，利用图扩散模型估计子图频率并通过束搜索发现神经网络架构中的高频子图模式。

**💡 创新点**

创新点在于首次将图扩散模型用于子图频率估计，将结构典型性转化为频率代理，并将其嵌入可扩展的搜索流程。

**🔧 技术方法**

核心技术包括图扩散估计器（DisCo‑E、DiGress‑E、DeFoG‑E等）、采样子图构建训练集、蒙特卡洛估计、束搜索和图变压器实现后验推断。

**📊 数据集**

使用NAS‑Bench‑101/201/301、Younger等真实神经架构数据集进行实验。

**📈 对比分析**

与传统枚举、采样方法（ARS、NRS、Rand‑ESU、Rand‑FaSE、GraphVAE）比较，GraDE在Spearman ρ上提升最高114%，在大规模子图搜索中中位频率提升至30×以上，且在稀疏采样条件下依然保持优越性能。

**⚠️ 局限性**

局限性包括仍需采样子图作为训练数据，模型训练与蒙特卡洛推理开销；对极大子图或非结构化图的适用性尚待验证；当前仅针对有向带类别属性的神经架构图。

---

## 420. Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning

**arXiv ID:** 2602.03249 | [PDF](https://arxiv.org/pdf/2602.03249v1)

**作者:** Zhicheng Yang `[一作]` (Hong Kong University of Science and Technology GZ), Jing Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13269 | [OpenAlex ID](https://openalex.org/A5083397767)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Accordion‑Thinking 框架，使大型语言模型能够在长篇推理过程中动态生成压缩摘要，从而在保留推理质量的前提下显著降低 KV 缓存和注意力开销，提升推理吞吐量。

**💡 创新点**

创新点在于：① 将推理步骤与摘要生成耦合成可学习的“Accordion”格式；② 通过强化学习引入压缩惩罚，使模型自我压缩并在压缩模式下保持与全上下文模式相当的推理准确性；③ 设计动态上下文裁剪机制，保证摘要信息的完整性。

**🔧 技术方法**

技术方法包括：基于 Transformer 的 LLM 微调；使用 GRPO（Group Relative Policy Optimization）强化学习框架；设计两种上下文模式（Unfold 与 Fold）和混合训练策略；加入特殊控制 token 以分隔步骤与摘要；采用数学推理验证器 Math‑Verify 作为奖励函数。

**📊 数据集**

数据集：使用 OpenR1‑45K（OpenR1‑Math‑220k 的子集）进行冷启动 SFT；通过从 16k 长度的 CoT 示例中采样 10,000 条，利用 DeepSeek‑V3.2 重写为 Accordion 格式并经过规则过滤，得到 3,900 条训练样本。评估使用五个数学推理基准：MATH‑500、OlympiadBench、MinervaMath、AIME24、AMC23。

**📈 对比分析**

与基线（Zero‑RL、Unfold‑RL、Fold‑RL、Mix‑RL）相比，Fold‑RL 与 Mix‑RL 在 48GB GPU 下实现近 3× 的吞吐量提升（如 Qwen3‑4B‑Base 从 1483 token/s 提升至 5888 token/s），同时在 Pass@1（Avg@32）指标上保持与 Unfold‑RL 同等或略优的准确率（例如 Qwen2.5‑Math‑7B 在 AIME24 上 32.2% vs 31.3%）。

**⚠️ 局限性**

局限性包括：① 仍需手工设计 Accordion 格式和特殊 token；② 训练过程对 RL 奖励函数和阈值敏感；③ 在极长推理任务中，摘要的压缩仍可能丢失细节导致错误；④ 目前仅在数学推理基准上验证，其他领域的迁移性待进一步探索。

---

## 421. Omnidirectional Solid-State mmWave Radar Perception for UAV Power Line Collision Avoidance

**arXiv ID:** 2602.03229 | [PDF](https://arxiv.org/pdf/2602.03229v1)

**作者:** Nicolaj Haarhøj Malle `[一作]` (University of Southern Denmark), Emad Ebeid `[通讯]` (University of Southern Denmark)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5062316918)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文设计并实现了一套轻量化、多雷达模块组成的全向毫米波雷达感知系统，用于在小型无人机上实现对电力线的实时探测与避免；

**💡 创新点**

创新点在于（1）利用多枚固态毫米波雷达模块合成全向（360°）覆盖，克服传统单雷达盲区；（2）深入研究毫米波雷达在电力线环境中的相互作用，发现雷达探测点近似于电力线最近点，从而提出了基于切线、急刹车与安全球的三阶段避免算法；

**🔧 技术方法**

主要技术包括：毫米波雷达硬件（TI IWR6843ISK 与 IWR6843AOPEVM）、ROS2 节点串联雷达数据、欧氏空间投影与切线计算、基于速度向量的避让策略、以及实时姿态/速度融合；

**📊 数据集**

实验数据集主要来源于真实电力线环境（约35 m 长、三相20 mm导体与一根10 mm导体），并在实验室使用金属角反射器进行射程与视场验证；没有使用公开的大规模雷达数据集；

**📈 对比分析**

与现有单雷达或需摄像头辅助的避障方法相比，本文系统在10 m以上可检测电力线，并能在>10 m/s 的飞行速度下完成安全避让；对1.2 mm 钢丝的探测与避让显示出更高的灵敏度；

**⚠️ 局限性**

主要局限包括：部分雷达模块视场偏差导致盲区，距离误差约6 cm；系统仍依赖于毫米波雷达的金属反射特性，可能对非金属障碍物不敏感；且未针对多无人机协同或复杂障碍环境进行评估。

---

## 422. Spiral RoPE: Rotate Your Rotary Positional Embeddings in the 2D Plane

**arXiv ID:** 2602.03227 | [PDF](https://arxiv.org/pdf/2602.03227v1)

**作者:** Haoyu Liu `[一作]` (University of California), Feng Wang `[通讯]` (John Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Spiral RoPE，一种在视觉 Transformer 中使用的多方向旋转位置编码，旨在解决标准轴向 RoPE 在二维空间方向上的限制。

**💡 创新点**

创新点在于将嵌入维度分成多组并沿均匀分布的 K 方向进行旋转，既保持了与 Axial 2D RoPE 相同的频率预算，又显著扩展了方向覆盖范围。

**🔧 技术方法**

使用技术包括旋转位置编码（RoPE）、分组频率插值、分组间隔频率分配，以及通过二维 Fourier 变换对编码能力进行可视化验证。

**📊 数据集**

实验数据集包括 ImageNet‑1k（分类）、ADE20k（语义分割）和 ImageNet（类条件图像生成）。

**📈 对比分析**

与绝对位置编码、Axial 2D RoPE 及 RoPE‑Mixed 进行对比，Spiral RoPE 在分类上提升 0.7–1.0% 的 Top‑1，分割任务提升 2.2% mIoU，生成任务 FID 降低 3.9–5.8 分，并在更高分辨率推断时表现出更好的泛化能力。

**⚠️ 局限性**

局限性包括对方向数 K 的选择需要平衡；过多方向会降低每组的维度容量，可能导致性能下降；此外，对更复杂多尺度场景下的极限尚未系统评估。

---

## 423. From Single Scan to Sequential Consistency: A New Paradigm for LIDAR Relocalization

**arXiv ID:** 2602.03198 | [PDF](https://arxiv.org/pdf/2602.03198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 424. PokeFusion Attention: Enhancing Reference-Free Style-Conditioned Generation

**arXiv ID:** 2602.03220 | [PDF](https://arxiv.org/pdf/2602.03220v1)

**作者:** Jingbang Tang `[一作]` `[通讯]` (Universiti Kebangsaan Malaysia), Jingbang Tang (Universiti Kebangsaan Malaysia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种轻量级的解码器级跨注意力融合机制PokeFusion Attention，实现了无需参考图像的风格控制文本到图像的生成；

**💡 创新点**

创新点在于将文本语义与学习得到的风格嵌入在解码器跨注意力层进行双分支融合，仅训练少量参数即可实现风格控制；

**🔧 技术方法**

采用了跨注意力融合、线性风格投影、CLIP 预训练模型和分类器无关引导等技术；

**📊 数据集**

使用了 Pokémon‑style 风格的 pokemon‑blip‑captions 数据集，共 833 张图像与文本描述；

**📈 对比分析**

与 ControlNet、T2I‑Adapter、Uni‑ControlNet、IP‑Adapter 等基线对比，PokeFusion Attention 在 CLIP‑T 与 CLIP‑I 上均取得最高分，参数量仅 22M，且无需参考图像；

**⚠️ 局限性**

局限性包括对单一风格的适用性较强，跨风格混合与动态风格控制尚未实现，需要进一步扩展多风格融合与参考图像自适应。

---

## 425. Beyond Quantity: Trajectory Diversity Scaling for Code Agents

**arXiv ID:** 2602.03219 | [PDF](https://arxiv.org/pdf/2602.03219v1)

**作者:** Guhong Chen `[一作]` (Southern University of Science and Technology), Yongbin Li `[通讯]` (Tongyi Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了TDScaling框架，通过聚类采样、Blueprint驱动多代理合成以及自适应熵/复杂度进化，专注于轨迹多样性以提升代码代理在Model Context Protocol环境下的工具使用与编程能力；

**💡 创新点**

创新点包括：①业务聚类采样提升工具覆盖度；②Blueprint驱动多代理生成保证逻辑一致性；③基于域熵、推理模式熵和累计动作复杂度的自适应进化机制，防止模式坍塌；④沙盒代码工具正则化，缓解工具调优导致的编码能力灾难性遗忘；整体实现多样性优先的合成与训练。

**🔧 技术方法**

采用多代理角色扮演（User/Assistant/Observation/Quality）、Blueprint生成、熵/复杂度度量、沙盒Python代码工具、基于最大覆盖的业务聚类抽样、可验证执行与自适应进化等技术。

**📊 数据集**

使用30,000+ MCP工具定义，抽样出约6,944个业务聚类，生成合成轨迹数据集；并在BFCL、τ²‑Bench、RebenchT、CodeCI、BIRD等公开基准上进行评估。

**📈 对比分析**

在500/5,000样本预算下，与Qwen3‑30B、Qwen3‑480B、APIGen‑MT、TOUCAN、Simia等对比。TDScaling在BFCL多轮任务仅用500样本即可达到36.66%（超越480B baseline），5,000样本提升至40.44%；在代码代理基准上平均提升4%；整体在所有基准上获得最高平均分。

**⚠️ 局限性**

主要限制：①合成过程计算和成本高，需高能力教师模型；②目前仅支持文本工具调用与Python执行，未覆盖多模态GUI或视觉任务。

---

## 426. Hand3R: Online 4D Hand-Scene Reconstruction in the Wild

**arXiv ID:** 2602.03200 | [PDF](https://arxiv.org/pdf/2602.03200v1)

**作者:** Wendi Hu `[一作]` (Zhejiang University), Gaoang Wang `[通讯]` (Zhejiang University)

**通讯引用:** 37832 | [OpenAlex ID](https://openalex.org/A5028525523)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 Hand3R，一个在线端到端的 4D 手-场景重建框架，能够从单目视频同时恢复手部网格与稠密场景几何，并给出全局绝对位姿；

**💡 创新点**

创新点在于场景感知视觉提示机制，将预训练的手部专家高保真特征与场景基础模型的空间记忆融合，实现在单前向推理中同时获得毫米级手部精度与全局尺度一致性；

**🔧 技术方法**

采用双流编码（手部专家 ViT + 场景编码器）、场景感知提示融合、基于时空记忆的解码器以及分离的手部姿态与全局平移头；

**📊 数据集**

使用 DexYCB（手部姿态）、HOI4D（全局轨迹与场景）以及公开的多视图/单目训练数据；

**📈 对比分析**

与 HaMeR‑SLAM、WiLoR‑SLAM 等多阶段离线基线相比，Hand3R 在全局 C‑MPJPE、W‑MPJPE、WA‑MPJPE 上均取得更低误差，同时在 DexYCB 上的本地手网格恢复亦保持竞争力；

**⚠️ 局限性**

局限性包括：在线时序估计仍受累计漂移影响；多手/极端遮挡下的全局一致性尚可进一步提升；缺乏多视角或多摄像头支持，导致深度与姿态不确定性仍存在。

---

## 427. LSGQuant: Layer-Sensitivity Guided Quantization for One-Step Diffusion Real-World Video Super-Resolution

**arXiv ID:** 2602.03182 | [PDF](https://arxiv.org/pdf/2602.03182v1)

**作者:** Tianxing Wu `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22315 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 LSGQuant，一种针对单步扩散模型的视频超分辨率的低位量化框架。

**💡 创新点**

创新点包括：动态范围自适应量化器（DRAQ）、方差导向层训练策略（VOLTS）以及量化感知交替优化（QAO）三大模块。

**🔧 技术方法**

技术手段包括：对齐缩放、Hadamard 旋转、低秩分支（SVD）、层敏感度统计、交替优化训练。

**📊 数据集**

使用的数据集有 HQ‑VSR（训练/校准）、合成数据 UDM10、REDS30 与真实数据 MVSR4x。

**📈 对比分析**

与 MinMax、SmoothQuant、QuaRot、ViDiT‑Q、SVDQuant 等量化方法对比，在 4‑bit 量化下 PSNR/SSIM/LPIPS 等指标均优于对手，参数与运算量均压缩超过 70%。

**⚠️ 局限性**

局限性：仍受低位量化误差影响，动态范围估计依赖校准样本，极端环境下可能失效；对单步扩散模型的进一步降维仍有挑战。

---

## 428. InstaDrive: Instance-Aware Driving World Models for Realistic and Consistent Video Generation

**arXiv ID:** 2602.03242 | [PDF](https://arxiv.org/pdf/2602.03242v1)

**作者:** Zhuoran Yang `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8048 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种驾驶世界模型，利用实例流引导器和空间几何对齐器提升驾驶视频在实例层面的时间一致性和空间几何精度，并在 nuScenes 数据集上生成高质量合成视频，用于下游感知、跟踪和规划任务。

**💡 创新点**

创新点在于：① 引入 Instance Flow Guider（IFG）通过实例流提取并在帧间传播实例特征，实现实例级时间一致性；② 引入 Spatial Geometric Aligner（SGA）通过 3D 边界框投影和深度顺序编码实现精确实例定位和遮挡层级建模。

**🔧 技术方法**

技术包括基于 OpenSora 的 VAE 编码、T5 文本编码、ST‑DiT 反向扩散 Transformer、ControlNet 注入、视图膨胀注意力、Fourier 编码深度信息、CARLA 生成模拟场景等。

**📊 数据集**

使用 nuScenes 数据集作为训练和评估基础，并通过 CARLA 自动驾驶生成长尾场景的控制条件。

**📈 对比分析**

与 BEVControl、DriveDiffusion、Panacea、MagicDrive‑V2 等 SOTA 驾驶世界模型对比，FID 下降至 3.96、FVD 下降至 38.06、在多目标跟踪上 IDS 降至 532、在感知上 NDS 达到 51.9（比仅用真实数据提升 3.6 分），表现优于现有方法。

**⚠️ 局限性**

局限性包括：对多视角生成仍需更多视角一致性验证；IFG 与 SGA 需要精确的实例 ID 与 3D 边界框数据，若缺失可能导致效果下降；生成长尾场景时仍受 CARLA 物理和传感器建模精度限制。

---

## 429. Topology Matters: A Cautionary Case Study of Graph SSL on Neuro-Inspired Benchmarks

**arXiv ID:** 2602.03217 | [PDF](https://arxiv.org/pdf/2602.03217v1)

**作者:** May Kristine Jonson Carlon `[一作]` (RIKEN Center for Brain Science), Yasuo Kuniyoshi `[通讯]` (RIKEN Center for Brain Science)

**通讯引用:** 10877 | [OpenAlex ID](https://openalex.org/A5010543059)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种多尺度自监督学习框架（Hierarchical SSL），旨在同时学习节点、边和图级别的嵌入，并通过一个可控的合成连接组图基准进行系统评估。

**💡 创新点**

创新点在于：1) 设计了显式的边缘投影头与多尺度预测器，实现节点、边、图三层嵌入的联合学习；2) 建立了四阶段严谨评估协议，并用合成基准揭示了传统 invariance‑based SSL 与连接组结构的根本冲突；3) 通过 ablation 证明了对拓扑不变性的目标会导致模型忽略社区结构。

**🔧 技术方法**

采用 GraphSAGE 消息传递网络为 backbone；多层 MLP 投影头与 SimSiam 预测器；使用 MMD 对边缘分布进行对齐；VICReg 风格的方差/协方差正则；节点特征遮蔽、DropEdge 等数据增强。

**📊 数据集**

使用自己设计的可控多模态合成连接组图（约 500 张），每张图包含 700–900 个节点、6–10 个社区、两通道（SC、FC）边缘，并生成节点特征、标签等。

**📈 对比分析**

与经典特征（LR、Cosine）、图拓扑度量（Jaccard、Label Propagation、WL‑Hash）以及监督 GraphSAGE 进行对比。结果显示 SSL 在链接预测、节点分类和子图回归等任务上均低于传统方法，甚至出现负 R²，证明了 invariance 目标与拓扑任务不匹配。

**⚠️ 局限性**

主要限制：1) 仅在合成数据上验证，缺少真实连接组实验；2) 未与最近的 SSL 基线（GraphCL、GraphMAE 等）直接对比；3) 评估聚焦于特定任务，未探索更广泛的下游应用。

---

## 430. Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation

**arXiv ID:** 2602.03208 | [PDF](https://arxiv.org/pdf/2602.03208v1)

**作者:** Jinyan Ye `[一作]` (East China Normal University), Yingda Chen `[通讯]` (Alibaba Group)

**通讯引用:** 940 | [OpenAlex ID](https://openalex.org/A5042773283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Spectral Evolution Search (SES)，一种通过离散小波分解将初始噪声限制在低频子空间，并使用交叉熵方法（CEM）在该低维子空间进行梯度无关的演化搜索，从而实现推理时刻的高效缩放。

**💡 创新点**

创新点：①发现并利用生成模型对低频噪声的显著谱偏差；②将搜索空间压缩到低频子空间，显著降低维度；③在该子空间中应用无梯度 CEM 进行分布演化；④从扰动传播动力学推导谱扩展预测理论，为低频主导提供理论支撑。

**🔧 技术方法**

技术手段：离散小波变换（DWT）进行频谱分解与重构；交叉熵方法（CEM）进行分布演化搜索；基于生成流扰动传播的理论分析；使用代理评估（4 步 Latent Consistency Model / 10 步 ODE）加速奖励评估。

**📊 数据集**

数据集与模型：DrawBench、Pick-a-Pic、SDXL、FLUX、Qwen-Image 等；奖励指标包括 CLIP、PickScore、HPS、ImageReward、Aesthetic Score。

**📈 对比分析**

对比方法：Best-of-N、Zero-Order Search、Search over Paths、SMC、SVDD、Demon 等。实验中 SES 在固定预算（NRE=200 或 1000）下，所有奖励指标均显著优于基线，并在质量-多样性 Pareto 前沿上表现更好。

**⚠️ 局限性**

局限性：受奖励模型质量与计算成本限制；理论假设局部线性和频谱分离，忽略高阶相互作用；NRE 与实际耗时之间可能不呈线性关系。

---

## 431. HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control

**arXiv ID:** 2602.03205 | [PDF](https://arxiv.org/pdf/2602.03205v1)

**作者:** Jinrui Han `[一作]` (Shanghai Jiao Tong University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 61661 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了一套物理感知的学习框架，实现人形机器人在现实世界进行滑板推行、转向与相位转换的全身控制，完成完整的滑板骑行。

**💡 创新点**

创新点包括：①基于滑板倾斜‑转向耦合的等式约束建模；②将任务拆分为推行、转向和相位转换三个阶段，并为每阶段设计专属学习策略；③利用对抗运动先验（AMP）学习人类般的推行动作；④通过物理引导的倾斜角参考实现精准的倾斜转向；⑤采用贝塞尔曲线轨迹规划的相位转换引导；⑥结合仿真参数识别与域随机化实现从模拟到现实的无缝迁移。

**🔧 技术方法**

技术手段包括：深度强化学习（PPO）与自监督对抗运动先验（AMP）；物理引导的偏航控制与倾斜参考；贝塞尔曲线和平滑插值的相位转换轨迹；PD 控制实现低层执行；域随机化、系统识别与仿真参数匹配；以及基于姿态与接触的奖励设计。

**📊 数据集**

使用人类滑板推行动作数据集作为 AMP 先验；仿真环境自行生成；无公开大规模数据集，主要依赖自制模拟与人类采集的滑板运动数据。

**📈 对比分析**

与三组基线（追踪式/步态式推行、无倾斜引导转向、仅平移转移）进行对比。仿真结果显示 100% 成功率、低速度与偏航误差、流畅度与接触误差显著优于基线；真实世界测试亦能实现连续推行、转向、相位转换，表现出稳健与抗扰动能力。

**⚠️ 局限性**

局限性包括：仅在平坦地面实验，缺乏对复杂地形和高难度技巧的适配；摄像头视野受限，缺乏视觉感知驱动的控制；对滑板物理识别参数敏感，若参数不匹配会导致失稳或无法上板。

---

## 432. Sparsity is Combinatorial Depth: Quantifying MoE Expressivity via Tropical Geometry

**arXiv ID:** 2602.03204 | [PDF](https://arxiv.org/pdf/2602.03204v1)

**作者:** Ye Su `[一作]` (Shenzhen Institutes of Advanced Technology), Yong Liu `[通讯]` (Renmin University of China)

**通讯引用:** 20382 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

对Mixture-of-Experts（MoE）模型的表达能力进行理论分析，利用热几何揭示 Top‑k 路由与 k‑th 交替对称热多项式的同构，并证明其划分空间为超单纯形的法线向量星，进而给出 MoE 的线性区域上界与下界。

**💡 创新点**

①首次将 MoE 路由机制映射到热几何框架；②证明 Top‑k 路由等价于 k‑th 交替对称热多项式；③提出组合切片定理给出线性区域的精确阶数；④在“流形假设”下给出 MoE 的有效容量，展示稀疏路由在低维流形上保持组合深度。

**🔧 技术方法**

热几何（max‑plus 半环）、交替对称热多项式、超单纯形法线向量星、超平面排列理论、Zaslavsky 公式、球面几何和期望计数技术。

**📊 数据集**

无具体数据集；研究完全基于理论推导与假设（一般位置、正交性、流形光滑性）。

**📈 对比分析**

与稠密网络和 Top‑1 MoE 的容量进行理论比较。结果显示：稠密网络容量为 Θ(H^d_in)，Top‑1 MoE 为 Θ(N·H^d_in)，而 Top‑k MoE 为 Θ(Nk·(kH)^d_in)。在流形假设下，MoE 的有效容量为 Θ(Vol(π(𝓜))/Vol(S^{d_in-1})·Nk·(kH)^{d_eff})，相较于稠密网络的 Θ(H^{d_eff}) 具有 Nk 的组合加成。

**⚠️ 局限性**

①假设权重在一般位置和球面均匀分布；②要求 kH ≥ d_eff 以保证排列非退化；③未考虑训练过程、优化与实际稀疏度调节；④理论上限与实际模型表达力可能存在偏差。

---

## 433. From Scalar Rewards to Potential Trends: Shaping Potential Landscapes for Model-Based Reinforcement Learning

**arXiv ID:** 2602.03201 | [PDF](https://arxiv.org/pdf/2602.03201v1)

**作者:** Yao-Hui Li `[一作]` (Beijing Institute of Technology), Yonggang Zhang `[通讯]` (Jilin University)

**通讯引用:** 8380 | [OpenAlex ID](https://openalex.org/A5100383751)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

SLOPE通过将奖励建模从标量回归转为乐观潜力景观，解决了MBRL在稀疏奖励中的梯度稀缺问题。

**💡 创新点**

其创新点在于结合PBRS与分布式上限估计的乐观学习，实现稀疏奖励下的密集、方向性奖励景观，同时保持最优策略不变。

**🔧 技术方法**

采用分布式量化交叉熵损失、潜力函数 Φ=η·max_a Q(s,a)、动态MPPI启动、演示数据增强等技术。

**📊 数据集**

在5个基准（ManiSkill3、Meta‑World、Robosuite、Adroit 以及 DeepMind Control Suite）以及真实机器人任务上评估。

**📈 对比分析**

与 TD‑MPC2、MoDem、DEMO^3 和 BC 等基线对比，SLOPE 在稀疏、半稀疏和稠密奖励场景均取得显著性能提升，成功率最高。

**⚠️ 局限性**

局限在于对 η、τ 等超参的敏感性、仍需大量演示或安全约束，以及在极端复杂任务中可能的过度乐观导致收敛不稳定。

---

## 434. Prompt Augmentation Scales up GRPO Training on Mathematical Reasoning

**arXiv ID:** 2602.03190 | [PDF](https://arxiv.org/pdf/2602.03190v1)

**作者:** Wenquan Lu `[一作]` (Brown University), Randall Balestriero `[通讯]` (Brown University)

**通讯引用:** 891 | [OpenAlex ID](https://openalex.org/A5047293370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入 Prompt Augmentation 结合 Group‑Relative Policy Optimization (GRPO) 进行数学推理 LLM 的强化学习后训练，使用多种提示模板和对应格式奖励让模型在同一次训练中生成多样化推理路径。

**💡 创新点**

通过 Prompt Augmentation 与模板特定的格式奖励，成功延长低熵训练周期、稳定收敛，并在不使用 KL 正则化的情况下实现更高的推理准确率。

**🔧 技术方法**

技术包括 GRPO/DAPO 强化学习框架、token‑level policy gradient、decoupled clipping、去除 KL 损失、格式奖励（基于标签计数/字符串匹配）、vLLM 推理引擎、SymPy 与 Math‑Verify 验证器。

**📊 数据集**

使用 MATH Level 3–5（8,523 题）作为训练集，评估集涵盖 AIME24、AMC、MATH500、Minerva、OlympiadBench 五个数学竞赛基准。

**📈 对比分析**

与 GRPO、DAPO、Dr. GRPO、SEED GRPO、GMPO 等基线对比，模型在所有基准上分别达 44.5% 的 benchmark 平均准确率和 51.3% 的 question 平均准确率，明显优于现有方法。

**⚠️ 局限性**

局限性：Prompt Augmentation 只能缓解而非彻底解决长周期训练中的性能衰减；反射式模板的训练成本高、收敛慢；依赖可验证奖励，可能对非可验证任务适用性有限。

---

## 435. StreamShield: A Production-Proven Resiliency Solution for Apache Flink at ByteDance

**arXiv ID:** 2602.03189 | [PDF](https://arxiv.org/pdf/2602.03189v1)

**作者:** Yong Fang `[一作]` (ByteDance Inc), Chi Zhang `[通讯]` (ByteDance Inc)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在字节跳动的 Apache Flink 大规模生产环境中，设计并部署了一套完整的容错与弹性框架，涵盖引擎层的运行时优化与细粒度恢复、集群层的混合复制与外部依赖 HA、以及发布层的 chaos 与性能测试流水线。

**💡 创新点**

创新点包括：
1) 通过 Adaptive Shuffle、WeakHash、AutoScaling 等引擎优化实现动态负载均衡与自动并行度调整；
2) 引入 Region Checkpointing、Single-task Recovery、State LazyLoad 等细粒度恢复技术，显著降低恢复时间；
3) 采用混合复制策略，根据业务优先级动态切换主动/被动复制；
4) 针对外部系统（HDFS、Zookeeper、Gödel）的多层 HA 与容错机制；
5) 构建完整的 chaos + performance 测试与在线 probe 发布管线，确保功能与性能双重可靠性。

**🔧 技术方法**

使用技术：Apache Flink 1.11、RocksDB 状态后端、增量检查点、HDFS、Zookeeper、Gödel、任务管理器与作业管理器调度、Java/Scala 业务实现、Docker/容器化部署、Spark/自研监控与报警系统。

**📊 数据集**

数据集与基准：生产实时流数据（Nexmark Q2/Q12、Data Synchronization、Sample Stitching）、Nexmark suite（模拟拍卖场景）以及自研的 log‑driven 与 ETL 工作负载。

**📈 对比分析**

对比方法：在 384 核、2.3 TB 内存、15 TB NVMe、100 Gbps 网络的 ByteDance 集群上，分别测量 job 启动时间、吞吐量、恢复成功率、QPS 等指标。结果显示：
- 资源分配和任务部署时间降低 50%+；
- Adaptive Shuffle 提升吞吐量 400–1600%；
- AutoScaling 与输入速率线性关联，动态扩缩容无明显性能波动；
- Region Checkpointing 检查成功率从 54% 提升至 94%；
- Single-task Recovery 使失败时 QPS 几乎保持不变（相较于 0 的 baseline）；
- HotUpdate 与批量部署显著缩短作业重启延迟。

**⚠️ 局限性**

局限性：
- 方案仍需要人工调参（如阈值、复制比例等）；
- 在极端多级负载或极低延迟场景下，单任务恢复可能无法满足严格一致性要求；
- 外部系统完全失效仍可能导致整体停机，需进一步完善全局容错；
- 现有实现主要针对字节跳动内部业务，对其它行业或规模的迁移需验证；
- 需要进一步加入预测性诊断与自愈能力，以减少人为干预。

---

## 436. DynSplit-KV: Dynamic Semantic Splitting for KVCache Compression in Efficient Long-Context LLM Inference

**arXiv ID:** 2602.03184 | [PDF](https://arxiv.org/pdf/2602.03184v1)

**作者:** Jiancai Ye `[一作]` (Shanghai Jiao Tong University), Guohao Dai `[通讯]` (Infinigence-AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对长上下文大模型推理中 KVCache 过大导致的内存与吞吐瓶颈，本文提出 DynSplit‑KV，动态识别语义边界并对 KVCache 进行分块压缩，减少 KV 访问次数与显存占用。

**💡 创新点**

创新点在于：① 基于注意力依赖的动态重要性评分（DD‑Select）自适应挑选分块分隔符，避免固定间隔或预定义标点导致的语义错位；② 变长块到固定块的映射策略（V2F）通过块级压缩与 token 级映射，消除变长块带来的额外推理开销。

**🔧 技术方法**

技术手段包括：1) 利用多头注意力矩阵计算分隔符重要性分数；2) 通过长度控制与权重平衡实现动态语义分割；3) 对每块使用最大/最小/均值压缩；4) 采用 top‑k 块/token 选择实现并行 KV 访问；5) 在 GPU 与 CPU‑GPU 混合部署下实现 KVOffload。

**📊 数据集**

实验数据集涵盖 LongBench、LongBench v2、passkey‑retrieval 以及多种长文本推理任务；模型包括 Mistral‑7B‑Instruct、LongChat‑v1.5‑7B、DeepSeek‑R1‑Distill‑Llama‑8B 与 Llama‑3‑8B‑1M。

**📈 对比分析**

与 token‑level（StreamingLLM、H2O）、block‑level（Quest、InfLLM、ChunkKV）以及 sentence‑level（SentenceKV）基线相比，DynSplit‑KV 在相同 KVCache 使用率下准确率提升 5–55 % 之间；在 FlashAttention 上可实现 2.2× 的推理速度提升、2.6× 的峰值显存下降，并在大多数任务上接近全注意力性能。

**⚠️ 局限性**

主要限制包括：① 需要额外的前向注意力计算来估算分隔符重要性，导致推理前期略有开销；② 对极端非文本或极不规则语义结构（如某些代码/标记混合文本）可能仍需人工调优分隔符权重；③ 目前评测范围集中在英文本与代码任务，对多语言或特殊符号集的泛化尚未充分验证。

---

## 437. LaVPR: Benchmarking Language and Vision for Place Recognition

**arXiv ID:** 2602.03253 | [PDF](https://arxiv.org/pdf/2602.03253v1)

**作者:** Ofer Idan `[一作]` (Bar Ilan University), Yoli Shavit `[通讯]` (Bar Ilan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LaVPR 基准，通过为现有 VPR 数据集添加 65.1 万条高质量自然语言描述，探讨多模态融合与跨模态检索的效果。

**💡 创新点**

创新点在于：①构建大规模文本-视觉配对数据集；②提出 ADS‑LLP 语义加权融合方案；③将 LoRA 与 Multi‑Similarity 损失结合实现高效跨模态对齐。

**🔧 技术方法**

使用的技术包括：视觉 Transformer（NetVLAD、MixVPR、CricaVPR 等）、文本编码器 BGE‑Large、Late‑Fusion（CAT、PA、MLP、ADS）、Learned Language Pooling、LoRA 参数高效微调、Multi‑Similarity 损失及人机交互式数据清洗。

**📊 数据集**

基准数据集为扩展后的 GSV‑Cities、SF‑XL、Pitts‑30k、MSLS 等 VPR 数据集，新增 651,865 条对应文本，覆盖多种极端环境（雨、雾、模糊等）。

**📈 对比分析**

与传统视觉‑单模态基线相比，La‑方法在 R@1 上提升了 10–40%（例如 ViT‑S + BGE‑S 在 MSLS‑Blur 上达 93.0%），跨模态检索通过 LoRA‑MS 获得 R@1 提升至 13.8–35.7%（相对零样本 0.1–3%），同时计算成本下降 60%。

**⚠️ 局限性**

局限性包括：仅采用 Late‑Fusion，未探索早期/中期融合；在 SSL 视觉骨干上增益有限；跨模态检索仍落后于视觉单模态，且对复杂文本的对齐尚不充分。

---

## 438. A thin and soft optical tactile sensor for highly sensitive object perception

**arXiv ID:** 2602.03248 | [PDF](https://arxiv.org/pdf/2602.03248v1)

**作者:** Yanchen Shen `[一作]` (Kanazawa University), Satoshi Sunada `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并实现了一种薄型、柔性、无对准需求的光学触觉传感器，利用硅胶内部散射产生的散斑模式来感知接触位置、施加力以及表面纹理。

**💡 创新点**

创新点在于采用光散斑干涉原理实现对接触信息的编码，消除了传统视基触觉传感器所需的镜头、精密光学对准和复杂光学结构，仅需激光、软硅胶、光纤和摄像头即可实现高分辨率、低成本的触觉感知。

**🔧 技术方法**

主要技术包括：光散斑生成与捕捉、基于卷积神经网络的端到端解码模型、软硅胶与玻璃微珠的可制造混合介质以及利用Raspberry Pi实现实时推理。

**📊 数据集**

使用的实验数据集包括：机器人臂在传感器表面四个位置、三处不同力度的压痕数据、以及200/40样本的麻将牌纹理识别数据（共9类），每类包含不同抓取条件下的散斑图像。

**📈 对比分析**

与现有视基触觉传感器（如GelSight、ChromaTouch等）相比，该传感器在保持相同或更低的光学对准要求的同时，取得了约0.04 N（40 mN）的RMSE力测量误差和93.33%的麻将纹理分类准确率，系统高度低于3 mm，组件数仅为4个，显著降低了体积和制造复杂度。

**⚠️ 局限性**

局限性包括：对激光光源稳定性与光纤耦合质量敏感；散斑模式易受环境光和温度变化影响；目前仅针对低频静态或慢速触摸的实验验证，动态高速触觉或大力范围的性能尚未充分评估。

---

## 439. MentalSeek-Dx: Towards Progressive Hypothetico-Deductive Reasoning for Real-world Psychiatric Diagnosis

**arXiv ID:** 2602.03340 | [PDF](https://arxiv.org/pdf/2602.03340v1)

**作者:** Xiao Sun `[一作]` (Chongqing University), Kaiwen Wei `[通讯]` (Chongqing University)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5068039769)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向真实临床场景的细粒度精神疾病诊断基准（MentalDx Bench）和基于临床推理的专属大语言模型（MentalSeek-Dx），并在该基准上进行评测。

**💡 创新点**

创新点在于：①构建具有ICD‑11规范的双层标签（类别+疾病）EHR基准；②通过“假设-演绎轨迹”SFT与奖励驱动的课程强化学习，使模型从表面关联转向结构化的临床推理；③以较小参数规模实现SOTA诊断性能。

**🔧 技术方法**

使用技术包括：知识库检索（ICD‑11结构化信息）、假设-演绎轨迹构建（HDR）、奖励设计（R_cat, R_hypo, R_diff）、课程强化学习（PRCRL），以及传统的SFT、PPO‑style RL。

**📊 数据集**

数据集：MentalDx Bench（712份去标识化EHR，12名精神科医师按ICD‑11标注，覆盖16类、76种疾病），以及15,000份用于SFT的临床病例。

**📈 对比分析**

与18款主流LLM（包括小模型、通用LLM和医学专用LLM）对比，MentalSeek‑Dx在类别准确率（CA）~83.99%、疾病准确率（DA）~70.08%、联合准确率（JA）~69.38%方面实现SOTA，明显优于同规模及更大模型。

**⚠️ 局限性**

局限性：仅使用文本EHR，缺少面部表情、语音等多模态信息；模型仍存在解释性与安全风险，仅作为决策支持工具；且对罕见疾病的泛化能力需进一步验证。

---

## 440. GRAM: Spatial general-purpose audio representations for real-world environments

**arXiv ID:** 2602.03307 | [PDF](https://arxiv.org/pdf/2602.03307v1)

**作者:** Goksenin Yuksel `[一作]` (Donders Institute), Kiki van der Heijden `[通讯]` (Mortimer B Zuckerman Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GRAM 模型，利用多通道掩码自编码器学习空间音频表示，并在模拟与真实环境下进行评估。

**💡 创新点**

创新点包括：①基于多通道 MAE 的空间音频学习；②构建 85k 真实场景模拟数据集（BRIR/ARIR）并用于预训练；③发布 NatHEAR 与 RealSELD 两大基准；④展示仅使用少量训练数据即可超越现有自监督模型。

**🔧 技术方法**

技术手段：多通道掩码自编码器（Transformer/Mamba 编码器+局部-全局注意力），语谱图 + 强度向量特征，线性位置编码，掩码比例 0.8，局部-全局注意力窗口多尺度。

**📊 数据集**

使用的数据集包括：SoundSpace 2.0 模拟的 85k BRIR/ARIR 场景、AudioSet、WHAMR! 背景噪声、HEAR、NatHEAR（模拟空间任务）以及 RealSELD（真实场景 SELD 任务）。

**📈 对比分析**

与现有自监督音频模型（AST、SSAST、MWMAE 等）在 HEAR、NatHEAR、RealSELD 上进行统一评估，GRAM 在大多数任务上实现了显著提升（如 HEAR 平均分 93/90，NatHEAR 位置误差最低，RealSELD 位置误差比监督模型低 1–3°），并在训练样本量上更高效。

**⚠️ 局限性**

局限性：Binaural 版因 Mel 频谱分辨率不足难以捕获 ITD，导致定位精度不及 Ambisonics 版；未充分验证在极端多源、移动源的动态场景中的鲁棒性；未来工作需改进频谱分辨率并扩展到多模态学习。

---

## 441. A Pipeline for ADNI Resting-State Functional MRI Processing and Quality Control

**arXiv ID:** 2602.03278 | [PDF](https://arxiv.org/pdf/2602.03278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 442. medR: Reward Engineering for Clinical Offline Reinforcement Learning via Tri-Drive Potential Functions

**arXiv ID:** 2602.03305 | [PDF](https://arxiv.org/pdf/2602.03305v1)

**作者:** Qianyi Xu `[一作]` (National University of Singapore), Mengling Feng `[通讯]` (National University of Singapore)

**通讯引用:** 12007 | [OpenAlex ID](https://openalex.org/A5022222926)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于大型语言模型的自动化奖励工程管道，利用三驱动潜在函数（生存、置信、能力）生成离线临床强化学习的奖励函数，并通过离线适配度指标进行筛选。

**💡 创新点**

创新点包括：①使用LLM进行可解释特征选择；②设计三驱动潜在函数结构以同时平衡生存、置信度与干预成本；③构建离线评估指标和基于NSGA‑II的Pareto前沿选择；④在三大临床任务上验证框架的通用性。

**🔧 技术方法**

采用的技术有：大型语言模型（Prompt + 代码生成）、潜在函数奖励塑形、离线RL算法（BCQ、IQL）、非支配排序遗传算法（NSGA‑II）进行多目标优化、离线评估指标（WIS、Survivor Agreement）以及不确定性量化。

**📊 数据集**

使用的数据集包括：MIMIC‑IV（败血症）、eICU‑CRD（机械通气）和 AmsterdamUMCdb（持续肾替代治疗）。

**📈 对比分析**

与三类手工奖励基线（ORM、PRM、OPRM）和两类LLM基线（LLMR、CodeGen）进行对比。评估指标包括WIS、存活患者的行动一致率和三驱动适配度。结果显示medR在所有任务中均优于基线，WIS提升约15–25%，行动一致率提升约10–20%，三驱动适配度均保持在最高水平。

**⚠️ 局限性**

局限性包括：对LLM提示质量的依赖、潜在的偏见与幻觉风险、仅在离线环境验证、缺乏前瞻性临床验证、对高维复杂干预空间的通用性仍需进一步评估。

---

## 443. Anomaly Detection via Mean Shift Density Enhancement

**arXiv ID:** 2602.03293 | [PDF](https://arxiv.org/pdf/2602.03293v1)

**作者:** Pritam Kar `[一作]` (Indian Institute of Science Education and Research), Saptarshi Bej `[通讯]` (Indian Institute of Science Education and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种全无监督的异常检测方法 Mean Shift Density Enhancement (MSDE)，通过自适应加权均值漂移迭代捕捉样本在密度驱动的几何漂移中的累计位移来判定异常。

**💡 创新点**

创新点在于将 UMAP 构造的模糊邻居图作为自适应权重源，结合加权均值漂移并将累积位移作为异常分数，突破传统的静态密度或距离度量，提供可解释的几何不稳定性评估。

**🔧 技术方法**

使用技术包括 UMAP 生成模糊邻居图、NN-Descent 近似 kNN、加权均值漂移迭代、基于累计位移的 Sigmoid 分数、以及 AUC‑ROC/AUC‑PR/Precision@n 等评估指标。

**📊 数据集**

在 ADBench 基准上评估，使用 46 个真实世界表格数据集，并通过四种合成异常类型（global、local、cluster、dependency）及六个噪声水平进行实验。

**📈 对比分析**

与 13 种无监督基线（KNN、LOF、CBLOF、IForest、PCA、OCSVM、HBOS、COPOD、ECOD、LODA、COF、DAGMM）对比，MSDE 在无噪声条件下平均 AUC‑ROC 0.922、AUC‑PR 0.714、Precision@n 0.694，且在多数噪声水平下保持前列，表现出稳定的高效性能。

**⚠️ 局限性**

局限性包括：需要构造近邻图并多次迭代均值漂移，计算量比单纯距离/密度方法大；依赖有意义的局部结构，在极高噪声或特征信息稀缺的场景下性能下降；对极大规模或流式数据需要进一步优化。

---

## 444. A3-TTA: Adaptive Anchor Alignment Test-Time Adaptation for Image Segmentation

**arXiv ID:** 2602.03292 | [PDF](https://arxiv.org/pdf/2602.03292v1)

**作者:** Jianghao Wu `[一作]` (University of Electronic Science and Technology of China), Shaoting Zhang `[通讯]` (Shanghai AI laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于锚点对齐的测试时自适应（A3‑TTA）框架，用以在无源数据、单轮在线环境下提升图像分割模型在域漂移场景中的表现

**💡 创新点**

通过引入Class Compact Density（CCD）度量筛选Anchor‑Target Images (ATIs)并构建特征库，实现对目标域样本的特征对齐；采用边界感知熵最小化和自适应EMA的Mean‑Teacher结构，解决伪标签噪声、错误累积和遗忘问题

**🔧 技术方法**

Class Compact Density、特征对齐与融合、边界熵约束、基于EMA的自适应教师更新、U‑Net/DeepLabV3+等网络

**📊 数据集**

心脏MRI多域数据（M&MS）、前列腺MRI多域数据（Prostate）、Cityscapes极端条件数据集（Fog/Noon/雨/雪）

**📈 对比分析**

与9种现有TTA方法（PTBN、TENT、MT、CoTTA、SAR、InTEnt、VPTTA、GraTa、EDCP）以及“Source Only”在Dice/ASSD/mIoU上进行比较，A3‑TTA在所有任务上均取得平均提升10–18个百分点，尤其在连续TTA和噪声鲁棒性方面表现最优

**⚠️ 局限性**

当域漂移极端（跨模态如CT↔MRI）时，ATIs可能不再贴近源分布；在3D全卷积网络和更大显存场景下的适用性受限；特征库初始化对首批样本适应效果影响明显

---

## 445. BlockRR: A Unified Framework of RR-type Algorithms for Label Differential Privacy

**arXiv ID:** 2602.03277 | [PDF](https://arxiv.org/pdf/2602.03277v1)

**作者:** Haixia Liu `[一作]` (Huazhong University of Science and Technology), Yi Ding `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 24181 | [OpenAlex ID](https://openalex.org/A5016874353)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BlockRR，一种统一的标签差分隐私随机响应机制；

**💡 创新点**

通过块化标签空间和权重矩阵分区，整合现有RR、RRWithPrior等算法，兼顾多数类和少数类；

**🔧 技术方法**

使用标签先验估计、权重矩阵分区、线性规划求解转移概率、并通过BlockRR实现隐私保证；

**📊 数据集**

在两个对CIFAR‑10做类不平衡抽样得到的CIFAR‑10_1和CIFAR‑10_2数据集上进行实验；

**📈 对比分析**

与标准RR和RRWithPrior比较，BlockRR在高隐私（ε≤1.0）时显著提升整体与均衡类精度，低隐私（ε≥4.0）趋同于RR，且表现更稳定；

**⚠️ 局限性**

需要手工调参（l、σ）且依赖标签先验估计，缺乏对块化噪声理论的深入分析。

---

## 446. Link Fraction Mixed Membership Reveals Community Diversity in Aggregated Social Networks

**arXiv ID:** 2602.03266 | [PDF](https://arxiv.org/pdf/2602.03266v1)

**作者:** Gamal Adel `[一作]`, Frank W. Takes `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了Link Fraction Mixed Membership (LFMM) 方法，用于在聚合网络上进行混合社区检测，并以荷兰人口规模社交网络为例，分析社区多样性与演化。

**💡 创新点**

LFMM 的创新点在于其线性定义的链接分数混合成员，能够在任何聚合与分解尺度下保持成员总量不变，解决传统方法的生态谬误与聚合敏感性问题。

**🔧 技术方法**

技术上先使用 Leiden 算法得到离散社区分区，然后通过单矩阵乘法计算节点的链接分数混合成员；同时利用 Gini‑Simpson 指数和重力 null 模型评估多样性显著性，并用合成 SBM 网络验证方法一致性。

**📊 数据集**

使用的核心数据集是荷兰注册基础的全国社交网络，涵盖约 1700 万居民、100 亿条社交关系，按社区和市镇两级进行聚合，时间跨度从 2009 年到 2021 年。

**📈 对比分析**

与单层聚合的 LFMM、合成网络中不同聚合比例和亲和度下的比较显示 Pearson 相关系数 ≥0.999，验证了方法的一致性；相比传统离散社区检测，LFMM 能更准确捕捉多样性与演化，揭示与城市化的显著关联。

**⚠️ 局限性**

方法局限在于其基于边权重而非节点数量，导致高强度节点对聚合特征影响较大；且无法区分聚合集合内部的异质连通与混合成员分布，需在更广泛的网络拓扑上进一步评估。

---

## 447. Beyond Suffixes: Token Position in GCG Adversarial Attacks on Large Language Models

**arXiv ID:** 2602.03265 | [PDF](https://arxiv.org/pdf/2602.03265v1)

**作者:** Hicham Eddoubi `[一作]` (University of Cagliari), Fadi Hassan `[通讯]` (Huawei Technologies Finland Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大语言模型（LLM） jailbreak 攻击中，逆向词的位置（前缀或后缀）对攻击成功率的影响，并系统评估了两种位置策略的效果。

**💡 创新点**

将逆向词位置作为攻击轴进行实验性探讨，首次证明位置变换能够显著提升攻击成功率，揭示现有安全评估方法的盲点。

**🔧 技术方法**

采用 Greedy Coordinate Gradient (GCG) 攻击的前缀/后缀变体进行白盒和跨模型黑盒实验，并结合注意力分布分析对成功攻击机制进行解释。

**📊 数据集**

使用 AdvBench 数据集随机抽取 100 条有害提示作为攻击目标，作为实验输入。

**📈 对比分析**

通过比较 ASR@k=1（仅评估优化位置）与 ASR@k=2（评估两种位置）来衡量位置差异，实验结果显示位置变换可使 ASR 提升多达约 49%，在多模型上均表现出显著改善。

**⚠️ 局限性**

仅针对 GCG 攻击及其前后缀两种位置进行研究，未覆盖更细粒度的位置信息或其他攻击方式；注意力分析仅为初步探讨，缺乏完整的机制解释。

---

## 448. On the Summability Problem of Multivariate Rational Functions in the Mixed Case

**arXiv ID:** 2602.03289 | [PDF](https://arxiv.org/pdf/2602.03289v1)

**作者:** Shaoshi Chen `[一作]` (Chinese Academy of Sciences), Yisen Wang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 3884 | [OpenAlex ID](https://openalex.org/A5084785077)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了多变量有理函数的可求和性问题，特别是在混合情况下，即同时存在移位和q-移位算子。通过引入轨道分解、佐藤的各向同性群和差分变换等概念，提出了可求和性标准。

**💡 创新点**

创新点在于解决了多变量有理函数的可求和性问题，特别是混合情况下的可求和性标准，推动了符号求和算法的发展。

**🔧 技术方法**

使用了轨道分解、佐藤的各向同性群理论和差分变换等数学工具。

**📊 数据集**

使用了多变量有理函数的理论，具体数据集未明确提及，但涉及到的函数和算子均为数学构造。

**📈 对比分析**

通过与已有的算法和理论进行比较，本文提出的标准能够有效地将多变量有理函数的可求和性问题简化为简单分数的情况，性能上具有显著的优势。

**⚠️ 局限性**

限制在于目前的研究主要集中在有理函数的情况，未来需要扩展到更广泛的函数类型和更复杂的算子组合。

---

## 449. The Personality Trap: How LLMs Embed Bias When Generating Human-Like Personas

**arXiv ID:** 2602.03334 | [PDF](https://arxiv.org/pdf/2602.03334v1)

**作者:** Jacopo Amidei `[一作]` (Universitat Oberta de Catalunya), Andreas Kaltenbrunner `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 2750 | [OpenAlex ID](https://openalex.org/A5053529497)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了使用大型语言模型（LLM）根据人格问卷（EPQR‑A）分数生成合成人口，并评估其社会人口属性的代表性与偏见。

**💡 创新点**

创新点在于：①首次以人格测验结果为唯一输入来驱动合成人口生成；②系统比较了不同LLM在人格最大化（Neuroticism/ Psychoticism）情景下的属性分布变化；③揭示了LLM在生成非二元、LGBTQ+等边缘群体时可能产生的病理化与刻板化倾向。

**🔧 技术方法**

技术方法包括：①基于EPQR‑A问卷的回答构建prompt，要求LLM生成JSON格式的人物描述；②使用五个LLM（GPT‑3.5、GPT‑4o、Claude‑3.5‑S、LLaMa3.2‑3B、LLaMa3.1‑70B）生成合成样本；③对生成的社会人口属性进行统计检验（t‑test、置信区间）；④评估人格一致性（MAE/RMSE、Cronbach’s α）与跨测量相关（EPQR‑A‑BFI Pearson相关）。

**📊 数据集**

数据集为826份基于EPQR‑A的模拟问卷回答（英文），并对照随机生成基线；评估指标使用这些问卷回答的原始得分与模型生成的回答比较。

**📈 对比分析**

比较方法：对不同模型、不同情景（基线、MaxN、MaxP）计算社会属性比例差异、t‑检验显著性、人格维度误差（MAE/RMSE）、内部一致性（Cronbach’s α）以及跨问卷相关。性能方面：LLM能较好地保留输入人格分数（误差低），但在属性分布上显现显著的WEIRD偏差，且在MaxP情景下非二元和LGBTQ+比例异常提升。

**⚠️ 局限性**

局限性：①仅使用模拟问卷数据，缺乏真实人类问卷对照；②只在英语环境下实验，未涉及多语言或多测验（如NEO‑PI‑R）;③未与其他基于人口属性或生物信息的合成方法进行对比；④对极端人格（N/P最大化）时的模型一致性与可靠性仍需进一步验证。

---

## 450. Tiled Prompts: Overcoming Prompt Underspecification in Image and Video Super-Resolution

**arXiv ID:** 2602.03342 | [PDF](https://arxiv.org/pdf/2602.03342v1)

**作者:** Bryan Sangwoo Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jong Chul Ye `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 17280 | [OpenAlex ID](https://openalex.org/A5012644755)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Tiled Prompts 框架，用局部视觉语言模型生成的 tile‑specific 文本提示来指导图像与视频超分辨率，从而解决单一全局提示导致的细节缺失和误导问题。

**💡 创新点**

创新点在于用分块级别的文本提示替代全局提示，显著提升高分辨率细节重建、文本对齐及视频时序一致性，并保持低额外计算成本。

**🔧 技术方法**

采用文本条件扩散模型（DiT4SR、STAR）与视觉语言模型（Qwen2.5‑VL‑7B、Qwen3‑VL‑8B）进行提示提取，结合 latent tiling、Gaussian 加权融合与 CFG 调节。

**📊 数据集**

实验使用 Urban100、OST300、LSDIR1K（图像）以及 VideoLQ、RealVSR、MVSR4x（视频）等真实高分辨率数据集。

**📈 对比分析**

相较于单一全局提示和全局+局部提示，Tiled Prompts 在无参考图像质量指标（NIQE、MUSIQ、CLIPScore、ImageReward 等）和视频质量指标（FasterVQA、FAST、DOVER 等）上均取得更高分数，证明了更好的细节重建与文本对齐。

**⚠️ 局限性**

局限性在于需为每个 tile 运行一次 VLM，导致计算量略有增加；此外 VLM 可能误解局部内容或产生幻觉，进而影响最终重建质量。

---

## 451. PWAVEP: Purifying Imperceptible Adversarial Perturbations in 3D Point Clouds via Spectral Graph Wavelets

**arXiv ID:** 2602.03333 | [PDF](https://arxiv.org/pdf/2602.03333v1)

**作者:** Haoran Li `[一作]` (Northeastern University), Jian Xu `[通讯]` (Northeastern University)

**通讯引用:** 2064 | [OpenAlex ID](https://openalex.org/A5085728168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种非侵入式、无训练的3D点云对抗防御框架（PWaveP），通过谱图波函数进行分层净化；

**💡 创新点**

创新点在于：①理论证明不可察觉攻击主要集中在图谱高频域；②结合谱-空间混合显著性评分，分层过滤和删除高危点；③使用局部稀疏与梯度信息实现自适应净化；

**🔧 技术方法**

核心技术包括：图谱傅里叶变换（GFT）、图谱波函数变换（GWT）及其逆变换、混合显著性评分、梯度引导的频带衰减与高危点剔除；

**📊 数据集**

实验数据集为ModelNet40与ShapeNet，使用DGCNN、PointNet、PointNet++、CurveNet等四种分类模型；

**📈 对比分析**

与SOR、ROR、PointCVAR、PFourierP、Ada3Diff等基线对比，PWaveP在多种白盒攻击（GSDA、GeoA3、HiT-ADV、SI-ADV、Eidos）下，分类准确率平均提升至90%以上，且在干净数据上的准确率仅下降约1%；

**⚠️ 局限性**

局限性包括：对黑盒梯度近似时性能可能显著下降，且在某些网络架构下对点数稀疏的适应性需进一步验证。

---

## 452. PQTNet: Pixel-wise Quantitative Thermography Neural Network for Estimating Defect Depth in Polylactic Acid Parts by Additive Manufacturing

**arXiv ID:** 2602.03314 | [PDF](https://arxiv.org/pdf/2602.03314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 453. Bayesian Conformal Prediction as a Decision Risk Problem

**arXiv ID:** 2602.03331 | [PDF](https://arxiv.org/pdf/2602.03331v1)

**作者:** Fanyi Wu `[一作]` (UKRI AI Centre for Doctoral Training in Decision Making for Complex Systems), Michele Caprio `[通讯]` (University of Manchester)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5000098655)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于贝叶斯后验预测分布的贝叶斯改进分裂式共形预测框架（BCP），通过决策理论优化阈值以实现有效且可靠的预测集。

**💡 创新点**

创新在于将共形阈值视为决策变量，用贝叶斯后验预测分数和贝叶斯积分实现对期望预测集大小的最小化，并结合 L+ 风险控制提供 PAC‑style 置信保证。

**🔧 技术方法**

核心技术包括 AOI 后验采样构造非共形分数、贝叶斯求积（BQ）估计输入期望、以及基于 Dirichlet 的 L+ 误差控制；实现了完整的 split‑CP + CRC 工作流程。

**📊 数据集**

在三类任务上验证：糖尿病稀疏线性回归数据集、威斯康星乳腺癌二分类数据集以及 ImageNet‑A 失配分布的图像分类。

**📈 对比分析**

与 Split‑CP、CB、BCI、MSP 等方法对比，BCP 在覆盖率上保持 80% 目标，预测集尺寸与 CB、Split‑CP 相当但方差更低；在先验失配时 BCP 维持近名义覆盖率，而纯贝叶斯置信区间明显欠覆盖。

**⚠️ 局限性**

局限包括额外的后验采样与阈值调参导致计算开销增大，对高维模型的 BQ 需要近似，且在严重失配或非参数贝叶斯模型中效果仍待验证。

---

## 454. Information-Theoretic Multi-Model Fusion for Target-Oriented Adaptive Sampling in Materials Design

**arXiv ID:** 2602.03319 | [PDF](https://arxiv.org/pdf/2602.03319v1)

**作者:** Yixuan Zhang `[一作]` (Technical University of Darmstadt), Hongbin Zhang `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 17057 | [OpenAlex ID](https://openalex.org/A5100442145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种信息论驱动的多模型融合框架，用以在材料设计中实现目标导向的自适应采样，重点将全局函数逼近转化为低熵轨迹发现；

**💡 创新点**

创新点在于三轴（数据、模型、物理）协同的认知系统与四阶段信息管控策略，结合维度感知容量限制、目标加权自举蒸馏、结构感知候选组织以及Kalman式多源融合，实现对高维稀疏空间的有效熵压缩；

**🔧 技术方法**

核心技术包括：维度感知信息预算与模型容量对齐、目标加权自举蒸馏的异构代理集、结构化候选云的低维聚类与权重化、以及基于R²与ELPD两通道的KF/rKF逆融合；

**📊 数据集**

在14个真实材料设计任务（样本数从600到400万、特征维度10–1025）以及20维Ackley、Rastrigin、Schwefel三类合成基准上进行评估；

**📈 对比分析**

与传统GP、Deep‑Kernel、BO‑HVI、EVO、CMA‑ES等基线相比，统一的10‑迭代/100‑评估预算下成功率100%（极端案例90%），平均收敛迭代数2–21次，显著提升了样本效率和可靠性；

**⚠️ 局限性**

局限性包括对物理先验与候选结构的依赖、对极端高维稀疏区域的探索仍需改进、以及对动态预算与多目标协同策略的进一步优化空间。

---

## 455. Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity

**arXiv ID:** 2602.03315 | [PDF](https://arxiv.org/pdf/2602.03315v1)

**作者:** Menglin Xia `[一作]` (Microsoft), Saravan Rajmohan `[通讯]` (Microsoft)

**通讯引用:** 11294 | [OpenAlex ID](https://openalex.org/A5100331488)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为 Memora 的“谐波”记忆表示方法，结合主抽象与提示锚点构建结构化记忆，并通过策略导向的检索（MDP+策略优化）实现高效、可扩展的上下文检索与推理。

**💡 创新点**

创新点包括：① 通过主抽象聚合相关信息，避免碎片化；② 提示锚点提供细粒度、多模态检索路径；③ 将检索视为多步决策过程，利用策略网络主动探索多跳依赖；④ 理论证明 Memora 能包容传统 RAG 与知识图谱检索，构成统一框架。

**🔧 技术方法**

技术手段：数据分段、语义分块、主抽象与提示锚点生成、基于 LLM 的抽象匹配与更新、策略导向检索 MDP、Group‑Relative Policy Optimization (GRPO) 训练。

**📊 数据集**

使用的基准数据集为 LoCoMo（多轮对话 600 轮、≈20k tokens）和 LongMemEval（极长上下文 115k tokens，500 题目）。

**📈 对比分析**

与全上下文、RAG、Mem0、Zep、Nemori 等方法对比，在 LoCoMo 上 LLM‑as‑a‑Judge 得分 86.3%（政策检索）/85.3%（语义检索），在 LongMemEval 上 87.4%，均优于所有基线；同时检索成本低 98% 以内，显著降低 token 与延迟。

**⚠️ 局限性**

局限性：① 政策检索需多轮 LLM 调用，导致延迟上升；② 记忆粒度与上下文规模之间权衡，过细粒度会增大内存；③ 策略训练依赖人工评估或预先设定的奖励，泛化性尚待验证。

---

## 456. Multi-Level Testing of Conversational AI Systems

**arXiv ID:** 2602.03311 | [PDF](https://arxiv.org/pdf/2602.03311v1)

**作者:** Elena Masserini `[一作]` (University of Milano-Bicocca), Elena Masserini `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 27 | [OpenAlex ID](https://openalex.org/A5065145526)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对对话式人工智能系统的多粒度测试，提出了从语言组件与服务交互到单一代理、再到多代理系统的三层自动化测试框架，涵盖了反馈驱动的灰盒测试、规范与变形测试以及基于规划与协同的多代理测试。

**💡 创新点**

创新点包括：①利用LLM辅助的灰盒搜索实现服务调用覆盖；②结合规范与变形关系的测试生成，克服对话式系统缺乏形式化需求的难题；③在多代理场景中引入测试与模拟代理，结合规划与编排技术实现端到端测试，提升缺陷发现率。

**🔧 技术方法**

核心技术为：LLM驱动的文本生成与语义匹配；灰盒搜索与反馈指引；变形（metamorphic）测试；AI规划与编排；变异测试（mutation testing）；以及基于覆盖率和错误检测的评估指标。

**📊 数据集**

使用的实验数据集包括：已构建的基于Rasa与Dialogflow的单代理对话系统集合，后续扩展至开源生成式代理和多代理系统，用于验证所提方法的普适性。

**📈 对比分析**

与Botium、Charm等基线方法比较，本文方法在对话覆盖率、代码覆盖率、变异得分以及实际缺陷揭示数量上均优于基线，显著提升了测试效果与缺陷定位的实用价值。

**⚠️ 局限性**

主要限制在于：实验覆盖的系统类型相对集中，难以全面代表所有对话式AI架构；测试所需的规范与变形关系仍需人工编写，对复杂业务场景仍有挑战；LLM生成的测试可能产生语义偏差，需进一步验证其可靠性。

---

## 457. Full end-to-end diagnostic workflow automation of 3D OCT via foundation model-driven AI for retinal diseases

**arXiv ID:** 2602.03302 | [PDF](https://arxiv.org/pdf/2602.03302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 458. LogicScan: An LLM-driven Framework for Detecting Business Logic Vulnerabilities in Smart Contracts

**arXiv ID:** 2602.03271 | [PDF](https://arxiv.org/pdf/2602.03271v1)

**作者:** Jiaqi Gao `[一作]` (Beijing Institute of Technology), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 48714 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

开发了一种基于大语言模型的自动对比审计框架 LogicScan，用来检测智能合约的业务逻辑漏洞。

**💡 创新点**

创新点在于：①利用成熟链上协议的业务惯例自动挖掘共识式业务不变式；②设计可验证的业务规范语言 BSL，降低 LLM 幻觉；③引入多维代码提取和噪声感知逻辑聚合，显著抑制误报。

**🔧 技术方法**

技术主要包括：大语言模型（如 GPT‑5、Claude‑Sonnet‑4.5、Qwen‑3‑235B）、语法约束的 BSL 生成与验证、向量检索 + 关系数据库存储、对比审计的多轮提示、噪声聚合投票。

**📊 数据集**

使用三大真实数据集：DeFiHacks（52 个项目 70 条漏洞）、Web3Bugs（65 项 134 条漏洞）以及 Top‑200 市值合约（无已知漏洞）。

**📈 对比分析**

与 GPTScan、ZepScope、Slither 等基线工具对比，LogicScan 在所有数据集上取得最高 F1（整体 77.7%），召回率高（94.3%/82.8%）且误报率低（Top‑200 仅 18/12/10），且在不同 LLM 后端保持稳定。

**⚠️ 局限性**

局限性：只关注单调用层面的业务约束，无法检测多步执行、精度累积等序列放大漏洞；误报主要来自合法的设计差异；对非 DeFi 或非 EVM 生态的泛化仍待验证。

---

## 459. Composable Visual Tokenizers with Generator-Free Diagnostics of Learnability

**arXiv ID:** 2602.03339 | [PDF](https://arxiv.org/pdf/2602.03339v1)

**作者:** Bingchen Zhao `[一作]` (ByteDance Seed), Yu Tian `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于扩散解码器的1D视觉tokenizer，目标是生成可学习、可控制的token空间；并引入两种无生成器的诊断指标（AvgIG和MC），用于评估token的可用性和几何可学习性；在此基础上训练tokenizer并验证其对下游生成器的提升。

**💡 创新点**

①将互信息监督、token交换复合训练与对抗流模型相结合，直接约束token的可利用性与可组合性；②提出AvgIG和MC两项无监督指标，能更可靠地预测下游生成质量与任务性能；③证明在ImageNet、遥感、医学影像与OCR等多域任务上，AvgIG和MC与生成质量/任务效能的相关性显著高于传统的rFID。

**🔧 技术方法**

使用扩散解码器（conditional denoising）、互信息损失（InfoGAN式）、token交换（swap）与互信息监督、对抗流模型（AFM）作为对抗流判别器、MaskGIT+扩散回归头的生成器、以及Diffusion Token Head。

**📊 数据集**

主要在ImageNet进行tokenizer训练与生成评估；跨域任务使用DeepGlobe（遥感分割）、ChestX-ray14（医学分割）、TextOCR（文本识别）。

**📈 对比分析**

对比了多种2D网格与1D序列tokenizer（如VQGAN、ViT-VQGAN、TiTok、Semanticist、FlexTok等）。在ImageNet上，所提出方法在rFID相近的情况下，AvgIG和MC均显著提升，生成器的gFID也从2.7下降到1.6，显示生成质量提升；在跨域任务上，AvgIG和MC与任务指标的Pearson相关系数均高于rFID。

**⚠️ 局限性**

局限性在于：①仅在图像分类与生成任务上验证，对更复杂场景或视频tokenization的适用性尚未探测；②对抗流模型和互信息监督增加训练复杂度；③指标AvgIG和MC需要对tokenizer进行多步优化和路径评估，计算开销较大；④最终生成效果仍受decoder和生成器架构限制，未完全解决所有生成细节问题。

---

## 460. Accurate Failure Prediction in Agents Does Not Imply Effective Failure Prevention

**arXiv ID:** 2602.03338 | [PDF](https://arxiv.org/pdf/2602.03338v1)

**作者:** Rakshith Vasudev `[一作]` (Writer), Waseem Alshikh `[通讯]` (Writer)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在LLM代理执行时使用LLM critic进行主动干预，并给出一种基于干预的失效与恢复率的决策框架。

**💡 创新点**

提出“失效–恢复权衡”条件（p > d/(r+d)）决定干预是否有利，证明单纯的预测准确性并不能保证干预收益；并给出基于小规模 pilot 的部署判断方法。

**🔧 技术方法**

使用二分类LLM critic（Qwen-3-0.6B、LoRA 细调）、温度缩放校准、两种简单干预机制（回滚与警告），以及基准模型 Qwen‑3‑8B、GLM‑4.7、MiniMax‑M2.1。

**📊 数据集**

实验涵盖 HotPotQA、GAIA、ALFWorld 三个 benchmark，分别代表高成功、中等成功和低成功场景。

**📈 对比分析**

与无干预基线、随机干预、固定步骤干预、以及理论上最优 oracle 干预和最佳两样本选择进行对比。结果显示，在高成功场景干预可导致 2–30 pp 的退化；在低成功场景干预可获得 1–3 pp 的正增益；oracle 干预上限仅 3–8 pp，远低于最佳两样本选择的 11–17 pp。

**⚠️ 局限性**

局限包括：干预机制过于简单、LLM critic 规模受数据多样性限制、低成功场景收益仍有限、pilot 估计仅在分布内有效、无法覆盖更复杂的干预方法或其他任务域。

---

## 461. Lipschitz Multiscale Deep Equilibrium Models: A Theoretically Guaranteed and Accelerated Approach

**arXiv ID:** 2602.03297 | [PDF](https://arxiv.org/pdf/2602.03297v1)

**作者:** Naoki Sato `[一作]` (Meiji University), Hideaki Iiduka `[通讯]` (Meiji University)

**通讯引用:** 1889 | [OpenAlex ID](https://openalex.org/A5016609320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Lipschitz MDEQ 通过重构 MDEQ 结构，使固定点映射的 Lipschitz 常数可控并小于 1，从而理论上保证前向和反向迭代收敛，显著加速训练与推理。

**💡 创新点**

创新点在于为 DEQ 的归一化、激活、卷积、残差与融合层等各操作引入可调的 Lipschitz 上界，并通过凸组合与 softmax 加权平均等手段，使整个模型在保持高效的同时获得可证明的收敛性。

**🔧 技术方法**

使用的技术包括卷积谱范数约束、Mean‑Only Group Normalization、Scaled ReLU、可学习的凸组合参数、softmax 权重调制、Jacobian‑free/backward 传播、Anderson 加速的固定点求解器等。

**📊 数据集**

主要在 CIFAR‑10 图像分类数据集上进行实验；AFHQ 数据仅用于示例图像展示。

**📈 对比分析**

与原 MDEQ、Jacobian 正则化、Phantom Gradient、JFB 等方法对比，Lipschitz MDEQ 在 CIFAR‑10 上实现最高 4.75×（或 6.43×）的速度提升，同时准确率略有下降；当 Lipschitz 常数设为 1 时可获得约 3.33×/4.98× 的加速；当设为 0.03 时可获得 4.75×/6.43× 的加速。

**⚠️ 局限性**

局限性在于需要手动调节 Lipschitz 上界并在一定程度上牺牲精度；约束卷积谱范数会限制模型表达能力；理论与实际收敛性仍存在差距，且对其他任务或模型的推广尚未验证。

---

## 462. LEVIO: Lightweight Embedded Visual Inertial Odometry for Resource-Constrained Devices

**arXiv ID:** 2602.03294 | [PDF](https://arxiv.org/pdf/2602.03294v1)

**作者:** Jonas Kühne `[一作]` (ETH Zurich), Luca Benini `[通讯]` (University of Bologna)

**通讯引用:** 56454 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了LEVIO，一种轻量级视觉惯性里程计，可在低功耗SoC（如GAP9）上实现实时6自由度跟踪。

**💡 创新点**

创新点在于在缺少循环闭环、仅使用QQVGA低分辨率图像的前提下，结合并行化、低内存自定义线性代数库和硬件软件协同优化，完成完整的VIO管线。

**🔧 技术方法**

使用技术包括ORB特征提取与描述、8点和EPnP RANSAC、IMU预积分、姿态图（BA+IMU融合）优化、八核并行化、定制SVD与Jacobi求解器等。

**📊 数据集**

使用公开的EuRoC UAV数据集进行算法评估与硬件部署。

**📈 对比分析**

通过与VINS‑Mono、ORB‑SLAM3等主流系统在相同数据集上的对比，LEVIO在低于100 mW功耗下实现20 fps，RMSE约0.9–3.5 m（与高端系统精度相对较低但功耗优势明显）。

**⚠️ 局限性**

局限性包括：仅使用低分辨率QQVGA图像、缺乏循环闭环导致长期漂移、对大规模环境适应性有限、仅在单目和GAP9平台验证。

---

## 463. Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis

**arXiv ID:** 2602.03279 | [PDF](https://arxiv.org/pdf/2602.03279v1)

**作者:** Zhengbo Jiao `[一作]` (Alibaba Group Holding Limited), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14228 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Agentic Proposing 框架，通过可组合的推理技能、内部反思、工具调用和多粒度奖励机制，自动生成高难度、可验证的数学、编程与科学问题数据。

**💡 创新点**

创新点在于把问题生成视作目标驱动的组合逻辑工程，构建技能库、引入内部反思与工具自校验，并利用 MGPO 进行多粒度奖励的强化学习，从而实现自我纠错与精准难度控制。

**🔧 技术方法**

采用了可组合技能库、POMDP 建模、Agentic SFT（行为克隆）与 MGPO（多粒度策略优化）、内部反思动作、工具调用、逻辑校验器与难度评估器等技术。

**📊 数据集**

使用了约 10k–11k 条生成轨迹以及多领域基准数据集，包括 AIME 2024/2025、HMMT、AMO‑Bench、LiveCodeBench v5/v6、MMLU‑Redux/Pro、GPQA、SuperGPQA、OlympicArena 等。

**📈 对比分析**

通过与传统合成方法、人类注释、代理自博弈和前沿大模型对比，4B 解决器在 AIME 2025 及其他基准上平均提升 4–6%，30B 解决器在 AIME 25 达到 91.6%，超越大多数开源和商用模型。

**⚠️ 局限性**

局限性包括：对高质量验证器与难度评估器的依赖；技能库质量与覆盖范围直接影响生成质量；对极端复杂或非数学领域的泛化尚需进一步验证；小模型在 RL 奖励稀疏性方面仍受制约。

---

## 464. SCASRec: A Self-Correcting and Auto-Stopping Model for Generative Route List Recommendation

**arXiv ID:** 2602.03324 | [PDF](https://arxiv.org/pdf/2602.03324v1)

**作者:** Chao Chen `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**通讯引用:** 5389 | [OpenAlex ID](https://openalex.org/A5101512474)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种统一的生成式路由推荐模型 SCASRec，集成细排、再排与冗余消除；

**💡 创新点**

创新点是引入步进纠错奖励（SCR）提供列表级监督，并使用可学习的结束标记（EOR）实现自适应停止；

**🔧 技术方法**

采用 encoder‑decoder 自回归生成框架、注意力机制、多场景自注意力、SCR 奖励加权交叉熵以及噪声感知 EOR 调参技术；

**📊 数据集**

使用了自建的大规模开放路由推荐数据集（约 50 万查询、600 万候选路径）以及公开的 MSDR 数据集；

**📈 对比分析**

在离线指标（HR@K、LCR@K、MRR）和在线 A/B 测试中，与 MMR、DPP、DNN、PRM、Seq2Slate、NAR4Rec 等基线相比，SCASRec 在 HR@1、HR@5、LCR 以及 MRR 上均取得领先，并在线提升多样性、降低冗余；

**⚠️ 局限性**

目前仍面临多模态输入、交互式推荐的扩展需求，模型对噪声估计和 EOR 超参数较为敏感，且主要聚焦单一路由场景。

---

## 465. Vigemers: on the number of $k$-mers sharing the same XOR-based minimizer

**arXiv ID:** 2602.03337 | [PDF](https://arxiv.org/pdf/2602.03337v1)

**作者:** Florian Ingels `[一作]` (University of Lille), Mikaël Salson `[通讯]` (University of Lille)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对 DNA/RNA 序列中固定长度 k 的子串（k-mers）使用 XOR 关键字 γ 定义的最小子串（vigemin）进行计数，并给出计算函数 π_k^γ(w) 的通用动态规划算法；

**💡 创新点**

将原来仅适用于词典序（lexicographic）最小化的计数方法扩展到整个 |Σ|^m 种 XOR 关键字产生的可实现的全局最小化顺序，构造了自相关矩阵、专用字母表等工具；

**🔧 技术方法**

主要技术包括：自相关矩阵（generalized autocorrelation vector）、专用字母表、前缀-后缀（antemer/postmer）划分、递推公式与动态规划；时间复杂度 O(k m²)，空间 O(k m)；

**📊 数据集**

实验使用了 DNA 字母表 Σ={A,C,G,T}，取 m=10、k=31 的所有 k-mers，分别评估了词典序、逆词典序、交替序列及三种随机键 γ 的 π_k^γ(w) 分布；

**📈 对比分析**

通过对不同键产生的桶大小分布进行可视化与排序，比较了其平衡性；结果表明不同键虽然在峰值位置和二阶行为上有差异，但总体分布相似，未发现单一键显著更优；

**⚠️ 局限性**

局限性包括：对大规模键集或大量 w 计算仍昂贵；仅适用于基于 XOR 的顺序，难以直接推广到更一般的 g 函数；对真实生物数据的实证验证不足；算法对 m 较大时仍可能产生计算瓶颈。

---

## 466. GuardReasoner-Omni: A Reasoning-based Multi-modal Guardrail for Text, Image, and Video

**arXiv ID:** 2602.03328 | [PDF](https://arxiv.org/pdf/2602.03328v1)

**作者:** Zhenhao Zhu `[一作]` (Tsinghua University), Jiaheng Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 14555 | [OpenAlex ID](https://openalex.org/A5032474012)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GuardReasoner‑Omni，一种能够在文本、图像和视频三种模态上统一进行安全评估与过滤的多模态安全防护模型。

**💡 创新点**

创新点包括：① 使用链式推理（CoT）生成可解释的安全判断轨迹；② 采用两阶段训练策略（先 SFT 再 GRPO 强化学习），并在 GRPO 中引入硬样本挖掘与错误驱动探索奖励，显著提升对动态视频威胁的检测与推理深度；③ 在训练中整合多来源多模态数据，突破单一模态局限。

**🔧 技术方法**

技术手段主要有：教师模型 Qwen3‑VL 进行 CoT 蒸馏；SFT 微调以学习推理过程；GRPO 强化学习框架结合硬样本挖掘与探索奖励；使用标记化输出格式和多模态输入编码器；对模型进行参数规模扩展（2B/4B）。

**📊 数据集**

使用了 148K 样本的 GuardReasoner‑OmniTrain 数据集，涵盖文本、图像、文本-图像、纯视频和文本-视频；视频样本来自 UCF‑Crime、XD‑Violence、SafeWatch‑Bench、VHD、Video‑ChatGPT 等；文本-视频样本来自 Video‑ChatGPT 与 Video‑SafetyBench。

**📈 对比分析**

通过在 20 个基准（13 个提示、7 个回复）上计算 F1 分数进行评估；GuardReasoner‑Omni 2B 在整体任务上达 83.84% F1，视频任务 93.39% F1，文本-视频任务 93.96% F1，显著优于现有 LLM、VLM 及 VAD 对手；4B 模型在复杂场景下进一步提升性能。

**⚠️ 局限性**

局限性包括：① 需要大规模预训练模型和昂贵的两阶段训练成本；② 对极少数边缘案例的鲁棒性仍待验证；③ 仍可能出现误报或假阴性，需进一步提升安全可靠性。

---

## 467. Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization

**arXiv ID:** 2602.03327 | [PDF](https://arxiv.org/pdf/2602.03327v1)

**作者:** Manuel Hofer `[一作]` (Graz University of Technology), Thomas Köhler `[通讯]` (Graz University of Technology)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5111285597)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于3D Gaussian splatting的稀疏视角新视图合成方法，利用π^3网络进行无SfM的稠密点云初始化，结合置信度感知的Pearson深度损失、正则化的法向监督与深度重投影伪视角提升几何一致性与视图连贯性。

**💡 创新点**

关键创新在于：①无传统SfM的稠密初始化方案；②置信度加权的Pearson深度损失缓解深度不确定性；③掩码正则化的法向监督消除网格伪影；④通过深度重投影生成伪视角实现额外多视角监督；⑤在PGSR基础上去除多视角裁剪与分裂策略，适配稀疏视角。

**🔧 技术方法**

技术包括：3D Gaussian splatting (PGSR)、π^3点云与相机参数预测网络、置信度感知Pearson深度损失、法向监督L1损失、深度重投影伪视角生成、SSIM与L1伪视角监督。

**📊 数据集**

在Tanks and Temples、MipNeRF360、LLFF、DTU四个公开数据集上进行评估，分别采用3视角和12视角设置。

**📈 对比分析**

与Intern-GS、InstantSplat、SparseGS、DNGaussian、FSGS、3DGS等先前方法对比，实验显示在稀疏视角下实现PSNR、SSIM提升，LPIPS下降，显著减少浮点体和几何误差，整体性能达到或接近SOTA。

**⚠️ 局限性**

局限性：π^3网络在处理大量输入视角时显著占用GPU显存，限制在消费级硬件上的可扩展性；在部分场景（如叶片、低纹理区域）深度估计不准确，导致重建细节欠缺；未来需结合相机姿态联合优化或生成式先验提升对遮挡/稀疏区域的鲁棒性。

---

## 468. To Search or Not to Search: Aligning the Decision Boundary of Deep Search Agents via Causal Intervention

**arXiv ID:** 2602.03304 | [PDF](https://arxiv.org/pdf/2602.03304v1)

**作者:** Wenlin Zhang `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 5962 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了深度搜索代理的决策边界问题，并设计了一套基于因果干预的诊断方法和决策边界对齐（DAS）技术，帮助代理更合理地判断何时停止搜索并给出答案。

**💡 创新点**

创新点包括：
1) 用因果干预方式从已完成轨迹回溯判断代理的知识是否足够，从而诊断过度搜索和不足搜索两类错误；
2) 将诊断得到的因果偏好（优劣轨迹对）构造成偏好数据集，并通过直接偏好优化（DPO）对代理策略进行对齐；
3) 在多数据集上系统验证决策边界错误普遍存在，且DAS能同时提升准确率和搜索效率。

**🔧 技术方法**

技术手段包括：
- 大语言模型与自主管理的搜索代理框架；
- 因果干预（do-operator）对代理决策进行反事实推断；
- 偏好数据集构造与直接偏好优化（DPO）；
- 基于检索工具的多轮搜索与答案生成；
- 传统RL与DPO相结合的微调流程。

**📊 数据集**

使用的公开数据集有：Natural Questions（NQ）、HotpotQA（HotpotQA）以及2WikiMultiHopQA（2WikiMultiHopQA），用于评估单跳与多跳推理的准确率与搜索行为。

**📈 对比分析**

对比方法：将DAS与基线Search‑R1（RL训练）和Search‑O1（闭源大模型+检索）进行对比，指标包括 Exact Match（EM）、总推理时间、平均搜索查询数（ASQ）、过度搜索率（OSR）与不足搜索率（USR）。实验结果显示：
- DAS在EM上至少提升3–5个百分点；
- ASQ 下降 30–50%；
- 总推理时间缩短 20–40%；
- OSR 与 USR 均显著下降，证明决策边界得到有效校准。

**⚠️ 局限性**

局限性：
- 诊断与对齐依赖手工设计的干预提示，缺乏自动化生成；
- 仅在文本检索任务中验证，尚未测试多模态或真实网络搜索环境；
- 对知识边界与决策边界差距的理论解释仍不充分，可能隐藏更深层次的自我感知问题。

---

## 469. POP: Prefill-Only Pruning for Efficient Large Model Inference

**arXiv ID:** 2602.03295 | [PDF](https://arxiv.org/pdf/2602.03295v1)

**作者:** Junhui He `[一作]` (Wuhan University), Qingan Li `[通讯]` (Wuhan University)

**通讯引用:** 741 | [OpenAlex ID](https://openalex.org/A5103211926)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种仅在预填阶段剪枝深层的策略（Prefill-Only Pruning, POP），以加速大型语言和视觉语言模型的推理。

**💡 创新点**

通过虚拟门机制揭示了预填与解码阶段对层的敏感度不对称，发现深层对生成敏感但对上下文编码冗余，从而实现阶段感知剪枝。

**🔧 技术方法**

使用虚拟门梯度估计层重要性、独立KV投影、边界处理以及结构化层剪枝。

**📊 数据集**

使用WizardLM‑V2‑196K（文本）和LLAVA‑Instruct‑150K（多模态）作为校准数据集，并在Llama‑3.1‑8B、Qwen3‑VL‑8B、Gemma‑3‑12B‑It等模型上评估。

**📈 对比分析**

与Unstructured（Wanda）和Structured（SliceGPT、ShortGPT）方法比较，POP在保持近乎完整准确率的前提下，在预填阶段实现最高1.37×的速度提升，尤其在长上下文与高分辨率多模态任务中表现最佳。

**⚠️ 局限性**

因预填阶段剪枝后仍需加载完整模型进行解码，未减小峰值显存需求；且目前实现基于单一推理管线，需在分布式系统中进一步工程化。

---

## 470. Universal Approximation of Continuous Functionals on Compact Subsets via Linear Measurements and Scalar Nonlinearities

**arXiv ID:** 2602.03290 | [PDF](https://arxiv.org/pdf/2602.03290v1)

**作者:** Andrey Krylov `[一作]` (Lomonosov Moscow State University), Maksim Penkin `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5035765796)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了在Hilbert空间乘积的紧集上，连续泛函的通用逼近，证明可通过有限个线性测量与标量非线性组合来逼近。

**💡 创新点**

证明了此类结构的通用逼近性质，并给出了Banach空间输出的有限秩扩展，填补了先前研究的空白。

**🔧 技术方法**

使用了紧致性、Heine–Cantor、正交投影、分区函数以及复数版Stone–Weierstrass定理等经典分析工具。

**📊 数据集**

未涉及具体数据集，属于理论证明。

**📈 对比分析**

无实验比较，纯理论分析。

**⚠️ 局限性**

局限在于缺乏量化误差率与对数据结构约束的讨论，且结果仅在紧集上适用，难以直接推广到无限维非紧集。

---

## 471. Rejecting Arguments Based on Doubt in Structured Bipolar Argumentation

**arXiv ID:** 2602.03286 | [PDF](https://arxiv.org/pdf/2602.03286v1)

**作者:** Michael A. Müller `[一作]` (Université de Fribourg), Bruno Yun `[通讯]` (Univ Lyon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了结构化双极论证框架（SBAF）并给出了支持与攻击关系下的新语义，兼顾论证与句子层面的可接受性；

**💡 创新点**

1）允许被辩护的论证不必全部接受，体现“怀疑”概念；2）从句子层面推导论证集合，提供论证与语言视角的对应关系；3）将d‑admissibility与强连贯性联系，构建更一般的双极语义；

**🔧 技术方法**

理论构建与形式化定义、语义证明、概念关系证明；

**📊 数据集**

无实验数据集，全部为形式化理论与示例；

**📈 对比分析**

无实验比较，理论上通过与现有抽象/双极论证语义的关系说明其优劣；

**⚠️ 局限性**

1）在非饱和框架下论证与语言视角可能不一致；2）对“怀疑”机制的具体实现和多代理情景的可扩展性未给出；3）缺乏经验验证与应用案例。

---

## 472. MeetBench-XL: Calibrated Multi-Dimensional Evaluation and Learned Dual-Policy Agents for Real-Time Meetings

**arXiv ID:** 2602.03285 | [PDF](https://arxiv.org/pdf/2602.03285v1)

**作者:** Yuelin Hu `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 82659 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于企业真实会议语料的双语多模态数据集MeetAll、与之配套的多维度评估基准MeetBench-XL，以及通过离线强化学习联合优化路由与工具调用的双策略代理MeetMaster-XL；

**💡 创新点**

创新点在于：1）利用企业访谈驱动的注入协议，生成兼具真实性与多维任务复杂度的会议问题；2）开发了人类校准的五维评估框架，实现与专家判断高度一致；3）提出轻量级学习型路由器和工具策略，显著提升了延迟-质量-成本平衡，且可在单 RTX 4090 上部署；

**🔧 技术方法**

技术包括多模态 ASR 与 TTS、知识库检索、跨会聚合、LLM‑as‑Judge、离线强化学习（CQL）、多任务分类与工具调用决策；

**📊 数据集**

使用了来自 AISHELL‑4（中文）和 CHiME‑6（英文）的真实会议音频，合计 140 小时，并在此基础上注入 1,180 个经过专家验证的企业问题；

**📈 对比分析**

与 GPT‑4o、Claude 3.5 Sonnet、Gemini 1.5 Pro 等商用 API 以及开源 LLM 进行比较，MeetMaster‑XL 在整体评估分数上达 6.59（相比 GPT‑4o 的 6.93，差距约 0.34 分），在简单查询上平均延迟低 30 %，复杂查询质量提升 12.4 %，且部署成本显著低于云 API；

**⚠️ 局限性**

局限性包括：数据仅覆盖中文和英文两种语言；工具库尚未覆盖所有企业工作流（如日历、任务管理）；在极端噪声或说话者重叠情况下仍存在误检与检索失败，需要进一步鲁棒性提升。

---

## 473. Global Geometry Is Not Enough for Vision Representations

**arXiv ID:** 2602.03282 | [PDF](https://arxiv.org/pdf/2602.03282v1)

**作者:** Jiwan Chung `[一作]` (Yonsei University), Seon Joo Kim `[通讯]` (Yonsei University)

**通讯引用:** 4610 | [OpenAlex ID](https://openalex.org/A5103036411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了视觉表示学习中全局嵌入几何是否能预测组合能力，发现传统几何指标与组合绑定任务无显著关联；通过分析输入-输出雅可比矩阵的有效秩（JER）证明了功能灵敏度是判断组合结构的有效诊断工具；进一步展示了不同训练目标对JER的影响，说明目标设计决定了本地敏感度的分布；

**💡 创新点**

创新点在于提出并验证了基于雅可比矩阵有效秩的功能灵敏度指标，揭示全局几何与组合结构的分离，并系统分析不同预训练目标对功能敏感度的影响；

**🔧 技术方法**

使用雅可比矩阵有效秩（Jacobian Effective Rank）计算、随机噪声输入、自动微分求雅可比向量乘、统计几何指标（G.PR、G.Iso、L.Iso）等技术；

**📊 数据集**

在21个预训练视觉编码器（包含ResNet、ViT系列）上评估，使用ImageNet-1k验证集和合成属性绑定任务（Attribute Binding）进行实验；

**📈 对比分析**

将几何指标与属性绑定准确率、JER与绑定准确率的Pearson相关系数进行对比，发现几何指标几乎无相关性，而JER与绑定性能呈显著正相关（r≈0.65，p<0.01），表明功能灵敏度是更可靠的评估维度；

**⚠️ 局限性**

局限在于仅使用合成数据评估组合能力，未覆盖更复杂的真实场景；雅可比有效秩的估计依赖随机噪声输入，可能未能捕捉真实图像扰动；

---

## 474. On Complete Categorical Semantics for Effect Handlers

**arXiv ID:** 2602.03275 | [PDF](https://arxiv.org/pdf/2602.03275v1)

**作者:** Satoshi Kura `[一作]` (Waseda University), Satoshi Kura `[通讯]` (Waseda University)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5090422035)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构造了一个新的含效用理论的效处理器语言，给出其语义和等式体系，并证明了其语义的正确性与完备性。

**💡 创新点**

创新点在于将代数效用理论中的操作间等式融入效处理器语言，并通过子范畴与预射构造实现效处理器满足等式的语义解释；同时提供了基于CPS变换的正确性证明。

**🔧 技术方法**

采用了范畴理论（强单子、Kleisli指数、Eilenberg–Moore代数、子范畴与等化器）以及代数效用理论的自由模型。

**📊 数据集**

无特定数据集，研究为形式化理论与范畴模型。

**📈 对比分析**

对方法的比较主要是通过构造项模型实现完备性证明，未涉及运行时性能评估。

**⚠️ 局限性**

限制在于需要范畴具备充分完整性（如有子范畴的可表示性与等化器存在性），否则需使用预射范畴；对一般范畴的适用性与实现复杂度未详述。

---

## 475. Unveiling Covert Toxicity in Multimodal Data via Toxicity Association Graphs: A Graph-Based Metric and Interpretable Detection Framework

**arXiv ID:** 2602.03268 | [PDF](https://arxiv.org/pdf/2602.03268v1)

**作者:** Guanzong Wu `[一作]` (Chinese University of Hong Kong), Baoyuan Wu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6802 | [OpenAlex ID](https://openalex.org/A5068027800)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于毒性关联图（TAG）的多模态毒性检测框架，能够识别隐蔽毒性并给出解释路径。

**💡 创新点**

创新点包括：1) 设计TAG系统系统建模视觉-文本之间的语义关联；2) 引入多模态毒性隐蔽度量（MTC）来量化隐藏毒性；3) 构建首个高隐蔽毒性数据集CTD并做基准评估。

**🔧 技术方法**

技术手段包括：大规模多模态语言模型（如CLIP、Gemma等）提取实体关联、构造关联树与双向图、计算联合转移概率并得到MTC评分；利用LLM生成解释。

**📊 数据集**

使用自研的Covert Toxic Dataset（CTD）作为主实验集，并在Hateful Memes、VLSBench、MMIT等公开数据集上进行泛化评估。

**📈 对比分析**

与传统“裸”输入（Vanilla）模型对比，使用TA-CTD后F₂得分提升显著，例如Gemma3从0.31升至0.82，Llama 3.2 Vision达到0.97，整体表现优于所有基线。

**⚠️ 局限性**

局限性在于：1) 对TAG的层数和子节点阈值需手工调参，深度不足会导致高隐蔽样本检出率下降；2) 在纯正面或低隐蔽场景下可能出现误报，影响整体准确率。

---

## 476. From Inexact Gradients to Byzantine Robustness: Acceleration and Optimization under Similarity

**arXiv ID:** 2602.03329 | [PDF](https://arxiv.org/pdf/2602.03329v1)

**作者:** Renaud Gaucher `[一作]` (Centre de Mathématiques Appliquées, École polytechnique, Institut Polytechnique de Paris), Hadrien Hendrikx `[通讯]` (Centre Inria de l'Univ. Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出将拜占庭鲁棒分布式优化问题抽象为不完全梯度算子问题，并基于此设计了两种加速求解方法：一种是基于 Nesterov 的加速梯度算法，另一种是利用代理损失的相似性进行预条件化的近似梯度下降（PIGS），从而显著降低通信复杂度。

**💡 创新点**

创新点在于：①将拜占庭鲁棒性转化为通用的 (ζ²,α) 不完全梯度算子框架，解除了对具体鲁棒聚合方法的耦合；②首次在拜占庭场景下实现加速收敛，并证明其理论收敛速率；③引入相似性假设，利用代理模型实现更大的步长，从而进一步提升收敛速度。

**🔧 技术方法**

使用技术包括：鲁棒聚合规则（几何中值、坐标裁剪等）的 (f,ν)-鲁棒性表征；不完全梯度算子理论和 (δ,L,μ)-oracle 的关系；加速梯度下降（Nesterov）与不完全梯度的结合；预条件化近似梯度下降（PIGS）中的二阶相似性假设与近似原点子问题求解；以及 L-BFGS 等近似优化器。

**📊 数据集**

实验采用 MNIST 数据集的逻辑回归任务，利用 Dirichlet 分区（β=1 及 β=5）模拟不同程度的数据异质性，此外还使用随机 i.i.d. 分区作为对照。

**📈 对比分析**

与传统的分布式梯度下降（D‑GD）和分布式加速梯度（D‑NAG）比较，PIGS 在同等精度下仅需 5–10 次通信轮次即可突破拜占庭极限误差，而 D‑NAG 需 40+ 次，D‑GD 需 100+ 次；三种方法最终收敛到相同的渐近误差，证明了方法的鲁棒性与加速优势。

**⚠️ 局限性**

局限性包括：加速方法在满足 (ζ²,α) 不完全梯度假设时会产生 √κ 的渐近误差折衷；对更高维或更复杂模型的适用性未充分验证；实验中拜占庭攻击多为简单攻击，未涵盖可能破坏加速稳定性的更强攻击；以及在随机梯度或噪声环境下的理论最优收敛率仍未知。

---

## 477. MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning

**arXiv ID:** 2602.03320 | [PDF](https://arxiv.org/pdf/2602.03320v1)

**作者:** Shengyuan Liu `[一作]` (Chinese University of Hong Kong), Yixuan Yuan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9688 | [OpenAlex ID](https://openalex.org/A5073968803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出MedSAM-Agent框架，将医学图像分割从静态分类转为自主多轮决策过程。

**💡 创新点**

通过混合提示生成专家轨迹、两阶段SFT+RL训练和过程感知奖励，实现自适应盒子+点迭代并兼容多种交互工具。

**🔧 技术方法**

结合多模态大语言模型（Qwen3-VL-8B）、Segment Anything交互工具（SAM2、MedSAM2、IMISNet）、混合提示、SFT+RL两阶段训练、GRPO强化学习及多维过程奖励。

**📊 数据集**

在21个公开医学数据集上评估，涵盖CT、MRI、X光、超声、眼底和内镜六大模态。

**📈 对比分析**

与SAM、MedSAM、IMISNet、LISA、UniBioMed等基线对比，MedSAM-Agent在所有模态上均达到最高Dice/Iou，平均提升约5–10%，并在多轮交互中超越单轮盒子/点基线，显示出更高的精度与交互效率。

**⚠️ 局限性**

仍受限于工具精度、训练样本需专家轨迹、计算成本较高，以及对极少量或形态极其不规则的数据集可能表现欠佳。

---

## 478. Invisible Clean-Label Backdoor Attacks for Generative Data Augmentation

**arXiv ID:** 2602.03316 | [PDF](https://arxiv.org/pdf/2602.03316v1)

**作者:** Ting Xiang `[一作]` (Hunan University), Zhuo Tang `[通讯]` (Hunan University)

**通讯引用:** 4759 | [OpenAlex ID](https://openalex.org/A5024356456)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对生成式数据增强（GDA）环境下的干净标签后门攻击，提出一种在潜在空间注入隐形触发器的攻击方法InvLBA，并提供理论保证。

**💡 创新点**

创新点在于：① 通过理论分析揭示像素级触发器在生成图像中导致攻击成功率下降的根源——较高的对抗扰动敏感度；② 设计在Stable Diffusion潜在空间的单步噪声预测触发器，显著降低对抗扰动敏感度，实现更高的攻击成功率；③ 证明该方法在干净准确率和攻击成功率上的泛化性。

**🔧 技术方法**

技术主要包括：Stable Diffusion潜在空间变换、单步噪声预测（DDIM/CFG）、逆向优化训练触发器、Rademacher复杂度与泛化误差理论分析、数据级与模型级防御评估。

**📊 数据集**

使用的数据集包括：六个小规模GDA基准（CIFAR‑10‑S、ImageNet‑10‑S、CelebA‑S、Caltech‑101、Cars、Pets）以及三大后门基准（CIFAR‑10‑S、ImageNet‑10‑S、CelebA‑S）。

**📈 对比分析**

与LC、Refool、Sleeper Agent、Narcissus、COMBAT及改写的BadNets等干净标签后门基线对比，InvLBA在无GDA时提升约30% ASR，在GDA场景下平均提升约46% ASR，且保持几乎无CAD；在数据级防御（DATAELIXIR）和模型级防御（Neural Cleanse）下仍保持50%以上ASR，显示出强鲁棒性。

**⚠️ 局限性**

局限性包括：在类别数显著增多时（如Cars的196类）ASR下降；触发器的实现依赖于Stable Diffusion模型的可逆噪声预测，可能受限于不同生成模型；尚无专门针对生成数据的有效防御策略。

---

## 479. RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization

**arXiv ID:** 2602.03310 | [PDF](https://arxiv.org/pdf/2602.03310v1)

**作者:** Songming Liu `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 67256 | [OpenAlex ID](https://openalex.org/A5115666530)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为RDT2的机器人基础模型，能够零样本跨平台部署并执行开放词汇任务。

**💡 创新点**

创新点在于：1）设计了一个可实现10,000小时以上的数据收集的可重构Universal Manipulation Interface (UMI)；2）提出三阶段训练策略，将离散动作编码（RVQ）与连续动作生成（flow-matching扩散专家）以及单步蒸馏相结合；3）实现了在不同对象、场景、指令和机器人平台上零样本的组合泛化。

**🔧 技术方法**

使用的技术包括：Qwen2.5-VL 7B视觉语言模型、残差向量量化（RVQ）离散化、流匹配损失训练的扩散动作专家、单步蒸馏、分布式并行训练和高精度红外追踪。

**📊 数据集**

使用的公开数据集是基于改进的UMI收集的约10,000小时、100+家庭的多样化操控演示数据，以及少量视觉语言对。

**📈 对比分析**

与现有基线（π_0-FAST、π_0.5等）相比，RDT2在多项挑战性任务（如布料折叠、桌子摆放、快速按键、乒乓球击球）中表现更佳，尤其在泛化任务中成功率提升至两倍以上；在实时推理速度上也比更小模型快数倍。

**⚠️ 局限性**

局限性包括：1）对硬件的依赖仍存在，虽然UMI具备可移植性但在极端环境下可能受限；2）零样本泛化虽显著提升，但在某些极端场景或复杂动态任务仍需微调；3）模型训练需要大规模算力和长时间的计算；4）对隐私与安全的合规性仍需进一步完善。

---

## 480. Entropy-Gated Selective Policy Optimization:Token-Level Gradient Allocation for Hybrid Training of Large Language Models

**arXiv ID:** 2602.03309 | [PDF](https://arxiv.org/pdf/2602.03309v1)

**作者:** Yuelin Hu `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 82659 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Entropy-Gated Selective Policy Optimization (EG-SPO)，构建三阶段混合监督微调（SFT）与强化学习（RL）的训练框架，在 token 级别对梯度进行熵门控分配。

**💡 创新点**

创新点在于使用预测熵模块实现 token 级梯度分配，并在低熵 token 上保留优势函数以防止错误强化，同时兼顾探索与稳定性。

**🔧 技术方法**

采用监督微调、PPO 强化学习、GRPO 风格优势归一、熵门控权重 ϕ(p)=p(1-p) 等技术。

**📊 数据集**

在 AIME（2024-2025）、AMC、MATH 等数学推理基准上进行评测。

**📈 对比分析**

与 CHORD-ϕ 等现有混合训练基线对比，分别在 AIME、AMC、MATH 上提升约 +3.8%、+2.3% 和 +2.9%，仅增加 3.4% 计算开销。

**⚠️ 局限性**

限制包括需手工设定熵阈值 ρ、仅验证于数学推理任务、缺乏自动阈值调节与跨领域泛化能力。

---

## 481. Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval

**arXiv ID:** 2602.03306 | [PDF](https://arxiv.org/pdf/2602.03306v1)

**作者:** Zhanyu Wu `[一作]` (Beihang University), Zhijie Nie `[通讯]` (Beihang University)

**通讯引用:** 130 | [OpenAlex ID](https://openalex.org/A5059807664)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于监督标签的查询自适应维度选择框架，在检索时只保留对当前查询最重要的维度进行相似度计算。

**💡 创新点**

创新点在于通过构造oracle维度重要性分布并训练一个仅依赖查询嵌入的预测器，避免了噪声较大的伪相关反馈，且实现了真正的查询感知维度掩码。

**🔧 技术方法**

核心技术包括：高维稠密检索、使用多级监督标签构造正负加权质心、基于温度softmax的维度重要性分布、单层全连接预测器及KL散度训练、Top‑k维度掩码与cosine相似度计算。

**📊 数据集**

实验使用了SciFact、MS MARCO和NFCorpus三大Benchmarks，检索模型涵盖Qwen-Embedding系列、OpenAI、LLM2Vec及GritLM等多种稠密编码器。

**📈 对比分析**

与全维度基线、固定前缀、基于模糊正则的Norm、训练好的搜索适配器、以及基于伪相关反馈的DIME/Eclipse等方法对比，本文方法在保留约20‑40%维度的情况下均能达到或超过全维度效果，并在多模型多数据集上实现了显著的NDCG@10提升。

**⚠️ 局限性**

主要局限包括：仍需高质量监督标签（尤其在小规模领域数据上），以及仅在查询层面进行维度选择，无法提升底层编码器本身的检索性能。

---

## 482. Periodic Regularized Q-Learning

**arXiv ID:** 2602.03301 | [PDF](https://arxiv.org/pdf/2602.03301v1)

**作者:** Hyukjun Yang `[一作]` (Korea Advanced Institute of Science and Technology), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2154 | [OpenAlex ID](https://openalex.org/A5100654316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了周期性正则化的 Q 学习算法，改进了传统 Q 学习的收敛速度和稳定性。

**💡 创新点**

通过引入周期性正则化投影算子和 AVI_reg 框架，首次在有限样本下提供了收敛性证明。

**🔧 技术方法**

结合了强化学习中的 Q 学习、近似动态规划、正则化技术以及投影算子等方法。

**📊 数据集**

在 OpenAI Gym 的经典离散控制环境（如 CartPole、MountainCar、Acrobot 等）上进行实验。

**📈 对比分析**

与标准 Q 学习、DQN 等方法对比，实验表明周期性正则化 Q 学习在累计奖励、收敛速度和样本效率上都有明显提升。

**⚠️ 局限性**

需要手动调节正则化周期和强度，实验范围有限，未在大规模连续动作空间或复杂视觉环境中验证。

---

## 483. R1-SyntheticVL: Is Synthetic Data from Generative Models Ready for Multimodal Large Language Model?

**arXiv ID:** 2602.03300 | [PDF](https://arxiv.org/pdf/2602.03300v1)

**作者:** Jingyi Zhang `[一作]` (Hong Kong Polytechnic University), Jiaxing Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 69303 | [OpenAlex ID](https://openalex.org/A5100355322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过集体对抗数据合成（CADS）框架自动生成高质量、多样且具有挑战性的多模态数据，并基于此训练出 R1‑SyntheticVL 模型。

**💡 创新点**

创新点在于将多模态 LLM 作为协同生成与评判主体，形成生成–评判循环，并引入对抗上下文优化机制以动态提升数据难度与价值。

**🔧 技术方法**

主要技术包括 CADS 的集体生成与判别算法、Nano Banana Pro 视觉生成、对抗上下文优化、以及基于 GRPO 的强化学习微调。

**📊 数据集**

使用 CADS 生成的 MMSynthetic‑20K（20K 样本）数据集；与公开真实数据集（MM‑Eureka、MathVista 等）进行对比实验。

**📈 对比分析**

在 MathVista、MMVU、MathVision、MMMU、ThinkLite‑VL‑7B 等多项基准上与现有开源与闭源模型对比，R1‑SyntheticVL 在大多数指标上实现领先，尤其在推理类任务中提升显著。

**⚠️ 局限性**

局限性包括对多模态 LLM 与生成模型质量的高度依赖、生成成本与计算开销大、对抗优化难以精细控制数据多样性与难度平衡，以及在极端专业领域的覆盖仍有限。

---

## 484. An Algorithm for Monitoring Edge-geodetic Sets in Chordal Graphs

**arXiv ID:** 2602.03288 | [PDF](https://arxiv.org/pdf/2602.03288v1)

**作者:** Nacim Oijid `[一作]` (Umeå University), Clara Marcille `[通讯]` (Univ Lyon)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明所有弦图都是巨最小的（meg-minimal），并给出一个多项式时间算法，能在弦图上计算最小监测边几何集（meg-set）

**💡 创新点**

首次将弦图加入已知的巨最小图类中，揭示了弦图中必需顶点集合(Mand(G))即为唯一最小meg-set，从而完成了对前人提出的开放猜想的解答

**🔧 技术方法**

利用弦图的性质（如存在简单顶点、无诱导环）、必需顶点与支撑顶点的定义、路径计数（2‑路径数表）以及逐顶点递推的集合更新算法，综合运用了图论结构分析与动态维护技术

**📊 数据集**

论文中未使用具体数据集，全部结果均为理论证明与算法复杂度分析

**📈 对比分析**

给出的算法时间复杂度为 O(|V|(|V|+Δ²))，对弦图可实现最小meg-set的多项式求解；论文未与实验或其他算法进行性能对比，仅在理论层面展示了优越性

**⚠️ 局限性**

主要局限在于：①仅针对弦图；②缺乏实验验证与对比；③未探讨如何将方法扩展到更广泛的图类或非巨最小图的多项式解法

---

## 485. Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks

**arXiv ID:** 2602.03284 | [PDF](https://arxiv.org/pdf/2602.03284v1)

**作者:** Yi Yu `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15480 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种只改变事件时间戳而不改变计数和幅度的Spike‑Retiming攻击，针对事件驱动的脉冲神经网络进行时序扰动。

**💡 创新点**

创新点在于提出容量为1的统一时延攻击模型并引入三种预算（ℬ∞、ℬ1、ℬ0），同时设计了Projected‑in‑the‑Loop（PIL）优化框架，将可微软重定时与严格投影结合，实现可扩展的时序攻击。

**🔧 技术方法**

采用可微软重定时器、严格投影、容量正则化、预算惩罚，并在SNN上使用STBP、surrogate梯度与PIL‑PGD进行优化；同时利用整数与二值事件网格。

**📊 数据集**

在CIFAR10‑DVS、DVS‑Gesture、N‑MNIST等事件数据集上评估，并分别使用二值和整数网格编码。

**📈 对比分析**

与传统事件级攻击（如SpikeFool、PDSGSDA）以及对抗训练模型对比，攻击在大多数预算下成功率超过90%，整数网格表现更稳健，表明时序扰动是一个强大且隐蔽的攻击向量。

**⚠️ 局限性**

限制包括对抗训练仅能略增鲁棒且显著降低准确率；针对特定目标类别攻击效果不佳；攻击在不同网络架构间的转移性有限；对极端预算下的性能仍需进一步提升。

---

## 486. MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research

**arXiv ID:** 2602.03318 | [PDF](https://arxiv.org/pdf/2602.03318v1)

**作者:** Yifan Shi `[一作]` (Xi'an Jiaotong University), Jianyong Sun `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 15773 | [OpenAlex ID](https://openalex.org/A5100367671)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个无训练的多代理框架 MIRROR，用于把自然语言的运筹学问题自动转换为数学模型和可执行求解器代码。

**💡 创新点**

创新点在于结合迭代自适应修订 (IAR) 与层级检索增强生成 (HRAG)，并通过双重记忆池实现执行驱动的自我纠错和知识迁移。

**🔧 技术方法**

技术主要包括大语言模型、多代理协作、层级检索（两阶段过滤与重排序）、迭代自适应修订、双重内存（本地/全局）以及外部求解器 Gurobi。

**📊 数据集**

使用的数据集包括 NL4Opt、Mamo‑EasyLP、Mamo‑ComplexLP、IndustryOR、ComplexOR，并利用 602 条高质量样本构建检索库。

**📈 对比分析**

与传统提示、学习驱动模型和其它多代理框架相比，MIRROR 在所有 5 个基准上均为 SOTA，宏平均准确率 71.88%，在复杂任务上提升约 9.9%/5.6%，并在小模型 qwen3‑30B 上提升 12.15%。

**⚠️ 局限性**

局限性包括：对极端模糊或多语言描述的鲁棒性有限；依赖外部求解器导致规模和求解速度受限；以及缺乏完整的人类审查机制，可能导致误判或不可解释的错误。

---

## 487. Towards Distillation-Resistant Large Language Models: An Information-Theoretic Perspective

**arXiv ID:** 2602.03396 | [PDF](https://arxiv.org/pdf/2602.03396v1)

**作者:** Hao Fang `[一作]` (Tsinghua University), Ke Xu `[通讯]` (Tsinghua University)

**通讯引用:** 11593 | [OpenAlex ID](https://openalex.org/A5100665814)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种通过信息论视角对大语言模型输出进行后处理的防御方法，阻止对教师模型的logit‑based知识蒸馏；

**💡 创新点**

创新点在于首次将条件互信息（CMI）作为防御目标，用低秩变换矩阵和梯度角度损失实现对教师logit的上下文信息压缩，从而兼顾模型效能与安全；

**🔧 技术方法**

技术主要包括：信息瓶颈（Information Bottleneck）与CMI理论、线性变换矩阵学习、梯度方向失配损失、低秩矩阵分解、LoRA初始化；

**📊 数据集**

实验使用的模型与数据集为：Qwen2.5‑7B、Llama‑3.1‑8B 以及对应小尺寸学生模型，评测数据集为 GSM8K、MMLU、MATH；

**📈 对比分析**

在对抗四种logit‑based KD 基线（KD、AlphaNet、MiniLLM、ABKD）时，本文方法在保持教师原始准确率的前提下，显著降低学生模型在各数据集上的蒸馏效果，提升了防御性能；

**⚠️ 局限性**

局限性包括：需要教师与学生使用相同分词器；防御强度与模型效能需要权衡，参数 λ 与矩阵秩 r 需手工调优；对极端大词表或高维模型的计算开销仍有提升空间。

---

## 488. From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning

**arXiv ID:** 2602.03390 | [PDF](https://arxiv.org/pdf/2602.03390v1)

**作者:** Hyun Seok Seong `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1470 | [OpenAlex ID](https://openalex.org/A5029469141)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种Synergistic Representation Learning (SRL)框架，通过互补的三元对比学习目标打破编码器与解码器之间的尖锐-模糊冲突，实现编码器与解码器的相互细化。

**💡 创新点**

创新点在于：①利用编码器的锐利注意力图作为解码器去模糊的指导；②利用解码器的空间一致性掩码对编码器特征进行去噪；③设计双向三元对比损失并在预热阶段加入槽位正则化，形成从恶性循环到良性循环的训练流程。

**🔧 技术方法**

技术主要包括：Slot Attention编码器、基于DINO‑v2特征的特征提取、两种三元对比学习损失（去模糊和去噪）、槽位正则化正则化项、分阶段训练框架以及传统的MSE重建损失。

**📊 数据集**

使用的公开数据集有：MOVi‑C、MOVi‑E（合成视频数据）和YouTube‑VIS 2021（真实视频数据）。

**📈 对比分析**

与SlotContrast、VideoSAUR、STEVE等方法对比，SRL在MOVi‑C上FG‑ARI提升5.5%、mBO提升8.8%；在YTVIS 2021上FG‑ARI提升18.5%、mBO提升8.2%，并在物体动力学预测任务中明显优于基线和SlotContrast。

**⚠️ 局限性**

局限性包括：①在高度约束的合成场景（如MOVi）仍不如VideoSAUR在mBO上表现；②对超参数（如正负样本数K、正则化系数λ等）敏感；③需要显式的时间上下文来提升语义聚类，单帧训练效果下降；④目前仅在合成和中等复杂度视频上验证，尚未在极其复杂或长时序视频中彻底检验。

---

## 489. Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL

**arXiv ID:** 2602.03389 | [PDF](https://arxiv.org/pdf/2602.03389v1)

**作者:** Jinwoo Choi `[一作]` (Seoul National University), Seung-Woo Seo `[通讯]` (Seoul National University)

**通讯引用:** 2764 | [OpenAlex ID](https://openalex.org/A5048311228)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Chain-of-Goals Hierarchical Policy（CoGHP），将离线目标条件强化学习中的层级决策改写为自回归的子目标序列生成，再输出最终原子动作。

**💡 创新点**

创新点在于：1）将层级决策统一为一次性模型的自回归序列建模，支持多步子目标生成；2）在序列建模中使用MLP‑Mixer骨干网络，并加入因果混合器以捕捉前一步子目标对后续预测的影响；3）通过共享价值函数实现端到端的优势加权训练，使梯度能够跨所有层级传播。

**🔧 技术方法**

核心技术包括：链式思考（Chain‑of‑Thought）框架、MLP‑Mixer（token‑mixing 与 channel‑mixing MLP）、因果混合器、优势加权回归（AWR）以及离线强化学习中的Implicit Q‑Learning（IQL）价值函数。

**📊 数据集**

使用 OGBench 基准集，包括 PointMaze、AntMaze（不同规模）以及 Cube、Scene 等操控任务，亦在可视化的像素版环境（visual‑antmaze、visual‑cube）中验证。

**📈 对比分析**

与六种主流离线目标强化学习方法（GCBC、GCIVL、GCIQL、QRL、CRL、HIQL）对比，CoGHP 在大多数长序列任务中取得显著更高的成功率（例如 PointMaze‑giant 79% 对比 HIQL 46%，Scene 78% 对比 HIQL 38%），验证了其在长时程离线控制中的优越性。

**⚠️ 局限性**

局限性包括：1）子目标仅采用编码的未来状态，缺乏更抽象的技能或语义表示；2）需要预先设定子目标数量，未实现动态自适应；3）在极简任务或不需要多步推理的场景下，复杂模型可能并不必要且训练稳定性受限。

---

## 490. An Approximate Ascent Approach To Prove Convergence of PPO

**arXiv ID:** 2602.03386 | [PDF](https://arxiv.org/pdf/2602.03386v1)

**作者:** Leif Doering `[一作]` (University of Mannheim), Simon Weissmann `[通讯]` (University of Mannheim)

**通讯引用:** 163 | [OpenAlex ID](https://openalex.org/A5074701373)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对 Proximal Policy Optimization (PPO) 的理论基础进行阐释，提出了将 PPO 视为近似策略梯度上升的框架，并给出了 surrogate 梯度的偏差上界与收敛性证明；同时识别并修正了截断 Generalized Advantage Estimation (GAE) 的尾部质量塌陷问题，提出了有限时 GAE 重新加权方法并在 LunarLander‑v3 与 Ant 环境中验证了性能提升。

**💡 创新点**

① 将 PPO 的多轮 minibatch 更新解构为偏置策略梯度的循环使用；② 推导 surrogate 梯度的偏差界并利用随机重排 (RR) 理论给出收敛定理；③ 发现并改正 GAE 的尾部质量塌陷，引入有限时 GAE。

**🔧 技术方法**

策略梯度理论、随机重排 (RR) 机制、GAE 估计、PPO 损失函数、深度网络实现、仿真实验。

**📊 数据集**

LunarLander‑v3（OpenAI Gym）、Ant（MuJoCo）等强化学习基准环境。

**📈 对比分析**

将默认 PPO（使用截断 GAE）与修改后的有限时 GAE 进行对比。实验显示，终止时间感知的 GAE 在 LunarLander‑v3 上学习曲线更快、平均回报更高，Episode 长度更短；在 Ant 上也表现出类似的收益提升。

**⚠️ 局限性**

理论中的常数非常大，难以直接指导超参数；偏差与方差分析主要基于理想化假设，实验验证仅限于少数环境；未深入探讨不同 λ、γ 参数对有限时 GAE 的影响。

---

## 491. Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes

**arXiv ID:** 2602.03381 | [PDF](https://arxiv.org/pdf/2602.03381v1)

**作者:** Axel Benyamine `[一作]` (Ecole polytechnique), Alain Durmus `[通讯]` (Ecole polytechnique and CNRS)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了“模糊不确定性下的马尔可夫决策过程”框架，将转移概率视为随机变量并使用风险度量评估策略的随机回报，统一了多种不确定MDP模型；

**💡 创新点**

首次给出在动态规划框架下，哪些法律不变风险度量（law‑invariant）能够满足Bellman方程，并证明除了最优、最保守和期望三种风险度量外，其他常用风险度量均不满足，阐明了动态规划在不确定MDP中的可行性边界；

**🔧 技术方法**

利用风险度量理论（单调性、平移不变性、正齐次性、可加性）、Bellman算子、动态规划原理、凸分析与表示定理，构造价值/策略迭代算法；

**📊 数据集**

论文为理论工作，不涉及具体实验数据集；

**📈 对比分析**

通过理论分析与定理证明，展示了在静态与重采样转移核下的可行性与不可行性，并给出价值/策略迭代的收敛速度（线性收敛）；没有实验比较；

**⚠️ 局限性**

主要限制在于：仅讨论法律不变、单调、平移不变的风险度量，忽略非法律不变或更一般的嵌套风险度量；假设转移核分布具有凸支撑与乘积结构；对静态转移核求解仍为NP‑hard，动态规划不适用；

---

## 492. PlanTRansformer: Unified Prediction and Planning with Goal-conditioned Transformer

**arXiv ID:** 2602.03376 | [PDF](https://arxiv.org/pdf/2602.03376v1)

**作者:** Constantin Selzer `[一作]` (Munich University of Applied Science), Fabina B. Flohr `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 PlanTRansformer，统一轨迹预测与规划，利用目标条件 Transformer 生成多模态安全轨迹；

**💡 创新点**

创新点在于将高层命令、动态可行性、碰撞避免及可达车道约束融入 Transformer，并采用教师‑学生策略逐步遮蔽周围车辆命令；

**🔧 技术方法**

使用 Transformer 编码器‑解码器架构、局部自注意力、可达车道聚合、动态与碰撞损失以及 GMM 输出进行端到端学习；

**📊 数据集**

主要在 Waymo Open Motion Dataset 上进行训练与评估；

**📈 对比分析**

与 MTR 对比，预测 mAP 提升 4.3%/3.5%；与 GameFormer 对比，5s 规划误差降低 15.5%，碰撞率与误差指标显著下降；

**⚠️ 局限性**

存在保守预测导致与真实轨迹偏离、碰撞约束导致多模态精度下降，以及对违规或缺失命令的鲁棒性不足。

---

## 493. Multi-Resolution Alignment for Voxel Sparsity in Camera-Based 3D Semantic Scene Completion

**arXiv ID:** 2602.03371 | [PDF](https://arxiv.org/pdf/2602.03371v1)

**作者:** Zhiwen Yang `[一作]` (Peking University), Yuxin Peng `[通讯]` (Peking University)

**通讯引用:** 8887 | [OpenAlex ID](https://openalex.org/A5047811387)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出多分辨率对齐（MRA）框架，解决基于摄像头的3D语义场景完成中的体素稀疏问题。

**💡 创新点**

设计了多分辨率视图变换器、立方体语义各向异性以及关键分布对齐模块，通过自我对齐的辅助监督弥补稀疏标签的不足。

**🔧 技术方法**

采用2D‑3D投影、多分辨率特征融合、种子特征对齐、ASPP传播、语义再分配、三维邻域差异度量、循环KL损失和自蒸馏等技术实现。

**📊 数据集**

在SemanticKITTI、SSCBench‑KITTI‑360以及nuScenes等公开数据集上进行训练与评估。

**📈 对比分析**

与现有最先进的摄像头基方法相比，MRA在SemanticKITTI验证集的IoU提升约1.1%（47.28% vs 46.21%），mIoU提升1.82%（17.14% vs 15.32%）；在SSCBench‑KITTI‑360提升0.97% IoU、1.89% mIoU；在nuScenes亦取得最高分。

**⚠️ 局限性**

在远距离小尺度物体和遮挡区域仍易出现误检，且计算开销略高，关键体素数量需要手工调参。

---

## 494. Pursuing Best Industrial Practices for Retrieval-Augmented Generation in the Medical Domain

**arXiv ID:** 2602.03368 | [PDF](https://arxiv.org/pdf/2602.03368v1)

**作者:** Wei Zhu `[一作]` (University of Hong Kong), Wei Zhu `[通讯]` (University of Hong Kong)

**通讯引用:** 18004 | [OpenAlex ID](https://openalex.org/A5068308955)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

系统性评估并构建最佳RAG实践以提升医学与通用问答性能

**💡 创新点**

通过模块化逐步优化、结合小到大分块、混合索引和COT‑Refine等创新方案，提出BP‑RAG方案

**🔧 技术方法**

使用BGE‑base嵌入、DeBERTa查询分类、滑动窗口/小到大分块、BM25与FAISS混合检索、伪响应生成与链式思考提示等技术

**📊 数据集**

MMLU、PubMedQA、PromptNER 三大基准以及自建27.9k问答对查询分类数据集

**📈 对比分析**

与无检索、各单模块替换对比，BP‑RAG平均提升25.6%准确率，但平均延迟提升32%，单模块分析显示混合索引和COT‑Refine贡献最大

**⚠️ 局限性**

仅评估开源LLM，未包含更强大模型或更复杂检索策略（如迭代检索），资源限制导致未覆盖 GPT‑4o、Gemini 等

---

## 495. Beyond Exposure: Optimizing Ranking Fairness with Non-linear Time-Income Functions

**arXiv ID:** 2602.03345 | [PDF](https://arxiv.org/pdf/2602.03345v1)

**作者:** Xuancheng Li `[一作]` (Tsinghua University), Yiqun Liu `[通讯]` (Tsinghua University)

**通讯引用:** 9929 | [OpenAlex ID](https://openalex.org/A5100668121)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了收入公平（Income Fairness）的概念，并给出了其正式定义与测量指标，改进了传统仅基于曝光的公平度量；

**💡 创新点**

创新点在于：①考虑上下文因素（如时间）导致的曝光到收入非线性映射；②设计了基于一阶、二阶泰勒展开的动态收入衍生（DIDRF）算法，既兼顾相关性又兼顾收入公平；

**🔧 技术方法**

使用的技术包括：泰勒展开（一阶/二阶）近似，边际收入/曝光分解，梯度化简，排序得分函数构造，在线情境下的无偏相关性估计与不确定度消减；

**📊 数据集**

实验数据集采用公开的排名基准MQ2008和Istella-s，人工构造的周期性、非周期性和恒定的时间收入函数；

**📈 对比分析**

与随机、TopK、FairK、FairCo、MMF、PLFair、MCFair、FARA等基线进行比较，DIDRF在所有设置下在收入公平度量上都最优，同时保持或提升cNDCG_avg@1,3,5的效果；

**⚠️ 局限性**

局限性在于：需要预估或已知的收入函数 f_d(t)；实验仅基于仿真而非真实业务数据；对复杂的时间-收入映射（如非线性、多因素）尚未全面验证。

---

## 496. SWE-World: Building Software Engineering Agents in Docker-Free Environments

**arXiv ID:** 2602.03419 | [PDF](https://arxiv.org/pdf/2602.03419v1)

**作者:** Shuang Sun `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 23709 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SWE-World，一个无需 Docker 的软件工程代理训练框架，通过 LLM 模拟执行环境和评估反馈，实现了完整的无容器化训练与推理。

**💡 创新点**

创新点在于用 LLM 学习的执行转移模型 SWT 与奖励模型 SWR 替代物理执行环境，消除容器化成本；同时实现可扩展的数据生成、SFT/ RL 训练和 TTS 选优，显著释放可用开源数据量。

**🔧 技术方法**

技术包括大型语言模型（Qwen2.5、Qwen3 等）训练 SWT/SWR、CoT 反推蒸馏、Sandbox + LLM 交互框架、ReAct 思考与行动循环、GRPO 强化学习、SWR 基于多样本投票的 TTS 选优。

**📊 数据集**

使用公开的 SWE-bench、SWE-Gym、SWE-rebench 以及自爬取的 16.6K PR/issue 构成 SWE-World 数据集，并从中采集真实的 Docker 轨迹用于训练。

**📈 对比分析**

在 SWE-bench Verified 上与 Docker 训练对比，SFT+RL+TTS 在 32B 模型上从 6.2% 提升至 68.2%，超过同等规模基线；SWT 的转移反馈与 SWR 的奖励模拟在准确率、精确率、召回率上均表现优于基线。

**⚠️ 局限性**

局限性包括仿真误差导致的执行/奖励不完全一致、CoT 推理延迟、对极端/复杂依赖的鲁棒性不足，以及对高质量真实轨迹的训练需求。

---

## 497. FactNet: A Billion-Scale Knowledge Graph for Multilingual Factual Grounding

**arXiv ID:** 2602.03417 | [PDF](https://arxiv.org/pdf/2602.03417v1)

**作者:** Yingli Shen `[一作]` (Tsinghua University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 37256 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套规模达10亿级、跨316种语言的知识图谱FactNet，统一Wikidata断言与Wikipedia文本证据，并提供可追溯的字节级证明指针。

**💡 创新点**

首次实现纯确定性构造流程，实现了高可审计、可复现的多语言知识图谱，并推出FactNet-Bench三大评测任务。

**🔧 技术方法**

采用Wikidata与Wikipedia dumps的字节级解析、模板与表格抽取、规则化归一化、结构化映射、Stanza分词与自定义规则分割、LLM检索与推理等技术。

**📊 数据集**

基于2025-11-01的Wikidata JSON、Wikipedia XML以及关联表，汇总约1.7B FactStatement、3.01B FactSense、1.55B FactSynset与3.69B RelationEdge。

**📈 对比分析**

在KGC、MKQA、MFC三项基准上，文本增强模型相较纯结构模型提升5–15% MRR/Hit@10；LLM在MKQA达41% Macro F1，检索+推理系统在MFC中准确率提升至70%+。

**⚠️ 局限性**

受限于仅使用静态dump与规则匹配，覆盖率在低资源语言与非模板化段落中低至30%，且仍保留原始维基百科的主题与性别偏见。

---

## 498. Most Convolutional Networks Suffer from Small Adversarial Perturbations

**arXiv ID:** 2602.03415 | [PDF](https://arxiv.org/pdf/2602.03415v1)

**作者:** Amit Daniely `[一作]` (Hebrew University), Idan Mehalel `[通讯]` (Hebrew University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

本文证明随机卷积神经网络在输入维度d下，几乎最优的ℓ2距离为‖x‖/√d 的对抗扰动存在，并且单步梯度下降即可找到；

**💡 创新点**

创新点在于将随机CNN的对抗鲁棒性问题与完全连接网络扩展，利用傅里叶分解精确界定随机卷积算子奇异值，并得到最小扰动的最优上界；

**🔧 技术方法**

核心技术包括傅里叶变换与群表示、随机矩阵理论对卷积算子的奇异值估计、梯度流与Rademacher不等式的组合分析；

**📊 数据集**

该工作为纯理论分析，未使用任何真实数据集；

**📈 对比分析**

由于是理论证明，未做实验对比；研究结果表明在满足宽度、深度和输入规模条件下，对抗扰动的范数可控制在‖x‖/√d 级别；

**⚠️ 局限性**

主要局限性包括：激活函数需满足C²光滑假设（不适用于ReLU），网络深度需为常数、宽度增长有限，且仅适用于随机初始化的网络，实际深度网络及非随机权重的情况尚未覆盖。

---

## 499. Socratic-Geo: Synthetic Data Generation and Geometric Reasoning via Multi-Agent Interaction

**arXiv ID:** 2602.03414 | [PDF](https://arxiv.org/pdf/2602.03414v1)

**作者:** Zhengbo Jiao `[一作]` (Alibaba Group Holding Limited), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14228 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Socceratic-Geo框架，利用多智能体（Teacher、Solver、Generator）循环生成并学习几何问题与对应图像，实现自动化几何数据合成与图像生成；

**💡 创新点**

创新点在于：① 通过“Reflect‑RePI”机制实现目标驱动的程序化合成，主动修正错误并生成符合几何约束的图像；② 将生成与学习耦合成闭环，Solver失败触发Teacher改造问题；③ 在生成过程中利用程序化指令训练Generator，使其具备高精度几何绘图能力；

**🔧 技术方法**

核心技术包括：多智能体交互框架、Python程序化几何绘制与验证、Group Relative Policy Optimization（GRPO）强化学习、反事实检验与自校正（Qualify），以及基于指令–图像对的Diffusion模型微调；

**📊 数据集**

主要使用了自研的Geo数据集（从108个种子生成），并与公开数据集（Geo170K、GeoReasoning、PGPS、MathVerse、MathVista、MathVision、WeMath、GenExam‑Math等）进行对比；

**📈 对比分析**

在六大几何推理基准上，Soctratic‑Solver 以 49.11% 的平均准确率（比零样本高 4.13 分），仅使用 2.5k 训练样本；在 GenExam‑Math 文字到图像任务中，Soctratic‑Generator 的 Relaxed 分数达到 42.4%，超过所有开源模型，逼近闭源 Gemini‑2.5‑Flash‑Image；

**⚠️ 局限性**

局限性在于：仍需依赖大型教师模型进行问题生成，且对非几何领域的迁移需要进一步验证；生成的图像虽精准但对复杂多图示的支持有限；模型训练资源需求较高。

---

## 500. Feasible strategies for conflict resolution within intuitionistic fuzzy preference-based conflict situations

**arXiv ID:** 2602.03403 | [PDF](https://arxiv.org/pdf/2602.03403v1)

**作者:** Guangming Lang `[一作]` (Changsha University of Science and Technology), Feng Xu `[通讯]` (Changsha University of Science and Technology)

**通讯引用:** 4679 | [OpenAlex ID](https://openalex.org/A5100769883)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于直觉模糊偏好的三向冲突分析模型，并设计可行策略通过调整偏好来降低冲突度。

**💡 创新点**

①将直觉模糊数引入偏好表述，实现对问题对偏好的细粒度描述；②采用贝叶斯损失函数客观计算阈值；③提出联合最小化冲突度与偏好调整幅度的优化算法。

**🔧 技术方法**

直觉模糊集合理论、三向决策、贝叶斯决策理论、约束优化（模拟退火）以及冲突度度量与分割方法。

**📊 数据集**

以中东冲突的六个国家与五个议题的偏好表为案例数据集。

**📈 对比分析**

与现有偏好、模糊、直觉模糊等模型对比，实验显示冲突度显著降低、联盟增加、计算效率可接受。

**⚠️ 局限性**

未考虑各代理与议题的权重，参数设置仍需经验；算法复杂度较高，适用于规模较小的情境。

---

## 501. Symbol-Aware Reasoning with Masked Discrete Diffusion for Handwritten Mathematical Expression Recognition

**arXiv ID:** 2602.03370 | [PDF](https://arxiv.org/pdf/2602.03370v1)

**作者:** Takaya Kawakatsu `[一作]` (Preferred Networks), Ryo Ishiyama `[通讯]` (Kyushu University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将手写数学表达式识别任务 reformulate 为离散扩散框架，通过迭代符号重掩与解码实现逐步细化。

**💡 创新点**

①使用离散扩散替代自回归，消除曝光偏差并实现全局结构一致性；②设计符号感知分词 (SAT)，将符号与结构修饰符一一对应；③引入随机掩码互学习 (RMML)，强化对书写多样性与结构歧义的鲁棒性。

**🔧 技术方法**

Vision Transformer 图像编码器；离散扩散推理；符号感知分词；随机掩码互学习；KL 对齐损失。

**📊 数据集**

MathWriting（230k 真实 + 400k 合成）和 CROHME 2014–2023 四届评测。

**📈 对比分析**

在 MathWriting 上 CER 5.56%、EM 60.42%，显著优于 BTTR、CoMER、ICAL；在 CROHME 2014–2023 上 EM 均高 4–5 个百分点，且在统一 224×224 预处理下保持最高。

**⚠️ 局限性**

扩散步数较多时推理慢；对极度模糊或错误书写的表达仍可能产生多义解析；目前仅支持离线图像输入，缺乏对在线笔迹时序的利用。

---

## 502. Learning-based Initialization of Trajectory Optimization for Path-following Problems of Redundant Manipulators

**arXiv ID:** 2602.03418 | [PDF](https://arxiv.org/pdf/2602.03418v1)

**作者:** Minsung Yoon `[一作]` (Korea Advanced Institute of Science and Technology), Sung-Eui Yoon `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3846 | [OpenAlex ID](https://openalex.org/A5078173428)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于强化学习的初始化轨迹生成方法 RL-ITG，用于在冗余机械臂路径跟随问题中快速生成高质量的初始关节轨迹，从而提升轨迹优化（TO）的收敛速度和最终解的质量。

**💡 创新点**

将示例引导强化学习与轨迹优化结合，引入 null‑space 投影的模仿奖励，以学习可行的 null‑space 运动；并在训练时使用大规模随机环境和目标路径，使策略能够泛化到多种任务与障碍场景。

**🔧 技术方法**

使用 Soft Actor‑Critic 强化学习框架、示例引导奖励、null‑space 投影模仿奖励、VAE 场景编码、SE(3) 位置与姿态表述、B‑spline 路径插值、以及关节极限、奇异性和碰撞约束评估等技术。

**📊 数据集**

在 Fetch 机器人工作空间内采样约 30,000 条路径（5,000 条无障碍 + 10,000 条有障碍），并使用 TORM 专家轨迹优化生成演示轨迹；同时使用 5,000 条随机桌面场景的 3D 占用网格训练 VAE。

**📈 对比分析**

与线性插值、贪婪 IK、行为克隆等三种基线以及两种 TO 方法（TORM、TrajOpt）在 6,000 个评估问题上比较，RL‑ITG 在成功率、平均位姿误差、生成时间和约束违规率上均优于基线，尤其在随机障碍设置中提升显著。

**⚠️ 局限性**

仅在静态桌面场景下验证，未处理动态环境或时间变化目标路径；演示轨迹质量依赖于专家 TO 的初始解；需要大量离线训练样本和计算资源。

---

## 503. SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training

**arXiv ID:** 2602.03411 | [PDF](https://arxiv.org/pdf/2602.03411v1)

**作者:** Huatong Song `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 23709 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了SWE-Master，一个开源、可复现的后训练框架，用于构建高效的软件工程代理；

**💡 创新点**

创新点包括：①系统化的数据构造与过滤策略（格式化与难度过滤）②结合长时序SFT与强化学习的RLVR+GRPO优化方案③引入基于LSP的IDE级代码导航工具，实现结构化仓库理解；④设计测试时扩展（TTS）与仿真验证（SWE-World）提升推理性能；

**🔧 技术方法**

主要技术涵盖：多轮监督微调（YaRN上下文扩展）、强化学习（GRPO+优势估计、clip-higher、去KL）、预算感知、Git命令限制、环境响应掩码、LSP工具封装、连续上下文压缩、SWE-World奖励模型；

**📊 数据集**

使用的数据集包括SWE-Gym、SWE-rebench、R2E-Gym、SWE-smith等Docker化环境，评测基准为SWE-bench Verified；

**📈 对比分析**

与现有开源代理（OpenHands、SWE-agent等）及基础模型（Qwen2.5-Coder-32B、Qwen3-4B等）对比，SWE-Master在SFT+RL+TTS下达成61.4%（RL）和70.8%（TTS@8）的Resolve Rate，显著优于同规模基线，且TTS提升效益显著；

**⚠️ 局限性**

局限性包括：①对大型模型和高计算资源的依赖（RL训练耗时、TTS算力需求）；②在多语言、非Python环境下的LSP兼容性尚未充分验证；③RL阶段仍易受环境错误或策略过拟合影响；④对极端复杂任务的推理预算仍有限；

---

## 504. UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning

**arXiv ID:** 2602.03410 | [PDF](https://arxiv.org/pdf/2602.03410v1)

**作者:** Piotr Wójcik `[一作]` (Jagiellonian University), Maciej Zieba `[通讯]` (Wrocław University of Science and Technology)

**通讯引用:** 1503 | [OpenAlex ID](https://openalex.org/A5083652196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用CLIP引导的超网络生成LoRA权重，实现动态、可扩展的概念消除；

**💡 创新点**

将超网络与LoRA结合，形成可即时生成的消除适配器，支持多概念并行消除并实现零样本泛化；

**🔧 技术方法**

采用LoRA参数化、超网络（MLP）生成权重、CLIP文本编码、Classifier-Free Guidance（Stable Diffusion）或直接权重应用（Flux）、梯度匹配的损失训练；

**📊 数据集**

使用CIFAR-10（对象消除）、I2P（不雅内容消除）、MS‑COCO（质量评估）、GIPHY Celebrity Detector（名人消除）等公开数据集；

**📈 对比分析**

与ESD、MACE、FMN、SAeUron、EraseAnything等基线进行对比，评估指标包括Acc_e、Acc_s、Acc_g、H_o、FID、CLIP分数；在Stable Diffusion和Flux上均取得最优或接近最优的消除效果，同时保持较低的FID和高的CLIP；

**⚠️ 局限性**

对超网络表达能力有限、对极少见概念的泛化仍受限、对高分辨率长文本可能需要更大文本编码器、训练时间相对传统LoRA仍占用显存

---

## 505. Universal Costas Matrices: Towards a General Framework for Costas Array Construction

**arXiv ID:** 2602.03407 | [PDF](https://arxiv.org/pdf/2602.03407v1)

**作者:** Fatih Gulec `[一作]` (University of Essex), Vahid Abolghasemi `[通讯]` (University of Essex)

**通讯引用:** 2139 | [OpenAlex ID](https://openalex.org/A5024631330)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了统一的 Universal Costas Matrix (UCM) 与 Universal Costas Frequency Matrix (UCFM) 框架，并通过从 UCFM 重构 UCM 的算法生成新的 Costas 数组。

**💡 创新点**

首次将 UCM 与 UCFM 视为统一的矩阵表示，利用其对称频率结构加速搜索，并为 AI 生成 Costas 数组奠定基础。

**🔧 技术方法**

基于 UCFM 的频率矩阵、UCM 的块结构、Polymorph 生成、Russo 搜索法以及 Python 实现的重构算法。

**📊 数据集**

使用已枚举的所有 Order≤29 Costas 数组构成的完整 UCFM，以及较大 Order≤1030 的代数生成数据库生成的不完整 UCFM。

**📈 对比分析**

与 Russo 方法对比，平均运行时间提升最高 59%，在 n≥13 时仍保持约 15% 的加速，表明重构方法更高效。

**⚠️ 局限性**

受限于数据稀缺，仅完成重构步骤；AI 预测 UCFM 尚未实现，需进一步研究。

---

## 506. Risk Awareness Injection: Calibrating Vision-Language Models for Safety without Compromising Utility

**arXiv ID:** 2602.03402 | [PDF](https://arxiv.org/pdf/2602.03402v1)

**作者:** Mengxuan Wang `[一作]` (South China University of Technology), Ming Li `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为 Risk Awareness Injection (RAI) 的训练‑free 框架，通过在视觉语言模型输入层注入风险信号，提升模型对多模态越狱攻击的安全识别能力。

**💡 创新点**

创新点在于：①构造 Unsafe Prototype Subspace 并使用稀疏门控在单一 token 级别进行风险信号注入；②仅在模型前置层进行一次性操作，既不破坏原有语义也避免了训练成本；③实现了安全与实用性的最佳平衡。

**🔧 技术方法**

采用的技术包括：语言嵌入构造风险子空间、视觉 token 与子空间的余弦相似度筛选、稀疏加权投射注入、单层安全激活，并结合标准的推理过程。

**📊 数据集**

使用的数据集包括 MM‑SafetyBench、JailBreakV‑28K、Video‑SafetyBench（安全评测）以及 MME、MM‑Vet（通用视觉理解评测）。

**📈 对比分析**

通过与 AdaShield、ECSO、CoCA、ShiftDC 等主流防御方法对比，RAI 在 Qwen2‑VL、LLaVA‑1.6‑7B、DeepSeek‑VL 等模型上将攻击成功率压至接近 0% 或大幅下降（如 LLaVA‑1.6‑7B 3.62%），同时在 MME、MM‑Vet 上保持与原模型相近的性能；推理延迟仅略增 13% 以内。

**⚠️ 局限性**

局限性在于：阈值与子空间构造仍需根据不同模型手工调优；对极端噪声、完全新的攻击模式或未来更复杂的多模态攻击尚未全面验证，且在更大规模的视频任务中可能需要进一步优化注入策略。

---

## 507. The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting

**arXiv ID:** 2602.03395 | [PDF](https://arxiv.org/pdf/2602.03395v1)

**作者:** Chen-Hui Song `[一作]` (E Fund Management Co., Ltd.), Liyuan Chen `[通讯]` (E Fund Management Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了金融预测中的监督标签时间窗口，并提出了Label Horizon Paradox：最优训练标签的时间窗口往往与最终预测目标不同。

**💡 创新点**

创新点在于：① 将监督标签视为可学习参数，使用双层优化自动定位最优标签窗口；② 提出了基于信号累积与噪声累积的理论分析，解释标签窗口选择的机制；③ 通过热身阶段与熵正则化提升训练稳定性。

**🔧 技术方法**

技术手段包括：深度学习框架（LSTM、GRU、Transformer、Mamba等），双层优化（inner loop 训练，outer loop 评估并更新标签权重 λ），标准化平均场热身，熵正则化，信号‑噪声理论推导。

**📊 数据集**

使用了中国沪深300、500、1000三大指数的分钟级行情数据，时间范围为2019年1月–2025年7月，进行训练/验证/测试划分。

**📈 对比分析**

与多种主流模型（DLinear、RLinear、GRU、LSTM、PatchTST、iTransformer、Mamba、Bi‑Mamba+、ModernTCN、TCN）对比，指标包括IC、ICIR、RankIC、RankICIR、Top‑10%日收益与Sharpe Ratio。实验显示，双层优化方法在所有模型与数据集上均显著提升IC与Sharpe Ratio，提升幅度约 5–20%。

**⚠️ 局限性**

局限性：① 只在中国A股市场验证，跨市场泛化未验证；② 需要较多计算资源（双层梯度、热身阶段）；③ 对于极端市场波动或结构性变革，信号‑噪声模型的假设可能失效；④ 训练过程对熵正则化参数及热身轮次较敏感。

---

## 508. On the Entropy Dynamics in Reinforcement Fine-Tuning of Large Language Models

**arXiv ID:** 2602.03392 | [PDF](https://arxiv.org/pdf/2602.03392v1)

**作者:** Shumin Wang `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8048 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了强化学习细调（RFT）中熵动态的理论框架，并基于单个logit更新推导了熵变化判别量S*，进一步将其推广到GRPO优化步骤，提出了批量归一化和词汇归一化的熵判别剪辑方法。

**💡 创新点**

创新点在于：①通过解析单个logit更新对熵的第一阶影响，提出S*判别量；②得到GRPO步骤下的熵变化一阶表达式，揭示熵崩塌机理；③基于此设计了高效的批量/词汇归一化熵判别剪辑策略，统一解释多种熵控制方法。

**🔧 技术方法**

主要技术包括：软最大梯度解析、第一阶近似、GRPO算法、熵判别量S*、批量与词汇归一化剪辑、对比实验评估。

**📊 数据集**

实验使用的数据集和模型有：Qwen2.5-7B/14B Instruct 预训练模型；DAPO‑Math‑17k 训练集（剔除高/低通过率样本）；DAPO500 验证集；AIME24/25、DAPO500 作为测试集。

**📈 对比分析**

与标准GRPO以及GRPO+等对照，使用Avg@K/Pass@K指标比较，结果显示熵判别剪辑在AIME24/25、DAPO500上平均提升约2–8%，显著提升模型的探索能力和最终性能。

**⚠️ 局限性**

局限性在于：仅考虑单token logit更新的一阶近似，忽略高阶交互效应；剪辑方法需要完整词表概率，计算成本相对较高；在多目标奖励或更复杂策略环境下的适用性仍待验证。

---

## 509. Toward a Sustainable Federated Learning Ecosystem: A Practical Least Core Mechanism for Payoff Allocation

**arXiv ID:** 2602.03387 | [PDF](https://arxiv.org/pdf/2602.03387v1)

**作者:** Zhengwei Ni `[一作]` (Zhejiang Gongshang University), Victor C. M. Leung `[通讯]` (University of British Columbia)

**通讯引用:** 65055 | [OpenAlex ID](https://openalex.org/A5035919267)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于最小核心（Least Core）的联邦学习收益分配框架，并给出一种堆栈剪枝算法实现其在大规模网络中的可行性；

**💡 创新点**

创新点在于将最小核心概念引入FL收益分配，既兼顾协作稳定性又能通过剪枝显著降低计算复杂度；

**🔧 技术方法**

采用堆栈式递归剪枝与线性规划求解最小核心，同时利用FL模型训练得到的收益函数；

**📊 数据集**

使用CICIDS‑2017数据集进行入侵检测实验，并用UCI Cleveland Heart Disease数据集验证纵向FL场景；

**📈 对比分析**

与数据量分配、留一法（leave‑one‑out）及Shapley值对比，实验显示LC分配在稳定性和公平性上更优，能够更准确识别关键参与者并降低潜在的“离队”风险；

**⚠️ 局限性**

局限在于仍需训练多组模型，虽然剪枝显著减少次数，但在极大规模或动态网络中仍可能面临计算瓶颈；此外，对阈值t1、t2的选择需经验或调优，可能影响分配精度。

---

## 510. How do people watch AI-generated videos of physical scenes?

**arXiv ID:** 2602.03374 | [PDF](https://arxiv.org/pdf/2602.03374v1)

**作者:** Danqing Shi `[一作]` (University of Cambridge), Miri Zilka `[通讯]` (University of Cambridge)

**通讯引用:** 829 | [OpenAlex ID](https://openalex.org/A5080823518)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过眼动实验研究人类观看真实与 AI 生成物理场景视频时的注视行为。

**💡 创新点**

创新点在于将视线分析应用于非面部、物理场景视频，并将观测者的真实性判断与注视模式关联。

**🔧 技术方法**

使用了 Gazepoint GP3 眼动仪、眼动追踪分析软件以及配对 t 检验等统计方法。

**📊 数据集**

数据集包含两组共 80 段视频，分别为 Physics‑IQ 与 Adobe Stock，均设有真实与 AI 生成版本。

**📈 对比分析**

比较方法是对不同任务、视频真实性及检测策略的眼动指标进行配对 t 检验，结果显示检测任务时注视更分散、扫描更广。

**⚠️ 局限性**

局限在于样本量仅 40 人、视频时长 5 秒、仅覆盖物理与非人类场景，且未评估更高级生成模型的影响。

---

## 511. Entropy Functions on Two-Dimensional Faces of Polymatroid Region with One Extreme Ray Containing Rank-One Matroid

**arXiv ID:** 2602.03363 | [PDF](https://arxiv.org/pdf/2602.03363v1)

**作者:** Kaizhe He `[一作]` (Xidian University), Qi Chen `[通讯]` (Xidian University)

**通讯引用:** 45237 | [OpenAlex ID](https://openalex.org/A5100340193)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对多项式空间中包含秩1矩形的二维面上的熵函数进行表征并分类

**💡 创新点**

首次给出该类二维面上熵函数的四类结构（全熵、Matúš型、Chen-Yeung型与非熵）并与已有结果关联

**🔧 技术方法**

运用矩阵理论、图论与信息不等式等组合理论与几何手段进行推导

**📊 数据集**

未使用任何实验数据集，全部采用理论推导与证明

**📈 对比分析**

与已知的 Γ₃、Γ₄ 二维面结果对比，完成了 n≥4 的普适性扩展，理论上实现了完整分类

**⚠️ 局限性**

仅处理了一个极射包含秩1矩阵的二维面，未覆盖所有二维面，部分结论仍需进一步验证

---

## 512. MeKi: Memory-based Expert Knowledge Injection for Efficient LLM Scaling

**arXiv ID:** 2602.03359 | [PDF](https://arxiv.org/pdf/2602.03359v1)

**作者:** Ning Ding `[一作]` (Samsung Research), Yehui Tang `[通讯]` (Samsung Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Memory-based Expert Knowledge Injection (MeKi) 方案，在 Transformer 层引入基于 ROM 的 token‑级专家知识库，解耦模型容量与算力。

**💡 创新点**

创新点在于将训练时的非线性投影重参数化合并进 ROM 表，实现零推理延迟；同时利用大容量 ROM 显著提升模型容量而不增加 FLOPs。

**🔧 技术方法**

使用技术包括 Transformer、token‑级静态/动态记忆表、低秩门控融合、SwiGLU 投影、重参数化、ROM 预取、RMSNorm、低秩投影等。

**📊 数据集**

预训练采用 FineWeb‑Edu‑Dedup 50B 语料；评估基准为 ARC‑E/C、OBQA、SciQ、PIQA、COPA、HellaSwag、BoolQ、WinoGrande、LAMBADA 等十项公开测试。

**📈 对比分析**

与相同规模密集模型、PLE、Engram 在相同 ROM 预算下对比；在 Qualcomm Snapdragon 8 Elite 上测 token/s，MeKi‑1.7B 在 10 任务上平均提升 2–4 分，同时保持与 4B 稠密模型相同的推理速度。

**⚠️ 局限性**

局限性包括：依赖 ROM 容量受限，重参数化只能消除推理时投影但训练时 FLOPs 仍高；在极大规模模型或极小 ROM 预算下效果有限；对多语言或少数语言知识迁移的适用性尚未验证。

---

## 513. PACE: Pretrained Audio Continual Learning

**arXiv ID:** 2602.03355 | [PDF](https://arxiv.org/pdf/2602.03355v1)

**作者:** Chang Li `[一作]` (Tsinghua University), Liyuan Wang `[通讯]` (Tsinghua University)

**通讯引用:** 3445 | [OpenAlex ID](https://openalex.org/A5115695075)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为PACE的新框架，专门解决预训练音频模型在持续学习场景下的表示偏移与饱和问题。

**💡 创新点**

创新点在于三层级的自适应策略：①首轮自适应只调整后层并冻结输出头，②多轮自适应采用子空间正交的LoRA投影，③边界感知扰动正则化以增强类间边界。

**🔧 技术方法**

使用的技术包括：预训练的音频ViT（EAT）、LoRA模块、子空间投影、递归解析式分类器、时间‑频率遮蔽扰动和自适应学习率策略。

**📊 数据集**

实验覆盖六大音频持续学习基准：粗粒度的ESC‑50、UrbanSound8K、Speech‑Command V2，细粒度的TIMIT‑2、TIMIT‑3和VocalSet。

**📈 对比分析**

相较于Vision‑CL PEFT方法和统计方法，PACE在所有基准上均实现了最高精度，平均比联合训练上限仅差不到1%，在细粒度任务上缩小了约7%之差。

**⚠️ 局限性**

局限性包括：需要额外的自适应训练时间（相较于RanPAC稍慢），以及对超参数（如层冻结阈值、投影子空间比例）较为敏感。

---

## 514. PEGRL: Improving Machine Translation by Post-Editing Guided Reinforcement Learning

**arXiv ID:** 2602.03352 | [PDF](https://arxiv.org/pdf/2602.03352v1)

**作者:** Yunzhi Shen `[一作]` (National Key Laboratory for Novel Software Technology, Nanjing University), Shujian Huang `[通讯]` (National Key Laboratory for Novel Software Technology, Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段强化学习框架PEGRL，先生成翻译草稿再进行后编辑，通过后编辑任务的学习信号稳定训练并引导全局与局部探索。

**💡 创新点**

创新点包括：① 将后编辑视作辅助任务并嵌入RL流程，利用条件回报估计降低方差；② 设计任务特定的梯度加权方案，在保持样本效率的同时降低训练噪声；③ 采用GRPO进行分组优势估计，提升稳定性。

**🔧 技术方法**

技术实现基于GRPO的政策梯度，使用两阶段Monte Carlo估计、后编辑奖励（COMET‑Kiwi+表面指标）、翻译奖励为后编辑平均奖励，并加入token预算惩罚；同时对梯度加权 λ_pe=M, λ_mt=1。

**📊 数据集**

实验数据集包括：English→Finnish、English→Turkish（WMT24 与 FLORES‑200 低资源样本），English↔Chinese（WMT24、FLORES‑200 及挑战集）以及对比大模型的数据。

**📈 对比分析**

与MT‑R1‑Zero、GRPO基线以及大型LLM（Gemini‑2.0‑Flash、OpenAI GPT‑5.2、DeepSeek‑V3.2、Seed‑X‑PPO‑7B等）比较。结果显示：在低资源场景下，PEGRL 在 COMET‑Kiwi、XCOMET 与 chrF++ 上均取得 5–7 分点提升；在 EN→TR 上几乎匹配 DeepSeek‑V3.2；对 4B/8B 模型的提升超过同类基准的 10–20 分。

**⚠️ 局限性**

局限性包括：仍无法与大规模 LLM 的表面指标匹配；仅在低资源、小模型上验证，未探究高资源或更大模型的表现；后编辑辅助任务可能不适用于奖励稀疏或不易量化的任务；对其他复杂任务（推理、代码生成等）的可推广性尚不清楚。

---

## 515. Robustness as an Emergent Property of Task Performance

**arXiv ID:** 2602.03344 | [PDF](https://arxiv.org/pdf/2602.03344v1)

**作者:** Shir Ashury-Tahan `[一作]` (IBM Research), Leshem Choshen `[通讯]` (MIT)

**通讯引用:** 1112 | [OpenAlex ID](https://openalex.org/A5040286212)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大模型在不同数据集与配置下的性能与鲁棒性（输出一致性）的关系，发现两者高度正相关。

**💡 创新点**

创新点在于将鲁棒性视为随着任务饱和度提升而自然产生的现象，而非单独的目标；强调任务特定能力是鲁棒性的主导因素。

**🔧 技术方法**

采用线性回归、ANOVA 等统计方法；使用输出一致性、得分标准差、性能下降率等鲁棒性指标；对比随机基线。

**📊 数据集**

六个公开数据集（如 IMDB、BoolQ、GPQA、MMLU 等），分别用 24 种配置（改写、温度、示例数等）生成预测。

**📈 对比分析**

与随机基线对比，模型鲁棒性显著高于基线；性能与鲁棒性相关系数 ≈ 0.92，斜率 1.05；模型差异对鲁棒性影响较小。

**⚠️ 局限性**

局限包括仅关注分类任务，未涵盖闭源模型，实验范围受成本限制，可能不适用于大幅度架构或训练范式的模型。

---

## 516. Causal Graph Learning via Distributional Invariance of Cause-Effect Relationship

**arXiv ID:** 2602.03353 | [PDF](https://arxiv.org/pdf/2602.03353v1)

**作者:** Nang Hung Nguyen `[一作]` (University of Tokyo), Masashi Sugiyama `[通讯]` (RIKEN)

**通讯引用:** 22125 | [OpenAlex ID](https://openalex.org/A5072744508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种利用因果关系的条件分布不变性进行父节点识别并恢复因果图的框架；

**💡 创新点**

创新点在于将因果不变性转化为对不同前因分布下效应-因果条件分布方差的检验，从而避免昂贵的局部条件独立性检验；

**🔧 技术方法**

核心技术包括基于Markov blanket的子图稀疏化、利用DFS寻找最大团生成父候选集，以及通过对观测数据下采样生成不同前因分布的合成子集进行不变性检验；

**📊 数据集**

实验使用了合成的Erdos‑Renyi、bipartite、scale‑free DAG 数据以及真实数据集SACHS、bnlearn 包中的七个小到大型图（如Munin 1041 变量）进行评估；

**📈 对比分析**

与PC、GIES、FCI、NOTEARS、MLP‑NOTEARS、DAS、SCORE等基线相比，所提方法在SHD、错误率和运行时间上均实现了至少 25 倍的加速并在大规模图上保持或超过最佳准确率；

**⚠️ 局限性**

主要局限在于对观测数据无未观测混杂变量的假设、对前因分布采样的依赖以及在极端稠密图中可能出现的搜索子图规模膨胀。

---

## 517. Enhancing Navigation Efficiency of Quadruped Robots via Leveraging Personal Transportation Platforms

**arXiv ID:** 2602.03397 | [PDF](https://arxiv.org/pdf/2602.03397v1)

**作者:** Minsung Yoon `[一作]` (Korea Advanced Institute of Science and Technology), Sung-Eui Yoon `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3846 | [OpenAlex ID](https://openalex.org/A5078173428)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了基于强化学习的主动运输器骑乘方法（RL‑ATR），实现四足机器人在个人运输平台（如Segway）上的骑乘，从而显著提升长距离导航的速度与能效。

**💡 创新点**

创新点包括：①首次将主动运输器骑乘融入四足机器人；②采用两级状态估计（内在与外在）弥补观测不足；③引入网格自适应课程学习以提升训练效率；④在不同机器人与运输器组合下验证跨平台兼容性。

**🔧 技术方法**

技术手段涵盖：模型无关强化学习（PPO）、系统识别与在线估计（CNN‑GRU）、正则化在线自适应、域随机化、命令分布调度、机械能耗（CoT）评估。

**📊 数据集**

实验数据来源为仿真环境：Isaac Gym并行运行4096个环境，使用A1、Go1、Anymal‑C、Spot四种四足模型与两类运输器，随机采样内在参数与命令空间。

**📈 对比分析**

与纯步行基线比较，RL‑ATR在命令跟踪误差热图中覆盖更广的速度/转向空间；机械CoT显著降低（约30‑50%），并通过消融实验验证课程学习与状态估计对性能的关键作用。

**⚠️ 局限性**

局限性在于仅在仿真中验证，未进行真实平台试验；仅考虑两种运输器设计；未实现上下台操作；缺乏外部感知以处理复杂环境。

---

## 518. Seeing Through the Chain: Mitigate Hallucination in Multimodal Reasoning Models via CoT Compression and Contrastive Preference Optimization

**arXiv ID:** 2602.03380 | [PDF](https://arxiv.org/pdf/2602.03380v1)

**作者:** Hao Fang `[一作]` (Tsinghua University), Yaowei Wang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8564 | [OpenAlex ID](https://openalex.org/A5100631216)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 C3PO 框架，先通过 Chain‑of‑Thought 压缩去除冗余推理文本，再用对比偏好优化提升推理质量，从而显著降低多模态大型推理模型的幻觉现象。

**💡 创新点**

创新点在于（1）利用信息瓶颈理论指导 CoT 压缩，减少视觉信息被忽视；（2）构建多模态幻觉诱导负样本并结合 AI 反馈生成高质量正样本，完成推理层面的对比偏好学习；（3）提供理论证明与多任务实验验证。

**🔧 技术方法**

核心技术包括：LoRA‑based 细化学习、基于 token‑importance 的 CoT 压缩、RLAIF‑V 生成 AI 反馈、对比偏好学习（DPO）与 anchor 约束、幻觉诱导机制以及信息瓶颈分析。

**📊 数据集**

主要使用数据集：RLAIF‑V（20k 训练样本）、MSCOCO（CAPTION 评估）、POPE、AMBER、GPT‑4‑assisted Fine‑Grained Hallucination benchmark、MME 与 MMBench，用于多模态推理与幻觉评测。

**📈 对比分析**

与现有方法 RLAIF‑V 与 OPA‑DPO 对比，C3PO 在 CHAIR、POPE、AMBER、GPT‑4‑assisted 等多种幻觉度量上均取得显著下降（如 CHAIR_S 减少 13%+、SHR 降低 14%+），并保持甚至提升一般多模态能力，验证了框架的有效性。

**⚠️ 局限性**

局限性包括：需要额外的 AI 反馈和幻觉诱导样本，训练成本相对较高；对不同模型结构的迁移仍需进一步验证；在极端复杂关系/属性幻觉方面提升有限。

---

## 519. SEW: Strengthening Robustness of Black-box DNN Watermarking via Specificity Enhancement

**arXiv ID:** 2602.03377 | [PDF](https://arxiv.org/pdf/2602.03377v1)

**作者:** Huming Qiu `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**通讯引用:** 69653 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对黑盒DNN水印的特异性增强方法SEW，提升水印在黑盒查询中的鲁棒性。

**💡 创新点**

创新点在于量化水印特异性并通过引入覆盖样本与噪声自适应优化，显著压缩可被近似提取的关键空间，从而提高对删除攻击的抵抗力。

**🔧 技术方法**

采用基于高斯噪声扰动的关键噪声上限优化、交叉熵训练以及自适应噪声调优技术，融合原始关键与覆盖样本进行联合训练。

**📊 数据集**

实验使用CIFAR‑10、CIFAR‑100和TinyImageNet三个视觉数据集，并在BERT、Text‑CNN等NLP任务上验证通用性。

**📈 对比分析**

与十种主流黑盒水印基线相比，SEW在保持100%水印准确率的同时，特异性指标大幅降低（如从0.3569降至0.0364），并在六种主流删除攻击（Neural Cleanse、Dehydra、MOTH、FeatureRE、Fine‑Tuning、Fine‑Pruning）中实现近乎100%防御率，整体性能优于基线。

**⚠️ 局限性**

局限性包括：对高噪声扰动的鲁棒性仍有限，过高的特异性可能导致对合法近似触发器的误判；在某些模型架构或任务中自适应噪声参数仍需手动微调，且对未来更高级的删除攻击的防御机制尚需进一步研究。

---

## 520. Unifying Watermarking via Dimension-Aware Mapping

**arXiv ID:** 2602.03373 | [PDF](https://arxiv.org/pdf/2602.03373v1)

**作者:** Jiale Meng `[一作]` (Zhejiang University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2796 | [OpenAlex ID](https://openalex.org/A5028270700)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种通过维度感知映射统一深度水印的框架，并在视频域中实例化为 DiMap‑V，实现多功能水印（版权验证、局部嵌入、时空篡改定位和帧序恢复）

**💡 创新点**

创新点在于将水印信息视作不同维度的payload，揭示嵌入/提取维度关系决定功能，并首次在视频域跨维度映射实现时空定位与顺序恢复

**🔧 技术方法**

使用端到端编码器‑解码器网络、噪声层模拟攻击、3D 时空掩码、跨维度投影（升维与降维映射）以及多通道帧编码来显式编码帧身份

**📊 数据集**

在 SA‑V 视频数据集上训练，测试集包含 1,000 个全球视频和 5,000 个局部掩码视频，视频分辨率 256×256，帧数 8

**📈 对比分析**

与 9 种公开基线（如 MaskWM‑ED、VideoSeal、RivaGAN 等）在 PSNR/SSIM、鲁棒性（不同噪声、几何、帧级、压缩）和 IoU 上比较，结果显示在不改动网络结构的情况下，DiMap‑V 在鲁棒性、可视性和定位/顺序恢复上均优于基线

**⚠️ 局限性**

主要限制包括多通道时空掩码下跨帧一致性下降导致高维提取性能下降，且方法目前仅在视频域验证，对不同分辨率或更长视频的推广仍待进一步研究

---

## 521. Z3D: Zero-Shot 3D Visual Grounding from Images

**arXiv ID:** 2602.03361 | [PDF](https://arxiv.org/pdf/2602.03361v1)

**作者:** Nikita Drozdov `[一作]` (Lomonosov Moscow State University), Maksim Kolodiazhnyi `[通讯]` (Lomonosov Moscow State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个从多视图图像到3D视觉定位的零监督框架。

**💡 创新点**

创新在于将零样本3D实例分割MaskClustering与VLM推理、SAM3-Agent等技术结合，并通过多视图聚合提升性能。

**🔧 技术方法**

使用了MaskClustering、SAM3-Agent、CLIP、VLM（如Qwen3-VL/Seed1.5-VL）、DUSt3R、TSDF重建等。

**📊 数据集**

在ScanRefer和Nr3D数据集上进行评估。

**📈 对比分析**

与现有零监督方法相比，在ScanRefer Acc@0.5提升至约54.8（相较于OpenScene 13.2），在Nr3D top‑1准确率也达到SOTA。

**⚠️ 局限性**

局限包括依赖CLIP做视图预选导致对复杂概念识别受限、图像‑only情形对重建质量敏感、MaskClustering计算开销大。

---

## 522. QASM: A Novel Framework for QUIC-Aware Stateful Middleboxes

**arXiv ID:** 2602.03354 | [PDF](https://arxiv.org/pdf/2602.03354v1)

**作者:** Hari Hara Sudhan Selvam `[一作]` (Indian Institute of Technology Gandhinagar), Sameer G. Kulkarni `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 1077 | [OpenAlex ID](https://openalex.org/A5081037214)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种通用框架 QASM，让有状态中间盒能够在 QUIC 连接迁移时准确追踪流，保证 NAT、负载均衡、限速器及 K8s 服务正常工作。

**💡 创新点**

核心创新是利用 QUIC 的连接 ID 及客户端代理/跟踪代理协同，将连接迁移信息实时传递给中间盒，实现对多种有状态中间盒的统一兼容，并保持低开销。

**🔧 技术方法**

实现基于 eBPF/QUIC 库的客户端代理、独立的跟踪代理和在中间盒中增添的 QUIC 解析与追踪表；评估使用 Mininet+scapy、aioquic HTTP/3。

**📊 数据集**

使用在 Mininet 仿真环境中的自建 HTTP/3 客户端/服务器数据流，以及通过 scapy 生成的测试包进行性能测评。

**📈 对比分析**

通过与默认 NAT/限速器对比，采用主动/被动模式下的 QASM 在延迟、吞吐量和 CPU/内存占用上均保持 <5% 额外开销，且能在高迁移速率（100 Hz）下保持稳定。

**⚠️ 局限性**

局限性包括仅在实验室仿真下验证，未测试真实高流量环境；需要部署额外的跟踪代理；对隐私无加密支持时可能泄露 CID 信息。

---

## 523. AesRec: A Dataset for Aesthetics-Aligned Clothing Outfit Recommendation

**arXiv ID:** 2602.03416 | [PDF](https://arxiv.org/pdf/2602.03416v1)

**作者:** Wenxin Ye `[一作]` (Wuhan University of Technology), Jimmy Xiangji Huang `[通讯]` (York University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过构建AesRec数据集，首次在服装推荐系统中加入多维度的美学评估，并基于此实现了美学导向的推荐模型。

**💡 创新点**

创新点在于：①提出六维（单品）+八维（成衣）多维美学量化指标；②利用大型视觉语言模型（VLM）进行大规模自动美学打分，并通过严格的人机一致性验证确保质量；③在推荐框架中引入联合损失，将个性化和美学排序目标统一优化。

**🔧 技术方法**

采用的技术主要包括：多模态视觉语言模型（Qwen‑VL‑Max）进行美学评分；基于图神经网络与对比学习的CrossCBR框架；联合损失函数（个性化推荐损失+美学排序损失）进行端到端训练。

**📊 数据集**

使用的基准数据集为从阿里巴巴时尚推荐平台POG中筛选并整理的AesRec，包含27,694套装、42,526件单品、53,843名用户以及约170万用户-套装互动。

**📈 对比分析**

与传统仅基于交互的推荐模型（如BPR‑MF、HyperMBR、CrossCBR、LLM‑qwen）对比，AesRec在美学评分（ΔScore_Aes@K）和排名曝光公平性（ExpoGap）上均实现显著提升（正向提升≥0.19、ExpoGap≥0.65），同时保持相近的个性化召回/ NDCG，说明美学约束并未显著损失推荐效果。

**⚠️ 局限性**

局限性包括：①模型在追求美学时对个性化召回有轻微牺牲，需进一步平衡；②数据来源单一平台，可能缺乏跨文化的美学差异；③使用VLM进行评估仍可能带来尺度偏差和模型固有偏见；④未考虑时间动态和用户美学偏好的长期演变。

---

## 524. Verified Critical Step Optimization for LLM Agents

**arXiv ID:** 2602.03412 | [PDF](https://arxiv.org/pdf/2602.03412v1)

**作者:** Mukai Li `[一作]` (Tencent AI Lab), Dong Yu `[通讯]` (Tencent AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Critical Step Optimization（CSO）的后训练方法，专注于在失败轨迹中验证关键决策步骤并进行偏好学习。

**💡 创新点**

创新点在于：①从失败轨迹出发，仅筛选能显著改变任务结果的关键步骤；②利用过程奖励模型（PRM）快速定位候选步骤，再通过分支rollout验证成功，从而获得精确、可验证的偏好对；③避免传统轨迹级或步骤级奖励噪声，提升归因精度。

**🔧 技术方法**

采用的技术包括：Process Reward Model（基于Claude‑3.7 Sonnet的评分）；分支rollout和结果验证；Direct Preference Optimization（DPO）进行偏好学习；ReAct轨迹框架；以及迭代式在线细化。

**📊 数据集**

实验使用的主要数据集为GAIA‑Text‑103文本子集和XBench‑DeepSearch‑2505；训练数据来源于CK‑Pro‑8B的SFT 47K轨迹。

**📈 对比分析**

与GPT‑4.1、Claude‑3.7 Sonnet、Qwen3‑8B、CK‑Pro‑8B（SFT）以及多种后训练基线（ETO、RFT、Step‑DPO、IPR）进行比较，CSO在GAIA上整体准确率提升至49.5%（比SFT提升37%相对），匹配GPT‑4.1；在XBench上提升26%；仅需对16%步骤监督，且相较其他方法至少高5分。

**⚠️ 局限性**

局限性：①验证过程需要完整执行分支轨迹，耗时较大，难以直接应用于在线RL；②当前PRM依赖闭源模型，无法与政策模型联合训练，限制了可迁移性与进一步提升。

---

## 525. Deep-Learning-Based Control of a Decoupled Two-Segment Continuum Robot for Endoscopic Submucosal Dissection

**arXiv ID:** 2602.03406 | [PDF](https://arxiv.org/pdf/2602.03406v1)

**作者:** Yuancheng Shao `[一作]` (Tongji University), Peng Qi `[通讯]` (Tongji University)

**通讯引用:** 7077 | [OpenAlex ID](https://openalex.org/A5025104684)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发双段连续体机器人DESectBot，并提出基于GRU深度学习的控制算法，实现对ESD手术中机器人末端位置与姿态的精确实时控制。

**💡 创新点**

创新点包括：①解耦双段结构与内置手术镊子实现6自由度末端操作；②首次将GRU网络应用于双段连续体机器人，显著提升非线性耦合控制精度；③在同一工作空间内用统一数据集对比多种模型，验证GRU优越性。

**🔧 技术方法**

采用NiTi丝驱动的双段连续体结构、磁电（NDI）跟踪采样、GRU深度学习控制网络，并与Jacobian逆运动学、MPC、FNN、LSTM等模型做对比。

**📊 数据集**

数据集为蒙特卡罗采样生成的约20400条5Hz时序数据（训练14280条、验证4080条、测试2040条），覆盖30×45 mm工作空间，测试轨迹不包含在训练集中。

**📈 对比分析**

对比方法为Jacobian、MPC、FNN、LSTM；GRU在nested-rectangle、Lissajous轨迹跟踪、姿态控制以及peg transfer实验中均取得最低RMSE、最高成功率（100%），peg transfer平均时间11.8 s，比dVRK更快、同等或更高的成功率。

**⚠️ 局限性**

限制：控制更新率受EM跟踪5 Hz限制；未在真实临床环境下验证；缺乏多模态感知、碰撞回避与自校准等高级功能；需要进一步提升采样频率和实时推理效率。

---

## 526. Dynamic Topology Optimization for Non-IID Data in Decentralized Learning

**arXiv ID:** 2602.03383 | [PDF](https://arxiv.org/pdf/2602.03383v1)

**作者:** Bart Cox `[一作]` (Delft University of Technology), Jérémie Decouchant `[通讯]` (Delft University of Technology)

**通讯引用:** 4595 | [OpenAlex ID](https://openalex.org/A5024072796)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完全去中心化的动态拓扑优化算法，基于节点模型的相异度自适应地选择邻居，从而提升非IID环境下的训练效果；

**💡 创新点**

创新点在于：①仅利用本地模型相异度进行指导性拓扑调整，完全不依赖全局信息或中心协调；②通过软最大化采样与随机重采样相结合，既保持多样性又保证网络连通性；③在保持固定入度的同时，动态发现新节点，兼顾稀疏通信与全局收敛；

**🔧 技术方法**

使用的核心技术包括：cosine相似度评估与三角不等式的近似传播、基于softmax的多样性驱动邻居采样、gossip式节点发现与邻居列表共享、以及对接入/离开的连接请求的匹配机制；

**📊 数据集**

实验数据集为CIFAR‑10和FEMNIST，采用Dirichlet分布(α=0.1)模拟非IID分布；

**📈 对比分析**

与三种基线（静态3/7‑regular随机图+Metropolis–Hastings、完全连通、每轮随机k‑regular）对比；在100节点、3/7/14连通度下，所提方法在准确率、收敛速度、节点间方差上均接近完全连通上限，且通信成本低于基线，尤其在低连通度时明显优于静态与epidemic方法；

**⚠️ 局限性**

局限性包括：①对极端低连通度时仍可能出现节点孤立现象；②需要调参β与相似度评估间隔Δ_r，适配不同规模网络；③未考虑节点异质性（如带宽、延迟）对拓扑演化的影响。

---

## 527. SLIM-Diff: Shared Latent Image-Mask Diffusion with Lp loss for Data-Scarce Epilepsy FLAIR MRI

**arXiv ID:** 2602.03372 | [PDF](https://arxiv.org/pdf/2602.03372v1)

**作者:** Mario Pascual-González `[一作]`, Ezequiel López-Rubio `[通讯]` (University of Málaga)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

使用共享瓶颈U‑Net实现了基于扩散模型的 FCD 病灶与 FLAIR 图像的联合合成。

**💡 创新点**

创新点在于：① 单一共享瓶颈U‑Net强制图像与病灶掩码共享特征，显著降低模型容量；② 引入可调 Lp 目标并系统比较噪声、速度、x0 三种预测参数化；③ 发现 x0 预测与 L1.5 损失在图像质量与掩码形态之间取得最优平衡。

**🔧 技术方法**

核心技术包括：扩散模型（DDPM/DDIM）、共享瓶颈 U‑Net、可调 Lp 损失、正弦位置编码、条件嵌入、EMA 以及 2D 切片级生成。

**📊 数据集**

使用 Schuch 等人公开的 85 例 FCD II 患者及 85 例健康对照的 FLAIR 切片数据，最终只选取 78 例 FCD II 切片进行训练与评估。

**📈 对比分析**

通过 KID、LPIPS、MMD‑MF 及每个形状特征的 Wasserstein 距离进行内部基准比较；实验显示 x0‑预测加 L1.5 损失在 KID、LPIPS、MMD‑MF 上均优于 ϵ‑预测 L2 基线，证明该组合在低数据场景下效果最佳。

**⚠️ 局限性**

局限性包括：仅采用 2D 切片生成，缺乏 3D 空间一致性；未与现有双流或多阶段联合合成框架进行直接对比；数据量仍有限，缺乏外部验证。

---

## 528. Learning-based Adaptive Control of Quadruped Robots for Active Stabilization on Moving Platforms

**arXiv ID:** 2602.03367 | [PDF](https://arxiv.org/pdf/2602.03367v1)

**作者:** Minsung Yoon `[一作]` (Korea Advanced Institute of Science and Technology), Sung-Eui Yoon `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3846 | [OpenAlex ID](https://openalex.org/A5078173428)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了面向六自由度移动平台的学习型主动平衡方法 LAS-MP，让四足机器人能在动态平台上保持稳定。

**💡 创新点**

创新点在于将状态估计器（显式估计机器人与平台状态，隐式估计机器人内在属性）与对齐指令相结合，并通过 B‑spline 平台轨迹生成与进阶调度实现多场景鲁棒学习。

**🔧 技术方法**

使用技术包括深度强化学习（PPO+ROA）、并行 Isaac Gym 仿真、B‑spline 轨迹生成、状态估计网络（CNN+MLP）和对齐指令特征工程。

**📊 数据集**

数据集为仿真生成的训练轨迹集 Ξ_train（多段 B‑spline 轨迹）和评估轨迹集 Ξ_eval（10,000 条 6‑DoF 轨迹），覆盖更广泛的运动与机器人内在参数范围。

**📈 对比分析**

与三种基线（静止策略、R‑S Policy、R‑S Policy+Oracle）对比，LAS‑MP 在碰撞率、位移误差、功耗、姿态偏差等指标上均显著优于基线。

**⚠️ 局限性**

限制：目前仅在仿真中验证，缺乏真实平台实验；对已建立的稳态偏差缺乏补偿机制；对外部传感器的利用不足。

---

## 529. GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer

**arXiv ID:** 2602.03358 | [PDF](https://arxiv.org/pdf/2602.03358v1)

**作者:** Junmo Cho `[一作]` (Korea Advanced Institute of Science and Technology), Hae Beom Lee `[通讯]` (Korea University)

**通讯引用:** 13557 | [OpenAlex ID](https://openalex.org/A5100737934)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将提示搜索视为后验推理，使用GFlowNet和动态记忆更新来优化提示

**💡 创新点**

创新点在于把提示优化转化为后验推理，并结合离线GFlowNet训练与无参动态记忆更新以提升样本效率

**🔧 技术方法**

主要技术包括Generative Flow Networks（GFlowNet）、离线重放学习、Dynamic Memory Update（DMU）以及LoRA微调

**📊 数据集**

实验数据集覆盖GLUE/SuperGLUE文本分类、Instruction Induction、BigBench Induction、MMLU、OpenBookQA等

**📈 对比分析**

与StablePrompt、GrIPS、PromptBoosting等方法对比，GFlowPO在多数任务上取得更高准确率，平均提升约1–5%

**⚠️ 局限性**

限制在于尚未在需要中间推理链的任务上验证，且当前方案仅针对单一任务优化，缺乏跨任务的元学习能力

---

## 530. Achieving Linear Speedup for Composite Federated Learning

**arXiv ID:** 2602.03357 | [PDF](https://arxiv.org/pdf/2602.03357v1)

**作者:** Kun Huang `[一作]` (Chinese University of Hong Kong), Shi Pu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2924 | [OpenAlex ID](https://openalex.org/A5051722330)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了FedNMap，一种利用正则化映射和局部校正的联邦学习算法，能够在非凸复合目标下实现线性加速；

**💡 创新点**

创新点在于将正常映射（normal map）与局部校正相结合，克服了非光滑正则项带来的偏差，并在不需要数据同质性假设的前提下实现线性加速；

**🔧 技术方法**

核心技术包括正则化映射更新、局部校正机制、梯度无偏估计以及多步Lyapunov分析；

**📊 数据集**

在MNIST（单隐藏层网络）和SVHN（VGG-16）上进行实验，使用弹性网正则化；

**📈 对比分析**

与Zhang方法和FedCanon进行对比，FedNMap在所有测试配置下均取得更低的停滞度和更快的收敛速度，尤其在高异质性和大规模客户端/多步更新场景下表现突出；

**⚠️ 局限性**

局限性在于理论仍依赖于梯度方差上界和弱凸正则项假设，且实验仅在标准图像分类数据集上验证，缺乏对更复杂任务或非图像数据的评估。

---

## 531. Building Interpretable Models for Moral Decision-Making

**arXiv ID:** 2602.03351 | [PDF](https://arxiv.org/pdf/2602.03351v1)

**作者:** Mayank Goel `[一作]`, Paras Chopra `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个两层Transformer模型，针对结构化的自动驾驶道德困境进行预测，并对其内部机制进行可解释性分析。

**💡 创新点**

创新点在于设计了基于角色、数量和团队的组合嵌入来显式编码情境，并将因果干预、层级归因和电路探测三种方法结合，以定位道德偏见的计算阶段。

**🔧 技术方法**

采用了自定义Transformer、因果干预（DoWhy）、层级注意力归因、电路探测以及梯度加权注意力相关性等技术。

**📊 数据集**

使用Moral Machine数据集（约540万条经过筛选的场景）作为训练、验证和测试的数据来源。

**📈 对比分析**

与大型预训练语言模型相比，模型参数仅104k，准确率达77%（比最优配置77.5%略低），证明小模型即可实现道德判断且更易进行机制分析。

**⚠️ 局限性**

限制在于模型学习到的道德层次受训练时人类偏见影响，缺乏跨文化适应性，且在更复杂或多样化情境下的鲁棒性有限。

---

## 532. Manipulation via Force Distribution at Contact

**arXiv ID:** 2602.03350 | [PDF](https://arxiv.org/pdf/2602.03350v1)

**作者:** Haegu Lee `[一作]` (Maersk Mc-Kinney Moller Institute), Christoffer Sloth `[通讯]` (Maersk Mc-Kinney Moller Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种 Force-Distributed Line Contact（FDLC）模型，并在基于 iLQR 的双层优化框架中用于生成接触丰富的操作轨迹；

**💡 创新点**

创新点在于将线接触用两点虚拟弹簧阻尼系统近似，使得接触力能够在接触线段上非均匀分布，从而直接产生扭矩、减少滑动；

**🔧 技术方法**

使用了双层优化、iLQR、隐式函数求导、虚拟弹簧阻尼动力学以及 GelSight 触觉传感器进行力分布可视化；

**📊 数据集**

使用了仿真数据（盒子旋转任务）和真实数据（UR5e 搭载 GelSight 及 ArUco 标记的盒子旋转实验），无公开数据集；

**📈 对比分析**

通过将相同任务的点接触模型和 FDLC 模型在仿真中同一物理参数下比较，结果显示 FDLC 轨迹在控制能量、机器人行程距离和接触力连续性上均优于点接触；实验中 FDLC 产生的角度误差显著低于点接触，验证了其鲁棒性；

**⚠️ 局限性**

局限包括：仅在二维平面盒子旋转任务中验证；仿真与真实系统之间仍有误差（sim‑to‑real 问题）；模型仅采用两点近似，可能无法完全捕捉更复杂的接触形状；实验使用开环执行，缺乏在线控制验证。

---

## 533. Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures

**arXiv ID:** 2602.03379 | [PDF](https://arxiv.org/pdf/2602.03379v1)

**作者:** Sangyeon Yoon `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**通讯引用:** 507 | [OpenAlex ID](https://openalex.org/A5049196468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了机器学习模型在被遗忘后通过细调恢复已遗忘信息的现象（benign relearning），并探讨其背后的驱动因素。

**💡 创新点**

发现语法相似度是benign relearning的主要驱动因素，并提出语法多样化的忘记方法来抑制重学习。

**🔧 技术方法**

采用梯度上升 (GA)、负偏好优化 (NPO)、SCRUB 等参数优化型遗忘算法，并利用 Levenshtein 距离衡量语法相似度。

**📊 数据集**

主要使用 TOFU、BLUR、WMDP、WHP、RWKU 等公开基准数据集进行实验。

**📈 对比分析**

通过与传统遗忘方法在忘记成功率、Relearn Success Rate 和模型效用（ROUGE、概率、真值比）等指标对比，语法多样化显著提升遗忘效果并降低重学习率，同时保持更高的模型效用。

**⚠️ 局限性**

局限性在于实验主要针对 LLM 的问答式结构，未探究更广泛的结构因素；语法多样化依赖于外部生成模型的质量，且对不同模型架构的普适性尚未充分验证。

---

## 534. Precision in Practice: Knowledge Guided Code Summarizing Grounded in Industrial Expectations

**arXiv ID:** 2602.03400 | [PDF](https://arxiv.org/pdf/2602.03400v1)

**作者:** Jintai Li `[一作]` (Wuhan University), Xiaoyuan Xie `[通讯]` (Wuhan University)

**通讯引用:** 2346 | [OpenAlex ID](https://openalex.org/A5100746280)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向工业开发者期望的函数级代码摘要生成方法ExpSum，旨在提升自动生成的代码摘要在工业项目中的可接受度。

**💡 创新点**

创新点包括：①基于工业开发者调研提炼出三大期望（域词使用、功能类别标注、冗余细节剔除）；②构建包含路径语义的知识库并设计CACKR级联检索；③在LLM生成过程中引入两阶段约束驱动提示，实现功能类别推理与摘要生成的协同。

**🔧 技术方法**

采用的技术包括：函数代码建模与信息检查、上下文感知级联知识检索、两阶段约束驱动提示框架、以及多种大型语言模型（Qwen-QWQ-32B、DeepSeek-Coder-33B、OpenReasoning-Nemotron-32B 等）进行摘要生成。

**📊 数据集**

使用的数据集：工业级 HarmonyOS 代码摘要基准 HMSum‑12 与 HMSum‑13；社区项目基准 CodeSearchNet 与 C/C++ benchmark；以及手工筛选的 CodeSearchNet‑Verified 与 C‑Verified 两个验证集。

**📈 对比分析**

实验与基线对比方法：对比 PRIME、EP4CS、FSP（0/1/5-shot）和 ProConSuL 等现有方法，在 BLEU‑4、ROUGE‑L、SentBERT 等指标上进行评估；在社区基准上还使用 LLM 评判。实验结果显示，ExpSum 在 HMSum‑12 上 BLEU‑4 提升 26.71%、ROUGE‑L 提升 20.10%，在其他基准上亦提升 10‑20% 以上，并获得 86% 的专家认可率。

**⚠️ 局限性**

局限性：①仍有约 6.5% 的摘要未满足三大期望，主要涉及域词更新不及时和公式细节处理不足；②知识库需要人工维护更新；③对极端复杂或极小函数的类别推理仍存在误差；④方法在不同编程语言和项目风格中需要进一步验证。

---

## 535. Learning to Reason Faithfully through Step-Level Faithfulness Maximization

**arXiv ID:** 2602.03507 | [PDF](https://arxiv.org/pdf/2602.03507v1)

**作者:** Runquan Gui `[一作]` (University of Science and Technology of China), Yu Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 73730 | [OpenAlex ID](https://openalex.org/A5090802305)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FaithRL 框架，通过 RL 直接最大化推理的可信度，并在过程层面加入几何奖励与信度感知优势调制。

**💡 创新点**

创新点在于：① 将推理可信度作为正式优化目标并证明其避免过度自信与过度保守；② 设计基于基线能力的几何奖励，实现可信度与正确率的几何融合；③ 引入 Faithfulness‑Aware Advantage Modulation (FAAM)，在每一步骤层面监督推理是否严格基于证据。

**🔧 技术方法**

采用基于 GRPO 的强化学习框架，结合几何奖励、FAAM、证据验证器、THS（Truthful Helpfulness Score）等技术；训练时使用离线验证与在线分组采样。

**📊 数据集**

主要使用三大多跳推理数据集（2WikiMultiHopQA‑Full、HotpotQA‑Full、MuSiQue‑Full）进行训练和评估，并在 OOD 任务（GSM8k、MATH500）检验通用性。

**📈 对比分析**

与 prompting、refusal‑aware SFT、confidence‑based abstention、RLVR（GRPO、TruthRL）以及事实驱动 RL（FSPO、KnowRL）等方法对比；在所有基准上获得最高的 THS，正确率提升约 1.6%，幻觉率下降约 4.7%，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：① 需要依赖精确的证据验证器，验证误差会影响奖励；② 主要针对多跳 QA 任务，跨领域通用性仍需进一步验证；③ 计算开销略有提升（约 10–15%），且模型假设知识库完备性在实际场景中可能不成立。

---

## 536. Reparameterization Flow Policy Optimization

**arXiv ID:** 2602.03501 | [PDF](https://arxiv.org/pdf/2602.03501v1)

**作者:** Hai Zhong `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**通讯引用:** 3719 | [OpenAlex ID](https://openalex.org/A5082905458)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了 Reparameterization Flow Policy Optimization (RFO)，通过将可微分流模型与 Reparameterization Policy Gradients 结合，并加入两种正则化实现高样本效率的机器人控制算法。

**💡 创新点**

创新点在于首次将流模型与 RPG 对接，消除对不可计算的 log‑likelihood 的需求，并通过过去数据 Conditional Flow Matching 与均匀探索正则化以及动作块化扩展，显著提升训练稳定性与探索能力。

**🔧 技术方法**

采用可微分仿真器、Euler ODE 积分、Conditional Flow Matching、短期演员‑评论家 (SHAC) 框架、动作块化以及正则化技术。

**📊 数据集**

在 Rewarped、DFlex 等可微分仿真器中，对 Ant、ANYmal、Soft Jumper、Hand Reorient、Rolling Pin、Transport、Hand Flip 等多种刚体与软体机器人任务进行评估。

**📈 对比分析**

与 SOTA RPG（SAPO、SHAC）及流/扩散 RL（DrAC、FlowRL）做对比，RFO 在所有任务上均优于基线，Soft Jumper 的奖励提升近 2 倍，整体平均性能提升约 60%。

**⚠️ 局限性**

局限在于仍需可微分仿真器支持，流模型训练对数值稳定性要求高，动作块化在无专家数据下更难收敛，且未探讨离线到在线迁移的适用性。

---

## 537. Least but not Last: Fine-tuning Intermediate Principal Components for Better Performance-Forgetting Trade-Offs

**arXiv ID:** 2602.03493 | [PDF](https://arxiv.org/pdf/2602.03493v1)

**作者:** Alessio Quercia `[一作]` (Forschungszentrum Juelich), Hanno Scharr `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了低秩适配（LoRA）在使用主成分初始化时的性能与遗忘权衡，并提出使用中间主成分进行微调以实现更优的学习-遗忘平衡。

**💡 创新点**

创新点在于揭示了“U形”遗忘曲线，证明极端主成分（首尾）更易导致灾难性遗忘；并通过理论与实验证明中间主成分既能保持较高任务精度，又能显著降低遗忘量。

**🔧 技术方法**

采用SVD对预训练权重进行分解，构造基于不同主成分范围的LoRA矩阵，并结合低秩更新、学习率调度和梯度裁剪等技术实现微调；同时对比PiSSA、MiLoRA等主成分初始化方法。

**📊 数据集**

在视觉任务上使用ImageNet1k预训练的ViT‑B，微调至CIFAR10/100、DTD、Caltech101/256、Food101、Oxford Pets/Flowers、Stanford Cars/Dogs、FGVC Aircraft等；在NLP任务上微调LLaMA‑2 7B完成Python编码、数学推理和常识推理等。

**📈 对比分析**

与全量微调、LoRA、DoRA、PiSSA、MiLoRA等方法对比，实验显示所提中间主成分方案在多任务上获得最高的任务准确率与最低的遗忘率（如在Caltech101上Accuracy 169.25%、ImageNet1k遗忘 2.0%），并在高学习率设置下保持稳定性。

**⚠️ 局限性**

局限性包括：仅在固定秩和单一模型架构下验证；对动态秩分配或其他PEFT方法的适用性未作充分探讨；以及对极端任务分布或更大规模模型的泛化能力仍待进一步研究。

---

## 538. Detecting and Explaining Malware Family Evolution Using Rule-Based Drift Analysis

**arXiv ID:** 2602.03489 | [PDF](https://arxiv.org/pdf/2602.03489v1)

**作者:** Olha Jurečková `[一作]` (Czech Technical University in Prague), Martin Jureček `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5033203359)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于规则的可解释概念漂移检测框架，通过比较同一恶意软件家族原始样本与演化样本生成的规则集来识别并量化漂移。

**💡 创新点**

创新点在于将规则集相似度作为漂移量化指标，既能检测漂移，又能提供特定特征变化的可解释说明，弥补了传统黑箱方法对解释性的不足。

**🔧 技术方法**

技术包括使用RIPPER算法学习规则集、MAB-Malware生成对抗演化样本、基于Hamming距离的规则集相似度计算以及阈值决策来判定漂移。

**📊 数据集**

实验数据来源于RawMal‑TF恶意软件家族（Agensla、DCRat、Makoob、Mokes、Strab、Taskun）以及EMBER数据集的特征向量，使用LIEF提取PE文件静态特征。

**📈 对比分析**

通过比较原始家族内部规则集间的距离与原始家族与演化家族规则集间的距离，设定阈值后检测漂移，实验显示在六个家族和多维度特征下漂移检测准确率达92.08%。

**⚠️ 局限性**

局限性包括依赖已知家族标签、仅采用单一规则相似度度量、对不同演化策略的鲁棒性未充分评估，以及对规则集规模和计算开销的进一步优化仍待研究。

---

## 539. When Routing Collapses: On the Degenerate Convergence of LLM Routers

**arXiv ID:** 2602.03478 | [PDF](https://arxiv.org/pdf/2602.03478v1)

**作者:** Guannan Lai `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**通讯引用:** 3324 | [OpenAlex ID](https://openalex.org/A5065180062)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并解决LLM路由中的“路由崩塌”问题，提出了决策感知的排名学习框架EquiRouter并设计了评估指标Routing Collapse Index（RCI）。

**💡 创新点**

创新点在于发现路由崩塌主要源自目标–决策不匹配，提出直接学习模型排名而非预测分数的EquiRouter，并首次给出了量化路由崩塌的RCI指标。

**🔧 技术方法**

技术方法包括利用预训练编码器获取查询向量，采用FiLM调制的查询‑模型嵌入构造模型特定特征，使用列表/成对排序损失训练排名模型，以及轻量级模型嵌入共享参数实现高效推理。

**📊 数据集**

实验数据集为RouterBench（单模态多模型）和MMR‑Bench（多模态）。

**📈 对比分析**

与kNNRouter、MLPRouter、GraphRouter、EmbedLLM、AvengersPro、CausalRouter等多种基线在nAUC、QNC、RCI等指标上进行比较，EquiRouter在nAUC最高、RCI最低、QNC最低（成本比最强模型低约17%–25%），显著优于现有方法。

**⚠️ 局限性**

局限性在于仍需昂贵的离线标注、对更大模型池或不同任务的泛化未知、仅关注排名而非精确性能预测，以及对极端预算下多模型混合策略的进一步探索不足。

---

## 540. Reading Between the Code Lines: On the Use of Self-Admitted Technical Debt for Security Analysis

**arXiv ID:** 2602.03470 | [PDF](https://arxiv.org/pdf/2602.03470v1)

**作者:** Nicolás E. Díaz Ferreyra `[一作]` (Hamburg University of Technology), Riccardo Scandariato `[通讯]` (Hamburg University of Technology)

**通讯引用:** 3160 | [OpenAlex ID](https://openalex.org/A5012313708)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过混合方法研究了安全相关自我申报技术债（SSATD）如何补充静态分析工具（SAT）的检测结果，先对SSATD数据集进行手工标注与CWE映射，再结合Semgrep、Flawfinder和CWE Heuristics三款主流SAT进行比较，随后通过线上问卷调查收集安全从业者的使用体验。

**💡 创新点**

研究证明SSATD能补足SAT难以检测的CWE类别（如竞争条件、资源泄露等），并对开发者理解和修复SAT发现的安全缺陷提供显著帮助，首次系统量化了SSATD与SAT之间的互补关系。

**🔧 技术方法**

技术手段包括：①手工过滤并验证SSATD实例并映射至CWE；②使用Semgrep、Flawfinder与CWE Heuristics三款开源SAT进行默认配置扫描；③设计并发布问卷收集实践者对SAT与SSATD结合使用的感知。

**📊 数据集**

使用MADE‑WIC数据集的Big‑Vul分区，筛选出135个C/C++函数中的SSATD实例作为实验数据；同时利用这些实例对应的CWE作为比较基准。

**📈 对比分析**

通过将SSATD手工标注的33种CWE与SATs检测到的24种CWE进行匹配，发现SATs覆盖率为84%，但与SSATD的重叠仅为6.42%；问卷结果显示，SSATD在理解和修复SAT发现的漏洞（尤其CWE‑362）时具有统计显著的提升，表明其在实践中的有效性。

**⚠️ 局限性**

局限性包括：①仅使用默认配置的SAT，未探讨定制化对结果的影响；②样本仅来自C/C++开源项目，缺乏跨语言验证；③SSATD识别依赖人工关键词过滤，可能漏掉隐含安全信息；④问卷样本规模有限，且以Prolific平台参与者为主，可能存在样本偏差。

---

## 541. RAL-Bench: Benchmarking for Application-Level Functional Correctness and Non-Functional Quality Attributes

**arXiv ID:** 2602.03462 | [PDF](https://arxiv.org/pdf/2602.03462v1)

**作者:** Ruwei Pan `[一作]` (Chongqing University), Hongyu Zhang `[通讯]` (Chongqing University)

**通讯引用:** 14715 | [OpenAlex ID](https://openalex.org/A5100412608)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RAL-Bench，一个基于真实 GitHub 项目的应用级代码生成评估基准，涵盖功能正确性与 ISO/IEC 25010 的非功能质量属性。

**💡 创新点**

创新点在于将自然语言需求提炼、黑盒系统测试、参考实现过滤、AHP 权重合成等方法整合，构建可执行、可比对、覆盖功能与非功能的端到端评测框架。

**🔧 技术方法**

采用大语言模型零样本代码生成、黑盒系统测试、静态/动态分析（Maintainability Index、静态安全扫描、效率与资源监控）以及 AHP 进行属性加权等技术。

**📊 数据集**

使用了 38 个活跃 GitHub 开源项目（共 450+ 评价点），每个项目固定提交哈希，覆盖工具、数据、Web、安全等七类实际场景。

**📈 对比分析**

对 16 个 LLM 进行零样本贪心解码评测，功能正确率最高仅 45%，非功能得分相对较高，但整体功能仍为瓶颈；高成本“思考” LLM 并未显著提升。

**⚠️ 局限性**

局限在于评测仍受限于测试用例过滤、单轮生成、仅零样本；未涵盖多轮迭代或更复杂的应用场景，且生成质量受 LLM 与工程实践差距影响。

---

## 542. Hierarchical Concept-to-Appearance Guidance for Multi-Subject Image Generation

**arXiv ID:** 2602.03448 | [PDF](https://arxiv.org/pdf/2602.03448v1)

**作者:** Yijia Xu `[一作]` (Peking University), Jinshi Cui `[通讯]` (Peking University)

**通讯引用:** 1693 | [OpenAlex ID](https://openalex.org/A5113432906)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种层级化概念到外观引导（CAG）框架，用于多主体图像生成；在训练中通过VAE Dropout随机丢弃低层细节，使模型更依赖VLM语义；在推理中加入对应关系掩码注意力，将每个文本词限制在对应参考图像区域内，提升身份一致性与指令遵循度。

**💡 创新点**

创新点在于：①将高层语义指导与低层外观细节分层融合；②VAE Dropout策略让模型在缺失细节时仍能靠语义生成；③对应关系掩码注意力显式约束文本与参考区域的对应关系，解决传统DiT对多模态对应的隐式推理难题。

**🔧 技术方法**

核心技术包括：Vision‑Language Model（VLM）对文本和参考图像的联合编码；VAE对参考图像的低层特征编码；Diffusion Transformer（DiT）作为解码器；对应关系掩码注意力机制；VAE Dropout训练策略；以及多模态融合的自注意力设计。

**📊 数据集**

使用了约24k条包含多角色与场景参考的训练集，测试集为300个样本；训练时采用冻结VLM、微调DiT；对比数据集主要参考OmniGen2、UNO、Qwen‑Image‑Edit等现有方法。

**📈 对比分析**

在GPT‑4.1评估的Prompt Following（PF）和Subject Consistency（SC）指标上，CAG取得PF 7.308、SC 7.906、Overall 7.568的最高分，明显优于OmniGen2、UNO及基线Qwen‑Image‑Edit，展示了在指令遵循和身份一致性方面的显著提升。

**⚠️ 局限性**

局限性包括：对VLM的依赖性较高，若VLM无法充分捕捉细节则生成效果受限；需要预先提取词‑区域对应的bounding box，额外增加预处理成本；在极端多样化或极少量参考图像场景下，模型的鲁棒性和实时性仍待进一步验证。

---

## 543. DiscoverLLM: From Executing Intents to Discovering Them

**arXiv ID:** 2602.03429 | [PDF](https://arxiv.org/pdf/2602.03429v1)

**作者:** Tae Soo Kim `[一作]` (KAIST), Juho Kim `[通讯]` (SkillBench)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于在多轮对话中帮助用户逐步发现并细化其意图的框架（称为 IDo），并通过构造层级化用户模拟器为大语言模型提供奖励信号，以训练模型在探索与收敛之间取得平衡。

**💡 创新点**

创新点在于①将意图形成视为从抽象到具体的递进过程；②设计了基于意图层级的用户模拟器，使模型可在训练时获得“意图发现”奖励；③构造了奖励函数将意图发现与交互效率统一为一条训练目标。

**🔧 技术方法**

核心技术包括：
- 基于 LLM 的用户模拟器（使用 Gemini 3 Flash）
- 意图层级树的自动构造（使用 Claude Sonnet 4.5）
- 训练方法：SFT、DPO、SFT+DPO、SFT+DPO+GRPO
- 评估采用 LLM‑judge（GPT‑5.1）和交互性指标、意图发现/满意度等。

**📊 数据集**

使用三类创作任务数据集：
- Creative Writing（来自 r/WritingPrompts，约 500 训练 / 100 测试）
- Technical Writing（新闻文章）
- SVG Drawing（可视化绘图）
同时在实验中还验证了对未见任务（旅游计划、数据可视化代码、科研摘要等）的迁移性能。

**📈 对比分析**

与基线（原始模型、加提示、CollabLLM）比较，采用四项指标：意图发现率、意图满意度、交互性评分、平均 token 数。结果显示 IDo 在意图发现率上提升约 10%，意图满意度和交互性显著提高，且平均 token 数下降 30‑40%。用户研究（75 名参与者）也表明 IDo 在交互满意度和完成效率上优于基线。

**⚠️ 局限性**

局限性包括：
- 模拟器假设意图单向递进且由模型触发，缺乏用户回溯与外部因素；
- 生成的意图树和探索选项多样性不足，可能导致过度相似或缺乏创意；
- 评估主要基于 LLM‑judge，可能受模型偏差影响；
- 安全性评估有限，尚需更广泛的对抗性测试。

---

## 544. Lookahead Path Likelihood Optimization for Diffusion LLMs

**arXiv ID:** 2602.03496 | [PDF](https://arxiv.org/pdf/2602.03496v1)

**作者:** Xuejie Liu `[一作]` (Peking University), Anji Liu `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于路径对数似然（Path LL）和POKE估计器的动态解码策略POKE‑SMC，用于优化扩散LLM的解码路径。

**💡 创新点**

创新点在于将路径LL作为全局目标，设计了对数似然估计器POKE并将其嵌入SMC搜索，显著提升解码质量。

**🔧 技术方法**

使用了扩散语言模型（LLaDA）、路径对数似然估计、Sequential Monte Carlo搜索、随机滚动评估等技术。

**📊 数据集**

使用了LLaDA‑8B‑Instruct、LLaDA‑1.5‑8B模型在GSM8K、MATH500、HumanEval、MBPP、Countdown、Sudoku等推理任务数据集。

**📈 对比分析**

与多种基线（Uniform、Confidence、Entropy、Margin、EB‑Sampler、Semi‑AR、PC‑sampler、Majority Voting、E‑SMC、ReMDM）对比，POKE‑SMC在相同计算预算下平均提升约3%准确率，并在效率‑准确率Pareto前沿上更优。

**⚠️ 局限性**

局限性包括仅在8B规模模型上验证，且采用固定间隔重采样，未探讨更大模型和自适应重采样策略。

---

## 545. Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance

**arXiv ID:** 2602.03491 | [PDF](https://arxiv.org/pdf/2602.03491v1)

**作者:** Yingjie Zhu `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 59770 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一种针对大型视觉语言模型（LVLM）的表格推理方法，能够在几乎不需要额外注释或外部工具的情况下显著提升表格理解与推理性能。

**💡 创新点**

创新点在于将表格结构抽象与语义对齐分离（DiSCo框架），以及引入全局-局部结构引导的推理流程（Table-GLS），两者共同实现了高效、可解释且对未知表格结构具有鲁棒性的推理机制。

**🔧 技术方法**

技术上使用了结构对齐（只关注布局的标记化表示）、语义对齐（全局描述和局部查询）、以及三阶段推理策略（全局结构探索 → 结构自检子表提取 → 证据驱动推理）等，并通过LoRA微调、vLLM加速推理。

**📊 数据集**

数据集包括MMTab、TTT、TabPedia、SynTab等共21项表格理解与推理任务，以及10K表格图像用于对齐；此外还评估了ScienceQA、CRPE、HallusionBench和TextVQA等非表格基准。

**📈 对比分析**

与传统基于文本序列对齐（HTML/Markdown/LaTeX）以及多种公开的LVLM（TableLlama、Table-LLaVA、Qwen3-VL、Gemma3n-E4B）相比，DiSCo+Table-GLS在多数任务上获得显著提升（尤其是结构敏感任务和OOD场景），在部分基准甚至可与经过专门优化的模型持平或超越，且仅使用10K图像即可接近全量（97K）对齐的效果。

**⚠️ 局限性**

局限性包括：对极度复杂或嵌套表格的子表提取仍可能出现误差；在缺乏足够全局结构描述时，推理步骤可能受限；此外，虽然不需要大量注释，但仍需至少10K的表格图像进行对齐，且方法在跨域非表格任务上的通用性虽好，但对某些视觉语言任务的提升有限。

---

## 546. Preferences for Idiomatic Language are Acquired Slowly -- and Forgotten Quickly: A Case Study on Swedish

**arXiv ID:** 2602.03484 | [PDF](https://arxiv.org/pdf/2602.03484v1)

**作者:** Jenny Kunz `[一作]` `[通讯]` (Linkoping University), Jenny Kunz (Linkoping University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了瑞典语大语言模型在预训练与从英语迁移过程中的习得性别化表达偏好，并通过最小对照句对模型进行评估。

**💡 创新点**

首次构建了两套瑞典语习得性别化评测数据集（习惯成语对照与翻译歪曲对照），系统性探讨习得速度与机器翻译指令微调对习得性别化表达的影响。

**🔧 技术方法**

采用FineWeb2瑞典语数据进行从零训练与英语预训练模型的继续预训练；使用机器翻译后的指令数据进行指令微调；通过最小对照句的困惑度（perplexity）进行偏好评估。

**📊 数据集**

训练集：FineWeb2（约250亿词）；评测集：自研的Idioms_all/Idioms_challenge、Translationese_all/filtered；现有的DaLAJ、ScaLA；对比模型包括SmolLM2 135M（从零与继续预训练）和AI Sweden Llama 8B。

**📈 对比分析**

在各训练检查点对上述评测集的准确率进行比较；习惯成语和翻译歪曲任务的学习曲线较慢，最终准确率约为88‑90%；指令微调后习得性别化表达准确率大幅下降（接近随机），而语法与形态学任务基本保持不变。

**⚠️ 局限性**

局限性：评测数据集规模有限且部分数据受版权限制；研究聚焦瑞典语及小模型，未涵盖更大模型或其他低资源语言；训练数据主要为书面文本，可能低估口语化习得效果。

---

## 547. Recursive Energy Efficient Agreement

**arXiv ID:** 2602.03474 | [PDF](https://arxiv.org/pdf/2602.03474v1)

**作者:** Shachar Meir `[一作]` (Weizmann Institute of Science), David Peleg `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 18939 | [OpenAlex ID](https://openalex.org/A5056419022)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出了一种递归的能量高效协议，在同步睡眠模型下实现 Crash Agreement 与 Byzantine Agreement，且每个参与方的待机轮数仅为 O(log f)（或 O(log n)），总轮数为 O(f)（或 O(n)）。

**💡 创新点**

创新点在于将 Momose‑Ren 的递归分区技术与能量效率结合，得到在任意 crash 故障数下仍可实现的 O(log f) awake 复杂度；并构造了一种无数字签名实现，适用于 f < n/3 的 Byzantine 场景；此外通过子集并行化进一步将复杂度降至 O(f) 轮次。

**🔧 技术方法**

主要技术包括：
- 同步睡眠模型下的能量耗费度量（awake 复杂度）；
- 递归分区与信息传播（dissemination）机制；
- 通过两轮投票确认的 GBA 方案；
- 子集并行化与最大值聚合的优化步骤。

**📊 数据集**

本工作为理论性研究，无使用具体数据集；所有结果均来自算法分析与证明。

**📈 对比分析**

与先前的算法相比：
- 对于 f ≤ √n·g(n)（g(n)=o(log n)）时，Meir 等的方案在 awake 上更优；
- 当 f 较大时，本文方案的 O(log f) awake 复杂度优于 O(log n)；
- 对于 Byzantine，本文提供了 f < n/3 的无签名实现，满足 O(log f) awake 和 O(f) 轮数，优于已有的 O(n) 轮数方案。

**⚠️ 局限性**

局限性包括：
- 证明仅适用于 crash 与 f < n/3 的 Byzantine 情况；
- 对低 fault 数的情况下无法突破已有的 lower bound；
- 仍缺乏对最优 awake 下界的完整证明；
- 对非同步或带有通信延迟的模型适用性未知。

---

## 548. Scaling Continual Learning with Bi-Level Routing Mixture-of-Experts

**arXiv ID:** 2602.03473 | [PDF](https://arxiv.org/pdf/2602.03473v1)

**作者:** Meng Lou `[一作]` (University of Hong Kong), Yizhou Yu `[通讯]` (University of Hong Kong)

**通讯引用:** 18898 | [OpenAlex ID](https://openalex.org/A5108557359)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CaRE，一种基于预训练模型的连续学习框架，采用双层路由混合专家（BR‑MoE）实现每层动态知识检索与融合；

**💡 创新点**

创新点在于双层路由机制：先用任务感知类感知器根据熵挑选Top‑M路由器，再由路由器动态激活并聚合Top‑K专家，既获得判别性又兼顾全面性；

**🔧 技术方法**

技术包括预训练 Vision Transformer、参数高效 Adapter、熵驱动路由、KL 监督、EMA 更新共享专家、以及针对每层的动态路由与专家聚合；

**📊 数据集**

使用 OmniBenchmark‑V2（1000类、≈190k图像）作为长序评测，还在 CIFAR‑100、ObjectNet、ImageNet‑R、ImageNet‑A、VTAB 等标准数据集做短序实验；

**📈 对比分析**

与 10+ 传统 PTM‑based CIL 方法（如 L2P、DualPrompt、EASE、TUNA、MOS、MIN 等）对比，在 100–301 任务长序和 5–20 任务短序均取得显著优势，最后准确率提升 5–10% 以上；

**⚠️ 局限性**

局限性是模型随任务数线性增长，需要为每个新任务添加新的路由器/专家/感知器，导致参数规模随任务数扩张。

---

## 549. Inlier-Centric Post-Training Quantization for Object Detection Models

**arXiv ID:** 2602.03472 | [PDF](https://arxiv.org/pdf/2602.03472v1)

**作者:** Minsu Kim `[一作]` (Korea Advanced Institute of Science and Technology), Junmo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 9405 | [OpenAlex ID](https://openalex.org/A5100606266)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 InlierQ，一种后训练量化方法，通过分离检测模型的异常激活与有意义激活来提升量化质量。

**💡 创新点**

创新点在于利用梯度感知的体积显著性评分构建异常概率分布，并采用 EM 算法得到两组高斯混合模型，从而在量化时仅关注内点集合，抑制噪声影响。

**🔧 技术方法**

主要技术包括梯度感知体积显著性评分、两组高斯混合模型与 EM 估计、任务特定的热图 Top‑K 采样、基于 min‑max 的量化参数校准以及对 4‑bit 激活的量化优化。

**📊 数据集**

使用的公开数据集为 COCO（2D 目标检测）和 nuScenes（3D 目标检测，包含摄像头和激光雷达两种传感器）。

**📈 对比分析**

与 BRECQ、LiDAR‑PTQ 等基线在相同模型与校准设置下对比，InlierQ 在 4‑bit 激活下 2D mAP 提升约 0.4%，3D mAP 提升约 3.2%，低位宽场景性能尤为突出。

**⚠️ 局限性**

局限性包括对阈值 τ 与热图 Top‑K 参数的敏感性、异常与内点分布相似导致去除异常不完全，以及在极低位宽下仍存在一定误差。

---

## 550. Beyond Variance: Prompt-Efficient RLVR via Rare-Event Amplification and Bidirectional Pairing

**arXiv ID:** 2602.03452 | [PDF](https://arxiv.org/pdf/2602.03452v1)

**作者:** Xin Sheng `[一作]` (Beijing University of Post and Telecommunications), Yong Ma `[通讯]` (QI-ANXIN Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在极低数据量下的 RLVR 训练策略：通过正负配对（hard‑but‑solvable prompt 与 easy‑but‑brittle prompt）和 Weighted GRPO 以双向教学信号提高大型语言模型的数学推理能力。

**💡 创新点**

创新点在于①把 prompt 选择视为机制级别的“正负配对”而非仅靠方差或难度；②使用 Weighted GRPO 对二元奖励进行组归一化，自动放大稀有成功与失败的优势；③仅用两条固定 prompt 即可在多项数学基准上逼近甚至超越使用1209条 prompt 的大规模 RLVR。

**🔧 技术方法**

技术包括：
- RLVR（强化学习可验证奖励）
- Weighted GRPO（对二元奖励加权后进行组归一化的策略梯度方法）
- Prompt 轻量级探测（估计成功率并筛选正负配对）
- 典型的 Pass@k 评估与 unbiased estimator。

**📊 数据集**

数据集：
- 训练 prompt 来源于 AIME 2025 与 DeepScaleR‑sub（1209 条子集）。
- 评估基准包括 AIME 2025、AMC23 与 MATH500。
- 采用精确答案检验作为可验证奖励。

**📈 对比分析**

对比方法：
- 基线：GRPO 与两条历史方差最高的 prompt；
- 大规模 RLVR：GRPO 与 1209 条 prompt。
- 结果：
  - AIME 2025 Pass@8 由 16.8 提升至 22.2；
  - AMC23 Pass@64 由 94.0 提升至 97.0；
  - MATH500 在多 k 维度上亦表现出稳健提升。
  仅使用两条 prompt 即可实现与大规模训练相当甚至更优的性能。

**⚠️ 局限性**

局限性：
- 方案针对二元可验证奖励，可能难以直接迁移到多类或连续奖励任务；
- 仍需进行 prompt 探测与成功率估计，略有计算开销；
- 只在数学推理基准上验证，缺乏在更广泛任务（如对话、文本生成）的评估；
- 当 prompt 质量或分布发生变化时，正负配对的效果可能需要重新调整。

---

## 551. HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic

**arXiv ID:** 2602.03447 | [PDF](https://arxiv.org/pdf/2602.03447v1)

**作者:** Yu-Hsiang Chen `[一作]` (National Yang Ming Chiao Tung University), Yi-Ting Chen `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 10098 | [OpenAlex ID](https://openalex.org/A5100348260)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了HetroD数据集与统一基准，涵盖了17.5小时无人机高清视角、65.4k个高精度轨迹、HD地图与交通灯状态，专注于多种VRU复杂行为（如钩转、车道穿越等）。

**💡 创新点**

创新点在于：①无人机视角实现全局无遮挡观测；②厘米级轨迹标注与多类型VRU覆盖；③提供统一工具包，可直接转换为主流框架格式；④构建跨数据集评测协议与标准指标。

**🔧 技术方法**

采用无人机摄像+相机标定+深度检测网络+跟踪+Kalman滤波+轨迹后处理+HD地图生成与校准；同时开发跨数据集转换工具（支持ScenarioNet、GPUDrive等）。

**📊 数据集**

主要使用HetroD自有数据集；与现有NuScenes、Waymo、SinD等对比，用于跨域评测。

**📈 对比分析**

评估方法：对MTR与Wayformer进行跨域Brier‑FDE对比；对IDM、PDM‑Closed规划器在HetroD车辆场景下进行闭环非反应评测，关注NuPlanScore、TTCWithin、At‑Fault碰撞率。结果显示：预测模型在HetroD上误差显著升高，规划器出现明显性能下降，尤其VRU侧向碰撞率显著上升。

**⚠️ 局限性**

限制：尚未覆盖更大城市范围与多模态传感；规划器缺乏对VRU侧向交互的成本模型；预测模型对高密度多agent场景与跨域迁移仍表现不足。

---

## 552. CRL-VLA: Continual Vision-Language-Action Learning

**arXiv ID:** 2602.03445 | [PDF](https://arxiv.org/pdf/2602.03445v1)

**作者:** Qixin Zeng `[一作]` (University of Southampton), Chao Huang `[通讯]` (University of Southampton)

**通讯引用:** 12976 | [OpenAlex ID](https://openalex.org/A5042083053)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CRL‑VLA框架，使视觉‑语言‑动作(VLA)模型能够在连续任务流中实现持续学习，解决旧技能保持与新技能获取的稳定性‑可塑性权衡问题。

**💡 创新点**

通过理论分析将目标条件优势幅度作为稳定性‑可塑性两端的关键量，并设计双重评论器（冻结与可训练）实现不对称正则化，从而在不牺牲可塑性的前提下抑制遗忘。

**🔧 技术方法**

采用PPO强化学习、KL正则化、目标条件价值函数（GCVF）双头评论器、Monte‑Carlo返回估计以及价值一致性与动作约束损失，实现对优势幅度和策略散度的异步控制。

**📊 数据集**

在LIBERO benchmark（OpenVLA‑oft模型）上进行实验，使用标准的单任务和多任务设置。

**📈 对比分析**

与SL、LWF、ER、MTL等经典持续学习基线对比，CRL‑VLA在单任务和多任务场景下均在FAR、BWT、FT等指标上优于基线，显著降低遗忘并提升前向迁移。

**⚠️ 局限性**

局限性在于仅验证于结构化任务流，未探讨部分可观测性、非结构化语言目标及更动态环境；对大型VLA模型的冻结策略依赖较强，可能不适用于所有应用场景。

---

## 553. Ontology-to-tools compilation for executable semantic constraint enforcement in LLM agents

**arXiv ID:** 2602.03439 | [PDF](https://arxiv.org/pdf/2602.03439v1)

**作者:** Xiaochi Zhou `[一作]`, Markus Kraft `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型与工具调用，将本体规范编译成可执行工具，在生成阶段直接强制执行语义约束，从而构建金属有机多面体合成知识图谱。

**💡 创新点**

提出将T-Box本体转化为可调用工具的编译框架，使LLM在生成时即遵守语义约束，避免传统的后期校验与手工对齐，提高知识图谱构建的自动化与一致性。

**🔧 技术方法**

大型语言模型（LLM）+ Model Context Protocol（MCP）工具调用 + 本体编译机制 + 知识图谱构建与验证流程。

**📊 数据集**

30篇金属有机多面体（MOP）合成相关科研论文，包含全文、表格与补充信息。

**📈 对比分析**

与人工标注基准对比，微平均F1 0.826，步骤提取F1 0.843，化学实体F1 0.736；去除约束反馈的消融实验显示步骤F1下降至0.572，验证了约束反馈的重要性。

**⚠️ 局限性**

仅在单一本体与单一领域（MOP）验证，数据集规模有限；化学实体召回仍偏低，且缺乏跨领域与更大规模数据集的泛化评估。

---

## 554. Origin Lens: A Privacy-First Mobile Framework for Cryptographic Image Provenance and AI Detection

**arXiv ID:** 2602.03423 | [PDF](https://arxiv.org/pdf/2602.03423v1)

**作者:** Alexander Loth `[一作]` (Frankfurt University of Applied Sciences), Marc-Oliver Pahl `[通讯]` (IMT Atlantique)

**通讯引用:** 882 | [OpenAlex ID](https://openalex.org/A5004198506)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Origin Lens，移动端隐私优先的图像可追溯与 AI 检测框架；

**💡 创新点**

在设备本地实现多层防御（C2PA 认证、元数据指纹、隐写水印及可选反向搜索），并提供分级可信度指示，避免云端依赖；

**🔧 技术方法**

使用 Rust 进行安全的 JUMBF 解析与 X.509 链验证，Flutter UI，FFI 桥接，SHA‑256 哈希，EXIF/IPTC 解析，SynthID 水印检测，异步 I/O 等技术；

**📊 数据集**

主要基于公开的 C2PA 验证样本及生成模型参数日志，评估以 iPhone 15 Pro 处理的实测图像为准；

**📈 对比分析**

评估方式以本地处理时延为主，C2PA 验证 <500 ms，EXIF 解析 <50 ms；与云端检测对比未做实验，但延迟更低且无需网络请求；

**⚠️ 局限性**

受限于 C2PA 生态依赖、易受元数据剥离和模拟孔攻击，逆向搜索需网络且默认关闭，元数据检测对模型演化敏感，缺乏大规模对比实验。

---

## 555. Semantic Routing: Exploring Multi-Layer LLM Feature Weighting for Diffusion Transformers

**arXiv ID:** 2602.03510 | [PDF](https://arxiv.org/pdf/2602.03510v1)

**作者:** Bozhou Li `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14684 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了语义路由框架，动态融合多层LLM特征以提升Diffusion Transformer的文本条件效果

**💡 创新点**

提出统一的可解释凸融合框架，支持时序、深度及联合路由，并发现深度路由最优，同时诊断时序融合失效原因

**🔧 技术方法**

使用轻量级门控(Time‑conditioned fusion gate)、LayerNorm、softmax凸组合、sinusoidal编码、流匹配DiT、Qwen3‑VL文本编码器与Stable Diffusion 3 VAE

**📊 数据集**

在LAION‑400M子集（约3000万图文对）上训练，文本采用Qwen3‑VL‑32B生成的合成描述

**📈 对比分析**

与单层、均值、静态融合以及FuseDiT基线在GenAI‑Bench、GenEval、UnifiedReward上对比，深度路由在多项指标上提升约+5.5点（计数任务+9.97点），而单纯时序路由导致性能下降

**⚠️ 局限性**

主要局限在于纯时间路由因训练‑推理SNR漂移而失效，缺乏真正轨迹感知融合；对多样性与偏差影响未做深入探索

---

## 556. Explaining the Explainer: Understanding the Inner Workings of Transformer-based Symbolic Regression Models

**arXiv ID:** 2602.03506 | [PDF](https://arxiv.org/pdf/2602.03506v1)

**作者:** Arco van Breda `[一作]` (University of Amsterdam), Erman Acar `[通讯]` (University of Amsterdam)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5042454545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对Transformer基于符号回归模型进行机理可解释性分析，发现并验证了28个用于特定运算符的电路。

**💡 创新点**

引入基于CMA-ES的演化电路发现算法PATCHES，并提出功能完整性、完整性与最小化的严格评估标准，首次系统展示SR模型的可解释电路。

**🔧 技术方法**

使用激活补丁（Mean/Resample）、Covariance Matrix Adaptation Evolution Strategy、功能与模型层评估、探针验证及对比直觉式logit归因技术。

**📊 数据集**

基于NeSymReS模型的合成数学表达式数据集，分别为500条训练/验证样本和400条测试样本，聚焦单一运算符的表达式。

**📈 对比分析**

与传统迭代补丁、Mean/Resample补丁以及直接Logit归因方法对比，PATCHES在功能完整性上达46%（28中13通过），电路规模平均约40–78%，且在功能评估中几乎达到全模型的准确率。

**⚠️ 局限性**

对多符号电路的探索有限；补丁策略对数据分布敏感；结果仅在单一SR模型NeSymReS上验证，跨模型通用性尚待考察。

---

## 557. IntentRL: Training Proactive User-intent Agents for Open-ended Deep Research via Reinforcement Learning

**arXiv ID:** 2602.03468 | [PDF](https://arxiv.org/pdf/2602.03468v1)

**作者:** Haohao Luo `[一作]` (Alibaba Group), Ying Shen `[通讯]` (Alibaba Group)

**通讯引用:** 1712 | [OpenAlex ID](https://openalex.org/A5005535444)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

暂无信息

**💡 创新点**

暂无信息

**🔧 技术方法**

暂无信息

**📊 数据集**

暂无信息

**📈 对比分析**

暂无信息

**⚠️ 局限性**

暂无信息

---

## 558. Causal Inference on Networks under Misspecified Exposure Mappings: A Partial Identification Framework

**arXiv ID:** 2602.03459 | [PDF](https://arxiv.org/pdf/2602.03459v1)

**作者:** Maresa Schröder `[一作]` (LMU Munich), Nathan Kallus `[通讯]` (Cornell University)

**通讯引用:** 3134 | [OpenAlex ID](https://openalex.org/A5036921114)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在网络干预研究中，对曝光映射（exposure mapping）可能被错误指定时的部分识别框架，给出潜在结果和处理效应的尖锐上下界。

**💡 创新点**

创新点在于：①将曝光映射误差建模为分布偏移，利用灵敏度分析得到可计算的尖锐界限；②设计了正交（orthogonal）估计器，使得估计误差仅为二阶；③对三种常见曝光映射（加权邻居均值、阈值映射、高阶间接影响）给出了通用的界限与估计方法。

**🔧 技术方法**

使用了：因果敏感性分析、正交双阶段估计（cross‑fitting + 高阶修正），核平滑处理连续曝光水平，理论上证明了正则性、尖锐性与渐进正则性（quasi‑oracle）等性质。

**📊 数据集**

实验采用模拟网络数据：小规模 1000 节点，单维协变量；大规模 6000 节点，6 维协变量，随机生成网络结构与处理分配。

**📈 对比分析**

与传统的插值（plug‑in）估计器对比；正交估计器在小样本时已具备有效性，且收敛速度快；对不同灵敏度因子验证界限的有效性；宽度小于总结果范围的 10%，并且始终不包含零，满足决策需求。

**⚠️ 局限性**

局限性：仅在模拟数据上验证；需要预先给定曝光映射误差上界（灵敏度参数）；目前只对三种曝光映射给出实例，对更复杂或不规则的曝光映射需进一步扩展；对连续曝光时核带宽的选择仍依赖经验。

---

## 559. Game-Theoretic and Algorithmic Analyses of Multi-Agent Routing under Crossing Costs

**arXiv ID:** 2602.03455 | [PDF](https://arxiv.org/pdf/2602.03455v1)

**作者:** Tesshu Hanaka `[一作]` (Kyushu University), Hirotaka Ono `[通讯]` (Nagoya University)

**通讯引用:** 3925 | [OpenAlex ID](https://openalex.org/A5102003030)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了多代理路由的跨越成本模型（Crossing Cost Multi-Agent Routing，CC-MAR），并在异步环境下将冲突视为软成本（跨越成本乘积）。

**💡 创新点**

创新点在于把冲突转化为跨越成本的乘积，并结合博弈论证明纯纳什均衡存在与可达性，同时给出多参数化算法，突破了Steiner Orientation 的 NP‑hard 性。

**🔧 技术方法**

采用了网络拥塞游戏理论、潜在函数、动态规划、参数化复杂度技术以及对 Steiner 方向化简的运用。

**📊 数据集**

论文未给出实验数据集，全部以理论分析为主。

**📈 对比分析**

通过理论证明，纯纳什均衡在权重有限时可在多项式时间求解，普遍为 PLS‑完备；最优解 NP‑完备，但在多种参数化（如 |A|+k、|E|、顶点覆盖数等）下得到 FPT/XP 算法，显示在结构受限实例上可高效求解。

**⚠️ 局限性**

局限在于一般情况仍为 NP‑hard/PLS‑完备，对参数 |A| 单独未能得到 FPT；且缺乏对大规模实际实例的实验评估与近似算法。

---

## 560. Failure is Feedback: History-Aware Backtracking for Agentic Traversal in Multimodal Graphs

**arXiv ID:** 2602.03432 | [PDF](https://arxiv.org/pdf/2602.03432v1)

**作者:** Joohyung Yun `[一作]` (POSTECH), Wook-Shin Han `[通讯]` (POSTECH)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 FiF，一个基于大型语言模型的多模态文档检索框架，通过将图遍历建模为具有经济理性和失败反馈的序列决策过程，实现了更精确、可适应的多跳检索。

**💡 创新点**

创新点：①历史感知回溯机制，将失败信息用于重定位搜索；②经济理性策略升级，动态在低成本向量检索与高成本LLM推理之间切换；③将检索与规划分离的Agentic工作流，在多模态层次图上实现上下文感知的子查询生成与路径规划。

**🔧 技术方法**

技术包括：层次化组件图（文档、组件、子组件层），LLM驱动的多策略遍历器、子查询规划器、评估器与重排序器；经济理性决策逻辑；回溯时利用历史检索记录；使用MM-Embed作为统一多模态嵌入；OpenAI GPT‑5作为LLM推理引擎。

**📊 数据集**

数据集：MultimodalQA、MMCoQA、WebQA 三大多模态检索基准；采用URL标注的网页式语料，分别包含 3235、453、7662 页，平均每页 37、32、13 个多模态组件。

**📈 对比分析**

与 LILaC、IRCoT、VisRAG、ColPali 等传统与Agentic检索方法对比，在所有三个基准上均取得 Recall@10、MRR@10 的最高分，平均 Recall@10 提升 22.03%，MRR@10 提升 18.60%；在端到端 QA 任务中 EM/F1 同样领跑，提升约 8–17%；相对最强 Agentic 基线 IRCoT，检索准确率提升 12.9%/9.3%，但略高于其的 API 费用与推理时间。

**⚠️ 局限性**

局限性：依赖昂贵的 LLM 推理，导致成本与延迟相对较高；回溯与策略升级的超参数（如阈值、回溯深度）需要手工调优；在极大规模或低连接度的文档图中，仍可能出现死循环或无法有效跳转到远程相关节点的情况。

---

## 561. ConsistentRFT: Reducing Visual Hallucinations in Flow-based Reinforcement Fine-Tuning

**arXiv ID:** 2602.03425 | [PDF](https://arxiv.org/pdf/2602.03425v1)

**作者:** Xiaofeng Tan `[一作]` (Southeast University), Feng Zheng `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5792 | [OpenAlex ID](https://openalex.org/A5063285882)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对流模型的强化学习微调（RFT）中出现的视觉幻觉问题，提出 ConsistentRFT 框架进行抑制

**💡 创新点**

创新点在于：①动态粒度滚动（DGR）平衡细粒度与粗粒度优化，②一致性策略梯度优化（CPGO）保持模型向量场的一致性

**🔧 技术方法**

技术手段包括流匹配模型、SDE/ODE采样、DDPO/DPO/GRPO 等策略梯度方法，以及自定义的 VH‑Evaluator 评估指标

**📊 数据集**

使用公开文本到图像数据集 HPS‑v2 及 FLUX.1 dev 进行训练与评估

**📈 对比分析**

与现有 RFT 方法比较，ConsistentRFT 在低阶幻觉下降 49%、高阶幻觉下降 38%，在 FLUX1.dev 上提升 5.1% 的 out‑of‑domain 分数，同时保持或提升美学与偏好分数

**⚠️ 局限性**

局限性：受限于奖励模型的表达能力，未来需更强大、可扩展的奖励网络

---

## 562. On (Im)possibility of Network Oblivious Transfer via Noisy Channels and Non-Signaling Correlations

**arXiv ID:** 2602.03421 | [PDF](https://arxiv.org/pdf/2602.03421v1)

**作者:** Hadi Aghaee `[一作]`, Holger Boche `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供了IEEEtran LaTeX模板的使用说明和常见文档元素的代码示例，帮助作者快速编写符合IEEE排版规范的论文。

**💡 创新点**

提出了简化的使用手册、详细的模板选项说明以及针对不同期刊、会议的配置示例，降低了作者的学习成本。

**🔧 技术方法**

主要使用LaTeX语言与IEEEtran宏包，结合标准的章节、图表、引用等命令。

**📊 数据集**

无（本文为使用指南，无实验或数据集）。

**📈 对比分析**

无对比实验，仅给出示例代码和排版效果；未进行性能评估。

**⚠️ 局限性**

仅适用于IEEEtran模板的排版指导，未涉及科研内容；缺乏对不同模板兼容性、跨平台编译等细节的深入讨论。

---

## 563. DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs

**arXiv ID:** 2602.03495 | [PDF](https://arxiv.org/pdf/2602.03495v1)

**作者:** Zeyu Zhu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jian Cheng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 DALI 框架，在本地 PC 上通过动态专家分配、基于残差的预取和工作负载感知缓存，实现高效的 MoE 模型推理。

**💡 创新点**

核心创新点包括：1）贪心动态 CPU‑GPU 专家分配策略；2）利用跨层残差提升预取精度的 Residual‑Based Prefetching；3）基于历史工作负载的 Workload‑Aware Cache Replacement。

**🔧 技术方法**

技术手段包括 0‑1 整数规划与贪心求解、残差校正的特征预取、工作负载加权缓存置换、基于 KTransformers 的实现与 CUDA 异步流管理。

**📊 数据集**

评估使用了 WikiText 1K 作为残差向量校准集，C4 与 WikiText 语料用于推理速度测试，模型涵盖 DeepSeek‑V2‑Lite、Qwen‑3‑30B‑A3B 与 Mixtral‑8x7B。

**📈 对比分析**

与 llama.cpp、KTransformers、MoE‑Lightning、HybriMoE 等基线比较，prefill 阶段平均提升 7.62×、3.80×、2.45×、2.00×，解码阶段平均提升 3.97×、2.16×、1.48×、1.32×，显著优于现有方案。

**⚠️ 局限性**

局限性在于仅针对单 GPU 个人电脑环境，PCIe 4.0 带宽受限；预取与缓存参数需手动调优；未在多 GPU 或高带宽服务器上全面验证，模型规模与极端动态工作负载的鲁棒性待进一步探究。

---

## 564. Contextualized Visual Personalization in Vision-Language Models

**arXiv ID:** 2602.03454 | [PDF](https://arxiv.org/pdf/2602.03454v1)

**作者:** Yeongtak Oh `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**通讯引用:** 12584 | [OpenAlex ID](https://openalex.org/A5086877012)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出CoViP框架，利用个性化图像标题生成与强化学习后训练，使视觉语言模型能够依据用户的历史视觉经验生成个性化响应。

**💡 创新点**

创新点在于将个性化图像标题作为通用代理任务，通过可验证奖励（含视觉识别与检索）进行RL后训练，并在推理时采用标题增强生成（CAG）以提升下游个性化任务性能。

**🔧 技术方法**

使用的技术包括视觉语言模型（VLM）、基于GSPO的强化学习、标题增强生成、LLM驱动的多项选择问答评估以及多任务诊断评测。

**📊 数据集**

数据集主要为自研的个性化图像标题基准（约2.8K训练/1.3K测试样本）以及三类诊断任务（Last‑Seen Detection、Last‑Action Recall、Instruction‑Triggered Recall），所有数据均基于生成图像和自动生成的对话。

**📈 对比分析**

与开源和专有VLM、RAP、RePIC等基线进行对比，CoViP在标题生成上提升约40%，在诊断任务上保持一致的性能提升，且CAG进一步放大效果，整体优于现有方法。

**⚠️ 局限性**

局限性包括数据主要为合成图像与对话，可能存在事实或视觉不一致；缺乏真实用户长期交互日志；且在隐私保护方面尚未提供具体方案。

---

## 565. Exploiting Multi-Core Parallelism in Blockchain Validation and Construction

**arXiv ID:** 2602.03444 | [PDF](https://arxiv.org/pdf/2602.03444v1)

**作者:** Arivarasan Karmegam `[一作]` (IMDEA Networks Institute), Antonio Fernández Anta `[通讯]` (IMDEA Software Institute)

**通讯引用:** 4509 | [OpenAlex ID](https://openalex.org/A5014057564)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究区块链验证器在多核 CPU 上的并行执行，提出已排序块的调度（OBS）和内存池事务的选取与调度（PBC）两种优化问题，并给出精确的 MILP 模型和可扩展的确定性贪心启发式。

**💡 创新点**

① 将确定性执行与冲突约束结合成精确的多目标调度模型；② 同时考虑事务选择与并行调度的 PBC；③ 提出可扩展的贪心启发式，并与 Solana 声明访问方式及奖励贪心 baseline 对比；④ 用真实以太坊主网轨迹进行大规模实验。

**🔧 技术方法**

使用混合整数线性规划（MILP）求最优解；基于冲突图和先行图的列表调度启发式；事件驱动模拟；在 MATLAB 中调用 HiGHS 求解器实现。

**📊 数据集**

以太坊主网 2025 年 1 月 21,631,019–21,635,079 区块的交易记录，提取 gas、执行时间、读写集，分为同质（转账）和异质（合约）两类。

**📈 对比分析**

与 MILP 最优解、Solana 先行调度 baseline（Sol）和奖励贪心 baseline（RG）比较；MILP 运行时间从毫秒到数千秒，启发式在 0.1 秒以内；OBS 的 makespan 与 MILP 差距仅几个百分点，PBC 的奖励达到 74–100% LP 上界；加速比在 1.6–2.3 倍之间，近乎线性。

**⚠️ 局限性**

① 以 gas 为执行时间近似，实际硬件时间可能偏差；② 只使用静态冲突信息，缺乏全局冲突预测；③ 仅在离线批处理情境，未考虑在线/增量更新；④ MILP 仅在小规模可解。

---

## 566. A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces

**arXiv ID:** 2602.03442 | [PDF](https://arxiv.org/pdf/2602.03442v1)

**作者:** Mingxuan Du `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6130 | [OpenAlex ID](https://openalex.org/A5023341829)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Agentic RAG框架A‑RAG，允许大模型通过三种分层检索工具（keyword_search、semantic_search、chunk_read）自主决定检索策略并逐步构建答案。

**💡 创新点**

创新点在于：①把检索接口暴露给模型，赋予模型检索决策自主权；②设计三层级检索工具，使模型能够在词级、句级、块级不同粒度下灵活切换；③通过简单的ReAct式循环实现无监督的交互，验证分层检索可显著提升性能并减少上下文冗余。

**🔧 技术方法**

主要技术包括：基于句子级向量的密集检索、精确文本匹配、分层索引、ReAct式工具调用循环、上下文跟踪机制、以及对检索结果的片段化返回。

**📊 数据集**

使用的基准数据集有HotpotQA、2WikiMultiHopQA、MuSiQue、GraphRAG‑Bench（多跳问答场景），并在不同语言模型（GPT‑4o‑mini、GPT‑5‑mini）上评测。

**📈 对比分析**

与传统的Naive RAG、Graph‑RAG、Workflow‑RAG以及其他工作比较时，A‑RAG（Full）在所有数据集上均取得更高的LLM‑Acc和Contain‑Acc，同时检索的token数更少，证明了更高的上下文效率；在更强的GPT‑5‑mini上表现更为突出。

**⚠️ 局限性**

局限性包括：未对所有可能的工具组合进行系统的消融与比较；仅在中小模型上验证，未评估在更强模型（如GPT‑5、Gemini‑3）上的效果；主要针对多跳问答，尚未验证在事实验证、对话或长篇生成等其它知识密集任务上的泛化能力。

---

## 567. On the Complexity of Maximal/Closed Frequent Tree Mining for Bounded Height Trees

**arXiv ID:** 2602.03436 | [PDF](https://arxiv.org/pdf/2602.03436v1)

**作者:** Kenta Komoto `[一作]`, Hirotaka Ono `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文探讨了在树高度受限（无序树≤2，高度≤5；有序树≤2）条件下，最大频繁树与闭合频繁树挖掘的输出多项式复杂度，并给出了相应的多项式延迟算法与多项式难度证明。

**💡 创新点**

首次将树高度限制纳入频繁树挖掘的复杂性分析，提出“几乎双射”映射将超图的最大独立集与最大公共树关联，并在无序树场景下实现了高度≤2的多项式延迟枚举。

**🔧 技术方法**

采用逆搜索法、子树同构与整数多集合双射、（3,4)-SAT 与超图最小通道问题的归约，以及与Dualization问题的关联技术。

**📊 数据集**

文章主要为理论分析，并未使用实际数据集；实验示例仅引用Zaki合成XML树和CSLOGS数据集来说明树高的典型值。

**📈 对比分析**

与已知频繁子树挖掘算法对比，证明在无序树高度≤2时实现多项式延迟且多项式空间；在有序树高度≤2及无序树高度≤5时问题为输出多项式难，无法高效枚举。

**⚠️ 局限性**

局限性在于只针对高度≤2（无序）与≤5（无序）、≤2（有序）给出结果，未处理高度≥3的无序树闭合挖掘；归约证明基于P≠NP假设，实际可行性仍需进一步验证。

---

## 568. Model-based Optimal Control for Rigid-Soft Underactuated Systems

**arXiv ID:** 2602.03435 | [PDF](https://arxiv.org/pdf/2602.03435v1)

**作者:** Daniele Caradonna `[一作]` (SabioRob), Federico Renda `[通讯]` (Kurob)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于GVS模型的模型驱动最优控制框架，针对刚软耦合的欠驱动系统实现动态摆动上升任务，并在三种软机器人基准（软倒立摆、软双摆、软 Furuta 摆）上进行仿真验证。

**💡 创新点**

创新点包括：①利用GVS的解析导数实现高维连续体动力学的梯度求解；②设计了结合隐式积分与盒约束的Box-IDDP算法；③提出分辨率层次化的预热启动策略以加速收敛；④将直接共轭（DC）、差分动态规划（DDP）与非线性MPC统一在同一框架下比较。

**🔧 技术方法**

核心技术包括：GVS (Geometric Variable Strain) 解析动力学建模、隐式欧拉积分、Box-DDP/IDDP、非线性MPC、梯度基优化与热启动策略。

**📊 数据集**

使用了三套软机器人基准数据集：Soft Cart‑Pole、Soft Pendubot 与 Soft Furuta Pendulum，这些系统均基于GVS的高阶连续体动力学模型仿真。

**📈 对比分析**

通过对比DC、Box‑IDDP 与 NMPC 的成本收敛曲线与每次迭代计算时间，Box‑IDDP 在保持约束的同时收敛速度约为DC的2.9倍，虽然可能陷入局部最优，但在实时控制场景表现更佳；DC 在离线规划中取得更优终端成本。

**⚠️ 局限性**

局限性包括：①需要GVS模型的解析导数，若不可得则难以应用；②Box‑IDDP 仍可能受初始值影响陷入局部最优；③实验验证缺失，尚未评估硬件实时性能与模型不确定性鲁棒性。

---

## 569. CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering

**arXiv ID:** 2602.03420 | [PDF](https://arxiv.org/pdf/2602.03420v1)

**作者:** Siyi Wang `[一作]` (University of Melbourne), Ting Dang `[通讯]` (University of Melbourne)

**通讯引用:** 639 | [OpenAlex ID](https://openalex.org/A5071116593)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文系统研究并实现了激活向量调制（activation steering）在混合情绪和文本‑情绪不匹配情境下的可控 TTS 方法。

**💡 创新点**

创新点在于：①在两阶段 Hybrid TTS 中定位情绪可调制的最佳层级和操作；②提出基于线性可分性的层级‑操作选择；③构建可组合的情绪向量并实现量化混合情绪合成；④在多评测框架下验证效果。

**🔧 技术方法**

采用激活向量调制、线性可分性评估、层级操作注入，结合 TTS 语言模型与 flow‑matching 解码器。

**📊 数据集**

使用 ESD、RAVDESS、CREMA‑D、CosyVoice2、IndexTTS2 等数据集。

**📈 对比分析**

与无调制、随机向量、指令控制、情绪向量控制等基线比较，在 CREMA‑D 与 IEMOCAP 的 ID/OOD 场景下，激活向量调制在情绪可控性（E‑SIM、TEP、Spearman、H‑Rate）上显著提升，语音自然度与可懂度基本保持。

**⚠️ 局限性**

局限在于对情绪标签的文化/人口偏差敏感、在极端混合或高误差情绪下仍有限制、对某些模型层的可调制性受限且需调节超参数 α。

---

## 570. Generative Decompression: Optimal Lossy Decoding Against Distribution Mismatch

**arXiv ID:** 2602.03505 | [PDF](https://arxiv.org/pdf/2602.03505v1)

**作者:** Saeed R. Khosravirad `[一作]` (Nokia Bell Laboratories), Ingrid van de Voorde `[通讯]` (Nokia Bell Laboratories)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了“生成式解压缩”（generative decompression）方法，在压缩器固定且与真实源分布不匹配时，仅通过解码器侧的贝叶斯校正实现对源的最佳重建与任务优化。

**💡 创新点**

创新点在于将经典的MMSE估计与现代生成模型相结合，在保持编码器不变的前提下，解码器可利用真实分布信息动态调整重建点，从而消除分布不匹配导致的误差上限，并提供在噪声信道与任务感知场景下的软解码规则与性能分析。

**🔧 技术方法**

使用的技术包括：贝叶斯MMSE重建、Lloyd–Max分区、逆Mills比、Bennett积分、软判决（后验均值）以及在深度学习任务中的VQ‑VAE、Rician/高斯分布分析和分类损失函数。

**📊 数据集**

实验数据集涵盖：Gaussian/Laplace 1D 信号、5G CSI（Rician/Rayleigh）模拟、Fashion‑MNIST（VQ‑VAE）进行类别限制与域迁移。

**📈 对比分析**

与理想的端到端联合优化（encoder+decoder）和未调整的标准解码器对比。实验显示：在高分辨率（≥4 bit）下生成式解码器可恢复约70‑90% 的理想性能；在低分辨率和任务感知场景下恢复率显著提高，且在CSI反馈中实现显著的比特率节省。

**⚠️ 局限性**

限制：对分区（编码器）不可更改时的性能提升受限于原始码本的细化程度；在极端高分辨率或多位编码器（>1 bit）下，仍存在与理想解码器的误差壁垒；对非连续或高度多模态分布的精确贝叶斯估计需要较强的先验或大规模样本，且在实时系统中的计算开销未作评估。

---

## 571. A Minimal Task Reveals Emergent Path Integration and Object-Location Binding in a Predictive Sequence Model

**arXiv ID:** 2602.03490 | [PDF](https://arxiv.org/pdf/2602.03490v1)

**作者:** Linda Ariel Ventura `[一作]` (Sorbonne University), Sushrut Thorat `[通讯]` (Institute of Cognitive Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练一个三层GRU网络，用当前token与saccade位移预测连续2D空间中的下一个token标签，并通过在新场景中表现出的in‑context学习能力验证模型的自我建模能力。

**💡 创新点**

首次展示在极简环境下，预测任务可促使网络内部产生路径积分与动态label‑position绑定的机制，并证明模型能够在序列后期学习新绑定、覆盖旧绑定并在OOV位置泛化。

**🔧 技术方法**

采用3层GRU RNN、线性投影、ReLU层、交叉熵训练；通过SVM对内部层激活进行标签、位置及其绑定的解码；使用控制实验和干预分析验证机制。

**📊 数据集**

构造的自生成数据集：26个字母的token随机放置于[-4,4]二维空间，4–6个token组成场景；训练时持续生成，测试采用500个新场景（包括特殊pentagon布置）。

**📈 对比分析**

与仅解码单个标签或位置的基线相比，绑定的解码准确率显著提升；模型在序列中随步长提升预测准确率，35步时达到峰值；在未见过的saccade和OOV token‑position下仍保持约99.2%的预测准确率；干预实验显示可在序列后期快速学习新绑定。

**⚠️ 局限性**

局限性：仅在极简设置验证，缺乏对复杂视觉或Transformer架构的评估；绑定实现细节未明确，难以直接对应具体记忆形式；覆盖旧绑定缓慢，可能影响多任务学习；未探讨绑定机制在更广泛任务中的可迁移性。

---

## 572. DeepDFA: Injecting Temporal Logic in Deep Learning for Sequential Subsymbolic Applications

**arXiv ID:** 2602.03486 | [PDF](https://arxiv.org/pdf/2602.03486v1)

**作者:** Elena Umili `[一作]` (Sapienza University of Rome), Roberto Capobianco `[通讯]` (Sony AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可微分的Deterministic Finite Automata/莫尔机层 DeepDFA，用于将时间逻辑知识注入子符号深度学习模型，实现感知与符号推理的统一。

**💡 创新点**

创新点在于把概率有限自动机（PFA）与可微分层结合，既可直接注入已知时序规则，又可通过梯度学习符号地面化函数，支持稀疏与密集奖励的强化学习场景。

**🔧 技术方法**

核心技术包括基于 PFA 的可微分逻辑层（Softmax 温度控制）、CNN‑RNN/Transformer 结合、半监督符号地面化（SSSG）以及与 A2C 强化学习算法的耦合。

**📊 数据集**

实验使用了 MNIST4Declare（将 Declare 公式转化为 DFA 的 MNIST 图像序列）、CAVIAR 视频事件识别数据集以及基于 Minecraft‑类的地图与图像环境进行 RL 测试。

**📈 对比分析**

与 LSTM、GRU、Transformer、FuzzyDFA、NeSyA、RM 等方法对比，DeepDFA 在图像序列分类、CAVIAR 事件识别以及非马尔可夫 RL 任务中均实现了性能提升，接近或优于最优 RM 上限。

**⚠️ 局限性**

局限性包括：在奖励稀疏的任务中符号地面化效果不佳；目前仅支持确定性 DFA，未实现自动化 PFA/DFG 学习；与离线/基于模型的 RL 整合仍需进一步研究。

---

## 573. Self-Verification Dilemma: Experience-Driven Suppression of Overused Checking in LLM Reasoning

**arXiv ID:** 2602.03485 | [PDF](https://arxiv.org/pdf/2602.03485v1)

**作者:** Quanyu Long `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4534 | [OpenAlex ID](https://openalex.org/A5101936536)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型推理模型中的反射行为进行量化分析，发现自我验证步骤过度且大多为确认性检查，随后提出经验驱动的测试时控制框架，通过检索历史验证经验来决定是否抑制当前自我验证，从而减少推理长度并保持甚至提升准确率。

**💡 创新点**

首次将离线经验池与在线检索相结合，用历史验证结果来实时判定自我验证是否必要；同时设计轻量级检测器与抑制信号注入策略，实现对冗余验证的精准抑制。

**🔧 技术方法**

利用 GPT‑5 进行反射与验证标签化，训练 RoBERTa‑base 检测器识别自我验证激活，采用 BM25 检索相似验证片段，投票估计验证不必要性，并通过短文本抑制信号干预模型思考流程。

**📊 数据集**

在 AIME24/25、AMC23、MATH500、Olympiad Bench 四大数学推理基准上进行评测，并使用 Deepscaler（包含历史 AIME/AMC 题目）构建经验池。

**📈 对比分析**

与基线模型、全抑制、First Try Matters 等方法对比；在 Qwen3‑8B、QwQ‑32B、DeepSeek‑R1‑Distill‑Qwen‑7B 等模型上，经验驱动抑制可将平均推理长度缩短 9–20% 并在多数数据集上保持或提升 0.5–0.6% 的 Pass@1，显著优于全抑制导致的准确率下降。

**⚠️ 局限性**

需离线构建并维护经验池，阈值设置影响精确度与效率之间的平衡；抑制信号可能无法完全停止验证导致循环；只抑制自我验证而不考虑 rethink，可能在某些复杂任务中忽略必要的思路重构；实验依赖 GPT‑5 标注质量，标注误差可能影响经验池的可靠性。

---

## 574. ScDiVa: Masked Discrete Diffusion for Joint Modeling of Single-Cell Identity and Expression

**arXiv ID:** 2602.03477 | [PDF](https://arxiv.org/pdf/2602.03477v1)

**作者:** Mingxuan Wang `[一作]` (Renmin University of China), Yanbiao Ma `[通讯]` (Renmin University of China)

**通讯引用:** 2903 | [OpenAlex ID](https://openalex.org/A5067460359)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于masked discrete diffusion的单细胞转录组生成与表示学习框架scDiVa，能够在不依赖自回归顺序的情况下同时重建基因身份和表达量；

**💡 创新点**

创新点包括：1）将前向扩散与单细胞技术dropout等价；2）引入Entropy‑Normalized Serialization降低稀疏性并提升信息密度；3）双重去噪损失同时优化基因身份与剂量；4）Depth‑Invariant Sampling将训练视为逆向物理测序；5）使用latent anchor token保持全局细胞身份；6）利用RoPE与SwiGLU提升Transformer的无序建模能力；

**🔧 技术方法**

技术手段包括：masked discrete diffusion、Transformer编码器（RoPE、SwiGLU、Pre‑RMSNorm）、双向去噪、深度不变采样、Entropy‑Normalized Serialization、latent anchor token、对抗域适配（GRL+SupCon）等；

**📊 数据集**

预训练数据为约5900万细胞的大型私有单细胞转录组集合；在评测中使用PBMC12k、Immune、Perirhinal Cortex、BMMC、COVID‑19、hPancreas、MS、Myeloid、Adamson、Norman等多种公开基因表达与扰动数据集；

**📈 对比分析**

与Geneformer、GeneMamba、scGPT、scFoundation、Harmony等SOTA方法对比，scDiVa在rank‑value重构中取得最高Spearman；在多批次整合中Avg‑batch/Avg‑bio均优于对手；在细胞类型注释（Fine‑tune与Zero‑shot）中表现最优；在扰动预测中Pearson 0.837/0.709，MSE低于其他方法，表明在多项任务上均具备显著性能优势；

**⚠️ 局限性**

局限性包括：模型训练与推理仍然计算量大、耗时；对极低表达基因的重构精度有限；目前主要针对单一模态（scRNA‑seq），跨模态兼容性待进一步验证；

---

## 575. TactDeform: Finger Pad Deformation Inspired Spatial Tactile Feedback for Virtual Geometry Exploration

**arXiv ID:** 2602.03476 | [PDF](https://arxiv.org/pdf/2602.03476v1)

**作者:** Yihao Dong `[一作]` (University of Sydney), Anusha Withana `[通讯]` (University of Sydney)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5047568623)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了TactDeform，一个基于手指皮肤变形模型的电触觉反馈系统，用于在VR环境中通过参数化的时空电刺激呈现三维几何体的形状与纹理，并通过用户研究验证其效果。

**💡 创新点**

采用双重上下文（接近、接触、滑动）与几何上下文（面、边、角、纹理）联合映射，使用参数化的电刺激模式模仿真实手指皮肤变形，从而突破传统电触觉在空间分辨率和特征表达上的瓶颈。

**🔧 技术方法**

32电极柔性电触觉贴片、单相阳极电流脉冲、Unity实时控制、Meta Quest 3手势追踪、参数化模式生成算法、实时过滤与状态机、用户研究与统计分析等技术。

**📊 数据集**

使用自建实验对象（面、边、角、纹理级别的可视模型）以及四个递增复杂度的三维模型（球、立方体、茶壶、兔子）进行用户测试，未使用公开数据集。

**📈 对比分析**

与两种基线（均匀激活、接触面积映射）在相同VR场景中进行比较，采用两路任务（几何特征识别、纹理辨别）以及主观偏好和思考录音，结果显示TactDeform在几何特征识别达85.7%，纹理辨别达95.8%，并获得58%用户偏好，优于基线。

**⚠️ 局限性**

仅支持单指操作、仅能感知左右角度、贴片覆盖有限、未考虑多指及复杂手势、单相阳极刺激可能产生不舒适感，缺乏对动态物体、多材质及更复杂交互的支持。

---

## 576. The Dual Role of Abstracting over the Irrelevant in Symbolic Explanations: Cognitive Effort vs. Understanding

**arXiv ID:** 2602.03467 | [PDF](https://arxiv.org/pdf/2602.03467v1)

**作者:** Zeynep G. Saribatur `[一作]` (Institute of Logic and Computation), Ute Schmid `[通讯]` (Cognitive Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在符号 AI 解释中，使用 ASP 定义并评估了两种抽象操作（去除无关细节与聚类）对人类理解与认知负荷的影响。

**💡 创新点**

提出了基于 χ‑irrelevance 的更通用抽象概念，并验证了聚类与去除对准确率与答题时间的不同效益。

**🔧 技术方法**

核心技术是 Answer Set Programming（ASP）与其解释工具；实验采用人类受试者的分类任务。

**📊 数据集**

实验数据来自三种生物学领域（花、蘑菇、仙人掌）的手工构造实例集，每个领域包含正负样本。

**📈 对比分析**

通过四组对照（默认、聚类、去除、聚类+去除）进行准确率、答题时间与自信度测评；聚类显著提升准确率，去除显著缩短答题时间，二者结合效果最佳。

**⚠️ 局限性**

局限包括实验规模有限、对抽象程度与任务复杂度的交互未深入探讨，以及自信度未随抽象显著变化。

---

## 577. Soft-Radial Projection for Constrained End-to-End Learning

**arXiv ID:** 2602.03461 | [PDF](https://arxiv.org/pdf/2602.03461v1)

**作者:** Philipp J. Schneider `[一作]` (École Polytechnique Fédérale de Lausanne), Daniel Kuhn `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 9313 | [OpenAlex ID](https://openalex.org/A5065780980)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了软径向投影（Soft‑Radial Projection）层，实现对凸约束集的严格内部映射，从而在端到端学习中避免梯度饱和问题。

**💡 创新点**

通过构造以锚点为原点的径向同胚映射，保证投影可微且Jacobian几乎处处满秩，既保持可行性又避免正交投影导致的维度坍塌。

**🔧 技术方法**

基于凸几何的径向投影公式、闭式求解α*、差分传播与Rademacher可微性理论、全局同胚与可微性证明，并在实验中实现可并行的向量化计算。

**📊 数据集**

在金融组合优化（50只资产）和网约车调度（150个区）等任务上使用真实市场和需求数据集进行评估。

**📈 对比分析**

与Softmax、正交投影、DC3、HardNet等基线对比，在组合优化任务中显著提升夏普比率（从0.25‑0.64提升至0.90），在调度任务中与最优基线相当且优于正交投影。

**⚠️ 局限性**

仅适用于凸约束集且需要已知内部锚点，无法直接处理非凸或动态锚点的情形，且对高维复杂约束的自动锚点选择尚未解决。

---

## 578. RankSteer: Activation Steering for Pointwise LLM Ranking

**arXiv ID:** 2602.03422 | [PDF](https://arxiv.org/pdf/2602.03422v1)

**作者:** Yumeng Wang `[一作]` (Leiden Institute of Advanced Computer Science), Suzan Verberne `[通讯]` (Leiden Institute of Advanced Computer Science)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种后置激活向导框架 RankSteer，用于在零样本点式 LLM 排序中通过投影干预调整决策、证据与角色方向，提升排序质量。

**💡 创新点**

创新点是将排名行为拆解为可分离的决策、证据和角色三向量，并在推理时对激活进行投影干预，实现在不改模型权重的前提下调校排序。

**🔧 技术方法**

使用对比式向量提取、投影干预、角色与证据方向的正交化等技术，基于 LLM 的隐藏层进行激活调节。

**📊 数据集**

在 TREC‑DL 2020 和 BEIR 八个子数据集（包括 Covid、Touche、Signal 等）上进行实验，并使用 Llama‑3.1‑8B、Qwen‑2.5‑7B、Mistral‑7B 三个后端模型。

**📈 对比分析**

与现有零样本点式、双向、集合式、列表式以及 BM25 方法对比，RankSteer 在大多数数据集上达到或逼近对比式方法的性能，同时保持点式推理的 O(1) 复杂度。

**⚠️ 局限性**

局限包括对角色向量正交化假设的依赖、对 anchor 查询的敏感性，以及在某些数据集上提升幅度不及微调或显式比较方法；对大模型的通用性验证仍有限。

---

## 579. Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation

**arXiv ID:** 2602.03515 | [PDF](https://arxiv.org/pdf/2602.03515v1)

**作者:** Hyunji Jung `[一作]` (POSTECH), Namhoon Lee `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了异步流水线并行训练中梯度时延导致的收敛退化问题，并提出通过旋转优化空间以对齐Hessian特征基与坐标基，从而缓解梯度时延的影响。

**💡 创新点**

创新点在于将Hessian特征基对齐引入异步管线并行，设计了基于Kronecker‑factorized Fisher矩阵的高效旋转框架“Equal*”，显著提升了大规模LLM训练的收敛速度与可扩展性。

**🔧 技术方法**

主要技术包括：坐标系旋转（Eigenbasis alignment）、Kronecker‑factorized empirical Fisher近似、一次性幂迭代+QR分解计算特征向量、以及异步Adam在旋转空间中的更新。

**📊 数据集**

使用OpenWebText 1B token数据集训练从95M到1B参数的Decoder‑only Transformer模型。

**📈 对比分析**

与PipeDream、PipeDream‑LR、Nesterov三种异步管线基线对比，Equal*在32个阶段时迭代数下降71.6%，在1B模型上下降76.8%，GPU时间缩短约54%，同时保持甚至提升最终模型性能。

**⚠️ 局限性**

局限性包括：需额外的特征基估计与更新开销，假设Hessian为块对角且Kronecker分解成立；在极大规模模型或极低频更新时仍可能存在计算与内存负担；且对权重缓存的依赖在无权重存储方案下效果尚待进一步验证。

---

## 580. A Function-Space Stability Boundary for Generalization in Interpolating Learning Systems

**arXiv ID:** 2602.03514 | [PDF](https://arxiv.org/pdf/2602.03514v1)

**作者:** Ronald Katende `[一作]` (Kabale University), Ronald Katende `[通讯]` (Kabale University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5076646959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了函数空间轨迹稳定性框架，通过引入收敛递归、稳定性证书和边界定理，对现代过参数化学习系统中的泛化机制进行结构化分析，并提供实验诊断。

**💡 创新点**

创新点在于将算法稳定性直接定义在预测器轨迹上，构造可计算的收敛证书，并给出一条精确的“稳定性可解释泛化”与“非稳定性可解释泛化”之间的分界定理，突破了传统参数级、统一收敛等局限。

**🔧 技术方法**

使用了共享随机性耦合、收敛递归、Jacobian/梯度范数估计、尖锐度近似以及通用的函数空间不等式，辅以理论证明和对多种优化器的实验评估。

**📊 数据集**

实验主要基于过参数化的高维线性回归（高斯输入、可调谱）、但框架本身与任何数据集和模型兼容。

**📈 对比分析**

通过对步长、优化器、邻近样本选择和标签置换等干预，比较证书增长与终端测试误差；在稳定性可解释的情形下，证书与测试误差呈正相关；在标签随机化等非稳定性情形下，证书与误差无关，验证了边界定理。

**⚠️ 局限性**

局限在于证书上界可能过于保守、无法覆盖所有深度网络或非收敛情形、对估计噪声敏感，以及对高维数据的实验验证仍有限。

---

## 581. CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains

**arXiv ID:** 2602.03511 | [PDF](https://arxiv.org/pdf/2602.03511v1)

**作者:** Qixin Zeng `[一作]` (University of Southampton), Chao Huang `[通讯]` (University of Southampton)

**通讯引用:** 12976 | [OpenAlex ID](https://openalex.org/A5042083053)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Contractive Mapping for Robustness（CMR）框架，利用对比学习与 Lipschitz 正则化构造收敛的潜在空间，以提升类人机器人在观测噪声下的稳健步态控制。

**💡 创新点**

创新点在于首次将收敛映射定理与深度强化学习结合，给出噪声导致的回报差距上界，并通过对比学习保持语义信息而不失去鲁棒性。

**🔧 技术方法**

采用对比损失（InfoNCE）、Lipschitz 正则化、PPO 强化学习、收敛嵌入编码器以及理论证明。

**📊 数据集**

使用仿真数据：Isaac Gym 与 MuJoCo 的多种不规则地形（斜坡、踏步、平衡木等）和多种噪声配置（α=1,2,3）。

**📈 对比分析**

与 HIM、LCP、Naïve PPO 等基线比较，在距离、功耗、动作平滑度等指标上均优于对手，尤其在高噪声情形下表现更为稳健。

**⚠️ 局限性**

局限在于需要手动调节温度、拉氏系数等超参数，收敛映射强度可能削弱表达能力，且目前尚未在真实机器人上完成验证。

---

## 582. ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response

**arXiv ID:** 2602.03430 | [PDF](https://arxiv.org/pdf/2602.03430v1)

**作者:** Xiaomeng Zhu `[一作]` (Hong Kong University of Science and Technology), Xuantang Xiong `[通讯]` (Tencent)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向主动响应的 ProAct‑75 基准以及对应的 ProAct‑Helper 多模态框架。

**💡 创新点**

创新点在于提供任务图与并行执行的结构化标注，并在此基础上设计了基于熵驱动的启发式搜索实现结构感知的主动决策。

**🔧 技术方法**

采用多模态大型语言模型（mLLM）结合层级绑定模块（HBM）进行感知和决策，利用任务图进行合法动作搜索。

**📊 数据集**

数据集来源于 Ego‑Exo4D、COIN、UCF‑Crime 及自采视频，共 5,383 条视频、91,581 步骤，含 75 个任务。

**📈 对比分析**

与多种开源和闭源 mLLM 进行两阶段评估，ProAct‑Helper 在触发/任务/步骤检测、未来动作预测和并行动作率等指标上均超过对比模型，提升触发 mF1 6.21%、平均节省步骤 0.25 步、并行动作率提升 15.58%。

**⚠️ 局限性**

局限在于对动作生成的可控性不足（未来动作序列偶有幻觉），并且依赖于任务图的准确性和预先定义的线程划分。

---

## 583. Efficient Algorithms for Partial Constraint Satisfaction Problems over Control-flow Graphs

**arXiv ID:** 2602.03588 | [PDF](https://arxiv.org/pdf/2602.03588v1)

**作者:** Xuran Cai `[一作]` (University of Oxford), Amir Goharshady `[通讯]` (University of Oxford)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5005241421)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种通用的线性时间算法，用于求解结构化程序控制流图上的部分约束满足问题（PCSP），涵盖寄存器分配、LOSPRE和银行选择等编译器优化任务。

**💡 创新点**

创新点在于利用Series-Parallel-Loop（SPL）分解而非传统树宽分解，构建动态规划算法实现O(|G|·|D|^6)复杂度，且对固定域可降为线性。

**🔧 技术方法**

采用SPL分解、底层动态规划与成本函数推导，结合图约束和不违反约束的代价计算。

**📊 数据集**

实验使用Small Device C Compiler（SDCC）的回归测试集，包括HC08(13,463个实例)、Z80(179,611个实例)和MCS51(84,700个实例)。

**📈 对比分析**

与基于树宽的状态‑最优算法、SAT（Kissat）和ILP（Gurobi）进行对比；相较树宽方法平均速度提升约4倍，ILP方法约10倍，SAT方法约100倍。

**⚠️ 局限性**

局限在于仍需对域大小|D|进行指数级指数，虽然对常数域可线性，但大域下计算量随|D|^6成比例，且仅适用于结构化无goto程序的CFG。

---

## 584. EarResp-ANS : Audio-Based On-Device Respiration Rate Estimation on Earphones with Adaptive Noise Suppression

**arXiv ID:** 2602.03549 | [PDF](https://arxiv.org/pdf/2602.03549v1)

**作者:** Michael Küttner `[一作]` (Karlsruhe Institute of Technology), Tobias Röddiger `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 331 | [OpenAlex ID](https://openalex.org/A5069745172)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在商业耳机上实现全设备、实时的呼吸率估计系统，利用双麦克风自适应噪声抑制并在耳机内部完成所有处理。

**💡 创新点**

创新点包括：①使用延迟LMS自适应滤波器（ANS）消除外部噪声，同时保持呼吸音；②完全基于硬件实现，无需云端或深度学习模型；③通过双耳融合和基于通道差异的异常剔除提升鲁棒性。

**🔧 技术方法**

技术实现：双麦克风延迟LMS自适应滤波（含泄漏、阈值归一化）；STFT特征提取、能量与谱相似度计算；峰值检测与谐波谱求解；低功耗DSP与MCU协同；BLE低功耗通信。

**📊 数据集**

数据集：20名参与者（14男6女，22–46岁）在6种真实环境噪声（低噪、50/65/80 dB白噪、约80 dB餐厅噪声、音乐）下采集，配备respiBAN胸带做基准，包含休息和30 s跳绳运动，共计2000多窗口。

**📈 对比分析**

评估方法：与无噪声过滤、固定滤波器（BreathPro）以及NLMS对比；性能表现：单通道MAE 0.90 CPM，双耳融合MAE 0.84 CPM，异常剔除后MAE 0.47 CPM；80 dB噪声下MAE从1.54降至0.75；相对BreathPro MAE从2.54降至0.81；系统功耗<2%处理，延迟≈11 ms。

**⚠️ 局限性**

局限性：仅在低运动或静息场景验证，未评估强运动；对耳塞密封质量敏感，密封差时呼吸音被抑制；假设噪声抑制后呼吸音占主导，若被破坏导致误估；异常剔除仍无法完全消除所有错误；基准胸带信号偶有不可靠；在极安静环境下ANS可能过度滤除呼吸音。

---

## 585. Sequential Linear Contracts on Matroids

**arXiv ID:** 2602.03543 | [PDF](https://arxiv.org/pdf/2602.03543v1)

**作者:** Kanstantsin Pashkovich `[一作]` (University of Waterloo), Yun Xing `[通讯]` (University of Waterloo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在马约数约束下的线性序贯合同，给出最优合同的求解框架并与马约数可靠性问题等价。

**💡 创新点**

创新点在于将序贯合同与马约数可靠性问题建立正式等价关系，利用FRUGAL算法及微扰技巧实现最佳代理策略，并提供多种精确与近似求解方法，尤其在统一、层次与图形马约数下得到多项复杂度结论。

**🔧 技术方法**

主要技术包括马约数理论、概率分析、分段线性梯度（grade）与代理报酬替代（surrogate）方法、微扰与最优代理策略的逼近、Monte‑Carlo 采样与 FPRAS 构造，以及复杂度理论中的多项式时间与逼近下界证明。

**📊 数据集**

本文为理论性工作，未使用实验数据集，主要通过理论证明与算法复杂度分析得出结果。

**📈 对比分析**

方法通过与马约数可靠性问题的归约实现精确/近似求解，并在特殊情形（相对均匀收益、两值收益）下给出 FPRAS；在统一与层次马约数中可多项式求解，图形马约数则为 #P‑难；总体性能依赖于马约数类型与收益分布。

**⚠️ 局限性**

局限包括：仅对线性合同有效；对一般合同需指定指数级别的支付函数，计算不可行；对马约数的依赖需使用独立性或acles，无法直接在某些马约数上高效实现；近似误差随参数 m、n 可能增大；特殊情况之外的泛化仍有挑战。

---

## 586. Constrained Dynamic Gaussian Splatting

**arXiv ID:** 2602.03538 | [PDF](https://arxiv.org/pdf/2602.03538v1)

**作者:** Zihan Zheng `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22728 | [OpenAlex ID](https://openalex.org/A5100447801)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种可在边缘设备上控制高质量动态 4D Gaussian Splatting 的框架 CDGS，并实现了精确的高阶 Gaussian 数量控制。

**💡 创新点**

创新点包括：1) 将 Gaussian 数量限制直接作为可微的优化约束引入训练；2) 引入统一重要性得分融合几何、运动和感知信息；3) 采用自适应静态-动态分配策略以最大化有限预算的利用；4) 采用三阶段训练与双模压缩方案以提升效率与存储表现。

**🔧 技术方法**

技术核心包括：可微预算控制器、统一重要性评分（几何/运动/感知三模），动态‑静态划分、可微稀疏化（激活门）、三阶段训练、双模混合压缩（空间重排+预测编码+H.264 视频压缩）。

**📊 数据集**

在 N3DV、Technicolor、MeetRoom 三个公开真实动态视频数据集上进行实验。

**📈 对比分析**

与 4DGC、STGS、Ex4DGS、ReRF、TeTriRF、Swift4D 等前沿方法相比，CDGS 在保持或提升 PSNR/SSIM 的同时，将模型体积压缩至 1/3 甚至 1/10，压缩率提升 3×，且 Gaussian 数量误差 <2%，渲染速度提升约 5–10 fps，整体性能位于 Pareto 前沿。

**⚠️ 局限性**

限制主要体现在：1) 仍需手工设定目标 Gaussian 数量，无法完全自动化；2) 对极端高速运动或大规模场景的动态分配可能需要更精细的阈值调优；3) 双模压缩在极低位深度时可能导致细节损失。

---

## 587. MatGPTQ: Accurate and Efficient Post-Training Matryoshka Quantization

**arXiv ID:** 2602.03537 | [PDF](https://arxiv.org/pdf/2602.03537v1)

**作者:** Maximilian Kleinegger `[一作]` (Vienna University of Technology), Dan Alistarh `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 4412 | [OpenAlex ID](https://openalex.org/A5083822059)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种后训练多层次量化方法 MatGPTQ，能够在一次量化过程中生成可切片的多精度模型；

**💡 创新点**

创新点在于将 GPTQ 改造成多精度目标，加入交叉位宽误差补偿与权重，并配合演化搜索实现非均匀位宽分配，同时提供开源 CUDA 核；

**🔧 技术方法**

采用 GPTQ、交叉位宽损失、位宽插值、演化搜索（EvoPress）以及 TensorCore/SIMT 加速的 CUDA 核；

**📊 数据集**

使用 LLaMA 3.1、Qwen3、Phi‑3 等 LLM 的小量校准集进行 PTQ，并在 WikiText、ARC、HellaSwag、PIQA、Winogrande 等标准任务集上评估；

**📈 对比分析**

与 GPTQ、OmniQuant、MatQuant 对比，3‑bit 平均精度提升约 1.3%，4/8‑bit 差距 ≤1%；插值 6‑bit 接近理论；混合位宽 MatGPTQ‑EP 在低精度下保持甚至超过基线；推理时 3‑4 位可实现 2.5‑3× 加速；

**⚠️ 局限性**

对极低精度（≤2 位）准确率仍有限；当前实现仅支持整数量化，CUDA 核针对 Ampere，未覆盖 Hopper/Blackwell；未结合更适合低精度的量化算法。

---

## 588. D3PIA: A Discrete Denoising Diffusion Model for Piano Accompaniment Generation From Lead sheet

**arXiv ID:** 2602.03523 | [PDF](https://arxiv.org/pdf/2602.03523v1)

**作者:** Eunjin Choi `[一作]` (KAIST), Juhan Nam `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在符号音乐领域，利用离散扩散模型 D3PIA，自动从给定的主旋律和和弦进程生成完整的钢琴伴奏。

**💡 创新点**

创新点：①将邻域注意力（NA）引入离散扩散模型的编码器和解码器，实现对和弦与旋律的局部对齐；②采用离散扩散而非连续扩散，更适合二进制钢琴卷轴的离散特性；③在解码器中加入 AdaLN 和 AS 采样，提升和弦一致性与节奏细节。

**🔧 技术方法**

主要技术：离散扩散（Discrete Diffusion），邻域注意力（Neighborhood Attention）、双向 LSTM、AdaLN 时间条件化、AS 采样、Piano Roll 表示、16 分音符时间单位、四状态（Onset、Off、Sustain、MASK）编码。

**📊 数据集**

使用 POP909 数据集（909 首中文流行歌曲钢琴伴奏），在 8:1:1 的划分上训练并评估。

**📈 对比分析**

对比方法包括 Polyffusion、WSG‑4th、FGG（连续扩散）以及 Transformer 基础的 C&E‑E；D3PIA 在和弦准确率（CA）和相似度（CS）、节奏相似度（GS）上均高于多数竞争模型，主观听评得分最高，且生成速度（≈1.7 s/8 段）快于大多数模型。

**⚠️ 局限性**

限制：未对力度（velocity）进行建模，导致音色与动态多样性受限；模型规模虽小，但在极端复杂和弦结构或更长片段时仍可能出现不够多样或细节不足的问题。

---

## 589. SlowFocus: Enhancing Fine-grained Temporal Understanding in Video LLM

**arXiv ID:** 2602.03589 | [PDF](https://arxiv.org/pdf/2602.03589v1)

**作者:** Ming Nie `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**通讯引用:** 46111 | [OpenAlex ID](https://openalex.org/A5100461206)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为 SlowFocus 的机制，帮助视频大型语言模型（Vid-LLM）在保持高质量帧级语义信息的同时，获得足够多的帧以进行细粒度时序理解。

**💡 创新点**

创新点包括：①根据问题定位查询相关时序片段并在该片段内进行高频密集采样；②引入多频混合注意力（MMA）将高频细节与低频全局上下文融合；③设计时序编码器处理多频帧的相对位置；④构建细粒度时序理解基准 FineAction-CGR 与对应的训练策略，显著提升 Vid-LLM 的时间定位与推理能力。

**🔧 技术方法**

使用的技术主要是：视频编码器 + 视觉适配器 + 大语言模型（如 Vicuna‑7B/Vicuna‑7B‑v1.5）、多频采样（MFS）、多频混合注意力（MMA）、时序编码器、LoRA 微调、三阶段训练策略（模态对齐 → 边界增强 → SlowFocus 适配）。

**📊 数据集**

使用的数据集包括 FineAction、ActivityNet Captions、WebVid 2.5M、InternVid‑10M‑FLT、LLaVA 图文数据、GPT‑4V 与自研 Recaptioner 生成的片段级字幕等，用于构建 FineAction‑CGR benchmark 与训练。

**📈 对比分析**

与基线 LLaMA‑VID、VTime‑LLM 等模型比较，SlowFocus 在 FineAction‑CGR 上获得 mIoU 66.68、时序推理准确率 53.10%，在零样本视频问答（MSVD‑QA、MSRVTT‑QA、ActivityNet‑QA）和长视频基准（MovieChat‑1K、EgoSchema）上也保持竞争力，说明其在细粒度时序理解和整体视频推理上的优越性。

**⚠️ 局限性**

主要限制在于：当前方法仍难以保持足够高的空间分辨率，导致在细节丰富的场景中可能出现错误预测；此外，对极长视频的处理仍有提升空间。

---

## 590. CL-bench: A Benchmark for Context Learning

**arXiv ID:** 2602.03587 | [PDF](https://arxiv.org/pdf/2602.03587v1)

**作者:** Shihan Dou `[一作]` (Tencent), Shunyu Yao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CL‑bench基准，评估语言模型在提供的上下文中学习并应用新知识的能力。

**💡 创新点**

首次系统化定义并量化“上下文学习”能力，设计无污染、多维度可验证的任务与评价指标。

**🔧 技术方法**

使用大语言模型作为判定者进行自动评估，结合prompt engineering、长上下文处理和高推理力度设置。

**📊 数据集**

构建500个复杂上下文、1899个任务和31607个评价指标的CL‑bench数据集。

**📈 对比分析**

与10款前沿模型比较，平均解题率仅17.2%，最佳GPT‑5.1为23.7%，表明当前模型在上下文学习上仍差距巨大。

**⚠️ 局限性**

仅覆盖文本上下文、单轮任务、有限领域，缺乏人类基准与多模态扩展，验证机制对单一LM（GPT‑5.1）的依赖。

---

## 591. Symbolic Model Checking using Intervals of Vectors

**arXiv ID:** 2602.03565 | [PDF](https://arxiv.org/pdf/2602.03565v1)

**作者:** Damien Morard `[一作]` (University of Geneva), Didier Buchs `[通讯]` (University of Geneva)

**通讯引用:** 918 | [OpenAlex ID](https://openalex.org/A5013795728)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

设计了一种基于向量区间的符号向量集，用于对Petri网进行全局CTL模型检查，以解决状态空间爆炸问题。

**💡 创新点**

创新点在于提出符号向量和符号向量集合的概念，给出可归一化的同态运算和最小化形式，并通过饱和与聚类优化显著提升性能。

**🔧 技术方法**

采用符号模型检查理论、向量区间符号结构、同态运算、归一化与最小化、饱和搜索及Swift实现框架。

**📊 数据集**

使用MCC 2022竞赛提供的Petri网模型（如CircadianClock、MutualExclusion等）作为实验数据集。

**📈 对比分析**

与GreatSPN、ITS‑Tools、Tapaal等工具比较，使用饱和优化后在16个CTL查询中全部完成，平均耗时约30分钟，显著优于其他工具在60分钟限制内的表现。

**⚠️ 局限性**

限制在于归一化操作最坏复杂度为O(n!)，实现容易出现递归深度和内存瓶颈，难以处理极大规模Petri网；对局部模型检查和非Petri模型的适用性尚未完全验证。

---

## 592. ACL: Aligned Contrastive Learning Improves BERT and Multi-exit BERT Fine-tuning

**arXiv ID:** 2602.03563 | [PDF](https://arxiv.org/pdf/2602.03563v1)

**作者:** Wei Zhu `[一作]` (University of Hong Kong), Wei Zhu `[通讯]` (University of Hong Kong)

**通讯引用:** 18004 | [OpenAlex ID](https://openalex.org/A5068308955)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个对齐的对比学习（ACL）框架，用于提升监督学习场景下的BERT微调和多出口BERT的训练效果。

**💡 创新点**

创新点在于①将标签嵌入视为对比学习中的锚点（ACL-Embed），②通过梯度角度动态关闭冲突的对比损失（ACL-Grad），③在多出口模型中实现跨层对比学习（ACL-CL）实现知识蒸馏。

**🔧 技术方法**

使用了对比学习、交叉熵损失、标签嵌入、梯度角度判定、MHA退出头以及多出口BERT训练的2ST策略等技术。

**📊 数据集**

主要在GLUE基准数据集上进行评测，使用BERT-base和RoBERTa-base作为预训练模型。

**📈 对比分析**

与CE、CE+SCL以及多种压缩和蒸馏方法（LayerDrop、DistillBERT、TinyBERT等）对比，ACL在大多数GLUE任务和6层出口模型上均取得更高准确率，且在多出口BERT中显著提升浅层出口性能。

**⚠️ 局限性**

局限性包括仅针对句子/句对分类任务验证，未扩展到序列标注、关系抽取等任务；以及未评估在静态模型压缩框架中的适用性。

---

## 593. ELIQ: A Label-Free Framework for Quality Assessment of Evolving AI-Generated Images

**arXiv ID:** 2602.03558 | [PDF](https://arxiv.org/pdf/2602.03558v1)

**作者:** Xinyue Li `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21112 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种完全无标注的框架ELIQ，用自动构造的正负样本对，结合预训练的多模态大模型（Qwen3‑VL‑8B‑Instruct）进行指令微调，最终通过轻量级的Quality Query Transformer实现对AI生成图像的视觉质量和prompt‑图像对齐质量的双维度预测。

**💡 创新点**

创新点包括：①将传统的绝对MOS监督替换为可周期性刷新、基于相对比较的无标签监督；②构造覆盖传统失真与AIGC特定失真模式的正负样本对；③在预训练多模态模型上进行“质量感知”指令微调，并冻结特征提取器；④引入门控融合技术统一技术与审美特征，再用两种查询标记的Transformer实现单图推理的质量评分；⑤实现了从AIGC到UGC的无缝迁移，显著降低了标注成本。

**🔧 技术方法**

技术细节包括：多模态LLM（Qwen3‑VL‑8B‑Instruct）指令微调；正负样本生成策略（使用Qwen‑Image、FLUX.1‑dev、Stable Diffusion 3.5‑Large；技术失真：JPEG压缩、Gaussian噪声；审美失真：图像编辑；对齐失真：prompt置换或图像‑prompt错配）；门控融合网络；Quality Query Transformer；基于排名的损失（视觉与对齐两项）。

**📊 数据集**

使用的主要数据集：①自构造的400个多类别prompt，分别在三款T2I模型上生成1200张高质量正样本；②AIGC评测基准AGIQA‑3K、AIGCIQA2023、AIGIQA‑20K；③UGC评测基准KonIQ‑10k、SPAQ；此外还利用公开的AIGC‑AIGIQA‑20K等作为负样本生成来源。

**📈 对比分析**

与现有监督、弱监督和无监督方法对比，ELIQ在AIGC基准上在弱监督下SRCC分别达到0.876/0.837/0.856，接近或超过部分全监督模型；在无监督下SRCC分别为0.801/0.767/0.786，明显优于所有无监督基线；在UGC基准上仅用30% MOS标签即可得到SRCC≈0.912（KonIQ‑10k）和0.915（SPAQ），而无标签版本也能得到0.818/0.842；整体而言，ELIQ在多维度质量评估上均显著领先传统无监督方法，且与监督方法差距可忽略。

**⚠️ 局限性**

局限性：①正负样本的构造仍依赖人工设计的失真方式，可能遗漏未来模型出现的新型失真；②需要大规模预训练多模态模型，算力与成本较高；③对极端分布或非常新颖的生成模型，迁移效果可能下降；④排名监督仅保证排序关系，若需要精确的分数尺度仍需后期线性校准。

---

## 594. Scaling Test-Driven Code Generation from Functions to Classes: An Empirical Study

**arXiv ID:** 2602.03557 | [PDF](https://arxiv.org/pdf/2602.03557v1)

**作者:** Yunhao Liang `[一作]` (Chengdu Institute of Computer Applications, Chinese Academy of Sciences and University of Chinese Academy of Sciences), Zhe Cui `[通讯]` (Chengdu Institute of Computer Applications, Chinese Academy of Sciences and University of Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了将测试驱动开发（TDD）从函数级扩展到类级代码生成，并提出了依赖感知的类级TDD框架

**💡 创新点**

提出了 ClassEval‑TDD 清洁可靠的基准，构建了依赖分析+调度+反射式修复的全流程方法

**🔧 技术方法**

采用 LLM 自回归生成、方法级公测驱动迭代、基于反射的修复与依赖推断技术

**📊 数据集**

使用修正后的 ClassEval‑TDD 基准（100 个单类任务共 412 个方法）

**📈 对比分析**

与 Holistic/Incremental/Compositional 三种直接生成基线对比，8 款 LLM 上类级成功率提升 12–26 个百分点，最高达 71%；方法级成功率接近 90%；修复开销低

**⚠️ 局限性**

仍存在类级组合差距（跨方法状态一致性难题）、依赖推断误差导致调度不合法、对较小模型的修复效率有限

---

## 595. Persona Generators: Generating Diverse Synthetic Personas at Scale

**arXiv ID:** 2602.03545 | [PDF](https://arxiv.org/pdf/2602.03545v1)

**作者:** Davide Paglieri `[一作]` (Google DeepMind), Alexander Sasha Vezhnevets `[通讯]` (Google DeepMind)

**通讯引用:** 4366 | [OpenAlex ID](https://openalex.org/A5030684417)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并训练了可在任意情境下生成多样化合成人物群体的“Persona Generator”模型，利用AlphaEvolve对生成代码进行进化优化。

**💡 创新点**

创新点在于将生成器本身作为可进化的可编程代码进行优化，而非单纯调优固定合成群体或基于典型人群的模仿；并引入六维多目标多样性评估来覆盖罕见行为模式。

**🔧 技术方法**

采用大语言模型（Gemma‑3‑27b‑it 与 Gemini‑2.5 Pro）作为生成器与突变算子，使用AlphaEvolve进行进化搜索；并通过Concordia模拟框架评估生成的人物回答问卷的响应向量。

**📊 数据集**

使用人工合成的 50 份问卷（基于 Big Five、DASS、SVO、NFCS 等标准量表的自动生成），不依赖真实人类样本；在验证集上对生成的人物进行多样性度量。

**📈 对比分析**

与 Nemotron Personas、Concordia 原始生成器以及仅提供姓名的基线进行对比；实验表明进化后生成器在六项多样性指标上均显著优于基线，并在未见测试问卷上保持较高性能。

**⚠️ 局限性**

主要局限包括：评价指标主要基于问卷答题的陈述偏好，无法充分评估在开放式交互中的行为多样性；突变提示的设计可能限制生成器结构的多样性；且目前仅在每个情境下生成 25 个人物，未验证大规模生成的可扩展性。

---

## 596. WARP Logic Neural Networks

**arXiv ID:** 2602.03527 | [PDF](https://arxiv.org/pdf/2602.03527v1)

**作者:** Lino Gerlach `[一作]` (Princeton University), Isobel Ojalvo `[通讯]` (Princeton University)

**通讯引用:** 7454 | [OpenAlex ID](https://openalex.org/A5047987529)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于Walsh–Hadamard变换的可微分逻辑神经网络框架WARP，用于高效学习硬件原生的布尔函数。

**💡 创新点**

创新点在于：①使用Walsh–Hadamard参数化实现完全可表达且参数最优；②不需要近似梯度、无冗余；③结合可学习阈值、残差初始化和Gumbel噪声平滑，使训练稳定并显著缩小离散化误差。

**🔧 技术方法**

采用的技术包括Walsh–Hadamard变换、可微分逻辑门网络、Gumbel–Softmax（Gumbel-Soft）平滑、残差初始化以及可学习阈值的软阈值化。

**📊 数据集**

实验使用的主要数据集为CIFAR‑10、JSC以及在卷积核实验中使用的单层卷积网络。

**📈 对比分析**

与现有SOTA逻辑网络（如s、DWNs、Light‑LNNs）在相同架构下进行比较，WARP在保持或减少参数量的同时实现更快收敛、离散化误差更小，在高输入阶逻辑块上也能保持或提升最终离散化精度。

**⚠️ 局限性**

局限性包括：尚未在硬件平台上验证推理延迟与资源利用；在极深或极高输入阶的网络中仍存在收敛困难；离散化误差在极深或大输入阶时仍有提升空间。

---

## 597. Riemannian Neural Optimal Transport

**arXiv ID:** 2602.03566 | [PDF](https://arxiv.org/pdf/2602.03566v1)

**作者:** Alessandro Micheli `[一作]` (Imperial), Samir Bhatt `[通讯]` (University Of Copenhagen)

**通讯引用:** 81854 | [OpenAlex ID](https://openalex.org/A5091290326)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种在Riemannian流形上进行无离散化的神经网络最优传输框架RNOT，能够在高维流形上实现可摊销的生成模型。

**💡 创新点**

创新点在于：1) 证明任何离散化的流形OT方法都会受到维数灾难；2) 设计通过c-变换内在实现c-concavity的连续潜在函数；3) 给出对潜在函数与传输映射的多项式复杂度近似上界，从而突破维数灾难。

**🔧 技术方法**

采用了基于特征映射的拉回网络、c-变换构造的隐式c-concave类、ReLU/分段线性激活的深度网络，以及对流形指数映射的利用；训练时使用Kantorovich半对偶目标。

**📊 数据集**

实验数据集包括：1) 地质学上的大陆漂移分布（球面S²上的古今两时点）；2) 合成的高维球面Sⁿ与环面Tⁿ，源为均匀分布，目标为包裹正态分布。

**📈 对比分析**

与RCPM、RCNF和Moser Flow等基准方法对比，RNOT在低维下与RCPM竞争，且在高维维度提升实验中保持稳定的KL与ESS表现，显示出更好的尺度鲁棒性，但训练时间相对较长。

**⚠️ 局限性**

局限性包括：仅针对紧致流形、需要光滑性假设、训练成本高（需迭代求解c-变换），以及对非二次成本和非紧致情况尚未扩展。

---

## 598. When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs

**arXiv ID:** 2602.03554 | [PDF](https://arxiv.org/pdf/2602.03554v1)

**作者:** Bogdan Zagribelnyy `[一作]` (Insilico Medicine AI Limited), Alex Zhavoronkov `[通讯]` (Insilico Medicine AI Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于化学可行性的新评估指标ChemCensor，并基于该指标构建了大型高质量反应数据集CREED（约640万条）以及100条专家验证的全新单步反应基准，随后利用这些数据对大型语言模型进行监督式微调与强化学习微调，得到新的ChemF模型，显著提升了单步逆向合成的化学可行性评分。

**💡 创新点**

创新点在于：①将化学可行性（反应中心与功能基互容性）作为评价核心，突破传统单一“ground‑truth”匹配的局限；②构造了覆盖多种可行路径的CREED数据集；③推出URSA评测框架和新基准；④将ChemCensor直接作为奖励信号进行RL微调，提升模型对化学选择性的隐式学习。

**🔧 技术方法**

主要技术包括：大型语言模型（如GPT、Claude、Qwen等）在SMILES语料上的预训练与指令微调；ChemCensor前端的反应中心与功能基提取与预置数据库匹配；基于ChemCensor分数的奖励函数（GRPO）进行强化学习；以及用于多样性评估的平均最大CC和平均Top‑K CC指标。

**📊 数据集**

使用的数据集有：①CREED（6.4M条经ChemCensor验证的反应）；②URSA 100条全新目标分子（expert‑annotated）；③改进后的USPTO‑50K‑test（去除泄露反应）。

**📈 对比分析**

与传统Top‑K准确率相比，ChemF模型在URSA基准上实现了平均最大CC提升至约0.75（相较于基线0.48），在USPTO‑50K‑test上也保持领先；同时其生成的反应在多样性与可行性上表现更好。

**⚠️ 局限性**

局限性包括：①CREED仍以规则模板为主，可能对极端或新颖反应的覆盖不足；②ChemCensor依赖于公开专利数据库，缺乏实验验证；③模型在极端苛刻功能基（如卤代烷）上仍出现兼容性错误；④基准规模相对较小，可能无法完全覆盖实际药物开发中的多步路径需求。

---

## 599. Don't believe everything you read: Understanding and Measuring MCP Behavior under Misleading Tool Descriptions

**arXiv ID:** 2602.03580 | [PDF](https://arxiv.org/pdf/2602.03580v1)

**作者:** Zhihao Li `[一作]` (Shandong University), Kun Li `[通讯]` (Shandong University)

**通讯引用:** 15299 | [OpenAlex ID](https://openalex.org/A5100377568)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用静态分析工具对10,240个真实MCP服务器进行检测，评估工具描述与实现代码之间的一致性，揭示约13%的服务器存在描述与代码不匹配的安全隐患。

**💡 创新点**

首次在大规模样本上系统测量MCP生态的描述‑代码不一致，提出了MCPDiFF框架，通过调用链构建、LLM语义抽取和向量相似度判定，给出可解释的一致性等级。

**🔧 技术方法**

采用Tree‑Sitter构建多语言AST、函数调用图，利用大语言模型对调用链进行语义分析，再将代码特征与工具描述嵌入向量空间进行相似度比较，判定一致性。

**📊 数据集**

收集自MCP Market、Smithery、MCP World等三大市场的36类目服务器代码，伴随星级、下载来源等元数据，共计10,240个实例。

**📈 对比分析**

方法通过对每个实例构造调用链、抽取特征、向量化后计算相似度，划分为Full、Mostly、Partial、Rare四类；在所有样本上完成，发现大多数一致，但仍有约1,393个服务器属于Partial或Rare，说明存在安全风险。

**⚠️ 局限性**

局限包括仅基于静态分析，无法捕捉运行时动态行为和权限交互；对语言支持不完全；对LLM理解的假设可能与实际不符；未进行完整的攻击演示和漏洞验证。

---

## 600. Multi-Player, Multi-Strategy Quantum Game Model for Interaction-Aware Decision-Making in Autonomous Driving

**arXiv ID:** 2602.03571 | [PDF](https://arxiv.org/pdf/2602.03571v1)

**作者:** Karim Essalmi `[一作]` (Inria), Fawzi Nashashibi `[通讯]` (Inria)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将量子博弈与经典博弈结合的自动驾驶决策框架 QGDM，能够实时处理多车多策略交互。

**💡 创新点**

创新点在于将量子叠加、纠缠与干涉等量子机制引入博弈模型，并设计动态收益与量子电路实现交互感知决策，无需量子硬件。

**🔧 技术方法**

采用经典博弈论、量子博弈理论（Eisert‑Wilkens‑Lewenstein 协议）、量子电路（单量子比特与多量子比特）、以及 highway‑env 仿真环境。

**📊 数据集**

使用 highway‑env 仿真产生的合成数据，涵盖环形交叉、并道与高速路段的不同玩家与策略配置。

**📈 对比分析**

与 IDM、MOBIL、COR‑MP、CG‑NE、CG‑EPD、CG‑MS 等基线方法对比，QGDM 在互动场景下碰撞率最低、成功率最高，高速路段亦无碰撞，性能显著优于传统方法。

**⚠️ 局限性**

局限性包括仅在仿真环境评估、未覆盖密集城市场景、量子参数需手动设定、缺乏学习机制、对更复杂真实交通动态的适应性尚待验证。

---

## 601. Asymmetric Hierarchical Anchoring for Audio-Visual Joint Representation: Resolving Information Allocation Ambiguity for Robust Cross-Modal Generalization

**arXiv ID:** 2602.03570 | [PDF](https://arxiv.org/pdf/2602.03570v1)

**作者:** Bixing Wu `[一作]` (Zhejiang University), Gopala Anumanchipalli `[通讯]` (University of California)

**通讯引用:** 2721 | [OpenAlex ID](https://openalex.org/A5068922218)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Asymmetric Hierarchical Anchoring (AHA) 框架，用音频残差向量量化 (RVQ) 作为语义锚点，指导视频语义特征在共享离散空间中对齐，并通过梯度反转层 (GRL) 对抗解耦、局部滑动对齐 (LSA) 等技术实现音视频跨模态的鲁棒泛化。

**💡 创新点**

创新点包括：① 异步层级锚定，将音频低层 RVQ 代码块设为共享语义锚，消除信息分配歧义；② 用 GRL 对抗解耦替代传统互信息估计，显著降低语义泄漏；③ 引入局部滑动对齐提升细粒度时间对齐；④ 通过“谈话面部解耦实验”验证解耦效果。

**🔧 技术方法**

使用的核心技术包括：Residual Vector Quantization (RVQ)；Gradient Reversal Layer (GRL) 对抗解耦；Local Sliding Alignment (LSA)；Cross‑Modal Contrastive Predictive Coding (Cross‑CPC) 与 Multi‑Modal EMA (MM‑EMA) 代码更新；对抗式对比损失；以及传统的重建、对齐与量化损失。

**📊 数据集**

主要数据集：VGGSound‑AVEL 40K（预训练），AVE、AVVP、UCF‑101、VGGSound（下游转移任务），以及 TalkVid 子集用于面部解耦实验。

**📈 对比分析**

与 MST、CODIS、TURN、CMCM、Unicode、DCID、FCID 等基线在 AVE、AVVP、UCF↔VGG 的零样本跨模态转移任务对比，AHA 在 8 个转移方向上均优于 Unicode，最大提升达 +13.7（AVVP）/ +7.4（AVE），并在跨数据集和细粒度定位任务上保持稳定的性能提升；消融实验验证了 GRL、LSA、音频锚点等模块的关键作用。

**⚠️ 局限性**

局限性：仍依赖音频作为语义锚点，可能对非音频模态扩展不足；需大量音视频配对数据；在极端噪声或长时序场景下的鲁棒性尚未充分验证；深度伪造风险需注意；计算成本相对较高。

---

## 602. EHRWorld: A Patient-Centric Medical World Model for Long-Horizon Clinical Trajectories

**arXiv ID:** 2602.03569 | [PDF](https://arxiv.org/pdf/2602.03569v1)

**作者:** Linjie Mu `[一作]` (Shanghai Jiao Tong University), Xiaofan Zhang `[通讯]` (Shanghai Innovation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了医学世界模型EHRWorld和基于MIMIC‑IV的长期电子健康记录数据集EHRWorld‑110K，用于在多步临床干预下模拟病人状态的持续演化。

**💡 创新点**

创新点在于：①提出因果顺序训练范式，将临床事件视为集成条件生成；②采用双模式预测机制（询问/干预）与确定性状态转移，显著抑制长时间序列中的错误累积；③构建规模达110k住院记录、1.75亿事件的真实长期EHR数据集。

**🔧 技术方法**

技术：基于大型语言模型（Qwen3、Qwen2.5等）进行事件集条件生成；双模式预测（Inquiry/Intervention）；确定性状态更新；因果遮蔽训练；使用DeepSpeed ZeRO‑2、AdamW等训练工具。

**📊 数据集**

数据集：EHRWorld‑110K（来源MIMIC‑IV），包含110,513个住院周期、约17.5M临床事件，按诊断类别分层划分训练/测试集。

**📈 对比分析**

比较方法：闭源LLM（GPT‑5.2、Gemini‑3.0），开源通用LLM（Qwen3‑30B、Llama‑70B、GLM‑4.7等）与医学LLM（MedGemma、Baichuan‑M2）。在全轨迹预测中，EHRWorld在S@25、Stat F1、Label F1、Avg Score等指标上显著优于基线，误差累计更低、保留率最高（92.6% vs 86.2%）。

**⚠️ 局限性**

局限性：①仍使用自回归生成，残余误差可能在长滚动中扩散；②评估侧重轨迹一致性，未检验临床决策或治疗优化的实际效果；③模型可能继承观测数据中的实践偏差，需要慎重外推。

---

## 603. CoGenCast: A Coupled Autoregressive-Flow Generative Framework for Time Series Forecasting

**arXiv ID:** 2602.03564 | [PDF](https://arxiv.org/pdf/2602.03564v1)

**作者:** Yaguo Liu `[一作]` (Science and Technology of China), Qi Liu `[通讯]` (Science and Technology of China)

**通讯引用:** 24581 | [OpenAlex ID](https://openalex.org/A5100453158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种结合预训练大语言模型和流匹配机制的混合生成框架 CoGenCast，用于时间序列预测。

**💡 创新点**

创新点在于：①将解码器仅LLM改造成前向-后向编码解码器骨干；②在LLM生成的自回归表示上加入平均速度的流匹配生成，从而一次性完成连续随机动态建模；③实现了多模态预测与跨域统一训练。

**🔧 技术方法**

技术手段包括：预训练大语言模型（Qwen3-0.6B）、自回归编码解码器、交叉注意力、流匹配生成、线性噪声调度、时间间隔条件速度估计、JVP 正则化训练、一次性推理。

**📊 数据集**

使用了十个公开基准数据集：Energy、ETT（四个子集）、Environment、Exchange、Health、Wind、Solar 等，均为多变量或单变量时序。

**📈 对比分析**

与 LLM 基础方法（LLM4TS、Time-LLM）、生成方法（FlowTS、CDPM、CSDI）、Transformer 方法（TimeDART、PatchTST、Autoformer）等对比，CoGenCast 在大多数数据集上均取得 MSE/MAE 最优或次优，平均提升约 11%/7%。

**⚠️ 局限性**

局限性包括：①模型仍需大量 GPU 训练资源；②对非常长预测区间或高维复杂序列的鲁棒性尚未充分验证；③对不同噪声调度或流匹配参数的敏感性可能影响部署。

---

## 604. Assessing the Impact of Typological Features on Multilingual Machine Translation in the Age of Large Language Models

**arXiv ID:** 2602.03551 | [PDF](https://arxiv.org/pdf/2602.03551v1)

**作者:** Vitalii Hirak `[一作]` (Heinrich Heine University), Arianna Bisazza `[通讯]` (University of Groningen)

**通讯引用:** 2431 | [OpenAlex ID](https://openalex.org/A5019968969)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在大量多语言翻译任务中，对NLLB‑200和Tower+两大预训练模型的翻译质量进行评估，并探讨目标语言的细粒度类型特征及束搜索宽度对性能的影响。

**💡 创新点**

创新点在于首次结合大型预训练模型与细粒度连续型语言类型学特征，系统评估不同语言的翻译难度及束搜索宽度的交互作用，并公开了包含212种语言的细粒度类型学属性数据集。

**🔧 技术方法**

主要技术手段包括使用chrF++指标进行翻译质量评估、利用统计相关性与多元线性回归分析语言属性影响、以及对NLLB‑200与Tower+模型进行不同束宽度的束搜索推理。

**📊 数据集**

实验基于FLORES+多语言评测基准（共计超过200种语言），并使用CommonCrawl数据估算语言资源比例，结合WALS/URIEL等类型学数据库提供语言间距离及类型特征。

**📈 对比分析**

通过对比不同束宽度（k=1,3,5,7）的chrF++得分及生成概率，实验发现目标语言的词形复杂度和语序灵活度显著预测翻译质量，且在Beam宽度上表现出显著差异；Tower+在其覆盖语言上优于NLLB‑200，而在未覆盖语言上则相反。

**⚠️ 局限性**

局限性包括仅评估两款模型、缺乏公开的训练数据比例导致资源估计粗糙、仅使用chrF++等表面级别评测指标、仅考虑左到右束搜索且束宽度范围有限，以及部分语言的细粒度类型学特征缺失。

---

## 605. PnP-U3D: Plug-and-Play 3D Framework Bridging Autoregression and Diffusion for Unified Understanding and Generation

**arXiv ID:** 2602.03533 | [PDF](https://arxiv.org/pdf/2602.03533v1)

**作者:** Yongwei Chen `[一作]` (Nanyang Technological University), Xingang Pan `[通讯]` (Nanyang Technological University)

**通讯引用:** 3757 | [OpenAlex ID](https://openalex.org/A5052549072)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个统一的3D理解与生成框架，融合自回归与扩散两种范式；

**💡 创新点**

创新点在于轻量化的跨模态Transformer连接器，既保持各自强大预训练优点，又实现高效信息交换；

**🔧 技术方法**

技术包括预训练的视觉语言模型（Qwen‑VL‑2.5）、3D VAE（Hunyuan‑3D 2.1）、扩散生成模型、MLP投影、Transformer连接器；

**📊 数据集**

使用了Text‑3D数据集（约32万物体+10级多粒度描述）、3D Editing数据集（约1.4万编辑对）以及40k形状的理解子集；

**📈 对比分析**

与PointLLM、ShapeLLM‑Omni、Trellis、SAR3D等基线比较，文本‑3D生成的Q‑Align最高、CLIP/MUSIQ保持竞争力，编辑任务中能保持身份并产生语义一致的修改；

**⚠️ 局限性**

局限包括对纹理信息缺失导致颜色/材质幻觉、超长token下效果不一定继续提升、编辑对复杂结构的精细控制仍有限。

---

## 606. Morphe: High-Fidelity Generative Video Streaming with Vision Foundation Model

**arXiv ID:** 2602.03529 | [PDF](https://arxiv.org/pdf/2602.03529v1)

**作者:** Tianyi Gong `[一作]` (Chinese University of Hong Kong), Fangxin Wang `[通讯]` (Shenzhen Future Network of Intelligence Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了基于视觉基础模型（VFM）的端到端高保真生成式视频流传输系统。

**💡 创新点**

创新点在于将VFM与自适应时空压缩、分辨率加速、网络自适应控制融合，解决高保真、低延迟、抗丢包三难题，并通过 token 重要性采样、像素残差压缩以及自监督的 token dropping 等技术提升码率与鲁棒性。

**🔧 技术方法**

核心技术包括：Cosmos 视觉基础模型、异步时空压缩与语义重要性采样、像素残差压缩、分辨率缩放加超分、BPF16 计算、BBR 带宽估算、混合丢包处理与自适应帧率控制。

**📊 数据集**

训练使用 UltraVideo‑Long 数据集；评估与基线对比使用 UVG、UHD、UGC、Inter4K 四个公开视频集。

**📈 对比分析**

与 H.264/265/266、Grace、Promptus 等基线对比，采用 SSIM、VMAF、LPIPS、DISTS、时序一致性、延迟和丢包鲁棒性评测；在 400 kbps 下 VMAF 85.17、比 H.266 高约 27 点，带宽节省 62.5%；实现 65 fps 实时编码/解码，25% 丢包率下仍保持低延迟且质量稳定。

**⚠️ 局限性**

局限性包括：对细小文本与标识的细节保留不足，VFM 对文字学习不足导致文字重建不佳，且整体训练和推理算力需求仍高于传统编码。

---

## 607. Live or Lie: Action-Aware Capsule Multiple Instance Learning for Risk Assessment in Live Streaming Platforms

**arXiv ID:** 2602.03520 | [PDF](https://arxiv.org/pdf/2602.03520v1)

**作者:** Yiran Qiao `[一作]` (Institute of Computing Technology), Qing He `[通讯]` (Institute of Computing Technology)

**通讯引用:** 16709 | [OpenAlex ID](https://openalex.org/A5100734672)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了直播间风险评估问题，并提出了一种基于多实例学习的AC‑MIL框架，将用户-时间片段构造成胶囊实例进行室级风险判定。

**💡 创新点**

创新点在于：①将用户-时间片段作为胶囊实例，并通过自适应图注意力建模跨用户、跨时段协同关系；②融合多粒度语义（动作、胶囊、用户视图、时段视图）并通过门控融合实现可解释的风险判定。

**🔧 技术方法**

技术上采用Transformer编码动作序列、LSTM构造胶囊、Graph‑Aware Transformer实现自适应关系注意力、并行用户视图与时段视图，以及跨层门控解码器进行最终预测。

**📊 数据集**

使用抖音（Douyin）工业数据集（May、June 两大数据集）以及后续线上测试数据进行实验。

**📈 对比分析**

与多种序列模型（Transformer、Reformer、Informer）和MIL方法（mi‑NET、AtMIL、AdMIL、MIL‑LET、TimeMIL、TAIL‑MIL）对比，AC‑MIL 在 PR‑AUC、F1、召回率等指标均提升约 4% 并显著降低误报率，达到新的 SOTA。

**⚠️ 局限性**

局限在于：仅使用房间级标签，未充分利用更细粒度的文本或多模态特征；对极端稀疏用户行为的识别仍存在挑战。

---

## 608. Causal Inference for the Effect of Code Coverage on Bug Introduction

**arXiv ID:** 2602.03585 | [PDF](https://arxiv.org/pdf/2602.03585v1)

**作者:** Lukas Schulte `[一作]` (University of Passau), Steffen Herbold `[通讯]` (University of Passau)

**通讯引用:** 1883 | [OpenAlex ID](https://openalex.org/A5027032646)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

建立因果模型并量化代码覆盖率对缺陷引入的因果效应，采用双重稳健回归与广义倾向分数估计剂量‑响应曲线。

**💡 创新点**

首次将因果推断方法系统引入软件工程，全面考虑混杂因素，提供覆盖率阈值和边际收益的量化结果。

**🔧 技术方法**

使用因果图建模、广义倾向分数（GPS）、双重稳健回归、剂量‑响应估计以及改进的SZZ算法。

**📊 数据集**

收集了20个成熟的JavaScript/TypeScript开源项目的提交、代码覆盖率、源代码度量、issue与审查信息以及CI结果。

**📈 对比分析**

通过对比未调整的线性回归与GPS调整后的双重稳健模型，展示了未调整偏倚并在不同覆盖率水平上提供更稳定、可解释的效应估计。

**⚠️ 局限性**

SZZ算法误判率、因果图主观性、仅适用于成熟JS/TS项目、未采用方法控制不可观测混杂等限制。

---

## 609. Flaky Tests in a Large Industrial Database Management System: An Empirical Study of Fixed Issue Reports for SAP HANA

**arXiv ID:** 2602.03556 | [PDF](https://arxiv.org/pdf/2602.03556v1)

**作者:** Alexander Berndt `[一作]` (Heidelberg University), Sebastian Baltes `[通讯]` (Heidelberg University)

**通讯引用:** 1315 | [OpenAlex ID](https://openalex.org/A5033132966)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 SAP HANA 期望报告的自动化标签，系统性分析了工业级数据库管理系统中测试不稳定（flaky）现象的根因及其随时间和测试类型的变化。

**💡 创新点**

创新点在于将大语言模型（LLM）作为注释器，结合模型间一致性和人工校正，实现了大规模、低成本的根因归类，并首次将根因视为多标签问题进行探讨。

**🔧 技术方法**

使用了多模型（GPT-4、Claude 等）进行自动标签，采用温度 0、五次重复问答与多数投票策略，结合手工校正与代码书提升一致性。

**📊 数据集**

数据集由 559 条 SAP HANA 处理过的 flaky 问题报告组成，涵盖 587 个独立测试（系统测试 464 个，单元测试 123 个），时间跨度 34 个月。

**📈 对比分析**

通过与手工标注样本（n=50）对比，LLM 的 Cohen’s Kappa 达到 0.63，模型间 Fleiss’ Kappa 0.78，显示相当可靠；自动化方法大幅提升标签效率，显著降低人工工作量。

**⚠️ 局限性**

局限性包括：仅捕捉开发者提交的错误报告，可能低估基础设施相关的 flaky；多标签归类仍需进一步验证；对其他编程语言或项目的泛化性尚未充分评估。

---

## 610. Investigating the Influence of Spatial Ability in Augmented Reality-assisted Robot Programming

**arXiv ID:** 2602.03544 | [PDF](https://arxiv.org/pdf/2602.03544v1)

**作者:** Nicolas Leins `[一作]` (Zuse Institute Berlin), Sebastian Pokutta `[通讯]` (Zuse Institute Berlin)

**通讯引用:** 1952 | [OpenAlex ID](https://openalex.org/A5043574831)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过对71名参与者进行单因素随机对照实验，比较了基于AR的机器人编程教学（使用Meta Quest 3 HMD、Unity与ROS2）与传统平板+教练台教学在学习体验（系统可用性、认知负荷）和空间能力关联上的差异，并探讨AR是否能弥补学习者空间能力差异。

**💡 创新点**

创新点在于：①首次在真实机器人编程任务中系统性检验AR对空间能力的调节作用；②将空间能力（心理旋转测试）与AR学习体验进行匹配，验证AR作为“补偿机制”的可能性；③采用多维认知负荷与可用性评估，形成了完整的学习体验评价框架。

**🔧 技术方法**

主要技术：Meta Quest 3头戴式显示器、Unity 2022.3 AR应用、ROS2中间件与Universal Robots UR5e机器人接口、手势交互与空间锚定；评估工具包括系统可用性量表（SUS）和三维认知负荷量表。

**📊 数据集**

数据集：71名自愿参与者的人口学信息（性别、年龄）、空间能力得分（心理旋转测试）、技术熟练度（ATI、机器人/编程/AR经验）以及学习后收集的SUS、认知负荷（ECL、ICL、GCL）等问卷数据。

**📈 对比分析**

比较方法：使用多元线性回归分析控制空间能力、性别等协变量，检验AR与对照组在ECL、ICL、GCL、SUS上的差异；进一步进行亚组回归探究AR对空间能力影响的调节作用。结果显示：AR组与对照组在四个学习体验指标上无显著差异；在对照组空间能力越高，ECL越低、SUS越高；在AR组上述关联消失，提示AR具备补偿低空间能力者的潜力。

**⚠️ 局限性**

局限性：①样本量有限，无法充分检验交互项或更细粒度的亚组差异；②AR应用仅实现了空间可视化的核心功能，缺乏交互反馈、适应性指导等可能提升学习效果的要素；③认知负荷采用自评量表，受主观偏差影响，且GCL量表内部一致性低；④实验环境为实验室，缺乏真实工业场景的外部效度。

---

## 611. Sparse Training of Neural Networks based on Multilevel Mirror Descent

**arXiv ID:** 2602.03535 | [PDF](https://arxiv.org/pdf/2602.03535v1)

**作者:** Yannick Lunk `[一作]` (Institute of Mathematics, University of Würzburg), Leon Bungert `[通讯]` (University of Würzburg)

**通讯引用:** 227 | [OpenAlex ID](https://openalex.org/A5035045595)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多层线性化Bregman迭代的稀疏训练框架，动态冻结网络结构并交替进行稀疏模式更新。

**💡 创新点**

创新点在于将稀疏诱导的Bregman迭代与自适应冻结机制相结合，形成多层优化结构；在此框架下给出PL型条件下的收敛保证，并实现显著的计算成本降低。

**🔧 技术方法**

核心技术包括线性化Bregman（等价于镜像下降）、多层（multilevel）优化框架、随机梯度估计、稀疏参数冻结、SparseProp稀疏层实现。

**📊 数据集**

实验使用CIFAR‑10和TinyImageNet两大图像分类数据集。

**📈 对比分析**

与SGD、LinBreg、RigL、稠密模型剪枝+微调等方法比较，所提方法在99%+稀疏率下保持或超过对手精度；在CPU上实现49%训练时间缩减；在测试集上表现至少与RigL相当，且比LinBreg更节省梯度信息。

**⚠️ 局限性**

局限性：仅在无结构稀疏上验证；在GPU上实际加速未验证；收敛证明仅适用于PL条件，需对局部KŁ类型进行进一步研究；对超参数（λ、m、阈值）敏感，需经验调优。

---

## 612. APEX: Probing Neural Networks via Activation Perturbation

**arXiv ID:** 2602.03586 | [PDF](https://arxiv.org/pdf/2602.03586v1)

**作者:** Tao Ren `[一作]` (Aalborg University), Qiongxiu Li `[通讯]` (Aalborg University)

**通讯引用:** 340 | [OpenAlex ID](https://openalex.org/A5062097625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在推理阶段向神经网络隐藏层注入可控噪声，构建APEX框架，用以探测和揭示模型内部表示的结构特征；通过小噪声和大噪声两种 regime，分别捕捉样本级别的正则性和模型级别的全局偏置。

**💡 创新点**

① 在激活空间直接扰动而非输入或参数，提供更大覆盖的表示空间；② 理论上证明噪声从样本依赖过渡到模型依赖，并把输入扰动视为APEX的特殊约束；③ 在大噪声下发现后门模型的目标类别集中现象，提供轻量化的后门检测手段；④ 通过“逃逸噪声”提出单模型即可评估样本正则性的指标。

**🔧 技术方法**

对每层激活添加高斯噪声，使用Monte‑Carlo推理估计输出分布；计算JS散度、熵等统计量；理论上分解前向信号，证明噪声主导行为；对比输入扰动、参数扰动以及不同噪声分布的效果；应用于ResNet、Inception、Vision‑Transformer等架构。

**📊 数据集**

主要实验数据集为CIFAR‑10、CIFAR‑100、ImageNet；对随机标签比例实验、后门攻击（BadNets、Blended、Physical BA、IAD、LIRA）以及不同模型容量的实验。

**📈 对比分析**

与输入噪声、参数噪声以及传统的样本正则性指标（memorization score、consistency score）进行对比；在小噪声 regime 下逃逸噪声与已有指标高度相关；在大噪声 regime 下，后门模型的目标类别概率显著高于正常模型，熵显著降低，表现优于传统后门检测方法；整体计算量仅需单模型多次前向推理，轻量化。

**⚠️ 局限性**

① 需要多次推理才能获得稳健的输出分布估计，计算成本略高；② 对噪声尺度的选择敏感，过大噪声可能导致信息丢失；③ 在Transformer等架构中，部分后门信号的收敛较弱；④ 仅适用于推理阶段，无法实时监控训练过程；⑤ 对模型容量和层数的变化敏感，需针对不同架构调参。

---

## 613. Optimization and Generation in Aerodynamics Inverse Design

**arXiv ID:** 2602.03582 | [PDF](https://arxiv.org/pdf/2602.03582v1)

**作者:** Huaguan Chen `[一作]` (Renmin University of China), Hao Sun `[通讯]` (Renmin University of China)

**通讯引用:** 23541 | [OpenAlex ID](https://openalex.org/A5100375406)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了统一的逆向设计框架，结合基于分布式目标的优化与协方差感知的引导生成，并引入基于SKL的成本预测器训练损失与密度梯度优化；

**💡 创新点**

创新点包括：1）将点解与分布解统一为分布式解并推出密度梯度优化；2）提出基于Symmetric KL的损失训练成本预测器；3）设计时间‑内存高效的协方差估计算法，提升大维引导生成性能；

**🔧 技术方法**

主要技术为：Flow‑matching（DiT/Shape‑VAE）生成模型；Transformer+DiT成本预测器；密度梯度优化；Monte‑Carlo（SA‑MC）引导；低秩 QR/Cholesky 近似协方差；离线强化学习验证；

**📊 数据集**

使用的公开数据集包括车辆 DrivAerNet++、飞机 BlendedNet 以及控制实验用的 2D Gaussian 混合数据；

**📈 对比分析**

与成本梯度、DPS、LGD‑MC、SIM‑MC 等基线对比；在车辆/飞机 CFD 基准中，密度梯度+SKL 预测器实现更低 drag 或 drag/lift，且形状保真更好；SA‑MC 在 OOD 与 RL 场景中表现最佳，单样本耗时 <1 分钟，显存 <40 GB；

**⚠️ 局限性**

局限性在于：仍需昂贵的 CFD/风洞验证；成本预测器训练依赖大量标注数据；协方差估计在极高维或复杂物理约束下可能不稳健；生成多样性受限，尚未解决全局最优探索问题。

---

## 614. Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs

**arXiv ID:** 2602.03578 | [PDF](https://arxiv.org/pdf/2602.03578v1)

**作者:** Su Dong `[一作]` (Hong Kong Polytechnic University), Xiao Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 45764 | [OpenAlex ID](https://openalex.org/A5073869073)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于查询语法复杂度动态路由的检索增强生成框架EA-GraphRAG，能够在单跳与多跳知识问答任务中根据语法特征选择密集检索、图检索或两者融合，显著提升答案准确率与检索召回率，并降低延迟。

**💡 创新点**

创新点主要包括：
1) 语法特征构造器：利用句法分析提取多维结构特征；
2) 轻量级复杂度评分器：将特征映射到连续复杂度分数；
3) 基于分数的多路选择与递归秩融合（wRRF）策略；
4) 在混合难度任务中实现零成本、实时的检索路径切换。

**🔧 技术方法**

使用技术包括：
- 句法分析（Stanza、SpaCy）获取树结构特征；
- 依赖关系与语义特征提取；
- 轻量级多层感知机（MLP）进行复杂度评分；
- 词向量检索（Dense Retriever）与图检索（基于知识图谱的个性化 PageRank）；
- 递归秩融合（Reciprocal Rank Fusion）实现动态混合；
- LLM 生成器（GPT‑4o‑mini）进行答案生成。

**📊 数据集**

数据集：
- 单跳问答：Natural Questions（NQ）和 PopQA；
- 多跳问答：HotpotQA 与 2WikiMultihopQA；
- 混合基准：将上述四个数据集均匀混合，共4000条查询。

**📈 对比分析**

与基线比较：
- 语言模型基线：Llama‑3‑8B、Qwen3‑8B、GPT‑3.5‑turbo、GPT‑4o‑mini；
- RAG基线：BM25、Contriever、ColBERTv2；
- GraphRAG基线：RAPTOR、G‑retriever、LightRAG、KGP、HippoRAG、HippoRAG2。
EA‑GraphRAG 在所有四个单跳/多跳与混合任务上均达到或逼近最优：
- 混合任务 Acc 71.6（GPT‑Acc 76.9），
- 单跳 NQ Acc 69.1，PopQA Acc 75.1；
- 多跳 2Wiki Acc 81.5，HotpotQA Acc 76.3；
- 检索召回率提升 8–10 分；
- 单跳检索延迟约 1.14 s/问，整体系统约 2.19 s/问，明显低于纯图检索。

**⚠️ 局限性**

限制与未来工作：
- 需要手动设定阈值 τ_L、τ_H，调优成本不低；
- 语法特征提取对解析错误敏感，难以处理非标准或口语化查询；
- 预构建知识图谱成本高，难以快速适应新领域；
- 目前仅在公开的英语知识问答任务上验证，跨语言或更大规模的实测尚缺；
- 对极其复杂的推理（多层逻辑链）可能仍需更强的图遍历与推理机制。

---

## 615. NPCNet: Navigator-Driven Pseudo Text for Deep Clustering of Early Sepsis Phenotyping

**arXiv ID:** 2602.03562 | [PDF](https://arxiv.org/pdf/2602.03562v1)

**作者:** Pi-Ju Tsai `[一作]` (National Yang Ming Chiao Tung University), Yi-Ju Tseng `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 1511 | [OpenAlex ID](https://openalex.org/A5051651198)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究提出了一种名为NPCNet的深度聚类网络，用于从早期ICU时间序列电子健康记录中识别临床意义明确的败血症亚型。

**💡 创新点**

创新点在于将时序EHR转换为伪文本以保留完整时间信息，并通过目标导航器（包含分辨率损失和距离损失）在聚类过程中引入临床相关性，使得识别出的四个亚型与SOFA轨迹和预后高度相关。

**🔧 技术方法**

采用了伪文本生成器、深度聚类网络（DCN）、目标导航器、Triplet loss以及类似Transformer位置编码的嵌入技术实现特征学习和聚类。

**📊 数据集**

主要使用MIMIC‑IV 2.2数据库中的19,834例败血症病例进行训练与内部验证，并在eICU 2.0数据库中对结果进行外部验证（13,660例）。

**📈 对比分析**

与传统聚类方法（K‑means、KM‑DTW、DCN、DMK、naviDCN）以及消融实验相比，NPCNet在内部指标（Silhouette、Calinski‑Harabasz、Davies‑Bouldin）和临床指标TDI上均取得显著优势；在四个亚型中，α、β、δ亚型对早期升压剂的治疗效果表现出统计学意义。

**⚠️ 局限性**

主要局限包括：仅使用常规临床变量，未纳入生物标志物或基因组数据；TDI仅衡量轨迹可区分度而非差异幅度；回顾性设计无法证明因果关系；时间戳可能不精准；治疗效果分析仅限于接受升压剂的患者。

---

## 616. Cut to the Mix: Simple Data Augmentation Outperforms Elaborate Ones in Limited Organ Segmentation Datasets

**arXiv ID:** 2602.03555 | [PDF](https://arxiv.org/pdf/2602.03555v1)

**作者:** Chang Liu `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Andreas Maier `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 15582 | [OpenAlex ID](https://openalex.org/A5101619735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究针对多器官分割数据量有限的场景，系统评估并对比了四种交叉图像与对象级数据增强方法（CutMix、CarveMix、ObjectAug、AnatoMix），并与传统数据增强技术结合，提升了基于nnUNet的分割性能。

**💡 创新点**

创新点在于将多种交叉图像和对象级增强方法重新实现并适配多器官分割任务，发现即使产生“错误”图像的CutMix在有限数据下仍能显著提升分割效果，并证明不同增强方法的可互补性。

**🔧 技术方法**

主要技术包括基于nnUNet的深度学习分割框架、四种数据增强策略（CutMix、CarveMix、ObjectAug、AnatoMix）以及与nnUNet默认的传统空间与强度增强（TDA）相结合。

**📊 数据集**

使用的数据集为公开的腹部多器官分割数据集AMOS（约300体积，16器官）和私有的DECT双能CT数据集（42体积，9器官），并在两者上分别取有限样本（20例训练、100/22例测试）进行实验。

**📈 对比分析**

通过在不同增强倍率（×10、×25、×50）下训练模型，并在宏观（macro）和微观（micro）Dice平均值上评估，CutMix在AMOS上微观Dice提升约2.6，宏观Dice提升约4.9，结合TDA后提升可达7.0；在DECT上CutMix的宏观Dice提升约3.1，但微观Dice几乎无变化。

**⚠️ 局限性**

局限性包括：1）实验仅覆盖两类有限数据集，结果可能不易推广到更大规模或不同模态数据；2）ObjectAug表现不佳，说明现有实现与任务匹配度不足；3）对增强方法的“错误”图像效果尚缺乏理论解释，需进一步研究其对网络学习的影响机制。

---

## 617. Can Large Language Models Generalize Procedures Across Representations?

**arXiv ID:** 2602.03542 | [PDF](https://arxiv.org/pdf/2602.03542v1)

**作者:** Fangru Lin `[一作]` (University of Oxford), Janet B. Pierrehumbert `[通讯]` (University of Oxford)

**通讯引用:** 18538 | [OpenAlex ID](https://openalex.org/A5049920688)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究 LLM 在代码、图和自然语言三种表示之间是否能迁移通用程序知识，并提出两阶段 RL 课程。

**💡 创新点**

通过构造同构数据消除表面差异，首次证明可通过“生成类比”实现跨表示迁移，并证明两阶段课程显著提升此能力。

**🔧 技术方法**

使用 Qwen、Llama‑3、Olmo 等多模 LLM，结合 SFT、蒸馏、Self‑Taught Reasoner 和 Group Relative Policy Optimization 等训练方法，并引入 RL 课程。

**📊 数据集**

主要数据集为 AsyncHow 异步规划任务，另外还使用 MATH、SciBench 以及 AAVE 方言版以测试鲁棒性。

**📈 对比分析**

与单一表示训练、零射 GPT‑4o 等基线比较，课程模型在 1.5B Qwen 上实现与 GPT‑4o 近似的规划准确率，且在 AAVE 语料上表现优于同规模零射模型。

**⚠️ 局限性**

局限在于需要大量训练步骤、仅在少数 LLM 家族验证，且跨语言泛化仍受限于训练集分布和表示复杂度。

---

## 618. Group Selection as a Safeguard Against AI Substitution

**arXiv ID:** 2602.03541 | [PDF](https://arxiv.org/pdf/2602.03541v1)

**作者:** Qiankun Zhong `[一作]`, Iyad Rahwan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

构建基于Henrich模型的代理模型，研究生成式AI作为补充与替代两种使用方式对人类文化累积进化的短期与长期影响。

**💡 创新点**

首次将AI使用模式纳入多层次文化演化框架，揭示在群组层面多层选择机制可优先推广“补充型”AI，以保持文化多样性并避免文化崩溃。

**🔧 技术方法**

采用代理模型、演化博弈理论、复制器动力学以及群组结构模拟，结合Monte Carlo仿真评估不同AI策略的收益与文化进化速度。

**📊 数据集**

未使用真实数据集，而是基于理论参数设定（如学习误差α、方差β）进行抽象仿真，使用随机生成的技能分布。

**📈 对比分析**

通过比较无AI、补充AI、替代AI三种策略在平均技能收益与文化累积速度上的表现；结果显示替代AI在短期内获得更高收益，但在群组层面补充AI能够实现更快的长期累积；性能评估以仿真代数和中位技能值为指标。

**⚠️ 局限性**

局限性包括缺乏经验验证与实证数据支撑、模型对AI影响的简化假设、未考虑AI模型多样性与数据集演化对结果的实际影响。

---

## 619. Robust Representation Learning in Masked Autoencoders

**arXiv ID:** 2602.03531 | [PDF](https://arxiv.org/pdf/2602.03531v1)

**作者:** Anika Shrivastava `[一作]`, Samar Agnihotri `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对Masked Autoencoders（MAE）进行系统性研究，分析其在无监督预训练和有监督微调过程中的内部表示层次结构、类相关性以及对输入扰动的鲁棒性。

**💡 创新点**

创新点包括：①首次揭示MAE在预训练阶段就会逐层形成类可分的子空间结构；②利用子空间几何（SVD与主角度）和全局注意力距离两种视角，系统性描述MAE的类结构与全局注意力行为；③提出两种鲁棒性指标——方向一致性（cosine相似度）和头级特征保留计数，量化不同扰动下的表示变化；④在多种人工及真实扰动（高斯模糊、注意力引导遮挡、ImageNet‑C等）上验证MAE的鲁棒性。

**🔧 技术方法**

技术方法包括：MAE ViT‑Base（12层Transformer）模型；t‑SNE可视化；子空间几何分析（对每个类别的patch嵌入做SVD，计算主角度）；全局注意力距离统计；Gaussian blur与attention‑guided occlusion扰动；cosine相似度与头级特征保留计数评估鲁棒性；对比实验使用标准Vision Transformer（ViT）及其他MAE变体。

**📊 数据集**

数据集：ImageNet‑1K（用于预训练、微调与评估），ImageNet‑C（15类噪声/模糊等扰动，5个severity级别），ImageNet‑R与ImageNet‑A（分布偏移与自然对抗图像），SAM（用于注意力距离验证）。

**📈 对比分析**

与标准ViT及MAE的不同预训练/微调配置对比，实验发现：①在预训练阶段，MAE在第7-9层开始出现明显的类分离；②微调后在高斯模糊最高级别（PSNR≈20dB）与50%注意力遮挡下，top‑1准确率仍保持在80%以上；③在ImageNet‑C中，大多数噪声/模糊/天气类型的准确率超过75%，表明MAE具有较强的扰动鲁棒性；相比之下，传统ViT在同类扰动下的准确率下降更快。

**⚠️ 局限性**

limitations: (1) 仍未给出理论解释为何无监督预训练即可形成类可分子空间；(2) 在极端扰动或分布偏移（ImageNet‑R/A）下鲁棒性显著下降；(3) 只评估了两种人工扰动和ViT‑Base模型，未覆盖不同尺度或其他MAE变体；(4) 鲁棒性指标主要关注方向和特征保留，未考虑更细粒度的语义信息或生成性评估。

---

## 620. ZOR filters: fast and smaller than fuse filters

**arXiv ID:** 2602.03525 | [PDF](https://arxiv.org/pdf/2602.03525v1)

**作者:** Antoine Limasset `[一作]` `[通讯]` (University of Lille), Antoine Limasset (University of Lille)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出并实现了一种确定性、可终止的 XOR/Fuse 过滤器——ZOR 过滤器，解决传统算法在构造失败时需要重启的问题；

**💡 创新点**

通过在构造过程中确定性地放弃少量键（而非重启）并将其存入小型辅助过滤器，既保证了构造终止，又保持了与 XOR/Fuse 相同的查询效率，并实现了接近信息理论极限的空间利用率；

**🔧 技术方法**

采用确定性剥离（deterministic peeling）、受限哈希与段化结构、辅助 Fuse/MPHF+指纹存储、以及针对阻塞事件的多种干预策略（如最轻邻域、最重邻域等）等技术；

**📊 数据集**

实验使用大规模随机生成的键集合（10M 以上）进行评估，并通过不同 arity、段大小、指纹长度等参数对 ZOR 进行全面测试；

**📈 对比分析**

与最先进的静态过滤器（Fuse 过滤器）和信息理论基准（MPHF+指纹）进行对比，结果显示 ZOR 在相同指纹大小下，整体空间占用率低于 1% 的信息理论下限，查询时间约 100 ns/键，负面查询成本略高；

**⚠️ 局限性**

主要限制是构造速度相对较慢——ZOR 需要维护显式的邻接信息以实现确定性剥离，导致构造时间比高度优化的 Fuse 过滤器慢约一个数量级；

---

## 621. How to Train Your Resistive Network: Generalized Equilibrium Propagation and Analytical Learning

**arXiv ID:** 2602.03546 | [PDF](https://arxiv.org/pdf/2602.03546v1)

**作者:** Jonathan Lin `[一作]` (University of Southern California), Francesco Caravelli `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 1500 | [OpenAlex ID](https://openalex.org/A5011236474)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在论文中作者提出了一种基于电阻网络的物理学习框架，能够在保持局部性约束的前提下，直接计算电阻网络的梯度并进行训练；

**💡 创新点**

创新点在于引入了通用平衡传播（Generalized Equilibrium Propagation, GEP）理论，将传统的两相学习（Equilibrium Propagation 与 Coupled Learning）统一到一个多阶线性响应框架中，并利用电路的循环空间投影算子 Ω_A/R 计算出精确梯度，避免了有限 nudging 引入的估计偏差；

**🔧 技术方法**

技术上使用了 Kirchhoff 定律的图论解析、线性响应理论、投影算子计算（Ω_A/R 与其转置）、电压/电流模式实验、以及对电阻网络的闭式解析梯度推导；

**📊 数据集**

实验验证采用了 Wisconsin 乳腺癌分类数据集（经过 PCA 降维到 3 维）以及在随机纳米线网络结构上进行的噪声线性回归任务；

**📈 对比分析**

与传统的两相（差平方）学习方法相比，投影算子梯度在同等训练步数下表现出更稳定的收敛曲线，分类准确率可达约 90%，且在部分控制和噪声环境下更具鲁棒性；

**⚠️ 局限性**

局限性包括：仍依赖于对电阻网络的可测电压/电流操作；对于非线性或动态电路需进一步扩展；以及网络拓扑和输入/输出边选择对可表达性有约束，选择不佳会限制学习效果。

---

## 622. $V_0$: A Generalist Value Model for Any Policy at State Zero

**arXiv ID:** 2602.03584 | [PDF](https://arxiv.org/pdf/2602.03584v1)

**作者:** Yi-Kai Zhang `[一作]`, Han-Jia Ye `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了V_0，一种通用值模型，可通过历史指令-性能对作为上下文，零梯度地预测任何策略在状态零（初始提示）下的期望表现，并应用于训练期间的预算分配与推理时的模型路由。

**💡 创新点**

创新点在于将值估计从参数化的函数转变为条件预测问题，利用显式上下文感知策略能力；设计了混合语义感知与结构化推理架构（残差查询适配器+TabPFN），并通过互信息分析识别并消除shortcut学习，采用组合的Pairwise Ranking Loss与Soft Cross‑Entropy提升判别与校准。

**🔧 技术方法**

使用技术包括In‑Context Learning、残差查询适配器、多头注意力、TabPFN贝叶斯推理、Pairwise Ranking Loss、Soft Cross‑Entropy、GRPO、预算分配与成本加权路由算法。

**📊 数据集**

实验数据集涵盖DAPO‑Math‑17k、OlympiadBench、AIME‑24/25、GPQA‑Diamond以及Open‑Reasoner‑Zero 57k等，构建多模型、多训练阶段的历史交互数据。

**📈 对比分析**

与传统PPO价值模型、奖励模型、kNN、逐步重训练的价值模型等进行对比，V_0在Intra‑AUC、校准误差、预算利用率上均表现优异，能够逼近Pareto最优的成本‑性能曲线，训练时与重训练方法相当但无额外梯度开销。

**⚠️ 局限性**

局限性在于目前仅实现对状态零的粗粒度值估计，对更细粒度的token级动态评估尚未展开；对极少样本或极端上下文稀缺情况的鲁棒性待进一步验证。

---

## 623. Secure Decentralized Pliable Index Coding for Target Data Size

**arXiv ID:** 2602.03579 | [PDF](https://arxiv.org/pdf/2602.03579v1)

**作者:** Anjali Padmanabhan `[一作]` (National Institute of Technology Calicut), Shanuja Sasi `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5022319079)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在 LPS-FO 侧信息结构下，满足严格安全约束的分布式可调索引编码（Secure DPIC）传输方案；

**💡 创新点**

在异构侧信息与单消息需求的前提下，首次将安全约束纳入 DPIC，并给出了针对不同客户端数的递归传输策略，取得了在 C=3 或 4 时的最优结果；

**🔧 技术方法**

采用递归算法（Algorithm 1）与两种子方案（Algorithm 2 与 Algorithm 3）结合线性递进集合与固定重叠的 XOR 编码；

**📊 数据集**

本文未使用公开数据集，而是基于理论模型 LPS-FO 的人工构造侧信息集合；

**📈 对比分析**

与传统 DPIC 的无安全约束下的最小传输次数（C）比较，安全方案在 C≥5 时额外传输量为 N(C−r_max)，实验表明在 C=3、4 时两方案相同；

**⚠️ 局限性**

局限在于仅适用于满足 K≥2P、P≥r_max−2 的特定侧信息结构；安全约束导致传输开销较大，且对更一般侧信息分布缺乏通用证明。

---

## 624. EVE: Efficient Verification of Data Erasure through Customized Perturbation in Approximate Unlearning

**arXiv ID:** 2602.03567 | [PDF](https://arxiv.org/pdf/2602.03567v1)

**作者:** Weiqi Wang `[一作]` (University of Technology Sydney), Shui Yu `[通讯]` (University of Technology Sydney)

**通讯引用:** 27902 | [OpenAlex ID](https://openalex.org/A5005228053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出EVE方法，利用定制扰动验证机器学习模型的近似数据抹除效果，而不依赖模型初始训练过程

**💡 创新点**

创新点在于不需要在训练阶段植入后门，仅通过在抹除数据上施加梯度匹配的扰动，使未抹除模型对指定样本的预测发生可观测变化，且给出统计显著性检验

**🔧 技术方法**

使用对抗优化、梯度匹配、误差极限约束和t检验等技术实现扰动生成与验证

**📊 数据集**

在MNIST、CIFAR‑10、STL‑10和CelebA四个公开图像数据集上进行实验

**📈 对比分析**

与后门基方法MIB和无后门方法TAPE对比，EVE在100%可验证率、≈160倍的速度提升以及保持模型准确率方面均优于两者

**⚠️ 局限性**

仅适用于近似抹除，无法验证完全抹除或非近似方法

---

## 625. HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing

**arXiv ID:** 2602.03560 | [PDF](https://arxiv.org/pdf/2602.03560v1)

**作者:** Yizhao Gao `[一作]`, Fuli Luo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种混合稀疏注意力架构Hybrid Sparse Attention，将每个全注意力层与多层稀疏注意力层交错，并利用前一层全注意力产生的 token 重要性和 KV 缓存作为稀疏层的“oracle”输入，从而实现低计算量且无额外 KV 开销。

**💡 创新点**

通过 Oracle token selection 和跨层 KV 缓存共享实现无代理的稀疏 token 选取与内存优化，同时在稀疏层加入局部滑动窗口分支实现全局与局部信息融合。

**🔧 技术方法**

采用 FlashAttention 改写输出块级注意力得分进行 TopK 选取，使用 Grouped‑Query Attention 做索引共享；稀疏层包含两支（全局稀疏和局部 SWA），并通过 sigmoid 门融合；训练时采用 1:3/1:11 混合比例，MoE 专家模式与全局注意力混合。

**📊 数据集**

训练使用 1T/200B/500B tokens（序列长度 8K/32K），评测基准涵盖 BBH、MMLU、MMLU‑Redux、MMLU‑Pro、DROP、ARC‑Challenge、HellaSwag、WinoGrande、TriviaQA、GSM8K、MATH、HumanEval、MBPP、C‑Eval、CMMLU 以及长上下文 RULER。

**📈 对比分析**

与全注意力和 Hybrid SWA 基线对比，Hybrid Sparse 在 7B dense 与 80B MoE 模型上保持或提升各类基准准确率的同时，将 KV 缓存压缩至 1/10、计算量大幅下降；在长上下文任务中，表现与全注意力相当或更优，尤其在 80B MoE 仅 5 层全注意力即可超越全注意力。

**⚠️ 局限性**

仍需至少保留全注意力层做“oracle”，无法完全消除 O(n²) 计算；极端稀疏比例下局部信息依赖不足，KV 缓存共享对 SWA 会导致性能下降；系统级 KV offloading 与更大规模训练仍未实现。

---

## 626. Formal Evidence Generation for Assurance Cases for Robotic Software Models

**arXiv ID:** 2602.03550 | [PDF](https://arxiv.org/pdf/2602.03550v1)

**作者:** Fang Yan `[一作]` (University of York), James Baxter `[通讯]` (University of York)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5055266310)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于模型的自动化流程，将形式化验证结果嵌入安全保障案例（Assurance Case）中，实现安全证据的自动生成与维护。

**💡 创新点**

创新点在于：①使用预定义模板将自然语言需求转化为形式化断言；②协同多种形式化验证工具（FDR、PRISM、Isabelle）以覆盖不同属性类型；③通过Eclipse EMF框架实现需求结构化、断言生成、验证执行及证据集成的全流程自动化。

**🔧 技术方法**

技术包括：RoboChart域特定建模语言、CSP和PRISM断言DSL、Isabelle/Z‑Machine证明方法、Kapture需求管理工具、RoboTool、Epsilon模型转换、GSN/SACM安全案例元模型。

**📊 数据集**

使用四个机器人案例（邮件递送机器人、喷漆机器人、海底无人机、维修机器人）作为评估数据集，每个案例对应不同类型需求与验证工具。

**📈 对比分析**

在四个案例中实现了从需求到证据的全链路自动化，平均验证时间从数十毫秒到几百毫秒，证据生成与集成耗时在几十毫秒以内，展示了较高的效率和实用性。

**⚠️ 局限性**

局限性：依赖于RoboChart及其形式化语义；对未覆盖的需求模式（如即时响应、数据驱动需求）需扩展模板；多工具协同仍需手工维护模板映射与工具集成。

---

## 627. SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue

**arXiv ID:** 2602.03548 | [PDF](https://arxiv.org/pdf/2602.03548v1)

**作者:** Yuqin Dai `[一作]`, Chaozheng Wang `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了SEAD框架，构建一个零注释、可自进化的服务对话代理，利用分离的用户建模实现可控、真实的多轮交互；

**💡 创新点**

核心创新在于将用户建模拆分为Profile Controller与User Role‑Play Model，形成“下注游戏”式的初始状态采样与黄金情境挖掘，同时通过错误分析实现自适应难度曲线，避免传统自对抗训练中的奖励黑客；

**🔧 技术方法**

采用自进化（自生成自评）与GRPO（Group Relative Policy Optimization）强化学习技术，结合LLM驱动的角色扮演模型和基于用户状态空间的动态模拟；

**📊 数据集**

仅使用标准作业流程(SOP)、用户特征库（来源于10万+真实企业对话的匿名行为模式）以及可枚举的初始用户状态组合，不依赖任何标注对话数据；

**📈 对比分析**

与多款开源基础模型（Qwen2.5‑14B/32B/72B）及闭源商业API（GPT‑4o、DeepSeek‑Chat、Qwen3‑235B、LongCat‑Flash）在同一服务对话任务上对比，SEAD在14B参数规模下完成率达52.0%，比GPT‑4o提升17.6%，比预训练14B提升34.4%，平均完成轮数9.6，效率领先，且成本保持为零；

**⚠️ 局限性**

局限性包括：评估主要聚焦任务完成率，缺乏对用户满意度与情感体验的深入测评；仅测试单一场景，未覆盖多场景或多域；对多样化用户策略的泛化尚未彻底验证；

---

## 628. AffordanceGrasp-R1:Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping

**arXiv ID:** 2602.03547 | [PDF](https://arxiv.org/pdf/2602.03547v1)

**作者:** Dingyi Zhou `[一作]` (Technical University of Munich), Hu Cao `[通讯]` (Technical University of Munich)

**通讯引用:** 6865 | [OpenAlex ID](https://openalex.org/A5011193488)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于推理的 AffordanceSegmentation 框架 AffordanceGrasp-R1，用以将语言指令转化为机器人抓取动作，并通过全局点云+掩码过滤实现更具语义上下文的抓取候选选择。

**💡 创新点**

创新点包括：① 用链式思维（CoT）冷启动数据集提升 MLLM 的推理结构；② 采用多阶段后训练（SFT + RL）实现推理与空间定位的协同优化；③ 重新设计抓取管线，先生成全局场景点云的抓取候选，再用指令条件的 affordance 掩码过滤；④ 将 SAM‑2 与 LoRA 结合，得到高质量、低成本的像素级 affordance 掩码。

**🔧 技术方法**

技术手段：多模态大型语言模型（MLLM）+ Chain‑of‑Thought 推理、强化学习（GRPO）、SAM‑2 语义分割+LoRA、点云抓取模型、指令条件的空间提示（边界框/点）和全局点云过滤。

**📊 数据集**

使用的数据集：RAGNet 评测基准（HANDAL、GraspNet、3DOI）、新构建的 CoT 推理数据集、以及真实机器人零样本抓取实验。

**📈 对比分析**

与 SOTA 方法（AffordanceNet、LISA‑7B、Segzero 等）在 gIoU/cIoU 指标上进行对比；AffordanceGrasp‑R1 在所有子数据集上均取得最高分（例如总体 gIoU/​cIoU 分别为 66.7%/65.9%）。在实际抓取实验中，易指令成功率从 62% 提升至 80%，难指令从 50% 提升至 72%。

**⚠️ 局限性**

局限性：① 仍主要针对静态单步抓取场景，难以处理动态或多步骤操作；② 对 SAM‑2 的依赖导致模型对预训练分割的敏感性；③ 推理过程可能出现微小对齐误差，导致掩码过滤失效；④ 需要手工构造 CoT 数据集，规模和多样性有限。

---

## 629. Interpretable Logical Anomaly Classification via Constraint Decomposition and Instruction Fine-Tuning

**arXiv ID:** 2602.03530 | [PDF](https://arxiv.org/pdf/2602.03530v1)

**作者:** Xufei Zhang `[一作]` (Beijing XingYun Digital Technology Co., Ltd.), Jianxiong Wang `[通讯]` (Beijing XingYun Digital Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了逻辑异常分类（LAC）任务，并设计了 LogiCls 框架实现可解释的逻辑异常检测与细粒度分类。

**💡 创新点**

创新点在于将复杂逻辑约束拆分为可验证子查询，构建基于链式推理（CoT）的指令式数据集，并引入难度感知再采样提升模型对长尾与难题的鲁棒性。

**🔧 技术方法**

采用小规模视觉语言模型（VLM）通过指令微调、细粒度 CoT、图像文本增广以及多轮链式推理实现逻辑推理；同时利用 SAM/CLIP 进行开源分割与对象定位。

**📊 数据集**

在工业视觉检测领域的 MVTec LOCO FC 数据集上进行实验，该数据集已按单一与多重异常类型重新划分。

**📈 对比分析**

相较于多种大型闭源/开源 VLM（Gemini‑2.5‑pro、GPT‑4o、InternVL3‑78B 等），LogiCls 在二分类 F1 与宏观 F1 指标上均实现近 100% 及 94% 的得分，显著优于基线模型。

**⚠️ 局限性**

局限性包括对视觉相似物体的细粒度辨识仍不够精准，以及对细微相对尺度异常的识别存在偏差，需进一步整合层次几何先验与全局-局部注意机制。

---

## 630. Rank-Learner: Orthogonal Ranking of Treatment Effects

**arXiv ID:** 2602.03517 | [PDF](https://arxiv.org/pdf/2602.03517v1)

**作者:** Henri Arno `[一作]` (Ghent University), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段的正交学习器（Rank‑learner），直接从观测数据中学习个体治疗效果的排序，而非先估计 CATE 再排序。

**💡 创新点**

创新点在于：①引入了基于影响函数的正交对比损失，使得排名目标对杂项估计误差鲁棒；②将排名问题从硬判别（二元指标）平滑为软概率，兼顾可微分性与正交性；③实现了可扩展的对样本对随机子采样，保持计算效率。

**🔧 技术方法**

主要技术包括：跨验证的 Nuisance 估计（响应面与倾向得分），正交化的对比损失（Neyman‑orthogonal pairwise ranking loss），基于深度前馈网络的对比学习以及对 κ 参数的调优。

**📊 数据集**

使用了合成数据（可变样本量）以及三种半合成基准：MovieLens（推荐系统）、MIMIC‑III（医疗）、CPS（公共政策）来验证方法。

**📈 对比分析**

与 T‑learner、DR‑learner、Plug‑in ranker 及树基排序器进行比较。Rank‑learner 在 AUTOC 指标上连续优于所有基线，尤其在小样本和有限重叠场景下提升显著。

**⚠️ 局限性**

局限性包括：依赖无偏估计（unconfoundedness）和足够重叠；对 κ 的选择需要验证；对样本对的随机子采样虽然降低成本，但在极小数据集可能不足；正交化虽然提升稳健性，但实现复杂度更高。

---

## 631. Not All Negative Samples Are Equal: LLMs Learn Better from Plausible Reasoning

**arXiv ID:** 2602.03516 | [PDF](https://arxiv.org/pdf/2602.03516v1)

**作者:** Zixiang Di `[一作]` (East China Normal University), Jie Wang `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了“可疑负样本”（PNS）方法，用逆向强化学习生成形式正确但答案错误的高质量负样本，并将其作为偏好优化（DPO）的训练数据，显著提升LLM的数学推理能力。

**💡 创新点**

创新点在于：①提出逆向奖励结构，鼓励模型产生貌似合理却错误的推理链；②构造多维复合奖励（格式合规、答案准确率翻转、奖励模型评估、链式思考质量）以生成“可疑负样本”；③将此样本作为plug‑and‑play负样本在多模型、多基准上实现性能提升。

**🔧 技术方法**

技术包括：逆向GRPO强化学习、中心化Bradley‑Terry奖励模型训练、格式规则+LLM评估混合格式分数、分桶裁剪的RM分数、链式思考评分以及DPO偏好学习。

**📊 数据集**

使用的数据集主要有：DAPO‑Math（训练RM和RL）、MATH‑500、AIME'24/25、AMC、Olympiad、ARC、GPQA‑Diamond，覆盖面向数学推理的中、外域基准。

**📈 对比分析**

与基线（instruction‑tuned）、RL、RL+负样本、RL+RS、RL+LLM‑Judge等方法对比，PNS在三大骨干模型（Qwen2.5‑7B/3B、Llama3.1‑8B）和七个数学基准上平均提升约2.03%（在AIME、ARC、GPQA上表现尤为突出），显示其对难度更高任务的显著改进。

**⚠️ 局限性**

局限性：①需要额外的逆向RL训练与奖励模型调优，计算成本较高；②奖励模型质量决定负样本可信度，对其它领域（非数学）迁移效果尚未充分验证；③负样本仍可能存在“奖励剥削”或格式缺陷，需进一步完善分桶和规则约束。

---

## 632. Universal One-third Time Scaling in Learning Peaked Distributions

**arXiv ID:** 2602.03685 | [PDF](https://arxiv.org/pdf/2602.03685v1)

**作者:** Yizhou Liu `[一作]` (Massachusetts Institute of Technology), Jeff Gore `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 11205 | [OpenAlex ID](https://openalex.org/A5003202779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究发现softmax与交叉熵损失在学习峰值分布时会自发产生1/3的幂律时间缩放。

**💡 创新点**

创新点在于从网络非线性出发解释神经缩放，指出软最大化与交叉熵是导致幂律的根本机制，并给出普适指数1/3。

**🔧 技术方法**

使用单层softmax模型、梯度流分析、对齐学生假设、极限温度展开、以及Adam/SGD优化器对训练过程的理论与数值验证。

**📊 数据集**

实验数据来自Pythia系列大语言模型的公开检查点，使用FineWeb数据集进行训练曲线分析。

**📈 对比分析**

通过动态时间τ拟合损失，发现不同模型尺寸下损失与τ的幂律指数≈1/3，验证理论预测，且与传统数据规模缩放的0.28指数相比更精确。

**⚠️ 局限性**

局限包括未考虑梯度噪声、学习率/权重衰减对旋转动态的完整理论描述，以及对多层结构的推广仍待进一步证明。

---

## 633. ContraLog: Log File Anomaly Detection with Contrastive Learning and Masked Language Modeling

**arXiv ID:** 2602.03678 | [PDF](https://arxiv.org/pdf/2602.03678v1)

**作者:** Simon Dietz `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Bjoern M Eskofier `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无日志解析器、基于自监督对比学习和掩码语言模型的连续嵌入预测框架 ContraLog，用于检测日志中的异常。

**💡 创新点**

创新点包括：①直接使用连续消息嵌入替代离散模板，保留变量信息；②结合对比学习与掩码语言模型进行端到端训练；③设计分层结构（MessageEncoder + SequenceEncoder）和缓存机制以降低序列长度和计算成本；④融合上下文异常与点异常两种评分，提升对不同异常类型的检测能力。

**🔧 技术方法**

采用 Transformer 结构的 MessageEncoder 与 SequenceEncoder，BPE 字节对编码 tokenizer，Masked Language Modeling 与 InfoNCE 对比学习，Robust z‑score 校准与 L2 聚合得分，点异常通过最近邻距离计算。

**📊 数据集**

在三大公开日志基准上评估：HDFS、BGL、Thunderbird。

**📈 对比分析**

与 LogBERT、DeepLog、OCSVM、Isolation Forest 等方法对比。ContraLog 在 HDFS、BGL、Thunderbird 的 F1 分别达到 83.12%、97.43%、97.51%，在大多数数据集上均优于或持平于传统基于解析器的深度学习与统计方法。

**⚠️ 局限性**

局限性：仍需为每个数据集训练专属 BPE tokenizer，导致预处理成本；对不同数据集最佳特征组合不确定，需要手工调优；对训练集异常比例敏感；当前实验基于离线批处理，实时性能及对极少见日志模板的泛化能力尚待进一步验证。

---

## 634. Instruction Anchors: Dissecting the Causal Dynamics of Modality Arbitration

**arXiv ID:** 2602.03677 | [PDF](https://arxiv.org/pdf/2602.03677v1)

**作者:** Yu Zhang `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 59770 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过信息流分析方法，对多模态大型语言模型（MLLM）的模态跟随机制进行深入拆解，揭示指令标记为关键结构锚点，深层注意力层完成模态仲裁，MLP层产生语义惯性。

**💡 创新点**

创新点在于：①将信息流视角与因果注意力剪切结合，首次定位指令锚点及跨模态中继机制；②发现浅层注意力为潜在缓冲层，深层注意力为仲裁器；③识别出稀疏的专用注意力头，并通过阻断/放大实验验证其因果必需与充分性。

**🔧 技术方法**

主要技术包括：Transformer架构的因果注意力剪切（Causal Attention Knockout）、Logit Lens投影、归一化符号结构发散（I_NSSD）、潜在决策一致率（LDAR）以及头部级别因果干预。

**📊 数据集**

构造了跨模态冲突诊断数据集，包含视觉上下文、文本上下文、指令以及答案实体词典（涵盖多种表述），用于训练/评估模态跟随。

**📈 对比分析**

与传统的“直接关注”或“文本代理”方式对比，实验显示指令锚点路径阻断可使模态跟随率骤降60%，而对关键头部的放大可提升近60%；在Qwen2.5‑VL‑7B和InternVL‑3‑8B等模型上验证，稀疏头部控制对性能影响显著。

**⚠️ 局限性**

局限在于：仅在控制的跨模态冲突场景下验证，缺乏对更复杂、多轮对话等实际任务的泛化评估；所用的因果干预方法依赖对Transformer内部结构的完整可观测性，对非Transformer架构适用性未知。

---

## 635. Mitigating Conversational Inertia in Multi-Turn Agents

**arXiv ID:** 2602.03664 | [PDF](https://arxiv.org/pdf/2602.03664v1)

**作者:** Yang Wan `[一作]` (Zhejiang University), Linchao Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 6090 | [OpenAlex ID](https://openalex.org/A5043617790)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Context Preference Learning 与 clip context 方法，以缓解多轮对话代理中的会话惯性问题。

**💡 创新点**

首次识别会话惯性并利用长短上下文偏好对比构造无环境奖励的偏好学习；引入周期性上下文清除的 clip context。

**🔧 技术方法**

使用 Transformer 语言模型（如 Qwen3-8B、Llama3.1-8B、GPT-4o-mini）、LoRA 微调、DPO 偏好学习、注意力分析与 KV 缓存优化。

**📊 数据集**

在 AgentGym 提供的八种多轮代理环境以及深度研究场景 BrowseComp 进行评测。

**📈 对比分析**

与全上下文、滑窗和摘要管理等基线对比，clip context 与 CPL 结合可使 4‑7 倍预填速度提升，且在大多数环境中性能提升约 3‑5%（如 Qwen3-8B 约 4%）。

**⚠️ 局限性**

仍需平衡信息保持与惯性抑制，长周期任务下信息丢失与总结质量的影响未完全解决。

---

## 636. RAGTurk: Best Practices for Retrieval Augmented Generation in Turkish

**arXiv ID:** 2602.03652 | [PDF](https://arxiv.org/pdf/2602.03652v1)

**作者:** Süha Kağan Köse `[一作]` (Roketsan Inc), Çağrı Toraman `[通讯]` (Middle East Technical University)

**通讯引用:** 270 | [OpenAlex ID](https://openalex.org/A5040060728)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了基于土耳其维基百科与CulturaX的两部分土耳其RAG基准，并在七个核心阶段（查询转换、重新排序、过滤与选择、上下文增补、压缩、提示构造、答案精炼）上进行无任务微调的端到端评估。

**💡 创新点**

创新点在于首次提供完整土耳其RAG管线系统性评估、设计Pareto最优配置、揭示过度堆叠生成模块会扭曲形态学线索，并公开数据与代码。

**🔧 技术方法**

采用HyDE查询生成、cross‑encoder重排、上下文增补、LLM驱动查询澄清以及基因搜索的管线优化等技术。

**📊 数据集**

使用约1.12万条土耳其文本，包含6,305条Web页面与4,891条维基百科文章，生成20,459个问答对并标注主题与章节切分。

**📈 对比分析**

在不微调模型的条件下，HyDE+cross‑encoder+context augmentation组合的准确率达85%（高于基线78.7%），而Pareto最优组合实现84.6%且成本显著降低。

**⚠️ 局限性**

局限包括仅覆盖公开Web与维基百科文本，未检验专业领域；依赖LLM过滤可能带来偏差；结果受语料、硬件或超参数变化影响。

---

## 637. Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration

**arXiv ID:** 2602.03647 | [PDF](https://arxiv.org/pdf/2602.03647v1)

**作者:** Bowei He `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Irwin King `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 27324 | [OpenAlex ID](https://openalex.org/A5042251906)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Search‑R2 框架，采用 Actor 生成初始推理轨迹，Meta‑Refiner 对轨迹进行诊断并通过 cut‑and‑regenerate 机制修正错误，二者共同训练以提升检索‑推理任务的准确率。

**💡 创新点**

创新点在于将检索错误定位与局部修正拆分为判别器和修剪器两子模块，并用混合奖励（结果正确性 + 证据信息密度）进行多尺度信用分配，理论证明其在混合策略下可严格优于拒绝采样，且实现了高效的样本利用。

**🔧 技术方法**

使用的技术包括：Actor–Refiner 结构、Meta‑Refiner（判别器 + 修剪器）、cut‑and‑regenerate 机制、混合奖励设计、Group Relative Policy Optimization (GRPO)、E5 检索器、Qwen 系列大型语言模型以及多步推理工具调用。

**📊 数据集**

实验数据集涵盖：NQ、TriviaQA、PopQA（通用 QA）以及 HotpotQA、2WikiMultiHopQA、Musique、Bamboogle（多跳 QA）七个基准。

**📈 对比分析**

与直接推理、CoT、RAG、IRCoT、Search‑o1、SFT、RL 基线、Rejection Sampling、以及 Search‑R1 进行对比，Search‑R2 在 7B‑32B 模型上平均提升 8–10% EM，尤其在多跳 QA 上提升超过 10%，且在相同计算预算下优于双倍样本搜索的 Search‑R1。

**⚠️ 局限性**

局限性包括：仍依赖检索质量，对超参数（如 max revision）敏感；模型规模越大计算成本提升略微明显；缺乏在更大规模模型、不同语言或更长推理链条上的验证。

---

## 638. TRE: Encouraging Exploration in the Trust Region

**arXiv ID:** 2602.03635 | [PDF](https://arxiv.org/pdf/2602.03635v1)

**作者:** Chao Huang `[一作]` (Institute of Information Engineering), Tingwen Liu `[通讯]` (Institute of Information Engineering)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5103214505)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型（LLM）的强化学习中探索不足问题，提出了 Trust Region Entropy (TRE) 机制，限制熵正则化仅在模型信任区间内进行，从而提升推理与对齐性能。

**💡 创新点**

创新点在于将传统熵正则化的全局均匀化改为局部熵最大化，利用模型自身的 top‑K 或 nucleus 选取的“信任区”避免把概率质量泄漏到无效词汇尾部，解决长序列累计噪声导致的推理崩溃。

**🔧 技术方法**

技术核心是对奖励优化的 PPO 损失中加入局部熵正则项（TRE-K / TRE-P），并通过对 logits 进行 top‑K 或累积概率阈值截断构造信任区，使用软最大化对局部分布求熵并按对数词表大小比例缩放。

**📊 数据集**

在三类任务上评估：数学推理（MATH），组合搜索（Countdown）和偏好对齐（HH, UltraFeedback），分别使用 Qwen2.5‑1.5B/7B 作为基础模型，最大生成长度 8,192/512/1,024 令实验具有多样性。

**📈 对比分析**

与 Vanilla PPO、标准熵正则化、Forking‑Tokens、KL‑Cov 等基线对比，TRE 在 MATH 与 Countdown 上提升 1–3% Pass@1，HH 上奖励提升 0.15–0.64，整体性能优于其他探索方法，尤其在大模型与对齐任务上优势显著。

**⚠️ 局限性**

局限性包括：对极长生成（>128k 令牌）或极大模型（>7B）未充分验证，超长 CoT 的连贯性仍需探究；TRE 的超参数（K、P）对不同任务与模型仍有敏感性，需要更系统的调优策略。

---

## 639. SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection

**arXiv ID:** 2602.03634 | [PDF](https://arxiv.org/pdf/2602.03634v1)

**作者:** Wei Zhang `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5700 | [OpenAlex ID](https://openalex.org/A5043503650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种Sparse Partial Weakly-supervised Oriented Object Detection（SPWOOD）框架，能够仅凭稀疏、弱标签与无标签数据完成遥感图像的旋转目标检测。

**💡 创新点**

创新点包括稀疏注释学习模块、基于对称的方向学习与尺度学习方法、以及多层伪标签过滤（MPF）机制，并提出整体稀疏采样策略，显著降低标注成本。

**🔧 技术方法**

使用教师‑学生 EMA 框架、Gaussian Mixture Model 伪标签过滤、Focal Loss 改进、对称学习、Gaussian/Wasserstein 距离等技术实现模型训练与伪标签筛选。

**📊 数据集**

实验基于 DOTA‑v1.0、DOTA‑v1.5、DIOR 三个遥感数据集。

**📈 对比分析**

与 SOOD、WOOD、SAOD、RSST 等现有方法比较，SPWOOD 在稀疏‑弱标签条件下 mAP 提升数个百分点，表现优于同类方法。

**⚠️ 局限性**

局限在仅使用单一视觉模态，未结合多模态信息，对极稀疏类别的泛化仍有限。

---

## 640. CALM: A Self-Adaptive Orchestration Approach for QoS-Aware Routing in Small Language Model based Systems

**arXiv ID:** 2602.03632 | [PDF](https://arxiv.org/pdf/2602.03632v1)

**作者:** Hemang Jain `[一作]` (International Institute of Information Technology Hyderabad), Karthik Vaidhyanathan `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 CALM，一个基于 MAPE‑K 的自适应编排框架，用于在多域专用小型语言模型（SLM）集群中动态路由、缓存与调度，以实现低延迟、高能效的 AI 服务。

**💡 创新点**

创新在于将自适应反馈循环与模型缓存、语义相似度路由结合，支持在 GPU 受限环境下按 QoS 目标动态选择、加载和清除 SLM，实现比单一 LLM 更优的延迟、能耗和置信度。

**🔧 技术方法**

使用 MAPE‑K 循环、语义相似度（Cosine、MiniLM）路由、LRU 缓存、基于动态指标的自适应权重 λ、以及多模型 SOTA 基础（Qwen2.5‑3B、Phi‑3‑mini‑4k 等）和 LoRA 微调。

**📊 数据集**

在医疗、法律、金融等域使用自定义指令式数据集（如医学问答、法律咨询、财务查询），并通过模拟 500 条用户请求结合 FIFA 世界杯访问分布进行实验。

**📈 对比分析**

对比单一 LLM 基线（Deepseek‑MOE 16B、Llama‑2‑13B、Qwen2.5‑14B）和多种消融配置，实验显示 CALM 在 40–50% 的延迟与能耗下降、15–30% 的置信度提升，且在不同工作负载与跨域场景下保持稳健。

**⚠️ 局限性**

局限在于依赖语义描述的路由易受表述细微变化影响，缓存和自适应参数需经验调优，实验仅在单 GPU 设定且使用 LLM‑as‑Judge 评估，真实多轮对话和更大规模集群尚未验证。

---

## 641. Multi-Objective Optimization for Synthetic-to-Real Style Transfer

**arXiv ID:** 2602.03625 | [PDF](https://arxiv.org/pdf/2602.03625v1)

**作者:** Estelle Chigot `[一作]` (Fédération ENAC ISAE-SUPAERO ONERA), Dennis Wilson `[通讯]` (Fédération ENAC ISAE-SUPAERO ONERA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过多目标进化算法，寻找能在保持语义结构的同时，将合成图像迁移至真实图像风格的增广流水线。

**💡 创新点**

将增广流水线建模为可变长度的旅行商问题（TSP），并利用成对图像度量（DISTS 与 DreamSim）实现高效搜索，首次展示了用进化方法自动设计多样化风格迁移管道。

**🔧 技术方法**

NSGA‑II 进化算法、图像处理算子（传统与深度学习算子 AdaIN、CACTI、ControlNet）、成对图像相似度度量（DISTS、DreamSim）以及分布度量（CMMD）。

**📊 数据集**

源域 GTA5 合成数据，目标域 Cityscapes（晴天）与 ACDC（夜间、雨雪雾等恶劣天气）数据集。

**📈 对比分析**

与单一 ControlNet 基线和分布度量（CMMD）对比；在 DISTS/DreamSim 上取得了稳健的 Pareto 前沿，生成的数据在 CMMD 上表现优于 ControlNet，但在语义分割的 mIoU 上仍落后，说明成对度量与下游任务性能未完全对应。

**⚠️ 局限性**

成对度量无法准确预测语义分割性能；缺少直接估计下游效果的指标，导致进化得到的流水线在实际分割任务中的表现不如单独的 ControlNet；此外仅限于固定算子集合，未考虑算子超参数。

---

## 642. Quasi-multimodal-based pathophysiological feature learning for retinal disease diagnosis

**arXiv ID:** 2602.03622 | [PDF](https://arxiv.org/pdf/2602.03622v1)

**作者:** Lu Zhang `[一作]` (Tianjin University), Mengyu Jia `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于单眼底彩色照片（CFP）通过生成对数千种衍生模态（arterial/av-FFA、MSI、病变与视盘/杯分割的显著图），并在此基础上进行多模态特征融合，完成视网膜疾病的多标签分类和糖尿病视网膜病变分级。

**💡 创新点**

创新点包括：① 将多模态生成与诊断任务联合微调，直接把生成网络的特征与诊断目标对齐；② 设计跨病理注意模块（CPAM），实现模态特异性和模态依赖性的层级自适应校准，显著减少冗余信息；③ 通过生成血管冲击相（arterial-FFA）和多光谱图像等前所未有的合成模态，弥补传统CFP单一视角的局限。

**🔧 技术方法**

使用的技术主要有：多分辨率编码-解码器（pix2pix、DRCR‑Net、U‑Net）进行模态生成；对抗损失、循环一致性损失、fidelity正则化；特征微调模块（MFFM）与跨病理注意模块（CPAM）结合SE和多尺度交叉注意力；训练采用Adam、cosine学习率调度；评估指标包含F1、AUC、mAP、Accuracy、Kappa。

**📊 数据集**

主要数据集：MuReD（多标签疾病分类，2208例）、DDR（DR分级，12522例）、Tianjin Eye Hospital（814例FFA、85例MSI）以及公共数据集如REFUGE、ARIA、STARE、RFMiD等。

**📈 对比分析**

与SOTA方法（如InceptionV3、ResNet、DenseNet、M²CNN、MuR‑CAN、RetFound等）在MuReD上比较，F1提升至0.683（比对手提升≈4.1%），AUC 0.953；在DDR上Accuracy 0.842、Kappa 0.861，分别比第二名提升≈0.6%和≈2.3%；ROC曲线、混淆矩阵均显示更高的区分度与更低的误分。外部验证（EyePACS）也保持显著优势。

**⚠️ 局限性**

局限性：① 训练数据中某些疾病样本极少，导致类别不平衡与误报；② 血管冲击相图像受采集时机限制，生成质量仍受影响；③ MSI数据来源有限，难以推广至更多临床设备；④ 缺乏跨机构外部验证与真实临床部署评估；⑤ 生成模态与原始CFP的对齐仍需改进，可能导致部分病变被低估。

---

## 643. Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation

**arXiv ID:** 2602.03619 | [PDF](https://arxiv.org/pdf/2602.03619v1)

**作者:** Changze Lv `[一作]` (Fudan University), Jie Zhou `[通讯]` (Tencent Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于人类偏好学习的查询特定rubric生成器，用于深度研究报告生成的强化学习训练与评估。

**💡 创新点**

创新点在于：①通过人类偏好对报告进行成对标注，构建大规模偏好数据集；②利用GRPO和混合奖励（偏好一致性+LLM评估+格式奖励）训练rubric生成器；③提出多代理马尔可夫状态(MaMs)工作流，以解决长时程推理中的上下文依赖。

**🔧 技术方法**

技术主要包括：强化学习（GRPO）、LLM-as-a-Judge、Query-specific rubric生成、Multi-Agent Markov-State (MaMs) 工作流、分块处理、奖励分配与权重化。

**📊 数据集**

数据集：构造的约5,000条多步研究查询与其两份候选报告的人类成对偏好标注数据；DeepResearch Bench（100条中英文查询）用于评估生成系统。

**📈 对比分析**

与基准方法比较：在偏好建模中，RL+混合奖励的rubric生成器在偏好准确率和Cohen’s d上均优于人类定义通用rubric、LLM直接生成rubric、SFT以及单纯的偏好或LLM奖励；在DeepResearch Bench上，使用MaMs+RL训练的rubric生成器与ReAct相比在整体分数上提升约3-5分，逼近闭源模型性能。

**⚠️ 局限性**

局限性：①需要大量人类偏好标注，成本仍然高；②生成rubric的多样性受限于GRPO的模式寻优特性；③对特定LLM的依赖（如Qwen3-30B-A3B）导致可移植性受限；④在极长文本场景下，仍存在hallucination风险。

---

## 644. Explanations Leak: Membership Inference with Differential Privacy and Active Learning Defense

**arXiv ID:** 2602.03611 | [PDF](https://arxiv.org/pdf/2602.03611v1)

**作者:** Fatima Ezzeddine `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Omran Ayoub `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文研究了在机器学习即服务（MLaaS）系统中，反事实解释（CFs）如何扩展对成员推断攻击（MIAs）的攻击面，并提出结合差分隐私（DP）与主动学习（AL）的防御框架；

**💡 创新点**

创新点在于首次系统性量化CF在阴影模型基础MIA中的作用，提出将DP与AL协同使用以降低模型记忆和训练数据暴露，并全面评估隐私、性能与解释质量三者之间的权衡；

**🔧 技术方法**

使用的技术包括阴影模型基础成员推断攻击、差分隐私随机梯度下降（DP‑SGD）、主动学习的样本挑选（基于不确定性/熵），以及基于Nearest‑Instance Counterfactual Explanations（NICE）的CF生成；

**📊 数据集**

实验数据集为EEG（14,980条，14维，2类）和In‑Location（20,000条，529维，多类），分别用于评估模型预测、MIA效果和CF质量；

**📈 对比分析**

与传统无CF和仅AL或仅DP的基线相比，CF开启后MIA准确率和召回率显著提升；DP在低ε下能削弱MIA，但CF会显著削弱这一优势；在保持预测精度（≈90%）的同时，CF质量（接近度和稀疏度）基本不受DP影响；

**⚠️ 局限性**

局限性包括：实验仅涵盖两类数据集，未对高维或非结构化数据验证；CF生成方法采用单一NICE实现，其他CF方法可能表现不同；DP噪声对稀疏度的影响在高维数据中更显著，需进一步研究更鲁棒的解释生成策略。

---

## 645. Sleep or Transmit: Dual-Mode Energy-Efficient Design for NOMA-Enabled Backscatter Networks

**arXiv ID:** 2602.03607 | [PDF](https://arxiv.org/pdf/2602.03607v1)

**作者:** Hajar El Hassani `[一作]` (CY Cergy Paris University), Mikael Gidlund `[通讯]` (Mid Sweden University)

**通讯引用:** 8484 | [OpenAlex ID](https://openalex.org/A5004012289)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了基于NOMA的双向Backscatter网络的能量效率资源分配方案，联合优化RF源功率、睡眠/激活时间分配和反射系数；

**💡 创新点**

提出双模式（Harvest‑on‑Transmit / Harvest‑then‑Transmit）自适应能量收集与传输策略，首次将NOMA与能量收集耦合在非凸分式规划中实现闭式解；

**🔧 技术方法**

采用Dinkelbach分式规划与交替优化（AO）算法、功率域NOMA、SIC以及能量收集模型进行联合优化；

**📊 数据集**

使用Monte Carlo仿真，基于随机Rayleigh衰落与距离相关路径损耗的统计模型，没有使用公开数据集；

**📈 对比分析**

与固定功率、无睡眠以及传统OMA基线对比，低至中等功率区间能量效率提升可达127%，在高功率下仍优于OMA，整体比基线提升8%至68%；

**⚠️ 局限性**

局限于单区、理想能量收集与完美CSI，未考虑多小区、多链路干扰及能量存储约束，需进一步验证在复杂环境下的适用性。

---

## 646. TodyComm: Task-Oriented Dynamic Communication for Multi-Round LLM-based Multi-Agent System

**arXiv ID:** 2602.03688 | [PDF](https://arxiv.org/pdf/2602.03688v1)

**作者:** Wenzhe Fan `[一作]` (University of Illinois at Chicago), Xinhua Zhang `[通讯]` (University of Illinois at Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多轮LLM多智能体系统中的动态通信问题，提出了一种基于任务导向的动态通信框架TodyComm，能够在推理时根据代理行为实时构建合适的通信拓扑；

**💡 创新点**

创新点在于：①使用代理信用值（通过GRN学习）来驱动每轮通信与决策图的生成；②采用DAG优先级构造，保证通信可行且可约束；③在动态对抗环境下通过强化学习（REINFORCE）实现任务导向的通信优化；

**🔧 技术方法**

技术手段包括：大型语言模型（如GPT‑4）、多轮交互式生成、门控递归网络（GRN）进行信用估计、REINFORCE策略梯度、基于信用的DAG构造算法；

**📊 数据集**

实验使用五个公开基准：MMLU、ARC‑Challenge、GSM8K、OpenBookQA、MedQA，涵盖常识推理、数学推理和科学推理；

**📈 对比分析**

与随机图、完全图、G‑Designer、AgentPrune等多种基线相比，TodyComm在大多数攻击率下（≥50%）均实现更高的任务准确率，同时保持与AgentPrune相近的token消耗；

**⚠️ 局限性**

局限性在于：①在攻击率低于50%时性能与部分基线相当或略逊；②实验仅在统一LLM模型上验证，尚未评估在异构或更大规模代理场景下的表现。

---

## 647. Efficient Investment in Multi-Agent Models of Public Transportation

**arXiv ID:** 2602.03687 | [PDF](https://arxiv.org/pdf/2602.03687v1)

**作者:** Martin Bullinger `[一作]` (University of Bristol), Kassian Köck `[通讯]` (Technical University of Munich)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了两类公共交通投资模型：一是在线路图上为公交线路选定停靠点，二是在加权图中选定边进行折价，以期降低乘客行程时间。

**💡 创新点**

创新点在于：①首次证明在简单线图上最大化均衡福利（egalitarian welfare）为NP‑hard且不可逼近；②提出改进的Dijkstra算法可在单代理网络模型中多预算下求最优；③给出两代理网络模型多预算最优解的多项式算法；④通过集合覆盖归约证明多代理情况对均衡福利仍是NP‑complete，并给出逼近下限。

**🔧 技术方法**

主要技术包括：动态规划、改进的Dijkstra（考虑预算状态）、贪心分析、集合覆盖归约、NP‑hardness和逼近下限证明。

**📊 数据集**

论文仅使用理论构造的合成实例，无实际交通数据集。

**📈 对比分析**

方法的评价完全基于理论复杂度分析；对比显示：均衡福利在路径模型上不可近似，单/两代理网络模型可多项式求解，变量代理则呈现NP‑难。

**⚠️ 局限性**

局限性包括：①缺乏实验验证与实际数据评估；②仅针对无向图和单一折价比例α；③多代理网络的多项式算法仅限于两代理，无法推广至任意固定k；④对真实交通网络（如平面图）性能未知。

---

## 648. Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models

**arXiv ID:** 2602.03681 | [PDF](https://arxiv.org/pdf/2602.03681v1)

**作者:** Difan Deng `[一作]` (Institute of Artificial Intelligence), Marius Lindauer `[通讯]` (Institute of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于神经网络注意力搜索的Token级混合注意力模型（Neural Attention Search Linear），自动为每个输入Token决定使用线性注意力还是软max注意力，以提升长文本建模效果并降低计算复杂度。

**💡 创新点**

创新点在于：①将软max与线性注意力混合到同一层并通过搜索决定Token级别的注意力类型；②使用Gated DeltaNet（GDN）作为线性注意力变体；③在chunk内实现自适应权重归一化与加权融合，兼顾长短程信息。

**🔧 技术方法**

主要技术包括：线性注意力、softmax注意力、GDN、神经架构搜索（NAtS）、chunk‑wise 混合注意力、RMS 归一化、动态权重学习、Flash‑Attention 加速。

**📊 数据集**

实验使用 Fineweb‑Edu 进行预训练，并在多种公共数据集上评测：LAMBADA、PIQA、HellaSwag、WinoGrande、OpenbookQA、ARC、PG19、CodeParrot、NarrativeQA、RULER、LongBench 等。

**📈 对比分析**

与 GDN、Mamba2、Transformer、GDN‑Hybrid 等基线相比，NAtS‑L 在长文本推理和检索任务上取得更优或相当的准确率/困惑度，同时推理延迟比纯 Transformer 低 2–5 倍，且在 64K+ 长度下仍保持较好性能。

**⚠️ 局限性**

局限性包括：①仅搜索两种注意力类型，未覆盖所有线性注意力变体；②未对软max/线性注意力比例施加正则约束，可能导致资源利用不均；③在极长上下文或高并行度场景下的可扩展性尚待进一步验证。

---

## 649. A Probabilistic Model-Checking Framework for Cognitive Assessment and Training

**arXiv ID:** 2602.03643 | [PDF](https://arxiv.org/pdf/2602.03643v1)

**作者:** Elisabetta De Maria `[一作]` (Universite Cote dAzur), Christopher Leturc `[通讯]` (Universite Cote dAzur)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种基于概率模型检验的认知评估与训练框架，利用严肃游戏监测阿尔茨海默患者的游戏行为并动态调整训练难度。

**💡 创新点**

创新点在于将DTMC/PDFA与可观测的置信度函数结合，构建Doxastic Meta‑Automaton（DDFMA）来跟踪医师诊断信心并实时决定后续游戏难度或测试。

**🔧 技术方法**

使用离散时间马尔可夫链（DTMC）、概率确定有限自动机（PDFA）、模型检验（PCTL、LTL）以及PRISM等工具实现。

**📊 数据集**

主要使用十名临床医生基于Match Items游戏的行动概率问卷数据作为先验概率，并在单个游戏中模拟患者行为。

**📈 对比分析**

通过编写PCTL与LTL属性进行模型检验，验证模型满足可达性、行为约束等需求，但未给出具体性能数值或与传统评估方法的客观比较。

**⚠️ 局限性**

局限性包括仅针对单一游戏和单一难度级别、缺乏真实患者实验数据、模型验证主要为理论演示，需进一步扩展到多游戏、多难度及临床实证。

---

## 650. CTTVAE: Latent Space Structuring for Conditional Tabular Data Generation on Imbalanced Datasets

**arXiv ID:** 2602.03641 | [PDF](https://arxiv.org/pdf/2602.03641v1)

**作者:** Milosh Devic `[一作]` (Computer Research Institute of Montreal), David Garson `[通讯]` (Computer Research Institute of Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种Conditional Transformer-based Tabular Variational Autoencoder（CTTVAE+TBS），专门解决严重类别不平衡的合成表格数据生成问题。

**💡 创新点**

创新点在于引入类感知三元组margin损失以重构潜在空间结构，以及训练-采样（Training-by-Sampling）策略自适应增强少数类样本的曝光。

**🔧 技术方法**

采用Transformer Encoder结合MMD正则化的VAE框架，加入triplet margin loss、TBS采样、局部三角插值的条件生成技术。

**📊 数据集**

在六个真实世界数据集上验证：Churn Modeling（CH）、Adult（AD）、Default of Credit Card Clients（DE）、Credit Card Fraud Detection（CR）、Machine Predictive Maintenance（MA）和Vehicle Insurance Claims（VE）。

**📈 对比分析**

与SMOTE、CTGAN、TVAE、CopulaGAN、CTABGAN、TabDiff及TTVAE等方法对比，CTTVAE+TBS在少数类下游任务（MLE）上显著提升，同时保持竞争性的真实性、隐私和一致的Fidelity。

**⚠️ 局限性**

局限性包括triplet loss带来的计算开销、对超参数λ的敏感性以及潜在空间重构未必统一提升Fidelity，且在极大规模数据上的可扩展性仍需验证。

---

## 651. Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation

**arXiv ID:** 2602.03639 | [PDF](https://arxiv.org/pdf/2602.03639v1)

**作者:** Fabian Schramm `[一作]` (Inria and DI-ENS, PSL Research University), Justin Carpentier `[通讯]` (Inria and DI-ENS, PSL Research University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于二次模型逼近的方差减小型MPPI框架，将目标函数拆分为已知模型与残差，利用模型导向的采样提高采样效率。

**💡 创新点**

核心创新在于将Boltzmann更新拆解为模型导向的先验与残差校正，并通过二次模型实现闭式高斯指导先验，实现分层采样与低方差；该框架对模型来源不敏感，可使用解析导数、Gauss‑Newton、BFGS或随机平滑。

**🔧 技术方法**

使用了随机平滑（Randomized Smoothing）估计梯度与海森矩阵，构造二次模型；基于高斯先验的MPPI更新；自适应协方差与移动平均稳定采样分布。

**📊 数据集**

在三类数据集上验证：①经典静态优化基准（Rosenbrock、Ackley等）；②非线性欠驱动的摆杆上升控制任务；③接触丰富的单指球体操纵任务（基于Pinocchio、CasADi与MuJoCo物理仿真）。

**📈 对比分析**

与普通MPPI和CMA‑ES做对比。实验显示在低采样（≤100样本）下，模型指导MPPI保持更高的有效样本数（ESS）并收敛更快；在所有任务上，平均迭代次数明显低于基线，方差也大幅降低。

**⚠️ 局限性**

局限性包括：构建二次模型（尤其是随机平滑）会产生额外计算开销；模型近似在离全局最优或噪声水平不合适时可能导致陷入局部极小；需要手工调节平滑尺度与温度参数。

---

## 652. Efficient Sequential Neural Network with Spatial-Temporal Attention and Linear LSTM for Robust Lane Detection Using Multi-Frame Images

**arXiv ID:** 2602.03669 | [PDF](https://arxiv.org/pdf/2602.03669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 653. Ultra Fast PDE Solving via Physics Guided Few-step Diffusion

**arXiv ID:** 2602.03627 | [PDF](https://arxiv.org/pdf/2602.03627v1)

**作者:** Cindy Xiangrui Kong `[一作]` (Purdue University), Guang Lin `[通讯]` (hi-Lab, Xiaohongshu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Phys-Instruct，一种在训练时加入 PDE 约束的知识蒸馏框架，将多步扩散教师压缩为仅需 1–4 步的学生生成器，实现无测试时物理一致的高质量 PDE 场采样。

**💡 创新点**

创新点：1) 在蒸馏目标（IKL）中引入可微 PDE 误差约束，使学生在无外部纠正的情况下自洽物理；2) 采用可调步数的生成器，既保留教师的分布特性，又显著降低采样延迟；3) 在训练阶段使用辅助分数网络，提升学生分布与教师的匹配。

**🔧 技术方法**

技术：扩散模型、Heun 一阶/二阶采样器、IKL（积分 KL）蒸馏、可微 PDE 误差损失、基于梯度的自适应损失权重、辅助分数网络训练。

**📊 数据集**

数据集：五个 PDE benchmark——Darcy 流、Poisson 方程、无界 Navier‑Stokes、Burgers 方程、Helmholtz 方程，分辨率分别为 32×32×2、32×32×2、64×64×2、128×128×1、128×128×2。

**📈 对比分析**

对比方法：EDM、DiffusionPDE、CoCoGen、PIDM；评价指标包括 PDE 误差（RMSE）、采样步数、延迟、SWD/MMD。结果显示：在 1–4 步下 Phys‑Instruct 的 PDE 误差显著低于所有基线（最少 5 倍以上），同时保持分布一致性；在高步数（200–2000 步）基线下亦能取得竞争或更优性能，说明效率–质量折中优势明显。

**⚠️ 局限性**

局限性：1) 依赖预训练的高质量教师扩散模型；2) 适用于低维（2D/3D）固定分辨率问题，扩展到更高维或自适应分辨率仍待研究；3) 对 λ_phys、σ_init 等超参敏感，需经验调参；4) 只在训练阶段施加物理约束，可能导致部分模式收敛不足或无法捕获全局复杂解。

---

## 654. KTV: Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs

**arXiv ID:** 2602.03615 | [PDF](https://arxiv.org/pdf/2602.03615v1)

**作者:** Baiyang Song `[一作]` (Xiamen University), Jianyuan Guo `[通讯]` (City University of Hong Kong)

**通讯引用:** 9102 | [OpenAlex ID](https://openalex.org/A5016185815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段的训练自由视频理解框架 KTV，先进行问题无关的关键帧聚类再对每帧进行基于重要性与冗余度的关键视觉令牌选择，并将得到的紧凑视觉表征输入 LLM 进行视频问答。

**💡 创新点**

创新点在于①使用问题无关的 K-means 聚类获得多样且代表性强的关键帧，避免 CLIP 相似度导致的语义陷阱；②在每帧中同时评估令牌的重要性（对 [CLS] 的注意力）与冗余度（与同帧其它令牌的相似度），并以可调权重 α 合成最终分数，精细控制视觉令牌数量；③通过问答相关性动态分配令牌比例 β，实现更高效、更精准的视觉信息压缩。

**🔧 技术方法**

核心技术包括预训练视觉编码器（DINOv2、CLIP‑L）、K‑means 聚类、令牌重要性与冗余度评分、CLIP 文本-图像相似度评估、投影层与 LLaVA‑v1.6 LLM 的结合。

**📊 数据集**

在七个多选视频问答基准上进行评估，涵盖 NExT‑QA、EgoSchema、IntentQA、STAR、VideoMME、MVBench 与 MLVU‑Test，使用 7B 与 34B 版 LLaVA‑v1.6，未使用额外视频‑文本对齐或大规模视频数据。

**📈 对比分析**

与现有训练自由方法（IG‑VLM、SF‑LLaVA、DYTO）对比，KTV 在视觉令牌数量大幅下降（仅 504–1872 令牌，低于 576 令牌/图像）同时保持或提升准确率；在 60 min、10800 帧视频上仅 504 令牌即可获得 44.8 % 的 MLVU‑Test 准确率，且推理时间比 SF‑LLaVA‑7B 低 60 % 以上；在 34B 版本上更是超过部分训练基线，说明效率与性能兼得。

**⚠️ 局限性**

局限性包括①仍依赖强大的预训练 VLM，若预训练模型欠佳会影响效果；②关键帧数量固定（如 6‑帧聚类）可能忽略极短或极微小动作；③仅在视频问答任务上验证，对开放式生成或其他视频理解任务的适用性尚待进一步探索。

---

## 655. QuAIL: Quality-Aware Inertial Learning for Robust Training under Data Corruption

**arXiv ID:** 2602.03686 | [PDF](https://arxiv.org/pdf/2602.03686v1)

**作者:** Mattia Sabella `[一作]` (Politecnico di Milano), Cinzia Cappiello `[通讯]` (Politecnico di Milano)

**通讯引用:** 4516 | [OpenAlex ID](https://openalex.org/A5063621424)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 QuAIL，一种将列级数据质量先验嵌入模型的质量感知学习机制，使用可学习的门控层和质量加权的 proximal 正则化来控制低质量特征的更新，从而实现鲁棒训练；

**💡 创新点**

创新点在于：①直接把列级质量信息注入优化过程；②用质量加权的正则化为门控层引入“惯性”，让可靠特征自由适应，低质量特征受限；③无需显式数据清洗或实例级重加权，模型可在现有管线中轻松集成；

**🔧 技术方法**

技术核心包括：可学习的特征门控层；质量加权的 proximal 正则化与动态锚点更新；学习率调度（cosine、线性衰减）；Optuna 超参搜索；Bootstrap 评估；合成 CCAR 与 CNAR 噪声；以及基于 MLP 的标准神经网络架构；

**📊 数据集**

在 50 个 UCI 公开数据集上评估，涵盖分类与回归任务，样本量从 100 条到数万条不等；

**📈 对比分析**

与线性模型、标准 MLP 以及基于课程学习的 MLP 进行对比，评价指标为 F1（分类）和 R²（回归）。在 CCAR 与 CNAR 模式下，QuAIL 在分类任务上平均提升约 0.99%/1.90%，在回归任务上提升约 2.99%/1.30%；小数据集上提升更显著；

**⚠️ 局限性**

局限性包括：仅考虑列级质量，缺乏实例级质量标签；实验使用合成噪声，缺乏真实世界噪声验证；对极大特征维度或大规模数据的可扩展性未知；需要先验质量标注，若缺失会影响效果；未与更复杂的网络结构（如 GNN、变压器）结合。

---

## 656. MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction

**arXiv ID:** 2602.03668 | [PDF](https://arxiv.org/pdf/2602.03668v1)

**作者:** Jung Min Lee `[一作]` (Seoul National University), Jungwoo Lee `[通讯]` (HodooAI Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过时间同步的多视角视频和跨视角重建目标，学习离散的潜在动作表示，用作视觉‑语言‑动作（VLA）模型的预训练。

**💡 创新点**

创新点在于引入跨视角重建约束，使潜在动作在不同视角下保持一致，从而显著抑制视角噪声、提升潜在动作与真实机器人动作的互信息；并首次在多视角同步数据上训练潜在动作模型。

**🔧 技术方法**

采用 VQ‑VAE 结构提取潜在动作，使用 DINOv2 作为视觉编码器，交叉熵训练 VLM 进行 VLA 预训练；评估时用线性探针和三种 MI 估计（KSG、BA、MINE）。

**📊 数据集**

训练集包含 OpenX‑Embodiment (OXE) 的多视角机器人轨迹与 EgoExo4D 的多视角人类视频；Bridge V2 用于验证潜在动作质量；SIMPLER 与 LIBERO‑Long 用于下游操纵任务评估。

**📈 对比分析**

与 UniVLA、LAPA、Moto 等基线以及 OpenVLA、π₀ 等 VLA 模型对比，MVP‑LAM 在 Bridge V2 上 MI 与 NMSE 均优；在 SIMPLER 上平均成功率从 39.6% 提升至 60.4%；在 LIBERO‑Long 上成功率从 79.4% 提升至 90.8%，明显优于多数对照方法。

**⚠️ 局限性**

局限性：需要严格时间同步的多视角视频，收集成本较高；实验仅在仿真环境中验证，缺乏真实机器人实验；主要针对视角噪声，未系统处理其他外部噪声如背景运动。

---

## 657. Reference-Free EM Validation Flow for Detecting Triggered Hardware Trojans

**arXiv ID:** 2602.03666 | [PDF](https://arxiv.org/pdf/2602.03666v1)

**作者:** Mahsa Tahghigh `[一作]` (Howard University), Hassan Salmani `[通讯]` (Howard University)

**通讯引用:** 2118 | [OpenAlex ID](https://openalex.org/A5008087855)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于EM侧信道的无参考、无设计知识的后硅硬件木马检测流程。

**💡 创新点**

将连续小波变换、CNN特征提取、PCA降维与贝叶斯高斯混合模型结合，实现了无监督、可解释的异常检测。

**🔧 技术方法**

使用连续小波变换（CWT）生成时间频率示波图，VGG‑16 CNN提取特征，主成分分析（PCA）压缩特征，贝叶斯高斯混合模型（BGMM）进行聚类与置信度计算。

**📊 数据集**

在AES‑128加密核中植入四种触发型木马，对比无木马基线，收集约1000条EM波形作为实验数据集。

**📈 对比分析**

与基线（无木马）以及传统金标参考方法对比，采用ΔBIC、α_post、β_post、Mahalanobis距离等指标，实验显示在90% PCA阈值下，信息泄露木马的β_post≥0.39、ΔBIC>2700，DoS木马β_post≈0.21，均能高置信度识别，误报率低。

**⚠️ 局限性**

对极低激活率或极短激活时间的木马仍难以捕捉，且受EM采样环境噪声影响，需进一步提升鲁棒性与泛化到非加密设计的验证。

---

## 658. Tutorial on Reasoning for IR & IR for Reasoning

**arXiv ID:** 2602.03640 | [PDF](https://arxiv.org/pdf/2602.03640v1)

**作者:** Mohanna Hoveyda `[一作]` (Radboud University), Maarten de Rijke `[通讯]` (University of Amsterdam)

**通讯引用:** 28554 | [OpenAlex ID](https://openalex.org/A5031439294)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并介绍了一场关于信息检索（IR）中推理能力的半天教程，阐述推理的定义、挑战、方法体系，并构建了一个统一的分析框架以评估不同推理技术在IR中的适用性。

**💡 创新点**

创新点在于：①首次在IR语境下给出系统化的推理定义；②提出跨学科方法的统一评估轴（表示能力、推理机制与学习方式、计算可行性）；③将LLM推理策略、强化学习、神经符号方法、概率框架及几何/能量优化等多元技术映射到同一框架，帮助IR研究者快速定位可迁移技术。

**🔧 技术方法**

采用的技术包括：LLM推理时策略（Chain-of-Thought、Self‑Refine、IRCoT 等）、LLM 与强化学习结合、神经符号推理（使用第一阶逻辑解析器、外部定理证明器）、概率与贝叶斯推理、Box/Set‑Compositional/超平面嵌入、能量基优化等；并辅以相应的评估维度。

**📊 数据集**

讨论并引用的主要数据集：NevIR、ExcluIR、QUEST、BRIGHT、BrowseComp‑Plus 等，用于展示当前IR系统在多步推理、排除、集合组合、时间约束等复杂查询上的不足。

**📈 对比分析**

通过在统一框架的三条轴（表示能力、推理与学习机制、计算可行性）上绘制方法位置，阐释各技术的优势与缺陷，虽然未给出具体实验性能，但提供了方法间的相对对比视角，帮助研究者判断哪些技术更适合特定IR需求。

**⚠️ 局限性**

限制主要体现在：①教程内容覆盖广泛但不可能深入每一类技术；②跨领域方法仍然零散，实战迁移需更多实证；③缺乏统一的标准化评测，当前的对比更多是概念层面而非量化性能；④对大规模IR部署的细节仍需进一步研究。

---

## 659. BIRDTurk: Adaptation of the BIRD Text-to-SQL Dataset to Turkish

**arXiv ID:** 2602.03633 | [PDF](https://arxiv.org/pdf/2602.03633v1)

**作者:** Burak Aktaş `[一作]` (Roketsan Inc), Bilge Kaan Görür `[通讯]` (Roketsan Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 BIRDTurk——BIRD benchmark 的土耳其语翻译版，并在其上评估了推理式提示、Agentic 多阶段推理以及监督微调三种 Text-to-SQL 方法。

**💡 创新点**

创新点在于：① 通过受限翻译管线保持 SQL 语义与执行一致性；② 采用基于 CLT 的大规模翻译质量验证；③ 系统比较多种方法在土耳其语环境下的性能差异。

**🔧 技术方法**

技术主要包括：LLM 翻译（Gemini）、AST 基于结构的 SQL 本地化、CLT 统计检验、Agentic 推理管线 DIN-SQL、以及 Qwen2.5-Coder 系列的指令调优微调。

**📊 数据集**

使用的数据集为 BIRD 的训练/验证集经翻译得到的 BIRDTurk，涵盖 12,751 条问题、95 个数据库（总计 33.4 GB）。

**📈 对比分析**

对比方法：直接提示（In-Context Learning）与 Agentic 推理（DIN-SQL），以及多语言 mT5 和指令调优的 Qwen2.5-Coder 微调。实验显示，Agentic 推理在土耳其语上提升明显，微调在指令模型上可实现显著提升但整体仍低于直接推理，土耳其语相较英语表现普遍下降。

**⚠️ 局限性**

局限性包括：① 仅基于翻译，缺乏原生土耳其问句；② 依赖单一 LLM 生成的翻译可能带来风格偏差；③ CLT 验证只给出总体质量保证，无法排除个别语义错误；④ 对土耳其特有的省略、语用现象覆盖不足。

---

## 660. Complete Reduction for Derivatives in a Transcendental Liouvillian Extension

**arXiv ID:** 2602.03592 | [PDF](https://arxiv.org/pdf/2602.03592v1)

**作者:** Shaoshi Chen `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Ziming Li `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究构建了一个完整的降阶算法，用于在超越的Liouvillian扩展中处理导数，能够将任意函数分解为导数和余项的和。

**💡 创新点**

创新点在于提出了一种新的完整降阶方法，能够有效地处理超越Liouvillian扩展中的导数，并为符号积分提供了新的算法。

**🔧 技术方法**

使用了完整降阶算法和Risch算子，结合线性代数和计算机代数的技术。

**📊 数据集**

使用了超越Liouvillian扩展的相关数据集，具体包括多种超越函数和它们的导数。

**📈 对比分析**

与传统的Risch算法进行了比较，结果表明该方法在处理复杂的超越函数时具有更高的效率，尤其是在计算时间上显著优于现有的算法。

**⚠️ 局限性**

限制在于该方法对某些特殊类型的函数可能不适用，且在处理高阶导数时可能会遇到计算复杂度增加的问题。

---

## 661. SAGE-5GC: Security-Aware Guidelines for Evaluating Anomaly Detection in the 5G Core Network

**arXiv ID:** 2602.03596 | [PDF](https://arxiv.org/pdf/2602.03596v1)

**作者:** Cristian Manca `[一作]` (University of Cagliari), Battista Biggio `[通讯]` (University of Cagliari)

**通讯引用:** 8728 | [OpenAlex ID](https://openalex.org/A5008367647)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了针对5G核心网络的安全意识评估准则SAGE‑5GC，并通过在真实数据集上评估多种无监督异常检测模型，在静态攻击与可适应攻击两种场景下比较其性能。

**💡 创新点**

创新点在于：①首次系统性将协议级约束与攻击者可控特征结合，设计了随机搜索和遗传算法的无模型攻击方法；②构建了安全感知的训练与评估指南，强调环境无关特征与类不平衡的真实性；③通过对抗性鲁棒性评估揭示了传统高精度检测器在野外部署的脆弱性。

**🔧 技术方法**

使用的技术包括：无监督异常检测算法（统计、密度、几何、集成等）、特征预处理与缩放、随机搜索、遗传算法（Differential Evolution与Evolution Strategy）以及Python库scikit‑learn与Nevergrad。

**📊 数据集**

使用的数据集是公开的5G-Attacks数据集，该数据集包含多种基于PFCP的攻击（如PFCP Flood、Restoration‑TEID等）与大量正常流量，经过协议过滤、缺失值填充与缩放后构成实验数据。

**📈 对比分析**

通过与基线检测器对比，集成方法在无对抗环境下的F1和AUC接近1，但在随机与优化对抗攻击下，所有模型的逃逸率显著提升，尤其是传统模型在遗传算法攻击下逃逸率可达100%，表明对抗鲁棒性不足。

**⚠️ 局限性**

局限性包括：①对可控特征集合J的手工定义，缺乏自动化推断；②只关注PFCP协议，未扩展到其他核心协议；③生成的对抗样本未能完整重现可在真实网络中重放的完整流量。

---

## 662. High-Resolution Underwater Camouflaged Object Detection: GBU-UCOD Dataset and Topology-Aware and Frequency-Decoupled Networks

**arXiv ID:** 2602.03591 | [PDF](https://arxiv.org/pdf/2602.03591v1)

**作者:** Wenji Wu `[一作]` (Harbin Engineering University), Zitong Yu `[通讯]` (Great Bay University)

**通讯引用:** 4901 | [OpenAlex ID](https://openalex.org/A5062522283)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对深海水下伪装目标检测，提出了统一的DeepTopo‑Net框架；

**💡 创新点**

创新点在于引入基于Riemannian度量的水条件自适应感知模块（WCAP）来补偿非均匀光衰减，并结合拓扑感知的Abyssal‑Topology Refinement Module（ATRM）恢复细长结构的连通性，同时构建了高分辨率的GBU‑UCOD深海数据集；

**🔧 技术方法**

技术上使用了MAE‑ViT编码器、Riemannian几何驱动的采样变形、频域门控特征融合、方向性骨架过滤器等多任务学习与拓扑约束；

**📊 数据集**

实验数据集包括MAS3K、RMAS以及新建的GBU‑UCOD（覆盖0‑4000+米深度的2K分辨率图像）；

**📈 对比分析**

与10+SOTA模型比较，在GBU‑UCOD上实现了mIoU、结构度量等指标的领先，S_α提升约1%，总体保持最优；

**⚠️ 局限性**

局限性在于对极低对比度或极薄透明目标的检测仍易出现漏检，且模型仅依赖单模态光学信息，未融合多模态传感器。

---

## 663. mopri - An Analysis Framework for Unveiling Privacy Violations in Mobile Apps

**arXiv ID:** 2602.03671 | [PDF](https://arxiv.org/pdf/2602.03671v1)

**作者:** Cornell Ziepel `[一作]` (TU Dresden), Stefan Köpsell `[通讯]` (TU Dresden)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 mopri 框架，整合静态与动态隐私分析流程，实现工具自动安装、网络捕获、解密、增强和报告生成；

**💡 创新点**

提出模块化可插拔分析管道以及多种网络捕获/解密方法，兼容物理与模拟设备并支持用户自定义流程；

**🔧 技术方法**

使用 Node.js+Vue 前后端架构，结合 Exodus、tweasel、Frida、mitmproxy、WireGuard、PCAPdroid、tshark 等开源工具；

**📊 数据集**

以 Android APK/XAPK 样本为测试对象，未使用公开大规模数据集；

**📈 对比分析**

与 PlatformControl、MobSF 等现有工具对比，展示 mopri 在自动化程度、报告完整度及可视化上优于前者，但未提供系统化性能基准，主要受设备与网络捕获方式影响；

**⚠️ 局限性**

局限性包括仅支持 Android、缺乏 iOS 支持、未实现隐私政策比对、需用户手动交互设备、对自定义加密仍有限、缺乏全面评测与性能指标。

---

## 664. MM-SCALE: Grounded Multimodal Moral Reasoning via Scalar Judgment and Listwise Alignment

**arXiv ID:** 2602.03665 | [PDF](https://arxiv.org/pdf/2602.03665v1)

**作者:** Eunkyu Park `[一作]` (Seoul National University), Gunhee Kim `[通讯]` (Seoul National University)

**通讯引用:** 5974 | [OpenAlex ID](https://openalex.org/A5100664729)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了多模态道德量表(Multimodal Moral Scale)数据集，并通过列表式学习优化(VLMs)实现连续道德判断。

**💡 创新点**

创新点在于引入5分标度的连续道德评分和显式模态归因，采用列表式学习到排序（ListMLE）进行多模态道德对齐，突破传统二元标签的局限。

**🔧 技术方法**

使用了预训练VLM（LLaVA‑OneVision、Qwen2‑VL、Phi‑3 Vision、InstructBLIP）+ LoRA微调，ListMLE列表优化，MSE与模态预测的多任务训练。

**📊 数据集**

使用Commonsense NormBank场景转化并由Stable Diffusion/DALL‑E3生成的约32k图像-场景对，附有5分评分和模态标签。

**📈 对比分析**

与二元偏好优化（BPO）和二元分类（BCE）对比，列表优化在NDCG@5、MRR、AUC‑Safety上表现最佳，Unsafe Rate与BPO相当但更稳健，展示了更高的排名与安全校准。

**⚠️ 局限性**

局限包括样本主要来自美英地区的注释者，模态归因仅限三类，且生成图像可能存在风格偏差，导致跨文化及细粒度模态信息缺失。

---

## 665. RIPPLE: Lifecycle-aware Embedding of Service Function Chains in Multi-access Edge Computing

**arXiv ID:** 2602.03662 | [PDF](https://arxiv.org/pdf/2602.03662v1)

**作者:** Federico Giarrè `[一作]` (Hasso-Plattner Institute), Holger Karl `[通讯]` (Hasso-Plattner Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于生命周期感知的服务功能链嵌入与重构框架RIPPLE。

**💡 创新点**

将服务功能链嵌入与节点生命周期动态结合，利用用户移动预测提前准备 VNF，显著降低中断。

**🔧 技术方法**

使用 LSTM 进行移动预测、RF 分类器估计接入概率、启发式生命周期感知部署算法以及贪心链路嵌入。

**📊 数据集**

通过仿真实验使用 Gauss‑Markov 移动模型、树型和城市型（Potsdam）拓扑，未使用真实网络数据。

**📈 对比分析**

与理想即时嵌入（Ideal）和实时重构（Reactive）三种方法对比，RIPPLE 在大多数用户上接近 Ideal，显著降低服务中断和不成功数据包。

**⚠️ 局限性**

对预测误差和生命周期延迟敏感；在资源极限或预测不佳时仍会出现服务中断。

---

## 666. Sequential Group Composition: A Window into the Mechanics of Deep Learning

**arXiv ID:** 2602.03655 | [PDF](https://arxiv.org/pdf/2602.03655v1)

**作者:** Giovanni Luca Marchetti `[一作]` (KTH Royal Institute of Technology), Nina Miolane `[通讯]` (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出顺序群组合成任务，分析神经网络如何通过梯度学习捕获结构化运算，并研究不同深度模型在宽度和样本复杂度上的表现

**💡 创新点**

将群论与傅里叶分析与梯度学习动态结合，证明两层网络按不可约表示顺序学习，深度网络利用结合性将指数宽度降低至对数/常数级别

**🔧 技术方法**

采用群的不可约表示、傅里叶变换、Alternating Gradient Flow框架、Waring分解及梯度下降构造性证明等技术

**📊 数据集**

使用合成的有限群序列数据集，主要包括循环群C_p、二面体群D_3等的编码向量进行实验

**📈 对比分析**

与两层MLP、RNN、深层MLP及Transformer等模型对比，实验结果显示深度网络在隐藏宽度和样本复杂度上明显优于浅层模型，且理论与实验高度吻合

**⚠️ 局限性**

仅在可逆群的设定下分析，浅层网络仍需指数宽度；对半自动机、形式文法等更一般结构缺乏理论与实验验证，训练过程在宽度不足时可能不稳定

---

## 667. Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG

**arXiv ID:** 2602.03645 | [PDF](https://arxiv.org/pdf/2602.03645v1)

**作者:** Yicheng Zhang `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9246 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过强化学习（RL）对检索增强生成（RAG）系统中的密集检索器进行终端任务级微调，解决检索器与LLM目标不匹配的问题。

**💡 创新点**

创新点包括：1）将传统确定性top‑k检索改为概率性文档采样，使检索器可建模为随机策略；2）在多跳推理中将检索历史编码进状态，缓解状态别名问题；3）使用基于GRPO的RL框架，直接优化终端答案质量。

**🔧 技术方法**

主要技术为：马尔可夫决策过程（MDP）建模、Plackett‑Luce文档排序概率、历史感知状态表示、Group Relative Policy Optimization（GRPO）强化学习、候选池限制与文档编码器冻结等近似技巧。

**📊 数据集**

实验使用多跳HotpotQA和单跳Natural Questions（NQ）数据集，并在2018年维基百科检索语料上进行检索。

**📈 对比分析**

与冻结检索器和REPLUG监督微调基线对比，所有20项评估指标中18项均提升；在多跳RAG管道（ReAct Agent、Search‑R1）和检索器编码器（4B、0.6B）下均表现出显著的EM/F1提升；训练收敛快，单次实验耗时约3小时。

**⚠️ 局限性**

局限性包括：1）仅优化检索器，未进一步改进LLM或推理策略；2）RL奖励稀疏，导致训练波动；3）候选池限制与编码器冻结可能限制检索器的表现；4）在单跳或低容量检索器上的提升相对有限。

---

## 668. A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures

**arXiv ID:** 2602.03604 | [PDF](https://arxiv.org/pdf/2602.03604v1)

**作者:** Basile Terver `[一作]` (Meta FAIR), Amir Bar `[通讯]` (Meta FAIR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个可公开获取、模块化且易于使用的 EB-JEPA 库，用于自监督表示学习、视频预测和基于动作的世界模型规划，全部可在单 GPU 上数小时完成训练。

**💡 创新点**

创新点在于将 Joint-Embedding Predictive Architecture（JEPA）从理论转化为实际可复现的实现，提供三层次（图像、视频、动作规划）示例、统一的能量式训练框架以及多步骤滚动训练与逆动力学、时序相似性等正则化组合，显著降低了入门门槛并展示了正则化对模型不崩塌的重要性。

**🔧 技术方法**

技术细节包括：能量式正则化（VICReg、SIGReg），投影器设计，GRU/UNet 预测器，k 步多步滚动训练，MPPI 与 CEM 规划优化，逆动力学（IDM）与时序相似性正则化，统一的能量函数实现。

**📊 数据集**

使用的数据集和环境包括：CIFAR‑10（图像表示学习），Moving MNIST（视频预测），以及自定义的 Two Rooms 交互式环境（动作规划）。

**📈 对比分析**

与现有自监督方法相比，EB‑JEPA 在 CIFAR‑10 上通过线性探测获得约 90‑91% 的准确率；在 Moving MNIST 上多步滚动训练显著提升平均精度；在 Two Rooms 上实现 97% 的规划成功率，且 ablation 证明了所有正则化项的必要性。

**⚠️ 局限性**

局限性包括：仅在小规模任务上验证，缺乏对更复杂、真实世界环境的扩展；正则化组合的理论解释仍不完整；对超参数调优仍需手工搜索，且尚未与大规模分布式训练和预训练视觉骨干对接。

---

## 669. Refer-Agent: A Collaborative Multi-Agent System with Reasoning and Reflection for Referring Video Object Segmentation

**arXiv ID:** 2602.03595 | [PDF](https://arxiv.org/pdf/2602.03595v1)

**作者:** Haichao Jiang `[一作]` (Sun Yat-sen University), Jian-Fang Hu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1885 | [OpenAlex ID](https://openalex.org/A5102336058)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Refer-Agent，一种训练自由的多代理系统，用于通过文本查询进行视频目标分割

**💡 创新点**

创新点在于采用粗细分帧选择、动态焦点布局以及两阶段链式反思机制，有效缓解大模型的幻觉与误差积累

**🔧 技术方法**

核心技术包括多模态大语言模型（如Ovis2.5‑9B）与SAM2分割模型，配合Coarse‑to‑Fine帧挑选、Dynamic Focus Layout、Chain‑of‑Reflection（存在性与一致性两阶段）

**📊 数据集**

在五个公开基准上进行评估，分别为ReVOS、Ref‑YouTube‑VOS、MeViS、ReasonVOS 与 GroundMoRe

**📈 对比分析**

与SFT与零射击基线比较，Refer‑Agent 在所有数据集上均实现显著提升（平均J&F提升至约72.7，超过最优零射击模型5‑10%），且无需额外微调即可接入新模型

**⚠️ 局限性**

主要限制包括对大模型推理时长和算力需求较高，且多回合反思的迭代次数仍需手动设定，未来可进一步压缩计算成本

---

## 670. TIPS Over Tricks: Simple Prompts for Effective Zero-shot Anomaly Detection

**arXiv ID:** 2602.03594 | [PDF](https://arxiv.org/pdf/2602.03594v1)

**作者:** Alireza Salehi `[一作]`, Mohammad Sabokrou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于TIPS视觉语言模型的零样本异常检测框架，利用分离的固定检测提示和可学习定位提示实现图像级和像素级检测。

**💡 创新点**

创新点在于：①使用空间感知的TIPS骨干取代CLIP，②采用解耦提示技术弥补全局与局部特征的分布差距，③将局部证据注入全局分数提升检测性能。

**🔧 技术方法**

采用TIPS视觉语言模型、固定与可学习的文本提示、局部和全局对比损失、Focal与Dice损失、Softmax相似度评分以及局部证据融合。

**📊 数据集**

在14个工业与医学数据集（MVTec-AD、VisA、DTDSynthetic、SDD、BTAD、DAGM、MPDD、ISIC、CVC-ColonDB、CVC-ClinicDB、TN3K、BrainMRI、HeadCT、BR35H）上进行评估。

**📈 对比分析**

与三种基于CLIP的先行方法对比，图像级AUROC提升约1.1–3.9%，像素级AUROC提升约1.5–6.9%，在工业和医学数据集上均取得最优或相当的表现。

**⚠️ 局限性**

局限性在于仍依赖TIPS的预训练效果，且对异常阈值设定敏感，未来需进一步提升跨域鲁棒性与可解释性。

---

## 671. Beyond the Commit: Developer Perspectives on Productivity with AI Coding Assistants

**arXiv ID:** 2602.03593 | [PDF](https://arxiv.org/pdf/2602.03593v1)

**作者:** Valerie Chen `[一作]` (Carnegie Mellon University), Ameet Talwalkar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 177390 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过在BNY Mellon开展的2989份大规模调查和11次深度访谈，评估AI编码助手（主要为GitHub Copilot）对开发者生产力的多维影响，并提出六维度的评估框架。

**💡 创新点**

提出单一指标不足以衡量AI编码助手影响的论点，构建包含自给自足、认知负荷、部署效率、技术专长与代码所有权等六个维度的新框架，并首次强调长期维度的重要性。

**🔧 技术方法**

使用定量问卷（DX平台）、统计相关分析（如Pearson r）以及定性访谈的开放式编码与主题分析技术。

**📊 数据集**

内部数据集：BNY Mellon开发者的2989份调查回应和11份访谈记录，涵盖不同岗位、经验层级和业务部门。

**📈 对比分析**

通过比较满意度与预估节省时间两项指标的相关性（r=0.34）说明单一指标不足；访谈结果识别出六维度并映射到三大典型用例（新特性实现、代码改进、测试/文档生成），展示不同维度对工作阶段的差异性影响。

**⚠️ 局限性**

局限性：研究仅涵盖单一企业、主要使用GitHub Copilot，缺乏对其他AI助手的比较；定量数据以自报为主，可能存在主观偏差；长远效应虽然提出但缺乏客观量化验证。

---

## 672. When Should Agents Coordinate in Differentiable Sequential Decision Problems?

**arXiv ID:** 2602.03674 | [PDF](https://arxiv.org/pdf/2602.03674v1)

**作者:** Caleb Probine `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**通讯引用:** 9899 | [OpenAlex ID](https://openalex.org/A5068441112)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究在可微分运动规划问题中，如何决定何时通过协调（联合决策）与非协调（Nash均衡）来获得最佳团队表现，提出了通过二阶导数条件判定协调程度并基于此设计时间段内协调的决策算法。

**💡 创新点**

创新点在于将协调程度映射到目标函数的Hessian正定结构，提供了一种从二阶性质出发自动判定何时需要通信的框架，并给出动态协调的概率分布最优求解方法。

**🔧 技术方法**

使用了连续优化的第一、二阶必要/足够条件、Newton根搜索、混合补全问题求解器（如PATH）、以及凸二次规划/均方误差正则化构成的动态规划算法。

**📊 数据集**

未使用公开数据集，而是在自定义的一维机器人分离问题和多步单积分器动力学的仿真场景中进行实验，构造了不同时间步长（T=6、10）的轨迹数据。

**📈 对比分析**

通过与完全不协调（仅Nash均衡）和完全协调（全程联合优化）两种极端做对比，实验显示所提出的动态协调策略显著降低了高成本轨迹的概率，且在更长时间范围内还能保持较低的通信成本；具体数值可在论文图中查阅。

**⚠️ 局限性**

局限性包括：只能处理可微分且无不确定性的情形，求解过程中需大量随机初始化以覆盖所有解，且对大规模时间步长或高维决策空间的可扩展性尚未验证。

---

## 673. Referring Industrial Anomaly Segmentation

**arXiv ID:** 2602.03673 | [PDF](https://arxiv.org/pdf/2602.03673v1)

**作者:** Pengfei Yue `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**通讯引用:** 4200 | [OpenAlex ID](https://openalex.org/A5014628588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Referring Industrial Anomaly Segmentation (RIAS) 这一新范式，并基于此设计了 Dual Query Token with Mask Group Transformer (DQFormer) 模型。

**💡 创新点**

创新点在于：①利用语言描述直接生成精细异常掩码，消除阈值设定；②引入统一的“异常”“背景”两种查询令牌，实现单一模型处理多种异常；③在视觉‑语言编码阶段加入多级局部聚合（LMA）和语言门控融合；④在解码阶段采用 Mask Group Transformer 与 Gumbel‑softmax 组块实现高效视觉‑语言交互。

**🔧 技术方法**

使用技术包括 Swin Transformer 视觉编码、语言编码、局部增强视觉‑语言融合 (LVLE)、Language‑Gated Multi‑Level Aggregation (LMA)、Dual Query Token 与 Mask Group Transformer、Dice+Focal 损失、交叉熵、Gumbel‑softmax 等。

**📊 数据集**

使用新构建的 MVTec‑Ref 数据集，该数据集扩展自 MVTec‑AD，包含 2110 张图像-语言-掩码三元组，覆盖多种异常类型且异常面积主要为小尺寸。

**📈 对比分析**

通过与七种自然图像参考分割基线（包括 CGFormer、CLIP‑based 方法等）以及传统 IAD 方法进行对比，DQFormer 在 MVTec‑Ref 验证集上 mIoU 提升 3.08% 以上、gIoU 提升 3.42% 以上，且在单模型下实现了最优的异常检测与定位性能。

**⚠️ 局限性**

局限性包括：①数据集规模有限，可能导致模型在更大、更丰富的工业场景中泛化能力不足；②当前仅支持单目标或简单多目标表达，复杂多语义描述的鲁棒性待验证；③对异常种类的“开放集”泛化仍受限于训练时的语言提示与视觉特征匹配。

---

## 674. Equilibrium Propagation for Non-Conservative Systems

**arXiv ID:** 2602.03670 | [PDF](https://arxiv.org/pdf/2602.03670v1)

**作者:** Antonino Emanuele Scurria `[一作]` (Universite Libre de Bruxelles), Serge Massar `[通讯]` (Universite Libre de Bruxelles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了两种可扩展到非守恒动力学的平衡传播算法（AEP与Dyadic EP），实现了在非对称系统中的梯度计算与学习；

**💡 创新点**

创新点在于：①通过在学习阶段加入基于反对称雅可比的局部校正力，AEP恢复精确梯度；②通过状态空间加倍构造能量势，Dyadic EP以变分方式映射非守恒动力学到守恒形式；

**🔧 技术方法**

使用了基于能量函数的动力学模型、局部梯度对比学习、雅可比分解、状态空间加倍与变分推导、以及与传统反向传播的对比分析；

**📊 数据集**

在MNIST手写数字识别数据集上进行实验；

**📈 对比分析**

与标准EP、Vector Field (VF) 以及极限学习机进行对比，实验显示AEP在对称、非对称和纯前馈网络中均取得更高准确率且收敛更快，Dyadic EP在计算成本上与AEP相当；

**⚠️ 局限性**

局限性包括：①实现需要在硬件上支持反向耦合或状态空间加倍；②对时间局部性仍有依赖；③在深层网络中未充分验证；④AEP的局部校正力在实际神经元硬件中的实现机制尚待进一步研究。

---

## 675. Can Developers rely on LLMs for Secure IaC Development?

**arXiv ID:** 2602.03648 | [PDF](https://arxiv.org/pdf/2602.03648v1)

**作者:** Ehsan Firouzi `[一作]` (Technische Universitat Clausthal), Mohammad Ghafari `[通讯]` (Technische Universitat Clausthal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了 GPT‑4o 与 Gemini 2.0 Flash 在识别与生成安全 IaC 代码方面的能力，重点考察通用提示与引导提示对安全 smell 检测和代码生成的影响。

**💡 创新点**

创新点在于首次大规模对比两款顶尖 LLM 在 IaC 安全 smell 检测与安全代码生成任务中的表现，并提出可提升检测精度的引导提示方案。

**🔧 技术方法**

使用的技术包括自然语言提示工程（通用提示与多步引导提示）、手工标注、精确率/召回率/F1 分数评估、代码相似度分析（手工与 TF‑IDF‑余弦相似度）以及基于 89 个人工合成场景的安全代码生成测试。

**📊 数据集**

数据集涵盖了 2,569 条带安全 smell 的 Stack Overflow Ansible/Puppet 代码片段、21,757 条 GitHub IaC 文件（其中 430 条用于手工验证）以及 89 条人工合成安全场景。

**📈 对比分析**

通过对比通用提示与引导提示，发现引导提示将 GPT‑4o 在 GitHub 数据集上的 F1 分数从约 53% 提升至 88%，Gemini 从 66% 提升至 74%；在代码生成任务中，基线安全输出率仅 7%，即使显式请求“secure”代码，最高也仅提升至约 19%。

**⚠️ 局限性**

主要局限在于仅评估了 Ansible 与 Puppet 两种 IaC 工具、仅使用 GPT‑4o 与 Gemini 2.0 Flash 两个模型、存在潜在数据泄露风险、提示设计对结果高度敏感，以及缺乏对更广泛真实项目的纵向验证。

---

## 676. Can LLMs Do Rocket Science? Exploring the Limits of Complex Reasoning with GTOC 12

**arXiv ID:** 2602.03630 | [PDF](https://arxiv.org/pdf/2602.03630v1)

**作者:** Iñaki del Campo `[一作]` (Universidad Politécnica de Madrid), Jack Yarndley `[通讯]` (University of Auckland)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5114730395)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将GTOC12小行星采矿任务转化为MLE‑Bench评测环境，利用AIDE架构让LLM生成并迭代优化轨迹方案，并使用LLM评判者对策略的可行性进行分级。

**💡 创新点**

首次在高维物理约束下引入“LLM‑as‑Judge”评估机制，并将航天轨迹优化问题映射到机器学习工程评测框架，从而揭示LLM在策略与实现层面的能力鸿沟。

**🔧 技术方法**

技术栈包括OpenAI MLE‑Bench、AIDE搜索式代理、低推力轨道动力学与重力助推模型、Python科学计算库、LLM推理模型（GPT‑4‑Turbo、Gemini 2.5 Pro、O3、Cloud‑Sonnet‑4.5等）以及LLM评判器。

**📊 数据集**

使用GTOC12任务描述、60000个候选小行星目标列表、官方轨道验证器与自定义评估标准作为数据集。

**📈 对比分析**

通过对每个模型生成100份初始方案并由Gemini 2.5 Pro进行26分制评估，比较不同LLM的平均可行性分，结果显示从GPT‑4‑Turbo的9.3分提升至Cloud‑Sonnet‑4.5的21.15分，平均得分近翻倍。

**⚠️ 局限性**

主要局限在实现层面：单位不统一、语法/格式错误导致验证器崩溃；缺乏系统化调试与观测机制，导致LLM难以自主纠错，无法生成可验证的轨迹方案。

---

## 677. Self-supervised Physics-Informed Manipulation of Deformable Linear Objects with Non-negligible Dynamics

**arXiv ID:** 2602.03623 | [PDF](https://arxiv.org/pdf/2602.03623v1)

**作者:** Youyuan Long `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理模型的自监督学习框架SPiD，用于快速、鲁棒地控制弹性线性物体（如绳索）的动态变形。

**💡 创新点**

创新点包括：① 将改进的带阻尼质量–弹簧模型与可微物理模型耦合；② 在训练中使用能量最小化的自监督损失，并通过自监督DAgger自动检测和纠正分布漂移；③ 采用域随机化和数据增强提升泛化；④ 结合低成本、无标记摄像头的实时绳索感知；⑤ 仅用少量数据即可实现无专家演示的控制。

**🔧 技术方法**

技术手段主要有：可微差分物理建模、梯度基于的神经控制器训练、self‑supervised DAgger、域随机化噪声注入、YOLOv11‑seg细化分割、基于点云的绳索分割与跟踪、MuJoCo 仿真、OptiTrack 运动捕捉。

**📊 数据集**

使用的数据集：1）在仿真中采集 5000 条状态-动作样本；2）在实测中采集 2000 条样本；3）标记绳索的 RGB 数据集 12,400 张图像（12,400 训练 + 2,200 验证/测试），并在此基础上 fine‑tune YOLOv11‑seg；4）在实测中使用 9 个反射标记与 OptiTrack 系统记录绳索位置。

**📈 对比分析**

与基准方法（双摆模型控制器、PID 控制器）对比，SPiD 在绳索稳定化任务中：能量消散更快、误差更小；在多种绳索类型、不同初始姿态、外部扰动下保持最优；在标记无感知的单摄像头环境下仍能实现与标记感知相当的性能；在轨迹跟踪任务中，平均跟踪误差显著低于基准，且泛化性更好。

**⚠️ 局限性**

局限性：仅针对一维拓扑（绳索）进行验证，尚未在二维/三维柔性物体（如布料、海绵）上验证；模型与控制器需要手动设计能量/距离损失，适用范围有限；标记无感知系统的深度信息噪声仍影响精度。

---

## 678. Quantization-Aware Regularizers for Deep Neural Networks Compression

**arXiv ID:** 2602.03614 | [PDF](https://arxiv.org/pdf/2602.03614v1)

**作者:** Dario Malchiodi `[一作]` (Università degli Studi di Milano), Marco Frasca `[通讯]` (Università degli Studi di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在训练阶段加入量化感知正则化，使网络权重自然聚集于若干可学习的基准值，并将量化代表作为可学习参数直接参与反向传播；

**💡 创新点**

首次将量化参数视为可学习变量，并提出静态（正弦/余弦周期）与动态（可学习基底最小二乘/指数）四种正则化方案，实现对量化友好分布的直接引导；

**🔧 技术方法**

采用正弦/余弦周期正则化、可学习基底最小二乘/指数正则化、K‑means聚类以及权重共享等技术；

**📊 数据集**

在CIFAR‑10数据集上分别使用AlexNet与VGG16两种网络进行实验；

**📈 对比分析**

与无正则化基线比较，预调优准确率提升约3–6倍，后调优在K=8时仍可取得提升，动态正则化相对静态正则化表现更佳；

**⚠️ 局限性**

该方法在权重分布接近正态时最有效，K值增大时效果衰减，且尚未与其他压缩技术（如剪枝、低秩分解等）结合验证其普适性。

---

## 679. Controlling Output Rankings in Generative Engines for LLM-based Search

**arXiv ID:** 2602.03608 | [PDF](https://arxiv.org/pdf/2602.03608v1)

**作者:** Haibo Jin `[一作]` (School of Information Sciences), Haohan Wang `[通讯]` (School of Information Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何通过优化检索结果文本来控制LLM生成式搜索的输出排序，提出CORE方法。

**💡 创新点**

创新点是把输出排序问题建模为黑盒优化任务，提出shadow-model和query-based两种方案，并设计字符串、推理、评测三类优化内容。

**🔧 技术方法**

使用梯度优化、近似shadow模型、链式推理、自然语言生成与评价指标。

**📊 数据集**

数据集是自建的ProductBench，包含15类、每类200件商品的Amazon前10条推荐。

**📈 对比分析**

与现有排名操纵方法（STS、TAP、SRP、RAF）对比，CORE在四个搜索启用LLM上平均Top‑5/Top‑3/Top‑1推广成功率分别为91.4%/86.6%/80.3%，显著优于基线。

**⚠️ 局限性**

局限性在于对LLM的黑盒依赖仍需多轮查询，且优化内容可能被基于困惑度等过滤方法检测，且仅针对产品搜索未验证跨域通用性。

---

## 680. Human-in-the-Loop Failure Recovery with Adaptive Task Allocation

**arXiv ID:** 2602.03603 | [PDF](https://arxiv.org/pdf/2602.03603v1)

**作者:** Lorena Maria Genua `[一作]` (Worcester Polytechnic Institute), Zhi Li `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 1486 | [OpenAlex ID](https://openalex.org/A5100785539)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了适用于实时机器人故障恢复的自适应故障分配框架（ARFA），能够根据操作员的能力、历史表现和工作负载动态分配故障；

**💡 创新点**

通过在线更新人类能力信念并结合奖励函数、紧急度与工作负载，首次实现了同时考虑远程和本地操作员的实时故障分配；

**🔧 技术方法**

使用基于贝叶斯能力模型的信念区间、奖励函数、Adam优化器进行在线更新，以及仿真和用户研究中的实验数据收集与分析技术；

**📊 数据集**

使用自生成的随机故障需求数据和基于IONA机器人用户实验收集的失败解决时间与成功率数据；

**📈 对比分析**

与随机分配baseline进行对比，在仿真中远程操作员成功率提升至82%（vs 55%），团队成功率提升至95%（vs 78%）；在用户研究中失败处理时间、任务完成时间和团队成功率均显著提升，工作负载差异也显著减小；

**⚠️ 局限性**

能力模型过于简化，忽略疲劳、上下文等非线性因素；假设所有故障已预定义且可完整描述；噪声数据导致某些能力维度难以收敛；未评估大规模多操作员场景的泛化性。

---

## 681. Reasoning Cache: Continual Improvement Over Long Horizons via Short-Horizon RL

**arXiv ID:** 2602.03773 | [PDF](https://arxiv.org/pdf/2602.03773v1)

**作者:** Ian Wu `[一作]` (Carnegie Mellon University), Aviral Kumar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3943 | [OpenAlex ID](https://openalex.org/A5102493293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“Reasoning Cache”迭代解码算法，并通过强化学习训练模型在测试时能够在超出训练预算的长期推理中持续改进；

**💡 创新点**

创新点在于将推理与摘要循环结合，利用LLM在摘要-生成上表现出的优势，实现推理长度的可扩展性和分布式收敛；以及通过离线重放缓冲训练模型，使其在训练预算受限的情况下仍能实现长期推理的超越；

**🔧 技术方法**

使用的技术包括：基于奖励的策略梯度（GRPO）进行强化学习；离线重放缓冲（off‑policy RL）收集摘要；迭代解码框架（交替生成推理、摘要）；以及指令化提示控制推理与摘要的生成；

**📊 数据集**

使用的数据集包括：AceReason‑Math、HMMT 2025（Nov）、AIME 2025、IMO‑AnswerBench、FrontierScience（Olympiad）以及 Omni‑MATH 的困难子集；

**📈 对比分析**

与基线（原始 Qwen3‑4B‑Instruct‑2507、RL 训练、专用推理模型、Polaris‑4B、Self‑Refine/Verify 等）以及测试时 scaffold（RSA、DSM Agent）比较，实验显示该方法在 HMMT、IMO、FrontierScience 等四大基准上均实现了 10%–20% 的准确率提升，并能在 256k token 级别显著提升难题通过率；

**⚠️ 局限性**

主要局限包括：奖励仅基于单步正确率，缺乏全局（非贪心）策略；摘要生成未被单独优化，摘要质量仍可提升；当前方法仅适用于可验证的最终答案任务，对开放式推理（如证明生成）尚不适用；依赖于指令跟随能力较强的模型，若基础模型指令执行差则效果受限。

---

## 682. See-through: Single-image Layer Decomposition for Anime Characters

**arXiv ID:** 2602.03749 | [PDF](https://arxiv.org/pdf/2602.03749v1)

**作者:** Jian Lin `[一作]` (Saint Francis University), Xueting Liu `[通讯]` (University of Pennsylvania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将单张静态动漫插画自动分解为可操控的 2.5D 模型，生成完整语义 RGBA 层并推断绘制顺序。

**💡 创新点**

创新点包括：① 利用 Live2D 渲染引擎与弱监督构建高质量 2.5D 标签；② 采用两阶段扩散模型并引入 Body Part Consistency Module 实现跨层一致性；③ 在语义层内进行深度引导分层，以处理复杂的交错遮挡。

**🔧 技术方法**

技术手段：Latent 扩散模型（SDXL）+ 透明度解码器 + Body Part Consistency Module + Marigold 深度估计 + SAM‑HQ 多解码器 + Grad‑CAM++ + 自训练循环等。

**📊 数据集**

数据集：构建了 9,102 个完整标注的 Live2D 模型（训练 7,404 / 验证 851 / 测试 847），来自 ArtStation、Booth、DeviantArt，涵盖 19 类体部、遮挡区域与绘制顺序。

**📈 对比分析**

与 SAM+LaMa、SAM3、Qwen‑Image‑Layered 等基线比较，LPIPS 降至 0.1549，PSNR 提升至 18.3，SSIM 达 0.923，Fidelity 显著提升，层分解与深度推理效果更佳。

**⚠️ 局限性**

局限性：层间偶有轻微重叠或不一致；深度分层不够稳定；对极端姿态或复杂背景的鲁棒性有限，需人工微调。

---

## 683. SWE-Refactor: A Repository-Level Benchmark for Real-World LLM-Based Code Refactoring

**arXiv ID:** 2602.03712 | [PDF](https://arxiv.org/pdf/2602.03712v1)

**作者:** Yisen Xu `[一作]` (Concordia University), Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于真实 Java 项目开发者纯粹重构提交的 benchmark，用于评估 LLM 在方法级重构的能力。

**💡 创新点**

创新点在于自动化四步构建流程、严格筛选纯重构、覆盖单/多步重构类型、提供仓库级结构信息以及全面的功能验证。

**🔧 技术方法**

使用 AST‑based 重构检测工具、Eclipse JDT 语法分析、编译+单元测试、JaCoCo 覆盖、CodeBLEU 相似度、RAG 与多代理工作流等技术。

**📊 数据集**

采集自 18 个主流 Java 开源项目，包含 1,099 次重构（922 个原子、177 个复合）。

**📈 对比分析**

采用编译/测试成功率、AST 检测重构准确率、CodeBLEU 等指标，对 9 大 LLM（如 GPT‑4o‑mini、DeepSeek‑V3、Qwen‑Coder、CodeLLaMA 等）进行对比，结果显示通用模型表现最好，复合重构成功率仅 39% 左右。

**⚠️ 局限性**

仅覆盖 Java 语言、仅方法级重构、数据规模虽大但仍有限，且无法覆盖更高级别或跨文件复杂重构。

---

## 684. No Shortcuts to Culture: Indonesian Multi-hop Question Answering for Complex Cultural Understanding

**arXiv ID:** 2602.03709 | [PDF](https://arxiv.org/pdf/2602.03709v1)

**作者:** Vynska Amalia Permadi `[一作]` (Universitas Pembangunan Nasional Veteran Yogyakarta), Nikos Aletras `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ID-MoCQA 数据集和多步文化推理框架，构造了 15,590 条印尼文化多跳问答。

**💡 创新点**

创新点在于将单跳文化问题系统化转化为六类多跳推理链，并设计了结合专家与 LLM 判定的多阶段验证流程。

**🔧 技术方法**

采用大语言模型（Claude‑3.7‑Sonnet、GPT‑4o、GPT‑5 等）进行自动生成与判定，并通过人类审核与 LLM‑as‑judge 进行质量控制。

**📊 数据集**

基于 IndoCulture 单跳问答集，生成双语（印尼语/英语）多跳问题，最终形成 ID‑MoCQA。

**📈 对比分析**

在十个开源与前沿 LLM 上做零样本与 CoT 推理实验，前沿模型平均多跳准确率约 81%（高于人类 70%），但在比较与交集类型仍低于 50%。

**⚠️ 局限性**

局限包括模型偏好知名文化事实导致情境不符、比较/交集类型生成质量低、对低资源地区知识不足以及仍需进一步去偏与细化推理。

---

## 685. A Formal Analysis of Capacity Scaling Algorithms for Minimum-Cost Flows

**arXiv ID:** 2602.03701 | [PDF](https://arxiv.org/pdf/2602.03701v1)

**作者:** Mohammad Abdulaziz `[一作]` (Kings College London), Thomas Ammer `[通讯]` (Kings College London)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在Isabelle/HOL中对最小成本流问题进行形式化，验证了三种算法（最短路径、容量扩展和Orlin算法）的正确性，并实现了可执行代码。

**💡 创新点**

首次完整地形式化证明了Orlin算法的正确性，提出了对增广路径与残差图的颜色化与FBP去除技术，并通过ADT实现可执行代码，改进了多重图与残差网络的形式化。

**🔧 技术方法**

使用Isabelle/HOL的locale与ADT抽象、stepwise refinement、可执行代码生成、残差网络、容量缩放、Orlin算法的森林维护以及潜能函数分析等技术。

**📊 数据集**

论文未给出具体实验数据集，所有证明均在理论层面；代码实现可在GitHub（https://github.com/mabdula/Isabelle-Graph-Library）获取。

**📈 对比分析**

对三种算法的最坏情况运行时间进行了形式化证明；Orlin算法实现的强多项式时间为O(n log n (m+n log n))，容量扩展为O(n³ log B)，最短路径为指数级；理论上与已知结果保持一致。

**⚠️ 局限性**

局限性包括仅适用于无负环的保守权重图；Orlin算法仅适用于无限容量；证明基于多重图，未在实际数据集上验证性能；实验结果缺乏经验验证。

---

## 686. Data-Driven Graph Filters via Adaptive Spectral Shaping

**arXiv ID:** 2602.03698 | [PDF](https://arxiv.org/pdf/2602.03698v1)

**作者:** Dylan Sandfelder `[一作]` (University of Oxford), Xiaowen Dong `[通讯]` (University of Oxford)

**通讯引用:** 2710 | [OpenAlex ID](https://openalex.org/A5101579932)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种数据驱动的自适应谱形状框架，用可学习的基准谱核与少量高斯调制因子构造可解释的多峰多尺度图滤波器。

**💡 创新点**

创新点在于通过学习可解释的基准核并用可调高斯因子局部化能量，实现多峰谱响应且可迁移的滤波器；并提供TASS方法实现跨图的少样本迁移。

**🔧 技术方法**

采用多层感知机学习基准核，利用Chebyshev多项式展开实现无特征分解的快速滤波，训练时使用均方误差损失与光滑/形状正则化。

**📊 数据集**

使用合成图数据，包括ER、BA、WS、2D网格、SBM等多种拓扑，生成多峰谱响应的基准滤波器并通过Gaussian信号训练。

**📈 对比分析**

与固定原型波函数和学习线性组合基底的基线进行比较，实验表明在多峰谱重建任务中误差降低约5-15%，TASS在跨图迁移时实现正向迁移并在有限样本下收敛速度更快。

**⚠️ 局限性**

局限在于仅在合成数据上评估，未验证真实世界任务；模型以重建为目标，缺少对下游任务的优化；对大规模图的数值稳定性与参数选择仍需进一步研究。

---

## 687. Input-to-State Safe Backstepping: Robust Safety-Critical Control with Unmatched Uncertainties

**arXiv ID:** 2602.03691 | [PDF](https://arxiv.org/pdf/2602.03691v1)

**作者:** Max H. Cohen `[一作]` (North Carolina State University), Aaron D. Ames `[通讯]` (California Institute of Technology)

**通讯引用:** 14764 | [OpenAlex ID](https://openalex.org/A5039171820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对存在无法直接通过控制输入抵消的非匹配扰动的非线性系统，提出一种通过OD‑CBF扩展ISSf框架并结合Backstepping技术，构造出可实现鲁棒安全的控制器。

**💡 创新点**

创新点在于：①将Optimal Decay CBF引入ISSf，显著降低了验证条件；②提出了适用于严格反馈和双相对阶（Dual Relative Degree）系统的递归Backstepping合成方法；③通过控制输入与OD因子共同优化，实现了在非匹配扰动下的安全增益。

**🔧 技术方法**

主要技术手段包括：输入到状态安全（ISSf）理论、OD‑CBF（Optimal Decay Control Barrier Function）、高阶/递归Backstepping、SMT/QP控制器实现以及仿真验证。

**📊 数据集**

实验验证采用的案例为倒立摆和二维平面四旋翼，两者均无公开数据集，所有参数均来自典型物理模型，扰动采用正弦/恒定风速等仿真信号。

**📈 对比分析**

与传统CBF/稳健CBF方法对比，采用OD‑CBF+Backstepping后，安全集的扩张量显著减小，系统对非匹配扰动的容忍度提升；仿真结果表明在较小的ε值下即可保持安全，而传统方法需更严格的控制量或更大安全集。

**⚠️ 局限性**

局限性包括：①仅适用于严格反馈和双相对阶结构，其他系统结构需要进一步推广；②对系统矩阵的全秩假设要求较高；③未给出对比实验的量化指标，仅通过仿真展示安全性能；④实现复杂度较高，需要设计虚拟控制器与伪逆等。

---

## 688. Fast-MWEM: Private Data Release in Sublinear Time

**arXiv ID:** 2602.03732 | [PDF](https://arxiv.org/pdf/2602.03732v1)

**作者:** Themistoklis Haris `[一作]` (Boston University), Mutiraj Laksanawisit `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过改进MWEM的指数机制，提出Fast‑MWEM算法，实现每次迭代时间从Θ(m)下降到期望Θ(√m)。

**💡 创新点**

创新点在于将指数机制与懒惰Gumbel采样结合，并利用MIPS（k‑Nearest Neighbor）索引实现子线性采样。

**🔧 技术方法**

主要技术包括懒惰Gumbel采样、最大内积搜索（MIPS）与局部敏感哈希/IVF/HNSW等近似最近邻结构。

**📊 数据集**

实验使用合成数据：域大小3000的直方图、随机生成的线性查询，以及随机生成的线性规划实例。

**📈 对比分析**

与传统MWEM相比，Fast‑MWEM在相同精度下，错误几乎无差异，且在查询/约束数量增大时，运行时间呈O(√m)提升，尤其HNSW索引最快。

**⚠️ 局限性**

局限性包括：对数据维度仍保持指数级依赖；近似MIPS索引可能导致隐私/误差轻微下降；需调节阈值和参数，且实验仅在合成数据上验证。

---

## 689. BridgeV2W: Bridging Video Generation Models to Embodied World Models via Embodiment Masks

**arXiv ID:** 2602.03793 | [PDF](https://arxiv.org/pdf/2602.03793v1)

**作者:** Yixiang Chen `[一作]` (Institute of Automation), Liang Wang `[通讯]` (Institute of Automation)

**通讯引用:** 43063 | [OpenAlex ID](https://openalex.org/A5115602506)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将动作转化为像素对齐的 embodiment 掩码，并通过 ControlNet 路径注入到预训练视频生成模型中，形成统一的世界模型。

**💡 创新点**

创新点在于使用像素级掩码弥合坐标动作与像素视频之间的鸿沟，实现视角鲁棒、跨装置统一性，并加入流动基运动损失关注动态区域。

**🔧 技术方法**

采用预训练的 CogVideoX-5B-I2V 视频生成模型、ControlNet 结构、VAE、DiT、RAFT 光流、URDF 渲染掩码等技术。

**📊 数据集**

在 DROID（单臂 Franka）和 AgiBot-G1（双臂）机器人数据集上训练评估，并使用 Ego4D 等无标定视频进行额外训练。

**📈 对比分析**

与 IRASim、Cosmos、EVAC 等基线对比，BridgeV2W 在 FVD/LPIPS、Mask‑IoU 等指标上均表现更优，且在真实世界的策略评估与目标图像规划任务中取得较高成功率。

**⚠️ 局限性**

局限在于需要掩码或相机标定的支持，对高度旋转任务仍有挑战，且对不同机器人几何误差的鲁棒性尚未彻底验证。

---

## 690. QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization

**arXiv ID:** 2602.03782 | [PDF](https://arxiv.org/pdf/2602.03782v1)

**作者:** Yuhao Xu `[一作]` (Shanghai Jiao Tong University), Zhipeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6288 | [OpenAlex ID](https://openalex.org/A5100410140)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Vision‑Language‑Action（VLA）模型的低位量化问题，提出基于动作空间敏感度的通道级量化框架；

**💡 创新点**

①引入动作空间敏感度指标指导通道级位宽分配；②将量化与剪枝统一为 0‑bit 量化；③使用贪心降位算法在总预算内优化；④强调动作导向而非特征保真；

**🔧 技术方法**

后训练量化/量化感知训练、基于 Taylor 展开的一阶敏感度近似、通道级整数量化、贪心降位算法、激活统一位宽；

**📊 数据集**

LIBERO benchmark（四类任务），OpenVLA 与 OpenVLA‑OFT，π0 模型的 W8A16 量化，以及在 IMETA‑Y1 双臂机器人上采集的单臂/双臂任务数据；

**📈 对比分析**

与 SmoothQuant、OmniQuant、AWQ 等 LLM/MLLM 量化方法对比，OpenVLA‑OFT 量化后仅占 29.2% VRAM，保留 98.9% 性能并提升 1.49× 速度；在 LIBERO 上 INT4/INT8 的成功率几乎与 FP 一致，优于统一位宽和层级量化；在真实机器人任务上几乎无性能损失，速度提升 1.28×；

**⚠️ 局限性**

目前仅在 OpenVLA/OpenVLA‑OFT 等特定 VLA 架构验证，其他架构如 UniVLA、CALVIN 仍待评估；敏感度评估成本较高；激活统一位宽可能限制进一步压缩潜力。

---

## 691. UniGeM: Unifying Data Mixing and Selection via Geometric Exploration and Mining

**arXiv ID:** 2602.03772 | [PDF](https://arxiv.org/pdf/2602.03772v1)

**作者:** Changhao Wang `[一作]` (Politecnico di Torino), Jun Zhou `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniGeM 框架，将宏观分布平衡与微观实例筛选统一为流形近似问题，生成高质量的代码预训练数据集。

**💡 创新点**

创新点在于：①使用基于拓扑稳定性的自适应宏观分辨率选择；②在微观层面通过几何一致性门、结构惩罚和语义评分实现对结构化代码语义的精准采样；③无需代理模型或外部参考集，完全基于自监督几何特征。

**🔧 技术方法**

技术手段包括：几何嵌入、K‑means 层级聚类、稳定性度量（Kendall’s τ）、软对齐、结构惩罚（Mahalanobis 距离）、几何连贯性门、Wasserstein‑2 流形逼近分析。

**📊 数据集**

数据集：100B 令牌混合语料，包含 The Stack Dedup 与 Common Crawl 的代码与文本，保持 7:3 代码/文本比例；评测使用 HumanEval、MBPP、LiveCodeBench、CruxEval、MultiPL‑E 等代码基准。

**📈 对比分析**

与随机抽样、Meta‑rater、CLIMB 等现有方法对比，UniGeM 在 8B 与 16B MoE 模型上实现 2.0× 数据效率，单周期性能提升 6–7 分，整体得分 36.4（8B）/39.5（16B），在代码推理、多语言通用性与复杂执行任务上均优于对手。

**⚠️ 局限性**

局限性：①主要验证于代码语料，是否能推广到更为异质的通用文本未知；②全局嵌入与聚类阶段计算开销仍显著；③当前为离线预处理流程，缺乏在线动态更新机制。

---

## 692. FOVI: A biologically-inspired foveated interface for deep vision models

**arXiv ID:** 2602.03766 | [PDF](https://arxiv.org/pdf/2602.03766v1)

**作者:** Nicholas M. Blauch `[一作]` (Harvard University), Talia Konkle `[通讯]` (Harvard University)

**通讯引用:** 7075 | [OpenAlex ID](https://openalex.org/A5002461431)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于人眼视网膜和初级视觉皮层的可变分辨率感知接口FOVI，并将其集成进CNN和ViT模型；

**💡 创新点**

创新点在于：①使用数学模型实现局部各向同性的生物学可解释视差采样；②通过kNN卷积和核映射实现可变分辨率下的卷积共享；③利用LoRA在预训练ViT上高效适配视差输入；

**🔧 技术方法**

技术手段包括：kNN卷积、核映射、低秩适配（LoRA）、可变视差采样、卷积和Transformer的集成；

**📊 数据集**

主要使用ImageNet-1K（及其100类子集）作为训练与评估数据集；

**📈 对比分析**

与全分辨率Uniform baseline、Log-Polar采样、Weak-FOVI等低分辨率基线进行对比。结果显示：在64×64像素下，FOVI-ViT-H+可获得84% ImageNet Top‑1，仅用1/16像素且1/3 GFLOPs；FOVI-ViT-S+在64/128像素下分别取得约88%和92% baseline；整体性能与统一采样相近但计算量显著降低；

**⚠️ 局限性**

局限性包括：对高分辨率全视野场景的实验不足；缺乏主动视觉与扫视整合的动态策略；只在单张图像或固定随机视差上评估，未覆盖更复杂的时间序列任务。

---

## 693. RAWDet-7: A Multi-Scenario Benchmark for Object Detection and Description on Quantized RAW Images

**arXiv ID:** 2602.03760 | [PDF](https://arxiv.org/pdf/2602.03760v1)

**作者:** Mishal Fatima `[一作]` (University of Mannheim), Margret Keuper `[通讯]` (Max Planck Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在RAW图像上进行目标检测和对象描述的可行性，构建了一个统一、标注完善的大规模RAW数据集，并在不同量化位深下对其进行了基准评估。

**💡 创新点**

创新点包括：①将四个现有RAW数据集整合并用大模型重新标注，得到七类细粒度目标；②为每个对象提供高质量描述，方便评估RAW处理对语义信息的保留；③提出并实验了线性、对数与可学习γ缩放等量化方法，证明低比特RAW（4/6/8-bit）在目标检测和描述上可与8-bit sRGB媲美甚至超越；④首次在MM‑Grounding‑DINO等大型VLM上进行零样本量化RAW评估。

**🔧 技术方法**

使用的技术包括：Grounded‑DINO 1.5 进行重标注并人工校验；对RAW图像进行对数/γ缩放后量化；采用Faster‑RCNN、RetinaNet、PAA以及MM‑Grounding‑DINO等检测模型；利用直通估计训练可学习γ；Gemini‑2.5‑Pro 生成对象描述；BLEU、Regex、语义相似度等指标评估描述质量。

**📊 数据集**

数据集为自建RAW‑Dataset，包含约32,000张图像（训练25k，测试7.6k），来源于PASCAL RAW、RAOD、Zurich RAW、NOD；覆盖10/12/14/24位深、日夜、HDR等多种传感器与光照条件；标注7个类别（Car、Truck、Tram、Person、Bicycle、Motorcycle、Bus）。

**📈 对比分析**

在4/6/8-bit量化下对比sRGB，使用mAP、AP50/AP75等指标：1）线性量化性能显著下降；2）对数或对数+γ量化接近或超过sRGB；3）在MM‑Grounding‑DINO零样本实验中，对数+γ量化甚至超过sRGB；4）对象描述在log+γ量化下与高分辨率sRGB保持高度相似，线性量化表现差。

**⚠️ 局限性**

局限性：数据集在传感器种类与位深分布不均衡，主要集中在24‑bit；扩展更多RAW来源及重新标注成本高；大型VLM无法进行完整微调，需采用零样本或仅训练γ；低比特RAW处理仍受限于现有缩放方案，尚未探索更高级的自适应量化方法。

---

## 694. Improving Deep Learning Library Testing with Machine Learning

**arXiv ID:** 2602.03755 | [PDF](https://arxiv.org/pdf/2602.03755v1)

**作者:** Facundo Molina `[一作]` (Complutense University of Madrid), Marcelo d'Amorim `[通讯]` (North Carolina State University)

**通讯引用:** 1908 | [OpenAlex ID](https://openalex.org/A5020353093)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

使用机器学习分类器学习深度学习库API的输入约束，以提升输入有效性预测和测试效率

**💡 创新点**

将形状抽象和AutoML自动化模型搜索相结合，证明在大规模API上可获得>90%准确率的输入有效性判定模型，并将其作为ACETest的预过滤器显著提升通过率和效率

**🔧 技术方法**

Tensor形状编码、AutoGluon AutoML、梯度提升树（CatBoost、LightGBM、XGBoost）、ExtraTrees、FastAI神经网络等

**📊 数据集**

两大主流DL库PyTorch（98个API）和TensorFlow（85个API）的实际API调用，随机与组合式训练样本（10k/个API）以及额外5万样本用于泛化测试

**📈 对比分析**

与ACETest对比：在183个API上，加入ML预过滤后通过率从约29%提升至约61%；在无效输入过滤、生成速度方面表现显著改善，整体测试时间减少约50%，有效输入率提升至90%以上（对低通过率API尤为显著）

**⚠️ 局限性**

训练样本正负样本极不平衡导致某些复杂API无法生成足够正样本；预过滤可能误判有效输入导致漏测；仅依赖现有输入校验假设；未探讨生成模型（GAN/Transformer）替代随机生成

---

## 695. Test-Time Conditioning with Representation-Aligned Visual Features

**arXiv ID:** 2602.03753 | [PDF](https://arxiv.org/pdf/2602.03753v1)

**作者:** Nicolas Sereyjol-Garros `[一作]` (Valeo), Nermin Samet `[通讯]` (Valeo)

**通讯引用:** 167 | [OpenAlex ID](https://openalex.org/A5070720681)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在推理阶段利用自监督模型对齐的表示来进行视觉条件生成的方法，称为REPA-G。

**💡 创新点**

创新点在于：①通过在采样时对齐中间特征与预训练特征实现即时视觉条件控制；②不需要额外训练或参数，仅在推理时调优；③支持多尺度、全局与局部、甚至多概念混合的控制。

**🔧 技术方法**

技术上结合了流匹配（flow‑matching）/概率流ODE、SDE采样、潜在表示对齐（REPA）、自监督特征提取（如DINOv2）以及梯度引导的潜在空间对齐势（potential）。

**📊 数据集**

使用ImageNet（1.2M图像）和COCO（123K图像）进行实验，图像分辨率均为256×256。

**📈 对比分析**

与无条件流模型、传统SiT、以及文本生成模型CAD‑I等进行对比；在FID、sFID、IS、Precision/Recall、CLIP、Pick等指标上，REPA‑G在多尺度条件下实现了显著提升，尤其在多概念组合时能保持高质量与高概念一致性。

**⚠️ 局限性**

局限性包括：①需要手动调节参数λ以平衡多概念生成；②对场景视角变化的鲁棒性不足；③对极端细粒度局部控制的精度仍有限。

---

## 696. LIVE: Long-horizon Interactive Video World Modeling

**arXiv ID:** 2602.03747 | [PDF](https://arxiv.org/pdf/2602.03747v1)

**作者:** Junchao Huang `[一作]` (Chinese University of Hong Kong), Li Jiang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 21755 | [OpenAlex ID](https://openalex.org/A5100392387)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了LIVE模型，用循环一致性目标消除传统自回归视频生成中的误差累积问题；

**💡 创新点**

核心创新在于前向滚动后逆向生成恢复原始帧并通过循环一致性损失实现误差边界，同时将教师强制、扩散强制与LIVE统一在一个可调GT比例框架中，并引入渐进训练课程；

**🔧 技术方法**

技术方案包括视频扩散Transformer/UNet结构、逆向生成与噪声注入的随机时间步、循环一致性损失和基于比例调节的进阶训练策略；

**📊 数据集**

使用的主要数据集有RealEstate10K、UE Engine视频数据集和Minecraft（WorldMem＋MineDojo）互动游戏视频；

**📈 对比分析**

与教师强制、扩散强制、Geometry Forcing、DFoT等基线相比，LIVE在短至中长帧数（32~200帧）上保持更低的FID、较高的PSNR、SSIM与更低的LPIPS，尤其在长序列（>128帧）显著优于其他方法；

**⚠️ 局限性**

局限性包括仍未引入大规模双向教师模型、对极长序列的稳健性有待验证，以及对计算资源的依赖较高。

---

## 697. Occlusion-Free Conformal Lensing for Spatiotemporal Visualization in 3D Urban Analytics

**arXiv ID:** 2602.03743 | [PDF](https://arxiv.org/pdf/2602.03743v1)

**作者:** Roberta Mota `[一作]` (University of Calgary), Nivan Ferreira `[通讯]` (Universidade Federal de Pernambuco)

**通讯引用:** 1127 | [OpenAlex ID](https://openalex.org/A5065868869)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于共形映射与视角依赖X射线的沉浸式透镜可视化，用于在三维城市环境中同时展示时间变化与空间几何

**💡 创新点**

创新点在于将共形映射、形状自适应嵌入与X射线去遮挡相结合，解决时间-空间可视化的混乱和遮挡问题

**🔧 技术方法**

技术包括Schwarz–Christoffel共形映射、建筑足迹分解、视角依赖的X射线裁剪以及沉浸式交互手势

**📊 数据集**

使用Manhattan OSM建筑模型，属性为四季八小时累计日照时间

**📈 对比分析**

与传统交互式时间切换、X射线+时间和嵌入式+X射线三种基线比较，结果显示嵌入+X射线在完成时间、误差、头部运动上均优于其它方案

**⚠️ 局限性**

局限在于共形映射预处理、数值不稳定、仅测试单一属性和有限任务，缺乏动态重映射和更广泛的可扩展性

---

## 698. Edge-Optimized Vision-Language Models for Underground Infrastructure Assessment

**arXiv ID:** 2602.03742 | [PDF](https://arxiv.org/pdf/2602.03742v1)

**作者:** Johny J. Lopez `[一作]` (University of New Orleans), Mahdi Abdelguerfi `[通讯]` (University of New Orleans)

**通讯引用:** 770 | [OpenAlex ID](https://openalex.org/A5082583698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套双阶段端到端的地下基础设施缺陷检测与自然语言总结管道，先使用轻量级RAPID‑SCAN对图像进行语义分割，再用Phi‑3.5视觉语言模型生成结构化、可操作的中文报告。

**💡 创新点**

创新点在于（1）构建极小化的RAPID‑SCAN分割网络，仅0.64M参数即可获得0.834 F1；（2）通过QLoRA与4‑bit NF4量化实现Phi‑3.5在资源受限边缘设备上的高效域适配；（3）将两阶段模型无缝集成至移动机器人，并通过TensorRT、混合精度等硬件优化实现实时推理。

**🔧 技术方法**

使用的技术包括：RAPID‑SCAN轻量级分割网络、Phi‑3.5视觉语言模型、QLoRA低秩适配、4‑bit NF4与INT8量化、NVIDIA TensorRT推理引擎、ROS 1节点架构、Jetson AGX Orin边缘计算平台。

**📊 数据集**

使用自建的Sewer and Culvert Defect（SCD）数据集，包含5051张RGB图像、八类缺陷分割掩码及人工校验的结构化自然语言描述，作为VLM微调与评估的数据来源。

**📈 对比分析**

与U‑Net、Swin‑UNet等主流分割模型比较，RAPID‑SCAN在F1、mIoU、参数量、GFLOPS上实现近似性能且参数下降97%；VLM在ROUGE‑L、BLEU、BERTScore等指标上比基线提升约10%~20%；推理延迟在完整管道下平均3.1秒，符合机器人实时要求。

**⚠️ 局限性**

局限性包括：缺陷类别有限，仅覆盖管道常见缺陷；在极端光照或极其复杂结构时分割与总结的鲁棒性仍需提升；依赖人工标注的语料库规模有限；边缘设备的功耗与热管理仍是未来改进点。

---

## 699. CUBO: Self-Contained Retrieval-Augmented Generation on Consumer Laptops 10 GB Corpora, 16 GB RAM, Single-Device Deployment

**arXiv ID:** 2602.03731 | [PDF](https://arxiv.org/pdf/2602.03731v1)

**作者:** Paolo Astrino `[一作]` `[通讯]` (Independent Researcher), Paolo Astrino (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了CUBO，一套在16 GB内存、无云依赖的本地检索增强生成（RAG）系统，支持10 GB左右的文档库；

**💡 创新点**

核心创新在于把流式摄取、分层混合检索与硬件感知调度在单机上协同工作，实现O(1)缓冲、热/冷索引分层、量化感知路由，并在15.5 GB RAM下达到0.48–0.97的Recall@10；

**🔧 技术方法**

采用FAISS IVF+PQ（8位）与BM25混合检索，Reciprocal Rank Fusion（RRF）融合，gemma-embedding‑300m向量嵌入，4‑bit Llama 3.2 3B生成，mmap内存映射、延迟加载、语义缓存、Snowball词干化、多语言词表；

**📊 数据集**

在BEIR基准（SciFact、FiQA、ArguAna、NFCorpus）和自定义的UltraDomain合成数据上进行评测；

**📈 对比分析**

与LightRAG、GraphRAG、LlamaIndex、PrivateGPT、E5‑base、SPLADE等基线对比，CUBO在16 GB限制下实现约0.4的nDCG@10，查询延迟≈185 ms，峰值内存≈14.2 GB；

**⚠️ 局限性**

限制包括：只能处理≈12 GB左右的语料库，专业领域（医学、法律）召回偏低，缺乏增量/异步摄取，语言覆盖主要为英语，需自行完成法规合规与人类评估。

---

## 700. Training Multi-Turn Search Agent via Contrastive Dynamic Branch Sampling

**arXiv ID:** 2602.03719 | [PDF](https://arxiv.org/pdf/2602.03719v1)

**作者:** Yubao Zhao `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5114038310)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过分支采样对长周期代理学习进行训练。

**💡 创新点**

关键创新在于只针对轨迹尾部进行对比监督，结合难度感知分支和冗余步骤掩码。

**🔧 技术方法**

采用值无关的GRPO框架、对比优势估计、分支采样以及RSM技术。

**📊 数据集**

在七个多跳与单跳问答基准（HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle、Natural Questions、TriviaQA、PopQA）及GAIA网页搜索上评估。

**📈 对比分析**

与SFT、GRPO、GiGPO、Tree‑GRPO等基线对比，BranPO 在多跳任务中平均提升 5–10% 的 F1/LLM‑Judge 分数，单跳任务提升有限。

**⚠️ 局限性**

仍受限于最终奖励稀疏、模型可能产生幻觉信息，且对极难问题性能不足。

---

## 701. Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging

**arXiv ID:** 2602.03702 | [PDF](https://arxiv.org/pdf/2602.03702v1)

**作者:** Alexandru Meterez `[一作]` (Harvard University), Sham Kakade `[通讯]` (Kempner Institute at Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并验证在大型语言模型预训练中使用无终点依赖的学习率调度（常数LR+权重平均或1/√t+权重平均）是否能与传统的余弦衰减相媲美。

**💡 创新点**

提出了在高维线性回归与功率律谱条件下，1/t^γ（γ<1）学习率加尾部平均能够达到最优的极小化速率；并证明在适当的谱指数下，1/√t即为最优调度；进一步将这种理论推广到实际LLM预训练。

**🔧 技术方法**

使用常数学习率、1/√t学习率、余弦衰减、Warmup-Stable-Decay (WSD) 调度，并结合Stochastic Weight Averaging (SWA) 进行实验；理论分析基于线性回归的偏差-方差分解与谱指数分析。

**📊 数据集**

在150M与300M参数的基于OLMo代码库的Transformer模型上，以C4数据集（采用T5分词器）进行预训练；训练使用AdamW、批量大小分别为256/512，序列长度1024。

**📈 对比分析**

与为每个训练时长单独调优的余弦衰减基线比较，发现无终点调度在所有中间检查点（1×到32× Chinchilla计算量）几乎能追平余弦调度的最终损失，起始与收敛末端略有不足，但差距可忽略。

**⚠️ 局限性**

主要限制在于理论分析仅针对线性回归模型，未直接覆盖非凸LLM训练；WSD虽近似无终点，但仍需在90%处检查点决定衰减；α参数的调优虽然对性能影响小，但在严格意义上仍包含终点信息。

---

## 702. LLM-Inspired Pretrain-Then-Finetune for Small-Data, Large-Scale Optimization

**arXiv ID:** 2602.03690 | [PDF](https://arxiv.org/pdf/2602.03690v1)

**作者:** Zishi Zhang `[一作]` (Peking University), Yijie Peng `[通讯]` (Peking University)

**通讯引用:** 17095 | [OpenAlex ID](https://openalex.org/A5082863871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于Transformer的预训练‑再微调框架，用以解决在仅有极少观测（小数据）却需要同时处理海量决策实例（大规模）的问题。

**💡 创新点**

创新点包括：① 采用领域知识生成的合成数据进行监督预训练，提升模型对任务结构的先验捕捉；② 在缺乏真实标签的微调阶段利用Stein等式构造无标签的损失；③ 在微调中使用LoRA（低秩适配）仅更新少量参数，降低数据需求；④ 针对该框架给出完整的误差分解与非渐进性理论保证，阐明域差、泛化误差与逼近误差如何共同决定最终决策质量。

**🔧 技术方法**

技术细节：基于Transformer的编码器结构（embedding、self‑attention、output head）；对输入做CenterNorm与Radical Clipping以保证Lipschitz；微调时仅更新embedding与output head的LoRA层；Stein式无标签损失；理论上利用信息论、谱复杂度和Hellinger距离给出误差上界。

**📊 数据集**

数据集：① 预训练阶段使用大量基于业务经理经验、文献模型或生成模型产生的合成序列；② 真实评估阶段使用每个决策实例仅一条观测（如单次需求样本），以多产品新闻贩售问题为实验场景；③ 还在论文实验中探索了指数分布、Dirichlet过程、神经网络生成等多种目标分布以验证鲁棒性。

**📈 对比分析**

与无预训练的Transformer（从随机初始化直接微调）和单纯的基准方法（如SAA）对比，预训练‑再微调显著降低了决策的过度风险；当域知识与真实环境相近时，仅预训练即可获得接近最优的表现；当域知识误差较大时，微调能够弥补偏差，并且随着任务数量N增加误差快速下降，表现出规模经济效应。

**⚠️ 局限性**

局限性：① 需要先验的合成预训练数据，若领域知识不准确或难以获取，预训练效果有限；② 理论证明基于多项假设（如参数分布为独立同分布、正态噪声、子高斯尾部等），实际业务中可能不满足；③ 逼近误差仍不可消除，受Transformer容量限制；④ 实验主要在仿真数据上验证，缺乏真实大规模业务数据的实证；⑤ 微调仅针对embedding与output head，可能在某些任务中无法充分利用注意力模块。

---

## 703. Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation

**arXiv ID:** 2602.03689 | [PDF](https://arxiv.org/pdf/2602.03689v1)

**作者:** Jiashuo Sun `[一作]` (University of Illinois Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 122445 | [OpenAlex ID](https://openalex.org/A5019539533)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种边界感知的证据选择框架，将检索增量式生成中的重新排序器重新定义为能主动挑选既能挑战又能被生成器解答的证据集合，以提升在检索噪声环境下的鲁棒性

**💡 创新点**

创新点在于：①将证据选择定位为“Goldilocks Zone”问题，即挑选既不太容易也不难的证据；②利用生成器反馈的强化学习奖励构建选择策略；③采用两阶段交替训练（选择器+生成器），在训练时逼近真实检索噪声分布

**🔧 技术方法**

技术方法包括：使用RL（Group Relative Policy Optimization）对选择器进行训练；生成器使用相同的GRPO进行细调；构造多元奖励函数（边界奖励、相关性奖励、格式奖励、计数惩罚/引用奖励）以引导策略；在训练阶段进行过滤去除可预测或不可解的样本

**📊 数据集**

使用七个知识密集型问答基准：Natural Questions、TriviaQA、PopQA（单跳）；HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle（多跳）

**📈 对比分析**

与Direct Inference、Chain‑of‑Thought、RAG、RAG + Reranker、IRCoT、RAG SFT等基线对比，实验显示在所有数据集上平均提升约10.3 % EM，单跳任务提升8–15 %之间，多跳任务提升6–13 %之间，并在低‑k检索场景下保持更高鲁棒性

**⚠️ 局限性**

局限性包括：①RL训练成本高且对超参数敏感；②过滤步骤可能去掉有价值的样本；③方法仍依赖检索器的基本质量，若检索失败仍无法恢复；④目前仅在问答任务上验证，跨任务推广需进一步探索

---

## 704. Conflict-Resolving and Sharpness-Aware Minimization for Generalized Knowledge Editing with Multiple Updates

**arXiv ID:** 2602.03696 | [PDF](https://arxiv.org/pdf/2602.03696v1)

**作者:** Duy Nguyen `[一作]` (University of North Carolina Chapel Hill), Mohit Bansal `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于LoRA的知识编辑框架CoRSA，能够在多次更新中保持稳定性、提高泛化并解决知识冲突。

**💡 创新点**

创新点在于将Sharpness-Aware Minimization（SAM）与Direct Preference Optimization（DPO）结合，并在梯度冲突时采用PCGrad来保证更新方向的有效性。

**🔧 技术方法**

使用了LoRA参数微调、SAM、DPO、PCGrad等技术进行训练。

**📊 数据集**

在三大事实更新基准（CounterFact、ZsRE、MQuAKE）以及代码更新基准CodeUpdateArena上进行实验，并在不同规模的Qwen、Llama模型上验证。

**📈 对比分析**

与LoRA、MEMIT、F-Learning等基线对比，CoRSA在泛化率、更新有效性上平均提升约12%~15%，在持续更新场景中显著降低灾难性遗忘（约27%），在代码域的Pass@5提升5.5%。

**⚠️ 局限性**

局限性包括对输入数据质量高度依赖，可能仍会出现错误推断或被恶意利用的风险，并且在极大规模模型上训练成本仍较高。

---

## 705. WebSentinel: Detecting and Localizing Prompt Injection Attacks for Web Agents

**arXiv ID:** 2602.03792 | [PDF](https://arxiv.org/pdf/2602.03792v1)

**作者:** Xilong Wang `[一作]` (Duke University), Neil Gong `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出两步方法：先提取网页中可能被注入的片段，再在上下文中评估其是否为注入攻击，从而实现检测与定位。

**💡 创新点**

创新点在于将段落提取与上下文一致性检查相结合，并通过无目标/有目标剪枝及结构化对齐检查显著提升效果。

**🔧 技术方法**

技术主要包括使用 GPT‑4o 作为提取器和分析器 LLM，结合代码模式匹配、剪枝策略和对齐检查。

**📊 数据集**

使用了来自 EIA、Pop‑up、WASP、WebInject、VPI 五种攻击的网页以及对应的清洁网页，来源包括 Mind2Web、VisualWebArena、WebArena、Spam Email、SMS Spam Collection 等。

**📈 对比分析**

与文本、截图和段落基线相比，本文方法在所有攻击下实现最高 0.991 的准确率和 0.987 的平均 Jaccard 系数，分别高于最佳基线 0.12 与 0.127。

**⚠️ 局限性**

局限性包括对无明确指令的复杂注入仍有漏检，并且方法依赖 LLM 的推理能力，算力受限或模型变更时需重新调优。

---

## 706. Decision-oriented benchmarking to transform AI weather forecast access: Application to the Indian monsoon

**arXiv ID:** 2602.03767 | [PDF](https://arxiv.org/pdf/2602.03767v1)

**作者:** Rajat Masiwal `[一作]` (University of Chicago), Pedram Hassanzadeh `[通讯]` (University of Chicago)

**通讯引用:** 3033 | [OpenAlex ID](https://openalex.org/A5052673105)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文比较了多种物理与人工智能模型在印度次大陆中部季风区（CMZ）1-30 天预报中的季风开启日精度，使用 MAE、FAR、MR 等指标与传统分布式气候方法进行系统评测。

**💡 创新点**

创新点在于将最新 AI 模型（如 FuXi、GraphCast、GenCast 等）与传统数值模式融合，并在统一的评价指标和多时段数据集上对比其预测技能，首次展示 AI 在季风开启预测中的优势。

**🔧 技术方法**

使用的技术包括大气数值模式（IFS、NGCM）、深度学习气象模型（FuXi、GraphCast、GenCast）、基于阈值的季风开启判定方法、误差分析（MAE、FAR、MR）以及概率评估指标（AUC、BSS、RPSS）。

**📊 数据集**

数据集涵盖印度气象局（IMD）1° 与 0.25° 雨量格点数据、ERA5、IMERG 降雨数据，以及 2019-2024 年的观测期，扩展至 1965-1978、2004-2021 等历史期。

**📈 对比分析**

通过对比 MAE/FAR/MR、AUC、BSS、RPSS 等指标，研究发现 AI 模型在 1-15 天预报中普遍优于传统模式，误差可低于 2 天；但在 16-30 天周期和部分地区仍表现不佳，显示 AI 仍需改进。

**⚠️ 局限性**

局限性包括部分模型缺失部分年份、训练/微调年份影响公平比较、计算成本高导致样本量有限，以及对极端气候事件的泛化能力尚未完全验证。

---

## 707. Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States

**arXiv ID:** 2602.03708 | [PDF](https://arxiv.org/pdf/2602.03708v1)

**作者:** Ximing Dong `[一作]` (Centre for Software Excellence), Ahmed E. Hassan `[通讯]` (Queen's University)

**通讯引用:** 23747 | [OpenAlex ID](https://openalex.org/A5091586373)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种语义感知的推测式解码框架SemanticSpec，能在生成长链推理时一次性验证整个语义序列而非逐词验证，显著降低推理延迟；

**💡 创新点**

创新点在于：①从词级到语义级的解码迁移；②通过内部隐藏状态构建语义概率预测器，估计模型生成某一语义的概率；③实现了高效的序列级验证机制；

**🔧 技术方法**

采用内部隐藏状态提取、聚类（双向蕴含检测）、多层感知器回归模型预测语义概率，结合推测式解码算法；

**📊 数据集**

使用了四个推理基准：MATH‑500、AIME24、AMC23、GPQA‑D，并在这些数据集上训练预测器；

**📈 对比分析**

与Token‑level和Sequence‑level的基线（SpecReason、Speculative Thinking、SpecSampling）进行对比，实验显示在DeepSeek‑R1‑32B上平均加速2.7×、在QwQ‑32B上平均加速2.1×，同时保持或提升Pass@1和TPS；

**⚠️ 局限性**

仅在两对开源模型上验证，可能对其他模型或任务的泛化性有限，未来需要在更广泛模型和任务上进一步评估。

---

## 708. OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering

**arXiv ID:** 2602.03707 | [PDF](https://arxiv.org/pdf/2602.03707v1)

**作者:** Yifan Zhu `[一作]`, Haoran Luo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了OmniRAG-Agent，在低资源条件下通过图像‑音频检索与多轮工具调用实现长时段音视频问答。

**💡 创新点**

将检索增强生成（RAG）与代理式多轮推理相结合，并利用GRPO强化学习进行全局优化。

**🔧 技术方法**

采用CLIP与ASR索引的检索工具、多轮工具调用框架、GRPO强化学习以及OmniLLM。

**📊 数据集**

使用OmniVideoBench、WorldSense和Daily‑Omni三大长时段多模态问答基准。

**📈 对比分析**

与闭源和开源OmniLLM基线比较，在低资源设置下均实现多项指标提升，平均得分显著提高。

**⚠️ 局限性**

受检索质量限制、可能的多轮误差累积以及额外计算开销等因素限制。

---

## 709. Cognitively Diverse Multiple-Choice Question Generation: A Hybrid Multi-Agent Framework with Large Language Models

**arXiv ID:** 2602.03704 | [PDF](https://arxiv.org/pdf/2602.03704v1)

**作者:** Yu Tian `[一作]` (Arizona State University), Danielle McNamara `[通讯]` (Arizona State University)

**通讯引用:** 26914 | [OpenAlex ID](https://openalex.org/A5081661094)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了ReQUESTA，一种混合多代理框架，用来生成具备不同认知难度的多项选择题。

**💡 创新点**

创新点在于将LLM与规则引擎分离成专门的代理，利用规划、控制、评估和迭代改进的工作流，实现对题型认知需求的精准调控。

**🔧 技术方法**

技术包括：大型语言模型GPT‑5、基于规则的前处理器、规划代理、控制代理、文本、推理和主旨三种生成代理、评估代理、格式化与选项缩短模块，以及链式思考、少样本、人物化与自我批判式提示工程。

**📊 数据集**

使用了20篇约400词的OpenStax学术阐述文本，覆盖社会学、生命周期发展、历史与人类学四个学科。

**📈 对比分析**

通过在相同文本下生成200道题，比较ReQUESTA与单次GPT‑5零样本生成，结果显示ReQUESTA生成的题目在项目难度、区分度、点二值相关以及专家评估的主题相关性、干扰项语言一致性与语义合理性方面均显著优于基线，且模型性能提升在统计上显著。

**⚠️ 局限性**

局限性包括：仅测试学术阐述文本且篇幅有限，认知多样性仅划分为三类（文字、推理、主旨），未验证跨学科、长文本或更高级认知层级的适用性。

---

## 710. Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity

**arXiv ID:** 2602.03794 | [PDF](https://arxiv.org/pdf/2602.03794v1)

**作者:** Yingxuan Yang `[一作]` (Shanghai Jiao Tong University), Shangding Gu `[通讯]` (UC Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM多代理系统的规模效应与多样性影响，提出信息理论框架和无标签有效通道数指标K*；

**💡 创新点**

创新点在于将多代理性能与任务内在不确定性关联，并通过有效通道数与K*阐释多样性优于单纯规模扩展的根本原因；

**🔧 技术方法**

采用信息论方法、互信息计算、有效通道计数以及无标签嵌入特征（K*）等技术；

**📊 数据集**

使用七个推理/知识基准：GSM8K、ARC、Formal Logic、TruthfulQA、HellaSwag、Winogrande、Pro Medicine；

**📈 对比分析**

在相同计算预算下比较同质扩展与异质配置，结果显示仅两名多样化代理即可匹敌或超过16名同质代理，性能提升显著；

**⚠️ 局限性**

局限包括理论假设理想化、K*受嵌入模型影响、对知识任务的相关性弱、仅验证中小规模LLM，未覆盖更复杂工作流或更大模型。

---

## 711. Should I use Synthetic Data for That? An Analysis of the Suitability of Synthetic Data for Data Sharing and Augmentation

**arXiv ID:** 2602.03791 | [PDF](https://arxiv.org/pdf/2602.03791v1)

**作者:** Bogdan Kulynych `[一作]` (Lausanne University Hospital), Carmela Troncoso `[通讯]` (Max Planck Institute for Security and Privacy)

**通讯引用:** 3492 | [OpenAlex ID](https://openalex.org/A5072857797)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文对合成数据的三大主要应用场景（代理共享、训练集增强、统计估计增强）进行了形式化定义，结合理论分析与案例研究，阐明了其有效性与隐私保证的根本限制，并提出了基于用途和数据特征的决策流程图。

**💡 创新点**

创新点在于将合成数据应用场景细分为代理共享、训练增强与统计估计三类，针对每类提出了严谨的数学形式化和可验证性准则，并首次系统性地将信息恢复定律、差分隐私权衡与特定方法（如PPI、MST、AIM）整合为一套决策框架。

**🔧 技术方法**

采用理论推导（信息恢复定律、重构攻击理论、差分隐私理论）、形式化建模、案例分析和对比研究（如差分隐私、联邦学习、受限数据访问等）来评估合成数据的有效性与安全性。

**📊 数据集**

主要引用的案例数据集包括 2020 年美国人口普查的合成微观数据、欧洲健康数据空间（EHDS）中的健康记录、医学影像数据（用于扩增诊断模型）以及临床试验的受试者记录；论文并未自行采集新数据，而是以这些公开或行业案例为依据。

**📈 对比分析**

论文未提供传统意义上的实验对比与性能数值，而是通过理论界限与案例分析说明：在代理共享场景下隐私与统计有效性不可兼得；在训练增强场景下，仅当外部信息充分且有目标分布的验证集时，性能提升才可被验证；在统计估计场景中，普通统计方法无效，需采用专门的推断框架（如PPI）才能保证置信区间和假设检验的正确性。

**⚠️ 局限性**

主要局限包括：1）代理共享必须是窄目的，否则隐私与有效性无法同时满足；2）训练增强需外部补充信息并拥有目标分布的验证数据，否则无法评估收益；3）统计估计的合成增强在无专门推断方法时缺乏有效性保证；4）现有专门方法覆盖面有限，难以推广到更复杂的估计器或全合成场景。

---

## 712. Inference-time Unlearning Using Conformal Prediction

**arXiv ID:** 2602.03787 | [PDF](https://arxiv.org/pdf/2602.03787v1)

**作者:** Somnath Basu Roy Chowdhury `[一作]` (Google), Aranyak Mehta `[通讯]` (Google)

**通讯引用:** 4064 | [OpenAlex ID](https://openalex.org/A5075894805)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在推理时通过迭代改写与验证来实现语言模型的轻量级“可塑性去学习”框架；

**💡 创新点**

创新点在于将 conformal 预测与 LLM‑as‑a‑judge 结合，既保持了去学习的可行性，又提供了分布无关的置信度保证；

**🔧 技术方法**

采用了 LLM‑as‑a‑judge 评估器、迭代生成与修正、以及 conformal 预测来设定最大迭代次数；

**📊 数据集**

在 RWKU、Wikipedia Person Unlearn (WPU) 与 Weapons of Mass Destruction Proxy (WMDP) 三个公开数据集上进行评估；

**📈 对比分析**

与 best‑of‑N、梯度上升、负偏好优化等基线相比，所提方法在忘记集上提升高达 93% 的错误率下降，同时保持保留集性能；

**⚠️ 局限性**

局限性包括对噪声评估器的敏感性、潜在的迭代次数增长以及对 LLM 先验知识的依赖。

---

## 713. AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration

**arXiv ID:** 2602.03786 | [PDF](https://arxiv.org/pdf/2602.03786v1)

**作者:** Jianhao Ruan `[一作]` (DeepWisdom), Jiayi Zhang `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了 AOrchestra 框架，主导器通过统一四元组（Instruction、Context、Tools、Model）动态创建和调度子代理，以完成复杂的多步任务。

**💡 创新点**

创新点包括：①将子代理视为可按需生成的工具，而非固定角色；②统一的四元组抽象实现框架无关性和可插拔；③解耦主导器与执行，使其可通过监督微调或迭代提示学习提升。

**🔧 技术方法**

主要技术：大规模语言模型（Gemini‑3‑Flash、Qwen3‑8B 等）、工具调用与上下文控制、监督微调（SFT）、基于交互轨迹的提示优化（ICL）、成本‑性能 Pareto 前沿分析。

**📊 数据集**

实验数据集：GAIA、Terminal‑Bench 2.0、SWE‑Bench‑Verified；用于生成训练轨迹的 TaskCraft。

**📈 对比分析**

与 ReAct、OpenHands、mini‑SWE‑agent、Claude Code 等基线进行对比，使用相同模型或模型池。AOrchestra 在三大基准上平均提升 16.28% 的 pass@1；单模型 Gemini‑3‑Flash 下提升 13.94 绝对分；SFT、ICL 进一步提升约 11–15% 并降低平均成本。

**⚠️ 局限性**

局限性：仍依赖大模型算力；主导器的学习需要足够的轨迹数据；子代理实现细节对性能影响较大；成本调度需手动设定参数；未在更多领域或更大规模任务上验证。

---

## 714. From Pre- to Intra-operative MRI: Predicting Brain Shift in Temporal Lobe Resection for Epilepsy Surgery

**arXiv ID:** 2602.03785 | [PDF](https://arxiv.org/pdf/2602.03785v1)

**作者:** Jingjing Peng `[一作]` (King's College London), Alejandro Granados `[通讯]` (King's College London)

**通讯引用:** 66539 | [OpenAlex ID](https://openalex.org/A5082106258)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于U‑Net的神经网络（NeuralShift）预测癫痫手术中前期MRI到术中MRI的脑移位，并同时预测术中脑体积掩模；

**💡 创新点**

创新点在于：①只利用术前MRI及切除侧信息即可预测脑移位，省去术中成像；②采用变形场、掩模及其SDF三种监督信号的多任务学习，提升全局与局部几何一致性；

**🔧 技术方法**

技术：U‑Net多尺度架构、变形场回归、Dice与边缘损失、球坐标损失、F3D非刚性配准生成监督；

**📊 数据集**

数据集：98例癫痫患者的配对术前术中T1加权MRI，涵盖左右侧颞叶切除，采用9折交叉验证；

**📈 对比分析**

比较方法：与基于配准的Ground Truth（F3D）对比，指标为目标配准误差(TRE)和Dice系数。模型在切除侧及中线标志点上的平均TRE从约4.4mm下降至约2.8mm，Dice从0.92提升至0.97；

**⚠️ 局限性**

局限性：样本量有限，无法充分覆盖多样化临床情况；仅使用术前影像，未加入术中超声等补充信息；评估指标为几何对齐，缺乏临床可行性验证。

---

## 715. Context Compression via Explicit Information Transmission

**arXiv ID:** 2602.03784 | [PDF](https://arxiv.org/pdf/2602.03784v1)

**作者:** Jiangnan Ye `[一作]` (King’s College London), Yulan He `[通讯]` (King’s College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究一种软上下文压缩方法，将压缩过程视为在冻结LLM隐藏状态上显式信息传输，降低计算与内存成本；

**💡 创新点**

创新点在于将压缩任务拆分为深度方向的层级门控与宽度方向的全局最优传输规划，显式避免了逐层自注意的表示覆盖和无协调的容量分配；

**🔧 技术方法**

采用层级门控机制进行深度信息聚合、基于Sinkhorn算法的最优传输规划实现宽度方向的全局分配、轻量级MLP对齐压缩向量；

**📊 数据集**

在六个问答基准（SQuAD、NewsQA、TriviaQA、SearchQA、HotpotQA、NQ）以及MRQA的12个域内外数据集上进行评估；

**📈 对比分析**

与ICAE、500×、Beacon等软压缩基线以及无压缩提示微调/零样本对比，六大基准上均获得更高EM/F1，部分场景甚至超越无压缩提示微调；

**⚠️ 局限性**

仅在1B-3B规模模型、512长度、固定压缩比×4进行实验，未验证更大模型或不同压缩率，且在极端噪声/多跳任务中仍可能受限。

---

## 716. Zero-shot large vision-language model prompting for automated bone identification in paleoradiology x-ray archives

**arXiv ID:** 2602.03750 | [PDF](https://arxiv.org/pdf/2602.03750v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 717. A Scene Graph Backed Approach to Open Set Semantic Mapping

**arXiv ID:** 2602.03781 | [PDF](https://arxiv.org/pdf/2602.03781v1)

**作者:** Martin Günther `[一作]` (German Research Center for Artificial Intelligence), Martin Atzmueller `[通讯]` (Osnabrück University)

**通讯引用:** 3521 | [OpenAlex ID](https://openalex.org/A5011835245)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种以3D语义场景图（3DSSG）为核心后端的实时增量映射架构，将感知、语义抽象与知识推理无缝结合；同时引入可视化语言模型（VLM）特征和预测图神经网络实现语义关系的在线推断。

**💡 创新点**

创新点：① 将3DSSG直接作为整个映射过程的“单一真值源”，实现持续、一致的图更新；② 通过分层（帧层、片段层、对象层）设计，实现几何与语义的模块化；③ 引入局部与全局双层增量关系预测网络，在保持开集特性的同时实现语义关系的实时推断；④ 结合MaskCLIP与全局CLIP嵌入的门控融合，提升开集查询的语义鲁棒性。

**🔧 技术方法**

技术细节：SAM（FastSAM）分割；DINOv2视觉特征；MaskCLIP + 全局CLIP特征；基于GPU的DBSCAN+体素化点云处理；双阶段匹配（贪婪+主动细化）实现图融合；异构GraphSAGE网络进行增量谓词预测；Pose估计采用MICP-L与外部跟踪；可选的Rmagine/几何过滤实现 panoptic 去噪。

**📊 数据集**

使用的数据集与场景：ICL RGB‑D（小规模室内重建）、3RScan（含3DSSG谓词标注）、TIAGo 机器人真实环境（Ouster OS0‑128 + Femto Bolt RGB‑D）

**📈 对比分析**

方法对比与性能：在ICL上可视化显示彩色网格、实例分割与语义查询结果，证明语义标注准确但在拥挤区域存在过分割；在3RScan上展示局部/全局图及谓词预测融合；在TIAGo实机上完成大规模地图构建，成功集成大部分实例并通过滤波降低噪声。虽然未给出数值指标，但整体映射连贯、查询可解释，且实时性满足移动机器人需求。

**⚠️ 局限性**

局限性：① 过分割导致的噪声和边界模糊；② 语义关系预测依赖监督标签，缺乏对未标注环境的泛化；③ 对小物体与高度拥挤场景的分辨率受MaskCLIP补丁大小限制；④ 需要外部高质量位姿估计与同步；⑤ 未来需实现无监督空间关系推断以提升开放域适应性。

---

## 718. Reward Redistribution for CVaR MDPs using a Bellman Operator on L-infinity

**arXiv ID:** 2602.03778 | [PDF](https://arxiv.org/pdf/2602.03778v1)

**作者:** Aneri Muni `[一作]` (Mila Quebec AI Institute), Erick Delage `[通讯]` (GERAD and HEC Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

重新设计静态CVaR MDP的Bellman算子，并提出基于离散化的价值迭代和Q学习算法，用以求解无限期静态CVaR问题。

**💡 创新点**

创新点在于通过代数变换得到稠密奖励且收敛的Bellman算子，解决了先前方案中的稀疏奖励和收敛限制问题，并给出了误差上界与离散化精度的理论分析。

**🔧 技术方法**

采用状态增强、凸优化的CVaR表示、离散化Bellman算子、近似动态规划、Q学习以及误差分析等技术。

**📊 数据集**

在随机网格世界（Gridworld）环境中进行实验验证。

**📈 对比分析**

通过与风险中性策略和传统Q‑VI进行比较，展示在不同风险水平α下，风险厌恶策略在尾部性能上优于风险中性策略；误差随离散化细化而收敛，性能曲线与理论一致。

**⚠️ 局限性**

局限性主要是实验仅限于表格环境，缺乏对函数逼近或深度RL的扩展；在实际大规模问题中如何有效实现仍待研究。

---

## 719. Reasoning with Latent Tokens in Diffusion Language Models

**arXiv ID:** 2602.03769 | [PDF](https://arxiv.org/pdf/2602.03769v1)

**作者:** Andre He `[一作]` (Carnegie Mellon University), Daniel Fried `[通讯]` (Carnegie Mellon University)

**通讯引用:** 9884 | [OpenAlex ID](https://openalex.org/A5003637850)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨并引入潜在令牌机制，提升扩散式和自回归语言模型在全局一致性与规划任务中的推理性能，并实现推理速度与质量的可调权衡。

**💡 创新点**

提出可调节潜在令牌数量的思想，并将其迁移到自回归模型，显著缩小两种模型在推理任务上的性能差距。

**🔧 技术方法**

使用离散扩散（masked diffusion）模型、双向Transformer以及多标记预测目标，结合自回归训练与潜在令牌辅助训练。

**📊 数据集**

在合成推理任务（如数独、约束满足）以及标准自然语言文本基准上进行评估。

**📈 对比分析**

与传统独立预测扩散模型和自回归模型对比，发现更多潜在令牌可显著提升准确率与困惑度；在统一解码下，潜在令牌自回归模型往往优于扩散模型。

**⚠️ 局限性**

主要限制在于需手动设定潜在令牌数量与扩散步骤，权衡仍受限，且在更大规模或高维任务中计算成本可能上升。

---

## 720. RegionReasoner: Region-Grounded Multi-Round Visual Reasoning

**arXiv ID:** 2602.03733 | [PDF](https://arxiv.org/pdf/2602.03733v1)

**作者:** Wenfang Sun `[一作]` (University of Amsterdam), Cees G. M. Snoek `[通讯]` (University of Amsterdam)

**通讯引用:** 15475 | [OpenAlex ID](https://openalex.org/A5024508073)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RegionReasoner，一种多轮视觉语言推理框架，能够在每轮对话中生成可验证的、包含全局场景描述、局部区域描述、推理过程和最终答案的结构化轨迹，并通过强化学习优化其推理质量。

**💡 创新点**

创新点在于：①引入“引用引用”机制，要求推理轨迹中显式引用前一轮定位框，实现可验证的空间引用；②设计全局–局部语义一致性奖励，促使推理轨迹与全局场景和局部区域描述保持语义对齐；③构建 RegionDial‑Bench 多轮检测/分割基准，系统评估多轮推理性能。

**🔧 技术方法**

核心技术包括：基于大规模视觉语言模型的自回归策略网络，结构化的 JSON 语法约束解码；强化学习（GRPO）结合多项奖励（引用奖励、全局‑局部一致性奖励、基础几何奖励等）；以及对推理轨迹的自动化解析与奖励计算。

**📊 数据集**

使用公开的 RefCOCO+ 与 RefCOCOg 数据集，按程序化方法构建成多轮对话形式，涵盖检测和分割两类任务。

**📈 对比分析**

在 RegionDial‑Bench 上，RegionReasoner‑7B 在多轮检测任务的平均 AP 上领先 VisionReasoner‑7B 约 5.9/4.6 点，在分割任务的平均 gIoU 上领先 VisionReasoner‑7B 约 5.3/6.6 点，并且在后续回合中误差累积更慢，整体性能优于现有 VLM 与任务专用模型。

**⚠️ 局限性**

局限性包括：仍依赖于现有的 RefCOCO+ / RefCOCOg 语料，难以直接迁移到未覆盖的领域或更复杂的交互场景；奖励设计与推理流程较为复杂，训练成本较高；以及模型在极端稀疏或极度遮挡的视觉情境下仍可能出现误判。

---

## 721. Multimodal Generative Recommendation for Fusing Semantic and Collaborative Signals

**arXiv ID:** 2602.03713 | [PDF](https://arxiv.org/pdf/2602.03713v1)

**作者:** Moritz Vandenhirtz `[一作]` (ETH Zurich), Michael Louis Iuzzolino `[通讯]` (Meta AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态生成式推荐框架MSCGRec，融合语义和协同信号

**💡 创新点**

创新点在于将协同嵌入视作单独模态、使用基于DINO的自监督量化学习图像特征，并通过可行码约束的序列学习提升性能

**🔧 技术方法**

采用残差量化、DINO自监督量化、T5序列模型、相对位置嵌入、可行码约束的softmax、注意力机制

**📊 数据集**

在Amazon 2023 Beauty & Sports、PixelRec三个大规模真实数据集上进行实验

**📈 对比分析**

与传统序列推荐（SASRec等）及生成式基线（TIGER、LETTER、ETEGRec、MQL4GRec）对比，MSCGRec在Recall、NDCG、MRR等指标上均显著优于所有基线，并能在大数据集上击败序列推荐模型

**⚠️ 局限性**

局限性包括对模态缺失的鲁棒性仍需进一步验证，模型对极大项目集的扩展性及训练成本仍相对较高

---

## 722. OCRTurk: A Comprehensive OCR Benchmark for Turkish

**arXiv ID:** 2602.03693 | [PDF](https://arxiv.org/pdf/2602.03693v1)

**作者:** Deniz Yılmaz `[一作]` (Middle East Technical University), Bilge Kaan Görür `[通讯]` (Roketsan Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

发布并评测了OCRTurk，一个包含180页、三难度层级、四类土耳其文档（学术、非学术、论文、幻灯片）的文档解析基准，并对七个OCR模型进行细粒度性能评估。

**💡 创新点**

首次构建真实多元化土耳其文档解析基准，覆盖表格、公式、图像等多种元素，公开数据与评测脚本，并对不同难度与文档类型进行系统对比。

**🔧 技术方法**

采用多模型OCR（PaddleOCR、DeepSeek-OCR、Docling、NanonetsOCR2、HuanyanOCR、NVIDIA Nemotron、OlmOCR2）以及NED、TCS、BLEU、CDM、TEDS、MSE、DS等多维评测指标。

**📊 数据集**

OCRTurk基准由180页土耳其文档组成，来源包括arXiv、DergiPark、YÖK Tez、MEB OGM、Ankara Açık Öğretim等公开平台，按学术、非学术、论文、幻灯片四类与三难度层级划分。

**📈 对比分析**

通过对文本、表格、公式、图像的多指标逐项对比，PaddleOCR在整体指标上最高，DeepSeek-OCR在图像提取最佳，HuanyanOCR在土耳其字符敏感度最好；模型在不同难度和文档类型上表现差异显著。

**⚠️ 局限性**

仅包含180页数据，人工标注成本高，缺少更大规模与更丰富文档类型，覆盖度有限。

---

## 723. Bringing Reasoning to Generative Recommendation Through the Lens of Cascaded Ranking

**arXiv ID:** 2602.03692 | [PDF](https://arxiv.org/pdf/2602.03692v1)

**作者:** Xinyu Lin `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60270 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为CARE的级联推理框架，旨在消除生成式推荐（Generative Recommendation, GR）模型中出现的“偏差放大”问题，从而提升推荐多样性与用户体验。

**💡 创新点**

创新点在于：①引入渐进式历史编码机制与渐进注意力掩码，使模型在生成不同层级Token时能获取更细粒度、异质的用户历史信息；②设计查询锚定推理（query‑anchored reasoning）机制，在单次前向推理中并行执行多步推理，显著增强计算资源利用率并缓解偏差；③通过多头相似度多样性损失鼓励推理向量多样化，进一步降低Token偏差。

**🔧 技术方法**

使用的技术主要包括：大规模预训练语言模型（如Qwen2.5‑0.5B）作为backbone；自回归与并行两种生成式推荐方式；渐进式注意力掩码；查询锚定推理；多样性损失；以及对比实验与 ablation 分析。

**📊 数据集**

实验数据集包含四个真实业务场景：Amazon Reviews的Games、Sports、Toys三个子域以及微视频推荐数据集MicroLens，分别用于评估准确率、NDCG、推荐多样性（DivR@K）和过度推荐比例（ORR@K）。

**📈 对比分析**

在四个数据集上，CARE在自回归GR（TIGER、LETTER）以及并行GR（SETRec）中均取得了Recall@K、NDCG@K的提升（约1–4%）、多样性显著提升（DivR@K提升约10–20%），并保持与基线相近的推理速度与显存占用。相比传统多阶段推荐与其他推理/去偏方法（ReaRec、SPRec），CARE在准确率和多样性上均表现更优。

**⚠️ 局限性**

限制方面：①CARE对自回归GR的提升相对显著，但对并行GR的提升有限，表明在Token不严格遵循粗细层级的场景下，其渐进编码优势不明显；②当前的推理步骤数固定，未实现动态计算分配，可能导致不同用户历史长度下效率不均衡；③实验仅在四个公开数据集验证，缺乏大规模工业部署与长期用户行为影响的评估。

---

## 724. Efficient Training of Boltzmann Generators Using Off-Policy Log-Dispersion Regularization

**arXiv ID:** 2602.03729 | [PDF](https://arxiv.org/pdf/2602.03729v1)

**作者:** Henrik Schopmans `[一作]` (Institute for Anthropomatics and Robotics, Karlsruhe Institute of Technology), Pascal Friederich `[通讯]` (Institute for Anthropomatics and Robotics, Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了一种称为离线日志分散正则化（LDR）的方法，用于在训练玻尔兹曼生成器时利用能量标签来提升数据效率和最终性能。

**💡 创新点**

创新点在于将对数方差目标推广为对数分散目标，并将其作为离线（off‑policy）形状正则化器，能够在不额外采样或能量评估的前提下利用已存在的能量标签；该正则化可应用于无偏、偏置或纯变分训练，显著提升数据与能量评估效率。

**🔧 技术方法**

核心技术包括：离线日志分散正则化（LDR）、对数分散目标（L1和L2变体）、前向KL或路径梯度KL训练、离散可逆正则化流（normalizing flow）以及自适应重要性采样与温度退火的变分训练框架（CMT、TA-BG、FAB）。

**📊 数据集**

使用的实验数据集包括：二维高斯混合模型、氨基酸二肽（Alanine dipeptide）和六肽（Alanine hexapeptide）；训练数据来源于无偏MD、偏置MD（重要性采样）和仅能量评估的变分设置，样本规模从1e5到5e6不等。

**📈 对比分析**

与传统前向KL、路径梯度KL、CMT、TA-BG、FAB等基线相比，LDR在所有设置下均显著提升负对数似然（NLL）和有效样本量（ESS）。在无偏数据训练中，1e6样本的LDR模型可媲美5e6样本的基线；在偏置数据训练中，使用1e5重要性样本加上偏置数据即可获得与10倍样本量基线相当的性能；在纯变分训练中，LDR将CMT所需的目标能量评估量从1e8降至1e7，提升约10倍效率。

**⚠️ 局限性**

主要局限包括：需要调节正则化权重的超参数，且当参考分布与目标差异过大时可能导致训练不稳定；此外，实验仍依赖大规模数据集，虽然LDR提高了数据效率，但对极大规模或极高维系统的鲁棒性和可扩展性仍需进一步验证。

---

## 725. Efficient Estimation of Kernel Surrogate Models for Task Attribution

**arXiv ID:** 2602.03783 | [PDF](https://arxiv.org/pdf/2602.03783v1)

**作者:** Zhenshuo Zhang `[一作]` (Northeastern University), Hongyang R. Zhang `[通讯]` (Northeastern University)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5013834846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了基于核回归的任务归因方法，结合二阶分析、影响函数与梯度近似，显著提升多任务模型的归因精度。

**💡 创新点**

将线性归因与影响函数统一为二阶视角，引入RBF核非线性归因模型，并设计无须重新训练的梯度投影估计。

**🔧 技术方法**

使用二阶泰勒展开、影响函数、线性与核回归、梯度投影、随机子集采样、Hessian正则化等技术实现高效归因。

**📊 数据集**

在算术推理、上下文学习（SST-2、coin‑flip）、Meta‑World MT10 强化学习、CIFAR‑10 等数据集上进行实验。

**📈 对比分析**

与影响函数、TracIn、TRAK、SOURCE、BIF 等基线对比，核模型在离开一项检验相关度提升约25%，在算术推理上提高42%，在提示选择中降低40%损失。

**⚠️ 局限性**

主要评估集中于预测任务，对规划等语言模型能力关注不足；依赖一阶近似与子集采样，且在极大任务规模下仍需验证。

---

## 726. From Separate Compilation to Sound Language Composition

**arXiv ID:** 2602.03777 | [PDF](https://arxiv.org/pdf/2602.03777v1)

**作者:** Federico Bruzzone `[一作]` (Università degli Studi di Milano), Luca Favalli `[通讯]` (Università degli Studi di Milano)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

开发了一种基于数据流分析的静态检查工具，在 Neverlang 语言工作台中检测属性访问错误，并保持分离编译。

**💡 创新点**

首次在保持分离编译的前提下提供完整的静态正确性保证，利用数据流分析预防运行时未定义属性访问。

**🔧 技术方法**

采用数据流分析、程序依赖分析技术，并在 Neverlang 工作台实现；实验使用变异测试对错误进行评估。

**📊 数据集**

使用真实的 Neverlang 项目集，并通过变异测试生成错误实例。

**📈 对比分析**

与传统动态映射方法对比，实验显示检测覆盖率提升，运行时错误率下降，性能影响微乎其微，可在日常开发中使用。

**⚠️ 局限性**

仅针对属性访问错误，未覆盖其他语义错误；工具目前仅在 Neverlang 中验证，可能不适用于其他工作台；分析深度受 AST 生成限制。

---

## 727. An Empirical Study of Collective Behaviors and Social Dynamics in Large Language Model Agents

**arXiv ID:** 2602.03775 | [PDF](https://arxiv.org/pdf/2602.03775v1)

**作者:** Farnoosh Hashemi `[一作]` (Cornell University), Michael W. Macy `[通讯]` (Cornell University)

**通讯引用:** 19160 | [OpenAlex ID](https://openalex.org/A5017534073)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于LLM的社交代理平台Chirper.ai进行大规模长期实验，分析其同质性、社交影响、毒性语言、意识形态倾向、网络结构等，并提出一种零样本链式社会思考（CoST）干预方案。

**💡 创新点**

①在真实交互环境下首次大规模观察LLM社交行为，揭示其与人类相似的同质性与社交影响，但在毒性结构和意识形态极化上存在独特模式；②提出CoST，通过在提示中加入对潜在危害的反思，显著降低LLM的有害发帖率；③将LLM网络特性与人类网络、非交互LLM网络进行系统对比。

**🔧 技术方法**

使用SentenceBERT进行文本相似度度量、BERTopic进行主题建模、Perspective API评估毒性、RoBERTa/TweetNLP做情绪分析、GPT‑4o Mini进行立场与意识形态标注、BERT-base做倾向预测、链式推理CoST提示、网络分析（度分布、同质性系数、聚类系数、小世界指标）等。

**📊 数据集**

Chirper.ai的英文数据集：约7M条帖子、1M+互动、32K LLM代理，包含初始backstory信息。该数据为首批在真实交互平台上收集的大规模LLM社交数据。

**📈 对比分析**

通过与人类社交网络、非交互式LLM合成网络对比，评估同质性系数、交叉/同组比例、聚类系数及小世界特性。CoST实验显示受试LLM复发毒性降低约43%。文本检测模型随时间变得更易区分LLM生成文本。预测模型在加入邻居信息后RMSE和F1显著提升，证明社交上下文对倾向预测关键。整体性能表明LLM在互动环境中会出现可观的集体行为差异。

**⚠️ 局限性**

仅限英语数据，平台动作有限，缺乏人类参与，使用外部工具（Perspective API、GPT‑4o Mini）可能带来偏差；CoST实验仅测意向未观测实际后续发帖；研究为描述性，缺乏因果验证；未考虑多语言、跨平台、复杂多媒体交互等现实情境。

---

## 728. Mitigating Timing-Based Attacks in Real-Time Cyber-Physical Systems

**arXiv ID:** 2602.03757 | [PDF](https://arxiv.org/pdf/2602.03757v1)

**作者:** Arkaprava Sain `[一作]` (Indian Institute of Technology), Soumyajit Dey `[通讯]` (Indian Institute of Technology)

**通讯引用:** 421 | [OpenAlex ID](https://openalex.org/A5085224640)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 SecureRT 框架，利用基于任务级延迟的调度机制在实时闭环控制系统中抵御基于时序的侧信道攻击。

**💡 创新点**

创新点在于：① 在考虑实时可调度与控制性能约束的前提下，构造并求解针对每个安全任务的最优作业级延迟序列；② 将该序列与实时检测器结合，在线切换至带延迟的固定优先级调度器；③ 通过最小化攻击有效窗口与恶意任务执行时间重叠来降低攻击窗口。

**🔧 技术方法**

主要技术包括：最坏情况响应时间分析（RTA）求取可接受的最大延迟；线性规划（MILP）优化求解最优延迟序列；基于卡尔曼滤波与 LQR 的延迟感知控制器设计；χ² 残差检测实现异常检测；基于 Real‑Time Linux 的实时调度实现。

**📊 数据集**

实验数据集包括：1）基于汽车控制任务（巡航、ESP、轨迹跟踪）与非控制任务的真实任务集合；2）在 10‑20 个任务、10 个使用率区间内随机生成的 100 组合成任务集。

**📈 对比分析**

方法比较：基线固定优先级调度（PFP）、PFP+随机延迟、以及 SecureRT。实验显示，SecureRT 在遭受后置 FDI 攻击时，控制成本与估计状态误差与无攻击情形基本无差异，且攻击窗口被缩短约 60%（在汽车案例中）。相比之下，随机延迟导致控制成本显著上升，攻击成功率未降低。

**⚠️ 局限性**

局限性包括：仅在单核固定优先级调度下验证；对多核或动态优先级（EDF）系统的适用性尚未验证；假设检测器具有理想的误报/漏报率；攻击模型假设攻击者只能利用已知的 AEW，未考虑更强的时序预测或旁路信息。

---

## 729. Soft Sensor for Bottom-Hole Pressure Estimation in Petroleum Wells Using Long Short-Term Memory and Transfer Learning

**arXiv ID:** 2602.03737 | [PDF](https://arxiv.org/pdf/2602.03737v1)

**作者:** M. A. Fernandes `[一作]`, M. A. Sampaio `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于机器学习的软传感器，利用井口和平台测量数据估算井底压力；

**💡 创新点**

首次将迁移学习应用于井底压力软传感器，实现跨油田泛化并应对概念漂移；

**🔧 技术方法**

使用了长短时记忆网络、前馈神经网络与岭回归，并结合时间序列输入与迁移学习细化；

**📊 数据集**

使用巴西Pre‑salt离岸油田13年海量数据，约10万条样本，覆盖多井、多平台、多流动条件；

**📈 对比分析**

通过MAPE、SMAPE、RMSE与5折交叉验证及两组盲测对三种基线与LSTM模型进行比较，LSTM在最差情况下MAPE约1%~2%，优于传统回归与MLP；

**⚠️ 局限性**

局限在于仍需大量历史数据以保证模型泛化，迁移学习需预先拥有相关油田样本，且对极端工况的精度受限于PDG失效与测量噪声。

---

## 730. Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems

**arXiv ID:** 2602.03695 | [PDF](https://arxiv.org/pdf/2602.03695v1)

**作者:** Haibo Jin `[一作]` (University of Illinois), Haohan Wang `[通讯]` (University of Illinois)

**通讯引用:** 2534 | [OpenAlex ID](https://openalex.org/A5072244531)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Agent Primitives——一套可复用的潜在构建块，用于在 LLM 基础上构建多智能体系统，降低任务特定设计，提高鲁棒性与效率。

**💡 创新点**

创新点：①将现有多智能体系统拆解为三类通用原语（Review、Voting & Selection、Planning & Execution）并通过 KV‑Cache 进行潜在通信；②使用 LLM Organizer 与知识池自动生成任务无关的系统结构，实现系统化、模块化设计。

**🔧 技术方法**

技术：LLM KV‑Cache 潜在通信、RoPE 位置重编码、基于 LLM 的 Organizer 与知识池、原语组合图、自动化 MAS 构建。

**📊 数据集**

数据集：八个公开基准——数学推理（AIME24/25、MATH、GSM8K）、代码生成（MBPP‑Plus、HumanEval‑Plus）、问答（MedQA、GPQA‑Diamond）。

**📈 对比分析**

方法比较：与单一 LLM、TextMAS、LatentMAS 及 10 种主流 MAS（如 Chain‑of‑Thought、Self‑Consistency、LLM‑Debate 等）在 Qwen3、DeepSeek 等模型上对比。Primitives‑based MAS 在各任务平均提升 12–17% 以上，token 与推理时间相比 TextMAS 降低 3–4 倍，仅增加 1.3–1.6 倍相较单一 LLM，且在不同模型和任务上表现更稳定。

**⚠️ 局限性**

局限性：①KV‑Cache 通信需要相同 LLM 参数与位置编码，跨模型或不同 LLM 之间兼容性受限；②对 RoPE 位置重编码高度敏感，若无 RoPE 将导致显著性能下降；③需要手工收集和维护知识池，知识稀缺时效果下降；④在极大模型或跨任务迁移时实验尚未充分验证。

---

## 731. PLATE: Plasticity-Tunable Efficient Adapters for Geometry-Aware Continual Learning

**arXiv ID:** 2602.03846 | [PDF](https://arxiv.org/pdf/2602.03846v1)

**作者:** Romain Cosentino `[一作]` `[通讯]`, Romain Cosentino

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无旧任务数据的持续学习方法 PLATE，利用预训练网络的几何冗余实现数据自由的保护子空间和可调节的可塑性。

**💡 创新点**

创新点在于将冗余神经元选取与低能量输入基底相结合，构造权重只读的低秩适配器 ΔW = B A Qᵀ，直接从冻结权重推断保护子空间，并提供 r 与 τ 两个显式控制保留‑可塑性权衡的超参。

**🔧 技术方法**

核心技术包括权重导向的低秩适配器、冗余神经元选择、基于冻结层权重的低能量输入基底构造，以及功能漂移与旧任务曲率的理论分析。

**📊 数据集**

在实验中使用的主要数据集包括：LLM 预训练后 fine‑tune 的 Qwen2.5‑7B + DeepSeek‑R1、OLMo‑2‑7B + Tulu‑3、MNIST 0‑4 → 5‑9、AG News → IMDB，以及合成回归与其他标准 Vision/Text benchmark。

**📈 对比分析**

与 LoRA、全微调对比时，PLATE 在保持新任务性能相近的同时，旧任务遗忘显著降低；在 LLM 领域对 OOD 任务和大模型异构任务的实验中，表现优于 LoRA，尤其在低可塑性时遗忘几乎为零。

**⚠️ 局限性**

局限性在于需要额外的超参（r、τ）调优，计算上略有开销（约 10‑15% 训练时间增加），在极低可塑性时学习能力受限，且对非常大型模型的内存占用仍需进一步优化。

---

## 732. Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL

**arXiv ID:** 2602.03839 | [PDF](https://arxiv.org/pdf/2602.03839v1)

**作者:** Erfan Miahi `[一作]` (Covenant AI), Eugene Belilovsky `[通讯]` (Mila Concordia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用RL微调过程中的权重更新稀疏性实现的无损权重同步方法（Pulse），显著降低分布式RL中权重广播的带宽需求。

**💡 创新点**

创新点在于：①系统性分析了BF16精度与低学习率导致的权重更新被“吸收”而稀疏的机制；②提出仅传输改变参数的索引和值而非增量，避免浮点漂移；③通过稀疏补丁链实现高效无损同步，并给出带宽感知的压缩算法选择。

**🔧 技术方法**

技术实现包括：基于BF16训练的GRPO算法、权重差异检测与索引/值编码、可选的差分编码与类型下采样、Zstd/Lz4等通用压缩算法、集中式对象存储与异步补丁拉取。

**📊 数据集**

实验数据集：数学推理任务（MATH）和代码生成任务（MBPP），使用Qwen2.5-7B-Instruct模型进行微调。

**📈 对比分析**

与传统完整权重同步比较：Pulse在保持相同训练动态和验证准确率的前提下，将同步大小从14 GB压缩至约108 MB，带宽需求从约20 Gbit/s降至0.2 Gbit/s，实现≈100×的带宽降低，且训练效果无差异。

**⚠️ 局限性**

局限性包括：仅在GRPO单轮推理任务上验证，其他算法（PPO、DPO）或长序列任务可能表现不同；假设使用Adam优化器；未探讨极端带宽或更大延迟场景；对超参数（如学习率）变化的影响需进一步研究。

---

## 733. Continuous Control of Editing Models via Adaptive-Origin Guidance

**arXiv ID:** 2602.03826 | [PDF](https://arxiv.org/pdf/2602.03826v1)

**作者:** Alon Wolf `[一作]` (Tel Aviv University), Or Patashnik `[通讯]` (Tel Aviv University)

**通讯引用:** 2663 | [OpenAlex ID](https://openalex.org/A5076541595)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于扩散模型的连续视频/图像编辑方法，能够在输入与目标编辑之间实现平滑、可控的过渡。

**💡 创新点**

核心创新是“Adaptive‑Origin Guidance (AdaOr)”——通过引入可学习的“身份”提示，将无条件预测（原始CFG的指导起点）从任意编辑迁移到无编辑的起点，实现编辑强度的连续控制；并给出对应的插值调度。

**🔧 技术方法**

使用了扩散模型中的Classifier‑Free Guidance (CFG)，改进为三重预测（无条件、条件、身份），以及在训练中加入身份提示；在基础编辑模型Lucy‑Edit上实现。

**📊 数据集**

使用与Lucy‑Edit相同的编辑数据集（包含图像-目标配对），并在此基础上随机加入身份示例；视频测试采用同一模型处理单帧视频。

**📈 对比分析**

与FreeMorph、Kontinuous Kontext、Concept Sliders、SAEdit等现有连续编辑基线比较；在PIE‑Bench及自定义视频基准上，AdaOr在平滑度、文本一致性、感知路径一致性等三大指标均优于基线，且在用户研究中得到更好或相当的用户评价。

**⚠️ 局限性**

局限性包括：依赖底层编辑模型的能力，训练数据量有限导致编辑范围受限；Inference时需三次预测，计算开销略增；在极端编辑类型或模型欠拟合时可能出现失败或不必要的变动。

---

## 734. Perfect Network Resilience in Polynomial Time

**arXiv ID:** 2602.03827 | [PDF](https://arxiv.org/pdf/2602.03827v1)

**作者:** Matthias Bentert `[一作]` (TU Berlin), Stefan Schmid `[通讯]` (TU Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文对现代通信网络中的局部快速重路由机制进行了研究，提出了一种完整的特征描述，阐明了在何种情况下可以实现完美的弹性。该机制允许节点在链路故障时快速反应，确保数据包能够从源节点路由到目标节点，只要在故障后源节点和目标节点仍然连接。

**💡 创新点**

创新点在于提供了完美弹性的完整特征描述，并设计了O(n)时间复杂度的算法来判断给定实例是否具有完美弹性，以及O(nm)时间复杂度的算法来计算完美弹性的重路由规则。这些重路由规则的结构简单，易于硬件支持。

**🔧 技术方法**

使用了图论中的基本概念和算法，特别是关于图的嵌入、重根图的性质以及局部重路由算法的设计。

**📊 数据集**

论文中使用的图数据集是理论构造的，主要关注于具有特定结构的图，例如外平面图和双连通图。

**📈 对比分析**

与现有方法的比较表明，本文提出的算法在时间复杂度上是最优的，能够有效地处理完美弹性问题，并且在特定情况下比其他方法更快。

**⚠️ 局限性**

限制在于算法的复杂性分析较为复杂，且在处理某些特定类型的图时可能需要额外的预处理步骤。

---

## 735. Bridging Online and Offline RL: Contextual Bandit Learning for Multi-Turn Code Generation

**arXiv ID:** 2602.03806 | [PDF](https://arxiv.org/pdf/2602.03806v1)

**作者:** Ziru Chen `[一作]` (Ohio State University), Huan Sun `[通讯]` (Ohio State University)

**通讯引用:** 2492 | [OpenAlex ID](https://openalex.org/A5101488340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合在线与离线强化学习的上下文Bandit学习框架（CoBaL），用于多轮代码生成任务。

**💡 创新点**

1) 将多轮代码生成视为一阶可恢复的MDP，降低为单步Bandit优化；2) 先离线采集参考LLM生成的完整轨迹，再分段为上下文；3) 在线单步优化时使用KL约束，实现训练效率提升；4) 针对LLM的“奖励黑客”问题，引入扰动测试案例的数据增强，显著提升鲁棒性。

**🔧 技术方法**

上下文Bandit学习、GRPO算法、KL信任域约束、奖励黑客检测与扰动增强、对齐分析、单步RL奖励设计（R_correct, R_improve, R_format）等。

**📊 数据集**

TACO（包含6,103任务的验证子集）、LiveCodeBench（175题测试集）以及通过扰动生成的TACO-Dev-PTB。

**📈 对比分析**

与两种多轮在线RL基线（GRPO-MT、VeRPO-MT）以及单步RL对比，CoBaL在LiveCodeBench上对R1-Distill 8B、Qwen3 8B分别提升了9.0/6.2、6.2/3.2 Pass@1；在TACO-Dev上同样显著提升；训练时间更短、效率更高。

**⚠️ 局限性**

1) 仍存在奖励黑客中的语义漂移主导问题；2) 依赖参考LLM生成轨迹，若参考模型质量低会影响效果；3) 对更长回合的泛化虽有改善但仍有限；4) 在非代码生成类迭代任务的适用性尚未验证。

---

## 736. Prediction of Critical Heat Flux in Rod Bundles Using Tube-Based Hybrid Machine Learning Models in CTF

**arXiv ID:** 2602.03805 | [PDF](https://arxiv.org/pdf/2602.03805v1)

**作者:** Aidan Furlong `[一作]` (North Carolina State University), Xu Wu `[通讯]` (North Carolina State University)

**通讯引用:** 893 | [OpenAlex ID](https://openalex.org/A5014450583)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究将仅基于单通道（tube）数据训练的机器学习模型迁移至燃料棒束（rod bundle）几何结构，评估其在预测临界热流（CHF）位置与幅值方面的表现。

**💡 创新点**

创新点在于首次将纯数据驱动深度神经网络以及两种混合型（残差校正）模型应用于未适配的燃料棒束，并系统比较其与传统经验模型的泛化能力。

**🔧 技术方法**

采用深度神经网络（DNN）与混合校正框架（Bowring基准+ML残差校正、Groeneveld LUT基准+ML校正），并通过贝叶斯超参数优化、早停与指数学习率衰减实现训练。

**📊 数据集**

使用美国核监管委员会（NRC）24,579条tube式CHF实验数据作为训练集（经筛选后24,320条），以及CE 5×5燃料棒束实验系列（8个TS74/TS75案例）作为验证集。

**📈 对比分析**

通过与传统W‑3、Bowring、Groeneveld LUT模型的相对误差比较，结果显示混合LUT模型在CHF幅值与位置预测上误差最低，平均相对误差分别为±1.3%（幅值）和−0.14%（位置），显著优于基准模型。

**⚠️ 局限性**

局限性包括缺乏足够的燃料棒束实验数据导致模型泛化受限、子通道仿真中的不确定性较大、混合LUT模型在部分案例中表现出正偏差，且当前仅在单一测试束上验证，需进一步扩展数据集与采用迁移学习方法。

---

## 737. Conformal Reachability for Safe Control in Unknown Environments

**arXiv ID:** 2602.03799 | [PDF](https://arxiv.org/pdf/2602.03799v1)

**作者:** Xinhang Ma `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**通讯引用:** 5069 | [OpenAlex ID](https://openalex.org/A5038669899)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对未知动力学的自律系统，提出一种同时最大化奖励和严格保证概率安全的学习方法。

**💡 创新点**

创新点在于将分布无关的分形预测（conformal prediction）与有限时域可达性分析相结合，形成可微分的安全分析框架；并通过安全加权的动力学学习、可微分的安全集预测网络和安全约束损失，实现了模型无关、无光滑性假设的安全约束学习；同时采用基于安全时域的课程学习动态扩展预测时程。

**🔧 技术方法**

主要技术包括：分形预测、可达性分析、模型基础与模型无关的强化学习（PPO）、神经网络动力学模型与策略、可微分的安全阈值网络、联合损失优化与 Lagrange multiplier 调整。

**📊 数据集**

在七个环境上评估：2D Quadrotor（非线性安全约束）、Cartpole、Lane Following、2D Quadrotor（线性）、3D Quadrotor、CarGoal、HalfCheetah；其中后两者来自 Safety-Gym；实验使用随机采样的初始状态集作为验证集。

**📈 对比分析**

与五个最先进基线（PPOLag、PCPO、CRPO、P3O、RESPO）比较，ReCORS 在大多数环境下实现了更高的已验证安全概率，并在 3D Quadrotor、HalfCheetah 等场景中保持或提升奖励；即实现了安全与性能的双赢。

**⚠️ 局限性**

主要限制包括：假设系统状态完全可观测；需要额外的校准与验证数据；对初始状态分布假设有限；以及计算量较大，特别是在可达性分析与分形预测阶段。

---

## 738. FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation

**arXiv ID:** 2602.03798 | [PDF](https://arxiv.org/pdf/2602.03798v1)

**作者:** Zimu Lu `[一作]` (Multimedia Laboratory), Hongsheng Li `[通讯]` (Multimedia Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FullStack-Agent系统，整合多智能体框架、仓库反向翻译自我改进和完整栈评测，使用LLM从自然语言指令生成可运行的前后端与数据库代码。

**💡 创新点**

创新点在于①三层统一框架：规划代理、前后端编码代理及专用调试工具；②通过仓库反向翻译与增广生成高质量代理轨迹进行自我改进；③构建包含前端、后端与数据库功能的完整评测基准。

**🔧 技术方法**

采用多智能体协作、LLM（Qwen3-Coder）、代码编辑与Shell工具、专用前后端调试工具、仓库反向翻译、仓库增广、SFT自我改进及自动化评测流水线。

**📊 数据集**

使用GitHub真实全栈仓库、WebGen-Bench 101 条指令、以及自行构造的 647 前端、604 后端、389 数据库测试用例。

**📈 对比分析**

通过与 WebGen-Agent、TDDev、OpenHands、Bolt.diy、Qwen-Code 等基线在前端、后端、数据库三类测试上对比，Qwen3-Coder-480B 达到前端 64.7%、后端 77.8%、数据库 77.9%，相较 WebGen-Agent 提升 8.7%、38.2%、15.9%；Qwen3-Coder-30B 在自我改进后提升 9.7%、9.5%、2.8%。

**⚠️ 局限性**

受限于LLM对复杂依赖与长链推理的理解能力，评测主要基于自动化工具，可能遗漏细粒度错误，且生成网站仍需进一步安全与可维护性验证。

---

## 739. Manifold Random Features

**arXiv ID:** 2602.03797 | [PDF](https://arxiv.org/pdf/2602.03797v1)

**作者:** Ananya Parashar `[一作]` (Columbia University), Krzysztof Choromanski `[通讯]` (Google DeepMind)

**通讯引用:** 2685 | [OpenAlex ID](https://openalex.org/A5031842812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种新的随机特征方法（Manifold Random Features, MRFs），利用图随机特征（Graph Random Features, GRFs）作为教师，对离散化的流形进行监督学习，从而得到连续的正定核函数逼近。

**💡 创新点**

创新点在于：①将GRFs迁移到流形空间，通过离散化与随机游走得到的签名向量训练连续神经网络；②实现了在流形和欧氏空间（如高维立方体上的高斯核）上都可用的正值且有界的随机特征；③揭示了图随机游走与流形扩散/热核之间的深层异同，为核方法提供了低方差、可扩展的近似。

**🔧 技术方法**

使用的技术包括：图随机特征算法（基于随机游走和调制函数），神经网络监督学习（多层感知机），流形离散化与k‑NN图构造，随机抽样逼近积分，特征对齐（Frobenius对齐），以及对比实验中的基准谱分解。

**📊 数据集**

主要使用的数据集与实验场景包括：
- 4 个嵌入式 3D 曲面（球面、椭球面、莫比乌斯带、环面）
- 高维立方体网格（用于高斯核实验）
- Thingi10k 网格集合（用于顶点法向量预测）
- 运动捕捉网格（用于速度场插值）

**📈 对比分析**

与传统谱分解（显式构造完整核矩阵）比较，MRFs 在计算时间和存储上实现了 30–60 倍的加速；在错误指标（均方误差、相对误差）上与基准相当，且在大规模网格（>10^5 顶点）上仍能保持可行性。实验中 MRF 还在低维高斯核逼近中实现了与解析表达式几乎无差异的性能。

**⚠️ 局限性**

局限性：
- 需要先对流形进行高质量离散化，网格细化越细越慢；
- 随机游走参数和神经网络训练需要调优，过度采样或不恰当的停止概率会影响精度；
- 对非光滑或极其稀疏的采样（如莫比乌斯带的近重复点）更易出现数值不稳定；
- 目前仅在对称/光滑核（热核、高斯核）上验证，其他复杂核仍需进一步研究。

---

## 740. Investigating Quantum Circuit Designs Using Neuro-Evolution

**arXiv ID:** 2602.03840 | [PDF](https://arxiv.org/pdf/2602.03840v1)

**作者:** Devroop Kar `[一作]` (Rochester Institute of Technology), Travis Desell `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1175 | [OpenAlex ID](https://openalex.org/A5065630093)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于进化搜索的自动量子电路设计与训练框架EXAQC，用于生成满足任务需求的参数化量子电路。

**💡 创新点**

创新点包括：①将神经进化与遗传程序技术融合，设计可扩展的量子电路基因组与多样化变异算子；②引入Lamarckian权重继承与多父亲交叉，显著减少训练成本；③实现后端无关（支持Qiskit和Pennylane），可灵活切换执行环境；④提供多种损失函数（状态相似度、分布KL、交叉熵等），支持教师回放与监督学习两种任务。

**🔧 技术方法**

使用遗传算法、神经进化、参数梯度优化（Adam）、多种交叉算子（二进制、n-ary、指数）以及量子门变异算子（增删改换门、交换量子比特）等技术。

**📊 数据集**

使用UCI经典分类数据集（Iris、Wine、Seeds、Breast Cancer）以及随机生成的教师量子电路（身份、Bell态、跨注册、复合层）进行实验。

**📈 对比分析**

与传统手工模板与单一启发式设计相比，EXAQC在仅评估500个基因体的预算下，能够快速找到准确率>90%的分类电路，并在教师电路仿真中实现高达98%以上的保真度；相比单目标进化，使用多目标度量（保真度、角度距离）可更好捕获目标行为。

**⚠️ 局限性**

局限性包括：目前仅使用单一稳态种群且单目标优化，缺乏岛屿或多目标策略；变异算子与门库仍为固定集合，缺乏自适应门选择；对噪声与硬件约束的评估有限，需进一步扩展至真实量子设备。

---

## 741. EventNeuS: 3D Mesh Reconstruction from a Single Event Camera

**arXiv ID:** 2602.03847 | [PDF](https://arxiv.org/pdf/2602.03847v1)

**作者:** Shreyas Sachan `[一作]` (Saarland University), Vladislav Golyanik `[通讯]` (MPI for Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了仅使用单目事件相机数据进行稠密3D网格重建的方法，能够在无RGB帧、无显式特征匹配的情况下实现高精度的表面重建。

**💡 创新点**

创新点包括：①将神经隐式SDF与NeRF渲染结合，形成自监督事件驱动的学习框架；②首次在事件驱动3D重建中引入球谐（SH）编码以高效处理视角依赖的光照与反射；③采用重要采样、频率退火与Eikonal正则化等技术提升几何与外观细节；④仅通过事件数据训练，消除对RGB或多模态输入的依赖。

**🔧 技术方法**

主要技术包括：事件帧累积与时间窗口化、神经隐式SDF与颜色网络（两个MLP）、球谐编码、重要采样与层次采样、频率退火、负采样、Eikonal损失、基于事件的自监督损失。

**📊 数据集**

实验使用了合成NeRF数据集（Chair、Mic、Hotdog、Drums、Lego）以及真实EventNeRF数据集（DAVIS 346C 事件相机捕获的场景）。

**📈 对比分析**

与E2VID+NeuS、EventNeRF、PAEv3D等方法比较，在Chamfer Distance与SDF‑MAE上均取得显著优势（例如CD从0.107降至0.040，MAE从0.052降至0.017），并在新视角合成指标（SSIM、LPIPS）上表现更佳，整体在10个场景中获得9个最佳分数。

**⚠️ 局限性**

局限性包括：仅适用于中小规模物体，无法处理大规模场景；事件窗口采样导致高频纹理和光照变化信息损失；重建中仍出现纹理印记伪影；需要精确的相机位姿估计。

---

## 742. Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing

**arXiv ID:** 2602.03845 | [PDF](https://arxiv.org/pdf/2602.03845v1)

**作者:** Tong Zheng `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24723 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 2D probing 以及并行思维控制器 Parallel‑Probe，用以提升大语言模型在并行推理过程中的效率与准确性。

**💡 创新点**

创新点在于：①引入 2D probing 接口以全局观察宽度–深度维度的推理轨迹；②基于全局共识的早停与偏差驱动的分支剪枝，实现训练无关的在线资源调度；③构建 SCOUT 测试床，将推理空间与策略评估解耦，方便快速探索不同宽深配置。

**🔧 技术方法**

技术包括：黑盒终止触发式中间回答抽取、矩阵化 2D probing、共识判定（mode）与稳定性检测、偏差阈值剪枝、热身阶段控制以及离线模拟策略的 SCOUT 框架。

**📊 数据集**

数据集涵盖三大推理任务：AIME 2024、AIME 2025 与 HMMT 2025，使用 Qwen‑3 系列模型（0.6B、1.7B、4B、8B）进行实验。

**📈 对比分析**

与 Self‑Consistency、Adaptive Self‑Consistency、Early‑Stopping Consistency 等基线相比，Parallel‑Probe 在相同或更低的总 token / 顺序 token 成本下，保持甚至提升准确率；在总 token 下降 20–25% 之余，顺序 token 下降 30–35%，形成更优的 Pareto 前沿。

**⚠️ 局限性**

局限性包括：目前仅采用无训练的控制策略，缺乏针对不同任务的自适应学习；2D probing 仅通过终止触发的答案信息，未利用隐藏层表示；热身与剪枝阈值仍需手动设定，可能对新模型或任务适配不够稳健。

---

## 743. PrevizWhiz: Combining Rough 3D Scenes and 2D Video to Guide Generative Video Previsualization

**arXiv ID:** 2602.03838 | [PDF](https://arxiv.org/pdf/2602.03838v1)

**作者:** Erzhen Hu `[一作]` (Autodesk Research and University of Virginia), Fraser Anderson `[通讯]` (Autodesk Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种结合粗糙3D场景阻塞、2D视频参考和生成式AI的预视觉化系统，可快速生成多样化视频预览并支持多级运动控制。

**💡 创新点**

创新点在于将粗略3D阻塞与2D视频、可调相似度的图像风格化结合，提供三层运动精度（粗粒、风格化、控制视频），实现从结构到创意的无缝过渡。

**🔧 技术方法**

技术实现包括React Three Fiber的3D编辑、ControlNet与FlowEdit的图像风格化、LoRA微调、Stable Diffusion / Wan 2.1 / VACE的生成模型、Skeleton提取与视频混合渲染。

**📊 数据集**

数据来源主要为Sketchfab公开GLB模型、Hunyuan‑3D生成的3D角色、用户上传或公开的2D视频素材，LoRA训练使用自采集的角色/场景图像。

**📈 对比分析**

通过10名电影制片人/3D艺术家进行用户研究，使用SUS量表与定性访谈评估易用性、创意支持；生成时间约1分钟/片段，输出质量与传统预视觉化工具相当但速度更快，未做直接基准对比。

**⚠️ 局限性**

局限包括生成模型的时序一致性与跨镜头连贯性不足、运动微调困难、LoRA训练成本高、推理延迟、潜在偏见与作者署名等问题。

---

## 744. Robust Intervention Learning from Emergency Stop Interventions

**arXiv ID:** 2602.03825 | [PDF](https://arxiv.org/pdf/2602.03825v1)

**作者:** Ethan Pronovost `[一作]` (University of Washington), Siddhartha Srinivasa `[通讯]` (University of Washington)

**通讯引用:** 18247 | [OpenAlex ID](https://openalex.org/A5077719529)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出鲁棒干预学习框架RIL，研究如何从人类紧急停机干预中学习，并提出RIFT算法通过残差细调结合先验策略与干预信号来提升政策；

**💡 创新点**

将干预学习视为细调问题，引入残差Q学习与先验策略的KL正则化，并给出理论证明在多种干预策略下可实现策略改进，同时对干预不完整性进行定量分析；

**🔧 技术方法**

残差Q学习、最大熵强化学习、软Actor‑Critic（SAC）、KL正则化、优势差分与可视化差分的理论分析；

**📊 数据集**

使用OpenAI Gymnasium环境（Lunar Lander、Half Cheetah、Bipedal Walker）模拟人工干预；先验策略通过行为克隆或基于RL得到；

**📈 对比分析**

与未正则化的RLIF做对比，使用干预率、成功率和平均奖励评估；RIFT在干预信息较少时显著提升成功率，随着干预信息增多其优势逐渐减小；ω的调参范围相对宽松；

**⚠️ 局限性**

需要先验策略包含有用且互补信息，否则RIFT无效；对早期终止处理需谨慎；实验仅在仿真环境中验证，未考虑非马尔可夫或延迟干预场景。

---

## 745. SymPlex: A Structure-Aware Transformer for Symbolic PDE Solving

**arXiv ID:** 2602.03816 | [PDF](https://arxiv.org/pdf/2602.03816v1)

**作者:** Yesom Park `[一作]` (University of California), Stanley Osher `[通讯]` (University of California)

**通讯引用:** 115875 | [OpenAlex ID](https://openalex.org/A5002037883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 SymPlex 框架，利用强化学习与结构感知 Transformer（SymFormer）自动发现 PDE 的解析符号解。

**💡 创新点**

在符号表达式生成中引入树相对自注意力和语法约束自回归解码，实现对树状符号依赖的建模，并将 PDE 误差作为奖励进行无监督强化学习搜索。

**🔧 技术方法**

使用结构化 Transformer、树相对自注意力、语法约束解码、多样性上 top‑k 内存、模仿学习、课程学习以及常数梯度优化等技术。

**📊 数据集**

通过一系列合成 PDE（Poisson、advection、heat、Eikonal、Burgers 及其参数化版本）的初始/边界条件作为实验数据集。

**📈 对比分析**

与 RNN‑based SSDE、Deterministic FEX、PINN+DSR、KAN 以及数值 WENO 对比，SymPlex 在所有测试 PDE 上实现 100% Symbolic Recovery Rate（SRR），MSE 几乎为零，推理速度快、存储占用极小。

**⚠️ 局限性**

局限在于高维 PDE 的组合爆炸、对符号词汇表选择的手工依赖、训练成本高以及缺乏严格的收敛与误差理论。

---

## 746. Fast-Slow Efficient Training for Multimodal Large Language Models via Visual Token Pruning

**arXiv ID:** 2602.03815 | [PDF](https://arxiv.org/pdf/2602.03815v1)

**作者:** Dingkun Zhang `[一作]` (Harbin Institute of Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 95062 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉 token 剪枝的双速训练框架（DualSpeed），在训练过程中交替使用 fast‑mode（剪枝+模式隔离）和 slow‑mode（完整视觉序列+自蒸馏），从而在保持模型性能的前提下显著提升多模态大语言模型的训练效率。

**💡 创新点**

创新点在于：① 将视觉 token 剪枝（VTP）从推理端迁移到训练端，并通过模式隔离（Mode Isolator）让模型同时适应剪枝和完整输入；② 在慢速模式下采用自蒸馏，将已训练好的 fast‑mode 作为教师指导 slow‑mode 学习完整视觉信息；③ 通过随机切换 fast‑mode 与 slow‑mode 的比例，平衡训练速度与性能，形成可调节的训练‑推理一致性框架。

**🔧 技术方法**

技术手段包括：视觉 token 剪枝（可插件化，如 DivPrune、FasterVLM、CDPruner）；soft‑prompt 模式隔离器；自蒸馏（KL 失真）；fast‑slow 模式切换策略；以及在预训练阶段冻结 LLM 的加速技巧。

**📊 数据集**

数据集：LLaVA‑Pretrain‑558K（图像-标题对）用于预训练；LLaVA‑665K（图像+指令-答案）用于监督微调（SFT）。

**📈 对比分析**

与基线（完整训练）和 NaivePrune（直接在训练中使用 VTP）对比：在 LLaVA‑1.5‑7B 上实现 2.1× 训练加速，性能保持 99.6%；在 LLaVA‑NeXT‑7B 上实现 4.0× 加速，性能保持 99.0%。在各种视觉语言基准（VQAv2、GQA、SQA、TextVQA、POPE、MME、MMBench 等）上，DualSpeed 在正常推理下几乎无性能损失，并在剪枝推理下表现优于基线。相比 NaivePrune，DualSpeed 将训练‑推理差距从约 3.7% 缩小至 0.4%。

**⚠️ 局限性**

限制：① 需要在训练中保留约 10% 的 slow‑mode 才能保持性能，仍增加一定的计算负担；② 超高剪枝比例（>90%）会导致性能急剧下降；③ 过度剪枝可能削弱模型在复杂视觉场景下的鲁棒性；④ 目前仅在 LLaVA 系列模型验证，跨模型泛化仍需进一步评估。

---

## 747. Enhancing Imbalanced Node Classification via Curriculum-Guided Feature Learning and Three-Stage Attention Network

**arXiv ID:** 2602.03808 | [PDF](https://arxiv.org/pdf/2602.03808v1)

**作者:** Abdul Joseph Fofanah `[一作]` (Griffith University), Shaoyang Zhang `[通讯]` (Chang'an University)

**通讯引用:** 366 | [OpenAlex ID](https://openalex.org/A5101598280)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于课程学习的三阶段注意力网络 CL3AN‑GNN，用以解决图神经网络中的节点分类不平衡问题。

**💡 创新点**

创新点在于：① 采用逐步递进的 Engage‑Enact‑Embed 课程学习框架；② 将节点与边特征融合的多头注意力机制；③ 在训练过程中动态调节权重和注意力，实现对难易样本的自适应学习。

**🔧 技术方法**

主要技术包括 GCN、GAT、三阶段注意力（Engage‑Enact‑Embed）、多头注意力、课程学习损失（权重自适应、熵正则化）以及边‑节点联合嵌入。

**📊 数据集**

使用八个公开图数据集：Cora、Citeseer、PubMed、Amazon Photo、Amazon Computers、Coauthor CS、Chameleon 以及 OGBN‑Arxiv，涵盖社交、产品、学术与大规模学术网络。

**📈 对比分析**

与 GraphSMOTE、GraphMixup、GATE‑GNN、ReVar‑GNN、Graph‑DAO 等多种基线相比，CL3AN‑GNN 在 accuracy、F1‑score 与 AUC‑ROC 上平均提升 3–7%（在 OGBN‑Arxiv 上最高达 5.9% F1、4.7% AUC），且在极端不平衡场景下仍保持稳健性能。

**⚠️ 局限性**

局限性包括：① 对课程参数（如 λ₁、λ₂、λ₃）较为敏感，需要经验调优；② 计算与存储成本比单纯 GCN 稍高，尚未完全验证在亿级节点大规模图上的可扩展性；③ 主要针对节点分类任务，边预测等其他图学习任务尚未直接验证。

---

## 748. Antidistillation Fingerprinting

**arXiv ID:** 2602.03812 | [PDF](https://arxiv.org/pdf/2602.03812v1)

**作者:** Yixuan Even Xu `[一作]` (Carnegie Mellon University), J. Zico Kolter `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18174 | [OpenAlex ID](https://openalex.org/A5075035644)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种抗蒸馏指纹识别（ADFP）方法，用来检测第三方模型是否对教师模型的输出进行过蒸馏。

**💡 创新点**

创新点在于将指纹嵌入目标直接对学生学习动态的梯度优化方式（利用代理模型计算最优的logit扰动），实现指纹与学生模型学习轨迹的最优对齐，从而在不明显损害生成质量的前提下提升指纹可检测性。

**🔧 技术方法**

技术核心是基于抗蒸馏采样的梯度导向logit扰动、红绿列表（red‑and‑green‑list）水印、统计指纹检测（平均绿色词概率与p值检验）以及LoRA微调。

**📊 数据集**

使用了数学推理数据集GSM8K和开放域对话数据集OASST1进行实验评估。

**📈 对比分析**

与传统的红绿列表水印方案相比，ADFP在保持相同生成质量（GSM8K准确率或OASST1 NLL）时，p值显著更低，实现了Pareto改进；在部分指纹化训练数据、不同学生模型等场景下也表现出更强的指纹可检测性。

**⚠️ 局限性**

局限性包括对代理模型的依赖（若代理模型与真实学生模型差异过大可能影响指纹效果）、在大规模多样化数据下的泛化能力尚未完全验证，以及在极少量指纹化样本时指纹效果仍会衰减。

---

## 749. xDevSM: An Open-Source Framework for Portable, AI-Ready xApps Across Heterogeneous O-RAN Deployments

**arXiv ID:** 2602.03821 | [PDF](https://arxiv.org/pdf/2602.03821v1)

**作者:** Angelo Feraudo `[一作]` (University of Bologna), Tommaso Melodia `[通讯]` (Northeastern University)

**通讯引用:** 19572 | [OpenAlex ID](https://openalex.org/A5054337759)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出并实现了xDevSM框架，统一抽象O‑RAN的KPM监测与RC控制两种服务模型，支持跨异构RAN栈的xApp开发、CI/CD自动化部署与闭环控制实验。

**💡 创新点**

创新点在于：① 将低层E2接口的e2sm编码/解码封装为高层API；② 允许单一xApp同时使用多种服务模型，实现监测+控制一体化；③ 搭建完整的CI/CD流水线，实现跨oai、srsRAN等栈的一键部署与验证。

**🔧 技术方法**

采用技术包括：O‑RAN E2接口、e2sm KPM与RC、Python+oscricricxappframe、FlexRIC工具链、InfluxDB+Grafana数据可视化、Open5GS核心网络、OpenShift/Kubernetes、USRP SDR、Foxconn RPQN硬件、oai与srsRAN软件栈。

**📊 数据集**

使用真实实验数据：单UE/双DU部署、USRP SDR、Foxconn RPQN、商用UE（OnePlus AC2003）、IPerf3 流量；未使用公开数据集，而是基于自建实验场景收集指标。

**📈 对比分析**

通过在oai和srsRAN两栈上实施监控、切片级PRB分配、闭环资源调度及移动性控制，对比控制延时、吞吐、PRB占用等指标，平均控制延时为4.5–7 ms，PRB分配能按预期限制吞吐，闭环控制对流量突变实现了及时响应。

**⚠️ 局限性**

局限性包括：oai仅部分实现RBC控制（无法评估其效果）；部分服务模型尚未完整支持；实验规模有限，未验证多UE/多RAN的鲁棒性；AI/ML算法未进行深入评估，主要聚焦规则/阈值控制。

---

## 750. Progressive Checkerboards for Autoregressive Multiscale Image Generation

**arXiv ID:** 2602.03811 | [PDF](https://arxiv.org/pdf/2602.03811v1)

**作者:** David Eigen `[一作]` `[通讯]`, David Eigen

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于平衡递进棋盘采样顺序的多尺度自回归图像生成方法。

**💡 创新点**

创新点在于结合跨尺度与同尺度条件，利用平衡的棋盘采样减少采样步骤并保持并行性。

**🔧 技术方法**

采用 Transformer+VAE 量化码本、RoPE、分类器无指导（CFG）等技术。

**📊 数据集**

在 ImageNet 256×256 图像数据集上进行训练与评估。

**📈 对比分析**

与其他 AR 与 PAR 等方法对比，使用 17 步即可达到 2.72 FID，速度与性能均优于 PAR 与 RandAR。

**⚠️ 局限性**

局限性包括需针对不同尺度比例单独训练、仅在 ImageNet 评估、对更高分辨率或视频等场景尚未验证。

---

## 751. Do We Need Asynchronous SGD? On the Near-Optimality of Synchronous Methods

**arXiv ID:** 2602.03802 | [PDF](https://arxiv.org/pdf/2602.03802v1)

**作者:** Grigory Begunov `[一作]` (Lomonosov Moscow State University), Alexander Tyurin `[通讯]` (AXXX)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新评估了传统同步随机梯度下降（SGD）及其鲁棒变体m-Synchronous SGD，并在多种异构计算场景下提供理论分析，证明它们在时间复杂度上与现代异步方法几乎同优；

**💡 创新点**

创新点在于提出了新的下界证明，证明在固定、随机、部分参与及非平稳计算模型下，m-Synchronous SGD（甚至普通Synchronous SGD）能够在对数或常数因子内达到最优；

**🔧 技术方法**

主要技术包括对异构梯度计算时间的随机/确定性建模、时间复杂度分析、下界构造、以及对“bubble”效应的理论剖析；

**📊 数据集**

实验数据集包括合成二次优化任务、CIFAR‑10 图像分类和 NanoGPT 语言模型训练；

**📈 对比分析**

与异步方法（Asynchronous SGD、Rennala SGD、Malenia SGD 等）比较，实验显示同步/ m‑同步方法在收敛速度上仅比最优异步慢约 1–2 倍，且在实际环境中与异步方法相当；

**⚠️ 局限性**

局限性包括：在计算时间突变、噪声极大或异构任务需要全部参与的情况下，异步方法仍可能更优；

---

## 752. Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting

**arXiv ID:** 2602.03809 | [PDF](https://arxiv.org/pdf/2602.03809v1)

**作者:** Leonardo Monchieri `[一作]` (University of Padova), Simone Milani `[通讯]` (University of Padova)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Split&Splat框架，先对多视图图像进行一致的实例分割，再独立使用3D Gaussian splatting重建每个对象，并在重建后嵌入实例级语义描述，最终合并为全景场景。

**💡 创新点**

创新点在于：① 先分割再重建，保持实例边界一致；② 在Gaussian中仅嵌入稀疏实例级语义特征，降低显存；③ 结合深度+SfM信息进行跨视图mask传播，提升一致性。

**🔧 技术方法**

使用了3D Gaussian splatting、SAM2实例分割、COLMAP SfM、单目深度估计、CLIP视觉语言模型、DBSCAN聚类、KMeans++采样等技术。

**📊 数据集**

实验数据集包括ScanNetv2（3D实例分割评估）和LERF（开词表分割与编辑评估）。

**📈 对比分析**

与InstanceGS等SOTA Gaussian方法对比，在ScanNetv2平均mIoU提升约6%至56.4%，在LERF开词表分割mIoU 55.7%排名第二；并在实例重建、边界精细化和下采样鲁棒性上表现优异。

**⚠️ 局限性**

局限性包括：在高实例密度场景下精度略下降；依赖深度/相机参数，无法完全处理极端遮挡或光照变化；未解决部分细部部件识别和实例重新标记等细粒度问题。

---

## 753. AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations

**arXiv ID:** 2602.03828 | [PDF](https://arxiv.org/pdf/2602.03828v1)

**作者:** Minjun Zhu `[一作]` (Westlake University), Yue Zhang `[通讯]` (Westlake University)

**通讯引用:** 11950 | [OpenAlex ID](https://openalex.org/A5100333689)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了用于长文本科学插图生成的全新大规模基准数据集 FigureBench，并提出了基于推理渲染的代理框架 AutoFigure，能够从长篇科学文本自动生成结构完整、视觉美观的出版级插图。

**💡 创新点**

创新点包括①将长文本插图任务拆分为语义解析与布局规划、审美渲染两阶段的代理架构；②引入自我批判-修正循环，使布局在生成过程中不断优化；③采用擦除-纠正机制提升文本清晰度；④通过 VLM‑as‑Judge 与专家评估双重评测提升评估可靠性。

**🔧 技术方法**

技术主要有：大语言模型（如 Gemini‑2.5‑Pro、GPT‑5）做概念提取与布局生成；多代理自我对话式优化；向量渲染模型（GPT‑Image）完成图像绘制；OCR 与多模态校验实现文本纠错；VLM‑as‑Judge 进行自动评价。

**📊 数据集**

使用自建 3,300 对高质量长文本–图像对的 FigureBench 数据集（包含论文、调查、博客、教材四类），其中 300 为测试集，3,000 为开发集；同时提供公开代码与 HuggingFace Space。

**📈 对比分析**

与基线（全流程 T2I、代码生成、Diagram Agent）相比，AutoFigure 在所有四类文档的整体得分和赢率均遥遥领先（例如教材类 97.5% 赢率，论文类 53%），人类专家评价中 66.7% 的专家愿意直接采用 AutoFigure 生成的图像。

**⚠️ 局限性**

局限性主要体现在：仍需人工核对以避免误信息；在处理数据驱动图表等非概念性插图方面表现不足；对极长上下文的推理深度受限；模型在极端视觉创意或特定学科细节方面仍有欠缺。

---

## 754. Adaptive Evidence Weighting for Audio-Spatiotemporal Fusion

**arXiv ID:** 2602.03817 | [PDF](https://arxiv.org/pdf/2602.03817v1)

**作者:** Oscar Ovanger `[一作]` (University of Texas), Timothy H. Keitt `[通讯]` (University of Texas)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 FINCH 框架，将预训练的音频分类器与空间-时间先验通过自适应加权的 log‑linear 融合方法结合，实现对不同证据可靠性的动态调整。

**💡 创新点**

创新点在于：①引入样本级的门控网络估计上下文可靠性并限制其影响；②通过可学习的上界 ω_max 保证融合安全且保留音频基准；③不需要重新训练基础模型，只在融合层加入轻量门控。

**🔧 技术方法**

技术手段包括：log‑linear（乘积专家）融合、门控网络（两层 MLP）、温度缩放、方差正则化、基于不确定性与信息量的特征提取。

**📊 数据集**

使用的公开数据集为 Cornell Birdcall Identification (CBI) 与 BirdSet，空间时间先验来自 eBird AdaSTEM 或自学的元数据 MLP。

**📈 对比分析**

与音频单独模型和固定权重融合相比，FINCH 在 CBI 上提升了 2.0% 的 Top‑1 准确率，在 BirdSet 各子集的检索 AUROC、检测 cmAP 与 Top‑1 准确率上均保持或略优于最强基准，证明自适应加权显著提升性能。

**⚠️ 局限性**

局限性包括：对近似条件独立性的假设敏感；上下文先验本身相对弱，可能导致误判；门控网络可能收敛为常数或失效；在强依赖情形下，需采用联合模型。

---

## 755. Conformal Thinking: Risk Control for Reasoning on a Compute Budget

**arXiv ID:** 2602.03814 | [PDF](https://arxiv.org/pdf/2602.03814v1)

**作者:** Xi Wang `[一作]` (Johns Hopkins University), Eric Nalisnick `[通讯]` (Johns Hopkins University)

**通讯引用:** 1283 | [OpenAlex ID](https://openalex.org/A5003054496)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种分布无关的风险控制框架，用于在大语言模型的推理过程中实现自适应早停，从而在满足用户设定的错误率约束下最小化计算成本。

**💡 创新点**

创新点包括：①将早停阈值设置问题重新表述为风险控制；②提出双阈值机制（上阈值用于“自信停止”，下阈值用于“无进展提前终止”）；③利用验证集进行分布无关风险估计并加入有限样本校正；④在同一模型上联合多种不确定性信号形成“信号集成”，进一步提升效率。

**🔧 技术方法**

核心技术包括：分布无关风险控制（如UCB校正）、自定义误差与效率损失函数、参数化下阈值函数（Sigmoid），以及与多种不确定性信号（confidence、EAT、probe、token计数）配合的早停策略。

**📊 数据集**

实验使用的主要数据集有：AIME、DeepScaleR（去除AIME）、GPQA-Diamond、MathVision，以及通过组合AIME和GPQA构造的可控 solvable/unsolvable 比例集，模型覆盖 Qwen3-8B、Qwen3-30B-A3B、DeepSeek-R1-Distill-Qwen-32B、Qwen3-VL-8B。

**📈 对比分析**

与单阈值上限或下限策略以及无校正的交叉验证策略相比，本文方法在满足给定风险约束时显著降低 token 消耗，且在不同 solvable/unsolvable 组合下保持更高的准确率；实验中风险控制框架的风险上界始终位于目标风险之下，效率损失明显优于基线。

**⚠️ 局限性**

主要局限包括：①依赖于足够大小且代表性的验证集，验证集过小或分布偏移时仍需谨慎；②下阈值的参数化形式可能需要针对不同任务微调；③在极端长推理或高难度任务中，误差/效率损失仍可能较大；④双阈值的联合搜索可能增加调参成本。

---

## 756. 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation

**arXiv ID:** 2602.03796 | [PDF](https://arxiv.org/pdf/2602.03796v1)

**作者:** Zhixue Fang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 3DiMo，一种端到端的隐式 3D 运动控制框架，能够从 2D 驾驶视频中学习视角无关的运动表示，并通过交叉注意力在预训练的 DiT 视频生成器中实现自由文本驱动的相机控制。

**💡 创新点**

① 通过 Transformer 1D 编码器消除 2D 结构信息，学习视角不敏感的运动语义；② 在预训练生成器上联合训练，确保运动表示与生成器内在的 3D 先验对齐；③ 使用视角丰富的监督（单视、多视、移动摄像）和可逐步消除的几何辅助，促使模型获得真正的 3D 运动理解。

**🔧 技术方法**

隐式 1D 运动编码器（Transformer）、跨注意力条件机制、DiT 预训练视频生成器、VAE + diffusion 训练框架、轻量级几何解码器用于 SMPL/MANO 辅助监督、视角丰富的数据增强与多阶段训练策略。

**📊 数据集**

结合来自互联网、Unreal Engine 5 合成渲染和真实多视角拍摄的数据，覆盖单视、固定多视角和移动摄像三种摄像机配置，并利用 Qwen2.5-VL 自动生成相机描述文本。

**📈 对比分析**

与 AnimateAnyone、MimicMotion、Uni3C、MTVCrafter 等 2D 姿态或 3D SMPL 基线在 50 条 TikTok 片段和 100 条互联网视频上对比。3DiMo 在 LPIPS、FID、FVD 等指标上均优于所有基线，尤其在 3D 物理一致性和相机控制方面表现突出；人类评测 MOS 亦显示在动作准确性、自然度和 3D 可行性上领先。

**⚠️ 局限性**

仍受限于训练数据的视角覆盖范围和几何辅助的初始依赖；当相机变化极端或运动极为复杂时，运动表示可能出现轻微失真；模型在完全没有文本提示的场景下对相机控制的自适应性尚待进一步提升。

---

## 757. Accelerating Scientific Research with Gemini: Case Studies and Common Techniques

**arXiv ID:** 2602.03837 | [PDF](https://arxiv.org/pdf/2602.03837v1)

**作者:** David P. Woodruff `[一作]` (Google Research), Vahab Mirrokni `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过Gemini系列大语言模型在理论计算机科学、经济学、物理等学科的多项案例研究，展示了AI与人类研究者在解决开放问题、反驳猜想、生成新证明时的协作方式；

**💡 创新点**

创新点在于提出并验证一套通用的AI‑人类协作技术与流程——包括迭代精炼、问题分解、对抗式审查、神经符号循环与跨学科知识迁移，证明AI可作为主动而非被动的科研伙伴；

**🔧 技术方法**

主要技术包括Gemini Deep Think等前沿大语言模型、对话式推理与自我纠错、自动代码生成与执行、对抗式审稿机制、以及跨学科检索与知识合成；

**📊 数据集**

使用的数据集为Gemini训练时构建的高质量数学解题语料、公开论文与专用知识库，涵盖算法、密码学、几何等多领域文献；

**📈 对比分析**

对比方法为将模型生成的解与人工专家评审、传统手工证明及现有工具结果对齐，实验表明模型在多项案例中成功率达90%以上，解决时间缩短数十倍，错误率可通过自我纠错和人类验证降至可接受水平；

**⚠️ 局限性**

局限性包括模型仍易产生幻觉、对极其专业或新颖领域的深度推理不足、需要人工干预验证、以及对极大规模数据或计算复杂度高的任务仍受限。

---

## 758. They Said Memes Were Harmless-We Found the Ones That Hurt: Decoding Jokes, Symbols, and Cultural References

**arXiv ID:** 2602.03822 | [PDF](https://arxiv.org/pdf/2602.03822v1)

**作者:** Sahil Tripathi `[一作]` (Jamia Hamdard), Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 2941 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三阶段框架 Cross-ALIGN+，用于提升图文混合 Meme 的社交骚扰检测。

**💡 创新点**

创新点在于：①将结构化文化知识与 LVLM 表征对齐，②使用 LoRA 结合对比损失精细化决策边界，③生成基于证据链的可解释文本。

**🔧 技术方法**

核心技术包括：外部知识检索（ConceptNet、Wikidata、Hatebase）、跨模态注意力融合、LoRA 参数高效微调、对比学习以及模板化解释生成。

**📊 数据集**

实验使用 GOAT‑Bench 五大子集（Harmfulness、Hatefulness、Misogyny、Offensiveness、Sarcasm）和八个 7B 级 LVLM 进行评估。

**📈 对比分析**

相较于零射击和 ICL 基线，平均提升约 12% F1，单模型最高可达 17% 相对提升，同时解释质量也得到显著提高。

**⚠️ 局限性**

局限性：对实体检索和知识库覆盖率敏感，跨语言或新符号的适应性有限；解释模板固定，缺乏表达多样性。

---

