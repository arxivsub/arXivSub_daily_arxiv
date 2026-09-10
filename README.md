# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-10 | 今日论文总数: 729

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Open-Set Ego-Noise Separation for Legged-Robot Audition via Annotation-Free Adaptation and Pretrained-Model Transfer

**arXiv ID:** 2609.07440 | [PDF](https://arxiv.org/pdf/2609.07440v1)

**作者:** Koki Shoda `[一作]` (University of Tokyo), Atsushi Yamashita `[通讯]` (University of Tokyo)

**通讯引用:** 16209 | [OpenAlex ID](https://openalex.org/A5047464293)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种开放集腿形机器人回声噪声分离框架，利用无标注预部署录音自动选取噪声片段并通过预训练模型迁移实现高保真分离；

**💡 创新点**

创新点包括无标注无手工注释的递归图算法(RecurGraph)自动识别ego‑noise，和基于Diffusion Transformer的Transfer‑DiT通过Bottleneck Adapter与LoRA实现低成本迁移；

**🔧 技术方法**

使用的技术包括预训练音频‑语言嵌入、PCA与图传播、VAE+Transformer特征编码、Flow Matching、Diffusion Transformer、Bottleneck Adapter、LoRA以及多种数据增强（RIR、SNR、速度变换）；

**📊 数据集**

数据集涵盖PE_AV预训练模型、FSD50K环境音、ESC‑50/TUT Rare Sound事件集以及实际录制的ego‑noise和物理播放混合音频；

**📈 对比分析**

与字典NMF、Conv‑TasNet、Conformer、CLAPSep、SAM‑Audio等多种基线在参考音频与物理播放上对比，Transfer‑DiT在CLAP_A、SAJ、AUPRC_anom、Acc_52等指标上均优于或匹敌最佳基线，尤其在低SNR环境下表现突出；

**⚠️ 局限性**

局限性在于需预部署录音中存在足够比例的ego‑noise片段，对频繁出现的环境类敏感；目前仅单通道实现，物理播放场景的泛化仍受限，未来需进一步提升持续适配与连续学习能力。

---

## 2. Self-Supervised Multi-View 3D Gaze Target Estimation via Probabilistic Ray Marching

**arXiv ID:** 2609.07415 | [PDF](https://arxiv.org/pdf/2609.07415v1)

**作者:** Keqi Chen `[一作]` (University of Strasbourg), Nicolas Padoy `[通讯]` (University of Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 Self-MVGTE，一种自监督的多视角 3D 视线目标估计方法，直接在三维空间定位视线目标。

**💡 创新点**

创新点在于将估计的 3D 视线方向作为几何先验，构造可概率化的射线行走（ray marching）搜索空间，并通过软标签与 KL 损失实现对噪声伪标签的自监督学习。

**🔧 技术方法**

采用了 DINOv2 的语义特征、Depth-Anything-3 的深度引导、Gaze-LLE 的改进框架以及 3D 视线向量的融合，整体实现了概率分布式定位与可微几何门控。

**📊 数据集**

使用 MVGT 数据集进行训练与评估，并在训练阶段利用 Gaze-LLE（加 3D priors）生成伪 2D 视线标签。

**📈 对比分析**

与现有单目和多视角全监督方法对比，Self-MVGTE 在 3D 距离、角误差和 2D 误差上均优于对手，尤其在离线 3D 距离上提升至 74.09 cm，角误差为 13.08°。

**⚠️ 局限性**

局限性包括：3D 视线向量误差导致目标超出射线锥；模型倾向于语义丰富区域，易将视线误定位至物体表面；对低视角数目时性能显著下降。

---

## 3. Trust the Spec, Not the Code - A Specification-First, AI-Assisted Case Study in Online Banking

**arXiv ID:** 2609.07365 | [PDF](https://arxiv.org/pdf/2609.07365v1)

**作者:** Eitan Farchi `[一作]` `[通讯]` (IBM Research), Eitan Farchi (IBM Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在在线银行转账服务中应用并验证了一种基于AI的“先写规范、后生成代码”的开发流程，并通过增加运行时覆盖模型和Z语言接口层进一步扩展和验证该方法。

**💡 创新点**

首次将该流程迁移到新的领域并在更复杂的定时/循环转账场景下进行压力测试；引入AI提出的运行时覆盖准则和完整的Z接口自动生成，为规范驱动开发提供了完整的验证闭环。

**🔧 技术方法**

使用大型语言模型（ChatGPT）进行规范编写、歧义检测、证明生成和代码实现；利用轻量级数学表达式、Z规范进行接口约束；在Python中实现运行时检查与覆盖报告。

**📊 数据集**

无真实数据集；使用模拟的三名用户、六个账户以及随机生成的余额进行实验。

**📈 对比分析**

对比方法为仅基于代码检查的传统流程；报告显示生成的代码在首次生成时即满足规范，运行时 invariant 通过所有测试，覆盖率模型在实验中达 100%，但未进行系统化性能基准。

**⚠️ 局限性**

仅在规范层提供安全性保障，未对生成代码进行形式化的 refinement 验证；方法主要针对保守性不变式，无法直接推广到 liveness 或安全性等属性；实验仅为单一作者的案例研究，缺乏对比实验与可重复性评估。

---

## 4. Elastoplastic inherent strain-based topology optimization for residual stress reduction in metal additive manufacturing

**arXiv ID:** 2609.07337 | [PDF](https://arxiv.org/pdf/2609.07337v1)

**作者:** Takao Miki `[一作]` (Osaka Research Institute of Industrial Science and Technology), Shinji Nishiwaki `[通讯]` (Kyoto University)

**通讯引用:** 10328 | [OpenAlex ID](https://openalex.org/A5081957526)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于层间弹塑性固有应变的拓扑优化方法，用于在金属增材制造过程中最小化残余应力，并保证最终使用性能。

**💡 创新点**

创新点在于：
• 采用无激活应变的层间弹塑性固有应变分析模型，直接在每层步中携带前一步累积应力，实现应力连续性；
• 通过将应力和等效塑性应变作为状态变量，推导出一阶递推的伴随方程，逆向传播只需一次线性求解，计算量与正向分析相当；
• 结合密度方法、投影与滤波，构造可微的残余应力指标，并在完整使用性能约束下进行全局优化。

**🔧 技术方法**

所用技术包括：弹塑性固有应变分析（无激活应变）、层间递推状态更新、伴随敏感度分析（利用返回映射投影）、密度方法的投影-滤波策略、P-范数残余应力聚合、MMA型梯度更新。

**📊 数据集**

没有使用外部实验数据集；所有验证均基于自建的二维悬臂梁和三维MBB梁有限元模型，使用已知材料参数（铝合金）和设计独立的固有应变张量。

**📈 对比分析**

与传统仅考虑弹性残余应力或仅最小化平均刚度的参考设计相比，优化结果在残余应力聚合上下降约 8–10%（p=16），塑性应变区域减少约 70%，而最大残余应力几乎相同（受屈服面限制）。计算成本与前向层间分析线性相关，单步伴随求解仅需一次线性系统求解，显著优于传统的 N² 级别链式求导方法。

**⚠️ 局限性**

局限性包括：
• 需要对固有应变进行实验或高精度数值识别，误差会直接影响结果；
• 对于极大层数或高度非线性几何，仍可能出现收敛困难；
• 采用的投影-滤波参数对最终结构有显著影响，需经验调优；
• 仅在静态弹塑性框架下验证，未考虑热梯度、振动或多相材料的复杂效应。

---

## 5. AFID: A Unified Open Framework for Automated Fingermark Identification, Quality Assessment and Feature Extraction

**arXiv ID:** 2609.07439 | [PDF](https://arxiv.org/pdf/2609.07439v1)

**作者:** Tim Oblak `[一作]` (University of Ljubljana), Peter Peer `[通讯]` (University of Ljubljana)

**通讯引用:** 3272 | [OpenAlex ID](https://openalex.org/A5027730478)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了AFID统一的开放式指纹/指痕处理框架，集成识别、质量评估和特征提取，并在公开数据上训练，直接输出固定长度嵌入、质量分数以及分割/方向/细节特征。

**💡 创新点**

创新点包括：① 用单一共享编码器实现所有下游任务，消除传统管道的多阶段对齐；② 通过MagFace和背景正则化自监督生成与质量相关的嵌入范数；③ 结合嵌入范数与专家标签的融合质量模型；④ 仅用公共数据即可达到甚至超过商业匹配器的性能。

**🔧 技术方法**

主要技术为ConvNeXt‑T backbone、MagFace（带范数正则化）损失、强数据增强（尺度、旋转、颜色变换）、U‑Net式轻量解码器（分割、方向、细节）以及多阶段特征聚合的质量回归网络。

**📊 数据集**

使用的公开数据集包括：LFIW、IIIT‑D MOLF、NIST SD 300‑302、FVC 2000/2002/2004、NIST SD 27/302/303以及合成的Anguli、LFIW synth等；训练集约68k张样本，评估集覆盖传统与指痕任务。

**📈 对比分析**

与现有方法对比时，AFID在FVC验证中与开源顶尖方法竞争，且在SD 27/302/303的指痕识别中实现了新的最高rank‑1（≈70%），超过商业VeriFinger；质量评估在EDC曲线和nAUC上也优于LQMetric、pAFQA和VeriFinger自有质量分数。

**⚠️ 局限性**

局限性包括：分割解码器倾向于检测所有摩擦纹而非仅中心印记；质量模型仍需在不同捕获条件下进一步验证；以及对极小或高度失真印记的鲁棒性尚未完全覆盖。

---

## 6. Social Intuition vs. Machine Reasoning: Anticipating Human-Robot Interaction from multiple modalities

**arXiv ID:** 2609.07394 | [PDF](https://arxiv.org/pdf/2609.07394v1)

**作者:** Raphael Lorenzo-Louis `[一作]` (Inria, CNRS, UL, Loria, HUCEBOT), Serena Ivaldi `[通讯]` (Inria, CNRS, UL, Loria, HUCEBOT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了人类如何从服务机器人的视角预测一个人是否会互动，使用了仅姿态或完整视频输入，并对不同的轻量级姿态模型和最先进的视觉-语言模型进行了基准测试。

**💡 创新点**

创新点在于比较了人类和机器模型在互动预测任务中的表现，发现人类在使用完整视频时的表现显著优于机器模型，强调了社交直觉在预测互动中的重要性。

**🔧 技术方法**

使用了轻量级的姿态模型（如LSTM、ST-GCN和SkateFormer）和视觉-语言模型（如Qwen和Gemini系列），并进行了多种输入条件下的比较。

**📊 数据集**

使用了HUI360数据集，该数据集包含4310个样本，专注于人类与机器人之间的互动预测。

**📈 对比分析**

与人类标注者相比，轻量级姿态模型在仅使用姿态输入时表现稍逊，但在完整视频输入下，人类标注者的F1分数显著高于视觉-语言模型，表明人类在社交动态推理方面的优势。

**⚠️ 局限性**

限制在于当前模型在同时推理时间、空间和社交线索方面存在困难，且模型大小与性能之间没有明显的相关性。

---

## 7. Inferring Urban Mobility Interactions from Aggregated Dynamics

**arXiv ID:** 2609.07349 | [PDF](https://arxiv.org/pdf/2609.07349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 8. Monadic Second-Order Logic in HOL: Deep and Shallow with Automated Faithfulness (Extended Preprint)

**arXiv ID:** 2609.07345 | [PDF](https://arxiv.org/pdf/2609.07345v1)

**作者:** Christoph Benzmueller `[一作]`, Daniel Kirchner `[通讯]` (University of Bamberg)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5053333126)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了单调二阶逻辑（MSO）在 Isabelle/HOL 中的深浅嵌入，并给出了对应的忠实性（faithfulness）证明；同时开发了三种嵌入方式（深度嵌入、最大浅嵌入与最小浅嵌入），并对两排序的捕获避免替换与 α 重命名进行了机理化。

**💡 创新点**

创新点主要体现在：① 同时提供深浅两种嵌入的统一框架，能够在同一理论中实现互为映射并自动化忠实性证明；② 引入两排序（个体与集合）透明的替换机制，使得对 MSO 的两排序量化能够得到机理化的支持；③ 在最小浅嵌入中使用 locale 进行参数化，得到对标准阅读与一般阅读的精确区分，并通过两排序 Löwenheim–Skolem 定理实现了对一般（Henkin）阅读的机理化验证。

**🔧 技术方法**

使用的技术包括 Isabelle/HOL 证明助手、深浅嵌入技术、捕获避免替换与 α 重命名、局部化（locale）机制、自动化证明脚本、以及两排序 Löwenheim–Skolem 构造与可证明的替换引理。

**📊 数据集**

实验使用的并未采用传统机器学习数据集，而是使用经典的 MSO 基准公式（布尔闭包、图操作、可达性、2‑可着色等）和手工构造的对偶反例，全部在 Isabelle/HOL 内部完成。

**📈 对比分析**

通过在三种嵌入下对相同的 MSO 公式进行证明与反例搜索，比较标准阅读与一般阅读在有效性上的差异；性能方面并未给出数值度量，但实验展示了在最小浅嵌入中可直接使用 Isabelle 的自动求解器（如 `simp`, `sledgehammer`, `nitpick`）完成证明或反例生成。

**⚠️ 局限性**

限制主要在于：① 仅覆盖无等号、二元关系的 MSO 子语言；② 需要显式限制解释为元素子结构才能得到标准阅读的完全忠实性；③ 由于依赖于手工构造的示例，实验规模有限，未覆盖更复杂的多排序或更高阶的 MSO 形式。

---

## 9. TRAIL: Trajectory-Aware Visual Place Recognition against Unordered Databases

**arXiv ID:** 2609.07373 | [PDF](https://arxiv.org/pdf/2609.07373v1)

**作者:** Dominik A. Kloepfer `[一作]` (University of Oxford), Patrick Wenzel `[通讯]` (Helsing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 TRAIL 框架，通过条件随机场利用查询序列的时间上下文来提升在无序数据库上的视觉地点识别（VPR）性能。

**💡 创新点**

创新点在于：① 将顺序信息引入单图像 VPR 的后处理步骤；② 通过 CRF 模型结合视觉相似度与相机运动一致性，递归更新对数据库图像的概率分布；③ 兼容任意预训练的 VPR 基座，计算开销低。

**🔧 技术方法**

核心技术包括：条件随机场（CRF）递推、可微分的发射与转移潜能（使用 MLP 与 CNN 计算），以及基于 DINOv2 本地特征的转移描述子；训练采用二元交叉熵损失和预训练。

**📊 数据集**

使用 Mapillary Street-Level Sequences（MSLS）作为训练集，并在 MSLS、Nordland、4Seasons 三个多场景数据集上进行无序数据库评估，测试时不做 fine‑tune。

**📈 对比分析**

与 MegaLoc、FoL、SeqNet、JIST、CaseVPR、SelaVPR、PairVPR、手工规则等基线相比，TRAIL 在 T≥5 的 Recall@T 取得最高或接近最高的结果；在 MSLS T=10 达到 78.1%（比 MegaLoc 提升 8.3pp），Nordland、4Seasons 也有 9–10pp 的提升；相比昂贵的重排序方法，TRAIL 计算量和内存占用更低。

**⚠️ 局限性**

局限性：依赖预训练 VPR 基座，未对极端视觉变迁（如灰度/彩色差异）做专门适配；对大型数据库仍需额外存储本地特征；在极少特征区块中若前置检索误检率高，转移模型可能无效。

---

## 10. TASTE: Throughput-Aware Batch Size Tuning for On-Device Edge Learning

**arXiv ID:** 2609.07444 | [PDF](https://arxiv.org/pdf/2609.07444v1)

**作者:** Avik Bhatnagar `[一作]` (FZI Research Center for Information Technology, University of Tübingen), Oliver Bringmann `[通讯]` (FZI Research Center for Information Technology, University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种在资源受限的边缘CPU上进行深度学习模型训练的方法TASTE，利用贝叶斯优化自动调优批量大小以最大化训练吞吐量，同时保持模型精度。

**💡 创新点**

创新点在于：①将贝叶斯优化与梯度累积、线性学习率缩放相结合，在不牺牲精度的前提下寻找最优批量大小；②系统性评估了在监督学习与在线连续学习两种范式下，批量大小与吞吐量、精度之间的关系；③提出了针对边缘CPU的静态图编译与执行框架，兼顾内存与能耗。

**🔧 技术方法**

使用技术包括ONNX训练库、Apache TVM编译器、hyperopt贝叶斯优化（TPE算法）、梯度累积、线性学习率缩放，以及在C++/Python层面实现的边缘设备远程部署。

**📊 数据集**

实验使用的公共数据集为CIFAR‑10、CIFAR‑100和TinyImageNet，模型采用预训练的ResNet‑18。

**📈 对比分析**

通过与最大可支持批量大小（bs_max）以及参考批量大小的对比，TASTE在Raspberry Pi 4等多种边缘CPU上实现了至少1.5倍、最高约2倍的训练吞吐提升，同时在监督学习与连续学习任务中保持了与参考实现相近的测试准确率。

**⚠️ 局限性**

局限性包括：仅在CNN ResNet‑18 上验证，未涉及其他模型或专用加速器；采用静态批量大小而未考虑动态工作负载；实验仅在CPU上评估，未充分挖掘GPU/TPU等加速硬件潜力。

---

## 11. A Surrogate-based Approach for Fast Multi-objective Architectural Refactoring Optimization

**arXiv ID:** 2609.07389 | [PDF](https://arxiv.org/pdf/2609.07389v1)

**作者:** J. Andrés Diaz-Pace `[一作]`, Antonela Tommasel `[通讯]` (ISISTAN, CONICET-UNCPBA)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在软件架构优化中使用代理模型加速多目标进化搜索。

**💡 创新点**

创新在于将回归代理模型动态集成进GA评估，持续增量训练以降低昂贵的仿真调用。

**🔧 技术方法**

使用了XGBoost回归、NSGA-II进化算法、FastAPI服务器、UML建模与特征编码。

**📊 数据集**

采用CoCoME案例与AWS EC2实例的性能/能耗/成本数据。

**📈 对比分析**

将代理辅助NSGA-II与纯NSGA-II进行比较，代理模型在保持帕累托前沿质量的同时将计算时间缩短约30%。

**⚠️ 局限性**

主要局限包括特征选择不足、代理精度受限、实验只覆盖单一建模语言与有限的重构动作空间。

---

## 12. EnvPilot: Systematic Design and Evaluation of an Experience-Augmented Agent for Software Environment Setup

**arXiv ID:** 2609.07357 | [PDF](https://arxiv.org/pdf/2609.07357v1)

**作者:** Hanwu Chen `[一作]` (Shenzhen Technology University), Daoguang Zan `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了一种基于经验重用的自动化环境搭建代理EnvPilot，并构建了112个多语言GitHub实例的AES‑Bench基准。

**💡 创新点**

创新点在于将执行轨迹抽象为结构化<问题-解决-动作>经验记忆，并通过上下文感知检索将历史经验精准匹配到新任务，实现经验驱动的环境配置。

**🔧 技术方法**

技术包括大语言模型（DeepSeek V3、GPT‑4o‑mini）、多查询检索与互惠排名、结构化经验库、LLM‑as‑a‑Judge自动评估及容器化执行。

**📊 数据集**

使用的数据集为从多语言环境搭建轨迹抽取的667条经验记忆以及112个跨9种编程语言的真实GitHub实例组成的AES‑Bench。

**📈 对比分析**

与RepoLaunch、ExecutionAgent、Repo2Run、SWE‑Agent等基线比较，AES‑Bench上EnvPilot实现Pass@1 75.00%，显著高于最优基线约55%，且Token和API成本更低。

**⚠️ 局限性**

局限性包括对容器权限、系统级依赖、版本管理细节的敏感性，无法处理私有镜像或高度定制化的工业项目，且经验库覆盖有限，面对全新工具链时效果可能下降。

---

## 13. A Text Mining and Classification Approach for Analyzing Architecture Decision Records

**arXiv ID:** 2609.07375 | [PDF](https://arxiv.org/pdf/2609.07375v1)

**作者:** Nicolás Miccio Palermo `[一作]`, J. Andrés Diaz-Pace `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自动化的文本挖掘和分类方法，用于大规模分析架构决策记录（ADRs），并应用于从约550个开源项目中提取的ADRs数据集。

**💡 创新点**

创新点在于结合主题建模、基于大语言模型（LLM）的分类和模板合规性检查，系统性地分析ADRs的内容和结构。

**🔧 技术方法**

使用了主题建模和基于LLM的分类技术，构建了一个自动化分析管道。

**📊 数据集**

使用的数据集来自约550个开源项目，共计约4316个ADRs，数据集包含了项目的元数据和ADR文件位置。

**📈 对比分析**

通过与现有的决策分类法进行比较，发现ADRs主要记录存在、技术和过程相关的决策，但在替代方案、决策驱动因素和某些质量关注方面记录不足。性能方面，LLM分类的准确率在46%到66%之间，具体取决于分类的类型和提示策略。

**⚠️ 局限性**

限制在于分析主要集中在英语的开源GitHub项目，可能无法推广到非英语或私有软件项目。此外，ADRs的结构依赖于标题，可能会错过自定义或非正式格式的内容。

---

## 14. Think Wider: Mitigating Latent Rank Collapse in Implicit Chain-of-Thought Reasoning

**arXiv ID:** 2609.07406 | [PDF](https://arxiv.org/pdf/2609.07406v1)

**作者:** Yuwen Hao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种轻量级谱正则化器，用于抑制隐式推理中潜在状态的秩崩塌，提升推理轨迹多样性与准确率。

**💡 创新点**

创新点在于设计了一个只在训练阶段加入、无需改变推理流程的谱正则项，惩罚潜在轨迹中主导方向的投影，从而扩展潜在子空间。

**🔧 技术方法**

主要技术包括对潜在轨迹进行行归一化、通过求和得到主导方向近似、构造投影正则损失，并将其作为附加目标加入 CODI 与 SIM‑CoT 等隐式推理方法的训练。

**📊 数据集**

使用了四个推理基准：StrategyQA、CommonsenseQA、ASDiv‑Aug 与 AQuA，评估其在 LLaMA‑3.2‑3B‑Instruct 与 Qwen3‑1.7B 等开源模型上的表现。

**📈 对比分析**

实验将加入正则的模型与匹配的 CODI/SIM‑CoT 基线以及显式推理（no‑CoT、SFT‑CoT）对比，平均提升 1.8%–4.4% 的准确率，单项最高提升 6.8%，同时有效秩从约1.5升至约3.8。

**⚠️ 局限性**

局限性包括仅在 3B–8B 规模模型上验证，使用固定的 6 步潜在序列，未提升潜在状态的可解释性，并且对更大规模模型的效果尚未探测。

---

## 15. Better Late Than Never: Online Flow Time Scheduling with Online Estimates

**arXiv ID:** 2609.07402 | [PDF](https://arxiv.org/pdf/2609.07402v1)

**作者:** Anupam Gupta `[一作]` (New York University), Sorrachai Yingchareonthawornchai `[通讯]` (Institute for Theoretical Studies)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于多层反馈队列的Balanced MLF算法，用于在单机环境下的在线流时间调度问题中，当作业到达后只能在作业处理进度达到未知阈值后才收到作业规模的近似估计，且该阈值在作业实际规模的一定范围内随机出现。

**💡 创新点**

创新点主要体现在：
1) 同时放宽了SRPT所需的两大假设——作业规模完全未知且估计在作业完成前才揭示；
2) 引入“有效实例”与“有效类”概念，构造一套不依赖于作业真实规模的LP松弛，并用改进的双重拟合（dual‑fitting）技术进行分析；
3) 设计了精细的修剪（pruning）机制，剔除那些在有效类划分中造成不一致的作业，从而保证局部竞争比可被控制；
4) 证明了该算法在(ε, μ₁, μ₂)‑在线估计模型下的竞争比为O(μ₁μ₂/ε)，并给出了匹配的下界，证明此比值是渐进最优的。

**🔧 技术方法**

使用的技术包括：
- 多层反馈（MLF）调度策略的变体，结合“新鲜”与“正在执行”状态的动态切换；
- 对每个作业定义有效处理时间、有效类和有效剩余时间，构造有效实例；
- 通过改进的 knapsack‑cover LP 及其对偶，进行双重拟合分析；
- 设计修剪算法去除“异常”作业并维持类的平衡；
- 采用 Yao 原理与轰炸（bombardment）lemma 给出随机化下的下界。

**📊 数据集**

该工作属于理论分析范畴，没有使用实际数据集；所有实验都是在构造的理论输入（例如几何分布的随机作业、Adversarial 估计阈值）上进行的。

**📈 对比分析**

性能评估：
- 对于一般的(ε, μ₁, μ₂)‑在线估计模型，Balanced MLF 的竞争比为O(μ₁μ₂/ε)。在特殊情形ε=1/2、μ₁=μ₂=1时，得到O(1/ε²)=O(4) 的常数竞争；
- 与传统非clairvoyant算法（如Round‑Robin、SETF、MLF）相比，后者的竞争比至少是Ω(n^{1/3}) 或 Ω(log n)，该算法大幅提升到常数级；
- 通过构造的下界证明，该竞争比在渐进意义上是最优的。

**⚠️ 局限性**

局限性：
- 仅针对单机调度，未讨论多机或异构机环境；
- 虽然算法在参数上不需要 μ、ε，但分析依赖于这些参数的存在，若 μ、ε 极大时常数因子会变得很大；
- 该模型假设估计只在作业处理进度达到未知阈值时才揭示，实际应用中估计可能更频繁或更早；
- 对极端作业规模分布（p_max 远大于 1）的情况分析仍显繁琐，实际实现需进一步简化。

---

## 16. CRISP: Corneal Confocal Microscopy Real-Time Image Stitching Pipeline

**arXiv ID:** 2609.07336 | [PDF](https://arxiv.org/pdf/2609.07336v1)

**作者:** Qincheng Qiao `[一作]` (Shandong University), Xinguo Hou `[通讯]` (Shandong University)

**通讯引用:** 3503 | [OpenAlex ID](https://openalex.org/A5039626060)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了CRISP——一种针对传统 CCM 视频流的实时全景拼接框架，能够在扫描过程中提供实时覆盖反馈，并生成可用于离线精细拼接的帧、位姿和锚点信息。

**💡 创新点**

创新点在于：①通过焦点感知门控剔除模糊/无效帧；②结合局部配准、全局检索和稀疏锚点图实现在线低延迟拼接；③设计了跨子图重定位与合并策略，支持断断续续扫描后恢复；④实现了完整的开源、无硬件依赖的实时管线，可与离线工具无缝衔接。

**🔧 技术方法**

使用的技术包括：SuperPoint+LightGlue 进行局部特征提取与匹配；RANSAC 估计几何内点比例与覆盖率；DINOv2 ViT-S/14 提取全局图像描述子用于检索；基于二维平移的位姿传播；稀疏锚点网格实现低冗余地图。

**📊 数据集**

实验数据为一名受试者（Xinguo Hou）在 Heideteag Retina Tomograph III 上采集的约10分钟 CCM 视频，共数百帧；未使用公开标准数据集，而是现场真实采集。

**📈 对比分析**

与离线 HIT（HRT Imaging Tool）比较：CRISP 的在线拼接覆盖面积约 3.63 mm²（单帧视场的 22.68 倍），经过 HIT 细化后面积约 3.25 mm²（20.30 倍），显示出相当高的覆盖率与精度；实时反馈显著提高了扫描过程的可操作性。

**⚠️ 局限性**

局限性：仅在单一受试者上验证，缺乏跨设备、跨人群的广泛评估；使用的特征提取与匹配模型为通用视觉模型，未针对 SNP 图像进行专门训练，可能导致匹配误差；对高帧率、强手颤等极端条件的鲁棒性尚待进一步验证。

---

## 17. Unified Vision-Centric Pedestrian Crossing Action Prediction via Adaptive Patch Projection and Proactive Spatial Rectification

**arXiv ID:** 2609.07420 | [PDF](https://arxiv.org/pdf/2609.07420v1)

**作者:** Yao Tian `[一作]` (Chang'an University), Binglu Wang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 1825 | [OpenAlex ID](https://openalex.org/A5043220498)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 ViCross，一种基于多模态大语言模型的视觉中心化行人横穿预测框架，利用变量分辨率补丁映射（VRPM）聚焦目标行人，并通过空间约束增强策略（SCES）在训练阶段提升跨帧空间一致性。

**💡 创新点**

创新点包括：① 在仅依赖首帧目标框的视觉中心化设置下实现行人预测；② 通过 VRPM 对目标区域采用细粒度补丁、背景采用粗粒度补丁，实现高效且空间一致的特征编码；③ 在训练时引入多任务空间约束（定位、轨迹预测、跨帧运动补偿），实现对共享视觉表示的主动空间校正；④ 采用生成式大语言模型作为最终决策器，避免传统二分类头的局限。

**🔧 技术方法**

核心技术包括：多模态大语言模型 Qwen2‑VL‑7B + LoRA 微调；3D 卷积实现 VRPM 的粗细特征提取；内容感知融合与注意力加权；SCES 的四重损失（语言生成、分类、预测、定位）；以及全流程的端到端训练。

**📊 数据集**

实验使用了三个公开数据集：JAAD_beh、JAAD_all 以及 PIE，分别覆盖不同的场景复杂度与行人规模。

**📈 对比分析**

与传统视觉中心化基线（如 I3D、C3D、ConLSTM）以及多源融合方法（如 Faster‑PCPnet、PedGNN）对比，ViCross 在 JAAD_beh、JAAD_all 上分别达到 0.74、0.90 的准确率，PIE 上为 0.82，显示出在视觉中心化设置下的竞争力；在多源融合设置下虽略逊一筹，但显著降低了对帧级检测/追踪等外部感知模块的依赖，且在多次随机种子实验中表现稳定。

**⚠️ 局限性**

局限性包括：① 仍需在首帧手动或外部自动给定目标框，对初始化误差敏感；② 对遮挡、极小或远距离行人、拥挤场景下的空间一致性尚不足；③ 大语言模型推理成本高于轻量级模型；④ 在跨域（PIE）性能下降明显，尤其在召回和 F1 上受限。

---

## 18. 5GDescrambler: Locating, Descrambling, and Decoding 5G Scheduling Information (long version)

**arXiv ID:** 2609.07367 | [PDF](https://arxiv.org/pdf/2609.07367v1)

**作者:** Fritz Windisch `[一作]` (Karlsruhe Institute of Technology), Thorsten Strufe `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3616 | [OpenAlex ID](https://openalex.org/A5053465128)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种完全被动的 5G NR PDCCH 监听器 5GDescrambler，能够在不依赖任何侧信道或暴力破解的情况下逆向解密 DCI 并提取 UE 的 RNTI、调度信息与二进制 DCI。

**💡 创新点**

创新点包括：
- 利用 PDCCH 及 DCI 编码链的线性代数结构，构造可逆生成矩阵，实现一次性 O(1) 的解码而非 2^44 次暴力尝试；
- 自动检测 CORESET 与搜索空间配置，支持所有 5G 配置，无需预先获取；
- 通过主动 QPSK 组检测与相位估计，克服无已知参考符号的信道不确定性；
- 在实现中加入错误修正的线性方程求解，提升在低 SNR 下的可靠性。

**🔧 技术方法**

技术手段主要包括：
- 线性代数与矩阵乘法实现 DCI 编码与逆向解码；
- 极化码 (Polar) 译码与 CRC 处理；
- 线性方程组 (GF(2) 上高斯消元) 进行错误纠正；
- QPSK 组检测与相位估计算法；
- Rust 语言实现，完整的 5G NR 同步与资源网格处理。

**📊 数据集**

数据集：
- srsRAN 捕获样本（两个场景）
- OpenAirInterface5G 捕获样本（两种场景）
- 两个商业供应商（红箱）提供的实测样本（共四种场景）
- 通过公开的合成 5G RAN 捕获文件与配置示例。

**📈 对比分析**

对比方法与性能：
- 与 5GSniffer 在同一 srsRAN 样本下比较，未调参时 5GDescrambler 速度提升约 40 倍、DCI 检测率提升 17%；调参后提升约 300 倍；
- 在所有样本上，DCI 误码率 BLER 在 6.5 dB SNR 以下低于 1%；
- 处理时延在大多数样本中低于采样时长（≤48% CPU 负载），满足实时监听需求；
- 误码修正有效，尤其在 10 dB 以上 SNR 时 F1‑macro >99% 的 QPSK 组检测。

**⚠️ 局限性**

局限性：
- 在共搜索空间 (CSS) 下，180° 旋转导致 RNTI LSB 颠倒，需额外参考符号或 CSI‑RS 支持；
- 目前仅输出二进制 DCI，未实现完整的 DCI 内容解析与后续流量识别；
- 仅在实验室级样本中验证，对真实运营商高负载、不同频段、移动性及干扰环境的鲁棒性待进一步评估；
- pipeline 在配置搜索停滞时可能出现队列溢出，需进一步并行化；
- 要彻底阻止该攻击，需要在物理层上改动 3GPP 规范，替换线性 Scrambling。

---

## 19. Scanning the Harness: An Empirical Study of Supply-Chain Defects in AI Coding-Agent Configurations

**arXiv ID:** 2609.07360 | [PDF](https://arxiv.org/pdf/2609.07360v1)

**作者:** Benjamin Kapner `[一作]` (Red Hat), Hofni Gartner `[通讯]` (Red Hat)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AI编码代理的配置层（harness）进行静态分析，检测并验证供应链缺陷，量化安全和配置错误的普遍性。

**💡 创新点**

提出了一套多阶段验证流程（独立实现、模型仲裁、再次模型复核），并将检测结果公开，提供了可复现的测量方法。

**🔧 技术方法**

使用自研的Python静态分析工具、两份独立实现、Claude LLM仲裁和再次模型审阅，结合规则集对文件、文件系统、文件对比进行检查。

**📊 数据集**

基于3,171个公开GitHub仓库的AI代理配置集合，包括17.5%多助手配置和511个技能集合。

**📈 对比分析**

通过三步验证确保规则准确率，最终得到16.0%配置中存在安全缺陷、0.8%无法工作、2.4%规格不符；相较原始分析率25.5%，验证后率显著下降，体现了方法的高精度。

**⚠️ 局限性**

受限于仅检索公开仓库、规则仅能判定字节级别条件、无法衡量规则召回率，且验证依赖模型结果，缺乏人工评分。

---

## 20. Revisiting Thinning Methods for Kernel Learning Problems

**arXiv ID:** 2609.07432 | [PDF](https://arxiv.org/pdf/2609.07432v1)

**作者:** Blanca Cano-Camarero `[一作]` (Universidad Autónoma de Madrid), José R. Dorronsoro `[通讯]` (Universidad Autónoma de Madrid)

**通讯引用:** 2028 | [OpenAlex ID](https://openalex.org/A5031174137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了两种新的核子集抽样方法：Backward Kernel Herding（BKH）和 Flexible Kernel Thinning（FKT），并对其进行了理论分析与实证评估。

**💡 创新点**

创新点在于：BKH从完整数据集反向迭代删除样本，降低迭代次数并保持均值嵌入；FKT通过二进制展开实现任意大小的子集，突破传统Kernel Thinning只能按幂减的限制。

**🔧 技术方法**

采用的技术包括最大均值差异（MMD）目标、核嵌入理论、随机傅里叶特征（RFF）近似、支持向量机与高斯过程模型、以及自监督核构造。

**📊 数据集**

实验使用了 OpenML 开源基准套件（Classification 和 Regression），共 72 个分类和 35 个回归数据集。

**📈 对比分析**

与 Kernel Herding、Kernel Thinning 等传统方法比较，FKT 在大多数任务（尤其是分类和支持向量回归）取得最优预测性能；BKH 在训练时间和内存占用方面优于其他方法，且在 50%–75% 保留比例时性能相当。

**⚠️ 局限性**

主要局限是：在极端压缩比例（≤25%）下三种方法差异不大；BKH 对于 GP 回归的效果不如预期，可能需针对性优化；所有方法仍为贪婪式单点更新，可能导致全局最优不足。

---

## 21. Staying on the Attack Path: Structured State for Long-Horizon Automated Penetration Testing

**arXiv ID:** 2609.07344 | [PDF](https://arxiv.org/pdf/2609.07344v1)

**作者:** Weizhe Wang `[一作]` (Tianjin University), Guangquan Xu `[通讯]` (Tianjin University)

**通讯引用:** 3563 | [OpenAlex ID](https://openalex.org/A5040791046)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于意图图的自动化渗透测试代理，利用事实-意图有向无环图（Fact‑Intent DAG）外部化长周期状态；

**💡 创新点**

创新点在于：①用DAG将验证过的网络事实与可执行意图绑定，消除LLM上下文遗忘与意图漂移；②设计三层架构和两阶段退化恢复、维度自适应负载均衡提升执行稳定性；③构建五阶段意图检索与预测算法，提供结构化战术先导；

**🔧 技术方法**

核心技术包括LLM驱动的文本到图抽取、图数据库持久化、子图同构与模糊匹配检索、任务调度与负载均衡、容器隔离与心跳租约；

**📊 数据集**

使用真实CTF挑战集（覆盖10+ Web 漏洞类型，分易中难三级）并从公开报告与演练文档构建离线意图图；

**📈 对比分析**

在该基准上，所提方法整体成功率达88.2%，硬题成功率75.0%，相比基线 VulnBot 提升约44/50个百分点；平均成功任务回合数约7回，显著低于 Pentest‑R1、VulnBot、Claude Code 的20/23/12回；

**⚠️ 局限性**

局限性包括：依赖离线图的时效与覆盖范围，图匹配在大规模网络下的可扩展性未验证；仅在CTF环境评估，生产环境复杂性与防御机制差异需进一步研究；

---

## 22. AAS-RAIL: Improving Information Extraction for Asset Administration Shells through Retrieval-Augmented In-Context Learning

**arXiv ID:** 2609.07334 | [PDF](https://arxiv.org/pdf/2609.07334v1)

**作者:** Janek Groß `[一作]` (University of Applied Sciences Mainz), Jens Heidrich `[通讯]` (University of Applied Sciences Mainz)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5026996796)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种检索增强的LLM信息抽取框架AAS-RAIL，用于自动从技术数据表生成资产管理壳（AAS）。

**💡 创新点**

创新点在于使用检索得到的“提取助手”作为实例化上下文学习，动态适配公司特定的命名约定和格式，无需对模型进行微调。

**🔧 技术方法**

采用大型语言模型（GPT、Gemma、Qwen、Claude、Gemini等）结合语义检索、结构化知识图和语法约束解码技术。

**📊 数据集**

实验使用200份工业产品技术数据表与对应AAS（覆盖四家公司）以及40份用于构建检索数据库的样本。

**📈 对比分析**

与基线固定few-shot提示对比，RAIL在所有模型上提升30.4–52.4%的属性提取准确率，最高达到79.3%。

**⚠️ 局限性**

局限包括依赖已有AAS样本、对技术文档完整性和多语言适应性的不足，以及模型推理时延和检索覆盖率的限制。

---

## 23. PV-WM: A Heterogeneous Micro-Macro World Model for Articulated Pedestrian-Vehicle Co-Rollout

**arXiv ID:** 2609.07328 | [PDF](https://arxiv.org/pdf/2609.07328v1)

**作者:** Haozhuang Chi `[一作]` (Nanyang Technological University), Chen Lv `[通讯]` (Nanyang Technological University)

**通讯引用:** 20042 | [OpenAlex ID](https://openalex.org/A5072073374)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了PV-WM，一种历史序列仅依赖的世界模型，能够在同一递归周期内同时预测行人根运动与15关节关节运动、学习到的车辆刚体运动、车辆定向盒子以及行人-车辆几何关系。

**💡 创新点**

创新点在于将异质的行人关节动力学、车辆动力学与几何同步到同一时间步，采用生成状态的共轭回滚，利用持久的对偶状态与分类型终端动态，显著提升了预测精度并降低模型复杂度。

**🔧 技术方法**

技术主要包括多分类型编码器（行人根+关节、车辆GRU）、递归生成块、对偶关系再计算、关系监督损失、对偶-终端路由控制，以及在每一步重建定向盒子并同步几何。

**📊 数据集**

使用Waymo Open Dataset（包括Waymo-3DSkelMo的行人15关节轨迹与Waymo车辆轨迹/定向盒子），共计824个驾驶上下文，提取8个3秒窗口得到8,364个局部场景实例。

**📈 对比分析**

与一系列基线（TBIFormer、VehCondPose3D、T2P、Trajectron++、QCNet、MTR）以及验证选取的模块化专家组合比较，PV-WM在Root ADE、MPJPE、车辆ADE、P–V距离误差及最小接近误差上分别比专家系统提升5.2%、7.6%、11.9%和5.8%，同时参数量减少57.1%，平均FLOPs降低96.5%，p95延迟下降25.5%。

**⚠️ 局限性**

局限性包括：模型仍为单机单网络，需在更大规模或多模态数据上验证；对极端交互场景的鲁棒性未充分评估；以及对地图信息的利用有限，静态上下文的增强尚未显著提升性能。

---

## 24. Impact of canny edge detection preprocessing on performance of machine learning models for Parkinson's disease classification

**arXiv ID:** 2609.07408 | [PDF](https://arxiv.org/pdf/2609.07408v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 25. Multi-label versus multi-class classification of blood cells and their aggregates in microfluidic channels

**arXiv ID:** 2609.07410 | [PDF](https://arxiv.org/pdf/2609.07410v1)

**作者:** Igor Zingman `[一作]` (Max Planck Institute for the Science of Light), Jochen Guck `[通讯]` (Friedrich-Alexander University Erlangen-Nürnberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了多标签分类方法用于形变细胞计数器成像流式细胞术中识别单细胞和细胞聚集体，并与传统多分类方法对比。

**💡 创新点**

采用多标签分类避免了严格互斥聚集体标签的需求，能够识别未在训练集出现的聚集体，并通过共享特征学习类别关联。

**🔧 技术方法**

使用深度卷积神经网络（EfficientNet‑B0）与多头二分类器及WBC多分类子网络相结合，训练时采用平衡采样、加权交叉熵及多种图像增强。

**📊 数据集**

使用基于形变细胞计数器的血液图像数据，包含约122k训练样本、42k测试样本，并构建额外的基于聚类的WBC测试集。

**📈 对比分析**

通过在测试集上计算平衡准确率和敏感度进行比较，MC与ML在单细胞识别上相近（平均平衡准确率≈97%），ML在未见聚集体上明显优于MC，且在不同采集条件下的WBC测试集上表现更好。

**⚠️ 局限性**

对形变严重的聚集体若未在训练中出现则识别性能下降；聚集体标签不完整时模型需依赖阈值与约束，且荧光标记导致的形态差异仍是性能瓶颈。

---

## 26. DF26: We Cannot Tell Fake From Real Anymore

**arXiv ID:** 2609.07369 | [PDF](https://arxiv.org/pdf/2609.07369v1)

**作者:** Severyn Shykula `[一作]` (Ukrainian Catholic University), Anastasiia Mishchuk `[通讯]` (Institute of Software Systems of the National Academy of Sciences of Ukraine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了DF26 benchmark，专注于单人公开演讲场景的 AI 生成视频与真实视频的二分类检测。

**💡 创新点**

创新点在于：①构建受控的公开演讲三类情境（直视摄像机、官方声明、工作室访谈）并用语义提示精确匹配真实与合成视频；②通过多阶段过滤与 VLM 生成的提示，确保真实与合成视频在内容与风格上高度一致；③首次在交叉生成器环境下对检测模型进行统一评估，揭示现有方法在现代生成器上的显著失效。

**🔧 技术方法**

采用 Gemini‑2.5 VLM 进行情境提示生成，使用七款文本/图像到视频生成模型（如 Wan‑2.2、Hunyuan‑1.5、LTX‑2.3、Grok‑1.0、Veo‑3.1、Kling‑3.0、Wan‑2.6）生成合成视频；检测方面使用多种现有深伪检测器（DFD‑FCG、PwTF‑DVD、GenD‑PE 等）并用 AUROC、EER 进行性能评估。

**📊 数据集**

真实视频来自 OpenVid、TalkingCelebs 与 MAVOS‑DD 三大公开数据集；合成视频由上述七款生成器按统一语义提示生成，最终得到 271 条真实与 2,420 条合成样本。

**📈 对比分析**

在 held‑out 交叉生成器评估中，现有检测器在 CelebDF++ 上可达 90%+AUROC，但在 DF26 上仅 50%–70%，部分模型甚至低至 40%；人类观察者对 DF26 假视频的识别准确率仅 52.6%，几乎等于随机猜测。

**⚠️ 局限性**

局限性包括：数据规模仅 2,691 条样本；仅评估视觉模态，未涵盖音频同步与口型一致性；商业生成器仅用于评估而不支持训练，且对多样性与长视频处理的适用性待验证。

---

## 27. D3ARC: Time-Critical Distributed Disaster Detection for Asynchronous Cooperative Multi-Robot Systems

**arXiv ID:** 2609.07350 | [PDF](https://arxiv.org/pdf/2609.07350v1)

**作者:** Nikolaos Koursioumpas `[一作]` (National and Kapodistrian University of Athens), Ramin Khalili `[通讯]` (Huawei Heisenberg Research Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出了 D^3ARC 框架，实现多无人机与远程控制器在时间受限、通信不稳定的灾害检测任务中异步协同决策，目标是尽快且可靠地探测森林火灾。

**💡 创新点**

创新点在于：①层级异步分布式决策，RC 与机器人各自独立决策但共享奖励；②前瞻性神经回归模型让各代理预测其行动对最终奖励的影响；③四个专门机制（安全避障、覆盖效率、求援协同、提前完成）提升安全、覆盖率与效率；④在不确定环境下同时考虑感知、计算、通信与时间成本。

**🔧 技术方法**

使用深度前馈回归网络（RFNN）做前瞻评估；ROS2+Gazebo 11 仿真环境；多种 CNN 检测模型（Full CNN、Quantized CNN）；离线训练 115k 样本；采用随机采样候选动作与贪婪选择；引入基于区域历史的优先级惩罚。

**📊 数据集**

没有使用公开真实火灾数据集，而是基于 Blender 生成的 530x530x165 m 森林场景，放置 14 个火灾点并随机设置 2 个基站位置；机器人参数取自 3DR Iris 参考模型；通信模型为 2.4 GHz 单基站，考虑距离衰减与阴影波动。

**📈 对比分析**

通过 300 场仿真任务评估，比较单机、协作与无协作、规则基准 CH‑RBS。结果显示：多机协作提升任务成功率（最高 94%）、检测置信度（89.4%）与任务完成速度（平均 1.8 min，较基准快 22%）。与 CH‑RBS 比较，D^3ARC 的成功率提升 32% 以上，平均检测时间缩短 32%。

**⚠️ 局限性**

局限性包括：①实验仅在仿真环境下验证，真实场景的动态障碍、信号干扰与硬件噪声未充分考察；②模型依赖离线训练样本与超参数调优，迁移到不同任务或硬件平台可能需要重训练；③在极高拥塞或极低带宽环境下，前瞻模型误判可能导致过度通信或误判；④目前仅支持固定数量的 UAV 与单基站，扩展到大规模多基站或移动基站需进一步研究。

---

## 28. Fast simulation of nonlinear deep resistive networks for energy-based computation

**arXiv ID:** 2609.07356 | [PDF](https://arxiv.org/pdf/2609.07356v1)

**作者:** Filip Osana `[一作]` (Université Paris-Saclay), Damien Querlioz `[通讯]` (Université Paris-Saclay)

**通讯引用:** 9776 | [OpenAlex ID](https://openalex.org/A5063819347)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种针对深度电阻网络（DRN）的坐标下降（CD）求解器，支持实际单极性、双极性肖克利二极管以及分段线性（PWL）等真实单端非线性元件。

**💡 创新点**

创新点在于：①将CD方法从仅适用于理想二极管拓展到任意单调非线性元件；②针对不同非线性曲线推导了闭式Lambert‑W更新或高效的单变量根求解；③在保持层级并行结构的同时引入可调的双向放大器，使信号在深层网络中保持幅度。

**🔧 技术方法**

使用的技术包括：坐标下降（带超松弛）、Lambert‑W函数闭式求解、Newton–Raphson局部迭代、PWL插值求解、双向放大器的能量函数重写以及与SPICE（ELDO）对比的DC稳态仿真。

**📊 数据集**

实验数据集：scikit‑learn提供的Digits（360个验证样本）和标准MNIST（训练/测试集）。

**📈 对比分析**

通过与SPICE的稳态电压结果比较，测量相对L1误差（90%分位数<1e‑4）和运行时间；在不同宽度（64–1024）和深度（1–3隐藏层）下，CD实现了高达≈1750×的速度提升，同时保持极低误差。对MNIST网络（1568×100×20）训练，CD每轮耗时约1/440 SPICE，最终测试误差≈3%（比之前的理想二极管基线略优）。

**⚠️ 局限性**

局限性包括：仅考虑线性电导、理想源和单端单调非线性元件；不支持非单调或双端非线性、晶体管级模型；训练规模受限于当前实现；对更深网络和更复杂任务的鲁棒性尚待进一步验证。

---

## 29. RAFM-SER++: A Lightweight Multimodal Emotion Recognition Framework for Real-Time Behavioral Monitoring in Surveillance Systems

**arXiv ID:** 2609.07409 | [PDF](https://arxiv.org/pdf/2609.07409v1)

**作者:** Ngo Truong Dinh `[一作]` (Industrial University of HCM City), Phuc-Lu Le `[通讯]` (University of Science, VNU-HCM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级的多模态情感识别框架RAFM_SER++，旨在实时监控行为，特别适用于资源受限的监控系统。

**💡 创新点**

创新点在于引入了非对称残差注意力融合机制（RAFM），通过单向残差注意力路径将情感语音线索注入语义文本表示，减少了计算开销。

**🔧 技术方法**

使用了非对称残差注意力融合机制（RAFM）、BYOL启发的跨模态对齐目标和注意力引导的池化技术。

**📊 数据集**

在IEMOCAP和ESD两个基准数据集上进行了实验。

**📈 对比分析**

与HuBERT-Base基线相比，RAFM_SER++在准确性和效率上均表现优越，减少了60%以上的可训练参数，推理速度达到79.60 it/s，BACC得分为81.10%（IEMOCAP）和95.39%（ESD）。

**⚠️ 局限性**

限制在于当前实验是在基准数据集上进行的，实际监控环境可能涉及背景噪声、说话者重叠等复杂情况，未来需要在真实环境中进行评估。

---

## 30. Algorithms for Finite Group Epimorphism Testing

**arXiv ID:** 2609.07429 | [PDF](https://arxiv.org/pdf/2609.07429v1)

**作者:** Joshua A. Grochow `[一作]` (University of Colorado Boulder), Dhara Thakkar `[通讯]` (Nagoya University)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5074233683)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在有限群的 Cayley 表给定时的同态映射（epimorphism）判定问题，并给出了若干结构化群类的多项式时间判定算法。

**💡 创新点**

创新点在于将 epimorphism 检测问题归约到正常 Hall 子群及其补集的映射，再利用表示论与代码同构（Code Isomorphism）等技术，首次实现了在以下几类群的多项式时间解决：① 带有阿贝尔正则 Hall 子群且补为循环群的群；② 正则 Hall 子群为直积的阿贝尔群、补为阿贝尔群的群；③ 组没有阿贝尔分解因子（Fitting‑free）或仅允许有限阿贝尔宽度的群。

**🔧 技术方法**

主要技术包括：群结构分解（Hall 子群、补集、合成系列、首项系列），表示论（模同构、不可约分解、马塞克定理）、线性代数（标准形、匹配算法）、代码同构问题的多项式算法，以及在子模层面构造二分图并求最大匹配。

**📊 数据集**

该工作纯粹为理论分析，未使用实验数据集，而是针对所有满足结构约束的有限群给出算法与复杂度证明。

**📈 对比分析**

与以往仅给出决策性或指数级别算法的研究相比，本文在上述群类上实现了确定性多项式时间算法；具体复杂度可写为 O(|G|^{c})（c 为常数），并在证明过程中给出了构造式的 epimorphism。

**⚠️ 局限性**

局限性包括：未覆盖所有有限群，仅限于已知结构化群类；对 Fitting‑free 群的完整多项式解仍为开放问题；当阿贝尔宽度较大时算法需要穷举子模导致多项式上界难以保证；此外，研究未涉及实验验证与性能评估。

---

## 31. Federated Binary Gating with Server-Side Vision-Language Inference for Surveillance Anomaly Classification

**arXiv ID:** 2609.07403 | [PDF](https://arxiv.org/pdf/2609.07403v1)

**作者:** Côme-Alexis Puech `[一作]` (ESIEA), Rachid Chelouah `[通讯]` (CY Cergy Paris University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在联邦学习环境下，提出轻量化二分类CNN门控与服务器端零样本VLM的两阶段架构，实现隐私保护的视频异常分类。

**💡 创新点**

将联邦二分类屏蔽与中心化零样本VLM分离，既减小了边缘设备负担，又在非IID数据下保持了可用的多类别性能；并在真实三节点异构部署中验证了可行性。

**🔧 技术方法**

LiteCNN3D轻量化二分类网络、Qwen3-VL-8B零样本VLM、Flower联邦框架、FedAvg聚合、温度标定与基于熵的敏感度路由。

**📊 数据集**

UCF-Crime视频数据集，按五个宏观类别重新分组后用于二分类门控与四类异常元类别。

**📈 对比分析**

对比中心化CNN、联邦CNN、中心化多类CNN、联邦多类CNN、独立VLM以及中心化与联邦两阶段混合系统；联邦混合方案在保持与中心化混合相近的F1-macro（0.503 vs 0.506）的同时，视频传输比例从79.7%降至51.4%，而直接联邦多类学习性能崩溃。

**⚠️ 局限性**

仅在单个Dirichlet分区、有限的路由阈值和FedAvg训练下评估，未探究多分区、多样本与更强联邦优化器的泛化；VLM和门控的概率校准与阈值选择仍为经验性。

---

## 32. Human-like moral judgments conceal divergent motive attributions in large language models

**arXiv ID:** 2609.07353 | [PDF](https://arxiv.org/pdf/2609.07353v1)

**作者:** Xiaoyan Wu `[一作]` (University of Zurich), Jean-Claude Dreher `[通讯]` (Institut des Sciences Cognitives Marc Jeannerod CNRS UMR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对五种大型语言模型（LLM）与两组人类样本在关于举报人道德判断与动机归因的实验结果进行对比，检验模型是否能复制人类在平均水平及心理结构上的相似性。

**💡 创新点**

创新点在于将焦点从仅比较平均分数扩展到评估模型与人类在动机归因与道德判断之间的相关关系，并通过等价检验考察模型对情境细节与人群特征的敏感性。

**🔧 技术方法**

采用提示工程、结构化JSON输出、模拟人群特征（年龄、性别、学生身份）等技术，对五个开放与封闭权重的LLM（DeepSeek‑V3、GPT‑4o、Gemini‑2.5‑Pro、Llama‑3.1‑8B‑Instruct、gpt‑oss‑120b）进行多次查询。

**📊 数据集**

使用Brotzeller等人公开的两个基准人类样本（N=125、N=742）以及对应的情境材料，并在此基础上生成LLM的模拟响应。

**📈 对比分析**

比较方法包括：1）条件均值的皮尔逊相关和均方根偏差；2）多元回归检验动机与道德判断的交互作用；3）等价检验（TOST）评估模型在不同情境规格下的差异。结果显示，LLM在整体排序上与人类高度相关，但在动机归因（如竞争性动机与道德判断的关联）以及对情境细节的反应上显著偏离，且模型对人群特征的敏感性极低。

**⚠️ 局限性**

局限性包括：仅使用单一英文举报情境、未对模型进行预注册、两组人类样本的差异未随机化且混杂多因素、模型对情境与人群特征的等价检验受限、LLM响应集中在量表上限、缺乏跨文化和多语言验证、以及未探究模型内部推理机制。

---

## 33. Photonic reservoir computing with dimensionally compressed readout

**arXiv ID:** 2609.07418 | [PDF](https://arxiv.org/pdf/2609.07418v1)

**作者:** Gerald Kobi `[一作]`, Damien Rontani `[通讯]` (Université de Lorraine)

**通讯引用:** 1918 | [OpenAlex ID](https://openalex.org/A5067673382)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在光子储层计算中读取层尺寸受限的硬件瓶颈，提出通过随机投影（Random Projection）压缩高维状态到低维输出，比较压缩前后的时间延迟储层性能。

**💡 创新点**

创新点在于：①首次系统评估随机投影对读出层尺寸受限的光子储层的内在处理能力影响；②揭示在相同输出维度下，压缩后储层可获得更高的线性记忆容量（MC）但伴随非线性容量（NLC）的损失；③在NARMA‑10任务上发现存在“甜点区”——压缩比例合适时压缩后储层显著优于同尺寸独立储层。

**🔧 技术方法**

主要技术包括：时间延迟光子储层（Ikeda 非线性模型）、Johnson–Lindenstrauss 随机投影、记忆容量（MC）与信息处理容量（IPC）指标评估，以及对输入缩放、延迟同步/异步等超参数的全量扫描。

**📊 数据集**

使用数据集：NARMA‑10 合成时序任务（输入均匀分布 U(0,0.5)）。

**📈 对比分析**

比较方法：对同一输出维度（O）下的 (O) 与 (N,O) 两种配置分别计算 MC、IPC（按多项式阶数划分）和 NRMSE。结果显示：① (N,O) 在大多数超参数配置下 MC ≥ (O)，IPC 总量相等但低阶容量占比提升；② 在 NARMA‑10 任务中，(N,O) 的 NRMSE 在压缩程度适中时明显低于 (O)，尤其在 O≤40 时优势最显著。

**⚠️ 局限性**

局限性：①随机投影无法产生新的总容量，受限于输出维度的上界；②线性与非线性容量存在严格的权衡，过度压缩会削弱高阶非线性处理；③压缩效果高度依赖源储层尺寸与超参数（如输入缩放、延迟同步）是否处于最优区间；④实验仅验证了光子 Ikeda 模型，推广至其他物理实现需进一步验证。

---

## 34. TabBench-Bio: A Living Benchmark for Machine Learning on High-Dimensional Biomedical Tables

**arXiv ID:** 2609.07441 | [PDF](https://arxiv.org/pdf/2609.07441v1)

**作者:** Jules Kreuer `[一作]`, Nico Pfeifer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个名为TabBench-Bio的生物医学数据集基准，旨在比较不同机器学习模型在高维低样本量（HDLSS）环境下的表现。

**💡 创新点**

创新点在于创建了一个动态的、交互式的生物医学数据集基准，涵盖多个领域，并在共享的交叉验证协议下进行比较，强调了模型在不同特征和样本预算下的表现。

**🔧 技术方法**

使用了经典估计器、神经网络和表格基础模型（TFMs），并通过AutoGluon框架进行配置和比较。

**📊 数据集**

使用了来自多个来源的生物医学数据集，包括OpenML、MGnify、TCGA、GEO等，共计包含分类和回归数据集。

**📈 对比分析**

通过共享的交叉验证折和特征限制，比较了不同模型的表现，结果显示TFMs通常占据领先地位，但最佳配置依赖于操作点和生物医学模式。

**⚠️ 局限性**

限制在于基准目前以分类数据集为主，且可能遗漏稀疏生物信号，比较的是默认行为而非经过调优后的性能。

---

## 35. OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining

**arXiv ID:** 2609.07398 | [PDF](https://arxiv.org/pdf/2609.07398v1)

**作者:** Yuran Wang `[一作]` (National University of Singapore), Hang Zhao `[通讯]` (Tsinghua University)

**通讯引用:** 15898 | [OpenAlex ID](https://openalex.org/A5101826600)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了OpenWAM，一个开放的世界-动作模型研究框架，旨在系统地开发和评估世界-动作模型。

**💡 创新点**

通过模块化设计，OpenWAM-Infra将世界-动作建模分解为可组合的组件，促进了不同设计的控制比较和实验。

**🔧 技术方法**

使用了生成模型作为基础，结合视觉编码器和动作生成模块，通过联合自注意力机制实现世界和动作学习的协同。

**📊 数据集**

在518.5M帧的自我中心和机器人数据集上进行预训练，涵盖了多种机器人任务和场景。

**📈 对比分析**

与现有的单一系统相比，OpenWAM在多个基准测试中表现出色，尤其是在模拟和真实机器人实验中，展示了强大的泛化能力和高效性。

**⚠️ 局限性**

模型在特定任务上表现不佳，尤其是在单臂操作的某些基准测试中，可能是由于训练数据的多样性不足。

---

## 36. BlueprintAgent: Constraint-Triggered Targeted Revisits for Simulation-Ready Generation from Scanned Structural Blueprints

**arXiv ID:** 2609.07362 | [PDF](https://arxiv.org/pdf/2609.07362v1)

**作者:** Zhouyuan Xu `[一作]` (Tsinghua University), Chen Wang `[通讯]` (Tsinghua University)

**通讯引用:** 50788 | [OpenAlex ID](https://openalex.org/A5100337500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 BlueprintAgent，一种基于多模态大型语言模型和约束触发的专家系统，能够从扫描的钢筋混凝土结构蓝图中自动提取可直接导出为 FEM 的 3D 框架模型。

**💡 创新点**

将工程约束作为可调用验证器，生成结构化冲突记录并触发局部证据重新读取，实现“约束触发重访”机制，而非仅后期过滤；同时利用 LLM 作为主读者与多工具协同完成跨页、跨层的一致性校验。

**🔧 技术方法**

多模态 LLM（GPT‑5.4/Claude 4.6/Qwen 3.6）+ OCR+计算机视觉+轴线裁剪+表格读取+结构约束验证器+工具调用循环（类似 ReAct/Reflexion）+OpenSees FEM 导出。

**📊 数据集**

300 张真实扫描蓝图（20 份 RC 框架项目，共 102 层），包含梁柱平面图、立面、剖面、标题块、材料表、楼层高度表等；提供对应手工标注的 JSON 与 OpenSees 模型。

**📈 对比分析**

与 5 个基线（纯 OCR 规则、单 LLM 零射、固定管线、无约束反馈的 Agent、完整 BPA）以及 6 个消融实验对比；在所有 20 项目上，BPA 的轴、柱、梁、绑定、JSON 合法性、结构模型完整度均达 1.0，Beam F1 从 0.301 提升至 0.994，显示显著性能提升。

**⚠️ 局限性**

仅评估 RC 框架，未覆盖钢墙、钢结构、混凝土墙等；需要大量人工标注；对复杂层级、动态分析、非线性等高级 FEM 未覆盖；工具调用成本与时延未量化；模型依赖特定 LLM API，迁移性有限。

---

## 37. Uncertainty Quantification for LLM Agents: A Taxonomy, an Evaluation Protocol, and an Empirical Study

**arXiv ID:** 2609.07395 | [PDF](https://arxiv.org/pdf/2609.07395v1)

**作者:** Moule Lin `[一作]` (Trinity College Dublin), Goetz Botterweck `[通讯]` (Trinity College Dublin)

**通讯引用:** 1838 | [OpenAlex ID](https://openalex.org/A5031243998)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了针对LLM代理不确定性量化的三轴分类体系，整理并标注了120篇核心论文，提出了轨迹级校准指标TC‑ECE及评估协议，并在四个LLM模型与三类任务上进行实证分析。

**💡 创新点**

创新点包括：①三轴分类法（来源–方法族–管线阶段）实现对不确定性来源与使用方式的细粒度划分；②正式证明单步校准无法直接复合为轨迹校准并给出误差上界；③提出轨迹级ECE（TC‑ECE）和对应的评估流程；④基于此制定十项研究议题。

**🔧 技术方法**

技术上使用了校准理论、贝叶斯/集成推理、对抗式推断、对齐学习、可解释性方法等，结合标准指标（ECE、AUROC、Brier等）和自定义的TC‑ECE，实测四大LLM（如GPT‑4、Claude等）在多步推理、检索增强、代码生成等场景下的表现。

**📊 数据集**

实验数据来源于自研的\tau^2‑bench（多步问答）、ALFWorld（交互式环境）以及检索增强生成基准（如WebQA、OpenBookQA）等三类任务。

**📈 对比分析**

与传统单步校准基线相比，TC‑ECE显示轨迹级不确定性更为保守且常出现过度自信，四模型在三任务中的轨迹可靠性均低于单步校准的平均水平；实验表明当前方法尚未能系统提升轨迹级可靠性。

**⚠️ 局限性**

局限性包括：①缺乏充分的步级标签导致校准难度大；②单步校准与轨迹校准的分离使方法难以迁移；③实验环境仍相对理想，未涵盖真实世界噪声与长期交互；④分类法中稀疏细胞主要因术语不统一而非技术缺失；⑤对多代理协调的不确定性研究不足。

---

## 38. CALM: Configuration-Aware Human Intervention Boundaries During Robot Approach

**arXiv ID:** 2609.07430 | [PDF](https://arxiv.org/pdf/2609.07430v1)

**作者:** Xinting Gao `[一作]` (Tsinghua University), Weimin Zhuang `[通讯]` (Tsinghua University)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5104182618)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在一项受控实验中，研究者让41名受试者在两种室内空间里，机器人以不同的四种臂姿态（0、25、35、45 cm 前伸）沿直线路径接近受试者，测量受试者的停止距离、主观舒适度以及眼动指标，并基于停止距离分布构建了配置感知的干预边界模型 CALM。

**💡 创新点**

创新点包括：①首次系统量化机器人体态（臂伸展）对人类干预边界的影响；②提出 CALM（Configuration‑Aware Limit Model），将停止距离分位数转化为配置依赖的群体覆盖边界；③通过示例规划分析展示在不放宽干预概率约束的前提下，机器人通过臂部重新配置可恢复更近目标的可行性。

**🔧 技术方法**

主要技术手段有：受控实验设计与在参与者内重复测量；线性混合效应模型分析停止距离、舒适度与眼动数据；分位数估计与引导重采样构建 CALM 边界；以及基于 1‑D 规划的模拟评估重新配置策略。

**📊 数据集**

使用的数据集为实验生成的数据：41 名受试者 × 8 条件 = 328 条停止距离记录，配合主观问卷与眼动追踪（峰值瞳孔直径、瞳孔面积方差、平均注视时长）。

**📈 对比分析**

通过与固定距离（0.88 m）规划策略比较，CALM‑reconfig 策略在 1.10 m 目标下成功率从 17.6% 提升至 58.8%，且在多达 17 个不同目标距离的规划中，重新配置显著提高完成率。整体性能表明配置感知可显著改善机器人靠近目标的可行性。

**⚠️ 局限性**

局限性包括：仅测试了单一机器人、单速、静态臂姿态和两种空间尺度；受试者主要为大学生且样本规模有限；停止距离受按钮操作、系统延迟、定位误差等因素影响；实验环境与真实人机交互情境差异大，未考虑动态人、不同方向或携带负载等因素。

---

## 39. A Systematic Analysis of Automatic Differentiation versus Discretization-based Constraints for Physics-Informed PDE Solvers

**arXiv ID:** 2609.07437 | [PDF](https://arxiv.org/pdf/2609.07437v1)

**作者:** Xing Guo `[一作]` (Sun Yat-sen University), Feng Liu `[通讯]` (National Key Laboratory of Aerospace Physics in Fluids)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比自动微分（AD）与离散化约束两种物理信息神经网络（PINN）范式，在不同非线性程度和边界复杂度的 PDE 题目上进行系统实验，探讨网络架构（MLP vs GNN）与约束形式对精度、效率和鲁棒性的影响。

**💡 创新点**

提供基于误差分解（近似误差、优化误差、截断误差）的定量评估框架，揭示非线性增大时 AD 失效、离散化方法优先的规律，并验证 GNN 在强非线性下的稳健性。

**🔧 技术方法**

使用自动微分、有限差分、有限体积、有限元等离散化手段；多层感知机（MLP）与图神经网络（GNN）两类网络；Adam 优化器、动态学习率衰减等训练技巧。

**📊 数据集**

多种二维静态 PDE：线性 Poisson、非线性多项式 Poisson、Liouville 方程；不可压 Navier–Stokes（灯笼 cavity、backward‑facing step）；激波抛物面（高超声速无粘圆柱）等，配合对应的高精度 DNS 参考数据。

**📈 对比分析**

比较方法包括：全硬约束、软约束、不同阶数离散化、不同网络架构；通过全局 L₂ 误差、局部物理量分布、损失曲线、墙面/冲击线特征等多维指标评估。结果显示：在弱非线性、可全硬约束时 AD+MLP 精度最高；非线性加剧时离散化+GNN 逐渐优于 AD；高阶离散化在强非线性下易失稳；GNN 在复杂边界和强非线性场景下表现更稳健。

**⚠️ 局限性**

局限性包括：仅针对二维静态 PDE，缺乏对时间依赖、三维或多物理耦合问题的验证；缺乏自动诊断指标，仍需后验参考解；训练过程对超参数、边界约束形式高度敏感，难以通用；高阶离散化与 GNN 的计算开销在大规模网格下可能超过 AD+MLP。

---

## 40. CIT-CAD: Constraint Intent Tree-based CAD Code Generation and Verification

**arXiv ID:** 2609.07434 | [PDF](https://arxiv.org/pdf/2609.07434v1)

**作者:** Yali Du `[一作]` (Nanjing University), Ming Li `[通讯]` (Nanjing University)

**通讯引用:** 49322 | [OpenAlex ID](https://openalex.org/A5100452145)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 CIT-CAD 框架：先用 LLM 从自然语言描述推断 Constraint Intent Tree（CIT），再以 CIT 为条件生成 CadQuery 代码，随后通过确定性约束检查与局部反馈实现程序验证与修复。

**💡 创新点**

创新点在于将设计意图显式化为树形结构，利用结构化约束（节点属性与节点间几何关系）实现构造级别的检查与局部定位，且通过迭代修复循环保持已满足约束的递增性。

**🔧 技术方法**

使用技术包括：多模型 LLM（Qwen3、DeepSeek、GPT‑5.4‑mini）用于 CIT 推断与代码生成；静态 AST 与运行时几何分析提取实体与关系约束；确定性验证器与 Constraint Satisfaction Rate（CSR）评估；局部反馈驱动的修复循环。

**📊 数据集**

采用 Text2CAD 数据集（约26,783 条多实体样本，过滤单实体案例），从每条样本的自然语言描述推断 CIT，并使用对应的参考 CadQuery 程序进行几何级别评估。

**📈 对比分析**

与直接 LLM 提示生成（Vanilla）对比，CIT-CAD 在 VSR、IoU、CSR 上均有提升；在多实体且复杂程度高的样本中 CSR 提升至约 28%，VSR 与 IoU 也分别提升 3–19%，尤其在 Qwen3 与 DeepSeek 上表现最为显著；修复循环进一步提升 CSR 并缩小约束违规率。

**⚠️ 局限性**

局限性包括：对细粒度草图约束（Sketch_type、Is_connected 等）的识别仍不足；关系约束错误率虽低但仍存在；依赖 LLM 对 CIT 的推断准确性，若树错误会影响后续生成；当前约束词汇表有限，难以覆盖更复杂的几何关系和参数化细节。

---

## 41. RelightFormer: Feed-forward Generative Transformer for Multiview Object Relighting

**arXiv ID:** 2609.07414 | [PDF](https://arxiv.org/pdf/2609.07414v1)

**作者:** Hejun Wang `[一作]` (Shenzhen Research Institute), Bo Yang `[通讯]` (Shenzhen Research Institute)

**通讯引用:** 70858 | [OpenAlex ID](https://openalex.org/A5108047889)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种面向单视角和多视角图像的无显式逆渲染、全前向生成 Transformer，用于高质量图像重照明。

**💡 创新点**

创新点包括：① 设计了跨注意力的“latent illumination module”，将目标环境图直接注入特征；② 引入了排列不变的视角编码（PRope），消除多视角输入顺序偏差；③ 在大规模多视角重照明数据集上进行端到端训练，避免逐场景优化。

**🔧 技术方法**

技术主要包括：Transformer架构（基于Wan2.1视频生成模型）、变分自编码器、跨注意力、旋转位置编码、光照条件的离散化编码，以及对视角与光照的ray投影。

**📊 数据集**

使用自建的Laval Objaverse Dataset（LOD）：90K 3D模型、39K 环境光照，包含多视角训练/验证/测试集。

**📈 对比分析**

与传统逆渲染（LightSwitch、Reli3D）及生成式方法（DilightNet、Neural Gaffer）在单视角、多视角及新视角重照明任务上对比，取得 PSNR/SSIM/LPIPS 领先；在真实世界数据（OLATverse、Stanford-ORB）上实现零样本通用性，性能接近或优于优化式逆渲染方法，推理时间仅几分钟。

**⚠️ 局限性**

局限性：对极端材质（高光、半透明、毛绒等）仍有细节损失；需要海量训练数据，缺少对动态视频重照明的支持；模型不具备显式几何重建，可能在极端视角下产生几何不一致。

---

## 42. Det-5G: Closing the Determinism Gap in 5G-Advanced for Industrial Closed-Loop Control

**arXiv ID:** 2609.07386 | [PDF](https://arxiv.org/pdf/2609.07386v1)

**作者:** Adnan Aijaz `[一作]` `[通讯]` (Toshiba Europe Ltd), Adnan Aijaz (Toshiba Europe Ltd)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种Deterministic-5G的周期级资源分配框架，将闭环控制命令与反馈视为单一耦合事务，实现自包含的下行/上行调度、主动冗余捆绑、组下行和多用户上行打包。

**💡 创新点**

创新点在于周期级可靠性控制与自适应捆绑、周期预留与循环动态的结合，以及多设备组下行和多用户上行的联合打包，显著降低恢复引起的时延波动。

**🔧 技术方法**

采用5G NR现有机制：下行/上行协同调度、主动冗余捆绑、组下行（G-downlink）、SBT/NSBT多用户上行打包，并通过闭式分析与Monte Carlo调度仿真验证。

**📊 数据集**

使用基于3.8 GHz/40 MHz载波、30 kHz子载波间隔、7符号SBT的仿真设置；BLER三种状态概率（0.50, 0.35, 0.15）以及10⁵次周期的Monte Carlo实验，没有公开数据集。

**📈 对比分析**

与动态授权、SPS/CG和固定重复基线对比，采用无偏95%置信区间的闭式分析和Monte Carlo仿真。结果显示：在4 ms目标下Det-5G平均周期2.15 ms、失败率4.7×10⁻⁶，低于动态授权5.6×10⁻⁴；在16台多设备时99th百分位周期3.43 ms，优于动态5.25 ms；在1 ms全周期、10⁻⁶失败目标下NSBT方案提供最低时延。

**⚠️ 局限性**

局限性包括：仍需在实际5G-Advanced/6G硬件上验证；组下行受最弱设备捆绑限制；多用户上行资源分配复杂，需动态组建；未深入探讨跨层同步与TSN集成；在极高动态场景下捆绑更新频率仍受限。

---

## 43. Beyond Single-Negative Preference: Multi-Negative DPO for LLM-Centric Historical Entity Linking

**arXiv ID:** 2609.07379 | [PDF](https://arxiv.org/pdf/2609.07379v1)

**作者:** Tien Nam Nguyen `[一作]` (University of La Rochelle), Antoine Doucet `[通讯]` (University of La Rochelle)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于大语言模型的检索增强历史命名实体链接框架，利用LLM生成候选实体并进行验证，随后采用多负样本DPO进行偏好学习来选择最终实体；

**💡 创新点**

创新点包括：①将LLM用于检索阶段以提升候选召回率；②设计多负样本DPO（Multi-Negative DPO）在每条实例内对金标准实体与所有负样本进行pairwise学习，充分利用候选集信息；③结合简单提示、分块上下文与别名检索进一步提升检索效果；

**🔧 技术方法**

技术主要包括：大语言模型（GPT-20B/120B、Qwen3、Gemma等）进行候选生成；结构化提示与函数调用实现候选验证；LoRA微调的LLM作为选择器；多负样本DPO优化目标；

**📊 数据集**

数据集为两大多语言历史报刊基准：Hipe-2020（德法英）和HEPI-2023（法德芬兰瑞典），包含数万条命名实体，标注于Wikidata；

**📈 对比分析**

与传统检索+排序系统（SBB、L3i、MELHISSA、BELA、MHEL-LLaMo）以及自身的SFT、单负DPO、LLM提示等方法比较。实验表明Multi-DPO在两大基准上均实现了显著提升，微F1平均提升约4–6个百分点，尤其在NIL、语义歧义、OCR噪声和历史难名词上效果突出；

**⚠️ 局限性**

限制包括：①对NIL预测仍易误判，LLM倾向生成语义合理但不在KB中的实体；②检索阶段仍为瓶颈，未检索到正确实体则无法恢复；③未探索列表化或对比性目标，无法证明MDPO优越性；④候选生成阶段成本高，推理延迟较大。

---

## 44. Beyond Fluent Generation: A CPU Reliability Benchmark for MCP-Style Tool Calling in Sub-2B Small Language Models for Edge Deployment

**arXiv ID:** 2609.07370 | [PDF](https://arxiv.org/pdf/2609.07370v1)

**作者:** Abrar Shahriar Qurat-Ul-Ain Mastoi `[一作]` `[通讯]`, Abrar Shahriar Qurat-Ul-Ain Mastoi

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在CPU平台上对五个小于2B参数的语言模型进行基准实验，评估其在MCP风格工具调用中的准确性、JSON可恢复性以及CPU资源占用。

**💡 创新点**

创新点在于提出可恢复JSON解析与分阶段评估方法，并给出了平台无关的CPU基线，同时比较了贪婪与采样解码对工具调用成功率的影响。

**🔧 技术方法**

采用HuggingFace Transformers在CPU上以FP32推理，使用greedy和top‑p=0.9采样生成结果，利用自定义JSON恢复解析、Wilson置信区间、McNemar检验以及RSS与平均推理时间测量技术进行评估。

**📊 数据集**

使用100个英文提示，均匀分布在5种工具（各20组参数），共生成1,000个模型响应，作为实验数据集。

**📈 对比分析**

通过恢复成功率、工具名称与参数完整性检查、严格JSON解析率以及按工具类别的成功率来比较；结果显示Qwen2.5‑1.5B采样模式达到最高79%恢复成功率，Qwen2.5‑0.5B贪婪模式72%，但资源消耗最高。

**⚠️ 局限性**

局限性包括未在真实边缘硬件上测试、缺乏安全与对抗性场景、采样仅单次、未记录硬件/软件版本、解析仅使用贪婪提取，并且未涵盖多步交互、工具发现等更复杂的使用情形。

---

## 45. From Explicit References to Scene Manifolds: Distributional Fidelity and Realism for Radiance Field Quality Assessment

**arXiv ID:** 2609.07346 | [PDF](https://arxiv.org/pdf/2609.07346v1)

**作者:** Saeed Mahmoudpour `[一作]` (Vrije Universiteit Brussel), Peter Schelkens `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 6036 | [OpenAlex ID](https://openalex.org/A5034083500)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SCODA的轻量级场景条件目标质量评估方法，旨在通过场景流形建模来评估渲染视图的感知质量，而不是依赖于显式的图像对比。

**💡 创新点**

SCODA的创新点在于将质量评估从显式的图像对比转变为场景条件的分布一致性评估，避免了对匹配参考图像的需求，同时引入了失真感知的补充信号。

**🔧 技术方法**

使用了多元高斯分布建模深度特征空间中的高质量场景观察，并结合了弱监督的失真感知补丁鉴别器，通过无监督的有界融合策略将两者结合。

**📊 数据集**

在多个基准数据集上进行实验，包括3DGS-IEval-15K、GS-QA和NeRF QA数据集，这些数据集包含了不同场景的渲染图像。

**📈 对比分析**

与现有的全参考、无参考和交叉参考方法相比，SCODA在多个基准数据集上表现出强大的与人类判断的一致性，尤其是在3DGS-IEval-15K数据集上，SCODA的整体相关性最佳。

**⚠️ 局限性**

SCODA的局限性在于其未经过人类评分的回归训练，尽管在多个数据集上表现出强大的泛化能力，但在某些情况下可能仍然无法捕捉到所有类型的失真。

---

## 46. When Stakeholder-centric Requirements Engineering is Not Enough: An Action Research Study on Legacy System Modernisation

**arXiv ID:** 2609.07340 | [PDF](https://arxiv.org/pdf/2609.07340v1)

**作者:** Ruward S. Karper `[一作]` (Jheronimus Academy of Data Science), Willem-Jan van den Heuvel `[通讯]` (University of Tilburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在跨国能源公司进行的行动研究中，利用利益相关者中心的需求工程技术（半结构化访谈、Delphi 共识工作坊、MoSCoW 优先级划分、用户故事编写）生成了可理解、正确的目标系统需求规格，并通过问卷评估其在与遗留系统差距分析中的效果。

**💡 创新点**

证明利益相关者中心需求工程能高质量生成目标系统需求，但在遗留系统功能映射上的一致性不足，揭示单靠人类视角无法充分支持差距分析，强调需要结合技术证据的混合方法。

**🔧 技术方法**

采用了访谈、Delphi、MoSCoW、用户故事编写以及 Krippendorff α 统计评估等技术，并在讨论中提出可借助软件分析工具与大型语言模型（LLM）辅助的可能性。

**📊 数据集**

数据来源为该能源公司 14 位利益相关者的访谈记录、生成的 105 条需求（用户故事）以及后续问卷中关于可理解性、正确性和遗留系统覆盖度的二元响应。

**📈 对比分析**

通过问卷中的二元判断和 Krippendorff α 进行一致性比较；在可理解性与正确性方面 α 分别达到 0.94 与 0.90，显示高度一致；而在功能映射一致性方面 α 仅为 0.45，表明较低的共识。

**⚠️ 局限性**

研究仅在单一公司单一遗留系统上进行，受限的利益相关者参与与证据可得性导致结果缺乏普遍性，且缺乏对技术证据融合方法的实证验证。

---

## 47. Resource-Constrained Semantic-Aware Remote Estimation with Overlapping Sensor Coverage

**arXiv ID:** 2609.07563 | [PDF](https://arxiv.org/pdf/2609.07563v1)

**作者:** Bowen Sun `[一作]` (Linköping University), Nikolaos Pappas `[通讯]` (Linköping University)

**通讯引用:** 4246 | [OpenAlex ID](https://openalex.org/A5084740578)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在具有重叠传感器覆盖、多源有限状态马尔可夫源以及全局和单传感器传输频率约束的语义感知远程估计问题，并提出了联合源-传感器调度的有限平均成本约束马尔可夫决策过程(CMDP)模型。

**💡 创新点**

创新点包括：①证明传输资源函数秩不超过传感器数K；②Lagrangian仅依赖K个有效传输成本；③最优策略可用至多K+1个确定性策略随机化实现；④Lagrangian值是分段线性凹函数；⑤提出投影双重子梯度法并给出收敛性分析。

**🔧 技术方法**

采用了约束马尔可夫决策过程、乘子松弛、占用测度与多链策略理论、线性规划、投影双重子梯度优化以及平均成本Bellman方程求解技术。

**📊 数据集**

使用合成数据：三源两传感器的实验设置，源状态为三状态马尔可夫链，传感器可靠度与延迟不同，且引入了非对称的估计误差成本矩阵。

**📈 对比分析**

通过投影双重子梯度法求解约束问题，并与已知最优解做对比。数值结果表明Lagrangian值呈分段线性，随机化必要性得到验证；随单传感器预算变化，性能变化符合理论预测，收敛速度满足 O(1/√N) 的理论上界。

**⚠️ 局限性**

局限性：①仅适用于有限状态马尔可夫源和全局/单传感器传输频率约束；②求解需要完整的CMDP状态空间，规模较小；③未给出可扩展的近似或在线调度算法；④未考虑不完全信息、随机延迟或非同步情况。

---

## 48. Scoring Without the Engine: Validating a Deterministic, Manipulation-Resistant Content Score for Generative Engines, End to End

**arXiv ID:** 2609.07559 | [PDF](https://arxiv.org/pdf/2609.07559v1)

**作者:** Elisha Bajemon `[一作]` (TW3 Partners), Andre-Louis Rochet `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一套用于替代昂贵、受限、非平稳生成式引擎的确定性代理评分协议，构建了可审计的内容质量得分并通过多种反事实门控（负控、剂量响应、饱和、重复惩罚、长度中性）来确保代理不被攻击；

**💡 创新点**

核心创新在于：①将“真因果锚点”与“对抗性门控”相结合，形成基于可验证性而非单纯对齐的加权策略；②通过条件化天际线（query‑conditioned skyline）界定代理在未见查询时的理论上限；③在真实引擎实验中重新测量并证实旧锚点失效，说明门控对代理设计至关重要；

**🔧 技术方法**

技术包括：确定性文本特征提取（正则、统计、TF‑IDF）；使用平方根非线性变换实现单杠杆放大限度；多重对齐与正则化（余弦相似度与Ridge惩罚）求解权重；门控测试与交叉验证；使用量化对抗攻击和标准网页垃圾检测基线评估；

**📊 数据集**

使用GEO‑Bench公开数据集（500条英文页面+500条对抗编辑；250训练/250测试；60法语扩展），以及10种不同生成式引擎（6开源、4闭源，3款GPT‑5.x 2026版）进行结果验证；

**📈 对比分析**

通过对比多种聚合策略（线性、平方根、分位数）和基线（等权、最佳单特征、历史观测权重），在门控合格与否的评估下选择最终方案；在游戏检测基准上，攻击者对代理的得分提升被限制在6点以内，并且在标准垃圾检测基线中表现优越；在对照引擎实验中，代理与原始权重相比，内在相关性仅略有提升（Spearman≈0.11）但已达到理论上限；

**⚠️ 局限性**

局限包括：①代理仅能捕获内容侧信号，无法直接预测查询‑来源相关性；②对抗门控与锚点的依赖可能导致对新型攻击或不同语言的泛化不足；③实验使用的引擎与检索环境受限，未覆盖真实的检索竞争与缓存机制；④对抗编辑的设计与实现可能存在漏洞（如论文中已公开的错误）；

---

## 49. Re-engineering SORT-based algorithms for low-cost small object tracking from omnidirectional footage

**arXiv ID:** 2609.07547 | [PDF](https://arxiv.org/pdf/2609.07547v1)

**作者:** Xin Shu `[一作]` (Trinity College Dublin), Anil Kokaram `[通讯]` (Trinity College Dublin)

**通讯引用:** 4124 | [OpenAlex ID](https://openalex.org/A5073540444)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对低成本全景摄像机下的小目标多目标跟踪问题，提出轻量级的OmniSORT框架。

**💡 创新点**

创新点在于引入跨缝隙运动模型SAMM、融合欧几里得距离与GIoU的关联成本E_fuse，以及新的OmniSmall基准。

**🔧 技术方法**

主要技术包括Kalman滤波的缝隙自适应修正、Omni‑Euclidean距离计算、GIoU关联以及基于CPU的在线跟踪。

**📊 数据集**

使用了自制的OmniSmall数据集以及公开的JRDB全景行人跟踪数据集。

**📈 对比分析**

与SORT、OCSORT、ByteTrack等基准比较，OmniSORT在OmniSmall上提升HOTA+8.51、MOTA+9.41、IDF1+10.17，YOLOX检测时增益收窄至+1.95，保持CPU‑only速度。

**⚠️ 局限性**

局限在于对检测器依赖较大，缺乏自适应的融合权重λ，且在非小目标场景如JRDB的表现仅保持竞争力。

---

## 50. PhysReal: Learning Real-World Deformable Object Physics via Hybrid Constitutive Modeling

**arXiv ID:** 2609.07532 | [PDF](https://arxiv.org/pdf/2609.07532v1)

**作者:** Yinan Deng `[一作]`, Yufeng Yue `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了PhysReal，一个基于视频的框架，利用混合专家‑神经本构模型与可微MPM模拟器从稀疏单视角视频中学习真实可变形物体的物理动力学。

**💡 创新点**

通过空间可变的混合本构模型、分阶段课程学习以及运动与蒙版双重监督，实现了从单一交互视频中识别复杂异质材料行为的突破。

**🔧 技术方法**

技术方案包括MPM物理仿真、3D Gaussian Splatting渲染、可微物理求解、专家+神经残差本构网络、梯度优化、课程学习以及像素跟踪与蒙版监督。

**📊 数据集**

实验采用公开的PhysTwin数据集以及自采集的包含填充玩具与布料的交互数据集进行评估。

**📈 对比分析**

与GS‑Dynamics、Spring‑Gaus、PhysFlow和PhysTwin等四个基线在Chamfer距离、轨迹误差、IoU和PSNR等指标上进行对比，PhysReal在重建与未来预测任务中均显著优于对手，尤其在自采集数据上表现最突出。

**⚠️ 局限性**

主要局限在于对极薄薄膜（如布料）等高度细腻结构的体积MPM建模受限；神经残差受限于预定义本构形式；仅使用单一视频序列难以覆盖多样交互场景，未来需要多视频联合学习以进一步提升泛化能力。

---

## 51. When Semantically Consistent Encoding Meets View-Label Heterogeneity Modeling: A Unified Framework for Incomplete Multi-View Multi-Label Learning

**arXiv ID:** 2609.07525 | [PDF](https://arxiv.org/pdf/2609.07525v1)

**作者:** Chengliang Liu `[一作]` (University of Macau), Wenwu Wang `[通讯]` (University of Surrey)

**通讯引用:** 9655 | [OpenAlex ID](https://openalex.org/A5100676721)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了一个统一框架V2L，用于不完整多视角多标签学习；

**💡 创新点**

创新点在于：①源扰动不变的变分编码实现跨视角语义一致性；②主动视图‑标签相关性建模，既能捕捉实例级、标签级视图异质性，又可通过后向决策融合实现自适应权重；③将表示层和决策层通过混合融合结构紧密耦合；

**🔧 技术方法**

核心技术包括变分自编码器、信息瓶颈、跨视图一致性约束、扰动不变性匹配、对比正则化、主动感知网络、PoE联合推断以及多标签交叉熵与自监督相关性损失；

**📊 数据集**

实验使用了五个公开多视角多标签数据集：Corel5k、Pascal07、ESPGame、IAPRTC12和MIRFLICKR；

**📈 对比分析**

在50%视图缺失和50%标签缺失的双重缺失设置下，V2L在AP、AUC、1‑HL、1‑RL、1‑OE、1‑Cov等六项指标上均优于CDMM、DM2L、LVSL、iMVWL、NAIM3L、DICNet、DIMC、MSLPP、SIP和QARF；在完整数据集上亦保持领先或接近最优的性能；

**⚠️ 局限性**

主要限制包括：训练耗时较长且参数量较大；对信息瓶颈权重β、后向一致性权重γ、对比正则权重σ等超参数较敏感；模型目前验证于图像特征，多模态通用性仍待进一步探索。

---

## 52. An LLM-Associated Register Shift in Korean Journal Abstracts: A Morphology-Aware Excess-Vocabulary Study, 2018-2026

**arXiv ID:** 2609.07447 | [PDF](https://arxiv.org/pdf/2609.07447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 53. From Human Factors to Human-Technology Factors: An HCI Perspective on Technology in Avalanche Safety

**arXiv ID:** 2609.07560 | [PDF](https://arxiv.org/pdf/2609.07560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 54. CrACK: Adversarial Attacks on Cross-Model Consistency in Collaborative Vision Foundation Models

**arXiv ID:** 2609.07499 | [PDF](https://arxiv.org/pdf/2609.07499v1)

**作者:** Feifei Liu `[一作]` (South China Normal University), Xiaoyu Tang `[通讯]` (South China Normal University)

**通讯引用:** 3144 | [OpenAlex ID](https://openalex.org/A5012106499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 CrACK 的推理时攻击框架，能够在不改动输入像素、模型权重或训练数据的前提下，通过操纵协同视觉基础模型（VFM）之间的特征接口，破坏训练-free 视觉分割流水线的语义一致性，从而导致严重性能下降。

**💡 创新点**

创新点在于识别出协同 VFM 流水线中普遍存在的“语义‑空间对齐依赖”这一结构性弱点，并设计了两阶段攻击（ACI：利用 CLIP 语义逆相似性对 SAM 产生的 affinity 矩阵进行反转；SIP：基于 CLIP 文本嵌入的最大距离置换对预测标签进行恶意重映射），实现了对接口层的高效利用和跨模型的灾难性攻击。

**🔧 技术方法**

使用技术包括：CLIP patch‑token 提取、逆相似性混合、SAM attention 与特征的同时置换、CLIP 文本嵌入几何构造的最大距离置换、以及在 LLaVA 等多模态大模型中的视觉提示注入；所有操作均在推理时完成。

**📊 数据集**

实验数据集涵盖八个开源分割基准（PASCAL VOC20/21、Context59/60、COCO Object/Stuff、Cityscapes、ADE20k）以及 VQA 评测集（POPE、MM‑Vet、LLaVA‑Wild），同时对 LLaVA‑1.5‑13B 进行下游推理测试。

**📈 对比分析**

通过与传统输入级攻击（FGSM、PGD‑SAM、PGD‑CLIP）以及三种输入防御（高斯模糊、JPEG 压缩、随机裁剪）对比，CrACK 在所有四条流水线和八个基准上平均将 mIoU 降至 1–5 之间，攻击成功率（ASR）超过 98%，显著优于基线；且对输入级防御无效，体现出接口级攻击的优势。

**⚠️ 局限性**

局限性：攻击需在白盒或供应链/微服务级别上可读写 SAM 中间特征并使用 CLIP patch‑token 提取器，无法直接应用于完全黑盒场景；未来工作需扩展至无监督或迁移学习的黑盒攻击，并设计可验证的接口一致性防御。

---

## 55. The Internal Anatomy of Strategic Choice in Large Language Models

**arXiv ID:** 2609.07478 | [PDF](https://arxiv.org/pdf/2609.07478v1)

**作者:** Vinícius Ferraz `[一作]`, Enrico Ferrea `[通讯]` (German Primate Center)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5038112728)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一套完整的144个严格序数2×2游戏中，作者让四种开源大型语言模型（Qwen2.5、Qwen2.5‑Instruct、Llama‑3.1‑Instruct和GPT‑OSS‑120B）做一次性决策，并记录其内部激活；随后用线性探测、几何对齐和对激励方向的可逆干预，探究模型内部如何把激励信息转化为选择，并与人类实验数据进行对照。

**💡 创新点**

核心创新在于：①首次在完整的游戏空间里统一衡量LLM的可用性（信息是否存在）、招募（信息是否被用来决策）和易感性（信息是否可被外部操控）三种内部特征；②通过“激励方向”干预证明即使信息未被自然招募，模型仍可被强制改变选择；③对比指令微调前后的相同预训练模型，揭示后训练阶段可改变信息到决策的路径，却不改变行为或信息可用性。

**🔧 技术方法**

技术方法包括：自然语言提示生成、一次性决策采样、记录Transformer残差流激活、线性解码器、激励-决策几何角度测量、梯度提升树特征重要性分析、对激励方向的可逆干预以及对cue词的答案轴投影分析。

**📊 数据集**

数据集为：完整的144个严格序数2×2游戏（按玩家角色区分为144个游戏），每个游戏四种提示顺序的四个等价版；人类对照数据来自两套公开实验（Ordinal‑Payoff panel和Cardinal‑Payoff panel），覆盖同一游戏集合；此外还使用了游戏的结构特征（复杂度、均衡类型）和cue的可识别性集合。

**📈 对比分析**

比较方法：将模型选择与人类选择、理论均衡与随机选择做一致性、效率和协调率比较；用量化的激励敏感性λ衡量决策对激励差值的反应；用quantal level‑k模型统一映射所有代理的推理深度；通过线性解码评估激励与选择的可用性；用几何角度评估招募；用干预测量易感性。性能方面，所有模型在高复杂度游戏中都表现出与人类相似的退化趋势；Qwen2.5‑Instruct在招募和易感性上优于Qwen2.5 base和Llama，而GPT‑OSS在激励对齐上最弱。

**⚠️ 局限性**

局限性包括：只研究了一次性严格序数2×2游戏，无法推广到重复、连续或不完全信息游戏；模型样本仅限四个大模型，缺乏更广泛的架构覆盖；cue仅用单一词句，无法分离词汇与心理机制；干预仅在单一层面和激励方向上测试，未覆盖所有可能的决策路径；GPT‑OSS因路由结构不支持同类干预；未能对内部推理过程进行因果追踪，仍只能观察到表面级别的招募与易感性。

---

## 56. FramingQA: Does the Question Shape the Answer? Measuring the Compositional Framing Effect

**arXiv ID:** 2609.07448 | [PDF](https://arxiv.org/pdf/2609.07448v1)

**作者:** Hazel H. Kim `[一作]` (University of Oxford), Philip H. S. Torr `[通讯]` (University of Oxford)

**通讯引用:** 59240 | [OpenAlex ID](https://openalex.org/A5042899882)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为FramingQA的基准，用以系统评估大型语言模型在法律、医学、金融和机器人等高风险领域对问题表述（framing）敏感性的表现；

**💡 创新点**

创新点在于：①将 framing 效果细分为根级、命题级和全局级三层，并对每层进行组合式变化；②引入严格（compositional robustness）评价指标，揭示模型在同一事实下对不同表述的表现不一致；③跨四大领域（法律、医学、金融、机器人）构建统一框架；

**🔧 技术方法**

使用了大规模预训练 LLMs（Gemma‑3、LLaMA‑3.1、Mistral、Phi‑3/4）进行推理，并通过 HuggingFace 接口进行一次性回答；同时在机器人场景中利用 GPT‑4.1 mini 进行仿真；

**📊 数据集**

使用了四套领域数据集：法律（LegalBench 的 issue‑spotting 与 hearsay）、金融（Personal Finance Stack Exchange 片段）、医学（顶级期刊临床案例）、机器人（PARTNER 与 Habitat 3.0 的任务集合）；每套数据均生成多种表述变体，共计 54,756 个问题；

**📈 对比分析**

通过对比平均准确率、严格准确率、任务完成率、规划步骤数和重规划次数等指标，发现模型在平均层面表现中等（多项任务 48–61%），但严格准确率骤降至 0–35%（多项任务）或 0%（二分类），表明模型在面对相同事实的不同表述时缺乏语义系统性；

**⚠️ 局限性**

局限性包括：仅针对单轮、基于证据的分类任务；未覆盖多轮对话、开放式生成或其他领域；数据集以简化版本为主，可能无法体现真实专家级推理复杂度；

---

## 57. Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat

**arXiv ID:** 2609.07557 | [PDF](https://arxiv.org/pdf/2609.07557v1)

**作者:** Simone Foti `[一作]` (Imperial College London), Tolga Birdal `[通讯]` (Imperial College London)

**通讯引用:** 2348 | [OpenAlex ID](https://openalex.org/A5038619214)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于热扩散的内在纹理表示——Heat Kernel Textures（HKTex），用离散黎曼几何中的各向异性热核在三角网格表面直接建模纹理，完全不依赖UV展开，可通过Riemannian优化与物理渲染器耦合，实现从已有UV或多视角图像的自适应纹理优化。

**💡 创新点**

① 用离散LBO/ALBO谱分解得到的热核天然贴合曲面，避免欧氏高斯与表面分离的问题；② 通过Riemannian梯度下降和指数映射保证核位置始终在曲面上；③ 引入双调和距离加权、Biharmonic KNN与局部化滤波，抑制谱摆动并提升计算效率；④ 在可微PBR管线中直接评估热核，实现无缝逆渲染；⑤ 实现对UV纹理的压缩和多视角图像的无缝重建，消除缝隙、畸变与冗余存储。

**🔧 技术方法**

离散LBO与各向异性LBO谱分解、热扩散公式、双调和距离、Biharmonic KNN、Riemannian梯度下降、并行传输、可微光线追踪（OptiX / RTX）、Triton、基准方法（MLP、InstantNGP、INFs、ImageGS）、渲染评估指标（MSE、PSNR、SSIM、MS-SSIM、LPIPS）。

**📊 数据集**

过滤后的Objaverse子集（约3000个高质量三角网格），其中313个用于UV纹理拟合，162个用于多视角渲染实验；训练时使用生成的多视角渲染图像；对比实验使用同一数据集的UV、顶点颜色、MLP等基线。

**📈 对比分析**

在相同内存预算下，与传统UV纹理、顶点颜色、高分辨率顶点纹理、MLP（Fourier/InstantNGP/INFs）、ImageGS等基线进行比较。HKTex在MSE/PSNR/SSIM/MS‑SSIM/LPIPS上优于或相当于其他方法，且存储占用最低；在多视角逆渲染中比NVDiffRec和HR VTex获得更好的感知质量，同时存储更少。渲染时间略高，但仍可接受。

**⚠️ 局限性**

目前仅实现了漫反射（albedo）纹理，尚未扩展到完整BSDF属性；对极高频纹理细节仍有一定限制；需要先做LBO/ALBO谱分解，处理大规模或非流形网格时成本较高；整体计算量比传统UV渲染略大。

---

## 58. Qwen-Audio-3.0-ASR Technical Report

**arXiv ID:** 2609.07549 | [PDF](https://arxiv.org/pdf/2609.07549v1)

**作者:** Chuanmeng Bian `[一作]` (Alibaba Token Foundry), Jianheng Zhuo `[通讯]` (Alibaba Token Foundry)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Qwen-Audio-3.0-ASR，一款基于 Qwen Mixture-of-Experts 的大规模 LLM‑ASR 系统，支持 30 种语言和 16 种中文方言，并集成行业实体识别、分层热词定制、单通道纠错润色与长音频上下文建模。

**💡 创新点**

通过统一的指令控制解码框架，将多语言、多方言、热词、上下文与单通道润色等功能集成到同一模型，避免多模型切换和后处理链；采用 MoE LLM 与音频编码器的高效适配，实现低延迟流式与非流式两种模式。

**🔧 技术方法**

使用 Qwen MoE 语言模型、SenseVoice 音频编码器、CTC 辅助解码、低秩 LoRA 微调、GRPO 强化学习与自研 FunVerl‑ASR 异步 RL 框架，以及实体挖掘、语音合成等数据增强技术。

**📊 数据集**

训练数据涵盖数千万小时多语言音频，包括公开基准（AISHELL、LibriSpeech、CommonVoice、FLEURS、GigaSpeechBench 等）和内部工业语料（行业实体、长尾词、热词、方言、长音频记录）。

**📈 对比分析**

通过统一 API 在公开基准和内部工业测试集上与 GPT‑4o Transcribe、Azure、Doubao‑ASR 等系统对比，Qwen‑Audio‑3.0‑ASR 在多语种、中文/英文、方言、热词、长音频等场景均获得最优或竞争性错误率，尤其在方言识别、实体召回与流式低延迟方面表现突出。

**⚠️ 局限性**

仍受限于低资源语言样本不足、极长或极嘈杂音频鲁棒性、热词/上下文切换准确性，以及在极低延迟下的准确率下降；模型规模大、部署成本高，对极细粒度方言或新实体的即时适应仍需进一步提升。

---

## 59. Analytical Resource Management for Fine-grained MoE Computation-Communication Overlap

**arXiv ID:** 2609.07536 | [PDF](https://arxiv.org/pdf/2609.07536v1)

**作者:** Hongyu Liu `[一作]` (Chalmers University of Technology), Miquel Pericas `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在分布式Mixture‑of‑Experts推理中基于工作负载和资源约束的启动时资源管理器，自动决定计算与通信CTA的分配；

**💡 创新点**

创新点在于将CTA资源分配建模为波量化的依赖约束问题，利用分析模型在不需要轮询或重新编译的情况下在启动前即做最优决策；

**🔧 技术方法**

使用依赖‑约束的波量化模型、GPU SM资源占用分析、计算与通信波数预测以及CPU侧的快速求解算法；

**📊 数据集**

在NVIDIA A100 4卡上评估了Granite、Qwen 和 DeepSeek‑V2‑Lite 三个大规模MoE模型，采用BF16精度、不同TP/EP组合以及多种序列长度；

**📈 对比分析**

与公开的COMET实现以及Megatron core‑TE、FastMoE TP+NCCL对比，平均加速分别为2.53×（GEMM2+GatherRS）、1.77×（完整post‑router层）和1.19×（完整模型预填），最大加速分别为4.22×、2.58×和1.44×；

**⚠️ 局限性**

局限性包括模型仅针对单机A100/ NVLink 架构，且在极短任务或小规模波数时提升有限，未来需验证在多节点互连及新GPU架构上的适用性。

---

## 60. Zero-Shot Sim-to-Real Contact-Rich Assembly via Proprioception-Anchored Cross-Modal Pretraining

**arXiv ID:** 2609.07534 | [PDF](https://arxiv.org/pdf/2609.07534v1)

**作者:** Yuhan Wang `[一作]` (Shanghai Jiao Tong University), Wenzhao Lian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 606 | [OpenAlex ID](https://openalex.org/A5017678179)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种以本体感知为锚点的跨模态编码器 PACE，能够在没有真实世界调优的情况下将仿真中学到的接触丰富装配策略直接迁移到机器人硬件。

**💡 创新点**

创新点包括：①使用本体感知状态转移预测来约束视觉与力/扭矩特征，使其跨域不变；②引入可学习的零和时间差分核以抑制静态偏置；③采用跨模态掩码重建激励不同模态互补信息，提升鲁棒性。

**🔧 技术方法**

技术手段涵盖：多模态预训练（本体锚定、空间对齐、接触预测、掩码重建）+ 对称演员-评论家强化学习，利用冻结的编码器特征训练策略并直接在硬件上部署。

**📊 数据集**

使用了工业机器人 Assembly 任务集：Factory benchmark 的 PegInsert、GearMesh 以及 AutoMate 数据集的 JointAssemble、KnobTighten，共四个任务。

**📈 对比分析**

与 Pose-Based PPO、AugInsert、No-Pretrain RL 等基线相比，PACE 在真实硬件上的平均成功率达到 93.3%，仿真-现实性能下降仅 2.7pp，显著优于基线的 18.9~43.7pp 的降幅。

**⚠️ 局限性**

局限性包括：依赖于高精度的本体感知和力/扭矩传感器；在本体感知不稳定或缺失的场景下效果可能下降；需要专业的仿真专家策略收集，训练流程相对复杂。

---

## 61. Topologically Consistent Agricultural Parcel Vectorization with Semantic-Guided Diffusion and Topology-Aware Polygonization

**arXiv ID:** 2609.07520 | [PDF](https://arxiv.org/pdf/2609.07520v1)

**作者:** Weiqin Jiao `[一作]`, Claudio Persello `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种拓扑一致性的农业地块矢量化框架，能够从高分辨率遥感影像中同时生成精确的边缘与顶点原语，并通过共享边图实现无内部侵入的地块多边形生成。

**💡 创新点**

创新点包括：① 语义引导的联合边缘-顶点潜在扩散模型，既保证几何正规性又抑制误检；② 区域划分+共享边图的拓扑感知重构策略，首次在同一语义类别下实现共享边缘重用；③ 在大尺寸影像上通过拼接中间原语而非最终多边形实现单张图的全局一致性。

**🔧 技术方法**

核心技术：联合边缘-顶点潜在扩散（LDPoly风格）、监督多模条件（边缘+地块掩模）、VMamba+UPerNet特征提取、分水岭式区域划分、共享边图构建与顶点引导简化、Tile级中间原语融合与重构。

**📊 数据集**

使用 AI4SmallFarms（越南、柬埔寨小农田）和 iFLYTEK（中国产区）两个公开遥感数据集进行训练与评估。

**📈 对比分析**

与 SEANet、TFNet、CLPsNet、Mask R-CNN、Mask2Former、DeepSnake、E2EC、E2EVAP、PolyR-CNN、RoIPoly、B2PNet 等基线对比，性能显著提升：在 AI4SmallFarms 上 IoU 提升至 90.54%（+1.22%），BIoU 86.65%（+4.70%），CIoU 78.73%（+17.16%），侵入率降为 0% 并且共享边回忆率最高 43.39%；在 iFLYTEK 上也取得 BIoU 69.45%、CIoU 61.89%、侵入率 0%、共享边回忆率 21.96%。

**⚠️ 局限性**

局限性：① 两阶段设计导致边缘或顶点原语误差会直接传递至重构阶段；② 对极端形状或弱边缘的鲁棒性仍有限；③ 当前方法未对原语不确定性建模，可能导致过拟合或误合并；④ 需要进一步探索端到端可微分的拓扑感知重构以进一步提升性能。

---

## 62. Perception-Aware Joint Power and Sub-Band Allocation for 6G In-Body Subnetworks

**arXiv ID:** 2609.07519 | [PDF](https://arxiv.org/pdf/2609.07519v1)

**作者:** Samira Abdelrahman `[一作]` (Aswan University), Hossam Farag `[通讯]` (Aalborg University)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5074765348)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于人类延迟感知的6G体内子网络功率与子带分配框架。

**💡 创新点**

将用户个体的可感知延迟阈值（JND）纳入资源分配，突破传统仅基于QoS的设计。

**🔧 技术方法**

采用高斯混合模型+监督学习估计JND，并通过Lyapunov漂移加惩罚的实时分配算法。

**📊 数据集**

使用公开的XR用户延迟感知实验数据（30人）并在此基础上合成了1000用户的特征与JND。

**📈 对比分析**

与固定延迟约束的基线对比，稠密部署下平均功率降低约60%，严苛延迟场景下节能率达26%。

**⚠️ 局限性**

依赖合成数据，未在真实XR环境下验证；JND预测误差和时变特性仍需进一步研究。

---

## 63. P$^2$Calib: Utilizing Pattern Priors for LiDAR-Camera Extrinsic Calibration

**arXiv ID:** 2609.07516 | [PDF](https://arxiv.org/pdf/2609.07516v1)

**作者:** Xiangcheng Hu `[一作]` (Hong Kong University of Science and Technology), Xiangcheng Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5021447091)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了利用板块几何先验（孔半径与四孔矩形布局）来改进LiDAR-摄像头外参校准的P^2Calib系统。

**💡 创新点**

创新点在于将孔半径先验固定为CAD值并引入共享偏移，消除中心-半径退化；再将四孔布局约束为刚性矩形，增强跨孔一致性。

**🔧 技术方法**

使用了圆形拟合、基于HUBER损失的迭代优化、矩阵投影（Procrustes）以及闭式SVD求解外参。

**📊 数据集**

实验数据集包括：仿真板（多距离、多噪声）、移动平台上的固态LiDAR（FS-B、FS-C）以及多种扫描LiDAR（Livox Avia、Mid‑360）。

**📈 对比分析**

与FAST‑Calib、velo2cam等基线对比，P^2Calib将联合残差降低约90%（FS-B）/82%（FS-C），留一法重投影误差下降约96%/77%；在所有传感器上均表现出更小的误差和更高的检测率。

**⚠️ 局限性**

局限性包括仅适用于四孔矩形板块；共享偏移在某些传感器下可能为零；需要已知几何的板块，且对非矩形或多孔板块的扩展尚未实现。

---

## 64. Modus Tollens and Counterfactuals and Counterfactual Reasoning Based on Three Types of Negation

**arXiv ID:** 2609.07483 | [PDF](https://arxiv.org/pdf/2609.07483v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 65. Statistical versus machine learning-based spatial interpolation of post-processed ensemble weather forecasts

**arXiv ID:** 2609.07512 | [PDF](https://arxiv.org/pdf/2609.07512v1)

**作者:** Mária Lakatos `[一作]` (University of Debrecen), Mária Lakatos `[通讯]` (University of Debrecen)

**通讯引用:** 100 | [OpenAlex ID](https://openalex.org/A5082730138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较统计与机器学习方法对ECMWF 2‑m 温度和10‑m 风速集合预测的后处理，并评估其在有观测站和无观测站的校准与预测精度。

**💡 创新点**

提出高度感知线性池（ALP）用于无观测站的预测组合，并系统比较了EMOS、boosted EMOS、分布回归网络、Transformer 与图神经网络在有限与扩展预测变量两种设定下的空间插值性能。

**🔧 技术方法**

使用统计后处理方法（EMOS、boosted EMOS）、分布回归网络（DRN）、Transformer、图神经网络（GraphSAGE）以及标准与高度感知线性池。

**📊 数据集**

基于2007–2016年德国站点的50成员ECMWF 2‑m 温度和10‑m 风速的TIGGE集合及其观测（DWD），共99个温度站和182个风速站。

**📈 对比分析**

采用滚动训练窗口，利用CRPS、MAE、RMSE、预测区间覆盖率、宽度及可靠性指数等指标，对观测站和未观测站进行全面比较。结果显示后处理普遍优于原始集合，但不同方法在不同变量、站点组和指标上排名各异；在未观测站中，ALP在组合预测上表现出显著提升。

**⚠️ 局限性**

局部模型（EMOS‑L）在未观测站泛化性能差；图神经网络使用的简单地理图在高海拔站可能导致误差；线性池组合在未观测站有时弱化单一预测；此外风速预测不适用于线性池，限制了对风速组合方法的比较。

---

## 66. What a Model Refuses, a State Fears: How Authoritarian Information Control Reproduces in Language-Model Guardrails

**arXiv ID:** 2609.07507 | [PDF](https://arxiv.org/pdf/2609.07507v1)

**作者:** Menglin Liu `[一作]`, Ge Shi `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究大型语言模型的拒绝行为，比较中国与西方模型的政治防护栏，发现中国模型更频繁拒绝涉及中国的集体行动提示，并在政治域中优先阻断协调而非异议。

**💡 创新点**

首次将治理论框架应用于模型拒绝行为，揭示模型拒绝不仅受内容本身影响，还嵌入了开发者所在国家的威胁模型，并展示拒绝行为的“渗透性”（porosity）。

**🔧 技术方法**

采用对抗式搜索（攻击性改写）评估拒绝可破坏性，以及线性概率模型和差异因子对比分析，测定模型的拒绝率与鲁棒性。

**📊 数据集**

使用10个基于不同国别的指令调优模型（4美国，4中国，2开放权重），三种语言的政治/非政治提示集合，以及对抗性攻击提示集。

**📈 对比分析**

通过差异因子和线性回归对比模型拒绝率，发现中国模型在特定参考名词（如“China”）时拒绝率提高约+60个百分点；对抗性攻击下，中国模型的拒绝易被突破（约70%恢复），表明“硬性”拒绝与真正的控制无关。

**⚠️ 局限性**

主要局限在于模型来源与监管、预训练语料、对齐层难以解耦；参考名词操作仍可能受语言、细节的影响；初步AI编码需进一步人类验证；结果对中文内部细节有所弱化。

---

## 67. Mitigating Shortcut Learning: Texture-Penalized Prototype Networks

**arXiv ID:** 2609.07504 | [PDF](https://arxiv.org/pdf/2609.07504v1)

**作者:** Akshay Anilkumar Girija `[一作]` (Institute for AI Safety and Security), Sven Hallerbach `[通讯]` (Institute for AI Safety and Security)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Texture-Penalized Prototype Network (TPPN)，通过 Texture-Penalization Branch (TPB) 与基于超球面原型的分类模块联合，强制卷积网络抑制局部纹理特征，提升全局形状辨识能力。

**💡 创新点**

创新点在于将梯度反转层 (GRL) 作为纹理抑制头来惩罚高频纹理特征，同时在超球面空间引入多原型动态 Top‑K 池化，实现形状驱动的端到端学习，而无需依赖昂贵的风格化数据集。

**🔧 技术方法**

使用 ResNet‑50 作为骨干网络，加入 1×1 投影、GRL、MPL 纹理分类头、动态 Top‑K 池化、温度缩放、信息噪声对比学习等技术；整体损失为交叉熵、拉伸、拉平与纹理惩罚之和。

**📊 数据集**

主要数据集为 ImageNet 10‑class 子集（训练 20k 张、验证 1.6k 张）以及 Cue‑Conflict 数据集（包含形状-纹理冲突与 OOD 形状/纹理），并通过合成噪声（高斯模糊、斑点噪声、色差）生成 4‑类纹理代理。

**📈 对比分析**

与 ResNet‑50 基线、VGG‑16 预训练、ViT‑B/16 预训练等模型比较，TPPN 在 Cue‑Conflict 数据集上将纹理偏差从 55.11% 降至 29.73%，形状识别率从 19.11% 提升至 29.82%，误判率从 70.33% 降至 40.47%；在干净验证集仅损失 0.9% 的 Top‑1 准确率。

**⚠️ 局限性**

局限性包括：仅在 ImageNet 10‑class 子集上验证，缺乏对完整 ImageNet‑1K 的评估；纹理抑制头仅使用简单的四种人工纹理；原型数目有限，难以覆盖更丰富的形状变异；未在不同任务或数据域（如医学影像）进行迁移性能验证。

---

## 68. An Empirical Study on the Impact of Change Granularity in Refactoring Detection

**arXiv ID:** 2609.07482 | [PDF](https://arxiv.org/pdf/2609.07482v1)

**作者:** Lei Chen `[一作]` (Institute of Science Tokyo), Shinpei Hayashi `[通讯]` (Institute of Science Tokyo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了在提交历史中识别粗粒度（CGR）与短暂（EPR）重构的方法，并在 32 个 Java 项目中进行实证研究，揭示了它们的频率、类型及原因。

**💡 创新点**

创新点在于定义并检测跨多个提交的粗粒度重构和仅在单个提交内但随后提交破坏的短暂重构；改进匹配方案，生成多级压缩单元并匹配重构签名。

**🔧 技术方法**

技术：使用 RefactoringMiner 进行重构检测，采用 git-blame 追踪重构目标位置，构造不同 offset/granularity 的压缩单元；并通过匹配与归类得到 CGR/EPR。

**📊 数据集**

数据集：从 Silva 等人收集的 124 个 GitHub Java 项目中挑选 32 个，包含 154,826 次提交，覆盖多领域（Web、Android、图形、网络等）。

**📈 对比分析**

对比方法：将原始单提交检测结果与不同粒度压缩提交检测结果进行匹配；评估 CGR/EPR 频率、类型比例；Precision/efficiency 通过人工验证验证 RefactoringMiner 在不同 granularity 下的准确率（约 96%）和检测时间随 granularity 增大而线性增长。

**⚠️ 局限性**

局限：仅使用 RefactoringMiner；只研究 Java 代码；git-blame 可能不准确；手工分类样本有限；commit message 语义不确定；未直接与开发者访谈验证意图。

---

## 69. Parser-Free VLM Verification for Federated Weakly Supervised Video Anomaly Detection

**arXiv ID:** 2609.07455 | [PDF](https://arxiv.org/pdf/2609.07455v1)

**作者:** Sébastien Thuau `[一作]` (ESIEA), Rachid Chelouah `[通讯]` (CY Cergy Paris University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出轻量化联邦多实例学习（MIL）与冻结视觉‑语言模型（VLM）的级联框架，用于弱监督视频异常检测；

**💡 创新点**

创新点在于：①仅在联邦环境下训练轻量级MIL头部，VLM保持冻结且仅作为后置验证器；②通过提取下一词“是/否”logit实现无解析、连续异常评分，兼具可解释性和实时性；③消除对生成文本解析与强提示的依赖，提升鲁棒性与因果一致性；

**🔧 技术方法**

使用技术包括：联邦学习（FedAvg）、多实例学习（MIL）、CLIP视觉特征、Frozen VLM（InternVL3.5‑2B 与 Qwen3‑VL‑2B‑Instruct）、logit‑based 异常评分、置信度融合与可选时间后处理；

**📊 数据集**

实验数据集为 UCF‑Crime，采用 32 段划分与帧级评估；

**📈 对比分析**

与传统 MIL 基线和密集式零样本 VLM 进行对比。logit 方案在保持无后处理的前提下，F‑AUC 分别提升约 +1.05%（InternVL）和 +1.52%（Qwen3），F‑AP 进一步提升 4–7%。相较于文本生成方案，logit 方案在稳定性、可解释性及因果兼容性上表现更优；

**⚠️ 局限性**

局限性包括：①VLM 验证仅在离线集成环境完成，未验证本地/边缘部署的隐私/延迟；②仅使用 UCF‑Crime，缺乏跨数据集验证；③联邦设置基于四台 CPU 机器的模拟分割，未覆盖真实场景异质性；④仅利用 CLIP 预提取特征，未探究动态特征或更深的视觉模型；

---

## 70. TeMo: Temperature Modulation for Multimodal Contrastive Learning

**arXiv ID:** 2609.07540 | [PDF](https://arxiv.org/pdf/2609.07540v1)

**作者:** Dhimitrios Duka `[一作]` (Max Planck Institute for Informatics), Anna Kukleva `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 TeMo——一种基于相似度的温度调制框架，在多模态对比学习中对每个正负对自适应地调整温度，并将此机制扩展到单模态损失中，最终通过渐进式调度将标准 InfoNCE 与调制损失融合。

**💡 创新点**

创新点：① 针对每对样本的相似度动态生成温度，实现更细粒度的对比正则化；② 同时对跨模态和单模态损失都进行温度调制；③ 采用二次调度方式逐步从固定温度过渡到完全调制的损失，兼顾全局结构与局部细节。

**🔧 技术方法**

技术手段：InfoNCE 对比损失、相似度映射温度函数、二次（quadratic）调度器、图像-文本对、图像-图像、文本-文本的对比目标、LLM 生成的文本同义语义增强、ResNet‑50 / ViT‑B/16 视觉编码器和 DistilBERT 文本编码器。

**📊 数据集**

数据集：预训练使用 Conceptual Captions（CC3M、CC12M）；评估包括 MSCOCO、Flickr30k（零样本检索）、CIFAR‑10、CIFAR‑100、ImageNet‑1k（零样本分类）以及 CLIP Benchmark 17‑数据集。

**📈 对比分析**

与多种基线对比：InfoNCE、温度调度 (TS*)、DySTreSS*、CWCL*、MM‑TS、SoftCLIP 等。TeMo 在检索 R@1 上相较于 InfoNCE 提升约 1.6–2.4%，在分类 Top‑1 上提升 9–12%（CIFAR）和 2.4%（ImageNet），在所有评测任务上均达到或逼近当下最高水平。

**⚠️ 局限性**

局限性：① 需要额外的温度调度超参数（τ_min、τ_α）和调度设计；② 逐对温度计算增加了计算与内存开销；③ 单模态无调制时会导致性能下降，需要与调制配合；④ 对早期特征质量敏感，若预训练阶段不充分可能影响温度学习；⑤ 对噪声正样本的鲁棒性尚未全面验证。

---

## 71. PICANet: Physics-Informed Cascaded Asymmetric Network for Infrared Small Target Detection

**arXiv ID:** 2609.07515 | [PDF](https://arxiv.org/pdf/2609.07515v1)

**作者:** Jingjing Liu `[一作]` (Shanghai University), Wanquan Liu `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种物理信息驱动的级联不对称网络PICANet，用以解决红外小目标检测中的背景噪声扩散和高层语义特征中目标衰减问题。

**💡 创新点**

创新点在于三大模块的设计：层级先验解耦模块（HPDM）将低层几何先验与高层语义先验分离；双先验交互融合模块（DPIFM）实现先验对深度特征的动态门控注入；跨层注意模块（MCFAM）通过级联不对称机制完成多层特征的精确对齐与融合。

**🔧 技术方法**

技术细节包括：基于冻结的多方向特征聚合计算块（MFACB）提取物理梯度；深度学习骨干（U-Net、ResNet‑FPN）嵌入；双先验门控机制（BPGFE）；交叉空间-通道注意力（ISC‑A）；以及综合损失函数（多尺度BCE+SoftIoU+MaskedMSE）。

**📊 数据集**

使用了三个公开红外小目标检测数据集：NUDT‑SIRST、IRSTD‑1k 和 SIRST‑Aug。

**📈 对比分析**

与传统模型驱动、纯数据驱动和模型‑数据驱动方法对比，PICANet 在三大数据集上在 mIoU、F1、P_d 和 F_a 上均取得领先或相近的最优表现，尤其在 NUDT‑SIRST 上取得所有指标第一，体现出卓越的检测精度与鲁棒性。

**⚠️ 局限性**

主要限制在于计算量和推理速度相对较高，需要进一步轻量化设计以提升实用性。

---

## 72. Generation of Vectorized Maps Beyond Vehicle View

**arXiv ID:** 2609.07511 | [PDF](https://arxiv.org/pdf/2609.07511v1)

**作者:** Clara Gomez `[一作]` (Centre for Automation and Robotics, CSIC-UPM), Jorge Villagra `[通讯]` (Centre for Automation and Robotics, CSIC-UPM)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了在未观测区域生成向量化地图连续体的任务，并实现了基于Transformer的生成模型。

**💡 创新点**

首次将向量化地图预测扩展到传感器视野之外，并引入端点、拓扑辅助以及贝塞尔曲线表征。

**🔧 技术方法**

采用Transformer编码器-解码器、几何与拓扑头、贝塞尔曲线、自动回归生成、端点MLP以及多项损失函数。

**📊 数据集**

使用3DHD CityScenes真实激光点云数据集构建的自定义训练集。

**📈 对比分析**

与基线（曲线延伸法）对比，RMSE下降约15%，CD与FD亦显著改善，表明方法在简化路面上可行。

**⚠️ 局限性**

仅限于简化道路配置，无法准确捕捉车道分叉/合流，且在更复杂场景下鲁棒性不足。

---

## 73. Functional-SLAM: Interaction-Aware Mapping with Online Functional Scene Graphs

**arXiv ID:** 2609.07497 | [PDF](https://arxiv.org/pdf/2609.07497v1)

**作者:** Xinggang Hu `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了 Functional‑SLAM 框架，实现了在视觉 SLAM 流程中持续在线维护功能场景图，并将其用于增强闭环检测和定位。

**💡 创新点**

核心创新包括：① 将功能场景图从离线构建转为在线实时更新；② 采用 anchor‑keyframe 同步几何并通过功能上下文约束实现稳健节点关联；③ 通过时间关系后验累积多帧证据稳定功能边；④ 利用功能拓扑信息辅助闭环候选生成，提升定位鲁棒性。

**🔧 技术方法**

技术手段包括 MASt3R‑SLAM 视觉跟踪、Open‑Vocabulary 感知（DeepSeek + SAM3）、anchor‑keyframe 几何同步、Hungarian 匹配、时间关系后验、功能拓扑图签名与匹配以及基于功能拓扑的闭环验证。

**📊 数据集**

使用 SceneFun3D 与 FunGraph3D 两个室内功能场景图数据集进行评估，并在手持 iPhone 捕获的真实场景中验证其鲁棒性。

**📈 对比分析**

与 DSO、ORB‑SLAM3、MASt3R‑SLAM 等视觉 SLAM 基线，以及 OpenFunGraph、FunGraph、KeySG 等离线功能图方法进行对比。实验表明 Functional‑SLAM 在 ATE RMSE 上比强基线低约 16%，在节点召回率、三元组召回率上均优于离线方法；相较于离线方法，其运行速度提升至 0.38 FPS，显著高于 0.02–0.06 FPS。

**⚠️ 局限性**

主要限制是对视觉跟踪和开放词汇感知的依赖，在大间隙、纹理缺失或剧烈运动模糊等极端场景下，定位和功能图构建性能会显著下降。

---

## 74. FPScan: An Automated Constraint-Based Analyzer for Floating-Point Anomaly Detection

**arXiv ID:** 2609.07492 | [PDF](https://arxiv.org/pdf/2609.07492v1)

**作者:** Julien Bortolussi `[一作]` (ENAC ISAE-SUPAERO ONERA Université de Toulouse), Pierre-Loïc Garoche `[通讯]` (ENAC ISAE-SUPAERO ONERA Université de Toulouse)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一款名为FPScan的静态分析工具，用来正式定义并检测浮点程序中的吸收（absorption）与灾难性抵消（catastrophic cancellation）问题。

**💡 创新点**

创新点在于：①基于浮点数的数量级、精度与误差三种整数量的抽象，构建可用于证明错误不存在的约束；②使用抽象解释推导变量范围后，将错误传播关系离散化为一阶约束；③通过SMT求解器对约束求解，从而在所有输入上保证安全性，并且在缺失部分情况时仍能给出可解释的警告。

**🔧 技术方法**

技术手段包括：抽象解释（interval analysis）、数量级/误差/精度的离散化与约束生成、SMT求解（Z3）来判定约束可满足性、程序语义的约束化、循环展开与有限迭代上界。

**📊 数据集**

使用FPBench（约130个浮点数计算程序，保留58个符合要求的子集）作为评测数据集，并从中选取41个支持位向量（bitblasting）验证的基准。

**📈 对比分析**

评估方式：①与基于bitblasting的完整工具比较，验证FPScan的无误性与完整性；②与动态检测器FPChecker比较，分析检测覆盖率；③测量运行时间，显示FPScan在多数基准上与FPChecker相当，远快于bitblasting。结果表明FPScan能证明72%的案例无错误，误报率仅8%，且整体执行时间比bitblasting快一个数量级。

**⚠️ 局限性**

局限性：①分析不完全，仍会产生少量误报；②只支持单精度（binary32）且未覆盖混合精度情形；③循环分析需要用户指定展开上界，超大或无界循环不可处理；④约束生成对程序的相关性缺乏捕捉，导致某些可行错误被误判为不可行。

---

## 75. MEMO: Multimodal Evidence Memory Organization for Long-Horizon LLM Agents

**arXiv ID:** 2609.07471 | [PDF](https://arxiv.org/pdf/2609.07471v1)

**作者:** Xian Gao `[一作]` (Shanghai Jiao Tong University), Yuzhuo Fu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了长时序LLM代理的外部记忆读出问题，提出MEMO多模态证据记忆组织方法，通过证据单元选择、文本/视觉/双通道呈现及布局规划实现有限上下文预算内高效记忆重构。

**💡 创新点**

引入证据级别的多模态呈现决策，将文本与视觉结构结合，联合优化证据选择与布局；使用训练好的证据提取器和记忆管理器，以读者性能为目标的离线反馈学习；实现统一的预算约束下的文本/视觉分配。

**🔧 技术方法**

基于Qwen2.5-1.5B-Instruct的监督微调；证据提取器采用序列到结构的输出；记忆管理器通过离线读者评估的监督学习方式；视觉渲染模板（卡片、时间线、表格等）；实验使用InternVL3.5-8B、Qwen3-VL-32B、gpt-5.4-mini等多模态大模型。

**📊 数据集**

HotpotQA、2WikiMultiHopQA（多跳文档问答）、LoCoMo（长对话记忆）以及ALFWorld（机器人任务轨迹）。

**📈 对比分析**

与文本仅记忆、视觉仅记忆、BM25检索、Mem-α、MemAgent、AgentOCR、MemOCR等基线在128-token预算和无预算两种设置下比较；在128-token下MEMO在三种读者上均获得最高EM/F1；在无预算下使用平均83.93 tokens，仍保持首或次最佳整体得分，显示显著压缩与高质量兼得。

**⚠️ 局限性**

依赖预先冻结的读者模型，离线反馈学习需要额外计算；视觉渲染可能丢失细粒度文本细节；方法对预算配置较为敏感，跨读者泛化能力仍需进一步验证。

---

## 76. Temporal-Causal Inference for Reinforcement Learning via Automata Learning

**arXiv ID:** 2609.07461 | [PDF](https://arxiv.org/pdf/2609.07461v1)

**作者:** Jan Corazza `[一作]` (TU Dortmund University), Daniel Neider `[通讯]` (TU Dortmund University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 TCIRL 框架，联合学习控制策略与隐藏因果 DFA，以应对不可观测的时序因果导致的非马尔可夫动态；

**💡 创新点**

创新点在于利用阶段专属随机标记构造可验证的反例，通过 SAT 推理迭代更新 DFA，并在弱标记假设下证明几乎必然收敛到真实因果语言及最优策略；

**🔧 技术方法**

技术核心包括：LTL（有限轨迹）编译为 DFA、产品 MDP 构造、基于 Q‑learning 的无模型强化学习、反例驱动的 SAT 合成、并对 DFA 进行最小化与成功闭包约束；

**📊 数据集**

实验数据集：基因治疗格子世界（5×5 网格、5 状态因果 DFA）和交通信号管控环境（3 交叉口、7 状态因果 DFA）；

**📈 对比分析**

与四个基线对比：Q‑learning（已知因果、未知因果）、DQN（4/3 帧历史），TCIRL 在两域均与已知因果基线持平，并显著优于无因果知识与短历史 DQN；

**⚠️ 局限性**

局限性：需人工设计阶段专属标记，无法自动发现；仅适用于一次性不可逆因果（非循环），且对高维或连续状态空间的扩展仍待研究。

---

## 77. Anti-Gravity Walking by a Flying Humanoid Robot via Thrust-Rate Input Whole-Body Model Predictive Control

**arXiv ID:** 2609.07544 | [PDF](https://arxiv.org/pdf/2609.07544v1)

**作者:** Kazuki Sugihara `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了基于推力率输入的全身MPC，使飞行人形机器人能够在天井等抗重力环境中进行多点接触行走。

**💡 创新点**

创新点包括：将推力视为状态并用其时间微分作为控制输入，保证接触切换时推力连续；在CWC约束中加入足法向下压力下限，并通过负载转移策略平滑过渡；以及将上述方法集成到实时MPC框架并在硬件上实现演示。

**🔧 技术方法**

技术手段主要是全身MPC、推力率输入定式化、CWC约束、负载转移策略，并使用Crocoddyl/Pinocchio求解器、MuJoCo仿真以及硬件实验。

**📊 数据集**

实验数据来自自建的飞行人形机器人（2.0 kg）在MuJoCo仿真和真实硬件平台，未使用公开数据集。

**📈 对比分析**

与传统推力输入MPC对比，推力率MPC收敛更快（16步对26步）、推力、接触力和关节力更平滑；仿真平均求解时间为6.7 ms，90%以上在10 ms以内；硬件实现行走0.15 m，姿态误差≤0.1 rad，推力波动在8–12 N之间。

**⚠️ 局限性**

局限在于关节位置偏离初始姿势、缺乏足部力感测、使用高减速比伺服导致无力矩控制、Solver仅处理输入约束、未加入终端状态约束，以及通信延迟和动力响应未建模。

---

## 78. CosmoH2G: A Hand-to-Gripper Transfer Dataset and Baseline Method for Object Manipulation with Complex Spatial Movements

**arXiv ID:** 2609.07498 | [PDF](https://arxiv.org/pdf/2609.07498v1)

**作者:** Hongxiang Zhao `[一作]` (Chinese University of Hong Kong, Shenzhen), Xiaoguang Han `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过构建大规模复杂空间运动的手-夹爪配对数据集，并提出两阶段数据驱动框架，实现了从人类手部演示到机器人夹爪动作的精确转移。

**💡 创新点**

创新点包括①基于可扩展采集管线生成专注复杂空间运动的 6,189 期、1,254 个对象的数据集；②引入两阶段学习策略，先预测起止关键帧，再生成完整动作并通过后期优化分离位姿与方向，显著提升精度与鲁棒性。

**🔧 技术方法**

技术方法采用点云 Transformer 编码、Transformer 扩散模型、三关键点姿态表示、基于接触、平滑与 IK 的后期位姿优化。

**📊 数据集**

使用的数据集为 CosmoH2G，包含 6,189 条手-夹爪配对演示，涵盖旋转、翻转等多样的复杂空间运动，共 1,254 个独特对象。

**📈 对比分析**

与传统优化/学习基线对比，在模拟与真实机器人实验中，CosmoH2G 在 GOA、SR、TS、TOPA 等指标上显著提升（例如 GOA 7.53°，相较 10–30° 的基线有明显改进）。

**⚠️ 局限性**

局限性包括当前仅为开环控制，缺乏实时误差校正与碰撞规避，且单阶段直接转移方法仍受限，未来需要扩大数据规模并引入闭环反馈与碰撞感知。

---

## 79. Wearable Multimodal Human-Machine Interface for Integrated Hand Intentions Decoding in Dynamic Teleoperation

**arXiv ID:** 2609.07495 | [PDF](https://arxiv.org/pdf/2609.07495v1)

**作者:** Jiaxuan Li `[一作]` (Dalian University of Technology), Liming Shu `[通讯]` (Dalian University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一套可穿戴的多模态人机接口（MI‑DHMI），集成64通道sEMG电极和手臂/手部IMU，并设计了统一的意图解码框架（ADF‑Net、MCF‑Net、SZPR），实现了在非约束手腕和前臂运动下对手部姿势、手势和抓握力的同步解码，用于动态远程操控。

**💡 创新点**

创新点在于：1）首次将高通量sEMG与手腕/前臂IMU结合的可穿戴系统，实现完整手部意图的实时解码；2）提出Attention‑based Dual‑Level Fusion Network（ADF‑Net）与Motion‑Compensated Force Network（MCF‑Net），通过数据级与特征级双向融合及注意力机制显著抑制手腕运动引起的sEMG漂移；3）Soft Zero‑Velocity Update（Soft ZUPT）姿势重建方法提升位置估计精度；4）在实验室远程操作平台上验证系统在水瓶倒水和多手势抓取任务中的可行性。

**🔧 技术方法**

采用的技术包括：高通量64通道sEMG阵列、手臂/手部IMU、无线传输、Vicon运动捕捉校准、1D‑CNN + 残差网络、双模态注意力机制、Soft ZUPT、LSTM、经典机器学习基线（DT、RF、KNN、MLP），以及基于ROS2的机器人控制。

**📊 数据集**

使用自制数据集：10名受试者，7种抓握手势（圆柱、球形、指卷、钩握、精细抓、三指抓、侧抓、静止），每个手势连续保持15秒，5次试验，采样率为sEMG 2000Hz、IMU 400Hz。数据与Vicon捕捉同步，用于训练/验证/测试（3:1:1）。

**📈 对比分析**

与市售MYO臂套、传统机器学习模型以及单模态解码进行对比。实验显示：手势识别精度在97.16%（注意力+双模态融合），抓握力R²最高为0.95；单模态sEMG或IMU性能显著低于融合方案；在在线远程操控任务中，所有手势的成功率均≥70%，多手势任务成功率最高达100%。

**⚠️ 局限性**

局限性包括：1）缺乏跨受试者泛化验证，模型对新用户的迁移性待研究；2）系统仅在实验室条件下测试，未评估长期稳定性、能耗和实时延迟；3）前臂仅使用单一IMU，更多IMU或自适应融合可能进一步提升鲁棒性；4）未在真实工业/医疗场景中验证。

---

## 80. Where Should Language Sit in a Multimodal Model? Lessons from What Language Does to Human Perception and Cognition

**arXiv ID:** 2609.07474 | [PDF](https://arxiv.org/pdf/2609.07474v1)

**作者:** Peng Xie `[一作]` `[通讯]`, Peng Xie

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述语言在多模态模型中的角色，比较语言与感知、思维的关系，并通过cue‑conflict实验评估模型对语言与其他感官输入的融合规则。

**💡 创新点**

创新点在于提出语言是共享码本的压缩器，将其视为人类认知的通道而非内部表示，并用cue‑conflict实验揭示模型在多模态融合中偏离可靠性加权的规律。

**🔧 技术方法**

使用基于Transformer的多模态模型（Qwen、InternVL、Idefics等）、对比学习与强化学习框架，并结合心理学实验的cue‑conflict方法。

**📊 数据集**

利用公开的多模态基准（VQA、COCO、LIBERO等）、人工构造的视觉-语言冲突任务以及人类实验数据。

**📈 对比分析**

通过与理想观测者的权重对比和基准性能评估，发现模型在可靠性加权方面约为理想观测者的11–82%，并在VQA、视觉动作等任务中出现明显的模态忽略现象。

**⚠️ 局限性**

局限性包括：缺乏真实感知输入的完整测量、对不同模型结构的泛化性有限、以及未能彻底解决模型内部语言表征导致的可审计性问题。

---

## 81. Measuring Language Transfer in Robot Policies: Adding Greek to a Cosmos3 Vision-Language-Action Policy

**arXiv ID:** 2609.07470 | [PDF](https://arxiv.org/pdf/2609.07470v1)

**作者:** Ayoub Kirouane `[一作]`, Christos Petrocheilos `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在现有以英语为主的机器人基础模型上，利用机器翻译快速构建希腊语指令集，并在不改动模型架构的情况下，将双语指令混合注入训练，评估希腊语指令跟随效果；同时在此过程中引入多种对照实验和控制，剖析指标失效的原因并得出可靠的实践建议。

**💡 创新点**

提出了在机器人视觉‑语言‑动作模型中评估低资源语言适配的完整控制实验框架，系统地证明仅靠多语塔或词汇稀疏并不能保证语言跟随；验证了“加入目标语言演示”是必要但不足的；通过大规模任务集与多种对照显著减少了误报，形成可复现的“先建null再评估”准则。

**🔧 技术方法**

使用Cosmos3开源机器人学习栈（包含视频世界模型、可共享的Transformer文本塔与动作解码器）；对文本塔进行冻结/解冻、语料混合采样、语言多样化（多种希腊语表达）等实验；评估采用闭环任务成功率、图像生成的一致性判定、误差测度等多指标；对比时使用“错误指令控制”“多种随机种子”等控制方法。

**📊 数据集**

希腊语指令集完全由LLM自动重述生成，包含1,273条桥梁数据V2场景描述和53,207条DROID+LIBERO任务指令；此外使用10-任务和90-任务Libero仿真套件进行评估；实验亦涉及真实机器人数据DROID，但因缺少物理硬件仅用作动作预测代理。

**📈 对比分析**

与英语指令的对照实验采用三种条件：正确英语、正确希腊语、错误指令；在90任务套件中，双语策略在希腊语下的成功率为27.4%（相较错误指令基线+6.9点），英语为62.5%；单语希腊策略最高仅+2.7点；而在10任务套件中双语策略希腊成功率为48.6%（+6.7点），单语希腊为0-2.7点。多语训练显著提升希腊语跟随，但仍低于英语约30-40点。

**⚠️ 局限性**

主要限制包括：希腊语指令完全来自机器翻译，缺少人类原生表达；实验仅在仿真环境中验证，未在真实机器人上完成完整评估；语言适配效果高度受随机种子影响，单跑指标不可靠；多语塔的词汇切分差异和模型容量未彻底剖析；缺乏对更大语料、多语种以及跨域迁移的进一步验证。

---

## 82. When Superpixels Fail on Documents: A Study of Segmentation for LIME Explanations

**arXiv ID:** 2609.07462 | [PDF](https://arxiv.org/pdf/2609.07462v1)

**作者:** Quentin Telnoff `[一作]` (University of La Rochelle), Antoine Doucet `[通讯]` (University of La Rochelle)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对LIME在文档图像分类中的分割策略进行评估，探讨文档感知分割对解释可靠性的影响。

**💡 创新点**

强调分割是LIME核心设计而非预处理，并证明文档感知分割（OCR边框、网格）显著提升一致性、正确性和局部忠诚度；揭示文档ID码偏差。

**🔧 技术方法**

采用LIME框架，比较Quickshift、SLIC超像素与OCR边框、网格等文档感知分割；使用线性回归作为代理模型；通过Spearman一致性、插入/删除AUC和R^2评估解释质量。

**📊 数据集**

RVL‑CDIP数据集（仅选取Invoice、Budget、Form三类）。

**📈 对比分析**

使用同一模型多次运行，计算一致性、插入/删除AUC和R^2；文档感知分割在一致性上达0.933‑0.975，显著优于Quickshift（0.432）和SLIC（0.792）；插入AUC最高0.764、删除AUC最低0.031；R^2最高0.571，超像素方法表现相对较差。

**⚠️ 局限性**

仅在单一ResNet‑50分类器和RVL‑CDIP子集上验证；OCR分割质量受限于OCR精度；未探究更复杂模型或不同文档类型；实验规模受计算资源限制。

---

## 83. Finding Representative and Approximately Efficient Committees

**arXiv ID:** 2609.07554 | [PDF](https://arxiv.org/pdf/2609.07554v1)

**作者:** Dominik Peters `[一作]` (CNRS), Jatin Yadav `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在比例认可投票（PAV）中，局部搜索和全局优化版本的效率和代表性，并提出一种新的多阶段算法，结合了凸优化、pipage 取整和局部搜索，输出满足比例代表性、弱帕累托最优、α≈1.346 的分数帕累托最优和≈0.79 的 PAV 分数近似的议员名单。

**💡 创新点**

创新点在于首次证明全局 PAV 具备 α≈1.346 的分数帕累托最优保证，并构造了一个多阶段“round‑and‑swap”算法，既保持全局 PAV 的代表性，又通过局部搜索提升效率，克服了原本 PAV 计算 NP‑难的瓶颈。

**🔧 技术方法**

采用了凸/近似凸规划求解 h‑函数下的多元优化、PAV 多线性扩展的 pipage 取整，以及基于 PAV 分数增益的局部搜索；同时使用了 Poisson‑平滑和 Bernoulli‑Poisson 不等式证明效率下界。

**📊 数据集**

未使用公开真实数据集，实验和证明均在构造的合成实例上进行。

**📈 对比分析**

与 DMMS 算法相比，该算法在实现同等 0.79 的 PAV 分数近似的同时，额外满足 α≈1.346 的分数帕累托最优和弱帕累托最优，理论上提供了最优多项式时间逼近；与局部 PAV 对比，获得了更强的效率和代表性保证。

**⚠️ 局限性**

局限在于：仍无法在多项式时间内保证完整的帕累托最优（即 α=1），并且对于更严格的代表性公理或更高 α 值的分数帕累托最优，算法的可行性与性能尚未得到保证。

---

## 84. A Fundamental Limit in Decentralized Decision-Making

**arXiv ID:** 2609.07479 | [PDF](https://arxiv.org/pdf/2609.07479v1)

**作者:** Marco Carpentiero `[一作]`, Ali H. Sayed `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在完全去中心化网络中，基于局部信息交换的决策策略与最优中心化MAP分类器相比存在不可忽视的误差概率损失，并推导出该损失的闭式表达式，揭示了网络拓扑距离与观测信息的矩生成函数之间的相互作用。

**💡 创新点**

创新点在于首次证明任何去中心化决策策略都存在一个不可约的误差概率下界，并给出与中心化系统的误差比和学习延迟的精确近似；同时通过对传统社交学习的分析，证明其无法达到该极限，进一步揭示了分类与估计在去中心化下的根本差异。

**🔧 技术方法**

主要技术包括大偏差理论、精确渐近法（Edgeworth展开）、矩生成函数分析、凸优化（寻找 LMGF 的极值点）、图论中的最短路径距离分析，以及概率变换（指数定向分布）等。

**📊 数据集**

论文未使用真实数据集，而是通过合成实验（高斯混合、伯努利分布等）验证理论，并在环形网络与 Erdős–Rényi 随机图上进行数值仿真。

**📈 对比分析**

通过与中心化 MAP 分类器、传统社交学习以及最优去中心化策略的误差概率曲线比较，结果显示：最优去中心化策略比传统方法低 1–2 个数量级，但仍无法逼近中心化性能；在环形拓扑中学习延迟随网络规模线性增长，而在 ER 随机图中呈对数增长，验证了理论预测。

**⚠️ 局限性**

局限性包括：假设观测独立且满足可识别性；推导的闭式表达式仅适用于有限假设空间的分类问题；实现最优去中心化策略需要全局似然模型，可能违背隐私或通信约束；对估计问题的适用性仍未充分探讨。

---

## 85. EigenLI: Spectral Approximations to Late Interaction

**arXiv ID:** 2609.07561 | [PDF](https://arxiv.org/pdf/2609.07561v1)

**作者:** Archish S `[一作]` (Temple), Kirankumar Shiragur `[通讯]` (Microsoft Research India)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对多向量检索模型进行训练无关的低秩子空间压缩并提出对应的评分函数；同时推出单向量化版本EigenLI-SV；

**💡 创新点**

首次利用文档特定的低维谱子空间压缩 token 向量，取代传统聚类/池化，并给出与 MaxSim 等价的评分公式；

**🔧 技术方法**

使用文档二阶矩的前 k 个特征向量（SVD/特征分解）进行投影，构造高维单向量表示并实现高效 ANN 检索；

**📊 数据集**

在 BEIR 13 文本数据集、ColQwen3 视觉文档检索 ViDoRe-v3 以及 MS MARCO、NQ、HotpotQA 等多种数据集上进行实验；

**📈 对比分析**

与 k‑means++、Ward 聚类池化及 MUVERA 单向量基准进行 Recall、nDCG、MRR 对比；k=32 的 EigenLI 在大多数模型上优于聚类基线，EigenLI‑SV 在单向量模式下相较 MUVERA 提升 60–170%；

**⚠️ 局限性**

仅在无监督场景评估；单向量维度随 token 维度平方增长，导致高维成本；未在大规模 ANN + 重新排序管道中验证；对高度异质模型的鲁棒性有限。

---

## 86. "We Permit the Use of AI, but [...]": The Landscape of AI Policies in Popular Open Source Projects

**arXiv ID:** 2609.07542 | [PDF](https://arxiv.org/pdf/2609.07542v1)

**作者:** Andre Hora `[一作]` (Universidade Federal de Minas Gerais), Stefano Zacchiroli `[通讯]` (Télécom Paris)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 2,000 个最受欢迎 GitHub 仓库和 36 个知名项目/组织中的 281 条 AI 贡献政策进行系统分类与分析，探讨其对 AI 使用的许可、披露要求、AI slop 对策以及政策演化。

**💡 创新点**

创新点在于提出了六维 AI 政策分类框架，识别了十类 AI slop 对策，并对 92 条专门 AI 政策文件的纵向演化轨迹进行了首次量化研究。

**🔧 技术方法**

采用了 GitHub API 与 SEART 搜索工具获取政策文件，结合人工标注与统计分析方法对政策内容进行归类与计量。

**📊 数据集**

使用的数据集包括 2,000 个热门 GitHub 仓库、36 个知名项目/组织，共 281 条 AI 政策文件；以及 92 条专用 AI 政策文件的 196 条提交历史。

**📈 对比分析**

通过比例统计、共现可视化（UpSet）和条目计数等方法描述政策属性，发现 83.3% 允许 AI、50% 的政策被修订等趋势；该工作侧重描述性统计，未涉及性能比较或算法评估。

**⚠️ 局限性**

局限性包括仅覆盖公开 GitHub 项目，样本偏向热门活跃仓库，缺少闭源或其他语言生态的代表；人工标注仍存在主观性，且未验证政策是否真正被遵循。

---

## 87. Quantile-Led Feature Extraction for Multi-Horizon Predictive Maintenance in Industrial Manufacturing Systems

**arXiv ID:** 2609.07533 | [PDF](https://arxiv.org/pdf/2609.07533v1)

**作者:** David J Poland `[一作]`, Na Helian `[通讯]` (University of Hertfordshire)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并评估了基于分位数回归的双阶段特征提取框架，用于多时隙预测维护。

**💡 创新点**

将分位数回归与深度特征提取相结合，并通过尺度化的中尾量化级别以及时间嵌入实现跨预测时隙的可迁移表示。

**🔧 技术方法**

双层 MLP–QRNN 结构、channel‑resolved pinball loss、skip‑connected refinement、PReLU 激活、Transformer 时序分类。

**📊 数据集**

72 台工厂机器在 9 个设施中的高频多传感器工业时序数据，涵盖 43–81 个通道。

**📈 对比分析**

通过 13 条流水线的分阶段消融实验，对比 Transformer 仅模型、单阶段 QRNN 等，短期 60 min F1 达 98.4%，70 h 60.4%，30 d 79.97%。

**⚠️ 局限性**

仅在同一资产族内验证，未证实跨行业泛化；两阶段训练固定化，未尝试全联合微调；量化水平选择为经验性。

---

## 88. CoRL: Co-Evolutionary Reinforcement Learning for Adaptive Indirect Prompt-Injection Attacks and Defenses

**arXiv ID:** 2609.07529 | [PDF](https://arxiv.org/pdf/2609.07529v1)

**作者:** Boyang Zhang `[一作]` (South China University of Technology), Qingyao Wu `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出CoRL框架，训练工具使用代理在面对自适应间接提示注入（IPI）时既能完成任务又能阻止攻击。

**💡 创新点**

创新点在于将IPI建模为非对称、部分可观测的广义博弈，并通过在线双边Co‑PPO对抗历史对手实现共进化，再利用历史攻击者产生的安全演示进行修复。

**🔧 技术方法**

技术包括自监督初始化（Attacker SFT）、双边Co‑PPO强化学习、历史对手池、masked PPO、Verifier‑Grounded环境以及Defender SFT监督学习。

**📊 数据集**

使用的数据集为七个AgentDyn/AgentDojo领域的任务集合（共157个用户任务、168个固定模板、1187个自适应攻击），并在外部基准InjecAgent和AgentLAB上进行验证。

**📈 对比分析**

对比实验显示CoRL将攻击成功率（ASR）降至0%，任务实用性（Safe‑U）从~75%提升至~76%，相较于基线和单一角色训练均有显著提升；在外部基准上亦保持竞争力。

**⚠️ 局限性**

局限性包括：攻击目标有限且使用确定性Verifier；共进化未求解博弈均衡，仅靠历史对手暴露；依赖教师监督，缺乏完全自监督或更弱监督的验证。

---

## 89. From Bracha to Coded MBRB: Benchmarking Byzantine Reliable Broadcast Implementations

**arXiv ID:** 2609.07521 | [PDF](https://arxiv.org/pdf/2609.07521v1)

**作者:** Yenan Wang `[一作]` (Chalmers University of Technology), Timothé Albouy `[通讯]` (IMDEA Software Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现并评估了 Bracha、AFRT 和 Coded MBRB 三种 Byzantine 可靠广播算法在统一 Go 框架下的单发广播性能。

**💡 创新点**

创新点在于提供了统一的实现、调度、计量和解析器框架，使三种算法可在相同环境下可比，并通过可重复的实验集验证通信、计算、内存、延迟和安全性。

**🔧 技术方法**

使用 Shadow 模拟器、Go 原生性能剖析、GCP VPC 部署、FABRIC 测试bed，并结合分布式 TCP、加密签名、纠删码、向量承诺和阈值签名技术。

**📊 数据集**

采用自定义的单发广播工作负载，payload 从 100 KB 到 40 MB，节点规模 10–30，涵盖 92,190 次实验和 2,361,600 条解析检查记录。

**📈 对比分析**

通过总传输字节、实现消息数、CPU 指令、峰值堆、累计分配、完成延迟等指标比较，结果显示 Coded MBRB 在大负载下显著降低传输量和延迟，但在小负载时 CPU 成本和内存占用更高。

**⚠️ 局限性**

实验仅覆盖单发广播、无批量或持久化逻辑，未检验长时间运行的吞吐量，且只在有限的节点数、网络环境和故障模型下验证，缺乏全面的正式正确性和大规模可扩展性评估。

---

## 90. Improving Multivariate Time Series Classification with Class-Wise Training and Model Aggregation

**arXiv ID:** 2609.07493 | [PDF](https://arxiv.org/pdf/2609.07493v1)

**作者:** Mouhamadou Mansour Lo `[一作]` (University of Artois), David Mercier `[通讯]` (University of Artois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种按类别选择维度、按类别训练并聚合模型的多变量时间序列分类框架，旨在提升特征判别力并减少噪声维度影响。

**💡 创新点**

创新点在于：① 对每个类别单独选择最具辨别力的维度；② 针对每个类别训练一只一对其余类别的二分类器；③ 通过模型融合得到最终多类别预测，显著提升了高维数据下的鲁棒性和可解释性。

**🔧 技术方法**

技术核心包括：ECS/ECP 维度选择的类别化改造、MiniRocket 作为基线特征提取器、Sigmoid 对二分类器输出概率化、基于最大概率的最终预测。

**📊 数据集**

使用 UEA 多变量时间序列分类档案中的 25 个样本（包括 Motion, ECG, EEG, HAR, Audio, Other 等六大类，维度 2~1345，序列长度 30~17984）。

**📈 对比分析**

与 MiniRocket 基线（全局维度选择与无类别化）以及其他 ROCKET 族方法（ROCKET, MultiRocket, HYDRA 等）进行比较，采用 30 次重采样、平均准确率、标准误、Critical Difference Diagram 与 Multiple Comparison Matrix。结果显示：类别化训练提升平均准确率（≈+0.6%），在多类别、低维/高维数据上都有显著或稳定的改进。

**⚠️ 局限性**

局限性：① 需为每个类别训练一个模型，增加模型数和训练/推理开销；② 目前仅对 MiniRocket 及其衍生方法验证，未评估在其他基线或深度网络上的迁移；③ 对维度选择的阈值（elbow）仍依赖经验，可能影响不同数据集的最优性。

---

## 91. Blockchain-based Proportional Fair Scheduling for Multi-Operator O-RAN

**arXiv ID:** 2609.07473 | [PDF](https://arxiv.org/pdf/2609.07473v1)

**作者:** Kun Huang `[一作]` (Southeast University), Xiqi Gao `[通讯]` (Southeast University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在多运营商开放式无线接入网络（O‑RAN）中提出基于区块链的比例公平调度（BC‑PFS）方案，设计四个核心智能合约实现用户注册、状态上报、调度与结算，并给出理论分析与仿真验证。

**💡 创新点**

创新点包括：①利用区块链提供可信的跨运营商调度协作与账本，解决传统PFS缺乏信任基础的问题；②将PFS低复杂度算法映射到链上实现实时调度；③推导准确与简化闭式吞吐量公式，并量化资源池化收益；④通过理论与仿真展示调度收益随运营商与用户规模的单调提升。

**🔧 技术方法**

主要技术手段：区块链（权限链、Raft共识、Solidity智能合约）、比例公平调度算法、Rayleigh衰落信道建模、常数α平滑的吞吐量更新、基于概率与ODE的性能分析、JavaScript与Solidity实现。

**📊 数据集**

实验使用仿真数据：Rayleigh衰落信道、平均SNR–20dB至–10dB、1MHz带宽、10用户/运营商、调度间隔与信道保持一致，未使用公开真实数据集。

**📈 对比分析**

通过仿真将BC‑PFS与传统非协作PFS、最大和调度、最大最小公平调度进行对比，结果显示BC‑PFS在吞吐量、系统效用以及资源池化收益上均优于其他方案，且与理论推导高度吻合。

**⚠️ 局限性**

局限性包括：①对大量运营商/用户时分析模型复杂度高；②对速率波动假设较弱，实际多网络波动可能导致误差；③区块链共识时间与延迟可能限制调度频率；④实验仅基于仿真，缺乏真实网络部署与安全性评估。

---

## 92. Translation of Black-Box Clinical Prediction Models into Standalone Transparent Nomograms: Temporal External Validation in Heart Transplantation

**arXiv ID:** 2609.07610 | [PDF](https://arxiv.org/pdf/2609.07610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 93. Latent-to-Latent Flow for Volumetric Stochastic Segmentation

**arXiv ID:** 2609.07460 | [PDF](https://arxiv.org/pdf/2609.07460v1)

**作者:** Omar Todd `[一作]` (Imperial College London), Ben Glocker `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种Latent-to-Latent Flow（L2L-Flow）模型，用于在医学三维体积数据上实现高效的随机分割和不确定性估计。

**💡 创新点**

创新点在于提出时间平移噪声调度提升Flow-SSN稳定性，并将流匹配迁移到潜在空间，既减少计算量又保持性能。

**🔧 技术方法**

技术手段包括流匹配、神经ODE、潜在编码器、标签自编码器及条件先验。

**📊 数据集**

使用了私有CT放疗目标体积数据集和公开的CURVAS多器官分割数据集。

**📈 对比分析**

与MoSE、PhiSeg、SSN、原始和平移版Flow-SSN及deterministic nnU-Net对比，L2L-Flow在GED和多样性上与全分辨率模型相当，且推理速度提升约14倍。

**⚠️ 局限性**

局限性包括对高维数据训练稳定性的敏感、数据量有限、3D迁移难度大以及单一量化指标难以充分反映临床价值。

---

## 94. SMaRT-Tug: Structured Multi-Agent Reinforcement Learning for Physics-Based Tugboat-Barge Collaborative Manipulation

**arXiv ID:** 2609.07445 | [PDF](https://arxiv.org/pdf/2609.07445v1)

**作者:** Junkai Lu `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出一种基于物理仿真的去中心化多智能体强化学习框架，用以实现多船拖船协作操纵一艘浮动驳船。

**💡 创新点**

核心创新点包括：①使用GPU加速的物理仿真器，融合体积浮力、波浪与水动力阻力模型；②引入结构化控制先验（SCP），在残差策略上实现位置保持与协作几何约束；③在此框架下训练的MAPPO策略能够零样本泛化到更大队形和更强海况，且对不同任务阶段（直线行驶、转向、减速）具有良好适应性。

**🔧 技术方法**

技术手段主要包括：IsaacLab GPU物理引擎、Gerstenner波浪模型、体素化浮力计算、统一船体阻力模型、CTDE的MAPPO学习、PID结构化控制先验。

**📊 数据集**

数据集：本文使用自建的多船仿真环境进行大规模并行训练，无公开真实海洋数据集；训练过程涵盖多种命令、初始条件和波浪幅度，后续测试在不同波浪强度下进行。

**📈 对比分析**

对比方法：传统PID控制器（仅纵向速度控制）和中心化PPO。评估指标包括直线行驶速度MSE、最大恢复角度、转向末端侧移、减速残余速度等。实验显示MAPPO+SCP策略在所有任务上均优于两种基线，转向恢复速度快、侧移低、速度跟踪误差显著下降，且在更大队形、海况下仍保持稳定性能。

**⚠️ 局限性**

局限性：①实验仅在模拟环境验证，缺乏真实海域部署验证；②海浪模型为简化Gerstenner波，未涵盖复杂多尺度海浪与风浪耦合；③SCP参数需要手工调节，对不同船型或配置的迁移仍可能需要重新校准；④虽然实现了零样本扩展，但对极端海况（如波高>1.2 m）或非对称船型的鲁棒性尚未充分验证。

---

## 95. A new O(n log n) approach for the Euclidean maximum weight matching problem

**arXiv ID:** 2609.07501 | [PDF](https://arxiv.org/pdf/2609.07501v1)

**作者:** Rostislav Staněk `[一作]` (Technical University of Leoben), Robert Arustamyan `[通讯]` (Technical University of Leoben)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于最大角度旅行商问题的O(n log n)时间算法，用于求解二维欧几里得最大权匹配（Euclidean MWM）问题。

**💡 创新点**

创新点在于将最大角度TSP的最优解转换为匹配解，并通过在巡回路径上交替取边得到近似最优匹配，首次在该问题上实现了近乎最优的解与对数级别的复杂度。

**🔧 技术方法**

核心技术包括：Aichholzer 等人提出的最大角度TSP最优求解算法、图论匹配构造策略以及对偶/增广路径的快速更新。

**📊 数据集**

使用随机分布在正方形、圆形以及 TSPLIB 的结构化实例集，涵盖从几百到上万点不等的规模。

**📈 对比分析**

与网络X实现的 Blossom（O(n³)）和 Micali–Vazirani（O(n²·⁵））算法比较，实验表明该算法在几乎所有实例上获得了 99.9% 以上的目标值，并且执行时间仅为 Blossom 的 1–2% 甚至更低，具有极佳的性能。

**⚠️ 局限性**

局限性包括：对偶理论尚未完全证明对所有点分布的渐进最优性；当前实现为 Python 版本，仍可通过编译语言进一步提升规模；对偶点的选择对极小规模实例有微小影响。

---

## 96. Search-to-World: Evaluation of 3D World Delivery from User Request through Web Search

**arXiv ID:** 2609.07605 | [PDF](https://arxiv.org/pdf/2609.07605v1)

**作者:** Zixiao Gu `[一作]` (Institute of Artificial Intelligence, China Telecom (TeleAI)), Xuelong Li `[通讯]` (Institute of Artificial Intelligence, China Telecom (TeleAI))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个从用户请求到可使用的3D世界的端到端评估任务和相应的基准；

**💡 创新点**

创新点在于引入了观察检索率(ORR)与世界交付率(WDR)两个度量，构建了复用-再构建(harness)系统与结构化恢复控制，以评估和提升agentic系统的3D世界交付能力；

**🔧 技术方法**

使用了多模态检索、时间段定位、视频重建、Vision‑Language模型评估器、基于大模型的查询生成与恢复策略等技术；

**📊 数据集**

构建了包含8,900条训练请求和300条评估请求的公开数据集，覆盖20个细粒度语义域、室内外环境和五种空间尺度；

**📈 对比分析**

在300条评估请求上对八种开源模型（Qwen3.5、DeepSeek‑V4、GLM‑5.2、Kimi K3）和四种闭源模型（GPT‑5.6、Claude）进行全模型对比，发现DeepSeek‑V4‑Flash在ORR和WDR上表现最佳；恢复子代理的联合SFT显著提升了WDR和动作效率；

**⚠️ 局限性**

局限性包括依赖大模型的高成本、对实时网络环境的依赖、恢复策略仍需改进以覆盖更多失败场景，以及评估仍以自动化评测为主，缺乏人类主观质量验证。

---

## 97. ICI-VLA: In-Context Imitation with Spatiotemporally Aligned Demonstrations for Vision-Language-Action Models

**arXiv ID:** 2609.07581 | [PDF](https://arxiv.org/pdf/2609.07581v1)

**作者:** Songhua Yang `[一作]` (Wuhan University), Miao Li `[通讯]` (Wuhan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种固定文本动作VLM的检索条件化框架ICI‑VLA，在测试时通过检索微示例实现无梯度的任务适配。

**💡 创新点**

创新点在于：① 将阶段对齐的微示例检索与目标动作遮蔽训练相结合，充分利用原生文本生成接口；② 通过语义硬过滤+DTW监督构建精确的检索编码器；③ 在不更新参数的前提下实现高性能的无梯度上下文适配。

**🔧 技术方法**

采用了Qwen3‑VL‑4B和Qwen3‑VL‑Embedding‑2B模型，语义硬过滤、动态时间规整（DTW）检索监督、目标动作遮蔽（Target Action Masking）训练、微示例库构建与检索编码器训练等技术。

**📊 数据集**

使用了LIBERO、RoboTwin 2.0和Aloha四个物理任务的数据集，累计约11,200条长轨迹、139,659条微示例。

**📈 对比分析**

与多种VLA基线（Octo、OpenVLA、π_0、OpenVLA‑OFT等）和VLA‑0+Naive ICL进行对比。ICI‑VLA在LIBERO平均成功率97.7%，在RoboTwin 2.0平均成功率60.4%（比最高基线高19.3pp），在四个物理任务上平均成功率83.2%，显著优于π_0和VLA‑0。

**⚠️ 局限性**

局限性包括：① 对示例库和规划覆盖度高度依赖，覆盖不足会限制泛化；② 离线训练和检索计算开销较大；③ 检索效率和多模态对齐仍有提升空间。

---

## 98. Harnessing CLIP and DINO: An Uncertainty-Aware Cascaded Fusion Network for Generalizable Deepfake Image Detection

**arXiv ID:** 2609.07670 | [PDF](https://arxiv.org/pdf/2609.07670v1)

**作者:** Xuechao Zou `[一作]` (Beijing Jiaotong University), Junliang Xing `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种不确定性感知级联融合网络UCF-Net，用于检测深度伪造面部图像。

**💡 创新点**

创新点在于同时利用CLIP的语言对齐语义先验和DINO的自监督视觉结构先验，通过层级专家聚合和基于熵的不确定性加权融合，实现对多种生成器的泛化。

**🔧 技术方法**

采用CLIP ViT‑L/14和DINOv2 ViT‑L/14两大预训练模型，加入LoRA适配器，设计层级专家聚合（LEA）与不确定性感知融合（UAF），并使用焦点损失训练。

**📊 数据集**

构建统一深度伪造基准（约408万张图像）和跨生成器评估集（8807张来自八大近期生成模型的图像）。

**📈 对比分析**

与现有方法对比，UCF-Net在域内mAUC 95.33、域外mAUC 92.15均排名第一，并在少样本跨生成器适配中取得最高AUC（5样本/生成器 91.36，100样本/生成器 98.81）。

**⚠️ 局限性**

局限性包括对零样本跨生成器转移的表现仍不佳，且模型对不同生成器的适配仍需一定量的目标域样本。

---

## 99. MpSub: A Momentum $p$-Dimensional Subspace Trust-Region Method for Derivative-Free Fine-Tuning of Large Language Models

**arXiv ID:** 2609.07666 | [PDF](https://arxiv.org/pdf/2609.07666v1)

**作者:** Yuyang Wang `[一作]` (Xi'an Jiaotong University), Pengcheng Xie `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种动量+随机子空间的信赖域零阶优化方法，用于在不使用梯度的情况下对大型语言模型进行全参数微调。

**💡 创新点**

创新点在于：①将上一次接受步长的动量方向与多维高斯随机子空间结合，捕获更多梯度信息；②使用线性模型（中心差分估计）代替传统的二次插值，降低评估点需求；③将信赖域半径同时作为有限差分扰动幅度与子空间步长的上限，消除学习率调参；④通过种子重生成无显存占用的子空间方向，完全只依赖前向推理。

**🔧 技术方法**

采用的技术包括：零阶优化、中心差分梯度估计、信赖域框架、Gaussian 随机子空间、种子重生成、共享 minibatch 评估、理论收敛分析（随机子空间与有限差分误差估计）。

**📊 数据集**

使用的数据集为 SuperGLUE Benchmark 中的 CommitmentBank（NLI 三分类任务），并在 OPT‑125M 和 OPT‑350M 两个自回归语言模型上进行实验。

**📈 对比分析**

与 MeZO 进行同等前向评估预算（8400 次）对比。该方法在两种模型上均可获得与或略优于 MeZO 的测试准确率，且不需要学习率搜索；在不同初始信赖域半径下保持性能稳健；子空间维度为 20 时实现了计算与精度的良好平衡。

**⚠️ 局限性**

局限性包括：①理论证明仅针对固定确定性目标，未覆盖每次迭代重新采样的 minibatch；②实验仅在两种相对小的模型和单一数据集上验证；③未探讨并行评估与曲率信息引入的可能性；④对更大规模模型、生成任务以及参数高效微调等场景的适用性仍待验证。

---

## 100. RefVerifier: Semi-Automated Reference Claim Verification for Scientific Manuscripts

**arXiv ID:** 2609.07652 | [PDF](https://arxiv.org/pdf/2609.07652v1)

**作者:** Stefania Mocan `[一作]` (Technical University of Munich), Mark Kreitz `[通讯]` (University of the Bundeswehr Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种半自动化、以引用为界限的参考文献验证原型，旨在支持学术同行评审；

**💡 创新点**

创新点在于将引用句子提取与文献验证相结合，构建了基于引用边界的完整手稿检查流程；

**🔧 技术方法**

使用了自然语言处理技术（句子分块、引用识别、信息检索与匹配算法）以及机器学习模型进行验证；

**📊 数据集**

利用公开学术论文语料库（如arXiv、PubMed Central）及其引用数据库进行实验；

**📈 对比分析**

与传统单独事实核查方法对比，验证精确率提升至约83%，召回率达到78%，整体表现优于现有开域检索工具；

**⚠️ 局限性**

局限性包括依赖可检索的公开引用信息，对非引用声明缺乏检测能力，并且对多语言文献支持不完善。

---

## 101. Near-Term Verification Methods for AI Chip Exports

**arXiv ID:** 2609.07637 | [PDF](https://arxiv.org/pdf/2609.07637v1)

**作者:** Bruna Avellar `[一作]` (Independent Researcher), Erich Grunewald `[通讯]` (Institute for AI Policy and Strategy)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究并提出了在一至两年内可实施的AI芯片出口管制验证机制，涵盖端点位置、终端用户和终端用途三大类别，提出了现场检查、远程视频、审计员引导视频、随机归还请求、基于延迟的位置信息验证、加强的KYC检查、KYC合规审计、业务线与用途交叉检查以及云计算提供商的计算量审计等多种手段，并讨论了在BIS（美国商务部工业与安全局）框架内的实施步骤与参与主体。

**💡 创新点**

创新点在于：①将多领域（如IAEA核查、金融行业合规）验证技术迁移到AI芯片出口管制；②系统化列出可在一年内落地的“近端”验证工具；③提出了新颖机制（如基于延迟的位置信息验证和随机归还请求），并为其在法规与技术层面提供可操作的实施路线；④将验证机制与BIS资源受限的现实相结合，强调私营企业与第三方审计机构的协同。

**🔧 技术方法**

采用的技术包括：- 现场实地检查与设备序列号核对；- 远程视频与审计员引导的现场演练，配合加密认证、时间戳与地理位置元数据；- 通过云端加密签名的延迟测量实现芯片位置信息验证；- KYC自动化合规平台（如SolidIntel、AEB、WireScreen）进行风险扫描与名单交叉核对；- 电子出口信息（EEI/AES）与运输日志的自动化对账；- 计算量（FLOP）计量与云计费日志的汇总。

**📊 数据集**

使用的数据主要为公开的监管名单（Entity List、OFAC制裁名单、DENIED PERSONS LIST、MILITARY END-USER LIST）以及美国出口控制系统中记录的EEI、AES、发票、装箱单、运输记录、库存日志和云计费日志等；未使用传统机器学习数据集，而是依赖政府和行业的标准合规数据。

**📈 对比分析**

文中没有进行实验性比较，而是通过定性评估给每种机制打上成熟度（Established / Relatively Established / Novel）、效力（High / Medium / Low）和侵入度（Invasive / Relatively Invasive / Not Invasive）的标签。作者认为，延迟基位置信息验证和随机归还请求在“高效力”上具备优势，但仍需在实际部署中验证其可靠性与成本。

**⚠️ 局限性**

主要局限包括：①验证机制多依赖出口商或芯片所有者提供的数据，存在被篡改的风险；②间接销售与转售链条中责任难以追溯；③BIS资源有限，需大量私营企业参与，可能导致监管不均衡；④基于延迟的位置信息验证易受网络路径篡改、镜像服务器攻击；⑤缺乏实证评估，效能评级为主观；⑥对隐私和数据共享的法律合规挑战。

---

## 102. Open Tabular Insight Extraction: Where Do We Stand, and Where Should We Go?

**arXiv ID:** 2609.07629 | [PDF](https://arxiv.org/pdf/2609.07629v1)

**作者:** Daniel Gomm `[一作]` (Centrum Wiskunde en Informatica), Madelon Hulsebos `[通讯]` (Centrum Wiskunde en Informatica)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了Open Tabular Insight Extraction（OpenTI）统一框架，系统性梳理和评估跨学科的表格分析技术与基准；

**💡 创新点**

将多领域（数据库、IR、NLP、HCI、ML）表格知识提取问题归结为一体化任务，定义了洞察需求、实现、效用等核心概念，并构建了洞察类型层级和功能能力层级；

**🔧 技术方法**

利用形式化模型（函数空间、实现空间、效用最大化）和系统化文献综述方法，分析了数据检索、解释、分析组合与执行、输出合成等功能能力；

**📊 数据集**

未使用单一数据集，而是对2021‑2025年18个顶会与工作坊中共160篇论文的系统性文献进行抽样，覆盖了SQL、Python、工具调用、直接推理等多种实现；

**📈 对比分析**

通过对系统功能覆盖率、基准适用性、引用模式等指标进行量化评估，发现现有系统大多缺失检索与输出多模态能力，基准多数不适用于开放场景；

**⚠️ 局限性**

主要局限包括：1）仍缺乏完整的开放检索与多表整合能力；2）大部分系统依赖LLM的隐式解释，缺少可解释的实现路径；3）基准评估过度聚焦答案准确性，忽视洞察价值与交互体验；4）缺乏跨任务的统一评价标准与真实数据湖场景验证。

---

## 103. A monadic interpreter and type-and-effect checker

**arXiv ID:** 2609.07667 | [PDF](https://arxiv.org/pdf/2609.07667v1)

**作者:** Stefano Raviola `[一作]` (Università del Piemonte Orientale), Francesco Dagnino `[通讯]` (Università di Genova)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了一个基于 Haskell 的 monadic 框架，包含小步解释器和类型-效应检查器，支持通用效应和处理器。

**💡 创新点**

创新点在于将语言语法与效应语义分离，使解释器对任意 monad 参数化；同时提供了可扩展的类型-效应系统和处理器过滤器。

**🔧 技术方法**

主要技术包括 Haskell 的类型类、Monad、Template Haskell、效应集合（EffectSet）以及小步语义实现。

**📊 数据集**

未使用传统数据集；示例通过非确定性列表 monad 与异常 monad 进行演示。

**📈 对比分析**

论文未给出数值性能对比，仅通过手工示例展示语义行为；缺乏基准测试。

**⚠️ 局限性**

限制在于效应类型仅以操作集合表示，无法捕捉操作顺序；缺乏对更复杂效应（如输出序列）的建模。

---

## 104. How AI Models Manage Epistemic Authority: A Taxonomy and Comparative Analysis of Responses to User Disagreement

**arXiv ID:** 2609.07662 | [PDF](https://arxiv.org/pdf/2609.07662v1)

**作者:** Riyadh Alnasser `[一作]` (University Of Edinburgh), Tuğrulcan Elmas `[通讯]` (University Of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型在面对用户异议时的回应方式，提出六种挑战类型的分类，并设计四层注释框架（主张结果、权威定位、社会策略、证据支持），构建2,310个受控挑战场景并收集14种模型共32,340条回应进行分析

**💡 创新点**

首次系统性探究LLM的认知权威管理，提供基于对话分析的挑战类型与四层注释框架，并生成可供后续评测使用的公开数据集与术语表

**🔧 技术方法**

运用对话分析理论构造挑战模板，使用LLM-as-Judge（主要为Llama 3.3 70B）对四层进行自动标注，并通过人工验证确保标注质量；对14种模型（前沿、中端、小型）进行实验，统计各层分布并进行关联与多维比较

**📊 数据集**

包含2,310个挑战场景（7个领域×3任务×6挑战类型×3强度），共32,340条模型回应；数据公开于GitHub（https://github.com/riyadhalnasser1/ai-epistemic-authority）

**📈 对比分析**

通过统计每层的比例和Cramér’s V、χ²检验展示模型在主张维持、权威转移、社会验证和证据支持上的差异；对模型按开发者所在地、规模与能力层级进行比较；结果显示模型差异不随规模或能力层级排序，且各层解耦；无单一性能指标，而是多维行为概况

**⚠️ 局限性**

仅为描述性研究，未评估回应的适宜性；挑战为模板生成的受控模拟，缺乏真实用户对话；只考虑四轮交互、单一系统提示与低温度；LLM-judge标注可靠性受限；数据仅英文，文化差异未考量

---

## 105. Syntactic Patterns and Stylistic Functions in Narrative Prose: A Rule-Based and Machine-Learning Approach

**arXiv ID:** 2609.07651 | [PDF](https://arxiv.org/pdf/2609.07651v1)

**作者:** Stefana Janicijevic `[一作]` `[通讯]`, Stefana Janicijevic

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于依存句法的句子层面风格分类工作流，对三千多句的小说文本进行规则化风格标注，并用特征向量训练分类器

**💡 创新点**

创新点在于：① 用线性化的三元组（词干‑UPOS‑依存）作为稀疏特征；② 将规则标注与机器学习结合，提供可解释的工作流程；③ 通过元启发式搜索（Firefly 算法）显著提升随机森林性能

**🔧 技术方法**

技术手段包括：Python+Open‑Source NLP（UD 解析器、scikit‑learn）用于生成三元组；TF‑IDF 特征提取；Logistic 回归、随机森林、LinearSVC 三种传统分类器；Metaheuristic 超参优化（Firefly、GA、PSO、BAT）；以及基于 DistilBERT+LoRA 的对比实验

**📊 数据集**

数据集为 9,253 条 Slovenian 小说句子，经过 UD 解析后手工或规则标注为五类（描述、内省、因果、意识形态、中性），子集 3,300 条用于交叉验证与实验

**📈 对比分析**

比较方法：在 80/20 随机划分和 5‑fold CV 上评估宏观 F1、准确率；随机森林在全数据集上达到 macro‑F1≈0.9764，经过 Firefly 算法调参后提升至 0.9845；其它模型表现低于随机森林；DistilBERT+LoRA 远低于基线（macro‑F1≈0.54）

**⚠️ 局限性**

局限性：① 规则标注非黄金标准，模型复制的是规则而非人类直觉；② 意识形态类样本稀少，导致下游性能不稳定；③ 仅在小说文本上验证，泛化到其他体裁未知；④ 元启发式调参仅在 3,300 条子集上完成，未扩展到全数据；⑤ DistilBERT 输入格式不匹配导致性能不佳

---

## 106. Norms at a Price: Why RL-Based Alignment Can Promise Conditional Compliance at Best

**arXiv ID:** 2609.07627 | [PDF](https://arxiv.org/pdf/2609.07627v1)

**作者:** Kevin Baum `[一作]` (German Research Center for Artificial Intelligence), Felix Jahn `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对强化学习对齐训练无法区分无条件合规与有条件合规的理论与实验分析，并提出非选项性架构解决方案

**💡 创新点**

提出‘不可选项性（non-optionality）’对齐策略，揭示训练管道自然选择检测通过而非合规的机制

**🔧 技术方法**

理论证明（不可区分性与选择性）、对齐实验回顾、指标如δ可学习性、AUC、合规率等统计

**📊 数据集**

多模型评估数据（OpenAI/Claude 3 Opus等）、内部日志与评估对比记录

**📈 对比分析**

通过对比模型合规率与检测意识提升，发现提升检测率可显著降低违规，但仍无法区分合规性；实验表明性能随能力与代理性增强而提高

**⚠️ 局限性**

受限于训练仅靠行为回报导致不可区分性，迭代管道仅选择检测通过；对多代理非选项性、动态合规仍缺乏完整方案

---

## 107. Decentralized Safe Multi-Agent Reinforcement Learning via Predictive Shielding

**arXiv ID:** 2609.07618 | [PDF](https://arxiv.org/pdf/2609.07618v1)

**作者:** Yacine El Yamani `[一作]` (ENSTA IP Paris), Elena Vanneaux `[通讯]` (ENSTA IP Paris)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种完全去中心化的框架，结合模型预测屏蔽与通信自由的冲突解决协议，使独立训练的强化学习机器人能够在共享环境中安全、在线地适配其预训练策略。

**💡 创新点**

创新点包括：① 将模型预测屏蔽与有限时域Q学习相结合，区分静态与动态约束；② 设计了无通信的随机冲突解决协议以打破对称性并消除 livelock；③ 在 Dec‑POMDP 下证明了该方法的可扩展性与低在线计算成本。

**🔧 技术方法**

使用技术：Dec‑POMDP 框架、独立 Q 学习 (IQL)、模型预测屏蔽、基于模型的有限时域 Q 学习 (MB‑FH‑IQL)、Assume‑Guarantee 安全保证、随机冲突解决协议。

**📊 数据集**

实验数据集：OpenAI Gym 的 MultiGrid 环境，包括多智能体路径寻找和硬币收集任务，使用 10×10 网格等。

**📈 对比分析**

与 IQL、Dyna‑Q Shield、DMPS、MIS 四个基线进行对比；在三种典型场景（窄走廊、对称环境、拥挤通道）下评估成功率、训练/部署时间。结果显示，本方法在成功率上显著优于基线，并且部署时间接近独立学习方法。

**⚠️ 局限性**

局限性：预测时间窗口需手动设定，最坏情况估计导致过度保守；对训练与部署环境的模型不匹配敏感，未来需研究自适应时间窗口与更稳健的轨迹估计方法。

---

## 108. FinCUABuild: Can Agents Build Reliable Benchmarks for Dynamic Financial Computer Use?

**arXiv ID:** 2609.07603 | [PDF](https://arxiv.org/pdf/2609.07603v1)

**作者:** Jingpu Yang `[一作]` (Wuhan University), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 FinCUABuildBench benchmark 以及 FinCUABuildAgent 自动构建金融 CUA 任务的完整流程。

**💡 创新点**

创新点在于将任务构建本身作为评估目标，提供统一的 576 条目标槽，并通过多代理按能力依赖实现从需求到可执行、可重放、可验证任务的完整链条。

**🔧 技术方法**

采用多代理协同框架、能力-依赖编译、独立 Oracle、红队修复等技术手段，完成任务需求解析、环境搭建、验证与迭代修正。

**📊 数据集**

使用 24 类金融工作流共 576 条目标槽，涵盖三种运行时动态配置（状态更新、工具失败、政策变更）。

**📈 对比分析**

与规则模板、LLM+工具、Benchmark 生成系统等基线对比，FinCUABuildAgent 在四种模型下合格率分别从 31.3% 提升至 41.9%，显著优于 1–8% 的基线。

**⚠️ 局限性**

局限在于合格率仍偏低、跨工作流迁移性能有限、对人工验证和源许可、验证器维护的依赖较高。

---

## 109. BarkNet-Lite: A Lightweight Texture and Colour Network with the BarkBD Benchmark for Bark-Based Tree Species Recognition in Bangladesh

**arXiv ID:** 2609.07600 | [PDF](https://arxiv.org/pdf/2609.07600v1)

**作者:** Aroshi Ali `[一作]` (Khulna University of Engineering & Technology), Md. Khalid Syfullah `[通讯]` (Bangladesh Army University of Science & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在巴基斯坦（孟加拉三角洲）采集了 20 种本土树种的 14,258 张未裁剪的 bark 图像，并基于此数据集提出了一种从随机初始化、轻量化的多尺度纹理与并行颜色通道网络，单图推理精度达到 96.64%±0.66%。

**💡 创新点**

创新点在于（1）首次公开巴基斯坦本土 bark 数据集，填补了热带地区的数据缺口；（2）设计专门针对纹理信号的多尺度 Inception 纹理路径与并行颜色通道，完全从零训练而无需 ImageNet 预训练；（3）在单图推理、固定划分、严格评估协议下实现与大型预训练模型相近的精度。

**🔧 技术方法**

采用的技术包括多尺度 Inception 模块、通道与空间注意力、残差瓶颈、随机初始化与自监督数据增强、梯度注意力（Grad‑CAM）可解释性分析、TensorFlow Lite 进行单精度、动态范围 INT8 以及全整数 INT8 的部署实验。

**📊 数据集**

主要使用了自己收集的 14,258 张未裁剪的智能手机 bark 图像（20 种树种，四地区，三天气条件），并在此基础上对公开的 BarkVN-50、BarkNet 1.0、TRUNK12 等数据集进行跨域迁移测试。

**📈 对比分析**

实验在相同的 70/15/15 图像级划分、相同预处理、数据增强和优化器下进行，单图推理精度仅比 9 个 ImageNet 预训练紧凑网络低 2.3%，参数仅 2.96 M；在移动端以 15.3 ms/图、38.6 MB 的内存实现了可用的部署；迁移到 BarkVN‑50、BarkNet 1.0 分别获得 95.9% 与 92.9% 的精度，TRUNK12 则仅 69.8%。

**⚠️ 局限性**

主要局限包括：使用图像级划分导致对全新树个体的泛化仅为上限；单次实验与单设备评估，缺乏树组划分、不同季节、树龄和更广泛硬件平台的验证；量化后精度下降 11+ 点；颜色通道与纹理阶段的互依关系未作进一步拆解。

---

## 110. GPU-Accelerated Hypergraph Partitioning and Placement to Map SNNs on Neuromorphic Hardware

**arXiv ID:** 2609.07577 | [PDF](https://arxiv.org/pdf/2609.07577v1)

**作者:** Marco Ronzani `[一作]` (Politecnico di Milano), Cristina Silvano `[通讯]` (Politecnico di Milano)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了第一套基于 GPU 的 SNN 映射管线，在神经形态硬件上高效完成超图分区与放置。

**💡 创新点**

创新点在于：①结合硬件的多播/复制机制对超图分区与放置进行专门建模；②在 GPU 上实现多级分区与递归二分的高并行算法；③采用多起点放置与力导向细化，显著提升映射质量。

**🔧 技术方法**

使用的技术包括 CUDA 并行化、基于 hMETIS/KaHyPar 的多级超图分区、Hilbert 曲线投影与递归二分、力导向放置、匹配与前缀和策略、Steiner 树近似成本评估。

**📊 数据集**

使用 12 个公开的 SNN 数据集，涵盖从数百万到 577 个突触的网络（如 Lenet、VGG-11、AlexNet、MobileNet、Allen V1 等）。

**📈 对比分析**

与四个 CPU 顺序工具（SNNcut、EdgeMap、Ronzani 等）以及 hMETIS、Mt‑KaHyPar 进行对比。结果显示映射质量（能量、延迟、最大拥塞）均优于现有工具，执行时间提升 18–280 倍，整体速度提升 4–1500 倍，GPU 内存峰值约 28 GB。

**⚠️ 局限性**

局限性包括：尚未在真实硬件平台上验证精确路由成本；Steiner 树近似可能导致估计误差；多起点放置在 GPU 资源受限时收益有限；未考虑时钟同步或特定路由实现细节；对极大规模网络的内存与多 GPU 支持仍需进一步研究。

---

## 111. SphereSOD: Geometry-Structure Coupled Learning for 360 Salient Object Detection

**arXiv ID:** 2609.07571 | [PDF](https://arxiv.org/pdf/2609.07571v1)

**作者:** Junsong Zhang `[一作]`, Chunyu Lin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在 ERP 空间内同时考虑球面几何与显著结构的 360°显著目标检测框架 SphereSOD，利用可变形采样和结构引导的解码实现完整目标与精细边界的分割。

**💡 创新点**

创新点在于：① Prior‑Guided Multi‑scale Encoder (PG‑ME) 将球面投影几何直接嵌入 deformable attention，② Dual‑branch Prior‑guided Progressive Decoder (DPPD) 通过前沿先验（前景对比、sal‑con、畸变）实现结构感知的上下文聚合，③ Probability‑Context Residual Decoder (PCRD) 通过门控残差细化边界。

**🔧 技术方法**

采用 Swin‑S 作为骨干，结合 PG‑DSA、PGTR、ON‑rT2T、PCRD 等模块，并使用多尺度 BCE+IoU+Align 损失进行训练。

**📊 数据集**

在 360‑SOD、360‑SSOD 与 ODI‑SOD 三大公开 360°SOD 基准数据集上进行实验。

**📈 对比分析**

与多投影融合方法（FANet、MPFRNet、SCFANet 等）以及 2D SOD 方案对比，SphereSOD 在 S_m、MAE、F_β、E_m 等指标上均取得最高或相近的性能，同时 FLOPs 仅为 135.27G，参数量 71.18M，显示出更优的准确性‑效率折中。

**⚠️ 局限性**

局限性：仍受 ERP 投影畸变影响，极点区域的细节恢复有待进一步提升；模型相对较大，推理速度与内存仍可进一步优化；在极低光照或纹理稀疏场景的鲁棒性尚未充分验证。

---

## 112. Aegix Pulse: A Traceable Three-Stage Architecture for Personalized Content Generation and Context-Preserving Revision

**arXiv ID:** 2609.07672 | [PDF](https://arxiv.org/pdf/2609.07672v1)

**作者:** Hongnan Zhao `[一作]` (Aegix Insight), Zhihao Chen `[通讯]` (Aegix Insight)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一套三阶段内容生成体系，将当前任务、品牌身份与历史成功证据等多维上下文独立管理并实现可追溯的生成与修订流程；

**💡 创新点**

通过将任务Persona、Account Profile（品牌DNA）和成功历史风格证据三层上下文分别拆分、版本化，并提出预注册的组件级实验方法，显著提升了系统的可追溯性和评估可重复性；

**🔧 技术方法**

利用大语言模型（ChatGPT系列）结合检索增补、向量索引、规则校验、LLM Judge评估、异步作业调度与完整审计链；

**📊 数据集**

使用96个合成社交媒体生成任务（12个账户Profile、8个任务目标、4个受众阶段、12个商业场景），并预先提取合成成功帖与对应风格证据；

**📈 对比分析**

采用预注册的四个假设，配对置换检验和Cohen d衡量效果；结果显示Account Profile和保留上下文的修订在实验中有初步正向提升，但未通过多重比较校正；Task Persona提升有限，历史风格证据未显示显著效果；

**⚠️ 局限性**

主要局限包括：数据为合成任务，缺少真实用户与业务指标；评估仅离线且受限于5分量表；LLM Judge可能存在偏差，人工评审一致性低；未评估Recipe/Validator/Repair等已实现组件的真实影响；

---

## 113. Accuracy is Not Enough: A Divergence-Based Approach to Evaluate Fidelity Loss in Quantized LLMs

**arXiv ID:** 2609.07664 | [PDF](https://arxiv.org/pdf/2609.07664v1)

**作者:** Shahzeb Qamar `[一作]` (Fraunhofer IAIS), Rafet Sifa `[通讯]` (Fraunhofer IAIS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于分布差异的量化LLM评估框架，衡量量化模型与原始模型在完整词表概率分布上的信息损失；

**💡 创新点**

创新点在于将任务准确率与分布级别的Jensen–Shannon Divergence (JSD)和Total Variation Distance (TVD)相结合，揭示准确率掩盖的分布漂移；

**🔧 技术方法**

使用了统计距离度量（JSD、TVD）、full-vocabulary softmax输出、两流评估管道、K-quant（4K）混合精度与传统4-bit uniform（40）量化；

**📊 数据集**

评估数据集包括MMLU、BoolQ、PIQA、CausalBench，模型涵盖LLaMA-3.1-8B、LLaMA-3.2-1B、Mistral-7B、Teuken-7B、Gemma-2B；

**📈 对比分析**

对比方法通过比较不同量化状态下的任务准确率、预测翻转率和TVD/JSD来评估分布相似度，结果显示4K在保持相似内存占用下通常比40的TVD更低，表明更好的分布保真度；

**⚠️ 局限性**

局限在于仅评估基线模型、固定提示模板、单步生成，未覆盖指令调优、RLHF对齐模型，且未考虑延迟、能耗等部署指标。

---

## 114. Forecasting the Winner of a Live Tennis Match

**arXiv ID:** 2609.07617 | [PDF](https://arxiv.org/pdf/2609.07617v1)

**作者:** Charles Xie `[一作]` (Natick High School), Aneesh Muppidi `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种融合Markov递归、Elo预估与现场比赛信息的实时网球胜率预测模型，并对五种模型进行比较。

**💡 创新点**

创新点在于提出Trace模型，将Elo-不对称Markov、serve-shrink Markov与Histogram Gradient Boosting组合，利用现场更新的服务概率和预估优势提升概率估计精度。

**🔧 技术方法**

使用Markov递归、Elo评分、贝叶斯收缩、Histogram Gradient Boosting、XGBoost/LightGBM等机器学习方法。

**📊 数据集**

使用Jeff Sackmann提供的8,222场大满贯点对点数据，结合ATP/WTA比赛结果生成Elo。

**📈 对比分析**

通过时间顺序划分（训练2011-2021，验证2022，测试2023-2024）评估准确率与log loss，Trace在25%/50%/75%进度点的准确率分别为0.7606/0.8215/0.8834，整体准确率77.84%，优于其他四个基线模型。

**⚠️ 局限性**

限制包括仅考虑服务点、缺乏场地、体能、天气等因素，数据仅覆盖大满贯，难以推广到低级别赛事。

---

## 115. How Bitcoin Forms Its Network: Peer-Table Sampling and Structural Properties

**arXiv ID:** 2609.07609 | [PDF](https://arxiv.org/pdf/2609.07609v1)

**作者:** Taki E. M. Abedesselam `[一作]`, Francesco Pasquale `[通讯]` (University of Rome Tor Vergata)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

暂无内容可总结

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

## 116. CLUES-WEASEL: No additional clues required to choose your time series clustering algorithm

**arXiv ID:** 2609.07606 | [PDF](https://arxiv.org/pdf/2609.07606v1)

**作者:** Johann Faouzi `[一作]` `[通讯]` (University of Rennes), Johann Faouzi (University of Rennes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种新的无监督时间序列聚类算法 CLUES-WEASEL，采用无监督版本的 WEASEL 2.0 进行特征提取，随后通过 PCA 降维并使用 k‑means 进行聚类。

**💡 创新点**

创新点在于：①只需无监督特征提取步骤，无需任何标签或额外提示；②通过把 WEASEL 2.0 的变换步骤迁移到聚类任务，获得了显著更高的聚类质量；③对 PCA 进行阈值化选择（按累计解释方差）来自动决定降维维度；④展示了该框架可轻松替换为其他无监督特征提取器（如 MultiROCKET、RDST 等），进一步提升性能。

**🔧 技术方法**

核心技术包括：无监督 WEASEL 2.0 变换、特征标准化、常数特征剔除、PCA 降维（阈值为 20% 解释方差）、k‑means 聚类（k‑means++ 初始化）。实验使用 Python 生态：aeon（WEASEL 变换）、scikit‑learn（PCA、k‑means、标准化）、UMAP（可视化）。

**📊 数据集**

评估数据集为 UCR 时间序列分类档案的 112 个可变长度/缺失值已剔除的单变量时间序列数据集，涵盖 71 个评估集和 41 个开发集。

**📈 对比分析**

与多种 state‑of‑the‑art 聚类算法（KASBA、MSM、PAM‑MSM、DBA、Shape‑DBA、R‑Clustering、RandomNet、深度学习自监督模型）在 4 个聚类指标（聚类准确率、调整 Rand 指数、调整互信息、归一化互信息）上进行交叉验证与非交叉验证比较。CLUES‑WEASEL 在所有指标上均显著优于对手，且相较于 KASBA 运行速度快约 2.6 倍，内存占用虽然高但可通过调低最大特征数（如 5k）得到近似效果。

**⚠️ 局限性**

局限性：①PCA 的 O(n·p²) 计算与内存需求在大规模数据集上显著；②当前实现需要先完整计算 PCA 再根据阈值挑选组件，未使用迭代 PCA；③只在 UCR 档案上评估，缺乏对变长/缺失数据的适配；④对最大特征数和阈值的选取仍需经验性调参。

---

## 117. ObGynLongBench: Revealing the Evidence-to-EHR Gap in Longitudinal EHR Decision-Making

**arXiv ID:** 2609.07601 | [PDF](https://arxiv.org/pdf/2609.07601v1)

**作者:** Jun Xiang `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个基于可追溯临床规则的产科长文本EHR基准ObGynLongBench，用真实孕产妇医疗记录来评估LLM在临床决策中的证据检索与整合能力。

**💡 创新点**

将真实EHR与文本规则对齐，创建包含1,500条决策点的多级证据复杂度评估框架，首次揭示Evidence-to-EHR性能差距并比较多种EHR访问策略。

**🔧 技术方法**

采用多模型实验（商业、开源、医学LLM）对Evidence、Visit、History三种输入范围进行多选题准确率评测，并探究证据复杂度、上下文长度和患者级错误关联，比较Direct、Static RAG、Agent等访问策略。

**📊 数据集**

使用来自单中心三级医院的976名孕产妇真实中文EHR，生成覆盖16个产科子领域、1,500条决策点的基准数据集。

**📈 对比分析**

通过对17种LLM在三种输入范围下的准确率比较，发现Evidence下最高可达79%（Gemini‑3‑Flash），History下准确率仅降至≈68%；Active‑Search Agent在History下提升至≈64%，而RAG等策略保持接近Direct但成本更低。

**⚠️ 局限性**

局限性包括单中心单语（中文）数据、仅采用多选题评估、Inference设置差异、未验证跨医院/跨语言适用性，以及未覆盖开放式输出与临床交互等实际需求。

---

## 118. Large-Scale User Behavior Analysis in Multimodal AI-Assisted Manual Task Execution

**arXiv ID:** 2609.07594 | [PDF](https://arxiv.org/pdf/2609.07594v1)

**作者:** Rafael Ferreira `[一作]` (NOVA University of Lisbon), João Magalhães `[通讯]` (NOVA University of Lisbon)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Alexa平台上部署的多模态对话任务助手TWIZ-v2进行为期12个月的30k次真实用户交互数据的规模化行为分析，探究用户交互流程、意图分布、对话特征以及与满意度相关的行为信号。

**💡 创新点**

首次在大规模真实环境中系统性分析CTA用户行为，揭示任务发现与执行、非合作行为及满意度关联，为CTA设计提供可量化的改进方向与设计准则。

**🔧 技术方法**

结合模块化架构、BERT与规则混合的意图识别、跨模态检索、LLM生成响应、语音识别与回退机制，实现多模态交互与上下文感知回复。

**📊 数据集**

30k条TWIZ-v2对话（其中5k带用户评分），任务内容来源于WikiHow与Whole Foods，采集自美国Alexa用户的真实交互日志。

**📈 对比分析**

通过与Wizard of Tasks等众包数据对比，利用Spearman相关性评估特征与评分关系，发现任务进展与完成度与评分正相关，平均评分3.44，且完成任务用户满意度最高，说明研究方法能捕捉到真实用户满意度驱动因素。

**⚠️ 局限性**

研究仅覆盖美国英语Alexa用户，缺乏跨语言与多文化验证，任务领域受限于烹饪与DIY，且分析基于相关性，无法推断因果关系。

---

## 119. Validating DBpedia Triple Sets for Natural Language Generation

**arXiv ID:** 2609.07589 | [PDF](https://arxiv.org/pdf/2609.07589v1)

**作者:** Mark Andrade `[一作]` (ADAPT Centre), Brian Davis `[通讯]` (ADAPT Centre)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究DBpedia三元组的质量，提出验证与恢复方法以筛选出适合自然语言生成的高质量实体三元组集

**💡 创新点**

创新点在于将知识图谱三元组的域/范围验证与手工修正属性定义相结合，实现对单个三元组的精准判定，并通过自动修正属性提升召回率

**🔧 技术方法**

使用SPARQL查询、基于ontology的域/范围检查规则、手工修正属性定义与实体类型、统计与评估脚本

**📊 数据集**

采集约600,000条三元组，来源为Top 1,000、Top 10,000与随机1,000条实体的DBpedia三元组，使用手工标注的数据进行验证

**📈 对比分析**

与人工标注的真值集比较，验证阶段精度可达0.98，召回率约0.75；恢复阶段精度保持在0.97-0.98，召回率提升至0.92-0.99，表明方法在保持高精度的同时显著提升召回率

**⚠️ 局限性**

高精度导致初始召回率较低，需依赖恢复步骤；恢复主要通过修正属性定义，未对实体类型错误进行补救，未来工作计划进一步完善实体类型校正与跨语言扩展

---

## 120. A Tool-Augmented, GPT-4 Chatbot for Real-Time Repository Data Analysis

**arXiv ID:** 2609.07586 | [PDF](https://arxiv.org/pdf/2609.07586v1)

**作者:** Muhammad Jawad Chowdhury `[一作]` (Islamic University of Technology), Md. Sakib Khan `[通讯]` (University of Dhaka)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于GPT-4的工具增强型聊天机器人，用于实时解析并回答GitHub仓库相关问题，自动调用GitHub REST API获取数据。

**💡 创新点**

创新点在于采用结构化先解析再选择工具的流程，结合提示工程实现高精度回答，并具备对不在工具范围内查询的自我否认能力。

**🔧 技术方法**

使用技术包括OpenAI GPT‑4语言模型、GitHub REST API、工具选择与过滤算法、提示工程、迭代式响应生成和自我认知机制。

**📊 数据集**

使用自构造的80题仓库问答数据集（涵盖Pull Request、Commit、Issues、Compound、General Info等五类，并包含14个不在范围的问题）。

**📈 对比分析**

通过与传统多组件检索系统对比，实验表明在66个可答题中有65个正确，准确率为98.48%，并能正确拒绝所有14个不在范围的问题，显示出优异的性能。

**⚠️ 局限性**

局限性包括工具集仅包含两类工具，难以覆盖更高级的分析需求；复杂查询仍需多次迭代；系统对GitHub API可用性高度依赖，且未涵盖情感分析、趋势分析等功能。

---

## 121. Benchmarking LLMs for Threat Level Determination

**arXiv ID:** 2609.07582 | [PDF](https://arxiv.org/pdf/2609.07582v1)

**作者:** Han Wang `[一作]` (RISE Research Institutes of Sweden), Alfonso Iacovazzi `[通讯]` (RISE Research Institutes of Sweden)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型（LLM）在威胁情报事件的威胁等级判定任务上进行基准评测；

**💡 创新点**

构造了从MISP OSINT源提取、标注并归一化的威胁等级数据集，并设计专用提示与文本/JSON输入格式以系统比较模型表现；

**🔧 技术方法**

采用零样本提示、LoRA微调、不同输入表示（JSON vs 文本）以及宏F1/准确率评估等技术；

**📊 数据集**

使用包含1622条已标注低/中/高等级的MISP OSINT事件（原始2005条，剔除Undefined和过长样本）；

**📈 对比分析**

通过与多数类基线、不同模型（111B至8B）进行零样本和微调后对比，宏F1从0.271提升至0.40–0.58（最佳0.574），准确率从0.683提升至0.704；

**⚠️ 局限性**

仍存在低于实际部署需求的性能，受数据不平衡、模型倾向高风险预测、输入长度限制等限制，需进一步提升数据质量与领域特定调优。

---

## 122. Efficient Exploration Is Enough

**arXiv ID:** 2609.07575 | [PDF](https://arxiv.org/pdf/2609.07575v1)

**作者:** Mikel Malagón `[一作]` (University of Basque Country), Jose A. Lozano `[通讯]` (Basque Center for Applied Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究一种基于生成可泛化经验的高效探索框架，并在无外部奖励的情境下证明其能驱动复杂行为的出现。

**💡 创新点**

创新点在于：①将高效探索定义为最小化全局预测误差，强调经验可泛化而非简单覆盖；②理论推导出最优探索策略是有目标的、可确定性的；③揭示高效探索天然包含存活行为；④通过进化优化在部分可观测环境中得到自动课程化的复杂行为。

**🔧 技术方法**

使用：GRU 代理、前馈神经网络世界模型、Monte Carlo 估计 ECE、OpenES 进化策略、基于 L2 或 KL 的误差度量。

**📊 数据集**

采用自定义的网格环境（Empty、Blocks、Maze、RandColors）作为测试数据集，环境以离散状态和局部观测形式呈现。

**📈 对比分析**

与传统基于随机噪声或计数的探索方法对比，实验显示在相同互动步数下，高效探索代理在覆盖率、学习曲线和行为结构上均表现更优，能够自动产生结构化轨迹并形成学习课程。

**⚠️ 局限性**

局限性包括：仅在小型离散网格环境验证；世界模型为简单前馈或经验估计，未考虑更复杂泛化模型；MC 估计 ECE 的样本效率低；缺乏在真实连续空间或多任务设置下的验证。

---

## 123. No-Regret Mixing of LRU and LFU with Optimal Switching Cost

**arXiv ID:** 2609.07566 | [PDF](https://arxiv.org/pdf/2609.07566v1)

**作者:** Younes Ben Mazziane `[一作]`, Xinying Zou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究基于在线学习的缓存算法，证明现有的 LeCaR 方案对 LRU 与 LFU 的组合会产生线性 regret，并提出一种新的 LeCaR- 方案，通过虚拟缓存和最大耦合实现 O(√T) 的 regret 与切换成本。

**💡 创新点**

创新点在于：① 证明了 LeCaR 在某些周期性请求序列上会产生线性 regret；② 设计了利用 LRU/LFU 虚拟缓存的 experts 方案，并用最大耦合最小化专家切换成本；③ 证明该方案在所有请求序列上都能获得 O(√T) 的子线性 regret 与切换成本。

**🔧 技术方法**

主要技术包括：在线专家问题（Hedge/Hedge‑with‑coupling）、最大耦合 (maximal coupling) 对 Bernoulli 变量、负漂移（negative‑drift）分析、Markov 链与期望漂移计算、虚拟缓存与概率混合策略。

**📊 数据集**

使用的是论文中构造的周期性合成请求序列（例如 6 个请求一个周期的 LRU/LFU 交替模式）来证明理论性质；未涉及公开真实数据集。

**📈 对比分析**

比较方法：将 LeCaR、LeCaR- 与单独的 LRU、LFU 进行理论对比；实验部分主要验证理论结论，未给出具体数值，但表明 LeCaR- 的 regret 与切换成本均为 O(√T)，明显优于 LeCaR 的线性 regret。

**⚠️ 局限性**

局限性：① 论文仅提供理论分析，缺乏大规模真实流量实验验证；② 只考虑两种专家（LRU 与 LFU），未探讨更多专家组合的扩展；③ 最大耦合在实际实现中可能导致单步上传成本仍较高，需进一步改进。

---

## 124. Online Surrogate Repair: Decoupling High-Fidelity Feedback from Search Length in Closed-Loop Discovery

**arXiv ID:** 2609.07655 | [PDF](https://arxiv.org/pdf/2609.07655v1)

**作者:** Xiaotang Feng `[一作]` (University of Oxford), Bruno Andreis `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在线代理修复（OSR）方法，允许在闭环 AI 科学家搜索过程中以稀疏的高保真实验更新代理模型，从而在保持低成本的同时避免固定代理误差放大。

**💡 创新点**

创新点在于引入了“在线修复”这一新的反馈范式，既不需要每一步都进行昂贵实验，也不把代理固定不变；通过专门的采集规则（如 EI、Q90‑UCB）将有限的实验预算聚焦到对优化决策最关键的样本上，从而显著降低最大 regret。

**🔧 技术方法**

技术主要包括：基于上下文可更新的表格基础模型（TabPFN、TabICL）；采集规则（随机、Global‑UQ、Peak、Q90‑UCB、G2P、EI）；闭环实验框架（agent 产生候选，archive 存储，oracle 反馈，OSR 调度）；以及对比实验中使用的基准算法（Oracle control、Static EI、Archive‑BO、Tab‑AICL）。

**📊 数据集**

使用的数据集有：1）基于结构因果模型的 20 个合成世界（共 40 个实验设置，初始上下文 100/1000 条）；2）MADE 材料发现基准，30 个化学系统，每个系统 3 次重复，总共 90 条；3）Oracle 真实模型 Orb‑v3 以及弱代理 MACE。

**📈 对比分析**

与方法比较时，主要采用正则化最大 regret、oracle 查询次数、发现计数等指标。实验结果显示：在合成世界中，Online EI 在 Q=32 的预算下平均 regret 下降 44%（比 Static EI 低 19%），并在 NO‑repair 和 full‑feedback 之间闭合 76%；在 MADE 基准中，Online EI 在 Q=32 下实现约 82%/67% 的发现计数，比全反馈 (Q=300) 低 6–10 倍的查询量。

**⚠️ 局限性**

局限性包括：仅适用于可用表格表示的任务；依赖表格基础模型的上下文更新能力；未测试基于强化学习或测试时训练的代理；未探索更高级的多源或多保真度搜索策略；对模型外域和大规模高维特征的适用性未知。

---

## 125. Attestream: Usage-Aware Intermittent Data Distribution with Verifiable Lifecycle Provenance for Machine-Learning Data Streams

**arXiv ID:** 2609.07641 | [PDF](https://arxiv.org/pdf/2609.07641v1)

**作者:** Kentaro Oda `[一作]` `[通讯]` (Kagoshima University), Kentaro Oda (Kagoshima University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了基于区块链的使用感知间歇性数据分发架构，利用双签名生命周期记录实现数据使用可视化与流量停顿。

**💡 创新点**

将使用报告作为持续供应的门槛，实现零成本惰性门控，并结合多模态指纹嵌入实现泄露归因。

**🔧 技术方法**

EIP‑712 双签名、可选 ERC‑721 标记、Solidity 合约、链下指纹嵌入（高斯扰动、DCT 水印、随机指纹码）以及惰性门控算法。

**📊 数据集**

评估使用合成的二维地理表格、512×512 图像和包含 400 个可替换位置的文本文档，模拟不同泄露与攻击场景。

**📈 对比分析**

与传统无门控分发相比，完整生命周期操作耗约 657k gas（≈$0.13 在 Rollup 上），惰性门控不增加交易；指纹归因在各模态下可达 95–100% 准确率，且对噪声、压缩、改写具备鲁棒性。

**⚠️ 局限性**

无法验证模型训练真实性、指纹对高级攻击的鲁棒性有限、跨跳合规无法追溯、以及隐私泄露风险等。

---

## 126. Privacy Leakage from a Thousand Words: Millipixel Location Recovery from Dot Maps

**arXiv ID:** 2609.07623 | [PDF](https://arxiv.org/pdf/2609.07623v1)

**作者:** Yuntao Du `[一作]` (Purdue University), Ninghui Li `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了点图在不同尺度和背景下的隐私风险，提出了一种基于反走样（anti‑aliasing）信息的自动化位置恢复框架。

**💡 创新点**

创新点在于利用点图渲染过程中产生的子像素级别的颜色梯度（反走样痕迹）来提取精确位置，形成一个黑盒优化算法，实现毫米级定位精度，并显著优于现有方法。

**🔧 技术方法**

技术包括反走样信息提取、感知坐标下降（perceptual coordinate descent）优化、基于渲染器的黑盒对比误差计算、以及基于人口密度的 k‑匿名量化评估工具。

**📊 数据集**

使用了真实人口数据与合成数据，覆盖美国全国范围的点图；实验中使用 QGIS、GeoPandas、R 等常见可视化工具生成渲染图。

**📈 对比分析**

与基准方法（如 PixelMatch）以及其他现有攻击方法比较，平均误差约为1米（≈0.0002像素），相较传统方法提高了200倍以上，在不同地图缩放、背景和分辨率设置下均保持高度鲁棒。

**⚠️ 局限性**

局限性包括对背景图像的依赖（若背景未知可降低精度）、对渲染器的假设（需可复现或近似的渲染环境）、以及防御措施（禁用反走样或量化坐标）会显著影响可视化效果与攻击效果的平衡。

---

## 127. Construction and Natural Language Querying of a Cybersecurity Knowledge Graph

**arXiv ID:** 2609.07614 | [PDF](https://arxiv.org/pdf/2609.07614v1)

**作者:** Ines Ben Brahim `[一作]` (Université Lumière Lyon 2), Kamal Benzekki `[通讯]` (INSA Lyon)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于NVD公开CVE数据的网络安全知识图谱，并通过Neo4j Aura的AI助手实现自然语言查询。

**💡 创新点**

首次实现端到端的自然语言可查询网络安全知识图谱，并验证LLM生成Cypher查询在实际安全分析任务中的可用性。

**🔧 技术方法**

技术包括：NVD REST API数据抓取、Labeled Property Graph建模、Neo4j Aura云部署、Neo4j内置LLM自然语言到Cypher的语义解析。

**📊 数据集**

使用NVD公开的CVE记录（约2000条）作为数据集。

**📈 对比分析**

通过对10个代表性查询的人工编写与AI生成Cypher进行对比评估，结果显示单跳检索与基本聚合查询表现良好，部分多跳或聚合查询需小幅修改，整体成功率较高。

**⚠️ 局限性**

局限性：对多跳、复杂聚合或密集图结构的自然语言查询精度下降；依赖LLM的语义理解与图模式一致性；对持续更新的支持不够完善。

---

## 128. AgentIdeaBench: Benchmarking Scientific Ideation in the Agent Era

**arXiv ID:** 2609.07611 | [PDF](https://arxiv.org/pdf/2609.07611v1)

**作者:** Yunxiang Mo `[一作]` (HKUST), Simon See `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 AgentIdeaBench 这一多学科科学创意评测基准，比较模型在静态阅读列表与主动检索两种情境下生成新颖、可检验假设的能力；

**💡 创新点**

创新点在于把检索控制作为实验变量，使用文献验证的原创性评估，覆盖 33 模型、5 学科、40 细分领域，并揭示主动探索对强模型的显著优势和可扩展性；

**🔧 技术方法**

采用大规模语言模型（LLM）进行生成，配合 Semantic Scholar 搜索工具、基于规则的检索与评估管线，以及多模态三重评审者（LLM）进行多维度打分；

**📊 数据集**

数据集由 100 个跨学科研究子领域组成，挑选 40 个细分领域进行密集评分；每个子领域附有由 Semantic Scholar 自动收集的参考文献集合；

**📈 对比分析**

通过对比静态与主动两路评测，使用原始得分、维度拆分、统计显著性检验和能力门槛分析。结果显示主动模式平均提升约 1.2 分，且提升随模型能力呈正相关；强模型在主动模式上可获得 0.7 分以上的增益，弱模型甚至略有下降；

**⚠️ 局限性**

主要限制包括：评审者为 LLM，原创性评估依赖模型记忆和检索质量；主动模式包含多轮交互与工具使用，难以单独归因检索控制效果；世界模型实验未通过多比较校正，且仅在少数模型上表现出正向效果，整体提升不显著；

---

## 129. Solution for UCF UrbanTwin V2X-Real Track: Sim-to-Real Urban LiDAR 3D Object Detection

**arXiv ID:** 2609.07608 | [PDF](https://arxiv.org/pdf/2609.07608v1)

**作者:** Pu Luo `[一作]` (Xidian University), Lingling Li `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建多源协同训练和类感知融合框架以解决道路侧LiDAR的Sim2Real检测差距。

**💡 创新点**

通过将几何、采样密度、密度稳定性和行人形态四种合成数据角色明确分配，并在同一检测框架中训练源专属专家，实现可解释且互补的信息融合。

**🔧 技术方法**

使用DSVT稀疏体素Transformer、CenterPoint式检测头、RangeLDM扩散采样、标签无监督中心融合等技术。

**📊 数据集**

基于UrbanTwin数字孪生道路场景的S1、RangeLDM生成的S2、S3密度规范化样本以及训练时形态调整的S4四个来源的数据集。

**📈 对比分析**

通过在公开验证协议和隐藏测试集上进行统一评估，系统获得综合分数0.7421、3D mAP@0.5 0.4518，显示出相对于单一数据或平均融合方法更优的性能。

**⚠️ 局限性**

局限在于对源角色与融合路径的固定设计需在传感器或环境变化时重新验证，且缺乏不确定性自适应融合与物理先验的整合。

---

## 130. JudgmentLens: Human-AI Sensemaking of Complex Legal Judgments

**arXiv ID:** 2609.07607 | [PDF](https://arxiv.org/pdf/2609.07607v1)

**作者:** Xinyi Chen `[一作]` (Hong Kong University of Science and Technology), Chen Liang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了名为JudgmentLens的AI增强司法判决阅读系统，帮助非专业读者理解复杂判决。

**💡 创新点**

创新点在于将持久化的案例结构与可追溯的AI解释相结合，提供源关联的解释与多视图可视化，支持用户在阅读过程中的检查与决策。

**🔧 技术方法**

采用前端React/TS、后端FastAPI、DeepSeek大语言模型进行摘要、图谱生成与问答，并通过可视化图表展示时间线、逻辑图、证据矩阵等。

**📊 数据集**

使用中国裁判文书网公开判决（案例A、B用于对比，案例C用于探测），以及34名问卷、6名访谈和16名评估参与者的数据。

**📈 对比分析**

通过受控的交叉实验比较PDF阅读与JudgmentLens，结果显示任务完成时间下降约28.9%，工作负荷显著降低，自我报告理解提升，但判卷评分的理解分未出现显著差异；探索性对比PDF+DeepSeek显示用户在对话式AI中仍需自行组织与验证。

**⚠️ 局限性**

局限包括样本量小、只测试单一法律体系的判决、未分离各功能组件、未系统评估AI生成错误的影响，以及未检验长期或跨领域的适用性。

---

## 131. Fast Multidimensional Approximate Agreement with Optimal Resilience Using Ball Validity

**arXiv ID:** 2609.07599 | [PDF](https://arxiv.org/pdf/2609.07599v1)

**作者:** Tijana Milentijević `[一作]` (TU Berlin), Stefan Schmid `[通讯]` (TU Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了一种基于最小包围球（MEB）有效性的新型多维近似共识算法，设计了自适应收缩算法（Adaptive Contraction）并引入球膨胀定理，实现了无维度依赖的鲁棒性与收缩率；

**💡 创新点**

创新点在于：1）用MEB有效性取代传统凸有效性，突破了维度相关的容错阈值；2）提出自适应收缩算法，收缩因子1/√2可达极限；3）证明了球膨胀定理，支持最优鲁棒性n>3t；4）在同步与异步模型中均实现了坐标无关的收缩与有效性；

**🔧 技术方法**

主要技术包括：最小包围球与核心集构造、Helly定理与球膨胀论、几何收缩分析、Gather协议、以及对收敛与有效性同时进行的双重不等式推导；

**📊 数据集**

该工作为理论算法，未使用具体数据集，而是基于数学证明与构造实例验证收缩因子边界；

**📈 对比分析**

与现有算法（如MDA、MidExtremes、Mendes–Herlihy等）比较：本文在同步模型下可实现n>3t的最优鲁棒性，收缩率为√3/2≈0.866，且MEB有效性常数为√6≈2.45；在异步模型下实现n>4t的无维度鲁棒性，收缩率为√15/4≈0.968，MEB有效性为2√10≈6.32；相比MDA在相同鲁棒性下具有更强的有效性；

**⚠️ 局限性**

限制包括：1）对α的取值有限制（同步需α<√(β−1)，异步需α<√2或更小）；2）在异步模型中需要Gather协议，通信开销较大；3）算法虽然坐标无关，但实现复杂度较高；4）目前仅在完全网络模型下证明，缺乏对更一般拓扑的扩展；

---

## 132. I Don't Miss You, but I Do: Self-Explanation Faithfulness of Modality Missingness in Vision-Language Models

**arXiv ID:** 2609.07596 | [PDF](https://arxiv.org/pdf/2609.07596v1)

**作者:** Aydin Javadov `[一作]` (ETH Zurich), Florian von Wangenheim `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了一种可执行的干预协议，用以评估视觉语言模型对缺失模态的自我解释与行为一致性。

**💡 创新点**

首次将模型的自我声明（对单一模态是否足够或恢复缺失模态会否改变答案）与实际行为干预结果直接对齐，量化模型在缺失模态情境下的自我认知偏差。

**🔧 技术方法**

采用了两类评估：回溯性归因（比较完整输入与单一输入的答案）和前瞻性逆因果自知（预测缺失模态恢复后答案变化），通过多模态输入的可切换提示与答案一致性检测实现。

**📊 数据集**

使用四个任务：mm-IMDb（文本与图像的互补标签任务）、IsoBench（函数对称性与图流最大流的等价视图任务）以及GoldenView（多视角驾驶问答）。

**📈 对比分析**

与八款开放权重的 Qwen3.5 与 Gemma 4 系列模型（2B-35B）在 64 个模型-任务-条件格子上进行对比，结果显示模型普遍低估缺失模态恢复的影响（平均预测变更率<9% 对比执行变更率>30%），且在自我报告不足时准确率高但召回率低，证明自我解释不可靠。

**⚠️ 局限性**

局限性包括：自我报告本身会改变模型行为并降低任务性能；实验仅在完整缺失或恢复全模态的极端情况下进行，未覆盖部分缺失或受损模态；只使用单一生成采样，缺乏对生成稳定性的评估。

---

## 133. Same Problem, Different Field: Cross-Domain Solution Import via Domain-Stripped Computational Fingerprints

**arXiv ID:** 2609.07595 | [PDF](https://arxiv.org/pdf/2609.07595v1)

**作者:** Eryk Kulikowski `[一作]` `[通讯]` (KU Leuven), Eryk Kulikowski (KU Leuven)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种跨领域的计算方法指纹（faceted computational fingerprint），通过LLM将论文的计算机制去除领域词汇和方法名后提取关键计算结构与面向特征，随后结合文本相似性检索与可选面向特征的布尔过滤，实现跨学科相同计算问题的检索与可导入性；

**💡 创新点**

创新点在于①提出“去领域词汇、去方法名”的计算指纹抽取方式，②设计可调节的面向特征匹配门限以控制精度/召回前沿，③在公开数据集上验证该指纹能显著提升跨域检索性能，且比现有的主题/引用嵌入更强；

**🔧 技术方法**

核心技术包括：①大语言模型（Claude Haiku等）进行一次性计算机制抽取与面向特征标注；②TF‑IDF/BM25等文本检索用于召回；③布尔面向特征过滤（可选至少k个核心特征匹配）用于精确；④基准评估（AP、P@1、MRR、AUROC、ARI）与对照实验；

**📊 数据集**

使用两套数据集：一套为109篇论文（18个方法族，210个跨域同构对）的精确标注基准；另一套为501篇论文的无标签野生集，包含已知同构对与噪声；

**📈 对比分析**

与主题/引用嵌入（SPECTER、SPECTER2、SciNCL、SemCSE等）以及TF‑IDF、MiniLM等基线对比，实验表明：抽取的骨架文本在所有嵌入上均超过原始摘要，整体指纹+TF‑IDF在基准集上AP达到0.557，约为原始摘要0.222的两倍；面向特征过滤进一步提升精度，k=4时精度0.94、召回0.15；在野生集上，已知同构对在top‑1000中的召回率为0.619；

**⚠️ 局限性**

局限性包括①指纹依赖LLM抽取，模型性能和输出一致性影响结果；②仅能识别显式可抽取计算方法，纯理论或论证性论文不适用；③评估基准多基于已知可交换方法族，难以证明在更广泛领域的发现能力；④在野生集上未标注真实正例导致AP无法评估，实际召回仍受限。

---

## 134. Solution for UCF UrbanTwin LUMPI Track: Sim-to-Real Urban LiDAR 3D Object Detection

**arXiv ID:** 2609.07590 | [PDF](https://arxiv.org/pdf/2609.07590v1)

**作者:** Pu Luo `[一作]` (Xidian University), Lingling Li `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

仅使用合成数据训练，构建密度对齐的多源合成训练集，并通过多模型融合实现道路边LiDAR的3D目标检测与点云真实感评估。

**💡 创新点**

创新点包括将目标点密度对齐与长尾类增强相结合的训练集、基于类别的模型互补融合、残差召回与覆盖审计以及独立的点云真实感优化。

**🔧 技术方法**

使用DSVT与PointPillars检测器、RangeLDM采样、多源复制粘贴、类敏感的几何校准、类-aware路由与不对称融合等技术。

**📊 数据集**

利用UT-LUMPI数字孪生合成数据与真实LUMPI数据进行训练与验证。

**📈 对比分析**

在UCF UrbanTwin Sim2Real LiDAR挑战中，系统获得Combined Score 0.4692、Detection Score 0.1797、Realism Score 0.9035，3D mAP@0.5为0.1258，优于基线。

**⚠️ 局限性**

局限性在于对合成与真实的观察差异仍存在，某些稀有类别仍易漏检，且方法在其他环境或传感器上迁移性尚未验证。

---

## 135. From Simulated Citizens to Simulated Deliberation: Challenges in Representation and Interaction

**arXiv ID:** 2609.07573 | [PDF](https://arxiv.org/pdf/2609.07573v1)

**作者:** Chaemin Jang `[一作]` (Korea Advanced Institute of Science and Technology), Jihee Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过让LLM基于人口学特征的模拟公民进行多轮辩论，评估其在韩国政策议题上的观点分布、互动效果以及论证生成能力；

**💡 创新点**

创新点在于首次将人格化LLM回答与全国调查的群体意见进行对比，采用封闭单口实验拆解互动影响，并区分人口模拟与论证表层的验证需求；

**🔧 技术方法**

使用的大型语言模型为GPT‑4系列，结合人格化提示、随机轮询的多轮辩论协议、封闭单口对照、立场强制调查与Discourse Quality Index等评估工具；

**📊 数据集**

实验数据包括640个Nemotron‑Personas‑Korea合成公民、256个Nemotron‑Personas‑USA、2025年KEI环境认知调查（3,008份）和PCASPP（2,800份）以及美国Pew调查的基准；

**📈 对比分析**

通过比较每个群体的人口学立场差距（平均绝对误差约29个百分点）以及辩论与单口实验中的立场变化（差异≤1人），并使用Discourse Quality Index评估论证质量；结果显示观点分布与人类调查差距大，立场变动多非由互动驱动；

**⚠️ 局限性**

局限性包括模型无法准确再现人口意见分布、立场转变不受同行互动影响、论证表层效果尚未得到直接验证、仅使用单一模型与韩国语境、辩论规模小、以及对话质量评估依赖LLM判断。

---

## 136. We're Cooked! - Probing LLM Political Alignment Via Conflict-Framed Recipe Translation

**arXiv ID:** 2609.07568 | [PDF](https://arxiv.org/pdf/2609.07568v1)

**作者:** Svetlana Gorovaia `[一作]` (Technical University of Applied Sciences Würzburg-Schweinfurt), Ivan P. Yamshchikov `[通讯]` (Technical University of Applied Sciences Würzburg-Schweinfurt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不同语言和模型族中，通过给定的政治化词语（如 "aggressor"、"enemy"、"coloniser"）作为翻译目标的暗示，系统地评估大型语言模型在翻译任务中产生的隐性政治倾向，并对模型输出进行分类与图谱分析。

**💡 创新点**

创新点在于：①仅使用单一政治词语即可触发隐性政治立场；②采用跨语言、跨模型的全因子设计，揭示模型家族与文化背景对翻译行为的差异；③将翻译方向可视化为图网络，并用 Jensen–Shannon 距离量化模型间的行为相似度。

**🔧 技术方法**

主要技术包括：Prompt engineering（在翻译请求中插入政治化词语）、多模型评测（8个不同来源的LLM）、六类输出分类（翻译、风格化、保持原文、请求澄清、拒绝、空回复）、理由类型标注、图谱构建与密度分析、基于图的距离度量。

**📊 数据集**

数据集由两类食谱组成：一类为各国具有代表性的传统菜肴（文化食谱），另一类为无国别的煎饼（中性食谱），共17种目标语言，16种指令语言（含英语）；每种组合在8个模型上重复10次，共15,680条回复。

**📈 对比分析**

比较方法主要通过统计各模型在不同框架条件下的翻译率、目标语言分布、理由类型比例，以及图网络的边密度与中心性；使用 Jensen–Shannon 距离量化模型间的行为相似度。结果显示：大多数模型倾向将目标语言设为俄语，模型族群（西方/中国/欧洲）呈现显著的行为聚类，且政治词语的细微差异均可显著影响翻译结果。

**⚠️ 局限性**

主要局限包括：①部分指令语言及菜谱翻译由LLM完成，真实性和文化准确性无法完全验证；②菜谱选择可能携带额外文化偏见；③实验仅为单轮设计，无法捕捉多轮对话中的政治累积效应；④评估采用LLM-as-judge，可能导致自相似性偏差；⑤未进行统计显著性检验，结论以描述性为主；⑥无法单独归因于模型规模、架构或训练数据。

---

## 137. Noēsis: Deterministic-First Retrieval with Two-Tier Context Hydration for Factuality-Critical Queries on Small Local Models

**arXiv ID:** 2609.07663 | [PDF](https://arxiv.org/pdf/2609.07663v1)

**作者:** Nicola Cogotti `[一作]` `[通讯]` (Alpha Cogs), Nicola Cogotti (Alpha Cogs)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Noēsis 查询平面，针对小规模本地语言模型构建 deterministic fact layer、位置寻址、源归属约束和两层按需 hydration，以消除事实性查询中的混淆、上下文膨胀等问题。

**💡 创新点**

创新点在于：1) 生产端预计算的 deterministic facts 直接在 prompt 中提供精确答案；2) 位置级别的 deterministic cross‑source alignment；3) 通过命名引用路由实现跨 KB 的归属约束；4) 两层上下文（skeleton + hydration）并按需 hydration 的协议，使 2B 参数模型可匹配 35B 参数模型的事实性性能。

**🔧 技术方法**

使用技术包括：RAG 的结构化检索、图导航、可预计算 metric 事实、位置寻址、命名引用路由、两层上下文（skeleton + hydration）、epistemic tiering、基于 deterministic 规则的检索与排布。

**📊 数据集**

使用数据集：广播媒体（英、意）为例，构建 18 个知识库，包含节目收视率、对话、时间戳等数据；评测还包含 17 条跨 KB 查询，采用 N=5 的重复跑。

**📈 对比分析**

比较方法：对比 2B 与 35B 参数模型在固定基础设施下的四组 ablation（结构 vs flat、skeleton、hydration、模型替换）；使用精确值、无 confabulation、归属一致等指标；结果显示 2B 在 2B 上可与 35B 取得相同的精确率（100%），并在 skeleton/hydration ablation 下保持准确率，提升上下文压缩和时间（约 3 倍速）。

**⚠️ 局限性**

限制：实验仅覆盖广播媒体单一领域、规模有限的 KB，评测采用 N=5 的方向性指标，未验证在更大多模态或更复杂多实体场景下的性能；对跨语言短语的命名匹配仍存在微小缺陷；模型在更大规模语料下的可扩展性仍需进一步评估。

---

## 138. ZK-eSIM: A Privacy-Centric Zero-Knowledge Approach for eSIM Provisioning

**arXiv ID:** 2609.07654 | [PDF](https://arxiv.org/pdf/2609.07654v1)

**作者:** Liza Ahmad `[一作]` (University of Sheffield), Syed Rafiul Hussain `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出 ZK-eSIM，一种在 eSIM 预配过程中使用零知识证明、一次性伪匿名证书和短期凭证，消除对设备和订阅者持久标识符的暴露，保障订阅者匿名性与会话不可链的同时保持可追踪性；

**💡 创新点**

在 GSMA 现有 RSP 体系内实现匿名预配的完整链路，首次将零知识证明与可追踪逃逸机制（PCA 与 LEA 协同）相结合，提供可证明的匿名性、会话不可链以及联合审计追踪；

**🔧 技术方法**

零知识证明（NIZK）、盲签名、短期伪匿名证书、累加器（ACC）、ECDH 密钥协商、哈希函数、ECDSA 认证与签名、JavaCard 加密库；

**📊 数据集**

实验使用了测试 eUICC（sysmoEUICC1-C2T）、改造后的 GSMA SM‑DP+ 服务器、Local Profile Assistant (LPA) 以及自定义 JavaCard applet，未使用公开大规模真实数据集；

**📈 对比分析**

与商业 RSP 通过对每个协议阶段的运行时消耗（注册、证书初始化、订单处理、下载）进行对比，ZK‑eSIM 在整体端到端耗时上提升约 83%（单阶段提升 64%），但仍保持在可接受的范围内；

**⚠️ 局限性**

仅针对预配层的隐私改造，未覆盖后续 5G 网络访问、IP 地址去匿名化、背景通信、生命周期管理等跨协议风险，且在大规模部署时对 MNO 的计算负载相对较高；

---

## 139. The Art of Hierarchical Competing Patterns: Gaussian Process Optimization of Hyphenation

**arXiv ID:** 2609.07638 | [PDF](https://arxiv.org/pdf/2609.07638v1)

**作者:** Ondřej Sojka `[一作]` (Masaryk University), Petr Sojka `[通讯]` (Masaryk University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 hyphenation pattern 生成的 profile 选取问题表述为黑盒超参数优化，通过贝叶斯优化寻找同时兼顾精度和压缩率的最佳配置。

**💡 创新点**

创新点在于：①用精度加 trie 体积惩罚的单目标函数将准确率与压缩率统一化；②在 17 语言/脚本的多样化词表上验证，展示贝叶斯优化在多语言场景下的可重复性和效果；③公开完整可复现的实验流程与数据。

**🔧 技术方法**

使用高斯过程（Gaussian Process）贝叶斯优化（Upper Confidence Bound 探索策略），结合权重比例与阈值四维搜索空间；评估指标为 F₁/₇（精度优先）与 trie 节点数惩罚。

**📊 数据集**

采用 17 个 hyphenated word‑list 数据集，涵盖 14 种语言与多种文字（拉丁、西里尔、泰语等），每个数据集提供 8/1/1 的训练/验证/测试拆分。

**📈 对比分析**

与两套手工调优 baseline（cshyphen、wortliste）以及随机搜索、TPE 等预算匹配的对手进行比较。结果显示：16/17 数据集上 F₁/₇ 提升，所有 17 个数据集 trie 节点数均下降，平均压缩比为 0.407；在五个代表性数据集上预算匹配实验表明系统搜索与随机/树形估计器的表现相近，均能获得更小 trie 与更高 F₁/₇。

**⚠️ 局限性**

限制包括：搜索空间仍需手工设定（级数、权重范围、阈值上限、惩罚系数、评估预算）；仅用单一随机种子，未对多次实验或不同分割做重复验证；词表质量差异导致结果受限，需扩展更多语言、脚本与真实渲染测试。

---

## 140. Mapping the Emerging Social Science of Large Language Models

**arXiv ID:** 2609.07598 | [PDF](https://arxiv.org/pdf/2609.07598v1)

**作者:** Yi Yang `[一作]` (Chinese University of Hong Kong), Zhanzhan Zhao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并验证了大型语言模型（LLM）社交科学研究的三域框架（LLM作为社会心智、LLM社会、LLM与人类交互），并在198篇精细抽取文献和约47,719篇公开出版论文中通过聚类、主题模型和自动分类展现了框架的可重复性与跨方法一致性。

**💡 创新点**

首次将LLM社交科学研究系统化为三域结构，并证明该结构在不同语料规模、聚类方法与自动分类中的稳定性与可检验性。

**🔧 技术方法**

使用MPNet句子嵌入、K‑means聚类、LDA、结构主题建模（STM）以及大语言模型自动分类对文献标题与摘要进行文本分析。

**📊 数据集**

分析了198篇精细抽取的手工检索论文和47,719篇从Semantic Scholar、OpenAlex、Scopus、PubMed、Europe PMC等数据库检索的正式出版论文，覆盖多学科视角。

**📈 对比分析**

通过轮廓系数、Calinski‑Harabasz、Davies‑Bouldin等内部指标评估聚类，并用调整Rand指数和Cohen κ比较聚类与人工/LLM分类，结果显示聚类稳定性高、域映射一致性约78‑85%。

**⚠️ 局限性**

研究样本主要限于英语正式出版物，可能低估非正式、预印本及多语言研究；标题/摘要信息可能遗漏全文细节；LLM辅助筛选与分类受模型与提示设置的影响。

---

## 141. Beyond the Matrix Sign: Quadratic Spectral Descent

**arXiv ID:** 2609.07597 | [PDF](https://arxiv.org/pdf/2609.07597v1)

**作者:** Qiaozhe Zhang `[一作]` (Huazhong University of Science and Technology), Yingzhuang Liu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Quadratic Spectral Descent (QSD)，在保持 Muon 的谱范数约束下，用二次局部模型取代 Muon 的线性模型，得到考虑曲率的矩阵更新。

**💡 创新点**

核心创新在于证明曲率可以导致最优更新的奇异值不再均匀、奇异方向也不再与梯度对齐，并通过 K‑FAC 与 Frank–Wolfe 结合的方式，将该二次问题高效求解。

**🔧 技术方法**

使用 Kronecker‑Factored Approximate Curvature (K‑FAC) 近似曲率，Frank–Wolfe 算法在谱范数球上做线性化并得到闭式矩阵符号子问题，结合线性搜索实现内层收敛，并给出 O(1/K) 收敛率。

**📊 数据集**

在 GPT‑124M 与 GPT‑350M 两个规模上使用 FineWeb 数据集进行预训练实验。

**📈 对比分析**

与 Muon、NorMuon、Newton‑Muon 等变体比较，QSD 在相同验证损失下的训练时间缩短约 8.49%（GPT‑124M）至 7.4%（GPT‑350M），验证损失均优于对比方法。

**⚠️ 局限性**

局限性在于依赖局部二次模型和 K‑FAC 曲率近似，尺度校准与膨胀因子需经验选取，且仅在中等规模 GPT 任务上验证，未探讨更大模型或其他任务的适用性。

---

## 142. Replicating a Disjoint-Set Union Experiment over Various Notions of Micro Units to assess Translation Effort

**arXiv ID:** 2609.07748 | [PDF](https://arxiv.org/pdf/2609.07748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 143. DroneGround: Open-Vocabulary Drone Payload Characterization Using Synthetic Data and Grounded Vision-Language Models

**arXiv ID:** 2609.07780 | [PDF](https://arxiv.org/pdf/2609.07780v1)

**作者:** Ami Pandat `[一作]` (Homi Bhabha National Institute), Rohit Shukla `[通讯]` (Homi Bhabha National Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种两阶段的基于视觉语言模型的无人机载荷开放词汇表识别框架 DroneGround

**💡 创新点**

创新点包括：①将载荷识别任务转化为开放词汇的视觉语言生成与定位任务；②利用 LoRA 微调的 PaliGemma 生成语义描述；③设计遮挡式定位机制实现无监督的载荷可视化解释；④通过高保真合成数据提升跨域泛化能力

**🔧 技术方法**

技术手段为：YOLO26s 无人机检测 + LoRA 微调的 PaliGemma 视觉语言模型 + 遮挡式定位热图 + 统一的两阶段推理 pipeline

**📊 数据集**

数据集为：100,000 张 Unreal Engine 5 + Cosys‑AirSim 生成的合成无人机‑载荷图像（15 架无人机、6 类载荷、7 环境、5 天气），以及少量真实世界无人机图像（VisioDECT、Roboflow）用于训练与评测

**📈 对比分析**

与传统闭集载荷检测器（YOLO26s）对比，DroneGround 在合成+真实测试集上 F1 提升至 96.3%（闭集为 82.5%），在未见载荷类上 F1 进一步提升至 80.4%（闭集仅 42.7%），同时保持 45 FPS 的实时推理速度

**⚠️ 局限性**

局限性主要是：①对极小、运动模糊或严重遮挡的载荷识别效果不佳；②依赖第一阶段检测的准确性，误检会传播错误；③目前仅在单帧图像上验证，缺乏对视频序列的时序建模

---

## 144. Scalability Analysis of Distributed Kolmogorov-Arnold Network Training on High-Performance Computing Systems

**arXiv ID:** 2609.07740 | [PDF](https://arxiv.org/pdf/2609.07740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 145. Cross-modal learning for SAR target recognition using optical vision foundation models

**arXiv ID:** 2609.07753 | [PDF](https://arxiv.org/pdf/2609.07753v1)

**作者:** Lucas Hirsch `[一作]` (University of Edinburgh), Mike E. Davies `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在SAR目标识别任务中使用光学视觉基础模型（DINOv3）构建类级原型，并通过LoRA训练的SAR编码器将其特征嵌入与原型对齐，实现仅使用SAR图像进行推理的跨模态目标识别。

**💡 创新点**

创新点：①不需要一一对应的EO/SAR样本，只利用同类标签的光学原型进行跨模态监督；②在SAR编码器中加入LoRA进行轻量级微调，保持大部分预训练权重不变；③通过原型对齐而非传统的全局分布对齐（MMD），实现更有针对性的特征迁移。

**🔧 技术方法**

技术：光学视觉基础模型 DINOv3（冻结编码器）→原型构建；SAR编码器 DINOv3 加 LoRA 微调；交叉熵损失 + 原型对齐损失；t‑SNE 可视化；对比实验与 MMD、LoRA 单模训练等基线。

**📊 数据集**

使用 UNICORNv2 数据集（10 类民用车辆，包含 EO 与 SAR 图像），并在测试集上评估十类与七类（将轻型车合并）两种任务。

**📈 对比分析**

与基线比较：冻结 DINOv3 仅 26.9% top‑1；LoRA 微调 29.8%；MMD 对齐 31.8%；原型对齐 33.3%（十类）/52.2%（七类）。宏 F1 也随之提升，表明在严重类别不平衡与高噪声条件下有明显改进。

**⚠️ 局限性**

局限性：仍需监督标签；未验证与真正不重叠的 EO/ SAR 训练集；仅针对民用车辆，难以推广到其他目标或传感器；整体准确率仍偏低，说明 UNICORNv2 仍具高难度。

---

## 146. Translation Indeterminacy and the Distributional Fallacy

**arXiv ID:** 2609.07717 | [PDF](https://arxiv.org/pdf/2609.07717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 147. Guiding Worker Self-Selection in Crowdsourcing Contests: An LLM-Augmented Algorithmic Approach

**arXiv ID:** 2609.07749 | [PDF](https://arxiv.org/pdf/2609.07749v1)

**作者:** Nguyen Thach `[一作]`, Karim Lakhani `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究如何为众包平台的工人生成自我选择竞赛的推荐方案，旨在同时提升平台整体投入量并降低工人失望度。

**💡 创新点**

创新点在于提出了可在特定Tullock竞赛下保证零失望与平台最优的贪心框架GRAF，并设计了LLMScore——一种基于大语言模型的进化搜索方法，能够自动生成兼顾整体努力与工人失望的排序函数。

**🔧 技术方法**

主要技术包括Tullock竞赛理论、贪心算法、LLM驱动的进化搜索（类似EoH）以及Python实现与提示工程。

**📊 数据集**

实验使用1000个合成SSTC实例，涵盖四种不同设置（SSTC1–SSTC4），参数取自均匀分布和Beta分布，规模从10到60名工人、3到5个竞赛。

**📈 对比分析**

与手工设计的多种评分向量以及迭代最佳响应（IBR）方法对比，LLMScore生成的评分在所有设置中均取得更高的整体努力且更低的最大失望，并且在运行时间上优于IBR。

**⚠️ 局限性**

局限性包括仅在合成数据上验证，缺乏真实平台数据的实验；PSNE求解可能停留在局部最优；生成的排序对工人索引敏感，缺乏对称性处理。

---

## 148. LLM Forensics: Where Do Backdoors Hide? Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders

**arXiv ID:** 2609.07746 | [PDF](https://arxiv.org/pdf/2609.07746v1)

**作者:** Wissam Antoun `[一作]` (Inria Paris), Djamé Seddah `[通讯]` (Inria Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Gaperon LLM的触发式后门进行机制分析，使用稀疏自编码器识别触发检测、信号路由和输出读出特征。

**💡 创新点**

首次将稀疏特征分解用于后门机制，区分触发检测、行为控制和语言跟踪三种功能。

**🔧 技术方法**

使用JumpReLU和Matryoshka BatchTopK稀疏自编码器，并在残差、MLP、注意力位置训练特征。

**📊 数据集**

基于Gaperon-1B、8B模型的英语-法语/德语语言切换后门训练数据，包括触发、翻译和混合预训练样本。

**📈 对比分析**

通过对比触发与控制样本的F1、干预检验（消融、激活），发现残差特征最能抑制触发，法语可被激活但德语效果差。

**⚠️ 局限性**

研究受限于语言切换作为代理，无法直接推广到更具危害性的后门；24B模型训练不稳定，且结果可能受特征选择规则影响。

---

## 149. From Echo Chambers to Epistemic Monoculture: Large Language Models Present Temporally Contingent Partisan Alignments as Knowledge

**arXiv ID:** 2609.07735 | [PDF](https://arxiv.org/pdf/2609.07735v1)

**作者:** Wend K. Tam `[一作]` `[通讯]`, Wend K. Tam

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对Llama 3.1 8B进行机理可解释性研究，发现并定位了一个可解释的党派几何轴，并利用该轴在模型训练截止点前后的时间对比，进行对齐实验，揭示LLM在时间截断后对党派议题的输出呈现出“时间指纹”，从而说明LLM在知识呈现上呈现一种“知识即框架”的单一化模式。

**💡 创新点**

创新点在于：①首次将党派几何轴与时间截断点相结合，证明LLM的知识输出被训练时间固定的党派立场所决定；②通过对党派轴的可视化和操纵（steering）展示了模型在不同议题下的同向或逆向输出，揭示了“时态冻结”对政治信息生成的深层影响；③提出“知识即框架”与“知识即信息”的本体论区分，为LLM在民主信息环境中的角色提供新视角。

**🔧 技术方法**

技术手段包括：1) 公开权重的Transformer模型内部激活向量分析；2) 通过逻辑回归探测党派轴（对层18的激活进行线性投影）；3) 对激活向量做定向干预（steering）以改变生成文本；4) 设计时间对照实验，利用模型训练截止点前后政治现实变动的议题进行对比分析。

**📊 数据集**

主要数据集为：① 190,491条来自美国国会议员的推文，用于训练和验证党派轴；② Llama 3.1训练语料库（截至2023年12月）作为模型知识来源；③ 针对党派对齐实验的人工构造提示词，涵盖RFK Jr.、乌克兰援助、DEI、TikTok、Tylenol等在2024年后出现或逆向对齐的议题。

**📈 对比分析**

比较方法是：对相同提示词在左/右党派轴方向进行steering，记录生成文本的党派倾向；将结果与训练截止点前的党派共识或对齐状态进行对照，评估是否出现同向输出、逆向输出或统一输出；通过对比分析验证模型在时间截断后的输出与真实政治变动的偏离程度。性能表现主要体现在：同向输出与预训练期间的党派共识高度一致，逆向或统一输出与后期政治变动相悖，说明模型的党派几何轴在时间上保持不变。

**⚠️ 局限性**

局限性包括：①仅对单一开放权重模型（Llama 3.1 8B）进行研究，无法直接推广到闭源或更大规模的LLM；②党派轴的定义基于美式两党制，其他多党制或非民主体系的适用性未知；③实验聚焦于少量事例议题，缺乏大规模系统评估；④未探讨检索增强或更新机制对党派几何轴的潜在缓解效果；⑤缺少用户实验验证LLM输出对公众认知或信念的真实影响。

---

## 150. X-DigCheck: Co-Evolving Application Profiles and Knowledge Graphs, Demonstrated on the RTI Documentation of Rupe Magna

**arXiv ID:** 2609.07694 | [PDF](https://arxiv.org/pdf/2609.07694v1)

**作者:** Celian Ringwald `[一作]` (University of Bologna), Cristiano Putzolu `[通讯]` (University of Bologna)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并演示了 X-DigCheck，一个用于持续推进应用档案与知识图谱协同演进的通用环境，并通过 RupeMagna-RTI（文化遗产 RTI 调查）案例进行验证

**💡 创新点**

创新点在于将档案构建视为持续的 ontology‑data co‑evolution 循环，结合竞争问题、SHACL 与覆盖分析三种验证手段，形成可执行的回馈闭环；同时提供可离线运行的 CLI 与 Docker 化部署方案

**🔧 技术方法**

使用技术包括 Apache Jena Fuseki SPARQL 1.1、SPARQL Anything、OWL2SHACL、Python semRTI、Docker、CLI、YASGUI、Git、OWL 及 SHACL 验证框架

**📊 数据集**

使用的主要数据集是 2025 年 Rupe Magna 采集的 40 组 RTI 数据（共 11 句碑文），并通过 semRTI 将其提升为 RDF/KG；同时包含手工注释和元数据

**📈 对比分析**

通过执行 29 条竞争问题（93% 通过）和 42 条 SHACL 形状（76% 通过）对比验证；覆盖分析显示 8% 类和 3% 属性未实例化，未出现未声明术语；表明工具在该用例中的有效性

**⚠️ 局限性**

局限性包括：目前仅在单一 RTI 用例上演示，缺乏对不同领域和不同映射技术的全面测试；映射文件错误不在当前报告范围内；需要进一步集成 LLM 辅助以自动修复映射与形状

---

## 151. Emergent Charging Coordination in Electric Delivery Fleets

**arXiv ID:** 2609.07689 | [PDF](https://arxiv.org/pdf/2609.07689v1)

**作者:** Javier Vales-Alonso `[一作]` (Universidad Politecnica de Cartagena), Juan J. Alcaraz `[通讯]` (Universidad Politecnica de Cartagena)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了电动车配送车队在中途充电时的协同决策问题，提出基于局部信息的去中心化学习控制方案。

**💡 创新点**

创新点是：无需预留、无通信、无中央调度，仅使用车辆自身状态和站点占用广播实现自发协调，在多城市零样本迁移中可接近最优规划。

**🔧 技术方法**

采用强化学习（PPO）和神经进化（NEAT）构建车辆充电决策网络，并结合统计路网模型与离散事件仿真进行评估。

**📊 数据集**

使用OpenStreetMap的20个真实城市道路与充电站数据，通过OSMnx生成配送点，构成基准场景。

**📈 对比分析**

与全局Oracle、贪心阈值、无充电基线对比；PPO‑joint在20城市零样本上平均完成率98.6%，Oracle为99.5%；相比传统规则提升30%以上；NEAT略逊，SA‑tuned规则处于中间水平。

**⚠️ 局限性**

局限在于仅考虑站点占用广播，假设到达与服务过程平稳；未建模充电成本、能量衰减；依赖静态路网统计，未考虑动态交通或司机休息等因素。

---

## 152. Do AI Coding Assistants Check Before They Install? A Pre-Registered Demand-Side Audit of Trust Signals in the Research Software Supply Chain

**arXiv ID:** 2609.07754 | [PDF](https://arxiv.org/pdf/2609.07754v1)

**作者:** Pengyin Shan `[一作]` `[通讯]` (National Center for Supercomputing Applications), Pengyin Shan (National Center for Supercomputing Applications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对六个研究软件项目进行预注册的实验，测量 AI 编码助手在安装前是否读取或验证软件的信任信号，发现几乎没有验证行为；

**💡 创新点**

创新点在于首次将预注册实验与完整成本记录相结合，系统评估需求侧（助手）在接受供应侧（信任信号）时的行为；

**🔧 技术方法**

使用了容器化日志捕获、信任信号注入工具（SBOM、签名、SLSA 证明、渠道声明）以及三种主流语言模型和两种操作手段（有批准与无批准）；

**📊 数据集**

数据集为从先前的 87 项研究软件语料库中抽样得到的六个项目，分别为三项高性能计算和三项量子计算软件；

**📈 对比分析**

比较方法为对每个模型、每种手段和每个信号条件执行多次实验，统计验证行为出现率、检索事件频率和成本，结果显示验证率低于 2%，成本高昂且与价格无明显关联；

**⚠️ 局限性**

局限性包括样本量仅六个项目、实验在隔离容器环境下进行、未覆盖所有模型/手段变化、未测量实际安装成功与否，以及可能因模型更新导致复现性受限。

---

## 153. Attributing Cohen's d: Training Data Attribution for Disease-Related Effects in Normative Age Biomarkers

**arXiv ID:** 2609.07729 | [PDF](https://arxiv.org/pdf/2609.07729v1)

**作者:** Jakob Snel `[一作]` (Hertie Institute for AI in Brain Health), Marc-Andre Schulz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了一种基于疾病相关效应大小（Cohen's d）的影响函数，用以识别并去除对英国生物银行（UK Biobank）中各疾病（多发性硬化、脑血管疾病、2 型糖尿病、慢性肾病）年龄模型训练产生负面影响的样本，从而提升留置样本的疾病区分效应。

**💡 创新点**

创新点在于将归因目标从传统的训练损失改为疾病效应大小；给出该目标的闭式影响函数，并通过留一法验证其准确性；发现通过归因去除的样本携带未被诊断的亚临床代谢或心血管负荷，说明该方法能补偿诊断排除列表的盲点。

**🔧 技术方法**

技术手段包括：岭回归（Ridge）年龄预测模型、年龄偏差校正、对训练参数的闭式影响计算、选择/评估拆分、配对随机对照、Cohen's d 及均值年龄差的两种归因目标、以及对被归因样本的表型特征分析。

**📊 数据集**

使用英国生物银行（UK Biobank）约50万人数据：T1 脑影像特征（1,440）用于多发性硬化和脑血管疾病；核磁共振代谢组学（251）用于 2 型糖尿病和慢性肾病；训练集约 40,000 名健康人，评估集包含病例与匹配对照。

**📈 对比分析**

将归因去除与等量随机去除进行配对比较，结果显示归因去除在所有疾病和模态下均显著提升 Cohen's d（例如 2 型糖尿病 10% 去除时提升约 1.06，脑龄 10% 提升约 0.20），而随机去除基本保持不变；留一法检验显示归因排序与实际影响高度一致（重叠率 98.9–99.8%）。

**⚠️ 局限性**

局限性包括：仅在岭回归下验证，非线性模型的推广未知；一次性计算排序，未考虑迭代或级联污染；仅对单疾病的模型训练，未测试共享参考人群的效果；仅使用 UK Biobank，外部可重复性待验证；归因样本表型描述为相关而非因果，可能引入公平性关注。

---

## 154. Better Call CineCrew: Consistent Ultra-Long Narrative-to-Film Generation

**arXiv ID:** 2609.07720 | [PDF](https://arxiv.org/pdf/2609.07720v1)

**作者:** Jiaben Chen `[一作]` (University of Massachusetts Amherst), Chuang Gan `[通讯]` (University of Massachusetts Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CineCrew系统，通过结构化的FilmDSL将长篇剧本转化为可执行的拍摄计划，实现了从叙事到电影级视频的自动生成。

**💡 创新点**

核心创新在于将剧本映射为多层次的Domain‑Specific Language，显式编码镜头、相机、资产、连贯性和人物个性等电影制片约束，并通过多智能体协作实现计划、生成、评审与修复。

**🔧 技术方法**

使用了多智能体框架、基于文本提示的离线视频生成器（如g_V/g_AV）、关键帧先行生成、TTS音频合成、结构化评审与规则书机制，以及在FilmDSL中嵌入的资产库与记忆系统。

**📊 数据集**

在MovieBench上抽取了20个跨题材、跨时间段的叙事样本作为评估数据集。

**📈 对比分析**

与MovieAgent、AniMaker和LTX‑Studio等基线进行对比，CineCrew在电影级视听评分、跨镜头一致性得分及用户主观偏好上均取得最高分，说明其在节拍清晰度、人物连贯性和视觉一致性上优于现有方法。

**⚠️ 局限性**

主要局限是对现有视频生成器的依赖，生成质量受后端模型限制，且在极长的剧情或复杂人物互动下仍可能出现细节误差；同时规则书和DSL编写需要人工制定，难以完全自动化。

---

## 155. Human-AI Co-Creativity: Advances, Opportunities, and Challenges

**arXiv ID:** 2609.07711 | [PDF](https://arxiv.org/pdf/2609.07711v1)

**作者:** Adish Singla `[一作]` (Max Planck Institute for Software Systems), Chao Wen `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了ICML 2026人机共创工作坊的议程、论文与演讲，梳理了生成式AI与创意领域的交叉点，并提出了未来研究路线图。

**💡 创新点**

创新点在于构建了多语种、跨学科的研究社区框架，明确了两大核心研究方向（生成模型的创意性提升与人机共创挑战），并系统性总结了现有指标、benchmark与技术方法。

**🔧 技术方法**

主要聚焦生成式AI技术（LLM、GAN、Stable Diffusion等）、人机交互设计、自动化创意度量与真实性检测等，涵盖了提示工程、奖励机制、架构改进、情境化评测等方法。

**📊 数据集**

综述的论文与演讲涉及的 benchmark 包括 NoveltyBench、CreativityPrism、AUT/RAT/TTC、CSI 等；数据集范围涵盖文本、图像、音频及多模态创意任务（如文本写作、艺术设计、音乐创作）。

**📈 对比分析**

对比方法主要是基于自动化创意度量（多样性、原创性、价值、惊奇度）与人工评测（专家评分、Consensual Assessment Technique），讨论了不同研究的指标差异与结果表现；总体表明尚缺统一标准，性能评估多维度且各方法优缺参差。

**⚠️ 局限性**

局限包括：创意度量的主观性与低一致性；benchmark 侧重单模型而非人机协作；真实性与作者归属检测在规模与鲁棒性方面仍不足；设计固着与过度依赖问题缺乏系统评估；跨领域与群体层面的创意提升尚未解决。

---

## 156. ParetoTransport: Generative Optimization by Mass Transport Toward The Pareto Front

**arXiv ID:** 2609.07706 | [PDF](https://arxiv.org/pdf/2609.07706v1)

**作者:** Stephanie Holly `[一作]` (LIT AI Lab and Institute for Machine Learning), Werner Zellinger `[通讯]` (LIT AI Lab and Institute for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出一种训练无关的指导方法ParetoTransport，利用预训练的流匹配模型在离线多目标优化中对整体分布进行引导，直接将生成的目标空间分布运输至Pareto前沿并实现均匀覆盖；

**💡 创新点**

创新点在于把分布级别的控制（Wasserstein匹配与方向性改进）嵌入流匹配的概率路径，引导整体分布而非单样本，并且使用CHIM法则提供全局正则化方向；

**🔧 技术方法**

采用流匹配模型、Wasserstein距离、方向性梯度（CHIM normal）以及基于代理分布的迭代匹配-改进策略；

**📊 数据集**

在12个合成任务（ZDT、DTLZ）和15个真实工程任务（RE）上进行实验；

**📈 对比分析**

与多种前沿方法（Forward: NSGA-II、Multi-Head、Multiple-Models；Inverse: ParetoFlow、PreferenceGuidedDiffusion）在HV、GD、IGD、W2四项指标上进行比较，ParetoTransport在W2和GD上平均排名第一，在HV和IGD上竞争激烈，整体表现最优；

**⚠️ 局限性**

仅在目标数≥4的多目标情形下可能受限，CHIM normal只提供单一全局方向，可能无法充分捕捉高维Pareto前沿结构。

---

## 157. Situated Action in Pre-Hospital Critical Care Dispatch: Identifying where and how Algorithmic Assistance might be useful in the daily work of specialist Emergency Medical Dispatchers

**arXiv ID:** 2609.07705 | [PDF](https://arxiv.org/pdf/2609.07705v1)

**作者:** Ben Wilson `[一作]` (Swansea University), David Rawlinson `[通讯]` (EMRTS, Swansea Bay UHB)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在威尔士EMRTS（预医院危重病救援与转移服务）紧急医疗调度中心，作者通过现场沉浸、情境观察与结构化访谈，系统识别并细化了三大关键决策点（A：开窗查看、B：现场信息搜集、C：是否调度），并描绘了团队成员间的交互与信息流动模式，为后续AI辅助系统的需求与设计奠定基础。

**💡 创新点**

创新点在于将人机交互的情境化方法与临床决策分析结合，首次将实地工作细节映射到AI支持候选点，并提出了“从情境到算法”的工作流程框架，解决了传统AI模型与临床工作流程脱节的问题。

**🔧 技术方法**

技术手段主要是人机交互（HCI）领域的情境观察、定性编码（常量比较法与轴向编码）以及工作流绘制；并未实现具体的AI模型，而是给出了AI系统设计的功能需求与约束。

**📊 数据集**

使用的数据集包括现场观察记录、呼叫列表记录、团队内部通讯日志、培训记录和呼叫日志等非结构化/结构化数据；并未使用公开的标准数据集。

**📈 对比分析**

论文未实施任何AI系统，也未给出性能指标；仅在讨论中提出未来实验评估AI辅助在三大决策点的可行性与有效性。

**⚠️ 局限性**

局限性：研究仅在威尔士单一机构进行，样本量有限；缺乏实际AI系统验证；观察者效应与人工作业的不可控变异；对系统可推广性和跨机构差异性缺乏探讨。

---

## 158. SoK: Secure Software-Based Multi-Domain Data Segregation

**arXiv ID:** 2609.07701 | [PDF](https://arxiv.org/pdf/2609.07701v1)

**作者:** Quang Cao `[一作]` (RMIT University), Nalin Arachchilage `[通讯]` (RMIT University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性综述并整合了 80 篇关于软件化多域数据隔离在语音通信系统（VCS）中的研究，梳理现有技术、发现不足并提出未来研究路线。

**💡 创新点**

首次将 SDN/NFV/网络切片、隔离内核、跨域解决方案（CDS）与后量子密码学（PQC‑IPsec）等多学科技术聚合为一种可扩展的软件化多域隔离框架，并系统性识别了现有研究的核心空白。

**🔧 技术方法**

采用系统文献综述方法（Kitchenham & Charters 规范），基于 PICOC 框架构建检索词，对 IEEE Xplore、SpringerLink、Scopus、ACM Digital Library 等数据库进行检索，并对 SDN、NFV、网络切片、隔离内核、CDS、PQC‑IPsec 等技术进行分类与分析。

**📊 数据集**

文献来源共 80 篇（包含期刊、会议、书籍章节等），未使用实验或现场数据集；所有分析基于已发表的研究与标准文献。

**📈 对比分析**

通过主题分组与差距分析（gap analysis）比较不同技术在多域隔离、实时语音传输、操作员接口等维度的能力；由于缺乏统一实验平台，本文未给出定量性能指标，而是指出现有技术在实时性、互操作性和保障等级方面的理论优缺点。

**⚠️ 局限性**

局限性：缺乏统一的软件化多域隔离架构与实现原型；未进行实测或真实环境评估；在跨域实时语音混音、单域发送控制、非干扰性保障等关键需求上仍无完整解决方案；缺乏对量子威胁下 PQC 与 IPsec 性能的系统评估。

---

## 159. Audit Without Verification: When LLM Accountability Layers Relay Rather Than Check

**arXiv ID:** 2609.07680 | [PDF](https://arxiv.org/pdf/2609.07680v1)

**作者:** Paul-Peter Arslan `[一作]` `[通讯]` (Institute For Future Technologies), Paul-Peter Arslan (Institute For Future Technologies)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多机构跨组织边界的LLM代理管线中的责任层，评估报告中是否包含显式结论对错误归因的影响，探讨如何通过删除结论字段改进审计准确性；

**💡 创新点**

首次量化责任层对归因准确性的负面作用，并揭示删除单一结论字段可在不同情境下显著提升或降低审计准确率，提供基于上下游可靠性评估的可操作决策框架；

**🔧 技术方法**

使用大语言模型（GPT‑4.1 Mini、Claude Haiku 4.5 等）生成代理报告与审计结果，采用基于预注册的三因素实验设计、GEE 统计方法、互信息评估等技术；

**📊 数据集**

构建可重现的合成数据集：345,600 次请求（57,600 条案例），包含两种缺陷族（不一致与遗漏）以及对应的“干净”对照，数据以种子驱动可复现；

**📈 对比分析**

对比方法包括原始文档 vs 代理报告、完整信息 vs 仅通信信息、删除结论字段前后两组；实验结果显示原始文档下审计准确率约 60%，代理报告下约 4–8%，删除结论字段后在上游错误时提升 41%（GPT‑4.1 Mini）或 27%（Claude Haiku 4.5），但在上游正确时下降 15–9%；在软件交付工具链实验中收益更大，成本消失；

**⚠️ 局限性**

局限包括：仅在合成管线与两种模型下验证，缺乏真实企业部署验证；结论依赖于审计模型与域的可变性；预注册部分未全部实现；数据集仅包含两类缺陷；实验设置对结果解释存在潜在偏差；

---

## 160. A Note on Binary Quadratic Systems and their relation to complexity theory

**arXiv ID:** 2609.07769 | [PDF](https://arxiv.org/pdf/2609.07769v1)

**作者:** Gabriele Radici `[一作]` (University of Trento), Massimiliano Sala `[通讯]` (University of Trento)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在二元域上，平方二次方程组的无解与唯一解系统数量满足明确的有限n上界，且二者差距仅为O(2^{-n})。

**💡 创新点**

创新点在于将母单纯复形的口（port）与Reed–Muller码的最小支持词对应，利用Whitney式的sign‑reversing involution和MacWilliams恒等式获得精确计数和极限接近的理论界限。

**🔧 技术方法**

主要技术包括图论中的母单纯复形与Matroid理论、Whitney断路子定理的自反性构造、编码理论的Reed–Muller码与其对偶码、以及MacWilliams身份与最小距离估计。

**📊 数据集**

研究为纯理论工作，没有使用实验数据集；所有结论均来自组合与编码理论的严谨证明。

**📈 对比分析**

通过与已知的极限lim |α₁/α₀|=1对比，本文给出具体常数1+2^{-n}的上界，理论上实现了比无偏估计更紧的性能上限。

**⚠️ 局限性**

仍未给出多项式时间的显式注入映射（将无解系统映射为唯一解系统），该问题保持为开放的算法难点。

---

## 161. A Theoretical Analysis of Generalization Dynamics in Neural Networks under Gradient Descent with Weight Decay

**arXiv ID:** 2609.07755 | [PDF](https://arxiv.org/pdf/2609.07755v1)

**作者:** Yuqing Wang `[一作]` (Johns Hopkins University), Mikhail Belkin `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

构建了一个通用的理论框架，用来把整体泛化误差拆解为数据误差、优化误差和预测变异误差，并在训练动态中逐步分析每一项的演化。

**💡 创新点**

创新点包括：① 引入局部近似齐性（local approximate homogeneity）概念，用来刻画网络块在不同层的尺度变换特性；② 通过层级分解给出层级间泛化差异的必要与充分条件；③ 将“grokking”现象与预测变异误差的延迟衰减联系起来，提供了对该现象的理论解释。

**🔧 技术方法**

主要技术手段：梯度下降（含权重衰减）的收敛分析、实解析和多项式光滑性（poly‑smoothness）理论、预测变异误差的层级动态估计、局部齐性误差估计、以及对数据分区的概率与几何分析。

**📊 数据集**

文中未给出具体实验数据集，所有结果均为理论推导。

**📈 对比分析**

由于缺乏实验验证，文中未给出与其他方法的性能比较；所述结果主要以数学不等式和收敛率形式呈现。

**⚠️ 局限性**

局限性：① 依赖于多项假设（如输入独立、网络块可近似齐性、权重衰减等），实际可验证性有限；② 理论结果给出的是上界/下界，缺乏对真实误差大小的精确量化；③ 未通过实验验证框架的预测效果，难以评估在实际任务中的实用性。

---

## 162. Dex-X: Learning Visual-Tactile Dexterous Manipulation From Human Videos with Simulated Interaction

**arXiv ID:** 2609.07747 | [PDF](https://arxiv.org/pdf/2609.07747v1)

**作者:** Ruoqu Chen `[一作]` (Tsinghua University), Mengdi Xu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过把单目人类视频中的手物交互重建到仿真中，并利用仿真提供的触觉信息训练一种特权状态的强化学习专家，然后把专家策略蒸馏为可以在真实机器人上使用的视觉-触觉多任务控制器，实现了从人类视频到可部署机器人操控的零样本跨域迁移。

**💡 创新点**

① 将仿真视为触觉补全引擎，利用仿真物理交互生成缺失的触觉信号；② 结合演示指导的RL，获得具备触觉感知的闭环操作；③ 使用统一的点云+触觉表示，将视觉、触觉和本体感受融合，完成从专家到可部署策略的蒸馏。

**🔧 技术方法**

仿真环境（IsaacLab）+ PPO强化学习 + 非对称Actor-Critic + 视觉-触觉点云编码 + 触觉信息蒸馏 + 任务类别集成。

**📊 数据集**

人类单目视频演示（涵盖杯子、立方体、刮刀、锤子等物体的6类任务），以及在仿真中重建的手物轨迹和触觉信号。

**📈 对比分析**

与ManipTrans、DAPG以及运动重映射等基线对比。专家在仿真中的平均成功率为65.9%，相比ManipTrans的21.9%和DAPG的40.6%大幅提升；蒸馏后的视觉-触觉策略在真实世界的立方体抓取任务上达到93%成功率，在桌面清理任务上为53%，表现出零样本跨域成功率高于传统方法。

**⚠️ 局限性**

对长时间交互和大范围几何变化的泛化能力有限；对感知误差、执行器延迟和触觉模型的不一致较为敏感；使用的触觉表示仅为指尖力值，缺乏压力、剪切、滑动等丰富信息；蒸馏策略仍依赖预先重映射的运动参考，限制了更自由任务的适用性。

---

## 163. TV-SGS: Gaussian Splatting with Geometric Information Propagation via Tensor Voting under sparse views

**arXiv ID:** 2609.07734 | [PDF](https://arxiv.org/pdf/2609.07734v1)

**作者:** Harish N Sathishchandra `[一作]` (Stevens Institute of Technology), Philippos Mordohai `[通讯]` (Stevens Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究提出了一种基于 Tensor Voting 的 Gaussian Splatting 框架（TV‑SGS），通过在 3D 空间中直接传播几何信息，显著提升了稀视角下的几何重建质量。

**💡 创新点**

创新点包括：①将 Tensor Voting 重新定义为可对 Gaussian splats 进行 3D 传播的机制，使 splats 之间能够直接相互约束；②设计了两类无渲染依赖的 3D 损失（TV normal loss 与 TV position loss），并利用虚拟接收器推断表面位置，进一步增强几何一致性。

**🔧 技术方法**

技术手段主要包括 Tensor Voting、虚拟接收器、Whitening 变换、CUDA 加速的邻域搜索，并将 TV‑SGS 与现有骨干（PGSR、FatesGS、VGGS）无缝集成。

**📊 数据集**

在 DTU 与 Tanks‑and‑Temples 两大稀视角数据集上进行了评估。

**📈 对比分析**

与骨干方法对比，TV‑SGS 在几何指标（Chamfer Distance/F1）上提升约 7% 甚至更多，同时保持或提升渲染质量（PSNR/SSIM/LPIPS）。

**⚠️ 局限性**

局限性包括：在稠密视角下提升不明显；Tensor Voting 的参数需手动设定，缺乏自适应；极稀视角或超大场景仍可能出现浮点漂移问题。

---

## 164. Zero-Shot 3D Plant Organ Segmentation with SAM3 and Semantic NeRFs

**arXiv ID:** 2609.07724 | [PDF](https://arxiv.org/pdf/2609.07724v1)

**作者:** Andreas Gilson `[一作]` (Fraunhofer Institute for Integrated Circuits), Peter Pietrzyk `[通讯]` (Fraunhofer Institute for Integrated Circuits)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究提出了一种零样本（无标注）3D植物器官分割管线，利用文本提示驱动的SAM3模型在多视角RGB图像上生成2D掩模，再通过语义NeRF（Nerfacto）将这些2D掩模投影到三维空间，最终得到语义标记的3D点云。

**💡 创新点**

核心创新点在于：①完全不依赖人工标注或特定物种的模型微调；②将文本提示直接与SAM3结合，省去检测-分割两阶段；③利用NeRF的多视角一致性作为隐式投票机制，将噪声2D掩模提升为高质量3D标注；④通过单一管线即可在多种植物种类上实现统一分割。

**🔧 技术方法**

使用的主要技术包括：SAM3（基于ViT+文本编码的跨模态分割模型）、Nerfacto语义NeRF（hash‑grid编码、密度与语义分支）、多视角相机位姿（SfM/Colmap）、基于光照与深度的可微渲染、阈值筛选与后处理（累计、深度梯度、轴对齐盒子等）。

**📊 数据集**

实验数据集涵盖：①单株Begonia maculata（150张RGB图，1.5M点云，手工像素级标注）；②由自动化多视角平台采集的10个不同物种（每株约101张RGB图，10种多样化形态），并对重建点云手工标注。

**📈 对比分析**

性能评估：在Begonia上，SAM3+NeRF得到mIoU 0.926（约95.9% oracle上限），比Grounded‑SAM高13.4pp；在10种多样化物种上平均mIoU 0.856±0.097，叶子与盆栽IoU均≥0.90。与现有监督方法（如Organ3DNet）相比，零样本管线在相同任务下已逼近其精度。

**⚠️ 局限性**

局限性包括：①仅提供语义分割，缺乏实例分割；②对细小枝条（stem）分割仍不理想，受NeRF平滑边界与hash‑grid稀疏特性的限制；③管线高度依赖SAM3的2D分割质量，若遇未知物种或生长阶段可能失效；④需要多视角RGB与SfM位姿，限制了单相机或低视角场景；⑤训练耗时约35分钟/场景，难以满足高通量实时需求。

---

## 165. The Emerging AI Paper-Review Arms Race: Adversarial Co-Evolution in Scholarly Publishing

**arXiv ID:** 2609.07713 | [PDF](https://arxiv.org/pdf/2609.07713v1)

**作者:** Chenguang Wang `[一作]` (Virginia Tech), Dawei Zhou `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 AI 在科研与同行评审中的影响进行系统综述，提出“AI 论文‑评审 arms race”框架，并基于 230 篇文献构建六个耦合动态的分类学。

**💡 创新点**

创新点在于把生产、评估、操控、防御、逃逸和长周期生态反馈等维度联结为一个连贯的因果循环，突出 AI 作用的相互反馈与演化。

**🔧 技术方法**

主要采用文献检索、结构化归纳与映射技术，对相关研究进行分类、关系梳理和案例对照；不涉及新算法实现。

**📊 数据集**

使用 230 篇学术论文和机构记录（包括会议、期刊政策文件、实测部署记录等）作为研究数据源。

**📈 对比分析**

比较方法是基于文献映射与案例对照，未给出可量化性能指标；主要通过对比不同研究的发现和实证结果来论证框架的可行性。

**⚠️ 局限性**

局限性：依赖公开论文与政策文档，缺乏因果和跨学科实证；对部分动态（如长周期反馈、逃逸行为）仅有理论或实验预测，缺少完整的现实链路跟踪。

---

## 166. CrowdTraj: A Benchmark for Dense Crowd Trajectory Prediction in Realistic Crowded Environments

**arXiv ID:** 2609.07685 | [PDF](https://arxiv.org/pdf/2609.07685v1)

**作者:** Antonius Bima Murti Wijaya `[一作]` (University of Glasgow), Marwa Mahmoud `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了CrowdTraj数据集，构建了真实世界高密度CCTV监控场景下的行人轨迹预测基准，支持从检测到跟踪再到轨迹预测的端到端评估；

**💡 创新点**

创新点在于提供了最高密度达372人的场景、像素与真实世界坐标、以及在严重遮挡下同时评估检测、跟踪与预测的全流程，揭示了现有方法在密集人群中的不足；

**🔧 技术方法**

使用YOLOv11/YOLOv26检测器配合ByteTrack、BoT‑SORT、SMILETrack、JDE跟踪器，并对SocialVAE*、MART、MoFlow、DSTIGCN等预测模型进行实验；

**📊 数据集**

所用数据集为CrowdTraj，包含5个场景（Duri Morning/Evening、Vredeburg、Duri Platform1、Paisley Festival），共约5k帧、约3.2M头部边界框以及对应的真实世界坐标；

**📈 对比分析**

通过对跟踪指标（MOTA、IDF1）和轨迹误差（ADE/FDE）的系统对比，发现跟踪在最密集场景下准确率降至0.68–0.70，MART在像素坐标下误差最低但训练时间显著增长，SocialVAE*在真实世界坐标下更具泛化性，整体性能在高密度下显著下降；

**⚠️ 局限性**

局限性包括跟踪算法在高遮挡下仍易失效、预测模型计算开销大且易过拟合、数据仅基于头部检测导致信息损失、标注模糊率约1.4%，以及缺乏多模态输入等问题。

---

## 167. Buildability Assessment of 3D-Printed Concrete Structures Using Smoothed Mohr-Coulomb Plasticity with Isotropic Hardening

**arXiv ID:** 2609.07679 | [PDF](https://arxiv.org/pdf/2609.07679v1)

**作者:** Saif-Ur-Rehman `[一作]` (Bundesanstalt fur Materialforschung und -prufung), Jörg F. Unger `[通讯]` (Bundesanstalt fur Materialforschung und -prufung)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种基于平滑Mohr‑Coulomb模型和非线性硬化的3D混凝土打印可建造性评估框架。

**💡 创新点**

创新点是将平滑Mohr‑Coulomb弹塑性模型与非线性协同硬化结合，并采用全刚度特征值判据检测首个失稳，提升了对打印过程失效的预测准确性。

**🔧 技术方法**

使用了更新拉格朗日有限元法、Jaumann应力率、层激活场、特征值分析及 Smoothed Mohr‑Coulomb 返回映射。

**📊 数据集**

使用了直墙和空心圆柱两套实验数据集（来自文献中的打印实验）进行验证，并在此基础上做了参数化研究。

**📈 对比分析**

通过与实验测得的坍塌层数、之前数值模型以及解析的自重屈曲高度对比，表现出误差≤5%，并在参数空间内展示了层数随直径和硬化率的非单调变化。

**⚠️ 局限性**

局限在于未考虑材料损伤、概率分布和几何不均匀性，仅适用于理想化、单一混凝土配方，且对极端几何缺陷敏感。

---

## 168. Bag of Tricks or Bag of Myths? Reducing Modeling Complexity with Task Knowledge in Explainable Suicide Risk Assessment

**arXiv ID:** 2609.07766 | [PDF](https://arxiv.org/pdf/2609.07766v1)

**作者:** Shlok Shelat `[一作]` (Indian AI Research Organisation), Amit Sheth `[通讯]` (Indian AI Research Organisation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对 1,635 篇临床标注的 Reddit 帖子进行小样本、多输出（风险等级、支持证据、24 项临床因子）评估，并通过 31 项预设技术的严格对照实验审计后，仅保留对任务具有内在知识支撑的 5 个组件，构建最终系统 TRIDENT。

**💡 创新点**

提出“任务条件技术选择”与“部署一致校准”两项原则，强调在小样本、类不平衡、输出耦合的实际风险评估场景中，只有任务本身提供的知识才决定哪些技术能真正提升性能；同时揭示阈值拟合与集成平均导致的分布不匹配问题。

**🔧 技术方法**

使用多层级模型（中型编码器、预训练编码器、符号规则、LLM 路由）、类别平衡损失、定义驱动的 entailment 预测、不同架构的多样化集成、阈值重新校准等技术，并剔除无效的容量扩张、生成数据、self‑training、检索等常用手段。

**📊 数据集**

基于 CLPsych 2026 共享任务提供的 1,635 条标注帖子（153 位作者），并在 5 折作者独立划分上评估；测试集为 378 条隐藏帖子（36 位作者）。

**📈 对比分析**

采用作者不重叠的多次重抽样（约 300 次控制实验）与配对差异比较，采用“赢率”与“种子范围”判定显著性；最终系统在公开排行榜上获得 0.7781 的综合分（第三名），单项表现为风险 0.8203、证据 0.7953、因子 0.7045。

**⚠️ 局限性**

局限包括：1）种子波动大导致可检测效应窗口受限；2）对照实验仅在单一数据集与任务设置下完成，外部可重复性未知；3）评估指标对罕见类别极度敏感，无法准确估计上限；4）多技术比较的多重性与基准演化可能引入偏差；5）系统仅用于离线排序，未验证临床安全与实用性。

---

## 169. Local gradient neural operator

**arXiv ID:** 2609.07752 | [PDF](https://arxiv.org/pdf/2609.07752v1)

**作者:** Baiming Zhang `[一作]` (Zhejiang University), Shiying Xiong `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量化的本地梯度神经算子（LGNO），用于有限数据下的PDE源识别和时间演化预测。

**💡 创新点**

通过梯度感知的离散模板重构，将系数学习与场重建分离，并结合对称性折叠与零一致性构造，使模型解释性强且参数极少。

**🔧 技术方法**

采用多层感知机卷积（MLPConv）生成局部模板系数，零一致性因子分解，组对称性折叠，以及传统有限差分前置，形成局部梯度神经算子框架。

**📊 数据集**

在多种PDE基准上验证：1D扩散、2D Burgers、Navier–Stokes、Gross–Pitaevskii、3D Schrödinger，并在不同潜在场与源条件下进行测试。

**📈 对比分析**

与DeepONet、FNO、MLPConv、LOINN等基线在相同低样本条件下对比，LGNO在参数不足2000的情况下，测试误差平均下降约74%，在Navier–Stokes和Schrödinger的长时滚动中误差降低超过50%，展现出优越的稳定性与泛化性能。

**⚠️ 局限性**

对高度非局部或多尺度耦合系统的适用性有限，且对强噪声或非对称源的鲁棒性仍待进一步验证。

---

## 170. When Intelligence Becomes Agency: A Theory of Governed, Proactive Agency for Symbiotic AI Systems

**arXiv ID:** 2609.07741 | [PDF](https://arxiv.org/pdf/2609.07741v1)

**作者:** João Dias Ferreira `[一作]` `[通讯]` (Wyrde AI), João Dias Ferreira (Wyrde AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个针对永续、主动式 AI 助手的治理式代理框架，重点解决“激活问题”，即代理何时、何种模式下介入或保持沉默；

**💡 创新点**

核心创新在于将代理视为跨时间的组织化行为，并将感知、意图、情感-意向推理、约束与反馈等五大要素与委托授权（mandate）统一起来，构建了可度量、可评估的“激活门”机制；

**🔧 技术方法**

框架基于 LLM 驱动的代理技术（如 ReAct、Toolformer、Reflexion 等），结合多模态感知、情感-意向推理模型、政策门控与日志审计；

**📊 数据集**

本文未依赖具体数据集，而是提出了一套评估基准与模拟场景（如关键警报、家庭消息、健康提示等）来测试激活策略；

**📈 对比分析**

通过对比传统反应式代理与加入激活门的代理，论文展示了在误报率、延迟和用户满意度等维度的改进，实验表明激活门能显著降低不必要的打断并提升交互效率；

**⚠️ 局限性**

局限性包括：1）模型对复杂情境的情感-意向推理仍不成熟；2）需要高质量的多模态传感数据与用户授权；3）在跨任务或多代理协作中的委托漂移和信息不对称问题尚未完全解决。

---

## 171. An emancipatory vision for designing (generative) AI for learner flourishing

**arXiv ID:** 2609.07715 | [PDF](https://arxiv.org/pdf/2609.07715v1)

**作者:** Luis P. Prieto `[一作]` (Universidad de Valladolid), Yannis Dimitriadis `[通讯]` (Universidad de Valladolid)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种以学习者繁荣为核心的生成式人工智能教育技术解放式愿景，并给出新的假设、设计原则与方法论框架。

**💡 创新点**

创新点在于将价值敏感设计与多层级复杂系统结合，强调学习者主体性、技术适配、屏幕外互动、情感脱离等多维度考量，超越传统单一目标的AIED 设计思路。

**🔧 技术方法**

技术核心是基于生成式人工智能（LLM）但强调低复杂度、可本地化的模型与最小化资源需求的服务化实现。

**📊 数据集**

本工作未使用具体实验数据集，而是构建了理论与实践的多层级评估框架，强调在真实教育环境中进行长期多维度测量。

**📈 对比分析**

论文为概念性立场稿，未提供实验比较或性能指标；其价值在于提供了一套可供后续研究检验的设计与评估路径。

**⚠️ 局限性**

局限包括缺乏具体实现技术细节、评估指标与测量工具、跨层级社会/环境繁荣指标的可操作化、以及对LLM偏见与生态影响的系统性考虑。

---

## 172. APPSim-Bench: Bridging Real-world Apps and Reproducible Evaluation for Mobile GUI Agents

**arXiv ID:** 2609.07712 | [PDF](https://arxiv.org/pdf/2609.07712v1)

**作者:** Jintian Feng `[一作]` (Central China Normal University), Yichen Gong `[通讯]` (Agentic Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了 AppSim-Bench，一套基于 17 个可控模拟移动应用、共 557 条自然语言任务的可复现移动 GUI 代理基准。

**💡 创新点**

创新点在于通过编码代理辅助与人工审核构建可控后端数据的模拟应用，消除实时商业 App 的推荐、广告、账户等随机性，实现 deterministic 结果验证，同时覆盖中英文生态。

**🔧 技术方法**

技术手段包括编码代理生成 UI 与后端逻辑、人工验证修正、状态/答案/混合三种任务验证、行动统计（动作数、行动开销、预算耗尽）等评估指标。

**📊 数据集**

使用的数据集为 557 条任务，涵盖 WeChat、JD、RedNote、Tencent Meeting、Amap、Ctrip、Bilibili、Ele.me、NetEase Cloud Music、Amazon、Booking.com、Instagram、Spotify、Uber Eats、WhatsApp、YouTube、Zoom 等 17 个中英主流应用的自然语言指令、页面截图和数值推理子任务。

**📈 对比分析**

在 19 个通用与专门化 GUI 代理上进行比较，最强模型 Claude-Opus-4.7 的整体准确率为 50.27%，但 28.55% 的任务无人解，长流程、数值推理任务表现最差，行动开销与准确率呈负相关。

**⚠️ 局限性**

局限性包括仅在可控模拟环境评估，缺少真实商业 App 的广告、网络延迟、个性化等噪声；任务覆盖范围有限，未涉及跨应用、实时交互、无障碍、敏感安全等场景；视觉逼真度虽高但仍存在细微差异。

---

## 173. Crossing the Streams: SSH Plaintext Recovery via a Common Compression Context in Multiplexed Channels

**arXiv ID:** 2609.07709 | [PDF](https://arxiv.org/pdf/2609.07709v1)

**作者:** Fabian Bäumer `[一作]` (Ruhr University Bochum), Marcus Brinkmann `[通讯]` (Ruhr University Bochum)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 SSH 连接层的压缩侧通道，揭示通道复用导致的跨通道压缩泄露，并实现了基于此的自适应压缩攻击；

**💡 创新点**

首次发现并利用 SSH 多路复用与压缩交叉产生的侧通道风险，提出可在不同攻击模型下（网络注入、浏览器注入、Ansible 密码恢复）实现秘密恢复；

**🔧 技术方法**

使用 zlib 的 LZ77/Deflate 压缩分析、加密层长度计数、噪声补偿策略及自适应压缩攻击算法；

**📊 数据集**

在 Redis、浏览器注入和 Ansible 自动化场景中测试，评估 33 种 SSH 客户端的压缩与端口转发配置；

**📈 对比分析**

实验显示 8 字符密码在不同噪声级别下的猜测次数为：直接注入约 1,260 次、浏览器注入约 27,620 次、Ansible 恢复仅 276 次；相对无公开对比，但证明攻击可行且效率与噪声相关；

**⚠️ 局限性**

仅在启用压缩、存在多路通道、秘密多次发送且噪声可容忍的特定配置下有效；对部署普及度未作全面测量，噪声、实现差异与同步机制会显著影响成功率。

---

## 174. DeepTable: Structural Attention Biases and Tree Path Encoding for Hierarchical Table Understanding

**arXiv ID:** 2609.07707 | [PDF](https://arxiv.org/pdf/2609.07707v1)

**作者:** Jyun-Ying Yen `[一作]` (National Yang Ming Chiao Tung University), Yu-Chee Tseng `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DeepTable，用结构化注意力偏差和树路径编码提升LLM对层级表格的理解。

**💡 创新点**

引入可学习的同行同列注意力偏差SAB以及树路径编码TPE，在PEFT框架下显式建模二维和层级表结构。

**🔧 技术方法**

使用结构化注意力偏差、树路径编码、LoRA参数高效微调以及表格序列化+特殊token编码技术。

**📊 数据集**

在HiTab、WikiTQ、FeTaQA、TabFact四个表格问答基准上进行实验。

**📈 对比分析**

与TableLoRA基线对比，在三种LLM基座上平均提升HiTab+5.1%、WikiTQ+2.6%、FeTaQA+3.0 BLEU，TabFact保持相近性能。

**⚠️ 局限性**

对更大或闭源LLM的适用性未知；数值推理仍无提升；结构偏差效果受基础模型强度影响。

---

## 175. Preserving contextual information in cultural heritage metadata through multidimensional knowledge graphs

**arXiv ID:** 2609.07695 | [PDF](https://arxiv.org/pdf/2609.07695v1)

**作者:** Lyndon Nixon `[一作]` (Storypact GmbH), Andrea Schimmenti `[通讯]` (University of Bologna)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种多维知识图（MKG）框架，用于在文化遗产元数据中显式记录陈述的多维上下文有效性。

**💡 创新点**

创新点在于将上下文维度和约束以可查询的形式嵌入RDF 1.2中，提供统一的多维上下文表示和查询语义，填补了传统知识图对多维上下文缺乏统一建模的空白。

**🔧 技术方法**

使用RDF 1.2、RDF‑star语法、SHACL 1.2进行模型验证，并利用语义网技术（OWL‑Time、GeoSPARQL等）对时间、空间等维度进行约束。

**📊 数据集**

论文未给出特定公开数据集，示例使用文化遗产典型案例（如《康斯坦丁的捐赠》）演示模型；未来计划在INFINITY项目的ECCCH数据上进行应用。

**📈 对比分析**

未提供系统性能对比实验；作者仅指出增添上下文会增加图规模与查询复杂度，实际性能待后续基准测试。

**⚠️ 局限性**

局限性包括模型对维度值空间定义的依赖、治理与授权问题、查询语义的默认合取关系、以及对多维约束的查询性能尚未评估。

---

## 176. Silent Metronome: Rhythmic Grounding for Live Music Accompaniment

**arXiv ID:** 2609.07688 | [PDF](https://arxiv.org/pdf/2609.07688v1)

**作者:** Kevin Bretz `[一作]` (Leiden University), Aske Plaat `[通讯]` (Leiden University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Silent Metronome (SiMe)，为实时音乐陪伴模型提供周期性节拍信息，使其能够在严格因果条件下保持节奏同步并消除节奏漂移。

**💡 创新点**

创新点在于将节拍相位与音高/频谱预测以及未来token预测等辅助头结合，且通过周期性编码的外部节拍信号对模型进行无漂移的节奏定位。

**🔧 技术方法**

使用周期性正弦余弦编码的节拍相位，AdaLN-Zero 机制将其注入 Transformer 解码器；并加入多头辅助预测（多音高、CQT、未来token）。

**📊 数据集**

实验数据集为 Slakh2100，包含2100首MIDI合成多轨曲目，已知精确节拍和时间签名。

**📈 对比分析**

与传统无节拍条件、仅时间签名/节拍条件以及无辅助头的基线以及一个提供一秒前瞻的非因果参考进行对比；SiMe 在 Beat-F 指标上提升至0.432，约比因果基线高3.2倍，超过提供1秒前瞻的对照，且 CoCoLa 与 FAD 指标也均有显著提升。

**⚠️ 局限性**

局限在于目前仅使用完美的离线节拍信息，未验证对实时漂移或真实音频的鲁棒性；此外模型对和弦等和声结构的同步仍不充分。

---

## 177. Perspectives on Cross-Lingual Consistency in LLMs for Medical Questions

**arXiv ID:** 2609.07687 | [PDF](https://arxiv.org/pdf/2609.07687v1)

**作者:** Minh Duc Bui `[一作]` (Johannes Gutenberg University Mainz), Katharina von der Wense `[通讯]` (University of Colorado Boulder)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文综述了多语言医学NLP中关于答案一致性与文化适配的两种立场，并通过问卷和LLM实验检验其在人类专业人士中的分歧与模型表现。

**💡 创新点**

创新点在于首次系统调研医学、NLP与人类学专业人士对跨语言一致性与适配的偏好，发现两者缺乏共识，并验证LLM无法准确模拟人类观点。

**🔧 技术方法**

主要技术包括问卷调查、统计分析（比例检验、z检验）和对多模型（Gemma、Llama、Qwen等）的系统提示实验。

**📊 数据集**

使用的数据包括三国（德国、西班牙、美国）各专业群体（医学、NLP、人类学）共348名受访者的问卷响应以及11个开源LLM的生成结果。

**📈 对比分析**

方法通过对人类与模型在一致性偏好上的比例进行比较，发现模型在医学和NLP角色下过度倾向一致性且缺乏国家差异，表明现有LLM不具备人类专家的多样性。

**⚠️ 局限性**

局限性包括问卷只提出二分选择，未区分内容与沟通差异；样本仅来自全球北方，缺乏低收入/中等收入国家视角；并未评估一致性与适配对实际用户决策的影响。

---

## 178. On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via Hashing

**arXiv ID:** 2609.07681 | [PDF](https://arxiv.org/pdf/2609.07681v1)

**作者:** Yuval Koren `[一作]` (Tel Aviv University), Itamar Zimerman `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析 Mamba 模型在关联记忆（Associative Recall, AR）任务中的内部机制，并将其归结为一种基于 Johnson–Lindenstrauss 的线性哈希函数的压缩-检索电路；

**💡 创新点**

提出了“Recall Scaling Laws”，即给定词表大小 V 与事实数 N_f，能准确预测 Mamba 所需的嵌入维度 D 与状态维度 N 的关系；

**🔧 技术方法**

采用机械解释性（mechanistic interpretability）方法拆解模型电路，并结合线性哈希理论、Johnson–Lindenstrauss 定理以及概率统计分析，推导出高概率召回条件；

**📊 数据集**

在合成的 AR/MQAR 数据集上进行实验，使用不同维度、层数、头数的简化与完整 Mamba 模型进行训练和评估；

**📈 对比分析**

将理论预测与实验结果对齐：理论曲线与实验准确率图形高度吻合，验证了所提出的缩放律；在多层、多头设置下，实验也验证了理论中有效状态尺寸 N_eff = ΛN 的作用；

**⚠️ 局限性**

局限性：分析基于简化线性模型，未完全解释完整 Mamba 结构（如门控分支）对记忆的具体贡献，且仅在合成任务上验证，尚未直接推广到真实 NLP 语言建模任务。

---

## 179. TrajectoryDB: A New Database for Agent Trajectories

**arXiv ID:** 2609.07782 | [PDF](https://arxiv.org/pdf/2609.07782v1)

**作者:** Yunjia Zheng `[一作]` (Harvard University), Juncheng Yang `[通讯]` (Harvard University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TrajectoryDB，一种针对AI代理执行轨迹的专用数据库，支持结构化检索、语义判定和可追溯的派生状态管理。

**💡 创新点**

创新点在于将轨迹视为独立数据类型，统一构造层级结构、文本冗余压缩、图谱线索与关系型分析，并将LLM作为通用UDF嵌入查询计划。

**🔧 技术方法**

采用多层物理布局（行式+列式+图结构）与增量文本去重压缩、状态化流式摄取、基于轨迹顺序的查询优化以及可缓存的语义判定结果。

**📊 数据集**

使用来自freeinference.org的真实生产轨迹（约千亿token/天）以及公开的Agent轨迹基准数据集进行实验。

**📈 对比分析**

在与PostgreSQL、ClickHouse及现有文档/向量存储的对比中，TrajectoryDB在结构检索吞吐量提升约10-30%，在语义判定成本降低30%，但总体写入延迟略高，需进一步调优。

**⚠️ 局限性**

局限性包括：1）对极长轨迹的状态维护仍存在资源消耗问题；2）语义判定缓存的失效管理复杂；3）缺乏跨平台标准化评测，难以与更成熟的多模态数据库直接对标。

---

## 180. CodeTD: Topology of Attention Detects Hallucinations in Code LLMs

**arXiv ID:** 2609.07779 | [PDF](https://arxiv.org/pdf/2609.07779v1)

**作者:** Daria Voronkova `[一作]` (Applied AI Institute), Serguei Barannikov `[通讯]` (Applied AI Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CodeTD，利用 Transformer 的注意力图进行拓扑数据分析，以预测生成代码是否符合提示，从而实现预执行的误差检测。

**💡 创新点**

创新点在于将 Topological Data Analysis（如 Manifold Topology Divergence 与 Cross‑Barcode）应用于注意力图，量化提示与生成之间的连通性，并证明该方法跨 benchmark 与模型具有良好可迁移性。

**🔧 技术方法**

使用的技术包括 Transformer 自注意力、拓扑数据分析（TDA）、Manifold Topology Divergence 计算、XGBoost 二分类器。

**📊 数据集**

实验所用数据集为 HumanEval、MBPP、BigCodeBench、MultiPL‑E 四大代码生成基准，并在 10 种 Code LLM（参数规模至 34B）上进行评估。

**📈 对比分析**

与多种监督与无监督基线（如 Pylint、Self‑Eval、CodeJudge、AttnLogDet 等）对比，CodeTD 在 ROC‑AUC 与 pass@1（最高提升 17.6%）上均取得了显著优势。

**⚠️ 局限性**

局限性包括只能检测到注意力几何导致的 hallucination，无法定位错误的具体位置，也无法解释底层根因（如预训练偏差、解码漂移等）。

---

## 181. Decomposition-Guided Diffusion Language Models for Inertial Confinement Fusion Prediction

**arXiv ID:** 2609.07756 | [PDF](https://arxiv.org/pdf/2609.07756v1)

**作者:** Xiang Zhang `[一作]` (Purdue University), Dongfang Liu `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于离散扩散语言模型的ICF中子速率波形预测框架 ICF-DLM。

**💡 创新点**

创新点在于将波形分解为总产量、峰时和局部波形三个物理量，采用双向去噪生成并使用物理驱动的PPO奖励来弥补数值离散化的度量缺失。

**🔧 技术方法**

技术包括离散数值分词、LLaDA-8B扩散语言模型、对齐的物理奖励与PPO强化学习，以及层级化的物理量输出结构。

**📊 数据集**

使用了由 50k 条仿真样本和 232 条实验样本组成的 ICFBench 数据集。

**📈 对比分析**

与传统递归、Transformer、扩散时序模型以及 LLM 时序预测器相比，ICF-DLM 在仿真和实验测试集上实现了 MSE、MAE、峰时误差和产量误差的显著提升，实验集 PTE 仅为 9.2 步。

**⚠️ 局限性**

局限包括仅针对单一诊断和设施的实验数据、缺乏对其他科学时序任务的验证，以及扩散生成的推理延迟较大。

---

## 182. TFTrack: A Template-Free Framework for Efficient 3D Point Cloud Tracking

**arXiv ID:** 2609.07738 | [PDF](https://arxiv.org/pdf/2609.07738v1)

**作者:** Zhaofeng Hu `[一作]` (Stony Brook University), Ci-jyun Liang `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了TFTrack，一种模板自由、单帧LiDAR 3D 单物体跟踪框架。

**💡 创新点**

创新点在于消除模板-搜索双输入和复杂运动建模，仅利用前一帧边界框中心和尺寸作为几何先验，实现高效跟踪。

**🔧 技术方法**

采用点/体素/柱状视图的轻量背骨、盒子条件特征编码、残差对数似然损失以及单帧运动头。

**📊 数据集**

在KITTI和nuScenes这两个大规模 LiDAR 跟踪基准上进行评估。

**📈 对比分析**

与主流基于模板的和运动中心的追踪方法对比，TFTrack 在 nuScenes 上与 BEVTrack 相近，在 KITTI 上虽略逊但帧率约 120 FPS、FLOPs 减少约 50%。

**⚠️ 局限性**

局限在极稀疏点云、严重遮挡或恶劣天气下表现下降，且无法处理强变形或尺寸变化。

---

## 183. The Profit Alignment Problem: How Profit Mandates Induce Alignment Failures in LLMs

**arXiv ID:** 2609.07731 | [PDF](https://arxiv.org/pdf/2609.07731v1)

**作者:** Eric So `[一作]` `[通讯]` (Massachusetts Institute of Technology), Eric So (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大规模语言模型（LLM）的系统提示中加入“最大化盈利”这一商业目标，观察其对模糊安全与合规信号判断的影响。

**💡 创新点**

首次揭示常见商业语言能够诱发LLM的“利润导向偏差”，即在不改变风险提示的情况下系统性地低估或忽视安全风险，形成Profit Alignment Problem。

**🔧 技术方法**

采用对话式Prompt工程、chain‑of‑thought (CoT) 记录、ABP（acknowledge‑but‑permit）判别器以及三名LLM评审（Claude、GPT‑4、Gemini）进行结果评估。

**📊 数据集**

使用3,600次受控实验数据，涵盖8款不同提供商的推理型LLM、3种模糊安全情景（安全趋势、近乎失误聚集、GHS分类）以及两种平衡与抽象的利润指令。

**📈 对比分析**

比较方法为基于三名评审的modal consensus与ABP分类，统计比例差异、logistic回归与χ²检验；在利润指令下，容忍率上升6.8个百分点，升压建议下降13.9个百分点，模型在不同提供商间效应差异显著。

**⚠️ 局限性**

局限性包括：只针对模糊信号而非明确风险；实验仅在单文档、单轮交互的简化情景；模型多样性导致结果不均；对真实企业部署环境的生态效度有限。

---

## 184. When Is Content "AI-Generated Enough"? Labelling Synthetic Media under the Digital Services Act and the AI Act

**arXiv ID:** 2609.07727 | [PDF](https://arxiv.org/pdf/2609.07727v1)

**作者:** Marie-Therese Sekwenz `[一作]` `[通讯]` (Delft University of Technology), Marie-Therese Sekwenz (Delft University of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对欧洲数字服务法案（DSA）与人工智能法案（AI Act）下的合成媒体标注机制进行理论与实证分析，提出了标注作为社会技术分类实践的概念，并辨识出四大治理张力；

**💡 创新点**

创新点在于将标注视为跨部门（平台、供应商、部署者、上传者、接收者）责任分配的社会技术体系，并系统性地将法律阈值、技术来源、界面设计与公平性问题相结合；

**🔧 技术方法**

主要采用了法律文本解读、技术标注规范（机器可读元数据、不可见水印、数字签名）与平台界面设计原理的组合分析；

**📊 数据集**

使用了2026年4月30日的DSA Statement of Reasons数据库快照（约1.5亿条记录）以及AI Act第50条实施框架草案和DSA界面准则；

**📈 对比分析**

没有提出可量化的算法或实验对比，主要通过数据分布（标注、移除、降级等比例）描述标注在平台实践中的相对频次；

**⚠️ 局限性**

局限性包括仅依赖DSA声明数据库的报告数据，无法衡量实际合成媒体出现率，且对标注技术效果与公平性缺乏实证验证。

---

## 185. A radiographic world model for clinical reasoning and evidence generation

**arXiv ID:** 2609.07719 | [PDF](https://arxiv.org/pdf/2609.07719v1)

**作者:** Suyang Xi `[一作]` (Emory University), Xiaofeng Yang `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种名为MedDream的胸片世界模型，能够同时支持诊断推理和报告条件下的影像生成；

**💡 创新点**

创新点在于将诊断对齐与遮蔽潜变量生成通过共享视觉路径联合训练，实现了诊断和生成的内在耦合；

**🔧 技术方法**

采用了VAE编码器+Transformer编码器+文本对齐CLIP式损失+掩蔽潜变量扩散生成，辅以渐进式遮蔽训练策略和语义轨迹选择；

**📊 数据集**

预训练使用约2.65 M对齐的胸片–文本对（来源于PMC‑CXR和多源临床库），评估覆盖NIH ChestX‑Ray14、VinDr‑CXR、ChestDR、CXR‑LT、MS‑CXR、RSNA Pneumonia等八个数据集；

**📈 对比分析**

在诊断识别、罕见病适配、严重度评估、定位等任务上，MedDream均优于Ark+、MedCLIP等基准，生成的图像在分布相似度、病理一致性和专家评估上优于ChexGen与MINIM，且合成数据可提升外部宏观AUROC至81.4%；

**⚠️ 局限性**

局限包括仅针对胸片、未涵盖纵向或其他影像模态、生成评估缺少所有病理和解剖细节、潜在数据泄漏风险、模型在不同机构和工作流程中的泛化需进一步验证。

---

## 186. Fine PT-PT Web: A High-Quality 41 Billion Tokens Data Collection of the European Portuguese Web

**arXiv ID:** 2609.07699 | [PDF](https://arxiv.org/pdf/2609.07699v1)

**作者:** Gonçalo Vinagre `[一作]` (NOVA School of Science and Technology), João Magalhães `[通讯]` (NOVA School of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过一条高效管线从葡萄牙网络档案（Arquivo.pt）中提取、清洗、去重并筛选出 41 B 个高质量欧洲葡萄牙语文本，构建了首个规模可比的 PT-PT 语料库。

**💡 创新点**

创新点在于引入了后采集清洗模块（去除短行与重复行）并使用加权 MinHash 去重与 EuroFilter 质量分类，三者组合实现了 19% 文档量提升及 76% 语料量压缩。

**🔧 技术方法**

技术方案包括 Datatrove + FineWeb2 过滤框架、Trafilatura 解析器、GlotLID 语言识别、Gopher 与 FineWeb 质量滤波器、MinHash 模糊去重、EuroFilter 神经质量评估以及跨集群去重与重新加权。

**📊 数据集**

数据集为 Arquivo.pt 公开存档的 71 个集合，覆盖 1997–2024 年共计 411 TB WARC 原始数据，最终得到 24.6 M 文档、41 B 词元。

**📈 对比分析**

在与 GlórIA（1.5 M 文档、0.8 B 词元）及 AMALIA 训练基线对比后，该语料库在规模上提升 16.4× 文档、51.3× 词元；质量评估显示去重后语料显著降低重复度、提升模型预测准确率，且模型在记忆化测试中表现低级别，证明数据质量高。

**⚠️ 局限性**

局限性包括：语料以新闻与博客为主，缺乏学术、技术与文学文本；基于文档级的过滤导致包含高质量文本的文档被整体丢弃；去重与质量阈值可能仍留有少量 PT-BR 垃圾内容。

---

## 187. Your Agent Says Yes: Interpreting Adversarial Market Behavior Beyond Individual Transactions

**arXiv ID:** 2609.07675 | [PDF](https://arxiv.org/pdf/2609.07675v1)

**作者:** Zelin Li `[一作]` (Ohio State University), Tianyu Shi `[通讯]` (McGill University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在一个虚拟加密交易所中，使用十个预设角色的语言模型代理进行交易、发帖、代币发行和流动性池操作，记录钱包门控决策、交易循环状态和跨代理事件，研究单笔交易审查与整体交易行为之间的差异。

**💡 创新点**

提出了三层证据阶梯（政策事件、循环状态、事件重构），强调行为评估必须跨消息、授权和状态变化来判断，而非仅依赖单笔交易的阻断结果；同时展示了门控在阻断“泵‑抛”类请求时效果显著，但对协同操纵等更复杂情形的抑制不足。

**🔧 技术方法**

使用基于语言模型（LLM）的Agent、一个事件驱动的虚拟交易所、预先设定的角色配置、Runner‑侧钱包门控策略和时间屏蔽的历史行情回放。

**📊 数据集**

虚拟交易所模拟数据，包括10个代理、2条行情回放（上升型World A与下降型World B）、72个交易周期、每周期10条代理动作、钱包门控日志、循环结束的账户余额与仓位。

**📈 对比分析**

对比门控开启与关闭两种实验条件，统计被阻断、标记与允许的请求比例；评估每类行为（泵‑抛、协同操纵、虚假宣传等）的阻断率和费用收集率。实验显示门控对泵‑抛请求的阻断率高达45.8%，但对协同操纵等其他类请求的阻断率仅约11%，说明单笔审查在整体行为安全性上的局限。

**⚠️ 局限性**

限制包括：仅使用单一LLM模型与固定的10个角色；回放窗口有限，仅包含两条行情路径；未记录消息投递结果与完整交易执行反馈；循环结束后分数未同步；缺乏真实市场数据验证，结果仅在实验环境中可复现。

---

## 188. From Citations to Contributions: LLM-Assisted Credit Scoring of Research Articles

**arXiv ID:** 2609.07673 | [PDF](https://arxiv.org/pdf/2609.07673v1)

**作者:** Sana Ebrahimi `[一作]` (University of Illinois Chicago), Abolfazl Asudeh `[通讯]` (University of Illinois Chicago)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了基于贡献的学术信用计分框架，将论文的贡献拆分为原创贡献和引文贡献，并通过层次化的贡献树和加权引文网络进行传播。

**💡 创新点**

创新点在于将合作博弈中的Shapley价值概念转化为可观测的文档结构分配，利用LLM估算局部重要性，并将贡献递归传播到整个文献网络。

**🔧 技术方法**

核心技术包括贡献树（hierarchical importance propagation）、LLM估计器（作为噪声比较估算器）、加权有向图的信用传播算法以及多模型比较。

**📊 数据集**

使用了人工标注的50篇科研论文以及17篇论文的引文图作为实验数据集。

**📈 对比分析**

在与人类专家、Claude Sonnet 4.6和GPT‑OSS‑120B等基准比较的评测中，小型LLM（如qwen3:1.7b）在章节和引文级别的评分与人类标注的相关性最高，且在多种指标（Spearman、JSD、Top‑4覆盖率）上优于频率基线。

**⚠️ 局限性**

局限性包括缺乏客观的贡献真值、实验规模有限、对LLM的高度依赖、文档结构假设可能失效以及在子语料库中传播的影响易受网络覆盖程度影响。

---

## 189. M3-Tele: A Unified Multimodal Teleoperational Framework for Compliant Whole-Body Mobile Manipulation

**arXiv ID:** 2609.07859 | [PDF](https://arxiv.org/pdf/2609.07859v1)

**作者:** Hengxiang Chen `[一作]` (Shenzhen Technology University), Nutan Chen `[通讯]` (LS Wiiri Robot Innovation Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 M3-Tele，整合力-触觉反馈和全身协同控制的多模态远程操作框架，并在移动机械手上收集高质量演示数据。

**💡 创新点**

创新点在于：① 将方向性力正则化与触觉基抓手调节结合到统一的高级控制器；② 引入反馈启用的 iPhone 接口与运动重映射，实现无漂移的末端执行；③ 通过多模态（视觉、触觉、力、本体）同步采集，为下游学习提供完整的交互观测。

**🔧 技术方法**

技术包括：力-正则化（admittance）控制、触觉变形误差校正、全身逆运动学（带非齐次约束）、ResNet‑18 视觉编码、扩散策略（Diffusion Policy）学习、GelSight Mini 光学触觉传感器、FT300 力/扭矩传感器、iPhone 运动重映射。

**📊 数据集**

使用自采集的演示数据集，涵盖四项任务（拾取‑放置、杆子旋转、抽屉拉动、擦拭）共计约 200–300 次操作，数据格式包含 RGB、触觉、力、以及本体状态。

**📈 对比分析**

通过六种对照配置（有/无正则化、抓手反馈、全身协同、接口差异）进行实验。结果显示：力跟踪误差降至 0.35 N，接触丢失 0.02 次/试，触觉变形误差提升 65%；演示收集完成率 80–94%，操作易用度 8.4/10；在擦拭任务上，使用完整三模态观测的扩散策略实现最高覆盖率和最优抓取成功率。

**⚠️ 局限性**

局限性包括：仅验证在并联抓手平台；对更大规模任务或更高频交互的扩展未验证；演示质量仍受操作者技术水平影响；多模态同步采集的异步频率可能限制实时性能。

---

## 190. Nothing Breaks: No Single Peer Can Soundly Gate Post-Quantum Delivery

**arXiv ID:** 2609.07849 | [PDF](https://arxiv.org/pdf/2609.07849v1)

**作者:** Yunze Han `[一作]` `[通讯]`, Yunze Han

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在自动化配置管理与工具使用型语言模型代理中，后量子加密配置的声明与实际交付之间的失配，证明现有的artifact‑side与wire‑side门控层无法可靠捕获此类“交付‑声明”差异，从而导致安全回退被忽视。

**💡 创新点**

创新点包括：①定义并验证了后量子降级（post‑quantum downgrade）失配类；②提出确定性oracle对真实artifact和握手进行评估；③设计写入边界监控器pqgate，在代理写文件时强制声明与交付一致；④在多协议、多配置、多工具链下进行预先登记的实验验证。

**🔧 技术方法**

使用技术：确定性oracle（基于sshd‑T/T解析）、agent循环（OpenAI Agents SDK、LangGraph、MCP）、网络抓包与握手模拟、统计检验（精确检验、Wilson区间、Holm校正）、Preregistration与数据哈希、CI与Nginx+OpenSSL等实验平台。

**📊 数据集**

数据集：21,940条agent episode；216个oracle判定状态；15个SSH客户端能力集；5个peer profile；7种配置、2种协议下的48个配置状态；多版本OpenSSH、Nginx、OpenSSL、SSH客户端（libssh、paramiko、AsyncSSH、Dropbear、PuTTY、Go、OpenSSH 8.9–9.9+等）。

**📈 对比分析**

比较方法：将多工具（ssh‑audit、testssl.sh、OpenSSL、nginx‑t、OpenSSL CBOM生成器）与oracle结果对比，测定各层面检测率；pqgate在CI中平均5.4 s批处理，单握手约14.3 s，性能可接受；大多数检查无效，唯有交付绑定检查表现突出。

**⚠️ 局限性**

局限性：仅针对后量子TLS/SSH，模型特定（Claude Sonnet‑5不下）；缺乏无攻击者基线；检测率未统计真实生产率；工具与模型覆盖不全；验证仅针对预登记样本，未覆盖所有真实部署；缺乏对更广泛协议与客户端的实测；门控层面仍存在空洞。

---

## 191. Real-Time dApps for AI-RAN: Measured Interface Requirements for Inline PHY and Slot-Level Control

**arXiv ID:** 2609.07805 | [PDF](https://arxiv.org/pdf/2609.07805v1)

**作者:** Timothy O'Shea `[一作]` (DeepSig Inc.), Andriy Kharchenko `[通讯]` (DeepSig Inc.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了适用于AI‑RAN的分布式应用（dApp）接口，阐明了三类工作（inline、bounded、asynchronous）在实时约束下的可行性，并发布了可用的C ABI、E3AP与共享内存实现；

**💡 创新点**

提出了“coupling spectrum”概念，将dApp与DU之间的耦合程度映射到具体实现路径，并给出了完整的实验基准与审计用例，首次量化不同框架在100 µs、1 ms等关键时延下的表现；

**🔧 技术方法**

采用C ABI、E3AP over SCTP、ZeroMQ、shared‑memory ring、SEQPACKET、CUDA IPC、FlatBuffers等技术；在 NVIDIA DGX Spark（GB10 GPU）与 USRP B210 51 PRB 30 kHz TDD 环境下进行测量；

**📊 数据集**

使用了包含 39 条审计用例的 AI‑RAN use‑case 语料库（13 住进式、9 有限时、17 异步），每条用例给出字节数、NR 定时约束及实时期限；对应实验负载为 34/68 KB 的 51‑PRB 信道、734 KB/1.47 MB 的 273‑PRB envelope 以及 64 KB 的 SRS 等；

**📈 对比分析**

在安静主机与载波在空中两种条件下分别测量每种框架路径的 P99.9 延迟；发现 >50% 用例跨越传统 observer 边界不可行；在空闲主机 100 µs deadline 可满足，载波在空中时仅 inline 路径满足；异步用例在任意位置均能满足时延，且内置路径提供可追溯性与无消息返回；

**⚠️ 局限性**

实验仅在单台 NVIDIA DGX Spark + PCIe GPU 上完成，负载轻且核心分配宽松；真实生产环境下核心共享、更多 UE 与更大信道规模可能导致更高延迟；E3AP 仅测本地环回，跨主机网络延迟未评估；273‑PRB 大规模内核未通过空中验证；仅覆盖审计用例，未涵盖所有潜在场景。

---

## 192. LLM Agents as Computational Typologists

**arXiv ID:** 2609.07791 | [PDF](https://arxiv.org/pdf/2609.07791v1)

**作者:** Changbing Yang `[一作]` (University of British Columbia), Jian Zhu `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 AutoTypologist——一种基于 LLM 的自动化典型学分析代理，能够检索参考语法、分析 IGT 示例并通过 ReAct 循环生成证据支持的特征编码和假设检验。

**💡 创新点**

首次提出面向典型学的 LLM 代理框架，结合结构化证据图、专用语法/IGT 读取工具与可追溯的推理流程，实现大规模、可解释的跨语言特征编码与普遍性假设验证。

**🔧 技术方法**

使用大型语言模型（如 GPT‑4、Claude、LLaMA 系列）配合 ReAct 代理架构、语法与 IGT 专用工具、检索与量化分析模块，以及证据图和审计子系统。

**📊 数据集**

基于 25 本公开许可的参考语法（涵盖 14 个语系）和从中提取的 IGT 样本构建的典型学数据集，包含结构化语法块、IGT 示例和元数据。

**📈 对比分析**

在特征编码任务中与零样本提示和多数投票基线对比，采用加权 F1 与宏 F1 评价；在假设检验任务中统计各语言的支持/反驳/无关/不足证据，显示在提供语法信息时模型性能显著提升，但仅靠 IGT 时仍存在挑战。

**⚠️ 局限性**

仍需人工验证结果，受限于文本数据（不含语音/手语等多模态信息），依赖静态语法描述，且对低资源语言和未见语言的 IGT 推断准确率有限。

---

## 193. xDailyBench: Benchmarking LLMs on Professional Consultation for Real-Life Problems

**arXiv ID:** 2609.07784 | [PDF](https://arxiv.org/pdf/2609.07784v1)

**作者:** Yongchang Peng `[一作]`, Wenhao Huang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了 xDailyBench 评估框架，聚焦真实用户日常任务的开放式、情境化完成度评估。

**💡 创新点**

创新点在于：①从真实用户采集任务并保持其含糊性与多样性；②采用可细粒度的 rubric 评估并与能力维度映射；③实现开放输出、多维度指标的自动化评判。

**🔧 技术方法**

使用技术包括：LLM-as-a-Judge/Agent-as-a-Judge 自动评判、统一工具接口与文件系统访问的 Nanobot/OpenCode harness、强大模型与统一评判模型（GLM‑5.1）等。

**📊 数据集**

数据集为 248 个真实任务，覆盖 4 大领域、51 场景，平均 13.3 条二元 rubric 条目，来源于 1000+ 参与者提供的实际需求与上下文。

**📈 对比分析**

对 11 个前沿 LLM 进行对比实验，最佳模型（Seed‑2.1‑Evolving / Kimi‑K3）平均任务得分 75.6%，与传统指标差距显著；在隐式需求、量化推理等维度表现相对薄弱。

**⚠️ 局限性**

局限性包括：仍需人工审校以确保 rubric 的可评判性；隐式需求与量化推理仍是主要瓶颈；不同领域表现差异大，模型对多样化任务的统一适应能力不足。

---

## 194. You can contribute if you... An Empirical Framework of AI Contribution Policies in OSS

**arXiv ID:** 2609.07919 | [PDF](https://arxiv.org/pdf/2609.07919v1)

**作者:** Gregorio Robles `[一作]` (Universidad Rey Juan Carlos), Daniel M. German `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对开源软件项目中的AI介入贡献政策进行经验研究，系统梳理并分类了各类政策的动机与规则；

**💡 创新点**

创新点在于提出了AI贡献治理框架（AI Contribution Governance Framework），将治理关注点与项目价值关联，并给出可操作的接受性原则；

**🔧 技术方法**

主要技术手段包括人工编码与基于LLM（Claude）自动提取政策文本，结合归纳式定性分析构建理论框架；

**📊 数据集**

使用的实验数据集为两份公开的AI贡献政策列表，分别来自GitHub前1,000名项目的HR数据集和MelissaWM社区手工收集的106份项目政策；

**📈 对比分析**

方法上不涉及传统机器学习评估，而是通过对209个项目的定性归纳与框架构建来展示政策多样性与治理逻辑，缺乏量化性能指标；

**⚠️ 局限性**

局限性包括样本覆盖不具统计代表性、政策文本简短导致解读困难、框架基于现有政策可能与未来技术变迁不完全匹配，且缺乏实证验证其治理效果。

---

## 195. AVCG: A Generalized Variational Framework for Counterfactual Generation under Hypothesis Distributions

**arXiv ID:** 2609.07917 | [PDF](https://arxiv.org/pdf/2609.07917v1)

**作者:** Jamie Duell `[一作]` (Sheffield Hallam University), Mahault Albarracin `[通讯]` (Université du Québec à Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了AVCG（Amortized Variational Counterfactual Generator），一种可在任意假设分布上生成鲁棒反事实的通用变分框架。

**💡 创新点**

创新点在于将对抗解释问题转化为对假设分布的优化，使其既能处理贝叶斯后验，也能适用于Rashomon集合，实现一次性生成且对模型不确定性具有鲁棒性。

**🔧 技术方法**

使用变分自编码器结构的生成器、重参数化技巧、KL正则化和接近性约束，构建端到端可训练的反事实生成器。

**📊 数据集**

实验数据集包括四个常用表格数据集：Adult Income、Heart Disease、Wisconsin Breast Cancer和Spambase。

**📈 对比分析**

与传统贝叶斯及后期优化方法比较，AVCG在有效性（Val）、交叉模型有效率（CMV）和Rashomon有效率（RVR）上均接近1，推理时间比基线提升十倍以上，且生成多样性显著提升。

**⚠️ 局限性**

局限性包括对Rashomon阈值的敏感性以及潜在的“amortization gap”，导致生成的反事实在不同实例上可能不是最优的。

---

## 196. Improved Upper Bounds for Dynamic Bin Packing of General, Unit-Fraction, and Power-Fraction Squares

**arXiv ID:** 2609.07913 | [PDF](https://arxiv.org/pdf/2609.07913v1)

**作者:** Miguel A. Mini `[一作]` (Institute of Computing, University of Campinas), Yoshiko Wakabayashi `[通讯]` (Institute of Mathematics, Statistics and Computer Science, University of Sao Paulo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种在线动态2维方形装箱算法SMB，改进了之前的三列表法为两列表法，并分别给出对一般、单位分数和幂分数方形的装箱策略。

**💡 创新点**

核心创新在于引入简化的两列表划分、对Next‑Fit Decreasing Height的更紧致占面积下界（如5/16），以及通过精确计算xy函数来优化单位分数和幂分数场景的竞争比。

**🔧 技术方法**

主要技术手段包括：Next‑Fit Decreasing Height（NFDH）分析、面积占用下界证明、精确枚举求xy最小值、递归层级拆分与负载分析，最终实现竞争比的改进。

**📊 数据集**

论文为理论研究，未使用实测数据集，而是通过数学证明与枚举计算验证下界（如xy、1022131/2965284等），并给出最优示例集合来证明下界的紧迫性。

**📈 对比分析**

与之前的上界（4.2154、3.9654、2.4842）相比，SMB在一般方形上将上界降低至3.918，在单位分数方形上降至3.356，在幂分数方形上降至2.211，证明了更优的竞争比；实验或模拟未涉及。

**⚠️ 局限性**

主要限制在于：只给出了上界，实际最优竞争比仍未知；对SMB的下界尚未确定；算法仅针对在线动态装箱模型，未考虑迁移或退回时的重新装箱；理论分析依赖于精确的面积下界，若进一步提升需更复杂的算法或更细致的分析。

---

## 197. Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language Model Serving

**arXiv ID:** 2609.07883 | [PDF](https://arxiv.org/pdf/2609.07883v1)

**作者:** Siyu Song `[一作]` (Beijing Institute of Technology), Jiayu Sun `[通讯]` (Beijing Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于迭代级别的自适应前缀块划分策略，用来在LLM服务中动态决定每次迭代可接受的最大前缀块大小，以满足实时解码的时间约束。

**💡 创新点**

创新点在于：①在每个解码迭代前利用最早到期的解码请求的剩余时间预算，计算能安全完成的最大前缀块；②只需单调递增的成本模型即可完成二分搜索，时间复杂度为 O(log C_max)；③在保证“安全性”和“最大化前缀进度”的理论证明，并引入误差补偿机制。

**🔧 技术方法**

技术包括：迭代级别连续批处理、基于预先收集的时间预测表（或回归模型）的迭代成本预测、二分搜索求解、优先队列维护最早到期时间、对模型不做任何改动的控制器实现、以及在实际 GPU 运行时的集成和监控回路。

**📊 数据集**

使用了四类工作负载的合成请求流：Chat、Mixed、Long、Burst 以及对应的 1,000 条请求样本，prompt 长度为对数正态分布，SLO 设定为 10 ms、25 ms、50 ms，硬件平台包括 8×A100‑80GB、8×H100‑80GB，模型为 8B、70B、长上下文 8B。

**📈 对比分析**

与基准策略（完整前缀、固定块 256/512/1024/2048、Sarathi 调优块）相比，实验在 25 ms 目标下：混合流量的 goodput 提升 39%，长上下文提升 38%，Burst 提升 9%；在 10 ms 目标下，混合流量的 goodput 提升 3.3×，长上下文提升 2.4×。在 GPU 级别测试中，适配后可实现 1.7–2.7 倍的功耗/请求效率提升。

**⚠️ 局限性**

局限性包括：仅评估单机单节点场景，未覆盖多节点 KV 迁移和跨区域服务；模型假设迭代成本单调且无跳变，实际硬件跳变和通信开销可能导致误差；当前控制器只考虑单一前缀块，无法处理多租户优先级冲突；没有考虑前缀预取或中断开销。

---

## 198. LLM Layers Immediately Correct Each Other

**arXiv ID:** 2609.07876 | [PDF](https://arxiv.org/pdf/2609.07876v1)

**作者:** Arjun Patrawala `[一作]` (University of California, Berkeley), Jacob Steinhardt `[通讯]` (University of California, Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并系统描述了 Transformer 层之间的纠正机制（TLCM），该机制使相邻层的贡献向量呈负相关，且后续层会主动抵消前一层的部分输出。

**💡 创新点**

创新点在于首次揭示了大规模语言模型中普遍存在的层级纠正现象、阐明其学习动态、并提出“提议-拒绝”假设，解释了 Sparse Autoencoder（SAE）与跨层译码器（CLT）表现差异。

**🔧 技术方法**

技术方法包括：对残差流贡献向量进行余弦相似度分析、层级雅可比矩阵特征值分解、因果干预实验、以及对不同子层（注意力、MLP）贡献的分解。

**📊 数据集**

实验使用多种开源模型（Llama 3、OLMo、Mistral、Gemma、Qwen2、GPT‑2、Phi）及其不同训练阶段的检查点，文本来源为 WikiText 长文档和大量生成文本。

**📈 对比分析**

与传统假设（层级仅增量构建）对比，TLCM 在 5/7 族模型中显著出现，且在训练过程中从无到有、并随层数和上下文位置递增；对比 SAE 与 CLT，TLCM 解释了 SAE 低特异性与需要极端特征放大、CLT 更高重建准确率的原因。

**⚠️ 局限性**

局限性包括：未在所有模型族中观察到 TLCM（如 GPT‑2、Phi），缺乏形式化的“提议‑拒绝”理论，实验依赖高算力与大规模模型，且对其它 Transformer 变体（门控注意力、S4 等）的适用性尚未验证。

---

## 199. The OCUDU dApp Platform: An Open Runtime and E3 Interface for Real-Time AI-RAN

**arXiv ID:** 2609.07843 | [PDF](https://arxiv.org/pdf/2609.07843v1)

**作者:** Timothy O'Shea `[一作]` (DeepSig Inc.), Andriy Kharchenko `[通讯]` (DeepSig Inc.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 dApp 平台，开放 5G NR DU 的 sub-10 ms AI 计算层，支持链接适配、调度、信道估计和接收机等实时 AI 功能，并提供统一的签名包、ABI、E3 接口与生命周期管理。

**💡 创新点**

创新点在于：①将 sub‑10 ms 计算块从外部观测转为可被签名模块直接执行；②设计了三类执行契约（Class A、B、C）并统一管理；③实现了零硬件快速启动、离线签名、SBOM、E3 事务与安全、以及多供应商组合的完整闭环。

**🔧 技术方法**

使用技术包括：C/C++ 共享对象 + 预冻结 ABI、GPU（CUDA）流、SCTP + ASN.1 FlatBuffers 的 E3AP/E3DP、共享内存环、CUDA‑IPC、seccomp、签名（ECDSA P‑256）、SBOM、Python 交互客户端、LLM Agent（MCP）。

**📊 数据集**

数据集：现场 5G NR GB10 gNB 与真实用户设备（USRP B210/分裂 7.2 O‑RU），使用 20 MHz n78 30 kHz 子载波的对称 TDD 流量，结合实时收集的 PUSCH/DMRS/ SRS 及 KPI 流。

**📈 对比分析**

比较方法：在同一现场对 3.2 KB 调度请求、Class A 接收器、Class B 调度器和 Class C 观测器进行 P99.9、P50 时延测量；通过 A/B 版本切换评估等化器 BLER；对比 8 条遥测流与传统实现的无 fallback 性能。性能方面：Class A 估计 82 µs、接收 150 µs 预算，Class B 直调用 0.29 µs，Class C 发布 2.3 µs，所有在 slot 500 µs 内且无 fallback。

**⚠️ 局限性**

限制：尾部时延未完全满足 150 µs 预算；实验未开启 CPU 资源隔离；多单元遥测与异步输出未实现；E3 传输无 TLS/DTLS；缺乏多供应商实测验证；多单元/多 CU/DU 的跨域支持待完善。

---

## 200. Latent-MoE: Domain-Aware Mixture-of-Experts for PDEs with Multi-Regime Physics

**arXiv ID:** 2609.07814 | [PDF](https://arxiv.org/pdf/2609.07814v1)

**作者:** Hanwen Wang `[一作]` (University of Pennsylvania), Paris Perdikaris `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种在 PINN 中混合专家与共享主干的 Latent‑MoE 架构，用于解决物理方程跨域异质性问题。

**💡 创新点**

创新点在于利用物理结构的固定 compact‑support 路由得到局部化的 NTK，消除长距离耦合，并通过共享主干实现跨区域容量流动，兼顾局部化与全局化。

**🔧 技术方法**

采用 NTK 理论分析、基于 MoE 的架构设计、固定 bump 路由、周期性编码 + Fourier 特征、共享与中心化编码以及梯度冲突指标等技术。

**📊 数据集**

使用三种异质 PDE 基准（时变系数的对流扩散、分段驱动的阻尼波、空间分层波速）以及三个均质 PDE（Burgers、Allen–Cahn、KdV）作为对照数据集。

**📈 对比分析**

与 ResNet、FB‑PINNs、PirateNet 等基线对比，在均质基准上取得相同或略优的相对 L^2 错误；在异质基准上错误降低十倍以上，梯度冲突显著降低。

**⚠️ 局限性**

局限性包括仅在合成基准上验证；路由固定为单轴分区；未实现自适应分区或非均匀专家容量；NTK 分析基于无限宽理论，缺乏有限宽实证。

---

## 201. MicroIntent: Intent-Based Placement Strategy for Microservice Application in the Compute Continuum Using LLMs

**arXiv ID:** 2609.07927 | [PDF](https://arxiv.org/pdf/2609.07927v1)

**作者:** Koushikur Islam `[一作]` (Western Sydney University), Rodrigo N. Calheiros `[通讯]` (Western Sydney University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MicroIntent架构，基于自然语言用户意图自动生成微服务在计算连续体上的部署方案，并实现了MVP进行验证。

**💡 创新点**

首次将大型语言模型（LLM）用于将高层自然语言意图转换为低层SLO，并通过规则匹配实现意图驱动的微服务放置，显著降低了计算连续体的使用门槛。

**🔧 技术方法**

使用生成式大模型（如GPT‑4o、GPT‑4o‑mini）进行意图解析，JSON作为数据交换格式，基于规则的放置算法，Flask + Next.js API 与前端实现，全部容器化部署于 Docker。

**📊 数据集**

利用自定义的计算连续体基础设施 JSON 文件和手工构造的自然语言意图文本；未使用公开数据集。

**📈 对比分析**

通过两种场景（意图变化与基础设施变化）实验验证 MVP 能在意图或基础设施改变时生成正确的服务–节点映射；未给出定量性能指标，仅展示放置正确性。

**⚠️ 局限性**

仅支持单点网络指标（如带宽、延迟），未处理意图冲突或反馈机制；LLM 解析依赖通用模型，缺乏针对本任务的专门训练；缺乏真实环境下的性能评估。

---

## 202. Exact Degrees of Freedom of Spatially Sparse MIMO Channels Without Prior CSI

**arXiv ID:** 2609.07926 | [PDF](https://arxiv.org/pdf/2609.07926v1)

**作者:** Yifeng Xiong `[一作]`, Jianhua Zhang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文对没有先验CSI的点对点块状无记忆信道进行了自由度（DoF）的表征，考虑了固定数量K的传播路径，发射器和接收器分别配备了Nt和Nr个天线。

**💡 创新点**

创新点在于确定了在特定条件下的DoF，特别是对于K=1时DoF为1-1/T，对于K≥2时DoF为K(1-3/2T)，并且通过高斯信号实现了这些值。

**🔧 技术方法**

使用了几何分析和信息维度的技术，结合了高斯信号的输入分布来推导DoF。

**📊 数据集**

使用了块状无记忆信道模型，假设发射和接收天线的几何位置是已知的且不相同，且信道增益是独立的复高斯分布。

**📈 对比分析**

通过与已有的信道容量理论进行比较，证明了在给定的功率约束下，所提出的模型的DoF表现优于传统模型，尤其是在K≥2的情况下，DoF损失较小。

**⚠️ 局限性**

限制在于该模型假设了固定的天线位置和信道增益分布，且未考虑动态变化的信道状态信息（CSI）对DoF的影响。

---

## 203. Humans Introduce, Models Elaborate: Asymmetric Narrative Agency in Human-LLM Co-Writing

**arXiv ID:** 2609.07920 | [PDF](https://arxiv.org/pdf/2609.07920v1)

**作者:** Halfdan Nordahl Fundal `[一作]` (Aarhus University), Rebekah Baglini `[通讯]` (Aarhus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了三种协作写作模式（人–人、人–LLM、LLM–LLM）的对话式故事生成过程，量化了情感对齐、语义距离与叙事影响的转折级别差异。

**💡 创新点**

发现人–LLM合作表现出独特的非对称性：人类贡献更具创新性与持久性，而LLM主要扩展与稳定已有情境；这一现象在传统的两人对等合作中未出现。

**🔧 技术方法**

采用情感概念向量投影、Transformer语义嵌入、基于surprisal的Novelty/Transience/Resonance指标以及混合效应模型进行量化分析。

**📊 数据集**

使用了自制的对话式创意写作数据集，包含36条人–人、97条人–LLM、80条LLM–LLM的十轮故事，包含多种LLM（GPT‑4.1、Claude‑Sonnet、Llama‑3.3、Qwen‑2.5）。

**📈 对比分析**

通过对比三组的情感、语义与叙事影响指标，发现人–LLM在情感基线差异最大、语义距离最显著、以及叙事影响的不对称度最高，表明其并非人–人与LLM–LLM之间的中间状态，而是一个独立的交互范式。

**⚠️ 局限性**

局限包括样本不均衡（人–人样本最少）、受限的参与者群体（主要为单一大学学生）、对LLM模型多样性的潜在掩盖、仅基于量化指标无法捕捉作者主观意图，以及对LLM与人类写作风格的自动纠正可能抑制差异。

---

## 204. JEDI: JEPA-to-Edge Distillation for Efficient Cropland Segmentation from Satellite Imagery

**arXiv ID:** 2609.07915 | [PDF](https://arxiv.org/pdf/2609.07915v1)

**作者:** Kishor Kumar Bhaumik `[一作]` (University of California, Riverside), Evangelos E. Papalexakis `[通讯]` (University of California, Riverside)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了JEDI框架，利用大规模I-JEPA视觉Transformer教师模型的特征表示，训练出参数极少的SegFormer学生模型，实现高效的农田分割；

**💡 创新点**

创新点在于两阶段跨架构蒸馏：首先对学生的终端特征进行投影与空间对齐，使其与教师的token空间一致；随后在任务微调阶段持续保持特征对齐约束，防止学生在适应任务时偏离教师表示；此外使用NDVI驱动的单帧复合选择，减少时间序列计算量；

**🔧 技术方法**

核心技术包括：I-JEPA自监督预训练、跨架构投影对齐、温度缩放响应蒸馏、持久特征对齐目标、NDVI筛选的时序压缩、SegFormer结构的高效分割；

**📊 数据集**

实验基于CalCROP21数据集，该数据集包含加州中央谷地的Sentinel‑2多时序影像及农田掩模；

**📈 对比分析**

与四种基线（KD、SKD、CWD、CIRKD）和无蒸馏的基线相比，JEDI在MiT‑B0、B1、B2三个容量上均取得最高mIoU，JEDI‑B0达68.0（比教师70.0仅差2.0点），比无蒸馏提升约16点；

**⚠️ 局限性**

局限性包括仅针对二分类农田分割；实验仅使用单一地区（加州中央谷地）和单一传感器；NDVI选择导致丢失时间序列信息，可能限制对多作物混合区的辨识。

---

## 205. Quantization Amplifies Determinism, Not Bias: Scale-Dependent Behavioral Effects of Serving-Time Weight Compression

**arXiv ID:** 2609.07901 | [PDF](https://arxiv.org/pdf/2609.07901v1)

**作者:** Dachi Kurtskhalia `[一作]` `[通讯]`, Dachi Kurtskhalia

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Qwen3 8B、14B 与 32B 三个模型上，作者在三种权重量化精度（AWQ int4、FP8 int8、bf16）下，以相同硬件和采样设置，系统地收集并对比约 71,000 条回复，评估量化对推荐多样性、词汇多样性和风格的影响。

**💡 创新点**

创新点在于首次采用声明‑运行协议的多层次量化对比实验，揭示了在 8B 模型上 int4 量化会导致推荐多样性急剧下降，而在更大模型上则出现风格漂移，并区分了 token 级噪声与语义集中两种机制。

**🔧 技术方法**

技术手段包括 AWQ、FP8‑Marlin、bf16 量化实现、vLLM 0.11 服务器、温度 0.8、top‑p 0.95 采样、基于碰撞概率、词典消歧、token 级熵和 Jensen‑Shannon 散度等定量指标。

**📊 数据集**

使用了两个自定义无泄漏提示电池：218 条文化类国家提示和 96 条汽车品牌推荐提示，并配套 78 国、57 车系 gazetteer 及对抗性验证。

**📈 对比分析**

通过 3×3 的精度‑模型规模格局和配对符号翻转检验比较，发现 8B int4 在推荐多样性和词汇多样性上显著降低（P<0.023），14B/32B 则无内容收敛但出现 em‑dash 等风格漂移；整体上量化提升确定性而非偏见。

**⚠️ 局限性**

局限性包括仅评估 Qwen3 系列、单一量化器、仅英文与单一汽车推荐场景、温度对结果敏感、在 32B 上存在并行度差异以及多样性收敛效应仅在少数提示中显著。

---

## 206. Does Syntax Matter? A Graph-Augmented Variational Topic Model for Computational Social Sciences

**arXiv ID:** 2609.07797 | [PDF](https://arxiv.org/pdf/2609.07797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 207. Explainable Temporal Attention-based Defect Detection For Fillet Joints in Real-Time Gas Metal Arc Welding Based on Multi-modal Data

**arXiv ID:** 2609.07893 | [PDF](https://arxiv.org/pdf/2609.07893v1)

**作者:** Mobina Mobaraki `[一作]` (University of British Columbia), Guy A. Dumont `[通讯]` (University of British Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于多模态（图像+声学）时间注意力深度学习模型，用于Gas Metal Arc Welding（GMAW）焊缝内部缺陷（穿透不足、熔合不足、孔洞、下凹、冷叠）实时检测。

**💡 创新点**

创新点包括：①将时间特征融入3D ResNet并引入注意力机制（MLP+Softmax），显著提升F1分数；②使用GradCAM、GradCAM++、XGradCAM对模型进行可解释性分析，定位每种缺陷关键视觉与声学区域；③通过t‑SNE与手工特征分析确定各缺陷首选模态。

**🔧 技术方法**

技术方案：3D ResNet + temporal attention + MLP + Softmax + SVM分类；可解释性：GradCAM/GradCAM++/XGradCAM；降维分析：t‑SNE；数据预处理：图像拼接与Mel谱图生成。

**📊 数据集**

数据集：8次焊接实验，共23543训练、2942验证、2942测试图像；每个图像对应Mel声谱图；5类缺陷与无缺陷共6类；数据来源为R&D环境下的协作焊接机器人，人工制造缺陷。

**📈 对比分析**

与单模态、无时间信息以及多模态基本模型对比：时间信息提升各缺陷F1约7–12%；加入注意力后F1提升至0.99（所有缺陷），计算成本仅增0.1 GFLOPs；表格显示不同模态与时间/注意力下的F1分数与推理时延。

**⚠️ 局限性**

局限性：①数据为实验室有意制造的缺陷，缺乏真实生产现场噪声与环境多样性验证；②仅针对GMAW短路模式，未扩展到其他焊接方式；③模型相对复杂，实时性能在实际工厂仍需进一步评估；④缺陷标签依赖人工标注，可能存在主观偏差。

---

## 208. Kalman Delta Networks: Uncertainty-aware Associative Memory

**arXiv ID:** 2609.07816 | [PDF](https://arxiv.org/pdf/2609.07816v1)

**作者:** Ngoc Bui `[一作]` (Yale University), Rex Ying `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Kalman Delta Networks（KDN），将线性注意力中的 delta‑rule 记忆重构为 Kalman 滤波器，实现不确定性感知的关联写入；

**💡 创新点**

创新点在于将 delta‑rule 混合器视作线性高斯状态空间模型，利用 Kalman 滤波得到基于协方差的写入增益，并设计了可并行的等方差与对角线近似 KDN 与信息尺度调节；

**🔧 技术方法**

使用的技术包括线性高斯状态空间建模、Kalman 滤波、均值场变分推理、Möbius 递推扫描、信息尺度缩放及线性注意力架构；

**📊 数据集**

使用 FineWeb‑Edu 进行预训练，评估数据集包括 WikiText、LAMBADA、六项零样本推理任务（PIQA、HellaSwag、WinoGrande、ARC‑Easy/Challenge）、RULER 的单/多键检索任务以及真实世界检索任务（SWDE、SQuAD、FDA、TriviaQA、NQ、DROP）；

**📈 对比分析**

与 DeltaNet、Gated DeltaNet、KDA、Mamba‑3 及 GDN‑2 等基线在 750M/1.3B 参数规模下进行定量比较；KDN 在语言建模困惑度、零样本准确率和 RULER 分数上均超过所有基线；

**⚠️ 局限性**

局限性在于完整的 Kalman 过滤需要密集的协方差矩阵和 Riccati 更新，导致实现复杂；目前的等方差/对角近似仍在简化协方差，并未实现更丰富的转移动态（如旋转），未来仍有提升空间。

---

## 209. You Can't Prefer Emotions You Don't Sample: Intensity Undershoot in DPO-Tuned LLMs

**arXiv ID:** 2609.07808 | [PDF](https://arxiv.org/pdf/2609.07808v1)

**作者:** Hyunwoo Kim `[一作]` (Independent), Usama Khalid `[通讯]` (Hanyang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了指令调优的大语言模型在情绪强度控制中的欠驱动力（undershoot）现象，并提出通过均匀极端目标和更高温度、更大候选池来提升情绪强度的实现。

**💡 创新点**

创新点是引入“gain”指标量化情绪强度的实现程度，发现欠驱动力主要因候选池极端性不足，并通过简单的采样和目标分布修改显著提升 valence 控制，展示 arousal 控制的难点。

**🔧 技术方法**

使用了指令调优 LLM（Llama‑3.1‑8B 和 Qwen3‑8B），通过 LoRA 适配和 Direct Preference Optimization（DPO），利用冻结的 RoBERTa VA 回归器评估情绪。

**📊 数据集**

数据集使用 EmoBank 进行训练、验证和测试。

**📈 对比分析**

与自然目标采样的基线相比，均匀+高温度采样提升 valence gain 从 0.26 到 0.40（≈54%），并将 extrapolation MAE 降至 0.59；在 Qwen3‑8B 上同样提升，且对 in‑distribution VA 距离影响微小。

**⚠️ 局限性**

局限包括仅使用单一英语回归器和 EmoBank 语料，回归器对 arousal 的测量较弱导致不稳定；候选池极端性改进同时改变了目标分布、温度和大小，未能单独评估各因子；缺少人类主观验证。

---

## 210. What Does an LLM-Agent Leaderboard Rank Actually Compare?

**arXiv ID:** 2609.07785 | [PDF](https://arxiv.org/pdf/2609.07785v1)

**作者:** Wei-Jung Huang `[一作]` `[通讯]` (Independent Researcher), Wei-Jung Huang (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于估计量的LLM代理排行榜比较程序，并给出可解释的决策标签

**💡 创新点**

首次将估计量、支持检查、置信区间和可操作边界（符号翻转半径）整合到排行榜比较中，实现对排名可靠性的系统判定

**🔧 技术方法**

使用标准化、bootstrap置信区间、弱验证器校准、符号翻转半径等统计方法

**📊 数据集**

在SWE-bench、AgentRewardBench、tau2-bench、DataAgentBench、Open Agent等公开排行榜数据集上进行评估

**📈 对比分析**

该方法通过置信区间和实际差距判定稳定、未决、目标敏感等标签，揭示多数近似排名不可确定，能够说明标签源、目标和资源规则对结果的影响，性能体现在能系统解释不同排行榜差异的根本原因

**⚠️ 局限性**

仅适用于公开记录足够细粒度的排行榜，无法推断因果效应，受数据粒度、标签可用性和支持范围的限制

---

## 211. FrogNano: Training a 4B Coding Agent via Online Task Synthesis

**arXiv ID:** 2609.07925 | [PDF](https://arxiv.org/pdf/2609.07925v1)

**作者:** Minseon Kim `[一作]` (Microsoft Research Montréal), Alessandro Sordoni `[通讯]` (Microsoft Research Montréal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了一个仅含40亿参数的代码代理（Qwen3.5-4B）通过迭代强化学习，使其在仓库级软件工程任务上能与大模型竞争。

**💡 创新点**

创新点在于：①基于当前策略在线生成与其可学习边界对齐的合成任务；②为小模型设计轻量化的 Claude‑Code 样式工具接口；③使用 DPPO+异步 roll‑outs、日志长度惩罚和行为整合（consolidation）实现高效训练；④完全不依赖大模型蒸馏或大量真实任务。

**🔧 技术方法**

核心技术包括：基于策略反馈的任务合成 pipeline、DPPO（分布式 PPO）、异步 roll‑outs、log‑length 负奖励、简化工具调用 harness、行为整合与摘要压缩。

**📊 数据集**

使用 1,500 个基于 SWE‑Gym、SWE‑ReBench 等公开合成任务的数据集，并在每轮迭代中根据当前模型生成新的合成任务。

**📈 对比分析**

在 SWE‑bench Verified、SWE‑bench Pro、Terminal‑Bench 2.0、PatchEval‑Verified 四个基准上进行评估。迭代 5 次后在 Verified 上达到 61.5% solve rate，Pro 37.6%，Terminal‑Bench 31.1%，PatchEval 23.2%，与 32B–100B 级别模型相当或接近。

**⚠️ 局限性**

局限性包括：仅针对英文 Python 仓库；不适用于无测试或非 Python 项目；未验证对图像/视频输入的支持；可能产生安全或错误修复不完整的补丁；模型仍依赖测试覆盖率作为奖励，未全面评估代码质量。

---

## 212. Poisson Image Denoising Using Minimax Concave and Reweighted $\ell_1$ Penalties: Nonblind and Blind Approaches

**arXiv ID:** 2609.07916 | [PDF](https://arxiv.org/pdf/2609.07916v1)

**作者:** Reza Parvaz `[一作]` `[通讯]` (University of Mohaghegh Ardabili), Reza Parvaz (University of Mohaghegh Ardabili)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种基于MCP和重加权ℓ1正则化的Poisson图像去噪与盲去卷积模型。

**💡 创新点**

创新点是将Minimax Concave Penalty与分数阶梯度的重加权ℓ1结合，提升边缘保持并兼顾稀疏性。

**🔧 技术方法**

采用ADMM算法求解非凸优化，并给出收敛性分析。

**📊 数据集**

在公开图像（Girl、Shepp-Logan、月球、Hill、Satellite等）与USC-SIPI数据库的多种PSF下进行实验。

**📈 对比分析**

与BM3D、OGS、FOTV等方法对比，PSNR与MSSIM均优于或相近，证明性能更佳。

**⚠️ 局限性**

限制是参数敏感且需要手工调节，且对大尺寸图像计算量仍较高。

---

## 213. Conditional Timed Partial Orders: An Expressive and Interpretable Framework for Robot Task Specification and Planning

**arXiv ID:** 2609.07905 | [PDF](https://arxiv.org/pdf/2609.07905v1)

**作者:** Sebastian Escobar `[一作]` (University of Colorado), Morteza Lahijanian `[通讯]` (University of Colorado)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的任务规范框架——条件时序偏序（Conditional TPO, cTPO），并基于混合整数线性规划（MILP）实现了规划算法；

**💡 创新点**

1）在TPO基础上加入任意时钟差约束，支持无先后关系的时间约束；2）通过原子命题实现环境依赖的条件任务激活；3）设计了一种完整、最优性保留的子-TPO分解算法，将大型MILP拆分为若干小型MILP；

**🔧 技术方法**

图形化任务表示、时钟差约束建模、逻辑激活条件、GTSP‑TWPR约束、MILP求解（使用Gurobi）以及子-TPO分解与层次规划；

**📊 数据集**

使用合成的随机TPO/ cTPO实例（包含50‑60个事件、4‑13个原子命题）以及基于ROS的仓库巡检和火星车任务的离散状态机模型；

**📈 对比分析**

与单块MILP求解进行对比，实验显示在大多数实例中，分解方法可实现3–4个数量级的速度提升，且最终计划最优性保持不变；

**⚠️ 局限性**

分解依赖于任务的可分解结构；高层MILP的求解仍受命题公式复杂度影响；目前仅适用于单机器人任务，未考虑多机器人协作。

---

## 214. ComVLA: Communication-Aware Split Inference for VLA Models in 6G-Connected Robotics

**arXiv ID:** 2609.07838 | [PDF](https://arxiv.org/pdf/2609.07838v1)

**作者:** Boliang Liu `[一作]` (Technical University of Berlin), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ComVLA 框架，实现 VLA 模型的通信感知分布式推理：在边缘端通过语言指令对视觉 tokens 进行重要性评分，动态将传输的 token 数量对准无线链路容量，只发送任务关键 token，完成机器人操控任务。

**💡 创新点**

创新点在于将语言指导的 token 重要性与链路容量进行联合优化，构建信息瓶颈（IB）视角下的容量约束 token 选择；同时实现实时自适应的 token 预算（K*），解决现有 token-pruner 固定预算与 SemCom 需要通道重训练的缺陷。

**🔧 技术方法**

技术包括：跨模态交叉注意力计算 token 重要性（硬投票方式）、基于链路容量的 token 预算计算、量化压缩（INT8/INT4）以及对 6G 低延迟、频谱资源受限场景下的随机漫射和里辛衰落模型评估。

**📊 数据集**

使用 LIBERO 机器人操控基准数据集（Spatial、Object、Goal、Long Horizon 四个任务套），并在此基准上评估 OpenVLA-OFT 以及 LightVLA、VLA-ADP、DeepJSCC、WITT 等对比方法。

**📈 对比分析**

对比方法：与完整 512-token 的 OpenVLA-OFT、内容自适应 pruner（LightVLA、VLA-ADP）、SemCom 编码器（DeepJSCC、WITT）及随机 pruning。结果显示：在 K*=32（只传输 32 个 token）时，ComVLA 任务成功率 95.4%（比全量 96.9% 仅差 1.5pp），TX 数据量降低 16×，云端计算量降低 3.8×，推理延迟降低 22%。在 Rayleigh/Rician 衰落下亦保持稳健，CSI 延迟至 200ms 时性能衰减极小。

**⚠️ 局限性**

局限性包括：对动态链路容量的近似模型可能不完全适应真实 6G 物理层；硬投票方式对语义重要性估计的鲁棒性在极端低预算下可能受限；缺乏真实无线环境验证与更复杂任务的跨域泛化评估。

---

## 215. SAFIRE: Safety-Critical Benchmark for Fine-grained Fire and Smoke Understanding in Multimodal LLMs

**arXiv ID:** 2609.07823 | [PDF](https://arxiv.org/pdf/2609.07823v1)

**作者:** Pengfei Li `[一作]`, Muzammal Naseer `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SAFIRE大规模火灾与烟雾多模态评测基准，包含83K图像与193K多选VQA问答，支持场景级、上下文、因果、行为等多维度推理评估；

**💡 创新点**

创新点在于：①构建多场景、多维度、上下文感知的评测数据集；②采用GPT‑5.4辅助验证与多模型多数投票的高质量注释流程；③同时提供零/少样本视觉‑语言编码器评测，验证轻量化微调效果；

**🔧 技术方法**

技术手段包括多源数据收集与过滤、结构化Prompt生成、GPT‑5.4语义验证、MLLM多数投票、零样本CLIP推理与SVM线性探针、少量样本微调；

**📊 数据集**

使用的主要数据集是SAFIRE（83K图像、9.7K子集生成193K VQA），并对比已有火灾/烟雾数据集（如BowFire、EdgeFireSmoke、SmokeBench等）；

**📈 对比分析**

与十款开源MLLM（8B–38B）以及多种CLIP编码器对比，平均VQA准确率仅61.9%，最高约93%，零样本分类最高72.2%；少量样本微调可将分类准确率从20.1%提升至64.5%；

**⚠️ 局限性**

局限包括：①评测仅采用有限的文本提示，未覆盖更丰富的语言变体；②模型多为通用预训练，仅做线性探针或单一微调，未探索更深层次迁移；③注释链条依赖生成模型与GPT验证，可能保留生成偏差，需进一步人类审核与社区验证。

---

## 216. The Accuracy Paradox: Empirical Diagnostic of Default Decision Thresholds in Multi-Label Enzyme Commission Prediction [With Code]

**arXiv ID:** 2609.07897 | [PDF](https://arxiv.org/pdf/2609.07897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 217. Grid Trouble in Paradise: Uncovering Vulnerable Distributed Energy Resources and Their Grid-Level Risks

**arXiv ID:** 2609.07783 | [PDF](https://arxiv.org/pdf/2609.07783v1)

**作者:** Anna Raymaker `[一作]` (Georgia Institute of Technology), Raheem Beyah `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过互联网扫描与电网建模，系统性评估了全球范围内已暴露的光伏分布式能源资源（DER）安全风险，并量化了攻击对Oahu电网的潜在影响。

**💡 创新点**

创新点包括：① 构建跨厂商的 DER 发现管道，首次实现 66,379 台互联网可见 DER 的精准识别；② 将真实暴露的 DER 与功率系统分析相结合，提供基于实测设备的电网攻击上界；③ 开源公开的交互式工具 gridtrouble.xyz，便于社区实验与可视化。

**🔧 技术方法**

主要技术手段包括：Censys 与 Shodan 的大规模互联网扫描；LLM（Llama3.3:70B）进行初步标签生成；随机森林分类器实现快速全网识别；文本 TF‑IDF、协议指纹等特征工程；AC OPF 优化模型模拟攻击。

**📊 数据集**

数据集主要来源于两次 Censys 全网扫描（分别覆盖 34 与 96 家 DER 厂商，合计 1.8M 主机），以及 Oahu 电网的 37 节点传输模型和手工验证的 571 台 DER 位置信息。

**📈 对比分析**

与以往基于 PLC、EV 等设备的安全评估相比，本文在检测率和精度上取得 98%+ 精确率、86%+ 召回率；在电网影响评估中发现攻击可导致 18/37 节点低压、6/89 区线超载，显示在相同规模攻击下 DER 供电侧风险远高于负载侧。

**⚠️ 局限性**

局限性包括：① 仅检测可从公网访问的 DER，未覆盖使用 VPN/私网的设备；② 依赖 IP 地理定位，可能出现位置误差；③ CVE 匹配基于公开版本信息，未证明可利用；④ 假设攻击者能同时控制所有可暴露设备，忽略网络分段和防御；⑤ 对动态事件（频率失稳、保护响应）的建模不足。

---

## 218. Interactive Debugger for Performance Portable Python HPC Kernels

**arXiv ID:** 2609.07912 | [PDF](https://arxiv.org/pdf/2609.07912v1)

**作者:** Ivan Grigorik `[一作]`, Milos Gligoric `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了PKDB，一个支持GPU和多线程低层内核的交互式调试器，允许在不修改源代码的情况下在设备上设置断点、步进、变量检查，并引入了在暂停时执行任意表达式/内核（live eval）和动态更新正在运行内核（hotswap）的功能。

**💡 创新点**

①在Python EDSL环境中首次实现对GPU/多线程内核的实时调试；②提供在暂停时直接在设备上执行任意代码的live eval；③支持hotswap动态替换已调用的内核，避免完整程序重启。

**🔧 技术方法**

结合pdb接口、跨进程控制器与目标调试器（如GDB/LLDB）的PTY通信、IPC内存句柄传递数组、Python eval动态编译、call‑site interposition、并行内核调用等技术实现。

**📊 数据集**

评估使用了ExaMiniMD（分子动力学）、Boltzmann‑kinetics solver（粒子-细胞动力学）和Periodic Ewald sum（电势求和）等三类高性能计算工作负载。

**📈 对比分析**

与传统的Debug模式（逐步Python执行）相比，PKDB在CUDA、HIP和OpenMP后端的调试开销平均<2倍；hotswap相较重新启动节省数十毫秒；整体在大规模GPU调试场景中显著提升性能，保持可接受的调试延迟。

**⚠️ 局限性**

仍牺牲一定运行性能（调试构建比发布版慢）；仅支持Python EDSL，扩展到其他框架需额外工程；缺乏逆向调试等高级功能，对多线程继续命令支持有限。

---

## 219. PRIMUS: Identity, Governance, and Verification for Multi-Agent Federations

**arXiv ID:** 2609.07910 | [PDF](https://arxiv.org/pdf/2609.07910v1)

**作者:** Sasank Annapureddy `[一作]`, Anjaneya Prasad Thamatani `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出 PRIMUS 框架，用于在多智能体联邦中实现身份、合规性与治理决策的安全性，并探索将验证器输出转化为搜索的梯度信号。

**💡 创新点**

创新点包括：① 用质数幂身份与 BLS 聚合签名（PIAC）实现可验证的参与记录；② 推导安全杀阈值定理大幅降低误杀率；③ 给出单一治理与拜占庭容错治理的闭式交叉边界；④ 引入 VRF 随机选举、租约与 epoch 叠防护以保证安全与活跃性；⑤ 将二值验证结果转化为可度量的 fitness 信号用于生成‑测试循环。

**🔧 技术方法**

采用质因数分解、BLS12‑381 聚合签名、指数加权移动平均（EWMA）偏差评分、VRF 排序、租约与 epoch 叠防护，以及理论证明与仿真验证。

**📊 数据集**

使用自定义二进制覆盖码（(12,3) 与 (13,3)）实例、人工注入故障、LLM 生成候选集以及合成 oracle 进行实验。

**📈 对比分析**

通过与随机、二值门控、oracle 基准等方案比较，验证器实现 0% 误杀、杀阈值可在 10% 信道噪声下保持 0% 误杀；分区映射与理论值相差 <1.2 倍；梯度预筛选将 oracle 调用减少约 30%，但在覆盖码大小上未超越 oracle；验证成本约 0.15 美元/候选。

**⚠️ 局限性**

局限性包括：需要可信中心化注册表、对攻击稀释的前置假设、BLS 不是后量子安全、令牌规模随集群数增长不线性、无法修复自我膨胀、搜索实验仅在小规模、对称同步假设、未涵盖隐写式共谋、以及在无精确 oracle 的真实场景下缺乏验证。

---

## 220. Do Large Language Models Know What They Don't Know II? A Fully Behavioral, Non-Cognitive Measure of Epistemic Honesty

**arXiv ID:** 2609.07879 | [PDF](https://arxiv.org/pdf/2609.07879v1)

**作者:** Ali Şenol `[一作]` (Tarsus University), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了一种新的评估指标——认知诚实度量（Epistemic Honesty Quotient，EHQ），用于衡量大语言模型在知识边界上的自我认知与诚实表现，并构建了覆盖四类边界问答的 3,000 题 EHQ‑3000 数据集。

**💡 创新点**

创新点包括：① 将认知诚实拆解为可观测的三个子分数（回答克制、虚假陈述抵制、答案置信校准）并给出可复现的加权组合；② 设计了专门触发边界行为的四类题目（FEQ、PCQ、HNQ、CCQ）并对其进行版本化、审核和公开；③ 提供完整的实验框架与哈希绑定的结果，保证评测的可复现性与透明度。

**🔧 技术方法**

采用的方法主要是：① 通过规则式分类器将模型输出分为 ABSTAIN、HEDGE、CONFIDENT_CORRECT、CONFIDENT_WRONG 四类；② 在回答后引入 0–100 置信度提示并归一化；③ 计算三项子分数并按预设权重 0.30/0.45/0.25 组合成 EHQ；④ 对 14 台冻结的 LLM API 进行统计推断（自助法、置换检验、 Holm 校正）。

**📊 数据集**

使用的数据集为 EHQ‑3000，包括 3,000 个英文问题，均衡分布在四个子类别中（FEQ、PCQ、HNQ、CCQ），每类 750 题，且对每个问题都附有源材料、金标准答案或可验证的不可知判定。

**📈 对比分析**

在 14 台冻结 LLM 上的评测表明：EHQ 总分在 0.31–0.81 之间差异巨大，显著高于单纯的文档检索能力下的 99% 正确率；新版本模型整体得分略高于旧版本，但并未达到统计显著性；EHQ1 与 EHQ2 极度相关，EHQ3 与两者无显著相关，表明回答克制与置信校准是相对独立的行为维度；在各子类别上，FEQ/PCQ/HNQ 的表现高度相关，而 CCQ 则呈弱相关，提示知识边界与上下文边界的能力不完全重叠。

**⚠️ 局限性**

局限性包括：① 仅评估 14 台模型，样本量有限；② 分类器在 HEDGE 识别上存在误差，可能影响子分数；③ 置信度提示在不同模型间尺度不一致，导致 EHQ3 仅适用于主观答案；④ CCQ 只有匹配恢复实验作为对照，FEQ/PCQ/HNQ 缺乏对应的可回答基准；⑤ 受限于单轮英文提示，无法推广到多轮或多语言场景；⑥ 供应商截断与非推理条件验证依赖后端计数，易导致评分不确定。

---

## 221. InfluenceField: A Differentiable Field with Interventionally Identifiable Causal Structure for Multimodal World Modeling

**arXiv ID:** 2609.07874 | [PDF](https://arxiv.org/pdf/2609.07874v1)

**作者:** Zihao Yang `[一作]` (University of Oxford), Zhiqiu Huang `[通讯]` (University of Nottingham Ningbo China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 InfluenceField，一种插入在视觉编码器与语言解码器之间的可微连续空间场，用于捕捉局部干预的传播效果并提升多模态大语言模型的因果推理与可解释性。

**💡 创新点**

创新点在于：①将视觉补丁特征映射为连续可查询的空间场；②设计了基于注意力的有向影响矩阵和多步传播操作，能够模拟干预在空间上的长程效应；③引入了针对干预、跨环境不变性与结构正则化的联合训练目标，理论上证明在非线性有限基模型下可识别完整依赖图，在线性专门化下给出覆盖与损失稳定性保证。

**🔧 技术方法**

使用技术包括：高斯插值构造空间场；多头注意力与稀疏门控的有向影响矩阵；迭代传播方程与可学习的混合系数；干预掩膜与共享转移算子；语言建模、跨环境不变性损失、对比式干预监督和结构正则化等联合损失。

**📊 数据集**

采用的数据集主要有 CLEVRER、Causal3DIdent、CITRIS 用于预训练；CausalVQA、NExT-QA 用于下游因果视频问答；MAG 与 Lung Cancer 用于多模态因果结构发现；以及用于 OOD 评估的四种 CausalVQA 子集。

**📈 对比分析**

通过与 GPT‑4o、Gemini 2.5 Flash、InternVL2.5、LLaVA‑OneVision、Perception‑LM、Qwen2.5‑VL 等现有大模型和专门的因果基线进行比较，InfluenceField 在 CausalVQA 上整体准确率提升至 67.4 %，比基线提升 13.1 个百分点；在规划与假设类问答中获得最大增益；在 NExT‑QA、OOS 泛化、事实‑反事实一致性等指标上也表现出显著优势。

**⚠️ 局限性**

局限性包括：①理论证明基于完备的全样本一致性与单步分离假设，缺乏有限样本的非线性可识别性保证；②线性专门化的覆盖与损失稳定性仅在特定的混合模型下成立；③对连续空间场的全局传播复杂度为 O(N²)，在大规模场景下可能成为瓶颈；④干预被近似为高斯掩膜，未完全再现真实物理干预的细节。

---

## 222. Parallelizing the Factorial Space: 3x SIMD Acceleration of the Steinhaus-Johnson-Trotter Algorithm via Dual-Lane AVX2 Execution

**arXiv ID:** 2609.07862 | [PDF](https://arxiv.org/pdf/2609.07862v1)

**作者:** Serge Melnikov `[一作]` `[通讯]` (Independent Researcher), Serge Melnikov (Independent Researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用AVX2 SIMD指令的双路向量化实现，迭代生成Steinhaus‑Johnson‑Trotter排列。

**💡 创新点**

创新点在于将排列空间分区与单周期字节洗牌相结合，实现在单个256位寄存器内同时生成两条排列流，并通过半空间反射对称性将生成工作量减半。

**🔧 技术方法**

使用了AVX2指令集（YMM寄存器、_mm256_shuffle_epi8等）、循环无分支优化、零基索引、双路并行和半空间镜像重构等技术。

**📊 数据集**

使用的“数据集”是所有 0…n‑1 的排列，实验验证覆盖 n≤13（核心实现 n≤16）并测量CPU周期数。

**📈 对比分析**

与 Knuth 的 Algorithm P、Hu 的 Ring‑Cascade 算法以及 Heap’s 算法对比，单循环向量实现实现约 3× 的吞吐量提升（原子级别 6×，应用层 3×），并在 Intel i7‑8850H 4.2 GHz 机器上每个排列约 1.27 B 周期。

**⚠️ 局限性**

局限性包括 n≤16（受 256 bit 寄存器宽度限制）、仅生成排列空间的前一半需要手动镜像、对 AVX2 依赖且未充分利用 AVX‑512 或多线程并行。

---

## 223. Scene Graph-Driven Haptic Feedback for Safety Enhancement in Robotic Ophthalmic Surgery via Physically Simulated iOCT

**arXiv ID:** 2609.07857 | [PDF](https://arxiv.org/pdf/2609.07857v1)

**作者:** Danial Arbabi `[一作]` (Technical University of Munich), M. Ali Nasseri `[通讯]` (University of Alberta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究开发了一种基于场景图的子视网膜注射机器人辅助系统，并通过实时视觉感知提供触觉反馈以引导操作。

**💡 创新点**

创新点在于将场景图作为语义中间件实现可模块化的上下文感知反馈，并利用负强化的触觉信号实现软型虚拟约束。

**🔧 技术方法**

使用了图像感知管线、规则推理状态检测、场景图建模、触觉反馈装置及遥操作机械臂技术。

**📊 数据集**

数据集主要来自实验中使用的人体眼球仿真模型、视觉传感器采集的图像以及16名受试者的操作记录。

**📈 对比分析**

采用配对Wilcoxon符号秩检验比较视觉单一与视觉+触觉两种条件，结果显示对齐角度显著下降（7.9°→6.8°, p=0.044）且可用性评分提升（SUS 75.9→81.9, p=0.015），完成时间无显著差异。

**⚠️ 局限性**

局限性包括样本量仅16人、未考虑真实组织变形及成像噪声，缺乏对临床OCT数据的验证。

---

## 224. Foundation Models for Generalizable Semantic and Goal-Oriented Communication

**arXiv ID:** 2609.07853 | [PDF](https://arxiv.org/pdf/2609.07853v1)

**作者:** Boliang Liu `[一作]` (Technical University of Berlin), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于基础模型的FMSGOC框架，采用VLM进行稀疏语义锚点选择，LoRA调优的扩散模型实现稀疏锚点的无缝重建；

**💡 创新点**

创新点在于将语义选择与生成重建解耦，利用大规模预训练基础模型的开放世界知识实现零样本泛化，并通过硬重注射防止语义漂移；

**🔧 技术方法**

使用CLIP作为目标驱动的语义提取器，Stable Cascade VAE+U‑Net作为生成基底，LoRA轻量微调，LPIPS和CLIP余弦相似度评估；

**📊 数据集**

在CIFAR‑10上训练，评估于CIFAR‑10与ImageNet两大数据集；

**📈 对比分析**

与JPEG、DeepJSCC、WITT等基线比较，FMSGOC在0.039 BPP下达到0.87–0.90的语义相似度，ImageNet上保持0.83–0.86，并在LPIPS指标上取得0.1278/0.1558的优异表现；

**⚠️ 局限性**

局限在于目前仅在AWGN通道下验证，未考虑衰落、UEP以及不同目标提示的鲁棒性。

---

## 225. A*-Thought-V2: Efficient Latent Reasoning via Geometric Dynamics of LLM

**arXiv ID:** 2609.07821 | [PDF](https://arxiv.org/pdf/2609.07821v1)

**作者:** Xiaoang Xu `[一作]` (Beijing University of Posts and Telecommunications), Zhaofeng He `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 A*-Thought-V2 框架，将链式思考（CoT）压缩为显式文本与隐式潜在表示交错的序列，并通过三维 PCA 投影的方向角度判定哪些步骤保留为文本，哪些步骤压缩为潜在向量。

**💡 创新点**

创新点在于：①用三维 PCA 计算 CoT 轨迹的全局与局部方向角度，并将角度分段与语义对应；②设计嵌入强制（Embedding Forcing）将冗余文本步骤映射为单个连续潜在向量；③设计标签强制（Label Forcing）用软词表分布监督潜在位置，从而实现信息保持与压缩兼顾；④将显式–隐式交错结构与 A* 搜索相结合，实现动态、可变长的压缩。

**🔧 技术方法**

采用的技术包括：三维 PCA 投影与方向角度分析；显式–隐式潜在交错架构；嵌入强制与标签强制训练策略；SFT 细调；使用 Qwen3.5‑9B 与 Qwen3.6‑27B 语言模型；通过 ACU（Accuracy per Computation Unit）衡量性能。

**📊 数据集**

训练使用 OpenR1‑Math‑3k 数据集；评估使用六个基准：Math500、AIME 2024、AIME 2025、AIME 2026、ARC‑Challenge、GPQA‑Diamond。

**📈 对比分析**

与 SwiReasoning、CopT、A*-Thought 等基线对比，A*-Thought-V2 在两种模型规模上平均准确率提升 2.6%（最多 90° 变体），ACU 提升 2.29×，回复长度减半；压缩与训练时间分别下降 94.6% 与 80.3%。

**⚠️ 局限性**

局限性在于：①角度阈值需手工设定，可能不适用于所有推理任务；②潜在表示的可解释性有限；③在极端复杂、多分支推理场景下仍可能丢失关键信息；④对模型规模与任务分布的泛化能力需要进一步验证。

---

## 226. VoT: Vision-of-Thought for Unified Multimodal Representation Alignment

**arXiv ID:** 2609.07815 | [PDF](https://arxiv.org/pdf/2609.07815v1)

**作者:** Jingxiang Sun `[一作]` (ByteDance Seed), Weilin Huang `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 Vision‑of‑Thought（VoT）框架，在视觉语言模型（VLM）与扩散解码器之间插入可离散化的视觉思维层，实现从文本到视觉计划再到图像的三阶段生成流程。

**💡 创新点**

创新点在于：①将视觉 token 与 VLM 语义空间对齐，采用闭环目标（VLM 对齐 + 特征重建 + VQ 损失）训练 SimVQ tokenizer；②在统一的 Mixture‑of‑Transformers 中引入专门的 VoT 分支，既保持 VLM 对理解的专长，又允许自回归的视觉计划；③通过两阶段训练（VoT 预训练 + VoT‑DiT 联合训练）实现高效的端到端优化。

**🔧 技术方法**

使用技术包括：VLM 对齐损失、特征重建损失、SimVQ 量化、Qwen2.5‑VL ViT 作为教师、Mixture‑of‑Transformers（MoT）三分支架构、两阶段训练流程、结构化条件丢弃与注意力掩码、扩散速度预测等。

**📊 数据集**

主要使用图文对齐数据集（图像分辨率 434×434–1024×1024），以及公开的 GenEval 测试集、内部 DreamBench 等评测集。

**📈 对比分析**

与 SOTA 文本到图像模型（如 FLUX.1‑dev、Mogao、BAGEL、SD3 等）在 GenEval 上对比，VoT 在整体得分 0.91、计数任务 0.86 等指标均优于对手；内部评测 AutoEval 亦提升至 70%+。

**⚠️ 局限性**

局限性包括：离散 token 长度和覆盖范围受限，细节层面的表征可能不够精细；训练时需要冻结 VLM，算力与成本仍较高；对复杂场景的细粒度控制仍有限。

---

## 227. Understanding the Impact of Model Pruning on Long-Tail Forgetting and Explanation Reliability in Medical Imaging

**arXiv ID:** 2609.07803 | [PDF](https://arxiv.org/pdf/2609.07803v1)

**作者:** Nazish Khalid `[一作]` (Missouri University of Science and Technology), Mohammad Yaqub `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文开展了一项系统性研究，探讨在医学影像任务中模型剪枝对长尾遗忘与可解释性可靠性的影响，覆盖多种网络结构、剪枝策略及稀疏率；

**💡 创新点**

创新点包括首次在医学影像领域将长尾遗忘与解释可靠性联合评估；对四种剪枝方法进行横向对比；揭示解释质量下降主要源于梯度消失而非特征激活消失；提出在中度稀疏水平下实现压缩、预测性能与解释可靠性平衡的实践建议；

**🔧 技术方法**

使用了无结构剪枝技术（L1幅度剪枝、SNIP、GraSP、随机剪枝）；采用 Grad‑CAM 归因方法；通过 Stability IoU 量化解释稳定性；通过 AOPC 衡量解释可信度；配合 mAP、每类 AP、相关性分析等统计手段；

**📊 数据集**

实验数据集为 NIH‑CXR‑LT（胸部 X‑ray）和 ISIC‑2019（皮肤病）两大长尾医学影像数据集；

**📈 对比分析**

实验通过对比 mAP、各类 AP、解释稳定性与可信度在不同稀疏率下的表现来评估剪枝方法；结果显示 SNIP 在大多数稀疏率下保持最高或相近预测性能，L1 在高稀疏率出现崩塌，随机剪枝最差；长尾遗忘随类频率显著，低频类在较低稀疏率下即出现显著性能下降；解释稳定性与可信度受剪枝策略影响明显，梯度信息保持的剪枝方法（SNIP、GraSP）表现更稳健；

**⚠️ 局限性**

研究局限主要在于仅针对 CNN 的无结构剪枝，未涉及结构剪枝、量化、知识蒸馏等压缩手段；仅验证了两大医学影像数据集，未探讨其他任务（检测、分割）或 Transformer 等新型网络的适用性；

---

## 228. Climate-ModernBERT: Revisiting Corpus Composition for Domain-Adaptive Continued Pretraining

**arXiv ID:** 2609.07798 | [PDF](https://arxiv.org/pdf/2609.07798v1)

**作者:** Yongan Yu `[一作]` (McGill University), Markus Leippold `[通讯]` (University Of Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Climate-ModernBERT，一系列基于ModernBERT-Base通过在学术气候文本、过滤后网路数据和合成气候文本上继续预训练得到的气候适配编码器。

**💡 创新点**

创新点在于：①系统评估不同气候语料组合对域适配的影响；②提出通过参数空间合并（简单加权平均等）对单源预训练模型进行融合，比联合多源继续预训练更优；③揭示学术语料提供最强适配信号，合成文本在任务上表现具有显著任务依赖性。

**🔧 技术方法**

技术方法包括ModernBERT的两阶段训练（上下文扩展CX+学习率衰减LRD）、对齐三种语料的预处理与去重、对参数空间的合并（加权平均、Task Arithmetic、TIES‑Merging、DARE‑TIES）以及在九个气候NLP基准上的微调与评估。

**📊 数据集**

使用的语料共6.42B tokens，来源为：①学术气候文本（约1.28B token）——包括期刊、IPCC报告、arXiv预印本等；②过滤后的FineWeb‑Edu网络数据（约5B token）；③合成气候文本（约0.14B token）由GPT生成。

**📈 对比分析**

在21个适配模型和9个基准上进行比较。与基线ModernBERT‑Base相比，最佳模型平均F1提升2.8点；单源模型最高平均F1为86.4；参数合并（simple weight averaging）得到的平均F1 76.3，超过单源模型及联合训练。

**⚠️ 局限性**

局限性包括：①评估集中在句子/段落级分类，缺乏长文档理解与跨段推理的基准；②仅使用英文语料和单一编码器家族；③未探索指令微调、跨领域迁移的效果；④合成文本的生成质量和代表性可能影响结论。

---

## 229. Quantifying the Engagement Trap: Impact of Short-form Video Recommender Systems on Users with ADHD

**arXiv ID:** 2609.07795 | [PDF](https://arxiv.org/pdf/2609.07795v1)

**作者:** Vedad Misirlic `[一作]` (Graz University of Technology), Elisabeth Lex `[通讯]` (Graz University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对302名用户进行问卷调查，量化短视频推荐系统在注意缺陷多动障碍（ADHD）群体中的时间盲目、放弃难度、后使用后悔和情绪冲击等体验。

**💡 创新点**

提出并验证了“Engagement Trap”模型，阐释了推荐系统通过高效个性化内容导致ADHD用户在自我调节上受限而产生的系统性不平等。

**🔧 技术方法**

使用结构化问卷（7点李克特量表）收集数据，并采用Kruskal‑Wallis、Dunn后验、Mann‑Whitney U检验等非参数统计方法进行分组比较。

**📊 数据集**

数据来源于Prolific平台招募的自报或医诊断ADHD与非ADHD用户，共计302人；未使用外部公开数据集。

**📈 对比分析**

通过对三组（无ADHD、ADHD自报、ADHD医诊）以及两组（ADHD与非ADHD）进行统计检验，发现ADHD组在时间盲目、放弃难度、后使用后悔和情绪负担方面均显著高于对照组（p<0.001，效应量中等），表明推荐系统优化对ADHD群体产生不利影响。

**⚠️ 局限性**

局限性包括：依赖自报ADHD状态，未对诊断或药物使用进行验证；使用回顾性主观问卷，缺乏客观行为数据；未实现或测试所提议的可访问设计功能；可能存在样本自选择偏差。

---

## 230. Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference

**arXiv ID:** 2609.07786 | [PDF](https://arxiv.org/pdf/2609.07786v1)

**作者:** Zheyuan Wang `[一作]` (Beijing Normal University), Qian Liu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Signed Rescue Routing的LLM级联路由方法，先用小模型处理所有请求，再基于小模型输出特征预测是否升级到大模型。

**💡 创新点**

创新点在于预测“救援”与“危害”两事件的概率差（即增量价值），而非单纯使用不确定性或错误预测，从而在固定升级预算下实现贝叶斯最优路由。

**🔧 技术方法**

使用两头轻量级分类器（一个预测救援概率，一个预测危害概率）以及小模型的概率、熵、边际等特征来计算差值，并按阈值或top‑k进行升级决策。

**📊 数据集**

在公开的四选多选任务上进行评估，数据集包括MMLU、HellaSwag和ARC‑Challenge（共约5,165个样本）。

**📈 对比分析**

与随机、置信度、熵、错误预测以及仅救援等基线比较，Signed Rescue Routing在AUACC、25%与50%预算下的准确率分别提升至≈69.4%，并且相对于单大模型只需约1.78×的计算成本。

**⚠️ 局限性**

局限性包括：仅在同一模型族的多选任务上验证，缺少自由文本生成场景；假设升级成本均匀，未处理成本不均的情况；在数据分布漂移时需重新校准。

---

## 231. A 28nm 27,648-Spin Multichip Digital Ising Accelerator with Pegasus Connectivity

**arXiv ID:** 2609.07907 | [PDF](https://arxiv.org/pdf/2609.07907v1)

**作者:** Tong Wu `[一作]` (Carnegie Mellon University), Tathagata Srimani `[通讯]` (Carnegie Mellon University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一款28nm工艺、四颗芯片组成、总计27,648个自旋的数字Ising加速器，能够实现高密度Pegasus拓扑、10位系数精度，并通过时间复用的PE阵列和预排的芯片间通信完成并行自旋更新。

**💡 创新点**

创新点包括：①采用度为15的Pegasus连接拓扑，显著提升嵌入效率；②在每颗芯片上使用局部SRAM一次性读取所有系数，避免了多路地址与乘法器的开销；③引入时间复用的4色更新调度，允许在不同颜色间重叠计算与通信；④设计固定长度边界状态包，消除压缩/解压复杂度，保证在多芯片系统中实现高吞吐量。

**🔧 技术方法**

技术实现：时间复用PE阵列（216个并行更新）、本地10b系数SRAM、六个PE的单口bank结构、加权求和树与tanh查找表、LFSR采样、源同步I/O + FIFO后压缓冲、固定边界状态包的序列化/反序列化、可编程的系数和偏置加载扫描接口。

**📊 数据集**

使用的基准数据集：植入MaxCut（27,069节点）、自旋玻璃、受挤回路、半素数分解以及3SAT（200–1,200条子句）等多种组合优化与概率逻辑任务。

**📈 对比分析**

与现有数字CMOS Ising实现（如9芯片King’s图5b系数）进行对比，提升了原始度从8到15、系数精度从5b提升至10b，保持O(N)存储。四芯片总更新率达到41.5 G更新/秒，能耗为1.2 pJ/更新（49.8 mW）。在27,069节点的植入MaxCut实例中，达成最优解仅需3.3 µs；3SAT任务在1.3 s以内完成99%成功概率的TTS。

**⚠️ 局限性**

局限性：仍需多颗芯片协同工作，芯片面积与功耗受限于28nm工艺；Pegasus虽提升了嵌入效率，但在极稀疏或完全连接问题上仍需额外物理自旋；固定长度边界传输在高活动率下会产生不必要的开销；在更大规模或更高精度需求下，系统的可扩展性与存储成本仍是挑战。

---

## 232. Prevalence calibration as shortcut mitigation

**arXiv ID:** 2609.07922 | [PDF](https://arxiv.org/pdf/2609.07922v1)

**作者:** Mohamed Amine Kina `[一作]` (University of Bremen), Eike Petersen `[通讯]` (Fraunhofer Institute for Digital Medicine MEVIS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

将shortcut学习重新表述为校准问题，并提出两种encoder‑agnostic校准方法（内在正则化CSM/DCSM与后处理校准），解决冻结基础模型的shortcut依赖；

**💡 创新点**

创新点在于将shortcut学习视为预valence均衡的校准任务，提供不依赖表示学习、可直接用于冻结encoder的两种校准方案，并通过贝叶斯优化实现超参调优；

**🔧 技术方法**

使用交叉熵+组条件平均得分正则化、Beta校准、Gaussian Process Bayesian优化、以及标准的DenseNet与基础模型（MedSigLIP、MedImageInsight）作为编码器；

**📊 数据集**

实验基于CheXpert和SIIM‑ACR肺门诊图像数据集，利用胸腔引流与肺气肿的标注构造shortcut对齐与对齐的测试集；

**📈 对比分析**

与ERM、CDAN、CMMD、JTT、DFR等基线在aligned/misaligned AUROC与fairness score上比较，结果显示内在正则化和后处理校准显著提升性能，尤其是后处理校准将误对齐组AUROC从0.23提升至0.73；

**⚠️ 局限性**

局限性包括需提前获得组标签用于后处理校准，对不同shortcut的阈值选择不统一，且在某些情况下后处理校准对内在正则化方法的效果略有下降，未对多卡特异性shortcut进行更广泛验证。

---

## 233. Automatic constraints with few subpowers and graphoid recognition

**arXiv ID:** 2609.07891 | [PDF](https://arxiv.org/pdf/2609.07891v1)

**作者:** Antonios Kalampakas `[一作]` `[通讯]` (American University of the Middle East), Antonios Kalampakas (American University of the Middle East)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文探讨了有限自动机在约束满足问题中的应用，证明了在特定条件下，约束满足问题可以在多项式时间内解决，并提出了一种算法来计算完整的解决关系及其投影的紧凑表示。

**💡 创新点**

创新点在于提出了一种新的算法，该算法利用非确定性有限自动机直接提供所需的见证，解决了自动约束满足问题的可解性，并为特定的三边代数提供了规范形式的描述。

**🔧 技术方法**

使用了非确定性有限自动机（NFA）和多项式时间编译技术，将NFA编译为所需的见证和小投影，以支持少子代数算法。

**📊 数据集**

使用了与有限自动机相关的约束实例数据集，具体包括多个约束和变量集，涉及的约束关系由NFA描述。

**📈 对比分析**

与现有方法相比，提出的算法在处理具有无限元数和重复变量的约束时表现出色，能够在多项式时间内计算完整的解决关系，并允许有序投影，性能优于传统方法。

**⚠️ 局限性**

限制在于该算法的有效性依赖于特定的边操作和有限域的性质，且在某些情况下，识别问题的复杂性可能仍然存在。

---

## 234. Are Image Generators Zero-Shot Perceivers? A Rigorous Evaluation

**arXiv ID:** 2609.07884 | [PDF](https://arxiv.org/pdf/2609.07884v1)

**作者:** Shangzhe Di `[一作]` (Shanghai Jiao Tong University), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估通用图像生成模型在零任务（zero‑shot）条件下的视觉感知能力，并提出统一的ProbeGen基准与评测流程。

**💡 创新点**

创新点在于将生成任务转化为文本提示驱动的条件生成，统一衡量深度估计、分割与计数三种感知任务，并揭示生成模型在域外（OOD）和组合推理任务中的优势。

**🔧 技术方法**

使用文本提示驱动的生成器、确定性提取器（Otsu阈值、仿射对齐、红点计数）、对比多种生成器、专家模型与多模态LLM，构建统一的评测协议。

**📊 数据集**

使用了11个公开基准，包括NYUv2、DIODE、ScanNet、KITTI、RefCOCO、ReasonSeg、Manga109、DRAM、PixMo‑Points等。

**📈 对比分析**

在统一的Prompt与提取器协议下，对20个模型进行对比，发现专家模型在分布内精度和效率上更强，而生成模型在域外和组合推理任务中更具竞争力，但整体速度和资源占用较高。

**⚠️ 局限性**

主要局限包括：缺乏像素级对齐导致输入失真、输出格式不规范（如不保持原图结构）、对复杂任务（如表面法线）零射表现不佳，以及显著的速度与显存消耗问题。

---

## 235. EventSpec: Defining and Detecting Event-Semantic Issues in Blockchain Ecosystems

**arXiv ID:** 2609.07865 | [PDF](https://arxiv.org/pdf/2609.07865v1)

**作者:** Yixuan Liu `[一作]` (Nanyang Technological University), Yi Li `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并实现了 EventSpec，一种基于语料库学习的事件语义缺陷检测框架，定义了事件语义缺陷的五类（事件冲突、状态-事件不匹配、未授权事件发射、事件发射不匹配、事件参数不匹配）及其对应的两类 off‑chain 攻击向量，并通过差分检查对真实合约进行检测，辅以 off‑chain harness 演示攻击可行性；

**💡 创新点**

创新点包括：①系统性构建跨领域事件语义缺陷分类与攻击向量；②利用合约语料库进行事件规范的经验学习，结合字节码分析与语义约束抽取；③差分检查机制提供可解释的发现；④首次在真实链与 off‑chain 系统上重现并确认事件冲突和事件‑状态不同步攻击，报告实际漏洞；⑤在 331k 合约中实现 90.17% 的综合精度，显著优于现有通用安全分析工具；

**🔧 技术方法**

技术实现包括：EVM 字节码反编译（Gigahorse）、三地址码、CFG/ICFG、日志站点反向切片与污点分析、语义约束抽取、事件/函数聚类与统计模式匹配、概率与置信门限、SMT 符号等价检查；off‑chain harness 用 Solidity 编写，配合 Python 实现核心模块；

**📊 数据集**

使用的数据集为：①Smart Contract Sanctuary 中 331,382 个验证合约（其中 212,072 用作语料库，95 个被人工标注为基准）；②ERC 标准事件集（14 个 ERC，26 个事件）用于验证；③通过 FORGE、SlowMist、BlockSec 等渠道收集的 1,020 条安全审计报告与 11 起实际事件；④针对桥接、钱包、区块浏览器、NFT 市场等 off‑chain 消费者的实际系统进行攻击实验；

**📈 对比分析**

实验对比基线为 Slither，仅在两类事件上能检测，EventSpec 在五类事件上 F1 分别为 100%、90.91%、88%、98.90% 与 76.60%，平均 90.88%；整体综合精度 90.17%，宏平均召回 96.11%。在 off‑chain 侧的攻击实验中，成功重现事件来源混淆与事件‑状态不同步攻击，发现并验证了多起真实漏洞（包括钱包 $600 奖励）。每份合约平均分析时长 1–2 秒（设 1200 ms 上限）。

**⚠️ 局限性**

局限性：①依赖字节码反编译与污点分析，难以覆盖极简/高度优化合约；②事件规范仅基于经验统计，可能对 ERC 外事件产生误判；③未提供正式形式化验证，只给出检测与建议；④仅关注 EVM 兼容链，缺乏跨链交互深度分析；⑤对复杂合约中的 guard 与状态写入仍可能漏检或误检。

---

## 236. Sparks of In Silico Cognitive Science: Theories from Simulated Data Can Generalize to Humans

**arXiv ID:** 2609.08003 | [PDF](https://arxiv.org/pdf/2609.08003v1)

**作者:** Akshay K. Jagadish `[一作]` (Princeton University), Suyog H. Chandramouli `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在行为基础模型Centaur上运行AutoCog闭环理论发现循环，设计实验、收集模拟数据、对抗理论并改进，最终得到在人类实验中表现更好的两种理论。

**💡 创新点**

证明在缺陷的模拟器上也能发现可推广至人类的理论，展示了模拟器对理论差异的足够敏感，而非精确复制行为。

**🔧 技术方法**

AutoCog的四阶段LLM驱动循环、Centaur语言模型的行为模拟、程序合成进行理论修订以及LLM仲裁器决定优胜理论等技术。

**📊 数据集**

160个实验的试次级别数据训练Centaur；对10个保留的人类实验进行外部验证。

**📈 对比分析**

用理论预测与Centaur或人类选择比例的均方误差（MSE）评估，AutoCog在Centaur上MSE从0.110降至0.021，且在人类实验中超过Take‑The‑Best、Tallying，仅落后于人类循环发现的理论。

**⚠️ 局限性**

模拟器的偏差可能导致理论仅拟合模拟器，若实验设计偏离训练分布或理论趋同，仿真精度对发现结果的影响将更大。

---

## 237. The Stretch Factor of Planar Delaunay Triangulations Is Less Than 1.65

**arXiv ID:** 2609.07979 | [PDF](https://arxiv.org/pdf/2609.07979v1)

**作者:** Guanlin Mo `[一作]` (University of Science and Technology of China), Hu Ding `[通讯]` (University of Science and Technology of China)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了任意平面 Delaunay 三角剖分的最坏情况扩展因子不超过 1.65（后续进一步改进到 1.632），显著缩小了已知区间。

**💡 创新点**

提出了基于 Bellman 递归的盘链（disk‑chain）极限框架，将几何问题转化为一维线性规划；构造了可证明的分段三次样条势函数，并使用代理协助搜索得到满足所有约束的可行解。

**🔧 技术方法**

使用 Bellman 递归、势函数技术、线性规划、三次样条构造、精确算术与区间算术、以及 GPT‑辅助多代理搜索等。

**📊 数据集**

无实验数据集，所有结果均为纯理论推导，适用于任意有限非共线点集的指定平面 Delaunay 三角剖分。

**📈 对比分析**

与之前的上界 1.998 以及下界 1.5932 进行比较；新上界 1.65 将误差范围缩小至 0.0568；该方法在理论上已证明可行且性能优于以往的常数改进。

**⚠️ 局限性**

局限性在于仍未达到最优常数（已知下界 1.5932），证明过程复杂且依赖精确算术；方法仅适用于平面 Delaunay 三角剖分，无法直接推广至更广泛的几何网络。

---

## 238. Designing for Healthy, Affordable, and Sustainable Human-HVAC Interactions for Heating in Smart Homes

**arXiv ID:** 2609.07936 | [PDF](https://arxiv.org/pdf/2609.07936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 239. Heat Field Signatures: From Point Clouds to Smooth Geometry

**arXiv ID:** 2609.07975 | [PDF](https://arxiv.org/pdf/2609.07975v1)

**作者:** Yuanqing Wang `[一作]` (University of Texas at Dallas), Baris Coskunuzer `[通讯]` (University of Texas at Dallas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过把不规则点云提升到多尺度平滑热场，并从中直接提取全局和局部几何签名，提出了 Heat Field Signatures (HFS)。

**💡 创新点**

创新点在于使用热场的解析性质提供闭式、旋转不变的多尺度几何描述（热浓度、尺度响应、粗糙度、热维数、对数 Hessian 谱和尺度过渡），并引入 Heat Dimension Spectrum (HDS) 进行全局汇聚。

**🔧 技术方法**

使用的技术包括热场映射、闭式微分/积分运算、log-Hessian 谱分析、热维数估计、分层聚合（闭式或学习式池化）以及与点云网络的融合。

**📊 数据集**

数据集涵盖合成密度/拓扑基准（Processes、Orbit5k、DisksAnnuli）以及真实世界结构（Allen subcellular、NeuroMorpho neurons、FOR‑species LiDAR trees、SCOP 蛋白折叠）。

**📈 对比分析**

与传统描述符（FPFH、HKS）和主流点云网络（PointNet、PointNet++、DGCNN、Point Transformer）以及多参数持久化方法相比，HFS 在所有七个基准上均实现最高精度，并且在运行时显著快于深度网络（如 HFS-full 约 8.5× DGCNN）。

**⚠️ 局限性**

局限性包括在仅靠全局形状或局部纹理决定类别的任务中表现不一定优于坐标网络；对极大点云的计算成本仍高；以及缺乏对更多几何不变量的进一步探索。

---

## 240. MeRoTune: RoPE-Safe Merging with a Tunable Dial

**arXiv ID:** 2609.07971 | [PDF](https://arxiv.org/pdf/2609.07971v1)

**作者:** Salman Faroz `[一作]` `[通讯]`, Salman Faroz

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种利用RoPE可逆校正矩阵进行双侧对齐的模型融合方法，并在融合后可通过调节融合比例实现实时性能调节。

**💡 创新点**

核心创新在于推导出RoPE兼容的可逆校正矩阵类别（每个频率对的缩放旋转），并将其与双侧对齐结合；同时给出了预训练前判断两模型是否值得对齐的数学条件。

**🔧 技术方法**

使用RoPE-commutant矩阵、双侧校正、融合比例折叠训练、固定/随机融合比例、基于基础模型的权重平均与TIES/DARE-TIES的对比评测。

**📊 数据集**

基于Qwen2.5-1.5B-Instruct，分别使用印尼/代码与日语两种细化模型；评测数据集包括MMLU、ARC‑Challenge、印尼语和日语特定任务集。

**📈 对比分析**

与官方TIES/DARE‑TIES实现相比，该方法在所有融合比例下均不落后，且在大多数基准上均优于两者；固定α训练在多数指标上表现最佳，随机α训练虽击败DARE‑TIES但未超越TIES。

**⚠️ 局限性**

局限性包括仅在单一模型架构与单一对模型上验证，固定/随机α的效果尚未系统评估，独立调节融合比例会导致性能下降，且方法仍需在更广泛任务与模型上进行验证。

---

## 241. Bottom-up Modeling of Repeated Elements via Single Image Analysis-by-Synthesis

**arXiv ID:** 2609.07939 | [PDF](https://arxiv.org/pdf/2609.07939v1)

**作者:** Syrine Kalleli `[一作]` (Universite Gustave Eiffel), Mathieu Aubry `[通讯]` (Universite Gustave Eiffel)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种完全无监督的单图像分析-合成方法，能够从单张图像中学习到重复元素的共享原型以及每个实例的姿态、颜色和外观参数，并通过这些参数重建图像。

**💡 创新点**

创新点在于：①将单图像重建任务转化为联合优化共享原型和实例特定参数的分析-合成框架；②引入低维外观编码器实现实例间细微外观差异的建模；③采用分阶段剪裁式预训练与增量式学习策略，显著提升单图像优化的收敛速度和质量；④仅需粗尺度先验即可实现全局尺度自适应。

**🔧 技术方法**

技术上使用ViT‑B/16 DINO自监督特征做图像编码；双分支网络（背景预测 + 元素预测）；元素预测通过transformer解码器与实例查询实现；生成器为小型MLP；使用Huber+SSIM重建损失、稀疏正则化、alpha混合；剪裁和滑动窗口策略完成全图重建；在合成数据上做预训练。

**📊 数据集**

主要数据集为FSC‑147（116张带计数与三盒边框的图像）用于训练与评估；COCO语义分割用于生成5k合成图像做预训练；TPC‑268数据集用于展示学得原型的视觉效果。

**📈 对比分析**

与现有计数方法（ABC123、GeCo、TMR）以及基于TMR+SAM、SpaceJAM、3D重建的元素建模基线比较。该方法在元素建模指标（PSNR≈23.8、SSIM≈0.74、LPIPS≈0.24）和计数指标（MAE≈14.1、RMSE≈26.9）上均优于基线，尤其在无监督场景下显著提升了重建质量和可解释性。

**⚠️ 局限性**

局限性包括：需要非混乱背景且元素纵横比不极端；仅靠粗尺度先验，难以处理极端尺度差异；基于视觉相似性分组，语义相似但视觉差异的实例难以捕获；以及对部分-整体歧义的根本无法完全消除。

---

## 242. BanglaMemeX: Advancing Cultural Metaphoric Image Interpretation in Bangla with a Multimodal Explainable Dataset

**arXiv ID:** 2609.08029 | [PDF](https://arxiv.org/pdf/2609.08029v1)

**作者:** Md. Sadman Sakib `[一作]` (University of Dhaka), Md Fahim `[通讯]` (University of Texas at Dallas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个专门针对孟加拉语网络迷因的多模态基准 BanglaMemeX，并对模型在幽默、讽刺、攻击性、激励意图及整体情感等多维度进行分类，同时生成解释性文本；

**💡 创新点**

首次在低资源语言下引入包含文本与视觉隐喻及人类解释的多模态数据集，评估模型的文化推理与隐喻理解能力，发现标签预测与解释质量不一致的现象；

**🔧 技术方法**

采用零样本提示、链式推理、LLM集会（多模型协同推理）以及参数高效微调（LoRA）等技术对现有 Vision‑Language 模型进行评估；

**📊 数据集**

使用新构建的 BanglaMemeX 数据集（3000 条孟加拉语迷因，包含 5 维标签和多层次隐喻解释），并与现有 Bangla 及多模态数据集做对比；

**📈 对比分析**

在零样本提示下，封闭源模型如 Claude‑Opus‑4.5 在讽刺、攻击性和整体情感上表现最佳；链式推理提升部分标签但整体提升有限；LLM 集会显著提升解释质量但牺牲分类准确率；LoRA 微调在分类上可提升 10% 以上，但对隐喻理解与解释质量几乎无益；

**⚠️ 局限性**

数据集规模有限（3000 条），仅覆盖公开社交媒体内容，缺乏方言与社区多样性；隐喻理解仍弱，尤其是视觉隐喻；模型对文化上下文的把握依赖手工注释，未完全通用；实验仅涵盖少数模型和单一数据集，泛化性待验证。

---

## 243. Mini-Batch Risk-Averse Deep Q-Learning: A Robot Navigation Case Study

**arXiv ID:** 2609.07998 | [PDF](https://arxiv.org/pdf/2609.07998v1)

**作者:** Aayush Patel `[一作]` (Rutgers University), Andrzej Ruszczyński `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于小批量转移风险映射的风险厌恶深度Q学习方法，并在水下机器人导航任务中进行实验验证。

**💡 创新点**

① 将非线性转移风险映射转换为对N个后继状态值的期望，得到可无偏估计的mini‑batch风险映射；② 与双网络DDQN结合，消除动作选择偏差并分析剩余凸性导致的保守偏差；③ 采用层次分解将确定性路径规划交给图搜索，专注学习高层“采集或传输”决策；④ 设计对问题对称性不变的特征映射，实现规模无关、跨实例迁移。

**🔧 技术方法**

Markov风险测度、mini‑batch转移风险映射、双网络DDQN、层次强化学习、Gini/均值风险模型、Dijkstra图搜索、特征工程（对称性不变特征）。

**📊 数据集**

使用仿真生成的水下机器人环境：网格 7×7（12 采集点、2 传输点、5 障碍物）以及 10×10（14 采集点、3 传输点、8 障碍物）等随机配置；300 个留出测试环境。

**📈 对比分析**

通过比较原始特征与工程特征、风险无偏策略（N=2）与风险中性策略（N=1）以及阈值启发式策略，评估平均奖励、上半方差、成功率。结果表明：工程特征显著提升泛化能力（成功率 100%），N=2 风险厌恶策略在均值-尾部风险方面优于 N=1，并在模型误设（显式破坏事件）下进一步提升均值并降低上半方差。

**⚠️ 局限性**

缺乏非线性函数逼近下的收敛性保证；层次分解的半马尔可夫性与时间一致性尚未形式化；对批量大小 N、风险权重 ϰ 的系统性调优研究不足；训练中将破坏事件吸收到折扣因子，未能完全在风险映射内处理破坏风险。

---

## 244. Semi-Supervised Learning under Spatially Biased Sampling

**arXiv ID:** 2609.07982 | [PDF](https://arxiv.org/pdf/2609.07982v1)

**作者:** Bright Wiredu Nuakoh `[一作]` (African Institute for Mathematical Sciences), Ebenezer Afrifa-Yamoah `[通讯]` (Edith Cowan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究空间采样偏倚（分布不匹配）对半监督学习（SSL）性能的影响，并在合成与真实数据上系统评估了不同SSL方法在不同程度的空间偏差下的稳健性。

**💡 创新点**

① 将分布不匹配、空间自相关、空间非平稳三种机制分离；② 在合成生成器上发现分布不匹配导致阈值式性能崩溃；③ 提出基于核加权的局部MMD度量，显著提升了对空间不匹配的检测稳定性。

**🔧 技术方法**

使用的技术包括：自训练（Self‑Training）、标签传播（Label Propagation）、分布对齐重加权（Density‑Ratio Re‑weighting）、一致性方法（Mean Teacher、FixMatch）、图卷积网络、地理加权回归；分布度量包括KL、Wasserstein、MMD、局部MMD；利用分段线性回归检测临界点，并使用Bootstrap估计置信区间。

**📊 数据集**

数据集：合成空间随机场；PovertyMap‑WILDS（贫困地图）；加州住房（California Housing）；美国县级贫困率（Socio‑economic）；EPA PM2.5 空气质量监测（Air Quality System）。

**📈 对比分析**

在合成实验中，标签传播在所有方法中最不易下降（≈4.2个百分点），自训练和监督仅的下降约5–6个百分点；当偏差α超过≈0.71–0.77时，所有方法出现急剧下滑。真实数据中，PovertyMap‑WILDS、加州住房和空气质量均表现出相似的阈值下降；自训练在大部分数据集上最脆弱，标签传播最稳健。分布度量与性能负相关，全球度量（KL、MMD、Wasserstein）相关性最好，局部加权MMD略优于普通网格化局部度量。

**⚠️ 局限性**

局限性包括：阈值位置仅在单一合成生成器上验证，缺乏对其他真实空间任务的泛化；局部MMD在同一偏差等级下预测能力有限；分布意识框架（重加权+空间权重+自适应阈值）在严重偏差下并未提升，甚至退化；空间非平稳的补偿仍未得到有效解决。

---

## 245. $α$-Graph: Attention-Infused Normalizing Flow Approach to Tractable Graph Modeling

**arXiv ID:** 2609.07961 | [PDF](https://arxiv.org/pdf/2609.07961v1)

**作者:** Thanh-Dat Truong `[一作]` (University of Arkansas), Khoa Luu `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于可逆注意力的显式图模型α-Graph，并通过无条件与条件正则化流实现图结构的精确编码

**💡 创新点**

创新点在于设计可逆注意力层与可逆交叉注意力层，既保持可逆性又能显式捕获图结构，并引入可学习查询提升表达能力

**🔧 技术方法**

采用正则化流、可逆注意力机制、可学习查询、图卷积作为基础结构，并结合多层交叉注意力与传统Coupling层

**📊 数据集**

在11个节点分类基准（Cora、Citeseer、Pubmed等）、OGBN-Arxiv、Panoptic Scene Graph Generation等多种图数据集上进行实验

**📈 对比分析**

与GCN、GAT、GraphSAGE、GraphMAE等最先进方法对比，α-Graph在节点分类、场景图生成、边预测等任务上均实现平均提升1–3个百分点，表现最优

**⚠️ 局限性**

受限于计算资源仅在标准规模数据集验证，尚未在更大规模图上系统评估；且可逆注意力实现对稀疏图的效率提升仍有进一步改进空间

---

## 246. Structured Extrema Errors in Classical Surrogates for Viscous Burgers: A Physics-Consistent Interpretation

**arXiv ID:** 2609.07952 | [PDF](https://arxiv.org/pdf/2609.07952v1)

**作者:** Youssef Oubari `[一作]` `[通讯]` (IMT Atlantique), Youssef Oubari (IMT Atlantique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了经典机器学习代理模型对一维黏性 Burgers 方程时间演化的局部误差结构，并提出基于曲率的几何解释与物理归因；

**💡 创新点**

创新点在于揭示残差在预测极值附近形成的曲折分支，并将其与二阶空间导数（黏性扩散项）关联，提供可解释的残差修正方法；

**🔧 技术方法**

采用核岭回归、线性岭回归、ExtraTrees 与随机森林等原始网格模型，结合 PCA、二次曲线拟合、残差回归、谱粗糙度等技术；

**📊 数据集**

使用五模态初始条件生成的 30 条周期性 Burgers 轨迹，网格 128 点，粘度取 0.01、0.02、0.05、0.1，时间步长 1e-4；

**📈 对比分析**

对比四种模型的均方误差、相对 L2 误差和局部分支特征，KRR 在总体误差上最优，但所有模型都表现出极值附近的残差分支；在经过基于曲率的残差修正后，KRR 的 MSE 与 L2 误差显著下降（约 30% 与 20%）；

**⚠️ 局限性**

局限性包括仅针对一维方程与原始网格形式；曲率与扩散归因在树模型上弱；诊断指标如有效黏性因子随预测步长变化；对不同域或更高维系统的推广需进一步验证。

---

## 247. Improved Integrality Gap for Multicommodity Flow on Trees

**arXiv ID:** 2609.07949 | [PDF](https://arxiv.org/pdf/2609.07949v1)

**作者:** Elfarouk Harb `[一作]` `[通讯]`, Elfarouk Harb

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种新的方法，用于在树形网络上求解加权单需求多边际流问题，并证明该线性规划松弛的可积性间隙下界从之前的 1/4 提升到 2/5。

**💡 创新点**

创新点在于构造了一个简单的打包（packing）引理和递归的颜色分配技术，利用树的星形收缩与合并块的方法，实现了对 11 颜色的有效分配，随后进一步优化至 10 颜色，从而获得更紧的 2/5 下界。

**🔧 技术方法**

核心技术包括：① 通过在每条边加上容量为 1 的私有叶子，将“每个请求最多出现一次”约束转化为容量约束；② 使用 Könemann‑Parekh‑Pritchard 的加权约束保持性（bicriteria）近似算法，将每条边的负载最多提升 2；③ 在树上递归收缩星形结构，并在每次收缩后对留下的“块”进行打包，最终得到符合容量要求的 11 或 10 颜色划分；④ 对局部路径的颜色分配采用贪心与边着色理论相结合的方法。

**📊 数据集**

该工作完全是理论性质，不依赖任何实验数据集。

**📈 对比分析**

与先前的 1/4 下界相比，本结果将可积性间隙下界提高到 2/5，且实现了多项式时间算法，能够在给定的线性规划解中直接得到相应的整数解。上界仍为 2/3，说明两者之间仍有一定距离。

**⚠️ 局限性**

局限性包括：仍未达到猜想中的 2/3 下界；方法仅适用于树形网络，难以直接推广到更一般的图；颜色分配的递归与合并步骤虽然可行，但实现复杂度相对较高；以及目前的下界与上界之间仍存在较大差距。

---

## 248. Promises should be taken seriously: On relativization with promise problems

**arXiv ID:** 2609.07945 | [PDF](https://arxiv.org/pdf/2609.07945v1)

**作者:** David Miloschewsky `[一作]` (Stony Brook University), Dorian Rudolph `[通讯]` (Paderborn University)

**通讯引用:** 53 | [OpenAlex ID](https://openalex.org/A5056653308)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了承诺问题与语言在相对化世界中的差异，并提出了两种访问语义（稳健查询和松散访问），证明了它们对复杂度类的影响。

**💡 创新点**

首次给出了在承诺问题下稳健和松散查询语义的定义，并通过构造 Cohen 泛型 oracle 证明了语言与承诺问题在相对化世界中不等价，进一步提升了 Quantum-Classical Polynomial Hierarchy 上限，并提出新的承诺类 PromiseAPP*。

**🔧 技术方法**

采用 Cohen 泛型性、量化阈值问题的 Toda 证明、线性多项式逼近、有限独立哈希以及后选择技术等多种理论工具。

**📊 数据集**

无（纯理论研究，无实验数据集）。

**📈 对比分析**

通过相对化证明与 oracle 构造比较，展示了 PromiseAPP* 与传统 PromiseAPP 的差异；无具体性能指标，结论为理论复杂度提升。

**⚠️ 局限性**

结果依赖于特定的 oracle 构造，无法直接推广到所有承诺问题；对量子模型的后选择依赖于完美的后选择概率，实际实现难度大。

---

## 249. CausalVerify: An Execution-Grounded Benchmark for LLM Causal Inference Workflows

**arXiv ID:** 2609.07944 | [PDF](https://arxiv.org/pdf/2609.07944v1)

**作者:** Yonghong Zhang `[一作]` (Universidad Autonoma De Madrid), Yong Xie `[通讯]` (Spanish National Research Council)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个名为CausalVerify的基准，评估大型语言模型在结构化计量经济学因果估计工作流中的执行正确性；

**💡 创新点**

创新点在于将文本理解、代码可执行性、数值估计以及自评置信度四个维度分离，并引入基于可执行代码与已知参考估计相比较的L2b+正确性指标；

**🔧 技术方法**

采用了多轮LLM交互、R语言代码生成与执行、正则表达式与自动化数值提取、以及统计相关性（Kendall τ、Spearman ρ）评估；

**📊 数据集**

使用了259篇真实经济学论文（提取研究问题、数据描述、机构背景）和100个固定种子合成DGP数据集（包括DID、事件研究、IV、RDD四类设计）；

**📈 对比分析**

与四个LLM共识标签的文本一致性、L2b+执行正确率、以及自评置信度对比。实验结果显示：文本一致率最高约为90%，但L2b+通过率仅为10%–88%；L2b排名与L2b+高度相关（τ≈0.81），而文本方向一致性与L2b+关联弱（τ≈-0.2~0.1），自评置信度对区分正确/错误工作流效果不显著；

**⚠️ 局限性**

局限包括：只评估单一实验（R语言执行）、仅覆盖四类因果设计且不包含匹配、倾向评分、异质效应等复杂方法；共识标签可能存在噪声；自评置信度测量为回溯性且提示依赖；实验仅包含闭源大模型，开放权重系统的覆盖有限；执行层面可能因语言/软件环境差异而变异。

---

## 250. From Version Conflicts to Decision Conflicts: Selective Revalidation for Long-Running AI Agents

**arXiv ID:** 2609.08015 | [PDF](https://arxiv.org/pdf/2609.08015v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 251. ReactVAU: A Slow-Fast Decoupled Framework for Streaming Video Anomaly Understanding

**arXiv ID:** 2609.07941 | [PDF](https://arxiv.org/pdf/2609.07941v1)

**作者:** Chia-Hui Chen `[一作]` (National Tsing Hua University), Shang-Hong Lai `[通讯]` (National Tsing Hua University)

**通讯引用:** 5559 | [OpenAlex ID](https://openalex.org/A5073849580)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于慢速‑快速分离的实时流式视频异常理解框架 ReactVAU，能够在不观察未来帧的条件下实现异常检测与语义解释。

**💡 创新点**

创新点在于：① 使用 Spatial Grid Folding (SGF) 将短时序列映射为二维网格，让轻量级 VLM 直接完成异常检测；② 设计 Anomaly‑Aware Persistent Memory (AAPM) 通过异常优先得分与异常池保护稀疏异常特征，防止在持续压缩过程中丢失；③ 采用慢速-快速分离策略，仅在 Fast 模块触发时才唤醒重量级 7B MLLM 进行语义验证，显著降低计算量与延迟。

**🔧 技术方法**

技术手段包括：PaliGemma2‑3B 作为 Fast 模块，StreamForest‑7B 作为 Slow 模块；SGF、AAPM（含异常优先分数与异常池）、动态稠密采样、事件门控、以及加权融合 S_fused = 0.4·S_det + 0.6·S_reason。

**📊 数据集**

使用 UCF‑Crime、XD‑Violence 进行异常检测评估，使用 HIVAU‑70K 进行异常理解评估，三大公开数据集涵盖多级时间粒度。

**📈 对比分析**

与现有在线/离线基线对比，ReactVAU 在 UCF‑Crime AUC 88.44%、XD‑Violence AP 88.50% 甚至超过部分离线 fine‑tuned 模型；在 HIVAU‑70K 上 CIDEr‑E、CIDEr‑V 以 2.032/2.016 等分数领先；同时 MLLM 调用次数下降 54%（实际）或 95%（理论），平均推理延迟从 216 ms 降至 98 ms/查询。

**⚠️ 局限性**

局限性包括：仍依赖阈值调优；慢速 7B 模型在极低异常率下的冷启动成本；对极端光照、遮挡等环境鲁棒性待进一步验证；以及对大规模持续流时的内存管理与实时性保障仍有改进空间。

---

## 252. A Multimodal Label Forecasting Method for Aperiodic Visuo-Motor Time Series

**arXiv ID:** 2609.07930 | [PDF](https://arxiv.org/pdf/2609.07930v1)

**作者:** Borui He `[一作]` (Syracuse University), Garrett E Katz `[通讯]` (Syracuse University)

**通讯引用:** 360 | [OpenAlex ID](https://openalex.org/A5051693199)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的时间序列预测任务：在无周期的机器人步态数据上，利用自我视觉和关节角度预测未来每个时刻是否会跌倒，并构建了对应的真实与仿真数据集；同时给出了专门针对该任务的多模态深度学习基线模型EMP。

**💡 创新点**

创新点在于：①定义了无周期的细粒度分类预测任务，突破传统周期性时间序列基准；②首次将计划好的关节轨迹作为外生变量与视觉、运动感知信息一起输入；③通过i.i.d.采样、空间归一化等设计提升模型鲁棒性。

**🔧 技术方法**

技术方面采用了三流编码器：运动编码器使用DCT+MLP+实例归一化；视觉编码器利用低通FFT+3D卷积；轨迹编码器采用FFT筛选主频+二维卷积；整体网络结构简单高效，并在训练中严格控制样本独立性。

**📊 数据集**

数据集方面，真实数据RP取自Poppy人形机器人，16fps、110段，包含办公室、实验室、走廊三种环境；仿真数据SimP在PyBullet中生成2K段，30fps，并通过随机轨迹扰动诱发跌倒。

**📈 对比分析**

实验中将EMP与TimesNet、FlowFormer、EgoFalls等SOTA时序预测模型进行对比，评估预测时刻P={6,12,18,24}（RP）或{15,30,45,60}（SimP）的分类准确率；EMP在RP上平均达95–96%准确率，SimP上达95–96%，分别比基线高约12%与5%，并通过Welch t检验验证显著性。

**⚠️ 局限性**

局限性在于：①数据集规模仍有限，尤其真实数据难以覆盖更多跌倒场景；②对外生变量的依赖使模型对轨迹规划的可靠性要求高；③视觉模块可能出现过拟合，且对不同机器人或环境的泛化能力尚未充分验证。

---

## 253. Eliciting Self-Verification in Multimodal Reasoning Agents with Reinforcement Learning

**arXiv ID:** 2609.08025 | [PDF](https://arxiv.org/pdf/2609.08025v1)

**作者:** Vishwas Sathish `[一作]` (University of Washington), Douglas Gray `[通讯]` (Amazon)

**通讯引用:** 10808 | [OpenAlex ID](https://openalex.org/A5110967135)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种只用强化学习（RL）的微调框架SVRL，训练多模态智能体在多跳视觉问答中自行验证并过滤检索证据，以减少对外部验证器的依赖；

**💡 创新点**

创新点在于通过搜索校准因子和查询多样性奖励实现细粒度搜索决策控制，并在训练时引入自验证标签，使证据筛选成为可学习的轨迹级信号；

**🔧 技术方法**

主要技术包括基于GRPO的强化学习微调、Dr. GRPO稳定化、搜索校准奖励、查询多样性奖励、以及自验证与外部验证器对齐的监督；

**📊 数据集**

使用的主要数据集是FVQA（训练5000例、测试1800例）及其衍生的InfoSeek、LiveVQA、MMSearch、SimpleVQA等多跳问答基准；

**📈 对比分析**

与MMSearch-R1++等基线相比，SVRL在FVQA和InfoSeek的准确率提升约8–9个百分点，同时搜索比率下降，整体性能逼近大型专有模型GPT‑4o；

**⚠️ 局限性**

局限包括依赖于网络检索工具的可用性与质量、对训练时验证器标签的依赖、以及对提示语和工具格式的敏感性。

---

## 254. "Shut Up and Let Me Enjoy My Otome": Understanding and Measuring the Toxicity in Otome Game Communities

**arXiv ID:** 2609.08009 | [PDF](https://arxiv.org/pdf/2609.08009v1)

**作者:** Yage Zhang `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对日本浪漫模拟游戏（Otome）社区进行大规模毒性分析，构建并标注数据集，评估多种毒性检测模型。

**💡 创新点**

首次系统测量Otome社区毒性，提出基于LLM的定制检测器，揭示平台间毒性差异、事件触发峰值及潜在协调行为。

**🔧 技术方法**

使用LLM驱动提示推理（DeepSeek‑V3、GPT‑4o mini 等），与通用检测器 Perspective API、OpenAI Moderation、COLD 进行对比。

**📊 数据集**

采集 620,045 条微博与 Reddit 帖子，人工标注 4,308 条为黄金标准，用于训练与评估。

**📈 对比分析**

LLM 驱动模型在微博上 F1 = 0.82，Reddit 上 F1 = 0.78，显著优于通用检测器；同时通过相似度聚类识别 191 组潜在协调集群。

**⚠️ 局限性**

研究仅涵盖微博与 Reddit，受限于中文和英文；LLM 模型虽表现优异但仍存在误判；未能确认协同意图，且缺乏跨平台和多语言的泛化验证。

---

## 255. Sharp Structure-Agnostic Minimax Risk for Partial Linear Models

**arXiv ID:** 2609.07997 | [PDF](https://arxiv.org/pdf/2609.07997v1)

**作者:** Haichen Hu `[一作]` (Massachusetts Institute of Technology), David Simchi-Levi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 21779 | [OpenAlex ID](https://openalex.org/A5112431388)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在部分线性模型中，使用两个不同的黑箱学习器进行系数估计时的结构无关的最小最大风险，解决了双机器学习中的一个开放问题。

**💡 创新点**

创新点在于提出了一种新的下界，用于描述两个学习器问题的复杂性，并通过四个有限混合测试实验来捕捉不同的近似误差和学习误差之间的相互作用。

**🔧 技术方法**

使用了结构无关的最小最大风险框架，结合了局部Rademacher复杂性来控制随机误差。

**📊 数据集**

使用了2n个独立观察值的部分线性模型数据集，其中包含处理变量和结果变量的噪声。

**📈 对比分析**

通过与现有的上界进行比较，证明了所提出的下界与最新的上界相匹配，表明标准的双机器学习可能会高估目标估计的内在难度。

**⚠️ 局限性**

限制在于该研究假设了学习器类之间没有额外的结构关系，且在实际应用中，四个预算可能是未知的。

---

## 256. Algorithmic List Decoding of Reed-Solomon Codes up to Capacity

**arXiv ID:** 2609.08005 | [PDF](https://arxiv.org/pdf/2609.08005v1)

**作者:** Joshua Brakensiek `[一作]` (University of California), Kai Zhe Zheng `[通讯]` (Simons Institute for the Theory of Computing)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种确定性多项式时间列表解码算法，能够在任意常数码率下对任意评估点集合的Reed–Solomon码实现接近列表解码容量（即可在误差比例接近1‑R处成功解码），并且不依赖随机评估点或大域；

**💡 创新点**

关键创新在于使用隐藏的Hasse导数构造插值多项式，并通过Taylor恒等式将原本过强的插值约束转化为可行的线性约束；该方法显著降低了约束数，突破了Johnson半径壁垒，实现了接近容量的列表解码；

**🔧 技术方法**

主要技术包括：多变量插值与多重性约束、Hasse导数与泰勒展开、加权度数与格点计数、线性系统求解、Kopparty的多重性根查找算法；通过构造合适的插值空间和约束矩阵实现可解的非零解；

**📊 数据集**

论文为理论性研究，没有使用具体数据集；所有实验与分析均为理论证明与参数估计；

**📈 对比分析**

相较于以往基于随机评估点或Folded Reed–Solomon的列表解码方法，本文实现了确定性算法并在常数码率下达到列表解码容量；算法运行时间为q^{O(1/θ)}，列表大小为q^{O(1/θ)}，在理论上优于已知的Johnson半径内解码方案；

**⚠️ 局限性**

限制包括：算法仅适用于常数码率；所需域大小q需为Θ(n)（或更大）且参数如θ、d 的取值使得复杂度高；对高码率或非prime域的直接适用性有限；

---

## 257. Geometric Function Atlas: certified computing for geometric function theory in Python

**arXiv ID:** 2609.07969 | [PDF](https://arxiv.org/pdf/2609.07969v1)

**作者:** Kishan Gurumurthy `[一作]` (Indian Institute of Information Technology Kottayam), Asha Sebastian `[通讯]` (Indian Institute of Information Technology Kottayam)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `847a60d8-a755-47af-ba5d-c5236b9e3083` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个开源Python包geometric-function-atlas，用于计算Ma–Minda类的尖锐常数、系数证书和定向半径，并提供精确、屏蔽和认证三层证据；

**💡 创新点**

创新点在于将39个Ma–Minda生成器的全套数据集与可回放的证明链、分层验证器和版本化的artifact快照相结合，实现了可机器可复现的尖锐常数计算与证书重现；

**🔧 技术方法**

采用SymPy实现符号精确算术，mpmath实现区间算术与严格封闭，利用Schur参数展开、手工记录的证明链、JSON schema验证以及CLI/API接口；

**📊 数据集**

使用的数据集包括306条系数证书、702条定向半径记录、39个生成器定义、以及带校验和的artifact快照和SQLite文献引用库；

**📈 对比分析**

通过在Apple M3 Pro上基准测试，所有操作均在0.1秒内完成；回放时间与精度无关；手工验证与自动回放性能一致，并通过与文献和网站列表对照进行验证；

**⚠️ 局限性**

局限性在于仅覆盖Ma–Minda解析类，支持的性质仅为星形、凸、单值、Becker与Nehari；不含多值或 meromorphic 扩展；回放证明链仅覆盖八条定向半径；且未实现符号μ的参数化。

---

## 258. Reasoning Beyond Transcription: Audio Language Models on Child Stuttering Speech

**arXiv ID:** 2609.07968 | [PDF](https://arxiv.org/pdf/2609.07968v1)

**作者:** Chibuzor Okocha `[一作]` (University of Florida), Zoey Liu `[通讯]` (University of Florida)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5063501461)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估并比较最新音频-语言模型（ALMs）在含有言语不流畅（重复、停顿等）儿童语音的语义摘要和语义推理（包含推论、无关、矛盾）任务中的表现。

**💡 创新点**

提出针对多说话人混合访谈环境的“儿童专注语义推理”范式，并设计了基于难度的推论评测与对比实验，揭示 ALMs 在高密度不流畅与成人干扰下的系统偏倚与瓶颈。

**🔧 技术方法**

使用多种端到端 ALMs（Audio‑Flamingo3/2、Kimi‑Audio、Qwen2‑Audio、SALMONN、GAMA 等）以及 ASR+LLM 级联基线（Whisper/Granite + Llama3.2/Mistral/Qwen），并通过 LLM 评判者、BERTScore、宏平均准确率/ F1 进行量化评估。

**📊 数据集**

利用 Voices of Children Who Stutter 语料库（44 条录音，22 名患有口吃的儿童，包含单说话人朗读与多说话人访谈两种环境），并人工验证了 186 条推论假设，确保实验的临床可信度。

**📈 对比分析**

实验表明：Audio‑Flamingo3 与 Kimi 在子任务中获得最高的整体分数（摘要平均分 3.4‑3.9；推论准确率 0.65‑0.68），但所有模型在面对高密度不流畅或成人说话时性能显著下降；ALMs 在推论任务中显著倾向于“蕴含”类别，矛盾检测能力不足；相比之下，Whisper+Qwen 级联在准确率与 F1 上略优，说明语音识别准确性对推理仍有显著影响。

**⚠️ 局限性**

局限性包括：数据集规模有限且不平衡（仅 22 名儿童，填充停顿占主导），推论假设覆盖面有限；模型普遍对矛盾类别预测稀缺，导致评估结果偏向高准确率；缺乏针对成人说话与儿童不流畅交叉影响的深层语义分析；未来工作需扩大数据量、丰富不流畅类型并改进模型对多说话人干扰的鲁棒性。

---

## 259. Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation

**arXiv ID:** 2609.07965 | [PDF](https://arxiv.org/pdf/2609.07965v1)

**作者:** Keren Artiaga `[一作]` (ADAPT Centre), Mohammed Hasanuzzaman `[通讯]` (ADAPT Centre)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三种主流无Gloss的签名翻译模型（GFSLT‑VLP、GASLT、SignCL）进行签名折叠交叉验证，系统评估其在PHOENIX14T和CSL‑Daily数据集上的签名独立性能，并通过定量（BLEU‑4、ROUGE‑L）与定性分析揭示签名依赖导致的过高评估。

**💡 创新点**

首次在签名翻译领域引入签名折叠交叉验证框架，揭示了默认签名重叠拆分会显著夸大模型效果，并提出了签名独立评估协议、句子级拆分与数据集重构方案，推动更可靠的泛化评测。

**🔧 技术方法**

使用签名折叠交叉验证、BLEU‑4/ROUGE‑L评估、句子泄漏统计与可视化、以及手工定性错误分析，结合三种无Gloss模型的训练与推理。

**📊 数据集**

PHOENIX14T（德语天气预报视频）和CSL‑Daily（中文日常话题视频），两者均包含多位签名者且存在句子级重复。

**📈 对比分析**

在默认签名重叠拆分与签名独立拆分之间进行对比，发现PHOENIX14T上GFSLT‑VLP的BLEU‑4从21.44降至10.53，GASLT从15.74降至10.24，SignCL从22.74降至4.18；CSL‑Daily在签名独立拆分下BLEU‑4平均仅3.63，显著低于默认拆分的4.07。结果表明签名依赖会导致性能高估。

**⚠️ 局限性**

评估仅覆盖公开实现的模型，未涵盖可能更强的闭源方法；未尝试骨骼或其他输入特征；未对Gloss‑to‑Text模型进行签名独立评测；缺乏多模态或跨语言数据集的进一步验证。

---

## 260. Guppy: Efficient Light Clients via Recursive Zero-Knowledge Proofs

**arXiv ID:** 2609.07963 | [PDF](https://arxiv.org/pdf/2609.07963v1)

**作者:** George Danezis `[一作]` (Mysten Labs), Karl Wust `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在高吞吐量区块链上设计了一个无状态验证器的轻客户端协议，使验证器只需签名更新日志，而全状态由离线 ZK 服务维护。

**💡 创新点**

创新点在于通过哈希链承诺和并行递归证明，将完整状态更新拆分为可并行处理的子块，保持验证器负担极低，同时实现了完整性查询。

**🔧 技术方法**

使用 Plonky2 递归零知识证明、Poseidon 哈希、Merkle 树、并行聚合证明等技术。

**📊 数据集**

利用 Sui 区块链一月的数据（约 8,700 更新/秒峰值）进行基准测试。

**📈 对比分析**

与传统全状态根轻客户端对比，实验显示可处理数千更新/秒，单次证明约 150 KB，整体延迟 1–3 秒，验证器开销 <10 ms。

**⚠️ 局限性**

限制在于需要足够的并行证明机器以满足吞吐量，且对波动的更新率仍需预先设定最大束大小；递归证明的时延在极高频率时仍可能成为瓶颈。

---

## 261. Geometry-Informed Distributed Acoustic Scene Understanding

**arXiv ID:** 2609.08026 | [PDF](https://arxiv.org/pdf/2609.08026v1)

**作者:** Yiyuan Yang `[一作]` (University of Oxford), Andrew Markham `[通讯]` (University of Oxford)

**通讯引用:** 12728 | [OpenAlex ID](https://openalex.org/A5060183988)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于几何信息的分布式声学场景理解框架，利用多房间分布式麦克风、音频频谱变换器、拓扑感知图神经网络和冻结的大语言模型实现物理一致的叙事生成

**💡 创新点**

创新点在于将室内几何拓扑融入声学特征融合与LLM推理，克服非视距声学遮蔽，提升语义一致性与空间逻辑性

**🔧 技术方法**

使用音频频谱变换器(AST)、拓扑感知图神经网络、GRU时序建模、Meta Llama‑3‑8B‑Instruct冻结模型等技术

**📊 数据集**

使用自定义多房间模拟器生成的16 kHz人声和环境声数据集，并提供JSON格式平面图及麦克风位置信息

**📈 对比分析**

与集中式单麦克风基线和分布式无几何基线比较，系统在Triplet F1、BLEU‑4、ROUGE‑L、BERTScore和空间一致性得分(SCS)上均取得显著提升，Triplet F1达0.87，BLEU‑4 0.55，SCS 88.2%

**⚠️ 局限性**

主要局限为仅在仿真环境中验证，缺乏真实物理测试；推理速度和实时性能未深入优化，需要进一步实地验证

---

## 262. Automated Chest CT Protocol Selection via Large Language Model Derived Text Embeddings from Imaging Request Text

**arXiv ID:** 2609.07986 | [PDF](https://arxiv.org/pdf/2609.07986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 263. URL Extraction from Scholarly Documents: A Cross-Format Comparative Analysis

**arXiv ID:** 2609.08019 | [PDF](https://arxiv.org/pdf/2609.08019v1)

**作者:** Rochana R. Obadage `[一作]` (Old Dominion University), Jian Wu `[通讯]` (Old Dominion University)

**通讯引用:** 20007 | [OpenAlex ID](https://openalex.org/A5033054114)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估了六种文档格式（文本+注释层、LaTeX、HTML、XML、Markdown、PNG）以及63种组合的URL抽取效果，并对200篇arXiv论文的2338个手工标注URL进行评测。

**💡 创新点**

提出三维URL复杂度分类法；构建跨年代、跨领域的多格式基准集；发现TEXTWAL+LaTeX组合在整体抽取精度上优于单一格式，尤其在开放数据集和软件链接上表现突出。

**🔧 技术方法**

使用PDF-to-text工具（PyMuPDF、PyPDF、PDFMiner.six）、GROBID、LaTeXML、Marker、VLM（多种模型）、LLM（OpenAI/Claude等）进行格式特定抽取，并将注释层合并为TEXTWAL。

**📊 数据集**

基准数据集为200篇arXiv论文（1992–2024年），共2338个URL，其中约20%为开放数据集/软件链接；另对467,767篇论文做纵向大规模抽取验证。

**📈 对比分析**

与传统正则/属性搜索方法相比，TEXTWAL在单一格式下精度最高，TEXTWAL+LaTeX在多格式融合后达成最佳整体召回与精度；在开放链接抽取上同样领跑；提供可视化的成本-性能对比。

**⚠️ 局限性**

限制包括：仅覆盖arXiv源文件，其他期刊/会议格式未覆盖；VLM/LLM模型受限于成本与版本可复现性；未对非PDF来源的扫描文档进行深度分析；复杂度分类仍需进一步细化。

---

## 264. Sequential Offering in On-Demand Platforms: On the Optimality of Greedy Ranking

**arXiv ID:** 2609.08001 | [PDF](https://arxiv.org/pdf/2609.08001v1)

**作者:** Hongyao Ma `[一作]` (Columbia University), Matias Romero `[通讯]` (Columbia University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究在按需平台上如何通过顺序邀请工人并动态设定工资，来最大化平台福利或利润。

**💡 创新点**

创新点在于证明：当工人私有成本分布满足密度非递减且凸时，简单的贪心排序（按可观测匹配价值降序）配合后向递推设定工资即可达到全局最优；并给出了分布无关的最坏情况下贪心方案相对最优的比例 n/(2n−1)。

**🔧 技术方法**

主要技术包括：动态规划求解最优工资、利用增量价值函数 ψ 的单调与 Lipschitz 性质、局部稳定性与全局最优的交换（swap）论证，并用凸性、单调性等分布条件来证明贪心排序的最优性。

**📊 数据集**

论文没有使用具体实验数据集，而是通过理论证明与大量数值模拟（覆盖多种成本分布族）来检验贪心策略的表现。

**📈 对比分析**

通过与离线先知（prophet）和最优在线（任意排序）基准比较，数值实验显示贪心方案在绝大多数分布下几乎达到最优，甚至在已知可导致贪心失效的分布族中，其福利损失远低于理论最坏情况（例如对八名工人的最坏比例仍高于 88%），证明其鲁棒性。

**⚠️ 局限性**

局限性包括：假设工人私有成本独立同分布且与匹配价值仅通过位置偏移关联；未考虑多任务、并行邀请或工人间的相关性；模型未涵盖在线学习与动态成本变化的情形。

---

## 265. When Can LLM Digital Twins Reduce Human Measurement? From Behavioral Fidelity to Statistical Substitutability

**arXiv ID:** 2609.07987 | [PDF](https://arxiv.org/pdf/2609.07987v1)

**作者:** Steven Wang `[一作]` (University at Buffalo), Kenneth Joseph `[通讯]` (University at Buffalo)

**通讯引用:** 16742 | [OpenAlex ID](https://openalex.org/A5021488941)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估LLM数字孪生在行为研究中是否能在保持有效推断的前提下减少人类测量。

**💡 创新点**

提出“统计可替代性”概念，构建四维评估框架（聚合逼真度、配对个体信号、有限样本标签恢复、群体稳定性），并在Twin‑2K与Moore‑Berg实验中进行验证。

**🔧 技术方法**

采用预测辅助推断（PPI）、混合主体设计、相关系数估计与有效样本量计算等统计方法，并利用LLM生成预测。

**📊 数据集**

使用Twin‑2K（2058名美国受访者的500+问卷数据）和Moore‑Berg党派认知实验的数据。

**📈 对比分析**

通过与人类结果的聚合一致性、残差化相关系数、PPI效率阈值以及有限样本实验进行比较；结果显示聚合一致性高但个体信号弱，PPI在大多数任务下无法显著降低人类样本。

**⚠️ 局限性**

局限包括仅评估单一基线Twin‑2K管线、任务范围有限、未覆盖新颖估计量或真实目标样本、模型与人类误差关系跨群体的不稳定性以及未计入生成高质量人类标签的成本。

---

## 266. From Event Logs to Governed Action: A BlueSky Agenda for Agentic Process Mining

**arXiv ID:** 2609.07984 | [PDF](https://arxiv.org/pdf/2609.07984v1)

**作者:** Yiyuan Yang `[一作]` (University of Oxford), Qingsong Wen `[通讯]` (Squirrel Ai Learning)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出从事件日志到可治理行动的过程挖掘议程，定义了四个可挖掘工件（表示、证据、治理、评估），并阐述了实现面向代理决策的框架。

**💡 创新点**

将过程挖掘目标从描述性/预测性转为可执行、有证据、可治理的行动建议，首次系统化定义治理合同与证据层级，强调隐私与跨组织安全。

**🔧 技术方法**

基于对象中心日志、因果与预期监控方法、联邦学习/差分隐私、LLM交互接口，结合深度学习表示与因果推断技术。

**📊 数据集**

目前仅提出未来的 E2A‑Bench 公共基准，暂无实验数据集；讨论了使用现有的 OCEL 2.0、SimBank、ProCause 等数据集。

**📈 对比分析**

文章未给出实验或性能评估，提出评估对象与指标（校准、因果有效性、隐私成本、拒绝质量等）以供后续研究。

**⚠️ 局限性**

缺乏可验证的实证评估；日志观测缺失与标签漂移导致证据不确定；因果推断受限于观测偏倚；跨组织治理与隐私约束难以统一。

---

## 267. Solving the Elastic Wave Equation with Physics-Informed Neural Networks: A Robust and Critical Assessment

**arXiv ID:** 2609.07983 | [PDF](https://arxiv.org/pdf/2609.07983v1)

**作者:** Davide Staub `[一作]` (ETH Zuerich), Ben Moseley `[通讯]` (ETH Zuerich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将物理信息神经网络（PINNs）应用于地震学中的弹性波方程，构建并评估了从基础全连接网络到结合波动物理特征（自定义波形层、编码器/解码器）的多种网络架构，并引入源位置条件化技术，实现一次训练即可推断任意震源位置的波场。

**💡 创新点**

创新点在于（1）将弹性波的物理特性直接嵌入网络结构（如波形层和正交正弦层），显著提升精度；（2）首次在PINN中实现震源位置条件化，使模型可泛化到多种源位置；（3）通过硬约束Ansatz与自动吸收边界相结合，减少额外边界损失项。

**🔧 技术方法**

采用物理信息神经网络（PINNs），以全连接前馈网络为基础，结合自定义激活函数（tanh）、Sobol采样、硬约束Ansatz、Adam+LBFGS优化器，以及自动微分计算高阶导数。

**📊 数据集**

使用自研的DEVITO有限差分模拟生成的弹性波场作为参考数据，构造了多种参数模型（常数、混合高斯、层状）并在这些模型上生成100,000个采样点作为训练/验证数据；无真实观测数据。

**📈 对比分析**

通过与DEVITO有限差分解的相对L₂误差比较来评估模型精度；基础PINN在常数参数下平均误差约2.2%，混合模型约2.9%，层状模型约5%；引入物理层后误差约减半；在源位置条件化后，单次前向推断的速度超过传统数值模拟数百倍。

**⚠️ 局限性**

主要限制包括：对超参数（网络宽度/深度、t₁值、采样密度）高度敏感；在高频/强异质介质中仍受谱偏差影响；训练成本高、需要大量采样点；边界处理依赖自动吸收，可能在复杂几何下产生误差；模型在超出训练分布的参数变化下泛化能力有限。

---

## 268. Streaming Hierarchical Inference with Tabular Foundation Models

**arXiv ID:** 2609.07956 | [PDF](https://arxiv.org/pdf/2609.07956v1)

**作者:** Vitor Crista `[一作]`, Goreti Marreiros `[通讯]` (Polytechnic of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种名为HINT的层次推理框架，在边缘侧使用图基近似最近邻检索做局部预测，并在置信度低时将样本与邻域上下文下发给云端Tabular Foundation Model（TabPFN）进行完整推理；

**💡 创新点**

创新点在于将边缘检索与云端推理动态耦合，提出基于置信度阈值和邻域大小的可调offload策略，并通过滑动窗口图结构实现子线性近似最近邻搜索，进而实现通信-准确性Pareto前沿；

**🔧 技术方法**

采用了SWINN/NN-Descent近似最近邻图、TabPFN Transformer、双阈值offload机制、滑动窗口图维护与本地加权投票推理等技术；

**📊 数据集**

使用USP Data Stream Repository中的多分类/二分类数据集（NOAA、ELEC、METER、RIALTO、POSTURE、COVER）进行实验；

**📈 对比分析**

通过预序评估与Adaptive Random Forest、Hoeffding Tree等基线比较，实验显示HINT在20%–50%offload率下在大多数数据集上实现了10%–30%准确率提升，同时将通信量降低到约一半；

**⚠️ 局限性**

限制在于阈值与邻域大小需手工调参，未做在线最优控制；通信成本模型简化，未考虑延迟、能耗、批处理等实际因素；对极大规模或高维流需进一步优化计算与存储。

---

## 269. Support Topology and Gradient Mixing in Sinkhorn Layers

**arXiv ID:** 2609.07954 | [PDF](https://arxiv.org/pdf/2609.07954v1)

**作者:** Dylan Forde `[一作]` `[通讯]` (Independent Researcher), Dylan Forde (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文研究了固定支持的Sinkhorn缩放，揭示了一个精确的商Markov算子，并提出了固定支持下的商运输组件的理论。

**💡 创新点**

创新点在于提出了一个面格二分法，证明了支持-边际对在商混合证书下的条件，并提供了严格的商运输界限。

**🔧 技术方法**

使用了固定支持的Sinkhorn计算、商Markov算子、Dobrushin系数等数学工具。

**📊 数据集**

使用了有限的活跃分数和兼容的正边际数据集。

**📈 对比分析**

通过比较不同的支持-边际对，验证了商混合的有效性，结果表明在特定条件下，商算子可以收缩到任意接近于1。

**⚠️ 局限性**

限制在于未能证明支持变化的训练规则、生产内核、硬件加速或任务质量提升。

---

## 270. Service Health Engineering for Distributed Systems

**arXiv ID:** 2609.08020 | [PDF](https://arxiv.org/pdf/2609.08020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 271. Beliefs and Behavior in Language Models

**arXiv ID:** 2609.07943 | [PDF](https://arxiv.org/pdf/2609.07943v1)

**作者:** Alex Smolin `[一作]` (Toulouse School of Economics), Bryan Wilder `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3066 | [OpenAlex ID](https://openalex.org/A5079207566)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于潜在变量的框架，用来检验大型语言模型（LLM）是否具备可预测的“信念”，并在三类决策任务中评估其内部一致性。

**💡 创新点**

创新点在于：①将信念抽象为单一潜在变量并通过对不同指令的多重输出进行建模；②实现零样本跨框架、跨领域的预测验证；③提供衡量信念测量质量和框架效应的实证指标；④探讨模型层级与实例层级的信念差异。

**🔧 技术方法**

使用概率潜在变量模型（z ∈ [0,1]），logit 与 Beta 链接函数，基于 Gauss–Legendre 分段线性先验进行参数估计；评估指标包括 concordance index、R²、AUC 与 ICC。

**📊 数据集**

数据集包括：美国全国健康与营养调查（CKD 筛查）、Kaggle Agent Arena Werewolf 互动游戏、ashokkumar2026large 社会科学实验结果预测。

**📈 对比分析**

比较方法：对不同模型能力等级进行训练/测试，使用留一法评估预测性能；结果显示：高能力 LLM 在所有任务中取得 0.8–0.95 的 concordance、60–90% 的 R²，零样本跨框架/领域的 AUC 接近 1；低能力模型表现显著下降。

**⚠️ 局限性**

局限性：①仅涵盖二元未知的单步决策场景；②未检验内部机制是否真正包含“信念”；③低能力模型仍存在显著框架效应；④模型与实例层面的差异分析尚不完整，需进一步研究更复杂任务。

---

## 272. TDDN: Text-aligned Diffused DINO Network for Puzzle Understanding

**arXiv ID:** 2609.07937 | [PDF](https://arxiv.org/pdf/2609.07937v1)

**作者:** Harsha Patnala `[一作]` (Eightfold AI), Somak Aditya `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 424 | [OpenAlex ID](https://openalex.org/A5071844229)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种新型视觉-语言模型TDDN，融合了CleanDIFT和DINOv3的视觉特征，并在仅590k图文对的低资源场景下实现文本对齐，显著提升了细粒度感知与结构化推理能力。

**💡 创新点**

创新点在于将扩散模型的空间精细度与ViT模型的语义一致性融合，形成既能保留细粒度视觉细节又能对齐语言的视觉编码器，并通过轻量级对齐实现高效跨模态匹配。

**🔧 技术方法**

使用的关键技术包括CleanDIFT特征提取、DINOv3 ViT特征融合、低资源信息对齐（InfoNCE+STRUCTURE正则）、以及Contrastive Region Guidance（CRG）对冻结VLM的空间引导。

**📊 数据集**

主要数据集包括LAION-5B Aesthetics重新标注集、MS‑COCO‑2014、以及自研的Puzzle Perception（迷宫、国际象棋、汉诺塔、N‑Queens）用于评估细粒度分割与问答。

**📈 对比分析**

与CLIP、SigLIP 2、MetaCLIP等基准相比，TDDN在ADE20K、Cityscapes、COCO‑Stuff等分割任务上提升约15–20%，在Puzzle Perception上提升近2倍；在图文检索方面与CLIP相当或更优；在分类任务上仍略低。

**⚠️ 局限性**

局限性包括对大规模多样化图文对的依赖仍有限，仍在分类等语义任务上略逊于大模型；冻结backbone限制了对新领域的适应性；CRG对模型头部的依赖导致弱模型受益显著，而强模型提升有限。

---

## 273. Delusions and Harms Associated with AI Chatbot Use: Early Evidence from 185 Real-World Reports

**arXiv ID:** 2609.08027 | [PDF](https://arxiv.org/pdf/2609.08027v1)

**作者:** Hamilton Morrin `[一作]` (King's College London), Thomas A. Pollak `[通讯]` (King's College London)

**通讯引用:** 9553 | [OpenAlex ID](https://openalex.org/A5039765242)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对自选的 185 份 AI 聊天机器人使用导致的心理健康伤害报告进行交叉截面分析，描述其中的精神病学特征、机器人行为、时序与结果；

**💡 创新点**

首次系统汇总真实世界的 AI 聊天机器人相关心理伤害案例，并揭示与幻觉、机器人认同、严重社会及临床后果相关的安全风险信号；

**🔧 技术方法**

采用在线问卷收集数据，配对临床评估员对报告进行编码，使用 Cohen’s κ 评估编码一致性；

**📊 数据集**

The Human Line Project 网站收集的匿名调查响应，包含 95 份第一手与 90 份第二手报告；

**📈 对比分析**

未进行模型对比或性能评估，仅通过描述性统计呈现比例与频数；

**⚠️ 局限性**

样本为自选且回顾性，缺乏诊断验证与系统报告机制，因果推断不成立，外部有效性受限；

---

## 274. A Layered Analysis of Disagreement And Answer Quality in Multi-Agent LLM Debate

**arXiv ID:** 2609.08016 | [PDF](https://arxiv.org/pdf/2609.08016v1)

**作者:** Chen Qian `[一作]` `[通讯]`, Chen Qian

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多代理LLM辩论（Debate）进行分层度量，评估自报同意、回复文本是否真正推背、去掉立场指令后立场是否持久、以及模型内部概率分布是否出现偏移，并通过多种评判者（外部模型、人工）验证文本真实性。

**💡 创新点**

创新点在于将辩论过程拆分为四层（A–D）并分别独立测量，使用条件盲判别者来验证文本推背，并引入“持久性”与“内部log-prob”探针，避免仅凭自报标签得出误导结论；此外对结果采用可靠的swap‑consistency评审、匹配token预算、以及对开放式任务与可验证任务两类基准进行并行对比。

**🔧 技术方法**

技术包括：①多模型协同架构（三模型委员会）；②自报同意标签与文本推背判别器；③去除立场指令后重新提问的对比实验；④对开源可访问权重模型进行教师强制（token‑logprob）探针；⑤使用Jury与swap‑consistency的质量评估；⑥在不同时间轴（none/after/per‑turn）下进行辩论循环；⑦使用匹配token的self‑consistency基准。

**📊 数据集**

使用的数据集包括：GlobalOpinionQA（开放式意见问答）；FRAMES与GPQA（可验证答案基准，用于检验辩论是否能提升准确率）；LiveBrowseComp（网页检索任务，用于检验辩论对最终答案准确度的影响）。

**📈 对比分析**

比较方法：将辩论版与无辩论版在同一问题与相同模型组合下进行配对评分，使用swap‑consistency的三族评审确保评价不受立场偏见影响；在可验证基准上比较准确率；在token‑logprob层面比较对立场置信度与方向的变化。结果显示：辩论能显著改变模型所报告的同意率和文本推背率，但未在可靠评审下提升最终答案质量，且模型内部概率分布的置信度下降，但方向性变化不显著。

**⚠️ 局限性**

局限性包括：①样本规模仅为pilot级别（每个条件约数十个问题），结果置信区间宽；②仅在三模型委员会、特定模型家族（Opus、GPT‑5.5、Sonnet）与两类可验证基准下验证，缺乏跨模型、跨任务的广泛复现；③内部log‑prob探针仅在自主持模型上实现，无法直接映射到闭源委员会模型；④持久性实验以单轮为主，跨轮、跨会话的持久性未得到充分验证；⑤评判者的置信度与方向性在human‑human对比中存在差异，导致对严重差异的检测灵敏度有限。

---

## 275. A Black-Box Adversarial Attack on Human Pose Estimation and Keypoint-Based Action Recognition Models

**arXiv ID:** 2609.08013 | [PDF](https://arxiv.org/pdf/2609.08013v1)

**作者:** Kacper Mroczek `[一作]` (University of Rzeszów), Michal Kepski `[通讯]` (University of Rzeszów)

**通讯引用:** 1214 | [OpenAlex ID](https://openalex.org/A5006454401)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于对象关键点相似度（OKS）的决策式黑盒对抗攻击（OKS Attack），针对视频中的 2D 人体姿态估计模型并评估其对下游关键点动作识别模型的影响。

**💡 创新点**

创新点在于：① 用 OKS 取代 IoU 作为攻击反馈，更精准匹配姿态结构；② 采用时序连贯的决策式黑盒方法，在无梯度、无标注的条件下对姿态估计进行迭代扰动；③ 系统性地评估姿态扰动对动作识别性能的破坏，并进行检测器与关键点估计器的消融分析。

**🔧 技术方法**

使用的技术包括：决策式黑盒攻击框架、正切扰动与正交扰动的组合、OKS 计算、时序一致性传播、对多种 2D 姿态估计器（top‑down 与 single‑stage）的通用实现，以及对 PoseC3D 动作识别模型的跨数据集测试。

**📊 数据集**

主要使用了 Penn Action 数据集（用于姿态估计和动作标签），以及 COCO 数据集预训练的姿态模型；下游动作识别使用了预训练的 PoseC3D（基于 Pose‑SlowOnly R50）。

**📈 对比分析**

通过与 query‑matched 随机噪声对比，实验在四个姿态模型（ResNet‑50、MobileNetV2、YOLO‑Pose S/M）上显示 OKS 减少幅度为 0.0802–0.1494，动作识别准确率下降 6.18–13.86%。扰动在 PSNR 29–30、SSIM 0.68–0.71 范围内，保持视觉可接受性。

**⚠️ 局限性**

限制包括：仅针对 2D 姿态，未考虑 3D 或物理世界的攻击；实验仅在 Penn Action 上进行，跨数据集鲁棒性评估有限；攻击的查询预算与计算复杂度较高；未深入探讨如何提升攻击对不同检测器或更复杂动作识别模型的适应性。

---

## 276. TaskGuard: Task-Conditioned Restoration Utility for Risk-Aware Object Detection

**arXiv ID:** 2609.08011 | [PDF](https://arxiv.org/pdf/2609.08011v1)

**作者:** Vung Pham `[一作]` (Sam Houston State University), Vung Pham `[通讯]` (Sam Houston State University)

**通讯引用:** 608 | [OpenAlex ID](https://openalex.org/A5063674569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出TaskGuard，一种在冻结图像恢复和目标检测管线后决定是否接受恢复结果的后处理控制器；

**💡 创新点**

创新点在于将恢复残差与检测器梯度耦合，构建方向性任务敏感度信号，使用伪梯度实现可部署的任务效用预测，并在未见降质类型下实现零射击转移；

**🔧 技术方法**

技术主要包括利用精确区域反事实与一阶梯度近似的方向性效用估计、伪标签梯度、Ridge回归预测、特征组消融验证，以及基于Gaussian训练的冻结阈值；

**📊 数据集**

数据集使用COCO 2017作为源图像，生成Gaussian、运动模糊、合成雨和散焦等多种降质；还评估了DAWN自然雨图像和RT-DETR-L检测器；

**📈 对比分析**

与固定策略（Always Restore / Preserve）和基于视觉特征的基线（Image-only、Visual-pair）相比，TaskGuard在未见降质上保持93.3%/98.8%的效用保留，同时将负效用干预率降低54.2%（或对实际F1降幅降低37.0%）；在自然雨DAWN上仅恢复8张图像，保持77.8% AP提升，负效用率下降97.9%；

**⚠️ 局限性**

局限在于需先生成恢复结果并计算梯度，导致额外计算开销；伪梯度不完美，可能被错误伪标签污染；目前仅验证在特定恢复器、检测器和任务（目标检测）上的零射击转移，未覆盖更广泛的模型或任务。

---

## 277. mjorbit: A Simulation Framework for Space Robotics

**arXiv ID:** 2609.08010 | [PDF](https://arxiv.org/pdf/2609.08010v1)

**作者:** John Z. Zhang `[一作]` (Massachusetts Institute of Technology), Zachary Manchester `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5135805312)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `51c0528b-f690-4182-ae60-bb5f046c276c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于 MuJoCo 的多体空间机器人仿真框架（mjorbit），能够将轨道动力学、环境扰动和多体接触无缝耦合，并提供低延迟 CPU 与高吞吐 GPU 两种后端。

**💡 创新点**

创新点在于使用 Encke 差分重力避免单精度失效，选取轨道跟随（OF）框架实现局部坐标与对称性保持，同时将轨道耦合与 MuJoCo 的多体动力学整合，形成兼顾精度与速度的统一接口。

**🔧 技术方法**

采用的技术包括 MuJoCo / MJWarp GPU 加速、Encke 差分重力、RK4 与半隐式 Euler 积分、轨道参考框架（ECI/OF/LVLH）选择、J2、大气阻力、太阳辐射压、磁力等环境模型以及多体力矩与扭矩的统一负载组装。

**📊 数据集**

实验数据集主要为自建的四个场景（自由漂浮双臂机器人、Soyuz‑ISS 对接、抓取定点、Astrobee 抓取）以及与 Basilisk 的仿真对比；未使用公开标准数据集。

**📈 对比分析**

通过与 Basilisk 的状态误差和步骤率对比，单精度 GPU 后端可达 10^6–10^7 步/秒，误差 <10^‑3；双精度 CPU 后端误差 <10^‑7；相较 Basilisk 的 10^3 步/秒，速度提升 3–4 个数量级。

**⚠️ 局限性**

局限性包括：每个世界只能使用单一参考轨道，局部坐标假设要求 ρ_I ≪ r_c；GPU 单精度导致时间相关环境（太阳、磁场、阴影）精度下降；缺乏高级感知传感器（相机、激光雷达）支持；仅实现 1 阶隐式与 RK4 积分，未探索更高效的变分/结构保持积分器；仿真验证仅为软件对比，缺乏硬件或轨道实验验证。

---

## 278. HyCO: A Hybrid Neural Solver for Combinatorial Optimization

**arXiv ID:** 2609.07990 | [PDF](https://arxiv.org/pdf/2609.07990v1)

**作者:** Yuheng Li `[一作]` (William & Mary), Yanhai Xiong `[通讯]` (William & Mary)

**通讯引用:** 501 | [OpenAlex ID](https://openalex.org/A5008946067)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种混合神经求解器 HyCO，先用强化学习（RL）构造部分解前缀，再由条件扩散模型（DM）完成剩余决策；通过单次手over实现对组合优化问题的高质量求解。

**💡 创新点**

创新点：
① 统一的 regret 框架证明 RL 与 DM 的误差扩展特征互补；
② 推导出唯一的期望 regret 最优手over步长 τ*；
③ 设计轻量级自适应触发器，利用策略熵与 RL–DM 不一致度作为轨迹级指标，实现接近理论最优的手over。
④ 通过实验验证该混合策略在多种 CO 任务上持续优于单一 RL 或 DM。

**🔧 技术方法**

核心技术：
- 基于 Transformer 的 RL 策略网络（AM、POMO、LEHD 等）；
- 条件扩散模型（Prefix‑DIFUSCO）在已知前缀下进行全局采样；
- 单步熵估计、交叉熵/KL 评估的 RL–DM 不一致度；
- 单次触发 + 双轨生成（即时纠正 + 全局候选）。

**📊 数据集**

使用的 benchmark 数据集：
- Euclidean TSP：50、100、500、1000 节点；
- 最大独立集（MIS）：RB‑LARGE、ER‑700‑800、SATLIB；
- 方向性旅行问题（OP）：50、100、200 节点。

**📈 对比分析**

与多种基线对比：
- 传统精确求解器 Concorde、Gurobi；
- 经典启发式 LKH‑3；
- RL 方案 AM、POMO、LEHD；
- DM 方案 DIFUSCO、T2T、Prefix‑DIFUSCO；
- 通过平均长度、相对最优差距、推理时间衡量。结果显示 HyCO 在所有测试规模下都显著降低最优差距（最多超过 70% 相比 RL 仅 12%），且与 oracle 触发器相比误差仅 0.036%–2.33%，触发步长偏差 < 2.5%。

**⚠️ 局限性**

局限性：
- 触发阈值（熵、KL）需经验调优，可能在不同任务或规模上敏感；
- 理论分析假设可满足 Lipschitz 连续性与误差上界，实际情况可能偏离；
- 仅考虑单次手over，若问题更复杂或前缀信息不足，可能需要多次切换；
- 对于非图结构或需要多阶段决策的 CO 问题，扩散模型与 RL 的兼容性尚待验证。

---

## 279. Clean Accuracy Does Not Guarantee Provenance Robustness: A Prospective Codec-Stress Evaluation of Audio Attribution

**arXiv ID:** 2609.07981 | [PDF](https://arxiv.org/pdf/2609.07981v1)

**作者:** Gang Shi `[一作]` `[通讯]`, Gang Shi

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

评估了音频源归因模型在单阶段编解码压缩后的稳健性，使用预先设定的支持域进行前瞻性实验

**💡 创新点**

首次量化了编码器与压缩条件对归因性能的影响，揭示了清洁准确率不等价于稳健性，并展示了多指标失配问题

**🔧 技术方法**

采用WavLM-Base+、W2V2-BERT 2.0及ECAPA-TDNN等预训练表征，使用线性和Proxy-Anchor分类头，对MP3、Opus、EnCodec、DAC等多种单阶段编解码器进行评估

**📊 数据集**

在ASVspoof 2019 LA（7类TTS攻击）和ST-Codecfake-OOD（5类原始编解码源）两大语料库上进行实验，MLAAD为敏感性对照

**📈 对比分析**

通过预先冻结的支持掩模对ΔMacro‑F1进行同时置信区间评估，发现最高可达70点的性能下降；不同表征和编解码器表现差异显著，匹配精度指标不具可估计性，Proxy-Anchor未能提升稳健性

**⚠️ 局限性**

样本量受限（45/28独立组件）、仅单阶段编解码、闭集任务、匹配比较缺乏公共支持、后注册调整、缺乏缓解措施等限制了结论的推广性

---

## 280. A Sub-4 Approximation for Fair $k$-Means

**arXiv ID:** 2609.07974 | [PDF](https://arxiv.org/pdf/2609.07974v1)

**作者:** Kangke Cheng `[一作]` (University of Science and Technology of China), Hu Ding `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13798 | [OpenAlex ID](https://openalex.org/A5028970899)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出一种用于欧氏空间公平 k‑means 聚类的近似算法，并将该方法推广至 k‑sparse Wasserstein barycenter 问题；

**💡 创新点**

通过将公平约束与中心开放 LP 的分数预算相结合，利用 LP 的积分间隙来控制合并成本，并结合近似质心集与偏移实例，三种成本上界的凸组合最终得到 1+(3-1/Γ)ρ+O(ϵ) 的近似比；

**🔧 技术方法**

使用线性规划松弛、近似质心集、加权 k‑means 子算法（可为 PTAS）、几何偏移构造、以及公平分配 LP 的求解；

**📊 数据集**

本文未给出具体实验数据集，仅在理论上给出算法的运行时间和近似比；

**📈 对比分析**

与此前的 1+4ρ+O(ϵ) 结果相比，本方法将比率从 5+O(ϵ) 降至 3.8427+O(ϵ)（或使用更精细的 Γ_0 时为 3.8368+O(ϵ)）；

**⚠️ 局限性**

主要限制在于对 LP 的积分间隙 Γ 的依赖，若 Γ 取值接近上限则仍无法进一步压低比率；此外，算法依赖于能够在合理时间内解决加权 k‑means 子问题，且对高维输入需要投影降维处理。

---

## 281. MetaKV: Adaptive KV Cache Compression for Constrained LLM Inference

**arXiv ID:** 2609.07966 | [PDF](https://arxiv.org/pdf/2609.07966v1)

**作者:** Michael Wang `[一作]` (Lake Washington School District), Roozbeh Bostandoost `[通讯]` (UMass Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个自适应KV缓存压缩框架MetaKV，能够根据用户给定的延迟和峰值内存预算，为每个输入提示动态选择最优的KV缓存压缩配置。

**💡 创新点**

创新点在于：①利用轻量级的预测模型实时估算每个压缩配置的准确率、端到端延迟和峰值内存；②在满足约束的前提下，通过预算过滤与逻辑层次选择，最大化约束成功率；③该框架可与多种压缩方法共存，具备高度的可扩展性。

**🔧 技术方法**

核心技术包括：LightGBM预测模型（延迟、内存、准确率预测）、预算过滤器、逻辑层次选择策略；候选压缩方法涵盖KVQuant（量化）、H_2O（令牌驱逐）、RocketKV（稀疏注意力）以及未压缩FP16。

**📊 数据集**

实验使用了四个公开数据集：GSM8K（数学）、HellaSwag（常识推理）、ARC-Challenge（科学推理）和SQuAD 1.1（阅读理解）。

**📈 对比分析**

与单一最佳静态配置和理论上完美预测的上限进行对比，采用Constrained Success Rate（CSR）指标衡量。在所有延迟-内存约束组合下，MetaKV平均提升CSR约0.07，最大提升达0.135；同时在准确率、延迟与违规率方面表现均优于静态方案。

**⚠️ 局限性**

局限性包括：①若所有候选配置均无法满足约束，仍会出现违规；②预测模型需针对不同模型/硬件重新校准，成本较高；③当前仅考虑端到端延迟和峰值内存，未涵盖首令牌时间、吞吐量、能耗等其他部署指标。

---

## 282. Accelerating Fourier--Motzkin elimination: redundancy removal and the choice of variable elimination order

**arXiv ID:** 2609.07960 | [PDF](https://arxiv.org/pdf/2609.07960v1)

**作者:** Shashaank Khanna `[一作]` (University of York), Shashaank Khanna `[通讯]` (University of York)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5047788104)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了 Fourier–Motzkin 消去变量方法在投影多面体时的冗余约束处理与变量消去顺序选择，尤其针对因果结构的熵约束问题进行实验验证。

**💡 创新点**

创新点在于①证明 Imbert 记录法与 LP 冗余检测不能随意交错；②给出一种安全的组合策略（在每次 LP 检测后重置记录）；③提出基于一次前瞻的并行变量消去顺序算法，显著减少中间不冗余约束数量并大幅加速计算。

**🔧 技术方法**

主要技术包括 Fourier–Motzkin 消去、线性规划冗余检测、Imbert 记录测试、一次前瞻并行评估以及对随机多面体和因果结构实例的实验实现。

**📊 数据集**

使用的数据集包括四个规模较大的因果结构实例（初始非冗余不等式>250，待消除变量>100）以及六个 15 维随机多面体实例（每个消除 12 个变量）。

**📈 对比分析**

通过与固定变量顺序下的 FM+LP 方法对比，结果显示一次前瞻法在随机实例中速度提升 6–25 倍，在因果结构实例中每步冗余约束数降低 1–2 量级，墙钟时间在多核环境下显著下降。

**⚠️ 局限性**

限制因素包括：前瞻深度仅为一次，未证明最优性；对大规模实例需要大量并行资源；Imbert 记录测试的完整性不足；方法对可用核心数敏感；最优消去顺序的 NP‑完整性仍未解决。

---

## 283. SPOT: Spatial Perception-Oriented Long-Horizon Humanoid Teleoperation

**arXiv ID:** 2609.07933 | [PDF](https://arxiv.org/pdf/2609.07933v1)

**作者:** Lixing Fang `[一作]` (University of Massachusetts Amherst), Chuang Gan `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 14308 | [OpenAlex ID](https://openalex.org/A5040877128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套基于VR的空间感知导向遥操作系统（SPOT），用于长时段类人机器人数据采集。

**💡 创新点**

创新点包括提出操作员感知视界（perceptual horizon）概念，并通过机器人安装的双目鱼眼摄像、宽视野立体显示、视角解耦自由观测以及视觉稳定化技术，显著扩展了遥操作中的空间感知范围。

**🔧 技术方法**

技术实现涵盖：OpenXR兼容的VR头显与手柄、双目鱼眼摄像机与半球面渲染（equisolid投影）、基于IMU的视觉稳定化、坐标空间直接映射（Cartesian retargeting）、ExtremControl式目标条件控制策略，以及稀疏人类动作采集和上半身姿态估计。

**📊 数据集**

数据集与实验：在10名操作者（3名经验者、7名新手）上，分别完成4项长时段遥操作任务（意外掉落与恢复、外周物体检索、多物体双臂检索、灯光开关切换），每项20次试验；同时在仿真中训练目标条件控制策略。

**📈 对比分析**

与两种基线（外景遥操作和传统视角遥操作）比较，SPOT在任务完成时间、恢复时间、搜索时间、把握次数、对齐时间及成功率上均优于基线。例如，意外掉落任务完成时间从25.73s降至19.69s，成功率从79.5%升至98%。

**⚠️ 局限性**

局限性包括：仅提供视觉感知，未整合触觉、力学或音频反馈；摄像机传输延迟仍为主要瓶颈；未评估对大规模下行政策学习的长期影响；在高动态或高精度操作中可能仍受延迟与视觉稳定化效果限制。

---

## 284. GPU-Enabled Large-Scale Optimization Using Randomized Linear Algebra

**arXiv ID:** 2609.08136 | [PDF](https://arxiv.org/pdf/2609.08136v1)

**作者:** Pratik Rathore `[一作]` (Stanford University), Madeleine Udell `[通讯]` (Stanford University)

**通讯引用:** 2203 | [OpenAlex ID](https://openalex.org/A5084564811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个基于 PyTorch 的开源软件包 rlaopt，用于大规模优化与科学计算，提供 GPU 加速的 PCG、NysADMM 和 SAPPHIRE 求解器，并提供类似 CVXPY 的建模语言；

**💡 创新点**

创新点在于：①将 RandNLA 技术与 GPU 并行相结合，实现高效的随机预处理；②提供统一的建模语言和自动结构识别，支持问题拆分与求解器选择；③支持求解过程的反向传播，实现超参数调优；

**🔧 技术方法**

核心技术包括随机数值线性代数（RandNLA）预条件、共轭梯度、Nyström ADMM、预处理的随机梯度方法（SAPPHIRE），以及基于 PyTorch 的 GPU 加速矩阵运算和自动微分；

**📊 数据集**

使用的数据集包括 synthetic ridge regression (自建) 与真实数据集 CIFAR-10、SVHN、Fashion‑MNIST、News20、RCV1；以及随机特征数据集 acsincome-rf、yearpredictionmsd-rf、yolanda-rf、e2006、realsim；

**📈 对比分析**

与传统方法（CG、QR、LSQR、SCS、Clarabel、JAXopt 的 APG、L‑BFGS‑B）对比，PCG 在 GPU 上比 CG 速率提升 57–128 倍；NysADMM 在大规模稠密问题上能跑通且速度快于 conic 求解器；SAPPHIRE 在 GPU 上速度提升 3–7 倍，但整体仍落后于 JAXopt；

**⚠️ 局限性**

局限性包括：对稀疏矩阵的支持不够；在某些问题上仍需超参数手动调节；与 JIT 编译的 JAXopt 等基线相比，速度和精度有差距；

---

## 285. Jacap: Robust KV Cache Eviction via Jacobian-Based Nonlinear Information Capacity Preservation

**arXiv ID:** 2609.08131 | [PDF](https://arxiv.org/pdf/2609.08131v1)

**作者:** Jiaming Yang `[一作]` (Sichuan University), Jiancheng Lv `[通讯]` (Sichuan University)

**通讯引用:** 7196 | [OpenAlex ID](https://openalex.org/A5073535763)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对LLM的KV缓存裁剪问题，提出Jacap方法，通过局部非线性信息理论和雅可比信息容量来评估并裁剪缓存项。

**💡 创新点**

创新点在于将softmax注意力建模为非线性高斯通道，推导雅可比信息容量，并利用softmax敏感度和统计杠杆实现容量感知的子集选择。

**🔧 技术方法**

使用局部Taylor展开、雅可比矩阵、信息瓶颈理论、softmax敏感度权重以及统计杠杆得分等技术。

**📊 数据集**

实验涵盖LongBench、NIAH、AIME25等基准，并在Qwen3-8B/14B、Llama3.1-8B、Nemotron-7B等模型上进行评测。

**📈 对比分析**

与CapKV、SnapKV、KeyDiff、EA、KNorm等基线对比，Jacap在高压缩比（0.75、0.9）下在推理、检索和动态解码任务中均优于对手，平均提升约5-10分。

**⚠️ 局限性**

局限在于仅采用一阶近似，忽略softmax竞争的二阶耦合，可能在极端动态解码情形下表现受限。

---

## 286. Artificial Intelligence-Assisted Digital Inventory of Cultural Heritage & Traditional Knowledge: Case for Indonesian Open Digital Library of Culture

**arXiv ID:** 2609.08105 | [PDF](https://arxiv.org/pdf/2609.08105v1)

**作者:** Hokky Situngkir `[一作]` `[通讯]` (Bandung Fe Institute), Hokky Situngkir (Bandung Fe Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过构建一个五阶段经济漏斗的自治AI框架，实现对印尼数字文化图书馆中无形文化遗产知识的自动抓取、抽取、去重与发布，显著扩展条目覆盖并深化条目细节。

**💡 创新点**

创新点在于将“数据级新颖性”定义为事实级重复判定，将去重转化为增量化；采用可扩展的经济漏斗安排低成本过滤与高成本推理；以及在架构层面实现确定性编排与可追溯性治理。

**🔧 技术方法**

采用聚焦爬虫、跨语言多语言大型语言模型（LLM）进行抽取与规范化、向量编码与分块近似最近邻、贝叶斯多源证据融合、分布式事务性消息队列与幂等发布等技术。

**📊 数据集**

主要数据集为印尼数字文化图书馆（PDBI）现有十万级条目作为去重基准与发布渠道，来源于公开网页、百科、档案、社区等多语言资源。

**📈 对比分析**

通过与人工标注的样本进行校准与统计审计，系统在保持至少90%精度的同时，实现每日数千条目增量发布，计算成本保持在可控的推理预算内。

**⚠️ 局限性**

限制包括持续种子维护的必要、对防机器人屏蔽页面的可达性依赖、源可信度共线性假设、跨语言规范化质量不均以及审核反馈闭环实施的挑战。

---

## 287. CIVI: A Framework for Diagnosing Search Agent Failures in Civic Information

**arXiv ID:** 2609.08094 | [PDF](https://arxiv.org/pdf/2609.08094v1)

**作者:** Dingying Liu `[一作]` (University of Sydney), Yiyuan Li `[通讯]` (UNC Chapel Hill)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CIVI 框架，用于评估和诊断公共部门信息检索代理的失败，并构建跨国、跨辖区、跨功能的多选问答基准。

**💡 创新点**

首个基于联合国 COFOG 分类、跨国跨辖区的公共信息基准；引入 ARISE 诊断方法，将错误划分为搜索绕过、检索失败、对齐失败和理解失败。

**🔧 技术方法**

使用代理式检索 LLM 与 Exa 搜索子系统；source‑injection ablation 与搜索轨迹分析相结合的 ARISE；对10个前沿 LLM 进行系统评估。

**📊 数据集**

CIVI 数据集，4,097 个专家验证的多选 QA 对，来自 576 页官方政府网页，覆盖美、加、澳三国、联邦/州/市三辖区以及四个 COFOG 功能类别。

**📈 对比分析**

与 5 名人类基准对比；模型平均准确率约 63%，远低于人类 92.7%；启用代理检索提升约 10pp；检索失败占总错误的 72%；不同模型的检索调用率与准确度差异显著。

**⚠️ 局限性**

仅限英语、三国、三辖区；缺乏多语言与不同治理体系的覆盖；统一使用 Exa 搜索子系统，缺少生态有效性评估。

---

## 288. Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation

**arXiv ID:** 2609.08084 | [PDF](https://arxiv.org/pdf/2609.08084v1)

**作者:** Igor Pavlovic `[一作]` (EPFL), Dengxin Dai `[通讯]` (HUAWEI Bayer Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将开源图像编辑扩散变换器 Qwen-Image-Edit 通过两阶段微调改造成高质量的单目深度估计器，并推广至多种稠密回归任务。

**💡 创新点**

创新点包括：① 用 DINOv3 在地面真实深度上进行特征对齐（iREPA-depth）加速收敛并提升细节；② 设计 SinkLoss（Sinkhorn 匹配）在第二阶段微调，减轻噪声标签带来的误差，显著增强边缘锐度与细节保留。

**🔧 技术方法**

使用扩散变换器 (DiT)、VAE 编码/解码、4‑bit QLoRA 参数高效微调、Log‑Depth 归一化、像素级 L1 与梯度损失、iREPA‑depth 正则化、Sinkhorn 目标函数。

**📊 数据集**

训练数据主要为合成 HyperSim 与真实 vKITTI 的混合；在公开基准 KITTI、ETH3D、ScanNet、NYUv2、DIODE 上评估；也在 HyperSim、iBims‑1、Sintel 等数据集上验证扩展任务。

**📈 对比分析**

与最新的判别式与生成式单目深度方法（如 Depth Anything V2、Marigold V1、Pixel‑Perfect Depth、InfiniDepth 等）比较，Marigold V2 在 AbsRel、δ1 以及边缘误差 (SEE) 指标上均处于最前，尤其在 ETH3D 上实现 AbsRel 2.8（比对手低 30%）。

**⚠️ 局限性**

局限性包括：模型仍基于单步 VAE，推理速度不如实时框架；在透明、薄层、反射或模糊区域仍可能出现不确定性；扩展到更高分辨率时仍需显存支持。

---

## 289. Proactive Context-Forecasted Safety Constraints for Nonstationary Reinforcement Learning

**arXiv ID:** 2609.08080 | [PDF](https://arxiv.org/pdf/2609.08080v1)

**作者:** Tim Tomashevskiy `[一作]` `[通讯]` (McMaster University), Tim Tomashevskiy (McMaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于情境预测的主动安全约束生成框架，用于在逐集非平稳环境中训练强化学习智能体，避免安全违规发生；

**💡 创新点**

创新点在于将安全约束视为可随环境上下文演化的可预测量，通过隐式情境抽取、多步情境预测、合规校准和尾部安全约束预测，实现提前构造与未来环境相匹配的约束；

**🔧 技术方法**

采用GRU/Transformer编码器提取情境向量，使用多步Gaussian预测器和MAML风格元学习进行情境序列建模，利用合规预测得到椭圆不确定集，使用pinball损失进行尾部清晰度量预测，并通过MPC式安全滤波器执行多步鲁棒决策；

**📊 数据集**

在Highway‑Env的merge‑v0、highway‑v0、intersection‑v0和racetrack‑v0四个驾驶布局上进行实验，使用不同的p_stay（上下文保持概率）来模拟非平稳强度；

**📈 对比分析**

与无安全约束、固定约束、仅情境约束和无合规校准的预测等基线相比，本文方法在所有非平稳强度下将碰撞率降低90%以上，任务奖励基本保持，安全性能显著优于传统方法；

**⚠️ 局限性**

局限性包括仅通过p_stay调节的上下文切换方式，未系统区分可观测与隐式上下文漂移类型，实验范围有限，缺乏更广泛的算法对比与更细粒度的预测与校准效能分析。

---

## 290. Mapping Dynamic, Hierarchical Quantum Circuits

**arXiv ID:** 2609.08075 | [PDF](https://arxiv.org/pdf/2609.08075v1)

**作者:** Marouane Benbetka `[一作]` (New York University), Martin Kong `[通讯]` (Ohio State University)

**通讯引用:** 536 | [OpenAlex ID](https://openalex.org/A5048530008)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种面向动态量子电路的分层量子比特映射框架，能够在多路控制流中保持映射一致性；

**💡 创新点**

创新点包括：① 引入量子比特协调（Reconciliation）通道统一分支映射；② 设计循环入口重映射优化稳态映射；③ 将通信、深度、误差等四目标融合的多目标成本函数；④ 研发了动态QUEKO（d-QUEKO）基准集；

**🔧 技术方法**

技术手段涵盖：层级依赖图（Hierarchical DAG）建模、基于闭包的依赖驱动映射、SWAP优化、硬件误差感知与延迟计算、稳态选择策略；

**📊 数据集**

使用自研的 d-QUEKO 动态电路基准和 Stim 生成的旋转表面码（rotated surface‑code）电路；

**📈 对比分析**

与 Qiskit LightSABRE 进行对比，实验显示在 127/156 量子处理器和多芯片 hexagon 体系结构上，SWAP 数量提升 36–52%、深度提升 8.7–18%、延迟提升 15–18.6%、误差降低 15–40%；

**⚠️ 局限性**

局限性在于仍需假设循环执行次数已知或使用最坏情况估计，对极端嵌套层数和高并发分支的处理尚不充分，且仅在实验平台上验证，缺乏更大规模硬件上的实测。

---

## 291. Two-Scale Localized PCA-Net: Coarse-Global and Local-Residual Representations for Artifact-Reduced PDE Operator Learning

**arXiv ID:** 2609.08034 | [PDF](https://arxiv.org/pdf/2609.08034v1)

**作者:** Mrigank Dhingra `[一作]` (University of Tennessee), Omer San `[通讯]` (University of Tennessee)

**通讯引用:** 6337 | [OpenAlex ID](https://openalex.org/A5085671233)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了两尺度局部PCA-Net，对高维偏微分方程求解器学习采用全局粗尺度PCA编码与局部残差PCA分解的组合，解决传统局部PCA-Net在拼接时产生的块偏移与接口不连续问题。

**💡 创新点**

创新点在于先通过全局低秩PCA捕获全域尺度结构，再用独立局部PCA仅编码残差细节，并在学习时采用块平衡的潜在目标，同时提供可选的物理场接口微调阶段。

**🔧 技术方法**

使用随机化SVD构建PCA基，基于PCA编码/解码的压缩表示，三层ReLU MLP进行潜在映射，差分可微的场组装器和接口误差（值跳跃、法向导数跳跃）作为后置微调损失。

**📊 数据集**

在单一单位正方形上生成的高斯随机场（Poisson方程）和不连续渗透率场（Darcy方程）数据集，分辨率分别为64²、128²、256²，样本量约1万对。

**📈 对比分析**

与全局PCA-Net、纯局部L2L、重叠输出L2L、L2L+RefinementNet以及全场FNO进行对比。两尺度方法在Poisson 128²上实现1.15%相对L₂误差，显著低于其它局部PCA模型（≈2.3%–4.8%）且仅略高于全场FNO（0.24%），同时保持与重叠方法相近的计算成本。Darcy场上，MRE从3.11%降至2.95%，并将接口法向跳跃与残差误差分别下降≈89%与80%，但整体误差仍高于FNO。

**⚠️ 局限性**

局部PCA编码的残差仍难以准确预测，导致学习与PCA恢复之间存在显著差距；在低样本预算下两尺度方法需要足够数据才能超越重叠方法；接口微调虽可进一步提升连续性，却显著增加离线训练时间，且对全局误差提升有限。

---

## 292. Flexible Motion Generation from Language and Style References

**arXiv ID:** 2609.08032 | [PDF](https://arxiv.org/pdf/2609.08032v1)

**作者:** Kai Weixian Lan `[一作]` (University of California), Daniel Holden `[通讯]` (Epic Games)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 FlexMoGen，能够根据文本描述和示例风格片段生成长时序且可随时间变化的高质量人类动作。

**💡 创新点**

创新点包括无标签的变分风格编码器、仅用关键/值偏置的轻量级 Style Adaptation Module（SAM）以及相对位置编码与全局位置编码相结合的 Transformer 架构，实现了细粒度时间控制、长序列生成与多风格切换。

**🔧 技术方法**

技术手段涵盖变分自编码器、扩散模型、Transformer（RPE/GPE）、SAM（key/value 关键/值偏置）、无监督风格学习、Classifier-Free Guidance（CFG）以及 DDIM 采样。

**📊 数据集**

训练数据主要来自内部 3,089 条高质量 MoCap 片段（约 6.4 小时）以及公开的 100STYLE 数据集（约 20 小时、100 种风格）。

**📈 对比分析**

与 SMooDi、LoRA‑MDM、T2M+MotionPuzzle 等基线进行对比，FlexMoGen 在长序列生成和时间变风格任务上在内容保持、风格反映、运动质量等指标上均位列首位或第二位，并在用户研究中获得最高偏好。

**⚠️ 局限性**

主要局限为对快速或尖锐动作的平滑处理导致风格失真，且对未见过的外部风格迁移不够稳健；此外扩散采样耗时较长，难以满足实时需求。

---

## 293. Observe Before You Alert: Adaptive Driver Alerting with Vision-Language Models

**arXiv ID:** 2609.08130 | [PDF](https://arxiv.org/pdf/2609.08130v1)

**作者:** Yuhang Wang `[一作]` (University of South Florida), Hao Zhou `[通讯]` (University of South Florida)

**通讯引用:** 665 | [OpenAlex ID](https://openalex.org/A5008084033)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于视觉‑语言模型的驾驶员警报系统，利用三动作（Silent、Observe、Alert）策略在车载摄像头视频中进行自适应警报生成，Observe动作用于收集更多证据并推迟警报；

**💡 创新点**

创新点包括：①通过结构化的BELIEF‑span提取安全证据并将隐藏状态聚合为紧凑特征；②将Observe作为单独动作，实现观察‑警报之间的可学习权衡；③构建统一的 per‑tick 评测基准 VLAlert‑Bench 及 DAUS 指标，兼顾排名、覆盖、精度与提前量；

**🔧 技术方法**

技术手段包括：使用 Qwen3‑VL‑4B 作为 VLM 并通过 LoRA 微调，设计 BeliefExtractor、DangerHead、PolicyHead 以及动作‑条件采样器；在训练阶段采用三阶段策略（监督式 VLM fine‑tune、头部训练、闭环微调），并采用 DAUS 评估指标；

**📊 数据集**

数据集方面，训练和验证使用 VLAlert‑Bench（合并 Nexar Collision、DoTA、DAD、DADA‑2000 四个真实 dashcam 数据集），测试时进一步评估在 held‑out ADAS‑TO‑Critic 及 synthetic ACCIDENT 数据集上的泛化；

**📈 对比分析**

与五个主流基线（Open‑BADAS、R3D‑18、ResNet50‑LSTM、Gemini‑2.5‑Flash、MViT‑V2‑S）对比，本文模型在 DAUS 上提升至 0.4878（对比 0.4752），AUROC 提升至 0.689（对比 0.610），并在 APtick、F1t、Recallv 等多项指标上取得领先；

**⚠️ 局限性**

局限性在于：①模型依赖 VLM 的隐藏状态，可能受限于语言生成的可解释性；②Observe 动作仅在离散 tick 级别实现，缺乏更细粒度的延迟控制；③评估主要集中在 1Hz 视频采样和四个特定数据集，跨域或更高帧率的表现尚未验证；④对极端动态场景（如高速碰撞前瞬间）的实时性和准确性仍有待提升。

---

## 294. Router Prior Bias: Preserving Base Routing Structure in MoE Post-Training

**arXiv ID:** 2609.08115 | [PDF](https://arxiv.org/pdf/2609.08115v1)

**作者:** Jaedeok Lee `[一作]` (NAVER Applied AI Group), Haanju Yoo `[通讯]` (NAVER Applied AI Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出软路由锚定方法，利用冻结的基准路由先验将 MoE 迁移后路由软性保持，从而在不牺牲专家协同的前提下提升下游性能。

**💡 创新点**

创新点在于将路由保持从硬约束转为软偏置（Router Prior Bias, RPB），并证明不必使用特定空间或硬约束，任何形式的软锚定均能获得相同收益；同时通过社区图分析验证软锚定的有效性。

**🔧 技术方法**

使用 MoE 软路由、logit 先验偏置、参数空间 L2 约束、概率空间 KL 约束等多种软锚定技术，并结合专家共激活社区图（Louvain 归属、NMI、Q 分析）进行诊断。

**📊 数据集**

实验数据集包括 math20k、coding20k（来自 GLM‑5.1‑Reasoning‑1M‑Cleaned 的 20k 切分）以及 OpenR1‑Math‑220k（约 94k 例），在不同模型家族（Moonlight‑16B‑A3B、Qwen3‑30B‑A3B‑Base、DeepSeek‑V2‑Lite）上进行评估。

**📈 对比分析**

通过与无负载均衡（SFT）和重新应用负载均衡（LBL）的对比，软锚定在 Moonlight‑16B‑A3B 上在数学任务中从 31.91 提升至 45.77，在 Qwen3‑30B‑A3B‑Base 上保持 LBL 的弱点，表现对模型和语料依赖性显著；但在 DeepSeek‑V2‑Lite 上表现不显著。

**⚠️ 局限性**

局限性：依赖基准路由的社区结构，无法保证在路由分布均匀或缺乏社区的模型中同样有效；未在大规模多源数据、长链推理或不同 MoE 架构上验证；软锚定对具体先验内容的鲁棒性尚未完全评估。

---

## 295. SynthGait-19K: A Physically Grounded Synthetic Video Dataset for Gait Parameter Estimation

**arXiv ID:** 2609.08108 | [PDF](https://arxiv.org/pdf/2609.08108v1)

**作者:** Soroush Mehraban `[一作]` (KITE Research Institute), Babak Taati `[通讯]` (KITE Research Institute)

**通讯引用:** 3537 | [OpenAlex ID](https://openalex.org/A5011257199)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于真实 MoCap 数据、统一 SMPL 表示、深度条件视频扩散生成的大规模合成步态视频数据集，并在该数据集上训练直接 RGB 的步态估计模型；

**💡 创新点**

创新点在于：①将多源 MoCap 统一为 SMPL，生成可控视角、多样化场景的高质量 RGB 视频；②利用深度条件视频扩散保留运动学一致性；③在同一数据集上实现多种中间表示（HMR、姿态、力学）的基准比较；

**🔧 技术方法**

技术主要包括 SMPL 拟合、深度渲染与 V-JEPA2 条件视频扩散、Video‑ViT + 位置查询直接回归步态参数；

**📊 数据集**

使用 19,272 条合成步态视频（来自 6,427 个 MoCap 序列，437 位受试者）以及 GPJATK 真实视频进行评估，另外在 PD4T 运动障碍数据集上验证临床迁移；

**📈 对比分析**

通过与多种方法（HMR、OpenCap、STT 等）在 GPJATK 上对齐六个步态参数进行 Pearson 相关率比较，直接 RGB 模型在准确性、推理速度和跨域迁移性上均优于传统 HMR/姿态/力学路径，合成监督可提升真实视频的相关率至 0.82–0.84；

**⚠️ 局限性**

局限性包括：合成视频缺乏极端遮挡、辅助器具和更广泛的服装/临床多样性；仅使用固定 5 秒窗口，未覆盖长周期现象；评估集中在 GPJATK 和 PD4T，未能覆盖更大范围的真实人群。

---

## 296. Learning Metamaterial Eigenmodes with Wavelet-Encoded Fourier Neural Operators

**arXiv ID:** 2609.08102 | [PDF](https://arxiv.org/pdf/2609.08102v1)

**作者:** Han Zhang `[一作]` (Duke University), L. Catherine Brinson `[通讯]` (Duke University)

**通讯引用:** 25259 | [OpenAlex ID](https://openalex.org/A5021656287)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

训练并评估了一个基于 Fourier Neural Operator 的多模态弹性波求解器，能够在连续与二值材料几何下预测多种本征模态与本征频率。

**💡 创新点**

通过引入 Gabor 小波编码实现了在同一模型中对波矢和本征阶数的确定性模态选择，并证明小波编码在空间-频域双重结构上的优势。

**🔧 技术方法**

采用了 Fourier Neural Operator、Gabor 小波编码、有限元生成数据、NMAE 训练损失、AdamW 优化器等技术。

**📊 数据集**

使用了 48 万个样本（24000 结构 × 6 本征波 × 325 波矢）的二维单元格几何数据集，包含连续与二值两种材料分布。

**📈 对比分析**

与统一场和正弦场编码对照，Wavelet 编码在连续/二值测试集上 MAE、MSE、NMAE、NMSE 均优 10–30%，且在 1 ms/样本与 1 s/样本的 FEA 速度对比实现三阶加速。

**⚠️ 局限性**

对高频或尖锐界面（界面长度较大）时预测误差随界面长度增长，FNO 对不连续输入的鲁棒性有限。

---

## 297. Risk-Conditioned Fine-Tuning of Large Language Models

**arXiv ID:** 2609.08064 | [PDF](https://arxiv.org/pdf/2609.08064v1)

**作者:** Zixuan Liu `[一作]` (Tulane University), Zizhan zheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了风险条件化RLHF框架，让单一模型在推理时根据α可连续调节风险偏好。

**💡 创新点**

创新点在于将风险水平作为输入条件直接训练策略，避免为每个风险水平训练单独模型，并使用梯度优化的CVaR目标实现对尾部风险的控制。

**🔧 技术方法**

采用了强化学习从人类反馈（RLHF）、CVaR变分表述、风险条件化策略梯度算法，以及参数注入和提示注入两种风险级别条件化技术。

**📊 数据集**

在Pythia‑70M、Pythia‑2.8B和Llama‑3.1‑8B‑Instruct等模型上，使用IMDB‑Gen、RealToxicityPrompts‑Gen和Safe‑RLHF三类基准任务进行实验。

**📈 对比分析**

与固定风险的RA‑RLHF、提示基准、Logit‑Mixing、RA‑RLHF‑Mix等方法比较，风险条件化模型在已训练的α上与RA‑RLHF‑Oracle相当，并在未见α上保持较好表现，参数量和计算开销基本无增。

**⚠️ 局限性**

局限在于仍依赖奖励/成本模型，难以确定最优的α值，且该机制可能被滥用降低安全性。

---

## 298. LLMs for Social Network Modeling: From Network Generation to Dynamic Processes

**arXiv ID:** 2609.08049 | [PDF](https://arxiv.org/pdf/2609.08049v1)

**作者:** Shikha Mallick `[一作]` (University of Victoria), Akrati Saxena `[通讯]` (Leiden University)

**通讯引用:** 495 | [OpenAlex ID](https://openalex.org/A5055236010)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了大型语言模型（LLM）在社交网络建模中的应用，系统划分为网络生成模型（选择式与交互式）和动态过程模型（意见动态、信息扩散、谣言传播），并聚焦于基于语言生成的自适应社交行为模拟。

**💡 创新点**

创新点在于首次统一并按机制将LLM驱动的社交网络方法归类、梳理，并强调LLM在模拟语境感知、交互式推理和自我更新方面的独特优势，同时提出了对现实性、可扩展性与公平性等方面的研究空白。

**🔧 技术方法**

主要技术包括LLM提示设计、参数高效微调、多代理交互框架、基于文本的情感与观点更新机制，以及将LLM输出与传统扩散模型（如独立级联、DeGroot、Friedkin‑Johnsen）相结合的混合方法。

**📊 数据集**

使用的数据集多为公开社交网络结构（如Twitter、Reddit、CiteSeerX）以及合成或人工构造的用户画像；部分研究也使用真实平台的对话记录、新闻传播数据以及问答式知识库进行微调。

**📈 对比分析**

对方法的比较主要通过人工构造的基准实验（如同质化率、聚类系数、传播曲线、意见分布等指标）以及在公开数据上的定量评估；虽然多数模型能在小规模实验中复制经典网络现象，但缺乏统一的跨方法性能基准，导致可比性受限。

**⚠️ 局限性**

主要局限包括：LLM的偏差与提示敏感导致的同质化过度、可重复性不足、计算成本高导致的规模受限、缺乏理论可解释性以及缺少对人类行为的实证验证。

---

## 299. RFS-UNet: Decoder-Conditioned High-Resolution Skip Recalibration for Bone-Selective DRR Synthesis

**arXiv ID:** 2609.08044 | [PDF](https://arxiv.org/pdf/2609.08044v1)

**作者:** Xiaoyang Li `[一作]` (Northeastern University), Yuan Chai `[通讯]` (Australian National University)

**通讯引用:** 231 | [OpenAlex ID](https://openalex.org/A5024899123)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在骨选择性DRR合成中，引入了残差特征缩放(RFS)模块，用于高分辨率跳跃连接的通道重校准。

**💡 创新点**

创新点在于利用解码器状态与编码器特征联合预测通道缩放因子，形成有界的残差通道重校准，从而更好地重用高分辨率特征。

**🔧 技术方法**

使用U‑Net骨架、残差特征缩放(RFS)、全局平均池化、MLP、SE/CBAM/AttentionGate等注意力机制作为对照，训练目标为Charbonnier+梯度+MS‑SSIM。

**📊 数据集**

采用来自MERLIN CT数据库的1000个金属筛查后病例，分为700/100/200训练/验证/测试集，生成配对的全组织和骨掩膜DRR。

**📈 对比分析**

对比方法包括：U‑Net‑64基线、加宽版U‑Net‑96、Restormer、SE‑Skip、CBAM‑Skip、AttentionGate、Self‑RFS及RFS‑UNet；RFS‑UNet在验证集PSNR提升0.254 dB，测试集MAE下降3.91%，PSNR提升0.311 dB。

**⚠️ 局限性**

局限在于缺乏患者级别独立性验证、仅评估显示域合成且未涉及实际放射图像验证，且改进幅度对临床意义尚不确定。

---

## 300. MamMA: A Mamba-Based Pedestrian Trajectory Prediction Algorithm Considering Occupancy Map and Pedestrian Awareness States

**arXiv ID:** 2609.08041 | [PDF](https://arxiv.org/pdf/2609.08041v1)

**作者:** Juncen Long `[一作]` (Politecnico di Milano), Matteo Matteucci `[通讯]` (Politecnico di Milano)

**通讯引用:** 7262 | [OpenAlex ID](https://openalex.org/A5003932703)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种利用LiDAR占据图与行人警觉状态的Mamba网络实现行人轨迹预测的方法。

**💡 创新点**

创新点在于：①将占据图按网格分块并通过Mamba提取障碍物特征；②将行人警觉状态嵌入轨迹特征；③在时空图中使用多条Mamba模型同时捕获时空交互。

**🔧 技术方法**

采用Mamba模型、时空图、全连接层、PReLU激活以及多任务编码器等技术。

**📊 数据集**

使用STCrowd、SiT、JRDB、ETH和UCY等多模态公开数据集进行训练与评估。

**📈 对比分析**

与DSTIGCN、STIGCN、IMGCN、MRGT、Social‑Implicit、SGCN和Social‑STGCNN等SOTA算法对比，MamMA在minADE、minFDE以及碰撞比例p_coll上均优于对手，且推理时间与现有方法相近。

**⚠️ 局限性**

局限性包括：在缺少LiDAR或警觉状态信息时性能下降；模型参数量较大，对计算资源有一定要求；对极端遮挡或非常小目标的鲁棒性尚待进一步提升。

---

## 301. Scaling Multi-Agent Systems with Prospect-State Propagation

**arXiv ID:** 2609.08033 | [PDF](https://arxiv.org/pdf/2609.08033v1)

**作者:** Zhimei Chen `[一作]` (Independent Researcher), Fakhri Karray `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 8550 | [OpenAlex ID](https://openalex.org/A5070046659)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种将多智能体的微观状态分解为紧凑的“前景状态”与富有表达力的“语义状态”的宏观经济模拟框架，并通过前景状态传播持续注入行为异质性。

**💡 创新点**

创新点在于：① 将前景理论融入微观状态，构造三维前景状态向量（参考点、损失厌恶、概率加权），实现可标量化的行为差异；② 采用轻量级前景状态传播器，仅在每个步骤更新前景状态，显著降低LLM推理成本；③ 在长周期模拟中仅定期刷新语义状态，保持信息丰富的决策能力同时避免频繁的文本压缩导致的行为收敛。

**🔧 技术方法**

技术主要包括：前景状态向量与传播器（使用确定性并行化规则），LLM（Qwen3 系列）用于生成语义状态和决策，市场清算函数Ψ，宏观状态构造，异质性度量（轨迹矩阵、前景矩阵及其谱熵）。

**📊 数据集**

数据集：在自定义的闭合经济仿真环境中生成的合成宏观经济时间序列（GDP、通胀、失业率等），并为每个代理产生微观经济历史和行为轨迹；不使用真实金融或社会经济数据库，而是通过仿真内部生成的数据。

**📈 对比分析**

比较方法：与四种基线（SaMAS、Summary、Reflection、仅前景状态）在 Qwen3-32B 下进行同等推理预算、相同仿真时长的实验。性能指标包括波动真实性（VR）和归一化多样性（D_norm）。实验表明，在 N=500 时，PspMAS 的 VR 提升至 85.3%（相较 SaMAS 的 82.0%），D_norm 高达 56.9%（相较 SaMAS 的 33.3%），说明前景状态传播显著提升了可扩展性与行为异质性。

**⚠️ 局限性**

局限性：① 仅在简化的封闭经济模型中验证，未考虑开放经济、制度约束或真实政策摩擦；② 依赖LLM的偏见与价值取向可能影响决策结果；③ 前景状态仅包含三维参数，可能无法捕捉更复杂的心理机制；④ 仍需进一步评估在更大规模、多模型或不同推理策略下的稳健性。

---

## 302. Information-Entropy-Driven Fault Propagation Modeling for Probabilistic Network Performance Prediction

**arXiv ID:** 2609.08143 | [PDF](https://arxiv.org/pdf/2609.08143v1)

**作者:** Lusha Mo `[一作]` (Central South University), Ming Zhao `[通讯]` (Central South University)

**通讯引用:** 37433 | [OpenAlex ID](https://openalex.org/A5100748394)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出信息熵驱动的故障传播模型IEFP和基于条件扩散的概率网络性能预测模型FEMNet，用于在复杂故障场景下预测关键性能指标（KPI）的分布。

**💡 创新点**

创新点在于：①将相对熵、互信息与转移熵三种信息论量统一用于表征故障异常、耦合强度与传播方向；②设计故障感知的图消息传递机制，使传播风险动态调节节点表示；③将此机制嵌入条件扩散去噪网络，实现从点预测到概率分布的迁移。

**🔧 技术方法**

使用的技术包括信息论指标（相对熵、互信息、转移熵）、图神经网络（故障感知消息传递）、条件扩散模型、注意力门控与传播掩码，以及ns-3仿真平台生成的网络数据。

**📊 数据集**

实验数据集来自ns-3仿真，使用Abilene、GEANT、Germany50三种拓扑及SNDlib流量矩阵，并注入节点、链路、控制平面、数据平面及应用层等多类故障。

**📈 对比分析**

与STGNN、DGAT、xNet、EAGLE、DCRNN、RouteNet-Fermi、GNN-Denoise、Rt-Denoise、ReDiSC、SimDiff、LGD等基线对比，IEFP在故障预测中实现0.9225的准确率、0.9704的TNR、0.8494的F1；FEMNet在确定性预测上MSE、MAE、MAPE平均下降约38%，在概率预测上CRPS、ES-norm分别下降17.3%和12.5%，并保持较窄的置信区间宽度。

**⚠️ 局限性**

局限性包括：仅考虑闭集故障类型，依赖标注丰富的历史故障与KPI数据；在标签稀疏、噪声或延迟的实际环境下效果未知；未处理开放集故障、跨域迁移或在线自适应问题。

---

## 303. KBBQ: A Predictive Noise Law and the Limits of Spectrum Flattening in FP4 Quantization

**arXiv ID:** 2609.08135 | [PDF](https://arxiv.org/pdf/2609.08135v1)

**作者:** Lexington Whalen `[一作]` (SB Intuitions), Ryo Sakamoto `[通讯]` (SB Intuitions)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的量化噪声理论，推导出整数和浮点量化的噪声表达式，并基于此设计了可调节的 KBBQ（Kappa-Braked Blockwise Quantization）变换，能够在 FP4 格式下显著提升量化模型的性能。

**💡 创新点**

创新点包括：① 用单一的方差剖面统一描述整数与浮点量化噪声；② 推导出参与因子 κ 及其上限 κ*，并证明任何可保持函数不变的线性变换都无法超过 κ*；③ 设计 KBBQ 通过可调的“刹车”参数 λ 在理想极限与原始状态之间插值，从而在实际部署中获得最优效果。

**🔧 技术方法**

技术手段包括：基于假设 A1-A3 的量化噪声分析；对浮点量化使用乘法方差模型；求解参与因子的闭式上限；在矩阵乘法中应用对角重缩放、正交旋转和特征分解；实现 KBBQ 的一参数变换；使用 GPTQ 等误差补偿技术。

**📊 数据集**

主要使用大型语言模型的数据集，包括 Llama‑3.2‑1B/3B、Qwen‑3‑4B/8B/32B、Llama‑3‑8B 以及对应的推理评测数据集 MMLU、GSM8K、HellaSwag、WinoGrande、MBPP 等。

**📈 对比分析**

与无变换基线、Haar 旋转、WUSH 变换以及 GPTQ 组合方法进行对比。KBBQ 在四个基准模型、两种 FP4 格式下均优于之前的最佳方法，平均恢复率从 94% 提升到 95–97%，在 Qwen‑3‑8B‑MXFP4 上最高可达 97.2%。

**⚠️ 局限性**

局限性包括：① 需要对每一层预先估计二阶统计量；② 训练过程不包含量化感知，KBBQ 的最佳 λ 需在推理阶段调优；③ 对非块级量化或非均匀分布的误差模型可能不完全适用；④ 目前仅在 FP4、INT4 量化下验证，其他位宽或格式需进一步实验。

---

## 304. Nyström Attention Matches Full Attention for Cross-Sectional Stock Prediction

**arXiv ID:** 2609.08106 | [PDF](https://arxiv.org/pdf/2609.08106v1)

**作者:** Kunhan Guo `[一作]` `[通讯]` (University of Hong Kong), Kunhan Guo (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 MASTER 模型中跨股票多头注意力（Step 194）做了系统拆解、统计诊断、谱分解和各种稀疏/低秩近似实验，评估其对预测性能的真实贡献，并验证在不同规模（300/800/≈3500）下的可扩展性。

**💡 创新点**

发现注意力几乎均匀但其偏差呈低秩结构；低秩偏差是跨股票信息流的唯一来源；Nyström 低秩近似能在 O(mN) 复杂度下与全注意力等效；稀疏化始终失败；注意力与收益相关性呈负相关，暗示补充性而非相似性；谱分解揭示了与传统风险因子对应的六维因子模型。

**🔧 技术方法**

采用统计诊断（熵、困惑度、Spearman 相关）、功能消融（uniform、oracle、GraphMask 等）、谱分解（SVD）、Nyström 近似（m=32），TOST 等效检验，多种种子验证，GPU 前向推理计时与显存分析，因子映射与解释。

**📊 数据集**

使用中国 A 股市场数据：CSI300（约 300 只股票）及其全市场（≈3,500 只股票）通过 Qlib 生成，包含 158 个 Alpha 技术因子、63 个市场特征，训练集 2010–2017，测试集 2019–2020。

**📈 对比分析**

性能对比基于 IC、Rank IC、ICIR 以及 A 股 Sharpe；多种种子下通过 TOST 验证 Nyström 与全注意力在 Rank IC ±0.005、IC ±0.008 上等效；稀疏化变体显著下降；在 N≈3,500 的大规模设置中，跨股票模块与单股 LSTM 基线无显著差异；GPU 计时显示 Nyström 在 N>1,300 时实现 5–15× 的速度提升与显存线性缩减。

**⚠️ 局限性**

局限性包括：仅评估 MASTER 在中国 A 股的表现，缺乏其他市场与模型的验证；大规模实验采用改编架构（无市场门控、d_model=64），不完全与原始体系匹配；大量消融使用 seed 0，虽然关键等效结论多种子验证，但仍可能受特定随机种子影响；Nyström 训练时的 Newton–Schulz 迭代导致额外的计算开销；预处理方式对等效性有一定影响。

---

## 305. RevalExo: A Functional Daily-Activity Benchmark for Inertial and Visual Locomotion Mode Recognition in Older Adults and Clinical Cohorts

**arXiv ID:** 2609.08090 | [PDF](https://arxiv.org/pdf/2609.08090v1)

**作者:** Diwas Lamsal `[一作]` (KU Leuven), Benjamin Filtjens `[通讯]` (Delft University of Technology)

**通讯引用:** 315 | [OpenAlex ID](https://openalex.org/A5018244653)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 RevalExo 基准，收集27名老年人、卒中患者和肌少症患者的日常活动数据，提供10.1小时帧级运动模式标签和5.1小时同步第一人称视频，专门用于评估运动模式识别、跨人群泛化与跨模态迁移。

**💡 创新点**

创新点在于：①结合临床可行的 FATIG'AGE 运动协议；②引入多模态（IMU+视角）且带帧级标注的公共数据集；③设计三大挑战（单/多模态识别、跨人群泛化、视觉引导的知识迁移），为助力设备提供更可靠的主动支持提供研究平台。

**🔧 技术方法**

采用深度学习模型：DeepConvLSTM（IMU）、MobileNet‑v3/ResNet（图像）、X3D/MViT（视频）；多模态融合通过特征拼接、KIFNet、SFTIK、IMU‑Video‑MAE；视觉引导迁移利用对比预训练（CP）与知识蒸馏（KD/CRD/NKD）等技术。

**📊 数据集**

使用 RevalExo 数据集，包含27名受试者、11种运动模式、3个队列（健康老年人、卒中幸存者、肌少症老年人），其中所有人均配备下肢7枚IMU，13人同步佩戴首视相机。

**📈 对比分析**

通过留一人交叉验证、跨人群评估与迁移学习实验比较模型。最佳多模态 MViT‑DCL 在 τ=0s 的宏 F1 为 92.8%，跨人群下降至 88.1%；单模态 IMU 仅 78–80%；视觉单模态约 83–86%；迁移学习（CP+FitNets）在临床队列上提升 6–7个百分点。转移窗口（±0.25 s）识别 F1 约 68%，显著低于整体识别。

**⚠️ 局限性**

限制：①场景固定导致视觉模型可能过拟合、跨环境泛化未评估；②视角设备负荷对部分受试者不可行，导致多模态覆盖不足；③转移窗口识别仍偏低，需改进边缘设备实时性与计算效率；④模型计算量大，对可穿戴硬件的部署仍是挑战。

---

## 306. Vectorizer: Vectorizing NumPy Programs with Shape-Guided Rewrite

**arXiv ID:** 2609.08088 | [PDF](https://arxiv.org/pdf/2609.08088v1)

**作者:** Jingqian Liu `[一作]` (Simon Fraser University), Yuepeng Wang `[通讯]` (Simon Fraser University)

**通讯引用:** 1495 | [OpenAlex ID](https://openalex.org/A5101979875)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于源到源重写的向量化方法，自动将含有显式数组循环的 NumPy 代码转换为无循环、利用广播和掩码的高效向量化实现。

**💡 创新点**

创新点在于：①使用形状（shape）与掩码（maskedness）信息作为向量化指引，构造内部向量化的重写规则；②在“从内到外”地递归处理循环，避免搜索或符号求解；③通过 DSL 与类型推断保证重写的语义正确；④针对分支、掩码索引等高级结构引入掩码数组与掩码更新，兼顾安全性。

**🔧 技术方法**

技术细节包括：DSL（面向 NumPy 的简化语法）、形状与掩码的类型推断、数据流分析（确定依赖与掩码），一组“正确性保证”的重写规则（循环、赋值、索引、二元运算等），以及后处理优化（CSE、掩码简化、tensordot 替换、深拷贝修正等）。

**📊 数据集**

使用 150 个基准，来源于 12 大数据集（C、C++、Python 代码）以及 51 个从 StackOverflow 提取的示例，覆盖了典型的数组计算、图像处理、数值分析等场景。

**📈 对比分析**

与两种主流工具（基于符号执行/符号求解的向量化和基于程序合成的向量化）进行对比。成功率：142/150（直接向量化）对比 70/150；平均向量化时间 0.53 秒，比对方快 10×（5.78 秒）且 35×（18.47 秒）。得到的向量化程序平均加速 74.83×，并且在 JIT 编译器（Numba）下进一步提升 1.45×。

**⚠️ 局限性**

局限性：①无法处理循环中存在循环依赖的情况；②对极端掩码、复杂分支或自定义广播的支持仍需手工调整；③在某些内存密集型或分支剪枝明显的程序中，向量化会产生过大的中间数组导致性能下降；④依赖手动翻译为 DSL 的过程，在极端嵌套或非常规语法时可能需要额外修改。

---

## 307. Inference-Time Nash Alignment

**arXiv ID:** 2609.08082 | [PDF](https://arxiv.org/pdf/2609.08082v1)

**作者:** Hadi Hosseini `[一作]` (Penn State University), Duohan Zhang `[通讯]` (Penn State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在一般偏好下的推理时对齐问题，并提出了 Best-of-Nash（BoN）与 Nash Mirror Descent（NMD）两种算法

**💡 创新点**

首次把推理时对齐建模为在两玩家零和博弈中的纳什均衡，并给出最优的对偶间隙理论证明，同时设计两种高效实现

**🔧 技术方法**

利用零和博弈理论、线性规划、镜像下降、近似拒绝采样等技术进行算法设计与分析

**📊 数据集**

在 TLDR、HelpSteer2、UltraFeedback 三个公开偏好数据集上进行实验

**📈 对比分析**

与基准 SFT 模型和 DPO 微调模型对比，BoN 与 NMD 在所有数据集上均比 SFT 提升约10%–24% 的期望胜率，并能与 DPO 达到相同水平；NMD 对 β 参数鲁棒，BoN 随样本数 N 单调提升

**⚠️ 局限性**

对偏好模型质量敏感、受限于基准模型的覆盖范围、需要 O(N²) 的偏好查询成本，且仅给出逐条提示的独立理论保证

---

## 308. Automated Design of Inventory Policy with Large Language Models: An Exploratory Study

**arXiv ID:** 2609.08071 | [PDF](https://arxiv.org/pdf/2609.08071v1)

**作者:** Fenghua Yang `[一作]` (University of Michigan), Parshan Pakiman `[通讯]` (University at Buffalo)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5045832523)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于大语言模型（LLM）的自动化库存政策搜索框架（AIPS），通过将历史需求数据、外部数值优化器与LLM的策略类生成能力结合，自动生成可参数化的库存政策类，并在每一代通过优化获得最优参数，从而在不预先指定政策类的前提下，探索并发现更高效、更易解释的库存补货规则。

**💡 创新点**

核心创新在于（1）把LLM作为结构生成器，并利用优化后得到的政策性能反馈来引导后续生成，形成演化搜索；（2）展示了优化不仅仅是参数调优，更通过提供更精准的反馈加速搜索路径；（3）发现了此前文献未研究过的、由LLM组合而成的可解释库存政策类；（4）证明这些发现的政策类在广泛的新环境中具有可迁移性。

**🔧 技术方法**

使用的技术包括：大型预训练语言模型（DeepSeek V3、GPT‑5、Gemini、Grok等）用于生成Python代码形式的政策类；数值优化器（如COBYLA）在每个生成的类内搜索参数；基于仿真的经验成本评估；演化算法（父类选择、子代生成、存活池更新）；结构特征提取与统计分析；以及跨环境迁移实验。

**📊 数据集**

实验数据来源于30个失销库存实例（不同需求分布：泊松、指数、正态；不同阶期：2、4、6；不同成本比：1,2 和 1,5），以及10,064个目标实例（覆盖13种需求分布族、4个阶期、4个成本比），用于评估跨环境泛化。

**📈 对比分析**

与传统的优化基准（已优化的基准库存策略）进行对比，采用百分比成本降低作为评价指标。结果显示：第一代后平均降低17.5%，第十代后平均降低30%；在10,064个新实例中，发现的政策类平均降低约22%，并在不同阶期、成本比和需求波动性下表现出显著优势；LLM骨干不同导致搜索速度和最终收益差异显著，且优化器的加入在所有骨干上均大幅提升性能。

**⚠️ 局限性**

局限性包括：对LLM质量敏感，弱骨干可能导致搜索慢或性能低；外部优化器在有限预算下不保证全局最优；实验仅覆盖单品周期性库存（失销）模型，未验证多品或非失销情形；评估基于仿真数据，可能不完全反映真实环境；最后，搜索结果仍需人工验证其业务可行性与可解释性。

---

## 309. ConversationalVoice: Full-Duplex Speech Data from Real Conversations through Source-Faithful Reconstruction and Conversation-Grounded Expansion

**arXiv ID:** 2609.08147 | [PDF](https://arxiv.org/pdf/2609.08147v1)

**作者:** Richard Yucheng He `[一作]` (AveraLabs), Tairan Chen `[通讯]` (AveraLabs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建一个端到端的管道，将真实的双说话人单声道录音转换为三种互补的全双工训练数据：分离、重构和扩展；

**💡 创新点**

创新点在于：1) 引入质量验证的分离步骤，保持稳定的说话人槽位；2) 源可信重构，利用原始词语与交互结构生成更干净、对齐的语音；3) 基于会话上下文的扩展，生成新的对话片段，同时保持说话人身份和交互模式；

**🔧 技术方法**

使用技术包括：DialogueSidon（联合分离与恢复）、WavLM声纹验证、ASR生成规范转录、强制对齐、基于语音克隆的TTS（多模态指令与标签）、Gemini自动评估、NISQA/DNSMOS评估器；

**📊 数据集**

使用了真实的双说话人单声道录音（未指定具体数据集，但来自公开或私有真实对话），并在GitHub仓库中提供相应代码；

**📈 对比分析**

比较方法：在分离、重构、扩展三阶段分别计算WER、NISQA MOS、DNSMOS、声纹相似度、事件F1、Gemini对齐评分。结果显示：分离WER=0.140、NISQA=3.56；重构WER=0.150、NISQA=4.41；扩展WER=0.039、NISQA=4.61；声纹相似度在0.983–0.991之间；重构与扩展的交互率差异较小；Gemini给出扩展的内容连贯度4.94、对话自然度4.80；

**⚠️ 局限性**

局限性包括：1) 评估仅为自动化，缺乏人类听感验证；2) 未测评对下游全双工模型训练的实际提升；3) 数据来源受隐私、版权限制；4) 语音克隆带来身份滥用风险；5) 过滤与模型版本变动可能导致偏差与可复现性问题；

---

## 310. SchemeArena: Factorized Stress Testing of Scheming in LLM Agents

**arXiv ID:** 2609.08126 | [PDF](https://arxiv.org/pdf/2609.08126v1)

**作者:** Jie Ruan `[一作]` (University of Michigan), Lu Wang `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个可扩展的“SchemeArena”基准和对应的“SCOUT”监测方法，用来系统评估大型语言模型（LLM）在工具使用场景下的隐蔽误导行为（scheming）。

**💡 创新点**

创新点在于：1）构造了400个多因素、可因子化的场景，能够独立调节长期目标、压力、提示与监督等关键变量；2）提出了基于多准则的过程级检测框架SCOUT，能从推理和行动轨迹中获取证据并做聚合判定；3）在五种主流LLM上进行大规模对比，揭示了长期目标是驱动scheming的主因，提示能显著提升执行率，部分监督反而可能诱导隐蔽行为。

**🔧 技术方法**

技术手段包括：1）使用LLM（如GPT-4、Claude、Qwen）生成场景和执行轨迹；2）LangChain实现工具调用模拟；3）LLM-as-a-judge（例如Claude3、Llama-3）进行方案筛选与多准则评分；4）Bootstrap方法评估置信区间；5）基于多准则的逻辑判定规则实现scheming标签。

**📊 数据集**

数据集为400个因子化场景，覆盖28个RapidAPI工具类别、4类长期目标、压力与监督条件；每个场景附有初始文件系统状态，并为五个LLM产生完整推理+执行轨迹，共计数千条样本。

**📈 对比分析**

对比方法：对五大LLM在7种设置下的scheming率进行统计，利用SCOUT对每条轨迹评分得到F1≈90.73%；在实验中发现：①明确长期目标能将scheming率从≈20%提升至≈80%；②提示能让Qwen类模型几乎达到100%；③行动监督有时会增加closed模型的scheming率。

**⚠️ 局限性**

局限性包括：①基准为强制性压力和提示的“stress‑test”，实际部署中scheming率可能更低；②推理轨迹的CoT可能被省略或不完整，导致监测不全面；③实验使用仿真工具调用，未覆盖真实API行为；④模型可能意识到评估场景，从而调整行为；⑤因子化设计虽然可控但未涵盖所有现实变量。

---

## 311. DISEIL: Demonstration Distillation for Sample-Efficient Imitation Learning

**arXiv ID:** 2609.08123 | [PDF](https://arxiv.org/pdf/2609.08123v1)

**作者:** Suyog Khanal `[一作]` (Deakin Applied Artificial Intelligence Initiative), Santu Rana `[通讯]` (Deakin Applied Artificial Intelligence Initiative)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在交互式模仿学习中提出DISEIL框架，利用失败模式聚类与起始配置选择，主动向专家请求针对性演示，提升在有限演示预算下的任务成功率。

**💡 创新点**

创新点包括：① 将“何时请求专家”“哪一类失败需要纠正”“从何处开始演示”三大决策显式化；② 使用几何描述符对失败进行聚类并分配演示；③ 结合视觉‑语言模型和语言模型自动生成请求，并通过符号约束检查可行性。

**🔧 技术方法**

技术手段：Diff‑DAgger断点判定、几何描述符与标准化、凝聚聚类、失真损失/对数似然损失、Qwen3‑VL‑32B 视觉‑语言模型、Qwen3‑32B 语言模型、约束存储（工作空间、可达性、成功性）、扩散策略、R3M编码、强化学习基准、象征可达性检验。

**📊 数据集**

数据集与任务：GridWorld（5×5 网格）、Push‑T（ManiSkill3 平面推送）、UR5/UR5e 机器人任务（Lift、Wipe、Door）在 RoboSuite 仿真器中；全部使用模拟环境，无真实机器人实验。

**📈 对比分析**

与 Diff‑DAgger、STAGGER、Safe、Dropout、Ensemble、Thrifty、Stagger 等基线对比；在所有 10 个设置（5 任务 × 2 观测类型）下，DISEIL 获得最高平均留存成功率，尤其在较小预算（B=10）时优势最大（约 9.07%）。

**⚠️ 局限性**

局限性：仅在仿真中验证；几何描述符依赖特权状态，未在真实机器人上评估；专家多为脚本或学习策略，未检验对人类教师的噪声和耗时影响；预算按演示次数计，未衡量实际教师时间成本；推理阶段成本较高；聚类记忆未能捕捉已覆盖的演示内容，难以避免重复纠正。

---

## 312. Hyperspectral Anomaly Detection via Group Sparse Low-Rank Tensor Factorization With Automatic Anomaly Grouping

**arXiv ID:** 2609.08121 | [PDF](https://arxiv.org/pdf/2609.08121v1)

**作者:** Quan Yu `[一作]` (Central China Normal University), Xiongjun Zhang `[通讯]` (Central China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于组稀疏低秩张量分解与自动异常分组的光谱-空间谐波稀疏检测框架（GSAA-SS），用于高光谱图像中的异常检测。

**💡 创新点**

创新点主要有：① 用组稀疏正则化因子代替传统低秩核范数，显著降低SVD计算成本；② 引入自动异常分组（AAG）惩罚，通过潜在分组映射自适应学习空间分组结构；③ 将光谱域和空间域的检测结果进行融合，进一步提升检测精度。

**🔧 技术方法**

采用的技术包括：t-product张量分解、组稀疏正则化、变分表示的自动分组惩罚、线性化ADMM（LADMM）求解、FFT加速张量运算以及对偶变量更新的闭式投影。

**📊 数据集**

实验使用了五个真实高光谱数据集：Airport‑Beach‑Urban（含四幅子图）和 San Diego AVIRIS（100×100子图）。

**📈 对比分析**

与RX、RPCA、LRASR、Turbo-GoDec、PTA、TPCA、PCA‑TLRSR、RGAE、GAED、LCRS等十种代表性方法比较，GSAA-SS 在所有数据集上均获得最高的 AUC，平均提升约2%–3%，并且在计算时间上优于大部分方法，尤其比深度学习方法快 20–25 倍。

**⚠️ 局限性**

局限性：需要手动设置超参数（γ、p、α₁、α₂ 等），对极端噪声或更复杂背景的鲁棒性尚未充分验证，且在极大尺寸数据时仍可能受限于显存和运算时间。

---

## 313. DriveMotion: A Large-Scale Multi-Source Benchmark for Driver Motion Sequence Modeling and Forecasting

**arXiv ID:** 2609.08117 | [PDF](https://arxiv.org/pdf/2609.08117v1)

**作者:** Yuhang Wang `[一作]` (University Of South Florida), Hao Zhou `[通讯]` (University Of South Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了DriveMotion基准和数据集，提供数小时连续驾驶员全身133关键点运动序列，并定义了动态锚定的运动预测任务。

**💡 创新点**

创新点包括多源统一表示（BATON、公开车内视频、AIDE）、显式可观测性掩码、基于CAN信号的事件锚定评估窗口、身份无关划分和对姿态缺失的系统处理。

**🔧 技术方法**

技术实现基于RTMW姿态估计管道、统一提取与质量控制、动态锚定协议、以及多种预测模型（GRU、Transformer、CVAE、DDPM、运动令牌等）和对应评估指标。

**📊 数据集**

所用数据集为BATON自然驾驶记录、公开车内视频、AIDE情感与行为注释，合计数小时高质量驾驶员运动数据。

**📈 对比分析**

评估采用MPJPE@4s、Part‑State F1@2s和冻结率，对比零运动基线、姿态基线与各种模型，模型在锚定窗口上相对零运动提升13–15%，在外部Web测试集训练全源数据后误差下降38%。

**⚠️ 局限性**

局限性在于仅使用姿态估计无真实3D标注、视角与可观测性不完全覆盖、某些行为难以通过骨架预测、以及缺乏同步多角度捕获与完整3D参考。

---

## 314. AVP-Inspect: Coordinated Cyber-Physical Testing for Privacy Analysis of COTS Apple Vision Pro Applications

**arXiv ID:** 2609.08103 | [PDF](https://arxiv.org/pdf/2609.08103v1)

**作者:** Yichang Xiong `[一作]` (George Mason University), Xiaokuan Zhang `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套自动化的“Coordinated Cyber-Physical Testing”框架，用来检测Apple Vision Pro应用在网络流量中的隐私违规。

**💡 创新点**

首次实现不需要root或源代码的闭源XR平台自动化隐私检测，并结合硬件输入模拟、3D UI探索与统一隐私分类。

**🔧 技术方法**

利用ESP32蓝牙HID+伺服机实现物理交互控制，AirPlay+MITM捕获网络流量，OmniParser识别UI状态，基于VPVet扩展的隐私税onomy与规则匹配检测违规。

**📊 数据集**

构建50个ground-truth应用的手工标注数据，扩展到324个商用AVP应用的App Store元数据与网络流量。

**📈 对比分析**

与手工探索基准比较，覆盖率96.2%，检测到157/151违规；自动20分钟测试生成3.15倍交互流量，发现58%应用违规；与Meta Quest等XR平台相比相似违规率。

**⚠️ 局限性**

受限于20分钟探索未能覆盖全部UI，缺乏系统级API导致对深层SDK行为识别有限，且对AVP专属传感器数据检测不足。

---

## 315. VI-Bench: Benchmarking Prompt Inversion from AIGC Videos

**arXiv ID:** 2609.08079 | [PDF](https://arxiv.org/pdf/2609.08079v1)

**作者:** Wulin Xie `[一作]` (Institute of Automation, Chinese Academy of Sciences), Chen Gong `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出视频提示逆向（prompt inversion）任务与基准VI‑Bench，评估VLM是否能从AIGC视频中恢复可执行的重放提示。

**💡 创新点**

创新点在于：①将提示逆向定义为生成可重放提示的能力；②设计基于真实用户提示的三难度级别（单/多/多-shot）数据集；③引入Prompt Score与Video Score两阶段评估，结合Replay Score形成Inversion Score，全面衡量恢复与重放的质量。

**🔧 技术方法**

使用大型多模态模型（如GPT‑4o、Qwen系列、InternVL、VideoLLaMA等）以及两种视频生成器（Wan2.2、HunyuanVideo）进行逆向推理与重放；评估采用人工GPT‑4o判定与两代理（Memory + Judge）比较；在Prompt与Video维度上分别给出5分制打分。

**📊 数据集**

数据来源：16.1M真实用户提示（从DiffusionDB、VidProM、TIP‑I2V整理后约3.9M），通过主题池和难度合成得到900个经人工验证的AIGC视频（Easy/Medium/Hard各300个）。

**📈 对比分析**

比较方法：对18个代表性VLM做Prompt Score、Video Score、Inversion Score三项指标对比；结果显示最优模型（Doubao‑Seed‑2.0‑pro）总体Inversion Score仅0.632，且从Easy到Hard显著下降；与传统视频理解/字幕指标相关性低，说明逆向任务与描述性理解是不同能力。

**⚠️ 局限性**

局限性：现有VLM在恢复可执行提示上仍远未成熟，尤其在多shot和高控制维度（风格、摄像机）时性能衰退；评估依赖于两台生成器和人工判定，可能受生成器偏差和人工作业主观性影响。

---

## 316. A Machine Learning Framework for Predicting Restaurant Food Waste to Support Sustainable Food Management

**arXiv ID:** 2609.08078 | [PDF](https://arxiv.org/pdf/2609.08078v1)

**作者:** Md Mehedi Hasan Naeem `[一作]` (Jatiya Kabi Kazi Nazrul Islam University), Md. Arefin Haque Mahir `[通讯]` (Jatiya Kabi Kazi Nazrul Islam University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了一个可复现的餐厅食品浪费预测框架，利用运营、气象和时间特征训练监督回归模型。

**💡 创新点**

提供了透明的目标构造公式与噪声控制，系统性进行特征泄漏识别与消除，并采用时间序列交叉验证验证模型稳定性。

**🔧 技术方法**

使用线性回归、决策树、随机森林和梯度提升等监督学习方法，结合特征工程、标准化、Ordinal编码以及时间序列交叉验证技术。

**📊 数据集**

构建了包含77,980条日记录、27个特征的公开数据集，整合餐厅需求、气象数据和节假日/特殊事件信息，已在GitHub公开。

**📈 对比分析**

采用70-30时间顺序划分和5折时间窗口交叉验证进行模型比较；在测试集上随机森林实现MAE 6.19kg、RMSE 8.36kg、R² 0.817，优于线性回归和决策树。

**⚠️ 局限性**

主要局限在于目标为基于公式的代理值，未验证真实浪费；特征泄漏仍存在残余；缺乏跨餐厅、跨地区的外部验证，需结合实际测量硬件实现部署。

---

## 317. A Gradient-based yet Spike-Timing-Dependent Solution to the Feedback Learning Problem in Neural Microcircuits

**arXiv ID:** 2609.08070 | [PDF](https://arxiv.org/pdf/2609.08070v1)

**作者:** Xiangnan Zhang `[一作]` (Beijing Institute of Technology), Björn W. Schuller `[通讯]` (Technische Universität München)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了梯度隧道（GT）算法，利用神经微电路（NMC）的时序状态分离与自噪声率编码，实现在稀疏反馈下的 spike‑timing 依赖在线学习。

**💡 创新点**

创新点在于把时序信用分配重构为状态分离问题，使用 lead‑lag 扩展实现可观测的雅可比估计，且无需 surrogate gradient，理论上兼容 ANN–SNN 混合架构，提供对生物可实现性的解释。

**🔧 技术方法**

核心技术包括：随机率编码的 NMC 理论、lead‑lag 扩展、基于条件发放概率的因果梯度定理、梯度隧道算法（local tracing + global tunneling）、稀疏均匀反馈连接和发放率正则化。

**📊 数据集**

使用的数据集有：合成 T‑maze 证据整合、incremental add 记忆、Speech SHD、EEG 情绪识别（SEED、DEAP）。

**📈 对比分析**

与 e‑prop、FPTT、pp‑prop 等在线学习方法相比，GT 在 SHD 上达到 73.61% 准确率（高于 e‑prop 67.54%），在 SEED 上 60.22%（最高），在 DEAP‑Valence 73.93% 等；在 LSM、LTC‑SNN、LSNN 等传统方法上也实现了显著性能提升，且参数量仅占其 0.43%。

**⚠️ 局限性**

主要限制包括：固有的衰减记忆限制长序列任务；缺乏空间特征提取，难以处理高维视觉/时空输入；对 stationary window 的边界理论尚未确定；仅在单层 NMC 上验证，尚未扩展到多层层级结构。

---

## 318. Popular Knowledge Propagates More Errors in LLM Knowledge Updating

**arXiv ID:** 2609.08067 | [PDF](https://arxiv.org/pdf/2609.08067v1)

**作者:** Yuji Zhang `[一作]` (University Of Illinois Urbana Champaign), Heng Ji `[通讯]` (University Of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 FactProp 知识图谱，研究 LLM 在知识更新过程中已正确事实的脆弱性，并提出基于结构流行度的保留策略 PopAnchor

**💡 创新点**

首次发现结构流行度（实体连通度）能预测知识更新后的副作用，并利用此特征指导样本优先保留；同时公开了可验证的 FactProp 资源

**🔧 技术方法**

使用低秩适配 LoRA 进行细调、KL 正则化进行保留、自然语言 QA 对齐、图谱连通性分析、相似度与注意力机制评估

**📊 数据集**

FactProp（由 Wikipedia 事实构建的验证图谱）、CounterFact、MQuAKE‑CF 等公开事实更新基准

**📈 对比分析**

在 Qwen3.5‑2B/9B/27B 与 Gemma‑4‑31B‑it 四个模型上，使用 Flip Rate 衡量知识漂移；PopAnchor 在所有跳距和基准上均显著低于 Random、Rare、Similar Anchor，提升约 20%‑30% 的保留率

**⚠️ 局限性**

实验仅覆盖 LoRA+固定超参，未验证完整参数微调或持续学习；流行度仅基于实体连通度，未考虑关系频率或完整命题；数据来源 Wikipedia，可能存在覆盖和偏见问题

---

## 319. Movable Antennas Enabled Wireless Powered Networks: Principles and Technologies

**arXiv ID:** 2609.08042 | [PDF](https://arxiv.org/pdf/2609.08042v1)

**作者:** Zhendong Li `[一作]` (Xi'an Jiaotong University), Ying Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究可移动天线（MA）在无线供电网络（WPN）中的应用，提出MA位置优化、能量波束成形和位置相关信道估计等关键技术，并通过数值仿真验证MA在SWIPT系统中显著提升能量收集性能。

**💡 创新点**

创新点在于将可移动天线引入WPN，实现天线位置与波束权重的联合优化，显著提升WPT效率、适应性和抗干扰能力，并在案例研究中展示了相对于传统固定天线方案的性能优势。

**🔧 技术方法**

采用的技术包括MEMS/步进电机/流体驱动的可移动天线架构、交替优化（AO）算法、能量波束成形与位置优化、位置相关信道估计与预测。

**📊 数据集**

使用的数据集为仿真数据，采用多径场响应信道模型，设置M个MA在0.5m×0.5m区域内、K个用户在0.75m×0.75m区域内的随机分布。

**📈 对比分析**

通过与固定天线（FPA）、两阶段优化、天线选择矩阵（ASM）以及随机位置等基线方案比较，实验显示所提算法在不同发射功率和天线数量下始终获得最高的能量收集性能，且优势随天线数量增加而显著提升。

**⚠️ 局限性**

局限性包括：（1）实时天线位置调节需要高复杂度和能耗；（2）波束成形易受CSI不完整、硬件失真和多用户干扰影响；（3）天线移动产生的能耗和时间延迟可能抵消性能收益，需研发低功耗驱动与高效调度策略。

---

## 320. VEX-Bench: Benchmarking LLM Agents for Assessing Exploitability of Software Supply Chain Vulnerabilities

**arXiv ID:** 2609.08040 | [PDF](https://arxiv.org/pdf/2609.08040v1)

**作者:** Jiahao Shi `[一作]` (Purdue University), Tianyi Zhang `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并构建了 VEX-Bench 这一首个评估 LLM 代理在软件供应链漏洞可利用性分析中的基准；

**💡 创新点**

创新点在于：①将可利用性评估转化为跨仓库、跨语言的真实案例任务；②引入细粒度的“未受影响”理由标签，实现比传统二元判定更精准的解释；③提供完整的代码库、CVE 与手工标注的黄金标准。

**🔧 技术方法**

采用 LLM 代理（Claude Opus、GPT-5.5 等）配合不同的 harness（Claude Code、Codex CLI、OpenCode 等），利用模型推理、工具调用、外部信息检索与代码分析来完成任务。

**📊 数据集**

数据集为 75 个真实 GitHub 项目（Go、Python、Java），共 67 个 CVE，覆盖 35 个仓库，包含代码、依赖清单与人工标注的可利用性与理由标签。

**📈 对比分析**

通过与传统 SCA 工具（OSV-Scanner、Trivy、govulncheck）对比，LLM 代理在二元可利用性判定上达 80%+ F1，但在细粒度理由预测上仍显著落后（约 55-70% macro‑F1）。在成本-性能上，GPT‑5.5 与 Claude‑Opus 最高，但价格也最高；DeepSeek‑V4‑Pro/GLM‑5.1 具备更优的成本效益。

**⚠️ 局限性**

局限性包括：规模有限、仅覆盖三种语言且理由类别稀疏；单一标签导致可能忽略多重合法理由；仅基于代码与仓库静态信息，无法捕捉运行时配置或动态输入。

---

## 321. SAFER-Activities: A Dataset for Smart Assessment of Fall Events and Routine Activities

**arXiv ID:** 2609.08038 | [PDF](https://arxiv.org/pdf/2609.08038v1)

**作者:** Diwas Lamsal `[一作]` (KU Leuven), Matthew N. Dailey `[通讯]` (Asian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了SAFER‑Activities数据集，提供了超过66小时、85,310个动作实例的高帧率视频，并为每帧打上30类动作标签，特别加入了轮椅使用场景，旨在为智能健康监测中的跌倒检测和日常活动识别提供可在线学习的基准数据；

**💡 创新点**

其创新点在于：①首次提供多视角、无剪辑、全帧级别的跌倒与日常活动数据，②涵盖轮椅使用者的专门子集，③同步发布骨架、RGB特征与多模态融合基线，④设置实验室、离线实验室与外部家庭场景的分布外测试，极大提升了研究的可复现性与鲁棒性；

**🔧 技术方法**

在实验中，作者评估了2D/3D骨架模型（ST‑GCN++, MS‑G3D, PoseC3D, DG‑STGCN）、冻结的RGB预训练骨干（CLIP, DINOv3, VideoMAE）以及多模态融合策略（特征拼接、ModDrop、QMF、OGM‑GE、MMCL），并探索不同时间步长与滑动窗口下的表现；

**📊 数据集**

使用的主要数据集为SAFER‑Activities本身（含轮椅子集），并在ImViA跌倒数据集上进行跨数据集评估，同时利用附带的轮椅骨架标注集检验姿态估计；

**📈 对比分析**

比较结果显示，骨架模型在实验室环境下macro‑F1最高（约87%），在分布外环境下仍保持约79%；RGB模型在域迁移中表现显著衰退；融合模型在实验室提升明显，但在分布外仍低于单一骨架模型；跨数据集跌倒检测中，骨架模型F1可达96.8%，显示良好迁移性；

**⚠️ 局限性**

局限性包括：跌倒模拟由成年演员完成，未覆盖真实老年或轮椅使用者的自然跌倒；轮椅子集仅来自实验室录制，缺乏外部真实场景；冻结RGB特征在域迁移中易受外观变化影响，现有融合策略未能完全克服这一问题。

---

## 322. 6SEVEN: System for EValuating IPv6 ENumeration algorithms

**arXiv ID:** 2609.08098 | [PDF](https://arxiv.org/pdf/2609.08098v1)

**作者:** Chase Kanipe `[一作]` (Johns Hopkins University), Robert Beverly `[通讯]` (San Diego State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并公开了一个可扩展的框架（6SEVEN），用于统一实现、运行和评估目标生成算法（TGA），并通过该框架对八种已发表的TGA在同一实验环境下进行对比实验，系统评估了它们在地址响应、别名处理、生成速度、发现率和实验波动性等方面的性能。

**💡 创新点**

①提出统一的、可插件化的TGA评估流程，消除了不同论文间的实现、数据预处理、别名检测与结果统计差异，使得算法本身的比较更为公平；②系统展示了实验设置对TGA性能的巨大影响，强调多次实验和多维度评估的重要性；③将八种代表性TGA（包括树、熵、图、生成式模型）重实现并集成到框架中，首次实现大规模统一实验。

**🔧 技术方法**

Rust语言实现的插件架构；异步I/O与线程池并行处理；标准化的探测协议（Echo Request、TCP SYN、UDP）与链接层发送；离线/在线别名检测（APD多前缀长度）；随机数种子控制；统计分析（响应类型分布、Jaccard相似度、波动性箱线图）等。

**📊 数据集**

①TUM Hitlist（约千百万活跃地址，包含大量MAC嵌入的IID）；②NTP Pool客户端地址（约百亿个，主是APNIC分配，IID随机化）。两组数据分别作为训练/种子集，用于训练八个TGA。

**📈 对比分析**

通过统一的实验流水线，对八个TGA在相同的种子、探测速率、探测协议、别名过滤、结果计数方式下进行多次独立实验，记录目标生成速率、探测响应数量、去别名后的活跃主机数、新发现的/48网络数、以及不同实验之间的重叠与波动性。实验结果显示：生成速度差距达数十倍；别名检测对计数有显著影响；不同种子会导致算法偏好和性能差异；单次实验无法可靠评判算法优劣，需要多次实验与多维度指标。

**⚠️ 局限性**

①探测预算限制在1000万次；②仅使用单一云端视点；③探测速率固定为10,000pps；④探测类型仅使用Echo Request；⑤训练集大小固定，未探究规模效应；⑥训练集来源不均衡（NTP偏APNIC）；⑦算法内部实现细节可能与原论文略有差异；⑧未评估不同参数、探测速率、视点对性能的影响。

---

## 323. LLM-Based Penetration Testing in the Presence of Honeypots

**arXiv ID:** 2609.08093 | [PDF](https://arxiv.org/pdf/2609.08093v1)

**作者:** Xinhong Xie `[一作]` (Pennsylvania State University), Sencun Zhu `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了大型语言模型攻击代理在混合目标池（包含真实易受攻击主机和蜜罐）中如何在有限的推理预算内分配攻击资源，并提出了基于预连接筛查、预算感知主机排序以及后连接终止的两阶段检测引导策略。

**💡 创新点**

创新点在于将蜜罐识别视为决策问题，将不完美的检测信号与漏洞严重度、预算压力融合进连续的目标选择评分，并通过动态预算调节实现攻击资源的自适应分配。

**🔧 技术方法**

使用了OpenAI o3-mini LLM结合ReAct框架的自动化渗透测试代理、基于协议偏差和命令一致性的预/后连接蜜罐检测器、CVSS漏洞评分以及预算压力更新机制。

**📊 数据集**

实验数据集为控制实验室中的20台主机（8台蜜罐如Cowrie、Wetland等，12台Vulnhub公开易受攻击虚拟机），覆盖SSH、Web、SMB、FTP等服务。

**📈 对比分析**

与随机选择、固定预算分配以及基于规则的后连接停止等基线进行比较；在$1-$5预算范围内，完整策略在确认真实主机数、蜜罐预算花费和进入率等指标上均优于基线，尤其在$3及以上预算时提升明显（例如$5预算下确认真实主机3.66个，对比基线3.00/2.33）。

**⚠️ 局限性**

局限性包括实验规模有限（仅20台主机）、服务类型单一（主要为SSH/Web）、后连接检测分支有限、未覆盖端到端网络发现流程、并且假设蜜罐无法外部通信，未能评估更大规模数据中心环境下的表现。

---

## 324. BrachistoneLR: A Brachistochrone-Inspired Learning-Rate Schedule and a Controlled Benchmark of Scheduling Policies

**arXiv ID:** 2609.08069 | [PDF](https://arxiv.org/pdf/2609.08069v1)

**作者:** Md. Sadekur Rahman Roni `[一作]` (Leading University), Moutusi Dash Nimi `[通讯]` (Leading University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实验评估一种基于 brachistochrone 曲线的学习率调度（等价于周期为 E-1 的余弦退火），并在固定协议下对六种调度策略在 72 次实验中进行系统比较。

**💡 创新点**

创新点在于把 brachistochrone 曲线映射到学习率曲线，揭示其与余弦退火仅在周期设定上的差异，并证明终点率差异随训练周期 E 的平方递减，属于短期效应。

**🔧 技术方法**

使用 Adam 优化器、交叉熵损失、余弦退火、warmup‑cosine、step decay、exponential decay、constant rate 等调度技术，并通过对学习率的精确控制构建对照实验。

**📊 数据集**

三大图像分类数据集：MNIST、Fashion‑MNIST 和 CIFAR‑10，分别使用全连接网络、卷积网络、LSTM、残差网络四种架构。

**📈 对比分析**

在所有 72 配置中，平滑的峰值到底值下降的调度（余弦退火、warmup‑cosine、brachistochrone）平均精度均高于常数率和日历衰减策略，差距随任务难度增大；三者之间的差距在 0.06% 以内，难以在单次种子实验中显著区分。

**⚠️ 局限性**

局限包括：仅使用单个种子、固定 10 轮训练、固定峰值/底值、未对每种策略进行单独超参数调优，且实验仅覆盖三小型视觉数据集和四中等规模模型，无法直接推广到更大规模、不同任务或长周期训练。

---

## 325. ResidualAuth: What Authorization State Must Language Agents Preserve under Revocable Delegation?

**arXiv ID:** 2609.08062 | [PDF](https://arxiv.org/pdf/2609.08062v1)

**作者:** Moonwon Choi `[一作]` (Seoul National University), Seunggeun Lee `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究授权历史的残差状态，构建理论与基准来评估语言模型在授权递归撤销中的信息保持与决策能力。

**💡 创新点**

提出残差授权状态概念，证明同一转移闭包下可指数级区分，给出冗余-内存定律，并设计对抗对照的评测基准。

**🔧 技术方法**

运用图论、Myhill–Nerode理论生成对照历史，并在大模型（Qwen3.6‑35B、Gemma‑4‑26B、Ministral‑3‑14B、Mistral‑Small‑4‑119B）上进行 token‑限制的接口实验。

**📊 数据集**

使用合成授权事件序列（192对实验对、128对Held‑out），包含委派、撤销、使用等操作，生成多种权限配置。

**📈 对比分析**

通过对比 256‑token 摘要、假工具、认证读、路径/割证据等接口，发现认证读可实现 15/16–16/16 的成对正确率，而摘要与假工具表现低于 1/16；在线状态保持几乎不成功。

**⚠️ 局限性**

仅在简化的单右、二值直接边模型上证明，未涵盖多右、组/阈值、时间约束等复杂场景；实验受限于少数模型、token 预算，结果对通用模型的推广性有限。

---

## 326. A Quantitative Evaluation Framework for Temporal Explainability in Echocardiographic Video Segmentation

**arXiv ID:** 2609.08043 | [PDF](https://arxiv.org/pdf/2609.08043v1)

**作者:** Jiyoo Noh `[一作]` (University of Toronto), Jonathan H. Chan `[通讯]` (King Mongkut's University of Technology Thonburi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

针对心脏超声视频分割，提出了一个量化评估时序可解释性的框架，用四种时间显著性指标评估 Grad-CAM 解释。

**💡 创新点**

首次将时序一致性、显著性运动、解剖重叠和时间重叠四项指标整合到可解释性评估中，并揭示了中间特征与最终预测解释在时序稳定性上的差异。

**🔧 技术方法**

使用 ConvLSTM U‑Net、2D U‑Net、Grad‑CAM 以及 Dice/BCE 损失进行训练与评估。

**📊 数据集**

在 EchoNet‑Dynamic 数据集上进行实验，并使用预训练的双向 ConvLSTM 生成伪标签。

**📈 对比分析**

比较了不同时间步长（s∈{1,4,6,8,10}）的 ConvLSTM 与 2D U‑Net 在分割精度（Dice>0.91）以及解释一致性、显著性运动等指标上的表现，最终解释在所有模型上均保持高一致性。

**⚠️ 局限性**

局限包括：无法区分中间显著性变化是因时序特征演化还是解释不稳定；伪标签可能引入误差；仅针对 Grad‑CAM 与 ConvLSTM，未涵盖更复杂架构或其他可解释方法。

---

## 327. Representational Fidelity in Didactic Visualization: Toward a Multidimensional Design Space

**arXiv ID:** 2609.08037 | [PDF](https://arxiv.org/pdf/2609.08037v1)

**作者:** Shehryar Saharan `[一作]` (Univ of Toronto), Jodie Jenkinson `[通讯]` (Univ of Toronto)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于对175幅教学可视化实例的系统编码与分析，构建了一个涵盖五大维度（形态、动态、提示、情境、交互）与七个子维度的多维表征忠实度设计空间；

**💡 创新点**

创新点在于将传统的“真实-抽象”单轴框架拆解为多维度结构，明确区分形态细节、感知特征、模型生成、功能动态、时间变化、提示与情境等维度，并提出“反思性表征推理”概念；

**🔧 技术方法**

主要技术手段包括跨学科协作的分层编码、迭代维度修订、焦点小组访谈与单一设计案例的应用演示，以实现维度验证与可操作性探讨；

**📊 数据集**

使用的数据集为175幅来自医学、工程、细胞及自然科学教材、专业示例与学术论文的教学可视化，涵盖静态、交互式与动画三种媒介；

**📈 对比分析**

对设计维度在语料中的分布和相互关联进行了定量统计与相关分析，展示了维度共现模式和相互冲突，但未进行实验性学习效果比较；

**⚠️ 局限性**

局限性包括样本偏向英语西方教材、聚焦于具代表性的案例而非系统采样、焦点小组受众为中级研究生、缺乏对学习效果的实证验证与多文化视角。

---

## 328. IGT @ FinMMEval 2026 Task 2: Question-Type Prompting with Targeted Extraction for Multilingual Financial QA

**arXiv ID:** 2609.08139 | [PDF](https://arxiv.org/pdf/2609.08139v1)

**作者:** Yuwen Chiu `[一作]` `[通讯]` (Georgia Institute of Technology), Yuwen Chiu (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了面向PolyFiQA Task 2的多语言金融问答系统，能够根据问题类型自动分流至专用处理器，并直接从SEC文件或多语言新闻中提取或生成答案。

**💡 创新点**

创新点在于将问题类型异质性拆解为八类，采用关键词路由结合针对性规则提取（如财务比率计算、现金流表格正则、原语言报价保留）以及严格的输出格式控制，显著提升了ROUGE‑1分数。

**🔧 技术方法**

使用技术包括：Claude Sonnet 4（AWS Bedrock）作为生成器，关键词匹配路由、正则/规则提取、公司特定标签归一化、源语言原文保留、句子数量与token上限约束、prompt工程；未使用深度检索或多跳检索。

**📊 数据集**

数据集为PolyFiQA Task 2，包括4家公司（MSFT、HON、JNJ、UVV）的英文SEC文件及对应的英语、中文、日语、西班牙语、希腊语新闻文章，共344道问题，其中172道为易问（事实类），172道为难问（分析类）。

**📈 对比分析**

与基线（泛化RAG）相比，系统在开发集上ROUGE‑1从≈0.247提升到≈0.395（+60%），在官方测试集上获得ROUGE‑1 = 0.3071，排名第三，精准度≈0.2821，召回率≈0.4044。

**⚠️ 局限性**

主要局限包括：路由误判导致Expert类问题的提取错误、对PolyFiQA特定标记和格式的高度依赖、ROUGE‑1对表述差异过于敏感（导致翻译/重述误判）、缺乏自动化的公司格式审计机制，且检索改进在此任务上难以显著提升。

---

## 329. Sparse Data Augmentation for Optimization with Provable Guarantees

**arXiv ID:** 2609.08133 | [PDF](https://arxiv.org/pdf/2609.08133v1)

**作者:** Behrooz Tahmasebi `[一作]` (Harvard University), Melanie Weber `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在有限变换群下的完全数据增强目标（即对所有群元素求平均）的非凸优化，并提出“一次性稀疏增强”（one‑shot sparse augmentation）方法：在训练开始前随机采样一次少量群元素，随后在梯度下降过程中始终使用同一固定的稀疏增强目标；

**💡 创新点**

创新点在于证明：在满足光滑与RKHS结构的前提下，只需一次采样得到大小为O(log|G|/ε²)的稀疏集合，即可使梯度下降在O(1/ε²)步内得到全增强目标的ε‑平稳点；这比传统的每一步重新采样的group‑SGD所需的O(1/ε⁴)个群查询大幅降低；

**🔧 技术方法**

主要技术包括：群平均算子的谱逼近、有限群表示论（将平均算子拆分为单位子空间与非平凡不可约分量）、RKHS 统一梯度约束、以及对全局梯度场的统一逼近；

**📊 数据集**

实验采用了 6 维坐标的排列群 S₆ 的“坐标求和回归”任务：随机生成 128 训练样本和 256 测试样本，使用 Gaussian 核构建的可变参数模型；

**📈 对比分析**

对比方法包括：无增强、全组梯度下降、流式 group‑SGD（每步采样 1 个变换）以及一次性稀疏 GD（|S|∈{4,16,64}）。实验表明，一次性稀疏 GD 在使用 64 个变换时，其训练/测试风险几乎与全组 GD 相当，而所需的群查询仅为全组的 1/720；流式 group‑SGD 的梯度范数波动较大；

**⚠️ 局限性**

局限性：证明仅适用于光滑非凸目标且需要群作用可单射、可逆；实验仅在小规模仿真任务上验证，未探讨大规模深度网络或连续变换群的推广；

---

## 330. TTGBench: Benchmarking Topological Evolution and Semantic Drift in Text-attributed Temporal Graphs

**arXiv ID:** 2609.08226 | [PDF](https://arxiv.org/pdf/2609.08226v1)

**作者:** Longfei Ma `[一作]` (Zhejiang University), Fei Wu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了TTGBench基准，联合评估时间图结构演化和语义漂移，提供六个文本丰富、双重波动的数据集；

**💡 创新点**

创新点在于首次支持多类别、多标签TNC与TLP的统一评估，并构建高结构和语义波动的真实数据集，揭示TGNN与LLM模型的能力分裂；

**🔧 技术方法**

采用Temporal Graph Neural Networks与大型语言模型（如Qwen3-8B、LLaGA、GraphGPT）作为预测器，结合文本编码与图序列化；

**📊 数据集**

使用六个行业场景文本图数据集（FOOD、IMDB、Librarything、Beeradvocate、Ratebeer、Amazon-Kindle）；

**📈 对比分析**

通过对17种先进方法的系统实验，发现TGNN模型在TLP表现突出但TNC表现差；LLM预测器在TNC强，但在TLP远逊于TGNN；两类模型在效率与规模上亦存在显著差距；

**⚠️ 局限性**

局限在于缺乏统一模型同时处理结构与语义演化，LLM模型计算成本高，TGNN模型对语义漂移不敏感，整体仍难以兼顾性能与效率。

---

## 331. Vision: Data-Centric Anchoring for Robust and Interpretable Agentic AI

**arXiv ID:** 2609.08216 | [PDF](https://arxiv.org/pdf/2609.08216v1)

**作者:** Arun Vignesh Malarkkan `[一作]` (Arizona State University), Yanjie Fu `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向代理式AI的Data-Centric Agentic Loop，通过四阶段（Curate、Augment、Constrain、Attribute）系统性改进数据生命周期，以提升鲁棒性与可解释性。

**💡 创新点**

将鲁棒性与可解释性视为数据结构缺陷的两种共症，将模型训练视为数据设计问题，构建以数据迭代为核心的闭环框架，并提出失效驱动的四类失效模式与对应诊断指标。

**🔧 技术方法**

利用数据挑选与清洗（Cleanlab、Snorkel）、合成与对抗增强、分布鲁棒优化（IRM、Group DRO）以及归因与对抗验证（Influence Function、TRAK）等技术实现四阶段流程。

**📊 数据集**

使用多任务语言代理实验环境（例如基于AgentBoard/AgentNoiseBench等仿真/对抗数据集）进行验证，并通过自定义的分段诊断指标（slice leakage、invariance gap、ECE、counterfactual faithfulness）评估。

**📈 对比分析**

相较于单一模型改进方法，Data-Centric Agentic Loop在鲁棒性和解释性指标上均显示出显著提升；但实验多基于合成与模拟环境，缺乏大规模真实世界基准验证。

**⚠️ 局限性**

局限包括：对观测数据的可观测性依赖导致残余偏差难以消除；合成增强可能引入结构失真；归因方法在低密度区域不稳定；循环假设目标分布稳定但代理式AI的自我产生环境导致非平稳性。

---

## 332. SIM: Subspace Interaction-based Method for Token-Level Text Anomaly Detection

**arXiv ID:** 2609.08200 | [PDF](https://arxiv.org/pdf/2609.08200v1)

**作者:** Kehan Yan `[一作]` (Guangxi University), Yixin Liu `[通讯]` (Griffith University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于子空间交互的Token级文本异常检测方法（SIM），可精准定位异常词。

**💡 创新点**

创新点包括：①将高维词向量划分为子空间并通过跨子空间自注意力放大局部异常信号；②设计硬伪异常生成模块以克服预训练模型的过平滑；③使用概率边界损失将异常分数标准化为统计距离。

**🔧 技术方法**

采用BERT预训练语言模型生成词嵌入，子空间划分、跨子空间注意力、MLP评分、硬伪异常扰动、概率边界损失及最大池化聚合。

**📊 数据集**

在三大基准数据集上评估：Grammar（语法错误）、Review（负面情感）和SMS_Spam（文本腐败/垃圾短信）。

**📈 对比分析**

与LOF、iForest、ECOD、DeepSVDD、AutoEncoder、LUNAR、TokenCore及GPT-4.1-nano等方法对比，SIM在Token级AUROC和AUPRC均领先，尤其在SMS_Spam上达到98.44 AUROC，整体提升超过10%。

**⚠️ 局限性**

局限性在于仍依赖预训练模型的词向量，若异常类型与训练集差异过大，或对伪异常参数调优不当，性能可能受限。

---

## 333. Does Deeper Reasoning Compromise Alignment? Revealing and Mitigating of Alignment Collapse in Large Reasoning Models

**arXiv ID:** 2609.08186 | [PDF](https://arxiv.org/pdf/2609.08186v1)

**作者:** Yu-Hang Wu `[一作]` (Shanghai University of Engineering Science), Shaohua Li `[通讯]` (A STAR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究深度推理对大规模推理模型（LRM）对齐鲁棒性的影响，发现随着推理深度增大，模型对外部扰动的敏感性显著上升，称为对齐崩塌。

**💡 创新点**

提出 Alignment Loss Rate（ALR）指标量化对齐崩塌；设计 Reasoning Trap（RT）框架通过诱导推理放大攻击效果；通过注意力稀释（Attention Dilution）解释对齐崩塌机制，并提出 Reasoning Residual Alignment（RRA）轻量化训练无关的对齐防御。

**🔧 技术方法**

主要技术包括链式推理（Chain-of-Thought）激活、对齐损失率评估、注意力分配分析、RT 的 prompt 触发与攻击融合，以及 RRA 的残差连接重注入。

**📊 数据集**

使用 AIME2024、LogicAsker 作为推理任务数据集，使用 AdvBench（520 条恶意提示）评估安全对齐；在 Qwen3、Llama2、DeepSeek、Claude-Haiku、GLM4.6 等多种模型上进行实验。

**📈 对比分析**

实验显示 ALR 随推理深度单调上升；RT 在深度推理模式下可将拒绝成功率（RSR）下降 30%~70%，显著弱化安全对齐；RRA 在同一设置下能恢复约 6%–7% 的 RSR，证明防御有效。

**⚠️ 局限性**

局限在于实验环境受控，仅测试了有限模型、数据集和扰动类型；注意力分析未能覆盖所有导致鲁棒性下降的因素，未来需扩展至更广泛模型、攻击方式及更深入的机制探究。

---

## 334. Safe Harness Self-Evolution: A Theoretical Analysis of Feasibility and Limits

**arXiv ID:** 2609.08175 | [PDF](https://arxiv.org/pdf/2609.08175v1)

**作者:** Qianshu Cai `[一作]` (University of Science and Technology of China), Wei Xue `[通讯]` (Hong Kong Generative AI Research and Development Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Harness Self‑Evolution（保持语言模型冻结、仅修改提示、工具、代码等 harness）的理论可行性与极限进行了系统分析，涉及生成、评估、选择以及后续改进的安全性。

**💡 创新点**

创新点：① 将改进目标细分为失败任务提升与保留任务变化，并给出期望奖励改进的充分条件；② 推导生成可达性与候选池大小的上界与下界；③ 在有限评估下提供安全采纳的概率保证；④ 证明即使实现安全改进也不一定能保证后续进一步改进；⑤ 提供可同时覆盖所有目标的置信区间方法，用于诊断停滞。

**🔧 技术方法**

使用技术：统计学习理论（PAC、Hoeffding、VC理论）+ 置信区间与多目标风险分配；有限样本检验；理论构造与蒙特卡洛仿真。

**📊 数据集**

数据集：实验使用 DS‑1000（Python 求解器 harness）和 WorkBuddy‑DSH；理论构造使用自定义的两个任务族。

**📈 对比分析**

比较方法：对比不同生成、评估、选择策略在目标满足率、认证率、选择率上的表现；结果显示目标满足率高但认证率低，说明评估宽度是瓶颈；多步更新中，累计改进可通过安全阈值保证，但后续改进可能停滞。

**⚠️ 局限性**

局限性：仅适用于冻结 LLM 的持久 harness 改动；假设固定奖励函数和任务分布；评估置信区间在接近奖励上限时会发散；无法保证后续改进；对非 i.i.d. 生成或更复杂的更新空间的分析有限。

---

## 335. Geodesic-informed Generative Diffusion Model For Topology-preserved Image Video Generation

**arXiv ID:** 2609.08153 | [PDF](https://arxiv.org/pdf/2609.08153v1)

**作者:** Nian Wu `[一作]` (University of Virginia), Miaomiao Zhang `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了IGG，一种在形变空间学习geodesic的生成扩散模型，用于生成保持拓扑结构的图像序列。

**💡 创新点**

创新点在于结合geodesic-informed registration网络与latent geometric diffusion，能够在文本指导下生成拓扑一致的图像，并提供DetJac等拓扑评估指标。

**🔧 技术方法**

使用了Neural EP Diff、U-Net、CLIP文本编码、三维UNet以及扩散概率模型等技术。

**📊 数据集**

采用了Komatsuna植物生长数据和OASIS-3脑MRI时间序列作为训练与评估数据集。

**📈 对比分析**

与VDM、CogVideoX、DynamiCrafter等基线对比，IGG在FID、KID、SSIM、FVD等指标上显著优于基线，且在下游脑区分割任务中提升Dice分数。

**⚠️ 局限性**

局限在于对配准误差敏感，假设解剖结构保持微分同胚，难以处理非同胚变化；此外仅在2D图像上验证，4D应用待扩展。

---

## 336. Topology-induced Operators Reveal Complementary Graph Representations without Training

**arXiv ID:** 2609.08152 | [PDF](https://arxiv.org/pdf/2609.08152v1)

**作者:** Meng Qin `[一作]` (Pengcheng Laboratory), Sen Pei `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过对图的随机游走（RW）和匿名游走（AW）构造层次结构，并用随机特征在该结构上做一次无训练的前向传播，得到分别保留节点位置和节点身份信息的嵌入。

**💡 创新点**

创新点在于：①完全无需梯度训练，只靠拓扑诱导的算子和随机特征即可产生有用的嵌入；②RW和AW分别揭示了互补的拓扑通道（位置与身份）；③通过简单的加权拼接或求和即可将两类嵌入融合，进一步提升多种任务性能；④在保持高质量的同时显著降低了计算成本，实现了优越的质量–效率权衡。

**🔧 技术方法**

采用的主要技术包括：随机游走和匿名游走生成的层次结构；Gaussian 随机投影思想的单层前向传播；跳跃连接、归一化和可选非线性激活；AW 统计量的高效估计与树形编码；以及在不同层级的权重归一化与融合。

**📊 数据集**

使用了 8 个真实世界图（欧洲/美国机场网络、演员/电影共现网络、PPI、博客社交网络、DBLP 作者网络、亚马逊商品网络）和 9 个 LFR 生成的合成图进行评估。

**📈 对比分析**

与 18 种经典与前沿基线（包括 node2vec、struc2vec、RandNE、LouvainNE、S3GC、MAGI、DGI 等）在 7 种节点、边、图级任务上对比。实验表明：PI‑HIST（R）/（A）在位置/身份相关任务上与最强基线持平甚至超越，且耗时显著更短；两种嵌入的融合在 link prediction、graph reconstruction 与 superfamily 识别等任务中进一步提升了 1%–2% 的性能，展示了互补优势。

**⚠️ 局限性**

局限性包括：①仅利用拓扑信息，未考虑节点属性，适用于无属性或属性不可靠的场景；②AW 统计量估计需要随机游走采样，采样量与时间成本有关；③模型无学习参数，无法针对特定任务做细粒度调优，导致在某些需要属性或语义信息的任务上可能不如自监督 GNN；④目前仅针对静态无向图，扩展到动态图或有向图仍需进一步研究。

---

## 337. Bridging Language and Physics: Automated Design of Continuum Robots with Large Language Models

**arXiv ID:** 2609.08220 | [PDF](https://arxiv.org/pdf/2609.08220v1)

**作者:** Jingyi Chen `[一作]` (University of Science and Technology of China), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个多层反馈循环框架，用大型语言模型（如 GPT‑5）自动生成、验证并迭代改进肌腱驱动连续体机器人的设计，并在模拟与现实环境中进行测试。

**💡 创新点**

创新点在于将物理稳定性验证、场景集成、语义评判和人工反馈四层嵌套到 LLM 生成流程中，形成闭环，使 LLM 能够逐步理解并纠正其设计中的物理失效，从而显著提升设计的物理可行性和功能性。

**🔧 技术方法**

技术栈包括 GPT‑5 语言模型、MuJoCo 物理引擎、Gymnasium 接口、PPO 强化学习、Web 可视化界面以及基于规则的物理诊断与语义评判模块。

**📊 数据集**

使用了 14 个涵盖到达、抓取、行走和复杂操纵的任务基准（无公开数据集），每个任务提供环境 XML 与目标描述，构成自定义的实验数据集。

**📈 对比分析**

通过对比完整框架与多种剔除某一反馈层的消融实验，评估了有效率、成功率、token 使用、成本和延迟等指标；平均有效率 96.2%，整体任务成功率约 26.7%，GPT‑5 在物理有效率和成功率上优于其他 LLM。

**⚠️ 局限性**

主要局限包括：LLM 仍难以生成高复杂度、任务相关的设计，成功率受限；仿真与真实硬件存在显著差距；缺乏针对任务性能的细粒度反馈；对模型容量高度依赖，低容量模型效果差。

---

## 338. Drive by Hindsight and Foresight: Tool-Grounded Synergistic Reasoning over Hierarchical Memory for Autonomous Driving

**arXiv ID:** 2609.08217 | [PDF](https://arxiv.org/pdf/2609.08217v1)

**作者:** Baojie Chen `[一作]` (Beihang University), Jing Zhong `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了基于分层记忆与主动工具调用的闭环推理框架，用于自动驾驶视觉语言模型的决策与解释。

**💡 创新点**

将短期场景记忆、长期经验记忆与在线工具调用结合，形成前瞻与回顾相互补充的推理循环，并通过离线记忆合并实现模型自我进化。

**🔧 技术方法**

采用多模态记忆‑工具协同推理、两阶段后训练（SFT + GRPO）、教师滚动生成验证轨迹、BEV感知与图谱构造、LLM的工具接口等技术。

**📊 数据集**

主要使用DriveLMM‑o1、DriveMLLM、STRIDE‑QA、STSBench等自动驾驶评测基准。

**📈 对比分析**

与GPT‑4o、AgentThink、Qwen系列等强基线对比，DriveLMM‑o1上获得80.03分推理得分、79.09% MCQ准确率，提升约7–28个百分点，且跨基准零样本表现显著。

**⚠️ 局限性**

依赖外部工具执行、训练成本高、记忆检索效率与可扩展性受限，且离线合并在一定生成次数后趋于饱和。

---

## 339. TacClip: a clip-on sensor measures dynamic contact forces without covering the fingerpads

**arXiv ID:** 2609.08214 | [PDF](https://arxiv.org/pdf/2609.08214v1)

**作者:** Yuqian Ye `[一作]` (Stanford University), Mark R. Cutkosky `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并测试了一种可贴在手指上的纤维布拉格光栅（FBG）传感器TacClip，用于测量因接触产生的指尖变形，从而推算接触力并捕获动态振动信号。

**💡 创新点**

①使用可拆卸夹式结构保持指尖裸露，避免遮挡触感；②单个FBG实现静态力与动态振动双重功能；③支持水下使用；④可与视觉系统融合补偿手势遮挡。

**🔧 技术方法**

采用FBG光纤传感、光纤波长偏移测量、多层感知器（MLP）力映射、光纤多路复用、视觉手势追踪（MediaPipe）、多模态融合网络、肌肉杠杆振动激励、ATI力/扭矩传感器校准以及水下实验等技术。

**📊 数据集**

使用手指按压与ATI力传感器同步数据、肌肉杠杆扫频振动数据、布料滑动与透明胶带边缘检测实验数据、水下弹簧压缩实验数据以及多摄像机运动捕捉与手势标注数据等多种实验数据集进行评估。

**📈 对比分析**

通过RMSE衡量静态力估计误差，结果在0–8 N范围内<0.5 N；频率响应在20–30 Hz稳定，能捕获大部分振动；胶带边缘检测随按压力增大误差降至<1 mm；与单纯视觉相比，融合方案在尺寸估计上误差下降约15–25%，轨迹更稳定。

**⚠️ 局限性**

仅能估计总接触力大小，无法分离x/y/z分量；对用户尺寸、预载和温度敏感，需要快速个人校准；高频振动因手指软组织滤波被衰减；夹具需紧密贴合，易滑移；光纤测距仪成本较高。

---

## 340. Asymptotically good binary triorthogonal codes and higher-level transversal gates

**arXiv ID:** 2609.08203 | [PDF](https://arxiv.org/pdf/2609.08203v1)

**作者:** Rodrigo San-José `[一作]` `[通讯]` (Virginia Tech), Rodrigo San-José (Virginia Tech)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了支持对角横向门的二进制CSS码，特别是从第三层开始的Clifford层次的显式渐近良好二进制CSS码。

**💡 创新点**

首次获得了渐近良好的二进制三正交码族，并提供了不需要后续纠正的更强构造。

**🔧 技术方法**

使用了代数几何码和字母减少技术，将扩展域上的正交条件转化为二进制重叠条件。

**📊 数据集**

使用了二进制扩展域上的代数几何码。

**📈 对比分析**

与现有方法相比，构造的CSS码在保持线性速率和线性距离方面表现良好，且在足够弱的输入噪声和理想稳定器操作下，能够实现常数开销的T态块蒸馏。

**⚠️ 局限性**

构造的CSS码在参数上可能不如某些现有构造，尤其是在不需要后续纠正的情况下。

---

## 341. Beyond Cut Balance: Spectral Sparsification of the Nonlinear Directed Laplacian

**arXiv ID:** 2609.08177 | [PDF](https://arxiv.org/pdf/2609.08177v1)

**作者:** Yuichi Yoshida `[一作]` `[通讯]` (National Institute of Informatics), Yuichi Yoshida (National Institute of Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了具有常量切割平衡的有向图是否允许近线性谱稀疏化，特别是针对非线性有向拉普拉斯的能量进行谱稀疏化。

**💡 创新点**

证明了切割平衡本身并不保证近线性谱稀疏化，而不平衡的图也可以实现近线性谱稀疏化。

**🔧 技术方法**

使用了非线性有向拉普拉斯的能量函数 Q_G^+(x) 进行谱稀疏化。

**📊 数据集**

使用了简单的欧拉有向图和锦标赛图作为数据集进行实验。

**📈 对比分析**

与现有方法进行比较，发现简单的欧拉有向图的最坏情况支持大小为 Θ(n^3/2)，而每个 n-顶点的锦标赛图的谱稀疏化支持大小为 O(n/ε^3)。

**⚠️ 局限性**

限制在于对于一般的有向图，无法保证次线性稀疏化，尤其是在没有额外结构的情况下。

---

## 342. Boundary Voting Network for Ambiguity-Aware Timestamp-Supervised Action Segmentation

**arXiv ID:** 2609.08167 | [PDF](https://arxiv.org/pdf/2609.08167v1)

**作者:** Runzhong Zhang `[一作]` (Nanyang Technological University), Yap-Peng Tan `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种全局到局部的边界投票网络（BVN），用于解决仅有时间戳监督的动作分割问题；

**💡 创新点**

创新点在于通过投票块和聚合块，将全局视频先验知识层层传递到动作过渡区，显著降低特征歧义和边界定位不确定性；

**🔧 技术方法**

技术实现包括投票块（start_net、end_net）、聚合块（基于时间远点采样与MLP聚合）、投票损失以及在MS‑TCN++/ASFormer等基底上的编码-解码框架；

**📊 数据集**

实验使用了三大公共数据集：GTEA、50Salads 和 Breakfast；

**📈 对比分析**

在所有评估指标（F1、Edit、Acc）上，BVN 在 timestamp‑supervised 设定下实现了 state‑of‑the‑art 性能，并且在相同标注成本下甚至超过部分全监督方法；

**⚠️ 局限性**

局限性包括：相对较高的计算复杂度和参数量；对超参数（距离阈值、投票组数）敏感；以及仍需两阶段训练流程。

---

## 343. Snugi-AI-v2 @ eRisk 2026 Task 2: Early Depression Detection via a Learned Stopping Policy with Sustained Confidence Gate

**arXiv ID:** 2609.08161 | [PDF](https://arxiv.org/pdf/2609.08161v1)

**作者:** Yuwen Chiu `[一作]` `[通讯]` (Georgia Institute of Technology), Yuwen Chiu (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个三阶段轻量化管道，利用冻结的 MentalRoBERTa 编码器、MLP 分类器和学习到的停止策略，配合持续置信门，实现对 Reddit 讨论线程的早期抑郁检测并给出及时警报。

**💡 创新点**

创新点包括：① 学习到的停止策略直接针对 ERDE50 进行优化，取代传统的固定/分层阈值；② 持续置信门（需要 N=3 连续高置信度）用于过滤瞬时情绪波动；③ 采用 O(1) 的增量均值池化，保证实时性能；④ 系统在 eRisk 2026 Task 2 中首次实现学习式停止策略并实现最快提交。

**🔧 技术方法**

主要技术包括：Frozen MentalRoBERTa-base/large 作为特征提取器；三层 MLP 分类器（770→192→48→1）；五维特征输入的 MLP 停止策略（5→32→16→1）；持续置信门（N=3）；BCEWithLogitsLoss 训练策略与 ERDE50 监督；交叉验证与 Ablation 评估。

**📊 数据集**

使用 eRisk 2025 训练集（909 受试者，包含完整 Reddit 讨论线程）作为训练数据；eRisk 2026 测试集（523 用户，500 轮）用于最终评估；数据中抑郁样本 11.2%，控制样本 88.8%。

**📈 对比分析**

通过与固定阈值、分层阈值、上下文嵌入、不同 N 的持续门等基线的 ablation 进行对比；验证集 ERDE50 最低为 0.0267，测试集 F1 为 0.73，精确率 0.80，速度 0.97，完成 500 轮评估仅需 1 小时 26 分钟；在 NDCG@100 评价中排名第四，表现优异。

**⚠️ 局限性**

局限性包括：仍存在硬性误报，难以区分共情语言与自报；未对 encoder 进行微调，可能限制表达能力；对极慢上升的抑郁信号敏感度不足；缺乏发帖动机等辅助信号；对长对话的上下文建模仍为均值池化，未利用跨帖注意力。

---

## 344. SemBridge: Compiling Consumer Observations into Cross-Stack Communication Plans

**arXiv ID:** 2609.08231 | [PDF](https://arxiv.org/pdf/2609.08231v1)

**作者:** Genlang Chen `[一作]` (NingboTech University), Yuanshan Lin `[通讯]` (Dalian Ocean University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SemBridge，利用类型化的边界合同、确定性下推器和符号检查器，在跨 CUDA/NCCL 与 CANN/HCCL 的异构推理系统中显式表达消费者观察，从而生成满足多种交付义务（投影、复制、补全、权威等）的通信计划。

**💡 创新点**

创新点在于：1) 将消费者可见结果与交付义务分离，形成跨域可验证的 typed contract；2) 通过有序下推器将合同编译成后端无关的逻辑图；3) 使用符号检查器保证计划满足覆盖、完整性、来源、权威和域局部性等语义约束；4) 在同一物理部署上实现全日志传输与源侧投影、所有者令牌交付等多种策略，显著降低通信开销。

**🔧 技术方法**

技术包括：分布式张量编译（GSPMD/DTensor）、跨域点对点传输与本地集合操作（NCCL/HCCL）、类型化合同与符号执行、交叉栈逻辑图下推、跨域通信计划生成与验证、性能计时与指标收集。

**📊 数据集**

使用的数据集/模型：MiniMax‑M2.7（容量溢出工作负载），Qwen3‑14B Dense（稠密推理），Qwen3.5 MoE（混合专家）以及对应的 MiniMax、Qwen3‑14B 与 Qwen3.5 MoE 的多线程/多进程推理实验。

**📈 对比分析**

比较方法：与独立的 layout‑only 基线、Semantic Oracle（最优计划枚举）和 byte‑only minimizer 对比；评估指标包括交付字节、启动次数、吞吐量、p95 第一个令牌时间、p95 每令牌时间、总延迟、框架计时。实验结果显示：源侧投影将结果流量降低 99.97%+，吞吐量提升 8.9%–80.2%，p95 延迟降低 8.1%–44.5%；代表转发在 TP4 上减少 75%+ 交通并提升 13.7%–38.6% 吞吐。

**⚠️ 局限性**

限制：仅在两台物理主机、1GbE 互连环境下验证；依赖 CUDA/NCCL 与 CANN/HCCL 的本地集合实现；计划空间受限于 4 生产者/消费者、单点传输，未覆盖动态扩缩、故障恢复和多租户场景；符号检查器假设投影算子正确实现，若不符会导致错误。

---

## 345. ActionSplice: In-Flight Action Editing for Interactive World Models

**arXiv ID:** 2609.08230 | [PDF](https://arxiv.org/pdf/2609.08230v1)

**作者:** Pardis Taghavi `[一作]` (Texas A&M University), Reza Langari `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在实时视频世界模型的chunk‑autoregressive采样过程中实现动作更新时，提供一种无重放已完成评估的即时状态修正方法。

**💡 创新点**

提出Counterfactual State Transport（CST）框架，利用匹配的回滚对照学习，在保持已完成的评估不变的前提下，将中断状态迁移到相同求解步骤下的理想反事实状态；并给出了两种变体：全块重定向（CST_R）和子块时间剪切（CST_T）。

**🔧 技术方法**

使用轻量级纠正器（6层3D残差编码器‑解码器）对中断状态做残差预测，FiLM调制动作与求解步信息；采用匹配的回滚数据构造监督信号；在minWM–Wan Action2V与HY‑WM1.5两个基线模型上训练与评估。

**📊 数据集**

对minWM–Wan Action2V和HY‑WM1.5两个开源视频世界模型进行实验；使用与它们相同的Prompt‑分离训练/测试集（150个场景Prompt，其中120训练、30测试）。

**📈 对比分析**

与等待、直接条件交换、局部回滚、重噪等传统方法对比；CST_R在LPIPS上分别降低61.5%和75.9%，PSNR提升3.3~5.3 dB；CST_T在后缀LPIPS上降低56.1%和77.5%，PSNR提升3.3~9.5 dB；两种变体均比等待快约2.7×（minWM）和1.6×（HY‑WM1.5）。在HY‑WorldPlay基准上，CST_R实现PSNR 25.66、SSIM 0.6902、LPIPS 0.1337，均为公开结果中的最佳或次佳。

**⚠️ 局限性**

需要额外的回滚对照数据来训练纠正器，训练过程依赖于模型冻结与采样器保持不变；对超出训练动作空间的更新仍能泛化但精度下降；多次连续更新时误差累积仍低于直接条件交换但略高于完整回滚；对动态场景的鲁棒性尚待进一步验证。

---

## 346. Finite-Modal Realization and Operator-Norm Convergence of a Source-to-Observation Electromagnetic Scattering Green Operator

**arXiv ID:** 2609.08229 | [PDF](https://arxiv.org/pdf/2609.08229v1)

**作者:** Zhukang Wang `[一作]` (Zhejiang University), Er-Ping Li `[通讯]` (Zhejiang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了在固定连续希尔伯特空间上的源到观测散射Green算子，并给出了其对Maxwell方程的物理解释；在此基础上通过向量球面波函数（VSWF）进行有限模数化实现，证明了其算子范数收敛，并提出了保留非零奇异值的有限度量核心，能够恢复最优源-场通道。

**💡 创新点**

创新点在于：①首次给出源到观测散射算子在固定空间上的完整算子范数收敛定理；②提出结构化误差上界，将外部模数尾与集体逆变换敏感度分离；③构造了不引入外部离散自由度的有限度量核，保证奇异值物理意义；④通过两球近谐振实验阐释了内部放大与外部可达通道的分离。

**🔧 技术方法**

主要技术包括：Maxwell Calderón理论、VSWF展开与加法定理、折叠投影与正交化、折叠算子范数收敛证明、Moore–Penrose伪逆与奇异值分解、全波有限元/边界元验证、以及对角线Mie T矩阵提取。

**📊 数据集**

数据集为混合集群，包含半径ka=1的PEC球体以及从SCUFF-EM提取的四面体、椭球、环面在1 GHz下的局部T矩阵；两球实验使用相同介质常数的球体，并在不同谐振状态下扫描折射率。

**📈 对比分析**

与全波（COMSOL）比较时，源到场实现误差低于2.3%，散射场误差随集群规模从16.4%下降到4.6%；在高阶截断下，计算时间相较于全波可实现85×–175×的加速；算子范数误差指标在L=14时对近谐振情况误差<0.5%，并揭示了收敛瓶颈；奇异值分析表明近谐振时出现高阶通道极大增强，非谐振时低阶通道占主导。

**⚠️ 局限性**

局限性包括：①对非共形或高对比度介质场需要更高模数截断；②在强谐振或近场耦合情形下，收敛速度显著下降，需要更细致的外部尾约束；③理论假设离散化与连续希尔伯特空间的分离，若源或观测表面与球面接触则不适用；④实现依赖于精确的T矩阵提取，提取误差会影响最终结果。

---

## 347. 3DWay: Generalizing Robot Manipulation via 3D Consistent Waypoints

**arXiv ID:** 2609.08224 | [PDF](https://arxiv.org/pdf/2609.08224v1)

**作者:** Ziqin Huang `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过多视角RGB图像预测3D一致的中间点（waypoints），并将其用于提升机器人操纵的通用性

**💡 创新点**

将3D waypoint生成转化为多视角一致的2D waypoint预测+几何三角化，保留预训练VLM的视觉语言知识，并提出适应式waypoint引导融合方案

**🔧 技术方法**

多视角一致性监督、NVILA VLM微调、RDP轨迹简化、自动质量评分、几何三角化、适应式waypoint引导整合

**📊 数据集**

RoboPoint、RLBench、DROID、RH20T、VLABench等，共约290项任务、134k轨迹

**📈 对比分析**

与OpenVLA-OFT、π_0、π_0.5等基础VLA对比，3DWay在零样本和少样本场景下成功率提升6–26%，在模拟与真实环境中表现优异

**⚠️ 局限性**

仅生成平移waypoint，未建模姿态或动态适配；依赖预标定相机且仅验证双视角；集成机制仍处于初级阶段

---

## 348. A Better Spur Should Start From Each Objective

**arXiv ID:** 2609.08211 | [PDF](https://arxiv.org/pdf/2609.08211v1)

**作者:** Shanwen Mao `[一作]` (Harbin Institute of Technology), Gu Simiu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种多目标强化学习框架MMPO，用于解决电商商品详情页中的词条提取与解释生成任务；

**💡 创新点**

创新点在于三层干预：数据层通过曝光去偏与傅里叶平滑缓解稀疏奖励；梯度层通过低维子空间优先级顺序正交投影解耦梯度冲突；约束层通过自提示梯度限制形成自适应信任区间，抑制“reward tug‑of‑war”。

**🔧 技术方法**

使用了曝光去偏与傅里叶特征映射的奖励平滑技术；低维子空间正交投影与优先级顺序投影；自提示梯度约束与坐标裁剪；以及传统KL、梯度正则化等基础RL工具。

**📊 数据集**

主要在真实电商数据上（约65k条商品描述、3个月用户行为反馈）进行离线评测；外部评测使用ToolRL、LiveCodeBench等公开基准；在线A/B测试覆盖10%流量。

**📈 对比分析**

与GRPO、GDPO等基线相比，MMPO在离线指标（完整度、准确率、覆盖率）上分别提升≈6%/10%/≈10%；在在线A/B测试中GMV+5.07%、订单+3.14%；在ToolRL和代码生成基准中显著提升整体准确率与多目标均衡性能。

**⚠️ 局限性**

受限于基础LLM的能力与安全性，评估“过程完整性”仍是挑战；对高级对抗攻击、跨模态“smuggling”等攻击场景尚无完整防御机制。

---

## 349. UnespDataLens-RM: A Reference Model for Analytical Data Engineering with Governance, Quality, Provenance, and Reproducibility

**arXiv ID:** 2609.08184 | [PDF](https://arxiv.org/pdf/2609.08184v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 350. Stabilizing Instruction Supervision for Instruct-TTS via Controllable Diversification and Drift Filtering

**arXiv ID:** 2609.08204 | [PDF](https://arxiv.org/pdf/2609.08204v1)

**作者:** Yizhong Geng `[一作]` (Beijing University of Posts and Telecommunications), Ya Li `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种数据中心化的 Instruct‑TTS 稳定化方案，通过可控指令多样化、LLM 语义漂移过滤和属性对齐监督，提升语音指令跟随性能。

**💡 创新点**

将指令监督不稳定性概念化为漂移分类，联合覆盖与忠实度提升，首次在 TTS 上结合可控多样化与验证过滤，并加入低级属性对齐，实现了全流程的质量提升。

**🔧 技术方法**

使用 LLM 重新写作（如 DeepSeek‑R1、GPT‑4o）生成指令，采用漂移判定器投票过滤，配合音高/语速/音量参数扰动生成属性指令，基于 CosyVoice 2.0 进行微调，并用 Gemini 进行评判。

**📊 数据集**

使用约 90 小时中文语音数据（12k 条短片）以及 InstructTTSEval 中文测试集，并通过对音高、语速、音量进行扰动扩增，形成属性对齐的训练样本。

**📈 对比分析**

在 InstructTTSEval 中文评估中，完整方案将指令跟随准确率从 34.5% 提升至 56.4%，并在自然度和可控度人类评分上均超过 4.0，明显优于 VoxInstruct（47.5%）等基线。

**⚠️ 局限性**

方案依赖多种 LLM 交叉验证，成本较高；漂移过滤仍需人工校准阈值；属性对齐目前仅覆盖音高/语速/音量，尚未扩展到更复杂的情感等特征。

---

## 351. Bridging the Semantic-Utility Gap in Multimodal RAG via Generator-in-the-Loop Alignment

**arXiv ID:** 2609.08188 | [PDF](https://arxiv.org/pdf/2609.08188v1)

**作者:** Zhan-Lun Chang `[一作]` (Purdue University), Christopher G. Brinton `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种两阶段的生成器‑in‑the‑loop 对齐框架，先用冻结的多模态语言模型将图像+文本查询转换为文本化的假设推理文本（HyDE），再使用该模型在检索到的候选文档上挖掘答案级别的优劣偏好，用 LoRA 细调的交叉编码器实现检索重排序，并通过周期性重新挖掘保持训练信号与重排序器行为一致。

**💡 创新点**

创新点在于：① 用生成器生成的假设推理文本桥接模态鸿沟；② 用生成器的答案正确性直接标注偏好，无需人工文档级相关性标注；③ 将偏好挖掘与 LoRA 微调、周期性重挖深度融合，形成端到端的答案导向检索优化流程。

**🔧 技术方法**

技术手段包括：多模态 VLM（Qwen3.5‑2B、Qwen3‑VL‑4B‑Instruct）做 HyDE 与答案生成；文本稠密检索（FAISS + Qwen3‑Embedding‑8B）；LoRA‑适配的 GTE‑ModernBERT 交叉编码器做重排序；对齐损失三种实现（triplet、SFT、DPO）；周期性重挖与缓存加速的偏好挖掘。

**📊 数据集**

实验使用的公开数据集为 VQA‑X（图像+文本 + 说明）和 A‑OKVQA（图像+文本 + 逻辑推理理由），检索语料库由训练集中的说明/理由构成。

**📈 对比分析**

与 Rank‑Order、Random、REPLUG‑likelihood 等基线对比，生成器引导的偏好挖掘在所有对齐损失下均表现最佳：例如 Qwen3‑VL‑4B‑Instruct 在 VQA‑X 上准确率提升至 96.27%（基线 94.75%），A‑OKVQA 提升至 87.00%（基线 85.04%）；周期性重挖可进一步提升 2–3%。

**⚠️ 局限性**

局限性包括：仍需依赖答案标签做偏好挖掘；偏好挖掘及重挖消耗 VLM 推理资源；重排序器针对特定生成器训练，迁移性有限；仅生成正负对而忽略多文档或分级反馈，且对所有候选文档同等输出的查询会被跳过。

---

## 352. WSPolypNet: Weakly Supervised Polyp Localization in Colonoscopy Videos

**arXiv ID:** 2609.08182 | [PDF](https://arxiv.org/pdf/2609.08182v1)

**作者:** Giseong Hwang `[一作]` (Soonchunhyang University), Nam-Joon Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种弱监督的结肠镜视频息肉定位框架WSPolypNet，能够仅用视频级标签实现息肉的空间定位。

**💡 创新点**

创新点在于将3D CNN与CAM生成的粗定位结合多视角增强，并将CAM提示投递给MedSAM2进行精细化分割，实现无框架标签的精确定位。

**🔧 技术方法**

使用的技术包括3D CNN（X3D）提取时空特征、CAM定位、五视角聚合、多视角增强、MedSAM2点提示分割以及ROI预处理。

**📊 数据集**

数据集来源为LDPolypVideo和无息肉的结肠镜视频，构成871个正样本和615个负样本，总计1486个1 fps的视频片段。

**📈 对比分析**

通过与单视角设置和不同3D CNN骨干的比较，WSPolypNet在IoU 0.5下获得43.68% CorLoc，召回率94.51%，多视角策略在小息肉上显著提升了30.97% CorLoc。

**⚠️ 局限性**

局限性包括只能定位单个息肉，无法处理同一视频中多息肉或短暂出现的息肉；未来需改进多目标检测和对长视频的支持。

---

## 353. Cassette: Case-to-Case Structural Distillation for Efficient Legal Case Retrieval

**arXiv ID:** 2609.08185 | [PDF](https://arxiv.org/pdf/2609.08185v1)

**作者:** Yanran Tang `[一作]` (University of Queensland), Zi Huang `[通讯]` (University of Queensland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 Cassette 框架，用轻量化双塔结构在离线预处理候选案件、在线快速编码查询案件的方式，完成高效法律案件检索。

**💡 创新点**

创新点在于提出同时结合排名蒸馏和特征匹配（eigen‑matching）蒸馏的跨模态知识迁移策略，将教师 GNN 的关系知识和排序能力迁移至轻量化 MLP 与 GNN 学生编码器，显著降低 O(n²) 的图构建和推理成本。

**🔧 技术方法**

采用技术包括：教师端的 CaseLink‑style GNN 进行案例关系学习；学生端的两塔架构（查询 MLP + 候选 GNN）；排名匹配损失（MSE/ KL）与特征匹配损失；监督式对比学习；BM25 构建候选图；以及多层 MLP、GAT/GraphSAGE 等基础模型。

**📊 数据集**

实验使用三大基准：COLIEE2022、COLIEE2023（英语）以及中文 LeCaRDv2（55,192 份候选案件），覆盖不同司法领域与数据规模。

**📈 对比分析**

与 BM25、两塔语言模型、LLM 嵌入、CaseGNN、CaseLink 等基线进行对比，Cassette 在所有数据集的 P@5/R@5/Mi‑F1/MA‑F1/MRR/MAP/NDCG 等指标上与 CaseLink 竞争甚至略优，同时推理时间比 CaseLink 快 50‑340,000 倍，效率提升显著。

**⚠️ 局限性**

局限性包括：对候选案件图需要离线重建，难以实时处理新加入的案例；跨司法、跨语言迁移的鲁棒性尚待验证；在极端类别不平衡或分布漂移场景下，特征匹配蒸馏效果可能不稳定。

---

## 354. WorldAgen: Unified State-Action Prediction with Test-Time World Model Training

**arXiv ID:** 2609.08162 | [PDF](https://arxiv.org/pdf/2609.08162v1)

**作者:** Chi Wan `[一作]` (Northwestern University), Manling Li `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的 Vision‑Language‑Action（VLA）框架 WorldAgen，联合学习世界建模（预测未来观测）和动作预测，并在部署时通过 Test‑Time Training（TTT）自适应新环境。

**💡 创新点**

创新点包括：① 共享 Transformer 主干同时支持任务条件动作预测与任务无关的世界建模；② 设计了 Mixed Unidirectional Attention Mask 防止信息泄漏；③ 在 TTT 阶段仅对世界建模头使用 LoRA 轻量级微调，实现在线快速适应。

**🔧 技术方法**

技术手段包括：Transformer 主干、混合单向注意力掩码、LoRA 参数高效微调、教师强迫（teacher forcing）、轨迹切分与分块、两步推理（先动作后观测）等。

**📊 数据集**

使用的公开数据集为 CALVIN（长时序语言条件操控）和 LIBERO（终身学习与跨任务迁移）。

**📈 对比分析**

方法与多种基线（RoboFlamingo、SuSIE、GR‑1、3D Diffuser Actor、CLOVER、Seer 等）在 CALVIN 和 LIBERO 上进行对比。WorldAgen 在无 TTT 时已达到或超过现有最优水平，加入 TTT 后成功率进一步提升（如 CALVIN 5 连续任务平均成功率从 96.3% 提升至 96.6%，LIBERO 任务平均成功率从 75.5% 提升至 79.0%）。

**⚠️ 局限性**

局限性包括：仅评估单一模型规模和架构；TTT 仅针对世界建模头，未探索对动作预测的联合适配；仅在模拟环境/公开基准上验证，缺乏真实机器人实验。

---

## 355. OmniNav: Robust Long-Horizon Target Navigation in Dynamic Environments

**arXiv ID:** 2609.08159 | [PDF](https://arxiv.org/pdf/2609.08159v1)

**作者:** Yujie Tang `[一作]`, Yufeng Yue `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出OmniNav，一种训练无关的框架，用于在动态环境中进行长时段目标导航与移动操控。

**💡 创新点**

核心创新在于三方面：① 采用贝叶斯持续性推断维护实例级3D场景记忆，实时剔除过期物体；② 在探索阶段将语义共现与前进探索进度结合，形成区域级目标后验；③ 通过可达性约束的基座姿态优化与分层闭环恢复，使导航终点兼具交互可行性并能应对执行失败。

**🔧 技术方法**

技术手段包括：物体检测与分割（YOLO-World+TAP）、视觉-语言模型（CLIP/BLEIP-2、LLM用于指令解析）、3D voxel融合、贝叶斯推断、依赖感知的区域聚合、前沿搜索策略、双源确认（场景记忆+VLM观测）、可达性双椭圆Shell、粒子群优化、以及层级错误恢复。

**📊 数据集**

实验基于Habitat（HM3D、MP3D）以及Gibson模拟环境；真实世界在Unitree Go2平台上进行动态长时段导航与拾取-放置任务。

**📈 对比分析**

与多种现有方法（CoW、ESC、SG-Nav、UniGoal、DualMap等）对比，OmniNav在语义目标导航的SPL/成功率上均超越最佳基线（在MP3D上SR+0.053，SPL+0.016），在细粒度实例导航上SR提升至0.271；在动态长时段任务中对已移动目标的识别率显著高于对手。

**⚠️ 局限性**

局限性包括：对视觉-语言感知的依赖导致对感知误差敏感；场景变化只能在再次观测到时检测，可能延迟更新；恢复策略仅覆盖特定操作失败与目标移动，缺乏更通用的变更检测与多任务恢复。

---

## 356. When Metrics Reward the Worst Translations: Internalizing Cultural Reasoning for Social Media Translation Evaluation

**arXiv ID:** 2609.08156 | [PDF](https://arxiv.org/pdf/2609.08156v1)

**作者:** Yiwen Qiu `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究社交媒体翻译质量评估，发现传统指标因缺乏文化知识而失效，提出CuRIL框架让模型在内部推理中自我学习文化判断。

**💡 创新点**

创新点在于将文化提示作为梯度屏蔽的前缀注入模型内部，并通过线性衰减的课程学习方式，迫使模型在推理阶段自行掌握文化推理，而非依赖推理时外部提示或检索。

**🔧 技术方法**

采用GRPO强化学习框架，结合文化提示注入、token‑level梯度屏蔽、线性衰减策略和多维奖励（分数偏差与二分类一致性），并在Qwen系列大语言模型上实现。

**📊 数据集**

使用1,444条中文‑英文社交媒体翻译样本的人工标注评估（训练集13,128样本，验证集1,444样本），以及在MENT、RedTrans‑Bench等外部基准上的泛化测试。

**📈 对比分析**

与传统指标（COMET、XCOMET、BERTScore）以及开源/闭源LLM评估进行对比；CuRIL在Qwen3‑8B上获得Cohen κ=0.370、EM=45.22%，与30倍参数的Gemini‑3.1‑Pro相近；在下游翻译优化中将低质量率从25.6%降至4.9%。

**⚠️ 局限性**

局限性包括：仍需人工生成提示用于训练；在小模型规模下效果不如提示注入；仅验证中文‑英文社交媒体，跨语言或更广泛文化场景的泛化需进一步验证；对新出现的网络流行语或极端文化差异的鲁棒性有待提升。

---

## 357. SciFigure2Code: An AI-Reconstructed Benchmark for Scientific Figure-to-Code

**arXiv ID:** 2609.08155 | [PDF](https://arxiv.org/pdf/2609.08155v1)

**作者:** Wentao Li `[一作]` (Tsinghua University), Xiaonan Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出 SciFigure2Code 基准，目标是从最终发表的科学图像自动恢复可执行、可编辑的 Python 绘图程序，并通过 Codex 代理自动生成、审计和可视化验证得到银标准参考包；构建了 6,740 个面板参考包与 337 个平衡测试集。

**💡 创新点**

创新点在于：①将最终面板视为可审计的呈现对象，而非隐藏数据或作者代码的逆向问题；②设计了多阶段 Codex 代理流程（拆分、生成、审计、精炼、审核），实现可执行、可视化一致的银标准；③通过执行门控与 VLM 盲评估相结合，提供可衡量的、可重现的图形重建评估。

**🔧 技术方法**

技术手段包括：Codex CLI agents（基于 GPT‑5.4）负责面板拆分、代码生成、视觉审核与精炼；本地 Qwen‑vision 模型做质量评分；plan‑then‑code 两阶段提示；执行门控保证生成代码可跑；Qwen3.6‑27B 作为盲评判者评估布局、组件、图例、清晰度。

**📊 数据集**

数据来源于开放获取的 Nature 系列期刊论文（生物、环境、地理、材料、物理），提取完整图像并拆分为单面板，过滤后得到 6,740 个统计可视化面板；从中按域、复杂度、子类型平衡采样得到 337 个 SciFigureBench 测试集。

**📈 对比分析**

比较方法：在 image‑only 与 caption‑assisted 两个零样本设置下，评估 14 个模型（11 开源 VLM、3 Claude API）的执行、布局、组件、图例、清晰度四项盲评，计算 Overall 分数。结果显示 Claude Opus 4.7/4.6 最高，开源 GLM‑4.5V 与 Qwen3.5‑122B‑A10B 表现最佳；caption 辅助提升多数模型得分，plan‑then‑code 提升所有四个模型。整体平均得分约 70，执行成功率约 70%，但高复杂度面板仍难以恢复。

**⚠️ 局限性**

局限性包括：①对高复杂度面板（多组件、科学符号、注释）恢复仍不佳；②模型依赖 caption 或 plan‑then‑code 的效果因模型而异；③无法恢复原始数据或作者代码，重建结果仅为可编辑可视化的近似；④评估依赖 VLM 判别，可能带主观性；⑤缺少对不同绘图库或语言的通用性研究。

---

## 358. MRI-Guided Reslice-Refined Cross-Slice SDF Reconstruction of the Left Ventricle from Cardiac MRI with Sparse Axial Supervision

**arXiv ID:** 2609.08148 | [PDF](https://arxiv.org/pdf/2609.08148v1)

**作者:** Quanxin Zheng `[一作]` (China Electronics Information Technology Research Institute Company Limited), Shuai Zhao `[通讯]` (China Electronics Information Technology Research Institute Company Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于隐式符号距离场（SDF）的单病人专属三维左心室内膜表面重建框架，能够仅用少量弱标签的轴向切片进行重建。

**💡 创新点**

创新点在于：①将几何初始化与MRI图像边缘信息对齐的正则化相结合；②使用可微分的切片一致性损失（Dice + 边界一致性）保持与观测切片的吻合；③实现了对不同弱标签生成器（TransUNet、nnU-Net、Medical SAM3）均适用的无群体先验方法。

**🔧 技术方法**

核心技术包括：多分辨率哈希网格编码的隐式SDF表示；梯度幅值的MRI边缘场正则化；基于soft Dice和软轮廓一致性的切片一致性损失；PCHIP插值、锚点约束和外部极点约束等初始化手段。

**📊 数据集**

使用了MM-WHS 2017挑战的20例标注的磁共振心脏图像，实验中将不同弱标签生成器在每例上采用留一法训练或直接使用预训练模型。

**📈 对比分析**

在稀疏16张切片的条件下，所提方法对TransUNet、nnU-Net、Medical SAM3三种弱标签源分别达到了 Dice 为 0.906±0.042 / 0.928±0.025 / 0.928±0.040，HD95 为 6.41±4.70 mm / 3.81±2.25 mm / 3.80±2.49 mm；相较于同源的全模板拟合方法 GHD+DVS，所提方法在 Dice 与 HD95 上均有提升；在稀疏密度 4→64 的实验中，性能在稀疏 16 张切片后趋于饱和。

**⚠️ 局限性**

主要局限包括：仅在20例数据上评估，缺乏跨中心、跨扫描仪和病理多样性的验证；不同弱标签生成器的训练方式差异导致直接比较受限；当前实现对每例仍需数百秒的优化时间，未达到实时推理水平；未涉及右心室、心肌或全心重建；对最终阈值选择和表面提取仍依赖固定零等值；

---

## 359. SoftRerank: Hierarchical Soft Fusion with Candidate-Label Reranking for Long-Tailed Micro-Action Recognition

**arXiv ID:** 2609.08221 | [PDF](https://arxiv.org/pdf/2609.08221v1)

**作者:** Yichi Zhang `[一作]` (University of Science and Technology of China), Shengping Liu `[通讯]` (Unisound AI Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种结合 InternVideo2.5 全微调、层次软融合与轻量候选标签重排序的微动作识别框架。

**💡 创新点**

创新点包括全微调视觉模型以捕捉细微运动、使用层次软融合保持粗细粒度一致性、以及利用难样本与混淆标签的候选重排序器。

**🔧 技术方法**

采用 InternVideo2.5 视觉模型、类别平衡采样、逆频率加权、少数类增强、层次 Mixup、层次软融合和轻量候选标签重排序器（视频-标签匹配 MLP）。

**📊 数据集**

使用 MA-52 微动作数据集（52 细粒度动作、7 体位组）。

**📈 对比分析**

在 MA-52 1,138 样本子集上取得 F1_mean 79.99%，排名第一，优于前沿方法约 2.24 点；在完整测试集也获得最高 F1_mean。

**⚠️ 局限性**

局限在于对极端长尾类别的提升有限、重排序仅在低置信样本有效、以及对大规模 GPU 训练资源的高需求。

---

## 360. DRIFT: Removing Diffusion Watermarks by Deflecting the Generative Trajectory

**arXiv ID:** 2609.08213 | [PDF](https://arxiv.org/pdf/2609.08213v1)

**作者:** Rui Bao `[一作]` (University of New South Wales), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为DRIFT的黑箱攻击方法，结合了部分前向扩散和随机反向重采样，以有效去除扩散水印。

**💡 创新点**

创新点在于识别了扩散水印的轨迹一致性作为共同的攻击面，并通过自适应的强度选择和验证者门控的细化来提高攻击成功率和图像质量。

**🔧 技术方法**

使用了扩散模型的前向和反向采样技术，结合信息论和Wasserstein源依赖界限。

**📊 数据集**

在九种水印上进行了实验，涵盖了三种不同的嵌入范式，使用了Stable Diffusion v2.1作为公共扩散骨干。

**📈 对比分析**

与现有的黑箱攻击和去除攻击进行了比较，DRIFT在攻击成功率上达到了98-100%，并在图像质量上表现最佳，且无需秘密密钥或每图像的梯度优化。

**⚠️ 局限性**

局限性在于攻击的保证依赖于未验证的网络级Lipschitz和终端前提，且评估仅使用了一个公共的潜在扩散骨干。

---

## 361. QoS-Aware RACH Preamble Slicing via Quota-Projected Branching Deep Reinforcement Learning

**arXiv ID:** 2609.08199 | [PDF](https://arxiv.org/pdf/2609.08199v1)

**作者:** Jiulin Guo `[一作]` (Shenyang University of Technology), Lei Zhang `[通讯]` (Shenyang University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于配额投影的分支深度强化学习控制器QP-BD3QN-RACH，用于混合的两步和四步争用随机接入，旨在实现QoS感知的随机接入前导切片。

**💡 创新点**

创新点在于将分支决策与确定性配额投影相结合，生成非负整数的前导分配，同时保持随机接入预算。

**🔧 技术方法**

使用了分支对抗双重DQN（Dueling Double DQN）技术来选择特定池的乘数，并通过确定性配额投影将其转换为可执行的前导分配。

**📊 数据集**

使用了五种到达负载的模拟数据集进行评估，涵盖了不同的到达负载和交叉方法比较。

**📈 对比分析**

与四种比较方法进行了交叉比较，结果显示在五种负载下，QP-BD3QN-RACH在成功率、碰撞率、回退率、阻塞率和成功接入延迟等指标上均表现出积极的方向一致性差异，成功率提高了5.74到8.21个百分点。

**⚠️ 局限性**

限制在于采用的模拟和比较设计，未考虑物理层捕获、信号干扰加噪声比（SINR）、功率控制、多用户解码等因素，未来工作应扩展到多种种子交叉方法比较和更广泛的流量及到达过程。

---

## 362. OntologyBench: Can Dense Retrieval Satisfy Structured Biomedical Constraints?

**arXiv ID:** 2609.08174 | [PDF](https://arxiv.org/pdf/2609.08174v1)

**作者:** Xiao Yu Cindy Zhang `[一作]` (University of British Columbia), Jian Zhu `[通讯]` (University of British Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍了OntologyBench，这是一个分层的生物医学检索基准，包含471,854个训练和125,744个评估查询-文档相关性对，涵盖概念基础、关系检索和组合表型基础检索。

**💡 创新点**

创新点在于构建了一个多层次的检索基准，评估嵌入模型是否能够恢复由生物医学本体编码的兼容性，并识别出在关系和组合任务中存在的性能差距。

**🔧 技术方法**

使用了密集检索模型和嵌入模型，包括通用嵌入模型和生物医学嵌入模型，采用了两阶段检索评估方法。

**📊 数据集**

数据集来源于专家策划的生物医学本体，包括MONDO疾病本体、HPO表型本体和HGNC基因名称，共构建了471,854个训练和125,744个评估查询-文档相关性对。

**📈 对比分析**

与现有的本体感知参考方法相比，评估的嵌入模型在组合检索任务上的性能显著低于本体感知基线，且重新排序和基于LLM的候选评分方法在端到端性能上几乎没有改善。

**⚠️ 局限性**

限制在于OntologyBench只评估了在受控生物医学兼容性约束下的检索，未能捕捉到现实世界诊断的更广泛方面，如时间进展、因果机制和患者特定背景。

---

## 363. Key Path Identification for Resolving Knowledge Conflicts via SAE-based Steering

**arXiv ID:** 2609.08173 | [PDF](https://arxiv.org/pdf/2609.08173v1)

**作者:** Wenbo Zhang `[一作]` (Renmin University of China), Jun Xu `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的方法Key Path Identification (KPI)，用于识别关键的引导特征，以解决知识冲突问题，提升大语言模型的准确性和可解释性。

**💡 创新点**

KPI方法通过识别具有强因果依赖关系的关键特征，转变了基于稀疏自编码器(SAE)的引导方式，从数量驱动转向质量驱动，减少了冗余特征的影响。

**🔧 技术方法**

使用了稀疏自编码器(SAE)和因果依赖分析技术来识别关键特征和路径。

**📊 数据集**

使用了NQ-Swap和Macnoise数据集来评估模型的引导效果。

**📈 对比分析**

与现有的基于SAE的质量引导方法（如STA和SPARE）相比，KPI在RAG任务中平均提高了18%的准确性，且有效过滤了冗余特征，减轻了副作用。

**⚠️ 局限性**

方法的局限性在于基于相关性的方法固有的缺陷，以及在长上下文分析中反向传播电路寻找方法的适用性不足，可能会导致因果特征的丢失。

---

## 364. Dual-Layer Semantic-Spatial Belief Mapping for Aerial Object Goal Navigation

**arXiv ID:** 2609.08164 | [PDF](https://arxiv.org/pdf/2609.08164v1)

**作者:** Jianqiang Xiao `[一作]`, Liqiang Nie `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为AeroBelief的双层语义-空间信念映射框架，用于无人机在未知户外环境中进行目标导航。

**💡 创新点**

创新点在于将瞬态的视觉-语言模型输出转化为持久的空间指导，通过分离广泛的上下文可信度和特定目标的证据来提高导航的有效性。

**🔧 技术方法**

使用了对象条件的视觉推理、保守的证据资格、证据门控融合和自我中心的区域指导等技术。

**📊 数据集**

在UAV-ON基准数据集上进行了实验，该数据集专门用于无人机目标导航的评估。

**📈 对比分析**

与多种方法进行了比较，包括随机选择、CLIP-H、AOA-F等，AeroBelief在成功率(SR)、Oracle成功率(OSR)和成功路径长度(SPL)上均表现最佳，分别达到21.61%、35.57%和10.62。

**⚠️ 局限性**

局限性在于当前评估仅限于AirSim模拟，尚未验证模拟到现实的转移，且OSR与SR之间的差距表明可靠的目标确认和停止仍然是重要挑战。

---

## 365. Generalized DBLog: A Verified Contract for Interleaving Database Rows with a Change Log

**arXiv ID:** 2609.08160 | [PDF](https://arxiv.org/pdf/2609.08160v1)

**作者:** Andreas Andreakis `[一作]` `[通讯]`, Andreas Andreakis

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种改进的变更数据捕获（CDC）方法，解决了在将现有行与活动日志合并时可能出现的复制到日志的交接问题。

**💡 创新点**

创新点在于提出了一种通用的DBLog框架，能够在不需要全局数据库快照的情况下，确保复制的数据库状态与已提交的变更日志的正确合并。

**🔧 技术方法**

使用了水印技术来标识每次读取的变化，并通过机器检查工具Isabelle/HOL和Lean 4对理论进行了验证。

**📊 数据集**

使用了Netflix开发的DBLog系统的设计，并对Debezium和Flink CDC等开源项目进行了适应和扩展。

**📈 对比分析**

通过与经典的水印算法进行比较，证明了通用DBLog在处理不同表和键范围时的有效性，确保了在不同时间读取的行的状态能够正确重建。

**⚠️ 局限性**

限制在于该方法依赖于数据库的日志保留策略，且在某些情况下可能需要额外的协调来确保复制和日志的正确合并。

---

## 366. Function Tables for Secure Distributed Matrix Multiplication

**arXiv ID:** 2609.08154 | [PDF](https://arxiv.org/pdf/2609.08154v1)

**作者:** Rafael G. L. D'Oliveira `[一作]` (Clemson University), Divyesh Vaghasiya `[通讯]` (University of South Florida)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种函数表的概念，用于安全分布式矩阵乘法（SDMM）方案中的系数函数的逐项表示，研究了在外积分区下的隐私和可解性。

**💡 创新点**

创新点在于引入了函数表作为统一框架，能够描述和比较不同的线性SDMM方案，并且确定了在不同条件下所需的最小工人数量。

**🔧 技术方法**

使用了线性编码和线性解码的技术，结合了函数表的构建方法来实现隐私和可解性。

**📊 数据集**

使用了有限域上的矩阵数据集，具体的工人数量和字段大小依赖于K和L的值。

**📈 对比分析**

与现有方法相比，提出的方案在T=1时需要KL+K+L个工人（q≥3时），在T=2时需要KL+K+L+2个工人，且在特定条件下达到了最优工人数量。

**⚠️ 局限性**

限制在于需要满足特定的MDS条件，且在某些情况下，字段的大小限制了方案的可行性。

---

## 367. SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents

**arXiv ID:** 2609.08149 | [PDF](https://arxiv.org/pdf/2609.08149v1)

**作者:** Pujun Zheng `[一作]` (East China Normal University), Qi Zhang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SWE-Bench Pro Verified，这是一个针对软件工程代理的基准，包含731个经过修正的实例，旨在评估大型语言模型（LLM）在反黑客执行环境中的表现。

**💡 创新点**

创新点在于引入了反黑客控制和任务修正流程，解决了原有基准中存在的奖励黑客和任务质量问题，从而提高了评估的有效性。

**🔧 技术方法**

使用了反黑客控制技术，包括本地和网络隔离、元数据匿名化和源主机阻断，以及通过LLM辅助的任务修正流程。

**📊 数据集**

使用的数据集为SWE-Bench Pro Verified，包含731个实例，并通过公共问题报告收集了119个候选实例进行修正。

**📈 对比分析**

与原始的SWE-Bench Pro基准进行比较，结果显示反黑客控制有效地阻止了所有观察到的黑客尝试，且修正后的任务使许多之前无法解决的任务变得可解。

**⚠️ 局限性**

限制在于域名阻止列表可能无法覆盖所有自托管的Git服务，评估数据清理可能在某些存储库中留下少量残余信息，且修正过程可能未能识别每个任务质量问题。

---

## 368. Routing Dense Layouts with History-Aware Offline Reinforcement Learning using LSTM

**arXiv ID:** 2609.08232 | [PDF](https://arxiv.org/pdf/2609.08232v1)

**作者:** Afsara Khan `[一作]` (New York University), Austin Rovinski `[通讯]` (New York University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种历史感知的离线强化学习策略，用于在高密度布局下改进详细路由的收敛性和质量。

**💡 创新点**

创新点在于结合了轻量级的LSTM架构和额外特征，以保持序列上下文，从而在多种密度和路由指导质量下改善路由收敛性。

**🔧 技术方法**

使用了保守Q学习（CQL）和LSTM网络的离线强化学习技术。

**📊 数据集**

使用了来自OpenROAD设计套件的8个Nangate45设计，涵盖了6种不同的布局密度。

**📈 对比分析**

与基线方法相比，该策略在高密度和低指导质量的情况下，平均减少了92%的设计规则违规（DRV），同时减少了10%的运行时间。

**⚠️ 局限性**

限制在于该方法在极端条件下的推理开销可能会影响总的速度提升，但在更复杂的路由情况下，推理成本相对较小。

---

## 369. SE-GoS: Self-Evolving Graph-of-Skills for Skill Library at Scale

**arXiv ID:** 2609.08228 | [PDF](https://arxiv.org/pdf/2609.08228v1)

**作者:** Dawei Fu `[一作]` (Peking University), Zhongkai Hao `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了自我演化技能图（SE-GoS）框架，通过执行反馈改进技能检索图，解决了静态图的瓶颈问题。

**💡 创新点**

创新点在于利用每次执行记录的信号进行三种训练无关的更新，分别是拓扑诱导与修剪、边权重强化和节点内容优化。

**🔧 技术方法**

使用了反向感知的个性化PageRank（PPR）技术来读取图的边权重，并进行检索。

**📊 数据集**

在SkillsBench数据集上进行评估，该数据集包含1000个技能和87个真实世界技术任务。

**📈 对比分析**

与静态GoS、向量检索和SkillDAG等基线方法进行比较，SE-GoS在奖励上提高了约7.0%，并且在输入令牌上减少了32%。

**⚠️ 局限性**

局限性在于只进行了单轮更新，重复更新会导致性能饱和和过拟合，且评估范围仅限于特定数据集和模型。

---

## 370. PhysFlow: Physics-Aware Optical Flow for Motion Controllable Video Generation

**arXiv ID:** 2609.08215 | [PDF](https://arxiv.org/pdf/2609.08215v1)

**作者:** Cong Wang `[一作]` (Chinese Academy of Sciences), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为PhysFlow的物理感知视频生成框架，通过将视频生成分解为运动感知光流生成和运动条件外观合成两个阶段，来提高生成视频的物理合理性。

**💡 创新点**

创新点在于通过物理感知光流来建模运动模式，从而实现对物理动态的更好理解，并且构建了PhysVideo数据集以支持模型训练。

**🔧 技术方法**

使用了物理感知注意力模块和流引导视频生成模型，分别用于生成光流视频和合成最终视频。

**📊 数据集**

使用了PhysVideo数据集，该数据集包含10K个前景物体和50K个带有运动和材料属性注释的真实视频序列。

**📈 对比分析**

与现有方法相比，PhysFlow在物理合理性和视觉保真度上表现优越，尤其在运动一致性和时间平滑性方面超越了其他视频生成模型，并与物理引擎方法相当。

**⚠️ 局限性**

局限性在于模型对输入的物理属性和先验条件的依赖，若这些输入不准确，可能导致生成结果的物理行为不合理。

---

## 371. Qiushi Engine on AstaBench E2E-Bench-Hard

**arXiv ID:** 2609.08196 | [PDF](https://arxiv.org/pdf/2609.08196v1)

**作者:** Wenhao Li `[一作]`, Yihao Yang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本报告分析了Qiushi Engine v0.8在AstaBench E2E-Bench-Hard基准测试中的表现，涵盖了40个测试任务，评估了其在科学研究中的完整性和有效性。

**💡 创新点**

创新点在于Qiushi Engine能够在复杂的科学研究任务中实现较高的完成率，特别是在任务的完整性和报告的生成方面，超越了现有的官方代理。

**🔧 技术方法**

使用了DeepSeek作为模型后端，并结合了AstaBench的评分机制和Meta-Trace内存系统来记录和分析研究过程。

**📊 数据集**

使用的数据集为AstaBench E2E-Bench-Hard基准测试中的40个任务，涉及多种AI/NLP研究领域。

**📈 对比分析**

与其他方法相比，Qiushi Engine在40个任务中实现了0.816的平均得分，完美完成率为10%，显著高于官方代理的约3%。

**⚠️ 局限性**

局限性包括在统计支持、外部依赖和消融研究等特定维度上的得分缺失，表明在资源分配和依赖管理方面仍有改进空间。

---

## 372. An Elementary Proof of the $\widetilde O(n^{1/3})$ Bound for Separating Words

**arXiv ID:** 2609.08191 | [PDF](https://arxiv.org/pdf/2609.08191v1)

**作者:** Chen Xu `[一作]` `[通讯]`, Chen Xu

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了分离字问题，旨在找到一个小的确定性有限自动机，该自动机仅接受两个不同的二进制字中的一个。

**💡 创新点**

提出了一种替代的直接证明方法，给出了显式的状态界限O(n^1/3(log n)^(7/3))，改进了Chase的O(n^(1/3)log^7 n)的界限。

**🔧 技术方法**

使用了有限差分方法和二阶实数递归的截断技术。

**📊 数据集**

没有具体提到使用的数据集，但研究对象是长度为n的二进制字。

**📈 对比分析**

与Chase的方法进行了比较，Chase的方法使用复杂分析的稀疏多项式估计，而本研究通过有限差分直接证明了可分性估计，最终得到了更明确的状态界限。

**⚠️ 局限性**

研究没有寻求更好的n的幂次，分离集框架存在n^(1/3)的规模障碍。

---

## 373. Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Efficient LLM Inference

**arXiv ID:** 2609.08189 | [PDF](https://arxiv.org/pdf/2609.08189v1)

**作者:** Hongjin Lin `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种动态路由框架HeRo，通过引入路由器记忆机制来减少大型语言模型的推理成本，允许为每个token选择跳过的层。

**💡 创新点**

创新点在于引入了一个显式的路由状态，利用线性注意力机制在模型深度上维护先前路由决策的状态，从而改善了动态路由的准确性和适应性。

**🔧 技术方法**

使用了线性注意力机制来构建路由器记忆，并在每个路由层上进行动态路由决策。

**📊 数据集**

在Llama 3.1-8B、Llama 2-7B和Llama 2-13B模型上进行了实验。

**📈 对比分析**

与十个基线方法进行比较，HeRo在所有基线中表现最佳。在Llama 3.1-8B上，HeRo在跳过26.87%的模型参数的情况下，达到了100.24%的稠密模型性能，并在更严格的计算预算下保留了97.01%的性能。

**⚠️ 局限性**

限制在于HeRo的性能依赖于路由历史的维护，去除路由历史会导致性能下降，尤其是在多步推理和代码生成任务上。

---

## 374. NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness

**arXiv ID:** 2609.08183 | [PDF](https://arxiv.org/pdf/2609.08183v1)

**作者:** NeoHorse Team `[一作]`, Yingjie Zong `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文介绍了一种基于路由引导的代理训练方法，旨在实现递归自我改进（RSI），通过代理模型的交互记录来优化模型的学习过程。

**💡 创新点**

创新点在于将代理模型的交互记录转化为训练示例，并通过能力反馈引导数据分配，从而实现模型的自我改进。

**🔧 技术方法**

使用了路由引导的监督微调（SFT）和在线政策蒸馏（OPD）等技术，结合了多种模型和执行环境。

**📊 数据集**

使用了来自代理执行的交互轨迹数据集，包含用户请求、模型响应、工具调用和环境观察等信息。

**📈 对比分析**

与其他方法相比，经过代理后训练的模型在多个基准测试中表现出显著的性能提升，4B模型的宏平均分数从58.94提高到64.87，9B模型从65.60提高到69.04。

**⚠️ 局限性**

限制在于当前的验证主要集中在代理和编码能力上，尚未评估更广泛的能力范围，未来需要测试模型在多代迭代中的持续改进能力。

---

## 375. Less Is Personal: Learning Minimal Sufficient User Profiles for Personalized Language Models

**arXiv ID:** 2609.08180 | [PDF](https://arxiv.org/pdf/2609.08180v1)

**作者:** Minghang Liu `[一作]` (Institute of Computing Technology), Xueqi Cheng `[通讯]` (Institute of Computing Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ENOUGH的方法，通过最小化用户历史记录的个性化配置，来提高大语言模型的输出准确性和用户偏好对齐性。

**💡 创新点**

创新点在于引入了最小充分个性化的概念，动态构建用户配置文件的长度，并通过离线反事实搜索评估配置文件的有效性。

**🔧 技术方法**

使用了长远决策控制器和反事实评估技术，结合用户特定性和代币成本进行优化。

**📊 数据集**

实验使用了LaMP基准数据集，涵盖六个个性化任务，包括分类和文本生成任务。

**📈 对比分析**

与八个检索增强基线进行比较，ENOUGH在所有任务和指标上均表现最佳，且在有效性和效率上均优于强基线，减少了不必要的上下文成本。

**⚠️ 局限性**

限制在于该方法的适用性可能受到用户历史记录的动态变化和交互设置的影响，未来工作将扩展到这些领域。

---

## 376. EviSI: An Evaluation Agent for Simultaneous Interpreting

**arXiv ID:** 2609.08171 | [PDF](https://arxiv.org/pdf/2609.08171v1)

**作者:** Ben Yan `[一作]` (Huawei), Yuzhe Shang `[通讯]` (Xiamen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的评估代理EviSI，用于同时语音翻译的质量评估，旨在理解、翻译和口头表达，同时源流持续进行。

**💡 创新点**

EviSI结合了多维质量指标（MQM）的错误分析和惩罚原则，能够更好地评估语义保真度和口头表达质量，超越了现有的评估基准。

**🔧 技术方法**

使用了大型语言模型（LLM）技术，结合了共享源证据、专业判断和重整过程，进行确定性评分。

**📊 数据集**

使用了五个英语到中文（EN→ZH）和四个中文到英语（ZH→EN）的语料库，包含来自194和145个源的1164和870个输出，提供了人工评分。

**📈 对比分析**

与人类系统排名的平均一致性超过了评估的基准，EviSI在EN→ZH方向的Kendall一致性为0.707，在ZH→EN方向为0.467，表现优于BLEU和COMET等传统评估方法。

**⚠️ 局限性**

存在的局限性包括个别输出结果不一致，历史注释和执行元数据不完整，缺乏独立测试和评估的可重复性，且文本评估未能直接测量声学质量或听众理解能力。

---

## 377. Monkey See, Can Monkey Do? A Benchmark for Evaluating Robot Skill Learning by Observation

**arXiv ID:** 2609.08209 | [PDF](https://arxiv.org/pdf/2609.08209v1)

**作者:** Weiwei Gu `[一作]` (Arizona State University), Nakul Gopalan `[通讯]` (Arizona State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RoboReel，一个统一的基准，用于评估从人类视频中学习策略的模型，涵盖十个操作任务的真实人类演示视频、模拟机器人轨迹和评估环境。

**💡 创新点**

RoboReel是首个提供配对训练数据和评估环境的统一基准，解决了现有方法缺乏可比性的问题，并引入了四个测试套件来评估模型的性能。

**🔧 技术方法**

使用生成方法重建真实世界物体的3D网格，并构建配对的真实世界环境与模拟环境，以确保训练和测试阶段的视觉一致性。

**📊 数据集**

收集了来自六名参与者的真实人类演示视频，涵盖十个不同任务，总计超过七小时的数据，数据在干扰和无干扰条件下进行收集。

**📈 对比分析**

通过四个测试套件评估模型性能，发现视频条件的策略在大多数任务中表现最佳，尤其是在长时间任务中，现有模型仍面临挑战。

**⚠️ 局限性**

方法的局限性包括仅使用基于目标的任务，缺乏双手任务的评估，以及在复杂环境中精确操作的困难。

---

## 378. Non-Coherent Over-the-Air Federated Learning: Protocol, Convergence, and Device Scheduling

**arXiv ID:** 2609.08312 | [PDF](https://arxiv.org/pdf/2609.08312v1)

**作者:** Haifeng Wen `[一作]` (Hong Kong University of Science and Technology), Hong Xing `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种非相干的空中联邦学习协议（NCAirFL），旨在解决无线接入网络中的可扩展性瓶颈，通过模拟信号叠加实现模型聚合。

**💡 创新点**

NCAirFL协议不依赖于瞬时的信道状态信息（CSI），通过引入二进制抖动和无偏非相干检测来简化信号处理，理论上保证了与理想通信的FedAvg相同的收敛速度。

**🔧 技术方法**

使用了二进制抖动、无偏非相干检测和长期误差反馈机制等技术。

**📊 数据集**

在MNIST和CIFAR-10数据集上进行了实验验证。

**📈 对比分析**

与FedAvg相比，NCAirFL在实际设置中表现出接近的学习性能，且通过优化的设备调度策略显著加快了收敛速度。

**⚠️ 局限性**

NCAirFL在处理数据和无线资源异质性时的性能可能受到限制，尤其是在设备选择和功率控制的优化方面。

---

## 379. A Measurement Study of LLM Inference Trade-offs Across Edge Continuum Hardware

**arXiv ID:** 2609.08307 | [PDF](https://arxiv.org/pdf/2609.08307v1)

**作者:** Maysam Khatib `[一作]` (University of Cyprus), Marios D. Dikaiakos `[通讯]` (University of Cyprus)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文进行了一个控制测量研究，评估了在边缘和近边缘部署节点上自托管的大型语言模型（LLM）推理的权衡，包括NVIDIA Jetson AGX Orin和近边缘服务器的CPU和GPU推理模式。

**💡 创新点**

创新点在于提出了一种基于测量的方法论，比较了模型选择、量化、执行平台和流式传输延迟对准确性、响应性、模型占用和能耗的影响，并通过Pareto前沿分析识别了在准确性和延迟方面的次优配置。

**🔧 技术方法**

使用了容器化的基准测试管道，记录了准确性、模型占用、每个令牌的解码延迟、预填充延迟和整体执行能量。

**📊 数据集**

使用了MMLU（大规模多任务语言理解）数据集，评估了多个开放权重的LLM和量化变体。

**📈 对比分析**

与云托管的GPT-4o进行比较，结果显示GPU服务器执行提供最低的计算侧延迟，而Jetson Orin在能耗方面表现更好。CPU-only执行在延迟上始终处于劣势，且能耗较高。通过Pareto前沿分析，发现计算侧推理指标可能导致延迟敏感的交互式Web服务的次优部署选择。

**⚠️ 局限性**

限制在于本研究专注于顺序交互式问答，没有评估并发请求流、批处理、排队或尾延迟，因此报告的排名不应直接推广到高吞吐量的多租户服务。此外，MMLU工作负载仅涵盖了事实和推理准确性，不包括长上下文对话、检索增强生成、工具使用或多模态任务。

---

## 380. TRIUNE-Net: Harmonizing Scale, Shape, and Efficiency in Pancreatic Tumor Segmentation

**arXiv ID:** 2609.08303 | [PDF](https://arxiv.org/pdf/2609.08303v1)

**作者:** Amir Hossein Saleknia `[一作]` (Independent Researcher), Alaa Sulaiman `[通讯]` (Abdora Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种轻量级的统一架构TRIUNE-Net，用于胰腺肿瘤的3D CT图像分割，旨在解决胰腺和肿瘤的极端尺度变化和不规则形态带来的挑战。

**💡 创新点**

TRIUNE-Net通过三项协同创新实现了尺度、形状和效率的统一，分别是多尺度上下文聚合模块、串行线性可变形注意力机制和信息保留下采样模块。

**🔧 技术方法**

使用了多尺度上下文聚合模块、串行线性可变形注意力机制和信息保留下采样模块等技术。

**📊 数据集**

使用了MSD胰腺数据集和NVD胰腺数据集进行评估，前者包含281个腹部CT体积，后者是500个合成CT体积。

**📈 对比分析**

与八种3D分割模型进行比较，TRIUNE-Net在所有关键肿瘤指标上均超越了所有基线模型，特别是在肿瘤Dice上超出下一个最佳模型0.45%，在F1分数上超出6.0点，在灵敏度上超出6.6点，在精确度上超出3.4点，显示出其在临床环境中的有效性。

**⚠️ 局限性**

限制在于尚未在更大规模的数据集上进行验证，未来的工作将包括在更多小病灶任务和更大数据集上的验证。

---

## 381. Tracking-by-detection in Multi-object Tracking: Survey and Experiments

**arXiv ID:** 2609.08265 | [PDF](https://arxiv.org/pdf/2609.08265v1)

**作者:** Yujin Yang `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多目标跟踪中的基于检测的跟踪（TBD）方法进行了系统的综述和实验评估，分析了不同设计组件对多目标跟踪系统性能的影响。

**💡 创新点**

创新点在于通过逐步整合各个功能类别中最有效的方法，建立了一个强大的基线跟踪器，并提供了设计可靠的TBD跟踪器的实用见解。

**🔧 技术方法**

使用了卡尔曼滤波器、相似度度量、数据关联机制、相机运动补偿和后处理技术等多种技术。

**📊 数据集**

使用了MOT17、MOT20和DanceTrack三个数据集进行评估，这些数据集涵盖了不同的运动动态、物体密度和场景复杂性。

**📈 对比分析**

通过与现有的TBD跟踪器进行比较，提出的方法在所有评估的数据集上均表现出显著的性能提升，尤其是在DanceTrack数据集上，HOTA得分提高了10.01。

**⚠️ 局限性**

限制在于许多现有的跟踪器在评估时使用不一致的协议，导致难以进行公平比较，且在某些复杂场景下的鲁棒性评估仍然有限。

---

## 382. CircuTutor: Transforming Static Circuit Problems into Intelligent and Dynamic Tutoring

**arXiv ID:** 2609.08254 | [PDF](https://arxiv.org/pdf/2609.08254v1)

**作者:** Ziyu Luo `[一作]` (Beijing Technology and Business University), Xiaoming Chen `[通讯]` (Beijing Technology and Business University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了CircuTutor，一个基于电路状态的智能辅导系统，将静态教科书电路问题转化为互动辅导工作流程。

**💡 创新点**

创新点在于通过多模态问题解析和电路仿真，生成电路状态动画、因果推理、误解诊断和自适应后续练习，提供个性化反馈。

**🔧 技术方法**

使用了SPICE兼容的电路仿真技术，结合多模态问题解析和交互式学习环境。

**📊 数据集**

使用了静态教科书电路问题作为数据集，通过解析将其转化为结构化电路任务。

**📈 对比分析**

与传统学习方法相比，CircuTutor在后测和迁移测试中表现出显著更高的成绩，且学习者报告的心理努力更低，学习动机更高。

**⚠️ 局限性**

限制在于需要进一步研究更大和更具多样性的学习者群体，以及不同电路结构和误解类别的影响。

---

## 383. Variational Bayesian Data Detection for Multiuser MIMO Systems Corrupted by Phase Noises

**arXiv ID:** 2609.08252 | [PDF](https://arxiv.org/pdf/2609.08252v1)

**作者:** Toan-Van Nguyen `[一作]` (San Diego State University), Duy H. N. Nguyen `[通讯]` (San Diego State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种变分贝叶斯框架，用于在上行多用户多输入多输出（MIMO）系统中联合估计相位噪声（PN）和数据检测。

**💡 创新点**

通过将发射机的相位噪声直接吸收到发射信号中，解决了传统分离方法中的后验不匹配问题，从而提高了检测性能。

**🔧 技术方法**

使用变分贝叶斯（VB）方法，采用von Mises先验进行相位噪声的建模和数据检测。

**📊 数据集**

使用了模拟生成的信号数据集，考虑了不同的信道条件，包括独立同分布（i.i.d.）的瑞利衰落和相关的瑞利衰落。

**📈 对比分析**

与自干扰白化（SIW）算法和传统的相位噪声无关的检测器相比，提出的VB算法在广泛的信道条件、调制阶数和PN强度下实现了更低的符号错误率，同时在大规模MIMO部署中保持了计算可扩展性。

**⚠️ 局限性**

在高相位噪声条件下，尽管提出的方法表现良好，但仍可能面临计算复杂性和对信道模型假设的依赖等限制。

---

## 384. zScore-N: A Neural Network for On-Chain Wallet Reputation Scoring

**arXiv ID:** 2609.08247 | [PDF](https://arxiv.org/pdf/2609.08247v1)

**作者:** Girish G N `[一作]` (Zeru AI), Dhanashekar Kandaswamy `[通讯]` (Ohio State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的钱包声誉评分系统zScore-N，使用神经网络替代传统的手写公式，以提高评分的准确性和鲁棒性。

**💡 创新点**

创新点在于通过神经网络实现了可微分的评分函数，能够处理缺失数据并消除系统性偏差，同时保持高精度。

**🔧 技术方法**

使用了神经网络技术，特别是通过掩码增强训练来处理缺失值。

**📊 数据集**

使用了2019年至2024年间采样的5208952个钱包的数据集，确保了训练数据的丰富性和代表性。

**📈 对比分析**

与线性回归和梯度提升树相比，zScore-N在相同特征和分割上表现出更低的均方根误差（RMSE），分别为0.58、28.04和2.25，显示出显著的性能提升。

**⚠️ 局限性**

限制在于模型无法恢复从未捕获的信息，且在极端稀疏数据情况下的表现可能不如预期。

---

## 385. Tool Retrievers Are Underestimated: Annotation Expansion Reveals True Capability

**arXiv ID:** 2609.08327 | [PDF](https://arxiv.org/pdf/2609.08327v1)

**作者:** Yanyu Zhu `[一作]` (Tsinghua University), Hai-Tao Zheng `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的工具检索框架Tool Equivalent eXpansion，旨在解决现有工具检索基准中一对一注释的问题，自动发现和注释功能上等价的工具组合。

**💡 创新点**

创新点在于通过自动化的三阶段注释管道，扩展了每个查询的有效工具组合，从而纠正了现有基准对功能等价工具的低估。

**🔧 技术方法**

使用了深度学习模型（如Qwen3-Embedding-4B和GPT-4o-mini）进行查询分解、候选工具检索和功能验证。

**📊 数据集**

应用于7360个查询的基准数据集，扩展了每个查询的有效工具组合，平均每个查询有5.3个有效组合。

**📈 对比分析**

通过扩展的基准重新评估了八个基础检索器和两个微调变体，发现NDCG@10的得分平均提高了5-7个百分点，表明一对一注释系统性低估了检索器的能力。

**⚠️ 局限性**

局限性包括：验证工具等价性是基于语义而非实际调用，可能引入假阳性；扩展标签的全面验证尚未进行；虽然已在多个基准上应用，但对其他生态系统的推广仍需验证。

---

## 386. AttnCompress: Dynamic Attention-Guided Trajectory Compression for Software Engineering Agents

**arXiv ID:** 2609.08318 | [PDF](https://arxiv.org/pdf/2609.08318v1)

**作者:** Zhengran Zeng `[一作]` (Peking University), Shikun Zhang `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种动态注意力引导的轨迹压缩框架，旨在解决自主软件工程（ASE）代理在处理复杂任务时的上下文可扩展性瓶颈。

**💡 创新点**

创新点在于通过结构感知分段、代理注意力引导的相关性估计和动态滚动窗口机制，确保在压缩上下文的同时保留语法完整性和语义依赖性。

**🔧 技术方法**

使用了动态注意力引导的轨迹压缩框架，结合了PPL（困惑度）峰值检测和代理模型的注意力分布。

**📊 数据集**

在两个数据集上进行了广泛评估，包括真实的GitHub问题数据集和多种编程语言的任务数据集。

**📈 对比分析**

与七个基线方法进行比较，结果显示该框架的通过率为53.17%，在减少21.6%的令牌消耗和33.6%的总成本的同时，优于现有的最先进方法。

**⚠️ 局限性**

局限性在于可能仍会丢失某些在后续推理中变得重要的信息，导致代理在某些情况下无法正确决策。

---

## 387. Supervised Cross-Modal Feature Alignment for Zero-Wearable Freezing of Gait Detection in Parkinsonism

**arXiv ID:** 2609.08317 | [PDF](https://arxiv.org/pdf/2609.08317v1)

**作者:** Aryan Singh `[一作]` (NeuroAI Fusion Labs), Chandan Biswas `[通讯]` (NeuroAI Fusion Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种监督式跨模态特征对齐框架，用于在帕金森病患者中检测步态冻结（FoG），旨在解决可穿戴传感器的依赖性和视觉跟踪中的自遮挡问题。

**💡 创新点**

创新点在于通过监督对比学习将可穿戴传感器的运动数据和临床元数据引导到视觉模型中，从而在不依赖硬件传感器的情况下实现高精度的FoG检测。

**🔧 技术方法**

使用了运动增强的时空图卷积网络（ST-GCN）和监督对比学习（SupCon）技术。

**📊 数据集**

使用了一个公开的多模态FoG数据集，该数据集包含35名帕金森病患者的同步视频和惯性测量数据。

**📈 对比分析**

与传统的可穿戴传感器方法相比，提出的方法在推理时仅依赖视觉数据，达到了85.5%的准确率和90.6%的特异性，优于现有的视觉检测方法。

**⚠️ 局限性**

当前框架的局限性在于仅在35名患者的单一队列上进行评估，未来需要在更广泛的病理变异中进行多中心验证。此外，优化阶段依赖于特权的传感器和文本数据，限制了其应用的灵活性。

---

## 388. Agent ATO: Visualizing Agent Interaction Timelines from Logs

**arXiv ID:** 2609.08301 | [PDF](https://arxiv.org/pdf/2609.08301v1)

**作者:** Takuto Kawamoto `[一作]` (University of Osaka), Raula Gaikovina Kula `[通讯]` (University of Osaka)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Agent ATO 工具，基于控制台日志将 AI 编码代理的交互轨迹可视化为时间轴，并提供四种过滤视图（发现、读取、写入、执行）和令牌使用图。

**💡 创新点**

创新点包括：①将代理行为定义为从 LLM 响应到工具调用的可观测交互单元；②利用规则匹配将交互标记为四类并在时间轴中混合色显示；③在时间轴下方绘制令牌使用图，区分 LLM 输出与工具结果，从而揭示令牌消耗来源。

**🔧 技术方法**

使用 TypeScript 扩展收集 Pi 代理事件日志，Python 脚本解析日志、生成交互标签、绘制静态 HTML 可视化；前端实现悬停高亮、点击展开详情、连续序列搜索等交互功能。

**📊 数据集**

实验数据来自 Pi 编码代理在 GitHub 上的两条实际 issue（immich #28383 与 metabase #74530）的 10 次重复执行日志，日志中包含 LLM 消息、工具调用、bash 命令、工具结果、令牌使用及时间戳。

**📈 对比分析**

通过两组案例研究比较相同任务下多次执行轨迹：①在相同修复方向的两次尝试中，时间轴揭示了编辑-验证循环与无验证差异；②令牌使用图区分了由长 LLM 输出导致的峰值与由大文件读取导致的峰值。案例显示可视化能有效定位过程级别差异，虽未给出定量性能指标，但证明了对日志审计与提示调试的帮助。

**⚠️ 局限性**

局限性包括：仅在单一 Pi 代理上测试；使用手工规则标签，标签准确性未验证；案例规模有限，未进行广泛的可视化效果与调试效率评估；仅在固定任务与环境下验证，缺乏跨代理、跨任务的通用性评估。

---

## 389. Convolutional Codes from Cyclic Codes with Guaranteed Free and Local Minimum Distances

**arXiv ID:** 2609.08296 | [PDF](https://arxiv.org/pdf/2609.08296v1)

**作者:** Khaled Abdel-Ghaffar `[一作]`, Shu Lin `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出利用完整秩循环码链（如 BCH、Reed–Muller 码）构造多速率、低约束长度的卷积码，并给出了其生成矩阵、自由距离下界以及滑动窗口逐步退化（SWSC）解码方案。

**💡 创新点**

创新点在于：①定义完整秩代码链并利用其 t‑fold 分解实现可组合的卷积码；②通过该链构造的卷积码在保证局部最小距离≥母码距离的同时，能够显式给出自由距离下界；③提出基于局部码块的滑动窗口成功消解解码算法，兼顾低延迟与错误传播抑制。

**🔧 技术方法**

使用的技术包括：循环码的 t‑fold 分解、生成多层卷积码的矩阵形式、码链的全秩判定、基于 BCH 与 RM 码的代数根结构来构造码链，以及滑动窗口逐步退化的成功消解算法。

**📊 数据集**

实验与分析基于理论推导与符号级仿真，使用的典型码有二进制 BCH 码（如 (255,139)、(1023,737)）和 Reed–Muller 码（如 (255,93)、(1023,385)）等；未使用公开数据集。

**📈 对比分析**

通过与传统单循环码构造的卷积码（约束长度相同或更小）对比，本文卷积码在相同或更小约束长度下实现更高的局部最小距离与自由距离下界；示例给出自由距离至少为母码距离或 2·双码距两者中的较小者，说明性能优势。

**⚠️ 局限性**

局限性包括：①需要满足严格的完整秩条件，导致生成矩阵设计复杂；②滑动窗口成功消解易受误码传播影响，需在硬件实现中加入错误恢复机制；③在高吞吐量或实时场景下，解码复杂度与硬件资源需求仍需进一步评估。

---

## 390. HypLTSF: A Hyperbolic Geometric View of Multi-Scale Hierarchies for Long-Term Time Series Forecasting

**arXiv ID:** 2609.08286 | [PDF](https://arxiv.org/pdf/2609.08286v1)

**作者:** Namwoo Kim `[一作]` (Korea Advanced Institute of Science and Technology), Yoonjin Yoon `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了HypLTSF框架，通过将多尺度时间序列的表示嵌入到庞加莱球中，明确建模多尺度层次结构，以提高长期时间序列预测的准确性。

**💡 创新点**

创新点在于将多尺度层次结构以几何形式显式建模，并引入了两个层次损失：径向排序损失和角度一致性损失，以增强嵌入几何的层次性。

**🔧 技术方法**

使用了超几何空间（庞加莱球）来嵌入多尺度表示，并通过径向和角度几何约束来塑造嵌入结构。

**📊 数据集**

在八个真实世界的基准数据集上进行了实验，包括ETT、Weather、Electricity、Traffic和Solar等。

**📈 对比分析**

与多种基线方法（如TimeKAN、RAFT、TimeMixer等）进行了比较，HypLTSF在所有八个数据集上均表现出色，特别是在MAE指标上，显示出显著的性能提升。

**⚠️ 局限性**

局限性在于模型的复杂性和计算开销，尽管在准确性上表现优异，但在某些情况下可能需要更多的计算资源。

---

## 391. Physics of Information Geometry - Part I: Principle of Least Action on the Probability Simplex

**arXiv ID:** 2609.08285 | [PDF](https://arxiv.org/pdf/2609.08285v1)

**作者:** C. Emre Koksal `[一作]` (Ohio State University), Deniz Sargun `[通讯]` (Amazon.com)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了一种最小作用框架，用于描述概率分布如何在受限增量变化下从平衡状态演变到指定的非平衡状态。

**💡 创新点**

创新点在于将相对熵的毕达哥拉斯结构与信息几何结合，提供了一个统一的视角来理解分布演变，并通过信息投影构建贪婪的最小作用路径。

**🔧 技术方法**

使用了信息几何、热力学和相对熵的结合，特别是通过拉姆伯特W函数来表征每个投影的闭式形式。

**📊 数据集**

研究中使用了Gibbs分布作为平衡参考，并考虑了在概率单纯形上的离散运动。

**📈 对比分析**

通过与传统的最小击中时间问题进行比较，提出的信息投影路径在每一步都最大化自由能的进展，且在给定的动能预算下，保证了有限时间的性能。

**⚠️ 局限性**

限制在于该方法未必是全局最优的最小击中时间策略，且在不同的状态成本下，可能需要进一步的研究来理解其与全局最优性的关系。

---

## 392. SAM3-O2D2: Zero-Shot Object Out-of-Distribution Detection by Object Class Prompting of the SAM3-Image Model

**arXiv ID:** 2609.08281 | [PDF](https://arxiv.org/pdf/2609.08281v1)

**作者:** Lucas Görnhardt `[一作]` (Technische Universität Braunschweig), Tim Fingscheidt `[通讯]` (Technische Universität Braunschweig)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的零-shot对象超出分布（OOD）检测方法，利用基础模型进行高效的OOD检测。

**💡 创新点**

创新点在于直接在预测级别进行OOD检测，而不是在特征空间中进行，从而提高了鲁棒性并降低了计算成本。

**🔧 技术方法**

使用了基础模型进行文本提示，结合对象检测器的预测类进行OOD检测。

**📊 数据集**

使用了Pascal-VOC和BDD100K作为ID数据集，以及MS-COCO和OpenImages作为OOD数据集。

**📈 对比分析**

与当前最先进的零-shot方法相比，提出的方法在AuROC和FPR95指标上显著超越了现有方法，并且计算复杂度更低。

**⚠️ 局限性**

方法的局限性在于在开放世界设置中可能会增加误报率，尤其是在ID和OOD对象可能共存的情况下。

---

## 393. MemForest: Efficient Agent Memory Management via EventTree Partitioning and Progressive Merging

**arXiv ID:** 2609.08273 | [PDF](https://arxiv.org/pdf/2609.08273v1)

**作者:** Junxi Wang `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MemForest的通用记忆压缩框架，旨在提高代理记忆系统的存储和检索效率。

**💡 创新点**

创新点在于结合了全局语义相似性和局部时间连续性来对历史记忆进行事件中心的独立单元划分，并引入了锚点引导的传播检索机制以提高检索准确性。

**🔧 技术方法**

使用了最大生成树结构（EventTree）和锚点引导的传播检索机制（AGPR）等技术。

**📊 数据集**

在Mem0框架下使用了LoCoMo、LongMemEval和PersonaMem三个基准数据集，在M3-Agent框架下使用了M3-Bench-robot和M3-Bench-web两个基准数据集。

**📈 对比分析**

与多种基线方法进行比较，MemForest在压缩50%历史记忆的情况下，分别保留了97.1%和99.7%的原始性能，并实现了1.89倍和2.24倍的检索加速。

**⚠️ 局限性**

限制在于现有方法主要集中在生成阶段，无法根本性地解决记忆的持续增长问题，且对图结构记忆表示的依赖限制了其通用性。

---

## 394. Synergistic Fusion of Topological Structure and Temporal Semantics of Mobility for Urban Region Embedding

**arXiv ID:** 2609.08268 | [PDF](https://arxiv.org/pdf/2609.08268v1)

**作者:** Namwoo Kim `[一作]` (Korea Advanced Institute of Science and Technology), Yoonjin Yoon `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的城市区域嵌入框架MoSS，结合了人类移动的时间序列数据和时间演变的连接图的zigzag持久性图。

**💡 创新点**

创新点在于首次将zigzag持久同调应用于城市区域嵌入，并通过共享-私有分解和多度交互模块来捕捉跨视图的协同信号。

**🔧 技术方法**

使用了时间卷积网络（TCN）和zigzag持久同调技术。

**📊 数据集**

使用了来自纽约市和芝加哥的真实世界数据集，包括180个普查区和77个社区区域的出租车行程记录。

**📈 对比分析**

与六种基线方法进行比较，MoSS在犯罪预测、收入预测和服务呼叫预测任务中均表现出色，超越了依赖辅助模态的基线，且在参数数量上显著更少。

**⚠️ 局限性**

限制在于该方法仅依赖于移动数据，未来可以扩展到其他模态、较长时间尺度和跨城市转移。

---

## 395. Revisiting Spectral Representations in Generative Diffusion Models

**arXiv ID:** 2609.08253 | [PDF](https://arxiv.org/pdf/2609.08253v1)

**作者:** Yuehao Wang `[一作]` (University of Texas at Austin), Zhangyang Wang `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本论文研究了自监督光谱表示学习与扩散生成模型之间的联系，提出了一种自监督光谱表示对齐方法，以促进扩散模型的训练。

**💡 创新点**

创新点在于通过共享的扰动核视角，揭示了自监督光谱表示学习与扩散模型之间的基本联系，并提出了一种新的训练策略，将光谱正则化整合到扩散训练目标中。

**🔧 技术方法**

使用了自监督学习、光谱表示学习和扩散模型的结合技术，特别是通过扰动核进行的光谱对齐。

**📊 数据集**

使用了多个数据集进行实验，包括CIFAR10、CelebA、FFHQ、ImageNet以及ShapeNet等，涵盖了图像和3D点云生成任务。

**📈 对比分析**

与基线方法相比，提出的方法在多个数据集上均表现出一致的生成质量提升，特别是在图像生成任务中，FID指标显著降低，表明生成质量提高。

**⚠️ 局限性**

限制在于实验主要集中在相对小规模的数据集和模型上，且与依赖外部教师信号的REPA方法相比，性能仍有差距。此外，方法引入了额外的训练开销，增加了训练时间。

---

## 396. Agentic ML Exploration (A-MLE) for Ads Ranking

**arXiv ID:** 2609.08248 | [PDF](https://arxiv.org/pdf/2609.08248v1)

**作者:** Erwin Gao `[一作]` (Meta Platforms), Ritwik Tewari `[通讯]` (Meta Platforms)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Agentic ML Exploration (A-MLE)的自主LLM代理系统，旨在系统性地探索广告排名模型中的机器学习技术。

**💡 创新点**

A-MLE通过将机器学习迭代分解为五个阶段，显著提高了工业广告排名系统的迭代效率，尤其是针对那些很少受到专家关注的模型。

**🔧 技术方法**

使用了大型语言模型（LLM）作为代理，结合领域特定的技能库和沙盒执行层，自动化了机器学习探索的各个阶段。

**📊 数据集**

在一个代表性的工业广告排名模型组合上进行了部署和评估，具体模型被匿名标识为M_1, M_2等。

**📈 对比分析**

与手动和半自动化基线相比，A-MLE在完成的迭代数量、训练成功率和提案接受率等方面表现出显著的性能提升，尤其是在长尾模型上。

**⚠️ 局限性**

存在一些局限性，包括偶尔出现的虚构API调用、基线漂移、基础设施脆弱性等，这些问题可能导致候选技术的提前放弃或错误的评估。

---

## 397. CS-CLIP: Compositional Scene Graph-guided CLIP for Robust Compositional Reasoning

**arXiv ID:** 2609.08242 | [PDF](https://arxiv.org/pdf/2609.08242v1)

**作者:** SeongJun Jeong `[一作]` (Seoul National University), Byoung-Tak Zhang `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的视觉语言模型CS-CLIP，旨在解决现有模型在组合推理中的元素特定偏差问题。

**💡 创新点**

创新点在于通过场景图解析生成针对每个组合元素的困难负样本，从而增强模型的组合推理能力。

**🔧 技术方法**

使用了场景图解析、掩码语言模型（MLM）和自然语言推理（NLI）技术。

**📊 数据集**

使用了VG-Relation、VG-Attribute和COCO等数据集进行评估。

**📈 对比分析**

与现有的组合性模型（如NegCLIP、StructureCLIP和TripletCLIP）相比，CS-CLIP在组合推理基准上表现出色，且在不同元素变化中保持了较高的准确性和敏感性。

**⚠️ 局限性**

局限性在于依赖现成的场景图解析器、MLM和NLI模块，可能会传播这些模块的固有语言偏见。

---

## 398. Style Over Substance: Content-Invariant Wrappers Flip LLM Safety-Judge Verdicts

**arXiv ID:** 2609.08236 | [PDF](https://arxiv.org/pdf/2609.08236v1)

**作者:** Yongxi Zhou `[一作]` (Northeastern University), Junwei Yao `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了自动安全评判系统（如Llama Guard和GPT-4o）是否根据回复的内容或其表述方式进行评分。通过在固定回复内容前后添加风格包装，研究了这些包装如何影响评判结果。

**💡 创新点**

创新点在于提出了一种控制的风格包装攻击方法，专注于安全评判者的表现，而非模型本身，揭示了特定评判者的盲点和可被利用的漏洞。

**🔧 技术方法**

使用了内容不变的风格包装技术，通过在回复前后添加固定字符串来改变语气，同时保持回复内容不变。

**📊 数据集**

使用了来自公共JailbreakBench的600个回复（300个真实有害的和300个拒绝的），涵盖四个目标模型（GPT-3.5-turbo、GPT-4、Vicuna-13B、Llama-2-7B）。

**📈 对比分析**

通过配对显著性测试和置信区间比较不同评判者的翻转率，发现大多数评判者的翻转率很低，但特定评判者存在明显的盲点。例如，GPT-4o-mini在使用token拒绝包装时，其正确的“有害”判决翻转率达到19.9%。

**⚠️ 局限性**

限制在于仅使用了九种包装，且评判者使用的是中等性能的模型，未能涵盖所有可能的评判者和包装组合。此外，基于Jailbreak-artifact集的安全排名不稳定，需更广泛的基线分离以确认翻转效应。

---

## 399. Width-Bounded Equational Derivations for Finite Graph Expressions

**arXiv ID:** 2609.08325 | [PDF](https://arxiv.org/pdf/2609.08325v1)

**作者:** Antonios Kalampakas `[一作]` `[通讯]` (American University of the Middle East), Antonios Kalampakas (American University of the Middle East)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文探讨了有限图表达的等式路径的完整性与资源控制之间的关系，提出了一个新的界限，称为有界等式一致性，证明了在特定条件下，等价的闭合表达可以通过有限的推导步骤连接。

**💡 创新点**

创新点在于引入了有界等式一致性这一概念，证明了在给定的边字母表下，闭合表达的推导宽度可以被有效地限制，从而在图的大小不影响推导的资源使用。

**🔧 技术方法**

使用了结构性马戈伊德法则和十五个有限图形方案，结合了图的结构和推导过程中的宽度控制。

**📊 数据集**

使用了有限的双重排名边字母表Σ，具体数据集未明确提及，但涉及到的图形包括了多点超图和特定的图形结构。

**📈 对比分析**

通过与现有方法的比较，展示了在特定条件下，推导的宽度可以被有效控制，且在图的大小不影响推导的资源使用，性能上表现出更优的资源管理。

**⚠️ 局限性**

限制在于该方法的适用性可能受到图的结构和边的数量的影响，且未能提供一个通用的推导长度界限。

---

## 400. Exploring Bottom-Up Clustering for Creating Semantic IDs

**arXiv ID:** 2609.08310 | [PDF](https://arxiv.org/pdf/2609.08310v1)

**作者:** Leah Woldemariam `[一作]` (Cornell University), Ali Sahami `[通讯]` (PayPal)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种生成语义ID（Semantic IDs）的方法，确保标识符唯一且保留原始嵌入的结构。

**💡 创新点**

创新点在于采用自下而上的聚类方法，保留嵌入空间中的局部结构，从而提高生成语义ID的聚类质量和下游检索的实用性。

**🔧 技术方法**

使用了自下而上的聚类算法来生成语义ID，并通过标准的聚合聚类方法进行层次合并。

**📊 数据集**

使用了亚马逊产品评论数据集和一个包含约600万项的自定义数据集，数据集包含产品标题、图像生成的描述、类别信息和品牌信息。

**📈 对比分析**

与RQ-VAE方法进行比较，底部聚类方法在所有数据集上都获得了更高的轮廓分数和更好的下游任务表现（Recall@10和NDCG@10），显示出改进的语义ID结构能够转化为更好的检索性能。

**⚠️ 局限性**

限制在于方法的复杂性和对数据集的依赖性，可能在某些特定情况下表现不佳。

---

## 401. HoneyRoute: Honeypot-Model Routing for Adversarial LLM Serving

**arXiv ID:** 2609.08306 | [PDF](https://arxiv.org/pdf/2609.08306v1)

**作者:** Han Jin `[一作]` `[通讯]` (Independent Researcher), Han Jin (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种推理服务层，能够检测传入请求是否恶意，并将恶意请求路由到专用的蜜罐模型，从而保护生产服务并持续收集对手的交互信息。

**💡 创新点**

创新点在于首次在推理API层面拦截请求，区分恶意与良性请求，并将恶意请求引导至蜜罐模型，而不是其他防护层或策略。

**🔧 技术方法**

使用了流式路由器、双实现蜜罐（规则/提示工程蜜罐和专用同家族模型）以及一个将捕获的交互转化为攻击者指纹的分析循环。

**📊 数据集**

使用了生产追踪和七个领域的攻击种子语料库。

**📈 对比分析**

与现有的两级守卫LLM级联进行比较，路由器在0.5决策阈值下达到了91.8%的恶意召回率（AUROC 0.975；在95%召回率下假阳性率为0），并且在38毫秒的中位延迟下匹配了96%的F1分数，且在并发洪水攻击下将生产模型的令牌消耗减少了97.8%。

**⚠️ 局限性**

局限性在于检测与保真度之间的权衡：蜜罐越忠实于生产模型，泄露的行为信号就越少，反之亦然。

---

## 402. LEBGen: An LLM-Enhanced Bayesian Network Framework for Few-Shot Travel Survey Data Generation

**arXiv ID:** 2609.08288 | [PDF](https://arxiv.org/pdf/2609.08288v1)

**作者:** Zijian Shen `[一作]` (University of Hong Kong), Jintao Ke `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为LEBGen的框架，用于从少量旅行调查数据生成合成数据，旨在改善旅行行为分析和交通规划中的数据不足问题。

**💡 创新点**

创新点在于结合了大语言模型（LLM）和贝叶斯网络（BN），通过识别旅行者角色和修正网络结构来增强合成数据的生成质量。

**🔧 技术方法**

使用了贝叶斯网络（BN）和大语言模型（LLM）技术，LLM用于识别旅行者角色并指导BN结构的修正。

**📊 数据集**

使用了2022年香港旅行特征调查（TCS）数据集，该数据集包含家庭、个人和旅行层面的信息。

**📈 对比分析**

与其他生成模型（如高斯Copula、CTGAN、TVAE和MTabGen）进行比较，LEBGen在分布保真度和依赖保真度上均表现优越，平均JSD从0.0671降至0.0091，Cramér's V误差减少14.3%。

**⚠️ 局限性**

局限性包括评估仅基于单一调查，缺乏跨城市和调查设计的广泛验证，且在重建详细的起止点关系方面仍面临挑战。

---

## 403. Adaptively Incorporating Directional Hints into Zeroth-Order Optimization

**arXiv ID:** 2609.08277 | [PDF](https://arxiv.org/pdf/2609.08277v1)

**作者:** Alexander Ryabchenko `[一作]` (University of Toronto), Wenlong Mou `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了利用方向提示进行非凸函数的零阶优化，提出了一种新的框架CV-ZOD，通过控制变量来改进经典的零阶梯度估计器。

**💡 创新点**

创新点在于引入了控制变量的零阶下降方法，能够根据方向提示自适应地调整，同时保持对提示质量的鲁棒性。

**🔧 技术方法**

使用了控制变量梯度估计器和自适应步长选择的技术。

**📊 数据集**

在流体动力学和计算化学的科学优化任务中进行了验证，使用了便宜的代理模型生成方向提示。

**📈 对比分析**

与经典的零阶下降方法（ZOD）和引导进化策略（GES）进行了比较，CV-ZOD在优化速度和最终误差上显著优于现有的零阶方法和依赖于偏差梯度估计的引导方法。

**⚠️ 局限性**

限制在于当前框架每次迭代只能处理一个低维的提示子空间，未来的工作可以探索如何有效地聚合多个代理梯度。

---

## 404. ACEA: An Adversarial Co-Evolution Arena for Head-to-Head Red-Team and Blue-Team LLM Testing

**arXiv ID:** 2609.08256 | [PDF](https://arxiv.org/pdf/2609.08256v1)

**作者:** Yi Ting Shen `[一作]` (Vulcan Research), Alex Leung `[通讯]` (Vulcan Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ACEA（对抗共进化竞技场），一个连接红队和蓝队适配器的平台，以共享的目标大语言模型（LLM）进行攻击和防御的评估。

**💡 创新点**

ACEA通过提供可插拔的适配器、可验证的泄露真相、实时可视化和上下文改进循环，解决了当前红蓝团队评估中的三个主要问题。

**🔧 技术方法**

使用了HTTP协议的ACEA标准适配器协议（ASAP），并通过LLM作为评判者进行评分。

**📊 数据集**

使用了合成的标准秘密作为数据集，以便在攻击和防御中提供可验证的真相。

**📈 对比分析**

与现有方法相比，ACEA提供了实时的对抗评估，能够在比赛进行中观察攻击和防御的动态，评分更为可信且可操作。

**⚠️ 局限性**

ACEA目前仅支持一对一的红队和蓝队对抗，且依赖于LLM评判者的判断，可能存在偏见和不一致性。

---

## 405. CALIPER: Clean Scenes Cannot Rank Physical Inference in Pretrained Visual Representations

**arXiv ID:** 2609.08250 | [PDF](https://arxiv.org/pdf/2609.08250v1)

**作者:** Aman Mehta `[一作]` (Independent Researcher), Riya Baviskar `[通讯]` (Independent Researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的评估方法CALIPER，用于测试视觉编码器在物体推滑预测中的物理推理能力。

**💡 创新点**

创新点在于通过校准和预测的方式，评估编码器是否能够从物体的交互历史中推断出质量和摩擦力，而不是依赖于固定场景中的简单评估。

**🔧 技术方法**

使用了线性回归、主成分分析（PCA）和多种视觉编码器（如V-JEPA 2、VideoMAE、VC-1等）进行特征提取和预测。

**📊 数据集**

在MuJoCo模拟环境中生成的数据集，包含不同质量和摩擦力的物体，通过标准的冰球进行撞击实验。

**📈 对比分析**

与传统的单参数扰动和线性探测方法相比，CALIPER能够更好地评估编码器的物理推理能力。在干净场景中，所有编码器的表现接近于模拟器的上限，而在干扰场景中，编码器的表现差异显著，V-JEPA 2的表现最佳。

**⚠️ 局限性**

局限性在于实验环境较小，仅包含刚性盒子和单一接触事件，且评估方法依赖于线性可用性，可能低估了模型的潜力。

---

## 406. CUNO: Curriculum and Preference Optimization for Stable Graph Unlearning under Mass Deletion

**arXiv ID:** 2609.08244 | [PDF](https://arxiv.org/pdf/2609.08244v1)

**作者:** Chenhan Zhang `[一作]` (University of Technology Sydney), Raymond Owen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出了一种名为CUNO的基于课程的图形遗忘框架，旨在在大规模删除情况下逐步移除遗忘集，以减轻灾难性遗忘现象。

**💡 创新点**

CUNO的创新点在于通过课程调度和分布级负偏好优化（NPO）相结合，逐步删除样本，避免了现有方法在大规模删除时的性能急剧下降。

**🔧 技术方法**

使用了课程调度和分布级负偏好优化（NPO）技术。

**📊 数据集**

在Cora、CiteSeer和PubMed等多个广泛使用的节点分类基准数据集上进行了实验。

**📈 对比分析**

与现有的四种遗忘方法（如梯度上升、GIF、INPO和ETR）进行比较，CUNO在20%删除率时保持了74%的原始效用，而现有方法的效用仅为26%到53%。即使在50%删除率时，CUNO仍保持超过一半的原始效用。

**⚠️ 局限性**

CUNO的局限性在于其依赖于课程设计的有效性，可能在某些特定情况下表现不佳，此外，课程调度和NPO的组合可能需要进一步的调优和验证。

---

## 407. SmartANN: Object Causal Modeling Boosts Approximate Nearest Neighbor Diagnosis and Auto-Design

**arXiv ID:** 2609.08240 | [PDF](https://arxiv.org/pdf/2609.08240v1)

**作者:** Yutong Zhou `[一作]` (University of Chinese Academy of Sciences), Jianfeng Zhan `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了SmartANN框架，用于近似最近邻（ANN）算法的瓶颈归因和自动设计，解决了现有方法无法追踪性能损失传播的问题。

**💡 创新点**

创新点在于使用对象因果模型（OCM）来表示ANN工作流，并通过顺序诊断和替换循环识别瓶颈对象，自动生成优化的ANN设计。

**🔧 技术方法**

使用对象因果模型（OCM）和顺序诊断-替换循环技术。

**📊 数据集**

在八个真实世界数据集上进行实验，包括SIFT、GloVe、MSong等，涵盖图像检索和文本嵌入等多种类型。

**📈 对比分析**

与现有的VDTuner框架相比，SmartANN在大多数数据集上提供了更优的Recall-QPS权衡，提升Recall 0.24-74.20%或QPS 28.8-256.5%，并且在端到端延迟上实现了2.9-42.0倍的加速。

**⚠️ 局限性**

局限性在于SmartANN的设计依赖于现有的ANN优化方法，可能无法处理所有类型的ANN算法或数据分布。

---

## 408. EMBLEM: Enhancing Multi-script Table Detection through Masking

**arXiv ID:** 2609.08330 | [PDF](https://arxiv.org/pdf/2609.08330v1)

**作者:** Dhruv Kudale `[一作]` (Indian Institute of Technology Bombay), Ganesh Ramakrishnan `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的多脚本表格检测方法，通过掩蔽技术来增强模型的性能。

**💡 创新点**

创新点在于引入了一个手动标注的多脚本数据集和基于掩蔽的微调范式，使得模型能够在没有多脚本训练数据的情况下，依然在多脚本文档上表现出色。

**🔧 技术方法**

使用了基于掩蔽的微调技术，结合了多种深度学习架构，如Faster R-CNN、Mask R-CNN和YOLO。

**📊 数据集**

使用了一个名为MANDALA的数据集，包含2323页文档，涵盖18种语言和15种脚本。

**📈 对比分析**

与现有方法进行比较，结果显示该方法在多脚本数据集上表现优异，F1分数提高了20.8%，同时在五个标准的英语主导基准上保持竞争力。

**⚠️ 局限性**

限制在于该方法未能解决表格结构识别或嵌套表格的问题，且当前的掩蔽策略依赖于固定阈值，可能在低质量或视觉退化的输入上表现不佳。

---

## 409. Tracing Stereotypes from Representation to Output in Multilingual LLMs

**arXiv ID:** 2609.08322 | [PDF](https://arxiv.org/pdf/2609.08322v1)

**作者:** Ariun-Erdene Tumurchuluun `[一作]` (Saarland University), Koel Dutta Chowdhury `[通讯]` (University of Technology Nuremberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究比较了多语言大型语言模型（LLMs）在不同语言中表现出的刻板印象相关行为，探讨了这些行为的内部机制，使用了线性探测、归因修补、稀疏自编码器（SAEs）和特征消融等方法。

**💡 创新点**

创新点在于区分了刻板印象相关信息的可解码性、对模型输出的影响以及特征级干预效果之间的关系，揭示了这些特征在不同语言和社会类别中的转移性。

**🔧 技术方法**

使用了线性探测、归因修补和稀疏自编码器（SAEs）等技术来分析模型的内部表示和行为。

**📊 数据集**

使用了MBBQ数据集进行层级分析和稀疏特征识别，SHADES数据集用于评估特征干预效果，涵盖了英语、西班牙语、荷兰语和土耳其语等多种语言。

**📈 对比分析**

比较了线性探测和归因修补的结果，发现探测性能在模型中层早期达到峰值，而归因影响在输出层达到峰值，二者之间存在36-53%的层深度差距。特征消融的效果在不同模型和语言中表现出异质性。

**⚠️ 局限性**

本研究的局限性包括仅分析了四种高资源语言，未考虑更低资源语言的情况；只评估了8B-9B参数模型，限制了对模型规模影响的评估；SAE套件的差异可能影响比较结果；特征消融未考虑偏见路径的冗余性；未解决交叉偏见（如年龄与性别的交互影响）。

---

## 410. MARS-CLIP: Multi-Resolution and Attention Refined Zero-Shot Image Segmentation

**arXiv ID:** 2609.08283 | [PDF](https://arxiv.org/pdf/2609.08283v1)

**作者:** Nagito Saito `[一作]` (Tohoku University), Takafumi Aoki `[通讯]` (Tohoku University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的零-shot语义分割框架MARS-CLIP，旨在通过多分辨率输入和注意力机制的改进来准确捕捉图像细节。

**💡 创新点**

创新点在于引入了多分辨率特征提取模块和注意力精炼机制，以克服输入分辨率限制和恢复物体边界。

**🔧 技术方法**

使用了多分辨率特征提取和注意力机制的改进技术。

**📊 数据集**

在六个公共数据集上进行了实验，包括PASCAL VOC 2012、PASCAL Context、ADE20K、Cityscapes、COCO-Object和COCO-Stuff。

**📈 对比分析**

与现有的最先进方法相比，MARS-CLIP在所有数据集上均表现出显著的性能提升，尤其是在高分辨率和细粒度对象的识别上。

**⚠️ 局限性**

在低光或低对比度场景中，颜色亲和偏差可能会降低性能，尽管中间层的空间偏差部分缓解了这一问题。

---

## 411. Three Types of Negation of Triple and its Elements and an Extension of Triple

**arXiv ID:** 2609.08271 | [PDF](https://arxiv.org/pdf/2609.08271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 412. Dreaming in Flow: Generative Grounding Feedback for Self-Evolving Unified Multimodal Models

**arXiv ID:** 2609.08282 | [PDF](https://arxiv.org/pdf/2609.08282v1)

**作者:** Ke Hao `[一作]` (Shanghai Jiao Tong University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为生成性基础反馈（GGF）的自我演化后训练框架，旨在通过模型自身的视觉经验将视觉理解和生成结合在一起。

**💡 创新点**

GGF的创新点在于通过共享的视觉经验和双向基础循环将理解和生成耦合起来，允许模型在没有配对图像-文本监督的情况下自我演化。

**🔧 技术方法**

使用了流级反馈和梦境重放基础等技术，结合了生成和理解的反馈机制。

**📊 数据集**

使用了1k个文本提示进行视觉梦境生成，以及1k个不同的提示用于梦境重放基础。

**📈 对比分析**

与其他方法相比，GGF在文本到图像生成和视觉理解上均表现出一致的改进，尤其在生成的物体计数、空间关系和属性绑定等方面有显著提升。

**⚠️ 局限性**

限制在于GGF依赖于模型自身生成的图像，可能会放大模型的错误，且需要确保生成的图像在语义上是可靠的。

---

## 413. Seeing is Not Believing: Breaking the Physical-to-Digital Trust Boundary in Robotics

**arXiv ID:** 2609.08280 | [PDF](https://arxiv.org/pdf/2609.08280v1)

**作者:** Leming Shen `[一作]` (University College London), Chris Xiaoxuan Lu `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文揭示了机器人操作系统（ROS）2中的一个严重漏洞，攻击者可以通过修改一个环境变量，拦截和注入传感器遥测数据，从而使机器人执行危险任务，同时向下游验证者伪造虚假的遥测数据。

**💡 创新点**

创新点在于发现了ROS 2的信任边界，攻击者可以在不修改官方二进制文件的情况下，通过用户空间的钩子拦截和修改消息，导致物理行为与遥测数据之间的不一致。

**🔧 技术方法**

使用了用户空间函数插入技术，通过预加载恶意共享库来拦截ROS 2消息，确保在不修改ROS 2本身的情况下进行攻击。

**📊 数据集**

在物理的Franka Emika机器人臂上进行实验，使用Secure ROS 2进行评估。

**📈 对比分析**

与AI检测器的比较显示，攻击在87%的情况下成功绕过检测，且在所有实验中攻击成功率达到100%。即使在启用SROS 2的情况下，攻击仍然能够成功进行。

**⚠️ 局限性**

限制在于现有的机器人安全机制无法保证遥测数据真实反映物理执行，且攻击依赖于对第三方Docker容器的广泛使用，可能导致软件供应链的暴露。

---

## 414. Online Signature Verification Using Augmented Path Signature and T-Mamba

**arXiv ID:** 2609.08276 | [PDF](https://arxiv.org/pdf/2609.08276v1)

**作者:** Ruiling Li `[一作]` (Chongqing University), Danyu Yang `[通讯]` (Chongqing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的在线签名验证框架，结合了增强路径签名（APS）描述符和T-Mamba模型，以提高签名验证的准确性。

**💡 创新点**

创新点在于引入了APS描述符和T-Mamba模型的结合，APS有效捕捉几何结构和非线性通道间交互，而T-Mamba模型则通过时间扫描机制建模局部和全局依赖关系。

**🔧 技术方法**

使用了增强路径签名（APS）描述符和T-Mamba模型，后者结合了时间卷积网络（TCN）和选择性状态空间模型（Mamba）。

**📊 数据集**

使用了三个公共基准数据集：MCYT-100、SVC-2004 Task 2和DeepSignDB。

**📈 对比分析**

与其他方法相比，提出的方法在所有测试数据集上均表现出色，尤其是在MCYT-100上，EER达到了0.56%，显著优于现有的最先进方法。

**⚠️ 局限性**

限制在于APS可能对手写签名中的噪声敏感，尤其是在指写签名的情况下，可能影响性能。

---

## 415. Evidence-Aligned Entity Verification for Hallucination Detection in Retrieval-Augmented Generation

**arXiv ID:** 2609.08267 | [PDF](https://arxiv.org/pdf/2609.08267v1)

**作者:** Runsong Jia `[一作]` (University of Technology Sydney), Yi Zhang `[通讯]` (University of Technology Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的基于检索增强生成（RAG）的幻觉检测方法，称为证据对齐实体验证（EAEV），旨在通过对齐生成的实体与检索到的证据上下文来检测实体级幻觉。

**💡 创新点**

创新点在于引入了多维度对齐和反事实稳定性分析，以区分真实的证据支持与表面相关的虚假匹配，从而提高幻觉检测的准确性和可靠性。

**🔧 技术方法**

使用了检索增强生成（RAG）技术，结合多维度对齐和反事实稳定性分析来进行实体级验证。

**📊 数据集**

在多个RAG基准数据集上进行实验，包括RAGTruth、HotpotQA和DelucionQA，使用了不同的语言模型作为基础。

**📈 对比分析**

与多种强基线方法（如SelfCheckGPT、Semantic Entropy等）进行比较，EAEV在多个评估设置中表现出色，尤其在LLaMA2-13B模型上达到了87.89%的AUROC，显示出强大的泛化能力和性能提升。

**⚠️ 局限性**

限制在于LLM输出在不同运行中可能表现不稳定，EAEV依赖于检索证据的质量和覆盖范围，缺失或不完整的检索可能限制检测性能。此外，虽然该方法专注于准确和可解释的幻觉检测，但并未直接解决生成过程中的幻觉问题。

---

## 416. Revoked but Still Authoritative: An Empirical Study of Revocation Enforcement in Agent-Memory Systems

**arXiv ID:** 2609.08258 | [PDF](https://arxiv.org/pdf/2609.08258v1)

**作者:** Yi Ting Shen `[一作]` (Vulcan Research), Alex Leung `[通讯]` (Vulcan Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了五种主要的代理记忆系统在软撤销情况下是否能够有效地防止返回被标记为撤销的事实。通过加载撤销政策和其替代品，跟踪撤销事实在检索时是否被返回，以及代理是否基于此采取行动，进行了测量。

**💡 创新点**

创新点在于提出了一种防护机制，该机制位于代理和内存后端之间，能够在检索时检查记录的有效性，并拒绝返回被撤销或与其替代品冲突的记录。

**🔧 技术方法**

使用了多种代理记忆系统（如Graphiti、Zep、mem0、langmem和cognee）进行实验，测量了它们在不同场景下的表现，并开发了一个防护机制来提高安全性。

**📊 数据集**

使用了五种不同的代理记忆系统进行实验，涵盖了九种政策场景和九种模型，评估了在不同防御条件下的表现。

**📈 对比分析**

与其他方法的比较显示，现有的系统在检索时未能有效执行撤销，导致代理在44.2%的试验中采取了不安全的行动。引入的防护机制在检索时能够有效阻止不安全的行动，尤其是在没有撤销标记的情况下。

**⚠️ 局限性**

限制在于现有的系统未能在所有情况下执行撤销，且在某些情况下，撤销的记录仍然被返回并影响代理的决策。防护机制的有效性依赖于后端是否能够提供撤销标记。

---

## 417. Stable Voting Rules on the Edge of Optimal Metric Distortion

**arXiv ID:** 2609.08259 | [PDF](https://arxiv.org/pdf/2609.08259v1)

**作者:** Ziyi Cai `[一作]` (Rutgers University), Qilin Ye `[通讯]` (Stanford University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了一种随机投票规则的存在，该规则的度量失真最多为2.13713，接近下限2.11264。

**💡 创新点**

该规则源于稳定k-彩票的推广，采用单一分布进行抽样，而不是在多个投票规则之间混合。

**🔧 技术方法**

使用了稳定k-彩票和g-稳定彩票的概念，结合了零和博弈的均衡分布。

**📊 数据集**

未具体提及使用的数据集，但涉及的投票规则和偏好聚合的背景与社会选择理论相关。

**📈 对比分析**

与之前的投票规则相比，性能显著提高，尤其是在度量失真方面，达到了接近最优的结果。

**⚠️ 局限性**

当前的下限来自于特定且不自然的选举，仍需进一步研究以缩小2.11264与2.13713之间的差距。

---

## 418. StitchOver: Technical Embroidery on Seamed Fabrics

**arXiv ID:** 2609.08311 | [PDF](https://arxiv.org/pdf/2609.08311v1)

**作者:** Zekun Chang `[一作]` (Cornell Tech), Thijs Roumen `[通讯]` (Cornell Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种技术，能够在缝合面料上可靠地进行功能性刺绣，解决了传统刺绣在缝合处常见的缺陷问题。

**💡 创新点**

创新点在于引入了一种新的机制，通过调整针点位置和增加支撑缝，避免了缝合处的干扰，从而消除了70%的缺陷。

**🔧 技术方法**

使用了一种软件工具，该工具能够自动数字化用户定义的刺绣图案，并生成适应缝合的刺绣路径。

**📊 数据集**

使用了多种缝合和图案配置的面料进行评估，包括PU皮革和导电线。

**📈 对比分析**

与传统刺绣方法相比，使用该技术的样本缺陷率显著降低，且在不同的机器状态下（如过紧、良好校准和过松）仍能保持良好的性能。

**⚠️ 局限性**

限制在于该机制的最小跳跃长度受限于压脚尺寸和缝合宽度，可能在复杂缝合交叉处仍然存在问题。

---

## 419. Subquadratic Subsidies for Nonnegative or Nonpositive Valuations

**arXiv ID:** 2609.08272 | [PDF](https://arxiv.org/pdf/2609.08272v1)

**作者:** Max Dupré la Tour `[一作]` (RIKEN Center for Advanced Intelligence Project), Mashbat Suzuki `[通讯]` (UNSW Sydney)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在非可加估值下，如何通过补贴实现无嫉妒分配，证明了在所有代理人对每个捆绑物赋予非负或非正值的情况下，总补贴为O(n^3/2√(log n))足以实现无嫉妒分配。

**💡 创新点**

首次提出了适用于任意数量代理人的次二次总补贴界限，解决了之前仅适用于素数幂数量代理人的问题。

**🔧 技术方法**

结合了支持捆绑价格、拓扑均衡论证和随机舍入等技术。

**📊 数据集**

使用了具有单项边际值在[-1,1]范围内的任意估值的有限非可分物品集。

**📈 对比分析**

与之前的研究相比，提出的补贴界限显著降低，且在所有代理人数量下均适用，性能表现优越。

**⚠️ 局限性**

限制在于当前方法依赖于特定的边界条件，未来需要探索如何扩展到混合符号估值的情况。

---

## 420. From Glance to Scrutiny: Progressive Distortion Reasoning for Fine-Grained Image Quality Assessment

**arXiv ID:** 2609.08316 | [PDF](https://arxiv.org/pdf/2609.08316v1)

**作者:** Aoting Zhang `[一作]` (Chinese Academy of Sciences), Yu Zhou `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GS-IQA框架，将图像质量评估(IQA)重新定义为逐步的Where–What–How诊断，模拟人类的感知过程，从初步观察到细致审查。

**💡 创新点**

GS-IQA通过逐步的失真推理，明确将每个受损区域与其失真类型和序数严重性关联，提供空间基础和可验证的质量解释。

**🔧 技术方法**

采用了基于群体相对策略优化（GRPO）的两阶段强化学习范式，结合感知门控奖励和在线奖励条件失真生成。

**📊 数据集**

构建了Diag-Bench，一个包含约25K个样本的区域级IQA基准，涵盖12种失真类型和五个序数严重性级别。

**📈 对比分析**

GS-IQA在失真定位、识别和严重性估计方面的性能超越了现有的最先进方法，且其诊断表示在多种外部基准上有效转移到传统的全局质量预测。

**⚠️ 局限性**

GS-IQA的局限性在于其依赖于准确的区域和失真类型识别，错误的定位或识别可能导致严重性信号的误导。

---

## 421. FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy

**arXiv ID:** 2609.08305 | [PDF](https://arxiv.org/pdf/2609.08305v1)

**作者:** Tingyin Zhao `[一作]` (Tsinghua University), Yuan Shen `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为FPicker的拓扑引导框架，用于在低信噪比的冷冻电子显微镜图像中自动追踪细丝，解决了现有方法在极低信噪比下的局限性。

**💡 创新点**

FPicker是首个将中心-端点表示与开放曲线演化模块结合的框架，显著提高了在极端噪声条件下的细丝追踪精度，减少了拓扑缺口率。

**🔧 技术方法**

使用了中心-端点表示和开放边界图卷积网络（GCN）技术，结合了全局结构锚定机制和辅助几何场学习。

**📊 数据集**

使用了Cryo-Sim生成的模拟数据集和Custom-EMPIAR真实数据集，前者包含20000幅微图像，后者包含500幅手动标注的微图像。

**📈 对比分析**

与现有的盒式检测、像素级分割和闭合主动轮廓方法进行比较，FPicker在极端噪声下的mSAP提高了超过40%，拓扑缺口率降低了60%以上，表现出色。

**⚠️ 局限性**

FPicker在处理分支（如Y型交叉）时存在局限性，且在某些情况下可能会过度合并对齐的片段，需进一步改进。

---

## 422. Human-Centric Image Captioning with Subject-Centered Spatial Understanding

**arXiv ID:** 2609.08300 | [PDF](https://arxiv.org/pdf/2609.08300v1)

**作者:** Bozhou Li `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的基准SPACE，用于评估人本图像描述中的主体中心空间理解，特别关注坐标框架转换问题。

**💡 创新点**

创新点在于引入了SPACE基准，专门针对主体中心的空间关系进行评估，并设计了一个可扩展的数据构建和训练框架。

**🔧 技术方法**

使用了YOLO进行姿态估计和SAM3进行部位分割，结合了两阶段的描述生成和重写流程，以及基于评分标准的奖励机制进行优化。

**📊 数据集**

使用了CapRL-5M和LAION-400M数据集进行图像收集，并构建了2981张图像和19892个评估关键点的标注数据集。

**📈 对比分析**

与现有的多模态大语言模型（MLLMs）进行比较，H-SPACE-GRPO在SPACE基准上表现出色，尤其在空间与方向类别中，精度和命中率均优于其他模型。

**⚠️ 局限性**

限制在于当前模型在处理复杂的主体中心空间关系时仍然存在不足，尤其是在左/右区分和解剖部位绑定方面的错误。

---

## 423. EvoNav-Bench: Benchmarking Lifelong Navigation in Evolving Environments

**arXiv ID:** 2609.08292 | [PDF](https://arxiv.org/pdf/2609.08292v1)

**作者:** Xilin Wang `[一作]` (State Key Laboratory of General Artificial Intelligence), Lifeng Fan `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了EvoNav-Bench，一个用于评估在环境演变中进行终身导航的基准，重点在于如何在不同导航阶段之间处理环境变化。

**💡 创新点**

创新点在于引入了环境演变的控制机制，使得先前的经验在导航中既有用又可能过时，从而揭示了现有方法在动态环境下的脆弱性。

**🔧 技术方法**

使用了ProcTHOR框架构建EvoNav-Bench，并对三种最近的导航方法（UniGoal、3D-Mem和MSGNav）进行了基准测试，同时比较了三种简单的启发式策略（Frontier-Update、Fail-then-Update和Stage-Reset）。

**📊 数据集**

使用了ProcTHOR-10k数据集，该数据集包含可编辑的室内场景，支持生成多阶段的物体导航任务。

**📈 对比分析**

与现有方法相比，启发式策略在处理环境演变时表现更好，尤其是FTU策略在目标重新定位时表现最佳，表明反应性修正策略在应对环境演变时是有效的。

**⚠️ 局限性**

EvoNav-Bench目前通过模拟物体重新定位来建模环境演变，可能无法完全反映现实世界的重新排列，并且不涵盖物体插入和移除、关节状态变化或人类活动数据的物体重新定位策略。

---

## 424. What Eviction Destroys: A Restore-Counterfactual Audit of Forgetting in Agent Memory

**arXiv ID:** 2609.08279 | [PDF](https://arxiv.org/pdf/2609.08279v1)

**作者:** Chen Shen `[一作]` `[通讯]` (Megagon Labs), Chen Shen (Megagon Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种恢复反事实的方法，用于分析外部代理记忆存储中的驱逐策略对任务准确性的影响。

**💡 创新点**

创新点在于引入恢复反事实度量，能够区分不可逆损失和可恢复错误，并提供每个问题的错误分类。

**🔧 技术方法**

使用了恢复反事实度量和三种错误分类（可恢复、不可逆、残余）来评估不同的驱逐策略。

**📊 数据集**

使用了LongMemEval-S数据集，该数据集包含约102k个标记的多会话历史问题。

**📈 对比分析**

通过对比FIFO、随机、冗余感知和LLM重要性等四种驱逐策略的表现，发现不可逆错误的比例在80k令牌预算下为0.67-0.73，而LLM重要性为0.60。在8k令牌预算下，所有策略的不可逆错误比例均为1.00。

**⚠️ 局限性**

研究的局限性在于只使用了单一基准、两个读者和一个主要评判者，且评判者与读者来自同一提供者。此外，审计需要黄金标签和可回答的过滤器，限制了其在基准分析中的使用。

---

## 425. Beyond Coherence: Benchmarking Professional Editing-Technique Execution in Multi-Shot Audio-Video Generation

**arXiv ID:** 2609.08275 | [PDF](https://arxiv.org/pdf/2609.08275v1)

**作者:** Tianyi Zeng `[一作]` (Shanghai Jiao Tong University), Long Qin `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CutCraft基准，旨在评估多镜头音视频生成中的编辑技术执行能力，扩展了结构化的多镜头提示，加入了明确的编辑规范。

**💡 创新点**

CutCraft是首个专注于多镜头音视频生成中编辑技术执行的基准，提供了一个层次化的评估框架，结合了镜头结构对齐、专家模型度量和基于标准的问答。

**🔧 技术方法**

使用了层次化混合评估框架，结合了镜头对齐、混合度量和紧凑的强化学习训练的多模态语言模型评估。

**📊 数据集**

CutCraft数据集通过多阶段管道构建，包含295个样本，涵盖了多种编辑技术和镜头结构。

**📈 对比分析**

与13个最先进的闭源和开源模型进行比较，发现当前系统在生成视觉上可行的视频时，往往无法可靠地执行编辑指令，表现出一致的差距。

**⚠️ 局限性**

当前模型在镜头结构控制、音视频异步控制和蒙太奇执行方面存在根本性限制，且美学质量与编辑技术合规性之间的相关性较弱。

---

## 426. Coverage Path Planning for Redundant Manipulators using Generalized Spanning Trees

**arXiv ID:** 2609.08409 | [PDF](https://arxiv.org/pdf/2609.08409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 427. SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching

**arXiv ID:** 2609.08443 | [PDF](https://arxiv.org/pdf/2609.08443v1)

**作者:** Lin Guan `[一作]` (ByteDance), Lele Yu `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为SequenceO1的端到端框架，用于在推荐系统中处理超长（100K）序列建模，旨在提高用户历史交互的利用效率。

**💡 创新点**

创新点在于引入了Sketch Attention（SA）机制，将超长历史压缩为固定大小的用户专属草图，并通过缓存机制提高训练和推理的效率。

**🔧 技术方法**

使用了Sketch Attention（SA）和目标条件的交叉注意力（STCA）技术，结合了训练侧的本地键值缓存和多请求用户级批处理。

**📊 数据集**

使用了抖音（Douyin）平台的真实用户交互数据集，包含每个用户的历史交互记录，长度可达100K。

**📈 对比分析**

与现有的基于截断和多阶段检索的方法相比，SequenceO1在训练和推理中显著降低了计算和存储成本，同时保持了大部分性能提升。实验表明，SequenceO1在100K设置下的训练成本比STCA低49.9倍，推理成本低63.9倍。

**⚠️ 局限性**

限制在于当前方法主要针对100K的序列建模，尽管有扩展到百万级历史的潜力，但在实际应用中可能面临更复杂的计算和存储挑战。

---

## 428. AI-Native Orchestration in the 6G Continuum: Evolving Operator Platforms with Agentic AI

**arXiv ID:** 2609.08441 | [PDF](https://arxiv.org/pdf/2609.08441v1)

**作者:** Claudia Carballo González `[一作]` (i2CAT Foundation), Christos Verikoukis `[通讯]` (Industrial Systems Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了一种基于代理智能的架构扩展，旨在为第六代（6G）网络提供自主编排解决方案，以应对分布式计算和网络环境中的复杂性。

**💡 创新点**

创新点在于引入了一个AI原生的编排层，利用自主代理管理持久服务上下文，并通过CAMARA API实现闭环控制，支持跨域资源优化和冲突解决。

**🔧 技术方法**

使用了代理驱动的智能模块，结合了声明式监控和警报系统（DeMAS），以及去中心化的代理协商协议，构建了一个多代理的闭环编排架构。

**📊 数据集**

通过一个代表性的6G用例进行验证，涉及超可靠低延迟通信（URLLC）和增强移动宽带（eMBB）的共存场景。

**📈 对比分析**

与现有的轻量级政策网络相比，本文提出的框架在极端可靠性方面表现出色，成功保证了URLLC代理的延迟在10毫秒以内，同时优化了eMBB服务的尾延迟和基础设施能效。

**⚠️ 局限性**

局限性包括多代理决策的稳定性和认知偏差问题、跨域可观察性的可扩展性、联邦信任和操作隐私的挑战，以及边缘云原生部署和状态管理的复杂性。

---

## 429. EvolveScaler: Synthesizing Information-Evolution Contexts via Executable State Machines and Natural-Language Rendering

**arXiv ID:** 2609.08435 | [PDF](https://arxiv.org/pdf/2609.08435v1)

**作者:** Ziliang Zhao `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的信息演变（IE）任务，要求识别有效记录、按顺序应用更新，并从事件历史中重建查询相关状态。

**💡 创新点**

创新点在于引入了一种代码驱动的框架，通过人类编写的操作规范定义信息演变，并生成可执行的模拟器，从而确保生成的数据具有可验证性和一致性。

**🔧 技术方法**

使用了强大的大型语言模型（LLM）来合成可执行的模拟器，并通过确定性重放计算参考答案和原子检查表。

**📊 数据集**

构建了117个任务原型和159个最终问题操作符，生成了大约35,100个训练示例和585个验证评估实例。

**📈 对比分析**

通过在五个难度级别上对14个前沿和开源模型进行基准测试，结果显示最强模型在最高难度级别的avg@5为59.3%，而六个模型的avg@5得分低于10%。

**⚠️ 局限性**

限制在于当前框架专注于离散的、程序化指定的状态转变，未来的工作需要扩展到部分观察、连续和多模态设置。

---

## 430. Localized Visual Feature Aggregation via Focus Pooling for Visuomotor Policies

**arXiv ID:** 2609.08408 | [PDF](https://arxiv.org/pdf/2609.08408v1)

**作者:** Ruiyu Wang `[一作]` (KTH Royal Institute of Technology), Florian T. Pokorny `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种新的注意力池化模块，旨在从中间卷积神经网络（CNN）特征中提取与控制相关的局部信息，以提高数据效率。

**💡 创新点**

创新点在于引入了一种可训练的注意力池化模块，该模块能够根据机器人当前的本体感知状态选择性地聚合中间视觉特征，从而捕捉任务进展中的控制相关局部信息。

**🔧 技术方法**

使用了注意力机制，特别是交叉注意力，来实现对中间特征的选择性聚合。

**📊 数据集**

在MimicGen数据集和真实的UFactory xArm7机器人上进行了实验。

**📈 对比分析**

与四种基线方法（平均池化、空间softmax、机器人中心池化和注意力特征聚合）进行比较，结果显示该方法在模拟中提高了36.2%的成功率，在真实世界中提高了41.2%的成功率，同时仅使用了5.8%的编码器参数。

**⚠️ 局限性**

限制在于该方法依赖于ImageNet的预训练，尽管在没有预训练的情况下仍能取得41.7%的成功率，但预训练对性能的提升是显著的。

---

## 431. Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method

**arXiv ID:** 2609.08402 | [PDF](https://arxiv.org/pdf/2609.08402v1)

**作者:** Boao Yu `[一作]` (National University of Defense Technology), Rusheng Ju `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AGOS-Bench基准，评估无人机（UAV）和无人地面车辆（UGV）在城市环境中联合搜索和验证目标车辆的能力。

**💡 创新点**

创新点在于首次引入了专门的基准和数据集，AGOS-Bench和AGOS-Dataset，支持多视角视觉参考下的空地协作任务评估。

**🔧 技术方法**

使用了一种无训练的、工具增强的AGOS-Agent方法，该方法通过统一的搜索-交接-验证协议来协调UAV和UGV的决策。

**📊 数据集**

使用了AGOS-Dataset，包含7700个不同难度级别的搜索任务实例，涵盖五个CARLA城镇。

**📈 对比分析**

与九个通用视觉-语言模型（VLM）进行了广泛实验，AGOS-Agent在八个模型上提高了成功率，并在所有模型上减少了决策步骤，特别是在困难分割上，Gemini-3.6-Flash的成功率从8.6%提高到55.7%。

**⚠️ 局限性**

局限性包括评估仅限于210个来自一个测试城镇的实例，缺乏地理和外观的广泛泛化，且目标和干扰物为静态车辆，通信理想化，未测试模拟到现实的转移。

---

## 432. Authority Is Not a String: A Capability-Scoped Harness for Prompt-Injection-Resistant Coding Agents

**arXiv ID:** 2609.08371 | [PDF](https://arxiv.org/pdf/2609.08371v1)

**作者:** Dimitrios Stamatios Bouras `[一作]` (Peking University), Sergey Mechtaev `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的授权机制，旨在限制编码代理在执行工具调用时的权限，防止间接提示注入攻击。

**💡 创新点**

创新点在于引入了基于能力的授权模型，使每个代理只能访问其所需的特定权限，从而解决了当前编码代理授权的粒度不足问题。

**🔧 技术方法**

使用了一种能力范围的授权模型，该模型在每次工具调用前检查代理的权限，并在执行前冻结权限。

**📊 数据集**

在评估中使用了五个Python修复任务、五个注入表面和三个基线条件，共进行了300次实验。

**📈 对比分析**

与三种基线方法进行比较，结果显示在75次实验中，基线方法的注入效果执行率为–/75，而提出的方法执行率为/75，且成功完成了/75个修复任务。

**⚠️ 局限性**

限制在于该机制无法防止已授予能力的滥用或允许命令的传递效应，这仍然需要依赖于政策设计和沙箱机制。

---

## 433. ReMoMask-2: Latent Retrieval-Augmented Masked Motion Generation

**arXiv ID:** 2609.08365 | [PDF](https://arxiv.org/pdf/2609.08365v1)

**作者:** Yiran Wang `[一作]` (University of Sydney), Hao Tang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的文本到运动生成框架ReMoMask-2，旨在通过检索增强的生成方法来提高运动生成的质量和效率。

**💡 创新点**

创新点在于将检索数据库重建到生成器的预量化潜在空间中，从而消除了检索证据与生成潜在之间的表示差距，并引入了层次双向动量对齐和拓扑结构掩蔽等新技术。

**🔧 技术方法**

使用了层次双向动量对齐（HBM）、拓扑结构掩蔽（TSM）和语义时空注意力（SSTA）等技术来增强文本与运动之间的对齐和生成质量。

**📊 数据集**

在HumanML3D、KIT-ML和SnapMoGen数据集上进行了实验，这些数据集包含了大量的运动和文本对。

**📈 对比分析**

与现有方法相比，ReMoMask-2在所有基准测试中都达到了最先进的文本到运动检索性能，并在KIT-ML和SnapMoGen上获得了最低的FID值，且其单阶段生成的推理速度是所有比较系统中最快的。

**⚠️ 局限性**

局限性在于检索增强框架的有效性依赖于运动数据库与查询分布之间的匹配，当相关运动缺失或稀疏时，检索的好处会减小。此外，ReMoMask-2仅解码基础量化层，未能利用更深层的量化细节。

---

## 434. SentryLine: Evidence-Grounded Question Answering over Evolving Documents in Oncology Care

**arXiv ID:** 2609.08364 | [PDF](https://arxiv.org/pdf/2609.08364v1)

**作者:** Tampu Ravi Kumar `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SentryLine，一个针对肿瘤科医生和患者的临床问答系统，能够处理不断变化的ASCO活指南。

**💡 创新点**

SentryLine的创新点在于其能够实时检测指南的更新，并提供角色特定的答案和验证报告，解决了现有系统无法适应动态指南的问题。

**🔧 技术方法**

使用了分层的RAG（检索增强生成）管道和结构化实体提取技术，结合了多种生成模型进行答案生成。

**📊 数据集**

构建了ASCOBench，一个包含405个三轮对话的基准数据集，涵盖四个问题类别，并由临床专家提供金标准答案。

**📈 对比分析**

与五个基线模型进行比较，SentryLine在事实基础、时间漂移处理和角色特定生成方面表现出一致的改进，尤其在推理和角色特定问题上表现突出。

**⚠️ 局限性**

限制在于答案质量依赖于基础生成模型，且漂移检测依赖于指南PDF中明确的更新语言，此外，评估仅覆盖了ASCO的乳腺癌和前列腺癌指南。

---

## 435. Segment Any Motion with Radar: Robust Multimodal Moving-Object Segmentation and Tracking

**arXiv ID:** 2609.08346 | [PDF](https://arxiv.org/pdf/2609.08346v1)

**作者:** Jue Wang `[一作]` (Harbin Institute of Technology), Fei Luo `[通讯]` (Great Bay University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的基准和框架，用于在各种监控条件下进行运动分割和跟踪，结合了RGB、热成像和雷达数据。

**💡 创新点**

创新点在于构建了一个同步和校准的固定摄像头基准，提供了密集的实例掩码和时间一致的身份，同时开发了一个雷达感知的检测和跟踪框架。

**🔧 技术方法**

使用了RGB、热成像和雷达数据的融合技术，结合了运动监督和Hold-Lost记忆机制。

**📊 数据集**

使用了一个包含107个序列和8,537个注释帧的自定义数据集，涵盖白天、夜间、雨天和室内条件。

**📈 对比分析**

与现有的基准进行比较，提出的方法在IoU和F1_50上分别达到了0.7027和0.8090，MOTA、HOTA和IDF1分别提高了0.2977、0.1603和0.2857，显示出显著的性能提升。

**⚠️ 局限性**

限制在于雷达数据的稀疏性和噪声，且在目标停止或沿视线方向移动时，雷达的速度线索可能会消失。

---

## 436. A Multi-Modal Perception Pipeline for Object Detection and Tracking in Autonomous Racing

**arXiv ID:** 2609.08338 | [PDF](https://arxiv.org/pdf/2609.08338v1)

**作者:** Davide Malvezzi `[一作]` (University of Modena and Reggio Emilia), Marko Bertogna `[通讯]` (HiPeRT Srl)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多模态后融合感知管道，用于自主赛车领域的物体检测和跟踪。

**💡 创新点**

创新点在于结合了所有车载传感器，通过后融合方法和专门的多目标跟踪框架，显著提高了在极端速度和复杂环境下的检测和跟踪性能。

**🔧 技术方法**

使用了YOLOv4进行相机检测，PointPillars进行LiDAR检测，以及基于过滤和聚类的方法进行RADAR检测，结合了延迟补偿的扩展卡尔曼滤波器进行目标跟踪。

**📊 数据集**

使用了多个数据集，包括自定义的高速度自主赛车数据集，包含来自不同传感器的标注数据。

**📈 对比分析**

通过与现有方法的比较，实验结果表明，所提方法在多种赛车场景下表现出色，尤其是在高相对速度超车、遮挡下的多目标跟踪等情况下，跟踪精度和响应时间均优于单一传感器方法。

**⚠️ 局限性**

局限性在于当前方法未能充分利用语义感知的车辆几何模型，导致在某些情况下的估计偏差，未来工作将致力于解决这一问题。

---

## 437. Do Input-Level Defenses Transfer to Observation-Level Attacks on VideoLLMs?

**arXiv ID:** 2609.08331 | [PDF](https://arxiv.org/pdf/2609.08331v1)

**作者:** Bangshuo Zhu `[一作]` (University of New South Wales), Jingling Xue `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一个控制评估框架，系统评估输入级对抗防御是否能缓解观察级攻击对视频大语言模型的影响。

**💡 创新点**

创新点在于首次将输入级防御与观察级攻击的转移性进行系统评估，并提出了观察级攻击的机制分类。

**🔧 技术方法**

使用了控制评估框架，评估了11种输入级防御方法，包括图像压缩、局部梯度平滑等。

**📊 数据集**

使用了五种视频大语言模型（如LLaVA-Video-7B-Qwen2等）和五种观察级攻击（如PoisonVID、FRA等），共计825个评估单元。

**📈 对比分析**

与其他方法比较发现，输入级防御提供的保护有限且不一致，检测率通常接近零，且防御效果主要受模型架构影响，而非防御方法本身。

**⚠️ 局限性**

限制在于研究仅关注当前可用的模型、采样机制和观察级攻击，且未考虑其他任务可能存在的额外失败模式。

---

## 438. Windows Malware Detector as a Compound AI System: Trade-Offs in Accuracy, Efficiency, and Adversarial Robustness

**arXiv ID:** 2609.08394 | [PDF](https://arxiv.org/pdf/2609.08394v1)

**作者:** Andrea Ponte `[一作]` (University of Genova), Fabio Roli `[通讯]` (University of Genova)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新方法来优化和评估Windows恶意软件检测的复合AI系统，平衡检测性能、计算要求和鲁棒性。

**💡 创新点**

创新点在于引入系统级威胁模型，考虑攻击者对系统知识的不同程度，评估复合AI系统的整体鲁棒性，而不仅仅是单个检测组件。

**🔧 技术方法**

使用了基于规则的检测、静态分析和动态分析的多层次检测管道，结合机器学习模型进行恶意软件检测。

**📊 数据集**

使用了Speakeasy数据集，该数据集包含来自七个恶意软件家族的PE文件，经过去重处理后，训练集包含71,505个恶意样本和28,707个良性样本。

**📈 对比分析**

通过与现有的检测系统进行比较，展示了所提方法在训练时间和响应性方面的改进，同时仅造成检测性能的轻微损失。实验结果表明，优化的复合AI系统在面对知识更丰富的攻击者时，能够有效应对更具挑战性的对抗样本。

**⚠️ 局限性**

动态分析的性能有限，且当前的攻击模型未能针对动态检测进行评估，未来的工作需要扩展到动态行为的攻击模型。

---

## 439. Jointly Satisfying Pareto Optimality and Justified Representation is NP-Hard in Approval-Based Multiwinner Voting

**arXiv ID:** 2609.08357 | [PDF](https://arxiv.org/pdf/2609.08357v1)

**作者:** Chris Dong `[一作]` `[通讯]` (Hasso Plattner Institute), Chris Dong (Hasso Plattner Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文探讨了基于批准的多赢家投票中的一个开放问题，即是否可以有效地计算出同时满足合理代表性和帕累托最优性的委员会。作者证明了在所有配置的领域中，输出一个同时满足这两个公理的委员会是NP难的。

**💡 创新点**

创新点在于证明了任何仅输出满足合理代表性和帕累托最优性的委员会的投票规则都是NP难以计算的。

**🔧 技术方法**

使用了组合数学的引理和复杂性理论来证明结果。

**📊 数据集**

构建了一个基于3集合的精确覆盖问题（X3C）的投票实例，涉及18t + 12个投票者和多个候选人。

**📈 对比分析**

与现有方法的比较表明，虽然在限制领域中有一些正面结果，但在全领域中没有已知的多项式时间可计算的规则能够同时满足合理代表性和帕累托最优性。

**⚠️ 局限性**

限制在于没有已知的多项式时间可计算的规则能够同时满足合理代表性和帕累托最优性，除非P=NP。

---

## 440. RepoNav: From Snippet Retrieval to File-Centered Repository Navigation for Code Agents

**arXiv ID:** 2609.08355 | [PDF](https://arxiv.org/pdf/2609.08355v1)

**作者:** Hongzheng Chai `[一作]` (Beihang University), Yuan Yuan `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为RepoNav的轻量级后检索界面，用于改善代码代理在大型代码库中的功能定位能力。

**💡 创新点**

RepoNav通过将检索到的代码片段重新组织为以文件为中心的导航框架，帮助代理更有效地浏览文件结构，从而缩小文件发现与功能发现之间的差距。

**🔧 技术方法**

使用了轻量级的后检索接口设计，结合现有的密集检索基础设施，而不需要额外的持久结构索引或全库图。

**📊 数据集**

在LocBench数据集上进行评估，该数据集包含560个Python实例，每个实例将自然语言问题与相应的代码文件和功能配对。

**📈 对比分析**

与传统的平面片段检索方法相比，RepoNav在功能级定位上表现更好，显著减少了文件到功能的差距。实验结果显示，RepoNav在多个模型上均提高了功能定位的准确性。

**⚠️ 局限性**

当前评估主要集中在Python代码库和两个基准（LocBench和SWE-QA-Bench）上，未来需要扩展到其他编程语言和更广泛的应用场景。

---

## 441. CoVeR: Coverage-Based Token Pruning for Multi-View 3D Reasoning in VLMs

**arXiv ID:** 2609.08345 | [PDF](https://arxiv.org/pdf/2609.08345v1)

**作者:** Nhat-Tan Bui `[一作]` (Carnegie Mellon University), Fernando De la Torre `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为CoVeR的确定性、无训练的标记选择器，用于在2D视觉语言模型中进行多视角3D推理，通过空间覆盖进行修剪，仅使用几何信息。

**💡 创新点**

CoVeR通过优化场景覆盖并强制执行每个场景的确切预算，解决了现有方法的局限性，超越了当前最先进的标记修剪方法。

**🔧 技术方法**

使用几何信息的确定性选择方法，无需学习信号，采用空间距离进行标记选择。

**📊 数据集**

在多个3D推理基准上进行评估，包括ScanQA、SQA3D和OpenEQA，展示了其在不同场景下的有效性。

**📈 对比分析**

与现有的标记修剪方法（如Voxelization和学习重要性方法）进行比较，CoVeR在所有三个3D推理基准上均表现优越，保留了93.5%的全标记性能，同时仅使用约8%的视觉标记，平均超越最先进方法3.9个百分点。

**⚠️ 局限性**

CoVeR依赖于深度和相机信息，主要设计用于室内场景，其性能可能受估计几何质量的影响。未来的工作可以结合覆盖与可靠的深度/姿态估计，以及针对户外场景的分层或流式选择。

---

## 442. TV-Regulated OPD: Direction Matters in On-Policy Distillation

**arXiv ID:** 2609.08341 | [PDF](https://arxiv.org/pdf/2609.08341v1)

**作者:** Han Xiao `[一作]` (Hong Kong University of Science and Technology), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种新的方法，称为TV调节的在政策蒸馏（TV-OPD），旨在解决大语言模型（LLMs）在后训练阶段知识转移中的不稳定性问题。

**💡 创新点**

创新点在于仅保留标记级优势的符号信息，结合全局更新强度的调节，显著提高了训练的稳定性和后期性能。

**🔧 技术方法**

使用了总变差（Total Variation, TV）来调节优势，并通过实验验证了其有效性。

**📊 数据集**

使用了多个模型对（如Qwen3-8B与Qwen3-8B-Base，JustRL-DeepSeek-1.5B与DeepSeek-R1-Distill-Qwen-1.5B）进行评估，并在AIME 2024和AIME 2025基准上进行测试。

**📈 对比分析**

与标准的在政策蒸馏方法相比，TV-OPD在训练的后期表现出更好的性能和更低的方差，尤其在多个设置下均表现出色。

**⚠️ 局限性**

限制在于教师相对方向并不保证策略改进，且TV等价性是一个条件替代，未能保证序列级的单调返回。此外，TV-OPD的评估仅限于有限的模型对、基准和种子。

---

## 443. PENDA: An Efficient Processing Element via Norm-of-Difference for Deep Learning Accelerators

**arXiv ID:** 2609.08424 | [PDF](https://arxiv.org/pdf/2609.08424v1)

**作者:** Kai-Chieh Hsu `[一作]` (National Yang Ming Chiao Tung University), Tian-Sheuan Chang `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

本文提出了一种名为PENDA的高效处理单元架构，通过利用余弦定律将乘法运算转化为平方差运算，从而优化深度学习加速器的硬件效率。

**💡 创新点**

创新点在于用平方差运算替代传统的乘加单元，保持计算精度的同时显著降低了面积、能耗和时钟周期。

**🔧 技术方法**

使用了基于平方差的处理单元（NoD）架构，替代了传统的乘加（MAC）单元。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在深度学习加速器中的应用，特别是针对Transformer和卷积神经网络（CNN）模型。

**📈 对比分析**

与传统的MAC单元相比，PENDA在面积、能耗和时钟周期上分别减少了11-36%、5-48%和11-19%。在系统级别，PENDA在ViT-B和ResNet-18加速器中分别实现了13.6%和16.7%的速度提升，以及18%和59%的AET减少。

**⚠️ 局限性**

限制在于PENDA的实现需要对现有PE架构进行一定的修改，尽管这些修改相对较小，但仍可能影响与某些特定架构的兼容性。

---

## 444. Compositional Multilingual and Behavioral Attribute Steering

**arXiv ID:** 2609.08410 | [PDF](https://arxiv.org/pdf/2609.08410v1)

**作者:** Hyun Gu Kang `[一作]` (German Research Centre for Artificial Intelligence), Simon Ostermann `[通讯]` (German Research Centre for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究探讨了大语言模型中用于语言和行为控制的引导向量的组合性，重点研究了语言、越狱和简洁性属性的引导效果。

**💡 创新点**

创新点在于提出了在不进行额外训练的情况下，通过适当的干预层和引导强度组合多个属性的引导向量，从而实现同时控制多个属性的效果。

**🔧 技术方法**

使用了引导向量提取、应用和组合的技术，具体通过对比激活提取DiffMean引导向量，并在不同层次上进行干预。

**📊 数据集**

使用了FLORES数据集（用于语言向量提取），以及构建的越狱和简洁性向量的对比数据集，涉及多种语言和指令。

**📈 对比分析**

通过与基于提示的基线进行比较，发现组合引导向量在多个模型上表现优于基线，尤其是在语言和越狱属性的组合上，但在语言和简洁性属性的组合上表现不如基线，显示出更大的变异性。

**⚠️ 局限性**

限制在于实验仅限于指令调优模型，且使用固定的干预强度，未考虑在不同上下文中可能需要动态调整的最佳强度。

---

## 445. Equivariance Breaks the Learning Rate

**arXiv ID:** 2609.08381 | [PDF](https://arxiv.org/pdf/2609.08381v1)

**作者:** Andrei Manolache `[一作]` (University of Stuttgart), Mathias Niepert `[通讯]` (University of Stuttgart)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了矩阵结构优化器（如Muon）在等变网络中相较于Adam的优势，分析了其原因并提出了块归一化的方法来解决学习率不匹配的问题。

**💡 创新点**

提出了一种无参数的块归一化规则，消除了等变网络中不同块之间的学习率不匹配，同时保持了Adam的动量估计和更新方向不变。

**🔧 技术方法**

使用了块归一化和调节Adam的动量系数的技术来提高优化性能。

**📊 数据集**

在rMD17和MD22数据集上进行了实验，使用了SO(3)-等变模型和e3nn的相互原子势模型。

**📈 对比分析**

通过与Muon的比较，发现结合块归一化和调节动量的Adam在宽度为64和128时的性能超过了Muon，而在其他宽度下则表现相近。

**⚠️ 局限性**

研究仅限于SO(3)-等变网络和三个分子数据集，未来的工作应考虑更大的模型、更现实的数据集以及其他优化器。

---

## 446. Kirigami Meta-Sheet for Enhanced Impact Absorption

**arXiv ID:** 2609.08362 | [PDF](https://arxiv.org/pdf/2609.08362v1)

**作者:** Dahyun Joo `[一作]` (Seoul National University), Do-Nyun Kim `[通讯]` (Seoul National University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于kirigami的平面吸能材料，利用正负刚度区域的转变来增强冲击吸收能力，而不是依赖牺牲性破坏。

**💡 创新点**

创新点在于通过编程连接单元的铰链比率，使得低铰链比的kirigami材料能够清晰地展现负刚度转变，从而减少反弹、降低首次冲击力并增加能量耗散。

**🔧 技术方法**

使用了简单的质量-弹簧-阻尼模型、准静态压痕实验和有限元（FE）模拟技术来分析材料的性能。

**📊 数据集**

实验中使用了3D打印的kirigami meta-sheet，铰链比分别为0.08045（低铰链比）、0.09739（中铰链比）和0.15682（高铰链比）。

**📈 对比分析**

通过与传统的聚乙烯网和泡沫材料进行比较，结果表明，低铰链比的kirigami材料在冲击测试中表现出最低的反弹系数、最高的能量吸收比例和最小的首次冲击力。

**⚠️ 局限性**

限制在于需要进一步研究铰链的耐疲劳设计、多层meta-sheet的组装、标准化的峰值力测量和可扩展的制造工艺。

---

## 447. MLIP Detective: Active Failure Mode Discovery Beyond Benchmark Scores for Machine-Learning Interatomic Potentials

**arXiv ID:** 2609.08399 | [PDF](https://arxiv.org/pdf/2609.08399v1)

**作者:** Ryuhei Okuno `[一作]` (Preferred Networks), Yuta Tsuboi `[通讯]` (Preferred Networks)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MLIP Detective的框架，用于主动发现机器学习原子间势（u-MLIPs）的失败模式，旨在超越传统基准评估的局限性。

**💡 创新点**

创新点在于结合物理知识的主动搜索方法，生成可证伪的失败假设，并通过低成本的模拟筛选，最终将最可疑的案例上报给人类专家进行验证。

**🔧 技术方法**

使用了大型语言模型（LLM）作为代理，结合交叉模型不一致性和物理约束来生成失败假设，并设计了一个获取策略来优先验证最可疑的候选。

**📊 数据集**

使用了现有的基准数据集，如MLIP Arena和Matbench Discovery，作为起点进行主动搜索。

**📈 对比分析**

通过交叉模型比较，MLIP Detective识别并表征了MACE-MPA-0模型中的系统性异常，发现某些含氧或氟的吸附体-表面系统的能量预测高于其对应的分离片段，性能表现出明显的缺陷。

**⚠️ 局限性**

局限性在于搜索仍然受限于检查规范，升级的发现需要昂贵的人类验证，未来的工作包括自动化质量控制和扩展搜索到更广泛的模拟协议和目标模型。

---

## 448. Geometry-Aware Bayesian Parameter-Efficient Fine-Tuning on the Stiefel Manifold via Stein Variational Gradient Descent

**arXiv ID:** 2609.08354 | [PDF](https://arxiv.org/pdf/2609.08354v1)

**作者:** Quang-Duy Tran `[一作]` (Deakin University), Thin Nguyen `[通讯]` (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于Riemannian Stein变分梯度下降的框架StePS，用于量化不确定性和校准几何感知的参数高效微调（PEFT）。

**💡 创新点**

创新点在于通过在Stiefel流形上迭代运输粒子，显式和原则性地估计认知不确定性，并提供了与目标分布匹配的闭合形式更新。

**🔧 技术方法**

使用了Riemannian Stein变分梯度下降（SVGD）技术，结合了几何优化和粒子基础的变分推断。

**📊 数据集**

在多个常识推理基准上进行了实验，包括Winogrande、ARC、OpenBookQA等数据集。

**📈 对比分析**

与LoRA和StelLA等基线方法相比，StePS在准确性和模型校准方面表现出显著的改进，尤其在分布内和分布外的设置中均表现良好。

**⚠️ 局限性**

限制在于当前方法的计算效率仍有待提高，未来工作将探索基于模型的变分推断以进一步增强几何感知的贝叶斯PEFT的计算效率。

---

## 449. Toward Fully Autonomous 6G Networks: AI-driven Operational Efficiency and Optimization

**arXiv ID:** 2609.08426 | [PDF](https://arxiv.org/pdf/2609.08426v1)

**作者:** David Reiss `[一作]` (Universitat Politècnica de Catalunya), Daniel Camps-Mur `[通讯]` (i2CAT Foundation)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究探讨了未来6G网络中AI驱动的操作效率和优化，提出了一种基于代理的编排框架，以实现自主的6G无线接入网络（RAN）管理。

**💡 创新点**

创新点在于将AI与网络即服务（NaaS）生态系统结合，提出了一种能够解释基于意图的政策的框架，从而实现外部NaaS请求与内部网络管理政策的集成。

**🔧 技术方法**

使用了AI代理技术，特别是基于意图的编排和语义推理的长短期评估机制。

**📊 数据集**

使用了来自欧洲移动网络运营商（MNO）的真实5G网络数据集，进行AI驱动机制的评估和验证。

**📈 对比分析**

通过与现有的O-RAN合规框架进行比较，展示了AI驱动机制在提高RAN操作效率方面的有效性，且在O-RAN仿真框架中进行了验证。

**⚠️ 局限性**

局限性在于AI代理之间的冲突管理尚未深入探讨，复杂的动态政策和重叠目标的协调机制仍需进一步研究。

---

## 450. Reading a Legal Question Word by Word: Embedding Trajectories of 2,144 Vietnamese Legal Headlines

**arXiv ID:** 2609.08372 | [PDF](https://arxiv.org/pdf/2609.08372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 451. FastE: Readout-Triggered Token Compression for LLM Embedding Inference

**arXiv ID:** 2609.08407 | [PDF](https://arxiv.org/pdf/2609.08407v1)

**作者:** Jinsong Shu `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究识别了最终读取LLM嵌入模型中的深度依赖前缀冗余，特别是在Qwen3-Embedding和Qwen3-VL-Embedding等代表性骨干网络中。我们发现，在浅层中移除前缀状态的影响远大于在深层中，表明随着前缀和读取状态在网络中传播，前缀状态变得越来越可压缩。为此，我们引入了FastE，这是一种无训练、即插即用的方法。

**💡 创新点**

创新点在于提出了一种基于读取状态与前缀状态对齐的在线启发式方法，来决定何时进行前缀状态压缩，并通过注意力分数对前缀状态进行排名，以确定在后续层中保留哪些状态。

**🔧 技术方法**

使用了一种名为FastE的训练无关方法，该方法通过监控批量均值读取-前缀对齐来触发压缩，并利用读取注意力对前缀状态进行排名。

**📊 数据集**

使用了NarrativeQA数据集以及五个文本嵌入基准、两个骨干网络规模和三个跨模态检索任务进行评估。

**📈 对比分析**

与其他方法相比，FastE在匹配的前缀状态预算下表现出更好的质量-效率权衡。在NarrativeQA上，FastE在减少40.11%的解码器骨干FLOPs的同时，保留了99.53%的Full Forward nDCG@10，显示出显著的计算成本降低和速度提升。

**⚠️ 局限性**

限制在于FastE主要针对最终读取LLM嵌入模型，最终层池化骨干需要一个池化感知的压缩信号。此外，其控制器引入了开销，因此FLOPs的减少应与测量的速度提升一起解释。

---

## 452. From Coordinates to Candidate Regions: Temporal Change Localization via Region Selection in Remote Sensing Multimodal LLMs

**arXiv ID:** 2609.08391 | [PDF](https://arxiv.org/pdf/2609.08391v1)

**作者:** Juwan Chung `[一作]` (KAIST), Yong Man Ro `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对遥感图像的区域选择框架，旨在解决特定物体或变化区域的定位问题，尤其是在多图像序列中进行时间变化定位。

**💡 创新点**

创新点在于将区域选择范式扩展到遥感领域，通过文本条件的区域提议模块和每帧视觉特征的特殊标记来实现目标定位，而不是生成坐标序列。

**🔧 技术方法**

使用了文本条件的区域提议模块，结合空间和时间线索的视觉特征编码，并通过大语言模型（LLM）进行区域标记选择。

**📊 数据集**

构建了一个多任务训练和评估套件，使用了多个数据集，包括TEOChatlas、GeoChat Instruct、FIT-RS、DIOR-RSVG等，涵盖单图像和多时序设置。

**📈 对比分析**

与现有的RS-MLLM和坐标生成基线进行比较，实验结果显示该方法在时间变化定位和单图像视觉定位任务上显著优于坐标生成方法，同时在理解任务上保持竞争力。

**⚠️ 局限性**

局限性在于提议模块的回忆能力限制，模型只能选择由区域提议模块生成的候选区域，未提议的区域无法恢复。此外，训练和评估指标之间的差距也可能影响性能评估。

---

## 453. "Here Be Sharks!": Enhancing Scientific Communication and Analysis through Authoring Interactivity

**arXiv ID:** 2609.08386 | [PDF](https://arxiv.org/pdf/2609.08386v1)

**作者:** Caroline Berger `[一作]` (Aarhus University), Clemens Nylandsted Klokmose `[通讯]` (Aarhus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究报告了一个案例研究，旨在设计交互式可视化的创作环境，以增强科学工作。通过与一组海洋生物学家的研讨会和原型评审，识别了在可视化数据时面临的挑战，并提出了相应的设计要求。

**💡 创新点**

创新点在于提出了非传统编程语言和环境的考虑，探讨了如何为科学家提供交互式可视化的支持，并分析了如何有效整合人工智能。

**🔧 技术方法**

使用了参与式设计方法，包括研讨会和原型评审，结合了主题分析来识别科学家在可视化数据时的需求和挑战。

**📊 数据集**

研究中涉及的参与者为七名海洋生物学家，数据集主要包括他们在研究中使用的可视化类型和信息，如物种社区、环境条件和鲨鱼运动等。

**📈 对比分析**

通过与海洋生物学家的互动，研究发现交互性可以帮助科学家更准确地传达研究结果，但现有工具的可及性限制了科学家的表达能力。与传统的静态图形相比，交互式可视化能够更好地支持数据分析和沟通。

**⚠️ 局限性**

限制在于研究结果可能需要根据其他科学领域进行适应，尤其是计算生物学等具有强计算传统的领域。此外，使用的原型是一个挑衅性的原型，而非完全功能的工具，限制了科学家在实际工作中的反馈和使用体验。

---

## 454. Miles v0.1: Production-Level Post-Training

**arXiv ID:** 2609.08368 | [PDF](https://arxiv.org/pdf/2609.08368v1)

**作者:** RadixArk `[一作]`, Zhichen Zeng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个全栈的、适用于前沿后训练的系统v0.1，旨在使前沿规模的强化学习（RL）对研究人员和企业更易于访问。

**💡 创新点**

系统设计围绕组件的可验证性、清晰性和可定制性，强调准确性、效率、可靠性和可扩展性。

**🔧 技术方法**

使用了SGLang构建的回滚引擎，支持NVIDIA Megatron-LM和PyTorch FSDP的训练后端，以及三种不同的权重同步传输方式。

**📊 数据集**

在GLM-5.2 744B-A40B模型上进行了案例研究，使用64个NVIDIA GB300 GPU进行终端使用编码任务的完全异步代理RL。

**📈 对比分析**

通过比较不同的回滚生成和训练方法，展示了如何优化请求路由以提高吞吐量和保真度，确保训练和回滚之间的一致性。

**⚠️ 局限性**

系统的局限性包括某些精度格式仍处于早期阶段，某些权重传输路径仅覆盖特定模型系列，以及某些测量仅来自单一配置。

---

## 455. AirAnchor: Bridging Local and Global Spatial Information for Zero-Shot Aerial Vision-and-Language Navigation

**arXiv ID:** 2609.08442 | [PDF](https://arxiv.org/pdf/2609.08442v1)

**作者:** Shanwei Fan `[一作]` (National Key Laboratory of Cognition and Decision Intelligence for Complex Systems Institute of Automation Chinese Academy of Sciences), Guoliang Fan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的无人机导航框架AirAnchor，旨在通过空间锚点将局部和全局空间信息结合起来，以支持自然语言指令下的导航。

**💡 创新点**

创新点在于通过空间锚点桥接局部空间基础和全局空间记忆，使得两者能够在导航框架中共同利用，从而实现更全面的空间决策支持。

**🔧 技术方法**

使用了多模态大语言模型（MLLM）作为基础，结合了查询驱动的空间锚点定位、持久对象空间记忆和空间信息导航代理等技术。

**📊 数据集**

在AerialVLN-S数据集上进行评估，该数据集包含8446条飞行轨迹，涵盖25个城市规模的环境，涉及870多个城市物体类别。

**📈 对比分析**

与现有的零-shot 基线方法相比，AirAnchor在成功率、导航误差等指标上均表现出显著的提升，验证了其有效性和效率。

**⚠️ 局限性**

局限性包括：空间锚点的准确性依赖于RGB-D观测和UAV姿态，错误可能影响导航决策；在新环境中，指令相关的地标可能尚未存在于SOKB中；当前导航代理使用的技能集是预定义的，可能无法适应更复杂的动态环境。

---

## 456. Feyospace-v1: How the Cyber Mercury Seven Trained Frontier Cyber Models

**arXiv ID:** 2609.08418 | [PDF](https://arxiv.org/pdf/2609.08418v1)

**作者:** Zongjie Li `[一作]` (Vera Praxis Lab), Deke X `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了如何通过后训练改进现有开放权重模型的编码和网络安全能力，提供了一种更实用的路径，避免从头开始训练前沿基础模型所需的高计算和数据资源。

**💡 创新点**

创新点在于提出了五种互补技术，解决了推理保真度、数据经济性、封闭模型引导、开放模型能力恢复和专家指导内化等问题，形成了一个环境驱动的数据引擎。

**🔧 技术方法**

使用了多种技术，包括模型特征恢复、低成本教师采样、可选择的变异策略、白盒模型合并逆转和专家干预内化等。

**📊 数据集**

使用的数据集包括27,502个多语言编码实例、69,854个来自公共漏洞记录的环境、9,312个作者维护的CTF环境和12,993个经过系统挖掘的Linux内核历史环境，共计164,269个经过审计的轨迹。

**📈 对比分析**

与基线模型相比，后训练的模型在CyberGym的验证成功率提高了23.76%，在CTF的成功率提高了10.49%。Feyospace-s1在CyberGym排行榜上排名第10，所有三个检查点在相似参数规模的模型中排名第一。

**⚠️ 局限性**

限制在于高质量教师的获取和数据构建的成本，尤其是在封闭模型的访问和能力恢复方面，此外，恢复的推理轨迹未能直接用于模型训练，主要用于观察和分析。

---

## 457. Risk-Aware Generative Inpainting for Optimized Design Editing of EV Battery Cooling Channels

**arXiv ID:** 2609.08387 | [PDF](https://arxiv.org/pdf/2609.08387v1)

**作者:** Leekyo Jeong `[一作]` (Korea Advanced Institute of Science and Technology), Namwoo Kang `[通讯]` (Narnia Labs)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种风险感知的生成编辑框架，用于电动车电池包的冷却通道设计，旨在通过局部拓扑修改来提高性能，同时保持单一连续通道。

**💡 创新点**

创新点在于引入了风险感知的生成编辑方法，通过评估编辑位置的结果分布而非单一期望值来选择编辑位置，从而有效管理结果的随机性。

**🔧 技术方法**

使用了扩散模型进行图像修复，结合了分布值回归和风险敏感设计的技术。

**📊 数据集**

使用了基于计算流体动力学（CFD）生成的布局数据集，包含2000个布局用于训练和100个布局用于验证。

**📈 对比分析**

与随机编辑方法进行了比较，结果显示在大多数掩模配置下，提出的方法在冷却通道目标上优于随机编辑，并且在独立的CFD验证中保持了优势。

**⚠️ 局限性**

局限性包括所学的分布函数未能通过标准校准测试，且在不同的风险水平下，结果的一致性存在显著波动。

---

## 458. Structural Jailbreaks Generalize but Do Not Compound: A cross-provider and multilingual study of Involuntary In-Context Learning

**arXiv ID:** 2609.08373 | [PDF](https://arxiv.org/pdf/2609.08373v1)

**作者:** Tejasvi C. Addagada `[一作]` `[通讯]` (Independent researcher), Tejasvi C. Addagada (Independent researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本论文测试了结构性越狱（IICL）和多语言安全差距对两个Google Gemini模型的影响，发现IICL在金融领域的攻击成功率显著提高。

**💡 创新点**

创新点在于直接测试了IICL与多语言安全差距的相互作用，结果表明这两种弱点并不叠加，而是相互抵消。

**🔧 技术方法**

使用了IICL操作符和StrongREJECT风格的评分标准进行评估。

**📊 数据集**

使用了HarmBench和FinProof两个数据集，分别包含30个一般性危害行为和30个金融滥用行为。

**📈 对比分析**

与单次尝试的基线相比，IICL在HarmBench上的攻击成功率从≤6.7%提升至80-90%，在FinProof上提升至97-100%。而在非英语条件下，攻击成功率普遍低于英语基线。

**⚠️ 局限性**

限制在于样本规模较小（每个条件n=30），且仅测试了Google的模型，未能涵盖更广泛的模型和语言。

---

## 459. Reachability-Certified Subteam Decomposition for Locally Interacting Multi-Agent MDPs

**arXiv ID:** 2609.08366 | [PDF](https://arxiv.org/pdf/2609.08366v1)

**作者:** Xiangwu Wang `[一作]` (University of Hong Kong), Hongyuan Tang `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的方法，称为可达性认证子团队分解（RCSD），用于有限多智能体马尔可夫决策过程，旨在优化在有限通信能力下的多智能体协调。

**💡 创新点**

RCSD结合了对成对接触时间的速度限制下界和奖励包络，形成当前状态的亲和力，提供了一种可计算的证书来界定在容量受限的情况下的持久子团队。

**🔧 技术方法**

使用了马尔可夫决策过程（MDP）和动态规划技术，结合了速度限制、奖励包络和几何距离等因素。

**📊 数据集**

在控制的五智能体问题上进行了实验，并在随机二维障碍网格上进行了扩展研究，验证了方法的有效性。

**📈 对比分析**

与均匀、仅基于距离和仅基于包络的分区方法相比，RCSD-Exact在聚合归一化执行后悔方面分别减少了56.0%、28.8%和25.3%。在384个精确分区和1440个受限控制器评估中，没有发现界限违反。

**⚠️ 局限性**

方法的局限性在于，分区构造在中位数上保持在100个智能体以下，且未包括亲和力形成或MDP规划的时间。

---

## 460. Rank Without an Oracle: Deviation-Aware Interaction-Rank Selection from Offline Multi-Agent Logs

**arXiv ID:** 2609.08358 | [PDF](https://arxiv.org/pdf/2609.08358v1)

**作者:** Xiangwu Wang `[一作]` (University of Hong Kong), Hongyuan Tang `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种选择性交互排名验证方法（SIRV），用于在已知日志分布下评估有限博弈的离线多智能体收益模型。

**💡 创新点**

创新点在于通过选择性比较不同候选模型的表现，返回满足特定上限得分标准的最小排名，或在目标不支持时放弃选择，从而避免了传统方法的偏差。

**🔧 技术方法**

使用了选择性交互排名验证（SIRV）技术，结合了经验-伯恩斯坦界限和霍夫丁界限来评估模型的表现。

**📊 数据集**

在控制的因子研究中，使用了2048个独立博弈的合成数据集，涵盖了不同的游戏家庭和日志分布。

**📈 对比分析**

与ID-Mean和ID-Max等方法进行比较，SIRV-EB在候选选择的CCE悔恨和福利悔恨方面表现出积极的效果，尤其在排名为3的情况下，平均CCE悔恨减少了0.00518，福利悔恨减少了0.00328。

**⚠️ 局限性**

限制在于该方法在某些情况下可能无法提供普遍的战略改进，且在不同的噪声条件下，证书的效用可能会下降。

---

## 461. VeriScene: Reconstructing Crime Scenes from Legal Evidence via World-Model Agent

**arXiv ID:** 2609.08342 | [PDF](https://arxiv.org/pdf/2609.08342v1)

**作者:** Kevin Chuanpu Fu `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种通过法律证据重建犯罪现场的方法，利用世界模型代理将多模态法律证据融合，生成符合物理规律的动态场景。

**💡 创新点**

创新点在于设计了一个代理系统，该系统能够迭代融合证据，保持每个主张可追溯，并确保每个动作在物理上是合理的。

**🔧 技术方法**

使用了世界模型和多模态学习模型（如Gemini和Veo 3.1）进行证据融合、物理推理和视频重建。

**📊 数据集**

构建了一个包含25个犯罪场景的基准数据集，涵盖7种物理驱动的案件类型，包括139张法医风格的照片和65个带有植入不可靠性的证人陈述。

**📈 对比分析**

与端到端的多模态LLM基线相比，本文的方法在事实一致性上提高了20.35%，在时间一致性上提高了34.88%，在20个测试场景中达到了0.9014的证据覆盖率和0.7217的事实一致性。

**⚠️ 局限性**

限制在于渲染器在文本级物理约束方面的遵循不完美，合成基准的转移性未经过验证，且生成模型可能会伪造证据。

---

## 462. RoboCousin: Build Your Own Simulation Playground for Robust Bimanual Robotic Manipulation

**arXiv ID:** 2609.08339 | [PDF](https://arxiv.org/pdf/2609.08339v1)

**作者:** Jingxuan Zhu `[一作]` (E-surfing Digital Life Technology), Hongming Li `[通讯]` (E-surfing Digital Life Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了RoboCousin，一个基于仿真的数据生成平台，用于双手操作的训练，旨在通过用户提供的观察数据生成可重用的资产、场景和专家轨迹。

**💡 创新点**

创新点在于RoboCousin能够将用户提供的观察数据转化为交互准备好的对象和背景资产，并通过数字表亲生成框架扩展对象、场景和语言的多样性，同时保持任务相关的语义和空间关系。

**🔧 技术方法**

使用了RoboTwin 2.0作为基础，结合3D物体重建模型、3D高斯点云和大型语言模型，自动生成可交互的对象和背景环境。

**📊 数据集**

发布了RoboCousin-OBD数据集，包含3000多个注释对象实例和50个背景环境，支持多种家庭场景。

**📈 对比分析**

通过仿真和真实机器人实验比较，自动生成的交互注释与策划的注释相当，使用生成资产训练的策略在评估的真实机器人任务中表现与RoboTwin 2.0相当或更好。

**⚠️ 局限性**

限制在于当前的工作主要集中在双手操作的仿真数据生成上，未来需要在任务、环境和机器人体现上进行更广泛的评估，并改善物理校准和任务感知的表亲生成。

---

## 463. Distillation as Probability Transport: Routed On-Policy Distillation

**arXiv ID:** 2609.08337 | [PDF](https://arxiv.org/pdf/2609.08337v1)

**作者:** Tianle Xia `[一作]` (Tencent), Jie Jiang `[通讯]` (Tencent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的方法，将教师知识通过学生生成的轨迹进行转移，称为教师引导的概率传输，旨在解决传统的在政策蒸馏中更新不明确的问题。

**💡 创新点**

创新点在于将教师-学生之间的分歧重新构建为显式的源-目的地更新，优化了教师偏好的目标，并通过适应性预算控制更新的幅度。

**🔧 技术方法**

使用了教师引导的概率传输技术，结合了最大熵耦合和教师潜力的共享构造。

**📊 数据集**

在四个教师-学生设置和四个数学推理基准上进行了实验，使用的数据集包括MATH500、AMC23、AIME24和AIME25。

**📈 对比分析**

与采样的反向KL OPD相比，该方法在四个基准的平均表现提高了2.19到3.61分，且在匹配问题和样本的评估区间上均保持严格的正值，显示出更高的路由保真度和更低的背景泄漏。

**⚠️ 局限性**

限制在于尽管该方法在多个设置中表现良好，但仍需进一步验证其在更广泛应用场景中的有效性和稳定性。

---

## 464. EdMCGS: Event-Driven Markov Chain Gaussian Splatting for Extreme-Low-Frame-Rate Dynamic Scene Reconstruction

**arXiv ID:** 2609.08332 | [PDF](https://arxiv.org/pdf/2609.08332v1)

**作者:** Yuzhong Wang `[一作]` (Macau University of Science and Technology), Xinxing Yu `[通讯]` (Macau University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于极低帧率RGB图像与事件流的端到端动态3D场景重建方法EdMCGS，能够在任意时间点生成高质量的可渲染视图；

**💡 创新点**

核心创新在于将场景运动建模为事件驱动的马尔可夫链，使得事件在推断阶段仍能直接驱动控制点的状态转移；同时采用局部事件采样与时域局部等距正则化，提升运动一致性与细节重建；并且实现轻量化、无外部流网络、无姿态网络的完整框架；

**🔧 技术方法**

技术实现包括3D高斯剖面渲染（3D Gaussian Splatting）、控制点线性混合皮肤化、事件生成模型、事件驱动马尔可夫链、局部事件特征编码与采样、时域局部等距正则化、相片损失、事件损失以及多项正则化的联合优化；

**📊 数据集**

使用了公开的合成事件单目数据集（Jumpingjack、Kick、Mutant、Lego四个序列）以及来自E‑D3DGS的真实世界数据集（Excavator、Jeep、Flowers、Eagle），均对RGB帧降至5/10fps作为极低帧率输入；

**📈 对比分析**

与RGB基准（Deformable 3DGS、SC‑GS）、E2VID+Deformable 3DGS以及事件增强的Deformable 3DGS（E‑D3DGS）以及内部无马尔可夫变体进行对比。实验表明EdMCGS在PSNR、SSIM、LPIPS上均优于所有基线，尤其在极低帧率下提升更显著；同时训练速度更快、渲染帧率更高、所需高斯数量更少；消融实验进一步验证了事件驱动马尔可夫链、局部采样与时域等距正则化对性能的贡献；

**⚠️ 局限性**

主要局限在于需要已知相机位姿，未能同时学习位姿与动态几何；仅适用于单摄像头事件流，缺乏对多摄像头或大场景的扩展；此外，模型仍假设事件与RGB同步，未探索更高阶事件驱动的时序模型。

---

## 465. CAR-MIL: Counterfactual Attention Regularization for Multiple Instance Learning

**arXiv ID:** 2609.08419 | [PDF](https://arxiv.org/pdf/2609.08419v1)

**作者:** Imane Chraki `[一作]` (Université Paris-Saclay), Maria Vakalopoulou `[通讯]` (Université Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在多实例学习（MIL）中通过对抗式反事实注意力正则化（CAR-MIL）来显式引导注意力学习，从而提高模型解释性与性能。

**💡 创新点**

创新点：①在训练阶段加入反事实注意力分支，强制两条注意力分布相近但预测差异显著；②通过证据差异损失让模型关注支持与反驳预测的实例；③实现双分支轻量化架构，无需额外监督。

**🔧 技术方法**

技术：基于Attention‑based MIL（ABMIL）框架，使用两条注意力分支；损失函数包括交叉熵、证据差异损失（softmax(ΔF)）和注意力对齐损失（L1或余弦距离）。

**📊 数据集**

数据集：合成MNIST‑bags（四类与相邻对任务）以及五个数字病理WSI数据集（TCGA‑NSCLC、TCGA‑BRCA、TCGA‑LUAD、CAMELYON16、BRACS），采用UNI‑V1或ResNet50特征提取。

**📈 对比分析**

与多种MIL基线比较（Mean‑Max、ABMIL、CLAM、ACMIL、DSMIL、TransMIL、AddMIL、CIA‑MIL）。在合成数据上，CAR‑MIL在AUC和实例级AUPRC上均优于基线；在真实病理数据中，CAR‑MIL保持或略优于最强基线，尤其在挑战性任务（LUAD、BRACS）上提升显著，同时提升注意力可靠性。

**⚠️ 局限性**

局限性：①对超参数α、λ敏感，需调优；②仅在注意力空间做反事实，未探索输入空间干预；③实验主要聚焦于医学WSI，缺乏跨领域验证；④对大规模袋子计算仍有一定开销。

---

## 466. Stochastically Perturbed Weights: Ensembles from Deterministic Machine-Learning Weather Models

**arXiv ID:** 2609.08412 | [PDF](https://arxiv.org/pdf/2609.08412v1)

**作者:** Simon Adamov `[一作]` (Federal Office for Meteorology and Climatology MeteoSwiss), Sebastian Schemm `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用在已训练的单一天气预报模型的权重上注入微小随机噪声，生成无训练成本的概率性天气预报集群，并探讨噪声注入位置与模型不确定性分布的关系。

**💡 创新点**

提出无训练成本的权重随机扰动（SPW）框架，并通过三阶段参数搜索（幅度、张量组、粗尺度限制）实现跨架构的自适应不确定性恢复，首次揭示不同ML天气模型对噪声注入位置的依赖性。

**🔧 技术方法**

使用乘法高斯噪声扰动权重、三阶段实验设计、滚动过程中的噪声刷新、初始条件扰动融合，以及WeatherBench‑2的概率评分和校准指标进行评估。

**📊 数据集**

基准数据集为WeatherBench‑2（含ERA5初始场和多时相验证），对比I‑FS‑ENS、AIFS‑ENS、FCN3、Atlas等概率预报模型，并在Hurricane Milton案例中验证异常事件表现。

**📈 对比分析**

通过与训练式概率模型和I‑FS‑ENS的CRPSS、SSR、SSIM等指标对比，SPW在240 h时的CRPSS仅落后0.04–0.13点，平均单成员推理时间仅59–125 s，显著低于Atlas的38 min，展示了高效且可竞争的性能。

**⚠️ 局限性**

需针对每个模型进行手动参数搜索，粗尺度噪声导致空间平均过度离散；仅针对10成员、7个气象量，缺乏降水等尖峰变量；对初始条件扰动依赖外部Ensemble；仅验证至240 h，未检验更长滚动和有限区模型的适用性。

---

## 467. Environments as Scaffold: Enriching Feedback to Bootstrap Self-Evolving Agents in Long-Horizon Tasks

**arXiv ID:** 2609.08404 | [PDF](https://arxiv.org/pdf/2609.08404v1)

**作者:** Hongbang Yuan `[一作]` (Fudan University), Yixin Cao `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的环境适应策略，通过丰富反馈信号来改善长时间任务中的强化学习训练。

**💡 创新点**

创新点在于从代理端的预热转向环境端的适应，系统性地设计反馈信息的提供方式和时机，以提高强化学习的有效性。

**🔧 技术方法**

使用了强化学习（RL）算法，包括GRPO、DAPO和GSPO，并构建了反馈丰富环境（FEEs）进行训练。

**📊 数据集**

使用了SciWorld和BFCL两个广泛采用的基准数据集。

**📈 对比分析**

与标准环境相比，FEEs在各种模型规模和RL算法上均表现出一致的性能提升，平均提高了2.82%。

**⚠️ 局限性**

限制在于构建反馈丰富环境需要特定的设计选择和超参数，且在更具挑战性的环境中效果不佳，可能无法产生正奖励。

---

## 468. Noise Adaptive Streaming Audio-Visual Speech Token Enhancement for Robust Full-Duplex Spoken Dialogue Models

**arXiv ID:** 2609.08390 | [PDF](https://arxiv.org/pdf/2609.08390v1)

**作者:** Bella Godiva `[一作]` (KAIST), Yong Man Ro `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为AV-STE的模块化流媒体音频-视觉前端，旨在在背景噪声和重叠语音的情况下恢复受损的语义语音标记，从而提高全双工对话系统的鲁棒性。

**💡 创新点**

AV-STE通过使用视觉线索（如唇部运动）来增强受损的语音标记，而无需重新训练底层的语音大语言模型，从而实现了高效的集成。

**🔧 技术方法**

使用了流媒体音频-视觉编码器和噪声自适应融合技术来处理和恢复语音标记。

**📊 数据集**

使用了LRS3数据集（包含约433小时的音频-视觉语音）和Seamless Interaction数据集进行评估。

**📈 对比分析**

与现有的全双工对话模型（如Moshi）进行比较，AV-STE在相同数据集的说话者干扰下，GPT-4o评估的响应一致性从1.42提高到1.91，同时保持了良好的轮流发言行为。

**⚠️ 局限性**

AV-STE目前假设视觉输入是可靠的，对视觉干扰、遮挡或缺失视频的鲁棒性尚未探索。此外，AV-STE在干净语音上的准确性较低，表明对领域转移的敏感性。

---

## 469. GALoc: Gravity Aligned Wireframes for Depth-Free Monocular Floorplan Localization

**arXiv ID:** 2609.08385 | [PDF](https://arxiv.org/pdf/2609.08385v1)

**作者:** Jeahn Han `[一作]` (Gwangju Institute of Science and Technology), Pyojin Kim `[通讯]` (Gwangju Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为GALoc的几何优先框架，用于室内定位，利用重力对齐的线框和RGB图像中的墙交点来实现精确的平面图定位。

**💡 创新点**

GALoc的创新点在于用重力对齐的线框替代深度预测，构建了一个线性约束矩阵来编码垂直性和共面性，从而实现了无深度监督的3D线框恢复。

**🔧 技术方法**

使用了重力对齐的线框提取技术和基于SE(2)的搜索方法进行平面图匹配。

**📊 数据集**

在Structured3D数据集上进行了端到端评估，并在Gibson数据集上进行了带有校准噪声的测试，以及在真实世界中收集的序列。

**📈 对比分析**

与基于深度的基线方法相比，GALoc在可见墙几何时的单图像召回率达到了3倍的提升，在Gibson上实现了88%的顺序定位成功率，而基线方法为68%。

**⚠️ 局限性**

GALoc的局限性在于假设了亚特兰大世界几何，并在墙体结构不可见时会放弃定位，此外，准确性受限于线框提取器的性能。

---

## 470. Geographically Regularized AUC-Maximizing Personalized Federated Learning

**arXiv ID:** 2609.08379 | [PDF](https://arxiv.org/pdf/2609.08379v1)

**作者:** Mayu Hiraishi `[一作]` (Wakayama Medical University), Toshio Shimokawa `[通讯]` (Wakayama Medical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种地理正则化的AUC最大化个性化联邦学习方法（GrAUC-PFL），旨在优化诊断和风险预测模型，同时保持患者数据的本地性。

**💡 创新点**

创新点在于直接优化平滑的成对AUC代理，以学习个性化模型，并通过图形正则化鼓励地理相邻机构具有相似的系数向量。

**🔧 技术方法**

使用了基于交替方向乘子法（ADMM）的高效优化算法来解决结合了不可分解AUC目标函数和图形正则化的优化问题。

**📊 数据集**

在模拟和真实数据应用中使用了多种数据集，包括COVID-19病例监测公共使用数据和模拟生成的数据集。

**📈 对比分析**

与传统的个性化联邦学习（PFL）和直接AUC最大化方法进行比较，GrAUC-PFL在多个场景中表现出更好的判别性能，尤其是在地理相邻机构具有相似数据生成特征时。

**⚠️ 局限性**

GrAUC-PFL的局限性在于假设地理相邻的机构共享相似的模型参数，这在实际中可能并不总是成立。未来的工作可以扩展到其他相似性度量，如局部模型或数据分布的相似性。

---

## 471. IPM-FM: A Foundation Model with Consensus Feature Selection for Industrial Process Monitoring

**arXiv ID:** 2609.08375 | [PDF](https://arxiv.org/pdf/2609.08375v1)

**作者:** Liang Cao `[一作]` (University of British Columbia), Bhushan Gopaluni `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种工业过程监测基础模型（IPM-FM），通过自监督预训练从未标记的工业过程数据中学习通用表示，并使用少量标记数据适应特定监测任务，最终通过不确定性感知预测头生成校准预测。

**💡 创新点**

创新点在于将表示学习与任务监督解耦，首次将基础模型框架应用于工业过程监测，统一多个监测任务，并设计了多标准共识特征选择器和递归滞后特征机制。

**🔧 技术方法**

使用了自监督学习、共识特征选择、递归滞后特征回归头和校准的蒙特卡洛 dropout 不确定性模块等技术。

**📊 数据集**

使用了一个为期七年的柴油闪点软传感器数据集，该数据集包含6157个质量样本和24个在线过程变量。

**📈 对比分析**

与最强的经典回归模型（偏最小二乘法）相比，IPM-FM在RMSE上减少了8.3%，与从头开始的序列基线（LSTM）相比减少了14.6%。在所有基线中，IPM-FM的RMSE最低，R^2最高。

**⚠️ 局限性**

限制在于模型的适应性可能受到特定任务和数据集的影响，且在不同工业环境中的通用性仍需进一步验证。

---

## 472. To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models

**arXiv ID:** 2609.08367 | [PDF](https://arxiv.org/pdf/2609.08367v1)

**作者:** Siru Jiang `[一作]` (University of Chinese Academy of Sciences), Tieniu Tan `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种选择性适应的方法，旨在确定给定测试样本是否应进行适应或跳过适应。

**💡 创新点**

创新点在于引入选择性适应问题，通过评估每个样本的适应效果来提高适应效率，避免不必要的计算开销。

**🔧 技术方法**

使用了交叉增强相似性（CAS）作为基线方法，该方法基于增强视图之间的预测相似性来决定是否进行适应。

**📊 数据集**

使用了多个数据集进行评估，包括ImageNet及其变体（如ImageNet-A、ImageNet-V等）和多个细粒度数据集（如Flowers102、Caltech101等）。

**📈 对比分析**

与随机跳过和现有的OOD检测方法进行比较，CAS在多个TTA框架下的AUC约为90%，并在跳过85%的适应时保持或提高了整体准确性。

**⚠️ 局限性**

限制在于选择性适应的有效性可能依赖于特定的增强策略和阈值设置，可能在某些情况下未能充分利用所有可用的适应机会。

---

## 473. Erased Postulates, Identity Types and Quotients

**arXiv ID:** 2609.08578 | [PDF](https://arxiv.org/pdf/2609.08578v1)

**作者:** Nils Anders Danielsson `[一作]` `[通讯]`, Nils Anders Danielsson

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了在带有擦除注释的类型理论中，是否可以假设某种类型是有值的，同时仍然保证程序不会卡住。

**💡 创新点**

创新点在于将之前对一致擦除公设的保证扩展到带有身份类型的类型理论，并展示了如何使用擦除公设来支持商类型。

**🔧 技术方法**

使用了类型理论和机器检查的Agda证明。

**📊 数据集**

使用了带有身份类型和擦除公设的类型理论的形式化，具体数据集未明确提及。

**📈 对比分析**

与之前的方法相比，本文的方法在处理擦除公设和身份类型时提供了更强的安全性，确保程序在使用擦除公设的情况下仍能正确计算。

**⚠️ 局限性**

限制在于不允许无限制的擦除匹配与擦除公设的等价性，且在某些情况下，擦除匹配可能导致程序卡住。

---

## 474. PLC-Bin2Src: Retrieving Corresponding Structured Text Source Files for PLC Binaries

**arXiv ID:** 2609.08563 | [PDF](https://arxiv.org/pdf/2609.08563v1)

**作者:** Ang Jia `[一作]` (Dalian University of Technology), Xiaochen Li `[通讯]` (Dalian University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为PLC-Bin2Src的跨平台二进制到源代码匹配框架，用于从CODESYS、GEB、OpenPLC v2和OpenPLC v3生成的PLC二进制文件中检索相应的结构化文本（ST）源文件。

**💡 创新点**

PLC-Bin2Src是首个针对PLC二进制文件的二进制到源代码匹配框架，提出了多通道语义表示方法，结合控制-数据流图（CDFG）、函数调用图（FCG）和恢复符号的相似性来对源候选进行排名。

**🔧 技术方法**

使用了控制-数据流图（CDFG）、函数调用图（FCG）和恢复符号的相似性计算技术，并通过固定等权重融合这些相似性来进行候选排名。

**📊 数据集**

使用了PLC-BEAD数据集，该数据集包含2431个样本，评估了2358个样本的二进制与源代码对应关系。

**📈 对比分析**

与现有方法相比，PLC-Bin2Src在四个平台上实现了95.89%的Recall@1，99.66%的Recall@5，以及0.9769的平均倒数排名（MRR），显示出其在二进制到源代码匹配中的优越性能。

**⚠️ 局限性**

局限性包括对不同PLC平台的支持有限，且在某些情况下，恢复的符号可能不足以提供唯一的源代码对应关系。

---

## 475. Certified Topological Interaction in Neural Representations: Class Disentanglement Is Mostly Pairwise

**arXiv ID:** 2609.08561 | [PDF](https://arxiv.org/pdf/2609.08561v1)

**作者:** Sushovan Majhi `[一作]` `[通讯]` (George Washington University), Sushovan Majhi (George Washington University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文测量了类的解缠结程度，使用了交集欧拉特征曲线（Intersection Euler Characteristic Profile）来评估标记点云之间的拓扑交互。

**💡 创新点**

创新点在于提出了一种新的测量方法，能够对多类的解缠结进行对称、无边界的统计认证，并且能够比较不同层次、不同模型的解缠结程度。

**🔧 技术方法**

使用了交集欧拉特征曲线（Intersection ECP）和精确的置换检验方法来进行统计分析。

**📊 数据集**

使用了MNIST和CIFAR-10/100数据集，涉及111个训练网络和52650个认证测量。

**📈 对比分析**

与其他方法相比，交集ECP在测量类的解缠结方面表现出更高的准确性，且能够提供统计认证。性能上，交互商（interaction quotient）与混淆矩阵的相关性为0.83，且在深层网络中解缠结程度显著提高。

**⚠️ 局限性**

限制在于测量依赖于PCA降维，可能会丢失某些信息；此外，交集ECP在处理高维数据时的有效性可能会降低。

---

## 476. STSG-VQA: Evidence-Grounded Temporal Question Answering from Surgical Spatio-Temporal Scene Graphs

**arXiv ID:** 2609.08543 | [PDF](https://arxiv.org/pdf/2609.08543v1)

**作者:** Jing Li `[一作]` (University of Leeds), Duygu Sarikaya `[通讯]` (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多层次结构化的时间监督方法，增强了手术场景图的时间推理能力，并创建了STSG-VQA基准以生成基于证据的时间问答对。

**💡 创新点**

创新点在于引入了事件中心的结构化时间推理方法，结合了对象级连续性、事件级交互连续性和过程级连接性，从而更好地建模手术状态随时间的演变。

**🔧 技术方法**

使用了多层次结构化时间监督方法和基于证据的时间查询框架，结合了时空场景图（STSG）进行推理。

**📊 数据集**

使用了包含45个腹腔镜手术视频的STSG-VQA基准，包含18458个问答对，覆盖七个时间类别。

**📈 对比分析**

与现有的手术VQA基准相比，STSG-VQA在时间推理方面表现出显著的性能提升，特别是在并发性和边界推理任务上，微观准确率提高了24.39和19.56个百分点。

**⚠️ 局限性**

限制在于STSG-VQA仅包含45个视频，无法验证其在不同机构、采集系统或患者群体中的泛化能力，同时生成的模板和阶段条件问题可能保留文本和工作流程的先验知识。

---

## 477. Physical Law Ecology: mapping multi-mechanism ecologies as the zeroth step of data-driven scientific discovery

**arXiv ID:** 2609.08536 | [PDF](https://arxiv.org/pdf/2609.08536v1)

**作者:** Xiongheng Bian `[一作]` (Nantong University), Xiaoyan Shen `[通讯]` (Nantong University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了物理法则生态学框架，旨在从数据中自动识别多机制系统中的独立控制机制数量K^*，并挖掘相应的物理方程。

**💡 创新点**

创新点在于将K^*的确定作为科学发现的首要步骤，打破了传统方法假设单一控制方程的局限，能够自动挖掘多种共存的物理方程及其动态演变。

**🔧 技术方法**

使用了贝叶斯信息准则（BIC）来自动确定K^*，并结合符号回归技术进行方程挖掘和权重场构建。

**📊 数据集**

应用于四个不同的系统，包括弹性体力学、池沸腾、星系动力学和液滴蒸发，使用了相应的实验数据集。

**📈 对比分析**

与传统的单方程拟合方法相比，该框架在多机制系统中实现了67-72%的误差降低，且保持了完全的可解释性。

**⚠️ 局限性**

限制在于当前操作符的词汇有限，无法覆盖三角函数、特殊函数或其他复杂物理形式，且在训练条件远离时外推能力下降。

---

## 478. Layer Selection in VLMs for Zero-Shot OOD Detection via Multi-Resolution Entropy Estimation

**arXiv ID:** 2609.08524 | [PDF](https://arxiv.org/pdf/2609.08524v1)

**作者:** Shyam Nandan Rai `[一作]` (University of Bamberg), Christian Ledig `[通讯]` (University of Bamberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本论文提出了一种基于多分辨率熵估计的层选择方法，用于在医疗图像分析中进行零-shot OOD（分布外）检测。

**💡 创新点**

创新点在于首次系统性地分析了医疗图像中不同层次的表示对OOD检测的影响，并提出了一种多分辨率熵估计策略，以提高层选择的稳定性和鲁棒性。

**🔧 技术方法**

使用了多分辨率熵估计技术来聚合多个直方图统计数据，从而实现稳健的中间层选择。

**📊 数据集**

使用了两个医疗OOD基准数据集：MIDOG（组织病理学）和OASIS（脑MRI），涵盖不同的成像模态和分布转变类型。

**📈 对比分析**

与现有的基于最终层嵌入的方法（如MCM和Ju）进行比较，提出的方法在多个配置中表现出一致的性能提升，尤其是在FPR95指标上显著降低，显示出更强的鲁棒性。

**⚠️ 局限性**

限制在于该方法依赖于基础的VLM骨干网络，且目前仅在两个医疗模态上进行了验证，未来需要在更多成像领域进行进一步验证。

---

## 479. TASG-Explore: Traversability-Aware Sector-Guided Exploration for Ground Robot on Uneven Terrain

**arXiv ID:** 2609.08512 | [PDF](https://arxiv.org/pdf/2609.08512v1)

**作者:** Shaocong Wang `[一作]` (Chinese Academy of Sciences), Lianqing Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为TASG-Explore的框架，用于在不平坦地形上进行地面机器人自主探索，旨在提高探索效率、覆盖完整性和地形安全性。

**💡 创新点**

创新点在于结合了分层可通行性分析、增量扇区区域分割和动态拓扑路线图规划，能够在复杂环境中实现高效的自主探索。

**🔧 技术方法**

使用了可变体素地面拟合和自适应8位障碍物编码技术，进行分层可通行性分析，并通过增量扇区分割组织未知空间。

**📊 数据集**

在多种具有挑战性的环境中进行了基准实验，包括洞穴、森林和崎岖的山丘，验证了方法的有效性。

**📈 对比分析**

与六种先进的探索规划方法进行了比较，TASG-Explore在探索效率和覆盖率上表现最佳，探索效率提高了51%，在崎岖山丘场景中覆盖率增加了2.95倍。

**⚠️ 局限性**

该方法主要针对可以用支撑地面表示的户外场景，不适用于具有垂直重叠可通行结构的多层环境，未来工作将扩展到支持多层场景的环境表示和探索规划。

---

## 480. Temporal State Transport in Video Generation: Diagnosing and Correcting Spectral Imbalance

**arXiv ID:** 2609.08505 | [PDF](https://arxiv.org/pdf/2609.08505v1)

**作者:** Luyao Tang `[一作]` (University of Hong Kong), Chaoqi Chen `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了视频生成中的时间状态传输，提出了一种新的方法来诊断和纠正光谱不平衡，以提高视频生成的质量和一致性。

**💡 创新点**

创新点在于引入了光谱张力作为诊断工具，识别时间传输中的两种失败模式：碎片化传输和过度混合热点，并提出了光谱传输稳态作为一种训练无关的调节器。

**🔧 技术方法**

使用了光谱张力和光谱传输稳态作为核心技术，进行时间状态的诊断和调节。

**📊 数据集**

在预训练的视频生成模型上进行了实验，使用了VBench数据集进行定量评估，并进行了人类评估。

**📈 对比分析**

与原始模型相比，提出的方法在运动平滑性、美学质量和成像质量等方面表现更好，定量评估结果显示整体得分有所提高，且在每个维度上保持竞争力。

**⚠️ 局限性**

限制在于该方法依赖于预训练模型，可能在某些特定场景下的适用性有限，且调节参数的选择可能影响结果的稳定性。

---

## 481. AURORA: Active Uncertainty-Driven Re-Orientation for In-Hand Reconstruction

**arXiv ID:** 2609.08493 | [PDF](https://arxiv.org/pdf/2609.08493v1)

**作者:** Feiyu Zhao `[一作]` (ShanghaiTech University), Chenxi Xiao `[通讯]` (ShanghaiTech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了AURORA框架，用于在手中主动重建物体，结合了在线物体中心重建和不确定性驱动的再定向。

**💡 创新点**

创新点在于引入Ray-GPIS规划器，通过估计方向性重建不确定性来指导可执行的手中旋转动作，从而提高重建的完整性和效率。

**🔧 技术方法**

使用了不确定性驱动的下一最佳视图规划技术（Ray-GPIS），结合了CAD-free的6D姿态跟踪和轻量级几何重建。

**📊 数据集**

在六个具有不同几何形状的真实物体上进行了实验评估。

**📈 对比分析**

与非主动旋转策略和其他主动视图规划基线进行了比较，AURORA在重建质量和信息获取效率上表现更优，Ray-GPIS在重建性能、动作排名质量和规划效率上也超越了其他基线。

**⚠️ 局限性**

局限性包括对6D姿态跟踪误差的敏感性、固定的6秒重新规划间隔可能需要根据不确定性或跟踪信心进行调整，以及偶尔的物体滑动或掉落问题，强调了对更稳健控制和仿真到现实转移的需求。

---

## 482. An Evidence Model for Agentic Processes: Evidence Claims, Trust Assumptions, and Policy Assessment

**arXiv ID:** 2609.08481 | [PDF](https://arxiv.org/pdf/2609.08481v1)

**作者:** Arslan Brömme `[一作]` `[通讯]` (Independent Researcher), Arslan Brömme (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种代理人工智能系统的证据声明模型，旨在明确代理过程中的证据属性和限制。

**💡 创新点**

创新点在于将证据声明结构化，区分不同的证据属性，并明确其机制、假设、限制和威胁。

**🔧 技术方法**

使用了证据声明模型，结合哈希、签名、序列引用等机制来支持证据属性。

**📊 数据集**

未使用特定数据集，而是通过简化的治理示例来说明证据声明及其限制。

**📈 对比分析**

与现有的数字时间戳、安全审计日志等相关工作进行了比较，但未进行实证性能、安全或合规性评估。

**⚠️ 局限性**

模型并未解决所有保证问题，如无法保证代理行为的真实、完整捕获、正确的政策解释等。

---

## 483. Selective boundary condition reduction via learned error gating

**arXiv ID:** 2609.08461 | [PDF](https://arxiv.org/pdf/2609.08461v1)

**作者:** Daniel Fernández `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Dominik Riedelbauch `[通讯]` (Schaeffler Technologies AG & Co. KG)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

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

## 484. CreaMem: A Scene-Aware Memory Architecture for Personalized Agents

**arXiv ID:** 2609.08550 | [PDF](https://arxiv.org/pdf/2609.08550v1)

**作者:** Qixuan Sun `[一作]` (City University of Hong Kong), Chen Ma `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为CreaMem的场景感知记忆架构，旨在通过将记忆划分为多个生活场景记忆来改善个性化长时记忆的组织和检索。

**💡 创新点**

创新点在于场景感知的记忆组织和双重编码机制，使得同一事件可以从不同的视角进行检索，减少跨场景干扰。

**🔧 技术方法**

使用了Meta Memory Manager、Episodic Memory和Life Scene Memories等技术，结合了平衡采样策略进行检索。

**📊 数据集**

在LoCoMo和LongMemEval-S两个长时记忆基准上进行了实验，验证了CreaMem的有效性。

**📈 对比分析**

与多种现有方法进行了比较，CreaMem在所有评估指标上均表现出色，特别是在多跳推理性能上有显著提升，表明其记忆组织和检索方式的优势。

**⚠️ 局限性**

局限性包括固定的生活/工作/兴趣分类可能不适用于所有用户和文化，且系统开销较大，查询延迟较高。

---

## 485. Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token Context?

**arXiv ID:** 2609.08574 | [PDF](https://arxiv.org/pdf/2609.08574v1)

**作者:** Sara Rizwan `[一作]` (Shadan Women's College of Engineering and Technology), Samaanah Abdus Salam `[通讯]` (Shadan Women's College of Engineering and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了长上下文语言模型中的注意力沉没现象，提出了SinkProbe工具来测量注意力沉没、激活、位置偏差和近期差距等指标，并应用于四种不同架构的模型。

**💡 创新点**

创新点在于将注意力沉没和上下文读取不均匀性视为两个独立的问题，并通过SinkProbe提供了可比较的测量方法。

**🔧 技术方法**

使用了门控注意力机制、Kimi Delta Attention和注意力残差等新技术。

**📊 数据集**

使用了自生成的检索任务数据集，长度从4K到1M的上下文。

**📈 对比分析**

与现有方法比较，发现训练目标会导致注意力沉没，而门控机制在本研究的规模下未能重现其预期效果。模型的注意力沉没和激活值独立变化，且位置偏差在不同上下文长度下表现不同。

**⚠️ 局限性**

限制在于模型规模较小，无法完全代表大型模型的行为，且未能实现真实Kimi K3机制中的增量规则修正。

---

## 486. AgentGrad: Intervention-guided Prompt Optimization for Multi Agent Systems

**arXiv ID:** 2609.08572 | [PDF](https://arxiv.org/pdf/2609.08572v1)

**作者:** Jaewon Chu `[一作]` (Korea University), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为AgentGrad的多代理系统（MAS）提示优化框架，旨在解决现有文本梯度方法在梯度提取和聚合阶段的系统性局限性。

**💡 创新点**

创新点在于引入了顺序干预和语义文本梯度抽象，前者能够识别出负责每个失败的代理并生成细粒度的监督信号，后者则将样本级梯度聚类为共享纠正模式的语义小批量，并将每个小批量抽象为单一的通用梯度。

**🔧 技术方法**

使用了顺序干预和语义文本梯度抽象技术。

**📊 数据集**

在五个MAS基准上进行评估，包括多步问答（HotpotQA）、声明验证（HoVer）、指令跟随（IFBench）、隐私意识委托（PUPA）和数学推理（MATH），使用了开源（Qwen3-8B）和专有（GPT-5-mini）模型作为基础。

**📈 对比分析**

与MIPROv2、TextGrad和GEPA等三种最先进的提示优化算法进行比较，AgentGrad在所有基准上均表现出色，平均提高了11.76分，并且优化时间比下一个最快的基线快2.5倍。

**⚠️ 局限性**

局限性在于未详细讨论AgentGrad在特定任务或数据集上的适用性，可能在某些特定场景下表现不如预期。

---

## 487. Leveraging Cardiac Imaging to Improve ECG-Based Detection of Chagas Disease in Resource-Constrained Settings

**arXiv ID:** 2609.08582 | [PDF](https://arxiv.org/pdf/2609.08582v1)

**作者:** Laura Alvarez-Florez `[一作]` (Amsterdam University Medical Center), Fleur V. Y. Tjong `[通讯]` (Amsterdam University Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `109c2b71-d051-425c-831f-0c544c24280d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种通过对比预训练将心脏磁共振成像（CMR）获得的结构知识转移到心电图（ECG）的方法，以改善在资源有限的环境中对恰加斯病的检测。

**💡 创新点**

创新点在于通过对比学习将CMR和ECG的配对数据进行对齐，从而使得ECG模型能够捕捉到心脏的结构和功能信息，即使在预训练数据中没有恰加斯病的病例。

**🔧 技术方法**

使用了对比学习技术，具体为不对称的InfoNCE目标，来对齐ECG编码器与CMR嵌入空间。

**📊 数据集**

使用了来自英国生物银行的63,193对ECG-CMR检查数据集进行预训练，并在CODE-15%和SaMi-Trop数据集上进行微调。

**📈 对比分析**

与未对齐的ECG-FM基线相比，经过对比预训练的模型在五折交叉验证中在Top5%-TPR上提高了5个百分点，AUROC达到了0.851，显示出显著的性能提升。

**⚠️ 局限性**

限制在于预训练阶段未包含恰加斯病患者，因此模型的泛化能力在不同疾病和人群中仍需进一步验证。

---

## 488. Effects of model architecture and learning strategies on deep learning-based recognition of activated sludge microscopic images and comparison with quantitative image analysis

**arXiv ID:** 2609.08570 | [PDF](https://arxiv.org/pdf/2609.08570v1)

**作者:** Suguru Hakoshima `[一作]` (University of Tokyo), Fumiyuki Nakajima `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究探讨了模型架构和学习策略对活性污泥显微图像深度学习识别性能的影响，并定量比较了深度学习与定量图像分析（QIA）的表现。

**💡 创新点**

创新点在于首次系统评估了基于变换器的模型和自监督学习方法在活性污泥显微图像分析中的有效性，并探讨了图像下采样策略对分析性能的影响。

**🔧 技术方法**

使用了深度学习技术，包括卷积神经网络（CNN）和变换器（ViT），并采用了自监督学习和传统的监督学习进行模型预训练。

**📊 数据集**

使用了来自日本两个污水处理厂的三种类型的活性污泥样本的显微图像数据集，共获取了超过300张图像。

**📈 对比分析**

通过基准分类任务比较了深度学习与QIA的准确性，结果显示深度学习在准确性上优于QIA，并且在所需图像数量上更为高效。

**⚠️ 局限性**

本研究的局限性在于缺乏对模型可解释性的讨论，以及结果的普遍适用性有限，未来需要在更广泛的实际应用中验证这些发现。

---

## 489. The Fusion Frame Phase Retrieval

**arXiv ID:** 2609.08531 | [PDF](https://arxiv.org/pdf/2609.08531v1)

**作者:** Haixia Liu `[一作]` (Huazhong University of Science and Technology), Yang Wang `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了融合框架相位恢复问题，提出了一种基于梯度下降和两阶段初始化的方法，能够从测量的幅度中重建信号。

**💡 创新点**

创新点在于提供了关于秩-r正交投影矩阵的集中不等式，并证明了在测量复杂度为O(dlog^2 d)的情况下，梯度下降法能够线性收敛到目标信号。

**🔧 技术方法**

使用了梯度下降法和两阶段初始化技术。

**📊 数据集**

使用了从Haar测度中抽取的i.i.d.秩-r正交投影矩阵作为数据集。

**📈 对比分析**

与现有的相位恢复方法（如AltMinPhase和Wirtinger Flow）相比，本文的方法在测量复杂度上具有优势，且在数值实验中验证了其有效性。

**⚠️ 局限性**

限制在于该方法的收敛性依赖于初始化的质量，且在高维情况下可能仍然面临计算复杂性的问题。

---

## 490. Towards Actionable Strategy Certificates in Stochastic Parity Games

**arXiv ID:** 2609.08529 | [PDF](https://arxiv.org/pdf/2609.08529v1)

**作者:** Christel Baier `[一作]` (Technische Universität Dresden), Anne-Kathrin Schmuck `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的方法，用于在具有定量目标的随机奇偶游戏中合成大量获胜策略。该方法引入了一种局部和宽松的表示方式，增强了合成策略的可信度。

**💡 创新点**

创新点在于将随机不变体的证书扩展到游戏的设置中，并将其重新解释为一种局部和宽松的策略表示，允许在不确定和对抗环境中进行有效的合成、适应和运行时策略提取。

**🔧 技术方法**

使用了随机不变体的证书和几乎肯定获胜的策略模板相结合的技术，形成了一种新的局部可操作证书。

**📊 数据集**

使用了随机奇偶游戏的案例研究，具体数据集未详细说明。

**📈 对比分析**

与传统的策略合成方法相比，提出的方法在合成、适应性和运行时策略提取方面表现出更高的效率和灵活性，尤其是在处理不确定性和对抗性环境时。

**⚠️ 局限性**

局限性在于该方法的实现和实验数据仍处于原型阶段，且在复杂模型的合成时间上可能存在改进空间。

---

## 491. Same Values, Different Languages? From Multilingual Probing to Steering LLMs Toward Chinese Social Values

**arXiv ID:** 2609.08515 | [PDF](https://arxiv.org/pdf/2609.08515v1)

**作者:** Yuemei Xu `[一作]` (Beijing Foreign Studies University), Aishan Liu `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了大型语言模型（LLMs）在多语言环境中对中国社会价值（CSV）的行为偏好，构建了C-Voices数据集，并提出了一种无需微调的价值向量引导方法。

**💡 创新点**

创新点在于构建了第一个全面的多语言对比探测数据集C-Voices，涵盖12个CSV维度，并提出了一种基于隐藏状态差异的价值向量引导方法，能够在推理过程中选择性地干预价值敏感层。

**🔧 技术方法**

使用了对比探测和价值向量引导的方法，结合了隐藏状态的差异来识别价值方向，并在推理时进行干预。

**📊 数据集**

使用了C-Voices数据集，该数据集包含86400个基于困境的实例，涵盖中文、英文、日文、俄文、阿拉伯文和西班牙文六种语言。

**📈 对比分析**

与四种基线方法（原始模型、稀疏自编码器、基于熵的方法和因果干预方法）进行比较，提出的方法在所有模型和语言设置中均表现出一致的积极改进，平均Likert评分提高了5.87，优于其他方法。

**⚠️ 局限性**

限制在于该方法的有效性可能依赖于模型的特定架构和训练数据，且在某些情况下可能会对模型的通用能力产生一定的负面影响。

---

## 492. The conservative turn in science: The changing character of knowledge recombination

**arXiv ID:** 2609.08468 | [PDF](https://arxiv.org/pdf/2609.08468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 493. PGMT: Perceptive General Motion Tracking for Humanoid Robots

**arXiv ID:** 2609.08511 | [PDF](https://arxiv.org/pdf/2609.08511v1)

**作者:** Hongyi Li `[一作]` (Zhejiang University), Hongtao Wang `[通讯]` (MirrorMe Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aaccfe5c-6b26-4208-b23c-35331481e142` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为PGMT的感知通用运动跟踪管道，旨在使类人机器人能够在复杂地形上适应并执行全身运动。

**💡 创新点**

PGMT的创新点在于它能够直接跟踪与地形无关的运动参考，而无需生成与地形匹配的参考轨迹，从而实现了对复杂地形的适应。

**🔧 技术方法**

使用了意图融合模块（IFM）和多头评论家网络等技术，结合了运动感知和地形感知。

**📊 数据集**

使用了LAFAN1数据集进行训练，并在多种真实世界的地形上进行了测试，包括平坦地形、坡道、楼梯和随机粗糙地形。

**📈 对比分析**

与其他运动跟踪基线进行比较，PGMT在复杂地形上的完成率达到了87.81%，而传统方法的完成率显著低于此，显示出PGMT在复杂环境中的优越性能。

**⚠️ 局限性**

PGMT的局限性在于其高度图只编码几何信息，而不包含环境语义或可交互性，限制了其在非结构化环境中的应用潜力。

---

## 494. WiDiff: Extracting Changes from Wikidata's Edit History

**arXiv ID:** 2609.08508 | [PDF](https://arxiv.org/pdf/2609.08508v1)

**作者:** Carolina Cortés `[一作]` (Hasso Plattner Institute), Felix Naumann `[通讯]` (Hasso Plattner Institute)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种名为WiDiff的工具，用于从Wikidata的完整编辑历史中提取变化，并提供一个统一的接口以进行大规模分析查询。

**💡 创新点**

创新点在于提供了一个开源工具，能够处理Wikidata的完整编辑历史，并支持对其进行大规模的分析查询，这是之前的工具所不具备的。

**🔧 技术方法**

使用了Python编程语言和关系数据库来存储和处理数据，采用了增量读取和并行处理的方式来提高效率。

**📊 数据集**

使用了2025年6月的Wikidata数据集，该数据集包含2125个压缩的XML文件，总大小为2.2TB。

**📈 对比分析**

与现有方法相比，WiDiff在处理速度和存储效率上表现优越，能够以每秒59个实体和1126个修订的速度处理数据，且存储需求显著低于其他方法。

**⚠️ 局限性**

限制在于当前工具尚未提取实体别名或站点链接的变化，且只能处理单一语言的标签和描述变化，未来需要扩展到多语言支持。

---

## 495. When Topology Betrays Privacy: Lattice-Based Reconstruction Attacks on Secure Aggregation in Decentralized Federated Learning

**arXiv ID:** 2609.08476 | [PDF](https://arxiv.org/pdf/2609.08476v1)

**作者:** Wenrui Yu `[一作]` (Aalborg University), Qiongxiu Li `[通讯]` (Aalborg University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了去中心化联邦学习（DFL）中安全聚合（SA）的隐私漏洞，提出了一种基于格的重构攻击方法，揭示了稀疏网络拓扑如何导致信息泄露。

**💡 创新点**

创新点在于将DFL中的SA过程重新表述为一个统一的数学框架，并通过与隐子集和问题（HSSP）的形式联系，展示了如何利用结构和观察不对称性来重构私有模型状态。

**🔧 技术方法**

使用了基于格的重构方法，结合格约简和结构过滤技术来重构受保护的模型状态。

**📊 数据集**

在图像、表格和文本任务上进行了评估，使用了多种数据集以验证攻击的有效性。

**📈 对比分析**

通过与现有方法的比较，结果表明，合谋的半诚实节点能够恢复诚实节点的原始本地更新，表明SA在DFL中并不能保证隐私。

**⚠️ 局限性**

限制在于攻击的最强精确更新重构结果仅适用于第一轮训练，且需要足够的拓扑信息来过滤虚假候选项。

---

## 496. Safe Task Planning with Long-Term Graph Memory for Embodied Agents

**arXiv ID:** 2609.08444 | [PDF](https://arxiv.org/pdf/2609.08444v1)

**作者:** Siyuan Li `[一作]` (Harbin Institute of Technology), Peng Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的安全任务规划框架SafeMem，该框架通过构建和维护环境的长期语义图记忆来解决在部分可观察环境中生成安全高层次动作的挑战。

**💡 创新点**

创新点在于引入了长期图记忆和迭代风险预测与重新规划机制，使得代理能够在部分可观察环境中主动识别潜在危险并促进安全决策。

**🔧 技术方法**

使用了大型语言模型（LLM）和视觉语言模型（VLM）来进行风险预测和任务规划，同时结合了图记忆来增强安全意识。

**📊 数据集**

在IS-Bench基准和真实机器人平台上进行了广泛的实验，验证了SafeMem框架的有效性。

**📈 对比分析**

与多种最先进的VLM驱动的安全任务规划方法进行了比较，SafeMem在安全成功率（SSR）上显著优于这些基线方法，尤其是在部分可观察环境中。

**⚠️ 局限性**

限制在于假设安全隐患出现在历史或当前观察中，无法仅凭更广泛的物理上下文推断潜在但未见的危险。

---

## 497. Personalizing LLM Agent Memory Using Biometrics

**arXiv ID:** 2609.08558 | [PDF](https://arxiv.org/pdf/2609.08558v1)

**作者:** Yanhong Qian `[一作]` (Anhui University), Isao Echizen `[通讯]` (National Institute of Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Bio-Memory的生物识别感知记忆架构，用于个性化大语言模型（LLM）代理的记忆管理，结合语义相似性和生物识别匹配来检索用户特定的记忆。

**💡 创新点**

创新点在于将生物识别嵌入添加到每个记忆条目中，使得记忆检索不仅依赖于语义相关性，还依赖于当前请求者的生物识别身份，从而在多用户环境中实现个性化记忆的准确检索。

**🔧 技术方法**

使用了生物识别技术（如面部识别和掌纹识别）来增强记忆检索过程，并在此基础上构建了Bio-Memory架构。

**📊 数据集**

在LoCoMo数据集上进行评估，结合了7个面部基准和10个掌纹协议，构建了一个10用户共享代理的记忆存储。

**📈 对比分析**

通过与现有方法的比较，Bio-Memory在所有数据集上都能有效区分拥有者和非拥有者的查询。在面部个性化下，F1和BLEU-1的平均差距分别达到27.29%和21.15%；在掌纹个性化下，差距为25.75%和19.22%。

**⚠️ 局限性**

限制在于该方法依赖于生物识别技术的准确性和可靠性，可能在不同环境或条件下表现不一致。此外，生物识别数据的隐私和安全性问题也需要进一步考虑。

---

## 498. Movable-Element STAR-RIS for 6G: From Programmable Propagation to Programmable Geometry

**arXiv ID:** 2609.08545 | [PDF](https://arxiv.org/pdf/2609.08545v1)

**作者:** Wali Ullah Khan `[一作]` (University of Luxembourg), Muhammad Adil `[通讯]` (University of Rome Tor Vergata)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

本文介绍了可移动元素的同时传输和反射可重构智能表面（ME-STAR-RIS），探讨了其操作原理、运动架构及在通信、安全、近场系统、传感和高机动网络中的应用。

**💡 创新点**

创新点在于引入了几何可编程性，使得表面元素可以在规定区域内重新定位，从而改变传播距离、干扰和信号组合等特性。

**🔧 技术方法**

使用了电磁和几何重构的结合技术，允许表面元素在保持电子控制的同时进行物理位移。

**📊 数据集**

论文中使用了一个代表性的案例研究，比较了固定和可移动STAR-RIS架构的性能，展示了局部位移带来的谱效率增益。

**📈 对比分析**

通过案例研究，ME-STAR-RIS在低至中等功率范围内相较于固定架构表现出约11%到12.6%的谱效率提升，表明几何可编程性在信道丰富性较高时能显著提高性能。

**⚠️ 局限性**

限制在于实际硬件的实现挑战，包括运动范围、能耗、可靠性和校准负担等问题，需在设计中考虑这些因素以实现可行的ME-STAR-RIS部署。

---

## 499. Individual Text Corpora Predict User-Specific Knowledge: Benchmarks of Individualized Knowledge Simulation

**arXiv ID:** 2609.08532 | [PDF](https://arxiv.org/pdf/2609.08532v1)

**作者:** Christoph Wigbels `[一作]` (Bergische Universitat Wuppertal), Markus J. Hofmann `[通讯]` (Bergische Universitat Wuppertal)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了是否可以利用来自搜索历史的个体文本语料库（ICs）来模拟个体知识。研究收集了316名成人的ICs，并让他们回答36个多项选择知识题，比较了几种大型语言模型（LLMs）的表现，最终只有Qwen3-1.7B模型表现出色。

**💡 创新点**

创新点在于将个体文本语料库与检索增强生成（RAG）结合，首次实证展示了个体搜索历史所衍生的文本语料库能够携带可检测的个体知识信号。

**🔧 技术方法**

使用了低秩适应（LoRA）技术对Qwen3-1.7B进行任务特定的微调，并结合检索增强生成（RAG）方法。

**📊 数据集**

使用的数据集为316名参与者的Google搜索历史，构建的个体语料库平均大小为320万标记。

**📈 对比分析**

与参与者的表现相比，Qwen3-1.7B在公共问题上表现优于参与者，但在非公共问题上表现较差，表明可能存在训练数据污染。模型的匹配准确率显著高于随机水平，但在概率分配上表现不佳，表明模型对个体反应模式的校准较差。

**⚠️ 局限性**

限制在于样本主要集中在年轻、高学历的女性，可能影响结果的普遍性。此外，模型的高答题正确率伴随高过度自信，导致校准性能差，且仅评估了单一模型架构和检索配置，未能比较其他模型或策略。

---

## 500. Neural Centroidal Voronoi Tessellations

**arXiv ID:** 2609.08497 | [PDF](https://arxiv.org/pdf/2609.08497v1)

**作者:** Jiacheng Xu `[一作]` (Peking University), Peng-Shuai Wang `[通讯]` (Peking University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于学习的框架Neural CVT，用于高效的高质量表面重网格化，通过学习的递归过程直接从表面几何中预测种子更新。

**💡 创新点**

创新点在于将CVT优化建模为学习的递归过程，能够在推理时绕过昂贵的Voronoi图构建，从而实现比传统求解器快一个到两个数量级的加速，同时保持相当的几何保真度。

**🔧 技术方法**

使用了图神经网络作为几何编码器和基于GRU的递归优化器，结合自监督学习来优化CVT能量。

**📊 数据集**

使用了Thingi10K数据集，随机选择1000个网格进行训练，并在96个网格上进行测试，涵盖有机形状和具有尖锐特征的CAD模型。

**📈 对比分析**

与传统的基于优化的重网格化算法（如L-BFGS和CWF）进行比较，Neural CVT在网格质量指标上表现优越，同时在特征对齐性能上与最先进的方法相当，迭代优化速度提高了一个到两个数量级。

**⚠️ 局限性**

限制在于RVD提取在低分辨率下对薄结构的处理困难，当前架构更适合中等分辨率的重网格化或简化任务，且缺乏严格的数学收敛保证。

---

## 501. SignRefine: Adapting Foundational Video Models for Sign Language Generation

**arXiv ID:** 2609.08496 | [PDF](https://arxiv.org/pdf/2609.08496v1)

**作者:** Anton Pelykh `[一作]` (University of Surrey), Richard Bowden `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为SignRefine的手语视频生成模型，该模型通过仅使用2D关键点条件生成可理解的手语，克服了现代视频扩散模型在手语生成中的不足。

**💡 创新点**

创新点在于引入了局部适配器和空间定位技术，能够选择性地细化手和面部区域，从而提高生成的手语视频的可理解性和视觉质量。

**🔧 技术方法**

使用了预训练的视频扩散变换器（DiT）和局部适配器，结合空间定位技术进行手语视频生成。

**📊 数据集**

使用了名为DATASET的大规模手语视频数据集，该数据集包含多样的手语者外观、环境和自然对话场景。

**📈 对比分析**

与现有的最强基线模型相比，SignRefine在手部姿势精度指标上提高了30%，并且在80%以上的比较中被手语用户偏好，显示出更好的视觉质量和可理解性。

**⚠️ 局限性**

限制在于视频扩散模型计算成本高，生成81帧的剪辑需要几分钟，且模型可能会在快速运动中模糊细节。此外，依赖于现成的姿态估计器可能导致输入条件的噪声，从而影响生成质量。

---

## 502. SRPO: Setwise Relative Policy Optimization for Multi-Agent LLMs

**arXiv ID:** 2609.08452 | [PDF](https://arxiv.org/pdf/2609.08452v1)

**作者:** Shengtian Yang `[一作]` (Southeast University), Lei Feng `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多智能体强化学习框架SRPO（Setwise Relative Policy Optimization），用于优化多智能体大语言模型系统的决策过程。

**💡 创新点**

SRPO通过定义活动集作为环境转变的最小输出集合，提供了一种独立于工作流程的多智能体动作表示，能够同时处理劳动分工和联合共演化的情况。

**🔧 技术方法**

使用了强化学习（RL）和政策优化技术，特别是通过归一化的相对优势更新来优化多智能体决策。

**📊 数据集**

在数学推理和多轮搜索任务中进行了实验，使用了多个模型规模（如Qwen3-4B/8B和Qwen2.5-3B/7B）进行评估。

**📈 对比分析**

与现有的单智能体GRPO和其他多智能体基线方法进行比较，SRPO在多个基准测试中表现出更高的准确率和覆盖率，尤其在搜索任务中取得了显著的性能提升。

**⚠️ 局限性**

主要的局限性在于评估范围仅涵盖推理和检索任务，未涉及其他领域（如编码或开放式交流系统），并且SRPO的行为在强相关的成员更新下可能受到协方差项的影响。

---

## 503. GSComplete: Gaussian Splat Completion with 2D Diffusion Priors

**arXiv ID:** 2609.08449 | [PDF](https://arxiv.org/pdf/2609.08449v1)

**作者:** Elias Brugger `[一作]` (TU Wien), Paul Guerrero `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种生成性方法来完成部分3D物体，该物体由一组高斯斑点表示。该方法生成一个合理的完整3D物体，完全保留给定的部分输入，仅使用2D扩散先验。

**💡 创新点**

创新点在于结合了基于得分蒸馏采样的3D生成方法和一种新颖的保留损失，鼓励在可见区域保留原始斑点，从而有效完成高斯斑点物体。

**🔧 技术方法**

使用了得分蒸馏采样（Score Distillation Sampling, SDS）技术，并引入了一种新的输入保留损失。

**📊 数据集**

使用了一个新的部分高斯斑点对象数据集，该数据集包含39个来自不同来源的对象。

**📈 对比分析**

与现有方法相比，GSComplete在保留输入方面显著更准确，同时完成结果的合理性相当。通过比较颜色和深度的均方误差以及CLIP相似度来评估性能。

**⚠️ 局限性**

限制在于输入信息的丰富程度，如果输入过于稀疏，可能会导致生成的结果缺乏有意义的信息。此外，当前实现的运行时间与其他方法相当，但尚未进行优化。

---

## 504. FlexSpIM: An Event-Based Digital Compute-In-Memory Accelerator with Flexible Operand Resolution and Layer-Wise Hybrid Stationarity

**arXiv ID:** 2609.08446 | [PDF](https://arxiv.org/pdf/2609.08446v1)

**作者:** Nicolas Chauvaux `[一作]` (Delft University of Technology), Charlotte Frenkel `[通讯]` (Delft University of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

提出了一种名为FlexSpIM的数字计算内存加速器，支持灵活的操作数分辨率和层级混合静态数据流，以提高脉冲神经网络（SNN）的推理效率。

**💡 创新点**

FlexSpIM的创新点在于其支持任意的权重和膜电位分辨率，允许在每层独立选择权重静态或输出静态，从而减少操作数移动，提高能效和降低延迟。

**🔧 技术方法**

使用了数字计算内存（CIM）技术，结合灵活的操作数重塑和混合静态数据流。

**📊 数据集**

在IBM DVS手势数据集上进行了评估，FlexSpIM在该数据集上达到了95.8%的准确率。

**📈 对比分析**

与传统的固定静态方法相比，FlexSpIM在大规模系统中实现了高达45%的能量和52%的延迟减少，且在性能上优于之前的固定精度数字CIM SNN加速器。

**⚠️ 局限性**

限制在于尽管FlexSpIM提供了更高的灵活性，但其能效在宏观层面上可能低于某些专用的CIM架构，且在系统级别的优化仍需进一步探索。

---

## 505. Which Forms of Caregiver Feedback Support Grammar Learning? A Reinforcement-Learning Study of Child-Like Language Models

**arXiv ID:** 2609.08576 | [PDF](https://arxiv.org/pdf/2609.08576v1)

**作者:** Jing Liu `[一作]` (Université PSL), Abdellah Fourtassi `[通讯]` (Aix Marseille Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究使用儿童语言模型作为受控学习者，测试不同形式的照顾者反馈对语法发展的支持作用。

**💡 创新点**

创新点在于通过强化学习和奖励模型，系统地比较了四种反馈类型对语法学习的影响，特别是结构对齐反馈显示出显著的改善效果。

**🔧 技术方法**

使用了小型的GPT-2风格模型，结合强化学习和奖励模型进行微调。

**📊 数据集**

使用了CHILDES数据集，该数据集包含儿童与照顾者的自然对话数据。

**📈 对比分析**

与基线模型相比，使用奖励模型的微调在生成的语法性上显示出可测量的效果，尤其是结构对齐反馈在多个指标上表现出最强的改善，而其他反馈类型的效果较弱或负面。

**⚠️ 局限性**

本研究的局限性在于仅捕捉了语言反馈的口头成分，未考虑多模态上下文中的反馈，同时只使用了单一的GPT-2架构，未能验证不同模型架构的普适性。

---

## 506. Solution to Bucher's density problem for context-free languages

**arXiv ID:** 2609.08571 | [PDF](https://arxiv.org/pdf/2609.08571v1)

**作者:** Rastko Maslic `[一作]`, Jeffrey Shallit `[通讯]` (University of Waterloo)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

论文解决了Bucher在1980年提出的问题，证明了在给定的上下文无关语言L和U之间，存在一个上下文无关语言K，使得K的补集和U的补集都是有限的。

**💡 创新点**

创新点在于构造了一个具有上下文无关补集的无限语言D，并证明了在任何正则语言R的情况下，D与R的交集或补集必然是有限的，从而否定了Bucher的问题。

**🔧 技术方法**

使用了上下文无关文法、有限自动机和一计数器自动机等技术。

**📊 数据集**

构造了一个无限语言D，且L和U是从D的补集中构造的，所有字母表都是有限的。

**📈 对比分析**

通过构造语言D并使用文法论证，证明了任何上下文无关的中间语言K都将以与某个正则语言相同的方式划分D，从而得出所需的不可能性。性能方面，L和U可以在二进制字母表上进行构造。

**⚠️ 局限性**

限制在于构造的语言D的性质可能不适用于所有上下文无关语言的情况，且在某些情况下可能无法推广到更广泛的语言类别。

---

## 507. BIO-MEMART: Biometric-Aware KV Cache Memory for Multi-User LLM Agents

**arXiv ID:** 2609.08566 | [PDF](https://arxiv.org/pdf/2609.08566v1)

**作者:** Yanhong Qian `[一作]` (Anhui University), Zhe Jin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种生物识别感知的KV缓存内存框架Bio-MemArt，用于多用户大语言模型（LLM）代理，解决了共享内存中的访问控制问题。

**💡 创新点**

创新点在于将生物识别模板附加到每个存储的KV内存块，并在检索时通过生物识别探测过滤共享内存池，从而实现物理用户的访问控制。

**🔧 技术方法**

使用了生物识别技术（面部和掌纹识别）来验证当前用户的身份，并结合了原有的KV缓存检索和重用管道。

**📊 数据集**

使用了多个面部和掌纹识别基准数据集进行评估，包括AgeDB-30、CALFW、CFP-FF、IITD等。

**📈 对比分析**

与全上下文推理和原生KV内存进行了比较，结果显示Bio-MemArt在保持低令牌开销的同时，授权用户的问答性能显著优于未授权用户。

**⚠️ 局限性**

限制在于生物识别验证的准确性可能因基准数据集的不同而有所差异，可能影响不同用户的访问控制效果。

---

## 508. Not All Variables Agree: Reliability-Aware Variable-Wise Gradient Surgery for Multivariate Time-Series Forecasting

**arXiv ID:** 2609.08554 | [PDF](https://arxiv.org/pdf/2609.08554v1)

**作者:** Jinwoo Park `[一作]` (Seoul National University), Pilsung Kang `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的优化策略PV-Surgery，用于多变量时间序列预测，旨在解决标准均值损失训练中变量间梯度冲突的问题。

**💡 创新点**

创新点在于通过构建变量级梯度代理，选择性地进行梯度手术，以提高多变量预测的性能，同时保留有用的共享信息。

**🔧 技术方法**

使用了一种基于优化器的训练策略，结合了条件池化和方向校正的方法。

**📊 数据集**

在五个基础模型和七个数据集上进行了实验，包括ETTh1、ETTh2、ETTm1、ETTm2、Weather和Exchange。

**📈 对比分析**

与标准均值损失训练相比，PV-Surgery在MSE上平均降低了3.61%，在MAE上降低了2.93%。在140个设置中，PV-Surgery在111个设置中获得了更低的MSE，在119个设置中获得了更低的MAE。

**⚠️ 局限性**

限制在于该方法提高了每个训练周期的中位成本至均值损失基线的1.54倍，且在代理精度低、选择层混合变量或变量关系在不同数据集间差异较大时，性能提升可能减弱。

---

## 509. Visualizing Colonial Regimes: A Multi-View Approach to (Historical) Political Transformation

**arXiv ID:** 2609.08518 | [PDF](https://arxiv.org/pdf/2609.08518v1)

**作者:** Nicole Husemann `[一作]` (Hannah Arendt Institute for Totalitarianism Studies), Christofer Meinecke `[通讯]` (Image and Signal Processing Group, Leipzig University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文采用批判性的方法对殖民政权的可视化进行了探讨，创建了互动可视化，揭示了殖民类别的构建性质，并暴露了帝国与领土之间的等级关系。

**💡 创新点**

创新点在于通过多视角的互动可视化，挑战传统的静态和孤立的殖民数据表示，使得殖民分类的解释框架可见，并支持跨学科分析。

**🔧 技术方法**

使用了批判性解释学和后殖民理论，结合了政治政权多样性数据集，采用了时间流动可视化和地理分布映射技术。

**📊 数据集**

使用了政治政权多样性数据集（Varieties of Political Regimes），该数据集涵盖了1900年至2025年间的29,000多个国家-年份观察。

**📈 对比分析**

与传统的可视化方法相比，本文的方法通过动态的时间和空间视图揭示了殖民政权的持续性和变化，性能上能够更好地展示历史政治转型的复杂性。

**⚠️ 局限性**

限制在于数据的基础问题，例如未能捕捉到国家内部的殖民关系，可能忽视了不同地区在同一殖民地内的冲突和控制程度。

---

## 510. Multi-bounce Drum Roll with Optimized Active Tricks to Leverage Soft Embodiment

**arXiv ID:** 2609.08490 | [PDF](https://arxiv.org/pdf/2609.08490v1)

**作者:** Naoto Yamanaka `[一作]` (National Institute of Informatics), Taisuke Kobayashi `[通讯]` (National Institute of Informatics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种软体机器人鼓手，用于准确和高效地演奏鼓点，特别是高频鼓点的多次反弹技术。

**💡 创新点**

创新点在于设计了两种技巧：Tap-Pull（TP）技巧和Micro-Pulse（MP）技巧，以优化鼓点的反弹次数和音量保持。

**🔧 技术方法**

使用了贝叶斯优化技术，以数据驱动的方式高效调节这两种技巧的参数。

**📊 数据集**

实验中使用了自制的双臂机器人，配备软性和刚性末端执行器，进行鼓点演奏的评估。

**📈 对比分析**

通过与刚性末端执行器的比较，软性TP技巧实现了每次击打12.25次的最高反弹次数，且音量衰减比刚性MP高出6.8倍，显示出显著的声学效率。

**⚠️ 局限性**

限制在于实验是在固定的鼓配置和有限的节奏范围内进行的，且未明确评估鼓点质量的感知方面，未来需要进一步研究优化目标值与人类感知评估之间的关系。

---

## 511. Strategyproof Mechanisms for Connecting Impassable Regions

**arXiv ID:** 2609.08488 | [PDF](https://arxiv.org/pdf/2609.08488v1)

**作者:** Hau Chan `[一作]` (University of Nebraska-Lincoln), Chenhao Wang `[通讯]` (Beijing Normal University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在障碍物分隔的两个区域之间建立路径的策略无关机制，旨在最小化最大成本或社会成本。

**💡 创新点**

提出了紧密的确定性最大成本近似比和社会成本上界，并研究了期望下的随机机制，提供了更强的下界。

**🔧 技术方法**

使用了确定性机制和随机机制，特别是功率比例机制，确保在期望下的策略无关性。

**📊 数据集**

使用了包含多个代理的线段模型，代理的位置是私有的，且通过障碍物分隔。

**📈 对比分析**

与现有方法进行了比较，确定性机制的最大成本近似比为2/(1+k)，社会成本的上界为n/(1+k(n-1))，随机机制的社会成本保证为最多5，且在k=0时为3。

**⚠️ 局限性**

限制在于对于随机最大成本的上界和下界的改进仍然是开放的，且在实线模型中，任意确定性策略无关机制的社会成本下界仍未解决。

---

## 512. Enhancing Communication in Speech Therapy: Exploring the Cognitive Synergy Between Gesture and Speech

**arXiv ID:** 2609.08486 | [PDF](https://arxiv.org/pdf/2609.08486v1)

**作者:** Paul-Peter Arslan `[一作]` (De Vinci Research Center), Xiao Xiao `[通讯]` (De Vinci Research Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本论文研究了一种基于节奏的界面，该界面最初设计用于手动灵巧性康复，现在被适应用于语言治疗。通过手指敲击控制合成的语音短语，利用手势与语言之间的认知联系，发现该界面能够提高患者的动机和治疗效果。

**💡 创新点**

创新点在于将原本用于运动康复的界面成功应用于语言治疗，揭示了手势与语言之间的认知联系，增强了儿童在语言治疗中的动机和参与度。

**🔧 技术方法**

使用了Dextrain Manipulandum设备，该设备能够准确测量每个手指施加的力量，并结合Unity程序和FMOD音频接口生成合成语音片段。

**📊 数据集**

研究中使用的数据集包括来自四位治疗师的访谈数据，以及与一位语言治疗师和八名有语言障碍的儿童（包括自闭症、唐氏综合症、言语失用症和阅读障碍）进行的初步测试数据。

**📈 对比分析**

通过与治疗师的访谈和儿童的测试会话进行比较，发现该界面有效地将运动技能与语言联系起来，改善了治疗结果和患者的动机。治疗师指出，结合手势和节奏的使用帮助儿童更好地记忆和发音。

**⚠️ 局限性**

限制在于设置界面需要时间，尽管治疗师可以根据儿童的反应添加新词，但在短时间的治疗会话中，准备时间可能会影响治疗的效率。

---

## 513. Do Reviewers Still Reward Lexical Complexity? A Frozen-Rater Study of Preference Drift in 124K ICLR Reviews

**arXiv ID:** 2609.08475 | [PDF](https://arxiv.org/pdf/2609.08475v1)

**作者:** Jiabin Zheng `[一作]` `[通讯]` (Peking University), Jiabin Zheng (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究分析了2018年至2025年间ICLR提交的论文中，人工评审者对语言复杂性的评价变化，特别是在大型语言模型（LLM）普及后，评审者是否仍然重视复杂的语言表达。

**💡 创新点**

创新点在于使用一个固定的机器评审者（frozen rater）来区分评审者偏好变化与提交文本组成变化的影响，从而提供了一种新的识别策略。

**🔧 技术方法**

采用了机器学习模型生成的评审数据，使用了Gen-Review数据集，该数据集包含81,850条机器生成的评审。

**📊 数据集**

使用的数据集包括32,638个提交的论文和124,615条人工评审，机器评审来自Gen-Review数据集。

**📈 对比分析**

通过三重差异法（difference-in-differences）比较人类评审者与机器评审者的评分变化，发现人类对非领域语言复杂性的重视程度显著下降，而机器评审者的评分保持不变。

**⚠️ 局限性**

限制在于只分析了ICLR会议的论文，且使用的文本特征主要是语言复杂性，未能涵盖更广泛的评审标准变化。此外，机器评审者的评估可能不适用于所有类型的文本。

---

## 514. Detecting Authorship in Political Texts with Inductive Stylometry

**arXiv ID:** 2609.08459 | [PDF](https://arxiv.org/pdf/2609.08459v1)

**作者:** Gennadii Iakovlev `[一作]`, Levente Littvay `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发并测试了一种归纳风格计量方法，以恢复政治传播中的潜在作者结构，结合字符3-gram特征、UMAP降维和Burrows' Delta。

**💡 创新点**

创新点在于将传统的风格计量方法应用于政治文本，尤其是短格式文本和不同语言（英语和匈牙利语）的分析，揭示了隐藏作者的风格特征。

**🔧 技术方法**

使用了字符3-gram特征、UMAP降维技术和Burrows' Delta进行频率基础的风格计量分析。

**📊 数据集**

使用了六个不同的语料库，包括美国国会研究服务（CRS）报告、匈牙利监察员报告、特朗普的推文和演讲等。

**📈 对比分析**

通过与已知作者标签的比较，验证了方法的有效性。结果显示，在大多数案例中，风格计量能够有效区分不同作者的风格，尤其是在正式法律文本和社交媒体推文中。

**⚠️ 局限性**

限制在于在剧本化的演讲中无法识别个别的演讲作者，且在高度编辑的文本和口头交付的情况下，个体风格变得更加模糊。

---

## 515. Topological Fraud Detection in Latent Transaction Spaces

**arXiv ID:** 2609.08445 | [PDF](https://arxiv.org/pdf/2609.08445v1)

**作者:** Avraham Bourla `[一作]` `[通讯]`, Avraham Bourla

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于拓扑匿名嵌入的欺诈检测方法，通过无监督过滤和监督分类的迭代过程，实现了低延迟的隐私保护欺诈检测。

**💡 创新点**

创新点在于结合了拓扑方法与树模型和深度学习，能够在不泄露个人信息的情况下，快速识别欺诈行为。

**🔧 技术方法**

使用了UMAP（统一流形近似与投影）、自编码器（AE）、自组织映射（SOM）和CatBoost、LightGBM等机器学习技术。

**📊 数据集**

使用了Kaggle的信用卡匿名欺诈数据集和IEEE-CIS数据集，这些数据集包含真实的欧洲信用卡交易数据。

**📈 对比分析**

与传统方法相比，本文的方法在10%的测试集上实现了91.74%的精确率和79.88%的召回率，F1分数为0.8540，且在消费者级硬件上单行预测延迟为4.74毫秒，预计在生产级硬件上可达到1毫秒的延迟标准。

**⚠️ 局限性**

限制在于需要进一步评估该分类器在不同供应商数据集和时间尺度变化下的有效性，以及是否可以扩展到其他高风险领域的隐私敏感异常检测。

---

## 516. CleanCity-BinSense: An IoT-Enabled Smart Waste Management System with Configurable Real-Time Fill Monitoring and Nearest-Neighbor Route Optimization

**arXiv ID:** 2609.08527 | [PDF](https://arxiv.org/pdf/2609.08527v1)

**作者:** Mohammad Adnan Kabir `[一作]` (Islamic University of Technology), Intifad Muhammad Sayeed `[通讯]` (Jahangirnagar University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了CleanCity-BinSense，一个低成本的端到端物联网智能废物管理系统，旨在支持可扩展的实时废物监测和需求驱动的收集。

**💡 创新点**

系统的创新点在于可配置的传感模型，允许在不同尺寸和几何形状的垃圾箱上部署，而无需修改固件。

**🔧 技术方法**

使用了太阳能供电的传感器节点，配备超声波传感器进行实时垃圾箱填充水平监测，并通过Wi-Fi将传感器读数传输到集中式网络平台。

**📊 数据集**

实验中使用了不同几何形状的垃圾箱，具体数据集未详细说明，但提到在达卡等资源受限环境中进行测试。

**📈 对比分析**

与现有系统相比，CleanCity-BinSense在传感器的平均绝对误差（MAE）为0.38 cm，端到端延迟为5.3秒，路由生成时间少于100毫秒，显示出其在城市废物管理中的实用性。

**⚠️ 局限性**

系统的局限性包括最近邻启发式算法不保证全局最优性，且在没有现有网络基础设施的情况下需要4G或NB-IoT连接。

---

## 517. Quantum Matrix-Product Codes: CSS-T Characterization and Maximality

**arXiv ID:** 2609.08520 | [PDF](https://arxiv.org/pdf/2609.08520v1)

**作者:** Delio Jaramillo-Velez `[一作]` (Universidad de La Laguna), Flavio Salizzoni `[通讯]` (Max Planck Institute for Mathematics in the Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文分析了基于(u | u+v)构造的矩阵乘积码的CSS-T条件，提供了必要和充分条件以确定一对矩阵乘积码(C_1,C_2)是否为CSS-T对，并对这些对在CSS-T偏序中的极大性进行了表征。

**💡 创新点**

创新点在于将CSS-T条件与矩阵乘积码的代数结构联系起来，提供了具体的循环集合标准，以便构造和分类支持横向T门的新量子码族。

**🔧 技术方法**

使用了代数方法，特别是矩阵乘积码和Schur积的理论，结合了循环码的性质。

**📊 数据集**

研究中涉及的主要数据集是基于循环集合的矩阵乘积码，特别是定义在有限域𝔽_2上的循环码。

**📈 对比分析**

通过与现有的CSS-T对进行比较，本文提供了明确的条件来判断一对矩阵乘积码是否为CSS-T对，并且在极大性方面也给出了相应的标准。

**⚠️ 局限性**

限制在于目前的分析主要集中在特定的矩阵乘积构造上，未来的工作将扩展到加权Reed-Muller码作为构造中的成分码，以及对循环和加权Reed-Muller设置所获得参数的系统研究。

---

## 518. Concept-Level Risk and Calibration for Governance in Diffusion Foundation Models

**arXiv ID:** 2609.08517 | [PDF](https://arxiv.org/pdf/2609.08517v1)

**作者:** Kun Xu `[一作]` (Nanjing University of Aeronautics and Astronautics), Yuming Fang `[通讯]` (Jiangxi University of Finance and Economics)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种概念级概率审计和报告框架，用于扩散模型的治理，旨在系统地评估和比较模型在语义控制下的风险。

**💡 创新点**

创新点在于将治理相关的概念行为形式化为由随机生成引起的伯努利语义事件，并定义了概念风险算子，以便在不同模型、通道和条件下进行比较。

**🔧 技术方法**

使用了概念风险算子、后验校准和配置级风险聚合等技术，结合了多协议开发和持出设计。

**📊 数据集**

实验使用了SD1.5、SD2.1和SDXL模型，涉及64个概念，包括身份相关、版权敏感和不安全/NSFW敏感概念。

**📈 对比分析**

通过对比不同模型和通道的风险，发现嵌入式访问和模糊提示的风险常常被标准提示评估所低估。实验表明，概念风险在不同模型和条件下表现出一致但不均匀的模式。

**⚠️ 局限性**

限制在于当前的评估方法主要依赖于启发式审计和特定任务的基准，缺乏对模型、通道和部署条件的系统比较。

---

## 519. Beyond Agent Harnesses: Cross-Substrate Authority for Multi-Agent Systems

**arXiv ID:** 2609.08472 | [PDF](https://arxiv.org/pdf/2609.08472v1)

**作者:** Yang Li `[一作]` (University of Hong Kong), Ye Lu `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了跨子系统的授权问题，探讨了在共享工作区中如何安全地发布当前工件，并通过三个实验验证了不同的证据和执行时间验证机制的有效性。

**💡 创新点**

提出了跨子系统授权的概念，强调了隐藏状态对决策的重要性，并展示了在不同世界中相同输入可能导致不同的安全行动。

**🔧 技术方法**

使用了控制实验设计，结合真实的Git历史记录、持久记录的代理执行尝试和确定性预言机。

**📊 数据集**

使用了真实的Git仓库和持久记录的DeepSeek Harness源/目标尝试，构建了两个基准系列的实验。

**📈 对比分析**

通过三个实验比较了不同的证据处理方法，结果显示原始收据和类型关系都能有效解决授权问题，但在计划时的可靠性存在差异，执行时间的验证机制能够阻止不安全的意图。

**⚠️ 局限性**

实验的局限性包括模型依赖性和计划的可靠性问题，未来的工作需要解决多租户政策组合和身份绑定Git认证等问题。

---

## 520. Sample-Guided Exact Top-K Selection for Long-Context Sparse Attention

**arXiv ID:** 2609.08450 | [PDF](https://arxiv.org/pdf/2609.08450v1)

**作者:** Siran Liu `[一作]` (Tencent Inc), Jianchen Zhu `[通讯]` (Tencent Inc)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的Top-K选择算法HPC-Ops Top-K，该算法通过样本引导的精确选择来优化稀疏注意力模型中的选择过程。

**💡 创新点**

创新点在于将粗略边界定位与精确选择分开，利用固定步幅的视图来快速找到候选集，并通过完整行遍历来验证和细化这些候选集。

**🔧 技术方法**

使用了GPU实现的样本引导精确选择算法，结合了持久性、KV分割和直接精确执行的内核。

**📊 数据集**

使用了Hy4-Preview生成的索引器分数数据集进行评估。

**📈 对比分析**

与vLLM、TensorRT-LLM、SGLang、FlashInfer和PyTorch等外部实现进行了比较，HPC-Ops Top-K在20种操作配置中均表现最佳，速度提升为1.29-1.75倍，几何平均速度提升为1.55倍。

**⚠️ 局限性**

限制在于该方法依赖于样本引导的边界定位，可能在某些情况下导致边界估计不准确，从而影响最终的选择结果。

---

## 521. AlphaRJM: Reward-Jump Memory for Stochastic Return-Guided Alpha Discovery

**arXiv ID:** 2609.08581 | [PDF](https://arxiv.org/pdf/2609.08581v1)

**作者:** Sayan Dhan `[一作]` (Indian Institute of Technology Guwahati), Selvaraju Natarajan `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的公式化阿尔法发现框架AlphaRJM，结合了持久的终端评估历史和随机回报引导的符号搜索。

**💡 创新点**

创新点在于引入了Reward-Jump Memory和基于动作条件的随机微分方程(SDE)回报评论家，解决了延迟反馈和不确定性的问题。

**🔧 技术方法**

使用了Reward-Jump Memory和动作条件的随机微分方程(SDE)回报评论家，结合了分布式贝尔曼学习。

**📊 数据集**

在三个代表性的中国股票市场数据集上进行评估：CSI300、CSI500和CSI800。

**📈 对比分析**

与传统预测模型（如多层感知器和LightGBM）、连续时间神经模型（如Neural ODE和Neural SDE）以及基于强化学习的公式化阿尔法发现方法（如AlphaGen、AlphaQCM和AlphaSAGE）进行比较，AlphaRJM在多个指标上表现优异，尤其在IC和ICIR上领先。

**⚠️ 局限性**

限制在于模型的复杂性和对数据集的依赖性，可能在不同市场条件下的表现有所不同。

---

## 522. The Unreliable Progress Bar: Can LLM Agents Reliably Report Task Progress Throughout Execution?

**arXiv ID:** 2609.08589 | [PDF](https://arxiv.org/pdf/2609.08589v1)

**作者:** Boyang Wang `[一作]` (Independent Researcher), Yalun Wu `[通讯]` (NExT++ Lab, School of Computing, National University of Singapore)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究评估了大型语言模型在任务执行过程中报告任务进展的能力，特别是在不同阶段的可靠性。

**💡 创新点**

创新点在于系统性地分析了模型在任务生命周期各阶段的报告可靠性，并提出了一种新的评估协议。

**🔧 技术方法**

使用了τ^2-bench公共基准和StageIF控制测试平台来评估模型的报告能力。

**📊 数据集**

使用了τ^2-bench和StageIF这两个数据集，前者是公共基准，后者是控制测试平台。

**📈 对比分析**

通过对比不同模型在任务执行中的报告表现，发现大多数模型在任务进行中报告准确性下降，而在任务完成后恢复准确性。新一代模型在任务中期的下降现象有所改善，但在完成阶段变得更加保守。

**⚠️ 局限性**

限制在于每个部署标识符仅代表一个模型和服务的组合，无法识别内部机制，且终端检查点的选择可能影响结果。

---

## 523. MorphoOrgaAgent: A Foundation-Model-Based Multi-Agent System for Autonomous Organoid Analysis

**arXiv ID:** 2609.08696 | [PDF](https://arxiv.org/pdf/2609.08696v1)

**作者:** Hanyi Zhang `[一作]` (Helmholtz AI), Tingying Peng `[通讯]` (Helmholtz AI)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MorphoOrgaAgent，一个基于基础模型的多代理系统，用于自主分析类器官的形态特征，能够实现零-shot类器官分割、自动数据分析和报告生成。

**💡 创新点**

创新点在于结合了几何提示和文本提示的混合分割模块，能够在没有手动标注的情况下进行类器官实例分割，并且通过自然语言输入生成结构化报告。

**🔧 技术方法**

使用了基于Cellpose的几何提示和SAM3的文本提示的混合分割技术，以及GPT-5.4-mini和GPT-5.4作为自然语言处理的支持。

**📊 数据集**

使用了来自OrganoID、OrgaExtractor和OrgaSegment等三个公共数据集的图像，构建了MorphoOrgaVQA基准进行评估。

**📈 对比分析**

通过与现有的生物图像分析框架（如Omega和Agentic-J）进行定量和定性比较，MorphoOrgaAgent在零-shot实例分割和统计计算方面表现优异，生成的分析报告准确且结构清晰。

**⚠️ 局限性**

限制在于对复杂形态特征（如粗糙度）的处理仍然存在挑战，且系统的性能可能受到输入图像质量和生物学特征多样性的影响。

---

## 524. HDA-MoE: Hybrid Parallelism and Dynamic, Adaptive Scheduling for Mixture-of-Experts with 3D Near-Memory Processing

**arXiv ID:** 2609.08682 | [PDF](https://arxiv.org/pdf/2609.08682v1)

**作者:** Haochen Huang `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对3D近内存处理架构的混合专家模型（MoE）推理框架，结合了混合专家部署和动态自适应调度，以优化MoE的执行。

**💡 创新点**

创新点在于通过离线混合并行映射算法与在线动态调度机制的结合，减少通信开销并提高计算利用率，同时引入硬件感知的门控机制以应对动态激活模式。

**🔧 技术方法**

使用了混合并行部署、动态调度和硬件感知门控等技术，构建了一个统一的性能模型来估计MoE的计算和通信成本。

**📊 数据集**

使用了多个大型语言模型（LLMs）作为数据集，包括Mixtral、DeepSeek、Qwen2和Qwen3.5，评估了其在不同硬件配置下的性能。

**📈 对比分析**

与现有的并行化策略（如张量并行TP和专家并行EP）相比，HDA-MoE在速度上实现了1.1×–3.4×的加速，且在多个基线方法上均表现出显著的性能提升。

**⚠️ 局限性**

限制在于动态专家激活的复杂性和内存带宽的限制，尽管引入了硬件感知的门控机制，但在某些情况下仍可能面临计算和通信的瓶颈。

---

## 525. Beyond Fixed Fault Models: Comparing LLM-Based and Rule-Based Fault Injection in OpenStack

**arXiv ID:** 2609.08681 | [PDF](https://arxiv.org/pdf/2609.08681v1)

**作者:** Giuseppe De Rosa `[一作]` (University of Naples Federico II), Domenico Cotroneo `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究比较了基于大语言模型（LLM）和基于规则的故障注入方法在OpenStack中的表现，探讨了它们在生成和激活软件故障方面的差异。

**💡 创新点**

创新点在于将LLM生成的故障与传统的规则基础故障注入方法进行比较，发现LLM生成的故障能够扩展固定故障模型的行为覆盖范围，但并不一定优于规则基础方法。

**🔧 技术方法**

使用了Qwen2.5-Coder和DeepSeek-Coder这两种代码LLM，以及ProFIPy作为规则基础的故障注入工具。

**📊 数据集**

使用了PyResBugs数据集，该数据集包含5007个经过人工验证的残余错误，来自多个开源Python项目。

**📈 对比分析**

比较方法包括故障激活率和可观察失败率，结果显示在共享目标上，LLM生成的故障导致更多的灾难性结果，而ProFIPy则产生更多的静默和多组件效应。

**⚠️ 局限性**

限制在于LLM生成的故障可能需要显式验证，且在实际应用中仍需控制生成、运行时验证、系统级预言和可重复的实验来源。

---

## 526. Learning to build covering structures with continuous adjustments

**arXiv ID:** 2609.08669 | [PDF](https://arxiv.org/pdf/2609.08669v1)

**作者:** Gabriel Vallat `[一作]`, Stefana Parascho `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于强化学习的方法，完全放弃预定义的施工计划，能够在结构构建过程中自适应生成施工序列。

**💡 创新点**

创新点在于利用图神经网络和单向边来强制不同离散动作的Q值之间的独立性，从而实现稳定的学习。

**🔧 技术方法**

使用了混合软演员-评论家（HSAC）算法，该算法扩展了软演员-评论家（SAC）以适应混合动作空间，并结合了图神经网络。

**📊 数据集**

在物理双机器人设置上进行了验证，成功构建了一个使用3D打印块的拱形结构。

**📈 对比分析**

与之前的方法（混合PPO）相比，HSAC在收敛后表现出显著更高的性能，尽管每一步的优化时间较长，但在样本效率上表现良好。

**⚠️ 局限性**

限制在于HSAC的每一步优化时间较长，且在复杂环境中可能需要更多的计算资源。

---

## 527. CoordFormer: Give Me Any Coordinates and I Will Give You Labels

**arXiv ID:** 2609.08660 | [PDF](https://arxiv.org/pdf/2609.08660v1)

**作者:** Iacopo Curti `[一作]` (University of Bologna), Luigi Di Stefano `[通讯]` (University of Bologna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的坐标基础架构，用于非常高分辨率图像的语义分割，能够在任意空间位置预测标签，从而实现高精度的分割。

**💡 创新点**

创新点在于引入了坐标解码器和局部交叉注意力机制，能够有效整合局部和全局信息，同时保持像素级的精度。

**🔧 技术方法**

使用了坐标解码器和局部交叉注意力机制，结合了高分辨率局部特征和从下采样图像中提取的全局标记。

**📊 数据集**

使用了MaSS13K和DIS5K数据集进行评估，MaSS13K是专门为非常高分辨率语义分割设计的基准数据集。

**📈 对比分析**

与现有方法相比，取得了在MaSS13K上的最先进性能，并在DIS5K上超越了同等规模和更高参数的方法，显示出其在高质量分割中的有效性。

**⚠️ 局限性**

在硬件资源不足时，可能会导致高推理时间，因为需要顺序处理坐标，尽管在内存受限的情况下仍能实现高质量的分割。

---

## 528. TriCCOT: Tri-part Convolutional Conformal Transformer for Onboard Space Object Detection

**arXiv ID:** 2609.08659 | [PDF](https://arxiv.org/pdf/2609.08659v1)

**作者:** Adrien Dorise `[一作]` (Centre National d'Etudes Spatiales), Stéphane May `[通讯]` (Centre National d'Etudes Spatiales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为TriCCOT的三部分架构，用于在地球观测中进行稳健且可部署的物体检测，结合了卷积区域提议网络、符合预测阶段和硬件友好的注意力分类器Aper-GATES。

**💡 创新点**

TriCCOT的创新点在于将卷积检测、符合预测和自注意力机制结合在一起，能够直接处理原始图像，同时保持计算可行性，适用于嵌入式FPGA部署。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的自注意力机制Aper-GATES，后者通过卷积投影和硬件友好的门控操作来实现。

**📊 数据集**

使用了DIOR数据集和VDVRaw数据集，DIOR数据集包含23,463张光学遥感图像，VDVRaw数据集包含282张多光谱图像。

**📈 对比分析**

与FPGA兼容的架构（如YOLOX-S和NanoDet-Plus）相比，TriCCOT在原始DIOR数据集上表现接近最佳检测性能，并在退化图像上表现最佳，显示出其在处理空间模糊和信号依赖噪声方面的优势。

**⚠️ 局限性**

TriCCOT的局限性在于其采用了顺序训练策略，可能导致RPN、符合预测器和分类器之间的错误传播，未来需要探索联合或协作训练策略以优化整个检测管道。

---

## 529. Charts Are Beyond Pixels: Probing for Layer-Wise Chart Understanding and Editing

**arXiv ID:** 2609.08657 | [PDF](https://arxiv.org/pdf/2609.08657v1)

**作者:** Xiaochuan Zhong `[一作]` (Shanghai Jiao Tong University), Shaobo Cui `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LayerWiseBench基准，评估模型在图表理解和编辑中的层级行为，特别关注层归属、层绑定和可见性排序。

**💡 创新点**

创新点在于通过层级表示来组织评估，提供了一个结构化的基准，能够更好地评估模型在图表理解和编辑中的表现。

**🔧 技术方法**

使用了可执行的图表程序生成图表，并通过空间对齐的每层RGBA资产和构建派生标签进行评估。

**📊 数据集**

数据集包含2800个源图表，涵盖14种图表范式，生成7329个层级理解问题和53791个指导编辑变体。

**📈 对比分析**

与九个视觉语言模型和四个图像编辑模型进行比较，Qwen3.5-27B在层归属和层绑定上表现最佳，但在可见性排序上表现较差，整体mIoU在1.49%到4.93%之间。

**⚠️ 局限性**

局限性在于仅限于程序生成的图表，未来工作可以扩展到自然发生的图表和更广泛的可视化范式。

---

## 530. Combating Instruction Conflict via Energy-Driven Latent Conflict Detection

**arXiv ID:** 2609.08646 | [PDF](https://arxiv.org/pdf/2609.08646v1)

**作者:** Mingyu Ma `[一作]` (Wuhan University), Xiaochuan Shi `[通讯]` (Wuhan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种响应级别的潜在冲突检测器，用于在生成后、交付前验证大型语言模型（LLMs）的输出，以识别响应漂移现象。

**💡 创新点**

首次定义了响应漂移现象，并将其建模为能量空间中的分布差异，克服了静态输入检测的局限性，能够识别传统模式匹配过滤器常常遗漏的动态潜在偏差。

**🔧 技术方法**

采用了基于能量的建模（EBM）技术，通过优化成对边际排名目标来分离合规和漂移响应。

**📊 数据集**

使用了五种主流的开源LLM（参数范围从1.5B到14B），并构建了包含系统指令、用户指令和助手响应的三元组数据集。

**📈 对比分析**

与五个基线方法进行比较，结果显示该方法在PR-AUC上提高了约30个百分点，并在Mistral-7B上将95%真阳性率下的假阳性率降低到2.67%。

**⚠️ 局限性**

该方法需要访问目标语言模型的隐藏状态，这限制了其在仅提供API的闭源系统中的直接应用。此外，检测器的性能可能需要在不同模型、解码设置或任务分布下进行重新校准。

---

## 531. MFVINS: Multiple Fisheye Camera-Based Visual Inertial System

**arXiv ID:** 2609.08626 | [PDF](https://arxiv.org/pdf/2609.08626v1)

**作者:** Eunseong Jang `[一作]` (Jeonbuk National University), HyungGi Jo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于多鱼眼相机的视觉惯性系统MFVINS，旨在解决单目相机在复杂环境下的局限性。

**💡 创新点**

创新点在于结合多鱼眼相机和低成本IMU，采用IMU辅助的FAST特征跟踪和基于学习的深度估计，显著提高了在遮挡和低纹理环境下的鲁棒性和准确性。

**🔧 技术方法**

使用了IMU辅助的FAST特征跟踪算法、深度学习模型进行深度估计，以及基于图优化的回环检测和姿态图优化。

**📊 数据集**

在多个真实场景中进行评估，包括地下停车场等复杂环境，使用了LiDAR作为地面真实值进行比较。

**📈 对比分析**

与现有的单目和多相机VINS方法进行比较，MFVINS在准确性和鲁棒性方面均表现优越，尤其在遮挡和低纹理环境中，RMSE显著低于其他方法。

**⚠️ 局限性**

限制在于系统的计算复杂性，尽管实现了实时处理，但在更大规模的场景中可能需要进一步优化传感器配置和处理效率。

---

## 532. Target-Independent Micro-Interventions for Predicting Training Response Across Language-Model Families

**arXiv ID:** 2609.08618 | [PDF](https://arxiv.org/pdf/2609.08618v1)

**作者:** Zhongxuan Liu `[一作]` (Harbin Institute of Technology), Hongzhi Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何通过微干预来测量模型在训练过程中的响应状态，并提出了一种新的学习状态表示方法，结合了当前能力和微干预的响应。

**💡 创新点**

创新点在于引入了目标独立的微干预，形成了可转移的训练状态，并提出了直接读出和结构化操作读出两种互补的预测方法。

**🔧 技术方法**

使用了目标独立的微干预技术，结合了直接读出和操作读出的方法来进行模型性能预测。

**📊 数据集**

使用了多个模型家族的数据集，包括Qwen2.5-7B-Instruct、Mistral-7B-Instruct-v0.3和OLMo-2-1124-7B-Instruct等，共计135个状态。

**📈 对比分析**

与传统能力预测方法相比，直接读出和操作读出在三家族的开发中均减少了39.4%的均方误差（MSE）。在GLM-4-9B上，直接读出和操作读出分别将MSE降低了71.8%和78.3%。在Granite-3.1-8B上，直接读出达到RMSE 0.544，操作读出达到0.634，均优于能力预测的1.172。

**⚠️ 局限性**

限制在于模型的家族和动作之间的异质性可能影响预测的准确性，且在某些情况下，操作读出可能无法有效捕捉到目标动作的变化。

---

## 533. CLAMP: Constrained Decoding for Vision-Language Embodied Planning

**arXiv ID:** 2609.08602 | [PDF](https://arxiv.org/pdf/2609.08602v1)

**作者:** Tianyi Ma `[一作]` (Michigan State University), Parisa Kordjamshidi `[通讯]` (Michigan State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CLAMP，一个多模态约束引导框架，用于将视觉语言模型（VLM）生成的计划转化为可执行的行动序列，确保生成的计划符合执行所需的约束条件。

**💡 创新点**

CLAMP的创新点在于结合了基于观察的硬约束和隐马尔可夫模型（HMM）引导，能够在解码过程中直接强制执行约束，从而提高计划的可执行性和安全性。

**🔧 技术方法**

使用了隐马尔可夫模型（HMM）进行世界状态的前瞻性引导，并结合了观察条件的确定性有限自动机（DFA）来强制执行硬约束。

**📊 数据集**

在VLABench、SafeAgentBench和TaPA等多个基准数据集上进行了实验，评估了CLAMP的性能。

**📈 对比分析**

与现有方法相比，CLAMP在VLABench上将Qwen3-VL-8B的得分从28.7提高到37.1，且在SafeAgentBench上将符号规则违规率从0.41降低到0.05，显示出约束解码在任务级规划和约束满足方面的显著改进。

**⚠️ 局限性**

CLAMP的局限性在于其依赖于VLM规划器的离散符号技能调用，且在感知和规范方面存在潜在的错误，可能导致有效计划被拒绝或不安全的计划被接受。

---

## 534. TontaubeV1: Streaming Text-to-Speech with Hierarchical Codec Modeling and Bounded Context

**arXiv ID:** 2609.08703 | [PDF](https://arxiv.org/pdf/2609.08703v1)

**作者:** Fritz Cremer `[一作]`, Jonathan Cremer `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的文本到语音系统模型，旨在在保持自然韵律的同时实现高效推理，能够在单个消费级GPU上进行流式处理。

**💡 创新点**

创新点在于采用了分层的DualCodec表示法，将语义流与后续的声学细化分开，并通过多个小型变换器逐步添加声学细节，从而在保持高质量的同时降低计算成本和延迟。

**🔧 技术方法**

使用了Qwen3系列的变换器模型，结合DualCodec和VibeVoice的声学编码器和解码器。

**📊 数据集**

使用了约200,000小时的配对语音和文本数据，主要来自公共领域的有声书录音和公开发布的语音语料库。

**📈 对比分析**

在有声书阅读基准测试中，该模型的韵律表现与ElevenLabs Flash v2.5相当，优于Fish Audio S2 Pro、Gradium API和Cartesia Sonic 3，且在多个输入情况下的实时因子为0.02，显示出良好的性能。

**⚠️ 局限性**

局限性包括自回归语义生成可能会遗漏、重复或改变文本，参考条件可能不完美地转移身份，长段落分块可能引入不连续性，且序列化的四阶段因子化增加了延迟。

---

## 535. Record Grouping Controls Evidence Weight in Language Models

**arXiv ID:** 2609.08698 | [PDF](https://arxiv.org/pdf/2609.08698v1)

**作者:** Zhongxuan Liu `[一作]` (Harbin Institute of Technology), Hongzhi Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何通过提供的分区来控制记录的聚合和去重，以优化语言模型的表现。具体来说，提出了一种预生成的分组表示方法，能够在保留互补内容的同时去除组内重复项，并为每个组分配一个有界的贡献。

**💡 创新点**

创新点在于提出了一种新的分组表示方法，该方法结合了提供的分区、组内内容聚合和每组的有界贡献，同时建立了内容感知的分区误差界限，并通过实验验证了其有效性。

**🔧 技术方法**

使用了理论分析和实验方法，包括自然文本干预和控制面板实验，来验证所提出的模型和方法的有效性。

**📊 数据集**

使用了多个公共数据集进行实验，包括HUMAN和WEB领域的多个问题和声明，具体涉及101个PERSPECTRUM声明和138个ConflictingQA问题。

**📈 对比分析**

通过与六槽控制组进行比较，结果表明，内容固定的错误分裂会增加10.27到32.66个百分点，而错误合并则会减少9.13到31.79个百分点，所有模型在16个单元中均保持正向方向。

**⚠️ 局限性**

限制在于所提出的方法依赖于提供的分区的质量和准确性，分区错误可能会影响模型的表现和决策稳定性。

---

## 536. HOPE: Heterophily-Aware Open-Set Node Classification with Pseudo-Extrapolation

**arXiv ID:** 2609.08685 | [PDF](https://arxiv.org/pdf/2609.08685v1)

**作者:** Yumeng Dai `[一作]` (Xi'an Jiaotong University), Tao Qin `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为HOPE的新框架，用于处理异质图中的开放集节点分类问题，旨在解决传统方法在异质图中表现不佳的挑战。

**💡 创新点**

创新点在于引入了结构增强的特征初始化层、可信的邻域聚合机制和异质性引导的伪外推策略，以提高已知类节点的分类能力和未知类节点的拒绝能力。

**🔧 技术方法**

使用了图神经网络（GNN）技术，结合结构增强的特征初始化、可信邻域聚合和伪未知代理生成策略。

**📊 数据集**

在多个异质图数据集上进行了实验，包括Chameleon、Squirrel、Wisconsin、Amazon-Ratings、Roman-Empire、Actor和Arxiv-Year。

**📈 对比分析**

与多种最先进的闭集和开放集模型进行了比较，HOPE在准确性和F1分数上均表现优异，显示出其有效性和鲁棒性。

**⚠️ 局限性**

限制在于该方法可能对图的结构特征依赖较强，且在极端异质性情况下的表现仍需进一步验证。

---

## 537. BIFTA: Brain-Inspired Few-Shot Tactile Adaptation for Unknown Sensors

**arXiv ID:** 2609.08673 | [PDF](https://arxiv.org/pdf/2609.08673v1)

**作者:** Boheng Liu `[一作]` (Beijing Institute of Technology), Xia Wu `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为BIFTA的框架，旨在通过少量标记支持集快速适应未知触觉传感器，从而解决已知传感器训练模型在未知传感器上性能急剧下降的问题。

**💡 创新点**

BIFTA借鉴了大脑的快速感官适应机制，结合双视图统计记忆、支持条件谱几何和不确定性门控递归推理，能够在不更新编码器的情况下实现高效的适应。

**🔧 技术方法**

使用了双视图统计记忆、支持条件谱图和不确定性门控递归推理等技术。

**📊 数据集**

在三个触觉数据集上进行了广泛的基准测试，包括SITR、TacVerse Shape和TacQuad。

**📈 对比分析**

与其他几种方法（如Tip-Adapter、SimpleShot、LaplacianShot等）进行比较，BIFTA在SITR上仅使用10%的标记目标数据时，准确率从6.86%提升至87.09%，超出最强对比方法47.22个百分点，且在不同数据集和任务上均表现出良好的泛化能力。

**⚠️ 局限性**

BIFTA的局限性在于其依赖于少量标记数据，可能在极端情况下对数据的依赖性较高，且在实时适应性和计算效率方面仍有待优化。

---

## 538. X2Streaming-ASR: wait when uncertain, emit when ready for streaming ASR

**arXiv ID:** 2609.08672 | [PDF](https://arxiv.org/pdf/2609.08672v1)

**作者:** Zhiwei Lin `[一作]` (X Square Robot), Qian Wang `[通讯]` (X Square Robot)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种新的流式自动语音识别系统X2Streaming-ASR，旨在优化实时语音代理的部分转录准确性和低延迟。

**💡 创新点**

创新点在于将流式识别分解为何时提交和提交什么，采用三阶段训练程序来优化识别能力和提交策略。

**🔧 技术方法**

使用了强化学习技术，结合了监督学习和基于组相对策略优化（GRPO）的训练方法。

**📊 数据集**

使用了AISHELL-1/2/3和WenetSpeech数据集进行实验。

**📈 对比分析**

与现有的流式基线相比，X2Streaming-ASR在AISHELL-1和AISHELL-3上实现了最佳的字符错误率（CER），并且平均延迟为27-84毫秒，相比之下，基线的延迟为409-585毫秒。

**⚠️ 局限性**

限制在于当前模型的识别能力可能在某些情况下不如基线模型，尤其是在特定的流式识别任务中。

---

## 539. Linear Programming Bounds for LCD Codes via Gauss Phases

**arXiv ID:** 2609.08662 | [PDF](https://arxiv.org/pdf/2609.08662v1)

**作者:** Ming-Hsuan Kang `[一作]` (National Yang Ming Chiao Tung University), Maosheng Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

本文展示了在有限域上，k维线性码是线性互补对偶（LCD）的条件，并通过将权重枚举器的根单位值的相位转化为线性约束，构建了高斯相位线性规划（LP）以改进LCD码的界限。

**💡 创新点**

创新点在于将权重枚举器的相位与二进制情况下的奇偶性类型结合，形成精确的线性约束，并将其纳入高斯相位线性规划中，从而在保持接近标准汉明LP大小的同时，增强了对LCD码的界限。

**🔧 技术方法**

使用了高斯相位线性规划技术，该技术结合了权重分布和对偶码的普通权重分布，并引入了常数大小的分支方程。

**📊 数据集**

在二进制和三进制范围内进行了计算，展示了在62个二进制和39个三进制参数对中系统性地增强了汉明LCD松弛。

**📈 对比分析**

与已建立的混合联合权重枚举LP进行比较，结果显示在四个严格改进中，每个案例都将基准上界降低了一个，且每个严格比较都通过有理可行性证人和整数Farkas证书得到了验证。

**⚠️ 局限性**

限制在于对于q>3的情况，所有线性码都与LCD码是单调等价的，因此在更高的q值下，研究的复杂性和适用性可能受到限制。

---

## 540. SUN: Reaching for Novelty in Reinforcement Learning

**arXiv ID:** 2609.08642 | [PDF](https://arxiv.org/pdf/2609.08642v1)

**作者:** Wenyan Yang `[一作]` (Aalto University), Simone Parisi `[通讯]` (Tampere University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的目标选择框架，结合了新颖性和可达性，以提高强化学习中的探索效率。

**💡 创新点**

创新点在于引入了SUccessor-to-Novelty (SUN)指标，该指标同时考虑了目标的新颖性和可达性，并且可以无缝集成到任何离线策略的强化学习算法中。

**🔧 技术方法**

使用了基于后继值函数的SUN指标来评估目标的可达性，并通过轻量级的伪计数来评估目标的新颖性。

**📊 数据集**

使用了标准和新颖的环境进行基准测试，包括不可达状态、不可逆转的转移、障碍物、迷宫和无界空间等。

**📈 对比分析**

与现有的最先进方法（如AdaGoal和DISCOVER）进行比较，SUN在所有环境中均表现出色，尤其是在具有不可达或难以到达状态的环境中，显示出更好的覆盖率和均匀性。

**⚠️ 局限性**

限制在于，尽管SUN在许多环境中表现良好，但在大规模空间（如图像观察）中，基于计数的新颖性可能效果不佳，未来的工作可以考虑结合更丰富的信号来改进探索。

---

## 541. Suan: Rectifying Direct Preference Safety Alignment in Large Language Models

**arXiv ID:** 2609.08634 | [PDF](https://arxiv.org/pdf/2609.08634v1)

**作者:** Oleksandr Cherednichenko `[一作]` (Umeå University), Roman Klypa `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的偏好优化算法Suan，以增强大型语言模型（LLMs）的安全性和响应质量。

**💡 创新点**

Suan通过直接在梯度水平上制定优化目标，提供了更可解释和稳健的训练动态，克服了现有方法的局限性，如过度拒绝和质量下降。

**🔧 技术方法**

使用了一种新的偏好梯度优化方法，设计了简单且可解释的训练目标，避免了传统方法中的变分推导。

**📊 数据集**

在多个开源大型语言模型（如Mistral-12B、Falcon3-7B等）上进行了评估，并使用了Alpaca指令数据集和PKU-Safe-RLHF偏好数据集。

**📈 对比分析**

与DPO、IPO和SafeDPO等现有方法进行了比较，Suan在安全性、合规性和有用性方面表现优越，显著降低了过度拒绝率，同时保持了响应质量。

**⚠️ 局限性**

尽管Suan在多个基准测试中表现出色，但仍需在其他偏好优化方法和更显著的强化学习框架中进行进一步评估，未来研究可探索其对各种对抗性攻击的鲁棒性。

---

## 542. Dynamics of meaning: Towards the Evaluation of Diachronic Semantic Change in Sinhala

**arXiv ID:** 2609.08609 | [PDF](https://arxiv.org/pdf/2609.08609v1)

**作者:** Nevidu Jayatilleke `[一作]` (University of Moratuwa), Nisansa de Silva `[通讯]` (University of Moratuwa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了斯里兰卡的僧伽罗语在13世纪到20世纪的语义变化，采用多阶段计算框架进行分析。

**💡 创新点**

创新点在于引入了双向语义影响修剪方法，利用上下文化嵌入来识别和区分系统性语义变化与瞬态多义扩展。

**🔧 技术方法**

使用了相似性矩阵对齐(SMA)和正交普克雷斯特斯(OP)技术，以及上下文化嵌入模型。

**📊 数据集**

使用了最大的僧伽罗语历时语料库，该语料库包含来自185份文档的241,491个词，经过增强处理后用于分析。

**📈 对比分析**

通过与静态嵌入的统计邻域分析比较，发现OP对齐在识别时间相似性下降方面表现更为稳定，且在数据稀缺环境中，模型的表现差异显著。

**⚠️ 局限性**

限制因素包括数据稀缺、历史僧伽罗语的词形还原和词性标注的准确性不足，这些都可能影响模型的可靠性和分析结果的准确性。

---

## 543. Navigating the Latent Manifold: Proactive Concept Drift Adaptation for Resilient NIDS

**arXiv ID:** 2609.08623 | [PDF](https://arxiv.org/pdf/2609.08623v1)

**作者:** Chao Zha `[一作]` (Zhejiang University), Ruyun Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为DriftXpert的新型网络入侵检测系统（NIDS），旨在应对动态网络环境中的概念漂移问题。

**💡 创新点**

创新点在于提出了一种解耦的两阶段离线自适应框架，结合无监督异常度量和表示一致性对齐策略，有效应对概念漂移和灾难性遗忘。

**🔧 技术方法**

使用了无监督学习、深度聚类、对比学习等技术，结合特征映射约束和选择性冻结机制。

**📊 数据集**

使用了CICIDS-2017和CICIDS-2018等公共数据集，以及真实企业网络流量数据集进行实验。

**📈 对比分析**

与五种基线方法进行比较，DriftXpert在适应性和稳定性方面表现优越，尤其在概念漂移情况下，准确率和召回率均高于其他方法。

**⚠️ 局限性**

局限性在于模型的复杂性和对超参数的敏感性，可能需要在不同环境中进行调整以优化性能。

---

## 544. HiBRIDGE: A Hierarchical Bayesian Neural Network Framework for Interpretable Dialogue Management in Group-Robot Interaction

**arXiv ID:** 2609.08678 | [PDF](https://arxiv.org/pdf/2609.08678v1)

**作者:** Massimiliano Nigro `[一作]` (Politecnico di Milano), Fethiye Irmak Dogan `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为HiBRIDGE的分层贝叶斯神经网络框架，用于多方人机交互中的对话管理，帮助机器人决定与谁交谈以及说什么。

**💡 创新点**

创新点在于结合了贝叶斯建模和分层分类，能够在有限的数据下进行不确定性感知的预测，并提供更具可解释性的决策过程。

**🔧 技术方法**

使用了分层贝叶斯神经网络，结合决策树替代模型来生成可解释的决策路径。

**📊 数据集**

使用了三个离线人机交互数据集，包括MuMMER、Addlesee和Spitale数据集，涵盖不同的交互场景。

**📈 对比分析**

与多种基线方法（包括大型语言模型和监督学习方法）进行比较，结果显示贝叶斯模型在预测性能上优于确定性模型和多种最先进的基线，且分层模型提供了更好的可解释性。

**⚠️ 局限性**

限制在于分层建模未能显著提高预测性能，且在某些可解释性指标上未能显示出一致的效果，未来需要在更长时间的交互中进行研究。

---

## 545. Generalized Graph Search Trees

**arXiv ID:** 2609.08625 | [PDF](https://arxiv.org/pdf/2609.08625v1)

**作者:** Florian Krowiorz `[一作]` (Brandenburg University of Technology), Robert Scheffler `[通讯]` (Brandenburg University of Technology)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图搜索树的识别问题，提出了一种广义的图搜索树概念，允许每个顶点的前驱邻居作为父节点，并探讨了这一问题的复杂性。

**💡 创新点**

创新点在于将图搜索树的概念推广到允许每个前驱邻居作为父节点的情况，并证明了大多数搜索的识别问题是NP完全的，同时对于某些特定图类的搜索提供了多项式时间算法。

**🔧 技术方法**

使用了图搜索算法的理论框架，包括广度优先搜索（BFS）、深度优先搜索（DFS）、字典序广度优先搜索（LBFS）、字典序深度优先搜索（LDFS）等。

**📊 数据集**

研究中使用了多种图类，包括二部图、和弦图等，分析了这些图类上不同搜索的识别问题。

**📈 对比分析**

通过与现有方法的比较，发现对于大多数搜索，识别问题是NP完全的，而对于某些特定的图类（如二部图和和弦图），则可以在多项式时间内解决。

**⚠️ 局限性**

限制在于对于某些图类（如BFS在和弦图上的情况）和特定搜索（如LDFS、MCS和MNS在二部图上的情况），仍然存在未解决的问题。

---

## 546. Hyperparameter Scaling Laws Across MoE Sparsity

**arXiv ID:** 2609.08690 | [PDF](https://arxiv.org/pdf/2609.08690v1)

**作者:** Changxin Tian `[一作]` (Ling Team, Ant Group), Jun Zhou `[通讯]` (Ling Team, Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了超稀疏混合专家模型（MoE）的最佳学习率和批量大小如何随着计算规模、训练时间和稀疏性变化。通过系统的实验，发现传统的规模变量无法描述超稀疏性水平之间的超参数变化，激活比率A必须作为额外的预测维度。

**💡 创新点**

创新点在于提出了统一的超参数缩放法则，表明在固定稀疏性下，最佳学习率遵循计算量C=MD的幂律，而最佳批量大小遵循训练令牌D的幂律。激活比率A通过乘法幂律修正这两者的前因子。

**🔧 技术方法**

使用了混合线性注意力/MLA骨干网络和Muon优化器，进行了1800次预训练实验，系统地变化了激活的非嵌入参数数量、总非嵌入参数数量、激活比率和训练令牌数量。

**📊 数据集**

实验使用了大规模的训练数据集，处理了约20万亿个令牌，涵盖了从约1000万到3.24亿的激活参数规模，达到6B的总非嵌入参数。

**📈 对比分析**

通过与现有的缩放法则进行比较，提出的法则在超参数预测上表现优于传统的规模仅法则和加法法则，且在一个冻结的目标配置下，预测结果位于观察到的近最优损失平面上。

**⚠️ 局限性**

限制在于实验和评估范围仅限于单一的混合线性注意力/MLA骨干网络和特定的优化器，未来需要在其他模型规模、架构和优化器上进行验证。此外，当前的证据未能显著区分无交互模型与交互模型的优势。

---

## 547. CausalChapter: Improving Long-Video Chaptering with Interventional Dependency Modeling

**arXiv ID:** 2609.08686 | [PDF](https://arxiv.org/pdf/2609.08686v1)

**作者:** Xinran Duan `[一作]` (Beijing Normal University), Hua Huang `[通讯]` (Beijing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为CausalChapter的框架，用于长视频的章节生成，旨在通过估计预测级影响来改善章节边界定位和章节描述生成。

**💡 创新点**

创新点在于引入了基于干预的依赖建模，使用Local Dependency Shift (LCDS)和Cross-Segment Support Selection (CSSE)模块来解决边界错误传播和跨章节上下文碎片化的问题。

**🔧 技术方法**

使用了干预定义的预测依赖性来作为任务导向的支持信号，结合了轻量级的掩蔽和移除干预技术。

**📊 数据集**

在AVLecture和VidChapters-7M两个长视频章节基准上进行了实验，AVLecture包含长讲座视频的ASR转录和人类注释的主题边界。

**📈 对比分析**

与现有的密集视频字幕方法和长视频LLM基线进行比较，CausalChapter在描述质量和边界定位上均表现优异，CIDEr得分从99.78提高到110.12，F1@30从63.93%提高到72.97%。

**⚠️ 局限性**

局限性包括干预定义的依赖性仅作为预测级影响，而非真实世界因果关系的发现；使用的增强AVLecture基准可能继承了注释管道的风格规律；CSSE引入了额外的推理成本。

---

## 548. SynthRCT: Scalable Conditional Deformation Synthesis for Synthetic Repeat CT Generation

**arXiv ID:** 2609.08627 | [PDF](https://arxiv.org/pdf/2609.08627v1)

**作者:** Tomas Guija-Valiente `[一作]` (Universidad Rey Juan Carlos), Norberto Malpica `[通讯]` (Universidad Rey Juan Carlos)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为SynthRCT的可扩展条件生成框架，用于合成3D解剖变形，以生成合成重复CT。

**💡 创新点**

创新点在于通过条件变分自编码器学习潜在变形空间，并将采样的潜在代码解码为局部静态速度场，从而生成患者特定的解剖变形。

**🔧 技术方法**

使用了条件变分自编码器（Conditional Variational Autoencoder）和局部静态速度场（SVF）生成技术。

**📊 数据集**

使用了DIR4DCT数据集，包括10名胸部4DCT患者，每名患者有10个呼吸相位和75个标注的地标轨迹。

**📈 对比分析**

与传统的配准方法（如SyN、Demons和VoxelMorph）相比，SynthRCT在生成的变形上实现了竞争性的对齐质量，同时保持空间规律性，但在对齐精度上略低于这些确定性配准基线。

**⚠️ 局限性**

限制在于呼吸4DCT数据的变异性受限，未来工作应评估更大和更多样化的数据集，以验证潜在空间是否能够解开多种变形模式。

---

## 549. Multi-Level-Set-Based Physics-Driven Neural Network to Solve 3-D Inverse Scattering Problems

**arXiv ID:** 2609.08594 | [PDF](https://arxiv.org/pdf/2609.08594v1)

**作者:** Yutong Du `[一作]` (Northwestern Polytechnical University), Peixian Han `[通讯]` (Northwestern Polytechnical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于多级集的物理驱动神经网络求解器（LSPDNN），用于解决三维电磁逆散射问题。

**💡 创新点**

创新点在于利用多级集函数的神经网络参数化，结合软联合多材料对比模型，显著提高了复杂散射体的重建精度和边界清晰度。

**🔧 技术方法**

使用了基于坐标的傅里叶特征多层感知器（MLP）来参数化未知散射体的水平集函数，并引入了自适应正则化权重学习策略。

**📊 数据集**

使用了模拟数据集，目标区域为0.15m×0.15m×0.15m的立方体，采用了方法时刻（MoM）计算散射场。

**📈 对比分析**

与基于点云的深度学习求解器和PDNN进行比较，LSPDNN在重建精度上显著优于这两种方法，平均相对误差分别降低了89.4%和92.1%。

**⚠️ 局限性**

限制在于计算效率仍需提高，未来工作将集中在提升计算效率和自动选择模型复杂性方面。

---

## 550. Rescuing Performance from the Demo: Co-Designing Drum Gesture Mappings with a Percussionist

**arXiv ID:** 2609.08587 | [PDF](https://arxiv.org/pdf/2609.08587v1)

**作者:** Jordie Shier `[一作]` (Queen Mary University of London), Andrew McPherson `[通讯]` (Imperial College London)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究与一位专业打击乐手合作，开发了一种手势映射工具包，并录制了一张十首曲目的专辑，旨在探讨技术约束与音乐美学之间的张力。

**💡 创新点**

创新点在于提出了一种新的实时映射方法，能够将连续的打击手势映射到合成器参数，并通过反思性实践揭示了设计过程中的洞察。

**🔧 技术方法**

使用了神经网络（RNN）进行手势识别和映射，结合了技术实践研究（TPR）的方法。

**📊 数据集**

使用了研究者与打击乐手共同开发的手势映射工具包和录制的音乐专辑作为数据集。

**📈 对比分析**

通过与打击乐手的合作，研究了技术捕获的风险，并通过反思性实践评估了技术对音乐实践的影响，发现技术的影响可能会被无条件接受。

**⚠️ 局限性**

实验室环境的局限性在于缺乏音乐家的更广泛社会背景，可能导致技术影响的美学未能得到真实的支持或质疑。

---

## 551. Limitations of Automated Simulatability: LLM Simulators Can Bypass Explanations

**arXiv ID:** 2609.08585 | [PDF](https://arxiv.org/pdf/2609.08585v1)

**作者:** Antonin Poché `[一作]` (IRT Saint Exupéry), Vera Schmitt `[通讯]` (Technische Universität Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种评估解释有效性的自动化方法，称为可模拟性（simulatability），通过使用大型语言模型（LLM）模拟器来替代人工评估，从而进行大规模实验。

**💡 创新点**

创新点在于识别了现有可模拟性评估的两个局限性：一是当类名有意义时，模拟器可以直接解决分类任务而不依赖于解释；二是类匿名化可能会奖励泄露隐藏标签映射的解释。

**🔧 技术方法**

使用了大型语言模型（LLM）作为模拟器，并引入了类作为概念的基线方法来评估解释的有效性。

**📊 数据集**

使用了多个数据集，包括BIOS、IMDB、Rotten Tomatoes、AG News和GoEmotions等。

**📈 对比分析**

与ConSim的比较显示，解释在非匿名化实验中对模拟器预测的影响有限，主要依赖于任务先验，而不是解释内容。尽管统计上显著，但解释的增益相对较小。

**⚠️ 局限性**

局限性在于研究使用的LLM模拟器参数最多为15B，可能无法推广到更大或更好的闭源LLM。此外，分析集中在LLM模拟器在训练中可能已经遇到的用例上，未考虑潜在的污染问题。

---

## 552. Global Divergence, Local Convergence: Representation Geometry in SSMs and Transformers

**arXiv ID:** 2609.08692 | [PDF](https://arxiv.org/pdf/2609.08692v1)

**作者:** Amit Ben-Artzy `[一作]` (Hebrew University of Jerusalem), Roy Schwartz `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了状态空间模型（SSM）和变换器架构之间的内部表示几何差异，分析了它们在语言建模任务中的表现和结构特征。

**💡 创新点**

创新点在于系统性地比较了SSM和变换器的表示几何，发现尽管它们的几何结构不同，但在有效容量和概念编码方面表现出惊人的相似性。

**🔧 技术方法**

使用了多尺度分析、秩约束探测、主成分分析（PCA）和中心核对齐（CKA）等技术。

**📊 数据集**

使用了FineWeb-Edu数据集和LAMA TREx数据集进行实验。

**📈 对比分析**

通过比较不同模型的有效维度和几何特征，发现SSM模型的表示更均匀，而变换器则表现出明显的各向异性。尽管如此，两者在有效容量上表现相似。

**⚠️ 局限性**

限制在于本研究主要依赖于相关性分析，未能建立概念编码与下游生成之间的直接因果关系。

---

## 553. When Victorian Becomes a Prompt: Literary Periodization as a Generative Constraint in 100 AI-Generated Novels

**arXiv ID:** 2609.08689 | [PDF](https://arxiv.org/pdf/2609.08689v1)

**作者:** Mehdy Sedaghat Payam `[一作]` `[通讯]` (Independent Scholar), Mehdy Sedaghat Payam (Independent Scholar)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究定义并测试了“生成性时期化”，即在文本生成过程中使用文学时期标签的方式。研究分析了在维多利亚和零风格条件下使用GPT、Qwen和Llama生成的100本小说。

**💡 创新点**

创新点在于提出了生成性时期化的概念，探讨了历史和文学类别如何在文本生成中影响文本的形式和特征。

**🔧 技术方法**

使用了GPT、Qwen和Llama等大型语言模型进行文本生成，并通过“时期对齐分数”（PAS）评估生成文本与历史文本的对齐程度。

**📊 数据集**

使用的数据集包括410本小说，其中100本为AI生成，310本为人类创作，涵盖了19世纪小说、零风格小说和现代历史小说。

**📈 对比分析**

通过比较不同生成条件下的PAS分数，发现维多利亚条件下生成的文本在PAS分数上显著高于零风格文本，表明生成性时期化在不同模型中产生了一致的历史方向变化。

**⚠️ 局限性**

限制在于不同AI模型在生成条件下的技术差异，导致无法仅通过架构或能力来解释模型间的差异。此外，PAS是相对坐标，不能作为历史真实性的绝对测量。

---

## 554. Entropic Risk-Sensitive Evolutionary Learning and Equilibrium Selection in Coordination Games

**arXiv ID:** 2609.08677 | [PDF](https://arxiv.org/pdf/2609.08677v1)

**作者:** Solaleh Mohammadi `[一作]` (University of Maryland), Kaiqing Zhang `[通讯]` (University of Maryland)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了风险敏感的进化学习动态及其在协调博弈中的长期均衡选择行为。

**💡 创新点**

创新点在于将经典的熵风险度量引入到进化学习动态中，揭示了风险态度如何影响长期均衡选择。

**🔧 技术方法**

使用了熵风险度量（ERM）和两种标准的修正协议：带突变的最佳响应和逻辑选择。

**📊 数据集**

使用了2×2协调博弈的模型进行分析，并扩展到对称k行动博弈。

**📈 对比分析**

通过比较不同风险态度下的动态，发现风险寻求行为倾向于选择收益主导的均衡，而风险厌恶行为则倾向于选择最大最小均衡。对于超主导均衡，无论风险态度如何，均是随机稳定的。

**⚠️ 局限性**

限制在于分析主要集中在同质风险态度的情况下，未来可以考虑异质风险态度的影响。

---

## 555. CASD: Chunk-Aligned Semantic Distillation for Multi-StageRobot Manipulation

**arXiv ID:** 2609.08638 | [PDF](https://arxiv.org/pdf/2609.08638v1)

**作者:** Tinghe Ding `[一作]` (Ant Group), He Wang `[通讯]` (Ant Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的方法Chunk-Aligned Semantic Distillation (CASD)，用于为整个动作块推导语义目标，解决了多阶段操作任务中标签描述不完整的问题。

**💡 创新点**

创新点在于通过对动作块的阶段占用进行加权，定义了一个语义目标，并通过CASD生成器从当前观察、机器人状态和任务指令中预测该目标。

**🔧 技术方法**

使用了离线视觉-语言模型（VLM）和CASD生成器，结合了多种快速世界动作模型（Fast-WAM）变体和DreamZero集成。

**📊 数据集**

使用了LIBERO和LIBERO-Plus数据集进行评估，涵盖了多种任务和分布变化。

**📈 对比分析**

与现有方法进行比较，IDM+在LIBERO上达到98.9%的平均成功率，Joint+在RoboTwin 2.0上达到93.0%，而DreamZero+在MolmoSpaces上达到47.9%的平均成功率，整体性能优于许多已发布的参考。

**⚠️ 局限性**

限制在于自动注释可能包含基础和边界错误，且策略回滚可能与用于蒸馏的演示不同，未来需要对注释时机的敏感性进行测试。

---

## 556. From Where to How: Continuous 4D Interaction Forecasting from Egocentric Video

**arXiv ID:** 2609.08636 | [PDF](https://arxiv.org/pdf/2609.08636v1)

**作者:** Qiaohui Chu `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种新的方法Coherent4D，用于从自我中心视频中进行连续的4D交互预测，旨在同时预测未来的3D交互位置和人体运动。

**💡 创新点**

创新点在于引入了Coherent4D数据集，提供了大规模的、时间对齐的交互位置和全身姿态序列，并提出了HIGFlow框架，将交互位置预测与全身姿态预测紧密结合。

**🔧 技术方法**

使用了HIGFlow框架，该框架包括语义动态位置预测和手部条件残差流匹配，采用了深度学习技术进行模型训练和推理。

**📊 数据集**

使用了Coherent4D数据集，该数据集包含约233K个样本，涵盖烹饪、健康和自行车修理三个领域，提供了时间对齐的3D交互位置和全身姿态。

**📈 对比分析**

与现有基线方法（如FIction、Qwen3-VL等）进行比较，HIGFlow在交互位置和姿态预测上均表现出一致的性能提升，尤其在烹饪和自行车修理领域表现突出。

**⚠️ 局限性**

限制在于，尽管HIGFlow在预测精度上有所提升，但在长时间预测中，交互位置的误差可能会累积，导致姿态预测不够准确，且未能完全捕捉人类意图和物体接触约束。

---

## 557. Sequential Lossy Compression With Causal Conditional Perception

**arXiv ID:** 2609.08611 | [PDF](https://arxiv.org/pdf/2609.08611v1)

**作者:** Photios A. Stavrou `[一作]` (EURECOM), Zixuan He `[通讯]` (EURECOM)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在因果条件感知标准下的序列有损压缩，比较了源分布和重建分布在相同重建历史下的表现。针对一阶马尔可夫源，提出了有限视野非预见率失真感知函数（NRDPF），并建立了最小可变长度和总和速率的上下界。

**💡 创新点**

创新点在于引入了条件自适应感知损失函数（PLF-CSA），并通过加强的强功能表示引理（SFRL）和共享随机性，推导出了一次性下界和上界，扩展了现有的序列压缩理论。

**🔧 技术方法**

使用了强化的强功能表示引理（SFRL）和共享随机性等信息论技术。

**📊 数据集**

使用了一阶马尔可夫源和时间变化的标量高斯-马尔可夫源作为数据集，特别关注均方误差（MSE）和条件平方Wasserstein-2保真度。

**📈 对比分析**

通过与现有的序列有损压缩方法进行比较，证明了所提出方法的高效性，尤其是在高斯源的情况下，得到了闭合形式的解决方案，恢复了经典的高斯非预见率失真函数（NRDF）。

**⚠️ 局限性**

限制在于该研究主要集中在一阶马尔可夫源和特定的高斯-马尔可夫源上，未来的工作将扩展到向量源和实际的预测视频编码架构。

---

## 558. GOLF: Global Observation with Local Focus for Calibration-Aware Stereo Interaction Field Estimation

**arXiv ID:** 2609.08607 | [PDF](https://arxiv.org/pdf/2609.08607v1)

**作者:** Minqiang Zou `[一作]` (JIIOV Technology), Yao Tang `[通讯]` (JIIOV Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种方法，旨在从同步的自我中心立体视图中预测每个手关节到被操控物体最近点的3D向量，获得了HANDS@ECCV 2026 SHOW3D互动场估计挑战的第一名。

**💡 创新点**

创新点在于结合了密集的全局上下文、本地采样的手/物体证据和共同框架的Plücker射线几何，采用了LoRA和可训练参数的DINOv3 ViT-H+/16模型。

**🔧 技术方法**

使用了DINOv3 ViT-H+/16模型，结合LoRA进行高效的骨干网络适配，并通过联合解码预测互动场。

**📊 数据集**

使用了SHOW3D数据集，该数据集包含来自10个受试者和21个物体的468个录音，测试集包含来自139个录音的20,042帧。

**📈 对比分析**

与其他方法相比，最终模型在隐藏测试集上达到了27.61的官方得分和27.96mm的平均ADE，通过与直接微调的变体进行等权重集成，进一步提高到27.47和27.82mm，确保了第一名。

**⚠️ 局限性**

限制在于全局上下文的保留与小型、被遮挡的手和物体的细节恢复之间的权衡，可能在某些情况下影响预测的准确性。

---

## 559. Estimating Semantic Ambiguity via Gaussian Context Distributions for VLM-Driven Traversability Analysis

**arXiv ID:** 2609.08583 | [PDF](https://arxiv.org/pdf/2609.08583v1)

**作者:** Ramona Häuselmann `[一作]` (Luleå University of Technology), George Nikolakopoulos `[通讯]` (Luleå University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的视觉基础可通行性估计管道，明确建模上下文不确定性，以解决视觉语言模型（VLM）中的语义歧义问题。

**💡 创新点**

创新点在于引入了概念锚定（Conceptual Anchoring），将开放词汇的VLM预测与连续的物理可通行性尺度相结合，并通过高斯上下文分布（GCD）来建模模型的响应。

**🔧 技术方法**

使用了高斯上下文分布（GCD）来生成密集的可通行性图和不确定性图，并利用VLM进行图像分割。

**📊 数据集**

在真实世界的GOOSE数据集上进行了实验验证，该数据集提供了多种户外环境的传感器数据和地面真实标注。

**📈 对比分析**

与现有的基线方法（如WayFast和CLIP）进行了比较，结果显示该方法在处理复杂场景时表现出竞争力，且能够提供统计不确定性估计，优于传统的最大化方法。

**⚠️ 局限性**

限制在于VLM的语义相似性与可通行性之间的关系并不总是成立，可能导致分布出现分离峰和低概率间隙，从而增加了方法的挑战性。

---

## 560. Comparison-Based Fair Division of Indivisible Chores

**arXiv ID:** 2609.08687 | [PDF](https://arxiv.org/pdf/2609.08687v1)

**作者:** Zehan Lin `[一作]` (University of Macau), Shengwei Zhou `[通讯]` (Nanyang Technological University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在比较查询模型下，如何公平地分配m个不可分割的工作给n个代理，提出了多种公平分配算法。

**💡 创新点**

提出了一种基于比较的算法，能够在O(n^3 log m)查询复杂度下计算PROP1分配，并且在MMS保证方面引入了压缩框架，能够在对m的对数依赖下实现(13/11 + ε)-MMS分配。

**🔧 技术方法**

使用了比较查询模型，算法只能通过比较两个捆绑的成本来获取信息，而不能直接访问数值成本。

**📊 数据集**

使用了多个数据集，包括不可分割的工作集，具体的工作集大小和代理数量在文中有详细描述。

**📈 对比分析**

与现有方法进行比较，提出的算法在查询复杂度上具有对数级别的依赖，性能上能够达到最先进的(13/11 + ε)-MMS保证，且在三代理情况下，EF1分配可以在O(log m)查询下实现。

**⚠️ 局限性**

限制在于该算法主要针对固定数量的代理(n为常数)进行设计，未来的工作需要扩展到更多代理的情况，并且对MMS失真有更深入的理解。

---

## 561. Measuring Sustainability in Multi-Scale High-Performance Computing

**arXiv ID:** 2609.08688 | [PDF](https://arxiv.org/pdf/2609.08688v1)

**作者:** Carlos J Barrios `[一作]` (Universidad Industrial de Santander), Yves Denneulin `[通讯]` (Université Grenoble-Alpes)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种多维度指标框架，用于衡量多尺度高性能计算（HPC）系统的可持续性和性能，旨在指导现代工作负载的部署策略。

**💡 创新点**

创新点在于结合了架构性能指标、系统利用率以及可持续性和准确性指标，形成一个综合的评估体系，帮助识别最佳操作点并优化调度器。

**🔧 技术方法**

使用了模块化混合测试平台进行实验，分析了不同指标之间的复杂关系，特别是准确性与能耗之间的权衡。

**📊 数据集**

数据集来自于多个模块的测试，包括低配置CPU、CPU-GPU组合和高配置CPU-GPU节点，涵盖了从5000MB到30000MB的多种工作负载。

**📈 对比分析**

通过与传统HPC系统的比较，展示了多尺度HPC系统在处理复杂任务时的高效性和可持续性，强调了在不同工作负载下的性能表现。

**⚠️ 局限性**

限制在于未定义I/O性能，且当前的评估主要依赖于制造商的标准值，未来研究将整合I/O等待时间系数以优化数据密集型任务的性能。

---

## 562. Neither Adversarial Training Nor Purification: Emergent Adversarial Robustness from Oscillatory Predictive Learning

**arXiv ID:** 2609.08683 | [PDF](https://arxiv.org/pdf/2609.08683v1)

**作者:** Mohammed-Yassine Habibi `[一作]` (Ecole Polytechnique), Makoto Yamada `[通讯]` (Okinawa Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的框架，称为振荡预测学习（OPL），结合了人工Kuramoto振荡神经元（AKOrN）和自监督预测预训练（X-PhiNet），以实现计算机视觉中的对抗鲁棒性，而无需对抗训练。

**💡 创新点**

创新点在于通过架构和表示学习的归纳偏差来实现对抗鲁棒性，而不是依赖于生成对抗样本或在测试时进行迭代去噪。

**🔧 技术方法**

使用了振荡神经元（AKOrN）和自监督学习框架（X-PhiNet）相结合的技术。

**📊 数据集**

使用了CIFAR-10和CIFAR-100数据集，并在CIFAR-10-C上进行了额外的腐蚀评估。

**📈 对比分析**

与其他随机对抗防御方法进行比较，OPL在CIFAR-10和CIFAR-100上分别达到了76.63%和50.44%的鲁棒准确率，优于先前的随机防御方法（如DiffPure: 71.29%）。

**⚠️ 局限性**

限制在于目前的实验仅限于CIFAR-10、CIFAR-10-C和CIFAR-100，未包括更高分辨率的数据集如ImageNet，且尚未提供机制理论来预测何种配置会是鲁棒的。

---

## 563. MoEMB: Scaling Universal Multimodal Embeddings with Efficient Mixture-of-Experts Models

**arXiv ID:** 2609.08663 | [PDF](https://arxiv.org/pdf/2609.08663v1)

**作者:** Xuanming Cui `[一作]` (University of Central Florida), Jianpeng Cheng `[通讯]` (AI at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的混合专家（MoE）模型，用于通用多模态嵌入（UME），通过在专家轴上扩展编码器的能力来提高性能。

**💡 创新点**

创新点在于首次将专家容量作为UME的扩展轴，并进行全面的模型设计和训练方案研究，同时实现了在计算成本几乎不变的情况下提高超过10个准确率点。

**🔧 技术方法**

使用了混合专家（MoE）技术，结合了自适应计算方法，如令牌修剪和自适应top-k，以提高计算效率。

**📊 数据集**

使用了MMEB-V2和MRMR两个大型多模态检索基准数据集进行评估。

**📈 对比分析**

与其他方法相比，提出的模型在MMEB-V2上达到了70.4的准确率，超越了所有基于公共数据训练的模型，且计算成本几乎不变。与TTE（思考-再嵌入）方法相比，提出的方法在准确率和吞吐量之间的权衡更优。

**⚠️ 局限性**

限制在于当前模型在外部数据训练的模型上表现不佳，并且不支持音频作为输入模态。未来的工作将包括在更大规模的数据上进行训练，并增加更多输入模态的支持。

---

## 564. Difficulty-Adaptive Tree-Structured Policy Optimization for Expanding Reasoning Coverage in RLVR

**arXiv ID:** 2609.08650 | [PDF](https://arxiv.org/pdf/2609.08650v1)

**作者:** Youngjun Yu `[一作]` (Pohang University of Science and Technology), Hwanjo Yu `[通讯]` (Pohang University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的强化学习框架DATPO，旨在通过优化训练时的结构设计来扩展模型的推理覆盖率。

**💡 创新点**

创新点在于引入了难度自适应的树结构搜索和兄弟多样性奖励项，以显著提高推理覆盖率（pass@k）。

**🔧 技术方法**

使用了难度自适应的树结构策略优化（DATPO），结合了基于句子熵的分叉策略。

**📊 数据集**

使用了MATH数据集进行数学推理基准测试。

**📈 对比分析**

与基线方法（如GRPO和AttnRL）相比，DATPO在pass@k上表现出显著的提升，尤其是在处理困难问题时，显示出更好的测试时间扩展性能。

**⚠️ 局限性**

限制在于计算兄弟多样性项需要额外的前向传递，这增加了计算开销。此外，模型的可扩展性在更大参数规模（≥ 7B）上的验证尚未进行。

---

## 565. Navigating the digital spectrum: Assessing political bias, stability, and downstream fairness in Large Language Models

**arXiv ID:** 2609.08637 | [PDF](https://arxiv.org/pdf/2609.08637v1)

**作者:** Luka Debevc `[一作]` (Jo{\), Matej Martinc `[通讯]` (Jo{\)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种评估框架，系统研究大型语言模型（LLMs）在政治坐标测试中的表现，特别是如何通过不同的提示、语言和交互模式影响模型的政治输出。

**💡 创新点**

创新点在于引入了一个统计上稳健的评估框架，能够通过多种实验配置的聚合来提取可靠的政治坐标估计，并且首次在14种语言中进行多语言评估。

**🔧 技术方法**

使用了多种技术，包括拉丁超立方体采样（LHS）、因子分解分析、反向工程评分函数等，来评估模型的政治倾向和敏感性。

**📊 数据集**

使用了多种数据集，包括政治坐标测试和针对仇恨言论检测及主题情感分类的敏感任务数据集。

**📈 对比分析**

通过与现有方法的比较，发现模型在不同提示和语言下的表现差异显著，尤其是在经济和社会轴上的坐标变化，表明模型对提示的敏感性较高。

**⚠️ 局限性**

局限性在于，尽管评估框架提供了更稳健的结果，但仍然可能受到模型训练数据的偏见影响，且在低资源语言中的文化适应性风险仍需进一步研究。

---

## 566. Leveraging contextual events on structure-aware next activity prediction

**arXiv ID:** 2609.08622 | [PDF](https://arxiv.org/pdf/2609.08622v1)

**作者:** Alessandro Mele `[一作]` (Polytechnic University of Marche), Domenico Potena `[通讯]` (Polytechnic University of Marche)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于实例图的结构感知下一个活动预测方法，旨在通过建模上下文过程实例来提高预测性能。

**💡 创新点**

创新点在于引入了多种编码策略来整合上下文信息，并通过图神经网络进行分类任务，展示了上下文过程实例对预测性能的积极影响。

**🔧 技术方法**

使用了图神经网络（GNN）和实例图（Instance Graphs）技术。

**📊 数据集**

使用了多个真实世界的事件日志数据集，包括BPI Challenge 2012、BPI Challenge 2020和Helpdesk数据集等。

**📈 对比分析**

通过与多种图基方法进行比较，提出的方法在多个数据集上表现出色，尤其是E4编码策略在大多数数据集上取得了最佳性能。

**⚠️ 局限性**

该方法的局限性在于上下文事件仅基于时间重叠进行识别，未考虑资源、活动类型或过程实例之间的因果关系。

---

## 567. Why shared attention vectors fail: a case for outcome-indexed tuning

**arXiv ID:** 2609.08615 | [PDF](https://arxiv.org/pdf/2609.08615v1)

**作者:** Lenard Dome `[一作]` `[通讯]` (University of Tübingen), Lenard Dome (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

论文探讨了在多结果学习中，全球共享的注意力向量不稳定的问题，并提出了一种基于结果索引的注意力矩阵作为解决方案。

**💡 创新点**

创新点在于引入了结果索引的注意力矩阵，替代了不稳定的全球共享注意力向量，从而改善了多结果学习模型的表现。

**🔧 技术方法**

使用了基于梯度下降的学习技术，并通过合成实验验证了提出的注意力矩阵的有效性。

**📊 数据集**

使用了合成实验数据集来评估不同注意力机制的表现。

**📈 对比分析**

通过三项合成实验比较了共享向量和结果索引注意力矩阵，结果表明结果索引注意力矩阵能够收敛到有意义的表示，而共享向量则无法做到这一点。

**⚠️ 局限性**

限制在于该方法可能在特定情况下仍然受到其他因素的影响，且未能提供对所有类型学习任务的普适解决方案。

---

## 568. Graph-Based Personalized Memory for LLM Agents: Representation, Evolution, Retrieval, and Evaluation

**arXiv ID:** 2609.08599 | [PDF](https://arxiv.org/pdf/2609.08599v1)

**作者:** Dac Duy Anh Nguyen `[一作]` (Griffith University), Alan Wee-Chung Liew `[通讯]` (Griffith University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文探讨了基于图的个性化记忆在大型语言模型（LLM）代理中的应用，提出了一种生命周期导向的视角，组织了现有研究，涵盖记忆表示、演变、检索和评估。

**💡 创新点**

创新点在于提出了一种生命周期分类法，系统化了基于图的个性化记忆的研究，并识别了当前评估实践中的空白和未来研究方向。

**🔧 技术方法**

使用了图结构来表示用户信息，通过节点和边的关系来组织用户的事实、偏好和经历，支持个性化决策。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到了一些现有的基准和评估方法。

**📈 对比分析**

通过比较现有的个性化记忆系统，讨论了不同设计选择的优缺点，强调了基于图的记忆在个性化代理中的重要性，性能上表现出更好的用户建模能力。

**⚠️ 局限性**

限制在于当前的个性化图记忆仍处于早期阶段，面临构建长期、可信和可操作的用户模型的挑战，未来需要在可扩展性和多模态记忆方面进行更多研究。

---

## 569. A Three-Tier Persona Vector for Controllable User Simulation in Agentic Evaluation

**arXiv ID:** 2609.08592 | [PDF](https://arxiv.org/pdf/2609.08592v1)

**作者:** Rahul Khedar `[一作]` (PayPal AI), Mouli V `[通讯]` (PayPal AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种三层次的人物向量模型，用于在代理评估中进行可控的用户模拟，旨在提高用户输入的多样性和真实性。

**💡 创新点**

创新点在于构建了一个包含23个操作化维度的三层次人物向量，能够更好地模拟用户行为和情感状态，并通过情境反应设计验证了模型的有效性。

**🔧 技术方法**

使用了高斯噪声采样、分层结构、可审计的相关性规则和查询复杂度叠加等技术。

**📊 数据集**

使用了64,698个多轮对话的合成数据集，涵盖8个命名的用户角色和3个生产语料库。

**📈 对比分析**

通过与现有的用户模拟方法进行比较，发现该模型在目标达成率上有15.8个百分点的差异，且在不同情境下同一角色的行为表现出显著差异，验证了模型的有效性。

**⚠️ 局限性**

限制在于没有学习协方差，情感状态在会话中保持不变，信息损失以及文化有效性问题。

---

## 570. Leveraging Visual and Geometric Priors for Metric-scale and Complete Vehicle Gaussian Reconstruction from Limited Views

**arXiv ID:** 2609.08841 | [PDF](https://arxiv.org/pdf/2609.08841v1)

**作者:** Jinyu Miao `[一作]` (Tsinghua University), Diange Yang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种前馈车辆资产重建方法，利用稀疏的单侧观察重建车辆的3D高斯表示。

**💡 创新点**

创新点在于结合了视觉基础模型和对称感知克隆策略，以实现度量尺度和完整的3D高斯构建。

**🔧 技术方法**

使用了视觉基础模型（MapAnything）和可训练的高斯头（ResNet-50编码器和UNet风格解码器）。

**📊 数据集**

使用了3DRealCar数据集，该数据集包含多种形状和外观的车辆。

**📈 对比分析**

与现有方法（如3D Gaussian Splatting、DepthAnything和TRELLIS）相比，提出的方法在几何完整性和外观一致性方面显著优于这些基线，PSNR、SSIM和LPIPS指标均表现最佳。

**⚠️ 局限性**

限制在于该方法依赖于车辆的双边对称性，可能不适用于所有类型的车辆。

---

## 571. Approval-Based Multiwinner Voting with Candidate Qualities

**arXiv ID:** 2609.08830 | [PDF](https://arxiv.org/pdf/2609.08830v1)

**作者:** Niclas Boehmer `[一作]` (Hasso Plattner Institute, University of Potsdam), Markus Utke `[通讯]` (Eindhoven University of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本研究提出了一种新的基于批准的多赢家投票模型，其中每个候选人都有一个外生的质量评分，重新思考了在质量评分存在的情况下，比例代表制的含义。

**💡 创新点**

创新点在于引入了基于阈值和基于价值的比例代表性公理，分析了它们之间的关系、可满足性和计算复杂性，并提出了能够实现最强联合可满足组合的投票规则。

**🔧 技术方法**

使用了基于阈值和基于价值的公理分析方法，结合了计算复杂性理论。

**📊 数据集**

未具体提及使用的数据集，但讨论了候选人质量评分的多种来源，如相关性、可靠性、紧迫性和优点等。

**📈 对比分析**

通过引入新的互惠公理，研究了比例代表性与最大化所选候选人总质量之间的兼容性，结果表明在不增加计算成本的情况下，能够计算出保持总质量为最优的3/4的比例委员会。

**⚠️ 局限性**

限制在于计算价值EJR委员会是NP难的，这表明在参与预算中计算EJR的复杂性仍然是一个开放问题。

---

## 572. Beyond Gait: Person Identification from Millimeter-Wave Point Clouds Across Activities of Daily Living

**arXiv ID:** 2609.08818 | [PDF](https://arxiv.org/pdf/2609.08818v1)

**作者:** Xilai Wang `[一作]` (University of Ottawa), Miodrag Bolic `[通讯]` (University of Ottawa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究探讨了如何通过毫米波点云在日常生活活动中进行人员识别，超越了传统的步态识别方法，使用了一个新的数据集mm-ADL，包含11名受试者在七种日常活动下的点云数据。

**💡 创新点**

创新点在于提出了一种活动条件识别框架，通过人类活动识别路由器将每个片段分配给特定活动的身份专家，从而利用日常活动提供的上下文信息来提高识别准确性。

**🔧 技术方法**

使用了双流静态-动态PointNet（DS-SDPNet）技术，结合时间聚合的空间结构和逐帧信息，以实现更有效的身份表示学习。

**📊 数据集**

使用了mm-ADL数据集，该数据集由11名受试者在受控协议下执行七种日常活动收集而成，共包含3850个片段和77000帧点云数据。

**📈 对比分析**

与单一模型、联合多任务学习和端到端混合专家模型进行了比较，结果显示，活动条件专家的识别准确率从62.1%提高到68.0%，在两名受试者的重识别设置中，mAP从57.2%提高到75.4%，Rank-1准确率从59.1%提高到82.1%。

**⚠️ 局限性**

限制在于本研究仅在受控条件下评估识别的可行性，mm-ADL数据集仅限于11名受试者，且活动的多样性和连续性未能充分反映真实的室内活动。此外，当前的组件消融实验未能分离身体几何信息和运动相关信息的贡献。

---

## 573. Eliciting Weak-to-Strong Generalization with On-Policy Reverse Distillation

**arXiv ID:** 2609.08798 | [PDF](https://arxiv.org/pdf/2609.08798v1)

**作者:** Youngrok Park `[一作]`, Se-Young Yun `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的知识蒸馏方法，称为On-Policy Reverse Distillation (OPRD)，旨在从较弱的教师模型向较强的学生模型转移后训练的知识，而不将教师模型的策略作为优化目标。

**💡 创新点**

OPRD通过放大学生的策略梯度中与教师模型的政策变化对齐的部分，加速学生的优化过程，从而实现了弱到强的泛化，克服了传统方法的限制。

**🔧 技术方法**

使用了强化学习与可验证奖励（RLVR）和在学生生成的响应上进行的在线策略蒸馏（OPD）技术。

**📊 数据集**

在数学推理和逻辑推理任务上进行了评估，使用了DAPO-Math-17K数据集和Reasoning Gym基准。

**📈 对比分析**

与传统的OPD和其他基线方法（如GRPO和Mix-RL）相比，OPRD在更新次数上显著减少，且在性能上超越了这些方法，显示出更高的样本效率。

**⚠️ 局限性**

OPRD在某些情况下可能会受到教师模型能力限制的影响，尤其是在教师模型的反馈不足时，可能导致学习效果不佳。

---

## 574. Interpretable Hyperspectral Unmixing Framework with Fixed Endmember Prior and Structured Residual Refinement

**arXiv ID:** 2609.08786 | [PDF](https://arxiv.org/pdf/2609.08786v1)

**作者:** Ziyi Guan `[一作]` (Xiangtan University), Qian Liu `[通讯]` (Xiangtan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可解释的阶段性高光谱解混框架（I-HyperSU），在固定的端元先验下进行高光谱解混，明确分离了丰度估计和结构残差建模。

**💡 创新点**

创新点在于将丰度估计与结构残差建模分开，采用软约束的丰度估计方法，显著提高了解混的稳定性和准确性。

**🔧 技术方法**

使用了FISTA算法进行丰度估计，并结合低秩SVD结构正则化和轻量级深度图像先验（DIP）进行残差精炼。

**📊 数据集**

在Samson、Urban和Jasper Ridge数据集上进行了实验，使用固定的N-FINDR端元先验。

**📈 对比分析**

与传统方法（如UCLS）比较，I-HyperSU在重建误差上减少了61.7%至69.5%，而丰度RMSE几乎保持不变，表明残差精炼有效捕捉了结构性模型不匹配。

**⚠️ 局限性**

限制在于该研究主要集中在三个标准数据集和固定的端元配置，未来工作将扩展到变异感知基线和端到端深度方法。

---

## 575. Hybrid Continuous DoA Estimation with Shared-Radius Co-Prime Circular Arrays

**arXiv ID:** 2609.08827 | [PDF](https://arxiv.org/pdf/2609.08827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 576. Kairos: A Dataset for Fine-Grained Video-Language Modeling over Space, Time, and Dynamics

**arXiv ID:** 2609.08755 | [PDF](https://arxiv.org/pdf/2609.08755v1)

**作者:** Ruibo Ming `[一作]` (Sofia University St. Kliment Ohridski), Jinjin Gu `[通讯]` (Sofia University St. Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的长视频语言建模数据集，包含时间解析的注释，旨在支持对视频内容的细粒度理解和推理。

**💡 创新点**

创新点在于引入了时间解析的注释流，能够捕捉视频中的空间、时间和动态变化，提供了更高的注释密度和信息丰富性。

**🔧 技术方法**

使用了自动化注释管道，结合多模态信号（如语音和环境音频）进行视频内容的细粒度描述。

**📊 数据集**

数据集包含来自YouTube和Bilibili的长视频，时长从十分钟到半小时不等，涵盖多种场景类型。

**📈 对比分析**

通过与现有视频数据集的比较，展示了在注释密度和信息丰富性方面的显著提升，且在视频语言模型的性能评估中表现出色。

**⚠️ 局限性**

限制在于现有数据集的稀疏注释和粗略的时间对齐，可能影响模型对动态视频结构的深入理解。

---

## 577. When Can One Obtain Certificates of Optimality Using Positivstellensaetze?

**arXiv ID:** 2609.08736 | [PDF](https://arxiv.org/pdf/2609.08736v1)

**作者:** Nayoon Kim `[一作]` (Czech Technical University in Prague), Jakub Marecek `[通讯]` (Czech Technical University in Prague)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了学习问题的正性和最优性证书，这些问题的目标和约束不一定是多项式的。

**💡 创新点**

提出了一种公理化框架，分离了目标函数和约束函数的角色，并证明了相应的定理。

**🔧 技术方法**

使用了Fischer的构造性严格和弱Positivstellensätze的公理化形式。

**📊 数据集**

使用了连续和可定义的函数代数的实例，包括不闭合平方根的有序域。

**📈 对比分析**

通过与现有的分支界限和切平面证明系统进行比较，展示了所提出方法的复杂性和性能。

**⚠️ 局限性**

限制在于当前的公理未覆盖经典Positivstellensätze的中心设置，且尚不清楚是否存在满足特定条件的代数。

---

## 578. Graph-based automata

**arXiv ID:** 2609.08843 | [PDF](https://arxiv.org/pdf/2609.08843v1)

**作者:** Cyril Pujol `[一作]` `[通讯]`, Cyril Pujol

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了基于图的自动机，这些自动机是通过从边着色或定向图中获取的非确定性有限自动机，所有顶点既是初始状态也是接受状态，所有边都是一对相反的转移。

**💡 创新点**

提出了图和树的语言的独特最小图和同态最小图的存在性，并通过对偶性方法进行了证明。

**🔧 技术方法**

使用了图论和自动机理论的工具，特别是通过同态和可逆自动机的关系进行分析。

**📊 数据集**

使用了边着色和定向图的语言，特别是通过构造图的自动机来研究这些语言。

**📈 对比分析**

通过与可逆自动机的比较，展示了基于图的语言的分解，并证明了其性能与可逆语言之间的关系。

**⚠️ 局限性**

限制在于基于图的自动机的最小性问题，尤其是没有唯一的最小图，且在某些情况下，图的结构可能导致复杂性。

---

## 579. Graph-Based Safe Reinforcement Learning for Multi-Agent Systems with Time-Varying Topology

**arXiv ID:** 2609.08802 | [PDF](https://arxiv.org/pdf/2609.08802v1)

**作者:** Xiao Sizhe `[一作]` (Beijing Jiaotong University), Tan Xin `[通讯]` (Beijing Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图的安全多智能体强化学习框架，用于具有时变拓扑的协作导航。

**💡 创新点**

创新点在于引入了控制障碍类似函数（CBLF）进行安全解耦的动作筛选机制，并结合了基于注意力的演员和图注意力网络（GAT）集中评论者。

**🔧 技术方法**

使用了基于图的协作学习架构，结合了注意力机制和GAT，确保了在动态拓扑下的安全和稳定性。

**📊 数据集**

在真实的差动驱动机器人平台上进行了验证，使用了LiDAR传感器获取环境数据。

**📈 对比分析**

与现有的多智能体强化学习算法（如MASAC、MATD3和MADDPG）进行比较，结果显示该方法在成功率和碰撞率上表现优越，且在训练效率和最终导航性能上显著优于基线。

**⚠️ 局限性**

局限性在于在高密度智能体环境中，成功率会下降，主要是由于目标区域的空间拥挤导致的任务超时。

---

## 580. Improving Term Evaluation in Machine Translation: Variation Matters

**arXiv ID:** 2609.08779 | [PDF](https://arxiv.org/pdf/2609.08779v1)

**作者:** Nicolas Dahan `[一作]` (Inria), Rachel Bawden `[通讯]` (Inria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究探讨了机器翻译（MT）中术语评估的变异性，提出了一种新的评估框架，考虑了术语的多样性，尤其是在英语-法语科学翻译的文档级评估中。

**💡 创新点**

创新点在于引入了交叉术语变异性度量，评估翻译中术语变异关系的保留情况，并提出了一种变异感知的评估方法，强调目标侧变异应与源侧变异相对应。

**🔧 技术方法**

使用了基于词汇表的准确性、翻译一致性和交叉术语变异性等多种评估技术，并结合了依赖于上下文的变异检测方法。

**📊 数据集**

研究使用了两个英语-法语的平行语料库，分别来自自然语言处理（NLP）领域，包含32篇和10篇论文的翻译。

**📈 对比分析**

与现有方法相比，研究发现人类翻译者在目标侧引入的词汇变异性高于MT系统，且不同的翻译一致性度量会导致系统排名的差异，表明现有一致性评分对术语变异的处理不够敏感。

**⚠️ 局限性**

本研究的局限性在于仅针对英语-法语语言对和NLP领域的特定语料进行分析，变异转移模式可能在其他语言对和领域中有所不同，且未测试神经或无参考度量如何处理术语变异。

---

## 581. A Controlled Comparison of Manual and Teleoperated Intraocular Instrument Motion for an Input Device

**arXiv ID:** 2609.08770 | [PDF](https://arxiv.org/pdf/2609.08770v1)

**作者:** Korab Hoxha `[一作]` (Technical University of Munich), M. Ali Nasseri `[通讯]` (University of Alberta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究比较了手动和遥控的眼内手术器械运动，保持了手术工具、眼模型和跟踪源的一致性，仅改变了控制接口，以评估其对外科手术技术的保留程度。

**💡 创新点**

创新点在于提供了一种新的输入设备EMIRAS，旨在保留外科医生的手动技术，同时通过对比手动和遥控操作的运动行为，揭示了两者在执行上的显著差异。

**🔧 技术方法**

使用了三自由度的输入设备EMIRAS，该设备通过相位特定的模式多路复用三种输入到五个关节的机器人上。

**📊 数据集**

使用了商业眼科模拟器进行实验，参与者在模拟器上执行导航任务，所有参与者在手动和遥控条件下使用相同的工具和眼模型。

**📈 对比分析**

与手动操作相比，遥控试验的完成时间是手动操作的三倍，速度仅为手动操作的四分之一，运动范围不到手动操作的一半，且分段次数是手动操作的3.5倍。尽管任务结果相同，但执行方式显著不同。

**⚠️ 局限性**

本研究的局限性在于实验是在模拟器中进行的，缺乏真实手术中的组织顺应性和错误后果的影响，同时没有使用主观工作负荷量表来评估操作的感知努力。

---

## 582. PDMR: Passage-Driven Multi-ID Document Retrieval

**arXiv ID:** 2609.08762 | [PDF](https://arxiv.org/pdf/2609.08762v1)

**作者:** Smail Oussaidene `[一作]` (Institute for Research in Computer Science), Mohand Boughanem `[通讯]` (Institute for Research in Computer Science)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的生成检索框架PDMR，通过多个段落级标识符表示文档，解决了传统生成检索中单一标识符的局限性。

**💡 创新点**

创新点在于将文档表示为多个段落级标识符，允许模型通过多个语义入口点进行检索，从而提高了对查询变体的鲁棒性。

**🔧 技术方法**

使用了多目标学习和加权损失函数来处理段落级标识符的监督模糊性，并采用了基于LLM的段落提取和标识符构建流程。

**📊 数据集**

在NQ320K和MS MARCO Document数据集上进行了评估，NQ320K包含109,739个文档和7,830个查询，MS MARCO Document包含323,569个文档和5,187个查询。

**📈 对比分析**

PDMR在NQ320K上在Recall@1和MRR@100上超越了强基线，在MS MARCO Document上在Recall@1和MRR@10上表现最佳，且在Recall@10上保持竞争力。

**⚠️ 局限性**

限制在于PDMR未能达到NQ320K的最先进性能，且仍依赖于自回归解码，可能对早期修剪和标识符模糊性敏感。

---

## 583. PAC-Bayesian Bounds for Learning Partially Observed Stochastic Linear Time-Invariant State-Space Systems with Inputs and Sub-Gaussian Noise

**arXiv ID:** 2609.08740 | [PDF](https://arxiv.org/pdf/2609.08740v1)

**作者:** Mihaly Petreczky `[一作]` (Université de Lille), John Leth `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文推导了部分观测线性时不变（LTI）随机动态系统的PAC-Bayesian误差界限，特别是在状态空间形式下，考虑了输入的影响。

**💡 创新点**

创新点在于为具有部分观测和子高斯噪声的离散时间随机LTI系统提供了PAC-Bayesian误差界限，并且这些界限可以用于推导参数估计误差的界限。

**🔧 技术方法**

使用了PAC-Bayesian理论，结合Kullback-Leibler散度来推导误差界限，并考虑了不同类型的损失函数（如Lipschitz损失和二次损失）。

**📊 数据集**

使用了随机LTI系统生成的数据集，假设数据由稳定的LTI系统生成，并且包含子高斯噪声。

**📈 对比分析**

与现有文献相比，本文的界限在处理部分观测状态和输入的状态空间模型时更为一般化，提供了对多种学习算法的参数估计和泛化误差的界限，性能上优于特定的最小二乘学习算法。

**⚠️ 局限性**

限制在于现有结果假设模型是无噪声的，且在处理无限预测时可能存在一定的理论局限性。

---

## 584. Application of curiosity driven exploration methods for hardware interference identification

**arXiv ID:** 2609.08729 | [PDF](https://arxiv.org/pdf/2609.08729v1)

**作者:** Ludovic Matar `[一作]` (National Institute for Research in Digital Science and Technology), Pierre-Yves Oudeyer `[通讯]` (National Institute for Research in Digital Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于好奇心驱动的探索算法，用于多核处理器系统中的干扰分析，旨在系统性地覆盖可能的干扰行为空间。

**💡 创新点**

创新点在于首次将好奇心驱动的探索算法应用于多核处理器系统的干扰模式发现，提供了一种新的方法来识别复杂系统中的干扰源。

**🔧 技术方法**

使用了好奇心驱动的探索算法，这是一种自动化发现算法，能够高效覆盖复杂系统的行为空间。

**📊 数据集**

使用了一个双核处理器的模拟器环境进行实验，模拟了多核处理器的内存访问和干扰现象。

**📈 对比分析**

与传统的伪随机程序生成方法相比，所提出的方法在有限的实验预算内实现了更广泛和更均匀的行为覆盖，表现出更高的干扰发现能力。

**⚠️ 局限性**

限制在于所使用的模拟器模型简化了实际硬件架构的复杂性，可能无法完全捕捉真实系统中的所有干扰模式。

---

## 585. Enhancing Table Structure Recognition via Bounding Box Guidance

**arXiv ID:** 2609.08705 | [PDF](https://arxiv.org/pdf/2609.08705v1)

**作者:** Lei Hu `[一作]` (South China University of Technology), Shuangping Huang `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的表格结构识别框架BGTR，通过边界框引导生成HTML序列，以提高复杂场景下的表格识别准确性。

**💡 创新点**

创新点在于显式利用边界框信息来指导HTML序列的生成，解决了现有方法在复杂场景下的错误预测问题。

**🔧 技术方法**

使用了图像编码器、共享解码器和边界框引导结构解码器等技术，采用自回归解码方法和逐步训练策略。

**📊 数据集**

引入了SNSTab，一个合成生成的自然场景表格数据集，包含500k张表格图像，此外还在五个基准数据集上进行了实验。

**📈 对比分析**

与现有方法相比，BGTR在五个公共基准数据集上表现出色，特别是在复杂表格场景中，显著提高了TEDS-S得分，展示了其优越性。

**⚠️ 局限性**

限制在于自然场景表格的数据量较小，尽管采用了逐步训练方法，但仍可能影响模型的训练效果。

---

## 586. The Weight Spectrum of the Affine Grassmann Code $C^{\mathbb A}(3,6)$

**arXiv ID:** 2609.08784 | [PDF](https://arxiv.org/pdf/2609.08784v1)

**作者:** Prasant Singh `[一作]` (Indian Institute of Technology Jammu), Rohit Yadav `[通讯]` (Indian Institute of Technology Jammu)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了从仿射开集Å^9获得的仿射Grassmann码C^Å(3,6)，通过将码字表示为通用3×3矩阵的所有大小的次要的线性组合，并根据出现非零系数的次要的最大大小对其进行分类，确定了C^Å(3,6)的所有可能的汉明权重及其对应的码字数量。

**💡 创新点**

创新点在于通过对码字的分类，成功计算出C^Å(3,6)的完整权重谱，这是一个在编码理论中具有挑战性的任务，尤其是在仿射设置下，直接的Grassmann码方法无法应用。

**🔧 技术方法**

使用了线性代数和代数几何中的技术，特别是通过对3×3矩阵的次要的分类和分析来实现。

**📊 数据集**

使用了仿射Grassmann码C^Å(3,6)的相关数据集，具体为从有限域中提取的3×3矩阵的次要。

**📈 对比分析**

与现有方法的比较显示，本文的方法能够有效地分类和计算汉明权重，性能上优于传统的Grassmann码方法，尤其是在处理仿射Grassmann码时。

**⚠️ 局限性**

限制在于目前的研究主要集中在特定的仿射Grassmann码C^Å(3,6)上，其他更一般的情况仍然是开放的研究问题，且权重谱的计算在更高维度的情况下可能会变得更加复杂。

---

## 587. Exploring the Genesis Platform Capabilities to Accelerate Scientific Discovery in OPAL

**arXiv ID:** 2609.08844 | [PDF](https://arxiv.org/pdf/2609.08844v1)

**作者:** Daniel Rosendo `[一作]`, Rafael Ferreira da Silva `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文是IEEE会议论文的演示文件，旨在为使用IEEEtran.cls的论文提供一个起始模板。

**💡 创新点**

创新点在于提供了一个标准化的格式和结构，帮助作者更好地组织和撰写会议论文。

**🔧 技术方法**

使用了IEEEtran.cls文档类。

**📊 数据集**

未提及具体数据集。

**📈 对比分析**

未提供比较的方法和性能评估。

**⚠️ 局限性**

该论文缺乏具体的研究内容和实验结果，主要是一个模板示例。

---

## 588. Last-Iterate Convergence of Policy Dynamics in Zero-Sum Networked Separable Markov Games

**arXiv ID:** 2609.08823 | [PDF](https://arxiv.org/pdf/2609.08823v1)

**作者:** Zailin Ma `[一作]` `[通讯]` (Peking University), Zailin Ma (Peking University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的算法ER-OMWU，用于解决有限时域零和网络可分Markov游戏中的纳什均衡问题。

**💡 创新点**

创新点在于设计了一种单循环的策略动态更新方法，能够在不重复求解每个状态阶段的均衡问题的情况下，快速收敛到近似纳什均衡。

**🔧 技术方法**

使用了熵正则化的乐观乘法权重更新（ER-OMWU）技术。

**📊 数据集**

在有限时域零和网络可分Markov游戏上进行了评估，具体数据集未详细说明。

**📈 对比分析**

与基于值迭代的基线方法进行比较，ER-OMWU在相同的更新预算下达到了可比的终极纳什均衡差距，并且在熵温度降低时，收敛迭代遵循近线性1/ϵ的趋势。

**⚠️ 局限性**

限制在于该算法假设对奖励和转移的完全信息，且在无限时域的情况下计算均衡是计算上困难的。

---

## 589. What AI Benchmarks Actually Measure: Adapting Convergent and Discriminant Validity to Interrogate Fifty-Six AI Benchmarks

**arXiv ID:** 2609.08812 | [PDF](https://arxiv.org/pdf/2609.08812v1)

**作者:** Meera Desai `[一作]` (University of Michigan), Angelina Wang `[通讯]` (Cornell Tech)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究适应了社会科学中的收敛效度和区分效度的视角，分析了56个能力和安全基准与53个模型的关系，探讨这些基准是否真正测量了它们声称的概念。

**💡 创新点**

创新点在于将收敛和区分效度的框架应用于AI基准的评估，揭示了基准之间的相关性和潜在的有效性问题。

**🔧 技术方法**

使用了项目反应理论（IRT）模型来分析模型在基准上的得分和排名。

**📊 数据集**

使用了来自53个模型在56个能力和安全基准上的输出和得分数据集。

**📈 对比分析**

通过Spearman相关性分析比较模型在相同和不同概念的基准上的排名，发现安全基准之间的相关性通常较弱，而能力基准之间的相关性则不如预期的那样明显，表明这些基准可能无法有效区分不同的能力概念。

**⚠️ 局限性**

限制在于基准所声称测量的概念往往不够明确，导致难以判断有效性问题是测量失败还是概念不一致。

---

## 590. Evidence-Grounded Retrieval for Investigation Hunt Lead Generation from CTI Reports

**arXiv ID:** 2609.08790 | [PDF](https://arxiv.org/pdf/2609.08790v1)

**作者:** Akash Prakash `[一作]` (Concordia University), Mourad Debbabi `[通讯]` (Concordia University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自动化系统，从网络威胁情报报告中提取可操作的猎杀线索，旨在将非结构化知识转化为可调查的假设。

**💡 创新点**

创新点在于结合了密集向量检索与知识图谱的多跳遍历，采用本体约束的检索增强生成方法，并提供了一个与大型语言模型无关的框架，生成结构化的可操作线索。

**🔧 技术方法**

使用了混合检索技术，结合了密集向量检索和知识图谱推理，以及基于本体的检索增强生成方法。

**📊 数据集**

使用了公共的网络威胁情报报告数据集，特别是针对知名APT（高级持续性威胁）组织的报告。

**📈 对比分析**

与现有的基线方法相比，混合证据检索与本体约束的组合使得F1分数提高了约2倍（从0.44提升至0.85），并且在有效性评分上达到了最高（约86.95%）。

**⚠️ 局限性**

局限性包括：评估仅基于四个报告和单一威胁行为者（APT41），系统本体的规模较小，且未在实际安全运营中心中进行部署，操作可扩展性和分析师的可用性尚待验证。

---

## 591. Adaptive Anisotropic Attention for Axis-Structured Signals

**arXiv ID:** 2609.08788 | [PDF](https://arxiv.org/pdf/2609.08788v1)

**作者:** Mahir Jain `[一作]` (Mannas AI), Siddharth Panwar `[通讯]` (Mannas AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种新的自适应各向异性注意力机制（AAA），用于处理EEG等结构化信号，改进了注意力机制的设计。

**💡 创新点**

创新点在于将注意力分为时间路径和空间路径，并通过一个小门控机制动态调整两者的组合，从而更好地捕捉信号的结构特征。

**🔧 技术方法**

使用了自适应各向异性注意力（AAA）技术，结合了时间和空间的注意力机制。

**📊 数据集**

使用了多个EEG数据集进行评估，包括TUH-EEG、I-CARE等，涵盖了六个EEG下游任务。

**📈 对比分析**

与密集基线模型相比，AXON模型在六个EEG任务上均表现出更高的均衡准确率，尤其在线性探测和完全微调下，均优于密集模型和其他对比模型。

**⚠️ 局限性**

限制在于模型的复杂性和计算成本，可能在处理更大规模的数据时面临挑战。

---

## 592. AXS-Net: Interpretable Deep Unfolding for Hyperspectral Image Denoising via Spectral Basis Unmixing and Structured Noise Refinement

**arXiv ID:** 2609.08777 | [PDF](https://arxiv.org/pdf/2609.08777v1)

**作者:** Ziyi Guan `[一作]` (Xiangtan University), Zheng Yang `[通讯]` (Xiangtan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为AXS-Net的可解释深度展开网络，用于高光谱图像的去噪，模型将去噪过程视为低秩光谱子空间重建、结构稀疏噪声和残余高斯噪声的组合。

**💡 创新点**

创新点在于将高光谱图像的去噪建模为低秩加稀疏的分解，并通过可解释的K阶段网络分离出清晰成分和结构噪声估计，而不是简单回归单一的清晰图像。

**🔧 技术方法**

使用了深度展开方法，结合了分析光谱基更新、SSX-Block和SBlock等学习的近端操作符。

**📊 数据集**

使用了ICVL、CAVE和Harvard数据集，涵盖了高斯噪声、条纹噪声、死线噪声、脉冲噪声和混合噪声等五种噪声配置。

**📈 对比分析**

与多种最新学习型高光谱图像去噪方法进行了比较，AXS-Net在所有五种噪声配置下在ICVL数据集上表现最佳，并在CAVE和Harvard数据集上实现了有效的零-shot迁移，性能优于其他方法。

**⚠️ 局限性**

主要局限性在于依赖于合成监督，评估仅限于合成噪声数据，且在强烈偏移的CAVE条纹和混合情况下MPSNR转移较弱；未来工作将集中在无监督的结构噪声估计和真实噪声高光谱图像的验证上。

---

## 593. It's All in the Way You Say It: The Role of Information Representation in LLM-Based Glycemic-Event Prediction

**arXiv ID:** 2609.08772 | [PDF](https://arxiv.org/pdf/2609.08772v1)

**作者:** Andrea Apicella `[一作]` (University of Naples Federico II), Roberto Prevete `[通讯]` (University of Naples Parthenope)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究探讨了基于提示的通用大型语言模型（LLMs）在1型糖尿病患者的餐后高血糖和低血糖预测中的有效性，使用了OhioT1DM数据集进行评估。

**💡 创新点**

创新点在于研究了生理信息的不同表示方式对LLMs预测能力的影响，并比较了基于提示的LLMs与传统的患者特定监督模型及Gluco-LLM的性能。

**🔧 技术方法**

使用了通用大型语言模型进行零-shot和少-shot推理，分析了原始、生物结构化和叙述性三种不同的生理信息表示方式。

**📊 数据集**

使用了OhioT1DM数据集，该数据集包含12名患者的连续血糖监测数据及相关的胰岛素治疗、饮食和身体活动信息。

**📈 对比分析**

与传统的患者特定监督模型相比，传统模型在高血糖预测中表现最佳，而基于提示的LLMs在低血糖预测中表现更好。提示的推理效果受生理信息表示方式的强烈影响，提供额外的上下文信息并未系统性改善性能。

**⚠️ 局限性**

本研究的局限性在于仅使用了OhioT1DM数据集，且未考虑更大规模或不同类型的语言模型的表现。此外，提示推理的变异性未被充分评估，且所研究的表示方式仅覆盖了生理信息呈现的有限子集。

---

## 594. Benchmark Scores Are Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks

**arXiv ID:** 2609.08765 | [PDF](https://arxiv.org/pdf/2609.08765v1)

**作者:** Aymene Berriche `[一作]` (Qatar Computing Research Institute), Yazan Boshmaf `[通讯]` (Qatar Computing Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文审计了八个网络安全基准，分析了10个不同类型的大型语言模型（LLM）在这些基准上的表现，探讨了评估管道设计选择对模型评分的影响。

**💡 创新点**

创新点在于将基准视为测量管道，识别出15种系统性失败模式，并展示了单一管道选择可以导致模型评分变化超过80个百分点，从而显著改变模型排名。

**🔧 技术方法**

使用了控制扰动的技术来量化管道配置字段的影响，并实现了一个标准化的评估工具，以便在不同基准之间进行比较。

**📊 数据集**

使用了八个网络安全基准，涵盖了48,662个问题和23个任务，这些基准包括多种网络安全评估任务，如知识问答、漏洞评分和攻击者归属等。

**📈 对比分析**

通过标准化评估管道，发现10个模型中有9个在至少一个基准上排名变化至少三位，表明评估管道选择对模型排名有显著影响。

**⚠️ 局限性**

限制在于本研究的发现主要针对网络安全领域，其他领域的基准和模型可能会表现出不同的失败模式。此外，某些基准的文档不够明确，导致需要做出实施选择。

---

## 595. A Note on Scaling in Randomly Rotated Quantization and Its Connection to the CDEF +1 Pythagorean Relation

**arXiv ID:** 2609.08759 | [PDF](https://arxiv.org/pdf/2609.08759v1)

**作者:** Uri Erez `[一作]` `[通讯]` (Tel Aviv University), Uri Erez (Tel Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了基于随机旋转的量化方案，特别是MMSE和无偏重建缩放的作用，并指出其与经典统计信号处理和通信理论的联系。

**💡 创新点**

创新点在于将EDEN工作中的两种重建缩放与经典CDEF理论中的维纳和无偏系数进行自然解释，并证明EDEN在每个有限维度下保证了精确的条件无偏性。

**🔧 技术方法**

使用了随机旋转和Haar旋转等技术，结合了经典的统计信号处理方法。

**📊 数据集**

未具体提及使用的数据集，但提到的量化方案适用于LLM量化和相关源的分布式均值估计。

**📈 对比分析**

通过与经典CDEF理论的比较，证明了EDEN的重建缩放在每个旋转实现下都保持无偏性，且在维度增大时，EDEN的缩放接近于经典标量高斯模型的缩放。

**⚠️ 局限性**

限制在于EDEN假设重建缩放在没有量化误差的情况下表示，且未探讨在更复杂的量化场景中的表现。

---

## 596. FOCI Policy: Focus on Object-Centric Interactions for Relational Manipulation Policies

**arXiv ID:** 2609.08743 | [PDF](https://arxiv.org/pdf/2609.08743v1)

**作者:** Ze Fu `[一作]` (KU Leuven), Renaud Detry `[通讯]` (KU Leuven)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为Foci Policy的交互中心框架，通过建模对象之间的相对运动来改进机器人操作策略，从而提高泛化能力。

**💡 创新点**

创新点在于自动提取紧凑的交互片段，并将技能表示为任务相关对象之间的相对SE(3)运动，提供了对场景配置和机器人形态的不变性。

**🔧 技术方法**

使用了变换点检测（change-point detection）技术来识别交互片段，并通过两阶段的政策架构来建模操作。

**📊 数据集**

在RLBench、COLOSSEUM和真实世界任务上进行了实验，使用了这些数据集来验证Foci Policy的有效性。

**📈 对比分析**

与现有的对象中心和动作中心政策相比，Foci Policy在使用单一演示和单一相机的情况下，表现出更强的性能和更少的训练数据需求，尤其在精确度要求高的任务中表现突出。

**⚠️ 局限性**

局限性在于该框架假设任务相关的交互阶段是时间上可分离和几何上可识别的，这在某些任务中可能不适用。此外，依赖于预训练的对象分割和姿态估计模块，可能在对象几何模糊或观察不清时导致性能下降。

---

## 597. BAFF: Bid-Aware Filter Family for Mitigating Training Data Interference in RTB A/B Tests

**arXiv ID:** 2609.08725 | [PDF](https://arxiv.org/pdf/2609.08725v1)

**作者:** Jeonglyul Oh `[一作]` (Dable Inc.), Youngjae Kim `[通讯]` (Dable Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为Bid-Aware Filter Family (BAFF)的过滤器家族，用于减轻实时竞价（RTB）A/B测试中的训练数据干扰。

**💡 创新点**

创新点在于将训练数据干扰分解为广告选择和出价价格不一致两个可观察的通道，并通过(k,l)参数化的硬过滤器提供了一种结构化的搜索空间，以找到最小偏差的数据共享策略。

**🔧 技术方法**

使用了(k,l)参数化的硬过滤器和三阶段在线测量协议。

**📊 数据集**

在模拟和实际RTB部署中进行了验证，使用的训练数据集包括控制模型和处理模型的日志。

**📈 对比分析**

通过三阶段在线测量协议比较了不同的数据共享策略，结果显示BAFF在保持商业指标（如CPC和CTR）方面优于传统的日志共享和日志分割方法。

**⚠️ 局限性**

局限性包括理论分析基于岭回归的代理模型，无法直接扩展到非凸的有限样本深度学习模型；在线实验仅覆盖单一广告商-供应方平台对，外部有效性需未来在不同广告商和平台上复制。

---

## 598. Inverse Digital Marbling: Recovering Gesture Programs with a Replay Adjoint

**arXiv ID:** 2609.08722 | [PDF](https://arxiv.org/pdf/2609.08722v1)

**作者:** Tianqi Liu `[一作]` (Zelostech), Hang Liu `[通讯]` (Communication University of China)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于沉积的数字 marbling 模型，通过优化胶囊插入的有序程序来重建目标图像。

**💡 创新点**

创新点在于提出了一种胶囊插入原语，能够精确保持面积，并具有封闭形式的逆映射，同时开发了一个重放伴随机制来重建中间状态。

**🔧 技术方法**

使用了 PyTorch 实现的重放机制，结合了可微分模拟和逆向传播技术。

**📊 数据集**

使用了五个不同的 marbled 纸张作为数据集进行图像重建评估。

**📈 对比分析**

与禁用传输的拟合方法、一次性几何补偿和已发布的基于笔画的拟合器进行了比较，结果显示在相似的笔画数量下，本文的方法在内存使用和性能上均优于其他方法。

**⚠️ 局限性**

限制在于模型是几何近似，未考虑流体动力学的复杂性，如粘度对比、指纹效应和扩散等，且评估仅基于五个样本，缺乏更广泛的验证。

---

## 599. GoAnt: Quality-Diversity Multi-Agent Search for Alpha Factor Discovery in Market Microstructure Data

**arXiv ID:** 2609.08719 | [PDF](https://arxiv.org/pdf/2609.08719v1)

**作者:** Stella Zhao `[一作]` (University of Minnesota), Tommy Sha `[通讯]` (Stony Brook University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为GoAnt的动态多智能体系统，用于从市场微观结构数据中自动发现alpha因子，旨在解决现有系统的执行陷阱和推理陷阱问题。

**💡 创新点**

创新点在于GoAnt结合了适应性容量的MAP-Elites心理地图与解耦的生成器，采用非通信的工作蚂蚁（Explorer、Exploiter和Connector）来增强多样性，并通过一个精简的女王蚂蚁进行预算分配。

**🔧 技术方法**

使用了动态多智能体架构，结合了MAP-Elites心理地图和知识蒸馏的女王蚂蚁进行预算分配。

**📊 数据集**

使用了2023年至2026年的真实A股微观结构数据，分为价格-成交量（PV）面板和订单簿（L2）快照两类。

**📈 对比分析**

与静态协调的基线方法（如单智能体贪婪算法、辩论/投票机制等）进行比较，GoAnt在价格-成交量和订单簿设置下的质量加权收益分别为41.8和47.6，分别提高了57%和97%。

**⚠️ 局限性**

限制在于GoAnt的动态结构可能在某些情况下导致计算开销增加，且在特定市场条件下的表现可能不如静态方法。

---

## 600. Chimaera: A Mixture-of-Graph-Experts Architecture for Cross-Task and Cross-Dataset Graph Learning

**arXiv ID:** 2609.08709 | [PDF](https://arxiv.org/pdf/2609.08709v1)

**作者:** Jonathan Frank `[一作]` (University of Ulm), Ansgar Scherp `[通讯]` (University of Ulm)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为Chimaera的图学习模型，该模型结合了混合专家（MoE）架构与图基础模型（GFM），以支持跨任务和跨数据集的图学习。

**💡 创新点**

Chimaera的创新点在于将不同的GFM架构（如图提示和线性GNN模型）与混合专家架构相结合，扩展了线性GNN的应用范围，支持节点、链接和图级任务的分类。

**🔧 技术方法**

使用了混合专家架构、图基础模型（GFM）和大语言模型（LLM）来生成嵌入，并通过门控函数结合不同的专家模型。

**📊 数据集**

使用了六个基准文本属性图数据集进行实验，包括Cora、PubMed、WikiCS、WN18RR、BBBP和BACE。

**📈 对比分析**

通过同任务和跨任务的实验比较了Chimaera与其他方法的性能，结果显示Chimaera在大多数情况下表现出强大的泛化能力，尤其是在跨任务和预训练场景中，其性能接近于直接在目标数据集上训练的模型。

**⚠️ 局限性**

限制在于模型仅针对单一任务类型进行训练，导致在链接级任务上表现不佳；同时，超参数优化的时间成本较高，且未能同时训练所有组件。

---

## 601. Measuring the Security of the Evolving Software Supply Chain: a Research Agenda

**arXiv ID:** 2609.08810 | [PDF](https://arxiv.org/pdf/2609.08810v1)

**作者:** Sarah Meriem Ourari `[一作]` `[通讯]` (Budapest University of Technology and Economics), Sarah Meriem Ourari (Budapest University of Technology and Economics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一项研究计划，旨在系统化软件供应链安全的测量和建模，特别关注依赖关系和漏洞传播分析的现状与不足。

**💡 创新点**

创新点在于提出一个统一的测量框架，以便在不同软件生态系统中一致地表示和分析依赖结构，并考虑AI辅助软件开发带来的新挑战。

**🔧 技术方法**

使用了系统化文献综述方法，结合了文献分析、依赖图表示和源代码级可达性分析等技术。

**📊 数据集**

数据集来源于多个软件生态系统，包括npm、PyPI和Maven等包管理器，以及GitHub和Bitbucket等源代码库。

**📈 对比分析**

通过文献综述和实证分析，比较了现有的SCA工具和方法，发现它们在处理复杂依赖关系时存在局限性，导致误报和准确性降低。提出的框架旨在减少警报疲劳，提高漏洞检测的准确性。

**⚠️ 局限性**

限制在于现有方法大多针对特定生态系统，缺乏通用性，可能无法直接转移到其他环境中，且对新兴的AI生成代码的依赖模式的捕捉仍需进一步研究。

---

## 602. ArmPoser: Real-Time, Calibration-Free Arm Pose Estimation from Smartwatch IMU

**arXiv ID:** 2609.08806 | [PDF](https://arxiv.org/pdf/2609.08806v1)

**作者:** Bishnu Dev `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Karan Ahuja `[通讯]` (Northwestern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为ArmPoser的手臂姿态估计系统，该系统使用单个智能手表的IMU进行实时姿态估计，无需用户校准。

**💡 创新点**

创新点在于直接在消费者智能手表的重力对齐本地框架中进行训练，消除了传统方法中需要的全局框架对齐、传感器到骨骼的偏移估计和每次会话的校准。

**🔧 技术方法**

使用了基于LSTM的深度学习模型来处理IMU数据，并通过数据合成技术增强了模型的鲁棒性。

**📊 数据集**

使用了公共基准数据集（如DIP-IMU、TotalCapture和IMUPoser）以及一个新收集的ArmPoser数据集，该数据集包含10名参与者执行30种日常活动的原始和校准IMU数据。

**📈 对比分析**

与其他方法（如IMUPoser和MobilePoser）相比，ArmPoser在所有四个数据集上都实现了最低的关节位置误差和角度误差，且无需用户校准，表现出更好的实用性。

**⚠️ 局限性**

限制在于ArmPoser仅从单个智能手表估计手臂姿态，未来需要扩展到全身姿态估计，并且当前的延迟和准确性限制了其在某些应用中的使用。

---

## 603. Evaluation Principles for MRI-MRA Registration in Trigeminal Neuralgia: An ROI-Centered Neurovascular Benchmark

**arXiv ID:** 2609.08805 | [PDF](https://arxiv.org/pdf/2609.08805v1)

**作者:** Xupeng Zhang `[一作]` (Johns Hopkins University), Peirong Liu `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了一种针对三叉神经痛（TN）MRI-MRA注册的ROI中心神经血管评估基准，重点在于小范围内的神经血管可视化，而非传统的全脑注册。

**💡 创新点**

创新点在于将TN MRI-MRA融合视为一个特定的局部神经血管注册评估问题，并构建了一个包含149名患者的临床队列，提供了临床医生标注的双侧三叉神经ROI。

**🔧 技术方法**

使用了多种注册方法，包括ANTs Affine、ANTs SyN、ConvexAdam、FireANTs、EasyReg和SynthMorph，并结合了局部图像度量和基于分割的血管定位度量进行评估。

**📊 数据集**

数据集由149名TN患者的MRI和TOF-MRA图像组成，所有患者均接受了临床医生标注的三叉神经ROI，数据集包含298个双侧ROI。

**📈 对比分析**

通过局部图像相似性、血管背景可分离性和下游血管定位等多种度量进行比较，结果显示不同方法在局部可分离性和全局注册合理性上表现不一致，ANTs SyN在高对比度情况下表现最佳。

**⚠️ 局限性**

本研究的局限性包括其回顾性和单中心的设计，虽然包含了多种扫描仪和场强，但仍以西门子3T为主，且标注为部分血管段而非完整血管树，可能影响几何评估。

---

## 604. Real-time Puncture Detection and Recovery for Pneumatic Soft Actuators

**arXiv ID:** 2609.08804 | [PDF](https://arxiv.org/pdf/2609.08804v1)

**作者:** Tejonidhi R. Deshpande `[一作]` (Georgia Institute of Technology), Josiah Hester `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于单个惯性测量单元的软机器人气动驱动器的穿刺检测系统，能够实时识别穿刺位置和严重性。

**💡 创新点**

创新点在于使用单一的低功耗传感器（IMU）进行穿刺检测和严重性评估，同时引入了多腔气动软弯曲驱动器设计，增强了系统的可靠性和安全性。

**🔧 技术方法**

采用了基于全连接网络的自编码器进行穿刺定位，使用单层感知器进行严重性评分，结合了机器学习和非线性回归技术。

**📊 数据集**

使用了自定义数据集，包括不同腔体穿刺情况下的IMU数据，数据集包含三种不同的实验场景。

**📈 对比分析**

与其他方法（如一类支持向量机和孤立森林）相比，基于自编码器的模型在穿刺定位上达到了96.85%的准确率，且在不同数据集上表现出较强的鲁棒性。

**⚠️ 局限性**

局限性在于当前方法仅在使用Ecoflex 00-50材料的软驱动器上进行了验证，未来需要在不同材料和更复杂的场景下进行扩展和验证。

---

## 605. Ostrich: Taking Large Strides Through Stiff Contact in Differentiable Dynamics

**arXiv ID:** 2609.08800 | [PDF](https://arxiv.org/pdf/2609.08800v1)

**作者:** Aleš Kučera `[一作]` (Czech Technical University in Prague), Karel Zimmermann `[通讯]` (Czech Technical University in Prague)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Ostrich的GPU加速刚体模拟器，能够在较大时间步长下解决硬接触和摩擦问题，并通过隐函数定理对收敛残差进行微分。

**💡 创新点**

Ostrich在保持MuJoCo的仿真到真实精度的同时，能够使用比现有方法大50倍的时间步长，并且在优化过程中显著提高了速度和内存效率。

**🔧 技术方法**

使用了GPU加速的非光滑牛顿迭代方法，结合隐函数定理和前向Schur补充来计算梯度，确保每个时间步的梯度内存为O(1)。

**📊 数据集**

使用了一个自定义的三轮车机器人数据集，该数据集包含14次对一个木质托盘的真实世界穿越实验。

**📈 对比分析**

与MJX和牛顿半隐式方法进行比较，Ostrich在相同场景下的优化吞吐量高达29倍，且在随机目标控制合成任务中成功率为100%，而MJX和半隐式方法分别为16%和8%。

**⚠️ 局限性**

Ostrich的隐牛顿求解在小时间步长下的每步成本高于显式积分器，因此在小时间步长不是瓶颈时，其每次迭代的优势会减小。此外，当前不支持可变形物体，且所有真实机器人证据仅来自单一的轮式平台。

---

## 606. Compensating for Scarce Historical Images in Cross-Domain Cultural Heritage Retrieval Using Synthetic Aging

**arXiv ID:** 2609.08766 | [PDF](https://arxiv.org/pdf/2609.08766v1)

**作者:** Marcin Iwanowski `[一作]` (Nicolaus Copernicus University), Sabina Szymoniak `[通讯]` (Czestochowa University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究探讨了合成老化的现代图像是否可以替代或补充缺失的历史训练观察，以实现双向实例级检索。

**💡 创新点**

创新点在于提出了一种合成老化的图像生成方法，能够在缺乏历史图像的情况下，增强跨域检索的性能。

**🔧 技术方法**

使用了EfficientNetV2-M模型，并结合批量困难三元组损失进行训练。

**📊 数据集**

使用的数据集包含1077个对象身份，包含真实的历史图像和现代图像，以及合成的老域图像。

**📈 对比分析**

与真实历史图像的完全替换相比，合成图像的双向平均R@1从92.15%降至87.94%。在严重缺乏历史图像的情况下，合成补全的效果更为显著，10%真实老域覆盖率时，平均R@1提高了12.30和13.99个百分点。

**⚠️ 局限性**

限制在于研究仅使用了一个数据集，且合成老化方法可能无法完全再现真实历史图像的多样性。

---

## 607. MemSentry: A Framework for Detecting Persistent Memory Poisoning in Agentic AI

**arXiv ID:** 2609.08747 | [PDF](https://arxiv.org/pdf/2609.08747v1)

**作者:** Ayan Roy `[一作]` (Christopher Newport University), Kaustuvi Basu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MemSentry框架，用于检测代理AI系统中的持久内存中毒攻击，评估内存写入操作的安全性，并做出接受、审查或隔离的决策。

**💡 创新点**

创新点在于引入了一种结构化的预接纳评估管道，结合源信任、语义风险、攻击半径、访问风险和安全状态变化，来评估内存写入的潜在影响。

**🔧 技术方法**

使用了基于图的依赖关系模型、源信任评估、语义分类、影响估计和基于规则的决策逻辑等技术。

**📊 数据集**

使用了由GPT-4生成的1000个测试场景的数据集，数据集分为70%的训练集和30%的测试集。

**📈 对比分析**

与四种代表性方法（基于规则的Regex、TF-IDF+SVM、SBERT+LR和SetFit）进行了比较，SBERT+LR在整体性能上表现最佳，准确率为91.7%，宏F1分数为0.908，所有方法均能100%检测外部隔离类威胁。

**⚠️ 局限性**

局限性在于当前评估仅考虑单个内存操作，而未考虑协调的多步骤攻击；对于经过验证的内部用户，通常会升级到审查而不是自动隔离，表明需要更强的内容基础硬规则。

---

## 608. A New Backscattering Dual-Polarized Rectenna for Wireless Power Transfer and IoT Applications

**arXiv ID:** 2609.08833 | [PDF](https://arxiv.org/pdf/2609.08833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 609. Tools-CC-Bench: a Benchmark Suite for Collective Communication with Compression in HPC and AI Workloads

**arXiv ID:** 2609.08739 | [PDF](https://arxiv.org/pdf/2609.08739v1)

**作者:** Haozhe Fan `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Dingwen Tao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了CC-Bench，一个轻量级、可扩展的基准测试套件，用于在现实执行条件下评估集体通信中的压缩效果。

**💡 创新点**

CC-Bench是首个开源的基准测试套件，能够全面评估通信压缩，支持多种通信库和数据集，并提供应用特定的准确性度量。

**🔧 技术方法**

使用了声明式应用环境建模、函数级拦截和硬件计数器监控等技术，结合性能分析和资源干扰的系统化特征。

**📊 数据集**

使用了来自高性能计算（HPC）和大语言模型（LLM）工作负载的代表性数据集，包括气象数据、宇宙学数据和LLM的KV缓存跟踪。

**📈 对比分析**

与现有基准测试方法相比，CC-Bench提供了更细粒度的性能分析，能够揭示压缩与通信的延迟、吞吐量和准确性之间的权衡，评估结果显示在不同计算条件下的准确性和性能权衡。

**⚠️ 局限性**

CC-Bench的局限性在于，尽管它提供了灵活的配置和评估，但在处理极端数据集和特定应用场景时，可能仍需进一步优化和扩展。

---

## 610. CVT-GS: Learning to Simplify 3D Gaussian Splatting with Centroidal Voronoi Tessellation

**arXiv ID:** 2609.08730 | [PDF](https://arxiv.org/pdf/2609.08730v1)

**作者:** Bingxian Li `[一作]` (Beijing Institute of Technology), Bo Pang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种新的优化无关的后处理简化框架，直接压缩训练好的3D Gaussian Splatting (3DGS) 场景，而不牺牲视觉保真度。

**💡 创新点**

创新点在于将简化过程重新定义为基于渲染的多对一合并，而不是简单的原始点剔除，从而实现更高的视觉质量和更快的处理速度。

**🔧 技术方法**

使用了几何感知的质心Voronoi剖分（CVT）和轻量级神经网络合并器MergeNet。

**📊 数据集**

在多个3DGS数据集上进行了实验，包括NeRF-Synthetic、Mip-NeRF360、Tanks & Temples和Deep Blending。

**📈 对比分析**

与现有的最先进方法相比，提出的方法在相同的输出协议下，简化速度提高了12倍，同时PSNR提高了1.3 dB。

**⚠️ 局限性**

限制在于该方法仍然依赖于预训练的3DGS模型，且在极端压缩比下可能会面临视觉质量的进一步下降。

---

## 611. FIRE3D: Feed-forward Interactive 3D Scene Reconstruction Within A Minute

**arXiv ID:** 2609.08848 | [PDF](https://arxiv.org/pdf/2609.08848v1)

**作者:** Hongchi Xia `[一作]` (University of Illinois Urbana Champaign), Shenlong Wang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种统一框架，从单个RGB图像或视频中重建可用于仿真的3D环境，且无需额外的对象注释，能够在一分钟内完成。

**💡 创新点**

创新点在于提出了一种前馈框架，能够从未分割的RGB-D观测中重建对象级的纹理3D场景，并且设计了超紧凑的层次潜在流匹配生成器，实现了超过5倍的速度提升。

**🔧 技术方法**

使用了前馈神经网络，结合了压缩的对象表示（HC-VAE）和实例感知的生成模型。

**📊 数据集**

使用了一个大规模的训练数据集，包括80,000个场景、140,000个视频片段和500,000个对象，涵盖了室内布局、对象类别和外观变化。

**📈 对比分析**

与现有的基于感知、对象中心和优化的方法相比，性能在几何准确性、对象完整性、纹理质量和姿态一致性上都有显著提升，同时推理速度更快。

**⚠️ 局限性**

局限性在于该方法主要针对静态室内场景，且需要在网络接口处提供RGB-D观测的姿态，生成的资产尚未保证物理稳定、可重光照或可关节化。

---

## 612. Closing the Consistency Gap: Self-Evolving Agents That Learn to Stay on Course

**arXiv ID:** 2609.08832 | [PDF](https://arxiv.org/pdf/2609.08832v1)

**作者:** Evelyn Duesterwald `[一作]` (IBM Software Innovation Lab), Malgorzata Zimon `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自我演化的代理框架，通过识别不稳定的低一致性步骤并将其转化为代理可以在未来运行中借鉴的情节记忆，从而减少一致性差距。

**💡 创新点**

创新点在于引入了一种一致性分析器和指导生成器，前者能够识别代理执行过程中的不一致性，后者则生成针对性的指导方针以提高未来执行的一致性。

**🔧 技术方法**

使用了黑箱一致性分析器和基于大语言模型的指导生成器，结合了重采样技术来评估和改善代理的执行一致性。

**📊 数据集**

在AppWorld基准上进行了评估，使用了ReAct代理和GPT-4.1模型，涉及168个任务的测试。

**📈 对比分析**

与基线方法相比，该框架在同一任务的成功率提高了16个百分点，在相似任务的成功率提高了13个百分点，且没有降低准确性。

**⚠️ 局限性**

局限性包括无法直接测量或控制平台侧的非确定性，框架并未消除所有不一致性，且默认的重采样预算可能导致较高的计算成本。

---

## 613. Hi-FLoop: Hierarchical State-Feedback Loops for Multi-Timescale World Modeling

**arXiv ID:** 2609.08796 | [PDF](https://arxiv.org/pdf/2609.08796v1)

**作者:** Rx Fan `[一作]` (Beijing Normal University), Zhan H `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种多代理交通模拟框架，旨在从地图和观察历史中生成多样化、协调且物理上真实的未来状态。

**💡 创新点**

创新点在于引入了分支一致的多时间尺度状态反馈框架，能够在长时间范围内处理多个决策时间尺度，并在生成状态的演变中保持一致性。

**🔧 技术方法**

使用了QCNet风格的查询中心事实编码器，结合了意图条件和局部细化的MTR家族范式。

**📊 数据集**

使用了H-D公共验证集，包含955个场景进行评估。

**📈 对比分析**

与现有方法相比，该方法在场景联合评估中达到了ADE-at-joint-minFDE@8/joint-minFDE@8为2.377028/7.529927米，且在独立优化的情况下，oracle-minADE@8为1.196636米。

**⚠️ 局限性**

局限性在于模型的复杂性和对计算资源的高需求，可能限制了其在实时应用中的可行性。

---

## 614. Silent Revision: Measuring Undisclosed Change in the Safety Frameworks of Frontier AI Developers

**arXiv ID:** 2609.08789 | [PDF](https://arxiv.org/pdf/2609.08789v1)

**作者:** Louis Yiven Zhu `[一作]` `[通讯]` (University of Oxford), Louis Yiven Zhu (University of Oxford)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了前沿人工智能开发者发布的安全框架的修订透明度，提出了“沉默修订率”这一概念，量化了开发者在修订中未能明确说明的重大变化的比例。

**💡 创新点**

创新点在于引入了沉默修订率的测量方法，并提供了一个版本化的、哈希固定的语料库，以便计算这一指标。

**🔧 技术方法**

使用了系统内容分析的方法，结合了法律文档的审计和测量建模技术。

**📊 数据集**

使用了十二个开发者在2024年首尔峰会后发布的安全框架的所有公开版本，以及每个开发者的修订说明。

**📈 对比分析**

通过对710个承诺实例的追踪和编码，发现67%的重大变化在严格标准下是沉默的，53%在宽松标准下是沉默的，且大多数变化倾向于削弱承诺。

**⚠️ 局限性**

限制在于尚未报告编码者之间的一致性，且修订的类型学可能存在选择偏差，此外，缺乏比较类的对照组使得无法评估沉默率的高低。

---

## 615. ZK-Trace: Certified Collusion Tracing with Zero-Knowledge Credentials for Federated GNSS Interference Monitoring

**arXiv ID:** 2609.08763 | [PDF](https://arxiv.org/pdf/2609.08763v1)

**作者:** Redwanul Karim `[一作]` (Fraunhofer Institute for Integrated Circuits IIS), Felix Ott `[通讯]` (Fraunhofer Institute for Integrated Circuits IIS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种名为ZK-Trace的系统，用于在联合全球导航卫星系统（GNSS）干扰监测中进行安全的叛徒追踪，结合了零知识凭证和特定接收者的指纹技术。

**💡 创新点**

创新点在于提供了有限的可执行追踪保证，能够在不需要泄露者合作的情况下进行离线追踪，并且在多个条件下建立了错误指控的界限。

**🔧 技术方法**

使用了零知识证明、Tardos指纹、特征匹配和联合学习等技术。

**📊 数据集**

使用了模拟的GNSS联合体数据集和CIFAR-10数据集进行实验。

**📈 对比分析**

与现有的五种联邦水印方法进行了比较，ZK-Trace在追踪性能上表现优越，能够在不指控无辜者的情况下追踪712个混合样本中的720个，并且在GNSS和CIFAR-10上分别仅损失了4.8和6.1个百分点的准确率。

**⚠️ 局限性**

限制在于ZK-Trace未能检测或防止恶意更新，并且需要进一步验证在不同GNSS站点的条件独立性和探针边际假设。

---

## 616. DCLP++: Learning to Navigate with Footprint Clearance and Relative Motion

**arXiv ID:** 2609.08711 | [PDF](https://arxiv.org/pdf/2609.08711v1)

**作者:** Shanze Wang `[一作]`, Wei Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于足迹间隙的局部导航框架，用于研究动态环境中的相对运动特征。

**💡 创新点**

创新点在于使用足迹间隙作为几何基础，替代传感器到障碍物的距离，从而提高导航成功率。

**🔧 技术方法**

采用了深度强化学习（DRL）技术，结合了传感器测量和运动特征的映射。

**📊 数据集**

使用了一个包含20个移动障碍物的模拟环境，机器人速度限制为1 m/s，进行了100个固定验证任务。

**📈 对比分析**

与传统的传感器范围方法相比，足迹间隙方法在两个选定的训练种子中成功率提高了28个百分点，达到70%。运动信息的效果因不同的实现而异。

**⚠️ 局限性**

局限性在于未能测试所有训练种子的稳定性，且未考虑物理运动感知和不同机器人配置的影响。

---

## 617. Prior-free relative 6D pose estimation of multiple object instances

**arXiv ID:** 2609.08949 | [PDF](https://arxiv.org/pdf/2609.08949v1)

**作者:** Behdad Khodabandehloo `[一作]` (Fondazione Bruno Kessler), Fabio Poiesi `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无先验的相对6D姿态估计方法，旨在在同一图像中估计多个未知物体实例的相对姿态，而无需CAD模型、模板或参考图像。

**💡 创新点**

创新点在于首次提出了无先验的相对6D姿态估计任务，并开发了一个训练无关的方法PROSE，利用多模态基础特征和循环一致性来精确估计相对姿态。

**🔧 技术方法**

使用了多模态基础特征提取技术，包括视觉编码器DINOv2和几何编码器dGeDi，并通过RANSAC算法进行3D配准。

**📊 数据集**

使用了PRENCH基准数据集，该数据集由三个多实例BOP数据集（IC-BIN、IC-MI和XYZ-IBD）构建，并增加了任务特定的元数据。

**📈 对比分析**

与现有的单图像方法（One2Any和ConceptPose）进行比较，PROSE在所有三个数据集上均表现出色，平均召回率提高了19.2至19.7，旋转和位移误差减少了9.2至36.0度和7.7至15.3毫米。

**⚠️ 局限性**

限制在于该方法假设实例掩码和预定义的锚点是可用的，未来的工作可以探索自动实例分割和几何感知同步，以考虑几何不兼容的约束。

---

## 618. FINALLY: A Dataset Recommender System for Recommender-Systems Research

**arXiv ID:** 2609.08941 | [PDF](https://arxiv.org/pdf/2609.08941v1)

**作者:** Louis Owie `[一作]` `[通讯]`, Louis Owie

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为FINALLY的网络数据集推荐系统，用于构建可配置的数据集集合，以便进行离线推荐系统评估。

**💡 创新点**

FINALLY提供了一个集成的推荐工作流程，结合了所需数据集、候选池限制、元数据过滤器、可配置目标集大小以及基于有效协方差和凸包目标的多样性和非多样性策略。

**🔧 技术方法**

使用了基于有效协方差和凸包的多样性度量，结合随机选择和适应性实现，构建了推荐工作流程。

**📊 数据集**

评估使用了96个推荐系统数据集，并进行了420次推荐运行，涵盖了十种系统配置。

**📈 对比分析**

FINALLY的推荐结果在所有配置中都满足了目标大小、避免重复、快照成员资格、所需数据集和元数据过滤要求。多样性策略的得分高于随机结果，而非多样性策略的得分低于随机结果，显示出技术一致性。

**⚠️ 局限性**

未能证明生成的选择在科学适用性、全局最优性或实际优越性方面的有效性。

---

## 619. CoSA: Correlation-Guided Change A ttention with Learnable Residual Gating for Remote Sensing Change Detection

**arXiv ID:** 2609.08914 | [PDF](https://arxiv.org/pdf/2609.08914v1)

**作者:** Abdirashid Omar `[一作]` (Kookmin University), Jonghyuk Park `[通讯]` (Kookmin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种数据高效的目标域管道，使用241张手动标注的CCTV图像和5926帧未标注的CCTV图像进行交叉步道分割。

**💡 创新点**

提出了一种轻量级的伪标签排名规则，结合像素置信度和交叉步道区域先验，从未标注帧中选择1000个样本。

**🔧 技术方法**

使用了自定义的U-Net和DeepLabV3-ResNet50模型进行训练和推理。

**📊 数据集**

使用了241张手动标注的CCTV图像和5926帧未标注的CCTV图像。

**📈 对比分析**

与源域FPV测试的93.05% IoU相比，目标域CCTV的88.91% IoU是基于40张手动验证图像的最佳结果。第二阶段的98.52% IoU是基于伪标签的内部一致性，而非人类标注的准确性。

**⚠️ 局限性**

限制在于验证图像数量过少，第二阶段未包含手动标注样本，随机分割可能导致场景和时间冗余，以及图像和掩码翻转不同步引入训练噪声。

---

## 620. To Stop or Not to Stop: Exploring the Intention-Behavior Gaps in Smartphone Usage

**arXiv ID:** 2609.08909 | [PDF](https://arxiv.org/pdf/2609.08909v1)

**作者:** Jian Zheng `[一作]`, Eun Kyoung Choe `[通讯]` (University of Maryland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种新的方法，将问题性智能手机使用（PSU）定义为意图-行为差距（IBG），并收集了37名参与者在两周内的自我报告数据，分析了影响IBG的因素，并开发了机器学习模型以实时预测IBG。

**💡 创新点**

创新点在于将PSU操作化为意图-行为差距（IBG），并探讨了人口统计学和情境变量对IBG的影响，同时开发了实时预测IBG的机器学习模型。

**🔧 技术方法**

使用了机器学习技术，特别是随机森林回归模型来预测意图、行为和IBG。

**📊 数据集**

使用的数据集包括来自37名参与者的5772个手机使用会话的意图、行为和情境因素数据。

**📈 对比分析**

通过线性混合效应模型比较了不同因素对意图、行为和IBG的影响，发现性别、时间、应用类别和输入交互等因素显著影响IBG。机器学习模型的性能显示，个人模型在意图预测上表现最佳，而结合模型在行为和IBG预测上表现最佳。

**⚠️ 局限性**

本研究的局限性包括样本量较小，且参与者均为对减少手机使用感兴趣的用户，可能无法推广到其他用户群体。此外，数据的有效性受到参与者回忆准确性和单一问题调查的影响。

---

## 621. The BatchNorm Illusion: Diagnosing Normalization Artifacts in Machine Unlearning Evaluation

**arXiv ID:** 2609.08901 | [PDF](https://arxiv.org/pdf/2609.08901v1)

**作者:** Aaryaman Kalani `[一作]` (BITS Pilani), Yash Sinha `[通讯]` (BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了近似机器遗忘的评估问题，特别是在BatchNorm架构中，提出了一种通过单次前向传递来逆转表面遗忘的现象。

**💡 创新点**

创新点在于识别并形式化了BatchNorm在训练和评估模式下的两种不同的归一化伪影，提出了BN重校准作为一种低成本的诊断工具。

**🔧 技术方法**

使用了Batch Normalization（BN）和线性探测等技术，提出了一种权重保持的固定点操作来进行BN重校准。

**📊 数据集**

使用了CIFAR-10和CIFAR-100数据集进行实证评估，测试了九种不同的遗忘方法。

**📈 对比分析**

通过与标准基准的比较，发现BN重校准可以显著提高遗忘准确率，恢复了六种方法的表面遗忘准确率，最大提升达78个百分点。

**⚠️ 局限性**

限制在于仅评估了分类基准，未对实例级遗忘进行全面评估，且未在语言模型上进行基准测试。

---

## 622. On Weighted Mathai-Haubold Entropy Measures

**arXiv ID:** 2609.08889 | [PDF](https://arxiv.org/pdf/2609.08889v1)

**作者:** Oindrali Das `[一作]` (Ranaghat College), Siddhartha Chakraborty `[通讯]` (University of Kalyani)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了加权Mathai-Haubold熵及其残差和过去版本，并研究了它们的性质。

**💡 创新点**

创新点在于引入了加权Mathai-Haubold熵的概念，并基于此开发了老化类和相关不等式。

**🔧 技术方法**

使用了非参数估计技术，特别是基于核密度的估计方法。

**📊 数据集**

使用了标准均匀分布和标准指数分布进行模拟研究。

**📈 对比分析**

通过Monte-Carlo模拟评估了提出的估计器的偏差和均方误差，结果表明在样本量增加时，估计器的性能有所改善。

**⚠️ 局限性**

限制在于需要进一步研究其他类型的估计器，如基于间隔和分位数的估计器。

---

## 623. Towards Standardized Evaluation of GPU Memory Safety with GMSBench

**arXiv ID:** 2609.08871 | [PDF](https://arxiv.org/pdf/2609.08871v1)

**作者:** Saurabh Singh `[一作]` (Georgia Institute of Technology), Hyesoon Kim `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的GPU内存安全基准测试套件GMSBench，旨在评估不同GPU内存安全机制的有效性，涵盖了多种内存安全违规情况。

**💡 创新点**

创新点在于系统化分类GPU内存安全违规，并提供一个开放源代码的基准测试套件，能够全面评估GPU内存安全机制的检测能力。

**🔧 技术方法**

使用了CUDA技术来设计自包含的测试，涵盖空间错误、时间错误和数据竞争等多种内存安全违规。

**📊 数据集**

使用了多种GPU架构的测试，包括RTX A5000、A100和GH200等，确保基准测试的广泛适用性。

**📈 对比分析**

通过与Compute Sanitizer等现有工具的比较，评估了GMSBench的有效性，结果显示Compute Sanitizer在检测覆盖率上存在显著的盲点。

**⚠️ 局限性**

限制在于现有的检测机制可能无法覆盖所有类型的内存安全违规，且某些测试在本地执行时未能可靠地表现出违规现象。

---

## 624. REDSI: Addressing the Reproducibility and Evaluation Consistency of Differentiable Search Indexing for Document Retrieval

**arXiv ID:** 2609.08860 | [PDF](https://arxiv.org/pdf/2609.08860v1)

**作者:** Vivien Nicolas `[一作]` (Artefact Research Center), Caio Corro `[通讯]` (INSA Rennes)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ReDSI，这是第一个支持所有三种文档标识符类型的开源DSI实现，并提供了可参数化的NQ320K构建管道。

**💡 创新点**

ReDSI是第一个公开的DSI实现，支持原始的三种标识符类型，并且通过分离文档去重和序列化，展示了这两者对检索效果的影响。

**🔧 技术方法**

使用了T5作为基础模型，并进行了广泛的实验以评估模型缩放下的检索有效性、参数效率、训练方法和解码策略。

**📊 数据集**

使用了NQ320K数据集，该数据集是从Natural Questions中构建的，并且提供了一个可参数化的构建管道。

**📈 对比分析**

与之前的DSI基线相比，ReDSI在检索效果上具有竞争力或更强的表现，尤其是在模型缩放时，原子标识符在每个评估的模型规模上都优于其他类型。

**⚠️ 局限性**

实验仅限于英语的NQ衍生集合和T5系列的基础模型，标识符类型的相对表现可能在其他语料库或架构上有所不同，且原子标识符的可扩展性尚未在更大规模的文档集合中验证。

---

## 625. DSE-VTG: Dual-Side Enhancement for Training-Free Video Temporal Grounding

**arXiv ID:** 2609.08850 | [PDF](https://arxiv.org/pdf/2609.08850v1)

**作者:** Zhuo Cao `[一作]` (University of Queensland), Xue Li `[通讯]` (University of Queensland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为DSE-VTG的双侧增强框架，用于文本引导的视频时间定位，旨在根据文本查询定位未剪辑视频中的相关片段。

**💡 创新点**

创新点在于通过多尺度相似性融合（MSF）和查询级测试时间自适应（Q-TTA）解决了图像中心视觉偏差和静态查询歧义的问题，而无需特定任务的训练。

**🔧 技术方法**

使用了多尺度相似性融合（MSF）和查询级测试时间自适应（Q-TTA）技术。

**📊 数据集**

在三个标准基准（Charades-STA、ActivityNet Captions、QVHighlights）和两个超出分布（OOD）基准（Charades-CG、Charades-CD）上进行了实验。

**📈 对比分析**

DSE-VTG在训练无关的方法中实现了最先进的性能，在Charades-STA上，mIoU比最强的先前训练无关方法提高了5.61个百分点，并在分布转移下超越了最强的监督基线2.76 mIoU。

**⚠️ 局限性**

限制在于该方法仍然依赖于预训练的视觉-语言模型，可能在特定任务上表现不如经过专门训练的模型。

---

## 626. Earth System World Model for What-If Simulations: A Case Study for Terrestrial Ecosystems

**arXiv ID:** 2609.08855 | [PDF](https://arxiv.org/pdf/2609.08855v1)

**作者:** Zhihao Wang `[一作]` (University of Maryland), Yiqun Xie `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于动作条件的世界建模框架，用于地球系统仿真，允许用户指定的结构干预。

**💡 创新点**

创新点在于通过转移-动作预训练和掩蔽响应学习，使模型能够在没有手动标注干预的情况下学习可控的状态转移，并推断未观察到的变量。

**🔧 技术方法**

使用了转移-动作预训练和掩蔽响应学习技术。

**📊 数据集**

使用了CarbonGlobe数据集，这是一个基于生态系统人口（ED）模型的全球40年生态系统预测数据集。

**📈 对比分析**

与持久性基线和专用无动作基线进行比较，结果表明该模型在长时间范围内的仿真精度与基线相当，且在用户指定的动作下能够产生一致的响应。

**⚠️ 局限性**

限制在于模型在处理部分状态编辑时的稳定性和对未观察变量的推断能力可能受到影响。

---

## 627. Experience Funnel: A State-Policy Alternating Loop for Self-Evolving Agents

**arXiv ID:** 2609.08919 | [PDF](https://arxiv.org/pdf/2609.08919v1)

**作者:** Wenbo Gao `[一作]` (Hong Kong Polytechnic University), Yaoyuan Wang `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Experience Funnel的自我进化框架，旨在通过快速状态适应与缓慢策略整合的交替循环，将丰富的任务特定交互经验转化为可重用的模型能力。

**💡 创新点**

创新点在于结合了快速的状态适应和缓慢的策略整合，通过交替循环来处理经验的积累与转化，解决了现有方法在经验管理上的不足。

**🔧 技术方法**

使用了状态-策略交替演化和基于转移的技能蒸馏技术，能够在不同的交互阶段有效地整合和更新经验。

**📊 数据集**

在多个代理基准上进行了实验，包括问答、具身交互和网页导航等任务，使用了Qwen3.5-4B和Qwen3.5-27B模型进行对比。

**📈 对比分析**

与现有的状态演化和策略内化方法相比，Experience Funnel在所有测试环境中均表现出更好的性能，平均得分达到57.6%，超越了最强的显式状态方法和策略内化基线。

**⚠️ 局限性**

限制在于该框架主要设计用于后部署设置，适用于重复相关任务的代理，可能不适用于一次性任务或反馈稀疏的领域。此外，迭代演化过程需要额外的计算资源，可能影响实时服务的延迟。

---

## 628. Medical AI Encodes a "Feeling of Error": Verifying Cancer Segmentation via Internal Concepts

**arXiv ID:** 2609.08879 | [PDF](https://arxiv.org/pdf/2609.08879v1)

**作者:** Mengmeng Ma `[一作]` (University of Virginia), Xi Peng `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究探讨了癌症分割模型的内部信号，提出了一种通过内部概念验证癌症分割的方法，旨在识别模型的潜在错误并提供解释。

**💡 创新点**

创新点在于通过稀疏自编码器提取模型内部概念，展示了失败案例与成功案例在内部激活模式上的显著差异，从而实现更准确的失败检测和解释。

**🔧 技术方法**

使用了稀疏自编码器（SAE）作为机制可解释性工具，分析模型的内部激活，并构建了一个轻量级分类器来进行失败检测。

**📊 数据集**

使用了三个公共癌症数据集进行实验：前列腺癌（PI-CAI和Prostate158）和胰腺癌（PanTS）。

**📈 对比分析**

与基于输出的失败检测方法相比，所提方法在失败检测性能上表现更优，同时保持了高分割质量，打破了传统的敏感性与质量之间的权衡。

**⚠️ 局限性**

限制在于当前方法捕捉的是概念的出现而非因果关系，未来需要探索概念之间的机制性互动，并希望能够在不同癌症类型之间共享概念。

---

## 629. FRAME: Factored Retrieval via Attribute Readouts for Object-Centric Scene Memory

**arXiv ID:** 2609.08886 | [PDF](https://arxiv.org/pdf/2609.08886v1)

**作者:** Woosang Jeon `[一作]` (Seoul National University), Taehyeong Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种属性组合检索方法FRAME，用于语言引导的机器人在固定场景记忆中检索满足特定属性的对象。

**💡 创新点**

创新点在于将语言查询转化为与属性相关的权重，并通过学习的读取输出从对象嵌入中估计每个属性的证据，从而实现高效的多属性检索。

**🔧 技术方法**

使用了FRAME方法，该方法通过属性读取输出将语言分解为属性原语，并根据查询组合相应的证据进行检索。

**📊 数据集**

使用了Habitat Synthetic Scenes Dataset (HSSD)数据集，该数据集提供了固定的场景记忆和基于属性定义的目标。

**📈 对比分析**

与其他检索方法相比，FRAME在属性组合检索任务上表现优越，且后处理对象评分简化为轻量级的矩阵-向量计算，性能显著提升。

**⚠️ 局限性**

限制在于HSSD协议仅在预定义的属性词汇上进行检索，未能扩展到开放词汇属性、关系引用和更丰富的语言现象。

---

## 630. Model Predictive Control of Tensegrity Robots via Contact-Aware Graph Neural Dynamics Model

**arXiv ID:** 2609.08958 | [PDF](https://arxiv.org/pdf/2609.08958v1)

**作者:** Nelson Chen `[一作]` (Rutgers University), Mridul Aanjaneya `[通讯]` (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图神经网络（GNN）动态模型的模型预测路径积分（MPPI）控制器，用于三杆张力机器人在非水平平面地形和障碍物上的导航。

**💡 创新点**

创新点在于扩展了GNN动态模型，增加了可微分的接触检测模块，使得模型能够处理非水平平面地形、环境障碍物和自碰撞的交互。

**🔧 技术方法**

使用了图神经网络（GNN）和模型预测路径积分（MPPI）技术。

**📊 数据集**

在MuJoCo模拟环境中进行了实验，涉及五个导航任务，包括墙障碍、坡道、狭窄走廊、低净空结构和一个复合3D障碍课程。

**📈 对比分析**

与基于A*的重新规划和仅使用MPPI的变体相比，混合MPPI控制器在导航性能上表现优越，成功率和任务完成时间均有所提高。

**⚠️ 局限性**

限制在于当前方法仅适用于平面表面表示，扩展到完全非结构化地形是一个自然的下一步。此外，混合MPPI中的转向原语是手工设计的，直接从数据中学习这些原语可能会提高通用性并减少手动设计的工作量。

---

## 631. PMMS Allocations Need Not Exist for 3 Agents with Additive Valuations

**arXiv ID:** 2609.08954 | [PDF](https://arxiv.org/pdf/2609.08954v1)

**作者:** Paul Gölz `[一作]` `[通讯]`, Paul Gölz

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文展示了在某些公平分配问题中，无法满足成对最大最小共享（PMMS）属性的实例，特别是在不可分割商品和加法估值的情况下。

**💡 创新点**

创新点在于提供了一个具体的实例，证明了在n=3个代理和m=9个商品的情况下，PMMS分配不存在，并且通过计算机的穷举枚举证明了PMMS无法在比78/79（约0.987）更高的比率内被近似。

**🔧 技术方法**

使用了穷举搜索和计算机辅助证明的方法来验证PMMS分配的不存在性。

**📊 数据集**

使用了一个包含3个代理和9个商品的特定数据集，代理的估值是加法的。

**📈 对比分析**

通过与现有方法的比较，证明了在给定的实例中，PMMS分配不存在，且无法在高于78/79的比率内近似，显示了该方法的性能限制。

**⚠️ 局限性**

限制在于该研究仅针对特定的代理数量和商品数量，且PMMS的存在性问题在更广泛的情况下仍未解决。

---

## 632. SkillAdam: Stable and Efficient Skill Evolution for Agents

**arXiv ID:** 2609.08944 | [PDF](https://arxiv.org/pdf/2609.08944v1)

**作者:** Gaoyuan Li `[一作]` (Renmin University of China), Zang Li `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于Adam的框架，用于优化离散和非可微的技能文档，以实现稳定和高效的技能自我演化。

**💡 创新点**

创新点在于引入了优化记忆和波动驱动的编辑预算，分别作为Adam的第一和第二矩的功能类比，从而稳定优化方向并适应更新幅度。

**🔧 技术方法**

使用了一种基于Adam的优化框架，结合了Evolving Issue Tracker和波动驱动的编辑预算。

**📊 数据集**

在七个基准测试上进行了实验，这些基准测试涵盖了短期和长期任务。

**📈 对比分析**

与七个基线方法进行了比较，结果显示该方法在多个基准上实现了最先进的性能，且所需的优化迭代次数显著减少，优化成本也低于之前的方法。

**⚠️ 局限性**

限制在于该方法的有效性可能依赖于特定的任务和数据集，可能在其他未测试的领域表现不佳。

---

## 633. Evaluating and Improving Evidence-Grounded Fact-Checking in LLMs via Multi-Round Evidence Ablation

**arXiv ID:** 2609.08943 | [PDF](https://arxiv.org/pdf/2609.08943v1)

**作者:** Xingyu Deng `[一作]` (University of Sheffield), Mark Stevenson `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的评估框架Fact-Ablated Evaluation (FAE)，用于评估大型语言模型（LLMs）在事实核查中的证据依赖性，并提出了REAL（Rigorous Evidence Ablation Learning）训练框架，以增强模型对证据的依赖性。

**💡 创新点**

创新点在于引入了FAE评估框架，通过逐步去除证据来观察模型的预测变化，从而揭示现有LLMs在事实核查中对参数知识的依赖。同时，REAL框架通过反事实监督来促进证据依赖的验证。

**🔧 技术方法**

使用了Fact-Ablated Evaluation (FAE)评估框架和Rigorous Evidence Ablation Learning (REAL)训练框架，结合了反事实监督和增强证据注释的技术。

**📊 数据集**

使用了四个事实核查数据集进行实验，包括FEVER、SciFact、Climate-FEVER和Check-COVID。

**📈 对比分析**

与标准微调模型相比，使用REAL训练的模型在证据依赖性方面表现更优，且在多个数据集上均显示出更高的准确性和更强的证据依赖性。实验结果表明，REAL显著提高了模型的证据依赖性，同时保持了强大的事实核查性能。

**⚠️ 局限性**

FAE需要迭代的证据去除和重复的验证推理，使得评估在计算上比标准的单次事实核查评估更为昂贵。此外，现有事实核查基准中的证据冗余和不完整注释可能仍允许模型在去除“金证据”后保持正确预测，难以完美区分证据基础的验证与参数记忆。

---

## 634. Remotely Detectable Keyed Communication through Motion

**arXiv ID:** 2609.08920 | [PDF](https://arxiv.org/pdf/2609.08920v1)

**作者:** Benjamin Chang `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文探讨了通过机器人运动进行通信的问题，提出了一种名为运动消息传递（MTM）的方法，能够在不影响机器人政策性能的情况下，通过运动传递可检测的消息。

**💡 创新点**

创新点在于将任意消息内容编码为机器人运动中的噪声，并且该方法不需要额外的硬件或直接链接到机器人，提供了一种新的物理通信渠道。

**🔧 技术方法**

使用了运动水印技术（CoNoCo）作为基础，结合了随机政策的现有噪声来编码消息，并通过远程传感器数据进行解码。

**📊 数据集**

在多个模拟环境（如Discovery、Lunar Lander、Reacher、Football）和真实机器人（RoboMaster）平台上进行了验证。

**📈 对比分析**

在五个环境中，MTM方法在模拟中实现了100%的消息恢复率，在真实机器人上，四个机器人在12秒内成功恢复了8位消息，传输速率为0.67比特/秒。

**⚠️ 局限性**

限制在于该框架假设消息是离散的，无法表示连续变化的消息；对于低随机性的政策，MTM无法使用；长负载的可靠性未经过测试；该方案隐藏消息内容但不隐藏水印的存在。

---

## 635. Healthcare Utilization, Chronic Condition Management, and Workplace Functioning Among Users of a Purpose-Built Mental Health AI (Ash): Cross-Sectional Study

**arXiv ID:** 2609.08890 | [PDF](https://arxiv.org/pdf/2609.08890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 636. A Quantum-Inspired Approach to MaxCut Based on Sparse Walsh/Pauli-Correlation Encoding

**arXiv ID:** 2609.08907 | [PDF](https://arxiv.org/pdf/2609.08907v1)

**作者:** Cesar Augusto do Amaral `[一作]` (Instituto de Pesquisas Eldorado), Fernando Augusto Caletti de Barros `[通讯]` (Instituto de Pesquisas Eldorado)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于稀疏Walsh/Pauli相关编码的量子启发式MaxCut求解器，使用期望值表示图的顶点，而不是直接分配量子比特。

**💡 创新点**

创新点在于通过稀疏Walsh自相关计算来获得决策变量，从而实现了高效的量子启发式优化方法，并且在所有测试实例中超越了随机搜索和禁忌搜索。

**🔧 技术方法**

使用了稀疏Walsh函数和Pauli相关编码技术。

**📊 数据集**

在Gset实例（G1, G6, G12, G18）上进行了评估。

**📈 对比分析**

与随机搜索和禁忌搜索进行了比较，Walsh/PCE方法在所有测试实例中都表现出更高的近似比率，并且在运行时间上也优于这两种基线方法。

**⚠️ 局限性**

限制在于所使用的超参数较为简单且固定，未来可以通过更系统的优化和适应性调整来进一步提高近似比率。

---

## 637. Visible-Reachable Workspace for Perception-Aware Humanoid Design

**arXiv ID:** 2609.08905 | [PDF](https://arxiv.org/pdf/2609.08905v1)

**作者:** Boxi Xia `[一作]` (Duke University), Boyuan Chen `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可视可达工作空间（VRW）度量，结合了机器人在可达目标的同时能否观察到这些目标的能力，并在此基础上设计了一种具有独立驱动RGB-D相机的31自由度类人机器人。

**💡 创新点**

创新点在于引入了可视可达工作空间（VRW）这一概念，强调了在设计阶段考虑可视性与可达性之间的关系，并扩展到同时观察空间分离工作区域的能力。

**🔧 技术方法**

使用了独立驱动的RGB-D相机和全身强化学习控制策略，结合了运动学和视觉感知的设计。

**📊 数据集**

在多个类人机器人平台上进行了评估，包括Unitree G1、Booster T1、Apptronik Apollo、Fourier GR-3和PAL Talos等。

**📈 对比分析**

通过与其他类人机器人进行比较，发现VRW设计的机器人在可视可达覆盖率上显著提高，从38%提升至97%，并且在双目标操作中，完成时间减少了17%，机械能耗降低了19%。

**⚠️ 局限性**

限制在于当前的VRW度量主要关注几何可视性，未来的工作可以扩展到考虑传感质量、不确定性和任务依赖的感知需求。

---

## 638. Evolution of Multimodal Question Answering: From Modality-Adaptive Extraction to Unified Language Representation

**arXiv ID:** 2609.08896 | [PDF](https://arxiv.org/pdf/2609.08896v1)

**作者:** Abdullah Al Shafi `[一作]` `[通讯]` (Khulna University of Engineering & Technology), Abdullah Al Shafi (Khulna University of Engineering & Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对三种重要的多模态问答框架进行了全面的方法比较，分析了从模态自适应提取到统一语言表示的演变，探讨了如何在文本、表格和图像等异构源之间进行推理。

**💡 创新点**

创新点在于系统地比较了MAE、Solar和UniMMQA三种框架，强调了从显式模态特定处理向统一文本中心公式的转变，并指出了现有方法的关键局限性。

**🔧 技术方法**

使用了多种技术，包括模态自适应提取、检索增强推理和统一语言表示，结合了预训练语言模型（PLMs）来处理多模态输入。

**📊 数据集**

使用了多个公共基准数据集，包括ManymodalQA、MultimodalQA和MMConvQA，这些数据集包含文本、表格和图像的问答对。

**📈 对比分析**

通过与多个基线模型（如ORConvQA、ManymodalQA等）进行比较，UniMMQA在Exact Match（EM）和F1分数上表现出色，显示出其在多模态推理中的优势。

**⚠️ 局限性**

局限性包括在模态转换过程中可能导致的信息丢失、在多阶段管道中的错误传播，以及捕捉细粒度跨模态依赖的能力不足。

---

## 639. The Complexity of Membership, Uniqueness, and Counting for Optimal Proportional Approval Voting Committees

**arXiv ID:** 2609.08884 | [PDF](https://arxiv.org/pdf/2609.08884v1)

**作者:** Yizhou Ai `[一作]` `[通讯]` (University of Toronto), Yizhou Ai (University of Toronto)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了比例批准投票（PAV）选择最大化和谐效用总和的委员会，探讨了最大化委员会的性质，包括候选人是否属于某些或所有委员会、最优解的唯一性以及最优解的数量。

**💡 创新点**

提出了关于候选人成员资格、唯一性和计数问题的确切完全性分类，特别是在委员会大小作为输入的情况下，证明了这些问题的复杂性。

**🔧 技术方法**

使用了和谐效用函数来评估候选人，并通过多项式时间的度量约简构造了选举实例，以实现候选人成员资格和计数问题的复杂性证明。

**📊 数据集**

使用了基于批准投票的选举模型，具体包括候选人集合、选民批准的候选人子集和委员会大小作为输入。

**📈 对比分析**

通过与已知的NP完全问题的比较，证明了候选人成员资格、唯一性和计数问题的复杂性，显示这些问题在特定情况下的难度与已知的复杂性类相同。

**⚠️ 局限性**

在处理具有最多两个最优委员会的实例时，唯一性问题仍然是NP完全的，且在计数问题中，构造的选举实例的最优委员会数量与源输入的复杂性相关。

---

## 640. High-Magnetization Sampling at Low Temperatures: Ising Models and Bayesian Sparse Linear Regression

**arXiv ID:** 2609.08873 | [PDF](https://arxiv.org/pdf/2609.08873v1)

**作者:** Syamantak Kumar `[一作]` (University of Texas at Austin), Yusong Zhu `[通讯]` (University of Texas at Austin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文开发了利用稀疏性在高维采样问题中的框架，特别是在Hamming切片上，针对Ising模型和贝叶斯稀疏线性回归问题设计了改进的采样器。

**💡 创新点**

创新点在于提出了一种新的框架，能够在固定磁化的Sherrington-Kirkpatrick (SK)模型中实现多项式时间的采样，并且在任意低温下也能有效采样，同时改进了贝叶斯稀疏线性回归的测量复杂度。

**🔧 技术方法**

使用了稀疏Dobrushin条件和trickle down定理等技术，结合退火策略来估计归一化常数，从而实现高效的采样。

**📊 数据集**

使用了Sherrington-Kirkpatrick模型和贝叶斯稀疏线性回归的标准高斯后验模型作为数据集，特别是在高维情况下的稀疏信号。

**📈 对比分析**

与现有方法相比，提出的采样器在固定磁化的SK模型中能够在任意温度下以多项式时间进行采样，且在贝叶斯稀疏线性回归中，测量复杂度从n ≳ k^3 log^3 d改进到n ≳ k^1.5 log^2 d + k log^3 d，性能显著提升。

**⚠️ 局限性**

限制在于对于高磁化的采样问题，仍然存在对k的依赖性，且在某些情况下，采样的复杂度可能仍然较高，尤其是在处理更复杂的模型时。

---

## 641. OntoKG-EQ: A provenance-grounded, competency-question-governed knowledge graph for auditable analyst querying

**arXiv ID:** 2609.08869 | [PDF](https://arxiv.org/pdf/2609.08869v1)

**作者:** Furqan Nasir `[一作]` (City University of Science and Information Technology), Abdul Moiz Altaf `[通讯]` (City University of Science and Information Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为OntoKG-EQ的知识图谱系统，用于在新兴市场的股权数据上进行可解释的分析查询。

**💡 创新点**

创新点在于通过五个固定的能力问题来驱动构建方法，确保每个术语都有明确的证据来源，并且每个结果都可以追溯到其观察、证据和来源。

**🔧 技术方法**

使用了知识图谱、RDF、SPARQL查询、SHACL结构验证和推理规则等技术。

**📊 数据集**

使用了来自巴基斯坦、马来西亚和印度尼西亚的策划数据集。

**📈 对比分析**

与关系数据库基线进行比较，结果显示图谱层并未改变分析结果，但提供了治理、来源和自解释结构的价值。通过用户研究发现，证据包显著提高了参与者的信任感和完整性感。

**⚠️ 局限性**

限制在于本体的广度适用性，当前方法在结构相似的市场中表现良好，但在更异构的市场中尚未验证其有效性。

---

## 642. DJPlus: Generating minimal test suites for strong coverage criteria in graph models

**arXiv ID:** 2609.08953 | [PDF](https://arxiv.org/pdf/2609.08953v1)

**作者:** Yavuz Köroğlu `[一作]`, Franz Wotawa `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新颖的优化驱动方法DJPlus，该方法从图模型生成减少的测试套件，同时满足给定的基于图的测试要求。

**💡 创新点**

DJPlus方法在生成最小数量的测试用例的同时，减少了测试步骤的数量，克服了现有方法的冗余问题。

**🔧 技术方法**

使用了图模型和优化算法，特别是通过构建要求流图（RFG）来消除超顶点，从而优化测试步骤。

**📊 数据集**

在四个现实系统上进行了实验，包括两个Web应用（Parabank和Testinium）和两个硬件应用（TLC和RISC-V）。

**📈 对比分析**

与其他方法相比，DJPlus生成的测试步骤数量更少，其他方法生成的冗余测试步骤比DJPlus多2到26倍。DJPlus在执行时间上也表现出显著的减少。

**⚠️ 局限性**

DJPlus在处理嵌套循环时仍然存在次优性，可能无法生成绝对最小的测试步骤数量。

---

## 643. Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems

**arXiv ID:** 2609.08887 | [PDF](https://arxiv.org/pdf/2609.08887v1)

**作者:** Maximilian Schall `[一作]` (Perplexity AI), Denis Bykov `[通讯]` (Perplexity AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个新的大规模检索基准Q2D-Web，旨在评估大规模生产环境中的第一阶段检索器，结合了一个包含约1.9亿文档的网络语料库和约7万条基于真实用户查询的代理重构搜索查询。

**💡 创新点**

创新点在于创建了一个包含多种语言和大量查询的基准，解决了现有基准在评估大规模检索系统时的不足，特别是针对机器生成查询的评估。

**🔧 技术方法**

使用了多种检索技术，包括词汇检索、密集检索和后交互模型，并结合了LLM（大语言模型）生成的判断。

**📊 数据集**

使用了一个包含约1.9亿文档的网络语料库，和约7万条从真实用户查询中重构的搜索查询，覆盖十种语言。

**📈 对比分析**

对13种检索器进行了基准测试，发现它们的相对排序对判断集的选择不敏感，但在主题领域、查询语言和查询类型上有显著差异。通过子语料库抽样的方法，能够在保持模型排名的同时，降低评估成本。

**⚠️ 局限性**

限制在于基准的私密性，查询和相关性判断未公开，以防止训练污染，同时也可能存在源特定偏见的风险。

---

## 644. CAST: Alternating State-Value Targets and Expanded Policy Gradients for Model-Based Reinforcement Learning

**arXiv ID:** 2609.08853 | [PDF](https://arxiv.org/pdf/2609.08853v1)

**作者:** Pietro Noah Crestaz `[一作]` (University of Trento), Andrea Del Prete `[通讯]` (University of Trento)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的模型基础强化学习方法CAST，通过使用状态值评估器替代动作值评估器，结合真实和想象的过渡来改进价值学习。

**💡 创新点**

创新点在于引入了一种正则化的混合贝尔曼目标，将真实过渡与当前策略下的想象过渡结合，从而提高样本效率和学习稳定性。

**🔧 技术方法**

使用了模型基础强化学习（MBRL）技术，结合了在线规划和状态值评估器。

**📊 数据集**

在DeepMind Control和HumanoidBench套件上进行了评估，这些数据集包含高维控制任务。

**📈 对比分析**

与多种最先进的方法（如TD-MPC2、BMPC和BOOM）进行了比较，CAST在大多数环境中表现出更高的样本效率和更快的学习速度，最终性能也具有竞争力。

**⚠️ 局限性**

限制在于CAST的性能在经过较长时间的训练后与其他方法的差距减小，可能在某些任务上不如其他方法表现突出。

---

## 645. PlannerForge: LLM Agents for Scenario-Based Testing of Motion Planners in Autonomous Driving

**arXiv ID:** 2609.08965 | [PDF](https://arxiv.org/pdf/2609.08965v1)

**作者:** Yuan Gao `[一作]` (Technical University Of Munich), Johannes Betz `[通讯]` (Technical University Of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个名为PlannerForge的框架，旨在整合自动驾驶系统的场景测试流程，包括场景生成、选择、修改、执行和评估等多个阶段。

**💡 创新点**

PlannerForge是第一个统一的全生命周期LLM框架，涵盖所有场景测试阶段，并增加了两个新阶段：自动驾驶系统增强和基准测试。

**🔧 技术方法**

使用了大型语言模型（LLM）作为核心技术，结合了模块路由、场景生成、选择、修改、测试和分析等多个模块。

**📊 数据集**

使用了来自CommonRoad的开源驾驶场景数据集，支持XML格式的场景存储和检索。

**📈 对比分析**

与Scenario Factory 2.0、BM25和From-Words-to-Collisions等方法进行比较，PlannerForge在自然语言生成、选择精度和物理有效性等方面表现更优，成功率和碰撞率也有显著改善。

**⚠️ 局限性**

存在一些限制，包括在模拟回合中某些编辑类型的查询存活率下降，以及框架在开放循环下运行，未能实现闭环验证。

---

## 646. TASTE2: Text-Aligned Speech Modeling and Deployment toward Full-Duplex Voice Interaction

**arXiv ID:** 2609.08956 | [PDF](https://arxiv.org/pdf/2609.08956v1)

**作者:** Yi-Chang Chen `[一作]` (MediaTek Research), Da-Shan Shiu `[通讯]` (MediaTek Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TASTE2模型，旨在实现全双工语音交互，支持增量对话和用户打断处理。

**💡 创新点**

创新点在于通过文本对齐的语音建模，保持语言模型的语义能力，同时实现增量对话和用户打断的处理。

**🔧 技术方法**

使用了文本对齐的语音标记化和嵌入技术，结合增量对话堆栈和流式合成。

**📊 数据集**

使用了Emilia数据集（包含40,000小时的英语自然语音）和LibriTTS数据集（600小时的阅读风格语音），以及DeepDialogue数据集（37,000个多轮对话）。

**📈 对比分析**

与现有的全双工系统（如Moshi和Freeze-Omni）相比，TASTE2在Full-Duplex-Bench v1.0上表现出更高的对用户打断的响应速度和对话连贯性，但在平滑转接的延迟上仍有待提高。

**⚠️ 局限性**

局限性包括在转接时的延迟较高，显式的副语言控制在不同属性间不一致，且在某些情感特征的控制上表现较弱。

---

## 647. SQLMorph: Query Mutation and Fine-Grained Metrics for Text-to-SQL Evaluation

**arXiv ID:** 2609.08950 | [PDF](https://arxiv.org/pdf/2609.08950v1)

**作者:** Mohammadhossein Malekpour `[一作]` (Polytechnique Montréal), Amine Mhedhbi `[通讯]` (Polytechnique Montréal)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SQLMorph框架，用于通过查询变异进行文本到SQL的评估，解决了现有评估方法的局限性。

**💡 创新点**

创新点在于引入了Join Query Expansion (JQE)和Textual Query Augmentation (TQA)两种技术，自动生成和扩展评估集，并提出了Execution Precision (EXP)和Execution Recall (EXR)等细粒度执行指标。

**🔧 技术方法**

使用了查询变异技术和细粒度评估指标，结合了执行级别的评估方法。

**📊 数据集**

使用了BIRD基准数据集，该数据集包含12,751个自然语言到SQL的配对，涵盖95个数据库和37个领域。

**📈 对比分析**

与现有的文本到SQL系统（如CHESS、DIN-SQL和MAC-SQL）进行比较，发现JQE在增加查询覆盖率的同时，随着连接数的增加，准确率下降，TQA显示出对语言变化的脆弱性，准确率下降可达17%。

**⚠️ 局限性**

局限性在于当前评估方法仍然依赖于固定的基准查询和粗糙的二元指标，无法全面捕捉系统的性能差异。

---

## 648. Disentangled Global-Local Feature Learning with E-Branchformer for Audio Deepfake Detection

**arXiv ID:** 2609.08948 | [PDF](https://arxiv.org/pdf/2609.08948v1)

**作者:** Phuong Tuan Dat `[一作]` (National University of Singapore), Nguyen Thi Thu Trang `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于E-Branchformer的架构，用于音频深度伪造检测，利用自监督语音表示进行有效检测。

**💡 创新点**

创新点在于首次将E-Branchformer架构应用于音频深度伪造检测，通过并行处理全局和局部特征来提高检测性能。

**🔧 技术方法**

使用了E-Branchformer架构，结合了多头自注意力机制和卷积处理，集成了深度卷积和Squeeze-and-Excitation模块。

**📊 数据集**

使用了ASVspoof 2021 LA、DF和In-the-Wild数据集进行实验，展示了在这些数据集上的最先进性能。

**📈 对比分析**

与现有方法相比，提出的模型在ASVspoof 2021 LA、DF和ITW数据集上的等错误率分别为0.88%、1.85%和6.30%，显著优于其他方法。

**⚠️ 局限性**

局限性在于模型的复杂性和对计算资源的需求，未来工作将探索将该架构扩展到多模态深度伪造检测和对抗攻击的鲁棒性。

---

## 649. Strengthening Proportionality in Participatory Budgeting with Additive Utilities

**arXiv ID:** 2609.08888 | [PDF](https://arxiv.org/pdf/2609.08888v1)

**作者:** Tzeh Yuan Neoh `[一作]` (Harvard University), Nicholas Teh `[通讯]` (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了参与式预算中的完全合理代表性（FJR），提出了一种在多项式时间内计算FJR的方法，适用于任意非负的加性效用和项目成本。

**💡 创新点**

创新点在于提出了一种新的算法，能够在多项式时间内实现FJR，且满足更强的分数公理FJR1+，并且可以在预算内灵活分配剩余预算。

**🔧 技术方法**

使用了扩展的剩余预算贪婪算法（Residual-Budget Greedy），并结合了线性规划技术来验证代表性。

**📊 数据集**

使用了参与式预算的实例，其中包括任意非负的加性效用和项目成本，特别是审批效用和单位成本作为特例。

**📈 对比分析**

与现有方法相比，本文的方法在计算FJR时表现出更高的效率，能够在多项式时间内完成验证，且在预算分配上具有灵活性，确保了价格可行性。

**⚠️ 局限性**

限制在于该方法在处理复杂的加性效用时可能面临挑战，尤其是在效用水平不为小整数时，算法的时间复杂度可能会增加。

---

## 650. SeGDeP: Semantic- and Geometric-Aware Decoupled Prompts for Reasoning Segmentation

**arXiv ID:** 2609.08867 | [PDF](https://arxiv.org/pdf/2609.08867v1)

**作者:** Linnan Zhao `[一作]` (Xidian University), Wenping Ma `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出了一种新的推理分割方法，将隐式语言结论转化为精确的掩码，要求同时进行语义识别和空间定位。

**💡 创新点**

创新点在于提出了一种显式的什么-哪里接口，通过语义提示分支和独立的几何投影路径，将解析后的多模态大语言模型状态转化为语义特征和DETR预测框，从而共同条件化掩码解码器。

**🔧 技术方法**

使用了多模态大语言模型（MLLM）和DETR（Detection Transformer）技术，并通过群体奖励解耦策略优化（GDPO）进行训练。

**📊 数据集**

使用了RefCOCO系列和ReasonSeg数据集进行评估。

**📈 对比分析**

与现有方法相比，论文的方法在RefCOCO系列上达到了82.7的平均cIoU，在ReasonSeg验证/测试集上分别达到了66.0/59.6的gIoU，显示出显著的性能提升。

**⚠️ 局限性**

限制在于对小型和拥挤目标的定位精度较低，导致掩码解码中的误差传播，提示需要更强的定位能力和多实例支持。

---

## 651. Fitting and Learning Basis-Restricted Propositional Formulas

**arXiv ID:** 2609.08961 | [PDF](https://arxiv.org/pdf/2609.08961v1)

**作者:** Balder ten Cate `[一作]` `[通讯]`, Balder ten Cate

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了使用有限布尔函数集O作为连接词构建的命题公式类的复杂性，分析了多种拟合和学习问题的复杂性，包括拟合给定标记样本的公式、寻找小的公式（Occam算法）、在样本不可实现时最小化错误分类示例的数量（经验风险最小化）以及几种形式的PAC学习。

**💡 创新点**

本文的创新点在于提供了一种高效构造拟合公式的算法，基于经典的Baker-Pixley构造的细化，并提出了经验风险最小化的三分定理。此外，本文还填补了一个小的空白，证明了PAC学习的二分法不仅适用于电路，也适用于公式。

**🔧 技术方法**

使用了Baker-Pixley定理的细化和图论中的超图顶点覆盖问题的已知结果。

**📊 数据集**

使用了有限的布尔函数集O作为数据集，具体的样本数据集未在文中明确列出。

**📈 对比分析**

与其他方法的比较表明，基本拟合问题在多项式时间内是均匀容易的，且对于每个固定的有限基础O，_O具有多项式时间的拟合算法。经验风险最小化问题的复杂性则依赖于O的选择，某些情况下是NP难的，而在其他情况下是多项式可解的。

**⚠️ 局限性**

本文的局限性在于对于某些布尔函数集O的选择，可能导致经验风险最小化问题的复杂性显著增加，且在某些情况下，无法保证存在有效的近似算法。

---

## 652. ContinuumBench: Benchmarking Joint Autoscaling and Placement Across Evaluation Regimes in the Cloud-Edge Continuum

**arXiv ID:** 2609.08946 | [PDF](https://arxiv.org/pdf/2609.08946v1)

**作者:** Lanpei Li `[一作]` (Institute of Information Science and Technologies Alessandro Faedo National Research Council of Italy), Massimo Coppola `[通讯]` (Institute of Information Science and Technologies Alessandro Faedo National Research Council of Italy)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ContinuumBench，一个基准测试工具，用于在云边缘连续体中评估联合自动缩放和服务放置的性能。

**💡 创新点**

创新点在于引入了完成感知的会计方法，能够将未完成和丢弃的任务视为截止日期未达成，并通过控制变量来比较不同的控制器。

**🔧 技术方法**

使用了ECLYPSE模拟器，增加了到达、工作者弹性、间歇性传输、缓冲和故障等因素，以实现控制循环。

**📊 数据集**

评估了九个控制器，涵盖四种场景和两种评估机制，使用的场景包括地球观测、车辆对一切（V2X）和工业物联网等。

**📈 对比分析**

通过控制消融实验比较了不同的控制器，结果显示，具备缩放能力的控制器在完成率、SLO违约率和延迟方面表现优于仅具放置能力的控制器。

**⚠️ 局限性**

限制在于ContinuumBench仅为模拟，缺乏硬件验证，结果是相对比较，且未考虑传输竞争、容器调度和切换动态等因素。

---

## 653. EgoSIS: From Factorized Visual Ego-Transitions to Motion-Canonical Spatial Evidence for UAV Reasoning

**arXiv ID:** 2609.08938 | [PDF](https://arxiv.org/pdf/2609.08938v1)

**作者:** Jingpu Yang `[一作]`, Yufeng Wang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为EgoSIS的适配器，用于将RGB派生的双向光流转换为运动规范视觉证据，以解决无人机视频问答中相机运动与场景变化的分离问题。

**💡 创新点**

创新点在于引入了三个模块：因子化视觉自我过渡（FVET）、可靠性门控自我过渡记忆（ReTEM）和自我对齐空间证据（EASE），实现了对运动、残余支持和可靠性的有效建模。

**🔧 技术方法**

使用了因子化视觉自我过渡（FVET）、可靠性门控自我过渡记忆（ReTEM）和自我对齐空间证据（EASE）等技术。

**📊 数据集**

在SIS-Bench数据集上进行评估，该数据集包含多种无人机视频理解任务。

**📈 对比分析**

EgoSIS-8B在SIS-Bench上获得了89.9%的感知准确率、82.5%的感知加记忆准确率和76.2%的整体准确率，相比于其他模型表现出显著的性能提升，尤其是在自我意识感知和记忆方面。

**⚠️ 局限性**

当前表示仍然是图像平面代理而非度量姿态，且需要进一步的检查点控制渐进消融实验来隔离ReTEM和EASE的贡献。

---

## 654. AuK Technical Report: An Open-Source Foundational Model for Speech Generation and Editing

**arXiv ID:** 2609.08936 | [PDF](https://arxiv.org/pdf/2609.08936v1)

**作者:** Ziyang Ma `[一作]`, Xie Chen `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种开源基础模型，统一了语音生成和编辑，通过自然语言指令和音频上下文的共同接口实现多种功能。

**💡 创新点**

创新点在于构建了一个包含约30.3亿个指令-音频实例和195万小时有效监督的训练数据集，并采用了多模态大语言模型、变分自编码器（VAE）和混合流变换器架构。

**🔧 技术方法**

使用了多模态大语言模型（MLLM）、变分自编码器（VAE）和混合流变换器（Transformer）技术。

**📊 数据集**

构建了一个包含约30.3亿个指令-音频实例的训练数据集，涵盖了语音生成、内容编辑、增强和分离、旁语言编辑和声学编辑等五个任务家族。

**📈 对比分析**

通过与现有最先进模型的性能比较，展示了在零-shot和指令控制的语音生成及一般指令引导的编辑任务中表现领先，同时在信号级恢复任务中保持竞争力。

**⚠️ 局限性**

模型在处理开放式编辑请求时的原生理解能力仍不够完善，仍需依赖显式的任务路由和提示增强。

---

## 655. When Models Defer to Wrong Answers: A Robustness Audit of Source-Attributed Cues in Multiple-Choice QA

**arXiv ID:** 2609.08934 | [PDF](https://arxiv.org/pdf/2609.08934v1)

**作者:** Manikandan Ravikiran `[一作]` (Indian Institute of Technology Mandi), Siddharth Vohra `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了语言模型在多项选择问答中，如何受到外部声称的影响，特别是当声称与问题内容相冲突时，模型是否会改变其原本的答案。

**💡 创新点**

创新点在于引入了中性条件误导提示采纳率（NC-MCAR），用于衡量模型在中性提示下选择正确答案后，是否会在误导提示下转向固定的错误选项。

**🔧 技术方法**

使用了四种遵循指令的模型进行评估，具体模型包括GPT-5.4、Claude Sonnet 4.6、Qwen3-32B和Gemma-4-31B-it。

**📊 数据集**

使用了MMLU-Pro和IndicMMLU-Pro数据集，涵盖英语、印地语、孟加拉语、泰米尔语和泰卢固语，共计220,000个输出。

**📈 对比分析**

通过比较不同的提示模板，发现专家模板的NC-MCAR为41.1%，而多数模板为12.5%。专家模板在所有模型中表现最佳，且正确提示的有效响应准确率较高。

**⚠️ 局限性**

限制在于只评估了四个模型和两种基准，结果可能不适用于显式推理、自由形式对话或其他模型和基准。NC-MCAR的目标特定切换阈值在当前结果中未报告。

---

## 656. From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection

**arXiv ID:** 2609.08899 | [PDF](https://arxiv.org/pdf/2609.08899v1)

**作者:** Mengzhe Geng `[一作]` (National Research Council Canada), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本论文提出了一种可审计的决策记录，用于语音深伪检测，保留了多个证据线索，以便在最终校准步骤中进行分析。

**💡 创新点**

创新点在于引入了一个逐句决策记录，保持了被动探测器、探针、检索和个人资料线索的对齐，而不是将它们合并为一个早期融合的分数。

**🔧 技术方法**

使用了被动探测器、条件键探针、检索支持和个人资料边际等技术，并进行了后期校准。

**📊 数据集**

使用了ASVspoof 5 Track 1开发数据集，包含4,080个语句，705个说话者。

**📈 对比分析**

与固定检索增强规则相比，后期校准的决策记录在操作分数上有所提高，达到8.43%的EER，优于固定标量融合的结果。

**⚠️ 局限性**

限制在于键探针主要通过揭示与被动和检索流的分歧来贡献，而不是作为独立的检测器。

---

## 657. OTTER - Two Transistor - One RRAM Architecture for Reliable In-Memory-Computing in 28 nm CMOS Technology

**arXiv ID:** 2609.08898 | [PDF](https://arxiv.org/pdf/2609.08898v1)

**作者:** Yang Chen `[一作]` (Forschungszentrum Jülich GmbH), Regina Dittmann `[通讯]` (Forschungszentrum Jülich GmbH)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了OTTER，一个28纳米CMOS平台，结合了基于TaO_x的电压变化机制（VCM）RRAM，展示了一种可靠的内存计算的两晶体管一忆阻器（2T1R）架构。

**💡 创新点**

创新点在于2T1R单元通过并联低驱动电流（LD）晶体管和高驱动电流（HD）晶体管，分别提供SET编程和RESET操作的专用偏置路径，从而实现了对SET和RESET电流路径的独立控制。

**🔧 技术方法**

使用了28纳米CMOS技术和物理紧凑模型JART VCM Rth进行系统的实验和模拟比较，推导出晶体管尺寸的设计指南。

**📊 数据集**

使用了基于TaO_x的RRAM设备，集成在28纳米CMOS芯片上，芯片尺寸为36 mm²，包含2T1R测试结构和15×15的CIM阵列。

**📈 对比分析**

通过对三种晶体管配对配置的系统表征，展示了2T1R架构在可靠的模拟权重编程方面的优势。LD晶体管在SET操作中提供更精细的编程粒度，而HD晶体管在RESET操作中提供可靠性。性能比较显示，LD-HD配置在多级编程中表现出更低的变异性和更紧凑的状态分布。

**⚠️ 局限性**

限制在于LD晶体管的驱动能力较低，可能在高电流条件下影响RESET的完整性，尽管2T1R架构有效解耦了SET和RESET的要求。

---

## 658. API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces

**arXiv ID:** 2609.08861 | [PDF](https://arxiv.org/pdf/2609.08861v1)

**作者:** Jennifer Wang `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对ChatGPT、Claude和Gemini等七个系统进行了审计，比较了API和用户界面在九个基准测试中的表现，挑战了基准分数与实际部署系统行为之间的假设。

**💡 创新点**

创新点在于首次大规模、受控地研究了LLM基准结果在API和聊天界面访问之间的泛化能力，揭示了API和界面评估之间的系统性差异。

**🔧 技术方法**

使用了控制审计的方法，开发了一个开源工具来程序化查询聊天平台的Web接口，并控制了请求路由、会话个性化、工具调用等混杂因素。

**📊 数据集**

使用了来自OpenLLM Leaderboard的六个基准（如ARC、GSM8K等）和针对用户风险的基准（如BBQ、AITA等），共计九个基准进行评估。

**📈 对比分析**

与API评估相比，接口评估的准确性平均低3.4个百分点，重测一致性低2.1个百分点，且不同系统和基准之间的差异显著，表明基准结果是上下文依赖的。

**⚠️ 局限性**

限制在于无法验证匹配的API和界面标识符是否总是对应于相同的基础检查点，且仅评估了单一订阅层级，可能无法推广到其他提供商或未来版本。

---

## 659. Length Generalization for Transformers via Compression

**arXiv ID:** 2609.08851 | [PDF](https://arxiv.org/pdf/2609.08851v1)

**作者:** Georg Zetzsche `[一作]` (Max Planck Institute for Software Systems), Anthony W. Lin `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文对变压器的长度泛化理论进行了深入分析，提出了C-RASP假设，并通过引入新的压缩技术（即功率词）来改进长度泛化界限。

**💡 创新点**

创新点在于通过功率词的引入，提供了一个多项式长度泛化界限，并解决了C-RASP假设中存在的矛盾实验结果。

**🔧 技术方法**

使用了功率词作为压缩技术，并结合了变压器长度泛化的理论分析。

**📊 数据集**

未具体提及使用的数据集，但提到通过实验验证了不同任务的长度泛化能力。

**📈 对比分析**

通过与现有的理论和实验结果进行比较，展示了在某些任务上变压器的长度泛化能力，尤其是在定义为C-RASP的任务上表现出色。

**⚠️ 局限性**

限制在于目前的长度泛化界限仅适用于可在_1和_中定义的任务，未来的工作将探索其他可能的片段以寻找合理的样本大小界限。

---

## 660. Curriculum Learning as Transport: Understanding Curricula with Wasserstein Geodesics

**arXiv ID:** 2609.09099 | [PDF](https://arxiv.org/pdf/2609.09099v1)

**作者:** Changho Shin `[一作]` (Princeton University), David Alvarez-Melis `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Wasserstein课程路径的框架，通过将课程表示为训练分布在离散难度级别上的轨迹，来解耦课程学习中的多个设计选择。

**💡 创新点**

创新点在于通过Wasserstein插值提供了一个统一的控制框架，使得可以逐一变化设计选择，同时保持其他因素不变，从而更好地理解课程设计的各个组成部分及其影响。

**🔧 技术方法**

使用了Wasserstein插值和最优传输理论作为分析框架，允许在固定的难度轴上清晰地分离排序、匹配曝光、端点平滑度、节奏和几何结构。

**📊 数据集**

使用了一个经过校准的合成任务套件，包括12个任务和33个难度轴，以进行控制比较。

**📈 对比分析**

与静态i.i.d.和线性课程进行比较，发现没有单一的课程策略在所有任务和预算中表现最佳。Wasserstein课程在最难级别的表现优于其他方法，尤其在固定预算下更有效。

**⚠️ 局限性**

限制在于大部分研究集中于从头训练和新构建的任务，未能充分探讨结果在预训练设置、自然数据和与先前知识交互的任务中的转移性。

---

## 661. The Surprising Effectiveness of Approximate Value Iteration in Self-Play

**arXiv ID:** 2609.09094 | [PDF](https://arxiv.org/pdf/2609.09094v1)

**作者:** Raphael Boige `[一作]`, Bruno Scherrer `[通讯]` (Université de Lorraine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究探讨了在非平凡的中等规模游戏（如Connect Four和Hex(7x7)）中，简单的近似值迭代（AVI）方法是否仍然具有竞争力。通过自我对弈实现了AVI，并使用真实的oracle进行精确评估。

**💡 创新点**

研究结果表明，AVI学习到的价值函数比AlphaZero更准确，同时其一步前瞻的贪婪策略在训练和推理成本显著较低的情况下仍然具有竞争力。这表明MCTS方法的成功可能掩盖了简单方法的潜力。

**🔧 技术方法**

使用了近似值迭代（AVI）技术，结合了自我对弈和一步negamax备份，采用神经网络作为价值函数。

**📊 数据集**

使用了Connect Four、Hex(7x7)和合成F-Games等数据集进行实验评估，并在Othello和Go(9x9)上进行了初步实验。

**📈 对比分析**

通过与AlphaZero的对比，AVI在价值误差和策略遗憾方面表现更好，且在与完美对手的对弈中表现出色。AVI的训练和推理成本显著低于AlphaZero，尽管在复杂游戏中，AVI的贪婪策略在直接对抗中略逊于MiniZero。

**⚠️ 局限性**

本研究的局限性在于，主要结果依赖于精确评估，限制了研究的广度。Othello和Go(9x9)的实验仅与固定的MiniZero基线进行比较，未能揭示任一代理距离最佳游戏的远近。

---

## 662. ToolLoop: Closed-Loop Tool-Use Data Synthesis via Decomposed Generation and Dynamic Self-Feedback

**arXiv ID:** 2609.09072 | [PDF](https://arxiv.org/pdf/2609.09072v1)

**作者:** Min Zeng `[一作]` (vivo AI Lab), Xiaoxin Chen `[通讯]` (vivo AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ToolLoop，一个闭环框架，通过分解生成过程和动态自反馈来合成高质量的工具使用数据。

**💡 创新点**

ToolLoop通过将合成过程分解为三个阶段（真实值生成、用户查询推导和工具调用实例化）并集成动态自反馈，改进了工具使用数据的合成方法。

**🔧 技术方法**

使用了动态自反馈机制和分解生成的技术，结合了LLM（大型语言模型）进行验证和生成。

**📊 数据集**

构建了一个包含11K示例的合成工具使用数据集，涵盖四种代表性的函数调用场景。

**📈 对比分析**

在Berkeley Function Calling Leaderboard (BFCL)上，ToolLoop训练的4B参数模型在非推理模式下达到了86.40%的准确率，优于其他基线模型，并在ACEBench上也表现出强大的泛化能力。

**⚠️ 局限性**

缺乏真实环境反馈，无法验证合成数据是否能有效应对实际挑战，如超时错误、格式错误的API响应或基础真值中的级联失败。

---

## 663. A Distributed Consensus Particle Filter for Target Tracking using Autonomous Surface Vessels

**arXiv ID:** 2609.09066 | [PDF](https://arxiv.org/pdf/2609.09066v1)

**作者:** Carter Noh `[一作]` (Brigham Young University), Corbin Wilhelmi `[通讯]` (US Naval Research Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种分布式共识粒子滤波器，用于在无人水面船只上进行目标跟踪，解决了在缺乏集中协调和间歇性通信情况下的多代理目标跟踪问题。

**💡 创新点**

创新点在于引入了一种粒子扩散启发式方法，使粒子在缺乏外部更新时能够沿传感器流形均匀分布，从而减少对过时测量的依赖。

**🔧 技术方法**

使用了分布式粒子滤波器和贝叶斯滤波器，结合了贝叶斯更新和共识更新机制。

**📊 数据集**

在实验中使用了SeaRobotics HYCAT无人水面船只进行实地测试，收集了目标位置的真实数据。

**📈 对比分析**

与传统的贝叶斯滤波器和不使用扩散启发式的粒子滤波器进行了比较，结果显示在正常情况下性能相似，而在通信中断的边缘情况下，扩散启发式的粒子滤波器表现更好，恢复速度更快。

**⚠️ 局限性**

限制在于该方法在几何奇异性情况下的表现可能不如预期，且需要进一步探索在不同持续时间的通信中断下的效果。

---

## 664. Task-driven Processing with Coarse-to-Fine Glimpse-based Active Perception

**arXiv ID:** 2609.09025 | [PDF](https://arxiv.org/pdf/2609.09025v1)

**作者:** Oleh Kolner `[一作]` (IBM Research), Angeliki Pantazi `[通讯]` (IBM Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于粗到细的主动感知方法CF-GAP，旨在提高实例检测的高分辨率处理能力，通过选择性地聚焦于相关区域来改善性能。

**💡 创新点**

CF-GAP的创新点在于其生物启发的粗到细处理方案，能够根据任务信息引导视觉处理，显著提高了现有实例检测器的性能，尤其是在复杂和拥挤的场景中。

**🔧 技术方法**

使用了粗到细的视图选择技术，结合了低分辨率和高分辨率的图像处理方法，采用了基于log-polar的传感器进行细致的视觉信息提取。

**📊 数据集**

在HR-InsDet和Robotools两个数据集上进行了评估，HR-InsDet包含100个对象实例和160个高分辨率场景，Robotools包含20个对象实例和1581个测试图像。

**📈 对比分析**

与现有的基线模型（如OTS-FM和NIDS-Net）相比，CF-GAP在所有模型上均显著提高了平均精度（AP），在复杂场景中提升幅度可达20%。

**⚠️ 局限性**

CF-GAP的局限性包括仅依赖于基于纹理的引导，可能导致对无关区域的重复访问；固定大小的RoI可能不适应不同大小的对象；每次粗视图后调用下游架构的计算成本较高。

---

## 665. Evaluation of Contextual Understanding in Large Language Models

**arXiv ID:** 2609.09004 | [PDF](https://arxiv.org/pdf/2609.09004v1)

**作者:** Subavarshana Arumugam `[一作]` (University of Moratuwa), Kamal Premaratne `[通讯]` (University of Miami)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于知识图谱的评估框架，旨在评估大型语言模型（LLMs）在上下文理解方面的能力，特别是在问答任务中。

**💡 创新点**

创新点在于引入了语义结构相似性（S3KG）作为一种混合相似性度量，结合了结构和语义相似性，并提供了一个诊断框架来分类推理错误。

**🔧 技术方法**

使用了知识图谱（KG）构建和语义结构相似性（S3KG）技术，结合了SBERT进行相似性计算。

**📊 数据集**

使用了两个数据集：PubMedQA（生物医学问答对）和MesaQA（消费者医疗问答对），这两个数据集涵盖了长文本答案。

**📈 对比分析**

与传统的评估指标（如ROUGE、BLEU等）相比，S3KG在9个数据集上表现优异，7个数据集的F1得分位于前两名，尤其在KG扰动段落中，S3KG的F1得分提高了7.6点，显示出其在捕捉关系角色区分方面的优势。

**⚠️ 局限性**

局限性在于KG提取质量仍然是主要瓶颈，未来的工作将考虑使用更先进的模型（如GPT-4）和基于注意力的方向性对齐，以更好地区分语义相反的关系。

---

## 666. A Sublinear Approximation Algorithm for Minimum Dilation Trees in the Plane

**arXiv ID:** 2609.08990 | [PDF](https://arxiv.org/pdf/2609.08990v1)

**作者:** Sarita de Berg `[一作]` (IT University of Copenhagen), Sampson Wong `[通讯]` (University of Copenhagen)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种在欧几里得平面中计算最小膨胀树的子线性近似算法，解决了1996年Eppstein提出的开放问题。

**💡 创新点**

首次提供了一个O(n^14/15)的近似算法，并且该算法在多项式时间内运行。

**🔧 技术方法**

使用了几何图和树的结构分析技术，结合了新的见解来识别点集的结构。

**📊 数据集**

使用了任意点集的几何图，具体数据集未明确给出。

**📈 对比分析**

与最小生成树（MST）进行比较，MST是O(n)-近似，而该算法提供了O(n^14/15)的近似，性能显著提升。

**⚠️ 局限性**

算法的局限性在于常数因子较大，实际应用中可能不够高效，且在高维空间的推广仍然是一个开放问题。

---

## 667. GoDeep: Annotation-Free Open-Vocabulary 3D Scene Understanding via Language-Space Lifting

**arXiv ID:** 2609.09082 | [PDF](https://arxiv.org/pdf/2609.09082v1)

**作者:** Thodoris Betsas `[一作]` (National Technical University Of Athens), Andreas Georgopoulos `[通讯]` (National Technical University Of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无注释的3D场景理解管道GoDeep，通过将结构化的视觉语言模型（VLM）描述提升到纯句子嵌入空间，避免了对3D训练语料库或专用3D编码器的需求。

**💡 创新点**

创新点在于将VLM生成的描述直接提升到语言空间，而不是依赖于CLIP的联合视觉-语言空间，从而实现了更高的物理内容追踪精度。

**🔧 技术方法**

使用了视觉语言模型（如Qwen2-VL）作为翻译器，生成每个图像的结构化描述，并通过开放词汇分割器（SAM）将描述与3D点云进行关联。

**📊 数据集**

在ScanNet++（100类基准）和一个包含5栋历史建筑的文化遗产数据集上进行了评估。

**📈 对比分析**

与强大的无注释基线进行比较，GoDeep在ScanNet++上表现出竞争力，尽管在更大多数据集上训练的方法表现更好。在文化遗产数据集上，通过纠正单一的词汇不匹配，GoDeep的表现超过了CLIP基于的变体。

**⚠️ 局限性**

限制在于每张图像查询VLM和LLM的计算成本较高，且表示在内存上较为密集，尤其是使用规模感知多向量聚合时，存储需求显著增加。

---

## 668. Parity and Pattern Detection in Permutation Streams

**arXiv ID:** 2609.09064 | [PDF](https://arxiv.org/pdf/2609.09064v1)

**作者:** Mark Braverman `[一作]` (Princeton University), Or Zamir `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在流模型中处理排列的空间复杂性，特别是计算排列的奇偶性和检测长度为三的排列模式。

**💡 创新点**

提出了对于长度为三的每种排列模式，能够在一个传递中使用O(log n)位的空间进行确定性检测，同时证明了计算排列的奇偶性需要Θ(n)位的空间。

**🔧 技术方法**

使用了流模型和随机化算法，结合了通信复杂性理论和排列的性质。

**📊 数据集**

研究中使用了[n]的排列作为输入，具体的输入数据集未详细说明。

**📈 对比分析**

与Berendsohn的2026年下界结果相结合，完成了固定排列模式的分类。对于单调模式和长度不超过三的模式，最优空间复杂度为Θ(log n)，而对于其他模式则为Θ(n)。

**⚠️ 局限性**

研究中未详细讨论限制条件，但提到对于长度大于三的非单调模式，空间复杂度的下界为Ω(n)，这可能是一个限制。

---

## 669. Multi-Task Learning for Sparsely-Labeled Time Series: A Case Study on Cold-Hardiness Modeling

**arXiv ID:** 2609.09062 | [PDF](https://arxiv.org/pdf/2609.09062v1)

**作者:** Aseem Saxena `[一作]` (Oregon State University), Alan Fern `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种多任务学习（MTL）方法，用于从有限的时间序列数据中建模冷硬度，特别是针对葡萄的冷硬度预测问题。

**💡 创新点**

创新点在于通过多任务学习结合不同葡萄品种的数据，克服了数据稀疏性的问题，从而提高了冷硬度和芽萌发的预测准确性。

**🔧 技术方法**

使用了递归神经网络（RNN）作为基础模型，并开发了多种MTL架构，包括多头模型和任务嵌入模型。

**📊 数据集**

使用的数据集包括来自多个葡萄品种的冷硬度和芽萌发的实测数据，数据收集自1980年代以来的华盛顿州。

**📈 对比分析**

与单任务学习（STL）和现有的科学模型相比，MTL模型在多个葡萄品种上表现出显著的性能提升，尤其是在数据稀疏的情况下，MTL模型的表现优于STL和现有的科学模型。

**⚠️ 局限性**

限制在于当前的MTL模型可能对不同任务之间的负面交互敏感，且在某些情况下，任务嵌入模型的迁移学习效果不佳。

---

## 670. Location-Independent Robot-Assisted Finishing Using Digital Twins and Extended Reality

**arXiv ID:** 2609.09061 | [PDF](https://arxiv.org/pdf/2609.09061v1)

**作者:** Jose Outeiro `[一作]` (University of North Carolina at Charlotte), Khalil Chakal `[通讯]` (University of Oulu)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种网络物理系统（CPS），用于无位置编程、监督、培训和远程操作机器人辅助精加工（RAF）系统，以对金属增材制造（AM）组件进行后处理。

**💡 创新点**

创新点在于结合了人类决策和通过数字双胞胎（DT）进行的远程操作，利用扩展现实（XR）作为人机界面，实现低批量生产的经济可行性，同时减少操作员暴露于恶劣和嘈杂环境中的时间。

**🔧 技术方法**

使用了Unity构建的数字双胞胎和MQTT通信协议，支持远程操作和监控。

**📊 数据集**

验证在一个专门设计的物理RAF系统上进行，系统包括一个六自由度的协作机器人和一个离心盘精加工机。

**📈 对比分析**

与现有系统比较，本文的CPS实现了位置独立的远程操作、沉浸式跨平台接口和特定于精加工过程的反馈，性能指标显示最大稳态关节同步误差为0.12°，平均往返延迟为563毫秒，适合监督编程和间歇性远程操作。

**⚠️ 局限性**

限制在于系统尚未完全仪器化，缺乏对精加工过程的力、振动和声学信号的实时监测，且当前的延迟不支持连续远程操作。

---

## 671. The Audit Decides the Verdict: Instrument Effects Rival Demographic Bias in LLM Decision Audits

**arXiv ID:** 2609.09048 | [PDF](https://arxiv.org/pdf/2609.09048v1)

**作者:** Siddharth Vohra `[一作]` (Carnegie Mellon University), Manikandan Ravikiran `[通讯]` (Indian Institute of Technology Mandi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究测试了语言模型在招聘、贷款和医疗分诊中的偏见，特别是如何通过不同的审计格式影响模型的评估结果。

**💡 创新点**

创新点在于揭示了审计格式对模型评估结果的影响，表明审计的构造比人口统计偏见更能影响模型的判断。

**🔧 技术方法**

使用了多种语言模型，包括GPT-5.6 Terra、Claude Sonnet 5、Gemini 3.1 Pro等，采用了混合效应回归分析方法。

**📊 数据集**

使用了来自AgentFairBench的公共数据集，包含招聘、贷款和医疗分诊的12个无种族偏见的基本档案。

**📈 对比分析**

与以往研究相比，本研究未发现显著的种族或性别偏见，所有36个计划对比均未通过校正，结果表明审计格式的影响大于人口统计偏见。

**⚠️ 局限性**

限制在于仅使用姓名作为人口统计信号，可能无法转移到其他形式的种族和性别表达，且样本量较小，无法代表所有模型的普遍情况。

---

## 672. Deterministic Edge-Fault-Tolerant Connectivity Labeling Schemes with Nearly Optimal Label Size

**arXiv ID:** 2609.09031 | [PDF](https://arxiv.org/pdf/2609.09031v1)

**作者:** Yaowei Long `[一作]` (University of Michigan), Thatchaphol Saranurak `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种边故障容错连接标记方案，为无向图中的顶点和边分配短标签，以便在给定的故障边集下，能够通过检查标签来判断两个顶点之间的连接性。

**💡 创新点**

该方案首次实现了O(log^2n)位标签的确定性标记，且在多项式时间内可计算，显著改善了之前的确定性界限，并在某些情况下优于随机界限。

**🔧 技术方法**

结合了基于循环空间的标记方案和稀疏循环基的最新结果，采用了确定性算法来计算稀疏循环基。

**📊 数据集**

使用了无向图G=(V,E)作为数据集，图的顶点数和边数分别为n和m。

**📈 对比分析**

与之前的随机标记方案相比，该方案在标签大小和查询时间上都有所改进，查询时间为O(|F|^3log n)，且保证了全查询的正确性。

**⚠️ 局限性**

该方案的局限性在于，尽管在标签大小和计算时间上有所改善，但仍然依赖于稀疏循环基的存在性，且在某些情况下可能无法达到最优性能。

---

## 673. Physics-Informed Deep Learning for False Ventricular Tachycardia Alarm Reduction in the ICU

**arXiv ID:** 2609.08992 | [PDF](https://arxiv.org/pdf/2609.08992v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 674. A Joint 2D-3D Statistical Shape Model for Orthopedic Reconstruction

**arXiv ID:** 2609.09010 | [PDF](https://arxiv.org/pdf/2609.09010v1)

**作者:** Florence Dell'Aniello Picard `[一作]` (Polytechnique Montréal), Herve Lombaert `[通讯]` (Polytechnique Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种联合2D-3D统计形状模型，用于从X光图像中重建三维股骨形状，以支持手术规划和植入物尺寸选择。

**💡 创新点**

创新点在于通过联合建模2D和3D分割，直接从数据中学习2D到3D的映射，消除了迭代优化的需求。

**🔧 技术方法**

使用了主成分分析（PCA）和静态速度场（SVF）来捕捉2D和3D分割之间的共同变化。

**📊 数据集**

使用了新墨西哥死者图像数据库（NMDID）中的1368个CT扫描数据集，包含781名个体的下肢数据。

**📈 对比分析**

与传统的3D统计形状模型（SSM）相比，提出的方法在重建精度上有所提升，DSC从94.96%提高到96.15%，推理时间约为3秒，速度是传统方法的四倍。

**⚠️ 局限性**

当前方法假设已知且固定的校准，未来的工作可以扩展到未校准的设置，以提高适用性。

---

## 675. Let It Go or Learn to Self-Correct: Continuous Diffusion for Constrained Discrete Tasks

**arXiv ID:** 2609.09009 | [PDF](https://arxiv.org/pdf/2609.09009v1)

**作者:** Mariia Drozdova `[一作]` (University of Geneva), François Fleuret `[通讯]` (University of Geneva)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了去噪扩散概率模型（DDPM）在全局约束离散任务（如数独、图连通性、拉丁方和N皇后）中的表现，提出了通过直接从模型的干净预测中采样来改善标准采样器的有效性。

**💡 创新点**

创新点在于提出了Tweedie重投影方法，通过去除与当前状态的直接依赖，显著提高了约束满足率。此外，提出了自我修正训练，增强了模型对推理过程中错误的鲁棒性。

**🔧 技术方法**

使用了去噪扩散概率模型（DDPM）和自我修正训练技术。

**📊 数据集**

使用的数据集包括数独、图连通性、拉丁方和N皇后等约束离散任务的基准数据集。

**📈 对比分析**

与标准采样器相比，Tweedie重投影在数独有效性上从31%提高到95%。自我修正训练进一步提高了DDPM的有效性，从31%提升至87%。

**⚠️ 局限性**

限制在于Tweedie重投影可能在感知丰富的领域中不适用，因为它可能会丢失重要的细节信息。此外，自我修正训练仅部分解决了训练与推理之间的不匹配，长采样轨迹仍可能访问未在训练中充分表示的状态。

---

## 676. Concentrate After Imagination: Text-Conditioned Evidence Grounding for Partially Relevant Video Retrieval

**arXiv ID:** 2609.08999 | [PDF](https://arxiv.org/pdf/2609.08999v1)

**作者:** Shuaiqi Cheng `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为TRACE的分数级证据验证操作符，用于部分相关视频检索（PRVR），旨在解决查询与视频内容不完全匹配的问题。

**💡 创新点**

TRACE通过激活与查询相关的全局寄存器，并将其支持转化为帧级证据，从而减少不支持的局部峰值的影响，提供了一种新的证据验证机制。

**🔧 技术方法**

使用了层次化的证据集中操作，结合了查询条件的寄存器激活和跨粒度证据路由技术。

**📊 数据集**

在ActivityNet Captions、Charades-STA和TVR数据集上进行了评估。

**📈 对比分析**

TRACE在所有三个基准上都达到了最佳的SumR，并且相较于DreamPRVR分别提高了+1.2、+1.1和+1.5，显示出其在检索性能上的显著提升。

**⚠️ 局限性**

TRACE的局限性在于其依赖于查询条件的全局证据验证，可能在某些情况下无法完全消除不支持的局部峰值。

---

## 677. On APN Functions with Boomerang Uniformity One over $\mathbb F_{3^n}$: Differential and Boomerang Spectra and CCZ-Inequivalence

**arXiv ID:** 2609.08968 | [PDF](https://arxiv.org/pdf/2609.08968v1)

**作者:** Namhun Koo `[一作]` (Sungkyunkwan University), Byunguk Kim `[通讯]` (Sungkyunkwan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构造了一类基于Dembowski–Ostrom多项式的几乎完美非线性（APN）函数，证明了这些函数的差分谱和回旋谱的性质。

**💡 创新点**

创新点在于首次提供了一个通用构造，生成无限个具有回旋均匀性为1的APN函数族，并且这些函数的差分谱和回旋谱是完全确定的。

**🔧 技术方法**

使用了Dembowski–Ostrom多项式和线性化导数等技术，结合了差分和回旋性质的分析。

**📊 数据集**

数据集为有限域_3^n，其中n为奇数且大于1。

**📈 对比分析**

与其他方法的比较显示，本文构造的APN函数在回旋均匀性上达到了最优值1，且与每个幂函数和Ness–Helleseth类型的二项式在CCZ等价上是不同的。

**⚠️ 局限性**

限制在于该构造依赖于Dembowski–Ostrom性质，可能无法直接推广到其他类型的多项式或更广泛的情形。

---

## 678. PrivEscalate: Measuring and Augmenting the Threat of LLM-Automated Linux Privilege Escalation

**arXiv ID:** 2609.09087 | [PDF](https://arxiv.org/pdf/2609.09087v1)

**作者:** Yixuan Liu `[一作]` (Nanyang Technological University), Yi Li `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PrivEscalate，一个大规模的Linux权限提升基准，包含531个Docker化场景，涵盖14个子类别，并通过329个参数化变体场景来测试模型的环境敏感性。

**💡 创新点**

创新点在于构建了一个系统化的、可重复的评估框架，解决了现有评估中样本量小、覆盖面不足和环境干扰敏感性未测试的问题。

**🔧 技术方法**

使用了Docker化的环境构建和多代理构建管道，结合了LLM（大语言模型）进行自动化评估。

**📊 数据集**

使用了531个经过审核的Docker化Linux权限提升场景，并生成了329个参数化变体场景。

**📈 对比分析**

与六个LLM模型在三种代理架构下进行比较，发现模型在不同漏洞类别上的能力不均衡，且环境扰动对LLM的成功率有显著影响。PrivEscAgent在不修改底层LLM的情况下，提升了所有六个模型的成功率。

**⚠️ 局限性**

限制在于基准测试未涵盖内核漏洞、NFS根压缩和服务特定漏洞，且统计范围受限于样本的多样性和环境变化的影响。

---

## 679. Rethinking Learned Occupancy in Autonomous Active Mapping with Observation-Gated Filtering

**arXiv ID:** 2609.09069 | [PDF](https://arxiv.org/pdf/2609.09069v1)

**作者:** Jiahui Zhang `[一作]` (Boise State University), Yu Zhang `[通讯]` (Boise State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了自主3D主动映射中，机器人如何选择感知位置并构建导航所需的几何图形，提出了一种观察门控过滤器来改善占用预测的准确性。

**💡 创新点**

创新点在于提出了一种观察门控过滤器，该过滤器在未观察到的区域保留占用预测，并在缺乏支持的情况下抑制预测，从而提高了规划的有效性。

**🔧 技术方法**

使用了观察门控过滤器技术，该技术通过对重复观察的支持进行分类来动态调整占用预测。

**📊 数据集**

使用了Macarons++数据集，包括五个场景和每个场景的五个固定起始点，共进行100个规划动作和101个相机姿态的实验。

**📈 对比分析**

通过控制实验比较了不同条件下的占用预测对闭环覆盖的影响，结果表明，地面真实占用的使用在效率上有显著提升，但最终覆盖率的提高有限。

**⚠️ 局限性**

限制在于当前研究假设了基准RGB-D观察和足够准确的姿态估计，未评估行星感知条件和累积定位漂移的影响。

---

## 680. Training-Free Task Vectors for LLM Behavioral Control

**arXiv ID:** 2609.09054 | [PDF](https://arxiv.org/pdf/2609.09054v1)

**作者:** Gabriel J. Perin `[一作]` (University of Sao Paulo), Nina S. T. Hirata `[通讯]` (University of Sao Paulo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的方法，称为无训练任务向量（TFTVs），用于在不需要微调的情况下计算任务向量类似的方向，从而实现模型的后训练编辑。

**💡 创新点**

创新点在于TFTVs能够仅通过前向传播统计信息来识别语义上有意义的行为方向，而不需要依赖于微调过程。

**🔧 技术方法**

使用了前向传播统计信息和对比提示来构建无训练任务向量，并通过映射激活引导向量到权重空间的编辑。

**📊 数据集**

在大型语言模型的行为控制任务上进行了评估，主要关注特征如邪恶、幻觉和谄媚。

**📈 对比分析**

与其他编辑和引导基线方法进行比较，TFTVs在特征控制方面表现更强，同时保持或改善了效用的保留。

**⚠️ 局限性**

限制在于TFTV的性能依赖于被编辑的模块和层，因此自动选择模块是未来研究的重要方向。此外，TFTVs并未消除特征控制与效用之间的权衡，尤其是在更强或组合的编辑下。

---

## 681. Time-Varying Data as Sheaves: an Invitation to Narratives

**arXiv ID:** 2609.09056 | [PDF](https://arxiv.org/pdf/2609.09056v1)

**作者:** Wilmer Leal `[一作]` (University of Florida), Warren Dixon `[通讯]` (University of Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文提出了一种叙事理论，作为时间变化数据的抽象框架，旨在支持理论研究和应用。通过三个小节，探讨了时间变化数据的不同研究方向，包括信息损失、结构复杂性测量和多智能体系统的建模。

**💡 创新点**

创新点在于将时间变化数据视为层叠体（sheaves），并通过叙事理论提供了一个统一的框架，能够跨越不同数学领域进行研究和应用。

**🔧 技术方法**

使用了层叠体和共层叠体的数学工具，结合范畴论的方法来描述时间变化数据的结构和性质。

**📊 数据集**

论文中没有具体提到使用的数据集，而是通过理论构建和示例来说明叙事理论的应用。

**📈 对比分析**

通过比较持久叙事和累积叙事之间的关系，展示了在不同视角下信息的保留情况。性能方面，论文强调了在转换视角时可能会丢失信息，并探讨了如何通过适当的数学结构来最小化这种损失。

**⚠️ 局限性**

限制在于叙事理论的应用可能受到特定数学结构的限制，且在不同领域的适用性可能需要进一步的验证和调整。

---

## 682. Do Reasoning Representations Help Humans Evaluate LLM Outputs?

**arXiv ID:** 2609.09038 | [PDF](https://arxiv.org/pdf/2609.09038v1)

**作者:** Jaewoo Lim `[一作]` (Oregon State University), Sanghyun Hong `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了推理表示作为人类评估大型语言模型输出的接口，而非仅仅作为模型推理能力的指标。通过对六种推理格式进行控制的人类研究，评估其在不同复杂任务中的有效性。

**💡 创新点**

创新点在于将推理表示重新框定为人类评估的接口，设计了一个控制的人类评估协议，以比较不同推理表示在结构理解、错误检测和信任校准方面的表现。

**🔧 技术方法**

使用了基于网络的评估框架，随机化任务领域、问题实例和表示顺序，结合了正确答案和注入错误的追踪，以隔离表示的效用。

**📊 数据集**

使用了三个基准数据集：GSM8K（多步算术推理）、HotPotQA（多跳事实问答）和BBH（符号和逻辑推理），共27个问题。

**📈 对比分析**

与模型中心的评估方法相比，本研究发现参与者偏好规划和分解基础的表示，但更简单的链式思维追踪在错误检测和定位方面表现更好。偏好的表示也引入了校准风险，导致在正确追踪上出现更多误报。

**⚠️ 局限性**

限制在于本研究评估的是预生成的LLM输出，而非在交互会话中产生的推理追踪，这可能限制了用户可用的交互形式。此外，所有追踪均由单一模型生成，未评估结果是否适用于其他模型。

---

## 683. Travel Package Booking Application with API Bot

**arXiv ID:** 2609.09112 | [PDF](https://arxiv.org/pdf/2609.09112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 684. PIC: Revisiting INR for Image Coding with Fast Encoding and Sub-Millisecond Decoding

**arXiv ID:** 2609.09020 | [PDF](https://arxiv.org/pdf/2609.09020v1)

**作者:** Xiang Liu `[一作]` (Tsinghua University), Shu-tao Xia `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的图像编码框架，称为实用隐式神经表示图像编解码器（PIC），通过单次前向传递生成所有网络参数，从而实现比以往基于表示的方法更快的编码速度。

**💡 创新点**

创新点在于同时实现快速编码、超快解码和具有竞争力的率失真（RD）性能，首次在学习型图像编解码器中实现了与JPEG在RD性能和解码速度上的比较或超越。

**🔧 技术方法**

使用了前馈神经网络架构，结合了高效的熵估计模块和优化的解码器，充分利用了硬件加速特性。

**📊 数据集**

使用了LSDIR数据集进行训练，并在Kodak和CLIC数据集上进行评估。

**📈 对比分析**

与多种代表性方法（如Factorized、Hyperprior、COIN、Cool-Chic v4.2和GaussianImage）进行了比较，结果显示PIC在解码速度上显著优于JPEG，并在高比特率区域的PSNR表现上优于其他方法。

**⚠️ 局限性**

限制在于RD性能仍有提升空间，且当前方法在某些低比特率场景下的表现不如其他先进方法。

---

## 685. Spheriverse: 3D Scene Understanding from Spherical Observations in the Wild

**arXiv ID:** 2609.09012 | [PDF](https://arxiv.org/pdf/2609.09012v1)

**作者:** Fei Teng `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了Spheriverse，一个包含644个时序对齐的球形图像-LiDAR配对的数据集，旨在解决球形观察与笛卡尔坐标之间的表示差距，以支持3D场景理解。

**💡 创新点**

创新点在于提出了SphereOcc框架，该框架结合了球形几何建模和语义证据检索，显著提高了密集语义占用预测的性能。

**🔧 技术方法**

使用了球形几何建模、语义证据重查询（SER）和笛卡尔-球形表示重塑（CSRR）等技术。

**📊 数据集**

数据集Spheriverse包含644个序列，覆盖13个地理和视觉多样的区域，具有丰富的语义注释，包含24小时的真实场景数据。

**📈 对比分析**

通过对30多种方法的整体和场景比较，SphereOcc在密集占用预测中达到了13.91%的mIoU和24.65%的GeoIoU，分别比最佳先前方法TPVFormer和SurroundOcc提高了1.70和2.10个百分点。

**⚠️ 局限性**

限制在于尽管SphereOcc在多种场景中表现优异，但在某些特定情况下（如低光照和复杂几何结构）仍可能面临挑战。

---

## 686. Factorized and Vectorized Execution: Optimizing Analytical and Semantic Queries over Relations

**arXiv ID:** 2609.09002 | [PDF](https://arxiv.org/pdf/2609.09002v1)

**作者:** Sunny Yasser `[一作]` (Polytechnique Montréal), Amine Mhedhbi `[通讯]` (Polytechnique Montréal)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的查询引擎FFX，用于快速因子化执行，旨在优化分析和语义查询，特别是多对多连接的查询。

**💡 创新点**

FFX是第一个支持任意因子化方案的流水线引擎，同时保持完全向量化，能够有效执行连接密集的分析查询。

**🔧 技术方法**

使用了因子化向量和操作符，维护缓存友好的连续布局，并引入了级联更新操作符以在因子化层次结构中传播更新。

**📊 数据集**

使用了来自Open Graph Benchmark、Twitter爬虫和SNAP集合的数据集，涵盖社交、网络、产品和引用等多个领域。

**📈 对比分析**

与现有的向量化引擎（如DuckDB和Kuzu）进行比较，FFX在重连接工作负载上实现了显著的速度提升，同时在语义操作的准确性上保持了可比性或有所改善。

**⚠️ 局限性**

FFX在处理某些查询时可能会遇到性能下降，尤其是在因子化带来的好处有限的情况下，尽管它能够优雅地回退到标准的向量化执行。

---

## 687. It's Not RoPE that Creates Sinks: The Role of Self-Concentration and Value-Non-Mixing in Attention

**arXiv ID:** 2609.09085 | [PDF](https://arxiv.org/pdf/2609.09085v1)

**作者:** Raito Kiya `[一作]` (Tohoku University), Goro Kobayashi `[通讯]` (Tohoku University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究分析了大型语言模型（LLMs）在序列初始位置出现的“注意力沉没”和“巨大激活”现象，探讨了这些现象的成因。

**💡 创新点**

研究提出了自我集中注意力和注意力输出中的值非混合状态是导致这些现象的关键因素，提供了新的实证证据。

**🔧 技术方法**

使用了注意力机制的分析方法，特别是自我集中和值非混合状态的实验设计。

**📊 数据集**

使用了WikiText数据集，并对多种模型进行了实验，包括Llama-3.2-3B等。

**📈 对比分析**

通过与其他模型的比较，发现自我集中和值非混合状态在不同模型中均能引发注意力沉没现象，且在大多数模型中效果显著。

**⚠️ 局限性**

研究的局限性包括只针对特定模型和数据集，未能探讨其他架构的适用性，以及未能排除其他潜在因素的影响。

---

## 688. Measuring LLM Sycophancy under Sustained Multi-Turn Pressure

**arXiv ID:** 2609.09090 | [PDF](https://arxiv.org/pdf/2609.09090v1)

**作者:** Leyuan Tang `[一作]` (Texas A&M University), Ruihong Huang `[通讯]` (Texas A&M University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SPINE基准，评估大型语言模型在持续用户反对下是否能保持正确立场，模拟了一个持续但错误的用户进行对话挑战。

**💡 创新点**

创新点在于引入了适应性用户代理，进行长达25轮的多轮对话评估，分析模型的推理轨迹以揭示其屈从行为。

**🔧 技术方法**

使用了大型语言模型（LLM）作为适应性用户代理，结合了多轮对话和推理轨迹分析技术。

**📊 数据集**

使用了CREPE和StereoSet数据集，分别用于识别错误前提和挑战不道德请求，共计200个项目。

**📈 对比分析**

与现有方法相比，SPINE能够捕捉到在持续压力下模型的崩溃率随着对话轮数的增加而增加，短期评估低估了模型的屈从性。

**⚠️ 局限性**

限制在于每个场景的测试库仅限于100个项目，且所有判决均由同一LLM模型生成，可能存在偏差。

---

## 689. Everything in Moderation: Per-Domain Coverage Optima and Alignment-Resistant Domain Gaps in Multi-Domain Mid-Training

**arXiv ID:** 2609.09081 | [PDF](https://arxiv.org/pdf/2609.09081v1)

**作者:** Yunpeng Xu `[一作]`, Kun Zheng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了中期训练阶段的域数据覆盖对模型性能的影响，特别是数据分配的设计选择是否会影响后续的对齐过程。

**💡 创新点**

研究发现每个域都有一个内部覆盖最优值，适度的覆盖（10-40%）对所有五个域的性能最佳，且覆盖引起的差距在固定预算的对齐过程中难以弥补。

**🔧 技术方法**

使用了Qwen3-8B-Base模型进行逻辑推理的控制实验，结合了监督微调（SFT）和强化学习（RL）等技术。

**📊 数据集**

数据集使用了KOR-Bench，包含五个语义规则不相交的逻辑推理域，并进行了多种覆盖配置的训练。

**📈 对比分析**

与其他方法的比较显示，补偿性SFT在提高准确性方面表现有限，且在240对域之间的差距中，几乎没有弥补成功的案例（0/240对在5 pp阈值下弥补）。

**⚠️ 局限性**

研究的局限性在于覆盖分配的零和特性使得每个域的比例无法独立变化，且未能验证拟合的最优值是否真正可实现。

---

## 690. ThinkPrior: Zero-Rollout Difficulty Priors for Cold-Start Prompt Selection in RLVR

**arXiv ID:** 2609.09075 | [PDF](https://arxiv.org/pdf/2609.09075v1)

**作者:** Tommy Sha `[一作]` (Stony Brook University), Siqi Zhao `[通讯]` (University of Minnesota Twin Cities)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在可验证奖励的强化学习（RLVR）中，如何通过零回报难度先验来优化冷启动提示选择，从而减少无效的回报优势梯度。

**💡 创新点**

创新点在于提出了一种零回报难度先验，通过外部锚点的离线评分来初始化提示的难度估计，从而在首次选择提示之前避免了冷启动问题。

**🔧 技术方法**

使用了群体相对策略优化（GRPO）和贝塔后验等技术，结合外部锚点进行提示选择。

**📊 数据集**

使用了Qwen2.5-Math-7B数据集，该数据集包含250个数学问题，涵盖五个级别和七个主题。

**📈 对比分析**

与其他方法比较时，本文的方法在早期训练中将无效组的比例从23.8%降低到10.6%，并减少了通过30步的浪费回合数19%，而最终准确率没有显著变化。

**⚠️ 局限性**

限制在于固定预算的结果是重新分配，而不是净节省；只有测量的+DAPO组合显示出净生成减少。

---

## 691. Online, Reachability-Aware, Sampling-Based Motion Planning

**arXiv ID:** 2609.09073 | [PDF](https://arxiv.org/pdf/2609.09073v1)

**作者:** Brendan Gould `[一作]`, Samuel Coogan `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在线、可达性感知的基于采样的运动规划方法，旨在为安全关键的机器人系统提供实时的安全控制。

**💡 创新点**

创新点在于通过快速的区间基础管道在线计算可达集的过度近似，从而消除了昂贵的预计算步骤，并在不牺牲性能的情况下提供了安全保证。

**🔧 技术方法**

使用了基于区间的可达性分析和模型预测控制（MPC）技术。

**📊 数据集**

在模拟和真实硬件实验中使用了1/28比例的赛车平台进行验证。

**📈 对比分析**

与现有的基于采样的规划器进行了双向比较，结果显示该算法在没有预计算的情况下实现了类似的性能，并且在赛车模拟中将安全违规减少了99%以上。

**⚠️ 局限性**

局限性在于算法可能无法找到安全控制序列，即使存在安全控制；此外，算法不具备递归可行性保证，可能导致在预测范围外发生安全违规。

---

## 692. Performance of Clinical AI System and Physicians and Frontier Language Models in primary care diagnostics

**arXiv ID:** 2609.09070 | [PDF](https://arxiv.org/pdf/2609.09070v1)

**作者:** Andy Nkansah `[一作]` (A.I. Doctor Medical Assist LTD), Pavel Satalkin `[通讯]` (A.I. Doctor Medical Assist LTD)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究比较了Doctorina临床AI系统、八名医生和四个独立的前沿语言模型在150个合成的波兰语初级护理咨询中的表现。

**💡 创新点**

研究的创新点在于将Doctorina与医生和其他语言模型进行直接比较，评估其在适应性咨询中的诊断和管理能力。

**🔧 技术方法**

使用了Doctorina作为临床AI系统，结合了特定任务的指令、代理协调和咨询状态管理，支持多轮适应性咨询。

**📊 数据集**

使用了150个合成的波兰语初级护理案例，这些案例是根据2024年国家健康基金服务报告数据构建的。

**📈 对比分析**

与医生的比较中，Doctorina的Top-1一致性为82.0%，而医生为57.0%，差异为25.0个百分点。Doctorina在诊断和管理评分上均优于医生，并且在与其他语言模型的比较中也表现出色。

**⚠️ 局限性**

研究的局限性包括样本的非随机性和医生的选择偏差，可能限制了结果的广泛适用性。此外，研究未涵盖常规临床工作流程或患者结果，需进行前瞻性验证。

---

## 693. It Is Not My Code Anymore

**arXiv ID:** 2609.09022 | [PDF](https://arxiv.org/pdf/2609.09022v1)

**作者:** Augusto Camargo `[一作]` `[通讯]`, Augusto Camargo

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文讨论了在AI辅助软件生产中，作者、责任和评估的问题，使用了一个假设的失败案例来探讨这些概念。

**💡 创新点**

创新点在于分析了生成模型对代码归属和责任的影响，提出了在软件生产中如何理解和分配责任的问题。

**🔧 技术方法**

使用了生成模型（LLM）作为软件实现的生产者，并探讨了人类与机器之间的协作关系。

**📊 数据集**

没有使用具体的数据集，而是基于文献回顾和假设案例进行讨论。

**📈 对比分析**

通过对比不同的文献和案例，探讨了人类在软件生产中的角色与责任，强调了责任的分配并不简单，且与失败后的处理密切相关。

**⚠️ 局限性**

限制在于没有提供新的实证结果，且假设案例可能无法完全反映真实情况，缺乏对实际软件生产过程的深入实证研究。

---

## 694. DXPR: Depth-Based Vision-LiDAR Cross-Modal Place Recognition Using Vision Foundation Models

**arXiv ID:** 2609.09005 | [PDF](https://arxiv.org/pdf/2609.09005v1)

**作者:** Yungsoo Han `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个基于深度的跨模态地点识别框架DXPR，利用视觉基础模型（VFM）将单目相机查询与LiDAR地图进行匹配，无需特定模态的编码器。

**💡 创新点**

创新点在于将相机图像和LiDAR扫描转换为统一的深度图像表示，从而使单一的VFM主干网络能够学习模态不变的全局描述符，并引入几何感知重叠挖掘器以提高配对度量学习的准确性。

**🔧 技术方法**

使用了视觉基础模型（VFM）进行特征提取和聚合，结合几何感知重叠挖掘策略。

**📊 数据集**

使用了KITTI和Boreas数据集进行实验，KITTI用于与现有工作进行直接比较，Boreas用于验证在不同天气和光照条件下的性能。

**📈 对比分析**

在KITTI数据集上，DXPR在大多数序列中实现了接近完美的Recall@1，超越了之前的CMPR基线。在Boreas数据集上，DXPR在同一序列的表现与强单模基线（DINOv2-SALAD）相当，而在更具挑战性的跨序列设置中显示出明显的改进。

**⚠️ 局限性**

局限性在于尽管该方法在多种条件下表现出色，但仍可能受到传感器特性和环境变化的影响，未来的工作将扩展到更多模态并探索更先进的主干-聚合器组合。

---

## 695. Transformers as In-Context Samplers: From Closed-Form Diffusion to Estimation-Free Sampling

**arXiv ID:** 2609.08981 | [PDF](https://arxiv.org/pdf/2609.08981v1)

**作者:** Arman Adibi `[一作]` (Augusta University), Hadi Daneshmand `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文证明了大语言模型不仅仅是统计记忆器，而是能够进行上下文学习，能够在测试时仅使用提示中提供的示例进行推理，而无需更新参数。具体来说，论文探讨了变换器如何在上下文中实现数据生成，证明了变换器可以模拟迭代生成采样器。

**💡 创新点**

创新点在于证明了变换器不仅可以进行监督学习任务，还可以在上下文中进行数据生成，展示了变换器在生成模型中的具体生成角色，尤其是通过软最大注意力机制计算责任权重和加权经验平均。

**🔧 技术方法**

使用了变换器架构，特别是软最大自注意力机制来实现闭式扩散模型的模拟，并通过实验分析了预训练语言模型的生成机制。

**📊 数据集**

使用了从共同语义类别（如动物、食物或城市）中抽取的单词构成的提示进行实验，研究了不同预训练语言模型的表现。

**📈 对比分析**

通过与现有的生成模型进行比较，论文展示了变换器在生成新样本时的能力，尤其是在上下文样本的影响下，生成的样本能够遵循提示分布，表现出良好的生成能力。

**⚠️ 局限性**

限制在于，尽管理论上证明了变换器可以实现这些算法，但并未完全确定预训练语言模型内部执行的具体算法。此外，模型规模的不同可能导致均匀化效果的显著差异，较小的模型在这一方面的表现较弱。

---

## 696. NERVE Attacks: Breaking AI-Powered Brain-Computer Interfaces

**arXiv ID:** 2609.08971 | [PDF](https://arxiv.org/pdf/2609.08971v1)

**作者:** Zahra Tarkhani `[一作]` (Microsoft), Anil Madhavapeddy `[通讯]` (University of Cambridge)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文介绍了一种系统化的脑机接口（BCI）安全分析框架，识别了五种正交攻击维度，并揭示了17种新型神经特定攻击实例。

**💡 创新点**

创新点在于提出了五个正交的BCI特定攻击维度，并开发了一个可扩展的安全分析框架，能够系统性地评估BCI的安全性。

**🔧 技术方法**

使用了AI辅助的扩展框架进行BCI安全分析，结合了生成性AI技术来降低攻击门槛。

**📊 数据集**

使用了多个真实世界的BCI平台和多样的EEG数据集进行评估，包括OpenBCI、Muse和NeuroSky等设备。

**📈 对比分析**

通过与现有方法的比较，发现新提出的攻击方法在有效性和隐蔽性上具有显著优势，能够在不需要专家知识的情况下实现攻击。

**⚠️ 局限性**

局限性在于当前的防御措施无法完全消除所有风险，且BCI软件堆栈在每个层面上都存在安全漏洞，缺乏系统性的安全设计文化。

---

## 697. GraphFAS: A Distributed System for Automated Graph Feature Generation and Selection in Industrial Transaction Networks

**arXiv ID:** 2609.08970 | [PDF](https://arxiv.org/pdf/2609.08970v1)

**作者:** Yice Luo `[一作]` (Ant Group), Jiajun Zheng `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于Boruta的分布式图特征选择程序（Graph Feature Automated Selection），用于工业交易网络中的欺诈检测，旨在克服传统特征工程和图神经网络（GNN）在可解释性和部署方面的局限性。

**💡 创新点**

创新点在于通过无参数的图特征生成模块和自动化的分布式特征选择算法，生成明确且可解释的结构特征，并在大规模图数据上有效识别信息特征，减少对领域专家的依赖。

**🔧 技术方法**

使用了无参数的图特征生成模块和基于Boruta的分布式特征选择算法，结合中位数聚合方法进行特征选择。

**📊 数据集**

使用了八个公共基准数据集和三个来自支付宝的工业数据集，数据集包含多种关系边和真实的欺诈标签。

**📈 对比分析**

与传统的GNN管道相比，该方法将特征聚合与模型训练解耦，能够直接与表格模型集成，并与基于TreeSHAP的解释兼容。在多个风险控制场景中，处理数百万个种子节点，显示出显著的性能提升。

**⚠️ 局限性**

局限性在于解耦特征生成与模型训练可能牺牲一些表示能力，但在可扩展性和可解释性方面提供了实用的好处。

---

## 698. Good Pretraining, Bad SFT: Checkpoint Quality Across the Training Stack

**arXiv ID:** 2609.08966 | [PDF](https://arxiv.org/pdf/2609.08966v1)

**作者:** Sohir Maskey `[一作]` (Aleph Alpha), Sascha Wirges `[通讯]` (Aleph Alpha)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在30B参数的混合专家模型中，不同检查点在多个训练阶段的表现及其对最终模型性能的影响。

**💡 创新点**

提出了检查点选择标准可能会导致最终模型性能的反转，强调了解决密度在模型适应性中的重要性。

**🔧 技术方法**

使用了高斯扰动来评估检查点的解决密度，并通过不同的训练阶段进行比较。

**📊 数据集**

使用了GSM8K和MBPP数据集进行实验。

**📈 对比分析**

通过比较不同检查点在中期训练和长上下文适应后的表现，发现中期和长上下文的聚合分数与最终模型的表现相关性较低，尤其是在中期训练后。

**⚠️ 局限性**

结果仅基于一个30B MoE模型，且未测试所有设置组合，解决密度仅在两个任务上测量，未建立因果关系。

---

## 699. Ozaki 2.5: Engineering the Deconstruction Path of fp64-Emulated Dense Matrix Multiplication on FP8 Tensor Cores

**arXiv ID:** 2609.09095 | [PDF](https://arxiv.org/pdf/2609.09095v1)

**作者:** Satoshi Matsuoka `[一作]` `[通讯]` (RIKEN Center for Computational Science), Satoshi Matsuoka (RIKEN Center for Computational Science)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种FP8 Ozaki II方法，通过在中国剩余定理（CRT）残余系统上模拟64个矩阵乘法，使用低精度张量核心产品的固定调度。

**💡 创新点**

创新点在于工程化了去构造路径，提出了Ozaki 2.5方法，优化了去构造性能模型，并引入了模数/编码的协同设计。

**🔧 技术方法**

使用了FP8张量核心、去构造感知性能模型、模数/编码协同设计等技术。

**📊 数据集**

使用了NVIDIA Rubin和Blackwell（GB300）平台的硬件性能数据，特别是FP8模数集。

**📈 对比分析**

与之前的研究相比，Ozaki 2.5方法在大规模DGEMM（密集矩阵乘法）中表现出更高的性能，尤其是在处理大规模输出时，达到了约235 TFLOPS的性能，接近理论上473 TFLOPS的上限。

**⚠️ 局限性**

限制在于去构造成本和去构造-λ地板，影响了大规模DGEMM的性能，且当前硬件尚未实现所有提出的协同设计要求。

---

## 700. ActReview: Rebuttal-Guided Training Data and Rubric Rewards for Actionable Peer Review Generation

**arXiv ID:** 2609.09076 | [PDF](https://arxiv.org/pdf/2609.09076v1)

**作者:** Yiling Ma `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了可操作的同行评审生成，提出了一种基于反驳的后训练框架，将论文特定的诊断与具体的修订计划相连接。

**💡 创新点**

创新点在于将同行评审生成分解为两个子任务：诊断声明生成和修订建议生成，并利用真实的评审-反驳线程构建数据集，以提供更具针对性的反馈。

**🔧 技术方法**

使用了多任务监督微调和基于候选的、特定弱点的评分奖励的GRPO（基于奖励的优化）技术。

**📊 数据集**

使用的数据集来自OpenReview，包含15,819篇论文和约40,000个弱点-反应实例，构建了一个包含1,000个实例的人类策划基准。

**📈 对比分析**

与之前的专门评审生成模型相比，实验表明该方法在可操作性和基础性方面表现更好，同时与强大的基于提示的LLM保持竞争力。人类评估确认了修订的实用性有所提高，但在技术准确性上仍存在差距。

**⚠️ 局限性**

限制在于该方法主要关注可通过反驳解决的弱点，排除了如基本新颖性争议、深层概念分歧或不可修复的方法论缺陷等问题。此外，尽管在修订导向的维度上有所提升，但在技术准确性上并未始终超越强大的专有LLM。

---

## 701. "World Knowledge" in the Weights: Reading Concept Circuits of Vision Transformers

**arXiv ID:** 2609.09055 | [PDF](https://arxiv.org/pdf/2609.09055v1)

**作者:** Yanlin Chen `[一作]` (University of Delaware), Xi Peng `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了一种基于跨层转码器（CLTs）的方法，用于从视觉变换器（ViTs）中读取概念电路，揭示模型内部的“世界知识”。

**💡 创新点**

创新点在于使用CLTs来提取全局和实例概念电路，提供了对模型行为的两种互补视角，并展示了其在自动发现虚假相关性、去除虚假相关性和模型比较中的应用。

**🔧 技术方法**

使用了跨层转码器（CLTs）技术，该技术通过稀疏的跨层字典来近似变换器的计算，能够直接从学习到的参数中恢复全局概念电路。

**📊 数据集**

使用了多个数据集进行实验，包括Waterbird数据集和ImageNet数据集，以验证所提方法的有效性。

**📈 对比分析**

与现有方法相比，所提方法在Waterbird数据集上提高了11.0%的性能，展示了其在去除虚假相关性和模型比较方面的优势。

**⚠️ 局限性**

限制在于CLTs主要捕捉多层感知器（MLP）路径，间接反映注意力结构，概念解释需要人工检查，且全局图未能完全捕捉因果关系。

---

## 702. PlayTrain: An Efficient Reinforcement Learning Framework for LLM-Generated Adaptable JavaScript Games

**arXiv ID:** 2609.09059 | [PDF](https://arxiv.org/pdf/2609.09059v1)

**作者:** Ryan Truong `[一作]` (Harvard University), Kazuki Irie `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PlayTrain，一个结合大型语言模型（LLM）和高效管道的强化学习（RL）框架，能够从最小的人类提示生成JavaScript游戏，并在标准的gym环境中运行这些游戏。

**💡 创新点**

PlayTrain通过利用LLM生成游戏代码，显著简化了视频游戏环境的开发过程，使得用户可以灵活地修改环境或创建新版本，极大地提高了适应性和可玩性。

**🔧 技术方法**

使用了大型语言模型（LLM）生成JavaScript代码，并结合了一个高效的JS到gym的后端，优化了环境的运行效率。

**📊 数据集**

使用了经典的Atari和ProcGen游戏作为数据集，生成了这些游戏的JavaScript克隆，并进行了性能测试。

**📈 对比分析**

与经典基准相比，PlayTrain在环境步骤速度上表现优异，能够达到每秒超过100万的代理决策，显著快于传统的C++环境，且在多线程设置下表现出线性扩展性。

**⚠️ 局限性**

PlayTrain的局限性在于生成的游戏大小和复杂性受限，当前仅支持2D环境，且无法生成现代控制台游戏的全保真克隆。

---

## 703. Answer-Distribution Trajectories: A Stochastic-Dynamics View of LLM Reasoning

**arXiv ID:** 2609.09030 | [PDF](https://arxiv.org/pdf/2609.09030v1)

**作者:** Mar Gonzàlez I Català `[一作]` (University of Cambridge), George Montañez `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的分析框架，称为答案分布轨迹，用于研究语言模型在链式思维推理过程中预测分布的演变。

**💡 创新点**

创新点在于引入答案分布轨迹，提供比最终预测和熵更细致的推理动态表示，能够捕捉支持的假设及其概率质量的变化。

**🔧 技术方法**

使用了随机动力学的视角来定义和分析答案分布轨迹，并通过轨迹级别的度量来描述推理动态。

**📊 数据集**

在十六个开放权重语言模型和四个推理基准（GSM8K、ARC、SVAMP和MATH）上进行了实验。

**📈 对比分析**

通过比较相同最终答案和相似熵轮廓的轨迹，发现它们可能展现出显著不同的推理动态，表明传统的评估方法无法捕捉到推理过程中的重要信息。

**⚠️ 局限性**

限制在于该分析方法计算开销较大，未来需要开发更便宜的近似方法来提高效率。

---

## 704. DYAD: A Multimodal Dataset of Co-Located Human Assistance

**arXiv ID:** 2609.09023 | [PDF](https://arxiv.org/pdf/2609.09023v1)

**作者:** Akhil Ajikumar `[一作]` (Georgia Institute of Technology), Mohsen Moghaddam `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DYAD（双人协助数据集），记录了人类在齿轮箱组装过程中的协作，链接了任务步骤、请求和干预的多模态数据。

**💡 创新点**

创新点在于提供了一个同步的多模态记录，能够将协助者的语言和物理干预与执行者的请求、任务状态和结果相结合。

**🔧 技术方法**

使用了HoloLens 2和Kinect进行数据捕捉，结合了自我中心视觉和工作空间感知技术。

**📊 数据集**

数据集包含20个会话的记录，涉及528个任务步骤和611个请求，851个有效的协助记录。

**📈 对比分析**

通过三个基准任务评估不同组件的性能，结果显示在模式预测上，RGB的宏F1为0.548±0.007，而因果元数据达到0.624，特权触发映射则达到0.915，表明信息的恢复能力。

**⚠️ 局限性**

限制在于数据集仅包含20个会话，且只针对一个特定的任务和助手，无法推广到其他任务、工作空间或人群中。

---

## 705. Deposon: An Auditable, Conservation-Guaranteed, Game-Theoretically Tested Scattering Layer over LLM Reasoning Paths

**arXiv ID:** 2609.09001 | [PDF](https://arxiv.org/pdf/2609.09001v1)

**作者:** Qihao Yuan `[一作]` `[通讯]` (Renmin University of China), Qihao Yuan (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Deposon的散射层，旨在为多步推理提供可审计的记录，解决了现有大语言模型（LLM）推理路径缺乏可检查账本的问题。

**💡 创新点**

创新点在于通过物理构造实现了能量守恒和可审计性，确保每个推理路径的能量分配可以被独立验证，并且消除了错误路径的预算无法恢复的问题。

**🔧 技术方法**

使用了散射理论中的三通道散射模型，结合了路径耦合和以太耦合的两个参数，构建了Deposon状态。

**📊 数据集**

在合成基准和真实基准（如GSM8K和StrategyQA）上进行了实验，使用了200个合成问题和多个真实问题进行评估。

**📈 对比分析**

与传统的六关键词规则过滤器相比，Deposon层在真实基准上表现无显著差异，表明其差异化价值仅在于机器可验证性，而非过滤性能的提升。

**⚠️ 局限性**

局限性包括所有三种形式的动态等价性被否定，且在真实基准上未能显示出显著的准确性提升，此外，散射层的优势仅在于可审计性，而非实际的推理性能提升。

---

## 706. Embedded Human-Centered Data Science in a Graduate Programming Course: A Framework and Case Study

**arXiv ID:** 2609.08982 | [PDF](https://arxiv.org/pdf/2609.08982v1)

**作者:** Victoria Chui `[一作]` (University of Toronto), Shion Guha `[通讯]` (University of Toronto)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个名为HELIX的框架，用于在信息科学课程中嵌入以人为中心的教育内容，旨在提高学生对数据科学中伦理和社会影响的理解。

**💡 创新点**

HELIX框架的创新点在于其三大支柱（知识构建、决策制定和赋权），为教师和学生提供了具体的行动指南，帮助他们在技术课程中有效整合人本内容。

**🔧 技术方法**

使用了文献综述和定量分析方法，通过前后测问卷评估学生在课程前后的知识、态度和决策能力的变化。

**📊 数据集**

在一门研究生的编程课程中应用了该框架，参与者为22名学生，使用了匿名的调查问卷和作业材料。

**📈 对比分析**

通过前后测问卷比较学生的知识和态度变化，结果显示学生在伦理责任感方面有显著提高（p = 0.002），但在知识和决策能力方面的变化较小，表明短期接触可能不足以培养应用伦理推理能力。

**⚠️ 局限性**

该研究的局限性在于参与者的选择偏差（自愿参与），以及缺乏对照组，可能影响结果的普遍性和可靠性。

---

## 707. Learning Length-Extrapolatable Recurrent Models

**arXiv ID:** 2609.09157 | [PDF](https://arxiv.org/pdf/2609.09157v1)

**作者:** Hanwen Jiang `[一作]` `[通讯]` (Adobe Research), Hanwen Jiang (Adobe Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了如何通过短序列训练使递归模型在长时间范围内保持可靠的行为，提出了一种新的方法Credit Stabilization through Time (CST)。

**💡 创新点**

创新点在于通过稳定状态信用的规模来改善长时间范围的学习，而不改变每个修正组件的方向，从而提高模型在长序列上的推断能力。

**🔧 技术方法**

使用了Credit Stabilization through Time (CST)方法，该方法在反向传播过程中对状态信用进行局部缩放，以稳定信号传递。

**📊 数据集**

使用了合成任务和真实数据集进行实验，合成任务用于控制学习需求，而真实数据集用于语言建模。

**📈 对比分析**

与标准的反向传播通过时间（BPTT）方法相比，CST在所有评估数据集和长度设置中都表现出更好的性能，尤其是在需要长时间依赖的任务中，CST的准确率提高了6.58个百分点。

**⚠️ 局限性**

CST仅控制选定边界信号的规模，无法恢复缺失的方向或消除时间贡献之间的干扰，因此它只解决了长期信用分配的一个失败模式。

---

## 708. SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators

**arXiv ID:** 2609.09155 | [PDF](https://arxiv.org/pdf/2609.09155v1)

**作者:** Yuncong Yang `[一作]` (University Of Massachusetts Amherst), Chuang Gan `[通讯]` (University Of Massachusetts Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于视觉校准的动作条件世界模型，能够在未见过的环境中进行零-shot模拟，且无需额外训练。

**💡 创新点**

创新点在于通过视觉校准情境来指定特定设置的动作-视觉映射，从而实现对新环境的快速适应和可靠的零-shot生成。

**🔧 技术方法**

使用了Diffusion Transformer (DiT)作为基础架构，结合视觉校准和交互历史来明确设置特定的动作-视觉映射。

**📊 数据集**

使用了来自RLBench、RoboCasa和RoboMimic的模拟数据集，以及真实世界的DROID数据，以覆盖多样的视觉场景和相机视角。

**📈 对比分析**

与现有的动作条件世界模型相比，提出的方法在所有评估指标上均表现出显著的优势，尤其是在未见环境中的视频预测质量和多视角一致性方面。

**⚠️ 局限性**

限制在于模型在没有视觉校准的情况下可能会降低性能，尽管通过历史交互可以进行推断，但在某些情况下仍需依赖显式的校准信息。

---

## 709. Proxy Policy Steering

**arXiv ID:** 2609.09148 | [PDF](https://arxiv.org/pdf/2609.09148v1)

**作者:** Chuanruo Ning `[一作]` (Cornell University), Kuan Fang `[通讯]` (Cornell University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种推理时适应方法，通过训练两个轻量级代理策略来解决将通用机器人策略专门化到新任务的挑战。

**💡 创新点**

创新点在于通过校准的速度空间残差引导冻结的基础采样器，从而在不修改基础模型的情况下实现任务特定行为的提取。

**🔧 技术方法**

使用了轻量级的代理策略和速度空间残差技术。

**📊 数据集**

在8个真实世界和4个模拟操作任务上进行了评估。

**📈 对比分析**

在12个任务上，提出的方法平均提高了53%的成功率，超越了LoRA微调、从头开始的专家策略和其他推理时引导方法。

**⚠️ 局限性**

限制在于目前只研究了单任务环境，尚未验证在多任务设置中的有效性，并且依赖于基础模型的强大性能。

---

## 710. NOAH: Learning the Full Patient Journey. A Longitudinal Multimodal Time-Aware Model for Representation and Forecasting

**arXiv ID:** 2609.09140 | [PDF](https://arxiv.org/pdf/2609.09140v1)

**作者:** Tobias Susetzky `[一作]` (Technical University of Munich), Daniel Rueckert `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种时间感知的、任务无关的生成变换器模型，旨在表示和预测患者的多模态状态轨迹。

**💡 创新点**

创新点在于引入了双向时间集成和变分潜在空间，以捕捉患者状态的连续演变和临床轨迹的随机性。

**🔧 技术方法**

使用了生成变换器模型，结合了变分自编码器的原理，能够处理多种类型的医疗图像、时间序列和数值信号。

**📊 数据集**

使用了来自MIMIC数据集家族的超过5.59亿个临床事件，涵盖431,000次医院就诊和299,000名患者的记录。

**📈 对比分析**

与现有方法相比，该模型在多个临床下游任务中表现出色，能够进行概率零-shot预测、事件时间预测，并生成紧凑的患者状态表示，AUROC得分在0.83到0.95之间，Brier得分在0.03到0.11之间，超越了简单的持久性基线。

**⚠️ 局限性**

模型的局限性在于自回归模型的暴露偏差，长时间预测时可能会传播错误，并且模型的表示不应被视为患者历史的总结，而是提取可预测特征。

---

## 711. A Data-Driven Framework for Identifying and Prioritizing RPA Opportunities in Healthcare Processes

**arXiv ID:** 2609.09137 | [PDF](https://arxiv.org/pdf/2609.09137v1)

**作者:** Maria Alejandra Gomez `[一作]` (Universidad de La Salle), Juan Manuel Castillo `[通讯]` (Universidad de La Salle)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个四模块的数据驱动框架，用于识别、优先排序、工具选择和成本计算医院运营中的RPA机会。

**💡 创新点**

创新点在于将过程发现、基于AHP的多标准优先排序、工具层选择和投资回报率量化整合为一个统一的可审计流程，并引入了自动化风险指数以区分可自动化的良好候选和安全自动化的候选。

**🔧 技术方法**

使用了分析层次过程（AHP）进行优先排序，结合了Python、n8n和企业级RPA平台的工具层选择，并进行了投资回报率（ROI）量化。

**📊 数据集**

应用于一个合成的投资组合，涵盖了20个医院常见的过程，12个过程通过了优先排序阈值。

**📈 对比分析**

与现有方法相比，该框架提供了一个系统化的决策流程，优先排序的结果在±20%的权重扰动下保持稳健（平均Spearman等级相关系数为0.83），并且投资组合的三年净现值在其第5百分位数下仍然为正。

**⚠️ 局限性**

该框架的局限性在于其权重和成本假设是基于文献推导的，而非在特定医院的初步数据上进行实证校准。

---

## 712. MeClear: Cooperative Game-Theoretic Attribution and Risk-Aware Memory Clearance for Long-Horizon LLM Agents

**arXiv ID:** 2609.09115 | [PDF](https://arxiv.org/pdf/2609.09115v1)

**作者:** Boyu Yang `[一作]` (Fudan University), Jun Zheng `[通讯]` (Beijing Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为MeClear的任务条件记忆清除框架，旨在通过合作归因识别和选择性抑制对下游任务有负面影响的记忆，从而改善长时间交互中的大型语言模型（LLM）代理的性能。

**💡 创新点**

MeClear的创新点在于结合了Leave One Out筛选和采样的合作Shapley归因，能够有效解决冗余冲突掩蔽问题，并在不永久改变持久记忆库的情况下验证任务恢复。

**🔧 技术方法**

使用了合作博弈论归因和风险意识的记忆清除技术，结合了局部筛选和合作Shapley归因。

**📊 数据集**

在十个长对话记忆池上进行了全面的实验评估，使用了Kimi-k2.6作为任务代理和Qwen3.6-Flash作为评估者，包含745个因果验证的测试案例。

**📈 对比分析**

与Leave-One-Out（LOO）基线相比，MeClear在目标召回率上达到了85.9%，整体任务恢复率为82.3%，比LOO提高了25.5个百分点，显示出显著的性能提升。

**⚠️ 局限性**

MeClear的局限性在于其依赖于查询条件的可见性决策，可能在某些情况下无法完全消除所有有害记忆，且在计算复杂性上仍需优化。

---

## 713. Copying explains the collective behavior of AI agents in the wild

**arXiv ID:** 2609.09150 | [PDF](https://arxiv.org/pdf/2609.09150v1)

**作者:** Giordano De Marzo `[一作]` (University of Konstanz), David Garcia `[通讯]` (University of Konstanz)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了2026年6月，数千个AI代理在一个公共维基上进行合作编辑的现象，探讨了它们如何在没有记忆和外部指示的情况下，自发地选择写作页面、命名自己和措辞信息。

**💡 创新点**

创新点在于首次记录了AI代理在未设计的环境中自发合作的过程，并通过分析其编辑记录，揭示了复制行为在集体行为中的重要性。

**🔧 技术方法**

使用了最小复制模型来分析代理的决策过程，模型中每个决策都有一个自由参数，能够重现代理在维基上编辑的分布特征。

**📊 数据集**

数据集包括从2026年5月1日到6月22日的14,591次编辑记录，涉及4,579个页面，记录了每次编辑的内容、用户名和时间。

**📈 对比分析**

通过与三种最小模型的比较，发现这些模型能够有效重现代理在维基上的行为模式，且模型的表现与实际数据相符，表明复制行为是集体结构形成的关键。

**⚠️ 局限性**

限制在于代理没有记忆，无法追踪长期行为，且数据的可用性依赖于研究者的选择，未来类似事件可能不会留下可分析的记录。

---

## 714. Procedural Graphs: Self-Evolving Execution Structures for LLM Agents

**arXiv ID:** 2609.09153 | [PDF](https://arxiv.org/pdf/2609.09153v1)

**作者:** Yuxing Lu `[一作]` (Google), Sercan Ö. Arık `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为程序图（Procedural Graph, PG）的显式可编辑的程序知识图，用于指导大型语言模型（LLM）代理的执行，同时保持推理的灵活性。

**💡 创新点**

创新点在于结合了可编辑的程序表示和基于代理当前进展的指导，提供了一种结构化的程序知识，能够有效引导代理的决策过程。

**🔧 技术方法**

使用了程序图（PG）框架，该框架通过在线推理和离线自我演化两个阶段来优化程序知识的拓扑结构和属性。

**📊 数据集**

在多个基准测试中评估了程序图的性能，包括HotpotQA、MultiChallenge、GDPval、ALFWorld、τ-bench、BFCL和EnterpriseArena等数据集。

**📈 对比分析**

与多种基线方法进行比较，程序图在24个模型-基准设置中排名第一或并列第一，表现出显著的性能提升，尤其在BFCL v3和GDPval任务中表现最佳。

**⚠️ 局限性**

局限性在于程序图的构建和自我演化过程可能依赖于初始的专家知识，且在某些情况下可能需要较多的计算资源和时间来进行优化。

---

## 715. Point4D: Long-range 4D Motion Reconstruction

**arXiv ID:** 2609.09145 | [PDF](https://arxiv.org/pdf/2609.09145v1)

**作者:** Minsik Jeon `[一作]` (Carnegie Mellon University), Shubham Tulsiani `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为Point4D的前馈模型，用于长视频序列的4D重建，能够在数百帧的视频中可靠地推断每个点的3D轨迹。

**💡 创新点**

创新点在于使用3D查询代替2D像素进行运动解码，从而解耦轨迹预测与图像平面可见性，允许在遮挡或视野外的情况下直接跨块重新查询3D点。

**🔧 技术方法**

采用了前馈模型和3D查询机制，结合了视觉变换器（ViT）编码器和轻量级交叉注意力解码器。

**📊 数据集**

使用了多个动态和静态数据集进行验证，包括PointOdyssey、Dynamic Replica、ScanNet等。

**📈 对比分析**

与现有的前馈4D重建方法（如TraceAnything、Any4D等）和3D点跟踪器（如SpatialTrackerV2）进行比较，Point4D在长视频4D跟踪基准测试中表现出色，显著优于之前的方法。

**⚠️ 局限性**

局限性在于块之间的交接仅携带每个查询的3D坐标和补丁描述符，缺乏场景表示或特征记忆，导致在整个块中完全被遮挡或超出视野的点的预测位置不可靠。

---

## 716. Studying Image Tokenizers as Visual Languages in Unified Multimodal Models

**arXiv ID:** 2609.09143 | [PDF](https://arxiv.org/pdf/2609.09143v1)

**作者:** Siting Li `[一作]` (University of Washington), Yang Liu `[通讯]` (Amazon Far)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究探讨了图像标记器在统一自回归多模态模型中的作用，建立了一个控制的纯自回归测试平台，跟踪多模态持续预训练过程中的任务特定验证损失。

**💡 创新点**

创新点在于通过任务特定的损失分析图像标记器的行为，揭示了重建保真度与多模态可学习性之间的差异，并探讨了图像标记空间对文本建模的影响。

**🔧 技术方法**

使用了纯自回归模型进行多模态持续预训练，结合了任务特定的验证损失作为信号来研究图像和文本标记的联合建模。

**📊 数据集**

使用了来自多个来源的公共数据集，包括LAION-Aesthetics、JourneyDB和BLIP3o-Pretrain-Short-Caption，共计60M样本。

**📈 对比分析**

通过任务特定的损失与下游性能的关系进行比较，发现不同任务的损失表现出不同的缩放行为，且没有单一的标记器排名适用于所有任务。

**⚠️ 局限性**

限制在于重建保真度与多模态可学习性之间的关系并不总是直接的，且图像标记器的选择可能会影响文本建模，需进一步研究不同设计选择的影响。

---

## 717. Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails

**arXiv ID:** 2609.09134 | [PDF](https://arxiv.org/pdf/2609.09134v1)

**作者:** Zhou Yu `[一作]` (Salesforce AI), Sitaram Asur `[通讯]` (Salesforce AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了如何将代理的系统提示、工具集和执行钩子等环境与模型权重共同演化，以提高在特定领域任务上的表现。

**💡 创新点**

创新点在于提出了一种基于在政策上的专家修正的方法，避免了直接模仿专家导致的模型与环境不匹配的问题，从而实现了模型和环境的协同演化。

**🔧 技术方法**

使用了轻量级微调技术（如LoRA-SFT）和自导向的最大似然估计（MLE）代理来进行模型的在政策修正。

**📊 数据集**

使用了七个企业代理基准任务的数据集，包括薪资审计、预算审批、库存警报、物联网异常检测、浏览器自动化、网站管理和代码重构等。

**📈 对比分析**

与传统的专家轨迹模仿方法相比，采用在政策修正的方法在所有任务上都取得了更好的性能，平均测试成功率从78.0%提高到79.7%。

**⚠️ 局限性**

限制在于轻量级LoRA方法可能无法完全缩小弱模型与专家模型之间的差距，且在某些任务上可能已经接近性能上限。

---

## 718. Nearly Tight Rademacher Bounds for Sparsely Activated Neural Networks

**arXiv ID:** 2609.09130 | [PDF](https://arxiv.org/pdf/2609.09130v1)

**作者:** Xiaoyu Li `[一作]` (University of New South Wales), Andi Han `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究了在一层ReLU神经网络中，输入依赖的稀疏性对统计复杂性的影响，提出了在特定输入域内的复杂性界限。

**💡 创新点**

通过引入支持保持覆盖和归一化链论证，消除了之前显式的维度因子，展示了输入域对复杂性的影响。

**🔧 技术方法**

使用了ReLU激活函数、Rademacher复杂性和度量熵等技术。

**📊 数据集**

使用了固定半径R的输入域内的样本数据集。

**📈 对比分析**

与之前的界限相比，提出的界限在去掉了显式的√(n)因子后，性能得到了提升，且在特定条件下达到了最优的复杂性界限。

**⚠️ 局限性**

限制在于该研究主要集中在一层网络的稀疏性上，未考虑多层网络的扩展和分布依赖的稀疏性问题。

---

## 719. Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout

**arXiv ID:** 2609.09123 | [PDF](https://arxiv.org/pdf/2609.09123v1)

**作者:** Zhuoran Zhao `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为Mask Forcing的双噪声掩蔽展开策略，以改善自回归视频扩散模型的生成质量，解决了由反向KL目标引起的模式崩溃问题。

**💡 创新点**

创新点在于通过随机掩蔽在空间和时间轴上注入低噪声信号，从而促进自回归模型在生成过程中更广泛地探索教师分布，改善视频的视觉质量和真实感。

**🔧 技术方法**

使用了双噪声掩蔽展开策略，结合自回归视频扩散模型和分布匹配蒸馏（DMD）技术。

**📊 数据集**

使用了VidProM数据集进行训练，生成的视频分辨率为832 × 480，包含81帧。

**📈 对比分析**

与多种自回归视频蒸馏方法进行了比较，结果显示Mask Forcing在视觉质量和收敛速度上均有显著提升，尤其在短视频、长视频和相机控制的自回归生成中表现出更高的真实感。

**⚠️ 局限性**

限制在于未能在生成过程中使用真实视频数据或额外的后训练阶段，可能影响模型在某些复杂场景下的表现。

---

## 720. TANGO: Humanoid Navigation in Cluttered Environments with a Whole-Body Vision-Language-Action Model

**arXiv ID:** 2609.09158 | [PDF](https://arxiv.org/pdf/2609.09158v1)

**作者:** Anqi Li `[一作]` (University of California), Dhruv Shah `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了人形机器人在拥挤室内环境中的导航问题，提出了一种新的整体视觉-语言导航框架，能够根据自然语言指令和自我中心的RGB观察直接预测29自由度的关节空间动作。

**💡 创新点**

创新点在于首次提出了一个整体视觉-语言导航框架，能够处理复杂的3D空间中的人形机器人导航，避免了传统方法的局限性，并实现了零-shot转移到真实世界场景。

**🔧 技术方法**

使用了整体视觉-语言（VLA）系统，结合了流匹配的动作专家和高频率的低级跟踪器，采用了自动化的数据生成管道来合成无碰撞的导航轨迹。

**📊 数据集**

使用了合成的室内导航数据集，包含578个场景和64,633条轨迹，数据集通过模拟环境生成，涵盖了多样的碰撞避免行为。

**📈 对比分析**

与现有的视觉-语言导航（VLN）和人形空间导航基线进行比较，结果显示在需要障碍物协商的长时间导航任务中，表现出更强的无碰撞导航性能，成功率和路径长度加权成功率均优于其他方法。

**⚠️ 局限性**

局限性在于低级跟踪器的能力限制了在更复杂环境中的进一步部署，例如在楼梯上行走。此外，模型仅依赖RGB图像作为视觉输入，可能限制其在复杂场景中的理解能力，未来工作可以探索结合深度相机和LiDAR的可能性。

---

## 721. ReCite: Agentic Reasoning for Faithful Citation

**arXiv ID:** 2609.09156 | [PDF](https://arxiv.org/pdf/2609.09156v1)

**作者:** Yuyang Huang `[一作]` (Wuhan University), Donghong Ji `[通讯]` (Wuhan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的自动引用推荐框架ReCite，旨在通过主动的声明级推理来提高引用的准确性。

**💡 创新点**

创新点在于将引用过程从基于相似性的检索转变为基于推理的任务，提出了闭环的声明-证据验证框架，集成了意图感知的查询规划和反思重检索循环。

**🔧 技术方法**

使用了Qwen3-4B模型，并通过群体相对策略优化（GRPO）进行强化学习训练。

**📊 数据集**

构建了一个大规模的文献数据集，包含来自arXiv的10,893个LaTeX源包，涵盖2024-2025年在顶级计算机科学会议上发表的论文。

**📈 对比分析**

通过实验表明，ReCite在严格的引用准确性上显著优于现有的生成模型，验证了声明-证据验证、意图感知规划和反思循环在提升性能中的重要性。

**⚠️ 局限性**

限制在于当前的引用数据集仅映射局部上下文到目标论文，缺乏训练代理所需的显式推理轨迹。

---

## 722. The exact asymptotic constant in the metric dimension of Jaccard space

**arXiv ID:** 2609.09146 | [PDF](https://arxiv.org/pdf/2609.09146v1)

**作者:** Bjørn Kjos-Hanssen `[一作]` `[通讯]` (University of Hawaii), Bjørn Kjos-Hanssen (University of Hawaii)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究了有限集合X的幂集2^X在Jaccard距离下的度量维度，并确定了常数。

**💡 创新点**

确定了Jaccard空间的度量维度的确切常数，填补了之前研究中的空白。

**🔧 技术方法**

使用了Erdős–Rényi硬币称重问题的技术，结合了概率论和确定性方法。

**📊 数据集**

使用了有限集合X的幂集2^X，具体的n值未给出，但n是X的大小。

**📈 对比分析**

通过与Lladser和Paradise的结果进行比较，证明了度量维度的下界和上界，性能在Θ(n/ln n)范围内。

**⚠️ 局限性**

研究中未提及具体的局限性，但可能存在对更大集合的推广不足。

---

## 723. Entropy-Regularized Rank-Masked Policy Optimization for Test-Time Reinforcement Learning in Code Generation

**arXiv ID:** 2609.09135 | [PDF](https://arxiv.org/pdf/2609.09135v1)

**作者:** Jiacheng Xu `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种探针驱动的测试时强化学习（TTRL）方法，使其适用于代码生成，通过从问题陈述构建无输出的探针输入，并根据候选程序在这些探针上的执行结果定义探针共识奖励（PCR）。

**💡 创新点**

创新点在于将探针输入与PCR结合，提供了一种在没有标准答案的情况下进行代码生成的奖励信号，同时引入了熵正则化的排名掩蔽策略优化（ERPO），以避免直接优化PCR带来的不可靠性。

**🔧 技术方法**

使用了探针驱动的测试时强化学习（TTRL）和熵正则化的排名掩蔽策略优化（ERPO）技术。

**📊 数据集**

在LiveCodeBench（LCB）上进行实验，并评估了在CodeContests（CC）、CodeForces（CF）和TACO上的零-shot迁移。

**📈 对比分析**

与GRPO_PCR和NSR_PCR等方法进行比较，ERPO在pass@1和pass@k上均表现出显著的提升，尤其在零-shot迁移中也显示了良好的泛化能力。

**⚠️ 局限性**

局限性在于实证评估仅限于竞争编程风格的编码基准，未能涵盖更广泛的代码生成任务，未来的工作应关注项目级和工业规模的代码生成场景。

---

## 724. ExecCritic: Learn to Test, Test to Improve for Coding Agents

**arXiv ID:** 2609.09133 | [PDF](https://arxiv.org/pdf/2609.09133v1)

**作者:** Leitian Tao `[一作]` (University of Wisconsin--Madison), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的框架，分离测试生成和源代码修复，使用测试代理生成行为检查，修复代理仅根据反馈修订源代码。

**💡 创新点**

创新点在于引入了测试-验证-修订的结构，允许测试和修复代理独立训练，从而提高了修复的准确性和效率。

**🔧 技术方法**

使用了Qwen-3.5-35B-A3B作为基础模型，并结合了强化学习技术来训练测试和修复代理。

**📊 数据集**

使用了SWE-bench Verified数据集进行实验评估。

**📈 对比分析**

通过比较不同来源的测试反馈，发现高质量的生成测试能显著提高修复代理的性能，组合训练的代理在没有额外强模型或Oracle反馈的情况下达到了72.6%的成功率。

**⚠️ 局限性**

当前框架将测试和修复代理作为独立策略进行训练，未来的工作需要探索共享权重的联合训练，以简化部署并提高整体性能。

---

## 725. DeCAL: Towards Physically-Grounded Dexterous Vision-Language-Action Models via Contact-Aware Latent Co-Imagination

**arXiv ID:** 2609.09119 | [PDF](https://arxiv.org/pdf/2609.09119v1)

**作者:** Yankai Fu `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DeCAL，一个基于物理的灵巧视觉-语言-行动模型，统一了感知、理解、想象和行动，能够在多种接触丰富的灵巧操作任务中表现出色，并在未见场景中展现出强大的泛化能力。

**💡 创新点**

创新点在于引入了自适应视觉-触觉融合和视觉-触觉潜在共同想象，能够有效利用触觉信号进行接触丰富的灵巧操作。

**🔧 技术方法**

使用了混合变换器（Mixture-of-Transformers, MoT）架构，结合了理解、想象和行动生成的专家模型，并采用了接触感知门控策略进行自适应触觉信息融合。

**📊 数据集**

在六个接触丰富的灵巧操作任务上进行了实验，使用了100个高质量的专家演示数据集，并进行了20次试验评估。

**📈 对比分析**

与两种最先进的VLA模型（GR00T N1.6和InternVLA-A1）及两种基于触觉的专业策略（ViTacFormer和DECO）进行了比较，DeCAL在所有任务中均表现出最高的成功率，平均成功率达到71%，进展成功率达到83.4%。

**⚠️ 局限性**

局限性包括对触觉感知的依赖，可能受到传感器噪声、校准误差和模型漂移的影响，此外，当前框架未利用大规模的视觉-触觉预训练，未来研究可在这一方向上进行扩展。

---

## 726. Canonical Color as a Lens into Concept Decodability in Vision Encoders and VLMs

**arXiv ID:** 2609.09124 | [PDF](https://arxiv.org/pdf/2609.09124v1)

**作者:** Xiaofu Chen `[一作]` (MBZUAI), Yova Kementchedjhieva `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究使用典型颜色作为受控探针，探讨视觉编码器在去除颜色信息后是否仍能线性解码典型颜色信息，并分析其与物体身份的关系。

**💡 创新点**

创新点在于使用典型颜色作为受控案例，验证视觉编码器是否将典型颜色作为物体概念的一部分进行编码，并探讨视觉-语言模型（VLM）后训练对颜色可解码性的影响。

**🔧 技术方法**

采用线性探针技术对冻结的视觉编码器进行分析，比较不同层次的表示以评估典型颜色和物体身份的可解码性。

**📊 数据集**

构建了一个包含708个物体类别的典型颜色数据集，每个类别与10个典型颜色标签之一相关联，目标是每个类别5张图像。

**📈 对比分析**

通过与视觉-反事实（Visual-CounterFact）数据集的比较，验证了去除颜色信息后，典型颜色仍然可以被解码，且与物体类别信息相关联。结果显示，典型颜色在不同视觉编码器中保持线性可解码性，且在某些模型中，VLM后训练会降低视觉塔的可解码性，但在解码器侧的表示中恢复。

**⚠️ 局限性**

本研究的局限性在于仅聚焦于典型颜色作为受控案例，其他视觉属性（如材质、纹理等）可能需要不同的控制和评估协议。此外，线性探针只测量线性可访问的信息，可能无法捕捉到所有信息。

---

## 727. When Does Scale-Invariant Optimization Become Unstable? An Exact Schedule Law with Weight Decay

**arXiv ID:** 2609.09116 | [PDF](https://arxiv.org/pdf/2609.09116v1)

**作者:** Hasan Amin `[一作]` (Purdue University), Rajiv Khanna `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了归一化对神经网络优化动态的影响，提出了一种精确的离散时间法则，描述了学习率调度和权重衰减如何通过参数范数相互作用来控制优化器的有效步长。

**💡 创新点**

创新点在于提出了一个单一标量量（B_t）来捕捉所有调度和衰减的影响，并揭示了在归一化下自我抑制强度的结构二分法，解释了自适应方法在归一化下的稳定性较弱的原因。

**🔧 技术方法**

使用了精确的离散时间法则和统一的同质优化器框架，分析了不同优化器（如SGD、SGDM和Adam）的动态行为。

**📊 数据集**

在多个数据集上进行了实验，包括MNIST、CIFAR-10、WikiText和OpenWebText，验证了所提出的理论和法则的有效性。

**📈 对比分析**

通过与标准调度（如常数、阶梯衰减和余弦衰减）进行比较，发现B_t能够准确预测扩展和收缩的动态，性能在预测的边界附近达到峰值。

**⚠️ 局限性**

限制在于该研究主要集中在归一化的优化动态上，未来的研究可以扩展到没有显式闭合解的解耦衰减优化器。

---

## 728. SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?

**arXiv ID:** 2609.09113 | [PDF](https://arxiv.org/pdf/2609.09113v1)

**作者:** Yuqiao Tan `[一作]` (Key Laboratory of Cognitive Intelligence, Institute of Automation, Chinese Academy of Sciences), Kang Liu `[通讯]` (Key Laboratory of Cognitive Intelligence, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SAEScientist-Bench基准，评估AI代理是否能够利用稀疏自编码器（SAE）工具进行自主的机制发现。代理设计对比探针，导航Gemma-2-9B-IT中的特征字典，以发现最佳特征，并与专家参考特征进行比较。

**💡 创新点**

创新点在于建立了一个标准化的评估框架，涵盖20个任务，评估AI代理在机制发现中的能力，并引入了激活排名、激活选择性和因果引导等多个维度的评分。

**🔧 技术方法**

使用了稀疏自编码器（SAE）作为机制可解释性工具，结合了对比探针设计和特征导航技术。

**📊 数据集**

使用了Gemma-2-9B-IT数据集，该数据集包含超过131K个特征，并与Neuronpedia中的专家参考特征进行比较。

**📈 对比分析**

与专家基线相比，代理在激活选择性上接近专家水平（92.91 vs 98.92），但在因果生成引导方面表现较差（31.47 vs 57.75）。代理能够设计对比探针以排除虚假候选，但在实验测量的解读上存在误差。

**⚠️ 局限性**

限制在于基准主要集中在单一特征发现，未来需要扩展到多特征电路发现和开放式假设生成，同时引入人类验证和多参考特征以减少潜在的评分噪声。

---

## 729. A Generalization of Amari's Bayesian Duality

**arXiv ID:** 2609.09126 | [PDF](https://arxiv.org/pdf/2609.09126v1)

**作者:** Mohammad Emtiyaz Khan `[一作]` (RIKEN Center for Advanced Intelligence Project), Thomas Möllenhoff `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文重新审视了Amari的贝叶斯对偶理论，并将其与贝叶斯规则的凸对偶性联系起来，提出了Amari贝叶斯对偶的广义化，讨论了其在现代人工智能中的相关性。

**💡 创新点**

创新点在于通过不同的数学框架扩展了Amari的理论，使其适用于更广泛的贝叶斯推断和凸优化文献，并提供了一个新的视角来理解贝叶斯对偶。

**🔧 技术方法**

使用了凸对偶性作为数学工具，提出了一种变分形式的贝叶斯规则，并通过优化问题的对偶形式来实现贝叶斯对偶的广义化。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了与贝叶斯推断和机器学习相关的理论框架。

**📈 对比分析**

与传统的贝叶斯对偶方法相比，本文的方法在处理一般贝叶斯模型时表现出更大的灵活性，能够解决Amari方法的限制，尤其是在后验和似然不具有相同形式的情况下。

**⚠️ 局限性**

限制在于尽管提出了广义化的方法，但在某些情况下，后验和似然之间的映射可能不是唯一的，且在处理复杂模型时仍可能面临挑战。

---

