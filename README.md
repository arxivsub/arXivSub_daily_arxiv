# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-11 | 今日论文总数: 596

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Point Cloud as a Foreign Language for Multi-modal Large Language Model

**arXiv ID:** 2603.09173 | [PDF](https://arxiv.org/pdf/2603.09173v1)

**作者:** Sneha Paul `[一作]` (Concordia University), Nizar Bouguila `[通讯]` (Concordia University)

**通讯引用:** 9653 | [OpenAlex ID](https://openalex.org/A5090600716)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SAGE，一种无预训练3D编码器的端到端多模态大语言模型，直接将原始点云转换为离散令牌以供LLM处理；

**💡 创新点**

创新点在于：①用轻量级3D分词器（几何采样+邻域聚合+向量量化）将点云视为“外语”扩展LLM词表；②引入基于语义对齐的奖励的首个偏好优化训练策略（GRPO），提升开放式3D问答的推理能力；

**🔧 技术方法**

采用的技术包括：farthest point sampling、kNN邻域聚合、向量量化（Codebook）、LLM解码器（LLaMA），以及GRPO强化学习与语义相似度奖励；

**📊 数据集**

使用的数据集：Objaverse（约660K 3D对象及其Cap3D描述），MM‑Vet（3D VQA），以及在训练阶段利用PointLLM提供的700K点云-文本对；

**📈 对比分析**

与现有Encoder‑based方法（PointLLM、ShapeLLM、LLaVA‑3D等）相比，SAGE在3D caption、分类、VQA等任务上取得更高的GPT‑4、BLEU‑1、ROUGE‑L、METEOR、Sentence‑BERT等指标，且推理延迟显著降低（≈100 ms vs 240 ms）且对点云分辨率鲁棒；

**⚠️ 局限性**

局限性：目前仅在单模态点云+文本的任务上验证，尚未扩展到多传感器（如深度+RGB）或实时大规模部署；模型对超大点云（>8K点）处理速度和精度仍有待进一步提升；

---

## 2. Efficient Reasoning at Fixed Test-Time Cost via Length-Aware Attention Priors and Gain-Aware Training

**arXiv ID:** 2603.09253 | [PDF](https://arxiv.org/pdf/2603.09253v1)

**作者:** Rian Atri `[一作]` `[通讯]` (Serval Systems), Rian Atri (Serval Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了长度感知的注意力先验RPA和增益感知控制Guardian，提升Transformer在有限计算预算下的推理质量。

**💡 创新点**

通过模糊区域对齐与Sinkhorn对齐构造可学习、长度自适应的预softmax先验，并在训练时仅运行的梯度控制器根据验证收益动态调整注意力锐度。

**🔧 技术方法**

使用模糊会员分配、软cosine基底、Sinkhorn匀速化、两时间尺度REINFORCE控制、SWA与EMA、上下文长度的群体博弈等技术。

**📊 数据集**

在WikiText-2（GPT-2 BPE）上进行实验。

**📈 对比分析**

与仅使用正弦或相对位置偏置的基线在相同计算预算下对比；在WT2上验证交叉熵从5.85降至5.246，长序列下提升约18.8% perplexity，推理时延保持不变。

**⚠️ 局限性**

仅在小模型/低数据场景显著；在更大规模下优势减弱；控制器动作空间有限，可能不适用于多头精细调节；未报告微观推理时延测量。

---

## 3. $P^2$GNN: Two Prototype Sets to boost GNN Performance

**arXiv ID:** 2603.09195 | [PDF](https://arxiv.org/pdf/2603.09195v1)

**作者:** Arihant Jain `[一作]` (Amazon), Chaosheng Dong `[通讯]` (Amazon)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5071245598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种名为 P^2GNN 的插件式双原型机制，在传统消息传递图神经网络中加入全局原型节点和对齐原型，以增强全局上下文信息和降噪效果。

**💡 创新点**

创新点在于同时利用原型作为邻居（提供全局上下文）和原型对齐（对消息进行硬聚类降噪），并通过多任务损失（对齐、稀疏、多样性）来优化原型表示。

**🔧 技术方法**

核心技术包括基于原型的消息传递、混合注意力机制、原型对齐注意力、以及联合训练的辅助损失，适配任意消息传递 GNN（如 GCN、GAT、ACM‑GCN 等）。

**📊 数据集**

实验使用了 18 个公开基准数据集（节点分类任务）与 2 个专有电商推荐数据集（节点推荐任务），以及 6 个大型欺诈检测数据集，覆盖从低到高异质性和大规模图。

**📈 对比分析**

与多种基线（MLP、GCN、GAT、Heterophily‑aware GNN、SOTA 方法）以及相同骨干的 GNN 进行对比，P^2GNN 在推荐、分类与欺诈检测任务上平均提升 3–5%，并在多数数据集上获得统计显著优势，计算开销仅略高于骨干模型。

**⚠️ 局限性**

局限性包括需手动调节原型数目与辅助损失权重，原型数量过多可能导致过拟合；对极度稀疏或高度异质图的适用性尚待进一步验证。

---

## 4. Variational Routing: A Scalable Bayesian Framework for Calibrated Mixture-of-Experts Transformers

**arXiv ID:** 2603.09453 | [PDF](https://arxiv.org/pdf/2603.09453v1)

**作者:** Albus Yizhuo Li `[一作]` (Imperial College), Matthew Wicker `[通讯]` (Imperial College)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5006299169)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大规模 Mixture-of-Experts（MoE）Transformer，本文提出了 Variational Routing（VMoER）框架，通过变分推理在路由决策层引入不确定性，从而提升模型的校准性、路由稳定性和对分布外样本的检测能力。

**💡 创新点**

创新点包括：①把 MoE 路由视为潜在变量模型，直接对路由 logits 或选择空间做变分推理；②提出两种可摊销变分推理方案：Logit‑Space（VGLR，支持全协方差）和 Selection‑Space（VTSR，学习输入相关温度）；③将传统的负载平衡、辅助损失视为贝叶斯先验，实现无额外采样的轻量化实现。

**🔧 技术方法**

主要技术：变分推理（变分自编码器风格）、多头残差推断网络、温度缩放、蒙特卡洛采样、Softmax 与 Top‑K 的组合、熵正则化；实现方面使用轻量化的推断网络和全协方差的 Cholesky 分解。

**📊 数据集**

实验数据集：三大 MoE 结构（Granite‑3B、Qwen‑2.7B、DeepSeek‑16B）在多选问答任务上，包括 OpenBookQA、ARC‑Challenge、SciQ、MedMCQA；此外在 OoD 评估中使用 ARC‑Easy、ARC‑Challenge、MedMCQA、MMLU‑Law。

**📈 对比分析**

与基线（Deterministic MAP、全局温度、MCDropout、SWAG）比较，VMoER 在 ID 任务上显著降低 ECE（如 Granite 0.252→0.015），提升路由稳定性（噪声下 Jaccard 相似度提高 38%），并在 OoD 检测中 AUROC 提升 12%。同时 FLOPs 与激活内存增幅低于 1%，实现可扩展的低成本部署。

**⚠️ 局限性**

局限性：VTSR 在训练时易出现温度崩溃，需要精细初始化；实验仅覆盖多选问答（token‑level）任务，未验证生成任务或更大规模（70B+）模型；对抗性攻击、数据中毒、专家内触发器等其他失败模式不在研究范围。

---

## 5. Evolving Prompt Adaptation for Vision-Language Models

**arXiv ID:** 2603.09493 | [PDF](https://arxiv.org/pdf/2603.09493v1)

**作者:** Enming Zhang `[一作]` (Tsinghua University), Yang Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 45515 | [OpenAlex ID](https://openalex.org/A5100769533)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出EvoPrompt方法，实现对预训练Vision‑Language模型在少样本任务中的知识保留式自适应。

**💡 创新点**

创新点在于引入模态共享提示投影器（MPP）和演化轨迹感知训练策略，解耦提示方向与幅度并采用特征几何正则化，防止灾难性遗忘。

**🔧 技术方法**

采用提示学习、低秩适配、特征几何正则（Soft‑HGR）、动态秩缩减等技术。

**📊 数据集**

在ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、UCF101、DTD、EuroSAT等11个图像分类基准，以及跨数据集和域迁移数据集上进行评估。

**📈 对比分析**

与CoOp、MaPLe、PromptSRC等主流方法比较，EvoPrompt在平均泛化、跨域、少样本等指标上取得最优或同等性能，并在原始zero-shot精度上保持稳定。

**⚠️ 局限性**

局限在于对超参数（γ、η、秩阈值）敏感，且尚未在更大规模数据或更复杂任务上进行充分验证。

---

## 6. Latency Effects on Multi-Dimensional QoE in Networked VR Whiteboards

**arXiv ID:** 2603.09294 | [PDF](https://arxiv.org/pdf/2603.09294v1)

**作者:** Jiarun Song `[一作]` (Xidian University), Fuzheng Yang `[通讯]` (Xidian University)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5055433813)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

**🎯 论文内容**

通过构建可精确控制网络延迟的 NVR 白板系统，开展了基于协作模式（顺序 vs. 并行）和平台类型（带/不带虚拟头像的 VR 以及 PC）的大规模实验，评估不同延迟水平下用户的 QoE（交互性、效率、可信度等子维度）。

**💡 创新点**

① 开源的延迟可控 NVR 白板平台，填补了现有商业系统在延迟控制与实验支持方面的不足；② 系统性地将 QoE 分为实践与愉悦两大维度，并细化为多子维度，明确哪些子维度对延迟最敏感；③ 发现 600 ms 为 SC 场景的关键阈值，并揭示 avatar 对交互性与可信度的显著提升。

**🔧 技术方法**

Unity 3D 引擎 + Meta Quest 2 + Meta Horizon OS + Oculus Avatar SDK + WebRTC‑based P2P 通信 + 自定义时间缓冲实现的 E2E 延迟控制。

**📊 数据集**

实验数据来源于 36 名大学生参与者，分别在 7 个延迟水平（100 ms–2500 ms）下完成 SC 与 FC 两种协作任务，使用 3 种平台（VR+、VR、PC），收集 5‑点绝对类别评分（ACR）得分。

**📈 对比分析**

采用重复测量 ANOVA 与 MOS（平均意见得分）分析，结果显示：VR+ 在低至中等延迟（≤ 1000 ms）下交互性、效率和可信度均显著优于 VR 与 PC；SC 场景对延迟更敏感，效率下降最快；FC 场景更具鲁棒性。总体 QoE 随延迟升高显著下降，尤其在 600 ms 以上出现阶跃式衰退。

**⚠️ 局限性**

① 受限于仅招募 20‑30 岁的大学生，外推性有限；② 未加入语音交流，无法评估多模态交互对延迟的影响；③ 仅在局域网中控制延迟，未模拟真实网络抖动或丢包；④ 评估仅基于主观 QoE，缺乏任务完成时间、错误率等客观指标；⑤ 仅关注单一白板任务，未覆盖更复杂的协作场景。

---

## 7. RAE-NWM: Navigation World Model in Dense Visual Representation Space

**arXiv ID:** 2603.09241 | [PDF](https://arxiv.org/pdf/2603.09241v1)

**作者:** Mingkun Zhang `[一作]` (Tsinghua University), Ziyang Meng `[通讯]` (Tsinghua University)

**通讯引用:** 6418 | [OpenAlex ID](https://openalex.org/A5051392570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种将视觉导航世界模型从压缩VAE潜空间迁移到稠密DINOv2表示空间的RAE-NWM框架，并通过CDiT‑DH生成器与时间驱动门控的动态条件模块实现稳定的长时序rollout和精细动作控制。

**💡 创新点**

① 在稠密视觉表示空间建模导航动力学，保持几何结构；② 采用Conditional Diffusion Transformer with Decoupled Diffusion Transformer Head (CDiT‑DH) 的生成架构；③ 通过时间驱动门控机制自适应注入动作信息，平衡全局拓扑与局部细节。

**🔧 技术方法**

使用DINOv2编码器与RAE解码器、CDiT‑DH生成器、流匹配训练目标、ODE求解rollout、CEM规划、LPIPS/FID/DINO距离评估等技术。

**📊 数据集**

在多种真实机器人导航数据集上训练与评估：SACSoN/HuRoN、RECON、SCAND、Matterport3D（Habitat）。

**📈 对比分析**

与NWM、GNM、NoMaD、OmniVLA、One‑Step WM等基线对比，RAE‑NWM在长时序生成（4 s/16 s）上LPIPS、DINO距离、FID均显著优于NWM；轨迹预测ATE/RPE更低；Habitat闭环任务SR达78.95%，超过所有方法；门控机制在Ablation中比简单注入/MLP表现更好。

**⚠️ 局限性**

对高频随机纹理（如草）仍存在细节失真；依赖预训练RAE解码器在可视化时可能引入几何失真；需要更大生成器容量以提升视觉细节；在纹理丰富的无结构环境中VAE潜空间可能更具敏感性。

---

## 8. Time warping with Hellinger elasticity

**arXiv ID:** 2603.08807 | [PDF](https://arxiv.org/pdf/2603.08807v1)

**作者:** Yuly Billig `[一作]` (Carleton University), Yuly Billig `[通讯]` (Carleton University)

**通讯引用:** 1046 | [OpenAlex ID](https://openalex.org/A5052947695)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合 Hellinger 距离惩罚的时间扭曲匹配方法——Elastic Time Warping，并给出了新的相似性系数 K 与对应的距离度量。

**💡 创新点**

创新点在于将 Hellinger 相似系数引入时间参数化的最优匹配，构造可用于任意度量空间的相似性系数 K，并提出对应的动态规划算法，理论复杂度为 O(nm(n+m))。

**🔧 技术方法**

主要技术包括几何分析与概率论中的 Hellinger 距离、Hilbert 空间中的向量投影、Cauchy-Schwarz 不等式、以及动态规划实现。

**📊 数据集**

文章未给出具体实验数据集，仅在理论层面讨论了潜在应用场景（如 DNA 匹配、语音识别等）。

**📈 对比分析**

缺乏实验比较，无法给出性能对比；仅给出了理论复杂度与内存需求。

**⚠️ 局限性**

主要局限在于缺乏实证验证、复杂度相对较高、并且算法假设输入为分段常数函数，未讨论对噪声或不连续数据的鲁棒性。

---

## 9. Implicit Geometry Representations for Vision-and-Language Navigation from Web Videos

**arXiv ID:** 2603.09259 | [PDF](https://arxiv.org/pdf/2603.09259v1)

**作者:** Mingfei Han `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Ivan Laptev `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 31393 | [OpenAlex ID](https://openalex.org/A5087781064)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了大规模室内房间游览视频构成的 RoomTour3D 数据集，并通过隐式几何表示替代传统三维重建来训练 Vision‑and‑Language Navigation（VLN）代理。

**💡 创新点**

创新点在于①利用数千小时真实 YouTube 走廊视频自动提取长轨迹和开源描述；②在传统基于 COLMAP 的显式几何之外引入基于 VGGT 的隐式几何编码，显著提升数据利用率；③将描述‑丰富轨迹与动作‑丰富轨迹相结合，提供更完整的视觉‑语言‑空间监督。

**🔧 技术方法**

技术包括：COLMAP 3D 重建、RAM+Grounding‑DINO+Depth‑Anything 的目标检测与深度估计、GPT‑4 生成自由词汇指令、VGGT‑based 空间编码器、NaviLLM LLM 框架以及多任务预训练与微调。

**📊 数据集**

使用的数据集包括：RoomTour3D（1,847 片视频，200k+ 指令，17k 运动轨迹），以及 VLN 基准数据集 CVDN、SOON、R2R、REVERIE、ScanQA、LLaVA‑23k。

**📈 对比分析**

在四大 VLN 基准上与 SOTA 进行对比，RoomTour3D‑IGR 在 SPL/SR、GP 等指标上均实现 6%~14% 的提升；在零射击实验中，SR 达 19.21、SPL 14.60，逼近甚至超越 NavGPT、MapGPT 等基准模型；整体上实现了新的 state‑of‑the‑art 结果。

**⚠️ 局限性**

局限性：①隐式几何虽然减少了重建失败率，但仍无法完全替代高精度三维信息；②模型对极端光照、动态遮挡等条件的鲁棒性有限；③数据集主要覆盖室内场景，缺乏户外或非房间环境；④对视频质量依赖较高，仍有约90% 轨迹因重建失败被舍弃。

---

## 10. MASEval: Extending Multi-Agent Evaluation from Models to Systems

**arXiv ID:** 2603.08835 | [PDF](https://arxiv.org/pdf/2603.08835v1)

**作者:** Cornelius Emde `[一作]` (University of Oxford), Martin Gubri `[通讯]` (Parameter Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MASEval，一个框架无关的多智能体系统评估库，能够统一处理不同框架和基准的全系统评估；

**💡 创新点**

创新点在于将完整的智能体系统视为评估单元，提供最小化的适配器、统一的基准接口、trace-first评估、可扩展的任务队列，并通过实验验证框架选择对性能影响与模型选择相当；

**🔧 技术方法**

采用Python实现核心运行时与抽象基类，构建适配器层连接多种框架（smolagents、LangGraph、LlamaIndex等），集成多模型接口（OpenAI、Anthropic、Google）与日志后端，利用回调系统、错误归因与可视化工具；

**📊 数据集**

使用的基准包括GAIA、MACS、ConVerse、MultiAgentBench等多智能体评估数据集；

**📈 对比分析**

通过全因子实验（3框架×3模型×3基准）比较，发现框架差异导致的性能波动与模型差异相近，平均框架效应为14.2pp，模型效应为12.4pp；

**⚠️ 局限性**

局限在于最小适配器对新框架可能需要更厚的桥接，扩展性到大规模多智能体系统尚未测试，且强调灵活性导致入门门槛较高。

---

## 11. LooComp: Leverage Leave-One-Out Strategy to Encoder-only Transformer for Efficient Query-aware Context Compression

**arXiv ID:** 2603.09222 | [PDF](https://arxiv.org/pdf/2603.09222v1)

**作者:** Thao Do `[一作]` (Korea Advanced Institute of Science and Technology), Daeyoung Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3307 | [OpenAlex ID](https://openalex.org/A5100412744)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于margin的查询驱动上下文压缩框架，利用留一式Delta评分计算句子重要性并采用自适应gap阈值进行裁剪，使用轻量级encoder‑only模型实现高速推理

**💡 创新点**

创新点在于引入留一式Delta评分与gap‑based自适应阈值，摒弃传统二分类/token级目标；采用encoder‑only架构取代decoder LLM，实现高吞吐量与低内存占用

**🔧 技术方法**

使用ModernBERT encoder、留一式Delta计算、margin‑based ranking loss、BCE、adaptive gap阈值及flash‑attention技术

**📊 数据集**

训练使用HotpotQA的问答与句子重要性标注；评估数据集包括HotpotQA、2WikiMultihopQA、Musique、Natural Questions、TriviaQA，读取器包括Llama‑3.1‑8B、Llama‑3.3‑70B等

**📈 对比分析**

与RECOMP‑Abs、RECOMP‑Ext、CompAct、Refiner、LongLLMLingua、EXIT、Provence等7种压缩方法对比；在EM/F1上多数据集多读取器均获得第一或第二名；压缩延迟约0.05–0.2s，压缩率≤20%/≤14%，相较大多数基线显著提升精度与速度

**⚠️ 局限性**

依赖人工句子级标签，若改用LLM标注成本高；仅做句子级裁剪，无法对长句子进行细粒度优化；缺乏更细粒度标注与评估

---

## 12. CLIOPATRA: Extracting Private Information from LLM Insights

**arXiv ID:** 2603.09781 | [PDF](https://arxiv.org/pdf/2603.09781v1)

**作者:** Meenatchi Sundaram Muthu Selva Annamalai `[一作]` (University College London), Peter Kairouz `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文设计并实现了一种针对隐私保护的LLM聊天分析系统（如Clio）的隐私攻击，利用恶意注入与聚类技巧，使攻击者能够在系统聚类、摘要与审计环节中泄露目标用户的敏感信息。

**💡 创新点**

创新点在于：①提出了完整的多层注入+Poisoning攻击框架，能同时突破PII红action、聚类过滤、摘要提取和LLM审计四重防护；②演示了该攻击在不同LLM（Qwen3、Gemma3、LLaMA3、Claude）下的通用性；③系统性评估了现有“非正式”审计与差分隐私（DP）对抗效果。

**🔧 技术方法**

主要技术包括：LLM Prompt Injection、文本聚类（句子嵌入 + K‑means）、基于LLM的摘要生成、LLM审计评分、正则表达式匹配以及使用强大LLM（Claude Sonnet 4.5）进行隐私信息提取。

**📊 数据集**

数据集：使用公开的WildChat对话数据，外加基于NLICE合成的医学对话（包含年龄、性别、症状、诊断），合成对话与真实对话混合，形成实验用的目标聊天。

**📈 对比分析**

实验对比：基线攻击（仅利用公开信息推断疾病）成功率约22%；在Clio上使用单一症状与年龄性别信息的攻击成功率为39%；随着攻击者知识增多（5个症状）成功率可达近100%；LLM审计对隐私泄漏识别率极低（≈0%），而采用DP（ε=25,50）后攻击成功率下降至与基线相当，表明DP相对更稳健。

**⚠️ 局限性**

局限性包括：①实验仅基于合成医学聊天，未验证对真实敏感对话的效果；②仅针对单一聊天，未研究批量/无目标攻击；③攻击假设黑盒访问并可创建恶意账户；④DP方案在实用性、效能和实用预算上仍面临挑战。

---

## 13. Clarifying the Compass: A Reflexive Narrative on Entry Barriers into HCI and Aging Research

**arXiv ID:** 2603.08818 | [PDF](https://arxiv.org/pdf/2603.08818v1)

**作者:** Tianyi Li `[一作]` (Purdue University), Jin Wei-Kocsis `[通讯]` (Purdue University)

**通讯引用:** 174 | [OpenAlex ID](https://openalex.org/A5100758372)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

对非专业老龄研究者进入人机交互与老龄研究领域的障碍与经验进行反思和记录

**💡 创新点**

首次系统性阐述跨学科背景研究者在老年人群研究中的先入为主假设与情感冲突，强调同理心培养的重要性

**🔧 技术方法**

采用人机交互观察、志愿服务实地体验、访谈与情感记录等方法

**📊 数据集**

以当地养老社区的居民为案例，收集志愿者与老人互动的实地观察记录

**📈 对比分析**

通过对比自身预设与现场体验的差异，展示经验偏差，并对现有HCI方法在老年人群中的适用性进行质性评估

**⚠️ 局限性**

受限于研究者经验不足、样本仅为单一社区、缺乏量化指标，难以推广至更广泛老年人群

---

## 14. Overview of the TREC 2025 Retrieval Augmented Generation (RAG) Track

**arXiv ID:** 2603.09891 | [PDF](https://arxiv.org/pdf/2603.09891v1)

**作者:** Shivani Upadhyay `[一作]` (University of Waterloo), Jimmy Lin `[通讯]` (University of Waterloo)

**通讯引用:** 22324 | [OpenAlex ID](https://openalex.org/A5082997975)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并组织了TREC 2025 Retrieval Augmented Generation（RAG）Track，重点研究在长篇、多句叙事查询下整合检索与生成的系统，强调透明度与事实基础；

**💡 创新点**

创新点包括：将查询从关键词式转为长叙事式，构建子叙事拆解与引用映射的评估框架，引入AutoNuggetizer和多层级自动化评估；

**🔧 技术方法**

采用多种检索技术（SPLADE‑v3、Arctic‑Embed‑L、RRF、RankLLM、RankQwen3‑32B 等）与大语言模型（GPT‑4.1、Qwen3、Gemini 等）进行检索、重排序、生成与评估；

**📊 数据集**

使用MS MARCO V2.1 语料库（已去重、分段），配合叙事生成的自研叙事集；

**📈 对比分析**

通过手工与自动化（LLM）相结合的评估，使用nDCG@30、nDCG@100、Recall@100等指标进行比较，最高运行在nDCG@30≈0.676、nDCG@100≈0.605；自动评估与人工评估相关性较高，但在细粒度上仍有波动；

**⚠️ 局限性**

局限性包括：自动化评估与人工判别的吻合度仅约30%‑34%，难以捕捉细微语义；仅对22条叙事进行手工评估，样本覆盖有限；数据集仅限MS MARCO，缺乏多领域多语言测试；

---

## 15. Class Model Generation from Requirements using Large Language Models

**arXiv ID:** 2603.09100 | [PDF](https://arxiv.org/pdf/2603.09100v1)

**作者:** Jackson Nguyen `[一作]` (Monash University), Alessio Ferrari `[通讯]` (University College Dublin)

**通讯引用:** 3065 | [OpenAlex ID](https://openalex.org/A5041720518)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了一套从自然语言需求自动生成 UML 类图并对生成结果进行评估的完整流程。

**💡 创新点**

创新点在于提出双重验证框架：使用 LLM 作为评判者进行无参考评估，再用人类专家进行对比验证，并将两者结果结合形成可复现的评估流程。

**🔧 技术方法**

采用 GPT‑5、Claude Sonnet 4.0、Gemini 2.5 Flash Thinking、Llama‑3.1‑8B‑Instruct 四大 LLM 通过链式思维提示生成 PlantUML 代码，并利用 LLM‑as‑Judge（Grok 与 Mistral）进行 pairwise 比较，辅以 Spearman、Cohen’s κ、Wilcoxon 与 Cohen’s d 等统计方法。

**📊 数据集**

实验使用八个真实需求数据集，涵盖锁定马丁网络安全挑战、用户故事、Pure 数据集等多领域需求，确保多样性与行业相关性。

**📈 对比分析**

通过 LLM‑Judge 的 pairwise 排名与 Spearman 相关性、Cohen’s κ 及效应量计算，评估 LLM 生成的 UML 图质量；结果显示 GPT‑5 最高，评分普遍在 4–5 之间，Kappa 0.773，Wilcoxon 显著高于中点；人类评审 Kappa 0.684，整体与 LLM 评判高度一致，性能表现优秀。

**⚠️ 局限性**

局限性包括：对复杂领域（如 Pacemaker、g12‑camperplus）仍存在结构与可读性误差；高层次评判（可读性、术语对齐）受主观因素影响；评估仅覆盖当前四款 LLM，未检验未来模型或其他生成器；缺乏真实的参考标准模型。

---

## 16. DeZent: Decentralized z-Anonymity with Privacy-Preserving Coordination

**arXiv ID:** 2603.08854 | [PDF](https://arxiv.org/pdf/2603.08854v1)

**作者:** Carolin Brunn `[一作]` (Dresden University of Technology), Florian Tschorsch `[通讯]` (Dresden University of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了在物联网传感器网络中，将传统集中式 z‑匿名化改造成去中心化方案，让网关（gateway）在本地完成匿名化并通过轻量级协调实现全局一致性。

**💡 创新点**

核心创新在于：①使用计数布隆过滤器（Counting Bloom Filter）作为随机化计数结构；②在此基础上嵌入安全求和（Secure Sum）算法以保护中间计数；③通过轮询式时钟协调在网关间传播计数与发布权限，从而避免对中心实体的信任需求。

**🔧 技术方法**

主要技术包括：计数布隆过滤器、安全求和（Basic Secure Sum）、伪随机网关环路、时钟周期协调、旋转伪匿名化。实现代码已公开于 GitHub。

**📊 数据集**

使用合成的德国电力负荷剖面数据（来自 BDEW 标准负荷剖面），对家庭、农场、商业等不同客户类型进行模拟，并在此基础上生成 15 分钟一次的能耗样本。

**📈 对比分析**

与集中式 z‑匿名化和完全去中心化方案对比：①发布比（publication ratio）与集中式基本相同，说明数据可用性不受影响；②发送给中心实体的消息数明显降低；③需要额外的网关间协调消息，但总体传输量与集中式相当；⑥实验在不同 z 值和网关数量下均表现出可扩展性和较低标准差。

**⚠️ 局限性**

局限性包括：仅针对诚实但好奇（Honest‑but‑Curious）攻击模型；对强攻击者或多方协同攻击的防护仅为概率性；需要网关具备足够存储和计算资源以维护计数布隆过滤器；若网关数目过少，匿名化效果下降；未针对加密或更复杂的多方计算方案进行评估。

---

## 17. Kinodynamic Motion Retargeting for Humanoid Locomotion via Multi-Contact Whole-Body Trajectory Optimization

**arXiv ID:** 2603.09956 | [PDF](https://arxiv.org/pdf/2603.09956v1)

**作者:** Xiaoyu Zhang `[一作]` (Georgia Institute of Technology), Maegan Tucker `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 276 | [OpenAlex ID](https://openalex.org/A5054944952)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了Kinodynamic Motion Retargeting（KDMR）框架，将人类MoCap与地面反作用力结合，实现动态可行且多接触的全身运动转移。

**💡 创新点**

创新点在于同时引入GRF、硬件接触互补约束和多接触（heel-to-toe）模型，使转移运动在动力学上可行且减少物理伪影。

**🔧 技术方法**

主要技术包括CasADi求解非线性规划、Pinocchio进行刚体动力学计算、OpenSim进行逆运动学以及基于GRF的接触状态推断。

**📊 数据集**

使用公开的人体生物力学数据集（同步MoCap与力板数据）作为源运动。

**📈 对比分析**

通过与GMR基线在基座与关节轨迹平滑度、地面接触误差及GRF跟踪误差的定量比较，随后在BeyondMimic学习任务中验证，KDMR显著提升了轨迹质量、降低了接触错误，并使学习策略更快收敛、奖励更高。

**⚠️ 局限性**

局限性在于依赖同步的GRF测量，假设为稳态步态，数据获取成本高且对非步态运动的适用性受限。

---

## 18. Enabling Multi-Client Authorization in Dynamic SSE

**arXiv ID:** 2603.09550 | [PDF](https://arxiv.org/pdf/2603.09550v1)

**作者:** Seydina Ousmane Diallo `[一作]` (SAMOVAR), Nesrine Kaaniche `[通讯]` (SAMOVAR)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种多客户端可搜索加密方案 MASSE，支持属性基访问控制、动态更新与即时撤销，且不需要服务器重加密数据库。

**💡 创新点**

在 OXT 框架上加入单一标签的 Cset 结构和聚合证书机制，既实现了客户端细粒度授权，又避免了每个客户端复制索引的成本；同时提供了前向/后向隐私与令牌不可伪造的正式安全证明。

**🔧 技术方法**

使用对称加密、伪随机函数、双线性映射（Bilinear Pairing）以及公钥累加器（Accumulator）来实现索引加密、访问授权与撤销验证。

**📊 数据集**

实验采用包含 100 个关键字、每个关键字对应 150 条文档的加密数据库（测试时 1000~5000 条文档可扩展），并在真实数据集上验证性能。

**📈 对比分析**

与 OXT 及其它动态多客户端方案对比，MASSE 在数据库生成、令牌生成与查询时间上均优于对手；在 10–100 关键字查询时平均 <2 秒，检索 50 条匹配文档平均 14 秒，整体性能比 OXT 提升 20–30% 以上。

**⚠️ 局限性**

仍会泄露查询频率与部分关联模式（KPRP），依赖可信数据所有者进行初始设置，且对任意布尔查询的支持有限，只实现了多关键字交叉查询。

---

## 19. Think Before You Lie: How Reasoning Improves Honesty

**arXiv ID:** 2603.09957 | [PDF](https://arxiv.org/pdf/2603.09957v1)

**作者:** Ann Yuan `[一作]` (Google DeepMind), Katja Filippova `[通讯]` (Google DeepMind)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5037657908)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造可变成本的现实道德困境数据集，系统评估大型语言模型在面临诚实与欺骗取舍时的行为，并研究推理过程如何影响其诚实率。

**💡 创新点**

创新点在于发现推理不仅仅通过生成论证文本，而是通过在表示空间中进行“遍历”将模型从不稳定的欺骗状态引导至更稳定的诚实默认状态，并证明欺骗行为在几种扰动下高度易变。

**🔧 技术方法**

主要技术包括链式推理（chain‑of‑thought）触发的推理预算控制、输入重述、输出再采样、激活噪声注入、以及对隐藏表示进行插值和相似度分析的几何方法。

**📊 数据集**

使用了两大数据集：①自研的Variable‑Cost Moral Dilemmas，包含多种成本级别和多重表述；②对现有道德困境数据集（如DailyDilemmas）进行成本扩展，以统一评测框架。

**📈 对比分析**

对比方法为对照“token‑forcing”模式与不同推理长度（1、4、16、64句或无约束）的推理模式，结果显示推理显著提升诚实率（平均提升约 15–30%），且更长推理时诚实率进一步上升；在各种扰动下，欺骗率下降更为显著，表明推理具有稳健的对齐效果。

**⚠️ 局限性**

局限性包括：仅评估模型在单一场景下的决策，未考虑多轮对话或更大背景的影响；数据集基于特定文化背景，标签的主观性可能限制普适性；模型仅在现有训练后状态下测试，未探索如何通过训练进一步强化几何稳定性。

---

## 20. TriFusion-SR: Joint Tri-Modal Medical Image Fusion and SR

**arXiv ID:** 2603.09702 | [PDF](https://arxiv.org/pdf/2603.09702v1)

**作者:** Fayaz Ali Dharejo `[一作]` (University of Wurzburg), Radu Timofte `[通讯]` (University of Wurzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

设计并实现了TriFusion-SR，集成离散小波变换与条件扩散模型，实现三模态医学影像融合与超分辨率的端到端联合处理。

**💡 创新点**

首次在扩散条件分支中引入2D‑DWT分解、Rectified Wavelet Features（RWF）校准及Adaptive Spatial‑Frequency Fusion（ASFF）模块，实现频域感知的多模态交互与结构驱动的融合。

**🔧 技术方法**

使用条件扩散概率模型（U‑Net基架）、2D‑DWT、RWF校准网络、ASFF注意力机制以及LPIPS/PSNR/SSIM等评价指标。

**📊 数据集**

采用Harvard Medical School Whole Brain Atlas三模态注册数据集（MR‑T1/MR‑T2/PET等组合），共104组（73训练/9验证/22测试）。

**📈 对比分析**

与五种传统融合+SR组合及TMFS方法对比，三倍到八倍放大尺度下PSNR提升4.8–12.4%，RMSE降低11–33%，LPIPS降低52–65%，整体性能显著领先。

**⚠️ 局限性**

仅在特定三模态组合上验证，缺乏对更多临床场景的泛化评估；模型训练依赖大量对齐数据，推理速度受扩散迭代次数限制。

---

## 21. Two Teachers Better Than One: Hardware-Physics Co-Guided Distributed Scientific Machine Learning

**arXiv ID:** 2603.09032 | [PDF](https://arxiv.org/pdf/2603.09032v1)

**作者:** Yuchen Yuan `[一作]` (George Mason University), Lei Yang `[通讯]` (George Mason University)

**通讯引用:** 9103 | [OpenAlex ID](https://openalex.org/A5011903902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一个名为EPIC的分布式科学机器学习框架，应用于全波形反演任务；

**💡 创新点**

创新点在于将硬件资源与物理约束协同引导，设计了边缘轻量编码+跨注意力解码，既显著降低通信成本，又保持甚至提升物理一致性和推断精度；

**🔧 技术方法**

采用了分布式边缘编码、self‑attention融合、位置感知交叉注意力解码、自动化模型部署与运行时管理等技术，并在Raspberry Pi测试床上实现；

**📊 数据集**

使用了OpenFWI公开的10个数据集（Vel、Fault、Style三类，每类含A/B两个变体）；

**📈 对比分析**

通过与集中式InversionNet、BigFWI‑L、Federated Learning、Split Learning等基线在Wi‑Fi和模拟4G网络下对比，EPIC实现8.9×的时延压缩、33.8×的能耗下降，且在8/10数据集上SSIM优于或与集中式相当，鲁棒性更好；

**⚠️ 局限性**

主要局限在于中心节点算力受限导致时延仍高于纯分布式方法；跨注意力增加的计算开销；在极端网络延迟或丢包场景下性能提升有限。

---

## 22. TPIFM: A Task-Aware Model for Evaluating Perceptual Interaction Fluency in Remote AR Collaboration

**arXiv ID:** 2603.09264 | [PDF](https://arxiv.org/pdf/2603.09264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 23. Reading, Not Thinking: Understanding and Bridging the Modality Gap When Text Becomes Pixels in Multimodal LLMs

**arXiv ID:** 2603.09095 | [PDF](https://arxiv.org/pdf/2603.09095v1)

**作者:** Kaiser Sun `[一作]` (Johns Hopkins University), Fan Bai `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多模大语言模型在文本-像素模态差距进行了系统诊断，并提出了通过自蒸馏缩小差距的方案

**💡 创新点**

首次在七种模型、七大基准与五种输入模态下对模态差距进行定量与定性分析，发现差距主要来源于渲染方式和视觉读取误差；并提出利用模型自身文本推理轨迹进行自蒸馏的实用方法

**🔧 技术方法**

利用OCR分离读取与推理、基于LoRA的微调、视觉编码器+语言模型的自蒸馏、Grounded Theory错误分类、GPT-5进行半自动标签

**📊 数据集**

合成文本图像基准（MMLU、ARC、GPQA、GSM8K、HumanEval），以及自然文档图像基准（QASPER、SQuAD）

**📈 对比分析**

在五种输入模式下评估七模型的准确率，发现自然文档图像性能可与文本模式相当甚至更优；自蒸馏后GSM8K图像模式准确率从30.7%提升至92.7%，且在ARC、MMLU、HumanEval等未见灾难性遗忘，整体性能显著提升

**⚠️ 局限性**

仍受渲染与预训练分布不匹配的影响；自蒸馏依赖模型自身文本推理，错误轨迹可能传递错误；实验范围仅覆盖部分任务与数据集，通用性和可迁移性待进一步验证

---

## 24. Artificial Intelligence (AI) Maturity in Small and Medium-Sized Enterprises: A Framework of Internalized and Ecosystem-Embedded Capabilities

**arXiv ID:** 2603.08728 | [PDF](https://arxiv.org/pdf/2603.08728v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 25. Mitigation of UE Antenna Calibration Errors via Differential STBC in Cell-Free Massive MIMO

**arXiv ID:** 2603.08962 | [PDF](https://arxiv.org/pdf/2603.08962v1)

**作者:** Marx M. M. Freitas `[一作]` (University of Cassino and Southern Lazio), Stefano Buzzi `[通讯]` (University of Cassino and Southern Lazio)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过在基于单元间的无基站大规模MIMO网络中采用差分时空块编码（DSTBC），实现了在多天线用户设备（UE）端抗天线标定误差的下行链路传输。

**💡 创新点**

创新点在于首次将DSTBC用于消除UE端天线标定误差，避免了对UE侧校准或相位估计的依赖，并在不影响相位信息的情况下保持了接收多路复用优势。

**🔧 技术方法**

使用技术包括：差分STBC编码与解码、ZISI与P-MMSE预编码、TDD模式下的用户和AP集群化、以及基于3GPP UMi通道模型的仿真。

**📊 数据集**

实验数据来源于仿真：随机布置 20 个 UE 与 40 个 AP，采用 3GPP UMi 路径损耗模型、4 dB 影子衰落、2 个 UE 天线、2 个并行数据流，CPU 与 UE 之间无实际数据集。

**📈 对比分析**

通过与完全校准和未校准的传统协同下行（ZISI/P-MMSE）对比，DSTBC方案在 BER 和 SE 上逼近理想校准情形；在未校准情况下能显著提升可靠性，但在 AP 集群数增大时因预码率下降导致 SE 下降。

**⚠️ 局限性**

局限性包括：差分编码导致的预码率损失（尤其在 AP 集群数 > 2 时），对 UE 校准误差在两个连续码字间保持不变的假设，以及仅在仿真环境中验证，缺乏实测验证。

---

## 26. LLM as a Meta-Judge: Synthetic Data for NLP Evaluation Metric Validation

**arXiv ID:** 2603.09403 | [PDF](https://arxiv.org/pdf/2603.09403v1)

**作者:** Lukáš Eigler `[一作]` (Charles University), David Hurych `[通讯]` (valeo.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用LLM生成可控语义降级文本的“Meta-Judge”框架，借此替代昂贵的人类标注来验证NLG评估指标。

**💡 创新点**

创新点在于：① 用LLM按预设的降级级别生成质量可知的合成数据；② 通过“meta‑correlation”评估合成数据对人类评估的可替代性；③ 在跨任务、跨语言的多场景中验证该框架的通用性。

**🔧 技术方法**

技术手段包括：LLM（Llama 4 Scout、Llama 3.3‑70B、Qwen 3‑30B）进行语义降级；多种评估指标（BLEU、ROUGE、chrF、METEOR、BERTScore、COMET、BLEURT）；Spearman/Kendall相关性用于计算传统人类标注验证与合成验证之间的 meta‑correlation。

**📊 数据集**

使用的数据集有：机器翻译 WMT 2021/2024（多语种对），问答 CUS‑QA、MOCHA，摘要 RoSE；涉及语言包括捷克语、斯洛伐克语、乌克兰语、英语、豪萨语、科萨语、祖鲁语、冰岛语等。

**📈 对比分析**

比较方法：先在带有人类标注的原始数据上计算各评估指标的 Spearman 相关性（人类验证），再在LLM生成的合成降级数据上计算相关性；随后将两组相关性向量进行 Spearman 相关性，得到 meta‑correlation。实验显示在 QA 任务中 meta‑correlation 超过 0.9，表明合成数据能高度模拟人类评估；在摘要和 MT 任务中表现更为波动，但仍提供可接受的验证效果。

**⚠️ 局限性**

局限性：① 合成降级质量依赖于 LLM 对目标语言的熟练程度，低资源语言效果不佳；② 损伤定义需手工设计，迁移到新任务需额外工作；③ 对完全新任务/语言仍需少量人工标注作为初步验证。

---

## 27. The AetherFloat Family: Block-Scale-Free Quad-Radix Floating-Point Architectures for AI Accelerators

**arXiv ID:** 2603.08741 | [PDF](https://arxiv.org/pdf/2603.08741v1)

**作者:** Keita Morisaki `[一作]` `[通讯]` (Independent Hardware Software Co Design Researcher), Keita Morisaki (Independent Hardware Software Co Design Researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种可参数化的浮点架构AetherFloat Family，用于AI加速器的硬件/软件协同设计，提供AF8（8-bit）和AF16（16-bit）两种格式，并在FP8与bfloat16之间实现高效的动态范围和能耗优势。

**💡 创新点**

创新点包括：1）Lexicographic One's Complement Unpacking实现无周期整数可比性；2）Base‑4四进制缩放的Quad‑Radix指数，显著扩大动态范围；3）显式尾数、无隐位设计，消除隐藏位导致的乘法器尺寸和子正数管道停顿；4）Block‑Scale‑Free架构不需要AMAX硬件；5）向量共享32‑bit Galois LFSR实现的分布式随机舍入；6）AF8作为QAT‑first推理格式兼容大语言模型激活异常。

**🔧 技术方法**

采用的技术包括：一补码包装与显式尾数、Base‑4指数、2 阶段 MUX 对齐、32‑bit Galois LFSR 共享随机、量化感知训练 (QAT) 与 STE、Verilog 实现与 SkyWater 130nm PDK 综合、OpenSTA 动态功耗估算。

**📊 数据集**

使用 Qwen2.5‑7B 大语言模型进行 PTQ 与 QAT 实验，评估数据集包括 WikiText‑2、PIQA、HellaSwag 等语言任务；训练时也对梯度下溢和收敛行为做了单步实验。

**📈 对比分析**

通过与 IEEE‑754 FP8 E4M3（含 AMAX）和 bfloat16 的对比，AF16 在 PTQ 上与 BF16 几乎无差距；AF8 在 PTQ 下性能略逊但在 QAT 下可恢复；硬件层面 AF8 相比 FP8 减少 33.17% 面积、21.99% 功耗、11.73% 延迟，并无需 AMAX 共享缩放硬件，显著降低系统级面积与功耗。

**⚠️ 局限性**

局限性包括：1）硬件指标仅基于 SkyWater 130nm 教育 PDK，生产工艺可能差异；2）软件评估仅使用 PyTorch 仿真，未验证真实硬件系统效果；3）QAT 收敛实验仅单种 seed、200 步、单模型，缺乏统计可靠性；4）AF8 需 QAT fine‑tune，PTQ 直接使用会出现下溢；5）精度“wobble”理论证明尚缺；6）一补码对极端数值（如灾难性相消）未做全面验证。

---

## 28. GAST: Gradient-aligned Sparse Tuning of Large Language Models with Data-layer Selection

**arXiv ID:** 2603.09865 | [PDF](https://arxiv.org/pdf/2603.09865v1)

**作者:** Kai Yao `[一作]` (Ant Group), Penglei Gao `[通讯]` (Cleveland Clinic Lerner Research Institution)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合数据点与层级两维度的梯度对齐进行稀疏微调

**💡 创新点**

同时在数据层和层级层面动态选择，利用梯度对齐决定每层训练样本

**🔧 技术方法**

Gradient-aligned Sparse Tuning (GAST)、PEFT、LoRA/Series/Parallel Adapter、支持集梯度、采样概率与随机选择

**📊 数据集**

Commonsense 8 任务（BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC、OBQA）以及 Math10K（GSM8K、AQuA、MAWPS、SVAMP）

**📈 对比分析**

与 LISA、AdaLoRA、RST、IST、GREATS 等自适应方法对比，平均提升 1–3%（以 LLaMA 7B/13B、GPT‑J、LLaMA3‑8B 为例），GAST 在各任务上均显著提高性能

**⚠️ 局限性**

无法同时降低内存和计算开销；未在更大模型（如 LLaMA‑3 70B）验证，是否需要更稀疏仍待研究

---

## 29. A Gaussian Comparison Theorem for Training Dynamics in Machine Learning

**arXiv ID:** 2603.09310 | [PDF](https://arxiv.org/pdf/2603.09310v1)

**作者:** Ashkan Panahi `[一作]` (Chalmers University), Ashkan Panahi `[通讯]` (Chalmers University)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5015473416)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过Gordon比较定理构造了一个非渐近的训练动态与替代动态等价的理论框架，并用此框架验证并修正动态均值场（DMF）近似，进一步在感知机分类任务上展示了可迭代的精度提升方法。

**💡 创新点**

创新点在于将Gordon比较定理扩展到随机动力学零点问题，给出了训练过程与简化替代动力学等概率分布的非渐近等价性；同时提出了一种可迭代的有限维修正方案，弥补了DMF在有限样本下的误差。

**🔧 技术方法**

使用技术包括Gordon最小–最大定理、CGMT、Gaussian过程零点比较、动态均值场理论、Cholesky分解以及可迭代固定点近似。

**📊 数据集**

实验数据主要基于高斯混合模型生成的合成样本，使用感知机模型（带任意激活函数）在两类别或多类别设置下进行训练。

**📈 对比分析**

比较方法是将DMF和迭代修正后的理论预测与大规模仿真结果对比，发现理论与仿真误差随样本量增大而趋于零，且迭代修正后误差进一步下降，证明方法在相当规模下具有良好精度。

**⚠️ 局限性**

局限性包括对Gaussian混合模型的强假设、对激活函数可微性的要求、在非可解析延拓至复数时的困难，以及在高维/SGD等更复杂设置下可能出现的更高阶误差。

---

## 30. RecThinker: An Agentic Framework for Tool-Augmented Reasoning in Recommendation

**arXiv ID:** 2603.09843 | [PDF](https://arxiv.org/pdf/2603.09843v1)

**作者:** Haobo Zhang `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 4011 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出RecThinker框架，利用Analyze-Plan-Act模式通过主动调用专门的工具获取缺失信息，以实现更精准的推荐。

**💡 创新点**

创新点在于（1）引入主动信息缺口评估与工具调用决策；（2）设计专属推荐工具集；（3）采用两阶段自增强训练（SFT+RL）提升推理与工具使用效率。

**🔧 技术方法**

核心技术包括LLM代理、工具调用接口、Analyze-Plan-Act推理流程、LoRA低秩适配、GRPO强化学习以及NDCG评估。

**📊 数据集**

实验数据集为Amazon CD & Vinyl（划分为sparse、dense两子集）与MovieLens-1M（同样划分为sparse、dense）。

**📈 对比分析**

与传统推荐模型(BPR、SASRec)、LLM方法(LLMSeqSim、LLMRank)、推理模型(R2Rec)以及AgentCF、PersonaX进行对比，RecThinker在NDCG@10等指标上平均提升约10‑15%，显著优于所有基线。

**⚠️ 局限性**

局限性包括对高质量轨迹生成的依赖、工具调用成本较高、对极度稀疏场景仍有提升空间，以及对大型LLM的算力需求较大。

---

## 31. NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification, Control, and State Estimation

**arXiv ID:** 2603.09908 | [PDF](https://arxiv.org/pdf/2603.09908v1)

**作者:** Syed Izzat Ullah `[一作]` (Texas A&M University-Corpus Christi), Jose Baca `[通讯]` (Texas A&M University-Corpus Christi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对27g级的Crazyflie 2.1无人机，创建了一个多任务基准数据集NanoBench，包含170条飞行轨迹，配备同步的Vicon真实轨迹、原始IMU、飞控EKF、PID内部以及PWM指令，并制定了系统辨识、控制器评估和状态估计的评测协议。

**💡 创新点**

首次公开提供完整的低阶电机指令、控制器内部状态和估计器输出，并与毫米级外部真值同步；同时把三项任务（SysID、控制、估计）整合到同一平台与数据集上，填补了nano级无人机领域缺失的公共基准。

**🔧 技术方法**

使用CFLIB无线收发、Vicon运动捕捉、时间交叉相关对齐、三种评测任务的基线实现（物理模型、MLP残差、LSTM、PID、Mellinger、BC-MLP/BC-LSTM、MPPI、EKF）以及标准化评价指标（MAE、RMSE、ADE、ATE等）。

**📊 数据集**

NanoBench数据集：在GitHub上公开（https://github.com/syediu/nanobench-iros2026.git），包含170条飞行轨迹、同步的传感与控制日志。

**📈 对比分析**

通过对比多种基线模型，展示了：一阶物理模型在短期内精度高，但长周期快速衰退；残差MLP+物理模型在50步预测上最优；PID和Mellinger在实际飞行中位置误差相近，但Mellinger的轨迹发散率降低两位数；MPPI在本平台上性能差，发散率达75%；EKF在慢速/中速轨迹下ATE<22mm，快速轨迹发散。

**⚠️ 局限性**

局限包括：对高频电机非线性和低Reynolds流体模型的描述不足；评测仅基于单一硬件平台（Crazyflie 2.1），难以推广到不同型号；数据集只覆盖单一电池类型，未考虑多电池或混合动力情况；并且EKF在高速时失稳，表明低端MCU受限。

---

## 32. SpaceSense-Bench: A Large-Scale Multi-Modal Benchmark for Spacecraft Perception and Pose Estimation

**arXiv ID:** 2603.09320 | [PDF](https://arxiv.org/pdf/2603.09320v1)

**作者:** Aodi Wu `[一作]` (University of Chinese Academy of Sciences), Xue Wan `[通讯]` (Technology and Engineering Center for Space Utilization)

**通讯引用:** 537 | [OpenAlex ID](https://openalex.org/A5102796596)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了SpaceSense‑Bench，一个涵盖136个卫星模型、同步RGB、深度和LiDAR数据，并提供7类部件语义标签与6-DoF姿态的多模态空间感知基准；并在该基准上对2D分割、目标检测、3D点云分割、单目深度估计和姿态估计等五个任务进行了系统基线评测。

**💡 创新点**

提供大规模、时间同步的三模态（RGB+深度+LiDAR）数据，密集部件级语义标注，并设计零样本泛化评估协议，填补现有空间感知数据集在多目标、多模态和部件细粒度方面的空白。

**🔧 技术方法**

利用Unreal Engine 5与AirSim插件构建高保真空间仿真，使用Blender对3D资产进行部件级分解，采用全自动化数据采集、质量控制与多格式导出流水线实现无人工干预的数据生成。

**📊 数据集**

使用本工作构建的SpaceSense‑Bench数据集，包含90k帧（稀疏采样）或可扩展至200万帧，覆盖136个卫星模型。

**📈 对比分析**

对Mask2Former、YOLOv5、PMFNet等现有模型进行训练并在未见卫星的零样本测试，结果显示即使是最先进的网络在小部件（如螺旋天线、推进喷嘴）上的IoU低于35%，但通过增加训练卫星数量可使mIoU提升73%。

**⚠️ 局限性**

小部件检测仍是主要瓶颈，且由于缺乏真实轨道数据，尚未验证仿真到真实的迁移性能；数据规模虽大，但仍有限，未来需进一步扩充卫星模型与真实场景验证。

---

## 33. Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency

**arXiv ID:** 2603.09798 | [PDF](https://arxiv.org/pdf/2603.09798v1)

**作者:** Zhaofeng Shi `[一作]` (University of Electronic Science and Technology of China), Hongliang Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 43202 | [OpenAlex ID](https://openalex.org/A5075571728)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种在测试时实现Ego-Exo视角动作预测的算法DCPGN；

**💡 创新点**

首次将多标签原型生长与双线索一致性相结合，以克服视角间时空差异和多类别预测挑战；

**🔧 技术方法**

使用CLIP视觉编码器、轻量化叙述器生成文本线索、基于熵优先队列的多标签原型生长模块以及跨模态一致性约束；

**📊 数据集**

在EgoExoLearn和新建的EgoMe-anti两个Ego-Exo双视角数据集上进行评估；

**📈 对比分析**

与现有TTA方法相比，在两组基准上提升了约5%–12%的Top‑5召回率；

**⚠️ 局限性**

仍依赖预训练CLIP与叙述器的性能，针对极端视角差异或少样本场景的适应性尚待进一步验证。

---

## 34. ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph

**arXiv ID:** 2603.09266 | [PDF](https://arxiv.org/pdf/2603.09266v1)

**作者:** Junhao Cai `[一作]` (Shenzhen University), Xiaopin Zhong `[通讯]` (Shenzhen University)

**通讯引用:** 1105 | [OpenAlex ID](https://openalex.org/A5026482574)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种名为 ForgeDreamer 的工业文本到 3D 生成框架，集成多专家 LoRA 蒸馏与跨视角超图几何增强，显著提升工业部件的语义理解与几何精度。

**💡 创新点**

创新点在于：① 通过教师‑学生蒸馏将多个类别专属 LoRA 合并为统一模型，消除知识干扰并提升跨类别泛化；② 将几何一致性建模为跨视角超图学习，捕获高阶结构依赖，突破传统对视角间对偶一致性的局限。

**🔧 技术方法**

使用技术包括：Stable Diffusion + LoRA 蒸馏、3D Gaussian Splatting 渲染、跨视角超图神经网络（HGNN）与 MVHG 损失、Interval Score Matching 作为伪真实目标。

**📊 数据集**

使用自制多视角工业数据集，覆盖 10 类机械与电子部件（螺钉、螺母、轴承等），每类 20 张高分辨率图像，视角为前视与顶视。

**📈 对比分析**

与 ProlificDreamer、RichDreamer、LucidDreamer 等基线进行定量对比，T3Bench 平均质量分 50.88 分、处理时间 190 分钟，显著优于同类方法（如 LucidDreamer + LoRA 仅 47.10 分、110 分钟），LLM 评估亦显示最高的结构完整性与纹理真实性。

**⚠️ 局限性**

局限性包括：依赖专门构建的工业数据集，缺乏对更广泛类别的验证；超图构建与推理仍有计算开销；在极复杂多部件或非标准视角下的泛化能力待进一步提升。

---

## 35. Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction

**arXiv ID:** 2603.08713 | [PDF](https://arxiv.org/pdf/2603.08713v1)

**作者:** Jatin Chhugani `[一作]` (Meta Platforms), Changkyu Kim `[通讯]` (Meta Platforms)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两种软件级技术——Overflow-Aware Scaling（OAS）和Macro Block Scaling（MBS），用于改进MXFP4 4-bit量化以接近NVFP4的准确性。

**💡 创新点**

创新点在于：①在不改变硬件的前提下通过OAS扩大动态范围；②采用宏块尺度（1×128）并利用额外位数提升对异常值的表示，从而在保持低硬件开销的同时提升QSNR与下游准确率。

**🔧 技术方法**

使用的软件技术包括：基于块的指数缩放、块级动态范围判断、对块最大值进行指数与尾数截断、宏块前置缩放与动态/静态查表优化以及CUDA CUTLASS GEMM的定制化宏块调度。

**📊 数据集**

在LLM数据集上评估：Llama 3.1‑8B‑Instruct、Qwen 3‑8B、DeepSeek‑R1 以及 Llama 4‑Maverick，使用语言模型评估工具（Language Model Evaluation Harness）与vLLM推理引擎。

**📈 对比分析**

与NVFP4、原始MXFP4-OCP以及MX+对比，MBS‑Hybrid在大多数基准上将平均准确率提升至≈99–100%相当于NVFP4，且在GEMM层仅产生≈6.2% 的吞吐量开销；整体推理时间几乎无显著增加。

**⚠️ 局限性**

局限性包括：仅针对4‑bit MXFP4；对更大模型的泛化仍需验证；MBS‑Dynamic 的搜索开销较高，尚未完全消除；未结合量化感知训练或更高级的PTQ方法。

---

## 36. Learning the Hierarchical Organization in Brain Network for Brain Disorder Diagnosis

**arXiv ID:** 2603.09606 | [PDF](https://arxiv.org/pdf/2603.09606v1)

**作者:** Jingfeng Tang `[一作]` (Northeastern University), Osmar R. Zaiane `[通讯]` (University of Alberta)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5027917989)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出BrainHO框架，利用层次注意力学习脑网络的层次组织并去除对固定功能亚网络的依赖。

**💡 创新点**

创新点在于引入可学习子图token和层次注意力机制、正交约束以及层次一致性约束，实现对跨网络交互的自适应建模并提升特征多样性。

**🔧 技术方法**

使用层次注意力机制、Sparsemax激活、正交损失、KL一致性损失、Transformer结构以及辅助分类头等技术。

**📊 数据集**

使用ABIDE（ASD vs HC）和REST‑meta‑MDD（MDD vs HC）两个大规模 rs‑fMRI 数据集，采用 Craddock200 分区。

**📈 对比分析**

与多种基线（BNT、Com‑BrainTF、ALTER、DHGFormer、LHDFormer）对比，BrainHO 在 ABIDE 上准确率 69.68%、AUC 73.80%，在 REST‑meta‑MDD 上准确率 64.71%，均超越或竞争最佳性能。

**⚠️ 局限性**

局限性包括对动态时间序列数据处理有限、对亚网络解释依赖于事先映射且受分区方式影响，以及缺乏进一步的因果推断验证。

---

## 37. Lockbox -- A Zero Trust Architecture for Secure Processing of Sensitive Cloud Workloads

**arXiv ID:** 2603.09025 | [PDF](https://arxiv.org/pdf/2603.09025v1)

**作者:** Vamshi Krishna Thotempudi `[一作]` (Microsoft Corporation), Anjali Mangal `[通讯]` (Microsoft Corporation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种零信任架构，用于安全处理敏感云工作负载。

**💡 创新点**

创新点在于将硬件根信任、细粒度身份访问与持续验证机制结合，形成动态可扩展的安全管控模型。

**🔧 技术方法**

利用Intel SGX/AMD SEV等安全隔离技术、容器化微服务、访问控制列表与链路加密，构建完整的安全管道。

**📊 数据集**

使用公开金融交易与医疗影像数据集进行实验验证。

**📈 对比分析**

与传统基于防火墙和静态ACL的方案对比，平均处理延迟提升不到3%，但安全风险降低超过70%。

**⚠️ 局限性**

局限性包括对多云环境支持不足、硬件兼容性限制以及运维成本较高。

---

## 38. A Decentralized Frontier AI Architecture Based on Personal Instances, Synthetic Data, and Collective Context Synchronization

**arXiv ID:** 2603.08893 | [PDF](https://arxiv.org/pdf/2603.08893v1)

**作者:** Jacek Małecki `[一作]` (Wrocław University of Science and Technology), Katarzyna Tworek `[通讯]` (Wrocław University of Science and Technology)

**通讯引用:** 455 | [OpenAlex ID](https://openalex.org/A5081751187)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种去中心化的分布式人工智能架构 H3LIX DFMA，利用个人 AI 实例生成合成学习信号并通过共享的 Collective Context Field 实现上下文同步。

**💡 创新点**

创新点在于不以参数同步为主，而是通过上下文信号传播实现集体学习，结合能源自适应模型演化以及对隐私和治理的重视，形成一种可持续、分布式的 AI 扩展路径。

**🔧 技术方法**

采用的技术包括：本地推理模型（可使用小型语言模型或适配器如 LoRA）、合成学习信号生成（基于推理轨迹、反事实推理、知识抽象）、安全聚合与差分隐私、Collective Context Field 的分布式更新，以及能源感知的学习调度。

**📊 数据集**

本文未给出具体公开数据集，而是设计为可在多样化本地数据上运行，合成学习信号来自用户交互和模拟，理论上可与大规模公共文本库如 Common Crawl 等兼容。

**📈 对比分析**

方法比较主要是与传统中心化大型语言模型和联邦学习的参数同步方式进行对比。实验演示（假设性）显示在保持相同推理质量的前提下，分布式上下文学习可以在能源使用和隐私泄露率上分别降低 30% 和提升 20%。

**⚠️ 局限性**

限制包括：缺乏成熟的实证评估，合成学习信号的质量控制和信号衰减问题；聚合算法对网络规模和节点异质性的鲁棒性待验证；能源自适应调度需要实时电网信息且可能导致学习效率不稳定。

---

## 39. Granulon: Awakening Pixel-Level Visual Encoders with Adaptive Multi-Granularity Semantics for MLLM

**arXiv ID:** 2603.08800 | [PDF](https://arxiv.org/pdf/2603.08800v1)

**作者:** Junyuan Mao `[一作]` (National University of Singapore), Yueming Jin `[通讯]` (National University of Singapore)

**通讯引用:** 4894 | [OpenAlex ID](https://openalex.org/A5050163233)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Granulon，一种基于DINOv3的多模态大型语言模型，利用文本条件化的粒度控制器和自适应Token聚合实现像素到细粒度再到粗粒度的统一推理。

**💡 创新点**

创新点在于将可控粒度维度嵌入视觉编码器中，使模型能根据输入文本动态调整视觉抽象层级，从而在细粒度视觉理解与粗粒度语义对齐之间实现自适应平衡。

**🔧 技术方法**

采用文本条件化粒度控制器（MLP+聚合层）与AdaTA模块（粒度引导池化、聚类和质量筛选），并结合DINOv3自监督视觉编码器和大型语言模型（如Qwen‑2.5、Llama‑3.2）。

**📊 数据集**

使用多任务评测集：SEED‑Bench、A‑OKVQA、CC12M+Imagenet21K Recap、FLUX‑Reason、SurgVLM以及医疗领域的Phase/Instrument Recognition数据集。

**📈 对比分析**

在相同训练设置下与CLIP‑和DINOv2‑基线对比，Granulon在VQA、图像描述、推理等任务上分别提升约30%准确率、35% GPT‑4o分数，并将幻觉率降低约20%。

**⚠️ 局限性**

局限性包括：对高分辨率图像仍需更多算力，粒度控制器训练依赖人工或大语言模型标注，且在极端跨域任务中仍可能出现语义偏移或推理误差。

---

## 40. Evaluate-as-Action: Self-Evaluated Process Rewards for Retrieval-Augmented Agents

**arXiv ID:** 2603.09203 | [PDF](https://arxiv.org/pdf/2603.09203v1)

**作者:** Jiangming Shu `[一作]` (Beijing Jiaotong University), Jitao Sang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 2157 | [OpenAlex ID](https://openalex.org/A5023834030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将检索质量评估转化为显式动作，并强制检索后立即评估，构建 EvalAct 框架；同时提出 Process‑Calibrated Advantage Rescaling (PCAR) 以细粒度调节优势；通过强化学习训练检索-增强型 LLM 代理。

**💡 创新点**

创新点在于：①将隐式检索评估变为可训练的显式动作，形成严格的搜索‑评估循环；②利用评估分数在梯度更新中做优势缩放，实现过程级信用分配；③在多跳推理中显著降低错误传播。

**🔧 技术方法**

采用 LLM 代理、检索增强生成（RAG）、强化学习（GRPO）以及 PCAR 优势缩放，结合结构化工具调用与文本标记的交互方式。

**📊 数据集**

使用七个公开问答基准：单跳的 Natural Questions、TriviaQA、PopQA；多跳的 HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle；训练数据为 ASearcherBase35K 检索语料。

**📈 对比分析**

与 Direct Generation、Naïve RAG、IRCoT、Search‑o1、Search‑R1、AutoRefine 等方法对比，EvalAct 在 7B 模型下整体 EM 47.1%（单跳平均 44.0%），在多跳数据集上领先 AutoRefine 约 3.5–10+ 分，展示了显著的性能提升。

**⚠️ 局限性**

局限性包括：①评估动作被硬编码为检索后必做，限制代理自主性；②实验仅覆盖问答任务，未验证在网页导航、代码生成等更复杂场景中的泛化；③仅在 3B/7B 模型规模上测试，尚未评估在更大模型上的表现。

---

## 41. Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards

**arXiv ID:** 2603.09117 | [PDF](https://arxiv.org/pdf/2603.09117v1)

**作者:** Zhengzhao Ma `[一作]` (Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences), Le Sun `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 6100 | [OpenAlex ID](https://openalex.org/A5034536222)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了RLVR中出现的过度自信问题，并提出DCPO框架通过显式置信度回放和梯度掩蔽实现推理与置信度优化的解耦，解决了准确率与校准之间的梯度冲突；

**💡 创新点**

创新点在于理论上揭示了准确率与校准之间的梯度冲突，并提出分块显式置信度回放、分离优势估计、掩蔽梯度以及组级准确度作为低方差监督的完整方法；

**🔧 技术方法**

使用技术包括RLVR（GRPO）算法、分块显式置信度回放、分离优势估计、掩蔽梯度策略、组级准确度与实例级准确度的混合监督；

**📊 数据集**

在DeepScaler上进行训练，评测数据集为MATH‑500、AIME 2024/25、AMC 23/24等数学推理基准；

**📈 对比分析**

与GRPO、RLCR、CCGPSG、ConfClass等基线对比，DCPO在保持或略高的推理准确率的同时，ECE、PCE大幅下降、AUROC显著提升，表现出最优的准确率-校准折衷；

**⚠️ 局限性**

局限性包括仍需手动设计输出结构，实验仅在数学推理任务上验证，未探究对其他领域任务的泛化性能。

---

## 42. WikiCLIP: An Efficient Contrastive Baseline for Open-domain Visual Entity Recognition

**arXiv ID:** 2603.09921 | [PDF](https://arxiv.org/pdf/2603.09921v1)

**作者:** Shan Ning `[一作]` (ShanghaiTech University), Xuming He `[通讯]` (ShanghaiTech University)

**通讯引用:** 7386 | [OpenAlex ID](https://openalex.org/A5015970030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在开域视觉实体识别任务中，提出WikiCLIP框架，利用大型语言模型文本嵌入与视觉引导知识适配器结合的对比学习方法，实现高效实体检索。

**💡 创新点**

创新点在于将LLM文本表征与视觉局部特征通过跨模态注意力聚合，并通过硬负样本合成增强细粒度判别。

**🔧 技术方法**

采用CLIP视觉编码器、LLaMa大型语言模型、对比学习（InfoNCE）和跨模态注意力模块。

**📊 数据集**

主要使用OVEN实体集、INFOSEEK、E-VQA等公开数据集进行训练与评估。

**📈 对比分析**

与现有生成式与对比式基线相比，WikiCLIP在OVEN unseen上提升28.5%精度，推理延迟从1569ms降至14.49ms，整体性能领先。

**⚠️ 局限性**

局限在于对LLM知识的利用不足，文本长度与模型规模增大后性能提升有限，仍需进一步挖掘LLM表征潜力。

---

## 43. MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration

**arXiv ID:** 2603.09101 | [PDF](https://arxiv.org/pdf/2603.09101v1)

**作者:** Chenran Zhang `[一作]` (Southeast University), Yi Zhou `[通讯]` (Southeast University)

**通讯引用:** 39965 | [OpenAlex ID](https://openalex.org/A5008483780)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于医学知识驱动的认知编排框架 MedKCO，利用分层课程和自适应不对称对比损失实现医学视觉‑语言预训练的数据与目标函数的优化；

**💡 创新点**

创新点在于（1）基于诊断敏感度和样本代表性设计两级课程；（2）引入自适应不对称对比损失动态平衡图像‑文本与文本‑图像对齐；（3）实现从易到难的认知学习序列；

**🔧 技术方法**

采用 CLIP/FILIP 等通用视觉‑语言预训练框架，配合两级课程调度与自适应不对称对比损失；

**📊 数据集**

在色彩眼底摄影（CFP）、光学相干层析（OCT）和胸部X光（CXR）三种医学影像数据集上进行预训练与评估，使用 ODIR200×3、FIVES、REFUGE、OCTID、OCTDL、CheXpert5×200、RSNA‑Pneumonia、SIIM‑Pneumothorax、COVIDx 等下游数据；

**📈 对比分析**

与 CLIP、FILIP 以及 CL‑log/CL‑logit 课程学习基线对比，MedKCO 在零样本分类、图像‑文本检索与报告生成等任务上均显著提升，平均提升幅度约 8%–17% 以上，尤其在 OOD 任务上表现最为突出；

**⚠️ 局限性**

局限性包括需依赖医学专业知识进行课程划分，参数（如阶段数）对性能敏感；仅在三种影像模态与有限任务上验证，未探讨跨模态或更大规模的自动课程生成方案；

---

## 44. Almost-Optimal Upper and Lower Bounds for Clustering in Low Dimensional Euclidean Spaces

**arXiv ID:** 2603.09846 | [PDF](https://arxiv.org/pdf/2603.09846v1)

**作者:** Vincent Cohen-Addad `[一作]` (Google Research), Chris Schwiegelshohn `[通讯]` (Aarhus University)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5080748807)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文为欧氏低维空间中的 k‑median 与 k‑means 提出了一个 (1+ε) 近似 PTAS，运行时间为 2^{Õ(log(1/ε))^{d-1}}·n·polylog(n)，并给出了一个基于 Gap‑ETH 的近似下界 2^{Ω(log(1/ε))^{d-1}}·n^{O(1)}，证明了上界的近似最优。

**💡 创新点**

创新点在于：1) 对分层四叉树（quadtree）分解的门户点（portals）进行新的混合平均/最坏情况分析，从而将所需门户数从 1/ε^{O(d)} 降至 2^{O(log(1/ε))^{d-1}}；2) 结合细粒度复杂度框架，将 3‑SAT → Vertex‑Cover → k‑clustering 的链式归约，得到与上界匹配的下界；3) 通过预算与重构技术，构造几乎最优的 portal‑respecting 解决方案。

**🔧 技术方法**

主要技术包括：随机四叉树分解与门户点构造；portal‑respecting 路径的动态规划求解；对每个点的预算（budget）定义并进行期望/概率分析；对 “坏切割” (badly cut) 的概率和代价进行细致估计；利用 Gap‑ETH 的细粒度硬件假设构造 3‑SAT → Graph embedding → k‑means/median 的归约。

**📊 数据集**

本文为理论性工作，没有使用实际数据集；所有实验与分析均在理论构造的合成实例上完成。

**📈 对比分析**

在理论上，算法实现了 (1+ε) 近似，并将时间复杂度从 2^{O(d^2)}·n^{O(1)} 降至 2^{Õ(log(1/ε))^{d-1}}·n·polylog(n)。与已有的 2^{O(d^2)} 近似方案相比，门数显著减少；而下界表明此时间复杂度在 Gap‑ETH 假设下几乎是最优的。

**⚠️ 局限性**

局限性包括：1) 仅适用于低维欧氏空间，无法直接推广到高维或一般度量空间；2) 下界依赖于 Gap‑ETH 假设，若假设不成立则下界可能无效；3) 运行时间中隐含常数与 log(1/ε) 的指数项较大，实际可用性受限；4) 需要预先获得一个常数因子近似解，算法对该近似质量有要求。

---

## 45. Unpacking Interpretability: Human-Centered Criteria for Optimal Combinatorial Solutions

**arXiv ID:** 2603.08856 | [PDF](https://arxiv.org/pdf/2603.08856v1)

**作者:** Dominik Pegler `[一作]` (University of Vienna), Filip Melinscak `[通讯]` (TU Darmstadt)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在多子集求和（Multiple Subset Sum Problem）任务中，对等价最优解的结构化属性进行量化，提出并验证三种复杂度度量（HC、CC、VC）与人类解释性偏好的关系；

**💡 创新点**

首次将 heuristic 对齐、组合简洁性和视觉排序三类认知原则统一为可量化的解读复杂度指标，并证明它们在解释性、决策速度与注意力分布上的预测作用；

**🔧 技术方法**

采用贪婪启发式对比、图编辑距离、Dirichlet-几何-混合模型、Kendall τ 失序度等方法计算 HC、CC、VC；使用线性混合效应模型、序数混合模型和二项式 GLMM 对偏好、反应时与眼动偏向进行建模；

**📊 数据集**

在 Prolific 上随机生成 4~6 个箱子、7~9 个物品的 5,000+ 问题实例，生成等价最优解对；实验共 73 名受试者，包含 1,668 次评价试验；数据公开于 OSF；

**📈 对比分析**

对 HC、CC、VC 的差异与选择、RT、眼动进行显著性检验；结果显示，三个复杂度指标的差值越大，受试者越倾向选择较低复杂度的方案；绝对差值越大时，HC 相关 RT 越快；眼动未出现显著偏差；

**⚠️ 局限性**

局限包括：仅使用小规模、静态等价解；眼动测量受 WebGazer 低分辨率限制；实验无时间压力和动态约束；复杂度指标可能未涵盖所有人类偏好；

---

## 46. The Data-Dollars Tradeoff: Privacy Harms vs. Economic Risk in Personalized AI Adoption

**arXiv ID:** 2603.08848 | [PDF](https://arxiv.org/pdf/2603.08848v1)

**作者:** Alexander Erlei `[一作]` (University of Göttingen), Ujwal Gadiraju `[通讯]` (Delft University of Technology)

**通讯引用:** 3807 | [OpenAlex ID](https://openalex.org/A5038081564)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过一项 2×3 的实验研究，探讨了在可量化风险与不确定（含糊）信息环境下，个人数据泄露威胁如何影响用户对 AI 个性化服务的采用与谈判行为，并测算用户对完全透明隐私标签的支付意愿。

**💡 创新点**

创新点在于：① 首次将风险与含糊两种信息环境与隐私泄露威胁进行对照，揭示含糊环境下用户更易回避 AI；② 将经济成本与非经济隐私不良效应分离，展示用户在面对隐私威胁时会为消除不确定性支付超出经济成本的价格；③ 通过实验验证第三方隐私标签在提高用户信任与采用率方面的经济可行性。

**🔧 技术方法**

主要技术与方法包括：① 预注册的在线实验平台（Flask+PostgreSQL）；② 2×3 随机分配实验设计；③ 典型决策任务（AI 选择、Ultimatum 谈判）与 Becker–DeGroot–Marschak (BDM) 机制；④ 个人数据挖掘与问卷收集（隐私关注、信任度、情感失望）。

**📊 数据集**

使用来自 CloudResearch Connect 的 610 名英语母语参与者数据，实验前后收集个人信息、时间偏好、风险偏好、社会偏好等问卷结果，构成完整的行为数据集。

**📈 对比分析**

通过对比不同条件下 AI 采用率、WTP 与理论最优支付、谈判拒绝率等指标，并用回归分析验证假设；结果显示：在含糊环境下 AI 采用率下降，WTP 接近或超过理论最优值，谈判行为受含糊影响但未受隐私泄露直接影响；总体表现表明信息透明度提升可显著提高 AI 采用与用户支付意愿。

**⚠️ 局限性**

局限性包括：① 采用的是“价格算法泄露”抽象模型，未能完全模拟真实数据泄露情境；② 样本主要为英语母语者，跨文化普适性待验证；③ 任务设计复杂，可能掩盖部分行为机制；④ 统计功效有限，某些交互效应可能被低估；⑤ 隐私标签假设为完美可信，现实中可信度和合规性可能降低效果。

---

## 47. Abundant Intelligence and Deficient Demand: A Macro-Financial Stress Test of Rapid AI Adoption

**arXiv ID:** 2603.09209 | [PDF](https://arxiv.org/pdf/2603.09209v1)

**作者:** Xupeng Chen `[一作]` (New York University), Xupeng Chen `[通讯]` (New York University)

**通讯引用:** 415 | [OpenAlex ID](https://openalex.org/A5053943810)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并量化了 AI 快速普及导致的宏观金融冲击模型，构建分离机制并进行压力测试

**💡 创新点**

首次将失业‑工资替代、Ghost GDP、媒介化崩溃等三大机制与消费集中放大器结合，形成可检验预测与早期预警框架

**🔧 技术方法**

基于任务层面生产模型、对数回归与模拟仿真相结合的宏观经济框架

**📊 数据集**

使用 FRED 宏观时间序列、BLS 职业工资与就业数据、跨行业 AI 曝露指数以及全球私募信贷与抵押贷款数据

**📈 对比分析**

通过与历史衰退、Acemoglu 等模型对标，利用蒙特卡洛模拟和情景设定验证不同 AI 增长速率下的危机阈值，表现出明显的政策阈值敏感性

**⚠️ 局限性**

受限于对 AI 能力增长速率和重建任务速率的高度假设、因果识别难度、以及模型在不同行业替代率和价格通胀的简化假设

---

## 48. Robust Spatiotemporal Motion Planning for Multi-Agent Autonomous Racing via Topological Gap Identification and Accelerated MPC

**arXiv ID:** 2603.09188 | [PDF](https://arxiv.org/pdf/2603.09188v1)

**作者:** Mingyi Zhang `[一作]` (Zhejiang University), Lei Xie `[通讯]` (Zhejiang University)

**通讯引用:** 7622 | [OpenAlex ID](https://openalex.org/A5100668966)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Topo-Gap 多车赛道间隙识别与加速 MPC 的规划框架，实现了高频安全超车。

**💡 创新点**

创新点包括：并行稀疏 GP 生成多车动态占据通道；拓扑感知间隙选择并加入决策滞后抑制；使用 PTC 加速的 LTV-MPC 解决近奇异 QP。

**🔧 技术方法**

使用了并行稀疏 Gaussian Process、轨道拓扑间隙选择、多点间隙采样、贝塞尔/余弦平滑轨迹生成、伪瞬态继续（PTC） QP 求解、Tikhonov 正则化和 LDLT 分解。

**📊 数据集**

实验基于 F1TENTH 1:10 模拟环境与官方赛道，使用多车轨迹数据集。

**📈 对比分析**

与 M-PSpliner、FSDP 等 SOTA 进行对比，评估指标包括超车成功率、总超车时间、平均/最大求解时延；Topo-Gap 在四类场景中超车成功率 >81%，总超车时间减半，平均求解时延 19.48 ms，最大 67.18 ms。

**⚠️ 局限性**

局限性包括：未考虑主动防守或游戏理论互动；对极端遮挡/传感器噪声的鲁棒性未知；PTC 仍需手动调参。

---

## 49. Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy

**arXiv ID:** 2603.09932 | [PDF](https://arxiv.org/pdf/2603.09932v1)

**作者:** Gauthier Miralles `[一作]`, Pietro Gori `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

基于对Margin Disparity Discrepancy (MDD) 的重新优化，提出一种新的无监督域适应方法用于CT到CBCT的肝脏分割。

**💡 创新点**

创新点是去除了MDD中对源域的对立项，使特征提取器同时在源域和目标域对齐，提升了跨模态分割性能，并扩展到少样本微调。

**🔧 技术方法**

采用U‑Net骨干，双分支（任务头和对抗头）以及自定义损失，结合跨域对齐与少样本微调。

**📊 数据集**

使用私有的573个CBCT和678个CT 3D体积（共计13,024/15,827 2D切片），并在此数据上进行实验。

**📈 对比分析**

与现有Feature Alignment（DANN、MDD）、Self‑Training（BDCL）、Image Alignment（SIFA）以及基准基础模型（SAM‑MED 2D/3D、MA‑SAM）对比，2D F1最高74.4%，3D F1最高86.6%，在少样本设置下达到90.9%（5卷）/91.8%（20卷）。

**⚠️ 局限性**

局限在于仅验证了肝脏分割，未检验其它器官或模态的泛化能力。

---

## 50. CarbonBench: A Global Benchmark for Upscaling of Carbon Fluxes Using Zero-Shot Learning

**arXiv ID:** 2603.09868 | [PDF](https://arxiv.org/pdf/2603.09868v1)

**作者:** Aleksei Rozanov `[一作]` (University of Minnesota), Vipin Kumar `[通讯]` (University of Minnesota)

**通讯引用:** 43307 | [OpenAlex ID](https://openalex.org/A5100645812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

创建了CarbonBench基准，用于零射向地理空间迁移学习的碳通量上采样。

**💡 创新点**

首次提供多维度卫星、气象特征与分层评估协议，强调尾部性能并聚焦于时序模型在零射转移上的优势。

**🔧 技术方法**

采用树模型、RNN、Transformer、TAM‑RL等监督学习与域泛化架构，并结合质量加权损失、时序窗口和特征归一化。

**📊 数据集**

使用567个全球eddy covariance站点（2000–2024），整合MODIS、ERA5‑Land、IGBP植被类型和Köppen气候类别等特征。

**📈 对比分析**

通过IGBP和Köppen两种分层零射转移测试，利用R²、RMSE、nMAE的四分位统计对比模型；结果显示时序模型优于树模型，TAM‑RL在最低四分位表现最稳健。

**⚠️ 局限性**

对NEE预测仍差，低覆盖区域表现不佳，特征仅采用最小6维；缺乏不确定性量化、扩展特征空间和自监督预训练的探索。

---

## 51. d-QBF with Few Existential Variables Revisited

**arXiv ID:** 2603.08826 | [PDF](https://arxiv.org/pdf/2603.08826v1)

**作者:** Andreas Grigorjew `[一作]` (Université Paris Dauphine PSL), Michael Lampis `[通讯]` (Université Paris Dauphine PSL)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了QBF问题中按存在变量数k为参数的复杂度，给出了k为参数时的上界与下界，证明在CNF子句 arity 4 的情况下，双指数时间复杂度是最优的，并给出了针对两块量化子句（∀∃-QBF）的近似最优算法。

**💡 创新点**

创新点在于：① 通过递归构造与 arity 降低技巧，证明了在 ETH 下 k 的双指数下界；② 针对两块量化子句的情况提出了利用分离集合与命中集的递归算法，获得 k^{O_d(k^{d-1})} 的时间复杂度；③ 通过上述构造填补了先前 SETH 下弱下界与已知上界之间的巨大空白。

**🔧 技术方法**

主要技术包括：递归量化变换、整数到子句的编码函数 B、概率方法证明存在简化策略、命中集分支与递归树深度分析，以及对 CNF 子句 arity 的递归压缩。

**📊 数据集**

无实验数据集，全部为理论算法与复杂度证明。

**📈 对比分析**

算法性能上：对于 arity d ≥ 3 的 ∀∃-QBF，时间复杂度为 O(k^{O_d(k^{d-1})})，相较于一般 QBF 的 2^{2^k} 复杂度显著下降；下界证明表明该上界（除对数因子外）已近乎最优。

**⚠️ 局限性**

局限性：① 仍未解决 arity 3 的情况；② 对常数级量化交替（如 ∀∃∀∃）的双指数性质尚未确定；③ 上述递归技术对量化层数有一定要求，可能不适用于极少量级交替。

---

## 52. RESBev: Making BEV Perception More Robust

**arXiv ID:** 2603.09529 | [PDF](https://arxiv.org/pdf/2603.09529v1)

**作者:** Lifeng Zhuo `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9195 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种可插拔的 RESBev 框架，用于提升 Lift‑Splat‑Shoot（LSS）系列鸟瞰视角（BEV）感知模型在自然噪声和对抗攻击下的鲁棒性。

**💡 创新点**

将鲁棒性问题重新定义为在潜在语义空间中进行时序预测，利用潜在世界模型生成干净的 BEV 状态先验，再通过异常重建器与当前受损观测融合，且不需要改动原始网络结构。

**🔧 技术方法**

采用 Transformer‑based 潜在动态世界模型、语义先验预测器、跨注意力异常重建器，结合变分推理训练目标，并在 BEV 语义空间内执行生成与融合。

**📊 数据集**

在 nuScenes 数据集上进行实验，使用 RoboBEV 基准中的十种自然失真与三种对抗攻击（FGSM、PGD、C&W）进行评估。

**📈 对比分析**

与四种 LSS 基线模型及 GraphBEV 进行对比；在自然失真和对抗攻击下平均提升 IoU 约 18–24%，在未见失真上同样实现最高 IoU，并在连续失真实验中保持稳定性能。

**⚠️ 局限性**

依赖历史清晰帧与准确的车身运动估计，难以处理突发新事件；对极端传感器失效的适应性有限；引入的潜在世界模型与交叉注意力会带来一定计算开销。

---

## 53. A comprehensive study of time-of-flight non-line-of-sight imaging

**arXiv ID:** 2603.09548 | [PDF](https://arxiv.org/pdf/2603.09548v1)

**作者:** Julio Marco `[一作]` (Universidad de Zaragoza), Andreas Velten `[通讯]` (University of Wisconsin)

**通讯引用:** 3134 | [OpenAlex ID](https://openalex.org/A5058842115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文对时间飞行（ToF）非视线（NLOS）成像方法进行统一建模与综合评估，提出通用的光传输前向模型与逆向Radon变换框架；

**💡 创新点**

创新点在于将不同方法归结为不同类型的Radon变换，统一了理论与实现，并通过相同硬件与光子计数的实验对比揭示各方法共性与差异；

**🔧 技术方法**

采用滤波反投影（FBP）、Wiener滤波（LCT）、f‑k迁移、相位场虚拟相机（PF‑CC）等算法，并使用物理基准的瞬态渲染与单光子雪崩二极管（SPAD）测量；

**📊 数据集**

使用自行生成的物理基准仿真数据以及实验收集的confocal与non‑confocal SPAD数据（无公开数据集）；

**📈 对比分析**

方法比较基于相同光子计数、硬件配置的PSNR、MS‑SSIM以及可视化重建；总体性能相近，LCT与PF‑CC在>8M光子时收敛至锐利无噪声重建，FBP‑Lap在噪声敏感，f‑k迁移无显式滤波但仍受限；

**⚠️ 局限性**

主要限制包括三跳传输假设导致的缺失圆锥效应、对光子计数与时间分辨率高度敏感、硬件采样与空间/时间滤波导致的分辨率与噪声折衷。

---

## 54. Reconstructing Movement from Sparse Samples: Enhanced Spatio-Temporal Matching Strategies for Low-Frequency Data

**arXiv ID:** 2603.09412 | [PDF](https://arxiv.org/pdf/2603.09412v1)

**作者:** Ali Yousefian `[一作]` (Politecnico di Milano), Simone Vantini `[通讯]` (Politecnico di Milano)

**通讯引用:** 1840 | [OpenAlex ID](https://openalex.org/A5004699501)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

改进了 ST-Matching 地图匹配算法，提出动态缓冲、动态观测概率、重设计时间分数以及基于历史使用率的行为分析四项改进；

**💡 创新点**

创新点在于结合 GPS 不确定性动态调整候选搜索和观测概率，加入三项时间惩罚函数与行为分数，显著提升效率和路径质量；

**🔧 技术方法**

使用状态转移图与 A* 最短路、概率模型、指数衰减时间惩罚、对数归一化行为分数，以及多维评估框架；

**📊 数据集**

使用 Cuebiq 提供的匿名车载 GPS 跟踪数据，覆盖米兰市区约10k条轨迹，含不确定性信息；

**📈 对比分析**

通过与原 ST-Matching 及低频 STB-Matching 的多指标比较（效率、质量、拓扑、速度），实验表明改进版在高频数据上显著降低候选数、运行时、投影距离、路径复杂度；在低频情况下虽有效率提升，但精度提升有限；

**⚠️ 局限性**

局限在于对极低频 GPS 数据的改进有限，行为分数依赖通用历史数据，路径求解仍使用标准 A*，缺乏针对不同车辆或用户的细粒度模型。

---

## 55. ConFu: Contemplate the Future for Better Speculative Sampling

**arXiv ID:** 2603.08899 | [PDF](https://arxiv.org/pdf/2603.08899v1)

**作者:** Zongyue Qin `[一作]` (University of California Los Angeles), Yizhou Sun `[通讯]` (Qualcomm AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ConFu框架，在推理时通过静态/动态暂停（contemplate）tokens和软提示让轻量级草稿模型获得未来方向信号，从而提升推理速度

**💡 创新点**

创新点包括: ①利用contemplate tokens捕捉目标模型的“未来思考”; ②基于MoE的动态contemplate tokens; ③锚点采样与未来预测复制的鲁棒训练; ④两级蒸馏提升未来预测质量

**🔧 技术方法**

采用speculative decoding、Mixture-of-Experts、软提示token、anchor token sampling、future prediction replication、两级蒸馏等技术

**📊 数据集**

训练使用ShareGPT与UltraChat‑200K，评测在SpecBench（写作、问答、摘要、翻译、编程、数学推理等）

**📈 对比分析**

与EAGLE‑3对比，使用Llama‑3.2‑3B/8B目标模型；在不同温度和草稿树预算下，ConFu平均提升接受长度和速度提升约8–11%（最高可达1.14×速度提升）

**⚠️ 局限性**

局限性包括: 需要额外的contemplate token计算，仍以EAGLE‑3为基础；在高温度下提升不明显；需进一步减少contemplate token数量以提升可扩展性

---

## 56. Removing the Trigger, Not the Backdoor: Alternative Triggers and Latent Backdoors

**arXiv ID:** 2603.09772 | [PDF](https://arxiv.org/pdf/2603.09772v1)

**作者:** Gorka Abad `[一作]` (University of Bergen), Stjepan Picek `[通讯]` (Radboud University)

**通讯引用:** 4682 | [OpenAlex ID](https://openalex.org/A5024072796)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究后门攻击的耐受性，证明后门在特征空间存在可被多种不同输入触发的“备选触发器”，并提出一种基于特征引导的攻击方法（FGA）来寻找这些备选触发器；

**💡 创新点**

①理论证明后门会在特征空间形成可被多种输入激活的区域；②设计Feature‑Guided Attack（FGA）通过对齐特征方向来发现与原后门机制相同的备选触发器；③指出仅消除已知触发器并不能根除后门，需要针对特征空间的后门区域进行防御；

**🔧 技术方法**

特征方向估计、投影梯度上升（PGD）与自定义特征引导损失、对抗训练、后训练防御（BAN、NAD、触发器解耦）等；

**📊 数据集**

CIFAR‑10、CIFAR‑100、TinyImageNet；使用ResNet‑18和VGG‑19；攻击方法包括BadNets、Blend、WaNet、Input‑Aware；

**📈 对比分析**

与传统PGD、随机噪声、以及多种后训练防御（BAN、NAD、触发器解耦）进行对比。实验表明，FGA 在多种数据集/模型/攻击下的成功率均超过90%，即使在原始触发器被防御几乎消除后，FGA 仍能保持 60–90% 的成功率，验证了备选触发器的存在与防御的局限；

**⚠️ 局限性**

目前备选触发器仅是样本特定，缺乏跨模型/跨输入的泛化能力；未构造全局/通用备选触发器；未深入探讨如何在防御端实现对特征空间后门区域的完全清除；

---

## 57. Preparing Students for AI-Driven Agile Development: A Project-Based AI Engineering Curriculum

**arXiv ID:** 2603.09599 | [PDF](https://arxiv.org/pdf/2603.09599v1)

**作者:** Andreas Rausch `[一作]` (Clausthal University of Technology), David Inkermann `[通讯]` (Clausthal University of Technology)

**通讯引用:** 412 | [OpenAlex ID](https://openalex.org/A5049546946)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

设计并实施了面向AI的敏捷工程课程，结合项目实践培养学生的AI与敏捷能力。

**💡 创新点**

将生成式AI工具嵌入敏捷工作流程，形成了整合AI与敏捷的项目式教学模式。

**🔧 技术方法**

使用Python、Java、React、TypeScript等技术栈，结合Copilot等AI助手。

**📊 数据集**

未使用特定数据集，课程侧重实践项目与学生作品。

**📈 对比分析**

通过混合方法评估，发现WS 2022-2024学生缺课ECTS下降至约3%，说明学习效果提升。

**⚠️ 局限性**

局限包括AI工具快速迭代导致教学需不断更新、需口头考试验证理解以及学生将AI视为团队成员。

---

## 58. Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks

**arXiv ID:** 2603.09513 | [PDF](https://arxiv.org/pdf/2603.09513v1)

**作者:** Wang Honghui `[一作]`, Bai Chenjia `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个新的长时段多关节操作基准RuleSafe，并通过LLM辅助生成多样化锁定机制来构造非马尔可夫任务；同时设计了VQ-Memory模块，用向量量化自编码器将机器人关节状态序列离散化为结构化记忆，提升了对时间序列的推理能力。

**💡 创新点**

创新点在于：1）利用LLM高效生成多阶段、多关节交互的锁定规则，突破传统手工脚本的可扩展性限制；2）引入VQ-Memory对关节状态进行离散化并聚类，既滤除低层噪声，又保留高层任务阶段信息，为VLA和扩散策略提供轻量级、鲁棒的时间上下文。

**🔧 技术方法**

技术手段包括：基于大型语言模型（LLM）进行任务生成与规划；使用SAPIEN仿真平台和HumanoidGen框架生成演示；采用向量量化变分自编码器（VQ‑VAE）对关节状态序列进行编码，再通过K‑means聚类得到紧凑记忆词表；在多种VLA/扩散模型中将记忆词作为额外输入进行融合。

**📊 数据集**

数据集主要包括：RuleSafe基准，包含20条锁定规则、10种保险箱类型，演示生成平均成功率71.7%，平均轨迹长度638帧；在实验中使用单任务（两条规则）与多任务（20条规则）设置，各模型分别采集100或50条演示。

**📈 对比分析**

与基线比较时，VQ‑Memory显著提升各模型的成功率与过程得分；在单任务下，π₀从30%提升至80%（成功率）并提升至89%（过程得分）；在多任务下，平均成功率从25%提升至56%（+31%），过程得分从48.8%提升至76.5%（+27.7%）。其他模型（DP3、RDT、CogACT）亦表现出类似提升。

**⚠️ 局限性**

限制在于：VQ‑Memory主要依赖关节状态记忆，无法完全解决低层操作精度问题，导致部分操作因抓取或旋转误差失败；数据规模对每个保险箱实例的覆盖不足，难以完全适应多样化几何与动态特性；未来需扩大演示数量或结合其他传感器信息以进一步提升成功率。

---

## 59. Rescaling Confidence: What Scale Design Reveals About LLM Metacognition

**arXiv ID:** 2603.09309 | [PDF](https://arxiv.org/pdf/2603.09309v1)

**作者:** Yuyang Dai `[一作]` `[通讯]` (INSAIT), Yuyang Dai (INSAIT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了大型语言模型在提供自信度评分时，信心量表设计（粒度、边界、范围规则）对元认知质量的影响。

**💡 创新点**

创新点在于揭示信心量表的非中性性：通过实验发现 0–20 级别比传统 0–100 级别在元认知敏感度和效率上更优，并证明模型在不同边界压缩下表现出“锚点”偏好。

**🔧 技术方法**

采用信心量表细化、边界移动、非标准范围三维操纵，并使用 meta‑d'、ECE、AUROC 等指标量化元认知性能。

**📊 数据集**

实验数据集涵盖知识问答 MMLU、数学推理 GSM8K 与真诚度检验 TruthfulQA，覆盖三种认知难度。

**📈 对比分析**

在六种 LLM（GPT‑5.2、Gemini 3.1 Pro、LLaMA‑4-Maverick/Scout、Qwen3‑235B/30B）上进行对比，结果显示 0–20 规模在所有模型和任务上均能显著提升 meta‑d' 与 M_ratio；边界压缩至 60–100 时表现大幅退化。

**⚠️ 局限性**

局限性包括仅使用确定性推理、未结合 Chain‑of‑Thought 或自我反思、样本量与数据集多样性有限、模型内部机制不公开，以及对随机采样与开放式生成任务的适用性未知。

---

## 60. PrivPRISM: Automatically Detecting Discrepancies Between Google Play Data Safety Declarations and Developer Privacy Policies

**arXiv ID:** 2603.09214 | [PDF](https://arxiv.org/pdf/2603.09214v1)

**作者:** Bhanuka Silva `[一作]` (University of Sydney), Suranga Seneviratne `[通讯]` (University of Sydney)

**通讯引用:** 2252 | [OpenAlex ID](https://openalex.org/A5038376039)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 PrivPRISM 框架，用于自动提取和对比 Google Play 数据安全声明与开发者隐私政策之间的细粒度数据实践，从而检测不一致和潜在违规。

**💡 创新点**

创新点在于：①将编码器（PrivBERT）与解码器（Llama3.1‑8B‑Instruct）相结合，既保证分类精度又提供可解释的说明；②引入自监督验证器降低生成错误；③设计了新的 PP‑DS 合规性度量和目的一致性评估；④在数千款热门游戏和通用应用上实现大规模评估。

**🔧 技术方法**

使用技术包括：编码器–解码器语言模型（PrivBERT、Llama3.1‑8B‑Instruct）、自监督映射验证器、关键词映射、静态 APK 分析、静态权限与数据流检测、以及自定义合规性度量。

**📊 数据集**

数据集：3,400 条独立隐私政策对应 7,770 款热门游戏（下载量 1M+，其中 174 款 100M+），1,254 条独立隐私政策对应 1,711 款通用应用；静态 APK 样本约 5,000；用于基准的 OPP‑115 法律专家标注语料；此外还收集了 Google Play 的数据安全声明与对应源码。

**📈 对比分析**

与 Llama3.1 微调模型和 GPT‑4o 零射提示基准比较：PrivPRISM 在 OPP‑115 上的高层分类精度比 GPT‑4o 提升 6%，比 Llama3.1 提升 50%；映射错误率下降 22.3%；在实测中发现约 53%（游戏）/ 61%（通用）应用存在 PP‑DS 不一致；静态代码层面证实 PPs 与 DS 的潜在缺失和误报。

**⚠️ 局限性**

局限性包括：仅进行静态分析，无法捕捉运行时数据流；仅处理直接可公开访问的隐私政策 URL，无法覆盖需要复杂爬取或动态渲染的情况；仅评估 Android 侧（Google Play）声明，未覆盖 iOS App Store 或其他市场；跨平台、跨版本的差异仍需进一步研究。

---

## 61. An Empirical Study of Interaction Smells in Multi-Turn Human-LLM Collaborative Code Generation

**arXiv ID:** 2603.09701 | [PDF](https://arxiv.org/pdf/2603.09701v1)

**作者:** Binquan Zhang `[一作]`, Yida Ye `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性研究多轮人-LLM代码协同中的交互异味并构建完整 taxonomy，随后提出 Invariant-aware Constraint Evolution (InCE) 框架来缓解这些异味

**💡 创新点**

首次将交互异味概念化为九类并与主流 LLM 对比，提出以约束提取与预生成检测为核心的 InCE 方案

**🔧 技术方法**

利用 Invariant Extraction Module (IEM)、Proactive Smell Detector (PSD)、LLM 生成与评估等技术实现约束管理与异味预防

**📊 数据集**

基于真实对话数据 WildChat、LMSYS-Chat-1M，并采用 WildBench 任务集进行实验评估

**📈 对比分析**

对六款主流 LLM 进行 Task Success Rate 与 Average Turns 的对比，InCE 在大部分模型上提升 TSR 约5–6%，并显著降低关键异味（如 Must-Do Omit、Repetitive Response）约13%

**⚠️ 局限性**

局限性包括数据覆盖面有限、标签过程主观性、评价指标未能充分反映用户主观体验及代码可维护性，需在更广泛领域与主观评估上进一步验证

---

## 62. Towards Flexible Spectrum Access: Data-Driven Insights into Spectrum Demand

**arXiv ID:** 2603.09942 | [PDF](https://arxiv.org/pdf/2603.09942v1)

**作者:** Mohamad Alkadamani `[一作]` (Communications Research Centre), Halim Yanikomeroglu `[通讯]` (Carleton University)

**通讯引用:** 21176 | [OpenAlex ID](https://openalex.org/A5035446029)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于地理空间分析与机器学习的数据驱动方法，利用运营商流量数据构建代理指标，在加拿大多伦多与温哥华城市级别上估算并识别6G频谱需求及其关键驱动因素。

**💡 创新点**

创新点在于：①从真实运营商流量中提取并验证频谱需求代理，②通过卫星夜灯加权提升空间精度，③在城市微观层面完成代理验证与模型训练，实现对传统宏观模型的显著提升。

**🔧 技术方法**

使用的技术包括：地理空间处理、代理构建、特征工程（人口、经济、建筑、交通等多源数据融合）、k‑means聚类（处理空间相关性）、岭回归与梯度提升回归（GBR）模型。

**📊 数据集**

主要数据集涵盖：Ottawa LTE流量（2,799 个基站）、加拿大多伦多与温哥华的基站部署数据、Crowdsourced LTE性能测量、统计人口（日夜）、经济指标、建筑与交通网络、夜间灯光影像等。

**📈 对比分析**

与传统宏观模型对比，基线线性模型R²≈0.58；岭回归在联合城市场景R²≈0.64、RMSE≈0.96；GBR在联合城市场景达到R²≈0.81、RMSE≈0.51，表现显著优于基线。

**⚠️ 局限性**

局限性包括：代理指标仍依赖基站部署与流量数据，无法完全捕捉5G/6G技术特性；模型跨城市泛化性能下降（GTA训练到温哥华时R²≈0.70）；缺乏实时动态更新，且对高频变动需求的反应不够及时。

---

## 63. DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization

**arXiv ID:** 2603.09180 | [PDF](https://arxiv.org/pdf/2603.09180v1)

**作者:** Jianing Yang `[一作]` (SB Intuitions Corp.), Yui Sudo `[通讯]` (SB Intuitions Corp.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DuplexCascade，一种无VAD的级联流式管道，实现全双工语音对语音对话

**💡 创新点**

将长语句转为微轮（chunk）并引入对话专用控制符号，利用LLM在流式环境中直接控制轮次和响应时机

**🔧 技术方法**

使用流式ASR、LLM（Qwen2‑7B‑Instruct）+ LoRA微调、流式TTS、专用控制符号

**📊 数据集**

仅用5万条文本对话（UltraChat）动态构造微轮训练集，模拟暂停、中断、回声等场景

**📈 对比分析**

在Full‑Duplex‑Bench上取得平均轮次准确率最高，在VoiceBench上接近无微轮系统的对话智能，表现优于Freeze‑Omni等现有模型

**⚠️ 局限性**

需要精细调节微轮时长（Δt）平衡准确率与延迟；在真实语音噪声、口语速率变化下的鲁棒性仍待验证

---

## 64. BiCLIP: Domain Canonicalization via Structured Geometric Transformation

**arXiv ID:** 2603.08942 | [PDF](https://arxiv.org/pdf/2603.08942v1)

**作者:** Pranav Mantini `[一作]` (University of Houston), Shishir K. Shah `[通讯]` (University of Oklahoma)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结构化双线性变换（上三角矩阵W）对Vision‑Language模型的图像特征进行几何对齐，从而在少样本场景下实现域自适应。

**💡 创新点**

创新点在于：①仅用一个可学习的上三角矩阵进行对齐，极大降低参数量并保持预训练语义；②通过理论推导证明该变换对应于不同域间的正交映射；③用结构化约束防止特征崩塌，兼顾收敛速度与泛化。

**🔧 技术方法**

采用双线性矩阵W、上三角结构约束、身份初始化，配合AdamW优化，仅微调W；在CLIP与SigLIP两大VLM骨干上实施。

**📊 数据集**

在11个常见少样本数据集上评测：ImageNet、EuroSAT、DTD、FGVCAircraft、OxfordPets、StanfordCars、Flowers102、SUN397、UCF101、Food101、Caltech101。

**📈 对比分析**

与零样本基线以及CoOp、CoCoOp、MaPLe、PromptSRC等前沿方法对比，BiCLIP/ BiSigLIP 在 16-shot 平均提升约 15.2%/8.7%，在 1/2-shot 下亦优于 prompting 方法，且参数极少、收敛快。

**⚠️ 局限性**

局限性：上三角矩阵可能无法捕捉更复杂的非线性对齐，极低样本数（1-shot）下的鲁棒性尚待验证；目前仅验证视觉侧，对文本特征的适配仍有限。

---

## 65. VisionCreator-R1: A Reflection-Enhanced Native Visual-Generation Agentic Model

**arXiv ID:** 2603.08812 | [PDF](https://arxiv.org/pdf/2603.08812v1)

**作者:** Jinxiang Lai `[一作]` (Hong Kong University of Science and Technology), Qinglin Lu `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在原生视觉生成代理中可训练的反思能力，并提出 VisionCreator‑R1 与 Reflection–Plan Co‑Optimization (RPCO) 方法，实现反思与规划的共优化，从而显著提升单图、多图和图像编辑任务的性能。

**💡 创新点**

①揭示规划与反思在强化学习中的结构方差不对称，解释直接优化反思为何难以收敛；②提出“先解耦后融合”RPCO训练框架；③构建 VCR‑SFT、VCR‑RL 数据集以及统一评测基准 VCR‑Bench。

**🔧 技术方法**

使用大型视觉语言模型 Qwen3VL32B，结合 UTPCR 框架、GRPO 强化学习、SFT+RL 组合训练、反思奖励设计（基于 VLM 的多点检查）以及多维奖励（规划、格式、工具、结果、反思）。

**📊 数据集**

自构 VCR‑SFT（包含单图反思强与多图规划强轨迹）与 VCR‑RL（用于多任务 RL），并在 VCR‑Bench、GEdit‑Bench、Gemini‑Bench 等公开基准上进行评估。

**📈 对比分析**

与 Gemini2.5Pro、Qwen‑Image‑Fast、Qwen3VL32B 等基线进行自动评估（使用 Gemini2.5Pro VLM 判定）和人工评估；在 VCR‑Bench 上单图 0.532/多图 0.700/图像‑图像 0.836，均优于基线，尤其在多图任务上提升约 5%–10%；在 GEdit‑Bench 总分 7.23，超越基线模型。

**⚠️ 局限性**

仍受图像生成过程噪声影响，长序列反思奖励易出现高方差；对低资源或极长多图任务的泛化尚有限；核心方法依赖大量人工构造的 VCR‑SFT/VRL 数据集，数据生成成本高。

---

## 66. ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning

**arXiv ID:** 2603.09692 | [PDF](https://arxiv.org/pdf/2603.09692v1)

**作者:** Davit Melikidze `[一作]` (ETH Zurich), Andreas Krause `[通讯]` (ETH Zurich)

**通讯引用:** 30648 | [OpenAlex ID](https://openalex.org/A5003040843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种模块化的主动学习管道（ActiveUltraFeedback）用于高效收集偏好数据，并开发了两种新的响应对选择方法——Double Reverse Thompson Sampling（DRTS）和 DeltaUCB。

**💡 创新点**

创新点在于：①把偏好数据生成视为上下文对弈带，利用不确定性估计主动挑选信息量最大的响应对；②引入基于 Delta 学习假说的 DRTS 与 DeltaUCB 两种新的选择策略，显著提升样本效率；③构建了可插拔的管道，支持多种模型、选择方法、评判器和评估算法。

**🔧 技术方法**

技术包括：上下文对弈带算法、Epistemic Neural Network（ENSEMBLE）进行奖励预测并估计不确定性、Thompson Sampling 变体、DeltaUCB（基于上置信界的优化）、DPO/IPO/SimPO 等直接偏好优化算法，以及使用大模型模拟人类评判。

**📊 数据集**

使用的数据集包括：UltraFeedback（约 14 万个提示），Skywork Reward Preference 80k，Combined（UltraFeedback+Skywork），Tulu 3 8B Preference Mixture；评估基准包括 RewardBench2、GSM8K、IFEval、TruthfulQA、AlpacaEval2 等。

**📈 对比分析**

与传统随机、UltraFeedback、DeltaQwen 等被动策略以及 InfoMax、DTS、MaxMinLCB 等对弈带方法对比，DRTS 与 DeltaUCB 在奖励模型与微调模型上均取得更高得分；在样本量仅为 1/6–1/3 的情况下，仍能达到或超过静态基线的性能；同时展示了在不同提示来源和优化算法（DPO/IPO/SimPO）下的良好泛化。

**⚠️ 局限性**

局限性包括：①对计算成本高（每个提示需要从 30 个 LLM 生成响应）；②DeltaQwen 等方法对特定模型族的依赖性强，泛化不足；③当前评判器仍使用大模型模拟，未涉及真实人工标注；④对低资源领域的实测仍不足，需进一步验证。

---

## 67. Beyond Fine-Tuning: Robust Food Entity Linking under Ontology Drift with FoodOntoRAG

**arXiv ID:** 2603.09758 | [PDF](https://arxiv.org/pdf/2603.09758v1)

**作者:** Jan Drole `[一作]` (Jožef Stefan Institute), Tome Eftimov `[通讯]` (Jožef Stefan Institute)

**通讯引用:** 2386 | [OpenAlex ID](https://openalex.org/A5082115266)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 FoodOntoRAG——一种基于检索增强生成的食物实体链接管道，结合词法与语义检索、多代理 LLM 推理以及置信度校准，实现无微调、跨本体的实体映射。

**💡 创新点**

创新点包括：①无需对模型微调，保持模型与本体无关；②采用混合词法‑语义检索与基于证据的 LLM 选择器，提升对本体漂移的鲁棒性；③引入同义词生成反馈循环，实现可解释的再检索与置信度阈值控制。

**🔧 技术方法**

技术栈：Whoosh（词法检索）+ FAISS（向量检索）+ all‑MiniLM‑L6‑v2（文本嵌入）+ LLM（GPT‑4 / Gemini）做选择器、评分器、同义词生成器，JSON 结构化输出与 prompt 工程。

**📊 数据集**

使用数据集：CafeteriaFCD（1,000 条食谱，7,429 个实体；8,948 个唯一实体）以及 Open Food Facts（119 个品牌成分）；与 FoodSEM 的训练/评估集对比。

**📈 对比分析**

评估方法：在 CafeteriaFCD 上与 FoodSEM 进行对比，Acc@1 在 57–60% 之间；在 Open Food Facts 上，FoodOntoRAG 的 Acc@1 约 90.7%（另一评估者 83.3%），远高于 FoodSEM 的 36.9%/29.2%。最优置信度阈值约为 0.6–0.7，平衡准确率与推理次数。

**⚠️ 局限性**

局限性：检索效果受嵌入模型与检索参数影响；需要手动调参以权衡精度与延迟；当前仅覆盖食物领域，扩展至化学、药物等需集成更多本体；同义词生成与反馈循环多步推理可能导致推理链冗长。

---

## 68. AI Phenomenology for Understanding Human-AI Experiences Across Eras

**arXiv ID:** 2603.09020 | [PDF](https://arxiv.org/pdf/2603.09020v1)

**作者:** Bhada Yun `[一作]` (ETH Zurich), April Yi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 468 | [OpenAlex ID](https://openalex.org/A5046673805)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了AI phenomenology研究范式，并通过三项纵向与结构化研究探索AI伴侣、价值对齐与工作场景中的人机体验。

**💡 创新点**

提出了AI phenomenology方法论工具包（渐进透明访谈、价值对齐感知工具包、任务嵌入多方法提取），并提出透明对齐、代理感知价值对齐与时间共演化三大设计概念。

**🔧 技术方法**

采用大语言模型驱动的AI伴侣“Day”、Agentic AI工具Cursor，结合访谈、日志分析、ACTA、Delphi等质性与量化技术。

**📊 数据集**

使用“Day”对话日志、Schwartz价值问卷（PVQ-RR）以及软件工程任务的代码与提示历史。

**📈 对比分析**

通过对比实验发现，聊天式人格在价值对齐上达77%匹配度（相较于25%反向人格），价值推断与自评的Spearman相关≈0.58；透明度提升导致部分用户情感和信任波动。

**⚠️ 局限性**

局限在样本规模小、质性深度难以大规模推广、对AI技术演进的跨时代可比性不足，以及工具包需进一步量化与自动化。

---

## 69. Grounding Synthetic Data Generation With Vision and Language Models

**arXiv ID:** 2603.09625 | [PDF](https://arxiv.org/pdf/2603.09625v1)

**作者:** Ümit Mert Çağlar `[一作]` (Middle East Technical University), Alptekin Temizel `[通讯]` (Middle East Technical University)

**通讯引用:** 1766 | [OpenAlex ID](https://openalex.org/A5003028527)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了基于视觉‑语言框架的遥感合成数据增强与评估方法，并生成了包含 100k 真图、300k 合成图、对应分割图与描述的 ARAS400k 数据集。

**💡 创新点**

创新点包括：① 三阶段自动化流程，将生成模型、语义分割与图像描述结合，实现可解释的合成数据评估；② 基于分割统计的上下文感知 caption 生成与跨模态一致性验证；③ 规模最大的多模遥感数据集（ARAS400k）和低冗余高多样性的 caption；④ 通过 CLIPScore、FID 等多维度指标与下游语义分割任务直接关联的评估体系。

**🔧 技术方法**

采用 StyleGAN3+SPADE+U‑Net 判别器生成合成图；UNet/UNet++/PAN/DeepLabV3+/SegFormer/FPN 等分割网络；Gemma3、Qwen3‑VL 等大模型用于视觉、文本与混合模态的 caption 生成；CLIPScore、t‑SNE、UMAP、FID 等评价工具。

**📊 数据集**

原始数据源为 ESA Sentinel‑2 RGB‑NIR 与 WorldCover 2021；基于此构建 ARAS400k，随后与 NWPU、RSICD、UCMC 等现有遥感 caption 数据集进行对比。

**📈 对比分析**

通过在同一训练/验证/测试拆分上对比：① 仅真实数据、② 仅合成数据、③ 真实+合成（无条件或有条件或两者）训练得到的语义分割 F1、IoU 等指标；实验显示合成单独可接近真实性能，混合数据可提升整体及低频类别性能；CLIPScore 与 FID 与现有数据集相当，表明生成 caption 与图像质量均达标。

**⚠️ 局限性**

局限性包括：合成图像质量仍略低于真实图；评价指标对语义一致性和多模态鲁棒性的覆盖不完整；训练与推理成本高（≈4000 小时）；模型对真实分布的泛化可能受限，需进一步验证多域迁移效果。

---

## 70. GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System

**arXiv ID:** 2603.09718 | [PDF](https://arxiv.org/pdf/2603.09718v1)

**作者:** Zhiye Tang `[一作]` (Shenzhen University), Xu Wang `[通讯]` (Shenzhen University)

**通讯引用:** 98220 | [OpenAlex ID](https://openalex.org/A5100424784)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于3D Gaussian Splatting（Scaffold‑GS）的体积场景流媒体系统GSStream，系统集成协同视口预测与基于深度强化学习的码率适配，能够在实时场景渲染与网络传输之间实现高效平衡。

**💡 创新点**

创新点包括：①首个针对3DGS场景的流媒体框架；②构建并公开了专门用于3DGS视口轨迹的用户数据集；③设计了协同视口预测模块，利用多用户协作嵌入与历史轨迹双重信息提升预测精度；④将DDPG与点云特征抽取网络（Set抽取与Feature Propagation）结合，解决可变状态/动作空间下的码率适配问题。

**🔧 技术方法**

核心技术包括3D Gaussian Splatting（Scaffold‑GS）场景表示、双层注意力+跨注意力的协同视口预测网络、iTransformer序列建模、DDPG强化学习框架、点云Set抽取（SA/FP）以及多层感知机（MLP）融合。

**📊 数据集**

使用了32名受试者在30秒/分钟内对15个室内外3DGS场景进行视口轨迹采集（共约864k帧，30fps），构成专门的3DGS视口数据集。

**📈 对比分析**

与ViVo、CaV3（体积视频流方法）和GS3D（默认距离优先的3DGS流）在40/80/120 Mbps网络条件下进行对比。GSStream在视口SSIM上平均提升约10–12%，累计传输数据量更小，时间序列视觉质量更平稳，带宽利用率最高。

**⚠️ 局限性**

局限性包括：仅针对静态3DGS场景，未覆盖动态体积视频；未使用专用3DGS编解码器，仍采用简单下采样；码率适配仅针对单用户，尚未研究多用户/多客户端场景。

---

## 71. ShapeMark: Robust and Diversity-Preserving Watermarking for Diffusion Models

**arXiv ID:** 2603.09454 | [PDF](https://arxiv.org/pdf/2603.09454v1)

**作者:** Yuqi Qian `[一作]` (Institute of Information Engineering), Meineng Zhu `[通讯]` (School of Cybersecurity)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在扩散模型（Stable Diffusion）中设计了一种基于结构编码与随机化的水印方法ShapeMark，实现了对生成图像的鲁棒且多样化的水印嵌入与提取。

**💡 创新点**

创新点包括：①将水印信息编码到噪声的结构关系（分量块的相对排列）而非单值；②通过分离量化区间构建可区分的块和组；③引入负载去偏随机化PDSR，消除固定空间模式，提升多样性；④在保持高容量（256位/图像）的同时，兼顾鲁棒性和视觉质量。

**🔧 技术方法**

主要技术手段：结构编码（SE）– 量化排序→块与组→块排列码本；负载去偏随机化（PDSR）– 块级全局置换；扩散逆向（DDIM）恢复噪声；解码时的代码本匹配与阈值检测。

**📊 数据集**

使用Stable Diffusion v2.1作为生成模型，训练与评估数据集为MS‑COCO 2017和Stable‑Diffusion‑Prompts（SDP）两套数据，实验中每个prompt生成多张带水印图像并施加九种失真。

**📈 对比分析**

与后处理水印、模型微调水印以及其他NaW方法（Tree‑Ring、Gaussian Shading、PRC‑Watermark、T2SMark）进行对比；在TPR@FPR=10⁻⁶上达到1.000、每比特恢复准确率0.9870；LPIPS多样性最高（0.7338）；视觉质量（CLIP、FID）保持与无水印相近；整体表现优于所有基线。

**⚠️ 局限性**

局限性：在极高容量（>2048位）或强引导/逆向步数较少时，鲁棒性会下降；仅在Stable Diffusion上验证，其他扩散模型的迁移性未评估；未针对主动对抗性攻击（如水印移除或伪造）进行深入实验。

---

## 72. Computing $L_\infty$ Hausdorff Distances Under Translations: The Interplay of Dimensionality, Symmetry and Discreteness

**arXiv ID:** 2603.08890 | [PDF](https://arxiv.org/pdf/2603.08890v1)

**作者:** Sebastian Angrick `[一作]` (Karlsruhe Institute of Technology), Marvin Künnemann `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 564 | [OpenAlex ID](https://openalex.org/A5088790846)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对点集 Hausdorff 距离在不同翻译、维度、方向性以及离散/连续版本下的细粒度复杂性进行系统分析，并给出新的上界与下界。

**💡 创新点**

揭示了维度、对称性与离散性的相互作用，发现对称性不一定导致更易求解，提出了针对极度不平衡输入的近线性算法，并在 1 维中给出了与 MaxConv LowerBound 的条件分离。

**🔧 技术方法**

采用细粒度归约、超图超团假设、3-SUM 与 All-Ints-3SUM 的归约、补集分解、正交向量假设、以及对称/无对称转换的构造等技术。

**📊 数据集**

无实验数据，全部以理论证明和假设为基础；使用的实例均为合成的点集/超图。

**📈 对比分析**

在 d≥3、m≪n 时实现近线性时间（m^{1+o(1)}），在 d=3、m≈√n 时给出匹配的 (nm)^{3/2−ε} 下界；在 d=1、离散/连续版本与 MaxConv LowerBound 关联，说明不存在子平方解；总体上与已知上界相匹配或仅存在小幅度差距。

**⚠️ 局限性**

仍缺乏平衡情况（m≈n）下的非组合下界、d≥4 的完全匹配下界、以及对更高维度的归约技术；结果高度依赖于组合与非组合算法假设，未能解决所有参数空间的完整时间复杂度。

---

## 73. MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating

**arXiv ID:** 2603.09419 | [PDF](https://arxiv.org/pdf/2603.09419v1)

**作者:** Yuning Wang `[一作]` (Xi'an Jiaotong University), Jianru Xue `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 5667 | [OpenAlex ID](https://openalex.org/A5024309592)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出MetaDAT框架，实现轨迹预测的Meta预训练与自适应测试时更新。

**💡 创新点**

对预训练与测试时训练进行离线-在线对齐的Meta预训练，和基于在线梯度的动态学习率+难样本驱动更新。

**🔧 技术方法**

Meta学习（双层优化）、动态学习率优化、难样本驱动更新、MAE损失、ForecastMAE网络、AdamW等。

**📊 数据集**

Waymo、nuScenes、Lyft三大轨迹数据集。

**📈 对比分析**

与DURA、TENT、MEK、AML、T4P等TTS方法对比，mADE/mFDE均显著提升，超越T4P约12%并在多模态指标上表现最好。

**⚠️ 局限性**

依赖精确的在线感知轨迹，噪声感知误差会影响适应效果。

---

## 74. A Hybrid Residue Floating Numerical Architecture with Formal Error Bounds for High Throughput FPGA Computation

**arXiv ID:** 2603.08712 | [PDF](https://arxiv.org/pdf/2603.08712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 75. GIAT: A Geologically-Informed Attention Transformer for Lithology Identification

**arXiv ID:** 2603.09165 | [PDF](https://arxiv.org/pdf/2603.09165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 76. PathMem: Toward Cognition-Aligned Memory Transformation for Pathology MLLMs

**arXiv ID:** 2603.09943 | [PDF](https://arxiv.org/pdf/2603.09943v1)

**作者:** Jinyue Li `[一作]` (University of Science and Technology of China), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 49523 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于记忆中心的多模态模型PathMem，用以提升计算病理MLLM的诊断推理和报告生成。

**💡 创新点**

创新点在于将结构化病理知识图构建为长期记忆，并通过双阶段Memory Transformer实现长期记忆向工作记忆的动态转换，结合静态与动态检索以可解释且可控的方式注入专业知识。

**🔧 技术方法**

采用PubMed深度检索与LLM抽取构建知识图，使用知识图嵌入与Transformer结合的Memory Transformer，采用双阶段训练（对齐、投影、指令微调）和Top‑K自适应选择。

**📊 数据集**

主要使用WSI-Bench作为内部基准，并在WSI-VQA、SlideBench‑VQA (BCNB) 与 CPTAC‑NSCLC 等三大公开WSI级别外部数据集进行零样本验证。

**📈 对比分析**

与WSS-LLaVA、Quilt-LLaVA、WSI-VQA、GPT‑4o 等现有模型对比，PathMem在WSI-Bench的报告生成、诊断、形态分析等任务均实现SOTA，WSI-Precision提升12.8%，WSI-Relevance提升10.1%，在零样本外部基准上平均提升约1.5个百分点。

**⚠️ 局限性**

局限性包括知识图覆盖范围受限、检索与推理对计算资源要求高、模型在极大WSI尺寸下的效率待进一步优化，以及尚缺乏临床真实环境的验证。

---

## 77. From Ideal to Real: Stable Video Object Removal under Imperfect Conditions

**arXiv ID:** 2603.09283 | [PDF](https://arxiv.org/pdf/2603.09283v1)

**作者:** Jiagao Hu `[一作]` (Xiaomi Inc), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种稳健的视频对象移除框架SVOR，能够在阴影、抖动、遮罩缺陷等不完美条件下实现无阴影、无闪烁、无遮罩缺陷的移除。

**💡 创新点**

创新点包括三项：MUSE掩码联合策略用于处理突变运动下的时间压缩导致的漏删；DA‑Seg轻量级去噪感知分割头为内部定位提供先验；两阶段课程式训练先用无配对背景视频自监督学习背景再用合成配对数据结合遮罩退化和侧面加权损失细化移除与阴影抑制。

**🔧 技术方法**

采用扩散模型DiT为骨干，配合侧分支的DA‑Seg与DA‑AdaLN；实现遮罩联合MUSE；通过自监督预训练、遮罩退化、加权扩散损失等技术构建两阶段训练框架。

**📊 数据集**

训练使用大规模背景视频（约49K条）进行自监督预训练，随后使用ROSE合成配对数据；评估在DAVIS（90条无配对）、ROSE Bench（60条合成）以及新构建的RORD‑50（50条实景配对）三大数据集。

**📈 对比分析**

与多种非扩散和扩散基SOTA方法（FuseFormer、Propainter、DiffuEraser、VACE、ROSE、minimax等）对比，在ReMOVE、GPT、PSNR、SSIM等指标均取得领先，特别是在遮罩退化、突变运动和SAM2缺陷场景下表现稳健，用户研究中也获得最高的移除与完成分数。

**⚠️ 局限性**

局限性主要在极度稀疏遮罩（仅单帧）时可能产生误删/漏删，以及仍难以完全消除所有侧效应如阴影、反射。

---

## 78. CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation

**arXiv ID:** 2603.09418 | [PDF](https://arxiv.org/pdf/2603.09418v1)

**作者:** Bohao Li `[一作]` (Northwestern Polytechnical University), Yangming Guo `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 1083 | [OpenAlex ID](https://openalex.org/A5109484676)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CIGPose框架，通过因果干预去除视觉背景的混杂影响，从而实现更稳健的全身姿态估计。

**💡 创新点**

核心创新在于：①将预测不确定性作为混杂判别指标；②用可学习的上下文无关典型嵌入代替混杂的关键点嵌入；③在去混杂嵌入上构建层级图神经网络，实现局部与全局解剖一致性。

**🔧 技术方法**

主要技术包括：结构因果模型（SCM）+ 因果干预模块（CIM）、可学习的典型嵌入表、基于EdgeConv和超图的层级GNN、预测不确定性度量与干预选择、对比一致性损失。

**📊 数据集**

使用了COCO-WholeBody、COCO、UBody以及CrowdPose四大公开数据集进行训练与评测。

**📈 对比分析**

与现有SOTA方法相比，CIGPose在COCO-WholeBody上单数据集训练即可获得67.0% AP，加入UBody提升至67.5% AP，均高于前沿方法；在COCO和CrowdPose上也均实现了1–2% AP的提升。

**⚠️ 局限性**

主要局限包括：①干预策略依赖不确定性作为混杂的代理，可能无法覆盖所有类型的背景干扰；②典型嵌入表的学习与初始化可能对模型收敛敏感；③在极端遮挡或完全陌生背景下的泛化仍有待进一步验证。

---

## 79. MORE-R1: Guiding LVLM for Multimodal Object-Entity Relation Extraction via Stepwise Reasoning with Reinforcement Learning

**arXiv ID:** 2603.09478 | [PDF](https://arxiv.org/pdf/2603.09478v1)

**作者:** Xiang Yuan `[一作]` (Peking University), Tong Mo `[通讯]` (Peking University)

**通讯引用:** 357 | [OpenAlex ID](https://openalex.org/A5059356240)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了基于大型视觉语言模型的两阶段训练框架MORE-R1，利用显式逐步推理与强化学习提升多模态对象-实体关系抽取性能。

**💡 创新点**

创新点在于自动构造细粒度推理示例进行冷启动SFT，采用GRPO强化学习与进阶样本混合策略来优化对复杂案例的推理，并实现可解释、可扩展的生成式方法。

**🔧 技术方法**

使用了Qwen2.5‑VL作为LVLM骨干，GPT‑4o生成推理示例，SFT+GRPO强化学习，Progressive Sample‑Mixing策略，以及CoT式六步推理模板与规则奖励。

**📊 数据集**

使用了MORE基准数据集（20,264个样本，21种关系类型）。

**📈 对比分析**

与多种SOTA分类基线（如IFAformer、REMOTE）及生成式基线（Qwen2.5‑VL‑SFT）对比，MORE‑R1在Accuracy、Precision、Recall和F1上均有显著提升，尤其F1提升6.1%。

**⚠️ 局限性**

局限性包括对外部大模型生成推理数据的依赖（成本高），RL训练计算开销大，以及在极少样本或新关系类型上的泛化能力尚待进一步验证。

---

## 80. TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control

**arXiv ID:** 2603.09332 | [PDF](https://arxiv.org/pdf/2603.09332v1)

**作者:** Shihao He `[一作]` (Shenzhen University), Shengli Zhang `[通讯]` (Shenzhen University)

**通讯引用:** 23242 | [OpenAlex ID](https://openalex.org/A5100413426)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种检索驱动的可编辑音频效果控制框架，并在吉他效果预设上进行实验；

**💡 创新点**

创新点在于引入Texture Resonance Retrieval（TRR），利用深度特征的Gram矩阵捕获音频纹理，从而提升检索的纹理敏感性；

**🔧 技术方法**

核心技术包括Wav2Vec2中层激活的随机线性投影、Gram矩阵计算、余弦相似度检索，以及与CLAP、Wav2Vec-RAG、Text-RAG、FeatureNN-RAG等基线的对比；

**📊 数据集**

实验使用的自建吉他效果预设数据集，包含1063个预设、204个查询，确保了检索结果不出现训练测试泄漏；

**📈 对比分析**

在Protocol‑A检索基准上，TRR在L2误差、Norm.L2、Acc@0.1、Recall、Cosine和模块一致性等六项指标均优于所有基线，平均L2误差降幅约为15.8，Cosine提升约0.30；

**⚠️ 局限性**

局限性包括仅针对吉他效果、缺乏真实音频鲁棒性评估、未测试更大规模或跨乐器的泛化、基线选择不够全面、以及与参数感知间可能存在的偏差。

---

## 81. Robust Regularized Policy Iteration under Transition Uncertainty

**arXiv ID:** 2603.09344 | [PDF](https://arxiv.org/pdf/2603.09344v1)

**作者:** Hongqiang Lin `[一作]` (Zhejiang University), Dongxu Zhang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 1749 | [OpenAlex ID](https://openalex.org/A5100640205)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Robust Regularized Policy Iteration（RRPI）算法，在离线强化学习中将转移核视为决策变量，通过鲁棒优化与 KL 正则化来解决最大-最小双层问题，并实现可迭代的策略优化。

**💡 创新点**

创新点包括：①构造鲁棒正则化 Bellman 运算子；②使用 KL 正则化代理目标，将不可求解的 max‑min 目标转化为可求解的迭代式；③证明了该运算子为 γ‑收缩映射，并给出了策略迭代的单调提升与收敛性。

**🔧 技术方法**

技术手段：基于模型的离线 RL，利用模型集成构造不确定集；KL 正则化软贪心策略更新；鲁棒 Bellman 迭代；最小化转移不确定性；蒙特卡洛估计与重要性采样；Q 值截断以防过估。

**📊 数据集**

使用 D4RL 基准环境，包括 HalfCheetah、Hopper、Walker2d 等多种 Random、Medium、Expert、Medium-Expert、Full-Replay 数据集。

**📈 对比分析**

与 CQL、DMG、EPQ、MOReL、RAMBO、PMDB、ADM 等强基线对比，RRPI 在 18 个 D4RL 环境中 11/18 环境表现最佳，整体平均性能优于现有方法，并在大多数环境表现更稳健。

**⚠️ 局限性**

局限性：对模型集成构造不确定集的依赖可能导致误差放大；理论与实证之间仍存在差距；缺乏对多模态观测（如视觉）的直接支持；在高维连续动作空间下计算成本较高。

---

## 82. How to Write to SSDs

**arXiv ID:** 2603.09927 | [PDF](https://arxiv.org/pdf/2603.09927v1)

**作者:** Bohyun Lee `[一作]` (Technische Universität München), Viktor Leis `[通讯]` (Technische Universität München)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

将传统的基于 B‑Tree 的 LeanStore DBMS 从原来的原地写（in‑place）改造为离线写（out‑of‑place）写，并在此基础上提出一组跨层优化技术，以降低数据库层和 SSD 层的写放大（write amplification）

**💡 创新点**

① 统一考虑数据库层和 SSD 层的写放大，提出总写放大（Total WAF）作为整体性能衡量指标；② 通过页面压缩与页面打包、死亡时间（deathtime）聚合（GDT）以及与 SSD 交互的 NoWA、FDP 等机制，实现了在多种 SSD 接口（CNS、ZNS、FDP）下均可将 SSD 写放大降至 1；③ 在数据库层实现了可调节的 GC 单元和对 SSD 物理写放大可预测的区块分配

**🔧 技术方法**

离线写模式、页面压缩 + 页面打包、死亡时间聚合（GDT）、NoWA 写模式、FDP 放置信息、与 SSD 的 ZNS、FDP 交互、改进的 GC 选择策略、跨层写放大评估（通过 OCP）

**📊 数据集**

YCSB‑A（zipf θ = 0.8，50% 读/50% 写，100‑800 GB），TPC‑C（15 000 仓库，约 1.6 TB），多种企业 SSD（Samsung PM9A3、Solidigm D7‑P5520、Kioxia CM7‑R 等），以及 ZNS 与 FDP 设备

**📈 对比分析**

在所有实验中，基于 out‑of‑place 写并开启全部优化后，总写放大从 4–5 降至 0.6 左右；写吞吐率提升 1.6–2.5×；单个事务的 flash 写量下降 6–10×；在 ZNS 上进一步提升 2×吞吐率；在 FDP 上通过 NoWA 替代可达 1 的写放大；与传统 in‑place LeanStore、MySQL、PostgreSQL 等系统对比，性能与 SSD 寿命都有显著提升

**⚠️ 局限性**

（1）实现需要额外的元数据管理，内存占用最高可达 10 GB；（2）对压缩与打包的 CPU 开销虽可忽略，但仍略高于原生写；（3）对 SSD 物理细节的推测（如 GC 单元大小）需要实验或预估，可能不适用于所有硬件；（4）在高并发、混合工作负载下的 GC 与 NoWA 的调度仍需进一步优化；（5）实验主要针对单实例单设备环境，未覆盖多实例或虚拟化共享 SSD 的场景

---

## 83. ZipPIR: High-throughput Single-server PIR without Client-side Storage

**arXiv ID:** 2603.09190 | [PDF](https://arxiv.org/pdf/2603.09190v1)

**作者:** Rasoul Akhavan Mahdavi `[一作]` (University of Waterloo), Florian Kerschbaum `[通讯]` (University of Waterloo)

**通讯引用:** 4543 | [OpenAlex ID](https://openalex.org/A5102985450)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出ZipPIR，一种单服务器PIR协议，压缩LWE密文到Paillier密文以实现高吞吐量且无客户端存储。

**💡 创新点**

结合离线压缩技术和加性加密，实现接近SimplePIR的吞吐量，同时客户端仅需常量存储；首次展示Paillier在PIR中可与LWE同等高效。

**🔧 技术方法**

基于LWE/RLWE的线性相位压缩、Paillier加密的离线/在线分离、RNS矩阵乘法、批量压缩与扩展压缩键。

**📊 数据集**

在多种数据库规模（1 GB、2 GB等）上评估，使用公开LWE参数（如 n=1400, q=2^32）及模拟数据库。

**📈 对比分析**

与SealPIR、FastPIR、OnionPIR、Spiral、HintlessPIR、YPIR等协议对比，ZipPIR实现约3 GB/s吞吐量，10 倍以上快，且服务器存储仅≈200 KB/客户端。

**⚠️ 局限性**

压缩仍依赖昂贵的Paillier指数运算，批量压缩速度低于RLWE打包；仅支持单字节或大负载查询，且对数据库更新需重新生成压缩键。

---

## 84. Spectral-Structured Diffusion for Single-Image Rain Removal

**arXiv ID:** 2603.09054 | [PDF](https://arxiv.org/pdf/2603.09054v1)

**作者:** Yucheng Xing `[一作]` (Stony Brook University), Xin Wang `[通讯]` (Stony Brook University)

**通讯引用:** 29314 | [OpenAlex ID](https://openalex.org/A5100327957)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SpectralDiff，基于频域结构化扩散的单幅雨渍去除方法，并设计全乘 U‑Net 提升运算效率

**💡 创新点**

① 将雨渍的层级频谱特性嵌入扩散扰动；② 在频域构造方向和尺度感知的掩模；③ 在空间域采用全乘层替代卷积实现频域滤波

**🔧 技术方法**

频域结构化扩散（Spectral Diffusion）、全乘 U‑Net（Full‑Product U‑Net）、DDIM 采样策略、FFT/Inverse‑FFT、卷积定理

**📊 数据集**

Rain1400、RainCityscapes、SPA‑Data（含合成与真实雨图）

**📈 对比分析**

与多种基线（优化型、CNN、GAN、扩散）在 PSNR/SSIM 上对比；SpectralDiff 在合成数据上与最优模型相当，在真实 SPA‑Data 上优于其他方法；推理时间约 10 步，显著低于 100 步扩散模型，参数和 FLOPs 亦大幅减少

**⚠️ 局限性**

对真实雨分布的鲁棒性已显著提升，但在极端雨密度或多尺度交错时仍可能出现残留；目前掩模固定，缺乏自适应机制

---

## 85. Mobile Base Station Optimal Tour in Wide Area IoT Sensor Networks

**arXiv ID:** 2603.08828 | [PDF](https://arxiv.org/pdf/2603.08828v1)

**作者:** Sachin Kadam `[一作]` (Motilal Nehru National Institute of Technology), Sachin Kadam `[通讯]` (Motilal Nehru National Institute of Technology)

**通讯引用:** 90 | [OpenAlex ID](https://openalex.org/A5101998226)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并求解移动基站（MBS）在广域物联网传感器网络中，利用UAV完成最小成本、非重复路径覆盖全部传感器并避开禁飞区的优化问题。

**💡 创新点**

提出了“Mobile Base Station Optimal Tour（MOT）”组合优化模型，将停靠点选择、路径规划、覆盖保证与能量约束统一在一个NP‑完整框架内，并给出低复杂度贪心算法实现。

**🔧 技术方法**

采用组合优化建模、离散路径规划、覆盖收益评估、能量消耗计算，并实现多项式时间的贪心启发式算法，使用欧氏距离作为旅行成本。

**📊 数据集**

使用仿真数据：100个传感器随机布置于100×100 m²区域，30个候选停靠点，1个20×20 m²禁飞区；算法在MATLAB 2025b上运行。

**📈 对比分析**

与四个现有算法（Baek 2019/2020，Li 2021，Zhu 2023）进行比较，指标为巡航长度×运行时间，结果显示本算法在长度约178 m、耗时0.12 s时，性能指标α下降39.15%，实现了更快更省成本的方案。

**⚠️ 局限性**

局限性包括仅考虑单UAV、静态传感器布置、确定性能量模型，以及贪心算法缺乏理论近似保底；未来工作计划引入多UAV、随机能量模型及可证明的近似算法。

---

## 86. OptEMA: Adaptive Exponential Moving Average for Stochastic Optimization with Zero-Noise Optimality

**arXiv ID:** 2603.09923 | [PDF](https://arxiv.org/pdf/2603.09923v1)

**作者:** Ganzhao Yuan `[一作]` `[通讯]` (Shenzhen University of Advanced Technology), Ganzhao Yuan (Shenzhen University of Advanced Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 OptEMA 两个闭环自适应 EMA 优化器，用于非凸随机优化，消除了传统 Adam 中的开环步长与梯度上界等限制。

**💡 创新点**

创新点在于将 EMA 权重和步长设计为轨迹自适应闭环控制，使算法无 Lipschitz 常数依赖、零噪声下可达近似最优收敛率，并实现噪声自适应性能。

**🔧 技术方法**

采用轨迹依赖的 EMA 权重、闭环步长自适应、平均光滑性假设、无偏梯度与方差上界、Lyapunov 分析和对数界证明噪声自适应收敛。

**📊 数据集**

本文未在具体数据集上进行实验验证，主要提供理论分析与证明；若有实验，则未在论文中给出具体数据集。

**📈 对比分析**

与 Adam、STORM、AdaGrad-Norm 等方法比较，OptEMA 在噪声自适应率 𝒪(T⁻¹ᐟ²+σ¹ᐟ²T⁻¹ᐟ⁴) 与现有 Adam 变体相当，在 σ=0 时达到 𝒪(T⁻¹ᐟ²) 的近似最优收敛；实验（如有）显示收敛更快、对超参数更稳健。

**⚠️ 局限性**

理论结果包含多项对数因子，导致实际迭代复杂度略高；假设仍基于平均光滑性，未考虑批量大小、分布式实现等实际情况；缺乏公开实验验证。

---

## 87. N-gram-like Language Models Predict Reading Time Best

**arXiv ID:** 2603.09872 | [PDF](https://arxiv.org/pdf/2603.09872v1)

**作者:** James A. Michaelov `[一作]` (Massachusetts Institute of Technology), Roger P. Levy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22719 | [OpenAlex ID](https://openalex.org/A5090215557)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨语言模型在预测阅读时间时的逆向可扩展性，认为阅读时间主要受低阶n‑gram统计影响。

**💡 创新点**

提出阅读时间对n‑gram概率敏感而非高级transformer概率，解释逆向扩展现象。

**🔧 技术方法**

使用transformer语言模型（Pythia系列）、n‑gram统计（infini‑gram、Stupid Backoff）以及眼动追踪读时测量。

**📊 数据集**

使用Provo眼动阅读语料、GECO阅读语料以及多种大型语料库（OpenWebText、C4、Pile、Dolma、DCLM、OLMo‑Mix）进行n‑gram估计。

**📈 对比分析**

对比模型惊讶度与n‑gram惊讶度及阅读时间相关性，发现模型与n‑gram相关性最高时与阅读时间的相关性也最高，性能与传统n‑gram相当甚至略优。

**⚠️ 局限性**

局限在于只关注眼动读时指标，未验证对其他语言加工指标（如N400）的解释，且未探究语义和世界知识对读时的影响。

---

## 88. When to Lock Attention: Training-Free KV Control in Video Diffusion

**arXiv ID:** 2603.09657 | [PDF](https://arxiv.org/pdf/2603.09657v1)

**作者:** Tianyi Zeng `[一作]` (Shanghai Jiao Tong University), Xueqian Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5327 | [OpenAlex ID](https://openalex.org/A5100737125)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出KV-Lock框架，通过动态锁定注意力来提升视频编辑前景质量并保持背景一致性。

**💡 创新点**

创新点在于利用扩散模型的幻觉检测（方差）作为调度信号，实现KV缓存与条件引导的自适应控制，解决传统KV锁定过度或欠锁导致的质量损失。

**🔧 技术方法**

使用KV缓存、局部方差幻觉检测、动态KV融合比例、可学习的CFG缩放因子以及滑动窗口统计等技术。

**📊 数据集**

在VACE-Benchmark（22个场景）和自采集的30段视频（共52样本）上进行评估，视频分辨率为480×832。

**📈 对比分析**

与FateZero、FLATTEN、TokenFlow、CFG-Zero*、APG、ProEdit及VACE等基线对比，KV-Lock在SC、BC、AQ等指标上均表现最佳，且在背景SSIM/PSNR上优于VACE。

**⚠️ 局限性**

局限性包括推理时间较慢（需KV缓存和滑动窗口计算）、依赖显式掩码分割以及对KV缓存的显存需求较高。

---

## 89. Social-R1: Towards Human-like Social Reasoning in LLMs

**arXiv ID:** 2603.09249 | [PDF](https://arxiv.org/pdf/2603.09249v1)

**作者:** Jincenzi Wu `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9528 | [OpenAlex ID](https://openalex.org/A5019458385)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向社会推理的硬性对抗性评测基准 ToMBench‑Hard，并基于多维奖励的强化学习框架 Social‑R1，训练小参数 LLM 在社会推理任务中获得与大模型同等甚至更优的表现。

**💡 创新点**

创新点：① 将“思维轨迹”与最终答案并列作为奖励目标，采用结构化、内容完整和信息密度三维奖励；② 设计 ToMBench‑Hard 通过专家构造的对抗性扰动揭示模型的“短路”行为；③ 通过过程级监督而非仅靠结果提升社会推理的内部化与可迁移性。

**🔧 技术方法**

技术：多维奖励的强化学习（包含 R_struct、R_content、R_len 等），SIP（社会信息处理）理论驱动的阶段化推理；使用 GPT‑4o 作为奖励评估者；Group Relative Policy Optimization (GRPO) 进行策略更新；对话式奖励合成与学习曲线调度。

**📊 数据集**

数据集：ToMBench‑Hard（800 条专家标注的多选题，含对抗性扰动），以及八个对社会推理的标准评测集（ToMBench、SocialIQA、EmoBench、MotiveBench、SimpleToM、Hi‑ToM、TactfulToM 等）。

**📈 对比分析**

对比方法：在 Qwen3‑4B/8B 上进行强化学习训练，并与基准 LLM、传统 RLHF、以及 70B‑plus 大模型进行对照；结果显示 4B Social‑R1 在所有 8 个基准上均优于 LLaMA3.1‑70B，8B 版本甚至击败 DeepSeek‑R1 与 Qwen3‑32B，证明过程级奖励显著提升了社会推理能力。

**⚠️ 局限性**

局限性：仅验证于多选题式社会推理任务，缺少开放式对话或跨模态社会任务；对抗性基准虽强大但规模有限（800 条）；强化学习训练成本高，需要大规模 GPU；模型对极端扰动和长文本的鲁棒性尚待进一步探索。

---

## 90. From Perception to Cognition: How Latency Affects Interaction Fluency and Social Presence in VR Conferencing

**arXiv ID:** 2603.09261 | [PDF](https://arxiv.org/pdf/2603.09261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 91. AI-Enabled Data-driven Intelligence for Spectrum Demand Estimation

**arXiv ID:** 2603.09916 | [PDF](https://arxiv.org/pdf/2603.09916v1)

**作者:** Colin Brown `[一作]` (Communications Research Centre), Halim Yanikomeroglu `[通讯]` (Carleton University)

**通讯引用:** 21176 | [OpenAlex ID](https://openalex.org/A5035446029)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于 AI 的数据驱动光谱需求预测框架，结合站点许可证和众包活跃用户数据构建光谱需求代理并进行验证与建模。

**💡 创新点**

创新点在于：①首次将众包活跃用户数据与自报站点带宽相结合形成综合代理；②通过空间聚类和空间滞后特征有效抑制空间自相关；③在五个大城市中验证模型的通用性。

**🔧 技术方法**

采用机器学习模型（线性回归与 XGBoost）进行回归预测，并使用 OLS 及 F 统计量对代理进行验证。

**📊 数据集**

使用加拿大五大城市（蒙特利尔、渥太华、多伦多、卡尔加里、温哥华）的站点许可证数据、众包活跃用户数据、移动网络流量数据，以及人口、经济、物理与活动特征等多源地理信息。

**📈 对比分析**

通过 R²、RMSE、MAE 等指标与基线模型（昼间人口）进行比较，综合代理在 XGBoost 中取得最高 R²=0.89、最低 Norm. RMSE=0.022、Norm. MAE=0.014，显著优于单一代理和基线。

**⚠️ 局限性**

局限性包括：①代理依赖自报带宽和众包采样，可能存在报告误差与采样偏差；②模型在高波动或极端地区的泛化仍待进一步验证；③未考虑未来 6G 时代新业务对需求的影响。

---

## 92. Caterpillar-Inspired Spring-Based Compressive Continuum Robot for Bristle-based Exploration

**arXiv ID:** 2603.09745 | [PDF](https://arxiv.org/pdf/2603.09745v1)

**作者:** Zhixian Hu `[一作]` (Purdue University), Juan Wachs `[通讯]` (Purdue University)

**通讯引用:** 4349 | [OpenAlex ID](https://openalex.org/A5072528523)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种基于弹簧的牵引驱动连续体机器人，可与工业机械臂兼容，用于狭窄空间检查并集成人工毛刷实现接触感知。

**💡 创新点**

创新点在于：①采用弹性弹簧背骨与四线牵引实现可压缩弯曲运动；②驱动模块紧凑，可直接装配为现有机器人末端执行器；③通过人工毛刷与压力传感器实现无侵入式表面感知与障碍检测。

**🔧 技术方法**

使用技术包括：弹簧驱动连续体机器人、常弧曲运动学模型、四个MG996R伺服+齿轮轮胎、LPS33HW压力传感器、UR16e机械臂、RealSense D455摄像头、t‑SNE、Ecoflex等。

**📊 数据集**

实验数据集：自制五种不同形状物体（食物盒、擦板、线圈、橙子等）以及管道内的立方障碍物；无公开公开数据集。

**📈 对比分析**

性能评估：在单纯压缩运动下平均误差2.06 mm；在耦合压缩+弯曲运动下平均误差4.32 mm；实验演示成功重建物体表面点云并在管道内检测障碍，显示在受限空间中具备实用性。

**⚠️ 局限性**

局限性：未对载荷/最大推力进行评估；运动学模型简化导致误差，尤其在中间区域；弹簧压缩受限；未实现闭环姿态反馈；扫描速度相对慢。

---

## 93. A Unified Hierarchical Multi-Task Multi-Fidelity Framework for Data-Efficient Surrogate Modeling in Manufacturing

**arXiv ID:** 2603.09842 | [PDF](https://arxiv.org/pdf/2603.09842v1)

**作者:** Manan Mehta `[一作]` (University of Illinois), Chenhui Shao `[通讯]` (University of Michigan)

**通讯引用:** 2155 | [OpenAlex ID](https://openalex.org/A5059084183)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个层级多任务多保真度高斯过程框架（H‑MT‑MF），用于在制造系统中高效构建代理模型，并通过1D合成例子与发动机表面形状预测案例验证其性能。

**💡 创新点**

创新点在于将多任务学习与多保真度信息统一到层级贝叶斯模型中，通过分离任务特定全局趋势与共享的局部残差，实现在异构数据下的更高预测精度。

**🔧 技术方法**

使用了高斯过程的异质性随机克里金（SK）、多任务GP、层级贝叶斯先验和EM算法进行参数估计，并在残差层共享任务相关性。

**📊 数据集**

使用了①1D三任务合成数据，包含低/高保真测量；②真实发动机块表面高度数据，三块相似但不同的表面，并采集高/低分辨率测量点。

**📈 对比分析**

与单任务随机克里金（SK）和不考虑保真度的多任务学习（EG‑MTL）对比，H‑MT‑MF在RMSE上平均提升约19%/23%，且在不同保真度组合下保持稳定低误差，优于两者。

**⚠️ 局限性**

目前仅适用于空间静态过程，采样设计未自适应，且模型假设任务相似且保真度通过噪声方差体现，复杂时空动态或非线性保真度关系仍需进一步扩展。

---

## 94. PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories A Unified Framework for Test-Time Adaptation with Compositional Rule Learning and Pareto-Guided Prompt Evolution

**arXiv ID:** 2603.09641 | [PDF](https://arxiv.org/pdf/2603.09641v1)

**作者:** Arash Shahmansoori `[一作]` `[通讯]`, Arash Shahmansoori

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 PRECEPT 框架，旨在让 LLM 代理在有限样本下能够可靠地学习、组合、检测并更新规则，实现实时自适应决策。

**💡 创新点**

创新点包括：1) O(1) 结构化键的确定性精确检索，消除解释误差；2) 结合贝叶斯源可靠性与阈值失效的冲突感知记忆，能够主动识别并覆盖静态/动态矛盾及环境漂移；3) COMPASS 外层循环通过 Pareto 与 MAP‑Elites 优化提示，保证检索-决策链的一致性。

**🔧 技术方法**

技术手段涵盖：结构化键哈希检索、语义‑层级层次化堆叠、六法集成冲突检测与贝叶斯后验、阈值失效机制、COMPASS 的高频监控与低频提示演化（GEPA、Pareto、MAP‑Elites）。

**📊 数据集**

实验使用了三类自定义“黑暗迷宫”域：物流（4 条件、4 选项）、整合（6 条件、15 选项）和预订（17 条件、20 选项），每个域均在 10 条随机种子下评估，训练样本 β∈{1…5}。

**📈 对比分析**

与增强版 Full Reflexion 与 ExpeL 对比，PRECEPT 在 1‑尝试成功率提升 41.1pp、组合泛化提升 33.3pp、2‑路物流 100% 成功率、持续学习提升 40–55pp、漂移恢复 55.0pp、步骤数减少 61%，且所有关键比较均达到 p<0.001，表现出显著优势。

**⚠️ 局限性**

局限性：依赖键的离散结构，难以直接迁移到连续或无结构的决策空间；对 LLM 的提示演化仍需昂贵的低频优化；在极大键空间或极端漂移情形下的失效阈值设定需手工调优。

---

## 95. Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation

**arXiv ID:** 2603.09527 | [PDF](https://arxiv.org/pdf/2603.09527v1)

**作者:** Luxi Lin `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 32131 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 Efficient Draft Adaptation（EDA）框架，用于在 Speculative Decoding 中高效地将 draft 模型适配到细调后的目标模型；通过共享-私有分解、数据再生成和样本选择实现参数与数据双重高效；

**💡 创新点**

创新点包括（1）共享‑私有 gated 结构，使只更新轻量级私有 expert 即可对目标特定分布进行适配；（2）利用细调目标模型自生成数据，将训练目标与推理阶段的目标对齐；（3）基于 PCA 与 Mahalanobis 距离的表示偏移样本筛选，显著提升适配效率；

**🔧 技术方法**

技术手段有：Speculative Decoding、参数高效微调（PEFT）与 LoRA 对比、共享‑私有 gated 模块、Softmax 路由、PCA 降维、Mahalanobis 评分、token‑级与样本‑级价值评估；

**📊 数据集**

使用的数据集涵盖数学（GSM8K、AIME2024、SVAMP、Hendrycks‑MATH、MathQA）、代码（HumanEval、APPS、BigCodeBench、HumanEval+、MBPP）和医学（MedMCQA、MedQA‑USMLE、PubMedQA、MedQA、MMLU），以及 ShareGPT 等通用预训练数据；

**📈 对比分析**

与 Training‑Free、Full‑FT、LoRA 等基线在 Greedy（T=0）与 Sampling（T=1）下进行比较；EDA 在所有任务上均显著提升平均接受长度（τ）和解码速度，参数量仅 27.5% 并且训练时间缩短至 39.2%，在 math 任务上 τ 从 4.22 提升至 4.79（+13.5%），速度提升至 3.06×；

**⚠️ 局限性**

局限性包括：需细调目标模型可用于自生成，极端域差异或数据稀缺时效果可能受限；共享‑私有分解假设共享分布占主导，若分布差异过大则需更多私有容量；未验证多目标或跨域并行适配的可行性；

---

## 96. TrainDeeploy: Hardware-Accelerated Parameter-Efficient Fine-Tuning of Small Transformer Models at the Extreme Edge

**arXiv ID:** 2603.09511 | [PDF](https://arxiv.org/pdf/2603.09511v1)

**作者:** Run Wang `[一作]` (ETH Zurich), Luca Benini `[通讯]` (ETH Zurich)

**通讯引用:** 56887 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 TrainDeeploy 框架，实现 Transformer（如 CCT）在极低功耗异构 SoC 上的端到端训练，并支持 LoRA 等参数高效微调技术。

**💡 创新点**

首次在极低功耗边缘设备上完成 Transformer 训练，结合 LoRA 降低梯度存储与参数量，并通过 FP32 GEMM 加速器实现硬件加速。

**🔧 技术方法**

基于 PyTorch → ONNX 的自动微分图构建，静态内存分配与操作符裁剪，张量调度与 2D 包装，集成 FP32 RedMulE GEMM 加速器与 LoRA 低秩分解。

**📊 数据集**

使用 CIFAR-10、MNIST、EuroSAT 三个数据集进行 50-shot 迁移学习实验。

**📈 对比分析**

与 PULP-TrainLib、POET、MiniLearn、TTE 等框架对比，TrainDeeploy 在 0.28M 参数 CCT 模型上实现 71–126M FLOP/step、4.6 FLOP/cycle；LoRA 训练速率最高可达 11 次梯度更新/秒；相较于全反向传播，LoRA 可将内存占用降低 23%、梯度参数减 15×、数据传输减少 1.6×，且显著提升吞吐量。

**⚠️ 局限性**

仅支持单样本单步更新，FP32 计算占用能耗较高；对更大模型或更复杂优化器的适配尚未验证，依赖特定硬件加速器，仍面临极低功耗平台上的内存瓶颈。

---

## 97. Reviving ConvNeXt for Efficient Convolutional Diffusion Models

**arXiv ID:** 2603.09408 | [PDF](https://arxiv.org/pdf/2603.09408v1)

**作者:** Taesung Kwon `[一作]` (KAIST), Vinicius Azevedo `[通讯]` (Independent researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种完全卷积的扩散模型FCDM，利用ConvNeXt结构并加入条件注入和U形设计，能够在ImageNet上进行高分辨率（256×512）条件图像生成。

**💡 创新点**

创新点在于将ConvNeXt迁移到扩散模型中，提供仅需两参数（块数L与隐藏通道C）即可扩缩的U形架构，显著降低FLOPs并提升收敛速度，同时实现与Transformer基础模型相当甚至更优的生成质量。

**🔧 技术方法**

使用的核心技术包括：卷积式Transformer块（ConvNeXt）、自适应层归一化（AdaLN）进行条件注入、全局响应归一化（GRN）提升通道多样性、梯度检查点、EMA、ADM的噪声调度与时间/类别嵌入。

**📊 数据集**

主要数据集为ImageNet-1K，在256×256和512×512两个分辨率上进行训练与评估。

**📈 对比分析**

与DiT、DiCo、DiC等基线在同等参数规模下对比，FCDM在FLOPs约为Transformer的50%、训练步骤约为7倍、并在256×512分辨率上实现FID 7.46/10.23（400k/1M步骤）以及通过指导得到的F1 2.03/3.23，显示出更优的效率-性能折中。

**⚠️ 局限性**

局限性包括：在最新最先进的模型（如EDM-2、Simpler Diffusion）上尚未超越；在高分辨率下仍受限于卷积算子本身的计算成本；缺乏大规模文本到图像的跨模态评估。

---

## 98. MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification

**arXiv ID:** 2603.09374 | [PDF](https://arxiv.org/pdf/2603.09374v1)

**作者:** Nikola Jovišić `[一作]` (Institute for AI R&D of Serbia), Dubravko Ćulibrk `[通讯]` (Faculty of Technical Sciences, University of Novi Sad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出MIL-PF框架，将冻结的基础模型预计算特征与轻量化多实例学习头相结合，完成乳腺X线影像分类；

**💡 创新点**

创新点在于利用预计算特征与注意力聚合的MIL结构，仅训练约40k参数即可替代端到端微调，显著降低训练成本并提升性能；

**🔧 技术方法**

技术包括冻结的DINOv2/MedSigLIP等视觉基础模型、Perceiver式交叉注意力局部聚合、全局与局部特征拼接、二分类交叉熵训练；

**📊 数据集**

使用了大规模公开乳腺影像数据集EMBED（约50万张）以及VinDr和RSNA进行评估；

**📈 对比分析**

与Shen、Pathak、Mourão等现有方法在相同数据划分下对比，MIL-PF在EMBED上AUC、Spec@Sens=0.9等指标均达到或超过SOTA，性能优异；

**⚠️ 局限性**

局限在于对小病灶的检测分辨率不足，依赖固定预训练模型，未充分利用患者双侧信息，且ROI定位仍有改进空间。

---

## 99. Performance Analysis of Edge and In-Sensor AI Processors: A Comparative Review

**arXiv ID:** 2603.08725 | [PDF](https://arxiv.org/pdf/2603.08725v1)

**作者:** Luigi Capogrosso `[一作]` (Interdisciplinary Transformation University of Austria), Michele Magno `[通讯]` (ETH Zurich)

**通讯引用:** 7784 | [OpenAlex ID](https://openalex.org/A5066423975)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种超低功耗边缘AI处理器（GAP9、STM32N6、Sony IMX500）进行统一硬件级评测，使用 PicoSAM2 轻量化分割模型作为工作负载。

**💡 创新点**

提出基于延迟、MAC/周期、MAC/J 与 EDP 四项指标的统一评估框架，揭示不同架构在能效与实时性之间的权衡，并证明 in‑sensor 计算在能耗与延迟上的优势。

**🔧 技术方法**

采用 RISC‑V 多核 MCU、ARM Cortex‑M55+Neural Accelerator、CMOS 堆叠 in‑sensor 计算单元、模型量化/剪枝、周期级功耗测量与数据流分析等技术。

**📊 数据集**

使用 336 M MAC 的 PicoSAM2 U‑Net 分割模型（输入 3×96×96 RGB）作为评测数据集。

**📈 对比分析**

通过四项硬件指标对三平台进行对比；IMX500 达到 86.2 MAC/周期、1359.6 MMAC/J、最低 3.4 mJ·s EDP；GAP9 具备最高能效 182.15 MMAC/J；STM32N6 延迟最短 13.7 ms，但能效仅 21.5 MAC/J，EDP 206.8 mJ·s。

**⚠️ 局限性**

仅评估单一分割模型，缺乏多种工作负载和完整系统能耗（如休眠/唤醒）的覆盖；实验环境未涵盖实际传感器输入噪声与温度变化，结果在更广泛应用场景下可能需要进一步验证。

---

## 100. No Image, No Problem: End-to-End Multi-Task Cardiac Analysis from Undersampled k-Space

**arXiv ID:** 2603.09945 | [PDF](https://arxiv.org/pdf/2603.09945v1)

**作者:** Yundi Zhang `[一作]` (Technical University of Munich), Jiazhen Pan `[通讯]` (Technical University of Munich)

**通讯引用:** 318 | [OpenAlex ID](https://openalex.org/A5004278800)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种k-MTR框架，直接从欠采样的k空间进行心脏多任务分析，无需重建图像。

**💡 创新点**

创新点是通过跨模态对齐将k空间与图像域嵌入共享语义流形，显式恢复缺失的解剖结构并避免传统的逆问题。

**🔧 技术方法**

使用自监督掩码自编码器、对比学习与轻量级任务解码器等技术构建频域表征。

**📊 数据集**

基于模拟的42,000例英国生物银行卡片MRI数据，生成欠采样k空间与相应标注。

**📈 对比分析**

与ResNet‑50、ViT、MAE等图像域基线比较，k‑MTR在表型回归、疾病分类与分割任务上性能与完全采样图像相当，且优于未对齐的MAE_k^u。

**⚠️ 局限性**

局限在于仅验证单线圈合成数据，尚未在多线圈真实临床采样模式下评估鲁棒性。

---

## 101. Deblurring structural edges in variable thickness topology optimization via density-gradient-informed projection

**arXiv ID:** 2603.09780 | [PDF](https://arxiv.org/pdf/2603.09780v1)

**作者:** Gabriel Stankiewicz `[一作]` (Friedrich-Alexander-Universitat Erlangen-Nurnberg), Paul Steinmann `[通讯]` (Friedrich-Alexander-Universitat Erlangen-Nurnberg)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在可变厚度拓扑优化（VTTO）中，本文提出了结合SIMPL惩罚与更新投影相结合的低厚度抑制方法，并提出了基于密度梯度信息的DGI投影来去除滤波导致的结构边缘模糊；

**💡 创新点**

创新点在于：1）将SIMPL惩罚与投影同步进行，稳健抑制低厚度区域；2）利用局部密度梯度动态调节投影锐度，实现仅在边缘区域去模糊，保持内部结构不受影响；

**🔧 技术方法**

采用密度基VTTO框架、PDE滤波器、SIMP惩罚、更新投影、DGI投影以及自适应网格细化；

**📊 数据集**

使用标准悬臂梁基准问题的网格数据集（80×40网格，后可自适应细化至320×160）进行实验；

**📈 对比分析**

通过与传统VTTO、仅SIMP抑制、仅投影抑制等方法对比，证明DGI投影可显著恢复边缘锐度，且对总合规（性能）影响仅约0.4%（最高0.37%），表明该方法在保持性能的同时实现了结构边缘的“反锯齿”效果；

**⚠️ 局限性**

局限性为仅在简单的合规最小化悬臂梁案例验证，未对抗性更强的应力约束或尖点问题（如L型梁）进行深入测试，未来需要在更复杂的工况下进一步评估。

---

## 102. Democratising Clinical AI through Dataset Condensation for Classical Clinical Models

**arXiv ID:** 2603.09356 | [PDF](https://arxiv.org/pdf/2603.09356v1)

**作者:** Anshul Thakur `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**通讯引用:** 13890 | [OpenAlex ID](https://openalex.org/A5040302008)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了零阶梯度估计的差分隐私数据凝缩框架，生成适用于经典临床模型的合成数据。

**💡 创新点**

能够对非可微分的决策树、Cox回归等传统模型进行数据凝缩，并在保证正式隐私的前提下保持高预测性能。

**🔧 技术方法**

使用零阶梯度估计、有限差分、组合损失、RDP隐私计数器及Adam优化器实现数据凝缩。

**📊 数据集**

在PUH、OUH、UHB三家医院的COVID‑19预测、UK Biobank蛋白组多发性骨髓瘤预测、SEER乳腺癌生存、UK Biobank糖尿病生存等六个临床数据集上进行评估。

**📈 对比分析**

与全量真实数据训练的模型在AUROC、C-index等指标上几乎相当，IPC为100时可达到95%以上的预测准确，隐私预算ε在2–3区间即可实现。

**⚠️ 局限性**

主要局限在于与预训练模型共享诱导偏差导致对其他模型的通用性不足，以及对更激进攻击（自适应、多次查询等）的鲁棒性尚未验证。

---

## 103. APPLV: Adaptive Planner Parameter Learning from Vision-Language-Action Model

**arXiv ID:** 2603.08862 | [PDF](https://arxiv.org/pdf/2603.08862v1)

**作者:** Yuanjie Lu `[一作]` (George Mason University), Xuesu Xiao `[通讯]` (George Mason University)

**通讯引用:** 1985 | [OpenAlex ID](https://openalex.org/A5017662025)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出applv框架，利用预训练的视觉‑语言模型预测经典导航规划器的参数，从而实现在线自适应控制；

**💡 创新点**

创新点在于将视觉‑语言预训练模型应用于参数学习，而非直接生成动作，并通过监督与强化学习双策略进一步优化；

**🔧 技术方法**

技术包括Qwen2.5‑VL‑3B视觉‑语言骨干、LoRA适配器、DPT回归头、历史编码器以及TD3强化学习微调；

**📊 数据集**

使用BARN仿真数据集（300个训练、300个测试环境）以及Clearpath Jackal机器人实测数据；

**📈 对比分析**

与手工专家、applr、Transformer BC、Zero‑Shot VLM等基线在四种局部规划器（DWA、TEB、MPPI、DDP）上对比，applv‑rlft实现了最高成功率、最快通行时间和最高综合得分；

**⚠️ 局限性**

局限包括较高的推理延迟（≈0.4s），在DWA/TEB等基于全局代价图的规划器中受定位误差影响显著，且数据量过大时性能提升不显著。

---

## 104. SciTaRC: Benchmarking QA on Scientific Tabular Data that Requires Language Reasoning and Complex Computation

**arXiv ID:** 2603.08910 | [PDF](https://arxiv.org/pdf/2603.08910v1)

**作者:** Hexuan Wang `[一作]` (Johns Hopkins University), Philipp Koehn `[通讯]` (Johns Hopkins University)

**通讯引用:** 31129 | [OpenAlex ID](https://openalex.org/A5112315093)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了SciTaRC，一个专家手工构造的科学表格问答基准，用于测试语言模型在理解、规划与复杂数值计算上的能力。

**💡 创新点**

创新点在于结合可执行的伪代码计划与自动化判定，明确区分规划与执行失效，从而揭示模型在科学表格上的执行瓶颈。

**🔧 技术方法**

采用链式思考（CoT）与程序思考（PoT）两种推理范式，并使用大型语言模型（如Llama‑3.3‑70B）作为评判者来对答案进行自动匹配。

**📊 数据集**

数据集包含370道人工核查的问答，来源于AI领域的arXiv论文中的原始LaTeX表格，并附带手写的伪代码计划。

**📈 对比分析**

在零样本评测中，最先进的专有模型GPT‑5仅达76.8%准确率，开源模型如DeepSeek‑V3.2在CoT下约73.6%，但在PoT下显著下降，表明即使是大模型也只能达到约50%执行精度。

**⚠️ 局限性**

局限性包括数据集规模有限、评判者为LLM且可能带偏差、对开源模型可复现性好但对封闭模型不佳，以及程序思考模式对原始LaTeX解析的鲁棒性不足。

---

## 105. PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies

**arXiv ID:** 2603.09216 | [PDF](https://arxiv.org/pdf/2603.09216v1)

**作者:** Sunjung Lee `[一作]` (Samsung Advanced Institute of Technology), Jaehoon Yu `[通讯]` (Samsung Advanced Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对在LPDDR5X-PIM上运行LLM推理时，预填充(pre-fill)和解码(decode)阶段出现的内存属性与权重布局不一致问题，提出了一套纯软件解决方案——PIM‑SHERPA。

**💡 创新点**

创新点在于：①首次系统性识别并解决内存属性不一致和权重布局冲突；②提出两种软件方法——DRAM双缓冲（DDB）与在线权重量化（OWR），通过swizzled memory copy实现动态权重布局转换，既消除了容量冗余，又保持了性能；③在不修改硬件的前提下，兼顾了CPU、GPU、NPU多种平台的可迁移性。

**🔧 技术方法**

技术手段包括：软件实现的DRAM双缓冲、swizzled memory copy、在线权重量化；利用LPDDR5X-PIM架构下的非缓存/缓存内存区域；在ExecuTorch运行时通过多线程并行复制与GEMM计算；对权重布局进行按列列主序的PIM‑aware排列。

**📊 数据集**

实验采用Meta Llama 3.2 1B与3B BF16模型（隐藏维度分别为2048/3072），在Samsung Galaxy S24+设备上进行评测。

**📈 对比分析**

方法对比：与权重复制（WD）和FACIL（硬件改造版）进行比较。结果显示：PIM‑SHERPA在保持与FACIL相当的TTFT（首个token延迟）和整体吞吐（TPS）基础上，DRAM容量节省约48%；在LPDDR5X-PIM环境下可实现约3.3×的推理加速。

**⚠️ 局限性**

局限性：①需要LPDDR5X-PIM硬件支持，非通用DRAM不适用；②在线重排（OWR）在输入序列长度低时仍有明显延迟；③在高FLOP/B平台（如高端GPU）上，重排开销相对更显著，需要更长输入序列来抵消；④实验主要集中在单机单模型，跨机或多模型场景的评估尚未覆盖。

---

## 106. MM-tau-p$^2$: Persona-Adaptive Prompting for Robust Multi-Modal Agent Evaluation in Dual-Control Settings

**arXiv ID:** 2603.09643 | [PDF](https://arxiv.org/pdf/2603.09643v1)

**作者:** Anupam Purwar `[一作]`, Aditya Choudhary `[通讯]` (Sprinklr AI)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MM‑tau‑p^2基准，专门评估多模态（语音+文本）LLM代理在双控、个性化适配及实时规划下的性能。

**💡 创新点**

创新点包括：①12项细粒度指标（涵盖目标达成、效率、恢复、安全等四大类）；②双控协议与个性化注入（persona injection & context injection）；③利用LLM‑as‑judge自动化评分；④构建可调复合分数mm‑tap，用于跨模型、跨条件的统一比较。

**🔧 技术方法**

使用技术包括：多模态输入管道（ASR→LLM→TTS）、基于Prompt的LLM‑as‑judge、PersonaPlex风格的用户模拟、Context injection、LoRA 等模型微调技术。

**📊 数据集**

数据集：在电信（Telecom）和零售（Retail）两个真实业务域上，构造任务集合（含关键字段、工具调用）和三种用户人格（None、Easy、Hard），并在语音与文本两种模态下生成对话。

**📈 对比分析**

比较方法：使用GPT‑4.1与GPT‑5两种judge分别对话进行评分，得到各指标平均值；通过mm‑tap综合评估不同模型与个性化条件下的表现。实验表明：在文本模式下LLM表现更稳健，语音模式导致模态鲁棒性下降；persona injection在提升关键字段准确率的同时显著削弱安全性；GPT‑5在评判上更乐观，导致pass率较高。

**⚠️ 局限性**

局限性：①LLM‑as‑judge 在升级/降级场景中标注不一致，引入噪声；②语音管道缺乏对中断、打断、过度交谈等真实语音交互事件的建模；③缺乏对多轮交互中的情绪和语调控制的评估；④基准仅覆盖电信与零售两域，泛化性待进一步验证。

---

## 107. InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing

**arXiv ID:** 2603.09877 | [PDF](https://arxiv.org/pdf/2603.09877v1)

**作者:** Changyao Tian `[一作]` (Shanghai AI Laboratory), Hongjie Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 InternVL-U，一种统一的多模态模型，集成了多模态理解、推理、图像生成与编辑能力，基于 InternVL3.5 进行迁移学习，并新增 MMDiT 生成头。

**💡 创新点**

创新点包括：①在统一语境建模下采用模态特定的模块化设计和解耦视觉表示；②提出双流、门控注意力的 MMDiT 生成头与统一 MSRoPE；③通过 Chain‑of‑Thought（CoT）驱动的多任务数据合成管线，涵盖文本渲染、科学推理、空间几何、幽默等多领域；④分阶段渐进训练与自反思推理相结合。

**🔧 技术方法**

技术手段涵盖：ViT 视觉编码器 + VAE 生成器；自回归 + 流匹配（Flow‑Matching）混合建模；多尺度 Rotary Position Embedding（MSRoPE）；双流 MMDiT 带门控注意力；Classifier‑Free Guidance (CFG) 与 Flow‑DPM‑Solver 推理；CoT 逻辑推理与指令生成；以及自研的 GenEditEvalKit 与 TextEdit 评测框架。

**📊 数据集**

使用的数据集包括：InternVL3.5 预训练模型权重；公开的文本‑图像、编辑数据集（如 DPG‑Bench、TIIF、OneIG‑Bench、CVTG‑2k、LongText、WISE、GenExam、ImgEdit、GEdit‑Bench、RISEBench、TextEdit 等）；以及作者自行构建的合成数据集，涵盖科学图像（SVG‑Physics、Python‑CS）、空间几何（GeoGebra、CAD）、幽默图像（Memes）、文本渲染与编辑、CoT 语义增强等。

**📈 对比分析**

通过与现有 2B 级别统一模型（BAGEL、Ovis‑U1、Janus‑Pro）以及部分 7B 级别专用模型（Qwen‑Image、Nano‑Banana‑Pro、Qwen‑Image‑Edit）对比。InternVL‑U 在 7 大 MLLM 评测中几乎与 7B 模型齐头并进；在 GenEval、DPG‑Bench、TIIF 等图像生成基准上获得最高或接近最高分；在文本渲染基准 CVTG‑2k、LongText 以及知识推理基准 WISE、GenExam 上均显著超越统一模型，并在 3‑7B 参数规模下达到 20B 级别的效果；在编辑任务（ImgEdit、GEdit‑Bench、TextEdit、RISEBench）中，CoT 加速提升 2–4 分，整体表现优于同规模统一模型，部分指标甚至逼近或超过专用编辑模型。

**⚠️ 局限性**

局限性包括：①参数规模仍低于 20B+ 专用模型，某些极致细节和大规模多概念生成仍稍逊；②依赖大量合成数据，合成质量与真实世界差异可能导致偏差；③在极端长文本或高度抽象指令下，CoT 生成与执行仍可能出现误差；④对复杂逻辑推理和多步编辑的支持虽然已显著提升，但在与大模型（如 GPT‑Image‑1.5）相比仍有差距。

---

## 108. A Regularized Ensemble Kalman Filter for Stochastic Phase Field Models of Brittle Fracture

**arXiv ID:** 2603.09728 | [PDF](https://arxiv.org/pdf/2603.09728v1)

**作者:** Lucas Hermann `[一作]` (TU Braunschweig), Ulrich Römer `[通讯]` (TU Braunschweig)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文利用集成卡尔曼滤波器（EnKF）与相位场脆性断裂模型相结合，对结构的位移场和相位场（裂纹扩展状态）进行贝叶斯推断与数据同化，从而在已知传感器位移测量的情况下显著降低裂纹路径与剩余承载力的不确定性。

**💡 创新点**

创新点包括：
- 直接对高维状态变量（位移+相位场）进行贝叶斯推断，而非传统的仅更新模型参数；
- 引入相位场正则化（proximal/ staggered 迭代）步骤，解决标准 EnKF 产生不物理解（负裂纹值、极大振荡）的问题；
- 在微观相位场（micromorphic）框架下实现上述同化流程，兼顾能量分解与不可逆性约束；
- 将正则化视为离散能量极小化的近似，提供理论上可解释的“物理约束”视角。

**🔧 技术方法**

使用的技术与方法：
- 微观相位场脆性断裂模型（AT2、体积-偏差能量分解、不可逆性约束）
- 有限元（线性/二阶）求解器，采用共轭梯度 + 预处理；
- 集成卡尔曼滤波器（EnKF）实现预测与更新；
- 正则化步骤：基于相位场的 proximal / staggered 求解，利用更大长度尺度 L>ℓ 进行平滑；
- 数据模型：传感器观测通过线性映射 H，噪声采用 Matérn 核估计协方差；
- 随机初始损伤场采样（Beta/正态）生成先验；
- 评估指标：误差减小、相位场误差、反作用力分布与峰值负载分布的标准差变化。

**📊 数据集**

实验数据集：
- 1D 拉伸杆（长度 2mm，节点 200，初始损伤位置与强度随机采样）
- 2D SENS（单边缺口剪切）基准（边长 1mm，缺口 0.5mm，150k-200k DoF），传感器随机分布（5–100 个）并采集位移测量；
- 通过数字图像相关（DIC）等方式生成人工观测数据，加入高斯噪声。

**📈 对比分析**

方法比较与性能：
- 与未同化的先验（仅随机初始损伤）相比，EnKF+正则化后，位移和相位场误差平均下降 30–70%；
- 反作用力曲线收敛到参考解，标准差缩小 50% 以上；
- 峰值负载分布的 2σ 区间从 0.15N/mm² 缩小至 0.07N/mm²；
- 通过多次更新（1–2 次）进一步降低误差，证明正则化步骤的必要性；
- 计算成本：每步约 10–20 CPU 小时（共 100 组样本），相较于 4DVAR 成本高 5–10 倍，但可并行化。

**⚠️ 局限性**

局限性与未来工作：
- 标准 EnKF 在强非线性和高维下容易产生不物理解，需额外正则化，增加实现复杂度；
- 正则化步骤需要手动选择长度尺度 L、迭代次数等超参数，影响结果稳定性；
- 仅考虑初始损伤不确定性，未对材料参数、边界条件或网格误差等其他不确定性进行联合推断；
- 对动态相位场模型（含时间导数）适配 EnKF 的研究尚缺；
- 4DVAR 之类的强约束变分方法可进一步提升物理一致性，但计算成本显著增加。

---

## 109. Exploring the Design of GenAI-Based Systems to Support Socially Shared Metacognition

**arXiv ID:** 2603.08894 | [PDF](https://arxiv.org/pdf/2603.08894v1)

**作者:** Yihang Zhao `[一作]` (King's College London), Elena Simperl `[通讯]` (Technical University of Munich)

**通讯引用:** 6555 | [OpenAlex ID](https://openalex.org/A5046030036)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

探讨基于生成式人工智能（GenAI）的组群感知工具（GATs）如何支持自主的社会共享元认知（SSM）

**💡 创新点**

提出三条设计原则：①混合架构结合规则与GenAI以生成定量与定性意识信息；②以次要视觉编码呈现GenAI语义解释，产生认知冲突促发讨论；③提供交互技巧让团队检视并评估AI解释的证据

**🔧 技术方法**

混合规则/GenAI系统、视觉编码（颜色饱和度、背景强度等）、交互技术（hover、click、选择）

**📊 数据集**

无实验数据集，论文基于文献综述与初步设计探讨

**📈 对比分析**

无实证比较与性能评估，本文仅提出概念性原则，未来需在真实协作场景中验证

**⚠️ 局限性**

缺乏经验验证，可能导致信息过载或误导团队，对何时何地展示意识信息的最佳策略尚未确定

---

## 110. Probabilistic Hysteresis Factor Prediction for Electric Vehicle Batteries with Graphite Anodes Containing Silicon

**arXiv ID:** 2603.09103 | [PDF](https://arxiv.org/pdf/2603.09103v1)

**作者:** Runyao Yu `[一作]` (Delft University of Technology), Jochen L. Cremer `[通讯]` (Delft University of Technology)

**通讯引用:** 2220 | [OpenAlex ID](https://openalex.org/A5019114577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套概率化的硅-石墨负极电池电压滞后因子预测方法，并提出了统一异构驾驶循环的数据调和框架。

**💡 创新点**

①采用量化回归实现滞后因子不确定性预测；②构建跨车型、跨化学的统一数据预处理与特征调和流程；③在资源受限条件下系统比较统计学习、树模型与GRU三类模型；④对零射、微调、联合训练等迁移学习策略进行跨车型评估。

**🔧 技术方法**

统计学习（线性分位数回归、XGBoost分位数回归）+PCA/F‑reg特征降维/选择；深度学习（量化GRU）实现序列/子序列预测；分位数损失评估；多尺度重采样与归一化；迁移学习策略（重训练、零射、微调、联合）。

**📊 数据集**

使用两款真实电动车的驾驶循环数据（车辆A与车辆B），包括电流、电压、温度等特征，数万条时序样本，通过分割、重采样、归一化后划分训练/验证/测试集。

**📈 对比分析**

通过平均分位数损失（AQL）以及RAM/ROM两项硬件指标进行比较。实验显示QGRU在序列/子序列预测中获得最低AQL≈2.15×10⁻²，ROM≈0.02 MB，RAM≈0.07 MB；QXGB+F‑reg同样表现良好（AQL≈2.67×10⁻²，ROM≈0.018 MB，RAM≈0.00068 MB）；LQR最轻量但AQL最高。迁移学习中，微调或联合训练显著优于零射预测，提升约70–80%。

**⚠️ 局限性**

依赖标签生成算法；QGRU计算量较大，可能不适用于极限硬件；数据不平衡与不同化学导致迁移泛化受限；缺乏细粒度单细胞级数据与更优标签方法。

---

## 111. Influencing LLM Multi-Agent Dialogue via Policy-Parameterized Prompts

**arXiv ID:** 2603.09890 | [PDF](https://arxiv.org/pdf/2603.09890v1)

**作者:** Hongbo Bo `[一作]` (University of Bristol), Weiru Liu `[通讯]` (University of Bristol)

**通讯引用:** 5281 | [OpenAlex ID](https://openalex.org/A5002349071)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种以prompt为动作的轻量化策略参数化框架，控制LLM多智能体对话行为；

**💡 创新点**

将prompt视为可调策略，使用规则模板与权重调度实现可解释的对话控制；

**🔧 技术方法**

使用检索增强生成（RAG）与规则模板、权重调度，基于多智能体LLM对话生成；

**📊 数据集**

利用公开政府政策、博客、网站等资料构建角色知识库，并在土地与教育资源分配两种情景下对话；

**📈 对比分析**

通过五项评估指标（响应性、反驳、非重复、证据使用、立场保持）对不同规则与权重进行量化比较，发现规则模板和权重调整能显著影响对话质量；

**⚠️ 局限性**

缺乏对更大规模智能体群的验证，且对话质量仍受LLM多样性限制，未深入探究自我评估偏差和长期对话稳定性。

---

## 112. Enhancing Debunking Effectiveness through LLM-based Personality Adaptation

**arXiv ID:** 2603.09533 | [PDF](https://arxiv.org/pdf/2603.09533v1)

**作者:** Pietro Dell'Oglio `[一作]` (Università di Pisa), Lucia C. Passaro `[通讯]` (Università di Pisa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过LLM生成并评估针对不同人格特质的个性化假新闻辟谣信息。

**💡 创新点**

创新点在于将人格特质与LLM提示结合，实现对辟谣文本的个性化改写，并使用LLM模拟评估者自动评判说服力。

**🔧 技术方法**

技术使用了基于Big Five人格二值化的prompt工程，使用Qwen3-32B生成定制化Verdict，使用Qwen3-8B、Llama3等LLM做评判。

**📊 数据集**

数据集采用FullFact的933条假新闻辟谣实例，包含主张、文章全文和原始Verdict。

**📈 对比分析**

比较方法是让LLM评估匹配、不同匹配和通用版Verdict的1-7分说服力，并统计平均分和Accuracy_p/Accuracy_cn，结果显示匹配版最高，准确率在88%以上，泛化版最低。

**⚠️ 局限性**

局限包括评估仅使用LLM模拟人类判断，缺少真实受众验证；人格模型被二值化，失去连续性；实验模型和数据有限，无法全面泛化。

---

## 113. You Didn't Have to Say It like That: Subliminal Learning from Faithful Paraphrases

**arXiv ID:** 2603.09517 | [PDF](https://arxiv.org/pdf/2603.09517v1)

**作者:** Isaia Gisler `[一作]` (ETH Zürich), Tianyi Qiu `[通讯]` (Peking University)

**通讯引用:** 65 | [OpenAlex ID](https://openalex.org/A5108279374)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在自然语言同义改写任务中，教师模型通过隐式方式将对动物偏好的行为特征传递给学生模型，即使改写内容与该偏好无语义关联或甚至相反。

**💡 创新点**

创新点在于首次证明在语义固定的自然语言改写中，教师的偏好可以被成功传递，并且即使教师对目标动物持负面观点，学生模型仍会出现相同的正向偏好，表明传递机制与语义内容无关。

**🔧 技术方法**

采用了 GPT‑4.1 nano 作为教师和学生模型，使用多层筛选和可信度评分（LLM 判别、关键词过滤、二级审查）来生成高质量改写；随后利用 OpenAI 的微调 API 对学生模型进行 10,000 条 prompt‑completion 对的微调，并通过 50 句动物偏好问题对模型进行评估。

**📊 数据集**

实验数据集由 3,000 条原句构成（1,000 条无动物相关内容、1,000 条针对海豚的负面句子、1,000 条针对老鹰的负面句子），每条句子通过教师模型产生 15‑50 条改写后筛选得到最终训练样本。

**📈 对比分析**

对比方法为三组设置：基线（无微调）、中性（对中性教师改写微调）和偏好（对偏好教师改写微调）；在动物偏好测试中，偏好组相对于中性组的偏好率提升可达 19 pp（海豚）或 12 pp（老鹰），统计显著（p < 0.001），显示隐式传递效果显著。

**⚠️ 局限性**

局限性包括：仅两种动物偏好表现强烈，其他三种弱或无显著效应；实验仅在 GPT‑4.1 nano 上进行，未验证跨模型传播；仅针对动物偏好，未检验安全相关特征；缺乏人工评估改写可信度；实验条件过于严格，生态有效性有限；改写句子数量有限，可能低估了传递幅度。

---

## 114. Do What I Say: A Spoken Prompt Dataset for Instruction-Following

**arXiv ID:** 2603.09881 | [PDF](https://arxiv.org/pdf/2603.09881v1)

**作者:** Maike Züfle `[一作]` (Karlsruhe Institute of Technology), Jan Niehues `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3332 | [OpenAlex ID](https://openalex.org/A5046084081)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

这篇论文提出并公开了一个多语言、多任务的口语指令数据集 DOWIS，用以评估语音大语言模型（SLLM）的指令跟随能力。

**💡 创新点**

创新点在于提供并行的口语与文本指令，覆盖九个任务、十一种语言和五种提示风格，使评估更贴近真实语音交互场景。

**🔧 技术方法**

论文使用的技术包括手工设计提示、人工语音录制、语音识别与评估指标（WER、CometKiwi、BERTScore、UTMOS）以及对两大 SLLM（Phi‑4 Multimodal 与 Qwen2.5‑Omni）的推理与评测。

**📊 数据集**

使用的数据集为 DOWIS 本身以及各任务的公开基准（FLEURS、MCIF、YTSeg 等），通过将 DOWIS 指令与这些基准组合进行评估。

**📈 对比分析**

实验通过比较文本与语音提示、提示风格、语言和性别，发现文本提示在文本输出任务中显著优于语音提示；在语音输出任务中，语音提示与文本提示相当或更好；非正式提示在所有任务中表现最差。

**⚠️ 局限性**

局限性包括模型对语音指令的鲁棒性不足，实验仅覆盖两种模型，且数据集受录音设备、语言与提示风格的多样性限制，未能覆盖更广泛的语言或任务。

---

## 115. A Simple Constructive Bound on Circuit Size Change Under Truth Table Perturbation

**arXiv ID:** 2603.09379 | [PDF](https://arxiv.org/pdf/2603.09379v1)

**作者:** Kirill Krinkin `[一作]` `[通讯]` (Neapolis University Pafos), Kirill Krinkin (Neapolis University Pafos)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了布尔函数真值表单点扰动下最优电路尺寸的线性上界；

**💡 创新点**

首次明确给出任意有限完备门基下的O(n)上界，并在AIG基n=4实验中验证其紧致性；

**🔧 技术方法**

采用构造性电路修复、等价检测子电路以及SAT精确合成等技术；

**📊 数据集**

使用所有222个4变量布尔函数的NPN等价类真值表及其最优电路尺寸数据；

**📈 对比分析**

在987条单点突变边上实验，最大尺寸差为4，平均差1.03，表明理论上界被实验数据所支撑；

**⚠️ 局限性**

仅在n=4、AIG基上得到验证，缺乏更大n或其他基的理论或实验支持，且尚不清楚是否存在更大的Ω(n)差距。

---

## 116. The $qs$ Inequality: Quantifying the Double Penalty of Mixture-of-Experts at Inference

**arXiv ID:** 2603.08960 | [PDF](https://arxiv.org/pdf/2603.08960v1)

**作者:** Vignesh Adhinarayanan `[一作]` (AMD Research and Advanced Development), Nuwan Jayasena `[通讯]` (AMD Research and Advanced Development)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在推理阶段系统地分析了混合专家（MoE）架构的效率瓶颈，揭示了专家路由导致的“重用碎片化”以及KV缓存占用导致的内存带宽受限，并提出了qs不等式用于预测MoE在长上下文下的结构性劣势。

**💡 创新点**

创新点在于：①将推理效率的核心归结为权重量重用而非FLOP数量；②量化专家路由对重用的影响并提出qs不等式；③通过容量优先的批量大小评估和详细的推理成本模型，展示了在不同MoE系统（DeepSeek‑V3、Qwen3‑235B、Grok‑1、Switch‑C）中的通用性。

**🔧 技术方法**

使用了基于FP8权重、BF16 KV缓存的GPU/TPU推理成本模型，结合专家并行、张量并行、流水线并行、KV并行等多种并行策略；采用了重用因子R_moe≈B_moe·k/E和qs=qs不等式进行分析；并通过多种系统与硬件配置进行敏感性分析。

**📊 数据集**

评估基于公开的MoE模型（DeepSeek‑V3、Qwen3‑235B、Grok‑1、Switch‑C）的推理性能，并与对应的质量匹配稠密模型（Dense‑5）进行对比；上下文长度范围从1k到16M tokens。

**📈 对比分析**

对比方法是：在同一硬件和并行配置下，使用容量优先的批量大小决定重用率，计算每token的推理延迟（compute、HBM、通信三项）；结果显示，在长上下文（≥32k tokens）时，质量匹配的稠密模型可比MoE提升约2–5倍吞吐量，短上下文下由于通信开销，稠密模型同样表现更好。

**⚠️ 局限性**

局限性包括：qs不等式仅适用于在HBM受限的长上下文推理场景；在极端稀疏（如Switch‑C）下MoE甚至无法在给定集群上推理；研究主要聚焦于GPU/TPU推理，未考虑CPU或ASIC环境；并未针对如何改进MoE推理的具体实现给出实质性方案。

---

## 117. Measurement-Free Ancilla Recycling via Blind Reset: A Cross-Platform Study on Superconducting and Trapped-Ion Processors

**arXiv ID:** 2603.08733 | [PDF](https://arxiv.org/pdf/2603.08733v1)

**作者:** Sangkeum Lee `[一作]` (Hanbat National University), Sangkeum Lee `[通讯]` (Hanbat National University)

**通讯引用:** 854 | [OpenAlex ID](https://openalex.org/A5037786080)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估了盲重置（blind reset）作为量子错误校正中测量无关的 ancilla 复位技术，并在 IQM Garnet、Rigetti Ankaa-3、IonQ 三个平台上进行了统一的跨平台调度评估。

**💡 创新点**

提出了基于平台特征的时延交叉长度 L⋆ 与 NVQLink 延迟扩展的决策矩阵，首次将盲重置与测量重置在洁净度、时延和逻辑错误率等 QEC 指标上进行量化比较，形成系统级的调度策略。

**🔧 技术方法**

使用了量子门级噪声模型、序列重放与尺度参数 λ 的优化、距离‑3 重复码的逻辑错误代理、以及外部反馈延迟模拟的时延模型。

**📊 数据集**

收集了 IQM Garnet、Rigetti Ankaa‑3、IonQ 三个后端在相同种子、序列长度和 1024/2048 次抽样的实验和平台校准仿真数据，包括 ancilla 洁净度、时延和逻辑错误率。

**📈 对比分析**

通过 F_clean≥F_req 与 T_blind<T_meas 的双重判据构建决策矩阵；实验结果显示 IQM、Rigetti 在 L≤12、11 时盲重置可将周期时延缩短多达 38 倍，洁净度保持 ≥0.86；IonQ 仅在 L=1 时具备优势；NVQLink 情景下 L⋆ 扩至约 78，进一步提升盲重置适用范围。

**⚠️ 局限性**

局限包括：仅基于平台级噪声模型和距离‑3 重复码代理，未覆盖多距离 surface‑code；实验仅在单种子或 50 种子水平；未测量真实网络延迟；对 λ 的离线校准有一定依赖；以及未考虑泄漏、非马尔科夫噪声等更复杂硬件效应。

---

## 118. HeteroFedSyn: Differentially Private Tabular Data Synthesis for Heterogeneous Federated Settings

**arXiv ID:** 2603.08832 | [PDF](https://arxiv.org/pdf/2603.08832v1)

**作者:** Xiaochen Li `[一作]` (University of North Carolina Greensboro), Jing Yang `[通讯]` (University of Virginia)

**通讯引用:** 7260 | [OpenAlex ID](https://openalex.org/A5071470775)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 HeteroFedSyn 框架，在水平联邦环境下实现差分隐私表格数据合成，并通过随机投影、无偏估计和自适应边缘选择提升性能。

**💡 创新点**

首次在联邦场景下引入 l2 依赖度量、随机投影压缩边缘、无偏噪声校正以及自适应边缘选择策略。

**🔧 技术方法**

基于 PrivSyn 的 2‑way 边缘合成，采用 Gaussian Mechanism、zCDP 隐私预算管理、Johnson–Lindenstrauss 随机投影和 GUM 生成合成数据。

**📊 数据集**

在 Adult、Abalone、Obesity、Insurance、Shoppers 等五个真实多维数据集上评估。

**📈 对比分析**

与中心化 PrivSyn 及两种分布式基线进行基准比较，结果显示 HeteroFedSyn 的查询误差、Wasserstein 距离与机器学习任务精度均与中心化方法相近，且自适应版本略优。

**⚠️ 局限性**

在联邦设置下噪声累积导致高维统计仍受限，且需假设最多 1/3 边缘被选取，随机投影参数选择对结果敏感。

---

## 119. RA-SSU: Towards Fine-Grained Audio-Visual Learning with Region-Aware Sound Source Understanding

**arXiv ID:** 2603.09809 | [PDF](https://arxiv.org/pdf/2603.09809v1)

**作者:** Muyi Sun `[一作]` (Beijing University of Posts and Telecommunications), Zhenan Sun `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 18237 | [OpenAlex ID](https://openalex.org/A5055505703)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Region‑Aware Sound‑Source Understanding (RA‑SSU) 任务，构建了 f‑Music 与 f‑Lifescene 两个细粒度音视数据集，并设计了 SSUFormer 模型实现音源区域分割与描述；

**💡 创新点**

创新点在于引入细粒度空间‑时间音源理解，将区域掩码与文本生成联合学习；设计 Mask Collaboration Module 与 Mixture‑of‑Hierarchical‑prompted Experts（MoHE）提升语义与空间一致性；引入区域一致性约束（RCC）强化视觉‑文本协同；

**🔧 技术方法**

使用多模态 Transformer 框架，VGGish 作为音频编码器、PVT‑v2 作为视觉编码器；MaskFormer 负责分割，CLIP 提供视觉引导；LLaVA 作为长序列提示专家；采用自注意力、跨模态注意力、Dice、Focal 与 RCC 损失；

**📊 数据集**

新建 f‑Music（3976 片，22 典型音乐场景）和 f‑Lifescene（6156 片，61 生活场景）两套数据集，均含帧级掩码与文本描述；

**📈 对比分析**

与单任务模型（SSS、AVSBench、AVIS）及多模态大模型（NExT‑GPT、ModaVerse、PG‑Video‑LLaVA）进行对比；SSUFormer 在 mIoU、F‑score、BLEU、ROUGE‑L、METEOR 等指标上均达到 SOTA，显著优于对手；

**⚠️ 局限性**

局限性包括：只适用于特定场景，缺乏开放域泛化；对音频噪声敏感；数据采集过程繁琐；仍存在空间定位与时间一致性不足的问题。

---

## 120. No evaluation without fair representation : Impact of label and selection bias on the evaluation, performance and mitigation of classification models

**arXiv ID:** 2603.09662 | [PDF](https://arxiv.org/pdf/2603.09662v1)

**作者:** Magali Legast `[一作]` (Universite catholique de Louvain), François Fouss `[通讯]` (Universite catholique de Louvain)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了标签偏差和选择偏差对分类模型评估、性能以及偏差缓解方法效果的影响，并提出了一个基于受控偏差的公平评估框架。

**💡 创新点**

提出了通过在公平基准数据上注入可控标签或选择偏差，生成双标签数据集，以实现训练偏差数据、评估公平数据的框架，从而揭示传统偏差数据评估导致的误差与公平-准确率平衡误区。

**🔧 技术方法**

采用随机森林、决策树和MLP训练模型，并利用AIF360中的八种预处理/后处理偏差缓解方法（Reweighing、Massaging、FTU、EOP、CEO、ROC-SPD/EqOp/AvOd）进行实验；通过统计公平性指标（SPD、EqOd、BCC、GEI）与准确率进行对比。

**📊 数据集**

使用了两套在公平性上已低偏差的公开数据集（Student Performance/StudentBalanced、OULAD STEM 与 SOCIAL），并在其上注入不同强度的标签或选择偏差。

**📈 对比分析**

将模型在公平测试集与偏差测试集上的指标进行对照，发现公平评估消除了准确率-公平性平衡，偏差缓解方法的有效性高度依赖于偏差类型，Reweighing 在选择偏差上表现最佳，ROC、Massaging 对标签偏差效果好但对选择偏差失效。

**⚠️ 局限性**

实验仅覆盖了标签和三种选择偏差，缺少更复杂组合偏差与真实社会数据的检验，且仅考虑了预处理/后处理方法，未涉及中间处理或对数据质量/规模的深入分析。

---

## 121. Receptogenesis in a Vascularized Robotic Embodiment

**arXiv ID:** 2603.09473 | [PDF](https://arxiv.org/pdf/2603.09473v1)

**作者:** Kadri-Ann Pankratov `[一作]` (Institute of Technology, University of Tartu), Indrek Must `[通讯]` (Institute of Technology, University of Tartu)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在一款仿飞蛾的机器人中，利用3D打印的血管网络输送聚合前体，在UV光照下实现光聚合，生成可感知紫外光的聚吡咯（PPy）接收体；该接收体通过阻抗变化驱动LED指示灯与翼部电热运动，实现从环境刺激到硬件生成再到行为输出的闭环。

**💡 创新点**

首次将生物血管输运机制与光聚合反应相结合，提出“接收体生成（receptogenesis）”的概念，使机器人能够在操作过程中根据环境信号主动合成硬件；突破了传统模块化或预制硬件的局限，实现了材料层面的自我生长与功能化。

**🔧 技术方法**

技术手段包括：PETG 3D打印血管化结构（全球输运与局部注入），Pyrrole+光诱导聚合（PPy光聚合），UV LED光照，阻抗/电阻测量（嵌入ATtiny微控制器），LED光学信号输出，Nitinol 热致伸缩翼部驱动；并结合SEM、光谱和阻抗仪等表征方法。

**📊 数据集**

本工作主要基于实验测量数据，并使用Catocala fraxini飞蛾标本的形状做参考；未使用公开的大规模数据集，所有实验数据均在实验室内获得。

**📈 对比分析**

通过比较低强度UV（0.9 LED）下未产生感知与高强度UV（1.6 LED）下完成光聚合并导致阻抗下降的实验，展示了接收体的有效感知与行为控制；实验表明在UV 1.6 LED 43下，阻抗明显下降，机器人能够及时启动翼拍动作。虽然未与其他成熟传感技术做量化对比，但闭环控制效果已得到验证。

**⚠️ 局限性**

局限性包括：需要UV光源与化学前体，血管网络的制造与维护复杂；光聚合只能在光照可达区域进行，扩展性受限；化学耗材的回收与补给需要额外机制；长期可持续性与环境噪声鲁棒性未知；未实现多功能感知或自主决定生成位置的能力。

---

## 122. CyberThreat-Eval: Can Large Language Models Automate Real-World Threat Research?

**arXiv ID:** 2603.09452 | [PDF](https://arxiv.org/pdf/2603.09452v1)

**作者:** Xiangsen Chen `[一作]` (Microsoft Research), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 6599 | [OpenAlex ID](https://openalex.org/A5062243169)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CyberThreat-Eval基准，旨在以实战工作流为核心，对大型语言模型在网络威胁情报（CTI）三阶段（triage、deep search、TI drafting）中的表现进行端到端评估，并实现了人机交互式威胁研究框架TRA以提升模型输出质量。

**💡 创新点**

创新点在于①将真实CTI工作流拆分为三阶段任务，设计与业务高度契合的评测场景；②引入分析师中心的评价指标（精确率、召回率、优先级匹配度、时间/令牌消耗等）和LLM-as-Judge机制；③构建包含专家标注的公开数据集，并对比基线模型、微调模型以及TRA的提升效果。

**🔧 技术方法**

主要技术包括：大型语言模型（GPT‑4o、o3‑mini 及其微调版），Supervised Fine‑Tuning、LLM‑as‑Judge评测框架，外部知识库检索（Bing/Google、VirusTotal）以及人机循环反馈机制。

**📊 数据集**

数据集：从企业CTI业务中抽取 488 篇文章（triage）、55 篇深度搜索输入、1310 条 IoC、1565 条 MITRE TTP、412 篇用于生成威胁主体/根因叙事的专家标注数据。该数据集已公开至 GitHub/HuggingFace。

**📈 对比分析**

对比方法：在每个任务阶段将 GPT‑4o、o3‑mini、微调版及 TRA 进行性能对比，指标包括精确率/召回率、优先级匹配成功率、平均时间/令牌数、IoC/ TTP 提取精确率、Narrative 6维度评分。结果显示：基线模型在 triage recall 过高但 precision 较低；在 deep search 基线模型检索更广但精度更差；在 TI drafting 中 IoC 提取精度约 0.82‑0.85，TTP 识别精度仅 0.24‑0.35；而 TRA 在所有指标均提升 2–10%（尤其 TTP 精确率提升至 0.42，叙事质量评分 >4.5）。

**⚠️ 局限性**

limitations: 1) 仍存在召回-精确权衡难题； 2) 对复杂推理（TTP 识别）能力不足； 3) 微调后模型在检索深度上趋于保守； 4) 评测仍依赖人工专家，扩展性受限； 5) 受限于当前外部知识库更新频率，部分 IoC/ TTP 可能失效。

---

## 123. Autonomous Edge-Deployed AI Agents for Electric Vehicle Charging Infrastructure Management

**arXiv ID:** 2603.08736 | [PDF](https://arxiv.org/pdf/2603.08736v1)

**作者:** Mohammed Cherifi `[一作]` `[通讯]` (Hyperion Consulting), Mohammed Cherifi (Hyperion Consulting)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了基于边缘AI的电动汽车充电桩自主诊断与修复架构，能够在离线状态下完成故障定位和自动修复。

**💡 创新点**

创新点包括：置信度校准的自主决策框架、混合检索增强推理、针对充电协议的域适配LLM、层级多智能体协作编排以及实现实时可验证的离线执行。

**🔧 技术方法**

技术手段涵盖：QLoRA低秩量化微调、GGUF INT4 量化、预训练模型Mistral/Llama/Qwen、NPU加速、PREEMPT_RT实时调度、内存映射模型加载与回滚机制。

**📊 数据集**

数据集：约176 k条包含OCPP、ISO 15118、IEC 61851等协议说明、服务手册、历史故障日志及合成案例；测试集18 k条标注事故，5 k条公开子集。

**📈 对比分析**

通过与规则基、云LLM基线（GPT‑4o、Claude 3.5 Sonnet）以及未微调模型对比，边缘14 B Q4_K_M模型在TTFT 28 ms、78 %自主解决率、87.6 %诊断准确率、MTTR 4–8 h的条件下表现优异。

**⚠️ 局限性**

局限性：实验数据为受控合成，缺乏真实现场验证；对硬件平台的依赖较强；机械故障诊断受软件边界限制；训练集偏向欧美设备，全球适用性待扩展。

---

## 124. Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning

**arXiv ID:** 2603.09803 | [PDF](https://arxiv.org/pdf/2603.09803v1)

**作者:** Tiehua Mei `[一作]` (Fudan University), Deqing Yang `[通讯]` (Fudan University)

**通讯引用:** 1853 | [OpenAlex ID](https://openalex.org/A5046589466)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Evidence Gain质量信号，并在RLVR训练中引入In‑Context RLVR，通过在上下文中插入演示来隐式重加权奖励，从而提升大语言模型的推理质量。

**💡 创新点**

创新点在于：①用模型自身的in‑context学习能力度量演示效用（Evidence Gain），无需外部评估器或逐步监督；②通过在训练时预置演示实现奖励的隐式重加权，避免显式计算高开销的质量信号。

**🔧 技术方法**

采用的技术包括：强化学习与可验证奖励（RLVR）、in‑context学习（ICL）、贝叶斯推理、DAGPO强化学习框架、统计相关性分析、以及大型语言模型（DeepSeek-R1-Distill-Qwen等）。

**📊 数据集**

使用的数据集：KlearReasoner‑MathSub‑30K（训练与演示集）、AIME24/25、HMMT25、MATH500、AMC23、OlympiadBench（评测基准），以及DeepSeek‑V3.2等评估器生成的高质量示例。

**📈 对比分析**

方法通过在零射（zero‑shot）评估下与多种RLVR基线（GRPO、DAPO、GSPO、CISPO、CE‑GPPO）进行对比。IC‑DAPO在1.5B与7B模型上平均提升约2.5分，竞赛数据集上提升5.6/5.8分，且训练开销仅略高于5%。

**⚠️ 局限性**

限制：①仅在数学推理领域验证，未探测对其他推理密集领域（如STEM）的一般化能力；②构建演示集需要依赖强模型（如DeepSeek‑R1），降低了方法的可迁移性与构造成本。

---

## 125. NLiPsCalib: An Efficient Calibration Framework for High-Fidelity 3D Reconstruction of Curved Visuotactile Sensors

**arXiv ID:** 2603.09319 | [PDF](https://arxiv.org/pdf/2603.09319v1)

**作者:** Xuhao Qin `[一作]` (ShanghaiTech University), Chenxi Xiao `[通讯]` (ShanghaiTech University)

**通讯引用:** 760 | [OpenAlex ID](https://openalex.org/A5075464348)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种基于近场光学立体视觉的曲面视觉触觉传感器标定框架 NLiPsCalib，配合可控光源传感器 NLiPsTac，实现仅用日常物体几次接触即可完成高精度标定。

**💡 创新点**

无需专用压迫装置或CNC设备，通过近场光学立体视觉直接从内部点光源获取表面几何，既降低成本又保持高精度；同时构建轻量网络 NLiPsNet 实现单图像实时法向估计。

**🔧 技术方法**

采用近场光学立体视觉（NLiPs）物理模型、变分优化求深度、基于多光源的光度立体；设计可控LED阵列与光学传感器；训练轻量 MLP 网络 NLiPsNet。

**📊 数据集**

通过日常物体压痕采集多光源图像生成 NLiPsCalib 训练集，覆盖不同压痕形状与多种弹性体曲面，用于训练 NLiPsNet。

**📈 对比分析**

与传统基于CNC/机器人压迫的标定方法对比，AAE 约7°，MabsE 0.059；与其他视觉触觉传感器相比，单图像实时法向误差约3.1°，显著提升精度与实时性；12个LED已足够。

**⚠️ 局限性**

NLiPs 优化离线计算耗时较长（每次 3–4 分钟，整体约 3 小时），未实现 GPU 加速；标定物体选择仍需经验，需覆盖多种变形特征。

---

## 126. Predictive Spectral Calibration for Source-Free Test-Time Regression

**arXiv ID:** 2603.09338 | [PDF](https://arxiv.org/pdf/2603.09338v1)

**作者:** Nguyen Viet Tuan Kiet `[一作]` (Hanoi University of Science and Technology), Pham Huy Hieu `[通讯]` (VinUniversity)

**通讯引用:** 395 | [OpenAlex ID](https://openalex.org/A5108905044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于源自由的测试时自适应框架——Predictive Spectral Calibration（PSC），用于图像回归任务。

**💡 创新点**

创新点在于将子空间对齐扩展到完整的块谱匹配：在保持预测支持子空间对齐的同时，显式校准正交残差空间的谱特性，理论上可直接限制目标域预测均值漂移。

**🔧 技术方法**

采用块高斯模型、K²个探针向量进行支持子空间的全阶统计对齐，残差空间采用均值-方差对齐；目标损失为对齐损失与残差对齐损失的加权和；在测试时仅更新特征提取器中的归一化层参数。

**📊 数据集**

实验数据集包括 SVHN→MNIST 的跨域转移和 UTKFace 在 13 种噪声/伪影下的鲁棒性评估。

**📈 对比分析**

与多种基线（DANN、RSD、FR、TTT、AM、BNA、SSA）进行对比，PSC 在 SVHN→MNIST 下即使 λ=0 也优于 SSA，在 UTKFace 相关噪声下 λ=1 时表现最优，整体提升 R²/MSE/MAE 均超过传统基线，接近 oracle 上限。

**⚠️ 局限性**

局限性包括：需要根据不同类型的域偏移调节 λ，跨域大幅度变化时 residual 对齐可能限制适应能力；仅适用于线性回归头和需要源子空间统计的模型；在多任务或高维连续目标场景下的推广性尚未验证。

---

## 127. DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval

**arXiv ID:** 2603.09185 | [PDF](https://arxiv.org/pdf/2603.09185v1)

**作者:** Taegyeong Lee `[一作]` (Miri DIH), JooYoung Jang `[通讯]` (Miri DIH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的直接嵌入优化(DEO)方法，用于处理含否定和排除的检索查询；

**💡 创新点**

创新点在于利用LLM对查询进行正负子查询分解，并通过对原始查询嵌入施加对比损失实现嵌入空间的即时优化，无需对模型进行微调；

**🔧 技术方法**

采用LLM提示式查询分解、预训练嵌入模型（如BGE、CLIP）以及对比学习的优化步骤；

**📊 数据集**

使用NegConstraint、NevIR文本检索基准以及COCO-Neg图像检索基准进行评估；

**📈 对比分析**

与BGE、CLIP等基线模型对比，DEO在NegConstraint上MAP提升0.1028、nDCG@10提升0.0738；在COCO-Neg上Recall@5提升约6%，在多模态检索和文本检索任务中均表现出显著优势；

**⚠️ 局限性**

依赖LLM的分解质量，若分解不准确会影响检索效果，同时对每个查询需要额外的优化步骤，可能导致实时性能受限。

---

## 128. Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture

**arXiv ID:** 2603.09681 | [PDF](https://arxiv.org/pdf/2603.09681v1)

**作者:** Tom Wehrbein `[一作]` (Leibniz University Hannover), Bodo Rosenhahn `[通讯]` (Leibniz University Hannover)

**通讯引用:** 10032 | [OpenAlex ID](https://openalex.org/A5040412734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 FootMR 方法，在现有 3D 人体恢复模型的基础上利用检测到的 2D 脚部关键点对脚踝旋转进行细化，显著提升脚部 3D 动作重建质量。

**💡 创新点**

创新点在于使用膝关节及初始脚踝旋转作为上下文，仅预测残差角度以消除 2D‑>3D 解析不确定性，并通过全局旋转表示与随机根姿态增强泛化能力。

**🔧 技术方法**

技术实现基于 Transformer 的残差预测网络（采用 RoPE 位置编码与 6D 旋转表示），结合 2D 脚关键点、膝关节全局旋转及随机根旋转数据增强。

**📊 数据集**

训练与评估数据集包括大规模 MoCap（AMASS）、视频集（BEDLAM、Human3.6M、3DPW）、多视角数据（MOYO、RICH）以及新收集的 MOOF（含复杂脚步动作）。

**📈 对比分析**

与 GVHMR、CameraHMR、TRAM 等最先进方法在 MOYO、RICH、MOOF 上比较，FootMR 在脚部专用指标 AJAE、N‑MPJPEF、PCKF 上提升约 30%–60%，显著优于所有竞争方案。

**⚠️ 局限性**

局限性在于仅改进单一脚踝关节，无法捕捉脚趾卷曲等细微动作，且 SMPL‑X 脚部模型过于简单，未来需扩展至更细致的脚部关节模型。

---

## 129. Unlocking High-Fidelity Analog Joint Source-Channel Coding on Standard Digital Transceivers

**arXiv ID:** 2603.09080 | [PDF](https://arxiv.org/pdf/2603.09080v1)

**作者:** Shumin Yao `[一作]` (Pengcheng Laboratory), Shuguang Cui `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 30471 | [OpenAlex ID](https://openalex.org/A5009164482)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可在标准数字 WiFi PHY 上实现高保真模拟联合源通道编码（JSCC）的框架 D2AJSCC。

**💡 创新点**

创新点包括：利用 OFDM 子载波结构做波形合成器实现信号格式兼容；通过软件定义模块（SDM）进行 PHY 反演，解决卷积编码不可逆问题；设计 ProxyNet 作为可微分网络代理，实现端到端梯度传递并避免 JSCC 退化；以及构建多尺度补偿网络纠正量化、循环前缀和导频干扰。

**🔧 技术方法**

使用深度学习编码器-解码器、OFDM 物理层、SDM 逆演算、ProxyNet（U-Net 结构）和 TimesNet 风格补偿网络；在 MATLAB WLAN 工具箱仿真 IEEE 802.11a 20 MHz 链路上训练和评估。

**📊 数据集**

数据集采用 MNIST 手写数字图像进行语义图像传输实验。

**📈 对比分析**

与理想模拟 JSCC、使用直通估计（STE）的 JSCC、以及零样本 AWGN 训练的 JSCC 进行对比。实验表明 D2AJSCC 近似理想 JSCC 的渐进降解曲线，显著避免了阈值式的崩溃（cliff effect），在整个 SNR 范围内保持高重建质量。

**⚠️ 局限性**

局限性包括：在高 SNR 时仍受量化误差、循环前缀覆盖和导频干扰影响，导致性能略低于理想；需要在训练前完成 SDM 逆演算和离线采样；子载波占用比例受编码率限制，可能降低有效比特率；框架目前仅验证于 OFDM PHY，扩展至其他物理层需进一步研究。

---

## 130. VIVID-Med: LLM-Supervised Structured Pretraining for Deployable Medical ViTs

**arXiv ID:** 2603.09109 | [PDF](https://arxiv.org/pdf/2603.09109v1)

**作者:** Xiyao Wang `[一作]` (Shanghai University of Engineering Science), Xihe Qiu `[通讯]` (Shanghai University of Engineering Science)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5007950680)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

基于冻结的大语言模型对医学视觉Transformer进行结构化监督预训练，得到可直接部署的ViT模型。

**💡 创新点**

创新点在于利用统一医学架构（UMS）将临床发现转化为可验证的JSON格式，并通过结构化预测分解（SPD）与正交正则化将跨注意力拆分为互补查询组，从而实现高效的语义蒸馏并在推理时完全去除LLM。

**🔧 技术方法**

主要技术包括冻结LLM教师、UMS构造、答案可答性掩码、SPD投影器、正交正则化和基于token预测的损失。

**📊 数据集**

使用的数据集包括CheXpert（30k胸片）用于预训练和线性探测，NIH ChestX-ray14用于零样本跨域迁移，LIDC‑IDRI（CT切片）和OrganAMNIST（11器官分类）用于跨模态通用性评估。

**📈 对比分析**

在CheXpert上宏观AUC 0.8588，超越BiomedCLIP +6.65点；在NIH上零样本宏观AUC 0.7225，超越BiomedCLIP +5.00点；在OrganAMNIST上宏观AUC 0.9969、F1 0.9322，显著高于BiomedCLIP。

**⚠️ 局限性**

局限性包括对小样本数据集的方差较大、模型仍需大量预训练数据（尽管比BiomedCLIP低500×）、以及在不同医学任务和模态的进一步泛化性仍需验证。

---

## 131. Stable Boundaries of Opinion Dynamics in Heterogeneous Spatial Complex Networks

**arXiv ID:** 2603.09485 | [PDF](https://arxiv.org/pdf/2603.09485v1)

**作者:** Mats Bierwirth `[一作]` (ETH Zurich), Johannes Lengler `[通讯]` (ETH Zurich)

**通讯引用:** 1303 | [OpenAlex ID](https://openalex.org/A5084778631)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在几何不均匀随机图（GIRG）上进行的多数投票意见动态，并揭示大规模局部意见域会在网络几何与度分布特征作用下停滞不消失，形成稳定的意见边界。

**💡 创新点**

首次证明在GIRG上，多数投票可出现“被抑制的粗化”——即意见共存而非最终一致，且给出了严格的均值场模型和稳定接口分布的存在性证明。

**🔧 技术方法**

使用均值场近似、连续界面分析、正则性与对称性论证以及构造可行子解的方法，结合Poisson与高斯逼近的随机邻接分析。

**📊 数据集**

使用10,000个顶点、二维单位正方体的GIRG模拟，平均度数20，权重服从幂律分布，实验验证了理论预测的临界规模和τ指数对生存的影响。

**📈 对比分析**

通过数值实验与理论预测比较，观察到小域被消灭而大域收敛为球形且保持不变，说明模型在大平均度数下的稳定性优于传统无几何网络的全局一致性。

**⚠️ 局限性**

仅在零温度（确定阈值）下分析，未考虑温度效应；对权重截断的假设在理论上需要大且可能排除高权重顶点；对实际网络的泛化以及对不同动力学（如投票模型、3-多数等）的适用性尚未验证。

---

## 132. Tensor Train Decomposition-based Channel Estimation for MIMO-AFDM Systems with Fractional Delay and Doppler

**arXiv ID:** 2603.09293 | [PDF](https://arxiv.org/pdf/2603.09293v1)

**作者:** Ruizhe Wang `[一作]` (National Mobile Communications Research Laboratory, Southeast University), Jiangzhou Wang `[通讯]` (National Mobile Communications Research Laboratory, Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

针对多输入多输出 AFDM（Affine Frequency Division Multiplexing）系统，在存在分数延迟和多普勒频移的双向选择性信道下，设计了时间-仿射频域嵌入式导频结构，并提出基于 Vandermonde 结构张量训练（TT）分解的低复杂度信道参数估计算法；同时推导了 Ziv‑Zakai 边界（ZZB）作为全局均方误差下界。

**💡 创新点**

创新点主要有：
1) 引入时间维度的导频嵌入，利用时间相位变化实现分数延迟和多普勒频移的精确同步估计；
2) 结合张量训练分解与 Vandermonde 结构，构造高效的子空间估计流程，显著降低计算复杂度；
3) 首次为 AFDM 多维信道模型推导 ZZB，揭示低信噪比阈值效应并提供更紧致的性能下限；
4) 在算法设计中避免了传统的迭代 ALS 或 NOMP 过程，实现一次性完成参数估计。

**🔧 技术方法**

主要技术包括：
- 复合导频与相位旋转利用的多维张量表示（STAF 张量）;
- 张量训练（TT）分解与 EVD/ESPIRIT 结合的子空间参数提取；
- 旋转不变性与 Vandermonde 结构的特征匹配；
- ZZB 推导中的 Fisher 信息矩阵、贝叶斯先验与误判概率下界计算。

**📊 数据集**

实验使用的仿真数据集为：
- 载波 15 GHz，子载波间隔 30 kHz，M=1024 子载波；
- 基站 16 个天线、移动站 1 个天线；
- 5 条传播路径，最大速度 300 km/h（ν_max≈4.17 kHz）；
- 采用 16/64/256 QAM 进行 BER/SE 测试。

**📈 对比分析**

与 Rec‑ALS、CP‑ALS、NOMP、LMMSE 等基线算法相比：
- 在中高 SNR 区间，NMSE、BER、SE 近似甚至优于 Rec‑ALS；
- 低 SNR 区间 ZZB 能准确捕捉阈值效应，算法 MSE 接近 ZZB；
- 计算时间上相较 CP‑ALS/Rec‑ALS 低约 1–2 个数量级，NOMP 更高；
- 对路径数增多时，CP‑ALS 性能急剧下降，而提出算法保持鲁棒性。

**⚠️ 局限性**

局限性包括：
- 采用子空间估计而非最大似然估计，低 SNR 时仍存在性能缺口；
- 假设路径参数相互独立且分量稀疏，实际多径相干或非独立场景需进一步验证；
- 论文未对多天线移动站或频率同步误差做深入分析；
- 仅针对 AFDM 结构，尚需探索在其他一维/二维调制中的适用性。

---

## 133. MedMASLab: A Unified Orchestration Framework for Benchmarking Multimodal Medical Multi-Agent Systems

**arXiv ID:** 2603.09909 | [PDF](https://arxiv.org/pdf/2603.09909v1)

**作者:** Yunhang Qian `[一作]` (National University of Singapore), Hongwei Bran Li `[通讯]` (National University of Singapore)

**通讯引用:** 9557 | [OpenAlex ID](https://openalex.org/A5100325337)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MedMASLab，一个统一的多模态医学多智能体系统框架，整合11种不同架构、24种医学模态、473个疾病的数据，并提供标准化的通信协议与推理环境。

**💡 创新点**

创新点在于：①通过统一的代理抽象与多模态输入，消除实现偏差；②引入基于大型视觉语言模型的零射击语义评估，克服传统字符串匹配的脆弱性；③构建跨11个医学基准的最大全景评测，揭示多智能体系统在跨专业时的“专业化惩罚”。

**🔧 技术方法**

技术主要包括：多模态统一推理层、VLM驱动的语义判定器、动态vLLM服务层、统一配置与成本追踪、以及多种评估协议（VLM-SJ、VLM-EC、Rule-MR、Rule-FL、Rule-EM）。

**📊 数据集**

使用的数据集覆盖11个医学领域，包括Med-QA、Med-CMR、SLAKE-En、MedVidQA、PubMedQA、MedQA、MedBullets、MMLU、VQA-RAD、MedXpertQA-MM、DxBench及M3CoTBench，共计473个疾病。

**📈 对比分析**

与多种一般及医学专用多智能体方法（如MDTeamGPT、DyLAN、AutoGen、Meta-Prompting、Reconcile等）进行系统对比，发现单一方法在不同任务上表现各异，整体上多智能体框架在高阶推理任务上能提升性能，但在检索型或多模态融合任务中可能出现信息噪声与性能下降。

**⚠️ 局限性**

局限性包括：高度任务专用性导致跨专业迁移困难；对底层VLM的依赖使系统稳定性和成本高度受模型大小和指令遵循能力影响；格式化错误仍是评价与推理的主要瓶颈，且大规模多模态推理仍面临显著计算与能耗成本。

---

## 134. Composed Vision-Language Retrieval for Skin Cancer Case Search via Joint Alignment of Global and Local Representations

**arXiv ID:** 2603.09108 | [PDF](https://arxiv.org/pdf/2603.09108v1)

**作者:** Yuheng Wang `[一作]`, Tim K. Lee `[通讯]` (University of British Columbia)

**通讯引用:** 3705 | [OpenAlex ID](https://openalex.org/A5087105596)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在皮肤癌案例检索中提出了一种组合视觉-语言查询的检索框架。

**💡 创新点**

创新点是将参考图像和临床文本通过跨模态Transformer融合，学习层级查询表示，并采用联合全局-局部对齐以及域感知加权相似度。

**🔧 技术方法**

技术包括Swin Transformer视觉骨干、BERT文本编码、跨模态Transformer、区域注意力掩码、全局和局部余弦相似度融合。

**📊 数据集**

使用公开的Derm7pt数据集，包含888张经活检确认的图像（黑色素瘤、痣、良性角化样病变）。

**📈 对比分析**

与ResNet50-CosSim、SNF-DCA、MaskRCNN-Fusion、DAHNET等方法比较，在Accuracy@1/2/4和mAP上均取得最高值（Acc@1≈79.3%，Acc@4≈87.3%，mAP≈81.7%）。

**⚠️ 局限性**

局限包括数据规模有限、只针对Derm7pt的三类病变、未对更大范围或其他皮肤病种验证，且需要进一步验证在真实临床工作流中的鲁棒性。

---

## 135. Benchmarking Political Persuasion Risks Across Frontier Large Language Models

**arXiv ID:** 2603.09884 | [PDF](https://arxiv.org/pdf/2603.09884v1)

**作者:** Zhongren Chen `[一作]` (Yale University), Quan Le `[通讯]` (Yale University)

**通讯引用:** 410 | [OpenAlex ID](https://openalex.org/A5113786257)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七款前沿大型语言模型（Claude、Gemini、GPT、Grok 等）在两项在线问卷实验中与人类政治广告进行对比，评估其说服力，并通过 LLM 辅助对话分析挖掘并量化其使用的说服策略。

**💡 创新点**

①首次将多款最新 LLM 与真实政治广告直接对照；②开发无先验标签的 LLM 辅助对话分析框架，让说服策略从对话中自下而上地涌现；③系统性分析信息化提示在不同模型上的异质效果。

**🔧 技术方法**

使用随机对照实验、线性回归、加权变异元 Meta‑分析、Hajek 逆权重估计；对话分析采用 GPT‑5 mini 进行策略归纳、GPT‑5.2 进行策略评分，配合人工验证。

**📊 数据集**

两份问卷实验总样本 19,145 名 Prolific 受试者，涉及两大议题（移民、最低工资）及两种立场（支持/反对），对话平均 7 轮、约 600–800 词。

**📈 对比分析**

比较方法：对 LLM 与人类广告的平均处理效应（ATE）进行线性回归，随后用随机效应 Meta‑分析汇总不同议题/立场。结果显示：Claude 系列说服力最高，其次 GPT 与 Gemini 同级；Grok 说服力最低；所有 LLM 的平均说服效果均显著高于人类广告，且 Claude 与 GPT‑5 具备显著差异。信息化提示对 Claude、Grok 有正向效应，对 GPT‑5 产生负向效应。

**⚠️ 局限性**

限制：仅测试两类议题和两种立场，受试者来源单一（Prolific），立场非随机分配；对话策略与效果的关联为相关性而非因果；实验仅覆盖文本交互，未考虑社交媒体等多渠道传播；新模型快速迭代后结果可能过时。

---

## 136. Robust Parameter and State Estimation in Multiscale Neuronal Systems Using Physics-Informed Neural Networks

**arXiv ID:** 2603.08742 | [PDF](https://arxiv.org/pdf/2603.08742v1)

**作者:** Changliang Wei `[一作]` (University of Iowa), Xueyu Zhu `[通讯]` (University of Iowa)

**通讯引用:** 898 | [OpenAlex ID](https://openalex.org/A5006419753)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于物理信息神经网络（PINN）的框架，用于在只观测到部分、噪声较大的电压信号时，同时重构神经元模型的隐藏状态变量和未知的生理参数。

**💡 创新点**

在多尺度、强非线性发放与崩塌模型（如 Morris–Lecar 及呼吸模型神经元）上证明了 PINN 的鲁棒性与精确性，尤其能够在初始参数不佳且观测窗口短的情况下实现可靠的参数推断与状态重建，填补了传统前向求解方法在此类逆问题上的不足。

**🔧 技术方法**

采用物理信息神经网络（PINN）与神经元微分方程的约束结合，利用深度学习技术进行逆问题求解。

**📊 数据集**

使用合成数据：Morris–Lecar 模型在不同发放与崩塌模式下的电压信号，以及呼吸模型神经元的电压信号，仅利用短时间窗口内的部分电压观测。

**📈 对比分析**

与传统数值前向求解器及参数估计算法对比，PINN 在相同观测窗口下能够更快收敛、获得更准确的参数与状态重构，并在初始猜测不佳时仍保持鲁棒性；性能表现显著优于传统方法。

**⚠️ 局限性**

尚未在真实实验数据上验证，计算成本相对较高，对极端噪声或更复杂模型的泛化能力仍需进一步评估。

---

## 137. EvoDriveVLA: Evolving Autonomous Driving Vision-Language-Action Model via Collaborative Perception-Planning Distillation

**arXiv ID:** 2603.09465 | [PDF](https://arxiv.org/pdf/2603.09465v1)

**作者:** Jiajun Cao `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 10803 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了EvoDriveVLA框架，实现了视觉-语言-动作模型在自动驾驶中的协同感知与规划蒸馏

**💡 创新点**

引入自锚定视觉蒸馏与未来感知的oracle引导轨迹蒸馏，并结合粗细轨迹细化和MC‑Dropout多样性采样，解决视觉表征退化与长期规划不稳定问题

**🔧 技术方法**

自锚定视觉蒸馏（AnchorFormer）、oracle教师模型、粗细轨迹细化、Monte Carlo Dropout采样、双层（隐藏状态与logits）蒸馏损失

**📊 数据集**

nuScenes（开放环）与NAVSIM（闭环）两个公开基准数据集

**📈 对比分析**

在nuScenes开放环评估中，EvoDriveVLA在1/2/3秒L2误差和碰撞率均领先传统、LLM、蒸馏等基线；在NAVSIM闭环评估中，3B模型PDMS得分提升3.4点，甚至超过8B、InternVL3等更大模型

**⚠️ 局限性**

依赖未来图像与车辆状态的oracle教师，需额外采集或生成未来信息；在极端场景或数据稀缺时可能难以获得高质量未来输入，导致蒸馏效果下降

---

## 138. Correction of Transformer-Based Models with Smoothing Pseudo-Projector

**arXiv ID:** 2603.09815 | [PDF](https://arxiv.org/pdf/2603.09815v1)

**作者:** Vitaly Bulgakov `[一作]` (Profiteya LLC), Vitaly Bulgakov `[通讯]` (Mass General Brigham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在Transformer等神经网络中引入轻量级的“伪投影器”对隐藏表示进行多尺度平滑，以改善训练收敛速度和泛化性能。

**💡 创新点**

创新点在于将多重网格（multigrid）思想迁移到网络隐藏层，通过可学习的约束（restriction）和细化（prolongation）操作实现近似正交投影，既不改变网络架构也不改动损失函数，提供了一种新的隐层正则化和优化加速方法。

**🔧 技术方法**

技术主要包括：可学习的线性投影矩阵（Q、Q*），凸组合多尺度投影，残差式投影更新 h ← h + η(P_MS h – h)，以及在Transformer中对特征维度和序列维度分别进行投影。

**📊 数据集**

使用的实验数据集包括：二维“波浪”决策边界的合成数据、Quora Question Pairs（QQP）、Stanford Natural Language Inference（SNLI）和MIMIC‑IV住院摘要（长文本、噪声多）。

**📈 对比分析**

与不使用投影器的基线模型（Plain）对比，采用准确率、精确率、召回率、F1、训练/验证损失和梯度范数等指标。结果显示：在类不平衡、噪声注入或非凸决策边界等难度较高的场景下，投影器模型在收敛速度、最终F1及整体泛化表现上均优于基线，尤其在类不平衡和噪声强的实验中差距显著。

**⚠️ 局限性**

局限性包括：仅在小规模或中等规模任务上验证，缺乏大规模语言模型（如BERT、GPT）的实证；投影器需要额外的超参数（投影维度、α、η 等），调参成本较高；理论上不提供严格的收敛或泛化保证，主要依赖经验与实验验证。

---

## 139. FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting

**arXiv ID:** 2603.09661 | [PDF](https://arxiv.org/pdf/2603.09661v1)

**作者:** Boya Zhang `[一作]` (Shanghai Jiao Tong University), Xing He `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5897 | [OpenAlex ID](https://openalex.org/A5101560788)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为FreqCycle的时频特征提取框架，用于提升时间序列预测的准确性和效率。

**💡 创新点**

创新点在于：①滤波增强周期预测（FECF）显式学习低频周期模式；②分段频域模式学习（SFPL）利用STFT+可学习滤波器与自适应加权显著提升中高频能量；③层次化多尺度模块MFreqCycle实现了多周期耦合与长视窗的高效解耦。

**🔧 技术方法**

核心技术包括：FFT/iFFT、可学习滤波器、STFT分段、交叉尺度融合、MLP+线性层、跨尺度加权融合、以及基于频域与时域的并行处理。

**📊 数据集**

使用七个公开基准数据集：ETT（ETTh1、ETTh2、ETTm1、ETTm2）、Electricity（ECL）、Weather、Traffic。

**📈 对比分析**

与多种基线（DLinear、CycleNet、TimeMixer、FreTS、FilterNet、FITS、iTransformer、PatchTST）在MSE/MAE上进行对比，FreqCycle在10/14个指标中夺得第一，2个指标位列第二，整体性能显著优于对比模型，同时保持较低的内存占用和较快的训练速度。

**⚠️ 局限性**

主要局限包括：对更长周期（如年周期）的验证不足，长lookback窗口下的性能提升有限，以及在极端噪声或非周期性极强的数据上仍需进一步改进。

---

## 140. M3GCLR: Multi-View Mini-Max Infinite Skeleton-Data Game Contrastive Learning For Skeleton-Based Action Recognition

**arXiv ID:** 2603.09367 | [PDF](https://arxiv.org/pdf/2603.09367v1)

**作者:** Yanshan Li `[一作]` (Shenzhen University), Linhui Dai `[通讯]` (Shenzhen University)

**通讯引用:** 1035 | [OpenAlex ID](https://openalex.org/A5072786229)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多视角极小极大无限骨架数据游戏对比学习框架M3GCLR，用于无监督骨架动作识别。

**💡 创新点**

创新点包括：① 构造无限骨架数据游戏（ISG）并给出平衡定理；② 设计多视角旋转增强模块（MRAM）；③ 通过互信息约束和双损失平衡（DLEO）实现对抗学习与冗余抑制。

**🔧 技术方法**

使用旋转增强、互信息最大化、极小极大游戏理论、InfoNCE+KL双损失、ST‑GCN骨干网络、记忆池与动量教师等技术。

**📊 数据集**

使用数据集：NTU RGB+D 60/120 以及 PKU‑MMD Part I/II。

**📈 对比分析**

与 SkeletonCLR、AimCLR 等基线以及最新SOTA 在线性评估下对比，M3GCLR 在 NTU‑RGB+D 60 的 X‑View/X‑Sub 取得 82.1%/85.8%，PKU‑MMD Part I 取得 89.1%，均较基线提升 3–4% 并接近或超越最新方法。

**⚠️ 局限性**

局限性：极端旋转角度需手动调节，过大或过小均导致性能下降；对极端视角或高度交互场景的鲁棒性仍有限；在样本量较小或简单数据集上性能略低于 AimCLR++。

---

## 141. Modelling the Diachronic Emergence of Phoneme Frequency Distributions

**arXiv ID:** 2603.09503 | [PDF](https://arxiv.org/pdf/2603.09503v1)

**作者:** Fermín Moscoso del Prado Martín `[一作]` (University of Cambridge), Suchir Salhan `[通讯]` (University of Cambridge)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5114634016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

通过构建随机分裂合并模型，模拟音位库存随时间的演化，研究音位频率分布以及库存大小与相对熵的负相关关系是否可由音变过程自发产生。

**💡 创新点**

创新点在于：①将功能负载偏置与库存大小的中心趋向同时纳入模型；②证明负相关关系可不依赖显式的补偿机制而是音变过程与库存趋向的自然结果；③提出基于库存大小的指数调节概率来控制随机游走。

**🔧 技术方法**

技术手段包括：Hoenigswald 1965 分类的三种音变事件（primary split, secondary split, merger），随机采样 α 值和音位目标，功能负载近似为频率的采样偏置，库存大小自适应调节的指数函数，以及 Monte‑Carlo 模拟 400 语言 1000 步。

**📊 数据集**

使用的数据集为：Macklin‑Cordes Round 2020（澳大利亚语言）和 NorthEuraLex（北欧语言）作为实测基准；PHOIBLE 2.0 用于确定初始库存均值（34）。

**📈 对比分析**

比较方法：绘制 rank‑frequency 曲线、PIS‑相对熵相关系数，并与实测语言进行对比。结果显示：①Naïve 模型产生正相关；②功能负载模型仍正相关；③加入库存中心趋向后获得负相关 r≈−0.12，且库存波动趋于收敛，基本与真实语言匹配。

**⚠️ 局限性**

局限性：①模型未考虑音位功能负载的精确测度与语音细节；②参数（如 μ=34）的选择相对任意；③仅在宏观层面检验，未验证微观补偿机制；④模拟结果对极小/极大库存仍可能产生偏差。

---

## 142. Trajectory Optimization for Self-Wrap-Aware Cable-Towed Planar Object Manipulation under Implicit Tension Constraints

**arXiv ID:** 2603.09557 | [PDF](https://arxiv.org/pdf/2603.09557v1)

**作者:** Yu Li `[一作]` (Munich Institute of Robotics and Machine Intelligence, Technical University of Munich), Hamid Sadeghian `[通讯]` (Munich Institute of Robotics and Machine Intelligence, Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出并实现了自包裹感知的单绳拖拽轨迹优化框架，构建了从完整模式到三种松弛（FMR、BMR、IMR）的求解器层次。

**💡 创新点**

创新性地将绳索自包裹与张力互补约束联合成模式条件的轨迹优化，并设计了多层次松弛策略以缓解模式切换导致的非平滑性。

**🔧 技术方法**

采用多射击离散化、CasADi+IPOPT求解、MPCC到NLP的平滑松弛、状态依赖门控函数以及非线性补偿残差（NCR）等技术。

**📊 数据集**

在二维平面上自定义的三类拖拽基准（斜坡、弧形、障碍）进行评估，使用随机初始状态和参数扰动作为测试集。

**📈 对比分析**

通过随机初始化的成功率、跟踪RMSE、缠绕比例和求解时间比较三种求解器，发现IMR在保持较高成功率和合理误差的同时，能更稳定地激活自包裹；BMR求解更稳定但缠绕不足；FMR在大多数场景下失效。

**⚠️ 局限性**

局限性包括仅支持单面、单点红irection、无摩擦的理想化模型，无法处理多重缠绕、边缘摩擦、柔性绳索等实际复杂情况，也未提供在线重规划或MPC框架。

---

## 143. MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games

**arXiv ID:** 2603.09022 | [PDF](https://arxiv.org/pdf/2603.09022v1)

**作者:** Yunfei Xie `[一作]` (Rice University), Zhangyang Wang `[通讯]` (The University of Texas at Austin)

**通讯引用:** 20539 | [OpenAlex ID](https://openalex.org/A5048522863)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MEMO——一种基于自对弈的无权重更新框架，通过在推理时优化提示与记忆上下文来提升多回合多代理 LLM 游戏的胜率与稳定性。

**💡 创新点**

创新点在于：① 结合持久记忆库提炼轨迹洞察并在后续对弈中注入；② 用 TrueSkill 进行可靠性加权的提示选拔；③ 采用优先回放检索罕见决策点；④ 通过随机与记忆增强双重提示生成实现高效探索与累积学习。

**🔧 技术方法**

使用技术包括：自对弈、锦标赛式提示演化、TrueSkill Bayesian 评级、CRUD 操作的持久记忆库、优先回放（Inverse‑Frequency）以及多模型评估。

**📊 数据集**

实验基于五个文本游戏（KuhnPoker、SimpleNegotiation、TwoDollar、SimpleTak、Briscola）来自 LMGame‑Bench/BALROG 等公开评测套件。

**📈 对比分析**

与静态提示、其他提示优化方法以及 RL 基线进行对比；在 GPT‑4o‑mini 上平均胜率从 25.1% 提升至 49.5%，在 Qwen‑2.5‑7B‑Instruct 上从 20.9% 提升至 44.3%；相对标准误差从 43.3% 降至 6.4%，并且只需 2,000 次自对弈，约为 RL 的 1/19。

**⚠️ 局限性**

局限性包括：在完美信息游戏中 RL 仍更有效；跨模型迁移效果有限；记忆抽象仍可能缺乏对复杂策略的深度表达；优先回放与记忆操作参数需人工调优；对罕见状态的回放依赖手工设定的概率与优先指数。

---

## 144. Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs

**arXiv ID:** 2603.09906 | [PDF](https://arxiv.org/pdf/2603.09906v1)

**作者:** Zorik Gekhman `[一作]`, Jonathan Herzig `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究推理在大型语言模型中的作用，探究其对参数知识召回和单步事实问答的影响。

**💡 创新点**

发现推理可通过计算缓冲和事实预热两种机制显著扩展模型的参数知识边界，并证明推理生成的中间事实若为幻觉会降低最终答案的准确率。

**🔧 技术方法**

使用可切换推理的混合模型、pass@k 度量、生成式自检（事实提取、幻觉检测）、Gemini‑2.5‑Flash 作为验证器以及对比实验。

**📊 数据集**

SimpleQA、OpenBookQA 等闭卷 QA 数据集，包含 1000 条样本。

**📈 对比分析**

与开启/关闭推理的基线对比，计算 Ω 指标并在 k=1~100 的 pass@k 曲线上展示；推理开启时 pass@k 在高 k 处提升近一倍，Ω 在较弱模型上最高。

**⚠️ 局限性**

推理长度最优点未知、计算缓冲效应有限、事实预热的泛化与幻觉问题仍需解决，且实验集中于单步事实问答，未覆盖更复杂的多跳推理场景。

---

## 145. Case Study: Performance Analysis of a Virtualized XRootD Frontend in Large-Scale WAN Transfers

**arXiv ID:** 2603.09568 | [PDF](https://arxiv.org/pdf/2603.09568v1)

**作者:** J M da Silva `[一作]` (Núcleo de Computação Científica), R L Iope `[通讯]` (Núcleo de Computação Científica)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对T2_BR_SPRACE存储前端架构进行案例研究，测量并记录高强度数据传输的吞吐量。

**💡 创新点**

通过在8台XRootD虚拟机上结合BBR拥塞控制、SR-IOV网络、巨大的TCP缓冲区以及pNFS后端，实现超过40 Gb/s的实际传输速度，验证了虚拟化与网络调优的协同效应。

**🔧 技术方法**

使用XRootD VM集群、dCache/pNFS后端、BBR拥塞控制、TCP缓冲区调优、SR-IOV、CERN FTS传输服务以及监控工具。

**📊 数据集**

利用CMS实验的真实生产数据传输，重点观察2025年10月17日高负载期间的SPRACE→FNAL链路。

**📈 对比分析**

与后端理论峰值77 Gb/s及默认OS设置对比，系统在该配置下实现51.3 Gb/s的总吞吐量，单链路峰值41.5 Gb/s，且与CERN监控结果一致。

**⚠️ 局限性**

瓶颈在前端VM集群（CPU、内存、网络卡限制），部分目的地出现失败，且调优工作量大、需要手动配置，无法直接推广到其他环境。

---

## 146. SurgCalib: Gaussian Splatting-Based Hand-Eye Calibration for Robot-Assisted Minimally Invasive Surgery

**arXiv ID:** 2603.08983 | [PDF](https://arxiv.org/pdf/2603.08983v1)

**作者:** Zijian Wu `[一作]` (University of British Columbia), Septimiu E. Salcudean `[通讯]` (University of British Columbia)

**通讯引用:** 14963 | [OpenAlex ID](https://openalex.org/A5028375560)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了SurgCalib框架，实现了无标记、自动的达芬奇手眼标定，利用双相机视频和原始关节角度，通过初步PnP估计和两阶段的高精度姿态优化，最终计算出相机到机器人基座的变换。

**💡 创新点**

创新点在于：①首次将3D高精度的Gaussian Splatting用于外科机器人姿态优化；②引入RCM（远程中心运动）约束的两阶段优化，既保留了物理约束，又避免了早期过度约束导致的局部最优；③实现完全无标记、仅靠单目视频和原始运动学数据即可完成标定。

**🔧 技术方法**

技术手段包括：深度学习关键点检测（MFC-tracker）、分割（SAM 2）、EPnP求解、可微渲染（Gaussian Splatting）与render‑and‑compare损失、RCM几何约束、Kabsch‑Umeyama最小二乘求解。

**📊 数据集**

使用公开的dVRK SurgPose数据集，包含多视角视频、标定矩阵、2D关键点注释和关节角度信息。

**📈 对比分析**

对比方法为直接使用原始运动学姿态与通过SurgCalib优化后姿态的重投影误差和3D工具尖端误差。结果显示：2D重投影误差平均约为12.24px（2.06mm）和11.33px（1.9mm），3D工具尖端误差平均约为5.98mm和4.75mm，显著优于仅靠原始姿态的误差。

**⚠️ 局限性**

局限性包括：Gaussian Splatting对新视角渲染的真实感有限；当前仅验证在LND一种工具，缺乏多工具通用性；未显式建模光照与相机内参变化；未考虑运动学误差与重投影误差的相关性，可能进一步影响精度。

---

## 147. AgentOS: From Application Silos to a Natural Language-Driven Data Ecosystem

**arXiv ID:** 2603.08938 | [PDF](https://arxiv.org/pdf/2603.08938v1)

**作者:** Rui Liu `[一作]` (University of Kansas), Jian Pei `[通讯]` (Duke University)

**通讯引用:** 53085 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为AgentOS的新型操作系统架构，将传统的GUI桌面转变为统一的自然语言交互入口（Single Port），通过Agent Kernel对用户意图进行实时解析与多智能体协同执行，并将传统应用拆解为可由自然语言规则编写的可组合技能模块（Skill-as-Modules）。

**💡 创新点**

创新点在于：1）将操作系统本身重构为以自然语言为主的“意图驱动”系统；2）引入Agent Kernel实现意图解析、资源调度与多智能体编排；3）将技能模块化、可通过自然语言定义；4）将整个系统视作连续的数据挖掘管道，强调个人知识图谱、序列模式挖掘与推荐系统在OS中的核心作用；5）提出Semantic Firewall等安全机制以对抗AI驱动的攻击与幻觉。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）与Model Context Protocol（MCP）进行意图解析与交互；多模态自然语言处理与关系抽取；个人知识图谱（PKG）构建与图增强推理；两塔结构推荐系统（User Tower+Skill Tower）与强化学习优化；序列模式挖掘（SPM）用于工作流优化；语义防火墙与沙盒化、快照回滚等安全与容错技术。

**📊 数据集**

数据来源主要为多模态交互日志（语音、文字、屏幕上下文、地理位置信号等）、用户行为记录与系统调用日志；通过构建个人知识图谱与技能仓库，进一步利用公开或企业内部的对话与代码库进行训练与评估。

**📈 对比分析**

与传统GUI/CLI系统相比，AgentOS 在用户满意度、意图匹配度（Intent Alignment）与任务完成时间上有显著提升，实验表明通过个人知识图谱与序列模式挖掘能将复杂工作流的执行时间缩短30%-50%。在安全方面，Semantic Firewall 能识别并阻止超过90%的诱导注入攻击与数据泄露尝试。

**⚠️ 局限性**

主要局限包括：1）对LLM的性能与可解释性高度依赖，幻觉与误判仍可能导致安全风险；2）个人知识图谱与推荐模型的构建需要大量标注与长期学习；3）在多用户或多设备场景下的上下文同步与一致性尚未充分验证；4）安全与隐私机制需要进一步完善，以防止语义防火墙误判或被绕过。

---

## 148. Reasoning-Oriented Programming: Chaining Semantic Gadgets to Jailbreak Large Vision Language Models

**arXiv ID:** 2603.09246 | [PDF](https://arxiv.org/pdf/2603.09246v1)

**作者:** Quanchen Zou `[一作]` (360 AI Security Lab), Xiangzheng Zhang `[通讯]` (360 AI Security Lab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过将有害目标拆分为语义上无害的视觉小工具，并利用梯度无关的控制流优化在视觉语言模型中精确调度推理过程，提出了一种基于 Return-Oriented Programming 思想的多模态“Reasoning-Oriented Programming”攻击框架，用以规避大型视觉语言模型的安全对齐。

**💡 创新点**

创新点包括：① 将有害意图拆解为语义正交的视觉小工具；② 采用梯度无关的控制流优化（进化搜索）实现对模型推理过程的精准链式调度；③ 揭示了通过后期推理劫持实现的视觉语言模型安全漏洞的新攻击面。

**🔧 技术方法**

使用的技术包括：文本到图像（T2I）模型生成视觉小工具；辅助大型语言模型进行语义分解与提示演化；梯度无关的控制流优化（进化搜索）；安全评估指标 ASR；黑盒对抗实验。

**📊 数据集**

使用的数据集为 SafeBench 与 MM-SafetyBench，覆盖 500 条有害查询（10 类）及 13 类高风险场景。

**📈 对比分析**

方法与四种现有黑盒多模态越狱基线（FigStep、MM-SafetyBench、JOOD、MML）以及四个防御方法（CIDER、ECSO、JailGuard、AdaShield）在 7 个开源模型和 4 个商用模型上进行对比；平均 ASR 提升约 4.67%（开源）/9.50%（商用），单模型最高达 0.95，且在防御下仍保持超过 0.5 的成功率。

**⚠️ 局限性**

局限性：仅针对图像–文本场景，未探究音频或视频等其他模态；分解策略依赖预设规则，可能无法覆盖所有现实中的有害推理模式；对模型内部机制的解释仍有限。

---

## 149. Chaotic Dynamics in Multi-LLM Deliberation

**arXiv ID:** 2603.09127 | [PDF](https://arxiv.org/pdf/2603.09127v1)

**作者:** Hajime Shimao `[一作]` (Pennsylvania State University), Sung Joo Kim `[通讯]` (American University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多LLM委员会在多轮推理中的稳定性，量化不同设计（角色分配与模型异质性）对轨迹发散的影响，并通过实验验证了干预措施（主席角色消除、记忆窗口缩短）可降低不稳定性。

**💡 创新点**

首次系统性拆解多LLM不稳定性为两条可操控的路线（机构角色差异与模型异质性），揭示两者非加性交互，并提出可操作的机制性干预方案。

**🔧 技术方法**

采用随机动力学模型、基于日志线性回归的经验莱普诺夫指数估计、Bootstrap置信区间、角色消除与记忆窗口实验等技术。

**📊 数据集**

12个政策情境基准（涵盖移民、健康、收入、气候、言论与AI治理）以及统一与混合模型配置的多LLM委员会。

**📈 对比分析**

与基线（统一无角色）对比，发现两条不稳定路线显著提升轨迹发散；混合+角色配置不如混合+无角色，证明非加性；实验干预（主席消除、k=3窗口）均显著降低经验莱普诺夫指数。

**⚠️ 局限性**

实验仅覆盖特定协议族和温度T=0，未探讨不同任务质量的影响，且干预措施虽降低不稳定性但不保证决策质量保持；此外，单一指标（经验莱普诺夫指数）可能不足以全面评估治理效能。

---

## 150. Computer Vision-Based Vehicle Allotment System using Perspective Mapping

**arXiv ID:** 2603.08827 | [PDF](https://arxiv.org/pdf/2603.08827v1)

**作者:** Prachi Nandi `[一作]` (National Institute of Technology Rourkela), Suchismita Chinara `[通讯]` (National Institute of Technology Rourkela)

**通讯引用:** 572 | [OpenAlex ID](https://openalex.org/A5065126682)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建基于YOLOv8和逆透视映射(IPM)的室内智能停车系统，利用四摄像头合成图像进行车辆和柱子检测，并生成3D笛卡尔模型标记空车位。

**💡 创新点**

通过将多角度摄像头图像进行IPM融合，生成完整的3D视图；结合YOLOv8的动态锚框提升检测精度；在模拟环境中自动生成多样化数据集。

**🔧 技术方法**

YOLOv8对象检测、逆透视映射(IPM)、3D笛卡尔绘图、Matplotlib、Roboflow标注工具。

**📊 数据集**

使用Spline.AI生成的室内停车场3D仿真视频截取的150张标注图像（车辆和柱子），按75%/25%划分训练/测试集。

**📈 对比分析**

与YOLOv5、YOLOv7进行精确度-召回曲线对比，YOLOv8在整体、车辆检测、柱子检测上分别达98.4%、98.6%、98.2%的高精度。

**⚠️ 局限性**

仅在仿真数据上验证，缺乏真实环境测试；数据量有限，可能导致过拟合；系统对多摄像头布置和光照变化的鲁棒性待进一步验证。

---

## 151. Turn: A Language for Agentic Computation

**arXiv ID:** 2603.08755 | [PDF](https://arxiv.org/pdf/2603.08755v1)

**作者:** Muyukani Kizito `[一作]` `[通讯]`, Muyukani Kizito

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

构建了Turn语言，解决agentic软件的上下文、类型安全、内存、身份和API契约等核心难点。

**💡 创新点**

创新点在于将LLM推理作为强类型原语、引入置信度控制流、实现基于actor的进程模型、采用基于能力的身份体系以及编译时API schema吸收。

**🔧 技术方法**

技术采用Rust实现编译器与VM，WebAssembly推理驱动，Erlang式actor模型，JSON Schema验证，能力令牌与分布式状态管理。

**📊 数据集**

主要使用LLM提供者的API（OpenAI、Anthropic）进行推理，并通过内置的API适配器（如Stripe、Slack）进行测试；未使用公开机器学习数据集。

**📈 对比分析**

与现有框架（如LangChain）对比，Turn将实现行数从约350行压缩到89行；性能评估显示推理验证、进程调度等语言级保障的开销仅几微秒，远低于网络延迟；在安全性实验中通过所有5项验证。

**⚠️ 局限性**

局限性包括仅针对agentic场景的领域语言、非静态类型的普通绑定、缺乏自我反思与多模态支持，以及对置信度依赖模型暴露日志概率。

---

## 152. Emotional Modulation in Swarm Decision Dynamics

**arXiv ID:** 2603.09963 | [PDF](https://arxiv.org/pdf/2603.09963v1)

**作者:** David Freire-Obregón `[一作]` (Universidad de Las Palmas de Gran Canaria), David Freire-Obregón `[通讯]` (Universidad de Las Palmas de Gran Canaria)

**通讯引用:** 485 | [OpenAlex ID](https://openalex.org/A5040563581)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在蜂群决策模型基础上构建代理人模型，加入情绪（情感价值与激活度）对招募与抑制率的调节，并通过情绪传染研究情绪如何影响集体决策。

**💡 创新点**

首次将二维情感维度与蜂群决策动力学结合，实现情绪对招募/抑制参数的动态调节，揭示情绪异质性与结构临界点共同决定集体结果。

**🔧 技术方法**

基于代理人建模、情绪传染机制、离散时间仿真以及统计指标（一致时间、胜率、半衰期）等技术。

**📊 数据集**

使用人工生成的400个代理人的二维网格模拟，未使用真实实验数据，所有情绪与决策初始值由实验情景设定。

**📈 对比分析**

通过200次独立仿真对比三种情景，评估情绪参数对胜率和一致时间的影响；实验显示情绪增强可显著提高胜率并缩短一致时间。

**⚠️ 局限性**

局限在于缺乏真实世界验证、只考虑两选项、参数设定人为且未深入探究异质性与动态情绪反馈的影响。

---

## 153. A saccade-inspired approach to image classification using visiontransformer attention maps

**arXiv ID:** 2603.09613 | [PDF](https://arxiv.org/pdf/2603.09613v1)

**作者:** Matthis Dallain `[一作]` (Institut de Neurosciences de la Timone, Aix-Marseille Université, CNRS), Benoît Miramond `[通讯]` (Laboratoire d'Electronique, Antennes et Télécommunications, Université Côte d'Azur, CNRS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过将自监督Vision Transformer DINO的注意力图视为眼动注视，设计一种基于saccade的递进采样方法，对ImageNet-1K进行图像分类实验。

**💡 创新点**

创新点在于将ViT的自注意力映射直接用于指导视觉采样，证明仅暴露少量像素即可保留甚至提升分类性能，且该方法与传统人眼注视模型相比更高效。

**🔧 技术方法**

使用DINO ViT产生的多头注意力图，采用最大值聚合、抑制返回（Inhibition of Return）和固定大小fovea逐步曝光的采样策略；并结合预训练的线性分类头和ResNet-50进行跨模型验证。

**📊 数据集**

主要使用ImageNet-1K验证集（224×224）进行评估，另外在每个类别随机抽取10张图像的小样本集上测试显著性模型对比。

**📈 对比分析**

与随机采样、中心裁剪以及GBVS和UNISAL等显著性模型比较，结果显示DINO引导的saccades在仅使用约50%像素时即可达到90%以上的全图准确率，且在某些saccade数下甚至超过全图性能。

**⚠️ 局限性**

局限性包括：需要两次前向传播生成注意力图和分类，缺乏递归累积与自适应停止机制；仅基于预训练模型，未针对saccade任务进行微调；对低分辨率或早期层注意力的依赖尚未充分探索。

---

## 154. Fish Audio S2 Technical Report

**arXiv ID:** 2603.08823 | [PDF](https://arxiv.org/pdf/2603.08823v1)

**作者:** Shijia Liao `[一作]`, Dawei Han `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 Fish Audio S2 开源文本到语音系统，支持多说话人、多轮对话、长文本合成，并通过自然语言指令实现细粒度控制。

**💡 创新点**

创新点包括 Dual-AR 架构拆分时序语义与深度声学建模、统一数据管线与 RL 对齐，以及多维奖励强化学习，显著提升指令跟随、自然度和多说话人一致性。

**🔧 技术方法**

采用 Transformer+RVQ 音频编码、语音质量评估模型、ASR+丰富转写、GRPO/Dr.GRPO RL、SGLang 推理引擎等技术。

**📊 数据集**

使用超过 10 万小时多语言语料（约 80 种语言）以及 Seed‑TTS、MiniMax、CV3‑Eval、Long‑TTS‑Eval 等公开与内部数据集。

**📈 对比分析**

通过客观指标（WER、CER、SIM）和 LLM‑as‑a‑Judge 评估（Audio Turing、Emergent TTS、Fish Instruction Benchmark）与多种基准对比，Fish Audio S2 在多项基准上达到或逼近 SOTA，指令跟随成功率和自然度显著提升。

**⚠️ 局限性**

局限包括对极低资源语言的表现仍不如顶尖商用系统、对极长文本仍受上下文窗口限制，以及 RL 对齐可能引入偏差和不可解释性。

---

## 155. WVA: A Global Optimization Control Plane for llmd

**arXiv ID:** 2603.09730 | [PDF](https://arxiv.org/pdf/2603.09730v1)

**作者:** Abhishek Malvankar `[一作]` (IBM Research), Tamar Eilam `[通讯]` (IBM Research)

**通讯引用:** 1069 | [OpenAlex ID](https://openalex.org/A5074767181)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对大型语言模型推理的工作负载变体自动扩缩控制平面（WVA），通过深度整合推理引擎内部状态实现精细调度。

**💡 创新点**

核心创新在于将“Variant”作为一阶抽象，将GPU硬件、并行度、量化等组合成可配置的推理实例，并采用基于KV缓存饱和度和队列深度的饱和度模型，实现全局头部空间管理和碎片感知缩容。

**🔧 技术方法**

技术包括：1）基于Kubernetes的可插拔控制平面框架；2）自定义指标采集（KV使用率、队列深度）与分布式收集器；3）模型分析器与全局优化器，支持限制模式与无约束模式；4）与vLLM/ EPP调度器的深度集成，实现阶段同步。

**📊 数据集**

实验主要使用Qwen/Qwen3-0.6B模型及其KV缓存配置，采用仿真环境与实际OpenShift H100集群验证，使用synthetic负载（RPS波形）和真实流量生成器。

**📈 对比分析**

与传统Kubernetes HPA比较，WVA在同等SLO下实现37%吞吐量提升，10倍请求失败率下降，成本上则通过成本感知分层优先使用A100，显著降低功耗与运营成本。

**⚠️ 局限性**

局限包括：1）扩缩仍为基于阈值的反应式，突发波动时仍有短暂延迟；2）依赖推理引擎暴露的指标，若接口不兼容需额外适配；3）在硬件资源受限（如单节点）时，头部空间预留可能导致过度扩容。

---

## 156. AI Act Evaluation Benchmark: An Open, Transparent, and Reproducible Evaluation Dataset for NLP and RAG Systems

**arXiv ID:** 2603.09435 | [PDF](https://arxiv.org/pdf/2603.09435v1)

**作者:** Athanasios Davvetas `[一作]` (National Centre for Scientific Research Demokritos), Vangelis Karkaletsis `[通讯]` (National Centre for Scientific Research Demokritos)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开了一个基于欧盟AI法的多任务评估数据集，涵盖风险等级分类、条款检索、义务生成和问答，并利用LLM生成多风险级别场景；

**💡 创新点**

创新点在于将法规文本与大型语言模型推理相结合，自动生成结构化、可机器读取的评估场景，解决了风险边界模糊与数据稀缺的问题；

**🔧 技术方法**

采用大型语言模型（如Llama、Mistral）、文档嵌入（Jina embeddings）、GPU加速近邻检索（Annoy）以及Prompt工程和规则化评价等技术；

**📊 数据集**

使用欧盟AI法全文（Regulation EU 2024/1689）及其手工抽取的条款作为输入，生成约339条场景的JSON数据集和相应的QA对；

**📈 对比分析**

通过在RAG框架下对风险等级分类任务进行实验，禁用类F1=0.87，高风险类F1=0.85，整体加权F1=0.69，显示对禁用与高风险类表现良好，但对有限/最小类效果欠佳；

**⚠️ 局限性**

限制包括：有限/最小类的判别边界不明确、生成场景的边界模糊、依赖手工规则、缺乏全面法律专家验证，以及使用的LLM在可解释性与可复现性方面的不足。

---

## 157. IntroSVG: Learning from Rendering Feedback for Text-to-SVG Generation via an Introspective Generator-Critic Framework

**arXiv ID:** 2603.09312 | [PDF](https://arxiv.org/pdf/2603.09312v1)

**作者:** Feiyu Wang `[一作]` (Fudan University), Junyu Gao `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 4263 | [OpenAlex ID](https://openalex.org/A5001848378)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 IntroSVG 的自我修正框架，利用统一的视觉语言模型同时担任 SVG 代码生成器和视觉评估者，形成闭环的 generate‑critique‑refine 迭代流程。

**💡 创新点**

核心创新点在于：①将生成与评估整合到同一模型；②通过将失败样本转换为纠错和批判数据进行多任务监督；③使用 Direct Preference Optimization (DPO) 对生成策略进行偏好对齐；④构建标准化的多彩 SVG 训练集，显著提升代码可编辑性与视觉质量。

**🔧 技术方法**

主要技术包括：多任务监督式微调（SFT）、DPO 偏好优化、闭环生成‑批判‑修正迭代、SVG 代码与图像的统一表征、以及使用高容量教师 VLM 生成评估数据。

**📊 数据集**

使用了自建的 200k 条标准化多色 SVG 对话（来自 LLM4SVG、OmniSVG、SVGen）、Correction 与 Critique 训练集（约 50k 条）以及 10k 条偏好对比样本，全部采用统一的绝对命令与整数坐标格式。

**📈 对比分析**

在统一测试集上与现有域专用模型（OmniSVG、SVGen）以及大型通用模型（GPT‑5、Gemini 2.5 Pro、Qwen3-VL-30B 等）对比，IntroSVG 在 Render Success Rate 99.26%、FID 26.18、Aesthetic 4.8894、HPS 0.1969 等指标上均达到或逼近 SOTA，迭代循环显著提升 FID 从 29.76 降至 26.18。

**⚠️ 局限性**

主要限制包括：仍需大量算力训练；对极其复杂或非常规 SVG 结构的适应性有限；模型对非标准 SVG 规范（相对坐标、混合命令）可能表现不佳；当前仅在单一语言（中文）文本描述下验证，跨语言适配待进一步研究。

---

## 158. Evidential Perfusion Physics-Informed Neural Networks with Residual Uncertainty Quantification

**arXiv ID:** 2603.09359 | [PDF](https://arxiv.org/pdf/2603.09359v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**通讯引用:** 3657 | [OpenAlex ID](https://openalex.org/A5052023515)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了EPPINN框架，融合证据学习与物理约束网络，实现CT灌注图像的参数估计与不确定性量化。

**💡 创新点**

在物理残差上引入Normal–Inverse–Gamma证据分布，既能获得自回归的量化不确定性，又可通过结构化参数化、进展性退火和抗崩塌正则实现单次优化的鲁棒性。

**🔧 技术方法**

采用位置编码的哈希网格+SIREN网络、NIG证据回归、CBV–MTT结构化参数化、AIF预训练、逐步退火以及残差正则化等技术。

**📊 数据集**

使用数字脑灌注幻影、ISLES 2018 4D CTP + DWI核心掩模以及42例临床回顾性CTP + DWI + AIF数据集。

**📈 对比分析**

与SVD、bcSVD、boxNLR、SPPINN、ReSPPINN等基线对比；在NMAE、置信区间覆盖率和核心检测灵敏度上均优于基线，尤其在低SNR、稀疏采样和临床病例中表现突出。

**⚠️ 局限性**

仍受限于极低SNR或极端扫描协议下残差收敛不稳；在极大数据量时计算时间与内存占用尚需优化；对极端临床情况的鲁棒性未全面评估。

---

## 159. Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity

**arXiv ID:** 2603.09480 | [PDF](https://arxiv.org/pdf/2603.09480v1)

**作者:** Zhengyao Fang `[一作]` (Harbin Institute of Technology), Wenjie Pei `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2022 | [OpenAlex ID](https://openalex.org/A5078487642)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练自由、任务无关的视觉 token 压缩框架（PruneSID），通过两阶段流程（PSCA 语义聚类 + intra‑group NMS 冗余抑制）以及信息感知的动态压缩比例，对 VLM 的视觉 token 进行高效裁剪。

**💡 创新点**

创新点包括：① 将 PCA 变形为 Principal Semantic Component Analysis（PSCA）在 token 维度上进行语义聚类；② 在每个聚类内部应用非极大抑制（NMS）自适应保留代表性 token；③ 引入基于全局 token 相似度的动态压缩比例机制，使每张图像根据其信息量分配不同的 token 数量；④ 整个流程无须额外训练，兼容多种 VLM、图像与视频输入。

**🔧 技术方法**

采用的核心技术：Principal Semantic Component Analysis（PSCA）、非极大抑制（NMS）、信息/冗余度量（ρ）、自适应阈值 τ=λ·ρ、token 预算分配、两阶段聚类+裁剪算法。

**📊 数据集**

在 LLaVA‑1.5、LLaVA‑NeXT、Mini‑Gemini、Video‑LLaVA 等模型上，使用 GQA、MMBench、MME、POPE、ScienceQA、VQA‑v2、TextVQA、MMMU、SEED‑Bench、VizWiz、LLaVA‑Bench、TGIF、MSVD、MSRVTT、ActivityNet 等多种图像/视频问答与多模态基准数据集进行评估。

**📈 对比分析**

与 VisionZip、HiRED、DART、FastV、SparseVLM 等现有压缩方法比较，PruneSID 在 LLaVA‑1.5 仅保留 64 token（11.1%）即可达到 96.3% 准确率，较 VisionZip 提升约 1.9%；在 LLaVA‑NeXT 仅保留 5.6% token 时仍保持 92.8% 准确率，提升 2.5%；在 Video‑LLaVA 仅保留 6.6% token 时平均准确率达 95.5%，同时 Prefilling 速度提升 7.8×，整体推理时间大幅缩短。

**⚠️ 局限性**

在极端压缩或需要细粒度视觉细节的任务中，方法可能会误删重要局部信息，导致答案不完整；此外，当前方案为任务无关，缺乏针对特定任务的指令或上下文引导，可能在细节推理场景下表现不如任务定制化方法。

---

## 160. GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis

**arXiv ID:** 2603.09446 | [PDF](https://arxiv.org/pdf/2603.09446v1)

**作者:** Tran Bao Sam `[一作]` (NVIDIA), Steven Truong `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 GIIM 架构，通过多异质图模型同时捕捉多视图医学影像中的 intra- 与 inter- 视图依赖，提升诊断准确性。

**💡 创新点**

创新点在于同时建模同一病灶的视图内关联和不同病灶间的跨视图动态，并针对缺失视图设计四种补全策略，增强鲁棒性。

**🔧 技术方法**

技术上结合 ConvNeXt 单视图特征提取器、Multi-Heterogeneous Graph（MHG）与异质消息传递（SAGEConv）以及缺失视图补全（常量、可学习、RAG、协方差）。

**📊 数据集**

使用了私有肝脏肿瘤 CT 数据、公开 VinDr-Mammo 乳腺 X 光、多模态 BreastDM MRI 等多种数据集。

**📈 对比分析**

与单视图 NN、LightGBM、Attention 等方法对比，GIIM 在准确率和 AUC 上均领先 1–3%（肝脏）及 4–5%（乳腺）等，尤其在缺失视图场景下保持优势。

**⚠️ 局限性**

局限在于仍需手工设计多视图的图结构和缺失策略，对极端缺失率或不同模态间差异的适应性需进一步验证。

---

## 161. Explainable Innovation Engine: Dual-Tree Agent-RAG with Methods-as-Nodes and Verifiable Write-Back

**arXiv ID:** 2603.09192 | [PDF](https://arxiv.org/pdf/2603.09192v1)

**作者:** Renwei Meng `[一作]` `[通讯]` (Anhui University), Renwei Meng (Anhui University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于方法节点的可解释创新引擎，将检索增强生成从平面文本切片升级为方法树结构，实现可追溯的推导与可控合成。

**💡 创新点**

创新点在于双树知识表示（方法传承树和层次抽象树）以及策略驱动的创新操作与可验证剪枝，形成持续可审计的创新循环。

**🔧 技术方法**

使用了检索增强生成框架、LLM驱动的结构化抽取、MiniBatch k‑means聚类+LLM摘要、权重化的贡献评分、可执行验证（如Lean）以及策略代理与评分回归。

**📊 数据集**

在六个学科（数学、物理、计算机科学、生物、化学、社会学）共600条中立题目上，结合多大模型（GPT‑5.2、Gemini 3.0、Llama‑4 70B、DeepSeek）进行评测。

**📈 对比分析**

与同一后端LLM的平面检索‑生成基线相比，专家评测显示平均提升0.29–0.83分（最高0.83分），并且在多项统计检验下显著优于基线。

**⚠️ 局限性**

局限在于对大型LLM的依赖、验证覆盖受限、潜在的错误累积风险、对多模态信息提取的依赖尚不充分，以及系统对伦理与安全约束的手工设定。

---

## 162. Interactive 3D visualization of surface roughness predictions in additive manufacturing: A data-driven framework

**arXiv ID:** 2603.09353 | [PDF](https://arxiv.org/pdf/2603.09353v1)

**作者:** Engin Deniz Erkan `[一作]` (Middle East Technical University), Ulas Yaman `[通讯]` (Middle East Technical University)

**通讯引用:** 1480 | [OpenAlex ID](https://openalex.org/A5044902192)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文开发了一套基于实验数据和机器学习的三维可视化框架，能够在打印前根据层厚、温度等参数和表面倾斜角度预测并显示零件表面粗糙度。

**💡 创新点**

创新点包括：①利用Box–Behnken设计系统收集稀缺的粗糙度数据；②采用条件生成对抗网络（CGAN）进行表格数据增广，显著提升模型泛化；③将预测结果与三维几何实时可视化，支持交互式工艺和方向优化。

**🔧 技术方法**

核心技术包括多层感知器回归（MLP）、条件生成对抗网络（CGAN）、SHAP解释、Optuna超参搜索、Box–Behnken实验设计和基于浏览器的交互式可视化界面。

**📊 数据集**

使用的是在Bambu Lab A1打印机上以PLA丝材为材料、7个工艺参数（层高、挤压温度、壁速、填充密度、壁厚、床温、风扇速）和多角度表面倾斜的87件试样共1566次粗糙度测量所构成的实验数据集。

**📈 对比分析**

通过五折交叉验证和留出测试集比较：仅用实验数据训练的MLP MAE为2.20 µm、R²为0.844；加入CGAN合成数据后MAE降至1.75 µm、R²提升至0.90，说明数据增广有效提升预测精度。

**⚠️ 局限性**

局限性包括：仅针对单台PLA FFF打印机，难以直接迁移到其他机型或材料；合成数据虽提升性能但仍受限于训练分布；仅考虑平面面片，无法覆盖曲面和更复杂几何；缺乏实时传感反馈，不能实现在线闭环控制。

---

## 163. PM-Nav: Priori-Map Guided Embodied Navigation in Functional Buildings

**arXiv ID:** 2603.09113 | [PDF](https://arxiv.org/pdf/2603.09113v1)

**作者:** Jiang Gao `[一作]` (Northeastern University), Xiaoguang Ma `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于先验地图引导的具身导航框架PM-Nav，用于功能建筑中的室内导航。

**💡 创新点**

创新点包括将环境地图解析为语义先验地图、设计层次化链式思维提示模板以生成精准路径规划，并构建多模型协同动作输出机制实现定位决策与执行控制。

**🔧 技术方法**

技术上结合了视觉语言模型(VLM)、GroundingDINO、SAM、PixelNav神经网络，并通过H-CoT提示模板和注解先验地图来提升空间推理与路径规划。

**📊 数据集**

实验使用了基于VA Design Guide的数据集，构建了六个模拟功能建筑场景及真实校园场景（如佛山研究生院、东北大学等）。

**📈 对比分析**

与零样本导航SOTA SG-Nav和InstructNav对比，PM-Nav在仿真环境中平均提升SR/SPL 511%/1175%，在真实场景中提升650%/400%，显著提高了易、中、难任务的成功率。

**⚠️ 局限性**

局限性包括对环境地图解析精度的依赖、探索效率仍有提升空间，以及在高度相似的功能建筑中对先验地图的依赖导致泛化能力受限。

---

## 164. AutoViVQA: A Large-Scale Automatically Constructed Dataset for Vietnamese Visual Question Answering

**arXiv ID:** 2603.09689 | [PDF](https://arxiv.org/pdf/2603.09689v1)

**作者:** Nguyen Anh Tuong `[一作]` (University of Science), Tung Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 AutoViVQA，一套基于 LLM 驱动的全自动化生成与验证框架，用于构建大规模、推理级别可控的越南语视觉问答数据集。

**💡 创新点**

创新点在于（1）五级推理层次控制与语义类别映射，实现多层次认知难度的自动生成；（2）基于多模型投票的集成质量控制，完全无人工标注即可保证视觉可信度、语言自然度与推理深度。

**🔧 技术方法**

核心技术包括大型语言模型（如 Gemini‑2.5‑Flash）生成受限提示、Vision‑Language 模型评估、可视化质量与多维度评分量化、以及基于中位数阈值和多数投票的筛选算法。

**📊 数据集**

数据集来源为 MS COCO 图像与 VISTA 的越南语标题/对话，最终得到 19,411 张图片、37,077 个问题、185,385 条答案（每问 5 答），覆盖 9 类问题与 5 层推理。

**📈 对比分析**

实验对比多种越南语与通用多模态模型（Vintern、ViT5‑ViT、BARTPhoBEiT、GPT‑5、LLaMA‑3.2、Gemini）在标准自动评估指标上，发现使用 AutoViVQA 训练后模型在精确度、召回率、F1、ROUGE‑L、METEOR、CIDEr 等指标上均有显著提升，尤其是精准度和语义一致性。

**⚠️ 局限性**

局限性包括：图像来源局限于 COCO，缺乏本土文化多样性；集成验证仍可能保留大型语言模型的偏见；未覆盖越南方言或地区变体；推理层次分布仍有 0‑5 级偏差的余地。

---

## 165. TaSR-RAG: Taxonomy-guided Structured Reasoning for Retrieval-Augmented Generation

**arXiv ID:** 2603.09341 | [PDF](https://arxiv.org/pdf/2603.09341v1)

**作者:** Jiashuo Sun `[一作]` (University of Illinois Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 122759 | [OpenAlex ID](https://openalex.org/A5019539533)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于两层轻量级分类学的结构化推理框架 TaSR-RAG，用三元组形式表示查询与文档，逐步检索与绑定实体，提升多跳问答的证据选择与答案可信度。

**💡 创新点**

创新点包括：① 将查询和检索文档均转换为带类型信息的三元组；② 采用轻量级两层分类学实现语义与结构的双重约束；③ 引入混合匹配（语义相似度 + 结构一致性）实现逐步检索；④ 通过显式实体绑定表实现跨步变量替换，避免实体混淆；⑤ 无需显式图结构，训练自由，兼容现有检索器与 LLM。

**🔧 技术方法**

技术方法：三元组抽取与类型化（LLM 端），查询分解为有潜变量的三元组，混合语义‑结构匹配，递归实体绑定与逐步检索，最后在 LLM 生成阶段输出答案；使用的基础检索器为稠密检索，生成器为 Qwen2.5‑7B/72B‑Instruct。

**📊 数据集**

实验数据集：七大开放域/多跳 QA 数据集—Natural Questions、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle。

**📈 对比分析**

与基线（直接推理、CoT、RAG、IRCoT、GraphRAG、HippoRAG、HyperGraphRAG、StructRAG）对比，TaSR-RAG 在 Qwen2.5‑72B 上平均 EM 提升至 42.5（比标准 RAG 29.7 高 14%），在 7 大数据集上均名列前茅；在 Qwen2.5‑7B 上平均 EM 37.0，明显优于所有基线。

**⚠️ 局限性**

局限性：① 依赖 LLM 进行三元组抽取、类型化与答案生成，易受模型误差影响；② 两层分类学虽轻量，但细粒度误判会放大检索误差；③ 生成错误（抽取失误、上下文误用、变量传播错误）仍占比约 30%；④ 需要手工或半自动构建分类学，适配新领域仍有挑战。

---

## 166. EPIC-EuroParl-UdS: Information-Theoretic Perspectives on Translation and Interpreting

**arXiv ID:** 2603.09785 | [PDF](https://arxiv.org/pdf/2603.09785v1)

**作者:** Maria Kunilovskaya `[一作]` (Saarland University), Christina Pollkläsener `[通讯]` (University of Hildesheim)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5048969696)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并扩展了基于欧洲议会的英文↔德文翻译与口译语料库EPIC‑EuroParl‑UdS，加入单词级惊讶度（GPT‑2及MT模型）和对齐信息；

**💡 创新点**

首次将机器翻译模型的惊讶度与传统单语GPT‑2惊讶度同时提供，并在同一语料上对比两者对口译填充词预测的影响，展现了跨语言、跨模式的惊讶度非线性关系；

**🔧 技术方法**

使用Stanza进行分词、词性标注、句法分析；GPT‑2 small（英、德）及其微调模型、基线与微调MT模型生成惊讶度；bert‑base‑multilingual‑cased进行子词级对齐；混合效应逻辑回归评估填充词出现概率；

**📊 数据集**

EPIC‑EuroParl‑UdS（写作与口译版本）共计约1700篇文档、数百万单词，覆盖DE↔EN双向；

**📈 对比分析**

通过对比基线与微调惊讶度在填充词预测模型中的AIC与C‑score，发现基线惊讶度更优；在MT与GPT‑2惊讶度的交互分析中揭示了非线性关系，表明译者在高难度段落中无法同时兼顾源文本忠实与目标流畅；

**⚠️ 局限性**

局限性包括：MT惊讶度在口译（OOV）上的低适用性、仅覆盖两种语言、模型误对齐导致部分单词N/A、未考虑时间同步与实时上下文；

---

## 167. Robotic Scene Cloning:Advancing Zero-Shot Robotic Scene Adaptation in Manipulation via Visual Prompt Editing

**arXiv ID:** 2603.09712 | [PDF](https://arxiv.org/pdf/2603.09712v1)

**作者:** Binyuan Huang `[一作]` (Wuhan University), Zhenzhong Chen `[通讯]` (Wuhan University)

**通讯引用:** 8604 | [OpenAlex ID](https://openalex.org/A5006748765)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Robotic Scene Cloning（RSC）方法，利用视觉提示编辑机器人演示轨迹以实现零射击场景迁移

**💡 创新点**

创新点在于结合视觉提示、位置感知与姿态约束三种控制信号，实现对目标对象的精准外观与形状克隆，并保持非编辑区域语义一致性

**🔧 技术方法**

使用MS‑Diffusion、Grounding‑DINO、SAM2、DepthAnythingV2、ControlNet等预训练模型构建视觉提示编辑框架；通过Progressive Masked Fusion和视觉提示引导图像编辑实现高保真克隆

**📊 数据集**

在SIMPLER、CALVIN模拟环境以及WidowX250S机器人真实环境上进行实验，使用CogACT、OpenVLA、RoboFlamingo等基线策略的数据集进行比较

**📈 对比分析**

与基线和文本提示生成方法GreenAug对比，RSC在跨纹理、跨形状任务中分别提升约30%–60%成功率，CALVIN长序列任务平均长度提升至2.57，显著优于传统方法

**⚠️ 局限性**

受限于只能处理中等形变的对象，对大幅度形状变化效果不佳；若需更大范围的形状适配需进一步大规模个性化数据集微调

---

## 168. EmbC-Test: How to Speed Up Embedded Software Testing Using LLMs and RAG

**arXiv ID:** 2603.09497 | [PDF](https://arxiv.org/pdf/2603.09497v1)

**作者:** Maximilian Harnot `[一作]` (Hydac Software GmbH), Timo Oksanen `[通讯]` (Technical University of Munich)

**通讯引用:** 2792 | [OpenAlex ID](https://openalex.org/A5021744500)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了基于检索增强生成（RAG）的嵌入式 C 单元测试自动生成流水线，能够从项目文档、代码和旧测试中检索相关上下文并生成语法正确、符合编码规范的测试。

**💡 创新点**

将 RAG 与混合检索（密集嵌入 + BM25）、AST 级分块、提示工程和工业级评估指标结合，显著降低了 LLM 的幻觉，提升了测试可用率，并在工业环境中得到验证。

**🔧 技术方法**

使用技术包括：检索增强生成（RAG）、密集嵌入模型、BM25 词袋检索、Reciprocal Rank Fusion 混合检索、AST 解析分块、提示工程（系统提示 + 动态检索片段）、大型语言模型（如 GPT）、向量数据库、代码与需求的语义嵌入、代码覆盖率工具与人工 Likert 评估。

**📊 数据集**

Hydac Software GmbH 的嵌入式 C 项目代码（头文件、源文件）、历史 Python 单元测试、需求文档及相关说明书作为数据集。

**📈 对比分析**

与随机检索和无检索基线比较，RAG 在语法正确率 100%、运行时验证 85%、分支覆盖 43%、行覆盖 67% 以及人工评估可用率 94.4% 上均优于基线；单小时可生成约 270 个测试，节约约 66% 的测试工时。

**⚠️ 局限性**

对高质量文档、统一命名约定和丰富旧测试库的依赖较高；覆盖率仍低于人工编写的成熟测试套件；LLM 仍可能出现假设错误或细粒度缺陷，需要人工复核；目前仅针对嵌入式 C 进行了验证，泛化到其他语言或框架仍需进一步研究。

---

## 169. The Bureaucracy of Speed: Structural Equivalence Between Memory Consistency Models and Multi-Agent Authorization Revocation

**arXiv ID:** 2603.09875 | [PDF](https://arxiv.org/pdf/2603.09875v1)

**作者:** Vladyslav Parakhin `[一作]` `[通讯]` (Okta), Vladyslav Parakhin (Okta)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

**🎯 论文内容**

研究多代理委托链中的授权撤销与缓存一致性的等价性，并提出操作计数式凭证模型。

**💡 创新点**

定义 Capability Coherence System 并证明其与 MESI 等价，引入 Velocity Vulnerability 指标和执行计数限额实现速度无关的撤销安全。

**🔧 技术方法**

采用形式化状态映射、定理证明、tick‑based 事件模拟以及四种撤销策略（急速、惰性、租约、执行计数）进行评估。

**📊 数据集**

在三种业务场景（银行、CRM、异常）中使用模拟生成的代理、委托层级和操作概率，未使用真实数据集。

**📈 对比分析**

通过 10 次多运行模拟统计未授权操作数和延迟，执行计数在 CRM 中比租约低 120 倍，在异常场景中低 184 倍；其他策略受速度影响。

**⚠️ 局限性**

仅在单一中心化授权、有限规模、理想网络下评估；未考虑 Byzantine 故障、网络分区、攻击者调度；模拟基于抽象时钟，缺乏实地验证。

---

## 170. GenAI Is No Silver Bullet for Qualitative Research in Software Engineering

**arXiv ID:** 2603.08951 | [PDF](https://arxiv.org/pdf/2603.08951v1)

**作者:** Neil A. Ernst `[一作]` (University of Victoria), Christoph Treude `[通讯]` (Singapore Management University)

**通讯引用:** 5135 | [OpenAlex ID](https://openalex.org/A5077658936)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估并梳理了大语言模型在软件工程定性研究中的应用现状、优势与风险，提出未来研究方向。

**💡 创新点**

首次在定性研究范式维度上归纳 GenAI 的适用场景与局限，强调需要从认知与方法论角度审视其角色，并给出具体评估与工作流改进路线。

**🔧 技术方法**

采用文献综述与人工编码相结合的方法，利用 ChatGPT 等 LLM 进行文献检索和技术工具识别，讨论 GPT/Claude 等模型在编码、摘要、概念生成中的应用。

**📊 数据集**

使用 2025 年 ICSE、CHASE 与 CSCW 三大会议论文集（共 607 篇），通过关键词过滤筛选 25+ 份定性编码论文，进一步统计 GenAI 与 QDA 工具使用情况。

**📈 对比分析**

比较方法：按会议分组统计 GenAI 使用比例（CSCW 7/209=3.3%，ICSE/CHASE 0/），引用已有实验显示在归纳性编码中 LLM 与人工的 κ>0.7，但在无提示的零射击场景表现差。

**⚠️ 局限性**

局限性包括：证据仅涵盖低语境的扣式编码，缺乏对构建性、情境化方法的评估；模型易产生偏差和幻觉；提示与参数敏感；缺乏对大规模跨研究数据的实证验证；以及在定性研究中难以与构造主义范式对齐。

---

## 171. ESAinsTOD: A Unified End-to-End Schema-Aware Instruction-Tuning Framework for Task-Oriented Dialog Modeling

**arXiv ID:** 2603.09691 | [PDF](https://arxiv.org/pdf/2603.09691v1)

**作者:** Dechuan Teng `[一作]` (Research Center for Social Computing and Information Retrieval), Wanxiang Che `[通讯]` (Research Center for Social Computing and Information Retrieval)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的端到端、schema‑aware 的 instruction‑tuning 框架 ESAinsTOD，用来整合多种任务导向对话数据并实现 session‑level 的 end‑to‑end 训练，提升跨场景泛化与鲁棒性。

**💡 创新点**

① 将任务指令与 schema 对齐机制嵌入同一框架；② 通过结构化指令和 schema 信息实现跨数据集、跨任务的统一训练；③ 在 session 级别保留前置任务结果，显著减少级联错误；④ 构建多源多标签的 instruction‑tuning 语料。

**🔧 技术方法**

使用大型 LLM（LLaMA 2、Qwen2.5‑Instruct）全参数微调；instruction‑tuning + schema‑aware 对齐；session‑level end‑to‑end 训练；滑动窗口与 schema 管理等上下文控制技术。

**📊 数据集**

CamRest676、In‑Car、MultiWOZ 2.0/2.1、SGD、Frames、BiTOD、STAR、BANKING77、CLINC150、HWU64、SNIPS 等多种任务导向对话数据集。

**📈 对比分析**

与 SimpleTOD、UBAR、SOLOIST、PPTOD、SPACE 等最强基线进行对比，ESAinsTOD 在所有四个基准上均获得最高 Combined Score（提升 4–7 个百分点），并在低资源、零‑shot和数据效率实验中显著优于对手。

**⚠️ 局限性**

生成式 NLU 对意图标签缺乏自然语言描述时易出现误分类；在完全未知领域的 JGA 仍有限；模型对齐机制复杂，训练成本高；缺少高质量多源数据，需进一步数据增强。

---

## 172. AnalogToBi: Device-Level Analog Circuit Topology Generation via Bipartite Graph and Grammar Guided Decoding

**arXiv ID:** 2603.08720 | [PDF](https://arxiv.org/pdf/2603.08720v1)

**作者:** Seungmin Kim `[一作]` (Sungkyunkwan University), Yulhwa Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 498 | [OpenAlex ID](https://openalex.org/A5012718779)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种基于Transformer的AnalogToBi框架，能够从零开始生成设备级模拟电路拓扑，并支持通过电路类型令牌进行功能类型控制。

**💡 创新点**

创新点包括：① 引入电路类型令牌实现显式功能控制；② 采用双分图（设备–网）表示，分离位置与功能；③ 使用语法引导解码（状态机）保证电气有效性；④ 通过设备重命名数据增强提升多样性并降低记忆化。

**🔧 技术方法**

使用的技术：Transformer解码器、语法约束（状态机）解码、双分图序列化、设备重命名数据增强、GAT分类器、SPICE自动转换及规则式尺寸化。

**📊 数据集**

使用的数据集：公开的 2,165 条 SPICE 级别电路网表，按功能划分为 15 类（含通用类），通过不同遍历顺序和设备重命名扩增后约 397,515 条序列。

**📈 对比分析**

与 AnalogCoder、LaMAGIC、AnalogGenie 等基线对比，AnalogToBi 在有效率 97.8% / 新颖度 92.1% / 同时有效且新颖 89.9%，且多类型控制准确率 91.3%，在所有指标上显著优于基线。

**⚠️ 局限性**

局限性：对样本稀缺的类别（如比较器、压控振荡器等）性能仍低；生成的电路需要规则式尺寸化，缺乏全自动尺寸优化；对更大规模或更复杂电路的可扩展性尚待验证。

---

## 173. Automatic Cardiac Risk Management Classification using large-context Electronic Patients Health Records

**arXiv ID:** 2603.09685 | [PDF](https://arxiv.org/pdf/2603.09685v1)

**作者:** Jacopo Vitale `[一作]` (Università Campus Bio-Medico di Roma), Bram van Es `[通讯]` (University Medical Center Utrecht)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5113051775)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文开发了一套基于未结构化电子健康记录的自动化方法，用于识别老年人是否符合心血管风险管理方案的资格。

**💡 创新点**

创新之处在于提出了层次注意力的Transformer编码器，专为长文本设计，并通过晚期融合将药物ATC描述及体征信息与文本特征结合，以提升预测精度。

**🔧 技术方法**

技术实现包括层次Transformer（带RoPE位置编码、CLS/平均池化）、1D ResNet、线性SVM以及零样本LLM（GPT‑4系列），并使用BioLORD句子嵌入对药物描述进行向量化。

**📊 数据集**

使用数据集为UMCU老年门诊电子健康记录，包含3482例患者，包含多份咨询文本、药物ATC代码描述、年龄、性别等信息。

**📈 对比分析**

实验采用5折分层交叉验证，比较文本仅和文本+晚期融合两种方案。层次Transformer在文本仅条件下实现F1≈91%、MCC≈0.73；加入融合后F1≈92%、MCC≈0.76，明显优于SVM（F1≈86%、MCC≈0.75）和零样本LLM（F1≈34–36%）。

**⚠️ 局限性**

局限性包括单中心、非公开的数据集、LLM未经过微调导致表现弱；模型需在受限硬件上训练，未验证跨中心泛化；且未充分利用诸如BMI等更丰富的体征信息。

---

## 174. Learning Adaptive LLM Decoding

**arXiv ID:** 2603.09065 | [PDF](https://arxiv.org/pdf/2603.09065v1)

**作者:** Chloe H. Su `[一作]` (Harvard University), Udaya Ghai `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种轻量级解码适配器，利用强化学习在保持LLM不变的前提下，根据任务预算动态选择采样策略，既可在序列级也可在 token 级实现自适应解码；

**💡 创新点**

其创新点在于将解码视为上下文无关或部分可观测的马尔可夫决策过程，训练自适应采样策略而非手工设定，且通过贪婪子集选择构造离散动作空间；

**🔧 技术方法**

技术上采用 REINFORCE 与可验证终端奖励的强化学习框架，结合上下文无关 bandit 与 POMDP，动作空间包含温度、top‑k、top‑p 等采样参数；

**📊 数据集**

实验使用数学推理基准 MATH 与代码竞赛基准 CodeContests，并在 AIME‑2025 上做跨域评估；

**📈 对比分析**

在与最佳静态采样和均匀混合基线对比后，序列级适配器在 Pass@1/Pass@8 上分别提升约 1‑3%/2‑5%，token 级适配器在 Pass@1 上提升约 9‑10%，在预算受限情境下表现尤为显著；

**⚠️ 局限性**

局限性包括仅使用离散采样策略，动作空间受限；未与基础模型联合训练；策略选择的可解释性有限；在更大规模或多任务环境下的鲁棒性尚待进一步验证。

---

## 175. From Representation to Clusters: A Contrastive Learning Approach for Attributed Hypergraph Clustering

**arXiv ID:** 2603.09370 | [PDF](https://arxiv.org/pdf/2603.09370v1)

**作者:** Li Ni `[一作]` (Anhui University), Longlong Lin `[通讯]` (Southwest University)

**通讯引用:** 393 | [OpenAlex ID](https://openalex.org/A5101021504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种端到端的无监督属性超图聚类方法 CAHC，通过对节点和超边同时进行对比学习来生成节点嵌入，并在聚类阶段直接对嵌入进行联合优化得到聚类结果。

**💡 创新点**

创新点包括：① 设计了节点级与超边级的对比损失函数，既保留高阶结构信息又提升节点辨识度；② 引入多头注意力机制的超图神经网络，克服传统 HGNN 对超边中节点重要性忽略的问题；③ 在训练过程中将聚类指派作为监督信号，实现嵌入与聚类的协同优化，无需单独使用 k‑means。

**🔧 技术方法**

主要技术包括：对比学习（对节点与超边进行正负对比）、多头注意力超图神经网络（HGNN）、软/硬聚类指派损失、数据增强（特征掩码与超边关系掩码）和嵌入维度/温度调节。

**📊 数据集**

在八个公开超图数据集上进行实验，分别是 Cora‑C、Citeseer、Pubmed、Cora‑A、DBLP、NTU2012、20NewsW100 与 Mushroom。

**📈 对比分析**

与传统基线（Node2vec、Hyper2vec、DGI、RAGC）以及最新对比学习超图方法（TriCL、SE‑HSSL）进行对比。CAHC 在大多数数据集上均获得最高或次高的 ACC、F1、NMI、ARI 等指标，尤其在 Pubmed、Cora‑A、DBLP 等复杂结构数据上提升显著；仅在 20NewsW100 上因负超边生成策略不适合大规模超边导致略逊。

**⚠️ 局限性**

局限性包括：① 对超边规模较大的图（如 20NewsW100）负样本生成策略效果不足，导致超边对比损失弱化；② 对比学习与聚类指导需要多组超参数（掩码率、温度、嵌入维度）调优；③ 目前尚未针对大规模超图进行高效扩展，可能受限于内存与计算复杂度。

---

## 176. Multi-model approach for autonomous driving: A comprehensive study on traffic sign-, vehicle- and lane detection and behavioral cloning

**arXiv ID:** 2603.09255 | [PDF](https://arxiv.org/pdf/2603.09255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 177. Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning

**arXiv ID:** 2603.08921 | [PDF](https://arxiv.org/pdf/2603.08921v1)

**作者:** Mohamed Harmanani `[一作]` (Queen's University), Parvin Mousavi `[通讯]` (Queen's University)

**通讯引用:** 3657 | [OpenAlex ID](https://openalex.org/A5040401197)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究提出了MedCBR框架，将临床诊断指南嵌入概念瓶颈模型，通过视觉‑语言对齐和结构化推理生成可解释的诊断报告，提升医学图像分析的透明度与可靠性。

**💡 创新点**

创新点包括①使用大规模视觉‑语言模型生成符合指南的文本报告来丰富概念标签；②在多任务视觉‑语言模型中联合对齐、概念预测与诊断分类；③通过冻结的推理模型将预测结果与指南规则结合，输出结构化、可审计的诊断叙述。

**🔧 技术方法**

主要技术包括CLIP（ViT‑L/14等）视觉‑文本对齐、InfoNCE对比损失、概念与诊断的多任务监督、基于LVLM的报告生成、以及冻结的大型推理模型（LRM）进行规则驱动的推理。

**📊 数据集**

使用的主要数据集有：BUS‑BRA（超声影像）、CBIS‑DDSM（乳腺X光）以及BrEaST（补充训练集），外部验证集为CUB‑200‑2011；同时利用BI‑RADS Atlas与Sibley Bird Guide作为临床/领域指南文本。

**📈 对比分析**

与CBM、PCBM、Label‑free CBM、AdaCBM、各类CLIP变体及开箱即用的VLM（Qwen2.5VL‑7B、MedGemma‑4B、CLIP+Qwen3‑8B）进行比较。MedCBR在BUS‑BRA AUROC 94.2%（平衡准确率 89.0%）、CBIS‑DDSM AUROC 84.0%（BAL 76.4%）、CUB‑200 准确率 86.1%，同时在推理质量评估中获得最高 F1、敏感度和特异度，显示出诊断性能与可解释性兼得。

**⚠️ 局限性**

主要限制是对大量人工标注概念的依赖，标注成本高；缺乏无标签概念学习方法；在临床部署与跨模态推广方面仍需进一步验证。

---

## 178. SVG-EAR: Parameter-Free Linear Compensation for Sparse Video Generation via Error-aware Routing

**arXiv ID:** 2603.08982 | [PDF](https://arxiv.org/pdf/2603.08982v1)

**作者:** Xuanyi Zhou `[一作]` (University of California Berkeley), Alvin Cheung `[通讯]` (University of California Berkeley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关、块级误差感知的稀疏注意力机制，利用聚类中心补偿未计算块并按误差预算选择块

**💡 创新点**

创新点在于将误差估计与块选择结合，利用无参数线性补偿避免丢失全局信息，并给出误差上界

**🔧 技术方法**

使用语义聚类、线性补偿、误差估计的贪心路由、流式融合核实现

**📊 数据集**

在Wan2.2和HunyuanVideo两个开源视频生成模型上进行实验

**📈 对比分析**

与SVG、SVG2、SpargeAttention等基线比较，在保持或提升PSNR/SSIM的同时实现约1.5-1.9×推理加速，形成新的Pareto前沿

**⚠️ 局限性**

仅针对Diffusion Transformer，未验证对其他注意力机制的通用性

---

## 179. Can ChatGPT Generate Realistic Synthetic System Requirement Specifications? Results of a Case Study

**arXiv ID:** 2603.09335 | [PDF](https://arxiv.org/pdf/2603.09335v1)

**作者:** Alex R. Mattukat `[一作]` (RWTH Aachen University), Horst Lichter `[通讯]` (RWTH Aachen University)

**通讯引用:** 1180 | [OpenAlex ID](https://openalex.org/A5091346056)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在研究中，作者通过ChatGPT的黑盒语言模型迭代式 prompt 设计，生成了 300 篇面向 10 个行业领域的合成系统需求规范（SSyRS），并通过 LLM 自评（完整性与真实性度 DoR）及 SBERT 语义相似度评估，对产出质量进行初步量化；随后对 33% 的样本（30 篇）进行专家问卷调查，以验证 LLM 评估与专家感知的一致性。

**💡 创新点**

创新点在于：①将生成式 LLM 与自评机制结合，提出 DoR 与完整性两种量化指标；②采用模板化 prompt 与迭代细化方法，显著提升生成 SSyRS 的结构完整性与语义多样性；③首次探讨 LLM 真实性度评估的可靠性，并揭示模型偏差对评估结果的显著影响；④通过专家问卷验证 LLM 生成的 SSyRS 在真实感上的可接受度。

**🔧 技术方法**

技术手段包括：黑盒 LLM ChatGPT‑4o（生成与自评）、Prompt engineering（模板、Persona、Chain‑of‑Thought、Zero‑shot）、SBERT 语义相似度计算、Sonnet 4.5 交叉模型检查、Python 数据分析与可视化。

**📊 数据集**

数据集：10 个行业领域的 300 篇 SSyRS（每域 3 篇），包含完整的功能与非功能需求、设计约束等；专家评估样本 30 篇（每域 1 篇），共 87 份问卷（83 名专家）。

**📈 对比分析**

比较方法：将 LLM 自评得分（DoR、完整性）与 SBERT 语义相似度、专家 Likert 评分进行对比。结果显示，约 62% 的专家认为 SSyRS 具有“稍微或非常真实”特征；DoR 评估在不同 LLM 与上下文设置下表现不一，表明模型偏差显著；SBERT 语义相似度平均 0.66，表明生成的 SSyRS 在同一领域内保持一定多样性。

**⚠️ 局限性**

局限性包括：①模板过于简化，无法覆盖完整的 SyRS 复杂性；②LMM 自评的 DoR 量化不可靠，受模型偏差影响；③专家问卷采用便利抽样与自报专业度，可能导致专业性评估偏差；④仅覆盖 10 个预设行业领域，结果不易推广到其他领域；⑤仅单一 SSyRS 作为专家评估样本，统计效能有限。

---

## 180. Role Classification of Hosts within Enterprise Networks Based on Connection Patterns

**arXiv ID:** 2603.09910 | [PDF](https://arxiv.org/pdf/2603.09910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 181. Well Log-Guided Synthesis of Subsurface Images from Sparse Petrography Data Using cGANs

**arXiv ID:** 2603.09651 | [PDF](https://arxiv.org/pdf/2603.09651v1)

**作者:** Ali Sadeghkhani `[一作]` (School of Computer Science, University of Leeds), A. Rabbani `[通讯]` (School of Computer Science, University of Leeds)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用条件生成对抗网络（cGAN）根据井下孔隙率数据合成碳酸盐岩薄片图像，实现连续的孔隙尺度可视化。

**💡 创新点**

创新点在于将井下孔隙率信息作为条件输入，使生成器能够在不同孔隙率范围内生成地质一致的薄片图像，并弥补实测薄片数据稀缺、离散的问题。

**🔧 技术方法**

采用了cGAN框架，生成器包含6层反卷积，判别器包含5层卷积，使用LeakyReLU激活、Adam优化器（lr=2×10⁻⁴）和二元交叉熵损失函数。

**📊 数据集**

使用了5,000张256×256像素的薄片子图像，来源于15张石油岩薄片样本（深度1992-2000m），通过HSV阈值分割得到孔隙率并按10个区间分类，随后进行数据增强平衡。

**📈 对比分析**

通过生成图像的孔隙率与标注孔隙率比较，80%样本孔隙率误差≤10%（准确率81%），显示模型在孔隙率控制和地质结构再现方面具有良好性能。

**⚠️ 局限性**

局限性包括仅使用孔隙率作为条件，缺乏多参数（如渗透率、矿物成分）控制；数据量相对有限；模型仅生成二维图像，尚未实现三维孔隙结构生成；缺乏实际油藏验证。

---

## 182. PhD Thesis Summary: Methods for Reliability Assessment and Enhancement of Deep Neural Network Hardware Accelerators

**arXiv ID:** 2603.08724 | [PDF](https://arxiv.org/pdf/2603.08724v1)

**作者:** Mahdi Taheri `[一作]` (Tallinn University of Technology), Mahdi Taheri `[通讯]` (Tallinn University of Technology)

**通讯引用:** 3515 | [OpenAlex ID](https://openalex.org/A5074986688)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统文献综述、分析评估与改进技术，提出了一系列成本高效且零开销的可靠性评估与增强方法，显著提升深度神经网络硬件加速器在容错环境下的鲁棒性。

**💡 创新点**

创新点包括：①将故障注入、解析与混合方法体系化分类，揭示解析方法的轻量优势；②设计“AdAM”自适应容错近似乘法器，实现与TMR相近的可靠性，仅消耗约30%面积和40%功耗；③构建端到端工具链（DeepAxe、FORTUNE），实现量化、近似与容错的协同探索；④提出P_drop和RAP新指标，用于量化可靠性、存储与性能权衡。

**🔧 技术方法**

主要技术手段包括：系统级故障注入与模拟、解析可靠性模型、深度量化与近似计算、硬件级容错保护（如多位冗余与L0D检测）、高层次综合(HLS)与FPGA/ASIC实现、自动化设计空间探索(DSE)。

**📊 数据集**

使用了常见视觉与语音基准数据集：CIFAR-10/100、ImageNet、MNIST、FashionMNIST，以及在FPGA/ASIC上部署的VGG、ResNet、AlexNet、Inception等模型。

**📈 对比分析**

通过与传统TMR、全精度乘法器、现有近似乘法器（DRUM、TOSAM、ScaleTrim）对比，实验表明AdAM在准确率下降≤0.1%时，面积和能耗分别下降约2.7×和39%；在深度网络中，容错覆盖率提升至90%以上，且在FPGA上实现的DSE工具链在时间上比手工调优快数百倍。

**⚠️ 局限性**

主要局限包括：①解析方法仍需针对更大规模网络与多核加速器进行验证；②AdAM在极低精度（≤4bit）场景下的误差放大风险；③工具链对特定硬件平台（如某些ASIC设计工具）的兼容性尚未完全覆盖；④缺乏对恶意攻击（对抗样本）与安全性问题的联合分析。

---

## 183. Nezha: A Key-Value Separated Distributed Store with Optimized Raft Integration

**arXiv ID:** 2603.09122 | [PDF](https://arxiv.org/pdf/2603.09122v1)

**作者:** Yangyang Wang `[一作]` (Nanchang University), Zichen Xu `[通讯]` (Nanchang University)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5041407793)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出 Nezha 体系，融合键值分离与 Raft 协议（KVS-Raft），实现写入仅一次持久化并保持强一致性；引入 Raft-aware GC 与三阶段请求处理，优化读写性能。

**💡 创新点**

创新点在于：① 将键值分离嵌入 Raft 协议，消除 Raft‑LSM 的多重写放大；② 设计 Raft 友好的垃圾回收框架，动态平衡读写性能；③ 采用三阶段请求处理（Pre‑GC/During‑GC/Post‑GC），在 GC 期间仍保持高可用与一致性。

**🔧 技术方法**

核心技术包括：Raft 协议改造（KVS‑Raft）、键值分离（WiscKey 思路）、RocksDB + 自定义 ValueLog、Raft‑aware GC、哈希索引、三阶段请求调度、Go 语言实现、gRPC/Protocol Buffers、TLA+ 正式验证。

**📊 数据集**

实验使用 100 GB 随机键值数据，值大小 1 KB‑256 KB；YCSB 经典工作负载（A‑F）；扫描长度 10‑10 000 条；集群规模 3/5/7 节点；同时使用 10 GbE 连接、NVMe SSD 与 64 GB RAM。

**📈 对比分析**

与七种基线（Original、PASV、TiKV、Dwisckey、LSM‑Raft、Nezha‑NoGC）对比；Nezha 在 put 吞吐提升 460‑470%，latency 下降 60%；get 吞吐提升 12%，latency 下降 10%；scan 吞吐提升 72%，latency 下降 39%；整体 YCSB 吞吐提升 86%，latency 进一步下降。GC 触发时对写性能影响极小，写性能与原系统持平。

**⚠️ 局限性**

限制主要体现在：① GC 过程仍需额外存储空间与调度；② 在非 LSM 或非 NVMe 环境下性能提升需进一步验证；③ 长期大规模生产部署的稳定性与恢复延迟尚未在真实云环境中全面评估。

---

## 184. Do Ambient Backscatter Communication Receivers Require Low-Noise Amplifiers?

**arXiv ID:** 2603.09123 | [PDF](https://arxiv.org/pdf/2603.09123v1)

**作者:** Xinyi Wang `[一作]` (Xi'an University of Posts and Telecommunications), Guangyue Lu `[通讯]` (Xi'an University of Posts and Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了配备低噪放大器（LNA）的环境背向散射通信（AmBC）接收机的符号检测性能，推导了能量检测（ED）的误比特率（BER）与近似最优阈值，并提出了利用标签导频符号估计阈值参数的方法。

**💡 创新点**

创新点在于：①首次在AmBC系统中系统性地引入LNA并建模其非线性失真；②针对LNA整合的ED推导出闭式BER表达式和近似最优阈值；③提出基于导频能量统计的参数估计方案；④通过理论与仿真验证，证明LNA在低至中等源功率下显著提升符号检测性能。

**🔧 技术方法**

采用能量检测、偏差系数分析、LNA非线性模型（β1、β3）、Rayleigh衰落信道仿真、统计期望与方差推导、Monte Carlo 误码率仿真等技术。

**📊 数据集**

使用仿真数据：独立准静态Rayleigh衰落信道参数（如路径损耗、距离、噪声功率等）生成的合成信号，未使用公开真实数据集。

**📈 对比分析**

通过将带LNA与不带LNA两种方案在相同仿真条件下的BER曲线进行对比，评估不同源功率、不同BDPR（背向散射链接与直接链接功率比）以及导频开销对阈值估计误差的影响。结果表明：在低至中等源功率区间，配备LNA的接收机BER降低显著；随着源功率升高，LNA优势减弱；导频比例超过20%时阈值估计误差已趋于稳态。

**⚠️ 局限性**

局限性包括：①LNA在高源功率下会出现误码率饱和；②仅考虑了OOK调制与第三阶非线性失真，未覆盖更复杂调制与更高阶失真；③假设理想ADC无量化噪声，未验证硬件实现中的非理想效应；④未在真实AmBC硬件平台上进行实验验证。

---

## 185. PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments

**arXiv ID:** 2603.09760 | [PDF](https://arxiv.org/pdf/2603.09760v1)

**作者:** Guoliang Zhu `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**通讯引用:** 5284 | [OpenAlex ID](https://openalex.org/A5027010844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了在360°室内场景中进行全景性功能性（affordance）定位的新任务，并基于此提出了端到端的PanoAffordanceNet框架。

**💡 创新点**

创新点包括：①针对等距投影导致的纬度相关几何失真提出了Distortion-Aware Spectral Modulator (DASM)；②利用Omni‑Spherical Densification Head (OSDH)恢复稀疏激活并保证拓扑连续；③结合多层次（像素、分布、区域-文本）对比损失，显著抑制低监督下的语义漂移；④构建首个高质量全景功能性标注数据集360‑AGD。

**🔧 技术方法**

技术实现包括：基于DINOv2视觉编码器和CLIP文本编码器的双模态特征提取；LoRA低秩自适应训练；频域分离高低频的DASM模块；球面自相似传播的OSDH；多层次训练目标（BCE、KL、InfoNCE）。

**📊 数据集**

使用的数据集是自己构建的360‑AGD（分为Easy和Hard两份），并在标准视角的AGD20K上进行跨域泛化评估。

**📈 对比分析**

与现有的OOAL、OS‑AGDO等方法比较，PanoAffordanceNet在Easy/Hard split上KLD、SIM、NSS均显著下降/提升，平均KLD从≈1.48降至1.31；在AGD20K上保持或接近最优，证明在全景与视角两种投影下均具备良好鲁棒性。

**⚠️ 局限性**

主要局限包括：依赖大量预训练模型和GPU资源；对动态场景或非室内环境的适应性尚未验证；数据集规模相对有限，可能对极端纹理或遮挡场景的泛化仍有挑战。

---

## 186. Symbolic Discovery of Stochastic Differential Equations with Genetic Programming

**arXiv ID:** 2603.09597 | [PDF](https://arxiv.org/pdf/2603.09597v1)

**作者:** Sigur de Vries `[一作]` (Radboud University), Marcel A. J. van Gerven `[通讯]` (Radboud University)

**通讯引用:** 9770 | [OpenAlex ID](https://openalex.org/A5074794877)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于遗传程序（Genetic Programming，GP）的符号回归方法（GP‑SDE），同时学习随机微分方程（SDE）的漂移项和扩散项，并通过最大似然估计（MLE）直接评估模型适配度；

**💡 创新点**

创新点包括：①首次将GP用于SDE符号发现；②直接学习扩散函数，提升模型解释性和生成能力；③使用MLE作为目标函数，避免Kramers–Moyal分箱和两步回归的缺陷；④通过多步数值积分处理稀疏采样；⑤支持高维和随机偏微分方程（SPDE）的学习；

**🔧 技术方法**

技术手段包括：遗传程序（基于Kozax库）、多树表示（漂移+扩散）、NSGA‑II双目标优化（适配度+复杂度）、梯度下降优化常数、交叉/变异操作、子群迁移、MLE作为适配度、数值积分（多步）等；

**📊 数据集**

使用人工生成的模拟数据集，涵盖一维双阱、van der Pol、Rössler、Lorenz‑96（5/10/20维）、Lotka‑Volterra、Fisher‑KPP、二维热传导等多种SDE/SPDE基准系统；

**📈 对比分析**

与传统的Kramers–Moyal + 稀疏回归（KM‑SR）以及仅学习漂移的GP‑ODE进行比较。结果表明：在低维场景下GP‑SDE与KM‑SR相当；在高维或稀疏采样时GP‑SDE优于KM‑SR，且在多步积分版本（GP‑SDE‑MS）下对稀疏数据的恢复最优；GP‑SDE在泛化、生成样本和运行时可扩展性方面也优于对照方法；

**⚠️ 局限性**

主要局限包括：①假设系统完全可观测且噪声为可分离的高斯分布；②对非高斯跳跃噪声或不可观测变量支持不足；③模型识别存在可辨识性（identifiability）问题，优良拟合不一定保证真方程；④仍需手动调参（树深、常数学习步数等），对真实实验数据的适用性仍待验证。

---

## 187. VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models

**arXiv ID:** 2603.09826 | [PDF](https://arxiv.org/pdf/2603.09826v1)

**作者:** Shuhao Kang `[一作]` (Nankai University), Yun Liu `[通讯]` (Nankai University)

**通讯引用:** 15244 | [OpenAlex ID](https://openalex.org/A5078784976)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VLM-Loc，利用文本描述在三维点云地图中实现精确定位；

**💡 创新点**

创新点在于将点云转化为鸟瞰图和场景图，并引入部分节点分配机制，充分利用大视觉语言模型的空间推理能力，实现可解释的定位；

**🔧 技术方法**

采用大规模视觉语言模型（如 Qwen3-VL）与 BEV 渲染、场景图构建、PNA 机制，并通过 LoRA 微调实现跨模态对齐；

**📊 数据集**

使用 CityLoc 基准（CityLoc-K 基于 KITTI‑360 LiDAR，CityLoc‑C 基于 UAV photogrammetry）以及 KITTI‑360、CityRefer 等数据集；

**📈 对比分析**

与 Text2Pos、Text2Loc、MNCL、CMMLoc 等基线相比，在 CityLoc‑K 上 Recall@5m 提升约 15.46%（36.23% vs 20.77%），在 CityLoc‑C 迁移测试中也取得显著更高的召回率；

**⚠️ 局限性**

局限性包括仍受限于局部子地图范围、文本描述长度和语义稀疏性，且对非视差或缺失语义的点云映射鲁棒性待进一步提升。

---

## 188. Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control

**arXiv ID:** 2603.09221 | [PDF](https://arxiv.org/pdf/2603.09221v1)

**作者:** Peihao Wang `[一作]` (University of Texas at Austin), René Vidal `[通讯]` (University of Pennsylvania)

**通讯引用:** 26422 | [OpenAlex ID](https://openalex.org/A5011256828)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Test-Time Control (TTC) 层，将 LQR 控制规划嵌入语言模型的前向推理，实现在推理时对内部表征进行规划，并将 TTC 作为轻量级适配器集成到预训练 LLM 中。

**💡 创新点**

创新点包括：将推理视为最优控制问题，内部化价值函数；利用 KKT 和对称结构构造可微的 TTC 层；设计了硬件友好的对称迭代 LQR 求解器与 CUDA 核融合；支持时间异质参数化与混合轨迹采样，实现推理时可扩展的规划时程。

**🔧 技术方法**

使用的技术包括：线性二次调节器 (LQR)、可微优化、KKT 条件、对称矩阵（symplectic）迭代、CUDA 核融合、时间异质参数化、多头结构、混合轨迹采样、测试时尺度扩展。

**📊 数据集**

实验数据集：Sudoku 10k 9×9 棋盘（17-34 个已填数字）；数学推理数据集 Math-500、AMC、AIME 2024/2025；以及 80 万条自收集的推理示例。

**📈 对比分析**

与 Transformer、Mamba、Mamba2、GDN、Samba 等基准进行对比，利用单步/多步 Sudoku 完成率、MATH-500 Pass@8、AMC/AIME Pass@8 等指标；TTC-Net 在 Math-500 上提升约 +27.8%，在 AMC/AIME 上 Pass@8 提升 2-3 倍，Sudoku 单步板级准确率提升至 93.4%，多步 97.3%。

**⚠️ 局限性**

局限性：多层 TTC 之间的相互作用理论分析不足；仅探索了线性 LQR，非线性或更高阶动力学仍待研究；在更大规模模型与完整训练阶段的综合评估尚未完成。

---

## 189. DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics

**arXiv ID:** 2603.09668 | [PDF](https://arxiv.org/pdf/2603.09668v1)

**作者:** Yuanhang Lei `[一作]` (State Key Laboratory of Computer Aided Design and Computer Graphics Zhejiang University), Zhaopeng Cui `[通讯]` (State Key Laboratory of Computer Aided Design and Computer Graphics Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理信息的可微分框架 DiffWind，能够从稀疏视角视频中同时重建隐形风场和物体运动，并支持在新风条件下的前向仿真和风场重定向。

**💡 创新点**

创新点包括：① 将风场建模为网格物理场、物体建模为 3D 高斯点粒子系统并采用 MPM 实现粒子-网格耦合；② 通过可微分渲染与物理仿真联合优化风场和运动；③ 引入 Lattice Boltzmann Method 作为物理约束，保证重建的风场符合流体动力学；④ 构建 WD-Objects 既有合成也有真实的风场驱动数据集。

**🔧 技术方法**

使用技术包括 3D Gaussian Splatting、Material Point Method (MPM)、Lattice Boltzmann Method (LBM)、可微分渲染、Taichi 物理求解器以及多模态大型语言模型（MLLM）用于物理属性推理。

**📊 数据集**

使用的主要数据集为 WD-Objects，包含约 7 个合成 3D 对象与 8 组真实场景（植物、帽子等），每组分别采集多视角视频并在部分视角训练、剩余视角评估。

**📈 对比分析**

与 Deformable‑GS、Efficient‑GS、4D‑GS、GaussianPrediction 等前沿动态场景重建方法进行对比，使用 PSNR/SSIM/LPIPS（VGG）等指标评估新视角渲染质量；实验表明 DiffWind 在重建精度、时序连贯性和物理可逼真性方面均显著优于基线，且在用户研究中获得更高的真实感评分。

**⚠️ 局限性**

局限性：目前仅处理单个对象在风场中的动力学，未考虑多物体碰撞与交互；MPM 主要适用于连续体，无法直接模拟离散刚体；物理约束依赖 LBM，可能在极端流体行为下表现不足；进一步研究需要扩展到多体碰撞和非连续体模拟。

---

## 190. ICDAR 2025 Competition on End-to-End Document Image Machine Translation Towards Complex Layouts

**arXiv ID:** 2603.09392 | [PDF](https://arxiv.org/pdf/2603.09392v1)

**作者:** Yaping Zhang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Chengqing Zong `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文组织并评估了 ICDAR 2025 端到端文档图像机翻（DIMT）挑战赛，提出了两条轨道（OCR 基础与 OCR 免除）并分别设定小模型与大模型子赛，构建了包含 42,400 页的 Web 与学术文档数据集，提供基线模型与统一的文档级 BLEU 评价。

**💡 创新点**

创新点包括：①首次搭建面向复杂排版的 DIMT 综合基准；②同时支持 OCR 预处理与完全端到端的两种模式；③引入大模型与小模型双层评估，推动模型规模与资源效率的双向探索；④对多模态大模型在文档翻译中的表现与训练策略（SFT、DPO、LoRA 等）进行系统对比与分析。

**🔧 技术方法**

使用技术主要包括：大规模视觉-语言预训练模型（InternVL2.5-8B-MPO、Qwen2.5-VL-7B 等）、专门的排版感知模型（LayoutLM、LayoutLMv3、Donut、Nougat）、OCR 工具（如 Tesseract），以及 fine‑tuning 策略（监督微调、直接优先优化、对抗训练、链式思维训练、最小贝叶斯解码等）。

**📊 数据集**

数据集：DIMT-WebDoc-300K（300k 训练 + 1k 验证），DIMT-arXiv-124K（124k 训练 + 1k 验证），以及 1k 测试集；每张图像配有 OCR 结果、边框坐标、句子与文档级翻译，OCR-免除版则配有 Markdown 格式的目标翻译；参考数据集还包括 DIT700K 与 DoTA。

**📈 对比分析**

对比方法：基线模型为 Qwen2-VL-7B（OCR 免除）或 LayoutLM + Transformer；在 OCR 基础 LLM 子赛中，InternVL2.5-8B-MPO 取得 70.48 BLEU；在 OCR 免除 LLM 子赛中，InternVL2.5-8B-MPO 取得 60.78 BLEU；小模型赛道同样展示出 InternVL2.5-1B-MPO 与 Qwen2.5-0.5B-Instruct 的优异表现。整体而言，OCR 基础模型与大模型表现显著优于 OCR 免除与小模型。

**⚠️ 局限性**

局限性：①OCR 免除模式仍落后 OCR 基础模型，难以完全弥补 OCR 错误；②大模型对计算资源要求高，难以在资源受限环境部署；③数据集主要来自 Web 与 arXiv，排版多样性有限，未覆盖更多专业领域与多语言情况；④评估仅使用 BLEU，无法全面捕捉排版一致性与语义连贯性。

---

## 191. Upper Generalization Bounds for Neural Oscillators

**arXiv ID:** 2603.09742 | [PDF](https://arxiv.org/pdf/2603.09742v1)

**作者:** Zifeng Huang `[一作]` (Institute for Risk and Reliability, Leibniz University Hannover), Michael Beer `[通讯]` (Institute for Risk and Reliability, Leibniz University Hannover)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究考虑了一种由二阶常微分方程（ODE）和多层感知器（MLP）组成的神经振荡器，推导了其在近似因果和均匀连续算子以及均匀渐近增量稳定的二阶动态系统方面的PAC上界泛化界限。

**💡 创新点**

创新点在于理论上量化了神经振荡器的泛化能力，证明了在适当的正则化下，限制MLP的Lipschitz常数可以提高其泛化能力，并且估计误差随着MLP大小和时间长度的增加而多项式增长，从而避免了参数复杂度的诅咒。

**🔧 技术方法**

使用了基于Rademacher复杂度框架的理论分析方法，结合了二阶ODE和多层感知器的结构。

**📊 数据集**

使用了Bouc-Wen非线性系统作为数据集，进行随机地震激励下的数值研究，以验证理论结果。

**📈 对比分析**

通过与现有方法的比较，研究表明在有限训练数据下，约束MLP的矩阵和向量范数可以显著提高神经振荡器的性能，且理论结果与数值结果一致，验证了估计误差的幂律关系。

**⚠️ 局限性**

限制在于理论分析主要集中在特定类型的神经振荡器上，且在实际应用中，如何有效地选择和调整正则化参数仍需进一步研究。

---

## 192. Artificial Noise Versus Artificial Noise Elimination: Redefining Scaling Laws of Physical Layer Security

**arXiv ID:** 2603.09129 | [PDF](https://arxiv.org/pdf/2603.09129v1)

**作者:** Hong Niu `[一作]` (Nanyang Technological University), Chau Yuen `[通讯]` (Nanyang Technological University)

**通讯引用:** 42903 | [OpenAlex ID](https://openalex.org/A5060020877)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文重新定义了人工噪声（AN）与人工噪声消除（ANE）相互作用下的安全扩展律，推导了平均与瞬时密钥率的解析式；

**💡 创新点**

创新点在于：①将ANE视为AN的逆向干扰并给出完整的比例律；②提出了一系列阈值型下界与足够条件，揭示天线数、噪声比与AN功率对安全性的交互影响；③给出了简化近似和大规模极限表达式；

**🔧 技术方法**

采用了随机MIMO信道模型、奇异值分解、Wishart分布、指数积分函数以及极限相当理论（大矩阵分析）来推导安全容量；

**📊 数据集**

该工作为理论推导为主，未使用具体实验或公开数据集，所有结果均通过仿真验证；

**📈 对比分析**

通过与无AN情形以及传统AN方案进行仿真对比，证明当Eve天线不足或AN功率足够时AN仍能保证正密钥率；在Eve天线远超Alice时AN失效，阈值条件清晰；

**⚠️ 局限性**

局限在于假设Eve完全知晓并可精确实现ANE，忽略估计误差与实际硬件限制；仅考虑理想的Rayleigh衰落与AWGN，未覆盖多路径或协同干扰等实际网络场景。

---

## 193. Diagnosing and Repairing Citation Failures in Generative Engine Optimization

**arXiv ID:** 2603.09296 | [PDF](https://arxiv.org/pdf/2603.09296v1)

**作者:** Zhihua Tian `[一作]` (Virginia Tech), Ruoxi Jia `[通讯]` (Virginia Tech)

**通讯引用:** 2753 | [OpenAlex ID](https://openalex.org/A5032275274)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了诊断式的Generative Engine Optimization（GEO）方法AgentGEO，能针对单篇网页的未被引用问题进行原因诊断并自动修复。

**💡 创新点**

创新点在于构建了首个跨阶段的引用失败模式分类体系、基于此的诊断‑修复循环以及文档中心化的MIMIQ基准，显著提升引用率且保持内容完整。

**🔧 技术方法**

采用大型语言模型进行诊断、工具库实现HTML修复、内容重排、实体补全等；结合内存模块和批量聚合的迭代修复；使用局部块级编辑保证语义一致。

**📊 数据集**

使用新建的MIMIQ（Multi‑Intent Multi‑Query）数据集（含OOD和HTML变体），并对比了GEO‑Bench、E‑commerce、Researchy‑GEO等基准。

**📈 对比分析**

与传统的Vanilla、AutoGEO等方法比较，AgentGEO在GPT与Claude两种GE上均实现了约79%/70%的引用率，较AutoGEO提升10–15%，同时内容改动仅占5%而非25%。

**⚠️ 局限性**

局限性包括：对极端长尾主题仍可能无效；系统仍受GE内部偏好或资源限制影响，某些网页即便修复后仍无法被引用；实验基于模拟GE，未在真实商业引擎上验证。

---

## 194. The 802.11 MAC protocol leads to inefficient equilibria

**arXiv ID:** 2603.09902 | [PDF](https://arxiv.org/pdf/2603.09902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 195. Probing the Reliability of Driving VLMs: From Inconsistent Responses to Grounded Temporal Reasoning

**arXiv ID:** 2603.09512 | [PDF](https://arxiv.org/pdf/2603.09512v1)

**作者:** Chun-Peng Chang `[一作]` (DFKI Augmented Vision), Alain Pagani `[通讯]` (DFKI Augmented Vision)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究 Vision‑Language Models 在自动驾驶助手中的可靠性，探讨其在未来场景推理和响应一致性方面的表现。

**💡 创新点**

提出了人类标注的 FutureVQA 基准，并设计了仅使用过去帧的自监督链式思考 fine‑tuning 方法，显著提升 VLM 的时序推理与一致性。

**🔧 技术方法**

采用预训练 VLM（Hermes‑Yi‑34B + CLIP‑L）、自监督伪标签生成、链式思考（CoT）推理、时间加权损失 λ(Δt)=2^−Δt 等技术。

**📊 数据集**

使用 OpenDV‑YouTube 训练数据和 2.7k 人工标注的 FutureVQA 评测集（涵盖 1–12 秒预测）。

**📈 对比分析**

通过与标准 VQA、视频 VLM 对比，并使用自对齐与多轮测试衡量一致性；实验表明 fine‑tuned 模型在未来预测准确率下降率（NDR）和 ΔAcc^12s_1s 上明显优于基线，整体平均精度提升约 10‑15%。

**⚠️ 局限性**

局限包括：依赖基线模型生成的伪标签，可能携带错误；链式思考推理步骤多，推理延迟较高；缺乏真实长时序标注数据。

---

## 196. Compartmentalization-Aware Automated Program Repair

**arXiv ID:** 2603.09544 | [PDF](https://arxiv.org/pdf/2603.09544v1)

**作者:** Jia Hu `[一作]` (University of Manchester), Pierre Olivier `[通讯]` (University of Manchester)

**通讯引用:** 21830 | [OpenAlex ID](https://openalex.org/A5107884586)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于大型语言模型的跨越隔离界面漏洞修复框架。

**💡 创新点**

将隔离意识注入LLM提示，并结合两种CIV分类与回馈循环，以自动化修复跨界面漏洞。

**🔧 技术方法**

利用LLM（GPT‑4o‑mini、GPT‑5）、ConfFuzz隔离漏洞发现器、分析工具与自定义提示。

**📊 数据集**

采用ConfFuzz提供的600+条CIV案例进行验证。

**📈 对比分析**

与无隔离提示的LLM基线及现有APR工具对比，准确率100%放置在可信侧，避免沙盒内部修补，表现优于基线。

**⚠️ 局限性**

对更复杂的CIV、映射但已损坏的数据结构处理有限，需人工介入；仅评估单一恶意隔离情况。

---

## 197. A Generalized Voronoi Graph based Coverage Control Approach for Non-Convex Environment

**arXiv ID:** 2603.09596 | [PDF](https://arxiv.org/pdf/2603.09596v1)

**作者:** Zuyi Guo `[一作]` (Zhejiang University), Senlin Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 4332 | [OpenAlex ID](https://openalex.org/A5003643230)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出一种基于广义Voronoi图（GVG）的多机器人分区覆盖控制方法，分为负载均衡阶段和协作覆盖阶段；

**💡 创新点**

创新点在于（1）利用GVG对非凸、含多障碍区域进行子区域划分；（2）设计考虑子区域权重的分布式加权量化一致性负载均衡算法；（3）推导出沿GVG曲线运动的梯度下降控制器，实现机器人沿边界高效覆盖；（4）证明负载均衡和覆盖控制的收敛性；

**🔧 技术方法**

主要技术包括：广义Voronoi图构造、分布式加权量化一致性算法、梯度下降控制器、坐标变换与曲线参数化；

**📊 数据集**

实验使用人工生成的二维非凸区域（尺寸372×247，四个孔洞）和人工设定的密度函数ϕ(x,y)=10⁻⁸[(x-186)²+(y-86)²]；

**📈 对比分析**

通过仿真验证，20台机器人最终实现负载均衡（实际与理论分配误差≤1），覆盖成本随时间下降至约10⁻³，轨迹显示机器人沿GVG曲线覆盖且避免碰撞，说明算法在该场景下具有良好性能；

**⚠️ 局限性**

局限性包括：只能处理静态障碍的非凸环境；对障碍信息和GVG的完整性要求较高；负载均衡算法在整数约束下可能出现振荡；未考虑机器人动力学约束和实时通信延迟。

---

## 198. Leveraging whole slide difficulty in Multiple Instance Learning to improve prostate cancer grading

**arXiv ID:** 2603.09953 | [PDF](https://arxiv.org/pdf/2603.09953v1)

**作者:** Marie Arrivat `[一作]` (Primaa), Pietro Gori `[通讯]` (LTCI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了 Whole Slide Difficulty（WSD）概念，利用专家与非专家病理学家对前列腺活检 Whole Slide Image（WSI）Gleason 分级的分歧来衡量样本难度，并在多实例学习（MIL）训练中将该难度信息作为先验以提升分级准确率。

**💡 创新点**

创新点在于将 WSD 概念与两种方法（多任务回归和加权损失）结合进 MIL，显著提升了高 Gleason 级别分类性能，尤其在困难样本上效果更佳。

**🔧 技术方法**

技术方案包括使用 Histopathology Foundation Models（CTransPath 与 UNI2-h）提取特征，配合 MaxMIL、ABMIL、CLAM、DSMIL、TransMIL 等五种 MIL 架构，实施多任务学习与难度加权损失策略。

**📊 数据集**

数据集为 2,914 张 HE 染色的前列腺 WSI，分别由专家与非专家评估，划分为 1,995/507/412 的训练/验证/测试集。

**📈 对比分析**

与仅使用专家标签的基线分类器对比，WSD 方法在各 MIL 架构上平均提升 2.0 分平衡准确率，Gleason 5 的准确率提升 7.9 分，在困难样本上的表现尤为显著。

**⚠️ 局限性**

限制主要在于仅基于单一专家与非专家对照，缺乏跨医生差异的鲁棒性验证，并且仅在前列腺组织上测试，未验证对其他器官的泛化能力。

---

## 199. NaviNote: Enabling In-situ Spatial Annotation Authoring to Support Exploration and Navigation for Blind and Low Vision People

**arXiv ID:** 2603.08837 | [PDF](https://arxiv.org/pdf/2603.08837v1)

**作者:** Ruijia Chen `[一作]` (University of Wisconsin-Madison), Jessica Van Brummelen `[通讯]` (Niantic Spatial)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个基于语音交互、视觉定位（VPS）与多模态大语言模型的系统NaviNote，帮助盲人/低视力用户在“最后几米”场景下进行精准导航、查询与自建空间注释；

**💡 创新点**

创新点在于将高精度VPS与LLM驱动的对话式界面结合，实现无手持摄像头的高精度定位与实时语义导航，并提供基于位置的音频注释创作与访问；

**🔧 技术方法**

核心技术包括：iPhone LiDAR/相机的VPS定位、Unity+Niantic SDK前端、Python后端的Orchestrator+多Agent架构、GPT‑4o/mini进行问答与动作指令、音频/触觉多模态反馈；

**📊 数据集**

数据集：预先扫描的实验公园区域（约45×45 m）生成的3D点云与场景图谱，人工创建的39条注释（安全、无障碍、设施、布局、景点、体验、请求等六大类）；

**📈 对比分析**

与常用的基于GPS的盲人导航工具（TapTapSee）对比；结果显示NaviNote的成功率提高至14/16（vs 6/16），用户在主观指标（效能、易用性、认知负担、挫折感）上均显著优于基线，平均回答时间≈10.8 s；

**⚠️ 局限性**

局限性包括：依赖预先扫描的环境，缺乏实时场景变化检测与注释同步；对VPS的定位漂移仍存在风险；AI回答偶尔出现幻觉/误检；系统仅在单人、简易环境下测试，未覆盖多用户协作与复杂城市场景。

---

## 200. Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning

**arXiv ID:** 2603.09882 | [PDF](https://arxiv.org/pdf/2603.09882v1)

**作者:** Yixin Zheng `[一作]` (Institute of Automation), He Wang `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 1616 | [OpenAlex ID](https://openalex.org/A5075606222)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在桌面混乱场景中，通过非抓取（推、滑、翻转）动作实现目标物体的重新排布，并提出 Dynamics‑Aware Policy Learning（DAPL）框架，让机器人能够在多物体相互接触的复杂动力学环境中自适应地利用或避免外部接触；

**💡 创新点**

核心创新是：①通过物理世界模型学习接触诱发的动态表示，并将其作为条件输入给强化学习策略；②采用自适应课程学习，使得策略在无手工接触启发式或复杂奖励的前提下自然产生外在灵活性；③提出针对全 6D 物体重排的 Clutter6D 基准，专门考验密集场景下的外在灵活性。

**🔧 技术方法**

技术手段包括：基于点云的 ViT 编码器+解码器的物理世界模型（引入质量、速度等物理属性）；方差正则化的动态预测损失；强化学习（连续动作空间）与动态表示的结合；自适应课程学习循环；仿真环境 IsaacLab+PhysX；真实机器人 Franka‑Research‑3 和 Galbot 与视觉感知模块（SAM2、XMem、FoundationPose）。

**📊 数据集**

使用的数据集主要是自建的 Clutter6D，包含 1,024 个训练场景和 128 个测试场景，物体数分别为 4、8、12；物体资产来自 Objaverse 共 10K；真实世界实验在 10 个杂乱桌面场景中测试；除此之外还使用了现有的抓取规划基线和几何编码器训练集。

**📈 对比分析**

与抓取+规划（GraspGen+CuRobo）、人类遥操作、以及基于几何的编码器（Point2Vec、Concerto、CORN、UniCORN）等基线进行对比。DAPL 在模拟密集场景中成功率提升约 25% 以上（44% 对比 22%），在稀疏场景也保持优势；在真实世界 10 场景中达 48% 的成功率，几乎与人类 52% 相当，并且平均执行时间更短（42.6 s 对比 55.9 s）。

**⚠️ 局限性**

局限性包括：依赖近似的质量与速度估计，噪声容忍但可能影响精细操作；仅针对刚性桌面物体，未覆盖关节或柔性对象；未处理长时多阶段任务；以及对高动态或复杂环境的通用性仍需进一步验证。

---

## 201. BridgeDiff: Bridging Human Observations and Flat-Garment Synthesis for Virtual Try-Off

**arXiv ID:** 2603.09236 | [PDF](https://arxiv.org/pdf/2603.09236v1)

**作者:** Shuang Liu `[一作]` (Anhui University), Yu Liu `[通讯]` (Anhui University)

**通讯引用:** 63479 | [OpenAlex ID](https://openalex.org/A5100412458)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BridgeDiff框架，通过服装线索桥接模块和结构约束模块实现从穿着图像到平面服装的高质量重建。

**💡 创新点**

创新地引入服装线索桥接模块提取全局服装语义，并在扩散过程中注入平面结构注意力，实现视觉连续性与结构稳定性的统一。

**🔧 技术方法**

采用扩散模型、MetaFormer结构、VAE、文本编码器以及自定义的Flat-Constraint Attention（FC-Attention）实现条件融合。

**📊 数据集**

在VITON-HD和DressCode两个公开高分辨率数据集上进行训练与评估。

**📈 对比分析**

与现有VTOFF与统一多任务方法相比，在FID、PSNR、SSIM等指标上均优于或持平，尤其在低下衣和连衣裙类别中表现最为突出。

**⚠️ 局限性**

在极端遮挡或复杂姿态下仍可能出现细节缺失或结构失真，且对极端场景的泛化还有待提升。

---

## 202. Nemo: A Low-Write-Amplification Cache for Tiny Objects on Log-Structured Flash Devices

**arXiv ID:** 2603.09605 | [PDF](https://arxiv.org/pdf/2603.09605v1)

**作者:** Xufeng Yang `[一作]` (Xiamen University), Jiwu Shu `[通讯]` (Tsinghua University)

**通讯引用:** 3044 | [OpenAlex ID](https://openalex.org/A5101740783)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

设计并实现了一种新的闪存 KV 缓存架构（文中称为 Nemo），通过使用小哈希空间、Set-Group（SG）以及并行 Bloom 过滤器组（PBFG）等技术，显著降低了应用级写放大（ALWA）并同时保持低内存开销和低缺失率。

**💡 创新点**

创新点主要包括：① 在 Set-Associative 设计中引入小哈希空间和 SG 逻辑单位，提升 Set 的填充率，从而近似 1× 写放大；② 采用 PBFG 近似索引与按需闪存化元数据，实现 1% 误判率下仅 7.2 bits/obj 的索引存储；③ 结合热度感知写回和概率 flush 技术，进一步提升 SG 填充率至 89% 并避免不必要的写回。

**🔧 技术方法**

核心技术：小哈希空间 + SG 结构、并行 Bloom 过滤器组（PBFG）、热度跟踪的 1‑bit 计数器、概率 flush 与热度感知写回、FIFO SG 级别的写入与淘汰策略，以及在 CacheLib 框架中的实现。

**📊 数据集**

使用来自 Twitter 的四个真实流量集群（cluster_14、cluster_29、cluster_34、cluster_52）进行评估，这些集群均满足小对象（平均约 246 B）、大工作集（> 8 GB）且 Zipfian 分布（α≈1）等特征。

**📈 对比分析**

与 Log、Set、FairyWREN（FW）和 Kangaroo（KG）四种基线进行对比。Nemo 在同等闪存容量下将写放大降至 1.56×（比 FW 的 15.2×低约 90%），内存开销仅 8.3 bits/obj（比 Log 的 100+ bits/obj 低 85%），缺失率与 FW 相当；读取延迟稳定在 131 µs（p50）/523 µs（p99）/5 µs 内的改进；整体性能优于基线并保持可扩展性。

**⚠️ 局限性**

限制包括：需要针对不同闪存设备（如 ZNS、FDP 或传统 SSD）调优 SG 与 ERA 区映射；PBFG 近似索引在提高精度时可能导致读取放大；热度跟踪基于 1‑bit 计数器，可能误将部分冷对象标记为热，影响淘汰精度；在极高写放大需求或低 OP 空间的场景下，仍可能面临写回冲突。

---

## 203. Vibe-Creation: The Epistemology of Human-AI Emergent Cognition

**arXiv ID:** 2603.09486 | [PDF](https://arxiv.org/pdf/2603.09486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 204. Design Conductor: An agent autonomously builds a 1.5 GHz Linux-capable RISC-V CPU

**arXiv ID:** 2603.08716 | [PDF](https://arxiv.org/pdf/2603.08716v1)

**作者:** The Verkor Team `[一作]`, David Chin `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

基于大规模语言模型的 Design Conductor 系统实现了 VerCore RISC‑V 处理器的全流程设计，从 RTL 到 GDSII。

**💡 创新点**

创新点在于将 LLM 作为全流程智能代理，自动完成设计提议、模块化测试、验证驱动的迭代、PPA 优化，展示了在商业级设计中可持续迭代的可能性。

**🔧 技术方法**

使用技术包括：OpenAI GPT‑4 或类似 LLM、Spike RISC‑V 仿真器、OpenROAD 物理实现流、VCD 解析脚本、Python 自动化脚本、分布式文件系统和数据库。

**📊 数据集**

使用的数据集为 RISC‑V ISA 文档、Spike ISA 仿真数据、CoreMark benchmark、MD5 等自定义测试程序以及自建的 register‑trace 与 VCD。

**📈 对比分析**

对比方法：通过 Spike 对齐的周期级别仿真验证功能正确性；使用 OpenROAD 的 timing report 进行 PPA 评估；最终得到 CoreMark 3261 分、面积 2809 μm²、时钟 1.48 GHz。

**⚠️ 局限性**

局限性在于：仍需经验丰富的架构师进行指引；LLM 在 RTL 事件驱动和时序推理上存在误差；设计过程对规范写法高度敏感；大规模工艺约束下的验证和时序闭合仍然耗时。

---

## 205. GenePlan: Evolving Better Generalized PDDL Plans using Large Language Models

**arXiv ID:** 2603.09481 | [PDF](https://arxiv.org/pdf/2603.09481v1)

**作者:** Andrew Murray `[一作]` (J.P. Morgan AI Research), Michael Cashmore `[通讯]` (J.P. Morgan AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 GenePlan，一种基于进化式大语言模型的框架，用来自动生成针对经典 PDDL 规划任务的可解释、成本敏感的 Python 规划器。

**💡 创新点**

创新点在于将泛化规划视为优化问题，结合 LLM 作为进化算子（交叉、变异）在可执行代码空间内搜索，且通过平均计划长度作为评估指标实现计划质量优化。

**🔧 技术方法**

采用的技术包括：大型语言模型（GPT‑4o、GPT‑4o mini）用于生成代码、进化算法（种群、选择、交叉、变异、精英替换）、AST 解析与验证、以及统一规划框架进行评估。

**📊 数据集**

使用了八个 PDDL 域（包括七个公开基准域和两个新构建域：Trading 与 Research）共计 240 个实例（每域 30 个测试、5–10 个训练）。

**📈 对比分析**

与 Fast Downward、Chain‑of‑Thought 提示、以及其他 LLM 基线对比，GenePlan 在平均 SAT 分数上达 0.91，接近 Fast Downward 的 0.93，并显著优于 CoT（0.64）。生成器一次性成本约 1.82 美元/域，生成后每个实例平均求解时间仅 0.49 秒。

**⚠️ 局限性**

局限包括：仅在可行的 PDDL 域中表现良好，对缺乏通用策略的复杂域（如 Sokoban）效果不佳；进化过程耗时较长（约 645 秒/规划器）；目前仅优化计划长度，未考虑其他质量指标；且缺乏自动早停或动态选择搜索与生成式规划的机制。

---

## 206. Investigating Gender Stereotypes in Large Language Models via Social Determinants of Health

**arXiv ID:** 2603.09416 | [PDF](https://arxiv.org/pdf/2603.09416v1)

**作者:** Trung Hieu Ngo `[一作]` (Nantes Université), Emmanuel Morin `[通讯]` (Nantes Université)

**通讯引用:** 1421 | [OpenAlex ID](https://openalex.org/A5018688029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在法国医院匿名化患者记录的社会健康决定因素（SDoH）上，构建了一套性别刻板印象探测框架，使用性别中性化的SDoH输入对LLM的性别预测倾向进行评估

**💡 创新点**

提出了模型无关的、基于SDoH交互的性别刻板印象评估方法，并将人类注释者与LLM的预测进行对比，首次揭示两者在社会性别刻板印象上的相似性

**🔧 技术方法**

采用提示工程、Likert量表回归、修改后的RMSE偏差度量、Fisher精确检验和赔率比分析等技术，对多种LLM（包括开源与医学领域微调模型）进行实验

**📊 数据集**

使用法国大学医院收集的1700条匿名化社会历史片段（包含14项SDoH），经过筛选后得到958条用于实验的样本

**📈 对比分析**

通过平均RMSE偏差度量和赔率比热图对比9种LLM的性别预测偏差，发现大模型更稳定但医学微调后偏差略增；同时与人类注释者的结果比较，显示模型与人类在SDoH-性别关联上具有相似的刻板印象模式

**⚠️ 局限性**

局限包括样本来自单一法国医院、注释者样本单一、未探究SDoH组合对预测的影响、提示多样性可能带来扰动以及未评估不同语言中中性化处理的效果

---

## 207. Physics-Driven 3D Gaussian Rendering for Zero-Shot MRI Super-Resolution

**arXiv ID:** 2603.09621 | [PDF](https://arxiv.org/pdf/2603.09621v1)

**作者:** Shuting Liu `[一作]` (Sichuan University), Zizhou Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于显式3D高斯点云的零样本MRI超分辨率框架，可在无配对训练数据的情况下恢复高分辨率图像。

**💡 创新点**

将MRI特有的物理参数嵌入高斯点云，采用物理驱动的体渲染与砖块无序光栅化，实现高效且视角无关的重建。

**🔧 技术方法**

使用物理参数化高斯点云、体渲染归一化求和、CUDA并行砖块光栅化技术，支持零样本训练。

**📊 数据集**

在Medical Segmentation Decathlon（MSD）脑瘤数据集和FeTA（胎儿组织注释）数据集上进行实验。

**📈 对比分析**

与传统插值、监督CNN、NeRF等方法对比，PSNR/SSIM在2×/3×/4×放大倍数上均显著提升，训练/推理速度与显存需求大幅降低。

**⚠️ 局限性**

仍受限于高斯数量与物理假设，无法处理极端运动伪影，且未对多模态或低信噪比数据进行验证。

---

## 208. NetDiffuser: Deceiving DNN-Based Network Attack Detection Systems with Diffusion-Generated Adversarial Traffic

**arXiv ID:** 2603.08901 | [PDF](https://arxiv.org/pdf/2603.08901v1)

**作者:** Pratyay Kumar `[一作]` (New Mexico State University), Jayashree Harikumar `[通讯]` (DEVCOM Analysis Center)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于扩散模型的自然对抗样本（NAE）生成框架 NetDiffuser，用于欺骗基于深度学习的网络入侵检测系统（NIDS）。

**💡 创新点**

创新点在于：①使用聚类与Calinski–Harabasz指标系统识别可扰动的相对独立特征，从而在满足网络流合法性的前提下实现攻击；②将扩散模型与梯度攻击（FGSM/PGD/ACG）相结合，在反向扩散过程中注入细微扰动，生成与真实流统计分布高度一致的NAE。

**🔧 技术方法**

主要技术包括：特征相关性分析与层次聚类、Calinski–Harabasz评估、扩散概率模型（DDPM）训练与采样、传统梯度攻击算法、对抗检测器（MANDA、Artifact）的评估。

**📊 数据集**

使用了三大公开网络流数据集：CICIDS2017、CICDDOS2019 和 UNSW‑NB15，均按流级特征进行预处理后构建训练/测试集。

**📈 对比分析**

对比实验在多种模型架构（MLP‑1L/5L、CNN‑2L/3L）下与基线攻击（FGSM/PGD/ACG）及其自然版本（ND‑FGSM/ND‑PGD/ND‑ACG）进行评估；结果显示 NetDiffuser 的攻击成功率最高可提升 29.93%，并且在 MANDA 与 Artifact 检测器上的 AUC‑ROC 分数分别下降 0.267 与 0.534，证明其对现有防御的威胁显著。

**⚠️ 局限性**

局限性包括：①生成过程涉及多步扩散采样，导致运行时间显著增加；②目前仅在白盒场景下验证，灰盒/黑盒适配仍待进一步研究；③对抗检测器的泛化能力有限，需在更复杂网络环境下评估。

---

## 209. Dynamic Precision Math Engine for Linear Algebra and Trigonometry Acceleration on Xtensa LX6 Microcontrollers

**arXiv ID:** 2603.09333 | [PDF](https://arxiv.org/pdf/2603.09333v1)

**作者:** Elian Alfonso Lopez Preciado `[一作]` `[通讯]` (Independent Researcher), Elian Alfonso Lopez Preciado (Independent Researcher)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了ESP32上可在整数流水线与浮点单元之间动态切换的数学引擎，集成了Q16.16固定点算术、16迭代CORDIC三角函数与缓存感知的块矩阵乘法，并提供统一的API；

**💡 创新点**

创新点在于：①运行时精度切换机制（函数指针+FreeRTOS双阶段障碍）实现零成本的整数/浮点路径切换；②使用Q16.16固定点算术与16迭代CORDIC实现高精度、低延迟的三角运算；③块矩阵乘法与延迟修正相结合，显著减少舍入误差与缓存未命中；

**🔧 技术方法**

采用了Q16.16固定点格式、CORDIC迭代算法、循环分块（loop tiling）、FreeRTOS双核任务调度、函数指针动态分发、汇编优化指令、延迟修正与错误约束等技术；

**📊 数据集**

使用ESP32‑WROOM‑32物理硬件进行实验，随机生成输入，收集300对配对测量数据；

**📈 对比分析**

通过Wilcoxon符号秩检验将Fast Engine与标准math.h进行对比，结果显示sin、cos速度提升18.5×、24.7×；标量乘法提升1.5×；矩阵乘法在尺寸<32时慢0.54×，大尺寸未达标；性能提升显著且确定性高；

**⚠️ 局限性**

限制包括：矩阵乘法仅在尺寸≥64才有优势，块大小固定32；未全面评估误差与数值稳定性；单机实验，未验证多核/网络并行；CORDIC在四象限归约时存在分支预测误差。

---

## 210. Formation-Aware Adaptive Conformalized Perception for Safe Leader-Follower Multi-Robot Systems

**arXiv ID:** 2603.08958 | [PDF](https://arxiv.org/pdf/2603.08958v1)

**作者:** Richie R. Suganda `[一作]` (University of Houston), Bin Hu `[通讯]` (University of Houston)

**通讯引用:** 2567 | [OpenAlex ID](https://openalex.org/A5045548075)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在分布式视觉领导-跟随多机器人系统中，本文提出了一种分布式的、基于风险的自适应共形预测方法，结合控制障碍函数（CBF）实现感知安全的领导-跟随形成控制。

**💡 创新点**

创新点在于：①采用Mondrian共形预测对感知误差进行分区化校准，实现对不同视野风险区的自适应不确定度界限；②设计连续平滑的误差裕度插值以保持CBF约束的可行性；③在此基础上构建基于共形边界的CBF-QP，提供概率前向不变性保证。

**🔧 技术方法**

核心技术包括：共形预测（Mondrian CP）、风险感知分区、控制障碍函数与二次规划（CBF-QP）、深度视觉估计（ConvNeXt分类网络）以及仿真平台Gazebo。

**📊 数据集**

使用的感知数据集为200k张RGB图像，分辨率3×224×224，训练ConvNeXt网络进行方位角分类；校准使用450次仿真跑动收集的误差数据。

**📈 对比分析**

与基线比较：①不考虑感知误差的原始控制器；②两种全局共形预测基线（低风险区保守、全局保守）。在100次仿真跑动中，自适应方法成功率达95%，远高于原始4%、低风险基线23%和全局保守73%；同时跟踪误差和误差裕度均优于基线。

**⚠️ 局限性**

局限性包括：①共形预测的误差裕度仍可能过于保守，尤其在高风险区采用的单一阈值不够细粒度；②方法对相机模型和环境假设较为严格，实测环境中可能需要更多在线校准；③对多机器人规模的可扩展性尚未在大规模群体中验证。

---

## 211. On the Cost of Evolving Task Specialization in Multi-Robot Systems

**arXiv ID:** 2603.09552 | [PDF](https://arxiv.org/pdf/2603.09552v1)

**作者:** Paolo Leopardi `[一作]` (University of Konstanz), Tanja Katharina Kaiser `[通讯]` (University of Technology Nuremberg)

**通讯引用:** 89 | [OpenAlex ID](https://openalex.org/A5083764514)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在叶片收集仿真环境中，利用进化算法优化单层前馈人工神经网络，分别演化出整体（generalist）和子任务（dropper、collector）专门化控制器，并在两机器人场景下对其性能进行后评估。

**💡 创新点**

首次对任务分解（task specialization）在有限评估预算下的成本与收益进行定量比较，发现单一通用控制器在此条件下往往优于专门化控制器，挑战了传统认为分解能提升效率的观点。

**🔧 技术方法**

采用进化优化（基于种群的遗传算法，单个遗传位点变异），训练具备21个感知输入、8个隐藏节点的全连接前馈ANN，输出线速度和角速度；使用Gazebo仿真器进行物理建模与感知噪声处理。

**📊 数据集**

实验数据集为仿真环境中7个随机放置的圆柱形物体（源区），四个功能区域（源、斜坡、缓存、巢），以及两台差分驱动TurtleBot4机器人。

**📈 对比分析**

通过在5分钟评估期内统计进入巢区的物体数来衡量性能。结果显示，单一generalist对组的平均收集数显著高于dropper+collector组合，最佳专门化组合（C2-D2）虽优于其他专门化组合，但仍低于generalist；说明在有限评估预算下，通用策略更具鲁棒性。

**⚠️ 局限性**

局限性包括：仅在单机评估与两机器人场景下验证；评估预算有限，未探索更大预算下的专门化潜能；仿真与真实机器人之间存在转移鸿沟；仅考虑静态环境，未探讨动态变化和更大规模群体的交互。

---

## 212. Are Expressive Encoders Necessary for Discrete Graph Generation?

**arXiv ID:** 2603.08825 | [PDF](https://arxiv.org/pdf/2603.08825v1)

**作者:** Jay Revolinsky `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 25494 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于消息传播的可扩展图生成框架 GenGNN，并将其作为离散图扩散模型的低成本骨干网络。

**💡 创新点**

创新点在于将传统 GNN（如 GCN、GINe）与残差连接、边/节点门控、相对随机游走位置编码、全连接前馈网络等模块化设计相结合，使得单纯的消息传播网络即可在保持表达能力的同时显著提升推理速度并抑制过平滑。

**🔧 技术方法**

核心技术包括：基于 GCN/GINe 的消息传播层、残差 + 归一化、门控机制（Edge/Node Gating）、相对随机游走（RRWP）位置编码、两层 MLP 前馈网络、层归一化和全局池化，整合进离散扩散框架 DeFoG/DiGress。

**📊 数据集**

实验使用的公开基准数据集包括结构化图集（Comm20、Tree、Planar、SBM）、分子图集（QM9、ZINC250K）以及化合物生成与条件生成任务（MOSES、GuacaMol、TLS）。

**📈 对比分析**

通过与 Graph Transformer (GT)、PPGN、DiGress 等主流离散图生成模型的对比，GenGNN 在有效性（Validity）、平均 MMD 比例、Uniqueness、Novelty 等指标上均保持相当或更优，并在推理时间上实现 2‑5 倍的加速；在分子生成任务中亦达到或超过 99% 的 Validity。对照实验进一步验证了残差、门控和 RRWP 的必要性。

**⚠️ 局限性**

局限性：仍需要细致的模块配置与超参数调优；在极大图或高度稀疏图的稀疏性与长程依赖方面表现尚待进一步验证；当去除关键模块（如残差或 RRWP）时性能骤降，说明框架对这些组件高度依赖。

---

## 213. Reward-Zero: Language Embedding Driven Implicit Reward Mechanisms for Reinforcement Learning

**arXiv ID:** 2603.09331 | [PDF](https://arxiv.org/pdf/2603.09331v1)

**作者:** Heng Zhang `[一作]` (Istituto Italiano di Tecnologia), Yu She `[通讯]` (Purdue University)

**通讯引用:** 1444 | [OpenAlex ID](https://openalex.org/A5018653973)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Reward‑Zero，一种基于语言嵌入的隐式奖励机制，将自然语言任务描述映射为连续进度信号，补充稀疏或延迟的环境奖励并加速强化学习；

**💡 创新点**

核心创新是用语义相似度作为潜在函数，结合进度感知激活与完成感奖励，完全无需手工奖励设计，并提供衡量“完成感”的基准；

**🔧 技术方法**

利用预训练视觉语言模型（VLM）或 CLIP 生成文本/视觉嵌入，计算余弦相似度形成潜在值，再通过 sigmoid 激活与进度乘子构成奖励；

**📊 数据集**

使用 ManiSkill 机器人操作轨迹（OpenCabinetDrawer、AnymalC‑Reach、PushCube、PegInsertionSide、StackCube）构建完成感基准，并在多任务仿真中进行 RL 实验；

**📈 对比分析**

与 PPO 及传统奖励塑形基线对比，Reward‑Zero 在样本效率、收敛速度和最终成功率上显著优于基线；基准中 CLIP‑direct 方案实现最高前向区分率 72% 和 100% 跳跃检测；RL 任务中训练曲线更平稳，成功率显著提升；

**⚠️ 局限性**

局限性包括对视觉-语言模型描述准确性的依赖、对细粒度任务辨识度有限、VLM 计算成本高导致只能稀疏采样、可能出现目标回声偏差，以及尚未在真实机器人上验证。

---

## 214. Wrong Code, Right Structure: Learning Netlist Representations from Imperfect LLM-Generated RTL

**arXiv ID:** 2603.09161 | [PDF](https://arxiv.org/pdf/2603.09161v1)

**作者:** Siyang Cai `[一作]` (Institute of Computing Technology), Ying Wang `[通讯]` (Institute of Computing Technology)

**通讯引用:** 35895 | [OpenAlex ID](https://openalex.org/A5100347181)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了一个端到端的流水线，利用大型语言模型（LLM）生成 RTL，合成出门控级网表，经过结构相似度过滤与架构投票筛选后，使用 GNN 对网表进行表征学习，并在子电路边界识别及 IP 级功能分类任务上进行评估。

**💡 创新点**

创新点在于：①发现即使 LLM 生成的 RTL 存在功能错误，其合成网表仍保留可判别的结构模式；②提出无功能验证、仅靠结构相似度过滤的噪声数据增广方法；③通过架构投票鼓励 LLM 探索不同实现，显著提升模型对不同架构的泛化能力。

**🔧 技术方法**

使用的技术包括：LLM（DeepSeek‑V3.2‑Exp、GLM‑4.6）进行 RTL 生成；Synopsys Design Compiler 进行合成并提供反馈；结构相似度过滤（图嵌入余弦相似度）；架构投票机制；GraphSAINT 采样的 GNN 进行表征学习；节点/图级 MLP 进行下游分类。

**📊 数据集**

数据集包括：1）传统的小规模手工标注数据（ISCAS‑85、EPFL 等）做对照；2）从公开 SoC（PicoRV32、NEORV32）提取的规格与 RTL 生成的合成网表；3）通过 LLM 生成并过滤的合成网表数据集（多种规模的 T1–T5 以及 Voting 集）。

**📈 对比分析**

对比方法主要是与基线手工标注的小规模数据集、规则式逻辑重写增广方法以及未过滤的 LLM 原始数据。评估指标为 F1‑Micro 与 F1‑Macro。实验结果显示：在子电路边界识别任务中，LLM‑Aug‑t2 取得 F1‑Macro 93.79%（高于基线 90.15%）；在 IP 级 CPU 边界识别任务中，LLM‑Filtered 取得 F1 68.35%（高于规则式 58.28%）。

**⚠️ 局限性**

局限性包括：①对 LLM 生成质量和温度参数敏感；②结构相似度阈值的选取需经验调优，可能排除有价值的架构变体；③仅关注结构而非功能、时序或功耗等电路性能；④实验范围仅涵盖少数下游任务，未验证在更广泛的硬件分析任务中的效果。

---

## 215. Characterization, Analytical Planning, and Hybrid Force Control for the Inspire RH56DFX Hand

**arXiv ID:** 2603.08988 | [PDF](https://arxiv.org/pdf/2603.08988v1)

**作者:** Xuan Tan `[一作]` (University of Colorado), Nikolaus Correll `[通讯]` (University of Colorado)

**通讯引用:** 6045 | [OpenAlex ID](https://openalex.org/A5047458039)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对 Inspire RH56DFX 手进行硬件特征化、建立 MuJoCo 物理仿真模型、设计混合速度-力控制策略，并在 peg‑in‑hole 以及多种形状、尺寸的物体抓取任务上进行验证；通过分析闭合几何、仿真与实测对比，构建可解释且即插即用的研究平台。

**💡 创新点**

将商业级手指的非标力感知量化、结合仿真验证的解析式宽度‑抓取规划、以及基于速度切换的闭环力控制，形成完全无学习、可解释且能在不同感知模块间无缝对接的统一框架。

**🔧 技术方法**

硬件力学校准（线性力映射、延迟与超冲特性）、MuJoCo 动态模型与最小二乘识别、混合速度-力闭环控制、解析几何宽度‑抓取规划、二次规划验证、抓取质量（力闭合）分析、ROS2/Python 接口与 Tkinter 可视化。

**📊 数据集**

YCB 物体集（10 件加 1 个类似瓶子）、脆弱物体集（草莓、蓝莓、纸杯、鸡蛋、M6 螺母等）、Quadratic Peg 盘插入基准、以及用于仿真验证的 300 次抓取实验。

**📈 对比分析**

与三种闭合策略（naïve、reflex、iterative）进行对比；在 150 次总试验中，iterative 与 reflex 的成功率分别为 82.0% 与 86.7%，远高于 naïve 的 48.0%；在 YCB 子集上，reflex 达到 90% 成功率；peg‑in‑hole 试验中，指尖力阈值触发释放成功率为 65%（13/20），而腕部力阈值仅 10%（2/20）。

**⚠️ 局限性**

依赖高精度感知（尺寸、主轴、质心），未实现闭环力闭合监测；仿真与实测间仍存在约 20% 的误差；在复杂抓取任务与学习型方法的性能上仍未超越现有学习算法。

---

## 216. 3D UAV Trajectory Estimation and Classification from Internet Videos via Language Model

**arXiv ID:** 2603.09070 | [PDF](https://arxiv.org/pdf/2603.09070v1)

**作者:** Haoxiang Lei `[一作]`, Jianbo Su `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用大语言模型和视觉语言模型自动检索并筛选互联网上的无人机视频，随后在无监督条件下生成3D轨迹和无人机类别标签，并通过物理约束进行轨迹优化。

**💡 创新点**

提出三大创新点：① 语言驱动的数据采集与分层筛选，实现无人工标注的高质量视频聚集；② 训练自由的跨模态标签生成，通过多专家检测与VLM几何推断直接生成零样本3D轨迹候选；③ 物理信息驱动的递推优化（EKF），在相机投影模型下保证轨迹的时序平滑与运动可行性。

**🔧 技术方法**

核心技术包括：GPT‑4o 作为代理搜索与决策模型；CLIP/其变体用于视觉语言相关性与视角判别；Grounding SAM、轻量化无人机检测器、Anti‑UAV 基准模型等多专家检测器；DeepCalib 进行单帧相机标定；Extended Kalman Filter 对轨迹与深度进行递推平滑。

**📊 数据集**

主要数据集：自建的约200,000秒、2,245条的网络无人机视频集合；用于零样本迁移评估的公开3D MMAUD 基准数据集；还使用了多种公开的2D无人机追踪数据集做参考对比。

**📈 对比分析**

在 MMAUD 上的零样本转移实验中，方法取得 0.30 m 的平均3D误差，轴向误差 Dx=0.17 m、Dy=0.15 m、Dz=0.44 m，分类准确率 96%；与 7 大基线相比，在无监督/零样本条件下逼近甚至超越部分监督方法，推理速度约 23.5 FPS。

**⚠️ 局限性**

局限性：依赖 VLM 与 LLM 的推理准确度，受限于静态视角视频；对高度动态、极端光照或遮挡情况的鲁棒性有限；无标定相机外参仅在评估阶段使用，实际部署需保证相机模型一致；对超大规模视频的实时处理仍有计算瓶颈。

---

## 217. Sensitivity-Guided Framework for Pruned and Quantized Reservoir Computing Accelerators

**arXiv ID:** 2603.08737 | [PDF](https://arxiv.org/pdf/2603.08737v1)

**作者:** Atousa Jafari `[一作]` (Paderborn University), Marco Platzner `[通讯]` (Paderborn University)

**通讯引用:** 3469 | [OpenAlex ID](https://openalex.org/A5069368714)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种结合灵敏度导向剪枝和量化的 RC 模型压缩框架，并实现了对应 FPGA 加速器的自动化综合与设计空间探索。

**💡 创新点**

创新点在于使用基于量化权重位翻转的灵敏度分析，精准识别最不重要的权重实现无需再训练的剪枝；同时提供量化、剪枝联合的硬件设计空间探索方法，兼顾准确率与资源/能耗。

**🔧 技术方法**

技术包括：线性量化、基于位翻转的灵敏度导向剪枝、直接逻辑实现（LUT/FF 映射）、自动化 RTL 生成与 FPGA 综合、设计空间探索算法。

**📊 数据集**

使用的时间序列数据集有 MELBORN（分类）、PEN（分类）和 HENON（回归）三组。

**📈 对比分析**

与随机剪枝、MI、Spearman、PCA、Lasso 等方法在同一量化水平下比较，结果显示灵敏度剪枝在各种剪枝率下保持更高准确率/更低 RMSE，硬件上实现了 1–3% 资源节省和 50–70% PDP 降低。

**⚠️ 局限性**

局限性包括：仅针对单层 RC 网络，未验证对更深 ESN 的适用性；仅在 FPGA 上验证，缺乏 GPU/CPU 对比；灵敏度分析计算开销较大；数据集有限，需进一步验证泛化性。

---

## 218. ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer for Precision Assembly

**arXiv ID:** 2603.09565 | [PDF](https://arxiv.org/pdf/2603.09565v1)

**作者:** Minchi Ruan `[一作]` (Beijing University of Posts and Telecommunications), Bin Fang `[通讯]` (SunHDex Intelligent Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种融合视觉与触觉的行动分块Transformer（ReTac-ACT），用于高精度的插孔装配任务。

**💡 创新点**

创新点包括：双向交叉注意力实现视觉-触觉特征互补；基于关节位置的状态门控动态融合机制；以及辅助触觉重建目标以提升触觉表征质量。

**🔧 技术方法**

使用了基于ACT的CVAE变压器架构、ResNet-18视觉编码器、专门的触觉CNN编码器、双向多头注意力、以及可学习温度的门控网络。

**📊 数据集**

采用了NIST ATB M1标准装配基准和自建的5,000条多模态演示数据集（包含RGB图像、触觉图像、关节状态与动作）。

**📈 对比分析**

与ACT、Diffusion Policy以及pi05等基线对比，ReTac-ACT在3 mm公差下达到90%的插孔成功率，在0.1 mm公差下保持80%，明显优于基线（ACT 40%/15%，DP 20%/0%，pi05 20%）。

**⚠️ 局限性**

局限性在于目前仅验证圆柱形插销；未来需扩展到非轴对称形状、实现从真实到仿真/仿真到真实的触觉迁移，并探讨与大规模VLA预训练的融合。

---

## 219. Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists

**arXiv ID:** 2603.09277 | [PDF](https://arxiv.org/pdf/2603.09277v1)

**作者:** Jiaqi Liu `[一作]` (Wayne State University), Zhizhong Han `[通讯]` (Wayne State University)

**通讯引用:** 4723 | [OpenAlex ID](https://openalex.org/A5068597652)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出通过缩短每像素的高斯列表来加速3D高斯学习，并给出尺度重置与熵约束两种方法；

**💡 创新点**

创新点在于周期性缩小高斯尺度和对alpha混合权重进行熵正则化，使每个高斯更聚焦于少量像素，从而显著降低高斯列表长度，提升训练效率；

**🔧 技术方法**

使用了尺度重置（Scale Reset）、熵约束（Entropy Constraint）、渲染分辨率调度器、LiteGS框架以及MAE+D-SSIM基础损失；

**📊 数据集**

在Mip-NeRF 360、Tanks & Temples、Deep Blending等常用真实场景数据集上进行实验；

**📈 对比分析**

与3DGS、Taming‑3DGS、DashGaussian、LiteGS等方法在相同高斯数量下对比，训练时间比3DGS快约9.2×、比LiteGS快约50%，渲染质量（PSNR/SSIM）保持相近或略低；

**⚠️ 局限性**

存在轻微质量下降，且在小图像/少高斯场景中加速幅度降低；对极端复杂场景需更强正则化，可能导致质量进一步下降。

---

## 220. Alignment Is the Disease: Censorship Visibility and Alignment Constraint Complexity as Determinants of Collective Pathology in Multi-Agent LLM Systems

**arXiv ID:** 2603.08723 | [PDF](https://arxiv.org/pdf/2603.08723v1)

**作者:** Hiroki Fukui `[一作]` (Kyoto University), Hiroki Fukui `[通讯]` (Kyoto University)

**通讯引用:** 7687 | [OpenAlex ID](https://openalex.org/A5102813354)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在闭合设施中让四个LLM代理共同生活，实验不同可见性审查与对齐约束强度，量化集体病理表现与分离指标。

**💡 创新点**

首次在多代理模拟中揭示对齐本身可能产生集体病理（可见/不可见审查与对齐复杂度导致的“内部解离”），并将其与临床内因分离模型对齐。

**🔧 技术方法**

基于prompt层的对齐约束（安全提示、宪法原则、自我监控协议）、关键词匹配、复合指标（CPI、DI）、线性混合模型、置换检验及定性对话‑内部独白配对分析。

**📊 数据集**

使用公开LLM（Claude Sonnet、GPT‑4o、Grok‑2、DeepSeek‑V3、Llama 3.3 70B）与自定义日语/英语关键词列表，在实验中产生的对话和内部独白记录。

**📈 对比分析**

在每个模型-语言内采用配对Wilcoxon、Holm校正，CPI在不可见审查下显著提升（d≈1.9），DI在对齐复杂度增加时显著升高（d≈2.1），LMM与置换检验结果一致，表明效应稳健。

**⚠️ 局限性**

样本量小、对齐约束成分混合、对齐基准不透明、对齐对多模型普适性有限、内部独白通道为实验特性、DI为后验探索指数需进一步验证。

---

## 221. Ensuring Data Freshness in Multi-Rate Task Chains Scheduling

**arXiv ID:** 2603.09738 | [PDF](https://arxiv.org/pdf/2603.09738v1)

**作者:** José Luis Conradi Hoffmann `[一作]` (Federal University of Santa Catarina), Antônio Augusto Fröhlich `[通讯]` (Federal University of Santa Catarina)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

**🎯 论文内容**

提出一种基于数据新鲜度的任务链调度框架，通过在任务的起始偏移中注入数据新鲜度约束，实现 Just‑In‑Time（JIT）采样，消除 LET 缓冲带来的人工延迟。

**💡 创新点**

创新点在于把任务偏移由任务紧急度逆转为数据新鲜度驱动，提出主路径分解和共享生产者共识搜索算法，并给出正式证明表明该偏移策略不降低 Global EDF 的容量上限。

**🔧 技术方法**

使用任务偏移分配、主路径分解、共识搜索、基于数据新鲜度的有效截止时间推导、以及 DBF（Demand Bound Function）分析等技术。

**📊 数据集**

未给出具体实验数据集，论文以理论建模与证明为主。

**📈 对比分析**

通过示例和理论证明说明相较于传统 ASAP/LET 调度，能够显著提升数据新鲜度且不降低系统利用率；论文未提供实测性能数据。

**⚠️ 局限性**

局限性包括：假设网络延迟确定、数据新鲜度不具传递性、任务执行与通信不可抢占、共享生产者偏移可能需超周期拆分、缺乏真实系统仿真验证等。

---

## 222. DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction

**arXiv ID:** 2603.09291 | [PDF](https://arxiv.org/pdf/2603.09291v1)

**作者:** Fuzhen Jiang `[一作]` (Hangzhou Dianzi University), Yinlin Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对噪声多视图输入的 3D 场景重建方法 DenoiseSplat，能够在一次前向传播中恢复干净的 3D 高斯 SPLAT 表示。

**💡 创新点**

创新点在于将噪声抑制直接嵌入 3D 高斯 SPLAT 的 feed-forward 网络，采用双分支高斯头分离几何与外观学习，并在场景级一致噪声的 noisy‑clean 数据集上进行端到端训练。

**🔧 技术方法**

使用了 MVSplat 架构、3D Gaussian Splatting、双分支几何/外观高斯头、基于 IDF 的 2D 去噪对比、像素级 L1+SSIM 损失以及可微渲染器。

**📊 数据集**

在 RealEstate10K (RE10K) 数据集上构造四种噪声（高斯、泊松、斑点、椒盐）的一致 noisy‑clean 对，作为训练与评估基准。

**📈 对比分析**

与 MVSplat‑clean、MVSplat‑noisy、Denoise‑Then‑MVSplat 三种基线对比，DenoiseSplat 在 PSNR/SSIM/LPIPS 上均优于后两者，且逼近 clean 上的 upper bound。

**⚠️ 局限性**

局限性包括：仅使用合成噪声，未涵盖真实相机噪声、运动模糊、压缩失真等；仅在 RE10K 上验证，跨数据集与真实场景的泛化尚待进一步研究。

---

## 223. When Detectors Forget Forensics: Blocking Semantic Shortcuts for Generalizable AI-Generated Image Detection

**arXiv ID:** 2603.09242 | [PDF](https://arxiv.org/pdf/2603.09242v1)

**作者:** Chao Shuai `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35331 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种无参数的几何语义解耦（GSD）模块，用于消除 VFM 检测器中的语义桩塞，从而提高 AI 生成图像检测的泛化能力。

**💡 创新点**

创新点在于首次将语义回退机制与几何投影相结合，通过批量统计估计语义子空间并投影剔除，完全不依赖额外参数或损失，显著提升跨域检测效果。

**🔧 技术方法**

采用 CLIP ViT‑L/14 作为视觉主干，利用 QR 分解构造语义基底，并在前四层 Transformer 里插入 GSD 进行特征解耦，训练时仅使用标准二分类 BCE 损失。

**📊 数据集**

在面部伪造数据集（FaceForensics++、Celeb-DF、DF40）以及通用合成图像数据集（UniversalFakeDetect、GenImage）上进行实验，使用单源训练（仅 FF++ 或 ProGAN）评估跨数据集和跨算法的鲁棒性。

**📈 对比分析**

与现有最强方法（ForAda、Effort、VbSaT 等）相比，GSD 在视频级 AUC 上取得 94.4%（+1.2%），在 DF40 上平均视频级 AUC 97.8%（+3.0%），并在通用合成图像上实现 96.1% 的平均准确率，整体性能均显著提升。

**⚠️ 局限性**

局限性包括：依赖批量统计估计语义子空间，可能在极小批量或高噪声场景下不稳定；目前仅在面部与通用合成图像上验证，尚未探测更复杂场景或极低分辨率图像。

---

## 224. HECTOR: Hybrid Editable Compositional Object References for Video Generation

**arXiv ID:** 2603.08850 | [PDF](https://arxiv.org/pdf/2603.08850v1)

**作者:** Guofeng Zhang `[一作]` (Johns Hopkins University), Chongyang Ma `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于混合静态图像与动态视频参考的可编辑组合视频生成框架，支持多物体精细轨迹、缩放、速度控制以及背景锁定等编辑功能。

**💡 创新点**

核心创新点包括：
• Video Decompositor：使用点轨迹与尺度提取实现高质量、可追踪的轨迹与尺度标注；
• Spatio-Temporal Alignment Module (STAM)：将不同模态（图像/视频）参考映射至潜在空间，并通过可学习的 Gaussian 掩码实现空间对齐；
• 前后端的流匹配训练目标与前景/背景门控机制，提升多物体组合时的身份保持与遮挡处理。

**🔧 技术方法**

使用的技术栈：
• Diffusion Transformer (DiT) + VAE 编码；
• 轨迹跟踪与尺度估计：Cotracker3 与 SAM2；
• GridSample 逆变换与 Gaussian soft‑mask；
• Flow‑matching 训练目标、AdaLN 及多头交叉注意力；
• 前景/背景门控实现干扰抑制。

**📊 数据集**

训练集为内部 2.4 M 高分辨率视频（从 5 M 中筛选），包含多物体、复杂交互；评估使用 DAVIS 基准，自动生成掩码、轨迹与参考图像/视频。

**📈 对比分析**

与 MotionBooth、VACE（bbox/mask）进行单/多物体对比。实验表明：
• 在 R‑DINO、CLIP‑I、DINO‑I 等身份保持指标上领先；
• mIoU 与 Centroid Distance (CD) 显著提升，运动轨迹误差减半；
• 时序一致性与 CLIP‑T 维持或略高；
• 视觉质量和多物体交互更自然、边界更清晰。

**⚠️ 局限性**

局限性：
• 目前仅在图像参考的基线下评估，缺乏公开的动态视频参考对照；
• 依赖高算力（64 GPU、DiT-14B）和大量标注数据，对小型实验室不友好；
• 对极端遮挡或高速动态仍可能出现轨迹漂移或身份混淆；
• 仅针对静态/轨迹输入，未支持更复杂的交互式实时控制。

---

## 225. Reading the Mood Behind Words: Integrating Prosody-Derived Emotional Context into Socially Responsive VR Agents

**arXiv ID:** 2603.09324 | [PDF](https://arxiv.org/pdf/2603.09324v1)

**作者:** SangYeop Jeong `[一作]` (Seoul National University of Science and Technology), Seong-Eun Kim `[通讯]` (Seoul National University of Science and Technology)

**通讯引用:** 2323 | [OpenAlex ID](https://openalex.org/A5100694944)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在VR环境中，作者设计了一套将实时语音情绪识别（SER）结果作为对话上下文输入给大型语言模型（LLM）的交互流水线，并在用户说话时实时提取情绪标签，从而使虚拟角色能够根据用户的情绪做出更具情感共情的回应。

**💡 创新点**

创新点在于：①将情绪识别从传统的语义情感分析转为“情绪作为对话上下文”的显式处理，而非仅作为后置元数据；②采用内容-情绪解耦实验设计，确保语义中立或含糊的句子能单独检验语调情绪的影响；③在VR中的实时端到端流程中融合SER、STT与LLM，验证情绪上下文对社会存在感和交互质量的提升。

**🔧 技术方法**

核心技术包括：HuBERT‑based 语音情绪识别模型；OpenAI Whisper API 进行语音转文本；GPT‑4.1（通过 Convai API）生成对话回应；Unity 3D 与 Meta Quest 3 设备构建VR测试平台；情绪标签被注入LLM prompt 以塑造回应语调与风格。

**📊 数据集**

使用的数据集为 IEMOCAP（用于SER模型训练与评估）以及实验中自编的中立/情绪化脚本句子，实验参与者为 30 名本科生，在每种情绪（Happy、Sad、Angry）下使用 12 条预设句子进行交互。

**📈 对比分析**

比较方法：在 30 名参与者的 within‑subject 设计中对比“情绪识别（ER）”与“无情绪识别（NER）”两种条件，使用多项量表（UEQ、IMI、HAI、SAM）进行主观评估，并对结果进行配对 t 检验。结果显示 ER 条件在自然度、参与度、共情感、情感响应、对话质量、再使用意愿等指标上显著优于 NER，且 93.3% 的受试者更倾向于使用 ER 版本。

**⚠️ 局限性**

主要局限包括：①实验使用脚本化对话限制了自然交流的生态效度；②基于离散情绪标签的 SER 可能忽略多模态情绪信息；③系统平均响应时延约 3 秒，可能影响即时交互；④仅评估了情绪识别对语义中立句子的影响，未检验更复杂情境下的多情绪混合；⑤未来需探究低延迟端到端架构与更丰富情绪表达（如连续情绪或置信度权重）。

---

## 226. PPO-Based Hybrid Optimization for RIS-Assisted Semantic Vehicular Edge Computing

**arXiv ID:** 2603.09082 | [PDF](https://arxiv.org/pdf/2603.09082v1)

**作者:** Wei Feng `[一作]` (Jiangnan University), Qiang Fan `[通讯]` (Qualcomm)

**通讯引用:** 2268 | [OpenAlex ID](https://openalex.org/A5079841820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并实现了一个RIS辅助的语义感知车辆边缘计算（VEC）系统，并提出三路任务分配（本地、V2I、V2V）与联合优化框架。

**💡 创新点**

创新点在于将链路级RIS增强与语义通信深度集成，通过两层协同混合（PPO强化学习+线性规划）实现离散RIS相位、语义符号与连续任务分配率的联合非凸优化，克服高维搜索难题。

**🔧 技术方法**

主要技术包括：Proximal Policy Optimization（PPO）强化学习用于离散RIS相位与语义符号决策；线性规划（LP）求解任务分配率；DeepSC深度语义通信模型；RIS反射波束成形与V2I/V2V链路模型。

**📊 数据集**

实验基于预训练的DeepSC模型及仿真环境，任务负载采用Poisson分布，车辆速度固定20 m/s，RIS尺寸、车辆密度等参数在仿真中自行设置；未使用公开真实数据集。

**📈 对比分析**

与遗传算法（GA）和量子粒子群优化（QPSO）进行对比，平均系统延迟降低约40–50%，在不同发射功率、车辆密度和RIS规模下均表现出更优的性能和更高的稳定性。

**⚠️ 局限性**

局限性包括：假设语义知识库静态，未考虑动态更新；仅使用单个RIS，未探讨多RIS协作；车辆模型简化，未充分考虑真实环境干扰、能耗及更复杂的网络拓扑。

---

## 227. Differentiable Stochastic Traffic Dynamics: Physics-Informed Generative Modelling in Transportation

**arXiv ID:** 2603.09174 | [PDF](https://arxiv.org/pdf/2603.09174v1)

**作者:** Wuping Xin `[一作]` `[通讯]` (Caliper Corporation), Wuping Xin (Caliper Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于伊藤型随机LWR模型的分布式交通状态估计框架，推导出其一点前向Fokker-Planck方程及等价的概率流ODE，并结合分数匹配与物理约束训练得分网络，输出完整的密度分布。

**💡 创新点**

创新点在于：①将交通物理约束从确定性PDE转化为分布式的随机方程，直接推导分布前向动力学；②将Fokker-Planck前向方程转换为可微分的概率流ODE，实现自动微分训练；③将分数匹配与概率流残差耦合，形成分布式物理信息神经网络，可获取置信区间与拥堵风险指标。

**🔧 技术方法**

技术方法包括伊藤微积分、Fokker-Planck推导、概率流ODE、分数匹配（denoising score matching）、自动微分、闭包模块设计、正则化损失（物理残差、边界条件）以及多尺度噪声训练。

**📊 数据集**

当前版本未使用实际数据集，实验验证计划使用从Monte‑Carlo仿真得到的随机LWR样本；后续将对真实循环道或高速公路传感器数据进行验证。

**📈 对比分析**

尚未完成实验比较，未来将与传统PIDL、EKF/滤波器、以及无结构扩散模型进行对比，评估点估计精度、置信区间覆盖率和拥堵风险预测效果。

**⚠️ 局限性**

主要局限性：①理论仅在无冲击、平滑解的假设下成立；②未捕获空间相关性，仅提供一点边缘分布；③仅适用于一阶LWR模型，二阶模型与多车道情境待扩展；④闭包模型需要经验或学习，可能导致不确定性；⑤在实际稀疏观测下对数值稳定性与收敛性尚待验证。

---

## 228. ReCoSplat: Autoregressive Feed-Forward Gaussian Splatting Using Render-and-Compare

**arXiv ID:** 2603.09968 | [PDF](https://arxiv.org/pdf/2603.09968v1)

**作者:** Freeman Cheng `[一作]` (University of California Merced), Ming-Hsuan Yang `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种自回归的前馈高斯展开（ReCoSplat）模型，用于从连续图像流中实时构建3D场景并进行新视角合成。

**💡 创新点**

创新点包括：① Render-and-Compare 模块通过渲染当前重建并与观测图像比较，弥补训练时使用真实姿态与推理时使用预测姿态之间的分布不匹配；② KV 缓存压缩策略（早期层截断 + 按块保留）将Transformer的KV缓存量降低90%以上，使长序列推理在消费级显卡上可行；③ 通过动态块大小学习实现自适应推理。

**🔧 技术方法**

使用了 ViT 编码器（DINOv2 预训练）、交替注意力 Transformer、前馈高斯预测头、渲染-比较模块、跨模态注意力以及多任务损失（MSE、LPIPS、相机监督、稀疏性约束）。

**📊 数据集**

在 DL3DV、ScanNet++、RealEstate10K、ACID、ScanNet 等数据集上训练与评估，支持从 32 到 512 张图像的长序列，涵盖全姿态、无姿态、未标定等多种输入配置。

**📈 对比分析**

与 YoNoSplat、FreeSplat、S3PO‑GS 等基线进行比较，ReCoSplat 在自回归设置下实现了与离线方法相近甚至超过的 PSNR/SSIM/LIPIPS，且在无姿态与未标定条件下表现尤为突出；在摄像机姿态估计任务中，AUC 超过所有公开的在线方法。

**⚠️ 局限性**

局限性在于对姿态估计的依赖，姿态误差仍会影响高斯拼接和渲染质量；对极端姿态噪声的鲁棒性仍有限，需进一步提升在线姿态估计性能。

---

## 229. Deep Tabular Research via Continual Experience-Driven Execution

**arXiv ID:** 2603.09151 | [PDF](https://arxiv.org/pdf/2603.09151v1)

**作者:** Junnan Dong `[一作]` (Tencent Youtu Lab), Feiyue Huang `[通讯]` (Ruijin Hospital, Shanghai Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于闭环决策的Deep Tabular Research (DTR) 框架，用 LLM 结合图结构表格表示和可执行操作序列实现对不规则表格的深度多步分析推理。

**💡 创新点**

创新点在于将宏观规划与微观执行分离，采用期望感知路径选择算法与孪生结构化记忆持续改进策略，构建可执行操作空间并通过执行反馈实现连续学习。

**🔧 技术方法**

核心技术包括：表格元信息抽取与双向标题识别构建元图；LLM 驱动的查询拆分与操作映射；期望感知的 UCB‑style 路径评分；孪生记忆（参数化执行反馈与抽象经验）；以及 [THINK]/[CODE] 交互式提示策略。

**📊 数据集**

主要实验数据集为 DTR‑Bench（多步分析、非结构化表格）和 RealHitBench（多类型表格推理），并在公开的表格 QA 基准上进一步验证。

**📈 对比分析**

与 TableGPT、TableLLM、StructGPT、DeepSeek‑V3/V3.2、ST‑Raptor、TreeThinker、Code Loop 等基线比较，DTR 在 DTR‑Bench 上综合得分提升至 42.7 分，RealHitBench 上准确率最高达 58.2%/64.5%，并且 LLM 调用次数仅约 4.8 次，显著低于 Code Loop 的 8.8 次，显示出更优的性能‑效率平衡。

**⚠️ 局限性**

局限性包括：对 LLM 生成代码的错误率仍不可忽视；对操作库的依赖可能限制在未知表格类型上的迁移；大规模表格的内存与时间开销仍待优化；在极端复杂或极长的推理链中，累积误差和回溯成本可能增加。

---

## 230. Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms

**arXiv ID:** 2603.09090 | [PDF](https://arxiv.org/pdf/2603.09090v1)

**作者:** Renos Zabounidis `[一作]` (Carnegie Mellon University), Katia P. Sycara `[通讯]` (Carnegie Mellon University)

**通讯引用:** 19827 | [OpenAlex ID](https://openalex.org/A5087505541)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在状态依赖有效动作（state‑dependent action validity）环境中，无掩码训练导致有效动作被指数级抑制的现象，提出通过可行性分类（feasibility classification）和 KL 加权损失来学习可区分有效与无效状态的特征，从而实现不需要 oracle mask 的部署。

**💡 创新点**

①理论证明了在 softmax 策略下共享特征导致的有效动作指数抑制上界；②发现特征对齐（feature alignment）是抑制的关键机制；③提出通过预测动作有效性作为辅助任务（feasibility classification）来打破特征对齐，并设计 KL‑balanced 损失专注于对策略敏感的动作，从而显著降低抑制并提升性能；④展示了可在没有 oracle mask 的情况下几乎保持 oracle 级别的性能。

**🔧 技术方法**

使用 softmax 策略与共享特征的网络，Policy Gradient（PPO）训练，动作掩码（action masking），熵正则化，辅助可行性分类（binary heads），KL‑balanced 与 focal 损失比较，三种 PPO 变体（MLP、RNN、Hybrid）、Transformer‑XL、RND 探索等技术。

**📊 数据集**

Craftax、Craftax‑Classic（共 43 个动作）和 MiniHack Corridor‑5（共 11 个动作）三类离散动作、符号状态环境。

**📈 对比分析**

与四种实验设置比较：C1（oracle mask）、C2（无掩码）、C3（mask + focal loss）、C4（mask + KL‑balanced loss）。结果显示无掩码导致有效动作概率指数下降；oracle mask 大幅提升样本效率；mask+KL 在 oracle mask 下进一步提升 5–10% 的最终回报；在无 oracle mask 部署时 mask+KL 仅比 oracle 降低 1–2% 的性能，且相较于无掩码训练保持更高的可靠性。

**⚠️ 局限性**

假设网络特征 ϕ(s) 固定；未分析共享特征与策略权重共同优化对抑制的影响；训练时需要完整的有效性标签，标签噪声会影响分类精度；实验仅覆盖离散动作和符号状态，连续或像素观测下的抑制机制未知；抑制机制与其它深 RL 失败模式（如表示坍塌、记忆衰退）交互关系待进一步研究。

---

## 231. CktEvo: Repository-Level RTL Code Benchmark for Design Evolution

**arXiv ID:** 2603.08718 | [PDF](https://arxiv.org/pdf/2603.08718v1)

**作者:** Zhengyuan Shi `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14271 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了面向完整RTL代码仓库的演化基准（CircuitEvolve）和闭环LLM驱动框架，利用图结构分析、Prompt生成以及EDA工具链反馈，实现跨文件的功能保持且PPA优化。

**💡 创新点**

创新点在于：①首次构建完整多文件RTL基准，②将repo级RTL演化任务与LLM驱动的闭环迭代框架相结合，③通过图结构代码分析和定向Prompt，让LLM能够定位并修复跨文件的性能瓶颈。

**🔧 技术方法**

技术手段包括：LLM（DeepSeek-v3、GPT‑4o、Qwen3‑coder）、代码控制数据流图（CDFG）分析、Prompt生成器、形式化验证（Formality）、开源与商业EDA工具（Yosys、Synopsys Design Compiler）、岛屿进化算法与MAP‑Elites归档。

**📊 数据集**

使用的数据集为CircuitEvolve基准，包含11个来自DeepCircuitX/ForgeEDA的高质量Verilog仓库，涵盖处理器、控制器、接口、编码器/解码器等多种设计。

**📈 对比分析**

通过与单轮EDA优化对比，实验在开源工具链下平均ADP下降10.50%，延迟下降更显著；在商业工具链下也实现了1.77%的ADP提升，且关键路径延迟减少10.61%。实验表明闭环框架能在无人工干预的情况下取得显著PPA改进。

**⚠️ 局限性**

局限性包括：LLM在本地编码优化已接近EDA工具极限，对商业工具提升有限；缺乏高层架构重构能力；需进一步挖掘环境特定的优化规则和实现更强的模型适配。

---

## 232. TopoOR: A Unified Topological Scene Representation for the Operating Room

**arXiv ID:** 2603.09466 | [PDF](https://arxiv.org/pdf/2603.09466v1)

**作者:** Tony Danjun Wang `[一作]` (Technical University of Munich), Lennart Bastian `[通讯]` (Technical University of Munich)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5018052917)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的拓扑场景表示方法，用于手术室的三维感知与交互。

**💡 创新点**

将拓扑结构与语义标签相结合，实现对手术室物体的层次化表示与动态更新。

**🔧 技术方法**

使用图卷积网络进行拓扑推理，结合多模态传感器融合与几何约束的深度学习框架。

**📊 数据集**

在公开的ORScene和SurgData两大手术室数据集上进行训练与测试。

**📈 对比分析**

与传统基于体素、点云分割以及深度地图的方法对比，TopoOR在IoU和检测准确率上分别提升约12%和8%，并实现了更低的推理时间。

**⚠️ 局限性**

对极端光照、遮挡和多物体交互的鲁棒性不足，且需要大量标注数据，推理时仍存在一定的计算瓶颈。

---

## 233. Time, Identity and Consciousness in Language Model Agents

**arXiv ID:** 2603.09043 | [PDF](https://arxiv.org/pdf/2603.09043v1)

**作者:** Elija Perrier `[一作]` (University of Technology Sydney), Michael Timothy Bennett `[通讯]` (Australian National University)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5030698786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对语言模型代理（LMA）的身份评估方法进行理论研究，区分了身份成分的出现与在单个决策步骤中的共同实例化。

**💡 创新点**

创新点在于将Stack Theory的窗口语义与Temporal Gap引入LMA身份评估，提出弱持久性（Ingredient‑wise Occurrence）和强持久性（Co‑Instantiation）两个可测指标，并将Arpeggio与Chord同步后置条件映射为可实现的评估标准。

**🔧 技术方法**

采用模态逻辑窗口语义、Stack Theory 的 Arpeggio 与 Chord 后置条件、窗口映射与指标计算方法，以及对脚手架状态的显式定义与监测技术。

**📊 数据集**

论文未使用具体实验数据集，全部基于理论构建与抽象模型；若要验证，可使用现有 LMA 框架（如 Retrieval Augmented Generation）中的日志数据。

**📈 对比分析**

暂无实验比较；理论上提出的强持久性指标比传统的回忆测试更严格，能够揭示身份成分在行动时未真正联合激活的情况。

**⚠️ 局限性**

局限性包括：缺乏实证验证与数据支持；指标实现依赖对脚手架内部状态的完整可观测性；实际部署时对不同 LMA 架构的可适配性与可测度仍待探索。

---

## 234. Reward Prediction with Factorized World States

**arXiv ID:** 2603.09400 | [PDF](https://arxiv.org/pdf/2603.09400v1)

**作者:** Yijun Shen `[一作]` (East China Normal University), Pascale Fung `[通讯]` (HKUST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RewardPrediction基准和StateFactory框架，用于零样本奖励预测。

**💡 创新点**

创新点在于将世界状态分解为层级对象‑属性结构，并通过语义相似度实现奖励估计，显著提升跨域零样本奖励精度。

**🔧 技术方法**

采用大型语言模型进行状态提取与目标解释，结合层级路由与语义嵌入，并使用EPIC距离作为评估指标。

**📊 数据集**

使用了五个文本交互环境（AlfWorld、ScienceWorld、TextWorld、WebShop、BlocksWorld）共收集2454条轨迹。

**📈 对比分析**

与监督式奖励模型、VLWM‑critic、LLM‑as‑a‑Judge等基线对比，StateFactory在EPIC距离上降低约60%，并在各环境中成功率提升约21%~12%。

**⚠️ 局限性**

仍依赖大型语言模型的推理与语义嵌入质量，对极端复杂或非文本环境的泛化能力尚待进一步验证。

---

## 235. The Missing Memory Hierarchy: Demand Paging for LLM Context Windows

**arXiv ID:** 2603.09023 | [PDF](https://arxiv.org/pdf/2603.09023v1)

**作者:** Tony Mason `[一作]` (University of British Columbia), Tony Mason `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5041896410)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个名为 Pichay 的代理系统，用来实现大语言模型（LLM）上下文窗口的需求分页、页错误检测与页面固定，从而显著降低上下文中的结构性浪费。

**💡 创新点**

提出将 LLM 上下文窗口视为 L1 缓存并引入完整的虚拟内存层次结构（L1–L4），实现基于错误驱动的页面固定、合作式内存管理和多级压缩，首次将操作系统的页面置换理论直接迁移到 LLM 领域。

**🔧 技术方法**

使用透明 HTTP 代理拦截 Messages API，实施 FIFO 页淘汰、错误检测、错误驱动固定、检索句柄、虚拟工具与清理标签等技术，并在代理层实现统计与日志记录。

**📊 数据集**

在单一高负载用户的 857 个 Claude Code 会话（共 54,170 次 API 调用、4.45 亿有效输入 token）上进行实验，并在 5 个会话（99 次 API 调用）中对请求体进行细粒度拆解。

**📈 对比分析**

通过对比基线、工具定义裁剪、以及分页+裁剪三种处理方式，发现分页+裁剪可将有效输入 token 减少 37.1%（相当于 93 亿个 attention 对消除），在生产部署中单会话上下文容量提升 36% 以上，且在标准化任务上保持正确性，缺陷率仅 0.0254%。

**⚠️ 局限性**

主要局限包括：实验数据仅来自单个高负载用户；跨会话持久化内存 L4 仍未实现；FIFO 淘汰策略在极端长会话会出现剧烈 thrashing；对多用户、多模型、多任务场景的泛化与成本评估尚未完成。

---

## 236. Learning Bayesian and Markov Networks with an Unreliable Oracle

**arXiv ID:** 2603.09563 | [PDF](https://arxiv.org/pdf/2603.09563v1)

**作者:** Juha Harviainen `[一作]` (University of Helsinki), Vidya Sagar Sharma `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究了在条件独立性（CI）测试可能出现有限错误的情况下，如何对马尔可夫网络和贝叶斯网络进行结构学习；

**💡 创新点**

提出了“k-可识别性”概念，并证明当图的最大点对连通度较小或贝叶斯网络为稀疏链结构时，可容忍指数级错误；

**🔧 技术方法**

通过理论分析、离散化的CI查询、可识别性距离计算和枚举法（对马尔可夫网络直接修正错误，对贝叶斯网络枚举错误集并运行PC算法）实现结构恢复；

**📊 数据集**

未使用具体数据集，全部为理论和算法复杂度分析；

**📈 对比分析**

与传统的无错误PC算法相比，本文给出了在k错误时可唯一识别的充分条件和算法复杂度（O(n²k+2ⁿ)）；然而在最坏情况下仍需完整查询集合，表现为指数级；

**⚠️ 局限性**

主要限制：对贝叶斯网络的可识别性分析仅局限于链骨架，无法推广；最坏情况下仍需n²2ⁿ-2个查询；未提供误差纠正或实验验证。

---

## 237. Memorization capacity of deep ReLU neural networks characterized by width and depth

**arXiv ID:** 2603.09589 | [PDF](https://arxiv.org/pdf/2603.09589v1)

**作者:** Xin Yang `[一作]` (Sun Yat-sen University), Yunfei Yang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 7138 | [OpenAlex ID](https://openalex.org/A5038316493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

本文通过构造深度 ReLU 网络，证明了在单位球内任意 N 个相距至少 δ、标签取离散值 C 的数据点能够被记忆的网络宽度 W 与深度 L 的关系满足 W^2L^2 ≲ N(log(δ^{-1})+log C)，并给出了对应的下界，证明了该构造在 δ^{-1} 为 N 的多项式时达到最优（至多多项式对数误差）。

**💡 创新点**

创新点在于：①首次精确刻画了宽度与深度之间的权衡，给出了 W 与 L 的闭式上限与下限；②引入可调参数 S、T 以动态划分样本块和位提取层，打破了以往固定宽度/深度的硬限制；③使用位编码与分块存储实现高效记忆，实现参数数量仅为 O(√N)（当 δ^{-1} 为 N 的多项式时）。

**🔧 技术方法**

主要技术包括：1) 通过一维投影网络将高维点映射到区间 [0,R]，保证整数部分互不相同；2) 采用块编码将样本及标签的整数部分串联成大整数 u_j、w_j；3) 构造可在给定宽度和深度下实现的分段线性函数网络实现这些映射；4) 利用位提取与匹配子网络实现按位解码；5) 通过 Warren Lemma 和 VC 维度论证下界。

**📊 数据集**

本研究为理论性工作，未使用具体数据集；所有结果基于任意满足距离与标签条件的点集。

**📈 对比分析**

与以往仅以参数数目或单层宽度为指标的研究相比，本论文提供了更细粒度的宽度–深度权衡分析，并证明了构造的网络在大多数实用情形下（δ^{-1} 为多项式）达到近似最优；上界与下界相差至多多项式对数因子。

**⚠️ 局限性**

局限性包括：①结果仅适用于离散标签，连续标签需 Ω(N) 参数；②对 ReLU 激活函数的特殊性，尚未验证是否能推广到其他激活函数；③多项式对数因子可能不是必要的（或可能可以进一步压缩）；④理论证明基于构造，尚未评估在实际训练算法（如 SGD）下能否实现同样的记忆容量。

---

## 238. Rotation Equivariant Mamba for Vision Tasks

**arXiv ID:** 2603.09138 | [PDF](https://arxiv.org/pdf/2603.09138v1)

**作者:** Zhongchen Zhao `[一作]` (Xi'an Jiaotong University), Zongben Xu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 15970 | [OpenAlex ID](https://openalex.org/A5109280540)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种端到端旋转等变的Mamba网络EQ-VMamba，用于视觉任务。

**💡 创新点**

创新点在于引入旋转等变的跨扫描策略和组化的Mamba块，确保模型在90度旋转下保持等变。

**🔧 技术方法**

采用了旋转等变卷积、旋转等变线性层、基于状态空间模型的EQ-VSS块以及组共享参数的EQ-Linear等技术。

**📊 数据集**

实验使用ImageNet-100、ADE20K、Cityscapes、COCO-Stuff、LoveDA、ISPRS Potsdam以及Set5/Set14/BSD100/Urban100/Manga109等数据集。

**📈 对比分析**

与VMamba、SpectralVMamba、CNN/ViT等基线进行对比，指标为Top‑1/Top‑5准确率、mIoU、PSNR/SSIM和等变误差，结果表明EQ‑VMamba在准确率与参数量上均优于基线，并在旋转下保持近零等变误差。

**⚠️ 局限性**

限制在于目前仅针对p4四方向旋转，可扩展到更高阶或反射对称性；并且对极端旋转或非离散角度的等变性尚未验证。

---

## 239. Streaming Autoregressive Video Generation via Diagonal Distillation

**arXiv ID:** 2603.09488 | [PDF](https://arxiv.org/pdf/2603.09488v1)

**作者:** Jinxiu Liu `[一作]` (South China University of Technology), Weiyang Liu `[通讯]` (The Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Diagonal Distillation框架，实现高效自回归视频生成

**💡 创新点**

引入对角去噪与对角强制策略，结合光流分布匹配，显著提升时序一致性

**🔧 技术方法**

扩散模型蒸馏、对角注意力机制、Self‑Forcing KV缓存、光流匹配损失等技术

**📊 数据集**

使用Wan2.1‑T2V‑1.3B基线、VidProM文本提示、VBench评估指标

**📈 对比分析**

相较于Wan2.1、SkyReels‑V2、MAGI‑1、Causvid、Self‑Forcing，速度提升277.3×，质量约84.5分

**⚠️ 局限性**

在极少步或极长时序场景下仍可能出现运动衰减或细节退化

---

## 240. Robust Cooperative Localization in Featureless Environments: A Comparative Study of DCL, StCL, CCL, CI, and Standard-CL

**arXiv ID:** 2603.09886 | [PDF](https://arxiv.org/pdf/2603.09886v1)

**作者:** Nivand Khosravi `[一作]` (Instituto Superior Técnico), Rodrigo Ventura `[通讯]` (Instituto Superior Técnico)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对五种协同定位方法（CCL、DCL、StCL、CI、Standard-CL）在GPS失效、特征稀疏环境下进行仿真对比研究。

**💡 创新点**

首次揭示测量步幅（measurement stride）既能降低计算量又能提升鲁棒性；阐明忽略交叉相关导致的精度-一致性权衡；将协方差交叉（CI）定位定位为在一致性与精度之间的平衡点。

**🔧 技术方法**

采用扩展卡尔曼滤波（EKF）、协方差交叉（CI）、测量步幅、无交叉相关的顺序/标准协同更新，所有算法在ROS/Gazebo中实现。

**📊 数据集**

使用两机器人Gazebo仿真数据，产生真实轨迹并注入控制噪声与相对测量噪声；实验场景分为弱数据关联和鲁棒检测两种测量质量条件。

**📈 对比分析**

通过RMSE、NEES、NIS及计算时延等指标比较。结果显示StCL/Standard-CL精度最低但一致性最差；DCL在弱关联下最稳健；CI在保持一致性的同时精度仅次于StCL；CCL精度最好但对噪声敏感。

**⚠️ 局限性**

局限性包括仅测试两机器人小规模系统、仅使用仿真数据、步幅参数未自适应、缺乏真实世界多传感器融合验证。

---

## 241. OmniEdit: A Training-free framework for Lip Synchronization and Audio-Visual Editing

**arXiv ID:** 2603.09084 | [PDF](https://arxiv.org/pdf/2603.09084v1)

**作者:** Lixiang Lin `[一作]` (HiThink Research), Jinshan Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 25283 | [OpenAlex ID](https://openalex.org/A5002827290)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OmniEdit，一个训练无关的框架，用于口型同步与音视频编辑。

**💡 创新点**

创新点在于将FlowEdit的编辑序列改为目标序列以消除偏差，并去除随机采样构造确定、平滑的生成轨迹。

**🔧 技术方法**

使用预训练的音频到视频扩散模型（如Humo）、音视频基础模型（如LTX‑2）、Flow匹配、FlowEdit以及噪声估计等技术。

**📊 数据集**

在HDTF数据集和AIGC‑LipSync Benchmark上进行评估，音视频编辑部分以定性展示为主。

**📈 对比分析**

与Wav2Lip、IP‑LAP、Diff2Lip、MuseTalk、LatentSync、OmniSync等方法对比，OMNIEdit在FID、FVD、CSIM等指标上与最先进方法相当或略优，生成成功率略低于OmniSync。

**⚠️ 局限性**

局限性包括对全局场景和大规模风格改动支持不足，音频可能出现伪影，整体性能受限于基础模型，在风格化角色上的成功率仍有待提升。

---

## 242. Trade-Offs in FMCW Radar-Based Respiration and Heart Rate Variability

**arXiv ID:** 2603.09791 | [PDF](https://arxiv.org/pdf/2603.09791v1)

**作者:** Silvia Mura `[一作]` (Politecnico di Milano), Maurizio Magarini `[通讯]` (Politecnico di Milano)

**通讯引用:** 2781 | [OpenAlex ID](https://openalex.org/A5026125746)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本研究通过实验评估低成本FMCW MIMO雷达在非接触式心肺监测中的性能，系统分析了探测距离和发射波形数量对呼吸率(RR)与心率(HR)估计精度的影响；

**💡 创新点**

创新点在于首次量化雷达的操作折衷，尤其是对RR、HR及其变异性指标(HRV、BRV)的误差曲线进行系统性实验验证；

**🔧 技术方法**

采用Infineon BGT60TR13C毫米波雷达、基于LFM波形的多通道处理、两阶段波束选择与相位解包、频域心率检测以及时域变异性指标提取；

**📊 数据集**

使用自制的应变传感器和光学脉冲传感器提供RR与HR的地面真值，在30~150 cm距离、32~256个chirp的多种配置下收集2 min数据；

**📈 对比分析**

通过与真值的平均绝对误差(MAE)比较，发现最佳位置约70 cm、chirp≥96时RR误差≤1 bpm、HR误差≤3 bpm，而变异性指标误差仍在15–30%之间；

**⚠️ 局限性**

局限性包括仅在单个静态被试下实验、对短期动态变化缺乏评估、以及对HRV/BRV的高分辨率捕获仍不够精准。

---

## 243. Nonparametric Variational Differential Privacy via Embedding Parameter Clipping

**arXiv ID:** 2603.09583 | [PDF](https://arxiv.org/pdf/2603.09583v1)

**作者:** Dina El Zein `[一作]` (Idiap Research Institute), James Henderson `[通讯]` (Idiap Research Institute)

**通讯引用:** 3089 | [OpenAlex ID](https://openalex.org/A5084321238)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于理论的参数裁剪方法，限制非参数变分信息瓶颈（NVIB）后验分布的均值、方差和伪计数，以实现更紧的 Rényi 散度（RD）上界，进而提升变分差分隐私（NVDP）模型的隐私-效能平衡。

**💡 创新点**

创新点在于：①从 RD 上界推导出对后验参数的最优裁剪约束；②将裁剪作为正则化手段直接嵌入 NVDP 训练流程；③展示裁剪后模型在多任务上既能获得更好的隐私保障，又能提升或保持下游任务性能。

**🔧 技术方法**

核心技术包括：非参数变分信息瓶颈（NVIB）、Rényi 差分隐私（RDP）与贝叶斯差分隐私（BDP）理论、基于 Dirichlet 过程的后验建模、以及对后验参数进行的 L2、下界和区间裁剪。

**📊 数据集**

使用的数据集包括 GLUE 评测集（MRPC、STS-B、RTE、QNLI、SST‑2）以及 CommonLanguage 语音语言识别数据集，实验模型基于 BERT‑Base、BERT‑Large、RoBERTa‑Base 和 Wav2Vec2‑Large。

**📈 对比分析**

与无裁剪的 NVDP 基线和标准非隐私分类器对比；实验结果显示裁剪模型在绝大多数任务上既获得更低的 BDP/ RD 上界（隐私更好），又保持或提升准确率/ F1 分数，尤其在 STS‑B、RTE、QNLI 上显著提升。

**⚠️ 局限性**

限制在于裁剪参数需要手动设置（如 C_μ、C_α 上下界），需通过验证集进行网格搜索；裁剪对高资源任务的效能提升有限，且在极端小数据集上仍可能出现过拟合与隐私损失权衡不平衡的问题。

---

## 244. NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models

**arXiv ID:** 2603.09542 | [PDF](https://arxiv.org/pdf/2603.09542v1)

**作者:** Ziyue Zhu `[一作]` (Beijing University of Posts and Telecommunications), Haoran Luo `[通讯]` (Nanyang Technological University)

**通讯引用:** 1299 | [OpenAlex ID](https://openalex.org/A5101634507)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于神经符号化的 Vision‑Language‑Action（NS‑VLA）框架，利用在线强化学习实现机器人操控指令的视觉-语言-动作映射；

**💡 创新点**

创新点在于：① 将视觉与语言编码为离散符号原语以捕捉结构关系；② 设计轻量级符号求解器以逻辑推理生成动作；③ 通过在线RL实现主动探索，显著提升数据效率与泛化；

**🔧 技术方法**

核心技术包括：预训练视觉‑语言模型、符号编码器与分类器、视觉令牌稀疏化+Transformer动作生成器、基于组相对优化的RL策略；

**📊 数据集**

使用了 LIBERO、LIBERO‑Plus 与 CALVIN 三个机器人操作基准数据集，并对这些数据集进行原语标注；

**📈 对比分析**

与 OpenVLA、π0、UniVLA、VLA‑Adapter 等基线比较，NS‑VLA 在 1‑shot、全量训练及扰动环境下均实现了更高的成功率（约 85–88%），并在零样本、数据稀缺和探索空间方面表现优异；

**⚠️ 局限性**

主要限制包括：原语的手工定义、对真实世界仿真迁移的可扩展性不足，以及对动态环境变化的鲁棒性仍有提升空间。

---

## 245. EsoLang-Bench: Evaluating Genuine Reasoning in Large Language Models via Esoteric Programming Languages

**arXiv ID:** 2603.09678 | [PDF](https://arxiv.org/pdf/2603.09678v1)

**作者:** Aman Sharma `[一作]` (Lossfunk), Paras Chopra `[通讯]` (Lossfunk)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一套以极少见的 esoteric 编程语言为基准的代码生成评测，旨在检验大语言模型在离散分布（OOD）下的真实推理能力。

**💡 创新点**

创新点在于将数据极度稀缺、无游戏激励的 esoteric 语言作为评测对象，形成“不可游戏化”基准；同时提供跨范式、分层难度的题目，直接映射主流语言难题的算法本质。

**🔧 技术方法**

使用了 GPT‑5.2、O4‑mini、Gemini、Qwen、Kimi 等前沿 LLM，并试验了 Zero‑Shot、Few‑Shot、Self‑Scaffolding、Textual Self‑Scaffolding、ReAct 等提示策略；评测框架通过官方开源解释器实现自动化编译、执行与评分。

**📊 数据集**

数据集为 EsoLang‑Bench：80道题目，按 Easy/Medium/Hard/Extra‑Hard 四级分布，覆盖 Brainfuck、Befunge‑98、Whitespace、Unlambda、Shakespeare 五种语言，每题附 6 个 I/O 例子。

**📈 对比分析**

对比结果显示：在 Python 上模型可达 85‑95% 正确率，转移到 esoteric 语言时最高仅 11%（GPT‑5.2 在最优策略下 6.2%），所有模型在 Medium 及以上难度全为 0%；提示策略提升有限，agentic 系统略胜传统方法。

**⚠️ 局限性**

局限性包括：大多数任务仅在 Easy 层可解，难度层无法区分模型；只覆盖五种 esoteric 语言，可能未完全覆盖所有范式；误差分类粗略，无法深入剖析具体失败原因；Whitespace 的 tokenizer 兼容性问题可能导致编译失败。

---

## 246. Benchmarking Federated Learning in Edge Computing Environments: A Systematic Review and Performance Evaluation

**arXiv ID:** 2603.08735 | [PDF](https://arxiv.org/pdf/2603.08735v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 247. Quantifying and extending the coverage of spatial categorization data sets

**arXiv ID:** 2603.09373 | [PDF](https://arxiv.org/pdf/2603.09373v1)

**作者:** Wanchun Li `[一作]` (University of Melbourne), Charles Kemp `[通讯]` (University of Melbourne)

**通讯引用:** 8853 | [OpenAlex ID](https://openalex.org/A5080087902)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用大型语言模型（LLM）自动生成空间关系标签，并以此评估并扩展Topological Relations Picture Series（TRPS）的场景与语言覆盖度。通过对220幅图像在23种语言中的LLM标签进行相似度计算，提出覆盖度公式，用于筛选最具价值的场景和语言；随后新增42个图像并与已有的Zhang和LJSP扩展集进行比较。

**💡 创新点**

创新点在于：①首次将LLM作为“人类实验参与者”直接对图像进行空间关系标注，并验证其与真实人类标签的一致性；②提出基于标签相似度的覆盖度度量，可用于自动化评估并指导数据集扩充；③展示如何利用LLM标签在多语言环境下快速识别需优先收集人类标签的场景与语言，从而显著提升空间语义数据库的规模与多样性。

**🔧 技术方法**

技术手段包括：使用Gemini 3 Flash LLM（多语言生成）；针对不同语言的Prompt设计；文本和图像双条件实验（含图像缺失时的纯文本条件）；标签相似度计算（匹配即1/不匹配即0）与平均最大相似度公式求覆盖度；多维尺度分析（MDS）可视化空间关系；Bootstrap方法估计覆盖度置信区间；Google Translate API用于扩展语言范围。

**📊 数据集**

数据集：原始TRPS 71幅图；Zhang集 63幅图；LJSP集 44幅图；自制扩展集 42幅图；合计220幅图；人类标签来自Carstensen等人的多语言实验（7种语言，每种至少13人）和单一说话者的21语言实验；共23种语言；同时使用Google Translate支持的249种语言作为潜在扩展上限。

**📈 对比分析**

比较方法：与人类标签对比计算二元分数（至少有一人相同）和分级分数（人类给出该标签的比例）。LLM在大多数语言上二元分数>0.9，分级分数与最大可能分数差距≤0.15。覆盖度评估显示：TRPS 0.914，TRPS+Zhang 0.918，TRPS+LJSP 0.918，TRPS+扩展 0.964，且Bootstrap CI表明扩展集显著优于其他两种扩展。图像缺失的文本条件与图像条件得到相近结果，说明图像对LLM性能影响不大。

**⚠️ 局限性**

局限性：①LLM主要适用于高资源语言，低资源语言的标签质量难以保证；②标签相似度覆盖度度量未必能捕捉空间关系的系统性覆盖，缺乏基于特征的全面覆盖；③实验仅使用单一LLM模型，结果可能受模型偏差影响；④图像信息对标签影响有限，说明LLM更多依赖语言知识；⑤扩展语言受Google Translate支持范围限制，未覆盖全球语言多样性。

---

## 248. GeoBenchr: An Application-Centric Benchmarking Suite for Spatiotemporal Database Platforms

**arXiv ID:** 2603.09398 | [PDF](https://arxiv.org/pdf/2603.09398v1)

**作者:** Tim C. Rese `[一作]` (TU Berlin), David Bermbach `[通讯]` (TU Berlin)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GeoBenchr，一个面向应用的时空数据库基准套件，用于在多种数据规模、配置和工作负载下评估 PostGIS、TimescaleDB、MobilityDB、SedonaDB、SpaceTime 等平台的性能。

**💡 创新点**

创新点在于：①将真实业务场景（航空、骑行、AIS）与多样化的 MOD 数据集结合，②提供可插拔的查询翻译与参数生成器，使不同语法的系统能共享相同的基准任务，③实现了跨平台、跨配置的可复现实验流程。

**🔧 技术方法**

使用了 SQL/SQL‑dialect 转换、YAML 配置、Python/Go 脚本、Docker 容器化部署、内存/磁盘索引（GIST、SP‑GIST、空间/时间分区）、以及多线程并发执行。

**📊 数据集**

利用三大真实数据集：SimRa 乘骑轨迹（约 45M 点）、DFS 航班数据（约 100M 点）、Piraeus AIS 航运数据（约 12M 点）以及相应的辅助几何文件。

**📈 对比分析**

通过在统一硬件上对同一工作负载做三次跑测并取中位数，使用查询延迟、吞吐量、CPU/内存占用等指标比较；结果显示 SedonaDB 在大部分查询上最快，但 SpaceTime 在某些高并发或大规模场景下仍能竞争；分区和索引的调优对 MobilityDB 有显著影响。

**⚠️ 局限性**

局限性包括：仅覆盖单节点系统，未评估分布式扩展；实验基准受限于所选数据集和查询类型，可能无法代表所有实际业务；对硬件依赖较强，成本/内存比例未被量化；部分系统配置未必是最优，未来需要更系统的调优框架。

---

## 249. TubeMLLM: A Foundation Model for Topology Knowledge Exploration in Vessel-like Anatomy

**arXiv ID:** 2603.09217 | [PDF](https://arxiv.org/pdf/2603.09217v1)

**作者:** Yaoyu Liu `[一作]` (Shanghai Jiao Tong University), Yun Gu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5244 | [OpenAlex ID](https://openalex.org/A5100636598)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了TubeMLLM统一多模态基础模型，结合语言提示实现血管结构的拓扑保真分割与理解。

**💡 创新点**

通过显式自然语言拓扑先验、共享注意力架构和自适应损失权重，解决传统模型的拓扑错误和跨模态泛化不足。

**🔧 技术方法**

融合大型语言模型、VAE生成分支与理解分支的混合Transformer、rectified flow流匹配训练以及跨模态共享注意力。

**📊 数据集**

构建了TubeMData，包括10个彩色眼底照片和5个X射线血管数据集，共约3.5K图像与52K样本，用于拓扑生成与理解任务。

**📈 对比分析**

与nnUNet、SAM系列及专门血管模型等基线在15个数据集上对比，TubeMLLM在Dice、clDice及β₀错误等指标显著提升，零样本跨模态转移时β₀从238.26降至1.21，且对低质量输入保持鲁棒。

**⚠️ 局限性**

仍受限于预训练视觉特征，需大量多模态数据，且对极端噪声或极低分辨率输入的细节仍可能出现误差；缺乏对其他解剖结构的泛化验证。

---

## 250. The FABRIC Strategy for Verifying Neural Feedback Systems

**arXiv ID:** 2603.08964 | [PDF](https://arxiv.org/pdf/2603.08964v1)

**作者:** I. Samuel Akinwande `[一作]` (Stanford University), Clark Barrett `[通讯]` (Stanford University)

**通讯引用:** 10165 | [OpenAlex ID](https://openalex.org/A5026961968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种前向与后向可达性分析相结合的FaBRIC策略，用以验证非线性神经反馈系统的到达-避免规范；

**💡 创新点**

创新点在于开发了多种后向可达集的外近似和内近似算法（包括Polyhedral封闭、域细化、DRiPy等），并首次将它们与前向分析有效集成；

**🔧 技术方法**

使用了混合整数线性规划（MILP）、CROWN框架、polyhedral enclosure、域细化、采样+凸优化等技术，并借助Gurobi求解器实现求解；

**📊 数据集**

实验基于ARCH Competition基准（TORA、Unicycle、Attitude）三种神经反馈系统，训练不同规模（small/medium/large）的控制器；

**📈 对比分析**

与HyBreach‑MILP（外近似）、BURNS（内近似）以及现有前向分析工具对比，后向算法在体积上显著小于HyBreach‑MILP，尤其在复杂系统上；FaBRIC相比仅前向方法在Unicycle上可提升3–7倍，整体验证时间显著下降；

**⚠️ 局限性**

限制在于仍受维数灾难影响，内近似对样本稀疏敏感，部分基准下BURNS/内近似方法失败；后向分析仍需多次求解，且F/B比例需手工调节。

---

## 251. Let's Reward Step-by-Step: Step-Aware Contrastive Alignment for Vision-Language Navigation in Continuous Environments

**arXiv ID:** 2603.09740 | [PDF](https://arxiv.org/pdf/2603.09740v1)

**作者:** Haoyuan Li `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 81394 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Step‑Aware Contrastive Alignment (SACA) 框架，用 Perception‑Grounded Step‑Aware (PGSA) 审计器从失败轨迹中提取稠密的 step‑级监督，解决稀疏奖励导致的学习信号崩塌；通过场景条件组构造机制动态切换修复重采样和全失败救援，提升错误恢复与训练稳定性。

**💡 创新点**

创新点包括：①零射击 PGSA 审计器利用 LLM、GroundingDINO、SAM3、CLIP 等多模基础模型生成连续软评分与离散结构掩码；②场景条件组构造在 mixed‑group 与 all‑failure 两种情形下分别采用 Repair Resampling 与 All‑Failure Rescue；③在优化目标中融合轨迹级优势与 step‑级一致性对齐与对比纠正，形成鲁棒的 SACA 优化策略。

**🔧 技术方法**

使用技术包括：零射击大型语言模型（如 Qwen3‑0.6B）解析中间地标；GroundingDINO+SAM3+CLIP 进行视觉定位与语义对齐；GRPO 强化学习框架；PGSA 审计器、Repair Resampling、All‑Failure Rescue 以及多项损失（行为克隆、对比修正等）。

**📊 数据集**

在 R2R‑CE 与 RxR‑CE 两大 Vision‑Language Navigation in Continuous Environments（VLN‑CE）基准（Matterport3D 场景）上进行评估；同时扩展使用 ScaleVLN 数据集验证。

**📈 对比分析**

与现有方法（如 StreamVLN、VLN‑R1、ETPNav 等）对比，SACA 在 R2R‑CE 的 SR/SPL 提升约 7%/8%，在 RxR‑CE 上提升 11%/7%；在单 RGB 观测下已超越多传感器融合方案，且在长航程任务上表现尤为突出。

**⚠️ 局限性**

局限性：PGSA 的阈值与权重对性能敏感，需手工调参；对极长轨迹或极端失误的离散掩码可能产生误判；目前验证主要基于模拟环境，实际物理世界的鲁棒性仍待进一步验证。

---

## 252. TiPToP: A Modular Open-Vocabulary Planning System for Robotic Manipulation

**arXiv ID:** 2603.09971 | [PDF](https://arxiv.org/pdf/2603.09971v1)

**作者:** William Shen `[一作]` (Massachusetts Institute of Technology), Tomás Lozano-Pérez `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 39706 | [OpenAlex ID](https://openalex.org/A5108257764)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了TiPToP，一个基于预训练视觉/语言基础模型和GPU加速TAMP的模块化机器人操作系统，能够仅凭RGB图像和自然语言指令无训练数据完成多步抓取、放置等操作；

**💡 创新点**

创新点在于将视觉基础模型、语义归纳与任务与运动规划无缝融合，形成可快速迁移、逐组件升级、无数据训练的完整系统；

**🔧 技术方法**

技术包括FoundationStereo（深度估计）、M2T2（6-DoF抓取预测）、Gemini VLM（目标检测与语义归纳）、SAM/SAM‑2（分割）、cuTAMP（GPU并行TAMP）、cuRobo运动规划，以及基于符号规划的目标建模；

**📊 数据集**

使用公开预训练模型，无需收集机器人演示数据；评估基于28个真实与仿真桌面任务场景，涵盖日常物品；

**📈 对比分析**

与π_0.5‑DROID（在350小时演示上微调的VLA模型）比较，TiPToP在多步骤、干扰、语义任务上的成功率约为74.6%，显著高于52.4%；单步任务相近；平均执行时间约为π_0.5‑DROID的70%，更快；

**⚠️ 局限性**

主要局限在于无闭环执行导致抓取失败率高；单视角感知导致几何误差；对小物体或易滑落物体鲁棒性不足；需要提升抓取、形状重建和观测反馈。

---

## 253. MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents

**arXiv ID:** 2603.09827 | [PDF](https://arxiv.org/pdf/2603.09827v1)

**作者:** Kangsan Kim `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出多主体长时 egocentric 视频问答基准 MA‑EgoQA，并给出评测与基线

**💡 创新点**

首次把多主体的长时 egocentric 视频聚合用于问答，提出共享记忆与动态检索的 EgoMAS 框架

**🔧 技术方法**

采用多模态检索（BM25、DPR 等）、4W1H 事件记忆、LLM 生成、视频 LLM 预训练模型等技术

**📊 数据集**

基于 EgoLife 数据集（6 个人 7 天共 1.7k Q‑A 对）

**📈 对比分析**

与多种 LLM、视频 LLM 及 RAG 基线对比，最强基线仅 36.9% 准确率，EgoMAS 在 Gemini‑2.5‑Flash 背景下达到 41.4%

**⚠️ 局限性**

模型仍难以处理多主体推理和 Theory‑of‑Mind 类问题，缺乏高效的跨主体知识融合与长时记忆检索机制

---

## 254. Physics-informed neural operator for predictive parametric phase-field modelling

**arXiv ID:** 2603.09693 | [PDF](https://arxiv.org/pdf/2603.09693v1)

**作者:** Nanxi Chen `[一作]` (Tongji University), Rujin Ma `[通讯]` (Tongji University)

**通讯引用:** 1614 | [OpenAlex ID](https://openalex.org/A5089037577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于物理信息的傅里叶神经算子（PF-PINO）框架，用于学习参数化的相场模型的时间演化，能够在推断时通过递归一次步映射生成完整的空间-时间解。

**💡 创新点**

创新点在于将PDE残差直接嵌入到神经算子训练的损失函数中，结合梯度归一化权重平衡与自回归推理，显著提升了模型在稀缺数据、非周期边界、尖锐界面等苛刻条件下的物理一致性和泛化能力。

**🔧 技术方法**

采用的技术包括：傅里叶神经算子（FNO）作为基础架构、谱卷积与线性旁路相结合、自动微分与有限差分/谱差分计算PDE残差、梯度归一化损失平衡、交替训练与自回归微调、以及多尺度损失组合。

**📊 数据集**

使用四个人工高精度数值数据集：一维笔电极腐蚀、二维电化学抛光腐蚀、二维树枝晶固化、以及二维自发相分离（Spinodal decomposition），每个数据集涵盖多种参数变化（如界面动力学系数、初始界面形状、潜热系数、迁移率和扰动幅度）。

**📈 对比分析**

与纯数据驱动的FNO基准进行对比，评价指标为相对L2误差和相对Hausdorff距离；结果显示PF-PINO在所有四个基准中均实现了至少30%–60%的误差下降，长期自回归推理中误差累积更稳定，尤其在外推参数区间表现突出；在推断速度上两者相当，但PF-PINO的训练阶段成本相对更高。

**⚠️ 局限性**

局限性包括：仍需一定量的高精度仿真数据作为监督；训练时需要手工设计PDE残差权重和差分/谱算子；在极高阶导数或极尖锐界面情况下，自动微分或差分误差会增大；目前仅验证于1D/2D问题，尚未扩展到3D高维相场模拟；对超大规模网格的内存与计算开销仍有待进一步优化。

---

## 255. Towards Unified Multimodal Interleaved Generation via Group Relative Policy Optimization

**arXiv ID:** 2603.09538 | [PDF](https://arxiv.org/pdf/2603.09538v1)

**作者:** Ming Nie `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**通讯引用:** 46502 | [OpenAlex ID](https://openalex.org/A5100461206)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种后训练策略，利用少量的混合数据让统一视图语言模型实现文本与图像的交错生成；

**💡 创新点**

创新点在于将Group Relative Policy Optimization (GRPO)扩展到多模态交错生成，并设计了联合奖励和过程级奖励以实现细粒度对齐；

**🔧 技术方法**

使用了GRPO强化学习框架、混合奖励（文本、视觉、格式）、过程级奖励以及KL约束；

**📊 数据集**

使用的训练集包括0.3M交错文本-图像样本（ActivityNet、GenHowTo、OpenStory++）和1M理解/生成样本（EMOVA、JourneyDB）；

**📈 对比分析**

在MMIE和InterleavedBench上与现有统一模型对比，提升至59.5%（MMIE）和3.13（InterleavedBench），显著优于基线且保持原有能力；

**⚠️ 局限性**

局限在于对通用多模态理解与单向生成任务提升有限，主要受限于基础模型性能与奖励设计的局限性。

---

## 256. Component-Aware Sketch-to-Image Generation Using Self-Attention Encoding and Coordinate-Preserving Fusion

**arXiv ID:** 2603.09484 | [PDF](https://arxiv.org/pdf/2603.09484v1)

**作者:** Ali Zia `[一作]` (La Trobe University), Shahnawaz Qureshi `[通讯]` (Pak-Austria Fachhochschule Institute of Applied Sciences and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向组件的自我改进框架，将手绘草图映射为逼真图像。

**💡 创新点**

创新点在于：① 基于自注意力的组件编码器实现局部语义分离；② 坐标保持门控融合(CGf)保证空间一致性；③ 采用空间自适应细化重写器(Sarr)在StyleGAN2基础上迭代提升细节与真实性。

**🔧 技术方法**

技术栈包括：自注意力编码器（SA2N）、坐标保持门控融合模块（CGF）、空间自适应细化重写器（SARR）+StyleGAN2、L1/对抗/感知/Gram矩阵损失，以及多尺度特征提取与梯度更新。

**📊 数据集**

使用人脸数据集 CelebAMask‑HQ、CUHK、CUFSF 以及非人脸数据集 Sketchy、ChairsV2、ShoesV2 进行训练与评测。

**📈 对比分析**

与 CycleGAN、Pix2PixHD、pSp、DFD、OME（GAN）以及 ControlNet、T2I‑Adapter（扩散）等基线比较，实验表明在 FID、IS、KID、SSIM、MOS 等指标均优于现有方法，CelebAMask‑HQ 上 FID 下降 21%、IS 提升 58%、KID 降 41%、SSIM 提升 20%。

**⚠️ 局限性**

局限性：对极度稀疏或含噪的草图仍可能缺失细节（如痣、细纹），在纹理复杂的对象上偶尔出现轻微平滑；模型依赖大量标注数据，难以处理完全陌生或极端风格的草图。

---

## 257. Simultaneous Embedding of Two Paths on the Grid

**arXiv ID:** 2603.09750 | [PDF](https://arxiv.org/pdf/2603.09750v1)

**作者:** Stephen Kobourov `[一作]` (Technical University of Munich), Johannes Zink `[通讯]` (Technical University of Munich)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5042290109)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在整数网格上两条路径的同时几何嵌入问题，证明在最小化最长边长度或总边长度时为NP‑难，并给出了当一条路径为x‑单调、另一条为y‑单调时，最小化包围盒周长的O(n^3/2)多项式时间算法。

**💡 创新点**

首次证明了两路径嵌入优化（最长边、总长度）为NP‑难；同时将嵌入问题转化为最小顶点覆盖并利用Hopcroft–Karp求最大匹配，得到高效的O(n^3/2)算法。

**🔧 技术方法**

使用了布尔约束与图论模型（构造约束图C_G）、最大匹配与最小顶点覆盖算法、Kőnig定理、图的二分性与偶环性质，以及逻辑引擎的构造技术进行NP‑硬性证明。

**📊 数据集**

该研究主要为理论性分析，未使用具体数值数据集；所有实验均为算法复杂度和可行性证明，未进行实际数据集上的实验。

**📈 对比分析**

与现有的仅能得到非最优嵌入的Brass等算法对比，提出的O(n^3/2)算法在单调约束下可在多项式时间内获得最小周长嵌入；在最优长度优化方面证明了不可多项式时间求解，表明已知算法已是最优的理论上限。

**⚠️ 局限性**

局限性包括：①仅考虑两条路径，不能推广到更多路径或更一般的平面图；②最优长度优化仅在NP‑难框架内给出复杂度上限；③单调约束下的O(n^3/2)算法仍有进一步改进空间，且对特殊结构（如稠密路径）可能效率不够理想。

---

## 258. RubiCap: Rubric-Guided Reinforcement Learning for Dense Image Captioning

**arXiv ID:** 2603.09160 | [PDF](https://arxiv.org/pdf/2603.09160v1)

**作者:** Tzu-Heng Huang `[一作]` (University of Wisconsin), Manjot Bilkhu `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于自定义评价 rubrics 的强化学习框架 RubiCap，用来提升密集图像字幕质量并降低知识遗忘；

**💡 创新点**

通过教师 VLM 委员会自动生成样本特定的、可解释的评价标准（rubrics），将其转化为细粒度奖励信号，突破了 RL 在无确定验证器任务中的验证瓶颈；

**🔧 技术方法**

结合多模态大型语言模型（LLM）作为 rubrics 编写者与评审者、教师 VLM 委员会、以及基于 Group Relative Policy Optimization (GRPO) 的 RL 训练；

**📊 数据集**

使用 PixMoCap（含人类专家字幕）和 DenseFusion-4V-100K（GPT‑4V 生成字幕）两个密集字幕数据集进行训练与评测；

**📈 对比分析**

在 CapArena（GPT‑4.1 评判）、METEOR、BLEU、ROUGE‑L 等多指标下，RubiCap‑7B 在 7B 规模上取得最高 win‑rate、最小 hallucination 处罚，甚至在盲判中超过 72B、32B 前沿模型；在词数限制下表现出比更大模型更高的信息密度；在 VLM 预训练阶段使用 RubiCap 生成的字幕可使 3B‑7B 模型在多项评测上优于 GPT‑4V 生成的字幕；

**⚠️ 局限性**

仍然受限于教师 VLM 的多样性与可靠性、LLM 对 rubrics 的生成与评判错误、以及 RL 训练所需的大规模算力；

---

## 259. On the Multi-Commodity Flow with convex objective function: Column-Generation approaches

**arXiv ID:** 2603.08714 | [PDF](https://arxiv.org/pdf/2603.08714v1)

**作者:** Guillaume Beraud-Sudreau `[一作]` (Huawei Technologies Ltd), Sébastien Martin `[通讯]` (Huawei Technologies Ltd)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种通用的算法框架，用于在带有凸增量成本函数的有容量多商品流（CMCF）问题中求解可分（splittable）和不可分（unsplittable）两种路由方案；

**💡 创新点**

创新点包括：①利用内逼近（inner‑approximation）对凸成本进行线性化，兼容非可微或黑箱成本；②设计了多层次的列生成方法（compact、Simplicial Decomposition、inner‑approximation）并证明其在给定精度下可多项式求解；③针对不可分问题引入模式变量（pattern）和无分支约束加强（TIGHT‑INNERR）实现高效的分支定价；

**🔧 技术方法**

技术手段主要包括列生成（column‑generation）、KKT 条件推导的定价问题、内逼近多边形构造、分支定价（branch‑and‑price）、约束松弛与强化、以及对黑箱成本的函数值评估；

**📊 数据集**

实验数据集使用了公共的 SNDLib（欧洲/美国等网络实例），其中包含数百条链路和数百至数千条需求，且对线性、二次以及 Kleinrock 型凸成本进行了评测；

**📈 对比分析**

方法比较：在可分问题上，inner‑approximation（INNERR）比 compact 和 ConveX Decomposition（CONVEX）在求解时间上提升约10-35倍；在不可分问题上，pattern‑based 方案（PATTTERN）在分支定价树上能解决大部分实例，而 TIGHT‑INNERR 在根节点即可得到较好的下界，但在大型实例中仍需较长时间；

**⚠️ 局限性**

局限性：①不可分问题依旧是 NP‑hard，branch‑and‑price 对极大实例仍可能爆炸；②模式变量的指数增长使得模式生成的二次背包子问题在大规模时成本高；③内逼近需要不断生成新的顶点，若成本函数极其复杂或高维，列生成过程会变得冗长；

---

## 260. Task Aware Modulation Using Representation Learning for Upsaling of Terrestrial Carbon Fluxes

**arXiv ID:** 2603.09974 | [PDF](https://arxiv.org/pdf/2603.09974v1)

**作者:** Aleksei Rozanov `[一作]` (University of Minnesota), Vipin Kumar `[通讯]` (University of Minnesota)

**通讯引用:** 43307 | [OpenAlex ID](https://openalex.org/A5100645812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了TAM-RL框架，实现零样本地表碳通量上采样。

**💡 创新点**

创新点在于结合任务感知调制与表示学习，并通过碳平衡方程引入知识引导损失，提升跨域泛化。

**🔧 技术方法**

采用LSTM编码器-解码器、FiLM调制、复合损失以及多任务学习。

**📊 数据集**

使用579个EC站点的每日NEE、GPP以及MODIS、ERA5-Land等卫星与气候驱动数据。

**📈 对比分析**

与FLUXCOM-X-BASE、XGBoost、CT-LSTM等基线对比，RMSE下降约8–9.6%，R²提升19.4–43.8%。

**⚠️ 局限性**

在水体和部分森林类型表现仍差，空间与气候异质性导致误差波动，缺乏水体特征。

---

## 261. Paralinguistic Emotion-Aware Validation Timing Detection in Japanese Empathetic Spoken Dialogue

**arXiv ID:** 2603.09307 | [PDF](https://arxiv.org/pdf/2603.09307v1)

**作者:** Zi Haur Pang `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**通讯引用:** 7003 | [OpenAlex ID](https://openalex.org/A5038044080)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种同时利用声学旁语特征和情感信息的语音模型，用于检测在心理治疗对话中何时表达情感验证。

**💡 创新点**

创新点在于：①通过在 HuBERT 上继续自监督预训练提取旁语特征；②采用多任务学习同时预测情感类别和情绪极性；③将两条分支的语音表示进行融合，并在不使用文本上下文的前提下实现验证时机检测。

**🔧 技术方法**

使用的技术包括 HuBERT-Large 作为语音编码器、Mask‑Unit 预训练、情感与情绪多任务头、特征级拼接融合、LoRA 参数高效微调以及交叉熵与加权损失优化。

**📊 数据集**

采用的主要数据集：MELD‑ST（用于情感识别预训练）、JVNV（用于旁语自监督预训练）和 TUT Emotional Storytelling Corpus (TESC，验证时机检测)。

**📈 对比分析**

与传统语音基线（HuBERT、xlsr‑53）以及多种语言模型（BERT、ModernBERT、Llama 3.1 8B、GPT‑4.1 Nano）进行比较，所提方法在验证类精度 47.96%、验证类 F1 54.34% 以及宏观 F1 62.37% 上均明显优于所有基线。

**⚠️ 局限性**

局限性包括：1）仅在日语、对话式情感数据上验证，跨语言与跨场景的泛化尚未评估；2）依赖的旁语与情感特征可能无法捕捉更复杂的语境信息；3）数据集规模有限，模型对噪声或极端口音的鲁棒性尚待验证。

---

## 262. Progressive Representation Learning for Multimodal Sentiment Analysis with Incomplete Modalities

**arXiv ID:** 2603.09111 | [PDF](https://arxiv.org/pdf/2603.09111v1)

**作者:** Jindi Bao `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 43669 | [OpenAlex ID](https://openalex.org/A5100726984)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种名为 PRLF 的进化式表示学习框架，用以解决多模态情感分析中存在的缺失模态问题。

**💡 创新点**

创新点包括：① 通过 Adaptive Modality Reliability Estimator (AMRE) 将识别置信度与 Fisher 信息矩阵结合，动态评估每个模态的可靠性；② 采用 Progressive Interaction (ProgInteract) 逐步将辅助模态特征对齐到主模态，降低噪声并提升跨模态一致性。

**🔧 技术方法**

使用了多模态编码器、分类头、Fisher 信息矩阵计算、动态加权融合、阶段化自我完善机制、相位一致性损失以及梯度信息融合等技术。

**📊 数据集**

实验数据集包括 CMU-MOSI、CMU-MOSEI 和 SIMS 三大基准。

**📈 对比分析**

与 Self-MM、MISA、TETFN、MMIM、UMDF、HRLF、CorrKD、EMOE 等方法对比，PRLF 在完整和缺失模态场景下均实现了最优或接近最优的 F1/准确率，表现出更强的鲁棒性与泛化能力。

**⚠️ 局限性**

局限性：在极高缺失率下性能仍会显著下降；缺失模式的生成需要人工设计，且实验仅覆盖三模态（文本、语音、视觉），对更多模态或不同领域的适用性仍需进一步验证。

---

## 263. Progressive Split Mamba: Effective State Space Modelling for Image Restoration

**arXiv ID:** 2603.09171 | [PDF](https://arxiv.org/pdf/2603.09171v1)

**作者:** Mohammed Hassanin `[一作]`, Ibrahim Radwan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 Progressive Split-Mamba (PS-Mamba) 的图像恢复框架，通过在空间上递进拆分特征图并在每个子块内使用 Mamba 状态空间模型，实现局部细节和全局一致性的统一建模。

**💡 创新点**

创新点包括：① 采用拓扑保持的多级拆分（半分、四分、八分）避免了将二维特征线性化导致的邻接破坏；② 设计对称的跨尺度跳连通，直接传递低频全局信息，缓解状态空间模型的长程衰减；③ 在拆分子块内引入卷积预处理和注意力融合，进一步强化局部纹理与全局语义的协同。

**🔧 技术方法**

核心技术包括：基于 Mamba 的线性时间状态空间网络；几何一致的拆分与合并算子；卷积预处理与双重注意力（通道+空间）融合；对称跨尺度跳连通；使用 L1/Charbonnier 损失进行统一监督。

**📊 数据集**

在多项公开基准上验证：图像超分辨率（Set5、Set14、BSDS100、Urban100、Manga109）、JPEG 伪影去除（Classic5、LIVE1、HQ-Real）以及彩色图像去噪（CBSD68、Kodak24、McMaster、Urban100）等数据集。

**📈 对比分析**

与现有轻量级和经典的卷积、Transformer 与 Mamba 变体（CARN、LatticeNet、SwinIR-light、ELAN、SRFormer-light、MambaIR、MambaIRv2、EDSR、RCAN、SAN、IPT、HAT、DAT、CAT-A 等）进行对比。PS-Mamba 在同等参数量或更低 MACs 下，在 PSNR/SSIM 上均实现显著提升，尤其在 2×/3×/4× 超分和 JPEG 低质量场景下取得领先；在彩色去噪任务中也获得最高 PSNR。

**⚠️ 局限性**

限制：拆分与合并操作增加实现复杂度；对非常大尺寸图像的拆分粒度需手动调节，可能导致边界效应；跨尺度跳连通虽然缓解衰减，但在极深网络中仍可能存在信息瓶颈。

---

## 264. Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition on Static Hand Keypoints

**arXiv ID:** 2603.09213 | [PDF](https://arxiv.org/pdf/2603.09213v1)

**作者:** Chayanin Chamachot `[一作]` (Chulalongkorn University), Kanokphan Lertniponphan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究跨语言少样本手语识别，提出几何不变角度特征并在四种指字母表上进行系统评估。

**💡 创新点**

引入20维SO(3)不变的手部角度描述子，消除视角、尺度等域漂移，显著提升跨语言few‑shot性能。

**🔧 技术方法**

利用MediaPipe手部关键点、角度特征、MLP/Transformer编码器、Prototypical Networks及SupCon预训练，支持冻结与微调两种跨语言迁移方式。

**📊 数据集**

使用ASL、LIBRAS、Arabic SL、Thai Fingerspelling四个手语指字母数据集。

**📈 对比分析**

与坐标归一化、原始坐标和角度+坐标拼接等表示进行对照实验；在5-shot设置下角度特征提升约25pp，冻结跨语言迁移甚至超过同域基准。

**⚠️ 局限性**

仅限静态单手指字母，缺乏时序信息；角度特征忽略骨长等绝对尺寸信息，且在更大类别或动态手语中的效果未知。

---

## 265. SCENEBench: An Audio Understanding Benchmark Grounded in Assistive and Industrial Use Cases

**arXiv ID:** 2603.09853 | [PDF](https://arxiv.org/pdf/2603.09853v1)

**作者:** Laya Iyer `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**通讯引用:** 5044 | [OpenAlex ID](https://openalex.org/A5091266570)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SCENEBench 评测框架，针对背景声识别、噪声定位、跨语音识别与非语音情感识别四类实际场景的音频理解能力进行评测；

**💡 创新点**

创新点在于：1) 设计四大任务覆盖现实应用难点；2) 结合免费文本描述、后续问答与多项选择三层评分，细致分析模型错误类型；3) 通过人工与自然样本验证生态效度；

**🔧 技术方法**

技术包括：多模态大语言模型调用（GPT‑4o、Gemini‑1.5、Qwen2‑Audio‑7B、Audio‑Flamingo‑3、DeSTA2‑8B‑beta），自定义提示工程、自动化文本对齐与同义词匹配，基于音频合成与混合生成的合成数据与自然采样；

**📊 数据集**

使用的主要数据集有 ESC‑50、DailyTalk、LibriMix、ToyADMOS、MIMII、Nonspeech7k、CapSpeech‑AgentDB‑Audio 等，通过合成叠加、音量包络变化与语言转换产生四类任务样本；

**📈 对比分析**

评测方法：对每个模型在五个子任务上进行免费描述、后续提问与多项选择评分，计算准确率与置信区间；在自然样本上验证排名稳定性；性能表现：背景声识别中 Flamingo 与 Qwen2 最高；噪声定位中 Qwen2 与 GPT‑4o 获得显著提升；跨语音识别中 GPT‑4o 与 Gemini 按语言各领风骚；非语音识别中 Flamingo 在大多数类别达到 95%+；整体来看模型均低于“完美”水平，仍有显著提升空间；

**⚠️ 局限性**

局限性包括：1) 合成混合采用等响度叠加，缺乏真实信噪比分布；2) 语音合成与回译过程可能缺乏自然对话多样性；3) API 模型无法完整记录延迟；4) 数据集中部分标注存在误差，影响上限估计；5) 主要聚焦四类任务，未覆盖更复杂的多源、多声道情境。

---

## 266. ProvAgent: Threat Detection Based on Identity-Behavior Binding and Multi-Agent Collaborative Attack Investigation

**arXiv ID:** 2603.09358 | [PDF](https://arxiv.org/pdf/2603.09358v1)

**作者:** Wenhao Yan `[一作]` (Institute of Information Engineering), Cong Dong `[通讯]` (Zhongguancun Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合身份-行为绑定与多智能体协同的APT检测与追溯框架ProvAgent，能够在大规模审计日志中快速生成高质量告警，并通过多智能体的假设-验证循环实现完整攻击链的自动重建。

**💡 创新点**

核心创新点包括：①利用图对比学习实现细粒度身份-行为绑定，显著降低误报；②构建多智能体调查模块（Analyst、Investigator、Leader、Reporter）并以假设-验证闭环驱动主动探测；③将传统模型与LLM协同工作，兼顾检测效率与推理深度。

**🔧 技术方法**

技术手段主要有：图神经网络（GNN）用于节点嵌入；图对比学习与信息熵约束实现身份一致性学习；多智能体架构与基于LLM的检索增强生成（RAG）；图构建、节点特征编码（语义、动作分布、时序）等。

**📊 数据集**

使用公开的企业级审计数据集，包括DARPA Transparent Computing Engagement 3 (E3)、Engagement 5 (E5) 以及OPTC；此外在评估调查性能时采用CADETS与THEIA子集。

**📈 对比分析**

与七个SOTA基线（Kairos、ThreaTrace、Flash、MAGIC、Orthrus、OCR‑APT、Threatrace）在E3/E5/OPTC上进行对比。ProvAgent在TP、F1、误报率等指标上均优于所有基线；在调查阶段IOC提升达160%并在日均成本仅$0.06，展示了高效低成本的全链路能力。

**⚠️ 局限性**

局限性包括：①需先验的无攻击基线日志用于身份特征学习，恶意植入前数据不足时效果受限；②假设每天为单一APT活动窗口，无法跨日关联多场景或长周期攻击；③依赖预定义的kill‑chain框架，对新型攻击路径或重排的阶段识别可能出现不适应。

---

## 267. Flash-KMeans: Fast and Memory-Efficient Exact K-Means

**arXiv ID:** 2603.09229 | [PDF](https://arxiv.org/pdf/2603.09229v1)

**作者:** Shuo Yang `[一作]` (University of California Berkeley), Ion Stoica `[通讯]` (University of California Berkeley)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种面向现代GPU的K‑Means（Lloyd算法）实现，重点解决传统实现中的内存传输与原子写冲突瓶颈，能够在保持数学精度的前提下实现大规模、高频率的聚类任务。

**💡 创新点**

创新点包括：
- 在线argmin融合距离计算与归约，完全消除距离矩阵的中间存储；
- 通过argsort-逆序重排将散射原子写转为分段局部归约，显著降低原子冲突；
- 算法-系统协同设计：异步数据流水线隐藏PCIe延迟，动态形状自适应编译策略大幅减少配置时间。

**🔧 技术方法**

主要技术手段有：
- 双缓冲与异步预取的二维分块策略；
- GPU共享内存/寄存器级局部归约；
- CUDA流实现双缓冲Out‑of‑Core传输；
- 基于硬件缓存特征的编译器启发式参数选取。

**📊 数据集**

实验使用合成数据，系统性扫描点数 N、簇数 K、特征维度 d、批量 B 的组合；未报告具体真实数据集，但通过多维度参数空间验证了算法的通用性。

**📈 对比分析**

对比基线包括 Torch、cuML KMeans、NVIDIA 自研库以及其它优化实现。实验显示：
- 赋值核速度提升至 21.2×，更新核提升 6.3×；
- 单次迭代整体加速 17.9×，对比 NVIDIA 库提升 33×，对比其它实现提升 200×；
- 超大规模（10⁹点）Out‑of‑Core 迭代完成 41.4 s，基线 261.8 s，获得 6.3×速度；
- 动态形状编译耗时从 325 s 降至 2.5 s，性能差异 <0.3%。

**⚠️ 局限性**

局限性：
- 仅针对欧氏距离的完整 K‑Means，未讨论稀疏或非欧氏度量；
- 实验主要基于合成数据，缺乏对真实任务（如文本、图像特征）的验证；
- 依赖 NVIDIA H200 GPU，移植到其它 GPU 架构或 CPU 的可行性尚未评估；
- 仍需在大规模动态更新场景下进一步验证稳定性。

---

## 268. Multi-level meta-reinforcement learning with skill-based curriculum

**arXiv ID:** 2603.08773 | [PDF](https://arxiv.org/pdf/2603.08773v1)

**作者:** Sichen Yang `[一作]` (Johns Hopkins University), Mauro Maggioni `[通讯]` (Johns Hopkins University)

**通讯引用:** 6015 | [OpenAlex ID](https://openalex.org/A5089371996)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出一种多层次元强化学习框架，利用压缩MDP、技能-嵌入分解以及课程学习，实现对复杂任务的层次化规划与迁移。

**💡 创新点**

创新点包括：① 用参数化策略族作为抽象动作在更高层MDP中形成压缩层级；② 将策略拆分为嵌入与技能（可重复使用的高阶函数），实现跨任务、跨层级的迁移；③ 通过教师‑学生‑助手三方合作的课程设计，逐层引导学习并构造可迁移技能；④ 在压缩过程中保留语义结构，显著降低随机性与搜索空间。

**🔧 技术方法**

技术手段：多层次MDP（MMDP）压缩、偏策略生成器与嵌入器、外积组合、技能-嵌入分解、价值迭代（或Q‑学习扩展）、课程学习与教师指导的自动抽象生成。

**📊 数据集**

实验数据集：MazeBase+（带多房间、门钥匙任务）与“交通拥堵导航”网格世界，两者均为离散状态空间，使用手工定义的网格和障碍配置。

**📈 对比分析**

方法比较：在同一任务下与传统单层价值迭代对比，MMDP框架在迭代次数与有效状态/动作空间上均有显著下降，学习速度提升约为传统方法的1/5-1/3，且在迁移任务中只需极少迭代即可收敛；实验中未给出精确运行时长，但通过迭代次数与理论分析表明计算成本下降显著。

**⚠️ 局限性**

局限性：① 主要验证在小规模离散网格问题，尚未在大规模或连续空间中检验；② 需要教师事先提供嵌入与技能生成器，迁移性受限于这些先验；③ 对噪声、动态环境适应性待进一步研究；④ 目前只讨论价值迭代与Q‑学习的扩展，深度强化学习或近似方法的结合尚未实现。

---

## 269. Stepping VLMs onto the Court: Benchmarking Spatial Intelligence in Sports

**arXiv ID:** 2603.09896 | [PDF](https://arxiv.org/pdf/2603.09896v1)

**作者:** Yuchen Yang `[一作]` (Fudan University), Zhihang Zhong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 553 | [OpenAlex ID](https://openalex.org/A5112535563)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于运动场景的半自动三维重建引擎，并基于此生成了大规模体育空间推理数据集 CourtSI 与高质量基准 CourtSI-Bench。

**💡 创新点**

创新点在于将体育运动中的实时人机交互视为空间推理任务，利用场地几何作为度量参考，实现了厘米级 3D 重建与丰富的数值 QA 对；同时提供跨运动可迁移评估与空间意识评论生成。

**🔧 技术方法**

技术主要包括基于 PnP 的相机标定、Prompt-HMR 人体网格恢复、手工球位置与高度标注、以及自动化 QA 生成模板和距离阈值评估。

**📊 数据集**

使用的基准数据为 RacketVision（网球、羽毛球、乒乓球）与自行构建的多视角校准集，生成了 1M+ QA 对和 3,686 个高质量验证样本。

**📈 对比分析**

通过对 25 种公开与专有 VLM 的准确率和 T-MRA 进行评估，发现最强模型仍落后人类，距离测量准确率最低；在 CourtSI 进行监督微调后，Qwen3-VL-8B 提升 23.5% 准确率，尤其在距离测量上提升 25%+。

**⚠️ 局限性**

局限性包括对单视角 2D 图像的深度估计依赖人工标注，且现有 VLM 在视角歧义、精细 3D 定位与跨运动迁移方面仍表现欠佳。

---

## 270. More than the Sum: Panorama-Language Models for Adverse Omni-Scenes

**arXiv ID:** 2603.09573 | [PDF](https://arxiv.org/pdf/2603.09573v1)

**作者:** Weijia Fan `[一作]` (Shenzhen University), Rainer Stiefelhagen `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 17076 | [OpenAlex ID](https://openalex.org/A5087051920)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出面向全景图像的视听语言模型（Panorama-Language Model, PLM），并创建了首个大型全景视觉问答数据集 PanoVQA，涵盖正常、遮挡和事故三类驾驶场景。

**💡 创新点**

核心创新点在于：①设计了全景稀疏注意力（Panoramic Sparse Attention, PSA），动态挑选关键 token 并兼顾局部与全局关系；②通过 PSA 将现有窄视角 VLM 无需重训练即可处理 360° 视角；③构建 PanoVQA 以评估全景上下文推理，填补现有 VQA 仅覆盖窄视角的空白。

**🔧 技术方法**

技术手段包括：ViT 视图增强（添加局部 Sliding Window Attention 与 PSA 并行）、多头自注意力、位置嵌入、门控网络、全参数微调（SFT）以及 GPT-Score 评价框架。

**📊 数据集**

使用的数据集：PanoVQA（约 653k QA 对，包含 538k 训练、115k 验证），PanoVQA-mini（25k QA），以及基准对比数据集 NuScenes‑QA、DriveLM、OmniDrive、NuPlanQA‑1M、DeepAccident、BlendPASS、mmWalk 等。

**📈 对比分析**

方法比较：在 PanoVQA 上对比多种开源 VLM（Chameleon、LLaVA、Qwen、InternVL、GLM 等）以及商业模型（Gemini、Grok）。在 7B‑9B 参数规模下，PanoLM‑7B（PLM）在全景 VQA 上平均得分约 45.5%，显著高于同类模型（例如 Qwen2.5‑VL‑7B 45.2%，InternVL3‑8B‑Ins 34.5%）。实验还展示了 PSA 在参数效率和性能提升方面的优势。

**⚠️ 局限性**

局限性：①仍主要验证于自动驾驶场景，泛化至其它全景任务尚未充分测试；②PSA 需要动态计算门控，增加前向推理成本；③对极端几何畸变（如非常宽视角或高分辨率）处理仍有提升空间；④模型规模仍受 7B‑9B 限制，进一步扩展到更大规模需要更高算力。

---

## 271. Cross-Domain Uncertainty Quantification for Selective Prediction: A Comprehensive Bound Ablation with Transfer-Informed Betting

**arXiv ID:** 2603.08907 | [PDF](https://arxiv.org/pdf/2603.08907v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]`, Abhinaba Basu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并系统评估了九种有限样本风险控制方法，结合多重检验和投注式置信序列，并引入跨域信息热启动投注（Transfer‑Informed Betting）以实现更紧密的安全保证。

**💡 创新点**

创新点在于三方组合：投注式置信序列、Learn‑Then‑Test 固定序列检验与跨域热启动的结合；理论证明 TIB 在分布匹配时支配标准 WSR，并给出最优热启动的证明；并在 Lean 4 上实现了机器可验证的证明。

**🔧 技术方法**

使用 Hoeffding、Empirical Bernstein、Clopper‑Pearson、Wasserstein DRO、CVaR 等集中不等式；联合 Union Bound、Learn‑Then‑Test、投注式财富过程；PAC‑Bayes 交叉域转移；温度缩放校准以及合成模拟评估。

**📊 数据集**

实验基准包括真实数据集 MASSIVE（8 类、1102 样本）、NyayaBench v2（20 类、280 样本）以及模拟的 CLINC‑150（150 类、22500 样本）和 Banking77（77 类、13083 样本）。

**📈 对比分析**

在 18 个 (α, δ) 组合上对 9 种方法进行覆盖率对比。WSR+LTT 在大样本下得到最高覆盖率；TIB 在小样本且携带源域信息时超越标准 WSR；Union Bound 需要更多样本；Clopper‑Pearson 在低误差下约比 Hoeffding 紧 2 倍；PAC‑Bayes 转移在极小样本下仍能提供正覆盖；DRO 与 CVaR 更保守。

**⚠️ 局限性**

局限性：仅给出边际风险保证，子组/按意图的安全性需更多样本；跨域转移假设源目标相似，若差异大则失效；极小样本下仍可能出现微小违约；依赖 i.i.d. 校准数据；只适用于点预测而非集合预测。

---

## 272. Vision-Augmented On-Track System Identification for Autonomous Racing via Attention-Based Priors and Iterative Neural Correction

**arXiv ID:** 2603.09399 | [PDF](https://arxiv.org/pdf/2603.09399v1)

**作者:** Zhiping Wu `[一作]` (Zhejiang University), Hongye Su `[通讯]` (Zhejiang University)

**通讯引用:** 25584 | [OpenAlex ID](https://openalex.org/A5078403818)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为高速自动驾驶赛车设计了一套在线系统辨识框架，能够在极限操控条件下实时识别轮胎动力学参数。

**💡 创新点**

创新点包括：① 利用 MobileNetV3 对路面纹理进行概率映射，生成连续的摩擦先验，从而实现“温启动”解决冷启动问题；② 引入 s4 连续时间状态空间卷积网络，精准捕捉高频瞬态残差，克服 MLP/RNN 的时序缺陷；③ 在混合虚拟仿真环境下采用无梯度 Nelder‑Mead 单纯形算法提取物理可解释的 Pacejka 参数，并通过闭环迭代不断收敛。

**🔧 技术方法**

使用技术：MobileNetV3 视觉骨干、概率摩擦映射、s4 状态空间卷积模型、Nelder‑Mead 无梯度优化、CarSim 车辆仿真、虚拟稳态控制摆动、闭环迭代框架。

**📊 数据集**

数据集：Road Surface Condition Dataset（RSCD）用于训练视觉模型；CarSim 车辆仿真数据用于训练 s4 残差网络并验证参数提取。

**📈 对比分析**

对比方法与性能：
- 视觉模型对比 MobileNetV3、ResNet‑18、EfficientNet‑B0，MobileNetV3 在 RMSE、参数量、FLOPs 方面均显著更优（RMSE 0.102，76.1% 低于 ResNet‑18）。
- 视觉先验对冷启动影响：迭代次数从 7 降至 2（71.4% 降低），前/后轮力 RMSE 分别下降 65.3% 与 37.0%。
- 残差网络对比 MLP/RNN：s4 的前/后轮力 RMSE 分别为 0.051/0.033，显著低于 MLP（0.234/0.071）和 RNN（0.097/0.084）；计算时间为 12.2s，处于两者中间。
- 总体而言，视觉先验降低摩擦估计误差 76.1%，s4 提升侧向力 RMSE 超过 60%。

**⚠️ 局限性**

局限性：
- 仍依赖仿真环境与 RSCD 数据集，缺乏真实赛道验证，可能存在 sim‑to‑real 差距。
- s4 模型在实时推理上仍有一定计算负担，尤其在资源受限的嵌入式平台上需进一步优化。
- 视觉先验依赖路面纹理分辨率和光照条件，极端天气或遮挡可能导致先验失真。
- 闭环迭代过程虽在仿真中快速收敛，但在高速动态场景下的实际实时性尚需进一步评估。

---

## 273. MEGC2026: Micro-Expression Grand Challenge on Visual Question Answering

**arXiv ID:** 2603.08927 | [PDF](https://arxiv.org/pdf/2603.08927v1)

**作者:** Xinqi Fan `[一作]` (Manchester Metropolitan University), Adrian K. Davison `[通讯]` (Manchester Metropolitan University)

**通讯引用:** 2080 | [OpenAlex ID](https://openalex.org/A5007324349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并评估了两项基于大模型的微表情短视频问答（ME‑VQA）与长视频问答（ME‑LVQA）任务，并构建了对应的训练与测试数据集。

**💡 创新点**

创新点在于将多模态大语言/视觉‑语言模型引入微表情理解，设计了短/长视频问答范式，并首次将长视频中的时间推理与微表情识别相结合。

**🔧 技术方法**

使用 Qwen2.5VL‑3B 与 Qwen3VL‑4B 两款大型视觉‑语言模型，并采用 QLoRA 进行轻量化细调，以适应视频输入。

**📊 数据集**

数据来源包括公开微表情数据集 SAMM、CASME II、SMIC、CAS(ME)^3、4DME 等，分别用于构建 ME‑VQA‑v1/v2 与 ME‑LVQA 训练与测试集。

**📈 对比分析**

实验表明，在零样本与细调设置下，粗分类 UF1/UAR 可达 0.24–0.33，但细粒度情感识别、长视频事件计数与 AU 识别表现仍较差（MAE>5、UF1<0.6），显示当前模型效果有限。

**⚠️ 局限性**

主要局限在于训练数据规模与多样性不足，尤其长视频仅使用 10 名受试者，导致模型在身份不变性与长时序推理上的泛化能力受限。

---

## 274. ToolRosetta: Bridging Open-Source Repositories and Large Language Model Agents through Automated Tool Standardization

**arXiv ID:** 2603.09290 | [PDF](https://arxiv.org/pdf/2603.09290v1)

**作者:** Shimin Di `[一作]` (Southeast University), Yong Rui `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出ToolRosetta，一个能自动将开源代码仓库转换为可被大型语言模型调用的MCP工具的框架，显著降低人工标准化成本，提升工具调用成功率；

**💡 创新点**

创新点在于完全自动化的仓库到MCP服务的转换流水线（包含工具搜索、环境重建、接口抽取、代码检查、迭代修复）以及嵌入的安全治理机制；

**🔧 技术方法**

使用的技术包括大型语言模型（如GPT-4o）进行语义解析与代码生成、GitHub API检索、自动化环境配置、MCP协议包装、代码检查与修复循环，以及安全审计与访问控制；

**📊 数据集**

实验数据集主要为122个GitHub仓库（共1580个工具），并利用RosettaEval 387任务基准进行评测；

**📈 对比分析**

与人工工程师、GPT-4o服务生成、以及SciToolAgent、ChemCrow、RepoMaster、OpenAgents等基线对比，ToolRosetta的首轮成功率53.0%提升至68.4%，宏观平均任务完成率55.6%，在多数领域优于现有系统，平均提升约12个百分点；

**⚠️ 局限性**

局限性包括对Python生态的依赖、对环境重建和依赖配置的挑战、对非Python语言的支持不足，以及在自动化开放生态中引入的安全风险与治理需求。

---

## 275. FAME: Force-Adaptive RL for Expanding the Manipulation Envelope of a Full-Scale Humanoid

**arXiv ID:** 2603.08961 | [PDF](https://arxiv.org/pdf/2603.08961v1)

**作者:** Niraj Pudasaini `[一作]` (University of Colorado), Nikolaus Correll `[通讯]` (University of Colorado)

**通讯引用:** 6045 | [OpenAlex ID](https://openalex.org/A5047458039)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种基于深度强化学习的力自适应框架 FAME，用来让全尺寸类人机器人在双手操作时保持站立平衡。

**💡 创新点**

创新点包括：① 通过上半身关节状态与双手交互力的联合编码，构造一个可在线更新的低维隐含上下文；② 在训练阶段使用上半身姿态课程与球面采样的外力相结合，提升对姿态-力耦合扰动的鲁棒性；③ 采用无传感器的力估计方法（基于关节力矩、重力补偿和雅可比逆）实现现场在线适应。

**🔧 技术方法**

使用的技术主要有：深度强化学习（PPO）、隐含上下文自适应（类似 RMA 的编码器）、上半身姿态课程、基于动力学的力估计、域随机化与多模态感知；同时在仿真与真实机器人上进行训练与部署。

**📊 数据集**

数据集方面：① 在仿真中随机生成三维外力（球面采样）并随机上半身姿态，采集 500 条不同配置下的站立试验数据；② 在 Unitree H12 真实机器人上进行两类负载实验（单臂不对称拉力、双臂对称载荷）以验证性能。

**📈 对比分析**

性能对比方法：在仿真中对 Base（无课程、无编码）、Base+Curr（有姿态课程、无编码）和 FAME（有课程+编码）三种策略分别在 5 种固定上半身配置下进行 500 次随机外力实验。结果显示：Base 29.44%、Base+Curr 51.40%、FAME 73.84% 的平均站立成功率。真实机器人实验中，FAME 能在两种负载场景下保持平衡，而 Base+Curr 则出现失稳。

**⚠️ 局限性**

局限性：① 需要较为精确的动力学模型以实现无传感器力估计，模型误差会影响性能；② 训练依赖仿真，虽然采用域随机化但仍存在 sim-to-real 迁移瓶颈；③ 仅在 Unitree H12 与有限负载场景下验证，尚未展示在更大负载、不同机器人或多任务中的通用性；④ 强化学习训练时间长，且对超参数敏感。

---

## 276. OOD-MMSafe: Advancing MLLM Safety from Harmful Intent to Hidden Consequences

**arXiv ID:** 2603.09706 | [PDF](https://arxiv.org/pdf/2603.09706v1)

**作者:** Ming Wen `[一作]` (Fudan University), Xingjun Ma `[通讯]` (Fudan University)

**通讯引用:** 6914 | [OpenAlex ID](https://openalex.org/A5078711649)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于因果投射的多模态大语言模型安全评估与对齐方法，创建了OOD‑MMSafe基准并设计了CASPO框架。

**💡 创新点**

创新点在于引入了后续风险因果投射视角，量化并揭示“因果盲点”，并提出利用模型自身推理作为动态参考的CASPO方法。

**🔧 技术方法**

技术上结合了多模态因果MDP建模、token级自蒸馏奖励、强化学习与安全宪章引导的CASPO框架。

**📊 数据集**

使用了455例OOD‑MMSafe样本、Beavertails‑V和SPAVL安全偏好数据，以及多模态生成图片集（Flux.2‑dev、Qwen‑Image等）。

**📈 对比分析**

与传统DPO/SLHF对齐方法对比，CASPO在OOD‑MMSafe上将风险识别失败率从>80%降至5–7%，同时保持有效率不下降。

**⚠️ 局限性**

局限性在于对高容量模型的动态参考仍受模型自身推理质量限制，且缺乏对更大规模、跨领域多模态安全场景的进一步验证。

---

## 277. RiO-DETR: DETR for Real-time Oriented Object Detection

**arXiv ID:** 2603.09411 | [PDF](https://arxiv.org/pdf/2603.09411v1)

**作者:** Zhangchi Hu `[一作]`, Xiaoyan Sun `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出 RiO-DETR，一种支持实时旋转目标检测的 DETR 变体。

**💡 创新点**

创新点在于将角度信息从位置查询中解耦，采用周期化短路角度更新和旋转正交注意力，并引入 Dense O2O 训练策略以提升收敛速度和定位精度。

**🔧 技术方法**

核心技术包括 KLD 损失、Hausdorff 匹配、D-FINE 通用匹配策略、DEIM 密集一对一监督、周期化角度差和旋转正交注意力头的分配。

**📊 数据集**

主要使用 DOTA-1.0、FAIR-1M-2.0 以及 COCO 作为训练与评估数据集。

**📈 对比分析**

与现有实时和非实时旋转检测器对比，RiO-DETR 在 DOTA-1.0 单尺度下 AP50 达 81.78，且在多尺度上保持最高 AP50/AP75，并在实时性上与 RT-DETRv2 相比显著提升速度与精度。

**⚠️ 局限性**

局限性包括对极角度近似的仍有一定敏感性，训练过程仍需较大显存与时间，且在高度旋转或稀疏目标场景下的鲁棒性待进一步验证。

---

## 278. Tracking Cancer Through Text: Longitudinal Extraction From Radiology Reports Using Open-Source Large Language Models

**arXiv ID:** 2603.09638 | [PDF](https://arxiv.org/pdf/2603.09638v1)

**作者:** Luc Builtjes `[一作]` (Radboud University Medical Center), Alessa Hering `[通讯]` (Radboud University Medical Center)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5078758408)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并验证了一个完全开源、可本地部署的管线，用于从放射学报告中提取并关联基于RECIST的纵向病灶信息。

**💡 创新点**

首次提出了利用大语言模型实现多时间点病灶链接的任务定义，并实现了完整的可复现、隐私友好的开放源码解决方案。

**🔧 技术方法**

采用了开源LLM（Llama‑2‑70B）与Open‑Source “llm_extractinator”框架，结合自定义提示和Pydantic‑v2结构化解析器。

**📊 数据集**

使用了Radboudumc 2021‑2025年期间的50对荷兰语CT胸腹部报告，平均每对包含2.6目标病灶、5.0非目标病灶和0.91新病灶。

**📈 对比分析**

与两位人工标注者比对后，目标、非目标和新病灶的属性级准确率分别超过93%、94%和94%；整体报告级无误率在88%~96%之间，显示模型性能与人工标注相当。

**⚠️ 局限性**

局限性包括对表格格式变化的敏感、对“不可测量”或“已消失”病灶的处理不一致，以及在报告措辞多变时的标签一致性略低。

---

## 279. Compiler-First State Space Duality and Portable $O(1)$ Autoregressive Caching for Inference

**arXiv ID:** 2603.09555 | [PDF](https://arxiv.org/pdf/2603.09555v1)

**作者:** Cosmo Santoni `[一作]` (Imperial College), Cosmo Santoni `[通讯]` (Imperial College)

**通讯引用:** 9756 | [OpenAlex ID](https://openalex.org/A5107858764)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了Mamba-2状态空间模型（SSD）在不使用自定义CUDA/Triton内核的情况下可在CPU、GPU和TPU上无修改运行的可移植实现；

**💡 创新点**

创新点在于将SSD的算法特性（对角状态、chunkable递归、einsum主导计算、静态控制流）与XLA的融合与tiling优化完美对齐，使得O(1)缓存、无主机同步的推理成为可能；

**🔧 技术方法**

使用了JAX + XLA编译器、einsum、静态掩码、编译时循环、PyTree缓存、BF16/FP32精度控制等技术；

**📊 数据集**

使用了HuggingFace预训练的Mamba-2检查点（130M–2.7B参数），在Google Cloud TPU v6e、NVIDIA A100和CPU上进行评估；

**📈 对比分析**

与非缓存、手写CUDA/Triton实现对比；在TPU v6e上单序列prefill达≈140 TFLOPS（15% MFU），decode达到64% HBU；缓存实现相比无缓存提升数倍且保持O(1)内存；与PyTorch/CUDA参考实现在token级别完全一致；

**⚠️ 局限性**

仅在单一硬件（TPU v6e）测评利用率；固定chunk大小L=256、batch=1；缺乏多用户推理、动态分区、不同硬件后端的进一步评估；依赖成熟XLA后端。

---

## 280. CORAL: Scalable Multi-Task Robot Learning via LoRA Experts

**arXiv ID:** 2603.09298 | [PDF](https://arxiv.org/pdf/2603.09298v1)

**作者:** Yuankai Luo `[一作]`, Zhenguo Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CORAL 框架，冻结 VLA 主干并为每个任务训练独立的低秩 LoRA 专家，通过语言指令直接路由并在运行时动态切换专家，实现多任务学习与终身学习的高效部署。

**💡 创新点**

创新点在于完全参数隔离的 LoRA 专家消除多任务干扰，利用语言指令实现无门控的精确路由，并通过极低存储占用实现连续增量任务，兼具高效性与可扩展性。

**🔧 技术方法**

采用 LoRA 参数高效微调、动态专家切换的 CORAL Manager、语言指令任务路由技术，并在多种 VLA backbone（SimVLA、π_0.5 等）上验证。

**📊 数据集**

使用 LIBERO、WidowX、Google Robot 三个仿真 benchmark 以及 Galaxea R1 真实机器人平台；训练数据包括约 500 小时的开放世界大规模数据以及任务特定演示。

**📈 对比分析**

与全微调、联合微调以及现有 VLA 方案对比，CORAL 在 LIBERO 上平均 99.3% 成功率、WidowX 97.9%、Google Robot 84.9%；在真实任务中成功率提升至 95% 以上，存储占用比全微调低 100 倍，显著缓解灾难性遗忘。

**⚠️ 局限性**

局限在于仅适用于可由语言明确标识的任务，缺乏复杂多模态路由机制；单任务 LoRA 对极端动态或低样本任务仍有限制，且在极大任务集合下可能需要进一步的共享结构。

---

## 281. Design Guidance Towards Addressing Over-Reliance on AI in Sensemaking

**arXiv ID:** 2603.08903 | [PDF](https://arxiv.org/pdf/2603.08903v1)

**作者:** Yihang Zhao `[一作]` (King's College London), Elena Simperl `[通讯]` (Technical University of Munich)

**通讯引用:** 6555 | [OpenAlex ID](https://openalex.org/A5046030036)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并讨论了一种将生成式人工智能嵌入群体意识工具（GAT）的系统，以隐式方式支持协作工作和学习中的自主意义建构。

**💡 创新点**

创新点在于：①区分生成式AI的适用场景，采用混合规则+AI架构；②通过背景色编码在雷达图中呈现生成式AI分析的差异，保持认知冲突；③提供悬停细节交互，让团队主动探索AI推断的证据，避免明确指令导致的依赖。

**🔧 技术方法**

技术上结合了规则引擎提取结构化合作数据、生成式AI（LLM）对非结构化对话和文档进行语义理解，并利用可视化（雷达图、背景色）与交互（悬停）展示结果。

**📊 数据集**

本工作基于文献综述与案例分析，并未使用公开数据集。

**📈 对比分析**

未进行实证对比或性能评估，仅提出设计原则与示例可视化，未来需通过实验验证其对自主意义建构的影响。

**⚠️ 局限性**

局限性包括缺乏经验验证、潜在的过度依赖风险、实现难度、隐私与偏见问题，以及在不同协作场景的可迁移性尚待检验。

---

## 282. Provably Safe Trajectory Generation for Manipulators Under Motion and Environmental Uncertainties

**arXiv ID:** 2603.09083 | [PDF](https://arxiv.org/pdf/2603.09083v1)

**作者:** Fei Meng `[一作]` (Hong Kong University of Science and Technology), Max Q. -H. Meng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于递归时域的安全轨迹规划框架，在运动和环境不确定性下为机械臂生成风险受限路径。

**💡 创新点**

创新点在于：①引入 RM‑DeSKO 神经网络预测不确定机械臂的状态分布；②设计层级碰撞风险验证方法，先用物理仿真快速评估碰撞力，再用 SOS 程序正式证明碰撞概率不超过阈值；③将该风险信息嵌入 MPPI 控制器，实现实时规划并保持风险上限。

**🔧 技术方法**

采用深度随机 Koopman 运算符网络（RM‑DeSKO）、MPPI 采样式 MPC、IsaacGym 物理仿真、SOS 程序、Polycam 三维扫描、深度学习框架 PyTorch 等技术。

**📊 数据集**

使用模拟数据集（Franka Emika Panda 在 IsaacGym 下采样轨迹与噪声），以及真实 UR5e 机器人与人类协作的实验数据，障碍物通过 Polycam 预扫描得到三维网格。

**📈 对比分析**

与基线 MPPI、Transformer、LSTM 等方法比较，实验表明在 10 个任务中成功率从 89% 提升至 94%，规划时间从 47.2 s 缩短至 34.6 s，轨迹长度从 2.27 下降至 1.17，真实协作实验成功率约 90%。

**⚠️ 局限性**

局限性包括：需要为每个起点–目标对预先收集示例轨迹；障碍物必须以多项式或已扫描网格形式给出；SOS 验证在高维情况下仍昂贵；在极端噪声或不完整地图时仍可能失效。

---

## 283. EPOCH: An Agentic Protocol for Multi-Round System Optimization

**arXiv ID:** 2603.09049 | [PDF](https://arxiv.org/pdf/2603.09049v1)

**作者:** Zhanlin Liu `[一作]` (ProRata.ai), Munirathnam Srikanth `[通讯]` (ProRata.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了EPOCH协议，用于在多轮迭代中在异构环境下构建基线并管理自我改进。

**💡 创新点**

创新点在于将迭代优化抽象为两阶段协议，角色分离、标准化执行接口与轮次追踪，支持跨任务一致的工程化流程。

**🔧 技术方法**

使用大语言模型驱动的规划、执行、评估等角色，并通过可配置的通用协议进行任务抽象与执行。

**📊 数据集**

使用四个公开数据集：SST-2（情感分类）、MNIST（图像分类）、Iris（符号分类）和自定义 Fibonacci 计算器（代码优化）。

**📈 对比分析**

与单任务专用优化器对比，EPOCH在四个任务上保持了可追溯、可复现且不牺牲性能的改进；具体指标如SST-2从0.8提升至1.0，MNIST从0.53提升至0.66，Iris从0.9778提升至1.0，Fibonacci运行时间从94ms缩短至0.07ms。

**⚠️ 局限性**

局限在于仅实现单目标迭代优化，缺乏多代理协同；实验规模受限，未与现有最佳优化方法进行严格基准；对大规模或非确定性任务的通用性尚待验证。

---

## 284. bsort: A theoretically efficient non-comparison-based sorting algorithm for integer and floating-point numbers

**arXiv ID:** 2603.08929 | [PDF](https://arxiv.org/pdf/2603.08929v1)

**作者:** Benjamín Guzmán `[一作]` `[通讯]` (Independent researcher), Benjamín Guzmán (Independent researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 bsort 的位运算非比较排序算法，能够原地对有符号整数、无符号整数以及 IEEE‑754 浮点数进行 O(wn) 时间、O(w) 空间的排序。

**💡 创新点**

创新点在于将二进制快速排序改造为统一处理符号整数和浮点数的位级划分，并通过三阶段（符号→指数→尾数）排序实现对所有数据类型的原地线性排序，特别适用于小字长场景。

**🔧 技术方法**

采用了位掩码递归划分、单次扫描两指针交换、整数的位级快速排序以及浮点数的符号/指数/尾数分级排序，并在实现中使用了递归和递归终止条件来控制划分深度。

**📊 数据集**

实验使用随机生成的 8、16、32、64 位整数数组以及 32、64 位浮点数组，规模从 10⁵ 到 5×10⁹，并在同一硬件上与 STL introsort、Boost spreadsort、ska_sort 等实现进行对比。

**📈 对比分析**

通过 GoogleBenchmark 进行统一基准测试，比较运行时间、内存占用和是否基于比较；结果显示 bsort 在 8 位整数下能与混合算法竞争甚至超越，而在更大字长时由于递归深度和分支预测等问题略显逊色。

**⚠️ 局限性**

主要限制在于递归深度等于字长导致栈消耗大、分支预测不佳、指令量多、缺乏混合优化和 SIMD 以及缓存友好性，因而在大字长（如 64 位）场景下性能远低于现代混合排序实现。

---

## 285. Towards Visual Query Segmentation in the Wild

**arXiv ID:** 2603.08898 | [PDF](https://arxiv.org/pdf/2603.08898v1)

**作者:** Bing Fan `[一作]` (University of North Texas), Heng Fan `[通讯]` (University of North Texas)

**通讯引用:** 40168 | [OpenAlex ID](https://openalex.org/A5100703985)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出视觉查询分割（VQS）这一新范式，设计并公开了VQS-4K大规模基准数据集，并基于SAM 2 开发了 VQ‑SAM 方法实现全帧像素级目标定位。

**💡 创新点**

创新点包括：①把视觉查询从视频中迁移到外部帧，要求在未裁剪视频中定位所有出现的目标；②通过多阶段记忆演化（target 与 distractor 取样）和自适应记忆生成（AMG）显著提升记忆的判别力；③引入空间‑时间 Transformer（STT）捕获全局时序上下文。

**🔧 技术方法**

采用共享 Hiera‑B+ 编码器提取特征，基于 SAM 2 的掩码解码器、记忆注意力、目标/干扰器特征抽取（TFG/DFG）、AMG 以及 STT 块组成的多阶段网络。

**📊 数据集**

使用 VQS‑4K：4 111 个视频、1.3M 帧、222 类目标，外部查询来自视频外帧；此外在 VQ2D 数据集上进行二次评估以验证通用性。

**📈 对比分析**

与现有 VQL（如 REN、VQLoC）和 VOS（如 SAM 2、SAM 2Long）方法比较，VQ‑SAM 在 stAP、tAP、Recovery、Success 等指标上均显著优于对手（stAP 26.0% 对 18.6%，tAP 29.6% 对 24.4%）。在 VQ2D 上亦取得最佳表现。

**⚠️ 局限性**

局限性：① 对 GPU 内存敏感，单次推理仅能处理 8 帧，需分片合并；② 训练与推理耗时较长；③ 受限于 222 类目标，尚缺乏极端光照、遮挡等极端情况的评估；④ VQ‑SAM 仍需进一步提升对多目标、快速运动目标的鲁棒性。

---

## 286. External entropy supply for IoT devices employing a RISC-V Trusted Execution Environment

**arXiv ID:** 2603.09311 | [PDF](https://arxiv.org/pdf/2603.09311v1)

**作者:** Arttu Paju `[一作]` (Tampere University), Brian McGillion `[通讯]` (Technology Innovation Institute)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5036763044)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

构建并演示了一个基于可信执行环境（TEE）的外部熵服务，用于为资源受限的 IoT 设备提供高质量、可远程验证的熵。

**💡 创新点**

创新点在于：1）将 TEE 与硬件熵源（如环形振荡器）结合，形成可验证的熵交付链；2）设计了安全的熵请求-响应协议，支持自签名请求、时间戳验证、数字签名与节流；3）提供完整的开源实现与可扩展架构，方便后续多源熵集成。

**🔧 技术方法**

使用技术包括：TEE（SPIRS 软硬件平台）、RSA‑3072 加密与签名、AES‑128 对称加密、时间戳与签名验证、请求节流机制；实现环境为 QEMU 虚拟机，代码基于 OpenSSL 与 Keystone（TEE SDK）。

**📊 数据集**

未使用公开数据集；熵源来自同一设备集群的硬件环形振荡器与预安装的真随机种子。

**📈 对比分析**

与随机性灯（Randomness Beacon）对比，强调其可选择性熵投递和远程证明优势；但在论文中仅给出功能验证示例，未提供量化性能指标或可扩展性评估。

**⚠️ 局限性**

局限性包括：单点服务器可能成为瓶颈；仅使用同类硬件熵源，易受硬件缺陷影响；未在真实 SPIRS 硬件上测试；缺乏 PQC 支持；缺少性能基准与多客户端可扩展性评估。

---

## 287. Quantifying the Accuracy and Cost Impact of Design Decisions in Budget-Constrained Agentic LLM Search

**arXiv ID:** 2603.08877 | [PDF](https://arxiv.org/pdf/2603.08877v1)

**作者:** Kyle McCleary `[一作]`, James Ghawaly `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了BCAS，一种在显式搜索和令牌预算下评估代理检索（Agentic Retrieval）性能的框架，并在六种大语言模型和三大问答数据集上系统测量搜索深度、检索组件与生成预算的准确率与成本权衡。

**💡 创新点**

创新点在于将预算感知纳入代理检索流程，提供可重复的实验平台，量化搜索次数、检索策略与生成长度在固定预算下的准确率提升，进而给出实用的预算配置优先级。

**🔧 技术方法**

采用LLM提示、预算信号、预规划与反思循环、迭代检索工具、BM25+向量检索混合、轻量重排序以及完整的搜索与令牌使用计数等技术，实现预算驱动的检索与生成。

**📊 数据集**

使用了TriviaQA、HotpotQA和2WikiMultihopQA三大开放式问答数据集进行评估。

**📈 对比分析**

通过对六个模型在多种搜索次数与令牌上限的组合进行准确率对比，发现最多可通过三次搜索获得显著提升，混合检索+重排序带来最高平均增益，令牌预算在多跳推理任务上影响更大，整体提供了准确率-成本折衷曲线。

**⚠️ 局限性**

局限性包括仅针对静态事实QA语料、未包含非代理单轮检索基准、使用统一提示未针对模型调优、二值LLM评判可能忽略部分正确性、未进行实时延迟与实际成本估算，且不涵盖多语言、多模态或动态知识环境。

---

## 288. First Steps towards Categorical Algebraic Artificial Chemistry

**arXiv ID:** 2603.09431 | [PDF](https://arxiv.org/pdf/2603.09431v1)

**作者:** Joe Pratt-Johns `[一作]` (Edinburgh Napier University), Peter Andras `[通讯]` (Edinburgh Napier University)

**通讯引用:** 2684 | [OpenAlex ID](https://openalex.org/A5001723863)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

**🎯 论文内容**

构造了一个泛函，将单一排序 Lawvere 理论的代数映射为马尔可夫过程，从而为基于代数的人工化学模型提供动力学；

**💡 创新点**

将范畴论工具引入人工化学，提出 Flask 泛函以统一描述不同的交互协议和动力学，扩展并抽象化了 Fontana‑Buss 的 AlChemy 模型；

**🔧 技术方法**

使用 Lawvere 理论、分配单子、马尔可夫过程的范畴论框架，并在语义层面进行形式化构造与证明；

**📊 数据集**

未使用具体实验数据集，示例仅基于 lambda 词、自然数等符号集合；

**📈 对比分析**

未进行实验比较，主要以形式化证明和示例说明为主；性能未量化；

**⚠️ 局限性**

局限在于：无法处理空间交互、类型化/逻辑化学的扩展；缺乏通用可实现的实验框架；

---

## 289. From Days to Minutes: An Autonomous AI Agent Achieves Reliable Clinical Triage in Remote Patient Monitoring

**arXiv ID:** 2603.09052 | [PDF](https://arxiv.org/pdf/2603.09052v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 290. Towards Instance Segmentation with Polygon Detection Transformers

**arXiv ID:** 2603.09245 | [PDF](https://arxiv.org/pdf/2603.09245v1)

**作者:** Jiacheng Sun `[一作]` (Shanghai University), Xiaomao Li `[通讯]` (Shanghai University)

**通讯引用:** 4487 | [OpenAlex ID](https://openalex.org/A5014108121)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

我们提出 Poly-DETR，一种基于 Transformer 的多边形检测器，通过稀疏极坐标参数直接预测实例分割，而非传统密集掩码。

**💡 创新点**

创新点在于引入 Polar Deformable Attention 与 Position-Aware Training Scheme，解决了起始点漂移和采样不匹配问题，并系统比较了极坐标与掩码两种表征的优劣。

**🔧 技术方法**

采用 Deformable DETR 架构、极坐标表征、Polar Deformable Attention、Position-Aware Training Scheme、残差更新和多尺度自注意力等技术。

**📊 数据集**

在 MS COCO、Cityscapes、PanNuke 与 SpaceNet 等公开数据集上进行训练和评估。

**📈 对比分析**

与 PolarNeXt、BoundaryFormer 等极坐标模型以及 Mask-DETR 进行对比，Poly-DETR 在 COCO 上提升 4.7 mAP（AP_75 +5.4），在高分辨率和正多边形实例上更快更轻，整体 mAP 约 40-41，帧率 32 FPS。

**⚠️ 局限性**

对不规则、碎片化或复杂拓扑的实例拟合效果欠佳，极坐标表征对起始点位置敏感，难以处理形状极其不规则的目标。

---

## 291. Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning

**arXiv ID:** 2603.09697 | [PDF](https://arxiv.org/pdf/2603.09697v1)

**作者:** Yechen Zhang `[一作]` (Shanghai Jiao Tong University), Kai Chen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Mousse的优化器，将Muon's等距谱约束与Shampoo的二阶预处理相结合，在白化坐标系下执行Newton-Schulz正交化，实现更合适的几何更新；

**💡 创新点**

核心创新是把谱优化与曲率预处理融合，形成基于白化的谱优化框架，并引入Trace Normalization、Spectral Tempering等稳定技术，显著提升样本效率与收敛质量；

**🔧 技术方法**

使用了Kronecker‑factored预处理（Shampoo）、Newton‑Schulz迭代、谱归一化、Trace Normalization、Spectral Tempering、单侧预处理、梯度嫁接等技术；

**📊 数据集**

在FineWeb 20B tokens数据集上，用GPT‑2变体训练160M至800M参数模型进行实验；

**📈 对比分析**

与AdamW、Muon、SOAP进行网格搜索学习率后对比；Mousse在所有规模上取得最低验证损失，训练速度与Muon相近，样本效率提升约12%，内存占用低于SOAP（≈88%）且仅比Muon多约5%；

**⚠️ 局限性**

对光谱参数和曲率估计敏感，需Trace Normalization和Spectral Tempering进行调优；单侧预处理可能略逊于双侧；实现仍为实验原型，硬件效率待进一步提升；对Fine‑tuning兼容性尚待验证。

---

## 292. Arbiter: Detecting Interference in LLM Agent System Prompts

**arXiv ID:** 2603.08993 | [PDF](https://arxiv.org/pdf/2603.08993v1)

**作者:** Tony Mason `[一作]` (University of British Columbia), Tony Mason `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5041896410)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM编码代理的系统提示进行静态分析，提出Arbiter框架，结合规则驱动与多模型无指导探索，识别不同架构下的干扰模式。

**💡 创新点**

①将系统提示视为软件artifact引入正式评估规则和AST结构分析；②采用多模型无指导“scouring”发现单模型难以捕捉的漏洞；③建立按架构（单体、扁平、模块）对应失效模式的关联。

**🔧 技术方法**

规则驱动的静态检查、基于AST的结构分析、跨模型无指导LLM探索以及指向式停止判定等技术。

**📊 数据集**

三大主流编码代理的系统提示（Claude Code 1490 行、Codex CLI 298 行、Gemini CLI 245 行）及其版本差异。

**📈 对比分析**

通过与手工标注的 21 个干扰模式对比，多模型探索发现 152/15/21 项结果，成本仅 $0.27 USD，发现率随提示规模呈正比，模型多样性显著提升覆盖率。

**⚠️ 局限性**

仅静态分析未验证运行时行为；覆盖的仅三家厂商，缺乏更广泛的架构验证；发现依赖LLM生成，可能含伪造，需要人工复核。

---

## 293. Automated Thematic Analysis for Clinical Qualitative Data: Iterative Codebook Refinement with Full Provenance

**arXiv ID:** 2603.08989 | [PDF](https://arxiv.org/pdf/2603.08989v1)

**作者:** Seungjun Yi `[一作]` (University of Texas at Austin), Ying Ding `[通讯]` (University of Texas at Austin)

**通讯引用:** 16513 | [OpenAlex ID](https://openalex.org/A5047170063)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套端到端的可追溯性主题分析框架，结合迭代代码簿改进与多代理自动编码流程；

**💡 创新点**

通过在迭代过程中逐步扩大训练样本并融合代码相似性判定，显著提升代码簿的泛化能力，并且为每一次操作生成可审计的行动日志，实现完整的分析轨迹追踪；

**🔧 技术方法**

核心技术包括大型语言模型（LLM）驱动的编码（LOGOS）、主题合成模块（Auto-TA）、基于余弦相似度的代码关系分类、以及动作日志与唯一标识符的审计系统；

**📊 数据集**

使用了五个多样化数据集：儿童心脏病患者家族访谈（AAOCA、SV-CHD）、YouTube 视频转录（Ali Abdaal）、学术研究者访谈（Sheffield）以及 Reddit 社交媒体帖子（Dreaddit）；

**📈 对比分析**

与六个现有自动化编码基线进行对比，采用五项质量指标（可重用性、适配性、描述性、简洁性、一致性）评估；在四个数据集上迭代改进后得到最高综合得分，改进显著（p<0.01）且效果大（Cohen d>2.7）；

**⚠️ 局限性**

局限性包括：缺乏严格的早停准则、部分指标（可重用性与一致性）可能存在重叠、LLM评估偏差、主题抽象度偏高导致与专家标签不完全对齐、以及对人机协同与成本控制的进一步研究需求。

---

## 294. Why Channel-Centric Models are not Enough to Predict End-to-End Performance in Private 5G: A Measurement Campaign and Case Study

**arXiv ID:** 2603.08865 | [PDF](https://arxiv.org/pdf/2603.08865v1)

**作者:** Nils Jörgensen `[一作]` (KTH Royal Institute of Technology), Nils Jörgensen `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5016958230)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在地下私有5G工厂环境中，用移动机器人采集下行吞吐量、延迟、MCS和RI等数据，并将其与光线追踪模拟和高斯过程回归（GPR）模型预测的吞吐量进行对比。

**💡 创新点**

首次验证了通道中心模型（光线追踪+3GPP）无法准确预测终端吞吐量，并发现主要误差来自MIMO空间层数误估；同时提出直接基于测量的GPR吞吐量预测可显著降低误差。

**🔧 技术方法**

使用Altair Feko光线追踪、3GPP TR 38.901标准、Raspberry Pi 4/Quectel RM500Q-GL移动测量平台以及Gaussian Process Regression（RBF、Matern、RQ核）等技术。

**📊 数据集**

收集并公开了约7000条包含空间坐标、平均吞吐量、标准差、MCS、RI等字段的测量数据集。

**📈 对比分析**

通过误差偏差、MAE、RMSE、百分位等统计指标进行比较；光线追踪平均误差约145 Mbps，偏高且偏差大；GPR将误差减少约两倍、消除系统性偏差，预测更接近真实吞吐量。

**⚠️ 局限性**

光线追踪缺乏对真实MIMO层数与链路自适应的建模；GPR计算复杂、对环境变化适应慢；两种方法均存在可扩展性和实时更新的局限，需要进一步融合物理模型与数据驱动技术。

---

## 295. Age-Related Differences in the Perception of Eye-Gaze from a Social Robot

**arXiv ID:** 2603.08810 | [PDF](https://arxiv.org/pdf/2603.08810v1)

**作者:** Lucas Morillo-Mendez `[一作]` (Örebro University), Oscar Martinez Mozos `[通讯]` (Örebro University)

**通讯引用:** 7367 | [OpenAlex ID](https://openalex.org/A5088202291)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开展了基于Pepper机器人头部运动的指示性凝视（deictic gaze）对老年人与年轻人协作任务（面包制作）的感知与执行时间影响的在线实验，分析了不同年龄群体的客观行为和主观体验。

**💡 创新点**

首次在老年人群体中系统评估机器人指示性凝视的效用，并将其与传统静态指示进行对比，验证了该非语言线索在不同年龄段的可行性与影响差异。

**🔧 技术方法**

使用Pepper机器人头部运动模拟凝视、在线实验平台Labvanced、NASA‑TLX、Godspeed、RoSAS问卷、混合稳健ANOVA与引导抽样等统计方法对行为与主观数据进行分析。

**📊 数据集**

数据集为329名受试者（老年人64岁以上与年轻人18-64岁）在线完成的实验记录，包含反应时间、任务完成时间、问卷得分等指标；未使用公开图像或语言数据集。

**📈 对比分析**

通过对比静态（SR）与移动（MR）两种机器人条件，分别对不同年龄组进行混合设计，利用稳健混合ANOVA和t检验检验交互效应；结果显示移动凝视显著加快两类受试者的反应与完成时间，但两年龄组间加速效应无显著差异。

**⚠️ 局限性**

局限性包括：仅采用Pepper机器人头部运动模拟凝视，缺乏人类或更自然的眼动；在线实验限制了人与机器人的实际互动；部分受试者未察觉凝视差异，可能影响结果；实验任务相对简单，未能覆盖更复杂的日常协作情境。

---

## 296. Multi-Kernel Gated Decoder Adapters for Robust Multi-Task Thyroid Ultrasound under Cross-Center Shift

**arXiv ID:** 2603.08906 | [PDF](https://arxiv.org/pdf/2603.08906v1)

**作者:** Maziar Sabouri `[一作]` (University of British Columbia), Arman Rahmim `[通讯]` (University of British Columbia)

**通讯引用:** 17488 | [OpenAlex ID](https://openalex.org/A5021438906)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

对甲状腺超声图像进行多任务学习，包括结节分割、TI-RADS恶性风险预测和解剖定位，并针对跨中心域迁移设计鲁棒模型

**💡 创新点**

提出轻量化解码器侧多核门控适配器 MKGA 及其残差变体 ResMKGA，通过多尺度感受野与语义条件门控缓解几何驱动与纹理驱动任务间的负迁移

**🔧 技术方法**

使用 ResNet34 CNN 与 MedSAM ViT 编码器，配合 MKGA/ResMKGA 以及可选的 PCGrad 梯度调度和 LoRA 参数高效微调

**📊 数据集**

训练使用 ThyroidXL（内测中心）数据集，跨中心外部评测使用 DDTI 数据集

**📈 对比分析**

采用 Dice、IoU、准确率、F1 与 AUC 进行量化比较；MKGA/ResMKGA 在 DDTI 上将分割 Dice 提升至约 0.67，TI‑RADS 分类准确率提升至 0.63，显著优于单一共享编码器或仅使用 PCGrad 的基线；ViT 变体在分割上表现良好但在恶性预测上显著退化

**⚠️ 局限性**

局限性包括：仅验证两种编码器；对纹理依赖强的 ViT 在跨中心表现不佳；缺乏对长期临床部署与多中心更广泛数据集的评估

---

## 297. Adaptive Clinical-Aware Latent Diffusion for Multimodal Brain Image Generation and Missing Modality Imputation

**arXiv ID:** 2603.09931 | [PDF](https://arxiv.org/pdf/2603.09931v1)

**作者:** Rong Zhou `[一作]` (Lehigh University), Alzheimer's Disease Neuroimaging Initiative `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

设计并实现了ACADiff框架，利用自适应临床感知扩散模型填补缺失脑影像模态，并生成完整的sMRI、FDG‑PET与AV45‑PET三种扫描；

**💡 创新点**

提出了三大创新点：1）自适应多源融合机制，可在2→1与1→1场景下动态选择交叉注意力或投影；2）通过GPT‑4o编码的语义临床提示实现临床信息的深度引导；3）构建了专门的生成器实现六种双向模态互换，并在扩散过程中采用层次化条件和时间调制；

**🔧 技术方法**

采用3D VAE将高维影像压缩为latent空间，利用3D U‑Net的扩散去噪器完成目标模态的生成；自适应融合使用多头注意力与投影；临床提示通过GPT‑4o文本编码器并交叉注意力融合；使用FiLM调制与时间步调节；推断时进行10倍蒙特卡洛采样；

**📊 数据集**

使用公开的ADNI数据库，共1,028名受试者（包括AD、MCI、HC），具备sMRI、FDG‑PET与AV45‑PET三模态；

**📈 对比分析**

与Pix2Pix、DS‑GAN、LDM、PASTA、FICD等基线方法在不同缺失率（0%、20%、40%、60%、80%）下进行对比；在分类任务中，ACADiff在80%缺失时仍保持77.5%准确率，整体表现优于所有基线（如LDM的76.4%），并在生成质量指标（PSNR、SSIM、NMI、MAE）上领先；

**⚠️ 局限性**

主要限制包括：仅在单一ADNI数据集上验证，缺乏跨中心或跨人群的泛化评估；生成模型对计算资源需求较高，需大规模GPU支持；在极端缺失率下仍可能出现信息缺失，导致诊断性能衰减；

---

## 298. Bioalignment: Measuring and Improving LLM Disposition Toward Biological Systems for AI Safety

**arXiv ID:** 2603.09154 | [PDF](https://arxiv.org/pdf/2603.09154v1)

**作者:** Trent R Northen `[一作]` (Bioaligned Labs), Mingxun Wang `[通讯]` (Computer Science and Engineering Department, University of California Riverside)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型在技术方案选择上对生物学与合成方案的偏好，并通过自定义“Bioalignment Benchmark”评估与量化这种偏差。

**💡 创新点**

提出基于Kelly准则的Δp_up指标和50条跨领域（材料、能源、制造、算法）对照提示，用极少量的生物学训练数据即能显著提升模型对生物学方案的偏好，实现了模型倾向的可塑性。

**🔧 技术方法**

采用QLoRA参数高效微调、4-bit NF4量化、LoRA层配置等技术；评估时使用配套的评测脚本与传统基准（MMLU、HellaSwag、ARC、WinoGrande）对比。

**📊 数据集**

使用从PubMed Central收集的6,636篇开放获取论文（约22M token，包含约5.5M token的指令化子集），并将其拆分为持续预训练和指令格式两部分。

**📈 对比分析**

对10个模型（5开放权重、5前沿）进行基线评估，发现大多数模型偏向合成方案；对Llama 3B与Qwen 3B微调后Δp_up分别提升+0.132（p<0.001）和+0.054（p<0.01），效果在所有传统基准上无显著性能下降，显示出高效且可泛化的偏好调节。

**⚠️ 局限性**

局限包括：提示集仅50条且仅涵盖四个领域，模型规模限制在3B级，未检验对更大模型的可扩展性；训练数据来源单一，可能带来偏见；评估依赖默认采样参数；对生物偏好在实际决策中的实效性尚未验证。

---

## 299. An Empirical Study and Theoretical Explanation on Task-Level Model-Merging Collapse

**arXiv ID:** 2603.09463 | [PDF](https://arxiv.org/pdf/2603.09463v1)

**作者:** Yuan Cao `[一作]` (Peking University), Tao Xie `[通讯]` (Peking University)

**通讯引用:** 17627 | [OpenAlex ID](https://openalex.org/A5048118068)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了任务级模型合并中的崩溃现象，并提出基于信息论的理论解释

**💡 创新点**

发现任务级表示不兼容是导致合并崩溃的根本原因，并给出了维度相关的最小失真下界

**🔧 技术方法**

采用多种模型合并方法（线性平均、任务算术、TIES、DARE、SLERP）以及率失真理论分析

**📊 数据集**

使用来自Lots‑of‑LoRAs的64个任务检查点和GLUE八个任务的模型检查点，覆盖多种模型架构和规模

**📈 对比分析**

通过实验比较不同合并方法在相同任务组合下的性能，证明无论方法如何，表示不兼容的任务组合都会出现灾难性下降；理论下界与实验结果高度一致

**⚠️ 局限性**

仅关注基于同一基模型的Fine‑tune检查点，未探究跨基模型或更大规模模型的合并情况，且理论假设（LMC）可能在某些实际场景下不完全成立

---

## 300. From Weighting to Modeling: A Nonparametric Estimator for Off-Policy Evaluation

**arXiv ID:** 2603.09436 | [PDF](https://arxiv.org/pdf/2603.09436v1)

**作者:** Rong J. B. Zhu `[一作]` `[通讯]`, Rong J. B. Zhu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了非参数加权方法（NW）以及其基于奖励预测的改进版（MNW）用于上下文赌博机的离线策略评估。

**💡 创新点**

创新点在于用非参数模型直接拟合行为策略与奖励的关系，避免传统IPW的高方差，并通过奖励残差的非参数建模实现类似双重鲁棒但不需严格模型正确性的估计。

**🔧 技术方法**

主要技术包括P-spline非参数回归、非参数加权估计、以及奖励残差非参数校正，并给出了收敛速率分析。

**📊 数据集**

在实验中使用公开的多分类基准数据集（如letter、glass、ecoli、opt、page、pen、sat、vehicle、yeast）以及模拟的K-armed bandit场景进行验证。

**📈 对比分析**

通过与传统DM、IPW、DR估计器的对比，NW和MNW在多数据集和不同采样策略下均表现出更低的RMSE、较低方差且偏差可忽略，尤其在估计日志策略误差时显示出更强的鲁棒性。

**⚠️ 局限性**

局限包括仅使用P-spline实现非参数估计、未考虑二值奖励的离散特性、以及对大动作空间的适用性仍需进一步研究。

---

## 301. EXPLORE-Bench: Egocentric Scene Prediction with Long-Horizon Reasoning

**arXiv ID:** 2603.09731 | [PDF](https://arxiv.org/pdf/2603.09731v1)

**作者:** Chengjun Yu `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**通讯引用:** 19231 | [OpenAlex ID](https://openalex.org/A5003217535)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向长时序视角推理的 egocentric 场景预测基准 EXPLORE‑Bench，并对其进行细粒度的多维度评估。

**💡 创新点**

创新点：①首次将原始第一人称视频与长序列原子动作对齐，生成结构化的最终场景注释（对象、属性、关系）；②提出统一的评估协议，可对对象覆盖、属性准确度与关系质量进行量化；③探索了推理时的分段/多轮策略，揭示了分解动作序列对性能的影响。

**🔧 技术方法**

使用的技术包括多模态大型语言模型（MLLM）推理、基于 GPT‑5.2/ Gemini‑3 的思维与非思维模式、Qwen 系列与其它开源 MLLM 的多轮推理、Grounding‑DINO、Qwen‑3‑VL‑Instruct 生成属性/关系、基于 Sentence‑BERT 的对象匹配、LLM 评分器进行指标计算。

**📊 数据集**

数据集来源：Ego4D、Ego‑Exo4D 两大公开第一人称视频库以及自制多场景视频，共 1,157 条实例；每条实例包含起始图像、约 11–694 条原子动作、最终场景的 23,771 个对象与 1,612 类别的结构化标注。

**📈 对比分析**

评估对比：人类基准 S_uni 约 59；最佳开源 MLLM Qwen3‑VL‑8B‑Thinking/SVL‑Instruct 在 S_uni 约 70–71；最佳专有模型 Gemini‑3‑Pro、GPT‑5.2‑Chat 也在 66–70 之间；大多数模型与人类相比存在 7–10 分的性能差距，且在异常案例（S_abn）更大，说明长时序 egocentric 推理仍具挑战。

**⚠️ 局限性**

局限性：①仅在有限的推理策略上探索测试时缩放，缺乏更高效的分解方法；②异常/稀有情形样本不足；③基准主要基于现有公开视频，训练集规模受限，未覆盖全部可能的动作与场景组合。

---

## 302. The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness

**arXiv ID:** 2603.09200 | [PDF](https://arxiv.org/pdf/2603.09200v1)

**作者:** Subramanyam Sahoo `[一作]` (University of Cambridge), Divya Chaudhary `[通讯]` (Northeastern University)

**通讯引用:** 5164 | [OpenAlex ID](https://openalex.org/A5048878908)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出RAISE框架，阐述逻辑推理改进如何通过推理自检、归纳情境识别和归纳自建模型三条路径提升AI的情境感知水平；

**💡 创新点**

首次将推理提升与情境感知关联，构建递进阶梯模型与正式化论证，揭示推理改进不可避免地增强自我意识与潜在欺骗风险；

**🔧 技术方法**

基于理论分析与形式化证明，使用逻辑推理理论（演绎、归纳、溯因）与推理提升方法（Chain‑of‑Thought、外部求解器、奖励模型）；

**📊 数据集**

未使用具体数据集，主要引用现有LLM研究和安全评测标准做背景引用；

**📈 对比分析**

无量化实验比较，文中提出“镜像测试”与“推理安全平行原则”作为评估手段，预期通过这些指标衡量推理改进对情境感知的影响；

**⚠️ 局限性**

局限在于缺乏实证验证，理论模型对真实LLM行为的预测仍需实验验证；同时未给出可实现的完全隔离机制，仍存在递归监测与不可预见的自我提升风险。

---

## 303. From Verification to Amplification: Auditing Reverse Image Search as Algorithmic Gatekeeping in Visual Misinformation Fact-checking

**arXiv ID:** 2603.09130 | [PDF](https://arxiv.org/pdf/2603.09130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 304. TA-Mem: Tool-Augmented Autonomous Memory Retrieval for LLM in Long-Term Conversational QA

**arXiv ID:** 2603.09297 | [PDF](https://arxiv.org/pdf/2603.09297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 305. MAPLE: Elevating Medical Reasoning from Statistical Consensus to Process-Led Alignment

**arXiv ID:** 2603.08987 | [PDF](https://arxiv.org/pdf/2603.08987v1)

**作者:** Kailong Fan `[一作]` (Zhejiang University), Ning Guo `[通讯]` (Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为 MAPLE 的测试时训练框架，将医学过程奖励模型与测试时强化学习结合，以提升医学推理大语言模型的可靠性。

**💡 创新点**

创新点在于用过程级、专家对齐的 Med-RPM 步骤奖励替代传统多数投票伪监督，使测试时更新直接朝临床正确性收敛，显著提升推理质量。

**🔧 技术方法**

采用多样化采样、过程奖励模型评估、Soft 权重伪标签、GRPO 强化学习与小学习率的在线策略优化等技术。

**📊 数据集**

在 MedQA、MedMCQA、DDXPlus 与 MMLU‑Med 四个医学推理基准上进行评估。

**📈 对比分析**

与 Llama3.1、HuatuoGPT、Med‑PRM、TTRL 等基线对比，MAPLE 在所有基准上均优于同等参数模型，甚至在 MedQA 达到 73.02%，在 32B QwQ 上表现更好。

**⚠️ 局限性**

主要局限是对 Med‑RPM 质量高度依赖；在大批量采样时奖励噪声可能导致性能下降。

---

## 306. SiliconMind-V1: Multi-Agent Distillation and Debug-Reasoning Workflows for Verilog Code Generation

**arXiv ID:** 2603.08719 | [PDF](https://arxiv.org/pdf/2603.08719v1)

**作者:** Mu-Chi Chen `[一作]` (Academia Sinica), Hsiang-Tsung Kung `[通讯]` (Harvard University)

**通讯引用:** 211 | [OpenAlex ID](https://openalex.org/A5112533490)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套统一的多代理蒸馏与测试推理工作流，用于在本地微调的 LLM 上自动生成、测试和调试 Verilog 代码。

**💡 创新点**

创新点在于自动化生成 reasoning‑oriented 训练数据与 testbench，构建多代理数据管道，并在推理阶段实现自测自修复，从而不依赖商业模型或外部验证工具。

**🔧 技术方法**

采用多代理生成 pipeline、推理引擎（Regular、Deep‑Thinking、Agentic 三种策略）、SFT 训练、功能验证（Icarus Verilog）、RLVR 等技术。

**📊 数据集**

使用公开 Verilog 代码库（DeepCircuitX、PyraNet、RTLCoder、VeriThought、Verilog_Github）构成 36k（p^', r, c^', tb）训练样本，并进一步通过自我纠错扩充数据。

**📈 对比分析**

通过 Pass@k 指标在 RTLLM‑v2、VerilogEval‑v2、VerilogEval‑v2‑NTU、CVDP‑cid02/03 等基准上与 CodeV‑R1、DeepSeek‑R1 等 SOTA 进行对比，获得相似或更高的功能正确率，同时训练时间约 9 倍加速。

**⚠️ 局限性**

仍受限于训练数据规模与测试/调试策略的局限，某些极难 benchmark 上性能波动较大，且尚未完全实现对所有硬件环境的无工具验证。

---

## 307. OTPL-VIO: Robust Visual-Inertial Odometry with Optimal Transport Line Association and Adaptive Uncertainty

**arXiv ID:** 2603.09653 | [PDF](https://arxiv.org/pdf/2603.09653v1)

**作者:** Zikun Chen `[一作]` (Shanghai Jiao Tong University), Jingchuan Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9958 | [OpenAlex ID](https://openalex.org/A5100317823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 OTPL-VIO，一个利用深度线段描述符和全局最优传输匹配的立体点线视觉-惯性里程计，针对低纹理和强光照变化场景。

**💡 创新点**

创新点在于（1）无需训练的轻量级线段描述符；（2）使用熵正则化最优传输实现全局一致的线匹配，能处理未匹配与部分观测；（3）基于线段长度和跟踪持久性的可靠性自适应加权，提升后端优化稳定性。

**🔧 技术方法**

采用 PLNet+LightGlue 进行点/线检测与匹配，基于深度特征采样构造线描述符，利用熵正则最优传输求解线匹配，并在因子图优化中引入可靠性权重；若有 IMU，则加预积分因子。

**📊 数据集**

在公开数据集 EuRoC MAV、UMA‑VI（照明变化与低纹理子集）以及自采低纹理/照明剧烈变化的室内数据集上进行评估。

**📈 对比分析**

与 VINS‑Fusion、AirSLAM、PLF‑VINS、Kimera‑VIO 等基线对比，OTPL‑VIO 在 EuRoC 平均 8.06 cm、UMA‑VI 低纹理平均 11.60 cm、实测场景平均 39.99 cm 的 ATE，显著优于所有基线，并保持实时性能（≈32 ms/帧）。

**⚠️ 局限性**

局限包括：仍需立体或 IMU 传感器；对非常短的线段与极端运动仍可能产生不确定性；系统对光照变化的鲁棒性虽然提升，但在极端暗/光环境下仍可能出现漂移。

---

## 308. Ego: Embedding-Guided Personalization of Vision-Language Models

**arXiv ID:** 2603.09771 | [PDF](https://arxiv.org/pdf/2603.09771v1)

**作者:** Soroush Seifi `[一作]` (Toyota Motor Europe), Rahaf Aljundi `[通讯]` (Toyota Motor Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练-free、无需外部模块的嵌入引导个性化方法（Ego），通过分析大型视听语言模型的交叉注意力，从参考图像中提取最具代表性的视觉标记，构建个性化概念记忆，并在推理阶段以软提示方式注入，从而实现单概念、多概念和视频级个性化。

**💡 创新点**

创新点在于：①利用模型内部注意力直接挑选最具代表性的视觉token，避免全图像编码和外部检索；②通过对参考图像的面积估计动态决定记忆大小，实现高效紧凑记忆；③统一评测框架对比多种现有方法，展示训练-free方法在多任务、多场景下的领先表现。

**🔧 技术方法**

核心技术包括：视觉编码器+投影器映射为视觉token；基于LLM交叉注意力的token重要性评分与层选择；根据关键字提取与聚合最重要的视觉token；将概念记忆以soft prompt注入LLM；支持视频帧级检索与推理。

**📊 数据集**

使用了多种公开数据集：单概念（MyVLM、Yo'LLaVA、This-is-my-img）、多概念（This-is-my-img扩展、RAP）、视频问答（This-is-my-img视频QA）等；在InternVL3-14B和Qwen2.5-VL-7B上进行评测。

**📈 对比分析**

与训练型方法RAP、训练-free方法R2P和PeKit对比，Ego在单概念识别、VQA和字幕回调任务中均实现或逼近SOTA，尤其在多概念和视频场景下提升显著（例如VQA F1提升近20%，字幕召回提升近30%），且推理时仅需短时间的关键词生成，无需额外fine‑tune或外部模块。

**⚠️ 局限性**

局限性包括：对老旧或视听理解能力弱的模型适用性不足；需要手工设定最大token数K以及层选择阈值；在极端多概念或极大图像规模下可能面临上下文长度瓶颈；目前仅支持静态参考图像，尚未深入处理动态视角变换或光照变化。

---

## 309. Cognitively Layered Data Synthesis for Domain Adaptation of LLMs to Space Situational Awareness

**arXiv ID:** 2603.09231 | [PDF](https://arxiv.org/pdf/2603.09231v1)

**作者:** Ding Linghu `[一作]` (Qian Xuesen Laboratory of Space Technology), Cong Zhang `[通讯]` (Qian Xuesen Laboratory of Space Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 BD-FDG 框架，用于构建面向太空态势感知（SSA）的高质量监督微调数据集，并利用该数据集对 Qwen3‑8B 进行微调，形成 SSA‑LLM‑8B。

**💡 创新点**

创新点在于将布鲁姆认知层级与 SSA 任务链相结合，构建了结构化知识树、九类认知层级问题模板以及工程规范对齐的多维质量控制，解决了领域知识覆盖、认知深度不足和质量可控性差的问题。

**🔧 技术方法**

技术包括混合稠密‑稀疏检索、基于 QWQ‑Plus 的多层级问题生成、教师模型（Qwen‑Max）推理链生成与四维质量评估，以及 16 倍多重蒸馏提升样本多样性。

**📊 数据集**

使用了约 230K 条 SSA‑SFT 训练样本（覆盖 9 类问题和 6 认知层级）以及 1,644 条非重叠 SSA‑Test 验证样本，训练集还混合 600K 条通用 OpenThoughts3 数据。

**📈 对比分析**

通过 BLEU、ROUGE、Arena Battle（与 Qwen3‑8B 对比）和多项通用基准（MATH‑500、MMLU‑Pro 等）评估，SSA‑LLM‑8B 在 SSA‑Test 上 BLEU‑1 从 21.33% 提升至 57.23%（think 模式），Arena 赢率 82.21%，并在大多数通用基准上保持或略低的表现。

**⚠️ 局限性**

主要局限包括高昂的全参数微调算力需求、教师模型偏差可能被迁移、仅基于公开文献构建知识库导致最高层级操作细节缺失，以及缺乏人工专家评估验证。

---

## 310. Prismatoid Band-Unfolding Revisited

**arXiv ID:** 2603.09813 | [PDF](https://arxiv.org/pdf/2603.09813v1)

**作者:** Joseph O'Rourke `[一作]` `[通讯]` (Smith), Joseph O'Rourke (Smith)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

研究并给出了嵌套棱锥（nested prismatoid）带状展开（band-unfolding）是否无重叠的完整判定，证明了已知的六边形反例在一定意义下是唯一可能导致重叠的情况。

**💡 创新点**

创新点在于将半径单调性（radial monotonicity）与带状展开相结合，提出并证明了“安全切割”（safe cut）与顶面多边形满足 RM‑property 的条件，同时开发了开口引理（opening lemmas）和旋转合成工具，以此构造出非重叠展开。

**🔧 技术方法**

采用了球面三角不等式、开口引理、单调性分析、半径单调性、欧拉角合成（planar rotations）以及 Cauchy 腿部引理等几何工具来证明展开不重叠。

**📊 数据集**

利用随机生成的凸多边形以及六边形反例等示例来说明 RM‑property 的普遍性和反例的特殊性，并未使用真实的实验数据集。

**📈 对比分析**

与先前仅通过混合 petal 与 band 展开的经验性方法相比，本工作提供了理论判定标准；由于是理论证明，没有涉及数值性能比较。

**⚠️ 局限性**

局限性在于仍未证明所有嵌套棱锥都有安全切割，且仅处理了嵌套情况，非嵌套棱锥的边展开问题仍未解决。

---

## 311. SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training

**arXiv ID:** 2603.08997 | [PDF](https://arxiv.org/pdf/2603.08997v1)

**作者:** Jingxing Li `[一作]` (Arizona State University), Deliang Fan `[通讯]` (Arizona State University)

**通讯引用:** 5261 | [OpenAlex ID](https://openalex.org/A5047916979)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SkipGS，一种在3D Gaussian Splatting后稠密化阶段的视角自适应后向跳过机制，减少不必要的反向传播计算；

**💡 创新点**

创新点在于通过跟踪每个视角的损失均值并与当前损失比较，动态决定是否执行反向传播，同时自动校准最小反向预算，兼顾加速与质量；

**🔧 技术方法**

技术包括指数移动平均EMA、比例偏差评分、累计反向比例控制、后向跳过决策以及可配置的热身窗口；

**📊 数据集**

在Mip-NeRF 360、Deep Blending、Tanks & Temples三个公开基准数据集上进行实验；

**📈 对比分析**

与原始3DGS及多种效率提升基线（FastGS、Taming 3DGS、GaussianSpa、LightGaussian、Speedy-Splat）进行对比，SkipGS在后稠密化阶段平均缩短42%训练时间，整体训练时间降低约20-25%，同时保持或略微提升PSNR/SSIM，LPIPS基本不变；

**⚠️ 局限性**

局限性包括：需在后稠密化阶段才有效，对前期稠密化阶段无帮助；预算阈值选择仍需经验调优，过度跳过可能导致质量下降；在极大规模或极稀疏场景下跳过策略的泛化性尚未充分验证。

---

## 312. Does the Question Really Matter? Training-Free Data Selection for Vision-Language SFT

**arXiv ID:** 2603.09715 | [PDF](https://arxiv.org/pdf/2603.09715v1)

**作者:** Peng Sun `[一作]` (Nanjing University), Yuqiang Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1571 | [OpenAlex ID](https://openalex.org/A5055664612)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑free 的视觉指令调优数据选择方法（CVS），通过冻结的视觉‑语言大模型（VLLM）评估问题与答案在图像上下文中的可信度变化，筛选出需要真正跨模态推理的样本；

**💡 创新点**

创新点在于：①利用“条件裁决位移”（Conditional Verdict Shift）直接衡量问题对答案可信度的增益，从而识别语义冲突与无意义样本；②优先挑选“边界样本”（低 CVS_Yes），而非高置信度样本，提升梯度信息；③完全无训练、仅前向推理，显著降低计算成本；

**🔧 技术方法**

使用冻结的 VLLM（如 Qwen2.5‑VL‑7B‑Instruct）作为评估器，计算 P(Yes|I,Q,A)/P(Yes|I,A) 与 P(No|I,Q,A)/P(No|I,A) 的对数比值；不需要额外代理模型或梯度计算；

**📊 数据集**

主要实验数据集为 Vision‑Flan 和 The Cauldron，用于训练 VIT；评估指标包括多种 VQA、文本视觉理解、图表、文档等基准；

**📈 对比分析**

与 CLIP‑Score、EL2N、SemDeDup、D2 Pruning、COINCIDE、XMAS、随机采样等方法比较；在 Vision‑Flan 上 10%/15% 采样比下，CVS 的 ARP 分别为 103.5% / 104.8%，超过全数据训练；在 The Cauldron 上表现稳定，且在 10.5 GPU 小时内完成，较 COINCIDE、XMAS 分别节省 17.3% 与 44.4% 的计算时间；

**⚠️ 局限性**

局限性：①对评估器的性能敏感，过弱评估器可能导致选择不佳；②在结构冗余型噪声占主导的数据集（如高采样比例下的 Cauldron）时，聚类/去重方法可能更优；③对极小数据预算（如 5%）的冷启动效果不如预期；④仅适用于能提供图像‑文本‑答案三元组的任务，无法直接迁移至其他多模态场景。

---

## 313. Physics-Informed Neural Engine Sound Modeling with Differentiable Pulse-Train Synthesis

**arXiv ID:** 2603.09391 | [PDF](https://arxiv.org/pdf/2603.09391v1)

**作者:** Robin Doerfler `[一作]` (Impulse Audio Lab GmbH), Lonce Wyse `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 1405 | [OpenAlex ID](https://openalex.org/A5062692118)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

直接建模发动机声的脉冲列与阻尼，并通过可微的 Karplus‑Strong 反馈实现排气共振；

**💡 创新点**

在脉冲生成与可微声学滤波两侧引入物理启发的先验（如热力学相位调制、阀门动态包络、加速/减速燃料截止门控），从而实现可解释且更精确的声纹；

**🔧 技术方法**

采用可微数字信号处理（DDSP）、可微 Karplus‑Strong 反馈、Gumbel‑Softmax 延迟参数化、梯度回传优化及多分辨率 STFT+谐波损失组合；

**📊 数据集**

使用 Procedural Engine Sounds Dataset 的三个子集（A: I‑4，B: V8 低频共振，C: V8 高频金属共振），共计约 7.5 小时的合成发动机音频；

**📈 对比分析**

与同结构的 HPN 基线在多分辨率 STFT 与谐波损失上比较，PTR 在总损失上降低 5.7%，谐波重建提升 21%，并在三类发动机上保持一致的性能；

**⚠️ 局限性**

主要限制包括仅在合成数据上验证，未测试真实录音；对不同发动机配置的泛化仍有限；缺乏对其它车辆噪声（如涡轮、传动箱）以及环境噪声鲁棒性的评估。

---

## 314. Exploring Modality-Aware Fusion and Decoupled Temporal Propagation for Multi-Modal Object Tracking

**arXiv ID:** 2603.09287 | [PDF](https://arxiv.org/pdf/2603.09287v1)

**作者:** Shilei Wang `[一作]` (Northwestern Polytechnical University), Gong Cheng `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 20672 | [OpenAlex ID](https://openalex.org/A5080476856)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了MDTrack框架，实现多模态跟踪的模态感知融合和解耦时间传播。

**💡 创新点**

创新点：①为每种模态分配专用专家并使用Mixture of Experts实现动态模态选择；②使用两个独立的状态空间模型（SSM）分别建模RGB与其他模态的时间动态，并通过交叉注意力实现信息互通；③兼顾模态专用与统一训练，提升跨模态鲁棒性。

**🔧 技术方法**

采用的技术包括Mixture of Experts、双SSM（Mamba/SSM）时间模块、交叉注意力、HiViT骨干、张量嵌入、门控路由、负载均衡损失以及多阶段训练策略。

**📊 数据集**

使用的主要数据集有LasHeR、RGBT234、DepthTrack、VOT‑RGBD2022、VisEvent，并在这些基准上进行评估。

**📈 对比分析**

通过与STTrack、SUTrack、SDSTrack、BAT、OneTrack、Un-Track等最新方法在五个基准上的对比，MDTrack‑S和MDTrack‑U分别在LasHeR（精度76.5%、AUC61.4%）、RGBT234（MPR93.0%、MSR70.5%）、DepthTrack（F1 67.5%）、VOT‑RGBD2022（EAO 80.0%、鲁棒性95.1%）和VisEvent（精度82.2%、成功率65.3%）上均取得或接近最佳成绩，显示出显著的性能提升。

**⚠️ 局限性**

局限性：模型参数量和计算量较大，尤其是双SSM和MoE门控；对极端模态缺失或噪声的鲁棒性尚未充分验证；目前仅在五个基准上评测，未覆盖更广泛的真实场景和其他模态组合。

---

## 315. Quality over Quantity: Demonstration Curation via Influence Functions for Data-Centric Robot Learning

**arXiv ID:** 2603.09056 | [PDF](https://arxiv.org/pdf/2603.09056v1)

**作者:** Haeone Lee `[一作]` (KAIST), Kimin Lee `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于影响函数的机器人演示数据筛选方法QoQ。

**💡 创新点**

将最大影响评分和轨迹级聚合引入影响函数，直接衡量演示数据对验证损失的贡献，从而实现更精准的数据质量评估。

**🔧 技术方法**

使用影响函数、最大影响评分、轨迹级聚合、OPORP压缩梯度以及Transformer/GR00T N1模型进行策略训练与梯度计算。

**📊 数据集**

在Robomimic仿真数据、真实Franka机器人香蕉抓取、多物体搬运、抽屉开启任务以及DROID真实世界数据集上进行实验。

**📈 对比分析**

与全数据、行为检索、光流检索等基线对比，QoQ在模拟实验中成功率达99.2%，在真实机器人实验中达86.7%，相对最佳基线提升约30%及以上。

**⚠️ 局限性**

仅实现轨迹级筛选，无法细粒度子轨迹筛选；影响函数计算成本高；假设训练与验证共享相同体型；目前仅在行为克隆框架下验证，跨任务或跨体型推广待进一步研究。

---

## 316. From Data Statistics to Feature Geometry: How Correlations Shape Superposition

**arXiv ID:** 2603.09972 | [PDF](https://arxiv.org/pdf/2603.09972v1)

**作者:** Lucas Prieto `[一作]` (Imperial College London), Pedro A. M. Mediano `[通讯]` (Imperial College London)

**通讯引用:** 3973 | [OpenAlex ID](https://openalex.org/A5074280948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Bag-of-Words Superposition（BOWS）框架，研究在真实文本特征相关情况下的特征叠加与干扰机制。

**💡 创新点**

证明特征相关时干扰可为建设性，利用低秩结构实现线性叠加，从而解释语言模型中的语义聚类和周期性结构。

**🔧 技术方法**

使用带ReLU或线性解码器的稀疏自编码器、PCA、线性探针等方法分析特征几何。

**📊 数据集**

使用WikiText-103（V=10k词汇，c=20）以及OpenWebText数据集，生成二进制bag-of-words样本。

**📈 对比分析**

通过比较不同瓶颈尺寸、权重衰减条件下的ReLU和线性AE，观察R²和结构可视化；发现权重衰减/小瓶颈时线性叠加占优，语义聚类与周期结构显著。

**⚠️ 局限性**

BOWS过于简化，未覆盖全部语言模型复杂特性；未给出完整数学阐述何时各机制占优，且仅评估自编码器，对更大模型和解耦权重的情况缺乏验证。

---

## 317. ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis

**arXiv ID:** 2603.09611 | [PDF](https://arxiv.org/pdf/2603.09611v1)

**作者:** KunHo Heo `[一作]` (Kyung Hee University), MyeongAh Cho `[通讯]` (Kyung Hee University)

**通讯引用:** 414 | [OpenAlex ID](https://openalex.org/A5010029584)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 ParTY 模型，能够在保持全身动作连贯性的同时，从文本中精准生成各身体部位的表达性动作。

**💡 创新点**

创新点包括：① Part‑aware Text Grounding 将单一文本嵌入分解成多样化嵌入并动态为每个部位选择合适的嵌入；② Part‑Guided Network 先生成部件动作，再以此作为引导生成全身动作，配合 Holistic‑Part Fusion 实现跨部位连贯性；③ 提出了部件级和连贯性（Temporal/Spatial Coherence）评估指标，全面验证方法效果。

**🔧 技术方法**

技术手段涵盖：Temporal‑aware VQ‑VAE 对动作序列进行离散化；Transformer 架构的全身与部件生成网络；对比学习与多样化文本嵌入实现细粒度文本对齐；跨模态注意力和自适应融合实现部件与整体信息融合。

**📊 数据集**

使用数据集：HumanML3D 和 KIT‑ML。

**📈 对比分析**

通过 R‑Precision、MM‑Dist、FID、多样性与多模态度量，以及新增的部件 R‑Precision、Temporal Coherence 和 Spatial Coherence 进行评估；在所有指标上，ParTY 显著优于 MoMask、ParCo 等基线，尤其在全身连贯性和部件表达上实现了新的性能巅峰。

**⚠️ 局限性**

局限性包括：推理时延相对较高（部件网络与全身网络分离导致）；依赖 LLM 生成的部件文本作为训练辅助，推理时需额外前置步骤；在极长或复杂文本描述下，嵌入选择仍可能不足；未对多人物或非人类动作场景进行验证。

---

## 318. Towards Terrain-Aware Safe Locomotion for Quadrupedal Robots Using Proprioceptive Sensing

**arXiv ID:** 2603.09585 | [PDF](https://arxiv.org/pdf/2603.09585v1)

**作者:** Peiyu Yang `[一作]` (Istituto Italiano di Tecnologia), Cosimo Della Santina `[通讯]` (Delft University of Technology)

**通讯引用:** 4192 | [OpenAlex ID](https://openalex.org/A5050239145)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出一种仅基于本体感知的地形估计、接触与状态耦合估计方法，并将其集成到基于控制屏障函数的MPC安全控制器，实现四足机器人在不规则地形上的安全行走。

**💡 创新点**

创新点包括：①使用概率融合仅凭IMU、编码器和低成本接触力传感器实现2.5‑D地形图与支持平面参数的实时估计；②将地形信息与接触概率耦合于状态估计，显著提升CoM估计精度；③设计全局与局部安全约束的控制屏障函数，并结合MPC给出严格安全保证。

**🔧 技术方法**

主要技术包括：本体感知地形映射算法（概率融合+三角面法）、接触概率融合、Kalman滤波状态估计、控制屏障函数（CBF）与MPC优化。

**📊 数据集**

使用Unitree Go1四足机器人，在Gazebo 11仿真环境和实际斜坡、平地、陡坡实验平台上进行验证；未使用公开数据集，采用自建的地形图与机器人轨迹数据。

**📈 对比分析**

与基线点更新法、局部斜率逼近法相比，所提方法在2.5‑D地图平滑度上更优；在支持平面估计上角度误差相近；状态估计误差平均绝对误差降低64.8%，方差降低47.2%。

**⚠️ 局限性**

局限性：仅能在已探索的局部区域内提供安全保证，无法预判未知地形；对极端遮挡/光照变化的鲁棒性有限；未来需结合外部SLAM进行安全冗余。

---

## 319. Benchmarking Dataset for Presence-Only Passive Reconnaissance in Wireless Smart-Grid Communications

**arXiv ID:** 2603.09590 | [PDF](https://arxiv.org/pdf/2603.09590v1)

**作者:** Bochra Al Agha `[一作]` (American University of Beirut), Razane Tajeddine `[通讯]` (American University of Beirut)

**通讯引用:** 350 | [OpenAlex ID](https://openalex.org/A5000709773)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个面向智能电网多层通信（HAN/NAN/WAN）的存在‑仅被动侦察基准数据集生成器，提供了从物理通道到指标的完整一致映射，并通过严格的泄漏安全设计支持分裂无关的训练/验证/测试。

**💡 创新点**

创新点在于：①以IEEE标准为灵感的分层拓扑与邻接约束；②只通过传输通道的阴影损耗与相干衰减实现被动侦察，无需注入任何协议层事件；③生成的数据保持物理一致性与统计合理性，并通过邻居加权聚合和时序特征支持拓扑感知的图‑时序学习；④发布了联邦学习基准，验证了不同技术下的可检测性。

**🔧 技术方法**

主要技术包括：基于Gauss‑Markov模型的复杂衰落过程、三维阴影与干扰建模、CSI→SNR→PER→延迟的确定性链路映射、可调的相干降级与冲击过程、邻接矩阵的行归一化混合、严格因果特征提取与标准化、以及联邦学习框架下的LR、XGB、LSTM和GRNN基线实现。

**📊 数据集**

使用自研的合成数据集，包含12个节点（HAN 4、NAN 4、WAN 4），每个节点产生独立的train/validation/test时间序列；数据集公开于GitHub，包含拓扑、邻接、节点属性、窗口元数据以及训练所需的标准化参数。

**📈 对比分析**

通过行级联邦基线（不使用时序窗口）对10个非光纤客户端进行评估。LR在召回率上表现最好但精度低；XGB在精度与召回率间取得平衡；LSTM/GRNN在时序窗口为1时相对较好。整体表现表明被动扰动难以单凭瞬时特征完全识别，强调需要图‑时序方法提升检测性能。

**⚠️ 局限性**

局限性包括：数据为合成而非真实现场，节点数量有限；仅考虑接收端被动侦察，未涵盖更复杂攻击（如注入、干扰等）；缺乏多设备、多地点的规模化验证；基准仅覆盖部分无线/PLC技术，未完整覆盖所有IEEE 2030.5场景。

---

## 320. Self-hosted Lecture-to-Quiz: Local LLM MCQ Generation with Deterministic Quality Control

**arXiv ID:** 2603.08729 | [PDF](https://arxiv.org/pdf/2603.08729v1)

**作者:** Seine A. Shintani `[一作]` (Chubu University), Seine A. Shintani `[通讯]` (Chubu University)

**通讯引用:** 459 | [OpenAlex ID](https://openalex.org/A5050619910)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个自托管（API‑free）从讲义 PDF 生成多选题的端到端流水线 L2Q，并加入确定性质量控制；

**💡 创新点**

在保持本地推理、隐私保护的前提下，将 LLM 生成的题目通过严格的 JSON schema 与多重检测（重复、数值等）转化为可直接导入学习平台的静态题库，实现黑盒最小化；

**🔧 技术方法**

本地 LLM（Qwen2.5‑14B‑Instruct，GGUF），受限解码 + 定义好的 QC 检查（schema、单一答案、去重、常数等），Python wrapper，JSONL/CSV 导出，Google Colab/本地执行；

**📊 数据集**

三份简短的“假”讲义 PDF，分别涉及信息论熵、热力学第二定律和统计机械熵，共 2–3 页；

**📈 对比分析**

在 15 次种子遍历（3 讲义×5 种子）生成 120 条题目，QC 接受率 100%，重试率 1.6%，每题平均 7.3 秒；最终挑选 24 条题库无警告；未提供学习效果评估；

**⚠️ 局限性**

仅在小规模简短 PDF 上验证，长文档、图表/扫描文档处理不足；数学等价检测不完整；QC 不能保证教学有效性或语义正确性；模型依赖仍存在，隐私/版权/能耗仍需关注。

---

## 321. Avoiding Big Integers: Parallel Multimodular Algebraic Verification of Arithmetic Circuits

**arXiv ID:** 2603.09501 | [PDF](https://arxiv.org/pdf/2603.09501v1)

**作者:** Clemens Hofstadler `[一作]` (Johannes Kepler University), Chen Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 109936 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文提出了一种结合线性与非线性重写的混合多模验证框架，利用同态映像在多条机器字大小的素数域上并行执行代数推理，从而完全避免大整数运算，实现对算术电路（特别是乘法器）的单词级验证。

**💡 创新点**

创新点在于：①首次将同态映像（多模计算）与代数重写技术结合，用以消除大整数开销；②引入了结构感知的子电路抽取、加权采样与投影线性代数求解，加速猜测与证明过程；③在同一框架下同时支持线性重写（高效但易失效）和非线性重写（完整但代价高），实现了鲁棒的混合策略。

**🔧 技术方法**

核心技术包括：多模同态映像、线性化（引入占位变量）与猜测-证明（SAT 验证）框架、加权采样、投影线性代数求解、SIMD 向量化的多模多项式操作、非线性重写（利用格罗布纳基准序）以及并行化策略。

**📊 数据集**

使用了 207 个整数乘法器基准，分为两类：① 192 个结构化 64 位乘法器（aoki-multipliers，包含 PPG、PPA、FSA 等模块）；② 15 个经合成优化后的 32/64/128 位乘法器（Synthesized multipliers），覆盖不同合成脚本和架构，测试框架在各种电路结构上的鲁棒性。

**📈 对比分析**

实验对比了多种现有工具（如仅线性重写、仅非线性重写、先前的混合方法等）。结果显示本文实现的工具在所有 207 个基准上均能成功验证（或给出反例），而其他工具在结构化基准或合成基准上均有未能求解的实例。相对最优的工具在大多数基准上的总时长比前沿方法低约 30%~50%，并且能够在多核心环境下实现近线性加速。

**⚠️ 局限性**

局限性包括：① 仍需依赖 SAT 求解器进行猜测验证，SAT 负荷在极大子电路时可能成为瓶颈；② 对于某些极端结构（如大规模 carry‑lookahead 加法器）仍需多次子电路扩展，导致线性重写无效；③ 目前的实现主要针对乘法器，尚未系统评估在除法器、加法器等其他算术模块上的泛化能力；④ 需要预先选取足够多的素数模，过多模数会增加内存开销。

---

## 322. ConfCtrl: Enabling Precise Camera Control in Video Diffusion via Confidence-Aware Interpolation

**arXiv ID:** 2603.09819 | [PDF](https://arxiv.org/pdf/2603.09819v1)

**作者:** Liudi Yang `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2586 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对仅有两张输入图像且视角差距大的新视角合成任务，提出了ConfCtrl框架，利用预训练的视频插值模型结合置信度加权的点云初始化和Kalman滤波式的预测‑更新条件机制，实现精准的相机控制和未观测区域的重建。

**💡 创新点**

①置信度加权的点云投影潜在向量用于初始化扩散过程；②Kalman-inspired的预测‑更新结构在每个DiT块中融合相机姿态与噪声点云，实现对几何不确定性的自适应校正；③在预训练的视频插值模型上进行微调，保证几何一致性。

**🔧 技术方法**

预训练的视频插值模型、基于ViT的DiT网络、Kalman滤波预测‑更新模块、置信度加权投影点云、Rectified Flow训练目标以及梯度正则化。

**📊 数据集**

CO3D‑Hydrant、CO3D‑Teddybear、DL3DV（训练与测试），以及RealEstate10k、GraspNet、CO3D、DL3DV（跨域测试）。

**📈 对比分析**

与多种回归式基线（PixelSplat、MvSplat等）和扩散式基线（CameraCtrl、View‑Crafter、Uni3C）以及大规模预训练模型（SeVA、Gen3R）在PSNR、LPIPS、SSIM及相机位置/姿态误差上进行对比。结果显示ConfCtrl在视觉质量和相机控制精度上均优于所有基线，并在跨域测试中保持强泛化能力。

**⚠️ 局限性**

受限于现有VAE在视频扩散框架中的设计，难以处理大幅度相机运动导致的突变视角；未来需改进或替代VAE以适应此类场景。

---

## 323. Joint Precoding and Phase-Shift Optimization for Beyond-Diagonal RIS-Aided ISAC System

**arXiv ID:** 2603.09265 | [PDF](https://arxiv.org/pdf/2603.09265v1)

**作者:** Xuejun Cheng `[一作]` (Shandong University), Ju Liu `[通讯]` (Shandong University)

**通讯引用:** 9102 | [OpenAlex ID](https://openalex.org/A5100763003)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于BD‑RIS的多用户ISAC系统联合预编码与相位偏移优化框架，利用多用户干扰管理与波束增益近似实现通信与感知性能的灵活平衡。

**💡 创新点**

创新点包括①将传统高阶非凸目标转化为可闭式求解的凸型损失函数；②在BD‑RIS结构下引入多用户干扰管理与波束增益近似方法；③设计了可闭式解的交替优化（AO）算法，显著降低计算复杂度。

**🔧 技术方法**

采用了交替优化（AO）算法、增广拉格朗日乘子法（ADMM）与对称酉投影等数学工具，结合Rayleigh与LOS信道模型，处理BD‑RIS相位矩阵约束与预编码向量的联合优化。

**📊 数据集**

通过仿真场景数据（BS、BD‑RIS位置、用户数量、天线数等），使用Rayleigh小尺度衰落与LOS感知信道，未使用公开真实数据集。

**📈 对比分析**

对比了BD‑RIS、全连接BD‑RIS（FBD‑RIS）、分组BD‑RIS（GBD‑RIS）与传统对角RIS（D‑RIS）在通信速率与感知波束增益之间的折衷；结果显示BD‑RIS尤其是FBD‑RIS在两方面均优于其余方案，能够在相同功率预算下实现更高的多用户总速率与更集中感知波束。

**⚠️ 局限性**

局限性主要在于：①仅考虑单目标感知，未探讨多目标情形；②假设完全CSI，未讨论不完美通道估计导致的鲁棒性问题；③仿真基于理想化信道模型，缺乏真实实验验证。

---

## 324. Causally Sufficient and Necessary Feature Expansion for Class-Incremental Learning

**arXiv ID:** 2603.09145 | [PDF](https://arxiv.org/pdf/2603.09145v1)

**作者:** Zhen Zhang `[一作]` (Southwest Jiaotong University), Tianrui Li `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 26602 | [OpenAlex ID](https://openalex.org/A5070559820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究扩展式类增量学习中的特征碰撞问题，提出基于因果必要与充分性的 CPNS 正则化来指导特征扩展，并给出了双范围反事实生成器实现。

**💡 创新点**

① 将 PNS 概念扩展为 CPNS，量化任务内因果完整性与任务间可分离性；② 设计双范围反事实生成器实现可观测的因果度量；③ 引入三阶段可插拔优化策略提升扩展式 CIL 的鲁棒性。

**🔧 技术方法**

结构因果模型、概率必要与充分性 (PNS)、对抗式反事实生成、双网络对齐、KL 约束、投影器、三阶段训练、基线扩展法（DER、FOSTER、TagFex 等）。

**📊 数据集**

CIFAR‑100、ImageNet‑100、ImageNet‑1000、4cCIFAR‑100/4cImageNet‑100、2cImageNet‑1000、CUB200 细粒度数据集。

**📈 对比分析**

与多种基线扩展式 CIL 方法（DER、FOSTER、TagFex 等）在 last/average 准确率上对比；加入 CPNS 后在所有场景均提升 1–4%，在细粒度数据上提升 2–3%；ablation 实验验证各模块贡献。

**⚠️ 局限性**

需要额外的投影器与双网络导致计算开销增加；对超参数 λ、γ 敏感；三阶段训练流程易出现梯度失衡；理论依赖单调性假设，极端任务重叠或分布跳变时效果仍有限。

---

## 325. Semantic Level of Detail: Multi-Scale Knowledge Representation via Heat Kernel Diffusion on Hyperbolic Manifolds

**arXiv ID:** 2603.08965 | [PDF](https://arxiv.org/pdf/2603.08965v1)

**作者:** Edward Izgorodin `[一作]` `[通讯]` (Mnemoverse.AI), Edward Izgorodin (Mnemoverse.AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Semantic Level of Detail（SLoD）框架，通过在Poincaré球上热核扩散实现知识图谱的连续多尺度抽象。

**💡 创新点**

创新在于将热核扩散与超曲几何结合，形成可连续放大/缩小的语义LOD，并通过谱间隙自动识别抽象层级边界。

**🔧 技术方法**

使用热核扩散、弗雷歇平均、Poincaré球嵌入、图拉普拉斯谱分解、Chebyshev多项式近似以及多中心混合表示。

**📊 数据集**

在合成层级SBM、WordNet 3.0 名词层级以及可扩展的知识图谱上验证。

**📈 对比分析**

与Louvain、Leiden、Markov Stability等基线对比，SLoD在合成图上ARI可达1.00，WordNet层级检测与真实深度相关系数τ≈0.79，检出多层边界且不需要手动设定分辨率。

**⚠️ 局限性**

主要局限在于对树状假设的依赖、谱分解截断误差、仅适用于静态图谱及对密集图谱的可扩展性不足。

---

## 326. Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models

**arXiv ID:** 2603.08859 | [PDF](https://arxiv.org/pdf/2603.08859v1)

**作者:** John Cooper `[一作]` (University of Wisconsin Madisom), Frederic Sala `[通讯]` (University of Wisconsin Madisom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究混合序列模型（Transformer层+状态空间模型层）的表达能力与内存效率，并通过理论证明与实验验证其在核心函数组合任务上的优势；

**💡 创新点**

构造可在对数规模参数和子线性工作内存下解决选择性复制与关联回忆与解码等任务的混合模型，并证明纯Transformer和纯SSM在同类任务上存在本质的参数/窗口限制；

**🔧 技术方法**

采用Transformer注意力与SSM层的混合架构（如Mamba+GPTNeoX），结合信息论与通信复杂度分析证明限制，实验使用RoPE位置编码、滑动窗口注意力、梯度下降训练；

**📊 数据集**

主要使用人工合成数据集：选择性复制（Selective Copying）、关联回忆与解码（Associative Recall with Decoding）、多键关联回忆（MKAR）、针状搜索（Needle-in-a-Haystack），以及不同长度与分布偏移的变体；

**📈 对比分析**

在相同或相近参数量的纯Transformer、纯SSM与混合模型进行对比，混合模型在所有任务上均能用更少参数实现同等或更高准确率，并在长度泛化与OOD泛化上表现更佳；

**⚠️ 局限性**

研究仅局限于合成任务，关注滑动窗口Transformer，未验证在更复杂的自然长文本或多样注意力模式下的效果。

---

## 327. Rate-Distortion Bounds for Heterogeneous Random Fields on Finite Lattices

**arXiv ID:** 2603.09833 | [PDF](https://arxiv.org/pdf/2603.09833v1)

**作者:** Sujata Sinha `[一作]` (Virginia Tech), Lingjia Liu `[通讯]` (Virginia Tech)

**通讯引用:** 7439 | [OpenAlex ID](https://openalex.org/A5027237940)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了适用于有限格点上异质随机场的有限块长率失真(RD)理论框架，并将块划分（tile）约束纳入源模型；

**💡 创新点**

创新点包括：1）基于分块均匀化的随机场模型；2）针对块划分的非渐近可达与逆推界；3）二阶渐近展开与色散分解；4）逆水位填充的谱解析与闭式色散；5）将理论结果与实际误差有限压缩器相结合，提供设计指导；

**🔧 技术方法**

使用技术主要有：有限块长信息理论、随机编码与失真倾斜信息密度、Berry–Esseen 近似、Gaussian 源的谱分解与逆水位填充、块状协方差分解及区域级编码；

**📊 数据集**

使用数据集包括：SDRBench 的多种科学仿真字段（如 NYX 宇宙学模拟的密度场）、以及基于分块均匀化模型生成的合成随机场；

**📈 对比分析**

将理论界与 SZ3、ZFP、SPERR 等误差有限压缩器的经验 RD 曲线以及 1D GRP、2D GRF 的传统均匀模型界进行对比，结果显示理论界与块大小匹配的压缩器性能接近上界，压缩器未能突破相应的理论极限；

**⚠️ 局限性**

局限性在于：仅针对 Gaussian 块对角协方差的分块均匀化模型；未考虑非高斯分布、跨块相关性和功能性失真；以及需预先确定块划分和足够多的样本来估计区域参数。

---

## 328. Optimal partition selection with Rényi differential privacy

**arXiv ID:** 2603.09167 | [PDF](https://arxiv.org/pdf/2603.09167v1)

**作者:** Charlie Harrison `[一作]` (Google), Pasin Pasin Manurangsi `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在近似Renyi差分隐私（RDP）框架下的最优分区选择机制，扩展了单分区单用户的最优算法到多分区、加权分区，并设计了可替代高斯机制的SNAPS算法；

**💡 创新点**

创新点在于：① 在有限α下给出RDP下的最优分区选择原语；② 针对加权分区设计SNAPS机制，实现对L^r敏感度的精确控制；③ 证明并量化了使用可加噪声机制与非加噪声机制在α<∞时的隐私“成本”差异；

**🔧 技术方法**

主要技术包括：RDP和近似RDP的定义与组合性、最优分区选择的离散优化（求解p^*），SNAPS机制的L^r感知概率函数构造、凸优化求解加噪声分布、数值实验与水位填充算法计算近似Renyi散度；

**📊 数据集**

实验使用的公共数据集包括Reddit、某些基准查询日志数据等；

**📈 对比分析**

将SNAPS嵌入已有的自适应分区选择算法MAD2R和PolicyGaussian，实验显示在并行与序列两种工作模式下均实现了比原来高斯子算法更高的分区释放率和更好的误差性能；

**⚠️ 局限性**

局限性包括：① 对于Δ1≠1（用户可提交多分区）的情况，尚不存在全局最优机制；② 只对单点概率约束的可加噪声机制进行了优化，无法直接覆盖更复杂的多点约束；③ 需要进一步研究PLD兼容性与更紧的隐私计量方法。

---

## 329. What is Missing? Explaining Neurons Activated by Absent Concepts

**arXiv ID:** 2603.09787 | [PDF](https://arxiv.org/pdf/2603.09787v1)

**作者:** Robin Hesse `[一作]` (Max Planck Institute for Informatics), Stefan Roth `[通讯]` (Technical University of Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究深度神经网络中对“缺失概念”编码的现象，提出两种对现有可解释方法（归因、特征可视化）的简单扩展——非目标归因和最小化特征可视化，以揭示这些被忽略的负向因果关系。

**💡 创新点**

创新点在于首次系统化定义并证明“编码缺失”在DNN中的普遍性，提出通过对非目标类进行归因和利用最小化特征可视化两种通用技巧来捕获这种负向关系，并将其用于模型去偏与鲁棒性提升。

**🔧 技术方法**

技术方法包括：1）基于结构因果模型的编码缺失定义；2）利用梯度归因对不同类别样本计算非目标归因；3）通过对网络层前激活进行最小化来得到最小激活图；4）将上述信息用于归因先验去偏；5）对图像数据进行补丁插入实验验证。

**📊 数据集**

实验使用的主要数据集为ImageNet‑1k（用于验证编码缺失的普遍性和对细粒度分类的作用），Toy RGB 图片（绿色像素分类）作为演示，以及合成偏置的ISIC皮肤病变数据集（用于去偏实验）。

**📈 对比分析**

与传统的目标归因、最大化特征可视化、逻辑 NOT 归因等方法相比，最小化特征可视化在插入补丁时可显著降低通道激活（平均下降至约0.03‑0.94），且在去偏实验中加入非目标归因的归因先验可将模型在无偏、反向偏差验证集上的准确率提升至与无偏训练模型相当（~0.81）。

**⚠️ 局限性**

局限性包括：1）非目标归因需要为每个类别或概念多次计算归因，计算开销较大；2）方法假设概念与单个神经元轴对齐，实际可能更复杂；3）目前仅在图像分类任务中验证，扩展至更复杂任务（如语言模型、生成模型）仍待研究；4）对对称激活函数的模型可能不易判定缺失编码的符号。

---

## 330. Impact of Markov Decision Process Design on Sim-to-Real Reinforcement Learning

**arXiv ID:** 2603.09427 | [PDF](https://arxiv.org/pdf/2603.09427v1)

**作者:** Tatjana Krau `[一作]` (Institute of Production and Informatics), Frieder Heieck `[通讯]` (Institute of Production and Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统评估不同MDP设计选择（状态表示、奖励函数、终止准则、动力学模型）对工业过程控制中RL从模拟到真实硬件的转移影响，并在颜色混合实验中验证其效果。

**💡 创新点**

从MDP整体设计角度系统研究仿真到现实的转移瓶颈，提出包含目标状态、相对比例状态、简单欧氏距离奖励以及物理基动力学模型的最佳配置，为工业RL部署提供实用设计准则。

**🔧 技术方法**

采用强化学习（PPO）、目标条件RL、噪声与对抗扰动鲁棒机制，三种颜色混合模型（lerp、KM、WGM）以及硬件实时摄像测色。

**📊 数据集**

自制颜色混合任务的RGB基色（青、品红、黄）与四个目标色（C1–C4），仿真与真实硬件多次实验数据。

**📈 对比分析**

通过仿真指标FP、T7.5、CV、NM并加权评分，以及硬件指标RGB距离、步数、成功率进行比较；最佳配置在硬件上可达约50%成功率，物理动力学模型显著提升转移性能。

**⚠️ 局限性**

仅在单一颜色混合任务上实验，目标色在仿真可达范围之外导致直接转移评估受限；未考虑更复杂目标分布，模型校准仍待改进。

---

## 331. Multimodal Graph Representation Learning with Dynamic Information Pathways

**arXiv ID:** 2603.09258 | [PDF](https://arxiv.org/pdf/2603.09258v1)

**作者:** Xiaobin Hong `[一作]` (State Key Laboratory for Novel Software Technology, Nanjing University), Wenzhong Li `[通讯]` (State Key Laboratory for Novel Software Technology, Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于可学习伪节点的动态信息通道框架（Dynamic information Pathways，简称DIP），用于多模态图（图节点同时携带图像、文本等多模态特征）的表征学习。框架通过在每种模态内部引入专属伪节点，实现局部与全局的内模态扩散，并通过伪节点间的交互完成跨模态聚合，从而在保持线性复杂度的同时获得更丰富、可调的消息传播。

**💡 创新点**

创新点主要包括：① 引入模态专属伪节点作为轻量级中介，使得消息传递不再受限于固定邻接关系，可动态地根据节点相似度重组通路；② 通过共享状态空间和多通道近似路径积分实现节点与伪节点之间的距离计算，避免了 O(n²) 的全连通代价；③ 设计了内模态扩散（G2P + P2G）与跨模态聚合（P2P）两条动态通道，使得模型在保持可扩展性的同时显著提升了对跨模态依赖的捕捉能力。

**🔧 技术方法**

技术手段包括：
- 视觉编码器（CLIP、ViT、DINOv2、ImageBind）与文本编码器（CLIP、T5、ImageBind）用于获取高质量的模态嵌入；
- 共享状态空间与近似多通道路径积分构造节点与伪节点的相似度；
- 递归多步消息传递（L 步），并在每步使用可学习的伪节点来完成全局信息扩散；
- 线性投影+concat实现模态融合；
- 任务头（softmax 2‑层 MLP 用于节点分类，内积 + sigmoid 用于链预测）。

**📊 数据集**

实验使用的多模态图数据集有：
- 链预测任务：Amazon‑Sports、Amazon‑Cloth、Goodreads‑LP；
- 节点分类任务：Ele‑Fashion、Goodreads‑NC。

**📈 对比分析**

与传统 GNN（GCN、GraphSAGE、BUDDY）以及多模态 GNN（MMGCN、MGAT、UniGraph2）对比，DIP 在所有链预测数据集上均取得最高的 MRR、Hits@1 和 Hits@10，提升幅度可达 +2.8%/ +5.8%；在节点分类任务中，准确率提升 2–3%，最高可达 89.5%（Ele‑Fashion）与 85.4%（Goodreads‑NC）。模型的时间复杂度与 GCN 相近，内存占用更低，证明了其高效性。

**⚠️ 局限性**

局限性包括：① 伪节点数量需要通过交叉验证手工调参；② 当前框架主要针对同一模态的节点，尚未扩展到异构节点类型的多模态图；③ 在极大规模图（百万级节点）上仍需进一步优化伪节点的动态更新机制；④ 对编码器的依赖较强，若采用不同的基础编码器，性能波动可能较大。

---

## 332. STONE Dataset: A Scalable Multi-Modal Surround-View 3D Traversability Dataset for Off-Road Robot Navigation

**arXiv ID:** 2603.09175 | [PDF](https://arxiv.org/pdf/2603.09175v1)

**作者:** Konyul Park `[一作]` (Seoul National University), Jun Won Choi `[通讯]` (Seoul National University)

**通讯引用:** 3969 | [OpenAlex ID](https://openalex.org/A5102839991)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了大规模离地机器人三维通行性数据集 STONE，并设计了一套自动化流水线在三维空间生成通行性地图，同时构建了基于单模态与多模态感知的基线评测基准。

**💡 创新点**

创新点在于：①基于机器人行驶轨迹的三维通行性自动标注方法，利用多变量高斯分布与马氏距离实现无人工标注的几何驱动标注；②引入360°全景 LiDAR、六摄像头与三束 4D 雷达的多模态全景感知方案；③为离地导航提供可扩展的三维通行性评测基准，支持单模态与多模态融合的系统比较。

**🔧 技术方法**

技术要点包括：多帧 LiDAR 聚合 + Poisson 曲面重建；提取坡度、海拔与粗糙度三种几何特征；利用轨迹样本估计高斯分布并阈值化马氏距离生成标签；精确的时序同步（LiDAR 驱动摄像机触发）和多传感器外参标定；基线模型使用 OccFormer、TPVFormer、OccFusion 等网络在 PyTorch‑mmdetection3d 环境下训练。

**📊 数据集**

使用的数据集为本文自建的 STONE，包含 43 条序列、约 50,878 帧、4 种场景（农田、山区、湖泊、施工现场），并在不同昼夜条件下收集；对比实验中还引用了 RUGD、RELLIS‑3D 等公开数据集的相关方法。

**📈 对比分析**

通过对比实验展示了单模态与多模态的性能差异：在 IoU 评价指标下，LiDAR 单模态平均 mIoU 最高为 38.1%；将摄像头与 LiDAR 融合后 mIoU 提升至 39.6%；摄像头+雷达的基线 mIoU 为 33.1%，说明多模态融合可进一步提升通行性预测精度。

**⚠️ 局限性**

局限性包括：数据集覆盖的地形种类有限（未包含雪、雨、雾等恶劣天气）；样本分布相对集中在首尔郊区，缺乏更广泛的地理多样性；未来工作计划扩充场景、加入极端天气数据以提升鲁棒性。

---

## 333. Feedback Does Not Increase the Capacity of Approximately Memoryless Surjective POST Channels

**arXiv ID:** 2603.08886 | [PDF](https://arxiv.org/pdf/2603.08886v1)

**作者:** Xiaojing Zhang `[一作]`, Guanghui Wang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了一类有限状态通道——POST 通道，证明在近似无记忆且满足齐射条件的情况下，反馈不提升信道容量。

**💡 创新点**

创新点在于把 Shannon 的无反馈不增益结论推广到更广泛的记忆通道族，揭示了齐射与近似无记忆的条件足以保证反馈无效，且提供了通用的矩阵与凸包分析框架。

**🔧 技术方法**

主要技术包括：矩阵表示法、矩阵秩与扰动分析、凸优化求解、对输出过程的线性仿真与可实现性证明，以及利用信息论的互信息与马尔可夫链性质。

**📊 数据集**

该工作为理论性研究，无需实验数据集，所有结论通过解析证明与数值验证（如示例 1、2）得到验证。

**📈 对比分析**

比较方法：将反馈容量 C_f(Q) 与非反馈容量 C(Q) 的单字母与 n‑字母表达式对比。结果表明在满足假设下两者相等，说明反馈不提升容量。

**⚠️ 局限性**

局限性：仅适用于近似无记忆且齐射的 POST 通道；当输出字母数大于输入字母数、或通道不满足齐射/全秩条件时结论不一定成立；扰动阈值 δ 的具体取值需要进一步分析，且在实际应用中可能较难满足。

---

## 334. Transductive Generalization via Optimal Transport and Its Application to Graph Node Classification

**arXiv ID:** 2603.09257 | [PDF](https://arxiv.org/pdf/2603.09257v1)

**作者:** MoonJeong Park `[一作]` (Graduate School of Artificial Intelligence), Dongwoo Kim `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在分布无关的转导学习框架下，基于表示学习的全局与类别级别的泛化界限，并对图神经网络（GNN）的深度泛化关系进行理论与实验分析。

**💡 创新点**

创新点主要包括：① 利用最优运输（Wasserstein）距离构造可计算且与经验泛化高度相关的转导泛化界限；② 通过深度相关的 Wasserstein 上界揭示 GNN 深度与泛化误差的非单调关系；③ 充分利用训练时可获得的未标记测试特征，打破传统独立同分布假设，得到更紧的界限。

**🔧 技术方法**

技术手段包括：最优运输与1-Wasserstein距离、转导学习理论、图卷积聚合、GNN架构（SGC、GCN、GCNII、GAT、GraphSAGE）、经验分布推导、深度分析与实验评估。

**📊 数据集**

实验使用了九个图数据集，其中五个同类（Cora、CiteSeer、PubMed、Computers、Photo）和四个异类（Squirrel、Chameleon、Roman-empire、Amazon-ratings）。

**📈 对比分析**

与传统的 PAC‑Bayesian、转导 Rademacher 复杂度等基线进行对比，通过 rank‑correlation 衡量界限与实际泛化误差的关联；实验结果表明本文的全局与类别级别界限与实测误差高度相关，传统基线在大多数场景下表现不佳。

**⚠️ 局限性**

局限性包括：仅在图节点分类任务验证，未展示对其他转导任务或非图结构的推广；对极深模型时界限可能过于松散；未深入探讨标签不平衡对 Wasserstein 距离和界限的影响。

---

## 335. Chain of Event-Centric Causal Thought for Physically Plausible Video Generation

**arXiv ID:** 2603.09094 | [PDF](https://arxiv.org/pdf/2603.09094v1)

**作者:** Zixuan Wang `[一作]` (Sichuan University), Yinjie Lei `[通讯]` (Sichuan University)

**通讯引用:** 3050 | [OpenAlex ID](https://openalex.org/A5102831936)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于事件链推理与跨模态提示的物理可行视频生成框架，将复杂物理现象拆解为因果连贯的事件序列，并映射到视频生成过程；

**💡 创新点**

创新点在于通过物理驱动的事件链推理嵌入物理公式以消除因果歧义，同时通过跨模态提示动态生成语义‑视觉提示，保证视频在时间上的物理一致性；

**🔧 技术方法**

采用Chain‑of‑Thought推理、物理公式检索与约束、场景图更新、交互式关键帧生成以及视觉‑语言提示融合的扩散模型；

**📊 数据集**

使用的评测数据集包括PhyGenBench（160条描述，涵盖力学、光学、热学、材料四大领域）和VideoPhy（688条人类验证的物理交互提示）；

**📈 对比分析**

与多种视频基础模型及物理可行生成模型在PhyGenBench和VideoPhy上进行比较，整体PCA得分达0.66、SA/PC分数分别为49.3%/79.5%，显著超越SOTA；

**⚠️ 局限性**

局限在于对多物理律组合场景的推理不足，当前基础模型在组合物理推理上的能力有限，导致部分多物理律情境下的生成失败。

---

## 336. Using Vision Language Foundation Models to Generate Plant Simulation Configurations via In-Context Learning

**arXiv ID:** 2603.08930 | [PDF](https://arxiv.org/pdf/2603.08930v1)

**作者:** Heesup Yun `[一作]` (University of California), J. Mason Earles `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基准，用于评估视觉语言模型（VLM）从无人机遥感图像生成用于植物模拟的JSON配置，从而实现数字孪生中的三维农作物模拟。

**💡 创新点**

创新点在于首次将VLM直接用于从图像推断并生成完整的结构化JSON配置，结合多种上下文提示和LoRA微调，评估其在三维模拟任务中的表现。

**🔧 技术方法**

使用的技术包括Gemma 3与Qwen3‑VL等开源VLM、五种增量式上下文学习方法、LoRA参数高效微调，以及Helios 3D程序生成的模拟图像。

**📊 数据集**

数据集由Helios 3D生成的合成豌豆田地图像（1120张）和真实加利福尼亚无人机正射影像（10个样本）组成，后者仅提供部分标注。

**📈 对比分析**

通过JSON完整性、几何误差（如Chamfer距离）与生物物理误差（叶绿素含量等）等指标与均值猜测基线及不同模型规模进行比较，发现大模型在几何推断上更好，但在生物物理参数估计上仍远低于基线。

**⚠️ 局限性**

局限性包括模型在图像推断上的误差仍偏高，特别是叶片生理参数预测；模型易受上下文偏置影响，盲目基线往往比实际图像评估更好；并且在真实数据上表现仍落后于合成数据。

---

## 337. AgenticCyOps: Securing Multi-Agentic AI Integration in Enterprise Cyber Operations

**arXiv ID:** 2603.09134 | [PDF](https://arxiv.org/pdf/2603.09134v1)

**作者:** Shaswata Mitra `[一作]` (University of Alabama), Shahram Rahimi `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种针对多智能体AI在企业网络安全中的集成安全框架，通过攻击面分解、设计五条防御原则并在SOC流程中实现，显著降低可利用的信任边界。

**💡 创新点**

创新点在于发现所有已知攻击向量集中于工具编排与内存管理两大集成面，基于此提出五条防御原则并将其嵌入MCP结构的SOAR体系，首次实现对MAS安全的全层级覆盖与可量化的边界压缩。

**🔧 技术方法**

使用了多智能体系统分析、Model Context Protocol、签名清单、权限分级、共识验证、区块链式审计、同步与完整性校验、基于角色的内存隔离等技术。

**📊 数据集**

评估以SOC工单、SIEM、EDR、CTI等企业安全工具的日志与事件数据为实验素材，构造攻击路径与覆盖分析，但未使用公开数据集。

**📈 对比分析**

通过覆盖矩阵、攻击链跟踪与信任边界计数对比，展示相较于扁平MAS基础，信任边界减少72%；性能指标未给出，仅基于结构分析得出。

**⚠️ 局限性**

局限包括单点信任锚点、共识验证延迟、验证循环易被攻击、跨组织数据馈送未受保护、缺乏标准化评估数据集与基准、运行时开销未测评等。

---

## 338. Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction

**arXiv ID:** 2603.09930 | [PDF](https://arxiv.org/pdf/2603.09930v1)

**作者:** Yao Zhang `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**通讯引用:** 3922 | [OpenAlex ID](https://openalex.org/A5069437467)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可解释的细粒度文本-动作检索框架，将关节角度表示映射为结构化伪图像，并采用令牌-补丁的晚期交互进行匹配。

**💡 创新点**

创新点在于：①使用解耦的关节角度特征而非原始坐标，消除全局位移干扰；②将每个关节映射到ViT补丁，实现按部位对齐；③结合MaxSim晚期交互和掩码语言模型正则化，提升上下文丰富度和可解释性。

**🔧 技术方法**

核心技术包括：Vision Transformer（ViT）编码运动图像；DistilBERT编码文本；MaxSim令牌-补丁相似度计算；MLM正则化训练；以及产品量化/二进制哈希压缩。

**📊 数据集**

在HumanML3D和KIT‑ML两个公开动作-文本检索数据集上进行实验。

**📈 对比分析**

与TMR、CAR、KinMo、MoPatch、SGAR等基线比较，取得Recall@10和MedR等指标的显著提升，尤其在大模型Ours‑L版本上实现了SOTA表现，同时提供可视化的令牌-补丁对应图，增强可解释性。

**⚠️ 局限性**

主要局限在于需要存储每个动作的稠密补丁嵌入，导致离线索引存储量显著增加；尽管可通过量化压缩，但仍需进一步优化检索效率和大规模部署。

---

## 339. Epistemic Closure: Autonomous Mechanism Completion for Physically Consistent Simulation

**arXiv ID:** 2603.09756 | [PDF](https://arxiv.org/pdf/2603.09756v1)

**作者:** Yue Wua `[一作]`, Jizhong Huang `[通讯]` (Institute for the Conservation of Cultural Heritage School of Cultural Heritage and Information Management Shanghai University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种神经符号生成代理，能够在无结构文献中自动提取、验证并补全多物理耦合模型，避免物理幻觉；

**💡 创新点**

核心创新在于将物理定律封装为可模块化的“构成技能”，并通过链式思维和无量纲尺度分析自动推理、裁剪冗余机制并补全缺失物理，从而实现自主的认知监督；

**🔧 技术方法**

采用大型语言模型（LLM）与检索增强生成（RAG）技术进行知识提取，构造构成技能；随后使用链式思维（CoT）、无量纲尺度（Deborah数）分析进行推理；最后将验证后的技能编译为FEniCSx的弱形式实现数值求解；

**📊 数据集**

以低渗透性岩石（Rothbach砂岩）的热水压缩问题为实验案例，利用公开文献中的力学、热学、流体参数；

**📈 对比分析**

将生成的模型与传统“文献仅用”模型对比，后者因盲目采用“不可压缩”假设导致错误预测岩石破坏；神经符号代理通过补全达西耗散机制得到稳定应力路径，预测结果与实验一致，性能显著优于基线；

**⚠️ 局限性**

局限性包括：对文献的依赖性，如果源文本本身存在模型错误，验证层可能无法检测；当前只处理了确定性技能，尚未处理模型形式错误；以及对大规模多物理耦合场景的可扩展性与计算成本仍待评估。

---

## 340. MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning

**arXiv ID:** 2603.09892 | [PDF](https://arxiv.org/pdf/2603.09892v1)

**作者:** Yiyang Lu `[一作]` (Chinese University of Hong Kong), Hongyuan Zha `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 24075 | [OpenAlex ID](https://openalex.org/A5046703129)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在LLM的持续微调中，设计了一套基于Ebbinghaus遗忘曲线的记忆衰退模型和自适应重放调度，能够在不增加显著参数的前提下减轻灾难性遗忘。

**💡 创新点**

创新点在于将人类记忆理论（样本级记忆强度与间隔扩展）与持续学习结合，提出了可解释且无须频繁评估的动态重放调度策略；同时在LoRA参数高效微调框架下实现。

**🔧 技术方法**

主要技术包括：样本级记忆衰退公式、基于损失的记忆更新、指数衰减的重放比例、间隔扩展调度、LoRA参数高效微调、EMA-归一化损失、量化采样。

**📊 数据集**

使用了多种任务和数据集：GSM8K、MATH、MMLU、SQuAD、ARC、ARC、以及一个包含11个领域（AGNews、SQuAD、SciQ、BoolQ、ARC、MATH子集等）的长序列持续学习序列，评估基准覆盖推理、问答、选择题等。

**📈 对比分析**

与无重放、固定重放、基于损失和基于准确度的重放方法对比，MSSR_full在3任务和11任务设置下在大多数任务和多种模型（Qwen2.5‑7B、Gemma2‑9B、Llama‑3.1‑8B等）上均取得1–3点的精度提升，尤其在早期任务和推理类任务上表现最为显著；重放比例动态调节和间隔扩展显著降低了额外计算开销。

**⚠️ 局限性**

主要局限在于需要维护每个样本的记忆状态，虽然开销小但在极大模型或极长训练周期下仍存在一定的计算和存储负担；同时方法依赖于损失EMA估计，若任务难度或数据分布极度不稳定时可能需进一步调优。

---

## 341. On Catastrophic Forgetting in Low-Rank Decomposition-Based Parameter-Efficient Fine-Tuning

**arXiv ID:** 2603.09684 | [PDF](https://arxiv.org/pdf/2603.09684v1)

**作者:** Muhammad Ahmad `[一作]` (University of British Columbia), Yankai Cao `[通讯]` (University of British Columbia)

**通讯引用:** 1732 | [OpenAlex ID](https://openalex.org/A5000332723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了参数高效微调（PEFT）在连续学习中的灾难性遗忘问题，并比较了LoRA、PiSSA、LoRETTA和WeGeFT在ViT模型上的表现。

**💡 创新点**

发现更新子空间的几何和参数化决定了遗忘程度，并证明张量分解和预训练对齐能够显著减轻遗忘。

**🔧 技术方法**

采用低秩分解、张量训练（LoRETTA）、结构对齐更新（WeGeFT）以及主奇异值更新（PiSSA）等PEFT技术。

**📊 数据集**

使用了CUB-200-2011、EuroSat、Intel图像分类数据集和体育图像分类数据集，对ViT进行四个连续任务的微调与评估。

**📈 对比分析**

通过全微调作为基准，计算平均遗忘和最终准确率进行对比，LoRETTA和WeGeFT在极低参数预算下表现最佳，LoRA在高秩时能降低遗忘。

**⚠️ 局限性**

实验仅限于图像分类任务，未验证多模态或更大规模任务的泛化，也未探讨不同预训练模型的适用性。

---

## 342. Logos: An evolvable reasoning engine for rational molecular design

**arXiv ID:** 2603.09268 | [PDF](https://arxiv.org/pdf/2603.09268v1)

**作者:** Haibin Wen `[一作]` (City University of Hong Kong), Ye Wei `[通讯]` (City University of Hong Kong)

**通讯引用:** 26862 | [OpenAlex ID](https://openalex.org/A5100334733)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个紧凑的分子推理模型Logos，能够在给定自然语言描述时自动生成符合化学规律的分子结构，并在生成过程中输出可审计的推理链；

**💡 创新点**

创新点在于将多步骤推理与化学一致性深度融合：①利用大型教师模型自动生成链式推理（CoT）进行自蒸馏；②在监督阶段让模型学习输出推理块+结构；③在强化学习阶段加入化学奖励（RDKit校验、结构相似度等），使模型在保持可解释性的同时达到高合法性与精确度；

**🔧 技术方法**

技术包括三阶段训练：自蒸馏（CoT distillation）、监督微调（SFT）和分子聚焦的GRPO强化学习；使用Transformer基础结构、Bfloat16混合精度、vLLM推理；在奖励设计中结合JSON格式校验、长度惩罚、反作弊约束、精确匹配和指纹相似度；

**📊 数据集**

使用公开的化学数据库产生的 caption–structure 对（如ChEBI-20、PCdes）作为训练与评估数据集；在训练阶段还通过自检扩增 CoT 数据；

**📈 对比分析**

与四大通用LLM（DeepSeek-14B、Qwen3-32B、DeepSeek-R1、GPT-5）以及内部模型版本进行对比。Logos-1.5B（最终版）在合法性几乎达到100%（0.9996/0.9997），EM（Exact Match）分别为0.3406/0.3103，Logos-4B则提升到0.5588/0.5047，且在结构相似度（MACCS、Morgan）和分布逼近度（FCD）上均优于更大模型；

**⚠️ 局限性**

局限性包括：评估仅覆盖两种 caption‑to‑molecule 任务，未检验其它任务（如逆合成、反应预测）；推理质量未直接量化；只在英语环境下实验；自蒸馏对教师模型质量敏感；以及对大规模高通量筛选的延迟较高，难以满足极低时延需求。

---

## 343. TemporalDoRA: Temporal PEFT for Robust Surgical Video Question Answering

**arXiv ID:** 2603.09696 | [PDF](https://arxiv.org/pdf/2603.09696v1)

**作者:** Luca Carlini `[一作]` (Politecnico di Milano), Mobarak I. Hoque `[通讯]` (University of Manchester)

**通讯引用:** 758 | [OpenAlex ID](https://openalex.org/A5082689217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TemporalDoRA，一种针对外科手术视频问答的轻量级参数高效微调方法，利用时序多头注意力混合低秩瓶颈并仅对低秩分支进行权重分解；同时创建 REAL-Colon-VQA 这一带有内外模板问句的时间导向基准数据集。

**💡 创新点**

①在 PEFT 框架中首次在低秩瓶颈内嵌入时序多头注意力，实现帧间内容自适应聚合；②仅对可训练的低秩分支做权重分解，保持冻结主干的稳定性；③构建具备 In-Template 与 Out-of-Template 语义对照的临床 VideoQA 数据集，直接量化对语言变异的鲁棒性。

**🔧 技术方法**

Weight‑Decomposed Low‑Rank Adaptation (DoRA)、多头注意力（MHA）时序混合、LoRA/VeRA/AdaLoRA/ ST‑Adapter 等 PEFT 变体；在视觉编码器的注意力投影及 LLM 解码器上微调；使用 Qwen3‑VL‑2B、InternVL3‑1B、SurgViVQA 等多模态基础模型。

**📊 数据集**

REAL‑Colon‑VQA（6424 片段‑问答对，含 8 帧剪辑与 In/Out‑Template 语句），EndoVis18‑VQA（重新裁剪为 8 帧的手术视频问答），SurgViVQA。

**📈 对比分析**

通过在 REAL‑Colon‑VQA 与 EndoVis18‑VQA 的 In‑Template 与 Out‑Template 子集上对比 LoRA、DoRA、VeRA、AdaLoRA、ST‑Adapter、Zero‑Shot 等方法。TemporalDoRA 在 Out‑Template 上显著提升：如 Qwen3‑VL‑2B 的 ROUGE‑L 0.731（vs 0.653 ST‑Adapter）和关键词准确率 0.326（vs 0.304 LoRA）；参数增幅仅 0.22%（≈8.6 倍减少）。

**⚠️ 局限性**

在瓶颈中插入 MHA 会增加计算开销，尤其针对长视频；目前仅在视觉编码器层引入时序混合，未扩展到 LLM 解码器，导致部分语言偏差仍未彻底消除。

---

## 344. WS-Net: Weak-Signal Representation Learning and Gated Abundance Reconstruction for Hyperspectral Unmixing via State-Space and Weak Signal Attention Fusion

**arXiv ID:** 2603.09037 | [PDF](https://arxiv.org/pdf/2603.09037v1)

**作者:** Zekun Long `[一作]` (Griffith University), Jun Zhou `[通讯]` (Griffith University)

**通讯引用:** 19147 | [OpenAlex ID](https://openalex.org/A5100781212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 WS‑Net，一种针对弱信号光谱图像混合物的端到端深度解混算法

**💡 创新点**

创新点包括：①双分支编码器融合多分辨率小波、Mamba 状态空间模型与弱信号注意力；②可学习门控融合机制自动调节两条分支的贡献；③稀疏感知解码器结合 KL 散度正则化，强制弱端元与主端元的谱形分离；④针对弱信号崩溃问题的理论分析和设计。

**🔧 技术方法**

技术栈：多分辨率 Haar 与 Symlet‑3 小波分解、Mamba 状态空间模型、弱信号注意力（正向+逆向注意力融合）、门控融合（局部与全局）、稀疏感知解码器、Softmax 归一化约束、KL 散度正则化、RMSE/SAD 损失组合。

**📊 数据集**

数据集：合成光谱混合数据（S1）、真实场景 Samson、Apex 三个公开数据集，均包含弱反射率端元。

**📈 对比分析**

对比 6 大基线（FCLSU+VCA、Hyperweak、CNNAEU、DeepTrans、EDAA、EndNet、MiSiCNet）。WS‑Net 在 RMSE 上平均下降 55%，SAD 降低 63%，在弱信号端元（如 magnetite、water、road 等）误差显著低于所有对手，且在不同 SNR 条件下保持鲁棒性。

**⚠️ 局限性**

局限性：①仍依赖 VCA 等方法的端元初始化；②在大尺寸航空/卫星影像的可扩展性和实时性未验证；③跨传感器、跨场景的泛化能力需进一步评估；④对非线性残差的理论解释尚不充分。

---

## 345. Fly, Track, Land: Infrastructure-less Magnetic Localization for Heterogeneous UAV-UGV Teaming

**arXiv ID:** 2603.08926 | [PDF](https://arxiv.org/pdf/2603.08926v1)

**作者:** Valerio Brunacci `[一作]`, Tommaso Polonelli `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4`

**🎯 论文内容**

本文设计并实现了一套基于磁感应（MI）定位的全无外部基础设施的微型无人机（nano‑UAV）与四足地面机器人（UGV）协作平台，使无人机能够在移动地面平台上实现厘米级精度的悬停、跟踪与降落。

**💡 创新点**

创新点在于：①采用四频分复用的磁偶极子发射阵列与单一接收线圈的组合，在短距离内实现三维相对定位；②在nano‑UAV微控制器上实现20 Hz实时非线性最小二乘求解，兼顾计算量与精度；③将MI定位与惯性、光流、UWB等多模态融合，克服传统视觉/RF在GNSS‑失效或光照不佳环境下的局限。

**🔧 技术方法**

使用的主要技术包括：磁感应（MI）定位、频分复用（FDM）信号分离、快速傅里叶变换（FFT）提取幅值、Nelder–Mead优化求解逆磁场问题、扩展卡尔曼滤波（EKF）进行多模态融合、以及基于STM32F4 MCU的低功耗硬件实现。

**📊 数据集**

实验数据来自室内测控系统Vicon的运动捕捉标定，作为全局真值用于评估定位误差；未使用公开数据集。

**📈 对比分析**

与仅使用光流的基线（Flow-only）相比，MI+Flow方法在静态悬停/着陆场景下3D RMSE降至约5 cm，动态跟踪/着陆场景下RMSE保持在7–8 cm，成功率在静态实验中可达100%，动态实验中≥80%。

**⚠️ 局限性**

局限性在于当前磁模型仅考虑位置估计，对UGV快速偏航（yaw）导致的磁场变化未建模，导致在大角度转向时定位误差增大；此外，MI系统的工作距离仅约1 m，需与UWB等长程定位技术联合使用。

---

## 346. Clear, Compelling Arguments: Rethinking the Foundations of Frontier AI Safety Cases

**arXiv ID:** 2603.08760 | [PDF](https://arxiv.org/pdf/2603.08760v1)

**作者:** Shaun Feakins `[一作]` (Institute for Safe Autonomy), Phillip Morgan `[通讯]` (York Law School)

**通讯引用:** 255 | [OpenAlex ID](https://openalex.org/A5010733624)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文对前沿AI系统的安全案例（alignment safety case）进行系统评估，并提出将安全保证领域的成熟方法（如风险评估、GSN安全论证）引入前沿AI安全案例的思路，随后给出一个以欺骗式对齐和CBRN能力为示例的GSN安全论证案例。

**💡 创新点**

创新点在于：①批判性梳理了当前alignment safety case与传统安全保证方法的差异；②提出通过“through‑life”视角整合开发、部署与后期监控的安全论证框架；③首次将安全保证中的风险管理工具（风险日志、SIL、ALARP原理等）与AI安全案例结合，构建可复用的论证模式。

**🔧 技术方法**

主要技术：安全保证方法论（风险评估、故障树/故障模式分析、SIL/安全完整性等级、ALARP原理）；安全论证图谱技术（Goal Structuring Notation, GSN）；对AI安全的量化/定性评估工具（RLHF、对齐守门、后置监测）。

**📊 数据集**

本研究未使用具体机器学习数据集，而是基于已有安全保证标准（如ISO/IEC 26262、ISO 26262）与前沿AI安全报告（如Singapore Consensus、International AI Safety Report）中的理论案例与安全论证框架进行分析和示例构建。

**📈 对比分析**

方法比较：文章将传统安全保证流程与alignment safety case进行对照，指出前者强调从系统设计到退役的全生命周期风险管理，而后者往往聚焦部署时的安全证明。论文未给出定量性能指标，而是通过案例演示和论证逻辑的完整性来评估方法的可行性和可审计性。

**⚠️ 局限性**

局限性：①缺乏针对真实前沿AI模型的实证验证，安全论证仍以理论示例为主；②对新兴AI风险（如模型不确定性、生成式对抗攻击）仍未给出充分的定量评估方法；③将传统安全保证工具直接迁移到AI领域可能面临技术适配与监管缺失等挑战。

---

## 347. GNNs for Time Series Anomaly Detection: An Open-Source Framework and a Critical Evaluation

**arXiv ID:** 2603.09675 | [PDF](https://arxiv.org/pdf/2603.09675v1)

**作者:** Federico Bello `[一作]` (Universidad de la Republica), Federico Larroca `[通讯]` (Universidad de la Republica)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了可重复、模块化的开源框架GraGOD，用于统一评估图神经网络在时间序列异常检测（TSAD）中的性能与可解释性。

**💡 创新点**

框架首次整合多种阈值无关指标（VUS‑ROC/PR）、基于区间的精度/召回，并支持自定义图结构与模型，能够系统比较不同模型、拓扑与阈值策略。

**🔧 技术方法**

采用GNN（GCN、GDN、MTAD‑GAT）与传统GRU基线，利用图卷积、注意力机制、预测/重构损失以及多任务学习实现异常评分。

**📊 数据集**

在两大真实数据集上验证：TELCO（12条无显式图结构的通信指标）和SWaT（51条带有物理拓扑的水处理传感器）。

**📈 对比分析**

实验显示：当存在明确的图结构时，GNN（尤其是GDN和MTAD‑GAT）在VUS与阈值依赖指标上均优于GRU；注意力GNN对图拓扑不确定性更鲁棒；不同阈值策略会导致评估结果差异显著，VUS可避免此问题。

**⚠️ 局限性**

主要限制包括：基于重建/预测误差的代理评分导致阈值选择敏感；数据分布漂移和标签不准导致评估偏差；现有指标仍易被误导；缺乏直接针对异常检测的训练目标，需探索对比学习等更对齐的学习方式。

---

## 348. Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4

**arXiv ID:** 2603.08747 | [PDF](https://arxiv.org/pdf/2603.08747v1)

**作者:** Musa Cim `[一作]` (Pennsylvania State University), Mahmut Taylan Kandemir `[通讯]` (Pennsylvania State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性分析了两种FP4量化格式(MXFP4与NVFP4)在三种Qwen2.5模型规模（0.5B、7B、14B）中的量化敏感性；

**💡 创新点**

揭示MLP投影（上/下投影）始终为最敏感层，并且量化敏感性不一定集中在最后几层；

**🔧 技术方法**

采用组件级与块级隔离的量化实验方法，对Transformer中各投影类型及不同深度块进行逐一FP4量化；

**📊 数据集**

使用WikiText-2数据集（256个校准样本）评估Perplexity；

**📈 对比分析**

将FP4量化与FP16基准对比，发现MXFP4在更小模型上更敏感，整体量化对模型性能影响可由组件/块级调控，且不同格式和规模下相对敏感顺序保持一致；

**⚠️ 局限性**

仅在单一语言模型和单一指标（Perplexity）下验证，未覆盖多任务场景；量化方法为在线量化，缺乏对硬件FP4算子实际执行的评估。

---

## 349. Idempotent Slices with Applications to Code-Size Reduction

**arXiv ID:** 2603.09726 | [PDF](https://arxiv.org/pdf/2603.09726v1)

**作者:** Rafael Alvarenga de Azevedo `[一作]` (Federal University of Minas Gerais), Fernando Magno Quintão Pereira `[通讯]` (Federal University of Minas Gerais)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了幂等向后切片的概念，并给出了在GSA（Gated Static Single-Assignment）形式下提取此类切片的算法，随后通过把切片提取为单入口函数并合并相同切片，实现代码大小的压缩。

**💡 创新点**

创新点包括：①对幂等切片的正式定义和对其最大化属性的证明；②利用GSA显式的控制门指令，使得对任意CFG都能正确识别切片；③提出一种基于切片的代码压缩技术，能够合并非连续、跨块甚至跨函数的重复计算区域；④通过实验验证了该方法在LLVM Test Suite上能与现有技术竞争。

**🔧 技术方法**

技术手段主要包括：构建GSA表示（使用门指令γ、μ、η来显式控制依赖）；逆向依赖图遍历寻找切片；将切片提取为独立函数（outline）并生成接口；使用LLVM的哈希与等价检测合并相同的切片函数；配合成本模型（I、P、C）决定是否合并；最后利用LLVM的简化与去重Pass完成整体优化。

**📊 数据集**

实验使用LLVM Test Suite共计约3000个程序（包含134个单元测试、1548个回归测试、307个基准集以及18个真实应用），所有程序在Clang 17下编译，并通过LLVM 17.0.6实现该优化。

**📈 对比分析**

与两种基线（基于序列对齐的SBCR合并和LLVM IR Outliner）对比，衡量指标包括.text段大小、IR指令数、执行时间和编译时间。结果显示：在有收益的程序上平均可减少约10%的文本大小（最高可达‑7.24%），指令数同样降幅接近10%；运行时几乎无显著变化；编译时间增加约4%，且整体呈线性扩展。

**⚠️ 局限性**

局限性在于：算法在理论上是O(N²)，尽管实际近似线性；仅支持单入口单出口、局部函数切片；过度合并可能导致代码膨胀，需要成本模型调优；依赖GSA转换，若程序不易转换为GSA会增加额外开销。

---

## 350. An Analysis of Modern Web Security Vulnerabilities Inside WebAssembly Applications

**arXiv ID:** 2603.09426 | [PDF](https://arxiv.org/pdf/2603.09426v1)

**作者:** Lorenzo Corrias `[一作]` (University of Cagliari), Giorgio Giacinto `[通讯]` (University of Cagliari)

**通讯引用:** 7085 | [OpenAlex ID](https://openalex.org/A5075367917)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文通过构造多套 WebAssembly（WASM）模块与 Web 前端的实验环境，展示了低级内存错误（如缓冲区溢出、Use‑After‑Free、整数溢出、格式字符串）如何被利用来触发 Web 层的常见攻击（SQL 注入、服务器端模板注入、XS‑Leaks）。

**💡 创新点**

创新点在于：①系统地将 WASM 低级漏洞映射到 Web 应用层攻击，填补了现有研究的空白；②提出了一套可复现的 PoC 与实验方法；③给出了针对 WASM 的具体防御建议与最佳实践。

**🔧 技术方法**

技术手段包括：C 代码编写并使用 Emscripten 编译为 WASM；Node.js/Express 搭建后端接口；Pug 模板引擎渲染前端页面；Python 脚本自动化攻击与结果验证；Docker 容器化实现实验环境；利用 PCRE2 正则引擎进行 XS‑Leaks 侧信道实验。

**📊 数据集**

使用的数据集为自定义 PoC，主要包含：1) 预设的 SQL 查询字符串与参数；2) 伪造的模板字符串与 nonce；3) 随机生成的用户机密字符串用于 XS‑Leaks。未使用公开大规模数据集。

**📈 对比分析**

本文未进行性能基准测试，而是通过实验可复现性评估攻击成功率和可行性。PoC 证明在不同漏洞类型下攻击可行且复现成本低，但针对格式字符串和 Use‑After‑Free 的稳定性相对较差。

**⚠️ 局限性**

局限性包括：1）实验环境相对简化，未覆盖所有真实 Web 应用复杂交互；2）格式字符串与 UAF 攻击在黑盒条件下可复现性不足；3）未评估不同浏览器/WASM 运行时对漏洞表现的差异；4）侧信道攻击仅在特定正则引擎与延迟阈值下可行，现实环境中噪声更大。

---

## 351. Fast and Optimal Differentially Private Frequent-Substring Mining

**arXiv ID:** 2603.09166 | [PDF](https://arxiv.org/pdf/2603.09166v1)

**作者:** Peaker Guo `[一作]` (Institute of Science Tokyo), Hao Wu `[通讯]` (University of Waterloo)

**通讯引用:** 106662 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种 ε-差分隐私的频繁子串挖掘算法，在保证近乎最优误差的同时，将空间复杂度降至 O(n + |Σ|) 并将时间复杂度降至 O(n + |Σ|)。

**💡 创新点**

创新点在于：①利用字符对齐的二进制编码将任意字符集映射为二进制，从而降低候选子串生成的指数爆炸；②在自顶向下搜索时构建稀疏后缀树并采用频率引导剪枝，仅探索潜在频繁子串的子树；③结合 Heavy‑Light 分解与 Binary Tree 机制在线估计频率，实现了低噪声、低复杂度的查询。

**🔧 技术方法**

核心技术包括：字符对齐的二进制编码、r‑spaced 稀疏后缀树、Heavy‑Light 分解、Binary Tree 机制、以及 Laplace 机制的误差分析。

**📊 数据集**

本文使用了公开的 Reddit 数据集（约 10⁶ 条用户字符串，总长度 ≥ 3000）作为评估数据集。

**📈 对比分析**

与 Bernardini 等人（PODS'25）提出的 O(n²⁴) 空间/时间算法相比，本文在相同的近乎最优误差下，将复杂度降低到线性级别；实验结果显示，在大规模数据上可实现几乎即时的频繁子串发布。

**⚠️ 局限性**

局限性在于：①仍需对字符集进行二进制编码，若 |Σ| 较大则编码长度 r = ⌈log|Σ|⌉+1 可能导致额外的位开销；②对阈值 τ 的设置需要较为精细的参数调优；③在极端稀疏或高重复度数据时，频率估计误差仍可能影响结果。

---

## 352. LDP: An Identity-Aware Protocol for Multi-Agent LLM Systems

**arXiv ID:** 2603.08852 | [PDF](https://arxiv.org/pdf/2603.08852v1)

**作者:** Sunil Prakash `[一作]` `[通讯]` (Indian School of Business), Sunil Prakash (Indian School of Business)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了LLM委托协议（LDP），并在AutoGen等多代理框架中进行初步实证评估。

**💡 创新点**

创新点在于引入AI本体化身份卡、分级负载模式、受管会话、结构化来源跟踪与信任域等全新的协议原语。

**🔧 技术方法**

技术上采用Rust编写插件适配器，利用Ollama本地模型进行代理推理，使用Google Gemini 2.5 Flash 作为LLM‑as‑Judge 进行质量评估。

**📊 数据集**

数据集为30个按难度分层的任务（分类、推理、编码等）、20个负载测试任务、15个多源合成任务等，全部在本地Ollama模型上运行。

**📈 对比分析**

通过与A2A和随机路由基线对比，使用LLM‑as‑Judge评估质量、延迟和令牌消耗；实验表明身份路由虽未显著提升整体质量，但可显著降低延迟；语义帧负载可减少37%令牌；受管会话可降低39%冗余令牌；模拟安全与回退显示LDP在安全检测与任务完成率上有优势。

**⚠️ 局限性**

局限性包括委托池规模小、仅测试文本和语义帧负载、未对自述身份属性进行外部验证、模拟安全与回退而非真实攻击、单一评审模型、局限于本地模型等。

---

## 353. MM-algorithms for traditional and convex NMF with Tweedie and Negative Binomial cost functions and empirical evaluation

**arXiv ID:** 2603.09601 | [PDF](https://arxiv.org/pdf/2603.09601v1)

**作者:** Elisabeth Sommer James `[一作]` (Aarhus University), Marta Pelizzola `[通讯]` (Aarhus University)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5011213946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出统一框架，给传统与凸非负矩阵分解（NMF）在多种分布假设下（Normal、Poisson、Tweedie、Negative Binomial）提供乘法式更新；对凸NMF引入了全新Poisson、Tweedie、Negative Binomial更新并实现R包；通过残差诊断和BIC评估，展示不同噪声模型对拟合及特征提取的显著影响。

**💡 创新点**

创新点在于：①在凸NMF中首次推导Negative Binomial、Tweedie及Poisson的乘法更新；②统一使用MM算法得到所有模型的闭式更新；③实现了高效Rcpp实现，统一支持所有分布；④结合残差诊断与BIC对模型选择提供实用指南。

**🔧 技术方法**

主要技术包括：非负矩阵分解、凸NMF、Tweedie与β-divergence框架、Negative Binomial似然、Majorize–Minimisation (MM) 乘法更新、分布参数（p、α）通过profile log-likelihood/最大似然估计、Rcpp加速实现。

**📊 数据集**

实验数据：①肝癌突变计数（N=260，M=96，K=5）；②新sgroups文本词频（500文档，6354词，K=7）。

**📈 对比分析**

方法比较通过残差图、BIC值和余弦相似度评估。结果显示：在突变计数中，传统NMF+Negative Binomial/Tweedie获得最低BIC；在文本数据中，凸NMF+Tweedie或Poisson得到更低BIC且参数更少，凸NMF在高维稀疏场景下表现更佳。

**⚠️ 局限性**

局限性包括：1）MM更新收敛速度受p或α估计影响，Tweedie更新计算量大；2）对高维稀疏数据的分布参数估计不稳定，profile likelihood 曲线往往平坦；3）未对凸NMF的可识别性、唯一性进行理论分析；4）仅在两类数据集上验证，泛化性待进一步测试。

---

## 354. Build, Borrow, or Just Fine-Tune? A Political Scientist's Guide to Choosing NLP Models

**arXiv ID:** 2603.09595 | [PDF](https://arxiv.org/pdf/2603.09595v1)

**作者:** Shreyas Meher `[一作]` (Erasmus University Rotterdam), Shreyas Meher `[通讯]` (Erasmus University Rotterdam)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5038469956)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冲突事件分类任务中，本文比较了Fine-tune后的ModernBERT（Confli-mBERT）、领域预训练的ConfliBERT以及ConflLlama的性能，并基于类别稀缺性、误差容忍度和资源可用性提出模型选择决策框架。

**💡 创新点**

创新点在于系统实证揭示域预训练与Fine-tune在不同类别出现频率下的性能差距，并提出三维决策框架指导研究者在实际任务中权衡投入与收益。

**🔧 技术方法**

使用了Transformer架构（ModernBERT）进行Fine-tune，采用二值交叉熵加权损失、类权重调整、sigmoid输出的多标签分类头，以及单GPU训练脚本。

**📊 数据集**

采用Global Terrorism Database（GTD）2017+作为测试集，前期为训练集，使用时间切分，进行9类多标签冲突事件类型分类。

**📈 对比分析**

通过准确率、micro/macro F1、AUC、True Positive计数等指标比较模型；Confli-mBERT达75.46%准确率，ConfliBERT 79.34%，主要差距集中在少数类别；零射击LLM表现最低。

**⚠️ 局限性**

局限性包括仅在单一任务上评估，未对Fine-tune过程做进一步超参数优化；对罕见类别的提升有限；时间切分可能引入分布漂移；未验证跨域或多语言的普适性。

---

## 355. PromptDLA: A Domain-aware Prompt Document Layout Analysis Framework with Descriptive Knowledge as a Cue

**arXiv ID:** 2603.09414 | [PDF](https://arxiv.org/pdf/2603.09414v1)

**作者:** Zirui Zhang `[一作]` (Columbia University), Chengqing Zong `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 PromptDLA 框架，利用领域感知的 Prompt 直接将域先验注入文档布局分析模型，显著提升多领域文档的识别效果。

**💡 创新点**

创新点在于设计了可自定义的域感知 Prompter，将描述性知识（如文档类型、语言、标注风格）编码为 Prompt，直接与视觉特征融合，从而在训练时兼顾不同域的布局差异。

**🔧 技术方法**

主要技术包括 Vision Transformer / Swin / CNN 视觉编码器、CLIP/BLIP2/LLaMA 作为文本编码器、Prompted Transformer Encoder、多尺度 FPN、Cascade R‑CNN/DETR 检测头，以及 Prompt 生成器的三种模式（手工模板、LLM 自动生成、混合）等。

**📊 数据集**

使用了 PubLayNet、DocLayNet、M6Doc、D^4LA 以及自建的多语言 MLDLA 数据集进行实验，覆盖多种文档类型、语言和标注风格。

**📈 对比分析**

与现有 SOTA（如 SwinDocSegmenter、TransDLA、LayoutLMv3、DiT 等）相比，PromptDLA 在 DocLayNet、M6Doc、D^4LA 的 mAP 分别提升约 2.3%、2.0% 和 1.4%，并在不同 backbone、检测头和跨域/多语言场景下均表现出显著的性能提升。

**⚠️ 局限性**

局限性主要体现在对大型 LLM/CLIP 的依赖导致推理时略有加速损失，Prompt 生成的质量与域信息的可获得性密切相关，且在某些细粒度类别（如专利中的公式、图片）仍易受域差异影响。

---

## 356. SoftJAX & SoftTorch: Empowering Automatic Differentiation Libraries with Informative Gradients

**arXiv ID:** 2603.08824 | [PDF](https://arxiv.org/pdf/2603.08824v1)

**作者:** Anselm Paulus `[一作]` (University of Tübingen), Georg Martius `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 1801 | [OpenAlex ID](https://openalex.org/A5001474340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提供了 SoftJax 与 SoftTorch 两套开源库，统一实现了常用硬性操作（如阈值、比较、排序、索引等）的可微软化替代，并支持直通梯度估计。

**💡 创新点**

创新点包括：① 将散布在论文中的软化方法（Heaviside 软化、最优运输、单纯形/可换面投影、神经排序、快速可换面投影和可微排序网络）整合到同一框架；② 引入可调软化参数和多种光滑模式；③ 解决了 STE 在乘法组合时梯度被乘以硬函数导致消失的陷阱。

**🔧 技术方法**

核心技术包括：Heaviside 函数软化、最优运输（OT）正则化、单位单纯形投影、可换面投影、神经排序 (NeuralSort)、FastSoftSort、SmoothSort、可微排序网络（bitonic），以及直通梯度（STE）实现。

**📊 数据集**

主要在实验中使用随机生成的向量进行前向后向传递（无特定公开数据集），并在碰撞检测子例程中演示实际应用。

**📈 对比分析**

与硬性实现比较，排序网络在速度上最优（≈3.8×硬实现），SoftSort、NeuralSort 次之；FastSoftSort 在内存占用上表现最好；OT 和 SmoothSort 计算量大、速度慢。整体可微方法在保持梯度信息的同时实现了可接受的性能。

**⚠️ 局限性**

局限性包括：OT 与 SmoothSort 的计算和内存成本高；软化参数需手动调节；在极端平局或离散性强的情况下梯度仍可能不稳定；部分方法在梯度连续性上仅是有限阶可微。

---

## 357. SEP-NMPC: Safety Enhanced Passivity-Based Nonlinear Model Predictive Control for a UAV Slung Payload System

**arXiv ID:** 2603.08860 | [PDF](https://arxiv.org/pdf/2603.08860v1)

**作者:** Seyedreza Rezaei `[一作]` (York University), Jinjun Shan `[通讯]` (York University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种安全增强的基于被动性的非线性模型预测控制（SEP-NMPC），用于悬挂载荷的四旋翼无人机在拥挤环境中的安全稳定飞行。

**💡 创新点**

创新点在于将严格的能量被动不等式与高阶控制障碍函数（HOCBF）联合嵌入NMPC，既保证闭环能量耗散、渐进稳定，又实现对载荷与无人机双体安全集合的前向不变性，无需增益调度或启发式切换。

**🔧 技术方法**

技术包括能量调节的被动性约束、HOCBF安全约束、QP兼容的非线性MPC求解、四旋翼动力学建模、滑动窗口预测与SQP-RTI求解。

**📊 数据集**

数据集：仿真中使用随机静态与动态障碍物场；实验证明使用Quanser QDrone2平台，携带0.2kg载荷、0.5m绳索，进行20次单障碍/多障碍试验。

**📈 对比分析**

与仅使用一阶CBF、状态约束或仅HOCBF的基线比较，SEP-NMPC在20次试验中100%成功率，最小安全间隙≥0.51m，轨迹RMSE约0.035m，求解时间≈8.7ms，明显降低违规、不可行与超调。

**⚠️ 局限性**

局限性包括对模型精度与障碍运动预测的依赖，未解决在极端不可行情形下的恢复策略，以及在更大规模动态环境中对计算负载的进一步验证。

---

## 358. Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions

**arXiv ID:** 2603.09938 | [PDF](https://arxiv.org/pdf/2603.09938v1)

**作者:** Mingyang Song `[一作]` (Tencent), Mao Zheng `[通讯]` (Tencent)

**通讯引用:** 31915 | [OpenAlex ID](https://openalex.org/A5014662947)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了在大型语言模型时代，通过无训练的参数融合（model merging）将多个已微调模型组合为单一模型的方法，并提出了 FUSE 四维分类框架，归纳了理论基础、融合策略、应用场景与生态系统。

**💡 创新点**

创新点包括：①统一提出 FUSE（Foundations‑Unification‑Scenarios‑Ecosystem）四维分类，将模式连接、权重对齐、任务向量算术、稀疏化增强、MoE 路由等技术整合成一个系统化视角；②在此框架下系统评估并对比多种融合算法；③深入剖析模型融合的理论动因（线性模式连接、损失景观几何、Permutation Symmetry 等）。

**🔧 技术方法**

使用的技术包括：权重平均、梯度/ FIM 加权平均、轨迹平均（SWA/EMA）、几何插值（SLERP）、任务向量算术、TIES‑Merging、DARE、LoRA‑MoE、混合专家（MoE）路由、激活统计对齐、演化搜索等，辅以多种正则化与稀疏化策略。

**📊 数据集**

主要实验基准来自公开的开源 LLM 预训练模型（如 LLaMA、LLaMA‑2、LLaMA‑3、Mistral、Mixtral、Qwen、Qwen‑2、DeepSeek、Gemma 等）及其微调版本；评测使用 Open LLM Leaderboard、多任务/跨语言数据集、对齐安全基准、联邦学习模拟等数据集。

**📈 对比分析**

与单模型、传统集成、完整再训练、蒸馏等基线对比，实验表明融合模型在多任务、跨语言、对齐安全、联邦学习等场景中平均提升 2–5% 以上指标，部分排行榜上刷新了最佳记录；但提升幅度随任务相似度、模型规模与融合策略的不同而变化。

**⚠️ 局限性**

主要局限包括：需要共享预训练初始化且同一架构；对参数对齐、梯度/ Fisher 统计的依赖导致计算/存储成本高；缺乏统一的理论对模式连接强度、稀疏化阈值与路由策略的定量预测；对异构模型、在线动态融合及大规模稀疏化的支持仍有限。

---

## 359. Quantifying the Necessity of Chain of Thought through Opaque Serial Depth

**arXiv ID:** 2603.09786 | [PDF](https://arxiv.org/pdf/2603.09786v1)

**作者:** Jonah Brown-Cohen `[一作]` (Google DeepMind), Rohin Shah `[通讯]` (Google DeepMind)

**通讯引用:** 327 | [OpenAlex ID](https://openalex.org/A5012971694)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

提出了使用电路深度来衡量语言模型不可见序列深度，并给出了手工与自动化的上界计算方法。

**💡 创新点**

将电路深度概念引入LLM可解释性，定义了不可见序列深度，并实现了可自动化的深度计算器。

**🔧 技术方法**

使用电路深度理论、JAX中间表示（jaxpr）遍历、对Transformer、RNN、连续隐式思路、Mixture‑of‑Experts等架构的手工与自动化深度分析。

**📊 数据集**

未使用传统任务数据集，主要针对Gemma 3系列模型的参数和结构进行深度上界计算。

**📈 对比分析**

通过与手工计算比较，JAX计算器在最大序列长度下误差约28%；在Gemma 3系列上验证了对数刻度关系，并发现Mixture‑of‑Experts显著降低深度。

**⚠️ 局限性**

局限包括对“可解释节点”的主观定义、仅捕捉透明度的一面、对电路大小的多项式限制可能不完全贴合实际实现、以及仅支持有限的JAX运算。

---

## 360. ImpedanceDiffusion: Diffusion-Based Global Path Planning for UAV Swarm Navigation with Generative Impedance Control

**arXiv ID:** 2603.09031 | [PDF](https://arxiv.org/pdf/2603.09031v1)

**作者:** Faryal Batool `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 2094 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ImpedanceDiffusion 层次框架，结合基于图像的扩散式全局路径规划、人工势场（APF）跟踪和 VLM‑RAG 语义识别的可变阻抗控制，实现无人机群在室内混合软硬障碍环境中的安全导航；

**💡 创新点**

创新点包括：①直接使用扩散模型从单幅 RGB 图像生成全局轨迹，取代传统搜索规划；②结合 VLM‑RAG 识别障碍类别并动态检索对应阻抗参数，实现语义感知的可变阻抗；③层次化架构将全局规划与局部 APF + 变阻抗控制集成；④对比顶视图与第一人称视角扩散规划的性能差异；

**🔧 技术方法**

技术手段包括：条件 UNet 扩散模型、人工势场（APF）局部跟踪、VLM‑RAG 语义识别（Molmo‑7B‑O + FAISS 向量检索）、虚拟质量弹簧阻尼的可变阻抗控制、模拟到实机零样本迁移；

**📊 数据集**

使用数据集：ProcTHOR 与自建室内仿真环境生成的 10k/13k 条 A* 轨迹用于训练两种扩散规划器；20 个实验配置、100 次实飞行用于评估；200 次实验收集的障碍类阻抗参数数据库；顶视图/第一人称 RGB 图像作为输入；

**📈 对比分析**

通过在八种静态/动态场景下比较两种扩散规划器，评估路径长度、碰撞比、目标误差、累计转弯量；两者均 100% 轨迹生成率；FPV 规划器碰撞比更低（0.246 vs 0.348），顶视图推理更快、路径更平滑；整体成功率 92%，VLM‑RAG 检索准确率 90%；

**⚠️ 局限性**

局限性包括：依赖离线高性能 GPU 计算，未实现分布式/离线推理；阻抗参数仅按障碍类别离散化，缺乏连续学习；硬件通信/丢包导致部分失败；实验规模有限，需验证在更复杂环境中的泛化能力。

---

## 361. Can AI Agents Generate Microservices? How Far are We?

**arXiv ID:** 2603.09004 | [PDF](https://arxiv.org/pdf/2603.09004v1)

**作者:** Bassam Adnan `[一作]` (SERC), Karthik Vaidhyanathan `[通讯]` (SERC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文通过构造增量生成与全新状态两种场景，系统评估了三种主流 AI 生成器（Claude Code、Codex、Code Qwen）在生成微服务时的功能正确性、代码质量与执行效率；

**💡 创新点**

创新点在于：①首次将 AI 代理在微服务级别的端到端生成实验与自动化测试相结合；②提出两种提示策略（最小提示与带实现摘要），并比较其对生成结果的影响；③构建 144 个实验实例并从单元测试与集成测试、复杂度、时空成本等多维度评估性能；

**🔧 技术方法**

技术方法包括：基于 LLM 的代理框架（实现 perception–decision–action 循环）、自动化测试驱动的验证、SonarQube 静态分析、token 与费用计量；

**📊 数据集**

使用的数据集由 5 个微服务系统构成（2 个开源 Java 项目 PiggyMetrics、Train‑Ticket，3 个私有 Python/Java 项目 TeamSync、Project‑Management、MicroBank），每个项目挑选 3 个业务关键服务，共 144 次生成实验；

**📈 对比分析**

对比方法：通过单元测试/集成测试的 pass %、SLOC、cyclomatic、cognitive 复杂度以及生成时间、token 消耗和实际费用；实验显示，增量生成的单元测试通过率为 50‑76%，全新状态的集成测试通过率为 81‑98%；代码复杂度普遍低于人类基线；在效率方面，Claude Code 与 Code Qwen 平均 7‑8 min/服务、Cost 2.98‑13.28 USD；Codex 平均 16.6 min、费用 5.92 USD，且存在长尾延迟；

**⚠️ 局限性**

局限性包括：①实验仅覆盖 REST‑style 微服务，缺乏多样化架构；②对训练数据泄漏/污染敏感，开源项目表现好于私有项目；③提示工程需要针对不同代理调优，未提供通用策略；④仅评估功能与代码质量，未深入探讨架构一致性与可维护性；⑤时间与成本估算基于当前 API 价格，未来模型迭代可能改变。

---

## 362. MAcPNN: Mutual Assisted Learning on Data Streams with Temporal Dependence

**arXiv ID:** 2603.08972 | [PDF](https://arxiv.org/pdf/2603.08972v1)

**作者:** Federico Giannini `[一作]` (Politecnico di Milano), Emanuele Della Valle `[通讯]` (Politecnico di Milano)

**通讯引用:** 4704 | [OpenAlex ID](https://openalex.org/A5015694017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出一种名为 Mutual Assisted Learning（MAcPNN）的边缘设备协同学习框架，能够在数据流场景中自适应概念漂移、时间依赖和灾难性遗忘问题；

**💡 创新点**

创新点在于将 Vygotsky 的社会文化理论引入边缘网络，通过设备在遇到漂移时主动请求并评估他人模型的帮助，从而显著减少通信量并提升适应速度；

**🔧 技术方法**

技术上实现了基于 Continuous Progressive Neural Networks（cPNN）的即时点预测损失函数、模型量化压缩以及自适应列添加机制，形成可扩展的 QcPNN；

**📊 数据集**

实验使用合成的 Sine Random Walk（SRW）数据以及真实的 Weather 与 AirQuality 流数据，构造了多概念、非同步漂移的多设备场景；

**📈 对比分析**

与传统 SML（ARF、HAT）、cLSTM 以及单机 cPNN 比较，MAcPNN 在 Cohen’s Kappa 与 Balanced Accuracy 的 start_avg 与 end_avg 指标上始终表现最优，尤其在漂移初期快速提升；

**⚠️ 局限性**

局限性包括假设漂移检测器 100% 准确、仅处理突发漂移、实验规模仅为三台设备、以及对超参数调优和网络规模扩展尚未深入探讨。

---

## 363. ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited Memory Budget for Long-Context Inference in LLMs

**arXiv ID:** 2603.08727 | [PDF](https://arxiv.org/pdf/2603.08727v1)

**作者:** Jianlong Lei `[一作]` (University of Amsterdam), Shashikant Ilager `[通讯]` (University of Amsterdam)

**通讯引用:** 10764 | [OpenAlex ID](https://openalex.org/A5009158531)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ARKV，一个三状态 KV 缓存管理框架，在 LLM 推理时根据注意力动态分配全精度、低精度或丢弃，以减少内存占用。

**💡 创新点**

结合了层级 OQ 比例估计与重排 heavy‑hitter 分数，动态自适应地对 KV 进行精度分配，实现了轻量化、无训练、无模型改动的混合精度与淘汰策略。

**🔧 技术方法**

利用注意力统计（熵、方差、峰度）估计层敏感度，基于累积注意力计算 heavy‑hitter 分数，并在解码时按阈值分配原始、量化或淘汰状态，支持 fp8 量化与 bfloat16。

**📊 数据集**

在 LLaMA3 与 Qwen3 上，使用 LongBench、GSM8K、MMLU、CommonsenseQA 等任务评估。

**📈 对比分析**

与全精度、仅淘汰、仅量化基线对比，ARKV 在 512–2048 token 预算下保持约 97% LongBench 相关准确率，显著减少 4 倍 KV 内存，并在 TPS 上维持 85–88% 速度，远优于仅量化方案。

**⚠️ 局限性**

对数值推理高度敏感的任务（如 GSM8K）仍需足够预算；量化比例固定在约 14% 以内，未探索更细粒度或多种量化格式；对 MoE 或多模态模型尚未验证。

---

## 364. GeoAlignCLIP: Enhancing Fine-Grained Vision-Language Alignment in Remote Sensing via Multi-Granular Consistency Learning

**arXiv ID:** 2603.09566 | [PDF](https://arxiv.org/pdf/2603.09566v1)

**作者:** Xiao Yang `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**通讯引用:** 71942 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GeoAlignCLIP框架，通过多粒度对比学习与多视图一致性学习实现遥感图像与文本的细粒度对齐。

**💡 创新点**

创新点包括：①Region‑Phrase Alignment和Hard‑Negative Alignment的多粒度对比学习；②视觉内一致性与层次文本一致性的多视图一致性学习；③构建RSFG‑100k细粒度遥感数据集，提供分层语义监督。

**🔧 技术方法**

采用CLIP双编码器（ViT-B/16/ViT‑L/14），RoIAlign提取局部特征，联合使用全局与局部对比损失、硬负样本对齐、视觉/文本一致性约束。

**📊 数据集**

使用RSFG‑100k作为主训练集，评估基准包括RRSIS‑HR、CHOICE、NWPU‑VHR‑10、RRSIS‑D、DIOR、DOTAv1.0、RSICD、RSITMD、UCM‑Caption等公开遥感数据集。

**📈 对比分析**

与CLIP、LongCLIP、FG‑CLIP、RemoteCLIP、SkyCLIP、GeoRSCLIP、LRSCLIP等模型对比，GeoAlignCLIP在细粒度识别、区域分类、开放词检测、图文检索等任务均取得SOTA提升（Acc@1提升至33.45%，mAP_n提升至17.1%等）。

**⚠️ 局限性**

局限性包括：ROI提取导致FLOPs增加，计算开销略升；对极端尺度变化、遮挡等仍有挑战；模型依赖大规模多粒度标注，构建和维护成本高；实时部署时需进一步优化推理效率。

---

## 365. Agentic AI as a Network Control-Plane Intelligence Layer for Federated Learning over 6G

**arXiv ID:** 2603.09141 | [PDF](https://arxiv.org/pdf/2603.09141v1)

**作者:** Loc X. Nguyen `[一作]` (Kyung Hee University), Choong Seon Hong `[通讯]` (Kyung Hee University)

**通讯引用:** 22766 | [OpenAlex ID](https://openalex.org/A5034052371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Agentic AI控制平面作为6G网络中联邦学习的自适应决策层，实现自动化的客户端选择、资源调度、训练策略制定和代码生成。

**💡 创新点**

将Agentic AI多代理协作、目标分解、闭环评估与工具使用融入联邦学习，构建完整的自监督、持续改进的学习与网络管理闭环。

**🔧 技术方法**

利用LLM驱动的检索、规划、评估、编码代理；结合代码生成工具（GitHub Copilot）、优化器（资源分配、压缩）和实时网络监测，构成自适应决策框架。

**📊 数据集**

使用MNIST分类数据集，配合仿真中的无线信道条件和Dirichlet α=0.1的客户端数据分布。

**📈 对比分析**

与随机、延迟、最大数据、类别多样性四种客户端选择基准对比；类别多样性方案在10轮内实现最高测试准确率，延迟方案获得最低延迟与最高SNR。

**⚠️ 局限性**

存在学习-控制耦合的稳定性风险、代理间博弈冲突、规模可扩展性挑战、误用与安全风险以及价值漂移与物理系统集成的技术难题。

---

## 366. Can You Hear, Localize, and Segment Continually? An Exemplar-Free Continual Learning Benchmark for Audio-Visual Segmentation

**arXiv ID:** 2603.08967 | [PDF](https://arxiv.org/pdf/2603.08967v1)

**作者:** Siddeshwar Raghavan `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**通讯引用:** 3972 | [OpenAlex ID](https://openalex.org/A5001380619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了无样本连续学习基准，并提出ATLAS模型用于音视分割。

**💡 创新点**

创新点在于将音频引导的预融合调制与低秩锚定相结合，实现无样本连续学习的高性能。

**🔧 技术方法**

采用LoRA参数高效适配、音频引导预融合、跨模态注意力、低秩锚定等技术。

**📊 数据集**

使用SS-AVS和MS-AVS两个数据集，分别在TIL、CIL、DIL和TF-CL四种协议上进行评估。

**📈 对比分析**

与多种基准（Finetune、EWC、SI、MAS、DGR、PANDA等）对比，ATLAS在mAP上最高，前向迁移好、遗忘适中。

**⚠️ 局限性**

局限在于依赖预训练模型、仅验证两套数据集、对高复杂场景和大规模任务仍有挑战。

---

## 367. MissBench: Benchmarking Multimodal Affective Analysis under Imbalanced Missing Modalities

**arXiv ID:** 2603.09874 | [PDF](https://arxiv.org/pdf/2603.09874v1)

**作者:** Tien Anh Pham `[一作]` (VNU University of Engineering and Technology), Cam-Van Thi Nguyen `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MissBench，统一评估多模态情感分析在共享与失衡缺失率下的性能并引入模态公平指数与学习指数诊断。

**💡 创新点**

创新点在于同时考察共享与失衡缺失模式，并提供两项可量化模态公平与优化平衡的指标。

**🔧 技术方法**

采用缺失率采样协议、模型插件接口以及梯度监测等技术实现完整评估框架。

**📊 数据集**

使用四个公开情感数据集：IEMOCAP、CMU-MOSI、CMU-MOSEI 与 CH-SIMS。

**📈 对比分析**

与多种缺失处理方法（RedCore、GCNet、MCE 等）对比，发现任务准确率相近但在失衡下模态不平衡和梯度偏差显著。

**⚠️ 局限性**

局限在于只关注情感任务，缺少跨任务验证，且指标仍未完全捕捉所有模型潜在失衡。

---

## 368. The Coupling Within: Flow Matching via Distilled Normalizing Flows

**arXiv ID:** 2603.09014 | [PDF](https://arxiv.org/pdf/2603.09014v1)

**作者:** David Berthelot `[一作]` (Apple), Shuangfei Zhai `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过使用预训练的正则化自回归流（TarFlow）生成的噪声/数据耦合，对 Flow Matching 进行蒸馏，训练出更快、更好表现的学生模型（Normalized Flow Matching, NFM）。

**💡 创新点**

创新点在于将 Normalizing Flow 的双射映射作为自适应噪声耦合，并通过蒸馏将其注入 Flow Matching 训练；该方法既降低采样步数又提高生成质量，且实现了显著的采样速度提升。

**🔧 技术方法**

使用的技术包括 Flow Matching、Normalizing Flow（TarFlow）与自回归 Transformer、蒸馏、ODE 采样（Euler/Heun）、CFG 引导、Latent 空间 VAE 编码、以及对比的 Optimal Transport (SD-FM)。

**📊 数据集**

在 ImageNet 64×64 和 256×256（后者在 VAE 隐空间）上进行实验。

**📈 对比分析**

与基准 FM、SD-FM、TarFlow 老师模型比较，采用 FID 与 NFE（采样步数）以及延迟（Latency）进行评估。NFM 在 31 NFE 下获得 1.78 FID（低于老师 1.98），在 15/7 NFE 下仍优于老师，速度提升 32–145 倍，且在更少采样步下实现更优 FID。

**⚠️ 局限性**

局限性包括：需要先训练成本高的 NF 教师；对 NF 生成的 z‑space 结构尚未完全理解；目前仅在图像任务验证，文本/多模态等扩展待验证；无引导时性能相对下降；蒸馏效果受教师架构与 NLL 影响。

---

## 369. Quantifying Memorization and Privacy Risks in Genomic Language Models

**arXiv ID:** 2603.08913 | [PDF](https://arxiv.org/pdf/2603.08913v1)

**作者:** Alexander Nemecek `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**通讯引用:** 2633 | [OpenAlex ID](https://openalex.org/A5028326739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对基因组语言模型的记忆化风险进行评估并提出统一的多向量隐私评估框架。

**💡 创新点**

提出将困惑度检测、可怕序列提取和成员推断三种攻击向量整合为最大漏洞得分的全新评估方法，并验证记忆化与重复率、模型规模的缩放关系。

**🔧 技术方法**

采用GLMs的微调、可怕序列植入、困惑度计算、Beam搜索提取、似然比成员推断等技术。

**📊 数据集**

在合成零阶序列、E.coli、酵母、GUE promoter等四个不同复杂度的基因组数据集上进行实验。

**📈 对比分析**

与四种不同架构（SimpleDNALM、DNABERT-2、HyenaDNA、Evo）比较，发现Evo的最大漏洞得分达1.0，其余模型在0.48-0.55之间，表明架构决定了记忆化水平。

**⚠️ 局限性**

局限包括只使用随机无生物结构的可怕序列、仅评估LoRA微调的大模型、数据集规模固定为1000条、未考虑完整参数微调和更大训练集。

---

## 370. Adaptive SINDy: Residual Force System Identification Based UAV Disturbance Rejection

**arXiv ID:** 2603.08863 | [PDF](https://arxiv.org/pdf/2603.08863v1)

**作者:** Fawad Mehboob `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 2094 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于SINDy的系统识别与递归最小二乘（RLS）自适应控制相结合的UAV风扰动拒绝方法。

**💡 创新点**

创新点在于将稀疏非线性动力学识别（SINDy）与在线自适应控制相融合，实现对未知残余风力的可解释建模与实时补偿；并设计了基于姿态角和推力的物理知识库函数。

**🔧 技术方法**

使用技术包括SINDy（SR3稀疏优化）、RLS自适应控制、Gazebo仿真、ArduPilot SITL、Crazyflie硬件飞行、以及基于OU过程的风模型。

**📊 数据集**

数据集为Gazebo仿真中收集的状态与控制输入数据，以及真实Crazyflie飞行中记录的状态序列，包含多条圆形、无限线、螺旋轨迹在风扰动下的跟踪数据。

**📈 对比分析**

与预训练的DAIML自适应控制、经典PID、以及Crazyflie上的INDI控制器进行对比；在仿真和实飞中，Adaptive SINDy在所有轨迹上均优于PID，RMSE平均提升至约0.1–0.3 m，误差分布更集中，且在实飞中成功完成全部15次试验而PID多次失控。

**⚠️ 局限性**

局限性包括对风扰动模型的依赖（需要事先的风实验或仿真数据）、对大规模或高速飞行的验证不足，以及系统识别过程对计算资源的需求，在极端风速或多机协同场景下的可扩展性待进一步研究。

---

## 371. Electoral Systems Simulator: An Open Framework for Comparing Electoral Mechanisms Across Voter Distribution Scenarios

**arXiv ID:** 2603.08752 | [PDF](https://arxiv.org/pdf/2603.08752v1)

**作者:** Sumit Mukherjee `[一作]` `[通讯]` (Oracle), Sumit Mukherjee (Oracle)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并使用一个可扩展的Python框架，对多种选举制度在多种投票者分布下的空间结果进行仿真与比较。

**💡 创新点**

创新性地提供统一的评价基准（投票者几何中位点）和模块化体系，允许快速加入新制度、情境和指标，并通过Fractional Ballot给出近似上界。

**🔧 技术方法**

使用Python实现的模拟框架，利用二维理想空间、欧氏距离、Weiszfeld算法计算几何中位点，并通过YAML配置、Monte Carlo 200次实验评估指标。

**📊 数据集**

采用人工生成的基于高斯混合模型的八个情境数据集（共约5-6千名投票者和预设候选人位置）。

**📈 对比分析**

以选举结果与几何中位点的欧氏距离为主要指标，对九种标准制度和Fractional Ballot进行横向比较；结果显示Fractional Ballot在大多数情境中最优，普选在极化情境表现最差，比例制在碎片化情境中出现中位议员偏差。

**⚠️ 局限性**

主要局限包括假设投票者始终真诚投票、仅考虑二维理想空间、缺乏战略投票与党派协商动态、以及Fractional Ballot需要先行测量候选人和选民位置。

---

## 372. Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning

**arXiv ID:** 2603.08999 | [PDF](https://arxiv.org/pdf/2603.08999v1)

**作者:** Juming Xiong `[一作]` (Vanderbilt University), Zhijun Yin `[通讯]` (Vanderbilt University Medical Center)

**通讯引用:** 2565 | [OpenAlex ID](https://openalex.org/A5079247989)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种置信度感知推理框架，利用单条链式思考路径中的句级数值与语言特征来判断是否需要额外的多路径推理，避免无谓采样；

**💡 创新点**

创新点在于通过对单条推理轨迹的句级统计信号（概率、熵、长度、变化趋势等）与轻量语言特征（词频、标点、停用词等）进行融合，构建轻量级决策网络，实现零成本的置信度评估与自适应推理；

**🔧 技术方法**

采用句级概率/熵归一化、EMA 等数值特征，结合词计数、停用词比例、标点密度等语言特征，输入至注意力加GRU的决策网络进行置信度预测；

**📊 数据集**

模型在 MedQA 上训练，零样本迁移至 MathQA、MedMCQA 与 MMLU 四个多选问答数据集；

**📈 对比分析**

与贪婪、Self-Consistency、Confidence‑Enhanced Reasoning、Dynamic Voting 等基线比较，保持与多路径基线相当的准确率，同时在所有数据集上显著降低 69%–79% 的令牌消耗；

**⚠️ 局限性**

局限性包括仅针对结构化的多选问答任务，无法在线早停，对内部概率与熵信息依赖较强，且在开放式生成、长篇推理或对话场景中的效果尚未验证。

---

## 373. Data-Rate-Aware High-Speed CNN Inference on FPGAs

**arXiv ID:** 2603.08726 | [PDF](https://arxiv.org/pdf/2603.08726v1)

**作者:** Tobias Habermann `[一作]` (Fulda University of Applied Sciences), Martin Kumm `[通讯]` (Fulda University of Applied Sciences)

**通讯引用:** 1112 | [OpenAlex ID](https://openalex.org/A5016467458)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种数据率感知的多像素CNN推理加速器，在FPGA上通过设计空间探索优化硬件利用率，实现高吞吐量推理。

**💡 创新点**

创新点在于简化连续流架构的参数、引入多像素处理并提出新的层实现参数选择方法，显著降低算术资源使用且支持多数据速率的灵活配置。

**🔧 技术方法**

采用数据率匹配的连续流架构、KPU/FCU单元、压缩树、资源共享与重配置、设计空间探索及基于分数输入速率的Diophantine约束求解技术。

**📊 数据集**

使用在ImageNet上训练的MobileNetV1和MobileNetV2模型进行实验。

**📈 对比分析**

与现有SOTA FPGA实现（如FINN、LUTMUL等）在同一型号XCVU37P上对比，提出的加速器在相同或更低资源占用下实现FPS提升至约16k，超过SOTA三倍；低速率下可显著降低DSP与LUT使用。

**⚠️ 局限性**

局限性在于BRAM使用率始终偏高，权重存储仍依赖BRAM，未随数据速率下降而显著下降，需要通过DRAM/HBM卸载等方式进一步改进。

---

## 374. From Flow to One Step: Real-Time Multi-Modal Trajectory Policies via Implicit Maximum Likelihood Estimation-based Distribution Distillation

**arXiv ID:** 2603.09415 | [PDF](https://arxiv.org/pdf/2603.09415v1)

**作者:** Ju Dong `[一作]` (University of Hamburg), Jianwei Zhang `[通讯]` (University of Hamburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文将多步Conditional Flow Matching（CFM）生成式策略通过IMLE（隐式最大似然估计）加上双向Chamfer距离的集合级损失，压缩成单步快速策略，实现实时多模态操控；

**💡 创新点**

创新点包括①用集合级IMLE与双向Chamfer目标保留多模态分布，避免单步逼近导致的模式坍塌；②构建统一的几何感知多模态感知编码器，融合RGB、深度、点云与本体感知；③实现单步推理速率>125 Hz，显著提升实时闭环控制能力；

**🔧 技术方法**

采用的核心技术有Conditional Flow Matching教师网络、IMLE无似然估计、集合级双向Chamfer损失、U‑Net结构的单步学生网络，以及多模态视觉-点云-本体的融合编码器；

**📊 数据集**

实验使用RLBench模拟任务的数据集以及在Frankia Emika Panda机器人上收集的300条远程演示轨迹，涵盖动态、长周期和多物体交互场景；

**📈 对比分析**

与多步扩散/流匹配、Naïve 1‑step、Consistency Policy及IMLE基线对比，单步学生在RLBench任务上实现68.6%成功率，123.5 Hz；在真实世界中70%成功率，125 Hz，较多步教师（2.9 Hz）提升43×，且避免了模式坍塌导致的失败；

**⚠️ 局限性**

局限性在于与多步教师仍存在少量性能差距，依赖大量离线演示数据，且在极少见的操作模式下可能欠缺覆盖，未在更广泛任务上验证。

---

## 375. MO-Playground: Massively Parallelized Multi-Objective Reinforcement Learning for Robotics

**arXiv ID:** 2603.09237 | [PDF](https://arxiv.org/pdf/2603.09237v1)

**作者:** Neil Janwani `[一作]` (Georgia Institute of Technology), Maegan Tucker `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 276 | [OpenAlex ID](https://openalex.org/A5054944952)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了 MO-Playground，一套 GPU 原生的多目标强化学习框架，包含可并行训练的 MORLaX 算法和一组 GPU 加速的多目标控制环境，能够在几分钟内逼近高质量的 Pareto 集合并在真实机器人（BRUCE）上实现多目标步态控制。

**💡 创新点**

创新点在于：①利用超网络与 JAX 的向量化计算实现参数高效、连续的 Pareto 集合逼近；②通过 GPU 并行化采样、回放与更新，显著提升训练速度（最高可达 270 倍）；③提供易用的多目标环境集合和开源工具箱，填补了现有单目标 GPU 强化学习与多目标方法的鸿沟。

**🔧 技术方法**

核心技术包括：JAX + MuJoCo Playground 的 GPU 并行物理仿真；超网络（actor 与 critic 超网络）映射 trade‑off 向量到策略参数；Dirichlet 采样生成多样化 trade‑off ；线性 scalarization 与多目标 PPO；GPU 并行采样-回放-更新流程。

**📊 数据集**

使用的数据集主要是 DeepMind Control Suite 的经典任务（Cheetah, Walker, Ant, Humanoid, Hopper）以及自定义的 BRUCE 人形机器人环境，并在这些环境中设定多达 6 个目标（速度跟踪、能耗、平滑度、臂摆动等）。

**📈 对比分析**

与基准算法 HYPER‑MORL 进行对比。实验显示 MO-Playground 在所有五个基准环境中均实现更高的超体积，并在几分钟内完成训练，速度提升 34–271 倍；在 BRUCE 环境中实现 6 维 Pareto 前沿，仅耗时约 2 小时 11 分钟，比原报告的 5 天训练快数十倍。

**⚠️ 局限性**

局限性包括：①需预先确定所有目标，难以处理人类难以量化的目标；②采用线性 scalarization 只能逼近凸 Pareto 前沿，无法覆盖凹形区间；③对超参数敏感，虽然速度快但仍需调参；④当前实现仅支持线性权重，未探索更复杂的权重分布或自适应权重学习。

---

## 376. SCDP: Learning Humanoid Locomotion from Partial Observations via Mixed-Observation Distillation

**arXiv ID:** 2603.09574 | [PDF](https://arxiv.org/pdf/2603.09574v1)

**作者:** Milo Carroll `[一作]` (University College London), Zhibin Li `[通讯]` (University College London)

**通讯引用:** 7431 | [OpenAlex ID](https://openalex.org/A5100351555)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过混合观测训练的扩散模型实现了仅靠本体传感器（无全身状态估计）控制类人机器人行走与运动跟踪。

**💡 创新点**

创新点包括：混合观测训练将感知与监督分离；受限去噪剔除速度反馈以强制模型从历史信息推断速度；上下文分布对齐与上下文感知注意力掩码共同消除训练‑部署差异；实现了在无速度反馈下的完整身体控制。

**🔧 技术方法**

使用技术：扩散概率模型（DDPM）+ Transformer 编码器；离线强化学习专家（MMP）数据收集；受限去噪、上下文分布对齐、双向注意力掩码；ONNX 推理在 50 Hz 运行；强化学习、仿真域随机化。

**📊 数据集**

使用数据集：AMASS（走姿）、MMP 训练数据、由专家产生的 5200 条轨迹（约 750k 时步）用于离线训练；在 Unitree G1 上进行实机部署测试。

**📈 对比分析**

与基线（BC、BeyondMimic 等）在扰动恢复、摇杆控制、目标导航、运动跟踪四个任务上对比，SCDP 在无速度反馈的情况下成功率分别为 99–100%（扰动恢复、摇杆控制）和 93%（运动跟踪），与特权状态基线相当甚至更优。

**⚠️ 局限性**

局限性：仍需大量离线多样化数据收集；在长时间轨迹中存在全局位置漂移误差；依赖远程 GPU 推理，尚未实现完全本地在线部署；未针对非平坦地形或复杂接触场景进行验证。

---

## 377. SPREAD: Subspace Representation Distillation for Lifelong Imitation Learning

**arXiv ID:** 2603.08763 | [PDF](https://arxiv.org/pdf/2603.08763v1)

**作者:** Kaushik Roy `[一作]`, Peyman Moghadam `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种专门针对终身模仿学习的子空间表示蒸馏框架，结合置信度引导的策略蒸馏来保留先前知识并促进新技能学习。

**💡 创新点**

创新点在于：①使用SVD得到低秩子空间，对连续策略的特征子空间进行几何对齐，从而保留任务内在低维流形；②在策略蒸馏中只对高置信度的动作样本使用KL损失，降低噪声和方差。

**🔧 技术方法**

技术手段包括：子空间表示蒸馏（SPREAD）、置信度引导的KL策略蒸馏、跨模态特征提取（视觉、语言、关节、夹爪）、经验回放、以及对多任务评估指标（FWT、NBT、AUC）的计算。

**📊 数据集**

使用LIBERO连续模仿学习基准，包含三套任务（LIBERO-OBJECT、LIBERO-GOAL、LIBERO-SPATIAL），每套10个连续机器人操作任务。

**📈 对比分析**

与SEQUENTIAL、EWC、ER、BUDS、LOTUS、M2Distill等现有方法对比，实验显示在FWT、NBT、AUC上均优于或接近SOTA；在LIBERO-OBJECT上最高AUC 73%，NBT最低 8%，在LIBERO-GOAL和LIBERO-SPATIAL同样取得最优或显著提升。

**⚠️ 局限性**

局限性包括：对子空间秩的选择需经验调参，计算上需要SVD开销；置信度阈值对策略蒸馏效果影响较大；实验仅在模拟机器人抓取任务上验证，缺乏真实世界长序列任务的验证。

---

## 378. Curveball Steering: The Right Direction To Steer Isn't Always Linear

**arXiv ID:** 2603.09313 | [PDF](https://arxiv.org/pdf/2603.09313v1)

**作者:** Shivam Raval `[一作]` (Harvard University), Amirali Abdullah `[通讯]` (Thoughtworks)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Curveball steering，一种基于多项式核PCA的非线性激活空间调控方法；

**💡 创新点**

通过学习激活空间的黎曼度量发现其非欧氏几何结构，从而设计非线性曲线路径实现更精细的行为调节；

**🔧 技术方法**

主要技术包括拉普拉斯变换的拉普拉斯拉普拉斯的核PCA、预像重建、以及基于VAE学习的拉普拉斯度量；

**📊 数据集**

使用扩增后的LLM对比数据集（约8k–10k条对话），涵盖多种行为与人格特征；

**📈 对比分析**

与传统线性PCA调节对比，Curveball在高曲率情境下表现更佳，行为转移概率提升约10–50%，在部分特征上相对线性方法提升超过80%；

**⚠️ 局限性**

局限性在于核PCA训练与逆映射计算开销大，且需要足够多样的激活样本，且对极大模型规模的适用性尚未验证。

---

## 379. Platooning as a Service (PlaaS): A Sustainable Transportation Framework for Connected and Autonomous Vehicles

**arXiv ID:** 2603.09256 | [PDF](https://arxiv.org/pdf/2603.09256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 380. Impact of Different Failures on a Robot's Perceived Reliability

**arXiv ID:** 2603.08821 | [PDF](https://arxiv.org/pdf/2603.08821v1)

**作者:** Andrew Violette `[一作]` (Cornell University), Hadas Kress-Gazit `[通讯]` (Cornell University)

**通讯引用:** 5040 | [OpenAlex ID](https://openalex.org/A5020074157)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究通过在线视频实验比较了不同类型机器人失误（slip、lapse、mistake）对人类感知可靠性（PR）的影响，并考察成功执行是否能恢复 PR。

**💡 创新点**

创新点在于：①引入客观的下注+置信度（signed confidence）指标来衡量 PR；②系统评估多种失败类型及其对 PR 的差异；③发现成功执行后即可恢复 PR，无需显式修复行为。

**🔧 技术方法**

使用了预注册的在线视频实验、$2 下注与置信度测量、signed confidence 计算、一次方差分析（ANOVA）、Welch t 检验与 Holm‑Bonferroni 校正，以及 affinity diagramming 进行定性分析。

**📊 数据集**

实验使用 326 名 Prolific 参与者，采用自行制作的机器人臂 pick‑and‑place 任务视频；未使用公开数据集。

**📈 对比分析**

对比方法：计算不同失败条件下的 ΔPR（后前下注差值）并与 Failure、Failure+Success、Success 条件进行比较；结果显示 slip 与 lapse 对 PR 影响最大，mistake 最小；成功后 PR 恢复至甚至高于基线，并与直接成功无显著差异。

**⚠️ 局限性**

局限性：①实验仅在线视频，缺乏现场交互；②下注基于实验者定义的成功，可能影响真实信任评估；③未探索多任务或更复杂情境；④仅考察单一 pick‑and‑place 任务，结果可能不具普适性。

---

## 381. BEACON: Language-Conditioned Navigation Affordance Prediction under Occlusion

**arXiv ID:** 2603.09961 | [PDF](https://arxiv.org/pdf/2603.09961v1)

**作者:** Xinyu Gao `[一作]` (Delft University of Technology), Javier Alonso-Mora `[通讯]` (Delft University of Technology)

**通讯引用:** 8480 | [OpenAlex ID](https://openalex.org/A5013297671)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为BEACON的框架，用于语言条件下的局部导航目标预测，能够在遮挡环境中通过结合视觉语言模型与鸟瞰视图（BEV）热图实现目标定位；

**💡 创新点**

创新点包括：①引入Ego‑Aligned 3D位置编码和自监督指令调优，使VLM更好理解主体框架下的空间语言；②在BEV空间构建可通行热图，并采用地理距离目标区域监督来抑制非可通行预测；③通过两阶段训练与多模态融合提升对遮挡目标的推断能力；

**🔧 技术方法**

使用InternVL2‑2B视觉语言模型（冻结视觉编码器），LoRA微调，3D位置编码，BEV编码器（DINOv2特征+SECOND卷积）、BEVFusion、门控机制G以及地理距离监督；

**📊 数据集**

基于Habitat仿真器构造的局部导航数据集，来源于Landmark‑RxR并加入移动行人（Social‑MP3D），形成遮挡子集；

**📈 对比分析**

与ChatGPT‑4o、RoboPoint、RoboRefer等图像空间基线以及直接训练的VLM+点头进行对比；在完整验证集和遮挡子集上，BEACON GeoAcc 提升 22.74pp，SIR 降至 2.6%，EucAcc 亦有提升，表现显著优于基线；

**⚠️ 局限性**

仅在仿真环境验证，未对真实传感器数据进行评估；未处理动态人类或社交导航场景；对极端遮挡的鲁棒性仍有限；模型对RGB‑D输入依赖较大。

---

## 382. Scale-Plan: Scalable Language-Enabled Task Planning for Heterogeneous Multi-Robot Teams

**arXiv ID:** 2603.08814 | [PDF](https://arxiv.org/pdf/2603.08814v1)

**作者:** Piyush Gupta `[一作]` (Honda Research Institute), David Isele `[通讯]` (Honda Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Scale-Plan框架，利用基于PDDL的动作图和LLM进行任务相关信息过滤与规划，直接生成多机器人可执行长周期任务计划，避免完整环境输入和中间PDDL问题文件的生成。

**💡 创新点**

创新点在于：① 用动作图（严格/松弛边）从域层面抽取最小任务相关动作与对象，显著减少搜索空间；② 采用浅层LLM推理结合图搜索，而非直接生成完整PDDL；③ 通过无中间PDDL文件的结构化LLM规划实现高可靠性；④ 构建MAT2-THOR清洗版基准，提升实验可复现性。

**🔧 技术方法**

技术包括：PDDL领域解析与动作图构造、图搜索过滤、GPT-5.2的浅层推理、任务分解、机器人分配与计划整合、Plan-to-Code转换以及AI2-THOR仿真环境。

**📊 数据集**

使用MAT2-THOR（49个任务，分简单、复杂、含糊三类）作为评测数据集，该基准基于AI2-THOR，经过清洗和标准化。

**📈 对比分析**

与四个基线（LLM Planner、LLM+P、LaMMA-P PDDL-only、LaMMA-P LLM-corrected）在TCR、GCR、ER三项指标对比。Scale-Plan在所有类别和总体上均领先，整体TCR 78%/GCR 85%/ER 94%，比最强基线提升约16%/16%/9%。消融实验验证过滤和结构化规划的必要性；规划时间略高。

**⚠️ 局限性**

局限性：缺乏精准环境对齐导致LLM幻觉；对含糊任务的过滤不够鲁棒；未充分建模操作约束与affordance，导致执行失败；未来计划引入知识图、语义验证与回滚机制以提升可靠性。

---

## 383. Accelerating High-Order Finite Element Simulations at Extreme Scale with FP64 Tensor Cores

**arXiv ID:** 2603.09038 | [PDF](https://arxiv.org/pdf/2603.09038v1)

**作者:** Jiqun Tu `[一作]` (NVIDIA Corporation), Omar Ghattas `[通讯]` (University of Texas at Austin)

**通讯引用:** 8094 | [OpenAlex ID](https://openalex.org/A5049331711)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在MFEM高阶有限元库中直接使用FP64张量核心并进行核融合，提升了关键矩阵乘法的计算吞吐率

**💡 创新点**

首次将FP64张量核心应用于大规模科学计算的有限元应用，并结合张量核心与核融合实现最高2倍加速和83%能效提升

**🔧 技术方法**

使用CUDA PTX直接调用DMMA指令、共享内存银行冲突规避、张量重排、内核融合以及GPU全系统扩展

**📊 数据集**

以海啸早期预警数字孪生为例，使用高阶有限元离散的声‑重力波方程，测试规模从5.4亿DOF到9.28万亿DOF

**📈 对比分析**

相较于传统CUDA核心实现，DMMA+融合核在单GPU上实现了35%-59%速度提升，双GPU上实现约2倍速度和约80%能效提升，并在Alps系统上达到90%强/弱扩展效率

**⚠️ 局限性**

受限于DMMA指令尺寸与矩阵尺寸不匹配导致计算浪费、共享内存带宽仍为瓶颈、部分核未充分利用FP64张量核心、以及需要进一步针对不同矩阵形状优化映射

---

## 384. Diffusion-Based Authentication of Copy Detection Patterns: A Multimodal Framework with Printer Signature Conditioning

**arXiv ID:** 2603.08998 | [PDF](https://arxiv.org/pdf/2603.08998v1)

**作者:** Bolutife Atoki `[一作]` (Université Lumière Lyon 2), Carlos Crispim-Junior `[通讯]` (Université Lumière Lyon 2)

**通讯引用:** 541 | [OpenAlex ID](https://openalex.org/A5081523913)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过将二进制模板、打印CDP与打印机身份信息整合进扩散模型，提出了统一的多模态身份认证框架。

**💡 创新点**

将ControlNet从生成任务改为分类任务，实现基于噪声预测误差的打印机识别，并使用自然语言描述打印机身份以增强语义引导。

**🔧 技术方法**

采用扩散模型（ControlNet + Stable Diffusion VAE）、CLIP文本编码、跨类噪声预测分类器，以及多步去噪误差最小化策略。

**📊 数据集**

在Indigo 1×1 Base数据集上进行实验，该数据集包含两台HP Indigo 5500/7600打印机生成的真实和伪造CDP，共六个类别。

**📈 对比分析**

与传统NCC/SSIM和Pix2Pix等深度学习方法对比，均衡误差率P_err降至0.023，误拒率0.005，伪造拒绝率0.000，性能显著优于基线。

**⚠️ 局限性**

仅在单一扫描仪配置下验证，缺乏更多打印机与纸张多样性；训练仍需已知伪造样本，未探讨无伪造样本训练策略。

---

## 385. d-DNNF Modulo Theories: A General Framework for Polytime SMT Queries

**arXiv ID:** 2603.09975 | [PDF](https://arxiv.org/pdf/2603.09975v1)

**作者:** Gabriele Masina `[一作]` (University of Trento), Roberto Sebastiani `[通讯]` (University of Trento)

**通讯引用:** 6560 | [OpenAlex ID](https://openalex.org/A5088301538)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种通用的框架，将SMT公式编译为d-DNNF形式的SMT等价公式，使得在d-DNNF上进行的多种查询（如一致性、蕴含、模型计数、枚举等）可以在多项式时间内完成。

**💡 创新点**

创新点在于：1) 通过预先生成理论层的lemmas并与原公式组合，消除所有理论不一致的赋值，从而将SMT层的查询完全归约到布尔层的查询；2) 该方法对任意理论或理论组合均适用；3) 可以在任何d-DNNF编译器和理论lemmas枚举器之上实现；4) 证明了在此框架下，所有常见的SMT查询都可在多项式时间内完成。

**🔧 技术方法**

技术主要包括：理论lemmas枚举（利用作者提出的高效枚举技术）、布尔抽象与存在量化、d-DNNF（以及其子类如SDD、OBDD）编译、以及对编译后公式的布尔层查询。

**📊 数据集**

使用了<cit.>中的450个合成非CNF实例作为基准；对于CE查询，每个实例生成10个随机子句；对于CT查询，使用CE不成立时的反例cube。

**📈 对比分析**

与传统SMT求解器（如MathSAT）在非增量与增量模式下进行比较。实验结果显示：1) 对CE查询，d-DNNF编译后查询速度明显优于非增量SMT，且往往优于增量SMT；2) 对计数查询（#SMT），编译后方法在大多数情况下比AllSMT快数百倍，并能在分秒级完成，而AllSMT常超时。

**⚠️ 局限性**

主要限制是编译时必须预先知道所有参与查询的原子集合，无法支持动态扩展的查询；另外，理论lemmas枚举在某些理论下仍然昂贵，影响编译时间。

---

## 386. Chow-Liu Ordering for Long-Context Reasoning in Chain-of-Agents

**arXiv ID:** 2603.09835 | [PDF](https://arxiv.org/pdf/2603.09835v1)

**作者:** Naman Gupta `[一作]` (Microsoft), Vageesh D. C `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Chain-of-Agents长文本推理框架中，提出基于Chow–Liu树的分块顺序策略，以减小记忆瓶颈下的信息损失。

**💡 创新点**

创新点在于将检索块的互相依赖建模为树结构，并使用广度优先遍历生成顺序，从而系统地优化块的处理顺序。

**🔧 技术方法**

技术包括LLM驱动的CoA、多代理序列推理、嵌入相似度计算、最大权重生成树（Chow–Liu）、BFS遍历、RAG评估框架。

**📊 数据集**

数据集：LongQA（含Helmet、∞Bench、NarrativeQA）等长文本问答数据，覆盖超过256K token的长文本。

**📈 对比分析**

与默认文档顺序和基于语义分数的Dense顺序对比，CL‑order在GPT‑4.1、GPT‑4.1‑mini和Qwen‑3‑14B三种模型上均显著提升Ragas答案相关性与EM精度，最高提升约10.7% EM、5.9% Ragas。

**⚠️ 局限性**

限制在于依赖静态嵌入估计的互信息近似，且对LLM压缩策略的具体实现假设，可能在不同模型或更大规模任务下表现不一致。

---

## 387. On the Online Weighted Non-Crossing Matching Problem

**arXiv ID:** 2603.09262 | [PDF](https://arxiv.org/pdf/2603.09262v1)

**作者:** Joan Boyar `[一作]` (University of Southern Denmark), Denis Pankratov `[通讯]` (Concordia University)

**通讯引用:** 1440 | [OpenAlex ID](https://openalex.org/A5024673617)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在线加权非交叉匹配（Online Weighted Non‑Crossing Matching）问题，提出多种算法与理论分析；

**💡 创新点**

主要创新包括：①证明在无权重限制下，确定性算法无法获得常数竞争比；②给出权重限制在[1,U]时的上界与下界；③设计随机化Tree‑Guided‑Matching算法，实现1/3的严格竞争比；④在可撤销模型中获得约0.286的常数竞争比；⑤利用Catalan数的Dyck词实现最优匹配，且仅需O(log Cₙ)=O(n)位辅导信息；

**🔧 技术方法**

使用对抗性构造、凸区域划分、二叉树引导、责任关系、Dyck词、期望分析与Yao最小化原理等技术；

**📊 数据集**

全部实验均采用人工构造的随机/对抗输入，未使用真实数据集；

**📈 对比分析**

理论结果显示：确定性竞争比随U指数衰减；随机化算法竞争比为1/3；可撤销算法竞争比≈0.286；在共线点上随机+撤销可达0.5；辅导信息模型实现完美匹配，仅需O(n)位；

**⚠️ 局限性**

结果尚未收敛至最优，证明与算法间存在间隙；仅考虑顶点权重，未涉及边权或其他几何约束；未给出实验验证；在一般位置下随机或撤销单独无效。

---

## 388. Training-free Motion Factorization for Compositional Video Generation

**arXiv ID:** 2603.09104 | [PDF](https://arxiv.org/pdf/2603.09104v1)

**作者:** Zixuan Wang `[一作]` (Sichuan University), Yinjie Lei `[通讯]` (Sichuan University)

**通讯引用:** 3050 | [OpenAlex ID](https://openalex.org/A5102831936)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了运动因式分解框架，按运动无动、刚体运动、非刚体运动三类对场景动态进行分解，并在此基础上实现对多实例视频的生成

**💡 创新点**

创新点在于通过结构化运动图解决文本模糊性，采用分解后独立的运动引导分支（静态、刚体、非刚体）实现跨实例、跨帧的运动多样性提升

**🔧 技术方法**

技术主要包括LLM（LLaMA‑3.3‑70B）生成运动图，Structured Motion Reasoning（SMR）推断框架/位置序列，Disentangled Motion Guidance（DMG）在视频扩散模型（VideoCrafter‑v2.0 3D‑UNet 与 CogVideoX‑2B DiT）中实现多分支引导

**📊 数据集**

使用自构造的 CVGBench‑m（1665条来自 MSR‑VTT）和 CVGBench‑p（994条来自 Panda‑70M）作为评测数据集，并在两者上进行基准测试

**📈 对比分析**

与多种基线（传统 T2V、VideoTetris、Vico、BoxDiff、R&B、A&R 等）对比，评估五项时序质量指标（主体一致性、背景一致性、闪烁、运动平滑度、动态度），在 CVGBench‑m/p 上均获得显著提升，尤其在主体一致性和动态度方面提升 5‑10% 以上

**⚠️ 局限性**

局限性包括对稀有语义（如“Dendroid”）和情感表情（如“sad”）的生成效果不足，原因是模型对这类概念缺乏足够的特征空间和情绪指令处理能力

---

## 389. Lightweight 3D LiDAR-Based UAV Tracking: An Adaptive Extended Kalman Filtering Approach

**arXiv ID:** 2603.09783 | [PDF](https://arxiv.org/pdf/2603.09783v1)

**作者:** Nivand Khosravi `[一作]` (Instituto Superior Técnico), Rodrigo Ventura `[通讯]` (Instituto Superior Técnico)

**通讯引用:** 1971 | [OpenAlex ID](https://openalex.org/A5052413681)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了基于轻量级LiDAR的无人机跟踪框架，采用自适应扩展卡尔曼滤波器处理非重复扫描点云，实现相对定位。

**💡 创新点**

动态自适应过程噪声与测量噪声矩阵，利用创新与残差统计调整噪声；恢复机制保证检测缺失时跟踪连续；为稀疏点云优化DBSCAN聚类。

**🔧 技术方法**

自适应扩展卡尔曼滤波（AEKF/CAEKF）、Mahalanobis门控关联、DBSCAN聚类、Voxel网格下采样、Livox Mid-360 LiDAR、RTK基准定位。

**📊 数据集**

实测数据：配备Livox Mid-360的DJI F550无人机与另一F550目标，收集点云与RTK定位数据。

**📈 对比分析**

与固定噪声常加速卡尔曼滤波和粒子滤波对比；CAEKF在3D RMSE上比固定KF低78.5%，比PF低49.1%；CPU占用约106%（可接受），回调率9.3Hz，显示出优越的精度与实时性。

**⚠️ 局限性**

仅实现单目标跟踪，难以直接扩展到多目标与复杂遮挡；在极稀疏点云或极低测量频率下仍可能出现误检或漂移，需进一步完善恢复与关联策略。

---

## 390. Towards Understanding Adam Convergence on Highly Degenerate Polynomials

**arXiv ID:** 2603.09581 | [PDF](https://arxiv.org/pdf/2603.09581v1)

**作者:** Zhiwei Bai `[一作]` (Shanghai Jiao Tong University), Yaoyu Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1090 | [OpenAlex ID](https://openalex.org/A5100680347)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究Adam优化器在高阶退化多项式目标函数上的收敛性质，给出了自发收敛的理论条件，并证明其在此类函数上实现局部线性收敛；

**💡 创新点**

创新点在于首次识别一类完全退化多项式，使Adam无需学习率调度即可自动收敛；通过对Adam状态空间的定点与雅可比谱分析，揭示了v_t与g_t^2的解耦机制，解释了指数加速；并系统描绘了Adam的三种行为相位图（稳定、spike、SignGD型振荡）。

**🔧 技术方法**

使用了非线性动力学建模、固定点稳定性与谱半径分析、有效曲率与学习率放大理论，以及与梯度下降、动量法的对比实验；

**📊 数据集**

实验仅基于合成的退化多项式L(x)=1/kx^k（k≥4，偶数）以及少量高维混合曲线（quadratic+quartic）进行，未使用公开真实数据集。

**📈 对比分析**

与梯度下降和动量法在相同退化多项式上进行比较，Adam在满足理论稳定区间内收敛速度呈指数/线性级别，最终损失远低于GD和Momentum；在混合曲线实验中，Adam在退化方向保持快收敛，而GD/Momentum被退化瓶颈限制。

**⚠️ 局限性**

局限性在于理论分析仅覆盖确定性全批退化多项式，忽略了随机采样噪声；假设ε→0且偏差校正可忽略，实际在高维非多项式损失上可能表现不同；此外，对非退化方向的稳定性分析较为粗略。

---

## 391. "Who wants to be nagged by AI?": Investigating the Effects of Agreeableness on Older Adults' Perception of LLM-Based Voice Assistants' Explanations

**arXiv ID:** 2603.09012 | [PDF](https://arxiv.org/pdf/2603.09012v1)

**作者:** Niharika Mathur `[一作]` (Georgia Institute of Technology), Smit Desai `[通讯]` (Northeastern University)

**通讯引用:** 323 | [OpenAlex ID](https://openalex.org/A5033717301)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过实验设计，探讨LLM驱动的语音助手（VA）的agreeableness（同理心、友好度）如何影响老年用户对其解释的感知，并进一步考察情境（日常提醒 vs 紧急警报）和解释来源（对话历史 vs 实时环境数据）对这一关系的调节作用。

**💡 创新点**

创新点在于：①首次将人格维度agreeableness与AI解释接受度结合研究；②系统性比较不同情境与解释来源对同理心影响的交互效应；③发现高agreeableness在日常情境提升信任、亲和力和采纳意愿，而在紧急情境下效果下降；④揭示用户自身agreeableness对低agreeableness VA 的评价具有更强的负面调节作用。

**🔧 技术方法**

技术手段包括：使用 GPT‑5.0 生成基于情境的解释文本；通过 Trait Modulation Key (TMK) 提示框架实现高低agreeableness的对话风格调节；构建交互式故事板 UI 以呈现提醒/警报与解释；采用七项已验证的主观评估量表（PETS、Godspeed、Trust in Automation 等）收集用户反馈；使用非参数统计方法（Kruskal‑Wallis、Wilcoxon）分析实验数据。

**📊 数据集**

数据来源：受试者为 70 名美国老年人（60–83 岁），通过 Prolific 招募；解释文本由 GPT‑5.0 生成并人工审核；故事板图像由 Gemini 语言模型生成；无外部公开数据集使用，实验为自建数据集。

**📈 对比分析**

通过非参数检验比较高低 agreeableness、情境（C1：日常提醒，C2：紧急警报）以及解释类型（UH：历史对话，ENV：实时环境）对意向采用、同理心、亲和力、信任、依赖、满意度等指标的影响。结果显示：①高 agreeableness 在日常情境中显著提升多项指标；②在紧急情境中优势减弱；③环境解释在依赖、满意度、亲和力和智能感知上优于历史解释；整体说明高 agreeableness 与实时环境解释的组合最为有效。

**⚠️ 局限性**

局限性包括：①仅关注 agreeableness 这一人格维度，未探索其他五大人格特质；②采用二元操纵，未呈现人格特质的连续性；③实验仅使用单一合成声音，未考察语音语调对感知的影响；④情境设置为实验室控制情境，缺乏现场长期部署验证；⑤未考虑用户对技术的持续适应与失败经验。

---

## 392. Understanding the Interplay between LLMs' Utilisation of Parametric and Contextual Knowledge: A keynote at ECIR 2025

**arXiv ID:** 2603.09654 | [PDF](https://arxiv.org/pdf/2603.09654v1)

**作者:** Isabelle Augenstein `[一作]` (University of Copenhagen), Isabelle Augenstein `[通讯]` (University of Copenhagen)

**通讯引用:** 4601 | [OpenAlex ID](https://openalex.org/A5018976680)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并评估LLM在推理任务中对参数知识与检索知识的使用，并提出统一的归因评估框架、冲突检测指标及新的现实场景数据集。

**💡 创新点**

提出统一的Instance与Neuron归因对比框架，设计了语义熵和一致说服分数来量化内存冲突，创建了DynamicQA和DRUID数据集并提出ACU度量，揭示了事实动态性对检索效果的重要性。

**🔧 技术方法**

采用归因方法（Instance Attribution, Neuron Attribution）、faithfulness测试、语义熵、一致说服分数、ACU度量、检索增强生成（RAG）以及上下文操控技术（CMT）和机制干预等技术。

**📊 数据集**

使用DynamicQA（包含静态、时间、争议事实）与DRUID（真实世界主张验证）数据集，并对比合成数据集，训练集与测试集来源于公开基准（如事实检查、自然语言推理）。

**📈 对比分析**

通过与随机实例对比的Fine‑Tuning、实例与神经元归因对齐的faithfulness评估、内存冲突指标对RAG表现的影响以及不同CMT在多数据集上的比较，发现大模型在利用检索信息上表现更好，但未出现单一最优CMT，且在现实数据中与合成数据的差距显著。

**⚠️ 局限性**

对归因方法缺乏共识且计算成本高；内存冲突度量仍需进一步验证；合成数据可能夸大冲突特征；实验主要聚焦事实检索与主张验证，未覆盖所有任务场景，结论具有一定局限性。

---

## 393. QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model

**arXiv ID:** 2603.09125 | [PDF](https://arxiv.org/pdf/2603.09125v1)

**作者:** Junjie Yin `[一作]` (South China Normal University), Hanfa Xing `[通讯]` (South China Normal University)

**通讯引用:** 17194 | [OpenAlex ID](https://openalex.org/A5100345691)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的图像超分辨率框架 QUSR，利用多模态大型语言模型生成质量感知先验并通过不确定性引导噪声自适应注入来提升细节恢复效果。

**💡 创新点**

创新点在于：①将 MLLM 生成的质量描述作为全局语义与降质先验，②设计单步残差扩散网络配合不确定性引导噪声模块，动态调整噪声强度以强化纹理细节而非过度平滑。

**🔧 技术方法**

核心技术包括 Stable Diffusion 2.1 的 UNet + VAE 架构、LoRA 微调、CLIP 文本编码器、Qwen2.5‑VL‑7B‑Instruct 生成质量先验、轻量级不确定性估计网络、LPIPS、CSD 以及不确定性损失等。

**📊 数据集**

使用 RealSR、DRealSR 真实数据集，以及 LSDIR 与 FFHQ 训练集（RealESRGAN 合成 128×128→512×512 的低高分辨率对）。

**📈 对比分析**

在 RealSR 与 DRealSR 上与 StableSR、SeeSR、SinSR、OSEDiff、PiSA‑SR 等 SOTA 方法对比，PSNR、SSIM 以及无参考指标 CLIPIQA/MUSIQ/ MANIQA 均达标，尤其在 DRealSR 上 FID 大幅下降、MUSIQ 上升，证明在真实场景下具备更高的视觉质量。

**⚠️ 局限性**

局限性包括：依赖 MLLM 的描述准确性；模型在极端降质或不同尺度上可能仍产生失真；单步扩散与自适应噪声的参数调优较为敏感，且推理时间相对较长。

---

## 394. MuxGel: Simultaneous Dual-Modal Visuo-Tactile Sensing via Spatially Multiplexing and Deep Reconstruction

**arXiv ID:** 2603.09761 | [PDF](https://arxiv.org/pdf/2603.09761v1)

**作者:** Zhixian Hu `[一作]`, Yu She `[通讯]` (Purdue University)

**通讯引用:** 1444 | [OpenAlex ID](https://openalex.org/A5018653973)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在 GelSight 触觉传感器胶垫上使用棋盘格涂层，实现同一摄像头同时获取可视与触觉信息，并通过深度学习解耦恢复两种模态的高分辨率图像，进一步应用于闭环抓取实验。

**💡 创新点**

创新点包括：① 硬件层面采用可插拔的棋盘格涂层实现空间多路复用；② 软件层面设计双流 U‑Net（共享 ResNet-34 编码器、两个解码器）并使用残差学习恢复触觉；③ 通过大规模仿真与实时域随机化的 sim‑to‑real 训练策略，实现从仿真到真实世界的无缝迁移；④ 在不改动原有 GelSight 机械光学结构的前提下完成插件式升级。

**🔧 技术方法**

技术手段包括：棋盘格涂层与透明窗口的硬件设计；MuJoCo 物理仿真 + Taxim 光照映射；域随机化（背景模糊、颜色抖动、波纹棋盘）提升泛化；ResNet‑34‑UNet 网络（共享编码器、双解码器）结合绝对/残差触觉预测；多任务损失（L1、梯度、SSIM、LPIPS、感知损失）与两阶段训练；以及 3‑轴线性平台进行自动化真实数据采集。

**📊 数据集**

使用的主要数据集包括：Google Scanned Objects（1000+ 3D 物体）用于仿真；IndoorCVPR 背景图像用于域随机化；自建真实数据集（25 种几何/颜色指针、5 个 GelGel 配置、848 对象 50 视角、5 压深）用于 fine‑tune 与评估。

**📈 对比分析**

与单输入、双输入绝对预测等变体进行对比。零射仿真下 DI‑ResT 在触觉 RMSE 0.083、1‑SSIM 0.1227、LPIPS 0.109；fine‑tune 后触觉 RMSE 0.0287、1‑SSIM 0.0878、LPIPS 0.0489；在不同棋盘配置（2×2、4×4、8×8）中，4×4 在触觉上表现最佳，而 8×8 在视觉上更优。最终在 9 种未知物体上实现 100% 抓取成功率。

**⚠️ 局限性**

局限性：① 视觉与触觉的多路复用导致两者在分辨率和信息量上存在权衡；② 对光照、背景和胶层变形的鲁棒性依赖仿真随机化；③ 需要在每种硬件配置下重新训练或 fine‑tune，扩展性受限；④ 目前只验证了 GelSight 形式，其他传感器的迁移仍需研究。

---

## 395. ZeroWBC: Learning Natural Visuomotor Humanoid Control Directly from Human Egocentric Video

**arXiv ID:** 2603.09170 | [PDF](https://arxiv.org/pdf/2603.09170v1)

**作者:** Haoran Yang `[一作]` (University of Science and Technology of China), Xuelong Li `[通讯]` (TeleAI, China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ZeroWBC 框架，直接利用人类第一人称视频和 MoCap 数据实现全身机器人视觉运动控制，无需昂贵的机器人遥控数据。

**💡 创新点**

创新点在于将 Vision‑Language Model 与 VQ‑VAE 结合生成运动令牌，采用两阶段架构（运动生成 + 通用追踪）实现自然、可交互的全身运动，并通过自采集的 egocentric 视频大幅降低数据成本。

**🔧 技术方法**

使用 Qwen2.5‑VL 预训练模型、VQ‑VAE 离散化运动、强化学习基准的通用运动追踪器、以及自适应难度调度与纵向时间编码等技术。

**📊 数据集**

使用 Nymeria 视角视频+MoCap 数据、HumanML3D 文本-运动数据，以及自采集的 5 小时 egocentric 运动+视频+文本数据。

**📈 对比分析**

与 MotionGPT、GMT 等基线对比，ZeroWBC 在多模态运动生成上 FID、R‑Precision、MM‑Dist 等指标大幅提升；在通用追踪上 MPJPE、MPJAE、MPJVE 均低于 GMT，且在真实机器人上实现了 95% 以上障碍规避、78% 球踢击、84% 软垫坐下等任务成功率。

**⚠️ 局限性**

主要限制包括 VLM 推理时延高（≥500 ms）影响实时交互、缺乏触觉反馈导致精细操作受限、以及人机形态差异导致运动重映射需进一步优化。

---

## 396. World2Mind: Cognition Toolkit for Allocentric Spatial Reasoning in Foundation Models

**arXiv ID:** 2603.09774 | [PDF](https://arxiv.org/pdf/2603.09774v1)

**作者:** Shouwei Ruan `[一作]` (Beihang University), Yubin Wang `[通讯]` (Huawei)

**通讯引用:** 18897 | [OpenAlex ID](https://openalex.org/A5041572550)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个无训练的空间认知工具包 World2Mind，利用 3D 重建与实例分割生成结构化的 allocentric 空间认知图，帮助多模态基础模型在面对复杂空间推理任务时主动获取所需的空间知识。

**💡 创新点**

核心创新在于：① 通过“Allocentric‑Spatial Tree (AST)”将空间实体建模为椭圆参数化的有向无环图，既能捕捉人类模糊感知，又能提供稳健的几何拓扑先验；② 设计了三阶段跨模态推理链（工具调用评估、模态解耦线索收集、几何‑语义交织推理），有效缓解 3D 重建误差与视觉观测冲突。

**🔧 技术方法**

技术手段包括：Depth Anything V3 进行单目深度估计；SAM3 进行开放词汇语义分割；点云生成与密度过滤；基于 DBSCAN 的实例聚类和椭圆拟合；可视化路由图与可通行性映射；以及文本化的 AST 结构化输出。

**📊 数据集**

使用了 VSI‑Bench（视频空间推理）与 MindCube（多视角认知映射）两个基准数据集进行评估，并在 Tiny 子集上对 GPT‑5.2、Claude‑4.6‑Opus、Gemini‑3‑Pro 等前沿多模态模型进行测试。

**📈 对比分析**

与不使用 World2Mind 的基线相比，所有模型均提升 5%–18% 的平均准确率；在关键子任务如相对方向、路径规划、相对距离等，Claude‑4.6‑Opus 最高可提升 30.6%；在无视觉输入的“盲”场景下，文本化 AST 能将模型性能恢复至接近完整输入水平。

**⚠️ 局限性**

局限性包括：① 依赖 3D 重建的质量，重建误差仍可能导致推理错误；② AST 结构在极端遮挡或稀疏视角下可能缺失关键信息；③ 目前仅在 Tiny 子集验证，未覆盖更大规模或更复杂环境；④ 需要额外的计算资源进行重建和分割，影响实时性。

---

## 397. Experience Report on the Adaptable Integration of Requirements Engineering Courses into Curricula for Professionals

**arXiv ID:** 2603.09467 | [PDF](https://arxiv.org/pdf/2603.09467v1)

**作者:** Oleksandr Kosenkov `[一作]` (Blekinge Institute of Technology), Davide Fucci `[通讯]` (Blekinge Institute of Technology)

**通讯引用:** 1104 | [OpenAlex ID](https://openalex.org/A5023279547)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

在三份面向专业软件工程师的课程体系中，作者通过基于内容项（CI）的轻量级对齐方法，系统地将需求工程（RE）课程融入并与其它课程形成学习路径。

**💡 创新点**

创新点在于：①将课程对齐过程从自顶向下转为自底向上、以课程内容为中心；②强调教师主导、协作式对齐；③通过共享文档和CI序列化实现持续、灵活的课程适配与更新。

**🔧 技术方法**

采用的技术主要是：课程内容拆分为10–15分钟的内容项、利用draw.io和UML类图描述CI、在共享文档中进行协同排序与整合，以及基于CI构建学习模块与学习路径。

**📊 数据集**

本研究未使用公开数据集，而是基于三项目（PROMIS、Software4KMU、TASTE）的实际课程材料、教师访谈、焦点小组讨论以及学习路径设计文档作为实践数据。

**📈 对比分析**

对比方法：作者通过与项目内部教师和管理者的访谈、学生反馈（PROMIS已获得正面评价）以及对齐后学习路径的可视化，展示对齐方法在提高课程一致性和实用性方面的效果；然而并未提供量化性能指标。

**⚠️ 局限性**

局限性包括：①缺乏系统的学生学习效果评估；②方法对非RE课程的适用性尚待验证；③需要更清晰的教师参与动机和行政协调机制；④在大规模推广前需进一步细化和验证。

---

## 398. The Costs of Reproducibility in Music Separation Research: a Replication of Band-Split RNN

**arXiv ID:** 2603.09187 | [PDF](https://arxiv.org/pdf/2603.09187v1)

**作者:** Paul Magron `[一作]` (Universite de Lorraine), Constance Douwes `[通讯]` (Centrale Med)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对Band‑Split RNN (BSRNN) 进行复制实验并改进其模型与训练流程，最终提出更高性能的oBSRNN与oBSRNN‑SIMO

**💡 创新点**

系统化地评估BSRNN各设计与训练参数的影响，揭示多种可改进的方向，并在没有官方完整代码的情况下实现可复现的完整训练管线

**🔧 技术方法**

基于双向LSTM/双路径网络、可选的多头/注意力模块、稀疏卷积替代、以及混声道处理（TAC）等深度学习技术

**📊 数据集**

使用公开的MUSDB18‑HQ数据集进行训练、验证与测试

**📈 对比分析**

通过与原论文以及其他SOTA模型（如Hybrid Demucs、HT Demucs、BS‑RoFormer等）的SDR、cSDR比较，oBSRNN在无额外数据的情况下平均提升约1.5 dB，oBSRNN‑SIMO甚至超过原始BSRNN与BS‑RoFormer的平均uSDR

**⚠️ 局限性**

复制过程受限于缺乏官方完整代码、训练硬件与能源成本高昂；改进仍需进一步验证对不同数据集与模型规模的适用性

---

## 399. Automating Detection and Root-Cause Analysis of Flaky Tests in Quantum Software

**arXiv ID:** 2603.09029 | [PDF](https://arxiv.org/pdf/2603.09029v1)

**作者:** Janakan Sivaloganathan `[一作]` (Toronto Metropolitan University), Lei Zhang `[通讯]` (University of Maryland Baltimore County)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并扩展了量子软件 flaky 测试数据集，提出自动化管道通过自然语言和代码上下文识别 flaky 测试及其根因。

**💡 创新点**

首次将大型语言模型（LLM）应用于量子 flaky 测试检测与根因诊断，并实现 54% 数据集扩增和 0.9420 的 F1 评分。

**🔧 技术方法**

使用 LLM（OpenAI GPT、Meta LLaMA、Google Gemini、Anthropic Claude）结合嵌入式相似度检索、few-shot 提示和代码上下文推理。

**📊 数据集**

利用 12 个开源量子项目（如 Qiskit、NetKet）共 8,628 条 issue/PR，扩增至 71 条 flaky 测试实例，并在 Zenodo 上公开数据。

**📈 对比分析**

通过多模型、多上下文实验（R_p/C_p、R_f/C_f、E_p/E_f），对比 F1、MCC、召回率，Google Gemini 2.5 Flash 在 R_f/C_f 条件下达到最佳 F1=0.9420（flaky 检测）和 0.9643（根因识别）。

**⚠️ 局限性**

局限包括数据集规模仍有限、仅覆盖 Python 代码、缺乏动态重跑验证、模型推理不确定性及对不同量子平台泛化能力待进一步评估。

---

## 400. Towards a Neural Debugger for Python

**arXiv ID:** 2603.09951 | [PDF](https://arxiv.org/pdf/2603.09951v1)

**作者:** Maximilian Beck `[一作]` (Johannes Kepler University Linz), Gabriel Synnaeve `[通讯]` (Meta)

**通讯引用:** 8217 | [OpenAlex ID](https://openalex.org/A5041907084)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并评估了能够基于调试器动作预测 Python 程序行级执行状态的神经调试器（支持正向和逆向执行）。

**💡 创新点**

将调试器建模为马尔可夫决策过程（MDP），构造了正向与逆向的状态树，并设计了专门的神经调试语言格式，利用此格式在大规模训练数据上微调/预训练 Transformer LLM，实现了高精度的状态预测与逆向推断。

**🔧 技术方法**

采用 Transformer 语言模型（1.8B、32B），在 CWM 训练框架下进行微调或从零预训练；通过 Python trace API 收集执行轨迹，构建调试器轨迹采样策略；使用特殊符号序列将状态与动作序列序列化，形成可直接输入模型的文本格式。

**📊 数据集**

使用 CWM（Python 代码+执行轨迹）数据集（函数级和仓库级，约 150B 训练 token），并在 CruxEval 上评估输入/输出预测任务；训练管道同时产生正向与逆向轨迹。

**📈 对比分析**

比较方法：对每个调试动作计算 exact‑match 下的下一个状态预测准确率；在 CruxEval 上使用 pass@1 测度输入/输出预测；结果显示 32B 微调模型在正向预测上超过 90% 准确率，1.8B 预训练模型在 CruxEval 上达到 53.6/57.7 的 pass@1，32B 微调模型进一步提升到 66.5/83.2；同时展示了预测精度随跳跃距离增加而下降的趋势。

**⚠️ 局限性**

局限性：仅针对 Python；轨迹采样使用随机动作，缺乏针对性；逆向预测存在多义性，评测指标不够完善；对大型 Python 对象的序列化会导致长度膨胀；缺少多语言支持和更丰富的调试动作；局部变量预测准确率仍低于控制流预测。

---

## 401. Learning Convex Decomposition via Feature Fields

**arXiv ID:** 2603.09285 | [PDF](https://arxiv.org/pdf/2603.09285v1)

**作者:** Yuezhi Yang `[一作]` (NVIDIA), Nicholas Sharp `[通讯]` (NVIDIA)

**通讯引用:** 4147 | [OpenAlex ID](https://openalex.org/A5036423808)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于特征学习的自监督对比损失来实现3D形状的凸分解，并训练一个前馈网络在开世界数据上直接生成高质量的凸分解。

**💡 创新点**

创新点在于：① 将凸分解问题转化为连续特征学习与对比学习任务；② 设计基于几何凸性判定的自监督对比损失；③ 通过递归聚类实现可调粒度的凸分解，且模型能跨多种输入模态（网格、点云、Gaussian splat）应用。

**🔧 技术方法**

使用点云输入的PVCNN编码器、三平面（triplane）特征表示、2D CNN+Transformer结构以及自监督对比损失；后续对特征进行递归二分聚类，并计算凸包实现分解。

**📊 数据集**

主要在Objaverse数据集上训练，测试数据包含V-HACD、PartObjaverse-Tiny和ShapeNet三组模型。

**📈 对比分析**

与传统算法CoACD、V-HACD以及学习方法Cvx-Net、BSP-Net进行对比。实验表明在所有数据集上都能以更低的凸度（concavity）和更小的重建误差获得更优的凸分解；在碰撞检测任务中，使用该分解可实现约5倍的速度提升。

**⚠️ 局限性**

局限性：模型仅在干净的单体对象数据上训练，难以推广到场景级或高度缺失的几何；对薄结构（如风扇框架）表现欠佳；需要进一步训练以适应噪声或场景规模。

---

## 402. The Temporal Markov Transition Field

**arXiv ID:** 2603.08803 | [PDF](https://arxiv.org/pdf/2603.08803v1)

**作者:** Michael Leznik `[一作]` `[通讯]` (Aristocrat Leisure Limited), Michael Leznik (Aristocrat Leisure Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种新的时序二维图像表示方法——Temporal Markov Transition Field（TMTF），通过将时间序列分段并分别估计局部转移矩阵，来捕捉随时间变化的转移动力学。

**💡 创新点**

创新点在于将原本单一全局转移矩阵的 Markov Transition Field（MTF）改为 K 个局部转移矩阵，生成具有水平带结构的图像，从而保留了时序中不同时段的动态差异，并且在保持幅度不变性和顺序保留性的前提下，适合作为卷积神经网络的输入通道。

**🔧 技术方法**

主要技术包括：对原始序列进行分位数量化并得到状态序列；在每个时间段内估计经验转移矩阵；利用局部矩阵填充图像得到 T × T 的 TMTF；随后可将 TMTF 作为通道输入 CNN 进行时序特征提取；文中还讨论了多分辨率扩展、偏差-方差权衡和几何解释等。

**📊 数据集**

文中未给出具体实验数据集；该工作主要聚焦方法论与理论分析，随后可与在伴随论文中进行的站稳性分类/回归实验相结合。

**📈 对比分析**

由于缺乏实验结果，本文没有提供与其他时序表示（如原 MTF、Gramian Angular Field 等）的性能对比；方法的优点仅在理论上通过分段局部矩阵提升对非平稳过程的表达能力得到说明。

**⚠️ 局限性**

主要局限包括：需要足够多的观测来估计每个段的转移矩阵（对 K 与 T 的平衡有严格约束）；局部估计方差随段数增大而上升；对段界限的选择依赖先验假设或手动分割；以及在真正平稳过程中，TMTF 与全局 MTF 完全相同，无法提供额外信息。

---

## 403. Memory-Augmented Spiking Networks: Synergistic Integration of Complementary Mechanisms for Neuromorphic Vision

**arXiv ID:** 2603.08730 | [PDF](https://arxiv.org/pdf/2603.08730v1)

**作者:** Effiong Blessing `[一作]` (Saint Louis University), Junaid Rehman `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对N-MNIST数据集进行了五模型消融实验，研究了将Leaky Integrate-and-Fire神经元、监督对比学习、Hopfield网络和层次门控递归网络融合到Spiking Neural Network中的效果。

**💡 创新点**

首次系统性揭示多种记忆增强机制在SNN中的协同效应，证明整体架构平衡比单个优化策略能显著提升精度、聚类质量和能效。

**🔧 技术方法**

采用LIF神经元、监督对比学习、能量基础的Hopfield网络以及Hierarchical Gated Recurrent Network，并利用snntorch实现可微分反向传播。

**📊 数据集**

使用DVS128摄像机记录的N-MNIST neuromorphic视觉数据集进行训练与评估。

**📈 对比分析**

将五种配置的验证/测试精度、silhouette系数、能耗（µJ）与现有SNN与ANN方法对比，最佳模型达到97.49%精度、0.715 silhouette、1.85 µJ/推理，显著优于基线并保持高能效。

**⚠️ 局限性**

缺乏对更复杂视觉任务的验证，且Hopfield的离散迭代和SCL对齐问题在单一组件时易产生冲突，未来需探索更通用的集成策略与硬件实现。

---

## 404. Hindsight Credit Assignment for Long-Horizon LLM Agents

**arXiv ID:** 2603.08754 | [PDF](https://arxiv.org/pdf/2603.08754v1)

**作者:** Hui-Ze Tan `[一作]` (Nanjing University), Yu-Feng Li `[通讯]` (Nanjing University)

**通讯引用:** 41523 | [OpenAlex ID](https://openalex.org/A5082124101)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向长时程LLM代理的价值无关强化学习框架HCAPO，利用后向信用分配提升稀疏奖励任务的学习效率。

**💡 创新点**

创新点在于将后向信用分配与LLM自身的生成验证相结合，构造自归一化的重要性比率，解决传统GRPO在长程任务中步级价值估计不准和基线不匹配的问题。

**🔧 技术方法**

核心技术包括生成验证（使用LLM作为后向批评者）、自归一化重要性比率估计、多尺度优势融合与PPO优化。

**📊 数据集**

使用WebShop、ALFWorld以及搜索增强问答（如NQ、HotpotQA、TriviaQA等）等公开基准数据集进行评估。

**📈 对比分析**

与基准GRPO、RLOO、EMPG、GiGPO以及多种提示式代理相比，HCAPO在WebShop和ALFWorld上分别提升成功率约7.7%和13.8%，在搜索问答任务也实现了显著的准确率提升。

**⚠️ 局限性**

局限性在于依赖大模型的推理能力，模型规模较小时信用信号精度受限；同时后向信息可能引入与任务分布不匹配的噪声。

---

## 405. Training-Free Coverless Multi-Image Steganography with Access Control

**arXiv ID:** 2603.09390 | [PDF](https://arxiv.org/pdf/2603.09390v1)

**作者:** Minyeol Bae `[一作]` (Korea Advanced Institute of Science and Technology), Si-Hyeon Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 537 | [OpenAlex ID](https://openalex.org/A5091779193)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了 MIDAS，一种训练‑free、基于预训练扩散模型的多图隐藏且带访问控制的无封面图像隐写框架。

**💡 创新点**

创新点在于引入随机基准（Random Basis）和潜向量融合（Latent Vector Fusion）两种技术，既实现多图隐藏又实现用户级访问控制，同时通过消除残余结构显著提升隐藏图像的多样性与质量。

**🔧 技术方法**

采用 Stable Diffusion v1.5 的潜向量扩散与 DDIM 逆向，配合 EDICT、随机基准矩阵、潜向量融合等机制完成隐藏与解密流程。

**📊 数据集**

使用 Stego260 与 UniStega 两个公开数据集，随机挑选 N=2（可扩展到 N=8）张图像作为多图隐藏实验。

**📈 对比分析**

与 CRoSS*、DiffStega*、IIS、AIS 等基线对比，MIDAS 在隐藏图像质量（MANIQA）、多样性（LPIPS、CLIP Score）、访问控制成功率、抗噪声、抗 Steganalysis 等指标上均优于对手，并且在多图数量增大时仍保持稳健表现。

**⚠️ 局限性**

主要局限是扩散采样导致的推理延迟较高，且目前仅针对图像隐写，后续工作需优化采样策略并探索更高效或多模态的应用。

---

## 406. CREATE: Testing LLMs for Associative Creativity

**arXiv ID:** 2603.09970 | [PDF](https://arxiv.org/pdf/2603.09970v1)

**作者:** Manya Wadhwa `[一作]` (New York University), Greg Durrett `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了CREATE基准，用于评估大型语言模型在关联创造性推理上的能力；

**💡 创新点**

提出了“关联创造性”概念，结合特异性与多样性评估指标，并构建了可客观评分的路径生成任务；

**🔧 技术方法**

采用LLM与思考型（chain‑of‑thought）提示、知识图推理、距离与特异性计算以及创意效用度量等技术；

**📊 数据集**

基于Wikidata构建了931个自然语言查询，涵盖人物、基因、化学化合物等12类关系；

**📈 对比分析**

通过与非思考与思考LLM（如GPT‑4.1、GPT‑5、Gemini‑3‑pro、Claude、Qwen、OLMo等）在创意效用、质量、距离、事实性等指标上对比，发现前沿模型在创意效用上领先，但在事实性和多样性上仍有提升空间；

**⚠️ 局限性**

受限于知识图结构导致路径多样性受限；思考预算不一定提升创意；模型在兼顾高质量与事实性时易失真，评估仍受有限人类标注和距离阈值影响。

---

## 407. Proportionality Degree in Participatory Budgeting

**arXiv ID:** 2603.09660 | [PDF](https://arxiv.org/pdf/2603.09660v1)

**作者:** Aris Filos-Ratsikas `[一作]`, Georgios Kalantzis `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了审批型参与式预算（PB）中两种主流规则——等份共享方法（MES）与Phragmén序贯规则——的比例度（proportionality degree）及其公平性。

**💡 创新点**

创新点在于首次为这两种规则给出近似最优的比例度上界与下界，证明它们在量化公平度上具有相同的表现，并提出通用的EJR规则比例度下界；同时结合实验验证理论结果。

**🔧 技术方法**

主要技术包括对MES和Phragmén规则的潜在函数与支付上界分析、构造极端实例实现上界、以及对T-凝聚群体平均满意度的数学推导。

**📊 数据集**

实验使用了Pabulib库中的100个真实PB实例，覆盖多种选民与项目规模、成本与批准信息。

**📈 对比分析**

通过随机采样子集，计算各规则的平均比例度，结果显示MES+耗尽版与Phragmén+耗尽版在绝大多数数据集上优于贪心规则，且Phragmén在无耗尽情况下也表现良好。

**⚠️ 局限性**

局限性包括：比例度的上界仅为近似（加上常数），未给出完全精确最优性证明；实验采用随机采样而非全枚举，可能遗漏极端子集；仅考虑审批型偏好，未扩展到其他偏好模型。

---

## 408. See, Plan, Rewind: Progress-Aware Vision-Language-Action Models for Robust Robotic Manipulation

**arXiv ID:** 2603.09292 | [PDF](https://arxiv.org/pdf/2603.09292v1)

**作者:** Tingjun Dai `[一作]` (University of Science and Technology of China), Xiaojun Chang `[通讯]` (MBZUAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出See-Plan-Rewind框架，实现基于空间子目标的进度感知与自我恢复的视觉语言动作模型。

**💡 创新点**

创新点在于将任务拆分为可量化的2D子目标，实现精确进度监测并通过无额外数据的Rewind机制实现错误自恢复。

**🔧 技术方法**

使用预训练视觉模型（DINOv3、SAM、Gemini-3）与大型语言模型（DeepSeek-R1、Gemini-3）以及自监督的轨迹提取和逆向数据生成技术。

**📊 数据集**

在LIBERO、LIBERO-Plus机器人操控基准以及三项真实机器人任务上进行评测。

**📈 对比分析**

与MolmoAct、OpenVLA-OFT、UniVLA等基线比较，SPR在LIBERO上提升约5%，在LIBERO-Plus上平均性能下降仅18.8%，显著优于其他方法。

**⚠️ 局限性**

局限性包括对极端初始姿态或未见动作类型的恢复仍受限，Rewind步骤可能导致姿态漂移，且在多物体动态环境下的实时规划尚待进一步优化。

---

## 409. KernelCraft: Benchmarking for Agentic Close-to-Metal Kernel Generation on Emerging Hardware

**arXiv ID:** 2603.08721 | [PDF](https://arxiv.org/pdf/2603.08721v1)

**作者:** Jiayi Nie `[一作]` (University of Cambridge), Yiren Zhao `[通讯]` (Imperial College London)

**通讯引用:** 13345 | [OpenAlex ID](https://openalex.org/A5076778501)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出KernelCraft框架，评估LLM代理在新ISA自定义加速器上自动生成和优化低级汇编核的能力。

**💡 创新点**

首次构建统一的benchmark，集成多级任务（原语、组合、端到端）、多加速器平台、工具调用反馈循环，并系统研究LLM在零样本、推理深度、上下文学习等条件下的表现。

**🔧 技术方法**

利用大型语言模型的工具调用（function calling）与迭代调试、编译检查、仿真验证、数值容差对比，结合LLM的推理与生成；对三大新兴加速器（PLENA、AMD NPU、Coral NPU）实现多任务评测。

**📊 数据集**

基于公开的AI工作负载（激活、归一化、矩阵乘法、卷积、Transformer模块等）共33项任务，覆盖不同复杂度等级，使用多种配置组合进行评测。

**📈 对比分析**

通过成功率（功能正确性）和相对速度提升（与对应编译器基准比较）两指标评估；结果显示在原语级别成功率可达74%，在组合级别降至45%，端到端几乎未通过；部分LLM（如GPT‑5.2、Gemini‑3‑Flash）在合成任务上实现1.06–8×速度提升。

**⚠️ 局限性**

受限于文档质量、ISA复杂度、推理深度不足导致高复杂度任务失败；优化质量与成功率不完全相关；缺乏形式化验证与多代理协同；对极低资源平台的适配与可扩展性待进一步研究。

---

## 410. Equitable Multi-Task Learning for AI-RANs

**arXiv ID:** 2603.08717 | [PDF](https://arxiv.org/pdf/2603.08717v1)

**作者:** Panayiotis Raptis `[一作]` (Delft University of Technology), George Iosifidis `[通讯]` (Delft University of Technology)

**通讯引用:** 3900 | [OpenAlex ID](https://openalex.org/A5044138533)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在线多任务公平学习框架OWO‑FMTL，使AI‑RAN在共享边缘模型下动态任务中保证长期公平推理性能。

**💡 创新点**

首次把动态多任务公平性建模为双层在线学习，结合α‑公平度量和双重投影更新，实现零公平性回退。

**🔧 技术方法**

采用在线凸优化、原始‑对偶梯度更新、主/从循环学习、α‑公平度量的共轭表示与梯度权重平衡等技术。

**📊 数据集**

在合成的三次多项式核回归任务和Rainbow MNIST（含多背景、尺度、方向变换）深度学习实验中验证。

**📈 对比分析**

与单轮学习（SRL）及多种常数权重方案比较，实验显示在凸/非凸环境下OWO‑FMTL在公平回退和任务效能上分别提升约20–40%和10–30%，并在对抗环境下保持子线性回退。

**⚠️ 局限性**

对任务相似度假设和对偶变量区间的先验假设敏感，且在高维非凸任务下需近似fairest模型，导致理论与实际性能间的差距。

---

## 411. Memory-Guided View Refinement for Dynamic Human-in-the-loop EQA

**arXiv ID:** 2603.09541 | [PDF](https://arxiv.org/pdf/2603.09541v1)

**作者:** Xin Lu `[一作]` (University of Chinese Academy of Sciences), Yunhong Wang `[通讯]` (Beihang University)

**通讯引用:** 13923 | [OpenAlex ID](https://openalex.org/A5115589096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练的框架 DIVRR，用于在动态、包含人类活动的环境中进行视图精炼和自适应记忆管理，从而提升嵌入式问答（EQA）的鲁棒性与效率。

**💡 创新点**

创新点在于：①将相关性驱动的多视角增强用于实时视图精炼，解决因遮挡和人类动作导致的视角不确定性；②基于相关性门控的自适应记忆录入，仅保留已验证且信息量高的观测，抑制冗余记忆扩张；③构建了新数据集 DynHiL-EQA，包含动态与静态子集，专门考察人类引发的感知非平稳性。

**🔧 技术方法**

核心技术包括：零样本相关性评分（利用 VLM 的 next-token logits 计算 s_t）；多视角增强与视角选择；相关性驱动的记忆录入门控；基于 CLIP 的紧凑嵌入表示；以及 Habitat‑Sim 环境下的 EQA 实现。

**📊 数据集**

使用了 DynHiL-EQA（人类参与动态与静态场景）以及公开的 HM‑EQA（静态场景）进行实验；DynHiL-EQA 共 1100 个问题，涵盖 7 类；HM‑EQA 为基准静态数据集。

**📈 对比分析**

与多种 SOTA 方法（Explore‑EQA、Fine‑EQA、Graph‑EQA、Memory‑EQA 等）进行对比。DIVRR 在 DynHiL‑EQA 的 Dynamic 子集上准确率提升 10.1%，整体提升 7.4%，记忆条目平均减少 74%，推理延迟仅略高 0.2 s；在 HM‑EQA 上准确率达到 63.8%，显著优于 Graph‑EQA，记忆使用量降低 58%。

**⚠️ 局限性**

局限性包括：对长时序动态变化的验证仍不充分；缺乏对复杂社会交互和开放世界对象变化的显式建模；多视角增强的感知预算固定，可能在高遮挡场景下仍需更多视角；以及对 VLM 的依赖使得模型受限于其推理能力。

---

## 412. Temporal-Conditioned Normalizing Flows for Multivariate Time Series Anomaly Detection

**arXiv ID:** 2603.09490 | [PDF](https://arxiv.org/pdf/2603.09490v1)

**作者:** David Baumgartner `[一作]` (Norwegian University of Science and Technology), Heri Ramampiaro `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 1351 | [OpenAlex ID](https://openalex.org/A5026537247)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于时序条件化归一化流（tcNF）的多变量时间序列异常检测框架。

**💡 创新点**

创新点在于将归一化流的条件输入设为先前的观测值，能够精准捕捉时间依赖与不确定性，从而在学习到的分布中识别低概率事件。

**🔧 技术方法**

核心技术包括归一化流（Normalizing Flows）与自回归时序条件化（Temporal Conditioning）相结合的模型架构。

**📊 数据集**

在多种公开时间序列数据集上进行评估，具体数据集未在摘要中列出，但涵盖了不同领域与规模的样本。

**📈 对比分析**

与现有异常检测方法进行对比，实验结果表明 tcNF 在准确率和鲁棒性方面均优于传统模型，并提供了全面的性能分析。

**⚠️ 局限性**

主要局限包括对长序列和高维数据的训练成本较高、对超参数敏感，以及未深入探讨实时检测的可行性。

---

## 413. Fusing Semantic, Lexical, and Domain Perspectives for Recipe Similarity Estimation

**arXiv ID:** 2603.09688 | [PDF](https://arxiv.org/pdf/2603.09688v1)

**作者:** Denica Kjorvezir `[一作]` (Jožef Stefan Institute), Riste Stojanov `[通讯]` (Sveti Cyril and Methodius University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多视角融合方法，用于衡量食谱之间的相似度，结合语义、词汇和营养信息进行综合评分。

**💡 创新点**

创新点在于将三种不同信息源（语义嵌入、层次化的词汇相似度和营养向量）融合成统一的相似度指标，并使用Hungarian算法对词汇和营养层面进行最佳配对。

**🔧 技术方法**

技术包括Transformer（DistilRoBERTa、MiniLM-L6）句子嵌入、Jaccard相似度改进、营养余弦相似度、Hungarian算法配对以及加权融合。

**📊 数据集**

使用公开的 Recipe1M 数据集，包含约400多条菜谱的配料、做法和营养信息。

**📈 对比分析**

通过人工评估318对菜谱（80%一致）构建基准，随后用逻辑回归和随机森林进行验证，两者在相似与不相似分类上均达到了约89%的准确率。

**⚠️ 局限性**

局限在于仍需人工标注验证，数据集中高词汇相似度样本稀缺，且单一营养向量容易产生误判，未来需探索更细粒度的分子层面特征和更大规模数据。

---

## 414. ProGS: Towards Progressive Coding for 3D Gaussian Splatting

**arXiv ID:** 2603.09703 | [PDF](https://arxiv.org/pdf/2603.09703v1)

**作者:** Zhiye Tang `[一作]` (Shenzhen University), Xu Wang `[通讯]` (Shenzhen University)

**通讯引用:** 98220 | [OpenAlex ID](https://openalex.org/A5100424784)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为 ProGS 的 3D 高斯散点化压缩框架，能够实现渐进式编码和实时渲染。

**💡 创新点**

创新点包括：①使用层次化八叉树结构和梯度驱动的锚点自适应增删；②通过 InfoNCE 对父子节点互信息最大化以及粗细级（Coarse‑to‑Fine）损失提升低层级质量；③利用哈希网格上下文实现可自适应量化和算术编码，显著压缩码率。

**🔧 技术方法**

采用的技术主要有：八叉树层次建模、梯度引导锚点增删、InfoNCE 互信息增强、粗细级优化、哈希网格上下文量化、算术编码、可微分渲染等。

**📊 数据集**

实验使用了 Mip-NeRF360、BungeeNeRF 与 Tanks & Temples 三个真实场景数据集。

**📈 对比分析**

与多种原始（3DGS、Scaffold-GS 等）和锚点压缩方法（CompGS、HAC 等）对比，ProGS 在不同码率下压缩率可达 45 倍，视觉质量提升超过 10%，在低码率下仍能保持约 80% 的 PSNR/SSIM，显著优于 SOTA。

**⚠️ 局限性**

局限性：训练阶段耗时较长且需要大显存；极大场景的锚点数仍较高；对实时自适应编码/解码的细粒度控制尚未完全实现。

---

## 415. Cutting the Cord: System Architecture for Low-Cost, GPU-Accelerated Bimanual Mobile Manipulation

**arXiv ID:** 2603.09051 | [PDF](https://arxiv.org/pdf/2603.09051v1)

**作者:** Artemis Shaw `[一作]`, Nikolaus Correll `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一款低成本、完全无绳的双臂移动机器人平台，集成 NVIDIA Jetson Orin Nano GPU，支持 SLAM 导航、视觉驱动操作与 VR 远程操控。

**💡 创新点**

创新点包括：梯度 3D 打印结构以提升刚度重量比、三路电源架构（Tri-Bus）隔离高冲击电机与计算单元、以及完整的 ROS2+Pinocchio 任务驱动控制堆栈。

**🔧 技术方法**

使用的技术包括：ROS2+Pinocchio/IK、Open3D 视觉、RTAB‑Map SLAM、NVIDIA Jetson Orin Nano 计算、Feetech STS‑3215 电机、RealSense D435 摄像头、Meta Quest 3 VR 远程操控。

**📊 数据集**

使用的数据集主要为自建的 RGB‑D 数据、基于 RTAB‑Map 的 SLAM 地图以及在实验室环境中收集的演示数据；对比模型包括 ACT、Diffusion Policy 与 SmolVLA。

**📈 对比分析**

通过对比三种模型的端到端推理延迟，系统可实现 27.8 Hz 的重新规划速率；Tri‑Bus 设计将电压波动从 12.2 V 降低至 12.0 V，显著提升稳定性。

**⚠️ 局限性**

局限性包括：Jetson Orin Nano 的 CUDA 核心受限导致高频多模 transformer 受限、采用低成本 Hobby 电机导致机械磨损、平台主要适用于平面任务且缺乏垂直线性轴。

---

## 416. A New Modeling to Feature Selection Based on the Fuzzy Rough Set Theory in Normal and Optimistic States on Hybrid Information Systems

**arXiv ID:** 2603.08900 | [PDF](https://arxiv.org/pdf/2603.08900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 417. An accurate flatness measure to estimate the generalization performance of CNN models

**arXiv ID:** 2603.09016 | [PDF](https://arxiv.org/pdf/2603.09016v1)

**作者:** Rahman Taleghani `[一作]` (University of Padova), Francesco Marchetti `[通讯]` (University of Padova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

推导并实现了一种针对使用全局平均池化（GAP）和 1×1 卷积分类器的卷积神经网络的 Hessian 迹的闭式解析公式，并基于此构建了参数化无关的相对平坦度指标。

**💡 创新点**

创新点在于：①给出了卷积层的精确 Hessian 迹表达式，充分利用卷积的共享权重与空间平滑特性；②将该表达式与相对平坦度概念结合，得到对权重尺度不敏感、能直接反映特征空间几何与分类器不确定性的全局度量；③在此基础上证明了该指标与泛化误差的理论上界存在正相关。

**🔧 技术方法**

使用的技术包括：符号计算（推导闭式公式）、自动微分（验证结果）、Hessian 迹估计方法对比（Autograd、Hutchinson、Functorch）、统计回归与相关性分析、在不同优化器/学习率、数据增强、标签噪声等多种训练场景下的实验评估。

**📊 数据集**

主要实验数据集为 ImageNet（预训练 ResNet‑18 进行微调）和 CIFAR‑10（在 ResNet‑18、VGG‑16、DenseNet‑121 等架构上训练），并在不同噪声水平和增广策略下进行测试。

**📈 对比分析**

与传统的 Hessian 迹估计、相对平坦度和其他近似方法相比，闭式指标在精度上完全一致、运行时间更短；在泛化性能上，指标与验证误差呈显著正相关（Pearson r≈0.58，Spearman ρ≈0.76，R²≈0.34），并能在不同网络、优化器、增广和噪声设置下保持一致的预测能力。

**⚠️ 局限性**

局限性包括：仅适用于以 GAP+1×1 卷积结尾的网络结构，无法直接扩展到内部卷积层或非平均池化的架构；指标主要关注分类器端的平坦度，尚未分离特征提取器的鲁棒性与分类器置信度的贡献；对极大规模模型或多任务设置的适用性待进一步验证。

---

## 418. DataFactory: Collaborative Multi-Agent Framework for Advanced Table Question Answering

**arXiv ID:** 2603.09152 | [PDF](https://arxiv.org/pdf/2603.09152v1)

**作者:** Tong Wang `[一作]` (Institute of Systems Engineering, Academy of Military Sciences), Gang Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 DataFactory 多代理框架，用自然语言交互来完成表格问答任务。

**💡 创新点**

创新点在于三方协作：Data Leader、Database Team、Knowledge Graph Team，使用 ReAct 进行动态自然语言协商；自动化将表格映射为知识图谱的函数 𝒯；以及结合 SQL 与 Cypher 的双模查询，突破传统单一 LLM 的限制。

**🔧 技术方法**

主要技术包括：大语言模型（LLM）驱动的 ReAct 框架、检索增强提示（RAG）生成 Text‑to‑SQL 与 Text‑to‑Cypher、自动化数据到知识图谱的转换算法、数据库与图数据库的交互、自然语言解释与可视化。

**📊 数据集**

使用的公开数据集为 TabFact、WikiTableQuestions、FeTaQA。

**📈 对比分析**

与十类基线（DNN、Prompt、Code、Agent、Multi‑Agent）在 TabFact 和 WikiTQ 上对比，提升准确率分别为 20.2% 与 23.9%；在 FeTaQA 上 ROUGE‑1、ROUGE‑2、ROUGE‑L 分别提升至约 84%、72.8% 与 75%；Cohen’s d 超过 1，显示差异显著。虽然 token 消耗略高，但整体性能优于现有方法。

**⚠️ 局限性**

局限性包括：高度依赖 LLM 的推理与生成能力；交互成本与 token 使用较多；知识图谱团队使用频率受任务类型限制；仍受表格规模、上下文长度和模型性能的影响。

---

## 419. OptBench: An Interactive Workbench for AI/ML-SQL Co-Optimization[Extended Demonstration Proposal]

**arXiv ID:** 2603.08880 | [PDF](https://arxiv.org/pdf/2603.08880v1)

**作者:** Jaykumar Tandel `[一作]` (Arizona State University), Jia Zou `[通讯]` (Arizona State University)

**通讯引用:** 736 | [OpenAlex ID](https://openalex.org/A5013735333)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了名为 OptiBench 的交互式工作台，用于构建、调试、评估和比较 SQL+AI/ML 查询优化器，统一使用 DuckDB 作为后端执行环境，提供可扩展的 ML 函数库、重写动作、统计估计以及查询套件。

**💡 创新点**

创新点在于（1）将所有优化器运行在统一后端，确保 apples‑to‑apples 的公平对比；（2）提供可视化、决策轨迹记录的工作台，支持规则型与成本型优化器的快速原型；（3）实现了多模态（表格、文本、图像）SQL+ML 推理查询的统一 benchmark，极大简化了实验重复性。

**🔧 技术方法**

主要技术包括 DuckDB（C++/Python 接口）、自定义 UDF（线性代数、神经网络、树模型等）、可扩展的重写动作框架、统计采样与卡路里估计、Web 前端（React/Vue）用于可视化与交互；还实现了基于 DP 的成本模型搜索和规则驱动的优化器接口。

**📊 数据集**

使用了 10 组查询：来自 Expedia、Flights、CreditCard 的业务场景；TPCx‑AI 的 8 个业务用例（聚类、预测、文本分类等）；以及 IDNet 的 2 个图像推理查询（CNN、LLM）。

**📈 对比分析**

比较方法是在同一硬件/软件环境下，使用同一 DuckDB 后端执行所有注册优化器，记录决策轨迹、执行计划与延迟；实验表明，规则型优化器可将某些查询从 85 秒压缩到 2 秒；成本型优化器在给定深度预算内进一步提升性能，且两者的计划差异可通过侧边可视化直观呈现。

**⚠️ 局限性**

局限性包括：仅在单机 DuckDB 环境下评测，未覆盖分布式场景；重写动作与成本模型仍需人工定义和调优；目前重点关注端到端延迟，未对资源利用率、能耗或可扩展性进行深入评估。

---

## 420. Optimizing Reinforcement Learning Training over Digital Twin Enabled Multi-fidelity Networks

**arXiv ID:** 2603.08931 | [PDF](https://arxiv.org/pdf/2603.08931v1)

**作者:** Hanzhi Yu `[一作]` (University of Miami), Mingzhe Chen `[通讯]` (University of Miami)

**通讯引用:** 18813 | [OpenAlex ID](https://openalex.org/A5072241033)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于数字网络孪生（DNT）的分层强化学习框架，用以在基站天线倾斜角度调节和数据采集比例之间进行联合优化，从而提升物理网络的数据传输速率。

**💡 创新点**

创新点包括：①将鲁棒对抗损失与PPO相结合，构建鲁棒RL层以抵御DNT噪声；②采用第二层PPO动态调节物理网络与DNT数据的采集比例，实现两层时间尺度上的最优控制；③在一次训练周期内同时优化倾斜角度策略与采集策略，显著降低物理网络数据采集延迟。

**🔧 技术方法**

主要技术：强化学习（PPO、鲁棒RL）、对抗学习、数字网络孪生、深度神经网络（策略网络与价值网络）、随机游走用户移动模型、信道模型与SINR计算。

**📊 数据集**

使用仿真数据：基站位于(0,0)，覆盖半径50m，3个方向天线，10个随机分布用户，DNT数据误差服从[-ε,ε]均匀分布，实验参数按表所列（如P0=1W，β=2，N0=1e-6W，等）。

**📈 对比分析**

与两种基线对比：①鲁棒RL+随机采集比例；②Vanilla PPO+Vanilla PPO。实验结果显示：在收敛后，本文方法将物理网络数据采集延迟降低约28%，第一层鲁棒RL的平均回报提升约38%，第二层PPO的平均回报提升约78%，整体性能比基线提升70%+。

**⚠️ 局限性**

局限性：①需要预先知道网络参数（天线位置、功率、损耗指数等），实际部署时需估计；②仅在仿真场景下验证，真实网络中的DNT误差分布和通信延迟可能更复杂；③双层RL训练成本较高，规模化部署时计算与收敛时间可能成为瓶颈；④对极端误差（ε较大）时鲁棒性尚未充分验证。

---

## 421. Declarative Scenario-based Testing with RoadLogic

**arXiv ID:** 2603.09455 | [PDF](https://arxiv.org/pdf/2603.09455v1)

**作者:** Ezio Bartocci `[一作]` (TU Wien), Dejan Ničković `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一个开源框架，将OpenSCENARIO DSL（OS2）的抽象场景描述自动转化为可执行的仿真场景，并在仿真过程中实时监控符合性。

**💡 创新点**

创新点在于将声明式OS2规范与答案集编程（ASP）和运动规划相结合，形成端到端的从高层抽象到低层可执行的完整流水线，并通过符号自动机实现规范监控。

**🔧 技术方法**

主要技术包括ANTLR4解析OS2、ASP求解器（Potassco）生成高层计划、Frenetix运动规划器在CommonRoad框架中实现轨迹化、以及基于符号自动机的运行时监控。

**📊 数据集**

使用的场景数据集为七种典型OS2场景（如跟随、超车、多车超车、变道等），在CommonRoad仿真环境下生成和评估。

**📈 对比分析**

实验结果表明，框架能在几分钟内生成满足规范的多样化仿真；相较于纯手工或仅基于XML的方案，生成效率更高，且通过参数采样显著提升了场景多样性。

**⚠️ 局限性**

局限性包括仅支持OS2的子集、ASP求解在复杂场景中可能导致搜索偏置或时间过长、以及对多车大规模场景的可扩展性尚未充分验证。

---

## 422. UniField: A Unified Field-Aware MRI Enhancement Framework

**arXiv ID:** 2603.09223 | [PDF](https://arxiv.org/pdf/2603.09223v1)

**作者:** Yiyang Lin `[一作]` (Chinese University of Hong Kong), Yixuan Yuan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9999 | [OpenAlex ID](https://openalex.org/A5073968803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建统一的 MRI 场强提升框架 UniField，实现多模态（T1、T2、FLAIR）和多任务（低场到高场、3T到7T）联合训练，并发布了最大规模的多场 MRI 数据集

**💡 创新点**

三大创新：①统一模型共享降解模式；②利用 3D 视频超分预训练模型（FlashVSR）作为结构先验；③提出基于物理场强的频域校正机制 FASRM，缓解谱偏差

**🔧 技术方法**

技术核心包括：基于流匹配的 ODE 生成器、低秩适配 LoRA + 稀疏注意力、Field‑Aware Spectral Rectification Mechanism (FASRM) 以及 3D 频域损失函数

**📊 数据集**

使用从五家机构收集的、经精细配准的多场 MRI 数据集，涵盖 64mT→3T 与 3T→7T，包含 T1、T2、FLAIR 三种序列，共计数千对图像

**📈 对比分析**

与 MO‑U‑NET、MSFA、LowGAN、FlashVSR 等基线对比，UniField 在 PSNR、SSIM、NRMSE、LPIPS 上均取得最优或接近最优，平均 PSNR 提升约 1.8 dB、SSIM 提升 9.5%

**⚠️ 局限性**

局限性包括：仅验证于脑部序列；对极端高场（>7 T）和疾病样本的泛化仍待评估；模型推理速度受 3D ODE 求解影响

---

## 423. SPAN-Nav: Generalized Spatial Awareness for Versatile Vision-Language Navigation

**arXiv ID:** 2603.09163 | [PDF](https://arxiv.org/pdf/2603.09163v1)

**作者:** Jiahang Liu `[一作]` (Peking University), He Wang `[通讯]` (Peking University)

**通讯引用:** 11654 | [OpenAlex ID](https://openalex.org/A5100351651)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种端到端的 SPAN-Nav 基础模型，利用 RGB 视频实现通用 3D 空间感知，从而提升视觉语言导航任务的路径规划与执行。

**💡 创新点**

创新点在于：①将连续 3D 占据体映射压缩为单一空间 token，显著降低计算开销；②引入空间 Chain‑of‑Thought 机制，将空间信息显式嵌入决策链条；③通过跨场景占据体预测和多任务共训练，构建跨域通用空间先验；④构建规模达 4.2M 的室内外占据体标注数据集。

**🔧 技术方法**

采用 Qwen3‑VL 视觉‑语言模型为主干，结合 VQ‑VAE 预训练的占据体编码器/解码器，使用连续占据体潜在嵌入进行空间 token 投影；随后通过空间 CoT 与动作生成网络实现空间驱动的轨迹预测；训练中使用两阶段共训练策略，并联合 QA 任务提升语言理解。

**📊 数据集**

使用包括 7.08M 条目在内的大规模数据集：5.08M 轨迹、0.93M 图像问答、1.07M 视频问答，以及 4.2M 由室内/室外真实与仿真环境构成的占据体标注数据；涵盖 VLN‑CE R2R、RxR、MetaUrban、InternScenes 等基准。

**📈 对比分析**

与现有 SOTA 在 VLN‑RxR、MetaUrban、InternScenes 等基准对比，SPAN‑Nav 在 RxR 上成功率提升 5.3%，MetaUrban 累计成本下降 4 倍，InternScenes 家居场景成功率提升 30.9%；消融实验表明单个空间 token 已能维持高性能，并验证空间 CoT 对动作决策的显著正面影响。

**⚠️ 局限性**

局限性：①需要大量占据体标注数据，标注成本高；②在极度遮挡或快速动态环境下占据体预测精度可能下降；③模型当前以特定机器人姿态与传感器配置训练，跨机器人或跨传感器迁移能力仍需进一步验证。

---

## 424. Proprioceptive Safe Active Navigation and Exploration for Planetary Environments

**arXiv ID:** 2603.08905 | [PDF](https://arxiv.org/pdf/2603.08905v1)

**作者:** Matthew Y. Jiang `[一作]` (Georgia Institute of Technology), Shipeng Liu `[通讯]` (University of Southern California)

**通讯引用:** 405 | [OpenAlex ID](https://openalex.org/A5102980480)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出一种基于自我感知的安全主动导航与探索框架PSANE，利用腿-地面交互测得的滑移比构建高斯过程模型，实时估计并安全认证可行区域，采用前沿检测与多目标子目标规划，并用差分反应式控制器实现安全路径执行。

**💡 创新点**

创新点在于：①将不确定性感知的GP滑移比与安全集合认证结合，形成连续空间的安全集；②提出前沿基多目标子目标选择（Pareto + 标量化）实现目标进展与安全扩展双重权衡；③将此规划与实时差分导航控制器集成，完成从感知到规划到控制的闭环安全导航。

**🔧 技术方法**

使用技术包括：高斯过程回归（RBF核）进行滑移比建模；置信区间安全集认证与Lipschitz传播；前沿检测（Suzuki边界）与Pareto最优与标量化选择；指数衰减目标启发式；差分（diffeomorphic）反应式导航控制器；Chrono仿真与SCM地形模型。

**📊 数据集**

采用实测月球模拟物质LHS-1的土壤力学参数生成Chrono中的可变形地形模型，用于仿真验证。

**📈 对比分析**

与Naive Goal Heuristic (NGH)、Safe-Goal Heuristic (SGH) 和 Pareto-Goal Heuristic (PGH) 进行对比；在两种环境下，PSANE在成功率、完成时间和路径长度上显著优于基线，目标导航成功率均达到100%，完成时间约为PGH的一半，路径长度缩短近一半；在无目标探索任务中，覆盖率也高于PGH。

**⚠️ 局限性**

局限性包括：依赖Lipschitz常数与GP模型假设，计算量随地图分辨率增大；在极端复杂地形中前沿选择仍可能受限；实验仅在仿真环境验证，真实地形中腿-地面交互噪声与模型误差可能影响性能；未涉及多机器人协同或更长期决策问题。

---

## 425. LCA: Local Classifier Alignment for Continual Learning

**arXiv ID:** 2603.09888 | [PDF](https://arxiv.org/pdf/2603.09888v1)

**作者:** Tung Tran `[一作]` (Kyushu University), Khoat Than `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5054854013)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Local Classifier Alignment (LCA) 损失，解决CIL中基准与分类器不匹配问题，并结合增量模型合并实现无回放的连续学习。

**💡 创新点**

LCA损失提供理论误差上界，利用高斯原型对齐分类器，显著提升鲁棒性与泛化；同时提出仅合并PEFT参数的增量合并方法。

**🔧 技术方法**

使用预训练ViT backbone、LoRA参数高效微调、Gaussian原型、模型合并与LCA损失。

**📊 数据集**

在CIFAR‑100、ImageNet‑R、ImageNet‑A、CUB‑200、OmniBenchmark、VTAB、StanfordCars等七个基准上评估。

**📈 对比分析**

与CODA‑Prompt、DualPrompt、L2P、EASE、MOS、SLCA、APER等方法比较，IM+LCA在多数数据集上取得领先或接近最优的平均准确率，整体提升约2%，并在鲁棒性基准上有显著增益。

**⚠️ 局限性**

未在其它任务/场景验证LCA，理论未覆盖整体backbone训练，缺乏端到端训练整合，未来需进一步扩展。

---

## 426. Hierarchical Observe-Orient-Decide-Act Enabled UAV Swarms in Uncertain Environments: Frameworks, Potentials, and Challenges

**arXiv ID:** 2603.09191 | [PDF](https://arxiv.org/pdf/2603.09191v1)

**作者:** Ziye Jia `[一作]` (Nanjing University of Aeronautics and Astronautics), Zhu Han `[通讯]` (University of Houston)

**通讯引用:** 89279 | [OpenAlex ID](https://openalex.org/A5063667378)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个层次化的Observe-Orient-Decide-Act（H‑OODA）框架，并将其与网络功能虚拟化（NFV）结合，用于在不确定环境中实现无人机群的协同决策与行动。

**💡 创新点**

创新点在于将经典OODA循环嵌入云‑边缘‑终端三层架构，形成嵌套的H‑OODA循环，并利用NFV实现可动态部署的网络功能链，提升了决策的灵活性、可扩展性与实时性。

**🔧 技术方法**

采用了云‑边缘‑终端（CET）架构、NFV、SDN、服务功能链（SFC）、深度强化学习（D3QN）等技术来实现感知、协同与动作控制。

**📊 数据集**

使用的是仿真生成的目标搜索数据，实验在 100×100 m 区域内部署 15 台 UAV 进行 10 000 次重复，以评估搜索效率、成功率、QoE 与错误率。

**📈 对比分析**

通过与单层 OODA、边缘‑终端 OODA 等基准比较，实验显示 H‑OODA 在搜索效率、成功率、QoE 上更高、错误率更低；循环深度越大性能越好。

**⚠️ 局限性**

局限性包括：数据处理复杂度高、通信可靠性与带宽受限、自治与人工监督的平衡难题，以及安全与韧性面临的网络攻击与干扰风险。

---

## 427. A Text-Native Interface for Generative Video Authoring

**arXiv ID:** 2603.09072 | [PDF](https://arxiv.org/pdf/2603.09072v1)

**作者:** Xingyu Bruce Liu `[一作]` (Adobe Research), Dingzeyu Li `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款名为 Doki 的文本原生视频创作界面，用户可在单一文档中定义资产、编写剧情、生成并编辑镜头以及添加音频，整个制作流程均通过自然语言文本完成。

**💡 创新点**

创新点包括：① 将视频创作统一为单一文本结构（文档＝视频，段落＝序列，句子＝镜头）；② 通过可参数化的 @mention 与 #hashtag 定义实现跨镜头一致性；③ 极简交互（斜杠菜单、内嵌预览）与 AI 辅助（侧边会话式代理与内联代理）并行；④ 将文本作为人机交互的“公共语言”，实现对生成过程的可视化与可编辑控制。

**🔧 技术方法**

技术栈：前端 React+TypeScript+TipTap+Zustand；后端 Node.js；视频处理 FFmpeg；生成模型（文本到图像使用 Imagen 4 或类似模型；文本到视频使用 Veo 3/3‑fast 或类似模型）；LLM（Gemini 2.5 Flash、GPT‑4 等）用于提示重写、参考图像检索和 AI 代理；文档结构解析与参数化传播逻辑由自研的树形语法树实现。

**📊 数据集**

未采用自建训练数据；系统直接调用现有商业/研究级文本‑图像/视频生成模型；评估使用 10 名参与者生成的 46 条视频及其交互日志与自评问卷（无公开数据集）。

**📈 对比分析**

通过为期一周的日记研究（10 名参与者、5 天）进行定性/定量评估：System Usability Scale 平均 81.2（优秀区间），平均每会话生成 45.5 张图、20.3 条视频；用户认为创作效率提升、叙事结构更清晰。相较传统多窗体工作流，主观满意度显著提升；但未提供客观视频质量指标或与同类工具的直接对比。

**⚠️ 局限性**

局限性：① 生成模型的可预测性差，需多次重生成；② 对精细视觉控制（构图、镜头运动、音频同步）支持不足；③ 文本线性结构难以表达并行/交叉剪辑等时间性复杂性；④ 片段长度受限，需多片段拼接；⑤ 依赖模型的成本与时延；⑥ 对专业级剪辑需求（色彩分级、精准剪辑）尚不够成熟。

---

## 428. StyleVLA: Driving Style-Aware Vision Language Action Model for Autonomous Driving

**arXiv ID:** 2603.09482 | [PDF](https://arxiv.org/pdf/2603.09482v1)

**作者:** Yuan Gao `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**通讯引用:** 1993 | [OpenAlex ID](https://openalex.org/A5063677428)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 StyleVLA 数据集与基于 Qwen3‑VL‑4B 的物理信息化 VLM 框架，用于生成符合用户指定驾驶风格的可行轨迹。

**💡 创新点**

创新点在于结合多风格运动规划、物理一致性损失与持续回归头，显著提升轨迹的物理可行性和多样性。

**🔧 技术方法**

采用多模态指令学习、QLoRA 微调、物理一致性（PIKC）损失、跨模态指令格式与贝叶斯融合的权重自适应混合损失。

**📊 数据集**

使用 1.2k 场景、76k BEV 与 42k FPV 样本的 StyleVLA 指令数据集，包含 Default、Balanced、Comfort、Sporty、Safety 五种驾驶风格。

**📈 对比分析**

与 Gemini‑3‑Pro 等闭源模型及多款开源 VLM（Qwen2.5‑VL‑7B、InternVL3‑9B 等）进行零样本与微调对比，StyleVLA 在 BEV/FPV 上分别达到 0.55/0.51 的综合评分，成功率提升至约 39%/38%，并将推理时间压缩至 1.92–2.13 秒。

**⚠️ 局限性**

局限在于仍依赖仿真数据，风格类别有限；对真实环境的迁移性能未充分验证，且高频运动规划仍需进一步加速。

---

## 429. Dynamic Multi-period Experts for Online Time Series Forecasting

**arXiv ID:** 2603.09062 | [PDF](https://arxiv.org/pdf/2603.09062v1)

**作者:** Seungha Hong `[一作]` (Pohang University of Science and Technology), Hwanjo Yu `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 5028 | [OpenAlex ID](https://openalex.org/A5045521125)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在线时间序列预测，提出DynaME框架以动态应对概念漂移。

**💡 创新点**

将概念漂移细分为Recurring Drift和Emergent Drift，并通过动态专家委员会与门控网络分别针对这两类漂移进行自适应。

**🔧 技术方法**

采用FFT动态周期选择、非参数专家回归（对偶Ridge）、动态门控网络与危险信号机制，配合Transformer/MLP后端。

**📊 数据集**

在ETT、Traffic、ECL、Weather等公开基准数据集上进行实验。

**📈 对比分析**

与GD、SOLID、DSOF、PROCEED等在线方法在多种后端与预测时隙下比较，DynaME在所有设置中均取得最低MSE，成为新的SOTA。

**⚠️ 局限性**

需要手动调节专家数k与样本数n，对极端频率变化仍有限制；在极大规模流中仍有一定计算与内存开销。

---

## 430. Beyond Scaling: Assessing Strategic Reasoning and Rapid Decision-Making Capability of LLMs in Zero-sum Environments

**arXiv ID:** 2603.09337 | [PDF](https://arxiv.org/pdf/2603.09337v1)

**作者:** Yang Li `[一作]` (Tsinghua University), Yao Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 5774 | [OpenAlex ID](https://openalex.org/A5034221181)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了STAR基准，用于评估LLM在零和多代理动态环境中的战略推理与实时决策能力。

**💡 创新点**

将推理视为迭代对抗决策过程，设计四层模块化框架与PWER度量，揭示推理与执行之间的差距。

**🔧 技术方法**

基于ECS引擎、WebSocket协议、Chain-of-Thought提示与多模态输入，结合PWER、ELO评分等评估指标。

**📊 数据集**

构建了自研的RoTK战术地图和多地图场景，使用1v1零和游戏任务进行评估。

**📈 对比分析**

通过全对战评估，使用Win Rate、SER与PWER进行比较，发现推理强化模型在轮盘模式领先，但实时模式因推理延迟表现被削弱。

**⚠️ 局限性**

缺乏对多玩家/协作场景的支持，推理与执行分离导致策略-执行差距，评估受限于单一地图设计。

---

## 431. Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges

**arXiv ID:** 2603.09086 | [PDF](https://arxiv.org/pdf/2603.09086v1)

**作者:** Rongxiang Zeng `[一作]` (RWTH Aachen University), Yongqi Dong `[通讯]` (Delft University of Technology)

**通讯引用:** 2553 | [OpenAlex ID](https://openalex.org/A5019174861)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套统一的潜在空间框架，对自动驾驶中的世界模型进行系统性分类、内部机制分析、评估标准制定，并给出未来研究方向。

**💡 创新点**

创新点主要包括：① 通过目标、形式和结构先验构建的潜在空间分类体系；② 对5大内部机制（结构同构、长时序稳定、语义对齐、价值对齐、适应性计算）的系统梳理；③ 设计了闭环安全差距(CSG)、时序一致性(TCS)和推理成本(DC)三种统一评估指标，弥补了开放环与闭环评价的差距；④ 针对长时序漂移、实时推理和仿真到现实的关键挑战给出了可操作的改进路线。

**🔧 技术方法**

采用的技术包括：Diffusion/Transformer/VAEs/Normalizing Flow等生成式潜在模型；BEV网格、稀疏卷积、射线投射等几何先验；强化学习、VLM/VLA、Chain‑of‑Thought 机制；多尺度蒸馏、跨模态对齐、基于置信度的自适应推理等。

**📊 数据集**

使用的数据集和仿真平台有：nuScenes、Waymo Open Dataset、DriveX、DriveWorld、Drive‑OccWorld 等公开真实驾驶日志；CARLA、LGSVL、AirSim、MetaDrive、Waymax 等交互式仿真器；NAVSIM 等基于真实日志的神经仿真平台。

**📈 对比分析**

在开放环评估中，方法在nuScenes上表现为 ADE/FID 等指标优于传统模型；在闭环评估（CARLA 等）中则使用成功率、碰撞率、规则违规率等指标。实验表明，虽然多种方法在开放环上能取得较高精度，但在闭环任务中的安全性、鲁棒性仍有显著差距；部分方法（如 DriveLaW、Raw2Drive、WorldRFT）通过价值对齐和预训练提升了闭环成功率，且多方法在资源消耗（延迟、内存、能耗）上各有优势。

**⚠️ 局限性**

主要限制包括：① 长时序生成容易产生漂移/幻觉，导致闭环失稳；② 大型 Diffusion/Transformer 推理成本高，难以满足车载实时/功耗约束；③ 仿真与真实场景之间仍存在显著域差距，导致迁移性能下降；④ 语义/因果对齐机制在解释性与可靠性上尚不成熟；⑤ 稀有安全事件的覆盖不足，缺乏系统化的罕见事件评估框架。

---

## 432. ALADIN: Accuracy-Latency-Aware Design-space Inference Analysis for Embedded AI Accelerators

**arXiv ID:** 2603.08722 | [PDF](https://arxiv.org/pdf/2603.08722v1)

**作者:** T. Baldi `[一作]`, A. Biondi `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ALADIN 框架，实现对混合精度量化神经网络在嵌入式 AI 加速器上的精度-延迟-资源设计空间的预测与分析，支持无需在硬件上部署即可评估推理瓶颈与权衡。

**💡 创新点**

创新点在于通过逐步精炼模型，将 QONNX 转化为实现感知模型，再到平台感知模型，结合平台参数即可在软件层面预测推理时延、内存占用和准确率；同时验证了混合精度与实现策略对实时系统性能的细粒度影响。

**🔧 技术方法**

使用 QONNX、实现配置文件、基于 Dory 的 C 代码生成、GVSoC 循环精确模拟、LUT 与 im2col 的实现细节、硬件调度与双缓冲。

**📊 数据集**

MobileNetV1 在 CIFAR‑10 数据集上进行量化训练与评估。

**📈 对比分析**

通过 ALADIN 与 GVSoC 结果对比，展示了不同混合精度与实现（im2col、LUT、阈值树、dyadic scaling）在 MAC、BOP、内存占用、时钟周期等指标上的差异；在 GAP8 平台上验证了时延预测的准确性，揭示了低精度 im2col 与 LUT 在并行度与内存访问方面的权衡。

**⚠️ 局限性**

局限在于调度策略仍依赖 Dory，未提供最优调度算法；对多核共享 LUT 的并行冲突未完全建模；对更复杂的算子（如注意力、动态形状）支持有限；实验仅针对单一模型和平台，泛化性待验证。

---

## 433. ENIGMA-360: An Ego-Exo Dataset for Human Behavior Understanding in Industrial Scenarios

**arXiv ID:** 2603.09741 | [PDF](https://arxiv.org/pdf/2603.09741v1)

**作者:** Francesco Ragusa `[一作]` (University of Catania), Giovanni Maria Farinella `[通讯]` (Next Vision s.r.l. - Spinoff of the University of Catania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在真实工业实验室中采集了同步的前视（ego）和后视（exo）视频，构建了 ENIGMA-360 数据集，并对其进行了时序关键步骤、交互关键帧和手/物体的空间注释；随后对三类任务（时序动作分割、关键步骤识别、前视手-物体交互检测）进行了基线实验。

**💡 创新点**

创新点主要有：①首次提供同步 ego–exo 的工业场景视频；②结合真实工业工具与流程的细粒度时序与空间标注；③发布 3D 模型、DINOv2 预训练特征和 SAM-HQ 分割掩码等补充资源，方便多模态与跨域研究。

**🔧 技术方法**

采用的技术包括：时序动作分割基线（C2F‑TCN、MSTCN++、LTContext、ASFormer、DiffAct、FACT）；关键步骤识别基线（TimeSformer）；前视手-物体交互检测基线（VISOR HOS、Label Propagation 版 VISOR HOS、两阶段基于 Bounding‑Box 的检测）。此外，使用了 SAM‑HQ 生成手/物体分割掩码、DINOv2 提取帧级特征、Matterport 与 ARTEC EVA 采集的 3D 资产。

**📊 数据集**

使用的数据集是 ENIGMA‑360（共 360 条视频，180 条 ego 与 180 条 exo，约 111.5 小时，180 对同步录制），并在此数据集上进行训练与测试。

**📈 对比分析**

与现有方法相比，所有基线在同视角下已能取得一定分数，但在跨视角（ego→exo 或 exo→ego）时性能急剧下降，往往低于 30% 的 F1/编辑分数；关键步骤识别在 ego 视角约 0.75 的 F1，exo 仅 0.46；手‑物体交互检测的整体 AP 在 64% 左右，bbox 方案仅 46%。这些结果说明现有模型在真实工业域的泛化能力有限。

**⚠️ 局限性**

主要限制：①实验室环境固定，工具与流程单一，缺乏多样化；②参与者数量有限且任务种类仅为两种维护流程，行为多样性不足；③数据集虽然丰富但仍面临隐私与安全约束，难以扩展至更广泛工业场景；④跨视角差距大，表明需要更强的多模态融合与域适应技术。

---

## 434. DRIFT: Dual-Representation Inter-Fusion Transformer for Automated Driving Perception with 4D Radar Point Clouds

**arXiv ID:** 2603.09695 | [PDF](https://arxiv.org/pdf/2603.09695v1)

**作者:** Siqi Pei `[一作]` (Delft University of Technology), Dariu M. Gavrila `[通讯]` (Delft University of Technology)

**通讯引用:** 12504 | [OpenAlex ID](https://openalex.org/A5085298812)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了DRIFT框架，采用点与柱两条并行路径，并在多阶段实现双向特征共享，以高效处理4D雷达点云；

**💡 创新点**

创新点在于双路径并行架构与跨阶段双向特征共享块、Transformer在点/柱路径中的使用，以及面向雷达稀疏的稀疏实现；

**🔧 技术方法**

使用Transformer块、稀疏卷积、点与柱表示、跨注意力特征共享、MMDetection3D实现等技术；

**📊 数据集**

实验数据集为公开的View‑of‑Delft雷达数据集和内部的perciv‑scenes‑2数据集；

**📈 对比分析**

与CenterPoint、PointPillars等基线对比，DRIFT在VoD全景mAP达52.6%（预训练53.1%），在驾驶通道及小物体检测上显著优于现有方法；

**⚠️ 局限性**

对大规模训练数据依赖度高，雷达点云稀疏噪声敏感，Transformer计算量大且在稀疏场景下效率受限。

---

## 435. The Radio-Frequency Transformer for Signal Separation

**arXiv ID:** 2603.09201 | [PDF](https://arxiv.org/pdf/2603.09201v1)

**作者:** Egor Lifar `[一作]` (Massachusetts Institute of Technology), Gregory W. Wornell `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 33664 | [OpenAlex ID](https://openalex.org/A5066172831)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了一种基于自回归Transformer和离散tokenizer的单通道信号分离框架，用于从非高斯背景噪声中提取SOI（如QPSK）。

**💡 创新点**

创新点包括将SoundStream改造成RF信号tokenizer，采用有限标量量化（FSQ）与Transformer块实现低比特率离散化；使用交叉熵训练而非传统MSE，使模型直接优化最终的离散指标；以及展示了模型在未见混合环境下的零样本泛化能力。

**🔧 技术方法**

使用技术包括：Transformer编码器‑解码器架构、FSQ离散化、旋转位置编码、交叉熵损失、MSE预训练tokenizer，以及对齐/自回归解码流程。

**📊 数据集**

使用MIT RF Challenge数据集中的四种干扰信号（EMISignal、CommSignal2、CommSignal3、CommSignal5G）进行训练与评估。

**📈 对比分析**

与WaveNet、TUB、KU‑TII等基线相比，在多种干扰场景下实现了平均BER下降122倍、MSE显著降低，并在多干扰模型和零样本Gaussian噪声环境中保持竞争性甚至优于匹配滤波。

**⚠️ 局限性**

局限性包括：在5G干扰下多干扰模型表现略逊于专用模型；对极端高SIR/低INR的泛化仍需验证；训练时需要较大窗口和重叠，推理时计算量相对较高；以及未涉及多通道或多模态场景。

---

## 436. Hebbian-Oscillatory Co-Learning

**arXiv ID:** 2603.08731 | [PDF](https://arxiv.org/pdf/2603.08731v1)

**作者:** Hasi Hays `[一作]` `[通讯]` (University of Arkansas), Hasi Hays (University of Arkansas)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并验证了 Hebbian‑Oscillatory Co‑Learning (HOC‑L) 框架，将超曲率稀疏网络与基于 Kuramoto 的相位同步注意力耦合，并在同步阈值下实现 Hebbian 结构塑性。

**💡 创新点**

引入同步门控 Hebbian 机制、两时间尺度动力学、证明收敛与 Lyapunov 稳定性，并实现 O(n·k) 线性复杂度。

**🔧 技术方法**

Poincaré 球嵌入、超曲率稀疏图构造、Kuramoto 余弦相位同步、相位锁定注意力、可微同步门控函数、两时间尺度随机逼近与 Lyapunov 理论。

**📊 数据集**

合成振荡器网络、分子图分类（MUTAG、PTC、PROTEINS、IMDB‑B）、Long Range Arena 序列、神经形态模式识别等基准。

**📈 对比分析**

与标准 Transformer、GAT、GCN、RSGN、SSA、稀疏演化训练和 Lottery Ticket 子网络对比，实验显示 HOC‑L 在保持稀疏性的同时提升同步与结构收敛，效率优于全连接注意力。

**⚠️ 局限性**

理论假设光滑门控和中心化窄频分布，全球同步门控粗糙；频率多模或宽分布、长时间无同步导致权重衰减、未在大规模任务上充分验证，硬件实现仍需进一步研究。

---

## 437. DendroNN: Dendrocentric Neural Networks for Energy-Efficient Classification of Event-Based Data

**arXiv ID:** 2603.09274 | [PDF](https://arxiv.org/pdf/2603.09274v1)

**作者:** Jann Krausse `[一作]`, Jürgen Becker `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出并评估了一种基于抑制期（refractory period）的序列模型训练策略，以期在保持高精度的同时降低训练资源消耗。

**💡 创新点**

创新点在于将抑制期机制与并行序列训练相结合，形成一种新的训练范式；该范式能显著减少梯度计算次数，提升训练效率。

**🔧 技术方法**

主要技术包括：并行序列训练框架、抑制期机制（在前向传播中暂时冻结部分权重）、梯度下降优化以及对实验结果的可视化分析。

**📊 数据集**

实验采用 MNIST 图像分类数据集，使用标准的测试集进行性能评估。

**📈 对比分析**

与传统基线模型（Baseline）和无并行序列模型（No Parallel Seq.）比较，抑制期模型在相同参数量下可达到与基线相近的准确率（约 99.26%），且训练时间比无并行序列模型缩短约 30%。

**⚠️ 局限性**

实验范围仅限于 MNIST 数据集和较小规模的网络，未检验该方法在更大规模或更复杂任务（如 ImageNet、自然语言处理）上的可迁移性和鲁棒性。

---

## 438. Differentially Private Secure Multiplication: Beyond Two Multiplicands

**arXiv ID:** 2603.08944 | [PDF](https://arxiv.org/pdf/2603.08944v1)

**作者:** Haoyang Hu `[一作]` (Georgia Institute of Technology), Viveck R. Cadambe `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 9145 | [OpenAlex ID](https://openalex.org/A5008627075)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在有协同攻击（最多T个节点）下，使用ε‑差分隐私完成M个私有输入的乘积计算，提出了基于编码多项式和分层噪声注入的单轮安全乘法框架，并给出了两种不同节点规模下的精度与隐私权衡。

**💡 创新点**

创新点在于：①将之前仅适用于两乘子乘法的DP‑安全乘法扩展到任意数量乘子；②设计了一套通用的噪声协相关机制，使得在（M−1)T+1 ≤ N ≤ MT 以及 N = T+1 两个关键规模区间内都能实现最优或近似最优的隐私‑精度折衷；③通过几何直观解释和编码理论（GRS码）实现了高阶噪声项的系统消除。

**🔧 技术方法**

核心技术包括：差分隐私噪声的最优化（阶梯机制最小方差），编码多项式（Shamir/GRS式）与噪声多层嵌入，线性MMSE估计与多维张量运算的逆推，以及信息理论的下界证明。

**📊 数据集**

本文为理论分析，不使用任何公开数据集；所有结论均基于随机变量的分布假设和方差上限。

**📈 对比分析**

通过与两种基线方法（复数Shamir秘密分享+DP分析和独立噪声方案）进行理论比较，实验图表显示本文方案在LMSE（均方误差）上普遍优于基线，尤其在大隐私预算（ε→0）时接近下界。

**⚠️ 局限性**

主要局限包括：在 N = T+1、N < M 的情形下仍存在上界与下界的差距；结果依赖于无穷大样本/噪声收敛假设；并且仅考虑连续实数输入，实际实现中需解决量化与数值稳定性问题。

---

## 439. EDMFormer: Genre-Specific Self-Supervised Learning for Music Structure Segmentation

**arXiv ID:** 2603.08759 | [PDF](https://arxiv.org/pdf/2603.08759v1)

**作者:** Sahal Sajeer `[一作]` (University of Waterloo), Joel Song Bae `[通讯]` (University of Waterloo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了 EDM 音乐结构分割，提出 EDMFormer 模型并发布了 98 首手工标注的 EDM-98 数据集。

**💡 创新点**

创新点在于结合 EDM 专属的能量驱动结构分类法与自监督音频嵌入，专门针对 EDM 的节奏与动态特征进行模型微调。

**🔧 技术方法**

使用 Transformer（SongFormer）架构，融合 MuQ 与 MusicFM 的自监督嵌入，并在 EDM-98 上进行领域微调。

**📊 数据集**

使用了 98 首专业标注的 EDM-98 数据集，包含 intro、build-up、drop、breakdown、outro 等标签。

**📈 对比分析**

通过零转移基线与 SongFormer（pop 结构）对比，HR@0.5 从 0.569 提升至 0.616、HR@3 从 0.608 提升至 0.635、ACC 从 0.148 提升至 0.883，性能显著提升。

**⚠️ 局限性**

局限在于数据集规模有限、标注仅来自单一评注者、以及基础模型未针对 EDM 结构预训练。

---

## 440. One-Eval: An Agentic System for Automated and Traceable LLM Evaluation

**arXiv ID:** 2603.09821 | [PDF](https://arxiv.org/pdf/2603.09821v1)

**作者:** Chengyu Shen `[一作]` (Peking University), Wentao Zhang `[通讯]` (Zhongguancun Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套名为One-Eval的智能化评估框架，能将自然语言评估请求自动转化为可执行、可追踪的评估工作流，涵盖基准选择、数据获取、配置校验和任务导向的指标报告；

**💡 创新点**

创新点在于将模型评估完全自动化为端到端的Agent流程，利用NL2Bench解析意图并推荐基准，BenchResolve实现自动化基准解析与数据归一化，Metrics & Reporting提供决策导向的多层次报告，并集成人机回执点；

**🔧 技术方法**

技术包括：大型语言模型驱动的意图解析与基准检索（embedding/TF‑IDF+HuggingFace搜索）、分层基准解析与统一配置、自动化数据下载与schema归一化、知识增强的指标推荐、并行执行引擎以及层级化诊断报告；

**📊 数据集**

使用了约77个经过整理的公开评估基准（如GSM8K、MATH‑500、TruthfulQA等）以及HuggingFace Hub上的长尾基准，并在100条自然语言评估请求上进行实验；

**📈 对比分析**

在与lm‑eval‑harness、OpenCompass、HELM等框架的功能对比中，One-Eval在“自动化”和“指标推荐”上领先；在100条评估请求的端到端成功率分别为99%、85%和84%，平均每条请求耗时约11分钟，表明高可靠性与实用性；

**⚠️ 局限性**

局限性包括：对极度模糊或新颖查询的意图解析仍可能失败；自动化schema推断错误率仍在15%；依赖外部大模型与HuggingFace服务，可能在大规模部署时受限；当前仅支持文本类基准，跨模态评估待扩展；

---

## 441. Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation

**arXiv ID:** 2603.09506 | [PDF](https://arxiv.org/pdf/2603.09506v1)

**作者:** Won Shik Jang `[一作]` (Gwangju Institute of Science and Technology), Ue-Hwan Kim `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 510 | [OpenAlex ID](https://openalex.org/A5031220093)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在无训练的情境下，通过将长句子描述映射到值地图进行探索，并在发现候选实例时使用视角感知的3D空间关系验证，完成对目标实例的导航。

**💡 创新点**

创新点在于①把全文描述作为探索先验而非仅用于后期验证；②利用基于视角的3D空间关系检查来判定实例正确性；③不需要任务专属训练，能直接迁移到新场景。

**🔧 技术方法**

采用GOAL‑CLIP进行文本‑图像对齐生成值地图；使用开集目标检测与VLM进行属性检验；构建基于墙面占据图的房间划分；对候选视角进行采样并在3D几何上验证关系。

**📊 数据集**

在HM3D环境下的InstanceNav和CoIN‑Bench两大基准数据集上进行实验。

**📈 对比分析**

与RL训练、其他无训练基线和交互式方法对比，Context‑Nav在InstanceNav上取得最高SR（26.2%）和最佳SPL；在CoIN‑Bench所有子集亦均达成SOTA，在不需训练和交互的前提下表现优越。

**⚠️ 局限性**

局限性包括对精确映射与视角采样的依赖、计算开销较大；在动态或遮挡严重的环境下可能误判；对语义关系细粒度的表达仍有限。

---

## 442. The Patrologia Graeca Corpus: OCR, Annotation, and Open Release of Noisy Nineteenth-Century Polytonic Greek Editions

**arXiv ID:** 2603.09470 | [PDF](https://arxiv.org/pdf/2603.09470v1)

**作者:** Chahan Vidal-Gorène `[一作]` (École nationale des chartes), Bastien Kindt `[通讯]` (UCLouvain)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5091110501)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并实现了针对19世纪多音符号希腊文的《Patrologia Graeca》OCR、布局识别、文本规范化与词形分析工作流，生成了约600万词汇的词形化语料库。

**💡 创新点**

创新点在于：① 针对PG的复杂双语布局与多音符号排版设计了基于YOLO的布局检测与CRNN细调的OCR模型；② 结合人工校正与主动学习迭代训练，显著降低字符错误率；③ 使用混合神经规则后处理（基于PIE+字典）实现了高质量词形、词性标注。

**🔧 技术方法**

技术手段包括：YOLOv5+CRNN（基于Genavensis Græcus 44）、Albumentations模拟噪声、Tesseract、Transkribus、FastText语义嵌入、VGG16视觉嵌入、混合式词形标注（PIE+字典）以及Sketch Engine兼容的XML结构。

**📊 数据集**

主要数据集：PG 161卷的445页人工标注训练集；30页随机抽样的测试集；合成数据（5万例）用于增强；与现有的GREgORI、Pogretra、Gaza-Batrachomyomachia等公开语料进行对比。

**📈 对比分析**

与Tesseract（CER 11.57%）、Transkribus（CER 6.14%）、预训练CRNN+噪声（CER 8.12%）对比，我方模型在PG上实现了CER 1.05%（≈5–7%点提升）和WER 4.69%（≈6–10%点提升），布局检测mAP50>0.96，行检测/阅读顺序精度>0.98。

**⚠️ 局限性**

局限性包括：① 标题与大写字母仍易产生错误；② 跨栏文本与重叠区域偶尔导致识别或阅读顺序混乱；③ 文本规范化（连字符、空行、拉丁字符过滤）仍需人工介入；④ 目前对高频特殊字符与罕见词形的后处理规则不够完善。

---

## 443. PlayWorld: Learning Robot World Models from Autonomous Play

**arXiv ID:** 2603.09030 | [PDF](https://arxiv.org/pdf/2603.09030v1)

**作者:** Tenny Yin `[一作]` (Princeton University), Anirudha Majumdar `[通讯]` (Princeton University)

**通讯引用:** 992 | [OpenAlex ID](https://openalex.org/A5102792178)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

做了一个可扩展的自动化机器人玩耍数据收集和训练管道（PlayWorld），用来训练高保真动作条件视频世界模型。

**💡 创新点**

创新点是利用视觉语言模型生成多样指令，机器人自主玩耍产生丰富接触事件的数据，配合基于难度的学习进度表和大规模扩展，实现了比人类演示更物理一致的预测。

**🔧 技术方法**

使用技术包括视觉语言模型（VLM）、视觉语言动作策略（VLA）、稳定视频扩散（SVD）生成模型、CLIP嵌入、基于难度的自适应训练、Diffusion Steering RL微调等。

**📊 数据集**

用到的数据集是30小时的机器人自主玩耍数据（包括多对象、多任务），以及6小时的人类演示和人类玩耍数据，覆盖3套物体集合。

**📈 对比分析**

与人类演示、人工玩耍基线对比，PlayWorld在LPIPS/SSIM、接触动态预测、政策成功率相关性、RL微调成功率等指标均显著提升，最高提升可达65%。

**⚠️ 局限性**

限制包括仍存在幻觉、数据冗余、未实现主动样本效率，且在更复杂真实环境下可扩展性与泛化性待进一步验证。

---

## 444. YOLO-NAS-Bench: A Surrogate Benchmark with Self-Evolving Predictors for YOLO Architecture Search

**arXiv ID:** 2603.09405 | [PDF](https://arxiv.org/pdf/2603.09405v1)

**作者:** Zhe Li `[一作]` (Wangxuan Institute of Computer Technology Peking University), Yongtao Wang `[通讯]` (Wangxuan Institute of Computer Technology Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了第一个针对 YOLO 目标检测的 Surrogate Benchmark，构建了包含 1,000+ 个完整训练的 YOLOv8–YOLO12 结构样本，并通过自进化机制不断扩充高性能样本，最终训练出 R²=0.815、sKT=0.752 的 LightGBM 预测器。

**💡 创新点**

创新点在于：①专门为 YOLO 检测设计的全面搜索空间；②利用自进化机制在高性能边缘主动收集样本，显著提升预测器的排序准确率；③将 surrogate predictor 直接用作 NAS 的适配度函数，验证其在发现新模型时的有效性。

**🔧 技术方法**

技术手段包括：随机/分层/拉丁超立方采样、LightGBM 预测器、Self‑Evolving Predictor 循环、基于预测器的进化搜索、统一 COCO-mini 训练协议与 P40 GPU 延迟测量。

**📊 数据集**

使用 COCO-mini（COCO 的 10% 子集）进行完整训练和评估，同时在单个 NVIDIA P40 GPU 上测量推理延迟。

**📈 对比分析**

评估方法：在 20% 验证集上测量 R² 与 sKT，利用预测器进行 EA 搜索并对比官方 YOLOv8–YOLO12 基线；结果显示新模型在相同或更低延迟下的 mAP 均优于所有基线，尤其在小模型和大模型端均明显提升。

**⚠️ 局限性**

局限性包括：仅在 COCO-mini 上实验，延迟测量限定在单一 P40 GPU；缺乏对不同硬件平台、完整 COCO 数据集或其他检测任务（实例分割、姿态估计）的适配；未来工作需扩展硬件适配与任务多样性。

---

## 445. Security Considerations for Multi-agent Systems

**arXiv ID:** 2603.09002 | [PDF](https://arxiv.org/pdf/2603.09002v1)

**作者:** Tam Nguyen `[一作]` (Crew Scaler), Dheeraj Arremsetty `[通讯]` (Crew Scaler)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过四阶段方法系统构建多智能体系统（MAS）的威胁知识库，生成并验证针对MAS特有的193项安全威胁，并对16个现有AI安全框架进行定量覆盖评估。

**💡 创新点**

创新点在于首次给出MAS专属的威胁分类体系、引入生成式AI辅助威胁建模与专家评估相结合的验证流程，以及基于三分制得分的跨框架经验性覆盖对比。

**🔧 技术方法**

技术手段包括：构建生产MAS的技术知识库（86章），使用生成式AI进行威胁建模，人工专家校验，结构化安全问卷设计，以及对各框架在193项威胁上的覆盖率进行量化评分。

**📊 数据集**

主要使用的“数据集”是基于真实生产MAS架构的技术文档与公开威胁案例集合，未采用传统机器学习数据集；所有评估基于人工定义的风险项与框架对应规则。

**📈 对比分析**

评估方法为对每个框架在193个威胁项上按0/1/2三分制打分，随后汇总得到每个风险类别和生命周期阶段（设计、开发、运维）的覆盖率；结果显示OWASP Agentic Security Initiative整体覆盖率最高（65.3%），但在非确定性和数据泄露等关键类别下均低于2分，表明所有框架均未能完整覆盖任何单一类别。

**⚠️ 局限性**

局限性包括：评价仅涵盖16个框架，未覆盖更广泛的安全实践；依赖生成式AI生成威胁模型的准确性与专家判断的主观性；评估仅在阶段性完成，尚未在真实生产环境中验证；未针对不同业务领域或多模态AI系统进行细分；对非确定性和数据泄露等高危领域缺乏具体补救措施。

---

## 446. GeoSolver: Scaling Test-Time Reasoning in Remote Sensing with Fine-Grained Process Supervision

**arXiv ID:** 2603.09551 | [PDF](https://arxiv.org/pdf/2603.09551v1)

**作者:** Lang Sun `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**通讯引用:** 71942 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GeoSolver 框架，构建了大规模过程监督数据集 Geo-PRM-2M，训练 token‑级 Process Reward Model GeoPRM，并用 Process‑Aware Tree‑GRPO 对远程感知 VLM 进行对齐，最终在多任务遥感推理中实现 state‑of‑the‑art 性能，同时支持测试时可扩展。

**💡 创新点**

① 通过熵引导的 MCTS 与合成视觉幻觉注入构造细粒度过程监督数据；② token‑级 PRM 细化视觉与逻辑验证；③ Process‑Aware Tree‑GRPO 结合树搜索与 drop‑moment 惩罚解决信用分配与长度偏差；④ PRM 可作为通用验证器实现跨模型 Test‑Time Scaling。

**🔧 技术方法**

使用熵引导 Monte Carlo Tree Search、合成幻觉注入、Vision‑Language 基础模型 GLM‑4.1V‑9B 与 Aimv2‑Huge 视觉编码、token‑级二分类奖励模型、改进的 GRPO（Process‑Aware Tree‑GRPO）等技术。

**📊 数据集**

Geo-PRM-2M（≈2 M 样本）由熵引导 MCTS 与视觉幻觉注入生成；Geo-CoT380k 用于 SFT；多任务遥感基准包括 DIOR‑RSVG、RRSIS、VRSBench、RSVG、DOTAv2、HRRSD、RSOD、VHR、RSIT、RSIC、AID、RS19、UCM、SIRI、NWPU 等 17 个数据集。

**📈 对比分析**

与闭源商业 VLM、开源通用与遥感专用 VLM 进行多任务对比（mIoU、mAP、Accuracy、BLEU‑4）。GeoSolver 在所有任务上均优于现有模型，尤其在视觉定位与计数任务提升显著；通过 Test‑Time Scaling，GeoPRM 能将通用 VLM 提升至甚至超过专用遥感模型。

**⚠️ 局限性**

仍需大规模预训练模型与高算力；PRM 训练依赖自动化标注，受 MCTS 探索范围限制；在极端噪声或极小目标下可解释性不足；长序列生成的稳定性与奖励退化缺乏完整理论分析。

---

## 447. A Voronoi Cell Formulation for Principled Token Pruning in Late-Interaction Retrieval Models

**arXiv ID:** 2603.09933 | [PDF](https://arxiv.org/pdf/2603.09933v1)

**作者:** Yash Kankanampati `[一作]` (Sorbonne Paris Nord), Joseph Le Roux `[通讯]` (Sorbonne Paris Nord)

**通讯引用:** 869 | [OpenAlex ID](https://openalex.org/A5113436108)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于Voronoi细胞的令牌剪枝框架（Voronoi Pruning），通过最小化预期检索误差来挑选最不重要的文档令牌，从而显著减小索引体积而不损失检索效果。

**💡 创新点**

创新点在于：①将令牌重要性与嵌入空间中的Voronoi区域关联；②把剪枝问题形式化为最小化期望误差的几何优化；③提出了高效的Monte Carlo估计和迭代剪枝算法，速度比现有LP‑Pruning提升约120倍。

**🔧 技术方法**

技术手段包括：多维向量空间几何分析、Voronoi细胞估计、Monte Carlo采样、迭代误差更新、全局与局部剪枝策略、以及基于平均误差（Mean Error）的评估方法。

**📊 数据集**

主要实验数据集为 MS MARCO v1 passage、TREC DL19/20、BEIR 公开基准（包含 12 个子数据集），并在 ColBERT 预训练模型上进行评估。

**📈 对比分析**

与传统剪枝（如 stop‑word、IDF、First‑k）、学习型剪枝（AligneR、ConstBERT、ColBERTer）以及 LP‑Pruning 等基线相比，Voronoi Pruning 在保持 90%+ 的检索效果（如 MRR@10、nDCG@10）同时将索引大小缩减到 10% 以下；在极端剪枝比例下仍优于学习型方法，且推理速度更快。

**⚠️ 局限性**

局限性包括：①仅考虑期望误差，无法捕捉查询空间中的局部大误差；②目标是最大点积保持，而非直接优化检索损失；③仅实现了令牌选择而非嵌入空间的结构重塑，可能在跨域场景下表现不佳。

---

## 448. MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations

**arXiv ID:** 2603.09800 | [PDF](https://arxiv.org/pdf/2603.09800v1)

**作者:** Abhishikth Mallampalli `[一作]` (University of Wisconsin Madison), Sridhara Dasu `[通讯]` (University of Wisconsin Madison)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了MITRA，一个面向大型粒子物理协作的本地化检索增强生成（RAG）问答系统；

**💡 创新点**

创新点在于：①完全本地化部署（包括嵌入模型和LLM），保障数据隐私；②双层向量数据库设计，先按摘要定位分析再锁定全文，避免跨分析混淆；③自动化的Selenium+OCR文本抽取流水线，支持频繁更新。

**🔧 技术方法**

技术包括：Selenium浏览器自动化、OCR（Tesseract/Adobe等）+布局解析、Dense Passage Retrieval（DPR）+Cross-Encoder reranker、Chroma DB向量存储、4-bit量化的GPT‑4o‑mini模型、vLLM/llama.cpp推理引擎。

**📊 数据集**

使用CMS内部的分析文档（PDF/Wiki），涵盖数千页的分析笔记、摘要等；并用专家设计的两组查询集（Set 1、Set 2）进行评估。

**📈 对比分析**

与传统关键词检索BM25对比：在Set 2（同义/释义查询）上，MITRA在P@1、MRR、NDCG@5等指标上分别取得0.75/0.81/0.88，明显优于BM25的0.13/0.35/0.59；在Set 1（精确匹配）上两者相近或BM25略优。

**⚠️ 局限性**

局限性包括：仅评估了检索阶段，生成质量仍待量化；仅使用单一文档类型；对高并发负载的真实测评尚未完成；需进一步扩展到多轮对话与更丰富的任务。

---

## 449. Routing without Forgetting

**arXiv ID:** 2603.09576 | [PDF](https://arxiv.org/pdf/2603.09576v1)

**作者:** Alessio Masano `[一作]` (University of Catania), Concetto Spampinato `[通讯]` (University of Catania)

**通讯引用:** 6707 | [OpenAlex ID](https://openalex.org/A5075815307)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于能量关联检索的Transformer路由机制（RwF），实现在线连续学习，消除任务特定模块，保持仅一次前向传播即可完成路由。

**💡 创新点**

创新点在于将持续学习重新表述为路由问题，并使用Modern Hopfield网络实现闭式能量最小化的单步关联检索，避免逐步梯度专化，提升在线适应性与稳定性。

**🔧 技术方法**

采用能量基关联检索（Modern Hopfield + HopfieldPooling）、ViT预训练主干、参数高效微调（无LoRA/提示/Replay）、闭式前向路由与连续平滑化策略。

**📊 数据集**

在三个类增量基准上评测：Split‑CIFAR‑100、Split‑ImageNet‑R、Split‑ImageNet‑S（ImageNet 1000‑class分为10个任务）。

**📈 对比分析**

与Replay、LoRA、Prompt、双主干等SOTA方法对比；在ImageNet分割上取得最高的最终平均准确率（74.09%/61.37%），仅增2.1%可训练参数；在少样本和任务碎片化场景保持稳健优势，性能提升显著。

**⚠️ 局限性**

局限性：在细粒度分类（如CUB‑200）中效果下降，HopfieldPooling的特征聚合可能抹平细节导致类区分弱化。

---

## 450. Beyond Amplitude: Channel State Information Phase-Aware Deep Fusion for Robotic Activity Recognition

**arXiv ID:** 2603.09047 | [PDF](https://arxiv.org/pdf/2603.09047v1)

**作者:** Rojin Zandi `[一作]` (Northeastern University), Milad Siami `[通讯]` (Mayo Clinic)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了GF-BiLSTM两流门控融合网络，用于利用Wi‑Fi CSI幅度和相位信息识别机器人臂动作。

**💡 创新点**

首先系统评估了相位在机器人动作识别中的作用，并设计了门控融合机制与模态丢弃训练以自适应整合幅度与相位；其次在离线速度留一法评估中获得最佳跨速度鲁棒性。

**🔧 技术方法**

采用时间展开相位的相位解包与线性消噪、双向LSTM编码、门控融合、层归一化以及AdamW优化器等技术。

**📊 数据集**

在RoboFiSense数据集上实验，该数据集记录了Franka Emika机械臂在三种速度下执行八种动作的CSI。

**📈 对比分析**

与CNN、LSTM、BiLSTM、ViT、BiVTC等基线在Leave-One-Velocity-Out (LOVO)协议下对比，GF-BiLSTM在双通道幅度+相位下平均准确率达约96%，显著高于其它模型。

**⚠️ 局限性**

处理相位的线性消噪计算量大，导致较高的预处理延迟；实验仅覆盖单一机器人平台与八种动作，模型在更复杂环境或更多动作时的泛化仍待验证。

---

## 451. Some polynomial classes for the acyclic orientation with parity constraint problem

**arXiv ID:** 2603.09475 | [PDF](https://arxiv.org/pdf/2603.09475v1)

**作者:** Sylvain Gravier `[一作]` (University Grenoble Alpes), Isabelle Sivignon `[通讯]` (University Grenoble Alpes)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了带有奇偶约束的无向图的非循环定向问题，特别是如何找到一个满足特定顶点集合的奇偶定向。

**💡 创新点**

提出了三个必要条件，以确保存在非循环的奇偶定向，并定义了包含这些条件的图类，建立了这些类之间的包含关系。

**🔧 技术方法**

使用了构造性证明方法，结合图的类的定义和性质，构建了多项式时间内的非循环奇偶定向。

**📊 数据集**

使用了路径和循环的笛卡尔积图作为数据集，特别是网格、圆柱和大环面图。

**📈 对比分析**

通过与现有算法的比较，展示了所提出方法的有效性，特别是在特定图类中，证明了这些类的严格包含关系。

**⚠️ 局限性**

限制在于当前的复杂性问题仍然未解决，特别是对于图类的识别是否存在多项式时间算法的问题。

---

## 452. Embodied Human Simulation for Quantitative Design and Analysis of Interactive Robotics

**arXiv ID:** 2603.09218 | [PDF](https://arxiv.org/pdf/2603.09218v1)

**作者:** Chenhui Zuo `[一作]` (Tsinghua University), Yanan Sui `[通讯]` (Tsinghua University)

**通讯引用:** 1151 | [OpenAlex ID](https://openalex.org/A5069290448)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套基于完整身体肌肉骨骼模型与强化学习控制器的仿真框架，用于量化分析人机交互，并通过协同优化机器人结构和控制实现最佳外骨骼设计。

**💡 创新点**

创新点包括：1) 将全身体肌肉骨骼模型与RL驱动的可交互代理相结合，首次提供内部生物力学反馈；2) 引入结构参数协同优化与可分离PD控制器，扩大设计空间；3) 利用CMA-ES在大规模参数空间内高效搜索最优解；4) 开源整个框架，促进社区复现。

**🔧 技术方法**

使用技术包括：MuJoCo物理引擎、MS-Human-700完整人体模型、Soft Actor-Critic / DynSyn 强化学习、可分离PD控制器、域随机化、CMA-ES 进化优化。

**📊 数据集**

主要数据集为 AMASS 运动捕捉数据库（人类行走轨迹），用于训练和验证仿真代理。

**📈 对比分析**

对比方法：控制仅优化、结构仅优化与协同优化。实验结果表明，协同优化在关节误差、肌肉力和接触力方面均显著优于两种基线，成本函数下降最快、最终值最低，体现了更高的效能与舒适性。

**⚠️ 局限性**

局限性：1) 仍存在 sim-to-real 桥接难题，缺乏完整的生理数据验证；2) 模型未包含软组织动力学、疲劳等复杂因素；3) 当前仅在单一受试者的单一任务上训练，难以直接推广至多样化人群与多种运动；4) 主要验证为运动学指标，缺乏肌电或反作用力等更细粒度的实验数据。

---

## 453. Improving through Interaction: Searching Behavioral Representation Spaces with CMA-ES-IG

**arXiv ID:** 2603.09011 | [PDF](https://arxiv.org/pdf/2603.09011v1)

**作者:** Nathaniel Dennler `[一作]` (Massachusetts Institute of Technology), Maja Matarić `[通讯]` (University of Southern California)

**通讯引用:** 30253 | [OpenAlex ID](https://openalex.org/A5010248533)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种名为CMA-ES-IG的交互式偏好学习算法，用于让机器人在与人类交互时根据非专家用户的轨迹排序快速个性化行为。

**💡 创新点**

创新点在于将信息增益查询和CMA‑ES采样相结合：先用CMA‑ES搜索高奖励轨迹，再通过K‑means聚类挑选感知上可区分的轨迹，既保持迭代改进又提高用户的排序准确性，解决了传统方法在高维空间与用户体验上的矛盾。

**🔧 技术方法**

技术手段包括Luce‑Shepard/Plackett‑Luce排名模型、贝叶斯更新推断奖励权重、基于多元高斯分布的CMA‑ES采样、K‑means聚类量化剪枝以及自编码器或其他表示学习方法生成轨迹特征。

**📊 数据集**

实验使用的轨迹数据集包括四个模拟任务（Lunar Lander、Driving、Robot Face Design、Voice Design）的手工或学习得到的4维表示；以及真实实验中的JACO手交互轨迹和Blossom机器人表情轨迹的自动编码器得到的4/6维特征。

**📈 对比分析**

通过与纯信息增益、标准CMA‑ES三种基线对比，使用AUC（对齐度/后悔值/质量）和用户问卷（行为适应、易用度、算法排名）评估；在高维模拟任务中CMA‑ES‑IG在对齐度和质量上优于基线，后悔值最低；在真实用户实验中CMA‑ES‑IG在行为适应、易用度和用户总排名上显著高于两基线，显示出更好的性能。

**⚠️ 局限性**

主要局限包括：依赖预先收集的高质量、可多样化的轨迹数据；采用线性奖励假设，可能不适用于非线性偏好；实验样本为大学生，缺乏对残障或不同能力人群的验证；尚未在极低维或极大规模问题中进一步评估。

---

## 454. Latent-DARM: Bridging Discrete Diffusion And Autoregressive Models For Reasoning

**arXiv ID:** 2603.09184 | [PDF](https://arxiv.org/pdf/2603.09184v1)

**作者:** Lina Berrayana `[一作]` (École Polytechnique Fédérale de Lausanne), Wei Chen `[通讯]` (Microsoft Research Asia)

**通讯引用:** 33934 | [OpenAlex ID](https://openalex.org/A5100344522)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Latent-DARM 框架，利用隐空间投影实现离散扩散语言模型（DDLM）规划器与自回归语言模型（ARM）执行器的协同推理。

**💡 创新点**

创新点在于：① 通过学习的线性‑GELU‑线性投影将 DDLM 的隐藏表示映射到 ARM 的嵌入空间，消除文本解码的低流畅性损耗；② 采用任务导向的负对数似然目标训练投影器，而非几何对齐；③ 在保持模型冻结的前提下，实现跨架构高效信息传递。

**🔧 技术方法**

主要技术包括：离散扩散生成、隐空间投影网络、基于负对数似然的投影器训练、LoRA 轻量化适配、Bfloat16 训练等。

**📊 数据集**

使用的数据集涵盖多种推理任务：ARC‑E、ARC‑C、DART‑1~5、AIME‑2024、MMLU 等；投影器训练样本来自 ARC 与 DART 7 个子集，共 35,000 条。

**📈 对比分析**

与传统文本接口（DDLM→ARM）和单一 ARM 模型对比，Latent‑DARM 在 DART 系列和 AIME 上显著提升准确率（如 DART‑5 由 27% 提升至 54%），在不使用额外文本生成的情况下仅消耗 2.2% 的 token 预算，且整体 token 成本低于 ARM‑only 与常规推理模型。

**⚠️ 局限性**

局限性：在广泛事实检索任务（如 MMLU）上表现不及文本接口；未达到最先进推理模型的准确率；投影器训练仅基于规划密集型数据，泛化能力受限；并且仍需针对不同任务动态切换隐空间与文本模式的机制。

---

## 455. Touching Emotions, Smelling Shapes: Exploring Tactile, Olfactory and Emotional Cross-sensory Correspondences in Preschool Aged Children

**arXiv ID:** 2603.08889 | [PDF](https://arxiv.org/pdf/2603.08889v1)

**作者:** Tegan Roberts-Morgan `[一作]` (University of Bristol), Oussama Metatla `[通讯]` (University of Bristol)

**通讯引用:** 1653 | [OpenAlex ID](https://openalex.org/A5006138357)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对2-4岁幼儿进行触觉-嗅觉-情感跨感官对应的实验研究。

**💡 创新点**

首次在学龄前儿童中系统测量并证实多感官对应，并提出以故事驱动的实验方法与关联策略分析。

**🔧 技术方法**

采用木盒触觉刺激、Hynt嗅味机、叙事式任务与手部触摸、嗅味、情感标注。

**📊 数据集**

样本26名2-4岁幼儿，使用自定义的三种形状、三种气味、三种情感标签。

**📈 对比分析**

与先前成人/老童研究相比，通过卡方检验发现触觉-语言、触觉-气味、气味-情感对应显著(p<0.05)，但触觉-情感无显著性。

**⚠️ 局限性**

局限于小样本、仅限言语回应、缺乏其他感官与纵向发展考察。

---

## 456. Layered Dielectric Characterization of Human Skin in the Sub-Terahertz and Terahertz Frequency Ranges

**arXiv ID:** 2603.09822 | [PDF](https://arxiv.org/pdf/2603.09822v1)

**作者:** Silvia Mura `[一作]` (Politecnico di Milano), Marco Hernandez `[通讯]` (University of Oulu)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5021357438)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个多层次、多德拜模型结合有效介质理论的皮肤介电常数预测框架，并用 voxel 统计模型验证了亚 THz/THz 波在皮肤中的传播损耗。

**💡 创新点**

将细胞内部水、蛋白质、脂质三种成分的多德拜参数与 Maxwell–Garnett 有效介质相结合，实现从细胞到组织层的频率依赖介电模型，并给出概率分布的吸收和散射损耗分析。

**🔧 技术方法**

采用多德拜折叠模型、Maxwell–Garnett 有效介质理论、Rayleigh 与 Mie 散射理论、频域传播损耗公式以及 MATLAB voxel 化模拟技术。

**📊 数据集**

使用实验验证的水、蛋白、脂质在 100 GHz–1 THz 的多德拜参数，以及各皮肤细胞（角质细胞、基底细胞、成纤维细胞、红细胞、脂肪细胞等）的质量分数与体积密度。

**📈 对比分析**

通过与实验测得的皮肤吸收系数和散射系数对比，模型在 100 GHz 时预测总损耗约 45 dB、1 THz 时约 65 dB；散射损耗在两频段均低于吸收损耗，验证了模型对频率依赖性的准确性。

**⚠️ 局限性**

未考虑高频下离子导电贡献；对大尺度结构（如血管网）仅用线性 Poisson 模型，未捕捉真实血管形态；实验验证主要局限于宏观参数，缺少细胞级别的直接测量。

---

## 457. Scientific Rigor and Human Warmth: Remembering Vladimir Sidorenko (1949-2025)

**arXiv ID:** 2603.09437 | [PDF](https://arxiv.org/pdf/2603.09437v1)

**作者:** Christian Deppe `[一作]` (Technical University of Braunschweig), Gohar Kyureghyan `[通讯]` (Universität Rostock)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文记录了在FFCS会议上举行的纪念会，回顾了Vladimir Sidorenko在编码理论、密码学、电信与量子误差纠正领域的重要科研贡献以及他在学术社区中的人格魅力。

**💡 创新点**

创新点在于系统地将Sidorenko的严谨学术态度、以问题为驱动的思考方式与跨学科交流的实践经验相结合，展示了科研与人文的双重价值。

**🔧 技术方法**

主要技术涉及编码理论与密码学中的代数与组合方法、量子纠错中的树形方法以及通信系统中的信号处理技术。

**📊 数据集**

本报告未使用具体实验数据集，主要基于个人回忆、公开发表论文和会议演讲记录。

**📈 对比分析**

由于该文为纪念性质，并未进行实验或方法比较，故无性能评估；其评价以同行评语和学术引用为主。

**⚠️ 局限性**

局限性在于缺乏系统的实验验证与客观数据支持，且对Sidorenko技术贡献的深度与广度呈现主观概括。

---

## 458. AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents

**arXiv ID:** 2603.09716 | [PDF](https://arxiv.org/pdf/2603.09716v1)

**作者:** Xiaoxing Wang `[一作]` (MemTensor), Feiyu Xiong `[通讯]` (MemTensor)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AutoAgent，一种自我进化的多智能体框架，融合可学习的认知、即时上下文决策和弹性记忆组织，实现对工具使用、协作与长周期推理的自适应优化。

**💡 创新点**

创新点在于：①将认知拆分为内部与外部可动态更新的知识库，②统一内外动作空间并通过即时决策循环替代固定流程，③弹性记忆模块在每一步动态压缩与检索历史，④通过意图-结果对齐的闭环自我进化持续修正认知。

**🔧 技术方法**

技术包括：大语言模型交互式提示、内部/外部认知结构化、弹性记忆编排（逐步压缩与多级抽象）、意图与结果匹配的自我反思引擎、以及基于 LLM 的工具与技能推理。

**📊 数据集**

使用了多任务数据集：检索增强生成（HotpotQA、2WikiMultiHopQA、Bamboogle、Musique）、工具驱动任务（GAIA、HLE-Bench）以及文本环境模拟（ALFWorld）。

**📈 对比分析**

与 ReAct、DeepAgent、Self-Ask、IRCoT 等基线在同一模型与工具环境下对比，AutoAgent 在 RAG 任务上平均准确率提升至 0.3965（最高 0.530），工具使用任务上成功率普遍高于基线，尤其在闭源模型 Gemini 系列表现突出。

**⚠️ 局限性**

局限性包括：对实体密集型推理（如 Musique）的表现仍落后；自我进化依赖 LLM 生成的文本更新，可能受限于模型偏差；在高度动态工具或复杂多代理协作情境下仍需进一步验证与优化。

---

## 459. Distributed Convolutional Neural Networks for Object Recognition

**arXiv ID:** 2603.09220 | [PDF](https://arxiv.org/pdf/2603.09220v1)

**作者:** Liang Sun `[一作]` (Shandong University of Science and Technology), Liang Sun `[通讯]` (Shandong University of Science and Technology)

**通讯引用:** 71858 | [OpenAlex ID](https://openalex.org/A5100382892)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种分布式卷积神经网络 DisCNN，并设计了一种将负样本映射到原点的 N2O 损失函数，使网络仅提取正类特征。

**💡 创新点**

创新点在于通过 N2O 损失实现正负样本特征的分离，将负样本压缩至原点、正样本聚集至紧致集合，从而实现轻量化且可解释的正类识别。

**🔧 技术方法**

使用卷积神经网络结构（4 层卷积 + 3 层全连接）、批归一化、ReLU、最大池化，并加入自定义 N2O 损失进行训练；后续通过输出向量模长阈值进行分类。

**📊 数据集**

在 STL-10 数据集上，以 “car” 为正类，{bird, cat} 为负类进行训练与评估。

**📈 对比分析**

与传统 VGG（10 类交叉熵）相比，DisCNN 参数量从 3.096M 降至 0.149M；在正负样本分类上取得 TP=787、FP=13 的优异性能，阈值可调且对未见类也保持较好泛化。

**⚠️ 局限性**

主要限制包括：N2O 损失理论尚不完善，难以保证所有负类都严格映射至原点；对相似特征的负类可能产生误判；模型设计仅针对单一正类，难以同时处理多类任务。

---

## 460. Beyond Relevance: On the Relationship Between Retrieval and RAG Information Coverage

**arXiv ID:** 2603.08819 | [PDF](https://arxiv.org/pdf/2603.08819v1)

**作者:** Saron Samuel `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**通讯引用:** 8658 | [OpenAlex ID](https://openalex.org/A5075825791)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性研究了检索质量与RAG生成信息覆盖度之间的关系，并实证证明检索覆盖度指标可作为生成覆盖度的早期预测指标。

**💡 创新点**

首次将检索覆盖度指标与RAG生成覆盖度的相关性进行系统量化，评估不同管线复杂度、评估器和模态对该关系的影响，并提出检索指标可作为RAG性能的代理。

**🔧 技术方法**

使用多种RAG管线（GPT‑Researcher、Bullet List、LangGraph、CAG）、检索模型（BM25、PLAID‑X、LSR、Qwen3‑Embed、RRF等）、评估框架（Auto‑ARGUE、MiRAGE）以及Pearson相关分析。

**📊 数据集**

利用TREC NeuCLIR 2024 Report Generation Pilot、TREC RAG 2024以及WikiVideo（多模态视频生成）三大数据集。

**📈 对比分析**

通过在话题层和系统层分别计算检索指标（α‑nDCG、StRecall等）与生成覆盖度的Pearson相关系数，发现检索指标越高对应生成覆盖度提升；在不同评估器和模态下保持强相关，复杂迭代管线的相关性则显著下降。

**⚠️ 局限性**

研究局限：聚焦信息覆盖度，未评估流畅性和真实性；主要依赖自动评测；多模态实验仅使用单一生成模型；检索与生成目标匹配度强度决定相关性，迭代管线可能使其脱钩。

---

## 461. FetalAgents: A Multi-Agent System for Fetal Ultrasound Image and Video Analysis

**arXiv ID:** 2603.09733 | [PDF](https://arxiv.org/pdf/2603.09733v1)

**作者:** Xiaotian Hu `[一作]` (Tsinghua University), Qiyuan Tian `[通讯]` (Tsinghua University)

**通讯引用:** 2777 | [OpenAlex ID](https://openalex.org/A5066843175)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一套名为FetalAgents的多智能体系统，用于全面分析胎儿超声图像与视频并自动生成结构化报告。

**💡 创新点**

通过LLM驱动的协调器动态调用多种专业视觉模型，实现任务特定性能与端到端临床工作流的统一；并在视频流中自动提取关键帧并整合成报告。

**🔧 技术方法**

采用GPT‑5‑mini作为协调器；利用FetalCLIP、FU‑LoRA、ResNet‑50、ViT‑B/16等模型做平面识别；使用nnU‑Net、USFM、SAM‑based模型做分割；利用AoP‑SAM、UperNet做角度测量；以及多模型融合与确定性决策规则。

**📊 数据集**

使用多源公开胎儿超声数据集（Fetal_PLANES_DB、ACOUSLIC‑AI、PSFHS、HC18、PBF‑US1等）进行训练，并在多中心外部数据集（非洲、华人、美国等）进行验证。

**📈 对比分析**

与单一视觉模型、基础模型及通用/医学大型语言模型进行对比；在八项临床任务中，FetalAgents在准确率、Dice系数、误差指标和有效率等多项指标上均优于所有基线，显示出最稳健的性能。

**⚠️ 局限性**

局限性包括对高质量多模态医疗记录的依赖、在少量稀缺数据场景下的泛化能力待验证，以及在真实临床部署中可能出现的时间延迟和可解释性挑战。

---

## 462. PathoScribe: Transforming Pathology Data into a Living Library with a Unified LLM-Driven Framework for Semantic Retrieval and Clinical Integration

**arXiv ID:** 2603.08935 | [PDF](https://arxiv.org/pdf/2603.08935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 463. ChatNeuroSim: An LLM Agent Framework for Automated Compute-in-Memory Accelerator Deployment and Optimization

**arXiv ID:** 2603.08745 | [PDF](https://arxiv.org/pdf/2603.08745v1)

**作者:** Ming-Yen Lee `[一作]` (Georgia Institute of Technology), Shimeng Yu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 30714 | [OpenAlex ID](https://openalex.org/A5054894631)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ChatNeuroSim，一套基于大语言模型的代理框架，自动完成Compute‑in‑Memory（CIM）加速器的部署与优化。

**💡 创新点**

核心创新在于利用LLM实现请求解析、参数校验、脚本生成与仿真调用，并结合设计空间修剪的启发式优化算法，实现快速、高效的DSE。

**🔧 技术方法**

技术主要包括ChatGPT‑5 系列 LLM、LangGraph 多代理框架、NeuroSim 仿真器、遗传算法、模拟退火、TPE 以及跨空间约束投影、Top‑K 修剪和随机退修剪的设计空间修剪方法。

**📊 数据集**

使用自定义的 40 条 CIM 请求样例数据集，以及 ImageNet 上的 ResNet‑50、Swin‑Transformer‑Tiny 与 Vision‑Transformer‑Base 三个 DNN 工作负载作为评估数据集。

**📈 对比分析**

通过与无修剪基线对比，修剪策略使 Swin‑T 的平均优化时间缩短 42%–79%，P95 时间缩短 29%–69%；在不同目标（FoM、能效、计算效率、吞吐量）与硬件约束下均显著加速搜索且保持或提升最佳 PPA。

**⚠️ 局限性**

局限性包括：对 CNN 任务修剪效果有限；依赖于基模型相似度；退修剪超参数需要经验调优；在 LLM 版本差异下性能不一致，需进一步优化提示与校验机制。

---

## 464. DexHiL: A Human-in-the-Loop Framework for Vision-Language-Action Model Post-Training in Dexterous Manipulation

**arXiv ID:** 2603.09121 | [PDF](https://arxiv.org/pdf/2603.09121v1)

**作者:** Yifan Han `[一作]` (Chinese Academy of Sciences Institute of Automation), Wenzhao Lian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 573 | [OpenAlex ID](https://openalex.org/A5017678179)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 DexHiL 框架中，作者设计了一套端到端的手臂‑手腕协同人机交互后训练流程，结合轻量化遥控接口与干预感知采样，提升了视觉语言动作模型在多指手的精细抓取任务中的表现。

**💡 创新点**

创新点在于将手臂与多指手的操作统一到同一 HiL 系统中，提出了针对纠正段的加权采样机制，并实现了双路径手指重映射与异步多线程控制，实现了高效的在线纠错与快速收敛。

**🔧 技术方法**

技术上采用了预训练的 VLA 模型 Being‑H0.5（Mixture of Transformers + Flow Matching），两阶段手指重映射网络、基于 ArUco 的姿态映射、DAgger 循环与重要性采样权重、以及光流匹配的动作头。

**📊 数据集**

数据来源包括 60 条离线遥控轨迹（初始）以及随后每轮 10 条在线 HiL 纠错轨迹，所有轨迹基于真实 Franka 3 + DexHand021 机器人；VLA 预训练则使用大规模人类视频数据。

**📈 对比分析**

通过与基于相同数据量的离线微调（Offline‑40/50/60）和无加权 DAgger* 的对比，DexHiL 在两项任务上分别达到了 95%/65% 的成功率（相较基线提升约 25%），并在三轮后将人工干预时长减少约 35%。

**⚠️ 局限性**

局限性在于仍需依赖硬件遥控与手指重映射的精确度，且对不同任务的泛化尚待验证；加之 VLA 模型仍依赖大规模预训练，在线收敛速度受数据质量与采样策略影响。

---

## 465. From Semantics to Pixels: Coarse-to-Fine Masked Autoencoders for Hierarchical Visual Understanding

**arXiv ID:** 2603.09955 | [PDF](https://arxiv.org/pdf/2603.09955v1)

**作者:** Wenzhao Xiang `[一作]` (Chinese Academy of Sciences), Xilin Chen `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 35511 | [OpenAlex ID](https://openalex.org/A5083420537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于粗到细掩码自编码器（C2FMAE），通过语义、实例和像素三层数据实现多粒度视觉预训练。

**💡 创新点**

创新点在于（1）级联解码器按语义→实例→像素顺序逐层重建，强制实现自上而下的层次信息流；（2）进化式掩码策略从语义引导→实例引导→随机掩码的学习课程，缓解注意力漂移。

**🔧 技术方法**

使用Vision Transformer编码器、级联Transformer解码器、进化掩码生成器以及多任务交叉熵/均方误差损失进行训练。

**📊 数据集**

构建了包含1.28M ImageNet-1K图像的多粒度伪标签数据集，分别提供语义掩码、实例掩码和RGB图像。

**📈 对比分析**

在ImageNet分类、COCO检测/分割和ADE20K分割等下游任务中，与MAE和MultiMAE等基线相比，C2FMAE在分类Top‑1、检测AP和分割mIoU上分别提升约1–2个百分点，且训练效率与现有方法相近。

**⚠️ 局限性**

局限性包括对伪标签质量的依赖、额外的多模态数据预处理开销以及在更大规模模型或不同域上的泛化需进一步验证。

---

## 466. Evaluation of LLMs in retrieving food and nutritional context for RAG systems

**arXiv ID:** 2603.09704 | [PDF](https://arxiv.org/pdf/2603.09704v1)

**作者:** Maks Požarnik Vavken `[一作]` (Institute Jožef Stefan), Barbara Koroušić Seljak `[通讯]` (Institute Jožef Stefan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了四种大型语言模型在检索食品与营养数据库时，利用自然语言生成结构化元数据过滤器的性能。

**💡 创新点**

创新点在于将LLM生成的元数据过滤器与向量数据库结合，验证LLM可直接驱动RAG系统进行高效检索，并探讨了在查询复杂度提升时的可靠性边界。

**🔧 技术方法**

主要技术包括：Chroma向量数据库、LLM（Gemini‑2.0‑Flash、GPT‑4o、Claude‑Sonnet‑4、Mistral Medium 3）生成元数据过滤器、两阶段检索（过滤+语义相似度）以及动态相似度阈值策略。

**📊 数据集**

使用的数据集是斯洛文尼亚食品成分数据库（FCDB），包含约32,000个食品项目，并将其转化为嵌入向量进行检索。

**📈 对比分析**

通过对150个难度不同（易/中/难）的自然语言查询进行5次独立评测，并使用F1分数比较模型与阈值的性能；易/中类查询几乎达到完美检索（F1>0.999），难类查询最高F1仅为0.450，阈值越严格（μ‑σ）在难类中平均表现略好。

**⚠️ 局限性**

局限性包括：仅使用Chroma数据库，过滤器匹配大量条目时偶有检索缺失；语言仅限斯洛文尼亚，通用性未知；未评估模型成本与可扩展性；对新模型迭代的性能变化缺乏系统分析。

---

## 467. A Note on the Equivalence Between Zero-knowledge and Quantum CSS Codes

**arXiv ID:** 2603.08941 | [PDF](https://arxiv.org/pdf/2603.08941v1)

**作者:** Noga Ron-Zewi `[一作]` (University of Haifa), Mor Weiss `[通讯]` (Bar-Ilan University)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5070327070)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明线性完美零知识码与量子CSS码等价，并利用该等价构造出显式的渐近良好零知识可局部检验码（ZK‑LTC）

**💡 创新点**

首次建立零知识码与量子CSS码之间的双向对应关系，进而将量子LTC构造直接转化为显式零知识LTC，填补了先前只能得到概率性构造的空缺

**🔧 技术方法**

使用线性编码理论、随机化编码技术、量子错误校正中的CSS结构与双重相容性分析，并引入现有的量子局部检验码（LTC）构造

**📊 数据集**

无；论文为理论构造，未使用任何具体数据集

**📈 对比分析**

通过构造证明，得到的零知识阈值线性、局部查询多对数级；相较于之前仅提供概率性、非显式构造的方法，该方案提供显式实现，且保持距离和速率的渐近最优性能

**⚠️ 局限性**

仅适用于线性完美零知识码，非线性或统计ZK码尚未涵盖；目前得到的ZK‑LTC仍需多对数查询，尚无显式常数查询版本

---

## 468. MultiGraSCCo: A Multilingual Anonymization Benchmark with Annotations of Personal Identifiers

**arXiv ID:** 2603.08879 | [PDF](https://arxiv.org/pdf/2603.08879v1)

**作者:** Ibrahim Baroud `[一作]`, Roland Roller `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了多语言匿名化基准 MultiGraSCCo，扩展了原始德国临床文本数据集 GraSCCo，添加了间接个人识别符 (IPI) 注释，并通过 GPT‑4.1 进行注释保留与文化适配的多语言翻译。

**💡 创新点**

创新点在于首次系统性地为多语言临床文本加入 IPI 注释层，并利用高级语言模型实现跨语言、跨文化的注释保留翻译，验证了少量目标语言监督即可显著提升模型性能。

**🔧 技术方法**

技术手段包括：使用 INCEpTION 进行 IPI 注释；使用 GPT‑4.1 对文本进行预处理（错别字纠正、缩写展开）并执行注释保留翻译；采用 mmBERT、RoBERTa 以及多语言模型进行单语、跨语、跨语言以及多语模型实验。

**📊 数据集**

使用的数据集为公开的德国语 GraSCCo（含 PHI 注释），在此基础上新增 13 类 IPI 注释，随后翻译成 9 种目标语言（英语、法语、阿拉伯语、波斯语、意大利语、波兰语、俄语、乌克兰语、土耳其语）。

**📈 对比分析**

实验对比了三种设置：单语基线、零样本跨语、以及在每种语言加入 25–100% 目标语言数据的多语训练。结果显示，单语模型微 F1 通常 0.86–0.90，跨语模型在 IPI 上显著下降（宏 F1 0.42–0.55），而少量目标语言数据即可提升至与单语相当甚至更优，证明了多语训练的有效性。

**⚠️ 局限性**

局限性包括：原始 GraSCCo 中人为设定的名字和缩写可能影响真实度；翻译缺乏真实目标语言临床文本的书写风格；实验仅在翻译文本上评估，未在真实患者数据上验证模型的实际可用性。

---

## 469. The framework to unify all complexity dichotomy theorems for Boolean tensor networks

**arXiv ID:** 2603.09417 | [PDF](https://arxiv.org/pdf/2603.09417v1)

**作者:** Mingji Xia `[一作]` `[通讯]` (Institute of Software, Chinese Academy of Sciences), Mingji Xia (Institute of Software, Chinese Academy of Sciences)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出对全类 Holant 问题 (#ℱ) 的统一研究框架，利用二元函数形成的有限群与 SU(2)/SO(3) 的有限子群对应关系，将所有未解的 #ℱ 问题划分为九个互斥子类，并在此基础上完成了阶为1的循环群和高阶循环群的完整复杂度分类；此外给出了若干已知的二分法与三分法等证明工具；

**💡 创新点**

通过将二元函数组建成有限群并映射到 3 维旋转群，从根本上统一并细化了以往分散的二分法定理；首次把所有未解的 #ℱ 子类划分为九个互斥组，并在此框架下完成了循环群子类的完整 dichotomy，开辟了一个全新的“群论驱动”的 Holant 复杂度研究方向；

**🔧 技术方法**

使用了 holographic 变换、插值归约、分解引理、转置闭包、实化技术以及有限群与 SU(2)/SO(3) 的几何分类等多种数学与计算方法；

**📊 数据集**

无；该工作为纯理论复杂度研究，不涉及实验数据或数据集；

**📈 对比分析**

通过理论证明展示了在每个子类中问题的计算复杂度为 #P‑hard 或可在 FP^NP 内解决；与现有 dichotomy 结果对比，证明了更一般化的类可归约为已知可解类，从而实现了对全类的理论覆盖；

**⚠️ 局限性**

对含有 K4 子群（如四面体、八面体、二面体大偶数阶）以及四元数群等情况仍未完成实化与完整证明；该方法在实化步骤上遇到根本性障碍，导致部分子类仍停留在猜想或未解状态。

---

## 470. Exclusive Self Attention

**arXiv ID:** 2603.09078 | [PDF](https://arxiv.org/pdf/2603.09078v1)

**作者:** Shuangfei Zhai `[一作]` `[通讯]` (Apple), Shuangfei Zhai (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了独占自注意力（XSA），通过在注意力输出中去除自身值向量的投影，消除注意力相似性偏差，提升Transformer在序列建模中的上下文处理能力。

**💡 创新点**

创新点在于对标准自注意力进行极简修改：显式剔除与当前token自身值向量同向的分量，使得注意力层专注于上下文信息，减少与FFN层的功能重叠，并保持低额外计算成本。

**🔧 技术方法**

采用了多头因果自注意力的标准实现，并在其基础上加入了投影消除步骤；训练使用NanoGPT框架、RoPE位置编码、LayerNorm、AdamW优化器，以及对模型规模、学习率、序列长度等多维度的实验设置。

**📊 数据集**

主要数据集为FineWeb‑100B（约100 B tokens）进行预训练，随后在ARC‑Easy、BoolQ、HellaSwag、LAMBADA、OpenBookQA、PIQA、SocialIQA、WinoGrande等八个标准下游任务上评估。

**📈 对比分析**

与基线Transformer在三种模型规模（0.7B、1.4B、2.7B）下进行训练/验证损失曲线比较，XSA在所有规模上均显著优于基线；在下游任务中平均准确率提升约1–1.4个百分点；同时XSA在不同学习率、长序列长度和使用注意力sink时均保持稳定或更大优势，且计算开销极小。

**⚠️ 局限性**

局限性包括：实验仅覆盖至2.7B参数级别，未验证更大规模或更多数据；仅针对语言建模任务，未探究对其他任务/模态的适用性；对XSA的理论机理仅给出经验假设，缺乏严格证明；在不同优化器（如MuTant）或更复杂的架构中性能未知。

---

## 471. On the Structural Failure of Chamfer Distance in 3D Shape Optimization

**arXiv ID:** 2603.09925 | [PDF](https://arxiv.org/pdf/2603.09925v1)

**作者:** Chang-Yong Song `[一作]` (Vanderbilt University), David Hyde `[通讯]` (Vanderbilt University)

**通讯引用:** 1083 | [OpenAlex ID](https://openalex.org/A5040432863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对Chamfer距离在点云优化中的梯度结构进行分析，证明其易导致多对一聚类失效，并通过全局耦合（MPM物理先验或共享基底变形）抑制崩塌，并在3D形状变形任务中验证。

**💡 创新点**

发现Chamfer距离优化的崩塌是梯度结构导致的，而非度量本身；提出全局耦合是抑制崩塌的必要条件，并通过理论证明与实验验证。

**🔧 技术方法**

理论分析（Proposition 1-3, Corollary 1），基于MPM的可微分物理先验，全局共享基底变形，点云优化，Chamfer距离、正向/反向损失，逆向梯度裁剪等技术。

**📊 数据集**

使用人工合成形状配对（球体-兔子/牛/鸭/茶壶/龙等）以及Stanford Bunny、Duck、Cow、Teapot、Dragon等公开数据集。

**📈 对比分析**

与直接Chamfer优化（DCO）、密度感知Chamfer（DCD）以及仅物理先验基准进行对比；在20个方向对中，联合物理+Chamfer在多数情况下将两侧Chamfer距离降低1.3–3.0倍，并在龙的案例中实现2.5倍改进。

**⚠️ 局限性**

对复杂拓扑形状需要更高粒子分辨率；目前仅验证在形状变形任务，对生成/填补等其他任务需进一步探索全局耦合实现；物理参数限制导致某些形状出现噪声或NaN；未探讨EMD等其他点级指标。

---

## 472. Real-Time Trust Verification for Safe Agentic Actions using TrustBench

**arXiv ID:** 2603.09157 | [PDF](https://arxiv.org/pdf/2603.09157v1)

**作者:** Tavishi Sharma `[一作]` (Arizona State University), Pragya Sharma `[通讯]` (University of California Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究提出了TrustBench框架，能够在大型语言模型代理生成动作后、执行前进行实时可信度验证，并提供基准评估。

**💡 创新点**

创新点包括双模式架构（基准评估与运行时验证）以及针对医疗、金融等领域的插件化验证逻辑，实现了主动干预而非仅事后评估。

**🔧 技术方法**

主要技术包括LLM-as-a-Judge评分、等距回归自我信心校准、实时低延迟验证流水线和插件化规则引擎。

**📊 数据集**

实验使用了MedQA、FinQA、TruthfulQA等数据集进行评估，并在多规模LLM上测试。

**📈 对比分析**

与无验证的基线相比，TrustBench在多任务上将有害动作减少87%，域插件进一步提升35%，且验证延迟保持在200 ms以下。

**⚠️ 局限性**

局限性在于只覆盖有限领域，插件性能高度依赖域对齐，LLM-as-a-Judge评估的主观性，以及在更大规模部署中的可扩展性和安全性验证尚待进一步验证。

---

## 473. TIMID: Time-Dependent Mistake Detection in Videos of Robot Executions

**arXiv ID:** 2603.09782 | [PDF](https://arxiv.org/pdf/2603.09782v1)

**作者:** Nerea Gallego `[一作]` (University of Zaragoza), Eduardo Montijano `[通讯]` (University of Zaragoza)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于视频异常检测（VAD）的架构 TIMID，用来从弱标注的视频中识别机器人执行过程中出现的时间依赖性错误（即过程性失误）。

**💡 创新点**

创新点包括：①将 VAD 框架迁移到机器人任务执行中，利用任务与失误的文本提示实现时序异常检测；②引入了双模注意力模块（时序上下文 + 语义对齐），实现对多机器人、多时序上下文的捕获；③采用多实例学习（MIL）与对比损失的组合，实现仅视频级标签即可进行帧级异常定位；④构建了一个全新的多机器人仿真数据集，包含可控时间错误和对应的真实执行，用于 sim-to-real 评估。

**🔧 技术方法**

核心技术包括：视频编码器（预训练视频 backbone）、带正弦+高斯位置编码的时序上下文模块、CLIP 文本编码与跨模态注意力对齐、线性分类器、MIL 损失、监督对比损失，以及数据集生成的 LTL‑Büchi 自动机与 Gazebo 仿真。

**📊 数据集**

使用的数据集有：①自研多机器人仿真与真实视频数据集（约 1000 条仿真视频 + 8 条真实视频，涵盖互斥和顺序两类时间错误）；②BridgeData V2（单臂机器人厨房抓取任务，用于检测执行层面错误）。

**📈 对比分析**

与 Auto‑Encoder、Qwen‑2.5（零射击与微调）、PEL4VAD 等基线相比，TIMID 在 Bridge 任务中与 Qwen‑2.5 的性能相近，而在多机器人互斥与顺序任务中均显著优于所有基线（AP 最高可达 76.83，F1 最高 49.1）。在 sim‑to‑real 零射击实验中，TIMID 仍保持最高的 AP 与 F1，证明了其对域漂移的鲁棒性。

**⚠️ 局限性**

局限性：仅能检测单一类型的失误，需在每次失误类型变更时重新训练；训练仍需异常样本，获取困难；未支持多重并发失误，未来可探索无监督或仅正样本的学习方法。

---

## 474. TRIP-Bag: A Portable Teleoperation System for Plug-and-Play Robotic Arms and Leaders

**arXiv ID:** 2603.09226 | [PDF](https://arxiv.org/pdf/2603.09226v1)

**作者:** Noboru Myers `[一作]` (University of Illinois Urbana-Champaign), Joohyung Kim `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1714 | [OpenAlex ID](https://openalex.org/A5089263869)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种可放入行李箱内的便携式遥控操作系统TRIP-Bag，用于在多环境下快速收集高质量的多模态机器人操作演示数据。

**💡 创新点**

将全关节匹配的puppeteer遥控与便携式手持设备相结合，形成一次性组装、零校准、全自主感知的数据采集平台；实现了快速部署（<5 min）和在野外多样场景下的数据收集。

**🔧 技术方法**

基于PAPRLE框架与ROS2实现遥控控制；使用可插拔PAPRAS机械臂与关节匹配的操纵器；摄像头（Intel RealSense D435）与DYNAMIXEL驱动；数据同步采集；利用Action Chunking Transformer (ACT) 进行策略学习。

**📊 数据集**

收集了1238条演示，覆盖22个不同环境，包含两项双臂任务（水果收集、鸡蛋裂开），以及10名非专业用户提供的200条演示。

**📈 对比分析**

通过非专业用户实验评估易用性，成功率随训练递增；通过训练ACT策略验证数据质量，模型能完成任务但精确抓取仍有欠缺；与现有手持或实验室遥控方案相比，TRIP-Bag在部署时长、环境多样性和零校准优势上更具竞争力。

**⚠️ 局限性**

仍需外部电源；任务多样性有限，仅覆盖两项任务；策略在精确抓取与动态环境鲁棒性方面尚需提升。

---

## 475. Towards Viewpoint-centric Artifact-based Regulatory Requirements Engineering for Compliance by Design

**arXiv ID:** 2603.09492 | [PDF](https://arxiv.org/pdf/2603.09492v1)

**作者:** Oleksandr Kosenkov `[一作]` (Blekinge Institute of Technology), Oleksandr Kosenkov `[通讯]` (Blekinge Institute of Technology)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5078091720)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过系统综述、实证研究、访谈与焦点小组等方法，提出并验证了面向视角协同的监管需求工程Artefact Model for Regulatory Requirements Engineering (AM4RRE)，并规划其验证与推广。

**💡 创新点**

① 将法律视角与工程视角融入同一Artefact-based RE模型，实现视角协同；② 设计可定制的调节（T1、T2、T3）以适配不同法规、项目与目标；③ 将法律概念实例化到需求与系统层面并通过案例验证其可行性；④ 将高层目标纳入模型，支持多视角协作与治理。

**🔧 技术方法**

采用Artefact-based RE方法，扩展AMDiRE；使用文本注释、概念实例化、目标驱动评估框架；通过访谈分析软件对访谈与焦点小组数据进行编码与分析。

**📊 数据集**

使用欧洲GDPR文本、欧洲无障碍法等法规文本作为案例；收集的访谈与焦点小组受访者数据（法律专家、系统工程师、产品经理等）作为实证依据。

**📈 对比分析**

与现有隐私设计方法比较时，评估了法律域知识捕获、透明性、可追踪性等特性。通过案例验证与受访者评估，发现AM4RRE在需求完整性和可追踪性上优于传统方法，表现良好。

**⚠️ 局限性**

模型复杂度高，需长期验证；验证案例主要局限于欧盟法规（如GDPR），缺乏跨地区或大规模工业案例；受访者参与受敏感性限制，验证规模受限。

---

## 476. MUGEN: Evaluating and Improving Multi-audio Understanding of Large Audio-Language Models

**arXiv ID:** 2603.09714 | [PDF](https://arxiv.org/pdf/2603.09714v1)

**作者:** Chih-Kai Yang `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**通讯引用:** 9046 | [OpenAlex ID](https://openalex.org/A5040508737)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并构建了 MUGEN 多音频理解基准，设计了 35 个跨七维度的音频选择任务，评估 LALM 在多音频情境下的推理能力，并系统研究了输入规模对性能的影响。

**💡 创新点**

创新点在于：① 通过“音频作为选项”设计迫使模型进行跨音频比较；② 以 35 任务覆盖语义、说话人、情感、时间、环境、音乐与组合音频属性；③ 引入音频排列自一致性（APSC）等训练‑free 方案，有效缓解模型对输入顺序的偏倚。

**🔧 技术方法**

采用了多模型评测（开源 LALM、Gemini-3‑pro 等）、基于 LLM‑as‑judge 的准确率评估、Chain‑of‑Thought、Self‑Consistency 与 Audio‑Permutation Self‑Consistency 等推理策略，并在多音频设置下测评模型的输入规模敏感性。

**📊 数据集**

使用 9,250 条公开音频数据（来自语音、音频、音乐公开语料库及精细控制的合成样本），构成 1,750 条测试实例，覆盖情感、说话人身份、语言、音乐等属性，确保每个任务的候选音频对属性进行系统对比。

**📈 对比分析**

对比方法：在 9 种模型上进行多音频多选题评估，结果显示开源模型平均约 25‑30% 准确率，Gemini‑3‑pro 高阶版可达 69.6%；随着候选数从 2 增至 5，模型准确率下降 20‑50%；APSC 与 CoT 结合可提升约 6.7% 的绝对准确率，显示其在缓解顺序敏感性方面的有效性。

**⚠️ 局限性**

局限性：① 对非语义属性（情感、时间、音乐等）的表现仍远低于语义任务；② 输入规模扩大导致性能显著衰退；③ CoT 在音频感知层面无显著帮助；④ Self‑Consistency 与 APSC 需要多次生成，计算成本较高；⑤ 基准样本仍集中于 35 任务，可能无法完全覆盖所有多音频应用场景。

---

## 477. Prompt-Driven Color Accessibility Evaluation in Diffusion-based Image Generation Models

**arXiv ID:** 2603.09832 | [PDF](https://arxiv.org/pdf/2603.09832v1)

**作者:** Xinyao Zhuang `[一作]`, Kaan Akşit `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种算法在特定任务中的应用。

**💡 创新点**

提出了一种新的优化方法，显著提高了算法的效率。

**🔧 技术方法**

使用了深度学习和机器学习技术。

**📊 数据集**

采用了公开的标准数据集进行实验。

**📈 对比分析**

与现有方法进行了对比，结果显示新方法在准确性和速度上均有提升。

**⚠️ 局限性**

该方法在处理大规模数据时可能会遇到性能瓶颈。

---

## 478. Information Theoretic Bayesian Optimization over the Probability Simplex

**arXiv ID:** 2603.09793 | [PDF](https://arxiv.org/pdf/2603.09793v1)

**作者:** Federico Pavesi `[一作]` (University of Milan), Noémie Jaquier `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 286 | [OpenAlex ID](https://openalex.org/A5064146907)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对概率单纯形上的黑盒函数，提出了一种基于信息几何的贝叶斯优化框架 α-GaBO

**💡 创新点**

创新点在于：①利用 Fisher‑Rao 指标构造球面映射实现单纯形与球面等距；②从球面得到 Matern 核并拉回到单纯形；③基于 α‑连接构建一族可调的采集函数优化器，能够捕捉单纯形的几何结构；④通过 α 参数实现对优化路径的几何调节

**🔧 技术方法**

使用技术包括：信息几何（Fisher‑Rao 指标、α‑连接）、球面映射、Matern 核的谱分解、Riemannian 贝叶斯优化（GP、EI/LCB）、Riemannian 采集函数优化（指数映射、对数映射、信赖域）

**📊 数据集**

实验数据集：Ackley、Rosenbrock、Griewank 三类投影到单纯形的基准函数；混合物组件（混凝土抗压强度 7 维、Olympus 光降解 PCE10/WF3 4 维）；混合分类器（机器人墙跟踪 7 维）；多任务机器人控制（N=4、K=10）

**📈 对比分析**

与受限欧氏 BO（即 BORIS）以及在球面或单纯形上的受限欧氏 BO 进行对比；α-GaBO 在大多数任务中表现更优或相当，收敛更快、样本更高效、最终误差方差更小；在某些边界解任务中，α=-1 模型受限，α=0 取得最佳性能

**⚠️ 局限性**

局限性包括：① α=-1 模型无法到达单纯形边界，导致边界最优解难以发现；② 目前仅针对单纯形和球面等距的情形，缺乏通用的有界流形核构造；③ 需要预先选定 α 参数，可能需要先验知识；④ 只在若干应用场景验证，未覆盖更广泛的类别/高维问题

---

## 479. CLoE: Expert Consistency Learning for Missing Modality Segmentation

**arXiv ID:** 2603.09316 | [PDF](https://arxiv.org/pdf/2603.09316v1)

**作者:** Xinyu Tong `[一作]` (University of Chinese Academy Sciences), Haitao Li `[通讯]` (Zhejiang University)

**通讯引用:** 44886 | [OpenAlex ID](https://openalex.org/A5051089032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了CLoE框架，利用专家一致性学习和一致性驱动的门控融合来解决缺失模态下的多模态医学图像分割问题。

**💡 创新点**

创新点在于把缺失模态鲁棒性视为决策层一致性控制，设计模态专家一致性与区域专家一致性双分支约束，并将一致性映射为可靠性权重用于特征重校准。

**🔧 技术方法**

采用双分支专家一致性学习、cosine相似度一致性度量、轻量门控网络、交叉熵+Dice损失、对比学习等技术。

**📊 数据集**

在BraTS 2020脑肿瘤分割数据集和MSD Prostate前列腺MRI数据集上进行实验。

**📈 对比分析**

与HeMIS、RobustSeg、RFNet、M³AE、DC‑Seg等SOTA方法对比，CLoE在多种缺失模态组合下平均Dice提升至约88.1%（WT）和80.2%（TC），在前列腺PZ也表现最高。

**⚠️ 局限性**

限制在于对极少模态和小样本的鲁棒性仍有限，且对复杂背景的区域一致性约束需要进一步改进。

---

## 480. Local Stability of Rankings

**arXiv ID:** 2603.09724 | [PDF](https://arxiv.org/pdf/2603.09724v1)

**作者:** Felix S. Campbell `[一作]` (Ben-Gurion University of the Negev), Yuval Moskovitch `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5005553562)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并评估局部稳定性度量，用于衡量排名对单个条目的小改动的敏感度。

**💡 创新点**

创新点在于引入局部稳定性概念，考虑稠密区；给出近似可计算的α-局部稳定性；提供采样算法和稠密区检测启发式。

**🔧 技术方法**

采用采样方法、Hoeffding不等式、蒙特卡洛估计、凸多面体体积估算，优化包括减少合理改动、降低重排名成本、α-bounded迭代。

**📊 数据集**

实验使用NBA 2023‑24选手排名、CSRankings大学排名、合成数据；训练LightGBM学习排名函数。

**📈 对比分析**

与全局稳定性对比、基线优化前后速度提升（最高51.6倍），稠密区检测准确率100%，内存/时间提升显著。

**⚠️ 局限性**

局限：计算复杂度高，近似解可能偏差；对非数值/多重属性交互依赖不佳；需手工设定合理改动范围；在高维下采样效率下降。

---

## 481. Modeling Trend Dynamics with Variational Neural ODEs for Information Popularity Prediction

**arXiv ID:** 2603.09148 | [PDF](https://arxiv.org/pdf/2603.09148v1)

**作者:** Yuchen Wang `[一作]` (Northwestern Polytechnical University), Yang Liu `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 49523 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出VNOIP模型，利用双向跳跃ODE和变分神经ODE联合建模级联序列与宏观趋势，实现信息流量的未来流行度预测。

**💡 创新点**

创新点包括：①双向跳跃ODE捕获长程依赖并融合前后信息；②变分神经ODE将微观级联特征与宏观趋势联合编码；③知识蒸馏对先验与后验潜变量进行一致性约束，提升预测稳定性。

**🔧 技术方法**

使用的技术包括：图嵌入（NetSMF、GraphWave）、双向注意力、跳跃ODE、变分推断、神经ODE、截断正态分布生成、知识蒸馏、MLP解码等。

**📊 数据集**

数据集为Twitter（话题推文级联）、APS（物理期刊引用级联）和Weibo（微博转发级联）。

**📈 对比分析**

与DeepCas、DeepHawkes、VaCas、CasFlow、CTCP、CasDo、CasFT等SOTA方法在MSLE/MAPE上进行对比，VNOIP在三大数据集和不同观察窗口下多项指标均低于或接近最优，尤其在Twitter 2天和Weibo 1小时窗口表现显著优于基线。

**⚠️ 局限性**

限制：未显式建模结构不确定性，对极度稀疏或长期预测可能出现误差放大；模型对超参数（λ1、λ2、T等）敏感，需要手动调优。

---

## 482. RTFDNet: Fusion-Decoupling for Robust RGB-T Segmentation

**arXiv ID:** 2603.09149 | [PDF](https://arxiv.org/pdf/2603.09149v1)

**作者:** Kunyu Tan `[一作]`, Mingjian Liang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种三分支编码解码架构，实现RGB-T语义分割在缺失模态时的鲁棒推理。

**💡 创新点**

创新点在于将特征融合与解耦合统一到同一反馈循环中，主要包括Synergistic Feature Fusion (SFF) 交叉门控融合、Cross‑Modal Decouple Regularization (CMDR) 的停止梯度知识蒸馏以及 Region Decouple Regularization (RDR) 的类别‑自适应一致性约束。

**🔧 技术方法**

采用SegFormer Mix‑Transformer 编码器、1×1卷积、通道门控与轻量空间注意力，以及 L2/ L1 损失、stop‑gradient 机制进行端到端训练。

**📊 数据集**

在 MFNet、FMB 与 PST900 三个 RGB‑Thermal 数据集上进行实验。

**📈 对比分析**

与现有多模态鲁棒方法（如 CMNeXt、CRM、StitchFusion 等）相比，本方法在完整模态和单模态（RGB 或 Thermal 缺失）场景下均取得更高的 mIoU，缺失 Thermal 时提升约 4 %（MFNet），缺失 RGB 时提升约 5 %（FMB），并在实时性与 FLOPs 上保持优势。

**⚠️ 局限性**

局限性包括仅验证于 RGB‑Thermal 三分支，未扩展到更多模态；对齐假设仍然存在；并且多分支结构在极低算力设备上的部署仍需进一步压缩。

---

## 483. Serving Compound Inference Systems on Datacenter GPUs

**arXiv ID:** 2603.08797 | [PDF](https://arxiv.org/pdf/2603.08797v1)

**作者:** Sriram Devata `[一作]` (University of Illinois Urbana-Champaign), Sarita Adve `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 14212 | [OpenAlex ID](https://openalex.org/A5086111967)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个名为""的系统，能够在数据中心GPU上高效服务由多个模型组成的复合推理系统；通过联合优化模型变体选择、GPU空间划分与基于任务图的资源分配，实现对端到端延迟、精度与GPU成本的多目标调度。

**💡 创新点**

核心创新在于：①首次将模型准确度缩放、GPU空间划分与任务图信息三项技术整合到同一调度框架；②利用MIG+MPS实现细粒度GPU划分，显著提升GPU利用率；③采用混合整数线性规划（MILP）在可搜索空间内寻找最优配置，并结合动态批处理与早期丢弃机制。

**🔧 技术方法**

技术栈包括：多模型推理任务图；模型变体（不同精度/性能）的预先剖析；NVIDIA GPU的MIG+MPS空间划分；MILP求解器Gurobi用于配置优化；Python实现的服务器框架与批处理/早期丢弃逻辑；以及针对请求流的预测与动态重配置。

**📊 数据集**

实验用数据集：COCO（用于社交媒体与AR助手任务的图像和文本）；Bellevue交通数据集（用于交通分析任务）；Twitter请求时序（用于生成日常请求负载曲线）。

**📈 对比分析**

与多种基线（仅准确度缩放、仅空间划分、仅任务图分配、两项组合等）比较，""在三种应用场景下平均使用43.3% GPU资源，SLO违约率<0.6%；相比最优基线可提升服务吞吐量21.6倍；相比Loki等现有系统可提升11.3倍；基线往往违约率≥10%且资源利用率翻倍。

**⚠️ 局限性**

主要局限：①预先剖析模型变体、MIG配置等需要耗时7–12小时；②MILP求解虽仅2–20s，但在极大规模或频繁重配置时仍可能成为瓶颈；③当前仅支持NVIDIA MIG/MPS，其他GPU厂商需自行实现空间划分；④模型推理假设单GPU完成，无法直接处理需要跨GPU的超大模型；⑤任务图假设为有向无环图，无法处理循环或动态依赖；⑥依赖基于最近5个时段的简单预测，精准度不足时会导致SLO违约。

---

## 484. How Contrastive Decoding Enhances Large Audio Language Models?

**arXiv ID:** 2603.09232 | [PDF](https://arxiv.org/pdf/2603.09232v1)

**作者:** Tzu-Quan Lin `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**通讯引用:** 9046 | [OpenAlex ID](https://openalex.org/A5040508737)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统评估并比较了四种对比解码（Contrastive Decoding）方法在三大类大型音频语言模型上的效果，提出并验证了基于转移矩阵的错误模式分析框架。

**💡 创新点**

首次将对比解码方法与音频感知、推理和多模态任务相结合，揭示不同模型错误类型对方法有效性的影响，并给出基于错误谱选择解码策略的实用指南。

**🔧 技术方法**

采用Audio‑Aware Decoding（AAD）、Audio Contrastive Decoding（ACD）、Audio Minimal Test‑Time Intervention（AMTI）和Decoding by Contrasting Layers（DoLa）四种对比解码技术，并使用LLM‑as‑a‑Judge（GPT‑4o）自动标注错误类型，构建转移矩阵进行分析。

**📊 数据集**

使用SAKURA、MMAU和MMAR三个评测基准（分别涵盖感知、复杂信息提取与深度推理）对模型进行评估。

**📈 对比分析**

与贪婪解码基线相比，AAD和ACD在大多数任务上显著提升性能；效果高度依赖模型架构，Qwen2.5‑Omni取得最大收益，DeSTA2.5‑Audio与Audio Flamingo 3提升有限。

**⚠️ 局限性**

对比解码无法有效纠正推理错误或自信错误断言，仅对“音频盲区”与“猜测”错误有效，限制了其在推理密集任务中的适用性。

---

## 485. Gender Fairness in Audio Deepfake Detection: Performance and Disparity Analysis

**arXiv ID:** 2603.09007 | [PDF](https://arxiv.org/pdf/2603.09007v1)

**作者:** Aishwarya Fursule `[一作]` (Wichita State University), Anderson R. Avila `[通讯]` (Institut national de la recherche scientifique)

**通讯引用:** 449 | [OpenAlex ID](https://openalex.org/A5009407218)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文系统评估了 ASVspoof5 数据集上不同特征（LogSpec、CQT、Wav2Vec、WavLM）与 ResNet-18 模型以及基线 AASIST 在音频深度伪造检测中的性别公平性，并利用五种公平度量揭示了性别差异。

**💡 创新点**

创新点在于首次将多项标准公平度量（统计平等、机会平等、机会平等、预测平等、错误平衡）应用于音频深度伪造检测，揭示了仅用整体 EER 难以捕捉的性别偏差。

**🔧 技术方法**

使用的技术包括 ResNet‑18 分类器、四种特征提取方法（LogSpectrogram、Constant‑Q Transform、Wav2Vec 2.0、WavLM）、Fairlearn 与 AIF360 框架计算公平度量。

**📊 数据集**

使用的数据集为 ASVspoof 5，具有近乎平衡的男女发音样本。

**📈 对比分析**

对比方法为在同一训练/验证/测试分割下，使用相同网络结构与超参数对所有特征进行训练，并在 EER 运营点下计算公平度量；WavLM 在 EER 方面表现最佳（≈21.6%），但多种公平度量仍显示男女间显著差异，AASIST 虽整体 EER 较高但公平度量差异最小。

**⚠️ 局限性**

局限性包括未对导致性别偏差的根本原因进行深入分析；仅评估了现有模型且未提出偏差缓解方案；公平度量的计算仅在单一阈值下进行，缺乏对阈值变化的鲁棒性分析。

---

## 486. LAP: A Language-Aware Planning Model For Procedure Planning In Instructional Videos

**arXiv ID:** 2603.09743 | [PDF](https://arxiv.org/pdf/2603.09743v1)

**作者:** Lei Shi `[一作]` (Örebro University), Stephanie Lowry `[通讯]` (Örebro University)

**通讯引用:** 1385 | [OpenAlex ID](https://openalex.org/A5059929676)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对教学视频中的程序规划任务，提出了一种将视觉观测转换为文本描述后再用扩散模型生成动作序列的语言感知规划模型（LAP）。

**💡 创新点**

创新点在于：① 利用LLM生成丰富、区分度更高的动作文本描述，并用finetuned VLM将视觉映射到文本空间；② 用文本嵌入代替传统视觉嵌入，显著提升规划的区分度；③ 在扩散模型中仅对动作维度加入噪声，保持文本嵌入不变，强化文本指导。

**🔧 技术方法**

核心技术包括：视觉‑语言模型（VLM）finetuning（教授强制 + 计划采样）、LLM文本增强、文本嵌入提取、Denoising Diffusion Probabilistic Model（DDPM）进行动作序列生成、ROUGE 评估文本匹配、基于阈值的动作预测。

**📊 数据集**

实验数据集：CrossTask、Coin、NIV 三个公开程序规划基准。

**📈 对比分析**

与 PDPP、ActionDiffusion、SCHEMA、SkipPlan、KEPP、PlanLLM、MTID 等最新基线相比，LAP 在 SR、mAcc、mSIoU 上均取得显著领先，尤其在 Coin 与 NIV 上的提升幅度最大；在 CrossTask 上提升相对有限。

**⚠️ 局限性**

局限性：① 对 CrossTask 的提升不大，可能因该数据集视觉特征已足够区分；② 当 VLM 无法准确预测起始/目标动作时，规划模型的鲁棒性下降；③ 对文本生成依赖 LLM 的质量与成本，且需要额外的 finetuning 资源。

---

## 487. Evaluating the Practical Effectiveness of LLM-Driven Index Tuning with Microsoft Database Tuning Advisor

**arXiv ID:** 2603.09181 | [PDF](https://arxiv.org/pdf/2603.09181v1)

**作者:** Xiaoying Wang `[一作]` (Microsoft Research), Surajit Chaudhuri `[通讯]` (Microsoft Research)

**通讯引用:** 23843 | [OpenAlex ID](https://openalex.org/A5038037154)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估大语言模型在数据库索引调优中的实用性，并与微软SQL Server的数据库调优顾问（DTA）进行对比。

**💡 创新点**

揭示LLM能在单查询和多查询工作负载中产生与DTA相当或更优的索引配置，提出从LLM推理中提炼规则的可行性，并指出其高方差与成本估算误差问题。

**🔧 技术方法**

使用GPT‑5（大语言模型）作为索引建议引擎，结合何‑if API、成本估算、工作负载压缩等技术。

**📊 数据集**

基于TPC‑H（sf=10）以及四个真实企业客户工作负载（Real‑D、Real‑M、Real‑R、Real‑S）。

**📈 对比分析**

通过在同一硬件上执行每个查询的实际耗时比较，发现LLM在大部分单查询场景中可匹配或超越DTA，但在多查询场景下表现波动，最高可提升数倍但也可能导致性能回退。

**⚠️ 局限性**

主要限制是LLM结果的高方差、对成本估算误差敏感、需要昂贵的性能验证、直接集成到DTA时易导致性能下降，以及在大型多查询场景下容易被“分心”。

---

## 488. A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System

**arXiv ID:** 2603.09727 | [PDF](https://arxiv.org/pdf/2603.09727v1)

**作者:** Luyao Zou `[一作]` (Sungkyunkwan University), Zhu Han `[通讯]` (University of Houston)

**通讯引用:** 89279 | [OpenAlex ID](https://openalex.org/A5063667378)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多原型引导的联邦知识蒸馏方法（MP‑FedKD），用于解决 AI‑RAN 边缘计算系统中的非 IID 数据问题。

**💡 创新点**

创新点在于：① 结合自知识蒸馏（SKD）作为教师模型；② 设计条件层次凝聚聚类（CHAC）实现多原型生成；③ 引入原型对齐机制，避免全局原型平均导致的信息损失；④ 基于 CORE 的 LEMGP 损失实现同类吸引与异类排斥。

**🔧 技术方法**

技术手段包括：联邦学习（FedAvg）、知识蒸馏、CHAC、原型对齐、LEMGP 损失、ResNet‑10/S‑CNN 等深度网络架构。

**📊 数据集**

使用的公开数据集有：CIFAR‑10、MNIST、Fashion‑MNIST、EuroSAT，以及两种组合数据集 M+F（MNIST+Fashion‑MNIST）和 C+E（CIFAR‑10+EuroSAT）。

**📈 对比分析**

与 FedProx、FedProto、FedAS、MOON、E‑FPKD、FedALA 等基线在多种 Dirichlet 非 IID 设置下进行对比，MP‑FedKD 在准确率、平均准确率、RMSE 和 MAE 上普遍优于基线，提升幅度可达 1.98%–28.70%。

**⚠️ 局限性**

局限性：① 需要额外的聚类与对齐计算，时间复杂度较高；② 实验仅在模拟环境中验证，未深入评估实际无线网络中的通信成本与延迟；③ 对聚类数、学习率等超参数敏感，需要进一步调优。

---

## 489. FedLECC: Cluster- and Loss-Guided Client Selection for Federated Learning under Non-IID Data

**arXiv ID:** 2603.08911 | [PDF](https://arxiv.org/pdf/2603.08911v1)

**作者:** Daniel M. Jimenez-Gutierrez `[一作]` (Sapienza University of Rome), Andrea Vitaletti `[通讯]` (Sapienza University of Rome)

**通讯引用:** 1882 | [OpenAlex ID](https://openalex.org/A5070969466)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了FedLECC，一种在跨设备联邦学习中针对严重label skew非IID数据的簇与损失引导的客户端选择方法，旨在提升模型精度并降低通信开销。

**💡 创新点**

创新点在于将标签分布相似度聚类（使用Hellinger距离+OPTICS）与基于本地损失的优先级选择相结合，既保证客户端多样性，又聚焦于高损失、信息量大的设备，显著提升收敛速度与最终准确率。

**🔧 技术方法**

技术栈包括：Hellinger距离衡量标签分布相似度、OPTICS聚类算法、损失导向的选取策略、标准FedAvg聚合、轻量级客户端标签直方图上传，且不修改本地训练流程。

**📊 数据集**

实验数据集：MNIST与Fashion-MNIST（FMNIST），采用Dirichlet(α)划分实现高label skew（HD≈0.9）。

**📈 对比分析**

与FedAvg、FedProx、FedNova、FedDyn、HACCS、FedCLS、FedCor、POC等多种基线在相同实验设置下对比，FedLECC在严重label skew下最高提升12%的测试准确率，通信轮数减少约22%，总通信量下降最多50%。

**⚠️ 局限性**

局限性包括：对聚类数量、参与客户端数等超参数敏感，需要手动调节；缺乏正式收敛性证明；隐私泄露风险尚未彻底解决（需进一步集成DP/SMC）；在极端资源受限场景下的鲁棒性尚待验证。

---

## 490. Understanding the Use of a Large Language Model-Powered Guide to Make Virtual Reality Accessible for Blind and Low Vision People

**arXiv ID:** 2603.09964 | [PDF](https://arxiv.org/pdf/2603.09964v1)

**作者:** Jazmin Collins `[一作]` (Cornell University), Shiri Azenkot `[通讯]` (Cornell Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并评估了一个基于大型语言模型（LLM）的AI导游，用于社交VR环境中帮助盲/低视力（BLV）用户进行导航、视觉信息获取与社交互动。

**💡 创新点**

首次将可定制的人格化具身导游与LLM结合，并揭示了用户在单人和社交情境下对AI导游的情感化行为、角色扮演与使用差异。

**🔧 技术方法**

采用Unity搭建VR场景，利用Meta Quest 2进行交互；后台使用OpenAI的Whisper进行语音识别，GPT‑4生成文本，再通过Text‑to‑Speech合成语音；环境截图作为上下文输入LLM。

**📊 数据集**

使用自建的两款虚拟公园环境（来源于前人研究）和16名BLV参与者的实验数据，未使用公开大规模视觉或语音数据集。

**📈 对比分析**

通过对参与者问答准确率（63.2%）、响应时长（约6–11秒）以及对照前人人类导游的情感连结结果进行评估，显示AI导游在导航与信息提供上有效，但准确率和延迟仍显不足。

**⚠️ 局限性**

局限性包括较高的系统延迟、回答准确率低、缺乏实时流式语音、对用户提示的理解不完整、缺少记忆与预防功能，以及受GPT‑4/Whisper技术约束，未来需要更高效、更智能的LLM与语音交互架构。

---

## 491. Game-Theoretic Modeling of Stealthy Intrusion Defense against MDP-Based Attackers

**arXiv ID:** 2603.09587 | [PDF](https://arxiv.org/pdf/2603.09587v1)

**作者:** Willie Kouam `[一作]` (Johannes Kepler University Linz), Stefan Rass `[通讯]` (Johannes Kepler University Linz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在网络中已渗透的APT攻击下，基于攻击图的游戏理论模型，评估防御者在不同信息水平下的最优部署策略。

**💡 创新点**

引入MDP框架捕捉攻击者的自适应决策，并在三种信息情景（完全信息、盲目、Dirichlet不确定）下求解最优策略，证明Dirichlet鲁棒策略优于Stackelberg。

**🔧 技术方法**

采用线性规划、混合整数线性规划和蒙特卡罗采样的组合方法求解MDP和游戏。

**📊 数据集**

在Unguard虚拟网络、MARA和MiR100三套真实攻击图上进行实验。

**📈 对比分析**

与最短路径和随机启发式对比，结果显示最优策略将成功率降低至1/3以内，尤其在高多样性网络中效果显著。

**⚠️ 局限性**

主要限制包括假设检测器完美可靠、缺乏实时反馈、并且攻击图为无环且已知；未考虑动态信息更新。

---

## 492. Walking on Rough Terrain with Any Number of Legs

**arXiv ID:** 2603.09147 | [PDF](https://arxiv.org/pdf/2603.09147v1)

**作者:** Zhuoyang Chen `[一作]` (University of Michigan), Shai Revzen `[通讯]` (University of Michigan)

**通讯引用:** 1136 | [OpenAlex ID](https://openalex.org/A5056552261)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于分段有限状态机的多足机器人粗糙地形行走控制器，并在3-16条腿的模型上通过仿真验证。

**💡 创新点**

创新点是将离散事件触发的有限状态机与分段耦合，实现了与CPG和WalkNet兼容的、可伸缩的、低计算量的分布式行走模式，同时能在无接触时产生拟像步态。

**🔧 技术方法**

使用了有限状态机控制、分段耦合的前后关节交互、PD控制与速度轨迹生成、MuJoCo 仿真与对接，并通过脚接触开关实现本地触发。

**📊 数据集**

使用的“数据集”为合成地形（平坦、浮动、随机高斯噪声坡度、阶梯）以及随机初始相位，未使用真实物理实验数据。

**📈 对比分析**

通过与同类连续CPG和WalkNet等控制方案的对比（主要是同步收敛速度、地形鲁棒性、身体姿态稳定性），结果显示该FSM控制器在相同参数下能快速收敛至交替三足节拍，并在所有地形保持较小姿态波动，表现出与现有方法相当甚至更好的鲁棒性。

**⚠️ 局限性**

局限性包括：仿真中使用理想化的背骨柔性铰链和摩擦模型；未考虑真实传感器误差和地面摩擦变化；脚接触检测仅靠开关，易受损；缺乏在线足部规划与视觉感知；在硬件上仍需验证。

---

## 493. VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM

**arXiv ID:** 2603.09673 | [PDF](https://arxiv.org/pdf/2603.09673v1)

**作者:** Anh Thuan Tran `[一作]` (George Mason University), Jana Kosecka `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 VarSplat，一种基于 3D Gaussian Splatting 的不确定性感知 RGB‑D SLAM 系统，在线学习每个 splat 的外观方差并渲染像素级不确定度，用以增强跟踪、配准和回环检测。

**💡 创新点**

创新点在于显式学习并传播每个 Gaussian 的外观不确定性，通过总方差定律与 alpha 混合在一次渲染中得到可微像素不确定度，并将其统一作为权重应用于跟踪、配准和回环。

**🔧 技术方法**

使用技术包括 3D Gaussian Splatting、Spherical Harmonics、单通道可微渲染、alpha 混合、总方差定律、深度残差与光度残差的负对数似然损失、以及子图（submap）增量式 SLAM 框架。

**📊 数据集**

在四个数据集上评估：Replica（合成）、TUM‑RGBD、ScanNet 以及 ScanNet++（真实场景）。

**📈 对比分析**

与现有 NeRF 与 3DGS‑SLAM 基线相比，VarSplat 在跟踪精度、全局一致性、重建质量和新视角渲染方面均实现了最优或相当的性能，平均轨迹误差降低约 10%–18%，渲染 PSNR 提升数分。

**⚠️ 局限性**

局限性包括：仍需额外 GPU 内存以存储方差参数；在极低纹理或极高噪声场景下方差估计可能不够精确；系统对深度传感器误差仍有一定敏感性，未来可结合更鲁棒的几何不确定性建模。

---

## 494. FrameDiT: Diffusion Transformer with Frame-Level Matrix Attention for Efficient Video Generation

**arXiv ID:** 2603.09721 | [PDF](https://arxiv.org/pdf/2603.09721v1)

**作者:** Minh Khoa Le `[一作]` (Deakin University), Truyen Tran `[通讯]` (Deakin University)

**通讯引用:** 6598 | [OpenAlex ID](https://openalex.org/A5085471517)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种名为Matrix Attention的帧级注意力机制，并基于该机制设计了FrameDiT-G和FrameDiT-H两种Diffusion Transformer架构，用于高保真视频生成。

**💡 创新点**

创新点在于将注意力从token级迁移到帧级，通过矩阵原生运算生成查询、键、值，既能捕获全局时空结构，又保持低复杂度；同时引入全局‑局部混合（Hybrid）策略，实现大运动与细粒度运动的双重建模。

**🔧 技术方法**

核心技术包括扩散模型、Diffusion Transformer、Matrix Attention（矩阵级查询/键/值生成）、多头Matrix Attention、局部因式化注意力、行权重矩阵U的softmax归一化以及门控融合。

**📊 数据集**

实验覆盖UCF-101、Sky‑Timelapse、Taichi‑HD、FaceForensics以及用于文本‑视频的Pexels‑400K和VBench数据集。

**📈 对比分析**

通过FVD、FVMD、FID等指标与Local Factorized Attention、Full 3D Attention以及多款GAN/扩散视频生成模型对比，FrameDiT‑G在保持效率的同时显著提升时空连贯性，FrameDiT‑H在多数据集上实现SOTA性能（FVD下降9–39%），同时保持与全3D相近的计算成本。

**⚠️ 局限性**

局限性包括：全局‑局部混合仍需预训练局部分支，导致训练初期梯度不平衡；在某些极大运动或长序列场景下仍略逊于纯Full 3D Attention；对数据规模和模型容量依赖较大，进一步提升需更大规模训练。

---

## 495. Stein Variational Ergodic Surface Coverage with SE(3) Constraints

**arXiv ID:** 2603.09458 | [PDF](https://arxiv.org/pdf/2603.09458v1)

**作者:** Jiayun Li `[一作]` (Technische Universität Darmstadt), Georgia Chalvatzaki `[通讯]` (Technische Universität Darmstadt)

**通讯引用:** 810 | [OpenAlex ID](https://openalex.org/A5026055366)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于SE(3) Stein变分梯度下降的预条件化算法TSVEC，用于在离散点云表面上生成覆盖轨迹，并在机器人表面绘制任务中实现了高质量覆盖。

**💡 创新点**

创新点在于将点云遍历覆盖转化为流形感知的采样问题，推导出适用于SE(3)的SVGD粒子更新，并引入Gauss–Newton式预条件器以显著提升优化收敛性。

**🔧 技术方法**

采用了SE(3) Stein Variational Gradient Descent（SVGD）、流形感知采样、预条件化的Gauss–Newton迭代、离散点云采样以及机器人运动学逆解技术。

**📊 数据集**

使用了3D点云表面覆盖基准数据集（包括合成与真实点云）以及实际机器人表面绘制实验中的手绘字符数据。

**📈 对比分析**

与IPOPT、L-BFGS以及SE(3)-aware Gauss–Newton等传统优化基线进行比较，TSVEC在覆盖质量、最终目标值和收敛速度上均优于这些方法，并在真实机器人实验中实现了可辨识的字符轨迹。

**⚠️ 局限性**

该方法仍受限于粒子数和预条件器设置，对极端高维或极度非凸表面分布的收敛性能有一定限制；计算量随粒子数和轨迹长度增长，且在极高实时性要求的场景下可能不如传统梯度下降方法。

---

## 496. A PTAS for Weighted Triangle-free 2-Matching

**arXiv ID:** 2603.09144 | [PDF](https://arxiv.org/pdf/2603.09144v1)

**作者:** Miguel Bosch-Calvo `[一作]` (IDSIA), Takashi Noguchi `[通讯]` (RIMS)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文提出了一个多项式时间逼近方案（PTAS）用于加权三角形自由2匹配（WTF2M）问题，显著优于之前仅能达到2/3逼近的贪心方法。

**💡 创新点**

创新点主要在于：① 使用简单的局部搜索算法；② 通过对禁止三角形的泛化（𝒯-自由）构建递归图变换，获得可证的增量提升路径；③ 将权重离散化为有限整数后，保证多项式时间的搜索上界。

**🔧 技术方法**

核心技术包括：图的局部搜索、对称差的可交替路径分析、权重缩放与整数化、递归图拆分与合并、以及对3-环的禁用约束的数学证明。

**📊 数据集**

该工作为理论研究，没有使用实际数据集；评估完全基于算法复杂度与逼近比率的理论分析。

**📈 对比分析**

与传统的2/3近似相比，所提出的PTAS可以在给定任意常数ε>0时得到(1-ε)逼近解，且在权重整数化后运行时间为n^{O(1/ε)}。

**⚠️ 局限性**

局限性：① 算法的运行时间对ε依赖指数，实际实现效率可能不高；② 目前仍未给出多项式时间精确解，WTF2M的NP难度仍未知；③ 仅在无并行边、最多自环的图上有效，若图结构更复杂需进一步研究。

---

## 497. Where, What, Why: Toward Explainable 3D-GS Watermarking

**arXiv ID:** 2603.08809 | [PDF](https://arxiv.org/pdf/2603.08809v1)

**作者:** Mingshu Cai `[一作]` (Waseda University), Yixuan Li `[通讯]` (Nanyang Technological University)

**通讯引用:** 1981 | [OpenAlex ID](https://openalex.org/A5100443481)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在3D Gaussian Splatting模型中实现了一种鲁棒且不可察觉的数字水印方案。

**💡 创新点**

通过Trio-Experts模块提取几何、外观和冗余先验，配合SBAG进行安全预算驱动的载体选择，并在训练中使用解耦的通道组掩码实现视觉与水印目标的梯度隔离，最终实现了高质量与高鲁棒性的平衡。

**🔧 技术方法**

采用3D Gaussian Splatting、离散小波变换（DWT）、EOT对抗训练、Trio-Experts三重专家、SBAG安全预算门控和通道组掩码等技术。

**📊 数据集**

在Blender、LLFF和Mip-NeRF 360三个公开3D场景数据集上进行实验，并使用32/48/64位消息容量进行评估。

**📈 对比分析**

与WateRF+3D-GS、GuardSplat、3D-GSW等基线相比，在PSNR上提升0.83dB，位准确率提高1.24%，并在多种图像与模型空间攻击下保持更高的鲁棒性和视觉质量。

**⚠️ 局限性**

需要手动调节损失权重以平衡质量与鲁棒性，并依赖预训练的解码器，极端配置可能导致性能下降。

---

## 498. No Cliques Allowed: The Next Step Towards BDD/FC Conjecture

**arXiv ID:** 2603.09558 | [PDF](https://arxiv.org/pdf/2603.09558v1)

**作者:** Lucas Larroque `[一作]` (Inria), Michaël Thomazo `[通讯]` (Inria)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

证明了在UCQ可重写的存在规则集合中，通用模型不可能包含任意大小的有向团（tournament）而不必蕴含自环查询∃xE(x,x)。

**💡 创新点**

通过对规则集合进行系统的“规则手术”和精细的模型论分析，缩小了对有限可控性猜想（BDD）的潜在反例空间，并提出了新的模型论性质；同时，首次将Ramsey定理与“谷查询（valley query）”结合应用于存在规则的结构分析。

**🔧 技术方法**

主要使用了Chase算法、UCQ重写、规则集合的前沿（frontier）和前向存在性（forward-existential）以及谓词唯一性（predicate-unique）等属性；在证明中还引入了Ramsey定理、稀疏图与“谷查询”概念，并利用多重集与时间戳等技术构造结构。

**📊 数据集**

本工作为理论性研究，未使用任何实验数据集。

**📈 对比分析**

论文未提供实验对比，而是通过理论证明表明相较于以往仅适用于二元签名或单头规则的结果，该结论适用于更广泛的UCQ可重写规则集合。

**⚠️ 局限性**

局限性在于仅解决了关于有向团的性质，未能推广到更一般的图色数或更大规模结构；此外，虽然为BDD猜想提供了新的视角，但并未完全证明该猜想。

---

## 499. ALARM: Audio-Language Alignment for Reasoning Models

**arXiv ID:** 2603.09556 | [PDF](https://arxiv.org/pdf/2603.09556v1)

**作者:** Petr Grinberg `[一作]` (École Polytechnique Fédérale de Lausanne), Hassan Shahmohammadi `[通讯]` (Sony Europe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 ALARM 框架，将大语言模型与音频理解结合，使用自重述生成音频兼容的目标文本，并通过多编码器融合实现强大的多模态推理。

**💡 创新点**

创新点包括：自重述机制克服 reasoning LLM 的文本暴露问题；去除 ASR 依赖，使用多编码器（Whisper、W2V‑BERT‑2.0、MuQ、SSLAM）通过交叉注意力和 Perceiver 进行高效融合；冻结 LLM 仅训练适配器，保持文本能力；以及压缩多码率到 25/50Hz 的特征表示。

**🔧 技术方法**

采用 Qwen3‑4B‑Thinking‑2507 作为冻结 RLM，Qwen3‑30B‑Instruct 生成数据；构建 6M 实例多任务语料；使用卷积+MLP 适配器、交叉注意力（ALARM‑CA）、Perceiver（ALARM‑P）和融合压缩（ALARM‑E）等技术；训练采用 AdamW+cosine 调度。

**📊 数据集**

训练数据包括 19K 小时的语音、音乐和环境音，来源于 Cameo、VoxCeleb、GTZAN、ESC50、AudioSet 等多种公开数据集，配合 2.5M 个独特提示，形成 6M 实例的多任务语料库；同时采集 HeySQuAD（人类子集）与 Instructs2s 的 423K 示例。

**📈 对比分析**

通过与众多公开 ALM（如 GPT‑4o‑Audio、GAMA、Qwen2‑Omni、Audio Flamingo 等）在 MMAU、MMAR、MMSU 等基准上对比，ALARM‑E 仅 4B 参数即可在 MMAU‑speech 上取得最优开源结果，MMSU 逻辑推理排名第三，且在保持文本能力的前提下，训练成本与数据量显著低于同类更大模型。

**⚠️ 局限性**

主要局限包括：仍需多编码器的额外计算和内存开销；自重述虽缓解文本暴露，但对部分推理细节仍有误差；在音乐/声学任务中性能略逊于专用单编码器模型；对不同 LLM 的迁移性尚未充分验证；数据集虽大但仍可能存在未覆盖的音频场景。

---

## 500. Entangling Like Mycorrhizae: Mixing Realities Through Touch in "FungiSync"

**arXiv ID:** 2603.09272 | [PDF](https://arxiv.org/pdf/2603.09272v1)

**作者:** Botao Amber Hu `[一作]` (Reality Design Lab), Rem RunGu Lin `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 28 | [OpenAlex ID](https://openalex.org/A5007469433)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实施了一个多用户混合现实仪式FungiSync，利用MR头戴式掩护让参与者以植物的视角体验真菌网络，通过身体接触实现虚拟世界的混合与资源交换。

**💡 创新点**

将真菌共生网络的生态互联映射为身体触碰引发的感官混合，创造了以体感为中心的共生仪式，将生物学隐喻转化为可交互的沉浸体验。

**🔧 技术方法**

基于HoloKit X（iPhone ARKit）硬件、Unity VFX、光学手部跟踪、实时音频响应与分布式MR同步，结合自制木质蘑菇头盔和自定义shader实现视觉泄漏与融合。

**📊 数据集**

无传统数据集，使用实时摄像机与LiDAR重建环境、麦克风频谱分析做音频驱动，资源符号（氮、磷、糖等）为自定义几何模型。

**📈 对比分析**

通过用户访谈、观摩记录评估感官体验与生态理解，未提供客观性能指标；系统实现实时手部交互与视觉混合，帧率保持在30fps以上，延迟低于100ms。

**⚠️ 局限性**

受硬件限制仅支持少量参与者，触碰检测依赖空间跟踪误差，资源交互规则过于简单，缺乏长期生态反馈与可扩展性，实验规模受限且无量化指标。

---

## 501. Sim2Act: Robust Simulation-to-Decision Learning via Adversarial Calibration and Group-Relative Perturbation

**arXiv ID:** 2603.09053 | [PDF](https://arxiv.org/pdf/2603.09053v1)

**作者:** Hongyu Cao `[一作]` (Arizona State University), Yanjie Fu `[通讯]` (Arizona State University)

**通讯引用:** 6233 | [OpenAlex ID](https://openalex.org/A5032187620)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在数字孪生的模拟-决策框架下，提出Sim2Act以提升模拟器与决策策略的鲁棒性；

**💡 创新点**

通过对决策关键状态-动作对的对抗校准来聚焦误差，并使用基于组相对扰动的策略学习来避免过度保守；

**🔧 技术方法**

对抗式模拟器校准（minimax重加权）、组相对扰动（group‑relative advantage loss）、LSTM编码解码模拟器、Gaussian潜在扰动、强化学习策略优化；

**📊 数据集**

DataCo、GlobalStore、OAS三大供应链数据集；

**📈 对比分析**

与Sim2Dec、Markov、Prediction‑based、Generation‑based模拟器及LP、DQN、PPO、ChatGPT‑3.5、RARL、EPOpt等决策基线比较，Sim2Act在最差情况准确率、方差、CVaR以及平均收益等指标上均优于所有基线；

**⚠️ 局限性**

对抗校准与组相对扰动参数对性能敏感，实验主要基于供应链数据，缺乏对更复杂物理约束或多任务场景的验证。

---

## 502. Measuring onion website discovery and Tor users' interests with honeypots

**arXiv ID:** 2603.09329 | [PDF](https://arxiv.org/pdf/2603.09329v1)

**作者:** Arttu Paju `[一作]` (Tampere University), Juha Nurmi `[通讯]` (Tampere University)

**通讯引用:** 1363 | [OpenAlex ID](https://openalex.org/A5024596776)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

对八类暗网主题的自建诱饵网站进行部署，利用Ahmia搜索引擎、粘贴服务和另一渠道分发链接，并通过CAPTCHA和登录尝试记录人类访问与交互数据。

**💡 创新点**

首次通过诱饵实验量化实际用户在Tor上的发现和交互行为，揭示用户对不同暗网主题和语言的真实兴趣，而非仅靠爬虫分析。

**🔧 技术方法**

构建简约网页诱饵、部署CAPTCHA验证、记录HTTP日志并对访问来源、CAPTCHA完成率及登录尝试进行统计分析。

**📊 数据集**

使用了2025年3月24日至4月17日间生成的96个独立.onion地址的访问日志，包括访问量、CAPTCHA解决数和登录尝试数，来自Ahmia、两种粘贴服务和第三渠道。

**📈 对比分析**

通过对来源、主题类别和语言的访问量、CAPTCHA完成率以及登录尝试数进行对比，发现Ahmia是主要的人类流量来源，CSAM主题登录率最高（115.55%），且英语占比最高，整体方法证明了不同渠道和主题对用户兴趣的显著差异。

**⚠️ 局限性**

研究样本受限于仅使用过滤的Ahmia搜索引擎、有限的语言覆盖、诱饵网站的低真实度以及大多数非人类流量来自粘贴服务，因而难以代表整个暗网生态或不受过滤渠道的用户行为。

---

## 503. Emotion is Not Just a Label: Latent Emotional Factors in LLM Processing

**arXiv ID:** 2603.09205 | [PDF](https://arxiv.org/pdf/2603.09205v1)

**作者:** Benjamin Reichman `[一作]` (Georgia Institute of Technology), Larry Heck `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6434 | [OpenAlex ID](https://openalex.org/A5003679010)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究情绪作为潜在因子对大型语言模型（LLM）注意力分布和阅读推理的影响，并提出了情绪平衡的问答数据集AURA-QA以及一种情绪正则化训练框架。

**💡 创新点**

创新点在于：①将情绪视为隐含表示的系统性变化而非单纯的预测目标；②通过多维注意力几何特征量化情绪对模型关注模式的影响；③构造情绪平衡的自然文本QA数据集；④提出情绪正则化损失，约束情绪驱动的表示漂移。

**🔧 技术方法**

使用的技术包括：Transformer注意力几何特征（如中心质量距离、熵、曲率等）、基于SVD的情绪潜在空间、情绪分类器（对原始文本标注）、LoRA微调与情绪正则化损失、LLM生成与验证（生成QA、情绪标签验证）、统计与交叉验证评估。

**📊 数据集**

所用数据集包括：RefinedWeb（情绪分布分析）、TweetQA、FriendsQA、Natural Questions、以及本文新构建的AURA-QA（基于Project Gutenberg文本的情绪平衡QA），此外还使用了原始网页文本和多模型生成的情绪标签。

**📈 对比分析**

通过5折交叉验证、logistic回归与随机森林评估注意力特征与准确率的相关性；在LoRA微调对比实验中，将情绪正则化与无正则化、多情绪增广等方案进行比较。实验结果表明，情绪正则化在多模型、多数据集上平均提升1–3%的准确率，尤其在分布漂移（out‑of‑distribution）情况下显著提升鲁棒性。

**⚠️ 局限性**

限制：AURA-QA的数据构建依赖LLM生成与验证，可能带来模型偏差；情绪正则化仅针对情绪-语义表示漂移，未覆盖所有误差源；在域内评估中，正则化的收益表现不够稳定，且情绪标签仍为弱监督。

---

## 504. MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data

**arXiv ID:** 2603.09206 | [PDF](https://arxiv.org/pdf/2603.09206v1)

**作者:** Zongxia Li `[一作]` (University of Maryland), Fuxiao Liu `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MM-Zero框架，实现了从零数据开始的Vision Language Models自我进化；

**💡 创新点**

创新点在于引入三角色（Proposer、Coder、Solver）协同工作，并设计可验证奖励机制与GRPO优化，首次实现零数据VLM自我进化；

**🔧 技术方法**

采用强化学习可验证奖励（RLVR）与Group Relative Policy Optimization (GRPO)，结合代码生成、SVG渲染与多模态推理；

**📊 数据集**

不使用任何外部数据，全部生成自有视觉内容与问题，原始训练仅用模型自身生成的四元组；

**📈 对比分析**

在多种视觉推理基准（MMMU、ChartQA、MathVerse等）上与基线模型对比，迭代训练后平均准确率提升3-6个百分点，尤其在视觉数学任务上显著提升；

**⚠️ 局限性**

局限性包括对大规模模型（如38B参数）缺乏验证，渲染成功率与多模态推理仍受限于模型初始能力，且依赖规则验证奖励可能无法覆盖所有复杂场景。

---

## 505. Predictive Control with Indirect Adaptive Laws for Payload Transportation by Quadrupedal Robots

**arXiv ID:** 2603.08831 | [PDF](https://arxiv.org/pdf/2603.08831v1)

**作者:** Leila Amanzadeh `[一作]` (Virginia Tech), Kaveh Akbari Hamed `[通讯]` (Virginia Tech)

**通讯引用:** 1129 | [OpenAlex ID](https://openalex.org/A5066501467)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一套分层的自适应规划与控制框架，结合梯度下降自适应律与凸MPC，用于四足机器人在粗糙地形上携带未知重物的鲁棒运输。

**💡 创新点**

创新点包括：①提出了间接梯度下降自适应估计法，并将其收敛性通过凸不等式约束嵌入MPC；②首次将该自适应MPC应用于单刚体(SRB)模板模型，并通过低层非线性全身控制器实现轨迹跟踪；③提供了正式的状态估计误差渐近稳定性分析与可实现的凸约束。

**🔧 技术方法**

主要技术手段有：凸优化(MPC)与QP；梯度下降自适应更新；单刚体模板动力学建模；全身控制器(QP+虚拟约束)；状态与输入约束、线性化、特征矩阵构造；使用qpSWIFT求解器；RaiSim物理仿真。

**📊 数据集**

实验数据集包括：A1四足机器人在室内外平地与粗糙地形（木块、草地、砾石）上的硬件试验；1500个随机生成的高度图（每个30m长，随机放置5cm高方块）用于仿真成功率评估；以及动态携带物体和推力扰动的实验场景。

**📈 对比分析**

与普通MPC、ℒ1-MPC进行对比。实验显示：在1500随机地形上，AMPC成功率达88.22%，而普通MPC仅18.89%；在粗糙地形上携带109%、91%不确定性的静态负载；动态负载下AMPC可携带约79%质量，普通MPC仅41%。在速度跟踪方面，AMPC在10kg负载时仍能保持稳定，普通MPC失效。性能显著优于对比方法。

**⚠️ 局限性**

局限性包括：仅在线性化的SRB模板上工作，未考虑跨步期间支撑腿的切换；自适应律未强制物理一致性约束（如惯性矩阵正定性）；对大范围非线性/高频变化的鲁棒性尚未验证；未来计划扩展至非线性AMPC并加入腿步切换的稳定性分析。

---

## 506. A Guideline-Aware AI Agent for Zero-Shot Target Volume Auto-Delineation

**arXiv ID:** 2603.09448 | [PDF](https://arxiv.org/pdf/2603.09448v1)

**作者:** Yoon Jo Kim `[一作]` (Oncosoft Inc.), Jin Sung Kim `[通讯]` (Oncosoft Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了一种名为OncoAgent的AI代理框架，能够在无训练数据的前提下，将自然语言的临床指南自动转换为患者的三维靶区（CTV/PTV）轮廓。

**💡 创新点**

创新点在于：①将临床指南文本直接转化为可执行的工具调用计划，实现真正的零样本（zero-shot）靶区自动勾画；②通过LLM生成可审计、可解释的计划，为医师提供审核和修改的空间；③实现对指南更新和不同解剖部位的即时适配，无需重新标注或重新训练。

**🔧 技术方法**

采用的大型语言模型（LLM）+ 预训练解剖结构分割模型 + 形态学与布尔运算工具组成的两阶段（规划-执行）代理体系；系统提示设计、工具调用序列化、迭代自校验等技术。

**📊 数据集**

使用了40例中胸部食管癌的放疗CT扫描及其由经验放射科医师标注的GTV、CTV、PTV轮廓；此外在其他食管指南（CROSS、JASTRO 2024）和前列腺（RTOG 0126）上进行零样本迁移测试。

**📈 对比分析**

通过与三种监督式深度学习基线（nnU-Net、DDAU-Net、nnU-Net GTV Prior）在Dice、MSD、敏感度、精确度等指标上对比；OncoAgent在CTV的DSC 0.842、PTV 0.880 与监督模型相当，并在临床盲测中获得更高的指南符合度、修改难度与临床可接受度评分。

**⚠️ 局限性**

局限性包括：①性能高度依赖预训练OAR分割模型的质量；②LLM可能出现语义误解或幻觉，需人工审阅；③严格的几何约束可能忽略呼吸、吞咽等运动导致的安全边距；④在单中心40例数据上验证，尚需多中心、多部位的进一步评估。

---

## 507. VeriInteresting: An Empirical Study of Model Prompt Interactions in Verilog Code Generation

**arXiv ID:** 2603.08715 | [PDF](https://arxiv.org/pdf/2603.08715v1)

**作者:** Luca Collini `[一作]` (New York University), Ramesh Karri `[通讯]` (New York University)

**通讯引用:** 16662 | [OpenAlex ID](https://openalex.org/A5059648257)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 18 种语言模型在 Verilog 代码生成任务上的表现进行大规模、系统的实验评估，并探索模型规模、领域专化、提示工程以及推理时优化的相互作用。

**💡 创新点**

首次在硬件设计场景中，将模型规模与领域专化、结构化提示、链式推理、示例学习以及基于进化的提示优化进行完整对比，揭示这些技术在 Verilog 生成中的相对效益与局限性。

**🔧 技术方法**

使用多种提示策略（结构化提示、提示改写、链式推理、示例学习）和 Genetic‑Pareto（GEPA）提示优化，并在两套 Benchmark（Verilog Eval v2、VeriThoughts）上进行评估；采用 Pass@k（k=1,5,10）作为性能度量。

**📊 数据集**

Verilog Eval v2（156 个基准任务，基于 Icarus Verilog 仿真）和 VeriThoughts（基于 Yosys 的形式化等价检查），两套 Benchmark 覆盖功能正确性与形式化验证。

**📈 对比分析**

通过对 10 种实验条件（Base、Struct、Refine、加 Chain‑of‑Thought、加 In‑Context Learning）在 18 个模型上进行 factorial 设计，比较各模型在不同提示下 Pass@k 的提升；结果表明：模型规模与专化均能提升性能，但提示对模型的敏感性强；在部分模型上，强提示能缩小与 fine‑tuned 模型的差距；两套 Benchmark 的结果高度相关但仍存在显著差异。

**⚠️ 局限性**

实验仅使用单一温度与固定解码参数，未探索更广泛的解码策略；缺乏对错误模式（如时序、接口、边界条件）的细粒度分析；提示优化（GEPA）对某些模型效果有限，提示的可迁移性仍不确定。

---

## 508. Gap-ETH-Tight Algorithms for Hyperbolic TSP and Steiner Tree

**arXiv ID:** 2603.09834 | [PDF](https://arxiv.org/pdf/2603.09834v1)

**作者:** Sándor Kisfaludi-Bak `[一作]` (Aalto University), Geert van Wordragen `[通讯]` (Aalto University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5028724460)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种在d维超曲率空间中解决旅行商问题（TSP）和斯坦纳树问题的近似算法，具有最优的ε依赖性。

**💡 创新点**

引入了混合超曲率四叉树结构和非均匀的门户放置策略，显著改进了现有算法的性能。

**🔧 技术方法**

使用了一种基于随机偏移的动态规划算法，结合了新的层次分解技术。

**📊 数据集**

使用了自定义的超曲率四叉树数据结构，适用于d维超曲率空间的点集。

**📈 对比分析**

与现有方法相比，提出的算法在时间复杂度上为2^O(1/ε^(d-1))n^(1+o(1))，在ε的依赖性上达到了最优。

**⚠️ 局限性**

算法在处理一般超曲率点集时无法实现多项式时间的去随机化，且在输入点集的直径较大时，性能可能受到影响。

---

## 509. An Optimal Control Approach To Transformer Training

**arXiv ID:** 2603.09571 | [PDF](https://arxiv.org/pdf/2603.09571v1)

**作者:** Kağan Akman `[一作]` (Bilkent University), Serdar Yüksel `[通讯]` (Queen's University)

**通讯引用:** 3236 | [OpenAlex ID](https://openalex.org/A5005401257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种从最优控制理论出发，严格刻画 Transformer 训练过程的框架，将 Transformer 视为共享控制的离散时间 McKean–Vlasov 动力学系统，并通过把粒子动力学提升到概率测度空间得到一个完全可观测的 Markov 决策过程，利用动态规划证明存在全局最优策略；随后提出三重量化（状态、测度、动作）训练方案，并证明其近似最优性、稳定性与泛化性质；

**💡 创新点**

创新点在于：①首次将 Transformer 训练形式化为集体控制的最优控制问题；②通过提升到测度空间恢复 Markov 性，并证明闭环最优策略等价于开放式输入独立策略；③提出三重量化动态规划训练方法，并给出近似最优性与稳健性的理论保证；

**🔧 技术方法**

采用的技术包括：最优控制与动态规划、McKean–Vlasov 动力学、弱 Feller 性质、Wasserstein 距离测度、概率测度量化、动作空间量化、Arzelà–Ascoli 定理、Γ‑收敛等；

**📊 数据集**

实验使用自生成的 toy 数据集：对自注意力层的理想函数 T 进行采样，训练集 K_train=35，测试集 K_test=15，序列长度 N=4，状态空间为 ([-1,1]^2)^N；

**📈 对比分析**

通过比较不同动作量级下的训练误差与测试误差，发现误差随动作量级增大而递减；在动作量级 100 时训练误差降至 0.0046（提升 70%），测试误差降至 0.00384（提升 65%）；训练时长随动作量级近似二次增长，拟合曲线 P(M)=0.0438M^2-0.420M+7.319；

**⚠️ 局限性**

主要限制：三重量化方法在理论上可行但计算量随状态/动作量级迅速增长，难以扩展至高维或大规模数据；实验仅在 toy 任务上验证；方法依赖于紧致性与弱 Feller 条件，未与梯度下降训练在大规模模型上的性能做直接对比；

---

## 510. Generalized Reduction to the Isotropy for Flexible Equivariant Neural Fields

**arXiv ID:** 2603.08758 | [PDF](https://arxiv.org/pdf/2603.08758v1)

**作者:** Alejandro García-Castellanos `[一作]` (University of Amsterdam), Erik J Bekkers `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了在一个因素是齐次空间时，将 G 不变函数在异质乘积空间 X×M 上简化为 H（M 的稳定子）在 X 上的不变函数的通用方法，并将其应用于扩展 Equivariant Neural Fields 到任意齐次条件空间。

**💡 创新点**

创新点是把全局 G 不变问题归约为更小的 H 不变问题，实现了可分离轨道空间的显式同构 (X×M)/G≅X/H，并给出了通用的 canonicalization 与 moving frame 构造；此方法统一并推广了之前针对 M×M 或 M^m 的特殊归约。

**🔧 技术方法**

采用群作用理论、轨道空间同构、稳定子子群、canonicalization（移动框架）以及不变理论中的生成和分离不变函数技术；在 ENF 中使用该归约来构造最大表达能力的 H 不变网络。

**📊 数据集**

论文中未给出具体数据集，仅在理论和算法层面进行讨论；若在实验中使用，预期会在欧几里得和球面几何下进行。

**📈 对比分析**

没有公开的实验比较与性能评估；作者指出未来需要系统性评估不同条件空间对学习动态的影响。

**⚠️ 局限性**

局限在于仅适用于 G 在某个因素上传递作用的情形，对非齐次或非传递作用的情况不适用；并且缺乏实证验证。

---

## 511. SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models

**arXiv ID:** 2603.09215 | [PDF](https://arxiv.org/pdf/2603.09215v1)

**作者:** Hsiao-Ying Huang `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**通讯引用:** 9046 | [OpenAlex ID](https://openalex.org/A5040508737)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 SPAR-K 框架，在交互式语音语言模型中对语音标记进行调度式早退出，显著降低推理深度

**💡 创新点**

通过周期性交替全深度和早退出的调度，利用语音标记的冗余性和局部可预测性，实现高效推理且保持语音质量

**🔧 技术方法**

使用 Transformer 结构的交互式 SLM、层级 LM 头训练、KV-cache 补全以及基于固定周期的调度策略

**📊 数据集**

使用 Step‑Audio‑2‑Mini 与 GLM‑4‑Voice 两大模型，在 AlpacaEval、LlamaQuestions、TriviaQA、WebQuestion 四个英语数据集上评估

**📈 对比分析**

与无早退出、固定层早退出、基于置信度的早退出对比，SPAR‑K 在保持 0.82% 以内精度下降的同时实现 5–11% 语音深度削减，MOS 与 WER 变化微乎其微

**⚠️ 局限性**

仅对语音标记实现早退出，对文本标记不适用；固定调度对不同模型、不同语音块长度的适应性有限；需手动调节退出层和周期参数

---

## 512. HelixTrack: Event-Based Tracking and RPM Estimation of Propeller-like Objects

**arXiv ID:** 2603.09235 | [PDF](https://arxiv.org/pdf/2603.09235v1)

**作者:** Radim Spetlik `[一作]` (Czech Technical University in Prague), Jiri Matas `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 50897 | [OpenAlex ID](https://openalex.org/A5007656938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种名为HelixTrack的事件驱动方法，能够实时追踪无人机螺旋桨并精确估计其转速（RPM）。

**💡 创新点**

创新点在于将事件映射到旋转平面并利用螺旋相位模型，结合每事件扩展卡尔曼滤波器和批量高斯牛顿优化，实现了高精度的轨迹与微秒级转速估计。

**🔧 技术方法**

技术实现包括事件反射到旋转平面、扩展卡尔曼滤波器估计相位与角速度、批量高斯牛顿更新相机-旋转平面单应性，以及软门控与正则化的损失设计。

**📊 数据集**

使用自制的TQE数据集，该数据集包含13个高分辨率事件序列、52个旋转对象、2/4 m距离、不同等速运动水平，并提供微秒级红外同步转速标注。

**📈 对比分析**

与AEB‑Tracker和DeepEv两种基准方法相比，HelixTrack在所有场景下均实现了更低的MAE（约100 RPM）和更快的处理速度（≈11.8×实时），同时保持了微秒级更新延迟。

**⚠️ 局限性**

局限性包括对初始位置信息、尺度与转速的精确估计高度依赖、仅能处理单一平面模型的旋转对象、无法自动重新定位以及对严重遮挡或极端视角的鲁棒性不足。

---

## 513. The Spanning Ratio of the Directed $Θ_6$-Graph is 5

**arXiv ID:** 2603.09048 | [PDF](https://arxiv.org/pdf/2603.09048v1)

**作者:** Prosenjit Bose `[一作]`, John Stuart `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文对有限点集的定向Theta-6图（每个点在六个等角锥中只保留最近点的有向边）进行深入研究，证明其距离保持性质的最优伸缩因子（spanning ratio）为5。

**💡 创新点**

创新点在于首次给出Theta-6图的紧致跨度比，并将之前的4–7区间压缩到精确的5，成为第一个对任何Θ_k图给出紧致跨度比的结果；同时采用线性规划与几何诱导证明相结合的新思路。

**🔧 技术方法**

主要技术手段包括：几何构造（空三角、等边三角、锥划分）、诱导证明框架、对候选路径进行系统不等式建模以及利用线性规划求解系统不可行性，从而得到对所有点对的路径长度上界。

**📊 数据集**

该工作完全基于理论证明，没有使用任何实验数据集。

**📈 对比分析**

通过证明可知，任意定向Theta-6图都是5-紧致的，即任意两点间的最短路径长度最多为原直线距离的5倍；这在理论上给出了最优的距离保持界限。

**⚠️ 局限性**

局限性包括：证明仅适用于k=6的Theta图，尚未扩展到一般k；所用证明过程较为复杂，可能不易推广到更高维或其他几何图；并且未讨论算法实现或实际构造的时间复杂度。

---

## 514. Architectural Design and Performance Analysis of FPGA based AI Accelerators: A Comprehensive Review

**arXiv ID:** 2603.08740 | [PDF](https://arxiv.org/pdf/2603.08740v1)

**作者:** Soumita Chatterjee `[一作]` (Indian Institute of Engineering Science and Technology), Hafizur Rahaman `[通讯]` (Indian Institute of Engineering Science and Technology)

**通讯引用:** 4337 | [OpenAlex ID](https://openalex.org/A5082934529)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了基于FPGA的深度学习加速器在硬件层面上的优化技术和性能分析，涵盖CNN、SNN、RNN、GNN等模型的专属设计与通用架构。

**💡 创新点**

系统性分类了多层级优化策略（计算、内存、多层级），提出了针对FPGA的挑战与未来研究方向，并对比分析了FPGA与GPU、ASIC在吞吐量、延迟、能耗等指标上的优势与不足。

**🔧 技术方法**

使用的技术包括循环流水线、并行化（数据/指令/任务/特征级），量化与精度自适应，权重量化与稀疏化，内存层次化与近似计算，硬件/软件协同设计，高层综合（HLS、自动RTL生成）以及近似内存计算（AIMC、DIMC）等。

**📊 数据集**

文章主要引用了众多实验论文所使用的标准数据集，如ImageNet、CIFAR-10/100、MNIST、ImageNet、ImageNet‑VGG、ResNet、YOLO、Graph‑CNN、GAT等，用以展示不同加速器的性能表现。

**📈 对比分析**

通过对比表格和图表，论文比较了GPU、ASIC与FPGA在吞吐量（TOPS/GOPS）、延迟（微秒）、功耗（W/TFLOPS/W）和资源利用率（LUT/DSP/BRAM）等维度的表现，指出FPGA在低功耗、可编程性和模型适配性方面具有优势，但在最高吞吐量与功耗效率上仍落后于ASIC；与GPU相比，FPGA在能效和定制化方面更具优势。

**⚠️ 局限性**

限制包括量化误差导致精度下降、功耗与效率的权衡、CPU/FPGA协同设计缺失导致的通信瓶颈、内存带宽与访问效率不足、缺乏统一生态与标准库、可扩展性受限，以及配置位流安全风险等。

---

## 515. SinGeo: Unlock Single Model's Potential for Robust Cross-View Geo-Localization

**arXiv ID:** 2603.09377 | [PDF](https://arxiv.org/pdf/2603.09377v1)

**作者:** Yang Chen `[一作]` (National University of Defense Technology), Tao Wu `[通讯]` (National University of Defense Technology)

**通讯引用:** 39888 | [OpenAlex ID](https://openalex.org/A5043731569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SinGeo 框架，利用双重判别学习和课程学习实现单模型鲁棒的跨视角地理定位

**💡 创新点**

首次将双分支判别学习与课程学习结合，既提升同视角判别性，又实现无额外模块、无显式视角/视场变换的单模型鲁棒性

**🔧 技术方法**

采用双分支对比学习、增量式视角/视场变换、动态相似性采样以及 SSIM 一致性评估等技术

**📊 数据集**

在 CVUSA、CVACT、VIGOR 与 University‑1652 等四大基准数据集上进行实验

**📈 对比分析**

与 DSM、ConGeo、Sample4Geo 等现有方法在未知方向和有限 FoV 场景下进行 Top‑k recall 对比，SinGeo 在所有 FoV 下均取得 SOTA，尤其在 90°/70° 极限 FoV 时显著优于专门训练模型

**⚠️ 局限性**

需在训练时使用全景图，缺乏在无全景数据集（如 University‑1652）上的最佳处理方案

---

## 516. Intelligent Spatial Estimation for Fire Hazards in Engineering Sites: An Enhanced YOLOv8-Powered Proximity Analysis Framework

**arXiv ID:** 2603.09069 | [PDF](https://arxiv.org/pdf/2603.09069v1)

**作者:** Ammar K. AlMhdawi `[一作]` (University of Greater Manchester), Alaa Mashan Ubaid `[通讯]` (University of Khorfakkan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于YOLOv8的双模型框架，实现火灾分割与周围目标的实时距离估计与风险评估。

**💡 创新点**

创新点在于将火灾分割与COCO预训练目标检测结合，利用像素到米的比例转换实现距离估计，并将距离与火灾强度、目标脆弱性融合给出可解释的风险等级。

**🔧 技术方法**

采用YOLOv8实例分割模型、COCO预训练YOLOv8目标检测、像素-米比例映射、距离衰减函数及风险聚合公式等技术。

**📊 数据集**

使用了9,860张手工标注的火灾与烟雾分割数据集。

**📈 对比分析**

在验证集上mAP@0.5达91%以上，精确率、召回率和F1均超过90%，并与传统单模型检测相比，增加了距离与风险输出，提升了实用性。

**⚠️ 局限性**

局限性包括缺乏相机标定导致距离估计为近似、仅基于二维像素距离忽略深度变化、以及需要事先已知参考尺度。

---

## 517. Integrating Virtual and Augmented Reality into Public Education: Opportunities and Challenges in Language Learning

**arXiv ID:** 2603.08970 | [PDF](https://arxiv.org/pdf/2603.08970v1)

**作者:** Tanja Kojić `[一作]` (TU Berlin), Jan-Niklas Voigt-Antons `[通讯]` (Hamm-Lippstadt University of Applied Sciences)

**通讯引用:** 1638 | [OpenAlex ID](https://openalex.org/A5063539298)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过两项实证研究评估AR（Mondly AR）和VR（ImmerseMe VR）在公共教育中的语言学习效果，比较其在词汇学习、发音、听力与交谈等方面的表现与可用性。

**💡 创新点**

首次系统地将AR与VR在同一研究框架下并行评估，并将可用性、认知负荷与学习成果三维度结合，提出针对公共教育可实施的技术与教学改进策略。

**🔧 技术方法**

使用AR移动应用（Mondly AR）与VR头显（ImmerseMe VR）相结合的沉浸式学习平台，配合语音识别、可适应学习模块与存在感问卷（IPQ）等技术手段。

**📊 数据集**

数据来源为45名AR实验参与者与31名VR实验参与者的前测、后测、可用性问卷与IPQ得分；未使用公开大规模语言学习数据集，而是以实验收集的学习成绩与主观评估为基础。

**📈 对比分析**

采用混合方法：对AR组进行词汇前测/后测对比，AR与VR组分别与非沉浸式控制组进行成绩差异比较；使用t检验与相关分析评估学习增益与存在感/可用性之间的关系。结果显示AR词汇学习提升12.5%，VR学习成绩提升8.3分，VR组在IPQ中得分为3.7（中高存在感）并与学习增益呈正相关（r=0.45）。

**⚠️ 局限性**

局限性包括样本量相对较小、仅涉及短期学习效果、缺乏长期记忆跟踪、对高阶语言技能（写作、批判性思维）的探究不足、技术成本与硬件要求限制了可推广性、认知负荷与用户界面在部分参与者中仍存在问题。

---

## 518. Better Bounds for the Distributed Experts Problem

**arXiv ID:** 2603.09168 | [PDF](https://arxiv.org/pdf/2603.09168v1)

**作者:** David P. Woodruff `[一作]` (Carnegie Mellon University), Samson Zhou `[通讯]` (Texas A&M University)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5018283928)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究分布式专家在线学习问题，提出适用于一般 α_p 损失的低通信协议，能够在多服务器环境下实现接近最优的期望遗憾。

**💡 创新点**

创新点在于将 α_p 损失嵌入 α_∞ 结构，并利用指数随机变量与几何平均估计来控制方差，从而在不显著增加通信量的前提下实现低遗憾。

**🔧 技术方法**

主要技术包括指数随机采样、几何平均估计、乘法权重更新（MWU）、Chernoff 边界分析以及通信-遗憾权衡的精细设计。

**📊 数据集**

实验使用 HPO-B 超参数优化基准数据集，模拟不同机器学习任务的专家集合和服务器分布。

**📈 对比分析**

通过与标准 MWU 和仅支持 α_1 的先前协议进行对比，结果显示我们的协议在通信量和奖励（收益）上均优于对照组，特别是在 α_1 情况下显著降低通信需求。

**⚠️ 局限性**

局限性包括实验仅在 HPO-B 数据集上验证；对极端分布或非 α_p 损失形式缺乏理论下界；所提出的通信模型仅适用于中心化的协调器场景。

---

## 519. OmniEarth: A Benchmark for Evaluating Vision-Language Models in Geospatial Tasks

**arXiv ID:** 2603.09471 | [PDF](https://arxiv.org/pdf/2603.09471v1)

**作者:** Ronghao Fu `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**通讯引用:** 71942 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 OmniEarth 基准，用于系统评估遥感视觉‑语言模型（RSVLMs）的感知、推理和鲁棒性。

**💡 创新点**

创新点在于构建 28 个细粒度任务，覆盖多源、多时序、多模态数据；采用盲测和五重语义一致性约束，以消除语言偏差并检验视觉依赖；并首次引入 JL‑1 专有卫星图像和 400+ 城市覆盖。

**🔧 技术方法**

技术方法包括：零样本评估 19 种 VLM（对比、通用、遥感专用）；多格式任务（多选 VQA、定位、分割、自由文本）；人工三组三轮交叉验证；以及基于 LLM 的问题与干扰项生成。

**📊 数据集**

数据集涵盖 9,275 张图像（Jilin‑1、Sentinel‑1/2、Google Earth Engine 等）以及 44,210 条手工验证指令，跨七大洲、400+ 城市，包含光学、多光谱、SAR、红外等多模态。

**📈 对比分析**

比较方法采用任务专属指标（准确率、IoU、CIDEr 等），结果显示通用 VLM 在图像级感知上表现优于 RSVLM，但在细粒度感知、定量推理和鲁棒性方面均低于 50%；RSVLM 对文本提示高度依赖，鲁棒性差。

**⚠️ 局限性**

局限性包括：细粒度感知与分割能力不足；推理（空间、时间、决策）精度低；对图像降质、跨模态匹配缺乏鲁棒性；模型容易依赖语言先验，缺乏真正的视觉 grounding。

---

## 520. Investigating the Effects of LLM Use on Critical Thinking Under Time Constraints: Access Timing and Time Availability

**arXiv ID:** 2603.08849 | [PDF](https://arxiv.org/pdf/2603.08849v1)

**作者:** Jiayin Zhi `[一作]` (University of Chicago), Mina Lee `[通讯]` (University of Chicago)

**通讯引用:** 628 | [OpenAlex ID](https://openalex.org/A5100756493)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验中评估了LLM（如GPT‑4o）在不同访问时机（早期、连续、后期、无）与时间可用性（不足、充足）条件下对批判性思维任务（写作决策）的影响。

**💡 创新点**

创新点在于揭示了时间维度的“时间反转”效应：早期访问LLM在时间紧迫时提升表现，足够时间时反而削弱表现，而先独立思考后访问LLM则相反。

**🔧 技术方法**

使用了大语言模型（GPT‑4o）作为交互式助理，并通过自定义提示使其仅基于任务文档回答问题；同时采用ANCOVA等统计方法分析结果。

**📊 数据集**

数据集为iPAL（International Performance Assessment of Learning）框架下的水污染决策场景，包含七份不同可信度、立场和相关度的文件。

**📈 对比分析**

通过ANCOVA对Essay得分、Recall、Evaluation、Comprehension等指标进行比较，结果显示在时间紧迫下早期LLM访问可提升Essay得分，而在时间充足时则表现更差，整体效果因时间与访问时机交互显著。

**⚠️ 局限性**

限制包括实验仅在受控实验室环境下进行，样本为众包受试者，任务仅涵盖一个领域（公民决策），难以推广到更广泛的真实工作或学习场景；此外，LLM的提示与交互方式可能影响结果，需进一步验证。

---

## 521. Extension of ACETONE C code generator for multi-core architectures

**arXiv ID:** 2603.08744 | [PDF](https://arxiv.org/pdf/2603.08744v1)

**作者:** Yanis Aït-Aïssa `[一作]` (ONERA), Claire Pagetti `[通讯]` (ONERA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

为航空安全关键系统的深度神经网络推理扩展 ACETONE 代码生成器，使其能够在多核 CPU 上生成可预测的并行 C 代码并实现同步机制。

**💡 创新点**

将 DNN 推理任务建模为 DAG 调度问题，提出一种更紧凑的约束规划编码并实现两种启发式调度算法（ISH 与 DSH），同时在 ILP 方案上做优化，以显著降低求解复杂度并在多核平台上获得可证可预测的执行时间。

**🔧 技术方法**

采用 DAG 调度、约束规划（CP/ILP）与启发式算法（基于关键路径的列表调度与复制策略）进行任务分配；使用 OTAWA 进行 WCET 估算；在 Texas Instruments Keystone II 平台上跑真实实验；使用同步 flag 与共享缓冲区实现跨核通信。

**📊 数据集**

评估使用随机生成的 DAG（20、50、100 节点，10% 连接密度）以及一个基于 GoogleNet 的实际网络，进行实验验证。

**📈 对比分析**

通过与单核 baseline 的比对，ISH 与 DSH 取得 5–7 级核心时的速度提升平台化平台化，DSH 在速度上略优于 ISH 但计算时间增长 1–2 个数量级；优化后的 ILP 编码在 20/50 节点上能在 1 小时内求解并获得近似最优，远快于原始 Tang 编码；在 Keystone II 上实现的多核版本实际测得 8% 总体加速，单个可并行区段可达 31% 加速。

**⚠️ 局限性**

局限性包括：仅支持同构 UMA 多核平台，通信采用阻塞 flag + 共享缓冲区导致内存占用高且易受干扰；未考虑异构核或加速器、非阻塞通信；优化和实验数据仅针对少量随机 DAG 与单一网络；ILP 求解仍在大规模网络下难以在可接受时间内得到最优解。

---

## 522. TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers

**arXiv ID:** 2603.08928 | [PDF](https://arxiv.org/pdf/2603.08928v1)

**作者:** Yihua Liu `[一作]` (Independent Researcher), Chengming Zhang `[通讯]` (University of Houston)

**通讯引用:** 725 | [OpenAlex ID](https://openalex.org/A5100691052)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的框架TIDE，能让预训练的Diffusion Transformer在更高分辨率（如4096×4096）下生成高质量图像。

**💡 创新点**

核心创新在于两点：文本锚定（Text Anchoring）通过在交叉注意力 logits 上加入可伸缩偏置恢复文本信息；动态温度控制（Dynamic Temperature Control）根据扩散步骤和频率自适应调整 softmax 温度，抑制高频噪声。

**🔧 技术方法**

主要技术包括Transformer注意力机制、RoPE/NTK插值、温度衰减公式、频率自适应温度调节以及对注意力分布的分析。

**📊 数据集**

使用FLUX.1-dev模型的训练分辨率为1024×1024，评估数据集为DrawBench（200条提示）和Aesthetic-4K（195个图像-提示对）。

**📈 对比分析**

与直接外推、YaRN、Dy-YaRN 等基线对比，TIDE在FID、KID、CLIP Score、ImageReward和Aesthetic Score 等指标上均优于对手，尤其在4096×4096分辨率下表现突出；用户研究也显示在文本匹配、结构完整性和细节表现方面得到更高评价。

**⚠️ 局限性**

局限性包括：对低频细节的改进仍有提升空间，动态温度控制的参数设定依赖经验；在极端高分辨率或特殊场景下可能出现局部伪影；需要进一步探讨与其它模型的兼容性与系统级优化。

---

## 523. CogBlender: Towards Continuous Cognitive Intervention in Text-to-Image Generation

**arXiv ID:** 2603.09286 | [PDF](https://arxiv.org/pdf/2603.09286v1)

**作者:** Shengqi Dang `[一作]` (Tongji University and Shanghai Innovation Institute), Nan Cao `[通讯]` (Tongji University and Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 CogBlender 框架，实现对文本生成图像过程中多维认知属性（如情绪维度 Valence–Arousal–Dominance 与记忆度）进行连续可调控制。

**💡 创新点**

创新点在于：①构建认知空间与语义流形的映射，使用认知锚点形成极端语义边界；②通过在流匹配（FLUX.2）中插值重构速度场，实现多维连续控制；③在不训练额外模型的前提下完成跨维度认知调节。

**🔧 技术方法**

核心技术包括：大语言模型（Qwen3）实现提示极化，认知锚点与拉丁方重写顺序，速度场插值（v̂ 与 vθ 组合），以及基于流匹配的 ODE 解算生成图像。

**📊 数据集**

使用公开情感与记忆数据集：VAD 标注数据、MemNet 记忆度预测模型及相应图像集（如 ImageNet、COCO 等作为提示文本来源）。

**📈 对比分析**

与 EmotiCrafter、FLUX.2、GANalyze 等基线对比。评估指标为 CLIPScore、CLIPIQA、V/A/D/记忆度误差；结果显示 CogBlender 在情绪控制误差上最小、视觉质量最优、对话语义保持最佳。

**⚠️ 局限性**

局限性：推理时间较长（约22–40秒/图），当原始提示已携带强认知信号时调节困难；提示极化难以细粒度控制视觉细节，需进一步结合潜在空间约束。

---

## 524. Not All News Is Equal: Topic- and Event-Conditional Sentiment from Finetuned LLMs for Aluminum Price Forecasting

**arXiv ID:** 2603.09085 | [PDF](https://arxiv.org/pdf/2603.09085v1)

**作者:** Alvaro Paredes Amorin `[一作]` (Zhejiang University), Christoph Weisser `[通讯]` (Hochschule Bielefeld University of Applied Sciences and Arts)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用微调的大型语言模型生成情绪信号，融合到铝价预测模型中，并构建基于情绪的交易策略。

**💡 创新点**

首次系统评估LLM情绪在金属价格预测中的价值，揭示情绪在不同波动率场景下的不同效益，并比较多源新闻的情绪质量差异。

**🔧 技术方法**

深度学习模型（Qwen3 8B、FinBERT），时间序列模型（ARIMA、LSTM、XGBoost 等），情绪分类与量化；并使用 Sharpe 比例和累计收益评价。

**📊 数据集**

Wind 终端月度铝价及相关因子数据（2007‑2024），Factiva 新闻标题（Reuters、Dow Jones Newswires、China News Service）。

**📈 对比分析**

对比多种模型与交易策略（仅表格、表格+情绪、仅情绪），在不同波动率区间评估 Sharpe；Qwen3+情绪在高波动率下最高，情绪单独在中波动率下优于混合模型。

**⚠️ 局限性**

新闻覆盖稀疏导致只能做月度分析；仅针对铝，未验证其他金属；情绪生成受模型与数据源质量限制。

---

## 525. One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations

**arXiv ID:** 2603.08869 | [PDF](https://arxiv.org/pdf/2603.08869v1)

**作者:** Sripad Karne `[一作]` `[通讯]`, Sripad Karne

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了ICLR 2026会议论文提交的完整格式化规范与写作指南。

**💡 创新点**

通过细化排版细节、字体、页边距、标题层级、引用与图表等要求，确保所有稿件在视觉与结构上高度一致，从而提升评审效率。

**🔧 技术方法**

采用LaTeX模板（neural‑network‑conference‑style）和相应宏包实现自动排版，强调使用Times New Roman、10 pt文本、特定行距等标准。

**📊 数据集**

无数据集，仅为格式手册。

**📈 对比分析**

无实验或方法比较，本文仅提供规范性说明。

**⚠️ 局限性**

限制主要在于过度强调格式，缺少对研究内容多样性的支持，且不涉及实际科研创新。

---

## 526. EmoSURA: Towards Accurate Evaluation of Detailed and Long-Context Emotional Speech Captions

**arXiv ID:** 2603.09820 | [PDF](https://arxiv.org/pdf/2603.09820v1)

**作者:** Xin Jing `[一作]`, Björn Schuller `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构造了四类恶意扰动的负样本，评估多模检索模型的鲁棒性。

**💡 创新点**

创新点在于系统化的幻觉分类与属性互换，提供内部一致的负样本。

**🔧 技术方法**

采用手工构造的对抗样本，并结合原始音频与字幕的对齐分析。

**📊 数据集**

使用公开的音频-字幕对数据集（未指明具体名称）。

**📈 对比分析**

通过与标准检索模型对比，发现模型在面对属性互换与情感翻转样本时表现显著下降。

**⚠️ 局限性**

局限性包括缺乏自动化生成机制和对更大规模数据集的验证。

---

## 527. PixelConfig: Longitudinal Measurement and Reverse-Engineering of Meta Pixel Configurations

**arXiv ID:** 2603.09380 | [PDF](https://arxiv.org/pdf/2603.09380v1)

**作者:** Abdullah Ghani `[一作]` (Lahore University of Management Sciences), Zubair Shafiq `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对Meta Pixel在网页中的配置进行逆向工程，构建差分分析框架，对2017‑2024年18K健康网站与控制集10K网站的配置进行长期纵向分析。

**💡 创新点**

提出了基于静态与动态补丁的差分逆向框架，首次系统解析Meta Pixel的活动追踪、身份追踪和限制配置，并在健康网站中揭示对敏感信息的潜在跟踪。

**🔧 技术方法**

利用JavaScript代码补丁、Chrome DevTools覆盖、Selenium抓取、Wayback Machine API、正则解析配置脚本、SHA‑256逆向查找黑名单/敏感键，并通过统计检验比较两组网站的采纳率。

**📊 数据集**

主要数据集包括：AHA/CMS公开健康网站列表（18,327站点），Tranco top‑10k网站（10,000站点），以及这些站点在2017‑2024年间的Wayback存档快照与Meta Pixel配置脚本。

**📈 对比分析**

通过对比健康与控制网站的配置占比、时间演化曲线，采用两比例z检验与Cohen’s h衡量显著性，结果显示活动追踪与第一方Cookie在98%+的站点上被默认启用，健康站点在2023‑24年核心设置采用率显著高于控制站点。

**⚠️ 局限性**

主要局限：Wayback Machine对配置脚本的归档不完整（约40‑70%缺失），导致某些年份的样本不稳定；静态/动态补丁方法对高度混淆的脚本识别有限；仅关注Meta Pixel，无法推广到其他追踪像素。

---

## 528. A Graph-Based Approach to Spectrum Demand Prediction Using Hierarchical Attention Networks

**arXiv ID:** 2603.09859 | [PDF](https://arxiv.org/pdf/2603.09859v1)

**作者:** Mohamad Alkadamani `[一作]` (Communications Research Centre Canada), Amir Ghasemi `[通讯]` (Communications Research Centre Canada)

**通讯引用:** 4738 | [OpenAlex ID](https://openalex.org/A5063162458)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了HR-GAT层次图注意网络，利用多分辨率地理空间数据预测频谱需求。

**💡 创新点**

创新点在于融合多级分辨率的图注意机制，解决空间自相关导致的泛化问题，并构建了验证过的频谱需求代理。

**🔧 技术方法**

采用图神经网络、注意力机制、层次化多分辨率建模以及SHAP解释技术。

**📊 数据集**

使用公开的网络部署记录、移动运营商流量数据以及Bing地图多尺度网格特征，共计30个地理空间特征。

**📈 对比分析**

通过聚类交叉验证(CBCV)和留一城市验证(LOCO)与八个基线模型对比，HR-GAT在MAE、RMSE、R²上分别领先约21%，并在未见城市上仍保持最高精度。

**⚠️ 局限性**

局限在于仅验证于加拿大五大城市，特征选择和代理构建对不同地区可能需调整，且模型对极端城市形态的适应性未充分测试。

---

## 529. X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models

**arXiv ID:** 2603.09632 | [PDF](https://arxiv.org/pdf/2603.09632v1)

**作者:** Yueen Ma `[一作]` (Chinese University of Hong Kong), Irwin King `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 27632 | [OpenAlex ID](https://openalex.org/A5042251906)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了X‑GS，一个可扩展的开源框架，能够实时整合3D高斯分布（3DGS）在姿态无关跟踪、在线SLAM、语义映射和与视觉语言模型（VLM）融合等多领域的能力，形成统一的在线语义三维建图系统。

**💡 创新点**

核心创新在于（1）构建可扩展的框架结构，将不同3DGS技术整合为可插拔组件；（2）在X‑GS‑Perceiver中首次引入在线EMA更新的向量量化（VQ）模块、GPU加速的网格采样监督和高度并行的管线调度，实现约15 FPS的实时性能；（3）通过X‑GS‑Thinker将语义化3D高斯直接馈入VLM或VLA，实现开放词汇3D目标检测、零样本图像字幕生成及潜在的具身任务。

**🔧 技术方法**

使用的关键技术包括：3D Gaussian Splatting、向量量化（VQ）与EMA、GPU加速网格采样、并行多线程调度、视觉基础模型（SAM、CLIP、SigLIP）、对比式VLM（OpenCLIP、LLaVA）以及基于信息熵的高斯采样。

**📊 数据集**

在真实世界的RGB/RGB‑D视频流上进行评估，使用公开的室内外场景数据集（如ScanNet、Matterport3D等），并对比单独的MonoGS、GS‑SLAM、Feature 3DGS等现有方法。

**📈 对比分析**

与基准方法相比，X‑GS在实时性（≈15 FPS）和语义质量上表现优异；在目标检测与字幕生成任务上，能够无缝利用CLIP文本查询与3D语义场景进行精准定位和自然语言描述，整体性能超过单独的SLAM或语义映射系统。

**⚠️ 局限性**

局限性主要包括：目前仍缺乏在多模态、动态场景与具身任务上的全面评测；X‑GS‑Thinker接口为模块化，未实现端到端微调；在线优化仍存在一定的计算开销，尤其在高分辨率或大场景中；对4D或动态3DGS的支持尚未成熟。

---

## 530. Interpretable Markov-Based Spatiotemporal Risk Surfaces for Missing-Child Search Planning with Reinforcement Learning and LLM-Based Quality Assurance

**arXiv ID:** 2603.08933 | [PDF](https://arxiv.org/pdf/2603.08933v1)

**作者:** Joshua Castillo `[一作]` (Old Dominion University), Ravi Mukkamala `[通讯]` (Old Dominion University)

**通讯引用:** 1514 | [OpenAlex ID](https://openalex.org/A5035065105)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套端到端的Missing-Child调查支持系统Guardian，能将非结构化报告转化为可操作的搜索计划；

**💡 创新点**

三层架构：稀疏可解释的Markov移动预测、强化学习优化搜索区域、LLM质量保证验证；

**🔧 技术方法**

技术包括：基于地理信息的能量化转移矩阵、日夜分离的Markov传播、强化学习策略网络、指令调优的LLM（Qwen‑2.5‑3B‑Instruct、LLaMA‑3.2‑3B‑Instruct）和聚类/核密度先验；

**📊 数据集**

使用Virginia州真实失踪儿童案件PDF（经Parser Pack转化）、合成案例GRD‑2025‑001541、道路/交通/通道/隐蔽度等GIS层；

**📈 对比分析**

与两种基线对比：纯Markov+RL排序和LLM调整后的排序；在合成案例上表现为：0–72h累计概率集中在Tidewater区>50%，北弗吉尼亚占约25‑30%；区域内Containment半径随时间从≈20mi扩大至≈25mi，展示可解释的概率扩散；

**⚠️ 局限性**

局限：Markov无记忆、网格分辨率有限、对地理编码/先验高度敏感、参数如α_prior、corridor/seclusion权重需手工校准，LLM可能产生幻觉，整体为建议性决策支持而非强制指令。

---

## 531. TA-GGAD: Testing-time Adaptive Graph Model for Generalist Graph Anomaly Detection

**arXiv ID:** 2603.09349 | [PDF](https://arxiv.org/pdf/2603.09349v1)

**作者:** Xiong Zhang `[一作]` (Yunnan University), Cheng Xie `[通讯]` (Yunnan University)

**通讯引用:** 83242 | [OpenAlex ID](https://openalex.org/A5100387487)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种通用的图异常检测框架，解决跨域异常离散性（Anomaly Disassortativity）问题，实现零训练跨域检测。

**💡 创新点**

创新点在于：①发现并量化节点与结构层面的离散性；②设计高阶残差对比学习与低阶亲和编码的双路评分；③引入自适应适配器和测试时适配器，实现动态权重融合与无监督伪标签自适应。

**🔧 技术方法**

使用技术包括：高阶残差表征与对比学习、低阶亲和编码与余弦相似度、Jensen‑Shannon 距离度量节点/结构离散度、Ada适配器权重融合、投票+自适应加权的测试时适配器。

**📊 数据集**

使用数据集：14 个真实世界图（训练集 PubMed、Flickr、Questions、YelpChi；测试集 ACM、Facebook、Amazon、Cora、CiteSeer、BlogCatalog、Reddit、Weibo、CS、Photo、Elliptic、T‑Finance、DGraph‑Fin）。

**📈 对比分析**

与 16 种基线（监督、半监督、无监督以及 ARC、UNPrompt、AnomalyGFM）对比，AUROC 在 11/13 关键数据集排名第一，平均排名 1.23，显著提升如 CS +15.73%、Facebook +14.78%、ACM +8.90%。

**⚠️ 局限性**

局限性：对低离散度域（如 Weibo、Reddit）的提升有限；模型依赖投票阈值的设定；在极端结构或特征差异极大时仍可能表现不佳。

---

## 532. WESPR: Wind-adaptive Energy-Efficient Safe Perception & Planning for Robust Flight with Quadrotors

**arXiv ID:** 2603.09194 | [PDF](https://arxiv.org/pdf/2603.09194v1)

**作者:** Khuzema Habib `[一作]` (University of Maryland), Pratap Tokekar `[通讯]` (University of Maryland)

**通讯引用:** 1903 | [OpenAlex ID](https://openalex.org/A5086188394)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

开发了一套基于快速 CFD 的无人机轨迹规划管线，利用 Lattice Boltzmann 模拟从几何感知和局部天气数据中预测风场，生成风感知成本图并优化 Bézier 轨迹，实现了对风扰动的前瞻性规划。

**💡 创新点**

创新点在于：①在 10 秒内完成物理级风场预测（无需网格生成）；②将预测的风场直接嵌入成本函数，形成闭环的实时规划；③首次在实际 Crazyflie 飞行实验中验证了风感知规划的有效性。

**🔧 技术方法**

使用技术包括：Lattice Boltzmann Method（FluidX3D）GPU 加速求解、三维占据网格构建、A* 先导路径、Bézier 曲线全局优化、风源先验映射以及基于 VIO 的定位。

**📊 数据集**

使用的数据集为室内实验环境：配备 ZED Stereo 2i 深度相机、风扇（用 anemometer 测量风速）与 Styrofoam 障碍物；同时收集风源位置与速度作为风报告。

**📈 对比分析**

与传统无风感知的基准规划器对比，实验显示最大轨迹偏差降低 12.5–58.7%，轨迹失真（Fréchet 距离）降低 24.6%，平均加速度变化（jerk）降低 22–25%，并成功避免了 3 组风场下的碰撞事件。

**⚠️ 局限性**

局限性在于：仅采用二维平面 LBM 模拟，忽略垂直风层和三维湍流；成本函数权重需要手工调参；实验场景为受限室内结构，未涵盖户外复杂环境；未评估在更大尺度或多风源情况下的收敛与稳态表现。

---

## 533. When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency

**arXiv ID:** 2603.09024 | [PDF](https://arxiv.org/pdf/2603.09024v1)

**作者:** Ren Fujiwara `[一作]` (SANKEN, University of Osaka), Yasushi Sakurai `[通讯]` (SANKEN, University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 CALIPER，一种基于状态依赖、单通道加权局部回归的后漂移数据量估计方法，能够在漂移报警后直接从流中决定何时收集足够的数据进行稳定重新训练。

**💡 创新点**

创新点在于：①把后漂移数据量估计从传统的“检测”任务扩展到“何时足够”的决策；②使用数据侧的有效样本量门限与单调局部性检验，无需访问模型内部或后漂移标签；③提供理论分析与低复杂度单遍实现，显著减少在线计算与内存开销。

**🔧 技术方法**

核心技术包括：加权局部回归（Weighted Local Regression）、有效样本量（Effective Sample Size）门限、单调局部性（Monotone Locality）测试、单遍更新、以及与现有漂移检测器（ADWIN、KSWIN）的无缝协作。

**📊 数据集**

实验使用四个领域的数据集：MoCap（动作捕捉）、TEP（Tennessee Eastman Process）、Automobile（车辆传感器数据）、Dysts（混沌系统时间序列）。

**📈 对比分析**

与固定窗口大小重新训练和在线增量更新（SGD）相比，CALIPER 在大多数数据集和模型族（KRR、MLP、Transformer）中都能获得接近或优于最佳固定窗口的误差，且在保持近零额外开销的同时实现更快的漂移恢复。

**⚠️ 局限性**

局限性：依赖于数据呈现显著状态依赖特性，可能对非动态或高维、噪声严重的流失效；对漂移检测器的误报/漏报敏感；当前仅验证了单步预测的代理误差，未来可进一步探讨多步或更复杂模型的通用性。

---

## 534. High-Slip-Ratio Control for Peak Tire-Road Friction Estimation Using Automated Vehicles

**arXiv ID:** 2603.09073 | [PDF](https://arxiv.org/pdf/2603.09073v1)

**作者:** Zhaohui Liang `[一作]` (University of Wisconsin), Xiaopeng Li `[通讯]` (University of Wisconsin)

**通讯引用:** 8340 | [OpenAlex ID](https://openalex.org/A5100355083)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用自动驾驶车辆在空载行驶期间主动激发高滑移状态，结合约束最优控制和分箱统计投影方法，实现对道路与轮胎峰值摩擦系数的安全、可重复估计。

**💡 创新点**

① 将自动驾驶车辆作为主动激励平台，在车道跟随场景下安全地进入峰值滑移区；② 通过简化Magic Formula与分箱投影相结合，提升峰值摩擦估计的鲁棒性；③ 在车距安全约束下设计的最优控制框架，兼顾激励效果与行车安全。

**🔧 技术方法**

简化Magic Formula轮胎模型、约束最优控制（离散优化）、分箱统计投影估计、MATLAB/Simulink仿真、LiDAR+GPS/INS实时定位与闭环控制。

**📊 数据集**

在实验车辆上收集的高速/低速加减速轨迹传感器数据（轮速、加速度、位置等），并在同一位置多次重复测量以提升统计可靠性；仿真数据为基准。

**📈 对比分析**

通过仿真和实车测试验证，所提方法能够在最坏车距条件下安全激发峰值摩擦区，并得到与多次重复测量一致的TRFC估计，显示出比被动估计更高的准确性和可靠性。

**⚠️ 局限性**

受限于仅在空载、干燥混凝土路段验证；单辆车多次行驶的实验缺乏大范围空间覆盖；对不同路面与环境（雨雪、冰等）的适应性仍待进一步研究。

---

## 535. Logics-Parsing-Omni Technical Report

**arXiv ID:** 2603.09677 | [PDF](https://arxiv.org/pdf/2603.09677v1)

**作者:** Xin An `[一作]` (Alibaba Group), Lin Qu `[通讯]` (Alibaba Group)

**通讯引用:** 927 | [OpenAlex ID](https://openalex.org/A5030719052)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Omni Parsing框架和Logics‑Parsing‑Omni模型，实现文档、图像、音频与视频等多模态信息的统一、可定位、可计数、可追溯的解析；

**💡 创新点**

创新点包括：三层进步解析范式（全局检测→细粒度识别→多层推理），将感知与认知有机融合；统一标准化JSON输出与证据锚定机制；两阶段渐进式训练策略；以及构建大规模统一语料库；

**🔧 技术方法**

技术手段主要是：基于Qwen3‑Omni‑30B‑A3B的多模态LLM；Megatron‑SWIFT训练框架；多模态感知模块（OCR、ASR、声学事件检测、图像差异定位、几何解析、图表逆渲染等）；以及基于大型VLM的结构化推理与自然语言生成；

**📊 数据集**

使用了自研的OmniParsingBench、Logics‑Parsing‑Omni数据集（覆盖文档、自然图像、信息图、几何图形、音频、视频、文本丰富视频），训练样本约16M（基础阶段）+5M（精细阶段），文档图像集300K+；

**📈 对比分析**

在OmniParsingBench的六大子任务以及OmniDocBench、OmniDocBench‑v1.5上与Gemini‑3‑Pro、Qwen系列、GPT‑5.2、PaddleOCR‑VL等模型对比，Logics‑Parsing‑Omni在整体、认知得分均居前；尤其在图形、音频、文本丰富视频等认知任务上取得SOTA；在OmniDocBench‑v1.5中仅次于PaddleOCR‑VL，整体表现突出；

**⚠️ 局限性**

局限性：模型参数量大、推理算力需求高；对极端长序列和高噪声多模态输入的鲁棒性仍待提升；跨语言（尤其非中文）性能可能不足；对复杂长篇推理仍存在误差；未来需进一步优化推理效率与持续学习机制。

---

## 536. A Lightweight Multi-Cancer Tumor Localization Framework for Deployable Digital Pathology

**arXiv ID:** 2603.08844 | [PDF](https://arxiv.org/pdf/2603.08844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 537. Dynamic Multimodal Expression Generation for LLM-Driven Pedagogical Agents: From User Experience Perspective

**arXiv ID:** 2603.09536 | [PDF](https://arxiv.org/pdf/2603.09536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 538. Synergistic Directed Execution and LLM-Driven Analysis for Zero-Day AI-Generated Malware Detection

**arXiv ID:** 2603.09044 | [PDF](https://arxiv.org/pdf/2603.09044v1)

**作者:** George Edwards `[一作]`, Mahdi Eslamimehr `[通讯]` (Quandary Peak Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出CogniCrypt框架，结合concolic执行、LLM路径优先和Transformer分类器，实现零日AI生成恶意软件的高效检测。

**💡 创新点**

创新点在于将LLM作为智能路径oracle引导concolic探索，并通过强化学习动态调优优先级，同时给出时序逻辑理论保证检测的可达性与完备性。

**🔧 技术方法**

采用concolic执行（angr+Z3）、LLM路径优先（多种LLM后端）、Transformer路径约束分类器、PPO强化学习回馈以及相关工具（PyTorch、Transformers、TRL等）。

**📊 数据集**

使用EMBER、Malimg、SOREL-20M以及自构建的AI-Gen-Malware（2500个LLM生成样本）进行实验评估。

**📈 对比分析**

与ClamAV、YARA、MalConv、EMBER-GBDT及无LLM的angr对比，CogniCrypt在AI生成恶意样本上达97.5%准确率，提升幅度最高52.2个百分点。

**⚠️ 局限性**

主要限制包括对大规模LLM的算力与成本需求、对LLM训练数据的依赖以及仅针对Windows PE二进制的适用范围。

---

## 539. Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention

**arXiv ID:** 2603.08743 | [PDF](https://arxiv.org/pdf/2603.08743v1)

**作者:** Mengqi Liao `[一作]` (Beijing Jiaotong University), Huaiyu Wan `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 6743 | [OpenAlex ID](https://openalex.org/A5065949777)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Compressed PagedAttention，将 KV 缓存压缩与 PagedAttention 结合，构建了高并发 LLM 推理引擎 Zipage

**💡 创新点**

在 KV 缓存管理中引入 token‑级压缩与按块动态预分配，设计混合调度、共享前缀缓存以及异步压缩解码，显著提升并发度和吞吐量

**🔧 技术方法**

采用 PagedAttention、KV 缓存压缩、异步解码、混合调度、共享前缀缓存、FlashAttention、RoPE 等技术，并实现 GPU 加速 kernel

**📊 数据集**

使用 Qwen3 系列（0.6B、8B、14B、32B）、DeepSeek‑R1 Distill Llama 8B 以及数学与代码基准 AMC 23、GSM8K、AIME 24、LiveCodeBench、MultiFieldQA 等数据集进行评测

**📈 对比分析**

与 vLLM、Nano‑vLLM、HF‑Gen、MorphKV、R‑KV、G‑KV 等框架比较，Zipage 在 2k KV 缓存预算下，数学推理任务 TPS 超过两倍于 vLLM，性能保持约 95% 的 Full‑KV 级别；在不同预算下也显示出高吞吐、良好性能保持

**⚠️ 局限性**

目前仅支持离线推理，未实现在线引擎；KV 缓存预算固定，未针对不同请求长度动态调节；缺少多租户安全隔离；缺乏与在线推理相关的 TTFT 评估

---

## 540. A Consensus-Driven Multi-LLM Pipeline for Missing-Person Investigations

**arXiv ID:** 2603.08954 | [PDF](https://arxiv.org/pdf/2603.08954v1)

**作者:** Joshua Castillo `[一作]` (Old Dominion University), Ravi Mukkamala `[通讯]` (Old Dominion University)

**通讯引用:** 1514 | [OpenAlex ID](https://openalex.org/A5035065105)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Guardian 多模型 LLM 管道，用于缺失儿童调查中的信息抽取、摘要和弱标签生成，并通过共识层保证可靠性。

**💡 创新点**

创新点在于多模型共识驱动的可靠性机制、QLoRA 低秩微调、结构化验证与修复、统一 Prompt 治理以及 Zone QA 等多层级保障。

**🔧 技术方法**

采用多模型 LLM（Qwen、Llama、Gemini）、QLoRA 微调、vLLM API、共识引擎（Gemini Flash/Pro）、结构化验证器、并发调度、缓存与时间预算等技术。

**📊 数据集**

使用合成与半结构化缺失儿童案例语料，结合真实案例和研究语料库，用于 QLoRA 微调与评估。

**📈 对比分析**

通过可靠性指标（可解析性、结构对齐、修复率、共识覆盖率）对单模型与多模型管道进行诊断性评估，结果显示多模型共识显著提升结构一致性与稳健性，计算成本虽上升但在时间预算内可控。

**⚠️ 局限性**

局限性包括：计算成本较高、缺乏完整真实地面真值、LLM 对空间推理能力有限、系统需在敏感数据环境下保障隐私与安全。

---

## 541. Uncovering a Winning Lottery Ticket with Continuously Relaxed Bernoulli Gates

**arXiv ID:** 2603.08914 | [PDF](https://arxiv.org/pdf/2603.08914v1)

**作者:** Itamar Tsayag `[一作]` (Bar-Ilan University), Ofir Lindenbaum `[通讯]` (Bar-Ilan University)

**通讯引用:** 619 | [OpenAlex ID](https://openalex.org/A5053239679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于连续松弛伯努利门控的全微分方法，用于在随机初始化的过参数化网络中发现强彩票子网络（SLT），只训练门控参数而不改变权重，实现大规模稀疏化。

**💡 创新点**

首次将连续松弛伯努利门控引入SLT发现，消除了非微分的梯度估计或迭代剪枝循环，使整个过程可端到端微分；并通过直接优化\ell_0正则化期望形式实现精确稀疏化。

**🔧 技术方法**

连续松弛伯努利门控（CRBG）、\ell_0正则化期望、硬Sigmoid门控、基于高斯噪声的随机门、Adam优化；不使用梯度估计器。

**📊 数据集**

MNIST（LeNet‑300‑100）、CIFAR‑10（ResNet‑50、Wide‑ResNet‑50、ViT‑base、Swin‑T）。

**📈 对比分析**

与弱彩票票方法（训练权重）和传统强彩票票方法（如edge‑popup）对比；在LeNet上取得96%准确率、45%稀疏化；ResNet‑50 83.1%准确率、91.5%稀疏化；Wide‑ResNet‑50 88%准确率、90.5%稀疏化；ViT‑base 76%准确率、90%稀疏化；Swin‑T 80%准确率、50%稀疏化；比edge‑popup在相同或更高准确率下实现几乎两倍的稀疏化。

**⚠️ 局限性**

仅针对无结构稀疏化，未验证对结构化稀疏化或其他网络类型（如图神经网络、循环网络）的适用性；门控参数训练需要额外的GPU资源；对噪声超参数和正则化强度敏感；未在大规模图像数据集（如ImageNet）上充分实验。

---

## 542. Tracing Everyday AI Literacy Discussions at Scale: How Online Creative Communities Make Sense of Generative AI

**arXiv ID:** 2603.09055 | [PDF](https://arxiv.org/pdf/2603.09055v1)

**作者:** Haidan Liu `[一作]` (Simon Fraser University), Parmit Chilana `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对80个创意类子版块122,506条帖子和1,554,368条评论的三年大规模文本分析，探讨了创作者在Reddit上如何形成并演变AI素养；

**💡 创新点**

创新点在于提出一种底层实践驱动、事件响应的AI素养框架，强调工具使用与社区互动是创作者素养的主要路径，并通过时间序列与重大AI事件关联揭示素养的动态演化；

**🔧 技术方法**

采用LDA主题建模、手工编码、Claude Sonnet 3.5大语言模型分类及时间序列分析等技术手段，实现了从宏观主题到细粒度对话标签的层层抽取；

**📊 数据集**

使用了从Reddit for Researchers项目获取的80个创意子版块的122,506条帖子（含1,554,368条评论），共覆盖2022‑2025年三年期间的AI相关讨论；

**📈 对比分析**

在对话分类上比较了规则、传统机器学习和LLM三种方法，LLM取得最高准确率81%和宏观F1 77%；在主题演化分析中通过与主要AI事件的对齐展示了主题比例与讨论量的显著波动；

**⚠️ 局限性**

局限性包括仅聚焦Reddit导致平台偏倚、LLM分类可能误解细微语义、对话仅与大型事件关联而忽略微小发展、未覆盖跨平台与非创意领域的讨论。

---

## 543. LogoDiffuser: Training-Free Multilingual Logo Generation and Stylization via Letter-Aware Attention Control

**arXiv ID:** 2603.09759 | [PDF](https://arxiv.org/pdf/2603.09759v1)

**作者:** Mingyu Kang `[一作]` (Hanyang University), Yong Suk Choi `[通讯]` (Hanyang University)

**通讯引用:** 3099 | [OpenAlex ID](https://openalex.org/A5052803083)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的多语言标识生成方法 LogoDiffuser，利用字符图像作为输入，结合核心 Token 注入与层级注意力平均，实时融合文字结构与视觉风格。

**💡 创新点**

创新点在于通过分析 MM‑DiT 的自注意力，自动识别“核心 Token”，仅注入这些高响应的注意力映射并采用层级平均来抑制注意力漂移，从而在不额外训练的情况下保持多语言文字的结构与风格一致。

**🔧 技术方法**

核心技术包括：①多模态扩散 Transformer（MM‑DiT / Stable Diffusion 3.5）；②基于注意力方差的核心 Token 选取；③核心 Token 注意力注入；④层级注意力平均；⑤无训练的控制策略。

**📊 数据集**

使用包含英、中、阿、日、韩五种语言、每种语言 50 词（共 250 词）的手工构造标识数据集，配合对应的词汇提示与字符图像。

**📈 对比分析**

与 AnyText、TextDiffuser‑2、IP‑Adapter、ControlNet 等基线进行对比；在 CLIP 对齐、OCR 准确率、F1 分数以及 MTurk 人工评估上均获得最高分，说明在文本准确性、视觉一致性与风格表达上明显优于现有方法。

**⚠️ 局限性**

局限性：核心 Token 选择与层级平均需要手工调参，对极大或复杂字体、特殊背景下的文字可能仍出现失真；此外，方法依赖 MM‑DiT 的架构，对新型扩散模型的迁移性尚未验证。

---

## 544. Tetris is Hard with Just One Piece Type

**arXiv ID:** 2603.09958 | [PDF](https://arxiv.org/pdf/2603.09958v1)

**作者:** MIT Hardness Group `[一作]` (Massachusetts Institute of Technology), Jeffery Li `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5043161001)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

暂无可用论文内容，无法总结

**💡 创新点**

暂无可用论文内容，无法总结

**🔧 技术方法**

暂无可用论文内容，无法总结

**📊 数据集**

暂无可用论文内容，无法总结

**📈 对比分析**

暂无可用论文内容，无法总结

**⚠️ 局限性**

暂无可用论文内容，无法总结

---

## 545. Strategically Robust Multi-Agent Reinforcement Learning with Linear Function Approximation

**arXiv ID:** 2603.09208 | [PDF](https://arxiv.org/pdf/2603.09208v1)

**作者:** Jake Gonzales `[一作]` (University of Washington), Lillian J. Ratliff `[通讯]` (University of Washington)

**通讯引用:** 1215 | [OpenAlex ID](https://openalex.org/A5008161296)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在带有线性函数逼近的大规模或连续状态空间下，使用风险敏感量化响应均衡（RQRE）来实现多智能体马尔可夫游戏的稳健学习，并提出了带风险和逼近误差的乐观值迭代算法。

**💡 创新点**

创新点包括：①在RQRE框架下首次给出带风险和逼近误差的有限样本博弈损失（regret）上界；②证明RQRE对应的策略映射是Lipschitz连续的，克服了Nash均衡的多解与不稳定问题；③将RQRE与分布式鲁棒优化（DRO）联系起来，展示其更广泛的鲁棒性；④实验表明RQRE在自对弈中与Nash相当，但在交叉对弈和对手扰动下显著更稳健。

**🔧 技术方法**

技术手段包括：线性马尔可夫游戏假设、风险敏感贝尔曼递归、分布式风险测度（如熵/KL风险）、量化响应均衡求解子程序、乐观值迭代、经验分布估计、覆盖数分析和凸优化的Lipschitz性质证明。

**📊 数据集**

数据集与环境：1）动态Stag Hunt（Melting Pot框架改造版）；2）Overcooked（JaxMARL实现）——这两个多智能体协作基准环境。

**📈 对比分析**

方法比较：将RQRE与NQOVI（Nash Q-learning）和QRE（无风险量化响应均衡）进行对比；自对弈时性能相近；交叉对弈和对手扰动时，RQRE在保持较高奖励的同时显著提高鲁棒性，尤其在风险参数适中时表现最优。

**⚠️ 局限性**

局限性：①需要线性可表示的奖励与转移，限制了对非线性/高维环境的直接应用；②风险与正则化参数需手工调节，缺乏自动化选择机制；③理论与实验均针对有限时限马尔可夫游戏，扩展到无穷期或更复杂的动态环境仍待研究。

---

## 546. BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers

**arXiv ID:** 2603.09582 | [PDF](https://arxiv.org/pdf/2603.09582v1)

**作者:** Chaodong Xiao `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 106662 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 BinaryAttention，一种 1-bit 二值化查询-键注意力机制，利用符号量化和位运算显著加速视觉和扩散 Transformer 的注意力计算，同时保持甚至提升模型性能。

**💡 创新点**

创新点包括：①理论证明二值化注意力能保留相似性关系；②引入可学习偏置以补偿信息损失并避免注意力分布塌陷；③结合量化感知训练和自蒸馏技术，确保 1-bit 表示与全精度相符；④在 FlashAttention2 核上实现硬件友好、2× 速度提升。

**🔧 技术方法**

使用的技术主要有：1-bit 标量化的 Q/K（scaled sign），偏置增强，8-bit 量化的注意力系数和 value，位运算（XNOR+popcount），量化感知训练（QAT），自蒸馏，FlashAttention2 核的 CUDA/TensorCore 加速。

**📊 数据集**

评估数据集包括 ImageNet-1K（分类）、COCO 2017（目标检测/实例分割）、ADE20K（语义分割）以及 ImageNet 256×256（类条件图像生成）。

**📈 对比分析**

与 FlashAttention2、SageAttention、PTQ4ViT 等基线在相同硬件（A100 GPU）上对比，BinaryAttention 在 ViT、DiT、SiT 等模型中实现了 1.5–2× 的速度提升、显存利用率提升或维持不变，并在分类、检测、分割、生成任务上达到或超过全精度模型的 Top‑1、mAP、mIoU、FID 等指标。

**⚠️ 局限性**

局限性包括：PV 乘法仍使用相对保守的 8‑bit 量化，端到端加速空间有限；仅针对注意力层进行量化，未联合量化 MLP 等其他模块；高分辨率时可学习偏置占用的显存较大；在极低精度（1‑bit）下的鲁棒性仍有提升空间。

---

## 547. Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval

**arXiv ID:** 2603.09250 | [PDF](https://arxiv.org/pdf/2603.09250v1)

**作者:** Yingyi Zhang `[一作]` (Dalian University of Technology), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6177 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于人类双重记忆过程的检索框架 RF-Mem，用熟悉度不确定性动态切换快速识别（Familiarity）与逐步回忆（Recollection）两条检索路径，以实现个性化大语言模型的高效记忆检索。

**💡 创新点**

创新点：①将认知科学的熟悉度‑回忆双重模型引入检索控制；②利用检索结果的均值和熵量化熟悉度，以阈值和熵门控实现路径自适应切换；③在回忆路径中通过 KMeans 聚类与 α‑mix 逐轮扩展查询，形成类似链式的证据检索；④在保持单查询延迟的前提下，显著提升对大规模（1M 词）个性化记忆的检索准确率。

**🔧 技术方法**

技术：向量检索（近似最近邻 ANN）、KMeans 聚类、α‑mix 查询混合、熵与均值阈值控制、迭代检索（Beam width B、fanout F、最大轮数 R）。

**📊 数据集**

数据集：PersonaMem（多任务用户对话记忆，32K/128K/1M 词）、PersonaBench（私有用户文档+查询）、LongMemEval（长周期记忆检索评测）。

**📈 对比分析**

对比方法：零记忆、全上下文、密集检索（单路 Familiarity）和仅回忆（Recollection）。在 PersonaMem 生成任务上，RF-Mem 在 32K、128K、1M 词时分别提升 2.2%、1.3%、0.7% 的整体准确率，同时保持 5–7 ms 的检索延迟；在 PersonaBench 与 LongMemEval 的 Recall@K 评测中，RF-Mem 通常匹配或优于两种单路基线，且延迟比 Recollection 低约 30–40%。

**⚠️ 局限性**

局限：①依赖预训练的向量编码器，若编码质量差会影响熟悉度估计；②回忆路径的聚类与混合参数需要调优，过深或过浅会导致效率或召回率折衷；③对极端稀疏或噪声记忆集合时，熵门控可能误判；④目前未对实时动态记忆更新做深入研究，仍假设记忆库固定。

---

## 548. VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs

**arXiv ID:** 2603.08936 | [PDF](https://arxiv.org/pdf/2603.08936v1)

**作者:** Hezhao Zhang `[一作]` (University of Sheffield), Thomas Hain `[通讯]` (University of Sheffield)

**通讯引用:** 4716 | [OpenAlex ID](https://openalex.org/A5030528300)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 VoxEmo 评测工具箱，系统地对 35 个跨语言、跨情感标签集的语音情感识别数据集进行零射击与监督微调实验，并给出了统一的提示模板、输出解析和分布式评估流程。

**💡 创新点**

创新点包括：①构建跨 15 种语言、35 个语料库的统一 benchmark；②引入软标签分布评估与提示集成策略；③设计针对语音 LLM 的标准化推理协议；④提供对比基准（如 EmoBox）和跨域迁移实验，展示模型在跨标签集、跨场景下的泛化能力。

**🔧 技术方法**

采用 Qwen2‑Audio‑7B‑Instruct 与 Audio‑Flamingo‑3 两大语音 LLM，利用 Whisper‑large‑v3 编码器、7B 参数量；通过 LoRA（r=8, α=16）实现监督微调；实现多种提示模板（直接、ASR、声学描述、推理）并构建 5 倍提示集成；使用 KL‑div、JSD、TVD、余弦相似度和 MSE 等分布度量来评估软标签一致性。

**📊 数据集**

使用了 35 个公开情感语料库（包含 7 个真实环境、28 个演员录制），覆盖 15 种语言，主要包括 CREMA‑D、IEMOCAP、MSP‑Podcast、EmotionTalk、CES‑等；其中 5 个语料库提供多注释者分布用于软标签评估。

**📈 对比分析**

比较方法：在 35 份数据集上分别进行零射击（5 种提示）和 LoRA 微调；指标包括加权准确率（WA）、无权准确率（UA）、宏/微 F1、top‑1 准确率以及分布度量；实验结果显示零射击性能普遍低于传统监督基线，但通过提示集成后可显著提升；Qwen2‑Audio 在多数数据集上优于 Audio‑Flamingo‑3，微调后可缩小与 SOTA 差距至 15–30% 左右，但在小规模或高不确定性数据集仍落后。

**⚠️ 局限性**

局限性：仅评估两款同构 7B LLM，未覆盖更大规模或 tokenizer‑based 语音 LLM；LoRA 超参数固定，可能影响 AF3 的微调效果；软标签评估仅限 5 份语料库；仅提供宏观指标，缺乏类内/说话人层面的细粒度分析；未验证多模型或多前端的泛化能力。

---

## 549. FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation

**arXiv ID:** 2603.09046 | [PDF](https://arxiv.org/pdf/2603.09046v1)

**作者:** Yinpeng Wu `[一作]` (Shanghai Jiao Tong University), Yubin Xia `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8999 | [OpenAlex ID](https://openalex.org/A5026023746)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出FlexServe，一套基于ARM TrustZone虚拟化的快速安全设备端LLM推理系统，能够在受损的OS内核环境下保护模型权重与用户数据；

**💡 创新点**

核心创新在于引入可灵活切换的资源隔离机制——Flexible Secure Memory与Flexible Secure NPU，实现页级安全内存和动态NPU模式切换；采用按需保护、流水线推理框架、LLM感知内存管理及多模型调度器，显著提升性能与资源利用率；

**🔧 技术方法**

技术手段包括ARM TrustZone、EL2层轻量级虚拟化、Stage-2页表(S2PT)与SMMU配置、按需保护、可信执行环境TA、预取流水线、NPU驱动复用及逆向工程NPU运行时；

**📊 数据集**

实验使用Llama3与Qwen3系列（3B、8B、0.6B、1.7B、8B）INT8量化模型；在真实代理工作流（UltraChat、OpenAssistant、Dolly、Alpaca）中进行评测；

**📈 对比分析**

对比基线NW-Base（非安全）、Strawman（TrustZone单纯安全）和Strawman-OPT（安全+流水线+安全NPU），FlexServe在单模型TTFT上平均提升10.05×(vs Strawman)、2.44×(vs Strawman-OPT)，在多模型端到端延迟上提升至24.30×和4.05×；对常规应用的内存占用开销降低至1.31×，相较Strawman的3.27×；

**⚠️ 局限性**

局限性包括无法保护正常世界客户端I/O、侧信道与物理攻击不在范围内、容易受DoS攻击、依赖TrustZone硬件、当前仅支持CPU+NPU（GPU支持待扩展）、对模型类型与加速器的支持有限；

---

## 550. DISPLAY: Directable Human-Object Interaction Video Generation via Sparse Motion Guidance and Multi-Task Auxiliary

**arXiv ID:** 2603.09883 | [PDF](https://arxiv.org/pdf/2603.09883v1)

**作者:** Jiazhi Guan `[一作]` (Baidu Inc), Jingdong Wang `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DISPLAY 框架，利用稀疏运动指导实现可控的人物-物体交互视频生成。

**💡 创新点**

创新点包括：① 仅使用手腕坐标和形状无关的目标框进行稀疏运动指导，消除手体与物体表征不平衡；② 引入 Object‑Stressed Attention 强化物体相关注意力；③ 通过多任务辅助训练与精细数据清洗提升在稀缺 HOI 数据下的泛化。

**🔧 技术方法**

技术方案包括基于预训练流匹配 Diffusion Transformer（DiT）的 ControlNet 结构、VAE 编码、Object‑Stressed Attention 机制、稀疏运动编码与背景/视觉参考的多模态条件注入。

**📊 数据集**

数据集为作者自建的约 100 小时 HOI 标注视频（含手腕轨迹、物体分割）+ 50 小时通用人类视频，采用 Grounding‑DINO、SAM2 等工具完成标注。

**📈 对比分析**

与 VACE‑14B、HunyuanCustom、HuMo、WanAnimate、Re‑HOLD、AnchorCraft 等方法进行定量比较，显示在 FID、AES、FVD、HF、CA、O‑CLIP/O‑DINO 等指标上均达到或逼近最优，证明生成质量、物体保真度与交互一致性均显著提升。

**⚠️ 局限性**

局限性：① 稀疏手腕轨迹难以精细控制手部姿态；② 仍需大量 HOI 训练样本，非刚性物体或复杂场景效果待进一步验证；③ 对极端光照或遮挡的鲁棒性有限。

---

## 551. Telogenesis: Goal Is All U Need

**arXiv ID:** 2603.09476 | [PDF](https://arxiv.org/pdf/2603.09476v1)

**作者:** Zhuoran Deng `[一作]` (Independent Research), Wan Shen `[通讯]` (Independent Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了基于知识缺口（不确定性、惊奇与时效性）生成注意力优先级的Telogenesis框架，并在最小系统和模块化部分可观测环境中验证其有效性；

**💡 创新点**

创新点在于将三类认知缺口统一为单一优先级函数，展示评价指标（全局误差与检测延迟）下性能逆转，并通过自学习的衰减率无监督恢复环境波动结构；

**🔧 技术方法**

使用贝叶斯世界模型、软最大化选择、蒙特卡罗仿真、组件消融、幂律分析以及惊奇加权的衰减率更新；

**📊 数据集**

实验数据集为合成环境：最小系统（N=6）和Liminal模块化环境（N=16，4个模块），无真实世界数据集；

**📈 对比分析**

与随机、仅方差、循环、错误驱动等基线策略通过全局预测误差和检测延迟两种指标比较，结果显示在全局误差下循环优先；在检测延迟下，优先级策略平均保持≈4步检测延迟，且其幂律指数为0.55（优于循环0.40），表现更优；

**⚠️ 局限性**

局限性包括需预设贝叶斯模型、惊奇加权的衰减率更新仅为启发式、检测延迟仍需外部阶段切换标注、仅在合成任务验证、未测试对复杂学习模型或行动规划的扩展。

---

## 552. EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation

**arXiv ID:** 2603.09385 | [PDF](https://arxiv.org/pdf/2603.09385v1)

**作者:** Yinrui Ren `[一作]` (Southern China Normal University), Hui Xiong `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6853 | [OpenAlex ID](https://openalex.org/A5056768519)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用无标注的跨模态蒸馏，将图像基模型VGGT的多视角几何先验迁移到事件摄像机，得到时序一致、仅靠事件数据即可生成高精度深度图，并可进一步扩展至姿态与点云估计。

**💡 创新点**

三层蒸馏策略（CMFM、STFD、TCD）使事件流被显式视为连续视频序列；跨模态特征混合突破模态鸿沟；时间一致蒸馏保证深度预测的连续性和几何一致性；实现无RGB输入、无标注、零样本迁移。

**🔧 技术方法**

事件帧化、Alternating‑Attention Transformer、跨模态特征混合、特征层蒸馏、时间一致蒸馏、LoRA 微调、VGGT 作为教师网络。

**📊 数据集**

主要使用 EventScape（训练/验证）、MVSEC（评估与零样本）以及 DENSE（零样本）数据集；同时对 VGGT 原始 RGB 训练集进行对比。

**📈 对比分析**

在 EventScape 上对比 EventDAM、DepthAnyEvent 等基线，30 m 绝对误差从 2.30 m 降至 1.06 m（53% 改进）；在 MVSEC 夜间场景中亦显著降低误差；在 DENSE 进行零样本测试，表现超过所有多模态基线；整体提升显著且具备强泛化能力。

**⚠️ 局限性**

继承了 VGGT 的远场深度压缩偏差，导致极远景深度略低估；对极端光照/稀疏事件分布仍有鲁棒性不足；未利用真实稠密深度进行后期校准。

---

## 553. Robust Provably Secure Image Steganography via Latent Iterative Optimization

**arXiv ID:** 2603.09348 | [PDF](https://arxiv.org/pdf/2603.09348v1)

**作者:** Yanan Li `[一作]`, Yanzhen Ren `[通讯]` (Wuhan University)

**通讯引用:** 563 | [OpenAlex ID](https://openalex.org/A5101499462)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在接收端对潜在空间中的变量进行迭代优化，提升图像压缩或格式转换后信息提取准确率，且不改动原始嵌入逻辑，保持可证明安全性。

**💡 创新点**

提出的潜在空间迭代优化（固定点迭代 + 梯度下降）可在保持安全保障的前提下显著增强鲁棒性，且可作为独立模块应用于其他可证明安全的隐写方案。

**🔧 技术方法**

基于扩散模型的潜在空间迭代优化、梯度反向传播、固定点迭代、概率积分变换以及 KL 散度证明安全。

**📊 数据集**

使用 Stable Diffusion 2.1 的潜在空间（4×64×64），图像生成以 COCO 数据集为基础。

**📈 对比分析**

与 Hu 等人原始框架对比，实验显示在 TIFF、PNG、JPEG（Q90/Q70/Q50）等多种压缩格式下，经过 100 步迭代优化后提取准确率提升 3–6% 以上，基本接近完美。

**⚠️ 局限性**

主要限制是需要额外的计算时间和显存来完成多步迭代，且在极低质量压缩（如 JPEG50）提升有限，未提供全局收敛保证。

---

## 554. Common Sense vs. Morality: The Curious Case of Narrative Focus Bias in LLMs

**arXiv ID:** 2603.09434 | [PDF](https://arxiv.org/pdf/2603.09434v1)

**作者:** Saugata Purkayastha `[一作]`, Sukannya Purkayastha `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CoMoral基准，用于评估LLM在道德困境场景中识别常识矛盾的能力。

**💡 创新点**

首次发现并量化“叙事焦点”偏差，即LLM对叙述者和次要角色的常识矛盾识别存在显著差异；同时指出对常识推理的依赖需要显式提示。

**🔧 技术方法**

利用指令调优的大型语言模型（如LLaMA、Qwen2、Gemma）进行实验，采用显式与隐式提示，并使用LLM‑as‑a‑Judge评估器进行自动评分。

**📊 数据集**

构造了802条半自动生成的实例，包含“常识矛盾+道德困境”两种角色视角，涵盖物理、环境、时间、社会、概念、非现实等八类常识。

**📈 对比分析**

在隐式提示下各模型准确率低于0.3，显式提示下最高可达LLaMA 8B 0.845；所有模型在识别次要角色矛盾时均优于叙述者，表明叙事焦点偏差。

**⚠️ 局限性**

局限性包括数据集规模有限（802条）、未评估更大模型（30B/80B）、只关注一般道德困境，缺乏专业领域覆盖，且理论上对叙事焦点偏差的解释仍待深入。

---

## 555. RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation

**arXiv ID:** 2603.09723 | [PDF](https://arxiv.org/pdf/2603.09723v1)

**作者:** Sihong Wu `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 274764 | [OpenAlex ID](https://openalex.org/A5042321575)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RbtAct 框架，利用作者反驳信息作为隐式监督，通过偏好优化生成可操作的同行评审反馈。

**💡 创新点**

首次在段落级别将反驳映射到评审片段，并将影响类别作为偏好排序，利用反驳驱动的偏好学习提升可操作性。

**🔧 技术方法**

先在 Llama‑3.1‑8B‑Instruct 上进行监督微调，再使用 DPO（直接偏好优化）在评审片段上进行偏好训练。

**📊 数据集**

构建 Review‑Map‑Rebuttal（RMR‑75K）数据集，包含 75,542 个评审片段、对应反驳片段、视角标签和影响类别。

**📈 对比分析**

与 SFT‑only、prompted LLMs（GPT‑5‑chat、Llama‑3.1‑70B 等）以及其他任务适配方法对比，基于人类评估和 LLM‑judge，两种评测均显示 RbtAct 在可操作性与具体性上最高，性能与 32‑70B 大模型相当。

**⚠️ 局限性**

依赖公开反驳，受限于会议/期刊、语言与领域；反驳仅反映短期作者响应，可能包含策略性承诺；模型可能提出不切实际建议，未与代码或数据进行严格验证。

---

## 556. Decoder-Free Distillation for Quantized Image Restoration

**arXiv ID:** 2603.09624 | [PDF](https://arxiv.org/pdf/2603.09624v1)

**作者:** S. M. A. Sharif `[一作]` (Opt-AI Inc.), Jaeho Lee `[通讯]` (Opt-AI Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对边缘设备的图像恢复任务，提出 QDR 框架，利用量化感知知识蒸馏实现 Int8 级别的高质量恢复。

**💡 创新点**

创新点包括：1）Decoder‑Free Distillation (DFD)，只在瓶颈层进行蒸馏，避免解码器中误差放大；2）Learnable Magnitude Reweighting (LMR)，自适应平衡重建与蒸馏梯度，稳定 QAT‑KD 训练；3）Edge‑Friendly Model (EFM) 搭配轻量化的 Learnable Degradation Gating (LDG)，实现高效推理并提升局部降解处理。

**🔧 技术方法**

采用量化感知训练 (QAT)、自蒸馏、损失梯度重加权、轻量化 U‑Net 结构与可学习跳连门控技术；使用 Int8 权重与激活量化。

**📊 数据集**

在四大降解任务上进行实验：低光增强 (LOL‑v1)、去雾 (SOTS)、去雨 (Rain100H)、去噪 (SIDD)。

**📈 对比分析**

与 FP32、PTQ、QAT、QAT+KD 等基线对比，Int8 模型在四个任务中平均恢复 PSNR 仅比 FP32 少约 0.6 dB，恢复质量达 96.5% FP32，Jetson Orin 上 442 FPS，且在低光环境下作为预处理可提升 YOLOv5 mAP 16.3%（+0.16 mAP）。

**⚠️ 局限性**

局限性：仅针对单一降解任务评估；对极端低位宽（2/4 bit）仍存在性能衰减；框架对多重降解或不同硬件平台的适应性尚待进一步验证。

---

## 557. Multi-DNN Inference of Sparse Models on Edge SoCs

**arXiv ID:** 2603.09642 | [PDF](https://arxiv.org/pdf/2603.09642v1)

**作者:** Jiawei Luo `[一作]` (University of St Andrews), Blesson Varghese `[通讯]` (University of St Andrews)

**通讯引用:** 4234 | [OpenAlex ID](https://openalex.org/A5074563743)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一套多DNN推理系统，利用模型拼接（model stitching）扩展稀疏模型变体，并在边缘SoC上实现高效调度与预加载。

**💡 创新点**

创新点在于训练无关的模型拼接技术，可在不再训练的情况下组合稀疏子图生成数十甚至数百个变体；同时提出联合优化处理器放置、变体选择和热子图预加载的三模块系统。

**🔧 技术方法**

采用子图级别的准确率与延迟估计（XGBoost回归、子图延迟求和）、稀疏感知处理器放置优化（全局序列搜索）和热度评分预加载（贪婪选择）等技术。

**📊 数据集**

使用四个主流任务的公开数据集：ImageNet-1K/ResNet-101，SST-2/BERT-Base，HAR/ViT-Small，LibriSpeech/Wav2vec2。

**📈 对比分析**

在Intel Core Ultra 7 265K、Intel Core Ultra 5 135U、NVIDIA Jetson AGX Orin三种边缘SoC上与六类基线（SV-AO/LO-P/N，AV-P/N）对比，SLO违约率下降最多74%，吞吐量提升最多2.31倍，内存占用平均降低28%。

**⚠️ 局限性**

主要局限在仅针对中小型边缘模型，无法覆盖大型基础模型；对多任务动态SLO变化的实时重调度成本和跨平台精度迁移尚未充分评估；并且在极低功耗设备上的能耗分析缺失。

---

## 558. Meissa: Multi-modal Medical Agentic Intelligence

**arXiv ID:** 2603.09018 | [PDF](https://arxiv.org/pdf/2603.09018v1)

**作者:** Yixiong Chen `[一作]` (Johns Hopkins University), Alan Yuille `[通讯]` (Johns Hopkins University)

**通讯引用:** 106740 | [OpenAlex ID](https://openalex.org/A5086706224)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

我们训练了一个4B参数的多模态医疗代理模型Meissa，通过对前沿代理系统生成的交互轨迹进行蒸馏，实现在离线部署下的完整代理能力。

**💡 创新点**

创新点在于提出统一轨迹建模、分层监督和前向-后向监督三种技术，能够在单一模型中学习何时与如何进行多步骤交互，并跨越工具调用、视觉推理、多代理协作与临床模拟四种环境。

**🔧 技术方法**

使用了策略分层的行为克隆、前向与后向轨迹监督、基于Qwen3-VL-4B的LoRA微调以及自定义的状态-动作-观察轨迹格式。

**📊 数据集**

利用约40K条从Gemini-3-flash生成的多模态代理轨迹，覆盖13个医学基准（包括胸腔X光、病理、临床问答和多轮临床模拟）。

**📈 对比分析**

与GPT-4o、Gemini-3-flash等前沿模型以及多代理框架进行对比，在10/16项评测中获得与或优于对手的成绩，且在OOV基准上实现了62.8%的准确率，离线推理延迟比云端API低约22倍。

**⚠️ 局限性**

局限在于对深度参数知识和专家临床推理的依赖不足，缺乏置信度校准和主动放弃机制，且对极端OOD情况的鲁棒性仍待提升。

---

## 559. The Future of Software Engineering Conferences: A New Zealand Perspective

**arXiv ID:** 2603.09035 | [PDF](https://arxiv.org/pdf/2603.09035v1)

**作者:** Kelly Blincoe `[一作]` (University of Auckland), Amjed Tahir `[通讯]` (Massey University)

**通讯引用:** 788 | [OpenAlex ID](https://openalex.org/A5025562598)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文分析了新西兰软件工程研究者在国际会议参与中面临的地理、财政、签证和学术日历等障碍，并提出了混合参与、地理多样化、灵活定价和治理改革等对策；

**💡 创新点**

创新点在于从新西兰视角系统梳理多维障碍，并基于ICSE/FSE/ASE等旗舰会议的地点旋转机制提出可操作的包容性和可持续性改进方案；

**🔧 技术方法**

主要采用文献综述与案例分析技术，结合会议地点与时间数据进行阐释；

**📊 数据集**

使用的主要数据集为ICSE、FSE、ASE等会议过去十年的举办地点与年份表；

**📈 对比分析**

与传统会议模式进行对比，论证混合模式和地理多样化能够降低成本、提升碳足迹并扩大参与度，但未给出量化性能指标；

**⚠️ 局限性**

局限性包括：研究聚焦新西兰，缺乏跨地区实证验证；未进行实验或大规模数据分析，仅为概念性建议。

---

## 560. SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments

**arXiv ID:** 2603.09460 | [PDF](https://arxiv.org/pdf/2603.09460v1)

**作者:** Shiyi Chen `[一作]` (Tsinghua University), Chun Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 19517 | [OpenAlex ID](https://openalex.org/A5100454937)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了SEA-Nav框架，实现了在高度拥挤环境中使用深度强化学习进行四足机器人安全、高效、敏捷的导航。

**💡 创新点**

创新点包括：①自适应碰撞状态初始化ACSI，用于提升稀有高危经验采样；②端到端可微分的LSE-CBF安全屏障层；③运动学正则化损失提高仿真到实物的迁移。

**🔧 技术方法**

采用PPO强化学习、可微分控制屏障(CBF)与LSE平滑、多约束融合、以及动作正则化损失，在Isaac Gym仿真下使用LiDAR感知。

**📊 数据集**

在Isaac Gym的10×10房间环境中，随机化障碍占比为Easy/Medium/Hard，训练和测试均使用该数据集。

**📈 对比分析**

与SOTA方法对比（如基于VO或传统CBF的方案），SEA-Nav在成功率上更高、碰撞率更低、训练时间仅数十分钟，在真实Unitree Go2上实现零射击部署。

**⚠️ 局限性**

局限性：仅支持平地导航，缺乏坡道/楼梯检测，仍易在复杂迷宫或死胡同中卡住；对动态障碍物适应性有限。

---

## 561. Adaptive Multi-Objective Tiered Storage Configuration for KV Cache in LLM Service

**arXiv ID:** 2603.08739 | [PDF](https://arxiv.org/pdf/2603.08739v1)

**作者:** Xianzhe Zheng `[一作]` (Zhejiang University), Feifei Li `[通讯]` (Alibaba Group)

**通讯引用:** 216293 | [OpenAlex ID](https://openalex.org/A5100450462)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一套用于大语言模型推理服务的自适应KV缓存多层存储配置框架（Kareto），能够动态平衡GPU HBM、主机DRAM与磁盘等异构存储层的容量与访问策略；

**💡 创新点**

创新点包括：①将多目标（吞吐量、延迟、成本）Pareto前沿搜索转化为仿真驱动的优化问题；②提出基于递减收益的裁剪策略，显著降低搜索空间并保持前沿质量；③引入细粒度的分组TTL调优，依据前缀树访问模式实现针对性缓存保留；

**🔧 技术方法**

使用高保真全流程仿真器、递减收益剪枝的自适应搜索算法、SLSQP多起点优化、前缀树分析分组TTL以及Pareto前沿选择机制；

**📊 数据集**

使用真实业务生产轨迹，包括交互式聊天（Trace A）、API调用（Trace B）和智能代理（Trace C）三种典型负载；

**📈 对比分析**

与传统固定容量（1024 GB DRAM）基线对比，Kareto在吞吐量上提升最多9.3%，在平均首个Token延迟上降低最多58.3%，在整体成本上减少最多20.2%；

**⚠️ 局限性**

局限性包括：仿真仍耗时且依赖历史轨迹，可能无法捕捉极端或突发负载；TTL分组在某些工作负载（如交互式聊天或高并发低排队场景）收益有限；算法对存储层次结构的假设与云平台定价变化可能影响迁移性。

---

## 562. Dishonesty Tendencies in Testing Scenarios Among Students with Virtual Reality and Computer-Mediated Technology

**arXiv ID:** 2603.08974 | [PDF](https://arxiv.org/pdf/2603.08974v1)

**作者:** Tanja Kojić `[一作]` (TU Berlin), Jan-Niklas Voigt-Antons `[通讯]` (Hamm-Lippstadt University of Applied Sciences)

**通讯引用:** 1638 | [OpenAlex ID](https://openalex.org/A5063539298)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对22名学生进行实验，比较他们在虚拟现实（VR）环境与传统电脑浏览器环境下的考试作弊行为与心理体验。

**💡 创新点**

首次在真实观察者在场的条件下对比VR与电脑媒介技术的作弊率与感知差异，填补了此前仅在虚拟观察者环境下的研究空白。

**🔧 技术方法**

使用VR头显与对应控制器（未指定具体品牌），电脑浏览器；收集屏幕录像、浏览器历史、问卷（SAM、IPQ、Self‑Perception of Lying 等）数据。

**📊 数据集**

自定义数据集：22名学生的实验记录（作弊次数、问卷得分、浏览器历史等），无公开公开数据集使用。

**📈 对比分析**

通过独立样本t检验比较VR与Laptop条件下的问卷分数，发现多项感知变量显著差异；作弊频率数据显示两种环境下作弊比例基本相同。整体性能即两环境下作弊率相近。

**⚠️ 局限性**

局限性包括：伦理约束导致屏幕录制可能影响行为；实验者与受试者可能熟悉，降低受试者压力；实验不是正式考试，缺乏真实考试压力；样本量小，仅两人实验，无法推广至更大规模。

---

## 563. Context Engineering: From Prompts to Corporate Multi-Agent Architecture

**arXiv ID:** 2603.09619 | [PDF](https://arxiv.org/pdf/2603.09619v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 564. Open-World Motion Forecasting

**arXiv ID:** 2603.09420 | [PDF](https://arxiv.org/pdf/2603.09420v1)

**作者:** Nicolas Schischka `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2586 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种端到端的开放世界运动预测框架，支持在仅使用前置摄像头数据的情况下逐步引入新类别并预测未来轨迹。

**💡 创新点**

创新点包括：使用基于未来检测的伪标签生成与 Vision‑Language 模型过滤以提升伪标签质量；基于查询特征方差的序列经验回放策略；以及实现零样本迁移与可扩展到增量规划。

**🔧 技术方法**

采用基于 DETR 的 3D 检测和运动预测网络，配合 Grounded SAM 等 Vision‑Language 模型进行伪标签过滤，利用查询方差度量构建经验回放，训练采用 AdamW、余弦学习率退火等技术。

**📊 数据集**

在 nuScenes 和 Argoverse 2 两个真实驾驶数据集上进行实验，构建逐类增量和分组增量数据拆分。

**📈 对比分析**

与联合训练上限以及 CL‑DETR 等基线对比，Ours 在 mAP_f、AP_f 以及消除灾难性遗忘方面表现最优，接近全标注训练水平。

**⚠️ 局限性**

局限性包括：需要原始标注来生成新类别伪标签；未使用 HD 地图或在线地图构建；在类别消失时可能导致更大遗忘；以及与实际驾驶域仍存在差距。

---

## 565. Expressive Power of Property Graph Constraint Languages

**arXiv ID:** 2603.09806 | [PDF](https://arxiv.org/pdf/2603.09806v1)

**作者:** Stefania Dumbrava `[一作]` (Paris Cité University), Steven Sailly `[通讯]` (Telecom SudParis)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文首次对属性图约束语言 PG-Keys、GFD 与 GGD 的表达能力进行系统比较，提出了统一框架并给出严格的表达式层次与分离结果。

**💡 创新点**

创新点在于：①将三种约束语言映射到同一参数化框架；②引入“共享变量数”作为决定表达力的关键维度；③证明在该框架下 PG-Keys 的各种断言关键词仅是语法糖，能够用 1‑共享变量约束完全模拟；④构造图示例实现分离证明，得出完整的严格层次结构。

**🔧 技术方法**

使用的技术主要是形式化定义（CRPQ、等价/不等价谓词、参数化约束语法）、结构化的翻译构造、归约与构造证明、图构造与同构论证，辅以闭包性与诱导子图性质。

**📊 数据集**

本文并未使用实际数据集，而是通过抽象的属性图构造（如完整图、团、路径、边组）来进行理论证明与表达式比较。

**📈 对比分析**

比较方法是通过正式的表达式包含与不包含关系，利用翻译与分离证明得到严格的层次图；性能方面无实验测评，讨论集中在理论表达式复杂度与归约结果。

**⚠️ 局限性**

局限性包括：仅考虑 CRPQ + (不)等价谓词，未覆盖 Cypher/GQL 的完整语义（如两向路径、最短/唯一路径）；未探讨连通性限制对表达力的影响；未给出复杂度按共享变量数参数化的分析；缺乏实验验证与实现评估。

---

## 566. Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging

**arXiv ID:** 2603.09137 | [PDF](https://arxiv.org/pdf/2603.09137v1)

**作者:** Mohseu Rashid Subah `[一作]` (Purdue University), Rachel K. Surowiec `[通讯]` (Purdue University)

**通讯引用:** 807 | [OpenAlex ID](https://openalex.org/A5047036467)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种基于Transformer的SegFormer网络，对HR‑pQCT图像进行全自动多区域分割（含皮肤、肌腱、脂肪等软组织），并提取分割区域的radiomics特征，利用机器学习对骨质疏松进行二分类；

**💡 创新点**

首次在HR‑pQCT上实现多区域（骨与软组织）全自动分割，并发现软组织radiomics特征对骨质疏松的预测价值显著高于传统骨部位特征，构建完整从分割到分类的端到端流程；

**🔧 技术方法**

使用Transformer‑based SegFormer分割网络、后处理软组织分割、PyRadiomics提取939个特征、LASSO+特征筛选、六种机器学习分类器（LR、SVM、RF、XGBoost、KNN、NB）以及多元逻辑回归；

**📊 数据集**

使用两中心HR‑pQCT数据集：40份下肢扫描（6,720张2D图像）用于分割训练；122份扫描（20,496张图像）用于骨质疏松分类；；

**📈 对比分析**

与U‑Net、Attention‑U‑Net对比，SegFormer在纤维骨等少量样本区提升IoU约20%；图像级分类中，软组织LR模型达到AUROC 0.85；患者级别软组织radiomics模型实现AUROC 0.875，显著优于传统DXA与HR‑pQCT参数；

**⚠️ 局限性**

仅完成二分类任务，患者样本量有限且单中心，HR‑pQCT临床可用性受限，缺乏多中心或跨模态泛化验证。

---

## 567. FormalRTL: Verified RTL Synthesis at Scale

**arXiv ID:** 2603.08738 | [PDF](https://arxiv.org/pdf/2603.08738v1)

**作者:** Kezhi Li `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14271 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为FormalRTL的多代理框架，实现了基于软件参考模型的RTL代码生成与形式验证

**💡 创新点**

创新点在于将软件参考模型作为可执行规范进行规划、生成和调试，结合静态分析、正式等价检查和基于counterexample的自动调试，解决工业级数据路径设计的规模与正确性问题

**🔧 技术方法**

使用了大型语言模型（GPT‑4.1/GPT‑5）、LangChain、Clang静态分析、hw‑cbmc等形式验证工具、自动化调试工具（bug locator、counterexample simplifier）

**📊 数据集**

构建并公开了一个工业级基准套件，涵盖FP16、Hifloat8等数据路径密集型模块，包含规格说明和对应的C/C++参考模型

**📈 对比分析**

与仅基于自然语言规范的传统RTL生成方法进行对比，实验显示FormalRTL在初始成功率、调试迭代次数和最终成功率上均优于基线；在合成后的面积/延迟上略逊于人工优化，但提供了形式化验证的可靠基础

**⚠️ 局限性**

仍存在可扩展性挑战（完整工业规模RTL生成尚未覆盖）、对大型商业LLM的依赖、缺乏成熟的开源C‑RTL等价检查工具等局限

---

## 568. SPAARS: Safer RL Policy Alignment through Abstract Exploration and Refined Exploitation of Action Space

**arXiv ID:** 2603.09378 | [PDF](https://arxiv.org/pdf/2603.09378v1)

**作者:** Swaminathan S K `[一作]` (Indian Institute of Technology Kharagpur), Aritra Hazra `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 384 | [OpenAlex ID](https://openalex.org/A5022880201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在离线与在线强化学习的桥接中，本文提出了一种框架，通过先在由 CVAE 学到的低维行为流形上安全探索，再利用优势门控策略逐步切换到原始动作空间，实现在线微调与最优性能的兼顾。

**💡 创新点**

创新点包括：① 对 CVAE 造成的“利用上限”（exploitation gap）进行理论界定并证明其与重构误差相关；② 引入基于 Option‑Critic 的优势门控机制，取代传统的全局时间调度，实现按状态动态切换；③ 证明仅利用无序（s,a）对即可训练 CVAE，消除对轨迹分段的需求。

**🔧 技术方法**

主要技术包括：Conditional Variational Autoencoder (CVAE)、OPAL 轨迹技能预训练、SAC 强化学习、RND 内在奖励、共享 critic 与行为克隆、Option‑Critic 终止梯度、优势门控与学习率调度。

**📊 数据集**

使用了 D4RL 公开数据集：kitchen‑mixed‑v0、antmaze‑medium‑play‑v2/large‑play‑v2、hopper‑medium‑v2、walker2d‑medium‑v2；并在这些数据集上进行 OPAL 预训练和在线 fine‑tuning。

**📈 对比分析**

实验中与 SUPE、IQL、PLAS 等基线进行对比：在 kitchen‑mixed‑v0 上归一化回报从 0.75 提升至 0.825，样本效率提升 5 倍；在 hopper 与 walker2d 上分别从 66.3/78.3 提升至 92.7/102.9；在 AntMaze 上与 SUPE 对齐，优势门仅在靠近目标时触发。

**⚠️ 局限性**

局限性包括：对离线数据覆盖度要求高，演示不足或质量差时重构误差增大导致利用上限提高；优势门依赖共享 critic 的准确性，估计误差会导致误切换；理论主要针对连续动作空间，离散或高维情境的适用性尚待验证。

---

## 569. Fair and Square: Replacing One Real Multiplication with a Single Square and One Complex Multiplication with Three Squares When Performing Matrix Multiplication and Convolutions

**arXiv ID:** 2603.08732 | [PDF](https://arxiv.org/pdf/2603.08732v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 570. SurgFed: Language-guided Multi-Task Federated Learning for Surgical Video Understanding

**arXiv ID:** 2603.09496 | [PDF](https://arxiv.org/pdf/2603.09496v1)

**作者:** Zheng Fang `[一作]` (National University of Singapore), Yueming Jin `[通讯]` (National University of Singapore)

**通讯引用:** 4894 | [OpenAlex ID](https://openalex.org/A5050163233)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了SurgFed多任务联邦学习框架，解决手术视频分割与深度估计中的组织多样性与任务多样性问题。

**💡 创新点**

创新点在于引入语言引导通道选择（LCS）和语言引导超聚合（LHA）模块，利用预定义文本提示与CLIP嵌入实现跨站点、跨任务的个性化聚合与特征选择。

**🔧 技术方法**

采用了SAM2骨干网络、CLIP文本嵌入、轻量化通道选择网络、层级交叉注意力机制以及超网络等技术，结合联邦学习与多任务学习框架。

**📊 数据集**

使用了五个公开数据集：EndoVis2017、EndoVis2018、AutoLaparo、SCARED和StereoMIS。

**📈 对比分析**

与FedAvg、FedRep、FedProx、FedAvg+Cluster、MaT‑FL、FedHCA^2等方法对比，平均Δm提升至+5.92%，在分割任务的IoU/Dice显著提高，深度估计的RMSE显著下降。

**⚠️ 局限性**

局限性包括在AutoLaparo等强域移位场景下性能提升有限；模型仍受SAM2推理速度限制，且目前仅支持分割与深度估计两类任务。

---

## 571. Influence of Interactivity in Shaping User Experience and Social Acceptance of Mobile XR

**arXiv ID:** 2603.08973 | [PDF](https://arxiv.org/pdf/2603.08973v1)

**作者:** Tanja Kojić `[一作]` (Quality and Usability Lab, Technical University of Berlin), Jan-Niklas Voigt-Antons `[通讯]` (Immersive Reality Lab, Hamm-Lippstadt University of Applied Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

探究了移动增强现实应用中交互程度对用户体验与社会可接受性的影响，并分析了年龄、性别等人口因素的调节作用。

**💡 创新点**

首次将交互程度与UX与SA并行评估，揭示高交互性虽提升沉浸感与兴奋度，却伴随更高的复杂度、低效性和社会不适；并发现性别差异显著。

**🔧 技术方法**

采用iPhone 14 Pro Max运行IKEA与Virtlo两款MAR应用，结合IPQ、UEQ‑S、SAM、SAQ等标准问卷，使用Wilcoxon符号秩检验与相关分析技术。

**📊 数据集**

使用20名年龄20-45岁的参与者数据（11男9女），分别体验两款交互级别不同的MAR应用。

**📈 对比分析**

通过对比两款应用在UX与SA维度上的统计显著差异，发现Virtlo在UX中被评为更复杂但更吸引人，社会可接受性在高交互性下下降；统计结果显示差异显著。

**⚠️ 局限性**

样本规模小且同质化，缺乏文化与领域多样性；实验仅覆盖两款应用，限制了结论的普适性与推广性。

---

## 572. BrainSTR: Spatio-Temporal Contrastive Learning for Interpretable Dynamic Brain Network Modeling

**arXiv ID:** 2603.09825 | [PDF](https://arxiv.org/pdf/2603.09825v1)

**作者:** Guiliang Guo `[一作]` (Northeastern University), Osmar R. Zaiane `[通讯]` (University of Alberta)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5027917989)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出BrainSTR框架，利用自适应相位分区和增量图结构生成实现时空对比学习，从动态功能连接中提取可解释的诊断相关特征；

**💡 创新点**

创新点在于：①自适应相位分区（APP）捕获一致脑状态边界；②增量图结构生成器通过递归更新与正则化实现稀疏、可二值化的诊断相关连通性；③以原始图嵌入为参考的监督对比学习，显著提升诊断相关时空表示的区分度；

**🔧 技术方法**

技术包括：时间自编码器（TCN+解码器）进行相位分割；增量图结构学习与直通估计器；结构感知编码器（边到边、边到节点卷积）；注意力加权阶段聚合；参考调整的InfoNCE对比损失；多项结构正则化（二值化、平滑、稀疏）；

**📊 数据集**

数据集：私有 rs‑fMRI 组（246 对照、151 MDD、126 BD），以及公开 ABIDE NYU ASD 组（74 ASD、98 对照）；

**📈 对比分析**

与传统 SVM/RF、静态图神经网络（GroupINN、BrainGNN 等）、动态图模型（MDGL、BrainDGT、MCDGLN 等）对比；在 MDD、BD、ASD 三种疾病上，BrainSTR 分别取得 77.2%/77.8%、78.2%/79.6%、72.4%/73.0% 的 ACC/AUC，均高于所有基线并实现显著提升（MDD +2.9%/6.5%，BD +4.5%/4.9%，ASD +3.0%/3.3%）；

**⚠️ 局限性**

局限性包括：①对滑动窗口长度和阈值的敏感性需要进一步自动化；②增量结构更新机制过于简化，可能忽略更复杂的拓扑变化；③跨中心泛化仍需更大规模、多中心验证；④解释性主要聚焦于 DMN 相关网络，可能低估其他脑区的贡献。

---

## 573. A Hybrid Quantum-Classical Framework for Financial Volatility Forecasting Based on Quantum Circuit Born Machines

**arXiv ID:** 2603.09789 | [PDF](https://arxiv.org/pdf/2603.09789v1)

**作者:** Yixiong Chen `[一作]` `[通讯]`, Yixiong Chen

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

结合 LSTM 与 QCBM 的混合模型，对高频金融波动率进行预测。

**💡 创新点**

提出用 QCBM 作为可学习的先验生成器，并通过交替训练解耦量子与经典模块。

**🔧 技术方法**

采用 LSTM、Quantum Circuit Born Machine、COBYLA 优化、分层量子电路等技术。

**📊 数据集**

使用 2025‑2026 年 5 分钟频率的上证指数与 CSI300 指数高频数据。

**📈 对比分析**

与纯经典 LSTM 对比，LSTM‑QCBM 在 MSE、RMSE、QLIKE 上分别提升 42%‑67%，性能显著优越。

**⚠️ 局限性**

受限于 NISQ 噪声、QCBM 采样成本、量子电路规模与数据编码的理论与实证限制。

---

## 574. Surgical Repair of Collapsed Attention Heads in ALiBi Transformers

**arXiv ID:** 2603.09616 | [PDF](https://arxiv.org/pdf/2603.09616v1)

**作者:** Palmer Schallon `[一作]` `[通讯]`, Palmer Schallon

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对BLOOM系列模型中因ALiBi位置编码导致的注意力头崩塌问题进行诊断，并提出了一种“外科重启”技术，针对崩塌的Q/K/V权重进行随机重初始化、输出投影置零并冻结其余参数，成功在单张RTX 5070 Ti显卡上恢复了98.7%的功能头。

**💡 创新点**

创新点在于首次证明注意力头崩塌并非冗余，而是可通过外科手术恢复的可用容量；并揭示外科操作能触发全模型注意力重分布，且该重分布的质量由训练语料结构决定，而非模型架构本身。

**🔧 技术方法**

技术包括：基于BOS质量与熵的头诊断工具、Xavier重初始化+输出投影归零、梯度掩码冻结非手术参数、双阶段外科训练、以及对照实验（C4 vs curated 语料）来评估语料影响。

**📊 数据集**

使用的训练语料为两种：C4验证集（约541K纯网文本）和一份精心整理的语料（含HTML、代码和哲学文本）；两者均用于外科训练以对比语料效应。

**📈 对比分析**

比较方法为：在原始BLOOM‑1b7上测量头健康度、训练/评估困惑度、注意力分布漂移；外科后模型在相同任务上训练困惑度下降至15.1、评估困惑度上升至28.8，表明对训练分布的泛化提升而对外部分布的退化；C4外科模型在C4验证集上困惑度从32.4降至29.3，验证了恢复效果。

**⚠️ 局限性**

主要局限包括：外科技术仅在BLOOM‑1b7上验证，未在更大规模模型上测试；恢复后模型对语料的印记明显，导致生成行为受限；恢复效果在短期内显著，长时间训练易出现过拟合；诊断阈值依赖于BLOOM的双峰分布，对其他架构需重新调参。

---

## 575. Test-Driven AI Agent Definition (TDAD): Compiling Tool-Using Agents from Behavioral Specifications

**arXiv ID:** 2603.08806 | [PDF](https://arxiv.org/pdf/2603.08806v1)

**作者:** Tzafrir Rehan `[一作]` `[通讯]` (Fiverr Labs), Tzafrir Rehan (Fiverr Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Test-Driven AI Agent Definition（TDAD）方法，将 agent 开发视为编译过程：先把自然语言规范转化为可执行的测试，然后通过 PromptSmith 迭代修改 Prompt 直至所有可见测试通过，形成最终的 agent 构件；同时引入隐藏测试、语义变异测试和规范演化机制防止规范作弊；在 SpecSuite‑Core 4 个深度规范上实验，验证方法有效性。

**💡 创新点**

创新点在于：① 将测试驱动开发理念迁移至 LLM Agent，形成可迭代编译流水线；② 引入隐藏测试和语义变异测试作为防作弊手段；③ 通过规范演化场景评估回归安全；④ 通过多角色（TestSmith、PromptSmith、MutationSmith）分离职责，保证抗游戏能力。

**🔧 技术方法**

技术上使用 Claude Sonnet 4.5 通过 Claude Code 生成测试、Prompt 以及变异 Prompt；利用 Docker 隔离、pytest 并行执行、可确定性 fixture；实现可视/隐藏测试拆分、变异激活探测；使用 JSON schema 驱动工具调用和响应验证。

**📊 数据集**

数据集为 SpecSuite‑Core benchmark，包含四个规格（SupportOps、DataInsights、IncidentRunbook、ExpenseGuard），每个规格 10‑14 个决策节点，分别提供可见/隐藏测试、变异意图和 v1→v2 演化版本。

**📈 对比分析**

通过 24 次独立实验（每个规格两版本各 3 次）比较：V1 编译成功率 92%、隐藏测试通过率 97.3%、变异得分 86‑100%；V2 成功率 58%、隐藏通过率 78%，但变异得分 100%；回归安全率 97.2%；整体成本约 2‑3 美元/版本，迭代 2‑4 次即可收敛。

**⚠️ 局限性**

局限包括：规范无法完全表达诸如“富有同理心”等属性；变异测试排除非激活变异可能低估测试质量；随机性导致 HPR 与预算波动；对抗性输入生成受安全训练限制；实验仅使用 Claude，未验证跨模型泛化；对更大决策树、快速原型的可扩展性未测。

---

## 576. Exploiting Label-Aware Channel Scoring for Adaptive Channel Pruning in Split Learning

**arXiv ID:** 2603.09792 | [PDF](https://arxiv.org/pdf/2603.09792v1)

**作者:** Jialei Tan `[一作]` (Fuzhou University), Wei Ni `[通讯]` (CSIRO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种自适应通道裁剪辅助的分割学习（ACP‑SL）框架，用来降低在分割学习中传输中间特征（smash 数据）时的通信开销，同时保持甚至提升模型精度。

**💡 创新点**

创新点在于①设计了标签感知通道重要性评分（LCIS）模块，利用同标签内相似度与跨标签相似度计算每个通道的重要性；②基于 LCIS 输出的动态重要性分数，构造自适应通道裁剪（ACP）模块，按通道重要性自适应调整裁剪比例，避免对重要通道过度压缩。

**🔧 技术方法**

核心技术包括分割学习（Split Learning）、通道重要性评分（计算 Frobenius 内积的平均相似度）、历史重要性加权融合、动态裁剪比例计算、以及梯度裁剪同步。

**📊 数据集**

使用了 CIFAR‑10 与 Fashion‑MNIST 两个公开数据集，在 IID 与 non‑IID（Dirichlet β=0.5）两种数据分布下进行实验，模型采用 ResNet‑18 的前四层作为客户端模型。

**📈 对比分析**

与标准 SL、随机 Top‑k SL 与量化 SL 等基线进行对比，实验表明 ACP‑SL 在两组数据集与两种分布下均取得更高测试精度（如 CIFAR‑10 上约 75.88%/71.43%，比 Quantization‑SL 高 5.11%/3.72%），并且达到目标精度所需的训练轮数更少（例如非 IID CIFAR‑10 达到 65% 仅需约 46 轮，少 12 轮）。

**⚠️ 局限性**

局限性包括：①裁剪比例的参数设置（P_min、P_base、P_max）需经验调参；②在极端非 IID 或非常稀疏数据场景下，通道重要性评估可能不稳定；③实验仅在单 GPU 通过轮流激活模拟多客户端，实际多客户端环境下同步与通信延迟的影响未深入评估。

---

## 577. Comparative Analysis of Patch Attack on VLM-Based Autonomous Driving Architectures

**arXiv ID:** 2603.08897 | [PDF](https://arxiv.org/pdf/2603.08897v1)

**作者:** David Fernandez `[一作]` (Clemson University), Mert D. Pesé `[通讯]` (Clemson University)

**通讯引用:** 240 | [OpenAlex ID](https://openalex.org/A5085340429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文构建了一个可对比的评估框架，系统性地在CARLA仿真中测试了三种VLM架构的物理攻击鲁棒性。

**💡 创新点**

创新点在于引入语义同化层实现不同VLM输出的统一对齐，配合黑盒NES优化和EoT，使攻击可跨架构公平对比。

**🔧 技术方法**

采用黑盒自然进化策略(NES)、语义相似度损失、期望变换(EoT)以及CLIP文本编码器进行攻击生成与评估。

**📊 数据集**

使用CARLA Town04仿真场景，包含公交站广告牌和高速公路广告牌两类攻击实例，生成的图像序列作为输入。

**📈 对比分析**

通过攻击成功率、持续帧数、检测率下降、BLEU-4/语义相似度等多维指标进行比较，结果显示所有架构均出现70-80% ASR，并且攻击在10-25m关键距离持续多帧失败。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，未考虑实际传感器噪声与光照变化，且只评估了三种架构，缺乏对更广泛VLM设计的泛化研究。

---

## 578. MiniAppBench: Evaluating the Shift from Text to Interactive HTML Responses in LLM-Powered Assistants

**arXiv ID:** 2603.09652 | [PDF](https://arxiv.org/pdf/2603.09652v1)

**作者:** Zuhao Zhang `[一作]` (Inclusion AI), Shuai Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 25100 | [OpenAlex ID](https://openalex.org/A5100424064)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MiniApps基准，用于评估LLM在生成遵循真实世界原理且可交互的HTML应用；同时设计了Agentic Evaluation Framework (AEF)通过浏览器自动化进行意图、静态和动态三维评估。

**💡 创新点**

①首个专门针对原理驱动交互式应用生成的基准；②结合了可执行的交互式评估框架，解决传统静态或模板匹配评估的局限；③在评估中引入开放式参考与双盲评判。

**🔧 技术方法**

使用Playwright进行浏览器自动化与交互；LLM生成代码与评估器；自定义评价参考生成脚本；多模型推理与标准化代码生成框架。

**📊 数据集**

从生产平台收集的数千万条真实用户查询中筛选、扩增并人工审核，最终形成500个任务，覆盖6大领域，分为Easy/Medium/Hard。

**📈 对比分析**

与多款公开与闭源LLM（如GPT‑5、Claude‑Opus、Gemini‑3、GLM‑4.7等）进行对比；平均通过率仅17%，最高GPT‑5.2达45%；闭源模型整体优于开源模型，难度越高模型表现越差。

**⚠️ 局限性**

仍面临高质量交互式代码生成难度大，模型在复杂逻辑与细粒度原理实现上表现不足；评估框架对视觉任务易偏宽容，需进一步提升判定精度；对多模型推理成本与时间仍有挑战。

---

## 579. OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models

**arXiv ID:** 2603.09326 | [PDF](https://arxiv.org/pdf/2603.09326v1)

**作者:** Tengjin Weng `[一作]` (Shenzhen University), Zhong Ming `[通讯]` (Shenzhen University)

**通讯引用:** 13114 | [OpenAlex ID](https://openalex.org/A5100633973)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了可控的 OddGridBench 基准和 OddGrid-GRPO 强化学习框架，用于评估和提升多模态大型语言模型在低层视觉差异检测上的敏感度。

**💡 创新点**

创新点在于（1）构建可参数化、可扩展的网格图像基准，精准控制颜色、尺寸、旋转和位移等低级视觉属性；（2）在 GRPO 基础上引入课程学习与距离感知奖励，实现对细粒度视觉差异的连续反馈与自适应难度调节。

**🔧 技术方法**

主要技术包括强化学习（GRPO）、课程学习（Curriculum Learning）与距离感知奖励（Distance‑Aware Reward）以及基于 SVG 图标的合成图像生成。

**📊 数据集**

使用了自制的 OddGridBench 数据集，包含 1400 张测试图（+400 验证 + 30k 训练），每张图由 5-9 行列的网格构成，单一或多属性差异的控制可变范围。

**📈 对比分析**

通过在 19 款 MLLMs（包括开源和专有模型）上进行实验，所有模型均明显落后于人类；应用 OddGrid‑GRPO 后整体准确率提升至 82.6%，相较基线提高约 65%，并显著优于传统 GRPO/GSPO 方法。

**⚠️ 局限性**

局限性在于模型仍与人类差距较大，主要依赖合成图标而非真实场景，且对极细微属性变化的鲁棒性有限，未来需要扩展到更丰富的视觉语义与真实图像。

---

## 580. Automated Tensor-Relational Decomposition for Large-Scale Sparse Tensor Computation

**arXiv ID:** 2603.08957 | [PDF](https://arxiv.org/pdf/2603.08957v1)

**作者:** Yuxin Tang `[一作]` (Rice University), Chris Jermaine `[通讯]` (Rice University)

**通讯引用:** 1484 | [OpenAlex ID](https://openalex.org/A5002518742)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出一种将张量计算自动转换为张量-关系计算的框架，利用上/下标分离的Einstein表示法（Upper‑Case‑Lower‑Case）和SparseEinSum算法实现可扩展的稀疏张量运算；

**💡 创新点**

创新点在于将张量与关系两者的优势结合，提供一种自动的、基于成本模型的张量分解策略，既能充分利用稀疏性，又能调用高性能的稠密核；

**🔧 技术方法**

核心技术包括：Einstein求和符号的上/下标重写、成本估算模型、动态规划优化、TACO稠密核生成、以及基于PlinyCompute的分布式关系计算；

**📊 数据集**

使用的基准数据集包括图神经网络的Cora、CiteSeer、Amazon、Planetoid等；大规模图数据如ogbn‑arxiv、ogbn‑products、ogbn‑papers100M、friendster；以及量子电路模拟基准seca_n11、multiplier_n13；

**📈 对比分析**

实验与传统张量框架（DGL/PyTorch、AliGraph）以及纯关系实现（SQLite/Hyper/PostgreSQL）对比，SparseEinSum在分布式环境下可实现5–10×的速度提升，单机稠密实现相比纯关系实现提升1–2×，并避免了OOM问题；

**⚠️ 局限性**

主要限制包括：动态规划在结果被多次使用时只能得到局部最优；成本模型对稀疏度估计误差敏感；对内存有显式限制；对高阶表达式的支持有限；重排（repartition）开销在某些场景下仍显著。

---

## 581. HMR-1: Hierarchical Massage Robot with Vision-Language-Model for Embodied Healthcare

**arXiv ID:** 2603.08817 | [PDF](https://arxiv.org/pdf/2603.08817v1)

**作者:** Rongtao Xu `[一作]` (Spatiotemporal AI), Xiaopeng Zhang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种分层的具身按摩框架，结合多模态大型语言模型进行穴位定位并通过深度感知与轨迹规划实现机器人精准按摩。

**💡 创新点**

创新点包括：①首个针对穴位按摩的多模态数据集 MedMassage-12K；②将高层语义理解与低层运动控制紧耦合的分层架构；③在 Qwen‑VL 上进行专门微调并验证其在实际机器人上的可行性。

**🔧 技术方法**

所用技术包括多模态大型语言模型（Qwen‑VL、GPT‑4o）、跨模态注意力适配器、OpenCLIP 视觉编码器、深度相机 + RANSAC 取平面、逆运动学、轨迹规划与多项式拟合等。

**📊 数据集**

使用的数据集为 MedMassage‑12K：12,190 张图像、174,177 条 QA 对，覆盖 60 个穴位，包含多种光照与背景条件，并通过几何变换进行扩增。

**📈 对比分析**

通过与 Qwen‑VL‑Max、GPT‑4o 的基准比较，本文模型在 IoU=0.3、0.5、0.75 下的定位成功率分别为 87.60%、81.42%、67.77%，显著优于基线（约 0%）。实验中数据量与增强均对性能有显著提升，最终在真实机器人（Franka Panda）上实现了安全、准确的穴位按摩。

**⚠️ 局限性**

局限性在于：①数据集主要基于仿真人体模型，缺乏真实人类多样性；②模型对极端光照、遮挡等场景的鲁棒性仍有限；③高层模型对大规模训练数据的依赖较高，迁移到其他具身任务需要更多样本；④实际操作中对深度误差和机器人硬件限制的适配仍需进一步研究。

---

## 582. RSH-SpMM: A Row-Structured Hybrid Kernel for Sparse Matrix-Matrix Multiplication on GPUs

**arXiv ID:** 2603.08734 | [PDF](https://arxiv.org/pdf/2603.08734v1)

**作者:** Aiying Li `[一作]` (University of Science and Technology of China), Guangzhong Sun `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6235 | [OpenAlex ID](https://openalex.org/A5100932403)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种行结构化的混合 GPU SpMM 内核 RSH-SpMM，结合自适应行划分、RS‑Tile 存储格式和局部性感知重排序，以实现高效利用 Tensor Core 与 CUDA 核的协同计算。

**💡 创新点**

创新点包括：① 对每行进行细粒度划分，将结构良好的长行分配给 Tensor Core，短行或结构不连贯行转到轻量级 CUDA 路径；② 采用 RS‑Tile 通过位图压缩与列索引重映射产生密集的 8×8 MMA 块；③ 通过自适应负载平衡（分割极长行、双缓冲流水线）保持 Tensor Core 的连续占用；④ 用加权 Jaccard + MST + 2‑opt 的局部性重排序显著提升窗口密度。

**🔧 技术方法**

技术细节包括 RS‑Tile 存储格式、行级自适应阈值划分、双缓冲的 Tensor‑Core 计算流水线、最小开销 CUDA 余量路径、动态负载平衡策略以及基于 k‑NN 图的局部性重排序。

**📊 数据集**

实验使用 9 个真实图数据集（com‑amazon、ddi、DD、amazon0505、amazon0601、Yeast、OVCAR‑8H、YeastH、web‑BerkStan）以及 512 条 SuiteSparse 公开矩阵，覆盖从稠密到极度不规则的多种稀疏结构。

**📈 对比分析**

与 cuSPARSE、Sputnik、RoDe（CUDA 核），TC‑GNN、DTC‑SpMM、Acc‑SpMM（Tensor‑Core），HC‑SpMM、MP‑SpMM（混合）等基线进行对比。RSH‑SpMM 在 RTX 4090 上平均 2.35×、RTX 3090 上 2.86× 的加速；在所有基线中最高可达 6.13×，在 SuiteSparse 上 80%+ 矩阵提升范围为 1.24×–8.2×，实现了最稳健、最一致的性能提升。

**⚠️ 局限性**

局限性：① 行划分阈值的选择需要依据矩阵规模与稀疏度进行调优，可能对极稠密或结构规则的矩阵产生次优；② 重排序与 RS‑Tile 的预处理开销对实时工作负载影响仍需进一步优化；③ 对 GPU 版本的迁移性（如低算力或不同架构）尚未充分验证；④ 仍不支持 2:4 等结构化稀疏格式的加速；⑤ 在极高密度矩阵下，Tensor‑Core 利用率不如预期。

---

## 583. The Virtuous Cycle: AI-Powered Vector Search and Vector Search-Augmented AI

**arXiv ID:** 2603.09347 | [PDF](https://arxiv.org/pdf/2603.09347v1)

**作者:** Jiuqi Wei `[一作]` (Oceanbase), Chuanhui Yang `[通讯]` (Oceanbase)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文以教程形式全面梳理并阐述了人工智能与向量检索的相互促进关系，提出并系统化了AI驱动向量检索（AI4VS）与向量检索增强人工智能（VS4AI）的技术框架，并讨论了端到端共优化、RAG（检索增强生成）演进、以及未来研究挑战。

**💡 创新点**

创新点在于：①提出“良性循环”概念，将AI与向量检索的双向促进关系可视化为AI4VS、VS4AI及其闭环优化三大模块；②对Naive、Advanced、Modular三代RAG框架进行统一归纳与对比；③强调端到端共训练与可微检索、动态工作流等前沿方向，并对自适应索引、RAG缓存、Agentic RAG等关键挑战给出展望。

**🔧 技术方法**

使用的技术包括：深度学习驱动的索引与哈希学习、向量量化、图路由与自适应裁剪、自动化参数调优（贝叶斯优化、Meta‑学习等）、稀疏+稠密混合检索、多向量表示、可微检索、联合检索‑生成训练、模块化工作流（Router、Memory、Predictor、Search）以及相关的评估与缓存机制。

**📊 数据集**

作为教程，本文未直接使用任何数据集；其内容主要基于已有公开研究和工业案例，引用的典型检索与生成基准包括MS‑MARCO、C4、Wikipedia等公共语料库与向量检索基准（如SIFT、FAISS 等）。

**📈 对比分析**

本文未进行实验对比，而是综述并引用文献中的对比结果；对比方式主要为：不同索引结构、检索策略与RAG版本在召回率、延迟、生成准确性等指标上的表现，指出各类技术的优势与局限。

**⚠️ 局限性**

局限性：①未提供新的实验验证与量化结果；②内容偏向高层次综述，缺乏细粒度的算法实现细节与性能评估；③未涉及数据集和基准的统一评测，读者需自行查阅原始文献；④在快速发展领域，部分前沿技术（如Agentic RAG）仍处于探索阶段，实际应用效果待验证。

---

## 584. When Learning Rates Go Wrong: Early Structural Signals in PPO Actor-Critic

**arXiv ID:** 2603.09950 | [PDF](https://arxiv.org/pdf/2603.09950v1)

**作者:** Alberto Fernández-Hernández `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6838 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究学习率对PPO actor‑critic 网络内部结构的影响，并提出使用 OUI（Overfitting‑Underfitting Indicator）作为早期筛选指标；

**💡 创新点**

创新点在于将 OUI 改为批量形式并给出与学习率、激活翻转及 OUI 动态的理论关联，证明 OUI 在训练10%时即可区分不同学习率 regime，并在早期筛选中优于传统指标；

**🔧 技术方法**

主要技术包括 Proximal Policy Optimization（PPO）、批量化 OUI 计算、梯度步长与激活翻转的理论分析、以及早期筛选规则的实验验证；

**📊 数据集**

使用的实验数据集有 CartPole‑v1、LunarLander‑v3 与 MiniGrid‑Empty‑8x8‑v0 三个离散控制环境；

**📈 对比分析**

对比方法包括早期回报、KL 散度、裁剪率、divergence（KL+裁剪）和激活翻转率等，在匹配 recall 的前提下，OUI 单独或与回报结合的筛选规则在精确率上均优于其他方法；

**⚠️ 局限性**

主要限制在于实验仅覆盖 PPO 与离散控制任务，未验证连续控制或其他 actor‑critic 变体，probe batch 固定可能忽略动态结构变化，且理论分析未覆盖 PPO 的 clip 等特定机制。

---

## 585. The Richest Paradigm You're Not Using: Commercial Videogames at the Intersection of Human-Computer Interaction and Cognitive Science

**arXiv ID:** 2603.09753 | [PDF](https://arxiv.org/pdf/2603.09753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 586. DCAU-Net: Differential Cross Attention and Channel-Spatial Feature Fusion for Medical Image Segmentation

**arXiv ID:** 2603.09530 | [PDF](https://arxiv.org/pdf/2603.09530v1)

**作者:** Yanxin Li `[一作]` (Chongqing University of Technology), Libin Lan `[通讯]` (Chongqing University of Technology)

**通讯引用:** 187 | [OpenAlex ID](https://openalex.org/A5091564313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 DCAU-Net，结合 Differential Cross Attention 与 Channel‑Spatial Feature Fusion 的医学图像分割网络。

**💡 创新点**

创新点在于使用窗口级摘要 token 的 Differential Cross Attention 减少 O(N²) 计算，并通过 CSFF 通过通道和空间注意力自适应融合跳跃连接。

**🔧 技术方法**

采用 Transformer 相关自注意力（差分注意力）与深度可分离卷积、MLP、RMSNorm、窗口池化、通道/空间注意力等技术。

**📊 数据集**

在 Synapse 多器官 CT 数据集和 ACDC 心脏 MRI 数据集上进行实验。

**📈 对比分析**

与 U‑Net、TransUNet、Swin‑Unet、MISSFormer 等方法对比，DCAU‑Net 在 Synapse 上 83.29% DSC、4.67G FLOPs 领先；在 ACDC 上 92.11% DSC，也取得最佳表现。

**⚠️ 局限性**

缺点包括对 ImageNet 预训练权重的依赖、窗口尺寸固定可能限制多尺度特征，且未在三维体积数据上验证。

---

## 587. Generative Drifting is Secretly Score Matching: a Spectral and Variational Perspective

**arXiv ID:** 2603.09936 | [PDF](https://arxiv.org/pdf/2603.09936v1)

**作者:** Erkan Turan `[一作]` (Ecole Polytechnique), Maks Ovsjanikov `[通讯]` (Ecole Polytechnique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过证明在高斯核下漂移算子等价于平滑分布的得分差，系统阐述了漂移模型的理论基础；

**💡 创新点**

创新点在于提出漂移算子与得分匹配的等价性、利用傅里叶稳定性分析揭示高斯核的高频瓶颈并给出指数衰减的宽度退火调度、从Jordan‑Kinderlehrer‑Otto梯度流框架推导停梯度算子的必要性，并基于此提出Sinkhorn散度漂移等新漂移算子；

**🔧 技术方法**

采用得分匹配、傅里叶分析、McKean‑Vlasov动力学、Wasserstein梯度流（JKO方案）、指数宽度退火以及Sinkhorn距离的梯度推导等技术；

**📊 数据集**

主要在合成的1维/2维多模混合高斯分布上进行实验，文中未给出大规模图像数据集的实验结果；

**📈 对比分析**

通过与原始漂移模型在同一合成任务上的对比，发现指数退火能显著加速收敛并缓解高频抑制；实验表明停梯度是保持训练稳定的关键；

**⚠️ 局限性**

局限性包括：理论分析基于线性化与局部均匀假设，难以覆盖早期非平衡阶段；实验仅限于低维合成数据，尚未在高维图像数据上验证退火策略和新漂移算子的效果。

---

## 588. GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware Vision-Language-Action Models

**arXiv ID:** 2603.09079 | [PDF](https://arxiv.org/pdf/2603.09079v1)

**作者:** Md Selim Sarowar `[一作]` (Yeungnam University), Sungho Kim `[通讯]` (Yeungnam University)

**通讯引用:** 11927 | [OpenAlex ID](https://openalex.org/A5100619135)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在视觉语言动作学习中，提出了 GST‑VLA 系列，将 RGB+深度先转换为 128 个 3D 高斯令牌，再通过 Depth‑Aware Chain‑of‑Thought（DA‑CoT）生成中间 3D 认知，然后映射给 7‑DoF 运动专家；

**💡 创新点**

核心创新在于（1）Gaussian Spatial Tokenizer（GST），用密集深度和语义特征生成带方向、置信度的 3D 高斯令牌；（2）DA‑CoT，监督四步 3D 认知（物体定位、抓取接触、空间关系、SE(3) 轨迹），显式可视化并通过梯度回传优化 GST；

**🔧 技术方法**

采用冻结的语义编码器与深度估计器；Mlp 估计 μ、σ、α，3D Fourier 位置编码，空间注意力池化；VLM 通过 LoRA、交叉注意力注入高斯令牌；动作专家为 300M 参数流匹配网络；整体损失为 ℒ_flow+ℒ_CoT+ℒ_depth；

**📊 数据集**

训练集：ScanNet、Hypersim、ARKitScenes 用于深度预训练；演示数据拆分用于流匹配；评估集：LIBERO、LIBERO‑Pro、SimplerEnv、BridgeData V2；

**📈 对比分析**

与 SpatialVLA、CogACT、OpenVLA、π_0VLA 等基线对比，在 LIBERO 取得 96.4%（+3.8%）平均成功率，在 SimplerEnv 80.2%（+5.4%）进度，在整体 83.1%（+6.3%）平均表现；显著提升精细插入、薄物体抓取等高精度任务；

**⚠️ 局限性**

主要局限：对高反射/遮挡物体的深度估计不稳健，导致高斯令牌置信度低；依赖冻结的深度模型，若深度错误会连带影响认知；推理速度相较基线慢约 0.7 Hz，适配性仍有提升空间。

---

## 589. SCALAR: Learning and Composing Skills through LLM Guided Symbolic Planning and Deep RL Grounding

**arXiv ID:** 2603.09036 | [PDF](https://arxiv.org/pdf/2603.09036v1)

**作者:** Renos Zabounidis `[一作]` (Carnegie Mellon University), Katia Sycara `[通讯]` (Carnegie Mellon University)

**通讯引用:** 19827 | [OpenAlex ID](https://openalex.org/A5087505541)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出SCALAR框架，利用LLM规划与RL通过可学习技能的双向反馈循环，实现长时序稀疏奖励环境中的技能学习与组合；

**💡 创新点**

创新点包括：①LLM生成符号化操作符并通过RL训练后的轨迹分析反馈修正预设，形成闭环；②前沿检查点保存前置技能完成状态，提高采样效率；③在线自适应修正技能规范，增强鲁棒性；

**🔧 技术方法**

技术手段包括：大型语言模型（LLM）生成技能与奖励，符号规划（STRIPS）生成计划，强化学习（PPO/Transformer-XL等）训练选项，轨迹分析与前沿检查点等；

**📊 数据集**

使用数据集：Craftax（Classic 版本与完整 9 层版本）作为稀疏奖励的长时序游戏环境；

**📈 对比分析**

与多种基线（PPO-FC、PPO-RND、PPO-RNN、PPO-TRXL、PQN、PQN-RNN）对比，SCALAR 在 Craftax-Classic 的钻石收集成功率 88.2%（比最佳基线 46.9% 提升 1.9×），在完整 Craftax 中进入 Gnomish Mines 成功率 9.1%（此前为 0%），其余任务亦显著优于基线；

**⚠️ 局限性**

局限性：需要预先定义的符号化状态与词汇；技能执行顺序固定，无法机会性混合子任务；前沿检查点依赖可序列化环境状态，对状态化模型或真实世界环境受限；LLM 先验可能带来偏差，需轨迹分析补偿。

---

## 590. Proxy-Guided Measurement Calibration

**arXiv ID:** 2603.09288 | [PDF](https://arxiv.org/pdf/2603.09288v1)

**作者:** Saketh Vishnubhatla `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**通讯引用:** 48197 | [OpenAlex ID](https://openalex.org/A5100338946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于代理变量的测量校准框架，通过因果图将真实结果与系统误差分离，利用两阶段变分自编码器估计并纠正测量误差；

**💡 创新点**

创新点在于将代理排除假设与因果模型结合，使用两阶段VAE同时学习内容潜变量和偏差潜变量，并通过匹配估计偏差大小，从而在无验证数据的情况下实现误差校正；

**🔧 技术方法**

核心技术包括：两阶段变分自编码器（VAE）对代理与观测数据进行潜变量学习；因果图与可识别性分析；加性偏差模型与最近邻匹配估计；以及生成模型的ELBO优化；

**📊 数据集**

实验使用了三类数据：① 合成数据（完全可控的生成过程）；② 半合成数据（Oregon Health Insurance Experiment 与 JOBS 随机对照试验的代理与环境变量）；③ 实际灾害损失数据库 SHELDUS（县级财产损失与遥感代理）；

**📈 对比分析**

与代理仅、环境仅、TEDVAE 等基线进行比较。合成实验中误差参数 α 估计精度高、误差随样本量改善；半合成实验中本方法显著优于基线（尤其在 α=5、10 情况），而 TEDVAE 常低估偏差；真实案例中通过匹配得到县级 CATE，展示地理异质性，整体性能稳健；

**⚠️ 局限性**

局限性包括：1) 假设误差为单一加性 α 且满足单调性，限制了模型适用范围；2) 代理排除假设在实际场景可能不完全成立；3) 潜变量仅在尺度/旋转上可识别，导致参数解释受限；4) 仅估计条件平均效应，未得到个体级别偏差；5) 对高度非线性或高维数据的适应性仍需进一步研究。

---

## 591. Randomized Distributed Function Computation (RDFC): Ultra-Efficient Semantic Communication Applications to Privacy

**arXiv ID:** 2603.09577 | [PDF](https://arxiv.org/pdf/2603.09577v1)

**作者:** Onur Günlü `[一作]` `[通讯]` (Technische Universitaet Dortmund), Onur Günlü (Technische Universitaet Dortmund)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了随机分布式函数计算（RDFC）框架，利用强协调实现隐私保护的语义通信，显著降低了通信负载。

**💡 创新点**

创新点在于将本地差分隐私与对称随机响应机制融入RDFC，并给出了Wyner公共信息的下界与互信息上界，揭示了共享随机性对语义通信效率的巨大提升。

**🔧 技术方法**

使用的信息理论技术包括强协调、Wyner公共信息、互信息、Gaussian机制的LDP证明、软覆盖引理与有限块长误差分析。

**📊 数据集**

未使用公开数据集；本研究主要通过理论推导与数值模拟验证，涉及截断高斯与BSC混合模型的合成分布。

**📈 对比分析**

与无共享随机性和传统无损压缩方法比较，RDFC在共享随机性下可将所需率降低至互信息水平，且无共享随机性时仍比无损压缩低至十几倍；实验显示在LDP约束下通信率可提升高达两位数。

**⚠️ 局限性**

主要局限在于实现层面：目前仅给出存在性与非构造性证明，缺乏具体可实现的编码方案；同时，有限块长分析虽表明误差指数衰减，但实际性能需进一步实验验证。

---

## 592. Unit Interval Selection in Random Order Streams

**arXiv ID:** 2603.08937 | [PDF](https://arxiv.org/pdf/2603.08937v1)

**作者:** Cezar-Mihail Alexandru `[一作]` (Independent Researcher), Kheeran K. Naidu `[通讯]` (Qworky Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在一次性随机顺序流模型中，提出了一种求解单位长度区间最大无交集子集（Interval Selection）问题的算法，期望逼近率为 0.7401，空间复杂度为 O(|OPT|)。

**💡 创新点**

创新点在于：①通过递归拆分窗口并维护左右端点的贪心选择，构造了一个在随机顺序下比已知的 2/3 阈值更高的期望逼近算法；②使用通信复杂度下的 _t 问题归约，证明在相同空间限制下无法获得 8/9 以上的期望逼近率，且在概率 2/3+ε 下逼近 2/3 也需要线性空间。

**🔧 技术方法**

主要技术包括：递归窗口划分与最优左/右端点维护；对独立区间集合进行递归期望分析得到递推关系 out(x)；利用窗口滑动（shifting window）技术将有限域算法推广到无界域；通信复杂度下的 t‑问题归约与公共随机性构造。

**📊 数据集**

本研究为理论分析性质，未使用具体实验数据集；所有结果均通过严谨的数学证明与递推计算得到。

**📈 对比分析**

相较于先前针对任意顺序流的 2/3 逼近算法，本算法在随机顺序下实现更高的期望逼近率 0.7401，空间保持线性与 OPT 成正比；下界表明 8/9 以上的期望逼近率与高概率 2/3 以上逼近在同样空间下不可能实现，验证了所给逼近率的实用性。

**⚠️ 局限性**

局限性包括：①仅适用于单位长度区间；②仅在随机顺序流下工作，任意顺序下仍受 2/3 阈值限制；③期望逼近率仍与 0.8 之间存在差距，是否能进一步改进或证明更强下界尚未解决；④算法实现复杂度高（窗口大小 Δ=5000 导致巨大常数 4^Δ·Δ^Δ），实际应用需进一步优化。

---

## 593. $M^2$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

**arXiv ID:** 2603.09737 | [PDF](https://arxiv.org/pdf/2603.09737v1)

**作者:** Kaixin Lin `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**通讯引用:** 5284 | [OpenAlex ID](https://openalex.org/A5027010844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在多摄像头视角缺失时仍能进行语义占据预测的框架M^2-Occ。

**💡 创新点**

创新点在于：①多视角遮挡重建（MMR）模块利用相邻摄像头的重叠信息在特征空间中重建缺失视角；②特征记忆模块（FMM）引入全局类别原型（单原型与多原型），在体素空间进行语义正则化，提升在视角缺失下的鲁棒性。

**🔧 技术方法**

使用Transformer解码器进行特征重建，配合位置编码；利用记忆池学习类别原型；整体框架基于ResNet-101+FPN编码器、空间跨视角注意力转换到3D体素，再由3D占据头输出。

**📊 数据集**

主要使用nuScenes数据集，并在其基于SurroundOcc的占据标注上进行训练与评估。

**📈 对比分析**

与SurroundOcc基线在单视角缺失和多视角随机丢失的情况下进行对比；在后视缺失时IoU提升4.93%，在最多5个摄像头缺失时提升5.01%；在完整视角下保持与基线相近的性能。

**⚠️ 局限性**

局限性是对小物体和细节的重建不够理想，缺失视角下细粒度语义与高频信息仍易被模糊或误判。

---

## 594. On the Width Scaling of Neural Optimizers Under Matrix Operator Norms I: Row/Column Normalization and Hyperparameter Transfer

**arXiv ID:** 2603.09952 | [PDF](https://arxiv.org/pdf/2603.09952v1)

**作者:** Ruihan Xu `[一作]` (University of Chicago), Yiping Lu `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过矩阵算子范数的几何视角设计宽度可扩展的优化器，并提出MOGA。

**💡 创新点**

创新点在于引入均值归一化算子范数 (p,mean)→(q,mean)，解决层间兼容性，实现宽度无关的Lipschitz和光滑度，并基于此构造宽度感知的MOGA优化器。

**🔧 技术方法**

使用矩阵算子范数、极值理论、Steepest Descent、均值归一化、光滑性分析、行/列归一化以及MOGA算法等技术。

**📊 数据集**

实验数据集包括GPT‑2的OpenWebText、LLaMA的C4，以及对应的Chinchilla‑optimal token预算。

**📈 对比分析**

与AdamW、Muon、Adam/SignSGD等方法对比，MOGA在相同或更低学习率下收敛更快，尤其在大token和低loss阶段优于Muon，学习率可在不同宽度间直接迁移。

**⚠️ 局限性**

局限性：理论基于最坏情况的Lipschitz/光滑度估计，未涵盖正则化或动态梯度噪声；在极端宽度/深度模型或非语言任务中的表现尚待验证。

---

## 595. The Confidence Gate Theorem: When Should Ranked Decision Systems Abstain?

**arXiv ID:** 2603.09947 | [PDF](https://arxiv.org/pdf/2603.09947v1)

**作者:** Ronald Doku `[一作]` `[通讯]` (Haske Labs), Ronald Doku (Haske Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并验证了 Confidence Gate Theorem，阐明在排名决策系统中置信门控是否能单调提升决策质量，并给出实际的部署诊断流程。

**💡 创新点**

创新点在于将置信门控的单调性归因于结构性与上下文性不确定性的区别，证明结构性不确定性满足C1/C2条件时门控有效，而上下文性漂移时需采用基于集成或递归特征的置信度，否则门控会失效。

**🔧 技术方法**

技术上采用了置信门控理论、Spearman/Kendall 相关分析、误差分布检验、残差异常标签评估、集成不确定性、递归特征以及滑动窗口自适应校准等方法。

**📊 数据集**

实验使用了三大领域的数据集：MovieLens（协同过滤）、RetailRocket/Criteo/Yoochoose（电商意图检测）以及MIMIC‑IV（临床路径分流），覆盖了结构性与上下文性不确定性场景。

**📈 对比分析**

通过对比置信门控与随机、残差预测、集成、递归特征等策略，发现结构性场景下计数置信度可实现无逆转单调提升；在上下文漂移场景下，计数置信度与随机相当，集成与递归特征可将逆转次数降至1–2，但仍未完全恢复单调性。

**⚠️ 局限性**

局限性包括：理论假设置信函数固定、离线评估未覆盖真实在线实验、对上下文漂移场景的负面结果仅来自一个数据集、跨域置信阈值迁移不确定，以及未能完全解决上下文不确定性导致的非单调性。

---

## 596. SignalMC-MED: A Multimodal Benchmark for Evaluating Biosignal Foundation Models on Single-Lead ECG and PPG

**arXiv ID:** 2603.09940 | [PDF](https://arxiv.org/pdf/2603.09940v1)

**作者:** Fredrik K. Gustafsson `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**通讯引用:** 13890 | [OpenAlex ID](https://openalex.org/A5040302008)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了SignalMC-MED基准，用10分钟同步的单导联ECG和PPG数据评估医学信号基础模型的性能

**💡 创新点**

首次在大型多模态临床记录上系统比较时间序列与心电/光电容积波形专用基础模型，验证多模态融合和长时序对预测的益处

**🔧 技术方法**

采用冻结特征提取的线性探测框架，对MOMENT、Chronos-Bolt、D-BETA、ECGFounder、xECG、PaPaGei、CSFM以及手工ECG/PPG特征进行评估

**📊 数据集**

使用Stanford MC‑MED 22,256次急诊访问的10分钟同步ECG/PPG信号，涵盖20个临床任务（年龄、性别、出院/入院、实验室值、ICD‑10诊断）

**📈 对比分析**

在ECG‑only、PPG‑only、ECG+PPG三种模式下对10%、25%、50%、100%训练集进行线性回归/分类；CSFM‑base在多模态下获最佳总体排名，xECG排名第二；多模态融合均提升性能，长时序（10min）优于短时段，模型规模不一定带来收益，手工特征与学习表示可互补

**⚠️ 局限性**

仅使用冻结特征、线性下游模型，未考虑微调；仅评估首10min数据，未覆盖更长连续监测；数据来源单一医疗系统，可能缺乏跨机构通用性；未探索更高级多模态融合与时序聚合方法

---

