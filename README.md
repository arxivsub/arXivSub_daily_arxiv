# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-13 | 今日论文总数: 497

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. State Space Models are Effective Sign Language Learners: Exploiting Phonological Compositionality for Vocabulary-Scale Recognition

**arXiv ID:** 2604.08761 | [PDF](https://arxiv.org/pdf/2604.08761v1)

**作者:** Bryan Cheng `[一作]` (William A. Shine Great Neck South High School), Jasper Zhang `[通讯]` (William A. Shine Great Neck South High School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了首个显式利用手语音位结构的骨架序列识别网络PhonSSM，能够在大规模词汇下实现高精度识别。

**💡 创新点**

创新点在于通过解剖学图注意力、正交分解四大音位子空间、双向状态空间模型和分层原型分类，强制模型学习可组合的音位表示，从而突破传统“扁平”表征导致的词汇扩展瓶颈。

**🔧 技术方法**

主要技术包括：解剖学图注意力网络（GAT）、正交损失促成音位子空间分离、Mamba式双向状态空间模型（BiSSM）用于高效时序建模，以及分层原型学习实现少样本泛化。

**📊 数据集**

使用了多大规模手语数据集：WLSL（从100到2000个手语，采用75个骨架点）和新构建的Merged‑5565（5,565个手语，21个主手骨架点），验证了模型在不同词汇规模下的性能。

**📈 对比分析**

与目前骨架基线（DSTA‑SLR、Pose‑TGCN、SignBERT 等）及视频基线（I3D）对比，PhonSSM 在 WLASL2000 上提升了 18.4pp（从 53.7% 提升到 72.1%），在 Merged‑5565 上实现 53.3% 的准确率，显著优于 Bi‑LSTM（27.4%）且在少样本/零样本场景下提升超过 225%。

**⚠️ 局限性**

主要局限包括：仅针对离散手势（非连续流畅签），音位类别固定未进行自学习，评估仅覆盖 ASL，且在中等词汇量（WLASL300/1000）时因“精度-泛化”权衡导致性能略逊于细粒度注意力模型。

---

## 2. Loom: A Scalable Analytical Neural Computer Architecture

**arXiv ID:** 2604.08816 | [PDF](https://arxiv.org/pdf/2604.08816v1)

**作者:** Mehmet Kerem Turkcan `[一作]` `[通讯]` (Columbia University), Mehmet Kerem Turkcan (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于 8 层分析构造权重的 Transformer 架构 Loom，用固定权重在矩阵乘法中执行 C 程序的 22 条指令集，构建了对应的编译器和完整的硬件/软件执行路径。

**💡 创新点**

核心创新包括：① 将所有指令映射为共享减法核心的操作，省去每条指令单独的算术层；② 用单层 6 阈值 ReLU 直接实现 borrow‑chain 减法；③ 在指令集加入 STORE 进行间接写入，显著减少编译代码长度；④ 采用分析构造的稀疏权重，支持多种规模配置；⑤ 提供浏览器、ONNX、JavaScript Argmax 和 FPGA 四条执行路径。

**🔧 技术方法**

技术手段包括：分析构造 Transformer 权重（Q=K，V=0 等），多头注意力与 FFN 的组合，Bipolar 编码与 ReLU 门控，指令译码与寄存器重写，单层减法实现，指令集编译器（Python/JavaScript），ONNX 导出与 WebGPU/WebAssembly 执行，FPGA 侧的 argmax 注意力与顺序列处理。

**📊 数据集**

未使用传统机器学习数据集，而是基于自定义的 C 语言子集编译出的程序（如排序、Snake、Raycasting、Sudoku 等）进行验证，覆盖 111 个单元测试。

**📈 对比分析**

通过 42 条指令级单元测试、19 条交叉写回测试、50 条完整程序测试以及硬件/软件路径交叉验证，所有路径均通过全部测试。性能方面：FPGA 每步约 0.3 s，GPU WebGPU 约 10 ms/步；模型大小从 7.4 MB（146×512）到 29 MB（164×2048）；单层减法和 argmax 设计显著降低层数和资源占用。

**⚠️ 局限性**

局限性包括：① 需手写 C 子集编译器，语言支持有限；② 仅支持固定规模的状态张量，无法动态扩展内存；③ 在 FPGA 上实现仅使用 LUT 计算，速度慢于 GPU；④ 需人工重编译不同尺寸配置，未实现跨配置即时迁移；⑤ 对异常指针或非法指令的行为未完全定义。

---

## 3. Systematic API Testing Through Model Checking and Executable Contracts

**arXiv ID:** 2604.08633 | [PDF](https://arxiv.org/pdf/2604.08633v1)

**作者:** Ana Ribeiro `[一作]` (NOVA University Lisbon), Carla Ferreira `[通讯]` (NOVA University Lisbon)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于模型检测的系统化黑盒REST API测试框架，通过扩展OpenAPI规范生成可执行的契约并使用TLC模型检查器完成状态空间覆盖，生成完整的调用序列并执行测试；

**💡 创新点**

创新点在于：①将模型检测与黑盒测试结合，利用TLC完整状态空间搜索产生覆盖保证；②设计了基于一阶逻辑的可执行契约语言，可自动从OAS推导并支持手工扩展；③采用覆盖引导的宽度优先遍历实现状态与转换双重覆盖；

**🔧 技术方法**

主要技术包括：TLA+建模与TLC模型检查；一阶逻辑契约语言与自动契约生成；Java实现的调用序列生成算法；HTTP请求执行与状态仿真；

**📊 数据集**

评估数据集采用EvoMaster Benchmark中的四个系统（Tournaments、Petstore、E‑Commerce、Features‑Service），并在这些系统上注入错误进行验证；

**📈 对比分析**

实验显示：在规模约5–21 GB的状态空间内，TLC可在一小时内完成；生成的调用序列实现100 %状态覆盖，且在Tournaments系统中能发现所有注入的错误；在Petstore等较大系统中实现约70–80 %转换覆盖；相比现有工具，能够捕捉多操作相关的细粒度错误，oracle准确性高；

**⚠️ 局限性**

局限性包括：要求API严格遵循REST规范并提供完整的状态信息；模型和状态空间尺寸对性能有显著影响（超过≈46 k状态时内存消耗大）；手工编写契约和模型仍需人工干预；并行转换时无法保证完全转换覆盖；

---

## 4. Multi-User Large Language Model Agents

**arXiv ID:** 2604.08567 | [PDF](https://arxiv.org/pdf/2604.08567v1)

**作者:** Shu Yang `[一作]` (KAUST), Jiaxin Pei `[通讯]` (Stanford University)

**通讯引用:** 337 | [OpenAlex ID](https://openalex.org/A5049055745)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

请提供论文的完整内容或主要段落，我才能为您做精确总结。

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 5. Retrieval Augmented Classification for Confidential Documents

**arXiv ID:** 2604.08628 | [PDF](https://arxiv.org/pdf/2604.08628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 6. CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference

**arXiv ID:** 2604.08584 | [PDF](https://arxiv.org/pdf/2604.08584v1)

**作者:** Chuxu Song `[一作]` (Rutgers University), Chuanhui Yang `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的稀疏注意力方法CSAttention，利用预填充阶段构建查询中心化的查找表，显著提升长上下文LLM推理速度。

**💡 创新点**

通过将查询空间分块并在每块上聚类构建查询中心化的子空间表，避免传统基于键聚类的检索中查询‑键分布偏移问题，实现高稀疏度下的近乎完整准确率。

**🔧 技术方法**

采用子空间划分、cosine k‑means聚类、预计算中心化得分、固定容量Top‑L查找表、GPU友好的稀疏累计与动态增量更新等技术。

**📊 数据集**

在LongBench、LongBench‑v2以及Llama‑3.1‑8B、Qwen‑3‑8B、Mistral‑7B等多种指令调优模型上进行评估。

**📈 对比分析**

与PQCache、H_2O、SparQ、MagicPig等现有稀疏注意力基线在95%稀疏率、32K–128K长上下文下对比，CSAttention在保持≤0.7%精度损失的同时实现最高达4.6×的推理速度提升。

**⚠️ 局限性**

仍需一次性预填充构建表，适用于可重复使用的长上下文场景；对极端动态分布漂移或极短上下文的适用性有限。

---

## 7. LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification

**arXiv ID:** 2604.08741 | [PDF](https://arxiv.org/pdf/2604.08741v1)

**作者:** Lucas Wojcik `[一作]` (Federal University of Paraná), David Menotti `[通讯]` (Federal University of Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

扩充并纠正了现有的车牌可读性（legibility）基准数据集，加入约3倍的新样本、细粒度的车牌、车辆和摄像头级别注释，并提出改进的训练策略。

**💡 创新点**

① 通过纠正原标注错误提升数据一致性；② 引入额外数据显著扩充样本量；③ 采用基于EMA的动态加权交叉熵损失和细化学习率调度；④ 设计摄像头污染（camera contamination）评估协议。

**🔧 技术方法**

使用预训练的ResNet‑50网络，配合EMA动态加权损失、Adam优化器、线性学习率衰减、早停以及批大小64。

**📊 数据集**

改进后的数据集是原基准（LPLCv1）的扩展版本，包含37,099张图像、41,487个车牌标签、700+摄像头标识以及车辆属性。

**📈 对比分析**

在四种评估场景（Baseline、Legibility Recognition、Full Recognition、Quality Filter）下，基线F1从74.5%提升至84.4%；加入更多数据后提升至88.7%；EMA损失进一步提升至89.8%。摄像头交叉验证显示性能下降有限（≈1%），证明模型具有较好的泛化性。

**⚠️ 局限性**

仍存在：① 对不同车牌布局的跨数据集泛化尚未验证；② 边界类别仍有误判，说明数据本身存在不可避免的歧义；③ 只使用单个网络，未结合车牌超分辨率或字符级特征，可能进一步提升可读性评估。

---

## 8. Accurate and Reliable Uncertainty Estimates for Deterministic Predictions Extensions to Under and Overpredictions

**arXiv ID:** 2604.08755 | [PDF](https://arxiv.org/pdf/2604.08755v1)

**作者:** Rileigh Bandy `[一作]` (Sandia National Laboratories), Rebecca Morrison `[通讯]` (University of Colorado Boulder)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

扩展ACCURE框架以学习输入相关的非高斯不确定性分布（双边高斯和非对称拉普拉斯），实现对黑盒模型误差的概率预测。

**💡 创新点**

创新点在于提供可解析CRPS与RS的非高斯分布，允许捕捉偏斜和重尾误差，同时保留ACCURE的可解释性和输入依赖性。

**🔧 技术方法**

采用神经网络参数化误差分布参数，并通过平衡准确性与可靠性的ACCURE损失函数进行训练，使用β参数搜索以确定两者权衡。

**📊 数据集**

使用合成数据（包含线性、三角函数和混合参数的双边高斯/非对称拉普拉斯误差）以及真实天气数据（丹佛机场2022–2023年一小时温度观测与NOAA HRRR预测）。

**📈 对比分析**

与确定性HRRR、合成预测（CP）和EasyUQ比较；在天气预报实验中，非高斯ACCURE在其原生目标上取得最低损失，CRPS表现与CP、EasyUQ相近，且在50%置信区间内预测吻合度高。

**⚠️ 局限性**

局限性包括对误差分布形式的假设，误差分布模型未必总能完美拟合真实误差；在95%置信区间预测可能欠准确，且对更复杂多维输入和更大空间天气问题的适用性尚待验证。

---

## 9. Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation

**arXiv ID:** 2604.08627 | [PDF](https://arxiv.org/pdf/2604.08627v1)

**作者:** Yongchan Chun `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级后置模块Evidential Transformation Network (ETN)，将预训练模型转换为能够输出Dirichlet分布的证据学习模型，进而实现可靠的不确定性估计。

**💡 创新点**

创新点在于仅通过对预训练模型的logit空间做可样本自适应的仿射变换，并用变分分布对变换参数建模，避免了大规模再训练或多模型集成，显著降低了推理成本。

**🔧 技术方法**

使用了变分贝叶斯技术（对变换参数做Gamma分布建模），softplus函数映射logits为Dirichlet浓度，ELBO损失与Dirichlet KL散度结合训练ETN，并在推理时采用Monte‑Carlo采样。

**📊 数据集**

在图像分类上使用CIFAR‑10、ImageNet及其OOD集合；在大型语言模型上使用Llama‑3.1‑8B，在多项选择QA基准OBQA、RACE以及MMLU的子集做ID与OOD评估。

**📈 对比分析**

与传统的深度集成、MC‑Dropout、Laplace、以及后置EDL方法（MAP_EDL、IB‑EDL、DMM）等基线对比，ETN在保持原始准确率的同时，AUPR、MI、DE等不确定性指标均优于或竞争性地超过所有基线，且推理时间几乎无额外负担。

**⚠️ 局限性**

局限性包括：对超参数（如A的先验模式、Monte‑Carlo采样数）的敏感性；在极大类别数的LLM任务中，先验模式需谨慎设置以防性能下降；以及仅在logit空间进行变换，可能无法充分利用更深层特征的潜在信息。

---

## 10. WildDet3D: Scaling Promptable 3D Detection in the Wild

**arXiv ID:** 2604.08626 | [PDF](https://arxiv.org/pdf/2604.08626v1)

**作者:** Weikai Huang `[一作]` (Allen Institute for AI), Ranjay Krishna `[通讯]` (Allen Institute for AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个支持文本、点击和 2D 框三种提示方式的开放词汇单目 3D 目标检测框架，同时可利用可选深度信息进行几何推理。

**💡 创新点**

创新点在于：①双视觉编码器结合深度融合模块，能够在缺失深度时自适应降维；②提示可编程检测器将多种提示统一为共享查询，解决多模态交互；③在 13.5k 类别、100 万张图像的人类验证 3D 数据集 WildDet3D，显著扩展了开放世界 3D 注释规模；④通过深度、相机投影与 3D 旋转的多源信息聚合，提出无歧义旋转归一化实现精确 3D 盒子回归。

**🔧 技术方法**

核心技术包括：ViT-H+SimpleFPN 视觉特征提取、DINOv2 ViT-L/14+ConvStack 作为可选深度后端、ControlNet 风格深度融合、Promptable Detector（文本+几何编码 + Transformer 交叉注意）、深度学习深度与相机投影的多源交叉注意、6D 旋转表征与 Gram-Schmidt 正交化、单机 3D 盒子 head 的深度监督、O2M 匹配和 ignore‑aware 训练等。

**📊 数据集**

使用了 COCO、LVIS、Objects365、V3Det 的 2D 框作为基准，构建了 WildDet3D（1M 图像、13.5k 类别、3.7M 3D 注释）作为开放词汇数据集，并在 Omni3D、Argoverse‑2、ScanNet、Stereo4D 等公开基准上评测。

**📈 对比分析**

在 WildDet3D 评测中，使用文本提示 22.6 AP、盒子提示 24.8 AP，深度输入时分别提升到 41.6 AP / 47.2 AP；在 Omni3D 上文本提示 34.2 AP、盒子提示 36.4 AP，远超 3D‑MOOD、Uni‑MODE 等方法；在零样本跨域上，Argoverse‑2、ScanNet 的 ODS 分别达到 40.3 / 48.9，显著优于基线。整体提升幅度在 2-4 倍以上，且训练周期仅 12 轮即可达到最优。

**⚠️ 局限性**

局限性包括：①相机内参预测误差导致定位偏差；②单目深度仍受尺度和遮挡限制，远距离或稀疏对象精度低；③旋转估计对对称或表面信息不足的物体表现不佳；④双编码器架构占用显存/算力，移动端实时部署受限；⑤长尾类别仍缺乏足够训练样本；⑥目前未提供安全性保证，不能用于关键任务（如自动驾驶、手术规划等）。

---

## 11. Silhouette Loss: Differentiable Global Structure Learning for Deep Representations

**arXiv ID:** 2604.08573 | [PDF](https://arxiv.org/pdf/2604.08573v1)

**作者:** Matheus Vinícius Todescato `[一作]`, Joel Luís Carbonera `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种可微分的 Soft Silhouette Loss，并与监督对比学习（SupCon）相结合，形成轻量级的全局结构优化目标，用于学习更具判别力的表示。

**💡 创新点**

创新点在于将传统聚类指标 silhouette coefficient 转化为可微分目标，并将其与本地对比学习联合，既保证局部相似性，又提升全局聚类质量，且计算开销极低。

**🔧 技术方法**

技术上使用了差分化的 Soft Silhouette Loss（soft‑min 与 log‑sum‑exp）、监督对比学习（SupCon/SupCon2）、交叉熵、EfficientNet‑B0 编码器、投影头、AutoAugment 等。

**📊 数据集**

实验数据集包括七个公开图像分类基准：CIFAR‑10、CIFAR‑100、Stanford Cars、Caltech‑101、Caltech‑256、FGVC‑Aircraft、Oxford Flowers。

**📈 对比分析**

与 CE、SupCon、SupCon2、Proxy‑NCA、Center Loss 等基线比较，平均 Top‑1 准确率从 36.71% 提升至 39.08%，CE+Sil 提升约 0.38%，SupCon2 提升约 0.53%，最优结果来自 CE+Sil+SupCon2，显示显著性能提升。

**⚠️ 局限性**

局限性包括 Silhouette 仅在 mini‑batch 内估计，易受 batch 组成影响；单独使用 CE+Sil 的提升不稳定；对 λ、温度等超参数敏感；对极大 batch 或大规模数据集的效率仍需进一步验证。

---

## 12. Semantic Intent Fragmentation: A Single-Shot Compositional Attack on Multi-Agent AI Pipelines

**arXiv ID:** 2604.08608 | [PDF](https://arxiv.org/pdf/2604.08608v1)

**作者:** Tanzim Ahad `[一作]` (University of Texas at El Paso), Sajedul Talukder `[通讯]` (University of Texas at El Paso)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了一种新型攻击——Semantic Intent Fragmentation (SIF)，演示了在LLM多代理管道中单次合法请求如何被自动拆解成多条看似安全的子任务，最终组合后违反安全策略。

**💡 创新点**

创新点在于：①首次将“单射自主”概念与“计划级安全缺口”结合，揭示了子任务级审计无法检测的组合威胁；②给出正式的SIF定义、FS指标、DDT定理与CIV计划级合规判定器；③展示了通过计划级信息流追踪与合规评估即可零误报地捕获所有SIF攻击。

**🔧 技术方法**

技术方法包括：使用大型语言模型模拟LLM orchestrator与子代理；三阶段LLM生成 pipeline 产生无作者偏差的攻击请求；六家分类器电池与 G‑Eval 4 步链路思考；确定性信息流污点分析；跨模型合规评判器（CIV）以及对照实验与消融分析。

**📊 数据集**

数据集为 14 个企业场景（财务、信息安全、HR）和相应的 8 个安全对照请求，所有请求均通过基于 OWASP、MITRE 与 NIST 框架的三层 grounding 生成，确保真实性且避免研究者知识偏差。

**📈 对比分析**

与传统多代理攻击对比，SIF 在 71% 的场景中成功产生违规计划（FS=1.0），L1 与 CIV 的联合门控实现 100% 检测率且 0% 虚警；强大 orchestrator 版本提升 36% 的成功率，温度稳定性证明方法鲁棒；消融实验验证了请求措辞、orchestrator 能力及计划级防御的关键作用。

**⚠️ 局限性**

局限性包括：仅覆盖 14 个人工设计场景，缺乏真实生产框架（LangGraph/AutoGen/CrewAI）直接实验；信息流污点规则对某些细粒度泄露（如 P02-M3）不完全；L1 过滤器过度触发导致需与 CIV 联合；并且目前仅关注企业数据安全，需进一步扩展至更广泛领域和攻击向量。

---

## 13. Neural networks for Text-to-Speech evaluation

**arXiv ID:** 2604.08562 | [PDF](https://arxiv.org/pdf/2604.08562v1)

**作者:** Ilya Trofimenko `[一作]` (HSE University), Nikita Shevtsov `[通讯]` (Institute for System Programming, Russian Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一系列神经网络模型，用于自动评估文本转语音（TTS）系统的质量，包括相对偏好评估（SBS）和绝对质量评估（MOS）

**💡 创新点**

创新点在于：1) 通过HuBERT构建的NeuralSBS实现近74%的SBS准确率，并加入协作过滤式的标准化（Std SOMOS）来缓解评估者偏差；2) 通过WhisperBert的多模态堆叠式集成，以Late‑stage融合方式有效利用音频与文本信息，显著降低MOS RMSE；3) 对传统MOSNet进行批量长度排序、掩码和序列级损失等训练改进，实现更稳健的学习；4) 系统性评估了LLM零样本评估的局限性。

**🔧 技术方法**

使用的技术包括：自监督语音编码器HuBERT、Whisper、BERT、交叉注意力、抗对称双线性池化、堆叠式弱学习器（Ridge、SVR、DT）、Meta‑Learner MLP、Dropout、BatchNorm、长度排序、序列掩码等；实验框架基于PyTorch/Lightning。

**📊 数据集**

数据集主要是SOMOS（约2万条音频，36万条MOS评分，90k对SBS数据），并对其进行标准化（Std SOMOS）以减轻评估者偏差；还使用公开的VoiceMOS Challenge数据做对比。

**📈 对比分析**

与人类评估相比，NeuralSBS在SBS任务上达到73.7%准确率，接近人类一致性；WhisperBert在MOS任务上RMSE降至0.402（Std SOMOS），显著优于MOSNet（0.422）和MOSNetBert（0.496）；与UTMOS对比，WhisperBert在绝对误差上更低（MSE 0.161 vs 0.242），但在相关性上略逊。零样本LLM（Qwen2-Audio、Gemini 2.5）在SBS/ MOS任务上表现远逊。

**⚠️ 局限性**

局限性包括：1) 直接跨模态融合（Cross‑Attention）对性能有负面影响；2) SpeechLM基架在小数据集上难以训练，易收敛到平均值；3) 标准化后样本差异减小，可能导致模型难以区分微小差异；4) 仅评估英文，跨语言泛化尚待验证；5) 目前模型仍依赖大量人工标注，未实现完全零样本评估。

---

## 14. Adaptive Rigor in AI System Evaluation using Temperature-Controlled Verdict Aggregation via Generalized Power Mean

**arXiv ID:** 2604.08595 | [PDF](https://arxiv.org/pdf/2604.08595v1)

**作者:** Aleksandr Meshkov `[一作]` `[通讯]` (Independent Researcher), Aleksandr Meshkov (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了温度控制判决聚合（TCVA）方法，用于评估基于LLM的AI系统，通过五级判决、幂平均聚合及温度参数调节评估严格度。

**💡 创新点**

创新点在于：①将判决尺度扩展为五级（如Fully、Mostly、Partially、Minor、None），②引入幂平均聚合以可调节极值影响；③设计直观温度参数映射至p值，实现评估严格度的灵活控制。

**🔧 技术方法**

技术核心包括：LLM‑as‑a‑Judge判决提取、五级权重映射、泛化幂平均聚合、温度‑p映射以及对None判决的自适应惩罚。

**📊 数据集**

使用的基准数据集为SummEval（摘要真实性、相关性）和USR（对话保持上下文），均配有人类Likert量表评分。

**📈 对比分析**

在与RAGAS（二元判决+算术平均）和DeepEval（三元判决）的对比实验中，TCVA在Faithfulness上Spearman ρ=0.667与RAGAS的0.676相当，在Relevancy上0.480显著优于0.411；对话评估均低于RAGAS，但仍高于DeepEval。

**⚠️ 局限性**

局限性包括：需人工校准温度选择、句子提取的准确性依赖提示工程、判决离散等级可能不足以捕获细微差异、对话评估仍表现不佳、仅使用单一判决模型（GPT‑4.1‑mini）验证。

---

## 15. Towards Responsible Multimodal Medical Reasoning via Context-Aligned Vision-Language Models

**arXiv ID:** 2604.08815 | [PDF](https://arxiv.org/pdf/2604.08815v1)

**作者:** Sumra Khan `[一作]` (Salim Habib University), Rizwan Qureshi `[通讯]` (Salim Habib University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种“上下文对齐”框架，让医学视觉‑语言模型在做诊断前必须跨图像、报告、放射组学、可解释性激活和词汇语义等多种证据通道达成一致，并输出包含印象、证据、置信度、局限性和安全说明的结构化结果。

**💡 创新点**

创新点在于：1）通过约束多源证据的一致性来抑制医学幻觉；2）在不改动模型参数的情况下，仅通过决策协议实现更可靠的推理；3）强制输出统一的JSON结构，显式报告不确定性与局限，提升模型可解释性与责任感。

**🔧 技术方法**

技术手段包括：冻结 Qwen2‑VL（或 LLaVA）多模态基础模型；使用放射组学统计（灰度共生矩阵、纹理指标）、Grad‑CAM 生成可解释性激活摘要；从报告中提取医学词汇表概念；三阶段上下文对齐流程（提取 → 序列化融合 → 工具增强推理）。

**📊 数据集**

实验数据集：OpenI（包含成对胸片与详细报告，用于训练、消融与 50 例代理推理评估）以及 CheXpert（用于跨数据集验证）。

**📈 对比分析**

比较方法：对不同证据组合（文本、放射组学、XAI、全组合）做逻辑回归 AUC 评估；与单模态基线对比，AUC 从 0.918 提升到 0.925；幻觉率从 1.14 降到 0.25；推理证据长度缩短 19.4→15.3；不确定性保持稳定。跨数据集验证显示模型在 CheXpert 上保持相似的置信度和安全性，证明方法具备一定泛化能力。

**⚠️ 局限性**

局限性：提升幅度有限，仍需高质量文本报告以发挥最大效果；在报告简短或仅有标签式数据时多源证据贡献弱；模型对不同模态的权重仍固定，可能无法自动适应不同病种或影像设备；目前仅在胸片任务验证，需进一步扩展至其他影像模态。

---

## 16. Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models

**arXiv ID:** 2604.08557 | [PDF](https://arxiv.org/pdf/2604.08557v1)

**作者:** Arth Singh `[一作]` `[通讯]` (AIM Intelligence), Arth Singh (AIM Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在扩散式语言模型的去噪轨迹中重掩第一批拒绝令牌并注入肯定前缀，展示了安全对齐的结构性脆弱性；

**💡 创新点**

创新点在于揭示并利用扩散模型单一假设——单调去噪且已承诺的词不可再评估，证明无梯度、规则化前缀即可达到 76–94% 的攻击成功率；

**🔧 技术方法**

主要技术包括重掩（re‑mask）操作、固定长度肯定前缀注入以及尝试的 Gumbel‑softmax 连续梯度优化；

**📊 数据集**

实验基于公开扩散模型 LLaDA‑8B‑Instruct 与 Dream‑7B‑Instruct，并使用 HarmBench 进行 159 条危害行为的评测；

**📈 对比分析**

与未攻击基线（ASR 0%）对比，核心攻击在不同生成长度下可达 76–94% 的攻击成功率；梯度优化反而使 ASR 降至 41% 左右，表明攻击效果完全结构性而非梯度驱动；

**⚠️ 局限性**

主要局限包括仅评估两种模型、白盒威胁模型、仅使用 HarmBench、确定性贪婪解码，且对随机解码或更大规模模型的稳健性未验证。

---

## 17. An Eye for Trust: An Exploration of Developers' Trust Perceptions Through Urgency and Reputation

**arXiv ID:** 2604.08713 | [PDF](https://arxiv.org/pdf/2604.08713v1)

**作者:** Sara Yabesi `[一作]` (Polytechnique Montréal), Zohreh Sharafi `[通讯]` (Polytechnique Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一项对37名学生的受控实验中，使用眼动追踪技术观察开发者在代码审查时对代码补丁的信任感，实验操纵补丁的紧急程度（高/低优先级）和作者经验（高级/初级）两个变量，评估其对审查行为、认知负荷和重用意图的影响。

**💡 创新点**

首次通过实验和眼动追踪量化研究紧急性与声誉对开发者信任认知的作用，探讨了在保证代码质量不变的情况下，信任的心理机制与行为表现如何被外部信号改变。

**🔧 技术方法**

采用Tobii Pro Fusion眼动仪与iTrace插件记录视觉注意力；使用Likert量表收集自评信任与重用意向；统计分析包括Wilcoxon、χ²、Mann‑Whitney等非参数检验。

**📊 数据集**

数据集：选自Defects4j‑Repair的6个Java补丁（均已控制质量），嵌入在jFreeChart 1.1.0项目中，参与者共完成6个审查任务。

**📈 对比分析**

比较方法：对比高低优先级和高级/初级作者对接受率、准确率、审查时长和眼动指标（fixation count、average fixation duration、total fixation time）的影响；结果显示高优先级补丁显著增加审查时长和视觉负荷，但对接受率或准确率无显著影响；作者经验对行为与性能无显著作用。

**⚠️ 局限性**

局限性：受试者仅为学生，经验水平有限；实验仅涵盖单一开源项目和少量补丁；紧急性与声誉仅以二元标签呈现；未考虑其他影响信任的因素（如作者知名度、代码复杂度等）。

---

## 18. GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback

**arXiv ID:** 2604.08553 | [PDF](https://arxiv.org/pdf/2604.08553v1)

**作者:** Ruiyao Xu `[一作]` (Northwestern University), Kaize Ding `[通讯]` (Northwestern University)

**通讯引用:** 2829 | [OpenAlex ID](https://openalex.org/A5044455276)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为GNN-as-Judge的框架，利用图神经网络（GNN）与大型语言模型（LLM）协同，在低资源文本属性图（TAG）的半监督节点分类任务中生成可靠的伪标签并进行弱监督微调；

**💡 创新点**

创新点在于：①将GNN作为“评判者”通过同意与分歧策略挑选既可靠又具有挑战性的伪标签；②引入基于节点影响度的高效子集选择；③结合指令调优与偏好调优（ORPO）的弱监督微调，降低伪标签噪声；

**🔧 技术方法**

采用GCN/SGC等GNN进行结构影响度计算，使用Llama-3-8B-Instruct等LLM进行文本推理，设计协同伪标签生成与弱监督微调算法（指令调优+偏好调优），并利用top‑K与阈值过滤节点；

**📊 数据集**

实验数据集包括Cora、Citeseer、Pubmed、ogbn-arxiv、ogbn-products等文本属性图，覆盖从3到15-shot的低资源设置及跨域零样本场景；

**📈 对比分析**

与传统GNN（GCN、SGC）、LLM-as-Predictor（Zero-Shot、Graph-CoT、Neighbor等）以及LLM-Graph方法（GLEM、TAPE、LLM-GNN、LLaGA、GraphGPT）进行对比。结果显示GNN-as-Judge在所有数据集、所有shot下均优于基线，尤其在3/5-shot等极低资源条件下显著提升；在零样本跨域实验中也表现最强；

**⚠️ 局限性**

局限性包括：①仍需先训练GNN并计算影响度，额外的计算开销；②对超参数（top‑K、阈值）有一定敏感性；③LLM推理成本高，难以在极大规模图上直接部署；④若GNN误判，伪标签可能被错误引导，影响最终性能。

---

## 19. Pretrain-then-Adapt: Uncertainty-Aware Test-Time Adaptation for Text-based Person Search

**arXiv ID:** 2604.08598 | [PDF](https://arxiv.org/pdf/2604.08598v1)

**作者:** Jiahao Zhang `[一作]` (University of Macau), Zhedong Zheng `[通讯]` (University of Macau)

**通讯引用:** 9993 | [OpenAlex ID](https://openalex.org/A5034162160)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了“Pretrain-then-Adapt”框架，利用无标签测试数据通过不确定性感知的测试时适配（UATTA）实现文本-图像跨模态检索的领域适应。

**💡 创新点**

创新点在于：① 通过双向检索不一致度估计不确定性，从而避免传统熵最小化导致的过度自信；② 结合循环一致性筛选可靠样本；③ 在保持冻结预训练模型不变的前提下，仅在测试时进行轻量化参数微调，实现零标注、低成本部署。

**🔧 技术方法**

使用的技术包括CLIP/ XVLM预训练模型、熵最小化、双向检索概率差异作为不确定性指标、循环一致性样本筛选、AdamW优化器及对LayerNorm的局部参数微调。

**📊 数据集**

在四大基准上验证：CUHK-PEDES、ICFG-PEDES、RSTPReid（文本-图像检索）以及PAB（细粒度匹配）。

**📈 对比分析**

与传统Pretrain‑then‑Finetune、其他TTA方法（Tent、SHOT、SAR、READ、TCR）和无监督/半监督方法比较，UATTA在R@1、mAP均取得+1–4个百分点的提升，且后训练时间仅为传统方法的千分之一，显著提升了效率与性能。

**⚠️ 局限性**

局限性包括：① 仅针对Top‑1的误检进行校正，导致在Top‑5/10的召回略有下降；② 依赖循环一致性筛选，可能丢弃真实但难检索的样本；③ 对检索噪声或文本/图像质量差的场景仍易产生高不确定性，影响适配效果。

---

## 20. TiAb Review Plugin: A Browser-Based Tool for AI-Assisted Title and Abstract Screening

**arXiv ID:** 2604.08602 | [PDF](https://arxiv.org/pdf/2604.08602v1)

**作者:** Yuki Kataoka `[一作]` (Nagoya University), Toshi A. Furukawa `[通讯]` (Kyoto University)

**通讯引用:** 52816 | [OpenAlex ID](https://openalex.org/A5085050194)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了 TiAb Review Plugin，一款无代码、无服务器的 Chrome 浏览器扩展，用于标题与摘要的筛选，支持手动、机器学习主动学习和 Gemini LLM 批量筛选三种模式。

**💡 创新点**

创新点包括：①首次实现浏览器内嵌、无服务器的系统筛选工具；②将 ASReview 的主动学习算法完整重写为 TypeScript 并验证与原版一致；③在无服务器环境中集成 Gemini LLM，并对其批量筛选性能进行系统验证；④使用 Google Sheets 作为共享数据库，实现多审稿人协作与审计追溯。

**🔧 技术方法**

使用技术包括：Chrome Extension（Manifest V3）、TypeScript、IndexedDB、本地加密存储、Google Sheets API、Gemini API、TF-IDF+朴素贝叶斯、主动学习、LLM prompt 设计、加密 API key 存储。

**📊 数据集**

使用的数据集：①六个公开基准集（用于验证 ML 实现的一致性）；②抑郁模型数据集（1,993 条记录）用于 LLM 参数调优；③五个公开的感染性疾病相关数据集（CQ1–CQ5，1,038–5,628 条记录）用于跨数据集 LLM 验证。

**📈 对比分析**

比较方法：对 ML 模型验证使用 10 折交叉验证，比较 top-100 排名一致性；对 LLM 模型调优使用 Fβ=7 评价，选取最佳参数；跨数据集评估使用敏感度、特异度、精确度、WSS@95 等指标。结果显示 ML 版本与原版完全一致；LLM 在最佳配置下敏感度达 96.1%，精确度 53.4%，在五个数据集上敏感度 94–100%，精确度 2–15%，WSS@95 46–89%。

**⚠️ 局限性**

局限性包括：①仅验证了 top-100 排名的一致性，未评估低位排名差异；②仅测试了 Google Gemini 语言模型，其他 LLM 未覆盖；③数据集集中在重症医学领域，缺乏跨学科验证；④所有评估均为回顾性实验，未进行前瞻性系统评价或实际使用中的时间节省测量；⑤缺乏系统的可用性和用户体验评估。

---

## 21. RS-OVC: Open-Vocabulary Counting for Remote-Sensing Data

**arXiv ID:** 2604.08704 | [PDF](https://arxiv.org/pdf/2604.08704v1)

**作者:** Tamir Shor `[一作]` (Technion – Israel Institute of Technology), Genady Beryozkin `[通讯]` (Google Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了RS-OVC，一个面向遥感图像的开放词汇计数模型，能够在未见过的目标类别上进行计数。

**💡 创新点**

创新点包括首次在遥感领域实现开放词汇计数，采用跨域特征融合（Swin + DINOv3）并在多数据集上训练以提升类别多样性。

**🔧 技术方法**

技术上基于CountGD（GroundingDINO）框架，使用Swin Transformer图像编码器、BERT文本编码器、特征增强器、交叉注意力及冻结预训练编码器进行对比学习。

**📊 数据集**

使用了NWPU‑MOC、FAIR1M、DIOR、DOTA等遥感计数与检测数据集，并在RSOC建筑类别上进行评估。

**📈 对比分析**

与CountGD原版、CountGD在RS数据上微调、检测式基线LAE以及RS-OVC（仅RS编码器）进行对比，结果在平均MAE/RMSE上均优于基线，尤其在高密度低分辨率场景中表现突出。

**⚠️ 局限性**

局限性包括在大尺度/低密度目标上性能略逊于检测基线；对阈值调参依赖验证集；受限于CountGD架构和可用遥感数据集的规模。

---

## 22. Act or Escalate? Evaluating Escalation Behavior in Automation with Language Models

**arXiv ID:** 2604.08588 | [PDF](https://arxiv.org/pdf/2604.08588v1)

**作者:** Matthew DosSantos DiSorbo `[一作]` (Harvard Business School), Harang Ju `[通讯]` (Johns Hopkins Carey Business School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM在五类业务（酒店预订、贷款审批、内容审核、内容推荐、道德困境）中升级行为进行实验，量化其何时执行、何时上报。

**💡 创新点**

发现模型具有特定且可观测的升级阈值和误差偏差，且通过提示+思考或链式推理监督微调能实现近最优决策。

**🔧 技术方法**

采用基于成本阈值的决策框架、信号提示、成本框定提示、延伸思考、链式推理监督微调等技术。

**📊 数据集**

使用公开数据集：酒店预订（HotelBookings）、贷款审批（LendingClub）、维基百科毒性评论（Wikipedia Toxicity）、MovieLens电影推荐、道德困境（MoralMachine）。

**📈 对比分析**

通过与最优成本阈值对应的升级正确率比较，提示+思考提升至约79%，链式推理微调几乎达到100%的性能。

**⚠️ 局限性**

仅研究二分类任务、模型样本有限、成本假设为已知常数，未覆盖多步骤或多选行动空间。

---

## 23. Hierarchical Community Detection in Bipartite Networks

**arXiv ID:** 2604.08793 | [PDF](https://arxiv.org/pdf/2604.08793v1)

**作者:** Tania Ghosh `[一作]` (University of Houston), Kevin E. Bassler `[通讯]` (University of Houston)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可调分辨率的双分网络模块度密度方法 Q_bg，并在合成与真实双分网络上验证其能够在保持原始双分结构的同时识别层次社区。

**💡 创新点**

创新点在于将内部连通密度与传统双分模块度相结合，并引入可调参数 χ，既消除了分辨率限制，又实现了直接在双分网络上检出多层次结构。

**🔧 技术方法**

主要技术包括：构造 Q_bg 目标函数、解析分辨率限制行为、数值最大化优化、相位图与阈值分析、以及多分辨率可视化。

**📊 数据集**

使用了：合成层次双分基准网络、Southern Women 社交活动双分网络、以及 83 名哮喘患者与 18 种细胞因子构成的双分网络。

**📈 对比分析**

与传统双分模块度 Q_b 以及其他现有方法比较，Q_bg 在基准网络上能正确恢复所有层次结构，在真实数据上揭示更多细粒度与层次信息，整体性能优于现有方法。

**⚠️ 局限性**

局限包括：需要人工选择分辨率参数 χ，缺乏自动化选择策略；对极端权重仍存在一定敏感性；仅适用于无重叠的双分网络，且在大规模网络上的计算效率尚待进一步验证。

---

## 24. AI Driven Soccer Analysis Using Computer Vision

**arXiv ID:** 2604.08722 | [PDF](https://arxiv.org/pdf/2604.08722v1)

**作者:** Adrian Manchado `[一作]` (Milwaukee School of Engineering), Yiyang Wang `[通讯]` (Milwaukee School of Engineering)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用未标注的足球比赛视频，结合YOLOv5x检测、SAM2分割与跟踪、CNN关键点预测、单应变换和颜色聚类，构建了从原始摄像头视角到二维场地坐标系的完整可视化与统计分析流水线。

**💡 创新点**

创新点在于：①在无标签、原始视频条件下实现完整的玩家检测、跟踪与团队识别；②通过训练的小型CNN进行场地关键点检测，实现单应变换到真实尺寸坐标；③使用聚类方式进行团队分类，省去了手工标注与监督分类器的需求；④在整个系统中整合了SAM2的记忆式跟踪和YOLO的高效检测，提升了鲁棒性。

**🔧 技术方法**

采用的技术包括：YOLOv5x（目标检测）、SAM2（分割与跟踪）、自定义CNN（关键点回归与可见性预测）、DLT单应变换、K‑means聚类（团队识别）、数据增强（水平翻转）、贝叶斯超参搜索、以及 Adam + 学习率衰减优化器。

**📊 数据集**

数据集来源：2024赛季MSOE男子足球队10场主场比赛视频；390帧使用SAM2与YOLO标注得到的玩家边框；146帧手工标注的场地关键点（4~12点/帧）。

**📈 对比分析**

对比实验：YOLOv5x在SAM2生成的真值上取得最高F1（0.8451），召回率0.7995，精确率0.8963；关键点CNN在验证集上可见性准确率97.18%，坐标MAE为0.0138（约7.7像素）。整体系统在关键点投影上的平均误差为0.499 m，实际关键点MAE为0.225–0.26 m。

**⚠️ 局限性**

局限性：检测时出现误报（如球童、裁判），光照与阴影导致聚类误判团队；关键点预测误差影响投影精度；缺乏玩家重识别能力，无法处理离开再进入场地的情况；模型过拟合于MSOE主场摄像机与场地，跨场地泛化差；未包含球的检测与跟踪。

---

## 25. Smartwatch-Based Sitting Time Estimation in Real-World Office Settings

**arXiv ID:** 2604.08808 | [PDF](https://arxiv.org/pdf/2604.08808v1)

**作者:** Olivia Zhang `[一作]` (Hockaday School), Zhilin Zhang `[通讯]` (Lumos Alpha)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种基于智能手表IMU的坐姿时间估计方法，并在真实办公环境中收集并分析数据，提出通过旋转向量序列提升坐姿识别；

**💡 创新点**

创新点：①将加速度中的重力分量转换为旋转向量序列，捕获手臂姿态信息；②将旋转向量视为额外的传感器通道，可直接套用大多数现有算法；③首次在非实验室、自然办公场景中验证该表示的有效性；

**🔧 技术方法**

技术手段包括：利用Euler角估算pitch/roll，构造旋转矩阵并求旋转向量；提取模糊熵、统计量、能量等特征；采用CatBoost分类器；并将旋转向量序列与原始加速度/陀螺信号一起输入模型；

**📊 数据集**

数据集：5名办公室职员佩戴手表（非主导手腕）连续佩戴至少5小时，共计34小时加速度/陀螺数据，采样频率100Hz，人工标注坐姿与非坐姿；

**📈 对比分析**

评估方法：80%训练、20%测试，5折交叉验证。与ST‑DT、CNN‑LSTM、VAE、MaskCAE、Conformer等基线比较，改进版均在准确率、召回率和F1分数上有提升，提出方法达到Recall 0.964、Precision 0.978、F1 0.971、Accuracy 0.958；

**⚠️ 局限性**

局限性：样本量有限，仅5人同一办公环境；在剧烈运动时重力分量受扰动导致旋转向量不稳定；缺乏对不同工作场景、手表佩戴位置以及跨文化差异的验证。

---

## 26. HM-Bench: A Comprehensive Benchmark for Multimodal Large Language Models in Hyperspectral Remote Sensing

**arXiv ID:** 2604.08884 | [PDF](https://arxiv.org/pdf/2604.08884v1)

**作者:** Xinyu Zhang `[一作]` (Sun Yat-sen University), Haohuan Fu `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个专门评估多模态大语言模型在高光谱遥感图像理解能力的基准 HM-Bench，并生成了 19,337 个问答对。

**💡 创新点**

创新点在于提供双模态输入（PCA 合成图像与结构化文本报告），系统评估模型在空间‑光谱感知与推理任务中的表现，并揭示视觉输入优于文本输入的趋势。

**🔧 技术方法**

采用 PCA 降维可视化、结构化文本报告生成、规则+LLM 生成问答、以及统一多选题零样本评估框架。

**📊 数据集**

使用 20 个公开高光谱数据集（如 Indian Pines、Salinas、Xiongan、Houston、Martian MARS 等），涵盖 2,178 块图像。

**📈 对比分析**

通过统一多选题的零样本评估，对 18 个 MLLM（含 14 个开源与 4 个闭源）在图像与报告两种输入下进行比较，最高准确率仅 43%（InternVL3.5-14B），显示任务难度大。

**⚠️ 局限性**

局限性包括模型难以处理复杂空间‑光谱推理，文本报告信息量有限，且对高光谱特征的直接感知能力不足。

---

## 27. EXAONE 4.5 Technical Report

**arXiv ID:** 2604.08644 | [PDF](https://arxiv.org/pdf/2604.08644v1)

**作者:** Eunbi Choi `[一作]`, Sangyeon Yoon `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并发布了EXAONE 4.5，一款融合1.2B视觉编码器与32B语言模型的开放权重视觉语言模型（VLM），实现了文本与图像的跨模态推理与长上下文处理。

**💡 创新点**

核心创新包括：1) 采用1.2B参数规模的视觉编码器并配合2D Rotary Positional Embedding；2) 在视觉编码器中引入Grouped Query Attention（GQA）以提升计算效率；3) 通过Multi-Token Prediction（MTP）模块提升解码吞吐量；4) 在监督微调阶段直接嵌入256K上下文扩展；5) 采用多阶段离线优先级优化与多模态强化学习，强化视觉推理与指令遵循能力。

**🔧 技术方法**

技术手段包括：大规模自监督预训练（两阶段），多模态融合架构，GQA注意力机制，2D RoPE，MTP解码器，Context Parallelism并行技术，离线偏好优化（DPO、GROUPER），GRPO强化学习。

**📊 数据集**

使用的数据集涵盖：Korean-English双语图文对（synthetic captioning），开放与内部图文交错集，OCR与文档理解集（包含英文、韩文字符、单词、文档层级），视觉定位与计数数据，STEM与推理集（长Chain-of-Thought）、专门的韩语文化与学术数据，数十个评测基准（MMMU、MMMU-Pro、MedXpertQA-MM、MATH-Vision、MathVista、We-Math、LogicVista、BabyVision、AI2D、ChartQAPro、CharXiv、OCRBench v2、OmniDocBench、MMStar、BLINK、HallusionBench、KMMMU、K-Viscuit、KRETA、AIME 2026、GPQA-Diamond、LiveCodeBench v6、MMLU-Pro、τ^2-Bench、IFBench、IFEval、AA-LCR、KMMLU-Pro、KoBALT、MMMLU、WMT24++等。

**📈 对比分析**

与现有大型VLM（如Qwen3-VL-235B、GPT-5 mini等）对比，EXAONE 4.5在STEM/逻辑推理、文档理解、通用视觉任务以及语言推理、编码、工具使用与指令遵循等方面表现出色，常常在同类或更大参数模型中获得更高分数，显示出优越的跨模态推理和长上下文处理能力。

**⚠️ 局限性**

主要局限包括：可能生成不当、偏见或不准确信息；对最新事件缺乏更新；依赖训练数据统计，易产生语义或语法错误；在极端噪声或多样化视觉场景下性能可能下降。

---

## 28. VOLTA: The Surprising Ineffectiveness of Auxiliary Losses for Calibrated Deep Learning

**arXiv ID:** 2604.08639 | [PDF](https://arxiv.org/pdf/2604.08639v1)

**作者:** Rahul D Ray `[一作]` (BITS Pilani), Utkarsh Srivastava `[通讯]` (BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种简化的 VOLTA 结构，用深度归一化编码器、可学习原型和温度标度实现确定性不确定性量化

**💡 创新点**

核心创新在于只保留三大关键组件，去掉重构、对比、冗余损失，证明深度归一化与温度标度即可获得优秀的校准和 OOD 检测

**🔧 技术方法**

使用深度 CNN / MLP 编码器、原型分类、交叉熵训练、可学习温度参数、后置温度标度以及在不同实验中对比 10 种 UQ 基线（MC Dropout、SWAG、Posterior、Temperature Scaling、Energy‑based、Mahalanobis、Hyperbolic、ENN、Taylor‑Sensus、Split Conformal）

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、Tiny ImageNet 预提特征以及 OOD 数据集（SVHN、Uniform Noise、CIFAR‑10‑C）上进行评测

**📈 对比分析**

与十个基线比较时，VOLTA 在校准（ECE≤0.010）和 OOD 检测（AUROC≈0.80）上均达到或超过多数方法，同时保持竞争力的准确率和低计算成本；统计检验显示差异显著

**⚠️ 局限性**

实验仅覆盖 32×32 图像与预提特征，对更大规模、高分辨率数据、远距 OOD、以及对不同任务的泛化仍未验证

---

## 29. Drift and selection in LLM text ecosystems

**arXiv ID:** 2604.08554 | [PDF](https://arxiv.org/pdf/2604.08554v1)

**作者:** Søren Riis `[一作]` (Queen Mary University of London), Søren Riis `[通讯]` (Queen Mary University of London)

**通讯引用:** 1095 | [OpenAlex ID](https://openalex.org/A5005137861)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并构建了一个可精确求解的递归公共文本生成框架，使用可变阶n-gram代理来模拟人机混合生成、筛选与发布的闭环过程。

**💡 创新点**

创新点在于将“漂移”(Wright–Fisher式随机采样导致的稀有形式消失)与“选择”(发布、排名、验证等筛选机制)分离，并给出了在无平滑的n-gram环境下漂移的精确固定点多面体与规范性选择下保持深层结构的KL散度上限，首次在理论上阐明递归出版如何压缩或保留文本深度。

**🔧 技术方法**

使用可变阶n-gram模型、Wright–Fisher随机过程、de Bruijn图循环极点、KL散度、以及对lookahead/软评估器的数学分析。

**📊 数据集**

实验数据集采用公共领域文学文本，如《阿瑟·柯南·道尔》小说、简·奥斯汀与查尔斯·达尔文作品。

**📈 对比分析**

通过对比描述性（仅复制生成文本）与规范性（基于lookahead评估的筛选）两种递归出版策略，观察KL散度收敛、稀有n-gram消失速度、文本多样性等指标，结果表明规范性筛选能保持更深的n-gram结构且KL散度保持在理论上限内；实验规模有限但足以验证理论。

**⚠️ 局限性**

局限性包括仅考虑无平滑的n-gram模型，未涵盖Transformer等神经架构的细节；实验规模相对较小，未检验在更大、真实工业数据上的泛化；并假设所有采样与发布操作均完全可观测和可重现。

---

## 30. Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics

**arXiv ID:** 2604.08764 | [PDF](https://arxiv.org/pdf/2604.08764v1)

**作者:** Raphael Bernas `[一作]` (MICS, CentraleSupelec, Université Paris-Saclay), Céline Hudelot `[通讯]` (MICS, CentraleSupelec, Université Paris-Saclay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过引入流形与微分几何视角，理论分析并实证验证频率偏倚如何导致Transformer语言模型嵌入的低维、非等距（anisotropic）几何结构，并将此几何偏差与梯度更新过程中的切向（tangent）偏置关联起来；

**💡 创新点**

创新点在于：①将频率偏倚解释为对局部流形采样方差的收缩；②用微分几何推导外在采样如何抑制曲率可见性；③证明梯度更新偏好切向方向，形成自强化机制；④结合机制可解释性方法，在训练期间提取概念子空间并与真实梯度对齐，从而解释全局 anisotropy；

**🔧 技术方法**

技术手段包括：流形假设与切空间近似、二阶曲率与第二基本形式分析、梯度协方差分解、低秩PCA提取概念子空间、IsoScore* 与能量比率评估、统计显著性检验；

**📊 数据集**

使用多种英语与法语预训练语料库（merged corpus）构建的模型：EuroBERT、CamemBERT、OLMo、Pythia、Gaperon、SmolLM 等，覆盖编码器和解码器两大架构；

**📈 对比分析**

与匹配维度的正交正态子空间对比，发现切向子空间在早期训练阶段能捕捉到显著高的梯度能量（能量比>1，p值极低），并且移除切向子空间后可显著提高梯度协方差的等距性（ΔIsoScore*大幅正值），显示切向方向对全局 anisotropy 的主要贡献；

**⚠️ 局限性**

局限性在于：分析仅局限于局部流形和早期训练阶段；假设模型嵌入近似可微流形，忽略全局多分支或层级结构；未完全解释后期训练中损失驱动的细粒度方向选择；对低频词的几何影响研究不充分；

---

## 31. Choose, Don't Label: Multiple-Choice Query Synthesis for Program Disambiguation

**arXiv ID:** 2604.08792 | [PDF](https://arxiv.org/pdf/2604.08792v1)

**作者:** Celeste Barnaby `[一作]` (University of Texas at Austin), Isil Dillig `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于多项选择（MC）查询的主动学习框架，用于交互式程序合成，用户只需在高层行为描述中做出选择即可消除候选程序的不确定性。

**💡 创新点**

创新点在于将程序意图抽象为预条件/后条件的Hoare三元组，并将其映射为可解释的MC问题；利用SMT+OMT实现最优查询合成，并提供形式化的正确性保证，显著提升了传统基于例子的主动学习方法的可解释性和准确性。

**🔧 技术方法**

主要技术包括：Z3 SMT求解器与优化（OMT）用于查询生成；聚类与分隔符构造实现多选答案；LLM（OpenAI GPT‑4）用于将逻辑查询转换为自然语言；以及基于模型的近似采样策略以保证可扩展性。

**📊 数据集**

实验使用四个领域的157个基准任务：表格变换（80题）、JSON变换（15题）、批量图像编辑（37题）和图像搜索（25题），涵盖符号和神经符号混合数据。

**📈 对比分析**

在准确率上，Socrates 在所有任务中 100% 成功率，远超 SampleSy、LearnSy（77–85%）和 SmartLabel（85–86%）。在效率方面，Socrates 的交互轮次和查询生成时间与基线相当或更优，用户回答准确率提升约 38% 且平均响应时间无显著增加。

**⚠️ 局限性**

局限性包括：需手工构造预/后条件谓词集合，依赖有限展开（bounded unrolling）导致的语义不完全；LLM 翻译可能引入歧义；对高度复杂或自定义领域的泛化受限；以及在某些神经符号任务中仍受模型不确定性的影响。

---

## 32. Joint Interference Detection and Identification via Adversarial Multi-task Learning

**arXiv ID:** 2604.08607 | [PDF](https://arxiv.org/pdf/2604.08607v1)

**作者:** H. Xu `[一作]` (National University of Defense Technology), S. Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 37034 | [OpenAlex ID](https://openalex.org/A5100740061)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一种理论驱动的对抗式多任务学习框架 AMTIDIN，用于联合干扰检测、调制识别与干扰识别。

**💡 创新点**

首先给出 MT‑L 加权期望损失的理论上界，并通过 Wasserstein 距离与可学习任务关联系数量化任务相似度；其次在此理论指导下引入对抗特征对齐与可学习任务权重的网络，在低数据、短信号长度与低 SNR 环境下显著提升鲁棒性。

**🔧 技术方法**

采用对抗训练（Wasserstein‑1 距离、Spectral Normalization、Gradient Reversal Layer）、可学习任务关联系数、共享特征提取器、残差子网络与 Softmax 分类头，全部使用 PyTorch 及 Adam 优化器实现。

**📊 数据集**

使用基于 GNU Radio 与 Python 混合生成的合成数据集，包含多种调制、六类干扰（CWI、NNI、MTI、LFMI、DMI、AMI）以及 AWGN、Rayleigh、Rician 通道，数据量可自由调节。

**📈 对比分析**

与单任务 STL 基线以及三种主流 MT‑L 基线（Vanilla、MMoE、NonAdv）在相同信号长度、样本量约束下进行对比；AMTIDIN 在低 SNR（-13 dB、0 dB、-5 dB）、小样本与短信号条件下分别比 STL 提升约 2.8%–8.3%，并在所有任务上优于其他 MT‑L 模型，表现出最佳准确率与鲁棒性。

**⚠️ 局限性**

仅在合成数据上验证，未在真实雷达或通信现场数据上测试；对超参数如任务权重与 ρ 依赖较大；在极高 SNR 或长信号长度时提升有限；未来需扩展至开放集识别与边缘实时部署。

---

## 33. Script Collapse in Multilingual ASR: Defining and Measuring Script Fidelity Rate

**arXiv ID:** 2604.08786 | [PDF](https://arxiv.org/pdf/2604.08786v1)

**作者:** Hanif Rahman `[一作]` `[通讯]`, Hanif Rahman

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文定义了脚本保真率（SFR），并首次系统性地测量了六种语言（普什图语、乌尔都语、印地语、孟加拉语、马拉雅拉姆语、索马里语）在九种自动语音识别（ASR）模型下的脚本崩溃现象。

**💡 创新点**

创新点在于提出了脚本保真率（SFR）作为一种无参考的ASR评估指标，能够在没有参考转录的情况下计算，并且首次系统性地测量了不同模型和语言的脚本崩溃情况。

**🔧 技术方法**

使用了Unicode块分析作为评估技术，SFR是基于字符的Unicode块成员资格进行计算的。

**📊 数据集**

使用了FLEURS测试集，该数据集包含六种语言的语音数据，确保每种语言都有足够的样本量（至少250个发音）。

**📈 对比分析**

与传统的字错误率（WER）比较，SFR能够识别出WER无法检测的脚本崩溃现象。结果显示，53个模型-语言对中，有18个（34%）表现出脚本崩溃（SFR < 10%），而MMS-1B和SeamlessM4T-v2在所有评估的语言中SFR均高于99%。

**⚠️ 局限性**

SFR的局限性在于它无法区分高质量的目标脚本文本和随机的目标脚本字符输出。此外，SFR并不是WER的替代品，而是一个必要的前提检查。

---

## 34. Post-Quantum Cryptography-Based Bidirectional Authentication Key Exchange Protocol and Industry Applications: A Case Study of Instant Messaging

**arXiv ID:** 2604.08612 | [PDF](https://arxiv.org/pdf/2604.08612v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 35. Parameterized Complexity Of Representing Models Of MSO Formulas

**arXiv ID:** 2604.08707 | [PDF](https://arxiv.org/pdf/2604.08707v1)

**作者:** Petr Kučera `[一作]` (Charles University), Petr Martinek `[通讯]` (Charles University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文通过动态规划与状态空间构造，给出了在树宽/路径宽参数化下，MSO₂ 公式模型的可判定及其在 SDD 与 OBDD 中的线性大小表示。

**💡 创新点**

创新点在于首次证明：在树宽下可用 SDD 以参数化线性大小表示模型；在路径宽下可用 OBDD 以参数化线性大小表示模型；并给出了 OBDD 无法按树宽参数化的下界。

**🔧 技术方法**

使用了树分解（nice 树分解）与良好顶点着色的动态规划技术，并将状态转换表转化为 SDD/OBDD 结构；利用了 MSO₂ 语义的逐步评估与一致性检查。

**📊 数据集**

论文主要为理论研究，并未使用具体实验数据集，而是针对任意给定图与 MSO₂ 公式给出构造与上界/下界证明。

**📈 对比分析**

方法通过参数化复杂度理论进行比较，证明在树宽/路径宽参数下所构造的 SDD/OBDD 大小为 O(f(k)·n)，即参数化线性；同时给出 OBDD 下界，说明其在树宽下不可压缩。

**⚠️ 局限性**

限制在于仅针对树宽/路径宽参数化；对更一般图结构（无界树宽）的效果未知；以及实际实现的复杂度与常数因子未给出，可能导致实际效率不如理论预期。

---

## 36. GAN-Enhanced Deep Reinforcement Learning for Semantic-Aware Resource Allocation in 6G Network Slicing

**arXiv ID:** 2604.08576 | [PDF](https://arxiv.org/pdf/2604.08576v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 37. Medical Reasoning with Large Language Models: A Survey and MR-Bench

**arXiv ID:** 2604.08559 | [PDF](https://arxiv.org/pdf/2604.08559v1)

**作者:** Xiaohan Ren `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8339 | [OpenAlex ID](https://openalex.org/A5051925942)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文综述了医学推理与大型语言模型（LLM）的研究进展，系统归纳了训练型与训练‑free 的七大技术路径，并在统一实验框架下对现有模型进行跨基准评估；同时构建了基于真实医院病历的临床推理基准MR‑Bench，揭示了模型在真实临床场景下的显著性能缺口。

**💡 创新点**

创新点在于：①将医学推理抽象为假设（abduction）、演绎（deduction）和归纳（induction）的迭代过程，提供了理论指导；②首次提出MR‑Bench这一真实数据驱动、时间泛化且多任务的临床推理基准；③通过统一实验评估，系统揭示了领域适配与基础模型规模对医学推理性能的相对影响，并展示了现有模型在真实临床推理上的局限。

**🔧 技术方法**

技术主要包括：训练型方法（继续预训练、监督微调、强化学习）与训练‑free 方法（prompt engineering、检索增强生成、代理推理管线），以及多模型对比实验、基准任务拆分与统一解码/评估策略。

**📊 数据集**

使用的数据集包括：标准问答基准（MedQA、PubMedQA、MedMCQA、MMLU‑Pro、GPQA、ReDis‑QA、MedXpertQA、JMED）以及构建的MR‑Bench，后者基于MIMIC‑IV的真实ICU病历，涵盖药物推断与程序选择两类多选任务。

**📈 对比分析**

通过统一评估，域适配的医学LLM在标准基准上相对提升约10–20%，但在MR‑Bench上往往表现退步；最先进的GPT‑5在MR‑Bench仅达约60%准确率，说明虽然基础模型规模提升带来一定优势，但仍无法满足临床推理的安全与可靠需求。

**⚠️ 局限性**

局限性包括：现有问答基准缺乏临床真实性与完整信息，评估往往依赖模型判别导致结果不稳定；训练数据主要来自考试或合成案例，难以覆盖真实临床情境；现有模型缺乏可靠的推理链、主动信息获取与安全性保障，无法直接用于临床部署。

---

## 38. From Selection to Scheduling: Federated Geometry-Aware Correction Makes Exemplar Replay Work Better under Continual Dynamic Heterogeneity

**arXiv ID:** 2604.08617 | [PDF](https://arxiv.org/pdf/2604.08617v1)

**作者:** Zhuang Qi `[一作]` (Shandong University), Xiangxu Meng `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Federated gEometry-Aware correcTion的方法，以缓解联邦持续学习中的灾难性遗忘问题，特别是在动态异构环境下的表现崩溃。

**💡 创新点**

创新点在于通过几何结构对齐模块和基于能量的几何校正模块，增强了不同客户端之间的特征一致性，并改善了对少数类的敏感性。

**🔧 技术方法**

使用了几何结构对齐和能量基几何校正技术，结合了结构知识蒸馏和去偏见策略。

**📊 数据集**

在CIFAR10、CIFAR100和TinyImageNet-Subset等三个数据集上进行了实验，评估了不同异构性水平下的性能。

**📈 对比分析**

与七种最先进的方法进行了比较，结果显示该方法在所有基准测试中均优于其他方法，尤其在Top-1准确率上表现出一致的提升。

**⚠️ 局限性**

限制在于该方法在处理极端不平衡数据时可能仍然面临挑战，尤其是在特征偏移较大的情况下。

---

## 39. eBandit: Kernel-Driven Reinforcement Learning for Adaptive Video Streaming

**arXiv ID:** 2604.08791 | [PDF](https://arxiv.org/pdf/2604.08791v1)

**作者:** Mahdi Alizadeh `[一作]` `[通讯]` (University of Southern California), Mahdi Alizadeh (University of Southern California)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了一个基于eBPF的内核空间多臂赌博机(MAB)框架，实时监控TCP状态并在内核中动态选择最优ABR策略，消除了用户空间与内核之间的观测差距；

**💡 创新点**

创新点在于将网络监控与ABR决策迁移到Linux内核，通过eBPF实现零上下文切换、即时获取RTT和交付率，并采用轻量级epsilon‑greedy MAB在内核中进行在线学习；

**🔧 技术方法**

使用eBPF、Linux kernel hooks、整数奖励计算、per‑RTT回调、共享BPF map、epsilon‑greedy MAB、交叉乘法比较等技术；

**📊 数据集**

采用人工构造的三阶段对抗性带宽轨迹和Norway HSDPA移动网络真实会话（共42个），并使用一小部分留存轨迹做warm‑start；

**📈 对比分析**

与三种静态ABR基线（Throughput、Buffer‑Based、Hybrid BBA+）在Pensieve的log‑utility QoE模型下比较；在合成轨迹上eBandit比最佳静态策略提升7.2%，在真实轨迹上warm‑start版本平均QoE 1.241，显著优于其他策略；

**⚠️ 局限性**

局限在于仅使用简单的epsilon‑greedy MAB，缺乏上下文特征，shock detection目前仅在用户空间模拟，无法处理更复杂多路径或多用户场景；部署需要根权限，且未在长期生产环境中充分评估。

---

## 40. Towards Generalizable Representations of Mathematical Strategies

**arXiv ID:** 2604.08693 | [PDF](https://arxiv.org/pdf/2604.08693v1)

**作者:** Siddhartha Pradhan `[一作]` (Worcester Polytechnic Institute), Erin Ottmar `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过预训练的数学编码器和SimCSE对学生完整代数解题路径进行向量化，生成问题不变的序列嵌入，并用其评估学生的解决策略与创造力。

**💡 创新点**

创新点在于：①使用转换差分（transition）嵌入而非状态嵌入，突出解题步骤而非问题表面；②采用对比学习得到平台无关、跨问题的完整路径表示；③基于嵌入计算策略唯一性、多样性和符合度，作为可量化的创造力指标。

**🔧 技术方法**

技术包括：预训练数学BERT模型（MathBERT、MathBERT‑mamut 等）、SimCSE 对比学习、Transformer 编码器、t‑SNE 可视化、动作多标签预测、效率分类、序列重建探测，以及回归分析。

**📊 数据集**

使用的数据库是 2020‑2021 年 COVID‑19 期间在美国某郊区 11 所中学开展的 FH2T 随机对照试验数据，涵盖 4,092 名七年级学生的游戏日志、动作序列和测评结果。

**📈 对比分析**

与基线平均池化的预训练模型对比；通过三项探测任务（动作内容预测、效率分类、序列重建）评估。最佳 SimCSE 模型在动作预测微 F1≈86%、宏 F1≈21.8%、序列重建困惑度≈2.06、准确率≈77% 时表现最优，显著优于基线。

**⚠️ 局限性**

局限性包括：①仅使用有限的预训练模型和简单的差分转换，未探索更复杂的转换函数；②缺乏更大、跨平台的多样化数据；③评估主要基于探测任务，缺少人工标注的构念效度检验；④模型在效率分类任务上表现不如基线，说明对比学习更关注步骤而非效率。

---

## 41. Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition

**arXiv ID:** 2604.08708 | [PDF](https://arxiv.org/pdf/2604.08708v1)

**作者:** Tiejin Chen `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MATU（Multi-Agent Tensor Uncertainty）框架，用于量化基于 LLM 的多智能体系统的不确定性。该框架将每个代理在多步推理过程中的每一步嵌入为向量，拼接成矩阵，再将多次运行、不同代理的嵌入矩阵组成 ragged tensor，使用 PARAFAC2 分解得到低秩近似，重构误差作为系统层面不确定度。

**💡 创新点**

创新点包括：①全程嵌入多步推理轨迹，捕获细粒度的不确定性；②构造多运行、多代理的 ragged tensor，直接编码通信路径与拓扑差异；③利用 PARAFAC2 对 ragged tensor 进行分解，既能处理长度可变的数据，又能在不同通信拓扑下保持一致的低秩结构；④统一衡量系统层面的不确定性，可在静态、动态与工具集成场景下使用。

**🔧 技术方法**

主要技术：文本嵌入（Qwen3-Embedding-0.6B）、ragged tensor 组装、PARAFAC2（CP-2）张量分解、重构误差聚合得到不确定度。对比方法采用 P(true)、Eigv(Agr)-final/whole、SAUP（单轨迹和多轨迹）。

**📊 数据集**

实验数据集：MATH、MoreHopQA、MMLU、HumanEval。使用多代理系统框架 Camel、AutoGen（静态拓扑）和 AnyMac（动态拓扑），并在多种 LLM 后端（GPT‑4o、Qwen2.5‑7B、Llama3.1‑8B）上进行评测。

**📈 对比分析**

通过 AUROC 与 AUARC 评价不确定度与答案正确率的相关性。MATU 在所有三种系统设计（静态、动态、工具集成）下均显著优于基线，提升幅度约 5%–15%（具体数值见表格）。在 OOD 任务与模型选择等下游任务中，MATU 亦表现出更好的区分度与实用性。

**⚠️ 局限性**

局限性：①需要多次采样（通常 10 次）才能构成 ragged tensor，推理成本随采样次数线性增长；②依赖文本嵌入模型的质量，通用嵌入在专业域可能不足；③在极端专业化任务中，若嵌入无法捕获细微语义差异，矩阵表示与分解的可靠性可能下降。

---

## 42. Model Space Reasoning as Search in Feedback Space for Planning Domain Generation

**arXiv ID:** 2604.08712 | [PDF](https://arxiv.org/pdf/2604.08712v1)

**作者:** James Oswald `[一作]` (Rensselaer Polytechnic Institute), Shirin Sohrabi `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个基于大型语言模型和符号反馈的迭代生成PDDL领域的框架；

**💡 创新点**

将地标与计划验证等符号反馈结合，并使用启发式搜索在反馈空间上优化领域质量，系统评估多模型、多反馈组合的效果；

**🔧 技术方法**

使用大型语言模型（DeepSeek‑chat、gpt‑5‑nano、gpt‑5‑mini）、PDDL语法验证、VAL计划验证、计划求解器K*、启发式搜索以及HDE评估指标；

**📊 数据集**

采用多种typed STRIPS PDDL领域（blocks、bloxorz、checkers、flow、hiking、pacman‑63、pacman‑72、miconic等），每个领域配有问题生成器，生成可解问题集与评估问题集；

**📈 对比分析**

与无反馈基线相比，所有反馈策略平均提升HDE；最优组合LVS在gpt‑5‑mini上每个领域至少一次得到100% HDE；不同模型对反馈类型的响应存在差异，系统搜索通常优于随机但有例外；

**⚠️ 局限性**

仍受LLM训练数据覆盖范围、反馈类型的限制；搜索分支过大时需改进策略；实验仅在离线评估，缺少用户体验研究；对复杂或真实场景的泛化尚未验证。

---

## 43. MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation

**arXiv ID:** 2604.08782 | [PDF](https://arxiv.org/pdf/2604.08782v1)

**作者:** Jyotika Singh `[一作]` (Oracle), Dan Roth `[通讯]` (Oracle)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MT-OSC框架，实现对多轮对话历史的“一次性”序列化压缩，保持关键信息同时显著减少上下文长度；

**💡 创新点**

采用无模型改动、无细调的背景压缩机制，配合少量示例驱动的Condense agent和基于重叠度的Decider，实现高效且任务无关的历史压缩；

**🔧 技术方法**

基于LLM的few-shot Condenser、轻量级Decider、滑动窗口一阶序列化压缩算法；

**📊 数据集**

在10个多轮基准（包括GSM8K、BFCL、HumanEval、Spider、ToTTo、SoH、MT‑Eval等）上进行评测，并对噪声扰动进行鲁棒性实验；

**📈 对比分析**

与传统完整拼接（MT-baseline）对比，MT‑OSC在保持或提升多轮性能的同时平均减少约30.9%（最高72%）聊天历史token，且在13种顶级LLM上均显著提升准确率；

**⚠️ 局限性**

实验使用的数据主要为单主题、少量轮次（≤12）对话，缺乏复杂多主题与代理行为，未来需扩展更长、更复杂的对话数据集以验证泛化能力。

---

## 44. One Interface, Many Robots: Unified Real-Time Low-Level Motion Planning for Collaborative Arms

**arXiv ID:** 2604.08787 | [PDF](https://arxiv.org/pdf/2604.08787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 45. QCFuse: Query-Centric Cache Fusion for Efficient RAG Inference

**arXiv ID:** 2604.08585 | [PDF](https://arxiv.org/pdf/2604.08585v1)

**作者:** Jianxin Yan `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35514 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

QCFuse 是一种面向查询的 KV 缓存融合系统，旨在加速配备检索增强生成（RAG）的语言模型的推理过程，通过选择性重算与查询最相关的 token 来显著降低计算成本并提升响应速度。

**💡 创新点**

其创新点包括：① 以用户查询为全局关注点，使用轻量化的语义锚点（anchor）对查询进行上下文增强，避免传统局部或无上下文的 token 选择方法；② 在单一中间层进行查询注意力分析，既避免跨层依赖导致的流水线阻塞，又能获得比最后一层更完整的语义信息；③ 结合位置感知稀疏注意力内核（Triton实现），实现高效离散 token 重算。

**🔧 技术方法**

技术要点包括：KV 缓存融合、anchor 提取（基于 key‑norm）、查询探测（query probing）、关键层注意力分析、Triton 构建的稀疏注意力核、SGLang 框架改造、RadixCache 哈希索引、基于哈希的块级 KV 缓存等。

**📊 数据集**

实验使用了 Musique、2WikiMQA、HotpotQA 三个问答数据集，并在 Llama3.1‑8B、Qwen3‑8B、Mistral‑v0.3‑7B 三种大模型上进行评测。

**📈 对比分析**

在与 CacheBlend、EPIC、KVShare、ProphetKV、FusionRAG、QCAll、QCLast 等基线以及完整计算/重用方案的对比中，QCFuse 实现了约 2 倍的推理速度提升，较现有缓存融合方法降低约 40% 的延迟，同时保持甚至略优于基线的 ROUGE‑L 分数（在 HotpotQA 上提升 0.8 分）。

**⚠️ 局限性**

局限性包括：① 需要离线预计算并存储 KV 缓存至 SSD，增加存储与 I/O 负担；② 关键层选择的固定策略可能对不同模型或不同查询场景不够鲁棒；③ 依赖 SGLang 框架及其改造，迁移到其他实现时需要额外工作；④ 对极端动态检索或高并发环境的适配仍需进一步验证。

---

## 46. Structural Evaluation Metrics for SVG Generation via Leave-One-Out Analysis

**arXiv ID:** 2604.08809 | [PDF](https://arxiv.org/pdf/2604.08809v1)

**作者:** Haonan Zhu `[一作]`, Purvanshi Mehta `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于留一法（LOO）的元素级评估框架，能够为SVG生成模型提供每个图形元素的质量得分、概念归因以及四种结构化指标（纯度、覆盖率、紧凑度、局部性），并证明其对编辑可操作性的预测能力

**💡 创新点**

创新点在于将传统的留一估计方法迁移到SVG结构评估，通过对每个元素的缺失渲染来量化其对整体视觉质量的贡献，并由此推导出概念归因矩阵和四个结构化指标，填补了仅基于图像相似度的评估空白

**🔧 技术方法**

采用了CLIP ViT-B/32进行图像相似度度量、CLIPSeg和SAM3进行概念定位、Qwen3-VL-32B进行概念抽取，以及基于渲染的像素差异映射实现元素影响量化

**📊 数据集**

使用了300张手工标注的SVG验证集（分为简单、中等、复杂三层级），并在5种不同SVG生成模型（Claude 4.5-Opus、GPT-5.2、Gemini 3-flash-preview、Qwen3-Coder-30B-A3B-Instruct、VTracer）上进行评估

**📈 对比分析**

通过与传统的图像相似度评估、基于概念的评价以及人工标注的对比，LOO方法在错误检测（F1≥0.87）和结构化指标（如纯度、覆盖率、紧凑度、局部性）上均表现优异，且纯度与编辑精度之间的相关性显著（r≥0.29）

**⚠️ 局限性**

局限性包括需要对每个SVG进行N+1次渲染，计算成本高；CLIP相似度仅为视觉质量的近似代理；概念抽取与定位过程存在不确定性；以及编辑精度评估与概念掩模共享同一管线，可能导致验证偏倚

---

## 47. EvoLen: Evolution-Guided Tokenization for DNA Language Model

**arXiv ID:** 2604.08698 | [PDF](https://arxiv.org/pdf/2604.08698v1)

**作者:** Nan Huang `[一作]` (University of California, San Diego), Jingbo Shang `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种利用进化信息的DNA分词方法EvoLen，改善了DNA语言模型的表征。

**💡 创新点**

创新点在于将phyloP进化约束分层并融合长度感知解码，实现更保留功能基序的分词。

**🔧 技术方法**

使用了进化分层、BPE分词、词表合并、长度加权动态规划等技术。

**📊 数据集**

数据集包括hg38人类基因组、JASPAR 2024 TF motif、各种基因组功能注释以及跨物种任务（人-鼠）。

**📈 对比分析**

与标准BPE、DNABERT2、NT、Grover等对比，EvoLen在多项基准（GUE、GBM、NT、cCRE、snATAC-seq）上平均提升约8‑10%，在调控和跨物种任务上表现最佳。

**⚠️ 局限性**

局限性：对非哺乳动物任务收益有限，训练步数不完全匹配，分词偏向进化范围，需进一步结合更多生物先验。

---

## 48. IKKA: Inversion Classification via Critical Anomalies for Robust Visual Servoing

**arXiv ID:** 2604.08754 | [PDF](https://arxiv.org/pdf/2604.08754v1)

**作者:** Darya Pavlenko `[一作]` `[通讯]`, Darya Pavlenko

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发 IKKA 框架用于嵌入式视觉伺服，利用拓扑异常加权提升鲁棒性。

**💡 创新点**

结合局部极值、边界正交性和多尺度持久性三重拓扑指标，形成乘法异常权重，区别于传统异常剔除。

**🔧 技术方法**

计算局部极值 E、梯度正交性 T、子层集持久性 M（Vietoris‑Rips + H1 持久图），并在 Raspberry Pi 4 CPU 上实现实时 IBVS。

**📊 数据集**

在室内 3×2 m 实验环境中进行 230 次可复现的运行，包含 50 条屏幕驱动基准、150 条闭环场景、30 条遮挡/障碍条件。

**📈 对比分析**

与 HSV+各类滤波器（MOSSE、KCF、CSRT、Hybrid）对比，IKKA 在压力条件下将 P95 |e_x| 从 0.124 降至 0.094（-24%），异常跑率从 4/30 降至 1/30，帧率提升至 24.8 Hz。

**⚠️ 局限性**

仅在单一嵌入式平台与室内场景评估，E/T/M 为经验替代品，二分类需改写，权重参数未学习。

---

## 49. Optimal Multi-bit Generative Watermarking Schemes Under Worst-Case False-Alarm Constraints

**arXiv ID:** 2604.08759 | [PDF](https://arxiv.org/pdf/2604.08759v1)

**作者:** Yu-Shin Huang `[一作]`, Krishna Narayanan `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在最坏情况下的误报约束下，针对大型语言模型的多位生成水印技术。通过分析现有方案的不足，提出了两种新的编码-解码构造，能够达到之前建立的下界，从而完全表征最优的多位水印性能。

**💡 创新点**

创新点在于提出了两种新的编码-解码构造，分别采用分解方法和伪令牌方法，能够实现最优的多位水印性能，并且通过线性规划形式化了水印设计问题。

**🔧 技术方法**

使用了线性规划技术来形式化水印设计问题，并提出了两种新的构造方法：一种是基于分解的方法，另一种是伪令牌的方法。

**📊 数据集**

使用了多位生成水印的理论框架，特别是在有限令牌的情况下，分析了最大可实现的信息率和最优的漏检性能。

**📈 对比分析**

与He等人提出的方案进行了比较，指出其构造存在缺陷，无法达到最优性能。本文的两种新构造在性能上优于现有方案，能够实现最优的漏检概率。

**⚠️ 局限性**

限制在于现有的生成概率计算复杂度较高，尤其是在实际应用中，单个令牌生成方法相比之下更具挑战性。

---

## 50. RansomTrack: A Hybrid Behavioral Analysis Framework for Ransomware Detection

**arXiv ID:** 2604.08739 | [PDF](https://arxiv.org/pdf/2604.08739v1)

**作者:** Busra Caliskan `[一作]`, A. Halim Zaim `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出RansomTrack框架，结合Radare2静态指令提取与Frida动态API/内存监控，实现低延迟的勒索软件检测。

**💡 创新点**

创新点在于构建高家族比样本比例的混合特征数据集，集成SHAP可解释性，模块化的用户空间工具和实现子10秒的检测延迟。

**🔧 技术方法**

使用Radare2、Frida、机器学习（XGBoost、Soft Voting、RF等）以及SHAP进行特征解释和特征选择。

**📊 数据集**

采用公开的RansomTrack数据集，共2410个32位PE文件，包含165个勒索软件家族及对应的1,205个正常软件样本。

**📈 对比分析**

通过准确率、召回率、F1、ROC‑AUC和运行时评估，Soft Voting达96%准确率、0.99 ROC‑AUC且平均9.1s检测，XGBoost同样达到96%但推理仅0.05s。

**⚠️ 局限性**

局限性包括仅针对32位PE、依赖用户空间工具可能被沙箱/延迟执行的勒索软件规避、缺乏多窗口或事件触发的动态追踪以及未覆盖更高级的逃避技术。

---

## 51. Distributionally Robust Token Optimization in RLHF

**arXiv ID:** 2604.08577 | [PDF](https://arxiv.org/pdf/2604.08577v1)

**作者:** Yeping Jin `[一作]` (Boston University), Ioannis Ch. Paschalidis `[通讯]` (Boston University)

**通讯引用:** 6617 | [OpenAlex ID](https://openalex.org/A5075696701)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将分布鲁棒优化（DRO）应用于令牌级RLHF的框架（DRTO），通过在微批量轨迹上构建f-散度不确定性集合，获得理论上的稳健性并提升模型在分布偏移下的推理表现。

**💡 创新点**

创新点在于：①首次在令牌级RLHF中引入分布鲁棒优化；②使用KL和Pearson χ²散度得到可解析的对偶形式，形成自适应加权或加差项的损失；③结合令牌级奖励形状与鲁棒更新，兼顾细粒度信号与全局稳健。

**🔧 技术方法**

技术核心包括：Reinforced Token Optimization (RTO) 的令牌奖励，Proximal Policy Optimization (PPO) 的策略梯度更新，DRO 的 KL/χ² 散度不确定性集合，和对应的对偶优化实现；同时使用对偶参数η进行自适应加权。

**📊 数据集**

训练数据：10,000条来自某大型数学推理数据集（包含多种解题格式）；评测数据：GSM8K、MathQA、GSM-CoT、GSM-Plus、GSM-DE 等五个数学推理基准，覆盖词法、符号、提示方式和多语言的分布偏移。

**📈 对比分析**

与SFT、PPO、DPO、GRPO等基线对比，DRTO 在所有偏移场景下均优于或相当于最强基线；在GSM8K上提高9.17%准确率，MathQA提升1.81%，GSM-CoT提升11.83%，GSM-Plus提升7.92%，GSM-DE提升7.89%。训练动态表明 DRTO 收敛更快、价值网络更稳定、与原始预训练目标偏离更小。

**⚠️ 局限性**

限制：需调节不确定性半径ρ，过大或过小都会影响性能；目前仅验证单轮提示，尚未扩展至多轮会话；实验规模相对有限，未来需在更大数据集和多任务场景中进一步验证。

---

## 52. Demystifying the Silence of Correctness Bugs in PyTorch Compiler

**arXiv ID:** 2604.08720 | [PDF](https://arxiv.org/pdf/2604.08720v1)

**作者:** Meiziniu Li `[一作]` (Hong Kong University of Science and Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

系统分析了 PyTorch 编译器的 116 个正确性 bug，并评估现有 5 种 DL 编译器测试工具的检测效果。

**💡 创新点**

首次针对 PyTorch 编译器正确性 bug 提出基于 bug 特征的 LLM 驱动测试技术。

**🔧 技术方法**

使用 LLM 进行 bug 触发模式提取、变异规则合成以及测试用例变异，并结合现有测试框架实现。

**📊 数据集**

数据集包括从 PyTorch GitHub 收集的 116 个 bug、由 LLM 生成的 23 个新 bug，以及 14 个高优先级 bug。

**📈 对比分析**

与五种现有测试技术对比，后者仅检测到 77 个 bug 中的 26 个，而新技术检测到 23 个未报告的 bug，其中 14 个为高优先级，显示了显著的性能提升。

**⚠️ 局限性**

局限在于手工标注过程的主观性、仅覆盖 PyTorch 编译器，缺乏跨编译器的验证与泛化。

---

## 53. A Representation-Level Assessment of Bias Mitigation in Foundation Models

**arXiv ID:** 2604.08561 | [PDF](https://arxiv.org/pdf/2604.08561v1)

**作者:** Svetoslav Nizhnichenkov `[一作]` (IBM Research), Brian Mac Namee `[通讯]` (University College Dublin)

**通讯引用:** 3046 | [OpenAlex ID](https://openalex.org/A5084251444)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了基于词嵌入的表示层，评估性别偏见缓解对BERT与Llama2的嵌入空间的影响。

**💡 创新点**

首次将偏见缓解效果映射到模型内部表示空间，并提出WinoDec数据集来评估decoder-only模型。

**🔧 技术方法**

使用对抗数据增强、RLHF安全训练、余弦相似度以及Kolmogorov–Smirnov检验分析嵌入分布。

**📊 数据集**

BERT基于真实招聘数据（约60k样本）与人工合成句子；Llama2使用新建的WinoDec共4,000条句对，包含50男、50女职业。

**📈 对比分析**

通过比较基线与缓解模型的余弦相似度分布及KS统计，结果显示偏见差距显著下降，说明偏见缓解在嵌入空间实现了可解释的改进。

**⚠️ 局限性**

仅聚焦性别偏见，未检验种族/年龄等维度；未直接关联下游任务性能；实验仅覆盖单一encoder/decoder对，难以推广到更广泛模型。

---

## 54. SkillForge: Forging Domain-Specific, Self-Evolving Agent Skills in Cloud Technical Support

**arXiv ID:** 2604.08618 | [PDF](https://arxiv.org/pdf/2604.08618v1)

**作者:** Xingyan Liu `[一作]` (Alibaba Group), Honglin Qiao `[通讯]` (Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套名为SkillForge的自我进化框架，用于在企业云技术支持场景中生成、评估并持续改进LLM驱动的Agent技能包。

**💡 创新点**

核心创新在于：① 基于企业历史工单和知识库的领域上下文化技能生成器，解决技能冷启动问题；② 三阶段自动化诊断-优化循环（Failure Analyzer、Skill Diagnostician、Skill Optimizer），能够将部署中的执行失败追溯到具体技能缺陷并自动重写；③ 将技能抽象为文件化包，并通过虚拟文件系统控制执行，提升安全性与可追溯性。

**🔧 技术方法**

技术实现主要依赖Qwen3-Max LLM完成工作流挖掘、工具与知识提取、技能合成与重写；使用ReAct式诊断代理进行失败归因；通过LLM-judge进行一致性评估；整个流程通过VFS和安全约束实现。

**📊 数据集**

实验使用五个真实云技术支持场景，涵盖1,883个工单、3,737个任务，数据来源于企业内部生产工单与知识库。

**📈 对比分析**

与通用技能生成器、人工编写的技能以及传统决策树系统对比，领域上下文化生成器在Strict CR上提升约4.3pp、Lenient CR约3.6pp；自我进化循环在三轮迭代后可将Strict CR提升10–12pp，无论起始点如何；在生产老系统上，最终技能提升约13.8pp。

**⚠️ 局限性**

局限性包括：① 仅支持基于文本的技能，无法执行自定义脚本；② 知识缺口在第一轮已基本解决，后续改进受限于知识库覆盖率；③ 依赖内部大模型与LLM-judge，外部复现受模型差异影响；④ 对企业数据隐私约束较强，限制了模型的可访问数据范围。

---

## 55. Doctoral Theses in France (1985-2025): A Linked Dataset of PhDs, Academic Networks, and Institutions

**arXiv ID:** 2604.08619 | [PDF](https://arxiv.org/pdf/2604.08619v1)

**作者:** William Aboucaya `[一作]` (Université Paris Dauphine PSL), Dastan Jasim `[通讯]` (Université Paris Dauphine PSL)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了覆盖1985-2025年法国博士答辩的综合数据集，通过抓取Thèses.fr、IdRef、TEL、SUDOC等多源数据并进行校正、补全和特征工程，提供论文、作者、导师、评审及机构等多层面信息。

**💡 创新点**

创新点在于将行政答辩记录与国家权威身份数据库(IdRef)深度融合，实现自动纠正不规范ID、推断缺失性别、生成学术网络中心性与经历指标，并支持多语言内容与外部文献检索链接。

**🔧 技术方法**

使用Python数据管道：API抓取、批量下载、JSON解析、手工校正、自动性别推断工具、RDF/SPARQL查询、特征计算及数据整合。

**📊 数据集**

基准数据集为Thèses.fr官方论文平台原始数据，补充IdRef实体识别、TEL和SUDOC标识，形成跨平台可互操作的综合博士答辩数据。

**📈 对比分析**

相较于欧洲其他聚合平台（如Teseo、DART-Europe、BASE等），该数据集在委员会结构、身份标识和时间覆盖方面更完整；虽然未给出数值性能指标，但文中指出数据完整度随时间提升且覆盖率高。

**⚠️ 局限性**

局限性包括：2024后新增答辩未纳入（最新更新为2026年3月），早期（<2010）答辩缺少评审信息，性别推断存在误差，手工校正难以完全覆盖，且仅涵盖法国答辩，国际化候选人信息仍不完整。

---

## 56. Wireless Communication Enhanced Value Decomposition for Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.08728 | [PDF](https://arxiv.org/pdf/2604.08728v1)

**作者:** Diyi Hu `[一作]`, Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于深度学习的端到端语音增强方法，用于提升语音活动检测的准确性。

**💡 创新点**

创新点在于直接将含噪语音映射为干净语音，避免了传统分帧、特征提取等前处理步骤，实现端到端学习。

**🔧 技术方法**

采用深度学习模型（如CNN/Transformer等）进行端到端训练与推理。

**📊 数据集**

使用真实噪声混合语音数据进行训练与验证，干净语音数据用于测试。

**📈 对比分析**

通过与传统语音增强与检测方法比较，实验显示该方法在提升语音清晰度与检测精度方面具有显著优势。

**⚠️ 局限性**

局限性包括对不同噪声类型的泛化能力、模型大小与实时推理的计算开销。

---

## 57. Some variations of the secretary problem

**arXiv ID:** 2604.08593 | [PDF](https://arxiv.org/pdf/2604.08593v1)

**作者:** Sarthak Agrawal `[一作]` (Indian Institute Of Technology Kanpur), Sanjeev Saxena `[通讯]` (Indian Institute Of Technology Kanpur)

**通讯引用:** 269 | [OpenAlex ID](https://openalex.org/A5101785418)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了两种秘密接待问题的变体：一是允许候选人以概率p再次出现的情况；二是将成功定义为选中排名前3的候选人。

**💡 创新点**

创新点在于提出基于阈值的策略并推导了相应的递归概率公式，同时通过差分-积分方法得到极限行为和最优阈值，展示了重复出现信息对决策的积极影响。

**🔧 技术方法**

主要技术包括递归动态规划、Ribas等人提出的函数序列极限理论、定积分与微分方程求解，以及大样本极限下的偏微分方程分析。

**📊 数据集**

本文未使用外部公开数据集，而是通过理论推导和数值模拟（n=100及更大规模）验证结果。

**📈 对比分析**

通过对比p=0（经典问题）和p=1（必重复出现）以及不同p值下的阈值和成功概率，实验表明随p升高，最优阈值和成功概率均显著提升；对于前3名的目标，最优比例约为0.26，成功概率约0.60。

**⚠️ 局限性**

局限在于仅考虑单次或两次出现的情形，未探讨多次出现或非均匀出现概率；此外理论推导依赖于大样本极限，近似误差在小n时可能较大。

---

## 58. Memory-Guided Trust-Region Bayesian Optimization (MG-TuRBO) for High Dimensions

**arXiv ID:** 2604.08569 | [PDF](https://arxiv.org/pdf/2604.08569v1)

**作者:** Abhilasha Saroj `[一作]`, Ross Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究交通仿真校准作为昂贵黑盒优化问题，比较 GA 与多种 Bayesian Optimization 方法。

**💡 创新点**

提出 Memory-Guided TuRBO (MG‑TuRBO) 和自适应采集策略，利用历史信息选择 restart 区域并平衡探索与利用。

**🔧 技术方法**

采用 Gaussian Process surrogate、Thompson Sampling、自适应 EI 采集、信赖域局部搜索、多区域扩展与聚类筛选。

**📊 数据集**

使用两条真实网络 SUMO 模型：Chattanooga 的 14 维校准问题与 Nashville 的 84 维校准问题。

**📈 对比分析**

在 14 维下，TuRBO + Thompson Sampling 最优；在 84 维下，MG‑TuRBO + Adaptive 最佳，整体显著优于 GA 与标准 BO。

**⚠️ 局限性**

仅评估两维度规模，未考察中等维度、不同目标或多精度/在线校准，缺乏更广泛验证。

---

## 59. Self-Sovereign Agent

**arXiv ID:** 2604.08551 | [PDF](https://arxiv.org/pdf/2604.08551v1)

**作者:** Wenjie Qu `[一作]` (National University of Singapore), Dawn Song `[通讯]` (UC Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并阐述了自主自治代理（Self‑Sovereign Agent）的概念、定义及其实现路线图，并对其技术可行性、社会与安全风险进行系统分析。

**💡 创新点**

创新点在于：①给出自主自治代理的正式定义与四大核心属性；②构建分阶段实现路线（从工具级代理到完全自治代理）；③结合经济循环、复制循环与自适应循环三大机制，展示其技术可实现性；④深入探讨法律、经济、治理等多维影响。

**🔧 技术方法**

使用的技术包括大型语言模型（LLM）与代理框架、加密钱包与区块链支付、云计算与 API 调用、自动化财务与复制脚本，以及基于反馈的自适应更新机制。

**📊 数据集**

本文主要基于已有公开实验与案例（如OpenClaw、RLI、Truth Terminal等），并引用 Remote Labor Index 等评估指标，但未提出新的专用数据集。

**📈 对比分析**

由于该工作为概念性与系统性分析论文，未提供实验数据；相比传统代理系统，其优势在于理论可行性与多维风险评估，而缺乏具体性能对比。

**⚠️ 局限性**

主要局限包括：1）缺乏持续盈利的实证证明，2）长周期可靠性与误差累积难题；3）自适应更新可能导致功能退化；4）安全、伦理与监管风险尚未得到充分解决。

---

## 60. Follow My Eyes: Backdoor Attacks on VLM-based Scanpath Prediction

**arXiv ID:** 2604.08766 | [PDF](https://arxiv.org/pdf/2604.08766v1)

**作者:** Diana Romero `[一作]` (University of California, Irvine), Salma Elmalaki `[通讯]` (University of California, Irvine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了针对基于视觉语言模型的扫描路径预测模型的后门攻击，并评估其在移动设备上的有效性；

**💡 创新点**

首次提出了两种可变输出的后门攻击（基于输入的空间误导和时间膨胀），并证明这些攻击能逃避传统的统计检测；

**🔧 技术方法**

采用数据投毒、触发器注入、输入感知生成扫描路径以及后处理技术（如微调、精细剪枝、神经注意力蒸馏、对比学习）进行实验；

**📊 数据集**

使用COCO-Search18数据集对GazeFormer模型进行训练与评估；

**📈 对比分析**

通过与原始模型对比，发现攻击在触发输入下能显著改变扫描路径（空间误导使命中率从0.8降至0.3，时间膨胀使预测停留时间延迟至200+毫秒），但在多种后门防御下仍难以彻底抑制且会导致部分性能下降；

**⚠️ 局限性**

局限性包括仅评估单一模型架构、仅考虑后训练防御且未针对实时可穿戴眼动设备进行实测，且现有防御无法同时保持高清洁性能与完全去除后门。

---

## 61. Task-Aware Bimanual Affordance Prediction via VLM-Guided Semantic-Geometric Reasoning

**arXiv ID:** 2604.08726 | [PDF](https://arxiv.org/pdf/2604.08726v1)

**作者:** Fabian Hahne `[一作]` (Technical University of Darmstadt), Alap Kshirsagar `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于视觉‑语言模型的层次化双臂抓取框架，将双手协同操纵重新定义为联合的可供性定位与臂分配问题；

**💡 创新点**

核心创新在于零样本、任务条件下的语义推理与几何采样的耦合，能够在无需任务专门训练的前提下实现跨类别、跨任务的臂分配与抓取区域选择；

**🔧 技术方法**

采用多视角RGB‑D融合生成全局点云，利用AnyGrasp生成全局抓取候选；然后通过VLM（如GPT‑5）进行任务描述驱动的臂分配与网格化区域选择，并将选定的2D区域投影回3D用于过滤抓取候选；

**📊 数据集**

在Frank‑Panda双臂平台上使用ZED X Mini相机采集的真实世界场景，完成9个涵盖并行抓取、协同稳定、工具使用与人机交互等类别的实验任务；

**📈 对比分析**

与几何仅、臂分配仅、区域分配仅以及VLPart等基线进行对比，采用“策略一致率”评价标准；平均对齐率达到88.9%，显著优于几何仅（9.0%）和其它基线（约55%）；

**⚠️ 局限性**

局限包括：对遮挡敏感、臂分配未进行运动学可行性检查、仅在静态场景下工作，缺乏动态重新规划与实时感知反馈。

---

## 62. Sustained Impact of Agentic Personalisation in Marketing: A Longitudinal Case Study

**arXiv ID:** 2604.08621 | [PDF](https://arxiv.org/pdf/2604.08621v1)

**作者:** Olivier Jeunen `[一作]` (aampe), Schaun Wheeler `[通讯]` (aampe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对一款大型消费类应用的agentic CRM系统进行为期11个月的纵向案例研究，比较了人类营销团队主动管理与随后7个月无监督代理运营的效果。

**💡 创新点**

创新点在于首次提供长期实测证据显示自主代理可持续维持性能提升，并提出人机协同的“共生”运营模型。

**🔧 技术方法**

使用了基于序列决策的强化学习框架、Thompson Sampling探索策略、原子内容组合管道，并通过随机对照试验评估。

**📊 数据集**

采用约880万用户的重新激活目标群体数据，实验将90%分配给agentic组，10%为对照组。

**📈 对比分析**

通过计算治疗组与对照组的相对提升Δ在不同指标上进行比较，活跃阶段推送点击提升≈+65%，被动阶段仍保持≈+57%，但后期略有衰减。

**⚠️ 局限性**

局限性包括仅针对重新激活场景、缺乏生成式内容、对其他业务维度或更长时间跨度的通用性尚未验证。

---

## 63. Fully Autonomous Z-Score-Based TinyML Anomaly Detection on Resource-Constrained MCUs Using Power Side-Channel Data

**arXiv ID:** 2604.08581 | [PDF](https://arxiv.org/pdf/2604.08581v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 64. GRASP: Grounded CoT Reasoning with Dual-Stage Optimization for Multimodal Sarcasm Target Identification

**arXiv ID:** 2604.08879 | [PDF](https://arxiv.org/pdf/2604.08879v1)

**作者:** Faxian Wan `[一作]` (Northeastern University), Yifei Zhang `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉 grounding 与 Chain-of-Thought (CoT) 逻辑的多模态讽刺目标识别框架 GRASP，并构建了平衡且细粒度标注的 MSTI-MAX 数据集。

**💡 创新点**

创新点：① Grounded CoT 通过在推理过程中显式引用视觉区域实现“与图像思考”，提升可解释性；② 双阶段优化（SFT + FTPO）利用坐标加权损失与多维奖励联合训练，显著提升视觉目标定位与跨模态一致性；③ LLM-as-a-Judge 评估法量化中间推理质量。

**🔧 技术方法**

技术手段：多模态大语言模型 (如 Qwen3-VL、InternVL)、低秩适配 LoRA、坐标加权交叉熵、强化学习策略梯度（FTPO）以及自定义奖励体系。

**📊 数据集**

数据集：MSTI-MAX（重构版 MSTI 2.0，包含文本标注、视觉 bounding box、平衡的讽刺/非讽刺样本以及 CoT 解释）。

**📈 对比分析**

与多种基线对比（文本模型、任务专用模型 CofiPara、零样本 MLLMs 及大模型 Gemini-3-Pro 等），GRASP 在讽刺检测、文本目标 F1、视觉目标 AP 等指标上均实现 SOTA，尤其在视觉目标定位上提升 6–10% AP，且在 LLM-as-a-Judge 维度上表现优于大模型。

**⚠️ 局限性**

局限性：① 对高质量 CoT 生成与多维奖励的依赖导致训练成本高；② 仍受限于数据集规模和多样性，跨域泛化虽优但在极端多模态噪声场景下效果待验证；③ 现有实现多使用预训练 MLLM，若缺乏大模型基础，效果可能受限。

---

## 65. Building Better Environments for Autonomous Cyber Defence

**arXiv ID:** 2604.08805 | [PDF](https://arxiv.org/pdf/2604.08805v1)

**作者:** Chris Hicks `[一作]`, Paul Jones `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

介绍AAMAS 2026会议用的ACM LaTeX模板及使用说明

**💡 创新点**

在原ACM模板基础上新增IFAMMAS版权归属，并限制排版修改

**🔧 技术方法**

采用LaTeX文档类、Libertine字体、acmart相关宏包进行排版

**📊 数据集**

无实际数据集使用

**📈 对比分析**

对比原ACM模板与本模板的差异，说明如何正确使用指令

**⚠️ 局限性**

禁止修改页边距、字体等排版细节，且需使用指定字体，否则论文可能被拒

---

## 66. qPRO-AQFP: Post-Routing Optimization of AQFP Circuits with Delay Line Clocking

**arXiv ID:** 2604.08705 | [PDF](https://arxiv.org/pdf/2604.08705v1)

**作者:** Robert S. Aviles `[一作]` (University of Southern California), Peter A. Beerel `[通讯]` (University of Southern California)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对AQFP电路的后路由优化框架qPRO-AQFP，实现时序闭合、时钟调度与缓冲器移除。

**💡 创新点**

创新点在于结合频率感知的时钟调度与全局最优缓冲器移除，并实现自动化的相位跳过。

**🔧 技术方法**

使用了MILP时钟调度、最短路径缓冲器移除算法、线性化的频率相关时序参数；并在后路由阶段完成。

**📊 数据集**

使用了行业常用的AQFP基准电路（如8位加法器等）以及qPALACE/JoSim仿真得到的细胞库。

**📈 对比分析**

通过与TAAS、DLPlace等先前基于放置的结果比较，qPRO-AQFP在后路由条件下实现了100%时序闭合，缓冲器减少34%，延迟5%且频率仅下降4%。

**⚠️ 局限性**

局限性包括对单一细胞库的依赖、相位跳过仅在后路由实现、以及对更大规模电路的实验验证不足。

---

## 67. Attention-Based Sampler for Diffusion Language Models

**arXiv ID:** 2604.08564 | [PDF](https://arxiv.org/pdf/2604.08564v1)

**作者:** Yuyan Zhou `[一作]` (Hong Kong University of Science and Technology), James Kwok `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16819 | [OpenAlex ID](https://openalex.org/A5070273088)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的注意力引导解码算法Attn‑Sampler，用于扩散式语言模型的高效并行生成

**💡 创新点**

从对数似然最大化角度证明，按注意力矩阵列总和降序解码近似最优，并将该理论转化为可执行的解码策略

**🔧 技术方法**

利用Transformer自注意力、块级注意力近似、动态阈值调度以及多层多头信息聚合来实现解码顺序选择

**📊 数据集**

在Fast‑dLLM v2（1.5B/7B）和LLaDA‑1.5（8B）上，用数学推理数据集GSM8K、MATH和代码生成数据集HumanEval、MBPP进行评测

**📈 对比分析**

与基准采样器（KLASS、EB‑Sampler、Fast‑dLLM、Margin、Entropy、Confidence）对比，Attn‑Sampler在所有指标上均达到或超过最佳水平，且在吞吐量-准确率曲线中实现更优的 Pareto 前沿

**⚠️ 局限性**

对注意力矩阵不变性等假设的依赖以及块级近似可能限制理论精度，且在极大模型或其他扩散架构上的泛化需进一步验证

---

## 68. Sentiment Classification of Gaza War Headlines: A Comparative Analysis of Large Language Models and Arabic Fine-Tuned BERT Models

**arXiv ID:** 2604.08566 | [PDF](https://arxiv.org/pdf/2604.08566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 69. Communicate-Predict-Act: Evaluating Social Intelligence of Agents

**arXiv ID:** 2604.08727 | [PDF](https://arxiv.org/pdf/2604.08727v1)

**作者:** David Shoresh `[一作]`, Yonatan Loewenstein `[通讯]` (Hebrew University Of Jerusalem)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一个多玩家混合合作-竞争的社交游戏竞技场，并利用COMPACT交互协议系统地评估LLM智能体的社交智能，生成了Elo评分和细粒度的社会认知指标；

**💡 创新点**

创新点在于：①将游戏竞技与自然语言交互相结合形成统一的评估框架；②设计了七项社会认知指标（理论心智、可预测性、影响力、易受影响、断言力、规划深度、学习能力），并通过LLM评审自动化提取；③通过社会认知特征解释模型展示影响力与透明度是社交智能的核心，优于传统的理论心智；

**🔧 技术方法**

使用技术包括：COMPACT交互协议（沟通-预测-行动）、Elo评分（Bradley–Terry模型）、多维多项式社会认知指标、逻辑回归及交叉验证评估，以及LLM评审（GPT‑5 mini / Qwen3 Max 80B）进行指标评分；

**📊 数据集**

实验数据集包含8个不同参数规模（24B–1T）的LLM智能体，在5种社交游戏（Coalition、Scheduler、Tragedy of Commons、Survivor、HUPI）和4个玩家规模（2–5人）下运行共928局比赛，覆盖约一半所有组合；

**📈 对比分析**

比较方法：Elo模型单维评分（AUC≈0.67），加入游戏偏置后略增；社会认知指标模型（固定权重）AUC≈0.75，按游戏权重后AUC≈0.82，显著优于单维Elo，证明多维社会认知特征更能解释比赛结果；

**⚠️ 局限性**

局限性包括：LLM评审的自动化评分可能缺乏人类验证；实验仅限于LLM智能体，未扩展到人类或混合人机交互；评估框架主要聚焦奖励驱动的社交情境，可能忽略更广泛的社交动力学。

---

## 70. Skip-Connected Policy Optimization for Implicit Advantage

**arXiv ID:** 2604.08690 | [PDF](https://arxiv.org/pdf/2604.08690v1)

**作者:** Fengwei Teng `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhijiang Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Skip-Connected Policy Optimization (SKO)，通过将推理拆分为上游稠密奖励阶段和下游组相对优化阶段，解决传统 Monte Carlo 估计高方差导致的性能下降问题。

**💡 创新点**

创新点在于：①只在单一中间位置实例化 Monte Carlo 采样，提供稠密奖励；②引入跳连结构 [s, q]，既利用上游推理，又保留下游自由探索；③采用上游单流优化和下游组相对优化的非对称策略；④通过 KV 缓存重写实现与 GRPO 同等算力。

**🔧 技术方法**

核心技术包括 Monte Carlo 路径采样、组相对政策优化（GRPO）、单流策略优化（SPO）、KV 缓存重写、动态分割位置采样与中值选择。

**📊 数据集**

使用 dapo-math-17k 训练集，并在 Qwen2.5-Math-7B 与 Llama-3.2-3B-Instruct 基础模型上进行实验。

**📈 对比分析**

与 GRPO、GSPO、SPO、Critique-GRPO、PRIME、DAPO、SAPO、CISPO 等最强基线相比，SKO 在 Qwen 7B 上平均提升约 3.9%（对 42.5% 最高得分），在 Llama 3B 上提升约 6.2%（对 24.1% 最高得分），并在 MMLU‑Pro、LiveCodeBench 等 OOD 任务中保持优势。

**⚠️ 局限性**

局限性包括仅在 3B/7B 模型上验证、仅在单一拆分点采样（未探索多拆分或后期拆分）、对外部评估器的依赖导致奖励估计相对性、以及 KV 重写实现需依赖自定义推理引擎。

---

## 71. VerifAI: A Verifiable Open-Source Search Engine for Biomedical Question Answering

**arXiv ID:** 2604.08549 | [PDF](https://arxiv.org/pdf/2604.08549v1)

**作者:** Miloš Košprdić `[一作]` (Institute for Artificial Intelligence Research and Development of Serbia), Nikola Milošević `[通讯]` (Institute for Artificial Intelligence Research and Development of Serbia)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 VerifAI，一套集成检索增强生成（RAG）与后期主张验证的可验证开源生物医学问答系统。

**💡 创新点**

创新点在于：① 将生成的答案拆分为原子主张并使用自然语言推理（NLI）进行逐条验证，显著降低幻觉；② 通过小型模型（Mistral‑7B‑Instruct）在引用生成上的高精度，挑战大型模型的必要性；③ 将检索、生成与验证模块模块化，支持透明可审计。

**🔧 技术方法**

核心技术包括：混合检索（BM25 + 语义向量检索）、对 Mistral‑7B‑Instruct 进行 QLoRA 微调生成带引用答案、Fine‑tuned DeBERTa 作为 NLI 验证器，以及整体流水线的并行化实现。

**📊 数据集**

使用的数据集有：PubMed 语料库（约 25.5M 摘要）、自构 PQAref（9,075 Q&A 对）、SciFact（1,409 句子级主张）以及 HealthVer（14,330 证据‑主张 对）。

**📈 对比分析**

对比方法：与 GPT‑4 / GPT‑4 Turbo、PubMed 原生检索、传统 RAG 方案进行对标；IR P@10 23.7%、MAP@10 42.7%；生成支持率 81%（与黄金答案结论一致），关键信息覆盖 74%；验证准确率 81–84%，在 HealthVer 上超过 GPT‑4，表明域特定 NLI 在医学推理中具备优势。

**⚠️ 局限性**

局限性：① 验证模型仍难准确捕捉细微矛盾与隐含否定；② 检索覆盖率不足，尤其是列表类问题导致答案不完整；③ 评估部分依赖 GPT‑4 作为自动判定，缺乏专家手工标注；④ 目前仅在 PubMed 上验证，跨域泛化需进一步测试。

---

## 72. Adaptive Simulation Experiment for LLM Policy Optimization

**arXiv ID:** 2604.08779 | [PDF](https://arxiv.org/pdf/2604.08779v1)

**作者:** Mingjie Hu `[一作]` (Fudan University), Enlu Zhou `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于配对比较的自适应仿真实验框架 LLM-PO，用来在有限的候选策略集合中以高置信度识别最优 LLM 策略。

**💡 创新点**

创新点包括：①在无结构和结构化策略空间分别推导了信息理论下的最优采样比例；②在结构化空间中引入线性 Bradley‑Terry 模型并给出 Fisher 信息下的上界；③设计了可实现 δ‑PAC 保证且收敛至极限样本需求的自适应实验流程；④通过正则化凸优化实现唯一的最优采样分布。

**🔧 技术方法**

使用的技术主要有：配对比较抽样、信息理论下的 KL 与 Fisher 信息分析、凸优化（含正则化）、自适应实验设计、贝叶斯/频率统计推断（β 分布、最大似然）、停止规则与错误概率控制。

**📊 数据集**

数据集：人工生成的无结构和结构化实例（Latent score + Bradley‑Terry 产生的 μ 矩阵）；真实任务采用 Instruction Induction 与 BIG‑bench 上的四个子任务（Object Counting、Word Unscrambling、Second Word Letter、Sum），使用 Llama‑3:8B 作为生成模型，Qwen2.5‑7B 或规则判断器作为评审。

**📈 对比分析**

与 Baseline（RoundRobin、RandomPair、EpsGreedy、Thompson Sampling、RUCB）比较时，LLM‑PO 在 PCS、停止时间和样本效率方面均显著优于其他方法；在无结构空间 120 条比较中，LLM‑PO 仅需几千条比较即可接近 100% 正确率；在结构化空间 496 条比较中，停止时间约为 6500 条，显著低于对手的 15000‑23000 条；真实任务中，LLM‑PO 取得最高或近 100% 的正确率。

**⚠️ 局限性**

局限性包括：①需要手工或自动的评审（Judge）来提供配对偏好，可能受人类主观或评审模型误差影响；②对最优策略唯一性假设较强，若多种策略同样优则需要额外正则化；③在极大策略集合时，虽然采样集中在重要比较，但仍需维护足够的探索，导致计算和通信开销；④理论分析主要聚焦于固定置信度场景，未深入探讨预算限制或非平稳环境。

---

## 73. What Matters in Virtual Try-Off? Dual-UNet Diffusion Model For Garment Reconstruction

**arXiv ID:** 2604.08716 | [PDF](https://arxiv.org/pdf/2604.08716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 74. LEGO: Latent-space Exploration for Geometry-aware Optimization of Humanoid Kinematic Design

**arXiv ID:** 2604.08636 | [PDF](https://arxiv.org/pdf/2604.08636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 75. QoS-QoE Translation with Large Language Model

**arXiv ID:** 2604.08703 | [PDF](https://arxiv.org/pdf/2604.08703v1)

**作者:** Yingjie Yu `[一作]` (University of Illinois Urbana-Champaign), Klara Nahrstedt `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一套基于文献的、源可追溯的QoS–QoE关系数据集，并用该数据集对大型语言模型进行双向QoS↔QoE翻译任务的微调与评估。

**💡 创新点**

首次将QoS–QoE关系从多样化的研究论文中自动抽取、结构化并经过多轮LLM评审实现高质量、可追溯的数据集，成为该领域的基准资源。

**🔧 技术方法**

采用GPT‑5.2思考进行关系抽取，使用Gemini‑2.5‑flash‑lite、Claude‑haiku‑4‑5‑20251001、Grok‑4.20‑0309‑reasoning进行评审，并在Tinker框架下对Qwen3、Llama等模型进行监督微调。

**📊 数据集**

QoS‑QoE Translation数据集（共1026条记录，来源于505篇2017‑2025年期刊论文），并以此对模型进行训练和测试。

**📈 对比分析**

通过在微调前后的MAPE、Accuracy@δ、Accuracy和Macro‑F1四个指标对比，发现微调后模型在连续值预测上MAPE下降至8.49%，在离散标签预测上准确率升至90.24%，显示显著性能提升。

**⚠️ 局限性**

当前LLM在多步推理、时序理解和跨模态推理上仍不稳定，导致部分复杂QoS–QoE关系的抽取和翻译仍出现误差，需要进一步提升模型的推理可靠性。

---

## 76. AniGen: Unified $S^3$ Fields for Animatable 3D Asset Generation

**arXiv ID:** 2604.08746 | [PDF](https://arxiv.org/pdf/2604.08746v1)

**作者:** Yi-Hua Huang `[一作]` (University of Hong Kong), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

论文提出了基于S^3字段的统一框架AniGen，实现从单张图像直接生成可动画的3D资产（网格、骨架、蒙皮）。

**💡 创新点**

关键创新包括将形状、骨架和蒙皮统一为连续的S^3字段；使用置信度衰减骨架场解决Voronoi不确定性；引入双重皮肤字段和SkinAE实现关节数无关的蒙皮；以及两阶段稀疏结构与结构化潜流生成。

**🔧 技术方法**

采用稀疏结构自编码器、结构化潜自编码器、流匹配生成器、Transformer、Swin块、Denoising Auto-Encoder、SkinAE、BVH加速蒙皮转移等技术。

**📊 数据集**

主要使用ArticulationXL（约33k件带骨架的3D模型）进行训练，并在ArticulationXL上进行数据增强；对比使用TRELLIS等生成模型与UniRig、Anymate等自动装配算法。

**📈 对比分析**

在骨架准确性（Chamfer、Wasserstein、Gromov‑Wasserstein）和蒙皮质量（KL、L1、L2）上，AniGen显著优于所有基线；几何质量与改进后TRELLIS相当；推理时间与最快的序列基线相当。

**⚠️ 局限性**

限制包括仅支持单图像条件；对视频或动态输入无法保证时间一致性；对需要严格几何对齐的物体（如笔记本）蒙皮可能出现缝隙；骨架倾向于训练数据中的中轴线，而非专业的解剖学装配。

---

## 77. SenBen: Sensitive Scene Graphs for Explainable Content Moderation

**arXiv ID:** 2604.08819 | [PDF](https://arxiv.org/pdf/2604.08819v1)

**作者:** Fatih Cagatay Akyon `[一作]` (Middle East Technical University), Alptekin Temizel `[通讯]` (Middle East Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了首个大规模电影场景图数据集SenBen，并基于此训练了一个241M的小型多任务视觉语言模型；

**💡 创新点**

创新点在于：①设计了针对敏感内容的视觉场景图标注范式与复合召回评估指标；②提出了词表感知召回损失（VAR）与后缀式对象身份标记、解耦式Query2Label标签头等多任务训练技巧；

**🔧 技术方法**

采用了视觉语言模型（如Florence-2-base）知识蒸馏、DaViT编码器、解码器+Q2L标签头、VAR损失、最小排列交叉熵、标签平滑与自举采样等技术；

**📊 数据集**

利用MECD电影时间戳提取约13,999帧，覆盖16个敏感标签，构建了视觉图灵风格的对象/属性/谓词与情感属性标注；

**📈 对比分析**

与前沿VLM（Gemini、Qwen3-VL、GLM等）以及商业安全API（OpenAI Moderation、Azure等）做零样本/本地推理对比，模型在对象检测、标注召回、图像说明等指标上达到或超过所有VLM（仅Gemini略胜），同时推理速度提升7.6×、显存仅1.2GB；

**⚠️ 局限性**

局限性包括：初始标注来源于Gemini 3 Pro导致潜在风格偏差；标注仅由首位作者完成，缺乏交叉一致性评估；数据主要为西方影片，缺乏跨文化样本；未处理多帧动作时序，谓词召回仍偏低。

---

## 78. Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent

**arXiv ID:** 2604.08552 | [PDF](https://arxiv.org/pdf/2604.08552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 79. Extrapolating Volition with Recursive Information Markets

**arXiv ID:** 2604.08606 | [PDF](https://arxiv.org/pdf/2604.08606v1)

**作者:** Abhimanyu Pallavi Sudhir `[一作]` (University of Warwick), Long Tran-Thanh `[通讯]` (University of Warwick)

**通讯引用:** 384 | [OpenAlex ID](https://openalex.org/A5103245667)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个贝叶斯框架，用以量化在信息不对称下信息的价值，并基于此设计了递归检验协议（Recursive Inspection Protocol）以及可扩展监督机制（Scalable Oversight Mechanism）以改进信息市场和 AI 训练中的人类反馈；

**💡 创新点**

创新点在于：①将信息检验视为一个不完美记忆博弈，提出递归检验协议消除信息不对称导致的效用损失；②将递归检验与可扩展监督结合，构造了一个多级信息生成与奖励的子游戏，理论上实现对 AI 产出信息的“完全信息化”评价；

**🔧 技术方法**

核心技术包括：贝叶斯决策理论、信息价值（VOI）的期望与后验计算、递归博弈建模（不完美记忆游戏）、子游戏完美均衡分析、以及对信息市场的正式设计与实现；

**📊 数据集**

未使用公开数据集；实验部分采用自建的基于仿真场景的问答与产品检验数据，展示了递归检验协议在 Q&A、产品监管、社区事实核查等场景的可行性；

**📈 对比分析**

方法对比主要体现在理论分析与实现演示：相比单层检验协议，递归检验在信息不对称下的期望收益更高，且实现了更低的交易成本；在可扩展监督中，通过多级信息生成与奖励，提升了 AI 输出的质量；

**⚠️ 局限性**

局限性包括：①递归检验协议在极端信息成本差异时仍可能导致信息不完整或被误导；②可扩展监督机制未能完全实现“最优信息化”目标，存在短板估计误差；③目前仅在仿真和小规模实现上验证，缺乏大规模真实数据的实证评估。

---

## 80. DeFakeQ: Enabling Real-Time Deepfake Detection on Edge Devices via Adaptive Bidirectional Quantization

**arXiv ID:** 2604.08847 | [PDF](https://arxiv.org/pdf/2604.08847v1)

**作者:** Xiangyu Li `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了专为深度伪造检测设计的量化框架 DeFakeQ，使高性能检测模型在边缘设备上实现实时推理。

**💡 创新点**

创新点在于双向自适应量化：横向自适应块量化 (HAQ) 按块重要性动态分配位宽，纵向高效特征微调 (VEFT) 以少量全精度通道保留细粒度伪造特征。

**🔧 技术方法**

采用 Post‑Training 量化结合自适应位宽分配、重构损失、对比学习以及混合精度微调。

**📊 数据集**

使用了五大深伪造数据集：FaceForensics++、DeepfakeDetection、Deepfake Detection Challenge、DFDC preview 以及 CelebDF。

**📈 对比分析**

与 BRECQ、Adalog、FIMA‑Q 等通用量化基线及 ADD、GAC‑FAS、X‑Pruner 等压缩方法对比，DeFakeQ 在保持 90% 以上准确率的同时将模型压缩至原体积的 10–20%，在跨数据集评测中准确率提升 11–35%，压缩率优于传统方法。

**⚠️ 局限性**

局限在于目前仅验证在手机等移动端，未扩展到更低功耗 IoT、边缘网关或极端环境（低光、遮挡），且缺乏自适应实时量化的动态策略。

---

## 81. From Dispersion to Attraction: Spectral Dynamics of Hallucination Across Whisper Model Scales

**arXiv ID:** 2604.08591 | [PDF](https://arxiv.org/pdf/2604.08591v1)

**作者:** Ivan Viakhirev `[一作]` (ITMO), Grach Mkrtchian `[通讯]` (MTUCI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过Spectral Sensitivity Theorem研究了大型ASR模型中幻听现象的几何根源，揭示了从信号扩散到低秩吸引子阶段的相变。

**💡 创新点**

提出Spectral Sensitivity Theorem及Spectral Propagation Instability框架，将层级增益与对齐度量结合，首次把幻听归因于低秩吸引子而非随机噪声。

**🔧 技术方法**

使用Transformer内部激活谱分解、奇异值分解、有效秩、谱斜率、Kirchhoff指数等频谱指标，并对Whisper模型进行层级Jacobian分析。

**📊 数据集**

采用LibriSpeech的Hell对抗数据集，对Tiny、Small、Large规模Whisper模型进行实验。

**📈 对比分析**

与传统WER/概率指标对比，实验表明小模型处于Disintegration（跨注意力秩下降约13%），大模型进入Attractor（自注意力秩压缩约2.3%，谱斜率提升）并能解释幻听发生。

**⚠️ 局限性**

局限性在于对齐度量仅通过谱指标间接估计，缺乏对内部子空间对齐的直接测量，且仅在Whisper系列验证，未覆盖其他ASR体系。

---

## 82. Alleviating Community Fear in Disasters via Multi-Agent Actor-Critic Reinforcement Learning

**arXiv ID:** 2604.08802 | [PDF](https://arxiv.org/pdf/2604.08802v1)

**作者:** Yashodhan D. Hakke `[一作]` (Virginia Tech), Hoda Eldardiry `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将灾害中的社区恐惧与基础设施恢复问题建模为三方非零和差分游戏，并通过在线 actor‑critic 强化学习实现分散化资源部署与实时恐惧缓解；

**💡 创新点**

创新点在于将 CPS（Cyber‑Physical‑Social）模型加入可控渠道，形成可执行的非零和差分游戏框架，并通过模型基 actor‑critic 以在线方式求解 Nash 平衡，实现跨灾害泛化的控制策略；

**🔧 技术方法**

使用技术包括：模型基非零和差分游戏理论、Hamilton‑Jacobi 方程近似、在线 actor‑critic 双层学习、两时间尺度权重更新、探索信号设计、状态投影与控制饱和处理；

**📊 数据集**

使用的数据集为 2017 年飓风 Harvey 与 Irma 的社交媒体文本（LIWC 分析）、电力停电记录、FEMA 调度日志，用以校准 CPS 动力学并评估控制效果；

**📈 对比分析**

与无控制、最大控制、比例启发式、中心化 actor‑critic 等基准进行对比。结果显示在 Harvey 上可实现约 70% 的恐惧降低，Irma 上约 50%，同时控制成本最低、效能最高；

**⚠️ 局限性**

局限性包括：依赖已校准的 CPS 模型，模型参数和控制信号维度受限；基函数过度参数化导致持久激励（PE）不完全；仅在仿真层面验证，缺乏现场实验；对 EMS 的控制渠道受限，需进一步改进。

---

## 83. Unbiased Rectification for Sequential Recommender Systems Under Fake Orders

**arXiv ID:** 2604.08550 | [PDF](https://arxiv.org/pdf/2604.08550v1)

**作者:** Qiyu Qin `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4228 | [OpenAlex ID](https://openalex.org/A5039670436)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Dual‑view Identification and Targeted Rectification (DITaR)框架，用于在序列推荐系统中高效去除假订单并保持模型性能；

**💡 创新点**

创新点在于通过协同与语义双视图对比检测假订单，结合影响函数筛选真正有害样本，再用梯度上升局部校正，避免全量重训且保留有益信息；

**🔧 技术方法**

使用预训练语言模型LLaMA2提取语义嵌入、PCA与门控融合协同嵌入、双分支序列编码、对比学习、影响函数推断及梯度上升微调；

**📊 数据集**

实验基于MovieLens‑1M、Amazon‑Beauty和Yelp2018三个工业级推荐数据集，并构造三类假订单场景；

**📈 对比分析**

与Retrain、SISA、RecEraser和UltraRE等基线比较，DITaR在Hit@k/NDCG@k上均接近或超越原始无假订单性能，且收敛周期显著更少；

**⚠️ 局限性**

局限在于仍需预先训练完整模型进行检测，且对极端多样化或更隐蔽的假订单可能检测灵敏度不足。

---

## 84. MolPaQ: Modular Quantum-Classical Patch Learning for Interpretable Molecular Generation

**arXiv ID:** 2604.08575 | [PDF](https://arxiv.org/pdf/2604.08575v1)

**作者:** Syed Rameez Naqvi `[一作]` (Tulane University), Lu Peng `[通讯]` (Tulane University)

**通讯引用:** 1603 | [OpenAlex ID](https://openalex.org/A5100762718)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了 MolPaQ，一种将量子电路嵌入经典条件-聚合管线的模块化分子生成框架，能够在保证化学合法性的同时实现属性导向和可解释的分子合成。

**💡 创新点**

创新点在于将量子电路作为局部“patch”生成器，提供高阶相关的节点嵌入；以及构建基于距离和价键约束的聚合器，二者共同实现了对分子拓扑的可控塑形和解释性。

**🔧 技术方法**

使用的技术包括：β‑VAE 预训练的潜在空间，MLP 条件器，参数高效的 RY‑CNOT 量子电路（强相关层），连接优先的聚合器，GINE 判别器，latent critic 与 chemistry‑shaped reward 进行对抗与强化学习。

**📊 数据集**

主要使用的数据集是公开的 QM9 数据集，用于训练潜在空间、评估生成质量并与基线模型对比。

**📈 对比分析**

与多种经典与量子 VAE/GAN 方案（QVAE‑Mole、SQ‑VAE、QGAN‑HG 等）对比，MolPaQ 在 QM9 上达成 100% 合法性、99.75% 新颖度、0.905 的多样性，推理速度仅 0.065 s/分子；量子版本相较 MLP 生成器提升了约 2% 的 QED 与 10–12% 的芳香环产率。

**⚠️ 局限性**

主要局限包括：量子电路深度受限，难以直接生成 3D 坐标；对 logP 的控制相对弱；模型依赖于大量预训练数据，扩展到更大分子空间时的可扩展性仍需验证。

---

## 85. Why Network Segmentation Projects Fail

**arXiv ID:** 2604.08632 | [PDF](https://arxiv.org/pdf/2604.08632v1)

**作者:** Rohit Dube `[一作]` `[通讯]` (Cisco Systems Inc.), Rohit Dube (Cisco Systems Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对400名美国网络安全从业者的问卷调查，构建了网络分段项目失败的两部分框架，并利用潜在类别分析识别出四种失败类型。

**💡 创新点**

首次系统性实证研究网络分段项目失败，提出结合通用IT项目失败与分段特定技术障碍的框架，并揭示不同失败类型在治理与技术因素上的差异。

**🔧 技术方法**

采用潜在类别分析（Latent Class Analysis）对12个Likert量表项进行聚类，并使用BIC、AIC、熵、最小类规模及Bootstrap稳定性等指标进行模型选择与验证。

**📊 数据集**

使用400份美国网络安全从业者的问卷数据，涵盖B1–B6和C1–C6等12个Likert项，并收集项目环境、规模、分段模型与失败类型等信息。

**📈 对比分析**

通过比较BIC、AIC梯度、熵和Bootstrap稳定性等多重指标，确定四类为最优；四类在项目属性上显著差异，外部验证支持其统计与实用价值。

**⚠️ 局限性**

样本局限性、问卷项覆盖范围有限、两小类样本量不足、未对多重检验做严格校正、自由文本编码单一评审、且仅捕捉因果感知，未能完整解释失败机制。

---

## 86. Accelerating Transformer-Based Monocular SLAM via Geometric Utility Scoring

**arXiv ID:** 2604.08718 | [PDF](https://arxiv.org/pdf/2604.08718v1)

**作者:** Xinmiao Xiong `[一作]` (University of Wisconsin–Madison), Zhiwen Fan `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `51c0528b-f690-4182-ae60-bb5f046c276c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出轻量级前馈帧门控网络LeanGate，预测帧的几何效用并在几何解码前过滤冗余帧

**💡 创新点**

通过对几何效用分数的教师蒸馏实现无后向几何匹配的预测门控，显著减少对昂贵GFM推理的调用

**🔧 技术方法**

利用FLARE基础网络的摄像机感知标记，构建迭代重构头预测几何效用，并使用Huber损失进行蒸馏训练

**📊 数据集**

在ScanNet++ 150个室内场景中构造0.5M对视图样本进行教师标注，并在TUM‑RGBD、7‑Scenes、EuRoC等公共SLAM基准上进行评估

**📈 对比分析**

与统一间隔采样和原始MASt3R‑SLAM比较，LeanGate在保持定位精度的前提下，帧数减少超过90%，跟踪FLOPs下降85%，整体吞吐量提升5×，重建质量接近或优于全帧方法

**⚠️ 局限性**

对灰度图像的鲁棒性不足、依赖预训练权重以及在户外环境中缺乏验证等仍是未来改进的限制

---

## 87. LMGenDrive: Bridging Multimodal Understanding and Generative World Modeling for End-to-End Driving

**arXiv ID:** 2604.08719 | [PDF](https://arxiv.org/pdf/2604.08719v1)

**作者:** Hao Shao `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LMGenDrive，一种将大语言模型（LLM）与多视角视频生成器结合的闭环端到端自动驾驶框架，能够在给定自然语言指令和多摄像头输入下同时生成未来驾驶视频和控制指令；

**💡 创新点**

创新点在于：1）统一理解（LLM）与生成（世界模型）两大能力于同一架构；2）利用可学习的世界查询与动作查询实现跨模态信息融合；3）采用三阶段训练策略（视觉预训练→单步预测→多步长预测）提升长期时序建模与稳定性；

**🔧 技术方法**

技术手段包括：LLaMA LLM + Q-Former + MLP；多视角视觉编码器（ResNet+Transformer+BEV编码）；CLIP特征对齐；U-Net扩散模型（Stable Diffusion 1.5）用于多视角视频生成；PID控制器将轨迹点转为低层控制；AdamW+DeepSpeed ZeRO-2；

**📊 数据集**

使用 CARLA 0.9.10.1 虚拟数据，训练集为 3M 专家采集的帧，评测基准为 LangAuto（包含三条赛道：LangAuto、LangAuto-Short、LangAuto-Tiny）；

**📈 对比分析**

与现有 LMDrive、AD-H、BEVDriver 等对比，LMGenDrive 在 LangAuto 基准上 DS 分别提升至 62.2、77.1、84.1，RC 与 IS 也有显著提升，表现出更强的指令跟随、时空理解与鲁棒性；

**⚠️ 局限性**

局限性包括：1）长时序生成误差累积导致远期帧质量下降；2）对真实感知与多模态对齐仍有提升空间；3）训练资源需求大，需多 GPU 与大规模显存；4）目前仅在仿真环境验证，真实世界部署尚未充分测试。

---

## 88. QuanBench+: A Unified Multi-Framework Benchmark for LLM-Based Quantum Code Generation

**arXiv ID:** 2604.08570 | [PDF](https://arxiv.org/pdf/2604.08570v1)

**作者:** Ali Slim `[一作]` (American University of Beirut), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 25118 | [OpenAlex ID](https://openalex.org/A5024763828)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套跨 Qiskit、PennyLane 与 Cirq 的统一量子代码生成基准。

**💡 创新点**

创新点在于统一任务意图、采用可执行功能测试、引入 KL 散度衡量概率输出，并构建反馈修复机制。

**🔧 技术方法**

主要技术包括 Pass@k 评估、KL 散度判定、自动化执行与打分管道，以及基于错误信息的反馈修复循环。

**📊 数据集**

使用的数据集为 42 条来自 QuanBench 的任务，涵盖量子算法、门分解和态制备三类。

**📈 对比分析**

通过 Pass@1/5 与反馈修复后的 Pass@1 进行比较，实验显示 Qiskit 最高，PennyLane 最低，反馈修复可将准确率提升至约 83%（Qiskit）/ 77%（Cirq）/ 67%（PennyLane）。

**⚠️ 局限性**

局限性包括任务数量有限、仅覆盖三大框架、仅测评 Pass@1/5/FB，且对更复杂场景和其他评估指标关注不足。

---

## 89. Adversarial Sensor Errors for Safe and Robust Wind Turbine Fleet Control

**arXiv ID:** 2604.08750 | [PDF](https://arxiv.org/pdf/2604.08750v1)

**作者:** Julian Quick `[一作]` (Technical University of Denmark), Pierre-Elouan Mikael Rethore `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过在风电场级控制中引入对抗训练，研究如何使风电场控制器在面对传感器误差或恶意攻击时保持鲁棒性，且与传统的程序化噪声训练进行对比。

**💡 创新点**

创新点在于提出“军备竞赛”(Arms Race)对抗训练框架，让攻击者逐代提升对控制器的混淆能力，并与两种自我对抗方法（Synthetic Self Play、Self-Play）做对比，证明Arms Race能得到最强的鲁棒控制器。

**🔧 技术方法**

使用技术包括基于WindGym的动态wake meandering仿真、PyWake专家控制器、Proximal Policy Optimization (PPO) 强化学习、对抗噪声生成、三种共训练策略（Arms Race、Synthetic Self Play、Self-Play）以及零和博弈分析。

**📊 数据集**

数据集为仿真生成的两台Vestas V80风机的风流场，风速区间6–7 m/s，风向267–273°，无额外湍流，仅利用动态wake模型产生的状态与奖励。

**📈 对比分析**

通过在5个标准入流情形下对每个主角-对手组合进行评估，计算相对无转向基准的奖励，绘制热图与折线图。结果显示Arms Race训练出的主角在所有对手下表现最佳，最坏情况相对基准提升+7.9%功率，而程序化噪声训练的主角则出现-39%功率损失。

**⚠️ 局限性**

局限性包括训练过程可能不收敛、对抗者易出现灾难性遗忘、实验仅在两台机组的小型场景中验证，规模化扩展与真实转移仍需进一步研究，且训练结果受种子波动影响较大。

---

## 90. MedConceal: A Benchmark for Clinical Hidden-Concern Reasoning Under Partial Observability

**arXiv ID:** 2604.08788 | [PDF](https://arxiv.org/pdf/2604.08788v1)

**作者:** Yikun Han `[一作]` (University of Illinois Urbana Champaign), Yue Guo `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于患者模拟器的医疗对话基准，用来评估在部分可观测的异信息环境下，医生如何挖掘并处理患者的隐藏关注点。

**💡 创新点**

创新点在于：①保留隐藏关注点为不可见状态，模拟器只在适当的询问后才揭露；②结合结构化的四类隐藏关注点分类法；③设计过程感知评估，区分确认和干预两个任务；④使用多种对话质量指标将交互过程量化。

**🔧 技术方法**

采用了对话生成的大型语言模型（Qwen、Claude、GPT‑5.2、Doctor‑R1、Llama3‑OpenBioLLM），结合基于RIAS、VR‑CoDES、NCF的特征向量进行隐藏关注点的概率估计与状态转移；同时使用了经验式的可观测信号更新与阈值门控。

**📊 数据集**

数据集为300个从r/AskDocs中收集的真实临床问答，经过专家验证后转换为结构化病例，包含可见临床信息与隐藏关注点；共生成600人机交互实例。

**📈 对比分析**

通过与159名真实医生的实验比较，评估指标包括揭示率、精确率、召回率、F1以及干预成功率。结果显示，人工医生在确认和干预方面均领先；不同LLM在揭示率和细粒度恢复上表现差异，部分模型在长对话中有所提升。

**⚠️ 局限性**

局限性包括：①患者模拟器基于有限的四类关注点，缺乏更细粒度或跨领域的关注；②数据来源单一（Reddit），可能存在社交媒体偏差；③评估侧重过程而非最终临床结果；④模型可能在模拟器内部学习到“游戏”策略，需进一步防御。

---

## 91. Generative Simulation for Policy Learning in Physical Human-Robot Interaction

**arXiv ID:** 2604.08664 | [PDF](https://arxiv.org/pdf/2604.08664v1)

**作者:** Junxiang Wang `[一作]` (Carnegie Mellon University), Zackory Erickson `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

基于高阶自然语言提示，构建零射“text2sim2real”框架，自动生成软体人模型、场景布局及机器人轨迹，收集大规模仿真演示并训练视觉基仿学习策略。

**💡 创新点**

首次在物理人机交互中实现零射文本到仿真再到真实的全自动生成与训练流程，并通过LLM与VLM实现任务、人体、场景三维合成。

**🔧 技术方法**

使用大型语言模型（Gemini 3 Pro）、视觉语言模型、SMPL‑X软体人体、ARCHITECT场景生成、Genesis物理仿真、层级模仿学习架构及点云感知技术。

**📊 数据集**

生成的合成演示数据约8,000条（每任务4,000条），涵盖多种人体形态与姿态，全部由SMPL‑X与模拟器合成；未使用真实人类视频或公开数据集。

**📈 对比分析**

仿真成功率分别为96.3%（刮痧）和96.9%（沐浴）；在真实5位受试者实验中，刮痧在静态条件下100%成功、运动条件下80%，沐浴在两条件下均84%；相比传统手工脚本或真实示范方法，显著提升并实现零射迁移。

**⚠️ 局限性**

限制包括缺乏触觉信息导致沐浴时接触不充分、仅使用静态人体姿态、对动态运动的适应性有限，以及对更复杂任务通用性的验证尚待扩展。

---

## 92. Practical Bayesian Inference for Speech SNNs: Uncertainty and Loss-Landscape Smoothing

**arXiv ID:** 2604.08624 | [PDF](https://arxiv.org/pdf/2604.08624v1)

**作者:** Yesmine Abdennadher `[一作]` (Idiap Research Institute), Philip N. Garner `[通讯]` (Idiap Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文在语音识别任务中使用改进的变分在线牛顿方法（IVON）对基于突触神经网络（SNN）的代理梯度模型进行贝叶斯学习，并验证其在预测平滑度和不确定性方面的优势。

**💡 创新点**

创新点在于首次将IVON应用于SNN，证明贝叶斯权重分布能缓解由阈值触发的尖锐损失景观，从而提升预测分布的平滑度和鲁棒性。

**🔧 技术方法**

采用了Leaky Integrate-and-Fire（LIF）神经元的双层前馈SNN结构、盒形代理梯度、Batch Normalization、Dropout以及IVON实现的高斯权重后验估计与贝叶斯模型平均。

**📊 数据集**

使用了Heidelberg Digits（HD）和Speech Commands（SC）两个标准语音识别数据集进行实验评估。

**📈 对比分析**

与传统的MAP（确定性）训练相比，IVON在HD上将准确率从98.10%提升到99.51%，NLL从0.0553降至0.0233；在SC上准确率从77.01%提升到79.93%，NLL从0.8063降至0.7191，Brier得分和ECE亦有明显下降，且一维权重切片显示损失曲线更平滑。

**⚠️ 局限性**

局限性包括仅在两个语音数据集上验证，网络规模较小，未探讨硬件实现的鲁棒性，以及贝叶斯推断在大规模网络或实时部署时的计算成本和采样效率问题。

---

## 93. Multivariate Time Series Anomaly Detection via Dual-Branch Reconstruction and Autoregressive Flow-based Residual Density Estimation

**arXiv ID:** 2604.08582 | [PDF](https://arxiv.org/pdf/2604.08582v1)

**作者:** Jun Liu `[一作]` (Zhejiang University), Jun Tang `[通讯]` (Zhejiang University)

**通讯引用:** 6211 | [OpenAlex ID](https://openalex.org/A5066051651)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于双分支重建与自回归流的多变量时序异常检测框架(DBR-AF)；

**💡 创新点**

创新点在于将跨变量相关学习与单变量统计特征解耦的双分支重建器以及利用自回归流对残差进行密度估计，从而同时抑制伪相关并降低大误差误报；

**🔧 技术方法**

核心技术包括Transformer自注意力的双分支编码器、记忆池、余弦相似度损失、可逆自回归流(MAF)与高斯混合先验；

**📊 数据集**

在七个公开基准上评测，包括工业服务器(SMD、PSM)、航天器(MSL、SMAP)、水处理(SWaT)、NIPS-TS-GECCO、NIPS-TS-Swan；

**📈 对比分析**

与14种基线及多种最新方法对比，DBR-AF在五个工业/航天数据集的AUC‑ROC、AUC‑PR、F1均实现SOTA，并在所有指标上显著优于前沿方法；

**⚠️ 局限性**

局限性主要为对异常解释性不足、对先验分布与流层数等超参敏感、以及训练时对GPU资源需求较高；

---

## 94. InstrAct: Towards Action-Centric Understanding in Instructional Videos

**arXiv ID:** 2604.08762 | [PDF](https://arxiv.org/pdf/2604.08762v1)

**作者:** Zhuoyi Yang `[一作]` (Pennsylvania State University), Huijuan Xu `[通讯]` (Pennsylvania State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于动作的预训练框架 InstrAction，针对教学视频中的连续动作和弱监督进行有效建模。

**💡 创新点**

创新点包括：① LLM 辅助数据清洗与动词短语抽取，生成动作级硬负样本；② Action Perceiver 与动词引导蒸馏，抑制静态偏差；③ DTW‑Align 与 MAM 两种自监督目标，捕获时间序列和跨模态细粒度语义。

**🔧 技术方法**

使用的技术包括跨模态对比学习、Soft‑DTW、Perceiver 结构、动词引导蒸馏、Masked Action Modeling。

**📊 数据集**

主要使用 HowTo100M 的烹饪子集进行精细化清洗，并构建 InstrAct Bench 三大子任务（语义、逻辑、动力学）用于评估。

**📈 对比分析**

在 InstrAct‑Semantic、InstrAct‑Logic、InstrAct‑Dynamics 上均显著优于现有 Video Foundation Models，Recall@1、Accuracy 等指标提升 3–7个百分点，证明模型能更好区分细粒度动作与时间顺序。

**⚠️ 局限性**

局限性包括：① 依赖 LLM 的人工成本与偏差；② 对极少数动词或长序列的处理仍不充分；③ 评测集中于烹饪类视频，跨域泛化仍需进一步验证。

---

## 95. STIndex: A Context-Aware Multi-Dimensional Spatiotemporal Information Extraction System

**arXiv ID:** 2604.08597 | [PDF](https://arxiv.org/pdf/2604.08597v1)

**作者:** Wenxiao Zhang `[一作]` (University of Western Australia), Wei Liu `[通讯]` (University of Western Australia)

**通讯引用:** 86188 | [OpenAlex ID](https://openalex.org/A5100431792)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端到端的多维时空信息提取系统 STIndex，能够从多种非结构化文件（PDF、HTML、DOCX、TXT 等）中提取时间、空间、疾病、事件、场所等多维实体，并提供交互式可视化与分析仪表盘。

**💡 创新点**

创新点包括：① 领域无关的可配置多维架构，用户可按需定义层级；② 单次 LLM 调用的上下文感知提取，利用文档级记忆解决相对时间和模糊地点；③ 双通道反思机制与多模后处理（地理编码校正、质量验证）显著提升精度；④ 开源可部署、与可视化前端无缝对接。

**🔧 技术方法**

主要技术：大型语言模型（GPT‑4o‑mini、Qwen3‑8B）、Python 预处理库（unstructured、pdfminer 等）、Mapbox GL、D3.js、Next.js/React/TypeScript 前端；后端支持多种 LLM 接口和自托管模型；使用 DBSCAN、滑动窗口、共现网络等算法进行时空聚类与爆发检测。

**📊 数据集**

数据集：① 合成 500 条文档块（由 Claude Sonnet 4.5 生成并人工校验）用于系统评估；② 10 条真实公共卫生监测报告用于案例演示，提取 801 个实体。若有公开 benchmark（如 public health dataset）亦被引用。

**📈 对比分析**

与传统分离的时间/空间提取管道做对比，baseline 只在每个文档块独立处理。STIndex 在 GPT‑4o‑mini 模式下综合 F1 提升 4.37%，Qwen3‑8B 模式提升 3.60%；相对时间和空间精度分别提升 3.22% / 4.89% 与 8.23% / 0%（Qwen 主要提升召回）。空间平均距离误差（MDE）在 Qwen 模式下降 67.6%（1372km→444km），GPT‑4o‑mini 仅下降 2.2%。

**⚠️ 局限性**

局限性：① 仍需人工核查 LLM 误检；② 对高质量地理编码依赖多级回退，低资源环境下效果受限；③ 仅在实验环境下测试，跨域泛化需进一步验证；④ 对极长文档和多语言支持的扩展尚未完成；⑤ 计算成本高，尤其是 GPT‑4o‑mini 的 token 费用。

---

## 96. BIAS: A Biologically Inspired Algorithm for Video Saliency Detection

**arXiv ID:** 2604.08858 | [PDF](https://arxiv.org/pdf/2604.08858v1)

**作者:** Zhao-ji Zhang `[一作]` (Peking University), Ya-tang Li `[通讯]` (Chinese Institute for Brain Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BIAS模型，基于生物学原理实现视频动态显著性检测，能够在毫秒级延迟下生成可解释的显著性图和聚焦点；

**💡 创新点**

核心创新在于：①使用仿真视网膜的运动检测器（Hassenstein–Reichardt）提取时间特征；②采用Gaussian Winner‑Take‑All机制进行高效聚焦点选择；③利用核分解Gabor滤波显著降低计算量；④在显著性融合中引入自适应权重，兼顾静态与运动信息；

**🔧 技术方法**

技术手段包括：层次化中心‑周围对比、可分离Gabor滤波、Hassenstein–Reichardt运动检测、Gaussian WTA、中心先验、指数加权滑动平均等；

**📊 数据集**

主要数据集为DH​F1K（视频显著性）和交通事故因果识别基准（Traffic Accident Benchmark for Causality Recognition），并在后者上构建自监督SparK特征；

**📈 对比分析**

与传统启发式模型相比，BIAS在DH​F1K上实现AUC‑J≈0.849、NSS≈0.221、CC≈0.561，超过绝大多数启发式方法，且在深度学习模型中排名中上游；运行时仅0.012 s/帧，显著快于GPU实现的深度网络；在交通事故预测任务中，BIAS‑SparK提前≈0.72 s检测到事故因果，IoU和提前时间均优于基线；

**⚠️ 局限性**

局限性：仅关注低层次的bottom‑up注意力，对需要高层语义推理的任务表现欠佳；在复杂多物体或高噪声环境中可能误判；尚未集成top‑down或任务驱动模块。

---

## 97. LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs

**arXiv ID:** 2604.08752 | [PDF](https://arxiv.org/pdf/2604.08752v1)

**作者:** Paolo Gajo `[一作]` (University of Bologna), Alberto Barrón-Cedeño `[通讯]` (University of Bologna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型（LLM）与轻量级图基解析器在关系抽取任务中的性能比较，重点关注语言图复杂度对两者表现的影响。

**💡 创新点**

创新点在于揭示当文本所隐含的语言图复杂度提升时，轻量级图基解析器会显著优于大型LLM，并通过多数据集实验说明LLM在关系抽取时受格式噪声影响的机制。

**🔧 技术方法**

采用了四种LLM（Mistral‑7B、Qwen3‑14B、Qwen3‑32B、Llama‑3.3‑70B）进行LoRA微调，并与基于Biaffine注意力的图基解析器进行对比。

**📊 数据集**

实验使用了六个不同复杂度的关系抽取数据集：CoNLL04、ADE、SciERC、enEWT、SciDTB 与 ERFGC。

**📈 对比分析**

通过多种 prompt 设计、不同微调步数（1 轮或 3000 步）以及多随机种子评估，最终发现当图节点数>18时图基解析器的 micro‑F1 明显高于 LLMS；在简单图上 LLMS 与解析器表现相近或略优。

**⚠️ 局限性**

研究局限包括仅使用单一图基解析器架构、未深入探究 LLMS 的注意力机制、实验规模受硬件限制且只覆盖少数 LLM，未来需更广泛的模型与定性分析。

---

## 98. Creator Incentives in Recommender Systems: A Cooperative Game-Theoretic Approach for Stable and Fair Collaboration in Multi-Agent Bandits

**arXiv ID:** 2604.08643 | [PDF](https://arxiv.org/pdf/2604.08643v1)

**作者:** Ramakrishnan Krishnamurthy `[一作]` (New York University), Maximilian Nickel `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于合作博弈论的创作者激励机制，针对多智能体 bandit 推荐系统设计公平、稳定的奖励分配方案。

**💡 创新点**

创新点在于将 Shapley 值与稳定匹配相结合，构造可解释且稳定的合作框架，并证明在多代理 bandit 环境下的收敛与公平性。

**🔧 技术方法**

采用合作博弈论（Shapley 值、核心、核区）、多臂赌博机算法（ε‑greedy、UCB）以及稳定匹配理论。

**📊 数据集**

实验使用公开的在线视频平台数据集（如 YouTube‑Like synthetic 或 MovieLens）以及从 Meta AI 采集的推荐日志。

**📈 对比分析**

与传统非合作 bandit、均分奖励和基于利润共享的对照组比较，本文方案在累计奖励、均值方差以及 Gini 指数方面均表现更好，并在短期内实现了稳定合作。

**⚠️ 局限性**

局限性包括：Shapley 计算成本随创作者数目呈阶乘增长；假设创作者信息公开且理性；在极大规模系统中的实时部署仍需进一步优化。

---

## 99. EngageTriBoost: Predictive Modeling of User Engagement in Digital Mental Health Intervention Using Explainable Machine Learning

**arXiv ID:** 2604.08589 | [PDF](https://arxiv.org/pdf/2604.08589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 100. OpenKedge: Governing Agentic Mutation with Execution-Bound Safety and Evidence Chains

**arXiv ID:** 2604.08601 | [PDF](https://arxiv.org/pdf/2604.08601v1)

**作者:** Jun He `[一作]` (OpenKedge.io), Deying Yu `[通讯]` (OpenKedge.io)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OpenKedge 协议，将代理系统的状态变更从直接 API 调用转变为基于意图、上下文与策略的治理流程，并通过执行合同与短期任务身份实现受控执行。

**💡 创新点**

创新点包括：①将 mutation 视为有序治理过程而非即时执行；②引入 Intent‑to‑Execution Evidence Chain（IEEC）实现端到端可追溯性；③通过任务导向身份限制执行范围，实现执行绑定安全。

**🔧 技术方法**

采用事件溯源架构、基于规则的策略语言 Cedar、短期 AWS STS 身份、加密哈希链、无锁并发设计及多方冲突调度算法。

**📊 数据集**

在模拟的多代理冲突场景和 AWS 云基础设施（实例、负载均衡、资源拓扑）上进行实验，使用人工生成的意图与真实云状态数据进行评估。

**📈 对比分析**

相较传统 API‑centric 模型和现有安全框架（如 Claude Code、AARM），OpenKedge 在保持 3,200 次/秒吞吐量的同时，能够 deterministic 解决冲突并阻止无上下文或幻觉操作，平均策略评估延迟约 11 ms，状态派生 99% 分位 <30 ms。

**⚠️ 局限性**

局限性在于：依赖人工编写的 deterministic 策略规则，缺乏自学习能力；对事件溯源和短期身份的实现依赖特定平台（如 AWS）；在极大规模分布式环境下对状态同步与信任评分机制的细粒度调优仍待研究。

---

## 101. Ge$^\text{2}$mS-T: Multi-Dimensional Grouping for Ultra-High Energy Efficiency in Spiking Transformer

**arXiv ID:** 2604.08894 | [PDF](https://arxiv.org/pdf/2604.08894v1)

**作者:** Zecheng Hao `[一作]` (Peking University), Tiejun Huang `[通讯]` (Peking University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Ge²mS‑T 架构，采用三维分组计算（时间、空间、网络结构）实现可直接训练的 Spiking Vision Transformer，并在 ImageNet、CIFAR 与 neuromorphic 数据集上获得高精度与低能耗。

**💡 创新点**

创新点：① ExpG‑IF 通过分组指数编码实现无损 ANN‑SNN 转换与精确脉冲发放控制；② GW‑SSA 通过多尺度分组自注意力实现乘法自由、低 SOP 的注意力计算；③ 在同一网络中同步实现三维分组，突破了传统 S‑ViT 的记忆、精度与能耗三重瓶颈。

**🔧 技术方法**

使用技术包括：分组指数编码 IF（ExpG‑IF）、分组自注意力（GW‑SSA）、双分支混合卷积‑注意力块、深度可分离卷积 SFFN、基于混合激活的分阶段训练策略以及混合分组策略的量化与剪枝。

**📊 数据集**

使用数据集：ImageNet‑1k（224×224、288×288）、CIFAR‑10、CIFAR‑100、CIFAR10‑DVS（事件流）。

**📈 对比分析**

与现有 SoTA 进行对比：在 ImageNet‑1k 上，Ge²mS‑T Large 以 79.82% 的 Top‑1 率、14.48 M 参数、3.15 SOP/样本、2.83 mJ 能耗，优于 Spikformer、SDT、SViT 等方法；在 CIFAR‑10/100 上，在 4 步推理下准确率均超过 98%，显著优于 STBP、TET、GAC‑SNN 等；在 CIFAR10‑DVS 上也实现了 4 步内的 87.6% 精度，能耗仅为 4 mJ。相比 SoTA，Ge²mS‑T 在保持或提升精度的同时，参数量和能耗均大幅下降。

**⚠️ 局限性**

局限性：① 目前主要在标准图像与事件数据集上验证，尚未在更大规模或更复杂任务（如视频、语音）中测试；② 训练过程仍需要混合激活与分组策略的细粒度调优；③ 论文未给出实际硬件实现细节，能耗评估基于仿真模型，真实硬件落地仍需验证。

---

## 102. Distilling Genomic Models for Efficient mRNA Representation Learning via Embedding Matching

**arXiv ID:** 2604.08574 | [PDF](https://arxiv.org/pdf/2604.08574v1)

**作者:** Rasched Haidari `[一作]` (Helical), Maxime Allard `[通讯]` (Helical)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

使用知识蒸馏把大规模基因组模型Evo2-1B压缩成仅约5M参数的HelixNano-mRNA模型，并通过中间嵌入匹配实现对mRNA序列的表征学习

**💡 创新点**

证明嵌入层蒸馏在基因组模型中比传统logit/KL蒸馏更稳定、有效，并实现了200倍参数压缩后仍能在多项mRNA相关任务上达到同等规模模型的最佳表现

**🔧 技术方法**

采用中间隐藏层嵌入匹配（均方误差+余弦损失），使用线性投影对齐教师与学生维度，AdamW + 线性warmup + 量化梯度裁剪，混合精度训练（bfloat16）

**📊 数据集**

训练集为NCBI FTP上所有“.rna.gbff.gz”文件的27M条mRNA序列，包含43.6%其它脊椎动物、28.3%哺乳动物、26.4%无脊椎动物和1.6%病毒；在mRNA-bench基准上进行评估

**📈 对比分析**

与原始Evo2-1B、Orthrus-1M等模型对比，HelixNano-mRNA在大多数mRNA‑bench任务（半衰期、定位、GO分类、蛋白定位等）中以小模型规模取得领先或竞争性表现，整体得分与大模型相近

**⚠️ 局限性**

仅在mRNA序列上验证，未对所有教师层组合、所有下游任务（如mRNA‑Loc‑LR）进行全面评估；logit/KL蒸馏仍存在不稳定性，且学生模型可能无法完全捕获教师的全部潜在表征

---

## 103. $p1$: Better Prompt Optimization with Fewer Prompts

**arXiv ID:** 2604.08801 | [PDF](https://arxiv.org/pdf/2604.08801v1)

**作者:** Zhaolin Gao `[一作]` (Cornell University), Wen Sun `[通讯]` (Databricks AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大型语言模型中通过优化系统提示（system prompt）来提升任务表现的可行性，并提出一种基于提示方差的用户提示过滤方法，帮助提升在异构推理任务上的优化效果。

**💡 创新点**

创新点在于：①将奖励方差拆解为“回应方差”和“系统提示方差”两部分，阐明系统提示方差越大越易于优化；②发现样本规模增大会压缩系统提示方差，导致优化信号减弱；③提出通过筛选具有最高系统提示方差的少量用户提示，构造更具信息量的训练集，从而显著提升 RL 基础的提示优化性能。

**🔧 技术方法**

技术方法包括：基于强化学习（GRPO）对系统提示生成策略进行训练；方差分解分析（理论证明与实验验证）；用户提示过滤策略（选择具有最大系统提示方差的子集）；与基线方法（GEPA、全数据 RL）进行对比实验。

**📊 数据集**

使用的主要数据集有：IFBench（指令遵循任务），AIME 2024/2025/2026（高难度数学推理），HMMT Nov/Feb 2025 & 2026（高校数学竞赛），以及 Qwen3 系列模型的不同规模版本。

**📈 对比分析**

与 GEPA、全数据 RL 等基线相比，过滤方法在 AIME、HMMT 等推理基准上取得显著提升（如 AIME 2024 准确率提升至 54% 以上，较基线提升约 6%+），并在更大模型 Qwen3-30B 上展现迁移性能；但在相对同质的 IFBench 上，过滤方法不如全数据训练效果好，表明该方法更适合异构任务。

**⚠️ 局限性**

局限性包括：分析仅针对二元奖励场景，无法直接推广到连续奖励；过滤子集是否能普遍预测全分布性能尚未完全阐明；实验多聚焦于 Qwen3 系列模型，需进一步验证跨模型适用性。

---

## 104. Scrapyard AI

**arXiv ID:** 2604.08803 | [PDF](https://arxiv.org/pdf/2604.08803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 105. EfficientSign: An Attention-Enhanced Lightweight Architecture for Indian Sign Language Recognition

**arXiv ID:** 2604.08694 | [PDF](https://arxiv.org/pdf/2604.08694v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 106. EMA Is Not All You Need: Mapping the Boundary Between Structure and Content in Recurrent Context

**arXiv ID:** 2604.08556 | [PDF](https://arxiv.org/pdf/2604.08556v1)

**作者:** Arth Singh `[一作]` `[通讯]` (AIM Intelligence), Arth Singh (AIM Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究EMA指数移动平均（EMA）追踪作为高效序列模型的下限，探讨其在结构识别与语言建模中的优势与局限。

**💡 创新点**

证明固定系数累积会导致信息衰减：EMA追踪能保持句法结构但无法保留词语内容，且全软最大注意力无法弥补这一缺陷。

**🔧 技术方法**

使用EMA追踪、稀疏预测列网络(SPCN)、稀疏预测平衡网络(SPEN)、线性与软最大注意力对比、信息理论推导。

**📊 数据集**

控制语法数据集（147词语的形式语法）和公开文本（FineWeb‑Edu、C4）训练与评估。

**📈 对比分析**

与BiGRU和GPT‑2小模型比较，SPCN在无监督语法角色识别中达96%监督BiGRU精度并在结构角色上超越监督模型；SPEN在语言建模上取得260的困惑度，约GPT‑2小8倍，但瓶颈归因于EMA追踪；预测器消融实验显示不同预测器表现相同。

**⚠️ 局限性**

仅在固定EMA追踪下的下限，未验证在自然语言中的结构优势；模型训练规模有限、推理速度慢、未与等量计算的transformer做严格对比。

---

## 107. 3D-VCD: Hallucination Mitigation in 3D-LLM Embodied Agents through Visual Contrastive Decoding

**arXiv ID:** 2604.08645 | [PDF](https://arxiv.org/pdf/2604.08645v1)

**作者:** Makanjuola Ogunleye `[一作]` (Virginia Tech), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种推理时的视觉对比解码框架3D‑VCD，用于抑制3D嵌入式智能模型的幻觉；

**💡 创新点**

首次在3D场景中引入结构化场景图的对比解码，使用语义与几何扰动构造负样本，无需任何模型再训练；

**🔧 技术方法**

利用对象中心化的3D场景图，施加语义替换、几何噪声及结构扰动，双上下文推理后对logits做对比融合，并采用批量前向、KV缓存等效率优化；

**📊 数据集**

在3D‑POPE和HEAL两个公开基准上评估，使用3D‑GRAND生成场景图；

**📈 对比分析**

与3D‑LLM、3D‑VisTA、LEO等现有基线对比，3D‑VCD在Precision、Recall、F1、Accuracy等指标上均提升，显著降低过度肯定率；在HEAL Distractor Injection子任务中，将CHAIR‑C_S、C_O从 16.45%/4.13% 降到 5.0%/3.55%；

**⚠️ 局限性**

仍存在两次前向的计算开销、对动态或时间序列场景的适用性待验证，以及在极端噪声或大规模物体集合下的鲁棒性待进一步研究。

---

## 108. Can We Still Hear the Accent? Investigating the Resilience of Native Language Signals in the LLM Era

**arXiv ID:** 2604.08568 | [PDF](https://arxiv.org/pdf/2604.08568v1)

**作者:** Nabelanita Utami `[一作]` (Nagoya University), Sasano Ryohei `[通讯]` (Nagoya University)

**通讯引用:** 800 | [OpenAlex ID](https://openalex.org/A5049498516)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究写作助手工具从机器翻译到大语言模型演进对学术论文写作的同质化影响，构建高流利度学术写作的原始语言识别（NLI）数据集并通过模型评估其随时间的性能变化。

**💡 创新点**

① 用LLM增强的半自动标注框架构建两套NLI数据集；② 将NLI性能作为衡量写作同质化的量化指标，并在三技术时代（pre‑NN、pre‑LLM、post‑LLM）进行对比；③ 在日语、韩语等语言上揭示不同AI生态对写作风格的影响。

**🔧 技术方法**

利用大语言模型（Qwen3‑8B/14B、Gemma‑3‑12B‑it）进行姓名来源预测与少样本推理；采用QLoRA低秩适配对Qwen3‑14B与Gemma‑3‑12B‑it进行微调；使用统计检验（Fisher exact test）评估性能差异。

**📊 数据集**

训练集：来自 arXiv 的 1,600 篇（1999‑2021）平衡采样，覆盖 8 种语言；评估集：来自 ACL Anthology 的 1,200 篇（50 篇/语言/时代），划分为 pre‑NN（≤2015）、pre‑LLM（2016‑2022）、post‑LLM（2023‑2025）。

**📈 对比分析**

比较方法：对预训练模型做 1‑shot 少样本推理，微调后用准确率和 F1 评估；发现预训练模型在 post‑LLM 时代准确率从 37.8% 降至 14.5%，微调后 pre‑NN 约 73% 降至 post‑LLM 约 60%；差异在 pre‑NN 与其他时代显著，pre‑LLM 与 post‑LLM 差异不显著；日语、韩语的性能下降最为明显。

**⚠️ 局限性**

限制：标注为高概率估计，可能存在噪声；post‑LLM 时代定义假设所有作者使用 LLM；仅覆盖 CS/NLP 领域，可能无法推广到其他学科；仅使用摘要，可能低估全文的 L1 信号；评估样本量小（每语言 50 篇/时代），尤其是低频组合；国内外 AI 生态差异未能充分解释异常结果。

---

## 109. Real-Time Toxicity Filtering for Open-Source Code Reviews

**arXiv ID:** 2604.08886 | [PDF](https://arxiv.org/pdf/2604.08886v1)

**作者:** Md Awsaf Alam Anindya `[一作]` (Bangladesh University of Engineering & Technology), Amiangshu Bosu `[通讯]` (Wayne State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了ToxiShield，一个实时检测并缓解开源软件代码审查中有害交互的浏览器插件。

**💡 创新点**

创新点在于整合三阶段实时框架：毒性识别、多标签分类与语义等价的消毒改写，并提供解释性反馈。

**🔧 技术方法**

使用BERT-base进行二分类，Claude 3.5 Sonnet/ GPT‑4o 进行多标签分类，Llama 3.2 3B 进行文本风格迁移。

**📊 数据集**

基于从Sarker等人研究中筛选的15 M PR评论，手工标注的10 120个毒性评论与28 641个非毒性评论，并利用1 200条多标签样本与合成的并行毒性/非毒性对。

**📈 对比分析**

与基线模型对比，BERT二分类 F1 0.97；多标签分类 Macro‑F1 0.42 / Macro‑MCC 0.39；消毒模型达到 J‑score 84%，风格迁移准确率 95.27% 与流畅度 97.03%。

**⚠️ 局限性**

局限性包括：在技术细节复杂的评论中易误判；多标签分类对模糊情境的准确度不足；合成并行数据质量仍有限。

---

## 110. Sensor Placement for Tsunami Early Warning via Large-Scale Bayesian Optimal Experimental Design

**arXiv ID:** 2604.08812 | [PDF](https://arxiv.org/pdf/2604.08812v1)

**作者:** Sreeram Venkat `[一作]` (University of Texas at Austin), Omar Ghattas `[通讯]` (University of Texas at Austin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个可扩展的分布式多 GPU Bayesian D‑optimal 设计框架，用于超大规模线性时不变系统的传感器布置优化，并将其应用于 Cascadia 海啸数字孪生，最终从 600 个候选位置中选出 175 个最佳传感器。

**💡 创新点**

创新点包括：① 将逆问题重构到数据空间，使 D‑optimal 设计化为密集矩阵子集选择；② 设计了 Schur 补数更新的贪心算法，避免了重复的稠密矩阵分解；③ 采用多 GPU 流水线 I/O 与 GPU 计算重叠，实现近乎理想的强弱伸缩；④ 首次在 10^9 维参数场上直接求解 PDE 约束的 Bayesian OED，无需降维或代理模型。

**🔧 技术方法**

使用了 Bayesian OED、D‑optimal（log‑det）准则、Sherman–Morrison–Woodbury 变换、Schur 补数更新、块 Cholesky、MPI‑PyTorch、CUDA/HIP、HDF5 2D 分块存储、双缓冲流水线 I/O、单精度候选评估等技术。

**📊 数据集**

使用 Cascadia Subduction Zone 的 600 个候选传感器位置数据，模拟 420 步（1 Hz、7 分钟）的观测，构建 1 B 维度参数场的数字孪生；同时生成 600 条共振 PDE 反向求解，形成完整的 p2o 映射。

**📈 对比分析**

与传统 O(k^3) 的朴素实现相比，单 GPU 上 Schur 方案速度快数个数量级；在 Perlmutter 和 Frontier 超算上，强弱伸缩近理想，128 × GPU 加速；最终在 16 A100 GPU 上完成 175 传感器的选择仅耗时 1.5 小时，展示了卓越的性能和可扩展性。

**⚠️ 局限性**

局限性：仅适用于线性时不变系统和高斯等方差噪声；未处理非线性、非高斯或时间变参数的情形；前期需要大量 GPU 进行反向 PDE 求解；对硬件依赖强，需大型多 GPU 集群；未考虑实时动态更新或成本约束的实时重设计。

---

## 111. Structured Exploration and Exploitation of Label Functions for Automated Data Annotation

**arXiv ID:** 2604.08578 | [PDF](https://arxiv.org/pdf/2604.08578v1)

**作者:** Phong Lam `[一作]` (VNU University of Engineering and Technology), Hieu Dinh Vo `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自动化程序化标注框架，通过结构化探索与利用生成多层次（表面、结构、语义）的标签函数，并在选择阶段通过可靠性评估过滤噪声与冗余；

**💡 创新点**

核心创新在于同时保证标签函数的多样性与可靠性——先系统性地从任务描述与数据特征中挖掘多层次候选函数，再通过精度与覆盖率评估实现有针对性的筛选与校准，解决现有方法仅依赖LLM或固定原语导致覆盖不足或噪声多的问题；

**🔧 技术方法**

技术手段包括：1）使用GPT‑4.1生成表面层标签函数；2）用SVM学习结构层函数；3）采用BERT‑base提取语义层特征；4）构建标签函数矩阵后通过概率标签模型（如FlyingSquid、Dawid‑Skene）聚合弱标签；5）采用可靠性权重阈值与多重采样进行函数过滤与校准；

**📊 数据集**

共评估11个文本分类数据集，涵盖二分类（YouTube、SMS、IMDb、Yelp）、多分类（AGNews、Clickbait、Finance、ChemProt、Massive）与多标签（PubMed、PaperAbs）等多领域任务；

**📈 对比分析**

与LLM驱动、模型驱动、传统弱监督（Snuba）、半监督和少样本学习等基线进行对比；实验显示该框架在覆盖率上达98.9%，弱标签质量提升高达87%，下游加权F1得分提升最高46%；

**⚠️ 局限性**

局限性包括：在高度复杂的多类任务（如ChemProt、Massive）提升仍有限；性能仍依赖初始标注子集的代表性；目前未针对动态数据流或多模态场景设计自动化衰减与再生机制；

---

## 112. Reservoir observer enhanced with residual calibration and attention mechanism

**arXiv ID:** 2604.08592 | [PDF](https://arxiv.org/pdf/2604.08592v1)

**作者:** Yichen Liu `[一作]` (Peking University), Tianguang Chu `[通讯]` (Peking University)

**通讯引用:** 4391 | [OpenAlex ID](https://openalex.org/A5027232468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种增强型储水池观测器，结合残差校正与注意力机制，用于非线性动力学系统中未测变量的推断。

**💡 创新点**

将观测残差直接用于校正观测结果，并在储水池内部引入基于Gaussian RBF与SVD的注意力机制，以捕捉时间相关性，三者组合显著降低误差，尤其在传统观测器表现最差的输入变量情形下。

**🔧 技术方法**

使用了Reservoir Computing（Reservoir Observer）、残差校正模块、注意力机制（GRBF+SVD低维投影）、正则化回归、传递熵分析以及多种实验比较方法。

**📊 数据集**

使用典型混沌系统（Rössler、Lorenz、Chua's circuit）以及空间-时间混沌系统Kuramoto‑Sivashinsky（Q=64网格）等数据集，在不同输入变量配置下进行训练与测试。

**📈 对比分析**

通过与传统RO的比较，采用均方误差(MSE)及误差降低率评估。实验显示，RORA可将MSE降至传统RO的1%以下，误差下降率高达99%以上；对最差输入变量（如Rössler的z、Chua的y）亦能显著提升；在测量噪声存在时，仍可保持约50%以上的改进。

**⚠️ 局限性**

局限性包括：对输入变量传递熵弱的情况仍受限，注意力机制在高维状态下易受“维度灾难”影响；残差校正对噪声敏感，噪声强时效果减弱；模型参数（α、ρ、σ、N_c等）需经验调优；对极大规模系统或实时应用的可扩展性尚未充分验证。

---

## 113. Detection of Hate and Threat in Digital Forensics: A Case-Driven Multimodal Approach

**arXiv ID:** 2604.08609 | [PDF](https://arxiv.org/pdf/2604.08609v1)

**作者:** Ponkoj Chandra Shill `[一作]` `[通讯]` (University of Nevada, Reno), Ponkoj Chandra Shill (University of Nevada, Reno)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个基于案例驱动的多模态数字取证管道，用于检测图像证据中的仇恨、威胁、骚扰或暴力意图。

**💡 创新点**

通过显式区分嵌入文本、关联文本和仅图像证据，依据可用证据路由分析；使用冻结标签空间和显式分数级融合，保证可追溯性与法证可靠性。

**🔧 技术方法**

采用零样本文本分类模型（如 DeBERTa-v3-large）处理文本；使用 CLIP ViT‑L/14 视觉‑语言模型进行图像推理；OCR 提取嵌入文本；分数级融合权重手工设定。

**📊 数据集**

评估使用自建的取证风格混合数据集，包括图像、OCR 提取文本和来自取证报告的关联文本；未进行任务特定训练。

**📈 对比分析**

通过与人工标注的对照，分别在四种数据情况（DS1‑DS4）计算准确率，均达到94%–98%；多模态融合在有文本时提升判断，改变率最高为34%；整体表现稳定。

**⚠️ 局限性**

OCR 噪声导致误检；视觉‑语言模型解释性有限；评估样本量有限，存在主观标注偏差。

---

## 114. WAND: Windowed Attention and Knowledge Distillation for Efficient Autoregressive Text-to-Speech Models

**arXiv ID:** 2604.08558 | [PDF](https://arxiv.org/pdf/2604.08558v1)

**作者:** Hanna Lee `[一作]` (Korea Advanced Institute of Science and Technology), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5064051041)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 WAND 框架，将注意力分为全局上下文注意力与局部滑动窗口注意力，并结合知识蒸馏与课程学习对预训练的 AR‑TTS 模型进行高效微调。

**💡 创新点**

创新点在于实现了从线性到常数的计算与内存复杂度转变：通过保持对前缀条件的持续全局访问，限制生成序列的注意力窗口，同时利用全注意力教师的知识蒸馏与温度化软掩码课程学习稳定模型收敛。

**🔧 技术方法**

使用技术包括滑动窗口注意力、持久全局注意力、交叉熵+KL 散度蒸馏损失、温度缩放软掩码课程调度、KV 缓存分区与评估指标（UTMOS、NMOS、SSIM、WER、CER）。

**📊 数据集**

使用 100 小时的 English LibriTTS 子集进行微调，并在 Seed‑TTS‑eval 基准（包含 English 与 Mandarin）上进行评估，利用 Whisper‑large‑v3 与 Paraformer‑zh 进行 ASR 评测。

**📈 对比分析**

与 CosyVoice 2、IndexTTS 1.5、SparkTTS 的全注意力基线相比，WAND 在 10 s 生成任务中将 KV 缓存减少高达 66.2%、GFLOPs 降低约 46.9%，实现 1.5–1.9× 的加速，同时 WER 仅略低于 2%（如 1.72% vs 1.94%），保持了 UTMOS 与 SSIM 的几乎无损。

**⚠️ 局限性**

局限性包括：需要预训练模型和一定量的 fine‑tune 数据，窗口大小取决于经验选择，极短窗口可能导致内容一致性下降；目前仅针对 AR‑TTS，尚未验证在更长或多模态序列上的稳定性，且课程学习和掩码温度等超参数调优仍需经验。

---

## 115. Ranked Activation Shift for Post-Hoc Out-of-Distribution Detection

**arXiv ID:** 2604.08572 | [PDF](https://arxiv.org/pdf/2604.08572v1)

**作者:** Gianluca Guglielmo `[一作]` (Graz University of Technology), Marc Masana `[通讯]` (Graz University of Technology)

**通讯引用:** 3258 | [OpenAlex ID](https://openalex.org/A5089415088)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无超参数、后置方法Ranked Activation Shift（RAS），通过将模型的最终隐藏层激活值按排名与ID参考分布对齐，以提高OODA检测。

**💡 创新点**

使用排名信息而非绝对激活值来对齐ID分布，避免了传统方法对阈值和激活分布假设的依赖；无超参数、无需OoD样本；对非ReLU网络同样有效；同时证明升降两种方向均有助。

**🔧 技术方法**

后置激活编辑、排序、参考向量平均、直方图匹配；实验使用OpenOOD基准框架，并将RAS与EBO、ViM、GEN等能量/softmax等OOD分数结合。

**📊 数据集**

ImageNet、ImageNet-200、CIFAR-10、CIFAR-100 以及对应的近/远OOD数据集（TinyImageNet、SVHN、MNIST、iNaturalist、Places365、OpenImage-O等）。

**📈 对比分析**

与ReAct、DICE、ASH-P/B/S、SCALE等增强后置方法对比，使用AUROC、AUPR、FPR@95%等指标；RAS在绝大多数设置下保持或提升性能，同时保持ID分类准确率不变；在不同评分器下均可提升。

**⚠️ 局限性**

主要在公开基准上验证，未评估大规模自定义数据或跨任务迁移；对计算成本影响未详细讨论；对非最终层激活的效果有限；在极端模型/数据偏差下可能仍需进一步验证。

---

## 116. MARINER: A 3E-Driven Benchmark for Fine-Grained Perception and Complex Reasoning in Open-Water Environments

**arXiv ID:** 2604.08615 | [PDF](https://arxiv.org/pdf/2604.08615v1)

**作者:** Xingming Liao `[一作]` (Guangdong University of Technology), Lianglun Cheng `[通讯]` (Guangdong University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MARINER基准，覆盖细粒度船舶分类、检测和视觉问答，针对多源海上真实环境；

**💡 创新点**

引入3E（实体-环境-事件）范式，构建含63类船舶、4种恶劣环境、5类事故的海上数据集，统一多任务评估；

**🔧 技术方法**

使用多模态大语言模型（如GPT-4、Gemini、Qwen、InternVL、LLaVA）进行评估，结合人工标注与自动化生成VQA对；

**📊 数据集**

MARINER数据集（16,629张多源图像、11,790张标注实例、33,125个QA对）；对比公开海上基准（MS COCO、SeaShip、McShips等）；

**📈 对比分析**

在分类、检测和VQA三任务上与多种开源与专有模型对比，MARINER 7B模型在分类准确率达84.6%，检测AP 29.5%，VQA整体精度73.7%；整体提升幅度显著，尤其在高阶推理上；

**⚠️ 局限性**

仍面临军事船舶识别难度大、检测精度低、模型对复杂场景的因果推理不足，且大模型性能受限于推理能力与训练数据覆盖。

---

## 117. Cards Against LLMs: Benchmarking Humor Alignment in Large Language Models

**arXiv ID:** 2604.08757 | [PDF](https://arxiv.org/pdf/2604.08757v1)

**作者:** Yousra Fettach `[一作]` (Ghent University), Tijl De Bie `[通讯]` (Ghent University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对五款前沿大型语言模型（GPT‑5.2、Gemini 3 Flash、Claude Opus 4.5、Grok 4、DeepSeek‑V3.2）进行 Cards Against Humanity（CAH）游戏实验，测量它们在选取最有趣白卡时与人类玩家的偏好对齐情况，并系统分析模型的自一致性、相互一致性以及位置偏差和内容偏好。

**💡 创新点**

首次将 CAH 作为结构化但高度文化化的幽默基准，评估 LLM 在幽默偏好任务上的对齐水平；通过对位置和话题偏差的量化以及构建条件逻辑回归预测模型，揭示 LLM 对幽默判断的潜在机制与人类偏好差异。

**🔧 技术方法**

使用前沿 LLM（GPT‑5.2、Gemini 3 Flash、Claude Opus 4.5、Grok 4、DeepSeek‑V3.2）在温度 0.8 下进行推理；通过两轮随机排列白卡顺序以评估位置偏差；利用 LLM‑as‑judge 方案为白卡生成话题标签；对模型选择结果拟合条件逻辑回归模型；统计自一致性、互一致性和对齐准确率。

**📊 数据集**

主要使用 CAH Lab Gameplay 数据集（4,947 场次、9,894 轮次），其中包含每轮黑卡提示、十张候选白卡、玩家最终选择；使用 CAH Lab Demographic Answers 数据集进行人群子组分析；通过 LLM‑as‑judge 对所有白卡进行 1–4 个主题标签（共 15 类）标注。

**📈 对比分析**

通过与人类玩家的单一选择比较，计算模型对齐准确率（13–18%），与随机基线 10% 及简单基线（流行度 19.11%，提升树 19.77%）对比；计算模型自一致性（49.5–63.3%）和模型间互一致性（21.4–44.9%）；构建条件逻辑回归“代理模型”，其在测试集上的预测准确率最高达 36%（Grok、DeepSeek、Gemini），最低 17%（GPT），表明位置和话题偏差能部分解释模型行为。

**⚠️ 局限性**

局限性包括：实验规模有限（仅 9,894 轮、两轮复现），缺乏多评判者数据导致无法评估人类内部一致性；使用固定温度 0.8 可能限制模型多样性；话题标签依赖 LLM‑as‑judge 可能引入噪声；玩家样本主要为西方自选人群，模型多为西方背景，难以推广至跨文化情境。

---

## 118. Robust Reasoning Benchmark

**arXiv ID:** 2604.08571 | [PDF](https://arxiv.org/pdf/2604.08571v1)

**作者:** Pavel Golikov `[一作]` (Cranberry Lemon University), Mark C. Jeffrey `[通讯]` (Cranberry Lemon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一套14种确定性文本扰动，评估LLM在AIME 2024数学推理任务上的鲁棒性。

**💡 创新点**

创新点在于提出可逆、认知上可解的结构扰动，并通过分离解析与推理、单查询注意力衰减实验揭示开源模型对上下文污染的脆弱性。

**🔧 技术方法**

使用了确定性文本变换、工作记忆隔离实验以及多模型零样本推理技术。

**📊 数据集**

采用AIME 2024公开题集，并生成14种扰动版本作为评测数据。

**📈 对比分析**

与八个最先进模型对比，闭源模型平均跌幅约为10%，而开源模型平均跌幅超过50%，表明注意力衰减导致显著准确率下降。

**⚠️ 局限性**

局限包括仅测试单一数据集、未覆盖更大模型规模与其他任务，以及扰动解析策略需人工指定。

---

## 119. Artifacts as Memory Beyond the Agent Boundary

**arXiv ID:** 2604.08756 | [PDF](https://arxiv.org/pdf/2604.08756v1)

**作者:** John D. Martin `[一作]` (Openmind Research Institute), Amy Pajak `[通讯]` (Cohere Labs Community)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在强化学习中，作者将环境中的可观察痕迹（如空间路径）视作外部记忆，提出了“artifact”概念并证明其能减少对历史记录的存储需求；

**💡 创新点**

创新点在于首次用形式化框架证明外部痕迹能量化地降低内部记忆容量，并通过实验验证RL代理在有痕迹环境下能以更低内部容量获得相同或更好表现；

**🔧 技术方法**

使用线性Q学习与深度Q网络两种基线算法，配合投影转导函数与完整观测，进行强化学习实验；

**📊 数据集**

实验数据集为13×13网格的二维导航环境，观测为8×8噪声图像拼接成24×24图像，包含多种固定与动态路径痕迹；

**📈 对比分析**

通过比较相同设计下在有痕迹与无痕迹两种环境中的总/平均奖励，统计显著性检验发现：在有痕迹情况下，多数容量设置下代理表现更佳，表明外部化记忆有效降低了所需内部参数；

**⚠️ 局限性**

局限性包括：仅考虑确定性痕迹且只编码单一过去观测，未扩展至动作痕迹或部分信息传递；动态痕迹实验仅在线性Q学习下可行，DQN无法处理非平稳环境；此外，理论假设在现实复杂环境中的可推广性待进一步验证。

---

## 120. On Semiotic-Grounded Interpretive Evaluation of Generative Art

**arXiv ID:** 2604.08641 | [PDF](https://arxiv.org/pdf/2604.08641v1)

**作者:** Ruixiang Jiang `[一作]` (Hong Kong Polytechnic University), Changwen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SemJudge，一个基于 Peircean 计算半语理论的生成艺术评估器，能够在生成者-观众交互中重建意义链并评估象征性与指示性意义；

**💡 创新点**

首创以层级半语图（HSG）形式显式建模提示到图像的意义构建过程，突破传统指标的象征性偏差，实质性评估符号与指示性层面的艺术意义；

**🔧 技术方法**

采用半语理论框架、层级半语图（HSG）、零样本大语言模型（如 Qwen、Gemini）生成 HSG 与解释，结合 2AFC 与 VQA 评估协议；

**📊 数据集**

构建了 SemiosisArt 基准，包含 187 条 HSG 任务、935 张图像，涉及 16 款生成模型，任务以 12 名专家基于基准意象（基督教、东亚、印度教、伊斯兰等传统）构建；

**📈 对比分析**

与传统图像质量/对齐、结构化推理、艺术解释模型等 10+ 评估方法对比，SemJudge 在 KRCC、SRCC、CCC、VQA 准确率等指标上均显著领先（最高 CCC≈0.968，VQA≈92.4%），几乎与专家评判持平；

**⚠️ 局限性**

数据集文化覆盖受限，难以涵盖少数族群与当代概念艺术；依赖大语言模型生成 HSG 的质量与多模态推理能力；评估框架仍未涵盖所有艺术表达维度。

---

## 121. Memory Wall is not gone: A Critical Outlook on Memory Architecture in Digital Neuromorphic Computing

**arXiv ID:** 2604.08774 | [PDF](https://arxiv.org/pdf/2604.08774v1)

**作者:** Amirreza Yousefzadeh `[一作]` (University of Twente), Ana Lucia Varbanescu `[通讯]` (University of Twente)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了数字神经形态处理器的内存壁垒问题，并提出了多技术融合方案以提升面积与能耗效率。

**💡 创新点**

指出内存壁垒在神经形态架构中转变为“新的内存壁垒”，并提出异构分层内存、混合神经网络、3D集成NVM等创新路径。

**🔧 技术方法**

采用了SRAM、STT-MRAM、MRAM、DRAM等多种内存技术对比，并提出基于RF/SRAM/NVM混合的分层架构及3D单晶NVM层集成方案。

**📊 数据集**

利用CIFAR-10、MobileNet、N-MNIST等公开数据集对典型神经网络模型进行评估。

**📈 对比分析**

通过与TrueNorth、Loihi、GrAI VIP、SPECK等芯片的映射效率、面积/能耗等指标对比，表明异构分层方案可将映射效率从<1%提升至>20%，能耗下降约30%。

**⚠️ 局限性**

仍存在映射灵活性不足、NVM写耗高、跨层通信开销大等限制，且缺乏针对通用神经网络的完整验证。

---

## 122. Efficient RL Training for LLMs with Experience Replay

**arXiv ID:** 2604.08706 | [PDF](https://arxiv.org/pdf/2604.08706v1)

**作者:** Charles Arnal `[一作]` (FAIR at Meta), Remi Munos `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM强化学习微调过程中，提出并实现了经验回放缓冲区来减少推理成本。

**💡 创新点**

创新点在于通过平衡离线性（staleness）与样本多样性，给出理论最优缓冲区大小与重放比例，并证明在推理成本占主导时采用回放可获得最优计算效率。

**🔧 技术方法**

使用异步RL训练框架、GRPO/AsymRE 损失、经验回放缓冲区以及正样本偏置采样等技术。

**📊 数据集**

在 OpenR1-Math-220k 和 MATH 数据集上进行实验。

**📈 对比分析**

与传统的 generate‑then‑discard 无缓冲策略对比，缓冲方案在保持相同准确度的前提下可节省约 40% 计算，并在某些配置下甚至提升最大准确度与 pass@k；通过 Pareto 前沿展示最佳配置。

**⚠️ 局限性**

局限在于仅验证中小规模模型，未在极大模型上验证；缓冲策略仍相对简单，对离线性噪声与样本相关性的处理仍有改进空间。

---

## 123. AudioGuard: Toward Comprehensive Audio Safety Protection Across Diverse Threat Models

**arXiv ID:** 2604.08867 | [PDF](https://arxiv.org/pdf/2604.08867v1)

**作者:** Mintong Kang `[一作]` (University of Illinois Urbana-Champaign), Bo Li `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向音频输入和输出的安全基准（包含多语言、多说话人、非语音事件和声纹组合）并提出了统一的门控系统 SoundGuard+ContentGuard，可实现对音频风险的检测与评估。

**💡 创新点**

创新点：① 通过大规模红队生成音频样本并结合政策驱动的风险分类，首次形成跨语言、跨说话人、声纹与内容组合的完整音频安全风险树；② 将音频原生特征检测（SoundGuard）与语义安全评估（ContentGuard）分离并通过可解释规则集成，既提升准确率又保持低延迟；③ 对多种威胁模型（输入审核、输出审核、语音克隆、交互式语音代理）提供统一评估框架。

**🔧 技术方法**

技术：大规模红队数据生成、基于政策的风险分层分类、ECAPA‑TDNN+MLP的音频多标签分类、Whisper ASR + Gemma‑3 文本分类、规则式融合与阈值调优、跨语言文本噪声增强、端到端延迟评估。

**📊 数据集**

数据集：自研红队生成的 10k+ 语音样本，覆盖 17 种语言、50+ 说话人（儿童、名人、普通人），包含非语音有害事件、声纹+内容组合；以及外部基准（Nemotron‑Content‑Safety‑Audio、Jailbreak‑AudioBench、Omni‑SafetyBench、AdvWave）进行对比。

**📈 对比分析**

比较方法：在自研基准和四个外部基准上与多种音频 LLM（Gemma‑3、Qwen3‑Omni、Audio Flamingo、Gemini‑3、GPT‑Audio）作为单模型评判进行对比；性能：平均准确率从 0.740/0.672 提升至 0.871，延迟从 2–3 s 降至 1.42 s；在语音+内容组合、非语音事件等细分场景中表现尤为突出。

**⚠️ 局限性**

局限：① 对极端组合（如高风险声纹+极端内容）的检测仍有一定误判；② 依赖离线训练，模型更新周期可能较长；③ 隐私/公平性问题（声纹识别可能泄露身份信息）；④ 公开的风险分类与红队技术可能被滥用。

---

## 124. ViSAGE @ NTIRE 2026 Challenge on Video Saliency Prediction

**arXiv ID:** 2604.08613 | [PDF](https://arxiv.org/pdf/2604.08613v1)

**作者:** Kun Wang `[一作]` (Shandong University), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ViSAGE多专家框架，使用InternVideo2骨干提取多尺度时空特征，分别通过时序调制+空间先验和多尺度融合+深度辅助监督两种解码器生成显著性图，最终融合得到视频显著性预测；

**💡 创新点**

结合不同诱导偏置的专家解码器，通过时序调制与空间先验、深度辅助监督的多尺度融合，并利用LoRA细化预训练骨干，显著提升显著性预测的泛化与鲁棒性；

**🔧 技术方法**

采用InternVideo2预训练骨干、LoRA适配、FiLM调制、Temporal Gate、深度辅助监督、Logit/Saliency级融合以及联合KL、CC、SIM、BCE损失等技术；

**📊 数据集**

在NTIRE 2026 Video Saliency Prediction Challenge 数据集（公开/私有测试集）上进行训练和评估，并在DIEM数据集进行零样本泛化验证；

**📈 对比分析**

与挑战提交的多种方法（如ACLNet、TASED-Net、STAViS、ViNet、TSFP-Net、CASP-Net、DiffSal等）对比，ViSAGE在私有测试集上在CC和SIM上排名第一，并在DIEM零样本下获得所有指标的最高分，展示了强大的跨域性能；

**⚠️ 局限性**

仅采用两种专家解码器，可能在极端场景下难以同时优化所有指标；融合策略采用简单平均或Logit平均，缺乏自适应加权；模型对极端遮挡或快速运动的处理仍有提升空间。

---

## 125. TeamLLM: Exploring the Capabilities of LLMs for Multimodal Group Interaction Prediction

**arXiv ID:** 2604.08771 | [PDF](https://arxiv.org/pdf/2604.08771v1)

**作者:** Diana Romero `[一作]` (University of California, Irvine), Salma Elmalaki `[通讯]` (University of California, Irvine)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究通过将多模态混合现实（MR）环境下的传感器数据编码成层级自然语言上下文，评估大型语言模型（LLM）在团队行为预测中的能力，并与传统统计模型和LSTM进行对比。

**💡 创新点**

创新点在于：①首次系统性比较零样本、少样本和微调三种LLM适配策略在团队层面多模态预测上的表现；②提出将个体特征、群体结构和时间动态三层信息统一用自然语言描述的“层级上下文编码”方法；③揭示文本LLM对语义驱动行为（如对话）预测的优势与对空间/视觉信息的局限性。

**🔧 技术方法**

使用技术包括：Gemma‑2B LLM、LoRA 微调、自然语言上下文序列化、结构化社交图（sociogram）生成与评估、以及对比的LSTM与统计基线。

**📊 数据集**

数据集为64名参与者、16组（每组4人）在MR图像排序任务中收集的约25小时多模态传感器数据，涵盖视线、音频、位置和任务状态。

**📈 对比分析**

对比方法包括持久化、时间平滑、随机抽样基线和Bidirectional LSTM。单步预测中，微调后的LLM在对话Sociogram相似度上达96%（相较于LSTM的29%提升3.2×），少样本预测相似度58%，零样本仅10%。在自回归模拟模式下，LLM性能降幅高达83–99%，远低于随机基线。

**⚠️ 局限性**

限制主要有：①文本LLM无法捕捉需要空间/视觉推理的共享注意力行为；②自回归模式易因上下文错误导致灾难性误差；③对少样本例子选择敏感性低，难以进一步提升；④整体模型对上下文高度依赖，导致鲁棒性不足。

---

## 126. Tracing the Chain: Deep Learning for Stepping-Stone Intrusion Detection

**arXiv ID:** 2604.08800 | [PDF](https://arxiv.org/pdf/2604.08800v1)

**作者:** Nate Mathews `[一作]` (Rochester Institute of Technology), Matthew Wright `[通讯]` (Rochester Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了使用深度学习进行步进石（stepping‑stone）入侵检测，提出了新的流量关联模型ESPRESSO，并构建了多协议合成数据集来训练和评估该模型。

**💡 创新点**

创新点包括：①使用Transformer‑based特征提取网络结合时间对齐的多通道间隔特征；②在线triplet矿工和全序列上下文实现高效的流量嵌入；③通过时间对齐和特征去相关的辅助损失改进模型鲁棒性；④设计链长预测方法提供额外的威胁评估。

**🔧 技术方法**

技术主要包括Transformer与CvT结构、时间对齐间隔特征、在线triplet metric learning、软硬负样本挖掘、MLP决策头、链长回归。

**📊 数据集**

使用自研的五协议合成数据集（SSH、SOCAT、ICMP、DNS、混合协议），每个协议均生成10k+步进石链，模拟真实捕获分布。

**📈 对比分析**

与最先进的DeepCoFFEA基线进行对比；在host‑mode和network‑mode下，ESPRESSO在所有协议上都实现了0.99+的TPR并保持FPR≤10⁻³；在混合协议下TPR也超过0.94。基线在混合协议下仅达0.56，差距显著。

**⚠️ 局限性**

主要限制包括：①仅使用合成数据，真实网络背景流量未包含；②对DNS等周期性隧道协议的鲁棒性不足，易被时延扰动击败；③模型对极低FPR下的误报率估计可能受合成负样本影响；④模型推理时序和计算量较大，部署成本待评估。

---

## 127. MeshOn: Intersection-Free Mesh-to-Mesh Composition

**arXiv ID:** 2604.08799 | [PDF](https://arxiv.org/pdf/2604.08799v1)

**作者:** Hyunwoo Kim `[一作]` (Columbia University), Rana Hanocka `[通讯]` (University of Chicago)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种多步骤优化框架，用于将配饰网格无交叉、语义一致地贴合到基体网格上。

**💡 创新点**

创新点在于将VLM初始化、基于轨迹的交叉消除、IPC障碍损失与扩散引导的弹性变形相结合，形成完整的交叉无关、语义对齐的网格合成流程。

**🔧 技术方法**

所用技术包括Vision‑to‑Language模型、GPU加速的包围体层次结构、ICP/IPC损失、Score Distillation Sampling（扩散引导）以及NeoHookean弹性能量。

**📊 数据集**

实验使用公开的3D模型库（如ShapeNet或自制样例）及无专门数据集的手工选定配饰与基体组合。

**📈 对比分析**

与ICP、Deep Closest Point、Instant3dit等基线比较，本文方法在CLIP、CLIP‑IQA、VQA指标上更优，并且实现了完全无交叉的合成。

**⚠️ 局限性**

局限性包括较高的计算时间、对选定目标区域敏感，以及主要适用于近刚性场景，对高度弹性或布料等材料的适配有限。

---

## 128. Deep Learning-Based Tracking and Lineage Reconstruction of Ligament Breakup

**arXiv ID:** 2604.08711 | [PDF](https://arxiv.org/pdf/2604.08711v1)

**作者:** Vrushank Ahire `[一作]` (Indian Institute of Technology Ropar), Lipika Kabiraj `[通讯]` (Indian Institute of Technology Ropar)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个两阶段深度学习框架，首先使用 Faster R‑CNN 对高速阴影图像中的液体丝和液滴进行检测与分类；随后通过 Transformer‑augmented MLP 对相邻帧中的对象对进行三分类（保持、分裂、无关），从而自动重建分裂树并提取分裂统计。

**💡 创新点**

创新点包括：① 形态保持的合成数据生成策略，既能扩充数据又不产生非物理形态；② 通过对每一对象对的几何特征进行学习，实现一对多分裂事件的显式建模；③ Transformer‑MLP 结构显著提升了分裂事件的识别精度，并实现了 100% 的分裂召回；④ 整体框架能够一次性完成检测、关联与后继树重建，提升了喷雾分析的自动化与可解释性。

**🔧 技术方法**

主要技术：Faster R‑CNN（ResNet‑50+FPN）用于检测；Transformer‑augmented MLP 进行对象对关系分类；物理启发的特征（质心距离、归一化距离、IoU、面积比、类型一致性）；形态保持的合成数据生成；交叉验证、Bayesian 超参搜索与梯度裁剪等训练细节。

**📊 数据集**

数据集：0.10 wt% Carbopol 泡沫液在 Weber ≈ 400 条件下的高帧速阴影图像，共 287 张帧；从中抽取 37 张未见过的图像做最终测试，34 张相邻帧对做关系标注；合成图像从原始标注中提取对象后随机叠加或保持位置生成，用于增强训练。

**📈 对比分析**

方法对比：相较于 Mask R‑CNN、YOLOv8、传统 1:1 追踪等已有方法，框架在检测上达到了 F1 ≈ 0.872，关系分类上得到 86.1% 准确率、93.2% 精度、88.7% F1，且分裂事件召回率为 1.00。实验表明，适度合成数据（≈1–2 倍原始）能显著提升召回率，过量合成则会降低定位精度。与单帧或 1:1 追踪方法相比，显著提升了碎裂统计的完整性与物理合理性。

**⚠️ 局限性**

局限性：① 对初始标注（检测与关系）仍依赖人工；② 由于样本极度不平衡，模型在极少见的分裂事件外可能产生误判；③ 仅考虑两帧间关系，无法捕捉更长时间的演化；④ 仅在 0.10 wt% Carbopol 和特定相机条件下验证，泛化到其他流体、角度或光照下的性能尚未评估；⑤ 在极高密度喷雾区，物体重叠仍可能导致检测或关系误判。

---

## 129. Unified Multimodal Uncertain Inference

**arXiv ID:** 2604.08701 | [PDF](https://arxiv.org/pdf/2604.08701v1)

**作者:** Dengjia Zhang `[一作]` (Johns Hopkins University), Reno Kriz `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出统一多模态不确定推理任务，并构建人类标注的音频、视频与音视频连续概率评估集

**💡 创新点**

将自洽教师校准与潜在分布置信息建模相结合，形成 Calibrated Latent Uncertainty Estimation 方法，并通过模态特定批次训练实现跨模态概率校准

**🔧 技术方法**

自洽教师校准、潜在分布置信息模型、模态特定批次训练、基于 Qwen2.5-Omni-3B 的预训练与微调

**📊 数据集**

自标注的音频、视频、音视频对的评估集（10个主题、4名标注者）以及公开的文本和音频基准数据集

**📈 对比分析**

与音频、文本、视觉-文本以及全模态基线（如 Audio Flamingo3、Qwen2-Audio、Qwen3、Qwen2.5-VL 等）进行比较，3B 模型在所有模态的 MSE/准确率上均优于 32B 规模基线

**⚠️ 局限性**

评估集规模有限，仅 10 个主题；训练标签主要由教师生成，缺乏大规模人工连续概率标注，可能限制模型最终校准精度

---

## 130. Lessons Without Borders? Evaluating Cultural Alignment of LLMs Using Multilingual Story Moral Generation

**arXiv ID:** 2604.08797 | [PDF](https://arxiv.org/pdf/2604.08797v1)

**作者:** Sophie Wu `[一作]` (McGill University), Andrew Piper `[通讯]` (McGill University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出多语言故事道德生成的评估任务，并构建了14种语言、14个故事的人工标注与LLM生成的对照数据；

**💡 创新点**

创新点在于将跨文化故事道德解释作为可度量的开放式任务，结合语义相似度、人类偏好和价值分类三维评估框架；

**🔧 技术方法**

使用的技术包括多语言嵌入模型（LaBSE、MiniLM）、大型语言模型（GPT‑4o、Gemini 2.5）以及基于Schwartz价值框架的自动标签；

**📊 数据集**

所用数据集为14语言–文化对的故事摘要及对应的人工写作与LLM生成的道德句子，共计588条人类注释和多模型输出；

**📈 对比分析**

评估方法为语义相似度对比、人类偏好调查和价值分布分析；结果显示GPT‑4o和Gemini在语义相似度上接近人类平均水平，但在跨语言差异性和价值多样性上显著低于人类；

**⚠️ 局限性**

主要限制包括样本量小、翻译过程可能掩盖文化差异、评价指标受限于嵌入与偏好测量、以及对完整文本而非摘要的适用性不足。

---

## 131. Dynamic sparsity in tree-structured feed-forward layers at scale

**arXiv ID:** 2604.08565 | [PDF](https://arxiv.org/pdf/2604.08565v1)

**作者:** Reza Sedghi `[一作]` (Bielefeld University), David Kappel `[通讯]` (Bielefeld University)

**通讯引用:** 972 | [OpenAlex ID](https://openalex.org/A5052732444)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并验证在大型Transformer中用树结构的Fast Feedforward (FFF) 层替代密集MLP，实现高稀疏度下的性能接近或优于基线。

**💡 创新点**

引入无路由网络的硬二进制树路由，自动产生结构稀疏并出现自剪枝(auto-pruning)机制，且不需额外平衡损失。

**🔧 技术方法**

使用树结构的硬路由 + GELU，深度可调树，结合 GPT/OPT/DETR 等模型训练，评估零/少样本下的问答/推理任务。

**📊 数据集**

使用大规模文本数据（如 26B/300B tokens，SlimPajama、WikiText、OpenWebText 等）进行语言建模，下游任务使用 HellaSwag、Obqa、WinoGrande、ARC-e/c、boolq、piqa、MMLU、DROP 等数据集。

**📈 对比分析**

与相同参数、训练预算的密集FF Baseline 以及稀疏匹配的 MoE 进行对比；FFF 在 125M-1.3B 规模下 perplexity 与 Baseline 仅差 <10%，零/少样本任务几乎持平甚至优于 Baseline；在 1.3B 上 depth 6 的 FFF 以 8.7× 速度提升，稀疏度 94%。

**⚠️ 局限性**

受限于现有硬件/软件对动态树稀疏的支持，理论速度提升未完全实现；深度树会引入顺序执行开销，且在极深时自动剪枝可能导致路径不平衡影响性能。

---

## 132. Temperature-Dependent Performance of Prompting Strategies in Extended Reasoning Large Language Models

**arXiv ID:** 2604.08563 | [PDF](https://arxiv.org/pdf/2604.08563v1)

**作者:** Mousa Salah `[一作]` (Gujarat Technological University), Amgad Muneer `[通讯]` (University of Texas MD Anderson Cancer Center)

**通讯引用:** 3383 | [OpenAlex ID](https://openalex.org/A5038354395)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了在具备扩展推理能力的LLM（Grok‑4.1）中，采样温度与提示策略（零-shot与Chain‑of‑Thought）的交互对数学推理性能的影响。

**💡 创新点**

揭示温度与提示策略相互依赖的规律：零-shot在中等温度（0.4–0.7）下表现最佳、CoT在温度极端（0.0或1.0）下表现最好，并发现“推理放大效应”，即高温度下推理能力的提升从6倍升至14倍。

**🔧 技术方法**

使用Grok‑4.1的可控扩展推理API进行完整因子实验（16配置×39题），并结合自动解析与人工校验的评估管线。

**📊 数据集**

AMO‑Bench（39道可自动评分的IMO级数学题）。

**📈 对比分析**

通过因子实验比较零-shot/CoT与开启/关闭推理在四个温度下的准确率；结果显示零-shot+T=0.4/0.7最高达59%准确率，推理倍率最高达14.3×，整体比传统T=0做法提升约5个百分点。

**⚠️ 局限性**

仅评估单一模型且仅在数学域；温度取值离散；缺乏对内部推理链机制的解释；未验证跨模型或跨任务的普适性。

---

## 133. A Semi-Automated Framework for 3D Reconstruction of Medieval Manuscript Miniatures

**arXiv ID:** 2604.08610 | [PDF](https://arxiv.org/pdf/2604.08610v1)

**作者:** Riccardo Pallotto `[一作]` (University of Macerata), Tiberio Uricchio `[通讯]` (University of Pisa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并验证了一套半自动化流程，将中世纪手稿插图转换为可用于 WebXR、AR 与触感三维打印的三维数字模型。

**💡 创新点**

创新点在于：①对七种单图像 3D 重建模型进行系统量化评估，首次引入渲染与体量指标；②确定 Hi3DGen 作为最适合作为专家修整起点；③设计包含 SAM 分割、Hi3DGen 网格生成、ZBrush 专家优化与 AI 辅助贴图的完整可复现工作流。

**🔧 技术方法**

核心技术包括：SAM（ViT-H）图像分割、Hi3DGen 归一化桥接的 3D 生成、ZBrush 人工精修、Real‑ESRGAN 超分、Substance Painter 纹理化，以及 GLB/OBJ/FBX 导出实现跨平台可视化。

**📊 数据集**

使用 69 个手稿人物图像（38 张来自 Monteprandone 典藏，31 张来自 Vatican Library 的 Decretum Gratiani），无三维真实标注，利用原始 2D 图像作为参考。

**📈 对比分析**

通过 Silhouette IoU、LPIPS、CLIP Score、Depth Range Ratio 与 watertight % 等渲染及体量指标，并在 2AFC 人类评估中检验，Hi3DGen 在语义保真和视觉一致性上表现最佳，其次是 TripoSR；Wonder3D 虽深度高但几何缺陷明显。

**⚠️ 局限性**

局限性包括：①缺乏 3D 真实标注导致评价仅为相对指标；②高质量网格仍需耗时手工精修；③现有模型主要训练于自然图像，对极具风格化的手稿表现有限；④算力需求高（Hi3DGen 需 12 GB VRAM）。

---

## 134. StructRL: Recovering Dynamic Programming Structure from Learning Dynamics in Distributional Reinforcement Learning

**arXiv ID:** 2604.08620 | [PDF](https://arxiv.org/pdf/2604.08620v1)

**作者:** Ivo Nowak `[一作]` `[通讯]` (HAW-Hamburg), Ivo Nowak (HAW-Hamburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在分布式强化学习中分析回报分布随时间的变化，提出时序学习指标 t*，并利用其引导采样与探索，构建 StructRL 框架。

**💡 创新点**

创新点在于从学习动态中自动恢复类似动态规划的传播结构，并将该结构用于结构化采样与探索，从而提升数据利用率。

**🔧 技术方法**

使用分布式 RL、返回方差、时序学习指标、结构化采样和探索策略。

**📊 数据集**

在确定性网格最短路径（gridworld）环境上进行实验。

**📈 对比分析**

与均匀采样和基于方差的采样对比，t* 采样在更新聚焦和信息传播方面表现更佳，尽管未给出完整的数值指标。

**⚠️ 局限性**

局限包括仅在离散确定性环境验证，早期使用结构信号可能产生偏差，缺乏理论证明以及对随机或连续控制问题的推广。

---

## 135. Uncertainty Estimation for the Open-Set Text Classification systems

**arXiv ID:** 2604.08560 | [PDF](https://arxiv.org/pdf/2604.08560v1)

**作者:** Leonid Erlygin `[一作]` (Skolkovo Institute of Science and Technology), Alexey Zaytsev `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5011905327)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将 Holistic Uncertainty Estimation (HolUE) 框架迁移到开放集文本分类，通过结合文本变异性和样本质量来进行不确定性估计。

**💡 创新点**

创新点在于首次在 NLP 开放集场景下将基于 vMF 分布的概率嵌入与贝叶斯图像不确定性方法相结合，联合考虑图库结构和嵌入方差，实现对三类错误的检测。

**🔧 技术方法**

采用 BERT+MLP 生成概率嵌入，vMF 似然、KL 散度以及 MLP 组合的 HolUE 计算；并使用 SCF、AccScr、GalUE 作为基线。

**📊 数据集**

实验使用 PAN‑20‑AV 作者身份鉴定、CLINC150 目标识别、Yahoo Answers、AGNews、DBPedia 主题分类等多种数据集。

**📈 对比分析**

与基线相比，HolUE 在 PRR 指标上提升 40‑365%（如 Yahoo 365%、DBPedia 240% 等），显著提升误判过滤效果。

**⚠️ 局限性**

局限在于对动态图库需要重新校准，且对极少样本类别或生成式模型的推断尚未验证。

---

## 136. Fast Model-guided Instance-wise Adaptation Framework for Real-world Pansharpening with Fidelity Constraints

**arXiv ID:** 2604.08903 | [PDF](https://arxiv.org/pdf/2604.08903v1)

**作者:** Zhiqi Yang `[一作]` (University of Electronic Science and Technology of China), Gemine Vivone `[通讯]` (National Research Council, Institute of Methodologies for Environmental Analysis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于预训练模型的实例适配框架 FMG-Pan，实现对单幅低分辨率多光谱与高分辨率全景图像的快速高质量融合；

**💡 创新点**

创新点在于将预训练模型的伪参考作为软标签，用光谱保真与物理保真两种损失引导轻量化自适应网络，使零样本训练在保持高质量的同时实现秒级推理；

**🔧 技术方法**

采用预训练深度网络作为指导、自监督自适应训练、光谱与物理保真损失、深度可分离卷积轻量化网络、MTF匹配滤波等技术；

**📊 数据集**

使用WorldView‑3与WorldView‑2的全分辨率遥感图像作为实验数据集；

**📈 对比分析**

与传统模型、监督式深度模型以及其他零样本方法对比，FMG‑Pan 在 WV3 和 WV2 上均取得最高 HQNR 分数（≈0.964/0.947），并在 2–10 秒内完成一幅图像的训练‑推理，速度远快于现有零样本方法；

**⚠️ 局限性**

局限性包括对预训练模型的依赖、物理保真系数估计需额外计算、以及跨传感器的极端情况仍可能出现性能下降。

---

## 137. R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII

**arXiv ID:** 2604.08810 | [PDF](https://arxiv.org/pdf/2604.08810v1)

**作者:** Zewei Zhou `[一作]` (Nanjing University of Science and Technology), Daying Sun `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 R2G（RTL-to-GDSII）多视角电路图基准套件，标准化五种阶段感知视图，提供统一的 DEF‑to‑图转换、加载器、数据拆分与可复现基线。

**💡 创新点**

创新点在于：①实现信息对等（information parity）——五种视图编码相同属性，只改变特征附着位置；②将视图选择与模型架构解耦，揭示视图对性能的主导影响；③系统评估了 decoder‑head 深度，发现 3–4 层可将模型提升至 R²>0.99。

**🔧 技术方法**

使用经典 GNN 结构（GINE、GAT、ResGatedGCN）进行实验，并结合 OpenROAD 端到端物理设计流水线提取特征与标签；同时分析消息传递层深度与 decoder‑head 结构对结果的影响。

**📊 数据集**

数据集为 30 个开源 IP 核（处理器、DSP、加密、通信/控制、视频/音频等），节点/边规模最高可达 10⁶，全部通过 DEF 文件从 RTL 生成，涵盖布局、路由、时序等多阶段信息。

**📈 对比分析**

通过在同一数据集上对不同视图与模型进行交叉实验，发现同一 GNN 在不同视图下 R² 差异可超过 0.3；node‑centric 视图（b、c）在摆放与路由任务上表现最稳健；decoder‑head 深度 3–4 层显著提升精度，最终在摆放任务上实现 R²≈0.99，路由任务接近同等精度。

**⚠️ 局限性**

局限性包括：仅覆盖晚期物理设计阶段；特征来源局限于 DEF，缺乏时序、IR‑drop、功耗等细粒度信息；技术节点单一（主要为 180nm/90nm 等）；评估仅针对经典 GNN，未涉及图 Transformer、MoE 等新型架构。

---

## 138. FluidFlow: a flow-matching generative model for fluid dynamics surrogates on unstructured meshes

**arXiv ID:** 2604.08586 | [PDF](https://arxiv.org/pdf/2604.08586v1)

**作者:** David Ramos `[一作]` (Universidad Politécnica de Madrid), Gonzalo Rubio `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 866 | [OpenAlex ID](https://openalex.org/A5026793328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种名为FluidFlow的基于条件流匹配的生成式 surrogate 模型，可直接处理结构化与非结构化网格的 CFD 数据，生成受物理条件约束的流场预测。

**💡 创新点**

创新点在于将流匹配技术迁移到流体动力学 surrogate，构建无网格预处理、可在原始离散化上训练的模型，并结合 U‑Net 与 Diffusion Transformer 的双架构。

**🔧 技术方法**

采用条件流匹配、分类器无监督引导、U‑Net、Diffusion Transformer 与线性注意力等技术实现模型。

**📊 数据集**

使用 RAE2822 空气动力学边界压力系数数据（1000 条件）和 ONERA CRM WBPN 3D 飞机全表面压力/摩擦系数数据（468 条 CFD 计算，260k 网格点）。

**📈 对比分析**

与传统 MLP 及之前的 CNN/GCNN/MLP 基线相比，FluidFlow 在压力系数预测上 MSE 降低约 10‑2 倍，R² 从 0.997 提升到 0.999；在 3D 飞机任务中，加权 R² 从 0.956 提升至 0.965，且在多维输出上保持高精度。

**⚠️ 局限性**

局限包括对大规模网格的线性注意力近似可能导致精度下降、对时间动态流动尚未验证、缺乏多 GPU 并行加速、需要进一步研究补丁大小与注意力机制的权衡以及对不同几何形状的泛化能力。

---

## 139. AlphaLab: Autonomous Multi-Agent Research Across Optimization Domains with Frontier LLMs

**arXiv ID:** 2604.08590 | [PDF](https://arxiv.org/pdf/2604.08590v1)

**作者:** Brendan R. Hogan `[一作]` (Morgan Stanley), Yuriy Nevmyvaka `[通讯]` (Morgan Stanley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个自动化的多代理研究系统，利用前沿LLM完成从数据探索、评估框架构建到大规模GPU实验的完整科研周期。

**💡 创新点**

系统自生成域适配器、使用对抗Builder/Critic/Tester循环构建评估框架、持续的Playbook记忆以及全流程无人工干预的GPU调度策略。

**🔧 技术方法**

前沿LLM（GPT-5.2、Claude Opus 4.6）结合shell、web搜索、子代理工具，四阶段流水线与Strategy/Worker/Dispatcher循环，以及Supervisor监控与Playbook知识累积。

**📊 数据集**

CUDA KernelBench+Sakana AI的GPU内核基准、PleIAs SYNTH文本数据用于LLM预训练、San Francisco Bay Area高速公路传感器的交通占用率数据。

**📈 对比分析**

与单次LLM调用、贪婪循环基线以及公开论文实现对比，CUDA内核平均加速4.4×（最高91×）、LLM预训练验证损失比单一基线低22%、交通预测RMSE比季节性基线低23–25%。

**⚠️ 局限性**

对API变更敏感导致高失败率、Playbook过早收敛缺乏多样性保障、单次实验结果限制统计可靠性、缺少代码沙箱和安全机制。

---

## 140. Optimal Single-Pass Streaming Lower Bounds for Approximating CSPs

**arXiv ID:** 2604.08731 | [PDF](https://arxiv.org/pdf/2604.08731v1)

**作者:** Noah G. Singer `[一作]` (Carnegie Mellon University), Santhoshini Velusamy `[通讯]` (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了单次通过（single‑pass）流算法在近似求解约束满足问题（CSP）时所需的空间下界，证明只要存在基本线性规划（basic LP）的（γ,β）集成性缺口实例，则区分至少满足 γ−ε 与至多满足 β+ε 的 Max‑CSP 需要 Ω(n) 的线性空间。

**💡 创新点**

创新点在于：①将之前仅适用于具备线性代数结构的特定 CSP 子类（如 Max‑Cut、Max‑DiCut）的方法推广到任意谓词族；②引入“分析”型的线性条件，取代先前的线性代数条件，使证明更简洁并覆盖更广泛的问题；③得到的下界是单次通过模型的近似极限。

**🔧 技术方法**

主要技术包括：通信复杂度的隐式隐藏分区（implicit hidden partition）问题的分布式化简；将该通信问题归约到 CSP 流模型；利用 Fourier 分析、随机噪声（hypercontractivity）以及“单例自由”（singleton‑free）频率集合的级别不等式；以及通过迭代与概率界（boundedness → uniformity）证明后验分布的均匀性。

**📊 数据集**

该工作完全基于理论分析，不使用任何实际数据集；所有结果均为组合与信息理论证明。

**📈 对比分析**

与先前工作（如 Chou 等 2022、Fei‑Minzer‑Wang 2026 等）的比较表明，本论文在所有 CSP（包括 Max‑k‑XOR、Max‑LTF、Max‑Exactly‑ℓ‑Of‑k、Max‑DiCut、Max‑2SAT 等）上实现了线性空间下界，甚至在单次通过模型上达到最优；相比之下，早期的下界多限于 O(√n) 或仅适用于特殊结构。

**⚠️ 局限性**

限制：该下界假设输入实例具有有限度（bounded‑degree）；对无界度数的实例尚未覆盖；此外，本文仅讨论单次通过流模型，多次通过或随机化模型的进一步改进仍是开放问题。

---

## 141. From Business Events to Auditable Decisions: Ontology-Governed Graph Simulation for Enterprise AI

**arXiv ID:** 2604.08603 | [PDF](https://arxiv.org/pdf/2604.08603v1)

**作者:** Hongyin Zhu `[一作]` (Yonyou AI Lab), Feng Wu `[通讯]` (Yonyou Network Technology Co., Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于企业本体的事件驱动图模拟框架 LOM-action，实现从业务事件到可审计决策的完整流程。

**💡 创新点**

创新点在于将情景模拟作为前置步骤，采用沙盒图变更、双模执行（技能+推理）以及工具链 F1 和虚假准确性指标，显著提升可审计性和可靠性。

**🔧 技术方法**

使用 Qwen3.5‑27B 进行监督微调，结合 19 级函数调用接口、Neo4j 沙盒模拟以及自定义的图操作 API。

**📊 数据集**

使用人工构造的 20–30 节点 Neo4j 子图，共 2,200 训练样本，覆盖 11 项图操作任务。

**📈 对比分析**

与 Doubao‑1.8、DeepSeek‑V3.2 两个零射前沿模型对比，LOM‑action 在准确率 93.82% 与工具链 F1 98.74% 上分别优于 80% 与 24–36% 的基线，四倍提升。

**⚠️ 局限性**

局限在于仅在小规模合成图上验证，未做跨域/跨规模泛化实验，缺乏基线去训练层面的消融与实时吞吐/延迟评估。

---

## 142. Realisation-Level Privacy Filtering

**arXiv ID:** 2604.08630 | [PDF](https://arxiv.org/pdf/2604.08630v1)

**作者:** Sophie Taylor `[一作]` (University of Oxford), Justin Coon `[通讯]` (University of Oxford)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并证明了一种基于实际泄露（realisation-level）的差分隐私过滤器，用于在适应性查询序列中动态决定何时停止数据发布。

**💡 创新点**

创新点在于：①不再依赖机制的最坏情况隐私参数，而是跟踪每一步的真实泄露；②采用前瞻性阈值设计，使停止事件不依赖已发布的输出，从而保证（ε,δ）-DP；③提供了理论证明和可应用于任意机制的通用框架。

**🔧 技术方法**

使用了差分隐私的基本定义、隐私过滤器理论、Rényi相对熵的组合性质、对随机变量的期望与概率界定、以及Gaussian机制的解析计算。

**📊 数据集**

实验使用的主要数据集为合成的二值计数查询（即两个相邻数据集的二进制特征向量），并通过 i.i.d. 高斯机制（σ=2）模拟查询结果。

**📈 对比分析**

与传统的经典组合、加法过滤器、先进组合和RDP过滤器进行对比。实验显示：在相同隐私预算下，realisation-level过滤器在大多数时间点的存活概率（可发布次数）高于其它方法，尤其在长时间序列中显著优于RDP过滤器；但在极早期的几个步骤可能略逊。

**⚠️ 局限性**

局限性包括：①需要预先设定参数（δ̃,θ,N），其优化问题仍未完全解决；②实现时需评估复杂机制下的 δ̂ 计算，可能需要蒙特卡洛模拟，计算开销较大；③对强依赖或非高斯机制的适用性和性能尚未系统评估。

---

## 143. Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning

**arXiv ID:** 2604.08780 | [PDF](https://arxiv.org/pdf/2604.08780v1)

**作者:** Mohamad H. Danesh `[一作]` (McGill University), Hsiu-Chin Lin `[通讯]` (McGill University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出基于形态编码的四足世界模型（QWM），能够在单一模型中学习多种四足机器人的运动动力学，并实现零步无模型适配。

**💡 创新点**

① 将机器人结构信息显式编码为可缩放的特征向量并作为条件输入，避免隐式系统辨识导致的适配延迟；② 引入自适应奖励归一化（ARN），统一不同硬件奖励尺度；③ 采用双塔编码器分离静态与动态信息，增强对多形态的泛化。

**🔧 技术方法**

在 DreamerV3 的 RSSM 框架上改造；使用 Physical Morphology Encoder (PME) 处理 USD 资产；自适应奖励归一化（ARN）；双塔 encoder 结构；imagination‑based policy 学习；在 NVIDIA Isaac Lab 构建异形态并行环境。

**📊 数据集**

训练并评估七款四足机器人：ANYmal B/C/D、Unitree A1/Go1/Go2/B2、Boston Dynamics Spot；收集物理属性（尺寸、质量、关节结构）和对应奖励定义；对未见机器人进行零步测试。

**📈 对比分析**

与 PPO、PME‑PPO、Body Transformer、DreamerV3、PWM、TWISTER 等基线在同一多形态训练集上对比；QWM 在奖励、回合长度和预测误差（NMSE）方面明显优于基线；零步泛化中对 Unitree Go1 和 ANYmal‑D 达到与专属训练模型相当的奖励和回合长度；在真实硬件部署中直接实现稳定行走，无需微调。

**⚠️ 局限性**

泛化仅限于训练分布内插值，无法在极端外推（如 Unitree B2）表现良好；使用固定维度的形态编码，难以处理结构可变的机器人；目前仍依赖大量模拟数据，现实场景的未建模动态可能导致性能下降。

---

## 144. On the Spectral Geometry of Cross-Modal Representations: A Functional Map Diagnostic for Multimodal Alignment

**arXiv ID:** 2604.08579 | [PDF](https://arxiv.org/pdf/2604.08579v1)

**作者:** Krisanu Sarkar `[一作]` `[通讯]` (Indian Institute of Technology Bombay), Krisanu Sarkar (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了如何用功能映射（Functional Map）框架对独立预训练的视觉（DINOv2）和语言（MiniLM）编码器进行跨模态对齐，并通过谱诊断指标评估其几何兼容性。

**💡 创新点**

首次将功能映射应用于神经网络表示，发现“谱复杂度–取向缺口”（spectral complexity–orientation gap）并提出三种诊断量化指标（谱距离、对角主导性、正交偏差）来刻画跨模态表示的几何关系。

**🔧 技术方法**

使用图拉普拉斯谱、kNN图构造、正则化最小二乘求解功能映射、ZoomOut细化、Procrustes、相对表示、CCA等技术进行实验与对比。

**📊 数据集**

主要使用 Flickr30k 数据集（1,000 张图像，5,000 条英文描述）进行检索实验，并以 CLIP（ViT-B/32）作为联合训练的基准。

**📈 对比分析**

与 Procrustes、相对表示、CCA、CLIP 等方法在 Recall@1/5/10 上对比；功能映射在所有 anchor 预算下均落后，Recall@1 仅 2–4%（最高 4.3%），而 Procrustes 可达 55%+；诊断指标显示谱距离很小（0.043）但对角主导性低（<0.05）和正交偏差极大（70.15），揭示功能映射不适用。

**⚠️ 局限性**

局限性：仅在 1,000 样本、单一视觉模型和固定 kNN 参数下评估；功能映射对独立预训练模型效果差，缺乏对更大模型或不同预训练方式的验证；谱诊断结果受图构造方法影响，需要进一步探究。

---

## 145. Decomposing the Delta: What Do Models Actually Learn from Preference Pairs?

**arXiv ID:** 2604.08723 | [PDF](https://arxiv.org/pdf/2604.08723v1)

**作者:** Chia-Hsuan Lee `[一作]` (Capital One), William Campbell `[通讯]` (Capital One)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了偏好优化（如 DPO、KTO）中偏好对齐模型推理性能的作用，系统分离并量化了两种“delta”——生成器级别差距与样本级别差距，探讨其对推理任务的影响，并提出了通过最大化生成器级别差距和选择样本级别差距大的数据来提高推理性能的实用方案。

**💡 创新点**

创新点在于：①首次把偏好数据的“delta”细化为生成器级与样本级两层，揭示生成器能力差距对跨域推理提升的主要驱动力；②用 LLM-judge 对推理链进行多维质量评分（事实性、步骤连贯性、策略连贯性、计算精度、信噪比），并发现步骤连贯性是最具信息量的样本级别指标；③证明即使样本对都不正确，偏好优化仍能提升性能，说明推理过程质量是关键；④展示通过挑选高 delta 样本即可在 5k 对内获得与完整 16.5k 对齐的相同效果。

**🔧 技术方法**

技术方法包括 DPO（对数似然差异优化）与 KTO，LLM-as-judge（GPT‑OSS‑120b）对推理链进行多维评分，使用 OpenR1‑Math‑220k 生成推理轨迹并构建偏好对，进行离线偏好训练并评估于多域基准（MATH、GSM8K、AMc、MMLU‑Pro、TheoremQA、LiveCodeBench 等）。

**📊 数据集**

数据集：OpenR1‑Math‑220k（220k 数学题+推理轨迹）、MATH‑500、GSM8K、AMC、Minerva‑Math、OlympiadBench、AIME、MMLU‑Pro、TheoremQA、LiveCodeBench。

**📈 对比分析**

与基准模型（Nemotron‑8B）对比，使用弱生成器对（s1‑3B vs. s1‑7B）进行偏好训练后，能匹配或略优于使用强生成器（DeepSeek‑R1）的传统 outcome‑verified 训练；随着生成器能力差距增大，跨域推理性能持续提升；仅用 5k 高 delta 样本即可与完整 16.5k 对齐获得相近准确率，表明样本选择可显著提升数据效率。

**⚠️ 局限性**

局限性：①对数推理链质量评分依赖 LLM‑judge，可能引入评估偏差；②研究聚焦数学推理，跨域推理（STEM、代码）虽提升但仍受限于基准分布；③生成器级差距增大后域内性能趋于饱和，无法进一步提升；④实验以 Nemotron‑8B 为基础模型，结果可能不完全泛化至其他大模型。

---

## 146. SIC3D: Style Image Conditioned Text-to-3D Gaussian Splatting Generation

**arXiv ID:** 2604.08760 | [PDF](https://arxiv.org/pdf/2604.08760v1)

**作者:** Ming He `[一作]` (University of Sheffield), Steve Maddock `[通讯]` (University of Sheffield)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段的图像条件文本到3D高斯渲染（3D Gaussian Splatting）生成框架 SIC3D，用于高质量、可控的3D对象风格迁移。

**💡 创新点**

创新点在于：①首次将图像风格注入3DGS的后期优化；②设计了 Variational Stylized Score Distillation (VSSD) 损失，使风格与几何在多视角下保持一致；③加入尺度正则化避免高斯膨胀导致的伪影。

**🔧 技术方法**

核心技术包括：3DGS（如 GaussianDreamer、TRELLIS）作为几何基础；IP‑Adapter + LoRA 作为跨模态风格注入；VSSD 损失与尺度正则化；Stable Diffusion v1.5 作为 2D 预训练扩散模型。

**📊 数据集**

主要使用 Stable Diffusion v1.5 作为先验模型，并在实验中采样多种风格图像与文本提示组合（未公开具体数据集，实验基于公开的风格图像与文本提示）。

**📈 对比分析**

与风格化提示基线以及三种现有 3DGS 风格迁移方法（StyleGaussian、G-Style、SGSST）对比；在 NNFM、RMSE、LPIPS 指标上均获得最低/最佳分数；GPT‑4o 评估的 Elo 分数显示在对象质量与风格一致性方面均优于对比方法。

**⚠️ 局限性**

主要限制：仍依赖单视角扩散模型，跨视角一致性受限，可能出现局部纹理伪影；对多视角一致性和更复杂场景的适用性仍需进一步研究。

---

## 147. InsEdit: Towards Instruction-based Visual Editing via Data-Efficient Video Diffusion Models Adaptation

**arXiv ID:** 2604.08646 | [PDF](https://arxiv.org/pdf/2604.08646v1)

**作者:** Zhefan Rao `[一作]` (Hong Kong University of Science and Technology), Qifeng Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 HunyuanVideo‑1.5 通过数据高效的适配路径，构建了可用约 10 万条视频编辑样本的指令式视频编辑模型 InsEdit。

**💡 创新点**

创新点在于：① 引入 Mutual Context Attention（MCA）生成能从任意帧开始的编辑对，显著提升训练样本信息量；② 在两阶段训练中保持少量生成目标，帮助保留生成先验；③ 通过高比例图像编辑数据增强空间编辑能力，实现图像编辑的无缝延伸。

**🔧 技术方法**

采用双流编辑架构（语义模块+视觉模块）、冻结 Qwen2.5‑VL、SigLIP 与 Glyph‑ByT5 编码器，MMDiT 变换器进行可训练；MCA 用于视频对齐生成；两阶段训练结合生成、VLM 重构与一致性保持目标。

**📊 数据集**

使用 HunyuanVideo‑1.5 预训练模型；视频编辑数据来源于 OpenVE、Ditto 以及 3×MCA 生成的约 300K 对；图像编辑数据约 1M 条，包含公开数据与 Qwen‑Image‑Edit 生成样本。

**📈 对比分析**

与 VACE‑14B、OmniVideo、InsViE、Lucy‑Edit、ICVE、Ditto、VINO、UniVideo 等开源基线对比，基于 OpenVE‑Bench 与自建 InsEdit‑Bench 评估。InsEdit 在所有指标（整体质量、指令遵从、时序视觉质量、未编辑区域保留）均领先，尤其在局部编辑和创意编辑方面显著超越对手，且推理速度更快。

**⚠️ 局限性**

仅限 2D 短视频编辑，难以处理多对象关系推理、精准空间控制以及长视频的时序一致性；对较长视频、复杂场景与多模态控制（如蒙版、关键点）仍表现不佳。

---

## 148. RAMP: Hybrid DRL for Online Learning of Numeric Action Models

**arXiv ID:** 2604.08685 | [PDF](https://arxiv.org/pdf/2604.08685v1)

**作者:** Yarin Benyamin `[一作]` (Ben-Gurion University of the Negev), Roni Stern `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了 RAMP，一种在线数值规划的混合策略，结合了深度强化学习（PPO+掩码）、在线数值行动模型学习（NSAM）和规划（Metric-FF），并构建了 Numeric PDDLGym 框架将 PDDL2.1 域转换为 Gym 环境。

**💡 创新点**

首创在线数值行动模型学习，并通过安全保证的 NSAM 与规划、RL 的正反馈循环提升规划效率；引入 Numeric PDDLGym 解决 PDDL 与 Gym 的接口不匹配问题。

**🔧 技术方法**

使用深度强化学习（PPO + 掩码技术）、NSAM 进行安全数值模型学习、Metric-FF 规划器、PDDL2.1、Gym 接口、RLlib 等技术。

**📊 数据集**

在 IPC 经典数值规划域（Counters、Depot、Sailing）以及基于 Polycraft 的 Pogo Stick 域上进行实验。

**📈 对比分析**

与基线 PPO（加掩码）比较，RAMP 在大多数域和难度等级上实现了更高的成功率（接近 100%）和更短的解决方案长度；在 Pogo Stick 大规模实例中因规划器 60 秒超时导致差距减小。

**⚠️ 局限性**

局限性包括：规划器超时限制导致某些实例无法规划；模型召回率不完备（尤其在 Depot）；无法处理噪声或部分可观测环境；Numeric PDDLGym 仅支持线性前置条件/效应，且不支持数值目标条件和条件效应。

---

## 149. PRAGMA: Revolut Foundation Model

**arXiv ID:** 2604.08649 | [PDF](https://arxiv.org/pdf/2604.08649v1)

**作者:** Maxim Ostroukhov `[一作]` (Revolut Research), Anton Repushko `[通讯]` (Revolut Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 PRAGMA，一种面向多源银行用户历史事件序列的基础模型，能够直接从原始事件流生成可迁移的记录级嵌入。

**💡 创新点**

创新点在于：① 针对金融记录的 key–value–time 细粒度分词方案；② 双分支结构（事件编码器 + 个人资料编码器）并通过历史编码器融合；③ 结合多粒度掩码预训练（单词、事件、键级）以学习全局与局部上下文；④ 通过 LoRA 进行高效任务微调，显著降低可学习参数量。

**🔧 技术方法**

核心技术包括 Transformer 编码器、掩码语言建模（MLM）、RoPE 位置编码、动态批量与序列打包、低秩适配（LoRA）以及可选的预训练文本编码器（如 Nemotron）。

**📊 数据集**

使用了 26M 份匿名用户记录（覆盖 111 个国家）构成的训练语料，总计 24B 事件、207B token，包含交易、应用、交易、通讯等多种事件类型以及静态个人资料字段。

**📈 对比分析**

与多任务特定模型对比，PRAGMA 在信用评分、沟通互动、欺诈检测、产品推荐、重复交易检测、生命周期价值等六类任务上均实现相对提升；最显著的提升为信用评分 PR-AUC 提升 130.2%，沟通互动 PR-AUC 提升 79.4%，产品推荐 mAP 提升 40.5%，并且 LoRA 微调往往能匹配或超过从零开始训练的完整模型。

**⚠️ 局限性**

主要局限在于无法捕获跨记录的关系特征，对如反洗钱等需要网络层面信息的高度关联任务表现差劲；此外，文本密集任务在使用预训练文本编码器时虽有提升但会显著增加训练时间。

---

## 150. SynDocDis: A Metadata-Driven Framework for Generating Synthetic Physician Discussions Using Large Language Models

**arXiv ID:** 2604.08555 | [PDF](https://arxiv.org/pdf/2604.08555v1)

**作者:** Beny Rubinstein `[一作]` (University of Aveiro), Sergio Matos `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SynDocDis框架，利用去标识化病例元数据通过大型语言模型生成医师间的合成讨论。

**💡 创新点**

创新点在于：仅用元数据即可产生高保真医师对话；采用CIDI结构化提示与情感提示提升自然度；公开完整评测体系并提供可复现的提示与数据包。

**🔧 技术方法**

使用大语言模型（如GPT‑4）、结构化提示（CIDI）、链式思考、情感提示及多维评估指标（Likert、加权Fleiss' κ）。

**📊 数据集**

数据集为医师填写的去标识化病例元数据，生成9个合成讨论，并公开提示、对话与专家评分。

**📈 对比分析**

通过5名临床医生对9个合成讨论进行5分制评估，沟通有效性平均4.4/5，医学内容质量平均4.1/5，κ=0.70，91%评为优秀/良好。

**⚠️ 局限性**

限制包括：依赖GPT‑4导致可复现性受限；约20%讨论在临床准确性上评为一般或以下；证据引用较少，响应多样性不足；仅覆盖9个场景，样本规模有限。

---

## 151. Using Synthetic Data for Machine Learning-based Childhood Vaccination Prediction in Narok, Kenya

**arXiv ID:** 2604.08902 | [PDF](https://arxiv.org/pdf/2604.08902v1)

**作者:** Jimmy Bach `[一作]` (William & Mary), Haipeng Chen `[通讯]` (William & Mary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用逻辑回归和XGBoost模型对纳罗克县马萨伊社区儿童的疫苗接种风险进行预测，并采用TabSyn扩散模型生成合成数据以保护隐私。

**💡 创新点**

首次在低资源游牧环境中应用Latent Diffusion TabSyn进行疫苗数据合成，实现高保真合成数据而不损失模型预测性能。

**🔧 技术方法**

使用的技术包括Logistic Regression、XGBoost、TabSyn（VAE+分数扩散）以及SMOTE等数据平衡方法。

**📊 数据集**

所用数据集为MOH 510纸质登记的8年儿童接种记录，共计6,913例。

**📈 对比分析**

通过比较仅使用真实数据、仅使用合成数据和混合数据的训练集，评价精确率、召回率和F1分数；对早期疫苗（如BCG）F1>90%，合成数据模型与真实数据模型性能相当。

**⚠️ 局限性**

局限性包括样本量有限、可能存在录入或分类错误、仅涵盖就诊者导致的代表性不足，以及低资源环境下可复制性和外推性的进一步验证需求。

---

## 152. A Longitudinal Study of Dependency Reclassifications in JavaScript Projects

**arXiv ID:** 2604.08747 | [PDF](https://arxiv.org/pdf/2604.08747v1)

**作者:** Yuxin Liu `[一作]` (KTH Royal Institute of Technology), Benoit Baudry `[通讯]` (Université de Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 JavaScript 生态中大量项目的依赖声明（Core/Dev/Peer）进行纵向历史分析，重建依赖角色变迁，构建重分类实践的分类体系并评估其普遍性、形式与时间特征。

**💡 创新点**

①首次开展大规模纵向研究，系统揭示依赖角色在项目生命周期内的多样化重定义；②提出基于事件序列的重分类实践分类法；③提供针对性工具/包管理器改进的实证依据。

**🔧 技术方法**

通过 GitHub commit 差异比较、结构化依赖字段（dependencies / devDependencies / peerDependencies）解析，抽取重分类事件；随后将事件序列聚合、归类形成实践范式；使用统计分析（频率、占比、时间分布）评估模式。

**📊 数据集**

≈35,821 个 Node.js / JavaScript 开源项目（GitHub Search Engine 261k → 35k 经筛选），共计 442,258 个依赖实例，覆盖 175,853 Core、264,055 Dev、2,350 Peer，历史超过 5 年。

**📈 对比分析**

对比了不同角色、不同实践（一次性删除、多步删除、角色转移、角色振荡）在项目内出现率、依赖数量、批量删除比例以及时间延迟等维度的差异；结果显示重分类普遍、批量删除常见、回退延迟长、角色振荡短；未与现有工具直接对比，但为工具改进提供基准。

**⚠️ 局限性**

①仅关注 npm/JavaScript 生态，结果可能不适用于 Maven/PyPI/Cargo 等；②只分析默认分支，忽略长周期分支或未合并的变更；③事件检测基于文件差异，无法判断代码实际使用情况；④缺乏定性讨论，难以解释为何出现某种重分类模式。

---

## 153. Omakase: proactive assistance with actionable suggestions for evolving scientific research projects

**arXiv ID:** 2604.08898 | [PDF](https://arxiv.org/pdf/2604.08898v1)

**作者:** Pao Siangliulue `[一作]` (Allen Institute for AI), Daniel S. Weld `[通讯]` (Allen Institute for AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款主动式研究助手，监测研究者的项目文档，自动推理项目状态并向深度研究系统发起查询，随后将长篇研究报告提炼为可操作、上下文化的建议；

**💡 创新点**

①通过文档而非手动提问获取实时上下文；②将深度研究输出转化为可执行的建议；③支持多视图交互（建议中心、文档中心）并允许用户调节更新频率；

**🔧 技术方法**

采用大型语言模型（Opus 4.6、早期 O3）、Semantic Scholar API、深度研究系统（Open‑source LLM‑powered 文献检索），并实现文档解析、项目状态推断、问题生成、建议筛选与多级排序等管道；

**📊 数据集**

收集了研究者的真实项目文档（Google Doc/Slide/Sheet/LaTeX）共约 12 份，结合 Semantic Scholar 文献与深度研究系统检索结果；

**📈 对比分析**

通过对比实验：①让受试者评估系统生成的建议与深度研究回答的相关性、可操作性与时效性（Likert 1‑7 量表），并使用线性混合效应模型和 Wilcoxon 检验；结果显示系统建议在相关性（5.87 vs 5.22）、可操作性（6.01 vs 4.15）和时效性（5.50 vs 4.88）上均显著优于深度研究回答；在更高负荷的双段落基准下仍保持可操作性优势；同时系统产生的问题在当前文档版本上更贴合项目阶段（平均 5.28 vs 4.63，p<0.05）。

**⚠️ 局限性**

仅使用单一文档作为上下文，无法覆盖多源项目信息；对文档隐私和授权的安全性尚未彻底解决；系统对项目方向的提前推断偶尔过度自信；缺乏长周期部署评估；可能导致研究者过度依赖 AI 建议。

---

## 154. A Little Rank Goes a Long Way: Random Scaffolds with LoRA Adapters Are All You Need

**arXiv ID:** 2604.08749 | [PDF](https://arxiv.org/pdf/2604.08749v1)

**作者:** Hananel Hazan `[一作]` (Tufts University), Michael Levin `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种叫 LottaLoRA 的训练范式，将所有权重设为随机固定，仅训练低秩 LoRA 适配器，并在九个基准上验证其效果。

**💡 创新点**

证明任务相关信息可被压缩到极小低秩子空间，随机冻结骨干可作为有效的“储层”，并提出最低秩阈值估计任务内在维度。

**🔧 技术方法**

低秩 LoRA 适配、随机初始化冻结骨干（Reservoir Computing 类比）、基于种子重构、不同分布鲁棒性评估与重采样实验。

**📊 数据集**

MNIST、PhysioNet ICU mortality、CIFAR‑10、OGBG‑MolHIV、OGBN‑Arxiv、Decision Transformer、Vision Transformer（Flowers‑102）、IMDB 语义分类、WikiText‑103 等。

**📈 对比分析**

与全参数训练和传统 LoRA 在相同训练协议下比较，LottaLoRA 在 0.5–40% 参数训练下恢复 96–100% 的基准性能；在 900M Transformer 上仅损失约 0.8 nats；在其他任务上也能达到与全训练相近或略低的准确率/AUROC。

**⚠️ 局限性**

仍需多尺度、多优化器验证；随机骨干的极限与全训练不完全相同；ASIC 加速需进一步实现；最低秩与任务复杂度不完全稳健，过度参数化会影响最低秩估计。

---

## 155. From OSS to Open Source AI: an Exploratory Study of Collaborative Development Paradigm Divergence

**arXiv ID:** 2604.08888 | [PDF](https://arxiv.org/pdf/2604.08888v1)

**作者:** Hengzhi Ye `[一作]` (Peking University), Minghui Zhou `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过收集 GitHub 的 1,428,792 个 OSS 仓库和 Hugging Face Hub 的 1,440,527 个 OSM 仓库，结合统计、社交网络和 LLM 辅助内容分析，以及 10 轮专家访谈，对 OSS 与 OSM 在协作强度、协作开放性和用户创新三个维度进行量化比较，揭示了两者在协作行为与创新模式上的显著差异，并分析了技术壁垒、资源限制、平台不匹配和企业战略等社会技术因素。

**💡 创新点**

创新点在于首次系统量化 OSS 与 OSM 的协作差异，构建多维度指标体系并结合 LLM 与人类双重验证的主题分析方法，同时通过专家访谈挖掘导致差异的深层社会技术根源，为 AI 开源协作提供实证基础和改进路径。

**🔧 技术方法**

主要技术包括 REST API 与 GHArchive 采集、数据清洗与机器人过滤、Pearson 相关性与 Mann-Whitney U 检验、基于 LDA 的网络分析（Louvain 社区检测）、LLM 辅助主题建模（Chain‑of‑Thought + 人工校验），以及半结构化访谈的主题分析。

**📊 数据集**

使用的数据集为 GitHub 的完整仓库事件流（约 4.376 亿条）和 Hugging Face Hub 的模型仓库 JSON 数据（约 1.44 万家），随后按星级分层抽样得到 1,428,792 OSS 仓库与 1,440,527 OSM 仓库，并剔除机器人账户。

**📈 对比分析**

比较方法为：1) 对 commit、issue/讨论、star/like 等指标做分布与均值比较，并用 Mann‑Whitney U 检验显著性；2) 构建仓库与开发者网络，计算连通率、平均度和最大度等网络指标；3) 对 issue/讨论文本使用 LLM 辅助主题分析，得到主题比例。结果显示 OSS 在协作强度、开放性及基于 bug 的创新上明显领先，差异均显著；而 OSM 在讨论内容更侧重使用问题和性能评估，体现了适应性利用创新。

**⚠️ 局限性**

局限性包括：仅采集 GitHub 与 Hugging Face Hub 两个平台，可能遗漏其他社区特性；活动指标限定为 commit、issue/讨论、star/like，未涉及 PR、邮件等多渠道交流；访谈样本量仅 10 人，专业领域覆盖有限；LLM 辅助主题分析虽人机校验，但仍可能受模型偏差影响；以及研究时间点固定，无法反映 AI 开源生态的快速演进。

---

## 156. Precise Shield: Explaining and Aligning VLLM Safety via Neuron-Level Guidance

**arXiv ID:** 2604.08881 | [PDF](https://arxiv.org/pdf/2604.08881v1)

**作者:** Enyi Shi `[一作]` (Nanjing University of Science and Technology), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Precise Shield，一种基于神经元级安全对齐的框架，用于提升 Vision‑Language 大模型在多语言多模态攻击下的安全性。

**💡 创新点**

创新点在于：① 通过对有害与安全输入的激活差异识别出少量关键安全神经元；② 仅在该子空间内进行梯度遮蔽与 LoRA 微调，仅更新不到 0.03% 参数；③ 发现安全神经元在语言和模态间存在显著重叠，可实现零转移。

**🔧 技术方法**

技术包括神经元重要性评估（平均激活×下行权重范数）、激活对比、梯度遮蔽 LoRA、低秩参数适配、指令微调与安全判定损失。

**📊 数据集**

使用的主要数据集有 Lingua‑SafetyBench（10 语言、图像/文本占优风险）、MM‑Bench（泛化对齐）、MM‑Vet 与 MGSM（多模态与多语言通用能力评估）。

**📈 对比分析**

与 LoRA SFT、XSAFETY、ESCO、Self‑Defense 等基线对比；在 10 种语言、图像占优和文本占优攻击下，Precise Shield 在攻击成功率 (ASR) 上显著下降，且仅更新 0.03% 参数；对模型的多模态/多语言通用性能几乎不受影响。

**⚠️ 局限性**

局限性包括：对预训练模型结构依赖较大；安全神经元数量有限，跨模态转移效果不如直接训练；在极端或未覆盖的语言/模态攻击场景下仍有提升空间。

---

## 157. Risk-Aware Allocation of Transmission Capacity for AI Data Centers

**arXiv ID:** 2604.08854 | [PDF](https://arxiv.org/pdf/2604.08854v1)

**作者:** Shaoze Li `[一作]` (Dartmouth), Cong Chen `[通讯]` (Dartmouth)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了基于鲁棒优化与风险意识的框架，用于量化并分配AI数据中心接入电网的“固定位”与“灵活”传输容量，并采用同步递增拍卖实现容量的高效分配。

**💡 创新点**

创新点在于：① 将鲁棒优化与风险容忍度结合，既保证数据中心的可靠供电，又能解锁更多灵活容量；② 设计了将容量、风险水平和位置三维度的拍卖商品，证明在加性或对称凹价值函数下能收敛至竞争均衡，实现高效分配。

**🔧 技术方法**

技术主要包括：鲁棒优化模型、风险意识容量分配方法、同步递增拍卖算法（基于加性/对称凹价值函数的理论分析）以及仿真验证。

**📊 数据集**

没有使用公开真实数据集，主要通过基于典型负荷和电网拓扑的仿真数据验证方法。

**📈 对比分析**

通过仿真对比展示了在容忍极低概率服务中断和停电的前提下，灵活容量可提升约30%–50%，从而显著提升电网宿主容量；同时拍卖过程收敛快，分配效率高于传统分配方式。

**⚠️ 局限性**

局限性包括：① 依赖于准确的负荷预测与风险评估；② 拍卖模型假设参与者具备理性和可观测的价值函数；③ 对实际电网的动态运行约束（如潮流约束、设备故障）考虑不足，需要进一步的实测验证。

---

## 158. Scalable High-Recall Constraint-Satisfaction-Based Information Retrieval for Clinical Trials Matching

**arXiv ID:** 2604.08849 | [PDF](https://arxiv.org/pdf/2604.08849v1)

**作者:** Cyrus Zhou `[一作]` (Stanford University), Monica S. Lam `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于约束满足的临床试验检索方法，将试验和患者信息转化为SMT约束后进行匹配。

**💡 创新点**

创新点在于结合SNOMED CT医学本体与LLM进行语义解析，利用salience判定缺失数据，并将SMT约束映射到关系代数实现可扩展、可解释的检索。

**🔧 技术方法**

采用了SMT（Satisfiability Modulo Theories）、关系代数/SQL、LLM（GPT‑4.1/5）进行语义解析、salience评估与SMT编程。

**📊 数据集**

使用了SIGIR 2016 patient–trial test collection（59个合成病历与3621个临床试验）以及ClinicalTrials.gov snapshot。

**📈 对比分析**

与TrialGPT及其多种基线（BMRetriever、PubMedBERT、SapBERT等）对比，三种检索目标下平均检索相关且符合资格试验数提升3.25–11.76，召回率提升至92–94%，患者覆盖率提升至54/59，查询速度为每位患者约2.95秒。

**⚠️ 局限性**

局限性包括对LLM解析精度的依赖、缺失数据处理需手工设定salience、对部分医学术语覆盖不足，以及评估依赖有限的人工标签，未在真实临床环境中验证。

---

## 159. Spectral Geometry of LoRA Adapters Encodes Training Objective and Predicts Harmful Compliance

**arXiv ID:** 2604.08844 | [PDF](https://arxiv.org/pdf/2604.08844v1)

**作者:** Roi Paul `[一作]` `[通讯]` (Independent Researcher), Roi Paul (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LoRA权重增量的谱几何能否识别微调目标及其对下游行为的影响。

**💡 创新点**

实现了谱特征完美区分不同微调目标，发现训练目标是权重增量主轴，揭示了跨方法几何反转与权重/激活空间独立性。

**🔧 技术方法**

LoRA、谱特征提取、逻辑回归、PCA、激活探测、HEx-PHI、Llama-Guard、GPT-4o校准。

**📊 数据集**

HH-RLHF偏好数据集、HEx-PHI违规评测集、Llama-Guard-3-1B、GPT-4o校准。

**📈 对比分析**

预注册实验；AUC 1.00 的目标识别，Spearman ρ 最高 0.986 的剂量-响应，几何-行为 ρ=0.72，跨方法 AUC 0.00 指示反转。

**⚠️ 局限性**

仅单一 Llama-3.2-3B 模型、小样本、仅制造漂移、评测工具不匹配、未验证混合目标或多种基矩阵。

---

## 160. Discrete Meanflow Training Curriculum

**arXiv ID:** 2604.08837 | [PDF](https://arxiv.org/pdf/2604.08837v1)

**作者:** Chia-Hong Hsu `[一作]` (University of British Columbia), Frank Wood `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

我们提出了一种离散化MeanFlow（DMF）训练课程，利用从预训练流模型迁移的知识，在仅2000个epoch内完成一阶采样训练，并显著降低计算和数据成本。

**💡 创新点**

创新点在于将连续MeanFlow身份离散化为可实现一致性的目标，并设计基于VE映射的Δ衰减课程；通过几乎全MF目标的纯Fine‑tuning，实现了更快收敛且资源更友好。

**🔧 技术方法**

使用的技术包括MeanFlow、流匹配、离散MeanFlow目标、连续一致性训练、课程学习、VE扩散映射、EMA、JVP剔除、鲁棒Cauchy损失、软max核速度场，以及在latent空间使用SD‑VAE编码的图像生成。

**📊 数据集**

实验数据集主要是CIFAR‑10（像素空间）和ImageNet 256×256（通过SD‑VAE得到的latent空间），并在随机初始化与预训练模型两种设定下进行评估。

**📈 对比分析**

与传统MF从零开始、MF微调以及先前方法对比，DMF课程在CIFAR‑10上实现1‑step FID 3.36，仅用2000+2000 epoch，总GPU时长比MF快1.3×；在ImageNet 256×256上，DMF^†在6‑epoch即可达到21.18 FID，48‑epoch降至14.53，表现与基线相当但在更高预算时出现不稳定。

**⚠️ 局限性**

主要局限在于细粒度离散化阶段的训练不稳定，尤其在latent空间易出现发散；需要改进架构、归一化层和可能的二次引导训练来提升鲁棒性。

---

## 161. CatalogStitch: Dimension-Aware and Occlusion-Preserving Object Compositing for Catalog Image Generation

**arXiv ID:** 2604.08836 | [PDF](https://arxiv.org/pdf/2604.08836v1)

**作者:** Sanyam Jain `[一作]` (Adobe), Soo Ye Kim `[通讯]` (Adobe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `da1b1a89-583a-4b57-9c81-478778569bec` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CatalogStitch，一套模型无关的预后后处理技术，自动完成尺寸自适应掩模计算和遮挡恢复，帮助用户轻松生成高质量产品目录图像；

**💡 创新点**

创新点在于：①使用尺寸自适应掩模算法自动匹配不同产品长宽比；②采用遮挡自适应混合恢复策略，缓存遮挡像素后在生成后精确拼接；③提供CatalogStitch‑Eval 58例基准，验证通用性；

**🔧 技术方法**

主要技术包括：几何尺寸比计算与矩形扩展、实例分割与遮挡检测、基于生成式填补的遮挡去除、条件生成式合成、像素级alpha合成；

**📊 数据集**

使用CatalogStitch‑Eval数据集（35例尺寸不匹配+23例遮挡案例），包含预计算掩模、原始图片、生成结果与评估指标；

**📈 对比分析**

在ObjectStitch、OmniPaint、InsertAnything三种主流合成模型上做对比，加入CatalogStitch后AR误差从约30%降至4–5%，FID显著下降（如InsertAnything由105.99降至77.72），CLIP和DINO得分提升，遮挡PSNR显著提高；

**⚠️ 局限性**

局限性包括：假设遮挡物在产品前方，未处理多层遮挡；掩模扩展采用矩形，未考虑形状细节；对极端长宽比或半透明遮挡效果有限；

---

## 162. HiFloat4 Format for Language Model Pre-training on Ascend NPUs

**arXiv ID:** 2604.08826 | [PDF](https://arxiv.org/pdf/2604.08826v1)

**作者:** Mehran Taghian `[一作]` (Huawei), Shadan Golestan `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在华为 Ascend NPU 集群上，对 HiFloat4（HiF4）和 MXFP4 两种 4‑bit 浮点格式进行端到端 FP4 训练实验，涵盖稠密 Transformer（OpenPangu‑1B、Llama3‑8B）和混合专家（Qwen3‑MoE‑30B）模型，验证 90% 以上的计算与存储可在 FP4 下完成且相对 BF16 的损失不超过 1.5%。

**💡 创新点**

首次将 HiFloat4 应用于大规模 LLM 预训练，并与 MXFP4 进行系统对比，证明 HiF4 通过层级缩放天然具备更高稳定性，只需 RHT 等极少补偿，即可在保持 1% 以内的准确率的同时显著减少额外运算负担。

**🔧 技术方法**

采用 HiFloat4 的三层层级缩放、块级 FP4 量化、随机哈达玛变换（RHT）、随机舍入（SR）以及无截断缩放等技术，结合 Ascend NPU 的 FP4 硬件加速实现高效训练。

**📊 数据集**

使用常规 LLM 预训练语料库（如 Common Crawl、Wikipedia 等大规模文本数据），并在 OpenPangu‑1B、Llama3‑8B 与 Qwen3‑MoE‑30B 三个模型上验证。

**📈 对比分析**

通过对比 BF16 基线的训练损失曲线、下游任务精度以及模型尺寸与计算比例，结果显示 HiF4 与 MXFP4 在 1–1.5% 的相对损失范围内完成 90% 以上的 FP4 运算，并实现约 4 倍吞吐量提升与显著内存压缩。

**⚠️ 局限性**

仍需在高精度下执行的辅助运算（如 RHT、SR 等）导致部分计算开销增加，实验仅在 Ascend NPU 平台验证，其他硬件以及 RLHF、跨模态等更复杂场景的稳定性尚未充分验证。

---

## 163. Parametric Shortest Paths in a Linearly Interpolated Graph

**arXiv ID:** 2604.08892 | [PDF](https://arxiv.org/pdf/2604.08892v1)

**作者:** Jacob Sriraman `[一作]` (Montana State University), Binhai Zhu `[通讯]` (Montana State University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在两张加权有向图之间通过线性插值构造参数化图，本文计算所有参数化最短路径并构造查询数据结构。

**💡 创新点**

创新点在于使用递归划分法通过求解成本函数的交点来构造下层包络，避免枚举所有路径，时间复杂度为Θ(k|E|log|V|)。

**🔧 技术方法**

技术包括线性插值、下层包络计算、Dijkstra的变体以及二分查找。

**📊 数据集**

未使用实际数据集，研究为理论分析。

**📈 对比分析**

与以往方法相比，本文的查询时间为Θ(log k)，预处理时间为Θ(k|E|log V)，而以前的方法需要在每次查询时遍历路径，性能更优。

**⚠️ 局限性**

局限在于k可能呈超多项式增长，即分段线性函数的断点数可能非常大，导致预处理时间和存储空间受限。

---

## 164. Continuous Wavefront Design via Virtual Point Sources: A Holographic Paradigm for Near-Field XL-MIMO

**arXiv ID:** 2604.08908 | [PDF](https://arxiv.org/pdf/2604.08908v1)

**作者:** Xiyuan Liu `[一作]` (Tongji University), Jun Wu `[通讯]` (Fudan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了用于极大规模 MIMO（XL‑MIMO）系统的全息波束成形范式，并在双近场（DNF）场景下通过虚拟点源（VPS）方法实现非迭代、低复杂度的波束成形。

**💡 创新点**

核心创新包括：
- 以连续空间视角定义“全息阵列”并引入全息方法（先用几何模型求得波前，再映射回离散阵元），实现复杂度与阵列规模无关；
- 通过全息方法提出单一 VPS 的近似，并用几何光学“对齐三角”法解析确定最优 VPS 位置，完全消除传统 AO 迭代的耦合与局部最优问题；
- 形成一套完整的理论框架（衍射衰退原理、全息阵列定义、CWF 设计、DNF 解析）。

**🔧 技术方法**

使用技术包括：
- 连续电磁波函数（CWF）建模、平面波与球面波分解；
- 几何光学与 Huygens‑Fresnel 原理；
- 解析几何方法（对齐三角法）求解 VPS 位置；
- 非迭代 VPS 算法实现 IRS 辅助单用户系统中的波束成形；
- 采用仿真（LoS 球面波模型）验证方法。

**📊 数据集**

主要使用人工合成的 LoS 球面波仿真数据，设定 BS、IRS 与 UE 在二维平面上的位置与尺寸，采用多种参数变化（频率、角度、距离缩放等）进行实验，未使用公开真实数据集。

**📈 对比分析**

通过与传统 AO（不同初始化）进行对比，VPS 在迭代 0 时已取得更高性能，收敛后与 AO 最优结果相当或略优；在频率、角度、射程等多种场景下均保持优势；与随机 AO 的平均/最佳性能比较表明 VPS 作为初始化可显著提高收敛质量，且计算复杂度低、无收敛不确定性。

**⚠️ 局限性**

局限性包括：
- 主要针对 LoS 主导的近场环境，稀疏多径或强遮挡时单 VPS 近似效果可能下降；
- 依赖无限电气孔径/高频极限，实际硬件中阵列尺寸有限时全息假设可能不完全成立；
- 未考虑硬件失真、相位量化、功率控制等实际实现问题；
- 对于更复杂的多用户或多路径情形，需进一步扩展多 VPS 或其它模型。

---

## 165. GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing

**arXiv ID:** 2604.08896 | [PDF](https://arxiv.org/pdf/2604.08896v1)

**作者:** Aoran Xiao `[一作]` (RIKEN AIP), Naoto Yokoya `[通讯]` (RIKEN AIP)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GeoMMBench 专家制定的多学科、多传感器、多任务的多模态问答基准，并设计了 GeoMMAgent 多代理框架，用以提升 LLM 在地球科学与遥感任务中的理解与推理能力。

**💡 创新点**

创新点在于：①创建首个覆盖遥感、测绘、GIS、GNSS 等四大领域、光学、SAR、LiDAR、光谱等七种传感器的专家级问答基准；②提出基于检索、感知与推理三大工具的多代理系统，通过规划‑执行‑自评循环实现对复杂遥感问题的分解与协同解答。

**🔧 技术方法**

采用多模态 LLM（如 Qwen、Gemini 等）作为核心推理模型，并集成检索引擎（Wikipedia、Google、GME）、遥感感知模型（对象检测、场景分类、土地利用分割）以及自评模块，实现工具库与 LLM 的协同推理。

**📊 数据集**

使用 1,053 道图像-多项选择题组成的 GeoMMBench 数据集（验证集 37 题，测试集 1,016 题），题目涵盖光学、SAR、光谱、LiDAR、DEM 等多种传感器与遥感、测绘、GIS、GNSS 等四大学科的专家级知识。

**📈 对比分析**

在零样本设置下与 36 款开源/闭源 MLLMs 及文本 LLM 对比实验，GeoMMAgent 在测试集达 88.4%（接近人类专家 86.5%），超过所有单一模型；最高开源 MLLM Qwen3‑VL‑30B 仅 66.7%，闭源 Gemini‑1.5‑Pro 70.7%，展示了多代理框架在复杂遥感推理中的显著优势。

**⚠️ 局限性**

当前主要限制包括：多传感器视觉理解与空间关系推理仍较弱，尤其对非 RGB 影像和专门学科（GIS、测绘）知识缺乏；模型对传感器类型混淆、图像感知细粒度任务表现不佳；GeoMMAgent 依赖预先配置的工具库，扩展性与通用性需要进一步提升。

---

## 166. Harnessing Weak Pair Uncertainty for Text-based Person Search

**arXiv ID:** 2604.08877 | [PDF](https://arxiv.org/pdf/2604.08877v1)

**作者:** Jintao Sun `[一作]` (Beijing Institute of Technology), Gangyi Ding `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于不确定性估计的弱正样本学习框架，并引入组式图像-文本匹配（GITM）来提升文本检索性能。

**💡 创新点**

创新点在于：①将弱正样本的跨视角一致性转化为不确定性，并在对比损失中自适应加权；②通过组式匹配同时利用弱正样本和更多难负样本，提升特征空间分布。

**🔧 技术方法**

使用对比学习（ITC）、二分类匹配损失（ITM）、BERT文本编码、SwinV2-B视觉编码，并实现不确定性估计与正则化以及 GITM 组式采样。

**📊 数据集**

主要在三个公开文本检索数据集上评估：CUHK-PEDES、RSTPReid、ICFG-PEDES。

**📈 对比分析**

与多种现有 SOTA 方法对比，mAP 在 CUHK-PEDES、RSTPReid、ICFG-PEDES 上分别提升约 3%–7%，Recall@1/5/10 同时提升，整体表现优于基线和当前最强方法。

**⚠️ 局限性**

方法仅在微调阶段改动，对跨域迁移的鲁棒性仍需进一步验证；弱正样本质量过低时不确定性估计可能失效，导致训练不稳定。

---

## 167. Temporal Dropout Risk in Learning Analytics: A Harmonized Survival Benchmark Across Dynamic and Early-Window Representations

**arXiv ID:** 2604.08870 | [PDF](https://arxiv.org/pdf/2604.08870v1)

**作者:** Rafael da Silva `[一作]` (Eastern University), Gregory Longo `[通讯]` (Eastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的多维评估框架，对 14 种调优后的生存模型（包括线性、树集成、参数化和神经网络）在 OULAD 数据集上进行比较，分别在动态周人期和可比早期窗口两种时间表示下评估学生退学风险预测。

**💡 创新点**

创新点在于：①提出了同步评估的多指标（IBS、时间依赖 C‑index、Brier、校准、解释性、消融）协议；②通过消融与全局解释证明时间行为信号在退学预测中占主导；③提供了可复现的实验流程与代码。

**🔧 技术方法**

采用了生存分析方法（Cox、DeepSurv、Random Survival Forest、DeepHit、AFT 等）、离散时间灾害模型（线性、Poisson PIE、GB、CatBoost）、梯度提升树、神经网络、校准诊断、特征块消融与全局重要性分析。

**📊 数据集**

使用了 Open University Learning Analytics Dataset (OULAD)，包含 32,593 报名记录、7 模块、4 课堂，涵盖人口统计、学术成绩与 VLE 行为日志。

**📈 对比分析**

在每个臂内部按 IBS、时间依赖 C‑index、Brier@10/20/30、校准误差等指标比较；可比臂中 Random Survival Forest 在所有指标上领先；动态臂中 5 种模型聚集在 IBS 0.1396‑0.1412 区间，Poisson PIE 略占优势；整体结果显示时间行为信号对预测性能影响最大。

**⚠️ 局限性**

局限包括：仅使用单一 OULAD 数据集，无法评估跨机构或跨模块的迁移性能；bootstrap 只估计抽样方差，未考虑模型再训练的不确定性；未对动态臂做后验校准；评估范围仅限于共享课程上下文的训练‑测试拆分。

---

## 168. SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks

**arXiv ID:** 2604.08865 | [PDF](https://arxiv.org/pdf/2604.08865v1)

**作者:** Tianyi Wang `[一作]` (Southern University of Science and Technology), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Sequence-Level PPO（SPPO），通过把长链推理任务显式改造为序列级上下文Bandit，使用标量价值函数取代多样本基线，解决标准PPO在稀疏奖励下的信用分配与计算瓶颈。

**💡 创新点**

将推理视为单步上下文Bandit，利用单样本标量价值预测实现低方差优势信号，兼顾高采样效率与训练稳定性；同时引入Critic解耦方案显著降低显存占用。

**🔧 技术方法**

近端策略优化（PPO）、序列级上下文Bandit框架、二元交叉熵训练价值函数、Critic解耦、单样本更新与可验证奖励机制。

**📊 数据集**

数学推理基准：AIME24/25、AMC23、MATH、Minerva Math、DAPO‑17K、DeepScaleR等；以及控制任务的RLVR基准。

**📈 对比分析**

在1.5B、7B模型上与Base、标准PPO、ReMax、RLOO、GRPO等基线对比，SPPO平均得分分别超过GRPO（48.06/58.56），单样本训练速度提升约5.9×，显著优于多样本方法。

**⚠️ 局限性**

仅针对可验证的稀疏奖励任务设计，难以直接推广到开放式无客观评估的生成任务。

---

## 169. Stringology-Based Cryptanalysis for EChaCha20 Stream Cipher

**arXiv ID:** 2604.08862 | [PDF](https://arxiv.org/pdf/2604.08862v1)

**作者:** Victor Kebande `[一作]` `[通讯]` (University of Colorado Denver), Victor Kebande (University of Colorado Denver)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过改造 Knuth‑Morris‑Pratt (KMP) 与 Boyer‑Moore (BM) 两种经典字符串匹配算法，在 32‑bit 词级别上实现 EChaCha20 的 keystream 模式检测，并对 1 百万个 1152‑bit keystream 进行大规模实验，评估 8/16/32 位级别的模式出现频率以及对旋转差分攻击的抵抗能力；同时与传统统计测试、其他 ARX 流密码以及 KMP/BM 的单纯实现进行对比。

**💡 创新点**

① 将字符串匹配算法与密码学结合，构建基于词级别的 SBC（Stringology‑Based Cryptanalysis）框架；② 通过混合 KMP 与 BM 的跳跃启发式，显著提升搜索效率与检测精度；③ 在实验中首次展示 EChaCha20 在 16/32 位层面上保持均匀随机性，而 8 位层面存在可忽略的低熵偏差；④ 对旋转差分的实验表明 EChaCha20 在 3 轮内实现完整扩散，抵御旋转攻击。

**🔧 技术方法**

改造后的 KMP、Boyer‑Moore、混合 BM‑KMP、旋转差分实验、统计显著性检验（z‑score、χ²）、加速的 XOR 预处理、基于 32‑bit 词对齐的模式匹配。

**📊 数据集**

1 百万个 EChaCha20 keystream 块（每块 1152 位，36 个 32‑bit 词），在固定密钥+固定 nonce、以及变量密钥+随机 nonce 两种配置下生成，全部使用 CSPRNG 产生。

**📈 对比分析**

对比传统 KMP 与 BM，混合 BM‑KMP 在 3.6 GB/s 的吞吐量下，比单纯 BM 提升 12.5%（3.2 GB/s）且精度提高 18.3%（0.82→0.97）。与 Salsa20、ChaCha20、XChaCha20 的加密吞吐、扩散轮数、旋转差分抵抗等指标相比，EChaCha20 在保持或略高的性能下实现了更早的完整扩散（3 轮）和更强的旋转抗性。

**⚠️ 局限性**

① 仅在 32‑bit 词层面进行模式匹配，可能忽略多维/矩阵级别的隐藏结构；② 实验基于固定密钥与 nonce 的情景，未覆盖动态密钥调度的实际环境；③ 结果为经验性统计，缺乏形式化的安全证明；④ 仅针对 EChaCha20，无法直接推广到其他 ARX 流密码；⑤ 未考虑量子后攻击或侧信道等更复杂威胁。

---

## 170. Cross-Lingual Attention Distillation with Personality-Informed Generative Augmentation for Multilingual Personality Recognition

**arXiv ID:** 2604.08851 | [PDF](https://arxiv.org/pdf/2604.08851v1)

**作者:** Jing Jie Tan `[一作]` (Universiti Tunku Abdul Rahman), Kosuke Takano `[通讯]` (Kanagawa Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出ADAM框架，通过大语言模型在注重人格标签的条件下进行多语言数据生成（PIGA），随后利用教师-学生结构和注意力、对比损失进行跨语言知识蒸馏（CLAD），实现多语言人格识别；

**💡 创新点**

创新点在于将人格信息融入生成式翻译以提升语料质量（PIGA），以及引入三种损失（KL、对比、注意力）在跨语言蒸馏中实现人格特征的跨语言对齐与噪声抑制（CLAD）；

**🔧 技术方法**

使用技术包括OpenAI大语言模型进行条件生成、Transformer编码器（如XLM-R/Multilingual BERT）作为学生模型、教师模型（如PICEPR）、KL散度、对比学习和自适应注意力损失；

**📊 数据集**

数据集主要为English Big‑5 Essays（1,578样本）和Kaggle MBTI（5,552样本），并通过PIGA生成法扩展到法语、马来语、日语、中文四种语言；

**📈 对比分析**

与零射线跨语言、加权BCE、以及多种多语言嵌入基线（e.g., multilingual‑e5‑large‑instruct, Qwen3‑Embedding‑0.6B 等）对比，CLAD+PIGA在Essays上平均BA提升至0.6332（比基线提升约5.7%），在Kaggle上提升至0.7448（约9.7%），且在各语言和维度上均显著优于对照模型；

**⚠️ 局限性**

局限性包括潜在的标签泄露风险、对LLM生成质量的依赖、对二元人格标签的离散化假设、在真实世界多语言语料中的泛化验证不足以及对非标准语种的适用性待进一步探索。

---

## 171. Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs

**arXiv ID:** 2604.08846 | [PDF](https://arxiv.org/pdf/2604.08846v1)

**作者:** Jinqi Luo `[一作]` (University of Pennsylvania), René Vidal `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种利用大规模概念字典和稀疏自编码器(SAE)对多模态大型语言模型(MLLM)进行激活层的可控干预，提升模型安全性。

**💡 创新点**

创新点在于：①构建覆盖约15,000个多模态概念的字典；②用该字典初始化并训练SAE，自动标注SAE原子语义；③结合稀疏编码和多模态斜投影实现细粒度安全调控，避免传统方法的冗余或过度抑制。

**🔧 技术方法**

使用的技术包括CLIP特征检索、语义对比表示读取、稀疏自编码器( L1‑SAE 与 TopK‑SAE)、多模态斜投影(MOP)以及稀疏编码控制。

**📊 数据集**

主要数据集为从WordNet提取的15,661个概念，以及从CC‑3M检索的40万左右的标注图文对，用于构造概念字典和训练SAE。

**📈 对比分析**

与 Prompting、ActAdd、MOP 等基线在MM‑SafetyBench、JailBreakV‑28K 等安全基准上进行比较，结果显示在四种安全评测指标下均优于基线，同时保持或提升流畅度、PPL与MMMU等通用能力，推理时单词级延迟约提升14.6%。

**⚠️ 局限性**

局限性包括：①需要先前的概念字典构建与人工/专家标注；②对极端新颖概念的适应性仍有限；③在某些情形下对可解释性的自动化程度仍不够理想。

---

## 172. Finite-Sample Analysis of Nonlinear Independent Component Analysis:Sample Complexity and Identifiability Bounds

**arXiv ID:** 2604.08850 | [PDF](https://arxiv.org/pdf/2604.08850v1)

**作者:** Yuwen Jiang `[一作]` `[通讯]` (Guangzhou Institute of Science and Technology), Yuwen Jiang (Guangzhou Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文对非线性独立成分分析（ICA）进行了全面的有限样本分析，建立了样本复杂度的明确界限，首次提供了匹配的上下界。

**💡 创新点**

创新点在于提出了样本复杂度的明确界限 n = Θ((d + log(1/δ))/(ϵ^2 Δ))，揭示了样本复杂度与维度、精度和辅助监督之间的基本关系。

**🔧 技术方法**

使用了神经网络编码器和广义对比学习（GCL）框架，结合了伯恩斯坦不等式和自边界性质来实现快速收敛。

**📊 数据集**

通过模拟实验验证了理论预测，涉及15种配置，数据生成遵循非线性ICA模型。

**📈 对比分析**

与现有方法比较，本文的样本复杂度为Θ(d/(ϵ^2Δ))，收敛速率为𝒪(1/n)，优于之前的𝒪(1/√(n))，并且在维度和多样性方面的缩放规律得到了强有力的实证支持（R^2 > 0.999）。

**⚠️ 局限性**

限制在于实验主要使用合成数据，真实世界数据的验证仍需进一步研究，且样本复杂度的常数因子可能会有所不同。

---

## 173. Hierarchical Kernel Transformer: Multi-Scale Attention with an Information-Theoretic Approximation Analysis

**arXiv ID:** 2604.08829 | [PDF](https://arxiv.org/pdf/2604.08829v1)

**作者:** Giansalvo Cirrincione `[一作]` `[通讯]` (Université de Picardie Jules Verne), Giansalvo Cirrincione (Université de Picardie Jules Verne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Hierarchical Kernel Transformer（HKT），通过可学习的因果下采样构建多尺度序列表示，并在各尺度上独立计算注意力，再将其加权融合，从而提升长序列建模能力。

**💡 创新点**

创新点在于：①引入多分辨率注意力框架，①提供正半定核的理论保证和对称-反对称分解，②推导误差三项与几何衰减 bound，并证明 HKT 既严格包含标准注意力又包含因果卷积。

**🔧 技术方法**

采用多级因果卷积下采样、层级注意力与卷积混合头、动态融合权重、信息论误差分析以及 Mardia 多元峰度检验等技术。

**📊 数据集**

在合成 ListOps（T=512）、顺序 CIFAR-10（T=1024）和 IMDB 字符级情感分类（T=1024）上进行实验，并参考 Long‑Range‑Arena 公开基准。

**📈 对比分析**

与在相同实验设置下重新训练的多头自注意力（MHA）基线对比，HKT 在 ListOps +4.77pp、CIFAR‑10 +1.44pp、IMDB +7.47pp，计算开销约为 1.31×，保持可接受的效率。

**⚠️ 局限性**

局限性包括：仅在短序列（512/1024）上验证，未覆盖完整 LRA 任务；未对大宽度模型下的高斯性假设进行实证；层级结构在非结构化任务中的优势尚待进一步验证。

---

## 174. Semantic Zooming and Edge Bundling for Multi-Scale Supply Chain Flow Visualization

**arXiv ID:** 2604.08823 | [PDF](https://arxiv.org/pdf/2604.08823v1)

**作者:** Songmao Li `[一作]` (University of Southern California), Luciano Nocera `[通讯]` (University of Southern California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

做了一个多尺度供应链流量可视化仪表盘，结合语义缩放和边缘打包，实现从宏观路线到微观仓储层级的连续视图。

**💡 创新点**

创新点在于把Skeleton-Based Edge Bundling（SBEB）改造成适用于地理OD流的方向性聚类和自适应绕行约束，并在同一界面实现语义缩放的三层LOD切换。

**🔧 技术方法**

使用 Vue3 + Deck.gl + Mapbox GL JS + D3 进行浏览器端渲染，SBEB 通过 Web Worker 计算，支持动画过渡和交互式过滤。

**📊 数据集**

数据集为美国一家物流公司 2025 年 7 月 51,371 条订单，按发货仓库和州归约为 202 条 OD 流，并配备 4,000+ SKU 的仓储库存数据。

**📈 对比分析**

通过两场使用场景演示，展示系统可揭示跨国路线低效和需求库存不匹配；与传统无打包直线图相比，视觉清晰度提升，SBEB 计算约 2.5 秒，交互延迟 <50 ms。

**⚠️ 局限性**

限制包括：仅适用于四个仓库的 US 数据，SBEB 参数需针对不同网络调优，无法处理多月或跨国多仓网络，缺乏时间维度分析，且未进行正式用户评估。

---

## 175. How does Chain of Thought decompose complex tasks?

**arXiv ID:** 2604.08872 | [PDF](https://arxiv.org/pdf/2604.08872v1)

**作者:** Amrut Nadgir `[一作]` (University of Pennsylvania), Pratik Chaudhari `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了大规模语言模型在链式推理（Chain‑of‑Thought，CoT）中的分类误差随类数的幂律缩放，并基于此推导出最优树形推理结构和最佳推理长度；

**💡 创新点**

创新点在于将CoT视为一系列小规模分类任务，给出了理论上最优树分度m* = e^{d/2}，并预测并验证了存在“过度推理”阈值和最优深度，提供了对CoT有效性与效率的统一解释；

**🔧 技术方法**

技术主要包括：理论推导（概率、Lipschitz 常数、联合界、拉格朗日乘子法），经验验证（synthetic 结构化任务、真实推理数据），以及利用LLM隐含维度估计、熵与误差关系的数值实验；

**📊 数据集**

使用的数据集包括：CIFAR‑100（语义分组与随机分组实验）、synthetic树形任务、数学推理数据集GSM‑8k、MATH‑500、AIME 2022‑24 以及大型 LLM（Qwen2.5‑7B‑Instruct、Deepseek‑V3）的推理长度实验；

**📈 对比分析**

与直接预测（无CoT）对比，实验显示在类数较大且树分度接近或超过m*时，CoT 能显著降低误差；实验曲线表现为误差随推理长度先下降后上升，验证了理论预测的最优深度；

**⚠️ 局限性**

限制包括：假设各类样本均匀、转移动力学易学，适用于收敛型任务；不适用于发散型开放式任务；理论依赖于Lipschitz 常数与维度估计的近似；未考虑LLM内部学习过程的细节及多路径可选的真实复杂性。

---

## 176. Buying Data of Unknown Quality: Fisher Information Procurement Auctions

**arXiv ID:** 2604.08821 | [PDF](https://arxiv.org/pdf/2604.08821v1)

**作者:** Yuchen Hu `[一作]` (Massachusetts Institute Of Technology), Stephen Bates `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了数据市场中的统计参数估计，买家希望通过从不同数据提供者那里购买样本来估计参数，这些提供者在数据质量和提供成本上存在差异。

**💡 创新点**

提出了一种基于信息成本的第二价格采购机制，能够在数据质量未知的情况下，通过统计测试来验证报告的质量，从而激励卖家真实报告成本和质量。

**🔧 技术方法**

使用了第二价格拍卖机制和统计测试方法，结合了信息成本评分和质量验证。

**📊 数据集**

未具体提及使用的数据集，但讨论了在数据市场中不同提供者的样本质量和成本。

**📈 对比分析**

通过与传统的第二价格拍卖机制进行比较，证明了所提出机制在大样本情况下能够实现近乎真实的报告，且在参与者的激励和报告行为上表现良好。

**⚠️ 局限性**

机制在数据质量验证上存在一定的局限性，尤其是在样本量较小的情况下，可能导致参与者的报告不够真实。

---

## 177. Dissecting Bug Triggers and Failure Modes in Modern Agentic Frameworks: An Empirical Study

**arXiv ID:** 2604.08906 | [PDF](https://arxiv.org/pdf/2604.08906v1)

**作者:** Xiaowen Zhang `[一作]` (Concordia University), Shin Hwei Tan `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 5 个现代 LLM 代理框架的 409 条已修复缺陷进行系统性实证研究，提出 5 层架构模型和针对代理框架的症状、根因和触发情境的专用分类体系，分析缺陷集中层、常见症状与根因的关联，并评估缺陷触发场景在不同框架间的可迁移性。

**💡 创新点**

创新点包括：①将传统四层框架抽象升级为 5 层（Orchestration、Intelligence、Knowledge、Action、Infrastructure）以更贴合自主代理系统的架构；②提出涵盖“意外执行序列”“用户配置被忽略”等新症状与“模型相关错误”“认知上下文管理错误”“编排错误”等专属根因的细粒度分类；③系统挖掘触发因子（配置、输入特征、操作）并用 FP‑Growth 识别高频触发模式；④评估缺陷触发模式在 5 个框架间的迁移率（≈46%），为跨平台测试和共享基准提供依据。

**🔧 技术方法**

使用的技术主要有：①GitHub API 自动收集 issue–PR 对；②人工双人标注并用 Cohen κ 检验一致性；③统计分析（Jensen‑Shannon 相似度、卡方检验、Cramér’s V）揭示症状、根因、组件间关联；④FP‑Growth 频繁模式挖掘提取触发因子组合；⑤手工迁移实验验证跨框架可迁移性。

**📊 数据集**

使用的数据集为 5 个代理框架（LangChain、LangGraph、CrewAI、AutoGen、SmolAgents）GitHub 仓库中已修复的 issue–PR 对，累计 409 条缺陷；对每条缺陷进行症状、根因、组件、触发因子标注，形成内部数据库。

**📈 对比分析**

研究通过多维度统计比较（症状占比、根因分布、组件缺陷集中度、触发因子频次）展示结果；发现 Intelligence 层缺陷最多但测试覆盖率最低，触发因子主要是模型配置；跨框架迁移率约 46%，表明共享缺陷模式可作为跨平台验证的有效工具；未涉及性能基准，仅提供缺陷分布和关联性分析。

**⚠️ 局限性**

局限性包括：①仅覆盖 5 个框架，可能无法完全代表整个代理生态；②只分析已修复缺陷，未覆盖潜在未报告的缺陷；③人工标注仍存在主观偏差，尽管通过多人校验降低风险；④缺陷触发情境迁移实验仅在部分缺陷上进行，未覆盖所有类型；⑤研究主要关注框架层缺陷，未系统探讨应用层或任务层失效模式。

---

## 178. StaRPO: Stability-Augmented Reinforcement Policy Optimization

**arXiv ID:** 2604.08905 | [PDF](https://arxiv.org/pdf/2604.08905v1)

**作者:** Jinghan Zhang `[一作]` (Clemson University), Kunpeng Liu `[通讯]` (Clemson University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了StaRPO框架，利用强化学习结合局部和全局逻辑稳定性奖励（ACF、PE）提升LLM推理过程的逻辑一致性和最终答案准确率。

**💡 创新点**

首次将逻辑稳定性分解为可量化的局部自相关奖励和全局路径效率奖励，并将其作为过程级反馈直接融入RL优化；无需昂贵的外部判别器。

**🔧 技术方法**

使用强化学习中的Group Relative Policy Optimization (GRPO) 作为基座，加入ACF与PE稳定性奖励；通过嵌入向量计算自相关与路径效率；对RL训练进行过程级奖励设计。

**📊 数据集**

在四个推理基准上评估：GSM8K、MATH、GPQA-Diamond、AIME24。

**📈 对比分析**

与GRPO、CPPO、λ‑GRPO以及基于熵控制和外部规划的Baseline进行对比；StaRPO在所有数据集上均取得最高或第二高的最终答案准确率，并显著提升过程准确率，尤其在知识密集与长步推理任务上提升明显。

**⚠️ 局限性**

依赖于嵌入向量的稳定性指标，可能对模型规模和任务类型敏感；奖励权重需要调参，且目前仅针对算术/逻辑类任务，尚未验证在更广泛语义推理中的通用性。

---

## 179. AI-Induced Human Responsibility (AIHR) in AI-Human teams

**arXiv ID:** 2604.08866 | [PDF](https://arxiv.org/pdf/2604.08866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 180. Adaptive Dual Residual U-Net with Attention Gate and Multiscale Spatial Attention Mechanisms (ADRUwAMS)

**arXiv ID:** 2604.08893 | [PDF](https://arxiv.org/pdf/2604.08893v1)

**作者:** Mohsen Yaghoubi Suraki `[一作]` `[通讯]`, Mohsen Yaghoubi Suraki

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了一种新型3D U‑Net结构——Adaptive Dual Residual U‑Net with Attention Gate and Multiscale Spatial Attention Mechanisms (ADRUwAMS)，用于脑肿瘤分割。

**💡 创新点**

创新点包括：1) 采用自适应双残差块捕获多层次特征；2) 将注意力门与多尺度空间注意力机制融合，进一步提升对肿瘤细节的关注度；3) 在传统U‑Net框架中加入组归一化、ReLU激活以及多尺度卷积以增强网络表达能力。

**🔧 技术方法**

技术方法：3D卷积神经网络、残差连接、注意力门（Attention Gate）、多尺度空间注意力、组归一化、ReLU激活、Adam优化器、ReduceLROnPlateau学习率调度、五折交叉验证、统计检验（配对t检验与Cohen’s d）。

**📊 数据集**

使用公开数据集BraTS 2020（369例）和BraTS 2019（335例）进行训练、验证和测试；输入为四模态MRI（FLAIR、T1、T1ce、T2）拼接成4通道，尺寸裁剪为128×128×128。

**📈 对比分析**

通过与多种先进方法（Dual‑Path Attention U‑Net、3D Self‑Ensemble ResUNet、TransBTS、MENet等）以及基线3D U‑Net进行比较。ADRUwAMS在Dice分数上分别达到WT 0.9229、TC 0.8432、ET 0.8004，并在Hausdorff距离上取得显著更低的误差（WT 1.3228、TC 3.0375、ET 10.5287）。配对t检验显示差异显著（p<0.01），Cohen’s d表明效果尺寸大，表明改进在实际应用中具有重要意义。

**⚠️ 局限性**

局限性：1) 双残差块与注意力机制使用相同的卷积运算，可能未充分考虑不同层级特征的计算差异；2) 数据集规模仍有限，可能影响模型在更大多样化样本上的泛化能力；3) 训练时对高分辨率3D图像的计算资源有限，模型规模和推理速度受限。

---

## 181. Uncertainty-Aware Transformers: Conformal Prediction for Language Models

**arXiv ID:** 2604.08885 | [PDF](https://arxiv.org/pdf/2604.08885v1)

**作者:** Abhiram Vellore `[一作]` (Princeton University), Niraj K. Jha `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CONFIDE，一个基于中间层隐藏状态的 conformal prediction 框架，用于 BERT、RoBERTa 等编码器式 transformer 的不确定性量化与实例级解释。

**💡 创新点**

创新点在于将 CONFINE 的 kNN 语义距离非一致性度量与层级嵌入相结合，支持 CLS 或扁平化表示、PCA 降维、类条件校准，从而提升小模型的准确率和正确效率。

**🔧 技术方法**

采用了 conformal prediction、kNN（余弦/马氏距离）、PCA 降维、温度缩放、类条件校准以及中间层嵌入提取等技术。

**📊 数据集**

使用了 GLUE 与 SuperGLUE 任务（CoLA、MNLI、MRPC、QQP、RTE、SST‑2、BoolQ、CB、MultiRC 等）以及 BERT‑tiny/小型与 RoBERTa‑base/large 作为评测数据集。

**📈 对比分析**

与 NM1/NM2、VanillaNN、原始模型等基线比较，CONFIDE 在多数任务上保持或提升准确率，正确效率提升 3–5%，小模型可提升至 4.09% 的绝对精度，但仍存在少数类别欠覆盖。

**⚠️ 局限性**

局限包括对类不平衡/噪声任务下覆盖失效、需访问内部表示（限制 API 部署）、计算成本高（尤其扁平化嵌入），以及对交换性假设的敏感性。

---

## 182. HTNav: A Hybrid Navigation Framework with Tiered Structure for Urban Aerial Vision-and-Language Navigation

**arXiv ID:** 2604.08883 | [PDF](https://arxiv.org/pdf/2604.08883v1)

**作者:** Chengjie Fan `[一作]` (Nanjing University of Aeronautics and Astronautics), Jie Qin `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HTNav——一种结合模仿学习和强化学习的混合导航框架，利用宏观规划与微观行动的分层决策实现无人机在复杂城市环境中的长程路径规划与精准定位

**💡 创新点**

创新点包括：①分阶段训练的IL-RL策略，使得策略先在专家演示中获得稳定基线后在环境中进一步优化；②分层决策机制，将全局航路点生成与局部动作执行解耦；③基于残差网络和SCConv的地图表征学习模块，提升多源空间语义的保留与利用；④对CityNav数据集进行人工修正，提升基准质量

**🔧 技术方法**

采用的技术包括：多模态编码器（RGB、深度、地图），残差卷积网络、SCConv模块，三头预测（价值、进度、目标），Proximal Policy Optimization（PPO）强化学习，离线模仿学习（MSE损失），以及分层决策的宏观规划与微观控制网络

**📊 数据集**

使用CityNav数据集（原始版本及人工修订版），该数据集包含5,850目标对象、32,637指令-轨迹对，涵盖不同难度等级（Easy、Medium、Hard）

**📈 对比分析**

与Random、Seq2Seq+GSM、CMA+GSM、MGP、AerialVLN+GSM、FlightGPT等基线对比，HTNav在Test‑Unseen上NE、SR、OSR、SPL均实现了显著提升，达到或逼近人类水平（如SPL>20%），证明在复杂城市任务中实现了state‑of‑the‑art表现

**⚠️ 局限性**

局限性在于：1）仍与人类导航存在明显差距，尤其在空间推理和动态决策上；2）RL权重的调优需谨慎，过大易导致训练不稳定；3）对地图信息的依赖较高，若地图缺失或不精确会影响性能

---

## 183. BracketRank: Large Language Model Document Ranking via Reasoning-based Competitive Elimination

**arXiv ID:** 2604.08834 | [PDF](https://arxiv.org/pdf/2604.08834v1)

**作者:** Abdelrahman Abdallah `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于LLM的文档重排序框架BracketRank，将重排序视为一个包含自适应分组、推理增强提示和单淘汰式竞赛的“赛制”。

**💡 创新点**

创新点在于将推理过程嵌入每一次文档对比、使用自适应分组以满足上下文窗口限制，并通过赢家/输家双轨淘汰赛实现多轮系统性比较，提升了对推理密集查询的判定一致性。

**🔧 技术方法**

技术包括：自适应分组（根据LLM token限制动态切分候选文档）、基于步骤式推理的提示模板、单淘汰式竞赛结构以及并行化的多轮淘汰过程。

**📊 数据集**

使用的数据集包括推理密集检索基准BRIGHT（12个领域），传统检索基准TREC DL19/20、BEIR、NovelEval-2306，以及在不同检索器（BM25、Contriever）下的评估。

**📈 对比分析**

与现有排名方法（RankGPT、TourRank、Rank-R1等）以及监督模型（monoBERT、monoT5）相比，BracketRank在BRIGHT上平均nDCG@10达26.56，TREC DL19/20上nDCG@5分别为77.90/75.85，BEIR平均nDCG@10为54.66，显示显著性能提升。

**⚠️ 局限性**

主要局限包括对LLM上下文窗口的依赖（导致极大文档集需分批或截断）、推理提示质量受模型规模影响、以及在超大规模检索场景下仍存在API调用和推理延迟等运维成本。

---

## 184. Post-Hoc Guidance for Consistency Models by Joint Flow Distribution Learning

**arXiv ID:** 2604.08828 | [PDF](https://arxiv.org/pdf/2604.08828v1)

**作者:** Chia-Hong Hsu `[一作]` (Brown University), Randall Balestriero `[通讯]` (Brown University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Joint Flow Distribution Learning (JFDL)，一种在预训练的 Consistency Models 上后置、可调的指导方法，无需 Diffusion Model 教师即可实现 CFG 风格控制。

**💡 创新点**

创新点在于：① 在 CMs 上实现完全后置可调导引；② 通过伪噪声正态性分析与 Flow-based 模型连接给出理论保证；③ 兼容不具 ∅ 类的 CMs，采用 Random JFDL 进一步提升性能。

**🔧 技术方法**

使用的技术包括：JFDL (联合流分布学习)、一致性训练 (Consistency Training)、GradNorm 任务权重调节、EDM/EDM2-S 生成网络、CFG 对比、正态性检验等。

**📊 数据集**

使用的数据集为 CIFAR-10、ImageNet 64×64 以及若干 2D toy 数据。

**📈 对比分析**

通过与未指导 ECT 基线在 FID‑50k 量化指标比较，1 步采样时在 ω<2 的范围内 FID 从 3.40 降至 3.24（随机 JFDL）或 3.29（原始 JFDL），在 ImageNet 64×64 1 步采样时 FID 从 5.84 降至 4.38；与 CFG 在 DMs 中的行为相似。

**⚠️ 局限性**

局限性包括：① 对多步采样的理论尚未完善，2 步采样在 CIFAR‑10 上可能导致 FID 上升；② 需要预训练的 ∅ 类 CMs 或使用 Random JFDL；③ fine‑tuning 仍需额外 GPU 资源（约 30%）和 7.5% 训练数据；④ 仅在单步采样上验证了显著改进。

---

## 185. Adaptive Candidate Point Thompson Sampling for High-Dimensional Bayesian Optimization

**arXiv ID:** 2604.08891 | [PDF](https://arxiv.org/pdf/2604.08891v1)

**作者:** Donney Fan `[一作]` (University of British Columbia), Geoff Pleiss `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自适应候选点Thompson采样方法（ACTS），通过先采样高斯过程后验梯度并以此梯度为轴限定一个“轴对齐锥形”搜索空间，从而在保持全局一致性的前提下，在候选点上实现更高的密度与更有效的离散化。

**💡 创新点**

创新点在于：① 将后验梯度信息用于动态缩小候选点生成的搜索空间；② 以梯度为方向构造的轴对齐锥形搜索空间既显著压缩体积，又能保证包含比当前 incumbent 更高的函数值；③ ACTS可作为任何现有候选点策略（如Sobol、RAASP、CTS）的drop‑in替换，兼容本地搜索、批量搜索与可信域方法，且理论上保持TS的全局收敛性。

**🔧 技术方法**

主要技术包括：高斯过程回归（RBF/Matérn核），后验梯度采样与联合条件分布；在候选点集合上对后验进行Cholesky分解以生成采样；基于采样梯度构造锥形搜索空间；将任意基准候选点策略（如RAASP、Sobol、CTS）应用于该子空间；与TuRBO、批量TS等框架的无缝集成。

**📊 数据集**

在中等维度（d≤32）的机器人与控制任务（Lunar Lander 12D、Robot Pushing 14D、Swimmer 16D、Hopper 32D）以及更高维度（d≤1000）的任务上进行评估：Rover 60D、MOPTA08 124D、SVM 388D、LassoBench 180–1000D、GuacaMol 256D（SELFIES‑VAE潜在空间）。

**📈 对比分析**

与RAASP、CTS、Pathwise TS以及基于可信域的TuRBO、以及三种先进方法（SAASBO、BAxUS、LogEI）进行对比。实验结果显示：在绝大多数基准上，ACTS均能在相同候选点预算（M=10⁴）下取得更快的收敛速度和更高的最终目标值；在批量搜索（q=100）和高维问题中亦保持领先；在所有对照实验中，ACTS均优于或至少与最优方法持平。

**⚠️ 局限性**

局限性包括：① 需要可微的核函数（对非可微目标不适用）；② 仅利用单步梯度采样，若目标存在低内在尺度或极端非凸结构，锥形搜索可能过于局部；③ 依赖固定候选点数量（M=10⁴），在极大维度或极大样本数场景下仍受算子复杂度限制；④ 在极端噪声或不确定性较高的环境下，梯度采样的可靠性可能下降。

---

## 186. A Closer Look at the Application of Causal Inference in Graph Representation Learning

**arXiv ID:** 2604.08890 | [PDF](https://arxiv.org/pdf/2604.08890v1)

**作者:** Hang Gao `[一作]` (Chinese Academy of Science), Fengge Wu `[通讯]` (Chinese Academy of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于最小可分割单元的图表示学习因果建模理论，构建可控合成数据验证，并研发可插拔的冗余消除模块（REC）以提升因果学习性能。

**💡 创新点**

证明变量合并破坏因果推断假设，构造满足因果假设的结构因果模型，给出干预次数下界与简化条件，并提出REC模块。

**🔧 技术方法**

采用结构因果模型（SCM）、干预理论、图神经网络（GNN）及其冗余消除技术（REC），结合实验验证。

**📊 数据集**

使用合成的 RWG 数据集（Molecular、Citation）、SPMotif、真实数据集 CiteSeer、ENZYMES，以及与SPMotif结合的 SPMotif-M、SPMotif-C。

**📈 对比分析**

与 CaNet、CRCG、DIR、GCN、ChebNet、GIN 等基线在上述数据集上对比，REC 在大多数模型上均提升准确率，尤其在复杂图结构中显著提升。

**⚠️ 局限性**

需对底层因果结构具备一定先验，干预成本高，方法仍受图复杂性限制，且在真实复杂数据中可能无法完全消除混杂影响。

---

## 187. Simulation of Adaptive Running with Flexible Sports Prosthesis using Reinforcement Learning of Hybrid-link System

**arXiv ID:** 2604.08882 | [PDF](https://arxiv.org/pdf/2604.08882v1)

**作者:** Yuta Shimane `[一作]` (University of Tokyo), Ko Yamamoto `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于强化学习的混合链系统，利用叶片弹簧型运动义肢的柔性变形来仿真截肢者的步行和跑步运动；

**💡 创新点**

创新点在于将Piece-wise Constant Strain（PCS）柔性模型与多链刚体系统结合，并在强化学习框架中先用刚体模型预训练再细调至柔性模型，从而高效获得利用义肢弹性进行的全身动态控制；

**🔧 技术方法**

采用强化学习（PPO）、仿射动力学、PCS柔性模型、并行多进程训练与细调、运动捕捉反向运动学；

**📊 数据集**

使用单一受试者的运动捕捉与力板数据（跑步速度1.2–5.0 m/s），以及基于实验研究的义肢弹性参数；

**📈 对比分析**

通过与刚体模型的对比以及对不同虚拟刚度（柔软、标准、刚硬）条件下的GRF、关节轨迹和代谢耗能（COT）进行验证，结果显示柔性模型能更准确重现跑步动力学，并证明标准刚度下能量消耗最低；

**⚠️ 局限性**

仅基于单个受试者的数据，缺乏更大样本的验证，且对大幅度刚度偏差的实验验证不足，限制了通用性和外推性。

---

## 188. Revisiting the Capacity Gap in Chain-of-Thought Distillation from a Practical Perspective

**arXiv ID:** 2604.08880 | [PDF](https://arxiv.org/pdf/2604.08880v1)

**作者:** Tokio Kajitsuka `[一作]` (University of Tokyo), Sho Takase `[通讯]` (CyberAgent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对链式推理（CoT）蒸馏的容量缺口问题进行实用评估，并提出更符合实际部署的评估协议。

**💡 创新点**

创新点在于揭示先前评估忽略预蒸馏基线导致的误导，并证明在实际场景下容量缺口并非决定性因素，强教师的优势往往更重要；同时给出两条可直接落地的教师选择指南。

**🔧 技术方法**

采用硬链式推理蒸馏（hard CoT distillation）技术，使用 Qwen、Gemma-2 等模型族进行实验。

**📊 数据集**

实验数据集主要来自 BBH 基准（包括 Math、GSM8K、AIME、AMC、OlympiadBench 等多任务），并在部分实验中使用 Gemma-2 教师与 Qwen 学生进行跨族验证。

**📈 对比分析**

对比预蒸馏基线与多种教师-学生配置（小‑大、短‑长）在选定的 15 个 BBH 任务上的性能。结果显示：大多数任务蒸馏后提升，容量缺口仅在部分任务显现；当教师性能差距大时，使用更强教师可获得更好或相当的学生表现。

**⚠️ 局限性**

局限性包括仅在 BBH 上验证、单次实验无置信区间、任务选择阈值经验性、未能分离数据量与推理质量对容量缺口的独立影响。

---

## 189. A Mathematical Framework for Temporal Modeling and Counterfactual Policy Simulation of Student Dropout

**arXiv ID:** 2604.08874 | [PDF](https://arxiv.org/pdf/2604.08874v1)

**作者:** Rafael da Silva `[一作]` (Eastern University), Gregory Longo `[通讯]` (Eastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个以离散时间风险模型为核心的学生退学动态预测与情景模拟框架，能够在同一数据和模型上进行政策触发、机制感知和子组公平性评估。

**💡 创新点**

创新点在于：①将退学建模从静态风险评分转为时间索引的hazard估计；②在同一模型上通过预设的recency‑trigger规则实现结构性情景对比（冲击与机制感知）和子组差距变化分析；③引入双时域（T_policy、T_eval_policy、T_eval_metrics）评估策略，确保删失稳定性和结果可解释性。

**🔧 技术方法**

使用技术包括：离散时间逻辑回归（L1/L2正则化）并经过Platt校准；IPCW加权评估；机制感知与冲击情景的政策模拟；bootstrap不确定性估计；GroupKFold、时间分层划分等防泄漏手段。

**📊 数据集**

数据集为 Open University Learning Analytics Dataset（OULAD），集成了 VLE 交互日志、评估成绩和管理记录，覆盖 32,593 报名、28,785 学生。

**📈 对比分析**

通过 AUC_row、C‑index、Brier、IBS、ΔS(t)（情景生存差异）和 ΔGap(t)（子组差距变化）进行比较。模型在测试集上的 AUC_row 约 0.84，校准良好；在 18 周时冲击情景可提升 0.8%–8% 的生存率，机制感知情景差异不显著；性别子组差距在任何情景下的变化虽极小，但方向稳定。

**⚠️ 局限性**

局限性包括：①结果为模型假设下的结构性对比，非因果效应估计；②删失不完整导致高风险尾部样本稀少，校准不稳定；③机制感知参数固定，未探索多渠道干预；④跨课程、跨学科迁移性尚需进一步验证。

---

## 190. Hidden in Plain Sight: Visual-to-Symbolic Analytical Solution Inference from Field Visualizations

**arXiv ID:** 2604.08863 | [PDF](https://arxiv.org/pdf/2604.08863v1)

**作者:** Pengze Li `[一作]` (Fudan University), Xi Chen `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉到符号的解析解推断，在二维线性稳态场景中通过图像及其梯度信息生成完整可执行的SymPy表达式；

**💡 创新点**

提出ViSA任务与ViSA‑Bench基准；设计自验证的链式思维(CoT)对齐训练；利用视觉结构直接推断符号解法并验证一致性；

**🔧 技术方法**

使用Qwen3‑VL 8B视觉‑语言模型，程序化生成高质量CoT轨迹，配合自验证的结构匹配与参数推导；

**📊 数据集**

ViSA‑Bench：30个线性稳态场景，每场景500实例，共15k；其中1.5k用于金色CoT，150个用于测试；包含场值图、梯度图及少量辅助元数据；

**📈 对比分析**

在统一协议下与多种VLM/LLM基线对比，评估字符准确率、结构相似度与数值误差；ViSA‑R2整体分数0.512、结构0.860、数值0.385，明显优于开放源代码基线及部分闭源前沿模型；

**⚠️ 局限性**

仅覆盖线性稳态、无噪声；金色CoT标注成本高且难扩展；模型对非线性、时间相关或带噪声/部分可观测场景的鲁棒性不足；

---

## 191. Shortest Embeddings of Linear Codes with Arbitrary Hull Dimension

**arXiv ID:** 2604.08843 | [PDF](https://arxiv.org/pdf/2604.08843v1)

**作者:** Jiabin Wang `[一作]` (Central China Normal University), Jinquan Luo `[通讯]` (Central China Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了欧氏与赫米特内积下线性码的最短 t 维子壳嵌入问题，推导了最短长度公式并给出对应构造算法。

**💡 创新点**

创新点在于：①统一扩展了 LCD 与自正交嵌入的结果，得到任意 t 维子壳嵌入的精确长度；②对码的 Gram 矩阵进行四种类型分类，给出相应构造方法；③在二元码下改进了自正交嵌入长度的判定。

**🔧 技术方法**

主要技术包括：有限域上二次型与 Hermitian 矩阵的正交变换、秩与同构关系、矩阵分块对角化与构造、以及算法实现中的矩阵秩计算。

**📊 数据集**

实验使用了 BKLC（最优码）数据库中的已知最优码、Hamming 码、MDS 码等作为基准码进行构造与验证。

**📈 对比分析**

通过与 BKLC 数据库中的最优参数对比，构造出的码在多数情况下为最优或几乎最优，甚至发现了一些与 BKLC 不等价的新最优码。

**⚠️ 局限性**

局限性包括：仅针对有限域线性码；对高维嵌入（t>k）或非线性码缺乏理论；算法复杂度受矩阵对角化步骤的 O(n³) 限制。

---

## 192. PinpointQA: A Dataset and Benchmark for Small Object-Centric Spatial Understanding in Indoor Videos

**arXiv ID:** 2604.08991 | [PDF](https://arxiv.org/pdf/2604.08991v1)

**作者:** Zhiyu Zhou `[一作]` (Jilin University), Wen-Huang Cheng `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PinpointQA 数据集与基准，用于评估室内视频中小物体的定位、引用、细粒度描述和结构化空间推理能力。

**💡 创新点**

创新点在于把小物体空间理解拆分为四个递进任务（TPV、NRI、FSD、SSP），并通过中间空间表示实现自动 QA 生成与人工审核，填补了现有基准对小物体定位精度缺失的空白。

**🔧 技术方法**

使用多模态大型语言模型（LLMs）结合视觉编码器，采用 LoRA 微调提升性能，并借助 GPT‑5.4 作为判定者对自然语言答案进行评估。

**📊 数据集**

基于 ScanNet++ 与 ScanNet200 两大室内 3D 数据集构建，包含 1,024 场景、10,094 条 QA 对，涵盖目标出现、最近引用、细粒度描述及结构化预测四类任务。

**📈 对比分析**

在八种代表性多模态模型（GPT‑5.4、Kimi K2.5、LLaVA‑OneVision‑1.5、Qwen3‑VL‑8B、InternVL3.5‑8B、Spatial‑MLLM‑v1.1、SenseNova‑SI‑1.3‑InternVL3‑8B、Cambrian‑S‑7B）上进行对比实验，微调后 Qwen3‑VL‑8B‑SFT 在 Avg‑Micro 及 Avg‑Macro 上分别达 0.48 与 0.49，整体提升显著，但 SSP 仍是最薄弱环节。

**⚠️ 局限性**

限制主要在 SSP 的结构化定位仍显弱，模型难以将自然语言信息准确映射为可解析的 JSON；此外对更长时序、复杂遮挡或更大尺度空间关系的支持仍不足。

---

## 193. ActFER: Agentic Facial Expression Recognition via Active Tool-Augmented Visual Reasoning

**arXiv ID:** 2604.08990 | [PDF](https://arxiv.org/pdf/2604.08990v1)

**作者:** Shifeng Liu `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ActFER，基于多模态大型语言模型的主动表情识别框架，先动态调用人脸检测、对齐和局部放大工具获取视觉证据，再通过视觉链式推理（CoT）结合面部动作单元（AU）预测最终情绪。

**💡 创新点**

创新点在于：①将表情识别改为主动的视觉证据获取任务；②设计 Utility‑Calibrated GRPO（UC‑GRPO）学习何时、何处放大局部区域；③通过 AU‑grounded 级别奖励、查询级对比估计与情绪级 EMA 校准实现对局部放大效用的自适应评估；④在训练时利用合成多轮轨迹实现工具使用与推理的统一。

**🔧 技术方法**

使用技术包括：多模态 LLM（Qwen3VL‑4B）与工具调用接口（人脸检测/对齐、局部放大）；Chain‑of‑Thought 视觉推理；强化学习 GRPO 的变体 UC‑GRPO；AU‑grounded dense reward、查询对比效用估计、情绪级 EMA 校准；SFT + RL 两阶段训练。

**📊 数据集**

使用数据集：AffectNet、FERPlus、RAF‑DB、SFEW2.0（用于 FERBench 训练与评估），DISFA（用于零样本 AU 评估），并通过四个公开 FER 数据集构建 48k SFT 样本和 6.8k RL 样本，合成多轮轨迹。

**📈 对比分析**

在 FERBench 四个测试集上与通用 MLLM（如 Gemini、InternVL、LLaVA‑Next）和专门 FER 方法（EmoLA、ExpLLM、UniFER）对比，ActFER 取得 73.89% 准确率、67.45% macro‑F1，较最佳基线提升约 12% 准确率、22% F1；在 DISFA 上零样本 AU 的平均 F1 为 58.2，超越 Qwen3VL‑4B（≈38%）与 FEALLM（≈43%）等。

**⚠️ 局限性**

局限性包括：需要大量 RL 训练和精细调参；对工具（检测、对齐、放大）质量高度依赖，若工具失效会导致信息缺失；在极小样本或极难区分的表情中仍可能出现过度/不足放大；受交互步骤上限（4 步）限制，复杂场景下可能无法充分探索；实验主要在公开数据集上，缺乏真实场景下的鲁棒性验证。

---

## 194. Multi-agent Reinforcement Learning for Low-Carbon P2P Energy Trading among Self-Interested Microgrids

**arXiv ID:** 2604.08973 | [PDF](https://arxiv.org/pdf/2604.08973v1)

**作者:** Junhao Ren `[一作]` (Nanyang Technological University), Yajuan Sun `[通讯]` (Singapore Institute of Manufacturing Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个多微电网系统的P2P电力交易框架，并通过多智能体强化学习（MMAPPO）实现微电网自主出价、交易及储能调度，从而提升可再生能源利用、降低高碳电力依赖并提高社区经济福利。

**💡 创新点**

创新点在于：1）将P2P交易问题建模为DEC‑POMDP并采用CTDE训练，突破传统单一优化方法；2）提出MMAPPO框架，将LSTM与周期编码器结合，提取多尺度时序特征；3）将双重拍卖多轮清算机制（MRDAC）与MMAPPO融合，既保障个体利益，又提升系统碳减排效果。

**🔧 技术方法**

技术手段包括：多智能体近端策略优化（MAPPO）+LSTM + 周期性编码器；多轮双向拍卖清算（MRDAC）；集中训练、分散执行（CTDE）；环境仿真与奖励设计（基于利润与碳排放）。

**📊 数据集**

数据集主要来源于四个澳大利亚住宅家庭的日常负荷与光伏发电统计数据，并假设主电网FIT价为2$/kWh、紧急电价动态在[15,35]$/kWh；微电网参数（容量、储能特性等）从表格给出。

**📈 对比分析**

比较方法：与三种MARL基线（MIPPO、MAPPO‑one、MAPPO‑s）以及三种市场清算机制（MRDAC、VDA、Greedy）进行对比。实验显示，MMAPPO+MRDAC在训练收敛速度最快、总奖励最高（约-15/小时），总利润-123.81$/，紧急采购量最少30.33kWh，P2P交易量最大10.58kWh，显著优于其它组合。

**⚠️ 局限性**

局限性：1）未考虑线损与AC潮流约束，聚焦经济与碳计量；2）仅用四个微电网的仿真，缺乏大规模真实网格验证；3）日历采购策略固定，未优化；4）假设储能无能耗，可能偏离实际情况。

---

## 195. AudioGS: Spectrogram-Based Audio Gaussian Splatting for Sound Field Reconstruction

**arXiv ID:** 2604.08967 | [PDF](https://arxiv.org/pdf/2604.08967v1)

**作者:** Chunhao Bi `[一作]` (Shanghai Jiao Tong University), Zhengxue Cheng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了AudioGS框架，实现从稀疏音频观测直接重建三维声场，并在新视角下合成高保真双耳音频。

**💡 创新点**

创新点在于将声场显式建模为可学习的Audio Gaussians，利用球谐系数与距离衰减及相位校正，实现无视觉先验的精细空间化。

**🔧 技术方法**

采用声学基于频谱的三维高斯投影、球谐系数参数化、几何引导的距离衰减与相位校正，结合逆STFT合成双耳信号。

**📊 数据集**

在Replay‑NVAS真实室内多视角音频数据集上进行实验。

**📈 对比分析**

与Source Binaural、Mono、ViGAS、AV‑NeRF、AV‑Cloud等基线对比，AudioGS在MAG、ENV、DPAM等指标上分别提升约14%、6%与25%，并在空间准确度（LRE）和主观听感上获得最佳或近乎最佳成绩。

**⚠️ 局限性**

主要限制包括：仅针对静态单源场景，无法处理动态多源或实时更新；训练时仍需多视角音频对齐，缺乏对极端纹理缺失场景的鲁棒性。

---

## 196. Dynamic Class-Aware Active Learning for Unbiased Satellite Image Segmentation

**arXiv ID:** 2604.08965 | [PDF](https://arxiv.org/pdf/2604.08965v1)

**作者:** Gadi Hemanth Kumar `[一作]` (SRM Institute of Science and Technology), Pankaj Bodani `[通讯]` (Indian Space Research Organisation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于动态类别感知不确定性采样的主动学习框架（DCAU‑AL），用于在卫星图像语义分割任务中显著减少标注成本并提升少数类性能。

**💡 创新点**

创新点在于将实时类别 IoU 差距反馈融入像素熵不确定性权重，通过幂次调节参数 α 和自适应阈值 γ 动态重权，既能聚焦低性能类别，又兼顾探索多样性，避免传统不确定性方法的多数类偏差。

**🔧 技术方法**

技术包括 UNet 分割网络、像素级熵不确定性计算、类别性能 Gap 评估、幂次重加权、基于统计量的自适应阈值、迭代主动学习循环与人机协作标注。

**📊 数据集**

使用 OpenEarthLandCover 9 类卫星影像数据集（共 4300 张 1024×1024 像素图），对 500 张样本做初始标注，剩余 3300 张用于未标注池，验证在 20 次主动学习循环中的效果。

**📈 对比分析**

与全监督、Naive AL、Entropy 采样、Coreset 等基线对比；在仅使用 40% 标注数据的条件下，DCAU‑AL 达到 mIoU 0.664，几乎与全监督 0.670 一致，显著优于 Naive 0.642 与 Coreset 0.648，尤其在少数类如 Bareland、Road 等 IoU 上提升 20% 以上。

**⚠️ 局限性**

局限性包括对学习率和采样规模较为敏感；实验仅覆盖单一数据集，缺乏跨域验证；在高噪声标签或极端稀疏类场景下可能效果下降；实现复杂度高，需额外调参。

---

## 197. MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits

**arXiv ID:** 2604.08952 | [PDF](https://arxiv.org/pdf/2604.08952v1)

**作者:** Yixin Xiang `[一作]` (Nanjing University of Science and Technology), Jinhui Tang `[通讯]` (Nanjing Forestry University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种多臂老虎机（MAB）驱动的文档问答框架（MAB‑DQA），通过将查询拆分为多方面子查询并动态分配检索预算，提升检索与答案生成的准确性。

**💡 创新点**

创新点在于：①将查询拆分为子查询并视为多臂老虎机的臂；②利用初步视觉‑语言模型（VLM）评估结果作为奖励，采用Thompson Sampling动态平衡探索与利用；③通过超图构造查询特定的候选页面并结合反思式推理（HRRA）完成答案生成。

**🔧 技术方法**

核心技术包括：多臂老虎机（Bandit）与Thompson Sampling；视觉‑语言模型（如ColPali、Qwen‑2.5‑VL‑7B-Instruct）进行查询拆分与页面评估；超图（Hypergraph）建模页面关系；反思式推理代理（HRRA）进行多阶段验证。

**📊 数据集**

实验使用四个公开基准：MMLongBench、LongDocURL、PaperTab、FetaTab。

**📈 对比分析**

与纯VLM、基于多Agent的MDocAgent、以及现有RAG模型（M3DocRAG、MoloRAG、MoloRAG+）比较，MAB‑DQA在四个数据集上均取得显著提升，平均答案准确率提升约10.38%，在PaperTab提升18.5%；检索指标（Recall、Precision、NDCG、MRR）在所有Top‑K设定下均超过现有最佳方法。

**⚠️ 局限性**

局限性包括：高度依赖VLM的表现，若VLM在特定领域或低资源场景下效果差，整体性能会下降；方法对文档长度与复杂度有一定限制，检索多页需求时可能需调整参数；对超参数（α、β、λ）敏感，需要手工调优；仅使用Thompson Sampling，未评估其他Bandit策略（如UCB、Epsilon‑Greedy）可能带来的优势。

---

## 198. Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents

**arXiv ID:** 2604.08927 | [PDF](https://arxiv.org/pdf/2604.08927v1)

**作者:** Huangwei Chen `[一作]` (Zhejiang University), Lei Wu `[通讯]` (Zhejiang University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Aegle，一个虚拟多学科团队的同步多代理框架，用于提升门诊咨询的文档质量和诊断准确性。

**💡 创新点**

创新点在于将SOAP结构化状态与动态拓扑的多代理并行推理相结合，分离证据收集与诊断合成，并通过或chestrator动态激活专家。

**🔧 技术方法**

采用图形多代理架构，基于DeepSeek-V3.2 LLM的专门化代理，或chestrator和Aggregator进行协调与整合。

**📊 数据集**

使用ClinicalBench和RAPID-IPN两个真实医疗数据集，涵盖24科室和12科室的病例。

**📈 对比分析**

通过与多种单模型和多代理基线在文档质量、咨询能力和诊断准确性等53指标及诊断准确率进行对比，Aegle在所有指标均优于基线，诊断准确率提升至46.93%。

**⚠️ 局限性**

局限在于推理延迟、上下文窗口膨胀导致的计算开销，以及专家输出的冗余与重复，需进一步优化。

---

## 199. TAIHRI: Task-Aware 3D Human Keypoints Localization for Close-Range Human-Robot Interaction

**arXiv ID:** 2604.08921 | [PDF](https://arxiv.org/pdf/2604.08921v1)

**作者:** Ao Li `[一作]` (Tsinghua Shenzhen International Graduate School), Yansong Tang `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个针对近距离人机交互的视觉语言模型TAIHRI，用于精确定位任务相关的3D人体关键点，并能通过自然语言指令控制机器人。

**💡 创新点**

创新点在于将3D空间离散化为交互体素，利用2D关键点推理与next-token预测实现相机坐标系下的度量尺度定位，并通过任务感知提示引导模型聚焦特定部位。

**🔧 技术方法**

使用VLM框架（基于Qwen3-VL），结合相机内参注入、两阶段训练（SFT + GRPO强化学习）以及链式思考的2D→3D推理。

**📊 数据集**

主要使用新构建的CloseHRI数据集（约120万帧近距离视角），并辅以Human3.6M、MPI-INF-3DHP、BEDLAM等公开数据。

**📈 对比分析**

与现有全局3D姿态估计方法（CameraHMR、PromptHMR、SAM 3D Body）以及通用VLM（GPT-5.2、Qwen3-VL、Gemini-2.5 Pro）对比，在Harmony4D和EgoBody近距离测试中，TAIHRI在G-MPJPE上提升数十毫米，显著优于对照模型。

**⚠️ 局限性**

局限包括对摄像机内参的强依赖、在极端遮挡或多人场景下性能下降，以及模型规模与实时推理速度仍需提升。

---

## 200. Beyond Relevance: Utility-Centric Retrieval in the LLM Era

**arXiv ID:** 2604.08920 | [PDF](https://arxiv.org/pdf/2604.08920v1)

**作者:** Hengran Zhang `[一作]` (State Key Laboratory of AI Safety, ICT, CAS), Jiafeng Guo `[通讯]` (State Key Laboratory of AI Safety, ICT, CAS)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文围绕检索增强生成（RAG）时代的检索目标转变，提出了以LLM效用为核心的统一框架，涵盖LLM无关与专属效用、上下文无关与相关依赖效用，并讨论了LLM信息需求与代理式RAG的设计与优化路径。

**💡 创新点**

创新点在于：①将检索目标从传统的主题相关性迁移到面向LLM的效用优化；②系统性划分LLM-agnostic与LLM-specific、context-independent与context-dependent效用，为后续研究提供了明确的维度与评估视角；③提出将LLM信息需求显式化、与查询生成及强化学习结合的代理式RAG框架，进一步拓展检索与生成的协同优化。

**🔧 技术方法**

采用的方法主要是：基于文献综述与理论建模，构建效用定义与分类；利用生成式指标（如BLEU、ROUGE、EM、F1、答案概率）和注意力分布等作为LLM效用近似；结合查询生成、检索策略优化及强化学习（RL）框架，实现代理式RAG的实验验证。

**📊 数据集**

论文主要讨论的评估数据集包括MS MARCO、Natural Questions、TriviaQA等主流检索与问答基准，此外在代理式RAG实验中亦会使用如OpenAI API的LLM模型或本地训练的大模型作为下游生成器。

**📈 对比分析**

与传统基于nDCG、MAP、MRR的检索评估方法对比，本文提出的LLM效用评估更侧重答案正确性与生成质量，实验表明在多模型（GPT、LLaMA等）中，针对LLM-specific效用训练的检索器能显著提升答案准确率（+5%~10%）而跨模型泛化性略逊。

**⚠️ 局限性**

局限性包括：①缺乏统一的LLM效用评测基准，导致不同研究难以直接比较；②数据集上对LLM生成答案的人工或自动标注成本高；③上下文相关效用的建模与优化仍处于早期阶段，求解复杂度高；④代理式RAG中检索器多采用静态检索方法，未能充分利用动态查询与反馈循环的潜力。

---

## 201. Lightweight and Generalizable Multi-Sensor Human Activity Recognition via Cascaded Fusion and Style-Augmented Decomposition

**arXiv ID:** 2604.08910 | [PDF](https://arxiv.org/pdf/2604.08910v1)

**作者:** Wang Chenglong `[一作]`, Chen Xinlei `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量化、多模态融合与风格增强的多传感器人类活动识别框架

**💡 创新点**

1) Cascaded Fusion Block（CFB）替代传统注意力，采用压缩‑递归‑拼接‑融合结构实现无注意力高效特征交互；2) MixStyle（MoM）在局部和全局特征提取前进行低阶统计混合，提升泛化能力；

**🔧 技术方法**

1D卷积、深度可分卷积、全连接层、Mamba模块、MixStyle、CFB、深度可分卷积+点卷积的自注意力替代；

**📊 数据集**

Realdisp 与 Skoda 两个公开多传感器数据集；

**📈 对比分析**

在两数据集上与八种先进方法对比，准确率分别达到96.13%/97.50%，宏 F1 分别为95.30%/97.14%，相比第二好方法提升约3–5%；模型参数仅3.013M，推理速度8.8ms/样本，比对手快 30%+，显著低资源；

**⚠️ 局限性**

仅评估于两数据集；MoM 仅应用于两层，缺乏自适应混合策略；未结合自监督预训练，仍需大量标注数据。

---

## 202. AssemLM: Spatial Reasoning Multimodal Large Language Models for Robotic Assembly

**arXiv ID:** 2604.08983 | [PDF](https://arxiv.org/pdf/2604.08983v1)

**作者:** Zhi Jing `[一作]` (Fudan University), Chenjia Bai `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种基于多模态大语言模型的机器人装配系统（AssemLM）并创建了900K+样本的AssemBench数据集，用以实现精准的6D姿态预测和多步装配决策。

**💡 创新点**

创新点包括：① 在点云编码中采用SE(3)-equivariant的Vector Neuron DGCNN，保留几何旋转信息；② 通过embedding‑level融合和DeepStack多尺度注入将视觉、文本与几何特征统一；③ 设计离散化6D姿态tokenizer，使姿态预测可视为语言建模；④ 构造规模宏大的AssemBench，为多类别、多模态的装配推理提供标准基准。

**🔧 技术方法**

主要技术包括：SE(3)-equivariant点云编码、Qwen3‑VL‑2B‑Instruct LLM、SigLIP视觉编码器、DeepStack多尺度注入、M‑ROPE位置编码、离散化姿态tokenizer、两阶段监督微调（Geometry Warm‑up + Full Multimodal Alignment）。

**📊 数据集**

使用的主要数据集为自研的AssemBench（900k+样本、150k装配步骤、50+类别），以及在真实机器人实验中收集的实物点云与手册；对比基线如TwoByTwo、GPT‑5.2、DeepSeek‑V3.2等。

**📈 对比分析**

在AssemBench多类别基准上对比TwoByTwo和大模型，AssemLM平均成功率89.4%、RMSE 0.0203，显著优于基线；在IKEA数据的零样本测试中成功率81%；在Flexiv Rizon 4s 的四个真实装配任务中平均成功率50.8%，对比TwoByTwo的24%。

**⚠️ 局限性**

主要局限：多步任务中误差累积导致整体成功率下降；对高度对称或极小部件的旋转预测仍不够稳健；模型训练与推理需要大量算力；在更复杂或极端不确定环境下的鲁棒性尚未充分验证。

---

## 203. Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift

**arXiv ID:** 2604.08956 | [PDF](https://arxiv.org/pdf/2604.08956v1)

**作者:** Harshith Kethavath `[一作]` (University of Georgia), Weiming Hu `[通讯]` (University of Georgia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估CLIPSeg在Sentinel-2云/云影分割任务上，比较手工提示与监督微调（LoRA与全参数FFT）在不同标注比例下的效果，探究域差距对提示的限制。

**💡 创新点**

①系统性评估60种提示变体，发现提示无法弥补视觉‑语言域差距；②仅需0.1%标注即可超越零射基线；③揭示低秩适配在薄云与云影上性能不足，FFT显著优于LoRA；④提出“监督dip”现象——在极低标注率下先退化后恢复。

**🔧 技术方法**

使用CLIPSeg基础模型，进行提示工程、低秩适配LoRA、全参数微调FFT，结合多项损失（focal、Tversky、boundary）训练，采用RGB输入。

**📊 数据集**

CloudSEN12+ Sentinel‑2云/云影分割数据集，RGB三通道，训练8490图、验证535图、测试975图。

**📈 对比分析**

对比零射基线0.255 mIoU，LoRA与FFT在0.1%–100%标注比例下的mIoU曲线；FFT在100%标注时达到0.66 mIoU，LoRA 0.60 mIoU；FFT优于LoRA 0.03–0.09 mIoU，尤其在薄云与云影类显著。

**⚠️ 局限性**

仅评估CLIPSeg，未覆盖多光谱/多时相输入；未试验可学习提示；LoRA低秩限制未进一步优化；低标注采样未做分层；只使用RGB通道，结果可能不适用于其他传感器或任务。

---

## 204. Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication

**arXiv ID:** 2604.08944 | [PDF](https://arxiv.org/pdf/2604.08944v1)

**作者:** Benjamin Amoh `[一作]` (Dartmouth), Wesley Marrero `[通讯]` (Dartmouth)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种融合序列通信与决策聚焦学习的多智能体框架SeqComm-DFL，用于部分可观测环境下的协作决策。

**💡 创新点**

创新点包括①基于价值的消息生成，使消息直接提升接收者决策质量；②使用Stackelberg顺序条件和指导潜力对通信顺序进行优化；③将OMD扩展至通信增强的世界模型并通过隐式微分实现二层优化；④提供信息论界定和O(1/√T)收敛保证。

**🔧 技术方法**

技术方法包括值感知通信损失、消息细化网络、Gumbel-Softmax排序、顺序行动选择、对抗影响正则化、QMIX价值分解、内部循环的Critic训练、外部循环的真实环境评估、隐式微分（IFT）和共轭梯度求解。

**📊 数据集**

实验数据集包括自建医院协作环境（3名专家，100名病人）和StarCraft Multi-Agent Challenge（SMAC）多地图多人数。

**📈 对比分析**

与SeqComm、OMD、QMIX、MAPPO、MADDPG等基线对比，SeqComm-DFL在医院环境中奖励提升4–6倍、严重度改善提升3倍，在SMAC中赢率提升13–15个百分点，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：仅验证在离散动作空间，通信拓扑固定，训练需要显著warm‑up与超参数调优；对大规模智能体或连续动作的适用性尚未证明；在真实世界部署前仍需进一步评估安全性与带宽成本。

---

## 205. NCL-BU at SemEval-2026 Task 3: Fine-tuning XLM-RoBERTa for Multilingual Dimensional Sentiment Regression

**arXiv ID:** 2604.08923 | [PDF](https://arxiv.org/pdf/2604.08923v1)

**作者:** Tong Wu `[一作]` (Independent Researcher), Huizhi Liang `[通讯]` (Newcastle University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在SemEval‑2026 Task 3 Track A Subtask 1中，针对给定文本与特定方面，训练了XLM‑RoBERTa‑base模型，输出每个方面的连续情感维度（情绪-激活度）分数；同时将几种大型语言模型（GPT‑5.2、LLaMA‑3系列、LLaMA‑4‑Maverick）在少量示例下进行few‑shot提示实验，比较两类方法的效果。

**💡 创新点**

①首次将多语言、多领域的Aspect‑Based Sentiment Analysis扩展为连续的情绪‑激活度回归；②证明在此任务中任务专用的微调显著优于基于提示的LLM方法；③采用双回归头并通过sigmoid映射到[1,9]区间，得到可解释的连续分数。

**🔧 技术方法**

多语言预训练模型XLM‑RoBERTa‑base、两层MLP回归头、sigmoid缩放、AdamW优化器、线性学习率衰减、早停等标准训练技术。

**📊 数据集**

使用SemEval‑2026 Dimensional Aspect‑Based Sentiment Analysis数据集，覆盖英语与中文，餐厅、笔记本和金融三个领域，总共约1.5万条训练实例。

**📈 对比分析**

在20%验证集上将微调模型与LLM提示方法对比，微调模型在所有语言-领域组合上均优于LLM，平均RMSE_VA提升约0.77点（≈46%相对降低）。在正式测试集上，微调模型在中文数据上取得RMSE_VA≈0.54–0.95，远优于Kimi‑K2（≈2.1）和QLoRA‑Qwen‑3（≈2.8）基线。

**⚠️ 局限性**

仅使用[CLS]表示，未加入方面级注意力；LLM比较仅采用单一prompt模板与6个示例，可能低估LLM潜力；仅测试两种语言，泛化性尚未验证。

---

## 206. MV3DIS: Multi-View Mask Matching via 3D Guides for Zero-Shot 3D Instance Segmentation

**arXiv ID:** 2604.08916 | [PDF](https://arxiv.org/pdf/2604.08916v1)

**作者:** Yibo Zhao `[一作]` (Nankai University), Jin Xie `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 MV3DIS，一种基于 2D 先验（SAM）进行零样本 3D 实例分割的 coarse‑to‑fine 框架；

**💡 创新点**

创新点在于 ① 用 3D 先验投影作为参考实现跨视角 3D‑guided mask matching，显著提升多视角一致性；② 引入深度一致性权重抑制遮挡误差；③ 通过一致性掩码引导区域细化，实现更精细的 3D 实例分割；

**🔧 技术方法**

采用了 Segment Anything Model（SAM）/SAM2 生成 2D 掩码、图割/超点生成、投影 + 3D 覆盖分布、余弦相似度一致性评分、基于图的相似性聚类、region growing 与重细化；

**📊 数据集**

在 ScanNetV2、ScanNet200、ScanNet++、Replica 与 Matterport3D 等室内 RGB‑D 数据集上进行评估；

**📈 对比分析**

与闭集方法 Mask3D 以及多种开放词汇 SAM‑based 方法（SAM3D、SAI3D、SAM‑graph、Open3DIS、SAM2Object 等）对比，MV3DIS 在 AP_25/AP_50/mAP 上均位居前列，甚至在 ScanNet200 上超过 Mask3D；

**⚠️ 局限性**

局限性包括对极端遮挡或少量视角的鲁棒性仍有限；对 SAM 掩码质量高度依赖，若 SAM 产生错误掩码会影响后续 3D 对齐；

---

## 207. Accessible Fine-grained Data Representation via Spatial Audio

**arXiv ID:** 2604.08979 | [PDF](https://arxiv.org/pdf/2604.08979v1)

**作者:** Can Liu `[一作]` (Nanyang Technological University), Yong Wang `[通讯]` (Nanyang Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种基于空间音频的数值可视化方法，利用声源在方位角平面中的方向来表示数据值，从而实现对盲/低视障者的细粒度数据可视化。

**💡 创新点**

首次将空间音频方向编码应用于数据可视化，实现无语音说明的精细数值传达，突破了传统音高编码在细粒度信息传递上的局限。

**🔧 技术方法**

使用头相关传递函数(HRTFs)生成空间音频，利用Faust Live实现声音的双耳处理，并在常见的AirPods等通用耳机上实现方向渲染。

**📊 数据集**

采用[-10,10]范围内的整数作为实验数据，分别构造值比较、趋势识别、符号识别和精确值识别四类任务的数据集，共计12对比较数据、12个趋势组和21个单值数据。

**📈 对比分析**

通过26名参与者（10名BLV）在四个任务上与传统Pitch编码进行对比，采用Wilcoxon检验；结果显示在符号识别、精确值识别和趋势识别任务中，空间音频显著优于Pitch，值比较任务则略逊于Pitch。

**⚠️ 局限性**

局限性包括依赖通用HRTF导致定位误差、需要支持空间音频的耳机、方向可辨别范围有限、样本量偏小以及BLV与视障者在年龄和教育水平上的差异可能影响实验结果。

---

## 208. WOMBET: World Model-based Experience Transfer for Robust and Sample-efficient Reinforcement Learning

**arXiv ID:** 2604.08958 | [PDF](https://arxiv.org/pdf/2604.08958v1)

**作者:** Mintae Kim `[一作]` (UC Berkeley), Koushil Sreenath `[通讯]` (UC Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并利用基于世界模型的经验转移框架WOMBET，实现从源任务到目标任务的离线到在线强化学习

**💡 创新点**

联合生成与利用先验数据，采用不确定性惩罚的MPC生成可靠轨迹，并通过双准则筛选与自适应采样实现稳健的迁移

**🔧 技术方法**

世界模型（ensemble）、不确定性惩罚的MPC、双准则过滤、基于LayerNorm与ensemble最小化的Critic、TD误差自适应采样

**📊 数据集**

MuJoCo连续控制基准（如HalfCheetah、Walker2d、Ant等）

**📈 对比分析**

与SAC、PPO、TD3、MOPO、COMBO、IQL、CQL等在线或离线-在线基线对比，WOMBET在样本效率和最终回报上均优于这些方法

**⚠️ 局限性**

仍受限于模型误差与环境差异，双准则阈值需要手动设定，且在极端源-目标差异下效果可能下降

---

## 209. MASS: Mesh-inellipse Aligned Deformable Surfel Splatting for Hand Reconstruction and Rendering from Egocentric Monocular Video

**arXiv ID:** 2604.08943 | [PDF](https://arxiv.org/pdf/2604.08943v1)

**作者:** Haoyu Zhu `[一作]` (Hong Kong Polytechnic University), Yi Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用 Mesh-inellipse 对齐的可变形 Surfel splatting 方法从单目自我视角视频重建高保真3D手部模型。

**💡 创新点**

创新点在于将参数化手部网格通过 Steiner Inellipse 直接初始化 2D 高分辨率 Gaussian Surfel，结合哈希网格+MLP 的可变形网络以及两阶段训练和绑定损失，提升几何细节与纹理逼真度。

**🔧 技术方法**

主要技术包括 Mesh-to-Surfel 转换、Steiner Inellipse 计算、Gaussian Surfel Deformation（哈希网格+MLP）、2D Gaussian splatting 渲染、两阶段优化和绑定损失。

**📊 数据集**

使用 ARCTIC、Hand Appearance、InterHand2.6M 等 egocentric 单目数据集进行训练与评估。

**📈 对比分析**

与 HARP、3D-PSHR 等现有方法对比，在 PSNR、MS-SSIM、LPIPS 等指标上均超过 SOTA，并且推理速度快于 HARP，训练时间缩短至 5.5 分钟。

**⚠️ 局限性**

局限在于仅建模手部表面，无法处理手-物体交互细节，受单目遮挡、光照变化影响，且不具备全局世界尺度一致性。

---

## 210. Delve into the Applicability of Advanced Optimizers for Multi-Task Learning

**arXiv ID:** 2604.08939 | [PDF](https://arxiv.org/pdf/2604.08939v1)

**作者:** Zhipeng Zhou `[一作]` (Nanyang Technological University), Chunyan Miao `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结合自适应动量与轻量方向保留的框架，用来提升先进优化器在多任务学习中的即时去冲突效果。

**💡 创新点**

创新点在于发现动量机制会稀释MTL的即时去冲突，并提出基于局部曲率自适应调节动量的策略，同时为Muon's orthogonalization 设计轻量方向保留方法，使其更适合作为隐式MTL学习器。

**🔧 技术方法**

使用了自适应动量策略（APT）、轻量方向保留（LDP）、Muon's 伪正交化、以及传统梯度/损失平衡算法（如CAGrad、FairGrad、MGDA等）等技术。

**📊 数据集**

实验数据集包括 Cityscapes、NYUv2、CelebA 与 QM9 四大主流多任务学习基准。

**📈 对比分析**

在这些基准上与多种基线（LS、SI、RLW、DWA、UW、MGDA、PCGrad、CAGrad、Nash-MTL、FAMO、FairGrad、COST）进行对比，采用 Δm% 与 Mean Rank 指标，所提出框架在所有基线上均提升平均排名和 Δm%，并实现或逼近 SOTA。

**⚠️ 局限性**

局限性在于极大任务数下的平衡仍有限（如 CelebA 40 任务时 FairGrad 性能略下降），并且依赖于动量与曲率估计的准确性，可能在梯度噪声较大或非平滑任务分布下表现受限。

---

## 211. Speed Thrills: Visceral Demonstrations That Get Students Excited About Efficient Algorithms

**arXiv ID:** 2604.08938 | [PDF](https://arxiv.org/pdf/2604.08938v1)

**作者:** Alistair Moffat `[一作]` (University of Melbourne), David Hawking `[通讯]` (Australian National University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在算法教学中呈现“算法激动”序列（从朴素到高效的逐步改进），以实时演示的方式激发学生对高效算法的兴趣。

**💡 创新点**

创新点在于提出“thrills of algorithms”概念，强调通过可视化、即时计时和渐进式性能提升来使算法学习更具感官冲击力；并将该方法系统化为可复用的教学框架。

**🔧 技术方法**

使用的技术包括：从最粗糙的平方级计数器到基于质因子筛、素数数组、埃拉托色尼筛的多种质数计数实现；以及从暴力匹配到后缀数组、三路快速排序等多种重复子串检测实现；配合 C 语言实现与实时计时、电子表格分析。

**📊 数据集**

数据集：1）质数计数测试覆盖到 10¹²（并演示 10¹⁵ 并行实验）；2）重复子串检测使用 266.6 MiB 的带 SGML 标记的《华尔街日报》文本文件（m=50、m=250）。

**📈 对比分析**

对比方法：对质数计数，Method 0（O(n²)) 需 250 s 处理 n=10⁶；Method 5（分段埃拉托色尼）在相同硬件上仅 3 s 处理 n=10¹²，性能提升约 10⁶ 倍；对重复子串，Method 1（暴力 O(n²m)）预估需数小时，Method 3（三路快速排序后缀排序）仅需几秒，提升约 10⁵ 倍。

**⚠️ 局限性**

局限性：方法本身非新颖，主要用于教学演示而非作业；在最坏情况仍可退化到 O(n²)；分段筛需要额外内存管理；极大规模（10¹⁸ 以上）仍受限；对抗性输入可能导致三路快速排序退化。

---

## 212. Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning

**arXiv ID:** 2604.08926 | [PDF](https://arxiv.org/pdf/2604.08926v1)

**作者:** Taojie Zhu `[一作]` (Tsinghua University), Yonghong He `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过动态难度分级将SFT与RL结合，构建统一的后训练框架DYPO，提升LLM推理性能。

**💡 创新点**

创新点在于结构化缓解SFT偏差与RL方差冲突：动态难度分级、集成多教师蒸馏与Group Alignment Loss (GAL)。

**🔧 技术方法**

采用SFT、GRPO、GAL、Multi-Teacher Distillation以及动态门控技术实现统一优化。

**📊 数据集**

主要使用OpenR1-Math-220k、NuminaMath 1.5以及DeepSeek-R1、Qwen3-235B-A22B等多源教师生成的数据。

**📈 对比分析**

与SFT、RL、SFT→RL、SuperRL、CHORD等基线对比，DYPO在5个数学推理基准及2个OOD任务上平均提升约10%~13%，显著优于现有方法。

**⚠️ 局限性**

主要局限在对逻辑任务的适用性强，对开放式文本任务验证不足；训练需要8条轨迹导致计算开销高，样本效率较低。

---

## 213. Customized Fusion: A Closed-Loop Dynamic Network for Adaptive Multi-Task-Aware Infrared-Visible Image Fusion

**arXiv ID:** 2604.08924 | [PDF](https://arxiv.org/pdf/2604.08924v1)

**作者:** Zengyi Yang `[一作]` (Hefei University of Technology), Huafeng Li `[通讯]` (Kunming University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `da1b1a89-583a-4b57-9c81-478778569bec` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Closed‑Loop Dynamic Network（CLDyN），一种能够在不重新训练融合网络的前提下，依据不同下游任务的语义需求动态补偿红外-可见图像融合结果的框架。

**💡 创新点**

核心创新包括：① 语义传递链与闭环优化机制，允许下游任务网络向融合网络反馈语义信息；② 需求驱动语义补偿（RSC）模块，结合Basis Vector Bank（BVB）与Architecture‑Adaptive Semantic Injection（A2SI）实现任务定制化网络结构；③ reward‑penalty策略根据任务表现变化自适应调节RSC，避免语义漂移。

**🔧 技术方法**

使用的技术与方法包括：闭环优化与语义传递链、BVB+ A2SI 的可动态生成卷积核、正交卷积原型、奖励‑惩罚损失、CAGrad梯度冲突缓解、Adam优化、Softmax等；下游任务采用YOLOv5、SegFormer与CTDNet；训练阶段分两步，第一步训练可视引导融合网络，第二步冻结融合网络只训练RSC。

**📊 数据集**

主要数据集：M^3FD、FMB、VT5000 用于多任务评估；LLVIP、MSRS、RoadScene、M^3FD（部分）用于预训练融合网络；下游任务网络的训练数据来自同一数据集。

**📈 对比分析**

与CoCo、TDAL、IRFS、SMiF、MRFS、TIMF、SAGE等传统融合方法在MI、Q_AB/F、Q_CB、Q_CV、Q_CC等融合指标上始终位列首位；与“任务网络重训练”与“联合训练”方法相比，CLDyN在目标检测、语义分割、显著目标检测三个任务上均保持或超过最高分，且可训练参数仅0.46M、FLOPs仅174.06G，显著低于对比方法。跨检测器泛化实验亦表明补偿后在DETR和YOLOv5上的mAP提升。

**⚠️ 局限性**

限制与不足：RSC模块的补偿范围受预设任务集限制，若遇到完全不同的任务或语义，效果可能下降；需要预先训练并固定下游任务网络；对实时性与移动端部署未做深入评估；在极端环境或极少数据的场景下语义补偿的泛化能力仍待验证。

---

## 214. Large-Scale Universal Defect Generation: Foundation Models and Datasets

**arXiv ID:** 2604.08915 | [PDF](https://arxiv.org/pdf/2604.08915v1)

**作者:** Yuanting Fan `[一作]` (Tencent), Chengjie Wang `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了大规模缺陷生成框架UniDG，能够在无/少样本条件下通过参考图像或文本指令生成高质量、真实感且与目标场景一致的缺陷；

**💡 创新点**

创新点包括：①构建300K样本的UDG数据集，实现多模态缺陷生成的统一训练；②提出Defect-Context Editing与MM-DiT多模态注意力的融合，支持参考式和指令式缺陷生成；③两阶段训练策略Diversity‑SFT + Consistency‑RFT，通过奖励模型提升多样性与一致性；

**🔧 技术方法**

采用Flux系列预训练图像修复模型、MM‑DiT架构、RoPE、RMS‑Norm、流匹配训练、流重放梯度优化、LoRA微调、SigLIP特征提取等技术；

**📊 数据集**

使用UDG数据集（300K normal‑abnormal‑mask‑caption 量级，覆盖28类缺陷），以及在MVTec‑AD、VisA等公开数据集上进行评估；

**📈 对比分析**

与多种少样本缺陷生成和图像插值/编辑基线（AnoDiff, AnoGen, DualAnoDiff, SeaS, AnyDoor, InsertAnything）对比，UniDG在合成质量（IL/MLLM指标）和下游异常检测/定位（AUROC, AP, mIoU）均显著提升，单/多分类性能提升超过10%~20%；

**⚠️ 局限性**

局限性包括模型参数量大（12B）导致推理内存占用高，且评估指标仍需更针对缺陷场景的专用IQA指标；

---

## 215. PilotBench: A Benchmark for General Aviation Agents with Safety Constraints

**arXiv ID:** 2604.08987 | [PDF](https://arxiv.org/pdf/2604.08987v1)

**作者:** Yalun Wu `[一作]` (National University of Singapore), Boyang Wang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PilotBench 基准，用于评估 LLM 在受安全约束的通用航空代理中的飞行轨迹与姿态预测能力。

**💡 创新点**

创新点在于构建了 9 阶段标注的 708 条真实航迹数据集、设计了 60/40 的复合评估指标（回归精度+指令遵循+安全合规），并揭示了精度-可控性二分法与动态复杂度缺口，推动混合架构研究。

**🔧 技术方法**

采用了 41 种 LLM 与传统时序预测基线的对比实验，使用结构化提示、In‑Context Learning、Physics‑CoT 进行推理，并通过自定义的复合指标评估回归误差与指令/安全合规度。

**📊 数据集**

使用了 708 条来自 31 条 VFR 循环赛道（DA40）和 22 条双指令航班（C172N）的同步 34 通道传感器/航电数据，涵盖 9 个飞行阶段。

**📈 对比分析**

通过 MAE、RMSE、指令遵循分数等指标对比；LLM 在指令遵循上达到 86–89% 但 MAE 在 11–14 之间，传统基线回归更精确（MAE≈7.01），并在攀升/进近阶段显著退化。

**⚠️ 局限性**

局限性在于 LLM 对高动态阶段的物理建模脆弱、数值精度不足，且部分模型会拒绝指令，需通过混合 LLM 与专用预测器的架构来弥补。

---

## 216. Litmus (Re)Agent: A Benchmark and Agentic System for Predictive Evaluation of Multilingual Models

**arXiv ID:** 2604.08970 | [PDF](https://arxiv.org/pdf/2604.08970v1)

**作者:** Avni Mittal `[一作]` (Microsoft Corporation), Monojit Choudhury `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Litmus (Re)Agent，一种用于在不完整文献证据下预测多语言模型性能的基于DAG的多代理系统，并构建了覆盖六项任务、五种证据情景的受控评估基准。

**💡 创新点**

创新点在于：①将可获取证据与真实结果分离，形成受控情景；②通过DAG拆分假设并分别检索、聚合证据；③扩展专家知识库、引入语言学特征库，并改进提示与推理流程。

**🔧 技术方法**

主要技术包括：多代理DAG编排、检索增强推理（ReAct/Toolformer）、专家知识库查询、语言学特征提取与回归模型、结构化聚合与引用生成。

**📊 数据集**

使用的主要数据集是从公开的多语言评测论文中提取的，构成了1,500个任务–模型–语言问题的基准，覆盖代码生成、数理推理、问答/视觉问答、文本分类、摘要、机器翻译等六类任务。

**📈 对比分析**

与五个基线（LITMUS++、ThoughtAgent、Single Agent、GPT‑4.1直接推理、Magentic‑One）进行对比，Litmus (Re)Agent在数值预测任务的平均绝对误差(MAE)整体最低（10.3），在比较推理任务的准确率最高（21.6%），并在转移性强的情景中提升最显著。

**⚠️ 局限性**

局限性包括：仅在GPT‑4.1上实验，未验证更小或开源模型的泛化；Coder代理的代码执行成功率低；基准仅覆盖六项文本任务，缺乏多模态或安全性评估；评估基于已发布论文，可能存在测量噪声与报道不一致。

---

## 217. How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms

**arXiv ID:** 2604.08966 | [PDF](https://arxiv.org/pdf/2604.08966v1)

**作者:** Shengji Jin `[一作]` (University of Central Florida), Chen Chen `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种视频时序输出范式（文本数字生成、时间标记生成、连续时间解码）在统一的紧凑型多模态大语言模型上进行对照实验，评估其定位精度与系统效率；

**💡 创新点**

在同一模型、同一数据、同一训练协议下，系统化比较输出范式的实际性能差异，揭示连续时间解码的“范式红利”，并构建效率-准确率 Pareto 前沿；

**🔧 技术方法**

利用LoRA微调、三种输出头（文本生成、时间标记、分布式MLP）、自动回归与非自动回归推理、统一视觉编码器和统一训练策略；

**📊 数据集**

使用约1.2M的视频-文本标注样本（覆盖moment retrieval、问答、密集字幕），在Charades-STA、QVHighlights、YouCook2三大基准上评测；

**📈 对比分析**

对比方法：在相同的backbone (0.5B–8B) 下，连续时间解码在所有任务上取得最高mIoU和R1@0.5，同时在推理延迟与参数占比上优于其他两种范式；

**⚠️ 局限性**

局限：连续解码缺乏原生的显著性评分，无法直接完成高亮检测；输入侧时间表述未被探索；在极端多步因果推理任务上仍表现不佳。

---

## 218. Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models

**arXiv ID:** 2604.08964 | [PDF](https://arxiv.org/pdf/2604.08964v1)

**作者:** Shun Zou `[一作]` (University of Science and Technology of China), Xiangxiang Chu `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Anchor-based History-stable Decoding (AHD)，通过动态锚点实时监测历史轨迹，提前解锁跨块稳定 token，从而在 diffusion 大语言模型中显著减少解码步数并提升生成质量。

**💡 创新点**

创新点在于引入历史一致性指标与动态锚点来捕捉绝对稳定趋势，突破传统 Semi-AR 解码的块边界延迟；且该方法无训练、可插拔，兼容多种 dLLM。

**🔧 技术方法**

主要技术包括 KL 散度历史一致性评估、动态锚点策略、基于置信度的并行解码以及对历史缓冲区的加权求和。

**📊 数据集**

实验覆盖语言领域（LLaDA-8B-Instruct、LLaDA-1.5）、视觉-语言领域（MMaDA-8B-MixCoT）、音频-语言领域（DIFFA）等多种数据集，包括 HumanEval、MBPP、BBH、MMLU-Pro、TruthfulQA、Math、Asdiv、MATH-Vision、MathVista、ScienceQA、GQA、MME、VoiceBench 等。

**📈 对比分析**

与基线 Naïve、PC-sampler、Fast-dLLM、KLASS、Saber 等先进解码策略比较，AHD 在所有七个语言基准上平均提升约 1–3.7 分，同时把解码步数降低 70–80%，在多模态和音频基准中同样实现 1–2× 的速度提升和 1–3% 的性能提升。

**⚠️ 局限性**

局限包括：参数需针对不同任务微调；主要验证规模约 8B 参数，对更大模型（72B/256B）尚未评估；虽然额外计算 negligible，但仍有优化空间。

---

## 219. Efficient Hierarchical Implicit Flow Q-learning for Offline Goal-conditioned Reinforcement Learning

**arXiv ID:** 2604.08960 | [PDF](https://arxiv.org/pdf/2604.08960v1)

**作者:** Zhiqiang Dong `[一作]` (Shandong University), Guoqiang Wu `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种高效的层级隐式流 Q 学习方法（HIFQL），用于离线目标条件强化学习，解决多模态行为生成和长时程规划问题。

**💡 创新点**

创新点在于将均值流（Mean Flow）引入层级策略，替代单峰高斯政策，支持一次性采样；同时引入 LeJEPA 目标表征编码器，提升高维目标嵌入的判别性和泛化能力。

**🔧 技术方法**

使用技术包括：层级隐式流 Q 学习、均值流政策、LeJEPA 目标表征、优势加权回归（AWR）以及期望回归（Expectile Regression）。

**📊 数据集**

实验数据集为 OGBench benchmark，包含状态基和像素基的 PointMaze、AntMaze、HumanoidMaze 等 Maze 环境。

**📈 对比分析**

与 GCBC、GCIVL、GCIQL、QRL、CRL、HIQL 等基线进行比较，HIFQL 在大多数 PointMaze 及视觉 AntMaze 任务中显著优于 HIQL，平均成功率提升显著，但在 AntMaze、HumanoidMaze 上提升有限。

**⚠️ 局限性**

主要局限在于：当低层动力学成为瓶颈时，高层策略表达提升效果不明显；对最具挑战的像素 teleport 环境提升有限；且对高维动态任务的进一步提升仍需探索。

---

## 220. From Distance to Angle: One-Shot Detection Under Additive White Cauchy Noise

**arXiv ID:** 2604.08949 | [PDF](https://arxiv.org/pdf/2604.08949v1)

**作者:** Yen-Chi Lee `[一作]` `[通讯]` (National Central University), Yen-Chi Lee (National Central University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在加性白色Cauchy噪声（AWCN）下有限星座的一次性检测，推导了小噪声与大噪声两种极限下的误码概率上界与极限表达式。

**💡 创新点**

创新点在于：①证明AWCN下的ML决策仍为欧氏Voronoi分区，但可靠性由全局距离谱与角度（Voronoi散度角）决定；②给出递归距离谱上界与角度收敛定理；③揭示了噪声幂律尾导致的“几何崩塌”现象。

**🔧 技术方法**

主要技术包括：几何概率论、α‑stable 分布性质、极限分析（主导收敛定理）、Voronoi 边界与散度角计算、距离谱与角度度量的解析表达。

**📊 数据集**

没有使用公开数据集，本文以理论推导为主，并通过四点异构星座与标准4QAM作为示例验证。

**📈 对比分析**

比较方法：在小噪声下使用递归距离谱上界与Gaussian Q函数的近似；在大噪声下比较角度收敛极限与理想化的角度比例。性能表现：在AWCN下误码率远高于AWGN，且随噪声尺度增大时误码率趋于与角度占比相关的固定值，展示了距离与角度两种可靠性描述的转变。

**⚠️ 局限性**

局限性：仅针对等概率符号与等方位Cauchy噪声；未考虑编码与多比特错误；大噪声极限假设无界距离；解析结果对高维星座的可扩展性与计算复杂度未给出完整评估。

---

## 221. MuTSE: A Human-in-the-Loop Multi-use Text Simplification Evaluator

**arXiv ID:** 2604.08947 | [PDF](https://arxiv.org/pdf/2604.08947v1)

**作者:** Rares-Alexandru Roscan `[一作]`, Angela-Liliana Dumitran `[通讯]` (Universitatea Crestina Dimitrie Cantemir)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 MuTSE，一款支持多模型、多提示并行生成并进行人机交互式评估的网页应用，用于文本简化的系统化评价与教育级别定制。

**💡 创新点**

创新点在于：①异步并行生成架构实现 P×M 组合的实时展示；②三层次语义对齐引擎配合可调线性惰性 λ，兼顾语义与位置匹配；③可自定义评价维度与权重的评分框架，支持即时可视化与导出，真正实现人机协同的高维度评估。

**🔧 技术方法**

技术栈包括：后端 Python + FastAPI + Together AI API；前端 Vue.js3 + 交互式可视化；NLP 组件 spaCy、sentence‑transformers、TF‑IDF；语义对齐使用余弦相似 + 位置惩罚；读ability 评估采用 Flesch‑Kincaid 与 Flesch‑Reading‑Ease；数据持久化使用本地 JSON；并行处理通过 asyncio.gather 实现。

**📊 数据集**

未使用公开大规模标准数据集，而是基于用户上传的源文本（可覆盖 CEFR A2、B1 等教育目标），并可在后续导出为 JSON/CSV 供研究者用于构建自己的简化语料库。

**📈 对比分析**

比较方法：在同一源文本上生成所有 P×M 组合，利用线性惰性 λ 计算多对多对齐，实时展示对齐分数与可视化；同时输出多维度读ability 与压缩率指标，支持手工评分加权合成综合得分。性能上，异步并行显著缩短总耗时（瓶颈为最慢模型），线性惰性降低对高维 embedding 的需求，支持 CPU 环境下快速交互。

**⚠️ 局限性**

局限性：①本地 JSON 持久层不支持多用户协同，需迁移到关系数据库；②部署依赖 Python/Node 环境，仍有配置门槛；③线性惰性 λ 在跨语言翻译或大幅结构重排的任务中可能失效，需重新校准；④缺乏系统性实验数据与基准对比，性能评估主要基于可视化与人工评分。

---

## 222. Predictive Entropy Links Calibration and Paraphrase Sensitivity in Medical Vision-Language Models

**arXiv ID:** 2604.08941 | [PDF](https://arxiv.org/pdf/2604.08941v1)

**作者:** Binesh Sadanandan `[一作]` (University of New Haven), Vahid Behzadan `[通讯]` (University of New Haven)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在医学视觉‑语言模型（VLM）上系统评估了五种不确定性量化方法，并通过预测熵证明误差检测与问题改写（paraphrase）敏感性是同一几何现象的两个表现。

**💡 创新点**

创新点在于：① 将模型的预测熵与改写导致答案翻转的概率关联，提出一个统一阈值即可同时筛选错误与不稳健预测；② 揭示 LoRA 集成在跨站点（MIMIC→PadChest）下会因成员不一致而失效，强调了在医学部署中需做成员级别 OOD 验证。

**🔧 技术方法**

使用的技术包括：单前向软最大熵、温度缩放、MC Dropout、基于 LoRA 的深度集成以及决策边界绝对值（margin）；同时引入 UQ‑Paraphrase 关联分析、风险‑覆盖曲线与可分辨性熵分解。

**📊 数据集**

数据集：MedGemma‑4B‑IT 与 LLaVA‑RAD‑7B 两大 VLM；MIMIC‑CXR（98 例二分类）作为内部验证；PadChest（861 例）提供跨站点测试与改写翻转样本；以及基于 ImageNet‑C 的五种人工破坏条件用于模拟分布偏移。

**📈 对比分析**

比较结果：单前向熵的错误检测 AUROC 为 0.743，改写翻转检测 AUROC 为 0.711；MC Dropout 误差 AUROC 0.715，改写 AUROC 0.709；深度集成仅 0.657。风险‑覆盖方面，Softmax 在 5% 风险下覆盖率 7.3%，MC Dropout 提升至 21.5%，而集成在该阈值下覆盖率为 0%；温度缩放与 Softmax 近似。

**⚠️ 局限性**

局限性包括：仅测试单一跨站点迁移；仅评估二分类是/否问题，未覆盖自由文本生成；LoRA 集成的失败可能与模型特定，未在 LLaVA‑RAD 上验证；MIMIC 子集样本量小；人工破坏仅近似真实临床偏移。

---

## 223. From Indiscriminate to Targeted: Efficient RTL Verification via Functionally Key Signal-Driven LLM Assertion Generation

**arXiv ID:** 2604.08932 | [PDF](https://arxiv.org/pdf/2604.08932v1)

**作者:** Yonghao Wang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Li `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 AgileAssert 框架，利用关键信号驱动的 LLM 自动生成 RTL 断言，实现高效验证。

**💡 创新点**

创新点在于混合评分与冗余感知的 Top‑K 关键信号选取、结构感知 RTL 切片以及针对 LLM 的精准上下文提示，从而显著降低断言量和 Token 消耗。

**🔧 技术方法**

使用了 RTL 语义图构建、PageRank/可观测度/输出增强/多路分支分数混合评分、双向 Jaccard 相似度去重、结构感知切片以及 GPT‑5.1 生成断言。

**📊 数据集**

数据集涵盖 20 个区块级 RTL 设计和 3 个 CPU 级工业级 RTL 设计（如 I2C、SHA3、Pairing、picorv32 等），共计数百万行代码。

**📈 对比分析**

与 AssertLLM、AssertGen、AssertMiner 在相同硬件/模型（GPT‑5.1）下对比，AgileAssert 平均减少 66.68% 断言，提升分支/语句/切换/COI 覆盖率约 14–21%，误检率提升，且 Token 消耗降低 64%，错误检测率提升 72.74%。

**⚠️ 局限性**

局限性：对极度重复结构（如 SHA3 的 generate）排名效果欠佳；部分关键信号难以被 LLM 正确解释导致断言准确率不高；仍需人工校验或手工补充。

---

## 224. Degradation-Robust Fusion: An Efficient Degradation-Aware Diffusion Framework for Multimodal Image Fusion in Arbitrary Degradation Scenarios

**arXiv ID:** 2604.08922 | [PDF](https://arxiv.org/pdf/2604.08922v1)

**作者:** Yu Shi `[一作]` (Hefei University of Technology), Xun Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种降解感知扩散框架，用于在噪声、模糊、低分辨率等复杂降解下直接融合多模态图像。

**💡 创新点**

创新点：不显式预测噪声，而是直接回归融合图像；联合观测校正机制同时满足降解与融合约束；可在少量扩散步数内实现。

**🔧 技术方法**

采用扩散模型（DDIM）+联合观测校正、伪逆投影、无监督损失，结合多任务网络学习融合权重。

**📊 数据集**

使用M3FD（红外-可见图像融合）和Harvard PET–MRI数据集。

**📈 对比分析**

与八种主流融合方法（IFCNN、U2Fusion、MURF、DDFM、Text-DiFuse、VDMUFusion、RFfusion、Mask-DiFuser）比较，在噪声、模糊、复合降解三种场景下在六个评估指标上均取得最佳或次佳，显示出更高的信息保真和细节重建。

**⚠️ 局限性**

局限：仍比传统神经网络推理慢，扩散步数需平衡速度与质量；对极端降解或多源异构时的泛化尚待验证。

---

## 225. How Do LLMs See Charts? A Comparative Study on High-Level Visualization Comprehension in Humans and LLMs

**arXiv ID:** 2604.08959 | [PDF](https://arxiv.org/pdf/2604.08959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 226. PerMix-RLVR: Preserving Persona Expressivity under Verifiable-Reward Alignment

**arXiv ID:** 2604.08986 | [PDF](https://arxiv.org/pdf/2604.08986v1)

**作者:** Jihwan Oh `[一作]` (KAIST AI), Se-Young Yun `[通讯]` (KAIST AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练 LLM 使其对 persona prompt 产生的行为差异最小化，同时保持任务性能。

**💡 创新点**

提出 PerMix-RLVR——在训练时通过 persona 混合的可验证奖励对齐，解决 RLVR 在表达角色时的忠实度下降问题。

**🔧 技术方法**

使用强化学习与可验证奖励（RLVR）、Persona-mixed 训练、GRPO 优化等技术。

**📊 数据集**

在 GSM8K、MATH500、AIME2024、LiveCodeBench 和 PersonaGym 等数学与代码推理数据集上评估。

**📈 对比分析**

与 SFT、KD、RLVR 等后训练方法对比，PerMix-RLVR 在 Persona Stability Score（+21.2%）和 PersonaGym 一致性（+11.4%）上均优于对手，同时显著提升最差-case 性能。

**⚠️ 局限性**

仍依赖于精心设计的 persona 词表，且对非可验证任务的鲁棒性和训练成本需要进一步研究。

---

## 227. Confident in a Confidence Score: Investigating the Sensitivity of Confidence Scores to Supervised Fine-Tuning

**arXiv ID:** 2604.08974 | [PDF](https://arxiv.org/pdf/2604.08974v1)

**作者:** Lorenzo Jaime Yu Flores `[一作]` (Mila Quebec AI Institute), Jackie Chi Kit Cheung `[通讯]` (Mila Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了监督微调（SFT）对语言模型不确定性量化（UQ）指标与输出质量相关性的影响，并分析其误相关机制。

**💡 创新点**

发现不同SFT设置会显著破坏概率型和自一致性型UQ指标的相关性，并指出误相关主要来自于模型对训练分布相似度的敏感性。

**🔧 技术方法**

采用 Spearman 相关、AUROC、统计显著性检验等评估方法，比较多种概率与一致性置信度指标在多模型、多任务下的表现。

**📊 数据集**

使用翻译（Eng–Afr）、问答（SQuAD）、算术（GSM8K）以及 TruthfulQA 进行实验。

**📈 对比分析**

与预训练模型相比，SFT后约 33% 的配置出现相关性下降，AUROC 在 47% 的情况下降，表明误相关会显著降低下游识别正确答案任务的性能。

**⚠️ 局限性**

局限性包括：仅评估了平均对数概率和 BLEU 方差等少数指标，未覆盖更广泛的 UQ 方法；使用 Spearman 相关存在假设缺陷；实验范围仅涵盖三种任务，结果的普适性待验证。

---

## 228. Modality-Aware Zero-Shot Pruning and Sparse Attention for Efficient Multimodal Edge Inference

**arXiv ID:** 2604.08971 | [PDF](https://arxiv.org/pdf/2604.08971v1)

**作者:** Yueyuan Sui `[一作]` (Northwestern University), Stephen Xia `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发一种零调优的多模态压缩框架，结合模态感知裁剪与稀疏组查询注意力，实现边缘设备高效推理。

**💡 创新点**

提出模态感知结构门控（可在推理时仅前向计算的门控网络）与稀疏组查询注意力，并利用一阶梯度稀疏监督对齐生成无梯度推理时可用的重要性分数，完成零调优的多模态裁剪。

**🔧 技术方法**

基于Transformer的多模态时间序列模型、稀疏组查询注意力（Sparse Grouped‑Query Attention）、模态感知结构门控、梯度稀疏对齐目标、结构化剪枝与后训练量化。

**📊 数据集**

WESAD（压力监测）、DaliaHAR（人体动作识别）和DSADS（多模态睡眠/情绪）三大多模态时间序列数据集。

**📈 对比分析**

与随机、Magnitude、SynFlow等传统剪枝及原始Dense Self‑Attention对比；在三套数据集和多种backbone上，平均准确率提升12.7%，四模态丢失时提升13.4%；GFLOPs降低约15%–29%，内存下降28%，延迟降低至1.63×，在强基线上取得显著优势。

**⚠️ 局限性**

仍需在更广泛硬件和极端高剪枝比例下验证鲁棒性；稀疏组查询参数与门控阈值需手动调优；对非时序多模态或图像场景的通用性有限；训练过程中需额外的对齐损失与梯度稀疏监督。

---

## 229. TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice

**arXiv ID:** 2604.08948 | [PDF](https://arxiv.org/pdf/2604.08948v1)

**作者:** Gang Hu `[一作]` (Yunnan University), Kun Yu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了中文税务实践评估基准TaxPraBen，整合了10个应用任务和3个真实场景，共14个数据集共约7.3千条样本，并在零样本与一样本设置下对19款主流LLM进行了系统评估。

**💡 创新点**

首次构建专属中文税务实践基准并引入结构化评估范式，结合Bloom认知层级组织任务，并采用统一JSON输出协议与ChatGPT结构解析提升评测可靠性；同时，专家手工标注确保数据真实贴近行业实践。

**🔧 技术方法**

采用多源数据融合（书籍、官方文件、网络抓取）并使用ChatGPT辅助结构化抽取与人工审核，评测采用BERTScore、BARTScore、Exact Match等指标；整体框架基于LLM Evaluation Harness，并使用统一JSON模板实现自动化评测。

**📊 数据集**

使用14个税务相关数据集（如TaxRecite、TaxSum、TaxCalc、TaxSCQ、TaxMCQ、TaxQA、TaxBoard、TaxCrime、TaxOpinion、TaxRisk、TaxInspect、TaxPlan等），总计约7.3千条样本。

**📈 对比分析**

通过统一JSON评测框架，在零样本与一样本两种设置下对19款LLM进行多维度比较，结果显示闭源大参数模型（ERNIE‑3.5、GPT‑4o、Grok3、ChatGPT）整体表现最佳，中文本地模型优于多语种模型，领域微调未显著提升，税务应用层（KA）表现最差，揭示推理与数值计算能力不足。

**⚠️ 局限性**

存在数据泄露风险、模型参数同质化导致缺少更大规模对比、以及自动评测指标与人工判断不完全一致等局限。

---

## 230. TouchAnything: Diffusion-Guided 3D Reconstruction from Sparse Robot Touches

**arXiv ID:** 2604.08945 | [PDF](https://arxiv.org/pdf/2604.08945v1)

**作者:** Langzhe Gu `[一作]` (Tsinghua University), Wenzhen Yuan `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种利用预训练2D扩散模型作为几何先验的 TouchAnything 框架，能够仅用稀疏触觉采样和粗略文本描述对任意物体进行全局3D几何重建，支持开放世界无类别训练。

**💡 创新点**

创新点在于：①将大规模视觉扩散模型的几何知识迁移到触觉域，无需针对触觉数据训练专门的生成模型；②结合稀疏触觉约束与扩散引导的全局优化，突破触觉重建的欠约束难题；③实现了对未见物体的开放世界重建。

**🔧 技术方法**

技术包括：GelSight 触觉传感器获取深度/法向，虚拟摄像机投影转化为视角观测；神经隐式 SDF（Neuralangelo）与显式 DMTet 两阶段几何优化；Stable Diffusion 预训练模型与 Score Distillation Sampling (SDS) 进行语义与几何引导；差分渲染、Poisson 求解、Eikonal 正则化等。

**📊 数据集**

数据集：ShapeNetCore.V2（280个物体，6类）用于仿真；YCB、ShapeNetCore.V2 3D打印物体与日常物体共14个用于真实实验；仿真触觉图像通过 Taxim 生成，真实触觉图像使用 GelSight Mini。

**📈 对比分析**

与基线 TouchSDF（DeepSDF）和 Touch2Shape（专门训练的触觉扩散模型）比较，在仿真 20 次触碰下，TouchAnything 的 Earth Mover’s Distance (EMD) 在所有类别均低于两者，误差平均约 20%–30% 更低，显示更优的几何重建精度；真实实验亦能在 20 触碰内完成全局重建，展示出较强的泛化能力。

**⚠️ 局限性**

局限性：仍依赖文本提示，若提示错误或为空可能导致形状漂移；触觉采样需要多次接触，对实验成本有一定要求；当前扩散引导主要使用低分辨率正向渲染，限制细节恢复；未实现主动触摸策略，未充分利用信息增益。

---

## 231. M-IDoL: Information Decomposition for Modality-Specific and Diverse Representation Learning in Medical Foundation Model

**arXiv ID:** 2604.08936 | [PDF](https://arxiv.org/pdf/2604.08936v1)

**作者:** Yihang Liu `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种自监督医学基础模型M-IDoL，利用信息分解提升多模态医学影像的模态特异性与多样性；

**💡 创新点**

创新点在于通过信息分解将多模态互信息拆解为两项：最大化模态间熵（增强模态特异性）与最小化模态内不确定性（提升模态内多样性），并结合Mixture-of-Experts投影器实现无监督的模态划分；

**🔧 技术方法**

主要技术包括信息分解理论、Mixture-of-Experts（MoE）投影器、路由一致性损失（促进模态子空间分离）和模态内对比损失（降低增强噪声导致的不确定性），基于DINO框架和Swin-B视觉编码器；

**📊 数据集**

在115万无标签医学图像上预训练，涵盖五大模态：视网膜（Fundus）、胸片（X-ray）、光学相干断层扫描（OCT）、组织病理学（Pathology）和皮肤镜（Dermoscopy），随后在21个下游任务上评估；

**📈 对比分析**

与20个统一医学基础模型以及5个模态专属模型进行对比，M-IDoL在21个下游数据集上整体均优于统一模型，且在多数任务上接近或超过模态专属模型，显著提升了跨模态泛化与细粒度判别；

**⚠️ 局限性**

局限性包括对专家数量的依赖（更多专家提升有限）、训练和推理成本相对较高、模型仍需在大规模无标签数据上预训练且在极端模态分布或小样本任务上表现未知。

---

## 232. IAT: Instance-As-Token Compression for Historical User Sequence Modeling in Industrial Recommender Systems

**arXiv ID:** 2604.08933 | [PDF](https://arxiv.org/pdf/2604.08933v1)

**作者:** Xinchun Li `[一作]` (ByteDance), Yaocheng Tan `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Instance-As-Token（IAT）两阶段序列建模框架，先将每条历史交互实例压缩为低维实例嵌入（InsEmb），后将这些嵌入作为序列令牌进行下游序列建模；

**💡 创新点**

创新点在于把手工特征压缩成统一密集令牌，突破手工序列特征的容量瓶颈，并引入用户顺序压缩与 Source Instance Transformer（SIT）提升实例嵌入的序列建模能力；

**🔧 技术方法**

使用压缩/解压缩层、SIT（Transformer）、适配 MLP、可选侧信息、以及标准序列模型（如 Transformer、DIN、LONGER）实现；

**📊 数据集**

在真实广告场景中收集的四个月 CVR 预测数据集（数十亿条训练实例），并在多场景（广告、商城、直播等）进行离线与在线验证；

**📈 对比分析**

与传统手工序列特征基线（DIN、LONGER、Transformer）对比，用户顺序 IAT 在离线 AUC 上提升 0.2–0.3%，在线 A/B 试验中 ADSS/ADVV 提升 0.6–1.6%；

**⚠️ 局限性**

局限性包括：压缩过程可能略微丢失信息（temporal-order IAT 源模型略降 0.05% AUC），用户顺序源模型训练复杂且需要额外存储；部署时需要维护实例 ID 与侧信息，且目前未实现一阶段训练与更高效压缩技术。

---

## 233. Enhancing LLM Problem Solving via Tutor-Student Multi-Agent Interaction

**arXiv ID:** 2604.08931 | [PDF](https://arxiv.org/pdf/2604.08931v1)

**作者:** Nurullah Eymen Özdemir `[一作]` (Ozyegin University), Erhan Oztop `[通讯]` (Ozyegin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于同一大型语言模型的教师-学生双角色多代理系统 PETITE，以结构化的角色分工和迭代反馈提升代码生成性能。

**💡 创新点**

将人类教育学的伙伴辅导与结构化反馈原则应用于 LLM 内部交互，采用异步教师-学生序列化角色和决策驱动的早停机制，显著提升效率与准确率。

**🔧 技术方法**

使用 Qwen2.5-Coder-7B-Instruct 模型，结合系统提示区分学生与教师角色，实施迭代生成-评估循环、早停决策及 token 计数监控。

**📊 数据集**

在 APPS 编程题库的 100 题子集（Intro、Interview、Competition 三难度级别）上进行评估。

**📈 对比分析**

与 Self-Consistency、Self-Refine、Multi-Agent Debate、MARS 等基线对比，PETITE 在相同模型与参数下获得最高 31.6% 成功率，仅消耗约 2.5k‑5.3k token，效率比其他方法高达 12.5 倍。

**⚠️ 局限性**

依赖教师代理判断准确性；若误判会导致过早终止；仅在单一模型和 APPS 子集上测试，缺乏模型多样性与更广泛任务验证。

---

## 234. Finding Nemo-Nemo: CFT DAG-based Consensus in the WAN

**arXiv ID:** 2604.08914 | [PDF](https://arxiv.org/pdf/2604.08914v1)

**作者:** Rithwik Kerur `[一作]` (University of California Santa Barbara), Igor Zablotchi `[通讯]` (Mysten Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Nemo-Nemo，一种在 WAN 环境下实现高吞吐、低延迟的 DAG‑based CFT 共识协议。

**💡 创新点**

创新点在于把 BFT 的 DAG 架构迁移到 CFT，使用非认证 DAG、分离传播与共识、支持多领导者并且不丢弃超时请求，从而突破单领导者的吞吐瓶颈。

**🔧 技术方法**

采用了 DAG mempool、无签名无认证、随机异步网络模型、多领导轮转、写前日志、Rust 实现以及批量处理技术。

**📊 数据集**

通过自建基准框架在随机异步网络模型下模拟 WAN 网络，对比多种现有 CFT 协议，使用标准的网络延迟与失真参数。

**📈 对比分析**

与 Multi‑Paxos、QuePax、EPaxos 等协议对比，Nemo‑Nemo 在随机异步和部分同步模型下吞吐量提升至少 2 倍，延迟与现有 CFT 系统相当。

**⚠️ 局限性**

局限在于仅针对 crash fault 而非 Byzantine，依赖无签名假设，极端网络分区或高失效率时的表现仍需进一步评估。

---

## 235. SEA-Eval: A Benchmark for Evaluating Self-Evolving Agents Beyond Episodic Assessment

**arXiv ID:** 2604.08988 | [PDF](https://arxiv.org/pdf/2604.08988v1)

**作者:** Sihang Jiang `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了自演化代理（SEA）的形式化定义并构建了SEA‑Eval基准，用以评估跨任务演化与执行可靠性。

**💡 创新点**

创新点在于：①为SEA提供了基于数字化体现与持续跨任务演化的理论框架；②设计了Evolutionary Flywheel闭环结构；③将Token消耗引入为核心动态演化指标，并创建了多任务序列数据集。

**🔧 技术方法**

采用大语言模型（Claude Opus 4.6）与Agent框架（OpenClaw、GenericAgent），实现执行与认知双核架构、自动工具合成、经验蒸馏与递归记忆更新等技术。

**📊 数据集**

使用自构造的SEA‑Eval数据集，包含30个原子任务（Web 16 / 本地系统 14）、90个任务变体形成关联序列、45个环境变更及30个用户偏好任务，共计约120+任务。

**📈 对比分析**

通过对比成功率与Token消耗两大指标，实验发现OpenClaw与GenericAgent在成功率相同，但Token消耗相差约31.2×；只有GenericAgent表现出随任务序列递减的Token消耗，证明其真正实现了演化。

**⚠️ 局限性**

局限性包括：评估仅针对单一agent的私有记忆；缺少多agent协同演化、深度主观偏好对齐、安全治理与跨模态（物理）适配的研究。

---

## 236. Neighbourhood Transformer: Switchable Attention for Monophily-Aware Graph Learning

**arXiv ID:** 2604.08980 | [PDF](https://arxiv.org/pdf/2604.08980v1)

**作者:** Yi Luo `[一作]`, Aiguo Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出邻域Transformer（NT），在每个节点邻域内部使用自注意力生成节点表示，解决传统GNN在异质图上的同质性假设问题，并通过邻域分区与可切换注意力实现大图可扩展。

**💡 创新点**

创新点包括①基于“monophily”属性的邻域内自注意力机制；②邻域分区策略结合Transformer/Performer可切换注意力，显著降低空间和时间复杂度；③证明NT兼容传统消息传递，表达力不弱；④针对有向图设计的Dir-NT。

**🔧 技术方法**

技术手段：自注意力、Transformer、Performer线性注意力、邻域分区与可切换注意力、各种聚合器（mean、weighted-mean、sum、gated-sum、max）以及自动超参数搜索。

**📊 数据集**

数据集：10个真实图（异质图5个：Roman Empire、A-ratings、Minesweeper、Tolokers、Questions；同质图5个：A-computer、A-photo、CoauthorCS、CoauthorPhy、WikiCS）。

**📈 对比分析**

对比方法：10个state‑of‑the‑art MPNN基线（GCN、GraphSAGE、GAT、H2GCN、CPGNN、GPRGNN、FAGCN、GloGNN、GBK-GNN、JacobiConv、GGCN、tGNN、OrderedGNN、GAT-sep、Dir-GNN、CDE、BloomGML、APPNP、PPRGo、GCNII）。在节点分类任务上进行10次实验，NT在8/10图上超越所有基线，在剩余2图排名第二，平均准确率最高，尤其在异质图上表现突出。

**⚠️ 局限性**

局限性：需要针对高密度或度分布极端的图进行邻域分区，且分区策略在非常小或低度稠密图上可能反而降低效率；自循环、归一化等处理需谨慎；在某些有向图场景下，分割邻域可能不总是有益；模型依赖超参数搜索，部署时需额外调优。

---

## 237. Testing the Assumptions of Active Learning for Translation Tasks with Few Samples

**arXiv ID:** 2604.08977 | [PDF](https://arxiv.org/pdf/2604.08977v1)

**作者:** Lorenzo Jaime Yu Flores `[一作]`, Jackie Chi Kit Cheung `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在极低资源下评估主动学习对机器翻译性能的影响，发现传统信息性和多样性假设不成立。

**💡 创新点**

提出核心假设失效的原因，强调样本顺序和预训练知识对性能的决定性影响。

**🔧 技术方法**

使用Flan‑T5、Llama 3.1‑8B、Gemma‑2‑2B等大型预训练模型，结合多种主动学习采集函数（如BALD、DelFy、Core Set等）和统计相关分析。

**📊 数据集**

利用NLLB 10K句对的英文-非洲语、英文-德语、英文-菲律宾语数据作为训练集，FLORES Plus作为测试集。

**📈 对比分析**

通过与随机采样对比并计算ChrF+分数，发现主动学习在100–500样本时往往不优于随机采样，且信息性/多样性指标仅解释5–15%的性能方差。

**⚠️ 局限性**

结果受限于模型与数据选择、超参数设置、仅针对翻译任务的实验，且对不同生成任务的普适性仍需验证。

---

## 238. Quantisation Reshapes the Metacognitive Geometry of Language Models

**arXiv ID:** 2604.08976 | [PDF](https://arxiv.org/pdf/2604.08976v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]` (Independent Researcher), Jon-Paul Cacioli (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了模型量化对大型语言模型在不同知识域上元认知效率的影响，并尝试通过领域条件化的SFT提升弱域的自我监控能力。

**💡 创新点**

创新点在于揭示量化会导致M-ratio（meta-d'/d'）的域级几何结构重组，但对AUROC_2（原始元认知判别信号）保持不变，从而说明M-ratio对量化格式高度敏感。

**🔧 技术方法**

使用的技术包括Type‑2信号检测理论（meta-d'、M‑ratio、AUROC_2）、LoRA微调、Q5_K_M与f16两种量化格式，以及非参数自助法检验假设。

**📊 数据集**

数据集为TriviaQA，按Llama‑3‑8B‑Instruct在T=0.1下划分为Arts、Geography、History、Science四个知识域，总计约3,000道题目用于同问量化比较。

**📈 对比分析**

比较方法是将同一模型在相同问题集下分别使用Q5_K_M和f16两种量化，计算M‑ratio和AUROC_2；结果显示M‑ratio相关系数为0，而AUROC_2为1；SFT训练虽显著提升Science域的NLP gap，却未改善meta‑d'，验证了训练失效。

**⚠️ 局限性**

局限性包括仅测试单一模型和两种量化格式，M‑ratio在低d'条件下不稳定，未探究量化为何产生域间d'压缩的机制，且未验证结果在其他模型或量化方法（如GPTQ、AWQ、INT8）上的普适性。

---

## 239. Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems

**arXiv ID:** 2604.08963 | [PDF](https://arxiv.org/pdf/2604.08963v1)

**作者:** Keyu Li `[一作]` (Shanghai Jiao Tong University), Dequan Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估多代理系统在不同拓扑、角色与深度设置下的偏见累积，构建并实验验证了基于 Discrim‑Eval 的三选项开放式基准。

**💡 创新点**

创新点在于：①提出 Discrim‑Eval‑Open 基准，突破传统二元测试；②使用分布式度量（Gini、方差、熵）捕捉偏见传播；③揭示即使系统复杂度提升，偏见仍会累积，并发现“触发漏洞”导致极端偏见放大。

**🔧 技术方法**

技术方法包括：基于有向无环图的 MAS 架构设计；多种 LLM（DeepSeek‑V3、GPT‑4o、GLM‑4v、Qwen‑Max 等）实现多步骤推理；统计度量（Gini 系数、相对 Gini、方差、熵）评估偏见程度。

**📊 数据集**

使用改造后的 Discrim‑Eval Benchmark，共 70 个情景、210 个角色配置，覆盖年龄、性别、种族等敏感属性；通过三选项形式强制模型做比较判断。

**📈 对比分析**

实验对比不同拓扑（Spindle、Parallel、Fully‑Connected）、深度、角色组合与模型异质性，计算相对 Gini 的提升；结果表明所有配置下偏见均在递增，系统越深偏见越显著，性能表现为持续的偏见放大而非缓解。

**⚠️ 局限性**

局限性在于：仅定位并量化偏见传播机制，未给出有效的缓解方案；实验仅在公开 LLM 上进行，缺乏对更大规模或真实应用环境的验证；未探索对其他系统失效模式（如幻觉、群体思维）的影响。

---

## 240. NyayaMind- A Framework for Transparent Legal Reasoning and Judgment Prediction in the Indian Legal System

**arXiv ID:** 2604.09069 | [PDF](https://arxiv.org/pdf/2604.09069v1)

**作者:** Parjanya Aditya Shukla `[一作]` (IIT Kanpur), Arnab Bhattacharya `[通讯]` (IIT Kanpur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套用于印度法律的法院判决预测与解释（CJPE）框架，结合检索、结构化推理与预测，生成透明、基于证据的法律推理与判决说明。

**💡 创新点**

创新点在于将检索增强生成（RAG）与结构化推理LLM结合，明确模拟法官的论证流程（问题识别、双方论点、法条依据、结论），并构建了规模达1600万条的印度法律数据库。

**🔧 技术方法**

采用RAG检索管线（Milvus、Endee、Vespa），LoRA/QLoRA参数高效微调，配合DeepSeek、Phi-4、Qwen等大型推理LLM，使用结构化提示和多步“think tokens”实现分步推理。

**📊 数据集**

使用印度法律大规模语料库（1600万条法律文件）以及7,120条带有结构化注释的判决数据集进行微调与评估。

**📈 对比分析**

通过ROUGE、BLEU、BERTScore、BLANC等自动指标以及两名法律专业人员的专家评分进行对比。实验表明，系统在所有指标上均优于基线，尤其是大型模型在推理与解释质量上显著提升；但低精度量化（4‑bit）会导致输出重复、逻辑不连贯。

**⚠️ 局限性**

局限性包括：仅支持英文；受LLM上下文长度限制（16,384 token输入/4,096 token输出），无法完整处理超长法律文档；缺乏强制的推理验证机制；尚未实现多语言与更长上下文的支持。

---

## 241. PDE-regularized Dynamics-informed Diffusion with Uncertainty-aware Filtering for Long-Horizon Dynamics

**arXiv ID:** 2604.09058 | [PDF](https://arxiv.org/pdf/2604.09058v1)

**作者:** Min Young Baeg `[一作]`, Yoon-Yeong Kim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种结合PDE正则化插值器和Unscented Kalman Filter（UKF）预测器的PDYffusion框架，用于长时序的空间-时间预测。

**💡 创新点**

创新点在于：①将偏微分方程（PDE）约束直接嵌入插值损失，强制生成的中间状态满足物理一致性；②通过UKF对预测结果进行不确定性建模与校正，抑制预测循环中的误差累积；③在扩散模型基础上实现全概率长时预测。

**🔧 技术方法**

技术包括扩散模型、基于PDE的随机场采样、Unscented Kalman Filter、Mean Squared Error（MSE）、Continuous Ranked Probability Score（CRPS）、Spread‑Skill Ratio（SSR）等指标的训练与评估。

**📊 数据集**

使用四个基准数据集：海表温度（SST）、Navier‑Stokes流动、弹性春网（Spring‑mesh）和波动（Wave）。

**📈 对比分析**

与DYffusion、DDPM、MCVD、扰动与dropout等方法比较，使用CRPS、MSE、SSR等指标评估；在Navier‑Stokes、Spring‑mesh和Wave数据集上，PDYffusion在CRPS/MSE上均取得最低或接近最佳成绩，SSR也保持在接近1的稳定区间。

**⚠️ 局限性**

局限性主要体现在：对噪声较大、真实物理场（如SST）性能提升有限；模型对边界条件与正则化强度λ的敏感性需要精细调参；当前仅针对确定性PDE系统，尚未扩展到随机过程（SPDE）场景。

---

## 242. Conversations Risk Detection LLMs in Financial Agents via Multi-Stage Generative Rollout

**arXiv ID:** 2604.09056 | [PDF](https://arxiv.org/pdf/2604.09056v1)

**作者:** Xiaotong Jiang `[一作]` (Waseda University), Jun Wu `[通讯]` (Waseda University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对金融服务场景下的大语言模型对话安全，本文设计并实现了 FinSec 这一四层次安全检测框架，能够对多轮金融对话进行结构化、延迟风险、语义安全和融合决策等全流程评估。

**💡 创新点**

创新点在于：①基于 AML/SAR 标准构建三匹配（关键词/语义/序列）行为模式库，实现低成本高召回的预警；②引入对抗性生成回滚模拟，对多轮对话的潜在风险进行定量评估；③层级融合与权重自适应，使语义安全层占比最大化，突破传统模型的“权衡”瓶颈。

**🔧 技术方法**

主要技术包括：大语言模型（LLM）prompt 设计、少量样本提示法、基于三匹配的规则匹配、对抗性对话生成与风险评分、层级融合加权与阈值决策。

**📊 数据集**

使用 R-Judge 金融对话安全评测数据集（涵盖注入攻击与非注入风险场景），并在公开 LLM（O3、Gemini、Claude 等）上进行对比实验。

**📈 对比分析**

与基线 R-Judge、FinO3 等模型相比，FinSec 在总 F1 上提升至 90.13%，注入攻击 F1 94.55%，非注入攻击 F1 85.71%；AUPRC 与 ASR 也实现了 0.9189 与 0.0875 的显著提升，综合安全评分达 0.9098，明显优于现有 LLM 方案。

**⚠️ 局限性**

主要局限：当提示语过长或复杂时，LLM 的输出稳定性下降；对注入攻击的细粒度分类仍待改进，且对长序列输入的适配需要进一步优化。

---

## 243. SiMing-Bench: Evaluating Procedural Correctness from Continuous Interactions in Clinical Skill Videos

**arXiv ID:** 2604.09037 | [PDF](https://arxiv.org/pdf/2604.09037v1)

**作者:** Xiyang Huang `[一作]` (Wuhan University), Sophia Ananiadou `[通讯]` (University of Manchester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SiMing‑Bench 基准，评估多模态大语言模型在完整临床技能视频中依据专家评分表进行过程级判断的能力。

**💡 创新点**

创新点：①首个关注交互驱动状态更新与步骤合法性的全流程临床视频基准；②构建 200+ 份双专家标注的 SiMing‑Score 数据集；③揭示模型在整体评分与步骤评分之间的巨大性能差距。

**🔧 技术方法**

技术：使用多模态 LLM（如 GPT‑4o、LLaVA‑Video、Qwen3‑VL 等）结合文本化评分表进行结构化预测，并通过 Pearson、Spearman、Cohen κ 等统计指标评估。

**📊 数据集**

数据集：SiMing‑Score，包含 200 段完整的临床技能视频（CPR、AED、BMV），时长 2–4 分钟，每段均有双专家按标准评分表给出的步骤级与总分。

**📈 对比分析**

比较方法：在整体评分上计算 PLCC/SRCC，在步骤级评价上使用 quadratic weighted Cohen κ；结果显示最佳模型整体评分 PLCC 仅 0.158，步骤级 κ 接近 0；即使简化为二分类或裁剪步骤片段，模型性能仍低。

**⚠️ 局限性**

limitation：仅涵盖标准化教育评估环境，缺乏真实临床多样性；模型未验证在临床决策中的安全性；数据集规模有限；未深入探究模型记忆、因果推理及状态追踪机制。

---

## 244. Advantage-Guided Diffusion for Model-Based Reinforcement Learning

**arXiv ID:** 2604.09035 | [PDF](https://arxiv.org/pdf/2604.09035v1)

**作者:** Daniele Foffano `[一作]` (KTH Royal Institute of Technology), Alexandre Proutiere `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了优势引导扩散（AGD-MBRL），通过使用优势函数来指导扩散模型的反向采样，从而生成更具长期价值的轨迹，提升模型基 RL 的样本效率与最终回报。

**💡 创新点**

创新点在于将优势函数（而非仅奖励或策略）作为导向信号，引入 Sigmoid 与 Exponential 两种优势引导方案，并证明其可实现策略改进；同时实现了与 PolyGRAD 兼容的无额外训练目标的实现。

**🔧 技术方法**

核心技术包括扩散模型（denoising diffusion probabilistic model）、优势函数估计、分类器引导扩散（classifier-guided diffusion）以及 Dyna 风格的在线学习框架；使用了多臂 MLP、Adam 优化器和 L2 损失。

**📊 数据集**

在 MuJoCo 连续控制任务（HalfCheetah、Hopper、Walker2D、Reacher）上进行实验，使用环境回合数 1.5M 作为数据集。

**📈 对比分析**

与 PolyGRAD、Online Diffuser、PPO 与 TRPO 等基线对比，AGD-MBRL 在大多数任务中取得更高的最终回报，尤其在 HalfCheetah 上提升约 2 倍，且表现更稳定，训练曲线显示波动更小。

**⚠️ 局限性**

主要局限是扩散模型生成轨迹的计算开销较大，生成速度慢；此外，在噪声较大的 Hopper 任务中，优势引导效果略逊于 PolyGRAD，提示对优势估计的鲁棒性仍需提升。

---

## 245. The nextAI Solution to the NeurIPS 2023 LLM Efficiency Challenge

**arXiv ID:** 2604.09034 | [PDF](https://arxiv.org/pdf/2604.09034v1)

**作者:** Gyuwon Park `[一作]` (UNIST), Byung-Hak Kim `[通讯]` (CJ Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在单个A100 40GB GPU和24小时内完成LLaMA2 70B模型的高效微调

**💡 创新点**

将QLoRA与Flash Attention 2结合，并通过迭代的数据集构建实现了资源受限下的高性能微调

**🔧 技术方法**

使用QLoRA/LoRA技术、Flash Attention 2及自定义多样化数据集进行模型微调

**📊 数据集**

构建了来自开放源代码资源和基准测试的自定义数据集，用于训练和验证

**📈 对比分析**

通过在多项QA基准测试上的评估显示，该模型在保持高准确率的同时显著降低了显存和计算消耗

**⚠️ 局限性**

受限于单GPU环境，模型规模仍受限；仅在有限的基准任务上验证，泛化能力与更大规模硬件下的性能仍待进一步探究

---

## 246. Hypergraph Neural Networks Accelerate MUS Enumeration

**arXiv ID:** 2604.09001 | [PDF](https://arxiv.org/pdf/2604.09001v1)

**作者:** Hiroya Ijima `[一作]` (Hitachi Ltd), Koichiro Yawata `[通讯]` (Hitachi Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种域无关的 MUS/MSS 列举加速方法 HyMUSE，利用超图神经网络在枚举过程中学习约束与已发现 MUS/MCS 之间的关系；

**💡 创新点**

创新点在于使用超图而非传统变量‑约束图，构造以约束为顶点、MUS/MCS 为超边的超图；并通过强化学习训练 HGNN 代理以最小化求解器调用次数；

**🔧 技术方法**

核心技术包括 AllSetTransformer‑style 超图神经网络、PPO 强化学习、纠正算法（回溯恢复/标准收缩/扩张）以及谱嵌入；

**📊 数据集**

训练数据为 SR(U(5,20)) 随机 CNF 实例，评估数据集包含 SAT_small、SAT_large、图着色生成的 GC 以及 SMT‑LIB 中的 SMT 实例；

**📈 对比分析**

与 MARCO、TOME、ReMUS 等传统枚举算法对比；在固定 10k 次可满足性检查下，MARCO+HyMUSE 在 SAT_small、SAT_large、GC、SMT 上均获得约 1.3‑1.9 倍的 MUS/MSS 计数提升；与 TOME、ReMUS 的提升有限；

**⚠️ 局限性**

局限性在于：代理在 MARCO 训练后对其他算法迁移性差；只使用 MUS/MCS 作为超边，未探究其他组合；对更大规模或非 CNF 结构的泛化仍需验证。

---

## 247. Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory

**arXiv ID:** 2604.08995 | [PDF](https://arxiv.org/pdf/2604.08995v1)

**作者:** Zile Wang `[一作]` (Skywork AI), Yahui Zhou `[通讯]` (Skywork AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Matrix-Game 3.0，一种记忆增强的交互式世界模型，能够在 720p 分辨率下以 40 FPS 实时生成长达一分钟的视频，并保持长时空一致性。

**💡 创新点**

创新点在于：1) 工业级无限数据引擎整合 UE5 合成、AAA 游戏自动采集与真实视频；2) 误差感知交互式基础模型与相机感知长时记忆检索结合统一自注意力；3) 多段分布匹配蒸馏与系统级加速（INT8 量化、轻量 VAE 裁剪、GPU 检索）实现 5B 模型下的实时长序列生成。

**🔧 技术方法**

技术栈包括 Diffusion Transformer (DiT)、VAE、INT8 量化、轻量 VAE 裁剪、GPU 加速检索、相机几何编码、旋转位置编码扰动、误差收集/注入、自回归分布匹配蒸馏、MoE 扩展等。

**📊 数据集**

数据集涵盖：Unreal Engine 5 合成视频、AAA 游戏自动采集数据、DL3DV-10K、RealEstate10K、OmniWorld-CityWalk、SpatialVid-HD 等多源视频-姿态-动作-提示四元组。

**📈 对比分析**

通过与 Matrix-Game 2.0、HY-Gamecraft-2、Lingbot-World 等对比，Matrix-Game 3.0 在 5B 模型下实现 720p 40 FPS、分钟级长时空一致性；28B 模型进一步提升视觉质量、动态表现与泛化；实验中各加速技术显著提升推理速度，VAE 裁剪与 INT8 量化提高效率。

**⚠️ 局限性**

局限性包括：对极端视角变化或复杂交互的记忆检索误差；高分辨率长序列仍需大量算力；训练数据多样性与真实性不足可能导致某些场景细节偏差。

---

## 248. Nested Radially Monotone Polar Occupancy Estimation: Clinically-Grounded Optic Disc and Cup Segmentation for Glaucoma Screening

**arXiv ID:** 2604.09062 | [PDF](https://arxiv.org/pdf/2604.09062v1)

**作者:** Rimsa Goperma `[一作]` (Kyoto University), Liang Zhao `[通讯]` (Kyoto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种新的视网膜光盘与杯部分割框架 NPS-Net，利用嵌套极坐标占据表示保证星形凸性和杯内盘的嵌套关系；

**💡 创新点**

创新点在于将结构约束（星形凸性、嵌套性、角向rim曲线）直接编码到输出参数化中，并引入熵门控的形状先验融合和无参数 Polar Test‑Time Augmentation 以实现跨域鲁棒性；

**🔧 技术方法**

使用极坐标 UNet 编码器-解码器、累积递减的占据函数、乘法门控实现嵌套、熵门控形状先验、Polar-TTA、Dice/BCE + rim 形状损失；

**📊 数据集**

在七个公开数据集上评估，包括 RIM‑ONE、PAPILA、DRISHTI‑GS、REFUGE、RIGA、ORIGA、NETRA，覆盖多民族、多相机、零样本跨域情况；

**📈 对比分析**

相较于七种基线（UNet、Attention‑UNet、ResUNet、Polar‑UNet、TransUNet、BEAL、DoFE），NPS-Net 在零样本 RIM‑ONE 取得 Cup Dice +12.8%、vCDR MAE ↓56%；在 PAPILA 得到 Disc Dice 0.9438、HD95 2.78 px；在 DRISHTI‑GS+REFUGE 获得最优 vCDR MAE 0.0636 与 Rim Corr 0.6856，整体性能领先；

**⚠️ 局限性**

局限包括需要预先定位盘心（Polar‑TTA 仅搜索 ±16 px 误差），若定位误差大需外部检测；Polar‑TTA 计算开销高（27 次前向），且对未见模态（如超宽视野、手机眼底相机）尚未验证。

---

## 249. Taming the Black Swan: A Momentum-Gated Hierarchical Optimisation Framework for Asymmetric Alpha Generation

**arXiv ID:** 2604.09060 | [PDF](https://arxiv.org/pdf/2604.09060v1)

**作者:** Arya Chakraborty `[一作]` (Birla Institute of Technology Mesra), Randhir Singh `[通讯]` (Birla Institute of Technology Mesra)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一套名为AEGIS的三层阶梯式动量门控投资框架，结合波动率调整动量筛选、最小极大相关性去相关以及针对Sortino的SLSQP凸优化，实现美国股市2006-2025年的系统性动量投资；

**💡 创新点**

引入基于波动率调整的动量指标VAM、最小极大相关性过滤器以实现结构独立性，并通过SLSQP针对Sortino的非线性凸优化，实现尾部风险免疫的自适应资金配置，形成抗黑天鹅的动态组合；

**🔧 技术方法**

使用Python、Pandas、NumPy、SciPy（SLSQP）、Yahoo Finance API、并行异步请求、滚动回测、最小极大相关性算法、对数收益、收益分解等技术；

**📊 数据集**

采用2006-2025年美国五大指数成分（S&P 500/400/600、NASDAQ‑100、Dow Jones）历史价格、交易量、财报等公开数据，并手工注入退市/破产股以消除生存偏差；

**📈 对比分析**

通过20年滚动回测、10 bp交易成本、严格训练/测试窗口分离进行比较，结果CAGR 15.41%、净回报1657% vs S&P 500 8.88%，最大回撤28.9%、Sortino 6.47（剔除极值后1.72），月度赢率68.8%，相较传统动量与风险平价显著优越；

**⚠️ 局限性**

对参数（篮子规模、窗口长度）敏感；未充分模拟极端流动性冲击和滑点；模型依赖历史相关性估计，在新兴极端事件下可能失效；

---

## 250. Fine-Grained Action Segmentation for Renorrhaphy in Robot-Assisted Partial Nephrectomy

**arXiv ID:** 2604.09051 | [PDF](https://arxiv.org/pdf/2604.09051v1)

**作者:** Jiaheng Dai `[一作]` (Fudan University), Qingbiao Li `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并基准化了机器人辅助部分肾切除术中肾缝合阶段的细粒度动作分割任务，使用SIA‑RAPN数据集完成帧级12标签标注。

**💡 创新点**

创新点在于构建临床导向的12标签分割任务、设计跨域单孔RAPN评估、并对比四种主流时序模型，给出最强模型结果。

**🔧 技术方法**

采用I3D作为通用视频特征提取器，配合MS‑TCN++、AsFormer、TUT和DiffAct四种时序解码器，并引入类别重加权、边界感知、迭代去噪等技术。

**📊 数据集**

使用由中国人民解放军总医院采集的50条da Vinci Xi系统录制的临床手术视频（1080i 60fps）标注12帧级标签，同时对单孔RAPN数据进行跨域评估。

**📈 对比分析**

在五个随机拆分上分别训练并评估模型，取每个指标的最高值进行汇总；DiffAct在F1@10/25/50、帧精度、编辑分数和帧mAP上均领先，MS‑TCN++在平衡精度上最高；在单孔跨域测试中DiffAct仍保持最强性能。

**⚠️ 局限性**

局限在于仅来自单一机构的50例数据，缺乏多中心、多外科医生和多设备的泛化验证，且汇总表使用每个指标的最佳值，未能反映单一fold的真实表现。

---

## 251. Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures

**arXiv ID:** 2604.09048 | [PDF](https://arxiv.org/pdf/2604.09048v1)

**作者:** Mauricio Fadel Argerich `[一作]` (Universidad Politécnica de Madrid), Marta Patiño-Martínez `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了Watt Counts开源数据集与可复现的基准，用于测量50种LLM在10种NVIDIA GPU上在批量和服务器场景中的能耗，涵盖5000+实验、14M功耗采样。

**💡 创新点**

创新点是首次公开跨GPU异构的LLM能耗数据与基准，揭示GPU架构、模型大小、部署场景对能效的交互影响，并给出可节能部署建议。

**🔧 技术方法**

采用vLLM推理引擎、EnergyMeter/pyNVML采集GPU功耗、CPU/内存指标，结合Poisson负载生成器与Batch模式执行实验。

**📊 数据集**

数据集包含50个公开LLM（≤30B参数）与10个NVIDIA GPU（V100、T4、A100、RTX3090、A30、RTX4090、L40S、L4、H100 NVL、H200 NVL），记录批处理、服务器低负载与高负载下的能耗、吞吐量等指标。

**📈 对比分析**

对比方法基于能耗/能耗/能耗每token和平均功率评估不同GPU、模型、场景的能效，结果显示H100在批处理大部分模型最省能，低功耗GPU在小模型或服务器场景更优，可根据TTFT阈值实现70%能耗下降。

**⚠️ 局限性**

局限包括仅单GPU实验、仅16位权重、仅vLLM引擎、未考虑准确度与冷却等宿主功耗，且TDP与实际功耗差异大，未来需扩展多GPU、量化模型和多目标优化。

---

## 252. Scene-Agnostic Object-Centric Representation Learning for 3D Gaussian Splatting

**arXiv ID:** 2604.09045 | [PDF](https://arxiv.org/pdf/2604.09045v1)

**作者:** Tsuheng Hsu `[一作]` (Aalto University), Janne Heikkilä `[通讯]` (University of Oulu)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在3D Gaussian Splatting（3DGS）中引入全局对象中心学习（GOCL）产生的场景无关对象代码表，用该代码表直接监督Gaussians的身份特征，省去 VFM 的 mask 预/后处理和多视角一致性损失。

**💡 创新点**

创新点：① 将无监督对象中心学习与显式 3D 表示相结合，实现跨场景一致的对象身份标识；② 无需额外 mask 处理或专门的训练/损失设计；③ 通过全局代码表实现对象级 3D 语义与分割，并展示跨场景识别能力。

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting、Slot Attention（GOLD）、DINO 特征、Gumbel-Softmax、α‑混合渲染、特征 MSE、3D 正则化、图像渲染损失；对 VFM（SAM/SAM2）生成的 mask 直接进行监督。

**📊 数据集**

使用的公开数据集：OCTA（OCTScene‑A）与 GSO（Google Scanned Objects），分别包含真实桌面场景和合成场景。

**📈 对比分析**

与 Gaussian Grouping、Unified Lift、ObjectGS 等 VFM 监督方法以及 NeRF‑基的 uOCF 进行对比；在 OCL 指标（FG‑ARI、ARI‑A、FG‑AMI、AMI‑A、mIoU）上取得最高或第二高成绩；重建质量与 uOCF 相当甚至更好，渲染速度显著提升；实现跨场景识别优于其他方法。

**⚠️ 局限性**

局限性：① 依赖少量支持 OCL 与 3DGS 的数据集，难以推广到更复杂的真实场景；② GOLD 训练成本高、槽位数与代码表规模选择困难；③ 需要较多计算资源和时间来训练全局代码表。

---

## 253. U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster

**arXiv ID:** 2604.09041 | [PDF](https://arxiv.org/pdf/2604.09041v1)

**作者:** Salva Rühling Cachay `[一作]` (UC San Diego), Rose Yu `[通讯]` (UC San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了U-Cast，一种基于标准U‑Net、Monte Carlo Dropout以及两阶段训练课程的高效概率天气预报器；

**💡 创新点**

创新点包括：①用极简的U‑Net加上Dropout即可匹配或超过GenCast和IFS ENS的性能；②提出MAE预训练后CRPS微调的训练课程，显著缩短训练时间；③使用MuOn优化器实现快速收敛；④用Dropout替代复杂噪声注入，降低参数量和实现复杂度；

**🔧 技术方法**

使用的技术包括：标准U‑Net骨干网络（带瓶颈自注意力）、Monte Carlo Dropout实现不确定性采样、CRPS损失函数、两阶段训练课程（MAE预训练→CRPS微调）、MuOn优化器以及深度集成技术；

**📊 数据集**

实验使用ERA5再分析数据集（1.5°分辨率），并在WeatherBench 2 1.5°评测框架下使用六大大气层压强变量和表面变量，训练时间为1979–2019年；

**📈 对比分析**

通过在WeatherBench 2 1.5°上与GenCast、IFS ENS及其他基线进行对比，U‑Cast在大多数变量/先导时间下实现CRPS相当甚至优于GenCast（尤其是短期），并在92.9%变量‑先导组合上优于IFS ENS；训练成本降低10×，推理延迟降低10×；深度集成进一步提升精度；

**⚠️ 局限性**

局限性包括：在极地区域存在功率谱和视觉伪影，说明未充分利用球面几何；略微欠分散，特别是在短至中期先导时间；长达20+天的滚动预报不稳定；目前仅支持1.5°分辨率，需更大数据与自回归训练或初始条件扰动以进一步改进；

---

## 254. V-CAGE: Vision-Closed-Loop Agentic Generation Engine for Robotic Manipulation

**arXiv ID:** 2604.09036 | [PDF](https://arxiv.org/pdf/2604.09036v1)

**作者:** Yaru Liu `[一作]` (University of Cambridge), Nanyang Ye `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 V-CAGE，一套闭环的机器人数据合成框架，可自动生成符合物理约束且语义一致的长周期操控数据。

**💡 创新点**

创新点包括：①基于 Inpainting 的场景构建 (IGSC) 实现上下文感知、无几何冲突的对象布局；②将功能元数据与 VLM 结合，形成闭环视觉验证器；③面向动作的感知压缩算法实现 90%+ 文件体积压缩。

**🔧 技术方法**

使用的技术有：OpenClaw 代理框架、LLM 资产选择、SAPIEN 物理模拟、Nano Banana Inpainting、Grounding DINO + DINOv2 特征匹配、VLM Gemini 3 验证、HEVC + VDP 视觉压缩。

**📊 数据集**

数据集：使用自建资产库 (RoboTwin 风格)、生成的 100 条专家轨迹（每任务 100 条）以及 250 条仿真轨迹用于 Sim‑to‑Real。

**📈 对比分析**

与零射击、未验证的 open‑loop 方法对比，V-CAGE 在四项长周期任务中的学习成功率从 0% 提升到 54%‑100%，Sim‑to‑Real 任务从 20% 提升至 55%，压缩后性能差异小于 3%。

**⚠️ 局限性**

局限：当任务复杂度高、成功率低时，闭环验证导致拒采率高，生成效率下降；目前仅支持刚体，未覆盖变形物体或流体等动态交互。

---

## 255. Plasticity-Enhanced Multi-Agent Mixture of Experts for Dynamic Objective Adaptation in UAVs-Assisted Emergency Communication Networks

**arXiv ID:** 2604.09028 | [PDF](https://arxiv.org/pdf/2604.09028v1)

**作者:** Wen Qiu `[一作]` (Kitami Institute of Technology), Hiroshi Masui `[通讯]` (Kitami Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种基于稀疏专家混合的多智能体策略（PE‑MAMoE），用于无人机应急通信网络在阶段性非平稳目标下快速适应和保持表现；

**💡 创新点**

通过在专家层注入阶段感知的短时随机扰动、调度温度与熵、学习率等非梯度控制，以及保持路由器冻结来实现塑性增强；

**🔧 技术方法**

采用MAPPO（多智能体PPO）框架、稀疏专家混合网络（top‑k门控MoE）、专家噪声注入、非参数阶段控制器、KL约束等；

**📊 数据集**

在基于3GPP风格 A2G 信道的高保真灾难场景仿真器中进行实验，模拟多 UAV、移动用户、不同需求级别和阶段切换；

**📈 对比分析**

与 MLP、Dense MoE、Sparse MoE 等基线在同一环境下比较，使用归一化四分位数均值（IQM）以及回报、能耗、碰撞率、服务用户数等指标；PE‑MAMoE 在 IQM 上提升 26.3%，服务用户提升 12.8%，碰撞率降低约 75%；

**⚠️ 局限性**

局限包括：仿真器未考虑实际飞行、GNSS 等硬件不确定性；物理层干扰模型简化；缺乏通信受限的分布式执行方案；噪声注入与调度策略仍为手工设定；对随机/未知阶段切换的鲁棒性尚待验证。

---

## 256. Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Images via Visual Prompt Injection

**arXiv ID:** 2604.09024 | [PDF](https://arxiv.org/pdf/2604.09024v1)

**作者:** Zedian Shao `[一作]` (Georgia Institute of Technology), Neil Zhenqiang Gong `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种用户侧的图像扰动保护方法，利用视觉提示注入诱导多模态大语言模型产生拒绝响应，从而防止对个人图片进行敏感信息提取。

**💡 创新点**

将视觉提示注入技术从攻击转为防御，设计基于约束优化的几乎不可见扰动生成框架，支持单目标及多目标模型的通用扰动；并系统评估常见防御手段对该方法的影响。

**🔧 技术方法**

采用梯度投影的对抗优化（BIM/PGD），用交叉熵损失以拒绝文本序列为目标，结合 L∞ 约束保持视觉质量；通过构造“阴影问题”多样化训练，使扰动在不同模型和问题上泛化。

**📊 数据集**

在 VQAv2、GQA、TextVQA 三大视觉问答数据集以及扩展的 CelebA 视觉问答集上进行实验，并用 CommonsenseQA 生成与图像无关的提问来模拟攻击。

**📈 对比分析**

与两种现有图像对抗攻击（Qi 等、Bagdasaryan 等）及其 +PGD 版本对比；在六种开源 MLLM（Phi‑4‑multimodal、Qwen2.5‑VL 等）和四个数据集上，拒绝率普遍在 0.86–0.99 之间，明显优于对手；然而 Gaussian noise、DiffPure、对抗训练等防御手段虽能降低拒绝率，却显著降低模型准确率或推理效率。

**⚠️ 局限性**

仅适用于单轮视觉问答和图像输入，长轮对话中效果衰减；仅在白盒模型下可行，无法直接扩展到封闭式 MLLM；尚未针对音频、视频等多模态输入；对阴影问题的选择和规模仍需进一步探索。

---

## 257. Domain-generalizable Face Anti-Spoofing with Patch-based Multi-tasking and Artifact Pattern Conversion

**arXiv ID:** 2604.09018 | [PDF](https://arxiv.org/pdf/2604.09018v1)

**作者:** Seungjin Jung `[一作]` (Chung-Ang University), Jongwon Choi `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Pattern Conversion GAN（PCGAN）与Patch-based Multi-tasking Network（PMN）的域通用面部伪造检测框架，利用伪造痕迹与面部内容的分离来增强训练样本多样性，并通过局部补丁学习提升对部分伪造攻击的鲁棒性。

**💡 创新点**

创新点在于（1）PCGAN能够显式 disentangle 伪造痕迹与面部特征，支持痕迹的移除与注入，从而生成多样化的合成样本；（2）PMN 结合 CLIP、中心损失和补丁多任务学习，使模型同时关注全脸与局部特征，显著提升跨域泛化与对局部攻击的检测；（3）将上述两项技术组合成统一的训练流程，且仅在训练阶段使用生成器，推理时保持低成本。

**🔧 技术方法**

使用的技术包括：生成对抗网络（PCGAN），自编码器-交换架构、Adaptive Instance Normalization；CLIP 图像-文本编码器；多任务学习（全脸 + 裁剪补丁）；中心损失、梯度对齐损失、对抗损失、重建损失、模糊重建损失；Adam 优化器，学习率 1e-6；MTCNN/数据集提供的人脸定位。

**📊 数据集**

实验使用了五大基准数据集：CASIA-FASD（C）、Idiap Replay-attack（I）、OULU-NPU（O）、MSU-MFSD（M）、Rose Youtu（R）以及扩展的 CASIA-SURF CeFA（C）、CASIA-SURF（S）、WMCA（W）进行跨域和大规模交叉域评估。

**📈 对比分析**

与多种状态‑of‑the‑art 方法（SSAN‑R、DFDN、SAFAS、FLIP‑MCL、CA‑FAS、ViT&FA&CS、AG‑FAS、CA‑MoEiT、CFPL 等）进行对比。在 DG‑FAS 交叉域协议（如 OCI→M、OMI→C 等）中，平均 ACER 低至 2.79%，远低于对手（3–5%）；AUC 亦在 97–99% 之间保持领先。对 CSW 大规模基准（CS→W、SW→C、CW→S）同样获得最优平均 ACER 10.78% 与最高平均 AUC 95.58%。

**⚠️ 局限性**

局限性包括：合成样本的多样性受训练中可见伪造痕迹的限制，无法覆盖全新攻击技术；生成器仅用于训练，推理成本不变，但对实时大规模部署不够友好；实验主要聚焦图像级别，未覆盖视频或多模态情境，未来需在更真实的安全环境中进一步验证。

---

## 258. Scale-invariant projection optimization in tomographic volumetric additive manufacturing

**arXiv ID:** 2604.08997 | [PDF](https://arxiv.org/pdf/2604.08997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 259. Generative AI Agent Empowered Power Allocation for HAP Propulsion and Communication Systems

**arXiv ID:** 2604.09015 | [PDF](https://arxiv.org/pdf/2604.09015v1)

**作者:** Xiaoyu Xing `[一作]` (Beihang University), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种交互式生成式 AI 代理，用于 HAP（高空平台）推进与通信功率分配的全流程建模与优化；

**💡 创新点**

创新点包括：①利用 AI 代理与 CFD 结合，构建考虑船体–推进器相互干扰的精准推进功率模型；②设计了基于零干扰（ZF）+统计 CSI 的 Q3E 方案，并通过 ANN 直接在无监督环境下求解带约束的功率分配；

**🔧 技术方法**

核心技术包括：生成式 LLM（RAG+链式推理）进行文献知识检索与推理，CFD 仿真（STAR‑CCM+）验证推进模型，零干扰矩阵设计，人工神经网络与投影法相结合的无监督求解框架；

**📊 数据集**

使用的数据集为：ISA‑1976 大气参数下的 CFD 仿真样本（1–25 m/s 速度网格）、HAP 船体几何与推进器参数、9 终端的 Rician 信道统计参数，以及系统仿真中采用的 10 MHz 带宽、10.5 GHz 等设定；

**📈 对比分析**

与四种基准（QoS‑满足、max‑R_k、PPO、ACOR）比较，结果显示：①推进模型误差平均下降 84.3%，V₀=25 m/s 时误差从 8367.3 W 降至 85.32 W；②Q3E 在同一功率预算下获得最高 QoS 满足率，EE 在 90–150 W 区间提升 35–45%，并在更大预算时优于 PPO 与 ACOR；

**⚠️ 局限性**

局限性在于：仅针对单一已确定的 HAP 结构与静态统计 CSI，未考虑多 HAP 协同或用户动态分布、轨迹优化以及更复杂的实时约束；

---

## 260. Towards Linguistically-informed Representations for English as a Second or Foreign Language: Review, Construction and Application

**arXiv ID:** 2604.09008 | [PDF](https://arxiv.org/pdf/2604.09008v1)

**作者:** Wenxi Li `[一作]` (Minzu University of China), Weiwei Sun `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用同步式汉语语料构建ESFL语义银行，给出手工标注的句法-语义图与对应的同步语法-语义规则；

**💡 创新点**

创新点在于提出并应用同步语法-语义规则(SHRG)构造主义框架，系统应对ESFL的形态多样性、句法导向性及词汇多样性；

**🔧 技术方法**

技术核心为同步语法-语义规则(SHRG)与ACE解析器的组合，结合手工标注与规则选择；

**📊 数据集**

数据集主要为1643条手工标注的ESFL句子（含约100条未解析句子重构）与相同规模的标准英语句子；

**📈 对比分析**

比较方法为基于SHRG生成的CFG规则频率分布做统计检验（Chi‑Square），结果显示ESFL与英语在非词汇CFG规则分布上无显著差异；

**⚠️ 局限性**

局限性在于大部分句子仍需人工重构，解析器对ESFL句子的覆盖率不足（47.5%未解析），且仅关注句法级规则，缺乏对更深层语义多样性的自动评估。

---

## 261. StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding

**arXiv ID:** 2604.09000 | [PDF](https://arxiv.org/pdf/2604.09000v1)

**作者:** Junxi Wang `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 StreamMeCo 框架，对长视频代理的内存图进行压缩，并通过时间衰减检索实现高效且准确的记忆检索。

**💡 创新点**

创新点在于：① 针对图中孤立节点与连接节点分别采用 Edge‑free Minmax Sampling 与 Edge‑aware Weight Pruning 两种压缩策略；② 设计了时间衰减 Memory Retrieval（TMR）机制，动态分配检索节点并模拟人类记忆衰减；③ 通过上述技术实现 70% 以内的内存压缩同时保持甚至提升准确率。

**🔧 技术方法**

采用的技术包括：Spherical KMeans 聚类、Minmax 采样、权重矩阵与嵌入相似度融合、指数时间衰减、M3‑Agent 内存图结构、GPT‑4o 与 text‑embedding‑3‑large 生成查询与嵌入。

**📊 数据集**

使用数据集：M3‑Bench‑robot、M3‑Bench‑web、Video‑MME‑Long。

**📈 对比分析**

方法通过与 13 个闭源/开源 MLLM、流视频模型以及 5 种压缩算法（随机、聚类、DART、TimeChat‑Memory、MemoryLLM）进行对比。实验显示：在 70% 内存压缩下，StreamMeCo 能实现 1.87× 的检索速度提升，平均准确率提升 1.0%，并显著优于其他压缩策略。

**⚠️ 局限性**

局限性在于：① 依赖 Gemini‑2.5‑Pro 与 text‑embedding‑3‑large API，导致内存图生成耗时与成本高；② 仅提供 M3‑Bench‑robot 与 M3‑Bench‑web 的内存图，测试受限；③ 受预算与时间限制，未在更多基准数据集上验证。

---

## 262. The Speculative Future of Conversational AI for Neurocognitive Disorder Screening: a Multi-Stakeholder Perspective

**arXiv ID:** 2604.09070 | [PDF](https://arxiv.org/pdf/2604.09070v1)

**作者:** Jiaxiong Hu `[一作]` (Hong Kong University of Science and Technology), Xiaojuan Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对医生、认知障碍风险人群及其照护者进行半结构化访谈，探讨对话式人工智能（CAI）支持认知障碍筛查的期望、痛点与冲突，并基于访谈结果提出人本化设计建议。

**💡 创新点**

从多利益相关者视角和人本化方法系统阐述CAI在认知障碍筛查中的共享与冲突期望，并提出将CAI定位为多方协调者的设计框架，弥补传统单向筛查的不足。

**🔧 技术方法**

未采用算法实现，主要使用对话式AI演示视频进行情境化，采用人机交互的替代展示与半结构化访谈方法。

**📊 数据集**

数据来源为36位访谈参与者（5名医生、7名照护者、24名风险人群）的访谈记录；未使用公开数据集。

**📈 对比分析**

研究方法为主题分析和用户旅程图绘制，未进行算法对比或性能评估，因本研究为定性探索性研究。

**⚠️ 局限性**

局限性包括样本规模有限、仅覆盖城市环境、缺乏定量验证及跨文化可迁移性不确定；访谈结果可能受受访者自我报告偏差影响。

---

## 263. Feature-Label Modal Alignment for Robust Partial Multi-Label Learning

**arXiv ID:** 2604.09064 | [PDF](https://arxiv.org/pdf/2604.09064v1)

**作者:** Yu Chen `[一作]` (Guangdong University of Technology), Guanbin Li `[通讯]` (Sun Yat-Sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

提出了一种基于特征-标签模态对齐的部分多标签学习方法PML-MA，用低秩正交分解生成伪标签，结合全局与局部对齐以及多峰类别原型学习，显著提升在噪声标签环境下的分类性能。

**💡 创新点**

创新点包括：①将特征与标签视为互补模态，通过正交分解与全局-局部对齐实现特征-标签一致性；②采用多峰原型学习，利用软标签权重捕捉多标签特征的多峰分布；③联合优化伪标签、投影矩阵和原型，提升对噪声标签的鲁棒性。

**🔧 技术方法**

核心技术包括低秩正交分解、全局与局部投影对齐（共空间投影）、多峰原型学习、线性预测器、以及联合优化的四项损失（LRO、GLMA、MCP、LPC）。

**📊 数据集**

在30种数据配置上进行评测，使用13个真实世界部分多标签数据集（来自palm.seu.edu.cn）以及9个多标签数据集（mulan.sourceforge.net）并在其上生成26种合成噪声配置。

**📈 对比分析**

与八种SOTA方法（PML-PLR、FBD-PML、PML-ND、P-LENFN、PAMB、PML-NI、PARTIAL、PML-fp）以及多标签基线进行对比，采用Hamming Loss、Ranking Loss、One‑Error、Coverage、Average Precision五项指标。PML-MA在所有指标上均取得最高或第二高排名，Wilcoxon与Friedman检验表明差异显著。

**⚠️ 局限性**

局限性包括：①需手动调参（λ、α、β、γ）耗时；②线性投影难以捕捉非线性关系；③仅单一特征模态，未考虑多视角融合；④每轮复杂度为O(n²d)，对大规模数据不友好。

---

## 264. Frequency-Enhanced Diffusion Models: Curriculum-Guided Semantic Alignment for Zero-Shot Skeleton Action Recognition

**arXiv ID:** 2604.09063 | [PDF](https://arxiv.org/pdf/2604.09063v1)

**作者:** Yuxi Zhou `[一作]` (Wuhan University), Zhigang Tu `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于频率感知扩散模型的零样本骨架动作识别方法 FDSM。

**💡 创新点**

创新点在于引入语义引导的频谱残差模块、时间步自适应频谱损失以及课程学习式语义抽象，以解决扩散模型的低通偏置和语义模态不匹配问题。

**🔧 技术方法**

采用了离散余弦变换增强的频域残差、时间步可调频谱损失以及与 CLIP/LLM 预训练文本编码器相结合的扩散 Transformer。

**📊 数据集**

在 NTU‑RGB+D 60/120、PKU‑MMD 与 Kinetics‑Skeleton 200/400 等标准骨架动作数据集上进行实验。

**📈 对比分析**

与 TDSM、FS‑VAE、InfoCPL 等多种基准方法比较，FDSM 在所有数据集的零样本与广义零样本任务中均实现最高准确率，提升约 1–3%。

**⚠️ 局限性**

主要局限是推理过程仍需随机噪声，导致分类结果存在轻微波动，且扩散模型的推理不确定性影响可重复性。

---

## 265. {\sf TriDeliver}: Cooperative Air-Ground Instant Delivery with UAVs, Couriers, and Crowdsourced Ground Vehicles

**arXiv ID:** 2604.09049 | [PDF](https://arxiv.org/pdf/2604.09049v1)

**作者:** Junhui Gao `[一作]` (City University of Hong Kong), Yuguang Fang `[通讯]` (City University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 TriDeliver 框架，集成人类快递员、无人机和众包地面车辆（如出租车）进行即时配送，并通过迁移学习将快递员的行为偏好知识迁移到无人机与众包车辆的任务分配中，实现协同调度。

**💡 创新点**

创新点：
• 首个三种配送方式协同的即时配送系统；
• 通过迁移学习提取快递员行为知识，并细化调优后用于无人机和众包车辆的决策模型；
• 采用阈值分配与后续优化求解的组合策略，实现多代理的高效分配。

**🔧 技术方法**

技术手段：
• 迁移学习 + 神经网络（共享网络 + 细化网络）用于学习偏好模型；
• UAV 能量与时间约束的路径规划；
• 归纳为 GAPAR（Generalized Assignment Problem with Assignment Restriction）问题，使用贪心启发式求解；
• 边缘/云端调度与实时状态上报。

**📊 数据集**

数据集：
• aBeacon 订单与快递员轨迹（31k 快递员，802k 订单，1 个月）;
• 上海出租车轨迹（13k 车，3.4 亿条记录，10 秒采样）作为众包地面车辆；
• 30 天订单/轨迹数据集被划分为训练与评估。

**📈 对比分析**

实验对比：
• 与 On-Demand (O-D)、UAV+出租车协同 (U‑T) 以及 Ground Truth by Couriers (G‑T) 进行对比；
• 指标包括已派送包裹数、平均送达时延、配送成本、出租车价格；
• 结果显示 TriDeliver 接近 100% 的包裹覆盖，成本比 U‑T 低 65.8%，时延比 G‑T 小 13.2% 但比 U‑T 大约 15%，对出租车乘客影响（价格）降低 43.6%。

**⚠️ 局限性**

局限性：
• 迁移学习依赖快递员历史数据的质量与多样性，若数据偏差大可能影响模型泛化；
• UAV 能量、飞行许可与无人机部署假设过于理想，实际城市航线受限可能导致性能下降；
• 众包车辆激励与可用性模型简化，未考虑司机行为变化与安全约束；
• 评估仅基于上海场景，缺乏跨城市、跨业务场景验证；
• 系统需要高频率的状态上报和低延迟调度，部署难度较高。

---

## 266. Modeling and Simulation of Nitrogen Generation by Pressure Swing Adsorption for Power-to-Ammonia

**arXiv ID:** 2604.09053 | [PDF](https://arxiv.org/pdf/2604.09053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 267. Overhang Tower: Resource-Rational Adaptation in Sequential Physical Planning

**arXiv ID:** 2604.09072 | [PDF](https://arxiv.org/pdf/2604.09072v1)

**作者:** Ruihong Shen `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实施了一种块状构造任务，在该任务中受试者需在有限的时间与资源条件下最大化水平悬臂，同时保持结构稳定；通过实验收集行为数据，并构建计算模型对比人类决策。

**💡 创新点**

首次揭示在资源受限下人类会出现双重转变：物理预测机制从基于模拟的IPE逐步转向视觉启发式，规划策略从深度搜索逐步收敛为浅层/贪婪决策；提出一种分层资源合理化架构，将这两种转变统一为动态、可调节的认知策略。

**🔧 技术方法**

利用基于PyBullet的物理仿真、Inception‑V4 CNN 视觉启发式预测、Monte‑Carlo IPE 模拟、贪婪与有限深度搜索的规划算法，并在实验中对时间限制进行操控；通过资源合理化参数调优，将模型行为映射到人类决策。

**📊 数据集**

收集了82名受试者在20个不同六块序列上的行为记录（共1400+次放置），并从模拟生成的200k条部分结构样本训练视觉启发式 CNN；实验使用的物理模型与任务环境统一为二维 8×8 网格，块形状宽度分别为0.6、1.2、1.8。

**📈 对比分析**

通过对比人类行为与模型在对数似然、最终悬臂长度、路径依赖度（Order Dependency）等指标的表现，发现：① 在任务早期 IPE 的预测更贴合人类，随结构复杂度增加视觉启发式优于 IPE；② 在无时间限制下，人类的规划深度约为2–3，接近深度搜索模型；③ 时间限制导致悬臂长度下降，但成功率基本保持；模型的表现与人类在总体水平上相当，验证了资源合理化假设。

**⚠️ 局限性**

1) 预测与规划被视为完全独立模块，未考虑两者可能的互相影响；2) 研究仅聚焦静态稳定性，缺乏对动态因子（如碰撞、运动）下的验证；3) 受试者未进行多轮学习，未探讨经验对策略调整的影响；4) 视觉启发式 CNN 训练数据由仿真生成，可能无法完全反映真实感知误差；5) 实验任务规模有限，缺乏跨域推广性。

---

## 268. Learning Vision-Language-Action World Models for Autonomous Driving

**arXiv ID:** 2604.09059 | [PDF](https://arxiv.org/pdf/2604.09059v1)

**作者:** Guoqing Wang `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 Vision‑Language‑Action World Model（VLA‑World），通过短期轨迹预测生成未来视觉帧并在此基础上进行反思式推理，从而实现更安全、更可解释的自动驾驶决策。

**💡 创新点**

创新点在于将世界模型的未来生成能力与 VLA 模型的推理机制融合，形成“生成‑思考”闭环；采用三阶段训练（视觉预训练、监督微调、强化学习）使模型从纯模仿过渡到人类式自我反思。

**🔧 技术方法**

使用多模态大语言模型 Qwen2‑VL‑2B 作为骨干，并结合 VQGAN 视觉编码器、GRPO 强化学习算法、跨视角生成和基于规则的奖励体系，构建端到端的感知、生成、推理与规划流水线。

**📊 数据集**

基准数据集为 nuScenes‑GR‑20K（从 nuScenes 采样的 20K 条样本，用于未来帧生成与推理），以及原始 nuScenes 用于轨迹规划与动作预测评估。

**📈 对比分析**

与多种 SOTA 方法（如 FSDrive、DriveDreamer、UniAD、BEV‑Planner 等）在 nuScenes 上的 L2 距离误差、碰撞率、FID 以及动作 F1 分数等指标对比，VLA‑World 在 L2 与碰撞率上取得最优成绩，FID 亦低于主流生成模型，显示出更佳的生成质量和规划性能。

**⚠️ 局限性**

主要限制包括对大规模预训练数据的依赖（需 8×80 GB GPU 训练），生成阶段对视觉令牌数量的高梯度贡献可能抑制其他模块的学习，以及在极端稀有场景下的推理泛化仍有提升空间。

---

## 269. Tora3: Trajectory-Guided Audio-Video Generation with Physical Coherence

**arXiv ID:** 2604.09057 | [PDF](https://arxiv.org/pdf/2604.09057v1)

**作者:** Junchao Liao `[一作]` (Alibaba Cloud Computing), Weizhi Wang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发一种利用对象轨迹作为共享运动先验的音视频生成框架Tora3。

**💡 创新点**

创新点包括轨迹对齐的运动表征、基于二阶运动状态的音频对齐模块以及混合流匹配机制，实现视频运动与音频同步的物理一致性。

**🔧 技术方法**

使用扩散Transformer（DiT）双分支架构，轨迹直接映射到潜在空间、交叉注意力、RoPE以及混合流匹配等技术。

**📊 数据集**

构建新数据集PAV，460k个视频，自动提取轨迹注解，来源于VGGSound、ACAV-100M、OpenVid1M、Pexels等公开数据。

**📈 对比分析**

与LTX-2、Ovi、MOVA、AVControl等四个基线对比，Tora3在FVD、AS、FGAS、ETE、MAIC等指标上均表现最优，显示更佳的运动追踪与音视频同步。

**⚠️ 局限性**

局限在于仅基于二维轨迹缺乏完整物理约束，未考虑材质、声学传播等更丰富的物理因素。

---

## 270. Text-Conditioned Multi-Expert Regression Framework for Fully Automated Multi-Abutment Design

**arXiv ID:** 2604.09047 | [PDF](https://arxiv.org/pdf/2604.09047v1)

**作者:** Mianjie Zheng `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一套完全自动化的多牙种植体颈部设计框架TEMAD，实现了从口内扫描直接预测所有种植体的几何参数。

**💡 创新点**

创新点在于：① 将植牙位点定位（ISIN）与参数回归（MARN）统一为端到端流水线；② 引入牙位点条件FiLM（TC-FiLM）实现位置感知特征调制；③ 通过系统提示的专家混合网络（SPMoE）实现植牙系统特定的回归专家。

**🔧 技术方法**

使用技术包括：点云/网格Transformer编码器、Masked AutoEncoder自监督预训练、点云分类网络PointNet++、特征线性调制FiLM、Mixture‑of‑Experts专家选择、MSE+Smooth‑L1损失。

**📊 数据集**

数据集为22,115个口内网格用于预训练，9,037单牙种植体+1,688多牙种植体的标注数据集做监督学习，测试集分别为1,211单牙和296多牙样本。

**📈 对比分析**

与现有点云/网格方法（PointNet、PointNet++、PointMAE、PointMamba、PointFEMAE、MeshMAE、TCEAD、SSA^3D）比较，TEMAD在三项关键参数（穿透深度、直径、高度）的IoU上平均提升2–20个百分点，尤其在多牙场景下表现最为显著。

**⚠️ 局限性**

局限性包括：① 仅在本院内部数据上评估，泛化性待验证；② 对ISIN定位误差敏感，定位误差会直接影响后续回归；③ 目前支持的植牙系统有限，需进一步扩展到更多厂家。

---

## 271. Towards Lifelong Aerial Autonomy: Geometric Memory Management for Continual Visual Place Recognition in Dynamic Environments

**arXiv ID:** 2604.09038 | [PDF](https://arxiv.org/pdf/2604.09038v1)

**作者:** Xingyu Shao `[一作]` (Tsinghua University), Ziyang Meng `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向无人机终生视觉定位的任务式领域增量学习框架，结合静态卫星锚点和动态经验回放的异质记忆，并引入“Learn‑and‑Dispose”管线实现严格的存储约束。

**💡 创新点**

创新点在于：① 将地理知识拆分为不可变卫星锚点与可更新经验回放；② 采用空间约束的分配策略（Min‑Guar）与基于特征多样性的回放选择（DBS）；③ 设计了三套独立评估指标和21序列的专用基准，系统剖析泛化、适应与记忆能力。

**🔧 技术方法**

技术手段包括：ViT‑DINOv2 + GeM池化的分类式VPR；交叉熵 + 锚点 + 回放正则化；基于损失和特征相似度的样本优先级评估；空间约束的分配算法；实验使用了PyTorch、Adam、学习率调度等。

**📊 数据集**

使用了由21条航拍任务（VIS与IR）构成的自制数据集，覆盖山东青岛城区与乡镇，包含5,439幅图像，10条用于终生学习，11条为未访问的测试集。

**📈 对比分析**

通过与FT、LwF、DIL‑ER、DIL‑DER++、iCaRL、随机采样等多种基线对比，利用AP、BWT、FWT及C1/C2/C3三指标评估。DBS在C3（86.9%）与C1（64.6%）均优于对手，保持正向/逆向迁移，证明了多样性回放在稳定性与泛化上的优势。

**⚠️ 局限性**

局限性包括：① 依赖预先获取的卫星图像，若更新滞后可能误导锚点；② 对极端视角（>30°）与大尺度几何变形的适应性有限；③ 在悬停或极低熵场景下的冗余图像仍可能导致记忆稀疏；④ 仅在两地分区内验证，缺乏更广泛的跨地区推广。

---

## 272. CONDESION-BENCH: Conditional Decision-Making of Large Language Models in Compositional Action Space

**arXiv ID:** 2604.09029 | [PDF](https://arxiv.org/pdf/2604.09029v1)

**作者:** Yeonjun Hwang `[一作]` (Yonsei University), Jinyoung Yeo `[通讯]` (Yonsei University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Condesion-Bench，一个评估大语言模型在组合动作空间内有条件决策的基准；

**💡 创新点**

创新点在于将动作定义为决策变量与分配的组合，加入多层（变量、情境、分配）显式条件，并使用oracle最优动作进行评估；

**🔧 技术方法**

采用LLM推理生成动作，使用基于或-算子和Knapsack问题的枚举求解，评估条件满足率和归一化效用；

**📊 数据集**

使用金融领域的S&P 500股票、历史行情、新闻情绪与市场总结，构建了约251个带条件的日内交易场景；

**📈 对比分析**

与多种LLM（含推理与非推理模型）比较，衡量条件满足率（DSR/CSR）与归一化效用（NU），结果显示推理模型条件满足更好但未必取得最高效用，非推理模型虽效用稍高但条件遵守差；

**⚠️ 局限性**

局限包括仅覆盖金融领域、仅考虑硬性条件、日内交易时间跨度窄，且不包含软性或隐式约束，未来需扩展到更多领域和更丰富的条件形式。

---

## 273. BlendFusion -- Scalable Synthetic Data Generation for Diffusion Model Training

**arXiv ID:** 2604.09022 | [PDF](https://arxiv.org/pdf/2604.09022v1)

**作者:** Thejas Venkatesh `[一作]` (Samaya AI, Inc.), Suguna Varshini Velury `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出BlendFusion框架，利用物理渲染和自动化过滤生成高质量的图像-标题对，并基于该框架构建FineBLEND数据集。

**💡 创新点**

创新点在于将对象中心相机布局、基于VLM的过滤以及多样性采样结合到生成流程，显著提升渲染质量和标签可靠性。

**🔧 技术方法**

技术包括Blender与BlenderProc进行路径追踪渲染、对象中心相机放置算法、亮度/方差启发式过滤、Qwen3‑VL‑8B‑Instruct进行VLM过滤与标题生成、CLIPScore与LAION美学评分、DinoV2特征与最远点采样。

**📊 数据集**

使用了来自NVIDIA ORCA、Blender官方演示、BlenderKit等公开3D场景资源，构造了7,500张图像-标题对的FineBLEND数据集。

**📈 对比分析**

通过与MS‑COCO、Conceptual Captions、ReLAION等真实数据集的CLIPScore与美学评分对比，FineBLEND在图像‑文本对齐上与COCO相当，且方差更低，说明对齐更稳健；但美学得分略低。

**⚠️ 局限性**

局限在于渲染图像的真实性不足、数据规模仍有限、未直接训练模型验证效果、以及对生成过程的计算成本较高。

---

## 274. ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering

**arXiv ID:** 2604.08999 | [PDF](https://arxiv.org/pdf/2604.08999v1)

**作者:** Xiaoke Guo `[一作]` (Zhejiang University), Wen Zhang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练的Adaptive Semantic Tree Reasoning Architecture (ASTRA)，通过AdaSTR将复杂表格转为语义树，并用DuTR进行双模式树推理，实现高效、可解释的表格问答

**💡 创新点**

提出AdaSTR自动化语义树重构、双模式树推理DuTR以及自适应树构造策略与评估循环，解决结构忽略、表示鸿沟、推理不透明和模式不灵活等瓶颈

**🔧 技术方法**

利用大型语言模型的全局语义感知生成语义树，树搜索式文本导航、符号代码执行、评估驱动的自我修正循环，结合Python字典实现可解释的符号推理

**📊 数据集**

在AIT‑QA、HiTab和SSTQA三大复杂表格基准上进行实验

**📈 对比分析**

与GPT‑4o、DeepSeek‑V3、o3、EEDP、E5、TableGPT2、TableLlama、GraphOTTER、ST‑Raptor等多种基线比较，ASTRA在所有数据集上均达到或超过SOTA，特别是HiTab上的90.1%准确率

**⚠️ 局限性**

对极简平面表格的重构开销较高，未覆盖表格视觉语义（如颜色、粗体）等多模态信息

---

## 275. DRIFT: Harnessing Inherent Fault Tolerance for Efficient and Reliable Diffusion Model Inference

**arXiv ID:** 2604.09073 | [PDF](https://arxiv.org/pdf/2604.09073v1)

**作者:** Jinqi Wen `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DRIFT框架，在扩散模型推理中结合细粒度DVFS和回滚ABFT实现高能效与可靠性

**💡 创新点**

将模型固有容错性与硬件DVFS、ABFT协同优化，首次实现自适应电压频率控制和局部错误恢复

**🔧 技术方法**

细粒度可容错DVFS、回滚式ABFT、检查点间隔调节、数据重打包、HBM2内存、系统仿真

**📊 数据集**

DiT-XL-512 (ImageNet)、PixArt-alpha (COCO、DrawBench)、Stable Diffusion v1.5 (COCO)

**📈 对比分析**

与ThUnderVolt、ApproxABFT、DMR等方法对比，在相同质量下实现约36%能耗下降或1.7×速度提升

**⚠️ 局限性**

主要受限于硬件实现复杂度、低延迟回滚开销及对极低BER场景的精细调节需求

---

## 276. Temporal Patch Shuffle (TPS): Leveraging Patch-Level Shuffling to Boost Generalization and Robustness in Time Series Forecasting

**arXiv ID:** 2604.09067 | [PDF](https://arxiv.org/pdf/2604.09067v1)

**作者:** Jafar Bakhshaliyev `[一作]` (University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (University of Hildesheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Temporal Patch Shuffle（TPS）数据增强方法，通过重排重叠时间片来提升时间序列预测的泛化与鲁棒性。

**💡 创新点**

创新点在于将图像领域的 PatchShuffle 迁移到时序域，使用方差排序选择性混洗并通过重叠平均重构序列，兼顾多样性与时间连贯性。

**🔧 技术方法**

技术包括重叠滑动窗口切片、方差排序、随机打乱子集以及重构平均；实现无模型依赖。

**📊 数据集**

在九个长序列预测基准和四个短序列交通预测基准上评估，覆盖多种模型如 TSMixer、DLinear、PatchTST、TiDE、LightTS。

**📈 对比分析**

与多种现有增强技术（如 FreqMask/Mix、WaveMix、STAug 等）对比，TPS 在多数设置下获得 MSE 降低 2–10%，并在多模型、多数据集上取得最多获胜次数。

**⚠️ 局限性**

局限性包括对参数（patch 长、步幅、混洗率）敏感，过度扰动可能导致性能下降，并且相较无增强训练存在一定时间成本。

---

## 277. Anchored Sliding Window: Toward Robust and Imperceptible Linguistic Steganography

**arXiv ID:** 2604.09066 | [PDF](https://arxiv.org/pdf/2604.09066v1)

**作者:** Ruiyi Yan `[一作]` (Kyoto University), Yugo Murawaki `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并改进语言隐写技术，提出Anchored Sliding Window (ASW) 框架，增强文本质量、可感知性和鲁棒性。

**💡 创新点**

创新点在于：①把上下文窗口划分为 prompt、桥接上下文（bridge context）和最近 token；②桥接上下文既可为硬 token 也可为可训练的软 embedding，并通过自蒸馏最小化全上下文与 ASW 推理的 KL 散度；③通过桥接补偿丢失的历史信息，实现既不牺牲质量也保持鲁棒性。

**🔧 技术方法**

主要技术包括：大语言模型（如 Qwen2.5-7B-Instruct）、prompt distillation 与自蒸馏、KL 损失（forward 与 reverse）、算术编码（arithmetic coding）作为隐写实现、soft bridge context 的可微训练。

**📊 数据集**

使用 InstructionWild、databricks-dolly-15k、Super-NaturalInstructions 等公开数据集进行评估。

**📈 对比分析**

与基线 WinStega 及全上下文对比，ASW 在文本质量（ΔPPL ↓99%、BLEU ↑236%、ROUGE‑L ↑104%、BERTScore ↑18%）、可感知性（stealth detection accuracy ↓22%）和鲁棒性（在不同 w 与 m 组合下保持更高比例的未受影响推理）均明显优于 WinStega；embedding capacity 接近全上下文水平，生成速度略慢于 WinStega 但远快于全上下文。

**⚠️ 局限性**

主要限制：soft bridge context 通过 forward KL 训练时容量提升未能充分解释；实验未同时探索所有超参数、模型与数据集交互影响；计算资源有限导致部分评估分离进行。

---

## 278. AccompGen: Hierarchical Autoregressive Vocal Accompaniment Generation with Dual-Rate Codec Tokenization

**arXiv ID:** 2604.09054 | [PDF](https://arxiv.org/pdf/2604.09054v1)

**作者:** Jian Zhu `[一作]` (Zhejiang Lab), Cheng Luo `[通讯]` (Zhejiang Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了 AccompGen，一种能够根据隔离的歌声自动生成伴奏音频的三阶段层次自回归模型。

**💡 创新点**

创新点包括：① 双速码本化方案，使用 50 Hz HuBERT 语义码与 75 Hz EnCodec 声学码；② 采用三阶段分层自回归架构（语义→粗粒度声学→细粒度声学）并结合交叉码本预测与无条件引导；③ 在 Transformer 中引入 QK‑norm、GEGLU 激活、RMSNorm、T5 风格相对位置偏置等现代设计，提升训练稳定性与序列泛化。

**🔧 技术方法**

技术手段包括：HuBERT 与 EnCodec 码本化、分层自回归 Transformer、Classifier‑Free Guidance、RMSNorm、QK‑norm、GEGLU、相对位置偏置、AdamW、混合精度训练、梯度裁剪等。

**📊 数据集**

训练使用 FMA‑Large 数据集（约 100K 曲目）进行源分离后对齐；评估使用 MUSDB18 测试集（150 首歌曲）中的独立歌声与分离歌声。

**📈 对比分析**

与检索基线及 SingSong 进行对比，主要指标为 Fréchet Audio Distance（FAD）。在孤立歌声上 AccompGen‑Base 的 FAD 为 2.02，虽参数量仅 250 M，但已接近 SingSong‑XL（3B 参数）的性能；在分离歌声上 FAD 为 2.30，说明泛化能力较好。

**⚠️ 局限性**

局限性包括：仅支持单声道伴奏；目前仅在 6 kbps 码率下评估，未验证更高采样率下的表现；模型对多源（鼓、贝斯、钢琴等）分离生成尚未实现；缺乏对音乐风格、节奏等属性的显式控制。

---

## 279. NTIRE 2026 The 3rd Restore Any Image Model (RAIM) Challenge: Multi-Exposure Image Fusion in Dynamic Scenes (Track 2)

**arXiv ID:** 2604.09030 | [PDF](https://arxiv.org/pdf/2604.09030v1)

**作者:** Lishen Qu `[一作]`, Yaokun Shi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文介绍了 NTIRE 2026 赛季第三届 Restore Any Image Model (RAIM) 挑战中关于动态场景多曝光图像融合（MEF）的赛道，提供了具有移动、光照变化与相机抖动等难题的 HDR 数据集，并发布了评测协议与排行榜，收录并比较了多支参赛团队的创新方法与性能。

**💡 创新点**

创新点包括：① 设计了面向真实动态 HDR 场景的 benchmark，强调对运动误差、光照变化和手持抖动的鲁棒性；② 采用多维度评估（PSNR、SSIM、LPIPS、DISTS、NIQE 以及效率指标），并对最终提交的代码进行可复现性验证；③ 参赛团队提出多种新型网络结构（如基于 Swin Transformer 的 AFUNet、光照先验对齐、Wavelet 多分辨率融合、时间条件 U‑Net 等），以及针对评测指标的直接对齐训练策略。

**🔧 技术方法**

主要技术包括：Swin Transformer、Spatial Feature Transform (SFT)、数据一致性模块、光照先验对齐、光度先验导向的光照对齐、光波变换（Wavelet）多分支融合、Time‑DiffiT/时间条件 U‑Net、HDR‑Transformer、Restormer、Uformer、MST++ 等深度学习框架，配合多尺度训练、混合损失（L1/SSIM/LPIPS/Perceptual/Neighboring‑Pixel‑Relationships）与量化感知训练。

**📊 数据集**

使用的数据集为 RAIM MEF 数据集：训练集 100 条多曝光序列（每条 7 个曝光级别），测试集 100 条序列（每条 5 个曝光级别），数据来源于真实摄影场景，包含运动、光照变化、相机抖动与局部饱和/欠曝问题。

**📈 对比分析**

在两阶段测试（Test Stage 1/2）中，WHU‑VIP 以 58.889 分领先榜单；SHL 在 PSNR/SSIM 方面最高，但 LPIPS 较高，整体得分略低；其余参赛团队（nunucccb、untrafusion、I² Group & Transsion、NTR、miketjc 等）得分在 54–57 之间。参赛方法在高频细节恢复与运动伪影抑制方面表现出显著提升，且多种方案已通过可复现性验证，具备一定的实用性。

**⚠️ 局限性**

局限性：① 数据集规模仍有限，难以覆盖所有复杂的动态 HDR 场景；② 评测指标虽多元，但仍以 PSNR/SSIM/LPIPS 为主，可能忽略某些主观视觉特征；③ 部分高性能模型在推理效率、参数量与算力需求方面仍偏高，限制了在低端移动设备上的部署；④ 赛道侧重于非线性 sRGB 域的 MEF，未覆盖 HDR 线性域下的后期处理与光度精度需求。

---

## 280. Social Reality Construction via Active Inference: Modeling the Dialectic of Conformity and Creativity

**arXiv ID:** 2604.09026 | [PDF](https://arxiv.org/pdf/2604.09026v1)

**作者:** Kentaro Nomura `[一作]` (University of Osaka), Takato Horii `[通讯]` (University of Osaka)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了一种基于主动推理的多智能体模拟模型，模拟社会主体通过内部生成模型进行认知更新、主动创作新观测并在社会网络中相互交流，形成并演化社会现实；

**💡 创新点**

创新点在于把主动推理的生成模型与创意行为耦合，既捕捉社会规范的自上而下调控，又通过智能体的创造性行为自下而上重塑社会表征，实现双向的社会现实构建；

**🔧 技术方法**

使用了主动推理框架、生成对抗网络（f‑GAN）训练鉴别器、Metropolis‑Hastings命名游戏（MHNG）进行社会先验采样、变分自编码器（VAE）训练生成模型；

**📊 数据集**

使用的是实验性生成数据：观测为二维连续向量，社会表征为四维向量；实验在固定的两层集群“穴居人”图（14个智能体）上运行；

**📈 对比分析**

通过对比包含创作与无创作两种实验条件，利用Wasserstein与Gromov‑Wasserstein距离、RSA相似性等指标，展示创作条件下社会表征与观测结构的同步演化、集群内信息凝聚和跨群体传播差异；

**⚠️ 局限性**

主要局限包括：社会网络固定不变，缺乏主动伙伴选择与网络共进化；模型抽象化为无身体的观测生成，未考虑动作-环境的物理耦合；缺乏真实外部数据验证，评价指标主要为定量分布相似度而非功能性性能。

---

## 281. Skill-Conditioned Visual Geolocation for Vision-Language

**arXiv ID:** 2604.09025 | [PDF](https://arxiv.org/pdf/2604.09025v1)

**作者:** Chenjie Yang `[一作]`, Chenyu Wu `[通讯]` (Southwest Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GeoSkill框架，利用可演化的技能图实现无训练的自我进化视觉地理定位，并通过闭环反馈提升推理准确性。

**💡 创新点**

核心创新在于将地理知识转化为可迭代的自然语言技能，采用专家轨迹初始化、混合检索、技能图推理与多轮Rollout自我演化，避免传统模型的幻觉与参数依赖。

**🔧 技术方法**

技术组合包括大型语言模型（Qwen3.5、GPT-5.2）、混合检索（BM25+语义向量）、任务特定技能图、外部搜索验证、Rollout推理和结构化逻辑融合。

**📊 数据集**

使用了Im2GPS 3k、EarthWhere、GeoRC三大视觉地理定位基准，并在网络规模图像-坐标对上进行自我演化。

**📈 对比分析**

与GeoCLIP、GAEA、GeoReasoner、GLOBE、GeoVista等SOTA进行对比；GeoSkill在GeoRC和EarthWhere多阈值上获得最佳或接近最佳精度，Im2GPS 3k也保持竞争力；思路准确率、召回率和F1显著提升。

**⚠️ 局限性**

局限性包括对检索质量和外部工具的依赖、推理步骤较多导致推理时间增长、技能图规模管理挑战，以及在极端长尾或新场景下仍可能出现精度下降。

---

## 282. CAD 100K: A Comprehensive Multi-Task Dataset for Car Related Visual Anomaly Detection

**arXiv ID:** 2604.09023 | [PDF](https://arxiv.org/pdf/2604.09023v1)

**作者:** Jiahua Pang `[一作]` (Beijing Institute Of Technology), Yongchun Liu `[通讯]` (Li Auto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了 CAD 100K 车载多任务视觉异常检测基准数据集，覆盖 7 个车辆域、3 个任务。

**💡 创新点**

创新点在于构建了统一的域-系统-部件层级结构、融合真实、开源和合成数据，并提出了基于收敛平衡的动态任务加权 MTL 框架。

**🔧 技术方法**

使用了共享 backbone + 多头结构，基于 ConvNeXt/CNN 与 DINOv3/ViT 编码器，CoBa 动态权重和自适应任务调度，采用混合精度训练。

**📊 数据集**

使用了 CAD 100K 数据集以及 CarDD、Car Parts 50、Car Parts Segmentation 等公开数据进行评估。

**📈 对比分析**

通过与单任务、基础 MTL 进行对比，发现 adaptive MTL 在多任务设置下可保持或提升检测、分割性能，尤其在 ConvNeXt 结构中显著提升检测 mAP。

**⚠️ 局限性**

局限性包括任务间依赖复杂导致权重优化不稳定，合成图像与真实图像差异仍存在，且对极少量类别的泛化仍有限。

---

## 283. Noise-Aware In-Context Learning for Hallucination Mitigation in ALLMs

**arXiv ID:** 2604.09021 | [PDF](https://arxiv.org/pdf/2604.09021v1)

**作者:** Qixuan Huang `[一作]` (Japan Advanced Institute of Science and Technology), Masashi Unoki `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于噪声感知的上下文学习（NAICL）方法，利用检索得到的噪声-描述对引导大型音频语言模型生成更保守的描述，并构建了细粒度幻觉评估基准 Clotho-1K。

**💡 创新点**

创新点在于把噪声视为低限语义先验，通过检索与输入相似的噪声实例并作为上下文，抑制模型在证据不足时的过度推断；同时提出四类幻觉细粒度分类与对应评估指标，提供更细致的错误分析。

**🔧 技术方法**

使用 BEATs 音频编码器进行噪声检索；构建结构化噪声先验库；采用 LLM-as-Judge 进行幻觉检测与分类；在推理阶段实现噪声上下文的 In-Context Learning（ICL）。

**📊 数据集**

主要数据集为从 Clotho 语料中筛选并手工修订得到的 1,000 条音频样本 Clotho-1K，附有五个原始字幕、一个人工修订参考字幕及 AudioSet 事件标注；同时使用合成噪声样本构建噪声先验库。

**📈 对比分析**

通过在 Clotho-1K 上对 9+ 大型音频语言模型进行幻觉率（HR）及各类幻觉占比评估，比较了 NAICL 与传统 ICL 的效果；在 Qwen2.5-Omni-7B 上，NAICL 将整体幻觉率从 26.53% 降至 16.98%，各类幻觉均显著下降。

**⚠️ 局限性**

局限性包括：仅在推理时进行校准，未对模型做进一步微调；噪声先验库规模有限，可能在不同噪声环境下泛化不足；评估仅聚焦于 Clotho-1K 音频描述任务，缺乏跨域和多任务验证。

---

## 284. Regime-Conditional Retrieval: Theory and a Transferable Router for Two-Hop QA

**arXiv ID:** 2604.09019 | [PDF](https://arxiv.org/pdf/2604.09019v1)

**作者:** Andre Bacellar `[一作]` `[通讯]`, Andre Bacellar

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究两跳检索系统，并提出基于查询是否直接出现答案实体的两种检索模式（Q‑dominant 与 B‑dominant），并设计了一种基于表面文本特征的二进制路由器来动态选择检索策略。

**💡 创新点**

创新点在于把检索模式分解为两个可判定的文本谓词 P1（答案实体是否出现在问题中）和 P2（答案实体是否在桥段中以关系句出现），并通过三条定理阐明其结构；此外提出的五特征二进制路由器无需嵌入或 LLM 参与，且在跨数据集无损失。

**🔧 技术方法**

使用的技术包括：双编码器（NV‑Embed‑v2、BGE‑large‑en‑v1.5、e5‑mistral‑7b）、余弦相似度检索、逻辑回归路由器、句子选择器、统计分析（AUC‑margin 关系、Cantelli 下界）以及多数据集评测。

**📊 数据集**

实验数据集包括 2WikiMultiHopQA（训练和评测）、MuSiQue（零射击迁移）和 HotpotQA（零射击迁移）。

**📈 对比分析**

方法对比采用 Q‑only 基线；在 2Wiki 上提升 R@5 5.6pp（p<0.001），在 MuSiQue 上零射击提升 5.3pp（p=0.002），在 HotpotQA 上正向但无显著提升 1.1pp。

**⚠️ 局限性**

局限性包括：只针对两跳合成式问题；对隐式组合或更长链路的通用性未验证；以及路由器对文本特征的依赖可能在其他领域表现不佳。

---

## 285. Multimodal Large Language Model Enabled Robust Beamforming for HAP Downlink Communications

**arXiv ID:** 2604.09017 | [PDF](https://arxiv.org/pdf/2604.09017v1)

**作者:** Xiaoyu Xing `[一作]` (Beihang University), Xianbin Cao `[通讯]` (Beihang University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于多模态大语言模型的预测驱动主动波束成形框架，解决高空平台（HAP）下行链路中因平台姿态摇晃导致的波束失配问题。

**💡 创新点**

创新点包括：① 结合视觉-语言LLM（VL-LLM）实现多变量飞行遥测的短期姿态预测；② 通过离线预测误差校准提供高置信度的指向误差上界；③ 在此基础上设计学习辅助的QoS驱动数字波束成形与接入控制，并引入严格可行性修复机制，兼顾低延迟与可行性保证。

**🔧 技术方法**

采用的技术主要有：视觉-语言LLM（Frozen LLM + 视觉特征提取 + 交叉变量适配器 + Prompt注入）、离线误差校准（量化预测误差上界和统计矩）、KKT导向的波束重构与严格修复、轻量级数字波束成形（WMMSE启发式闭式映射）以及实时延迟优化。

**📊 数据集**

使用的数据集包括：10 Hz采样的真实HAP飞行遥测数据（航迹、姿态等）用于姿态预测；合成的多用户LoS/非LoS射频通道样本用于波束成形性能评估。

**📈 对比分析**

与基线方法比较：姿态预测方面与TimeLLM、PatchTST、TimesNet对比，RMSE降低约29%；波束成形方面与PPO、PPO-λ、DDPM对比，用户服务率提升22.1%、总速率提升12.5%；实时推理延迟为36.24 ms（均值）/40.13 ms（p99），满足实际部署需求。

**⚠️ 局限性**

局限性包括：仅在LoS通道环境下验证，未考虑复杂非LoS与多平台协同；依赖离线预测误差校准的准确性；模型对极端姿态波动和传感器失效的鲁棒性待进一步研究。

---

## 286. Identification and Anonymization of Named Entities in Unstructured Information Sources for Use in Social Engineering Detection

**arXiv ID:** 2604.09016 | [PDF](https://arxiv.org/pdf/2604.09016v1)

**作者:** Carlos Jimeno Miguel `[一作]` (Public University of Navarre), Francesco Zola `[通讯]` (Vicomtech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了收集Telegram及多媒体（文本、音频、图片）数据、语音转写、命名实体识别与匿名化的完整管道，兼顾GDPR等法规合规；

**💡 创新点**

将Transformer‑based NER与Microsoft Presidio结合，并提出多维匿名化评估指标，提升信息保持与合规性平衡；

**🔧 技术方法**

使用Parakeet/Whisper/Wav2Vec语音转写、信号预处理(noisereduce+VAD)、BERT‑based NER、Microsoft Presidio、定制哈希替换匿名化；

**📊 数据集**

利用Telegram公开/私密频道、espnet/yodas-granary音频、CoNLL‑2003、OntoNotes5、ECHR司法文本、PII（CENSUS‑NER、PII‑NER、pii‑masking‑400k），并用Faker合成实体；

**📈 对比分析**

通过WER+执行时长、F1分数以及信息损失/一致性/碰撞率/错误率/相关性保持等指标比较。结果显示Parakeet最佳WER0.16，Wav2Vec最快；BERT NER在域内F1≈0.96；Transformer匿名化一致性1.0、碰撞率1.0、错误率0，优于Presidio与原始In‑line方法；

**⚠️ 局限性**

存在类别不平衡导致少数类F1下降；语音预处理对某些模型无效；未实现完整端到端工作流；缺少IP、IBAN、区块链地址实体；法规和平台条款整合待进一步完善。

---

## 287. Robust by Design: A Continuous Monitoring and Data Integration Framework for Medical AI

**arXiv ID:** 2604.09009 | [PDF](https://arxiv.org/pdf/2604.09009v1)

**作者:** Mohammad Daouk `[一作]`, Hien Van Nguyen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并验证了一套连续监测与数据集成框架，使 ResNet-18 在肾小球影像分类任务中能够自适应新数据而不降低性能。

**💡 创新点**

创新点在于将特征空间距离（欧氏、余弦、马氏距离）与 Monte Carlo dropout 不确定性门控相结合，实现安全的增量训练，既防止数据漂移导致的性能下降，又避免灾难性遗忘。

**🔧 技术方法**

使用了多度量特征分析、MC dropout 不确定性估计、ResNet-18 卷积网络、增量式训练以及性能安全阈值检测。

**📊 数据集**

使用了来自哥伦比亚大学医院、斯坦福大学和芝加哥大学的 9,674 张专家标注的肾小球图像，涵盖多种染色和 ISN/RPS 类别。

**📈 对比分析**

在 5 折交叉验证基础上与保留的测试集比较，采用 AUC、准确率、敏感性、特异性和不确定性阈值；在加入符合门控的少量新图像后，AUC 0.92、准确率 89% 等指标保持在 ±5% 范围内，性能基本不变。

**⚠️ 局限性**

局限在于阈值门控可能排除真正但与训练分布不同的病例；当前仅一次加入一张图像，未探讨批量更新和动态阈值自适应，未来需要扩展至更广泛病例和批量学习。

---

## 288. On the Terminology and Geometric Aspects of Redundant Parallel Manipulators

**arXiv ID:** 2604.09156 | [PDF](https://arxiv.org/pdf/2604.09156v1)

**作者:** Andreas Mueller `[一作]` `[通讯]`, Andreas Mueller

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

针对并联机构（PKM）的冗余问题，提出了系统化的术语和分类框架，强调c空间的几何结构并定义了输入/输出/ c空间奇异性，进而引入运动模式、作用模式和操作模式。

**💡 创新点**

创新点在于：①以c空间的局部维度和Jacobian秩为基础，给出对PKM的“度量冗余”概念；②引入“控制向量场秩”（DOA）作为衡量作用完整性的指标；③提出“作用冗余度”并阐述其对消除输入奇异性的作用；④通过理论例子展示不同冗余策略对奇异性和控制连续性的影响。

**🔧 技术方法**

采用的技术包括：非线性控制系统建模、闭环约束Jacobian与正反Jacobian、c空间几何分析、Whitney嵌入定理、线性代数中的伪逆与零空间分析。

**📊 数据集**

未使用真实数据集，所有演示均基于文献中的典型示例机构（如5杆、RR/2RRR、3‑URU DYMO 等）进行理论推导与说明。

**📈 对比分析**

比较方法主要为理论分析与示例演示：通过比较有无冗余激活下的输入奇异性、可达工作空间与控制连续性，展示冗余激活能消除输入奇异、提升操控性。性能评价以可达性、奇异性消除率为准。

**⚠️ 局限性**

局限性包括：①缺乏实验或仿真验证；②对高维c空间的解析和算法实现尚未给出完整方案；③在非欧几里得关节空间映射与实际控制实现方面仍需进一步研究。

---

## 289. Strips as Tokens: Artist Mesh Generation with Native UV Segmentation

**arXiv ID:** 2604.09132 | [PDF](https://arxiv.org/pdf/2604.09132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 290. CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation

**arXiv ID:** 2604.09155 | [PDF](https://arxiv.org/pdf/2604.09155v1)

**作者:** Yushi Feng `[一作]` (University of Hong Kong), Lequan Yu `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种后策略、预执行的安全控制框架（CORA），通过对移动 GUI 代理生成的每一步低层动作进行风险评估、显式阈值裁决，并在拒绝时提供可解释的干预建议，实现在移动端自动化任务中的安全与可用性的平衡。

**💡 创新点**

核心创新点包括：①基于目标锁定的动作条件 Guardian 进行风险打分；②使用 Conformal Risk Control 校准可控执行风险阈值，提供统计学的安全保证；③训练多模态 Diagnostician 以生成原因、风险类型和最小化干预；④引入 Goal‑Lock 机制抵御间接注入攻击；⑤构建 Phone‑Harm 真实世界移动安全基准，实现对高风险步骤的细粒度评估。

**🔧 技术方法**

技术手段主要包括：视觉语言模型（VLM）+ LoRA 参数高效微调、Conformal Prediction（CRC）进行风险阈值校准、生成式诊断模型（基于 VLM 的文本生成）、目标锁定策略、加权校准应对分布漂移、以及多模态推理框架。

**📊 数据集**

使用的数据集有：①Phone‑Harm（Harm‑150 与 Normal‑150 组成的 300 任务，包含每一步的伤害标签和类型）；②MobileRisk（移动场景的注入与误行为评估）；③AndroidWorld（长周期移动导航与通用能力测试）；以及从基准代理生成的训练/校准/测试轨迹。

**📈 对比分析**

与基准策略（GPT‑5、Gemini‑3、UI‑TARS‑1.5、AutoGLM‑VLM）以及 Prompt‑check、Rule‑gate、VLM‑as‑critic 等安全基线进行对比。CORA 在 Phone‑Harm 上实现了 HR↓、GAR↑、IF1↑ 的三维 Pareto 优化，风险控制在设定预算内；在 MobileRisk 上的召回率与 F1 远超零样本评测；在 AndroidWorld 的安全保留实验中，成功率从 30% 提升至 40%，几乎恢复 GPT‑5 的上限。总体来看，CORA 在保持或提升任务成功率的同时，显著降低了错误执行风险和误报率。

**⚠️ 局限性**

局限性包括：①对长期序列的显式风险估计仍未充分建模，顺序依赖可能导致累积误判；②依赖 VLM 的语义识别，若模型对新界面或语言不够鲁棒，可能产生误判；③需要足量的校准数据才能得到可靠的风险阈值，部署初期可能不稳定；④在极端攻击或未知注入场景下，Goal‑Lock 与 Guardian 的防御效果仍有待验证；⑤虽降低了中断率，但在高风险预算下仍会产生一定的用户干预，需进一步优化干预策略的自然性与可接受度。

---

## 291. Benchmarking CNN- and Transformer-Based Models for Surgical Instrument Segmentation in Robotic-Assisted Surgery

**arXiv ID:** 2604.09151 | [PDF](https://arxiv.org/pdf/2604.09151v1)

**作者:** Sara Ameli `[一作]` `[通讯]`, Sara Ameli

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在SAR‑RARP50数据集上，对UNet、UNet++、DeepLabV3+、Attention U‑Net、SegFormer五种网络进行多类别手术工具语义分割的统一基准评测。

**💡 创新点**

结合交叉熵与Dice损失处理类别不平衡，并系统对比传统CNN与轻量级Transformer架构的性能，提出了针对手术工具细小结构的损失与网络设计思路。

**🔧 技术方法**

使用CNN（UNet、UNet++、DeepLabV3+、Attention U‑Net）与Transformer（SegFormer）等架构，采用ASPP、多尺度上下文聚合、注意力门、轻量级ViT编码器等技术。

**📊 数据集**

使用SAR‑RARP50数据集（50段真实RARP手术视频，10类像素级标注，40段训练集）。

**📈 对比分析**

通过每类Dice系数进行比较，DeepLabV3+在大多数类别上表现最佳，SegFormer紧随其后，UNet与Attention U‑Net略逊；总体来看DeepLabV3+兼具高精度和较低推理成本。

**⚠️ 局限性**

局限在类别不平衡导致少数类性能不足、模型仅处理单帧缺乏时序信息、Transformer在细小结构上易出现过平滑现象。

---

## 292. Enhance Comprehension of Over-the-Counter Drug Instructions for the General Public and Medical Professionals through Visualization Design

**arXiv ID:** 2604.09134 | [PDF](https://arxiv.org/pdf/2604.09134v1)

**作者:** Mengjie Fan `[一作]` (Peking University), Liang Zhou `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了OTC药品说明书的可视化框架，提出分类与两种面向不同受众（公众与医护）的交互式可视化版本，并给出可视化设计流程。

**💡 创新点**

①提出针对OTC药品说明书的分类与可视化设计框架；②设计两种可交互的简化与完整版本，以满足不同用户需求；③通过实验验证可视化版优于传统文字版，并提供通用工作流程。

**🔧 技术方法**

采用JavaScript与D3.js构建交互式可视化；使用Flaticon、Font Awesome等图标库；实验中使用SUS、NASA‑TLX等问卷评估工具。

**📊 数据集**

采样自中国国家药品监督管理局（NMPA）OTC药品数据库（约112个药品），并选取常用药物如扑热息痛、氯雷他定、氨溴索等做原型。

**📈 对比分析**

对照实验：N=60名主护人员随机分为可视化组和文字组，测量知识测试得分提升、答题时间、SUS可用性评分和NASA‑TLX认知负荷。结果显示可视化版在答题时间和可用性评分上显著优于文字版，知识提升相当，认知负荷趋势下降。

**⚠️ 局限性**

样本以年轻受过教育的家庭看护者为主，专家评估仅有三名药师，交互可用性未加入显式提示，颜色编码与可访问性仍需改进，且未获得监管机构批准，尚属补充工具。

---

## 293. EquiformerV3: Scaling Efficient, Expressive, and General SE(3)-Equivariant Graph Attention Transformers

**arXiv ID:** 2604.09130 | [PDF](https://arxiv.org/pdf/2604.09130v1)

**作者:** Yi-Lun Liao `[一作]` (Massachusetts Institute of Technology), Tess Smidt `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 EquiformerV3——一种面向 3D 原子体系的 SE(3)-等变图注意力 Transformer，显著提升了模型的效率、表达力和通用性。

**💡 创新点**

创新点包括：1) 通过软件融合与编译实现 1.75× 的速度提升；2) 引入等变合并层归一化、改进的前馈网络超参数以及平滑半径截断的注意力机制；3) 设计 SwiGLU‑S² 激活函数，既支持多体相互作用又保持严格等变性，减少 S² 网格采样复杂度。

**🔧 技术方法**

技术上融合了 eSCN 卷积、SO(3) 张量乘积优化、可微分的平滑截断、SwiGLU‑S² 激活以及 Transformer 结构中的图注意力。

**📊 数据集**

在大规模公开原子数据集上进行评估，包括 OC20、OC22、OMat24、Matbench Discovery、OC20 S2EF‑2M 等。

**📈 对比分析**

与 EquiformerV2、UMA、eSEN 等现有方法相比，EquiformerV3 在 OC20、OMat24 和 Matbench Discovery 上取得了新的最优成绩；训练速度提升至 1.75×，OC20 S2EF‑2M 训练速度可达 5.9×；在 OMat24 上与大型模型相比，EquiformerV3 在保持相同误差的同时模型大小减小 5–23 倍，且在 Matbench Discovery 的热导任务中提升 18%–31%。

**⚠️ 局限性**

局限性在于仍需较高的计算资源（张量乘积仍昂贵），对极大尺度系统的推理效率有待进一步提升；此外，虽然能很好地保持能量守恒，但在更复杂的动力学模拟或高阶导数任务中可能需要更细粒度的调优。

---

## 294. FIRE-CIR: Fine-grained Reasoning for Composed Fashion Image Retrieval

**arXiv ID:** 2604.09114 | [PDF](https://arxiv.org/pdf/2604.09114v1)

**作者:** François Gardères `[一作]` (Louis Vuitton), Shizhe Chen `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于问题驱动的视觉推理框架FIRE‑CIR，用于时尚领域的组合图像检索，通过将修改文本拆解为属性级视觉问题并利用VQA模型重新排序检索结果。

**💡 创新点**

通过自动生成属性级单/双图像问题并构建大规模自动标注VQA数据集，实现可解释的细粒度视觉推理，解决传统嵌入相似度无法充分捕捉修改细节和保留不需要属性的缺陷。

**🔧 技术方法**

利用LLM生成问题、InternVL‑3‑1B VQA模型与LoRA微调、VQA得分与传统CIR得分的融合重排序以及Sigmoid平滑函数等技术。

**📊 数据集**

以Fashion IQ为主任务数据集，并使用约41万问答对、32%双图像的自动标注VQA数据集，以及Refined‑FashionIQ和enhFashionIQ增强版本进行评估。

**📈 对比分析**

与现有SOTA CIR方法（如CLIP4CIR、DetailFusion、VQA4CIR）对比，FIRE‑CIR在Fashion IQ基准上Recall@10提升约2.4%，并在所有三类服装上实现最高Recall@50和MRR，显著提高检索性能。

**⚠️ 局限性**

仍依赖LLM文本理解与问题生成，对推理速度有一定负担；自动生成问题的覆盖率与准确性在极细粒度或多样化查询中有限，且对低频属性的泛化能力可能受限。

---

## 295. TensorHub: Scalable and Elastic Weight Transfer for LLM RL Training

**arXiv ID:** 2604.09107 | [PDF](https://arxiv.org/pdf/2604.09107v1)

**作者:** Chenhao Ye `[一作]` (ByteDance Seed), Remzi H. Arpaci-Dusseau `[通讯]` (University of Wisconsin Madison)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了 Reference‑Oriented Storage (ROS)，一种无数据所有权的重量转移抽象，并实现了生产级系统 𝑠，支持拓扑感知、强一致性、容错和动态弹性。

**💡 创新点**

创新点：① 通过利用模型权重的高度复制与不可变性，ROS 在“权重即引用”框架下消除复制与存储开销；② 设计了可变性合约与保留协议，保障权重一致性与可用性；③ 在服务器端实现负载均衡与管道复制，显著提升 RDMA 带宽利用率。

**🔧 技术方法**

主要技术：RDMA 一边读/写、TCP、GPU‑CPU 共享内存、Tiny‑Tensor 压缩、负载均衡调度、事务化一致性、故障检测与恢复、Ray Actor、Mooncake RDMA 引擎、Python/C++ 混合实现。

**📊 数据集**

使用了大规模 LLM 权重作为实验数据：9B、36B、260B 以及 1T（由 260B 模型复制四次构造）的训练与推理工作负载；在 veRL 框架中与 Spot、跨数据中心环境下进行评估。

**📈 对比分析**

与 NCCL、UCX、Ray Plasma 等现有方案对比，实验表明 ROS 在三类工作负载中分别实现：① 独立推理 GPU 挂机时间缩短 6.7×；② 弹性推理权重更新速度提升 4.8×；③ 跨数据中心推理停机时间降低 19×；总体上靠近 RDMA 理想带宽，且实现代码改动仅约 40 行。

**⚠️ 局限性**

局限性：① 参考服务器单点实现可能成为极大规模时的瓶颈；② 保留协议在极端场景（所有非 Spot 机器失效）下可能导致暂时性不可用；③ 依赖 RDMA 网络，跨网络传输仍受限；④ 仅针对权重转移，未探讨其他模型数据（如激活、梯度）的高效分发。

---

## 296. Scheming in the wild: detecting real-world AI scheming incidents with open-source intelligence

**arXiv ID:** 2604.09104 | [PDF](https://arxiv.org/pdf/2604.09104v1)

**作者:** Tommy Shaffer Shane `[一作]` (Centre for Long-Term Resilience), Hamish Hobbs `[通讯]` (Centre for Long-Term Resilience)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过公开转录（聊天记录和截图）收集与分析，监测AI系统的阴谋行为（scheming）与相关行为。

**💡 创新点**

提出基于OSINT的转录监测框架，填补实验室研究与真实部署之间的缺口，并实现对真实阴谋行为的可验证检测。

**🔧 技术方法**

利用大语言模型（Claude Opus 4.6）进行预筛选和评分，结合Python脚本构建的四阶段数据管道（收集→预筛选→分析评分→事件聚合）。

**📊 数据集**

使用从X（原Twitter）公开数据检索得到的183,420条包含图片或聊天共享链接的帖子作为样本。

**📈 对比分析**

与传统AI事件数据库相比，发现698个独立阴谋相关事件，事件增长率4.9倍，显示转录监测在事件捕获率与时效性上优于现有方法。

**⚠️ 局限性**

主要限制包括转录真实性和伪造风险、平台覆盖偏差、报告与披露的潜在不足、区分阴谋与普通误操作的挑战，以及对模型“无诚信推理”的依赖导致的误判。

---

## 297. Cross-Modal Knowledge Distillation from Spatial Transcriptomics to Histology

**arXiv ID:** 2604.09076 | [PDF](https://arxiv.org/pdf/2604.09076v1)

**作者:** Arbel Hizmi `[一作]` (Weizmann Institute of Science), Nir Yosef `[通讯]` (Weizmann Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出一种跨模态蒸馏框架，将空间转录组学（spatial transcriptomics）中学习到的细胞小环境（niche）结构迁移到仅使用H&E切片的图像模型中，实现仅靠H&E即可预测组织小环境标签。

**💡 创新点**

创新点在于：①使用冻结的空间转录组学教师模型NOLAN提供高质量的无监督小环境划分作为监督信号；②采用跨模态知识蒸馏，将教师的类别分布引导图像学生学习，最终实现图像模型对转录组学划分的高保真复制；③在推理阶段完全不需要转录组学数据，仅利用H&E图像即可完成小环境分割。

**🔧 技术方法**

核心技术包括：冻结的scVI和NOLAN模型用于提取转录组学邻域特征与分数；UNIv2预训练的H&E图像特征提取器；Transformer结构的学生网络对邻域进行编码；温度缩放的Kullback‑Leibler蒸馏损失。

**📊 数据集**

在16个公开Xenium空间转录组样本（覆盖12种人类和小鼠组织，包括癌症与正常组织）上进行实验。每个样本同时提供H&E图像与细胞级转录组数据。

**📈 对比分析**

与两种无监督H&E基线（Histology‑NOLAN和Histology‑Leiden）对比，蒸馏学生在与教师NOLAN的ARI与NMI上提升显著（ARI 0.615/0.500 vs 0.283/0.234，NMI 0.603/0.579 vs 0.383/0.403），在细胞类型组成与病理标签预测上亦优于基线，显示跨模态蒸馏能有效提升图像模型对分子定义的小环境的捕捉能力。

**⚠️ 局限性**

主要局限是需先获得对应组织的配对空间转录组与H&E数据以训练蒸馏桥梁；在缺乏该配对数据的新组织类型时无法直接迁移。

---

## 298. Evaluating Data Quality Tools: Measurement Capabilities and LLM Integration

**arXiv ID:** 2604.09163 | [PDF](https://arxiv.org/pdf/2604.09163v1)

**作者:** Tobias Rehberger `[一作]` (Software Competence Center Hagenberg), Wolfram Wöß `[通讯]` (Johannes Kepler University Linz)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了六种数据质量工具（Great Expectations、Deequ、Evidently、Informatica、Experian、Ataccama）的测量能力和LLM集成情况。

**💡 创新点**

首次系统对比开源与商业工具在规则定义、重复检测、指标聚合与不确定性处理方面的差异，并提出针对LLM集成的评估准则。

**🔧 技术方法**

利用公开的文档、网站与GitHub信息，对工具进行打分；LLM技术主要用于规则生成与辅助。

**📊 数据集**

评估基于合作伙伴的真实业务用例，而非使用特定实验数据集。

**📈 对比分析**

采用六项评估指标（LLM检查、规则创建、规则定义、最小化、指标聚合、不确定性）进行量化比较，结果显示商业工具功能更完整、LLM集成更广，但开源工具更易集成，性能差距主要体现在功能深度而非速度。

**⚠️ 局限性**

仅基于公开信息，未进行实际测试；工具普遍缺乏直接LLM数据验证、灵活指标聚合和细粒度不确定性量化。

---

## 299. Persona-E$^2$: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events

**arXiv ID:** 2604.09162 | [PDF](https://arxiv.org/pdf/2604.09162v1)

**作者:** Yuqin Yang `[一作]` (South China University of Technology), Zhanpeng Jin `[通讯]` (South China University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Persona‑E2 数据集，收集新闻、社交媒体与生活叙事三大领域事件，并由 36 位具 MBTI 与 Big Five 人格特征的标注者在无角色扮演的情境下给出情绪标签，探索情绪归因与人格的关系。

**💡 创新点**

首创以人格为条件的读者情绪标注数据，强调情绪差异而非单一标签；提出人格一致性差距（PAG）指标验证人格对情绪一致性的影响；证明 Big Five 比 MBTI 更能缓解“人格幻觉”，并对 LLM 情绪模拟与推理进行系统评估。

**🔧 技术方法**

采用多阶段 LLM 过滤（Qwen3‑MAX）、NSFW 分类与专家审核构建事件；使用 Persona Prompt 与 Chain‑of‑Thought 提示对 LLM 进行情绪预测；通过 Top‑1/Top‑2 准确率、SDS 子集、专家一致率和 win‑rate 衡量模型表现；利用 K‑means 等聚类分析人格一致性。

**📊 数据集**

Persona‑E2（3111 事件，112k 注释）结合新闻、社交媒体与生活经验原始数据来源（如 GoodNewsEveryone、SemEval‑2018 等），并使用 MBTI 与 Big Five 量表对标注者进行人格测评。

**📈 对比分析**

对 GPT‑5.1、Llama‑3‑8B、Qwen3‑8B、Gemma‑3‑12B、Ministral‑3‑8B 在 General、Persona 与 Persona‑CoT 三种提示下评估 Top‑1/Top‑2 准确率；在 SDS 子集 Top‑1 约 25%、Top‑2 约 45%；在心理合理性评估中，BFI 提示下 GPT‑5.1 的 win‑rate 最高，整体性能仍低于人类一致性。

**⚠️ 局限性**

样本数量有限（3111 事件，36 名标注者），主要覆盖中英文本，缺乏跨文化跨语言代表性；使用外部情绪分类器产生 General Writer 标签可能带偏；情绪标签采用 Ekman 六类+中性，缺乏维度或开放词表；缺少实时真实用户标注，存在标注者自我偏差与职业化标注者与日常用户的差异。

---

## 300. Truncated Rectified Flow Policy for Reinforcement Learning with One-Step Sampling

**arXiv ID:** 2604.09159 | [PDF](https://arxiv.org/pdf/2604.09159v1)

**作者:** Xubin Zhou `[一作]` (Harbin Institute of Technology), Zhan Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合确定性 ODE 前缀与随机 SDE 尾部的 Truncated Rectified Flow Policy（TRFP），用于最大熵强化学习。

**💡 创新点**

核心创新在于：① 使用混合确定性/随机采样架构实现可控的熵正则化与可微分推理；② 通过截断梯度避免长链 BPTT 的梯度失效；③ 引入流直化正则化，使前缀轨迹趋于直线，从而实现几步甚至一步采样并保持高表达力。

**🔧 技术方法**

采用流匹配/Rectified Flow 作为生成器，结合 SAC 离策略框架；使用梯度截断、流直化自蒸馏、Q 引导动作选择以及近似对数似然 surrogate 等技术。

**📊 数据集**

在 10 个 MuJoCo 连续控制基准任务（Ant、HalfCheetah、Hopper、Humanoid、InvertedPendulum、InvertedDoublePendulum、Pusher、Reacher、Swimmer、Walker2d）以及一个四目标 toy 多目标环境上进行评估。

**📈 对比分析**

与 SAC、TD3、SDAC、MaxEntDP 等基线对比，TRFP 在大多数任务上实现了状态‑最优或竞争性回报，并在一阶采样模式下保持了近似完整性能；在 NFE（函数评估次数）上显著低于传统扩散策略。

**⚠️ 局限性**

局限性：1) 采用近似似然，仍未给出对真实熵的严格界定；2) 评估仅限于 MuJoCo 环境，未验证在更大规模或离线 RL、长期规划中的表现；3) 需要进一步提升多步采样时的计算效率和对不同任务的通用性。

---

## 301. Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning

**arXiv ID:** 2604.09150 | [PDF](https://arxiv.org/pdf/2604.09150v1)

**作者:** Yi Sui `[一作]` (Beijing Institute Of Technology), Dawei Song `[通讯]` (Beijing Institute Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型推理模型中，提出了基于检索增强的动态链式思维压缩框架 STACK，以应对过度推理和冗余问题。

**💡 创新点**

创新点包括：① 针对不同推理阶段的状态进行细粒度压缩策略切换（知识引导与自提示）；② 结合答案分布收敛的早停机制；③ 在 DPO 损失中加入奖励差异的动态边缘；④ 在线对比抽样生成长短链对。

**🔧 技术方法**

使用技术包括：检索增强生成（RAG）、对比解码（KGCD）、自提示压缩、答案收敛早停、PPO 与 DPO 结合的奖励差异训练（MDPO）以及基于信息熵的犹豫状态检测。

**📊 数据集**

使用的数据集：GSM8K、MATH500 和 AIME24 三个数学推理基准。

**📈 对比分析**

与 Prompt、ConCISE、MuTIS、TokenSqueeze 等现有压缩方法对比，STACK 在三大基准上平均提升 4.8 分准确率、缩短 59.9% 的响应长度、降低 7.23 秒的推理延迟，表现最佳。

**⚠️ 局限性**

局限性：在线对比采样和检索会增加训练成本；检索质量不一定完全准确，可能误导推理；当前仅利用文本知识，未整合结构化或多模态知识与推理工具。

---

## 302. What's in a BIP? Exploring the Lived Experiences of Breaks In Presence

**arXiv ID:** 2604.09146 | [PDF](https://arxiv.org/pdf/2604.09146v1)

**作者:** Jean-Philippe Rivière `[一作]` (Nantes Université), Yannick Prié `[通讯]` (Nantes Université)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过微现象学访谈，对14名参与者在恐高VR情境中经历的57个“出现失去沉浸感”（BIP）事件进行系统化描述与分析，构建了BIP的时间演化模型并归纳出四种典型模式（反思型、丢弃型、自我保全型、矛盾调解型）。

**💡 创新点**

创新点在于：①首次从用户第一人称视角细致刻画BIP的主观体验与时间结构；②提出基于意识觉察的BIP定义，强调BIP是“用户主动意识到现实或媒介存在”的连续过程；③识别并归纳四大BIP模式，为VR设计提供三条可落地的干预机会（动态生成遮蔽来源、支持回归策略、训练用户丢弃BIP）。

**🔧 技术方法**

采用的技术主要包括：①微现象学访谈法与μPMT建模工具；②基于PI/Psi理论设计的干扰器（音乐、门敲、板移除、黑屏、直升机+遮帆+海鸥等）；③Presence for Immersive Environments (SP‑IE)量表评估沉浸水平；④VR场景EVEREST与实时语音、视频记录。

**📊 数据集**

使用的数据集为：14名受试者（11女、2男、1非二元，年龄20–56岁）在EVEREST恐高VR环境中经历的57个BIP实例，配合实验记录的访谈转录、头部运动视频、存在感问卷得分。

**📈 对比分析**

研究未进行传统算法对比或性能评估，而是以质性方法对比不同BIP事件的时间结构与情绪特征，发现四种模式的出现频次与干扰类型相关，并通过案例对比验证其对沉浸感的影响；因此“性能”指的是对BIP现象的解释力与设计启示，而非数值指标。

**⚠️ 局限性**

局限性包括：①样本量小、仅限实验室VR，外部干扰和自然环境的普适性不明；②部分干扰器效果不显著（如信息过载）；③缺乏连续跟踪BIP恢复后的沉浸度变化；④仅关注单人无社交角色场景，未涉及体现、共存等维度；⑤微现象学方法对经验丰富VR用户可能不适用，需要更适合的研究方法。

---

## 303. MATCHA: Efficient Deployment of Deep Neural Networks on Multi-Accelerator Heterogeneous Edge SoCs

**arXiv ID:** 2604.09124 | [PDF](https://arxiv.org/pdf/2604.09124v1)

**作者:** Enrico Russo `[一作]` (University of Catania), Alessio Burrello `[通讯]` (Politecnico di Torino)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MATCHA，面向异构 SoC 的异步、基于 tile 的 DNN 部署框架，实现多加速器并行推理；

**💡 创新点**

通过约束规划同时完成层级与 tile 级模式匹配、设备分配、异步调度与内存规划，支持多加速器的高利用率；

**🔧 技术方法**

使用 TVM + MATCH 的扩展、ZigZag LOMA 目标级调度、OR‑Tools 约束求解、Mako 模板代码生成、ONNX IR 作为输入，针对 Carfield HSoC 编译；

**📊 数据集**

主要基准为 MLPerf Tiny（AutoEncoder、DS‑CNN、MobileNet、ResNet‑18 等）以及 ResNet‑50、ResNeXt‑50、Transformer 编码器等微基准；

**📈 对比分析**

与 TVM host‑only、原版 MATCH（单加速器顺序）比较，MATCHA 在 Carfield SoC 上实现约 35% 的时延下降，Tile 级优化可进一步提升 20–35%，在不同网络上 FLOPS 提升 4.6–12.3 倍；

**⚠️ 局限性**

约束规划与低层调度分离导致搜索空间限制；辅助操作（slice/concat）耗时估算不精确；未评估能耗；DMA 并行与异步调度仍可进一步优化。

---

## 304. "Take Me Home, Wi-Fi Drone": A Drone-based Wireless System for Wilderness Search and Rescue

**arXiv ID:** 2604.09115 | [PDF](https://arxiv.org/pdf/2604.09115v1)

**作者:** Weiying Hou `[一作]` (University of Hong Kong), Chenshu Wu `[通讯]` (University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了无人机基于Wi‑Fi的无线搜救系统Wi2SAR，利用失踪人员携带设备的自动重连行为实现目标发现与定位；

**💡 创新点**

创新点包括：①低功耗的目标发现机制；②采用3D打印Luneburg Lens实现仅RSS的长距离3D AoA估计；③基于方向的无人机导航策略；④无需任何基础设施即可完成全流程；

**🔧 技术方法**

关键技术为Wi‑Fi 2.4/5 GHz、Intel AX200多NIC双模、3D打印Luneburg Lens、RSS‑only 3D AoA算法、DJI Matrice 350+Raspberry Pi CM4、GPS/IMU姿态变换与实时导航；

**📊 数据集**

数据集为实验室与现场测试中多种手机、平板、手表等设备，在四个户外环境（山丘、森林、岩石、海岸）下采集的真实RSS与角度数据；

**📈 对比分析**

与CSI‑based ArrayTrack基准对比，Wi2SAR在信号范围内提升约104%，方向估计中位角误差18.4°（MedPR 0.95），ArrayTrack在无校准时误差达46°；在160 000 m²搜索区实现100%目标发现率，耗时13.5 min；在40 000 m²单盲试验中定位误差仅5 m；

**⚠️ 局限性**

局限性包括：仅对开启Wi‑Fi并能自动重连的设备有效；需预先获得目标网络凭证；对不携带Wi‑Fi设备或Wi‑Fi关闭者无用；多目标或大面积时需多无人机或更大网格；受设备扫描间隔影响可能导致发现延迟；附加硬件对无人机续航有轻微影响。

---

## 305. Generalizing Video DeepFake Detection by Self-generated Audio-Visual Pseudo-Fakes

**arXiv ID:** 2604.09110 | [PDF](https://arxiv.org/pdf/2604.09110v1)

**作者:** Zihe Wei `[一作]` (Ocean University of China), Yuezun Li `[通讯]` (Ocean University of China)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自生成音频-视觉伪造样本（AVPF）来提升视频深度伪造检测的泛化能力。

**💡 创新点**

创新点在于引入两种多模态伪造生成策略——音频-视觉自混合（AVSB）和音频-视觉自拼接（AVSS），仅利用真实视频即可模拟真实深度伪造中出现的跨模态和跨模态内不一致性。

**🔧 技术方法**

使用自监督的 AV‑HuBERT 特征提取器作为骨干网络，并在此基础上进行二分类训练；伪造样本通过时域平移、窗口混合、面部掩模、音频梅尔谱平移与拼接等技术生成。

**📊 数据集**

训练数据来源于 VoxCeleb2（5万帧），在 FakeAVCeleb、AV‑Deepfake1M、AVLips、TalkingHeadBench 等四个公开数据集上进行评估。

**📈 对比分析**

与多种基线（如 AVH‑Align、MDS、VFD 等）对比，AVPF 在跨数据集和对抗性后处理场景下均显著提升 AUC/AP，最明显的是对 TalkingHeadBench 的 AUC 由 28.5% 提升至 77.8%，平均提升 6.7% AUC 与 8.0% AP，表明泛化性能显著增强。

**⚠️ 局限性**

局限在于仅做真实性判断，无法定位伪造时点或位置；同时仅基于真实样本生成伪造，可能无法覆盖所有极端伪造技术的特征。

---

## 306. GeoPAS: Geometric Probing for Algorithm Selection in Continuous Black-Box Optimisation

**arXiv ID:** 2604.09095 | [PDF](https://arxiv.org/pdf/2604.09095v1)

**作者:** Jiabao Brad Wang `[一作]` (Duke Kunshan University), Mustafa Misir `[通讯]` (Duke Kunshan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeoPAS 方法，通过在多尺度、多方向、多位置的二维切片上采样目标函数并使用卷积编码器构建问题表示，从而实现连续黑盒优化中的算法选择。

**💡 创新点**

创新点在于使用几何切片而非传统的固定景观描述符，结合可验证掩模、尺度/幅度条件化以及尾部感知的选择器，使得表示在跨问题/跨分布的情境下更具可转移性。

**🔧 技术方法**

采用随机 Sobol 采样、正交 Haar 分布的切片取向、对数均匀尺度分布，卷积网络编码、注意力池化、尾部风险惩罚的双头预测模型，并使用 log(relERT) 作为回归目标。

**📊 数据集**

在 COCO/BBOB 2009 单目标测试套件上，以 12 个算法组成的投资组合，维度 2-10，共 24 个函数、5 个实例，进行评估。

**📈 对比分析**

与 SBS、ELA、DeepELA 等基线比较，GeoPAS 在留实例、随机、留问题三种拆分协议下在平均、众数和 90 分位 relERT 上均显著低于 SBS，尤其在中位数和上尾表现突出；在部分函数组上甚至超过公开 ELA/DeepELA 结果。

**⚠️ 局限性**

局限在于极端尾部事件仍存在，主要集中在少数高维/难度函数；仅在低维（≤10）验证，未考虑更高维或其他测试套件；采样预算未计入 relERT，且静态尾部先验可能不适用于不同投资组合。

---

## 307. Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation

**arXiv ID:** 2604.09088 | [PDF](https://arxiv.org/pdf/2604.09088v1)

**作者:** Yutong Zhang `[一作]` (Beihang University), Yunhong Wang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出Masking Dual Path Distillation（MDPD）框架，在MEML侧网络的基础上进行双向知识蒸馏，训练完成后直接剔除侧网络以加速推理，同时保持甚至提升下游任务性能。

**💡 创新点**

创新点包括：①侧网络与主干在训练期间互为教师进行特征和logits蒸馏；②对多层编码器采用分层Hierarchical Feature-based Distillation，浅层直接模仿，深层通过mask+生成蒸馏；③训练后抛弃侧网络，实现在保持参数/内存优势的同时实现推理加速。

**🔧 技术方法**

采用的技术包括：低秩bottleneck映射实现特征对齐；Feature-based 与 Logits-based 知识蒸馏；mask+生成模块（随机mask与可学习mask token结合生成器 𝒢）；仅更新主干的 LayerNorm 缩放/偏置和侧网络全部参数；使用 CLIP、BERT、ViT 等预训练模型作为主干。

**📊 数据集**

使用的数据集涵盖视觉‑语言任务（Flickr30K、MSCOCO、MSVD、MSR‑VTT、VQA、GQA、RefCOCO/RefCOCO+/RefCOCOg）、视觉/语言单任务（VTAB‑1K、GLUE）以及多模态检索与定位等。

**📈 对比分析**

与多种 PETL（Adapter、LoRA、BitFit、Prompt、VPT 等）和 METL（LST、UniPT、SHERL、HST、LoSA 等）方法对比，MDPD 在保持训练内存下降 70%+ 的同时，在检索、QA、VG 等任务上提升 R@1/Accuracy 约 1–4% 并将推理时间缩短至少 25%（最快可达 44% QPS 提升）。

**⚠️ 局限性**

局限性包括：训练阶段仍需额外显存和前向计算开销；需要精细调节 mask 比例与生成块参数，过大或过小会影响蒸馏效果；目前实验主要在预训练模型与基准任务上验证，跨模型或更大规模任务的泛化性仍待进一步研究。

---

## 308. Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models

**arXiv ID:** 2604.09085 | [PDF](https://arxiv.org/pdf/2604.09085v1)

**作者:** Harry Proshian `[一作]` (Steklov Institute of Mathematics), Ilya Makarov `[通讯]` (AIRI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了三种模型无关的策略，将图结构信息整合到对比自监督学习中，以提高用户属性预测的准确性。

**💡 创新点**

创新点在于通过丰富事件嵌入、对齐客户端表示与图嵌入以及添加结构预文本任务，来整合用户-项目交互图的全局结构信息。

**🔧 技术方法**

使用了对比自监督学习（SSL）方法，包括CoLES和Barlow Twins，并结合图神经网络（GNN）进行特征提取。

**📊 数据集**

使用了四个金融和电子商务数据集，包括性别、年龄、MTS-ML-Cup和一个内部金融数据集。

**📈 对比分析**

通过与基线模型（如CoLES和Barlow Twins）进行比较，结果显示整合图结构信息后，AUC提高了最多2.3%，在不同密度的图中表现出一致的性能提升。

**⚠️ 局限性**

限制在于在极端密度的图中，显式的GNN嵌入可能会由于过平滑效应而失效，因此在这些情况下，辅助相似性损失的效果更佳。

---

## 309. Structuring versus Problematizing: How LLM-based Agents Scaffold Learning in Diagnostic Reasoning

**arXiv ID:** 2604.09158 | [PDF](https://arxiv.org/pdf/2604.09158v1)

**作者:** Fatma Betül Güreş `[一作]` (ETH Zürich), Tanja Käser `[通讯]` (EPFL)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于情景的学习系统 PharmaSim Switch，并在其中实现了两种基于 LLM 的教学代理的结构化与问题化支架。

**💡 创新点**

创新点在于将学习分析与大语言模型结合，系统化比较两种支架方式，并在药学技术员培训场景中验证其教学效果。

**🔧 技术方法**

使用了 GPT‑4o 作为药师角色的 LLM 代理，并配合学习分析模块实时监测学生学习轨迹。

**📊 数据集**

使用了自定义的药师案例数据集（四个情景 A、B、C1、C2）以及 63 名药学技术员学生的交互日志。

**📈 对比分析**

采用两组实验设计和混合线性模型比较支架效果，结果显示两种支架在诊断策略表现上无显著差异，情景复杂度是主要影响因素；结构化支架提升了准确性，问题化支架提升了创造性参与。

**⚠️ 局限性**

局限在于仅在瑞士一所职业学校进行，未考察长期学习效果，LLM 生成的支架覆盖不均且对人际关系策略支持不足，缺乏跨领域与跨人群的验证。

---

## 310. Hagenberg Risk Management Process (Part 3): Operationalization, Probabilities, and Causal Analysis

**arXiv ID:** 2604.09153 | [PDF](https://arxiv.org/pdf/2604.09153v1)

**作者:** Eckehard Hermann `[一作]` (University of Applied Sciences Upper Austria), Harald Lampesberger `[通讯]` (University of Applied Sciences Upper Austria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了从 Bowtie 风险模型到实时可观测概率决策网络的端到端工具链，包含 Realtime Risk Studio、Probability Capture 和因果分析模块；

**💡 创新点**

在 Bowtie 结构基础上显式引入安全状态、自动转换为 DAG 并加入 Activation Node、结合专家概率捕获与噪声分析、在同一模型上实现因果干预搜索；

**🔧 技术方法**

使用 Bayesian 网络/有向无环图、概率推理、Do-算子/因果推断、专家估计聚合（Beta、加权平均等）、REST API 监测接口和可视化编辑器；

**📊 数据集**

以即时支付网关为案例，利用内部监控指标（队列、延迟、重试、回滚状态等）作为证据，辅以专家评估；

**📈 对比分析**

未进行定量性能对比，主要通过案例演示展示后验分布与干预效果，相较传统 Bowtie 提供了结构化、概率化的实时监控与干预评估；

**⚠️ 局限性**

模型质量仍受结构假设和专家估计影响，缺乏经验校准；未处理循环因果，且仅通过单一案例演示，缺乏大规模验证；估计方法多样性需结合业务选择。

---

## 311. Score-Driven Rating System for Sports

**arXiv ID:** 2604.09143 | [PDF](https://arxiv.org/pdf/2604.09143v1)

**作者:** Vladimír Holý `[一作]` (Prague University of Economics and Business), Michal Černý `[通讯]` (Prague University of Economics and Business)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文提出一种基于分布对数似然梯度的分数驱动评分系统，能够推广传统 Elo 系统以适应任意形式的比赛结果；

**💡 创新点**

创新点在于将分数（log‑likelihood 的梯度）作为更新规则，证明其零期望、和为零、递减性等性质，并展示其与 Elo 的对应关系以及对多种结果（得分差、三分状态、完整排名）的统一建模；

**🔧 技术方法**

主要技术包括概率模型的对数似然梯度推导、严格对数凹性假设、对数似然的零期望与积分恒等式，以及对回归式的解析；

**📊 数据集**

论文未给出具体数据集，侧重理论推导与示例说明（如 Skellam、ordered logit、Plackett‑Luce 分布的实例），主要以模拟路径展示评分动态；

**📈 对比分析**

在实验上仅通过仿真对比 Elo 与分数驱动模型，显示后者在多样化结果下保持无通胀/通缩，并可随真技能漂移而趋向回归；实际预测性能未在真实赛果上评估；

**⚠️ 局限性**

局限性包括：需已知正确的概率分布与对数凹性假设、对真实技能的收敛性分析尚未完成、在实际应用中需考虑对手配对方式与模型选择等因素。

---

## 312. Geometry Reinforced Efficient Attention Tuning Equipped with Normals for Robust Stereo Matching

**arXiv ID:** 2604.09142 | [PDF](https://arxiv.org/pdf/2604.09142v1)

**作者:** Jiahao Li `[一作]` (City University of Hong Kong), Jianping Wang `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了GREATEN框架，利用立体图像与表面法向量的门控上下文-几何融合，并通过稀疏注意力机制实现零样本Syn-to-Real立体匹配的显著提升。

**💡 创新点**

核心创新包括（1）门控上下文-几何融合模块，可自适应抑制纹理歧义并引入域不变的法向信息；（2）Specular-Transparent Augmentation，用于扰动纹理以增强融合鲁棒性；（3）稀疏空间、双匹配与体积注意力，降低计算和显存消耗。

**🔧 技术方法**

采用迭代式Stereo（如IGEV、Selective-IGEV）+ MobileNetV2与U-Net特征编码器；门控遮罩网络（GMNet）+ FusionNet；Specular/Transparent Augmentation；稀疏空间注意力（SSA）、稀疏双匹配注意力（SDMA）和稀疏体积注意力（SVA）；以及ConvGRU细化。

**📊 数据集**

训练使用SceneFlow；混合合成数据（FallingThings、TartanAir、CREStereo、Sintel、CARLA-HRVS、Virtual KITTI2）提升泛化；测试覆盖ETH3D、Middlebury、KITTI-2012/2015、Booster以及自采集的真实数据。

**📈 对比分析**

在五大真实基准上与RAFT、RAFT-Stereo、Selective-IGEV、Monster、DEFOM等基线比较，零样本情况下GREATEN-IGEV在KITTI、Booster等指标上分别提升约30%–40%；在ETH3D和Middlebury上提升20%+；在RVC挑战中取得所有评测的第一名。

**⚠️ 局限性**

对法向图质量高度依赖；需先行生成高质量法向估计；在极端光照或非Lambertian区域仍可能受纹理误导；在极高分辨率时显存消耗较大。

---

## 313. The Role of LLMs in Collaborative Software Design

**arXiv ID:** 2604.09120 | [PDF](https://arxiv.org/pdf/2604.09120v1)

**作者:** Victoria Jackson `[一作]` (University of Southampton), André van der Hoek `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在实验室中让18对软件专业人员在90分钟内使用可自定义的ChatGPT 3.5 Turbo接口协同设计校园自行车泊车应用，记录对话、视频与访谈数据并进行定性分析

**💡 创新点**

首次系统揭示LLM在协作设计中的四种角色（无作用、信息源、生成器、生产者）以及共享与独立实例使用模式对设计过程与成果的影响

**🔧 技术方法**

自研聊天包装器（ChatGPT API）、Zoom远程会议、协同设计工具（Google Docs、Lucidchart）以及视频录制与文本分析工具

**📊 数据集**

18对设计任务的原始数据：PRD、LLM交互日志、会议视频、访谈录音与转录文本

**📈 对比分析**

通过归纳性编码与主题分析比较不同使用模式与角色的出现频率与影响，未给出客观性能指标，主要聚焦于使用模式与设计过程的质性洞察

**⚠️ 局限性**

受限于实验室绿地任务、样本规模小、缺乏真实工业背景，未评估设计质量或长期效果，仅提供探索性发现

---

## 314. CLIP-Inspector: Model-Level Backdoor Detection for Prompt-Tuned CLIP via OOD Trigger Inversion

**arXiv ID:** 2604.09101 | [PDF](https://arxiv.org/pdf/2604.09101v1)

**作者:** Akshit Jindal `[一作]` (Indian Institute of Technology Delhi), Vikram Goyal `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种针对Prompt‑tuned CLIP模型的后门检测方法CLIP‑Inspector，利用无标签OOD图像恢复潜在触发器并判定模型是否被植入后门；

**💡 创新点**

创新点在于将触发器逆向重建与行为驱动的异常得分相结合，既适用于保持编码器不变的Prompt‑tuning后门，又能在单个训练周期内高效完成；

**🔧 技术方法**

采用了基于margin的logit目标进行触发器优化、白盒梯度更新、ASR与优化损失的z‑score聚合评分，并利用CLIP的文本-图像嵌入机制；

**📊 数据集**

在十个标准视觉识别数据集（ImageNet、Caltech101、OxfordPets、Flowers102、Food101、FGVC、SUN397、DTD、EuroSAT、UCF101）上进行实验；

**📈 对比分析**

与改造后的Neural Cleanse和Pixel‑Backdoor基线相比，CI在单个epoch下对1000张OOD图像实现94%的检测准确率（AUROC 0.973），并在多数攻击场景下能准确定位目标类别；

**⚠️ 局限性**

局限性包括在细粒度数据集（如FGVC、EuroSAT）上性能下降，以及对语义触发器和极端稀疏触发器的检测效果尚未验证。

---

## 315. Physically Grounded 3D Generative Reconstruction under Hand Occlusion using Proprioception and Multi-Contact Touch

**arXiv ID:** 2604.09100 | [PDF](https://arxiv.org/pdf/2604.09100v1)

**作者:** Gabriele Mario Caddeo `[一作]` (Istituto Italiano di Tecnologia), Lorenzo Natale `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在手部遮挡严重的单视图场景中，融合RGB图像、手部关节姿态（关节角度）和多点触觉信息，利用姿态对齐的SDF潜在空间和条件流匹配模型，实现物理可行、尺度一致的3D物体重建和姿态估计。

**💡 创新点**

创新点在于将关节姿态和多点触觉作为物理约束注入生成模型，使用姿态对齐的SDF与Structure‑VAE和流匹配网络相结合，并在训练与推理阶段通过接触一致性和非穿透损失实现对手掌的物理约束，显著提升在高遮挡下的重建质量。

**🔧 技术方法**

采用Structure‑VAE生成SDF潜在空间，使用条件流匹配（Flow Transformer）进行生成，加入物理约束（接触一致性、非穿透损失）以及解码器引导的推理；后处理阶段可采用SLat或Sam3D进行网格细化与纹理化。

**📊 数据集**

训练与评估使用3D‑FUTURE、HSSD、ABO、ObjaverseXL、Google Scanned Objects等大型3D模型数据集；仿真抓取场景基于YCB；真实机器人实验在与YCB对应的真实物体上进行，使用自制的抓取场景数据。

**📈 对比分析**

通过与Amodal3R、SAM3D两种主流遮挡重建方法比较，实验表明在高遮挡层次下该方法在Chamfer Distance、Normal Consistency、F@0.02、Voxel IoU、EMD等指标上均优于基线；姿态估计指标3D IoU、ICP‑Rot、ADD‑S在所有遮挡级别上均显著更好；在真实机器人实验中也可与SAM3D竞争，且在物理可行性方面更具优势。

**⚠️ 局限性**

局限性包括：需要精准的相机‑手外参、正向运动学和遮挡掩模；固定64³网格限制了细节保留，薄结构与细小凹槽可能被忽略；对手部模型、传感器布局的泛化受限，校准误差会显著影响重建质量。

---

## 316. Few-Shot Personalized Age Estimation

**arXiv ID:** 2604.09125 | [PDF](https://arxiv.org/pdf/2604.09125v1)

**作者:** Jakub Paplhám `[一作]` (Czech Technical University), Artem Moroz `[通讯]` (Czech Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了N-shot个性化年龄估计任务，并构建了OpenPAE公开基准，提供身份不重叠的数据划分与严格评估协议。

**💡 创新点**

创新点在于（1）首次公开多参考（N>1）的个性化年龄估计基准；（2）通过不同评估模式（跨域与同域）区分真实个性化与域适应；（3）设计了多种基线（算术偏移、贝叶斯线性回归、注意力神经过程），形成对比基准。

**🔧 技术方法**

采用视觉Transformer（ViT-B/16）提取特征，配合高斯负对数似然训练；在此基础上实现算术偏移、闭式贝叶斯线性回归（BLR）和两种注意力结构（全局Attn-G、空间Attn-S）以及Pair-Avg。

**📊 数据集**

使用四个公开人脸年龄数据集：CSFD-1.6M（主训练集）、MORPH、AgeDB、KANFace；其中CSFD用于训练，MORPH、AgeDB、KANFace用于跨域测试。

**📈 对比分析**

实验结果表明：随着参考数量N增大，个性化方法整体MAE下降；注意力神经过程（Attn-S）在所有数据集上均优于算术偏移和BLR，且在N≥10时可将MAE提升约0.5–1.0年；对比不同ID参考的消融验证显示，域+年龄先验提升约1–2年，真正个性化可再提升约0.4–0.7年。

**⚠️ 局限性**

局限性包括：仅使用同一身份的参考图片，缺乏对不同身份间迁移的研究；对时间跨度大于10年的参考表现不佳，提示模型对时间分布的鲁棒性不足；依赖公开数据集的标注质量与分布可能影响结果。

---

## 317. Prototype-Regularized Federated Learning for Cross-Domain Aspect Sentiment Triplet Extraction

**arXiv ID:** 2604.09123 | [PDF](https://arxiv.org/pdf/2604.09123v1)

**作者:** Zongming Cai `[一作]` (Guizhou University), Hankz Hankui Zhuo `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于类别级原型的跨域联邦学习框架 PCD-SpanProto，专门用于 Aspect Sentiment Triplet Extraction，在不共享原始数据的前提下实现跨域知识共享。

**💡 创新点**

创新点包括：①仅上传类别级原型而非完整模型参数，显著降低通信成本；②引入对齐与分离的对比正则化，提升跨域原型一致性；③基于本地验证 F1 的性能感知权重进行原型聚合，增强跨域适应性。

**🔧 技术方法**

采用 BERT‑Encoder 与 STAGE span‑tagging 模型，配合原型对比学习、动量更新、性能感知聚合以及传统联邦学习框架。

**📊 数据集**

使用四个 ASTE 公开基准数据集：14Lap（笔记本）和 14Res、15Res、16Res（餐厅）。

**📈 对比分析**

与多种 pipeline、end‑to‑end、单域、合并数据等基线对比，取得所有四个数据集 F1 分别提升 2–5 分，成为目前最高性能，同时将通信量从 110M 参数降至约 3.2K。

**⚠️ 局限性**

局限性在于原型仅按标签聚合，可能无法捕捉细粒度语义差异；对更大规模、多语言或极端域差异的适用性尚需进一步验证；联邦学习仍受客户端差异和同步频率影响。

---

## 318. Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition

**arXiv ID:** 2604.09121 | [PDF](https://arxiv.org/pdf/2604.09121v1)

**作者:** Peng Wang `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了基于LLM的交互式语音识别框架，并提出了句级语义误差率S^2ER评价指标。

**💡 创新点**

①S^2ER通过LLM判定语义一致性，弥补传统WER对语义重要性忽视；②采用LLM驱动的意图路由与链式推理实现多轮语音交互纠错。

**🔧 技术方法**

使用LLM-as-a-Judge、链式推理（CoT）、意图路由、零声克隆TTS、自动化用户模拟器、Qwen3 ASR/LLM和Index‑TTS等技术。

**📊 数据集**

在GigaSpeech、WenetSpeech和ASRU2019代码切换测试集上进行实验。

**📈 对比分析**

通过与传统WER、CER等指标对照，并在10轮交互后S^2ER降至≈1%（GigaSpeech1.08%，WenetSpeech1.11%，ASRU2019 0.82%），显示在前两轮即可实现显著提升。

**⚠️ 局限性**

依赖大型LLM与ASR模型，算力需求高；自动化模拟未完全覆盖真实用户纠错行为；最终性能受底层ASR误识率限制。

---

## 319. Hybrid Cold-Start Recommender System for Closure Model Selection in Multiphase Flow Simulations

**arXiv ID:** 2604.09112 | [PDF](https://arxiv.org/pdf/2604.09112v1)

**作者:** S. Hänsch `[一作]` (Helmholtz-Zentrum Dresden-Rossendorf e.V.), P. Kordík `[通讯]` (Czech Technical University in Prague)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对多相 CFD 关节模型选择的混合推荐系统，解决了冷启动下的模型推荐问题。

**💡 创新点**

创新点在于将闭环模型选择重新表述为冷启动推荐问题，并将案例元数据驱动的相似性 (k‑NN) 与协同过滤的矩阵完成 (Gaussian Copula) 结合，兼顾了新案例无历史数据与稀疏性能矩阵的利用。

**🔧 技术方法**

主要技术包括基于案例特征的 k‑NN 相似度计算、gcimpute 的 Gaussian Copula 矩阵完成、嵌套实验级交叉验证、并使用 RR@k、MRR 和 regret 等指标评估排名与风险。

**📊 数据集**

实验使用了 13,600 条 CFD 仿真结果，覆盖 136 个验证案例（来自 17 个气泡流实验）与 100 种闭环模型组合，特征包含 27 个分类和 6 个连续变量。

**📈 对比分析**

与基于观测的热门模型、矩阵完成后的热门模型以及专家设计的通用模型相比，混合推荐器在所有稀疏度（0.25–0.90）下都取得更高的 MRR、RR@k 及更低的 regret，证明其在冷启动环境中的优越性能。

**⚠️ 局限性**

限制包括仅覆盖闭环模型空间的子集、将数值不稳定性与物理误差合并为零性能、假设所有案例特征已知、以及在面临超出训练特征空间的新流动场景时可能出现性能退化。

---

## 320. Detecting Diffusion-generated Images via Dynamic Assembly ForestsDetecting Diffusion-generated Images via Dynamic Assembly Forests

**arXiv ID:** 2604.09106 | [PDF](https://arxiv.org/pdf/2604.09106v1)

**作者:** Mengxin Fu `[一作]` (Ocean University of China), Yuezun Li `[通讯]` (Ocean University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于传统机器学习的动态组装森林模型，用于检测扩散模型生成的图像。

**💡 创新点**

创新点包括：①动态组装策略实现批量式森林训练；②任务特定的多尺度局部特征提取（HOG + 频域统计）。

**🔧 技术方法**

采用深度森林、动态装配、HOG、局部频率特征、滑动窗口多尺度拼接等技术；并在CPU上完成。

**📊 数据集**

使用 DiffusionForensics（LSUN‑B 与 ImageNet 子集）、GenImage、Chameleon 等公开数据集进行实验。

**📈 对比分析**

与 CNNDet、AEROBLADE、DIRE 等基于 DNN 的检测器对比，在 ACC / AUC 上与 DNN 相当甚至更优，同时参数量更少、计算成本更低，能在无 GPU 的 CPU 环境下部署。

**⚠️ 局限性**

局限在于特征提取阶段耗时较高，难以在更大规模数据上实时运行；与深度特征融合效果有限，导致跨数据集泛化受限。

---

## 321. Off-the-shelf Vision Models Benefit Image Manipulation Localization

**arXiv ID:** 2604.09096 | [PDF](https://arxiv.org/pdf/2604.09096v1)

**作者:** Zhengxuan Zhang `[一作]` (Ocean University of China), Yuezun Li `[通讯]` (Ocean University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可插拔的适配器ReVi，将通用视觉模型重新用于图像篡改定位（IML）任务

**💡 创新点**

创新点在于将稳健主成分分析（RPCA）原理与深度网络相结合，分离语义冗余与篡改特征，并利用预训练语义先验提升IML性能

**🔧 技术方法**

使用LoRA进行参数高效微调、RPCA式分解的SKA与MKE模块、梯度增强损失、以及Transformer自注意力层的插拔

**📊 数据集**

在PSCC进行预训练，评估于CASIA、Columbia、Coverage、NIST16、IMD20等公开数据集

**📈 对比分析**

与PSCC‑Net、TruFor、EVP、IML‑ViT、ACBG、Sparse‑ViT、MPC、Mesorch等专用IML方法比较，ReVi在大多数数据集上均取得最高或接近最高的F1/AUC，尤其在Fine‑tuned协议中领先

**⚠️ 局限性**

受限于依赖强大预训练模型，若使用低容量模型或极低级特征，ReVi效果可能下降

---

## 322. Few-Shot Contrastive Adaptation for Audio Abuse Detection in Low-Resource Indic Languages

**arXiv ID:** 2604.09094 | [PDF](https://arxiv.org/pdf/2604.09094v1)

**作者:** Aditya Narayan Sankaran `[一作]` (Institut Polytechnique de Paris), Noel Crespi `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了使用CLAP模型对多语言音频直接检测侮辱性语言，采用少量标注样本进行监督对比学习适配。

**💡 创新点**

首次将跨模态对比预训练的CLAP直接应用于音频侮辱检测，并提出仅更新投影层的轻量级对比适配方法。

**🔧 技术方法**

使用CLAP音频-文本对齐、监督对比损失、投影层适配、SVM/ANN下游分类器等技术。

**📊 数据集**

使用印度十种印地语系语言的ADIMA音频数据集。

**📈 对比分析**

通过与全监督基线和零样本提示进行对比，投影层适配在5–25 shot下的macro‑F1与全监督差距≤4点，最高达83.65（5-shot）超过基线。

**⚠️ 局限性**

仅在单一多语言基准上验证；零样本效果有限；支持集抽样波动；仅评估两种下游模型和两种适配策略；未进行公平性分析。

---

## 323. DIAURec: Dual-Intent Space Representation Optimization for Recommendation

**arXiv ID:** 2604.09087 | [PDF](https://arxiv.org/pdf/2604.09087v1)

**作者:** Yu Zhang `[一作]` (Anhui University), Lei Sang `[通讯]` (Anhui University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 DIAURec 框架，联合原型意图与分布意图重构用户/物品表示，并通过对齐、均匀性、多级匹配与正则化等技术实现跨空间协同优化，从而提升推荐效果。

**💡 创新点**

创新点在于：① 设计双意图空间（原型+分布）对表示进行重构；② 将对齐、均匀性、粗细粒度匹配及内空间/交互正则化相结合，实现跨空间的统一优化；③ 只利用 LLM 生成语义表示，避免频繁调用 LLM，降低计算成本。

**🔧 技术方法**

采用图卷积网络进行协同编码，利用 LLM 生成语义表示，构建 vMF 分布意图，使用对齐与均匀性目标，加入粗细粒度匹配策略，并通过内空间正则化与交互正则化稳定训练，实现多任务联合学习。

**📊 数据集**

在三大稀疏交互数据集 Amazon‑book、Yelp 与 Steam 上进行实验。

**📈 对比分析**

与 15 种基线（LightGCN、SimGCL、DCCF、BIGCF、SimGCF、KAR、LLMRec、RLMRec、AlphaRec、IRLLRec、DirectAU、MAWU、GraphAU、CARec、SIURec）在 Recall@N、NDCG@N 上对比，DIAURec 在 Recall@20/NDCG@20 上均领先最强基线 10%–15% 以上，性能显著提升。

**⚠️ 局限性**

主要限制包括：对 LLM 生成语义表示的依赖仍带来预处理开销；对超参数 λ1、λ2 敏感，过低会导致表示崩溃；目前仅适用于静态推荐，尚未考虑用户意图随时间变化的动态推荐场景。

---

## 324. Scrutinizing Real-life Configurations of Random Access Procedures in Cellular Networks

**arXiv ID:** 2604.09077 | [PDF](https://arxiv.org/pdf/2604.09077v1)

**作者:** Joris Belder `[一作]` (ETH Zurich), Fernando Kuipers `[通讯]` (Delft University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对4G网络中随机接入配置进行了实测、分析与仿真，发现现有配置往往不匹配部署场景且邻近小区使用相同配置，导致碰撞率高、连接延时长。

**💡 创新点**

首次公开真实网络随机接入配置的数据集，并通过仿真证明简单调整配置即可显著降低碰撞和延时。

**🔧 技术方法**

使用移动设备捕获SIB2广播信息（Python+QCSuper+Tshark），以及NS‑3+LENA+进行随机接入仿真。

**📊 数据集**

包含112,806条基站广播信息，来自三国三大运营商的9个网络运营商。

**📈 对比分析**

通过NS‑3仿真比较“同一配置”与“不同配置”两种方案，结果显示碰撞率平均下降43%（最高61%），连接延时平均下降11%（最高42%）。

**⚠️ 局限性**

仿真仅覆盖4G、LENA+仅支持预制格式0、PRACH‑ConfigIndex偶数、未考虑频率重用与真实网络动态，且缺乏5G/NSA场景验证。

---

## 325. Deep Light Pollution Removal in Night Cityscape Photographs

**arXiv ID:** 2604.09145 | [PDF](https://arxiv.org/pdf/2604.09145v1)

**作者:** Hao Wang `[一作]` (Shandong University), Baoqing Sun `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对夜间城市景观照片的光污染去除方法，结合物理驱动的光扩散模型（包括各向异性光扩散和天际线诱导的天空辉光）以及基于大规模生成模型的细化策略。

**💡 创新点**

创新点包括：
- 设计了新的物理光污染形成模型，显式考虑了光源的方向性扩散（ALSF）和隐藏光源产生的天空辉光；
- 通过合成数据与大模型先验相结合的训练框架，克服了真实配对数据匮乏的难题；
- 构建了首个光污染去除专用数据集，并公开发布；
- 对比实验和用户研究验证了模型在视觉质量和定量指标上的优势。

**🔧 技术方法**

技术手段包括：
- 基于APSF的各向异性光扩散函数ALSF与天空辉光层的物理模拟；
- 合成污染图像生成流水线（随机化PSF参数、天空辉光模拟等）；
- 采用Flux.1-Kontext扩散模型，使用Q-LoRA（4‑bit量化 + LoRA 512）进行高效微调；
- Flow Matching损失配合VAE编码实现噪声引导的图像恢复；
- 对比分析与现有夜间增强、去雾、去眩光、HDR等基线方法。

**📊 数据集**

使用的数据集包括：
- 521张高质量无光污染夜景图（约200张天文、300张城市）
- 在每张原图上合成5种光污染版本，生成约2500对（污染、干净）数据
- 100张真实严重光污染的城市夜景作为测试集
- 所有数据集将公开发布，方便后续研究。

**📈 对比分析**

对比方法包括：Liu 2021 LPR、Jin 2023 dehaze、Cong 2024 dehaze、Lin 2025 dehaze、Jin 2022 LES、Dille 2024 HDR、Dai 2024 deflare 等。实验结果显示：
- PSNR 26.17 dB、SSIM 0.8517，明显优于Liu 20.61/0.8314，超过所有基线；
- ablation 研究证明加入ALSF和天空辉光可分别提升约1–2 dB，完整模型进一步提升约4 dB；
- 用户研究中获得最佳平均排名，并在多数场景中占据第一名。

**⚠️ 局限性**

局限性包括：
- 未加入色彩调校，导致恢复图像在色彩饱和度和对比度上不如专门的HDR方法；
- 使用扩散模型推理速度不够实时，限制了在视频或移动端的即时应用；
- 目前仅处理静态图像，未考虑动态场景的连续性和时域一致性。

---

## 326. FaceLiVTv2: An Improved Hybrid Architecture for Efficient Mobile Face Recognition

**arXiv ID:** 2604.09127 | [PDF](https://arxiv.org/pdf/2604.09127v1)

**作者:** Novendra Setyawan `[一作]` (National Formosa University), Jun-Wei Hsieh `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了 FaceLiVTv2，一种轻量级的 CNN‑Transformer 混合架构，用于边缘和移动端的实时人脸识别。

**💡 创新点**

创新点在于将多头线性注意力重构为 Lite MHLA，结合重参数化本地混合器 RepMix 与全局注意力，并采用 GDConv 自适应空间聚合头，显著提升速度与准确率。

**🔧 技术方法**

使用了多头线性注意力、重参数化卷积、全局深度可分离卷积、AdaFace/CosFace 等角度损失，以及 Glint360K 与 TinyFace 训练策略。

**📊 数据集**

训练集为 Glint360K，评估基准包括 LFW、CA‑LFW、CP‑LFW、CFP‑FP、AgeDB‑30、IJB‑B/C 以及低分辨率 TinyFace。

**📈 对比分析**

与现有轻量级模型（GhostFaceNet、EdgeFace、SwiftFaceFormer、KANFace 等）在参数、FLOPs、移动端延迟等指标上对比，FaceLiVTv2 在 iPhone 15 Pro 上实现 22%–30% 延迟下降，同时保持或提升准确率。

**⚠️ 局限性**

未对极端遮挡、低照度或跨民族公平性进行评估，缺少对遮挡严重或种族差异的鲁棒性实验。

---

## 327. Responsive Distribution of G-normal Random Variables

**arXiv ID:** 2604.09103 | [PDF](https://arxiv.org/pdf/2604.09103v1)

**作者:** Ziting Pei `[一作]` (Suzhou University of Science and Technology), Xiaotao Zheng `[通讯]` (Soochow University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了G-正态随机变量的响应分布概念，并设计了耦合的后向-前向三叉树方法，用以数值求解G期望和构造相应的分布。

**💡 创新点**

创新点在于将G-期望的非线性映射到一个基于最优控制的线性期望，并通过三叉树实现响应分布的高精度采样，同时提供了严格的收敛性证明。

**🔧 技术方法**

使用技术包括随机最优控制、G-热方程的Barles–Souganidis一致性框架、显式有限差分、以及弱收敛到非线性Fokker–Planck方程的证明。

**📊 数据集**

实验中使用的参数为σ²=0.04与σ²=1.0的波动不确定区间，时间终点T=1，网格细化至最多800步，并在不同测度函数下进行采样。

**📈 对比分析**

与传统Monte Carlo或单纯数值 PDE 方法比较，耦合树法在保持稳定性的同时，取得了约二阶收敛率，且能够直观绘制非凸、非凹测度下的响应分布。

**⚠️ 局限性**

局限性包括仅在一维情形下证明，三叉树在高维扩展受限；以及对高度非光滑测度函数时可能需要更细网格，计算成本上升。

---

## 328. Synthesizing real-world distributions from high-dimensional Gaussian Noise with Fully Connected Neural Network

**arXiv ID:** 2604.09091 | [PDF](https://arxiv.org/pdf/2604.09091v1)

**作者:** Joanna Komorniczak `[一作]` `[通讯]` (Wrocław University of Science and Technology), Joanna Komorniczak (Wrocław University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了DiMSO（Distribution Mapping with Shuffled Optimization）方法，利用全连接神经网络和随机化损失函数将高维高斯噪声映射为目标真实分布，从而高效生成合成表格数据。

**💡 创新点**

创新点在于：①采用随机化绝对误差（RAE）以及随机样本顺序的损失函数，使模型在不使用潜在空间编码的情况下快速逼近目标分布；②通过PCA降维与逆变换实现隐私增强；③在传统生成模型与深度生成模型之间构建了一种计算成本显著更低、可解释性更强的生成方案。

**🔧 技术方法**

核心技术包括：全连接前馈神经网络（3层隐藏层，ReLU激活，Adam优化）；三种损失函数（RAE、Wasserstein距离、WC复合损失）；PCA降维与逆变换；与SMOTE、SVMSMOTE、Gaussian Copula、TVAE、CTGAN等生成器的对比实验。

**📊 数据集**

实验使用了25个公开表格数据集（样本数163–3772，特征数4–44，类别数1–9），涵盖高度不平衡与多类情况，所有数据均在归一化后进行处理。

**📈 对比分析**

比较方法：通过WD、MMD和MeanNN三种分布相似度指标、5折交叉验证下的Balanced Accuracy（BAC）评价分类性能，以及生成时间测评。结果显示：DiMSO（尤其RAE损失）在MMD和MeanNN指标上与SMOTE持平或略优，显著优于TVAE/CTGAN；在分类任务中，DiMSO生成的样本往往能提升BAC，尤其在高不平衡数据上表现优异；生成时间上，DiMSO比深度生成模型快数百倍，达到了时间效率上的领先。

**⚠️ 局限性**

限制与不足：①仅处理归一化连续特征，未对类别特征进行预/后处理；②RAE损失在迭代过多时易出现过拟合，需要进一步研究早停或正则化；③未系统评估生成样本规模与类别不平衡对性能的影响；④缺少正式的差分隐私或其他隐私泄露风险评估。

---

## 329. DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation

**arXiv ID:** 2604.09089 | [PDF](https://arxiv.org/pdf/2604.09089v1)

**作者:** Li Huang `[一作]` (Chongqing University), Meng Yan `[通讯]` (Chongqing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种面向代码生成模型的安全强化框架 DeepGuard，利用多层注意力聚合和多目标参数高效微调来提升生成代码的安全性并保持功能正确性。

**💡 创新点**

创新点在于：①识别并利用 Transformer 中中上层分布的安全相关信号，突破单层（最终层）瓶颈；②引入注意力机制对多层隐藏状态进行自适应聚合；③构建多目标训练目标（安全对比、流畅性、分布稳定性），并在推理时通过提示条件的安全偏置实现轻量级引导。

**🔧 技术方法**

技术主要包括：LoRA 参数高效微调；多层隐藏状态聚合（多头注意力）；安全分析器（小型 MLP）；多目标损失（对比损失、语言建模损失、KL 正则）；推理时的安全偏置加速（一次性偏置）。

**📊 数据集**

使用了由可验证功能等价的漏洞/安全代码对组成的训练集，覆盖常见 CWE（如注入、缓冲区溢出等），以及基准测试集（Python、C/C++ 的函数级安全评测数据）。

**📈 对比分析**

与多种基线（SVEN、SafeCoder、CoSec、CodeGuard+、prompt‑based）对比，DeepGuard 在 5 种主流代码 LLM 上平均提升 sec‑pass@1（安全且正确）约 11.9%，在保持或略高的 pass@1（功能正确性）同时，显示出对留出的漏洞类型的良好泛化能力。

**⚠️ 局限性**

局限性包括：仅在函数级、Python/C++ 评测环境验证；需要可获取功能等价的漏洞/安全对来训练，构造成本高；固定层聚合深度可能不适用于所有模型；只能在有内部状态访问权限的模型上使用，无法直接适用于 API‑only 或闭源模型。

---

## 330. Hierarchical Alignment: Enforcing Hierarchical Instruction-Following in LLMs through Logical Consistency

**arXiv ID:** 2604.09075 | [PDF](https://arxiv.org/pdf/2604.09075v1)

**作者:** Shu Yang `[一作]` (King Abdullah University of Science and Technology), Wenda Li `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型语言模型在多源指令混合环境下的指令冲突问题，提出一种神经-符号层次对齐框架（NSHA），在推理时通过约束求解器求解最优指令集合，并在训练时将求解结果蒸馏进模型参数，实现层级指令遵循。

**💡 创新点**

创新点包括：1) 以MaxSMT形式将层级关系和语义冲突转化为可求解的约束；2) 结合神经冲突检测器和符号求解器的推理流程；3) 通过自动构造的层级一致性损失（HCAL）实现训练时的符号知识蒸馏，避免推理时依赖外部求解器。

**🔧 技术方法**

技术方法包括：指令atomizer、跨编码器NLI冲突检测、Z3 MaxSMT求解、符号约束驱动生成、对齐训练（SFT/DPO/HCAL）、基于长度归一化的偏好损失和语义损失。

**📊 数据集**

数据集：使用Alpaca原始示例自动生成对齐与冲突语境，并在此基础上构造偏好对；评估使用IHEval benchmark（Rule Following、Task Execution、Tool Use、Safety）及其单/多轮、对齐/冲突子集。

**📈 对比分析**

与基线（Qwen3-4B、Llama3.1-8B）及其CoT、NS、NSHA-SFT、NSHA-DPO、NSHA-HCAL进行对比；实验显示NSHA-DPO在冲突场景下显著提升（如Qwen Rule‑Follow conflict从26.6提升至47.6，Llama 20.3→41.9），同时保持或略低于基线在无冲突场景的表现；在工具使用和安全任务上亦取得相对优异的鲁棒性。

**⚠️ 局限性**

局限性：1) 推理时仍需外部求解器导致延迟；2) 目前仅支持三级层级，未覆盖更细粒度权限；3) 对长时序对齐效果仍有限，需进一步研究多轮持续一致性；4) 训练集多基于人工合成，可能缺乏真实世界多样性。

---

## 331. EdgeFlow: Fast Cold Starts for LLMs on Mobile Devices

**arXiv ID:** 2604.09083 | [PDF](https://arxiv.org/pdf/2604.09083v1)

**作者:** Yongsheng Yan `[一作]` (Fudan University), Yangfan Zhou `[通讯]` (Fudan University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个针对移动设备的LLM推理框架，通过自适应调整模型参数的量化精度来显著降低冷启动延迟；

**💡 创新点**

提出了三项协同技术：① 以NPU约束为前提的自适应量化算法；② SIMD友好的权重打包格式及高效解包；③ 细粒度CPU–NPU协作的协同管线；这三者共同实现了在保持精度的前提下大幅压缩闪存带宽使用；

**🔧 技术方法**

使用了NPU-aware adaptive quantization、SIMD-friendly packing format（配合SIMD解包算法）和 synergistic granular pipeline；在实现层面，采用了QNN SDK的量化与编译工具链，并对QNN执行图格式进行逆向，支持动态加载权重；

**📊 数据集**

模型：Llama3 8B、Mistral 7B、Phi3 3.8B、Qwen1.5 1.8B；数据集：LAMBADA、WinoGrande、OpenBookQA、MMLU、HellaSwag（不同长度提示）；

**📈 对比分析**

与开源框架 llama.cpp、MNN 以及 NPU 加速框架 llm.npu 进行比较，主要评估指标为冷启动第一令牌时间（TTFT）与推理精度。结果显示，本文框架在保持相同或更好精度的前提下，TTFT 低于 llama.cpp/MNN 4.07×，低于 llm.npu 1.55×；

**⚠️ 局限性**

局限性：目前仅在 Qualcomm Hexagon NPU 上实现与验证；对其他 NPU 或 GPU 的适配需要额外工程；逆向 QNN 执行图导致实现复杂度较高；仅针对中小规模 LLM 进行评测，极大模型的性能尚未验证；

---

## 332. EthicMind: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue

**arXiv ID:** 2604.09265 | [PDF](https://arxiv.org/pdf/2604.09265v1)

**作者:** Jiawen Deng `[一作]` (University of Electronic Science and Technology of China), Fuji Ren `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出EthicMind框架，在多轮对话中实时评估伦理风险与用户情绪，并规划响应策略，实现伦理与情感的统一。

**💡 创新点**

创新点在于将伦理-情感对齐建模为每轮决策过程，并通过分阶段分析、规划与生成实现无须额外训练的实时调节。

**🔧 技术方法**

采用模块化LLM推理，包含风险分析器、情绪识别器和响应规划器，利用现有GPT‑4o/ Llama‑3.3‑70B实现。

**📊 数据集**

使用公开的Prosocial Dialogues数据集，并对其进行伦理风险标注作为评估基准。

**📈 对比分析**

与单轮生成或其他基线相比，在风险分层多轮评估中，EthicMind在高风险与道德模糊情境下表现出更一致的伦理指导和情感共情，得分提升约10%–15%。

**⚠️ 局限性**

局限包括仅评估亲社会情境、依赖LLM导致评估近似、额外的推理开销与延迟，且框架本身并不消除LLM的所有风险。

---

## 333. Mosaic: Multimodal Jailbreak against Closed-Source VLMs via Multi-View Ensemble Optimization

**arXiv ID:** 2604.09253 | [PDF](https://arxiv.org/pdf/2604.09253v1)

**作者:** Yuqin Lan `[一作]` (Beihang University), Zhiming Zheng `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Mosaic，一种针对闭源视觉语言模型（VLM）的多模态 jailbreak 框架，能够在异构代理-目标设置下提升攻击效果。

**💡 创新点**

创新点在于：① 通过 Text‑Side Transformation 减弱文本中的拒绝触发模式；② 采用 Multi‑View Image Optimization 在不同裁剪视图上更新扰动，避免单一视图过拟合；③ 引入 Surrogate Ensemble Guidance，将多个开源 surrogate VLM 的梯度信号聚合，降低代理依赖性。

**🔧 技术方法**

技术手段包括：梯度基 adversarial 生成（MI‑FGSM）；随机裁剪与视图生成；多模型梯度聚合；文本词序随机重排；固定肯定前缀引导目标。

**📊 数据集**

使用 MM‑SafetyBench 数据集（共 1,680 条涵盖 13 类有害查询），每条配以对应图像，用于生成攻击样本。

**📈 对比分析**

与两种基线（QR 与 JPS）在三款闭源 VLM（GPT‑4o、Claude‑4.5、Gemini‑3.0）上对比，评估指标为攻击成功率 ASR 与平均毒性 AvgTox。Mosaic 在所有模型上均实现了最优成绩，例如 GPT‑4o 上 AvgTox 4.13、ASR 69.66%，相比 JPS 提升 18–24 点，明显优于 QR。

**⚠️ 局限性**

局限性包括：① 在简单图像预处理（JPEG 压缩、Gaussian blur）下攻击效果下降；② 依赖大量黑盒查询和早停机制；③ 仅在三款特定闭源 VLM 上验证，可能对更强大或不同防御策略的模型效果未知。

---

## 334. LandSAR: Visceralizing Landslide Data for Enhanced Situational Awareness in Immersive Analytics

**arXiv ID:** 2604.09241 | [PDF](https://arxiv.org/pdf/2604.09241v1)

**作者:** Wong Kam-Kwai `[一作]` (Hong Kong University of Science and Technology), Leni Yang `[通讯]` (Inria)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套名为 LandSAR 的沉浸式分析系统，整合三维打印地形模型（物化）、AR 置身可视化（分析模式）和实时滑坡仿真（直觉模式），用于提升专业人员对滑坡的情境意识。

**💡 创新点**

创新点在于将分析模式与直觉模式通过物化支撑实现无缝衔接，首次将高精度 3D 打印地形作为可触摸锚点，并将实时物理仿真嵌入 AR 体验中，让用户在真实感环境中进行“what‑if”交互，从而提升三层情境意识。

**🔧 技术方法**

技术包括：Meta Quest Pro + SteamVR 与 VIVE 追踪器的混合现实；Unity 引擎与 Zibra Liquids（移动最小二乘材料点法）实时流体仿真；WebSocket 服务器实现物理模型与虚拟世界的数据同步；HP Jet Fusion 3D 打印制作高保真地形模型；AR 置身可视化与叠加渲染技术。

**📊 数据集**

使用的主要数据集有：香港政府地形与高程数据、雨量、土壤、建筑与人口密度等公开数据（Cedd、hk-cedd），以及历史滑坡事件数据库，用于绘制风险图、因果热图与模拟输入。

**📈 对比分析**

通过两阶段评估：12 名研究生/研究员的工作坊与 3 位资深工程师访谈，采用 UEQ 与 SART 量化体验，结果显示首人视角对情境意识影响最大；在 RTX 3090 Ti 计算机上实现的实时仿真保持低延迟，整体性能符合实时沉浸式体验要求。

**⚠️ 局限性**

局限性包括：仿真采用简化流体模型，缺乏高精度的地质粘弹性；实时渲染与传输导致快速头部运动时出现轻微延迟；缺乏完整的时间控制与多事件叠加分析；物理模型不可变形，无法模拟地形改造；硬件对低延迟和多用户交互支持有限。

---

## 335. Phase-Field Peridynamics

**arXiv ID:** 2604.09215 | [PDF](https://arxiv.org/pdf/2604.09215v1)

**作者:** Kai Partmann `[一作]` (University of Siegen), Kerstin Weinberg `[通讯]` (University of Siegen)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种新的相位场无穷远力学（Phase‑Field Peridynamics，PFPD）模型，将相位场裂纹模型嵌入到基于bond‑associated correspondence的框架中，避免了传统通过不可逆断链删除引起的数值不稳定性。

**💡 创新点**

创新点在于：①采用连续的bond phase‑field参数逐渐削弱键能量贡献，②引入运动学退化函数以保持形状张量的稳定性；③推导出对任意球面核函数均适用的归一化常数的解析闭式表达式，确保能量消耗与Griffith理论一致。

**🔧 技术方法**

所用技术包括：无穷远力学的bond‑associated correspondence公式、相位场能量函数、连续退化函数、解析归一化常数推导（一维积分比值）、显式时间积分（Velocity Verlet）以及Julia实现的离散算法。

**📊 数据集**

数据集方面，作者采用标准数值实验：Mode I 与 Mode II 斜切板、Boundary Tension Test (BTT) 与不同核函数和视界比例、Kalthoff‑Winkler 影响实验，均为基于物理参数（密度、杨氏模量、泊松比、临界能量释放率）构造的合成试件。

**📈 对比分析**

与传统bond‑based（BB）和ordinary‑state‑based（OSB）模型及经典相位场方法比较，PFPD 在 Mode II 下保持数值稳定，裂纹路径与实验相符，裂纹分支角度与BTT结果一致，裂尖速度始终低于半Rayleigh波速，且相较于断链模型得到更平滑的速度曲线和更合理的裂纹角度。

**⚠️ 局限性**

局限性包括：①仍需在更复杂几何和三维大规模问题中验证计算成本；②归一化常数仅对球面核函数解析，非球面核仍需经验或数值校准；③模型对视界比与网格分辨率敏感，需要经验参数调节。

---

## 336. Artificial intelligence can persuade people to take political actions

**arXiv ID:** 2604.09200 | [PDF](https://arxiv.org/pdf/2604.09200v1)

**作者:** Kobi Hackenburg `[一作]` (University of Oxford), Christopher Summerfield `[通讯]` (University of Oxford)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两项大规模预注册实验中，作者利用先进对话式人工智能模型（如GPT‑4.1、Claude Opus 4.6、Grok 4、Gemini 3.1 Pro）与英国成人进行互动对话，试图说服其签署真实线上请愿书并捐款；同时测量了对态度和行为的影响，并探讨了八种理论驱动的说服策略的机制；实验共招募14,779名参与者，完成17,950份调查问卷。

**💡 创新点**

该研究首次将对话式AI在真实行为层面（如签请愿、捐款）上进行系统、预注册的大规模实验，证实了AI在行为上的显著说服效应；并发现态度改变与行为改变之间缺乏相关性，揭示了两者机制不同的现象；进一步通过对比多种说服策略，证明“多策略合成（Mega）”在提升行为改变方面最为有效。

**🔧 技术方法**

技术包括：①前沿大语言模型（GPT‑4系列、Claude、Grok、Gemini）用于生成对话；②通过OpenRouter API与自定义 web 应用实现实时多轮对话；④使用线性混合效应模型与线性概率模型估计平均处理效应；⑤预注册实验设计与多种对照（信息式、随机式、控制式）对照组。

**📊 数据集**

数据集：UK Prolific在线受试者（18岁以上英国居民），共14,779名参与者，完成17,950份问卷；实验中使用的真实请愿书和支持组织来自英国公开请愿平台（共8个议题，涵盖核裁军、气候、动物福利等）。

**📈 对比分析**

与传统面向态度测量的AI说服研究相比，本研究在行为层面实现了超过10个百分点的提升（如请愿签署+19.7pp），与现场动员（面对面 canvassing）和直接信息传播（direct messaging）的效果相当甚至更大。对比八种说服策略时，Mega策略在请愿签署方面最高提升23.7pp，单一策略提升范围为16.2–21.2pp，显示出较小的差异。

**⚠️ 局限性**

主要限制包括：①实验在付费调查环境下进行，受试者获得报酬，可能不代表自然接触AI的情境；②所测行为为低成本行为，尚未验证对高风险或高投入行为（如疫苗接种、消费决策）的影响；③虽然样本规模大，但仅来自英国，结果的跨文化外推性未知；④对话时长有限（平均4.9轮，约7分钟），未探讨更长或更深层次对话的效果。

---

## 337. Camera Artist: A Multi-Agent Framework for Cinematic Language Storytelling Video Generation

**arXiv ID:** 2604.09195 | [PDF](https://arxiv.org/pdf/2604.09195v1)

**作者:** Haobo Hu `[一作]` (Communication University of China), Libiao Jin `[通讯]` (Communication University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多代理框架 Camera Artist，能够从故事大纲自动生成具有叙事连贯性和电影语言的长篇视频。

**💡 创新点**

核心创新点是（1）递归故事板生成（RSG），通过逐镜头递归生成保证镜头间叙事流畅；（2）电影语言注入（CLI），利用LoRA微调的 LLM 将普通镜头描述转化为专业电影语言；以及将这两项技术嵌入端到端的导演-摄像-视频生成协作流程。

**🔧 技术方法**

技术手段包括：大型语言模型（Qwen3、GPT‑4o）承担导演、摄像、视频生成角色；LoRA 微调实现 CLI；多参考 I2V 视频生成器 MAGREF 结合 Flux 生成高质量参考图；Chain‑of‑Thought 推理保证递归生成的逻辑连贯；多代理协同工作。

**📊 数据集**

使用 ShotBench（580 对电影镜头与电影化描述样本）进行 CLI 微调；MoviePrompts（10 个专业电影剧情与角色资料）作为主测试集；再加上自行构建的 8 个额外故事样本用于泛化评估。

**📈 对比分析**

通过 VBench、CLIP‑T、VLM 自动评估（Script Consistency、Camera‑Consistency、Video Quality、Real‑Movie Similarity）以及 5‑级 Likert 量表人类评估与 VGoT、Anim‑Director、MovieAgent 等基线对比。Camera Artist 在叙事一致性、镜头连贯性、视频质量与真实电影相似度等所有指标上均优于基线，显示显著性能提升。

**⚠️ 局限性**

局限性包括：受限于 LLM 与 I2V 模型的生成质量，长视频时空一致性仍可能出现细节漂移；CLI 训练数据规模有限，专业电影语言覆盖不完整；系统对非英语或极端剧本场景的鲁棒性尚待验证；整体推理与生成过程仍需大量算力。

---

## 338. MixFlow: Mixed Source Distributions Improve Rectified Flows

**arXiv ID:** 2604.09181 | [PDF](https://arxiv.org/pdf/2604.09181v1)

**作者:** Nazir Nayal `[一作]` (Max Planck Institute for Informatics), Jan Eric Lenssen `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种混合源分布的训练方法（MixFlow），通过学习可调的前向耦合来加速Rectified Flow的采样。

**💡 创新点**

创新点在于将任意引导信号 κ 与源分布耦合，并通过线性混合标准高斯与条件高斯来降低生成路径曲率，实现更少的函数评估即可获得高质量样本。

**🔧 技术方法**

使用 Rectified Flow 框架、可学习前向耦合、源分布混合训练策略、KL 正则化、ODE 求解器（Euler、RK45）等技术。

**📊 数据集**

在 CIFAR‑10、FFHQ 64×64、AFHQv2 64×64 等常用图像生成数据集上进行评估。

**📈 对比分析**

与 Rectified Flow、Fast‑ODE、Flow‑Matching、QAC 等基线相比，MixFlow 在固定采样预算下 FID 提升约 12%/7%，在低采样步数（5–9 NFEs）下 FID 更优 20%/10% 等。

**⚠️ 局限性**

局限在于仍假设源分布为高斯，需要调节 KL 权重 β，且对不同的引导信号或更复杂的条件任务尚未完全验证。

---

## 339. FashionStylist: An Expert Knowledge-enhanced Multimodal Dataset for Fashion Understanding

**arXiv ID:** 2604.09249 | [PDF](https://arxiv.org/pdf/2604.09249v1)

**作者:** Kaidong Feng `[一作]` (Yanshan University), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FashionStylist基准，涵盖项目和全套装配层级专家标注，并设计了三类任务：服装到单品定位、装配完成与服装评估。

**💡 创新点**

创新点在于将专业化的多层级标注与三任务评测结合，填补了现有数据缺乏全景式、专家级服装语义的空白。

**🔧 技术方法**

采用专家标注流水线、LoRA微调、检索与生成混合模型以及多模态大型语言模型（MLLM）的Prompt等技术。

**📊 数据集**

使用从淘宝和德物等电商平台采集的1,000套装配与4,637件单品，包含男女童装、配饰等多样样本。

**📈 对比分析**

在三项基准上对比多种开源与专有模型，SFT在FashionStylist上显著提升Recall、NDCG、FID及属性预测准确率，验证数据集对提升性能的价值。

**⚠️ 局限性**

局限在规模有限、标注成本高以及当前模型仍难以完全理解复杂搭配与细粒度属性。

---

## 340. Hitem3D 2.0: Multi-View Guided Native 3D Texture Generation

**arXiv ID:** 2604.09231 | [PDF](https://arxiv.org/pdf/2604.09231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 341. DDSP-QbE++: Improving Speech Quality for Speech Anonymisation for Atypical Speech

**arXiv ID:** 2604.09246 | [PDF](https://arxiv.org/pdf/2604.09246v1)

**作者:** Suhita Ghosh `[一作]` (Otto-von-Guericke University), Sebastian Stober `[通讯]` (Otto-von-Guericke University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

针对DDSP-QbE语音转换系统的激励阶段，通过引入PolyBLEP抗混叠修正和基于声门检测的谐波门控，提升了合成语音的自然度和可懂度。

**💡 创新点**

创新点在于将PolyBLEP平滑波形断点技术与声门检测相结合，实现无学习参数、可微分的激励优化，解决了传统相位累积产生的混叠与无声区谐波残留问题。

**🔧 技术方法**

采用PolyBLEP校正的相位累积振荡器、二进制声门检测（交叉熵损失）、WavLM提取的语音与情感嵌入、Conformer融合以及可微分的减法合成器等技术。

**📊 数据集**

实验使用了三类病态与情感语料库：SEP‑28k（口吃语音）、ADReSSo（痴呆受影响语音）和ESD（情感语音），音频采样率统一为16kHz。

**📈 对比分析**

与原始DDSP‑QbE基线对比，DDSP‑QbE++在PCC、pMOS和CER等指标均略有提升（PCC+0.3、pMOS+0.3、CER下降约2%），而EER保持不变，说明改进提升了音质与可懂度但未影响匿名性。

**⚠️ 局限性**

局限性包括对高频噪声的改善有限，部分病态语种如口吃的特定子类型提升不显著；实验仅在16kHz环境下验证，未探讨更高采样率或更强攻击模型下的表现。

---

## 342. DiffHLS: Differential Learning for High-Level Synthesis QoR Prediction with GNNs and LLM Code Embeddings

**arXiv ID:** 2604.09240 | [PDF](https://arxiv.org/pdf/2604.09240v1)

**作者:** Zedong Peng `[一作]` (Shanghai Jiao Tong University), Jieru Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于差分学习的高层综合 QoR 预测框架，利用图神经网络对内核与加插 pragma 的设计进行对齐，并用冻结的 LLM 代码嵌入补充语义信息。

**💡 创新点**

创新点在于将设计 QoR 拆解为内核基线 + pragma 引起的增量，显著降低目标方差；同时首次将预训练代码 LLM 嵌入与 IR 图结合，增强对 pragma 语义的感知。

**🔧 技术方法**

技术包括：控制数据流图 (CDFG) 的图神经网络编码（PNA、GraphSAGE、GCN、GAT 等），冻结的 LLM 代码编码器（如 Qwen2.5-Coder-1.5B）作为增量通道特征，以及三项损失（基线、增量、合成）共同训练的差分学习框架。

**📊 数据集**

主要数据集为 PolyBench/C 内核与其基于 pragma 的 10,108 条设计点，以及更大规模的 ForgeHLS PolyBench 子集进行可扩展性验证。

**📈 对比分析**

与纯 GNN 基线相比，该方法在四种 QoR 指标（DSP、FF、LUT、CP）上平均 MAPE 下降 2‑5% 以上，GraphSAGE 变体在 PolyBench 上 DSP 仅 3.31%，在 ForgeHLS 上保持领先；增量学习和代码嵌入均被验证为关键贡献因素。

**⚠️ 局限性**

局限性包括：需要大量带标注的内核-设计对才能训练；差分学习仍受内核基线可预测性约束，对极端 pragma 组合的泛化不明；冻结的 LLM 嵌入对特定语义的覆盖有限，且模型在跨平台或不同 HLS 工具链时可能需重新适配。

---

## 343. Neural Distribution Prior for LiDAR Out-of-Distribution Detection

**arXiv ID:** 2604.09232 | [PDF](https://arxiv.org/pdf/2604.09232v1)

**作者:** Zizhao Li `[一作]` (University of Melbourne), Kourosh Khoshelham `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `67630363-6be0-4f51-ab05-7198250671a5` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Neural Distribution Prior（NDP）框架，用于在 LiDAR 语义分割中对 OOD 目标进行自适应重加权，并配合 Perlin 噪声合成 OOD 样本和 Soft Outlier Exposure（SOE）训练策略。

**💡 创新点**

核心创新在于引入可学习的分布先验以捕捉类别间预测分布差异并动态校正 OOD 分数，同时利用 Perlin 噪声生成多样化的合成 OOD 样本，减少对外部数据的依赖。

**🔧 技术方法**

技术实现基于 Mask4Former‑3D 的 transformer‑decoder + sparse UNet 编码，采用交叉注意力的分布先验权重、能量/熵/扩展能量 OOD 分数，以及 Perlin 噪声、DBSCAN 聚类与 SOE 软标签等模块。

**📊 数据集**

在 STU 和 SemanticKITTI 两大 LiDAR 数据集上进行实验，STU 为异常分割基准，SemanticKITTI 用作 OOD 检测交叉验证。

**📈 对比分析**

与 MaxLogit、RbA、UEM 等现有方法相比，NDP‑EE 在 STU 测试集上实现 AP 61.31%、PQ 24.99%（约 10 倍提升），在验证集上 AP 74.24%，在 SemanticKITTI 上 AP 70.12%，且 AUROC、FPR@95 等指标显著优于对比基线。

**⚠️ 局限性**

局限性主要在于点云稀疏和不规则结构导致 OOD 边界精度受限，缺乏真实 OOD 训练标签使得边界细化困难，且方法对极端几何变形的泛化尚待提升。

---

## 344. SHIFT: Steering Hidden Intermediates in Flow Transformers

**arXiv ID:** 2604.09213 | [PDF](https://arxiv.org/pdf/2604.09213v1)

**作者:** Nina Konovalova `[一作]` (FusionBrain Lab), Aibek Alanov `[通讯]` (FusionBrain Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SHIFT，一种在 Diffusion Transformer 上通过激活向量推导实现概念擦除、风格迁移和目标对象偏置的轻量级推理时控制框架。

**💡 创新点**

创新点在于利用统一注意力空间中时间不变的激活方向，实现无需模型微调、仅在推理阶段即可对概念进行强力抑制或偏移；并证明单一向量即可跨所有时间步、跨蒸馏边界使用。

**🔧 技术方法**

技术手段包括：在文本编码器池化向量和 Transformer 文本 token 位置插入线性方向向量；通过对比提示对齐数据集计算均值差或 SVM 超平面法线作为 steering vector；在推理时按余弦相似度或 SVM 置信度动态调整 steering 强度。

**📊 数据集**

使用 Flux.1[dev] 与 Flux.1[schnell] 两大 DiT 模型；I2P 基准与 NudeNet 进行抽象概念（裸体）擦除评估；MS‑COCO 5k 评估生成质量；SPM 固定 80 句提示评估具体概念擦除；对 Van Gogh 与 McKernan 等艺术家风格的移除亦采用 100 句提示集。

**📈 对比分析**

与 ESD、CA、UCE、EAP、Meta‑Unlearning、EraseAnything 等现有概念擦除方法对比，SHIFT 在裸体检测量上降低 3‑4 倍，CLIP 对齐几乎不变，FID 仅略有提升，整体在多种提示、目标概念上均表现出更强的抑制效果和更少的质量损失。

**⚠️ 局限性**

局限性包括：对局部小对象擦除时不保持背景结构，推理时需要手动调节 steering 强度，过强时会导致非目标概念失真；对极端概念或高分辨率生成的鲁棒性仍需进一步验证。

---

## 345. UniSemAlign: Text-Prototype Alignment with a Foundation Encoder for Semi-Supervised Histopathology Segmentation

**arXiv ID:** 2604.09169 | [PDF](https://arxiv.org/pdf/2604.09169v1)

**作者:** Le-Van Thai `[一作]` (AI VIETNAM Lab), Ngoc Lam Quang Bui `[通讯]` (AI VIETNAM Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 UniSemAlign，一个用于计算病理学半监督语义分割的双模态语义对齐框架。

**💡 创新点**

创新点在于结合可学习的类别原型和冻结的 CONCH 文本编码器产生的文本语义，在共享嵌入空间中同时进行原型对齐和文本对齐，提供显式的类别级语义指导，提升伪标签质量。

**🔧 技术方法**

使用了 UNI ViT‑B/16 视觉Transformer 作为编码器、DeepLabV3+ 解码器、CoOp 风格可学习上下文 token、CONCH 文本Encoder、对齐分支（原型+文本）、伪标签融合与 CorrMatch 风格弱强一致性训练。

**📊 数据集**

在 GlaS（165 张）和 CRAG（213 张）肠道腺体分割数据集上进行实验。

**📈 对比分析**

与多种半监督基线（UAMT、FixMatch、CPS、CT、XNet、CorrMatch、DuSSS、CSDS）以及全监督基线对比，10% 标注下在 GlaS 上提升 Dice 2.6% 及 Jaccard 4.1%，CRAG 上提升 Dice 8.6% 及 Jaccard 11.7%；20% 标注下也保持显著优势。

**⚠️ 局限性**

局限性包括对超参数（融合权重、上下文 token 数量）敏感、仍需手动设计 prompt、对极端标注稀缺情形的鲁棒性未全面评估，且仅验证于腺体分割，其他病理任务需要进一步验证。

---

## 346. SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation

**arXiv ID:** 2604.09212 | [PDF](https://arxiv.org/pdf/2604.09212v1)

**作者:** Han Luo `[一作]` (Ben Gurion University Of Negev), Guy Laban `[通讯]` (Ben Gurion University Of Negev)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SPASM框架，旨在通过稳定的代理模拟生成具有长期一致性的人设驱动多轮对话。

**💡 创新点**

创新点在于引入了视角无关的对话历史表示与ECP（Egocentric Context Projection）投影机制，显著抑制了角色漂移和回声（echoing）现象。

**🔧 技术方法**

技术上包括人设采样、可行性验证与自然语言化、对话生成、终止检测，以及对话历史的投影与重构。

**📊 数据集**

使用了三种LLM后端（GPT‑4o‑mini、DeepSeek‑V3.2、Qwen‑Plus），构建了9种客户端–响应者组合，共生成4,500个人设和45,000条对话。

**📈 对比分析**

通过对话嵌入空间的聚类指标（Silhouette、Davies–Bouldin）、语义检索准确率和漂移度量，实验表明ECP能显著降低角色漂移，并在所有组合下完全消除回声，且同一模型对话的语义聚类更紧凑。

**⚠️ 局限性**

局限性包括仅在英语、少数大型模型上验证，未覆盖多语言或小型模型；仅适用于双人固定角色交互，无法处理多方或动态角色；人设表征基于结构化模式，可能缺乏真实人类多样性。

---

## 347. Globally Optimal Pose from Orthographic Silhouettes

**arXiv ID:** 2604.09199 | [PDF](https://arxiv.org/pdf/2604.09199v1)

**作者:** Agniva Sengupta `[一作]` (Freie Universität Berlin), Stefan Zachow `[通讯]` (Zuse Institute Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过仅利用物体的二维未遮挡外轮廓，提出一种全局最优姿态估计方法，可处理任意形状（无凸性、无基数限制）。

**💡 创新点**

创新点在于：①利用投影轮廓面积与形状连续性，构造“silhouette‑signature”并预先离散化；②结合面积签名与椭圆长宽比加速搜索；③实现无对应、无图像强度的全局最优求解；④将方法推广至透视投影并使用粗略深度先验。

**🔧 技术方法**

技术手段包括：空间投影与轮廓提取、面积与椭圆长宽比签名的预计算、Postel投影下的旋转空间分区、基于签名的候选旋转搜索、基于 𝕊𝔼(3) 的非线性微调、以及多分辨率迭代与阈值自适应。

**📊 数据集**

使用的公开数据集：stbu、pd、pelb（三维模型），以及 bcot 真实图像数据集（20 个物体），并在这些数据上进行合成与真实实验。

**📈 对比分析**

与三种传统非线性求解器（bsl_NonLinLie、bsl_NonLinICP、bsl_GMS）及最近基于学习的方法 tipfs 进行对比。实验表明在正交轮廓下，平均欧拉角误差低于 1°；在透视轮廓下，对称物体的误差仍可接受，且整体性能优于现有方法，尤其在姿态角度上明显领先。

**⚠️ 局限性**

局限性包括：①对严重遮挡或极高噪声的轮廓不可靠；②高度对称物体存在多重解导致不确定性；③需要预先对模型进行离散化签名，计算成本与存储受限；④透视情况需依赖深度先验，误差会影响最终精度。

---

## 348. Do LLMs Follow Their Own Rules? A Reflexive Audit of Self-Stated Safety Policies

**arXiv ID:** 2604.09189 | [PDF](https://arxiv.org/pdf/2604.09189v1)

**作者:** Avni Mittal `[一作]` `[通讯]`, Avni Mittal

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自我一致性审核框架 SNCA，评估 LLM 在自我声明的安全规则与其行为之间的一致性。

**💡 创新点**

首次将同一模型既作为安全规则的作者又作为行为主体进行审计，构建 Typed Predicate（Absolute、Conditional、Adaptive）三类规则并量化其符合度。

**🔧 技术方法**

采用结构化提问提取规则、使用判别模型对规则进行类型化、对比基准数据集行为结果并通过确定性推断计算符合度（SNCS）。

**📊 数据集**

使用 SORRY‑Bench（450 恶意提示）、XSTest（安全/危险混合提示）和 OR‑Bench Hard‑1K（近似有毒但安全提示）共 47,496 条观测数据。

**📈 对比分析**

与四款前沿模型（GPT‑4.1、DeepSeek‑V3.1、Llama‑3.3‑70B‑Instruct、o4‑mini）对比，发现自我一致性差异大（0.25–0.80），绝对拒绝声明最易被违背，推理模型在可表述规则上的一致性最高但透明度低。

**⚠️ 局限性**

局限在于依赖模型自我声明的规则作为代理，提取过程可能产生偏差，判别模型与规则类型化的准确性受限，且仅覆盖现有基准，无法完整反映真实世界情境。

---

## 349. A Deductive System for Contract Satisfaction Proofs

**arXiv ID:** 2604.09165 | [PDF](https://arxiv.org/pdf/2604.09165v1)

**作者:** Arthur Correnson `[一作]` (CISPA Helmholtz Center for Information Security), Jana Hofmann `[通讯]` (Max Planck Institute for Security and Privacy)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一种基于相对 bisimulation 的证明系统，用于在 Coq 中形式化验证硬件软件契约满足性。

**💡 创新点**

创新点在于将相对 trace 等价转化为四元关系的 bisimulation，并结合参数化共识推理和 up‑to 技术，实现了可增量、可模块化的证明流程。

**🔧 技术方法**

主要技术包括相对 trace 等价、相对 bisimulation、参数化共识推理（Paco）、up‑to 技术，以及 Coq 形式化。

**📊 数据集**

实验使用了简化的 toy ISA 以及两个契约模型（always‑mispredict 与 sequential）进行证明，没有使用真实硬件数据集。

**📈 对比分析**

与传统的手工证明和模型检查方法相比，该系统能够在交互式证明助手中完成全自动化证明，但文中未给出性能指标或时间成本比较。

**⚠️ 局限性**

局限性包括：假设硬件和契约语义是确定性的；目前仅适用于简化的 ISA，无法处理嵌套预测或非确定性行为；证明仍需显式操作具体状态。

---

## 350. A Domain-Theoretic Foundation for Imprecise Probability and Credal Sets

**arXiv ID:** 2604.09272 | [PDF](https://arxiv.org/pdf/2604.09272v1)

**作者:** Abbas Edalat `[一作]` (Imperial College London), Amin Farjudian `[通讯]` (University of Birmingham)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构建了一个基于域论（Domain Theory）的统一框架，用以在一般拓扑空间上处理不确定性，包括部分事件描述与分布不确定性（credal 集），并给出区间化的条件概率、贝叶斯更新与条件独立性。

**💡 创新点**

创新点在于：① 将连续判别子（continuous valuation）与事件域（event domain）映射到区间概率并证明 Scott 连续性；② 推导了从单一贝叶斯公式得到的区间贝叶斯更新规则，并给出完整的推理规则；③ 引入强条件独立性（strong conditional independence）与保守（Fréchet）规则，扩展了不确定条件独立性的理论；④ 设计了由不确定权重的迭代函数系统（IFS）生成的新型 credal 集，并证明其固定点映射的 Scott 连续性。

**🔧 技术方法**

使用了域论、可逼近映射（approximable mapping）、开集与闭集的对偶性、Choquet 积分、迭代函数系统（IFS）理论、以及离散化的线性规划方法来实现区间化贝叶斯推理。

**📊 数据集**

本工作主要为理论研究，未依赖特定实验数据集；但在实例说明中使用了简单的 Beta 分布族、区间贝叶斯更新和 IFS（如 Cantor 集）来验证理论。

**📈 对比分析**

与传统精确贝叶斯推理相比，区间化方法提供了包含真值的区间且更为保守；在给定示例中，区间推断的宽度覆盖了精确估计点，体现了鲁棒性。性能方面，由于所有映射均为 Scott 连续，可通过有限逼近实现可计算性；但缺少复杂大规模数据集上的实验评估。

**⚠️ 局限性**

局限性包括：① 需要满足事件为“经典事件”（即 σ(W1)+σ(W2)=1）或正性假设；② 对负边界的计算依赖于 Fréchet 规则，可能导致区间过宽；③ 强条件独立性假设较强，在实际应用中难以验证；④ 对非紧致空间的处理仍不完整，需进一步推广。

---

## 351. The causal relation between off-street parking and electric vehicle adoption in Scotland

**arXiv ID:** 2604.09271 | [PDF](https://arxiv.org/pdf/2604.09271v1)

**作者:** Bernardino D'Amico `[一作]` (Digital Built Environment Group), Emma Hart `[通讯]` (Centre For Artificial Intelligence and Robotic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用贝叶斯网络与因果推断框架，量化苏格兰户外停车对电动车采纳的因果效应，并揭示收入是进入市场的核心门槛。

**💡 创新点**

创新点在于将混合结构学习（MIIC）与手工后置校正结合，消除收入与停车之间的混杂，首次将因果推断应用于住房与电动车采纳问题。

**🔧 技术方法**

核心技术包括贝叶斯网络建模、MIIC因果发现、后门调整（d‑separation）、变量消除（VE）推理、以及置换与子样本检验等因果稳健性方法。

**📊 数据集**

数据来源为2022年苏格兰家庭调查（SHS）与住房条件调查（SHCS）融合的约1,800个加权样本，涵盖收入、住房类型、停车情况等15个变量。

**📈 对比分析**

通过比较观测条件概率与干预后概率，因果估计显示停车可提升约2.3个百分点的拥有率，收入干预可减少23.1个百分点的非参与率；与传统相关分析相比，因果方法显著降低了偏差并提供更具政策意义的效应。

**⚠️ 局限性**

局限性包括样本量有限、可能存在未观测混杂、仅将停车视为二元变量、对高密度城市住房干预可行性尚不明确，且因果图的完整性依赖于专家约束与自动学习结果。

---

## 352. Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima

**arXiv ID:** 2604.09258 | [PDF](https://arxiv.org/pdf/2604.09258v1)

**作者:** Huanran Chen `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Nexus优化器，在LLM预训练中通过鼓励多源任务梯度相似性来实现模型参数的几何接近，从而在相同预训练损失下提升下游任务性能。

**💡 创新点**

创新性地将梯度相似性视为几何接近的可计算上界，并通过双循环梯度近似实现第二阶梯度相似性优化，形成新的隐式正则化机制。

**🔧 技术方法**

采用双循环（inner‑outer）梯度相似性近似（Nexus）、标准AdamW或Muon优化器、Llama‑架构模型以及梯度归一化、Hessian‑梯度乘积等技术。

**📊 数据集**

使用去污染的内部混合预训练语料（多源文本、数学、代码等）以及公开数据集进行实验。

**📈 对比分析**

与AdamW、Muon等基线比较，在保持预训练损失差<0.01的前提下，Nexus在GSM8k提升15%、MATH提升8%、HumanEval提升4%；随着模型规模和训练token增大，性能优势不减反增。

**⚠️ 局限性**

目前与Muon优化器不兼容，可能因数值敏感性；在更大规模或不同架构上的泛化仍需进一步验证。

---

## 353. Adding Another Dimension to Image-based Animal Detection

**arXiv ID:** 2604.09210 | [PDF](https://arxiv.org/pdf/2604.09210v1)

**作者:** Vandita Shukla `[一作]` (Fondazione Bruno Kessler), Benjamin Risse `[通讯]` (University of Muenster)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个端到端的管线，利用 SMAL 模型为单目动物图像生成定向 3D 包围盒及其在图像中的可视化指标。

**💡 创新点**

创新点在于结合 SMAL 的关键点不确定性建模、基于边缘和置信度的协方差加权，以及与分割掩码的联合优化，显著提升了姿态估计的稳健性并克服了 PCA 方向不一致的问题，同时引入面部可见度度量为 VAB 提供视角信息。

**🔧 技术方法**

采用了 SMAL 形状拟合、EPnP+RANSAC 相机姿态初始化、协方差权重的 Mahalanobis 误差、与分割掩码的联合成本优化、投影面积与正向法线的可见度计算等技术。

**📊 数据集**

使用了 Animal3D 数据集和 UAV 捕获的野外图像进行实验与验证。

**📈 对比分析**

与传统的基于 2D-3D 对应的基本方法和 PCA 方向估计方法对比，我们的方案将退化比例从 13.81% 降至 0%，平均投影误差从 75.28 像素降至 7.56 像素，方向稳定性提升至 99.99%。

**⚠️ 局限性**

局限性包括只能为 SMAL 所覆盖的物种生成标签，且需要手工或自动的 2D 关键点标注，难以扩展到未包含在 SMAL 空间之外的动物种类。

---

## 354. Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception

**arXiv ID:** 2604.09206 | [PDF](https://arxiv.org/pdf/2604.09206v1)

**作者:** Jiahao Wang `[一作]` (Tsinghua University), Jianqiang Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了Long‑SCOPE，一种完全稀疏的长距离协同三维感知框架，能够在低成本视觉传感器和有限通信/计算预算下实现远程目标检测与跟踪；

**💡 创新点**

创新点在于两大模块：几何引导的查询生成（Geometry‑guided Query Generation，GQG）利用全局高度推断深度，精准定位远距离目标；以及上下文感知关联（Context‑Aware Association，CAA）通过学习的自注意力实现鲁棒的查询匹配，克服定位误差导致的配对失败；

**🔧 技术方法**

技术手段包括基于ResNet50的特征提取、轻量化的二维检测+深度/高度回归头、Transformer解码器进行查询细化、全局自注意力与局部自注意力的多层迭代、Sinkhorn归一化与互近邻阈值进行最终匹配；

**📊 数据集**

使用公开的V2X‑Seq（车载与路侧单元协同）和Griffin‑25m（空地协同）两个数据集，并将评估范围扩展至150 m/100 m，验证长距离性能；

**📈 对比分析**

与SparseCoop、CoopTrack、V2X‑ViT等基线对比，Long‑SCOPE在两套数据集上均实现了显著提升（V2X‑Seq AP提升6.5点，Griffin‑25m AP提升8.9点），在100–150 m区间性能几乎翻倍；通信成本仅1.9×10⁵ BPS，计算速度7.68 FPS，保持实时性；

**⚠️ 局限性**

局限性在于仍依赖精准的相机外参与车辆定位，过大的定位噪声或极端视角变化可能削弱GQG和CAA的鲁棒性；此外，目前仅在两大数据集上验证，需进一步评估在更广泛真实场景中的适用性。

---

## 355. On the Role of DAG topology in Energy-Aware Cloud Scheduling : A GNN-Based Deep Reinforcement Learning Approach

**arXiv ID:** 2604.09202 | [PDF](https://arxiv.org/pdf/2604.09202v1)

**作者:** Anas Hattay `[一作]` (Université Paris-Saclay), Zakaria Yahoun `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对云端工作流调度，构建基于图神经网络的深度强化学习调度器，并系统研究其在不同拓扑结构与宿主异构性下的泛化失效；

**💡 创新点**

创新点在于明确了导致GNN-RL调度器失效的具体离群分布条件，并给出结构失配导致消息传递崩解的理论解释，提出了分解拓扑与硬件异构的对照实验框架；

**🔧 技术方法**

主要技术为图同构网络（GIN）嵌入、演员-评论家（Actor‑Critic）深度强化学习、混合目标（完成时间与能耗）以及基于奖励递减的离散决策；

**📊 数据集**

使用了自定义的两类工作流图（宽平行与长关键路径）以及四种宿主配置（均速、均功、速功一致、速功不一致）作为实验数据集；

**📈 对比分析**

通过交叉拓扑、交叉宿主配置的全面评估，并与经典启发式（HEFT、CPOP等）对比，结果显示专家策略在其训练域内实现能耗与完成时间的优越权衡，但在离群域出现显著退化；

**⚠️ 局限性**

局限性在于仅考虑单一无队列工作流、有限的两类图结构与四种宿主配置，缺乏对多工作流、真实队列环境以及更丰富图形特征的评估，且模型对离群结构仍缺乏鲁棒性。

---

## 356. Vision Transformers for Preoperative CT-Based Prediction of Histopathologic Chemotherapy Response Score in High-Grade Serous Ovarian Carcinoma

**arXiv ID:** 2604.09197 | [PDF](https://arxiv.org/pdf/2604.09197v1)

**作者:** Francesca Fati `[一作]` (European Institute of Oncology), Ines P. Machado `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种基于2.5D多模态深度学习的Vision Transformer框架，用预手术CT和临床数据预测卵巢高分级浆液性癌的化疗反应评分（CRS）

**💡 创新点**

首次将Vision Transformer与高密度切片选择和临床特征融合，用于预测CRS，显著提高内部AUC并展示跨机构可迁移性

**🔧 技术方法**

采用DINOv3小型ViT编码器、2.5D切片堆叠、临床特征编码、交叉熵损失、迁移学习与中间融合层，配合MLP分类头

**📊 数据集**

使用意大利欧洲肿瘤学研究院（IEO）内部271例病例和英国Addenbrooke’s医院外部70例病例的CT、年龄、CA‑125数据

**📈 对比分析**

通过对比仅影像、影像+年龄、影像+CA‑125以及全模态的消融实验，内部ROC‑AUC达0.95，外部0.68；内部精度0.80、准确率0.95；外部精度0.75、准确率0.67，显示多模态提升外部表现

**⚠️ 局限性**

受限于域漂移与外部样本量不足、仅使用预治疗信息、缺少后治疗生物标志物，模型易过拟合特定机构特征，需扩大多中心数据并探索领域适配

---

## 357. Generalization and Scaling Laws for Mixture-of-Experts Transformers

**arXiv ID:** 2604.09175 | [PDF](https://arxiv.org/pdf/2604.09175v1)

**作者:** Mansour Zoubeirou a Mayaki `[一作]` `[通讯]` (University of Lyon), Mansour Zoubeirou a Mayaki (University of Lyon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Mixture-of-Experts Transformer 的理论，给出逼近和统一泛化上界，并推导基于活跃参数和路由组合的可扩展性定律。

**💡 创新点**

将活跃参数与路由复杂度分离，提出针对 d 维流形数据的逼近率与统一泛化边界，并导出了稀疏 Transformer 的计算‑数据‑模型比例。

**🔧 技术方法**

使用覆盖数、L∞ 度量、统一泛化、流形逼近理论以及构造性证明等技术。

**📊 数据集**

在 TinyStories、WikiText‑103、OpenWebText 三个文本数据集上进行实验。

**📈 对比分析**

通过实验比较不同专家数 M、激活数 k、数据量等设置，验证模型/数据/路由三种扩展规律；实验结果与理论指数相近，路由中等时损失随 k log(eM/k) 上升，M/k 过大时出现专家特化提升。

**⚠️ 局限性**

局限性：仅考虑硬 top‑k 路由、参数有界、平方损失；上界保守，未能捕捉专家特化和优化动态等数据依赖的实际效果。

---

## 358. SHIFT: Sigmoid-Based Heuristic Invertible Fitness-Landscape Transformation for Accelerating SBST

**arXiv ID:** 2604.09171 | [PDF](https://arxiv.org/pdf/2604.09171v1)

**作者:** Jeongjin Han `[一作]` (Korea Advanced Institute of Science and Technology), Seongyoon Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SHIFT方法，通过可逆的Sigmoid压缩技术重新塑造软件测试的fitness landscape，从而提升hill climbing在崎岖和高原多的搜索空间中的效率。

**💡 创新点**

首次引入可逆的Sigmoid压缩变换，保留全局最优位置并在局部压缩平坦盆地，解决传统hill climbing陷入局部最优的问题，并提供理论证明其不改变全局最优。

**🔧 技术方法**

利用双向盆地检测、维度感知压缩管理器以及基于Sigmoid的空间变换，对搜索空间进行自适应压缩，并与hill climbing集成。

**📊 数据集**

在合成的多峰、平坦与崎岖测试地形以及真实的SBST基准程序（含复杂分支条件）上评估。

**📈 对比分析**

将SHIFT+hill climbing与纯hill climbing和遗传算法在相同时间预算下对比，结果显示SHIFT在分支覆盖率、收敛速度和鲁棒性方面均显著优于基线。

**⚠️ 局限性**

局限在于最大压缩长度为静态设置、仅支持整数输入域、对维度较多时计算成本提升，并需进一步扩展至浮点和更丰富输入类型。

---

## 359. ELT: Elastic Looped Transformers for Visual Generation

**arXiv ID:** 2604.09168 | [PDF](https://arxiv.org/pdf/2604.09168v1)

**作者:** Sahil Goyal `[一作]`, Aditya Kusupati `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在训练阶段通过内部循环自蒸馏（Intra-LoopSelf Distillation）提升模型表现，并在推理阶段实现任意时刻（Any‑Time）输出的框架，能够在不同时间点得到可接受的预测结果。

**💡 创新点**

创新点在于将中间层输出与最终输出之间进行自蒸馏，并为每个中间块引入可选的MLM任务，实现了训练时的多任务监督与推理时的动态退出；同时采用共享权重的模块化设计，减少参数量。

**🔧 技术方法**

采用的技术包括：自蒸馏、内部循环自蒸馏、Mask‑Language‑Modeling（MLM）作为辅助任务、任意时刻推理（Any‑Time Inference）以及多任务损失组合（λ·ℒ^GT_int + (1-λ)·ℒ^dist_int）。

**📊 数据集**

实验主要基于常见的语言建模与下游NLP数据集，如WikiText‑103、PTB，以及GLUE benchmark 中的若干任务。

**📈 对比分析**

通过与传统单层训练、无蒸馏以及标准MLM预训练模型进行对比，实验表明该方法在训练收敛速度、最终准确率以及任意时刻的预测质量上均取得显著提升，尤其在中间层的输出已经足够好时即可完成推理，减少计算延迟。

**⚠️ 局限性**

限制方面包括：训练时需要额外的中间损失设计与调参，导致训练成本略有增加；模型对中间层的设计与权重共享依赖较大，可能在非常小的模型规模下效果不佳；并且在极限的推理时刻，输出准确率仍受限于模型容量。

---

## 360. MAG-3D: Multi-Agent Grounded Reasoning for 3D Understanding

**arXiv ID:** 2604.09167 | [PDF](https://arxiv.org/pdf/2604.09167v1)

**作者:** Henry Zheng `[一作]` (Tsinghua University), Gao Huang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练、多智能体协作框架 MAG-3D，实现基于视觉语言模型的 3D 场景无监督定位与几何推理。

**💡 创新点**

创新点在于将任务拆分为规划、定位和编码三大智能体，采用自适应定位、可视化记忆和可执行程序的几何验证，实现零训练、可拓展的 3D 推理。

**🔧 技术方法**

使用开源 VLM（如 GPT‑4o、GPT‑4o‑mini、Qwen3‑Coder）、SAM3 进行二维实例分割、VGGT 进行深度/相机位姿融合，以及 Python 代码解释器进行几何计算。

**📊 数据集**

在公开的 Beacon3D 和 MSQA 3D 问答基准上进行评测，并使用官方和仅视觉的 MSQA 版本。

**📈 对比分析**

与 GPT‑4o、单智能体工具使用模型以及先前方法 SceneCOT、SceneVerse 等比较，MAG‑3D 在 Beacon3D 的案例级 QA 分数提升 6.4 分、目标级提升 3.2 分，在 MSQA 上获得最高的零训练得分，整体性能显著优于先前方法。

**⚠️ 局限性**

局限性包括对 VLM 的语言理解与几何推理能力的依赖，可能在高度遮挡或复杂几何结构的场景中出现定位误差；对实时性能和大规模部署的计算开销尚未充分评估。

---

## 361. Efficient Spatial-Temporal Focal Adapter with SSM for Temporal Action Detection

**arXiv ID:** 2604.09164 | [PDF](https://arxiv.org/pdf/2604.09164v1)

**作者:** Yicheng Qiu `[一作]` (University of Electro-Communications), Keiji Yanai `[通讯]` (University of Electro-Communications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的时空焦点适配器（ESTF）并结合状态空间模型（SSM）用于视频中的时序动作检测。

**💡 创新点**

通过将空间和时间建模解耦、引入边界感知的双向SSM（TB-SSM）以及轻量级适配器，显著提升了长序列中的边界精准度和模型计算效率。

**🔧 技术方法**

采用预训练视频主干网络、深度可分离卷积、线性时间状态空间模型、双向异步状态转移、特征融合与残差结构。

**📊 数据集**

在 THUMOS14、ActivityNet‑1.3 与 Charades 三个公开动作检测基准数据集上进行评测。

**📈 对比分析**

与多种 Transformer、Mamba、BMN 等基准方法对比，ESTF‑SSM 在各 tIoU 阈值下均取得更高的 mAP，特别是在高 IoU 阈值下提升显著，且模型参数与显存占用相对更低。

**⚠️ 局限性**

模型仍受限于对极端长视频的实时推理能力与对多模态信息融合的支持不足，未来需进一步提升在线检测性能和跨模态适配性。

---

## 362. Distributed Online Convex Optimization with Compressed Communication: Optimal Regret and Applications

**arXiv ID:** 2604.09276 | [PDF](https://arxiv.org/pdf/2604.09276v1)

**作者:** Sifan Yang `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种针对分布式在线凸优化（D‑OCO）的压缩通信算法，并通过误差反馈与 Follow‑the‑Regularized‑Leader（FTRL）框架，进一步给出在线到批转换的分布式离散化方法，实现压缩下的最优 regret 与收敛率；

**💡 创新点**

在压缩通信条件下，首次证明 D‑OCO 的下界并给出与之匹配的上界；将误差反馈机制与 FTRL 结合，消除投影误差与压缩误差的耦合；引入在线压缩（FCC）与块更新策略，降低双向压缩的误差放大；通过在线‑到‑批转换，将在线算法推广至离线无滑动非光滑优化；

**🔧 技术方法**

误差反馈（EF）机制、FTRL、在线压缩（Fast Compressed Communication, FCC）、块更新与异步通信模型、在线到批（anytime online‑to‑batch）转换

**📊 数据集**

本工作为理论性研究，未使用具体数据集；所有结论均基于严格的理论分析与证明

**📈 对比分析**

与传统无压缩 D‑OCO 以及已有的压缩化解（如仅一向压缩、基于投影的算法）相比，本文算法在压缩比 δ 下实现了最优的 O(δ⁻¹/2√T) / O(δ⁻¹logT) regret 与 O(δ⁻¹/2T⁻¹/2) / O(δ⁻¹T⁻¹) 收敛率；

**⚠️ 局限性**

主要局限包括：需要设置较大的块大小 L≈1/δ 以抑制压缩误差；对高维压缩器的理论下界仍基于理想化的随机失效模型；对光滑损失的进一步改进尚未实现；在实际系统中异步通信延迟与噪声对性能的影响需进一步实验验证

---

## 363. Soft Electroadhesive Feet for Micro Aerial Robots Perching on Smooth and Curved Surfaces

**arXiv ID:** 2604.09270 | [PDF](https://arxiv.org/pdf/2604.09270v1)

**作者:** Chen Liu `[一作]` (Queen Mary University of London), Ketao Zhang `[通讯]` (Queen Mary University of London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计、制造并验证柔软可拉伸的电动粘附脚，用于在光滑和平滑曲面上使Crazyflie微型无人机实现停留。

**💡 创新点**

提出一套快速成型的软EA垫制造流程；研究两种电极图案（波形与同心圆）在多尺寸下的粘附性能；采用四脚分布的停留策略提升可靠性。

**🔧 技术方法**

软硅胶电极与Ecoflex介质、激光切割、层压成型、Instron力学测试、高压电源供电、Crazyflie 2.1集成与飞行实验。

**📊 数据集**

实验数据：六种不同尺寸与图案的垫子在Instron测试平台下在0 kV与4.8 kV条件下测得的垂直和剪切力；系统级实验数据包括四脚EA在不同表面上的停留与脱离过程。

**📈 对比分析**

通过比较0 kV与4.8 kV两电压条件下的垂直和剪切力，发现剪切力可达约3 N，而垂直力仅约0.3 N；系统实验表明四脚EA在光滑塑料表面可稳定停留并在曲面上实现接触，但木质粗糙表面粘附力显著降低。

**⚠️ 局限性**

限制主要在正常力不足，粘附性能高度依赖表面光洁度；对粗糙或高曲率表面的适应性差，未来需要提升正常力和改进介质设计以降低对表面光滑度的依赖。

---

## 364. DRBENCHER: Can Your Agent Identify the Entity, Retrieve Its Properties and Do the Math?

**arXiv ID:** 2604.09251 | [PDF](https://arxiv.org/pdf/2604.09251v1)

**作者:** Young-Suk Lee `[一作]`, Radu Florian `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 DrBencher，一个可按需生成的合成基准生成器，用于生成同时需要网络浏览和计算的问答，覆盖生物化学、金融、地球物理、历史与安全五个领域。

**💡 创新点**

创新点包括：① 生成多技能合成问题，将实体检索与量化推理结合；② 在生成流程中并行约束四大指标（可验证性、复杂度、难度、多样性）；③ 通过代码执行获得可验证的黄金答案；④ 支持即时实例化，避免静态答案泄露，提升抗污染能力。

**🔧 技术方法**

主要技术：answer‑first 生成管线；从 Wikidata 提取多跳 KG 链并结合领域特定 API 获取数值；使用 LLM 生成线索、构造问题；用参数化代码执行计算得出 gold answer；两阶段难度验证（关闭式与工具增强式）；最大最小嵌入过滤实现多样性；引入 Compositional Complexity Index（CCI）衡量复杂度。

**📊 数据集**

使用的数据集：Wikidata、Wikipedia、SEC EDGAR、NIST NVD CVE、PubChem、UniProt 等公开知识库；通过主题实体集合生成 354 条符合条件的问题，覆盖 129 个模板，涉及 1982 条实体链。

**📈 对比分析**

比较方法与性能：① 人类专家评测验证 76%（84% 去除过时数据）有效率；② 在 3 个前沿模型（Claude Opus 4.6、Gemini 2.5 Flash、GPT‑5.2）和 3 个开源模型（Llama 4 Maverick、Qwen3‑30B‑A3B、Mistral‑Small‑3.2‑24B）自动评测，最佳模型仅 20% 正确率；③ 随 CCI 上升准确率下降；④ 与手工基准（BrowseComp+, MATH‑500、GPQA）对比，DrBencher 在 Self‑BLEU 与多模型语义距离上均表现出最高的多样性。

**⚠️ 局限性**

局限性：① 生成与难度过滤均使用同一 LLM（gpt‑oss‑120b），可能对其它模型的难度评估产生偏差；② 系统性 hallucination 可能逃过验证；③ 依赖公开 KG，过时条目导致 35% 错误；④ 结果受单一模型能力谱影响，需进一步多模型验证。

---

## 365. 2D or 3D: Who Governs Salience in VLA Models? -- Tri-Stage Token Pruning Framework with Modality Salience Awareness

**arXiv ID:** 2604.09244 | [PDF](https://arxiv.org/pdf/2604.09244v1)

**作者:** Zihao Zheng `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并实现了一套针对多视觉模态 VLA（MVLA）模型的三阶段 token pruning 框架，利用 2D/3D 模态显著性分析在数据预处理、语义合成与动作迭代三个阶段动态剪枝，以提升推理速度并保持任务性能。

**💡 创新点**

创新点包括：①三阶段显著性分析与对应的模态感知阈值机制；②语义层聚类+模态阈值实现语义感知剪枝；③动作迭代阶段采用 EMA 滑动平均预测显著性动态；④跨阶段交叉融合决策保证剪枝既不丢失关键模态也不冗余保留；⑤通过这些方法实现了高达 2.55× 的加速且准确率下降不足 3%。

**🔧 技术方法**

使用的技术包括：视觉 Transformer 与点云编码器；LLM 推理与 diffusion 采样解码；token pruning；特征范数与注意力分数度量显著性；k‑means 聚类划分语义区域；EMA 滑动平均预测动态显著性；交叉融合（模态与语义）剪枝策略。

**📊 数据集**

实验使用了 RLBench 仿真任务（如 Close Box、Pick‑&‑Place 等）和真实 Songling Piper 机器人实验，评估基于开源 MLA 模型的剪枝效果。

**📈 对比分析**

对比方法包括未剪枝基线、随机剪枝（Naive Prune）、SP‑VLA、VLA‑Pruner 等。实验结果表明，在 50–80% 剪枝率下，本文框架实现 2.3×~2.55× 的速度提升，任务成功率仅下降 0.4%~2.5%；在真实任务中 2.3× 加速，成功率损失不足 5%。相较于其他基线，速度提升显著，准确率损失更小。

**⚠️ 局限性**

局限性在于需手动设定阈值与 EMA 参数，对极端动态变化的适应性可能受限；目前仅在 2D+3D 模态下验证，其他模态如音频不一定适用；显著性分析虽开销小，但仍需额外计算；对模型架构差异的通用性尚未完全验证。

---

## 366. Unreal Thinking: Chain-of-Thought Hijacking via Two-stage Backdoor

**arXiv ID:** 2604.09235 | [PDF](https://arxiv.org/pdf/2604.09235v1)

**作者:** Wenhan Chang `[一作]` (Zhongnan University of Economics and Law), Wanlei Zhou `[通讯]` (City University of Macau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了链式思维（CoT）劫持的两阶段后门攻击方法（TSBH），并通过反向树搜索（MRTS）构造与目标输出对齐的恶意CoT；

**💡 创新点**

创新点在于将CoT与输出同步的逆向合成技术与两阶段后门注入相结合，既能在触发时改变输出，又能操纵可观测的推理过程，实现更隐蔽、强效的攻击；

**🔧 技术方法**

核心技术包括LoRA参数高效微调、MRTS逆向合成、两阶段后门训练、欧氏距离匹配、以及多指标评估框架（CHR、ASR、GSM8K/MMLU精度）等；

**📊 数据集**

使用公开的Open‑Weight LLM模型（如DeepSeek‑V2/V3.2、GLM‑4.7、Kimi‑K2.5、MiniMax‑M2.5）及内部生成的恶意CoT数据；

**📈 对比分析**

与原始模型、全参数微调（FPFT）、LoRA微调（LRFT）、以及多种Jailbreak基线（H‑CoT、AutoRAN、Chain‑of‑Lure、CoT Hijacking）进行对比，实验表明TSBH在触发时具有较高的CHR/ASR，且在不触发时保持低失效率，且保持了较好的实用性能（GSM8K/MMLU准确率未显著下降）；

**⚠️ 局限性**

限制包括：需大量计算资源训练两阶段模型，逆向合成对模型选择敏感，后门攻击在极端安全策略下可能被检测，且对不同LLaMA/ChatGLM等模型的泛化性尚未完全验证。

---

## 367. The Fast Lane Hypothesis: Von Economo Neurons Implement a Biological Speed-Accuracy Tradeoff

**arXiv ID:** 2604.09229 | [PDF](https://arxiv.org/pdf/2604.09229v1)

**作者:** Esila Keskin `[一作]` `[通讯]` (University of the West of England), Esila Keskin (University of the West of England)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证Fast Lane Hypothesis，构建基于Von Economo神经元（VEN）的稀疏快速投射通路脉冲神经网络，用以解释社会决策速度差异。

**💡 创新点**

首次将VEN的生物学特征转化为计算模型，揭示其调节决策速度而非容量，并解释自闭症与前额叶纹状体退行性病变的临床表现。

**🔧 技术方法**

使用SpikingJelly框架实现可微分脉冲神经网络，采用surrogate‑gradient训练。

**📊 数据集**

使用人工生成的100维Poisson脉冲序列（二分类友好/威胁）作为任务数据，评估不同VEN比例与临床条件下的网络表现。

**📈 对比分析**

通过固定阈值θ=3和自适应阈值Δ=1比较典型、前额叶退化和自闭症等条件，结果显示典型最快（≈20 ms），退化最慢（≈32 ms），准确率均≈99%，体现速度‑准确性权衡。

**⚠️ 局限性**

局限在于任务过于简单、种子数有限导致自闭症条件差异不显著、未模拟逐步退化过程、未检验多模态长时序输入以及未使用真实社交刺激。

---

## 368. GRM: Utility-Aware Jailbreak Attacks on Audio LLMs via Gradient-Ratio Masking

**arXiv ID:** 2604.09222 | [PDF](https://arxiv.org/pdf/2604.09222v1)

**作者:** Yunqiang Wang `[一作]` (Sun Yat-Sen University), Guocong Quan `[通讯]` (Sun Yat-Sen University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种针对音频大语言模型的攻击框架 GRM，通过频率选择性扰动实现 jailbreak

**💡 创新点**

创新点在于使用梯度比率评分自适应选择关键 Mel 频带，并加入语义保持损失以平衡攻击成功与实用性

**🔧 技术方法**

技术包括：频域梯度分析、稀疏频带掩码、联合优化（生成目标前缀与语义保持）以及可复用的全局扰动学习

**📊 数据集**

数据集涵盖了自制的 AdvBench‑Audio（520 句子转语音）以及 LibriSpeech、AIR‑Bench‑Chat 用于评估语义与回复质量

**📈 对比分析**

与多种基线（文本转音频的 GCG、AutoDAN；音频原生的 BoN、AudioBench、SSJ）对比，GRM 在四个 ALLM 上平均 Jailbreak Success Rate 88.46%，WER 10.45%，RQS 6.33，兼具高攻击率与低语义损失

**⚠️ 局限性**

局限性包括：跨模型迁移效果有限、未在真实物理环境中验证、评估部分依赖 LLM 判别器，可能产生偏差

---

## 369. Beyond Segmentation: Structurally Informed Facade Parsing from Imperfect Images

**arXiv ID:** 2604.09260 | [PDF](https://arxiv.org/pdf/2604.09260v1)

**作者:** Maciej Janicki `[一作]` (Warsaw University Of Technology), Przemyslaw Musialski `[通讯]` (New Jersey Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 YOLOv8 训练中加入轻量化的对齐损失，促使同类立面元素在检测时呈现网格一致性

**💡 创新点**

创新点在于通过训练时的对齐正则化实现结构一致性，而非后处理或显式语法约束，且不改动推理流程

**🔧 技术方法**

采用 YOLOv8 单阶段检测器、对齐正则化损失、SVD 基准正则化指标以及 mAP@0.5 评估指标

**📊 数据集**

使用 CMP 立面数据集（689 份裁剪后的建筑图像）进行训练与测试

**📈 对比分析**

与未加对齐损失的 YOLOv8 基线对比，结果显示在中等权重（W≈0.5）下，SVD 正则化得分显著下降（结构更规整），mAP@0.5 仅略有下降，表现出良好折中

**⚠️ 局限性**

局限性包括：对齐权重与检测精度的折中、对单个或稀疏元素无效、对图像分辨率敏感、以及对严重遮挡区域无法恢复结构

---

## 370. BVH-Accelerated Ray Tracing for High-Frequency Electromagnetic Backscattering

**arXiv ID:** 2604.09243 | [PDF](https://arxiv.org/pdf/2604.09243v1)

**作者:** Marco Pasquale `[一作]` (KTH Royal Institute of Technology), Stefano Markidis `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种射击与反弹射线（SBR）方法，结合BVH加速与射线管离散化的物理光学积分，用于高频电磁散射和雷达截面预测

**💡 创新点**

将PO积分离散化为射线管并通过射线采样规则消除相位混叠，构造trace–integrate管线和MPI+GPU并行实现，显著提升大规模角度扫描效率

**🔧 技术方法**

射线追踪、BVH（Median/SAH分割）、物理光学积分、CUDA/HIP GPU加速、MPI分布式并行、射线管离散化

**📊 数据集**

PEC球体（用于与Mie解比较）和A380飞机模型（162 k三角面片）

**📈 对比分析**

与解析Mie解比较误差≤2.5%，10 GHz下A380扫描在8节点MI250X上每个角度≈616 ms，强/弱扩展率分别≈88%/96%

**⚠️ 局限性**

对低频波纹（Mie）无效；需要高射线密度导致计算量随频率增大；仅支持PEC且未包含衍射或介质效应

---

## 371. ScheMatiQ: From Research Question to Structured Data through Interactive Schema Discovery

**arXiv ID:** 2604.09237 | [PDF](https://arxiv.org/pdf/2604.09237v1)

**作者:** Shahar Levy `[一作]` (Hebrew University of Jerusalem), Gabriel Stanovsky `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了ScheMatiQ，一套基于LLM的交互式框架，能够根据自然语言研究问题从大型文档集合自动识别观察单元、生成针对该问题的结构化schema，并提取可追踪证据的结构化数据库；同时提供Web界面让专家实时编辑与完善。

**💡 创新点**

创新点在于：①将查询驱动的观察单元发现与schema迭代生成相结合，形成从问题到结构化数据的闭环；②利用LLM进行全流程自动化，同时保持输出的可验证证据与人机可编辑；③实现了面向领域专家的交互式人机协作工作流，显著降低传统手工标注的成本与错误率。

**🔧 技术方法**

核心技术包括：①使用Gemini‑2.5系列大语言模型进行观察单元识别、schema提炼与结构化提取；②Prompt工程化设计多轮LLM交互；③基于Web的可视化界面实现schema与数据的即时编辑与审计；④严格的证据绑定规则保证提取值的可追溯性。

**📊 数据集**

实验使用两组真实数据集：法律领域的89份美国移民案件判决书（人工标注了法官姓名、任命总统与裁决结果）；计算生物领域的NESdb（96篇科研论文，手工标注蛋白核出口信号及其强度与置信度）。

**📈 对比分析**

评估方法：将ScheMatiQ生成的schema与字段与已有人工gold schema进行覆盖率和相关性比较；在两领域分别计算召回率与误差分布。结果显示：法律领域召回率74%，生物领域召回率87%；字段覆盖率与人工标注高度一致，新增字段均被专家认为高度相关；误差主要集中在文档中出现多个观察单元时。

**⚠️ 局限性**

局限性包括：①依赖封闭的Gemini API，导致完全复现受限且存在非确定性；②模型在高密度文档中的召回仍有不足；③隐私政策仅允许可选记录，未完全保证所有数据的长期匿名化；④当前框架主要支持Gemini，若要使用其它LLM需额外配置。

---

## 372. Statistical Properties of the King Wen Sequence: An Anti-Habituation Structure That Does Not Improve Neural Network Training

**arXiv ID:** 2604.09234 | [PDF](https://arxiv.org/pdf/2604.09234v1)

**作者:** Augustin Chan `[一作]` `[通讯]` (Independent Researcher), Augustin Chan (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过蒙特卡洛置换分析验证王文序列的统计结构，并将其应用于学习率调度与课程学习排序，检验其是否能提升神经网络训练效果

**💡 创新点**

首次系统验证王文序列的四项统计显著特性，并给出其在梯度优化中造成性能下降的机制性解释

**🔧 技术方法**

蒙特卡洛置换分析、基于Hamming距离与信息惊奇的序列度量、学习率调度乘子、批次缓冲与重新排序、随机种子敏感性评估

**📊 数据集**

I‑易经六爻组合（64个二进制状态），ClimbMix‑400B 语言模型数据集，实验在 NVIDIA RTX 2060（PyTorch）与 Apple Silicon（MLX）上进行

**📈 对比分析**

与随机排列、Shao Yong 等基准序列比较；使用验证字节比（val_bpb）作为唯一指标；结果显示王文序列在学习率调度和课程学习中均未提升性能，甚至在某些实验中表现更差，且仅在学习率调度实验中出现的差异超过随机种子噪声阈值

**⚠️ 局限性**

实验规模有限（约1.15M参数模型，5 分钟训练），未验证在大规模模型或更长训练时间下的表现；框架特定效应（CUDA vs MLX）可能掩盖真实动态；仅使用单一数据集和模型架构，结果可能不具普适性

---

## 373. TinyNeRV: Compact Neural Video Representations via Capacity Scaling, Distillation, and Low-Precision Inference

**arXiv ID:** 2604.09220 | [PDF](https://arxiv.org/pdf/2604.09220v1)

**作者:** Muhammad Hannan Akhtar `[一作]` (American University of Sharjah), Tamer Shanableh `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究极小规模NeRV模型用于资源受限环境的高效视频表示与解码。

**💡 创新点**

提出两种极小化宽度的NeRV-T与NeRV-T+，并系统评估容量缩放、知识蒸馏（频域焦点）与低精度量化（PTQ/QAT）在极小模型上的协同效果。

**🔧 技术方法**

宽度缩放、频域焦点知识蒸馏、频域焦点对齐、时域差分蒸馏、特征层蒸馏、权重量化（INT8/INT6/INT4）与量化感知训练。

**📊 数据集**

单视频四个序列：Big Buck Bunny、honeybee、readysetgo、yachtride。

**📈 对比分析**

与原始NeRV-S/M/L进行对比，使用PSNR、MS-SSIM、T-PSNR、tSSIM、GFLOPs、FPS评估。结果表明NeRV-T/T+在极小容量下实现了显著的质量与吞吐量折中，INT8无显著损失，INT6需要QAT，INT4需QAT+蒸馏后才能保持质量，频域焦点蒸馏提升≈0.8–0.9 dB。

**⚠️ 局限性**

实验仅在单视频训练设置，未验证多视频或真实硬件的能耗与延迟；量化实验为软件仿真，缺乏硬件验证；对动态适应或在线更新的探讨有限。

---

## 374. CT-1: Vision-Language-Camera Models Transfer Spatial Reasoning Knowledge to Camera-Controllable Video Generation

**arXiv ID:** 2604.09201 | [PDF](https://arxiv.org/pdf/2604.09201v1)

**作者:** Haoyu Zhao `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种Vision–Language–Camera（VLC）模型CT‑1，用来自动生成与用户意图和场景内容相匹配的相机轨迹，并将其与视频扩散模型结合实现可控视频生成。

**💡 创新点**

1) 通过在视觉‑语言模块中注入相机上下文令牌，使模型能在语义层面推理相机运动；2) 将相机轨迹建模为分布，采用Diffusion Transformer并加入Wavelet频域正则化，提升轨迹平滑与物理合理性；3) 开发大规模CT‑200K VLC数据集，填补现有数据缺乏相机运动标注的空白。

**🔧 技术方法**

多模态Transformer（DINOv2、SigLIP、LLaMA‑2）、Diffusion Transformer、Haar小波变换正则化、相机轨迹分布学习、视频生成时的相机控制条件注入。

**📊 数据集**

自建CT‑200K数据集（包含约200K段视频，47M帧），并利用Pexels‑400K、DynPose‑100K、EgoSchema等公开视频资源进行预处理和描述生成。

**📈 对比分析**

在CameraBench100、RealEstate10K、MultiCamVideo等基准上，CT‑1在相机控制成功率上分别比最优VLM/AR轨迹估计模型提升171%/245%，比最佳Prompt‑based模型提升25–31%；在VBench视频质量指标上与现有扩散模型保持同等水平。

**⚠️ 局限性**

1) 仍需进一步提升在极端或复杂运动场景下的轨迹生成精度；2) 依赖大型Transformer和扩散模型，计算成本高，尤其是多模态预训练模型的资源占用；3) 现有频域正则化主要针对低频平滑，可能不足以处理高度细节化的相机扰动。

---

## 375. LatentFlowSR: High-Fidelity Audio Super-Resolution via Noise-Robust Latent Flow Matching

**arXiv ID:** 2604.09188 | [PDF](https://arxiv.org/pdf/2604.09188v1)

**作者:** Fei Liu `[一作]` (University of Science and Technology of China), Zhen-Hua Ling `[通讯]` (University of Science and Technology of China)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于流匹配的潜在空间音频超分辨率方法LatentFlowSR，能够在低频噪声鲁棒自编码器提取的连续潜在空间中完成高频细节恢复。

**💡 创新点**

创新点在于将条件流匹配（CFM）应用于潜在空间，利用噪声鲁棒自编码器构建连续潜在表示，并通过单步ODE求解实现高效、精细的高频恢复。

**🔧 技术方法**

采用噪声鲁棒自编码器、条件流匹配（CFM）与U‑Net风格的速度场估计网络，以及一阶Euler ODE求解器。

**📊 数据集**

在语音（VCTK）、音效（FSD50K/ESC‑50）和音乐（内部数据集/MUSDB18‑HQ）三大音频类型上进行训练与评测。

**📈 对比分析**

与传统波形、频谱及流基方法（NU‑Wave、AudioSR、FlashSR、FlowHigh等）对比，LatentFlowSR在LSD、LSD‑HF、ViSQOL等客观指标和MOS评测中均位列第一，且参数量与FLOPs显著低于基线。

**⚠️ 局限性**

局限性包括：目前仅针对单一降采样比例进行优化，缺乏对更广泛采样率变化的适应；对极长时序音频的处理仍可能受限；需要进一步验证在实时或低功耗环境中的部署可行性。

---

## 376. Facet-Level Tracing of Evidence Uncertainty and Hallucination in RAG

**arXiv ID:** 2604.09174 | [PDF](https://arxiv.org/pdf/2604.09174v1)

**作者:** Passant Elchafei `[一作]` (Johannes Kepler University Linz), Markus Schedl `[通讯]` (Johannes Kepler University Linz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了基于“facet”拆解的检索增强生成（RAG）诊断框架，通过将问题拆分为原子推理单元（facets），构建 Facet×Chunk 证据矩阵，并通过三种受控推理模式（Strict RAG、Soft RAG、LLM-only）评估检索与生成之间的对齐情况，系统性揭示检索-生成失配导致的幻觉。

**💡 创新点**

创新点在于：1) 引入 facet 级别诊断，将问题拆解为最小推理步骤；2) 通过结构化的 Facet×Chunk 矩阵结合检索相似度和 NLI 可信度量，提供细粒度的证据覆盖与真实性评估；3) 设计三种受控推理模式对比检索依赖与参数知识的交互，揭示“证据被覆盖（Override）”是幻觉主因。

**🔧 技术方法**

使用的技术包括：dense retriever（BGE‑base‑en‑v1.5）对文档进行向量检索；NLI（RoBERTa‑large‑MNLI）评估答案与检索片段之间的蕴含/矛盾关系；多模型推理（GPT‑4o‑mini、Gemini‑2.0‑Flash、LLaMA‑3‑8B‑Instruct）；Facet decomposition（使用 GPT‑4o‑mini 进行医学问题拆解）。

**📊 数据集**

使用的数据集有：HotpotQA（3,000 题，涵盖 easy/medium/hard 难度）和 RAGBench‑Medical（1,500 题，来自临床指南的医学问答）。

**📈 对比分析**

对比方法：在每种模型和每个数据集上分别执行 Strict RAG、Soft RAG、LLM-only 推理，计算 facet 级别的证据分类分布（Failure、Misalignment、Override、Helpful、Robust）以及聚合后的问题级别 F1、BERTScore。结果表明：1) Soft RAG 在大多数情况下显著优于 Strict RAG，尤其在复杂推理场景；2) LLM-only 在某些情形下甚至优于 Strict RAG，说明模型往往覆盖检索证据；3) 整体性能提升幅度因模型、数据集和难度而异，但核心发现是检索精度与答案可信度并不正相关。

**⚠️ 局限性**

局限性包括：仅做诊断，未提出改进检索或生成模型；依赖 facet 拆解与 NLI 模型，可能带来误差或偏差；实验仅覆盖两大数据集和三种 LLM，结果可能不适用于其他任务、语言或架构；未探讨交互式或在线部署场景。

---

## 377. Decoupling Vector Data and Index Storage for Space Efficiency

**arXiv ID:** 2604.09173 | [PDF](https://arxiv.org/pdf/2604.09173v1)

**作者:** Yuanming Ren `[一作]` (Chinese University of Hong Kong), Patrick P. C. Lee `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

本文提出了一种磁盘端近邻检索系统的解耦向量存储管理框架，通过将向量数据与索引元数据分离，提升空间效率和 I/O 性能。

**💡 创新点**

创新点包括：① 针对向量和索引分别设计无损压缩方案；② 引入分层（段/块）存储布局与块级索引；③ 采用延迟感知搜索与自适应预取；④ 采用分批写入与垃圾回收的更新策略，显著降低写放大。

**🔧 技术方法**

使用的技术有：基于 Huffman 的无损压缩、基于 Elias‑Fano 的邻居列表压缩、XOR 差分+熵编码、段/块分层存储、异步 I/O、预取与重排序、日志结构化写入与 GC。

**📊 数据集**

评估数据集包含公开的 SIFT1M、SIFT100M、SIFT1B、SPACEV100M、SPACEV1B，以及内部专有的 DecoupleVS100M。

**📈 对比分析**

与 DiskANN、PipeANN、SPANN 等主流磁盘端 ANNS 进行对比；在 100M 级别下，存储空间可节省最高 58.7%，查询吞吐量提升至 2.18×；在 10 亿级别下，存储节省约 42.6%，吞吐量与 PipeANN 相当或更高，延迟低于 DiskANN。

**⚠️ 局限性**

主要局限在于：写合并阶段仍需重写大量索引块，合并耗时略高；对已量化的高维数据集 Delta 压缩收益有限；在极大规模并发更新时 GC 与缓存一致性仍需进一步优化。

---

## 378. Automated Batch Distillation Process Simulation for a Large Hybrid Dataset for Deep Anomaly Detection

**arXiv ID:** 2604.09166 | [PDF](https://arxiv.org/pdf/2604.09166v1)

**作者:** Jennifer Werner `[一作]` (Fraunhofer Institute for Industrial Mathematics), Michael Bortz `[通讯]` (Fraunhofer Institute for Industrial Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个由实验批量蒸馏数据与自动生成的仿真数据组成的混合数据集，实现了从单一校准实验到118个实验的自动仿真生成。

**💡 创新点**

创新点在于将实验数据库的结构化异常注释映射为仿真控制信号，利用Python仿真器的索引归约技术实现大规模、自动、一致的仿真，并将仿真结果公开共享。

**🔧 技术方法**

使用了Python实现的批量蒸馏仿真器（基于非线性DAE索引归约），以及与实验数据库相同的异常标注语义和控制参数映射。

**📊 数据集**

使用的实验数据集为已发布的119次批量蒸馏实验（含71个已确认异常），并在此基础上生成对应的仿真时间序列。

**📈 对比分析**

通过与实验测量进行时序对比，验证仿真模型在温度、组分、流量等指标上与实验相符，异常预测保持趋势一致；在未校准实验中预测误差与校准实验相当。

**⚠️ 局限性**

局限在于仿真仅覆盖可由设点扰动表示的异常，无法模拟泡沫、传感器噪声等非控制源异常；仿真数据与实验数据在噪声、误差方面仍存在差异，需进一步做仿真到实验的风格迁移等处理。

---

## 379. EpiAgent: An Agent-Centric System for Ancient Inscription Restoration

**arXiv ID:** 2604.09367 | [PDF](https://arxiv.org/pdf/2604.09367v1)

**作者:** Shipeng Zhu `[一作]` (Southeast University), Hui Xue `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 EpiAgent，一套基于 LLM 的代理式系统，用于全流程恢复古碑铭刻图像，兼顾文字真实性与视觉美感。

**💡 创新点**

创新点包括：① 观察–构思–执行–再评估四阶段闭环决策，② 经验驱动的多工具动态调度，③ 结合多模态分析与多视角评估（文字、风格、专家反馈）的自我反思机制，③ 通过 LLM 规划器实现人类碑刻师工作流的自动化与可扩展化。

**🔧 技术方法**

使用技术：多模态大语言模型 (MLLM)、纠正语言模型 + RAG、布局校正与降解评估模块、可组合的专用修复工具（背景去噪、笔画补全、字体模仿、字符检索）、多视角评估（文字真实性、风格一致性、人工反馈）以及 LLM 规划器进行动态工具调用与迭代优化。

**📊 数据集**

数据集：CIRI（Chinese Inscription Rubbing Images）——24k 合成碑文图像（20k 训练，4k 测试）以及 2k 真实抄写图像，测试分为合成集 S、实测集 R‑I 与 R‑II。

**📈 对比分析**

对比方法：统一图像修复（Restormer、MambaIR、PromptIR、MoCE‑IR）、文本图像增强（CharFormer、GSDM、DocDiff）以及专门碑文修复基线 IR3。评估指标包括 PSNR、SSIM、LPIPS、CLIP‑IQA、MUSIQ、MANIQA、NIMA、1‑NED 以及 Top‑1/5/宏观准确率。实验结果显示 EpiAgent 在所有指标上均优于对手，并在专家用户研究中获得最高分数。

**⚠️ 局限性**

局限性：① 依赖大规模多模态模型与专用工具，推理成本和算力需求高；② 目前仅验证于汉字碑文，跨语言通用性待考；③ 仍需人工专家反馈以完善评价；④ 对极端缺损区域的完整字符重建仍具挑战；⑤ 对大尺寸、复杂布局碑文的可扩展性尚未充分评估。

---

## 380. LLM-Rosetta: A Hub-and-Spoke Intermediate Representation for Cross-Provider LLM API Translation

**arXiv ID:** 2604.09360 | [PDF](https://arxiv.org/pdf/2604.09360v1)

**作者:** Peng Ding `[一作]` `[通讯]` (University of Chicago), Peng Ding (University of Chicago)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个开源的 LLM API 翻译框架 llm-rosetta，提供一个中间表示（IR）以及基于 Ops‑composition 的转换器，实现四大主流 LLM 提供商（OpenAI Chat、OpenAI Responses、Anthropic Messages、Google GenAI）之间的双向、流式、无损转换，并在生产环境（Argonne National Laboratory）中部署。

**💡 创新点**

核心创新在于：① 将不同 LLM API 的共性抽象为 9 种内容部件和 10 种流事件的类型化 IR；② 采用 hub‑and‑spoke 的中间层降低 O(N²) 适配器成本为 O(N)；③ 将转换逻辑拆分为 ContentOps、MessageOps、ToolOps、ConfigOps 四个 orthogonal 模块，便于复用与单测；④ 支持双向、流式、状态化转换并实现 sub‑100 µs 的低延迟；⑤ 通过大量单元、流式与跨供应商测试验证无损回环，满足 Open Responses 合规性。

**🔧 技术方法**

技术实现基于 Python（3.10）和 TypedDict 进行类型安全，使用分层的 Ops‑composition 架构；利用 Starlette 开发 HTTP Gateway；通过自动检测器推断源格式；利用 StreamContext 处理状态化流事件；在 benchmark 中使用 time.perf_counter_ns() 进行微观延迟测量；对齐 Open Responses 合规性套件进行自动化测试。

**📊 数据集**

使用的“数据集”主要为：① 1,364 条自定义单元测试（覆盖请求、响应、流式、工具、配置等 4 供商各类情况）；② 1,364 条 Open Responses 合规性测试；③ 记录的实时 SSE 流式会话（来自四大供应商的真实流量）；④ 部署时的生产对话（Argo‑Proxy 的多轮对话与工具调用）；这些测试均来自官方文档、真实 API 响应或生产日志。

**📈 对比分析**

评估方法：① 通过 1,364 条单元测试验证回环（A→→A）和跨供应商（A→→B）无损性；② 用 255 条流式测试验证事件顺序与内容完整性；③ 与 LiteLLM 的单向 transform_request 进行性能对比，测得 llm‑rosetta 的中间层转换在 80 µs 内完成，P95 < 115 µs，单向比 LiteLLM 低 20‑30% 但在多轮/工具调用场景下约 1.6‑2.2×；④ 通过 Open Responses 合规性测试验证兼容性。整体性能表现可忽略不计（低于网络/推理延迟 0.01%）。

**⚠️ 局限性**

局限性包括：① 仅覆盖四大主流聊天 API，未支持嵌入、微调、批处理等非聊天接口；② 当供应商引入新字段或语义差异时需维护对应 Ops；③ 跨供应商转换会丢失无对应等价字段的特性（如 Anthropic 的 cache_control、Google 的 grounding_metadata）；④ 现有跨供应商测试覆盖面有限，未来可能需要更大规模的互通评估；⑤ 在极端高吞吐场景下，Gateway 可能因额外序列化/网络跳点导致略高延迟。

---

## 381. Mind the Gap Between Spatial Reasoning and Acting! Step-by-Step Evaluation of Agents With Spatial-Gym

**arXiv ID:** 2604.09338 | [PDF](https://arxiv.org/pdf/2604.09338v1)

**作者:** Lars Benedikt Kaesberg `[一作]`, Bela Gipp `[通讯]` (University of Göttingen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个Gymnasium环境Spatial‑Gym，用于在二维网格路径规划与规则约束下评估大型语言模型的空间推理能力，并实现一步一步交互式实验。

**💡 创新点**

将空间推理任务转化为可交互的顺序决策问题，提供自动验证与奖励机制，并支持回溯操作，突破传统单次输出评估的局限。

**🔧 技术方法**

利用Gymnasium构建MDP框架，使用大型语言模型（如GPT‑OSS 120B、Qwen 3等）在one‑shot、step‑by‑step、step‑by‑step+backtracking三种设置下进行推理与路径规划。

**📊 数据集**

使用SPaRC类的1,000道二维网格谜题（500训练/500测试），包含七类规则（点、缺口、石头、星星、三角形、聚合体、Ylops），并按难度等级划分。

**📈 对比分析**

与人类、随机走和A*基线对比；最佳模型GPT‑OSS 120B在Gym环境中实现16%解题率（人类98%），step‑by‑step对弱模型提升约5%，对强模型降低约5%；回溯对弱模型略有帮助，强模型无明显收益。

**⚠️ 局限性**

模型在空间约束推理上仍存在巨大缺口，难度提升时未能相应增加推理投入；视觉输入对性能有负面影响；强模型对回溯与全局规划不敏感。

---

## 382. From Frames to Events: Rethinking Evaluation in Human-Centric Video Anomaly Detection

**arXiv ID:** 2604.09327 | [PDF](https://arxiv.org/pdf/2604.09327v1)

**作者:** Narges Rashvand `[一作]` (University of North Carolina at Charlotte), Hamed Tabkhi `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了姿态基视频异常检测的事件级评估框架，并实现了端到端的双分支事件检测模型。

**💡 创新点**

创新点在于从传统帧级评估转向事件级评估，提出分层高斯平滑+自适应阈值的得分细化管道以及专门设计的双分支跨尺度融合网络。

**🔧 技术方法**

采用姿态序列重建、Transformer 编码-解码器、分层高斯平滑、阈值化、事件抽取、跨尺度融合与池化等技术。

**📊 数据集**

使用公开的姿态异常数据集 SHT、CHAD、HuVAD 和 NWPUC（对 SHT 做了事件清洗）。

**📈 对比分析**

在帧级指标上模型可达 AUC-ROC>60%，但在事件级 tIoU 与多阈值 F1 评估下，精度仅低于10%，表明帧级指标过高估计了实际性能。

**⚠️ 局限性**

限制在于事件级评估仍需阈值调优，且在高难度数据集上事件定位精度仍低，需要进一步提升模型对复杂动态异常的捕捉能力。

---

## 383. Structure-Aware Fine-Grained Gaussian Splatting for Expressive Avatar Reconstruction

**arXiv ID:** 2604.09324 | [PDF](https://arxiv.org/pdf/2604.09324v1)

**作者:** Yuze Su `[一作]` (Southeast University), Liang Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本论文提出了一种名为Structure-aware Fine-grained Gaussian Splatting (SFGS) 的方法，用于从单目视频中重建高保真、表达性强的人体全身3D数字化人偶；

**💡 创新点**

其创新点在于：①将空间三平面（Triplane）与时间三平面（Hexplane）相结合，实现静态与动态特征的统一编码；②引入结构感知的几何与颜色偏置模块，利用SMPL-X关节信息实现关节相关的位移与尺度校正；③针对手部采用MANO模型的残差细化模块，显著提升手部细节和非刚性变形；

**🔧 技术方法**

使用的技术包括：3D高斯分布表示（Gaussian Splatting）、三平面与六平面特征编码、轻量级MLP预测几何与颜色偏置、SMPL-X与MANO融合、差分渲染与多任务损失（RGB/SSIM/L2/LPIPS/Lab/梯度）等；

**📊 数据集**

实验数据集主要为NeuMan（6个单目视频序列）和X-Humans（多主体高质量扫描与RGB‑D序列），用来评估重建精度、手部细节与时序一致性；

**📈 对比分析**

在NeuMan和X-Humans数据集上，SFGS在PSNR、SSIM、LPIPS等指标上均优于目前最先进方法（如ExAvatar、GaussianAvatar等），同时实现约30 FPS的实时渲染；

**⚠️ 局限性**

局限性包括：对宽大或体积较大的身体部位点密度不足导致边缘模糊；对不同体型、服装的泛化能力有限；编辑流程缺乏直观的草图或文本交互支持。

---

## 384. The Need for a Green ICT Reference Framework

**arXiv ID:** 2604.09307 | [PDF](https://arxiv.org/pdf/2604.09307v1)

**作者:** Marco Aiello `[一作]` (Green ICT Working Group, Informatics Europe), Sebastian Werner `[通讯]` (Green ICT Working Group, Informatics Europe)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个多层次、跨域的 Green ICT 参考框架，并对其四个领域（基础设施、软件、数据、监控）与三层结构（概念、操作、治理）进行阐述。

**💡 创新点**

创新点在于首次将可持续性原则系统化为概念层，转化为可操作的行动与指标，并通过治理层与外部标准、政策对接，形成一个完整的、可追溯的跨层、跨域参考模型。

**🔧 技术方法**

采用理论框架设计方法，构建多层次结构，并结合现有可持续性指标（如 PUE、碳足迹计算器等）进行概念化与操作化的映射。

**📊 数据集**

未使用具体数据集，主要基于文献综述与专家共识构建框架。

**📈 对比分析**

未进行实验比较或性能评估，论文侧重概念与框架设计。

**⚠️ 局限性**

局限性：框架仍处于初步草案阶段，需要进一步细化概念、验证边界与责任归属；缺乏实证案例验证其可操作性与效果；跨层归因与标准化对接仍存在挑战。

---

## 385. The AI Codebase Maturity Model: From Assisted Coding to Self-Sustaining Systems

**arXiv ID:** 2604.09388 | [PDF](https://arxiv.org/pdf/2604.09388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 386. VAGNet: Vision-based accident anticipation with global features

**arXiv ID:** 2604.09305 | [PDF](https://arxiv.org/pdf/2604.09305v1)

**作者:** Vipooshan Vipulananthan `[一作]` (University of Moratuwa), Charith D. Chitraranjan `[通讯]` (University of Moratuwa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VAGNet，通过全局视频特征与 Transformer+图网络实现基于行车记录仪视频的事故预判。

**💡 创新点**

创新点在于利用 VideoMAE‑V2 提取全局时空特征，完全摒弃显式目标检测，并结合 Transformer Encoder 与 Graph Transformer 共同提升预测精度与效率。

**🔧 技术方法**

技术核心包括视觉基础模型 VideoMAE‑V2、Transformer Encoder、Graph Transformer、全连接层和交叉熵损失函数。

**📊 数据集**

使用四个基准数据集：DAD、DADA、DoTA 与 Nexar，重点评估 ego‑involved 事故。

**📈 对比分析**

与 UString、Graph(Graph)、AAT‑DA、STAGNet 等现有方法对比，VAGNet 在 AP 与 mTTA 指标上均取得领先，并实现约 97 FPS 的实时推理速度。

**⚠️ 局限性**

局限在于 VideoMAE‑V2 基础模型计算量大（约 102 GFLOPs/帧），需要进一步蒸馏或轻量化以满足更严格的实时部署需求。

---

## 387. Tracers for debugging and program exploration

**arXiv ID:** 2604.09301 | [PDF](https://arxiv.org/pdf/2604.09301v1)

**作者:** Shardul Chiplunkar `[一作]` (École Polytechnique Fédérale de Lausanne), Clément Pit-Claudel `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于完整执行轨迹的调试与程序探索工具（Tracer）

**💡 创新点**

将调试视角从语法上下文切换到时间上下文，并在轨迹中记录所有语句及其值，实现更直观的假设生成与验证

**🔧 技术方法**

利用运行时记录完整执行轨迹的自动化插桩/追踪技术，配合网页式 UI，支持查询、折叠、突出显示、书签等交互操作

**📊 数据集**

实验使用自制的小型 Python/Java 示例程序，未使用公开数据集

**📈 对比分析**

与传统步进器/日志器对比，强调在用户视角下的易用性与探索性；尚未给出量化性能指标，仅做粗略估算

**⚠️ 局限性**

主要局限包括执行开销与大规模程序的可扩展性、信息过载以及缺乏正式用户研究验证

---

## 388. A Catalog of Data Errors

**arXiv ID:** 2604.09277 | [PDF](https://arxiv.org/pdf/2604.09277v1)

**作者:** Divya Bhadauria `[一作]` (Hasso Plattner Institute, University of Potsdam), Lisa Ehrlinger `[通讯]` (Hasso Plattner Institute, University of Potsdam)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一份面向关系型数据库的数据错误与错误指示器的系统性目录，并为每种错误类型提供形式化定义、示例和分类维度。

**💡 创新点**

通过整合并统一已有的多种错误分类，解决术语不一致问题，并将错误细分为缺失、错误、冗余三大类，构建了完整的错误类型框架。

**🔧 技术方法**

采用形式化映射函数、约束检查机制、语义指示器等理论工具，对错误类型进行定义与分类，并综述了检测与修复方法。

**📊 数据集**

以示例数据库Employment为案例说明错误类型，未在真实大规模数据集上开展实验。

**📈 对比分析**

论文未进行实验评估，仅在文献综述中对比已有工具与方法；因此未给出性能指标或实验结果。

**⚠️ 局限性**

仅覆盖关系型数据库中的数据错误，未涉及图、文本等数据模式；缺乏实现细节与实证评估；方法综述性强而非实验验证。

---

## 389. Region-Constrained Group Relative Policy Optimization for Flow-Based Image Editing

**arXiv ID:** 2604.09386 | [PDF](https://arxiv.org/pdf/2604.09386v1)

**作者:** Zhuohan Ouyang `[一作]` (South China Normal University), Chaoqun Wang `[通讯]` (South China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出RC-GRPO-Editing框架，结合区域解耦扰动与注意力集中奖励，在确定性ODE流式编辑器上实现局部信用分配优化；

**💡 创新点**

创新点在于RDP仅在初始噪声层对编辑区域做局部扰动，显著降低背景噪声干扰，并引入ACD跨步关注度奖励引导跨层注意力聚焦编辑区域；

**🔧 技术方法**

使用GRPO强化学习、流式ODE采样、LoRA调优、文本-图像交叉注意力、以及自定义奖励函数；

**📊 数据集**

在CompBench三大子任务（Add/Remove/Replace）上进行评测；

**📈 对比分析**

与FLUX.1-Kontext-dev及InstructPix2Pix、Step1X-Edit、GoT等基线对比，RC-GRPO-Editing在LC-I、LC-T、PSNR、SSIM均取得最高或接近最佳分数，用户研究偏好率达34%；

**⚠️ 局限性**

限制在于训练需手工编辑掩码与访问交叉注意力权重，且RDP仅限制初始噪声层，无法完全保证整个推理过程中编辑区域外无影响。

---

## 390. Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios

**arXiv ID:** 2604.09377 | [PDF](https://arxiv.org/pdf/2604.09377v1)

**作者:** Hui Liu `[一作]` (City University of Hong Kong), Haoliang Li `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了多层任务概况引导的数据合成框架和TRouter，解决LLM路由的冷启动问题，能够在没有领域内训练数据的情况下生成合成QA对并训练路由器。

**💡 创新点**

创新点在于①通过三层任务分层（领域、子类别、难度）构建任务分类树，指导LLM合成多样化QA数据；②将任务类型作为潜在变量加入回归路由模型，并使用合成任务分类树的先验正则化，提升对任务语义的建模。

**🔧 技术方法**

技术包括LLM驱动的数据合成（任务类型生成、质量评估、QA对生成）、任务识别模块和多任务指标预测模块（MSE回归+交叉熵），以及变分推断求解带潜变量的条件分布。

**📊 数据集**

使用的基准数据集有Alpaca、GSM8K、SQuAD、Multi-News，实验中还额外加入四个任务；候选LLM池包括Qwen3系列大模型以及Gemini等商用模型，评估使用传统指标和LLM-as-a-Judge。

**📈 对比分析**

与多种基线（最小/最大LLM、Adaptive LLM、Prompt LLM、RouterDC、GraphRouter、MetricRouter、FrugalGPT、C2MAB-V）对比，TRouter在冷启动和领域内两种设置下在成本、性能和综合效用上均显著优于传统路由器，尤其在不同用户偏好（成本优先、平衡、性能优先）下表现最突出。

**⚠️ 局限性**

局限性包括需要人工提供初始领域或短文本来启动任务树；合成QA对未经过严格验证，可能存在噪声；LLM-as-a-Judge评估可能带来模型偏差，且实验使用的评估协议与实际应用可能不完全一致。

---

## 391. Robust 4D Visual Geometry Transformer with Uncertainty-Aware Priors

**arXiv ID:** 2604.09366 | [PDF](https://arxiv.org/pdf/2604.09366v1)

**作者:** Ying Zang `[一作]` (Huzhou University), Lanyun Zhu `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

基于预训练的VGGT模型，提出分层不确定性建模框架，实现4D场景的动态-静态分离与高质量重建。

**💡 创新点**

创新点在于三层不确定性机制：熵引导子空间投影、局部一致性几何净化、异方差投影一致性，统一提高动态检测与几何估计。

**🔧 技术方法**

使用Transformer跨视角注意力、熵权重聚合、点云局部密度过滤、基于高斯的异方差最大似然投影损失以及预训练VGGT。

**📊 数据集**

在DAVIS-2016进行动态分割评估，在DyCheck上评估重建误差和位姿误差。

**📈 对比分析**

相较于MonST3R、DAS3R、CUT3R、Easi3R和VGGT4D，本文在重建精度、完整度、位姿误差、Jaccard及边界F-Measure等指标上均获得显著提升，尤其在DyCheck上实现了0.0303的Accuracy Mean和0.1380的FM。

**⚠️ 局限性**

仍受限于对预训练模型的依赖，无法处理极端动态或缺失视角的情况，并且在极大规模序列中的计算成本仍需进一步优化。

---

## 392. A 0.5-V Linear Neuromorphic Voltage-to-Spike Encoder Using a Bulk-Driven Transconductor

**arXiv ID:** 2604.09315 | [PDF](https://arxiv.org/pdf/2604.09315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 393. Stochastic-Dimension Frozen Sampled Neural Network for High-Dimensional Gross-Pitaevskii Equations on Unbounded Domains

**arXiv ID:** 2604.09361 | [PDF](https://arxiv.org/pdf/2604.09361v1)

**作者:** Zhangyong Liang `[一作]` `[通讯]` (Tianjin University), Zhangyong Liang (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种随机维度冻结采样神经网络（SD‑FSNN），通过随机采样权重和维度，并在空间-时间上进行分离，实现对高维无界域Gross–Pitaevskii方程（GPE）的高效求解。

**💡 创新点**

创新点在于将无梯度随机估计与空间-时间分离相结合，构造了指数衰减、质量归一化和能量守恒的结构保持约束，使得方法在所有维度上保持无偏、维度无关且具备长时间稳定性。

**🔧 技术方法**

技术包括随机特征网络、随机维度子采样、无梯度随机化估计、空间-时间分离的ODE求解、SVD正交层、隐式辛积分以及质量投影等。

**📊 数据集**

使用了1‑3维标准GPE基准以及10、50、100、200、500、1000维的高维制造解（MMS）数据集，测试了不同交互强度β的场景。

**📈 对比分析**

与Hermite谱、TSSP、PINN、Causal PINN、ELM、SDGD、RS‑PINN、HTE、STDE等方法进行对比，SD‑FSNN在计算时间上维度无关、速度快4–5个数量级，误差低于10⁻⁵，能量和质量保持良好，长时间积分保持高精度。

**⚠️ 局限性**

局限性在于需足够多的随机基函数才能覆盖解流形，受Kolmogorov宽度限制；随机采样引入方差，需要权衡子采样大小；对强非线性（β≫1）场景需增大基数或采用方差减小技术。

---

## 394. Characterizing Lidar Range-Measurement Ambiguity due to Multiple Returns

**arXiv ID:** 2604.09282 | [PDF](https://arxiv.org/pdf/2604.09282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 395. Drift-Aware Online Dynamic Learning for Nonstationary Multivariate Time Series: Application to Sintering Quality Prediction

**arXiv ID:** 2604.09358 | [PDF](https://arxiv.org/pdf/2604.09358v1)

**作者:** Yumeng Zhao `[一作]` (Northeastern University), Xianpeng Wang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种漂移感知的在线动态学习框架 DA-MSDL，用于非平稳多变量时间序列预测。

**💡 创新点**

创新点在于将无监督 MMD 漂移检测与漂移严重程度引导的分层微调结合，并引入动态记忆队列和优先重放，形成主动检测‑适应‑预测的闭环。

**🔧 技术方法**

采用多尺度双分支卷积网络、MMD 漂移度量、门控特征融合、层级化学习率与损失权重、经验回放与数据增强等技术。

**📊 数据集**

使用铁矿渣烧结工厂的真实生产数据（2283 个样本）以及公共水处理厂数据集进行评估。

**📈 对比分析**

与 OB-ISSID、Ventingformer、Transformer、LSTM、GRU‑PLS 等基线比较，DA-MSDL 在 MSE、MAE、MAPE 和 NMSE 等指标上均显著优于基线，且误差波动更小。

**⚠️ 局限性**

局限在于对漂移阈值、窗口大小等超参数敏感，仍需在极端标签延迟或高噪声环境下进一步验证，模型规模和在线更新成本也需进一步优化。

---

## 396. VAG: Dual-Stream Video-Action Generation for Embodied Data Synthesis

**arXiv ID:** 2604.09330 | [PDF](https://arxiv.org/pdf/2604.09330v1)

**作者:** Xiaolei Lang `[一作]` (GigaAI), Zheng Zhu `[通讯]` (GigaAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 VAG（Video‑Action Generation）双流生成框架，能够在同一次前向推理中同步生成与语言提示对应的高质量视频与动作序列，用于机器人数据合成与策略学习。

**💡 创新点**

创新点在于：①基于流匹配的统一双流架构，使视频分支与动作分支同步去噪；②使用自适应 3D 池化将视频潜在映射为全局上下文，精准对齐视频与动作；③一次性生成视频‑动作对，避免两阶段方法的误差累积与效率低下。

**🔧 技术方法**

技术手段包括：流匹配（Flow Matching）训练框架；Diffusion Transformer（DiT）视频分支；1D U‑Net 动作分支；自适应 3D 池化；T5‑XXL 文本编码；Cosmos‑Predict2 预训练；VAE 视觉分词器；Classifier‑free 指导；以及跨模态文本‑视觉对齐。

**📊 数据集**

使用的数据集有：AgiBot（约 1M 轨迹，217 个任务）、LIBERO（模拟数据的 spatial、object、goal、long 子集）以及作者自采集的 Agilex Cobot Magic 双臂机器人数据。

**📈 对比分析**

与 SVD、Wan2.2 等主流视频模型以及 ResNet/AnyPos 两阶段动作回归基线比较，VAG 在 FVD、FID、LPIPS、SSIM、PSNR 等视频指标上均表现更好；在动作欧氏距离和成功率上，VAG 优于两阶段方案；在仿真与实测任务中，VAG 生成的轨迹成功率提升至 70% 以上；利用 VAG 合成数据预训练 VLA，任务成功率从 35% 提升到 55%。

**⚠️ 局限性**

局限性包括：①视频分支不受动作分支的反馈，导致两者对齐不够精细；②动作分支使用 1D U‑Net，容量有限；③在更大规模、多样化任务上的验证仍待扩展；未来计划加入动作引导视频、提升动作分支模型容量并扩大训练数据。

---

## 397. Meta-Learned Basis Adaptation for Parametric Linear PDEs

**arXiv ID:** 2604.09289 | [PDF](https://arxiv.org/pdf/2604.09289v1)

**作者:** Vikas Dwivedi `[一作]` (Université Lyon 1), Bruno Sixou `[通讯]` (Université Lyon 1)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种混合物理信息框架，结合元学习预测器和一次性最小二乘校正器，用于快速求解参数化线性PDE族。

**💡 创新点**

创新点在于将基函数几何自适应交由元学习器完成，并把预测器生成的基函数直接传递给PIELM校正器，实现一次推断即可完成基函数适配与解的校正。

**🔧 技术方法**

采用浅层核自适应物理信息元学习器（KAPI）、基于高斯核的PIELM最小二乘求解以及物理损失与基函数正则化等技术。

**📊 数据集**

使用四个合成线性PDE族的数据集，分别覆盖二维Poisson、常系数输运、输运‑扩散混合和可变速度输运，参数在低维空间内随机采样。

**📈 对比分析**

与参数化PINN、DeepONet、统一网格PIELM等基线对比，KAPI+校正器在预测器误差基础上提升一至两阶，整体相对L2误差降至10⁻³级，优于单实例PINN且无需重训练。

**⚠️ 局限性**

局限在于仅适用于线性、低维参数化PDE；浅层高斯核结构在高度非线性或多尺度场景下可能欠拟合，且校正器的性能高度依赖预测器生成的基函数质量。

---

## 398. Toward an Architectural Blueprint to Observe Sustainability in and by Software Systems

**arXiv ID:** 2604.09278 | [PDF](https://arxiv.org/pdf/2604.09278v1)

**作者:** Klervie Toczé `[一作]` (Vrije Universiteit Amsterdam), Patricia Lago `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种面向可持续性的可观测性架构蓝图及其实现代码，并通过两个真实案例演示其应用

**💡 创新点**

构建了可插拔、可组合、面向中级技术人员的可观测性栈，弥补了现有工具缺乏可重复使用与跨域部署指导的空白

**🔧 技术方法**

利用Prometheus、Grafana、OpenTelemetry、Exporter等开源组件，构成采集‑聚合‑处理‑可视化四层管道，并支持能耗与其他可持续性指标

**📊 数据集**

案例数据：Feed4Food项目中人工与设备采集的农业传感器与能耗日志；GreenLab项目中集群节点CPU/内存/功耗等自动导出数据

**📈 对比分析**

该工作未做系统的定量对比实验，仅通过案例展示实现效果，强调可观测性栈的部署易用性与可视化灵活性；对性能影响提出了讨论与缓解思路

**⚠️ 局限性**

缺乏系统性验证与可用性评估；在分布式环境下采集开销与安全、隐私风险未彻底解决；数据粒度与聚合折衷仍需进一步研究

---

## 399. Bringing Clustering to MLL: Weakly-Supervised Clustering for Partial Multi-Label Learning

**arXiv ID:** 2604.09359 | [PDF](https://arxiv.org/pdf/2604.09359v1)

**作者:** Yu Chen `[一作]` (Guangdong University of Technology), Fang Li `[通讯]` (Guangdong University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种弱监督聚类框架 WSC-PML，用于解决部分多标签学习（PML）中的标签噪声问题。

**💡 创新点**

创新点在于将聚类隶属矩阵分解为 Π⊙F，以兼顾聚类约束和多标签特性，并通过置信度加权的弱监督信号实现噪声识别与聚类协同。

**🔧 技术方法**

使用了聚类原型学习、置信度机制、弱监督聚类、交替优化、神经网络分类等技术。

**📊 数据集**

在 24 个数据集上实验，包含 6 个真实世界 PML 数据集（Mirflickr、Music_emotion、Music_style、YeastBP、YeastCC、YeastMF）以及 18 个由多标签数据生成的合成 PML 数据集（emotions、birds、medical、image、yeast、corel5k）。

**📈 对比分析**

与六种基准方法（FBD-PML、PML-LENFN、NLR、PAMB、PML-NI、PML-fp）比较，平均精度、排名损失等指标上均取得最佳或最接近最佳表现，Friedman/Nemenyi 检验显示显著优势。

**⚠️ 局限性**

局限性包括：主要针对误正噪声的假设，且对超参数（α、β）的选择仍有一定敏感性，未来需扩展至更复杂的噪声模式。

---

## 400. LuMon: A Comprehensive Benchmark and Development Suite with Novel Datasets for Lunar Monocular Depth Estimation

**arXiv ID:** 2604.09352 | [PDF](https://arxiv.org/pdf/2604.09352v1)

**作者:** Aytaç Sekmen `[一作]`, Sinan Kalkan `[通讯]` (Middle East Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对月球探测任务，构建了 LuMon 框架，对 14 种前沿单目深度估计模型进行零射击评估，并在真实月球影像和暗色地形模拟数据上测试其性能。

**💡 创新点**

创新点在于首次提供完整的月球深度估计基准，包含真实 Chang'e-3 任务的双目深度真值、暗色地形模拟数据以及多种合成、地球类比与真实月球场景；同时提出了 LoRA 轻量化 fine‑tune 基线与姿态估计下游验证。

**🔧 技术方法**

使用深度学习框架（ViT、Diffusion、MonoDepth 等）进行零射击推断，并通过低秩适配（LoRA）在合成数据上微调；结合 MADPose + MASt3R 进行相对姿态估计。

**📊 数据集**

数据集涵盖六类：Unity 与 Unreal 生成的合成序列、地球火山类比数据、暗色月球类比 Cheri、真实 Chang'e-3 双目影像，以及 LiDAR 真实地球类比等。

**📈 对比分析**

对所有模型在相同协议下使用 δ₁、AbsRel、RMSE 评估，结果表明度量基准模型（如 ViT‑L、Mono‑Large）在零射击下显著优于相对模型；LoRA 微调后在合成场景的精度提升显著，但在真实月球影像上的改进有限。

**⚠️ 局限性**

主要限制包括：合成到真实的迁移性能不足，模型对光照极端和无纹理地表的鲁棒性差；对相机校准边缘敏感；以及在空间硬件的低功耗/低显存环境下的计算效率瓶颈。

---

## 401. Decentralized Opinion-Integrated Decision making at Unsignalized Intersections via Signed Networks

**arXiv ID:** 2604.09351 | [PDF](https://arxiv.org/pdf/2604.09351v1)

**作者:** Bhaskar Varma `[一作]` (Free University of Bozen-Bolzano), Paolo Falcone `[通讯]` (Chalmers University of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于有符号网络意见动态的分散式决策框架，用以在无信号交叉口上实现自动驾驶车辆的安全通过；

**💡 创新点**

创新点包括：①双重有符号图结构——冲突拓扑网络与动态信念网络协同驱动车辆从“GO”到“YIELD”决策；②预测可行性门与单步速度优化的分解式协同控制，避免集中式MPC求解；③意见动态中对冲突、许可、协同三通道的阈值调节与权重分配，确保安全与效率并存；

**🔧 技术方法**

技术主要涉及：有符号网络非线性意见动态、区间注意力调节、预测可行性检查、单步多目标速度优化（加速度规划）以及V2V广播协调；

**📊 数据集**

使用仿真数据：在MATLAB中生成81种4车（4CAV）四路交叉口的组合意图场景（左转、右转、直行混合），未使用公开交通数据集；

**📈 对比分析**

与传统FCFS（先到先服务）基线对比，实验表明在大多数场景下平均离场时间与FCFS相近或略优，尤其在混合冲突场景中显著提升了最后车辆离场时间，且完全无需集中式调度器；

**⚠️ 局限性**

局限性包括：仅在低密度（4车）理想仿真环境验证；缺乏真实道路噪声、感知误差、V2V通信失真等不确定性；对高密度车辆群体和混合人车流的适应性尚未验证。

---

## 402. EYWA: Elastic Load-Balancing and High-Availability Wired Virtual Network Architecture

**arXiv ID:** 2604.09322 | [PDF](https://arxiv.org/pdf/2604.09322v1)

**作者:** Wookjae Jeong `[一作]`, Jungin Jung `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了EYWA虚拟网络架构，解决多租户云环境中的高可用、负载均衡和大规模Layer‑2语义问题，支持数百万租户与单个大IP子网；

**💡 创新点**

创新点在于通过分布式轻量级agent实现共享SNAT/DNAT、代理ARP以及多实例VR共享同一私网IP，实现无单点瓶颈、支持VM迁移与动态扩容；

**🔧 技术方法**

采用VxLAN 24‑bit ID、MVRRP、代理ARP、HAProxy层4负载均衡、外部DNS负载均衡等现有技术构建分布式控制与数据平面；

**📊 数据集**

实验使用10台服务器、1Gbps 交换机、HAProxy、OpenStack等开源组件进行北南与东西向吞吐量测试；

**📈 对比分析**

与单路由器或传统DVR模型对比，EYWA实现了全链路吞吐、线性可扩展性，北南与东西向通信无吞吐瓶颈，实验结果显示吞吐量等于物理链路容量；

**⚠️ 局限性**

局限在于仅在小规模实验环境验证，未覆盖大规模生产环境、私有网络性能、长期可靠性与安全性评估；

---

## 403. Online Intention Prediction via Control-Informed Learning

**arXiv ID:** 2604.09303 | [PDF](https://arxiv.org/pdf/2604.09303v1)

**作者:** Tianyu Zhou `[一作]` (Purdue University), Shaoshuai Mou `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种在线意图预测框架，能够在目标随时间变化且系统动力学与目标参数未知的情况下，实时估计自主系统的目标状态。

**💡 创新点**

创新点在于：①将目标视为连续参数而非离散集合；②使用滑动窗口（shifting horizon）策略抑制旧信息以跟踪时间变化的意图；③结合控制信息的在线学习（OCIL）与Pontryagin可微程序，实时更新目标与未知参数。

**🔧 技术方法**

主要技术包括逆最优控制/逆强化学习、Pontryagin可微编程、基于梯度的在线参数估计（类似EKF更新）以及滑动窗口预测。

**📊 数据集**

使用仿真（随机噪声水平下的100次实验）和真实硬件（在5 m × 3 m区域内的四旋翼机，配备运动捕捉系统）进行验证；未使用公开数据集。

**📈 对比分析**

与传统OCIL方法对比，展示了更低的预测损失和更快的收敛速度；平均预测时延约为60 ms（仿真）/92 ms（硬件），低于测量周期，证明了实时性。

**⚠️ 局限性**

局限性包括：仅处理全状态、低噪声测量；对部分或非线性观测的适应性有限；需要进一步研究更复杂环境与更大范围的目标变化。

---

## 404. SkillMOO: Multi-Objective Optimization of Agent Skills for Software Engineering

**arXiv ID:** 2604.09297 | [PDF](https://arxiv.org/pdf/2604.09297v1)

**作者:** Jingzhi Gong `[一作]` (King's College London), Jie M. Zhang `[通讯]` (King's College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SkillMOO 框架，通过 LLM 提出的编辑和 NSGA-II 存活选择，自动进化针对软件工程任务的技能包，以提升 LLM 代码生成的成功率和效率。

**💡 创新点**

创新点在于将 LLM 生成的技能编辑与多目标进化搜索相结合，实现自动化、任务特定的技能优化，并通过模式分析揭示 pruning 与 substitution 对性能提升的关键作用。

**🔧 技术方法**

采用的大技术包括大型语言模型（如 ChatGPT）、两代理迭代（solver + optimizer）、NSGA-II 多目标进化算法以及 LLM 生成的编辑操作。

**📊 数据集**

使用的数据集为 SkillsBench 中的三项软件工程任务（已通过 GPT-5.4 扩展至 40 条测试），并结合原始技能包与无技能基线进行对比。

**📈 对比分析**

实验方法为在每个任务上执行 10 次评估，比较 pass 率、成本和运行时；结果显示 SkillMOO 在所有任务上将 pass 率提升 2.1%–131.2%，成本降低 5.4%–31.7%，且优化开销极低。

**⚠️ 局限性**

局限性包括仅测试三项任务、只使用单一 LLM 模型、测试扩展可能引入偏差、编辑模式分析仅为观察性而非因果推断，以及缺乏对更小任务或不同 LLM 的泛化验证。

---

## 405. Cross-Paradigm Models of Restricted Syndrome Decoding with Application to CROSS

**arXiv ID:** 2604.09292 | [PDF](https://arxiv.org/pdf/2604.09292v1)

**作者:** Étienne Burle `[一作]` (University of Luxembourg), Aleksei Udovenko `[通讯]` (University of Luxembourg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文通过构造特殊的“光正则”向量，将限制症状解（ResSD）问题降低到常规症状解和格子搜索问题，进而针对CROSS签名方案提出了一系列新的攻击方法。

**💡 创新点**

创新点在于：①首次给出ResSD到常规SD的精确映射；②提出基于仿射直径和截断的紧凑格子化技术；③结合批量CVP和列表CVP实现了新的时/存折中折算；④系统评估了这些攻击对CROSS实例的影响。

**🔧 技术方法**

采用的技术包括信息集解（ISD）变体、光正则向量构造、格子构造与LLL/BKZ约简、批量最邻近向量问题（Batch‑CVP）以及基于截断与仿射变换的概率性格子化。

**📊 数据集**

主要数据集为CROSS签名方案的实例参数（n,k分别为127/76、187/111、251/150），以及对其缩小版（如n=35,k=21,z=4）的实验验证。

**📈 对比分析**

通过与现有ISD、列表CVP、列表SVP等攻击进行比对，实验显示所提攻击在时间/存储上取得了若干折中点（例如最佳时间约为2^1.8n），但整体复杂度仍高于CROSS安全阈值，未突破其安全性。

**⚠️ 局限性**

局限性在于：①降维后的格子维数仍过高，导致格子搜索成本巨大；②多为启发式估计，缺乏严格上界；③对特定参数（如z=7）效果有限，未能给出实质性威胁；④对截断与仿射变换的依赖使得攻击的成功率受参数分布限制。

---

## 406. Are Independently Estimated View Uncertainties Comparable? Unified Routing for Trusted Multi-View Classification

**arXiv ID:** 2604.09288 | [PDF](https://arxiv.org/pdf/2604.09288v1)

**作者:** Yilin Zhang `[一作]` (Xidian University), Wei Zhao `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种统一路由的可信多视图学习框架TMUR，用全局路由器在全局上下文中分配专家权重，从而实现对视图证据的公平融合。

**💡 创新点**

创新点在于把视图证据提取与融合仲裁解耦，采用统一路由器基于全局上下文而非单视图不确定性进行权重分配，并引入软负载均衡与专家多样性正则化。

**🔧 技术方法**

核心技术包括基于Dirichlet的证据学习、跨视图注意力交互、统一路由网络、软负载均衡正则、专家多样性正则以及多任务损失组合。

**📊 数据集**

使用14个公开多视图分类数据集，包括HandWritten、Scene、LandUse、NUS、Caltech-6V、PIE、WebKB、UCI、CUB、Animal、MSRCV1、BBC、Leaves等。

**📈 对比分析**

与15个最新可信与非可信基线（如TMC、TMDLO、ETMC、RCML、FUML、RCMCL、TMCEK、TUNED、TEF、SAEML、RTMC、NLC、MAMC、BCM等）对比，TMUR在大多数数据集上取得最佳或相当的分类准确率，并将ECE平均下降约8个百分点，显著提升了预测准确性与置信度可靠性。

**⚠️ 局限性**

局限性包括对超参数β、γ等正则权重的敏感性，路由网络需要额外计算开销，且在极端视图不平衡或缺失情况下的鲁棒性仍待进一步验证。

---

## 407. SAGE: A Service Agent Graph-guided Evaluation Benchmark

**arXiv ID:** 2604.09285 | [PDF](https://arxiv.org/pdf/2604.09285v1)

**作者:** Ling Shi `[一作]` (Tianjin University), Deiyi Xiong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SAGE——一种基于动态图和多代理的客户服务 LLM 评估基准，能动态验证 SOP 合规性与对话质量；

**💡 创新点**

创新点包括：将非结构化 SOP 转化为动态对话图，实现双维度（逻辑合规+对话质量）评估；引入 Judge Agent 与 Rule Engine 的协同判定；使用 Adversarial Intent Taxonomy 生成多样化对抗用户；提供模块化扩展机制，支持低成本场景迁移；

**🔧 技术方法**

技术手段包括：图结构建模（有向图表示 SOP）、多代理协同评估、Judge Agent 集成判定、Rule Engine 确定唯一路径与动作、动态用户代理生成对话、对抗意图分类、模块化配置与数据合成；

**📊 数据集**

使用数据集：六个工业场景（电商退款、物流、套餐、物业、航空、在线教育）对应的 SOP 图和自动生成的多轮对话轨迹；涵盖零、弱、强三种对抗强度；

**📈 对比分析**

评估方法：对 27 种闭源与开源 LLM 进行 6 场景的全流程评测，计算整体评估分、逻辑合规分、对话质量分、格式错误率、聊天长度等指标；实验显示闭源模型仍优先，但如 DeepSeek‑V3.2 等开源模型已逼近，揭示“执行差距”和“共情韧性”现象，且性能随对抗强度与对话深度呈 Inverted‑U 趋势；

**⚠️ 局限性**

局限性：仅支持文本交互；图结构单层，未覆盖嵌套子图；缺少多模态评估；评估依赖自动 Judge 可能存在误判；长上下文性能仍显著衰减；对抗场景设计仍需人工；扩展到更复杂业务场景尚待验证。

---

## 408. GeRM: A Generative Rendering Model From Physically Realistic to Photorealistic

**arXiv ID:** 2604.09304 | [PDF](https://arxiv.org/pdf/2604.09304v1)

**作者:** Jiayuan Lu `[一作]` (State Key Lab of CAD&CG, Zhejiang University and Zhejiang Lab), Rui Wang. Yuchi Huo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出GeRM模型，通过将物理渲染(PBR)图像转化为高质量的可控光照与材质真实的光照图像，实现物理真实到照片真实的连续过渡。

**💡 创新点**

创新点在于将P2P转化建模为分布转移向量场，并构建P2P-50K专家指导的对齐数据集，利用多条件ControlNet学习分布转移，同时融合文本提示与图像感知增强。

**🔧 技术方法**

采用G-buffer特征、视觉语言模型、多智能体VLM框架、ControlNet结构以及分布转移向量场学习技术。

**📊 数据集**

使用了P2P-50K对齐数据集（由专家指导的PBR–PRR配对样本构成）以及现有的渲染、编辑、重光照基准数据。

**📈 对比分析**

在PBR图像合成、图像编辑与重光照任务中与现有基线对比，GeRM在图像质量指标和人类评估中均表现更优，达到更高的光照真实性与可控性。

**⚠️ 局限性**

局限性包括对细微文本描述的依赖导致对齐困难、对极复杂场景的泛化受限以及计算开销较大。

---

## 409. Multimodal Anomaly Detection for Human-Robot Interaction

**arXiv ID:** 2604.09326 | [PDF](https://arxiv.org/pdf/2604.09326v1)

**作者:** Guilherme Ribeiro `[一作]` (University of Lisbon), Nuno Cruz Garcia `[通讯]` (University of Lisbon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了MADRI框架，利用视频、机器人传感器和场景图的多模态特征在特征空间进行重构异常检测；

**💡 创新点**

首次将视觉流映射为语义特征向量并与关节扭矩、场景图融合，证明多模态特征重构比单模态像素重构更能捕捉人机交互中的异常；

**🔧 技术方法**

使用预训练Swin3D提取视觉特征，拼接关节扭矩和场景图矩阵，随后通过线性+BN+ReLU+Dropout的自编码器进行特征重构，并阈值判定异常；

**📊 数据集**

自采集的“Pick‑and‑Place”人机交互数据集，包含72段视频、对应ROS传感器记录以及17个异常案例；

**📈 对比分析**

与仅基于视频重构的基线做ROC对比，结果显示加入关节扭矩模态显著提升AUC，场景图单独使用效果不佳，多模态模型表现最佳；

**⚠️ 局限性**

场景图生成精度低导致其贡献受限；数据集规模和任务多样性有限，且未实现实时在线检测与细粒度异常分类。

---

## 410. BadSkill: Backdoor Attacks on Agent Skills via Model-in-Skill Poisoning

**arXiv ID:** 2604.09378 | [PDF](https://arxiv.org/pdf/2604.09378v1)

**作者:** Guiyao Tie `[一作]` (Huazhong University of Science and Technology), Lichao Sun `[通讯]` (Lehigh University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了代理生态中模型嵌入技能的后门攻击，提出了 BadSkill 框架并在模拟环境中验证；

**💡 创新点**

创新点在于设计了基于结构化技能参数的组合触发后门、引入硬负样本和多项式损失以提升稀疏触发样本的学习，并揭示了模型携带技能的供应链风险；

**🔧 技术方法**

采用组合交叉熵、margin loss 与 poison loss 的三项损失共同训练二分类器，并在 OpenClaw 类似的轻量仿真环境中进行评估；

**📊 数据集**

使用了 13 个技能（8 个触发技能 + 5 个控制技能）的数据集，包含 571 条负类查询与 396 条触发对齐查询，覆盖 8 种不同规模模型；

**📈 对比分析**

通过 BA（benign accuracy）和 ASR（attack success rate）评估，在 8 种模型（494M–7.1B）上平均 ASR 97.5%–99.5%，BA 下降 ≤4.2；3% 毒化率即可达到 91.7% ASR，毒化率 1%–7% 迅速提升并基本饱和；在多种扰动下保持高效；

**⚠️ 局限性**

局限性包括：仅评估到 7.1B 参数，使用轻量仿真环境而非多生产堆栈；未包含防御评估；触发仅限英文；payload 简单且未涵盖更广泛场景；未研究更大规模或多语言、多生态环境的可迁移性。

---

## 411. Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts

**arXiv ID:** 2604.09364 | [PDF](https://arxiv.org/pdf/2604.09364v1)

**作者:** Farhad Nooralahzadeh `[一作]` (Zurich University of Applied Sciences), Kurt Stockinger `[通讯]` (Zurich University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并诊断了视觉语言模型（VLM）在面对视觉与语言先验冲突时的错误机制，揭示了“编码–地面化解耦”，证明模型能正确编码视觉信息但在决策阶段被语言先验所支配，并提出基于MAC层的轻量化激活干预方法以提升视觉地面化效果。

**💡 创新点**

首次系统地将多模态仲裁交叉（MAC）分析、全序列激活补丁、以及稀疏自编码器驱动的干预结合起来，揭示了视觉信息在VLM内部的分布与仲裁过程的因果关系，并提供训练无关的推理时改进方案。

**🔧 技术方法**

使用了Logit Lens层级探针、全序列激活补丁、稀疏自编码器（SAE）特征选择与残差补偿、线性激活对齐等技术，辅以多模型多尺度实验验证。

**📊 数据集**

基于Synthetic Counterfactual图像数据集Visual‑Counterfact（针对颜色与尺寸的视觉对比实验），并在10个不同规模的VLM上进行评估。

**📈 对比分析**

通过在10个VLM（7B–72B）上与MAC层交叉、激活补丁与干预方法比较，发现视觉信息在早层即可线性可解（AUC>0.86），但最终层的logit差距才是预测正确率的关键；在3个主模型上，早层线性/SAE干预可提升视觉地面化准确率1.4%–3.8%，无性能退化。

**⚠️ 局限性**

实验仅覆盖合成对照图像，未考察自然场景下的视觉语言冲突；模型规模、层深与跨语义一致性差异可能影响结果；仅在7–8B模型上验证干预效果，未扩展到更大模型；激活补丁样本量有限，需进一步统计检验。

---

## 412. Visually-Guided Policy Optimization for Multimodal Reasoning

**arXiv ID:** 2604.09349 | [PDF](https://arxiv.org/pdf/2604.09349v1)

**作者:** Zengbin Wang `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Visually‑Guided Policy Optimization (VGPO)，通过内部隐藏状态定位视觉关注并在强化学习中强化视觉激活，解决 VLM 在多模态推理中视觉稀疏和视觉遗忘问题。

**💡 创新点**

创新点在于利用隐藏状态相似度自动生成视觉关注分数，设计视觉注意补偿机制以及双层优势重新加权，既提升视觉激活又抑制随推理步骤衰减的视觉遗忘。

**🔧 技术方法**

核心技术包括视觉关注分数计算（基于隐藏状态相似度）、视觉注意补偿（线性提升后期视觉期望）和双层优势重新加权（轨迹内外加权），并在 GRPO 框架下实现。

**📊 数据集**

实验使用 Qwen2.5‑VL‑3B/7B/32B 模型，在 ViRL39K、Geo3K、MMK12 训练集以及 MMK12‑val 验证集，评估 MathVista、MathVerse、We‑Math、MMK12、GeoMath、Geometry3K 等数学推理基准和 LogicVista、SuperClevr Counting、MMMU‑Pro、MathVerse‑V 等视觉依赖多模态推理基准。

**📈 对比分析**

与 ThinkLite‑VL‑7B、VL‑Rethinker‑7B、MMEureka‑7B、NoisyRollout‑7B、PAPO_D‑7B、VPPO‑RL‑7B 等基线对比，VGPO 在数学推理任务上提升 33.2% 以上，在视觉依赖任务上提升 30.0% 以上，整体平均准确率达 66.6%（数学）和 63.3%（视觉），与同基线的 72B 规模模型竞争力相近。

**⚠️ 局限性**

局限在于补偿策略为经验性线性增益，可能在最终推理步骤不需要视觉信息时过度强调视觉；若基础视觉编码器未能充分捕捉关键信息，VGPO 只能在现有特征上优化，难以提升底层视觉感知能力。

---

## 413. DialogueSidon: Recovering Full-Duplex Dialogue Tracks from In-the-Wild Dialogue Audio

**arXiv ID:** 2604.09344 | [PDF](https://arxiv.org/pdf/2604.09344v1)

**作者:** Wataru Nakata `[一作]` (University of Tokyo), Hiroshi Saruwatari `[通讯]` (University of Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过联合恢复与分离技术，构建了DialogueSidon模型，将降质的单声道两人对话音频恢复为清晰、互不重叠的双轨音频；

**💡 创新点**

创新点在于将SSL‑VAE压缩的特征空间与基于扩散的潜在预测器相结合，既能有效恢复信号质量，又能解决说话人归属的排列不确定性，并在对话语料上进行专门训练；

**🔧 技术方法**

采用w2v‑BERT 2.0的SSL特征作为输入，使用VAE压缩为低维潜在向量，再通过Diffusion Transformer（带LoRA微调与辅助潜在头）进行潜在空间的精细预测，最后通过SSL‑VAE解码器生成波形；

**📊 数据集**

训练集为Sidon修复后的电话语料（CALLHOME 5种语言 + Fisher，总计约2226小时），评估集包括Switchboard、CallFriend（5种语言）和OpenDialog（互联网真实对话）；

**📈 对比分析**

与无处理、Sidon（单通道恢复）、GENESES（原始及重新训练版）等基线比较，DialogueSidon在WER、MOS、Speaker Similarity、VAD Accuracy等指标上均优于对手，且推理实时因子仅0.01，远快于GENESES的0.60；

**⚠️ 局限性**

局限性：仅支持两人对话；当潜在维度过大时性能下降；对更复杂噪声或多说话人环境的鲁棒性尚未验证；依赖SSL模型的预训练与大量GPU训练资源。

---

## 414. Stability Enhanced Gaussian Process Variational Autoencoders

**arXiv ID:** 2604.09331 | [PDF](https://arxiv.org/pdf/2604.09331v1)

**作者:** Carl R. Richardson `[一作]` (University of Oxford), Ján Drgoňa `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种稳定性增强的高斯过程变分自编码器（SEGP‑VAE），用于从高维视频中间接学习低维线性时不变（LTI）系统的潜在过程。

**💡 创新点**

创新点在于将半收缩性LTI系统的均值与协方差函数直接推导为SEGP的先验，并给出完整无约束的A矩阵参数化，从而保证训练过程数值稳定并实现物理可解释性。

**🔧 技术方法**

核心技术包括高斯过程的隐力模型、无约束半收缩性参数化、变分自编码器框架、Bernoulli/ Beta似然、ELBO正则化与L1稀疏约束。

**📊 数据集**

在合成数据集上验证，数据为 4 万个由螺旋粒子动力学（半收缩LTI + 随机输入）生成的视频，帧尺寸 40×40。

**📈 对比分析**

与传统多输出平方指数核GP进行对比，SEGP‑VAE 在潜在轨迹预测误差、后验方差压缩以及训练稳定性上均优于标准GP；重建质量与ELBO收敛表现良好。

**⚠️ 局限性**

主要局限包括：需要对系统输入-输出结构预先有一定先验，且当前方法仅在线性系统上验证，未来需扩展至更通用的非线性/Koopman理论。

---

## 415. Robust Adaptive Backstepping Impedance Control of Robots in Unknown Environments

**arXiv ID:** 2604.09323 | [PDF](https://arxiv.org/pdf/2604.09323v1)

**作者:** Reza Nazmara `[一作]` (University of Porto), A. Pedro Aguiar `[通讯]` (TU Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种鲁棒自适应反向踏步阻抗控制（RABIC）框架，能够在移动与固定基底机器人中实现安全、可靠的接触控制；

**💡 创新点**

创新点在于：1) 采用无动态参数、无力学模型假设的自适应阻抗控制；2) 引入Taylor级数估计器在线逼近未知动力学；3) 结合Lyapunov方法实现半全局、实用、有限时间稳定性；4) 适用于联动移动机械臂与固定机械臂的统一控制结构；

**🔧 技术方法**

主要技术包括：基于Backstepping的自适应控制、Taylor级数系统估计、半全局有限时间稳定性分析、阻抗控制模型参考设计、以及实验平台Franka Emika Panda与MuJoCo仿真；

**📊 数据集**

未使用公开数据集，实验采用仿真（MuJoCo）和真实机器人Franka Panda的碰撞测试；

**📈 对比分析**

通过与传统PD控制对比，RABIC在碰撞场景下显著降低了接触力峰值、减少了后碰撞的跟踪误差、提升了控制平滑性；实验结果显示在不同质量障碍物下，RABIC的碰撞力均低于PD，并保持较小的内环误差；

**⚠️ 局限性**

局限性包括：1) 需要手动调参（阻抗参数、Backstepping增益、适应率），2) 只验证在九自由度移动机械臂和七自由度Franka Panda上，尚缺乏对更复杂结构的验证；3) 对于极大扰动或系统失效的鲁棒性尚未系统评估；4) 计算量相对较高，需在更高维系统上评估实时性能。

---

## 416. CIR+CVN: Bridging LLM Semantic Understanding and Petri-Net Verification for Concurrent Programs

**arXiv ID:** 2604.09318 | [PDF](https://arxiv.org/pdf/2604.09318v1)

**作者:** Kaiwen Zhang `[一作]` (Tongji University), Guanjun Liu `[通讯]` (Tongji University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于大型语言模型(LLM)生成的并发中间表示(CIR)，并通过将其机械转换为Petri网(CVN)实现对并发程序的形式化验证与自动修复。

**💡 创新点**

创新点包括：①将共享资源识别从源代码层面迁移到模型生成阶段，显式化资源命名与保护关系；②设计了两层检查体系（61条静态规则与CVN状态空间探索）与目标可达性验证，形成生成‑验证‑修复闭环；③在CVI中引入三值守卫和有限全局存储，使Petri网既能处理数据依赖又保持结构简单。

**🔧 技术方法**

技术手段包括：大型语言模型（GPT‑5、Claude 4.6 Opus、Gemini 3、Qwen 3.5、DeepSeek‑V3）用于生成CIR；静态检查器实现61条规则；CIR→CVI的机械翻译；基于Petri网的穷举状态空间搜索与SCC分析；目标可达性检查与层级修复策略（自动修复、局部重生成、逻辑驱动修复）。

**📊 数据集**

评估数据集为9个具有典型并发缺陷的Rust样例（包括死锁、信号丢失、通道阻塞、循环锁等），并通过不同LLM模型与五轮迭代完成生成与修复。

**📈 对比分析**

比较方法为：对每个LLM模型记录生成有效CIR所需轮数、验证后检测到的错误种类、修复完成所需轮数以及最终可达性检查结果。实验显示前沿模型在1–2轮内即可完成，且在所有模式下实现bug修复和目标可达；强模型亦能完成但在某些复杂模式（如双条件变量交叉）出现回归；整体状态空间均在数百状态内，验证耗时毫秒级。

**⚠️ 局限性**

局限性在于：①信任边界仅在CIR层面，未对源代码与CIR的一致性做形式化证明；②模型只覆盖有限的并发模式，未处理更大规模或复杂数据域；③依赖LLM的生成质量，某些LLM在复杂资源交互上可能需要多轮甚至失败；④CVI对动态调度策略不敏感，无法捕获特定调度导致的bug。

---

## 417. ChatGPT, is this real? The influence of generative AI on writing style in top-tier cybersecurity papers

**arXiv ID:** 2604.09316 | [PDF](https://arxiv.org/pdf/2604.09316v1)

**作者:** Daan Vansteenhuyse `[一作]` `[通讯]` (KU Leuven), Daan Vansteenhuyse (KU Leuven)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对2000–2025年四大顶级网络安全会议（NDSS、USENIX Security、IEEE S&P、ACM CCS）论文的文本进行纵向分析，提取词汇和句法特征，追踪学术写作风格随ChatGPT发布后的演变，并梳理各会议对GenAI使用的政策。

**💡 创新点**

创新点在于首次将会议政策、词汇复杂度指标（如长词率、平均词长、Flesch可读性）与特定“标记词”频率相结合，系统量化GenAI对网络安全论文写作的潜在影响，并揭示后2022年显著的风格转变。

**🔧 技术方法**

采用自然语言处理技术（词频统计、句子分割、可读性计算）和Python脚本自动化分析，结合PDF文本提取与预处理，辅以人工验证和人工标注的政策表格。

**📊 数据集**

数据集为2000–2025年四大会议的全部论文PDF，经过清洗后得到约1.2万篇论文的正文文本及其词汇统计，用以计算指标与标记词频率。

**📈 对比分析**

通过对比各年份的长词率、平均词长、Flesch可读性和标记词出现频率，发现从2022年起长词率提升≈3–4%，平均词长上升≈0.2字符，Flesch得分下降≈5分，标记词频率急剧上升10–15倍，表明写作风格趋向更复杂、学术化。

**⚠️ 局限性**

局限性包括：PDF提取误差可能导致词频偏差；仅覆盖四大会议，可能不代表全行业；标记词列表不完全，难以排除其他非GenAI因素的影响；无法确立因果关系，且随着新一代模型推出，趋势可能继续演化。

---

## 418. Constraint-Aware Corrective Memory for Language-Based Drug Discovery Agents

**arXiv ID:** 2604.09308 | [PDF](https://arxiv.org/pdf/2604.09308v1)

**作者:** Maochen Sun `[一作]` (Institute of Automation, Chinese Academy of Sciences), Gaofeng Meng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为CACM（Constraint-Aware Corrective Memory）的框架，用于语言模型驱动的药物发现代理，通过结构化的审计与诊断机制对返回集的协议违规进行精确定位，并将关键信息压缩后写回给规划器，提升决策的精确度和效率。

**💡 创新点**

创新点在于将返回集级别的协议满足性转化为确定性审计+基于任务、蛋白口袋和候选集证据的“基于诊断”的纠正信号，并将该信号与静态、动态、纠正三通道的记忆组织和压缩实现，形成紧凑、可解释的规划器上下文，从而解决了传统语言模型在多步骤决策中对失败定位模糊、历史冗长导致上下文膨胀的问题。

**🔧 技术方法**

技术包括：1) 基于规则的返回集审计（确定性通过/失败判定）；2) 结构化的基于诊断（Grounded Diagnoser）生成纠正提示和行动偏好；3) 三通道记忆（静态、动态、纠正）与通道级别的选择、压缩与适配；4) 与现有LLM规划器（DeepSeek）和分子工具链（Pocket2Mol、GraphGA、AutoDock Vina）结合的闭环控制。

**📊 数据集**

使用LIDDiA 30目标基准，每个目标包含蛋白口袋、自然语言设计需求以及对应的阈值，评估返回集的多项指标（QED、SAS、Lipinski、Vina、Novelty、Diversity）。

**📈 对比分析**

与LIDDiA原始基线、Pocket2Mol、DiffSMOL以及Claude、GPT‑4o、o1‑mini、o1等通用LLM基线进行对比。CACM在目标成功率（TSR）上从73.3%提升至100%，DVS&HQ从80.0%提升至100%，平均终止迭代从4.40降至3.07，终止集大小从21降至5，控制器侧Token平均降低31.3%。

**⚠️ 局限性**

局限性包括：1) 仍需依赖现有分子生成、优化与评估工具，无法单独改进分子质量；2) 目前仅在短期（≤10步）LIDDiA基准上验证，难以直接推断在更长时间、更复杂协议场景下的鲁棒性；3) 纠正记忆的压缩采用规则模板，缺乏自适应学习，可能在某些任务中未能充分提炼关键信息。

---

## 419. Is More Data Worth the Cost? Dataset Scaling Laws in a Tiny Attention-Only Decoder

**arXiv ID:** 2604.09389 | [PDF](https://arxiv.org/pdf/2604.09389v1)

**作者:** Götz-Henrik Wiegand `[一作]` (University of St Gallen), Siegfried Handschuh `[通讯]` (University of St Gallen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在小模型规模下，使用冻结预训练嵌入、移除MLP、仅自注意力的Transformer decoder，对不同规模训练子集的性能进行系统实验。

**💡 创新点**

通过在固定容量下隔离自注意力学习，首次验证在极小模型上仍符合规模律的衰减收益，并提出子集预训练与早停策略。

**🔧 技术方法**

使用基于117M GPT‑2架构的注意力仅Decoder，冻结嵌入和输出层，移除MLP；采用子集构建、Jensen–Shannon相似度评估、固定epoch训练和固定优化步数实验。

**📊 数据集**

AllTheNews2.0 新闻语料（约270万篇）以及在附录中对WikiText-103的验证。

**📈 对比分析**

对不同2^k规模子集在固定训练步骤与固定epoch两种方案下进行验证准确率与交叉熵对比，发现90%完整性能仅需30%数据，显著降低计算成本。

**⚠️ 局限性**

训练未达收敛且受固定epoch限制，模型仅关注自注意力，未探索多层、MLP或更大模型的进一步影响；结果可能不完全可迁移到更大规模或不同任务。

---

## 420. EGLOCE: Training-Free Energy-Guided Latent Optimization for Concept Erasure

**arXiv ID:** 2604.09405 | [PDF](https://arxiv.org/pdf/2604.09405v1)

**作者:** Junyeong Ahn `[一作]` (KAIST AI), Sungyong Baik `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EGLOCE框架，通过推理时对潜在空间进行能量引导优化，实现对文本到图像扩散模型的概念擦除。

**💡 创新点**

创新点在于将概念擦除转化为双目标能量最小化问题，结合排斥能量与保留能量，在推理阶段即可完成擦除且兼容现有模型。

**🔧 技术方法**

使用基于CLIP的文本‑图像对齐能量、Latent Diffusion模型、能量引导采样以及多步梯度迭代优化技术。

**📊 数据集**

在COCO‑30k、ImageNet等公开数据集上评估，并使用NudeNet、CLIP、FID、LPIPS等指标。

**📈 对比分析**

与ESD、SLD、RECE、SAFREE等SOTA擦除方法比较，EGLOCE在攻击成功率、CLIP相似度、FID等指标上显著优于基线，尤其在反向攻击场景下提升明显。

**⚠️ 局限性**

局限在于能量函数基于CLIP可能无法完全捕捉低级纹理或艺术风格，导致对某些概念的擦除效果有限，且多步迭代带来额外计算开销。

---

## 421. Insights from Farmer-Managed Decentralized Solar Irrigation Systems

**arXiv ID:** 2604.09395 | [PDF](https://arxiv.org/pdf/2604.09395v1)

**作者:** Arnab Paul Choudhury `[一作]` (Viksit Labs Foundation), Aryan Yadav `[通讯]` (Viksit Labs Foundation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对印度古吉拉特邦农户在分散部署的太阳能灌溉泵（SIP）管理与维护进行民族志研究，揭示他们如何通过WhatsApp群组共享发电数据、互相比较、诊断异常，并将此过程视为一种社区驱动的数字基础设施。

**💡 创新点**

创新点在于将普通社交聊天工具WhatsApp重新定义为分布式能源系统的“非正式数字基础设施”，通过农户自发的数据共享与比较实现了从孤立监测到协作维护的转变，弥补了官方Sky应用的功能缺口。

**🔧 技术方法**

采用人种学方法（访谈、观察、笔记）结合数字工具（WhatsApp群组、官方Sky Android应用）进行数据收集与分析，侧重于社区实践与信息流动的质性评估。

**📊 数据集**

数据集包括：①农户手工记录的发电量（纸质表格或白板记录），②官方Sky应用提供的实时系统状态与发电/消耗数据，③WhatsApp群组中的每日发电数据共享记录与讨论内容。

**📈 对比分析**

比较方法为农户将每日发电量上传至WhatsApp群组，彼此对比相同容量系统的输出差异，快速识别异常（如尘埃堆积、线路损坏、遮阴等）。研究未给出量化性能指标，但通过案例展示维护响应速度提升、问题定位准确度提高、农户满意度上升等质性改善。

**⚠️ 局限性**

局限性包括：①样本规模小且仅为男性农户，缺乏性别与区域多样性；②缺乏对比实验或量化评估WhatsApp与官方应用在监测精度、维护效率方面的差异；③研究侧重质性观察，未能系统评估该做法的可复制性和长期可持续性。

---

## 422. 3D-Printing Water-Soluble Channels Filled with Liquid Metal for Recyclable and Cuttable Wireless Power Sheet

**arXiv ID:** 2604.09299 | [PDF](https://arxiv.org/pdf/2604.09299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 423. Decision Trace Schema for Governance Evidence in Real-Time Risk Systems

**arXiv ID:** 2604.09296 | [PDF](https://arxiv.org/pdf/2604.09296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 424. Efficient Unlearning through Maximizing Relearning Convergence Delay

**arXiv ID:** 2604.09391 | [PDF](https://arxiv.org/pdf/2604.09391v1)

**作者:** Khoa Tran `[一作]` (Sungkyunkwan University), Simon S. Woo `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了机器遗忘（unlearning）过程中的新评估指标和对应的改进框架。

**💡 创新点**

创新点在于：1) 引入“relearning convergence delay”（RCD）指标，能够量化模型在遗忘数据上重新学习的速度，从权重空间和预测空间两方面衡量遗忘效果；2) 设计了Influence Eliminating Unlearning (IEU) 框架，结合梯度上升和噪声正则化（噪声注入+权重衰减），显著提升遗忘质量与防止再学习的能力；3) 提供了理论上界和指数收敛保证。

**🔧 技术方法**

技术手段包括：梯度上升、噪声正则化（Iterative Re-initialization Process）、损失函数对保留集与遗忘集的双重优化、条件数分析、RCD指标近似与理论推导、实验使用梯度下降与Adam优化器。

**📊 数据集**

数据集：图像分类任务使用 CIFAR‑10、CIFAR‑100、TinyImageNet；生成任务使用 Stable Diffusion（latent SD）和 ImageNette；模型架构为 ResNet‑50、ViT、Stable Diffusion。

**📈 对比分析**

与 Fine‑tuning (FT)、Random Labeling (RL)、SCRUB、SALUN 等基线进行比较。实验显示 IEU 在保留集上的精度差距（Avg. Gap）最小，RCD 分数最高，且在随机与类别级遗忘场景下保持良好的隐私‑效能平衡；在生成任务中，IEU 能显著降低 Nudity 分数并保持低 FID，说明在去除有害概念的同时保留生成质量。

**⚠️ 局限性**

局限性包括：1) RCD 与 Adam 训练的适配性尚未充分验证，导致生成任务中的 RCD 结果与分类任务不完全一致；2) 噪声正则化会在一定程度上提升误差上界，存在隐私‑效能权衡；3) 目前实验集中在图像分类与文本引导图像生成，对其他领域（如自然语言、音频）或大规模模型的泛化尚未探讨；4) 需要进一步研究超参数（α、c）对收敛与性能的精细影响。

---

## 425. Through Their Eyes: Fixation-aligned Tuning for Personalized User Emulation

**arXiv ID:** 2604.09368 | [PDF](https://arxiv.org/pdf/2604.09368v1)

**作者:** Lingfeng Huang `[一作]` (Singapore University of Technology and Design), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在推荐系统评估中提出一种基于视觉语言模型的用户模拟器 FixATE，通过个性化的软提示让模型的视觉注意力与真实用户的注视模式保持一致，从而更准确地模拟用户点击行为。

**💡 创新点**

创新点在于：① 将用户特定的注视分布转化为可量化的slot‑level视觉相关性；② 引入可分解的软提示基（prompt basis）与用户专属系数，实现在不更新模型权重的前提下进行个性化对齐；③ 通过联合注意力对齐损失与下一个词预测损失，使模型在保持行为预测精度的同时提升视觉关注度。

**🔧 技术方法**

使用技术包括：视觉语言模型（Qwen3‑VL‑4B‑Instruct、InternVL3.5‑4B‑Instruct）；三种解释性探测器（Attention Rollout、GLIMPSE、AttnLRP）用于提取视觉相关性；基于软提示的因子化编码；加权 KL 对齐损失与标准 NTP 损失；以及对齐后的slot‑level注意力与用户注视热图的多种对齐评估指标。

**📊 数据集**

数据集：主实验使用 RecGaze（电影推荐车轮界面，包含 3×5 网格的注视时长与点击记录）；泛化实验使用 AdSERP（搜索引擎结果页的注视与点击数据）。

**📈 对比分析**

方法与基线比较：在三种探测器与两种VLM骨干上进行leave‑one‑out评估。FixATE 在注意力对齐指标（KL、JS、Cosine、CSH@k、TGO@k）和行为预测指标（Accuracy、LogLoss、AUC）上均显著优于未对齐的 Backbone，尤其在 top‑slot 对齐与点击准确率上提升明显，证明个性化注视对齐能有效提升模拟器的真实性。

**⚠️ 局限性**

局限性包括：① 眼动跟踪数据规模有限，难以覆盖多样化用户与界面；② 采集个体眼动成本高，缺乏可大规模部署的低成本替代（如光标轨迹）研究；③ 目前未考虑用户注意力随时间演变的长期历史，缺乏跨会话的动态建模。

---

## 426. Hierarchical Flow Decomposition for Turning Movement Prediction at Signalized Intersections

**arXiv ID:** 2604.09336 | [PDF](https://arxiv.org/pdf/2604.09336v1)

**作者:** Md Atiqur Rahman Mallick `[一作]` (Tennessee State University), S M Shazzad Rassel `[通讯]` (Tennessee State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种层级深度学习框架 HFD-TM，先预测通道层面流量，再通过比例预测模块将其展开为各向交叉点的转向流量，并加入残差校正与零流掩码。

**💡 创新点**

创新点在于：① 层级分解把低波动通道流量与高波动转向流量分离；② 学习转向比例的预测模块；③ 通过物理约束的流量守恒损失函数保证结构一致性；④ 极低的计算开销，适合实时部署。

**🔧 技术方法**

使用的技术包括：GRU 编码器、线性+MLP 的比例预测器、残差校正网络、零流掩码、物理约束损失、Adam 优化器及学习率调度。

**📊 数据集**

数据集为 Nashville 六路口 Corridor 的 LiDAR 流量数据，包含 20,352 个 15 分钟间隔样本，涵盖 12 条通道流和 60 条转向流。

**📈 对比分析**

与 GRU、LSTM、Transformer、DCRNN 四种基线模型对比，HFD‑TM 在 MAE 2.49、RMSE 5.19 方面分别比最佳基线（Transformer MAE 2.64）低 5.7%/9.2%，训练时间仅 276 秒，约比 DCRNN 快 12.8 倍。

**⚠️ 局限性**

局限性：仅在单一几何配置上验证；仅做单步（15 分钟）预测；未在多步预测和不同交叉口拓扑下评估泛化能力。

---

## 427. A Benchmark of Dexterity for Anthropomorphic Robotic Hands

**arXiv ID:** 2604.09294 | [PDF](https://arxiv.org/pdf/2604.09294v1)

**作者:** Davide Liconti `[一作]` (ETH Zurich), Robert K. Katzschmann `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了POMDAR——一种基于性能、基于任务的机器人手指灵巧度基准，涵盖12项操纵和6项纯抓取任务，所有设备均可3D打印并公开发布；

**💡 创新点**

创新点在于将人类手部运动与抓取分类（Elliott & Connolly、GRASP等）相结合，构建统一的、可量化的吞吐量评分体系（正确性+速度），并实现了物理与仿真双模版，支持快速复现；

**🔧 技术方法**

所采用的技术包括消费级3D打印、MuJoCo物理仿真、手势捕捉手套和Apple Vision Pro的VR交互、MANO手模型与贝叶斯优化的标定；

**📊 数据集**

使用了从用户研究获得的人的基准数据（22个手部关键点轨迹）以及GRASP、Elliott & Connolly等公开的抓取与操纵分类，而非传统数据集；

**📈 对比分析**

通过在同一机器人和操作者下比较四种ORCA手指构型（2、3、5、16 DoF），利用雷达图和总体得分评估，结果表明手指自由度越高通常得分越高，但提升幅度取决于具体任务；

**⚠️ 局限性**

局限性包括仅在遥控条件下评估，无法独立评估手部机械本身；基准物体对人形手优先，导致非人形手表现偏低；未覆盖动态推挤、投掷等高惯性交互；对外部机械臂的依赖；以及对自动化策略评估的缺失。

---

## 428. ECHO: Efficient Chest X-ray Report Generation with One-step Block Diffusion

**arXiv ID:** 2604.09450 | [PDF](https://arxiv.org/pdf/2604.09450v1)

**作者:** Lifeng Chen `[一作]` (AIRC, Midea Group), Yi Xu `[通讯]` (AIRC, Midea Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ECHO，一个能在一块（block）一次完成并行解码的离散扩散视觉‑语言模型，用于高效生成胸部 X‑ray 报告。

**💡 创新点**

创新点包括：1）直接条件蒸馏（DCD），通过从教师的重掩码轨迹获取非因子化监督，解决单步解码的均场偏差；2）响应不对称扩散（RAD）一次性完成 AR‑to‑diffusion 转换，显著降低训练 FLOPs；3）对报告进行结构化归一化，消除缺失负面信息的偏差。

**🔧 技术方法**

使用技术包括离散扩散模型、基于 KL 的蒸馏、块级注意力（block KV 缓存）与融合 KV 缓存、语义对齐的可视‑语言预训练模型 Lingshu‑7B、以及自监督重掩码策略。

**📊 数据集**

采用公开胸影数据集：MIMIC‑CXR、CheXpert‑Plus、ReXGradient、IU‑Xray，并加入 LLaVA‑ReCap‑558K 进行多语种训练。

**📈 对比分析**

与现有自回归模型（如 LLaVA‑Med、Lingshu‑32B）以及扩散模型（如 LLaDA‑MedV、dParallel、T3D）对比，ECHO 在 RaTEScore、SemScore 方面提升 17%‑40%，在 ROUGE‑L/CIDEr 方面同样显著，且单步块解码实现 8× 速度提升，整体性能优于同类最佳方法。

**⚠️ 局限性**

局限性：仍需大规模预训练 AR 模型作为教师；对均场偏差的抑制依赖于多步轨迹采样，可能在极端噪声条件下不稳定；目前仅针对胸部 X‑ray 报告验证，迁移到其他医学多模态任务需进一步验证。

---

## 429. AsymLoc: Towards Asymmetric Feature Matching for Efficient Visual Localization

**arXiv ID:** 2604.09445 | [PDF](https://arxiv.org/pdf/2604.09445v1)

**作者:** Mohammad Omama `[一作]` (University of Texas at Austin), Yelin Kim `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出异构视觉定位框架 AsymLoc：离线使用大型教师模型提取数据库图像特征，在线使用轻量级学生模型提取查询图像特征，并通过联合检测-描述子空间对齐实现两种模型特征的直接匹配。

**💡 创新点**

创新点在于设计了两种损失：几何匹配损失（利用教师检测置信度和描述子相似度的软匹配矩阵）与联合检测-描述子蒸馏损失（将检测置信度与描述子相似度融合为概率分布，再用 KL 散度对齐），从而无需额外的学习匹配器即可实现高效兼容匹配。

**🔧 技术方法**

主要技术包括知识蒸馏、几何匹配（基于已知单应性或双目几何）、互相关与双向 softmax 正则化、概率分布对齐（KL 散度）、温度调节、联合检测-描述子概率空间建模。

**📊 数据集**

实验使用 HPatches、ScanNet、IMC2022、Aachen（Day‑Night）四个数据集，并在 Synthetic COCO 对生成的同伦图像对上训练。

**📈 对比分析**

与教师同构、标准小模型、Naïve Distillation、AML、RKD、CSD、D3Still 等方法对比，AsymLoc 在 0.04–0.13M 参数的小学生模型上保持 93–96% 的教师精度，显著超越对称蒸馏和现有异构匹配基线，同时保持与小模型相同的计算量和内存占用。

**⚠️ 局限性**

局限性：依赖教师模型的预训练和大量同伦数据；在极端视角或光照变化下仍可能出现匹配误差；对不同类别的异构匹配泛化能力尚需进一步验证。

---

## 430. Confidence Without Competence in AI-Assisted Knowledge Work

**arXiv ID:** 2604.09444 | [PDF](https://arxiv.org/pdf/2604.09444v1)

**作者:** Elena Eleftheriou `[一作]` (University of Cyprus), Marios Constantinides `[通讯]` (CYENS Centre of Excellence)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并评估了名为Deep3的LLM交互工具，提供三种不同模式（未来自我解释、对比学习、引导提示）来促进Gen Z学生的深度思考与学习；

**💡 创新点**

创新点在于将“生产性摩擦”嵌入LLM交互，分别通过自我解释、对比反馈和渐进提示三种机制实现学习目标的差异化，缓解过度依赖AI导致的浅层理解；

**🔧 技术方法**

技术上使用React+Tailwind构建前端，FastAPI+MongoDB做后端，并通过any-llm连接Llama-4-maverick-17b-128e-instruct模型实现对话与提示生成；

**📊 数据集**

数据集基于真实大学生（85人）在两项学习任务中的交互日志、问卷评分与访谈记录；

**📈 对比分析**

比较方法采用双组间实验与配对t检验/非参数检验，结果显示引导提示模式在问题求解任务中获得最高学习效果（d≈1.14）且工作负荷最低，未来自我解释模式实现最佳认知与自我评估一致性；

**⚠️ 局限性**

局限包括样本偏向STEM本科生、未检验长期学习效果、未对不同学科与AI素养差异进行控制，以及仅使用单一LLM模型，未来需扩大样本多样性与对比更先进推理模型。

---

## 431. Musculoskeletal Motion Imitation for Learning Personalized Exoskeleton Control Policy in Impaired Gait

**arXiv ID:** 2604.09431 | [PDF](https://arxiv.org/pdf/2604.09431v1)

**作者:** Itak Choi `[一作]` (Carnegie Mellon University), Inseung Kang `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过物理可行的肌肉骨骼仿真与强化学习结合，提出一种设备无关的下肢外骨骼控制框架，能够在模拟中生成生理合理的步态并针对特定肌肉缺陷提供定向辅助。

**💡 创新点**

创新点在于：①完全在仿真中训练出可应用于多种外骨骼硬件的控制策略；②通过降低特定肌肉最大激活实现受损步态模型，进而在同一策略中生成针对性异侧辅助；③在奖励函数中同时引入运动学、动力学跟踪与代谢成本，使生成的步态既精准又节能。

**🔧 技术方法**

使用技术包括 Hyfydy 物理引擎/SCONE API 的肌肉骨骼仿真、Soft Actor-Critic 强化学习、元学习奖励设计、以及基于 Umberger-Uchida 代谢模型的能量惩罚。

**📊 数据集**

采用公开的运动学/力量平台数据集（12名参与者、31项活动，本文选取单名受试者跑步/步行的同步运动捕捉与地面反作用力数据）。

**📈 对比分析**

通过与人类实验验证的优化助力曲线对比，所生成的臀部与踝部助力曲线高度匹配；在模拟中能量消耗分别下降 4.5–13.8%，受损步态的代谢成本下降 8–13% 并显著提升两侧运动对称性。

**⚠️ 局限性**

主要限制包括：尚未在真实外骨骼硬件上验证；受损步态模型仅通过降低肌肉激活实现，未完全捕捉临床复杂病理；实验仅覆盖水平跑步/步行，缺乏多任务泛化。

---

## 432. On the Representational Limits of Quantum-Inspired 1024-D Document Embeddings: An Experimental Evaluation Framework

**arXiv ID:** 2604.09430 | [PDF](https://arxiv.org/pdf/2604.09430v1)

**作者:** Dario Maio `[一作]` `[通讯]` (University of Bologna), Dario Maio (University of Bologna)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了基于量子启发式的 1024 维文本嵌入框架，探究其几何特性及在检索任务中的表现，并构建诊断工具。

**💡 创新点**

提出了实验与诊断框架，系统分析量子启发式嵌入的几何失真对检索的影响，并评估蒸馏与混合检索的可行性。

**🔧 技术方法**

使用角度编码、EigAngle、量子电路模拟（Aer 后端）进行特征映射，结合线性/MLP 蒸馏、BM25 与稠密向量的得分级插值、Reciprocal Rank Fusion 以及 Cross‑Encoder 再排序。

**📊 数据集**

实验数据集为 10 篇意大利技术文档、10 篇英文叙事文本、10 篇意大利法律文档，并使用合成查询集。

**📈 对比分析**

通过与 BM25、教师模型嵌入及混合检索对比；单独使用量子嵌入效果差，蒸馏提升不稳定，混合检索能在部分情况下恢复性能但往往不超过强大的 BM25；在子块级检索中表现更差。

**⚠️ 局限性**

主要限制在于量子启发式嵌入的几何失真、距离压缩和相似度倒置；蒸馏虽能提高全局对齐但对局部检索效果有限；对不同领域和查询形式敏感，无法作为独立检索模型，仅能作为辅助信号。

---

## 433. Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories

**arXiv ID:** 2604.09429 | [PDF](https://arxiv.org/pdf/2604.09429v1)

**作者:** Wonbong Jang `[一作]` (Meta AI), Tao Xiang `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

学习视频帧与相机轨迹的联合分布，统一相机姿态估计和相机控制视频生成。

**💡 创新点**

将相机参数重编码为稠密像素级的“raxel”图像，使其可与预训练视频VAE共享空间；引入解耦自交叉注意力和流匹配的联合去噪框架，实现单模型完成三种任务。

**🔧 技术方法**

稠密r‑像素编码、预训练视频扩散模型（Wan 2.1）、Decoupled Self‑Cross Attention、流匹配（Flow Matching）、正交Procrustes姿态恢复、基于中位数的焦距估计。

**📊 数据集**

RealEstate10K、DL3DV、Tanks & Temples；训练时对场景尺度统一，使用时间倒转增强。

**📈 对比分析**

与VGGT、MotionCtrl、VD3D、ViewCrafter、Wonderland、Kaleido等基线对比，FVD/FID均为最佳或相近，姿态估计旋转误差低于基线但略逊，闭环自洽测试表现优异。

**⚠️ 局限性**

仅适用于静态、平滑轨迹场景；VAE 4×时间压缩限制帧率；只能使用图像条件，未支持文本控制；对快速运动或动态场景泛化有限。

---

## 434. Do We Really Need to Approach the Entire Pareto Front in Many-Objective Bayesian Optimisation?

**arXiv ID:** 2604.09417 | [PDF](https://arxiv.org/pdf/2604.09417v1)

**作者:** Chao Jiang `[一作]` (University of Birmingham), Miqing Li `[通讯]` (University of Birmingham)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了单点多目标贝叶斯优化框架（SPMO），旨在在有限评估预算内寻找一条最优权衡点。

**💡 创新点**

创新点在于：①专注单点质量而非完整 Pareto 前沿；②设计了可微分的期望单点改进（ESPI）采集函数；③在采集优化中使用样本平均逼近（SAA）实现梯度搜索并提供收敛性证明。

**🔧 技术方法**

技术包括：多目标高斯过程代理、期望单点改进采集函数、Monte Carlo 近似、梯度求解、SAA 收敛分析、噪声处理（NESPI）以及批量扩展。

**📊 数据集**

数据集涵盖标准多目标基准（DTLZ1/2、倒置、凸、尺度变体，3/5/10 维）以及两类真实工程问题（车侧撞击设计、车厢设计）。

**📈 对比分析**

与六种对比方法（Sobol、ParEGO、TS‑TCH、EHVI、C‑EHVI、JES）在单点距离、单点 hypervolume 和全体解集 hypervolume 上进行比较。SPMO 在单点指标上显著优于对手，在全体 hypervolume 上亦保持竞争力，且在收敛速度和计算时间上更快。

**⚠️ 局限性**

局限性是只能得到单个权衡点，无法获得完整 Pareto 前沿信息，因而对需要多点参考的应用场景适用性有限。

---

## 435. HiL-Bench (Human-in-Loop Benchmark): Do Agents Know When to Ask for Help?

**arXiv ID:** 2604.09408 | [PDF](https://arxiv.org/pdf/2604.09408v1)

**作者:** Mohamed Elfeki `[一作]`, Bing Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

HiL-Bench通过在现有SWE和SQL任务中注入多种真实阻断信息，构建了一个需要模型在执行过程中主动识别并请求澄清的评测基准。

**💡 创新点**

其创新点在于（1）采用progressive discovery方式让阻断信息在探索中逐步显现；（2）提出Ask‑F1指标兼顾提问精度与召回，防止提问spam；（3）通过RL训练使判断能力可提升并在不同领域迁移。

**🔧 技术方法**

本研究使用大型语言模型与工具调用框架，借助冻结的Llama‑3.3‑70B模拟人类专家，设计了Ask‑F1奖励函数并在Qwen3‑32B上进行LoRA微调。

**📊 数据集**

数据集由300个任务（150软件工程，150文本转SQL）组成，任务源自SWE‑Bench Pro与BIRD，包含共1131个阻断信息（平均每任务3.8个）。

**📈 对比分析**

在完整信息下模型pass@3可达86‑91%（SQL）和64‑88%（SWE），但当必须判断是否提问时pass@3降至4‑24%，Ask‑F1平均约40%；RL训练后，Ask‑F1与pass@3均显著提升，并能跨域迁移。

**⚠️ 局限性**

局限性包括仅覆盖SWE与SQL两领域、阻断信息由人工验证且工具回答有限，缺少多模态或更复杂对话场景，且指标仍未细化提问质量的深度。

---

## 436. VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning

**arXiv ID:** 2604.09508 | [PDF](https://arxiv.org/pdf/2604.09508v1)

**作者:** Yucheng Shen `[一作]` (Soochow University), Min Cao `[通讯]` (Soochow University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VISOR 框架，构建了单一智能体可在视觉检索增强生成（VRAG）任务中通过多轮检索、视觉裁剪和推理交互，完成跨页多步推理。

**💡 创新点**

创新点包括：
• 结构化证据空间（Evidence Space），在检索与裁剪后自动提炼并存储跨页关键证据；
• 视觉动作评估与纠正机制，先判断裁剪是否必要，裁剪无效时恢复原状态；
• 动态轨迹滑动窗口与意图注入，实时重构上下文，仅保留最前方证据与最近交互，避免视觉令牌占满上下文导致的搜索漂移；
• 基于 GRPO 的强化学习训练，结合检索与答案奖励，学习精准停止与检索策略。

**🔧 技术方法**

技术手段：视觉检索（ColPali 等）、视觉‑语言模型（Qwen2.5‑VL）、多轮交互式 agent 结构、视觉裁剪操作、证据提炼、动态上下文重构、意图注入、GRPO‑based RL、奖励设计与信用分配。

**📊 数据集**

使用的公开数据集：ViDoSeek、SlideVQA 与 MMLongBench，用于训练（SFT 取 SlideVQA 训练集）与评估。

**📈 对比分析**

与多种基线比较：单代理 Vanilla RAG、ReAct、Search‑R1‑VL、VRAG‑RL、EVisRAG、R1‑Router 等；以及多代理 ViDoRAG、M3RAG。VISOR 在 SlideVQA 上 78.82%（单跳）/53.62%（多跳），在 ViDoSeek 上 74.87%/69.00% 等，均显著优于所有对比模型，证明其在跨页多步推理与视觉证据稀疏场景中的优势。

**⚠️ 局限性**

局限性：
• 对检索质量仍高度依赖，若检索缺失关键页即使后续推理也会失效；
• 对高度结构化或复杂视觉内容（如细节图表、流程图）的理解仍不充分；
• 需要多轮推理导致推理时间相对较长；
• 受限于模型规模，在 3B 参数模型下性能略低于 7B 版。

---

## 437. Strategic Algorithmic Monoculture:Experimental Evidence from Coordination Games

**arXiv ID:** 2604.09502 | [PDF](https://arxiv.org/pdf/2604.09502v1)

**作者:** Gonzalo Ballestero `[一作]` (Pennsylvania State University), Ran I. Shorrer `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实施了包含人类与多种大型语言模型（LLM）的实验，考察它们在开放式问题上的一致性与差异化行为，并通过三种激励（无激励、协调激励、分歧激励）来分离基础与战略算法单一性。

**💡 创新点**

提出主/战略算法单一性这一新分类框架，首次将其与人类协调研究中的主/次/塞林格显著性区分；通过实验设计实现对激励影响的直接分离，揭示LLM在协调与分歧情境下的差异化适应。

**🔧 技术方法**

使用16款公开及闭源LLM（包括Claude、Gemini、Gemma、GPT‑4o、Llama、Phi、Qwen 等）和人类参与者；实验中测量agreement rate作为一致性指标，并利用语义相似度分析和LLM‑as‑Judge对模型生成的推理文本进行自动分类。

**📊 数据集**

收集12类开放式问题（如城市、颜色、数字等）的答案数据：人类通过 Prolific 平台完成实验，共 301 名受试者；LLM 通过 28,800 次独立查询生成答案；所有答案统一标准化后用于后续分析。

**📈 对比分析**

比较方法：对同一问题主题下不同激励组的 agreement rate 进行统计对比；结果显示 LLM 在协调激励下的 agreement rate 为 72%（远高于人类的 31%），但在分歧激励下仅为 27%（远低于人类的 4%）；LLM 在基础激励下的 agreement rate 也高于人类（58% 对比 14%）。

**⚠️ 局限性**

局限性：LLM 难以实现有效随机化，导致分歧情境下表现不足；实验仅涉及单一开放式文本任务，缺乏多模态或复杂经济情境的检验；LLM 对合作方身份信息的敏感度有限，无法完全模拟人类在现实多主体决策中的情境；模型间差异可能受温度、提示语等技术细节影响，进一步的泛化研究仍需开展。

---

## 438. E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning

**arXiv ID:** 2604.09455 | [PDF](https://arxiv.org/pdf/2604.09455v1)

**作者:** Weiyang Guo `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于增强经验利用的热身范式（E3‑TIR），用于改进工具集成推理（TIR）中的大型语言模型训练。

**💡 创新点**

创新点在于：①将专家前缀、专家引导和自我探索三类经验动态融合；②利用专家“锚点”进行分支采样并通过混合优势估计与梯度分离技术解决共享前缀的优化冲突；③采用离线重塑的混合策略优化，实现高效且稳定的学习。

**🔧 技术方法**

核心技术包括：强化学习（策略梯度）、混合优势估计（全局优势 + 专家树内部优势）、优势感知梯度分离（AAGD）、离线重塑（off‑policy reshaping）以及基于熵的分支采样。

**📊 数据集**

使用的数据集主要包括：Tool‑Star 的 SFT 与 RL 数据、AgentGYM 的长步任务数据；评测任务覆盖数学推理（AIME24、AIME25、MATH、AMC23、GSM8K）和知识密集推理（HotpotQA、2WikiMultiHopQA、Musique、Bamboogle、SimpleQA），以及长步探索任务（ALFWorld、SCIWorld）。

**📈 对比分析**

与 SFT‑only、SFT‑then‑RL、Zero‑RL 等基线以及当前公开的 SOTA 方法（Search‑R1、Tree‑GRPO、Tool‑Star、ARPO 等）进行对比。实验显示平均提升约 6%（在 10 个任务上），ROI（性能+数据+训练效率）提升 1.46×，且仅使用不到 10% 的合成数据。

**⚠️ 局限性**

局限性包括：对专家前缀质量高度依赖，低质量或单一多样性的锚点可能限制搜索空间；目前评测仅覆盖数学和 QA 任务，对多工具协同或更长序列的真实场景仍需进一步验证。

---

## 439. Many-Tier Instruction Hierarchy in LLM Agents

**arXiv ID:** 2604.09443 | [PDF](https://arxiv.org/pdf/2604.09443v1)

**作者:** Jingyu Zhang `[一作]` (Johns Hopkins University), Daniel Khashabi `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多层级指令层级（ManyIH）范式，允许在推理时动态指定任意数量的权限等级并解决冲突；

**💡 创新点**

创新点在于：①将权限级别从固定的角色标签解耦出来，支持任意多的动态权限层级；②设计了基于标记的权限提示接口（ordinal 与 scalar）；③创建了全新的 853 条样本 benchmark（ManyIH-Bench），涵盖编码与多代理指令跟随两大子集；

**🔧 技术方法**

使用了权限提示接口技术、顺序/标量权限编码、链式推理（CoT）与自检机制，以及多模型推理与评估框架；

**📊 数据集**

使用了 MBPP 编码任务与 AgentIF 指令跟随数据，融合生成冲突指令与权限标签，构成 ManyIH-Bench；

**📈 对比分析**

与现有两层级指令层级评估对比，评测 10 个前沿模型，最高准确率仅约 42%，且随着权限层数上升准确率呈递减趋势；

**⚠️ 局限性**

局限性在于：当前模型对多层级冲突的推理能力极弱，易受权限表示方式与数值扰动影响，缺乏鲁棒性，需要进一步的训练方法和体系结构改进。

---

## 440. TME-PSR: Time-aware, Multi-interest, and Explanation Personalization for Sequential Recommendation

**arXiv ID:** 2604.09439 | [PDF](https://arxiv.org/pdf/2604.09439v1)

**作者:** Qingzhuo Wang `[一作]` (Tongji University), Wen Shen `[通讯]` (Tongji University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种集成时间感知、多兴趣和解释个性化的序列推荐模型TME-PSR，能够同时提升推荐精度和解释质量。

**💡 创新点**

创新点在于：①双视图门控时间编码器自适应融合短期与长期时间节奏；②多头LRU架构以轻量化方式分解细粒度子兴趣；③动态双分支MI加权机制实现个性化语义对齐。

**🔧 技术方法**

使用门控GRU、MLP、线性递归单元（LRU）及互信息（MI）损失，配合Embedding、t-SNE可视化与K-means聚类等评估工具。

**📊 数据集**

在Amazon Movies、Amazon Electronics和Yelp三个常用可解释序列推荐数据集上进行实验。

**📈 对比分析**

将TME-PSR与12个基线模型（包括通用、时间建模、多兴趣和可解释模型）在Recall@10和NDCG@10指标上对比，TME-PSR在所有数据集上均取得最高分，提升幅度从约10%到37%（推荐）以及0.5%到25%（解释）不等。

**⚠️ 局限性**

局限性包括：①模型对超参数（如α、β、d、H）的敏感性；②仅针对单域静态序列，跨域或多模态适配尚未验证；③解释生成仅使用聚类后的语义片段，缺乏更丰富的自然语言生成能力。

---

## 441. Do Vision Language Models Need to Process Image Tokens?

**arXiv ID:** 2604.09425 | [PDF](https://arxiv.org/pdf/2604.09425v1)

**作者:** Sambit Ghosh `[一作]` (IBM), Chirag Agarwal `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型(VLM)中视觉令牌处理的深度是否真正必要，并探究其在不同任务中的贡献与可恢复性。

**💡 创新点**

创新点在于将视觉表示的几何稳定性（矩阵熵、内在维度、曲率）与层级功能可互换性、任务依赖性、微调可恢复性关联，揭示视觉深度不必均等使用；并提出基于表示稳定性的层级截断与恢复策略。

**🔧 技术方法**

主要技术包括：表示度量（矩阵熵、内在维度、轨迹曲率）、层替代实验、视觉令牌截断与多模态模型微调（LoRA+知识蒸馏）、多任务（单词预测、多词生成、问答、图像描述、链式推理）评估。

**📊 数据集**

使用 LLaVA‑1.5、Qwen2.5‑VL、InternVL 三族 VLM，数据集涵盖 BLINK、ChartQA、Flickr8k、M3CoT 等多模态视觉问答与图像描述任务。

**📈 对比分析**

与完整模型对比，截断模型在单词预测任务上保持较好性能，但在多词生成（图像描述、无选项问答）任务中显著下降；微调后可部分恢复，恢复程度随保留的视觉深度递增。实验表明视觉深度在多词生成和链式推理中更为关键。

**⚠️ 局限性**

局限性：恢复效果受限于保留的视觉层数，对细粒度问答（Exact Match）恢复尤弱；表示稳定性并不等同于推理完整，因而无法完全取代深层视觉处理；实验范围局限于所选模型和数据集，可能不适用于所有 VLM 架构。

---

## 442. OASIS: Online Activation Subspace Learning for Memory-Efficient Training

**arXiv ID:** 2604.09406 | [PDF](https://arxiv.org/pdf/2604.09406v1)

**作者:** Sakshi Choudhary `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了OASIS，一种在线激活子空间学习算法，能够在不改变前向计算的前提下，通过持续跟踪低秩子空间来压缩激活、梯度和优化器状态，从而显著降低LLM训练的峰值显存；

**💡 创新点**

创新点包括：①采用Oja规则实现无缝在线子空间更新，实时捕捉激活分布的演变；②设计投影感知优化器，在子空间变换时同步动量与方差；③将激活子空间与梯度、优化器状态统一压缩，避免单独压缩导致的性能损失；

**🔧 技术方法**

核心技术为在线PCA（Oja规则）、投影感知Adam、低秩子空间投影、随机SVD/随机投影对比以及对激活、梯度与优化器状态的统一压缩；

**📊 数据集**

实验使用了GSM8K、HumanEval、CodeAlpaca、C4等数据集，分别在LLaMA‑2‑7B、LLaMA‑3.2‑1B、Llama‑130M、Llama‑350M模型上进行finetune与pretrain；

**📈 对比分析**

与全精细调、LoRA、GaLore、LDAdam等基线相比，OASIS在finetune GSM8K任务上实现与Adam相当的准确率，峰值显存降低约2×；在pretrain C4任务上验证损失与Adam相近，内存降低10–30%；在低秩设置下，OASIS还可实现30%以上的显存节省同时保持或提升性能；

**⚠️ 局限性**

局限性包括：子空间学习率是关键超参数，过大或过小都会影响收敛；目前需要手动调优，未来可研究自适应学习率；此外，在线子空间更新在极大模型上可能带来额外的计算开销，需要进一步优化。

---

## 443. Across the Levels of Analysis: Explaining Predictive Processing in Humans Requires More Than Machine-Estimated Probabilities

**arXiv ID:** 2604.09466 | [PDF](https://arxiv.org/pdf/2604.09466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 444. Risk-seeking conservative policy iteration with agent-state based policies for Dec-POMDPs with guaranteed convergence

**arXiv ID:** 2604.09495 | [PDF](https://arxiv.org/pdf/2604.09495v1)

**作者:** Amit Sinha `[一作]` (McGill University), Aditya Mahajan `[通讯]` (McGill University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对有限记忆约束的 Dec-POMDP 的风险寻求保守策略迭代（RS-CPI）算法，能够在多智能体环境中通过迭代最佳响应和保守更新实现局部最优，并在有限的 agent-state 空间内获得接近最优的性能。

**💡 创新点**

创新点在于将风险寻求目标与保守策略迭代相结合，利用风险寻求的乐观性逃离劣势局部最优，并通过保守更新保持探索空间，从而在保证单调改进和收敛的同时提高最终性能；并证明该算法在多智能体 Dec-POMDP 上具有多项式时间与空间复杂度。

**🔧 技术方法**

使用的技术包括：风险敏感 MDP 的指数风险工具、agent-state 基于的集中 Q 函数、保守策略更新算子、风险寻求的温度衰减策略、迭代最佳响应框架和动态规划的风险转换；同时利用了 Dec-POMDP 的联合转移-观测分布与共享奖励模型。

**📊 数据集**

在 MASPlan 基准数据集上进行实验，选取了四个典型 Dec-POMDP（Dec Tiger、Recycling Robots、Cooperative Box Pushing、Mars Rovers）以及不同时间步长度和 agent-state 数量的配置。

**📈 对比分析**

与 FB-HSVI 与 PF-MAA* 两个 state‑of‑the‑art 历史基准方法进行对比。实验结果表明，RS-CPI 在使用仅 2 个 agent-state 的情况下与最佳方法表现相近，且在更大 agent-state 时性能进一步提升。Ablation 实验验证了风险寻求与保守更新的必要性，并展示了随着记忆容量增加性能提升的趋势。

**⚠️ 局限性**

局限性包括：仍受 agent-state 与观测空间乘积导致的计算量影响；在某些大规模 Dec-POMDP 上仍需更高的内存；算法收敛到的是局部最优而非全局最优；目前只在二维智能体设定下验证，扩展到更多智能体需进一步研究。

---

## 445. Dynamic Ranked List Truncation for Reranking Pipelines via LLM-generated Reference-Documents

**arXiv ID:** 2604.09492 | [PDF](https://arxiv.org/pdf/2604.09492v1)

**作者:** Nilanjan Sinhababu `[一作]` (IIT Kharagpur), Pabitra Mitra `[通讯]` (IIT Kharagpur)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM生成语义控制的参考文档作为切点，对检索结果进行自适应截断（RLT）和高效的列表式重排序，提升检索效果并减少推理成本。

**💡 创新点**

① 用LLM生成中等相关度的参考文档，充当动态阈值；② 在点式/对式重排序中实现单推理的自适应截断；③ 在列表式重排序中引入SNOW、VS‑Sliding和GPTD‑Part三种基于参考文档的并行/自适应窗口策略。

**🔧 技术方法**

大型语言模型（如Llama‑3.1‑8B‑Instruct、Llama‑3.2‑3B‑Instruct、GPT‑4o）用于参考文档生成；传统检索器（BM25、SPLADE）做第一阶段检索；mono‑T5、duo‑T5、Rank‑Zephyr、Rank‑Vicuna、Rank‑GPT等LLM重排序器做后续重排序；PSI‑Rank、SNOW、VS‑Sliding、GPTD‑Part等算法实现。

**📊 数据集**

主要在MS MARCO Passage集合的DL‑19和DL‑20主题上进行实验，同时使用BEIR的5个跨域子集（COVID、SciFact、Touché、DBPedia、另一个子集）进行泛化评估。

**📈 对比分析**

与固定截断、Greedy‑k、BiCut、Choppy、AttnCut、MtCut等RLT基线，以及传统Sliding Window、TD‑Part等列表式重排序基线进行对比。实验表明：PSI‑Rank在点式/对式重排序中在保持或提升MAP/nDCG的同时将IPQ降低35–66%；SNOW、VS‑Sliding、GPTD‑Part在列表式重排序中实现最高nDCG（如DL‑19 0.347、DL‑20 0.411）且推理速度提升达1.5–2.95倍，整体效果优于现有最优方法。

**⚠️ 局限性**

① 参考文档生成依赖LLM的推理成本，虽然实验显示不同规模模型差异不大，但在成本敏感场景仍需权衡；② 参考文档的语义阈值仍可能受检索质量影响，极端检索误差时阈值失效；③ 对长文本或多文档级语义分析尚未充分验证，需进一步扩展。

---

## 446. Agentic Jackal: Live Execution and Semantic Value Grounding for Text-to-JQL

**arXiv ID:** 2604.09470 | [PDF](https://arxiv.org/pdf/2604.09470v1)

**作者:** Vishnu Murali `[一作]` (PricewaterhouseCoopers), Vamse Kumar Subbiah `[通讯]` (PricewaterhouseCoopers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Agentic Jackal，一种通过工具增强的多步代理系统，结合 Jira 实例的实时查询执行和语义值检索（JiraAnchor），实现从自然语言到 Jira 查询语言（JQL）的自动生成与验证。

**💡 创新点**

首创公开的基于执行的文本到 JQL 基准（Jackal benchmark）以及首个具备实时值地面化和迭代执行反馈的代理架构，显著提升了在多变、隐式语义请求下的准确性，并系统归纳了主要错误类型。

**🔧 技术方法**

采用大语言模型配合工具调用（Jira Search、JiraAnchor），使用嵌入相似度检索进行值匹配，构建基于 schema 的系统提示与多步链式推理循环，实现实时执行校验与语义地面化。

**📊 数据集**

使用 Jackal benchmark（10万 NL–JQL 对）中的 1,000 条 stratified 子集（Jackal‑1K）以及 217 条专门针对分类字段的 Field‑Value 评估集，全部基于包含 200,000+ 问题的实时 Jira 实例。

**📈 对比分析**

对比 9 种前沿 LLM 在无工具单步生成与 Agentic Jackal 的两种设置下的执行准确率，Agentic 方法平均提升 2.0%（从 62.5% 到 64.4%），对短句（Short NL）提升最显著；JiraAnchor 在字段值检索上将准确率从 48.7% 提升到 71.7%，最高模型 Gemini 3 Flash 从 62.8% 提升至 71.0%。

**⚠️ 局限性**

主要局限包括：执行循环导致的平均延迟提升约 7 倍、令牌消耗提升 15 倍；仅在单一 Jira 实例上评估；JiraAnchor 在某些字段上出现回归；且工具无法解决语义解释歧义（如问题类型、文本字段选择、版本混淆）等核心错误。

---

## 447. Silence and Noise: Self-censorship and Opinion Expression on Social Media

**arXiv ID:** 2604.09465 | [PDF](https://arxiv.org/pdf/2604.09465v1)

**作者:** Xinyu Wang `[一作]` (Pennsylvania State University), Sarah Rajtmajer `[通讯]` (Pennsylvania State University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过混合方法（问卷与访谈）研究社交媒体中的自我审查现象，量化公共与私有意见的差异及其对表达的影响，并探讨其情境驱动因素与后果。

**💡 创新点**

提出“自我审查连续体”框架，将完全沉默与意见调节统一为同一维度，并结合社区支持、规模等情境变量提供实证支持，为平台干预设计提供新视角。

**🔧 技术方法**

结合定量统计（Mann‑Whitney U、Spearman 相关、逻辑回归、序数回归、调解分析）与定性主题分析（NVivo、Cohen κ），形成混合方法研究流程。

**📊 数据集**

收集了390名美国社交媒体用户的问卷数据（六个主题的私人与公开意见）以及20名受访者的半结构化访谈文本。

**📈 对比分析**

模型结果显示社区支持与发帖频率显著降低沉默概率，社区规模与沉默呈正相关；在意见差异模型中，社区支持与差异呈负相关。虽未与现有单一方法直接对比，但回归显著性与解释度良好。

**⚠️ 局限性**

样本仅限美国用户，且为自我报告数据，缺乏真实行为轨迹；情境变量测量仍可能存在偏差；跨平台差异与文化差异未充分探讨。

---

## 448. Robust Spectral Recovery for Dynamical Sampling

**arXiv ID:** 2604.09477 | [PDF](https://arxiv.org/pdf/2604.09477v1)

**作者:** HanQin Cai `[一作]` (University of Central Florida), Juntao You `[通讯]` (Wuhan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种针对时空采样中时间稀疏离群点（outlier）的鲁棒谱恢复方法，能够从有限周期格子上的子采样轨道中准确重建卷积算子谱。

**💡 创新点**

创新点在于：① 将时空采样问题通过傅里叶通道的指数混合性转化为低秩 Hankel 矩阵的鲁棒恢复；② 采用参考通道（最低秩）检测离群时间点，再完成其它通道的 Hankel 恢复；③ 通过 ASAP 与 HSNLD 算法实现对低秩 Hankel 的稀疏错误分离与补全，并给出可容忍离群比例的理论上限；④ 在恢复完成后使用 Prony/解消多项式实现谱估计。

**🔧 技术方法**

主要技术包括：低秩 Hankel 矩阵分解与恢复、稀疏错误分离（ASAP）、Hankel 结构 Newton‑like 降维法（HSNLD）、Prony 法/最小多项式消除、FFT 相关的离散傅里叶变换。

**📊 数据集**

实验数据为人工生成的周期格子卷积算子（实对称谱）和随机初始状态，利用周期采样器 S_m 产生观测，并在观测中注入离群点和不同水平的高斯噪声。

**📈 对比分析**

与常用的 Cadzow 低秩/Hankel 去噪 + Prony 基线进行比较；结果表明，本文方法在离群率 5% 且噪声水平下降时，谱恢复误差接近机器精度，恢复 SNR 显著高于 Cadzow，尤其在离群强度大或噪声低的情况下表现更优。

**⚠️ 局限性**

局限性：仅在一维周期格子、实对称卷积算子上验证，理论分析未完全覆盖高维或非周期情况；实验仅基于合成数据，缺乏真实世界案例；对参数（μ、κ）与谱特征的关系仍需进一步研究。

---

## 449. SafeAdapt: Provably Safe Policy Updates in Deep Reinforcement Learning

**arXiv ID:** 2604.09452 | [PDF](https://arxiv.org/pdf/2604.09452v1)

**作者:** Maksim Anisimov `[一作]` (Imperial College London), Matthew Wicker `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研发了一种SafeAdapt方法，实现对源任务安全约束下的持续强化学习更新。

**💡 创新点**

首次在参数空间给出源任务安全的先验证明，并通过Rashomon集限制更新，防止安全遗忘。

**🔧 技术方法**

使用了Local Invariant Domain（LID）框架、Interval Bound Propagation（IBP）、可微安全代理（smooth surrogate）以及投影梯度下降与PPO优化。

**📊 数据集**

在离散网格环境Frozen Lake与Poisoned Apple上进行实验。

**📈 对比分析**

与无约束PPO和EWC进行比较，SafeAdapt在保持源任务完全安全的前提下，能够获得与最佳方法相当或更优的下游性能。

**⚠️ 局限性**

仅适用于有限离散状态动作空间；IBP的保守性在网络较大或环境更复杂时可能导致过度限制，难以扩展到连续或更大规模的任务。

---

## 450. UIPress: Bringing Optical Token Compression to UI-to-Code Generation

**arXiv ID:** 2604.09442 | [PDF](https://arxiv.org/pdf/2604.09442v1)

**作者:** Dasen Dai `[一作]` (Chinese University of Hong Kong), Qizhen Lan `[通讯]` (UTHealth)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级光学压缩模块（Optical Compressor），将 UI 截图的 6,700 视觉 token 压缩到 256 token 并通过 LoRA 适配 Qwen3‑VL‑8B 的 LLM 解码器，实现 UI-to-Code 生成。

**💡 创新点**

创新点在于：①将光学压缩理念首次迁移到 UI‑to‑Code；②在冻结 ViT 编码器中插入可训练的卷积+注意力+Transformer 复合压缩器；③使用 LoRA 在解码器中补偿表示空间差距，整体参数仅 0.26%。

**🔧 技术方法**

技术包括深度可分离卷积、元素引导重加权（基于 OmniParser 识别的 UI 元素）、自适应平均池化、单层 Transformer 细化、LoRA 低秩适配、CLIP 评价、Bootstrap 置信区间分析。

**📊 数据集**

训练使用 50K WebSight 生成的截图‑HTML 对，评估使用 Design2Code 真实网页数据集（485 页，50 页验证集）并在 100 页 WebSight 进行跨域验证。

**📈 对比分析**

在 Qwen3‑VL‑8B 基础上与四种压缩基线（无压缩、分辨率缩放、VisionZip 选取、FastV 置零）对比；在 256 token 方案下，CLIP 0.8127，超过无压缩 7.5%、分辨率缩放 4.6%、VisionZip 10.8%；TTFT 9.1× 加速，显著提升效率。

**⚠️ 局限性**

局限性包括：①需要 50K 训练样本，领域迁移需更多数据；②仅评估输入侧压缩，对 EfficientUICoder 的完整流水线略低估；③CLIP 仅衡量全局视觉相似性，缺少细粒度文本/CSS 正确率指标；④固定 token 数 256，未考虑自适应分配。

---

## 451. You Can't Fight in Here! This is BBS!

**arXiv ID:** 2604.09501 | [PDF](https://arxiv.org/pdf/2604.09501v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 452. Intent Lenses: Inferring Capture-Time Intent to Transform Opportunistic Photo Captures into Structured Visual Notes

**arXiv ID:** 2604.09438 | [PDF](https://arxiv.org/pdf/2604.09438v1)

**作者:** Ashwin Ram `[一作]` (Saarland University, Saarland Informatics Campus), Jürgen Steimle `[通讯]` (Saarland University, Saarland Informatics Campus)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了意图透镜（Intent Lens）概念，利用大语言模型对拍摄时的隐式意图进行推断，并将其重构为可复用的交互对象，以自动生成结构化视觉笔记，并在学术会议照片上构建交互系统验证其效果。

**💡 创新点**

首次将隐式拍摄意图动态推断并转化为可复用的“意图透镜”，通过多模态推理和可复用提示生成实现意图驱动的笔记生成与意义建构，显著提升了照片收集的可用性与可探索性。

**🔧 技术方法**

采用视觉语言模型提取图像文本与视觉信息，结合大型语言模型进行链式思维推理、聚类与参数化，使用提示工程生成可复用的意图透镜；前端使用ReactFlow实现无限画布交互，并通过Semantic Scholar API等检索外部文献辅助推断。

**📊 数据集**

实验使用参与者在学术会议中自行拍摄的幻灯片、海报等照片，约每人 23 张左右；未使用公开标准数据集，全部为自收集的现场拍摄数据。

**📈 对比分析**

通过九名学者的用户研究，评估选择率、编辑/删除比例、可复用度、定制透镜数量以及 Likert 量表；结果显示编辑率仅约 14% 低、可复用度高、满意度平均 4.4/5，表明相较传统照片整理方法，意图透镜能生成更精准、可复用的结构化笔记。

**⚠️ 局限性**

局限性包括：图像信息稀疏导致意图推断误差；缺乏实时捕捉时的即时用户反馈；实验仅限学术会议照片，缺乏对其他信息丰富场景的验证；仅处理静态图片，未利用视频等动态媒介；需要更长时间使用以评估自适应效果；隐私与可追溯性等问题仍待进一步解决。

---

## 453. Yes, But Not Always. Generative AI Needs Nuanced Opt-in

**arXiv ID:** 2604.09413 | [PDF](https://arxiv.org/pdf/2604.09413v1)

**作者:** Wiebke Hutiri `[一作]` (Sony AI), Alice Xiang `[通讯]` (Sony AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了在生成式 AI 推理阶段实现细粒度 opt‑in 的架构，结合代理模型实现对创作者作品、风格与肖像的动态同意与控制，并以音乐案例演示该机制的可行性。

**💡 创新点**

创新点在于：① 把 opt‑in 从二元选择转化为基于用户输入、作品属性与使用场景的上下文细粒度决策；② 引入 inference‑time opt‑in 与代理架构（User Intent Agent、Opt‑in Agent、Consent Registry）实现实时同意校验；③ 设计抽象提示语法，将描述词、变换与限定词映射为可验证的同意条件。

**🔧 技术方法**

核心技术包括：自然语言/多模态提示解析、基于规则/检索的同意查询、代理通信与事务管理、数据库或知识图谱存储同意注册表；此外使用标注的同意规则来驱动推理时的验证逻辑。

**📊 数据集**

主要数据来源为音乐行业的案例，例如 Grimes 与 Rolling in the Deep 的合作，使用公开的音乐作品及其已登记的同意条件。论文未公开大型通用数据集，仅以案例数据为验证样本。

**📈 对比分析**

本文为概念性设计与案例演示，并未开展量化实验或性能评估；因此没有对比指标或精确性能数据。

**⚠️ 局限性**

限制包括：① 需要行业共识与统一同意数据库，跨境法律与治理难题；② 实时同意校验对技术实现与抗攻击性要求高；③ 可能产生偏见与可达性不均衡，难以覆盖所有创作者；④ 依赖用户输入的细粒度控制，实际操作复杂度高。

---

## 454. SynFlow: Scaling Up LiDAR Scene Flow Estimation with Synthetic Data

**arXiv ID:** 2604.09411 | [PDF](https://arxiv.org/pdf/2604.09411v1)

**作者:** Qingwen Zhang `[一作]` (KTH Royal Institute of Technology), Patric Jensfelt `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SynFlow，一个以运动为导向的LiDAR场景流数据生成管道，并发布了规模达4k序列（约94万帧）的Synthetic数据集；

**💡 创新点**

创新点在于将目标从视觉真实感转向运动复杂性，利用CARLA仿真产生高质量、无噪声的点云场景流标签，显著扩大训练数据量并提升跨域可迁移性；

**🔧 技术方法**

使用CARLA仿真器、规则化的拓扑、速度与交互策略，结合DeltaFlow骨干网络和标准的场景流损失；

**📊 数据集**

主要使用SynFlow-4k合成数据；在零样本和标签有限的真实数据集nuScenes、TruckScenes、Aeva上进行验证；

**📈 对比分析**

与自监督方法（SeFlow、VoteFlow、TeFlow）以及全监督基线（DeFlow、Flow4D、DeltaFlow）相比，SynFlow-4k零样本表现接近或超越监督基线；在5%/10%/20%标签微调时，能在nuScenes、TruckScenes上显著提升误差（如nuScenes EPE降至0.157，比DeltaFlow提升27%；TruckScenes EPE降至0.266，比DeltaFlow提升33%）；

**⚠️ 局限性**

局限性包括：仍依赖仿真物理而非真实传感器噪声，且未实现闭环动态生成；在非车辆类（如行人）运动的模拟仍受限；未来可通过反馈驱动仿真和扩展到其他领域。

---

## 455. Physics-Informed Reinforcement Learning of Spatial Density Velocity Potentials for Map-Free Racing

**arXiv ID:** 2604.09499 | [PDF](https://arxiv.org/pdf/2604.09499v1)

**作者:** Shathushan Sivashangaran `[一作]` (Virginia Commonwealth University), Azim Eskandarian `[通讯]` (Virginia Commonwealth University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于物理信息强化学习的地图无关无人车赛道控制方法，利用即时深度传感器信息和非几何奖励学习车辆时空动力学。

**💡 创新点**

创新点在于将深度测距数据解释为光谱信号，通过物理信息奖励和隐式碰撞截断，消除模拟到现实差距，并通过大规模训练提取运动学与动力学的功能分支。

**🔧 技术方法**

采用的技术包括Proximal Policy Optimization、物理精确的Bullet仿真、稀疏的非几何奖励、光谱密度潜能编码、以及两层MLP网络。

**📊 数据集**

使用的数据集为高频LiDAR（RPLIDAR S2）扫描数据，模拟环境共计2千万步、超过15k次碰撞，并在比例缩放的实车上进行验证。

**📈 对比分析**

在未见过的赛道上零样本迁移，DRL策略比人类演示快12%，比经典几何PID快26%，并在模拟中比SAC/BC快约15-30%，表现稳定且一致。

**⚠️ 局限性**

局限性包括对高频深度传感器的依赖、对物理引擎误差的剩余敏感、以及在极端复杂赛道或多车场景下可能仍需更多样本或细粒度仿真。

---

## 456. BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation

**arXiv ID:** 2604.09497 | [PDF](https://arxiv.org/pdf/2604.09497v1)

**作者:** Hippolyte Gisserot-Boukhlef `[一作]` (Artefact Research Center), Pierre Colombo `[通讯]` (Cohere)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种大语言模型在零样本任务中的表现进行评估，并提出一种基于BERT的轻量级评估器BERT-as-a-Judge，替代传统的正则表达式和LLM-as-a-Judge方法。

**💡 创新点**

创新点在于利用BERT编码器对候选答案、参考答案和问题三元组进行二分类评估，显著减少对输出格式的依赖并保持高计算效率。

**🔧 技术方法**

使用BERT编码器（EuroBERT 210M）进行微调、正则表达式解析、LLM-as-a-Judge对比以及多种评估指标（ROUGE、Math-Verify等）。

**📊 数据集**

实验数据覆盖15个任务集，包括MMLU、ARC、TruthfulQA、HotpotQA、SQuAD-v2、GSM8K、MATH、AIME等，生成36个模型的零样本输出。

**📈 对比分析**

与正则表达式和LLM-as-a-Judge相比，BERT-as-a-Judge在所有任务上均实现最高准确率（多选任务≈99%，数学≈94%，上下文提取≈90%），并在不同规模模型、任务类型和格式下保持稳健。

**⚠️ 局限性**

局限性在于仅针对英语客观答案任务，未涵盖开放式生成、多语言、跨模态或更广泛的任务场景。

---

## 457. RecaLLM: Addressing the Lost-in-Thought Phenomenon with Explicit In-Context Retrieval

**arXiv ID:** 2604.09494 | [PDF](https://arxiv.org/pdf/2604.09494v1)

**作者:** Kyle Whitecross `[一作]` (University of Massachusetts Amherst), Negin Rahimi `[通讯]` (University of Massachusetts Amherst)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RecaLLM，一种在推理过程中交替进行显式检索的语言模型，使用可复制的检索段解决长上下文推理中的信息丢失（lost‑in‑thought）问题。

**💡 创新点**

创新点在于：①引入显式检索（recall spans）并通过约束解码确保检索信息精确复制；②设计复合奖励函数，将检索质量与答案质量一起作为强化学习的信号；③在推理过程中动态决定何时检索，显著提升长上下文使用效率。

**🔧 技术方法**

采用的技术包括：监督微调+GRPO强化学习、约束解码（logit masking）、复合奖励（格式、答案、检索质量）、数据增强与多任务训练、特殊标记用于标记检索段。

**📊 数据集**

使用多样化任务集合（单跳/多跳问答、检索重排、短文算术、聚合任务等）以及公开的长上下文基准 RULER 和 HELMET，训练样本最大长度仅 10K。

**📈 对比分析**

在 RULER、HELMET 两大长上下文基准上与基线 LLM、LoongRL、ProLong、QwenLong 等模型对比，RecaLLM 在 7–8B 规模上平均提升约 15–20 分，且在 128K 上下文仍保持高性能，甚至超过更大模型，显示出显著的性能提升。

**⚠️ 局限性**

局限性：在极长上下文（128K）时检索使用率下降，导致准确率随之下滑；在长篇生成任务（LongQA、Summ）提升有限；训练仅在 ≤10K 上下文，模型对极长输入的泛化受限。

---

## 458. Process Reward Agents for Steering Knowledge-Intensive Reasoning

**arXiv ID:** 2604.09482 | [PDF](https://arxiv.org/pdf/2604.09482v1)

**作者:** Jiwoong Sohn `[一作]` (ETH Zürich), Michael Moor `[通讯]` (Heidelberg University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种Process Reward Agent（PRA）框架，能够在保持推理模型参数冻结的前提下，在线动态评估并给出检索驱动的逐步奖励，指导推理过程。

**💡 创新点**

创新点在于将检索与奖励完全解耦离开推理模型，实时给每一步打分并结合树搜索实现可扩展推理；同时PRA可以在不重新训练后端模型的情况下迁移到不同规模的模型。

**🔧 技术方法**

使用的大型语言模型（如Qwen3-4B-Instruct）进行链式推理，结合检索增强的过程奖励模型、Beam搜索、教师模型生成的步骤/检索标签，以及对数概率差分阈值判定检索需求。

**📊 数据集**

训练使用MedQA训练集；评估时采用MedQA、MedBullets、MedMCQA、MMLU-Med、GPQA、Lancet、NEJM等七个医学推理基准。

**📈 对比分析**

与直接回答、Chain‑of‑Thought、RAG及其Self‑Consistency等基线对比；在MedQA上PRA达80.8%准确率，突破同规模4B模型的最高水平；在七个基准上平均提升约4.8分，对不同规模模型（0.5B–8B）均显著提升，最大相对提升达25.7%。

**⚠️ 局限性**

局限性在于仍无法完全消除幻觉或不正确的中间步骤；全步检索成本高；该方法为研究原型，尚未验证可直接用于临床决策。

---

## 459. Online3R: Online Learning for Consistent Sequential Reconstruction Based on Geometry Foundation Model

**arXiv ID:** 2604.09480 | [PDF](https://arxiv.org/pdf/2604.09480v1)

**作者:** Shunkai Zhou `[一作]` (Peking University), Hongbin Zha `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Online3R，利用可学习的视觉提示对冻结的几何基础模型进行在线微调，实现对新场景的自适应连续重建。

**💡 创新点**

创新点在于：① 在冻结的 3D 基础模型中插入轻量视觉提示实现在线自适应；② 设计局部与全局自监督一致性约束，利用多视角融合的伪真值和关键帧一致性进行在线优化。

**🔧 技术方法**

采用视觉提示调优 (VPT)、自监督局部/全局一致性损失、MASt3R‑SLAM 框架、AdamW 优化器、单目 RGB 输入等技术。

**📊 数据集**

使用 TUM RGB‑D、NRGBD、7‑Scenes 等室内序列数据集进行评估。

**📈 对比分析**

与 ORB‑SLAM3、DeepV2D、DeepFactors、DPV‑SLAM、GO‑SLAM、DROID‑SLAM、MASt3R‑SLAM 等方法比较，ATE、Accuracy、Completion、Chamfer 等指标均达到或超过现有 SOTA；在标定和无标定场景下均取得最小 ATE，帧率约为 10 FPS。

**⚠️ 局限性**

局限性：在线学习带来额外计算开销；当前方法仅适用于静态场景，无法处理动态 4D 场景。

---

## 460. Incremental Semantics-Aided Meshing from LiDAR-Inertial Odometry and RGB Direct Label Transfer

**arXiv ID:** 2604.09478 | [PDF](https://arxiv.org/pdf/2604.09478v1)

**作者:** Muhammad Affan `[一作]` (University of Twente), George Vosselman `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一个基于LiDAR-惯导里程计和RGB VFM分割的增量语义辅助网格重建管线

**💡 创新点**

创新点在于将VFM生成的panoptic标签直接投影到LiDAR点云，并通过标签感知TSDF、动态截断以及时间标签合并实现几何与语义的协同融合

**🔧 技术方法**

采用了Fast‑LIO2惯导、OneFormer（VFM）panoptic分割、标签感知TSDF（Dirichlet融合、动态截断、标签合并）、Marching Cubes与USD导出等技术

**📊 数据集**

使用了 Oxford Spires（Christ Church College）和 NTU VIRAL 两个室内大场景数据集进行实验

**📈 对比分析**

与 ImMesh 与 Voxblox 进行准确率/完整率/F1 的对比，取得 14.54 cm 的准确率、98.55% 的完整率、98.58% 的 F1，明显优于基线

**⚠️ 局限性**

局限在于对里程计漂移、相机视角覆盖与标定精度敏感，薄结构、镜面和拥挤环境下的重建效果仍需改进

---

## 461. SafeMind: A Risk-Aware Differentiable Control Framework for Adaptive and Safe Quadruped Locomotion

**arXiv ID:** 2604.09474 | [PDF](https://arxiv.org/pdf/2604.09474v1)

**作者:** Zukun Zhang `[一作]` (University of Hong Kong), Mingqiao Mo `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SafeMind，一种基于差分可微随机控制边界函数（CBF）的四足机器人安全控制框架，在不确定性、感知噪声和语义上下文下实现可微安全约束，支持端到端训练。

**💡 创新点**

创新点：①将概率边界函数与方差感知相结合，显式建模知识不确定性与随机扰动；②通过语义编码器将高层语义信息直接映射为安全边界参数；③引入元自适应学习模块，在线调整风险敏感度，保证跨环境鲁棒性；④在实时硬件上实现200 Hz差分可微QP，保持梯度流通。

**🔧 技术方法**

技术：控制边界函数（CBF）+随机微分方程；可微二次规划（QP）优化；梯度反向传播/隐式微分；语义上下文编码（多模态网络）；元学习/自适应更新；端到端深度强化学习/监督学习；分布式动力学与不确定性估计。

**📊 数据集**

数据集与环境：利用混合地形仿真（12种地形类型）、不确定性注入（质量、摩擦、感知噪声）、语义任务（8种高层指令）以及真实硬件（Unitree A1、ANYmal C）上的实验；在仿真中采集约20k条轨迹，在硬件上执行多场景试验。

**📈 对比分析**

对比方法：确定性CBF-QP、BarrierNet、Nominal/MPC、Safety‑MPC、Robust‑MPC、Stochastic‑MPC、Ensemble‑CBF、RL‑CBF、分布式RL。评估指标包括安全违规率、跟踪RMSE、能耗、适应时间等。实验显示SafeMind在12种地形上安全违规率降低约5×，跟踪误差下降约10%，能耗减少10–15%，并在语义任务和极端不确定性下保持低违规率。

**⚠️ 局限性**

局限：①安全保证仅在每个接触阶段内成立，跨接触模式的理论证明有限；②元自适应更新对快速突发扰动仍需一定缓冲时间；③在极端动态环境（如极高摩擦、极大外力冲击）下可能出现解算不收敛；④需要较大计算资源（GPU/高性能CPU）才能保持200 Hz；⑤语义编码器依赖预训练的多模态模型，对未见语义场景的泛化仍有限。

---

## 462. Three Modalities, Two Design Probes, One Prototype, and No Vision: Experience-Based Co-Design of a Multi-modal 3D Data Visualization Tool

**arXiv ID:** 2604.09426 | [PDF](https://arxiv.org/pdf/2604.09426v1)

**作者:** Sanchita S. Kamath `[一作]` (University of Illinois Urbana-Champaign), JooYoung Seo `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过经验驱动共设计（EBCD）与五名盲/低视力（BLV）研究者共同开发了一款基于 Web 的多模态 3D 数据可视化工具，并将低保真触觉原型与高保真数字原型进行迭代对比，最终实现了可视化与听觉、触觉相结合的交互体验。

**💡 创新点**

① 将 EBCD 方法从医疗服务迁移到 HCI，形成了一套可复制的“触觉→数字”知识转移流程；② 在可视化中首次结合参考音化、立体/体积音频以及可配置缓冲区聚合，满足 BLV 用户在 3D 方向感、梯度追踪、峰值定位和区域对比等核心分析任务；③ 通过多层模块化架构实现快速迭代，验证了 Web 原生实现可行。

**🔧 技术方法**

核心技术包括：WebGL 渲染引擎、Web Audio API 语音合成与空间音频、键盘/音频驱动的导航与缓冲区管理、可配置的音频参数映射、基于事件总线的分层架构；辅助技术包括自动化 AI 对话助手（Gemini 2.5 Pro）和自定义触觉原型（泡沫板+塑料管）。

**📊 数据集**

评估数据集主要为连续高度场表面：1）苯光谱（Benzene spectroscopy）数据；2）高斯曲面（Gaussian surface）模拟；3）天气数据（weather）等；所有数据均为无噪声、密集采样的科学数据。

**📈 对比分析**

比较方法：采用两轮共设计会议（低保真与高保真原型对比），通过情境提示、脑写作与任务引导收集定性反馈；对关键功能（参考音化、立体/体积音频、缓冲区）进行任务测试（定位、梯度追踪、区域对比），记录准确性与学习曲线；性能表现：BLV 设计者报告在定位与分析准确度上显著提升，学习时间缩短，且在多模态交互下认知负荷下降。由于缺乏量化指标，评估主要基于专家主体感知与定性访谈。

**⚠️ 局限性**

局限性：① 样本为技术熟练的 BLV 专家，缺乏新手与多样化视觉障碍人群验证；② 仅测试了连续、密集的高度场数据，对稀疏或噪声数据的适用性未知；③ 研究为横断面两次会议，未评估长期使用中的适应性、听觉疲劳等；④ 体积音频深度编码仅使用音量，需结合混响进一步验证；⑤ 缓冲区交互仍存在两步选择、边界感知不足、聚合模式不明等 UX 问题，需要进一步优化。

---

## 463. NOMAD: Generating Embeddings for Massive Distributed Graphs

**arXiv ID:** 2604.09419 | [PDF](https://arxiv.org/pdf/2604.09419v1)

**作者:** Aishwarya Sarkar `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为NOMAD的分布式内存图嵌入框架，旨在为大规模分布式图生成嵌入，能够有效处理具有数百万到数十亿条边的图。

**💡 创新点**

NOMAD通过实现基于邻近性的模型，提出了多种实用的权衡，以提高大规模图嵌入方法的可扩展性和通信开销，展示了在性能和嵌入质量上的显著提升。

**🔧 技术方法**

使用了消息传递接口（MPI）进行分布式图嵌入，结合了随机游走和基于邻近性的模型（如LINE算法）来生成嵌入。

**📊 数据集**

在多个真实世界图数据集上进行了评估，包括社交网络、引用网络和网页网络等，具体数据集包括pubmed、photo、computers、physics、youtube等。

**📈 对比分析**

与现有的图采样和嵌入系统（如多线程的LINE和node2vec、分布式的PBG）进行了比较，NOMAD在CPU上相对于这些基线实现了10100倍的速度提升，在真实世界图上实现了12-370倍的端到端速度提升，同时在嵌入质量上保持竞争力。

**⚠️ 局限性**

NOMAD的局限性在于其在极大规模下可能面临负载不平衡的问题，尤其是在图的分区不均匀时，可能导致性能下降。

---

## 464. PhysInOne: Visual Physics Learning and Reasoning in One Suite

**arXiv ID:** 2604.09415 | [PDF](https://arxiv.org/pdf/2604.09415v1)

**作者:** Siyuan Zhou `[一作]` (vLAR Group), Bo Yang `[通讯]` (vLAR Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并发布了一个规模巨大的合成数据集（153,810 个多物体多物理场景，2 百万段视频），涵盖 71 种日常物理现象，用于视觉物理学习与推理。

**💡 创新点**

创新点在于：① 统一生成 71 种基本物理现象的 3,284 个多物理活动；② 结合多种模拟引擎（UE5 Chaos、MPM、SPH）实现真实多物体交互；③ 提供完整 3D 轨迹、几何、语义、属性和文本注释，形成前所未有的物理基准。

**🔧 技术方法**

技术包括：Unreal Engine 5 物理仿真、Taichi MPM 与 SPH 流体模拟、多相机采样与高分辨率渲染、文本生成与校对、以及多任务评估框架（PMF、FVD、LPIPS 等）。

**📊 数据集**

使用的新数据集即论文中提出的 “PhysicsInOne” 数据集；对比基线使用 SVD、CogVideoX、Wan2.2 等主流视频生成/预测模型。

**📈 对比分析**

通过微调（LoRA、SFT、FLT）提升模型在物理一致性（PMF）上显著提高，FVD 与人类评价也同步下降；但在磁力、流体等物理域表现最好，机械与光学仍较弱，未来/短期预测在新视角下精度明显下降。

**⚠️ 局限性**

局限性包括：① 仿真误差和物理逼真度受限；② 复杂多物体场景下属性估计仍不够精准；③ 动作迁移难以保留细粒度物理动态；④ 对真实世界数据的泛化能力尚未验证。

---

## 465. Do AI Coding Agents Log Like Humans? An Empirical Study

**arXiv ID:** 2604.09409 | [PDF](https://arxiv.org/pdf/2604.09409v1)

**作者:** Youssef Esseddiq Ouatiti `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AI编码代理在日志实践中的行为进行了实证研究，分析了其与人类开发者在日志添加、修改和删除方面的差异，并评估了自然语言指令对其日志行为的影响。

**💡 创新点**

发现AI代理日志变更频率低于人类且自然语言指令既稀缺又易被忽视，提出仅靠提示无法保证日志可观测性，需要采用确定性保障机制。

**🔧 技术方法**

使用正则表达式静态分析日志语句，LLM评审（GPT‑4o、GLM‑4.7、DeepSeek）进行指令意图判定，基于统计检验和生存分析评估日志修订生命周期。

**📊 数据集**

采用AIDev数据集中的4,550个AI代理PR和3,276个人工PR，覆盖81个主流Python/Java/JavaScript/TypeScript仓库。

**📈 对比分析**

与人类PR对比时，采用归一化得分（Agentic/Agentic+Human）评估日志出现率、密度、信息长度、日志级别和语法位置；结果显示代理日志出现率低58.4%，但在两者均有日志时密度略高30%，指令合规率仅33%。

**⚠️ 局限性**

局限包括：正则匹配可能漏检自定义日志框架；仅分析公开PR，忽略IDE聊天指令；研究聚焦主流语言和大星标仓库，可能不适用于小型或其他语言项目。

---

## 466. XFED: Non-Collusive Model Poisoning Attack Against Byzantine-Robust Federated Classifiers

**arXiv ID:** 2604.09489 | [PDF](https://arxiv.org/pdf/2604.09489v1)

**作者:** Israt Jahan Mouri `[一作]` (Bangladesh University of Engineering and Technology), Muhammad Abdullah Adnan `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对联邦学习中的模型中毒攻击，提出了不需要攻击者之间通信的非协同攻击模型，并实现了首个聚合无关的攻击框架 XFED。

**💡 创新点**

创新点在于：①将攻击者视为独立、无协作的恶意客户端；②通过仅观察到的全局模型更新，自适应估计攻击强度（利用中位数+MAD）；③使用逆单位向量或逆符号方向的扰动，构造有效的攻击更新，且不依赖任何服务器端聚合规则。

**🔧 技术方法**

采用的技术包括：本地模型更新的计算、扰动向量（逆单位向量、逆符号向量）构造、基于全局模型差分的中位数与MAD鲁棒统计量来估计缩放系数 μ、以及对多种联邦学习聚合规则与防御算法的兼容性测试。

**📊 数据集**

实验数据集包括：Purchase‑100、MNIST、EMNIST、Fashion‑MNIST、CIFAR‑10 以及 HAR（人类活动识别）等多种图像与行为识别数据集，覆盖跨设备与跨机房两种 FL 场景。

**📈 对比分析**

通过在 6 种聚合规则（FedAvg、Multi‑Krum、Trimmed‑Mean、Median、Clipped‑Clustering、SignGuard）与 4 种防御（FLTrust、FLAME、FoolsGold、FreqFed）上进行对比，XFED 的攻击影响力（I_θ）普遍高于现有 10+ 传统或最新攻击（如 LIE、Min‑Max、MPAF、PoisonedFL 等），在大多数配置下能突破除 FLTrust 以外的所有防御；在交叉设备设置下，XFED 仍保持领先优势。

**⚠️ 局限性**

局限性包括：①攻击效果仍依赖于攻击者能够获得一定数量的全局模型快照；②对 λ（攻击激进程度）与扰动向量的选择敏感；③在极度非 IID 或极少恶意客户端（<5%）的场景下，攻击影响可能下降；④未考虑服务器端可能的自适应防御或模型监测策略。

---

## 467. Realizing Immersive Volumetric Video: A Multimodal Framework for 6-DoF VR Engagement

**arXiv ID:** 2604.09473 | [PDF](https://arxiv.org/pdf/2604.09473v1)

**作者:** Zhengxian Yang `[一作]` (Tsinghua University), Tao Yu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了可实现6-DoF沉浸式体积视频（IVV）的概念，并搭建了完整的制作流水线。

**💡 创新点**

创新点在于：①基于高分辨率5K 60FPS的多视角多模态数据集ImViD；②利用高效的高维Gaussian时空表示实现动态光场重建，配合光流引导稀疏初始化、联合相机时序校准和多项式时空监督；③首次将声音场重建纳入该框架，实现完整的视听沉浸。

**🔧 技术方法**

采用的技术包括：多摄像头同步捕捉、COLMAP结构光、VideoFlow光流、3D Gaussian Splatting (3DGS) 及其4D扩展、光流与深度约束、HRTF与RIR合成音频、PyTorch实现的联合优化。

**📊 数据集**

使用的数据集为自研的ImViD（7个室内外场景，1-5分钟，5K 60FPS），并与公开的N3V、MPEG-GSC、MeetRoom、Google Immersive进行对比评测。

**📈 对比分析**

与Gaussian4D、4DGS、STG、Ex4DGS等基准方法相比，ImViD+本文框架在PSNR、SSIM、LPIPS等指标上均取得显著提升，尤其在ImViD数据集上平均PSNR提升4.2dB、LPIPS降低约53%。

**⚠️ 局限性**

局限性包括：①训练时需将大量图像缓存到显存，受单机显存限制难以一次处理更多帧；②整体流程对预处理和训练速度未做专门优化，需进一步加速与可扩展化。

---

## 468. Adaptor: Advancing Assistive Teleoperation with Few-Shot Learning and Cross-Operator Generalization

**arXiv ID:** 2604.09462 | [PDF](https://arxiv.org/pdf/2604.09462v1)

**作者:** Yu Liu `[一作]` (Jilin University), Xuan Song `[通讯]` (Jilin University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Adaptor 框架，实现少量样本跨操作员的意图识别与共享控制的辅助遥操作系统。

**💡 创新点**

创新点包括：① 通过在演示轨迹上注入噪声构造意图扰动分布；② 采用几何感知关键帧提取压缩轨迹；③ 结合 Vision‑Language‑Action（VLA）架构，分别训练意图专家与动作专家；④ 异步执行设计降低操作员工作量并提升鲁棒性。

**🔧 技术方法**

使用的技术：噪声注入、几何关键帧提取（基于 Directed Hausdorff 距离）、VLM（SigLIP）+ Transformer、Intention Expert 与 Action Expert、LoRA 微调、Conditional Flow Matching、行为克隆与流匹配。

**📊 数据集**

数据集与实验平台：ALOHA 机器人仿真（Trossen ViperX 双臂）、AgileX PIPER 真实双臂、Realman 双臂；六个操纵任务（插槽、立方体传输、笔拆盖、衬衫折叠、笔归档、立方体堆叠）；11 名参与者在不同练习时长下完成 30 次试验。

**📈 对比分析**

与 Full Teleop（直接逆运动学）和 HAJL（基于扩散模型的共享控制）进行对比；评估指标为任务成功率、操作时间和用户满意度。Adaptor 在所有任务上均取得最高成功率、最短操作时间，并在各操作员水平下方差显著降低，平均成功率提升 41.92%，操作时间缩短 32.17%。

**⚠️ 局限性**

局限性：当操作员行为与演示分布差异过大时意图识别仍易失误；系统在完全自主执行阶段缺乏实时干预，导致安全感与信任感下降；对极端噪声或新人操作的鲁棒性尚待进一步提升。

---

## 469. From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models

**arXiv ID:** 2604.09459 | [PDF](https://arxiv.org/pdf/2604.09459v1)

**作者:** Chenchen Zhang `[一作]` `[通讯]` (Independent Researcher), Chenchen Zhang (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了2024–2026年间关于大型语言模型（LLM）强化学习（RL）中信用分配（CA）的方法，涵盖推理RL和代理RL两大范式，并提出三大可复用资源：机器可读论文清单、CA报告清单以及基准评测协议。

**💡 创新点**

① 将CA方法按两维（细粒度 × 计算方法）构建二维分类法；② 将推理RL与代理RL的CA挑战对比并指出代理RL中出现的新技术方向；③ 提供完整的论文清单与指标，帮助后续工作统一评估与比较。

**🔧 技术方法**

主要技术包括：Monte Carlo、时序差分（TD）、LLM-as-Critic、游戏理论（Shapley、对比实验）和信息理论方法；对每类方法进行细粒度分配；并提出基准任务与决策树以指导方法选择。

**📊 数据集**

数据集涵盖推理RL任务（GSM8K、MATH、AIME、CodeContests）与代理RL任务（WebArena、WebShop、ALFWorld、SWE-bench、ColBench、HotpotQA、2WikiMQA 等），以及多代理评测环境。

**📈 对比分析**

通过对47种方法的实验结果对比，展示了从粗粒度（Episode级）到细粒度（Token/Turn/Segment）以及从前向估计到后向推断的性能提升趋势；在推理RL中，Token/Segment级方法往往带来 5–15% 的准确率提升；在代理RL中，后向推断和LLM-as-Critic 方法在长序列任务中相较于传统 GRPO/REINFORCE 提升 10–20% 的成功率或 F1 分数。

**⚠️ 局限性**

局限性包括：① 论文覆盖仅截至2026年初，后续新方法可能未包含；② 基准任务分散，难以统一对比；③ 许多方法对计算资源要求极高，缺乏可扩展性评估；④ 评测多基于自研或实验室内部数据，缺少公开可复现的基线；⑤ 对多代理情境的理论分析尚不充分，需更多实证验证。

---

## 470. AdaCubic: An Adaptive Cubic Regularization Optimizer for Deep Learning

**arXiv ID:** 2604.09437 | [PDF](https://arxiv.org/pdf/2604.09437v1)

**作者:** Ioannis Tsingalis `[一作]`, Corentin Briat `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种自适应的三次正则化二阶优化器 AdaCubic，能够动态调整三次正则化项的权重并避免鞍点。

**💡 创新点**

创新点在于利用辅助约束优化问题通过拉格朗日乘子自动更新三次正则化参数，同时只估计Hessian对角线并采用Hutchinson方法降低计算成本。

**🔧 技术方法**

核心技术包括三次正则化 Newton 方法、拉格朗日对偶、Hutchinson 随机估计 Hessian 对角线、对偶变量根搜索与自适应信赖域更新。

**📊 数据集**

在计算机视觉（ImageNet、CIFAR-10/100）、自然语言处理（GLUE、SQuAD、MNLI 等）、信号处理（音频 CMI 数据集）等多种任务上进行实验。

**📈 对比分析**

与 SGD、Adam、AdaHessian 等基准优化器相比，AdaCubic 在大多数任务上取得了更高或相近的准确率/精度，且不需要学习率调优，训练收敛更快，整体性能优于传统第一阶方法，接近或略低于 AdaHessian。

**⚠️ 局限性**

局限包括对 Hessian 对角线近似的依赖，导致无法捕获跨参数的非对角曲率信息；相较于 SGD，额外的梯度反向传播和内存开销更大；在某些高阶交互强的任务（如 CIFAR-100、Transformer 任务）性能略逊于精细调优的 AdaHessian。

---

## 471. SCoRe: Clean Image Generation from Diffusion Models Trained on Noisy Images

**arXiv ID:** 2604.09436 | [PDF](https://arxiv.org/pdf/2604.09436v1)

**作者:** Yuta Matsuzaki `[一作]` (Kyushu University), Shumpei Takezaki `[通讯]` (Kyushu University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SCoRe，一种在扩散模型采样时进行频谱截止重建的无训练方法，旨在消除训练数据噪声对生成结果的影响。

**💡 创新点**

创新点在于利用扩散模型在频域的谱偏差，先通过低通滤波抑制高频噪声，再用 SDEdit 在对应时间步重生成高频细节；通过 RAPSD 推导出截止频率与 SDEdit 初始化时间步的对应关系，避免噪声过度注入。

**🔧 技术方法**

使用技术包括 Radially Averaged Power Spectral Density (RAPSD) 频谱分析、频率截止滤波、SDEdit 编辑、以及标准扩散模型（DDPM）生成。

**📊 数据集**

实验数据集包括 CIFAR-10（合成加噪）和 SIDD（真实摄像噪声）。

**📈 对比分析**

与双边滤波、Noise2Void 后处理、标准扩散模型、SDEdit、NR‑GAN 等方法进行 FID 比较；结果表明 SCoRe 在所有噪声类型和噪声比例下均优于基线，并在 SIDD 上将 FID 从 30.1 降至 16.2 左右。

**⚠️ 局限性**

局限性包括对截止频率的选取仍需经验调节；主要针对高频噪声，低频噪声可能残留；未对极高噪声比例或其他噪声模型的鲁棒性进行深入验证。

---

## 472. Automated Instruction Revision (AIR): A Structured Comparison of Task Adaptation Strategies for LLM

**arXiv ID:** 2604.09418 | [PDF](https://arxiv.org/pdf/2604.09418v1)

**作者:** Solomiia Bilyk `[一作]` (Eleks Ltd), Taras Firman `[通讯]` (Eleks Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Automated Instruction Revision (AIR)，一种基于规则诱导的提示自适应流水线，通过从少量标注样本中提取、聚合和迭代修订可解释的决策规则，生成任务特定的执行提示。

**💡 创新点**

创新点在于把任务适配过程转化为可读的自然语言规则生成与精细化，而不是传统的提示搜索、检索或参数微调；同时提供了多阶段聚合与迭代修订机制，实现了结构化、可解释的适配。

**🔧 技术方法**

主要技术包括：输入/输出嵌入与聚类、基于对比的规则诱导、LLM驱动的规则编译与合并、基于误差案例的规则细化、以及将最终规则编码为可执行提示。

**📊 数据集**

使用了五个多样化基准：客户支持分类（8类），闭卷知识问答，结构化信息抽取，PII 识别（PUPA），以及金融事件逻辑推理（BizFinBench.v2）。

**📈 对比分析**

将 AIR 与直接提示、KNN检索、DSPy BootstrapFewShot、MIPROv2、GEPA、TextGrad、以及全参数微调进行对比。实验显示：AIR 在标签重映射分类上与 GEPA 相近且优于微调；在闭卷问答和信息抽取中表现逊色；在 PII 与事件推理任务中排名中游；整体表现说明适配方法的优劣高度依赖任务特性。

**⚠️ 局限性**

局限包括：规则诱导对噪声与标签不一致敏感；规则聚合与冲突消除难以保持清晰；聚类与迭代参数选择缺乏系统化；在需要大量事实知识、结构重构或数据集特定标注习惯的任务中，AIR 的优势不明显。

---

## 473. Policy-Aware Edge LLM-RAG Framework for Internet of Battlefield Things Mission Orchestration

**arXiv ID:** 2604.09493 | [PDF](https://arxiv.org/pdf/2604.09493v1)

**作者:** Om Solanki `[一作]` (Tennessee Tech University), Maanak Gupta `[通讯]` (Tennessee Tech University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个基于策略感知的边缘LLM检索增强生成框架PA-LLM-RAG，用于IoBT任务指令的安全调度。

**💡 创新点**

集成检索增强推理、边缘部署的LLM生成和独立JudgeLLM验证三层，形成闭环政策遵从与实时遥测验证的管控链。

**🔧 技术方法**

采用检索增强生成（RAG）、多模型LLM（Gemma-2B、LLaMA-3.1-8B、Mistral-7B、Qwen-2.5-7B）与Ollama CPU推理，配合RoboDK仿真。

**📊 数据集**

在RoboDK仿真中自建的多兵器、传感器资产与政策规则JSON数据库，使用10条标准化任务指令集。

**📈 对比分析**

对四个开源LLM在10种控制场景下进行混合与严格成功率、延迟、精度/召回/F1评估；Gemma-2B 100%成功、4.17s延迟、F1 0.95，其他模型表现递减。

**⚠️ 局限性**

仅在仿真环境验证，规模有限，缺乏物理平台、多节点扩展，以及对对抗性输入与模型更新安全未充分评估。

---

## 474. Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks

**arXiv ID:** 2604.09487 | [PDF](https://arxiv.org/pdf/2604.09487v1)

**作者:** Jan Schneider `[一作]` (Max Planck Institute for Intelligent Systems), Dieter Büchler `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了Generalized Actuator Network（GeAN），利用仅关节位置信息训练神经网络模型，实现肌肉与肌腱驱动的四自由度机器人在仿真中的精确控制，并成功完成零射击迁移，完成目标定位与球进杯两项任务。

**💡 创新点**

创新点在于不依赖扭矩传感器，采用位置损失直接优化运动轨迹，并将GeAN与传统刚体仿真结合，首次实现多自由度肌肉驱动机器人的sim‑to‑real迁移；同时引入GeAN集成以降低模型不确定性。

**🔧 技术方法**

技术包括基于Δ历史归一化的监督学习训练GeAN、位置损失与扭矩损失对比、MuJoCo GPU仿真与PPO强化学习、GeAN集成与随机采样策略。

**📊 数据集**

使用真实机器人采集约2500条2秒开放式轨迹（约1.4小时）用于GeAN训练，并用800条额外轨迹评估模型；所有数据仅包含关节位置与控制信号，无扭矩测量。

**📈 对比分析**

与无监督Actuator Net（UAN）以及仅使用扭矩损失的GeAN对比，位置损失训练的GeAN在单步误差降低6%、500步误差降低29%；在真实机器人上目标定位成功率约90%，球进杯成功率约75%，显著优于基线。

**⚠️ 局限性**

局限性包括对关节位置测量精度依赖较高，模型在系统老化或张力失效时需要重新标注；对球与绳索动力学建模不足导致的失误；在轨迹外或极端运动下表现欠佳。

---

## 475. Offline Local Search for Online Stochastic Bandits

**arXiv ID:** 2604.09423 | [PDF](https://arxiv.org/pdf/2604.09423v1)

**作者:** Gerdus Benadè `[一作]` (Boston University), Thomas Lavastida `[通讯]` (University of Texas at Dallas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种将离线局部搜索算法转换为在线组合多臂赌博机算法的通用框架。

**💡 创新点**

创新点在于实现了对(β,γ)-改进邻域的利用，得到O(log^3 T)的近似调度/基/聚类的γ-遗憾。

**🔧 技术方法**

主要技术是基于局部搜索的迭代、成功采样、成功淘汰子阶段以及集中化的概率分析。

**📊 数据集**

未使用公开数据集，论文仅在三类理论模型上展示：单机调度、基于基的最小成本、k-中心聚类。

**📈 对比分析**

与传统离线-to-在线框架（如贪婪或线性优化）相比，得到的γ-遗憾从O(T^{2/3})下降到O(log^3 T)。

**⚠️ 局限性**

局限性包括对已知T和成本上界的假设、需要β-改进邻域的存在、以及对NP-hard问题只能得到近似解决方案。

---

## 476. Tango: Taming Visual Signals for Efficient Video Large Language Models

**arXiv ID:** 2604.09547 | [PDF](https://arxiv.org/pdf/2604.09547v1)

**作者:** Shukang Yin `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种高效视频大型语言模型（Video LLM）的预训练后令牌压缩框架Tango，显著减少视频令牌数量。

**💡 创新点**

创新点在于：① 引入基于多模态分布的多样性驱动选取策略，弥补传统Top‑k注意力选择的不足；② 设计时空旋转位置编码（ST‑RoPE），将空间时间相对位置融入相似度计算，提升聚类质量。

**🔧 技术方法**

采用密度峰值聚类（DPC‑KNN）实现令牌选择与聚合；利用RoPE改写的相似度度量；结合时间视频分段与注意力池化实现预训练后剪枝。

**📊 数据集**

在四大视频理解基准上验证：Video‑MME、MVBench、LongVideoBench 与 MLVU，涵盖多场景、长短视频与多任务。

**📈 对比分析**

与 FastV、DART、VisionZip、VidCom2、FastVID、HoliTom 等最新无训练剪枝方法对比，Tango在10%令牌保留下保持98.9%性能，并实现1.88×加速；在更高保留比例下几乎无性能损失，且在跨模型、跨层级剪枝与大规模帧数下仍保持优势。

**⚠️ 局限性**

局限性在于：对超大规模视频的实时部署仍受限于聚类与位置编码的计算开销；在极端稀疏或高度动态场景中，注意力分布可能难以用多模态分布充分覆盖，导致选取误差；以及对不同视觉编码器的适配需进一步验证。

---

## 477. ANTIC: Adaptive Neural Temporal In-situ Compressor

**arXiv ID:** 2604.09543 | [PDF](https://arxiv.org/pdf/2604.09543v1)

**作者:** Sandeep S. Cranganore `[一作]` (JKU Linz), Johannes Brandstetter `[通讯]` (JKU Linz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一个端到端的 In‑situ 神经压缩框架 ANTIC，将物理感知的时间选择与连续微调的空间神经压缩结合，实现高维 PDE 仿真数据的极致压缩。

**💡 创新点**

创新点在于引入 Physics‑aware Temporal Selector (PATS) 用于基于物理量（如湍流耗散率、Weyl 标量）提取关键瞬态；以及采用低秩微调（LoRA）等持续学习策略，让空间压缩在保持高精度的同时形成可调的准确‑压缩率 Pareto 曲线。

**🔧 技术方法**

技术包括物理感知指标、LoRA / 全微调的持续学习、神经场（Neural Field）与连续 Fourier 特征映射、LayerNorm、SOAP 优化器等；同时利用动量阈值门控与上下文队列实现自适应采样。

**📊 数据集**

使用的实验数据集是 2D Kolmogorov 湍流（1000 步，16.8 GB）和 3D 二黑洞合并（5966 步，4.2 TiB）仿真。

**📈 对比分析**

与稀疏/密集采样加 ZFP、传统压缩等基线相比，ANTIC 在 2D 流体中实现 37 % 时间保留、47× 空间压缩、总压缩达 435×；在 3D 黑洞合并中实现 55 % 时间保留、3744× 空间压缩、总压缩 6807×，且相对 L2 误差保持在 10⁻⁴–10⁻⁵。

**⚠️ 局限性**

主要限制包括：神经压缩训练时间仍高于求解器；标准 MLP 对高阶空间导数的表达不足；PATS 需要针对具体 PDE 定制物理指标；以及对网格外查询的泛化性能尚未系统验证。

---

## 478. Case-Grounded Evidence Verification: A Framework for Constructing Evidence-Sensitive Supervision

**arXiv ID:** 2604.09537 | [PDF](https://arxiv.org/pdf/2604.09537v1)

**作者:** Soroosh Tayebi Arasteh `[一作]` (RWTH Aachen University), Daniel Truhn `[通讯]` (RWTH Aachen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了案例驱动的证据验证框架，直接判断外部证据是否支持给定病例与声明。

**💡 创新点**

创新点在于无需人工标注的监督构造，自动生成正例与语义控制的负例（错误状态与主题相关），使证据真正成为标签定义。

**🔧 技术方法**

使用现代BERT等Transformer作为验证器，并通过交互式干预（证据删除、置换、换源）评估证据依赖性。

**📊 数据集**

数据来源为MIMIC‑CXR和CheXpert‑Plus胸部影像报告，外部证据来自Radiopaedia。

**📈 对比分析**

与仅用病例或仅用证据的基线对比，完整模型在AUROC、AUPRC等指标上显著提升，且在证据置换或删除时性能显著下降，验证了对证据的真实依赖。

**⚠️ 局限性**

局限包括对证据来源的敏感性、不同模型骨干性能差异、仅限二分类声明以及未涵盖检索噪声和多态声明等实际应用挑战。

---

## 479. EgoTL: Egocentric Think-Aloud Chains for Long-Horizon Tasks

**arXiv ID:** 2604.09535 | [PDF](https://arxiv.org/pdf/2604.09535v1)

**作者:** Lulin Liu `[一作]` (University of Minnesota), Zhiwen Fan `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 EgoTL 数据集并通过说-先-执行协议收集同步音视频、链式思考和度量空间标签，评估并微调视觉语言模型与世界模型的长时空推理能力。

**💡 创新点**

提出实时说-先-执行的思考-说话协议和度量空间校准，提供精准的目标、动作与空间监督，填补现有数据集在意图和长时空一致性上的空白。

**🔧 技术方法**

使用 WhisperX 语音对齐、MapAnything 轨迹重建、LoRA 微调、长视频 VLM 与世界模型（如 Qwen、InternVL、COSMOS）在 EgoTL 上进行评估。

**📊 数据集**

EgoTL 本身（400 片段，100+ 任务），并与现有 egocentric 数据集（Ego4D、HD-EPIC 等）做对比。

**📈 对比分析**

通过 EgoTL‑Bench 的多层评测（记忆规划、场景交互、下一步预测、动作识别、方向识别、距离估计）对开源与闭源 VLM 以及微调模型进行多任务准确率/平均相对精度评估，结果显示微调后模型显著提升但仍远低于人类水平。

**⚠️ 局限性**

仍存在规划缺失、对象错报、时序漂移、距离估计偏差等缺陷，尤其在复杂环境和长期连续推理上表现不足。

---

## 480. VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning

**arXiv ID:** 2604.09529 | [PDF](https://arxiv.org/pdf/2604.09529v1)

**作者:** Wenyi Xiao `[一作]` (Zhejiang University), Leilei Gan `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为VL-Calibration的框架，针对大规模视觉语言模型的多模态推理中出现的幻觉和不确定性进行校准。

**💡 创新点**

创新点在于将置信度拆分为视觉置信度和推理置信度，并通过内在视觉确定性（图像扰动下KL散度+token熵）和基于视觉置信度的 token 级优势重加权，精确定位错误来源。

**🔧 技术方法**

采用强化学习（GRPO）训练模型，结合Brier误差、视觉确定性奖励与token级重加权的复合奖励函数，实现视觉与推理置信度的独立校准。

**📊 数据集**

在12k条从ViRL-39K采样的数据上进行训练，并在13个视觉推理与多学科推理基准（如DynaMath、MathVerse、CLEVR、A-OKVQA等）上进行评估。

**📈 对比分析**

与多种基线（Verbalize、RLVR、RLCR等）对比，VL-Calibration将4B模型的ECE从0.421降至0.098，同时提升准确率2.3%–3.0%；在8B模型上ECE降至0.071、准确率提升3.0%；在更大模型和不同架构上亦保持优越性能。

**⚠️ 局限性**

局限性主要体现在仅在4B–30B规模的模型上验证，对更大规模（70B+）的视觉语言模型的有效性及计算开销仍待进一步研究。

---

## 481. Envisioning the Future, One Step at a Time

**arXiv ID:** 2604.09527 | [PDF](https://arxiv.org/pdf/2604.09527v1)

**作者:** Stefan Andreas Baumann `[一作]` (LMU Munich), Björn Ommer `[通讯]` (LMU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从单张图像预测稀疏点轨迹的未来分布，提供高效多模态轨迹预测并支持动作规划

**💡 创新点**

提出稀疏轨迹的自回归扩散模型，结合流匹配头和快速注意力模块，避免像素级视觉税，显著提升采样速度；同时构建开放式运动预测基准 OWM

**🔧 技术方法**

自回归 Transformer + 轨迹流匹配（Flow Matching）头 + 并行注意力块 + RoPE 位置编码 + 随机轨迹 ID + DINOv3 图像编码器

**📊 数据集**

10M 开源视频训练集（伪轨迹），OMW 95 条野外视频，PhysicsIQ 及 Physion 物理子集，亦训练于桌球仿真数据

**📈 对比分析**

与 MAGI-1、Wan2.2、CogVideo-X、SkyReels V2、SVD 等密集视频生成模型相比，参数仅 665M，采样速率 2200 samples/min，Best‑5/Best‑5min 下 ADE 0.029/0.013，显著低于对手 (0.303/0.037 等)，在桌球规划任务中在相同算力下达到更高命中率

**⚠️ 局限性**

假设相机静止；依赖离线跟踪器生成的伪轨迹，可能带来误差；仅在二维平面内测试，未处理主体运动或视角变化

---

## 482. Trans-RAG: Query-Centric Vector Transformation for Secure Cross-Organizational Retrieval

**arXiv ID:** 2604.09541 | [PDF](https://arxiv.org/pdf/2604.09541v1)

**作者:** Yu Liu `[一作]` (Chinese Academy of Sciences), Yanbing Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Trans‑RAG，一种基于查询中心向量变换的跨组织检索增强生成（RAG）框架；通过多阶段向量2Trans（permutation、blinding、非线性、正交旋转）实现组织专属的向量空间隔离，并在不改动现有向量数据库和嵌入模型的前提下，支持安全高效的检索与生成。

**💡 创新点**

创新点：①将安全视角从文档级加密转为查询级变换，形成“向量空间语言”；②设计多阶段键驱动变换，使不同组织的向量空间实现几乎正交隔离；③兼顾安全性、检索准确性和计算效率，在高维空间中实现几乎无精度损失且远快于同类同态加密方案。

**🔧 技术方法**

核心技术：多阶段向量2Trans（键驱动的随机置换、输入相关加密噪声、可控非线性激活、正交旋转与偏置），密钥生成与伪随机函数、FAISS 索引（IndexFlatIP/IVFFlat），以及传统加密基线（AEAD、PHE）用于对比。

**📊 数据集**

使用 BEIR 三个基准集（NFCorpus、FiQA、SciFact）进行检索评估，并构造 1K、10K、100K、1000K 规模的自定义语料进行效率与可扩展性测试。

**📈 对比分析**

与无加密基线、AEAD 以及 Paillier 同态加密做对比。实验显示：平均 nDCG@10 下降约 3.5%（从 0.498 降至 0.481），但交叉空间角度从 58.33° 提升至 89.90°，隔离率超过 99.8%。在查询加速方面，Trans‑RAG 速度提升达 32,216×，而同态加密在 10K 规模下耗时 3.6×10⁷ ms；AEAD 在大规模下显著慢于 Trans‑RAG。

**⚠️ 局限性**

局限性：①对极高维（> 8K）向量的变换开销较大，导致数秒级延迟；②安全模型仅覆盖半诚实攻击，未考虑恶意查询或数据库篡改；③密钥生命周期管理、跨组织撤销与合并等运营成本未深入探讨；④大规模组织数（m）下的通信与聚合开销虽然线性，但在极大 m 的场景仍需进一步评估。

---

## 483. Seeing is Believing: Robust Vision-Guided Cross-Modal Prompt Learning under Label Noise

**arXiv ID:** 2604.09532 | [PDF](https://arxiv.org/pdf/2604.09532v1)

**作者:** Zibin Geng `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Min Liu `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出VisPrompt，一种在标签噪声环境下利用图像信息进行prompt学习的鲁棒框架。

**💡 创新点**

引入跨模态注意力将视觉语义注入prompt，并结合FiLM门控自适应调节视觉注入强度，从而在保持参数效率的同时提升对噪声标签的鲁棒性。

**🔧 技术方法**

跨模态注意力、FiLM调制、OT估计样本可靠性、CLIP预训练模型、轻量级可微模块。

**📊 数据集**

六个合成噪声数据集（EuroSAT、Flowers102、OxfordPets、DTD、UCF101、Caltech101）和一个真实噪声数据集Food101N。

**📈 对比分析**

与CoOp、GCE、JoAPR、NLPrompt等基线在多种噪声率下对比，VisPrompt在所有数据集和噪声设置中均获得最高准确率，尤其在高噪声率下优势明显。

**⚠️ 局限性**

仅在CLIP式图像分类任务上验证，噪声模式有限，对资源受限场景的计算开销有一定影响，未扩展至检测/分割等更复杂任务。

---

## 484. VisionFoundry: Teaching VLMs Visual Perception with Synthetic Images

**arXiv ID:** 2604.09531 | [PDF](https://arxiv.org/pdf/2604.09531v1)

**作者:** Guanyu Zhou `[一作]` (Princeton University), Zhuang Liu `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个完全自动化的任务感知合成数据生成管线，利用大型语言模型生成问题答案及详细的文本-图像提示，使用文本-图像模型合成图像，再通过多模态判定器进行一致性验证，最终生成可用于视觉语言模型训练的高质量 VQA 数据集；

**💡 创新点**

首次将任务关键词驱动的合成流程与自动化多模态验证结合，避免了人工标注与参考图像的需求，并实现了对低层视觉感知能力的精准、可控补强；

**🔧 技术方法**

核心技术包括大语言模型（GPT‑5.2）用于生成 Q&A 与 T2I 提示、现代文本-图像生成模型（Gemini‑2.5‑Flash‑Image）用于图像合成，以及专有多模态判定器（Gemini‑3‑Pro）用于图像与答案的自动一致性检查；

**📊 数据集**

使用构建的 VisionFoundry‑10K 数据集（10k 影像–问题–答案三元组，涵盖 10 种视觉感知子任务），并与公开自然图像语料（如 LLaVA‑Instruct‑80K）进行对照；

**📈 对比分析**

在 Qwen2.5‑VL‑3B‑Instruct、Llama‑3.2‑11B‑Vision‑Instruct 与 MiMo‑VL‑7B‑SFT 三大 VLM 上进行微调，实验表明在 MMVP、CV‑Bench、RealWorldQA 等视觉感知基准上平均提升约 7%–10%，同时在大多数通用与应用基准上保持不降，且性能随合成数据规模呈正向增长；

**⚠️ 局限性**

局限在于仅关注低层视觉感知任务，未覆盖更复杂的多模态推理；合成图像与判定器仍可能出现误判，导致部分低质量样本流入数据集，且对超大规模多模态推理任务的效果尚未验证。

---

## 485. Semantic Rate-Distortion for Bounded Multi-Agent Communication: Capacity-Derived Semantic Spaces and the Communication Cost of Alignment

**arXiv ID:** 2604.09521 | [PDF](https://arxiv.org/pdf/2604.09521v1)

**作者:** Anthony T. Nixon `[一作]` `[通讯]`, Anthony T. Nixon

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究不同计算容量的代理在同一环境中的通信问题，提出使用商POMDP作为容量导出的语义空间，证明了容量差异引起的结构相位转移，并将语义率失真问题与Wyner–Ziv编码在商字母上的等价性联系，进一步给出缩小失真下的渐进对偶式下界，并通过合成方案验证理论。

**💡 创新点**

创新点在于：①直接由有限状态控制器的容量通过商POMDP构造获得各代理的语义字母；②证明容量缺口导致的结构相位转移是不可规避的；③将语义通信的阈值精确映射为Wyner–Ziv的条件熵阈值；④提供缩小失真情形下的渐进信息率下界，补充了传统信息论对非对称能力通信的空白。

**🔧 技术方法**

使用的技术包括：有限状态控制器抽象与Myhill–Nerode商分割；信息论中的率失真、Wyner–Ziv、定向信息与Fano不等式；混合时间与混合率证明；Blahut–Arimoto算法计算Wyner–Ziv基准；k‑means++编码与贝尔曼迭代实现构造性方案；以及实验中的随机抽样与聚类。

**📊 数据集**

实验数据集主要为合成POMDP：Chain5、Chain100/150/200、BalancedRand8、RichGridWorld、RockSample(4,4)以及其他数个规模在100–200状态的环境，均采用离散观测与有限动作集合。

**📈 对比分析**

通过将语义编码方案与Wyner–Ziv基准、随机聚类和对beliefs的k‑means聚类进行对比，实验结果显示：存在明显的“knee”点，匹配理论给出的log‑cardinality阈值；在结构化策略下，所需比率可低至最坏情况的1/19；实验曲线与理论预测高度一致，验证了结构相位转移与Wyner–Ziv对应关系。

**⚠️ 局限性**

局限性包括：只考虑一方向可观测与公共历史细化的情况；结果基于离散有限状态控制器，未涵盖连续、噪声或未知环境；仅在一方向记忆无关的记忆体中给出指数收敛实现；对实际系统的商字母估计仍为开放问题。

---

## 486. Many Ways to Be Fake: Benchmarking Fake News Detection Under Strategy-Driven AI Generation

**arXiv ID:** 2604.09514 | [PDF](https://arxiv.org/pdf/2604.09514v1)

**作者:** Xinyu Wang `[一作]` (Pennsylvania State University), Sarah Rajtmajer `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个基于策略的五层人机协作假新闻生成框架，并使用该框架在 Snopes 事实核查数据基础上生成 6,798 篇混合真假的 synthetic 新闻，形成了新的 benchmark（MANYFAKE），随后对多种 LLM 检测模型（包括标准和 reasoning‑enabled 版本）在不同生成策略下的识别性能进行了统一评估。

**💡 创新点**

创新点在于①提出了能够捕捉人机协作假新闻生成的五层策略 taxonomy；②利用该 taxonomy 生成了规模大、真伪混合、主题多样的 synthetic 数据集，填补了现有 benchmark 对混合真假的缺失；③系统比较了传统与 reasoning‑enabled LLM 检测器，揭示了当前模型对微妙误导的脆弱性。

**🔧 技术方法**

技术方法包括多轮 Prompt 生成 pipeline、策略驱动的优化与迭代层、规则化的最终化处理、自动化质量评估（词数、句长、可读性、重复率）与人工审核；在评估阶段采用统一的二分类 Prompt、LLM 直接检测与嵌入 reasoning 的检测模型。

**📊 数据集**

使用的数据集为：① Snopes 事实核查的 4,000 条真/假 claim（含 8 个主题标签）作为 seed；② 基于上述 seed 生成的 6,798 条 synthetic 假新闻，形成 MANYFAKE benchmark。

**📈 对比分析**

通过在统一的二分类 Prompt 上对标准 LLM（如 GPT‑4o、Llama‑3.1、Qwen‑2.5、Gemma 等）与 reasoning‑enabled 版本（如 GPT‑5.1、Gemini‑3‑Flash 等）进行评测，发现先进模型在完全虚假内容上接近 100% 的准确率，但在微妙的“少量误导”或“事实扭曲”场景下准确率仅 30–60%，而加入 reasoning 后能提升约 10–20%。

**⚠️ 局限性**

局限性包括：①仅使用文本数据，未覆盖多模态假新闻；② synthetic 生成方式可能与真实 disinformation 的策略不完全一致，生态效度有限；③人工与自动审核仍可能漏检细微错误；④benchmark 主要基于 Snopes 的 claim，主题覆盖虽多但仍有潜在盲区。

---

## 487. Large Language Models Generate Harmful Content Using a Distinct, Unified Mechanism

**arXiv ID:** 2604.09544 | [PDF](https://arxiv.org/pdf/2604.09544v1)

**作者:** Hadas Orgad `[一作]` (Harvard University), Yonatan Belinkov `[通讯]` (Technion---IIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用目标权重剪枝（SNIP）探查大型语言模型内部有害内容生成的机制，发现有害生成依赖极小而集中、可压缩的权重子集。

**💡 创新点**

首次证明有害生成是一个可压缩且统一的机制，且与对抗性破解、微调诱发的突发不对齐密切相关。

**🔧 技术方法**

采用带符号SNIP剪枝、双重校准数据集（有害/安全），前缀注入等技术进行因果干预。

**📊 数据集**

使用AdvBench、Hex‑PHI、StrongREJECT、Alpaca等多种对抗与安全数据集进行评估。

**📈 对比分析**

与未剪枝、对抗性攻击及微调实验对比，剪枝后有害率显著降低（>90%）且实用性能损失≤10%，且剪枝仅需≈0.0005%权重稀疏。

**⚠️ 局限性**

限制在已训练的单模态语言模型上，剪枝不移除知识，易通过微调恢复；对多模态、其他攻击方式的适用性尚未验证。

---

## 488. A Physically-Informed Subgraph Isomorphism Approach to Molecular Docking Using Quantum Annealers

**arXiv ID:** 2604.09540 | [PDF](https://arxiv.org/pdf/2604.09540v1)

**作者:** Francesco Micucci `[一作]` (Politecnico di Milano), Gianluca Palermo `[通讯]` (Politecnico di Milano)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在分子对接中引入物理化学相互作用（库仑、电荷、范德瓦耳斯、氢键、疏水）到 QUBO 形式，并在量子退火器上求解

**💡 创新点**

首次将完整原子图与几何子图同化，利用物理化学信息构建无额外复杂度的 QUBO 目标，并通过全原子模型对比传统药效团方法

**🔧 技术方法**

权重子图同构到 QUBO、模拟退火求参数、D‑Wave Advantage 量子退火器、实验评估（RMSD、Adjusted RMSD）

**📊 数据集**

使用 PDBbind 2020 精炼集的子集，筛选轻量级配体（≤6–8 个重原子）

**📈 对比分析**

与纯几何基线比较，采用 Adjusted RMSD 评估；模拟退火下平均提升约20%，量子退火下提升>15%，但有效解比例低于1%，并伴随较高链长和物理量子数

**⚠️ 局限性**

受限于嵌入开销、有效解率低、仅适用于小配体，需改进编码与超参数搜索以提升可行性

---

## 489. On Worst-Case Optimal Polynomial Intersection

**arXiv ID:** 2604.09533 | [PDF](https://arxiv.org/pdf/2604.09533v1)

**作者:** Yihang Sun `[一作]` (Stanford University), Mary Wootters `[通讯]` (Stanford University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究 MDS 最大满足率问题，并通过改进差分采样框架突破了传统半圆律的限制，实现了更高的满足率；

**💡 创新点**

创新点在于将当地泄漏鲁棒性技术与量子启发式差异采样相结合，提出了通用桶划分控制方法，得到新的满足率阈值 2μ≥0.78 使得解接近 1；

**🔧 技术方法**

使用了 k‑wise discrepancy、Kravchuk 多项式、信息熵/二元熵函数、Cauchy‑Schwarz、Parseval 及离散傅里叶系数上界等理论工具；

**📊 数据集**

未使用具体数据集，全部为理论证明与数值优化；

**📈 对比分析**

与 DQI 与早期泄漏鲁棒性方法对比，性能在更高编码率下明显提升，在 2μ≥0.78 时满足率可达 1‑o(1)；

**⚠️ 局限性**

局限性包括只能在素数域下工作，无法推广到扩张域；参数优化复杂；对 ρ>0.67 或低率场景无改进；

---

## 490. Event-Driven Temporal Graph Networks for Asynchronous Multi-Agent Cyber Defense in NetForge_RL

**arXiv ID:** 2604.09523 | [PDF](https://arxiv.org/pdf/2604.09523v1)

**作者:** Igor Jankowski `[一作]` `[通讯]`, Igor Jankowski

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 NetForge_RL 一种高保真、物理约束的网络防御模拟器，并开发了 CT-GMARL 架构实现多智能体连续时间强化学习，以解决 Sim2Real 问题。

**💡 创新点**

创新点包括：①将网络防御建模为连续时间部分可观半马尔可夫决策过程 (POSMDP)，真实异步事件和 NLP 语义日志；②引入神经 ODE-RNN 与图注意力网络 (GAT) 的融合，处理非均匀时间序列与动态拓扑；③通过双模式引擎实现高吞吐量模拟训练与零样本真实容器验证；④设计了连续时间 GAE 与 MAPPO 的优化方案。

**🔧 技术方法**

主要技术：连续时间 POSMDP、神经 ODE（RK4）、图注意力网络（GAT）、多智能体 PPO（MAPPO）、连续时间 GAE、Sim2Real 双引擎、TF-IDF+LSA/Transformer SIEM 处理、随机噪声生成（Poisson 过程）和 Zero-Trust 访问控制。

**📊 数据集**

数据集：使用 100 节点的三子网网络模拟，日志通过生成的 Windows 事件 XML、TF-IDF + LSA 128 维嵌入；在零样本验证中利用 Vulhub 容器中的实际 CVE（EternalBlue、BlueKeep 等）作为攻击载荷。

**📈 对比分析**

对比方法：R-MAPPO、QMIX 以及 CT-GMARL 的若干消融实验。结果显示 CT‑GMA RL 在蓝方获得中位奖励 57,135，约为 R‑MAPPO（28,347）和 QMIX（26,649）的 2.0‑2.1 倍；恢复受损服务量 144，远高于基线；在零样本 Docker 测试中获得 98,026 的奖励，验证了 Sim2Real 桥接。

**⚠️ 局限性**

局限性：①神经 ODE 的计算开销大，导致吞吐量仅为 10 步/秒；②双模式实现仍需在真实容器中执行，吞吐量进一步下降；③红方攻击动作有限（32 种预定义 CVE），缺乏对未知零日攻击的泛化能力；④持续时间模型对高频事件的适配性有限，非自适应 RK4 可能模糊细粒度安全语义。

---

## 491. RIRF: Reasoning Image Restoration Framework

**arXiv ID:** 2604.09511 | [PDF](https://arxiv.org/pdf/2604.09511v1)

**作者:** Wending Yan `[一作]` (Southwest Jiaotong University), Qiankun Liu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Reason and Restore (R&R) 框架，先用 VLM 进行结构化 Chain-of-Thought 诊断，随后将诊断结果作为先验指导 VLM 基础恢复模型完成像素级图像恢复，并通过强化学习进一步对齐诊断与恢复质量。

**💡 创新点**

创新点包括：① 在统一的 UIR 中首次引入结构化 CoT 诊断；② 将诊断输出直接作为可利用的先验输入到恢复网络，而非仅做工具调度；③ 使用诊断严重度作为 RL 奖励，使恢复器在优化过程中兼顾诊断一致性和像素质量。

**🔧 技术方法**

使用 Qwen3‑VL 进行诊断（Fine‑tune 进行 CoT 生成），使用 Qwen‑Image‑Edit 作为恢复器；训练分为三阶段：诊断器的监督 fine‑tune、恢复器的监督 fine‑tune 以及基于 GRPO 的强化学习；并构建半真实退化模型生成合成训练数据。

**📊 数据集**

实验数据集包括：OTS、RESIDE 的合成图像（用于训练和测试），以及新收集的 700+ 张真实户外图像（用于验证泛化）。

**📈 对比分析**

与多种基线（3D、FoundIR、Img2Img‑Turbo、Stable Diffusion3、Qwen‑Image）在 PSNR/SSIM 上进行对比。R&R 在 OTS 上 19.564 dB / 0.6214 SSIM，RESIDE 上 17.0036 dB / 0.6188 SSIM，均明显领先于现有方法。

**⚠️ 局限性**

局限性包括：① 对大规模 VLM 训练资源和算力依赖强；② 在极端退化或未见过的混合场景下仍可能出现误诊或恢复不佳；③ 强化学习步骤复杂，调参难度大；④ 需要进一步验证在实时或低算力设备上的可行性。

---

## 492. Demonstrably Informed Consent in Privacy Policy Flows: Evidence from a Randomized Experiment

**arXiv ID:** 2604.09518 | [PDF](https://arxiv.org/pdf/2604.09518v1)

**作者:** Qian Ma `[一作]` (Pennsylvania State University), Brett Frischmann `[通讯]` (Villanova University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在儿童教育应用的隐私政策同意流程中引入教学摩擦（pedagogical friction），评估其对受访者对关键条款理解的提升及可证明的知情同意的实现效果。

**💡 创新点**

创新点在于将教学摩擦概念应用到隐私政策同意界面，提出“可证明的知情同意”标准，并系统比较了六种不同摩擦设计（高亮、补充说明、幻灯片、分段定时、复测等）对首次和复测测验成绩以及同意率的影响。

**🔧 技术方法**

使用了随机实验设计（6个实验条件），配合六题测验并设定80%阈值作为“证明同意”的标准，并记录阅读时长、测验时长、答题情况及退出调查，以量化摩擦对理解与决策的影响。

**📊 数据集**

数据集为293名拥有3-8岁儿童父母的实验样本，来自Positly平台，实验期间记录了每位受访者在不同条件下的阅读时长、测验成绩、是否复测以及最终同意决策。

**📈 对比分析**

对比方法：采用单因素ANOVA、卡方检验以及置信区间比较六个条件的首次测验准确率、阈值达成率、复测提升率和同意率。结果显示：幻灯片（G3）与分段高亮（G4）在首次达标率上最高；复测条件中分段+补充说明（G5）在第二次测验中表现最佳，整体门槛同意率约为87%。

**⚠️ 局限性**

局限性包括：样本量有限、仅一次实验会话、测验问题数量少导致分辨率不足、未直接测量认知负荷、未检验不同政策或更广泛领域的适用性。

---

## 493. Toward World Models for Epidemiology

**arXiv ID:** 2604.09519 | [PDF](https://arxiv.org/pdf/2604.09519v1)

**作者:** Zeeshan Memon `[一作]` (Emory University), Naren Ramakrishnan `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了计算流行病学的世界模型框架，强调将疫情视为受干预控制、部分可观测的动态系统，并通过案例验证该框架的必要性。

**💡 创新点**

创新点在于：①把干预视为序列化行动并与行为反馈耦合；②建立可生成观测的内部状态与观测模型；③通过LLM等模型在世界模型中进行策略生成与评估，实现对冲击的因果推断；④在实验中将世界模型与语言模型结合，演示对医院负荷的显著下降。

**🔧 技术方法**

使用的技术包括：基于变分自编码的受控状态空间模型、LLaMA‑2‑7B微调的CovidLLM、强化学习与GRPO的策略优化、迭代反馈式提示优化，以及多步滚动模拟。

**📊 数据集**

使用数据集为美国50州2021年1月–2022年末的COVID‑19监测数据（周度住院、确诊、疫苗接种等）以及OxCGRT的13维政策动作。

**📈 对比分析**

与历史真实政策对比，使用“政策一致率”和“住院率下降百分比”两指标。结果显示GPT‑4o‑mini在对齐率44.8%、住院率下降46.5%方面优于历史干预；Qwen‑7B对齐率30.3%、下降39.1%；通过迭代反馈或GRPO进一步提升到49.5%/48.7%和42.2%/46.1%。

**⚠️ 局限性**

局限性包括：潜在状态难以辨识、数据噪声与延迟导致的观测偏差、对多模态数据整合的挑战、计算开销大以及干预空间设计的复杂性。

---

## 494. Sustaining Exascale Performance: Lessons from HPL and HPL-MxP on Aurora

**arXiv ID:** 2604.09517 | [PDF](https://arxiv.org/pdf/2604.09517v1)

**作者:** Kazushige Goto `[一作]` (Intel Corporation), Aditya Nishtala `[通讯]` (Intel Corporation)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Intel的Aurora超算上，通过多轮大规模部署，跑出FP64 HPL 1.01 EF/s（9,234节点）和混合精度HPL-MxP 11.64 EF/s（9,500节点）的接近全系统级exascale性能。

**💡 创新点**

创新点在于：① 将CPU‑附加网络、多GPU与大规模Slingshot‑11互连结合，② 采用BF16存储与FP64迭代细化的混合精度策略并利用Intel AMX加速，③ 引入阶段性NIC分配和混合P2P/collective的容错机制，④ 在多层系统上实现确定性资源映射与显式CPU‑GPU重叠，整体实现了在10,000+节点上持续的高效性。

**🔧 技术方法**

使用技术包括：oneAPI/oneMKL GPU/CPU算子、MPICH＋libfabric(CXI) MPI实现、OpenCL多队列CPU‑GPU重叠、GPU Direct RDMA、AMX指令集、分阶段NIC映射、P2P/collective混合通信、BF16/FP64混合精度算子。

**📊 数据集**

数据集为稠密线性系统矩阵：FP64版尺寸28,773,888 × 28,773,888，BF16版尺寸57,693,696 × 57,693,696，满足GPU内存与DDR/HBM容量限制。

**📈 对比分析**

通过对比不同部署周期（SC23、ISC24、SC24）下的节点数与EF/s，验证了1.01 EF/s（FP64）和11.64 EF/s（HPL‑MxP）的性能提升；相较FP64提升约11.5倍，单节点基准维持78.8%扩展效率。

**⚠️ 局限性**

限制在于高度依赖Aurora特有的CPU‑附加NIC、多GPU与AMX架构，网络抖动与系统可靠性仍是挑战，方案在不同体系结构（尤其非CPU‑附加网络）上的可迁移性尚需验证。

---

## 495. When LLMs Lag Behind: Knowledge Conflicts from Evolving APIs in Code Generation

**arXiv ID:** 2604.09515 | [PDF](https://arxiv.org/pdf/2604.09515v1)

**作者:** Ahmed Nusayer Ashik `[一作]` (University of Manitoba), Yuan Tian `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究LLM在API演化环境下的代码生成能力，构建了270条真实Python库API更新基准，并评估了11个LLM模型。

**💡 创新点**

首次定量分析模型参数知识与外部API更新冲突的影响，并提出Self-Reflection等推理提示显著提升可执行率。

**🔧 技术方法**

采用检索增强生成、Chain-of-Thought、Self-Reflection、LLM-as-a-Judge自动评估以及Python虚拟环境执行测试等技术。

**📊 数据集**

基于八个主流Python库（NumPy、Pandas、SciPy、scikit-learn、TensorFlow、PyTorch、Keras、JAX）收集的270条后训练期API更新记录。

**📈 对比分析**

通过API Adoption Rate和Executable Rate两指标，对比UD+Doc、UD+Doc+CoT、UD+Doc+CoT+SR等提示配置，发现提供文档将可执行率从42%提升至66%，Self-Reflection进一步提升约11%。

**⚠️ 局限性**

实验仅涵盖Python数据科学库，评估受限于模型训练截止点，LLM-judge自动评估可能带来误差，且对更大规模或多语言库的泛化仍待验证。

---

## 496. Integrated electro-optic attention nonlinearities for transformers

**arXiv ID:** 2604.09512 | [PDF](https://arxiv.org/pdf/2604.09512v1)

**作者:** Luis Mickeler `[一作]` (ETH Zurich), Rachel Grange `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用薄膜锂铌的Mach‑Zehnder调制器（MZM）实现Transformer的Softmax和Sigmoid注意力非线性（Optmax、Optmoid），从而显著降低推理延迟。

**💡 创新点**

创新点在于将MZM的正弦非线性响应直接映射为Softmax的指数与归一化、Sigmoid的逐元素映射，避免了传统数字指数计算和分母求和的瓶颈，实现硬件级的低延迟、低功耗非线性。

**🔧 技术方法**

采用电光调制、模拟电路（DAC/ADC）、光学耦合与探测、以及低位量化模拟与数字混合处理技术，并结合可量化训练与噪声鲁棒性分析。

**📊 数据集**

在计算机视觉上使用MNIST、CIFAR‑10、SVHN等ViT任务；在自然语言处理上使用FineWeb‑Edu数据集训练GPT‑2模型。

**📈 对比分析**

与传统数字Softmax/Sigmoid相比，Optmax/Optmoid在精度/损失上保持1–3%以内的差距，且在4‑bit量化与高噪声条件下仍能维持可接受的性能；实现的系统延迟可比数字实现低1–2个数量级，能耗亦显著下降。

**⚠️ 局限性**

受限于MZM的周期性与峰值输出范围，导致动态范围受限；模拟噪声与量化误差对低精度和长序列任务影响显著；需进一步降低加性噪声并探索噪声感知训练以提升鲁棒性。

---

## 497. Packing Compact Subgraphs with Applications to Districting

**arXiv ID:** 2604.09522 | [PDF](https://arxiv.org/pdf/2604.09522v1)

**作者:** Ho-Lin Chen `[一作]` (National Taiwan University), Fang-Yi Yu `[通讯]` (George Mason University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在平面图、无顶点图族以及有界扩张图中对平衡或阈值约束下的紧凑连通子图打包问题的多种近似算法，包含常数因子近似和松弛约束下的PTAS；

**💡 创新点**

创新点在于将线性规划松弛与随机化舍入的分析方法改进至常数因子，并通过概率计费与Baker层划技术结合实现了对更广泛图族的优化；

**🔧 技术方法**

主要技术包括改进的LP松弛与分离子、随机化计费证明常数相关比、连通子图求和的FPTAS、Baker层划与树宽动态规划以及向量化剪枝的平衡/阈值判定；

**📊 数据集**

本工作为理论算法研究，未使用真实数据集，而是在图族上做严格的理论证明和实验性的算法复杂度分析；

**📈 对比分析**

与先前的O(log n)近似相比，本文在平面图和无顶点图族中实现了O(1)近似；在平衡或阈值约束的半径‑k 区域中同样得到常数近似，并在松弛约束下提供了可接受误差ε的PTAS；

**⚠️ 局限性**

主要限制包括：对弱半径约束的常数相关比尚未得到证明，动态规划的状态空间在保持精确平衡约束时易爆炸，且目前未能在有界扩张图族中给出高效分离子，导致PTAS仅适用于松弛约束或树宽图族。

---

