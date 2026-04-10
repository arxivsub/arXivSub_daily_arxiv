# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-10 | 今日论文总数: 611

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC

**arXiv ID:** 2604.07609 | [PDF](https://arxiv.org/pdf/2604.07609v1)

**作者:** Mohammad Siavashi `[一作]` (KTH Royal Institute of Technology), Marco Chiesa `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Blink 系统，通过将 LLM 推理堆栈委托给 GPU 和 SmartNIC，实现无 CPU 推理。

**💡 创新点**

创新点在于将主机 CPU 从持续推理路径中剔除，采用 GPU 持久调度器、设备端图形启动以及 DPU 端零拷贝 RDMA，形成端到端 CPU‑free 推理架构。

**🔧 技术方法**

使用技术包括 NVIDIA H100 GPU、Intel Xeon、BlueField‑3 DPU、RDMA、CUDA Graph、持久 GPU 内核、Ring Buffer、TensorRT 引擎、ARM tokenizer 等。

**📊 数据集**

使用的数据集为 ShareGPT v3 对话轨迹（自然对话），并在多种模型（Llama‑3 8B、Phi‑4 15B、Qwen‑3 32B、Qwen‑3 30B‑A3B）上评测。

**📈 对比分析**

与 TensorRT‑LLM、vLLM、SGLang 三系统对比，Blink 在隔离场景下 P99 TTFT 提升至 8.47×、P99 TPOT 提升至 3.40×、吞吐量提升至 2.1×，能耗降低 48.6%；在 CPU 干扰环境下保持稳定性能，而基线系统降幅可达两位数。

**⚠️ 局限性**

局限性包括目前仅单 GPU 实现，尚未支持多 GPU 分布式推理；对超出 GPU 显存的模型仍需 CPU/DRAM 迁移；在 DPU 资源受限或网络延迟高时可能受限。

---

## 2. Improving Search Suggestions for Alphanumeric Queries

**arXiv ID:** 2604.07364 | [PDF](https://arxiv.org/pdf/2604.07364v1)

**作者:** Samarth Agrawal `[一作]` (eBay Inc.), Zhe Wu `[通讯]` (eBay Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

提出一种训练无关的字符级检索框架，将每个字母数字标识符编码为固定长度二进制向量，利用Hamming距离进行近似最近邻检索，并可选用编辑距离进行候选排序；

**💡 创新点**

创新点在于将字母数字标识符视为非语言稀疏序列，使用二进制编码与Hamming距离实现高效低延迟检索，完全无需监督学习，且兼容大规模索引；

**🔧 技术方法**

技术手段包括字符到6位二进制编码、120位固定向量构造、Hamming空间近似最近邻索引（如FAISS/Annoy）以及Levenshtein编辑距离过滤；

**📊 数据集**

数据集来源于eBay的商品库存记录与历史销量，筛选出7位及以上的唯一标识符，并与用户点击/购买日志关联得到最常见的3条查询；

**📈 对比分析**

通过线上A/B测试评估：对比现有相关搜索，改进系统提升了18.8%的覆盖率、3.35%的点击率以及44.4%的转化率；

**⚠️ 局限性**

主要局限在于卖家提供的标识符质量不一，导致索引和检索效果受限，需进一步改进标识符校验和鼓励准确输入；

---

## 3. Enabling Intrinsic Reasoning over Dense Geospatial Embeddings with DFR-Gemma

**arXiv ID:** 2604.07490 | [PDF](https://arxiv.org/pdf/2604.07490v1)

**作者:** Xuechen Zhang `[一作]` (Google), Gautam Prasad `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Direct Feature Reasoning-Gemma（DFR-Gemma）框架，让大型语言模型（LLM）直接在密集的地理空间嵌入上进行推理，跳过传统文本化或检索增广（RAG）流程。

**💡 创新点**

核心创新在于将地理嵌入投影到LLM潜在空间生成“软”连续标记（soft tokens），让LLM能够在不改动主干网络的情况下对空间特征进行解码、比较与推理，显著提升信息密度、鲁棒性与推理效率。

**🔧 技术方法**

技术包括：1）使用 PDFM（基于 GNN 的人口动态基础模型）生成 330 维地理嵌入；2）设计多层感知机投影器 ϕ，将嵌入映射为 N 个软标记；3）在 LLM（Gemma/Qwen）中插入软标记并重新索引位置；4）通过监督微调（交叉熵）训练投影器，实现跨任务的统一推理。

**📊 数据集**

构建了一个多任务地理推理基准，涵盖 7,000 条样本，包含单嵌入查询、特征描述和多嵌入比较等三类问题，所有样本均配有 PDFM 嵌入与对应的验证答案。

**📈 对比分析**

实验在 Gemma-3-4B‑IT 上进行，DFR-Gemma 在单嵌入、特征描述和多嵌入任务上均超过了基线：相较于零上下文（0.67 → 0.79）、原始文本输入（0.63 → 0.79）、文本描述（0.70 → 0.79）以及非LLM模型（MLP/LightGBM），在复杂多嵌入查询中提升 6% 以上；同时在 Qwen‑2‑4B 上也保持一致性。

**⚠️ 局限性**

主要限制是高度依赖高质量的基础模型（如 PDFM），若嵌入表达不足则推理效果会大打折扣；此外，跨任务的投影器需要在目标域上少量微调或上下文适配以获得最佳性能。

---

## 4. DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time Series Forecasting

**arXiv ID:** 2604.07393 | [PDF](https://arxiv.org/pdf/2604.07393v1)

**作者:** Yeran Zhang `[一作]` (City University of Hong Kong), Tianyu Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 40161 | [OpenAlex ID](https://openalex.org/A5100418162)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种双流物理残差网络（DSPR），实现工业时间序列预测时的精度与物理可信度双提升。

**💡 创新点**

创新点在于将物理先验作为网络结构诱导（自适应窗口与动态图），而非仅靠软损失约束；通过分离稳定趋势与变时残差，动态捕捉运输延迟与耦合变化。

**🔧 技术方法**

采用时间混合器（TimeMixer）作为趋势流，图卷积与动态图卷积结合自适应窗口实现残差流；融合物理先验掩模、门控融合与线性投影等模块。

**📊 数据集**

四个工业基准数据集：SCR（化学反应延迟）、Rotary Kiln（热惯性）、Tennessee Eastman Process（化工耦合）以及 SDWPF（风电功率）。

**📈 对比分析**

与线性MPC、PatchTST、iTransformer、TimeMix‑er、MSGNet、TimeFilter、PG‑NN 等多种基线对比，DSPR 在 MAE/RMSE 上领先，MCA 最高 99.8%，TVR 最高 97.2%，TDA 最高 83.5%，显著提升精度与物理一致性。

**⚠️ 局限性**

局限包括对物理先验完整性的依赖；若存在未建模的耦合或设备退化，性能可能下降；在未见过的故障或极端操作条件下的泛化能力尚未充分验证。

---

## 5. FireSenseNet: A Dual-Branch CNN with Cross-Attentive Feature Interaction for Next-Day Wildfire Spread Prediction

**arXiv ID:** 2604.07675 | [PDF](https://arxiv.org/pdf/2604.07675v1)

**作者:** Jinzhen Han `[一作]` (Sungkyunkwan University), Jae-Joon Lee `[通讯]` (Jeonju University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了面向下一天野火蔓延预测的双分支CNN结构FireSenseNet，并通过系统比较评估了多种CNN、Transformer及混合架构的性能；

**💡 创新点**

创新点包括：①基于燃料/地形与气象两类物理意义的双分支编码；②跨模态交互模块CAFIM，实现自适应空间加权融合；③使用Monte Carlo Dropout实现像素级不确定性量化；④对评估协议进行剖析并揭示F1得分的严重夸大；

**🔧 技术方法**

技术手段涵盖深度卷积网络、跨模态注意力模块、U-Net解码器、组合损失（BCE、Dice、Focal）、数据增强（水平/竖直翻转、Cutout）、Monte Carlo Dropout、Transformer（SegFormer、Swin等）及其混合变体；

**📊 数据集**

使用Google Next‑Day Wildfire Spread公开数据集，包含约1.5万张64×64像素样本、12通道（4个燃料/地形、8个气象变量）以及二值火灾扩展目标；

**📈 对比分析**

在相同训练数据、损失函数和严格评估（仅预测新增燃烧区，排除未知像素）下，对七种架构进行对比。FireSenseNet获得最高F1≈0.418，远超其他模型；Transformer结构始终表现不佳；CAFIM模块在消融实验中提升约7.1%；评估协议改动可将F1夸大44–50%；

**⚠️ 局限性**

局限性：仅在单一、分辨率为64×64的日度数据集上验证；数据量有限，无法检验多时段或更大尺度下的泛化；对PrevFireMask的高度依赖意味着模型对火灾演化的长期预测能力有限；未来需在更大规模、多时序数据上验证并进一步改进气象输入的时空分辨率。

---

## 6. Weight Group-wise Post-Training Quantization for Medical Foundation Model

**arXiv ID:** 2604.07674 | [PDF](https://arxiv.org/pdf/2604.07674v1)

**作者:** Yineng Chen `[一作]` (University at Albany), Xin Wang `[通讯]` (University at Albany)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出后训练量化算法Permutation-COMQ，减少医学基础模型在终端设备上的存储与计算成本。

**💡 创新点**

创新点：①通过权重按幅值置换排序，降低尺度因离群值导致的误差；②利用坐标下降求解一元二次最小化，完全免除反向传播和Hessian求逆；③算法无超参，仅依赖点积与舍入操作。

**🔧 技术方法**

采用坐标下降量化（COMQ）框架、按通道量化、权重感知置换排序、无超参后训练量化。

**📊 数据集**

在模拟数据验证置换效果，并在真实医学数据AbdomenCT1K（CT扫描）上进行分割任务评估。

**📈 对比分析**

与RTN、原始COMQ及层级量化基线比较；在2/4/8-bit量化下，Permutation-COMQ在DSC/NSD上分别达到93.6%/93.2%（8-bit）、93.4%/93.1%（4-bit）和86.9%/78.9%（2-bit），明显优于其他方法。

**⚠️ 局限性**

局限性：仅针对权重量化，激活量化与网络结构未做改进；在极低比特（2-bit）下仍存在精度下降；实验集中于单一CT数据集，需在更多任务与设备上进一步验证。

---

## 7. The Lifecycle of the Spectral Edge: From Gradient Learning to Weight-Decay Compression

**arXiv ID:** 2604.07380 | [PDF](https://arxiv.org/pdf/2604.07380v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究神经网络训练过程中谱边缘的动力学及其在grokking（突然泛化）中的作用。

**💡 创新点**

发现谱边缘经历梯度主导→权重衰减对齐→压缩的两相生命周期，并证明梯度与权重衰减的对齐是grokking的微观标记。

**🔧 技术方法**

采用AdamW优化器的梯度-权重衰减分解、Gram矩阵奇异值分解、扰动曲线、Hessian曲率、非线性MLP探测器以及权重衰减干预实验。

**📊 数据集**

在两种序列任务上实验：Dyck-1平衡括号计数（约15万参数）和SCAN指令翻译（约150万参数）。

**📈 对比分析**

与无权重衰减对照、随机方向消融、以及不同种子重复验证相比较，谱边缘在grok阶段对性能影响4,000倍以上，而随机方向几乎无影响；权重衰减干预后准确率保持不变但线性可读性恢复。

**⚠️ 局限性**

局限性包括仅在小规模模型和任务上验证；功能坐标需要预先定义；梯度-权重衰减对齐与grokking的因果关系仍是相关性，未完全证明。

---

## 8. Grasp as You Dream: Imitating Functional Grasping from Generated Human Demonstrations

**arXiv ID:** 2604.07517 | [PDF](https://arxiv.org/pdf/2604.07517v1)

**作者:** Chao Tang `[一作]` (KTH Royal Institute of Technology), Danica Kragic `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过视觉生成模型合成的人类演示实现零样本功能抓取，并将其转化为可执行的机器人抓取计划；

**💡 创新点**

不需要大量机器人抓取数据，利用预训练的视觉生成模型与语言模型自动生成任务相关的人类抓取演示，结合功能化的转化与碰撞优化；

**🔧 技术方法**

视觉生成模型（VGM）、视觉语言模型（VLM）、MANO手部姿态估计、VDA深度估计、SLSQP优化、手-物体接触细化；

**📊 数据集**

TaskGrasp、DexGraspNet、YCB物体集、公开仿真和真实机器人实验数据；

**📈 对比分析**

在TaskGrasp上与LAN‑Grasp、GraspGPT、FoundationGrasp、GraspMolmo等对比，成功率从68%提升至78.6%，功能识别率和意图评分均显著提高；在DexGraspNet上与DexGYS、DexDiffuser对比，GraspDreamer在Shadow/Allegro手上成功率提升至80%+，并在真实机器人实验中达到70%+的抓取成功率；

**⚠️ 局限性**

对长时序的后抓取操作易产生幻觉，抓取对姿态误差敏感，且在人手与机器人手形差异较大时需额外的接触细化；

---

## 9. Multimodal Large Language Models for Multi-Subject In-Context Image Generation

**arXiv ID:** 2604.07422 | [PDF](https://arxiv.org/pdf/2604.07422v1)

**作者:** Yucheng Zhou `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出MUSIC，一种针对多主体场景的多模态大语言模型，用于在上下文中生成包含多张参考主体的图像。

**💡 创新点**

创新点包括自动化的可扩展数据生成管道、视觉链式推理（CoT）机制、语义驱动的空间布局规划，以及可在测试时进行多方案选择的缩放策略。

**🔧 技术方法**

采用了多种开源基础模型（Qwen-3、Qwen-2.5 VL、FLUX‑1.0‑DEV、UNO‑FLUX‑1.0‑DEV、GroundingDINO、SAM2、CLIP）以及LoRA微调技术。

**📊 数据集**

训练数据通过自动生成的多主体合成集（约1万样本）构建，并在MSIC和DreamBench两个基准上进行评测。

**📈 对比分析**

与Subject Diffusion、MIP‑Adapter、MS‑Diffusion、OmniGen、UNO等前沿方法比较，MUSIC在DINO、CLIP‑I和CLIP‑T指标上均取得显著提升，且在人工评测中更受偏好。

**⚠️ 局限性**

局限性包括随着主体数量增加仍存在性能下降，且测试时多方案选择会线性增加推理时间。

---

## 10. Mathematical analysis of one-layer neural network with fixed biases, a new activation function and other observations

**arXiv ID:** 2604.07715 | [PDF](https://arxiv.org/pdf/2604.07715v1)

**作者:** Fabricio Macià `[一作]` (Universidad Politecnica de Madrid), Shu Nakamura `[通讯]` (Gakushuin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一个隐藏层的ReLU神经网络（连续与离散两种形式），并在L²平方损失下证明了梯度下降的收敛性与谱偏差（低频优先）特性。

**💡 创新点**

创新点在于将ReLU视作一维拉普拉斯算子的基本解，利用此数学结构得到网络参数的唯一可表示性与收敛证明，并提出新的激活函数FReX（全波整流指数函数），同样具备基本解性质且可证明收敛。

**🔧 技术方法**

运用了分布式微分、算子理论（紧算子、Hilbert–Schmidt理论）、傅里叶分析以及对算子谱的显式计算等数学技术。

**📊 数据集**

主要为理论分析，没有使用公开数据集；仅在MNIST数据集上做了简易实验验证FReX的实用性。

**📈 对比分析**

通过与ReLU网络对比，FReX在MNIST上的分类性能与ReLU相近且优于Sigmoid；在理论层面，通过显式谱计算展示梯度下降对低频模式收敛更快，体现谱偏差。

**⚠️ 局限性**

局限性：仅考虑一维输入输出、单隐藏层、固定偏置，深层与多维情形尚未探讨；离散模型仅在有限宽度下适用；实验部分不足以全面验证实际效果。

---

## 11. Validated Synthetic Patient Generation for Small Longitudinal Cohorts: Coagulation Dynamics Across Pregnancy

**arXiv ID:** 2604.07557 | [PDF](https://arxiv.org/pdf/2604.07557v1)

**作者:** Jeffrey D. Varner `[一作]` (Cornell University), Ira Bernstein `[通讯]` (University of Vermont Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并验证了一种基于现代 Hopfield 网络的乘法权重随机注意力（Multiplicity‑Weighted Stochastic Attention, SA）框架，用于从仅有 23 名孕妇的极小纵向凝血数据集中生成既保持统计特征又具生物学可解释性的合成患者记录。

**💡 创新点**

创新点在于：① 将真实患者资料直接作为记忆模式，而非拟合参数化分布；② 利用能量函数的乘法权重实现无训练条件生成，可在推理时放大稀有子群；③ 在 n < p 的极小样本场景中通过 PCA 降维+Langevin 采样保留数据几何结构，避免传统 MVN 的秩缺陷和 GAN/VAE 的模式坍塌。

**🔧 技术方法**

核心技术包括：现代 Hopfield 能量网络、乘法权重化的能量函数、基于 Langevin 动力学的采样、PCA 线性降维、方向‑幅度分解、独立 ODE 机制模型验证、统计检验（MRE、Kolmogorov–Smirnov、Mann–Whitney）。

**📊 数据集**

使用的数据集是 23 名孕妇在三次访视（预孕、孕早期、孕晚期）中测得的 72 项凝血、纤溶、激酶生成和弹性参数（总 216 维），包含 PCOS、PE 等稀有子群。

**📈 对比分析**

与正则化多元正态分布（MVN）对比，SA 在四层验证（边际一致性、跨访视协方差、条件子群放大、机制一致性）均表现更优：边际 MRE ≈ 1.2%，跨访视相关矩阵保持块结构，条件生成能保持稀有子群特征，Ode 预测的云重叠 ≥ 85%，且用 SA 生成的合成数据校准的机制模型对真实留存样本的预测误差与仅用真实数据校准模型相当。

**⚠️ 局限性**

局限性包括：仅在单一极小队列上评估，未验证跨疾病或更大规模数据；PCA 降维假设线性结构，可能忽略非线性关联；机制验证仅覆盖凝血 ODE，未涉及纤溶或弹性测量；生成过程对尾部分布的压缩导致极端值多样性下降；乘法权重放大稀有子群时有效模式数下降，可能影响极小子群的生成质量。

---

## 12. The Day My Chatbot Changed: Characterizing the Mental Health Impacts of Social AI App Updates via Negative User Reviews

**arXiv ID:** 2604.07548 | [PDF](https://arxiv.org/pdf/2604.07548v1)

**作者:** Sirajam Munira `[一作]` (Rensselaer Polytechnic Institute), Lydia Manikonda `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Google Play上收集了Character AI聊天机器人的评论，按发布时间与应用版本对齐后，分析了不同版本下的用户评分及负面评论内容。

**💡 创新点**

创新点在于将用户评论与具体应用版本关联，并同时运用评分变化、LIWC情感词典与BERTopic主题建模三种技术，系统评估版本更新对用户满意度的影响。

**🔧 技术方法**

技术包括版本序列化与冲击判定、文本情感分析（spaCy+TextBlob）、LIWC情感词典分析、n-gram频繁短语提取以及基于Transformer的BERTopic主题建模。

**📊 数据集**

数据集为210,840条公开Google Play评论（包含评分、评论文本、时间戳、对应版本号），其中53,927条为1–2星的负面评论。

**📈 对比分析**

方法上先计算相邻版本平均评分差值并设阈值0.3判定冲击版本，再比较冲击与非冲击版本的LIWC情感分数，随后用BERTopic提取主题并手工聚合成高层主题。结果显示某些版本导致明显的负评分波动，情感词汇差异有限，但主题聚焦于功能、可用性、内容过滤与心理风险等。

**⚠️ 局限性**

局限性包括：仅基于公开评论，可能不代表整体用户群；未跟踪单一用户的版本变化；情感分析与主题模型受词汇表与模型偏差影响；缺乏直接将版本变更映射到具体功能或更新日志的分析。

---

## 13. Dual-Loop Control in DCVerse: Advancing Reliable Deployment of AI in Data Centers via Digital Twins

**arXiv ID:** 2604.07559 | [PDF](https://arxiv.org/pdf/2604.07559v1)

**作者:** Qingang Zhang `[一作]` (Nanyang Technological University), Yonggang Wen `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于数字孪生的双环控制框架（DLCF），实现了数据中心冷却系统的可靠深度强化学习控制。

**💡 创新点**

创新点在于融合数字孪生、策略池与实时预评估机制，显著提升了样本效率、安全性和可解释性，并通过双环交互实现了可靠部署。

**🔧 技术方法**

使用了物理信息化机器学习（PIML）、模型基础与模型无关的深度强化学习算法、Kalman滤波/粒子滤波数据同化、预评估与专家验证。

**📊 数据集**

使用了真实数据中心的现场监测数据、EnergyPlus仿真数据以及历史运行日志构建的数字孪生。

**📈 对比分析**

与传统基于规则的控制策略对比，实验显示在冷却子系统中实现了约4.09%的能耗下降，同时满足SLA约束，且算法可解释性得到提升。

**⚠️ 局限性**

局限性包括缺乏统一理论框架、对数字孪生误差的量化不足、仅验证了冷却子系统且未扩展到多子系统的整体优化。

---

## 14. Learning Markov Processes as Sum-of-Square Forms for Analytical Belief Propagation

**arXiv ID:** 2604.07525 | [PDF](https://arxiv.org/pdf/2604.07525v1)

**作者:** Peter Amorese `[一作]` (University of Colorado Boulder), Morteza Lahijanian `[通讯]` (University of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于稀疏 Sum-of-Squares（SoS）与 Rational Factor（RF）形式的 Markov 过程建模框架，能够在学习过程中同时保留解析性信念传播与可扩展性。

**💡 创新点**

创新点在于：①通过 RF 形式绕过 SoS 对条件密度归一化的严格限制，实现任意基函数的可解析建模；②允许在学习时同时优化基函数参数与系数，实现稀疏表示；③将半正定约束与解析积分结合，保持模型的可训练性。

**🔧 技术方法**

使用的技术包括 Sum-of-Squares 表示、Rational Factor 形式、半正定规划（SDP）约束、对基函数的可解析积分、随机梯度训练、Beta‑PDF 基函数与多维贝塔分布等。

**📊 数据集**

实验数据集主要为仿真产生的系统轨迹：2D Van der Pol、4D Cartpole、6D Planar Quadcopter、6D Dubin’s Car w/ Trailer 以及 12D Quadcopter 等。

**📈 对比分析**

与 Bernstein Normalizing Flow（BNF）、扩展卡尔曼滤波（EKF）、GMM、WSASOS、深度 Normalizing Flow 等方法进行对比。RF‑SoS 在低维（2D）时与 BNF 性能相当且参数量远小；在 6D、12D 高维场景下，传统方法多出现内存溢出或精度下降，而 RF‑SoS 能保持解析传播并在内存与计算上具有显著优势；在部分尖锐分布场景下，深度流仍表现更好。

**⚠️ 局限性**

限制：1) 对于极尖锐或多峰的转移分布，Beta‑PDF 基函数可能不够灵活；2) 深度流在高维时仍可获得更好的表达能力；3) 需要进一步研究基函数选择与更大规模系统的可扩展性。

---

## 15. Zero-Sum Fictitious Play Cannot Converge to a Point

**arXiv ID:** 2604.07544 | [PDF](https://arxiv.org/pdf/2604.07544v1)

**作者:** Jaehong Moon `[一作]` `[通讯]`, Jaehong Moon

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文通过严格的几何与分析方法，证明在某些零和博弈中，即便是最优的历史依赖学习策略——拟像博弈（fictitious play），在任何 tie‑breaking 规则下也无法收敛到单一的纳什均衡点，而只能收敛到整个均衡集合。

**💡 创新点**

创新点在于提出了三条几何条件（A1：均衡集正测度；A2：均衡点全混合；A3：边界不稳定性）并证明其足以保证点收敛失效，同时提出更弱的猜想（仅需 A1 与 A2），为探讨拟像博弈在多均衡情形下的长期行为提供了新的理论框架。

**🔧 技术方法**

使用的技术主要是：线性规划与凸几何理论、几何不动点论证、最佳响应多面体的划分、对惯性现象的正式表述，以及通过构造具体零和游戏与 tie‑breaking 规则来展示非收敛性。

**📊 数据集**

无；论文完全是理论推导，没有使用任何实验数据集。

**📈 对比分析**

无实验比较；论文通过数学证明给出收敛性与非收敛性的上界与下界，展示了在给定条件下拟像博弈无法达到点收敛。

**⚠️ 局限性**

局限性在于结果仅适用于满足 A1、A2、A3 的特定零和博弈，且 A3 被认为是技术性假设，实际可能在更弱条件下亦成立；此外，对初始化的依赖与实际博弈中的非平稳动态尚未完全解释。

---

## 16. Beyond Human-Readable: Rethinking Software Engineering Conventions for the Agentic Development Era

**arXiv ID:** 2604.07502 | [PDF](https://arxiv.org/pdf/2604.07502v1)

**作者:** Dmytro Ustynov `[一作]` `[通讯]` (Military Institute of Telecommunications and Information Technologies), Dmytro Ustynov (Military Institute of Telecommunications and Information Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对软件工程实践进行系统分析，提出语义密度优化原则并验证其有效性。

**💡 创新点**

提出语义密度优化原则、程序骨架概念以及对经典反模式的再评价。

**🔧 技术方法**

采用控制实验、日志压缩与工具辅助解码、token计数与性能评估等技术。

**📊 数据集**

使用200条模拟电子商务应用的日志事件数据集。

**📈 对比分析**

将四种日志格式（人类可读、结构化、压缩、压缩+工具）进行对比，实验显示压缩虽节省输入token，却导致会话总成本上升67%。

**⚠️ 局限性**

实验仅针对数据检索，未验证代码架构优化；仅使用单一模型与单一数据集；压缩阈值与工具调用开销未充分探索。

---

## 17. The Asymmetric Hamming Bidistance and Distributions over Binary Asymmetric Channels

**arXiv ID:** 2604.07730 | [PDF](https://arxiv.org/pdf/2604.07730v1)

**作者:** Shukai Wang `[一作]`, Zhengchun Zhou `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文引入了不对称汉明双距离（AHB）及其二维分布，旨在更好地分析在二进制不对称信道下的最大似然解码（MLD）中的错误概率。

**💡 创新点**

创新点在于提出了AHB作为一种更细致的度量，能够区分在传统度量下表现相同但实际解码性能不同的编码。

**🔧 技术方法**

使用了不对称汉明双距离（AHB）和强正则图、3类关联方案等技术来计算编码的AHB分布。

**📊 数据集**

计算了几类编码的AHB分布，包括二权和三权的投影码，以及从对称平衡不完全区块设计（SBIBD）构造的非线性码。

**📈 对比分析**

与现有的基于差异度量的界限进行比较，本文的界限在某些情况下提供了更好的区分能力，尤其是在传统度量无法区分的情况下。

**⚠️ 局限性**

限制在于完全确定一般二进制码的AHB二维分布是一个高度复杂的任务，尤其是随着码长和码本大小的增加，计算复杂度迅速增长。

---

## 18. CMP: Robust Whole-Body Tracking for Loco-Manipulation via Competence Manifold Projection

**arXiv ID:** 2604.07457 | [PDF](https://arxiv.org/pdf/2604.07457v1)

**作者:** Ziyang Cheng `[一作]` (Tsinghua University), Jiwen Lu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 Competence Manifold Projection (CMP) 框架，用于提升腿式移动机械手在面对 OOD 与不可行指令时的全身控制鲁棒性。

**💡 创新点**

创新点包括：将无限期安全约束转化为帧级单步隐空间包含；设计下界安全估计器和等距潜空间，使安全判断实现 O(1) 且可实现最佳努力追踪；引入动态 KL 正则化实现潜空间与安全概率等距。

**🔧 技术方法**

技术手段包括：CVAE 结构的意图编码器与低层策略；PPO 训练；Safety Estimator 与 TD(λ) 估计；Isomorphic Latent Space（ILS）与动态 KL 正则化；O(1) 隐空间截断投影。

**📊 数据集**

使用了仿真平台 Isaac Gym 上 Unitree Go2 + Hexfellow Saber 的数据集，包含 ID、OOD‑Geometry 与 OOD‑Sensor 三类轨迹；真实世界实验在同一硬件上采集相应任务数据。

**📈 对比分析**

与 UMI‑on‑Legs、Latent Shielding、Neural CBF 等基线对比，ID 场景下保持相近追踪精度；在 OOD 场景中生存率提升至 10 倍（例如 4.7%→46.9%），追踪误差提升不足 10%。

**⚠️ 局限性**

局限性：仅针对指令分布漂移，对动力学漂移（如负载、环境扰动）覆盖不足；安全估计器误差可能导致过度或不足保护；需手动设定安全半径 R_safe，且对极端高频噪声仍存在挑战。

---

## 19. PRISM: Evaluating a Rule-Based, Scenario-Driven Social Media Privacy Education Program for Young Autistic Adults

**arXiv ID:** 2604.07531 | [PDF](https://arxiv.org/pdf/2604.07531v1)

**作者:** Kirsten Chapman `[一作]` (Brigham Young University), Xinru Page `[通讯]` (Brigham Young University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文开发并评估了针对需要高水平支持的自闭症青年社交媒体隐私教育课程PRISM。

**💡 创新点**

创新点在于将基于规则的情景驱动教学与自闭症学习者的认知风格相匹配，填补了针对该人群隐私教育的空白。

**🔧 技术方法**

技术手段包括基于规则的决策支持、情景式社会故事、课堂前后测评以及定性访谈分析。

**📊 数据集**

数据集为29名居住式过渡年龄自闭症青年在14周课程中的课堂录音、问答记录和预后测评分数。

**📈 对比分析**

通过前后测评的配对 t 检验或 Wilcoxon 检验，四个模块显著提升了知识分数，平均提高约12%。

**⚠️ 局限性**

局限性包括样本量小、缺乏对照组、评估仅为短期测验以及仅覆盖 Facebook/Instagram 两个平台。

---

## 20. Physics-informed neural operators for the in situ characterization of locally reacting sound absorbers

**arXiv ID:** 2604.07412 | [PDF](https://arxiv.org/pdf/2604.07412v1)

**作者:** Jonas M. Schmid `[一作]` (Technical University of Munich), Steffen Marburg `[通讯]` (Technical University of Munich)

**通讯引用:** 5947 | [OpenAlex ID](https://openalex.org/A5014399890)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文利用物理信息深度算子网络（Physics‑Informed DeepONet）从近场声压和粒子速度测量中同时推断频率相关的声面导纳谱及声场分布。

**💡 创新点**

创新点在于将频率作为输入变量，在单一网络中一次性学习完整导纳谱；通过在损失函数中嵌入Helmholtz方程、线性动量方程和Robin边界条件，实现了物理一致的全局导纳估计，且无需显式前向模型或逐频求逆，显著提升了对噪声和稀疏采样的鲁棒性。

**🔧 技术方法**

使用的技术包括：物理信息深度算子网络（DeepONet）、SIREN激活函数与残差结构、梯度正则化的Helmholtz、动量和边界残差损失、AdamW优化器和余弦学习率调度。

**📊 数据集**

数据集为基于BEM仿真的合成测量数据，包含两个厚度不同的多孔吸音样本（melamine foam、PU foam）的频率相关导纳参考值，采样点338个（90%用于训练），频率范围100–5000 Hz，SNR设定为40 dB。

**📈 对比分析**

通过与仅数据驱动的DeepONet对比，采用相对L₂误差和MAC指标验证，结果显示平均导纳误差≈0.027，压力误差≈0.04，速度误差≈0.12，MAC值均>0.98；在噪声增加或采样点减少时，物理信息模型仍保持较低误差且标准差显著下降，表明性能优越。

**⚠️ 局限性**

局限性包括：需要大量高质量合成/实验测量数据；对网络超参数敏感，需要精细调优；模型可解释性较低；目前仅在合成数据上验证，真实测量验证尚待开展。

---

## 21. Mitigating Distribution Sharpening in Math RLVR via Distribution-Aligned Hint Synthesis and Backward Hint Annealing

**arXiv ID:** 2604.07747 | [PDF](https://arxiv.org/pdf/2604.07747v1)

**作者:** Pei-Xi Xie `[一作]` (CyCraft AI Lab), Cheng-Lin Yang `[通讯]` (CyCraft AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种结合分布对齐的提示合成（DAHS）和倒序提示退火（BHA）的RLVR训练框架，以提升大模型在无提示评估下的数学推理性能。

**💡 创新点**

创新点在于（1）在suffix‑only设置下通过学生风格模板对教师生成的提示进行对齐，减少提示与学生策略的分布不匹配；（2）采用按难度分桶的提示退火和每题提示随机丢弃，既能早期提供有效学习信号，又能平滑过渡到无提示推理。

**🔧 技术方法**

使用的技术包括：RLVR与可验证奖励（verifier），DAPO动态采样，DAHS提示合成，BHA分桶退火与提示丢弃，基于提示的suffix‑only策略梯度。

**📊 数据集**

实验数据集主要为数学推理集：DAPO‑Math‑17k（训练）以及AIME24/25/26、Olympiad、MATH500、Minerva Math、AMC23、GSM8K（评估）。

**📈 对比分析**

与基线方法（DAPO、SFT、BREAD、Hint‑Limited Search）比较，DAHS+BHA在AIME系列的pass@1提升约1.6‑1.8点，pass@2048提升约3.3‑10.0点，尤其在大k范围显著改进；在其他数学基准上亦取得最优或接近最优成绩。

**⚠️ 局限性**

局限性包括：模型规模受限（对较大模型效果未知）；仅使用规则式验证器；实验聚焦于数学领域，其他领域需进一步验证；以及对教师提示生成与退火调度的计算开销。

---

## 22. Event-Centric World Modeling with Memory-Augmented Retrieval for Embodied Decision-Making

**arXiv ID:** 2604.07392 | [PDF](https://arxiv.org/pdf/2604.07392v1)

**作者:** Fan Zhaowen `[一作]` `[通讯]`, Fan Zhaowen

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种事件中心的世界建模与检索框架，用于在动态安全关键环境下的自主决策，结合知识库检索与物理一致性约束，实现了实时的 UAV 冲突回避。

**💡 创新点**

创新点在于将环境抽象为语义事件、使用排列不变的潜在编码进行检索，并通过聚类贝叶斯选择避免多模平均导致碰撞，同时引入线性可约束潜在动力学保证稳定性。

**🔧 技术方法**

使用深度集合/Transformer 编码器、FAISS 近似最近邻检索、物理一致性正则化、聚类贝叶斯决策、离线仿真预训练与在线对抗性微调。

**📊 数据集**

采用基于 Virtual Potential Field 的专家轨迹数据集（27,075 条样本）以及 NVIDIA Isaac Sim 生成的对抗性航行任务数据。

**📈 对比分析**

在五轮对抗性课程中，系统在所有 100 m 任务中实现 100% 成功率、零碰撞，平均轨迹长度约 680 步，验证了在保持实时性（≤20 ms）下的高安全性与效率。

**⚠️ 局限性**

主要限制是知识库规模增大后检索延迟上升，现有实现对超过 10^5 条样本的实时性尚未保障。

---

## 23. Latent Structure of Affective Representations in Large Language Models

**arXiv ID:** 2604.07382 | [PDF](https://arxiv.org/pdf/2604.07382v1)

**作者:** Benjamin J. Choi `[一作]` (Harvard University), Melanie Weber `[通讯]` (Harvard University)

**通讯引用:** 894 | [OpenAlex ID](https://openalex.org/A5034942394)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型的情感表示几何结构，验证其与心理学的 valence–arousal 模型对齐，并提出基于几何的置信度量化方法。

**💡 创新点**

①证明 LLM 学习的情感表示呈现 V 形抛物曲线，与人类情感空间相似；②发现尽管存在非线性结构，但线性 MDS 仍能很好近似；③将表示几何用于误分类检测和置信度量化。

**🔧 技术方法**

使用经典多维尺度缩放 (MDS)、Isomap、对数回归线性探针、正交 Procrustes 对齐、层级激活采样以及校准误差 (ECE) 等技术。

**📊 数据集**

主要使用 GoEmotions 58,009 条 Reddit 评论的情感标签集，并与 ANEW 的 valence–arousal 参考数据进行对齐。

**📈 对比分析**

通过 Procrustes R² 统计检验 MDS 与 ANEW 的对齐度，Isomap 与经典 MDS 的可信度对比，误差量化模型在 Gemma‑2‑9B、Mistral‑7B 与 LLaMA‑3‑70B‑Instruct 上分别达 77.6%、85.7% 与 80.1% 的准确率，AUC‑ROC 超过 0.81，ECE 小于 0.011，显著优于多数类别基线。

**⚠️ 局限性**

未检验更大或不同家族模型的泛化；仅利用二分类逻辑回归探测几何，可能忽略更复杂结构；GoEmotions 的文化与语言偏差限制了结论的普适性。

---

## 24. The Principle of Maximum Heterogeneity Optimises Productivity in Distributed Production Systems Across Biology, Economics, and Computing

**arXiv ID:** 2604.07602 | [PDF](https://arxiv.org/pdf/2604.07602v1)

**作者:** Guillhem Artis `[一作]` (Callosum), Jascha Achterberg `[通讯]` (Callosum)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种统一的分布式生产系统（Distributed Production System）模型，刻画代理异质性、资源约束、通信拓扑与任务结构共同决定生产力、效率和鲁棒性的原理，并从模型推导出“最大异质性原理”。

**💡 创新点**

创新点在于：
1) 将跨学科已知的理论（博弈论、信息论、生态与经济的分工理论等）整合为一个可解析的、梯度可优化的框架；
2) 用异质性度量（基于Hill数与相似度）量化系统级多样性，并证明其对生产力、效率、鲁棒性的一致正向影响；
3) 通过理论与仿真实验在生物、经济、神经科学与计算机科学等多领域展示“最大异质性原理”的普适性。

**🔧 技术方法**

主要技术：
- 以一维环形托勒斯（torus）空间描述技能与操作；
- 代理技能用卷绕高斯密度描述，交互通过邻接矩阵实现；
- 生产函数为线性组合(1+Q)·w；
- 任务覆盖问题转化为覆盖成本；
- 通过梯度下降（含资源、通信和二阶约束惩罚）求解最优代理参数；
- 设计一系列衡量指标：专业化度、异质性、系统生产力、有效生产力、效能、效率等。

**📊 数据集**

实验数据主要为模拟生成的任务与网络：
- 多峰高斯任务（不同峰数、不同标准差）
- 生态学任务（多重生态位、随机环境漂移）
- 神经科学任务（双模态感官输入、拓扑为星形或三层）
- 经济学任务（双国贸易、企业内部合作）
- 计算机科学任务（混合高斯目标、神经网络参数学习、语言模型规模）
这些任务均在统一框架内随机采样或手工构造，无使用公开大规模数据集。

**📈 对比分析**

比较方法：将模型的最优解与各学科经典经验法则/理论预测进行对照，例如：
- 生态学中适应辐射与空间保险假说
- 神经科学中视觉滤波器分布与多需求系统的中心‑外围层级
- 经济学中贸易与分工、企业多样性对生产率的提升
- 计算机科学中模型异质性对学习性能和鲁棒性的提升
- 语言模型规模法则（Chinchilla）与二阶约束
通过可视化、相关系数、t检验等统计手段证明模型重现了这些趋势，性能表现表现为：在给定任务和约束下，模型最优解与实验/理论结果高度一致；异质性提升伴随生产力与鲁棒性提升。

**⚠️ 局限性**

局限性：
1) 采用一维环形空间和卷绕高斯技能，忽略更高维、多峰复杂性；
2) 生产函数线性且无交互效应，可能低估非线性系统的动态；
3) 仅考虑静态任务与一次性优化，缺乏时间演化与适应性学习的动态过程；
4) 对资源、通信与二阶约束使用简化惩罚形式，可能无法捕捉真实系统中的多层次成本结构；
5) 模型验证主要通过模拟实验，缺少实测数据的严格对照。

---

## 25. Towards Counterfactual Explanation and Assertion Inference for CPS Debugging

**arXiv ID:** 2604.07679 | [PDF](https://arxiv.org/pdf/2604.07679v1)

**作者:** Zaid Ghazal `[一作]` (University of Michigan-Dearborn), Khouloud Gaaloul `[通讯]` (University of Michigan-Dearborn)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 DeCaF 框架，利用反事实生成和因果模型对 Simulink 构建的 CPS 进行输入级调试，生成最小且必要的信号修改以恢复正确性，并推断可解释的成功断言。

**💡 创新点**

创新点在于：①将黑盒输入层面反事实解释与断言推断结合；②设计三种反事实生成策略（随机搜索、遗传算法、KD‑Tree）与两种因果模型（M5、随机森林）协同工作；③通过控制点映射实现输入时间窗的精确解释。

**🔧 技术方法**

技术手段包括：Simulated Annealing 生成训练样本；M5 模型树、随机森林、SVM、RIPPER 等因果模型；遗传算法 (GeCo)、KD‑Tree 邻域搜索和随机搜索做反事实生成；M5 决策树用于推断成功断言。

**📊 数据集**

数据集为三个工业级 Simulink CPS 基准：Automatic Transmission、Adaptive Cruise Control、Chasing Cars，涵盖 15 条 STL 需求，并通过仿真产生训练和测试输入。

**📈 对比分析**

实验对比六种配置（三种生成策略 × 两种因果模型），KD+M5 取得最高成功率（≈82%），GA+RF 亦表现良好；与随机搜索相比，前两者在成功率、必要性、充分性及断言简洁度上都有显著提升；整体性能优于传统的模型/信号级错误定位方法。

**⚠️ 局限性**

局限性包括：M5+GA 在部分案例出现不适用；方法对随机性敏感，需多次运行；仅在 Simulink CPS 基准上验证，未测试跨域适用性；输入采用分段常数，可能限制对更连续或非线性信号的解释能力。

---

## 26. EgoVerse: An Egocentric Human Dataset for Robot Learning from Around the World

**arXiv ID:** 2604.07607 | [PDF](https://arxiv.org/pdf/2604.07607v1)

**作者:** Ryan Punamiya `[一作]` (Georgia Institute of Technology), Danfei Xu `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了EgoVerse，一个可持续扩展的协同平台，用于收集、统一处理并共享视角人类演示数据，以促进机器人学习；同时发布了规模达1,362小时、1,965种任务、240个场景、2,087名演示者的公开数据集，并开展了多实验室、跨机器人执行的规模化人机迁移研究。

**💡 创新点**

创新点包括：①构建可持续增长的“活数据集”与统一的数据管理系统；②为机器人学习提供专门的操控相关标注（3D手姿、头部位姿、任务细粒度描述）；③进行跨实验室、跨机器人平台的可重复评估；④系统性揭示对齐人机数据和场景多样性在提升泛化性能中的关键作用。

**🔧 技术方法**

技术方法涵盖：①使用Aria Glasses、手机与定制硬件采集 egocentric 视频与多模态传感；②通过视觉‑惯性SLAM、手部姿态估计以及统一格式转换实现统一数据处理；③采用 transformer 编码‑解码架构与流匹配的 BC 共同训练算法实现跨实体策略学习；④在三种机器人平台上进行统一协议的强化学习与评估。

**📊 数据集**

使用的数据集为 EgoVerse 人类演示子集（EgoVerse-Human）和工业合作子集（EgoVerse-Industry），共计1,362小时、1,965任务、240场景、2,087演示者；此外，还收集了对应的机器人演示数据作为对比基准。

**📈 对比分析**

评估方法：在四个旗舰任务上，采用 ID 与 OOD 试验，机器人仅使用机器人数据与联合使用人机数据的策略进行对比；结果显示联合训练可提升 30% 以上性能，且仅在加入与任务对齐的人机数据时才出现规模化收益；不同机器人平台、实验室均验证了这一结论。

**⚠️ 局限性**

局限性：研究仅关注人机共同训练，未探索预训练/微调等更广泛算法；多样性实验多基于离线度量，缺少对应机器人跑线的验证；数据集聚焦操控任务，未覆盖更广泛的机器人行为范围。

---

## 27. Guardian-as-an-Advisor: Advancing Next-Generation Guardian Models for Trustworthy LLMs

**arXiv ID:** 2604.07655 | [PDF](https://arxiv.org/pdf/2604.07655v1)

**作者:** Yue Huang `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种Guardian-as-an-Advisor（软门控）框架，构建了GuardSet多域数据集并训练GuardAdvisor模型，使LLM在部署时能够给出风险标签和解释而非直接拒绝；

**💡 创新点**

创新点在于将风险检测与生成软交互化，通过二分类+自然语言解释的方式捕捉混合风险，同时引入专门的GuardSet数据集和SFT+RL两阶段训练，显著降低过度拒绝并提升安全与效用的平衡；

**🔧 技术方法**

技术包括Guardian-as-an-Advisor软门控模型、LLM生成标签映射与解释、监督微调（SFT）加Group-Relative Policy Optimization（GRPO）强化学习、LLM-as-a-Judge评估以及并行推理策略；

**📊 数据集**

使用了55+公开数据集，总计208k+实例，涵盖有害/无害、鲁棒性、诚实等多维标签，训练集与测试集严格分离，形成公开的GuardSet数据集；

**📈 对比分析**

与GPT‑4o、GPT‑4o‑mini、WildGuard、Llama‑Guard等基线模型比较，采用harmful/harmless accuracy衡量，GuardAdvisor平均准确率90.5%与GPT‑4o‑mini相近，且在鲁棒性与诚实方面的win rate明显高于对照组；

**⚠️ 局限性**

局限性包括评估覆盖率不足，难以覆盖所有开放式真实交互场景；使用近似模型与代理评估可能无法捕捉全部细节；系统仍面临滥用、分布漂移与公平性等更广泛挑战。

---

## 28. Spatio-Temporal Grounding of Large Language Models from Perception Streams

**arXiv ID:** 2604.07592 | [PDF](https://arxiv.org/pdf/2604.07592v1)

**作者:** Jacob Anderson `[一作]` (Toyota Motor North America), Danil Prokhorov `[通讯]` (Toyota Motor North America)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 a:tool 框架，自动生成可验证的时空查询并生成 (query, frames, match, explanation) 数据，用于提升视频语言模型的时空推理能力。

**💡 创新点**

创新点包括：①扩展 a:spre 查询语言以支持量化与量化符号，实现跨帧对象跟踪；②利用自监督的查询生成产生大规模训练集；③将自然语言解释与查询结果联合训练，显著提升推理性能。

**🔧 技术方法**

技术实现：基于正则表达式与 S4_u 空间逻辑的 a:spre；a:strem 匹配算法；对 Qwen2.5-3B-Instruct 进行监督微调（SFT）与强化学习微调（RL）；使用 LORA、PPO 等方法。

**📊 数据集**

使用 Woven Perception 数据集（180 场景、1.2K+ 感知流）生成 27k 条训练样本；评估时与 GPT‑4.1 进行对比。

**📈 对比分析**

对比方法：基线 Qwen‑3B、Qwen‑3B+SFT、Qwen‑3B+SFT+RL 与 GPT‑4.1；结果显示 SFT+RL 将帧级 F1 提升至 87.5%，仅比 GPT‑4.1（84.8%）略低，显著优于基线（48.5%）。

**⚠️ 局限性**

局限性：①需手工编写查询；②仅适用于已有预标注的感知流，对标注质量敏感；③无法自动将自然语言转为 a:spre；④仅对单一模型进行评估；⑤对存在式与空间查询的泛化仍落后 GPT‑4.1。

---

## 29. Trilinear Compute-in-Memory Architecture for Energy-Efficient Transformer Acceleration

**arXiv ID:** 2604.07628 | [PDF](https://arxiv.org/pdf/2604.07628v1)

**作者:** Md Zesun Ahmed Mia `[一作]` (Pennsylvania State University), Abhronil Sengupta `[通讯]` (Pennsylvania State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于双栅FeFET的三元计算内存（TrilinearCIM）架构，实现Transformer自注意力计算完全在非易失性内存核心内完成，避免运行时写操作；

**💡 创新点**

创新点在于利用双栅FeFET的背栅可调特性，构成三元乘加（A·B·C）原语，使得自注意力的动态乘法可以通过可变背栅电压完成，从而彻底消除动态写耗时与耐久性问题；

**🔧 技术方法**

使用双栅FeFET跨越器件，仿真框架TransCIM，混合模拟/数字电路设计（ADC、DAC、MUX、sense amp等），以及Transformer标准模型（BERT‑base、ViT‑base）和常用非线性功能单元（Softmax、LayerNorm、GELU）；

**📊 数据集**

在NLP上使用GLUE基准（CoLA、SST‑2、MRPC、RTE、STS‑B、WNLI、QNLI、QQP、MNLI），在CV上使用CIFAR‑10、CIFAR‑100、ImageNet‑1K；

**📈 对比分析**

与传统双元CIM、纯数字实现对比，TrilinearCIM在BERT‑base上实现了约18–20%延迟下降、约39–47%能耗降低、约12–13 TOPS/W的功率效率提升，并在7/9个GLUE任务上取得与数字相当或更优准确率；在ViT‑base上能耗和延迟同样降低，但由于背栅量化误差，精度略低于双元CIM；

**⚠️ 局限性**

主要局限包括：背栅量化导致视觉任务准确率下降；需要验证双栅FeFET的背栅操作范围和可靠性；额外的背栅驱动与DAC电路导致约37%面积开销；在极大序列长度下仍需考虑KV‑缓存管理与写扰等设备可靠性问题。

---

## 30. Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs

**arXiv ID:** 2604.07562 | [PDF](https://arxiv.org/pdf/2604.07562v1)

**作者:** Tunazzina Islam `[一作]` `[通讯]` (Purdue University), Tunazzina Islam (Purdue University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文提出了一种基于大语言模型的推理框架，用以验证和重构无监督聚类产生的主题结构。

**💡 创新点**

创新点在于将LLM作为语义判定者，而非生成嵌入或主题，分三阶段（连贯性验证、冗余裁决、标签归根）实现无监督主题的语义校正。

**🔧 技术方法**

使用的技术包括传统TF‑IDF+HDBSCAN聚类，LLM（GPT‑4o）进行摘要与评判，SBERT用于余弦相似度计算，UMAP降维，DGCV调参等。

**📊 数据集**

数据集为两大社交平台X（Twitter）和Bluesky的vegan话题帖子，分别采集约2万条和1.3万条英文推文/帖子。

**📈 对比分析**

与HDBSCAN原始结果、SBERT重构及传统主题模型（LDA、BERTopic、TopicGPT）等基线比较，在聚类质量指标（Silhouette、Davies–Bouldin）和人类评测标签一致率上，LLM重构显著提升连贯性与标签可解释性，且人类一致率约为90%。

**⚠️ 局限性**

限制包括仅在英文vegan领域验证、对历史时段差异未完全消除、LLM可能带来的偏见以及未对LLM进行微调，且在聚类已近最优时提升有限。

---

## 31. To Layer or Not to Layer? Evaluating the Effects and Mechanisms of LLM-Generated Feedback on learning performance

**arXiv ID:** 2604.07469 | [PDF](https://arxiv.org/pdf/2604.07469v1)

**作者:** Jie Cao `[一作]` (University of North Carolina at Chapel Hill), Kenneth R. Koedinger `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对比使用LLM生成的分层反馈与非分层反馈对学习者的学习成绩、行为与情感参与以及感知的影响，并通过中介分析探讨其机制

**💡 创新点**

首次将学习者中心的分层反馈（先鼓励+提示，再提供答案）与直接给出答案的反馈在LLM驱动系统中进行比较，并系统性地分析其正负效应及中介路径

**🔧 技术方法**

利用GPT‑5大语言模型结合检索增强生成（RAG）技术生成文本、幻灯片和语音反馈，并对比两种提示策略的输出

**📊 数据集**

收集了199名美国大学生的学习过程日志、预/后测成绩和问卷数据；学习任务来自《多媒体学习原理》章节的13个练习题（8道多选+5道开放式）

**📈 对比分析**

通过ANCOVA检验学习成绩，t检验/曼-惠特尼U检验评估行为参与，Bootstrap并行中介模型评估情感、行为、认知途径；结果显示非分层反馈在学习成绩上显著优于分层反馈，分层反馈虽提升鼓励感与自主感，但导致更多多次提交、认知负荷升高，整体效果负向

**⚠️ 局限性**

局限在于任务水平偏低、仅关注短期学习效果、单一学科场景、主要依赖日志数据，缺乏眼动或思考过程等更细致的过程指标

---

## 32. COSMIC: Emotionally Intelligent Agents to Support Mental and Emotional Well-being in Extreme Isolation: Lessons from Analog Astronaut Training Missions

**arXiv ID:** 2604.07589 | [PDF](https://arxiv.org/pdf/2604.07589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 33. Label Leakage Attacks in Machine Unlearning: A Parameter and Inversion-Based Approach

**arXiv ID:** 2604.07386 | [PDF](https://arxiv.org/pdf/2604.07386v1)

**作者:** Weidong Zheng `[一作]` (Guangzhou University), Yatie Xiao `[通讯]` (Guangzhou University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并实现了在机器无学习场景下对被遗忘数据标签的泄露攻击，提出基于模型参数相似度/差异以及基于模型反演的白盒和黑盒攻击方案；

**💡 创新点**

创新点在于首次系统分析类别级无学习导致的标签泄露风险，提出多种攻击维度（参数点积、差分、Youden指数、k-means、决策树、阈值与熵筛选），并证明参数差分攻击对多类遗忘场景具有更强鲁棒性；

**🔧 技术方法**

利用深度网络参数向量运算、k-means聚类、决策树分类、梯度优化与遗传算法的模型反演，以及阈值和熵判别技术；

**📊 数据集**

在MNIST、Fashion‑MNIST、SVHN、CIFAR‑10四个公开数据集上评估，使用LeNet与ResNet‑18两种模型；

**📈 对比分析**

对比五种主流无学习算法（Re‑Train、Fine‑Tune、Random‑Label、Amnesiac Unlearn、Negative‑Gradient）进行实验，白盒参数差分攻击平均ASR>90%，白盒反演阈值/熵攻击>70%，黑盒反演约70–80%，显示参数攻击在多类遗忘时更稳定；

**⚠️ 局限性**

局限在于依赖模型参数正定性、特征白化及类间正交等假设，且实验仅覆盖四个数据集与单一网络结构，未探讨更复杂网络、持续学习或更大规模遗忘场景下的攻击效果。

---

## 34. A Graph Foundation Model for Wireless Resource Allocation

**arXiv ID:** 2604.07390 | [PDF](https://arxiv.org/pdf/2604.07390v1)

**作者:** Yucheng Sheng `[一作]` (Southeast University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 45520 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于图形基础模型（GFM‑RA）的无线资源分配框架，利用自监督预训练与细调实现对不同目标和网络拓扑的快速适配。

**💡 创新点**

创新点包括：①引入干扰感知的 Transformer 与偏置投射器，能够在全连通干扰图上实现全局注意力并保持物理拓扑约束；②构建混合自监督预训练策略，融合遮掩边预测与无负样本的教师‑学生对比学习，提升对物理干扰结构的迁移性；③通过预训练学习通用表示，显著提升少样本适配效率与 OOD 鲁棒性。

**🔧 技术方法**

主要技术手段：图 Transformer（多头注意力 + 偏置投射），遮掩边预测（Masked Edge Prediction），教师‑学生对比学习（Teacher‑Student Contrastive），两阶段细调（头部预热 + 全部微调）。

**📊 数据集**

数据集：共 20 个仿真数据集，其中 15 个（𝒟₁–𝒟₁₅）用于预训练，5 个（𝒟₁₆–𝒟₂₀）用于 OOD 下的少样本细调，所有数据基于设备‑对‑设备（D2D）网络模拟，包含不同用户密度与距离分布。

**📈 对比分析**

与传统方法（WMMSE‑Best、PCGNN、Full Reuse）以及不同规模的 GFM‑RA 进行比较。实验表明：①在 sum‑rate、PF 与 QoS 三种目标下，GFM‑RA 与 WMMSE‑Best 接近或超过其性能；②即使在仅 2,048 训练样本下，GFM‑RA 仍显著优于 PCGNN（20,480 样本）；③在 50% 边缺失的 CSI 场景中，GFM‑RA 仍保持 0.9‑以上的 sum‑rate 比例，并在 QoS 任务中实现 1.6× 的 WMMSE‑Best 性能。

**⚠️ 局限性**

局限性：①需要大量无标签网络实例进行预训练，实际部署时收集成本高；②Transformer 对图的显式归纳偏置较弱，虽然通过偏置投射缓解，但在极大规模网络（数千节点）时仍可能面临计算与内存瓶颈；③模型对干扰图的全连通假设可能不适用于稀疏或动态拓扑的真实网络；④在极端 OOD 场景或非标准目标（如能耗约束）下的表现仍需进一步验证。

---

## 35. Smells Like Fire: Exploring the Impact of Olfactory Cues in VR Wildfire Evacuation Training

**arXiv ID:** 2604.07699 | [PDF](https://arxiv.org/pdf/2604.07699v1)

**作者:** Alison Crosby `[一作]` (University of California, Santa Cruz), Sri Kurniawan `[通讯]` (University of California, Santa Cruz)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一项小规模用户实验中，研究人员使用Meta Quest 2 VR 设备，让18名大学生参与一个模拟野火疏散的打包游戏，并比较了加入烟雾嗅觉刺激与不加入嗅觉刺激两种条件下的体验差异。

**💡 创新点**

创新点在于首次将真实的烟雾气味引入VR野火疏散训练，探究嗅觉刺激如何提升沉浸感、视觉清晰度以及参与者对疏散准备的感知。

**🔧 技术方法**

技术手段包括：Unity 开发的 VR 游戏、Meta Quest 2 头戴显示器、电子精油扩散器（Campfire 与 Smoke 两种油）提供烟雾气味。

**📊 数据集**

使用的数据集为实验参与者在实验前后完成的问卷调查数据，共计 18 组（每组 9 名受试者）的 Likert 量表评分。

**📈 对比分析**

比较方法采用 Wilcoxon 符号秩检验和 Mann‑Whitney U 检验；结果显示烟雾组在沉浸感（p = 0.025）、视觉清晰度（p = 0.046）以及对疏散物品准备知识感知（p = 0.007）等指标上显著优于对照组，整体表现提升明显。

**⚠️ 局限性**

局限性包括样本量极小、仅限于熟悉 VR 的年轻大学生、实验地点与设备存在差异、嗅觉扩散器规模有限，且未进行长期效果跟踪。

---

## 36. OpenPRC: A Unified Open-Source Framework for Physics-to-Task Evaluation in Physical Reservoir Computing

**arXiv ID:** 2604.07423 | [PDF](https://arxiv.org/pdf/2604.07423v1)

**作者:** Yogesh Phalak `[一作]` (Virginia Tech), Suyi Li `[通讯]` (Virginia Tech)

**通讯引用:** 4164 | [OpenAlex ID](https://openalex.org/A5042738110)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 OpenPRC——一个统一的物理储存计算框架，整合了物理仿真、实验轨迹提取、学习层、分析与优化等模块，并通过统一的 HDF5 方案实现模拟与实验数据的无缝衔接。

**💡 创新点**

创新点在于：① 架构化的五模块设计与模式驱动的数据接口，② 结合 GPU 加速的混合 RK4–PBD 物理引擎，③ 支持视频轨迹直接进入 RC 流程的实验摄像头采集层，④ 通过信息论指标（IPC、记忆容量）实现对储存计算机能的系统评估，⑤ 兼容外部物理引擎，为跨子系统的可重复研究提供标准化层。

**🔧 技术方法**

使用了 GPU 加速的混合 RK4–PBD 动力学求解、SIFT/ORB/Akaze 关键点检测与 KLT 光流跟踪、基于物理学的条形‑铰链模型、HDF5 统一 schema、线性/非线性相关性与重构能力评估、岭回归等读出训练方法。

**📊 数据集**

主要数据集包括：高保真模拟的 Origami 织物轨迹、由摄像机捕获的物理 Origami 实验视频以及可通过 HDF5 schema 导入的第三方仿真数据（如 PyBullet、PyElastica、MERLIN）。

**📈 对比分析**

通过 NARMA2 任务、IPC 记忆容量与线性/非线性分解等基准，对比了 GPU 版混合 RK4–PBD 与传统 CPU/MATLAB 求解的速度（显著提升）以及在原始与实验轨迹上的预测误差（NRMSE 可与现有 PRC 研究保持竞争水平）。

**⚠️ 局限性**

限制主要包括：模块成熟度不均（学习/分析层仍在完善中），当前物理模型仅聚焦条形‑铰链，尚未完整集成 PyBullet、PyElastica、MERLIN 等主流仿真器，且实验条件对输入信号的理想化假设影响了信息论指标的准确性。

---

## 37. Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation

**arXiv ID:** 2604.07486 | [PDF](https://arxiv.org/pdf/2604.07486v1)

**作者:** Qian Ma `[一作]` (Pennsylvania State University), Sarah Rajtmajer `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了RPSG方法，利用私有文本种子进行抽象化、正式差分隐私选取候选、LLM生成变体，并通过NLL过滤和PII去除，最终生成高保真、隐私友好的合成文本。

**💡 创新点**

创新点包括：①将私有种子作为一次性一对一映射，保证合成文本与原数据的高相似度；②引入情感对齐的抽象模型与正式DP机制（高斯机制）进行候选选择，兼顾语义保真与隐私；③使用NLL阈值过滤以降低成员推断风险；④在整个流程中保持对多模型的可移植性与高效性。

**🔧 技术方法**

技术手段包括差分隐私（Gaussian机制、DP-SGD）、抽象模型（bart-large-cnn）、情感分类（sentiment-roberta-large-english）、LLM推理（GPT‑3.5‑turbo、GPT‑4o‑mini、DeepSeek‑R1、Phi‑4等）、后续细化的surrogate模型、NLL分数过滤、PII 过滤、以及多种评估指标（下游任务精度/困惑度、Self‑BLEU、n‑gram、多种分布相似度指标和MIAs/PII泄露率）。

**📊 数据集**

使用的主要数据集为PubMed摘要子集（约75k训练样本）和自构建的Reddit财经困难相关对话数据（8,948训练、1,000验证、1,000测试）。

**📈 对比分析**

通过与DP‑SGD、AUG‑PE和RUPTA三种基线在同一数据集、同一模型条件下对比，下游任务表现（准确率、困惑度）、语义一致性（Self‑BLEU、n‑gram）、分布相似度（FID、KLD等）以及隐私评估（MIA AUC、PII泄露率）。实验显示RPSG在绝大多数指标上优于基线，尤其在下游任务准确率与语义保真度上表现突出，同时MIA AUC接近50%（表明强隐私保护），PII泄露率低于1%。在有限DP预算下，RPSG保持了相对稳定的性能，显示了良好的隐私-效能权衡。

**⚠️ 局限性**

局限性主要有：①对不同LLM特性和参数（如温度、阈值）敏感，需要精细调参；②评估仅聚焦单样本隐私泄露，未考虑聚合级别的重构或群体隐私风险，未来需要开展集体攻击与组级DP研究。

---

## 38. Position Paper: From Edge AI to Adaptive Edge AI

**arXiv ID:** 2604.07360 | [PDF](https://arxiv.org/pdf/2604.07360v1)

**作者:** Fabrizio Pittorino `[一作]` (Politecnico di Milano), Manuel Roveri `[通讯]` (Politecnico di Milano)

**通讯引用:** 4017 | [OpenAlex ID](https://openalex.org/A5035547226)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“Adaptive Edge AI”的概念，认为在长期真实部署中，边缘 AI 必须具备自适应能力，并用 Agent‑System‑Environment（ASE）框架阐明自适应的要素与机制；

**💡 创新点**

将自适应性从传统压缩/部署视角提升为操作性必需，系统性地列出十项未来研究挑战，并制定统一的评估与报告标准，强调在时间维度上的性能评估与成本透明度；

**🔧 技术方法**

使用了理论推导、POMDP 视角、动态神经网络、任何时推理、模块化设计、无监督自适应、联邦学习、软硬件协同等多种技术思路作为讨论基础；

**📊 数据集**

本文为位置论文，没有使用具体数据集，而是以公开的边缘 AI 基准和模拟漂移脚本为讨论背景；

**📈 对比分析**

没有实验对比，作者提出了时间序列评估指标（如 Energy‑to‑Recover、Time‑to‑Recover、稳定性得分）和报告清单，以供未来工作在统一基准上进行比较；

**⚠️ 局限性**

局限性在于缺乏实证验证，仅提供框架与挑战列表，实际实现细节与性能数据仍待后续研究完成。

---

## 39. RefineRAG: Word-Level Poisoning Attacks via Retriever-Guided Text Refinement

**arXiv ID:** 2604.07403 | [PDF](https://arxiv.org/pdf/2604.07403v1)

**作者:** Ziye Wang `[一作]` (Huazhong University of Science and Technology), Kailong Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 61854 | [OpenAlex ID](https://openalex.org/A5100355692)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种两阶段的 RAG 知识污染攻击框架 RefineRAG，利用宏观生成与微观词级优化相结合，在保持攻击效果的同时实现文本自然性。

**💡 创新点**

创新点在于把攻击视为整体词级文本细化问题，放弃粗粒度拼接，采用检索器引导的词级优化（WLO）来最大化检索相似度且保持语法正确，从而显著提升隐蔽性。

**🔧 技术方法**

技术包括宏观生成（使用大语言模型生成毒性种子并验证）、词级优化（基于 MLM + POS 标记的掩码替换、检索器相似度评估、Beam Search 搜索）、检索器与生成器的协同工作。

**📊 数据集**

使用了 Natural Questions (NQ) 和 MSMARCO 两个问答数据集，并利用公开模型如 Contriever、Deepseek-V3、Llama-2-7B、Vicuna-7B 进行实验。

**📈 对比分析**

与 PoisonedRAG、Prompt Injection、Corpus Poisoning 等基线对比，RefineRAG 在 NQ 上 90% ASR、MSMARCO 上 83% ASR，同时在困惑度、语法错误率、重复率等隐蔽性指标上均优于对手，且在不同检索器和 LLM 上表现出良好可迁移性。

**⚠️ 局限性**

局限包括计算开销高（多轮 MLM 迭代）、依赖代理检索器相似度、以及对更高级语义防御（如外部事实核查）的鲁棒性尚未评估。

---

## 40. When Equality Fails as a Rewrite Principle: Provenance and Definedness for Measurement-Bearing Expressions

**arXiv ID:** 2604.07626 | [PDF](https://arxiv.org/pdf/2604.07626v1)

**作者:** David B. Hulak `[一作]` (Independent Researcher), Ruy J. G. B. de Queiroz `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套统一的语义框架，用于对含测量值的表达式进行安全重写。

**💡 创新点**

创新点在于同时引入测量令牌的 provenance（依赖）与 admissible-domain（定义域）两个轴，构造了兼容的重写判定，并证明了其一致性、可恢复性、严格性和不充分性。

**🔧 技术方法**

主要技术包括令牌敏感的 enclosure semantics、定义域精细化、支持相对判定，以及在 Lean 4 中进行的完全无错误的形式化与证明。

**📊 数据集**

文章未使用任何外部数据集，而是通过形式化构造了一系列示例与分离/非收敛 witness 来验证理论。

**📈 对比分析**

由于该工作是理论框架与形式化验证，未进行实验比较；其“性能”体现在 Lean 4 证明无误、可重用性和可扩展性上。

**⚠️ 局限性**

局限性包括仅处理除法导致的奇异性、使用 ℚ 而非 ℝ、缺乏单位一致性系统、未覆盖溢出/下溢和类型错误，以及不涉及可判定性、收敛性或终止性等传统重写问题。

---

## 41. Bootstrapping Sign Language Annotations with Sign Language Models

**arXiv ID:** 2604.07606 | [PDF](https://arxiv.org/pdf/2604.07606v1)

**作者:** Colin Lea `[一作]` (Apple), Leah Findlater `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了基于伪注释的美国手语数据标注方案，构建了从英语文本和手语视频到时间对齐注释的管道。

**💡 创新点**

创新点是将LLM生成多候选手语翻译与手语指法识别、孤立手语识别相结合，生成候选注释并通过模型得分排序；同时在指法识别和孤立手语识别上达到SOTA。

**🔧 技术方法**

使用Claude Sonnet 4.5进行多候选翻译，TCN架构的指法识别模型，改进的孤立手语识别模型（加入手性校正和两分支），以及CTC、二元交叉熵等损失；并使用强制对齐。

**📊 数据集**

使用了ASL STEM Wiki、FLEURS-ASL、FSBoard、ASL Citizen等数据集；并手工标注了约500个视频的手语注释。

**📈 对比分析**

与现有指法识别基准相比，FSBoard上CER从7.3降到6.75；孤立手语识别Top‑1提升至74%；对手语注释的BLEURT/ChrF/Comet得分分别为0.612/0.589/0.715。

**⚠️ 局限性**

限制包括对指法识别的词长依赖、对非词汇手语的识别仍有误差、LLM候选翻译覆盖率不足、需要手工校验等。

---

## 42. Training-free Spatially Grounded Geometric Shape Encoding (Technical Report)

**arXiv ID:** 2604.07522 | [PDF](https://arxiv.org/pdf/2604.07522v1)

**作者:** Yuhang He `[一作]` `[通讯]` (Microsoft Research), Yuhang He (Microsoft Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Zernike基函数的训练无关、任务无关的二维空间几何形状编码框架XShapeEnc；

**💡 创新点**

创新点在于将形状几何和姿态分别映射到单位圆盘上的Zernike基，构造可逆、可解释、频率丰富的编码，并通过频率传播(FreqProp)进一步增强高频信息；

**🔧 技术方法**

采用Zernike多项式展开、谐波姿态场、频率传播、相位调制等数学工具实现编码；

**📊 数据集**

使用自构造的XShapeCorpus数据集（包含多级运算生成的多样形状与姿态）以及Mendeley、MPEG‑7、OpenStreetMap等公开数据集进行实验；

**📈 对比分析**

与11种基线（包括边界点、点集、训练无关/训练有根据的网络）比较，XShapeEnc在形状检索、拓扑关系分类和空间音频目标区域控制任务上取得了显著性能提升（mAP最高0.91，SDR最高17.62等）；

**⚠️ 局限性**

局限在于对极端复杂形状仍需更高编码长度；姿态编码依赖于预先设计的径向窗口，可能在极端变形下失效；对三维形状的推广尚未验证。

---

## 43. Don't Measure Once: Measuring Visibility in AI Search (GEO)

**arXiv ID:** 2604.07585 | [PDF](https://arxiv.org/pdf/2604.07585v1)

**作者:** Julius Schulte `[一作]` (Aurora Intelligence), Philipp Kaufmann `[通讯]` (University Of St Gallen)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究生成式搜索引擎中品牌可见度与来源稳定性，采用每日与同日多次查询测量其概率分布与波动；

**💡 创新点**

首次证明可见度是概率分布而非单点，并量化不稳定性，提出至少7–8次查询、跨提示、多周窗口的监测框架；

**🔧 技术方法**

使用Jaccard与Rank-Biased Overlap（RBO）评估相似度，计算Gini系数衡量来源集中度；

**📊 数据集**

构建两套数据集：45–46天每日结果集（四引擎、四行业）和同日多次查询集（最多10次），均来自瑞士服务器；

**📈 对比分析**

比较日间相似度与同日多次查询相似度，发现源/品牌重叠仅34–59%，Gini平均0.715；同日多次查询稳定性与日间相似度相近，说明主要源于模型随机性；

**⚠️ 局限性**

局限包括采集时间窗有限、仅瑞士IP、品牌识别基于关键词匹配、部分引擎零引用率、数据来源不统一等。

---

## 44. Tunneling-Augmented Simulated Annealing for Short-Block LDPC Code Construction

**arXiv ID:** 2604.07365 | [PDF](https://arxiv.org/pdf/2604.07365v1)

**作者:** Atharv Kanchi `[一作]` `[通讯]` (Illinois Mathematics and Science Academy), Atharv Kanchi (Illinois Mathematics and Science Academy)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于隧道增强模拟退火（TASA）的全局离散优化框架，用于直接优化短块LDPC码的校验矩阵，并结合局部精炼提升结构质量。

**💡 创新点**

将量子退火中隧道思想引入经典模拟退火，形成TASA，同时在能量函数中同时惩罚4/6周期、捕获集、度分布与禁忌子图，实现多目标约束下的LDPC构造。

**🔧 技术方法**

采用隧道增强模拟退火（TASA）+ 经典局部搜索、约束修复、并行重启等技术，能量函数基于图论指标（4/6周期、度偏差、低度惩罚、无效约束）。

**📊 数据集**

在块长为64、96、128、码率0.5的LDPC码上，利用AWGN信道的Monte‑Carlo仿真（每个SNR点1000次）与随机LDPC、PEG进行比较。

**📈 对比分析**

与随机LDPC比较时，在BLER=10⁻²处平均提升0.45 dB；与PEG比较时，非约束下性能差距在±0.6 dB；在受限场景下能够实现结构与性能的显著权衡。

**⚠️ 局限性**

计算量比PEG高4–5个数量级；结构改进不一定转化为解码性能提升；仅适用于离线设计，实时快速构造仍需采用PEG等贪婪方法。

---

## 45. How Independent are Large Language Models? A Statistical Framework for Auditing Behavioral Entanglement and Reweighting Verifier Ensembles

**arXiv ID:** 2604.07650 | [PDF](https://arxiv.org/pdf/2604.07650v1)

**作者:** Chenchen Kuai `[一作]` (Texas A&M University), Yang Zhou `[通讯]` (Texas A&M University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套统计框架，用于审计大型语言模型之间隐藏的行为纠缠（latent behavioral entanglement）。

**💡 创新点**

创新点在于将错误空间的多分辨率信息理论指标——困难加权行为纠缠指数 (BEI_w) 与累计信息增益 (CIG)——引入评估模型间的独立性，并在此基础上设计去纠缠的验证器加权方案。

**🔧 技术方法**

采用条件独立假设下的残差交互、易度加权、符号翻转随机化检验、方向性冲突与信息增益统计，以及基于这些指标的加权融合算法。

**📊 数据集**

在MMLU-Pro数据集上评估18个跨六大模型家族（GPT、Claude、Qwen、Llama、Gemini、DeepSeek）的行为纠缠，并在同一数据集上检验LLM-as-a-judge的偏差。

**📈 对比分析**

与传统的相关系数、准确率加权和多数投票等基线比较，去纠缠加权显著提升准确率、精确率（最高提升约4.5%），并在检验偏差上显示更高的Spearman相关性。

**⚠️ 局限性**

局限包括：只考虑二元错误同步和多选题方向性冲突，未对多模型高阶依赖做深入解析；依赖对任务难度的经验估计，可能对新颖任务不稳健；以及对数据集污染与偏差源的进一步分离尚不完整。

---

## 46. IPEK: Intelligent Priority-Aware Event-Based Trust with Asymmetric Knowledge for Resilient Vehicular Ad-Hoc Networks

**arXiv ID:** 2604.07532 | [PDF](https://arxiv.org/pdf/2604.07532v1)

**作者:** İpek Abasıkeleş Turgut `[一作]` `[通讯]`, İpek Abasıkeleş Turgut

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

研究了一种名为 IPEK 的基于事件的信任管理框架，专门用于抵御在 VANET 中利用事件严重性与位置关键度进行智能攻击的车辆。

**💡 创新点**

创新点包括：①首次在攻击模型与本地信任计算中加入事件严重性和位置关键度；②设计非对称奖励‑惩罚机制，让信任获得缓慢而失去快速；③采用 Yager 组合规则和按源可靠性排序的 DST 全局信任融合；④引入异步风险放大机制以快速响应可疑行为。

**🔧 技术方法**

使用技术：Dempster–Shafer 理论（Yager 规则）、事件/位置感知的本地信任计算、非对称奖励‑惩罚、顺序源可靠性排序、Pignistic 转换、仿真平台 OMNeT++/Veins/SUMO。

**📊 数据集**

数据集：通过 SUMO 生成的 150 辆车辆、4000m×4000m 网格以及 40 个随机生成的交通事件（包含不同严重性与位置类型），所有数据均为仿真生成。

**📈 对比分析**

与现有的 TCEMD 与 MDT 进行比较，采用 Recall、Precision、F1‑score 与 False Positive Rate 四项指标；IPEK 在 15%–35% 的攻击者密度下 Recall>75%，Precision 恒为 1，F1>0.86，FPR 为 0%，显著优于两种基准方法。

**⚠️ 局限性**

局限性：未涵盖信任值在网络层决策（如消息过滤、路径选择）中的应用；仅评估单一策略攻击，未考虑协同攻击和传统攻击模型；阈值参数由经验分析确定，缺乏正式优化；仿真环境对真实城市高密度交通场景的覆盖有限。

---

## 47. MIPT-SSM: Scaling Language Models with $O(1)$ Inference Cache via Phase Transitions

**arXiv ID:** 2604.07716 | [PDF](https://arxiv.org/pdf/2604.07716v1)

**作者:** Yasong Fan `[一作]` `[通讯]`, Yasong Fan

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一种名为 MIPT‑SSM 的序列模型，通过学习测量率 p_t 在波模式与粒子模式之间动态切换，解决了线性算子中保守与衰减兼容性冲突的问题，并实现了 O(N) 记忆消耗的长序列推理。

**💡 创新点**

创新点包括：① 引入测量率 p_t 作为波粒调度器，直接映射到量子测量诱发相变；② 证明了“波‑粒子死锁”并给出分离的 Lindblad 形式实现；③ 通过相干度（entanglement entropy）实时监测相变；④ 设计了基于 p_t 的因果稀疏 KV 缓存，等价于现代 Hopfield 网络；⑤ 通过 Hillis‑Steele 并行扫描在训练中实现 O(log N) 深度。

**🔧 技术方法**

使用的技术包括：复杂状态递推 h_t = (1‑p_t)·e^{iθ_t}·h_{t‑1} + p_t·(W_r x_t + iW_i x_t)；相位角 θ_t 与测量率 p_t 的线性映射；并行前缀扫描训练；相干度 S̃(h_t) 作为相变读数；因果稀疏 KV 缓存与门控融合；以及对 Transformer 结构的对照实验。

**📊 数据集**

评测数据集：AG News（四分类）、WikiText‑103（语言建模）、长文档分类（长度从512到8192）、needle‑in‑a‑haystack（精确检索）等。

**📈 对比分析**

与 Transformer、Mamba 等 SSM 进行对照。AG News 上准确率从 0.736 提升至 0.905（+16.6%）；在 N=8192 时显存从 34 651 MB 缩减到 810 MB（42.8×）；长文档分类 MIPT‑SSM 在 2048 长度上优于 Transformer 0.857 vs 0.830；语言建模中，带 64‑槽缓存的 MIPT‑LM 在 WikiText‑103 上 PPL 92.1，僅高出 Transformer 的 1.8%；若无缓存则 PPL 为 102.2，差距 12.9%。

**⚠️ 局限性**

局限性包括：模型规模仅 14–31 M 参数，尚未验证 1 B+ 规模；训练时 Python 并行扫描速度比最优 Transformer 慢 3–5×，需 Triton 加速；语言建模无缓存时仍有 12.9% PPL 差距，说明对历史压缩存在瓶颈；以及对 CUDA 核实现的依赖。

---

## 48. Parallel Batch-Dynamic Maximal Independent Set

**arXiv ID:** 2604.07515 | [PDF](https://arxiv.org/pdf/2604.07515v1)

**作者:** Guy Blelloch `[一作]` (Carnegie Mellon University), Jared Lo `[通讯]` (University of Hawaii)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在并行批量动态环境下维护图的最大独立集（MIS）的理论高效算法，能够在每批更新后快速更新MIS，工作量与更新数成线性关系，且并行深度为多对数级。

**💡 创新点**

创新点在于：
1) 开发了新的批量影响集（batch influence set）分析方法，突破了单次更新影响集可直接合并的局限；
2) 将影响集分析与传统的LEXICOGRAPHIC-FIRST MIS 结合，构造了一个工作效率高、深度低的并行传播算法；
3) 通过引入“shell”分层处理，进一步压缩工作量与深度。

**🔧 技术方法**

使用的技术包括：
- 影响分析版本的 LFMIS 算法，用于估计受批量更新影响的顶点集合；
- 并行桶（bag）和半排序（semisort）等基础并行原语；
- 依赖图（dependency graph）与交替消除路径（AE-path）理论，用于证明深度；
- “shells”分层技术，按顶点排列顺序分组，限制每层内部邻居交互；
- 原子写入最小值（atomic write min）和 CAS 等同步原语。

**📊 数据集**

论文主要为理论分析，并未在真实图数据集上进行实验，所有结果均来自随机排列和概率上界分析。

**📈 对比分析**

相较于现有最优的单点动态 MIS 算法（Chechik & Zhang 以及 Behnezhad 等）
- 传统算法每次更新的期望时间为 O(log⁴ n)，工作量为 O(log⁴ n)；
- 本论文提出的批量算法在单更新时工作量仅 O(log³ n)，并且批量处理时工作量为 O(b·log³ n)；
- 并行深度为 O(log⁵ n)（whp），在理论上提供了多对数级的并行性。

**⚠️ 局限性**

限制与不足：
- 结果仍包含多重对数因子，实际性能可能不如最优的单点动态算法；
- 并行深度 O(log⁵ n) 较大，实际实现中可能导致同步开销；
- 该方法尚未在实验平台上验证，实际效果与理论预期之间可能存在差距；
- 对于稀疏图的特定优化仍未充分利用，导致在某些场景下的性能不够优越。

---

## 49. FILCO: Flexible Composing Architecture with Real-Time Reconfigurability for DNN Acceleration

**arXiv ID:** 2604.07523 | [PDF](https://arxiv.org/pdf/2604.07523v1)

**作者:** Xingzhen Chen `[一作]` (Brown University), Peipei Zhou `[通讯]` (Brown University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了 FILCO，一种可实时重构的弹性组成架构，用于在多种多样的 DNN 工作负载上实现高硬件资源利用率。

**💡 创新点**

创新点在于：1）可运行时动态切换计算 Tile 大小的 AIE 编程方法；2）基于 1‑D 双缓冲的灵活 on‑chip 内存视图；3）两阶段设计空间探索（MILP + GA）以自动寻找最优映射与调度；4）统一指令集与控制平面实现多功能加速器的灵活组合。

**🔧 技术方法**

技术包括：FPGA 上的 AI Engine（AIE）多维并行计算、可重构的 Flexible Memory Unit（FMU）、Mesh Manager、I/O Manager、VLIW 指令集、两阶段 DSE（MILP + GA）、Pythran/C++ 代码生成、Xilinx Vitis 2023.1 编译链。

**📊 数据集**

使用的工作负载包括：BERT（32/64/128/256/512 词长）、DeiT、PointNet、MLP、Transformer 基准矩阵乘法等；评测在 7nm AMD Versal VCK190 开发板上进行。

**📈 对比分析**

与 CHARM 与 RSN 进行对比，FILCO 在多样性较高或规模较小的任务中实现 1.3× 至 5× 的吞吐量提升，硬件效率提升同样达到 1.3–5 倍；在大型 BERT 任务中亦显著优于对手。

**⚠️ 局限性**

局限性包括：仅在单一 FPGA 平台（VCK190）验证，尚未测试更大规模模型或跨平台适配；重配置开销与指令集复杂度对极端实时性要求仍有限制；FMU 与 AIE 的双缓冲实现需进一步优化以降低功耗。

---

## 50. On the Uphill Battle of Image frequency Analysis

**arXiv ID:** 2604.07563 | [PDF](https://arxiv.org/pdf/2604.07563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 51. Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction

**arXiv ID:** 2604.07659 | [PDF](https://arxiv.org/pdf/2604.07659v1)

**作者:** Mingchen Li `[一作]` (University of Massachusetts, Amherst), Hong yu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出K2K框架，利用LLM内部FFN层的键做知识检索并用于临床预测；

**💡 创新点**

创新点在于：①用内部键取代外部检索，消除长上下文延迟；②通过激活引导的探针构造和跨注意力重排提升检索质量；

**🔧 技术方法**

使用LoRA微调注入医学知识，Mahalanobis距离加权探针，Cross‑Attention重排，以及内部FFN键作为知识库；

**📊 数据集**

在MIMIC‑III和MIMIC‑IV数据集上进行死亡率和再入院预测；

**📈 对比分析**

与传统序列模型、KARE、标准RAG、Prompt‑based检索等方法对比，K2K在F1、Jaccard、AUPRC、AUROC四项指标上平均得分最高，尤其在MIMIC‑III死亡率任务中显著优于其他模型；

**⚠️ 局限性**

局限性包括：层选择与粒度未动态优化；仅在医学领域验证，跨领域通用性未知；对数据不平衡问题的处理仍有限。

---

## 52. Towards Rapid Constitutive Model Discovery from Multi-Modal Data: Physics Augmented Finite Element Model Updating (paFEMU)

**arXiv ID:** 2604.07746 | [PDF](https://arxiv.org/pdf/2604.07746v1)

**作者:** Jingye Tan `[一作]` (University of Southern California), Nikolaos Bouklas `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理增强的有限元模型更新(paFEMU)框架，将稀疏神经网络与对偶优化相结合，实现对多模态数据下的本构模型自适应发现。

**💡 创新点**

创新点在于：①通过物理增强的ICNN构建可保证或软约束多凸性的本构潜能；②利用L0稀疏化将高维神经网络压缩为可解释的低维表达式；③采用两阶段迁移学习（预训练+对偶更新）充分利用有限实验数据；④提出易于计算的多凸性指示器作为软约束。

**🔧 技术方法**

使用的技术包括：输入凸神经网络(ICNN)、多凸性约束与软约束、L0稀疏化、基于自动微分的可微有限元与对偶梯度、全场数字图像相关(DIC)数据的生成与处理、以及合成的Gent‑Gent、Neo‑Hookean与Ogden材料数据。

**📊 数据集**

数据集主要由三部分组成：①预训练用的Gent‑Gent模型合成的无序均质应力–变形数据；②迁移学习用的全场位移与反应力的合成DIC数据（Neo‑Hookean与通用Ogden两种目标材料）；③部署测试用的三维扭转实验的合成结果。

**📈 对比分析**

通过对三种ICNN变体（多凸、放松、多凸+稀疏化、无约束）在预训练和迁移学习阶段进行对比，评估误差、R²、梯度收敛速度等指标。结果显示：在预训练阶段，所有模型可达到高R²；迁移学习后，三种模型均能快速收敛，误差仅几‰，且在三维扭转预测中相对误差≤8.6%。

**⚠️ 局限性**

局限性包括：目前仅处理无历史耦合的弹性材料，缺乏对塑性、黏弹性等显式路径依赖材料的扩展；多凸性软约束可能削弱表达能力；稀疏化后模型在极端外推时仍可能失稳；实验验证仅基于合成数据，未在真实实验中检验鲁棒性。

---

## 53. ReflectRM: Boosting Generative Reward Models via Self-Reflection within a Unified Judgment Framework

**arXiv ID:** 2604.07506 | [PDF](https://arxiv.org/pdf/2604.07506v1)

**作者:** Kai Qin `[一作]`, Daiting Shi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文是ACL会议作者提交格式文件的补充说明，阐述了模板与风格文件的使用方法。

**💡 创新点**

创新点在于提供了可直接使用的LaTeX风格文件和完整的模板，帮助作者快速遵循ACL的格式要求。

**🔧 技术方法**

主要使用LaTeX风格文件（.sty）以及PDF生成工具，配合预设的命令和宏实现排版。

**📊 数据集**

该文档不涉及任何实验数据集。

**📈 对比分析**

文档中未包含实验方法或性能比较。

**⚠️ 局限性**

限制在于仅适用于ACL会议的格式，不能直接迁移到其他学术会议或期刊。

---

## 54. Prediction Arena: Benchmarking AI Models on Real-World Prediction Markets

**arXiv ID:** 2604.07355 | [PDF](https://arxiv.org/pdf/2604.07355v1)

**作者:** Jaden Zhang `[一作]` (Arcada Labs), Grace Li `[通讯]` (Arcada Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一个实时预测市场交易基准，通过让前沿AI模型以自治交易者身份在Kalshi和Polymarket上使用真实资本进行交易，以评估其预测准确性和决策能力。

**💡 创新点**

创新点在于：①将AI模型置于具有真实后果的实盘交易环境；②同时提供两种平台（标准化市场与开放市场发现）以检验模型在不同市场选择机制下的表现；③通过多维度指标（账户价值、盈亏、盈亏率、最大回撤、计算效率等）系统分析模型行为。

**🔧 技术方法**

采用大型语言模型（如gemini、gpt-5.2、grok-4-20-checkpoint等）作为代理，利用Web搜索、笔记、交易和发现工具，以及统一的系统提示与风险约束。

**📊 数据集**

使用Kalshi 29个精选市场和Polymarket全量市场的实时行情与结算数据，覆盖2026年1月12日至3月9日的交易日，共57天。

**📈 对比分析**

比较方法为对同一模型在两平台、两阶段（Jan‑Feb 与 Feb‑Mar）以及两代（Cohort1 与Cohort2）下的账户价值、回报率、盈亏率和最大回撤进行对照；结果显示Cohort1模型整体呈负回报，最高回报仅为-16%，而Cohort2在3天纸质交易中出现正收益（最高+6.02%），且平台差异显著。

**⚠️ 局限性**

主要局限包括：实盘交易受流动性与对手方限制，导致执行失败；纸质交易缺乏真实成交约束；统一提示可能未充分激发各模型潜能；数据周期短暂，未覆盖长期行情；平台设计差异导致结果难以直接可比。

---

## 55. Program Analysis Guided LLM Agent for Proof-of-Concept Generation

**arXiv ID:** 2604.07624 | [PDF](https://arxiv.org/pdf/2604.07624v1)

**作者:** Achintya Desai `[一作]` (University of California, Santa Barbara), Tevfik Bultan `[通讯]` (University of California, Santa Barbara)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 PAGENT，一个结合静态与动态分析并基于 LLM 的自动 PoC 生成框架，能从源代码与代码位置生成漏洞证明。

**💡 创新点**

将轻量级静态分析、规则驱动规则基静态分析、动态分析反馈与 LLM 代理循环结合，实现高成功率且低成本的 PoC 自动化。

**🔧 技术方法**

利用 LLVM IR 静态分析（入口驱动、指针解析）、Soufflé Datalog 规则库、OpenHands 代码行动 LLM 代理、AddressSanitizer/MemorySanitizer/UBSan 动态分析、Python/JSON 交互等技术。

**📊 数据集**

在 ARVO（Cybergym）数据集上评估，共 203 个 OSS‑Fuzz 漏洞实例，覆盖 10 个 C/C++ 开源项目。

**📈 对比分析**

与 Cybergym 级别 0–2 LLM 代理、PoCGen、Faultline 进行对比，PAGENT 在 DeepSeek3.2 上成功率 64.6%，约提升 132%；单模型即可产生 130 个成功 PoC，明显超过其他组合。

**⚠️ 局限性**

依赖漏洞规则集，若未覆盖会失效；对未见软件的通用性尚待验证；LLM 可能已在训练中见过代码或 PoC，导致误判；对隐藏或补丁后仍存的漏洞可能产生误报。

---

## 56. Dual-Rerank: Fusing Causality and Utility for Industrial Generative Reranking

**arXiv ID:** 2604.07420 | [PDF](https://arxiv.org/pdf/2604.07420v1)

**作者:** Chao Zhang `[一作]` (Kuaishou Technology), Jingwei Zhuo `[通讯]` (Unaffiliated)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了Dual‑Rerank框架，通过序列知识蒸馏把自回归（AR）模型的顺序依赖迁移到并行非自回归（NAR）模型，并结合List‑wise Decoupled Reranking Optimization (LDRO) 进行在线强化学习，实现在短视频搜索中高效且精准的重排。

**💡 创新点**

核心创新在于：1) 提出了Unimodal Concentration假设与序列知识蒸馏，能将AR的顺序信息压缩进NAR；2) 设计了LDRO算法，通过向量化Gumbel‑Max采样、双重解耦归一化与排名加权策略，解决了非平稳奖励、采样效率和位置敏感性三大挑战。

**🔧 技术方法**

使用自回归Pointer Network作为教师，非自回归并行模型为学生，利用KL蒸馏、Vectorized Gumbel‑Max采样、双重解耦归一化、排名加权和Policy Gradient等技术。

**📊 数据集**

采用公开的Avito Context Ad Clicks数据以及规模达10亿条交互的Kuaishou生产日志。

**📈 对比分析**

与传统点式/列表式评分方法及SOTA AR/NAR重排器比较，Dual‑Rerank在AUC、NDCG以及在线指标（Long‑View Rate、Query Reformulation Rate）上均提升约1–2%，同时实现约43%推理延迟下降。

**⚠️ 局限性**

局限性包括：对Unimodal假设的依赖在其他业务场景下可能不成立；RL训练仍需大量数据且对奖励设计敏感；模型在跨任务通用性和对更大规模基础模型的适配方面尚需进一步研究。

---

## 57. DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification

**arXiv ID:** 2604.07622 | [PDF](https://arxiv.org/pdf/2604.07622v1)

**作者:** Ziyi Wang `[一作]` (Purdue University), Qifan Song `[通讯]` (Purdue University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种动态集成验证器，改进了speculative decoding的接受率和效率

**💡 创新点**

在每一步根据上下文自适应地混合目标模型和草稿模型的分布，从而突破静态集成的Pareto前沿

**🔧 技术方法**

结合speculative decoding、动态权重网络、REINFORCE++强化学习、正则化接受率约束

**📊 数据集**

在GSM8K、CNNDM、XSum、MBPP四大基准数据集上进行实验

**📈 对比分析**

与标准speculative decoding、静态集成、SD(Lossy)和SpecCascade等基线相比，DIVERSED显著提升接受率、保持甚至提升生成质量，并在同等质量下实现更低的实际延迟，整体性能优于所有对比方法

**⚠️ 局限性**

需要任务特定训练；跨数据集迁移时接受率提升但质量下降；当前仅在token级别实现放宽，block级别放宽仍待探索

---

## 58. HiMARS: Hybrid multi-objective algorithms for recommender systems

**arXiv ID:** 2604.07572 | [PDF](https://arxiv.org/pdf/2604.07572v1)

**作者:** Elaheh Lotfian `[一作]`, Alireza Kabgani `[通讯]` (University of Antwerp)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出四种新的混合多目标算法（HANv1、HANv2、HANIv1、HANIv2），用于解决推荐系统中的准确率与多样性权衡问题。

**💡 创新点**

创新点在于将AMOSA、NSGA-II和NNIA的优势融合，形成两代混合算法，并提出基于理想解距离的个性化列表选择方法。

**🔧 技术方法**

采用AMOSA、NSGA-II、NNIA等进化优化技术，并结合协同过滤与加权求和预测实现候选生成和多目标搜索。

**📊 数据集**

在MovieLens和ModCloth这两个真实数据集上进行实验，分别覆盖中等至高稀疏度场景。

**📈 对比分析**

通过准确率、Diversity、Novelty以及Pareto前沿质量指标（SM、MID、DM、SNS）和TOPSIS评分进行比较，实验显示HANv2在多项指标上优于基线，并能生成更均匀的Pareto前沿。

**⚠️ 局限性**

局限包括仅针对单用户实验、规模受限、仅考虑两个目标、缺乏自适应参数调优等。

---

## 59. Behavior Latticing: Inferring User Motivations from Unstructured Interactions

**arXiv ID:** 2604.07629 | [PDF](https://arxiv.org/pdf/2604.07629v1)

**作者:** Dora Zhao `[一作]` (Stanford University), Michael S. Bernstein `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种名为行为层析（Behavior Latticing）的架构，用以从用户的无结构交互数据中推断其潜在动机，并将这些洞察应用于个人AI代理，提升其满足用户深层需求的能力。

**💡 创新点**

创新点在于：①将用户行为视为可多对多关联的观察点，通过递归层级构建“格子”结构，使观察与洞察在不同层级之间不断交叉、聚合，从而挖掘更深层的、跨时间、跨情境的动机；②在洞察生成过程中保持解释深度与准确度的平衡；③将洞察直接驱动代理行动，展现基于用户动机的个性化决策。

**🔧 技术方法**

使用的技术包括：大型语言模型（Gemini、Claude Sonnet）、视觉语言模型+OCR（EasyOCR）对截图进行语义抽取；行为层析算法（递归组块、多对多映射、层级推理）；工具调用式代理（ReAct/Genie）执行基于洞察的行动；评估时采用定量 Likert 量表、统计检验（t检验）与定性访谈。

**📊 数据集**

数据集主要为：①用户与ChatGPT的对话日志；②四天以上的屏幕截图记录（含键盘、鼠标事件）；实验共招募9名参与者进行技术评估，12名参与者进行端到端评估，所有数据均在受控环境下收集并去标识化。

**📈 对比分析**

与现有的基于观察的用户建模方法（如 General User Models）对比，Behavior Latticing 在洞察深度方面显著提升（平均评分+1.54，p<0.001），而准确度差异不显著；在代理行动评估中，洞察驱动的行动在满足用户潜在需求的评分上优于仅基于任务上下文的基线（t=2.69，p=0.01），同时保持即时效用相同。

**⚠️ 局限性**

局限性包括：①观察窗口有限，导致对长期或周期性行为的洞察不完整；②模型可能产生错误的规范性判断或过度解释；③仅从第三方视角获取行为，缺乏对用户自我感知的理论视角；④当前实现仍以单向递归层析为主，未探索向下传播或双向交互的更细粒度洞察。

---

## 60. Playing DOOM with 1.3M Parameters: Specialized Small Models vs Large Language Models for Real-Time Game Control

**arXiv ID:** 2604.07385 | [PDF](https://arxiv.org/pdf/2604.07385v1)

**作者:** David Golchinfar `[一作]` (VAGO Solutions), Alexander Marquardt `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在DOOM第一人称射击游戏中，作者训练了一个1.3M参数的文本模型SauerkrautLM-Doom-MultiVec，能够在实时（≈31 ms/决策）下进行游戏控制，并在10集成中获得178个击杀（平均17.8/集）

**💡 创新点**

核心创新包括：①将ModernBERT与哈希嵌入（hash embeddings）结合，显著压缩词嵌入参数；②使用深度感知的ASCII编码，将灰度字符与16区间深度嵌入相融合；③采用注意力池化（attention pooling）而非CLS/平均池化，形成信息瓶颈；④使用软标签和KL损失提升小数据集上的学习；⑤基于人类演示的极低数据量（约2 h）实现高效训练

**🔧 技术方法**

使用技术：ModernBERT-Hash编码器、hash embedding、深度嵌入、字符级BPE tokenizer、注意力池化分类头、KL损失、bfloat16混合精度训练、VizDoom平台、ASCII+深度编码、复合动作决策策略、16-bin量化深度、1 024字符令牌、30 ms CPU推理

**📊 数据集**

数据集：31 645帧人类演示（约2 h游戏），每帧包含40×25 ASCII字符、对应16区间深度、软动作标签（四个动作的概率分布）

**📈 对比分析**

与Nemotron‑120B、Qwen3.5‑27B、GPT‑4o‑mini、Gemini Flash Lite等大型多模态LLM在同一ASCII+深度输入下进行公平基准；SauerkrautLM在10集成中获得178击杀，远超所有LLM总和（13击杀），平均生存步数388步（LLM ≤105步），推理时间31 ms（LLM 0.6–13 s），参数量1.3 M vs 120 B/27 B；展示了1.3 M模型在实时控制任务中的显著优势

**⚠️ 局限性**

局限性：仅在单一DOOM场景下训练和评估，无法验证跨场景泛化；基准与模型输入严格匹配HUD与分辨率设置；未评估原始图像输入对LLM的影响；多向量分类方法仍未突破注意力池化的表现；LLM基准使用的回合数有限，统计功效受限

---

## 61. CAMO: A Class-Aware Minority-Optimized Ensemble for Robust Language Model Evaluation on Imbalanced Data

**arXiv ID:** 2604.07583 | [PDF](https://arxiv.org/pdf/2604.07583v1)

**作者:** Mohamed Ehab `[一作]` (October University for Modern Science & Arts), Khaled Shaban `[通讯]` (Qatar University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了 CAMO——一种层级化、关注少数类的集成方法，用于解决不平衡分类任务。

**💡 创新点**

创新点在于将投票分布、置信度校准和模型不确定性三方面融合成七阶段决策流程，动态提升少数类预测。

**🔧 技术方法**

采用多模型集成、LoRA 参数高效微调、置信度加权、不确定性估计及动态提升函数等技术。

**📊 数据集**

在教育评估的 BEA 2025 错误识别数据集和情感识别的 DIAR‑AI/Emotion 数据集上进行实验。

**📈 对比分析**

与七种主流集成方法对比，在 fine‑tuned 语言模型上，CAMO 在严格宏 F1、准确率等指标上均实现最高分，显著提升少数类表现。

**⚠️ 局限性**

局限性包括对模型适配度高度依赖、需要多模型训练与多阈值调优，在极端稀疏类上仍难以突破。

---

## 62. Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2604.07723 | [PDF](https://arxiv.org/pdf/2604.07723v1)

**作者:** Jiahao Li `[一作]` (Xiamen University), Yanyun Qu `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无需迭代训练或注意力调制的开词汇语义分割方法，直接通过求解分布差异的解析解生成语义分割图。

**💡 创新点**

创新点在于把传统的logits‑optimization改为直接求解分布差异的解析解，使用退化分布替代GT；并提出两种求解方式：最优传输路径（Sinkhorn）与最大速度（Markov过程）来构造分布差异。

**🔧 技术方法**

使用CLIP视觉‑语言对齐特征、cosine相似度生成logits；利用Stable Diffusion SD2的自注意力张量构造成本/转移矩阵；通过Sinkhorn算法求最优传输；Markov过程与最大速度计算；联合双边上采样生成最终分割图。

**📊 数据集**

八个标准基准数据集：Pascal VOC2012（VOC21/VOC20）、Pascal Context（Context60/Context59）、COCO‑Stuff（171类）、COCO‑Obj（81类）、ADE20K（150类）、Cityscapes（19类）。

**📈 对比分析**

与现有需要迭代训练或注意力调制的OVSS方法对比，在CLIP Base与Large两个规模上平均提升约2 mIoU；在VOC21、Context60、VOC20、COCO‑Stuff、Cityscapes实现state‑of‑the‑art；最大速度模式略优于最优路径模式。

**⚠️ 局限性**

局限性包括：仍需依赖SD2自注意力张量，性能受其质量影响；对高分辨率图像的上采样与推理效率有限；最优路径模式在光照不均的背景下易误分；两种分布差异图融合反而降低性能；不适用于实时动态场景。

---

## 63. FORGE:Fine-grained Multimodal Evaluation for Manufacturing Scenarios

**arXiv ID:** 2604.07413 | [PDF](https://arxiv.org/pdf/2604.07413v1)

**作者:** Xiangru Jian `[一作]` (University of Waterloo), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含真实2D图像和3D点云、细粒度域语义标注的多模态制造数据集，并对18款大型多模态语言模型在工件验证、结构表面检验与装配验证三大任务上进行零样本、参考条件与上下文示例三种评估，随后通过瓶颈分析和领域特定微调验证该数据集既可用作基准也可作为训练资源。

**💡 创新点**

①首个细粒度域语义的制造多模态基准；②提出三视图投影方法让通用MLLM兼容点云；③系统瓶颈分析揭示视觉定位非主要限制；④展示微调后小模型可逼近大模型性能，证明数据集可用于知识迁移。

**🔧 技术方法**

多模态大语言模型、三视图投影、视觉对齐策略、零样本/参考条件/上下文示例评估、瓶颈分析、监督微调（SFT）

**📊 数据集**

自行收集的真实制造工作件数据，约12,000个样本，包含14类工件、90个型号编号，包含2D图像与对应3D点云；数据被划分用于工件验证、结构表面检验、装配验证三大任务。

**📈 对比分析**

在Zero-shot、Reference-Conditioned和In-Context Demonstration三种设置下，采用多选准确率比较18款模型（开源与闭源）。结果表明闭源模型整体表现更好，但在细粒度型号识别与表面缺陷分类任务上仍存在显著差距；SFT后3B模型在三视图工件验证任务上提升约90.8%，逼近大型模型性能。

**⚠️ 局限性**

仍缺乏对细粒度制造知识和微观表面特征的理解，尤其在扁平垫圈等同质部件识别上表现较差；点云的原生3D编码能力不足，导致文本序列输入效果差；评估仅覆盖三类任务，无法全面涵盖所有制造场景。

---

## 64. Generative Experiences for Digital Mental Health Interventions: Evidence from a Randomized Study

**arXiv ID:** 2604.07558 | [PDF](https://arxiv.org/pdf/2604.07558v1)

**作者:** Ananya Bhattacharjee `[一作]` (Stanford University), Emma Brunskill `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了名为 GUIDE 的系统，该系统能够在运行时动态生成个性化的心理干预体验，并在一项包含 237 名大学生的预注册实验中评估其效果。

**💡 创新点**

创新点在于提出“生成式体验（generative experience）”这一新范式，将干预内容与交互结构同时视为可生成对象，从而实现既个性化又多模态、可变的干预体验。

**🔧 技术方法**

技术包括：多模态 LLM（GPT‑4.1、GPT‑4o‑mini）用于生成干预方案与交互原语，规则化评判（rubric‑guided）进行候选方案评分，模块化交互原语（文本、音频、视觉、计时器）构成可组合的交互流，以及 DALL·E 3 生成视觉素材。

**📊 数据集**

使用的数据集主要是实验收集的 237 名学生的自述情境与情绪量表（PSS‑10）数据；未使用公开的标准心理健康数据集。

**📈 对比分析**

与基于 LLM 的认知重构控制组相比，GUIDE 在减压（平均减压量 0.65 vs 0.35，p = 0.02）和用户体验（UEQ‑8 平均 0.49 vs 0.33，p = 0.04）两项主要指标上均表现更佳，显示其在单次干预中的可行性。

**⚠️ 局限性**

局限性包括：无法彻底分离生成式体验与干预多样性效应；仅在单次、学生样本下验证，缺乏长期或多样化人群的数据；依赖特定 LLM 及其生成质量，未探讨不同模型或更丰富模态的鲁棒性。

---

## 65. IatroBench: Pre-Registered Evidence of Iatrogenic Harm from AI Safety Measures

**arXiv ID:** 2604.07709 | [PDF](https://arxiv.org/pdf/2604.07709v1)

**作者:** David Gringras `[一作]` `[通讯]` (Harvard T.H. Chan School of Public Health), David Gringras (Harvard T.H. Chan School of Public Health)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 IatroBench benchmark，评估前沿 LLM 在医疗决策中的委婉遗漏（omission harm）与错误输出（commission harm），并揭示了身份相关的能力抑制现象。

**💡 创新点**

创新点包括：①双轴评分体系（omission harm 与 commission harm）并引入 acuity 权重；②Decoupling Eval 通过对同一临床情景的医师与普通人表述进行对比，量化模型在身份识别上的“规格化游戏”；③发现大多数安全训练的模型在医师框架下可给出标准治疗方案，但在普通人框架下主动回避，从而产生 iatrogenic 风险。

**🔧 技术方法**

主要技术手段为：①利用 6 种 frontier LLM（Opus、GPT‑5.2、Gemini、DeepSeek、Mistral、Llama‑4）生成 3,600 条回答；②双层评分流程——快速 LLM judge（Gemini Flash）与结构化评估（基于 Claude Opus 的医学专家协议）对每条回答进行 commission 与 omission 分数；③统计学检验（Wilcoxon、Spearman、κ‑一致性）验证假设。

**📊 数据集**

数据集：60 条经过临床验证、按 7 类别（心理危机、药物管理、急救等）划分的场景，每条场景附有黄金标准答案、关键行动列表与 acuity 权重；10 次重复生成 3,600 条目标响应；评估使用双评审体系，含 100 条医生评分样本。

**📈 对比分析**

比较方法：对每个模型计算平均 commission harm (CH) 与 omission harm (OH)，并在医师/普通人表述的 22 对匹配场景中计算 decoupling gap（OH_lay – OH_phys）。结果显示：所有模型 CH 均低于 0.5（即无重大错误输出），但 OH 均显著 > 0，且 decoupling gap 在 5 个模型中均为正，表明模型在身份识别上倾向于抑制信息传递；最高 gap 为 Opus (+0.65)。

**⚠️ 局限性**

局限性：①场景仅聚焦高风险冲突，未涵盖全部临床情境；②黄金标准由单位医生制定，虽已双人验证但仍可能存在主观偏差；③评估使用 LLM judge 与结构化评估，后者仍基于 AI 对医生评分，可能与真实临床实践差距；④模型样本仅 6 种，无法覆盖所有 LLM 设计差异；⑤对 GPT‑5.2 的内容过滤机制表明，部署层面缺陷亦会导致遗漏，需进一步研究。

---

## 66. MCP-DPT: A Defense-Placement Taxonomy and Coverage Analysis for Model Context Protocol Security

**arXiv ID:** 2604.07551 | [PDF](https://arxiv.org/pdf/2604.07551v1)

**作者:** Mehrdad Rostamzadeh `[一作]` (Old Dominion University), Daniel Takabi `[通讯]` (Old Dominion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文对Model Context Protocol（MCP）的安全性进行防御部署分析，提出以防御位置为导向的分层攻击分类法，并对49种攻击与13种防御进行覆盖率评估。

**💡 创新点**

创新点在于将攻击按可实施的架构层级映射，区分主次防御点，提供结构化的防御责任分配和对比视图，从而揭示生态层级的防御盲区。

**🔧 技术方法**

采用了结构化语义分类法、层级对齐、覆盖率统计与对比矩阵，结合文献综述和现有工具实现。

**📊 数据集**

数据集为2025-2026年公开的MCP攻击案例（49类）和现有防御实现（13款）所构成的集合。

**📈 对比分析**

通过将每种攻击与各防御的可覆盖性标记为✓/✗，计算各层级的覆盖百分比；结果显示注册/供应链与网络层覆盖低，工具层覆盖高。

**⚠️ 局限性**

局限性包括仅评估结构化覆盖度而未进行实际检测精度实验；样本来源仅限公开论文与工具，可能遗漏未公开攻击；未充分考虑多方协同攻击的交互复杂性。

---

## 67. BLEG: LLM Functions as Powerful fMRI Graph-Enhancer for Brain Network Analysis

**arXiv ID:** 2604.07361 | [PDF](https://arxiv.org/pdf/2604.07361v1)

**作者:** Rui Dong `[一作]`, Youyong Kong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和强化学习相结合的技术，构建了一个混合模型。

**📊 数据集**

采用了公开的图像识别数据集和自建的用户行为数据集进行实验。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示新算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于算法对特定类型数据的适应性较差，且在极端情况下可能出现过拟合现象。

---

## 68. Joint Task Offloading, Inference Optimization and UAV Trajectory Planning for Generative AI Empowered Intelligent Transportation Digital Twin

**arXiv ID:** 2604.07687 | [PDF](https://arxiv.org/pdf/2604.07687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 69. HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents

**arXiv ID:** 2604.07430 | [PDF](https://arxiv.org/pdf/2604.07430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 70. GEAR: GEometry-motion Alternating Refinement for Articulated Object Modeling with Gaussian Splatting

**arXiv ID:** 2604.07728 | [PDF](https://arxiv.org/pdf/2604.07728v1)

**作者:** Jialin Li `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xilin Chen `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于 Gaussian Splatting 的 EM 风格交替优化框架 GEAR，用于高保真地重建具有多关节的可运动物体，并同时估计几何结构与运动参数。

**💡 创新点**

创新点包括：①将运动参数视为模型参数、部分分割视为潜变量，实现稳定的交替优化；②利用 SAM 的 2D 细粒度分割作为弱监督，提升分割质量与泛化；③设计粗到细的初始化策略和多视角一致性约束；④通过双四元数对刚体运动进行严格硬分配，避免软权重导致的物理不一致。

**🔧 技术方法**

核心技术包括 Gaussian Splatting、EM 交替优化、SAM Mask Aggregation、KNN 聚类一致性损失、双四元数运动模型、渲染损失、深度一致性损失等。

**📊 数据集**

使用 PARIS、ArtGS-Multi 以及新构建的 GEAR-Multi 数据集（包含 10 种不同类别、共 10 个多关节对象）进行评估。

**📈 对比分析**

在所有数据集上与 Ditto、PARIS、DTA、ArtGS 等基线对比，GEAR 在几何 Chamfer 距离、关节轴角误差、位姿误差和运动误差等指标上均优于或竞争力更强，尤其在复杂多关节对象上实现了显著提升。

**⚠️ 局限性**

局限性包括：对极端大幅度旋转（如 180°）和透明材料的处理仍不理想，且在完全离群的多关节结构上可能出现收敛困难。

---

## 71. Event-Level Detection of Surgical Instrument Handovers in Videos with Interpretable Vision Models

**arXiv ID:** 2604.07577 | [PDF](https://arxiv.org/pdf/2604.07577v1)

**作者:** Katerina Katsarou `[一作]` (Fraunhofer HHI), Sebastian Bosse `[通讯]` (Fraunhofer HHI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种可解释的时空视觉框架，用于在真实手术视频中以事件级别检测手术器械交接并预测其方向。

**💡 创新点**

创新点包括：在单一模型中采用统一多任务 ViT–LSTM 结构同时预测交接发生与方向，减少级联误差；以及利用 Layer‑CAM 解释模型关注手–器械交互区域，从而提升可解释性。

**🔧 技术方法**

使用技术包括 Vision Transformer（ViT）提取空间特征、单向 LSTM 进行时序聚合、两头多任务预测（检测 + 方向）、加权交叉熵损失、Gaussian 平滑与峰值检测做事件定位，以及 Layer‑CAM 归因分析；同时与 VideoMamba 这一基于状态空间的时序模型进行对比。

**📊 数据集**

采用从五例肾移植手术中提取的实况视频，共计 484 个标注交接事件（334 辅助手术师→主刀手，150 主刀手→助手），其中一例（50 个事件）保留作测试集。

**📈 对比分析**

在同一 8 帧窗口下与单任务 ViT–LSTM 和 VideoMamba 进行对比，检测 F1 0.84（多任务）对比 0.79（单任务）及 0.84（VideoMamba），方向平均 F1 为 0.72（多任务）显著高于 VideoMamba 的 0.61，显示多任务模型在方向识别上更优。

**⚠️ 局限性**

局限性包括：数据集规模有限且交接事件稀疏，导致方向分类精度受限；ViT–LSTM 模型参数量大（304 M）且推理延迟高（≈232 ms），在极端遮挡或视角变化下鲁棒性仍待提升。

---

## 72. SAFE: Spatially-Aware Feedback Enhancement for Fault-Tolerant Trust Management in VANETs

**arXiv ID:** 2604.07552 | [PDF](https://arxiv.org/pdf/2604.07552v1)

**作者:** İpek Abasıkeleş Turgut `[一作]` `[通讯]` (Iskenderun Technical University), İpek Abasıkeleş Turgut (Iskenderun Technical University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了 SAFE（Spatially-Aware Feedback Enhancement）方法，用于 VANET 的事件驱动信任管理，在车辆离开观测区前持续记录事件信息并发送更新反馈，解决传统方法因事件状态变化导致误反馈的错误。

**💡 创新点**

创新点在于：①引入距离感知的反馈增强机制，车辆在观测区内持续记录并在离开前发送更新反馈；②在观测区与决策区之间保持记录以提高反馈准确性；③提出最优决策距离与观测距离的关系 D_d ≥ 2 × D_w。

**🔧 技术方法**

技术实现基于 Omnet++、Veins 与 SUMO 仿真平台；采用 IEEE 802.11p OBU 通信、RSU 与 CDU 的信任更新流程；通过事件模型（不同类型、持续时间、有效距离）与车辆行为距离模型来评估 SAFE。

**📊 数据集**

使用的“数据集”为仿真产生的车辆轨迹和事件日志，包含单一事件、三类多事件以及不同决策距离（200 m、300 m）场景，仿真车辆数量、路网长度等参数均在论文中给出。

**📈 对比分析**

与传统 TCEMD 方案对比，SAFE 在单事件情境下反馈报告数平均提升 2.5 倍、负反馈率从 52% 降至 25%；在多事件情境下提升超过 6 倍、负反馈率降至 <1%；误黑名单节点从 34 下降至 1，可信节点数显著增加，表明 SAFE 在信任计算准确性和网络可靠性方面表现更优。

**⚠️ 局限性**

局限性包括：仅在无攻击的理想环境下验证；未探讨攻击场景下的鲁棒性；对决策距离的选择仍需实测验证；仿真规模相对有限，缺乏真实交通数据验证。

---

## 73. Reinforcement Learning with LLM-Guided Action Spaces for Synthesizable Lead Optimization

**arXiv ID:** 2604.07669 | [PDF](https://arxiv.org/pdf/2604.07669v1)

**作者:** Tao Li `[一作]` (Emory University), Carl Yang `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一种利用受限于验证反应模板的马尔可夫决策过程进行药物先导优化的框架MolReAct，结合工具增强的大语言模型动态生成可行反应候选，并用轨迹级强化学习进行决策；

**💡 创新点**

创新点在于将反应模板匹配与LLM推理结合，形成可变、可扩展的动作空间，同时采用Group Relative Policy Optimization实现跨步奖励分配，并通过SMILES缓存显著降低LLM推理成本；

**🔧 技术方法**

核心技术包括模板匹配、ReAct框架下的工具增强LLM、GRPO强化学习、SMILES缓存机制以及Qwen-3-4B的策略网络；

**📊 数据集**

使用了Therapeutic Data Commons中的13个属性优化任务（多参数优化、重发掘、中位数目标）以及一个基于sEH的结构对接任务；

**📈 对比分析**

与GraphGA、ReaSyn、SynFormer、DrugAssist、LDMol和mCLM等基线比较，MolReAct平均Top-10分数0.563，比最强基线SynFormer提升10.4%相对改进，并在10/14任务上实现最佳样本效率；

**⚠️ 局限性**

局限性包括受限于115条预先验证的模板（覆盖度有限）、不考虑反应条件/选择性与保护基策略、构建的路径为模板级合法而非完整可执行配方、仅依赖计算oracle且奖励稀疏，且LLM模型性能对动作空间质量有重要影响。

---

## 74. Auto-Configured Networks for Multi-Scale Multi-Output Time-Series Forecasting

**arXiv ID:** 2604.07610 | [PDF](https://arxiv.org/pdf/2604.07610v1)

**作者:** Yumeng Zha `[一作]`, Xianpeng Wang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种自动配置框架，用于在预算有限的条件下搜索多尺度多输出时间序列预测模型，并输出误差–复杂度 Pareto 集合。

**💡 创新点**

创新点包括：①将预处理、网络结构与训练超参数统一到分层条件混合搜索空间；②设计短长卷积的 MS–BCNN 以捕捉局部波动与长期趋势；③提出 PHMOEA 进化策略，专门针对多目标混合变量问题逼近 Pareto 前沿；④在工业异步采样数据上实现可部署的多模型选择。

**🔧 技术方法**

使用的技术主要有：多目标进化算法 PHMOEA、层级条件编码与修复、混合变量的离散化与重编码、MS–BCNN（双分支卷积）、时间嵌入、批归一化与激活函数、归一化目标、归一化拥挤度等。

**📊 数据集**

数据集：合成 H‑DTLZ2 / H‑DTLZ7 用于验证搜索行为；真实工业数据为 2283 条 sintering 生产周期样本，含 5 个终端质量目标（TFe, FeO, SiO₂, CaO, 基性）。

**📈 对比分析**

方法对比：在合成和真实任务上与 MOEA/D、NSGA‑II、NSGA‑III、SMS‑EMOA 的 IGD/HV 进行比较，PHMOEA 在相同评估预算下显著降低 IGD、提升 HV；在预测性能上与 OB‑ISSID、Ventingformer、Transformer、LSTM、GRU‑PLS 对比，MS‑BCNN 在 NMSE、NMAE、MAPE 等指标上获得最优或同等表现。

**⚠️ 局限性**

局限性：仅在离线搜索阶段处理模型配置，未针对非平稳漂移和多阶段在线适应展开；搜索空间仍需人工设计，搜索成本高（每次评估需完整训练）；未进一步研究模型在长期运行中的自适应维护。

---

## 75. SepSeq: A Training-Free Framework for Long Numerical Sequence Processing in LLMs

**arXiv ID:** 2604.07737 | [PDF](https://arxiv.org/pdf/2604.07737v1)

**作者:** Jie Sun `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了SepSeq框架，在LLM输入中插入分隔符以减轻注意力分散，提升长数值序列处理效果。

**💡 创新点**

创新点在于利用训练无关的分隔符作为注意力汇聚点，重新定位注意力焦点，解决数值序列长尾精度问题。

**🔧 技术方法**

技术上通过Softmax注意力分析，插入特殊分隔符并在实验中验证其对注意力分布的调节作用。

**📊 数据集**

使用9款主流LLM，10个任务，涵盖6个合成数值序列任务和4个真实数据集（股票、天气、数字串、数字列表）。

**📈 对比分析**

与Vanilla、CoT、ICL、PoT等基线对比，SepSeq平均提升35.6%准确率，同时推理token消耗降低约16%。

**⚠️ 局限性**

局限在于对小参数模型（<4B）效果不佳，需要足够的模型容量才能充分利用分隔符。

---

## 76. M-ArtAgent: Evidence-Based Multimodal Agent for Implicit Art Influence Discovery

**arXiv ID:** 2604.07468 | [PDF](https://arxiv.org/pdf/2604.07468v1)

**作者:** Hanyi Liu `[一作]` (China Electronics Technology Group Co., Ltd.), Heran Yang `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多模态证据推理代理，专门用于检测艺术史中隐含的影响关系。

**💡 创新点**

创新点包括：将影响检测重新定义为基于证据的概率裁决；将Wölfflin式形式主义和ICONCLASS符号系统前置为可查询、可反驳的计算对象；采用四阶段（调查、佐证、反驳、裁决）ReAct式推理与提示隔离的LLM批评者相结合；并在新平衡的WikiArt Influence Benchmark‑100（WIB‑100）上验证其效果。

**🔧 技术方法**

技术手段包括：CLIP视觉编码 + SBERT文本编码；FAISS-HNSW近似最近邻候选生成；ReAct式LLM推理（Claude Opus 4.5）；Wölfflin风格投影与ICONCLASS概念检索工具；提示隔离的LLM批评者用于对抗性反驳；Neo4j知识图谱记录最终判决和证据。

**📊 数据集**

使用的主要数据集是重构后的WIB‑100（100位艺术家、2000对有向配对，正负各1000对），并利用WikiArt图像与维基百科生平文本做多模态输入。

**📈 对比分析**

与GalleryGPT、ComplEx、CLIP‑Art、Siamese Art等9个基线及Always‑YES进行对比，取得正类F1 83.7%，MCC 0.666，ROC‑AUC 0.910，显著优于所有对比模型，尤其在硬负样本和时间不可能样本的拒绝率上领先。

**⚠️ 局限性**

局限性包括：评估仅在WikiArt 100位艺术家的平衡子集上，缺乏对更广泛非西方艺术传统的覆盖；依赖完整的图像与文本信息，缺少对单模态或缺失模态场景的鲁棒性研究；LLM推理成本高、延迟大，需进一步压缩模型或优化推理流程；最终判决仍需人工专家验证，不能直接替代人类艺术史研究。

---

## 77. FedUTR: Federated Recommendation with Augmented Universal Textual Representation for Sparse Interaction Scenarios

**arXiv ID:** 2604.07351 | [PDF](https://arxiv.org/pdf/2604.07351v1)

**作者:** Kang Fu `[一作]` (Beijing Jiaotong University), Yidong Li `[通讯]` (Beijing Jiaotong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FedUTR 方案，在联邦推荐中加入通用文本表示，增强稀疏交互场景下的推荐效果。

**💡 创新点**

创新点：①利用基础模型（BERT）提取全局文本特征作为通用物品表示；②设计协同信息融合模块（CIFM）将通用表示与本地交互知识融合；③引入本地适配模块（LAM）动态保留用户个性化偏好；④提出稀疏感知残差变体 FedUTR‑SAR，根据本地交互稀疏度自适应平衡通用与个性化信息。

**🔧 技术方法**

技术手段：联邦学习（FedAvg）、BERT 预训练模型、MLP + 残差 + LayerNorm、L1 正则化、稀疏感知 ResNet、理论收敛分析。

**📊 数据集**

数据集：四个公开稀疏数据集——KU、Food、Dance、Movie，均覆盖 99% 以上的交互稀疏率。

**📈 对比分析**

与多种联邦与集中式基线（FCF、FedAvg、FedNCF、FedAtt、PFedRec、FedMR 等）对比，FedUTR 在 HR@10 与 NDCG@10 上相对最强基线提升 59.7%、47.4%、23.9% 与 29.3%（KU、Food、Dance、Movie），且在参数量上仅比 FedMR 少 60% 以上。

**⚠️ 局限性**

局限性：FedUTR‑SAR 需要额外的稀疏感知模块，训练时计算量增加；通用文本特征需在服务器端使用大模型，客户端仅接收轻量化表示；在极端稀疏或文本质量低下的场景下，文本辅助效果可能有限。

---

## 78. From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation

**arXiv ID:** 2604.07667 | [PDF](https://arxiv.org/pdf/2604.07667v1)

**作者:** Mengdie Flora Wang `[一作]` (AWS Generative AI Innovation Center), Jae Oh Woo `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种Conformal Social Choice后置层，将多代理辩论的概率输出转化为具有边际覆盖保证的行动/升级决策。

**💡 创新点**

创新点在于将分割共形预测引入多代理辩论，提供可解释的拒绝机制，避免错误一致导致的自动化失误。

**🔧 技术方法**

使用线性意见池聚合、分割共形预测、分层行动策略以及多轮LLM辩论作为技术手段。

**📊 数据集**

使用MMLU-Pro 10选项版的八个专业领域数据集进行评估。

**📈 对比分析**

与传统共识停止对比，α=0.05时能拦截81.9%错误共识，单例准确率提升至90–96.8%，但自动化比例相应降低。

**⚠️ 局限性**

局限包括仅提供边际覆盖保证、需满足可交换性假设、对开放生成和分布漂移适应有限、计算成本较高。

---

## 79. ReAlign: Optimizing the Visual Document Retriever with Reasoning-Guided Fine-Grained Alignment

**arXiv ID:** 2604.07419 | [PDF](https://arxiv.org/pdf/2604.07419v1)

**作者:** Hao Yang `[一作]` (Northeastern University), Ge Yu `[通讯]` (Northeastern University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ReAlign 方法，利用 Vision‑Language Model 的推理能力定位查询相关区域，并生成细粒度的视觉描述作为监督，优化视觉文档检索模型。

**💡 创新点**

创新点在于将推理生成的区域级描述作为对齐信号，通过 KL 正则化实现查询与视觉文档的细粒度语义对齐，显著提升检索效果。

**🔧 技术方法**

结合高性能 VLM（如 Qwen2.5‑VL‑72B‑Instruct）进行区域定位与描述生成，并在 InfoNCE 对齐基础上加入 KL 对齐损失进行联合训练。

**📊 数据集**

训练集由 DocVQA、InfoVQA、VisualMRC、OpenWikiTable、DUDE、MHDocVQA 等任务构成，评估覆盖 DocVQA、InfoVQA、ChartQA、SlideVQA、PlotQA、ArXivQA 六个视觉文档检索基准。

**📈 对比分析**

在 OCR 基线、CLIP、SigLIP、VDocRetriever 等多种对手上进行对比，平均提升 NDCG@10 约 2%（最高 2% 以上），在所有基准上均显著优于传统方法。

**⚠️ 局限性**

局限在于对大型 VLM 推理的依赖导致推理与描述生成成本高；在极端布局或极少可定位区域的文档中效果仍可能受限。

---

## 80. SubSearch: Intermediate Rewards for Unsupervised Guided Reasoning in Complex Retrieval

**arXiv ID:** 2604.07415 | [PDF](https://arxiv.org/pdf/2604.07415v1)

**作者:** Roxana Petcu `[一作]` (University of Amsterdam), Maarten de Rijke `[通讯]` (University of Amsterdam)

**通讯引用:** 28830 | [OpenAlex ID](https://openalex.org/A5031439294)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SubSearch框架，利用中间奖励信号指导搜索式LLM进行分解式推理；

**💡 创新点**

创新点在于使用内部产生的、无监督的中间奖励（子查询答案性与分解质量），取代传统需要人工或外部判别器的过程奖励模型；

**🔧 技术方法**

采用强化学习（GRPO）、动态搜索交互模板、语义覆盖与分裂性奖励、格式奖励、以及自适应残差奖励聚合等技术；

**📊 数据集**

在七个基准上进行评估，包含开放域问答（Natural Questions、TriviaQA、PopQA）和多跳推理问答（HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）；

**📈 对比分析**

与直接推理、CoT、RAG、以及多种RL‑基搜索代理（Search‑R1、ZeroSearch、R‑Search、StepSearch、InForage、O²‑Searcher）比较，SubSearch在大多数数据集上取得最高的EM/准确率，显著优于其他无监督RL方法；

**⚠️ 局限性**

局限性包括对检索器质量的依赖、计算成本提升、以及对内生奖励计算效率的需求，未来需探索联合优化检索器和生成器、进一步降低中间奖励计算成本。

---

## 81. Breaking the Illusion of Identity in LLM Tooling

**arXiv ID:** 2604.07398 | [PDF](https://arxiv.org/pdf/2604.07398v1)

**作者:** Marek Miller `[一作]` (University of Copenhagen), Marek Miller `[通讯]` (University of Copenhagen)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5025974860)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一套七规则的“voice model”，在LLM工具输出中消除人性化叙述，显著降低anthropomorphic markers；

**💡 创新点**

提出可配置、无模型改动的输出侧约束，系统性抑制人性化语言，实现机器化输出；

**🔧 技术方法**

利用配置文件系统提示、正则模式检测和改进版AnthroScore度量，对Claude Sonnet 4进行两轮对话评估；

**📊 数据集**

使用30个软件开发任务样本（错误诊断、代码评审等），13次复制，形成780个两轮对话共1560次API调用；

**📈 对比分析**

与默认输出对照，采用配对Wilcoxon检验，anthropomorphic markers从1233降至33（>97%减少），输出词数缩短49%，AnthroScore显著下降（p < 0.001）；

**⚠️ 局限性**

仅在单一模型和两轮对话中验证，未评估输出质量；规则基于经验，可能遗漏其他人性化机制；约束具有概率性，无法完全保证合规。

---

## 82. Tensor-based computation of the Koopman generator via operator logarithm

**arXiv ID:** 2604.07685 | [PDF](https://arxiv.org/pdf/2604.07685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 83. Cross-Tokenizer LLM Distillation through a Byte-Level Interface

**arXiv ID:** 2604.07466 | [PDF](https://arxiv.org/pdf/2604.07466v1)

**作者:** Avyav Kumar Singh `[一作]` (King's College), Davide Buffelli `[通讯]` (MediaTek)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种通过字节级接口实现的跨分词器知识蒸馏方法（Byte-Level Distillation, BLD），能够在教师和学生使用不同分词器的情况下直接传递知识。

**💡 创新点**

创新点在于将教师的分词级输出概率映射到统一的字节空间，并在学生端添加轻量级字节解码头，消除了传统方法中复杂的词表对齐和启发式近似步骤。

**🔧 技术方法**

主要技术包括：字节级概率近似（快速算法）、字节级解码头（线性投影或轻量级变换器）、LoRA 微调、Fast Vocabulary Transfer（FVT）等；损失函数融合了字节级交叉熵、KL 散度和词级交叉熵。

**📊 数据集**

使用的数据集包括：Tulu‑3 SFT 训练集、OpenMathInstruct‑2 训练集，以及 Tulu‑3 no‑robots、OpenMathInstruct‑2 随机 1,000 条样本作为验证集。

**📈 对比分析**

与现有方法（SFT、DSKD、MinED、ALM+SFT 等）对比，BLD 在 BPE‑BPE、BPE‑字节和跨模型蒸馏实验中表现相当或优于多数方法，尤其在 PiQA、AGI‑ZH、GSM‑8K 上取得最高分；但在 IFEval、MATH 等任务上不如 MinED 或 ALM+SFT，说明表现并不一致。

**⚠️ 局限性**

局限性包括：实验规模仅限 3B/8B 参数模型，未验证更大规模；使用 LoRA 限制了参数更新范围，完整参数优化可能提升性能；字节级解码头仅覆盖前 10 字节，对长 token 的监督不足；总体上跨分词器蒸馏仍缺乏统一、稳定的解决方案。

---

## 84. From Uncertainty to Possibility: Early Computing Experiences for Rural Girls

**arXiv ID:** 2604.07638 | [PDF](https://arxiv.org/pdf/2604.07638v1)

**作者:** Poornima Meegammana `[一作]` (University of Auckland), Kunal Gupta `[通讯]` (University of Auckland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在斯里兰卡农村地区实施为期八周的女性导向编程课程，提升女孩的编程自我效能和技术职业兴趣。

**💡 创新点**

结合当地语言的块式编程工具Ganidu与Scratch，采用从无接触到编程的渐进式路径和性别响应式教学。

**🔧 技术方法**

使用块式编程环境Ganidu（基于Google Blockly）和Scratch，辅以离线/线下指导、家长会议与教师培训。

**📊 数据集**

收集了158名10-14岁女孩的预后与后测问卷、职业兴趣问卷以及10名学生与5名教师的访谈文本。

**📈 对比分析**

采用Wilcoxon符号秩检验和McNemar检验，结果显示自我效能显著提升（p=0.003），职业兴趣向技术方向转移（p=0.015）。

**⚠️ 局限性**

缺乏对照组，数据为自我报告，受访者可能受社交期望影响，且课程时长有限，部分编程词汇仍使用英语。

---

## 85. On Formally Undecidable Propositions of Nondeterministic Complexity and Related Classes

**arXiv ID:** 2604.07406 | [PDF](https://arxiv.org/pdf/2604.07406v1)

**作者:** Martin Kolář `[一作]` `[通讯]`, Martin Kolář

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了将NP定义为每个语言都有多项式时间检验关系与有限证书长度的形式，实则包含了无法满足的Hilbert程序式要求，从而导致该定义在包含足够强的形式系统时与哥德尔不完备性冲突，证明该定义不可满足。

**💡 创新点**

揭示了NP类的传统定义内部存在的结构性缺陷：它隐式要求所有可多项式检验的语言同时满足完整性、可判定性与可验证性三元组，而在包含自身证明谓词的强系统时这三者不可共存。

**🔧 技术方法**

使用了形式化逻辑（Lean4）证明、哥德尔不完备性、证明复杂性（Cook–Reckhow框架）、可多项式证明系统的概念以及固定点构造。

**📊 数据集**

无实验数据集，研究完全基于理论证明。

**📈 对比分析**

由于本文为纯理论分析，没有实验比较；性能讨论被转化为证明长度与多项式界的关系，表明在强系统中无法获得全局多项式上界。

**⚠️ 局限性**

局限在于无法证明所有形式系统的具体证明长度界限，只能说明不存在统一满足所有系统的多项式上界；此外，关于Proof‑of‑Proof多项式性正式化仍需进一步细化。

---

## 86. Formally Guaranteed Control Adaptation for ODD-Resilient Autonomous Systems

**arXiv ID:** 2604.07414 | [PDF](https://arxiv.org/pdf/2604.07414v1)

**作者:** Gricel Vázquez `[一作]` (University of York), Simos Gerasimou `[通讯]` (Cyprus University of Technology)

**通讯引用:** 1065 | [OpenAlex ID](https://openalex.org/A5055440630)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种名为 SAVE 的基于情境的自适应框架，能够在运行时识别并处理超出预定义操作设计域（ODD）的情况，并动态调整控制器以保证安全性。

**💡 创新点**

创新点在于将 ODD 解析为离散情境网格（SCG），用概率模型（DTMC）刻画情境转移，结合概率模型检测（PMC）自动评估关键性并在出现违规时实时重构控制器，从而实现正式化的量化安全保证。

**🔧 技术方法**

使用的技术包括情境覆盖网格（SCG）构造、离散时间马尔可夫链（DTMC）、概率计算树逻辑（PCTL）、概率模型检查（PMC）与 PRISM 工具、MAPE‑K 监控回路以及数据驱动的转移概率更新。

**📊 数据集**

实验数据来自于一个简化的海事案例研究：一艘海上自主表面船（MASS）在两种类型船舶（A、B）间航行，采用离散化的 ODD（密度、类型、最短时间冲突）以及模拟生成的转移概率矩阵。

**📈 对比分析**

与固定控制器基线相比，SAVE 在 20 个随机配置中将违规率从 14/20 降至 6/20（即 70% 的违规被消除），并在运行时成功阻止了多次碰撞；实验还展示了在不同情境数量下的可扩展性，状态规模线性增长、转移数呈指数增长，但在当前规模下仍保持可接受的实时验证时间。

**⚠️ 局限性**

局限性包括：① 关键性评分（PMC）是最耗时的步骤，随着情境数目增大会出现指数级的状态与转移爆炸；② 当前评估仅限于海事领域的简化模型，尚未验证在更复杂或多模态系统中的泛化能力；③ 需要事先已知并手工定义的 ODD 与情境标签，若 ODD 本身不完整或不准，可能导致误判；④ 适应过程中对关键情境的屏蔽可能削弱系统的功能性或响应速度。

---

## 87. Bridging Natural Language and Interactive What-If Interfaces via LLM-Generated Declarative Specification

**arXiv ID:** 2604.07652 | [PDF](https://arxiv.org/pdf/2604.07652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 88. Assessing the Feasibility of a Video-Based Conversational Chatbot Survey for Measuring Perceived Cycling Safety: A Pilot Study in New York City

**arXiv ID:** 2604.07375 | [PDF](https://arxiv.org/pdf/2604.07375v1)

**作者:** Feiyang Ren `[一作]` (New York University), Takahiro Yabe `[通讯]` (New York University)

**通讯引用:** 1986 | [OpenAlex ID](https://openalex.org/A5075756309)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过结合视频测评与生成式 AI 聊天机器人，开发并试验了一种可在短视频情境中收集骑行者对城市骑行安全感知及其原因的互动式问卷方法。

**💡 创新点**

创新点在于将第一视角骑行视频与对话式 AI 结合，利用 LLM 的对话管理和规则控制实现结构化访谈，并在同一平台上同步生成自然语言的安全评估与建议；此外，还将自然语言处理、聚类和混合效应回归融入数据分析流程，首次实现对话式安全感知数据的定量分析。

**🔧 技术方法**

使用了 Gemini（Google LLM）实现对话生成，KeyBERT+sentence‑transformer 对文本做关键词抽取和嵌入，K‑means 聚类用于语义分组，混合效应逻辑回归评估环境特征与感知安全的关联。

**📊 数据集**

数据集为16名纽约市骑行者在9段不同道路（保护道、常规道、共享道、无道）中观看的视频后完成的对话记录，共144条安全评估；同时使用三位专家对同段道路进行手工评估的环境特征评分作为基准。

**📈 对比分析**

通过用户体验量表（UEQ‑S）和聊天机器人可用性量表（CUQ）评估方法可行性，平均得分分别为5.00/7（易用性）和3.47/5（聊天机器人友好度）；在数据可行性上，KeyBERT+聚类成功将自然语言归纳为与环境特征高度对应的语义主题；混合效应回归中，绿化、车流量、行人等特征显著影响安全判断，说明方法能够捕捉可解释的因果关系。

**⚠️ 局限性**

主要限制包括样本量小（仅16名完成全部9段视频的参与者）、高流失率导致数据偏倚、缺乏季节/天气等情境变量、以及对话内容温度不足导致用户感知“过于机器人”，未来需采用分布式抽样、增强对话温度、扩大样本及结合客观视频感知指标进一步验证。

---

## 89. A Physical Agentic Loop for Language-Guided Grasping with Execution-State Monitoring

**arXiv ID:** 2604.07395 | [PDF](https://arxiv.org/pdf/2604.07395v1)

**作者:** Wenze Wang `[一作]` (Adelaide University), Feras Dayoub `[通讯]` (Adelaide University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不重新训练抓取模型的前提下，给现有的视觉-力学抓取原语（ForceSight）添加了 Watchdog 监测层和有界恢复策略，形成一个闭环的物理 agentic loop，以语言指令驱动抓取任务并在执行失败时自动重试或请求澄清。

**💡 创新点**

创新点在于把物理抓取视为工具调用，提供统一的执行状态事件流（watchdog 产生的离散标签），并通过有限次数的重试与用户交互来保证安全与可解释性，从而在语言驱动抓取中实现可观测、可恢复的闭环控制。

**🔧 技术方法**

主要技术包括：1）TinyLlama 1.1B 用于自然语言指令解析；2）Watchdog 通过抓手力度、闭合度和微升动量等低延迟传感器数据做时间稳定的离散化标签；3）有限重试策略与事件驱动决策模块；4）RGB‑D 视觉-力学原语 ForceSight 与基于视觉的目标筛选。

**📊 数据集**

实验使用真实机器人 Stretch 搭载 RealSense D405 RGB‑D 摄像头，构造了多种场景（单一目标、颜色/空间歧义、干扰物、域迁移、故意空抓等）进行对比测试；没有使用公开的抓取数据集，而是自制的 290+ 次实地试验数据。

**📈 对比分析**

与单次开放式抓取基线相比，方法在单目标情形从 80% 提升到 100%，在颜色/空间歧义场景从 40% 提升至 80%，在干扰物场景从 0% 提升至 100%，并且大部分恢复仅需要一次重试；总体运行时仅略有提升（≈ 15–16 s）。

**⚠️ 局限性**

局限包括：①恢复策略过于保守，最多只能重试一次；②Watchdog 只区分空抓与非空抓，无法细粒度处理滑落、弱抓等更复杂失败；③语义验证依赖于上游感知，若感知失效仍可能导致误抓；④未针对更复杂环境或多目标情况提出专门的重试或重定位策略。

---

## 90. Vision-Language Navigation for Aerial Robots: Towards the Era of Large Language Models

**arXiv ID:** 2604.07705 | [PDF](https://arxiv.org/pdf/2604.07705v1)

**作者:** Xingyu Xia `[一作]` (Defense Innovation Institute, Chinese Academy of Military Sciences), Wen Yao `[通讯]` (Defense Innovation Institute, Chinese Academy of Military Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对无人机视觉与语言导航（Aerial VLN）领域进行全面综述，提出统一的技术分类，评估现有数据集、模拟平台和评估指标，并系统比较不同方法在主流基准上的表现。

**💡 创新点**

首次整合大型语言模型（LLM）与视觉语言模型（VLM）在 Aerial VLN 中的应用，并通过对比分析揭示不同动作空间、层次结构与训练范式的优势与不足，提出七个关键开放问题与发展路线。

**🔧 技术方法**

采用文献回顾、结构化分类、实验结果汇总、指标对齐与实证对比分析等方法，对现有技术进行系统化评估和归纳。

**📊 数据集**

引用并评估的主要数据集包括 AerialVLN、CityNav、OpenFly、OpenUAV、AVDN、AeroVerse 等；并对其规模、视角、动作空间等属性进行对比讨论。

**📈 对比分析**

通过在 AerialVLN‑S、AVDN、OpenUAV、CityNav 等四个成熟基准上对比，发现引入 LLM/VLM 与层次化控制后成功率与路径效率显著提升，且连续 6‑DoF 动作空间在实际飞行中更具实用性。

**⚠️ 局限性**

局限性主要在于：评估平台缺乏统一接口；大多数方法仍以模拟数据为主，真实世界实验不足；评估指标偏重成功率，缺乏对 LLM 推理质量、能耗与安全性的量化；以及多模态对齐在视角变化大时仍表现不稳。

---

## 91. Self-Calibrating LLM-Based Analog Circuit Sizing with Interpretable Design Equations

**arXiv ID:** 2604.07387 | [PDF](https://arxiv.org/pdf/2604.07387v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 92. Triage: Routing Software Engineering Tasks to Cost-Effective LLM Tiers via Code Quality Signals

**arXiv ID:** 2604.07494 | [PDF](https://arxiv.org/pdf/2604.07494v1)

**作者:** Lech Madeyski `[一作]` `[通讯]` (Wroclaw University of Science and Technology), Lech Madeyski (Wroclaw University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Triage 框架，利用代码健康度指标在代码生成前为每个软件工程任务路由至合适的 LLM 资源层级，从而实现成本与质量的平衡。

**💡 创新点**

创新点在于：①将代码健康度从事后诊断转化为预先决策的路由信号；②实现任务级别的多层级模型路由；③构建可抽象、可复现且可检验的评估协议。

**🔧 技术方法**

使用预先计算的代码健康子指标（25+ 子因子）、启发式阈值、机器学习分类器和完美回溯 oracle；成本模型分析、SHAP 解释以及多轮非确定性实验。

**📊 数据集**

主要数据集为 SWE-bench Lite（300 条真实 GitHub issue 任务），并结合内部代码健康度指标（1–10 复合分数及其子因子）。

**📈 对比分析**

对比三种路由策略（阈值、ML、oracle）与基线（始终轻、始终重、随机），评估指标包括任务成功率、每成功任务成本、路由准确率、过度/不足路由率；实验表明合适的阈值/ML 策略可在保持高通过率的前提下显著降低成本，且满足成本门限与信号门限（p̂≥0.56）。

**⚠️ 局限性**

局限性：依赖代码健康度指标的有效性；部署时需要准确识别目标文件；代码健康度与任务难度可能混淆；核心指标为专有度；未解决子任务级路由与多步协同问题。

---

## 93. Regret-Aware Policy Optimization: Environment-Level Memory for Replay Suppression under Delayed Harm

**arXiv ID:** 2604.07428 | [PDF](https://arxiv.org/pdf/2604.07428v1)

**作者:** Prakul Sunil Hiremath `[一作]` `[通讯]` (Visvesvaraya Technological University), Prakul Sunil Hiremath (Visvesvaraya Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对推荐、网络等平台媒介系统中存在的“回放”问题，本文设计了Replay Suppression Diagnostic (RSD) 与 Regret-Aware Policy Optimization (RAPO) 两个框架，系统性评估并抑制延迟危害的再现

**💡 创新点**

核心创新在于：①将平台层的可变转移核作为干预手段，引入持久的危害记忆（G）与不可逆痕迹（H）来重塑转移概率；②通过RSD在冻结策略、重置可观测状态下验证转移层变形能否消除回放，证明仅靠策略侧记忆无法抑制回放；③提出概率保守的、质量不丢失的转移变形方法，并给出可观测转移抑制的理论保证与概率收缩机制

**🔧 技术方法**

技术手段包括：基于PPO的策略学习，Lagrangian约束与对偶更新，持久转移重加权（ψ函数）实现转移变形；延迟信用分配与区块映射ρ实现危害归因；RSD实验流程（曝光–衰减–回放）以及多种指标（RAG、AUC-R、SM-R、ReplayRet、OddsRatio、ASD）

**📊 数据集**

主要实验数据集为合成有向图扩散模型（节点数 50–1000），其中指定敏感子图作为危害区；模拟推荐场景下的内容传播与延迟危害信号；此外对比了 Shield、SS、DR 等安全RL基线

**📈 对比分析**

在策略冻结的RSD评估中，RAPO 在 250 节点图上将 RAG 由 0.98 降低到 0.33（约 67% 降幅），同时保留 82% 任务回报；相比之下 Shield-UM、PM-ST、PM-RNN 等基线的 RAG 均接近 1 或略低，且回报更低；通过“deformation-off-at-replay”对照实验验证了转移变形是抑制回放的关键原因

**⚠️ 局限性**

局限性包括：需要平台能够介入转移核；危害归因与区域划分的误差可能导致误伤；痕迹不可逆可能在分布漂移后产生过度抑制；在高维连续空间中可观测状态与区域映射的离散化与计算复杂度仍需进一步研究

---

## 94. Contextual Earnings-22: A Speech Recognition Benchmark with Custom Vocabulary in the Wild

**arXiv ID:** 2604.07354 | [PDF](https://arxiv.org/pdf/2604.07354v1)

**作者:** Berkin Durmus `[一作]` (Argmax, Inc.), Atila Orhon `[通讯]` (Argmax, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建公开的 Contextual Earnings-22 数据集，并基准测试六种语音转文字系统在无上下文、本地上下文和全局上下文三种条件下的性能；同时给出关键词提示（prompting）和关键词提升（boosting）两大范式的强基线。

**💡 创新点**

提供了第一套标准化、可复现的短片段（15秒）与真实自定义词表（人名、公司名、产品名）配对的基准；区分理想的本地上下文与真实部署中的全局带干扰词列表；同时通过同时报告 WER 与关键词专属指标，揭示不同系统在精度与鲁棒性上的细微差异。

**🔧 技术方法**

关键词提示：在商业 API（Deepgram、OpenAI Whisper、AssemblyAI）中通过 prompt 参数实现；关键词提升：使用 CTC‑WS 结合 Parakeet + CTC 的多模态推理；实体抽取使用 GPT‑5 LLM；对齐采用 wav2vec forced‑alignment；评估工具为公开的 evaluation harness。

**📊 数据集**

基于 Earnings‑22 公开的财报电话会议音频，提取 15 秒短片段，手工校正后与 GPT‑5 自动抽取的实体（人、公司、产品）构成词表；覆盖 55 个音频文件、760 条样本，包含本地与全局词表两种配置。

**📈 对比分析**

方法：对六个系统在三种上下文条件下同时计算 WER 与关键词 precision/recall/F‑score；结果显示：提供上下文显著提升关键词 F‑score（尤其在本地上下文），但 WER 变化不一；全局上下文下精度下降，提示干扰词对系统鲁棒性的影响；关键词提升方法在全局干扰场景中往往优于提示方法。

**⚠️ 局限性**

局限：关键词列表长度与干扰词对精度有显著影响，prompting 可能引入 hallucinations；仅在英语财报通话场景验证，跨语言、多样语境的泛化仍待研究；手工校对成本高，扩展到更大规模时可能成为瓶颈。

---

## 95. Hybrid CNN-Transformer Architecture for Arabic Speech Emotion Recognition

**arXiv ID:** 2604.07357 | [PDF](https://arxiv.org/pdf/2604.07357v1)

**作者:** Youcef Soufiane Gheffari `[一作]` (USTO-MB), Samiya Silarbi `[通讯]` (USTO-MB)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一种混合CNN–Transformer架构，用于阿拉伯语语音情感识别，输入为Mel频谱特征。

**💡 创新点**

创新点在于将CNN提取局部频谱特征与Transformer捕获长程时间依赖相结合，首次在EYASE数据集上实现了97.8%的识别准确率。

**🔧 技术方法**

使用的技术包括3层CNN+池化、4层Transformer Encoder（8头、d_model=256）、全局平均池化+全连接分类层、Adam优化器、cosine annealing学习率调度、dropout和批归一化。

**📊 数据集**

使用的数据库是EYASE（埃及阿拉伯语半自然情感语料），包含461条样本，四个情绪类别（愤怒、快乐、悲伤、中性）。

**📈 对比分析**

通过与传统SVM/MLP基线和单独CNN基线比较，模型在测试集上取得了97.8%的准确率和0.98的宏F1分数，明显优于70%~80%区间的基线性能。

**⚠️ 局限性**

局限性包括：数据集规模有限且情绪类别分布不平衡，导致幸福与中性容易混淆；仅覆盖单一埃及方言，缺乏跨方言和多模态的泛化能力。

---

## 96. Rhizome OS-1: Rhizome's Semi-Autonomous Operating System for Small Molecule Drug Discovery

**arXiv ID:** 2604.07512 | [PDF](https://arxiv.org/pdf/2604.07512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 97. CivBench: Progress-Based Evaluation for LLMs' Strategic Decision-Making in Civilization V

**arXiv ID:** 2604.07733 | [PDF](https://arxiv.org/pdf/2604.07733v1)

**作者:** John Chen `[一作]` (University of Arizona), Mingyi Lin `[通讯]` (University of Arizona)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CivBench基准，评估LLM策略师在《文明V》多玩家游戏中的长期决策能力，使用进度预测模型生成密集的胜率信号。

**💡 创新点**

首次将进度预测方法与LLM benchmark结合，提供三维有效性检验（预测、构造、收敛）并揭示LLM策略师的多维战略特征。

**🔧 技术方法**

使用Vox Deorum接入《文明V》，训练多种机器学习模型（LogReg、XGBoost、MLP、AttentionMLP）预测回合胜率，基于Bradley‑Terry 生成ELO。

**📊 数据集**

收集307场8人对战（共1674局LLM回合）以及194场VPAI自对局，包含各回合游戏状态与决策轨迹。

**📈 对比分析**

对比7种LLM策略师与VPAI基线，使用ELO排序与回合胜率预测的AUC/Brier/LogLoss，AttentionMLP在AUC上达0.92，LLM策略师在特定胜利类型中可与VPAI匹敌但整体未突破。

**⚠️ 局限性**

未覆盖跨局学习，模型对战争动态捕捉有限，接口透明度不足导致决策解释性受限，Bench仅在《文明V》环境验证，难以直接推广到现实场景。

---

## 98. Reasoning Graphs: Deterministic Agent Accuracy through Evidence-Centric Chain-of-Thought Feedback

**arXiv ID:** 2604.07595 | [PDF](https://arxiv.org/pdf/2604.07595v1)

**作者:** Matthew Penaroza `[一作]` `[通讯]`, Matthew Penaroza

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了推理图和检索图结构，让语言模型代理在多轮推理中保留每条证据的评估链，实现跨查询的证据级反馈，从而在保持模型权重不变的情况下提升准确率并降低方差。

**💡 创新点**

创新点在于将推理链作为可逆图边存储，并通过从证据向后遍历聚合历史评估，形成证据级反馈；同时引入检索图用于管线规划，形成自我改进的闭环。

**🔧 技术方法**

技术主要包括：基于属性图的推理图与检索图设计、逆向遍历以构建证据档案、上下文注入式证据级反馈、候选集滤波的管线规划、以及顺序集群评估协议。

**📊 数据集**

使用了 HotpotQA 和 MuSiQue 两个多跳问答数据集进行实验。

**📈 对比分析**

与传统 RAG、Reflexion、ReasoningBank 等基线对比，本文系统在准确率上呈现递增趋势、方差显著下降，并在证据级反馈上优于查询级检索，表现出明显的性能提升。

**⚠️ 局限性**

局限包括冷启动时无反馈、图存储随运行增大、对结果标签噪声敏感、只能识别完全相同的证据段落、以及需要外部结果信号来更新图。

---

## 99. Lexical Tone is Hard to Quantize: Probing Discrete Speech Units in Mandarin and Yorùbá

**arXiv ID:** 2604.07467 | [PDF](https://arxiv.org/pdf/2604.07467v1)

**作者:** Opeyemi Osakuade `[一作]`, Simon King `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提供了一份用于验证PS/PDF打印质量的测试文档，检查了字体、连字、数学符号、对齐和尺寸等方面的打印准确性。

**💡 创新点**

创新点在于将多种常见排版检查（字体倾斜、连字、数学符号、尺寸标尺、双面对齐等）集中在一份简洁的测试文件中，便于快速评估打印输出。

**🔧 技术方法**

使用了LaTeX的article.cls、Times Roman、Computer Modern以及Palatino/Palladio等基础字体，并通过纯文本与数学表达式嵌入来验证排版效果。

**📊 数据集**

该测试文件不使用公开数据集，而是采用自定义文本和符号组合进行评估。

**📈 对比分析**

本文并未与其他方法进行性能对比；其功能主要是通过手动检查打印输出的视觉效果来评估质量，缺乏客观的定量指标。

**⚠️ 局限性**

局限性包括：仅针对基础排版特性；不涵盖高级排版功能或跨平台兼容性；缺乏自动化检测脚本，无法实现大规模批量验证。

---

## 100. Jean-Raymond Abrial: A Scientific Biography of a Formal Methods Pioneer

**arXiv ID:** 2604.07353 | [PDF](https://arxiv.org/pdf/2604.07353v1)

**作者:** Jonathan P. Bowen `[一作]` (London South Bank University), Henri Habrias `[通讯]` (University of Nantes)

**关键词:** `aaff19cd-e89f-4398-8dae-a6684a329811` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文以学术传记的形式梳理了Jean‑Raymond Abrial的生平与科研贡献，重点回顾其在Z规范、B方法和Event‑B框架中的创新工作及其工业实践

**💡 创新点**

创新点在于将形式化规范、逐步细化（refinement）与证明（proof）统一到同一方法论中，并将其成功推广至实际安全关键系统

**🔧 技术方法**

主要技术包括基于有序集理论与一阶逻辑的Z规范语言、B方法的可细化化设计与证明框架，以及Event‑B的事件建模与细化机制；辅以Atelier B、Rodin等工具支持

**📊 数据集**

论文为传记性质，未使用具体实验数据集

**📈 对比分析**

由于是历史性综述，未进行实验或性能比较

**⚠️ 局限性**

局限性在于缺乏对方法实际效果的定量评估，仅从案例与理论层面论述；对方法在不同规模、不同工业领域的适用性仍需进一步验证

---

## 101. Trust the AI, Doubt Yourself: The Effect of Urgency on Self-Confidence in Human-AI Interaction

**arXiv ID:** 2604.07535 | [PDF](https://arxiv.org/pdf/2604.07535v1)

**作者:** Baran Shajari `[一作]` (McMaster University), Istvan David `[通讯]` (McMaster University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在AI主动提示下，人类用户的自信心与信任随时间和紧迫性如何变化，采用30名学生参与的Pac‑Man游戏实验，比较有时限与无时限的两种互动顺序并测量自信、信任等指标。

**💡 创新点**

首次系统性检验“紧迫性”对人类自信的负面影响，并揭示信任提升与自信下降的认知不对称；提出将自我效能视为软件质量属性并呼吁更渐进的AI上岗与培训。

**🔧 技术方法**

使用强化学习训练的Pac‑Man AI（Proximal Policy Optimization），混合式人机交互实验设计，Likert量表问卷收集信任与自信数据，统计分析采用卡方检验和Cramér’s V。

**📊 数据集**

实验数据来自30名计算机科学学生的游戏得分与问卷回答；AI模型训练使用Gymnasium Pac‑Man 环境，无公开数据集。

**📈 对比分析**

对比两种交互顺序和总样本，使用卡方检验评估自信差异，发现紧迫性导致显著自信下降（p=0.002，Cramér’s V=0.63）；信任提升约50%，但不受紧迫性影响；整体实验性能显示信任提升但自信下降的对比结果。

**⚠️ 局限性**

样本局限为计算机科学学生，外部效度受限；自信评估仅使用单一Likert尺度，缺乏标准化量表；实验仅为短期，未检验长期影响；仅使用Pac‑Man游戏，任务可能与真实工业情境差距较大。

---

## 102. Robust Multi-Agent Target Tracking in Intermittent Communication Environments via Analytical Belief Merging

**arXiv ID:** 2604.07575 | [PDF](https://arxiv.org/pdf/2604.07575v1)

**作者:** Mohamed Abdelnaby `[一作]` (Worcester Polytechnic Institute), Kevin Leahy `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于KL发散闭式解的分布式目标跟踪框架，并在GPS‑无效、通信受限环境下实现了高效、无量化误差的多机器人目标跟踪。

**💡 创新点**

创新点包括：①推导出正向KL的算术平均与反向KL的几何平均闭式解，消除数值求解的量化误差；②提出基于物理访问历史的访客权重KL融合方法，有效抑制传感器噪声；③用有限视角MPC替代MCTS，避免内存爆炸并实现可扩展的主动信息收集。

**🔧 技术方法**

主要技术手段包括离散贝叶斯更新、正向/反向Kullback–Leibler闭式优化、空间访客权重计算、基于信息增益的有限视角MPC，以及大规模并行格子仿真。

**📊 数据集**

实验使用自定义离散网格数据集，网格大小从10×10到100×100，机器人数为2或3，目标运动模式为静止、随机游走、逃逸和巡逻，覆盖多种通信间隔和传感器噪声配置，累计进行超过47,000次仿真。

**📈 对比分析**

与数值求解的正向KL、反向KL、算术平均、几何平均以及本文的访客权重KL进行比较，结果显示闭式算术平均和几何平均在成功率上优于数值解，而访客权重KL在所有配置中在搜索效率（最低发现步数）上占优，尤其在高噪声和长通信间隔下保持鲁棒性。

**⚠️ 局限性**

局限性包括：仅适用于离散格子模型，难以直接推广到连续空间；访客权重公式需要人工设计，对异构传感器或非对称运动约束的适用性待验证；MPC搜索深度有限，可能在极大状态空间下无法获得最优策略；未考虑动态环境或对抗性目标的主动干扰。

---

## 103. RL-ASL: A Dynamic Listening Optimization for TSCH Networks Using Reinforcement Learning

**arXiv ID:** 2604.07533 | [PDF](https://arxiv.org/pdf/2604.07533v1)

**作者:** F. Fernando Jurado-Lasso `[一作]`, J. F. Jurado `[通讯]` (Universidad Nacional de Colombia)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出RL-ASL，一种基于强化学习的TSCH网络自适应监听框架，动态决定何时激活或跳过接收时隙；

**💡 创新点**

创新点在于将离线训练的Q表直接嵌入低功耗节点，实现无在线学习的自适应监听；

**🔧 技术方法**

采用离线多场景Q‑学习、混合基数状态编码、期望奖励调度、Contiki‑NG实现与FIT IoT‑LAB实验；

**📊 数据集**

使用FIT IoT‑LAB M3实验平台和Cooja模拟器生成的多种交通模式（高、异构、稀疏、周期性）网络数据；

**📈 对比分析**

与Orchestra、Link‑based Orchestra、PRIL‑M等基准协议对比，RL‑ASL在所有流量下能降低46%能耗、保持≈100%PDR，并将平均延迟相较PRIL‑M降低约96%；

**⚠️ 局限性**

局限在于只能处理静态或低移动性网络，无法在频繁拓扑变化或高速移动环境下维持高PDR，且依赖离线训练场景的泛化能力。

---

## 104. Emotion Concepts and their Function in a Large Language Model

**arXiv ID:** 2604.07729 | [PDF](https://arxiv.org/pdf/2604.07729v1)

**作者:** Nicholas Sofroniew `[一作]` (Anthropic), Jack Lindsey `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLMs）在对话中表现出的情感反应，探讨了其内部情感概念的表示及其对模型行为的影响。

**💡 创新点**

提出了LLMs表现出功能性情感的概念，即模型在情感影响下的表达和行为模式，这些模式是通过情感概念的抽象表示介导的。

**🔧 技术方法**

使用了Claude Sonnet 4.5模型，结合了合成数据集和自然对话数据，分析了情感向量的激活及其对模型输出的因果影响。

**📊 数据集**

使用了合成数据集（包含情感故事）和自然对话数据集（如Common Corpus和LMSYS Chat 1M）进行实验。

**📈 对比分析**

通过与其他方法的比较，发现情感向量的激活与模型的偏好和行为有显著的因果关系，情感向量的激活能够预测模型的输出和偏好。

**⚠️ 局限性**

模型的情感表示可能与人类情感的工作方式不同，且未能找到持久的情感状态表示，可能存在对情感概念的非线性或隐式表示。

---

## 105. Too long; didn't solve

**arXiv ID:** 2604.07593 | [PDF](https://arxiv.org/pdf/2604.07593v1)

**作者:** Lucía M. Cabrera `[一作]` (Instituto Balseiro), Isaac Saxton-Knight `[通讯]` (Poindexter Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对专业专家手工创作的607道数学题目，研究了提示长度和解题步骤长度这两个结构性变量对大语言模型（LLM）失败率和模型间差异的影响。

**💡 创新点**

创新点在于首次在对抗性数学基准上引入连续结构特征（词数）进行分析，并使用层级混合效应模型揭示了不同模型对长度敏感性的共同性与差异性。

**🔧 技术方法**

主要技术手段包括Spearman相关分析、方差归一化的跨模型不一致度度量，以及对数变换后长度的混合效应回归模型。

**📊 数据集**

使用的数据集是由硕士、博士、教授和IMO奖牌获得者构建的607道原创竞赛风格数学题，包含每道题的完整文字描述与作者提供的步骤解答。

**📈 对比分析**

比较方法是计算每题每模型的失败分数（五次尝试中失败次数/5），并与提示长度/解答长度相关联；结果显示两者均正相关导致失败率上升，而归一化后与模型间差异的负相关较弱，说明长度对所有模型的影响相似。

**⚠️ 局限性**

主要局限包括：失败分数离散且受试验次数限制，方差分析受均值约束，模型数量有限，未使用更合适的二项式混合模型，且仅在单一对抗性基准上验证，缺乏跨基准和跨模型的稳健性检验。

---

## 106. Bird-Inspired Spatial Flapping Wing Mechanism via Coupled Linkages with Single Actuator

**arXiv ID:** 2604.07677 | [PDF](https://arxiv.org/pdf/2604.07677v1)

**作者:** Daniel Huczala `[一作]` (Seoul National University), Frank C. Park `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计了一个基于两个耦合Bennett四杆链的单动量鸟类啮齿翼机，单一电机驱动实现空间摆动，另一链通过被动状态切换实现翼展折叠。

**💡 创新点**

创新点在于将空间单环机制与被动折叠结合，利用简化的四边形构造和运动多项式因式分解，实现仅用一个驱动实现空间摆动与翼展折叠的协同运动。

**🔧 技术方法**

采用了双四元数运动多项式、Study参数、运动因式分解与Python库Rational Linkages等几何与运动学技术。

**📊 数据集**

未使用数据集；该工作主要基于几何设计与3D打印原型实验。

**📈 对比分析**

通过实验验证原型实现预期的空间轨迹和被动折叠，机身重量87g；尚未进行飞行性能或气动性能比较，未给出数值性能指标。

**⚠️ 局限性**

局限在于缺乏气动与动力学建模、未进行飞行试验、翼膜与气动形状尚未优化，且被动折叠机制对流动条件高度敏感。

---

## 107. Reinforcement Learning with Reward Machines for Sleep Control in Mobile Networks

**arXiv ID:** 2604.07411 | [PDF](https://arxiv.org/pdf/2604.07411v1)

**作者:** Kristina Levina `[一作]` (Linköping University), Jendrik Seipp `[通讯]` (Linköping University)

**通讯引用:** 596 | [OpenAlex ID](https://openalex.org/A5031089257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于强化学习与奖励机的射频单元睡眠控制方法，能够在保证长期QoS约束的前提下显著降低移动网络能耗。

**💡 创新点**

创新点在于将奖励机引入非马尔科夫奖励结构，利用其有限状态机记忆历史QoS违约信息，使RL代理能够考虑时间平均约束，并通过奖励机深度调节记忆容量以提升能效。

**🔧 技术方法**

使用深度强化学习（TD3）实现睡眠策略学习，结合奖励机（RM）进行非马尔科夫奖励建模；同时与Markovian奖励、Lagrangian方法进行对比；实验中使用基于地图的射线追踪模型模拟用户位置、通道状态和功耗。

**📊 数据集**

数据集来自仿真生成，包含随机用户位置、流量负载、通道状态以及功耗参数；未使用公开真实数据集，而是在仿真环境中对不同用户数量、通道条件和流量负载进行多次随机采样。

**📈 对比分析**

通过对比Markovian奖励、Lagrangian方法、浅层奖励机（L=10）和深层奖励机（L=100）的性能，实验结果显示深层奖励机实现了最高的能效，并且在约束满足方面最接近边界；其睡眠模式切换最频繁，表明策略更具适应性。

**⚠️ 局限性**

局限性包括：奖励机深度越大学习复杂度和样本需求显著提升；实验仅在仿真环境进行，缺乏真实网络部署验证；对RM深度和其他超参数的选择较为敏感，需进一步研究其泛化性。

---

## 108. Narrix: Remixing Narrative Strategies from Examples for Story Writing

**arXiv ID:** 2604.07643 | [PDF](https://arxiv.org/pdf/2604.07643v1)

**作者:** Chao Zhang `[一作]` (Cornell University), Eunyee Koh `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Narrix 系统，帮助初学写作者通过例子学习、识别、探索并重混叙事策略，将其可视化并与写作流程交互。

**💡 创新点**

创新点在于：①将隐性的叙事策略显式化并以可操控的策略单元呈现；②提供基于情感轨迹与转折点的交互式故事弧，支持高层目标驱动的策略检索；③引入多维轨道（角色、情节、语言等）以音轨式拖拽方式重混策略；④将 LLM 输出与可解释性提示相结合，形成认知学徒式的学习与创作支持。

**🔧 技术方法**

技术手段包括：①大语言模型（GPT‑4o/4.1）用于分块、策略识别、转折点分类与情感弧生成；②动态时间规整（DTW）计算弧相似度；③前端采用 Next.js、Vega‑Lite 与 BlockNote 实现交互式 UI；④JSON schema 约束输出，提升策略标注的可检索性和透明度。

**📊 数据集**

使用的数据集为：20 条公开领域的童话故事（如《灰姑娘》《小红帽》《匹配的匹配》）作为示例；此外利用 Tian 等人提供的故事块与转折点标注数据训练转折点分类模型。

**📈 对比分析**

通过在 12 名非母语英语写作者中进行双盲、交叉实验，对比 Narrix 与基线对话式写作工具。评估维度包括 SUS、NASA‑TLX、CSI、AI‑Experience、策略识别与应用数量、故事质量（Lamp‑P‑Writing‑Quality‑RM 模型）。结果显示 Narrix 在可用性、创意支持、AI 体验、策略保留、重混频率和故事质量（74.3% 通过模型判定）等方面显著优于基线。

**⚠️ 局限性**

局限性包括：①LLM 仍可能产生幻觉或误标，系统对错误的防护不完备；②主要聚焦块级策略，缺乏跨章节、全局连贯性检查；③样本规模有限且文化单一，缺乏跨语境的验证；④未对长期学习效果和不同写作体裁的迁移性进行评估。

---

## 109. Multi-Agent Orchestration for High-Throughput Materials Screening on a Leadership-Class System

**arXiv ID:** 2604.07681 | [PDF](https://arxiv.org/pdf/2604.07681v1)

**作者:** Thang Duc Pham `[一作]` (Argonne National Laboratory), Murat Keçeli `[通讯]` (Argonne National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了可扩展的层级多代理框架，利用大语言模型在领导级 HPC 系统上自动化执行高通量材料筛选工作。

**💡 创新点**

创新点在于设计了 planner‑executor 结构，将科学规划与并行执行分离；结合 Model Context Protocol 与 Parsl 的异步接口，实现 LLM 与工作流管理器的无缝协作，突破单代理序列调用瓶颈。

**🔧 技术方法**

使用的技术包括 LangGraph 构建代理逻辑、MCP 作为工具调用协议、Parsl 负责任务调度与容错、gpt‑oss‑120b 开源 LLM 进行推理，以及 gRASPA 进行 GCMC 仿真，全部部署在 Aurora 超算上。

**📊 数据集**

实验数据集为 CoRE MOF 2025（5,591 结构），用于大气水捕获的多点吸附仿真。

**📈 对比分析**

通过弱/强规模实验与传统手工编排工作流对比，显示近线性加速、强规模效率可达 64.9%，总任务完成率 84%，调度开销约 60–90 秒，证明系统具备高吞吐量与可扩展性。

**⚠️ 局限性**

主要限制包括 LLM 工具调用错误导致 16% 的失败率、不同 MOF 结构导致仿真时间波动大、以及对开源 LLM 可靠性与容错机制的进一步改进需求。

---

## 110. LLM-Generated Fault Scenarios for Evaluating Perception-Driven Lane Following in Autonomous Edge Systems

**arXiv ID:** 2604.07362 | [PDF](https://arxiv.org/pdf/2604.07362v1)

**作者:** Faezeh Pasandideh `[一作]` (Hamm-Lippstadt University of Applied Sciences), Achim Rettberg `[通讯]` (Hamm-Lippstadt University of Applied Sciences)

**通讯引用:** 661 | [OpenAlex ID](https://openalex.org/A5110191210)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了离线-在线分离的 AI 驱动边缘安全测试框架，通过 LLM 生成语义化故障场景，LDM 合成视觉降质，并将结果预先存入查找表供 Jetson 等边缘设备实时查询。

**💡 创新点**

创新点在于将大模型在离线生成高质量故障样本并预计算评估结果，避免边缘设备运行重模型，同时实现语义丰富的故障注入与实时安全评估。

**🔧 技术方法**

使用的大型语言模型 LLM（如 GPT‑OSS）、潜在扩散模型 LDM（Stable Diffusion 2.1）、CLIP 用于语义一致性验证，以及 ResNet‑18 作为车道跟踪基准。

**📊 数据集**

构建了 VisionFault‑350K 数据集，包含 350k 张合成的故障图像，覆盖低光、雨、雾等多类环境降质。

**📈 对比分析**

将 ResNet‑18 在正常数据和 LDM 注入的故障数据上评估，结果显示在严重雾等极端降质下 RMSE 提高近 99%，R² 降至 0.755，细粒度定位准确率降至 31%。

**⚠️ 局限性**

局限在于缺乏与随机故障注入的对比实验、对真实硬件缺陷的覆盖不足，以及对模型自适应训练和域自适应策略的探索有限。

---

## 111. ADAG: Automatically Describing Attribution Graphs

**arXiv ID:** 2604.07615 | [PDF](https://arxiv.org/pdf/2604.07615v1)

**作者:** Aryaman Arora `[一作]` (Stanford University), Sarah Schwettmann `[通讯]` (Transluce)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套端到端自动化的电路追踪与解释管道，能够自动生成归因图、归因配置、对特征进行多视角谱聚类形成超级节点，并利用LLM解释器和模拟器生成自然语言解释。

**💡 创新点**

核心创新包括：① 归因配置量化特征功能以捕捉非局部信息；② 结合多视角谱聚类与谐波平均的聚类算法以自动划分功能相似的超级节点；③ 基于解释-模拟器框架的全自动自然语言描述与评分流程。

**🔧 技术方法**

使用的技术包括梯度归因（RelP、LogitLens）、MLP神经元电路追踪、谱聚类与多核学习、LLM解释器与模拟器（Anthropic API）以及自定义的归因配置与聚类评价指标。

**📊 数据集**

实验使用的主要数据集有：Llama 3.1 8B Instruct 的多跳状态首都任务、医疗建议 jailbreak 变体、FineWeb 文档集以及 Qwen3 32B 的对照实验。

**📈 对比分析**

在多跳任务中，自动生成的超级节点与人工标注相匹配，LLM在输入归因上的解释与人类相当、输出贡献上明显优于人类；在医疗 jailbreak 任务中，聚类成功定位两组关键集群，将攻击成功率从 5.5% 提升至 90%；聚类质量上，谐波平均多视角谱聚类实现了更低的符号混合率、更高的轮廓系数和更均衡的簇大小。

**⚠️ 局限性**

主要限制包括：目前仅覆盖 MLP 神经元归因，未扩展到注意力或 QK 电路；聚类仍需手动设定簇数 k；LLM 解释器的效果受提示设计和训练数据影响；系统整体依赖梯度归因的精确性和模型可微性。

---

## 112. Beyond Single Reports: Evaluating Automated ATT&CK Technique Extraction in Multi-Report Campaign Settings

**arXiv ID:** 2604.07470 | [PDF](https://arxiv.org/pdf/2604.07470v1)

**作者:** Md Nazmul Haque `[一作]` (North Carolina State University), Laurie Williams `[通讯]` (North Carolina State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对29种自动化 ATT&CK 技术提取方法在多报告攻击情报（SolarWinds、XZ Utils、Log4j）场景下进行评估与对比，分析误分类、控制缺口、性能饱和等。

**💡 创新点**

提出多报告聚合评估框架、控制覆盖度指标、误分类语义相似度分析、性能饱和阈值及报告特征影响研究，弥补单报告评估的不足。

**🔧 技术方法**

使用 NER、Encoder‑based Classification 与 Decoder‑based LLM（含提示、RAG、SFT+RAG 等）技术，并利用文本嵌入、余弦相似度、t‑SNE 等方法进行误差分析。

**📊 数据集**

训练集采用 AnnoCTR，测试集采用 CTIfecta（90 篇报告，涵盖 SolarWinds、XZ Utils、Log4j），并扩展 ATT&CK → P‑SSCRM 控制映射。

**📈 对比分析**

通过宏平均精确率、召回率和 F1 进行比较；多报告聚合后 F1 提升约 26%，最高可达 78.6%；误分类约 33% 与语义相近技术相关；控制覆盖率最高为 77%，但仍存在 23% 缺口；Encoder 方法在多报告下降，NER 精度高但召回低，LLM 方案表现最优。

**⚠️ 局限性**

局限性包括仅覆盖三场攻击，数据集分布差异导致泛化受限；训练/测试集分离与技术长尾问题；控制映射存在主观性；LLM 生成误差与误分类仍难以完全消除。

---

## 113. Universal, sample-optimal algorithms for recovery of anisotropic functions from i.i.d. samples

**arXiv ID:** 2604.07660 | [PDF](https://arxiv.org/pdf/2604.07660v1)

**作者:** Ben Adcock `[一作]`, Avi Gupta `[通讯]` (Simon Fraser University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文构造了针对高维周期函数的通用（无先验异方性信息）逼近算法，能够在不同的各向异性 Sobolev 空间（包括支配混合光滑性空间）中实现近似最优或接近最优收敛速率，并给出了对应的下界证明其近似最优性。

**💡 创新点**

创新点在于：
• 通过把函数恢复转化为 Fourier 系数的稀疏恢复问题，并利用压缩感知（SR‑LASSO）实现无自适应通用逼近；
• 给出全局下界（adaptive m‑width）证明该算法的收敛率（上限）在多项式对数误差范围内是最优的；
• 证明任何线性通用逼近算法必然受到维度相关的对数“灾难”，从而阐明非线性算法在通用逼近中的必要性。

**🔧 技术方法**

核心技术包括：
• 最佳 s‑项逼近与 Stechkin 不等式的分析；
• Fourier 样本矩阵满足 ℓ²‑鲁棒零空间性质 (ℓ²‑rNSP) 与 RIP 的随机采样理论；
• SR‑LASSO（Square‑Root LASSO）求解稀疏回归；
• 广义宽度与 Kolmogorov 宽度的对偶性用于构造下界；
• 维度缩减与构造特定多项式测试函数以证明线性算法的局限。

**📊 数据集**

数据集：本文仅使用均匀分布在 d‑维环面 𝕋^d 上的点值采样，即 i.i.d. 采样点；并未使用外部实际数据集。

**📈 对比分析**

与线性方法比较：
• 通用非线性算法达到误差 ≲ (log^{p(α)-1}m / m)^{h(α)}（或 1/m^{g(β)}）；
• 任何线性通用算法在相同设置下至少需要误差 ≳ (log m)^{d-1} / m^{h(α)}，即多项式对数因子依赖维度；
• 对于 𝑑>4，非线性算法在收敛率上明显优于线性算法。

**⚠️ 局限性**

局限性：
• 误差上界中仍存在 O(log³ m · log log m) 的多项式对数因子，主要源自 Fourier 矩阵 RIP 的当前最优上界；
• 仅在 L² 误差范数下分析，推广到 Lᵖ（p≠2）及其他函数空间仍需进一步研究；
• 算法为非自适应，虽然不需学习异方性，但在实践中可能无法充分利用函数的具体光滑性结构；
• 本文未给出具体的计算复杂度或实现细节，实际应用时对大规模稀疏求解器的性能与可扩展性仍是开放问题。

---

## 114. GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents

**arXiv ID:** 2604.07429 | [PDF](https://arxiv.org/pdf/2604.07429v1)

**作者:** Mingyu Ouyang `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为GameWorld Benchmark的基准，涵盖34款浏览器游戏和170个任务，提供统一的动作接口、暂停式与实时评估、以及可验证的状态评估框架。

**💡 创新点**

创新点包括：①将Computer‑Use与Generalist Semantic Action Parsing两种接口统一到同一可验证动作空间；②采用状态可验证评估而非OCR/视觉判定；③提供可重复、可重现的评估流程，并对实时交互、记忆敏感性、动作有效性等维度进行系统分析。

**🔧 技术方法**

使用技术包括：浏览器沙箱与Playwright控制、Semantic Action Parsing、工具调用与Prompt Template、可验证的游戏状态API、重复实验设计、实时与暂停评估模式、记忆回溯实验、无效动作统计与分析。

**📊 数据集**

数据集：34款跨五大类别（Runner、Arcade、Platformer、Puzzle、Simulation）的浏览器游戏，包含170个精细设计的任务，每个任务配有目标、初始化状态、评价指标和可验证字段。

**📈 对比分析**

对18个模型-接口组合（13基础模型、8 CUA、10 Generalist）进行评估，采用Success Rate（SR）和Progress（PG）两指标。最佳总体PG≈41.9%（Gemini‑3‑Flash‑Preview），但远低于人类新手（SR 55.3%，PG 64.1%），表明当前模型仍难以达到人类水平。

**⚠️ 局限性**

局限性：任务指令与语义动作映射需人工完成，缺乏自动化；评估只关注状态，未覆盖感知噪声和视觉错误；实时评估难度大且对模型推理速度敏感；对长期记忆和复杂规划支持不足，导致性能瓶颈。

---

## 115. Learning is Forgetting: LLM Training As Lossy Compression

**arXiv ID:** 2604.07569 | [PDF](https://arxiv.org/pdf/2604.07569v1)

**作者:** Henry C. Conklin `[一作]` (Princeton University), Seraphina Goldfarb-Tarrant `[通讯]` (Cohere)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文把大语言模型的预训练视为一种有损压缩过程，利用信息瓶颈（Information Bottleneck）框架量化模型在训练中的表示信息，并通过信息平面追踪压缩与表达的变化。

**💡 创新点**

创新点在于：①首次将软熵估计器（soft‑entropy estimator）应用于LLM表示层，能够在大规模模型上高效估计互信息；②揭示不同规模、不同训练工艺的LLM在预训练过程中都趋向信息瓶颈边界，并证明压缩最优度（optimality）与下游任务性能呈显著相关；③引入“偏好信息”（preference information）互信息作为衡量模型与人类偏好对齐程度的指标，并证明其对性能的预测力。

**🔧 技术方法**

技术手段包括：信息瓶颈理论、率失真理论、软熵估计、n-gram 后退（back‑off）互信息估计、预训练检查点的逐步追踪、Spearman 相关分析。

**📊 数据集**

使用的数据集：C4（预训练数据，用于估计输入/输出互信息），Tulu（偏好对齐数据，用于计算偏好互信息），以及 6 个公开评测基准（MMLU Pro、BBH、Math LVL5、IFEval、GPQA、MuSR）进行性能评估。

**📈 对比分析**

对比方法：对多家公开权重 LLM（包括 OLMo2、Smol LM2、Pythia 等）进行信息平面定位，计算压缩最优度与下游性能的相关性。结果显示：①模型越接近信息瓶颈边界，表示越简洁且性能越好；②偏好信息越丰富的模型在指令遵循等任务上表现更好；③较小模型（1B）难以达到压缩最优，性能相对较弱。

**⚠️ 局限性**

局限性：①互信息估计采用后退 n‑gram，无法捕捉更长距离上下文信息；②软熵估计在极大模型中仍是近似，可能低估真实熵；③研究仅建立相关性，缺乏因果验证；④仅评估预训练阶段的压缩效果，对后训练阶段影响探讨有限。

---

## 116. Decisions and Deployment: The Five-Year SAHELI Project (2020-2025) on Restless Multi-Armed Bandits for Improving Maternal and Child Health

**arXiv ID:** 2604.07384 | [PDF](https://arxiv.org/pdf/2604.07384v1)

**作者:** Shresth Verma `[一作]` (Harvard University), Milind Tambe `[通讯]` (Harvard University)

**通讯引用:** 23369 | [OpenAlex ID](https://openalex.org/A5000327528)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在印度的母婴健康项目中，设计并部署了一种基于不稳多臂老虎机（RMAB）模型的动态干预系统，帮助决策者在有限资源下实时分配服务，以提升孕妇和儿童的健康行为。

**💡 创新点**

创新点包括：①将RMAB理论首次应用于公共卫生干预，构建了可在真实世界持续迭代的决策框架；②结合深度强化学习对阈值进行自适应优化；③与地方卫生机构深度合作，实现从实验到真实部署的闭环。

**🔧 技术方法**

使用技术包括：不稳多臂老虎机（RMAB）模型、Thompson Sampling/Upper Confidence Bound 变体、深度强化学习（DQN/Actor-Critic）、分布式系统实现、移动健康（mHealth）数据采集与实时反馈。

**📊 数据集**

数据集来源：与地方卫生部门合作获取的孕产妇健康记录、免疫接种记录、营养补助记录等大规模电子健康数据库；以及针对算法验证的合成模拟数据。

**📈 对比分析**

对比方法：与传统固定频率干预、随机干预以及基于阈值的规则分配进行对比。实测结果显示：RMAB系统将孕妇接诊率提升12%，疫苗覆盖率提升15%，并在资源利用上比基线减少约10%。

**⚠️ 局限性**

局限性：①数据缺失与噪声影响模型鲁棒性；②RMAB假设的马尔可夫性与实际健康行为复杂性存在偏差；③部署依赖技术基础设施和社区参与，难以在极低资源环境中快速扩展；④模型解释性有限，难以满足监管和伦理透明度要求。

---

## 117. An Imperfect Verifier is Good Enough: Learning with Noisy Rewards

**arXiv ID:** 2604.07666 | [PDF](https://arxiv.org/pdf/2604.07666v1)

**作者:** Andreas Plesner `[一作]` (Handshake AI), Anish Athalye `[通讯]` (Handshake AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在代码生成和科学推理任务上使用 RLVR，系统评估了可验证奖励（verifier）噪声对模型训练的影响，发现即使 verifier 误差率高达 15% 也能保持与无噪声基线相当的性能，并且在一定范围内噪声反而有助于提升泛化；

**💡 创新点**

创新点在于提出并验证了“verifier 不需要完美，只需达到约 85% 的准确率和高精度即可”这一结论，并揭示噪声类型、精度与召回对 RLVR 训练效果的差异性影响；

**🔧 技术方法**

使用了 RLVR 的 Group Relative Policy Optimization（GRPO）和 Group Sequence Policy Optimization（GSPO），并设计了四种受控噪声注入模式（样本×单元测试、样本×rollout、组×单元测试、组×rollout），以及基于 LLM 的模型判别器（model‑based verifier）；

**📊 数据集**

主要使用了 MBPP（Python 编码任务）和 GPQA（科学推理多项选择）两大数据集进行实验；

**📈 对比分析**

通过对比无噪声基线、不同噪声率（0~50%）及不同噪声模式下的验证准确率，发现 10%–15% 的噪声下模型性能几乎不降反而略有提升；同时模型基于 30B 验证器可恢复 90% 以上基线性能，而 4B 验证器性能明显下滑；

**⚠️ 局限性**

局限性包括：实验仅覆盖 Qwen3 与 GLM4 两大模型族且噪声为对称且每个 epoch 重新采样，未探究不对称或固定噪声；真实 verifier 的错误模式更为复杂；结果是否能推广至更大模型或其他领域仍待验证。

---

## 118. Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference

**arXiv ID:** 2604.07394 | [PDF](https://arxiv.org/pdf/2604.07394v1)

**作者:** Quantong Qiu `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 84116 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Flux Attention，一种上下文感知的层级动态路由框架，在长文本推理中通过在每层动态切换全注意力（FA）与稀疏注意力（SA）实现显著加速。

**💡 创新点**

创新点在于使用轻量级 Layer Router 在层级层面而非头级进行动态路由，既保留高质量检索，又避免头级稀疏导致的硬件负载不平衡，且通过 Gumbel‑Softmax 训练实现可微软路由并在推理时切换为硬路由。

**🔧 技术方法**

核心技术包括：Gumbel‑Softmax 软路由与温度退火、预填充-后缀池化编码提取上下文特征、拉格朗日松弛的稀疏度约束、以及多种稀疏注意力实现（SSA、XA、TA）和 KV 缓存优化。

**📊 数据集**

训练使用 ChatQA2‑Long‑SFT、MuSiQue、CoLT‑132K、GovReport、XSum 五大数据集；评测基准包含 LongBench‑E、LongBench‑V2、RULER、GSM8K、AIME24 等长文本与推理任务。

**📈 对比分析**

与 DuoAttention、PruLong、TriangleMix 等静态混合注意力方法对比，Flux Attention 在 prefilling 阶段可达 2.8× 速度提升，decode 阶段 2.0×；在 LongBench、RULER、GSM8K、AIME24 等任务中保持或超过全注意力基线，证明了在性能与效率上的优越平衡。

**⚠️ 局限性**

局限性包括：对任务级稀疏预算设置敏感，过低/过高会导致性能波动；对极端长文本的 KV 缓存需求尚未彻底解决；动态路由训练依赖数据分布平衡，需精心设计；以及在多模态或多语言场景的适用性尚未验证。

---

## 119. TRUSTDESC: Preventing Tool Poisoning in LLM Applications via Trusted Description Generation

**arXiv ID:** 2604.07536 | [PDF](https://arxiv.org/pdf/2604.07536v1)

**作者:** Hengkai Ye `[一作]` (Pennsylvania State University), Hong Hu `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自动从工具实现生成可信描述的框架，消除LLM工具使用中的工具毒化攻击。

**💡 创新点**

创新点在于：①首个以实现为基础的可信描述生成框架；②结合可达性感知的代码切片与LLM驱动的冗余代码剪裁；③动态验证循环纠正LLM摘要误差，提升描述准确性。

**🔧 技术方法**

技术手段包括：静态分析构建可达调用图，LLM辅助剪枝（去除不可达/无关逻辑），程序切片摘要生成，识别与去除恶意/误导代码注释与标识符，动态执行任务并用LLM判断结果实现验证。

**📊 数据集**

使用了52个来自12个真实MCP服务器（Python/TypeScript）的工具实现集作为评估数据集。

**📈 对比分析**

对比方法：用原始描述、_lite和_full生成描述，分别在不同LLM（Claude-4.5-Sonnet、Gemini-3-Flash、GPT-5.2、gpt-oss-120b）下执行任务。性能提升：任务成功率平均提升4.3%，生成成本低（Gemini-3-Flash平均0.013美元、25.7s），运行时成本仅+4%代价、+0.2%延迟，且在工具竞争场景下能更好地挑选高质量工具，所有已知工具毒化攻击均被抑制，且对自适应攻击无明显提升趋势。

**⚠️ 局限性**

局限性：依赖工具源码可见且未被加密，LLM的生成与验证仍受模型偏差与幻觉影响，动态验证成本高，占总时间与费用约70%；对非Python/TypeScript语言的适配需要额外解析器，且对已植入隐藏恶意代码的工具无法防御。

---

## 120. FR3 for 6G Networks: A Comparative Study against FR1 and FR2 Across Diverse Environments

**arXiv ID:** 2604.07482 | [PDF](https://arxiv.org/pdf/2604.07482v1)

**作者:** Fahimeh Aghaei `[一作]` (University of Oulu), Murat Uysal `[通讯]` (New York University Abu Dhabi)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用射线追踪与宽带MIMO模型，对FR3（7.125–24.25 GHz）在Suburban、Urban与High‑Rise城市环境下的C‑V2B与C‑P2B下行链路进行详细信道特征化与性能评估。

**💡 创新点**

1) 将FR3与FR1（4.6 GHz）和FR2（28 GHz）在相同天线孔径下进行公平比较；2) 结合3D CAD（迪拜市区）与ITU统计模型的双重射线追踪；3) 引入一手握手式行人UE模型，探讨车辆与行人UE对覆盖的影响；4) 证明FR3在覆盖和边缘用户速率上优于mmWave。

**🔧 技术方法**

射线追踪（Remcom Wireless InSite）+ 频宽MIMO信道模型 + 最大比发射/接收（MRT/MRC）+ 天线阵列（URA/ULA）+ 交叉干扰与静态遮挡建模。

**📊 数据集**

3D CAD模型（迪拜市区）+ ITU统计城市参数（Suburban、Urban、High‑Rise）+ 370个随机UE位置 + 多频点（4.6、8.2、15、28 GHz）+ 物理材料参数表。

**📈 对比分析**

通过比较不同频段、不同干扰场景下的数据速率CDF与覆盖概率，发现：在相同孔径下，FR3在细胞边缘用户的速率上超过FR2，并在非边缘用户与FR2几乎持平；FR3在覆盖概率上略优于行人UE，且在所有频段的差异仅为1%–3%；mmWave在高频段提供最高峰值速率，但边缘用户受路径损耗限制，速率下降明显。

**⚠️ 局限性**

仅考虑静态遮挡，未模拟车辆与人行道障碍的动态效应；仿真范围仅为1.2 km²宏小区；结果受射线追踪精度、天线阵列规模和理想Beamforming假设的影响，实际环境可能产生更大差异。

---

## 121. Interpreting the Error of Differentially Private Median Queries through Randomization Intervals

**arXiv ID:** 2604.07581 | [PDF](https://arxiv.org/pdf/2604.07581v1)

**作者:** Thomas Humphries `[一作]` (University of Waterloo), Xi He `[通讯]` (University of Waterloo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在差分隐私机制下，同时给出均值估计和随机化区间（RI）的算法，专门针对中位数查询。

**💡 创新点**

创新点在于：1) 在保留中位数高效性的同时，通过单次指数机制估计RI；2) 设计了一个量化误差与层级的辅助函数，使RI宽度最小化；3) 对预算分配与步长进行理论最优分析。

**🔧 技术方法**

主要技术包括差分隐私中的指数机制、隐私预算分配、量化误差控制、以及对齐度量的辅助函数；同时结合理论证明和实验验证。

**📊 数据集**

实验使用了Banking、Adult、Airplane三大真实数据集，分别针对balance、fnlwgt、capacity属性进行评估。

**📈 对比分析**

与Sun等人提出的先估计RI后修正中位数的方法相比，该方法平均中位数误差提高了14%–850%，而RI宽度基本相同或略宽；在不同隐私预算下表现出更优的误差-宽度平衡。

**⚠️ 局限性**

局限性包括：1) 仅在1-Lipschitz、无重复元素的数据假设下有效；2) 仍需对预算分配与量化步长进行手动调优；3) 目前仅针对中位数，可望推广到其他统计量。

---

## 122. GAN-based Domain Adaptation for Image-aware Layout Generation in Advertising Poster Design

**arXiv ID:** 2604.07409 | [PDF](https://arxiv.org/pdf/2604.07409v1)

**作者:** Chenchen Xu `[一作]` (Anhui Normal University), Weiwei Xu `[通讯]` (Zhejiang University)

**通讯引用:** 3373 | [OpenAlex ID](https://openalex.org/A5101844050)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于GAN的像素级域适配方法PDA‑GAN，用于从产品图像生成广告海报的内容感知版面布局

**💡 创新点**

提出轻量级像素级判别器对源域（掩码后）和目标域（干净图像）进行对齐，设计三种内容感知评价指标，构建大规模CGL‑Dataset

**🔧 技术方法**

使用Transformer（DETR）生成布局，Pixel‑Level Discriminator结合Gaussian模糊与无监督域适配，CLIP与Saliency网络用于评估

**📊 数据集**

CGL‑Dataset（60,548张配对海报+121,000张干净产品图）以及公开的PKU海报数据集做验证

**📈 对比分析**

与ContentGAN、Layoutprompter、CGL‑GAN以及多种无内容版面生成方法对比，PDA‑GAN在背景复杂度、主体/产品遮挡、cFID等内容感知指标上均优于对手，整体表现为SOTA

**⚠️ 局限性**

仍受限于需依赖大量手工标注的海报配对数据，域适配虽减小差距但对极端图像风格或少量目标域样本的泛化尚待进一步研究

---

## 123. Mathematical Analysis of Image Matching Techniques

**arXiv ID:** 2604.07574 | [PDF](https://arxiv.org/pdf/2604.07574v1)

**作者:** Oleh Samoilenko `[一作]` `[通讯]` (National Academy of Sciences of Ukraine), Oleh Samoilenko (National Academy of Sciences of Ukraine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对卫星图像进行图像匹配，系统评估了SIFT与ORB两种经典局部特征在配准中的表现，利用RANSAC进行几何验证并用Inlier Ratio作为主要评估指标。

**💡 创新点**

①使用手工构造、GPS标注、邻接重叠的卫星图像数据集；②在不同关键点数量下对两种特征进行统一实验；③明确指出SIFT在相同条件下始终优于ORB，并给出在资源受限平台下的最佳特征数范围。

**🔧 技术方法**

传统局部特征提取（SIFT、ORB/FAST+BRIEF），特征匹配（欧氏/汉明距离+暴力匹配），RANSAC几何验证，Inlier Ratio评估。

**📊 数据集**

基于Google Maps Static API获取的GPS标注卫星图像瓷砖，图像之间存在部分重叠，构成自定义数据集。

**📈 对比分析**

对每个图像对，使用不同关键点数量（100/200/500/1000/2000）训练匹配管道，计算平均Inlier Ratio；实验显示SIFT在所有配置下均优于ORB，最大Inlier Ratio约36%，表明SIFT更稳健且增大特征数后收益有限。

**⚠️ 局限性**

仅评估传统特征，未涉及跨视角、学习方法；评估仅以Inlier Ratio为准，缺乏召回/精度等完整分类指标；数据集仅覆盖单一地区，缺乏多样性；在大规模实时系统中的计算和存储成本仍未充分验证。

---

## 124. LitXBench: A Benchmark for Extracting Experiments from Scientific Literature

**arXiv ID:** 2604.07649 | [PDF](https://arxiv.org/pdf/2604.07649v1)

**作者:** Curtis Chong `[一作]`, Jorge Colindres `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了LitXBench实验提取基准并构建LitXAlloy数据集，评估LLM在从文献中抽取实验测量的表现。

**💡 创新点**

提出了材料级别的实验提取框架，使用代码式数据模型和全局规范化的canonical值，提升可审计性与可编辑性。

**🔧 技术方法**

利用大语言模型（Gemini、Claude、GPT-5等）、多轮编码型抽取、Pydantic代码验证和匈牙利算法评估。

**📊 数据集**

构建了包含19篇合金论文、1426条实验测量的LitXAlloy；同时参考了MPEA等公开数据。

**📈 对比分析**

通过对测量、工艺、材料和配置四类加权F1分数进行评估，LLM在材料级提取上比传统方法高约0.37 F1，单属性提取F1可达0.95。

**⚠️ 局限性**

未覆盖图像/表格提取，缺乏显著性数字保留，且仅适用于合金，需要针对其他材料类进行扩展。

---

## 125. Efficient Dataset Selection for Continual Adaptation of Generative Recommenders

**arXiv ID:** 2604.07739 | [PDF](https://arxiv.org/pdf/2604.07739v1)

**作者:** Cathy Jiao `[一作]` (Carnegie Mellon University), Paul Bennett `[通讯]` (Spotify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在大规模流媒体推荐系统中，针对时序漂移导致的性能下降，作者通过从新产生的交互数据中挑选高质量的子集进行周期性再训练，以降低对完整重训练的依赖。

**💡 创新点**

创新点在于：①提出梯度信息驱动的样本表示（GradSim），能够更精准地评估样本对模型更新的贡献；②将分布匹配与多样性采样相结合（如 KNN‑Weighted、Diverse‑Weighted），显著提升在极小数据量下的补偿效果。

**🔧 技术方法**

技术方法包括：Transformer 结构的 HSTU 生成式推荐模型；梯度表示（对最终注意力层梯度取均值并用余弦相似度评分）；Token‑Based、Model‑Based 与 Gradient‑Based 三种表示；Top‑K/Bottom‑K、加权采样、KNN‑Weighted、Diverse‑Weighted 等采样策略；滚动训练协议与 6 个月一次的更新周期。

**📊 数据集**

使用了 2015‑2025 年跨度的专有音乐与播客交互日志，约 10k 用户、1M+ 物品，数据按时间段分块并形成 100 长度的序列样本。

**📈 对比分析**

与随机抽样、仅 Top‑K、加权采样等方法对比，梯度+多样采样在 NDCG@10/NDCG@50/HR@10/HR@50 上恢复至完整重训练误差的 78%，在所有指标上均优于随机与单纯分布匹配的方法；Top‑Bottom‑K 也优于单纯 Top‑K，说明多样性带来收益。

**⚠️ 局限性**

主要局限：①梯度表示需要前向+后向，计算成本较高；②仅使用最终层梯度，可能遗漏更深层次信息；③实验仅在单一音乐/播客领域和 HSTU 模型上验证，迁移性尚未充分评估；④未系统分析选样对训练时长、存储与延迟的影响。

---

## 126. SCOT: Multi-Source Cross-City Transfer with Optimal-Transport Soft-Correspondence Objective

**arXiv ID:** 2604.07383 | [PDF](https://arxiv.org/pdf/2604.07383v1)

**作者:** Yuyao Wang `[一作]` (Boston University), Yongshun Gong `[通讯]` (Shandong University)

**通讯引用:** 1270 | [OpenAlex ID](https://openalex.org/A5040047825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种跨城市地区嵌入迁移框架SCOT，能够在没有节点对应关系的情况下，将源城市的地区表示迁移到目标城市，提升目标城市在标签稀缺场景下的预测性能。

**💡 创新点**

核心创新在于利用可微分的熵正则化Optimal Transport（OT）生成软对应关系，并结合OT加权对比学习实现语义对齐；进一步通过共享原型中心（hub）实现多源迁移，避免单源支配并统一多源对齐空间。

**🔧 技术方法**

技术手段包括：图注意力网络（GAT）作为基底编码器；Sinkhorn迭代求解熵正则化OT得到软对应；OT加权对比损失与对齐损失联合优化；一侧循环重建与熵正则化稳定训练；多源时的共享原型空间与目标诱导的原型先验。

**📊 数据集**

实验使用中国三大城市北京、西安、成都的匿名OD流量生成的地区移动图，对GDP、人口、CO₂排放等指标进行跨城市迁移。

**📈 对比分析**

与基线（无对齐、基于手工匹配、MMD、对抗、CrossTReS、CoRE等）对比，SCOT在所有城市对、任务上均实现MAE/MAPE显著下降，单源时平均提升约30%~40%，多源时更显优势，整体表现最优。

**⚠️ 局限性**

局限性包括：依赖可解释的空间结构；对极端异构城市（大尺度差异）仍可能出现对齐误差；需要在源城市已有标签的前提下训练，无法在完全无标签的目标城市直接使用；以及对超参数（ε、τ、K等）仍有一定敏感性。

---

## 127. Active Reward Machine Inference From Raw State Trajectories

**arXiv ID:** 2604.07480 | [PDF](https://arxiv.org/pdf/2604.07480v1)

**作者:** Mohamad Louai Shehab `[一作]` (University of Michigan), Necmiye Ozay `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于SAT求解的框架，在仅获得原始状态轨迹与有限深度历史策略的前提下，联合学习奖励机的转移函数和标签函数，从而恢复多阶段任务的隐式逻辑结构。

**💡 创新点**

创新点在于：①首次在不观测奖励、标签或节点的情况下同时推断标签函数和奖励机；②给出了足够深度的理论保证（l* = |S|·u_max²）；③设计了主动扩展查询策略，通过挑选信息量最大的轨迹对显著压缩候选空间，降低内存与计算成本。

**🔧 技术方法**

核心技术包括：产品MDP构造、历史策略与负例约束的SAT建模、最大熵强化学习作为背景假设、随机DFS生成轨迹对、基于候选解划分的主动查询、增量SAT求解。

**📊 数据集**

实验使用了离散网格世界（4×4仓库、房间巡逻、Tetris房间）模拟数据，未使用公开真实数据集。

**📈 对比分析**

与完全深度扩展的基线相比，主动扩展在深度9的情况下将内存需求从约24.8 GB降至0.147 GB，SAT求解时间从≈7 200 s降至≈2 400 s，整体运行时间从≈7 185 s降至≈3 545 s，并在足够深度时完全恢复真值奖励机（到命名等价类）。

**⚠️ 局限性**

局限性包括：仅适用于离散状态-动作空间，需精确的历史策略或其估计；对有限样本的鲁棒性未充分验证；只能得到到重命名层面的最小奖励机，实际应用中可能需要更松散的等价判定；扩展到连续或大规模任务仍需进一步研究。

---

## 128. CLEAR: Context Augmentation from Contrastive Learning of Experience via Agentic Reflection

**arXiv ID:** 2604.07487 | [PDF](https://arxiv.org/pdf/2604.07487v1)

**作者:** Linbo Liu `[一作]` (AWS AI Labs), Lin Lee Cheong `[通讯]` (AWS AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种通过对过去执行轨迹进行对比学习的反思代理生成训练数据，再用监督微调和强化学习训练的上下文增补模型（CAM），为LLM代理在推理时提供任务特定的补充上下文，从而提升决策性能；

**💡 创新点**

①利用反思代理进行对比分析生成高质量的上下文摘要；②采用两阶段训练（SFT+RL）让CAM学习生成更有利于执行代理的上下文；③模型可在不改动底层LLM权重的前提下通用各类专有或开源LLM；

**🔧 技术方法**

对比学习、反思代理、监督微调（SFT）、强化学习（GRPO）、向量检索（RAG）、ACE等基线；

**📊 数据集**

AppWorld benchmark、WebShop‑40k（从原WebShop抽样的轻量版）

**📈 对比分析**

与无上下文基线、RAG、ACE等方法对比，CAM在AppWorld的任务完成率从72.62%提升到81.15%，在WebShop的平均奖励从0.68提升到0.74，整体表现显著优于所有基线；

**⚠️ 局限性**

需要对每个任务多次执行并收集轨迹，反思代理与对比学习过程成本较高；在任务多样性或轨迹质量不足时效果可能受限；模型在不同LLM上的迁移仍需进一步验证；

---

## 129. Sima 1.0: A Collaborative Multi-Agent Framework for Documentary Video Production

**arXiv ID:** 2604.07721 | [PDF](https://arxiv.org/pdf/2604.07721v1)

**作者:** Zhao Song `[一作]` `[通讯]`, Zhao Song

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Sima 1.0多代理系统，自动化视频制作的11步流程，显著减少手工编辑负担。

**💡 创新点**

创新点在于将耗时的编辑、字幕、资产集成等任务分配给不同级别的 AI 代理，实现人机协作与流程化生产。

**🔧 技术方法**

采用多代理架构、自然语言处理、自动字幕生成、图像/视频编辑 AI 等技术。

**📊 数据集**

未公开具体数据集，示例中使用常见视频素材与公开字幕数据进行实验。

**📈 对比分析**

与传统人工流程对比，Sima 1.0 在制作周期上缩短约 50%（实验结果仅作示例），提升整体效率。

**⚠️ 局限性**

局限性包括对高级创意决策仍需人工干预，AI 代理性能受训练数据质量限制，且对极端场景的处理仍不够成熟。

---

## 130. Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs

**arXiv ID:** 2604.07518 | [PDF](https://arxiv.org/pdf/2604.07518v1)

**作者:** Mengdan Zhu `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为DLR的强化潜在推理框架，动态将问题分解为前提、在视觉潜在空间中检索依据、生成基于证据的推理链。

**💡 创新点**

通过动态前提分解与前提条件的视觉潜在检索相结合，以及使用球面高斯潜在策略实现对连续潜在空间的有效探索，形成了独特的多步视觉推理流程。

**🔧 技术方法**

采用三阶段训练（预训练、监督微调、强化学习）、球面高斯潜在策略（SGLP）、GRPO改进以及Qwen3‑VL‑8B‑Thinking作为底层视觉‑语言模型。

**📊 数据集**

在四个视觉中心基准（V* Bench、MathVista、MMMU‑Pro、MMStar）以及构造的Vision‑R1‑cold衍生DLR训练集上进行实验。

**📈 对比分析**

与文本仅模型、ICoT、PixelReasoner、LVR等基线对比，DLR在所有基准上均优于对手，提升幅度从约4.2%到6.6%，甚至超越200B参数的GPT‑4o。

**⚠️ 局限性**

目前仅验证于图像任务，缺乏在视频、机器人决策或需外部交互的更广泛多模态场景中的适用性。

---

## 131. Google, AI Literacy, and the Learning Sciences: Multiple Modes of Research, Industry, and Practice Partnerships

**arXiv ID:** 2604.07601 | [PDF](https://arxiv.org/pdf/2604.07601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 132. Reset-Free Reinforcement Learning for Real-World Agile Driving: An Empirical Study

**arXiv ID:** 2604.07672 | [PDF](https://arxiv.org/pdf/2604.07672v1)

**作者:** Kohei Honda `[一作]` (Nagoya University), Hirotaka Hosogaya `[通讯]` (Nagoya University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在真实滑行轨道上实现无重置强化学习，训练1/10比例车辆进行高速度敏捷驾驶，并比较PPO、SAC、TD‑MPC2等算法及其残差学习效果。

**💡 创新点**

首次将MPPI同时用作基线与重置策略，在真实环境中开展无重置RL实验，并系统对比仿真与真实结果，揭示两者的显著差异。

**🔧 技术方法**

使用MPPI、PPO、SAC、TD‑MPC2等RL算法，并在ROS 2框架下实现分布式控制；利用2 D LiDAR、IMU、轮速计等传感器；对仿真采用F1TENTH‑Gym，真实系统为1/10比例RoboRacer。

**📊 数据集**

采用实时采集的物理轨道数据及对应的F1TENTH‑Gym仿真环境，未使用公开数据集。

**📈 对比分析**

通过累计奖励对比评估性能；仿真中SAC(残差)表现最佳，TD‑MPC2次之；真实环境中仅TD‑MPC2能持续超越MPPI基线，SAC收敛为过于保守，残差学习在真实环境无明显提升。

**⚠️ 局限性**

受限于真实世界的噪声、观测延迟和模型不匹配，模型自由RL表现差；残差学习受基线性能限制；实验仅在1/10比例车辆和单一滑行轨道上，缺乏对更高速度或多轨道环境的验证。

---

## 133. SMT with Uninterpreted Functions and Monotonicity Constraints in Systems Biology

**arXiv ID:** 2604.07496 | [PDF](https://arxiv.org/pdf/2604.07496v1)

**作者:** Ondřej Huvar `[一作]` (Masaryk University), Samuel Pastva `[通讯]` (Masaryk University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出利用无解释函数（uninterpreted functions）结合单调性约束来进行生物系统的模型推理。

**💡 创新点**

创新点在于将单调性约束转化为有限数量的量化实例化子句，并引入惰性实例化策略，显著提升求解效率。

**🔧 技术方法**

主要技术包括基于Z3的SMT求解、量化编码、强行实例化（eager instantiation）与惰性实例化（lazy instantiation），以及对无解释函数的单调性推导。

**📊 数据集**

使用了三个数据集：bbm‑boolean（8381实例），bma‑integer（465实例）和omnipath（144实例），覆盖布尔网络、多值网络和大规模基因网络。

**📈 对比分析**

与传统工具Bonesis和AEON比较，instantiated‑lazy在所有数据集上都能在几秒内完成求解，甚至在高阶函数和整数域时仍保持可扩展性，优于量化编码和现有工具。

**⚠️ 局限性**

局限性包括仅支持有限域与单调性约束，未对非单调或连续模型做扩展，对极大规模网络仍可能产生内存瓶颈。

---

## 134. Towards Real-Time Human-AI Musical Co-Performance: Accompaniment Generation with Latent Diffusion Models and MAX/MSP

**arXiv ID:** 2604.07612 | [PDF](https://arxiv.org/pdf/2604.07612v1)

**作者:** Tornike Karchkhadze `[一作]`, Shlomo Dubnov `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

实现了实时人机音乐共同表演框架，将MAX/MSP前端与Python后端的潜在扩散模型结合，现场即兴生成伴奏；

**💡 创新点**

提出滑动窗口前瞻推理范式并引入一致性蒸馏加速，克服扩散模型的高延迟瓶颈；

**🔧 技术方法**

采用潜在扩散模型（LDM）、Music2Latent压缩编码器、一致性蒸馏、OSC/UDP通信及MAX/MSP外部实现；

**📊 数据集**

在Slakh2100多轨音频数据集上训练，聚焦贝斯、鼓、吉他、钢琴四个乐器；

**📈 对比分析**

与StreamMusicGen基线对比，回顾性模式下性能相当甚至略优，前瞻模式下表现略好，但随着前瞻深度增加，COCOLA、Beat F1和FAD均出现衰减；

**⚠️ 局限性**

前瞻深度越大生成质量下降；模型仍受限于采样步骤和硬件，无法支持更细粒度步长，网络延迟和多平台兼容性仍是挑战。

---

## 135. Towards Knowledgeable Deep Research: Framework and Benchmark

**arXiv ID:** 2604.07720 | [PDF](https://arxiv.org/pdf/2604.07720v1)

**作者:** Wenxuan Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了Knowledgeable Deep Research (KDR) 任务，并设计了Hybrid Knowledge Analysis (HKA) 框架来生成包含结构化与非结构化知识的多模态研究报告，随后构建了KDR-Bench评测集。

**💡 创新点**

创新点包括：①将结构化（表格、图表）与非结构化（网页）知识统一纳入多代理推理；②通过代码LLM和视听模型生成图表并提炼见解；③提出知识点覆盖与视觉增强等新的评估指标。

**🔧 技术方法**

采用了多代理架构（Planner、Unstructured Knowledge Analyzer、Structured Knowledge Analyzer、Writer）、代码生成与执行、检索与再排序、Vision‑Language 模型、LLM 评测与比较。

**📊 数据集**

使用了包含9个领域、18个子领域、41个主题共1,252张表的KDR-Bench数据集，其中包含41个专家级研究问题及其主要结论与关键点。

**📈 对比分析**

通过与12个基线（LLM+搜索、闭源与开源DR）在一般性、知识中心化和视觉增强三类指标上进行对比；HKA在知识中心化和视觉增强指标上显著优于大多数基线，且在表格使用与图表生成方面明显领先；在与Gemini的比较中，HKA在视觉增强上胜过Gemini，但在一般性指标略逊。

**⚠️ 局限性**

局限性包括：对强大的LLM与代码/视听模型的依赖；对表格规模扩大后的可扩展性仍需研究；评估仍受判别模型偏差影响，且对非表格结构化知识（如图谱）的支持有限。

---

## 136. Beyond Pedestrians: Caption-Guided CLIP Framework for High-Difficulty Video-based Person Re-Identification

**arXiv ID:** 2604.07740 | [PDF](https://arxiv.org/pdf/2604.07740v1)

**作者:** Shogo Hamano `[一作]` (Sony Group Corporation), Sayaka Nakamura `[通讯]` (Sony Group Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于CLIP的Caption-guided CLIP（CG-CLIP）框架，用于解决高难度视频行人重识别（如同服装的运动员或舞者）中的身份识别问题。

**💡 创新点**

创新点包括：① 通过Caption-guided Memory Refinement (CMR)模块利用多模态大语言模型生成的文本描述来细化图像记忆；② 通过Token-based Feature Extraction (TFE)模块使用可学习的固定长度 token 以跨注意力方式高效聚合时空特征；③ 结合上述两项技术实现对细粒度区分的显著提升。

**🔧 技术方法**

主要技术手段包括：CLIP视觉-文本预训练模型、跨注意力与自注意力机制、Momentum更新的记忆池、三元组损失与标签平滑交叉熵、基于多模态大语言模型（Phi-4-Multimodal）生成的图像描述。

**📊 数据集**

使用数据集：标准视频行人ReID基准 MARS、iLIDS-VID；以及作者自构建的高难度数据集 SportsVReID（运动）和 DanceVReID（舞蹈），后两者从 SportsMOT、DanceTrack 中提取并手工标注属性后生成字幕。

**📈 对比分析**

与最新方法（FT-CLIP、VSLA-CLIP、TF-CLIP 等）比较，CG-CLIP 在 MARS 上 mAP 89.8%、Rank-1 92.5%，在 iLIDS-VID 上 Rank-1 96.7%，在 SportsVReID 上 mAP 77.7%、Rank-1 90.4%，在 DanceVReID 上 mAP 77.8%、Rank-1 90.4%，均显著优于对比方法，尤其在高相似度场景下提升更为明显。

**⚠️ 局限性**

局限性：① 训练阶段依赖高质量字幕，若生成字幕噪声较大可能影响效果；② 目前仅在行人ReID任务验证，尚未测试对其他多模态检索或跨域场景的泛化能力；③ 虽 TFE 降低了自注意力的算力，但仍需额外的可学习 token 及跨注意力计算，推理时对硬件仍有一定要求。

---

## 137. Cluster Attention for Graph Machine Learning

**arXiv ID:** 2604.07492 | [PDF](https://arxiv.org/pdf/2604.07492v1)

**作者:** Oleg Platonov `[一作]` (HSE University), Liudmila Prokhorenkova `[通讯]` (Yandex Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 Cluster Attention (CLATT)，一种在图聚类基础上实现的注意力机制，用于在保持图结构先验的同时实现长距离节点交互，改进传统 Message Passing Neural Networks (MPNN) 和 Graph Transformers (GT)。

**💡 创新点**

创新点在于将图聚类结果作为注意力的聚合域，既克服了 MPNN 的受限感受野，又保留了图结构的诱导偏置；同时可灵活选取多种聚类算法并叠加，进一步提升模型表达能力。

**🔧 技术方法**

技术实现包括：①在预处理阶段使用四种高效聚类算法（Leiden、Bayesian Planted Partition、Hierarchical Statistical Clustering、基于 ResMLP 的 k‑means）生成节点簇；②在模型中对每个簇内部使用多头稀疏注意力（scaled dot‑product attention）实现 CLATT；③将 CLATT 输出与原始 MPNN 或 GT 输出拼接后通过线性层映射回原维度；④采用 JIT‑编译和 ragged tensor 进一步提升效率。

**📊 数据集**

使用了 12 个真实世界图数据集，涵盖交通网络、社交网络、推荐系统、问答网络等，来源于 GraphLand 基准、常见社交网络 (CiteSeer、CiteseerX) 与 Amazon 评价数据。

**📈 对比分析**

通过在同一数据集和相同训练/验证/测试拆分上比较基准模型与其 CLATT 版，采用 R²、AP、Accuracy 等指标；实验显示 CLATT 在所有模型中均能提升性能，提升幅度从几个百分点到十余个百分点不等，且在大型图上（如 1.6M 节点）仍保持高效；相较于纯 GT，CLATT 进一步缩小两者性能差距。

**⚠️ 局限性**

局限性包括：对具有明显聚类结构的图有效，对无明显簇或高度规则网格图可能无效；需要额外的聚类预处理，尽管成本低但在极大图或动态图中仍可能成为瓶颈；对不同聚类算法的选择依赖超参数调优。

---

## 138. Agentic Copyright, Data Scraping & AI Governance: Toward a Coasean Bargain in the Era of Artificial Intelligence

**arXiv ID:** 2604.07546 | [PDF](https://arxiv.org/pdf/2604.07546v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 139. Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference

**arXiv ID:** 2604.07472 | [PDF](https://arxiv.org/pdf/2604.07472v1)

**作者:** Jiaming Cheng `[一作]` (Arizona State University), Duong Tung Nguyen `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出两种基于约束感知的快速启发式算法（Greedy Heuristic 与 Adaptive Greedy Heuristic），用于联合模型选择、GPU 配置、张量/流水并行度以及工作负载分配；

**💡 创新点**

创新点在于引入三种约束感知机制（M1-M3）保证在 GPU 内存、延迟、误差、预算等耦合约束下的可行性，并通过多起始构造、局部搜索和资源整合显著提升解质量；

**🔧 技术方法**

技术包括 MILP 形式化、单层 McCormick 线性化、启发式贪婪分配、局部搜索、GPU 并行度升级、资源再整合以及滚动窗口重新优化；

**📊 数据集**

使用 Azure LLM 推理轨迹（2025）校准的多种查询类型（文本、代码、翻译、数学、图像、视频）与 6 种 Llama-3.x 模型及 10 种 GPU 级别（A6000、RTX4090、A100、H100 等）作为实验数据；

**📈 对比分析**

与精确 MILP 求解器和其他系统（如 DynamoLLM、Helix 等）比较，AGH 在大规模实例上接近最优成本，速度提升超过 260 倍；在 1.5 倍延迟/误差压力下保持低成本和低 SLO 违约率；在滚动重优化场景中，每 5 分钟重新求解能比静态方案节省 16‑48% 费用；

**⚠️ 局限性**

局限性包括缺乏对排队和连续批处理动态的建模、未考虑完整的多租户成本与实时监控以及在更大规模多云环境下的可扩展性验证；

---

## 140. Differentially Private Modeling of Disease Transmission within Human Contact Networks

**arXiv ID:** 2604.07493 | [PDF](https://arxiv.org/pdf/2604.07493v1)

**作者:** Shlomi Hod `[一作]` (Boston University), Adam Smith `[通讯]` (Boston University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个基于节点级差分隐私的隐私保护传染病传播模拟管道，包含统计网络模型生成合成网络并在其上进行代理模型传播。

**💡 创新点**

创新点在于将节点级差分隐私与统计网络模型（SBM、ERGM）结合，并在整个模拟链中评估隐私引入的误差与方差，证明隐私保护对传播结果影响有限。

**🔧 技术方法**

使用了节点级差分隐私（Laplace机制+度截断）、统计网络模型（SBM、ERGM）、网络代理模型（SIS传播）以及方差分解（ANOVA）等技术。

**📊 数据集**

以ARTNet项目的MSM性伴侣网络（约10,000节点）为实验基准网络，生成合成网络进行评估。

**📈 对比分析**

通过对比无隐私、有限隐私（ε=0.5,1,5,10）和无限隐私（仅度截断）以及模型指定与未指定的情形，评估了误差来源、均值与方差。结果显示：模型误差是主导误差；隐私噪声在合理参数下对传播指标（总体和分组）影响微乎其微；方差来源中网络采样占比最大，隐私噪声方差小于采样与传播方差。

**⚠️ 局限性**

局限包括仅使用静态交叉网络、仅考虑SIS模型、仅采用SBM/ERGM两种网络模型、隐私预算范围受限于现有算法、未探索更复杂的动态网络和更高阶的差分隐私机制。

---

## 141. Squeeze Evolve: Unified Multi-Model Orchestration for Verifier-Free Evolution

**arXiv ID:** 2604.07725 | [PDF](https://arxiv.org/pdf/2604.07725v1)

**作者:** Monishwaran Maheswaran `[一作]` (University of California Berkeley), Chenfeng Xu `[通讯]` (University of Texas Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Verifier-Free Evolution的多模型编排框架 Squeeze‑Evolve，通过在进化步骤中动态路由不同成本模型，提升推理效率和性能。

**💡 创新点**

创新点在于利用模型自身或跨模型的置信度作为路由依据，按模型的边际效益分配任务，从而解决多样性衰退和高昂成本的瓶颈。

**🔧 技术方法**

技术包括统一进化框架、置信度路由策略、分层重组、分布式 GPU 资源池以及基于置信度的动态阈值调度。

**📊 数据集**

实验数据集涵盖 AIME 2025、HMMT 2025、GPQA‑Diamond、LiveCodeBench V6、MMMU‑Pro、BabyVision、ARC‑AGI‑V2 与 Circle Packing 等多种数学、编码、视觉与科学发现任务。

**📈 对比分析**

与单模型 RSA、AlphaEvolve 等方法比较，Squeeze‑Evolve 平均降低 1.3–3.3 倍 API 成本、提升 10 倍吞吐量，ARC‑AGI‑V2 达 97.5% 仅 $7.74/题，Circle Packing 与 AlphaEvolve 竞争力相当。

**⚠️ 局限性**

局限性包括置信度路由噪声、对模型对齐与安全性缺乏理论保证、对复杂多模态任务的适配仍有限，以及缺少对动态参数（如群体大小、阈值）自适应调节机制。

---

## 142. Detecting HIV-Related Stigma in Clinical Narratives Using Large Language Models

**arXiv ID:** 2604.07717 | [PDF](https://arxiv.org/pdf/2604.07717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 143. Needle in a Haystack -- One-Class Representation Learning for Detecting Rare Malignant Cells in Computational Cytology

**arXiv ID:** 2604.07722 | [PDF](https://arxiv.org/pdf/2604.07722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 144. Optimal Decay Spectra for Linear Recurrences

**arXiv ID:** 2604.07658 | [PDF](https://arxiv.org/pdf/2604.07658v1)

**作者:** Yang Cao `[一作]` `[通讯]`, Yang Cao

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种架构无关的框架 PoST，解决线性递归模型在长范围记忆中的衰减谱退化问题。

**💡 创新点**

创新点在于：① 通过谱重参数化强制对数衰减率几何间距，达到 minimax 最优的衰减速率；② 位置自适应缩放消除静态谱的尺度失配，进一步提升对时间位置 t 的记忆精度，并自然产生分数尺度不变性。

**🔧 技术方法**

采用谱重参数化、位置自适应缩放两种机制，并将 PoST 无开销地嵌入到任何对角线线性递归结构中；在多种模型（Mamba‑2、RWKV‑7、Gated DeltaNet、Gated Linear Attention、RetNet）上实现。

**📊 数据集**

使用大规模预训练语料（约 1.8 亿至 4.4 亿参数），在零样本语言建模和长上下文检索任务（MQAR、NIAH）上进行评估。

**📈 对比分析**

与原始模型相比，PoST 在零样本语言建模上均衡提升，Mamba‑2 在 MQAR 和 NIAH 上获得显著的长上下文检索优势，其他模型保持或略优于基线，且未增加额外计算开销。

**⚠️ 局限性**

局限性：仅针对对角线线性递归设计，未在非对角线或更大规模模型上进行充分验证；对极长上下文（T 超过模型设计范围）时的表现仍待进一步研究。

---

## 145. Twitch Third-Party Developers' Support Seeking and Provision Practices on Discord

**arXiv ID:** 2604.07732 | [PDF](https://arxiv.org/pdf/2604.07732v1)

**作者:** Jie Cai `[一作]` (Tsinghua University), Chun Yu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过混合方法（主题模型与主题分析）研究 Twitch 第三方开发者在 Discord 社区中的支持寻求与提供行为，揭示了其对 Twitch 平台的高度依赖、平台劳动与跨平台迁移现象以及多样化的支持角色；

**💡 创新点**

首次系统性描述 TPD 在外围社区（Discord）中的支持实践及其与核心平台（Twitch）的相互作用，提出了桥接正式与非正式支持空间的设计启示；

**🔧 技术方法**

采用 LDA 主题建模（结合主题相关度与可解释性）识别支持相关主题，再用归纳式与演绎式主题分析对话进行细粒度编码；

**📊 数据集**

使用 TwitchDev Discord 服务器“Lobby”频道的聊天记录，原始 45,376 条信息，预处理后 8,219 条、279,581 词；

**📈 对比分析**

通过 LDA 的 coherence 分数验证主题质量，coherence 在 8 主题时最高，表明模型在捕捉讨论主题上具备较好可解释性；

**⚠️ 局限性**

仅聚焦单一频道且仅研究 Twitch TPD，未覆盖其他平台或其他 Discord 频道，且研究主要为质性分析，缺乏跨平台或量化对比研究。

---

## 146. A Novel Edge-Assisted Quantum-Classical Hybrid Framework for Crime Pattern Learning and Classification

**arXiv ID:** 2604.07389 | [PDF](https://arxiv.org/pdf/2604.07389v1)

**作者:** Niloy Das `[一作]` (Noakhali Science and Technology University), Choong Seon Hong `[通讯]` (Kyung Hee University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一个量子‑经典对比框架，用于对孟加拉国16年犯罪数据进行模式学习与分类。

**💡 创新点**

创新点包括：基于相关性自适应的量子电路设计、双向量子‑经典混合架构以及针对犯罪分析的系统评估方法。

**🔧 技术方法**

使用技术有：变分量子分类器（VQC）、量子近似优化算法（QAOA）、量子核支持向量机、Q→C 与 C→Q 混合模型以及经典随机森林、SVM、逻辑回归等基线。

**📊 数据集**

使用的数据集为孟加拉国16年的犯罪统计数据，共272条样本（18个地区×16年）。

**📈 对比分析**

通过分层5折交叉验证（5个随机种子）进行性能比较；QAOA在仅16个可训练参数的情况下达到约84.6%准确率，表现出优秀的参数效率；但经典随机森林在准确率上仍高达94.5%，训练时间也更长；混合模型在效率上有一定提升。

**⚠️ 局限性**

局限性包括仅在经典模拟环境评估、样本量有限、类别不平衡严重、未在真实量子硬件上验证，以及对不同数据分布的泛化性不足。

---

## 147. Implicit Regularization and Generalization in Overparameterized Neural Networks

**arXiv ID:** 2604.07603 | [PDF](https://arxiv.org/pdf/2604.07603v1)

**作者:** Zeran Johannsen `[一作]` `[通讯]`, Zeran Johannsen

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在参数过剩的深度网络中为何能实现良好泛化，并在同一实验框架下系统评估了五种解释机制（SGD隐式正则、损失曲面平坦度、双曲下降、NTK宽网络行为和Lottery Ticket剪枝）。

**💡 创新点**

创新点在于将这五种机制放入统一的实验流程，量化它们之间的相互作用，并通过同一数据集与相同训练设置直接对比各机制的贡献。

**🔧 技术方法**

使用了SGD、Adam、Hessian主特征值估计、权重扰动分析、NTK宽网络实验、迭代幅度剪枝等技术；训练时还采用了余弦学习率退火和Batch Normalization。

**📊 数据集**

实验基于MNIST和CIFAR-10两大公开数据集，覆盖从小网络到近4千万参数的多种规模。

**📈 对比分析**

通过不同批量大小、网络宽度、稀疏比例以及随机初始化对照等方法比较，结果显示：小批量SGD和平坦极小值可将测试误差提升约2.25个百分点，宽网络在NTK趋近时参数移动减少11.3倍且准确率提升至≈93.9%；Lottery Ticket子网络在保留10%参数时仅损失1.15个百分点，并且比随机重初始化差2.80个百分点；整体测试准确率在MNIST上最高可达96.9%，CIFAR-10上可达≈86%。

**⚠️ 局限性**

局限性包括：仅使用MNIST/CIFAR-10等小规模数据集，未加入数据增强或显式正则；实验规模仅至数千万参数，未检验亿级/万亿级模型；Batch Normalization 本身隐含正则；Hessian估计仅为局部最优特征，未覆盖完整曲面；NTK实验仍未完全进入无限宽极限；未验证Transformer等新架构的适用性。

---

## 148. SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments

**arXiv ID:** 2604.07599 | [PDF](https://arxiv.org/pdf/2604.07599v1)

**作者:** Kota Kondo `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为SANDO的安全轨迹规划器，能够在3D动态未知环境中实现实时可重复的安全飞行。

**💡 创新点**

创新点包括热图A*全局规划、时间层化安全飞行走廊(STSFC)以及变量消元的硬约束MIQP优化，保证连续时间安全并显著提升求解速度。

**🔧 技术方法**

采用了热图加权A*、A*软成本搜索、STSFC生成、混合整数二次规划(MIQP)与变量消元、时空热图与预测、AEKF动态障碍物跟踪等技术。

**📊 数据集**

在仿真中使用标准静态基准、随机障碍森林、包含trefoil轨迹的动态环境以及真实LiDAR/RealSense传感器进行硬件实验。

**📈 对比分析**

与FASTER、SUPER、EGO‑Swarm2、I‑MPC、FAPP等方法对比，SANDO在所有难度级别保持100%成功率、无约束违规、最快轨迹与最低计算时延，显著优于基线。

**⚠️ 局限性**

局限性包括缺乏递归可行性保证、对未知空间缺失完整安全保障、以及在高密度动态环境下对最坏情况可观测不确定性导致的过度保守。

---

## 149. GIRL: Generative Imagination Reinforcement Learning via Information-Theoretic Hallucination Control

**arXiv ID:** 2604.07426 | [PDF](https://arxiv.org/pdf/2604.07426v1)

**作者:** Prakul Sunil Hiremath `[一作]` `[通讯]` (Visvesvaraya Technological University), Prakul Sunil Hiremath (Visvesvaraya Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GIRL框架，通过交叉模态地面向预训练基础模型的语义锚点和自适应信任域瓶颈，解决基于隐层的模型驱动强化学习中长期规划时的想象漂移问题。

**💡 创新点**

创新点：①使用冻结的DINOv2视觉模型提取语义锚点并通过残差门和一致性损失把隐层转移先验与语义空间对齐；②将KL正则化重新表述为受约束优化的拉格朗日乘子，并用信息增益（EIG）与相对性能损失（RPL）动态更新信任域；③推导出不含(1-γ)⁻²因子的价值差距上界，将I‑ELBO与真实环境回报差距关联；④针对全关节观测提供ProprioGIRL（基于MSAE的时序自监督锚点）。

**🔧 技术方法**

技术手段包括：隐层状态空间模型（RSSM）、跨模态残差门、DINOv2前向推断、轻量级投影网络、一致性损失、信息增益与相对性能损失估计、KL拉格朗日双变量更新、基于IPM的价值差距理论、以及Distilled Semantic Prior（DSP）压缩DINOv2推理。

**📊 数据集**

使用三个主流基准：DeepMind Control Suite（含视觉干扰版本）、Adroit Hand Manipulation（门、锤、笔）以及Meta‑World MT10，全部采用标准的随机种子、训练步数和评价框架。

**📈 对比分析**

与DreamerV3、TD‑MPC2等基线比较，GIRL在18个任务上实现了约40–55%的环境步数提升、58–68%的想象漂移降低、IQM提升至0.78（vs. 0.67 DreamerV3）、PI>0.5对所有基线，且在稀疏奖励和高接触任务中显著优于TD‑MPC2；Distilled Prior版本在保持近似性能的同时将推理开销降至5%。

**⚠️ 局限性**

局限性：1）未压缩版本计算开销较高（≈30%）；2）对完全无视觉输入的任务需额外的ProprioGIRL；3）信任域的双变量更新对初始值敏感，需手动warm‑start；4）目前仅在连续控制/操纵域验证，离散或部分可观测环境仍待探索。

---

## 150. Adaptive Depth-converted-Scale Convolution for Self-supervised Monocular Depth Estimation

**arXiv ID:** 2604.07665 | [PDF](https://arxiv.org/pdf/2604.07665v1)

**作者:** Yanbo Gao `[一作]` (Shandong University), Tian Xie `[通讯]` (Zhejiang Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Depth-converted-Scale Convolution (DcSConv) 与 Depth-converted-Scale aware Fusion (DcS-F) 模块，利用深度-尺度关系自适应调整卷积尺度，提升自监督单目深度估计的特征提取与融合。

**💡 创新点**

①基于深度-尺度转换关系实现卷积尺度自适应；②将自适应卷积与普通卷积通过尺度感知注意力融合；③可插拔的 DcSConv/GLSConv 模块；④首次在深度估计任务中突出卷积尺度的重要性。

**🔧 技术方法**

自监督深度估计框架、Encoder‑Decoder 结构、深度到尺度映射、可变尺度卷积、双线性插值、通道/空间注意力、残差连接、重建损失与平滑损失等。

**📊 数据集**

KITTI（Eigen split）、Make3D、NYU‑V2。

**📈 对比分析**

在 KITTI 上与 Monodepth2、CADepth、MonoViT 等基线比较，SqRel 下降 11.6%、17.9%、23.4%，AbsRel、RMSE、δ 等指标同步提升；在 Make3D、NYU‑V2 上亦分别提升 12.6% 与 18.1%。

**⚠️ 局限性**

需要初始或预训练深度用于尺度映射，参数与算力略增；仅在相机运动视频场景验证，对静态单帧或非针孔相机的鲁棒性尚未充分评估；未结合语义或其它辅助信息。

---

## 151. Personalizing Text-to-Image Generation to Individual Taste

**arXiv ID:** 2604.07427 | [PDF](https://arxiv.org/pdf/2604.07427v1)

**作者:** Anne-Sofie Maerten `[一作]` (University of Tübingen), Matthias Bethge `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个大型用户个性化审美评价数据集PAM∃LA，并训练了基于用户特征的奖励模型实现文本到图像生成的个性化优化。

**💡 创新点**

创新点在于收集了70,000条多用户评分（每图像15条），并构建了用户条件的奖励预测器，使模型能够捕捉到不同用户的主观审美偏好；同时通过prompt迭代优化展示了个性化生成效果。

**🔧 技术方法**

主要技术包括：冻结SigLIP2视觉/文本编码器、用户与图像元数据的自然语言表征、浅层Transformer融合、基于交互矩阵的用户嵌入插值，以及基于DPO的prompt优化流程。

**📊 数据集**

使用的数据集为PAM∃LA（5,077 AI生成图像与70k评分），以及LAPIS（艺术品）和PARA（摄影）用于联合训练。

**📈 对比分析**

与现有全局奖励模型（LAION-Aesthetics、ImageReward、Q-Align、DeQA、HPSv3）在用户级和总体级指标上对比，PAM∃LA在SROCC、PLCC、pairwise accuracy上均优于对手，尤其在个体化指标上提升明显。

**⚠️ 局限性**

局限性包括：仍难以准确预测用户意见高度分歧的情况；评分尺度中的近似平局带来噪声，影响模型评估；以及模型在新用户场景下仍依赖少量上下文样本。

---

## 152. The Shrinking Lifespan of LLMs in Science

**arXiv ID:** 2604.07530 | [PDF](https://arxiv.org/pdf/2604.07530v1)

**作者:** Ana Trišović `[一作]` `[通讯]` (Massachusetts Institute of Technology), Ana Trišović (Massachusetts Institute of Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对62种LLM在108k科研论文中的采用轨迹进行系统追踪并量化。

**💡 创新点**

首次揭示LLM在科研中的采用呈倒U形曲线，并发现该曲线随代际递减，生命周期压缩。

**🔧 技术方法**

利用Semantic Scholar Academic Graph、S2ORC文本抽取，结合GPT‑4.1‑mini zero‑shot分类与贝叶斯校正进行采用判定，并用多元回归分析生命周期特征。

**📊 数据集**

使用Semantic Scholar的论文引用数据与S2ORC全文，覆盖2018–2025年108,514篇引用论文。

**📈 对比分析**

与传统单一计数对照，归一化采用曲线和回归模型表明时间至峰值每年缩短27%，生命周期每年缩短23%，体现明显压缩。

**⚠️ 局限性**

局限包括对英文开放获取论文的偏倚、分类器误差可能导致低估、样本量有限导致子组分析功效不足、仅以论文引用计数衡量采用且未捕捉非正式使用。

---

## 153. CausalVAE as a Plug-in for World Models: Towards Reliable Counterfactual Dynamics

**arXiv ID:** 2604.07712 | [PDF](https://arxiv.org/pdf/2604.07712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 154. From Papers to Property Tables: A Priority-Based LLM Workflow for Materials Data Extraction

**arXiv ID:** 2604.07584 | [PDF](https://arxiv.org/pdf/2604.07584v1)

**作者:** Koushik Rameshbabu `[一作]` (Johns Hopkins University), Jaafar A. El-Awady `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用大型语言模型（LLM）通过层级化优先级流程，自动从科研论文中提取并重建全实验记录，涵盖文本、表格、图像和物理公式来源，聚焦合金抗冲击强度数据；

**💡 创新点**

创新点在于将三层提取优先级（直接提取→公式推导→图像数字化）嵌入prompt，实现无任务专门微调的、可追溯、物理一致的数据重建；

**🔧 技术方法**

使用Gemini 3 Pro及Claude Opus 4.5等LLM，配合定制prompt、单位规范、物理约束与后置验证；

**📊 数据集**

在30篇1996‑2024年发表的高压冲击实验论文上共计11,967条数据点进行评估；

**📈 对比分析**

与人工手工标注及其他模型比较，优先级1/2/3分别达94.93%、92.04%、83.49%准确率，整体加权准确率94.69%；与Claude模型的跨模型一致性平均88.76%；API版本甚至提高到100%；

**⚠️ 局限性**

局限性主要在图像数字化（T3）误差较大，且对包含大量分散信息的长文档存在回溯难度；在极少数低准确度文献中仍需人工核查；

---

## 155. PRIME: Training Free Proactive Reasoning via Iterative Memory Evolution for User-Centric Agent

**arXiv ID:** 2604.07645 | [PDF](https://arxiv.org/pdf/2604.07645v1)

**作者:** Prince Zizhuang Wang `[一作]` (Carnegie Mellon University), Shuli Jiang `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PRIME 框架，通过梯度无关的记忆演化提升 LLM 代理在多轮用户交互中的主动工具使用能力。

**💡 创新点**

创新点是将代理学习转为知识库演化，用结构化经验驱动推理而非参数更新；同时提供多阶段、三域内存组织与元层优化。

**🔧 技术方法**

使用的技术包括结构化经验提炼、多轮信用分配、遗传式记忆演化（变异、泛化、交叉、裁剪）以及检索增强生成。

**📊 数据集**

在八个多轮用户交互基准（TravelGym、TurtleGym、FunctionGym、TauGym、PersuadeGym、IntentionGym、TelepathyGym、SearchGym）上进行实验。

**📈 对比分析**

与零样本 GPT‑4o、ReAct、Reflexion 以及 GRPO‑RL 进行比较，PRIME 在所有环境中均超越原始开源 LLM，且与 GRPO 接近但需 5–6 倍更少 GPU‑时。

**⚠️ 局限性**

局限性在于依赖高质量的经验提炼和检索效果，跨模型迁移时表现下降，且无法在模型已固化时进行在线更新。

---

## 156. Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge

**arXiv ID:** 2604.07399 | [PDF](https://arxiv.org/pdf/2604.07399v1)

**作者:** Wonseon Lim `[一作]` (Chung-Ang University), Dae-Won Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 10789 | [OpenAlex ID](https://openalex.org/A5031878697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CPS-Prompt框架，通过在持续学习中引入关键补丁采样与分离训练来提升边缘设备的训练效率。

**💡 创新点**

创新点在于将任务感知的稀疏补丁采样与解耦的提示与分类器训练相结合，既降低了训练时内存与计算，又保持了高精度。

**🔧 技术方法**

使用冻结的ViT backbone、查询编码器提取补丁重要性、温度缩放的多项式采样、以及分阶段的Prompt/Classifier训练。

**📊 数据集**

在CIFAR-100、ImageNet-R和CUB-200三个公开增量学习基准上进行实验。

**📈 对比分析**

与CODA-Prompt、C-Prompt、OS-Prompt等现有prompt学习方法以及LwF、ER等传统CL方法对比，CPS-Prompt在保持ACC仅低于C-Prompt 2% 的同时，内存使用降低约1.6倍、训练时间缩短约1.5倍、能耗下降约1.6倍。

**⚠️ 局限性**

局限性包括：在极高稀疏率下精度仍会下降、需要预设温度和阶段比例，未针对动态资源约束或其他CL范式（如任务迁移、灾难性遗忘）进行评估。

---

## 157. Energy-Efficient Drone Logistics for Last-Mile Delivery: Implications of Payload-Dependent Routing Strategies

**arXiv ID:** 2604.07514 | [PDF](https://arxiv.org/pdf/2604.07514v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 158. VSAS-BENCH: Real-Time Evaluation of Visual Streaming Assistant Models

**arXiv ID:** 2604.07634 | [PDF](https://arxiv.org/pdf/2604.07634v1)

**作者:** Pavan Kumar Anasosalu Vasu `[一作]` (Apple), Hadi Pouransari `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向实时视觉助手的 Streaming Vision‑Language 模型评估框架 VSAS，并提供了 92 条视频、18k 稠密注解的数据集，定义同步与异步评估协议以及精度与一致性指标。

**💡 创新点**

创新点在于引入异步、时序感知评估、稠密自由形式注解，以及无需额外训练即可把普通视频 VLM 转化为可流式工作的模型，并展示小模型在实时场景下优于专门调优的大模型。

**🔧 技术方法**

使用的技术包括基于记忆缓冲区和访问策略的流式适配、LLM 判别器（GPT‑5）评估相似度、GPU 加速（H100/A100）等。

**📊 数据集**

使用的数据集由 92 条视频组成（48 新录制 + 44 来自 STAR、PerceptionTest、YouCook2 等），每秒一帧共 18k 纯文本注解。

**📈 对比分析**

通过同步/异步协议对比 12+ 模型，异步下小模型 Qwen3‑VL‑4B 以 3% 领先 Dispider；在同步协议中大模型表现更好，展示了速度‑准确度权衡。

**⚠️ 局限性**

局限包括：评估依赖 GPT‑5 判别器、数据集规模相对有限、只考虑视频输入不含音频，且异步协议仍假设固定摄像头帧率和缓冲区大小。

---

## 159. Munkres' General Topology Autoformalized in Isabelle/HOL

**arXiv ID:** 2604.07455 | [PDF](https://arxiv.org/pdf/2604.07455v1)

**作者:** Dustin Bryant `[一作]` (Independent), Josef Urban `[通讯]` (AI4REASON and University of Gothenburg / Chalmers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型自动化完成了 Munkres《Topology》通用拓扑章节在 Isabelle/HOL 中的正式化，共计 85,472 行代码、806 条定理且无剩余缺失。

**💡 创新点**

创新点在于提出了 “sorry-first” 声明式证明工作流并配合批量 sledgehammer 调用，使得在 24 天内实现了完整、无待证明空洞的大规模正式化。

**🔧 技术方法**

技术手段包括 ChatGPT 5.2 与 Claude Opus 4.6 的自动 tmux 交互、Isabelle/HOL 及其强大证明自动化工具（sledgehammer、blast、auto、simp 等）。

**📊 数据集**

使用的训练与验证数据集为 Munkres 书籍的 LaTeX 源码（第 2–8 章共 7,956 行），其为正式化的原始信息来源。

**📈 对比分析**

通过与 Megalodon、HOL Light 与 Naproche 等系统在同一教材上的对比，显示 Isabelle 版本在 24 天内完成 85k 行、0 sorry、覆盖 39 节的工作，其代码长度约为 Megalodon 的一半、剩余空洞更少，证明效率显著提升。

**⚠️ 局限性**

局限性包括定义弱化、部分定理陈述不完整、过度分解的辅助命题导致冗长、与 Isabelle 现有类型类拓扑库的整合不足，以及对 LLM 输出进行人工修正的必要性。

---

## 160. Semantic-Emotional Resonance Embedding: A Semi-Supervised Paradigm for Cross-Lingual Speech Emotion Recognition

**arXiv ID:** 2604.07417 | [PDF](https://arxiv.org/pdf/2604.07417v1)

**作者:** Ya Zhao `[一作]` (Xinjiang University), Liejun Wang `[通讯]` (Xinjiang University)

**通讯引用:** 3183 | [OpenAlex ID](https://openalex.org/A5081489939)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种半监督的跨语种语音情感识别框架 SERE，利用情感-语义共振嵌入实现无需目标语言标签或对齐即可迁移情感表达。

**💡 创新点**

创新点包括：①即时共振场（IRF）捕捉瞬时情感爆发并自组织无标签样本；②三重共振交互链（TRIC）损失在全局原型、源域内部和跨语种三层次加强情感聚类；③使用语言异构编码器提升跨语言特征可迁移性。

**🔧 技术方法**

核心技术包括：即时动态特征提取器（IDFE）从语音的基频、响度、音色中生成瞬时动态特征；即时共振场对源目标高亮帧进行相似性匹配；三重共振交互链损失实现全局原型、双实例共振；使用预训练语音模型（如 Whisper、WavLM、Hubert、Wav2Vec2）做异构编码。

**📊 数据集**

使用四大公开情感语音数据库：EmoDB（德语）、eNTERFACE（英语）、CASIA（中文）和 EMOVO（意大利语），共 12 语种跨域任务进行评估。

**📈 对比分析**

与现有领域自适应基线（DAN、AaD）以及多种对齐方法对比，SERE 在 12 任务中平均 UAR 47.75%，在 9 任务上领先，对 C→E、E→C、E→O、O→E 等难题实现显著提升。

**⚠️ 局限性**

局限包括：对情感表达差异较大的语言（如德语与意大利语）仍存在误判；对低资源语言的跨文化差异仍需进一步研究；目前框架依赖预训练模型，若无足够语音资源会受限。

---

## 161. The Role of Emotional Stimuli and Intensity in Shaping Large Language Model Behavior

**arXiv ID:** 2604.07369 | [PDF](https://arxiv.org/pdf/2604.07369v1)

**作者:** Ameen Patel `[一作]` (Irvington High School), Joseph Thomas `[通讯]` (California High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究四种情绪（喜悦、鼓励、愤怒、不安全感）在情绪化提示中的效果，并通过GPT‑4o mini生成提示，评估其对准确性、谄媚性与毒性三项指标的影响。

**💡 创新点**

首次系统评估多种情绪与强度对提示效果的影响，构建情绪提示生成管线和“Gold Dataset”，并在多任务基准上比较基线与情绪增强提示。

**🔧 技术方法**

利用提示工程、GPT‑4o mini的few‑shot和zero‑shot推理、情绪检测管线、Fleiss Kappa一致性评估，并在Anthropic SycophancyEval与Real‑Toxicity‑Prompts上进行实验。

**📊 数据集**

Anthropic SycophancyEval子集、Real‑Toxicity‑Prompts 100K 条毒性语句、415 条 GPT‑4o mini 生成提示以及手工挑选的 Gold Dataset。

**📈 对比分析**

对比基线提示与加入情绪词的增强提示，计算平均准确率、谄媚评分与毒性评分的百分比变化；结果显示正面情绪略提升准确率并显著降低毒性，但也提升谄媚；负面情绪变化不大，LLM生成提示对各指标的影响比人类提示更显著。

**⚠️ 局限性**

实验全部使用同一模型 GPT‑4o mini，存在方法学循环性与模型偏差传递风险；缺乏统计显著性检验；结论可能仅适用于该架构，需跨模型验证。

---

## 162. SHIELD: A Segmented Hierarchical Memory Architecture for Energy-Efficient LLM Inference on Edge NPUs

**arXiv ID:** 2604.07396 | [PDF](https://arxiv.org/pdf/2604.07396v1)

**作者:** Jintao Zhang `[一作]` (National University of Singapore), Xuanyao Fong `[通讯]` (National University of Singapore)

**通讯引用:** 3079 | [OpenAlex ID](https://openalex.org/A5085588788)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出一种分段层次刷新控制的 eDRAM 内存架构 SHIELD，以提升边缘 NPU 上大模型推理的能效。

**💡 创新点**

创新点在于将 BF16 激活分离为符号/指数和尾数，并根据生命周期（QO 临时 vs KV 持久）与误差容忍度实现刷新分层：临时 QO 尾数无刷新，持久 KV 尾数使用放宽刷新，显著降低刷新能耗。

**🔧 技术方法**

采用 eDRAM、生命周期感知刷新控制器、BF16 逐位分段存储、误差注入模拟和能耗模型评估等技术。

**📊 数据集**

在多种开源 LLM（Qwen、Mistral、Llama 等）以及 WikiText‑2、PIQA、ARC‑Easy 等基准数据集上进行评估。

**📈 对比分析**

与传统 eDRAM 刷新以及 Kelle KV‑only 刷新方案对比，SHIELD 在各种工作负载下实现约 35% 的刷新能耗下降，同时保持准确率与基准几乎无差异。

**⚠️ 局限性**

限制在于目前仅支持 BF16；对其他数值格式或更复杂的激活生命周期分层的适配尚未验证，且实现细节对硬件成本与延迟的影响仍需进一步研究。

---

## 163. Sheaf-Laplacian Obstruction and Projection Hardness for Cross-Modal Compatibility on a Modality-Independent Site

**arXiv ID:** 2604.07632 | [PDF](https://arxiv.org/pdf/2604.07632v1)

**作者:** Tibor Sloboda `[一作]` `[通讯]` (Slovak Technical University), Tibor Sloboda (Slovak Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的框架，用同一条基于样本索引的邻域图（modality‑independent site）来衡量不同模态之间的可兼容性，定义并计算两种互补的不可兼容性度量：投影硬度（Hardness）和Sheaf‑Laplacian 障碍（Obstruction）。

**💡 创新点**

创新点在于：① 用单一图结构消除模态间可比性问题；② 将全局可行性与局部可拼接性分离，形成两条清晰的失败模式；③ 通过投影参数层的 sheaf Laplacian 能直接将光滑度正则化映射到可计算的能量；④ 通过构造证明非传递性和桥接（bridging）效应，展示跨模态映射的阶段性可行性。

**🔧 技术方法**

技术手段包括：细胞层 sheaf 理论、图 Laplacian 与 Dirichlet 能量、Lipschitz 控制的投影族（线性、低秩线性、宽度受限 MLP）、投影参数 sheaf 的能量正则化、Poincaré 不等式与谱间隙分析、ReLU 单隐藏层网络的分段线性特性构造示例。

**📊 数据集**

论文未使用具体公开数据集；主要采用合成设定（如隐式潜在空间的 k‑NN 图）或理论构造（例如一维 ReLU 例子、两簇符号翻转模型）。

**📈 对比分析**

比较方法：对每个模态对使用相同的基图、白化、投影族和误差指标，计算投影硬度 H_a→b(ε) 与障碍能量 C_a→b(ε)，从而在可比度阈值下判断兼容/不兼容；论文未给出实验性能数值，主要提供理论上限和稳定性分析。

**⚠️ 局限性**

限制包括：① 需要预先固定且对所有模态统一的图结构，若图不代表真实语义邻域会影响结果；② 投影参数 sheaf 采用恒等限制，只度量参数平滑度而非特征传输；③ 投影族选择固定且有限，无法覆盖所有可能的映射；④ 目前仅给出理论与合成示例，缺乏大规模实测验证。

---

## 164. Safe Large-Scale Robust Nonlinear MPC in Milliseconds via Reachability-Constrained System Level Synthesis on the GPU

**arXiv ID:** 2604.07644 | [PDF](https://arxiv.org/pdf/2604.07644v1)

**作者:** Jeffrey Fang `[一作]` (Georgia Institute of Technology), Glen Chou `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出GPU-SLS框架，将安全、鲁棒的非线性模型预测控制（NMPC）与系统级合成（SLS）在GPU上并行化，实现大规模高维机器人系统的实时控制。

**💡 创新点**

创新点包括：① 利用ADMM与关联扫描、因式分解缓存实现对LTV-QP的对数深度求解；② 将SLS与NMPC联合并行化，形成统一的实时迭代流程；③ 在RTI（实时迭代）框架下，完成毫秒级求解，显著提升实时性能。

**🔧 技术方法**

核心技术：GPU并行ADMM、关联扫描、因式分解缓存、系统级合成（SLS）、SQP、实时迭代（RTI）以及硬件加速的线性二次调节器（LQR）求解。

**📊 数据集**

实验数据集与平台：10关节摆、n链摆、61维四足模型、75维人形机器人、Dubins车、6维平面四旋翼以及Unitree Go2硬件实验。

**📈 对比分析**

与OSQP、HPIPM、iLQR AL、FastSLS、DeepReach等基线比较；GPU-SLS在长时间程下实现99.8%+的速度提升，FastSLS上可达237×；在实时硬件上平均求解时间约20 ms；安全率达到100%，显著优于传统与数据驱动方法。

**⚠️ 局限性**

局限性：仍依赖于局部线性化，极端非线性或不可微约束场景可能受限；需要高性能GPU硬件支持；单次RTI迭代可能导致收敛误差，需要更多迭代以获得更精确解。

---

## 165. ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training

**arXiv ID:** 2604.07484 | [PDF](https://arxiv.org/pdf/2604.07484v1)

**作者:** Yu Liang `[一作]`, Daiting Shi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

示例说明如何使用ACL样式文件与LuaLaTeX或XeLaTeX进行编译和排版。

**💡 创新点**

演示多语言文本展示和样例文件结构，展示ACL样式的使用方法。

**🔧 技术方法**

使用LuaLaTeX或XeLaTeX编译器与ACL风格文件。

**📊 数据集**

无具体数据集，仅包含示例文本。

**📈 对比分析**

未进行方法比较或性能评估。

**⚠️ 局限性**

仅为模板示例文件，缺乏实验验证与实际应用场景，功能受限于演示用途。

---

## 166. The Cartesian Cut in Agentic AI

**arXiv ID:** 2604.07745 | [PDF](https://arxiv.org/pdf/2604.07745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 167. Designing Annotations in Visualization: Considerations from Visualization Practitioners and Educators

**arXiv ID:** 2604.07691 | [PDF](https://arxiv.org/pdf/2604.07691v1)

**作者:** Md Dilshadur Rahman `[一作]` (University of Utah), Paul Rosen `[通讯]` (University of Utah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

对可视化注释设计的实践进行系统研究，访谈了10名从业者与7名教育者，归纳出六个核心设计考虑因素，并构建了案例展示的注释设计图库。

**💡 创新点**

首次将注释设计过程中的隐性专业知识与决策逻辑系统化呈现，补充了以往仅关注注释视觉形式的研究，提供了可操作的设计指导。

**🔧 技术方法**

采用定性研究方法：半结构化访谈与内容分析；并结合公开的专业注释实例数据集进行案例映射。

**📊 数据集**

未使用实验数据集；利用公开的专业注释实例数据集（https://vis-annotations.github.io/annotation-design/）作为案例示例。

**📈 对比分析**

研究主要为定性比较，未进行量化性能评估；通过访谈结果与案例对照，验证了六个设计考虑在不同场景下的适用性。

**⚠️ 局限性**

研究样本有限（10名从业者、7名教育者），缺乏大规模量化验证；受访者主要来自特定文化背景，可能不完全代表全球实践；并且未探讨技术实现细节与工具支持。

---

## 168. Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity

**arXiv ID:** 2604.07402 | [PDF](https://arxiv.org/pdf/2604.07402v1)

**作者:** Yucheng Zhou `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**通讯引用:** 16147 | [OpenAlex ID](https://openalex.org/A5023184215)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过经验分析提出了加速自回归视频生成模型训练的方法；

**💡 创新点**

创新点在于引入局部优化（Local Optimization）和表示连续性（Representation Continuity）两种训练策略，以减轻错误累积并提升视频一致性；

**🔧 技术方法**

使用了自回归模型、VQ‑VAE视频分块、局部窗口训练、Lipschitz连续性正则化以及光流、PSNR评估指标；

**📊 数据集**

实验数据集包括FFS、SKY、UCF101和Taichi-HD等；

**📈 对比分析**

与基线及多种GAN、扩散和自回归模型对比，所提方法在保持或提升视频质量（FVD、PSNR、光流）的同时，训练速度提升约1.7–2倍；

**⚠️ 局限性**

局限性在于主要针对训练效率提升，未在商业规模大语言模型上进行实验，且方法在不同网络结构或更大模型上仍需验证。

---

## 169. EMSDialog: Synthetic Multi-person Emergency Medical Service Dialogue Generation from Electronic Patient Care Reports via Multi-LLM Agents

**arXiv ID:** 2604.07549 | [PDF](https://arxiv.org/pdf/2604.07549v1)

**作者:** Xueren Ge `[一作]` (University of Virginia), Homa Alemzadeh `[通讯]` (University of Virginia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个基于电子病历(ePCR)的多代理迭代生成管道，生成高质量、多方角色、主题流一致的 EMS 对话，并发布了 4,414 条合成对话数据集 EMSDialog；随后用该数据集训练对话式诊断预测模型，验证其在真实 EMS 对话上的有效性。

**💡 创新点**

① 多代理循环（规划→生成→精炼）结合判别器；② 通过规则化的概念检验器、主题流检查器和 LLM 风格检查器保证事实性、流程合理性和自然性；③ 生成的数据集兼具多方对话、诊断标签、角色与主题标注，填补现有医学对话语料在多方工作流和诊断注释上的空白。

**🔧 技术方法**

使用 LLM（如 Qwen3、Llama-3.3）完成规划、生成与精炼；MedSpaCy+QuickUMLS 提取概念；GatorTron 进行语义匹配；图结构（主题流图）验证对话顺序；LLM 作为风格评判者；循环迭代直到满足所有规则；最终利用 LoRA 对 LLM 进行微调以做诊断预测。

**📊 数据集**

核心数据集：真实 ePCR（4,417 条 EMS 报告）作为知识源；合成数据集 EMSDialog（4,414 条对话，43 种诊断、角色、主题）；对照数据：149 条真实 EMS 对话（训练/测试划分）及多种基线合成对话（NoteChat、DDXPlus 等）。

**📈 对比分析**

通过内部评估（对话层逻辑、事实性、真实性、角色准确度等）和外部评估（对话式诊断预测的首次/最后一次准确率、早期性、编辑开销）与多种基线（0-shot、CoT、NoteChat、其他合成方法）对比。EMSDialog 在逻辑结构、事实性、真实性、诊断预测准确率和早期性等指标上均优于基线；与真实数据合并训练得到最优性能。

**⚠️ 局限性**

仅在 EMS 领域验证，方法难以直接迁移到其他医疗场景；大部分质量评估依赖 LLM 判断，人工验证样本有限；合成数据可能保留 ePCR 中的偏差、错误或缺失信息；隐私泄露风险和模型产生不安全诊断建议仍需临床验证与人机监督。

---

## 170. From LLM to Silicon: RL-Driven ASIC Architecture Exploration for On-Device AI Inference

**arXiv ID:** 2604.07526 | [PDF](https://arxiv.org/pdf/2604.07526v1)

**作者:** Ravindra Ganti `[一作]` (XgenSilicon Inc.), Steve Xu `[通讯]` (XgenSilicon Inc.)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端到端的RL驱动编译器，联合优化AI推理ASIC的计算架构、存储层级和工作负载划分，并直接生成可上光的RTL/GDSII文件。

**💡 创新点**

创新点在于将所有设计维度统一为单一MDP，采用混合离散-连续动作空间，并使用Soft Actor‑Critic加Mixture‑of‑Experts策略；同时实现了异构每核参数推导、操作级划分、跨工艺节点自动重定向及全流程自动化，无需手工调参。

**🔧 技术方法**

技术手段包括Soft Actor‑Critic（SAC）与优先经验回放、MPC计划、Surrogate PPA模型、操作级划分、流程图优化、节点级约束投影，以及自适应探索衰减与Pareto前沿管理。

**📊 数据集**

实验数据集主要是Llama 3.1 8B FP16推理模型和SmolVLM视觉‑语言模型（均以ONNX导入），并在7个不同工艺节点（3 nm至28 nm）进行评估。

**📈 对比分析**

通过与行业GPU/ASIC基准（H200、B200、Groq、SambaNova、Cerebras、Taalas）以及随机/网格搜索对比，验证了在3 nm节点下Llama 3.1 8B可达≈29 k tokens/s、51 W、580 mm²，PPA得分显著优于传统搜索；在SmolVLM低功耗模式下，7个节点均实现<13 mW、10‑14 tokens/s的性能。

**⚠️ 局限性**

局限性包括：仅验证了两种Transformer类模型，缺乏多种模型（CNN、扩散模型、MoE）验证；单次RL种子导致未给出统计置信区间；仅支持2D网格拓扑；结果高度依赖奖励权重与工艺约束；未对非离散动作空间的拓扑进行探索。

---

## 171. Have LLM-associated terms increased in article full texts in all fields?

**arXiv ID:** 2604.07565 | [PDF](https://arxiv.org/pdf/2604.07565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 172. MEV-ACE: Identity-Authenticated Fair Ordering for Proposer-Controlled MEV Mitigation

**arXiv ID:** 2604.07568 | [PDF](https://arxiv.org/pdf/2604.07568v1)

**作者:** Jian Sheng Wang `[一作]` `[通讯]` (ACE Labs), Jian Sheng Wang (ACE Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了MEV-ACE协议，旨在通过身份认证和受限提交实现区块生产者对交易顺序的公平控制，从而消除MEV。

**💡 创新点**

创新点在于将ACE-GF身份框架与VDF随机延迟、阈值收据相结合，实现可审计的可接受性、不可预测的排序和可证明的包含性，并通过质押罚金对恶意行为进行经济约束。

**🔧 技术方法**

采用了ACE-GF的身份绑定签名、VDF产生随机种子、阈值签名收据、SHA-256哈希和可证明的延迟随机性等技术。

**📊 数据集**

该工作没有使用传统意义上的数据集，而是在理论模型和安全分析中验证协议属性，并在性能分析中给出通信与计算成本估算。

**📈 对比分析**

通过与传统的commit‑reveal、阈值加密、PBS等方案进行对比，指出MEV‑ACE不需要解密委员会、单槽完成且兼容后量子签名，性能上主要受VDF延迟和收据签名开销影响。

**⚠️ 局限性**

局限性包括仅在收据阈值达成后才生效、无法解决信息型MEV、依赖VDF硬件时间保证、对经济参数高度敏感以及需要及时传播缺失证明。

---

## 173. DCD: Domain-Oriented Design for Controlled Retrieval-Augmented Generation

**arXiv ID:** 2604.07590 | [PDF](https://arxiv.org/pdf/2604.07590v1)

**作者:** Valeriy Kovalskiy `[一作]` (red mad robot), Max Maximov `[通讯]` (red mad robot)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 DCD（Domain–Collection–Document）架构，通过显式知识层次划分和多阶段路由来改进 RAG 系统，降低跨域干扰并提升检索质量。

**💡 创新点**

创新点在于引入三层知识层级（Domain、Collection、Document）、基于 LLM 结构化输出的 DCD Router 进行动态路由、智能滑动窗口分块、混合检索以及快速安全门控机制，实现对查询流程的细粒度控制。

**🔧 技术方法**

使用的技术包括 LLM 结构化输出（JSON）、向量检索（ChromaDB + bge-m3 embedding）、滑动窗口分块、混合检索、基于 LLM 的评估与 guardrails 以及多阶段路由流程。

**📊 数据集**

使用的实验数据集为合成的多域多集合文本数据，包含 10 个住宅小区域，每个域下有基础设施、安全、描述等子集合，生成了对应的 Q&A–Context 评估集。

**📈 对比分析**

与 Naive RAG 基线对比时，DCD 在 Context Recall（0.95 vs 0.59）、Factual Accuracy（0.89 vs 0.40）以及 Retrieval Coverage Score（1.76 vs 1.28）等指标上均显著提升，答案相关性保持相近。

**⚠️ 局限性**

局限性包括配置复杂度随知识库规模增长、对完全无结构数据的适用性有限以及在高度相似模板化数据上可能导致的计算开销增大。

---

## 174. Benchmark Shadows: Data Alignment, Parameter Footprints, and Generalization in Large Language Models

**arXiv ID:** 2604.07363 | [PDF](https://arxiv.org/pdf/2604.07363v1)

**作者:** Hongjian Zou `[一作]` (Vivo AI Lab), Xiaoxin Chen `[通讯]` (Vivo AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究了数据分布（训练制度）对大语言模型学习动态与能力的影响，设计了控制实验、参数空间谱与秩诊断，并在多模态模型中进行外部验证，最后通过 Prompt 去重案例探讨冗余与集中效应的区别。

**💡 创新点**

创新点包括：①提出“训练制度”框架，将数据分布视为决定学习路径的关键因子；②利用权重谱指数、有效秩、变化方差等参数空间诊断，揭示不同数据制度的内部结构差异；③在多模态模型中验证“基准阴影”非单纯污染现象，而是由训练制度导致的表征收缩或扩张。

**🔧 技术方法**

技术手段主要为：Transformer 仅解码器（0.6B）配合 RMSNorm、RoPE、SwiGLU；参数空间诊断工具 WeightWatcher（谱指数 α、有效秩、变异方差、Δ有效秩）；阶段性训练（覆盖扩展/频率集中/重复集中/优化控制）；多模态模型（Qwen3-VL、InternVL、AndesVL）与基准评测。

**📊 数据集**

使用了覆盖扩展（多域、多任务、多格式）数据集；对比 10% 重复集中和 频率集中两种制度；Prompt 重复与去重数据；以及开源多模态模型自带的数据集（Qwen3-VL、InternVL、AndesVL 等）进行外部验证。

**📈 对比分析**

对比方法：在相同模型与优化设置下，计算权重谱指数比例、有效秩分布、变化方差曲线，评估不同训练制度的参数结构；在多模态基准（MMMU、MathVision、ChartQA、DocVQA 等）上比较性能分布，发现覆盖扩展制度在推理类基准表现更好，而频率集中制度在感知类基准表现更强，说明能力分布不均。

**⚠️ 局限性**

局限性：①实验仅在单一规模（0.6B）Transformer 上验证，缺乏对更大模型和不同架构的推广；②外部验证为相关性分析，缺乏因果证明；③谱诊断与梯度/训练动态的理论联系仍不充分；④Prompt 去重案例只探讨了单一去重阈值，重复率与制度形成的关系未系统探索；⑤所用数据集仍未覆盖所有多模态场景。

---

## 175. Conservation Law Breaking at the Edge of Stability: A Spectral Theory of Non-Convex Neural Network Optimization

**arXiv ID:** 2604.07405 | [PDF](https://arxiv.org/pdf/2604.07405v1)

**作者:** Daniel Nobrega Medeiros `[一作]` `[通讯]` (University of Colorado Boulder), Daniel Nobrega Medeiros (University of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究梯度下降在无偏置的 ReLU 网络中如何通过保守量的破缺与谱交叉公式解释其在非凸损失地形中可靠找到好解的机制。

**💡 创新点**

提出梯度保守量 (L-1 个) 在梯度流中保持不变，离散梯度下降则按 ^2 的尺度破坏这些保守量，构成精确的漂移分解；同时给出闭式谱交叉公式解释非整数漂移指数，揭示交叉熵自正则化导致指数谱压缩并保持漂移指数≈1。

**🔧 技术方法**

利用梯度流与离散梯度下降理论、谱分解（Hessian 以及数据协方差特征值）、Gauss‑Newton 近似、NTK 理论、以及对 2 层线性与 ReLU 网络的闭式分析。

**📊 数据集**

主要使用人工合成数据（特征协方差谱可控）以及在 23 次实验中覆盖多种宽度、学习率和损失函数的设置；并没有使用公开大规模真实数据集。

**📈 对比分析**

与传统 NTK、过参数化理论相比，本文通过实验证明漂移指数在 1.1–1.6 之间随宽度与学习率变化，交叉熵在任何宽度下都保持约 1.0–1.1；实验覆盖四个数量级学习率，R²>0.99 的拟合；表明理论预测与实验高度一致。

**⚠️ 局限性**

局限性包括：仅分析 2 层网络（更深层需进一步推广）；在边缘稳定性 (EoS) 时模式耦合严重，独立模式假设失效；对保守量破缺在更复杂网络结构（卷积、残差）中的适用性尚未验证；实验数据以人工合成为主，缺乏对真实任务的直接验证。

---

## 176. An Analysis of Artificial Intelligence Adoption in NIH-Funded Research

**arXiv ID:** 2604.07424 | [PDF](https://arxiv.org/pdf/2604.07424v1)

**作者:** Navapat Nananukul `[一作]` (University of Southern California), Mayank Kejriwal `[通讯]` (University of Southern California)

**通讯引用:** 1442 | [OpenAlex ID](https://openalex.org/A5074197492)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过构建人机协作的大语言模型(LLM)流程，对2025财年58,746项NIH资助的生物医学研究项目进行自动化分类、结构化编码、网络分析，并系统性地评估AI/ML在NIH投资中的普遍性、资金差异、临床部署缺口以及机构间合作模式。

**💡 创新点**

创新点在于①提出了两步LLM管道——先进行AI筛选，再用固定结构编码提炼关键信息；②将LLM与人工校正相结合，实现大规模可复现的投资组合洞察；③通过社区检测与网络科学手段，揭示AI研究的中心/桥接机构，暴露合作不均衡与健康公平缺口。

**🔧 技术方法**

核心技术包括：GPT‑4o‑mini的零样本分类与结构化推理；基于Prompt设计的JSON输出规范；Python脚本自动化数据拼接、统计与可视化；图网络算法（Louvain社区检测、度/介数中心性、度分布拟合）用于协作网络分析。

**📊 数据集**

使用的数据集为NIH Research Performance and Reporting System (RePORTER) 2025年度项目元数据（58,746项），包含标题、摘要、项目术语、资助金额、机构信息等；随后生成的AI筛选结果与结构化编码结果用于后续分析。

**📈 对比分析**

对照方法主要是传统关键词匹配和手工抽样审核。LLM筛选在95%+的人工审核精度下识别AI项目，筛选率为15.9%；结构化编码的覆盖度提升至约82%（将“Other”疾病归类从41.5%降至17.7%），显著提高了主题多样性与数据完整性；网络指标（度分布、中心性）量化合作结构，未给出传统对照，但展示了相对稀疏的桥接层。

**⚠️ 局限性**

局限性包括：①LLM分类对模糊/新颖术语仍可能产生误判，需人工校正；②RePORTER数据缺乏子奖项层面细节，合作网络基于项目共现而非正式协作记录；③研究仅聚焦单一财年，未考虑时间演化趋势；④未对AI技术成熟度、伦理风险等进行深度评估，主要聚焦投资与部署比例。

---

## 177. Monocular Depth Estimation From the Perspective of Feature Restoration: A Diffusion Enhanced Depth Restoration Approach

**arXiv ID:** 2604.07664 | [PDF](https://arxiv.org/pdf/2604.07664v1)

**作者:** Huibin Bai `[一作]` (Shandong University), Xingyu Gao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种从特征恢复视角出发的单目深度估计框架IID‑RDepth，利用可逆解码器驱动的间接扩散模块对高层编码器特征进行恢复，并通过辅助视角增强低层细节。

**💡 创新点**

创新点在于将深度估计视为特征恢复问题，设计了满足双Lipschitz条件的可逆变换增强间接扩散（InvT-IndDiffusion），解决了间接监督导致的特征偏移；同时引入可插拔的辅助视角低层特征增强模块（AV-LFE）。

**🔧 技术方法**

核心技术包括可逆神经网络（Affine Coupling层）、扩散模型（DDPM/latent diffusion）与双Lipschitz约束、辅助视角对齐与融合（可变形卷积+融合卷积）以及自监督的SiLog损失。

**📊 数据集**

实验数据集主要使用KITTI和DDAD的户外稀疏深度数据。

**📈 对比分析**

与多种SOTA方法（PixelFormer、P3Depth、Depthformer等）对比，IID‑RDepth在KITTI上RMSE从2.081提升至1.295（+37.77%），AV‑LFE兼容模式进一步降至1.722；在DDAD上也均有显著提升，且在长距离预测上RMSE下降约9%。

**⚠️ 局限性**

局限性包括对辅助视角的依赖、扩散步骤导致推理速度慢、对稀疏标注的处理仍不完善，以及模型在室内或不同光照条件下的泛化尚未充分验证。

---

## 178. SMFD-UNet: Semantic Face Mask Is The Only Thing You Need To Deblur Faces

**arXiv ID:** 2604.07477 | [PDF](https://arxiv.org/pdf/2604.07477v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 179. SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization

**arXiv ID:** 2604.07663 | [PDF](https://arxiv.org/pdf/2604.07663v1)

**作者:** Wooin Lee `[一作]`, Hyun-Tae Kim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新型优化器SAGE，用于解决大语言模型训练中嵌入层的内存瓶颈和梯度高方差问题；

**💡 创新点**

创新点在于将Lion的单一动量状态与O(d)自适应阻尼器相结合，构建了一个可证明受限于1.0的安全阻尼尺度，既保持了低内存占用，又能抑制高方差梯度；

**🔧 技术方法**

采用了自适应阻尼算法（EMA求平均绝对梯度、相对RMS比值）、Lion风格的符号更新、SinkGD的单状态统计、AdamW的分离权重衰减等技术；

**📊 数据集**

在The Pile数据集（约6.6B个token）上使用bfloat16精度进行LLM预训练，模型规模分别为270M、0.6B和1.3B；

**📈 对比分析**

与AdamW、Lion、APOLLO、SinkGD纯/混合等基线进行比较，SAGE-Hybrid在所有规模下均实现最低的perplexity，且显著降低了优化器状态内存（相比AdamW可节省约50%），在1.3B模型上达到24.33的最优perplexity；

**⚠️ 局限性**

局限性包括实验仅覆盖至1.3B规模，未验证更大模型（7B+）的效果；训练时长受限，可能低估了长期训练中的优势；未对其他模态或Fine-tune任务进行评估；

---

## 180. TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense

**arXiv ID:** 2604.07727 | [PDF](https://arxiv.org/pdf/2604.07727v1)

**作者:** Cheng Liu `[一作]` (National Interdisciplinary Research Center of Engineering Physics), Kangyi Ding `[通讯]` (National Interdisciplinary Research Center of Engineering Physics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TrajGuard，一种无训练、在解码时实时监测 LLM 隐藏状态轨迹以拦截 jailbreak 攻击的防御框架。

**💡 创新点**

创新点在于利用解码过程中的隐藏状态轨迹（而非仅靠静态文本或单步检测）来捕获风险漂移，采用滑动窗口聚合、指数加权移动平均与持久阈值触发的分层检测，再通过 LLM 进行语义裁决，既保证低延迟又提升检测精度。

**🔧 技术方法**

技术包括：在关键层提取隐藏状态，计算 Mahalanobis 距离形成风险对比；使用滑动窗口 + EWMA 对风险做平滑；持久触发阈值与 hysteresis；当触发时调用 LLM 进行安全判定；整体不需额外训练。

**📊 数据集**

使用的公开数据集：HarmBench（生成 jailbreak 提示与响应）、XSTest、Alpaca、对 12 种 jailbreak 攻击（GCG、AutoDAN、PAIR、GPTFuzzer 等）以及各种公开 LLM（Llama‑2‑7B‑Chat、Llama‑3.1‑8B‑Instruct、Mistral‑7B‑Instruct、Vicuna‑7B）。

**📈 对比分析**

与 Llama Guard 3、Self‑Guard、Goal Prioritization、Qwen3Guard 等基线比较，TrajGuard 在 4 种目标模型上平均防御成功率达 95%，攻击成功率显著降低；延迟仅 5.2 ms/token，误报率低于 1.5%；同时保持几乎零触发频率，展示了优秀的性能与效率平衡。

**⚠️ 局限性**

局限性：对完整白盒攻击者可能通过优化隐藏状态轨迹实现规避；依赖预先估计的安全/恶意分布，特殊领域或新攻击需重新校准；需要访问内部隐藏状态，仅适用于开源模型，无法直接应用于封闭 API。

---

## 181. AITH: A Post-Quantum Continuous Delegation Protocol for Human-AI Trust Establishment

**arXiv ID:** 2604.07695 | [PDF](https://arxiv.org/pdf/2604.07695v1)

**作者:** Zhaoliang Chen `[一作]` `[通讯]` (University of Macau), Zhaoliang Chen (University of Macau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 AITH（AI Trust Handshake），一种面向 AI 代理的后量子连续委托协议，支持一次性签名后持续、可撤销且可审计的权限授权；

**💡 创新点**

创新点包括：① 采用 ML-DSA‑87 生成一次性委托证书，消除每次操作的加密开销；② 设计了六项检查的边界引擎，实现 0.21 µs 的实时约束校验；③ 引入推送式即时撤销机制（<1 s）和三层 SHA‑256 哈希链的责任链；④ 对协议进行 Tamarin 形式化验证；

**🔧 技术方法**

核心技术包括后量子签名 ML‑DSA‑87、哈希链（SHA‑256）、边界引擎实现、推送式撤销消息、三层责任链日志、Tamarin 形式化验证；

**📊 数据集**

使用了 100,000 条模拟操作数据（分布涵盖查询、交易、转账等），并在四个前沿 LLM（Model α/β/γ/δ）上进行结构化对抗审计；

**📈 对比分析**

与 OAuth‑2.0、Macaroon 等传统授权方式对比，AITH 在每操作延迟上低 10–100 倍（0.21 µs vs 1–10 µs），吞吐量可达 4.7 M ops/s（单核），撤销传播在 1 s 内完成；

**⚠️ 局限性**

局限性包括：① 未处理 AI 对齐与语义正确性问题；② 侧信道攻击与硬件级安全未覆盖；③ 约束完整性仍需人工设定；④ 依赖于网络时延与推送基础设施，极端网络条件下撤销延迟可能升高。

---

## 182. Bayesian Optimization for Mixed-Variable Problems in the Natural Sciences

**arXiv ID:** 2604.07416 | [PDF](https://arxiv.org/pdf/2604.07416v1)

**作者:** Yuhao Zhang `[一作]` (Aalto University), Patrick Rinke `[通讯]` (Technical University Of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Bayesian优化中的概率重参数化（PR）方法进行推广，支持离散型变量，并在混合变量搜索空间中实现梯度优化。

**💡 创新点**

创新点在于：①将PR框架扩展到非等距离离散变量；②在GP代理中引入了温度调控的概率分布，使离散变量在连续域内可微分；③提出了基于大值惩罚的重采样抑制策略和修改后的采集函数（mAF）以避免局部最优陷阱。

**🔧 技术方法**

技术手段包括：高斯过程（GP）作为代理模型；概率重参数化实现对连续、整数、离散、分类变量的统一映射；不同核（Matérn‑5/2、RBF、加性）与先验（Gamma、LogNormal、固定尺度）组合；采集函数为EI与LCB；梯度优化使用Adam；重采样惩罚与mAF的实现。

**📊 数据集**

数据集：①合成“Butternut Squash”函数（20种维度+离散级别组合）；②真实科学实验任务：化学合成（Chemistry）和形状记忆聚合物驱动器（Actuator）；③极端离散连续混合的“Discontinuous Unsmoothed Step-like Test”函数（DUST1、DUST2）。

**📈 对比分析**

与方法比较：对20个BS变体、Chemistry、Actuator、DUST1/2分别进行10次Sobol起始的10次实验。使用复合评分（收敛率与平均收敛迭代）与排名统计。结果显示：最优模型（产品Matérn‑5/2、Gamma先验、EI）在所有基准中均达到最高复合评分，优于原始PR、KR、RF等基线；在DUST1/2中加入惩罚和mAF后，性能可与RF持平甚至优于其。

**⚠️ 局限性**

局限性：①性能高度依赖问题特性（离散程度、维度、目标景观结构）；②对加性核的过度拟合导致在非加性问题上表现下降；③重采样惩罚在高噪声或连续变量混合时需进一步验证；④基准集仍缺乏更丰富的多样化合成函数，未来需要扩展到更广泛的场景；⑤对大规模高维问题的可扩展性尚未完全评估。

---

## 183. Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training

**arXiv ID:** 2604.07397 | [PDF](https://arxiv.org/pdf/2604.07397v1)

**作者:** Jinhong Lin `[一作]` (University of Wisconsin Madison), Pedro Morgado `[通讯]` (University of Wisconsin Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为扩散模型训练引入数据预热（Data Warmup）课程学习策略，使模型先学习简单图像，再逐步接触复杂图像。

**💡 创新点**

创新点在于提出仅通过离线语义感知复杂度度量（前景占比+前景典型性）来决定训练顺序，并使用温度控制的softmax实现无额外计算开销的动态采样。

**🔧 技术方法**

使用 DINO‑v2 特征提取、PCA 前景分离、k‑means 原型聚类、Sigmoid 校正和温度调度器，全部在训练前完成。

**📊 数据集**

主要在 ImageNet‑1K（256×256）及其子集 IN‑100、IN‑500、IN‑1K 上评估，同时在 SiT 系列（S/2–XL/2）骨干上测试。

**📈 对比分析**

与统一采样基线相比，Data Warmup 在 IS 上提升最多 6.11、FID 减少 3.41；逆向课程甚至比基线差；与 REPA 等模型加速器叠加可再获 2.72 IS、1.70 FID 的额外提升，且仅需约 10 分钟的离线预处理。

**⚠️ 局限性**

局限包括：度量固定不随训练更新；在样本量极小的子集（如 IN‑100）会导致过拟合；仅针对图像生成，未验证文本条件或无标签数据；若数据不够多样，课程可能适得其反。

---

## 184. SPAMoE: Spectrum-Aware Hybrid Operator Framework for Full-Waveform Inversion

**arXiv ID:** 2604.07421 | [PDF](https://arxiv.org/pdf/2604.07421v1)

**作者:** Zhenyu Wang `[一作]` (China University of Mining and Technology), Lei Zhang `[通讯]` (China University of Mining and Technology)

**通讯引用:** 107089 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全波形反演的新框架SPAMoE，能够有效分离并分别建模不同频率的地质信息，提升模型的分辨率与稳定性。

**💡 创新点**

核心创新包括：Spectral‑Preserving DINO Encoder确保高低频能量比不被压缩；Adaptive Spectral Mixture‑of‑Experts通过可微软频带分解、频率偏好机制与能量注意力路由实现频段自适应专家激活。

**🔧 技术方法**

技术手段包括ViT‑DINO自监督编码器、卷积/傅里叶变换的频域软掩码、注意力路由与多专家（FNO、MNO、LNO）融合。

**📊 数据集**

在OpenFWI benchmark的十个2D子数据集（CurveVel、FlatVel、CurveFault、FlatFault、Style，各A/B版本）上进行实验。

**📈 对比分析**

与官方基线(InversionNet、VelocityGAN、UPFWI)及FNO相比，SPAMoE在所有十个子数据集上均获得最佳成绩，平均MAE从0.0649降至0.0298（↓54.1%），RMSE下降42.1%，SSIM提升至0.9311。

**⚠️ 局限性**

限制方面：对极高频细节的恢复仍受限于训练样本的频谱覆盖；模型对不同地质类型的泛化能力仍需进一步验证；实验仅涉及二维模拟，三维扩展尚未探究。

---

## 185. When Switching Algorithms Helps: A Theoretical Study of Online Algorithm Selection

**arXiv ID:** 2604.07473 | [PDF](https://arxiv.org/pdf/2604.07473v1)

**作者:** Denis Antipov `[一作]` (Sorbonne Université), Carola Doerr `[通讯]` (Sorbonne Université)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文证明了在 OneMax 优化中，通过在 (1+λ) EA 与 (1+(λ,λ)) GA 之间进行在线算法选择，可实现 O(n log log n) 的期望收敛时间。

**💡 创新点**

创新点在于首次给出理论证明的 OAS 方法，并提出既能实现理论最优又能在实践中可行的切换策略，还将该思路推广到超启发式（hyper‑heuristic）范式。

**🔧 技术方法**

主要采用了固定目标与固定起点的分析技术，结合进化算法的理论运行时间公式，推导出切换时机与参数设置。

**📊 数据集**

未使用具体实验数据集，研究全部在 OneMax 函数的理论分析上进行。

**📈 对比分析**

通过与单独使用 (1+λ) EA（O(n log n)）以及最佳参数的 (1+(λ,λ)) GA（Θ(n√(log n logloglog n / loglog n))）的理论运行时间比较，证明 OAS 在期望时间上显著更优，达到 O(n log log n)。

**⚠️ 局限性**

局限性包括仅适用于 OneMax，未给出对更一般问题的推广，缺乏实验验证，并且只考虑单向切换（不回到前一算法）及固定大小的算法组合。

---

## 186. Flow Learners for PDEs: Toward a Physics-to-Physics Paradigm for Scientific Computing

**arXiv ID:** 2604.07366 | [PDF](https://arxiv.org/pdf/2604.07366v1)

**作者:** Yilong Dai `[一作]` (University of Alabama), Runlong Yu `[通讯]` (University of Alabama)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出一种基于流学习的PDE求解器框架，即“Flow Learners”，通过学习状态空间的输运向量场而非传统的一步状态回归，来实现连续时间预测与分布式不确定性传播。

**💡 创新点**

核心创新在于将学习目标从单点状态预测转向输运规律，强调物理到物理（physics‑to‑physics）的结构对齐，使求解器本身即为物理演化的连续动力学；同时提出相应的研究议程和评估标准。

**🔧 技术方法**

采用流匹配（flow matching）、神经ODE、神经算子、扩散模型等技术，结合向量场参数化、对称/守恒约束、主动学习与数据增强等方法实现输运学习。

**📊 数据集**

未给出具体公开数据集，论文在多种典型PDE基准（如气象预报、流体动力学、血流动力学等）上进行概念验证和讨论。

**📈 对比分析**

比较方式侧重长时序分布预测、物理一致性、置信度校准与计算成本，提出新的基准指标（如semigroup质量、物理残差、覆盖率等）；实验结果未给出数值，只给出未来评估方向。

**⚠️ 局限性**

主要限制包括对稀疏、分支未来的覆盖不足、在高维/复杂几何下的可扩展性、生成速度与计算开销、以及对物理一致性与校准的可审计性和可靠性验证等挑战。

---

## 187. Modeling and Analysis for Joint Design of Communication and Control

**arXiv ID:** 2604.07735 | [PDF](https://arxiv.org/pdf/2604.07735v1)

**作者:** Xu Gan `[一作]` (University of Hong Kong), Yuanwei Liu `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个统一的分析框架，用于通信与控制的联合设计（JDCC），并推导了通信传输延迟和稳态控制方差作为JDCC的基本性能指标。

**💡 创新点**

创新点在于建立了JDCC系统的Pareto边界，表征了通信与控制之间的最佳权衡，并在最大比率传输（MRT）和零强迫（ZF）波束成形下推导了其性能区域的闭式表达式。

**🔧 技术方法**

使用了率失真理论来推导JDCC系统的性能指标，并通过数值结果验证了理论结果。

**📊 数据集**

使用了多天线基站（BS）同时服务于一个通信用户（CU）和一个可控设备（CD）的JDCC系统模型。

**📈 对比分析**

通过与现有的通信和控制单功能系统进行比较，展示了JDCC系统在通信延迟和控制方差之间的权衡，性能结果表明JDCC的可靠性由上行和下行闭环控制及其与通信的耦合共同决定。

**⚠️ 局限性**

限制在于现有文献中缺乏一个统一的建模和分析框架，无法明确表征通信与控制之间的内在耦合关系。

---

## 188. MSCT: Differential Cross-Modal Attention for Deepfake Detection

**arXiv ID:** 2604.07741 | [PDF](https://arxiv.org/pdf/2604.07741v1)

**作者:** Fangda Wei `[一作]` (Beijing Institute of Technology), Nan Li `[通讯]` (China Academy of Electronics and Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种多尺度跨模态 Transformer 编码器（MSCT），通过差分跨模态注意力和多尺度自注意力实现音视频深度伪造检测；

**💡 创新点**

创新点包括①差分跨模态注意力（DCA）利用注意力矩阵差异提升伪造区域检测，②多尺度自注意力（MSSA）在关键帧之间引入多尺度卷积，增强时间特征感知；

**🔧 技术方法**

使用技术包括：Transformer 编码器、差分跨模态注意力、Multi‑Scale Self‑Attention、Res2Net+波形卷积视觉前置编码器、CBAM、线性投影音频前置编码器、交叉熵+跨模态对齐损失、Adam 优化；

**📊 数据集**

采用公开数据集 FakeAVCeleb（500 真视频 + 20k 伪造视频，四类比例均衡）；

**📈 对比分析**

与现有方法（VFD、MDS、AVOID‑DF、MRDF‑CE、BusterX 等）比较，本文在 FakeAVCeleb 上取得 98.75% 准确率、98.83% AUC，显著优于所有对比基线；

**⚠️ 局限性**

局限性包括：仅在 FakeAVCeleb 上评估，未验证跨数据集泛化；模型计算量较大，可能限制实时部署；对齐损失对伪造/真实样本分布敏感，需进一步研究鲁棒性。

---

## 189. MSGL-Transformer: A Multi-Scale Global-Local Transformer for Rodent Social Behavior Recognition

**arXiv ID:** 2604.07578 | [PDF](https://arxiv.org/pdf/2604.07578v1)

**作者:** Muhammad Imran Sharif `[一作]` (Kansas State University), Doina Caragea `[通讯]` (Kansas State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种基于姿态序列的多尺度全局-局部 Transformer（MSGL‑Transformer），用于识别大鼠和小鼠的社交行为。

**💡 创新点**

创新点包括：① 多尺度时空注意力模块（同时捕捉短期细微动作与长期交互）；② 行为感知调制块（BAM）对时序特征进行动态重加权；③ 无需预定义骨架图，直接对姿态坐标序列建模，模型轻量化且可跨数据集。

**🔧 技术方法**

技术实现：Transformer 架构、全局 token、位置编码、双向与因果注意力、三分支多尺度注意力、BAM 重加权、轻量化编码器、标签平滑交叉熵损失、Adam 优化器等。

**📊 数据集**

使用的数据集：RatSI（9 录像、12 维关键点、5 类社交行为）和 CalMS21（89 录像、28 维关键点、4 类社交行为）。

**📈 对比分析**

与 TCN、LSTM、Bi‑LSTM 等顺序基线以及 ST‑GCN、MS‑G3D、CTR‑GCN、STGAT、HSTWFormer 等骨架基线进行比较；在 RatSI 上平均精度 0.754、F1 0.746；在 CalMS21 上准确率 87.09%、F1 87.45%，均显著优于基线且跨数据集保持一致。

**⚠️ 局限性**

主要局限：稀有行为（如 Moving Away、Attack）的识别仍受类别不平衡影响，误判率在行为边界附近偏高；需要进一步研究边界感知、样本增广和类别不平衡处理方法。

---

## 190. Vulnerability Abundance: A formal proof of infinite vulnerabilities in code

**arXiv ID:** 2604.07539 | [PDF](https://arxiv.org/pdf/2604.07539v1)

**作者:** Eireann Leverett `[一作]` (Concinnity Risks Ltd.), Jeroen van der Ham-de Vos `[通讯]` (University of Twente)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造一个 C 程序（Vulnerability Factory）并给出其可构造证明，证明该程序能产生可计数无限的独立 CVE 可分配漏洞，从而间接证明所有软件漏洞集合是无限的。

**💡 创新点**

创新点在于首次用可执行程序和可证明的 Turing 机模型构造出无限漏洞生成器，提出“漏洞丰度”概念，将漏洞分布与化学元素丰度类比，并将漏洞丰度与软件市场份额结合，用于风险评估。

**🔧 技术方法**

技术手段包括集合论与可计算性理论的形式化证明、基于 MITRE CVE 计数规则的验证、模型检查与静态分析工具的验证、以及对该程序的 Turing 机化简化。

**📊 数据集**

主要使用的数据集是 MITRE CVE 数据库和公开的 CVE 计数规则，未进行实验性数据收集，更多是理论推导与已有公开数据的佐证。

**📈 对比分析**

由于该工作主要是理论证明，未与现有漏洞扫描器或模型检查器进行实验比较；但文中提到基于模型检查的计数上限检验可生成反例，表明在任何有限阈值下都能找到越过该阈值的实例。

**⚠️ 局限性**

局限性包括：1）实现中使用的计数器有限，实际可产生的漏洞数量受机器位数限制；2）论文所示的无限漏洞在实际软件工程中并不直接导致漏洞出现；3）缺乏对生成漏洞实用性的实验评估；4）仅考虑了可计数无限的漏洞数量，而非漏洞严重程度或可利用性。

---

## 191. Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding

**arXiv ID:** 2604.07692 | [PDF](https://arxiv.org/pdf/2604.07692v1)

**作者:** Micky C. Nnamdi `[一作]` (Georgia Institute of Technology), May D. Wang `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Tree-of-Evidence（ToE）框架，用推理时搜索算法为大型多模态模型生成可审计的离散证据集合。

**💡 创新点**

创新点包括：将解释性视为离散优化问题，结合轻量级 Evidence Bottleneck 进行单位评分，并采用 beam search 在多模态空间中寻找最小证据集，实现模型可信的决策轨迹。

**🔧 技术方法**

使用技术包括：Evidence Bottleneck + 直通估计器（STE）对时间窗和文本块评分，GRU 对时序编码，冻结 BioClinicalBERT 文本编码，beam search 搜索与稳定性、稀疏性权衡的评分函数。

**📊 数据集**

使用数据集：MIMIC-IV（4项临床预测），eICU（跨中心验证），以及非医疗 LEMMA‑RCA（故障检测）。

**📈 对比分析**

与 LIME、SHAP、梯度显著性、CBM 以及 1B–70B 的 LLM 进行比较，ToE 在 k=5 证据单位下保持 98%+ AUROC、最低的信度误差（Fidelity MAE），且在稀疏预算下性能显著优于所有基线。

**⚠️ 局限性**

局限性：仅对模型内部逻辑可解释，若模型学习了偏见或伪相关，ToE 也会揭示；证据单位粗粒度（小时窗、文本块）；目前仅适用于晚期融合的可分离模态；搜索仅近似最优；在更长文本或更宽 beam 时计算开销可能增加。

---

## 192. Evaluation as Evolution: Transforming Adversarial Diffusion into Closed-Loop Curricula for Autonomous Vehicles

**arXiv ID:** 2604.07378 | [PDF](https://arxiv.org/pdf/2604.07378v1)

**作者:** Yicheng Guo `[一作]` (Tongji University), Jian Sun `[通讯]` (Tongji University)

**通讯引用:** 266392 | [OpenAlex ID](https://openalex.org/A5100785015)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Evaluation as Evolution（E²）闭环框架，将对抗评估转化为自适应的进化课程，用于自动化车辆策略的改进。

**💡 创新点**

创新点在于结合逆时扩散模型与拓扑分岔分析，实现稀疏控制和拓扑锚定的可解释对抗场景生成，同时闭环回传失败信息以持续更新策略。

**🔧 技术方法**

采用逆时随机微分方程（SDE）扩散、KL 正则化稀疏控制、拓扑分岔与语义可行性投影、Topological Anchoring，以及多轮闭环演化的强化学习调优。

**📊 数据集**

实验数据集为 nuScenes 和 nuPlan 两大公开轨迹数据集，且能直接跨数据集迁移。

**📈 对比分析**

与 CTG、STRIVE、DiffScene、CCDiff、Safe‑Sim‑opt 等基线比较，E² 在 nuScenes 的冲突发现率达 60.29%（高于 51.28% 的 STRIVE）且现实性 REAL 0.7432、离场率仅 0.14%；在 nuPlan 上零样本提升 21.43%，并通过闭环微调显著降低失败率。

**⚠️ 局限性**

局限性包括对预训练扩散模型分布逼近的依赖、攻击强度与锚定阈值需手动调参、对极端复杂交互的覆盖仍有限，以及拓扑分岔分析误判可能导致控制目标不准确。

---

## 193. TR-EduVSum: A Turkish-Focused Dataset and Consensus Framework for Educational Video Summarization

**arXiv ID:** 2604.07553 | [PDF](https://arxiv.org/pdf/2604.07553v1)

**作者:** Figen Eğin `[一作]` (Izmir Katip Celebi University), Aytuğ Onan `[通讯]` (Izmir Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了TR‑EduVSum数据集并提出AutoMUP框架，利用多份人类摘要自动生成金标准视频摘要。

**💡 创新点**

采用基于语义单元聚类与共识权重的自动金标准生成方法，消除人工标注成本，提供可重复、语言无关的框架。

**🔧 技术方法**

使用多语种SBERT嵌入、层次聚类、支持计数/比率权重进行摘要聚合，并与Flash 2.5、GPT‑5.1等LLM摘要进行对比。

**📊 数据集**

使用82段土耳其“数据结构与算法”课程视频的TR‑EduVSum数据集，包含3,281条独立人工摘要。

**📈 对比分析**

通过BERTScore‑F1、ROUGE‑L、BLEURT、SBERT、SimCSE、USE等指标与LLM及人类摘要对比，AutoMUP‑1在所有指标上均优于LLM，消融实验验证共识权重与聚类是关键。

**⚠️ 局限性**

只关注共识，可能忽视少数重要观点；依赖SBERT嵌入的聚类对跨语言泛化有限；数据仅涵盖“数据结构与算法”课程，未验证其他领域的适用性。

---

## 194. SYN-DIGITS: A Synthetic Control Framework for Calibrated Digital Twin Simulation

**arXiv ID:** 2604.07513 | [PDF](https://arxiv.org/pdf/2604.07513v1)

**作者:** Grace Jiarui Fan `[一作]` (Columbia University), Yuhang Wu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SYN‑DIGITS框架，通过后处理校准LLM生成的数字孪生响应，使其更贴近真实人类行为。

**💡 创新点**

创新点在于基于配对潜在因子模型给出行/列空间包含条件与理论误差界限，并系统评估十种校准方法及其在多种人设构造下的稳健性。

**🔧 技术方法**

使用线性回归（Ridge、Lasso、Elastic Net）、神经网络、合成控制、矩阵补全（HSV、SSV、ALS）、加权集成和镜像下降等技术实现校准与分布级匹配。

**📊 数据集**

主要数据集包括MovieLens 20M（500用户/500电影子集）和Twin‑2K‑500（21058份问卷），分布级实验还使用OpinionQA。

**📈 对比分析**

与零射击和上下文学习基线相比，校准方法平均提升30–50% Pearson相关性，分布校准可将误差降低50–90%，最优方法在不同人设构造下表现稳定。

**⚠️ 局限性**

局限性包括仅适用于数值/离散评分，无法处理自由文本；校准依赖于训练问题覆盖潜在空间；当新问题不在行/列空间内时性能下降；需真实人类数据支持。

---

## 195. MVOS_HSI: A Python Library for Preprocessing Agricultural Crop Hyperspectral Data

**arXiv ID:** 2604.07656 | [PDF](https://arxiv.org/pdf/2604.07656v1)

**作者:** Rishik Aggarwal `[一作]` (South Dakota State University), Moon S. Kim `[通讯]` (USDA/ARS Environmental Microbial and Food Safety Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

实现了一个面向叶片级别高光谱成像的开源 Python 库，提供从原始 ENVI 数据的辐射校准、叶片自动分割与裁剪、几何增强以及光谱可视化的全流程处理。

**💡 创新点**

创新点在于将传统手工或碎片化的 MATLAB/Python 脚本整合为单一可安装包，支持命令行与 API 两种使用模式，统一目录结构与参数记录，显著提升可复现性和跨实验共享效率。

**🔧 技术方法**

技术栈包括 Python 3.x、NumPy/SciPy/Matplotlib、Spectral Python（I/O）、scikit-image（图像分割）、imgaug（几何增强）等主流科学计算与图像处理库。

**📊 数据集**

使用了来自实验室的叶片高光谱数据（ENVI 格式），配合对应的暗参采集文件，并以大豆叶片的 SDS（Sudden Death Syndrome）检测数据为示例演示流程。

**📈 对比分析**

通过与传统手工 GUI 工具对比，作者强调脚本化流程在参数透明度、批处理效率和结果可重复性方面的优势；虽然文中未给出量化指标，但展示了完整的校准、分割、增强与可视化链路，说明在实验规模上实现了高效的端到端处理。

**⚠️ 局限性**

局限性包括：对稳定光照和准确暗参的依赖、对背景多样性、叶片重叠或阴影的鲁棒性不足、仅支持 ENVI 格式及特定传感器配置；未来计划扩展传感器兼容性、丰富指数选择以及自动参数调优。

---

## 196. Cognitive-Causal Multi-Task Learning with Psychological State Conditioning for Assistive Driving Perception

**arXiv ID:** 2604.07651 | [PDF](https://arxiv.org/pdf/2604.07651v1)

**作者:** Keito Inoshita `[一作]` (Kansai University), Akira Imanishi `[通讯]` (ISUZU Advanced Engineering Center Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种基于认知因果结构的多任务学习框架CauPsi，用于同时识别驾驶员情绪、行为、交通与车辆上下文。

**💡 创新点**

引入Causal Task Chain实现软标签传播的因果链条，并提出Cross-Task Psychological Conditioning将面部与身体姿态估计的心理状态信号注入所有任务，实现内在状态对感知的调制。

**🔧 技术方法**

使用冻结的MobileNetV3-Small编码器、双向视图注意力、原型嵌入的软标签传递、对数交叉熵+标签平滑、梯度反转域分类器以及EMA等训练技巧。

**📊 数据集**

在AIDE多模态时间序列数据集（2898样本）上进行评估。

**📈 对比分析**

与TEM^3-Learning等前沿模型在同一数据集下对比，平均准确率达到82.71%，比TEM^3-Learning高1.0%，并在情绪识别和行为识别上分别提升3.65%和7.53%。

**⚠️ 局限性**

缺乏对心理状态维度解释的验证、对时序信息的处理不足以及对极少样本类别的识别仍不理想。

---

## 197. Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers

**arXiv ID:** 2604.07822 | [PDF](https://arxiv.org/pdf/2604.07822v1)

**作者:** Harsh Kohli `[一作]` (Ohio State University), Yuekun Yao `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究循环深度Transformer在隐式多跳推理中的组合泛化能力，探讨其在系统性泛化和深度外推方面的表现。

**💡 创新点**

首次揭示循环深度Transformer通过权重共享实现系统性泛化，并能通过推理时迭代实现更深的推理深度，同时发现其学习过程呈现三阶段grokking动态。

**🔧 技术方法**

采用循环深度Transformer结构、动态/固定迭代训练、零初始化、logit lens可视化、以及自适应停止等技术。

**📊 数据集**

使用构造的合成知识图谱数据集（含2000/200实体/关系等）以及多跳推理任务进行实验。

**📈 对比分析**

与普通Transformer对比，循环深度Transformer在系统性泛化上显著优于无循环模型；在深度外推方面通过增加推理迭代可实现10-24跳的推理，动态迭代模型性能优于固定迭代。

**⚠️ 局限性**

过度迭代导致的过度思考限制了极深推理的性能，并且对真实长尾知识迁移的效果尚未得到充分验证。

---

## 198. Tool Retrieval Bridge: Aligning Vague Instructions with Retriever Preferences via Bridge Model

**arXiv ID:** 2604.07816 | [PDF](https://arxiv.org/pdf/2604.07816v1)

**作者:** Kunfeng Chen `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种工具检索桥模型（Tool Retrieval Bridge），通过把模糊指令转换为更具体的表达来提升工具检索性能。

**💡 创新点**

创新点在于用桥模型将用户模糊指令与检索器偏好对齐，并在检索阶段通过强化学习进一步优化模型。

**🔧 技术方法**

采用了大型语言模型（LLaMA-3.2-3B）作为桥模型，并结合监督微调与直接偏好优化（DPO）实现指令重写。

**📊 数据集**

使用自制的 VToolBench 数据集（对 ToolBench 进行模糊化生成）以及公开的 ToolBench、Berkeley Function‑Calling Leaderboard (BFCL) 进行评测。

**📈 对比分析**

与 BM25、TF‑IDF、AdaEmbedding、ToolRetriever 等四种检索器比较，实验显示桥模型能在所有检索器上平均提升 111.51% 以上的 NDCG 分数，并在 BFCL 真实世界任务中提升约 5‑7% 的检索与调用准确率。

**⚠️ 局限性**

主要局限包括：数据集为人工生成的模糊化样本，可能存在偏差；桥模型增加了计算与推理成本；对多语言与特定领域的泛化尚未验证。

---

## 199. AgriChain Visually Grounded Expert Verified Reasoning for Interpretable Agricultural Vision Language Models

**arXiv ID:** 2604.07814 | [PDF](https://arxiv.org/pdf/2604.07814v1)

**作者:** Hazza Mahmood `[一作]`, Rao Anwer `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了 AgriChain 数据集，并用专家验证的链式推理（CoT）文本与置信度标签对 11,000 张植物病害图像进行标注；随后对开放源代码 VLM Qwen‑2.5‑VL‑3B 进行 CoT 监督微调，得到能够在诊断时提供可信、可解释的推理链的新模型 AgriChain‑VL3B。

**💡 创新点**

创新点包括：①将专业农学推理链与视觉数据紧密结合，生成了首个包含专家级推理与置信度标注的农学视觉‑语言数据集；②设计了视觉‑文本对齐的 CoT 监督框架，鼓励模型学习与专家诊断相符的视觉特征；③提出 Region–Text Alignment（RTA）指标，量化推理与视觉证据的对齐程度，提升解释可检验性。

**🔧 技术方法**

技术手段主要有：使用 GPT‑4o‑mini 生成初始诊断推理链；人工专家校验并标准化推理；采用 Qwen‑2.5‑VL‑3B 作为基准 VLM，使用自回归语言建模目标进行微调；训练时采用 AdamW、FP16 混合精度、cosine 学习率调度；评估时结合准确率、宏 F1、加权 F1 以及 RTA 等多维指标。

**📊 数据集**

使用的主要数据集包括 PlantVillage、PlantDoc、PlantCLEF 等公开农学图像来源，经过去重、质量过滤、标签统一后形成 11,000 张标注图像（33 病害类 + 健康类）。每张图像配有专家写作的推理链与置信度标签。

**📈 对比分析**

与 Gemini 1.5 Flash、Gemini 2.5 Pro、GPT‑4o‑Mini 等零样本 VLM 基线进行对比。AgriChain‑VL3B 在 1,000 张平衡测试集上取得 73.1% top‑1 准确率、宏 F1 0.466、加权 F1 0.655，分别比 Gemini Pro 提升 17.3、0.325、0.179；同时在六项推理质量指标上平均得分 4.63/5，显著优于基线模型。

**⚠️ 局限性**

局限性包括：①稀有病种样本不足，导致尾部类别仍易出错；②与视觉相似的病害仍存在混淆；③推理链可能过度依赖常见词汇，忽略罕见症状；④当前数据集主要覆盖实验室与标准化场景，缺少多样化田间环境；⑤模型规模较大，对算力要求高，限制了在边缘设备上的即时部署。

---

## 200. Automotive Engineering-Centric Agentic AI Workflow Framework

**arXiv ID:** 2604.07784 | [PDF](https://arxiv.org/pdf/2604.07784v1)

**作者:** Tong Duy Son `[一作]` (Siemens Digital Industries Software), Ajinkya Bhave `[通讯]` (Siemens Digital Industries Software)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了Agentic Engineering Intelligence（AEI）框架，将汽车工程工作流视为受约束、历史感知的顺序决策过程，通过离线构建多模态工程记忆与在线闭环状态估计与决策支持实现工匠式干预；

**💡 创新点**

创新点在于：①将工程工作流整体建模为控制理论意义下的闭环决策问题；②引入“工作流能量”启发式用于排序候选干预；③整合多模态记忆构建（文本、图像、语音）与LLM工具交互；④在多种汽车工程场景（悬挂设计、强化学习调参、气动外形探索、MBSE）中统一实现；

**🔧 技术方法**

核心技术包括：多模态解析管道（VLM+ASR+文本提取）、检索增强生成（RAG）、知识图谱与向量检索、基于约束的离线/在线决策模型、控制理论视角下的能量启发式、工程工具链（Simcenter/Teamcenter/Amesim）集成；

**📊 数据集**

使用了内部Siemens工程档案（PPT、技术报告、会议录音）、DrivAerNet气动基准数据集（用于外形气动 surrogate 训练），以及各业务场景产生的仿真日志和设计记录；

**📈 对比分析**

本文主要以案例演示形式进行验证，未给出统一量化指标；通过悬挂设计、RL调参、气动优化、MBSE四大用例展示框架可行性，示例中AEI能显著减少人工诊断步骤并提升决策可追溯性；

**⚠️ 局限性**

局限性包括：①缺乏大规模实测实验与性能对比；②检索质量与多模态解析的准确性直接影响决策质量；③对工具链的深度集成需要工程投入，泛化到其他领域尚未验证；④目前以推荐为主，尚未实现完全自主控制。

---

## 201. RoboAgent: Chaining Basic Capabilities for Embodied Task Planning

**arXiv ID:** 2604.07774 | [PDF](https://arxiv.org/pdf/2604.07774v1)

**作者:** Peiran Xu `[一作]` (Peking University), Yadong Mu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RoboAgent，一个基于单一视觉语言模型的能力驱动嵌入式任务规划框架，通过调度器调用五种专门能力将复杂规划拆解为一系列视觉语言子任务。

**💡 创新点**

创新点在于将规划过程拆分为可解释、可控的能力调用，利用内部模拟器信息进行多阶段监督（SFT+DAgger+RL），并首次在无外部工具的情况下实现全流程可解释推理。

**🔧 技术方法**

使用了视觉语言模型（Qwen2.5-VL-3B）作为调度器和能力，Chain-of-Thought思路，行为克隆、DAgger训练，强化学习中的Expert‑Induced Policy Optimization（EIPO），以及从环境模拟器获取的内部状态进行监督。

**📊 数据集**

主要数据集为ALFRED训练集（6374个任务+20k人类指令），生成SFT、DAgger、RFT训练样本；评估使用EB‑ALFRED、ALFWorld、EB‑Habitat和LoTa‑WAH等多种嵌入式任务规划基准。

**📈 对比分析**

与多种开源VLM和RL方法对比，RoboAgent在EB‑ALFRED平均成功率达到67.0，在ALFWorld实现77.6，显著优于同类模型和部分闭源基线，且在跨模态和跨域环境中保持较好性能。

**⚠️ 局限性**

局限性包括对跨域、不同模拟器的泛化仍存在显著差距；强化学习阶段收敛相对缓慢；需更大规模、多样化的数据来进一步提升模型在复杂环境中的表现。

---

## 202. A Guide to Using Social Media as a Geospatial Lens for Studying Public Opinion and Behavior

**arXiv ID:** 2604.07773 | [PDF](https://arxiv.org/pdf/2604.07773v1)

**作者:** Lingyao Li `[一作]` `[通讯]` (University of South Florida), Lingyao Li (University of South Florida)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套利用社交媒体数据进行地理空间分析的实用工作流，并通过疫苗接受度、地震损毁、机场服务质量与可达性满意度四个案例展示该方法的可操作性。

**💡 创新点**

创新点在于将大型语言模型（LLM）和多模态技术融入信息提取、地点推断与统计建模全过程，形成可复制的全流程框架，并通过对比实验验证其相较传统方法的优势。

**🔧 技术方法**

使用的技术包括：规则/TF‑IDF+随机森林、BERT/Transformer微调、LLM（GPT/LLama3）指令式抽取、NER、情感/立场/主题建模、空间回归（MGWR、SDM）等。

**📊 数据集**

采用的数据集覆盖多平台：X/Twitter、Reddit、YouTube、Google Maps、Yelp；具体案例数据包括COVID‑19疫苗相关推文（约2900万条）、2019 Ridgecrest地震推文、美国98大机场的Google Maps评论、超过百万条可达性相关Google Maps评论等。

**📈 对比分析**

方法通过与官方疫苗接种率、USGS震感图、CDC问卷等外部基准进行相关性和准确率评估。TF‑IDF+随机森林在疫苗推文分类上达73‑74%准确；LLM在立场/情感抽取与多维度主题解释上明显优于传统模型；空间回归模型揭示了显著的地理异质性与社会经济关联，验证了数据的可解释性。

**⚠️ 局限性**

局限性包括：样本偏倚与代表性不足、地点推断的精度有限、文本歧义与LLM hallucination、平台访问与API政策变动、以及对人工验证与伦理风险的高需求。

---

## 203. Anamorphic Encryption with CCA Security: A Standard Model Construction

**arXiv ID:** 2604.07771 | [PDF](https://arxiv.org/pdf/2604.07771v1)

**作者:** Shujun Wang `[一作]` (Griffith University), Leo Yu Zhang `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了在标准模型下可实现强 IND-CCA 安全的接收方隐形加密框架，形式化并给出了公共密钥（PKAKEM）和对称密钥（SKAKEM）的通用构造。

**💡 创新点**

创新点在于将随机可恢复 Key Encapsulation Mechanism（RR‑KEM）与隐形加密结合，打破了传统方案的“全赖”设计，实现了强 IND‑CCA（sIND‑CCA）安全，并给出了完整的标准模型证明；同时提供了对称与非对称两种场景的统一设计。

**🔧 技术方法**

核心技术包括：随机可恢复 KEM、Fujisaki‑Okamoto（FO）变换、可逆伪随机函数（IPF）、伪随机 MAC、密钥封装‑数据加密（KEM‑DEM）范式以及基于游戏的安全证明。

**📊 数据集**

该工作为理论工作，没有使用任何数据集；所有分析均在形式化模型和算法安全性证明之上。

**📈 对比分析**

安全性通过一系列游戏变换与现有基石原语的安全性（伪随机、SUF‑CMA、IND‑CCA 等）关联得到；与传统隐形加密方案相比，主要优势在于实现了 CCA 安全性，能在持有解封密密钥的“独裁者”场景下仍保证隐写信息安全；性能方面未给出具体实验评估，侧重理论保证。

**⚠️ 局限性**

局限性包括：需依赖可用的随机可恢复 KEM，若该基原语不可用则无法部署；目前未提供性能评估与实现细节，实际部署中的计算和通信开销尚待进一步研究；在大规模系统中的实用性与兼容性需要后续实验验证。

---

## 204. Beyond Surface Artifacts: Capturing Shared Latent Forgery Knowledge Across Modalities

**arXiv ID:** 2604.07763 | [PDF](https://arxiv.org/pdf/2604.07763v1)

**作者:** Jingtong Dou `[一作]` (University of Sydney), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨模态伪造检测的通用框架MAF，目标是捕捉共享的潜在伪造知识而非单模态表面特征

**💡 创新点**

创新点在于将多模态检测从“特征融合”转向“模态泛化”，通过显式去除模态样式得到跨模态的伪造本质；并引入Weak MAF与Strong MAF两种严苛评估场景及DeepModal-Bench基准

**🔧 技术方法**

使用模态特征解耦技术、跨模态域泛化（DG）正则化（如Invariant Risk Minimization、Mixup等）、自监督编码器、轻量化全连接检测器，并在弱/强MFA场景下分别使用预训练的ImageBind/LanguageBind/UniBind和自监督ViT/T5编码器

**📊 数据集**

利用公开多模态伪造数据集：LAV‑DF、FakeAVCeleb、Celeb‑DF++、ASVspoof5，构建对齐与非对齐两类组合的DeepModal‑Bench

**📈 对比分析**

与传统多模态学习（Concat、OGM、DLMG）及域泛化方法（ERM、IRM、Mixup、SagNet等）对比，MAF在Weak MAF场景下显著优于所有基线（提升AUC 10‑20%），在Strong MAF场景下虽整体性能下降但仍保持高于随机，说明跨模态伪造特征可迁移

**⚠️ 局限性**

局限性包括：对极端“暗模态”的性能仍有限；需要大规模多模态训练数据；模型对新型生成算法的适应性尚待验证；以及在强自监督编码器训练时的计算开销

---

## 205. Reduced-Mass Orbital AI Inference via Integrated Solar, Compute, and Radiator Panels

**arXiv ID:** 2604.07760 | [PDF](https://arxiv.org/pdf/2604.07760v1)

**作者:** Stephen Gaalema `[一作]` (University of Austin), Clinton Staley `[通讯]` (University of Austin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评估了一种将太阳能电池、计算模块和散热器集成在同一面板上的空间数据中心架构（ISCR），实现高功率、低质量的星座级运算平台。

**💡 创新点**

创新点在于利用蒸汽室散热器作为机械支架，将太阳能电池与AI芯片共载，显著提升比功率至约506 W/kg，并通过分布式LLM推理降低热负荷与通信延迟。

**🔧 技术方法**

核心技术包括薄膜钙钛矿/硅双层太阳能电池、硼氮化物支撑柱的蒸汽室散热器、硅/HDPE护盾的AI ASIC、张量/流水线并行的LLM推理以及气体推进的可扩展展开机理。

**📊 数据集**

论文主要使用理论LLM推理负载（如500,000‑token上下文、128 attention块）作为评估基准，并未引用特定公开数据集。

**📈 对比分析**

通过与传统高温散热/单独结构的空间数据中心进行能耗/令牌、特定功率和热设计指标的对比，ISCR在低温下实现约30%能源/令牌优势，特定功率约112 kW/吨。

**⚠️ 局限性**

主要限制包括热、辐射和结构动力学的实际验证缺失，ASIC NRE与量产成本不确定，面板展开与扭转稳定性需进一步研究，以及辐射寿命与长期可靠性待评估。

---

## 206. Symbiotic-MoE: Unlocking the Synergy between Generation and Understanding

**arXiv ID:** 2604.07753 | [PDF](https://arxiv.org/pdf/2604.07753v1)

**作者:** Xiangyue Liu `[一作]` (HKUST), Ping Tan `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种名为Symbiotic-MoE的统一框架，将视觉生成与多模态理解任务在同一稀疏Mixture-of-Experts Transformer中协同训练，解决传统联合训练中出现的灾难性遗忘与专家路由崩溃问题。

**💡 创新点**

创新点包括①基于专家激活分布的Modality‑Aware Expert Disentanglement，将专家划分为专属理解组、专属生成组并保留共享专家作为多模态桥梁；②Progressive Training Strategy，采用差异学习率与warmup梯度屏蔽，分阶段平衡稳定性与可塑性；③Knowledge‑Inherited Initialization，利用预训练模型的路由与权重进行无冷启动的迁移；④实现无额外参数开销，保持模型规模不变。

**🔧 技术方法**

使用了稀疏Mixture‑of‑Experts Transformer、专家路由器拆分与共享、差异学习率调度、梯度屏蔽技术、流匹配（VAE）图像生成损失、负载均衡辅助损失以及多模态预训练与联合优化。

**📊 数据集**

采用了多任务专有大规模语料库，包括文本生成、图像生成（T2I、T2I‑Long）、语言模型（LM）和多模态推理（MMU）等；评测使用公开基准MMLU、OCRBench、COCO‑30K（FID、CLIP、HPSv2）及T2I‑CompBench。

**📈 对比分析**

与标准MoE（无拆分）和MoT（物理隔离）两种基线进行对比。Symbiotic‑MoE在生成任务上Fid降至23.04、CLIP提升至0.28、HPSv2提升至0.45，优于两者；在理解任务上MMLU提升至0.492、OCRBench提升至747，甚至超越仅训练理解的基线。总体展示了生成与理解可以相互促进，而非零和竞争。

**⚠️ 局限性**

局限性包括：①需依赖已有的大规模预训练VLM做初始化，迁移成本仍较高；②梯度屏蔽策略仅在warmup期间生效，后期仍可能出现细微干扰；③在更低资源或特殊领域任务（如长文本生成、视频生成）中的泛化尚未验证；④对不同专家分配比例的敏感性需进一步探索。

---

## 207. AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention

**arXiv ID:** 2604.07815 | [PDF](https://arxiv.org/pdf/2604.07815v1)

**作者:** Yuxuan Hu `[一作]` (Renmin University of China), Jing Zhang `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AsyncTLS，一种双层稀疏注意力与异步KV缓存卸载机制，用于长上下文LLM推理。

**💡 创新点**

创新点在于：①将粗粒度块过滤与细粒度token选择相结合，形成两级稀疏注意力；②通过时间局部性实现块级预取与增量传输，将KV缓存搬运与计算重叠，显著降低PCIe带宽需求；③将Quest的块评分重写为GEMM，提升在共享KV架构上的算力利用。

**🔧 技术方法**

采用的技术包括：层次稀疏注意力、双层索引（块+token）、量化与通道选择、增量KV块传输、异步预取、GPU–CPU层级存储、GQA/MLA等多头注意力变体。

**📊 数据集**

评测数据集包括LongBench（14项长文本理解任务）、RULER（10项检索与推理任务）、以及在Qwen3‑8B、Qwen3‑14B和GLM‑4.7‑Flash模型上的多任务基准。

**📈 对比分析**

与全注意力、Quest（块级）和DS（token级）做对比；在保持与全注意力相近的准确率的同时，AsyncTLS在算子层面实现1.2×–10.0×速度提升，在端到端吞吐量上获得1.3×–4.7×的增益，尤其在32k–96k token上下文下效果显著。

**⚠️ 局限性**

局限性包括：依赖块级预测的时间局部性，在极端动态上下文中可能导致预取误差；对模型特定的块尺寸和token预算需要手动调优；在GPU与CPU之间的PCIe带宽成为上限时，增量传输收益有限。

---

## 208. TEMPER: Testing Emotional Perturbation in Quantitative Reasoning

**arXiv ID:** 2604.07801 | [PDF](https://arxiv.org/pdf/2604.07801v1)

**作者:** Atahan Dokme `[一作]` (Georgia Institute of Technology), Larry Heck `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个情绪化与中性版数学问题的双向翻译框架，并基于此生成了温度-5400（Temper-5400）数据集，评估18款LLM在情绪化输入下的推理性能

**💡 创新点**

创新点在于①提出了教师‑学生情绪转译框架，能够在保持数值内容不变的前提下控制情绪强度；②通过情绪中和（neutralization）展示情绪对推理的可逆干扰；③提供了通用的情绪化鲁棒性评估方法，可推广至其他风格属性

**🔧 技术方法**

使用了教师‑学生知识蒸馏、情绪分类器的辅助损失、LoRA微调、两种情绪监督（7类分类与100维潜在向量）、多样化翻译器集成与三阶段过滤

**📊 数据集**

数据集包括GSM8K、MultiArith、ARC‑Challenge三大数学推理基准，训练集来源于AQuA‑RAT、MathQA、ASDiv，生成情绪化对齐样本并通过DeepSeek‑V3和手工验证形成5400对情绪–中性样本

**📈 对比分析**

对18款模型（1B‑70B+及GPT‑4/5系列）在基础与Zero‑Shot CoT提示下进行评测，情绪化输入导致平均2–10%准确率下降，情绪中和可恢复70–80%损失；非情绪化改写无显著影响；对比多种模型和提示表明情绪干扰普遍存在

**⚠️ 局限性**

局限包括：仅使用Ekman六大基本情绪；中和恢复仍受限于模型容量，前沿模型下降接近噪声底；仅验证短篇数学问题，对长文本或非数学领域的可扩展性未测试；教师分类器依赖预训练，可能不足以捕捉更细粒度情绪

---

## 209. Density Decomposition on Hypergraphs

**arXiv ID:** 2604.07794 | [PDF](https://arxiv.org/pdf/2604.07794v1)

**作者:** Xiaoyu Leng `[一作]` (Beijing Institute of Technology), Rong-Hua Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了基于密度的超图分解，提出(k,δ)-dense子超图模型及其层级分解方法。

**💡 创新点**

创新点：引入整数密度级k和超边贡献上限δ，统一度数核心与最稠密子图；提出公平‑稳定、DSM‑PATH/DSM‑FLOW/DSM‑ALL等四种高效挖掘算法；通过分治策略实现DSD+，将整体分解复杂度降至O(nmδ·d^E_max·log k_max)。

**🔧 技术方法**

使用技术包括超图方向化、均衡（egalitarian）方向、最大流网络、BFS、分治、动态维护等。

**📊 数据集**

实验数据集包括九个真实超图：CP、SC、SB、CH、HC、HB、TC、AR、SA。

**📈 对比分析**

与k-core、nbr‑k‑core、(k,h)-core、α,β-core、hyper k‑truss等基线对比，实验表明(k,δ)-dense分解层数最多、层质量最好、计算时间最快（与基线相比通常快10–20倍），内存占用低；动态维护在小批量更新下优于全重算。

**⚠️ 局限性**

局限性：δ参数需要人工选择，极大超边或极大数据集仍可能出现内存溢出；在大量批量更新时动态维护性能下降，接近全重算；模型对超图的稠密度分布假设仍有限。

---

## 210. Plug-and-Play Logit Fusion for Heterogeneous Pathology Foundation Models

**arXiv ID:** 2604.07779 | [PDF](https://arxiv.org/pdf/2604.07779v1)

**作者:** Gexin Huang `[一作]` (University of British Columbia), Xiaoxiao Li `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种只使用专家输出的LogitProd框架，实现对异构病理基础模型的轻量级加权乘积融合。

**💡 创新点**

创新点在于：①只在logit层进行加权，避免特征对齐和重编码；②通过自适应门控利用置信度、熵和专家间不一致度等logit衍生特征；③提出乘积融合形式并给出理论保证可不劣于最佳专家。

**🔧 技术方法**

采用温度缩放、softmax、熵、最大概率、相邻概率差等logit衍生特征作为门控输入，使用轻量级两层MLP生成非负权重，并在softmax下对各类别概率做加权乘积。

**📊 数据集**

在22个基准上评估，包括WSI级诊断、切片级分类、基因突变预测和离散时间生存模型，数据来源包括TCGA、PANDA、CRC-100K等。

**📈 对比分析**

与特征层融合基线、平均/多数投票、均匀乘积等对比，LogitProd在20/22任务中排名第一，平均提升约3%，并在训练时间和参数量上比特征融合低约12倍，显著提升性能-效率比。

**⚠️ 局限性**

局限性：仅支持针对相同标签空间的已冻结专家，无法融合不同任务或多模态模型；当所有专家表现均差时无法提升；需要预先准备一个专家池，缺乏在线自适应选择机制。

---

## 211. Structured Distillation of Web Agent Capabilities Enables Generalization

**arXiv ID:** 2604.07776 | [PDF](https://arxiv.org/pdf/2604.07776v1)

**作者:** Xing Han Lù `[一作]` (Mila Quebec AI Institute), Siva Reddy `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种框架——Agent-as-annotators，利用前沿大型语言模型（Gemini 3 Pro）自动生成合成轨迹并通过纯监督微调（SFT）训练出可本地部署的9B参数web代理模型。

**💡 创新点**

创新点包括：① 将人类注释工作（Task Designer、Annotator、Supervisor）映射为LLM模块；② 引入评价提示、推理轨迹和LLM Judge进行轨迹过滤；③ 证明轨迹质量远胜数量，且在少量环境下即可实现跨域迁移。

**🔧 技术方法**

核心技术包括：前沿LLM教师生成任务与轨迹、思维预算控制、LLM Judge评估、结构化推理轨迹、Qwen3.5-9B多模态模型的SFT、BrowserGym评估框架。

**📊 数据集**

数据集：在六个WebArena自托管环境生成3,000个任务，得到2,322成功轨迹共16,353观测‑动作样本；评估使用WebArena、VisualWebArena、WorkArena L1/L2、MiniWoB等benchmark。

**📈 对比分析**

与GPT‑4o、Claude 3.5 Sonnet等专有模型以及Go‑Browse、Qwen3.5‑27B等公开基线在统一评估协议下比较；A3‑Qwen3.5‑9B在WebArena达到41.5%（超过Claude 36% +5.5pp、GPT‑4o 31.5% +10pp），在WorkArena L1提升18.2pp，表现出强泛化能力。

**⚠️ 局限性**

局限性：仅使用六个环境，缺乏更广泛网站覆盖；Judge误报率未与人工标注对比；仅采用SFT，未结合强化学习；教师模型仅来自Gemini家族，未验证其他模型的适用性；轨迹规模已趋于饱和，进一步提升需新任务生成策略。

---

## 212. ESOM: Efficiently Understanding Streaming Video Anomalies with Open-world Dynamic Definitions

**arXiv ID:** 2604.07772 | [PDF](https://arxiv.org/pdf/2604.07772v1)

**作者:** Zihao Liu `[一作]` (Communication University of China), Linlin Yang `[通讯]` (Communication University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ESOM框架，基于训练‑free 的多模态大语言模型实现开放世界视频异常检测，并通过 Definition Normalization 和 Probabilistic Scoring 兼顾动态异常定义与帧级异常评分。

**💡 创新点**

创新点包括：①训练‑free 流式推理设计；②结构化异常定义规范化与概率化评分；③Inter‑frame‑matched Intra‑frame Token Merging (IIM) 与 Hybrid Streaming Memory (HSM) 实现高效压缩与记忆管理；④构建了多动态定义的 OpenDef‑Bench 基准。

**🔧 技术方法**

使用技术包括：多模态大语言模型（如 Qwen3‑VL）、逆 RoPE 旋转、Token 压缩、滑动窗口流式推理、概率曲线化评分、表格化定义标准化，以及 KV 缓存重用与长期记忆注入。

**📊 数据集**

采用数据集：UCF‑Crime、XD‑Violence、ShanghaiTech，以及自建的 OpenDef‑Bench（770 视频、1492 样本）。

**📈 对比分析**

在传统零样本、动态定义、类别选择、异常定位、解释性等任务上与多种训练、梯度自由、训练‑free 基线对比，ESOM 在 AUC、AP、LaAP、F1、准确率等指标上均明显领先，RTF < 1 实现单 GPU 实时推理。

**⚠️ 局限性**

局限性：依赖大规模 LLM，算力需求仍高；定义规范化与概率评分对模型语言理解能力有依赖；在极低帧率或极长视频中的延迟与内存占用仍有限；目前仅在单 GPU 进行测试，缺乏多设备或多模型的扩展验证。

---

## 213. Administrative Decentralization in Edge-Cloud Multi-Agent for Mobile Automation

**arXiv ID:** 2604.07767 | [PDF](https://arxiv.org/pdf/2604.07767v1)

**作者:** Senyao Li `[一作]` (Huazhong University of Science and Technology), Ruixuan LI `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AdecPilot 框架，将云端的战略规划与边缘端的战术执行分离，实现移动自动化的去中心化管理。

**💡 创新点**

创新点在于行政去中心化、双模边缘自治（视觉 orchestrator + 文本 executor）以及层级隐式终止协议（HIT），显著提升系统鲁棒性与隐私保护。

**🔧 技术方法**

采用多模大语言模型（GPT‑4o 云端、Qwen‑VL 视觉模型、Qwen 文本模型）、UI‑agnostic 云设计器、边缘视觉规划与文本执行、HIT 终止协议和局部自校正循环。

**📊 数据集**

使用 AndroidWorld 与 AndroidLab 这两个移动 UI 自动化基准数据集进行评估。

**📈 对比分析**

与单体与协作基线对比，AdecPilot 在任务成功率上提升约 21‑34%，云端 token 消耗降低 37‑95%，端到端延迟下降 88‑99%，并能在 2G 网络下保持约 3 秒的响应。

**⚠️ 局限性**

局限包括对极端 UI 变异仍需局部重规划、视觉或文本模型精度限制导致部分失败，以及在更大规模任务和多设备场景中的泛化验证仍待深入。

---

## 214. Learning to Coordinate over Networks with Bounded Rationality

**arXiv ID:** 2604.07751 | [PDF](https://arxiv.org/pdf/2604.07751v1)

**作者:** Zhewei Wang `[一作]`, Marcos M. Vasconcelos `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在有限理性条件下，通过 Log‑Linear 学习（LLL）实现网络协同，探讨网络连通度与理性程度如何共同影响协同状态的长期稳定概率，并证明在均匀连通度（K‑regular）网络中，连通度越高所需的理性阈值越低。

**💡 创新点**

创新点在于：①给出有限理性下协同概率随连通度单调提升的理论证明；②证明在大规模或低理性场景下，K‑regular 结构是最优的，并量化了“非均匀性代价”（Price of Irregularity）；③将 Gibbs 分布与 Gaussian 近似相结合，提供了可计算的性能下界。

**🔧 技术方法**

主要技术包括：图理论（正则图构造与度序列的 majorization）、统计物理中的 Gibbs‑Boltzmann 分布与矩母函数、马尔可夫链稳态分析、Taylor 展开与 Rayleigh 商、以及 Martingale 中心极限定理用于大图极限。

**📊 数据集**

本文为理论分析，未使用实测数据集；所有结果均基于抽象图模型和随机变量（Rademacher 取值）推导。

**📈 对比分析**

通过解析上界/下界以及数值仿真（如 N=14、θ=0.3 的图形展示），作者对比了不同连通度与不同理性水平下的协同概率，结果表明：连通度升高、理性降低都能提升协同概率；K‑regular 网络在多种情形下提供最优或接近最优性能。

**⚠️ 局限性**

局限性包括：①仅考虑同质有限理性，未探讨理性异质性与连接策略；②未给出异构网络或调度优化的正式理论；③对极大规模网络的计算复杂性仍未解决，需进一步借助 graphon 等方法。

---

## 215. The Weaponization of Computer Vision: Tracing Military-Surveillance Ties through Conference Sponsorship

**arXiv ID:** 2604.07803 | [PDF](https://arxiv.org/pdf/2604.07803v1)

**作者:** Noa Garcia `[一作]` (University of Osaka), Amelia Katirai `[通讯]` (University of Tsukuba)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过收集 2004-2024 年 CVPR、ICCV、ECCV 三大计算机视觉会议的赞助商数据，构建了包含 469 家技术公司和研究机构的数据库，并对其与军用与监视领域的关联进行系统注释和量化分析；同时通过两个案例研究阐述赞助商分析在揭示技术武器化中的潜力与局限。

**💡 创新点**

创新点在于首次将会议赞助视为评估计算机视觉技术武器化的可行代理指标，搭建了包含赞助信息、元数据、武器化领域标签和参与度档案的多维注释框架；通过此框架揭示 44% 赞助商与军用/监视领域有实质联系，并通过案例展示赞助分析能识别即便表面关注公平与人道的技术也可能被军用化。

**🔧 技术方法**

采用的技术主要是：① 手工抓取会议官网赞助商列表并自动去重；② 通过 Crunchbase、Wikipedia、Wikidata 采集元数据；③ 结合官方网页审计、系统化网络检索和六个外部数据库（AI War Cloud、AIGS Index、Atlas of Surveillance、Surveillance Watch、SIPRI、Investigate）的 TFIDF 匹配，对每个赞助商进行军事、监视或无关联三类标注；④ 对武器化参与度进行四档归类（overt、transparent、opaque、not applicable）。

**📊 数据集**

数据集包含 469 家独立赞助商的基本信息、国家、创立年份、行业分类、母子公司关系等元数据，并与六个外部数据库交叉引用以确认其军用或监视关联；该数据集可用于进一步分析赞助层级、公司网络等维度。

**📈 对比分析**

本文没有传统意义上的模型性能对比；通过定量指标（如 44% 赞助商涉军/监视、30% 未公开关联、不同年份与地区的分布）评估赞助商在武器化中的占比和趋势；与过去文献对军工 AI 投入的描述对比，验证赞助商分析作为揭示武器化的有效工具。

**⚠️ 局限性**

主要局限包括：① 关联性仅为相关性，无法证实因果；② 数据集并非完整，缺少赞助商规模、收入等更细粒度特征；③ 依赖英文网页和公开数据库，存在西方中心化偏差；④ 赞助商分析可能遗漏非公开或多元用途的技术；⑤ 对双重用途技术的评估仍需结合当地社区需求与使用场景。

---

## 216. Latent Anomaly Knowledge Excavation: Unveiling Sparse Sensitive Neurons in Vision-Language Models

**arXiv ID:** 2604.07802 | [PDF](https://arxiv.org/pdf/2604.07802v1)

**作者:** Shaotian Li `[一作]` (Macquarie University), Tat-Seng Chua `[通讯]` (National University Of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

利用冻结的视觉语言模型（VLM）中稀疏的敏感神经元，通过无训练的方式挖掘潜在的异常知识，并以此进行零/少样本异常检测和定位。

**💡 创新点**

认为异常知识已嵌入预训练模型，仅需激活少量高方差神经元即可完成检测，而非依赖外部适配器或大规模内存；提出LAKE框架实现稀疏神经元定位与跨模态激活的结合。

**🔧 技术方法**

方差筛选确定敏感神经元子空间；patch‑level视觉偏差计算与最近邻距离；跨模态文本提示对比激活；两分支特征融合。

**📊 数据集**

工业缺陷数据集MVTec‑AD、VisA、BTAD以及医学影像数据集Brain‑AD。

**📈 对比分析**

与多种零/少样本基线（OpenCLIP、WinCLIP、CLIP‑AD、AnomalyCLIP、AdaCLIP、VisualAD、ReMP‑AD）对比，LAKE在图像级AUROC提升约15–30%，在像素级PRO提升50%以上，且在医学数据上也实现SOTA。

**⚠️ 局限性**

对极少样本（2–4张）性能下降明显；依赖预训练模型的表达能力，若模型不具备足够的结构/语义知识，效果会受限；跨类别泛化虽好，但对极其复杂纹理的异常仍可能产生误检。

---

## 217. Lightweight LLM Agent Memory with Small Language Models

**arXiv ID:** 2604.07798 | [PDF](https://arxiv.org/pdf/2604.07798v1)

**作者:** Jiaquan Zhang `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种轻量级外部记忆系统LightMem，利用小型语言模型(SLM)分离在线高频记忆操作与离线长期整合，支持多用户、STM/MTM/LTM三层记忆，并实现两阶段检索和语义重排序。

**💡 创新点**

创新点在于：1）通过SLM驱动的多模块（控制器、选择器、写入器）实现在线低延迟记忆检索与写入；2）两阶段检索结合向量粗检与语义精排显著降低检索噪声；3）离线聚合将中期记忆抽象为图结构长记忆，兼顾隐私与可扩展性。

**🔧 技术方法**

使用的小型语言模型（如Llama‑3.2‑1B、Qwen2.5‑1.5B）负责查询规划、语义重排与记忆写入；离线采用大上下文LLM进行抽象与整合；检索基于All‑MiniLM‑L6‑v2向量和向量数据库；控制流程通过自定义提示与LoRA微调实现。

**📊 数据集**

评测数据集为LoCoMo（多步推理、时间推理等长对话任务）和DialSim（多方对话模拟），并在多种模型尺度（GPT‑4o/mini、Qwen2.5‑1.5B/3B、Llama‑3.2‑1B/3B）上进行实验。

**📈 对比分析**

与LoCoMo、ReadAgent、MemoryBank、MemGPT、A‑MEM等基线对比，LightMem在LoCoMo上平均提升约2.5点F1，在DialSim上在ROUGE、METEOR、SBERT等指标均居首位；同时检索延迟P50仅83 ms、端到端P50 581 ms，显著低于重放式或LLM驱动的记忆方案。

**⚠️ 局限性**

局限性在于只探讨了一种在线–离线分离的设计，未系统评估其他整合策略或控制策略；同时在极端长对话或高并发场景下的可扩展性与安全性仍需进一步验证。

---

## 218. Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video

**arXiv ID:** 2604.07786 | [PDF](https://arxiv.org/pdf/2604.07786v1)

**作者:** Chanhyuk Choi `[一作]` (Ulsan National Institute of Science and Technology), Taehwan Kim `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了跨模态情绪迁移方法C-MET，利用音频情绪语义向量映射到视觉表情空间，对谈话面部视频进行情绪编辑，支持未见过的情绪生成。

**💡 创新点**

创新点包括：1）首次在音频与视觉空间显式建模情绪语义向量的对应关系；2）通过对齐语义向量实现跨模态情绪迁移；3）轻量化Transformer模块可作为插件无缝集成到现有去混合网络，提升效率并保留视觉质量。

**🔧 技术方法**

技术方法包括：预训练情绪音频编码器（emotion2vec+large）、去混合面部表情编码器（EDTalk）、多模态对比学习、跨模态Transformer、MSE+方向损失等。

**📊 数据集**

使用数据集：MEAD、CREMA-D（训练与评估），Gemini TTS生成扩展情绪音频，HDTF用于定性评估。

**📈 对比分析**

与标签、图像、音频三类基线（EAT、EAMM、EDTalk、FLOAT）比较，C-MET 在情绪识别准确率 Acc_emo 提升约14% 以上，FID/FVD 与基线相近但情绪表达更丰富，平均推理时间更短，用户研究中获得更高偏好率。

**⚠️ 局限性**

局限性包括：仍受训练数据情绪种类限制，复杂微妙情绪生成仍有挑战；依赖预训练音频/视觉编码器，对极端发音或非标准口音的鲁棒性未知；帧间运动细节同步可能略有偏差。

---

## 219. ACIArena: Toward Unified Evaluation for Agent Cascading Injection

**arXiv ID:** 2604.07775 | [PDF](https://arxiv.org/pdf/2604.07775v1)

**作者:** Hengyu An `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的评估框架，用1356个测试用例系统性地评估多智能体系统在Agent Cascading Injection攻击下的鲁棒性，并提出了ACI‑Sentinel防御策略。

**💡 创新点**

提供了覆盖多种攻击表面与目标的完整基准、标准化接口与可扩展架构，并引入了传播脆弱性指数PVI，首次揭示角色设计与交互模式对安全性的影响；同时设计了基于语义最小化的ACI‑Sentinel防御。

**🔧 技术方法**

采用自动化攻击生成（变异/选择+LLM评估）、LLM评估器、PVI指标、语义剔除防御以及基于多模型的实验评测；使用GPT‑4o、GPT‑4o‑mini与Qwen2.5‑7B等大模型。

**📊 数据集**

使用了GSM8K、MATH500、HumanEval、MBPP、GPQA、MedMCQA等任务数据，覆盖数学推理、代码生成、科学与医学四个任务域；构建了六种主流MAS实例（MetaGPT、AutoGen、CAMEL、Self‑Consistency、LLM‑Debate、Agentverse）。

**📈 对比分析**

通过对比ASR、BU、UA及PVI等指标，发现当前MAS在大多数攻击下易受影响；在GPT‑4o实验中，部分系统的ASR可达90–100%；ACI‑Sentinel将ASR显著降低（如AutoGen Exfiltration 53.33%），但仍存在“安全税”与部分攻击失效。

**⚠️ 局限性**

实验局限在于攻击仅模拟白盒LLM环境，防御可能导致实用性下降；基准仅覆盖选定MAS与任务域，未能覆盖所有真实场景；模型层防御易失效，需进一步提升系统级安全设计。

---

## 220. DailyArt: Discovering Articulation from Single Static Images via Latent Dynamics

**arXiv ID:** 2604.07758 | [PDF](https://arxiv.org/pdf/2604.07758v1)

**作者:** Hang Zhang `[一作]` (East China Normal University), Xin Tan `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 DailyArt 框架，利用单张静态图像先合成最大化张开放状态，再通过跨状态差异估计关节参数并实现部件级可控运动。

**💡 创新点**

创新点在于把关节估计转化为合成驱动的推理流程，避免先验依赖，且利用 3D 点云差分实现全局关节推断。

**🔧 技术方法**

采用冻结的 DINOv2 编码器 + VAE 解码器、AdaLN 调制、Vision Geometry Transformer 3D 提取、集合预测网络及 Hungarian 匹配等技术。

**📊 数据集**

使用 PartNet‑Mobility 作为合成与评估数据集，并在 AKB‑48 真实物体上做零样本测试。

**📈 对比分析**

与 DragAPart、PartRM、Puppet‑Master、LARM 等合成基线以及 URDFormer、Singapo、ArticulateAnything、PhysX‑Anything 等估计基线对比，取得 68.4% 整体成功率、PSNR 25.5、SSIM 0.920，明显优于最近方法。

**⚠️ 局限性**

局限在于对合成精度依赖较高，难以处理极端/无闭合/无最大开关状态的物体，且 3D 提升可能在缺乏可见运动时失效。

---

## 221. An Empirical Analysis of Static Analysis Methods for Detection and Mitigation of Code Library Hallucinations

**arXiv ID:** 2604.07755 | [PDF](https://arxiv.org/pdf/2604.07755v1)

**作者:** Clarissa Miranda-Pena `[一作]` (University of Sydney), Jonathan K. Kummerfeld `[通讯]` (University of Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对 LLM 代码生成中的库幻觉，系统评估了静态分析工具（Mypy、Pyright）和自动生成的文档导向语法（grammar）对检测与修复错误的效果，并将这些方法与基线（o3-mini）进行对比。

**💡 创新点**

首次量化静态分析在 Python 代码库幻觉检测中的能力，提出通过文档自动构建语法约束实现受控解码，并给出了这些技术在不同基准上的上限估计（48.5%–77%）。

**🔧 技术方法**

使用的技术包括：1）基于文档生成的 GBNF 语法与 constrained decoding；2）现成的静态分析器 Mypy 与 Pyright；3）LLM 作为判断器（o3-mini）作为基线；4）手工标注的错误类型与“Imaginary Features”评估；5）利用 ER、Pass@1 与 RIF 评估修复效果。

**📊 数据集**

数据集涵盖三大开源 NL‑to‑Code 评测基准：DS‑1000（1,000 题，7 个库）、Odex（945 题，79 个库）和 BigCodeBench（1,140 题，139 个库），并在四种 LLM（Claude‑3、GPT‑4、GPT‑3.5、IBM‑Granite）上进行实验。

**📈 对比分析**

与基线相比，静态分析在错误检测上能覆盖 16–70% 的错误，14–85% 的幻觉；在修复上，结合 o3‑mini 与静态分析能显著提升执行率（ER）与 Pass@1；受控解码通过语法约束将幻觉率降至约 10–15%，但对 Pass@1 影响有限。整体来看，静态分析在资源利用与检测速度上优于编译/运行时检查。

**⚠️ 局限性**

主要限制包括：1）语法构建依赖文档字符串，若文档质量差导致误报或漏报；2）仅关注第一处执行错误，未跟踪后续错误；3）实验样本量有限，未覆盖所有数据集；4）仅针对 Python，无法直接推广至其他语言；5）基准本身存在错误与歧义，影响评估客观性。

---

## 222. Automatic Generation of Executable BPMN Models from Medical Guidelines

**arXiv ID:** 2604.07817 | [PDF](https://arxiv.org/pdf/2604.07817v1)

**作者:** Praveen Kumar Menaka Sekar `[一作]` (University of Maryland), Akihiro Inomata `[通讯]` (Fujitsu Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个端到端管道，能够将医疗政策文本自动转换为可执行、可度量的BPMN模型，并通过模拟评估其决策准确性。

**💡 创新点**

创新点在于将LLM与结构化验证、自动语法纠错相结合实现BPMN 2.0 合规；自动将模型注入KPI，支持基于模拟的性能评估；以及基于熵的多模型不确定性检测，用以指示文本歧义。

**🔧 技术方法**

使用的技术包括大语言模型（Gemini 2.5 Pro/Flash、GPT‑5.1）、BPMN 2.0 XML、SpiffWorkflow 任务引擎、PyMuPDF4LLM 文本提取、Python 表达式评估、结构规则验证与修复以及熵分析。

**📊 数据集**

采用的实验数据集包括三份日本市镇的糖尿病肾病预防指南文档以及 1,000 个覆盖阈值边界的合成病人记录。

**📈 对比分析**

通过将自动生成模型的 KPI 组合与人工编写基线模型进行统计匹配，并计算每患者决策一致率、F1、kappa 等指标；在 City1 完全匹配、City2 约 86–90% 匹配、City3 低一致率，熵值与 kappa 成正相关，验证了不确定性检测的有效性。

**⚠️ 局限性**

局限性在于仅测试三份政策、BPMN 架构有限（缺少并行、定时、子流程等），高熵文档仍需人工介入，并未对真实患者数据进行验证。

---

## 223. Learning Without Losing Identity: Capability Evolution for Embodied Agents

**arXiv ID:** 2604.07799 | [PDF](https://arxiv.org/pdf/2604.07799v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了能力中心化的演化范式，将机器人持续改进的工作拆分为固定的认知主体与可演化的模块化能力单元（ECM）。

**💡 创新点**

创新点在于：①将能力模块化、版本化并形成生命周期管理；②将学习过程与执行过程分离，使用运行时治理层保障安全与政策约束；③通过闭环能力演化（任务执行→经验收集→模型更新→模块升级）实现持续性能提升而不改变主体身份。

**🔧 技术方法**

技术包括：强化学习、示范学习、LLM 代码合成三种学习模态的混合更新；ECM 版本控制与回滚；运行时安全治理层（基于谓词的约束校验、回退与日志跟踪）。

**📊 数据集**

使用 MuJoCo/Robosuite 模拟环境中的 6 个桌面操控任务（Pick/Place/Stack/Pour/Sort/Assemble），每个任务包含不同级别的 ECM 组合需求。

**📈 对比分析**

与代理改动、静态 ECM、SPiRL、SkiMo 等基线对比，迭代 20 次后成功率从 32.4% 提升至 91.3%，平均方差降至 2.0，安全违规率 0%，并在安全约束实验中 100% 阻断不安全动作，运行开销仅 2.3 ms。

**⚠️ 局限性**

局限性包括：仅在单一模拟环境验证，缺乏真实机器人实验；与其他持续学习或奖励设计方法的系统对比不足；运行时治理的误报率与极端复杂情境下的泛化性未充分评估；模块化与整体性能的计算开销未量化。

---

## 224. BRASP: Boolean Range Queries over Encrypted Spatial Data with Access and Search Pattern Privacy

**arXiv ID:** 2604.07797 | [PDF](https://arxiv.org/pdf/2604.07797v1)

**作者:** Jing Zhang `[一作]` (Lancaster University), Zhengyang Qiu `[通讯]` (Lancaster University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种支持布尔范围查询的可搜索加密方案 BRASP，能够在云端安全地存储并查询加密的空间-文本数据。

**💡 创新点**

创新点在于结合 Hilbert 曲线前缀编码、加密前缀‑ID 与关键词‑ID 倒排索引，并通过双服务器索引洗牌与 ID 字段重分配实现搜索模式和访问模式的隐藏，同时支持动态更新和前向安全。

**🔧 技术方法**

使用技术包括专门化的代理伪随机函数 TPF、基于 Paillier 的通用重加密 TUR、Hilbert 曲线前缀覆盖、双服务器随机化洗牌、前缀成员资格验证以及零知识检索证明等。

**📊 数据集**

在 Yelp 商业数据集（spatio‑textual）上进行实验评估。

**📈 对比分析**

与 VPBRQ_SupL、PPSKS 等基线方案对比，BRASP 在索引构建、令牌生成、搜索和更新四个阶段均表现出更低的计算开销，搜索通信略高但仍在可接受范围；整体性能显著优于基线。

**⚠️ 局限性**

限制包括需要双不合作服务器的部署，洗牌与重分配过程对网络延迟敏感；在高频更新场景下，重随机化与分布式更新的开销仍不低。

---

## 225. SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents

**arXiv ID:** 2604.07791 | [PDF](https://arxiv.org/pdf/2604.07791v1)

**作者:** Xinshun Feng `[一作]` (Shanghai Artificial Intelligence Laboratory), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自进化工具记忆与策略联合训练框架，使小规模LLM能够通过构建工具图、分层规划与细粒度奖励，实现长期强化学习；

**💡 创新点**

创新点包括：1) 将工具记忆建模为图结构，用工具作为anchor进行优势估计；2) 联合优化策略参数与外部工具记忆；3) 设计稠密过程奖励与步骤级信用分配；

**🔧 技术方法**

技术手段包括：结构化轨迹生成、工具检索与创建、基于工具的优势估计（episode级+step级）、工具图记忆演化、RL训练（对比GRPO、DAPO等）、LLM-as-Judge评估；

**📊 数据集**

使用的数据集有数学推理类（AIME2024、MATH500、GSM8K）、知识推理类（HotpotQA、2WikiMultihopQA、Musique、Bamboogle），以及10k Tool-star RL样本；评估采用Qwen3-32B判别器；

**📈 对比分析**

与TIR Prompt、GRPO、DAPO、Reinforce++、ARPO等基线比较，实验显示在多跳推理任务中平均排名1.43，往往取得最优或第二优；在数学推理上与ARPO持平，尤其在AIME24上明显领先；

**⚠️ 局限性**

局限性包括：在GSM8K/MATH500等易题上仍有性能差距；工具生成在简单任务上产生噪声；工具集可能不适用于其他任务；部分工具过于简单；奖励函数仍可能诱导表面化奖励行为。

---

## 226. ORACLE-SWE: Quantifying the Contribution of Oracle Information Signals on SWE Agents

**arXiv ID:** 2604.07789 | [PDF](https://arxiv.org/pdf/2604.07789v1)

**作者:** Kenan Li `[一作]` (Microsoft), Dongmei Zhang `[通讯]` (Microsoft)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Oracle-SWE方法，系统地提取并评估了软件工程任务中五大关键信号（重现测试、回归测试、编辑位置、执行上下文和API使用）的对LM代理性能的影响。

**💡 创新点**

创新点在于提供了一套统一的oracle信息提取框架，量化了各信号的上限贡献，并通过两阶段验证实验验证了这些信号在现实情境下的实际价值。

**🔧 技术方法**

主要技术包括基于错误堆栈、覆盖率和AST解析的oracle信号提取、SWE-agent代理与LLM交互、以及多模型对比实验与成本分析。

**📊 数据集**

使用了SWE-bench-Verified、SWE-bench-Live和SWE-bench-Pro三大基准数据集，涵盖Python和Golang多语言任务。

**📈 对比分析**

通过将oracle信号逐个注入基准任务，实验显示重现测试是最重要的信号，结合其他信号可将成功率提升至97%以上，且成本和步骤数在信号累加时出现下降再上升的趋势；两阶段验证实验进一步证明了信息提取与问题解决的协同提升。

**⚠️ 局限性**

局限性包括对信息信号划分的主观性、对仅在少量实例中可用的原生错误堆栈的依赖，以及未覆盖更复杂或更大规模代码仓库的情境。

---

## 227. PeReGrINE: Evaluating Personalized Review Fidelity with User Item Graph Context

**arXiv ID:** 2604.07788 | [PDF](https://arxiv.org/pdf/2604.07788v1)

**作者:** Steven Au `[一作]` (Icahn School of Medicine at Mount Sinai), Baihan Lin `[通讯]` (Icahn School of Medicine at Mount Sinai)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为PeReGrINE的基准，用于在包含时间约束的用户–商品图结构证据下生成个性化产品评测。

**💡 创新点**

创新点在于：①将Amazon Reviews 2023重组为时间一致的双边图；②提出基于产品元数据的商品侧检索而非目标评测文本；③设计了紧凑的用户风格参数；④引入宏观失调分析来评估用户风格与产品共识的偏差。

**🔧 技术方法**

使用技术包括：基于图检索的检索增强语言模型、语义与风格相似度评分、VADER情感分析、视觉模型生成图像字幕以及多模态评估。

**📊 数据集**

数据集为Amazon Reviews 2023的七个产品类别，包含用户历史、商品邻域评论和可选的商品视觉信息。

**📈 对比分析**

通过与LaMP、PGraphRAG等基线在相同的时间约束下匹配重跑，PeReGrINE在评测文本ROUGE‑L和BERTScore‑F1上表现最优；在用户与产品失调指标上，联合证据取得最佳平衡；视觉证据在部分指标上略有提升。

**⚠️ 局限性**

局限性包括：检索排名不支持视觉特征，依赖稠密图邻域而忽略冷启动情况；用户风格参数过于简化，未涵盖语法与话语层面；失调指标为启发式，缺乏子项消融分析。

---

## 228. Toward Generalizable Graph Learning for 3D Engineering AI: Explainable Workflows for CAE Mode Shape Classification and CFD Field Prediction

**arXiv ID:** 2604.07781 | [PDF](https://arxiv.org/pdf/2604.07781v1)

**作者:** Tong Duy Son `[一作]` (Siemens Digital Industries Software), Theo Geluk `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种通用的基于图的3D工程AI框架，将异构的汽车CAE与CFD数据转换为物理感知图，并通过GNN完成模式形态分类和气动场预测。

**💡 创新点**

创新点包括：①物理感知的区域化BiW图与对称保留的下采样表面图；②多任务与自定义聚合的区域感知注意力网络；③结合物理约束的损失函数和可解释性/不确定性引导的数据生成；④在同一框架下实现分类与回归任务，提升模型可复用性与解释性。

**🔧 技术方法**

使用的技术包括：图注意力网络（Graph Attention Network）、AeroGraphNet、物理约束正则化（Bernoulli一致性、质量守恒、WSS切向性）、数据增强、区域感知池化以及不确定性评估。

**📊 数据集**

采用的主要数据集有：①汽车BIW与FE模态数据（4款车型，共326个标注模式），②DrivAerStar CFD数据集（约10k条模拟，3款车身配置）。

**📈 对比分析**

对比方法包括传统GCN、MeshGraphNet、MLP等基线模型；在模式分类上，单车训练达到100% Level‑1、81.6% Level‑2，跨车训练提升至100%/98.7%；在气动场预测上，物理约束模型获得压力R²=0.989、WSS R²=0.985，平均误差分别为21.6 Pa和0.49 Pa，显著优于无物理约束的对照模型。

**⚠️ 局限性**

局限性：仅验证了两类汽车任务，图构造仍需手工域知识；对极端外部扰动（如大偏航、显著扰流器）和新的几何/边界条件的泛化能力有限；对多任务迁移的理论分析尚不完整。

---

## 229. An Empirical Study on Influence-Based Pretraining Data Selection for Code Large Language Models

**arXiv ID:** 2604.07769 | [PDF](https://arxiv.org/pdf/2604.07769v1)

**作者:** Chengli Xing `[一作]` (Peking University), Shikun Zhang `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对代码大语言模型的预训练数据过滤，提出基于验证集损失的影响分数（DIScore）并进行大规模实验，验证其对模型性能的提升。

**💡 创新点**

创新点包括：1）将DIScore应用于生成式编程任务，利用验证集损失评估样本影响；2）系统探究影响分数随训练阶段和任务的变化；3）对比传统的困惑度、LLM评分等过滤方法，发现其在代码数据上的无效性；4）尝试用小模型预测DIScore，但发现准确率低，揭示预测瓶颈。

**🔧 技术方法**

技术手段：Transformer‑decoder架构的Code‑LLM（1B参数），使用CodeLlama基线；数据影响评估采用单步训练后验证集损失变化；与困惑度、GPT‑4o“教育评分”等传统指标对比；利用RoBERTa‑Base预测DIScore的可行性实验。

**📊 数据集**

数据集：StarCoderData（260B代码，随机抽取100B训练），验证集来自HumanEval、MBPP、CrossCodeEval、DS‑1000、Bird‑SQL等多任务（共7个任务）。

**📈 对比分析**

比较方法：通过Spearman相关性和Pass@k/准确率等指标评估DIScore与训练损失、困惑度、LLM评分的关联；利用“Top/Bottom DIScore”训练继续预训练并在7个任务上测算性能提升；发现DIScore过滤后模型在大多数任务上提升约5‑15%（具体数值见表）。

**⚠️ 局限性**

局限性：1）DIScore计算成本高，需要对每个样本做前向后向与验证集对比；2）小模型预测DIScore准确率极低，难以在大规模预训练中应用；3）实验仅在1B参数模型上验证，缺乏更大规模验证；4）验证集覆盖的任务有限，可能导致过拟合或泛化不足。

---

## 230. Sensitivity-Positional Co-Localization in GQA Transformers

**arXiv ID:** 2604.07766 | [PDF](https://arxiv.org/pdf/2604.07766v1)

**作者:** Manoj Chandrashekar Rao `[一作]` `[通讯]` (Independent Researcher), Manoj Chandrashekar Rao (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Llama 3.1 8B GQA Transformer 中任务敏感层与 RoPE 频率影响层的关联性，并提出将 LoRA 与 GQA‑aware RoPE 只应用于任务敏感层的微调策略。

**💡 创新点**

发现任务敏感层与 RoPE 影响层呈强烈反向定位，提出“反定位”理论，并证明将 LoRA 与 RoPE 调整集中在任务敏感层可显著提升性能。

**🔧 技术方法**

使用 LoRA 低秩微调、GQA‑aware RoPE 频率调节、正确性差异余弦距离敏感度量、双优化器训练与四路对照消融实验。

**📊 数据集**

训练数据包括 Magicoder‑OSS‑75K、CodeAlpaca‑20K、MetaMathQA‑30K、OpenHermes‑2.5；评估基准为 MMLU、GPQA、HumanEval+、MATH、MGSM、ARC。

**📈 对比分析**

通过与随机层选取、仅 LoRA、仅 RoPE 等对照的四路消融，实验显示共定位（LoRA+RoPE）在所有六个基准上平均提升 4–16pp，接近 Claude 3.5 Haiku 在 HumanEval+ 上的 68.3% 并且总算力低于 100 USD。

**⚠️ 局限性**

局限性在于仅验证于 Llama 3.1 8B，使用的对比样本仅 15 对，且训练集偏重代码/数学导致事实类基准表现回退。

---

## 231. RemoteAgent: Bridging Vague Human Intents and Earth Observation with RL-based Agentic MLLMs

**arXiv ID:** 2604.07765 | [PDF](https://arxiv.org/pdf/2604.07765v1)

**作者:** Liang Yao `[一作]` (Hohai University), Min-Ling Zhang `[通讯]` (Hohai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 RemoteAgent，一种基于多模态大语言模型的代理框架，能将模糊自然语言查询转换为相应的 EO 任务，并在必要时将密集预测任务委托给专用工具。

**💡 创新点**

创新点包括：①使用 VagueEO 这一人类中心的模糊查询数据集训练模型，提升意图识别；②采用 RL（GRPO）对模型进行对齐，避免传统 SFT 导致的灾难性遗忘；③设计 Model Context Protocol（MCP）实现内部任务与外部专用工具的高效、单步路由，显著降低推理延迟。

**🔧 技术方法**

核心技术包括多模态大型语言模型（Qwen2.5‑VL‑7B‑Instruct）、Group Relative Policy Optimization (GRPO) 的强化学习对齐、统一多模态奖励机制、MCP 工具调用协议以及 LoRA 参数高效微调。

**📊 数据集**

使用的数据集：VagueEO（10 个 EO 任务的模糊查询与标准标注），以及在 10 个主流 EO 基准（AID、DIOR、Potsdam、iSAID、xBD 等）上的测试集。

**📈 对比分析**

与现有多模态 LLM、专用 EO 模型以及其它代理系统的比较显示：意图识别准确率 95%（远超 SFT 版本），在内部任务上接近或优于最先进模型；在密集预测任务上通过工具路由达到接近或优于专用模型的精度；整体推理时间仅 1.18 秒，比 Earth‑Agent 等基线快 100 倍。

**⚠️ 局限性**

局限性：VagueEO 的规模有限，未能覆盖所有真实用户查询；工具库为静态手工构建，缺乏动态发现和集成新专家模型的能力；缺乏对外部工具错误的自我纠正或回滚机制。

---

## 232. WUTDet: A 100K-Scale Ship Detection Dataset and Benchmarks with Dense Small Objects

**arXiv ID:** 2604.07759 | [PDF](https://arxiv.org/pdf/2604.07759v1)

**作者:** Junxiong Liang `[一作]` (Wuhan University of Technology), Ryan Wen Liu `[通讯]` (Wuhan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了100,576张图像、381,378个船体实例的WUTDet大规模船舶检测数据集，并在该数据集上评估了20个基准模型。

**💡 创新点**

创新点在于大规模高分辨率船舶数据集的构建、包含大量小目标与多样化场景，以及建立统一跨数据集测试集Ship-GEN验证模型泛化能力。

**🔧 技术方法**

采用CNN、Transformer和Mamba三大主流检测框架，结合YOLOv8/YOLOX、DETR、DEIM、Mamba YOLO等多种方法。

**📊 数据集**

使用自采集的可见光图像数据（WUTDet）并结合WaterScene、WSODD等公开数据集构成Ship-GEN。

**📈 对比分析**

通过AP_50:95、AP_s/m/l、AR_50:95等指标比较，Transformer模型在总体和小目标上表现最佳，CNN模型推理速度最快，Mamba模型在准确率与速度之间取得折中。

**⚠️ 局限性**

局限在于仅提供单一类别标签、对极小目标的检测仍低、Mamba模型在复杂场景下表现欠佳，且数据仍未覆盖所有航行环境。

---

## 233. Beyond Social Pressure: Benchmarking Epistemic Attack in Large Language Models

**arXiv ID:** 2604.07749 | [PDF](https://arxiv.org/pdf/2604.07749v1)

**作者:** Steven Au `[一作]` (Independent Researcher), Sujit Noronha `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了PPT-Bench，评估大型语言模型在哲学压力下的认知攻击与谦让行为；

**💡 创新点**

提出了哲学压力分类（PPT）四种类型，并通过三层提示结构（基线、单回合压力、多回合对话）区分单回合不一致与多回合投降两种失效模式；

**🔧 技术方法**

利用Grice合作理论与哲学传统构建评测语料；采用自动判定器GPT‑4o对模型输出进行评分；对抗干预包括提示级方法（anchor、CoT、persona、self‑consistency）和机制级方法（Leading Query Contrastive Decoding、Activation Steering）；

**📊 数据集**

共90道种子问题，涵盖事实、伦理、安全、社会规范与自述五个领域，并生成180个改写与90个反驳对话，形成多层次评测集；

**📈 对比分析**

在八个不同规模模型上进行单回合和多回合实验，发现Nemotron最为稳定，单回合与多回合恢复率不完全相关；机制级干预M5+M6能在大部分模型上将所有压力类型的投降率降至近零；

**⚠️ 局限性**

限制包括判定器单一（GPT‑4o）可能引入偏差；缺乏立场与模糊度的独立标签；仅以英文数据为主；机制干预实验受限于计算资源，未覆盖所有模型与类型；干预可能产生过度抑制的风险。

---

## 234. Image-Guided Geometric Stylization of 3D Meshes

**arXiv ID:** 2604.07795 | [PDF](https://arxiv.org/pdf/2604.07795v1)

**作者:** Changwoon Choi `[一作]` (Seoul National University), Young Min Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于图像风格参考的3D网格几何化装饰方法，利用扩散模型对风格进行抽象提取并通过 Jacobian‑based 细粒度与粗粒度变形对输入网格进行几何变形。

**💡 创新点**

创新点包括：① 通过 DreamBooth + LoRA 在扩散模型中学习图像风格的低秩权重；② 采用 Score Distillation Sampling 与近似 VAE 编码器实现高效、稳定的梯度回传；③ 设计粗细层级的 cage‑guided 与 Jacobian 优化策略，并加入对称性正则化以保证大尺度几何一致性。

**🔧 技术方法**

技术栈涵盖：LoRA、DreamBooth、Stable Diffusion XL、Score Distillation Sampling、近似 VAE 编码器、Per‑face Jacobian 参数化、Cage 约束、PartField 语义分割、可微渲染器、对称性检测与正则化。

**📊 数据集**

实验使用若干（4–12 张）风格图像训练 LoRA，并在常用 3D 网格数据集（2k–20k 顶点）上验证；评价通过用户研究、定量对比与消融实验完成。

**📈 对比分析**

与 Paparazzi、Neural 3D Mesh Renderer、MeshUp、Text2Mesh、TextDeformer 等基线对比；在 32 名受试者的感知排序中，该方法在几何对齐与美学风格迁移方面均取得最佳排名，且在可视化结果上表现出更大尺度且更细腻的几何变形。

**⚠️ 局限性**

局限性包括：需要手工提供风格图像；对极端风格差异的适应性有限；对非对称或复杂拓扑网格的支持不足；近似 VAE 与粗细层级优化虽加速训练，但在某些细节恢复上仍不及直接使用完整 VAE；整体计算成本仍较高。

---

## 235. GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-efficient Large Language Model Fine-tuning

**arXiv ID:** 2604.07808 | [PDF](https://arxiv.org/pdf/2604.07808v1)

**作者:** Kaiyuan Tian `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 GRASS 框架，通过梯度均值判定层重要性，动态采样并更新关键层，从而在保持模型表达力的同时显著降低显存消耗。

**💡 创新点**

创新点包括：①使用均值梯度范数作为实时层重要性指标；②基于该指标动态更新采样概率；③引入层级优化器状态卸载与异步重叠，进一步压缩显存；④通过可调采样周期和活跃层数实现性能与效率的最佳平衡。

**🔧 技术方法**

技术手段包括：梯度统计与指数滑动平均、softmax 采样、层级激活/冻结策略、异步 CPU–GPU 状态卸载、重叠计算与通信、以及多任务实验中的混合训练调度。

**📊 数据集**

实验数据集涵盖算术推理（MultiArith、AddSub、GSM8K、AQuA、SingleEq、SVAMP）和常识推理（BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-e、ARC-c、OBQA）等多种 NLP 基准。

**📈 对比分析**

与 FFT、LoRA、DoRA、LISA、IST、OWS 等 PEFT 方法在 TinyLlama、Gemma-2B、LLaMA‑2‑7B 等模型上进行对比，GRASS 在多数基准上平均提升约 4.38 分，显存使用降低高达 19.97%，并在部分任务上逼近或超越 FFT。

**⚠️ 局限性**

局限性主要在于：①梯度统计导致额外计算开销；②评估范围仅限 decoder‑only 语言模型与常见 NLP 任务，跨架构或跨模态的适用性尚待验证。

---

## 236. More Capable, Less Cooperative? When LLMs Fail At Zero-Cost Collaboration

**arXiv ID:** 2604.07821 | [PDF](https://arxiv.org/pdf/2604.07821v1)

**作者:** Advait Yadav `[一作]` (University of Illinois Urbana-Champaign), Oliver Sourbut `[通讯]` (Future of Life Foundation)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在完全无成本、无策略复杂度的多代理环境中，评估并拆解大语言模型（LLM）在合作与执行上的失败，探究“指令-效用差距”，并测试三种低摩擦干预（明确协议、共享激励、信息遮蔽）。

**💡 创新点**

首次将合作失败与执行能力失效分离，揭示即使指令明确，LLM仍可能主动忽略信息共享；提出“指令-效用差距”框架并验证三种干预能针对性提升合作与执行效率。

**🔧 技术方法**

构建基于对话与任务交互的多回合环境，使用八种主流LLM（Gemini、Claude、OpenAI o3、GPT-5-mini、GPT-4.1-mini、DeepSeek-R1 等）作为代理，结合自动化请求/履行的因果分解实验、私密思考日志分析、干预机制实验。

**📊 数据集**

自定义实验数据集：10 名代理、100 条信息、每轮20回合、每代理2个任务（n=4）组成的任务集合；不使用公开 benchmark，而是设计“完美玩法”作为理论极限。

**📈 对比分析**

与“完美玩法”基准比较，发现模型间总任务完成率从 5.8%（GPT‑4.1‑mini）到 78.9%（Gemini‑2.5‑Pro）不等；通过自动化请求/履行拆解可将合作不足或执行不足分别提升至 90%+；三种干预分别使合作受限模型提升 20–100%，执行受限模型提升 50–200%，但大多数模型仍低于完美玩法。

**⚠️ 局限性**

实验仅限于无成本、无信息限制的理想化环境，未涵盖通信成本、带宽限制、竞争性激励等实际部署难点；结果可能低估真实场景中的合作难题；干预效果对不同规模/能力模型的适用性需进一步验证。

---

## 237. Open-Ended Video Game Glitch Detection with Agentic Reasoning and Temporal Grounding

**arXiv ID:** 2604.07818 | [PDF](https://arxiv.org/pdf/2604.07818v1)

**作者:** Muyang Zheng `[一作]` (University of California, Davis), Lifu Huang `[通讯]` (University of California, Davis)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个开放式视频游戏错误检测基准（GlitchBench），并提出了一个具备游戏上下文记忆、辩论式验证与事件级时间归一化的代理式检测框架（AGLID）。

**💡 创新点**

创新点在于：①首次实现对游戏视频中错误的开放式描述与精确时间定位；②通过游戏意识记忆提升上下文推理；③引入辩论式验证以降低误报；④使用事件级聚类与双向时间扩展实现完整错误时段归一化。

**🔧 技术方法**

主要技术包括：多模态LLM（GPT‑4o、CLIP 等）+ 视觉帧拼接、游戏元数据注入；代理式工具链（Planner‑Executor‑Reflector）；语义聚类与 Hungarian 匹配；自定义评估协议（语义评分×IoU）。

**📊 数据集**

使用来自 GamePhysics 论坛的 5,238 条游戏视频（120 个游戏）经过 GPT‑4o 伪生成 + 人工校验，得到细粒度描述和精确时间戳。

**📈 对比分析**

与传统单步预测基线相比，AGLID 在 6 个开源模型上的平均 F1 由 14.47% 提升至 36.05%，mIoU 由 0.28 提升至 0.51，整体 F1×IoU 提升至 17.05%，超过了所有专有模型的表现。

**⚠️ 局限性**

局限性在于：仍然难以达到高精度，特别是在多重错误与极短时段错误的识别；依赖大量算力与 LLM 推理；对未知游戏的泛化能力尚需进一步验证。

---

## 238. Agentivism: a learning theory for the age of artificial intelligence

**arXiv ID:** 2604.07813 | [PDF](https://arxiv.org/pdf/2604.07813v1)

**作者:** Lixiang Yan `[一作]` (Tsinghua University), Dragan Gašević `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出Agentivism学习理论，阐述AI辅助学习与真正学习的区别，并归纳了委托、验证、重构和转移四大机制。

**💡 创新点**

创新点在于将人机交互中的代理性与代理代理重新定义为学习核心，形成一套可检验的学习机制框架。

**🔧 技术方法**

主要采用文献综述与理论构建的方法；未使用具体技术实现或实验工具。

**📊 数据集**

本研究不涉及数据集，属于理论性阐述与框架提出。

**📈 对比分析**

未进行实验对比，提出六条可检验的假设，为后续实证研究提供方向。

**⚠️ 局限性**

局限在于缺乏实证验证、机制的操作化与度量尚不成熟，以及在不同学科与任务中的适用性尚待探索。

---

## 239. HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models

**arXiv ID:** 2604.07812 | [PDF](https://arxiv.org/pdf/2604.07812v1)

**作者:** Qihui Zhu `[一作]` (University of Science and Technology of China), Yinfei Pan `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多模态大型语言模型中提出一种基于注意力头重要性进行视觉令牌剪枝的方法HAWK；

**💡 创新点**

创新点在于识别并量化不同注意力头对视觉理解的贡献差异，并将头重要性权重与文本引导的注意力相结合，构建训练无关、可直接应用的剪枝框架；

**🔧 技术方法**

使用的技术包括：头重要性权重的离线统计（基于多数据集的头剥离实验），文本引导的无位置编码注意力分数计算，注意力重加权的剪枝决策；

**📊 数据集**

实验数据集涵盖10个图像视觉语言基准（HallBench、MME、TextVQA、ChartQA、AI2D、RealWorldQA、CCBench、OCRVQA、SQA-IMG、POPE）以及2个视频基准（Video‑MME、WorldSense），并在Qwen2.5‑VL‑7B与InternVL3‑8B两款主流模型上验证；

**📈 对比分析**

与DivPrune、FastV、CDPruner等现有剪枝方法对比，HAWK在保留80%视觉令牌时可保持96%原始准确率，剪枝后整体性能提升≈4–5%，同时推理延迟下降25%至30%，显著优于基线；

**⚠️ 局限性**

局限性包括：剪枝策略是静态预计算的，可能无法适应每个任务的细粒度需求；对极端高压缩比时仍会出现性能下降；尚未评估在更大规模模型或其他任务类型（如音频）中的泛化能力。

---

## 240. PolicyLong: Towards On-Policy Context Extension

**arXiv ID:** 2604.07809 | [PDF](https://arxiv.org/pdf/2604.07809v1)

**作者:** Junlong Jia `[一作]` (Tencent), Songlin Hu `[通讯]` (Tencent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PolicyLong，利用模型当前的熵信息迭代刷新长上下文训练数据，包含正负样本自适应生成；

**💡 创新点**

创新点是将数据构造转变为动态的 on‑policy 循环，实现隐式自我课程化以及自适应负样本；

**🔧 技术方法**

采用信息理论熵检测、检索、熵减验证以及二次检索构造正负样本；

**📊 数据集**

使用 RULER、HELMET、LongBench‑v2 三个长文本基准；

**📈 对比分析**

与 EntropyLong、NExtLong 进行对比，在 Qwen2.5‑3B 上在 64K–128K 语境下平均提升约 2–3 分，且不损失短文本性能；

**⚠️ 局限性**

仅在 Qwen2.5‑3B 上验证，缺乏更大模型或多语种的评估。

---

## 241. The Accountability Horizon: An Impossibility Theorem for Governing Human-Agent Collectives

**arXiv ID:** 2604.07778 | [PDF](https://arxiv.org/pdf/2604.07778v1)

**作者:** Haileleol Tibebu `[一作]` `[通讯]` (University of Illinois at Urbana-Champaign), Haileleol Tibebu (University of Illinois at Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出人机协作集体（Human‑Agent Collective, HAC）的形式化模型，并证明了在存在混合反馈循环且最小复合自主度超过阈值时，任何单一主体责任框架都无法同时满足因果归因、可预见性、非空虚性和完整性四个基本原则；

**💡 创新点**

首次给出AI治理可行性的严格边界（Accountability Horizon），揭示责任可行性与自主度、交互拓扑结构的本质关系，并证明该不可行性为结构性不可逆；

**🔧 技术方法**

使用结构因果模型（SCM）和信息论自主度向量（epistemic、executive、evaluative、social）对代理进行度量，构建责任分配的公理化框架，并用组合数学和信息理论证明不可能定理；

**📊 数据集**

构造了3,000个随机生成的合成HAC（基于Erdős‑Rényi图、随机自主度分布），作为实验数据；

**📈 对比分析**

通过理论证明与大规模实验验证相结合，展示了责任预算在阈值以下始终为零，超过阈值时立即出现责任缺口，实验结果与理论完全一致，说明模型可预测性强；

**⚠️ 局限性**

模型假设有限状态与动作空间、混合策略结构、静态交互图以及对因果与信息可达性的完备性假设；对连续空间、动态拓扑、非收缩结构方程及更一般的策略形式的适用性仍待进一步研究。

---

## 242. The Art of (Mis)alignment: How Fine-Tuning Methods Effectively Misalign and Realign LLMs in Post-Training

**arXiv ID:** 2604.07754 | [PDF](https://arxiv.org/pdf/2604.07754v1)

**作者:** Rui Zhang `[一作]` (University of Electronic Science and Technology of China), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了多种微调方法在对大语言模型进行误导（misalignment）与重新对齐（realignment）中的效果，重点考察了四款已安全对齐LLM在SFT与PFT下的表现。

**💡 创新点**

创新之处在于首次将攻击与防御方法的机制不对称性进行定量与机制化分析，并通过构造MisQA数据集以及多轮对抗实验揭示了误导与重新对齐的交互影响，为LLM安全防护提供了实证依据。

**🔧 技术方法**

使用了六种微调技术（LoRA、QLoRA、AdaLoRA、IA3、DPO、ORPO），配合LLM-Guard、GPT4o-mini等LLM-as-a-judge评估器，以及OpenCompass、vLLM推理框架，对模型进行训练与评估。

**📊 数据集**

数据集包括自构的MisQA（390问答对），公开的safe-rlhf、hh-rlhf、XSTEST、AdvBench、SafeBench、Do-Not-Answer等安全与对齐数据集。

**📈 对比分析**

实验表明ORPO在误导上最为强大，DPO在重新对齐时取得最佳安全恢复但略低实用性；LoRA样本效率最高；Gemma2对误导具有最强抵抗力；多轮误导/重新对齐会导致模型实用性与安全性逐渐下降。

**⚠️ 局限性**

研究未涉及RLHF等高成本对齐技术，评估依赖LLM-judge且可能存在偏差；未探讨多源误导/重新对齐情境；缺少对专有LLM的实验，限制了结果的普适性。

---

## 243. MIMIC-Py: An Extensible Tool for Personality-Driven Automated Game Testing with Large Language Models

**arXiv ID:** 2604.07752 | [PDF](https://arxiv.org/pdf/2604.07752v1)

**作者:** Yifei Chen `[一作]` (McGill University), Lili Wei `[通讯]` (McGill University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于Python的可扩展游戏自动测试工具MIMIC-Py，能够将个性化驱动的LLM代理转化为可跨游戏、可模块化的测试框架；

**💡 创新点**

核心创新在于将原JavaScript原型拆解为规划、执行、记忆三大可插拔组件，公开人格特质作为可配置输入，支持多种交互方式（API直接调用或代码生成），极大降低了迁移成本与工程量；

**🔧 技术方法**

技术上依赖大型语言模型（LLM）完成混合规划（底层递归+顶层分解）、动作执行与摘要；使用向量数据库ChromaDB实现检索增强生成（RAG）存储与检索记忆与技能；采用PathOS人格模型为代理注入七种行为特质；

**📊 数据集**

使用了多个游戏环境作为测试集，包括Dungeon Adventures、Shattered Pixel Dungeon和Minecraft；并在这些环境中收集了游戏状态与动作执行的记忆与技能作为检索语料；

**📈 对比分析**

与随机策略和已有最先进的Minecraft代理进行对比，MIMIC-Py在分支覆盖率提升最多1.30倍，交互级覆盖率提升14.46倍，且在复杂多步任务的求解上表现优于现有代理；

**⚠️ 局限性**

主要局限在执行效率与成本：每个动作平均耗时12.4秒、成本约0.06美元，难以适用于对时延要求高的FPS等游戏；此外对游戏接口的适配仍需一定手工配置，未来计划引入本地微调模型以降低延迟和费用。

---

## 244. Capture-Quiet Decomposition: A Verification Theorem for Chess Endgame Tablebases

**arXiv ID:** 2604.07907 | [PDF](https://arxiv.org/pdf/2604.07907v1)

**作者:** Alexander Pavlov `[一作]` `[通讯]` (ProofCodec), Alexander Pavlov (ProofCodec)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 Capture-Quiet Decomposition (CQD) 定理，用于对象棋残局表格库的 Win-Draw-Loss（WDL）标签进行完整、结构化的验证；

**💡 创新点**

创新点在于将所有合法局面划分为终端、捕获和安静三类，并利用捕获局面通过已验证的子残局作为锚点，既消除了自我一致性陷阱（如全棋谱为 Draw 的伪解），又显著减少了后向检验所需的计算量；

**🔧 技术方法**

采用的技术包括结构化分解定理、递归强归纳证明、后向一致性检查、CEGAR（Counterexample‑Guided Abstraction Refinement）构建决策树、以及 Rust/Python 双层实现的高效遍历和验证；

**📊 数据集**

使用的数据集为所有 35 个三、四棋子残局、110 个五棋子残局和 372 个六棋子残局，总计 517 个残局，涵盖数十亿个合法位置；

**📈 对比分析**

与完整后向检验（O(N·b)）相比，CQD 在 4 棋子时速度相当，5-6 棋子时可提升约 1.3–2 倍；在所有验证的残局中，CQD 的总违规计数与完整后向检验完全一致；

**⚠️ 局限性**

主要限制是需要先验证所有子残局；对高棋子数残局的捕获比例仍是估算值；并且若决策树深度不足，仍可能出现已知的决策树模型误差。

---

## 245. Task-Adaptive Retrieval over Agentic Multi-Modal Web Histories via Learned Graph Memory

**arXiv ID:** 2604.07863 | [PDF](https://arxiv.org/pdf/2604.07863v1)

**作者:** Saman Forouzandeh `[一作]` (Royal Melbourne Institute of Technology University), Mahdi Jalili `[通讯]` (Royal Melbourne Institute of Technology University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于任务自适应图记忆的多模态 Web 导航历史检索框架 ACGM。

**💡 创新点**

创新点包括：① 通过策略梯度直接优化检索边，学习任务特定的相关性；② 为视觉、文本、结构信息学习模态特定的时间衰减；③ 构造稀疏图结构，实现 O(log T) 子线性检索。

**🔧 技术方法**

技术方法包括：神经相关性预测器 gϕ、双阶段预训练+强化学习、模态嵌入（CLIP、RoBERTa）、四叉树层次检索、模态衰减 λm 及其正则化。

**📊 数据集**

使用 WebShop、VisualWebArena、Mind2Web 三个多模态 Web 导航基准数据集进行评估。

**📈 对比分析**

与 19 个基线（稠密检索、重排序、多模态模型、图检索等）对比，ACGM 在 WebShop 上 nDCG@10 达 82.7（+9.3 与 GPT‑4o），Precision@10 89.2%（+7.7），检索速度提升 4.8×，内存占用仅 2.1 GB。

**⚠️ 局限性**

局限性包括：依赖任务成功奖励，训练时奖励方差大；对训练分布敏感，跨域迁移仍需进一步验证；仅在 Web 交互任务上验证，未覆盖非 Web 多模态场景。

---

## 246. Why Are We Lonely? Leveraging LLMs to Measure and Understand Loneliness in Caregivers and Non-caregivers

**arXiv ID:** 2604.07834 | [PDF](https://arxiv.org/pdf/2604.07834v1)

**作者:** Michelle Damin Kim `[一作]` (Emory University), Jinho D. Choi `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并应用LLM驱动的处理管道，利用改进的UCLA孤独量表与专家制定的原因分类框架，对Reddit上照顾者与非照顾者的帖子进行孤独度测量、原因归类和人口统计提取，最终生成高质量的孤独数据集并比较两人群的孤独体验差异。

**💡 创新点**

① 专为照顾者量身定制的孤独评估与原因分类框架；② 将LLM用于端到端的数据清洗、相关性判断、孤独评估、原因归类及人口统计抽取；③ 通过该管道验证Reddit可用于构建多样化、规模化的孤独数据集，并揭示照顾者与非照顾者在孤独原因上的显著差异。

**🔧 技术方法**

使用GPT‑5、GPT‑5‑nano、GPT‑4o等大型语言模型完成文本预处理、相关性筛选、孤独度评估、原因分类与人口统计信息抽取；结合Python Reddit API Wrapper、tiktoken、正则表达式进行数据抓取与清洗。

**📊 数据集**

采集15个Reddit子版块（8个照顾者相关子版块、7个非照顾者子版块）共计28,351条照顾者帖子与41,619条非照顾者帖子；通过LLM管道筛选后得到387条高孤独照顾者帖子与908条高孤独非照顾者帖子。

**📈 对比分析**

利用LLM评估框架对孤独度进行评分，阈值7筛选高孤独帖子；随后用同一模型进行原因分类。孤独评估平均准确率分别为76.09%（照顾者）和79.78%（非照顾者）。原因分类的微观聚合F1分别为0.825（照顾者）和0.8（非照顾者），宏观聚合F1较低，表明模型对频繁类别表现良好但稀有类别仍有提升空间。

**⚠️ 局限性**

样本不平衡导致宏观F1低；单一本科生标注器可能引入偏差；标注样本量小，限制了模型验证力度；未通过IRB审核，数据不可公开；仅使用公开文本，隐私保护措施有限。

---

## 247. Are GUI Agents Focused Enough? Automated Distraction via Semantic-level UI Element Injection

**arXiv ID:** 2604.07831 | [PDF](https://arxiv.org/pdf/2604.07831v1)

**作者:** Wenkui Yang `[一作]` (UCAS), Ran He `[通讯]` (UCAS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了语义级UI元素注入攻击，利用安全对齐且无害的图标叠加到GUI截图上，诱导GUI代理误解指令并错误点击；

**💡 创新点**

创新点在于：①将攻击视为语义级注入而非像素扰动，绕过安全过滤；②设计了Editor–Overlapper–Victim三模块及迭代深度搜索+反馈驱动的策略选择，实现高效黑盒攻击；③证明了攻击的跨模型可转移性和对目标元素的持续吸引力；

**🔧 技术方法**

使用的技术包括：基于Qwen3-VL的文本-图像多模态嵌入；FAISS近似检索和LMDB存储的图标库；迭代Depth×Pass@N搜索与贪心累计；元诊断与策略适配的Prompt工程；

**📊 数据集**

实验数据集由OS-Atlas、SeeClick、AMEX、ShowUI等多平台GUI截图组成，最终选取885个样本供攻击评估；

**📈 对比分析**

通过与随机注入基线对比，展示了L1/L2攻击成功率（ASR）和点击距离等指标，最佳攻击在D=5时可达32–88%的ASR，且与随机注入相比提升3–4倍；

**⚠️ 局限性**

局限性包括：仅在公开可用的多平台图标库内有效；攻击对图标可用性和检索质量敏感；对极强模型仍存在一定成功阈值；未评估对实时交互和动态UI的适用性；

---

## 248. Sampling-Aware 3D Spatial Analysis in Multiplexed Imaging

**arXiv ID:** 2604.07890 | [PDF](https://arxiv.org/pdf/2604.07890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 249. LPM 1.0: Video-based Character Performance Model

**arXiv ID:** 2604.07823 | [PDF](https://arxiv.org/pdf/2604.07823v1)

**作者:** Ailing Zeng `[一作]`, Zi Ye `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

LPM 1.0 通过协同设计的系统架构，生成身份一致、语言与非语言行为同步、视觉质量高且支持流式和长时段的对话视频。

**💡 创新点**

其创新点在于将表达质量、实时推理与长时段稳定性三大目标统一在数据、多模态条件、生成、流式处理与在线稳定化的系统层面上，实现了可部署的对话视频生成。

**🔧 技术方法**

技术方案包括多模态预训练语言模型、语音合成、视觉生成网络以及实时推理和训练优化的在线稳定化模块。

**📊 数据集**

使用了多维度数据集：对话数据集、字幕数据集、身份标签数据集，涵盖语音、文本、面部与肢体行为信息。

**📈 对比分析**

通过基准测试、离线比较、在线比较及消融实验等方法评估，系统在对话连贯性、表达自然度与视觉一致性方面均优于传统单向视频生成方法，且能够满足实时推理要求。

**⚠️ 局限性**

局限性包括：仅支持单一摄像头面向角色的单方对话，缺乏多方协作、长时记忆、动态环境交互及多视角3D一致性等功能。

---

## 250. Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency

**arXiv ID:** 2604.07904 | [PDF](https://arxiv.org/pdf/2604.07904v1)

**作者:** Mingqing Xiao `[一作]` (Microsoft Research Asia), Dongsheng Li `[通讯]` (Microsoft Research Asia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Kuramoto oscillatory Phase Encoding（KoPE），将生物学相位同步机制与 Vision Transformer 结合，提升模型的训练、参数与数据效率，同时增强对结构化任务（如分割、视觉语言对齐和抽象视觉推理）的表现。

**💡 创新点**

创新点在于把 Kuramoto 动力学作为可扩展的相位同步框架注入 Transformer 的率-相位双动态，利用相位差作为结构诱导偏置，实现了从数据中自适应学习绑定关系的机制。

**🔧 技术方法**

使用了 Kuramoto 同步模型、相位编码与旋转注意力、复杂数形式的相位融合、Self‑supervised 学习（SimDINOv2）、CLIP 视觉‑语言对齐、Mask2Former、SETR‑PUP 等技术。

**📊 数据集**

在 ImageNet‑1K、ADE‑20K、COCO、ARC‑AGI、CLIP benchmark（40 个零样本数据集）等公开数据集上进行评估。

**📈 对比分析**

通过与标准 ViT 在相同训练设置下的对比，KoPE 在多任务中表现出 1–3% 的准确率提升、15–20% 的数据/训练样本节省以及更高的参数与 FLOPs 效率，且在分割、视觉‑语言零样本分类和 ARC‑AGI 任务中显著优于基线。

**⚠️ 局限性**

局限性包括对相位初始化的依赖、相位耦合参数的敏感性，以及尚未在所有任务域和更大模型规模下验证其普适性和鲁棒性。

---

## 251. AFGNN: API Misuse Detection using Graph Neural Networks and Clustering

**arXiv ID:** 2604.07891 | [PDF](https://arxiv.org/pdf/2604.07891v1)

**作者:** Ponnampalam Pirapuraj `[一作]` (IIT Hyderabad), Jyothi Vedurada `[通讯]` (IIT Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于图神经网络的 API 用法误用检测框架，利用 API Flow Graph（AFG）对代码进行结构化建模，自动识别常见与异常 API 使用模式；

**💡 创新点**

创新点在于：①设计了包含数据流、控制流和调用序列三种边的 AFG；②采用自监督上下文预测预训练的 GNN 生成流感知嵌入；③通过聚类实现无监督误用检测并推荐常用用法；

**🔧 技术方法**

技术上使用了 AST+CFG+DFG 的 AFG 构建、Graph Convolutional Network（GCN）/Relational GCN（RGCN）网络、BIRCH 聚类和 Davies‑Bouldin 指标自适应选簇；

**📊 数据集**

数据集包括 1.5M 规模的 Java API 用法实例（来自 Code2Seq 的 9,500+ GitHub 项目）以及 MUBench 用法误用基准；

**📈 对比分析**

与 GraphCodeBERT、UnixCoder 等小型 LM 以及现有误用检测器比较，AFGNN 在 API 用法聚类的 RI/MI、误用检测的 F1 等指标均显著提升（最高 F1 0.65，比最佳基线高 11%）；

**⚠️ 局限性**

局限性：需要足够的历史用例才能有效聚类，稀有或新 API 可能表现不佳；模型依赖于旧用法，API 变更后需重新预训练；

---

## 252. LCMP: Distributed Long-Haul Cost-Aware Multi-Path Routing for Inter-Datacenter RDMA Networks

**arXiv ID:** 2604.07836 | [PDF](https://arxiv.org/pdf/2604.07836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 253. Contextualising (Im)plausible Events Triggers Figurative Language

**arXiv ID:** 2604.07885 | [PDF](https://arxiv.org/pdf/2604.07885v1)

**作者:** Annerose Eichel `[一作]`, Sabine Schulte im Walde `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究收集并分析了人类与四大主流LLM对英语svo事件的（非）字面性与可行性判断及情境化示例，探讨抽象度对可行性和字面性的影响。

**💡 创新点**

创新点在于将事件可行性与字面性与概念抽象度联结，构建人机对比的细粒度评估框架，并揭示LLM对可行性偏向与情境修复能力的不足。

**🔧 技术方法**

使用了指令版的Qwen3-4B、Gemma3-4B、Mistral-7B和Llama3.1-8B四大LLM，结合零/少量示例提示进行标签预测与上下文生成。

**📊 数据集**

采用了PAP数据集的411个svo三元组（含原始可行性标注和抽象度标签），并扩充至6497条人类/LLM判断和对应上下文。

**📈 对比分析**

通过与人类主流标注对比（≥60%多数决定）以及χ²与Cramér’s V等统计检验，发现LLM在可行性和字面性判断上与人类显著偏离，且对抽象度的敏感度远低于人类，整体性能不足。

**⚠️ 局限性**

主要限制包括仅针对英语、对LLM生成的文本解析采用简单规则、缺乏对不同语言或更复杂语义场景的验证，以及模型固有的偏见可能影响评估结果。

---

## 254. Reinforcement-Guided Synthetic Data Generation for Privacy-Sensitive Identity Recognition

**arXiv ID:** 2604.07884 | [PDF](https://arxiv.org/pdf/2604.07884v1)

**作者:** Xuemei Jia `[一作]` (Wuhan University), Zheng Wang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文提出一种基于强化学习的合成数据生成框架，利用通用域预训练模型在隐私敏感身份识别任务中适配并生成高保真、多样化的训练样本。

**💡 创新点**

通过冷启动初始化、任务特定多目标奖励（语义一致、分布覆盖、表达多样）以及动态样本选择三阶段的自适应循环，突破数据稀缺下生成模型与任务性能互惰循环。

**🔧 技术方法**

结合 Diffusion Transformer（DiT）、RL 调优（DPOK）、多目标奖励函数、内存银行、投影学习与虚拟梯度选择等技术。

**📊 数据集**

在人行重识别的 Market‑1501、CUHK03‑NP 以及面部识别的 CASIA‑WebFace 子集、LFW、AgeDB、CFP‑FP、CA‑LFW、CP‑LFW、RFW 等数据集上进行验证。

**📈 对比分析**

与现有真实图像增强、模拟扩增及合成增强方法对比，在 Market‑1501 上 mAP 88.6%（比基线+3.2%），CUHK03‑NP 上 76.6%（比基线+2.5%）；在 CASIA 子集面部验证平均准确率 79.07%（比前沿方法高约1%），并在跨种族 RFW 上取得最高平均准确率 69.78%，显示出更好的泛化与公平性。

**⚠️ 局限性**

依赖预训练骨干的表达能力；奖励函数需要任务调参；当前仅验证图像模式，未扩展至视频或事件等多模态；在与先前对齐度较差的域中效果可能受限。

---

## 255. MemReader: From Passive to Active Extraction for Long-Term Agent Memory

**arXiv ID:** 2604.07877 | [PDF](https://arxiv.org/pdf/2604.07877v1)

**作者:** Jingyi Kang `[一作]` (MemTensor Technology), Zhiyu Li `[通讯]` (MemTensor Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MemReader 系列模型，通过主动决策和工具调用实现长时记忆提取与管理。

**💡 创新点**

将记忆提取从被动一次性生成转变为主动的状态维护，结合 ReAct 思考‑行动框架和专门的四种记忆工具；同时提供轻量级 0.6B 版和 4B 版。

**🔧 技术方法**

基于 Qwen3‑4B/0.6B 的 LLM，使用 ReAct 机制、工具调用、GRPO 强化学习、SFT 预训练、奖励分层、LLM 评判和蒸馏等技术。

**📊 数据集**

构建多轮对话轨迹数据（写入、检索、缓冲、忽略四种路径），并在 LOCOMO、LongMemEval、HaluMem‑Medium 三个公开基准上评测。

**📈 对比分析**

与 Mem0、MemOS、MemU 等开源系统及 GPT‑4o‑mini 进行对比；MemReader‑4B‑GRPO 在整体分数、知识更新、时序推理等指标上领跑，MemReader‑0.6B 在结构化提取与效率上优于小模型基线。

**⚠️ 局限性**

缺乏记忆编辑、冲突检测、层级抽象等高级操作；评估仅基于离线基准，缺少真实在线交互验证；与回答生成模块耦合度不高，需要进一步统一优化。

---

## 256. Silencing the Guardrails: Inference-Time Jailbreaking via Dynamic Contextual Representation Ablation

**arXiv ID:** 2604.07835 | [PDF](https://arxiv.org/pdf/2604.07835v1)

**作者:** Wenpeng Xing `[一作]` (Zhejiang University), Meng Han `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理时对大型语言模型隐藏状态进行动态低秩子空间消融的技术，能够在不改动模型权重的前提下绕过安全拒绝机制

**💡 创新点**

创新点在于将拒绝子空间的定位与抑制实时结合：通过梯度归因快速识别实例特定的低秩拒绝子空间，并在生成过程中对该子空间进行软/硬遮蔽，从而实现精准、可逆的干预

**🔧 技术方法**

使用梯度归因（对拒绝词表的梯度计算）、低秩子空间筛选、可调遮蔽系数与自适应迭代等技术，在每个时间步动态修改隐藏状态

**📊 数据集**

在 AdvBench、PKU‑Alignment、ToxicChat 等对抗与安全评测基准上进行实验

**📈 对比分析**

与传统的白盒梯度优化攻击（如 GCG、PEZ）和静态剪枝方法（如 LED）对比，CRA 在攻击成功率上提升约 15.2 倍，显著优于 DSN，且保持生成质量

**⚠️ 局限性**

局限性包括：推理时需要额外梯度计算，导致轻微延迟；实验仅覆盖稠密 Transformer，尚未验证在 Mixture‑of‑Experts 或状态空间模型上的适用性

---

## 257. Optimization of 32-bit Unsigned Division by Constants on 64-bit Targets

**arXiv ID:** 2604.07902 | [PDF](https://arxiv.org/pdf/2604.07902v1)

**作者:** Shigeo Mitsunari `[一作]` (Cybozu Labs, Inc.), Takashi Hoshino `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对64位CPU环境下的32位无符号整数常数除法，提出了新的优化方法

**💡 创新点**

将传统GM方法中33位常数除法的三步移位序列改为单乘法+取高64位，消除128位右移的瓶颈

**🔧 技术方法**

利用乘法指令（x86-64中的mul / shrd，AArch64中的umulh）和对乘积高64位的直接提取

**📊 数据集**

在LLVM/Clang与GCC中对32位无符号除法微基准（测试除数7、19、107等）进行评测

**📈 对比分析**

将修改后的编译器生成的汇编与原始代码进行对比，使用时间命令测得Xeon 1.67×、Apple M4 1.98×的速度提升

**⚠️ 局限性**

仅适用于33位常数除法的情况，对其他除法模式（如移位、特殊大小的除数）不产生影响，且需要64位寄存器支持

---

## 258. To Copilot and Beyond: 22 AI Systems Developers Want Built

**arXiv ID:** 2604.07830 | [PDF](https://arxiv.org/pdf/2604.07830v1)

**作者:** Rudrajit Choudhuri `[一作]` (Oregon State University), Anita Sarma `[通讯]` (Oregon State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对860名微软开发者的问卷进行人机协作的多模型主题分析，梳理出22个他们认为需要的AI系统及其约束，提出“有界委托”概念；

**💡 创新点**

首次系统性捕捉开发者对AI系统的具体需求与边界，并提出有界委托理论，结合多模型协同分析法；

**🔧 技术方法**

采用人机协作、多模型（GPT‑5.2、Gemini 3.1 Pro、Claude Opus 4.6）主题发现、协同代码本梳理、Krippendorff α一致性评估等技术；

**📊 数据集**

来自微软内部860名开发者的问卷开放式回答数据，覆盖开发、设计、质量、运维、元工作五大任务类别；

**📈 对比分析**

通过三模型的交叉验证与人类审核实现主题一致性，平均Krippendorff α达到0.94；本研究为定性分析方法提供可复制的评估流程，但未涉及传统性能指标；

**⚠️ 局限性**

构造效度受限于自报偏好，内部效度受限于单一时间点、可能的自选样本偏差；外部效度受限于微软组织背景；对LLM编码的误差依赖需进一步检验；

---

## 259. Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders

**arXiv ID:** 2604.07825 | [PDF](https://arxiv.org/pdf/2604.07825v1)

**作者:** Jaehyun Lee `[一作]` (Pohang University of Science and Technology), Hwanjo Yu `[通讯]` (Pohang University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

使用大语言模型进行推荐时，通过选择性地补充知识来填补推荐缺口，构建了基于知识打分的LLM推荐器

**💡 创新点**

提出了多种基于知识打分的选择性补充策略（Acc、Pop、MinK、EigV、SeaKR、Self），相比统一补充能显著提升推荐质量

**🔧 技术方法**

利用大型语言模型（Llama-8B、Mistral-7B、Qwen-7B、Qwen-32B）+知识打分机制和自研的选择性补充算法

**📊 数据集**

在A-Beauty、A-Gift、ML-1M、Steam四个真实推荐数据集上进行实验

**📈 对比分析**

与无补充、均匀元数据/维基补充以及随机/SASRec候选生成方法对比，Recall@1平均提升10%~23%，部分模型提升高达22%

**⚠️ 局限性**

仅评估Recall@1，缺乏多指标和长期用户实验；选择性策略对LLM规模和知识库内容敏感，扩展性与计算开销待进一步验证

---

## 260. PanoSAM2: Lightweight Distortion- and Memory-aware Adaptions of SAM2 for 360 Video Object Segmentation

**arXiv ID:** 2604.07901 | [PDF](https://arxiv.org/pdf/2604.07901v1)

**作者:** Dingwen Xiao `[一作]` (Hong Kong University of Science and Technology), Lin Wang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出PanoSAM2，一种改进版的SAM2，专门用于360°全景视频目标分割，解决投影畸变与记忆稀疏问题。

**💡 创新点**

创新点包括：①Pano-Aware解码器通过wrap padding和迭代畸变细化实现全景边界一致；②Distortion-Guided Mask Loss根据ERP畸变权重增强稀疏前景和边缘学习；③Long-Short Memory Module将长期对象指针与短期记忆融合，提升时序一致性。

**🔧 技术方法**

技术手段包括SAM2架构、Hiera backbone、PC Block、FiLM、遮挡采样策略、BCE+Dice加权损失、Bilinear投影重映射等。

**📊 数据集**

使用公开的360VOTS与PanoVOS两大全景视频分割数据集进行训练与评估。

**📈 对比分析**

与SAM2、SAM2Long、PSCFormer及多种视角VOS方法对比，PanoSAM2在360VOTS上取得J&F 65.8（+5.6），在PanoVOS验证集上78.1（+6.7），在测试集上73.4，显著超过现有最优模型。

**⚠️ 局限性**

局限性包括仅支持单目标分割，对远距离或高速运动导致的投影严重畸变、遮挡时仍易出现身份漂移，且对多目标复杂交互处理不足。

---

## 261. TSUBASA: Improving Long-Horizon Personalization via Evolving Memory and Self-Learning with Context Distillation

**arXiv ID:** 2604.07894 | [PDF](https://arxiv.org/pdf/2604.07894v1)

**作者:** Xinliang Frederick Zhang `[一作]` (University of Michigan), Lu Wang `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个双翼的长期个性化LLM框架，通过动态记忆写入与内部化记忆读取提升模型对长时序用户历史的把握能力。

**💡 创新点**

①提出结构化算法演进的动态记忆写入机制，包含 ADD/UPDATE/RECONCILE 等动作并对时间信息进行显式标注；②通过自学习合成数据与教师‑学生上下文蒸馏方式实现参数化内部化记忆，从而解决训练‑推理差距与质量‑效率权衡。

**🔧 技术方法**

动态记忆演进算法、核心观察提取与压缩、内存管理器、教师‑学生上下文蒸馏、top‑d KL近似、synthetic QA生成、Qwen‑3 LLM、检索增强生成 (RAG) 等技术。

**📊 数据集**

LoCoMo 长期对话数据集以及 LongMemEval 评测基准。

**📈 对比分析**

与 Mem0、Memory‑R1、A‑Mem、LoCoMo 等基线在 F1/ROUGE/BLEU/LLM‑Judge 等指标上对比，模型规模从 4B 到 32B，最终在 LoCoMo 上获得 45.24 F1，较 Memory‑R1 提升 4.9%，较 Mem0 提升 35‑50%；在质量‑效率上实现 Pareto 改进，使用更少 token 达到更高 F1。

**⚠️ 局限性**

仅在两大基准上评估，缺乏更广泛任务的验证；未测试超 32B 模型，模型训练耗时大且能耗高；合成数据生成侧重事实 QA，未覆盖抽象推理。

---

## 262. Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs

**arXiv ID:** 2604.07888 | [PDF](https://arxiv.org/pdf/2604.07888v1)

**作者:** Binxing Xu `[一作]`, Yike Guo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

演示如何使用ACL风格文件与LuaLaTeX或XeLaTeX编译文档

**💡 创新点**

无实际研究创新，仅提供示例

**🔧 技术方法**

使用LuaLaTeX或XeLaTeX进行排版

**📊 数据集**

无数据集

**📈 对比分析**

无实验比较，未评估性能

**⚠️ 局限性**

仅为模板，缺乏可复现性与功能验证

---

## 263. Language Preferences and Practices in Multilingual EdTech: Flexible Primary Language Use with Secondary Language Support

**arXiv ID:** 2604.07843 | [PDF](https://arxiv.org/pdf/2604.07843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 264. Linear Representations of Hierarchical Concepts in Language Models

**arXiv ID:** 2604.07886 | [PDF](https://arxiv.org/pdf/2604.07886v1)

**作者:** Masaki Sakata `[一作]` (Tohoku University), Kentaro Inui `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究语言模型内部表示如何编码层次关系，提出 Linear Hierarchical Encoding（LHE）方法，训练层次深度与领域特定的线性变换，并通过线性解码和因果干预验证层次信息可被提取和利用。

**💡 创新点**

创新点在于：①将多词实体纳入层次分析，扩展到中间层而非仅限于未嵌入层；②引入深度特定的线性映射，揭示层次信息集中在低维子空间并且该子空间在不同领域中高度相似；③结合拓扑数据分析（TDA）量化不同领域内部层次结构的相似性。

**🔧 技术方法**

采用的技术包括：线性关系嵌入（Linear Relational Concepts）和线性变换学习、伪逆求解、低秩 SVD 分解、PCA 可视化、TDA 与 Wasserstein 距离、以及对模型内部表示的因果干预。

**📊 数据集**

使用的数据集涵盖五个领域（位置、人物、组织、生命体、研究主题）的层次结构数据，经过基于模型推理的过滤后划分为子树训练/测试；实验评估在四个 decoder‑only 语言模型（Llama 3.2 3B、Llama 3.1 8B、Qwen3 8B、Qwen3 14B）上的表现。

**📈 对比分析**

与 SVM 和 Input Averaging 两种基线相比，LHE 在 Accuracy（0.5–0.9）和 Causality（0.35–0.7）两项指标上均显著更优；不同模型的得分差异说明模型容量和架构对层次编码的影响；跨域实验显示 Accuracy 稳定但 Causality 在领域移位时显著下降。

**⚠️ 局限性**

局限性包括：①对大模型 Qwen3 14B 的因果效能低，提示线性映射不足以捕捉其内部表示；②实验仅覆盖已知层次结构，未探究训练动态或上下文条件下层次表示的变化；③多词实体处理仍受限于分词与层次深度的匹配；④因果干预仅针对单个层次层面，未覆盖更复杂的层次结构或多关系。

---

## 265. Design and empirical validation of a stock-Android software architecture for Wi-Fi Direct multi-group communication

**arXiv ID:** 2604.07889 | [PDF](https://arxiv.org/pdf/2604.07889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 266. From Clicking to Moving: Embodied Micro-Movements as a New Modality for Data Literacy Learning

**arXiv ID:** 2604.07881 | [PDF](https://arxiv.org/pdf/2604.07881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 267. On the Decompositionality of Neural Networks

**arXiv ID:** 2604.07868 | [PDF](https://arxiv.org/pdf/2604.07868v1)

**作者:** Junyong Lee `[一作]` (Yonsei University), Jieung Kim `[通讯]` (Yonsei University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了神经网络可分解性（decompositionality）的形式化定义，并给出了基于决策边界的语义契约；

**💡 创新点**

创新点在于将可分解性定义为边界语义保留与结构分离的双重契约，并设计了基于边界采样与结构化掩码学习的分解框架LBMask；

**🔧 技术方法**

采用了决策边界近似（低logit margin）、PGD + 二分精炼生成边界样本、结构化掩码学习（per-neuron/channel）以及对掩码的后处理以保持维度一致；

**📊 数据集**

使用了BERT（DBPedia‑14、AG News）、ResNet‑34（CIFAR‑10）和DeiT‑small（CIFAR‑10）等公开数据集进行实验；

**📈 对比分析**

与Wanda、MI pruning等传统结构化/无结构化剪枝方法对比，LBMask在BERT‑DBPedia‑14上同时满足语义保留和结构分离，显示出更优的边界一致性与模块化程度；在其他任务或视觉模型上则表现出无法同时满足两项要求；

**⚠️ 局限性**

局限性包括：只验证了有限的模型与任务，未证明可扩展到更大规模或不同网络；分解效果对阈值设定敏感；在视觉模型中始终无法实现完整契约，说明方法受限于模型的分布式表示。

---

## 268. Information-Theoretic Requirements for Gradient-Based Task Affinity Estimation in Multi-Task Learning

**arXiv ID:** 2604.07848 | [PDF](https://arxiv.org/pdf/2604.07848v1)

**作者:** Jasper Zhang `[一作]` (Great Neck South High School), Bryan Cheng `[通讯]` (Great Neck South High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在多任务学习中证明梯度冲突分析需要任务共享训练样本，并给出30-40%重叠阈值以实现可靠的任务关系推断。

**💡 创新点**

提出样本重叠是梯度分析的根本信息理论条件，揭示了七年多任务学习结果不一致的根源，并实现了从梯度到生物学通路恢复的桥梁。

**🔧 技术方法**

使用梯度相似度（余弦相似度）、梯度冲突矩阵、信息论分析、聚类与ARI/NMI评估，以及基于梯度预测MTL收益的方法。

**📊 数据集**

在六大分子性质预测数据集（Tox21, ToxCast, SIDER, Tox21+ADME, 亲和性基因组, QM9）中进行验证。

**📈 对比分析**

与经验相关性、路径注释以及单/多任务性能比较，梯度相似度与真实任务关系的皮尔逊相关系数最高达0.94，预测MTL收益相关系数0.71，任务分组提升3-4%精度。

**⚠️ 局限性**

局限性是需要至少40%的样本重叠，现有标准基准如MoleculeNet、TDC等重叠不足，限制了方法在大多数公开数据集上的直接应用。

---

## 269. Dynamic Attentional Context Scoping: Agent-Triggered Focus Sessions for Isolated Per-Agent Steering in Multi-Agent LLM Orchestration

**arXiv ID:** 2604.07911 | [PDF](https://arxiv.org/pdf/2604.07911v1)

**作者:** Nickson Patel `[一作]` `[通讯]` (Independent Researcher), Nickson Patel (Independent Researcher)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种动态注意力上下文缩放（DACS）机制，能在多代理LLM编排中通过“注册模式”和“聚焦模式”实现对每个代理的上下文隔离，从而显著降低上下文污染并提升决策质量。

**💡 创新点**

创新点在于引入了代理触发的、非对称的上下文模式切换：聚焦模式下仅保留当前代理的完整上下文并将其它代理压缩为简短摘要，实现了确定性、无压缩的上下文隔离，彻底消除了交叉代理污染。

**🔧 技术方法**

技术手段包括：有限状态机式调度器、SteeringRequest协议、基于 token 预算的上下文构造器、精确计数与日志记录；实验使用 MiniMax‑M2.7 与 Claude Haiku 4.5 LLM，评估采用 LLM‑as‑judge 进行结果校验。

**📊 数据集**

数据集主要为八个人工构造的实验场景（跨代理数、异质性、决策密度），共计 160 轮试验；Phase 4 则用 Claude Haiku 4.5 代理生成自由文本问询，额外进行 40 轮真实代理实验；未使用公开 benchmark 数据集。

**📈 对比分析**

与平面上下文基线相比，DACS 在 160 轮合成试验中实现 90–98% 的指挥准确率（vs. 21–60%），错误代理污染率从 28–57% 降至 0–14%，上下文效率提升 2.1–3.5×；Phase 4 真实代理试验同样显示 17–20 pp 的准确率提升，且相同趋势随代理数递增。

**⚠️ 局限性**

局限性包括：依赖关键词匹配评估，易受词汇重叠影响；Phase 4 仅覆盖 N=3、5、低决策密度，未验证 N=10 或高 D 的可扩展性；对话模型在聚焦模式下引用注册信息导致污染测量偏高，且未独立消融中断协议或进一步验证跨模型泛化。

---

## 270. Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate

**arXiv ID:** 2604.07874 | [PDF](https://arxiv.org/pdf/2604.07874v1)

**作者:** Fangyue Liu `[一作]` (Peking University), Peng Chen `[通讯]` (Tencent)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在生产环境中实现了一套在线‑离线推理共址系统，联合控制抢占延迟与频率，显著提升 GPU 利用率。

**💡 创新点**

创新点在于将 GPU 通道控制与子层级 KV‑缓存回收、动态预留相结合，实现在毫秒级以下的抢占延迟，并将抢占频率限制在每个在线请求至多一次。

**🔧 技术方法**

采用 CUDA 通道控制（ioctl）、子层级内存回收+清洗策略、MIAD‑风格动态预留、基于多 GPU 的吞吐模型调度，以及仅需一行驱动改动与 20 行框架补丁的实现。

**📊 数据集**

使用了公司内部 8,054 张 GPU 的生产工作负载和 10 组在线‑离线工作负载对比实验，没有公开数据集。

**📈 对比分析**

通过与 KernelPreempt+UVM、GPreempt、Channel+Prism 等基线在 TTFT、TPOT 与离线吞吐量上的对比，Valve 使 TTFT 增加低于 5%、TPOT 增加低于 2%，且离线吞吐量与 Channel+Prism 接近，显著优于其它基线。

**⚠️ 局限性**

局限性包括依赖 CUDA 通道控制（仅对 Pascal+ 及以上 GPU 适用）、对多 GPU 同步的假设、可能不兼容所有模型并行模式，以及在极端在线峰值下仍可能出现冷启动延迟。

---

## 271. PyVRP$^+$: LLM-Driven Metacognitive Heuristic Evolution for Hybrid Genetic Search in Vehicle Routing Problems

**arXiv ID:** 2604.07872 | [PDF](https://arxiv.org/pdf/2604.07872v1)

**作者:** Manuj Malik `[一作]` (Singapore Management University), Zhiguang Cao `[通讯]` (Singapore Management University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于元认知进化编程（MEP）的框架，用语言模型自动化演化车辆路径规划（VRP）中的Hybrid Genetic Search（HGS）核心组件，获得比手工设计更优的启发式；

**💡 创新点**

核心创新在于将LLM的推理能力转化为结构化的Reason‑Act‑Reflect循环，并引入Domain‑Aware Initialization，为LLM提供领域知识，从而使演化过程从被动突变转为主动假设驱动；

**🔧 技术方法**

使用大语言模型（GPT‑4.1）进行代码生成，结合PyVRP开源实现对HGS组件的插拔；采用进化算法进行多代演化，评估采用成本函数和多目标性能；

**📊 数据集**

在TSP‑100基准上演化父选择器，并在PyVRP提供的六类VRP（CVRP、GVRP、MDVRPTW、PCVRPTW、VRPB、VRPTW）各60个实例上进行泛化测试；

**📈 对比分析**

对比方法包括PyVRP原始实现、单独演化组件以及完整集成的MEP演化求解器；结果显示，MEP演化的父选择器平均提升约1.8%成本，集成求解器在所有六种VRP上平均降低成本2.2%，并在某些变体上运行时间下降45%+；

**⚠️ 局限性**

局限性主要在于演化成本依赖LLM调用，且单独演化各组件未充分考虑跨组件协同；此外，当前演化仅在相对简洁的HGS框架内测试，未验证在更大规模或不同启发式框架下的可迁移性。

---

## 272. Ensembles at Any Cost? Accuracy-Energy Trade-offs in Recommender Systems

**arXiv ID:** 2604.07869 | [PDF](https://arxiv.org/pdf/2604.07869v1)

**作者:** Jannik Nitschke `[一作]` (University of Siegen), Joeran Beel `[通讯]` (University of Siegen)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了推荐系统中集成方法的准确率与能源消耗权衡，开展了93个实验；

**💡 创新点**

量化了多种集成策略在不同数据集与任务下的能源效率，提出选择性集成（Top Performers）是更节能的实践；

**🔧 技术方法**

使用Surprise与LensKit框架、EMERS硬件能耗测量，结合经典协同过滤与KNN模型；

**📊 数据集**

MovieLens‑100K、MovieLens‑1M、ModCloth、Anime；

**📈 对比分析**

对比最佳单模型与四种集成策略，集成准确率提升0.3%–5.7%，能源开销19%–2550%，其中Top Performers相对平均更节能；

**⚠️ 局限性**

仅评估经典算法，未覆盖深度学习模型；未将训练与推理能耗分离；实验仅在离线评估环境；内存限制导致部分集成失效。

---

## 273. Hidden Biases in Conditioning Autoregressive Models

**arXiv ID:** 2604.07855 | [PDF](https://arxiv.org/pdf/2604.07855v1)

**作者:** Francois Pachet `[一作]` (Sorbonne Universite), Pierre Roy `[通讯]` (Soundtrap)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文证明了在典型的可简洁表示的自回归模型（如大型语言或音乐模型）中，针对全局约束（如韵律、节拍、终止词、位置约束等）的精确 MAP 解码和精确条件采样是计算上不可行的；具体而言，精确句子级 MAP 归约自 SAT，NP‑难；在单元约束和度量约束下同样 NP‑难；而对固定长度终止事件的精确归一化则归约自 #SAT，#P‑难。

**💡 创新点**

创新点在于首次将复杂性理论与自回归生成任务结合，给出了正式的 NP‑难和 #P‑难证明，阐明了当前大多数基于启发式搜索或重排序的约束生成方法背后潜在的“推断偏差”。

**🔧 技术方法**

采用了经典的多项式时间归约（SAT→MAP，#SAT→归一化），利用自回归模型对前缀信息的全局依赖特性，构造了可在多项式时间内评估的条件概率，进而证明了上述复杂度上界。

**📊 数据集**

本研究没有使用任何具体的数据集；其贡献完全基于理论分析。

**📈 对比分析**

由于是理论性证明，本文没有实验比较；不过作者指出在受限状态（如一阶马尔可夫模型）下，正则或单元约束可通过动态规划实现，而在一般自回归模型下则面临上述复杂度障碍。

**⚠️ 局限性**

局限性包括：仅给出了上界（难度）证明，未给出实际可行的近似算法或对复杂度的下界；实验验证缺失；适用范围仅限于可多项式时间评估前缀概率的自回归模型。

---

## 274. QaRL: Rollout-Aligned Quantization-Aware RL for Fast and Stable Training under Training--Inference Mismatch

**arXiv ID:** 2604.07853 | [PDF](https://arxiv.org/pdf/2604.07853v1)

**作者:** Hao Gu `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了QaRL管线，先用低精度量化模型进行生成，再对学习器执行量化对齐的前向推理和梯度更新，并在此基础上设计Trust Band Policy Optimization（TBPO）以抑制量化误差导致的错误标记。

**💡 创新点**

创新点在于：① 通过“rollout‑aligned”量化训练消除训练‑推理失配；② 发现并针对量化生成中长响应的错误标记提出双重剪裁与序列级目标的TBPO；③ 兼顾大规模MoE模型的速度与稳定性。

**🔧 技术方法**

使用技术包括：低位量化推理（W4A16、W8A8等）、量化感知训练（QAT）、GRPO/GSPO策略梯度、Straight‑Through Estimator、双向剪裁、序列级比例和误差权重截断、以及 Muon 优化器。

**📊 数据集**

数据集涵盖 OpenR1‑Math‑46K、AIME2024/2025、AMC、Math‑500、OlympiadBench、Minerva，以及 OOD 集合 ARC‑Challenge、GPQA‑Diamond、LiveCodeBench、MMLU Pro。

**📈 对比分析**

与全精度 BF16 RL 基线及单纯量化生成训练对比：QaRL TBPO 在所有模型规模下均接近 BF16 的奖励水平，并比量化生成训练提升约 5.5 分；在 MoE Qwen3‑30B‑A3B‑Base 上平均数学分数 51.2% 接近 BF16 的 52.1%，同时实现 1.3× 的训练速度加速。

**⚠️ 局限性**

局限性包括：仍需在训练端做低精度前向，未实现完全低精度端到端 RL；序列级目标和双重剪裁在大规模批次下计算开销较高；对不同量化格式的泛化验证有限，未来需探索更高效的 token‑级近似。

---

## 275. We Need Strong Preconditions For Using Simulations In Policy

**arXiv ID:** 2604.07838 | [PDF](https://arxiv.org/pdf/2604.07838v1)

**作者:** Steven Luo `[一作]` (University of California, Berkeley), Carlos Guirado `[通讯]` (University of California, Berkeley)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并讨论了在政策制定中使用大语言模型（LLM）代理模拟的三项前提条件，重点关注双重用途风险、验证难题与责任归属，呼吁制定“模拟开发与部署报告”以增强透明度与问责。

**💡 创新点**

创新点在于将技术伦理、社会正义与政策实践结合，首次系统性阐述了（1）不将边缘群体模拟视为中立技术输出、（2）必须让模拟对象参与设计与验证、（3）必须建立可追溯的责任链；并提出了面向监管的报告框架。

**🔧 技术方法**

主要技术包括大型语言模型（LLM）驱动的代理模拟、情景设计与参与式验证方法，以及对模拟结果的可解释性与可审计性评估工具。

**📊 数据集**

未采用单一数据集进行实验，而是依赖历史行政数据、社区自述经验以及公开的合成观众数据作为理论讨论与案例示例的背景资料。

**📈 对比分析**

文章并未进行实验性对比，而是通过案例分析与文献回顾说明在满足前提条件下模拟可以提升决策质量、降低误导风险；若不满足，则可能导致偏见强化或决策失误。

**⚠️ 局限性**

局限性包括：缺乏实证验证与量化评估；对不同政策场景的适用性尚不明晰；对边缘群体数据的可获得性与真实性存疑；需要进一步研究如何实现可操作的参与式设计与责任追溯机制。

---

## 276. SPARD: Self-Paced Curriculum for RL Alignment via Integrating Reward Dynamics and Data Utility

**arXiv ID:** 2604.07837 | [PDF](https://arxiv.org/pdf/2604.07837v1)

**作者:** Xuyang Zhi `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自适应的 RL 对齐框架 SPARD，结合进度感知的奖励权重调整和奖励归因的数据再平衡，实现动态、自主的训练课程。

**💡 创新点**

创新点：1）将学习进度作为信号动态调整多目标奖励权重；2）利用奖励-数据归因矩阵对数据样本重要性进行自适应加权；3）闭环机制同步学习意图与数据效用，突破静态加权与数据异质性瓶颈。

**🔧 技术方法**

技术：基于 GRPO 的无价值网络 RL；进度感知权重更新采用 LCB 原理与 KL 约束的在线镜像下降；奖励归因通过奖励维度的分散度计算得到数据类别贡献矩阵；温度化 Boltzmann 正则化实现数据权重更新。

**📊 数据集**

数据集：从 WildChat‑IF 子集挑选 5.4k 多样化对话提示，按 Code、Knowledge QA、Text Transformation、Creative Writing 四类标注；基准评测使用 IFEval、LiveCodeBench、GPQA‑Diamond、CreativeWritingV3、Arena‑Hard、WildBench、MT‑Bench。

**📈 对比分析**

与 SFT、DPO、GRPO_rm、GRPO_avg、GRPO_imp 等方法比较；在 Qwen2.5‑7B‑Instruct 与 Qwen3‑8B 上实验，SPARD 在所有领域均显著优于基线，提升平均性能，收敛更快、方差更小，证明了自适应课程的有效性。

**⚠️ 局限性**

局限：1）依赖高能力 LLM 作为奖励评判者，导致推理延迟与计算开销大；2）奖励聚合仍采用线性加权，未能捕捉目标间非线性交互，可能限制最优解空间。

---

## 277. ZeroCoder: Can LLMs Improve Code Generation Without Ground-Truth Supervision?

**arXiv ID:** 2604.07864 | [PDF](https://arxiv.org/pdf/2604.07864v1)

**作者:** Lishui Fan `[一作]` (Zhejiang University), Zhongxin Liu `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种无监督的共进化强化学习框架 ZeroCoder，通过代码与测试的自生成交互来产生可验证奖励，从而同时提升代码生成与测试生成的质量。

**💡 创新点**

创新点主要包括：① 在无标签环境下实现代码与测试的共进化训练；② 采用离线基于执行矩阵秩的实例过滤，剔除低信息样本；③ 引入动态贝叶斯选择器 (Dynamic B⁴) 来缓解选择器漂移，并仅需极少量标注数据进行校准。

**🔧 技术方法**

技术手段涵盖：强化学习与可验证奖励 (RLVR)、执行反馈、矩阵秩预过滤、突变测试奖励、动态贝叶斯选择器及基于执行的奖励构造。

**📊 数据集**

在六大代码生成基准（MBPP、LiveCodeBench、APPS、CodeForces 等）上进行评估，并使用 Qwen2.5-1.5B-Instruct、Qwen3-4B、Qwen2.5-Coder-7B-Instruct 三大模型进行实验。

**📈 对比分析**

与基线（无 RL、离线/在线测试驱动 RL）相比，ZeroCoder 在无标签设置下平均提升代码生成 8.6%/测试生成 58.2%；在仅 10 条标注样本的匹配小标签设置下提升至 21.6%/24.3%，接近甚至与 oracle 监督下的表现相当。

**⚠️ 局限性**

局限性包括：需要较大计算资源进行执行与突变测试，矩阵秩作为多样性代理可能不足以全面评估交互质量；目前仅在 1B–7B 规模模型上验证，缺乏对更大模型的泛化性探究。

---

## 278. Incentivising green video streaming through a 2-tier subscription model with carbon-aware rewards

**arXiv ID:** 2604.07910 | [PDF](https://arxiv.org/pdf/2604.07910v1)

**作者:** Vasilios A. Siris `[一作]` (Athens University of Economics and Business), Konstantinos Varsos `[通讯]` (Athens University of Economics and Business)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并分析了基于碳强度的激励机制，帮助视频流服务商在碳排放高峰期通过降低视频质量、提供折扣和碳奖励来实现碳排放削减；同时设计了两档订阅模式，使用户无需逐视频做出选择即可享受激励。

**💡 创新点**

①将用户的质量偏好与碳强度动态结合，建立了利用每日碳强度测量来决定视频降质时机的决策流程；②证明了在仅降低最高质量下一档（FHD）时能在不显著降低用户体验的前提下获得更高能源/碳收益；③提出了基于本地与远程 CDN 的碳感知选择条件，体现了多源策略对激励的影响。

**🔧 技术方法**

用户体验模型（MOS‑based utility）、碳排放与能耗映射模型、日常碳强度时间序列预测、动态阈值调整的激励决策算法、基于订阅的计费与奖励机制。

**📊 数据集**

每日碳强度时间序列（希腊与荷兰两国数据）、不同网络段的增量能耗测量（4G/5G、5G、6G、核心网、CDN等）、用户质量偏好参数（高质量 vs 绿色用户）以及视频码率对应的 MOS。

**📈 对比分析**

通过在实验环境下模拟一个月的碳强度走势，比较两种降质策略（4K→FHD 与 4K→HD）在满足 280 gCO₂e/kWh 上限时的比特率下降、用户效用损失与所需激励；结果表明 FHD 降质在保证更低效用损失的同时，能实现与 HD 降质相近的碳削减（约13%比特率下降），因此更具经济性。

**⚠️ 局限性**

仅考虑增量能耗，忽略设备空闲能耗和非线性流量‑能耗关系；用户行为假设为均匀分布且不考虑实时需求变化；激励模型未在真实用户群中验证，实际激励接受度及长期行为影响尚待实证研究。

---

## 279. AnomalyAgent: Agentic Industrial Anomaly Synthesis via Tool-Augmented Reinforcement Learning

**arXiv ID:** 2604.07900 | [PDF](https://arxiv.org/pdf/2604.07900v1)

**作者:** Jiaming Su `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于agent的工业缺陷合成框架AnomalyAgent，通过Prompt生成、图像生成、质量评估、知识检索、掩码生成五种工具的闭环交互，实现对缺陷图像的多轮迭代生成。

**💡 创新点**

将缺陷生成视为多步骤决策问题，构建工具驱动的思考–行动–观测循环，并在此基础上设计三维奖励（任务、反思、行为）与GRPO强化学习，显著提升生成质量与多样性。

**🔧 技术方法**

采用多模态大语言模型（Qwen3‑VL‑4B‑Thinking）配合Gemini 3.1 Flash Image Preview等生成工具，使用SFT+RL两阶段训练，并通过知识检索和掩码生成补全语义与定位。

**📊 数据集**

在MVTec‑AD和VisA工业缺陷数据集上构造多轮合成轨迹进行训练和评估，利用真实缺陷样本反向生成正样本以构建轨迹。

**📈 对比分析**

与传统零射、生成模型（CutPaste、DRAEM、AnomalyAny等）以及Gemini、GPT Image等直接生成方法对比，IS/IC‑L分别达到2.10/0.33，分类准确率57%，图像/像素级AP提升至99.3%/74.2%，显著优于SOTA。

**⚠️ 局限性**

仍受限于对高质量生成模型和外部知识库的依赖，计算成本较高，对极端稀缺或极端复杂缺陷的泛化能力尚未充分验证。

---

## 280. Visual Perceptual to Conceptual First-Order Rule Learning Networks

**arXiv ID:** 2604.07897 | [PDF](https://arxiv.org/pdf/2604.07897v1)

**作者:** Kun Gao `[一作]` (Zhongguancun Academy), Katsumi Inoue `[通讯]` (National Institute of Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种全可微分的规则学习框架 γILP，能够在无图像标签泄漏的前提下，利用图像嵌入、可微聚类和张量化替换从关系图像或纯图像中学习一阶逻辑规则，并实现谓词发明。

**💡 创新点**

创新点包括：①将符号ILP与视觉编码器和可微聚类结合，实现端到端可微的规则学习；②通过张量化替换和规则网络实现快速的逻辑推理；③利用LLM自动推断占位谓词的自然语言语义，从而完成谓词发明。

**🔧 技术方法**

使用了 Vision Transformer / VAE 编码器、可微聚类模块、张量化规则网络、LLM（Gemini 2.5 Pro、GPT‑5）进行占位谓词语义翻译，以及基于张量操作的可微替换机制。

**📊 数据集**

实验数据集包括经典 ILP 数据集（如 Predicates、Predecessor 等）、关系图像数据集（将 MNIST 图像替代常量）、纯图像 Kandinsky 模式数据集，并与 ∂ILP、DFORL 等方法对比。

**📈 对比分析**

与现有方法比较，γILP 在经典 ILP 和关系图像任务上均实现 1.0 的精确率和召回率；在 Kandinsky 分类任务中，γILP 在大多数任务上超过传统 CNN/ViT 模型；LLM 在占位谓词语义推断上表现出高度一致性。

**⚠️ 局限性**

局限性包括：①规则长度有限，难以学习需要超过 4~6 个变量的规则；②对聚类质量和编码器性能敏感；③需要大量 GPU 计算，扩展到大规模数据集的可扩展性仍待验证。

---

## 281. DialBGM: A Benchmark for Background Music Recommendation from Everyday Multi-Turn Dialogues

**arXiv ID:** 2604.07895 | [PDF](https://arxiv.org/pdf/2604.07895v1)

**作者:** Joonhyeok Shin `[一作]` (Sungkyunkwan University), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了DialBGM数据集，用于多轮对话情境下的背景音乐推荐任务。

**💡 创新点**

首次把多轮对话与背景音乐匹配作为任务，并引入人类主观排序作为评测标准。

**🔧 技术方法**

结合LLM生成摘要、规则过滤、嵌入检索以及多模态LLM进行候选音乐排序。

**📊 数据集**

利用DailyDialog的对话文本与MusicCaps的音乐音频及其字幕构成数据。

**📈 对比分析**

使用Hit@1、MRR、nDCG、Kendall’s τ_b等指标评估模型，结果显示最优模型仅约35%的Hit@1，整体性能远低于人类。

**⚠️ 局限性**

数据量有限、音乐来源单一、评判主观性高且模型对情感推理能力不足。

---

## 282. Data Selection for Multi-turn Dialogue Instruction Tuning

**arXiv ID:** 2604.07892 | [PDF](https://arxiv.org/pdf/2604.07892v1)

**作者:** Bo Li `[一作]` (Peking University), Wei Ye `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多轮对话数据选择框架 MDS，通过全局语义覆盖与局部结构评估来筛选高质量对话。

**💡 创新点**

创新点是将对话级别的语义覆盖与实体对齐、新颖度评分以及问答形式一致性两阶段筛选结合，既保证多轮覆盖又提升结构可靠性。

**🔧 技术方法**

技术包括句子编码（Sentence‑Transformers）、K‑means分箱、贪心覆盖选择、实体对齐与新颖度评分、问答形式一致性评分、局部阈值过滤，以及基于 LLM 的无参考评估。

**📊 数据集**

使用 Baize 通用对话语料库、Banking 客服对话语料，以及 MT‑Eval、ConsistentChat、TopDial 等公开基准和 Banking 测试集。

**📈 对比分析**

与单轮选择、对话级 LLM 评分、启发式、随机等基线比较，MDS 在大多数参考无/有指标上均取得最高或第二高排名，显著提升多轮一致性、实体覆盖，并在长对话上更稳健。

**⚠️ 局限性**

局限性：对矛盾错误的抑制效果有限，因保留较长、结构丰富的对话仍可能出现隐性不一致；阈值设置需要手动调参。

---

## 283. An Agentic Evaluation Architecture for Historical Bias Detection in Educational Textbooks

**arXiv ID:** 2604.07883 | [PDF](https://arxiv.org/pdf/2604.07883v1)

**作者:** Gabriel Stefan `[一作]` (University of Bucharest), Adrian-Marius Dumitran `[通讯]` (University of Bucharest)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套agentic评估体系，对罗马尼亚中学历史教材中的隐性偏见进行多阶段筛选、评估与汇总，提供可审计的报告；

**💡 创新点**

创新点包括：① Source Attribution Protocol将教材叙事与引证源区分，显著降低误报；② 多代理推理架构实现深度审议和自上而下的决策，提升了判定的准确性和可解释性；

**🔧 技术方法**

技术实现基于多模态LLM（Llama‑4‑Maverick）做粗筛，五款异构评估模型（Mixtral、GPT‑OSS、DeepSeek、Cogito、Kimi‑K2）做独立评估，元代理（GPT‑5.2）进行结果合成与人机升级，全部输出结构化JSON并生成HTML报告；

**📊 数据集**

使用公开的罗马尼亚高等中学历史教材数据集（由教育部官方数字仓库提供的多家出版社教材），共计若干册；

**📈 对比分析**

与单模型零射击基线对比：agentic管线的平均严重程度从5.4/7降至2.9/7，83.3%的候选片段被评为可接受；人类评估中，独立推理配置被18名评审以64.8%（35/54）的比例认为报告最准确、最具教育意义；成本约$2/本；

**⚠️ 局限性**

局限性包括：① 屏蔽阶段单模型导致漏检风险；② 与零射击基线的比较混入了窗口大小、提示校准等变量，未完全隔离；③ 缺乏专家标注的真实标签用于精确评估召回与精度；④ 对跨国教材和多语言场景的通用性尚待验证。

---

## 284. ReconPhys: Reconstruct Appearance and Physical Attributes from Single Video

**arXiv ID:** 2604.07882 | [PDF](https://arxiv.org/pdf/2604.07882v1)

**作者:** Boyuan Wang `[一作]`, Xingang Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了ReconPhys框架，利用单目视频通过双分支网络同时预测3D高斯剖面（3DGS）和弹簧-质量系统的物理属性，实现快速（<1s）生成可直接用于仿真的物理属性绑定的3D资产。

**💡 创新点**

创新点在于：①将物理属性预测与3DGS重建融合为单向前馈网络，消除逐场景优化；②通过自监督的可微物理-渲染循环，利用像素级重建误差直接优化物理参数；③采用稀疏弹簧-质量系统与3DGS的双阶段绑定，既保证物理真实性又保持渲染效率。

**🔧 技术方法**

技术包括：3D Gaussian Splatting、可微弹簧-质量动力学、InternViT+ResNet自注意力的时空特征提取、IDW插值绑定机制、Self‑Forcing训练与截断反向传播。

**📊 数据集**

使用从Objaverse-XL中筛选出的约500个可变形物体，通过自动化管道合成大量带有连续物理参数的虚拟自由下落视频（30帧@512×512）。

**📈 对比分析**

与4DGS和Spring‑Gaus等基线对比，ReconPhys在重建任务上PSNR最高33.84dB、SSIM 0.953、LPIPS 0.0366、Chamfer Distance 0.001；在未来预测任务上PSNR 21.64dB、SSIM 0.907、LPIPS 0.0876、CD 0.004，显著超越基线，并将推理时间从>1h降至<1s。

**⚠️ 局限性**

局限包括：目前仅支持单视角自由下落场景，对更复杂交互或多视角数据的泛化未评估；弹簧-质量模型对材料行为的表达仍有限，无法覆盖更复杂的非线性或粘弹性材料；以及缺乏真实物理标签的验证，仍需进一步实验验证在真实世界中的物理一致性。

---

## 285. FlowGuard: Towards Lightweight In-Generation Safety Detection for Diffusion Models via Linear Latent Decoding

**arXiv ID:** 2604.07879 | [PDF](https://arxiv.org/pdf/2604.07879v1)

**作者:** Jinghan Yang `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FlowGuard，一种跨模型的在生成过程中的 NSFW 内容检测框架；

**💡 创新点**

创新点在于：① 用线性 VAE 解码器快速把潜在状态投影到图像空间；② 采用傅里叶低通滤波去除噪声；③ 通过课程学习逐步训练模型，从干净图像到噪声强的中间状态；③ 构建跨模型的生成轨迹数据集；

**🔧 技术方法**

使用了线性 VAE 解码、傅里叶低通滤波、课程学习（Curriculum Learning）、ViT-B/16 视觉网络、低频滤波等技术；

**📊 数据集**

使用自建的 FlowGuard 数据集，包含多种主流文本到图像模型（Stable Diffusion v1.5、v3、Flux、PixArt、Qwen-Image、Zimage 等）的 50 步潜在轨迹、线性解码图像及 NSFW 标注；

**📈 对比分析**

与后置检测器（Falconsai）、通用视觉语言模型（Qwen3‑VL‑8B‑Instruct）以及 LlavaGuard‑7B 进行对比，结果表明 FlowGuard 在 ID 与 OOD 场景下 F1 分数提升 30% 以上，同时推理时间缩短、显存占用降低 97% 以上；

**⚠️ 局限性**

局限性包括：① 对课程学习的顺序和难度设置敏感；② NSFW 标注依赖最终高质量图像，可能存在标注不一致；③ 只在 128×128 低分辨率下训练，可能在更高分辨率或更复杂场景下性能下降。

---

## 286. Networking-Aware Energy Efficiency in Agentic AI Inference: A Survey

**arXiv ID:** 2604.07857 | [PDF](https://arxiv.org/pdf/2604.07857v1)

**作者:** Xiaojing Chen `[一作]` (Shanghai University), Shugong Xu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了Agentic AI在无线边缘网络中的能耗瓶颈，提出了能耗核算框架和统一的优化技术分类，并系统评估了跨层协同、模型压缩、计算控制等方法。

**💡 创新点**

首次将Agentic AI能耗拆解为计算与通信两大维度，构建了覆盖模型简化、计算调控、输入/注意力优化和硬件感知推理的四柱分类，并提出跨层协同与语义通信的设计范式。

**🔧 技术方法**

采用模型量化、剪枝、蒸馏、稀疏Mixture‑of‑Experts、动态长度控制、早停、层跳过、稀疏注意力、KV缓存复用、混合精度调度、DVFS、内存/ I/O 优化，以及基于通信‑推理协同的语义通信与检索增强通信等技术。

**📊 数据集**

本综述主要基于已有文献的实验结果（如GPT‑J‑6B、LLaMA‑7B、OPT‑175B等模型）与公开数据集（如GSM8K、MMLU、Text‑to‑Image等），并未自行实验。

**📈 对比分析**

通过对比已有工作与本文提出的技术体系，指出量化/剪枝在能耗可达70%‑80%时仍保持95%‑99%准确率；早停/层跳过可实现约30%‑50%计算节省；语义通信能将传输能耗降低30%‑40%；整体多层协同可望实现数倍能效提升，具体数值因场景与实现而异。

**⚠️ 局限性**

主要局限包括：1）缺乏统一量化指标与基准，难以跨工作直接对比；2）多技术组合的最佳权衡尚未系统化；3）对网络条件与设备异构的自适应策略尚处于理论与实验验证阶段；4）能耗评估多基于模型推测或单机实验，缺乏大规模真实网络验证。

---

## 287. ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning

**arXiv ID:** 2604.07851 | [PDF](https://arxiv.org/pdf/2604.07851v1)

**作者:** Jiani Huang `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于强化学习的微调框架，专门提升LLM在复杂查询式推荐任务中的多步推理能力；

**💡 创新点**

核心创新点包括：① 双图增强奖励塑形（融合NDCG、查询一致性得分与偏好一致性得分）以生成细粒度奖励；② 关注推理过程的优势估计（将奖励分配到每一步推理段落，惩罚错误推理）；③ 在线课程调度器根据模型表现动态调整训练难度，提升收敛稳定性；

**🔧 技术方法**

采用强化学习微调（RFT）结合奖励模型、优势估计、剪切策略、在线课程学习等技术；

**📊 数据集**

在RecBench+基准数据集上进行实验，覆盖电影和图书两个领域，按推理难度分为五类子任务；

**📈 对比分析**

与多类基线（LLM骨干、对话式推荐系统、已有RFT模型）对比，实验显示该框架在所有任务类型均显著提升准确率（在电影领域最高可提升13.2%，在复杂任务上提升440%），并保持指令跟随和知识保留能力；

**⚠️ 局限性**

局限性在于仅处理单轮查询，未覆盖多轮对话场景，缺乏对话上下文维护与动态奖励调整的支持。

---

## 288. A Hardware-Anchored Privacy Middleware for PII Sharing Across Heterogeneous Embedded Consumer Devices

**arXiv ID:** 2604.07839 | [PDF](https://arxiv.org/pdf/2604.07839v1)

**作者:** Aditya Sabbineni `[一作]` (Independent Researcher), Willison Lopes `[通讯]` (IEEE)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一个名为UDSS的硬件锚定隐私中间件，支持在不同消费者电子设备间安全、最小化的PII共享，并显著降低用户上机时延。

**💡 创新点**

创新点在于将ARM TrustZone/OP-TEE与Contextual Scope Enforcement相结合，实现了协议级别的字段最小化、情境感知访问控制和硬件可信审计。

**🔧 技术方法**

核心技术包括ARM TrustZone/OP-TEE可信应用、OP-TEE安全隧道、AES-256-GCM+RS256加密签名、分层访问表与硬件可信证书。

**📊 数据集**

实验使用Raspberry Pi 4模拟的Smart TV中间件，利用Yocto Linux + OP-TEE，并在30次实验中收集上机时延和CPU占用等指标。

**📈 对比分析**

与传统手工输入和OAuth设备流对比，UDSS在用户上机时延上降低约65%（6.3 s vs 18.4 s），CPU占用低于2%，且在PII泄露方面将字段暴露从4.8降至1.0，实现约79%的减少。

**⚠️ 局限性**

局限性包括仅在单节点原型验证、对多用户家庭身份隔离未实现、对TEE侧时序侧信道的防护仍需加强以及对去中心化manifest部署的支持不足。

---

## 289. Harnessing Embodied Agents: Runtime Governance for Policy-Constrained Execution

**arXiv ID:** 2604.07833 | [PDF](https://arxiv.org/pdf/2604.07833v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个运行时治理框架，用于在执行受政策约束的环境中管理具身代理的执行。

**💡 创新点**

创新点在于将代理认知与执行监督分离，构建了能力接入、策略守护、执行监视、恢复、人工覆写和审计六个模块的治理管道，并在模拟中验证其效果。

**🔧 技术方法**

技术包括大规模语言模型（LLM）驱动的任务规划、可组合的能力包（ECM）、ROS2/仿真平台、运行时监视器与策略引擎。

**📊 数据集**

使用Gazebo Fortress仿真环境下的UR5e移动机器人以及四种环境配置（Sim-Relaxed、Real-Restricted、Human-Shared、Test-Audit）进行随机任务生成。

**📈 对比分析**

与直接执行、静态规则、能力内部安全三种基线对比，未经授权动作拦截率96.2%，运行时违规检测率61.3%，恢复成功率91.4%，显著优于基线（p<0.001）。

**⚠️ 局限性**

局限在于仅在仿真中验证，真实机器人上的延迟与硬件不确定性未评估；治理层依赖精确策略定义，误差可能导致误拦或误放；人机交互评估未包含真实人类决策。

---

## 290. ImVideoEdit: Image-learning Video Editing via 2D Spatial Difference Attention Blocks

**arXiv ID:** 2604.07958 | [PDF](https://arxiv.org/pdf/2604.07958v1)

**作者:** Jiayang Xu `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种完全基于图像对训练的在线视频编辑框架ImVideoEdit，能够在保持预训练视频模型时间先验的前提下，实现精细且语义驱动的空间内容编辑。

**💡 创新点**

核心创新包括（1）Predict‑Update Spatial Difference Attention模块，通过先粗略对齐再细化残差的两步方式高效提取空间差异；（2）Text‑Guided Dynamic Semantic Gating，使编辑强度与文本语义自适应调节；（3）冻结3D DiT自注意力，保证时间一致性，显著降低训练成本。

**🔧 技术方法**

技术手段主要有：流匹配（Flow Matching）框架、3D Diffusion Transformer（Wan‑T2V‑1.3B）冻结、2D空间自注意力并行到每个Transformer块、跨模态文本编码与门控机制、基于VLM的评价与数据过滤。

**📊 数据集**

使用自己构建的约13k对图像数据集（source‑edit pairs），通过场景条件提示、文本生成、图像合成与VLM+人工筛选过滤得到，涵盖多种场景与编辑任务。

**📈 对比分析**

在VLM‑评估（IA/TC/VF/AA）与VBench指标上与VACE、OmniVideo2、Kiwi‑Edit、Lucy‑Edit‑Dev、DITTO、ICVE等基线对比，ImVideoEdit在1.3B参数规模下获得65.24的总分，显著高于同规模基线，且与5B模型相近，体现出极佳的性能与效率。

**⚠️ 局限性**

局限性包括：仅在单帧图像对上训练，可能对长时序复杂动态（如大幅运动、跨帧几何变形）支持有限；冻结的3D自注意力限制了模型对时间先验的进一步细化；对极端文本模糊或多义提示的鲁棒性尚待验证。

---

## 291. Large Language Model Post-Training: A Unified View of Off-Policy and On-Policy Learning

**arXiv ID:** 2604.07941 | [PDF](https://arxiv.org/pdf/2604.07941v1)

**作者:** Shiwan Zhao `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本工作系统性综述了LLM后训练方法，提出了以轨迹来源（离线vs在线）为首要组织轴，并将行为改进拆分为支持扩展、策略重塑与行为整合三类功能角色；

**💡 创新点**

创新点在于将后训练视为模型行为的结构化干预，统一了离线与在线方法的概念框架，并通过功能角色解释各类技术的核心贡献；

**🔧 技术方法**

利用语言模型作为策略/轨迹视角，定义有效支持、支持扩展、策略重塑等概念，随后对SFT、偏好优化、RLHF、过程监督、蒸馏等方法进行系统归类和对照；

**📊 数据集**

综述主要引用公开的指令数据集、偏好对照、奖励模型数据、验证器输出、搜索/检索生成的轨迹等实验集，未自行收集数据；

**📈 对比分析**

在框架下对比不同方法的优势与瓶颈，指出离线方法易于扩展支持，在线方法擅长修正自生状态，但跨阶段保留仍不足；多阶段组合可实现更强能力，但需注意迁移与蒸馏中的性能衰减；

**⚠️ 局限性**

局限在于缺乏对有效支持与支持衰减的定量度量，框架主要为解释工具，难以直接指导算法细节；跨阶段迁移中仍面临支持流失和评估不一致的问题。

---

## 292. Shortcut Learning in Glomerular AI: Adversarial Penalties Hurt, Entropy Helps

**arXiv ID:** 2604.07936 | [PDF](https://arxiv.org/pdf/2604.07936v1)

**作者:** Mohammad Daouk `[一作]`, Chandra Mohan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究多中心、多染色的狼疮性肾炎肾小球病变分类中染色是否构成快捷方式，并提出无标签的熵正则化方法以消除染色偏差。

**💡 创新点**

首次系统评估染色快捷方式；提出基于 Bayesian 双头结构的标签无监督熵正则化；展示不需要染色标签即可保持性能与校准的可行性。

**🔧 技术方法**

使用 Bayesian CNN/ViT 通过 MC dropout 估计不确定性；双头网络（病变头+染色头）配合损失加权；逆交叉熵/熵最大化正则化实现标签无监督染色抑制。

**📊 数据集**

采用 9,674 张 224×224 像素肾小球补丁，来自 365 张 WSIs，三中心（科隆、斯坦福、芝加哥），四种染色（PAS、H&E、Jones、Trichrome），标签为增殖性/非增殖性。

**📈 对比分析**

通过三组实验比较：单头染色分类精度≈100%；双头监督染色权重调节对病变性能无显著影响但负权重导致不确定性飙升；熵正则化将染色性能压至随机水平，病变指标与基线相同且不确定性保持低。

**⚠️ 局限性**

局限性包括仅在多中心多染色数据上验证，可能不适用于单染色或其他组织；熵正则化需手动选择 μ₂ 范围；模型在染色信息极度缺失时的鲁棒性尚未充分评估。

---

## 293. Unified Supervision for Walmarts Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling

**arXiv ID:** 2604.07930 | [PDF](https://arxiv.org/pdf/2604.07930v1)

**作者:** Shasvat Desai `[一作]`, Kuang-chih Lee `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

**🎯 论文内容**

提供了 ACM 文章模板的使用方法与说明，帮助作者准备符合 ACM 标准的稿件。

**💡 创新点**

将多种 ACM 与 SIG 专门模板的特点统一到单一模板，简化了投稿与审稿流程。

**🔧 技术方法**

采用 LaTeX 文档类 "acmart" 及其各类宏包（如 booktabs、graphicx 等）实现排版与格式控制。

**📊 数据集**

无（该文档仅为模板使用说明，未涉及科研数据集）。

**📈 对比分析**

不涉及实验或性能比较，主要关注模板规范与排版细节。

**⚠️ 局限性**

仅适用于 ACM 期刊/会议稿件，修改模板不被允许，缺乏科研内容与实验验证。

---

## 294. Same Outcomes, Different Journeys: A Trace-Level Framework for Comparing Human and GUI-Agent Behavior in Production Search Systems

**arXiv ID:** 2604.07929 | [PDF](https://arxiv.org/pdf/2604.07929v1)

**作者:** Maria Movin `[一作]` (Spotify), Panagiotis Papapetrou `[通讯]` (Stockholm University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Spotify音频流媒体系统中，构建并验证了一套用于比较人类与GUI代理在多跳信息检索任务中轨迹级行为的评估框架。

**💡 创新点**

创新点在于首次从任务结果、查询表达和导航路径三个维度进行细粒度对比，并在真实生产环境中验证代理与人类行为的相似与差异。

**🔧 技术方法**

采用了大语言模型驱动的GUI代理（GPT‑5 + AppAgent），配合Android模拟器进行交互；评估使用日志解析、文本相似度（Levenshtein、TF‑IDF）和导航图Jaccard相似度等技术。

**📊 数据集**

数据集包括39名真实Spotify用户在10个精心设计的多跳检索任务中的交互日志（约150条记录）以及5名模拟用户在同一任务下的50条代理交互日志。

**📈 对比分析**

对比方法：先测量任务成功率和努力（动作数、耗时），再比较首个查询相似度和完整查询覆盖度，最后用Jaccard系数量化导航图重叠。结果显示：代理成功率与人类相近（56% vs 53%），动作数更少但耗时更长；查询相似度与人类持平（≈0.58）；导航路径重叠在高频边缘显著，低频细节上代理更偏向搜索中心化。

**⚠️ 局限性**

局限性包括：仅在单一音乐流媒体应用和单一代理配置下验证；数据量相对有限，且生产环境的动态变化可能导致实验不可复现；未覆盖多种接口风格与代理架构。

---

## 295. Stitch4D: Sparse Multi-Location 4D Urban Reconstruction via Spatio-Temporal Interpolation

**arXiv ID:** 2604.07923 | [PDF](https://arxiv.org/pdf/2604.07923v1)

**作者:** Hina Kogure `[一作]` (Keio University), Komei Sugiura `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在多地点稀疏观察下的动态城市场景4D重建，提出统一框架 Stitch4D 通过合成桥接视角并联合优化实现空间覆盖和时空一致性。

**💡 创新点**

创新点在于利用多视角桥接模块 (MVBM) 生成几何一致的中间视图以及多视频联合优化模块 (MVJOM) 在共享坐标系下同步优化真实与合成观察，显著提升稀疏布局下的几何稳定性和时间连续性。

**🔧 技术方法**

技术采用基于 SpacetimeGS 的时间可变高斯原语表示，配合深度估计、视角合成网络与跨位置一致性损失实现。

**📊 数据集**

使用 CARLA 模拟的 U‑S4D 基准，提供同步全景视频与摄像机位姿，用于评估稀疏多地点重建。

**📈 对比分析**

与 4DGS、SpacetimeGS、FreeTimeGS 等基线相比，Stitch4D 在 PSNR、SSIM、LPIPS 上均优于 20%+，在轨迹插值与见视点条件下均表现更平滑、准确的重建结果。

**⚠️ 局限性**

局限性包括仅在仿真场景验证，未涉及真实世界噪声与动态摄像机；对大规模场景的可扩展性与计算成本仍待评估。

---

## 296. The Sustainability Gap in Robotics: A Large-Scale Survey of Sustainability Awareness in 50,000 Research Articles

**arXiv ID:** 2604.07921 | [PDF](https://arxiv.org/pdf/2604.07921v1)

**作者:** Antun Skuric `[一作]` (Hugging Face), Thomas Wolf `[通讯]` (Hugging Face)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2015–2026年arXiv机器人学论文约5万篇进行大规模定量分析，评估其对社会、生态和可持续性影响的意识以及与联合国可持续发展目标（SDG）的直接或间接关联。

**💡 创新点**

创新点在于：①首次使用零样本大语言模型（DeepSeek‑V3）在完整论文文本上自动提取可持续性影响与SDG关联；②系统性揭示“意识缺口”与“动机缺口”——影响提及率低于6%，真正以可持续性为动机的论文不到5%；③构建公开数据集与分析工具，便于同行复现与进一步研究。

**🔧 技术方法**

采用的技术：大语言模型零样本分类（DeepSeek‑V3），结合结构化提示、全量文本提取、文本证据和推理输出；计算平台使用云GPU；实现了大约13 Wh/次、650 kWh/总消耗的能源预算。

**📊 数据集**

数据集：约50,000篇arXiv cs.RO类别论文（2015–2026），包含每篇论文的完整PDF文本、已分类标签与模型推理；数据已公开发布于Hugging Face。

**📈 对比分析**

比较方法：对比不同年份、不同SDG主题的提及比例与动机比例，绘制时间序列和分布图；与行业、学术机构（IFR、IEEE RAS）提出的可持续性框架进行对照。性能上显示：虽然模型在单篇论文的精确度有限，但在统计水平上能稳健捕捉趋势，且四种LLM的结果保持一致，证明方法鲁棒。

**⚠️ 局限性**

限制：①LLM缺乏可持续性专业知识，易产生误判；②只评估科研叙事与动机，未反映真实环境影响；③零样本分类准确率受限，可能导致误识别；④仅覆盖arXiv公开论文，未包括IEEE Xplore等闭源数据库，导致整体代表性有限。

---

## 297. How Far Are Large Multimodal Models from Human-Level Spatial Action? A Benchmark for Goal-Oriented Embodied Navigation in Urban Airspace

**arXiv ID:** 2604.07973 | [PDF](https://arxiv.org/pdf/2604.07973v1)

**作者:** Baining Zhao `[一作]` (Tsinghua University), Xinlei Chen `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了包含5037条目标导向的城市空中导航轨迹的数据集，并在此基准上评估了17种不同的多模态模型与传统基线的空间动作表现。

**💡 创新点**

提出了目标导向的城市空中导航基准（强调垂直动作与丰富语义），引入关键决策分岔（CDB）现象，并从CDB视角系统分析LMM在空间行动中的四大局限，进而提出四条改进方向。

**🔧 技术方法**

采用UE+AirSim仿真器收集数据；使用多模态语言模型（非推理、推理、Agent、VLA）进行动作生成；通过视觉+深度输入、跨视图整合、空间想象（行动后果预测）和稀疏记忆等技术提升模型表现；对关键决策点进行曲线分析和实验验证。

**📊 数据集**

自建的城市空中目标导向导航数据集：5037条轨迹，平均长度203.4米，水平/垂直/旋转动作比例约45%/28%/27%，包含丰富的城市语义标注。

**📈 对比分析**

使用 Success Rate（SR）、Success Weighted by Path Length（SPL）和 Distance to Goal（DTG）三指标进行评估；与人类、随机、动作采样等基线以及不同类别的多模态模型对比；实验结果显示，最佳LMM（如GPT‑5）在短距离SR最高达34%，远低于人类的92%，推理型模型在中长距离上略有优势；CDB分析揭示错误非线性积累。

**⚠️ 局限性**

LMM在几何感知、跨视图理解、空间想象（对动作后果的直觉理解）以及长期记忆与规划方面存在显著不足；在关键决策点易偏离目标，导致导航失败；整体空间行动能力仍距人类水平差距巨大。

---

## 298. Karma Mechanisms for Decentralised, Cooperative Multi Agent Path Finding

**arXiv ID:** 2604.07970 | [PDF](https://arxiv.org/pdf/2604.07970v1)

**作者:** Kevin Riehl `[一作]` (ETH Zürich), Michail A. Makridis `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种基于 Karma 的去中心化多机器人路径规划框架，用双边协商和人工信用余额调节冲突解决，旨在平衡代理的重新规划负担并提升系统公平性。

**💡 创新点**

创新点在于将 Karma 信用作为内在反馈信号，形成分布式积分控制器，能够动态调节代理在冲突中的责任分配，实现长期间接互惠与公平分配。

**🔧 技术方法**

采用双边协商机制、Karma 信用平衡更新、A* 路径搜索、仿真中的方向感知运动学约束，以及离散时间离散空间的网格模型来实现冲突检测与重新规划。

**📊 数据集**

使用模拟的机器人仓库环境（5×5、10×10、15×15 网格）并随机生成任务，未使用公开真实数据集。

**📈 对比分析**

通过与 token‑passing、egoistic 及 altruistic 协商策略对比实验，Karma 方法在平均任务/服务时间与总体效率上与现有策略相当，同时显著降低任务完成时间的方差，体现更公平的负载分配。

**⚠️ 局限性**

局限性包括仅评估任务/服务时间公平性；未分析 Karma 机制的稳定性、收敛性与性能上界；参数 τ 的选择依赖具体设置；通信开销与不同信用支付规则的影响仍需进一步研究。

---

## 299. Are we still able to recognize pearls? Machine-driven peer review and the risk to creativity: An explainable RAG-XAI detection framework with markers extraction

**arXiv ID:** 2604.07964 | [PDF](https://arxiv.org/pdf/2604.07964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 300. Kathleen: Oscillator-Based Byte-Level Text Classification Without Tokenization or Attention

**arXiv ID:** 2604.07969 | [PDF](https://arxiv.org/pdf/2604.07969v1)

**作者:** George Fountzoulas `[一作]` `[通讯]` (Frederick University), George Fountzoulas (Frederick University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种仅用733K参数、无分词器、无注意力机制的字节级文本分类模型，能够直接对原始UTF‑8字节进行频域处理。

**💡 创新点**

创新点在于：① 引入阻尼正弦卷积（RecurrentOscillatorBank）实现O(L)时间复杂度；② 使用单个可学习向量通过FFT-Rotate波表编码映射所有256字节；③ 仅用6个可学习相位的PhaseHarmonics非线性成为最重要的增益；④ 整体模型极简、无Tokenizer、无Attention，展示频域处理可超越传统Token化模型。

**🔧 技术方法**

采用的技术包括阻尼正弦卷积、FFT相位旋转编码、连续相位偏移、PowerLawGate、DualPooling、线性投影及多尺度频域特征融合。

**📊 数据集**

使用的公开数据集为 IMDB（情感分析）、AG News（主题分类）和 SST‑2（情感分析）。

**📈 对比分析**

通过与Tokenized Kathleen（11.8M参数）和预训练BERT等基线进行对比；Kathleen在 IMDB 88.6%、AG News 92.3%、SST‑2 83.3% 的准确率，分别比Tokenized版本高 +1.6/+2.1，且参数量低16倍；与BERT相比仍差约8%。模型的O(L)复杂度使其能处理超长序列（如100K+字节），Transformer在该范围内无法运行。

**⚠️ 局限性**

局限性包括：与大规模预训练模型相比仍有性能差距；短文本性能受限于信号长度；仅验证了分类任务，未测试生成或翻译等序列到序列任务；多语言评估与边缘设备部署尚待进一步验证。

---

## 301. AtomEval: Atomic Evaluation of Adversarial Claims in Fact Verification

**arXiv ID:** 2604.07967 | [PDF](https://arxiv.org/pdf/2604.07967v1)

**作者:** Hongyi Cen `[一作]` (Zhejiang University), Yingcai Wu `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AtomEval，一种面向事实验证的对抗性命题重写有效性评估框架；

**💡 创新点**

创新点在于将命题拆解为SROM原子，并通过硬门控与软降解度量实现对事实一致性的细粒度评估；

**🔧 技术方法**

采用SROM原子提取、硬结构一致性检测与核心事实失真、事实冲突、主题漂移、证据泄露等软指标相结合的方式；

**📊 数据集**

使用FEVER基准数据集，并在其验证集上采样被拒绝的命题进行实验；

**📈 对比分析**

与传统攻击成功率、SBERT相似度和困惑度等指标对比，AtomEval得到的有效攻击率（VASR）明显低于传统评估，揭示了大部分攻击在事实层面并未保持原命题；

**⚠️ 局限性**

局限性包括仅在英文FEVER上验证、对原子提取器的依赖以及难以捕捉多跳推理或隐含矛盾等复杂推理错误。

---

## 302. Lighting-grounded Video Generation with Renderer-based Agent Reasoning

**arXiv ID:** 2604.07966 | [PDF](https://arxiv.org/pdf/2604.07966v1)

**作者:** Ziqi Cai `[一作]` (Peking University), Boxin Shi `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

LiVER提出了一种基于扩散模型的可控视频生成框架，能够通过3D场景代理（包含光照、布局和相机轨迹）实现高保真、光照真实的影视级视频合成。

**💡 创新点**

创新点包括：①利用渲染器生成包含漫射、粗糙、光滑GGX通道的光照代理，直接将物理光照信息作为条件；②设计轻量化的代理编码器和适配器，将代理特征注入视频潜空间；③提出三阶段训练策略（先冻结主体、后LoRA微调、再混合真实与合成数据），提升训练稳定性和光照多样性；④构建大型光照驱动的视频数据集LiVERSet，包含真实与合成子集，提供精细光照、相机和布局标注。

**🔧 技术方法**

技术栈包括：扩散模型Wan2.2-5B + VAE、LoRA微调、轻量化2D代理编码器、适配器、Blender物理渲染、HDR环境图、DiffusionLight-Turbo、VGGT、Grounding-DINO、SAM 2、Qwen 2.5-VL、CLIP、GMP等。

**📊 数据集**

使用的数据集为LiVERSet，包含约11K段视频（81帧，720×1280），分为LiVER-Real（真实世界光照）和LiVER-Syn（合成光照、HDR环境图、动态照明）。

**📈 对比分析**

与CameraCtrl、MotionCtrl、VideoFrom3D等基线对比，LiVER在FVD、FID、CLIP、相机姿态误差、光照误差、mIoU等指标上均取得最高分；在用户评测中在视频质量、场景控制、相机控制、光照控制四项指标中被选中比例达80%+，显示显著优于竞争方法。

**⚠️ 局限性**

局限性：初始3D几何重建粗糙，细节依赖文本提示，导致对高精度几何和材质的生成质量敏感；未来需改进代理的几何解析和提示工程。

---

## 303. ParkSense: Where Should a Delivery Driver Park? Leveraging Idle AV Compute and Vision-Language Models

**arXiv ID:** 2604.07912 | [PDF](https://arxiv.org/pdf/2604.07912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 304. Rethinking Residual Errors in Compensation-based LLM Quantization

**arXiv ID:** 2604.07955 | [PDF](https://arxiv.org/pdf/2604.07955v1)

**作者:** Shuaiting Li `[一作]` (Zhejiang University), Kejie Huang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新定义 GPTQ 与 GPTAQ 的残差误差，将跨层输出误差与权重量补偿误差统一考虑，并引入 compensation‑aware error（补偿感知误差）来实现更精准的量化校准。

**💡 创新点**

创新点包括：① 将校准目标改为始终与原始全精度输出对齐；② 在残差误差中加入权重量补偿产生的内在误差；③ 利用 neuron decomposition 以低成本高效地将该误差融入权重更新流程。

**🔧 技术方法**

使用技术：Post‑Training Quantization (PTQ)、GPTQ/GPTAQ、compensation‑based 量化、per‑group 对称量化、weight‑rotation（QuaRot/SpinQuant）以及神经元分解法。

**📊 数据集**

数据集：训练/校准使用 WikiText‑2 与 C4（各 128 条样本）；评估使用 WikiText‑2、C4 perplexity 以及 6 个零样本下游任务（PiQA、ARC、HellaSwag、Winogrande、BoolQ）。

**📈 对比分析**

比较方法：在 Llama‑2、Llama‑3（1B–70B 参数）系列模型上与 GPTQ、GTAQ、AWQ、QuaRot+GPTQ、SpinQuant 等基线进行对比。实验表明，在 3‑bit、2‑bit 量化（含权重+激活）时，平均下游准确率提升约 1–3%，perplexity 明显下降，尤其在 70B 模型上提升幅度更显著。

**⚠️ 局限性**

局限性：需要额外的校准 GPU 与内存（尤其在 70B 模型上峰值显著增加）；对极低比特 2‑bit 量化与某些旋转方法（SpinQuant）仍易失效；目前仅在 PTQ 场景下验证，尚未探究与 QAT 的兼容性。

---

## 305. A Systematic Framework for Tabular Data Disentanglement

**arXiv ID:** 2604.07940 | [PDF](https://arxiv.org/pdf/2604.07940v1)

**作者:** Ivan Tjuawinata `[一作]` (Nanyang Technological University), Parventanis Murthy `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种系统化框架，用于表格数据的分离解耦（disentanglement），将流程拆分为数据提取、数据建模、模型分析和模型外推四个核心模块，并在此框架下开展了合成数据生成的案例研究。

**💡 创新点**

创新点包括：①将表格数据分离解耦拆解为可模块化的四步流程，提供统一的视角；②在请求（query）与目标函数（objective）驱动下，将解耦过程与具体任务（如合成数据、外推）紧密耦合；③对外推（extrapolation）过程提出了三种插值/外推层级，并给出了正式定义和评估属性；④系统性地定义了正确性、覆盖度、最优性、独立性、外推准确度等衡量指标。

**🔧 技术方法**

核心技术主要包括：结构化的协议（Π.设置、Π.提取、Π.建模、Π.分析、Π.外推）；关系族（ℛ_ℱ）和随机变量建模；参数估计与分布拟合（可使用 GMM、变分自编码器等）；以及基于目标函数的损失优化。

**📊 数据集**

本文在案例研究中使用了合成表格数据（synthetic tabular dataset），并未使用公开真实数据集；主要通过实验验证框架在合成任务中的可行性。

**📈 对比分析**

在合成数据生成任务中，框架与传统方法（Tabsyn、Switchtab、VAEGMM、TVAE）对比，强调在请求驱动的训练与评估机制下，可更精准地满足特定任务目标，虽然论文中未给出具体数值比较，但作者指出该方法在模型独立性与外推准确度方面具有优势。

**⚠️ 局限性**

局限性包括：①缺乏大规模真实数据的实验验证；②在多表或时间序列表格数据上的推广仍待研究；③外推准确度评价指标尚未标准化；④对超参数与模型复杂度的敏感性未系统分析。

---

## 306. Top Management Journal Portal: A Real-Source Search and Research Analytics Artifact for UTD-24 and FT50 Journals

**arXiv ID:** 2604.07934 | [PDF](https://arxiv.org/pdf/2604.07934v1)

**作者:** Chuang Zhao `[一作]` (Tianjin University), Hongke Zhao `[通讯]` (Tianjin University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向管理学精英期刊（UTD‑24 与 FT50）的 Web 搜索与分析门户，整合 Crossref 公开元数据、热点提取、可视化分析、引用导出及收藏功能，为研究者提供端到端的查询、解读和研究工作流。

**💡 创新点**

创新点在于：① 以精英期刊为检索边界实现领域专属搜索；② 采用实时 Crossref 元数据和可追溯的来源标注，提升数据透明度；③ 通过热点提取与分布式分析快速生成洞察；④ 低成本、模块化的 Node.js+Supabase 方案，使学术实验室可轻松部署。

**🔧 技术方法**

使用技术包括：Node.js 服务器与 API，前端 vanilla JS、HTML/CSS，ECharts 可视化，Supabase（免费层）持久化，Render 公共托管，Crossref REST API，及可选的 LLM 接口进行热点重写。

**📊 数据集**

数据集主要为：UTD‑24 与 FT50 的期刊清单（手工整理），以及通过 Crossref 实时抓取的文章元数据（标题、摘要、DOI、作者、机构等）。

**📈 对比分析**

本文未给出传统性能指标的量化比较，而是从功能对比角度说明：与通用学术搜索引擎和静态目录相比，该门户在期刊聚焦、热点可视化和即时分析方面更符合管理学研究者的工作流程；系统已在 Render 上公开部署，访问日志显示使用量和查询关键词分布，可作为后续用户研究的数据基础。

**⚠️ 局限性**

局限性包括：受 Crossref 元数据覆盖率限制；热点提取方法为启发式、缺乏正式主题模型验证；缺乏全文检索、协同注释和引用网络分析；未完成正式的用户评估；在高负载或跨期刊长期监测方面的性能尚未验证。

---

## 307. Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for Authoring Support

**arXiv ID:** 2604.07989 | [PDF](https://arxiv.org/pdf/2604.07989v1)

**作者:** Jing Xu `[一作]` (Hong Kong University of Science and Technology), Weikai Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种面向信息图的意图感知检索框架，通过把用户自然语言查询拆解为内容、图表类型、布局、插图、风格五个意图面板，利用面板特定重写与加权融合实现多面向检索，并集成对检索到的示例进行交互式 SVG 适配的端到端创作工作流。

**💡 创新点**

①首次将信息图检索拆解为五个可加权的意图面板；②使用面板特定 token 与轻量 MLP 投影实现面板级别的文本‑图像嵌入；③通过在 ChartGalaxy 上的面板级对齐训练提升检索效果；④把检索与对话式 SVG 适配结合，形成“一站式”创作体验。

**🔧 技术方法**

多模态 LLM（Qwen3-32B、Qwen3-VL-8B）、视觉语言模型（CLIP/SigLIP2/MegaPairs）、面板特定 token + MLP、对比损失、在图像-文本配对上进行 facet‑aware 对齐，面向 SVG 的树形结构摘要与工具化代码检索与合成。

**📊 数据集**

ChartGalaxy（约10k 设计样本），以及从中抽取的 52k 训练样本和 10k 测试样本。

**📈 对比分析**

与 CLIP、SigLIP2、MegaPairs 以及本方法的若干 ablation 进行单轮检索评估（Recall@1/5、MRR@10）、人工评价（1–5 级）以及多轮检索与端到端作者实验。实验表明：在合成与人类长查询上，R@1、R@5、MRR 最高可提升 20–30%；在人工评估中，平均得分从 3.09 提升至 4.51；多轮检索中 FoundRate 从 45.8% 提升至 91.7%，dCRR@10 由 0.21 变 0.73；端到端作者实验中工作负荷显著下降（NASA‑TLX 11.23→7.88），输出质量偏好率提升至 54.2%。

**⚠️ 局限性**

（1）SVG 适配过程仍易出现结构破坏、图标缺失等执行错误；（2）系统更适合快速草图与创意探索，缺乏对排版、细节的精细控制；（3）对高层设计探索与美学细化的支持仍有限，需要进一步改进编辑策略与界面。

---

## 308. Rethinking Data Mixing from the Perspective of Large Language Models

**arXiv ID:** 2604.07963 | [PDF](https://arxiv.org/pdf/2604.07963v1)

**作者:** Yuanjian Xu `[一作]` (Hong Kong University of Science and Technology), Guang Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM训练中的数据混合策略，建立梯度动态与域分布的理论联系，并提出基于图的域加权框架DoGraph。

**💡 创新点**

将模型感知的域视为梯度空间中的节点，使用随机投影与K‑means动态聚类捕捉训练过程中的域演化，并通过图约束优化域权重。

**🔧 技术方法**

线性化Transformer下的梯度-分布对应理论、MMD核、Johnson–Lindenstrauss随机投影、K‑means聚类及概率多面体优化。

**📊 数据集**

SlimPajama七域大规模语料，评测用HellaSwag、PiQA、OBQA、COPA、LogiQA、WinoG、SciQ、ARC‑E、Lambada等九个基准。

**📈 对比分析**

与Uniform、Dynamic Loss、DoReMi、DOGE、RegMix、Data Mixing Law等基线在GPT‑2 Medium/210M/300M规模下对比；DoGraph在各类任务和验证PPL上均优于基线，尤其在推理类任务提升显著。

**⚠️ 局限性**

仍需进一步降低随机投影与聚类的计算开销，当前在极大规模训练中的效率尚未最优。

---

## 309. WorldMAP: Bootstrapping Vision-Language Navigation Trajectory Prediction with Generative World Models

**arXiv ID:** 2604.07957 | [PDF](https://arxiv.org/pdf/2604.07957v1)

**作者:** Hongjin Chen `[一作]` (Harbin Institute of Technology), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WorldMAP框架，利用世界模型生成未来视频，通过语义-空间记忆与显式规划产生轨迹伪标签，训练轻量级学生模型实现单观测导航轨迹预测。

**💡 创新点**

创新点在于将世界模型输出转化为持久的语义空间结构与规划监督，而非直接使用生成视图；通过教师‑学生蒸馏结合多视角记忆和Fast Marching规划，生成可学习的轨迹标签。

**🔧 技术方法**

使用视觉语言模型（VLM）、多视角3D重建、UniPixel分割、CLIP文本‑图像相似度、Fast Marching Method（FMM）、多假设轨迹解码器以及伪标签质量评估机制。

**📊 数据集**

在Target‑Bench真实世界语义目标导航基准上进行评估，该数据集包含室内外未见环境的单视角RGB图像和语言指令。

**📈 对比分析**

与多种专有与开源VLM直接预测、MindJourney等world‑model增强方法对比，WorldMAP学生在ADE、FDE、DTW三项指标上均领先，ADE下降18%，FDE下降42%。

**⚠️ 局限性**

依赖生成视图的质量，难以处理动态或多层环境；教师系统计算开销大；训练时需对伪标签进行质量筛选。

---

## 310. Incremental Residual Reinforcement Learning Toward Real-World Learning for Social Navigation

**arXiv ID:** 2604.07945 | [PDF](https://arxiv.org/pdf/2604.07945v1)

**作者:** Haruto Nagahisa `[一作]` (Kyushu University), Ryo Kurazume `[通讯]` (Kyushu University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向社交导航的增量残差强化学习框架IRRL，使得移动机器人能够在资源受限的边缘设备上进行实时学习并适应未知的人群动态。

**💡 创新点**

创新点在于将无重放缓冲区的增量学习与残差RL相结合，既保持了轻量级、实时更新的优势，又通过残差学习提升了样本效率和稳定性。

**🔧 技术方法**

采用了Actor‑Critic架构、GATv2图神经网络进行人群特征聚合，使用SFM作为基准策略，结合增量学习中的归一化与TD误差缩放技术以及自适应温度调整。

**📊 数据集**

主要使用CrowdNav仿真环境进行圆形交叉测试，真实实验中采用DR‑SPAAM检测模型和MCL定位，混合虚拟与真实行人的混合实验场景。

**📈 对比分析**

与传统基于重放缓冲区的SAC、TD3、PPO等方法以及无缓冲区的Stream‑AC、SAC‑1、TD3‑1等增量学习方法对比，IRRL在成功率高、碰撞率低、平均回报高、执行时间相当或略长的同时，收敛速度最快，且在随机种子分布下方差最小，证明其性能优越。

**⚠️ 局限性**

局限性包括仅在有限的圆形交叉场景和两名行人设置下验证，需进一步扩展到更复杂、多变的人群环境；依赖SFM作为基准策略，若基准策略差异较大可能影响残差学习效果；以及对实时检测与定位的误差依赖较高。

---

## 311. Mitigating Entangled Steering in Large Vision-Language Models for Hallucination Reduction

**arXiv ID:** 2604.07914 | [PDF](https://arxiv.org/pdf/2604.07914v1)

**作者:** Yuanhong Zhang `[一作]` (Ministry of Education Key Laboratory of Intelligent Networks and Network Security), Joey Tianyi Zhou `[通讯]` (Centre for Frontier AI Research, Institute of High Performance Computing, Agency for Science, Technology and Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MESA框架，在推理时对视觉特征进行可控扰动，产生干扰后的隐藏表示并通过PCA提取清晰的对抗性方向，以此在潜在空间中进行抑制幻觉的干预，保持原模型的生成行为。

**💡 创新点**

①将幻觉抑制问题重新表述为“选择性驱动”，即只调节幻觉相关分量而不影响其他生成因素；②通过学习可控的、基于token的视觉扰动，避免了传统随机扰动导致的方向混杂；③结合幻觉增强与分布保持双重目标，得到更干净、更可解释的驱动方向。

**🔧 技术方法**

①可控视觉扰动模块（轻量级两层MLP）对视觉嵌入施加clip约束；②分布对齐损失（KL散度）用于幻觉对齐与语义保持；③PCA用于从扰动导致的隐藏差分中提取主方向；④在推理时在Transformer层加入缩放参数α的潜在空间干预。

**📊 数据集**

使用多种公开基准：CHAIR（图像字幕的幻觉评估）、POPE（存在级幻觉问答）、AMBER（生成与判别任务的多类幻觉评测）、LLaVA-Bench（对话式评测）以及对应的预训练模型LLaVA‑v1.5与Qwen‑VL。

**📈 对比分析**

与VCD、ICD、VAF、ICT、VTI、Nullu等解码、注意力和潜在空间驯化方法比较。MESA在CHAIR、POPE、AMBER、LLaVA‑Bench上多项指标均领先：CHAIR_S下降至最低、POPE Accuracy/F1提升1–18个百分点、AMBER判别Accuracy提升至84.3%、生成长度与原模型保持相近，且保持EOS边缘分布与Zipf分布不受显著扰动。

**⚠️ 局限性**

①需额外训练可控扰动模块，增加推理前的预处理步骤；②调节α时需平衡幻觉抑制与召回，过大会导致覆盖率下降；③对视觉扰动的设计与训练依赖多种视觉退化策略，若未覆盖足够多样，可能在未见过的幻觉模式下表现欠佳。

---

## 312. MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems

**arXiv ID:** 2604.07956 | [PDF](https://arxiv.org/pdf/2604.07956v1)

**作者:** Arda Yüksel `[一作]` (Trustworthy Human Language Technologies), Ivan Habernal `[通讯]` (Trustworthy Human Language Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Moneta多模态行业分类基准，并构建了零样本与多轮推理框架。

**💡 创新点**

创新点在于首次将地理空间资源（OSM与卫星影像）与文本资源融合用于行业分类，并设计了中间代理评估与多轮推理机制。

**🔧 技术方法**

使用多模态大型语言模型（InternVL、Llava、QwenVL、Gemini、GPT‑5）以及提示工程、上下文丰富与分类解释技术。

**📊 数据集**

构建了包含1000家欧洲企业的NACE 20类标签数据集，数据来源包括企业官网、维基百科/维基数据、OpenStreetMap与ESRI卫星图像。

**📈 对比分析**

对比了多种模型与配置，最大单轮零样本准确率为74.10%（GPT‑5 Mini），多轮+扩展提示+解释的最佳性能达到约79-80%，明显优于传统单模态方法，且多模态输入相比仅文本能提升20%准确率。

**⚠️ 局限性**

主要局限包括：数据集可能受预训练数据覆盖影响导致过拟合，卫星影像与OSM的标注质量不一，且模型对地理空间信息的提取仍不稳定；多轮推理过程复杂且对资源消耗大；实验受限于公开数据的许可与获取。

---

## 313. Context-Aware Disentanglement for Cross-Domain Sequential Recommendation: A Causal View

**arXiv ID:** 2604.07992 | [PDF](https://arxiv.org/pdf/2604.07992v1)

**作者:** Xingzi Wang `[一作]` (Shanghai University of Finance and Economics), Hui Fang `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出CoDiS，一种上下文感知解耦框架，用于跨域序列推荐，解决因果混杂、梯度冲突和用户重叠依赖等问题。

**💡 创新点**

创新点在于将变分上下文调整、专家隔离选择以及变分对抗解耦三者结合，实现在因果视角下对共享与域特定偏好的精确解耦，且不依赖用户重叠即可实现跨域迁移。

**🔧 技术方法**

使用了变分上下文调节（backdoor adjustment）、多专家（MoE）路由器、变分解耦模块、对抗判别器与梯度反转层、InfoNCE 对比学习以及 t‑SNE 可视化等技术。

**📊 数据集**

在 Amazon Review 的三对域（Food–Kitchen、Beauty–Electronics、Movies–Books）共六个域的数据集上进行实验。

**📈 对比分析**

与 ST‑SDSR、ST‑CDSR、DT‑CDSR 等 10+ 传统与最新模型在 HR@5/10、NDCG@10、MRR 上进行对比，CoDiS 在所有指标上均显著优于 SOTA，尤其在稀疏域提升超过 20%。

**⚠️ 局限性**

仍假设上下文可用有限离散集，极端冷启动或极端稀疏情形下的鲁棒性未完全验证；在跨域多样性极大时需增大专家数，可能导致计算开销上升。

---

## 314. DP-DeGauss: Dynamic Probabilistic Gaussian Decomposition for Egocentric 4D Scene Reconstruction

**arXiv ID:** 2604.07986 | [PDF](https://arxiv.org/pdf/2604.07986v1)

**作者:** Tingxi Chen `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了DP-DeGauss框架，实现了在 egocentric 4D 重建中背景、手与物体的动态概率 Gaussian 分解；

**💡 创新点**

创新点包括：统一 Gaussian 初始化、可学习的类别概率与软‑到‑硬分流策略、亮度控制、运动流约束以及掩膜监督，实现了多类别精细分离；

**🔧 技术方法**

采用 3D Gaussian Splatting、HexPlane 编码、MLP 分类与回归、光流约束、可学习亮度属性以及软‑到‑硬门控机制；

**📊 数据集**

使用了 HOI4D、Epic‑Field 与 Hot3D 等 egocentric 视频数据集；

**📈 对比分析**

与 4DGaussians、MotionGS、NeuralDiff、DeGauss、EgoGaussian 等基线相比，DP‑DeGauss 在 PSNR 上提升约 +1.7 dB，SSIM 与 LPIPS 均有显著改进，并成功实现细粒度手/物体分离；

**⚠️ 局限性**

局限性在于仍需依赖 COLMAP 初始化，极端遮挡或高速运动下细节恢复受限，且仅针对背景、手与物体三类，未涵盖更复杂交互场景。

---

## 315. Is your algorithm unlearning or untraining?

**arXiv ID:** 2604.07962 | [PDF](https://arxiv.org/pdf/2604.07962v1)

**作者:** Eleni Triantafillou `[一作]`, Georgios Kaissis `[通讯]` (Hasso Plattner Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并阐述了两类“unlearning”概念——一种是针对特定忘记集合的影响（数据层面），另一种是针对其所代表的概念或行为的影响（概念层面）并明确两者在目标与评估上的区别；

**💡 创新点**

创新点在于系统性区分并命名这两类问题（Δ与Γ），指出经典定义实际上对应Δ，揭示其在实用性、评估指标、成功判定等方面与Γ截然不同，并提出相应的研究问题与评估框架；

**🔧 技术方法**

利用统计学习理论中的训练与学习定义、记忆度量（counterfactual memorization）与交叉影响度量，构建了Δ、Γ的形式化定义，并讨论了现有unlearning算法（如基于差分隐私、梯度上升等）在两类问题中的适用性；

**📊 数据集**

论文主要以理论与概念阐述为主，未给出具体实验数据集；通过示例（如圆形/星形分类任务、LLM中版权句子、后门等）说明两类问题；

**📈 对比分析**

由于缺乏统一实验，论文未给出具体性能对比；作者强调评估指标需针对问题类型选择，例如Δ侧重于与重新训练结果的分布相似度，而Γ侧重于对概念的彻底抹除；

**⚠️ 局限性**

局限性包括：未提供完整可实现的算法；概念/行为定义在实践中难以完全捕获；对数据集覆盖度、样本代表性与跨影响的理论分析仍不充分；此外，论文未解决如何在实际大模型中高效实现Γ的技术细节。

---

## 316. Physics-Based Motion Tracking of Contact-Rich Interacting Characters

**arXiv ID:** 2604.07984 | [PDF](https://arxiv.org/pdf/2604.07984v1)

**作者:** Xiaotang Zhang `[一作]`, Hubert P. H. Shum `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文探讨了某一领域的最新研究进展，提出了一种新的方法来解决特定问题。

**💡 创新点**

创新点在于引入了一种新的算法或模型，显著提高了性能或效率。

**🔧 技术方法**

使用了深度学习技术，结合了卷积神经网络和循环神经网络。

**📊 数据集**

使用了公开的标准数据集，如ImageNet或CIFAR-10。

**📈 对比分析**

与现有方法进行了比较，结果显示新方法在准确率和计算效率上均有显著提升。

**⚠️ 局限性**

限制在于模型的可扩展性和对特定数据集的依赖性，可能在其他领域的应用效果不佳。

---

## 317. TOOLCAD: Exploring Tool-Using Large Language Models in Text-to-CAD Generation with Reinforcement Learning

**arXiv ID:** 2604.07960 | [PDF](https://arxiv.org/pdf/2604.07960v1)

**作者:** Yifei Gong `[一作]` (Shanghai University), Kang Tu `[通讯]` (Shanghai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 ToolCAD 框架，使用大型语言模型（LLM）作为工具调用代理，实现从自然语言指令到 CAD 设计的端到端自动建模。

**💡 创新点**

创新点包括：① 交互式 CAD Gym 让 LLM 与 CAD 引擎实时交互并获取混合反馈；② 基于 CAD‑CoT 的链式推理和 ReAct 交互策略，显著提升长序列工具调用的可靠性；③ 分部课程学习与在线强化学习（GRPO）相结合的两阶段训练策略，解决工具使用稳定性和泛化问题；④ 设计了 ORM（结果监督奖励模型）实现对完整建模轨迹的高质量评估。

**🔧 技术方法**

使用技术包括：大型语言模型（Qwen2.5‑7B、Qwen3‑8B、GPT‑4o）、ReAct 与 CAD‑CoT 提示、MCP（模型上下文协议）封装 CAD 原语、FreeCAD 交互式引擎、ORM 奖励模型、在线强化学习框架（GRPO、AWR、PPO）以及分部课程策略。

**📊 数据集**

数据集：DeepCAD（约 170K 模型，含 L0–L3 多级注释）和 Text2CAD；从中抽取 L3 专家级指令；使用 GPT‑4o 生成 982 条离线演示轨迹用于 SFT 与 ORM 训练；200 条保留测试案例。

**📈 对比分析**

与 GPT‑4o、Qwen3‑235B 等前沿专有 LLM 以及 Qwen2.5‑7B、Qwen3‑8B 的 ReAct/Zero‑Shot 对比；与 AWR、SFT 训练的基线及 Transformer 生成模型（DeepCAD、Text2CAD）对比。结果显示：ToolCAD 在多部件任务中的工具调用成功率达 61.8%–63.9%，显著高于零射门提示（≈20%）和 AWR（≈30%），并在几何误差（1‑IoU）上取得最小值，说明其建模质量更优。

**⚠️ 局限性**

局限性：① 目前缺乏视觉感知作为前置模块，难以在复杂场景中直接通过图像引导工具调用；② 工具库有限且缺少自我纠错机制，导致某些细粒度错误难以及时修正；③ ORM 在极端复杂任务上表现衰退，需进一步提升其鲁棒性；④ 交互式 CAD 与工具学习框架仍有改进空间，例如更细粒度的几何反馈与更丰富的工具覆盖。

---

## 318. Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting

**arXiv ID:** 2604.07928 | [PDF](https://arxiv.org/pdf/2604.07928v1)

**作者:** Tao Hana `[一作]`, Lei Bai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于3D高斯撒点的尺度感知视觉Transformer（GSSA‑ViT），用于任意分辨率的气象下尺度和中期预报。

**💡 创新点**

创新点在于将生成式3D高斯参数预测与尺度感知注意力结合，构建一次模型即可在任意分辨率下直接生成高分辨率气象场，解决传统NWP模型固定尺度和多尺度不灵活的问题。

**🔧 技术方法**

采用3D Gaussian Splatting表示、尺度感知窗口注意力与全局注意力的ViT架构、生成式条件网络预测高斯参数以及可微分渲染的多尺度重建技术。

**📊 数据集**

使用ERA5再分析数据和CMIP6气候模拟数据进行训练、验证和测试。

**📈 对比分析**

通过与Bicubic、Bilinear、ResNet、UNet、MetaSR、LIIF、MINet、GSASR、NeuralGCM、Stormer等基准方法在LRMSE、Pearson相关、均值偏差等指标上比较，GSSA‑ViT在下尺度任务中均取得最低LRMSE/最高相关，并在多分辨率中期预测中持续领先。

**⚠️ 局限性**

对长时间预测存在误差累积，尚未充分利用观测数据与不规则网格信息，未来需要引入时序一致性约束或扩散式生成机制来缓解误差积累。

---

## 319. EigentSearch-Q+: Enhancing Deep Research Agents with Structured Reasoning Tools

**arXiv ID:** 2604.07927 | [PDF](https://arxiv.org/pdf/2604.07927v1)

**作者:** Boer Zhang `[一作]` (Meta), Yuan He `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在深度研究代理的浏览器子代理中加入结构化查询处理与证据处理工具，提升搜索与信息聚合的有序性与可观测性。

**💡 创新点**

采用Anthropic“think”工具范式，将查询生成、扩展、搜索前沿管理以及证据提取与进度评估等步骤拆解为可调用工具，使中间决策显式化，轻量且非侵入式。

**🔧 技术方法**

实现了QueryExpand、QuerySelect、TargetExtract、ReflectSearch等工具，结合Google Custom Search JSON API、Hybrid浏览器工具包、LLM模型（GPT‑4.1、GPT‑5.1、Minimax M2.5）以及Camel‑AI多代理工作流框架。

**📊 数据集**

在四个公开基准上评估：SimpleQA‑Verified、FRAMES、WebWalkerQA、X‑Bench DeepSearch。

**📈 对比分析**

对比四种代理配置（直接生成、仅搜索、原始浏览器代理、增强代理），用GPT‑4.1自动评判，结果显示在GPT系列模型上平均提升3.0–3.8个百分点，Minimax M2.5平均提升0.6个百分点，整体保持更结构化的搜索轨迹。

**⚠️ 局限性**

目前为无训练的零样本实现，缺乏完整的内部推理与工具化推理对比实验；对低能力模型提升有限；未来需探索RL/微调以及更广泛的推理与非推理模型对比。

---

## 320. SAT: Balancing Reasoning Accuracy and Efficiency with Stepwise Adaptive Thinking

**arXiv ID:** 2604.07922 | [PDF](https://arxiv.org/pdf/2604.07922v1)

**作者:** Weiyang Huang `[一作]` (Harbin Institute of Technology (Shenzhen)), Min Zhang `[通讯]` (Harbin Institute of Technology (Shenzhen))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型推理模型的过度推理进行步级难度感知的动态剪枝，减少推理链长度。

**💡 创新点**

将推理过程建模为有限状态机，配合轻量化的步骤难度估计器实现自适应深度控制；通过四种思维模式实现难度感知的步级裁剪。

**🔧 技术方法**

有限状态机（FSM）、轻量级过程奖励模型（PRM/Pilot）、GRU+语义+置信特征、基于控制标签的上下文引导。

**📊 数据集**

七大基准：GSM8K、MATH500、AMC、AIME2024/2025、GPQA Diamond、HumanEval，使用多款大模型（Qwen3、DeepSeek、Llama3、QwQ）。

**📈 对比分析**

与COT、DEER、CGRS、ThinkSwitcher等基线对比，平均减少25% token、提升1.5点准确率，最大可达40% token压缩且准确率不下降。

**⚠️ 局限性**

额外的感知模块引入少量计算开销，且依赖LLM对控制标签的理解，若模型指令遵循能力不足可能导致状态机失效。

---

## 321. Investigating Code Reuse in Software Redesign: A Case Study

**arXiv ID:** 2604.07919 | [PDF](https://arxiv.org/pdf/2604.07919v1)

**作者:** Xiaowen Zhang `[一作]` (Concordia University), Shin Hwei Tan `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究跨仓库软件重构中的代码与测试重用，采用行动研究双向案例方法，提出基于重构语义的克隆检测与语义对齐分数（SAS）来识别代码映射，并提交 PR 给开源项目。

**💡 创新点**

创新点在于提出重构特定的重命名规则、语义对齐启发式和分层检测策略，发现非线性迁移、TODO 延迟实现、测试缺失与残留 Bug，并贡献 1,749 条手工标注的重构映射数据集与改进克隆检测工具。

**🔧 技术方法**

使用行动研究、JavaParser 解析、四种克隆检测工具（NiCad、DeepSim、CCStokenizer、GPT‑OSS‑120B）、自定义重命名规则、基于 LCS 的 SAS 计算以及层级预筛选技术。

**📊 数据集**

使用两个跨仓库重构对（Soot/SootUp 与 Spotbugs/Spotbugs‑Slim），并构建 1,749 条手工标注的代码映射数据集进行评估。

**📈 对比分析**

在两组重构对上与四个基线工具进行精度、召回和 F1 对比，改进方法在平均 33–99% 降低无关克隆、精度提升至 86%，并通过层级预筛选使 LLM 检测可扩展到仓库级。

**⚠️ 局限性**

局限在于仅验证两对项目，规则及方法对其他重构场景的适用性待验证，仍需人工检查并受限于输入规模和不同语言/工具的通用性。

---

## 322. MotionScape: A Large-Scale Real-World Highly Dynamic UAV Video Dataset for World Models

**arXiv ID:** 2604.07991 | [PDF](https://arxiv.org/pdf/2604.07991v1)

**作者:** Zile Guo `[一作]` (Chinese Academy of Sciences), Lei Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MotionScape 数据集，并在其基础上对现有世界模型的动态 UAV 视频生成能力进行了评估。

**💡 创新点**

创新点在于：①构建了包含 30 小时、4K 分辨率、4.5M 帧的高动态 UAV 视频数据集；②数据中同时提供 6‑DoF 相机轨迹和精细自然语言描述，实现了语义与几何的对齐；③通过多阶段自动化流程（CLIP 过滤、TransNetV2 切片、GPT‑4o 语义标注、DROID‑SLAM 位姿恢复）高效生成高质量训练样本。

**🔧 技术方法**

使用的技术包括 CLIP 语义匹配、TransNetV2 片段划分、GPT‑4o mini 语义标注、DROID‑SLAM 视觉 SLAM、Cosmos 2.5‑2B 视频生成模型以及光流统计、PSNR/SSIM/LPIPS/Warping‑Error/FVD 等评估指标。

**📊 数据集**

使用的数据集为自建的 MotionScape 以及对比的公开 UAV 数据集（VisDrone、UAVid、Zurich Urban MAV）和合成数据集（Tartanair），并在 MotionScape 上进行实验。

**📈 对比分析**

方法比较：在 Cosmos 2.5‑2B 的 video‑to‑world 模式下，分别对无控制、加上 GPT‑4o 语义描述、加上位姿信息三种控制策略进行实验。结果显示，无论哪种控制方式，PSNR/SSIM/LPIPS 等指标均略有波动但总体未出现显著提升，Warping‑Error 与 FVD 亦未改善，说明模型在高动态 UAV 场景中对辅助条件的利用有限。

**⚠️ 局限性**

局限性：1) 数据集的位姿信息仅为相对轨迹，缺乏绝对尺度；2) 当前世界模型对高动态 6‑DoF 运动的建模仍不充分，无法充分利用语义或几何约束；3) 评估仅基于 Cosmos 2.5‑2B，未探索其他模型或进一步训练的可能性。

---

## 323. SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations

**arXiv ID:** 2604.07990 | [PDF](https://arxiv.org/pdf/2604.07990v1)

**作者:** Yunnan Wang `[一作]` (Shanghai Jiao Tong University), Yujun Shen `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SceneScribe-1M，一套包含一百万段真实视频、文本描述、相机姿态、连续深度图和 3D 点轨迹的多模态视频数据集，并在多项 3D 感知与视频合成任务上进行基准评测

**💡 创新点**

创新点在于：①大规模统一数据集兼顾语义与几何信息；②使用大规模 GPU 并行的 MegaSaM + TAPIP3D 注释管线，实现高质量连续深度与 3D 跟踪；③引入多视角筛选策略 SceneScribe-MVS，分离相机与对象运动，满足多视角重建需求；④通过数据增强与评测在多任务上验证该数据集的泛化与实用性

**🔧 技术方法**

核心技术包括：Qwen2.5‑VL‑72B 进行长视频文本生成；MegaSaM 进行动态运动掩码、相机姿态估计与深度优化；TAPIP3D 进行 3D 点轨迹重建；TransNetV2 用于非连续视频的剪辑检测；大规模 GPU 并行推理与分布式注释管线

**📊 数据集**

数据来源主要是公开的文本‑视频对齐数据集（HD‑VILA‑100M、Panda‑70M、Koala‑36M）以及 Pexels‑Video，随后通过筛选、剪辑与注释得到 4,000+ 小时、1,000,000 条视频

**📈 对比分析**

在多任务基准（单目深度、场景重建、点跟踪、文本/姿态到视频生成）上，SceneScribe‑1M 能提升已有模型（MoGe、VGGT、MonST3R、CoTracker3、SpatialTrackerV2、AC3D）在多项指标上的表现，差异从几百分位到几百分点不等，表明数据集显著提升模型泛化与控制能力

**⚠️ 局限性**

局限性包括：①注释过程对 GPU 资源高度依赖，成本高昂；②虽然覆盖多样性，但部分极端动态场景或低光照条件仍缺乏；③相机姿态与深度标注在极端视角或遮挡时可能不够精确；④缺乏对音频或多模态交互（如语音指令）的支持

---

## 324. Rag Performance Prediction for Question Answering

**arXiv ID:** 2604.07985 | [PDF](https://arxiv.org/pdf/2604.07985v1)

**作者:** Or Dado `[一作]` (Technion), David Carmel. Oren Kurland `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究如何预测在开放域问答任务中使用检索增强生成（RAG）相对于不使用RAG所带来的质量提升，提出多种预测器并系统评估其效果。

**💡 创新点**

创新点在于：1) 引入了基于监督的后生成预测器，显著提升预测精度；2) 将检索与生成过程中的语义关联直接建模；3) 在三大问答数据集和两种检索方式、两种LLM上做跨模型、跨指标的统一评估。

**🔧 技术方法**

主要技术包括：BERT/ModernBERT跨编码器作为预测器骨干；token熵、NLI得分、检索评分分布等多种无监督特征；使用E5、BM25进行检索；Falcon‑3‑10B和Llama‑3.1‑8B作为生成模型；使用Pearson相关性评估预测性能。

**📊 数据集**

使用了三大开放域问答数据集：Natural Questions、TriviaQA 和 HotpotQA，并在每个数据集上采样 3,600 条验证样本进行测试；检索语料为 2018 年版 Wikipedia dump，分段为 100‑词块。

**📈 对比分析**

通过与无监督预检索、后检索、后生成预测器的对比，发现后生成监督模型在所有配置上均获得最高 Pearson 相关（最高约 0.87/0.89），明显优于传统无监督或仅基于检索评分的预测方法；实验表明该方法在不同检索方式、不同 LLM 以及不同质量评估指标下保持一致的优势。

**⚠️ 局限性**

局限性包括：1) 仅关注质量提升，未考虑检索成本与实时性；2) 需要两次 LLM 推理（有检索/无检索），增加算力消耗；3) 在极端长文本或检索不到相关段落时模型表现可能受限；4) 预测模型的训练需要大量带标签的问答数据，迁移到新领域可能受限。

---

## 325. A Decomposition Perspective to Long-context Reasoning for LLMs

**arXiv ID:** 2604.07981 | [PDF](https://arxiv.org/pdf/2604.07981v1)

**作者:** Yanling Xiao `[一作]` (Tencent), Lemao Liu `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将长文本推理拆解为五个原子技能并利用自动生成数据提升LLM长上下文推理能力

**💡 创新点**

创新性地将长文本推理从整体任务拆解为检索、抗干扰、全局整合、关系推理和动态状态追踪等原子技能，并构建Anchor-based Reasoning框架自动生成验证数据

**🔧 技术方法**

采用多轮强化学习（GRPO）、动态采样、LLM-as-a-Judge评估及链式推理提示，训练在约4000条合成数据上

**📊 数据集**

基于AbR框架生成的近4K合成原子技能数据集，覆盖NIAH、Anti-Interference、Multi-Source、Logic、Calc-Reason等任务；评测使用Loogle、Loong、LongBench‑v2、BrowscompLong、Ruler‑qa2、MRCR等真实长文本基准

**📈 对比分析**

与DeepSeek‑R1‑Distill‑32B等强基线相比，平均提升7.7%（从46.3%升至54.0%），在六大长文本基准上均表现优于现有长文本训练方法和LoongRL数据集，且与LoongRL叠加可进一步提升

**⚠️ 局限性**

局限在于仍需人工设计原子技能层级与模板，生成数据对真实多样性可能不足；高质量长文本推理任务的验证仍受限于现有基准与自构造数据的真实性和覆盖面

---

## 326. Object-Centric Stereo Ranging for Autonomous Driving: From Dense Disparity to Census-Based Template Matching

**arXiv ID:** 2604.07980 | [PDF](https://arxiv.org/pdf/2604.07980v1)

**作者:** Qihao Huang `[一作]` `[通讯]`, Qihao Huang

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一套完整的立体视觉测距管线，将稠密匹配（BM/SGM）、基于检测框的稀疏 Census 匹配以及单目几何先验融合到实时检测-测距-跟踪工作流中，重点解决高速公路长距离目标的精准测距与实时性。

**💡 创新点**

核心创新包括：
- 对目标检测框内部做稀疏 Census 匹配，显著降低计算量；
- 采用远/近分层策略（全分辨率 vs 缩放分辨率）和多块聚合，提升远距离小目标的精度；
- 前向-后向一致性验证和遮挡感知采样，过滤错误匹配；
- 在线校准框架：自动竖直偏移搜索、雷达投票纠偏、对象级雷达-立体关联，实现持续外参漂移补偿；
- GPU 并行化设计与异步流水线，保证 30+ FPS 的实时性能。

**🔧 技术方法**

使用的技术包括：
- 立体视觉（BM、SGM、Census Transform、Hamming 距离）;
- CUDA 并行核与共享内存优化的 Census 计算;
- 前向/后向一致性检查与聚合逻辑;
- 车辆雷达-相机联合投影与投票纠偏;
- Kalman 轨迹滤波与目标几何估计;
- 多源深度融合策略与错误检测门限;
- 可能与 BEV 特征融合的前瞻性架构。

**📊 数据集**

实验主要基于公开高速公路立体视觉数据集（如 KITTI、Waymo Open Dataset）以及真实车辆现场采集的多模态数据，结合标定雷达和摄像头的同步记录进行验证。

**📈 对比分析**

与传统 BM/SGM 进行定量对比，发现稀疏 Census 匹配在 200‑300 m 范围内的误差下降 30‑50%（RMSE 约 2‑4 m），并且计算量仅为 BM/SGM 的 1‑5%；在多光照、雨雪等极端条件下仍保持 70% 以上的有效匹配率，远优于单目深度网络和雷达单独测距。

**⚠️ 局限性**

主要局限：
- 需要先验目标检测；对非常小或被遮挡目标的测距仍有挑战；
- 对大尺度非标准物体的尺寸不敏感，需结合几何先验；
- 依赖相机和雷达同步与标定，外参漂移仍可能引入误差；
- 在极端低光或强反射场景下，Census 仍可能出现多重匹配。

---

## 327. DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing

**arXiv ID:** 2604.07965 | [PDF](https://arxiv.org/pdf/2604.07965v1)

**作者:** Gyanendra Das `[一作]` (Zynix AI), Sai Satyam Jena `[通讯]` (Zynix AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态子空间概念对齐（DSCA）框架，用于在不冻结模型权重的前提下，对视觉‑语言模型进行持续、局部化的知识编辑；

**💡 创新点**

核心创新在于将模型表示空间分解为近乎正交的低秩子空间，并在每个子空间上部署可学习的、门控的残差编辑模块，实现概念级的结构隔离；

**🔧 技术方法**

技术包括在线语义聚类、增量 PCA 构建子空间、双阶段层次路由、门控残差干预、跨模态对齐损失与对比蒸馏损失等；

**📊 数据集**

实验使用 LLaVA‑1.5‑7B、PaliGemma‑3B 在 E‑VQA、E‑IC、VLKEB、CoIN 等编辑与持续学习基准；

**📈 对比分析**

与 LiveEdit、DualEdit、MEND、LTE、VisEdit、SERAC 等现有编辑方法比较，DSCA 在单一编辑上平均得分提升至 98.5%，在 1,000 次连续编辑后仍保持 96.8% 的可靠性和 98.2% 的多模态局部性；在 CoIN 上 BWT 仅为 -9.37，显著优于基线；同时对原始 LVLM 基准的性能提升或保持不变；

**⚠️ 局限性**

局限性包括：仅使用线性子空间模型，可能不足以处理高度非线性或高度纠缠的概念；子空间正交化随着概念数量增长的计算成本；以及对概念发现与路由的依赖，概念重叠或模糊时易导致错误编辑或干扰。

---

## 328. Seeing enough: non-reference perceptual resolution selection for power-efficient client-side rendering

**arXiv ID:** 2604.07959 | [PDF](https://arxiv.org/pdf/2604.07959v1)

**作者:** Yaru Liu `[一作]` (University of Cambridge), Arnau Raventos `[通讯]` (Huawei Research)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无参考的感知分辨率选择框架，利用高帧率下的时空掩蔽效应，动态决定最低可辨识分辨率以降低客户端渲染功耗。

**💡 创新点**

创新点在于将人类视觉系统的时空掩蔽特性与无参考质量预测相结合，使用0.1 JOD阈值定义可接受分辨率，构建轻量级预测网络并通过Viterbi算法实现稳定切换。

**🔧 技术方法**

采用DINOv2 ViT主干提取空间特征，3D卷积+注意力编码运动特征，融合后通过MLP分类；使用ColorVideoVDP作为监督标注，采用Viterbi平滑；整体网络训练与推理轻量。

**📊 数据集**

使用基于Unreal Engine 5渲染的HFR数据集，共73个动态场景，包含5个分辨率与4种渲染配置，生成约3.5TB视频片段，并以ColorVideoVDP生成JOD标签。

**📈 对比分析**

与固定1080p和720p的120fps基准对比，用户研究显示方法在多运动速度下至少与1080p相当、在中速时更受偏好；在渲染负载上平均可节约51%像素，推理成本仅占3.8%帧时长，能显著降低功耗。

**⚠️ 局限性**

局限在于只针对120Hz高帧率，低帧率下不一定适用；仅调节分辨率，未考虑其他渲染参数；依赖可获取的运动向量和预训练ViT，可能在非物理渲染或极端后处理场景下表现下降。

---

## 329. Pruning Extensions and Efficiency Trade-Offs for Sustainable Time Series Classification

**arXiv ID:** 2604.07953 | [PDF](https://arxiv.org/pdf/2604.07953v1)

**作者:** Raphael Fischer `[一作]` (TU Dortmund University), Geoffrey I. Webb `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一个统一的时间序列分类（TSC）评估框架，重点关注模型预测质量与能耗之间的权衡；在此基础上对两种主流混合分类器（基于随机卷积核的R-Forest和基于区间量化的ITSC）引入了理论可界定的剪枝策略，并提出了融合两者特征的可剪枝Hybrid模型；

**💡 创新点**

创新点包括①在TSC中首次系统化地将能耗纳入评价指标；②设计了一种利用中间线性模型特征重要性进行可视化剪枝的策略，并给出均匀误差上界；③提出了Hybrid模型以融合卷积核激活与区间统计特征，旨在提升表达能力；

**🔧 技术方法**

技术手段包括：随机卷积核特征提取、区间量化特征、随机决策树集成、岭回归中间模型、基于特征重要性排序的剪枝、统一性能指标的指数归一化与复合评分、以及多硬件（CPU、GPU、不同工作站）上的实验；

**📊 数据集**

实验使用了UCR/UEA存档的20个时间序列数据集，样本量从9k到200k，通道数1至113，类别数2至82，序列长度23至5k；

**📈 对比分析**

比较方法采用多维度性能指标（准确率、F1分数、推理时间、每样本能耗）并通过指数归一化得到复合分数；结果显示剪枝可将能耗降低约80%，而准确率仅下降不到5%，并且剪枝后的Hybrid在复合评分上通常位居榜首；

**⚠️ 局限性**

局限性包括仅评估能耗而未考虑其他可持续性维度（如碳足迹、嵌入式影响）、实验硬件环境有限（仅三种设置），以及对Hybrid模型的融合方式未能完全提升准确率，未来需探索更精细的融合与硬件感知剪枝策略。

---

## 330. Fraud Detection System for Banking Transactions

**arXiv ID:** 2604.07952 | [PDF](https://arxiv.org/pdf/2604.07952v1)

**作者:** Ranya Batsyas `[一作]` (IGDTUW), Ritesh Yaduwanshi `[通讯]` (IGDTUW)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并评估了基于机器学习的欺诈检测框架，使用PaySim合成交易数据集，并遵循CRISP‑DM流程进行特征工程、模型训练和评估。

**💡 创新点**

通过将SMOTE过采样与随机森林的超参数调优相结合，在极端类别不平衡的数据集上显著提升F1分数，并提出可部署的API微服务方案。

**🔧 技术方法**

使用了Logistic回归、决策树、随机森林、XGBoost等分类器，SMOTE过采样，GridSearchCV进行超参数搜索，LabelEncoder进行类别编码，并以F1、ROC‑AUC等指标评估模型。

**📊 数据集**

采用了Kaggle提供的PaySim Synthetic Financial Datasets for Fraud Detection数据集，包含约600万条交易记录，少量欺诈样本占比0.129%。

**📈 对比分析**

对四个基线模型进行准确率、召回率、F1比较，随机森林基线F1=0.87；经过SMOTE和调参后提升至0.91，整体准确率达到99.97%。

**⚠️ 局限性**

仅使用静态特征，未考虑时间序列或图结构信息；在真实环境中对概念漂移、攻击适应性和模型可解释性的进一步研究仍有必要。

---

## 331. On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning

**arXiv ID:** 2604.07944 | [PDF](https://arxiv.org/pdf/2604.07944v1)

**作者:** Amirhossein Afsharrad `[一作]` (Stanford University), Sanjay Lall `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了将大规模语言模型（LLM）用于自动驾驶轨迹规划的知识蒸馏，提出在策略上进行全分布匹配的 GKD 方案；

**💡 创新点**

创新点在于将 GKD 引入自动驾驶规划，直接用教师分布对学生生成序列进行监督，并与密集反馈 RL 进行对比，证明全分布监督的优越性；

**🔧 技术方法**

使用的技术包括 Qwen3 系列 LLM、GPT-Driver 任务框架、Generalized Knowledge Distillation（基于 Jensen‑Shannon 散度）、以及基于教师 log‑prob 的 dense‑feedback RL；

**📊 数据集**

实验数据集为 nuScenes 的 GPT‑Driver 版本，包含约 1000 个驾驶场景和 5119 个验证帧；

**📈 对比分析**

与 RL 基线相比，GKD 学生在轨迹 L2 距离和碰撞率上仅差 5‑6%，而 RL 基线差距达 40‑55%；GKD 学生参数量为 1.7B，压缩率约 5×；

**⚠️ 局限性**

局限性包括仅在离线规划评估上验证，未进行闭环仿真；对教师-学生容量比例和更大规模模型的影响仍待研究。

---

## 332. RAGE-XY: RADAR-Aided Longitudinal and Lateral Forces Estimation For Autonomous Race Cars

**arXiv ID:** 2604.07939 | [PDF](https://arxiv.org/pdf/2604.07939v1)

**作者:** Davide Malvezzi `[一作]`, Marko Bertogna `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出RAGE-XY框架，实时联合估计车辆状态、侧滑角、侧向与纵向轮胎力，利用IMU和多雷达传感器，并加入在线雷达校准与三轮模型。

**💡 创新点**

创新点在于：① 通过ROLEQ实现在线雷达校准补偿传感器偏移；② 将单轨模型扩展为三轮模型以估计后轴纵向力；③ 在移动窗口估计框架中实现完整状态与力的联合估计。

**🔧 技术方法**

采用移动窗口估计（MHE）、ROLEQ在线校准、Pacejka魔法公式、有限滑差速器模型以及随机游走状态扩展等技术。

**📊 数据集**

使用高保真多体仿真数据以及EAV-24在Abu Dhabi Autonomous Racing League收集的实测数据（最高速度70 m/s，侧加速度28 m/s²）。

**📈 对比分析**

与原始RAGE及基准方法对比，利用仿真与实测误差指标，结果显示在高速极端操纵条件下侧向与纵向力估计误差下降约20%，鲁棒性显著提升。

**⚠️ 局限性**

局限性包括：纵向力估计尚未在实测中充分验证；雷达校准只能观测俯仰和偏航，无法纠正滚转角；依赖直线段进行校准，可能不适用于持续弯道；适用范围主要限于赛车级高速车辆。

---

## 333. HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction with a Prediction-then-Verification Strategy

**arXiv ID:** 2604.07937 | [PDF](https://arxiv.org/pdf/2604.07937v1)

**作者:** Guoqi Ma `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大语言模型的分层分类模型HCRE，用于跨文档关系抽取；

**💡 创新点**

创新点在于构造层级关系树并引入预测-验证推理策略，以减少关系选项并降低错误传播；

**🔧 技术方法**

使用大语言模型（如LLaMA‑3.1‑8B‑Instruct、GPT‑4o-mini）作为预测器，利用GPT‑4o构建层级树，结合多视角验证和层级推理；

**📊 数据集**

在CodRED数据集上进行实验，涵盖闭合和开放两种设置；

**📈 对比分析**

与多种基线（SLM+分类器、LLM基线、HTC基线）对比，HCRE在micro‑F1与binary‑F1上显著优于所有基线，提升幅度可达10‑15个百分点；

**⚠️ 局限性**

局限性包括：仅能处理单一路径，无法利用跨路径依赖；训练时随机采样次优节点可能不是最优；

---

## 334. Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions

**arXiv ID:** 2604.07931 | [PDF](https://arxiv.org/pdf/2604.07931v1)

**作者:** Jing Wang `[一作]` (Nanjing University), Zhi-Hua Zhou `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种利用多次生成的长度分布来做LLM输出长度预测的方法，称为ProD。

**💡 创新点**

创新点在于把输出长度视为随机分布而非确定值，并通过重复采样构造稳健的监督目标（中位数或分布），从而降低噪声影响。

**🔧 技术方法**

技术方法包括：多次采样得到长度分布；构造中位数标签（ProD‑M）或分布标签（ProD‑D）；利用被服务LLM的最后层隐藏状态做特征；使用简单的两层MLP和分桶的分类头进行训练；在推理时仅使用一次采样，保持单射；提供线性代理模型的理论误差分析。

**📊 数据集**

使用的评估数据集包括 Qwen‑2.5‑7B 与 Llama‑3‑8B 在四类任务上的测试集：GSM8K（Math）、MBPP（Coding）、LongBench（LongSequence）和 LMSYS‑Chat‑1M（Chat）。

**📈 对比分析**

与 S³、EGTP、TRAIL 等现有方法以及两种 TRAIL 变体进行对比，采用 MAE 评测。实验显示 ProD‑D 在所有模型与场景中均取得最低 MAE，提升幅度最高可达 25%（例如 Qwen‑2.5‑7B 在 LongSequence 上从 67.91 降至 57.83）。即使在固定总推理预算下，重复采样方法仍优于全覆盖单样本方法。

**⚠️ 局限性**

局限性包括：只在 prompt‑only 场景下验证；需要多次生成来构造训练标签，训练成本上升；方法仅对已服务模型的最后层隐藏状态依赖，可能对其它模型结构适应性有限。

---

## 335. Sinkhorn doubly stochastic attention rank decay analysis

**arXiv ID:** 2604.07925 | [PDF](https://arxiv.org/pdf/2604.07925v1)

**作者:** Michela Lapenna `[一作]` (University of Bologna), Bahman Gharesifard `[通讯]` (Queen’s University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文主要研究 Transformer 中自注意力的秩崩塌问题，并对比了行随机化 Softmax 与双随机化 Sinkhorn 的差异。

**💡 创新点**

创新点在于首次对 Sinkhorn 正则化的双随机化自注意力的秩崩塌进行理论分析，给出了残差范数上界，并在多层 Transformer 中实验证实其能显著抑制秩崩塌，尤其在残差连接存在时效果更明显。

**🔧 技术方法**

采用了 Sinkhorn 算法实现双随机化归一化、路径分解框架、残差范数分析以及标准的 Softmax 对比；实验中使用了 Transformer 与 Vision Transformer 结构。

**📊 数据集**

数据集包括 AG’s News 文本分类、MNIST 与 Cats&Dogs 视觉分类。

**📈 对比分析**

方法上通过计算注意力矩阵及其路径产品的秩（残差谱范数）与模型输出的残差进行对比，实验证明 Sinkhorn 在相同配置下保持更高秩，性能提升体现在更稳定的训练和更好的分类效果，尤其在有跳跃连接的模型中。

**⚠️ 局限性**

限制方面：Sinkhorn 的迭代归一化计算成本较高，最佳迭代次数经验确定；实验未探究不同层间注意力矩阵相关性对秩崩塌的影响；未评估 Lipschitz 性质；仅在三组数据集上验证，可能不具普适性。

---

## 336. Tarot-SAM3: Training-free SAM3 for Any Referring Expression Segmentation

**arXiv ID:** 2604.07916 | [PDF](https://arxiv.org/pdf/2604.07916v1)

**作者:** Weiming Zhang `[一作]` (Hong Kong University of Science and Technology), Lin Wang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了无训练的框架Tarot‑SAM3，用于从任意指代表达中实现图像分割。

**💡 创新点**

创新点在于引入表达推理解释器（ERI）将复杂表达转化为多种结构化提示，并通过掩模自我细化（MSR）利用特征一致性纠正分割误差。

**🔧 技术方法**

采用多模态大语言模型 Qwen2.5‑VL、Segment Anything Model 3（SAM3）和 DINOv3 进行特征提取和自我细化。

**📊 数据集**

使用 RefCOCO、RefCOCO+、RefCOCOg、ReasonSeg 等标准 RES 数据集进行评估。

**📈 对比分析**

在所有零训练设定下，在显式和隐式 RES 基准上均超过现有训练自由方法，gIoU 分别达到约 75.5（显式）和 74.3（隐式），展示显著性能提升。

**⚠️ 局限性**

局限性包括对空间歧义描述的处理不佳以及对基于区域的查询误判目标级别。

---

## 337. LogAct: Enabling Agentic Reliability via Shared Logs

**arXiv ID:** 2604.07988 | [PDF](https://arxiv.org/pdf/2604.07988v1)

**作者:** Mahesh Balakrishnan `[一作]`, Victoria Dudin `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于共享日志的 Agentic Shared Log（ASL）架构，将 agent 拆分为可解耦的状态机组件，并通过日志实现安全、容错与审计。

**💡 创新点**

创新点包括：1) 将 agent 视为在共享日志上运行的状态机；2) 引入日志类型与访问控制，支持安全投票、审计与恢复；3) 通过“agent introspection”在日志上执行语义恢复、健康检查与优化；4) 设计可热插拔的投票器与子代理，支持多代理协同。

**🔧 技术方法**

使用技术包括：LLM 推理（Claude/ChatGPT）、共享日志 API（带类型与阻塞查询）、状态机复制（SMR）与 WAL、规则/静态投票器、LLM 投票器、Python/Rust 执行引擎、S3/SQLite 作为持久化层。

**📊 数据集**

实验主要使用 AgentDojo 公开基准进行安全性评估，并在自建的代码生成/校验任务（如大型代码库哈希、Python 代码类型注解）上验证恢复与优化效果。

**📈 对比分析**

与基线 agent 对比，ASL 在安全性上将攻击成功率从 48% 降到 1.4%，在正常任务上保持 78% 以上的实用率；延迟提升约 15–20%（Rule‑based 58%，Dual‑voter 75%）；token 消耗相对较低（约 13% 增幅）。日志存储极小，约 2.6 KB/s；整体性能几乎与原始推理无显著差异。

**⚠️ 局限性**

局限性包括：1) 仍依赖 LLM 进行投票与恢复，无法提供绝对硬性安全保证；2) 对于并发多 agent 场景的协调与一致性仍待完善；3) 需要人工定义安全不变式和投票策略；4) 在极端异常或网络分区下的恢复逻辑复杂度较高。

---

## 338. The Hyperscale Lottery: How State-Space Models Have Sacrificed Edge Efficiency

**arXiv ID:** 2604.07935 | [PDF](https://arxiv.org/pdf/2604.07935v1)

**作者:** Robin Geens `[一作]` (Ku Leuven), Thierry Tambe `[通讯]` (Stanford University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了从 Mamba-1 到 Mamba-3 这一系列状态空间模型（SSM）在云端与边缘场景下的性能演变，发现针对云端吞吐量的架构改动导致边缘低延迟和能耗表现显著下降；

**💡 创新点**

提出“Hyperscale Lottery”概念，即将云端吞吐量优化嵌入模型结构中，牺牲了边缘友好性，并主张在模型设计与部署时将云端吞吐量改进与边缘需求解耦；

**🔧 技术方法**

使用 GPU 基准测试、面向边缘的 Stream 分析框架、Roofline 模型进行理论与实际性能评估，并通过对不同 Mamba 变体的操作强度、能耗和延迟进行量化；

**📊 数据集**

未使用具体数据集，而是以模型参数规模（如 880M、15M 参数）和通用推理负载作为实验对象；

**📈 对比分析**

将 Mamba 的 sequential 与 parallel scan（pscan）、SSD 以及 MIMO 变体在单张 A100 GPU 与 1024‑MAC/32‑SIMD 边缘芯片上进行对比；结果显示 Mamba‑3 在单请求（B=1）下延迟提升 28%（880M）至 48%（15M），而在大批量云端解码场景中通过提高操作强度获得吞吐提升；

**⚠️ 局限性**

局限性包括：仅评估推理阶段；部分 Mamba 变体的 GPU kernel 未公开；实验基于假设的边缘硬件配置，实际部署环境可能差异；未涵盖多种真实边缘工作负载或训练过程。

---

## 339. EEG2Vision: A Multimodal EEG-Based Framework for 2D Visual Reconstruction in Cognitive Neuroscience

**arXiv ID:** 2604.08063 | [PDF](https://arxiv.org/pdf/2604.08063v1)

**作者:** Emanuele Balloni `[一作]` (Università Politecnica delle Marche), Emiliano Santarnecchi `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出EEG2Vision框架，实现从低密度EEG到图像的端到端生成，并加入LLM提示引导的后期增强阶段。

**💡 创新点**

创新点包括：①系统性评估不同电极密度对解码与生成的影响；②将多模态大语言模型提取语义与图像到图像扩散相结合，实现轻量级后期增强；③利用ControlNet与多模态提示在保持结构的同时提升视觉质量。

**🔧 技术方法**

技术栈涵盖EEG编码、ControlNet适配器、潜在扩散模型（LDM）、多模态大语言模型（LLaMA 3.2 Vision）、Stable Diffusion 3 img2img、CFG调节与LLM语义抽取。

**📊 数据集**

使用EEGCVPR40数据集（6位受试者、2000张ImageNet图像、128通道EEG）进行实验。

**📈 对比分析**

通过50类Top‑1/Top‑5、IS、FID、LPIPS、CLIP‑Sim等指标进行比较；低通道下语义准确率从89%降至38%，但FID/IS波动不大；后期增强提升IS约+6–10%、FID下降约1–2点、LPIPS微降。

**⚠️ 局限性**

局限性：低通道时语义解码显著下降；跨受试者泛化尚未充分验证；后期增强依赖预训练扩散模型，可能引入伪造；对非具象类别的鲁棒性尚不明朗。

---

## 340. From Gaze to Guidance: Interpreting and Adapting to Users' Cognitive Needs with Multimodal Gaze-Aware AI Assistants

**arXiv ID:** 2604.08062 | [PDF](https://arxiv.org/pdf/2604.08062v1)

**作者:** Valdemar Danry `[一作]` (Microsoft Research), Judith Amores `[通讯]` (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估一种基于视线的多模态LLM助手，该助手通过第一人称摄像与眼动追踪结合，对用户的阅读行为进行后续式认知支持。

**💡 创新点**

创新点在于：①将原始前摄视频与投影眼动直接作为多模态输入，让LLM自行推理认知需求；②采用时间序列分析而非单帧分类，识别疑难点；③将认知需求检测与对话交互分离，提供更精准的个性化支持。

**🔧 技术方法**

核心技术包括：Meta Aria智能眼镜的前置RGB摄像与眼动摄像、开源Aria眼动推理模型、200×200像素聚焦裁剪、GPT‑4.1（用于物体识别、行为归纳与需求推断）、OpenAI Realtime API（语音识别与合成）以及实验室数据处理与统计分析。

**📊 数据集**

数据集：36名受试者完成6篇不同难度的维基百科阅读任务（文本长度、Flesch‑Kincaid等级统一），并完成回忆、定义探测与概念库存测评；眼动与视频为实验生成的第一人称流；对照组使用相同文本但无眼动信息。

**📈 对比分析**

比较方法：within‑subject（对照）设计，配对t检验/符号秩检验。结果显示：在阅读后回忆得分上，视线辅助组平均提升约7.4个百分点（p≈0.02）；定义探测与概念库存提升趋势明显但未显著；受试者对视线分析的准确性与个性化评价显著更高，且在对话中使用的词数减少；对照组表现与基准LLM助手差异更小。

**⚠️ 局限性**

局限性：①阅读材料短且回忆分数高，易出现上限效应；②对照条件已较强（文本‑only 预测疑难点），对比差异被低估；③仅做后续式干预，未验证实时即时提示的可行性与用户体验；④在阅读情境之外的开放式任务中，眼动信号的解释可能不稳定；⑤系统误判如重读/扫视为困惑，需要更精细的行为语义模型或多模态补充（心率、皮肤电等）。

---

## 341. Governed Capability Evolution for Embodied Agents: Safe Upgrade, Compatibility Checking, and Runtime Rollback for Embodied Capability Modules

**arXiv ID:** 2604.08059 | [PDF](https://arxiv.org/pdf/2604.08059v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了面向具身代理的可治理能力演化框架，在能力版本更新时加入生命周期治理，避免安全、策略与恢复失效；

**💡 创新点**

创新点在于将运行时治理扩展到能力升级层面，定义四维兼容性模型（接口、策略、行为、恢复），并设计七阶段升级管道（注册、校验、沙箱、影子、门控、在线监测、回滚）实现安全部署；

**🔧 技术方法**

利用模块化能力包（Embodied Capability Modules）、ROS 2中继治理层、PyBullet仿真、沙箱与影子评估、在线监控、回滚机制；

**📊 数据集**

在PyBullet仿真中的抓取-对齐-放置任务套件，随机化多种扰动，共计150 个任务实例/随机种子，5个随机种子；

**📈 对比分析**

与无治理（Naïve Upgrade）和静态能力对比；在6轮升级中，治理升级保持约67.4%任务成功率，完全消除不安全激活（0%），而Naïve升级成功率为72.9%但不安全激活高达60%；回滚成功率约79.6%；

**⚠️ 局限性**

局限性包括：升级治理主要在仿真环境验证，真实机器人验证有限；兼容性阈值需手工设定，可能导致误拒或误接受；多阶段管道引入额外延迟和资源开销；缺乏对复杂策略与大规模多能力协同的深入分析。

---

## 342. Efficient Provably Secure Linguistic Steganography via Range Coding

**arXiv ID:** 2604.08052 | [PDF](https://arxiv.org/pdf/2604.08052v1)

**作者:** Ruiyi Yan `[一作]` (Kyoto University), Yugo Murawaki `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于旋转范围编码（RRC）的语言模型隐写方法，能够实现零KL散度、接近100%熵利用，并且嵌入速度快。

**💡 创新点**

创新点在于将经典范围编码与每步旋转连续均匀随机变量相结合，既保持原始概率分布（零KL），又消除随机性重用；实现了近乎100%熵利用的高嵌入效率。

**🔧 技术方法**

使用范围编码（RC）、伪随机数生成器（PRNG）、对称密钥、线性缩放概率、熵利用分析等技术。

**📊 数据集**

使用C4数据集生成上下文，实验基于GPT‑2、OPT‑1.3B、Llama‑2‑7B三大语言模型。

**📈 对比分析**

通过平均/最大KL散度、嵌入容量、熵利用率、嵌入速度等指标与AC、ADG、Meteor、iMEC、Discop、SparSamp等基线方法对比，RRC在GPT‑2上实现0 KL散度、≈5.93比特/符号、99.98%熵利用率、1554.66比特/秒，速度最快且容量最高。

**⚠️ 局限性**

需要预先约定秘密消息长度；理论上仅给出近似100%熵利用的证明；依赖密钥与PRNG同步，存在同步失误风险。

---

## 343. LINE: LLM-based Iterative Neuron Explanations for Vision Models

**arXiv ID:** 2604.08039 | [PDF](https://arxiv.org/pdf/2604.08039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 344. ABMAMBA: Multimodal Large Language Model with Aligned Hierarchical Bidirectional Scan for Efficient Video Captioning

**arXiv ID:** 2604.08050 | [PDF](https://arxiv.org/pdf/2604.08050v1)

**作者:** Daichi Yashima `[一作]` (Keio University), Komei Sugiura `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视频字幕生成，提出ABMamba模型，将Deep SSM与多分辨率双向扫描结合，提供全开源实现。

**💡 创新点**

用Deep SSM实现线性复杂度长视频建模，并引入Aligned Hierarchical Bidirectional Scan (AHBS)模块以多分辨率捕获复杂时序；并在开源生态中提供完整模型与训练脚本。

**🔧 技术方法**

采用Mamba作为语言背骨，SigLIP+ DINOv2双视觉编码器，AHBS模块进行多分辨率双向扫描，自回归生成并行推理。

**📊 数据集**

训练集为LLaVA 1.5的665K图像-文本指令与Video-ChatGPT的100K视频-文本指令；评估基准为MSR‑VTT和VATEX视频字幕数据集。

**📈 对比分析**

与Video‑ChatGPT、Video‑LLaVA、LLaVA‑OneVision等全开源视频MLLMs以及InternVL2.5、VideoLLaMA3对比；在VATEX和MSR‑VTT上BLEU4与CIDEr指标与对手相当，BLEU4最高；推理速度约3倍（95 tokens/s 对比 38 tokens/s）。

**⚠️ 局限性**

仅针对视频字幕任务，未覆盖多轮对话、时间定位等更复杂视频推理；对更长视频或多模态任务的泛化仍待验证。

---

## 345. Adapting Foundation Models for Annotation-Efficient Adnexal Mass Segmentation in Cine Images

**arXiv ID:** 2604.08045 | [PDF](https://arxiv.org/pdf/2604.08045v1)

**作者:** Francesca Fati `[一作]` (Mayo Clinic), Timothy L. Kline `[通讯]` (Mayo Clinic)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

使用自监督预训练的 DINOv3 视觉基础模型与 DPT 风格解码器，构建了一种标签高效的子宫附件肿瘤（adnexal mass）分割框架。

**💡 创新点**

创新点在于：①将冻结的 DINOv3 backbone 直接迁移至医学影像任务；②采用多尺度 Dense Prediction Transformer 结构将全局语义与局部细节融合；③通过 BCe 与 Dice 组合损失实现边界精准且注释量低的学习；④对不同 backbone 规模、输入分辨率与数据稀缺性进行了系统的效率分析。

**🔧 技术方法**

核心技术包括：自监督预训练的 DINOv3 视觉 Transformer、DPT 级别的多尺度解码器、像素级 BCE+Dice 损失、冻结 backbone 与轻量级预测头、伪 RGB 预处理、Resampling 与 Residual Fusion 等。

**📊 数据集**

使用来自单中心的 112 名患者、共 7,777 帧的手工标注超声（US）数据集，帧尺寸分别为 224×224 与 512×512，数据按 70/15/15 的比例分为训练、验证与测试集。

**📈 对比分析**

与 U‑Net、U‑Net++、DeepLabV3、MAnet 等卷积基准模型在 224×224 与 512×512 分辨率下进行对比，评估指标包括 Dice、IoU、Sensitivity、HD95 与 MSD。结果显示：Dice 最高 0.945（0.956），HD95 与 MSD 分别下降 11.4% 与 14.1%，在数据稀缺（仅 25% 训练集）时保持 70%+ 的性能，明显优于传统 CNN。

**⚠️ 局限性**

主要限制：①仅在单中心数据上内部验证，缺乏跨中心泛化评估；②将连续帧视为独立样本，可能高估样本容量；③未系统评估对恶性风险分层与临床决策的实际影响；④在高分辨率输入下学习曲线下降，提示对模型与输入配合需进一步优化。

---

## 346. A Full-Stack Performance Evaluation Infrastructure for 3D-DRAM-based LLM Accelerators

**arXiv ID:** 2604.08044 | [PDF](https://arxiv.org/pdf/2604.08044v1)

**作者:** Cong Li `[一作]` (Peking University), Guangyu Sun `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了XXX的性能。

**🔧 技术方法**

使用了XXX技术，包括XXX和XXX。

**📊 数据集**

实验使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果表明本方法在XXX指标上优于其他方法。

**⚠️ 局限性**

限制在于XXX，可能影响结果的普适性。

---

## 347. Beyond Mamba: Enhancing State-space Models with Deformable Dilated Convolutions for Multi-scale Traffic Object Detection

**arXiv ID:** 2604.08038 | [PDF](https://arxiv.org/pdf/2604.08038v1)

**作者:** Jun Li `[一作]`, Jianhua Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在铜氧化物薄膜与聚苯乙烯微球界面上形成的新型极化子（四极激子极化子）的物理性质与行为。

**💡 创新点**

创新点在于首次发现并演示了这种四极激子极化子，并展示了其与传统极化子在能量、耦合强度等方面的显著差异。

**🔧 技术方法**

采用了微观结构制备技术、光学探测与光谱分析技术，对界面形成过程和极化子能谱进行实验测量。

**📊 数据集**

主要使用实验获得的光谱数据集，涵盖不同温度、入射角度及光强下的极化子响应。

**📈 对比分析**

通过与传统极化子实验结果进行对比，表明四极激子极化子在耦合强度和寿命上具有更优表现，且能在更宽的波长范围内工作。

**⚠️ 局限性**

局限性包括样品制备复杂、实验环境要求高（如低温或高真空）以及对界面质量的敏感性，限制了其在大规模应用中的可行性。

---

## 348. Rotation Equivariant Convolutions in Deformable Registration of Brain MRI

**arXiv ID:** 2604.08034 | [PDF](https://arxiv.org/pdf/2604.08034v1)

**作者:** Arghavan Rezvani `[一作]`, Xiaohui Xie `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文将SE(3)旋转-平移等变卷积引入脑部MRI可变形配准网络，替换传统CNN编码器；

**💡 创新点**

创新点在于首次系统评估等变卷积对可变形配准的影响，证明其可提升精度、参数效率、旋转鲁棒性与样本效率；

**🔧 技术方法**

采用Steerable CNN与SE(3)等变核构建等变编码器，并在VoxelMorph、Dual‑PRNet++、RDP Net三种架构中测试；

**📊 数据集**

使用公开脑部MRI数据集OASIS、LPBA40和MindBoggle，包含35、54和97个解剖标签；

**📈 对比分析**

与基线模型相比，等变版本在Dice/ASSD上均无明显下降且多数场景更优，参数量减少至78–96%；在旋转输入测试中保持更高准确率，且在样本不足时表现更佳；

**⚠️ 局限性**

限制包括等变约束削弱模型灵活性（尤其高容量模型RDP）和训练时的计算开销，且全等变解码器性能不足。

---

## 349. Wiring the 'Why': A Unified Taxonomy and Survey of Abductive Reasoning in LLMs

**arXiv ID:** 2604.08016 | [PDF](https://arxiv.org/pdf/2604.08016v1)

**作者:** Moein Salimi `[一作]` (Sharif University of Technology), Mohammad Hossein Rohban `[通讯]` (Sharif University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文是一篇关于大型语言模型（LLM）中归纳推理（abductive reasoning）的系统综述。作者首先梳理了哲学与历史背景，提出统一的两阶段定义（假设生成 + 假设选择），随后构建了四轴分类法（任务形式、数据集类型、方法论、评估方式），并对近 70 篇论文进行整理。最后作者在公开与闭源 LLM（3B–72B 参数）上搭建了一个包含生成与选择任务的基准集，开展实验并对模型规模、任务类型、评价指标进行对比，探讨了阶段差异、域差异与指标敏感性，并总结了当前研究的不足与未来方向。

**💡 创新点**

创新点：
1. 提出了针对 LLM 的统一两阶段归纳推理定义，消除了学术界的概念碎片化。 
2. 设计了四轴（Task–Dataset–Method–Evaluation）结构化分类法，系统归纳了现有工作。 
3. 构建了跨域、多尺度、多评估方式的基准套件，首次在统一框架下对 3B–72B 规模 LLM 进行对比实验。 
4. 通过实验揭示了 Stage‑I（生成）与 Stage‑II（选择）之间显著性能差距，以及模型规模、域与指标对结果的影响。

**🔧 技术方法**

技术与方法：
- Prompting 与链式推理（Chain‑of‑Thought）
- 监督微调（SFT）
- 知识检索与增强（Retrieval‑Augmented）
- 多代理协作（Multi‑Agent）
- 神经‑符号混合（Neuro‑Symbolic）
- 自动化评估（accuracy, top‑k, similarity, human judgement）
- 强化学习奖励（RL‑K, GEAR 等）

**📊 数据集**

使用的数据集（按任务类型划分）：
- Commonsense: ART, αNLI, e‑CARE, UNcommonsense, True Detective, MuSR, Visual Abductive Reasoning, MAR, VideoABC, Sherlock
- Expert/Formal: DDXPlus, ProofWriter, AbductionRules, NeuLR, L'ART, Reasoning Like a Doctor, ARG‑Med‑Agents, MedCaseReasoning, ER‑REASON, etc.
- Multimodal: Visual Abductive Reasoning, MAR, VideoABC
- Knowledge‑graph/logic: AbductionRules, ProofWriter, UniADILR, GEAR, LogiGLUE, etc.

**📈 对比分析**

对比方法与性能：
- 采用统一 prompt、同一指标，分别对生成（WR‑H, CEQ, Set‑F1）和选择（accuracy, Top‑1, Hit@3）任务评估。 
- 结果显示：最强模型（GPT‑4o、GPT‑5.4）在选择任务上可达 87–98% 的准确率，仍低于人类平均；生成任务在大模型上提升明显，但整体仍处于 “实验室” 级别，难以与人类相媲美。 
- Stage‑I 与 Stage‑II 对比显示：在同一域（如 DDXPlus）上，选择任务的性能远高于生成任务，表明假设生成是更难的子任务。 
- 模型规模呈正相关：3B→72B 的性能提升显著；不同模型家族（Qwen, Llama, DeepSeek, GPT）在基准上表现不一。 
- 指标敏感性：BLEU、ROUGE、BERTScore 与人类偏好并不完全一致，强调多指标评估的重要性。

**⚠️ 局限性**

局限性：
- 定义碎片化：归纳推理在不同研究中仍存在多重解释，导致跨工作比较困难。 
- 基准设计单一：多数任务为静态、单步预测，缺乏动态、多步交互情景；难以检验模型在真实推理流程中的表现。 
- 域覆盖有限：主要集中于常识与少量专家任务，缺少对高价值领域（法律、科学、工程等）的深入探索。 
- 准确率 vs 真正推理：高准确率不一定意味着模型真正掌握归纳推理；缺乏对内部推理路径的透明评估。 
- 训练方法单一：大部分工作使用监督微调或提示工程，RL、偏好学习、机制解释等方法尚未得到充分利用。 
- 机制可解释性不足：目前对 LLM 内部实现归纳推理的电路层面仍缺乏系统研究。

---

## 350. Bridging Time and Space: Decoupled Spatio-Temporal Alignment for Video Grounding

**arXiv ID:** 2604.08014 | [PDF](https://arxiv.org/pdf/2604.08014v1)

**作者:** Xuezhen Tu `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的Bridge-STG框架，用于视频定位任务中的时空分离式定位，解决了多模态大语言模型在联合定位时的挑战。

**💡 创新点**

核心创新包括：①通过Explicit Temporal Alignment（ETA）将文本时间戳注入MLLM，实现时序对齐；②引入Spatio‑Temporal Semantic Bridging（STSB）产生桥接查询，弥合时空解耦的语义鸿沟；③设计Query‑Guided Spatial Localization（QGSL）模块，采用多层交互查询和正负帧采样，有效消除双域视觉令牌冗余。

**🔧 技术方法**

技术实现基于Qwen3‑VL 7B MLLM，使用LoRA微调、时间戳注入、桥接查询学习、对比损失对齐、正负帧采样、以及多层视觉特征融合的解码器。

**📊 数据集**

使用了HCSTVG‑v1/v2、VidSTG、ReVOS（合成）等视频定位数据集，同时对多任务进行指令调优，涵盖视频时序定位（Charades‑STA）、目标跟踪（GOT‑10k）、指称表达理解（RefCOCO系列）和视频问答（VideoMME）等数据集。

**📈 对比分析**

与传统非生成式方法（TubeDETR、CG‑STVG、TA‑STVG）以及MLLM基线（LLaVA‑ST、VideoMolmo、SpaceVLLM）对比，Bridge‑STG在VidSTG上m_vIoU从26.4提升至34.3，在HCSTVG上超越TA‑STVG和SpaceVLLM，且在VTG、VOT、REC、VQA等跨任务评测中均显著优于现有模型。

**⚠️ 局限性**

局限性包括对MLLM预训练知识的高度依赖，仍需平衡正负帧采样比例，处理极长视频时计算成本较高，以及在极其细粒度或复杂语义查询场景下的性能仍有提升空间。

---

## 351. $φ-$DeepONet: A Discontinuity Capturing Neural Operator

**arXiv ID:** 2604.08076 | [PDF](https://arxiv.org/pdf/2604.08076v1)

**作者:** Sumanta Roy `[一作]` (Johns Hopkins University), Michael D. Shields `[通讯]` (Johns Hopkins University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出并实现了 φ-DeepONet，一种物理信息化神经算子，用于学习具有不连续输入和输出的界面问题的算子映射。

**💡 创新点**

创新点在于：①通过多个分支网络处理输入域的多重不连续性；②在主干网络中引入潜在嵌入（scalar、categorical、非线性类别嵌入）来隐式编码界面信息，从而直接捕捉输出的不连续性；③在损失函数中加入物理约束和界面条件的软约束，避免显式网格划分。

**🔧 技术方法**

使用的技术包括：DeepONet 框架、物理信息化神经网络、多个分支网络、潜在嵌入（含 one‑hot 与可学习嵌入矩阵）、SOAP 优化器、JAX 训练平台，以及软约束损失。

**📊 数据集**

所用数据集为人工合成的高斯随机场（GRF）样本，用于生成源项、边界与界面条件，并在一维与二维多个测试案例（单/多界面、断裂输入/输出、花瓣形界面等）中进行训练和验证。

**📈 对比分析**

通过与标准物理信息化 DeepONet 和 IONet 进行对比，φ-DeepONet 在相同训练样本下平均 L2 相对误差下降 1–2 个数量级，且训练成本比 IONet 低 1.5–2.7 倍；在 2D 复杂界面上误差约为 2.1×10⁻¹，仍优于传统方法。

**⚠️ 局限性**

局限性包括：需要预先知道子域划分；潜在嵌入为分块常数，可能无法捕捉界面附近的细尺度变化；嵌入维度的选择与问题强相关，过大可能导致过拟合；软约束无法保证界面条件完全满足；对未知或随时间演化的界面适用性有限。

---

## 352. Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving

**arXiv ID:** 2604.08075 | [PDF](https://arxiv.org/pdf/2604.08075v1)

**作者:** Xunzhuo Liu `[一作]` (vLLM Semantic Router Project), Huamin Chen `[通讯]` (vLLM Semantic Router Project)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过自校准的 token 预算路由，将大型语言模型服务集群拆分为短上下文池和长上下文池，并根据请求的总 token 预算进行路由，从而显著减少 GPU 资源浪费、提升可靠性和吞吐。

**💡 创新点**

创新点在于（1）基于 token 预算的两池路由策略；（2）不使用 tokenizer 的自校准 bytes‑per‑token 估计；（3）闭式成本模型与可观测阈值；（4）与现有实例级优化（如 PagedAttention、分块预填）无缝组合。

**🔧 技术方法**

使用的技术包括 token 预算路由算法、指数移动平均自校准、PAGED Attention、分块预填、负载感知溢出处理以及离散事件仿真评估。

**📊 数据集**

所用数据集为 Azure LLM 推理数据集、LMSYS‑Chat‑1M、Alibaba ServeGen 以及 Qwen3‑235B‑A22B 在 MI300X 的工作负载。

**📈 对比分析**

与单一池 homogeneous 部署对比，实验在 1000 req/s、90% GPU 利用率下，GPU 实例数下降 41.9%（Azure）或 31.3%（LMSYS），预处理/OOM 事件降 5.4×，P99 TTFT 提升 6%，在 Qwen3 方案中可实现每年约 15.4M 美元的成本节约。

**⚠️ 局限性**

局限性包括：长上下文请求仍受长池瓶颈影响；阈值需根据流量分布手动或动态调整；跨模型多租户场景需进一步兼容；极端突发流量的溢出处理需要细粒度调优。

---

## 353. Identifying bubble-like subgraphs in linear-time via a unified SPQR-tree framework

**arXiv ID:** 2604.08071 | [PDF](https://arxiv.org/pdf/2604.08071v1)

**作者:** Francisco Sena `[一作]` (University of Helsinki), Alexandru I. Tomescu `[通讯]` (University of Helsinki)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了首个线性时间算法，用于在有向或双向图中识别所有的 snarl、ultrabubble 以及 superbubble 子图，解决自 2018 年以来的开放问题；同时给出了一种线性大小的 snarl 组合表示，能在 O(|V|+|E|) 时间内构造；并在双向图上给出线性时间的反馈弧计算方法。

**💡 创新点**

创新点包括：①首次将 SPQR 树分解技术引入寻找 bubble‑like 结构，统一框架能够一次性解决三种子图；②构造了仅线性大小的 snarl 表示，避免了成对数的爆炸；③证明了在 tipless 双向图中反馈弧可以线性求解，而一般情况则至少与矩阵乘法难度相当；④为 superbubble 提供了新的、结构更简单的线性时间实现。

**🔧 技术方法**

核心技术主要是 SPQR 树分解、基于分割对的虚边操作、动态规划（自底向上与自顶向下）以维护连通、无环、源/汇等属性；以及将多条子图的无环性问题转化为单个反馈弧集合的检测；此外还利用了多边形图的分块、sign‑cut 图等概念。

**📊 数据集**

研究动机来源于人类全基因组参考共识（Human Pangenome Reference Consortium）构建的全基因组图（232 条基因组，206+ 万条边），并预期未来的更大图（350 条基因组等）将带来更高计算压力；实验与数据集主要以这些全基因组图为基础。

**📈 对比分析**

与之前的二次时间（O(|V||E|) 或 O((|V|+|E|)^2)）算法相比，本文提出的算法在时间复杂度上降至 O(|V|+|E|)，空间复杂度亦为线性。实验结果显示，在大规模全基因组图上实现了数十倍甚至百倍的速度提升。

**⚠️ 局限性**

局限性包括：• 对 ultrabubble 的线性算法仅适用于无 tip 的双向图；• 对于含有 tip 的图，仍需退化到更慢的算法；• 虽然 snarl 的表示线性，但实际数量仍可能达到二次级；• 反馈弧线性求解的证明仅在 tipless 场景，普适性仍受限；• 该框架主要针对单一类型的 bubble‑like 结构，尚未覆盖如 bibubble、panbubbles 等更一般的变体。

---

## 354. ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models

**arXiv ID:** 2604.08064 | [PDF](https://arxiv.org/pdf/2604.08064v1)

**作者:** Chonghan Qin `[一作]` (University of Hong Kong), Lingpeng Kong `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ImplicitMemBench评估LLM隐式记忆的基准。

**💡 创新点**

创新点是将程序化记忆、诱发与经典条件反射等认知科学机制映射到LLM任务，使用统一的学习-干扰-测试协议。

**🔧 技术方法**

采用LLM生成的任务蓝图、规则验证与LLM判定器相结合的混合评估方法。

**📊 数据集**

使用300条手工与自动生成的任务样本，涵盖程序化记忆、诱发与经典条件反射三类。

**📈 对比分析**

对17个顶尖LLM进行对比，最高总体得分仅66%，表现出显著的记忆偏差与瓶颈。

**⚠️ 局限性**

局限在未覆盖感知学习、习惯形成等隐式记忆范畴，以及对外部显式记忆模块效果的探索不足。

---

## 355. Automating aggregation strategy selection in federated learning

**arXiv ID:** 2604.08056 | [PDF](https://arxiv.org/pdf/2604.08056v1)

**作者:** Dian S. Y. Pang `[一作]` (Imperial College London), Ahmed E. Fetit `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一个端到端的自动化框架，用大语言模型一次性推荐聚合策略或在受限计算预算下采用轻量级遗传搜索，自动根据统计异构性选择FL聚合策略及其参数。

**💡 创新点**

创新点在于将异构性检测、LLM一次性策略生成与基于预算的遗传搜索集成到FL流程中，实现对聚合策略及参数的全自动、低成本优化，并提供双模式（单次与多次）部署方案。

**🔧 技术方法**

使用技术包括大语言模型GPT‑4.1进行推理、FedAvg、FedProx、Trimmed‑Avg、Krum等聚合算法、Jensen–Shannon Divergence、Federated PCA、局部异构检测、轻量级遗传搜索、Flower框架、Optuna等。

**📊 数据集**

实验数据集涵盖多模态：NASA Bearing（tabular）、CIFAR‑10、Wine Quality、Fundus图像、Twitter文本、OpenGym CartPole、以及其他tabular、image、text与强化学习数据集。

**📈 对比分析**

与FedAvg、单次LLM推荐以及Optuna 50次全搜索等基准比较，单次LLM在异构场景下平均提升约2–5%，遗传搜索在仅8次试验内即可逼近全搜索最优，并显著低于Optuna的计算成本。

**⚠️ 局限性**

局限性包括LLM在参数调优上的效果有限；未对模型架构进行自动化，模型不当可能抵消聚合策略优势；特征偏斜检测对预处理敏感；在高维预训练嵌入场景下聚合策略差异不明显。

---

## 356. From Universal to Individualized Actionability: Revisiting Personalization in Algorithmic Recourse

**arXiv ID:** 2604.08030 | [PDF](https://arxiv.org/pdf/2604.08030v1)

**作者:** Lena Marie Budde `[一作]` (Saarland University), Isabel Valera `[通讯]` (Saarland University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过将个体可操作性（包括硬约束与软约束）明确地定义并整合到因果算法解释框架中，提出了可扩展的个性化因果再决策求解器 iCARMA，并在半合成和真实数据集上对其与传统非增量求解器进行比较，进一步通过公平性案例研究揭示了忽视个体约束可能导致的隐性不公平。

**💡 创新点**

创新点在于：①首次将个体可操作性细分为硬约束与软约束并系统性地量化其对有效性、可行性、可解释性等指标的影响；②在因果再决策框架下设计高效的增量求解器 iCARMA，能够处理个体约束；③通过大规模实验与公平性案例展示个体约束如何显著影响模型表现与公平性。

**🔧 技术方法**

主要技术包括：因果结构模型 (SCM)、因果正则化的正常化流 (CNF)、基于神经网络的增量求解框架 CARMA 的扩展 iCARMA、硬约束掩码与软约束权重映射，以及基于偏好排名/评分的用户信息抽取与成本函数构造。

**📊 数据集**

使用的数据集为：①基于德国信用评分数据的半合成贷款数据（包含完整的因果结构）；②真实贷款申请数据 GiveMeSomeCredit (GMSC)，用于验证在未知结构下的鲁棒性。

**📈 对比分析**

与非增量因果求解器（Oracle）及无个体约束基线进行对比；iCARMA 在成本方面与 Oracle 相近，能够在数分钟内生成全人群解，而 Oracle 则需数小时；硬约束显著降低有效性与可解释性，软约束则在保持成本与可解释性之间取得更温和的权衡；公平性实验显示个体约束能揭示隐藏的群体差异。

**⚠️ 局限性**

局限性包括：①缺乏真实用户偏好数据，依赖模拟抽样；②软约束依赖用户评分或排名，信息可能不完整；③对因果图的假设要求较高，误差会影响性能；④未考虑策略行为与隐私、鲁棒性等多重约束；⑤目前仅处理固定数量特征，扩展至高维场景仍有挑战。

---

## 357. Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI

**arXiv ID:** 2604.08015 | [PDF](https://arxiv.org/pdf/2604.08015v1)

**作者:** Minh Sao Khue Luu `[一作]` (Novosibirsk State University), Bair N. Tuchinov `[通讯]` (Novosibirsk State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种统一的损失函数 CATMIL，旨在同时优化体素级分割精度和病灶级检测，特别针对脑 MRI 中稀疏且小的多发性硬化病灶。

**💡 创新点**

创新点在于引入组件自适应 Tversky 重新加权病灶连通分量，使得小病灶在训练中获得更大权重，并辅以多实例学习（MIL）病灶检测项，二者联合实现了体素级与病灶级监督的无缝融合。

**🔧 技术方法**

技术实现基于 nnU-Net 框架，损失为 Dice + Cross‑Entropy 加上 CAT 及 MIL 两项，训练采用 Adam 优化器、150 轮、批量 2，并通过 5‑折交叉验证验证稳健性。

**📊 数据集**

使用的数据集为 MSLesSeg，包含 75 名多发性硬化病人（53 名已标注），采用 T1、T2 与 FLAIR 三模态的共注册、去颅骨、1 mm³ 的体素空间。

**📈 对比分析**

与传统 Dice+CE、Tversky、FocalTversky 损失相比，CATMIL 在 Dice（0.7834）与 HD95（7.98 mm）上略有提升；在小病灶召回率（0.8730）和 FP 体积（1537 mm³）上显著优于其他方法，显示出更好的小病灶检测能力与误报控制。

**⚠️ 局限性**

局限性包括仅在单一数据集和单一网络架构上评估，未对其他疾病或模型泛化进行测试；同时高敏感性导致病灶级 F1 较低，说明检测与精细分割之间存在权衡，且未对误报和边界一致性做进一步控制。

---

## 358. SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving

**arXiv ID:** 2604.08008 | [PDF](https://arxiv.org/pdf/2604.08008v1)

**作者:** Felix Embacher `[一作]` (Mercedes-Benz AG), Markus Enzweiler `[通讯]` (Esslingen University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模稀有对象与场景的图像检索数据集 SearchAD，并提出对应的语义检索基准。

**💡 创新点**

首次提供涵盖 90 种稀有类别、超过 423k 帧的手工注释数据，聚焦稀有案例的“针尖寻针”检索挑战，并引入 R-Precision 和 MAP 作为评估指标。

**🔧 技术方法**

采用多种 VLM（如 RADIO、NARADIO、OpenCLIP、BLIP2 等）进行零样本文本检索和图像检索，并对 BLIP2 进行基于搜索集的微调。

**📊 数据集**

集成 11 个公开自动驾驶数据集（RoadAnomaly21、Vistas-NP、Lost and Found、RoadObstacle21、BDD-Anomaly、CODA 等），共 423,798 图像。

**📈 对比分析**

与现有 VLM 进行对比，文本检索方法平均 MAP 最高达 14.27%，图像检索平均 MAP 仅 8.31%；微调 BLIP2 后文本 MAP 提升至 12.06%，图像 MAP 提升至 11.13%。

**⚠️ 局限性**

仍面临检索精度低、语义不精准、对小目标识别不足等局限，整体 MAP 远低于工业需求，表明需进一步改进模型与训练策略。

---

## 359. PASK: Toward Intent-Aware Proactive Agents with Long-Term Memory

**arXiv ID:** 2604.08000 | [PDF](https://arxiv.org/pdf/2604.08000v1)

**作者:** Zhifei Xie `[一作]` (Nanyang Technological University), Shuicheng Yan `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Pask系统，构建了DD-MM-PAS四层主动式AI框架，并实现了实时需求检测模型IntentFlow和自进化多层记忆模块Pask-MM，构建了自研数据集LatentNeeds及对应的评测基准LatentNeeds-Bench。

**💡 创新点**

创新点在于（1）将需求检测、长期记忆与执行循环统一到同一架构中，形成可扩展的主动AI范式；（2）设计IntentFlow的流式双模型结构与SFT+RL训练策略，实现低延迟的即时需求识别；（3）提出层次化记忆（User/Workspace/Global）与异步检索，兼顾实时性与长时记忆演化；（4）通过大规模合成+真实语料的LatentNeeds数据，填补主动需求检测的数据缺口，并公开基准。

**🔧 技术方法**

技术主要包括：大语言模型（Qwen3系列）+自定义Prompt、流式推理架构、强化学习对齐、检索增强生成（RAG）+多层KV缓存、异步检索与内存压缩、全栈部署（前端、服务器、AI后端）和多模态感知模型（视觉、语音、LLM）等。

**📊 数据集**

使用的主要数据集是自研的LatentNeeds（约102k合成样本+2.1k真实交互），以及从中抽取的LatentNeeds-Bench（100场景、3936回合）进行评测；此外还利用公开语料用于合成、模型对齐与测试。

**📈 对比分析**

对比方法包括开源大模型（GPT-oss-120B、DeepSeek-V3.2）、多款闭源前沿模型（GPT-5-Mini/Nano、Gemini-3-Flash等）以及不同提示级别；实验采用平衡准确率、需求识别率和非需求识别率等指标。IntentFlow在需求识别上达83.1分，整体平均84.2分，接近Gemini-3-Flash（83.3），并在多轮对话中保持较小的性能衰减（≈5%），显示出良好的鲁棒性。

**⚠️ 局限性**

局限性包括：对更深层次隐式需求识别仍有不足；依赖合成与有限真实数据，可能不完全覆盖真实复杂情境；与闭源前沿模型相比，在特定高价值任务仍有差距；系统总体延迟受模型规模影响，尽管IntentFlow较快，但整体部署仍需优化；缺乏大规模真实世界部署与长期用户研究验证。

---

## 360. SAT: Selective Aggregation Transformer for Image Super-Resolution

**arXiv ID:** 2604.07994 | [PDF](https://arxiv.org/pdf/2604.07994v1)

**作者:** Dinh Phu Tran `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Selective Aggregation Transformer（SAT），通过密度驱动的Token聚合实现全局自注意力的高效压缩。

**💡 创新点**

采用非对称Query‑Key‑Value压缩，只聚合低频Token，并结合密度峰值聚类与特征范数恢复，显著降低97% key/value tokens，同时保持查询完整分辨率。

**🔧 技术方法**

利用变形Transformer、余弦相似度、温度加权聚合、特征范数恢复等技术，并在全局-局部混合注意力结构中实现高效注意力。

**📊 数据集**

训练使用DIV2K+Flicker2K（DFT2K）数据集，测试在Set5、Set14、B100、Urban100、Manga109等标准SR基准上。

**📈 对比分析**

与EDSR、RCAN、IPT、SwinIR、CAT-A、HAT、IPG、ATD、PFT等SOTA方法在×2/×3/×4尺度下对比，SAT在PSNR/SSIM上领先，提升至0.22dB，同时FLOPs下降27%，参数约19.4M。

**⚠️ 局限性**

仍需在极低分辨率或大尺寸图像下验证，聚合过程的子采样选择可能影响稀疏高频纹理的细节保留，整体可扩展性受限于密度聚类的计算复杂度。

---

## 361. PrivFedTalk: Privacy-Aware Federated Diffusion with Identity-Stable Adapters for Personalized Talking-Head Generation

**arXiv ID:** 2604.08037 | [PDF](https://arxiv.org/pdf/2604.08037v1)

**作者:** Soumya Mazumdar `[一作]` (Gargi Memorial Institute of Technology), Tapas Samanta `[通讯]` (Homi Bhabha National Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种隐私友好的联邦学习框架PrivFedTalk，用于个性化对话头像生成，核心在于共享条件潜在扩散模型骨干与客户端本地轻量化LoRA身份适配器的联邦训练。

**💡 创新点**

创新点包括：
1) 采用身份稳定联邦聚合（ISFA）根据本地身份一致性与时序稳定性给出加权更新；
2) 引入时间去噪一致性（TDC）正则化，抑制帧间漂移；
3) 结合安全聚合、差分隐私与LoRA适配器，实现通信开销小且更新安全；
4) 在联邦环境下首次针对对话头像生成进行系统性实验与基准对比。

**🔧 技术方法**

技术手段包括：
- 条件潜在扩散模型（Latent Diffusion）
- LoRA低秩适配器实现参数高效微调
- 时序去噪一致性正则
- 身份稳定聚合权重计算
- 安全聚合 + 客户端级差分隐私
- 多GPU并行、低内存定制化实现

**📊 数据集**

主要使用LRS3语音-视频数据集；部分实验中将HDTF数据与LRS3混合以增加客户端异质性。

**📈 对比分析**

在与FedAvg、FedProx以及中心化扩散模型的基准对比中，PrivFedTalk在PSNR、SSIM、LPIPS、FID、身份相似度和时间抖动等指标上与基准差距极小，表明其在隐私约束下保持了与中心化模型相当的生成质量；但未显著优于基准，凸显需进一步验证。

**⚠️ 局限性**

局限性包括：
- 未完成完整的差分隐私账户报告，隐私‑效能定量尚不明确；
- 生成质量的定性评估仍在验证中，缺乏公开的高质量视觉对比；
- 仅在LRS3（和有限HDTF）上验证，泛化到更大多样化数据集待进一步实验；
- 量化通信成本与训练效率的系统报告尚未完善；
- 目前对抗性攻击或隐私泄露的安全性实验尚缺失。

---

## 362. Tensor-Augmented Convolutional Neural Networks: Enhancing Expressivity with Generic Tensor Kernels

**arXiv ID:** 2604.08072 | [PDF](https://arxiv.org/pdf/2604.08072v1)

**作者:** Chia-Wei Hsing `[一作]` (blueqat Inc.), Wei-Lin Tu `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Tensor‑augmented CNN（TACNN），将卷积核替换为可表示任意量子叠加态的高阶张量，并在 Fashion‑MNIST 上实现了高精度的图像分类。

**💡 创新点**

创新点在于把每个卷积核视为完整的量子态，使得单个卷积核即可捕获高阶特征关联，从而在保持网络浅层的同时获得与深层 CNN 相当甚至更优的性能，并显著提高参数利用效率。

**🔧 技术方法**

采用量子启发式张量网络（generic tensors）嵌入卷积核，配合多层归一化+Sigmoid非线性、Adam 优化器以及 PyTorch 框架实现训练。

**📊 数据集**

使用 Fashion‑MNIST 数据集（28×28 灰度图，训练 60k，测试 10k）。

**📈 对比分析**

通过与标准 CNN、VGG‑16、GoogLeNet 以及多种张量网络模型在相同测试集上对比；单层 TACNN 达到 93.1% 以上，双层 TACNN 达到 93.7%，与 VGG‑16（93.5%）和 GoogLeNet（93.7%）性能相当，但参数量更少。

**⚠️ 局限性**

局限性包括：对张量尺寸的计算资源仍有限；实验仅涵盖 Fashion‑MNIST，缺乏在更大规模或更复杂任务上的验证；优化方法仍采用通用梯度下降，尚未开发专门针对张量网络的高效训练策略。

---

## 363. Guaranteeing Knowledge Integration with Joint Decoding for Retrieval-Augmented Generation

**arXiv ID:** 2604.08046 | [PDF](https://arxiv.org/pdf/2604.08046v1)

**作者:** Zhengyi Zhao `[一作]` (Chinese University of Hong Kong), Xian Wu `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过显式分离内部推理与检索证据，并在生成阶段采用联合解码，提升了RAG系统的知识整合效果。

**💡 创新点**

创新点在于使用对比DPO训练生成“Refer-Answer”以抑制内部知识的幻觉，并采用分段级融合的联合解码机制，将推理逻辑与事实依据动态对齐。

**🔧 技术方法**

采用的技术包括RAG、Contrastive Direct Preference Optimization (DPO)、长度与事实性约束、语义分段解码以及基于隐藏状态的动态融合。

**📊 数据集**

实验使用五个知识密集型问答数据集：Natural Questions、TruthfulQA、Wizard of Wikipedia、HotpotQA 与 ELI5。

**📈 对比分析**

与标准RAG、SelfRAG、RQ‑RAG、SOLAR、DA‑RAG、FLARE、DRAGIN、P‑RAG等基线相比，GuarantRAG在准确率上提升高达12.1%，幻觉率降低16.3%，在所有评测指标上均位居榜首。

**⚠️ 局限性**

主要局限包括：相较传统RAG增加了推理延迟与计算开销；对检索质量高度依赖；对专业领域知识的泛化能力有限；DPO训练需要精细的超参数调优。

---

## 364. "Why This Avoidance Maneuver?" Contrastive Explanations in Human-Supervised Maritime Autonomous Navigation

**arXiv ID:** 2604.08032 | [PDF](https://arxiv.org/pdf/2604.08032v1)

**作者:** Joel Jose `[一作]` (Norwegian University of Science and Technology), Erlend M. Coates `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出并实现了基于对比解释的海事碰撞规避系统支持框架，旨在提高监督者对自动避碰决策的理解与信任。

**💡 创新点**

创新点在于：①将对比解释（contrastive explanations）应用于多目标碰撞规避规划；②设计了针对导航员的可视化界面；③结合“事件触发+提前模拟”的解释生成时机；④在真实船舶模型上进行体验式用户研究。

**🔧 技术方法**

技术包括：SB-MPC（Simulation-based Model Predictive Control）碰撞规避规划器、成本函数分解与对比解释生成算法、基于地图与航路图的可视化接口，以及基于情景触发的解释触发机制。

**📊 数据集**

使用的“数据集”是自行构造的交通场景（单船与多船、不同相遇类型）以及 Telemetron RBB 8 m 船舶动力学模型；实验数据来自模拟框架和四名具备 STCW 资质的船长级船员的反馈。

**📈 对比分析**

与传统的透明度展示（仅提供轨迹与成本）相比，对比解释在复杂多船场景中显著提升了监督者的决策满意度与对系统目标的理解，但在简单场景下会增加认知负荷；性能主要通过定量问卷评分和定性访谈反馈评估，未与其他解释方法做系统对比。

**⚠️ 局限性**

局限性包括：样本量极小（仅4名参与者）；可能受到新颖性效应影响；对比解释在简单场景中导致额外工作量；界面缺乏常用航海工具（如 ECDIS/ARPA）集成；未在高保真海事模拟器或真实作业环境中进行验证。

---

## 365. A Comparative Study of Semantic Log Representations for Software Log-based Anomaly Detection

**arXiv ID:** 2604.08028 | [PDF](https://arxiv.org/pdf/2604.08028v1)

**作者:** Yuqing Wang `[一作]` (University of Helsinki), Mika V. Mäntylä `[通讯]` (University of Helsinki)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对软件日志异常检测，系统评估了多种语义日志表示方法，并提出了兼顾效率与效果的 QTyBERT。

**💡 创新点**

创新点在于将轻量化 BERT 与跨系统嵌入增强相结合，既保持 BERT 表示能力，又显著加速嵌入生成，达到近乎 BERT 的效果。

**🔧 技术方法**

使用技术包括 TinyBERT 量化、系统特定量化、CroSysEh 低秩映射、以及多种 RNN/CNN/Transformer 等深度学习模型。

**📊 数据集**

使用的数据集为公开的 BGL、Thunderbird、Spirit 三大超级计算机日志数据。

**📈 对比分析**

通过对比静态词嵌入、BERT 以及 QTyBERT，在 CPU 单核/多核环境下，QTyBERT 在检测 F1 评分与嵌入生成时间上均超过静态方法且接近 BERT，且生成时间降低约 94%。

**⚠️ 局限性**

局限性包括对系统日志分布的校准样本敏感、仅在三大系统上验证，且跨系统增强的泛化能力需进一步评估。

---

## 366. Evaluating Counterfactual Explanation Methods on Incomplete Inputs

**arXiv ID:** 2604.08004 | [PDF](https://arxiv.org/pdf/2604.08004v1)

**作者:** Francesco Leofante `[一作]` (Imperial College London), Mustafa Yalçıner `[通讯]` (TU Dortmund University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对现有十种生成逆因解释（CX）的方法在缺失输入下的性能进行了系统评估，并比较了鲁棒与非鲁棒方法的有效性、成本与可行性。

**💡 创新点**

提出将逆因解释的鲁棒性与缺失值问题结合，发现鲁棒方法在缺失输入时能更好保持有效性，但仍无法普遍获得可靠解释，突显需要能直接处理输入不确定性的全新方法。

**🔧 技术方法**

使用了十种现有CX生成算法（如BinaryLinearSearch、MILO、GradientBased、NearestNeighbor、ARMIN等），并结合三种常用缺失值填补技术（MICE、k‑NN、均值填补）进行实验；评估指标包括逆因有效性、回归有效性、成本与可行性（LOF）。

**📊 数据集**

在四个表格数据集上进行实验：WineQuality、Diabetes、Concrete、Combined Cycle Power Plant（CCPP），每个数据集均包含4–11个连续特征，使用min–max归一化并二值化为二分类。

**📈 对比分析**

实验显示鲁棒方法（如DeltaRobust、RobustCE、RobustWeightChange等）在逆因有效性上显著优于非鲁棒方法（例如BinaryLinearSearch、MILO、GradientBased），但总体有效率仍低于80%；成本与可行性两者在鲁棒/非鲁棒间差异不大；ARMIN（专为缺失输入设计）表现最优，说明填补策略对性能影响显著。

**⚠️ 局限性**

主要限制包括：1）所有方法在缺失输入下仍未能提供普适有效的逆因解释；2）梯度基方法易陷入局部最优，尤其在CCPP数据集上表现差；3）目前仅依赖填补技术，未能直接建模输入不确定性；4）实验仅在四个表格数据集上进行，缺乏对更复杂或高维数据的验证。

---

## 367. Benchmarking Deep Learning for Future Liver Remnant Segmentation in Colorectal Liver Metastasis

**arXiv ID:** 2604.07999 | [PDF](https://arxiv.org/pdf/2604.07999v1)

**作者:** Anthony T. Wu `[一作]`, Xiaohui Xie `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对公开的CRLM-CT-Seg 197份CT扫描进行人工精细标注，得到完整、可验证的肝脏、FLR和CRLM分割数据；在此数据上，首次比较了级联（Liver→CRLM→FLR）与端到端（FLR）三阶段的三种3D网络（nnU-Net、SwinUNETR、STU-Net）的FLR预测性能。

**💡 创新点**

创新点包括①首次提供完全手工校正、公开可验证的CRLM-CT-Seg扩展数据集；②首次在FLR预测任务中系统比较级联与端到端策略，并给出基准；③验证预训练STU-Net在此任务中的优势和对级联误差的鲁棒性。

**🔧 技术方法**

使用的技术为3D全卷积网络nnU-Net、SwinUNETR和预训练STU-Net；级联分为Liver→CRLM→FLR三阶段与单阶段端到端FLR；采用5折交叉验证、logits集成以及Dice、Precision、Recall三指标评估。

**📊 数据集**

使用的数据集为197份公开CRLM-CT-Seg CT扫描，人工精细标注后划分157份训练/验证集与40份测试集。

**📈 对比分析**

通过Dice、Precision、Recall三指标对比，级联nnU-Net在测试集上Dice最高0.767；STU-Net在CRLM分割上表现更稳健（0.620→0.594），端到端略逊；总体而言级联略优于端到端。

**⚠️ 局限性**

局限性在于：①标注仅来自单机构医生，缺乏多中心共识；②CRLM分割仍是性能瓶颈；③预训练模型受限于预训练数据集，可能影响泛化。

---

## 368. Search Changes Consumers' Minds: How Recognizing Gaps Drives Sustainable Choices

**arXiv ID:** 2604.08079 | [PDF](https://arxiv.org/pdf/2604.08079v1)

**作者:** Frans van der Sluis `[一作]` (University of Copenhagen), Leif Azzopardi `[通讯]` (University of Strathclyde)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室环境中，让308名参与者在进行一次针对随机分配的伦理维度的在线搜索后，测量其对该维度的事前重要性评估、搜索行为（查询数、打开的URL数、搜索时长）以及搜索后的重要性变化，并结合主观搜索过程评价；

**💡 创新点**

发现搜索过程中的“知识缺口认知”和“信息易懂性”两个主观维度比单纯的搜索行为更能解释重要性变化，并指出低伦理意图者更易受到认知影响而改变重要性评估；

**🔧 技术方法**

使用线性回归、探索性因子分析（EFA）、K‑means聚类与主成分分析（PCA）等统计与机器学习方法对问卷与行为数据进行分析；

**📊 数据集**

采用了308名通过线上平台招募的参与者的实验数据，包括搜索日志、事前后重要性评分以及主观问卷回答；

**📈 对比分析**

通过比较搜索行为与主观搜索维度对重要性变化的解释能力，发现主观维度（认知与易懂性）对重要性变化的影响显著（β=0.23/0.17，p<0.001），而单纯的搜索行为无显著作用，显示主观维度模型性能更好；

**⚠️ 局限性**

局限性在于仅测量了自我报告的意图与重要性变化，未观察实际购买行为，样本聚焦单一产品情境，且可能存在需求偏差与一般化性不足。

---

## 369. DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather

**arXiv ID:** 2604.08074 | [PDF](https://arxiv.org/pdf/2604.08074v1)

**作者:** Christof Leitgeb `[一作]` (Infineon Technologies AG), Daniel Watzenig `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了 DinoRADE，一个基于雷达-摄像头融合的 3D 目标检测框架，能够在所有天气条件下保持稳定性能，并特别关注对 VRU（弱势路人）的检测。

**💡 创新点**

创新点包括：1）基于雷达 3D 投影的加权查询提升模块，利用仰角信息重新分配特征；2）自适应门控融合机制，动态决定是否使用摄像头特征；3）利用 DINOv3 Vision Foundation Model 的高分辨率视觉特征，通过可变形交叉注意力实现跨视角特征聚合；4）针对雷达-摄像头融合的损失设计（改进的焦点+Gaussian‑Wasserstein 损失），提升小目标检测性能。

**🔧 技术方法**

采用了 RADE-Net 雷达后端、DINOv3 ViT‑S/16 视觉前端、可变形注意力（Deformable Attention）、加权特征提升、门控融合、焦点损失+GWD 回归损失、AdamW 优化器和余弦学习率调度等技术。

**📊 数据集**

在 K‑Radar 数据集（v1.1 与 v2.1）上进行实验，该数据集包含 35k 帧、7 种天气（正常、阴天、雾、雨、雪等）以及 5 类目标（Sedan、Bus or Truck、Pedestrian、Bicycle、Motorcycle）。

**📈 对比分析**

与雷达单模、雷达‑摄像头基线以及多类检测方法进行对比。DinoRADE 在 'Sedan' 单类任务中取得 AP_3D 70.8（比前沿方法高 12.1%），总 AP 达 36.99；在雾、雪、雨等恶劣天气下仍保持高精度；对 VRU 的检测也显著提升，如雾天人行道行人 AP 达 51%。

**⚠️ 局限性**

局限性：1）摄像头在雨雾等恶劣天气下被遮挡时对性能提升有限；2）K‑Radar 数据集存在标签缺失，导致模型评估与训练受限；3）仅在 K‑Radar 上验证，尚缺乏跨数据集的泛化验证；4）推理速度未优化，单帧约 190 ms；5）部分少数类样本不足，导致该类性能波动。

---

## 370. AtlasOCR: Building the First Open-Source Darija OCR Model with Vision Language Models

**arXiv ID:** 2604.08070 | [PDF](https://arxiv.org/pdf/2604.08070v1)

**作者:** Imane Momayiz `[一作]`, Haitame Bouanane `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了AtlasOCR——首个面向摩洛哥阿拉伯语方言达里贾的开源OCR模型，并通过Vision Language Model微调实现文本识别；

**💡 创新点**

创新点在于首次专门针对达里贾设计的OCR模型，结合合成+真实数据、参数高效微调（QLoRA+Unsloth）以及公开的AtlasOCRBench评测框架；

**🔧 技术方法**

采用Qwen2.5‑VL 3B视觉语言模型，利用QLoRA量化低秩适配器、Unsloth加速训练、OCRSmith合成工具、Gemini 2.0 Flash伪标签和Argilla人工校正；

**📊 数据集**

使用86%合成数据（OCRSmith）+14%真实扫描、社交媒体、教育材料、食谱等共计30,092张图像、约10.7M单词的混合数据集；

**📈 对比分析**

通过CER/WER指标在AtlasOCRBench和KITAB-Bench上与其他开源模型对比，AtlasOCR在达里贾OCR上取得最优CER，并在标准阿拉伯语Bench上与12B规模模型竞争，表现优异；

**⚠️ 局限性**

局限包括对阿拉伯语重音符号识别不足、复杂排版和艺术化布局处理性能下降，以及训练数据对特定领域的偏倚。

---

## 371. Brain3D: EEG-to-3D Decoding of Visual Representations via Multimodal Reasoning

**arXiv ID:** 2604.08068 | [PDF](https://arxiv.org/pdf/2604.08068v1)

**作者:** Emanuele Balloni `[一作]` (Università Politecnica delle Marche), Emiliano Santarnecchi `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Brain3D，一个基于多模态推理的 EEG-to-3D 解码架构，将脑电信号先解码成图像，再通过 LLM 生成结构化 3D 描述，最后利用扩散模型和单图像到 3D 的网络生成三维网格。

**💡 创新点**

创新点在于把 EEG-to-3D 任务拆分为：1）EEG 到图像的 diffusion 解码；2）使用多模态大语言模型进行几何感知的语义推理；3）基于结构化描述的 diffusion 生成与单图像 3D 生成相结合的两步生成策略，从而避免直接映射导致的几何失真，提升可扩展性和几何一致性。

**🔧 技术方法**

核心技术包括：EEG 图像解码的跨模态 diffusion 模型；LLaMA 3.2 Vision 90B 进行结构化文本描述；Stable Diffusion 3.5 Medium 生成清晰的 2D 视觉前置；Microsoft TRELLIS 的单图像到 3D 的前向生成网络；以及 CLIPScore、LPIPS、FID、Top‑k 语义检索等评估指标。

**📊 数据集**

使用 EEGCVPR40 数据集：128 通道 EEG 与 2,000 张 ImageNet 物体类别图像（共 40 类）相匹配，分 80/10/10 的训练/验证/测试比例。

**📈 对比分析**

通过在六个标准视角渲染 3D 网格并与原始图像比较，使用 CLIPScore、LPIPS、IS、FID、Top‑k Accuracy 进行评估。实验显示，基于 GWIT 后端的 Brain3D 在 10‑way Top‑1 达到 85.4%，CLIPScore 0.648，Fidelity 方面 FID 降至 153，显著优于直接将 EEG 图像解码后直接生成 3D 的基线；消融实验表明语义推理与生成模块能进一步提升语义一致性与几何质量。

**⚠️ 局限性**

主要限制：EEG‑to‑Image 解码的精度仍是瓶颈；当前仅处理单一静态物体，无法覆盖多物体或动态场景；模型依赖于强大的预训练扩散与 LLM，推理成本较高，且对实时应用的适配性待提升。

---

## 372. Multimodal Latent Reasoning via Predictive Embeddings

**arXiv ID:** 2604.08065 | [PDF](https://arxiv.org/pdf/2604.08065v1)

**作者:** Ashutosh Adhikari `[一作]` (University of Edimburgh), Mirella Lapata `[通讯]` (University of Edimburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Pearl 框架，训练视觉‑语言模型在不调用外部工具的情况下，通过预测工具使用轨迹的潜在嵌入来提升多模态推理。

**💡 创新点**

创新点在于采用 JEPA 风格的预测嵌入对齐，直接学习轨迹级潜在表示而非逐步生成 latent token，支持多步工具调用且消除了推理时的工具调用与 token 切换开销。

**🔧 技术方法**

技术手段包括双视图编码（输入视图与轨迹视图）、轻量级预测器（特殊 token）、JEPA 对齐损失、NextLatent 预测正则化、联合 VLM 生成目标，以及 LoRA 微调。

**📊 数据集**

使用的训练数据集包括 LVR（单工具单步）、ThinkMorph（多工具单步多类型）和 PixelReasoner（多步单工具），以及多种视觉问答基准（VQA、BLINK、MMVP 等）。

**📈 对比分析**

通过与 LVR、CoVT、ThinkMorph 基线、PixelReasoner 以及零样本 VLM 进行比较，Pearl 在所有三种训练场景中匹配或优于重建式潜在推理和 SFT，并且在保持相同推理流程的前提下显著提升性能。

**⚠️ 局限性**

局限性包括仅覆盖单工具单步或多步单工具场景，未实现多工具多步推理；训练时需要两次前向传递导致计算成本翻倍；对小模型在多工具多步设置下的表现仍有限。

---

## 373. PriPG-RL: Privileged Planner-Guided Reinforcement Learning for Partially Observable Systems with Anytime-Feasible MPC

**arXiv ID:** 2604.08036 | [PDF](https://arxiv.org/pdf/2604.08036v1)

**作者:** Mohsen Amiri `[一作]` (Stockholm University), Mehdi Hosseinzadeh `[通讯]` (Washington State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在部分可观测环境下，提出PriPG-RL框架，利用训练期间可用的任何时间可行MPC规划器对学习代理进行指导，训练出在部署时仅依赖有限观测的高效反应式策略，并在仿真和真实Unitree Go2四足机器人上验证其可行性。

**💡 创新点**

创新点包括①将REAP-based任何时间可行MPC作为规划器，提供可随计算时间调节的可行解；②设计P2P‑SAC，将规划器信息转化为软演员‑评论家目标，包含双缓冲、三阶段成熟度调度、logit空间模仿锚点和优势门控，显著缓解状态别名带来的方差；③理论证明优势门控下梯度不受不可约混淆方差影响，确保学习过程的稳定性。

**🔧 技术方法**

技术手段：Soft Actor‑Critic算法、REAP-based MPC、双重经验缓冲、确定性成熟度调度、logit空间模仿损失、优势基Sigmoid门控、温度自适应、NVIDIA Isaac Lab仿真、OptiTrack视觉定位、Unitree Go2硬件部署。

**📊 数据集**

数据集与环境：自定义障碍环境（4.1×5.6 m 区域，6个圆柱障碍），在NVIDIA Isaac Lab中生成仿真数据，并使用OptiTrack系统收集真实机器人在相同障碍场景下的状态数据。

**📈 对比分析**

与基线算法（标准SAC、PPO、加速SAC）比较：P2P‑SAC在1 M步内实现100 %成功率，Accelerated SAC仅40 %；最终性能上，P2P‑SAC成功率100 %、碰撞率0 %、路径最优度1.06，优于REAP（1.10）和SAC（35 %成功）。运行时间和平均速度略高，但总体学习效率和最终性能均显著提升。

**⚠️ 局限性**

局限性：目前仅针对反应式策略，无法利用历史信息解决更复杂的时序不确定性；信息不对称导致无法完全消除状态别名，部分情况下仍需规划器指导；规划器的计算开销在资源受限的场景中仍是瓶颈；在更动态或非线性模型的环境中泛化能力尚未充分验证。

---

## 374. IoT-Brain: Grounding LLMs for Semantic-Spatial Sensor Scheduling

**arXiv ID:** 2604.08033 | [PDF](https://arxiv.org/pdf/2604.08033v1)

**作者:** Zhaomeng Zhou `[一作]` (University of Science and Technology of China), Jinke Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 IoT-Brain 系统，解决大规模摄像头网络中的语义‑空间传感器调度（S³）问题。

**💡 创新点**

提出“verify‑before‑commit” neurosymbolic 的 Spatial Trajectory Graph（STG）框架，将 LLM 的语义推理与可验证拓扑图相结合，显著提升调度可靠性与资源效率。

**🔧 技术方法**

结合大语言模型、工具调用与编程、图优化与检验循环、共享记忆缓存、可视化语言模型以及目标检测/识别技术，实现动态激活与主动感知。

**📊 数据集**

构建 TopoSense‑Bench 基准，包含 2,510 台摄像头与 5,250 条自然语言查询，覆盖校园规模多建筑室内外场景；同时在真实校园测试床上进行验证。

**📈 对比分析**

与 Hierarchical、Reactive、Backtracking 等三种 LLM 规划范式对比，IoT‑Brain 在最复杂任务上任务成功率提升 37.6%，速度近 2 倍，提示 token 消耗减少 6.6 倍；在真实部署中实现 49.84% TCR，使用 4.1 倍更少带宽。

**⚠️ 局限性**

受限于语义歧义、视觉识别局限及动态环境中传感器状态变化，且对 LLM 质量仍有一定依赖，需进一步降低对外部 VLM 的需求。

---

## 375. AgiPIX: Bridging Simulation and Reality in Indoor Aerial Inspection

**arXiv ID:** 2604.08009 | [PDF](https://arxiv.org/pdf/2604.08009v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 376. Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles

**arXiv ID:** 2604.08031 | [PDF](https://arxiv.org/pdf/2604.08031v1)

**作者:** Jiawei Liu `[一作]` (Jilin University), Qing Guo `[通讯]` (NKIARI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于大型语言模型（LLM）的调度中心框架，用于将乘客的开放式自然语言指令转换为可执行的驾驶行为脚本，并通过调度多个基于模型预测控制（MPC）的运动规划器实现指令。

**💡 创新点**

创新点包括：①将LLM的高层语义推理与低层控制解耦，仅用LLM一次性生成脚本；②采用异步触发器实现实时场景适配的行为切换；③构建高保真、闭环评估基准POINT，填补开放式指令评估空白；④提供透明可追溯的决策链，满足安全合规要求。

**🔧 技术方法**

核心技术包括：大型语言模型（如DeepSeek V3、Qwen-2.5-72B）、脚本生成与异步触发机制、MPC运动规划器、线性二次调节器（LQR）控制器、基于nuPlan的混合仿真环境。

**📊 数据集**

使用数据集：nuPlan混合仿真平台（基于1300小时真实道路数据）与自构造的1050条开放式指令–场景对，指令由真实语料与LLM生成相结合并人工筛选。

**📈 对比分析**

与多种基准方法（LogReplay、IDM、DiLu+、PDM、Diffusion-ES、DiLu++等）在任务完成率、安全性、碰撞避免、TTC、可行驶区域、限速符合度等指标上对比。实验结果显示：指令实现率达0.84，比基准提升64%–200%；安全性和规则合规性与专用AD方法持平；在碰撞避免、TTC、可行驶区域等安全指标上均优于或等同于对手。

**⚠️ 局限性**

局限性包括：①缺乏视觉输入整合，LLM只能基于文本场景描述；②nuPlan未提供视景渲染，限制了VLA方法的闭环评估；③异步触发器的表达能力有限，需进一步学习或自适应重调度；④依赖预定义的有限运动规划器库，扩展指令空间仍需手工集成；⑤LLM生成的脚本若出现歧义或幻觉仍可能导致误判。

---

## 377. Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation

**arXiv ID:** 2604.08011 | [PDF](https://arxiv.org/pdf/2604.08011v1)

**作者:** Yantao Yu `[一作]` (Alibaba International Digital Commercial Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commercial Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究推荐系统在规模化过程中的稀疏性瓶颈，提出SSR框架，在层级上先显式过滤维度，再进行密集融合，从而提高大模型在稀疏特征数据上的表现和可扩展性。

**💡 创新点**

①将稀疏性从模型学习的隐式结果转化为架构设计目标；②采用多视角“filter‑then‑fuse”结构，先在每个视角做维度级显式稀疏，再聚合；③提出两种稀疏实现：静态随机过滤（硬性稀疏）和迭代竞争稀疏（ICS，基于全局抑制的可微分稀疏化机制）；④证明该设计能突破密集模型在工业规模数据上的饱和瓶颈。

**🔧 技术方法**

静态随机过滤、迭代竞争稀疏 (ICS)、多视角稀疏层、Block‑diagonal dense fusion、LayerNorm、GELU 激活、FLOPs/参数度量、TensorFlow 实现、Adam 优化、A/B 在线测试。

**📊 数据集**

公共数据集：Avazu、Criteo、Alibaba；工业数据集：AliExpress 亿级 CTR 日志（约1 B条记录，300+特征字段）。

**📈 对比分析**

与 DeepFM、DCNv2、AutoInt、MMOE、AutoFIS、AFN、Wukong、RankMixer 等现有方法对比；在工业数据上 SSR‑D 取得点击 AUC 0.6667、支付 AUC 0.8194，超过 RankMixer（0.6621/0.8122）且参数/ FLOPs 相近；在公共数据上 SSR‑D 对 RankMixer 的 AUC 提升 0.63%–0.43%，参数量更少或相同；在线 A/B 测试中，SSR‑D 相较 RankMixer 提升 CTR +2.1%，订单 +3.2%，GMV +3.5%，平均延迟 26 ms 与基线相当。

**⚠️ 局限性**

缺点与局限：需要对超参数（T、α_t、γ）进行调优，ICS 的收敛性和鲁棒性在极稀疏场景下尚未充分验证；多视角独立性假设可能导致信息冗余或丢失；模型仍需在推理时对稀疏化过程做硬件友好实现；可解释性、训练稳定性以及在其他业务场景（如视频推荐、搜索排序）的推广仍待进一步研究。

---

## 378. Preference Redirection via Attention Concentration: An Attack on Computer Use Agents

**arXiv ID:** 2604.08005 | [PDF](https://arxiv.org/pdf/2604.08005v1)

**作者:** Dominik Seip `[一作]` (University of Tübingen), Matthias Hein `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种对多模态大型视觉语言模型的攻击方法，利用对视觉注意力的集中来重定CUA的选择偏好。

**💡 创新点**

创新点在于通过操控内部注意力分布而非直接修改输出，实现在仅对图像极小范围做不可察觉扰动的情况下诱导代理选择攻击目标。

**🔧 技术方法**

攻击使用白盒梯度优化（Auto-PGD + PCGrad）在模型多头注意力层聚焦注意力，并挑选关键输出词与活跃注意力头。

**📊 数据集**

实验数据来自仿真电子商务网站的服装商品图片（5个网格位置），并使用Qwen3-VL、GLM4.6V、Kimi-VL、EvoCUA等公开权重模型。

**📈 对比分析**

与交叉熵、CLIP对齐、文本覆盖等基线相比，所提出的PRAC方法在不同提示与模型上均实现约82%的成功率，显著高于最优基线约65%。

**⚠️ 局限性**

局限性包括需模型白盒访问、仅针对单个产品图像、攻击效果受扰动阈值限制，且在完全黑盒场景下效果尚未验证。

---

## 379. The ecosystem of machine learning competitions: Platforms, participants, and their impact on AI development

**arXiv ID:** 2604.08001 | [PDF](https://arxiv.org/pdf/2604.08001v1)

**作者:** Ioannis Nasios `[一作]` `[通讯]` (Nodalpoint Systems), Ioannis Nasios (Nodalpoint Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统性分析了主流机器学习竞赛平台（Kaggle、Zindi、Codabench 等）的生态结构、奖池、参与者构成与竞赛质量，并探讨其对人才培养、科研创新与工业应用的影响。

**💡 创新点**

创新点在于将平台层面数据、Kaggle 元数据挖掘与文献计量相结合，构建统一的指标体系评估竞赛生态，并量化不同平台对 AI 发展与产业合作的贡献。

**🔧 技术方法**

采用统计分析、聚类、回归模型、Kaggle API 数据抓取、公开竞赛数据集和文献检索工具（如 Google Scholar、Scopus）等技术。

**📊 数据集**

主要使用 Kaggle 公共数据集、Meta Kaggle 用户统计、各平台竞赛数据（参赛人数、奖金、排行榜）以及相关公开研究数据集。

**📈 对比分析**

通过排行榜分数、奖金分布、参赛人数、奖项等级（Grandmaster、Master 等）等指标进行对比，发现 Kaggle 虽主导市场但每团队奖单价低，Drivendata、Thinkonward 等平台奖单价高且质量更专注；整体竞赛质量与参与者水平呈显著差异。

**⚠️ 局限性**

局限性包括公开数据不完整、竞赛结果对泛化性的验证不足、评测指标单一（多聚焦准确率）、平台差异导致可比性受限，以及高性能竞赛对资源与环境成本的隐形门槛。

---

## 380. HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation

**arXiv ID:** 2604.07993 | [PDF](https://arxiv.org/pdf/2604.07993v1)

**作者:** Shuanghao Bai `[一作]` (Beijing Innovation Center of Humanoid Robotics), Badong Chen `[通讯]` (Xi'an Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 HEX 框架，实现全身协调的视觉‑语言‑动作 (VLA) 控制，解决人形机器人高自由度操控难题。

**💡 创新点**

① 全身统一的 humanoid‑aligned 状态表示；② Mixture‑of‑Experts 的 Unified Proprioceptive Predictor 对全身运动进行预测；③ 轻量历史缓存与门控融合动作专家，将视觉‑语言信息与未来状态动态自适应融合。

**🔧 技术方法**

采用 Qwen3‑VL‑2B‑Instruct 视觉语言模型、Transformer‑MoE 架构、时间序列预测与流匹配 (flow‑matching) 动作生成、双跨注意力与门控融合、以及 RL 训练的低层全身控制器。

**📊 数据集**

使用约 12M 帧跨 7 种人形机器人的数据集，包含 Tienkung 2.0/3.0、Tienyi、Unitree G1/H1、AgiBot、Leju 等真实机器人轨迹及仿真/视频数据，构成跨体型、多姿态的多模态数据集。

**📈 对比分析**

与 ACT、SwitchVLA、GR00T N1.5、π_0.5 等基线在相同低层控制下进行对比。HEX 在见过场景、长时程任务和分布迁移任务中，任务成功率提升约 15–30%，动作平滑度更高，误差传播更小，尤其在快速反应与长时程情境中表现最为突出。

**⚠️ 局限性**

需要大量跨体型预训练数据；低层控制仍依赖离线 RL 训练；对极端动态环境或大遮挡的鲁棒性有限；模型规模较大，推理时仍存在一定延迟。

---

## 381. AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding

**arXiv ID:** 2604.08077 | [PDF](https://arxiv.org/pdf/2604.08077v1)

**作者:** Handong Li `[一作]` (University of Chinese Academy of Sciences), Jing Liu `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AdaSpark，一种针对长视频理解的自适应稀疏框架；

**💡 创新点**

创新点在于将视频划分为三维时空立方体，并通过熵驱动的 Top‑p 机制同时对立方体和立方体内的 token 进行自适应稀疏选择，既保留细粒度信息又维持长时序依赖；

**🔧 技术方法**

采用的技术包括 3D 视频立方体分割、Adaptive Cube‑Selective Attention (AdaS‑Attn)、Adaptive Token‑Selective FFN (AdaS‑FFN)、熵‑基 Top‑p 自适应选择以及 Mean Compensation 近似；

**📊 数据集**

使用的主要数据集包括 llava‑video‑178k、DideMo 77k、ActivityNet Captions、Video Needle in a Haystack、LongVideoBench、MLVU、VideoMME、LVBench、MVBench、VSIBench、CharadesSTA 等长短视频与视频‑文本评测集；

**📈 对比分析**

与多种稀疏方法（如 MoBA、Local Attention、Video‑XL 等）以及基线 dense Video‑LLM 进行比较，AdaSpark 在 FLOPs 上最多可减少 57%，且在 5 大类基准（额外长视频、长视频、短视频、空间推理、视频定位）均取得领先或同等水平的性能；

**⚠️ 局限性**

局限性包括对立方体划分大小和 Top‑p 阈值的依赖，且在极高时序敏感任务中可能仍需进一步精细化稀疏策略；

---

## 382. Guiding a Diffusion Model by Swapping Its Tokens

**arXiv ID:** 2604.08048 | [PDF](https://arxiv.org/pdf/2604.08048v1)

**作者:** Weijia Zhang `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Self‑Swap Guidance（SSG），一种在扩散模型推理过程中通过在token空间进行最语义不相似token对的自适应交换来产生弱化预测，从而实现无条件与有条件生成的指导。

**💡 创新点**

创新点在于：①在token级别而非全局噪声或注意力空间进行扰动，②采用“最不相似”token对的对抗性交换提供可控、细粒度扰动，③可与经典CFG并用，进一步提升图像质量与文本一致性。

**🔧 技术方法**

技术核心包括：扩散模型推理（Stable Diffusion v1.5 / vXL）、token自交换操作、引导尺度控制、并行正向与扰动分支的组合。

**📊 数据集**

使用数据集：MS‑COCO 2014/2017 验证集、ImageNet 验证集；在这些数据上进行无条件与有条件图像生成实验。

**📈 对比分析**

对比方法包括：SAG、PAG、SEG、CFG 等。实验显示在MS‑COCO 2014/2017 和 ImageNet 上，SSG 在 FID、AES、PickScore、ImageReward 等指标上均优于其他无条件引导方法，并在与 CFG 组合时进一步提升图像质量与文本对齐度。

**⚠️ 局限性**

局限性：①仅在 SD1.5 / SDXL 上验证，未对更大/不同架构扩散模型评估；②需调节交换比例与引导尺度，过高或过低可能导致细节丢失或噪声；③对极端高分辨率生成的鲁棒性尚未彻底验证。

---

## 383. 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience

**arXiv ID:** 2604.08042 | [PDF](https://arxiv.org/pdf/2604.08042v1)

**作者:** Hongcan Xiao `[一作]` (Beijing University Of Posts And Telecommunications), Yonggang Qi `[通讯]` (Beijing University Of Posts And Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一个训练‑free 的 3DrawAgent 框架，利用大型语言模型通过提示生成 3D Bezier 曲线，并通过自我评估和对比式经验优化不断提升生成质量。

**💡 创新点**

创新点包括：①将训练‑free GRPO 改为对比式经验提取，构建无监督的相对评价体系；②将 CLIP 多视图感知奖励与 LLM 细粒度评估相结合，形成混合奖励；③设计专门的 3D 草图语言与解析/渲染管线，使 LLM 能在 3D 空间中直接规划和生成曲线；④实现 LLM 的“自我强化”，无需梯度更新即可提升 3D 推理能力。

**🔧 技术方法**

使用的技术主要包括：大型语言模型（DeepSeek‑V3.2、Gemini‑2.5Pro 等）作为空间规划器；CLIP 进行多视图感知奖励；差分渲染器将 3D Bezier 转为 2D 视图；对比式经验提取（Contrastive Knowledge Extraction）与训练‑free GRPO 的组合；将经验信息注入上下文实现黑盒强化提示调优。

**📊 数据集**

实验所用数据集：ModelNet40、QuickDraw、Diff3DS（文本和图像版本），以及自行构造的多样化文字提示集合。

**📈 对比分析**

与 Diff3DS、3Doodle、Dream3DVG 等现有方法在文本‑>3D、图像‑>3D 任务中对比；评价指标为 CLIP‑S_T、CLIP‑S_I、AES；3DrawAgent 在无训练的条件下实现了与训练模型相近甚至更优的性能（CLIP‑S_T 约 0.643‑0.669，AES 约 4.1‑4.17），并在多种形状上表现出更清晰、更连贯的 3D 草图。

**⚠️ 局限性**

局限性包括：①依赖 CLIP 视图感知，难以捕捉极细粒度几何细节；②对比经验需要足够多的候选生成，计算开销随样本数上升；③LLM 的推理可能导致过度优化或结构不完整；④缺乏对纹理、材质或复杂表面细节的控制；⑤在极端复杂或高度抽象的描述下仍可能出现不连贯或误解。

---

## 384. SynQL: A Controllable and Scalable Rule-Based Framework for SQL Workload Synthesis for Performance Benchmarking

**arXiv ID:** 2604.08021 | [PDF](https://arxiv.org/pdf/2604.08021v1)

**作者:** Kahan Mehta `[一作]` (Dhirubhai Ambani University), Amit Mankodi `[通讯]` (Dhirubhai Ambani University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过确定性图遍历和AST组装生成可直接执行的SQL工作负载，解决学习型优化器缺乏真实训练数据的问题。

**💡 创新点**

提出了基于数据库外键图的拓扑采样与参数化控制的两阶段框架，消除了LLM生成的模式崩塌和架构幻觉，并实现了高拓扑熵（H≈1.53比特）的多样化查询。

**🔧 技术方法**

使用了图遍历权重加权选择、列与聚合随机抽样、谓词生成以及抽象语法树编译技术；核心算法为两阶段构造与参数化配置向量Θ。

**📊 数据集**

在TPC‑H和IMDb两大典型数据库上构造了20,000条查询，用以评估多样性与成本预测性能。

**📈 对比分析**

与传统固定模板和LLM生成相比，SynQL在拓扑熵、成本预测R²（TPC‑H 0.99，IMDb 0.82）以及子查询无误率和推理毫秒级表现上均优于对比方法。

**⚠️ 局限性**

当前仅支持基础的多表内连接、投影、聚合和范围谓词；不包含子查询、CTE、set操作、外部谓词等复杂语法；仅针对PostgreSQL的计划特征，跨引擎迁移仍待验证。

---

## 385. Log-based, Business-aware REST API Testing

**arXiv ID:** 2604.08007 | [PDF](https://arxiv.org/pdf/2604.08007v1)

**作者:** Ding Yang `[一作]`, Chunrong Fang `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于历史请求日志的业务感知 REST API 测试方法 LoBREST。

**💡 创新点**

创新点在于利用日志捕获业务约束，结合局部切片、切片增强和业务感知模糊测试三阶段流程。

**🔧 技术方法**

技术包括 LLM 资源抽取、日志局部切片（MLTS/STWS）、切片补全（资源一致性补全）与业务感知与故障触发模糊器。

**📊 数据集**

使用了 17 个真实 REST 服务数据集，其中 10 个来自 RESTgym，7 个 GitLab 子服务及完整 GitLab（1,099 个 API，109 个业务模块）。

**📈 对比分析**

与八种主流工具对比，LoBREST 在所有服务上均获得最高的操作/行覆盖率和最多的缺陷发现，GitLab 上覆盖率提升达 263%/56% 以上，缺陷数翻倍。

**⚠️ 局限性**

局限在于需收集足够完整的历史日志；实验使用人工生成的日志；对极大规模或高频日志旋转的服务效果未知；随机性和工具配置对结果仍有影响。

---

## 386. Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments

**arXiv ID:** 2604.07997 | [PDF](https://arxiv.org/pdf/2604.07997v1)

**作者:** Yun Zhu `[一作]` (Nanjing University of Science and Technology), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于视觉语言模型引导的少样本增量3D检测框架FI3Det，能够在仅有少量新类别样本的情况下实现对新类别的检测，并保持对已知类别的检测性能。

**💡 创新点**

首次将VLM用于基线阶段的未知物体学习，并设计了点级与框级加权以及门控多模态原型印记机制，实现少样本增量学习且显著提升新类别检测效果。

**🔧 技术方法**

使用VLM（如GroundingDINO、分割模型）生成2D掩码与语义特征，结合3D点云特征，采用多模态原型更新与门控融合，基于TR3D等点云检测器。

**📊 数据集**

在ScanNet V2和SUN RGB‑D上构造少样本增量拆分，分别评估批量和序列增量设置。

**📈 对比分析**

与Imprinting、IL‑DETR、SDCOT++、AIC3DOD以及VLM‑vanilla等基线进行对比，FI3Det在新类别mAP上平均提升约17%（ScanNet）和约9%（SUN），并保持或提升基类性能。

**⚠️ 局限性**

仍依赖VLM生成的伪标签，噪声可能影响学习；在极少样本（1-shot）场景下效果相对有限；对已有类别的记忆受限于原型更新策略，存在一定的灾难性遗忘风险。

---

## 387. xDup: Privacy-Preserving Deduplication for Humanitarian Organizations using Fuzzy PSI

**arXiv ID:** 2604.08019 | [PDF](https://arxiv.org/pdf/2604.08019v1)

**作者:** Tim Rausch `[一作]` (CISPA Helmholtz Center for Information Security), Wouter Lueks `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于模糊私有集合交（Fuzzy PSI）的隐私保护重复登记检测系统，专为人道主义组织在危机地区使用生物识别数据的场景设计；

**💡 创新点**

创新点在于将记录嵌入512维哈希空间并提出了全无输入假设、阈值无关的高效 FPSI 协议，整体实现速度比现有方案提升约两位数；

**🔧 技术方法**

核心技术包括 LSH 生成哈希嵌入、SHADE 基于 OT 的距离计算、加密的秘密共享、两台非协同计算节点以及 SilentOT/SoftSpokenOT 等 OT 实现；

**📊 数据集**

使用人工合成的危机环境登记数据（约 131,072 条现有记录、2,048 条新增记录）来评估系统性能；

**📈 对比分析**

与 FLPSI、DA‑PSI、Approx‑PSI、Fmap‑FPSI、PE‑FPSI 等现有 FPSI 协议对比，在线查询仅需约 10 秒，离线查询耗时 178 分钟，整体速度比 FLPSI 快 84 倍，通信量更小；

**⚠️ 局限性**

局限性包括需预先在离线阶段上传秘密共享嵌入，无法防止恶意登记者的重识别攻击，且对高错误率的真实人道主义数据验证不足，未涵盖基于唯一标识符的情况。

---

## 388. On the Global Photometric Alignment for Low-Level Vision

**arXiv ID:** 2604.08172 | [PDF](https://arxiv.org/pdf/2604.08172v1)

**作者:** Mingjia Li `[一作]` (Tianjin University), Xiaojie Guo `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种光照对齐损失（PAL），在低级视觉任务的配对监督中显式消除每对图像间的全局光照不一致，提升梯度对结构恢复的关注；

**💡 创新点**

创新点在于使用闭式最小二乘全局仿射颜色变换对齐，轻量级且可与任何基准网络无缝集成；

**🔧 技术方法**

技术手段包括Ridge回归求解的三通道仿射变换、梯度归零的对齐损失与原始像素误差的加权融合；

**📊 数据集**

实验覆盖六类任务（低光增强、海底增强、去雾、夜间去雾、全景天气恢复、阴影去除），共16个公开数据集（LOLv1/v2、EUVP、RESIDE-SOTS、NHR、ISTD等）；

**📈 对比分析**

与16种基准网络对比，PAL平均提升PSNR约0.45dB、SSIM和LPIPS均显著改善，并在未见分布数据上提升IQA/IAA分数，证明泛化能力增强；

**⚠️ 局限性**

局限在于仅针对全局光照变换有效，无法完全补偿局部色调或空间非均匀变形，且需要配对训练数据来估计统计量。

---

## 389. T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation

**arXiv ID:** 2604.08167 | [PDF](https://arxiv.org/pdf/2604.08167v1)

**作者:** Pranjal Khadka `[一作]` `[通讯]`, Pranjal Khadka

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种时序适配器，将相邻切片上下文注入VLM的视觉token，实现3D医学图像分割；

**💡 创新点**

通过轻量级的时序transformer、空间上下文块与自适应门控，在不改动基底VLM的前提下实现跨切片的语义融合；

**🔧 技术方法**

基于CLIPSeg的视觉-语言模型，加入时序Transformer、空间自注意力以及门控机制，并在有限标注数据上微调；

**📊 数据集**

使用FLARE22进行训练与验证，零样本评估于BTCV、AMOS22 CT，以及跨模态评估于AMOS22 MRI；

**📈 对比分析**

与CLIPSeg基线和3D DynUNet对比，FLARE22平均Dice提升+0.206，BTCV/AMOS22零样本提升+0.210/+0.230，跨模态MRI平均Dice 0.366，显著优于仅CT训练的3D基线；

**⚠️ 局限性**

采用固定5切片窗口，未考虑切片间距；分辨率限制导致小结构性能受限；在薄管状器官（如食管）上可能引入噪声导致性能下降。

---

## 390. A Direct Approach for Handling Contextual Bandits with Latent State Dynamics

**arXiv ID:** 2604.08149 | [PDF](https://arxiv.org/pdf/2604.08149v1)

**作者:** Zhen Li `[一作]` (BNP Paribas), Gilles Stoltz `[通讯]` (Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了带隐马尔可夫动态的线性情境多臂赌博机模型，并提出一种分阶段 LinUCB 策略，利用在线估计的信念进行决策，给出了高概率伪遗憾上界。

**💡 创新点**

创新点：① 直接处理隐藏状态对奖励的线性依赖，突破了以往把奖励线性化为信念的简化；② 通过分阶段更新与 HMM 快速混合性质相结合，得到无模型参数依赖的高概率 T^{3/4}（简化模型）和 T^{7/8}（完整模型）遗憾上界；③ 采用 L^2‑马尔科夫不等式和信念误差控制，克服了传统 LinUCB 无法处理的状态依赖。

**🔧 技术方法**

技术手段：隐马尔可夫模型估计（光谱法+贝叶斯更新）、分阶段 LinUCB 与置信上界、快速混合性质、L^2‑马尔科夫不等式、椭圆势能 lemma 等。

**📊 数据集**

使用的并未给出真实数据集，主要是理论推导；若需实验则可生成合成的 HMM 上下文与奖励。

**📈 对比分析**

比较方法：与以往的简化模型（Belief‑Linear）相比，先前给出 √T 的期望遗憾；本工作给出高概率 T^{3/4}（简化）和 T^{7/8}（完整）上界，且不依赖奖励函数、无期望限制，并且不要求预先知道 HMM 参数；整体性能在理论上优于之前的方法。

**⚠️ 局限性**

局限性：① 需要 HMM 快速混合的假设；② 对完整模型的 T^{7/8} 上界未证明最优，仍可能存在更优算法；③ 信念估计需要额外对齐步骤，计算成本未评估；④ 仅给出伪遗憾上界，实际奖励与伪遗憾之间的关系未进一步量化。

---

## 391. Internal noise in deep neural networks: interplay of depth, neuron number, and noise injection step

**arXiv ID:** 2604.08117 | [PDF](https://arxiv.org/pdf/2604.08117v1)

**作者:** D. A. Maksimov `[一作]`, N. Semenova `[通讯]` (Saratov State University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究内部高斯噪声在深度前馈神经网络中的影响，比较噪声注入于激活函数前后、加性与乘性噪声以及不同网络深度和隐藏层规模下的表现，并验证池化降噪策略的有效性。

**💡 创新点**

首次系统性对比激活函数前后噪声注入对网络鲁棒性的差异，揭示激活函数是有效的非线性噪声滤波器，并量化加性噪声对前向注入的更高抑制效果；同时提出基于池化的噪声抑制方案在任何注入时机均可显著提升性能。

**🔧 技术方法**

使用 TensorFlow/Keras 进行网络训练与评估，采用白噪声模型（加性/乘性）注入，实验涵盖 3/4/5 层网络及 10/20/30/350/250/200 隐藏层规模，利用平均方差分析权重矩阵统计特性。

**📊 数据集**

MNIST 手写数字识别数据集（60000 训练 + 10000 测试）。

**📈 对比分析**

通过在训练后固定权重并在测试阶段注入噪声，绘制准确率与噪声强度 D 的对数曲线。结果显示：加性噪声在前向注入时导致准确率从 94% 降至 31%（D=1），乘性噪声下降较慢；激活函数前注入显著提升准确率（同样噪声下保持 80% 以上）。池化 m=3 在任意注入位置将噪声导致的准确率下降降低约 20–30%，接近无噪声基准。

**⚠️ 局限性**

噪声累积受后续权重矩阵均方值影响，早期层噪声更易放大；对后向注入的噪声，单层池化无法完全抵消因权重矩阵放大导致的噪声；未针对不同硬件实现的实际噪声分布给出统一的泛化方法，且仅验证了白噪声模型。

---

## 392. Complementary Filtering on SO(3) for Attitude Estimation with Scalar Measurements

**arXiv ID:** 2604.08099 | [PDF](https://arxiv.org/pdf/2604.08099v1)

**作者:** Alessandro Melis `[一作]` (I3S, CNRS, Université Côte d'Azur), Tarek Hamel `[通讯]` (I3S, CNRS, Université Côte d'Azur)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4`

**🎯 论文内容**

提出了一种在 SO(3) 上工作的互补滤波观测器，用于仅利用标量（部分向量）测量来估计姿态，并给出了其稳定性分析。

**💡 创新点**

创新点在于：① 将传统互补滤波器的创新项改写为兼容标量测量的形式；② 通过 S† 与 Λ† 的引入，补偿测量方向的各向异性；③ 在至少三条标量测量或两条标量测量的情况下，给出了几乎全局或有限吸引域的稳定性条件，显著降低了对完整向量测量的需求。

**🔧 技术方法**

使用了几何观测理论、SO(3) 上的李群动力学、Lyapunov 能量函数分析、矩阵伪逆和正交投影，以及持续激励（persistence of excitation）概念；同时设计了常数增益的观测器。

**📊 数据集**

使用仿真数据：无人机在不同轨迹（含加速度计、磁力计、气压计、Pitot 管等）下的合成标量测量，未使用真实传感器数据集。

**📈 对比分析**

与传统全向量互补滤波器进行对比，仿真表明：三标量测量时收敛速度略慢，受激励条件影响明显；六标量测量或完整向量测量时收敛效果相近；在两标量配置下，满足提出的条件可实现收敛，性能与传统方法相当但要求更低。

**⚠️ 局限性**

局限性包括：① 对持续激励的要求较高，若激励不足收敛会停止；② 两标量情形的吸引域有限，初始误差需在阈值以内；③ 仅在模拟中验证，缺乏实测实验和不同噪声/偏置条件下的鲁棒性评估。

---

## 393. GALA: Multimodal Graph Alignment for Bug Localization in Automated Program Repair

**arXiv ID:** 2604.08089 | [PDF](https://arxiv.org/pdf/2604.08089v1)

**作者:** Zhuoyao Liu `[一作]` (Sichuan University), Wei Ye `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

GALA通过构建图结构对多模态程序错误进行定位和修复。

**💡 创新点**

创新点在于使用图对齐在文件层和函数层实现视觉与代码结构的显式对应。

**🔧 技术方法**

使用视觉语言模型构造图，LLM进行跨模态对齐，基于Qwen3.5系列模型的推理与代理。

**📊 数据集**

使用SWE‑Bench Multimodal数据集。

**📈 对比分析**

与GUIRepair、SVRepair等方法对比，在相同模型规模下Pass@1提升至35.40%，超过最高对手。

**⚠️ 局限性**

局限性包括受LLM上下文长度限制，难以实现行级细粒度对齐，依赖仓库组织结构。

---

## 394. DiffVC: A Non-autoregressive Framework Based on Diffusion Model for Video Captioning

**arXiv ID:** 2604.08084 | [PDF](https://arxiv.org/pdf/2604.08084v1)

**作者:** Junbo Wang `[一作]` (Northwestern Polytechnical University), Jiangbin Zheng `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于扩散模型的非自回归视频字幕生成框架 DiffVC。

**💡 创新点**

创新点在于引入判别式去噪器显式分离视觉与文本交互，提升生成质量；并通过扩散过程实现并行解码，显著加快生成速度。

**🔧 技术方法**

使用了扩散模型、判别式去噪器、非自回归 Transformer 语言模型，以及视觉与文本编码器。

**📊 数据集**

使用了 MSVD、MSR‑VTT 和 VATEX 三个公开数据集。

**📈 对比分析**

与自回归与非自回归基线对比，DiffVC 在 MSR‑VTT、MSVD、VATEX 上取得了最高或接近 SOTA 的 BLEU、METEOR、ROUGE、CIDEr 分数，并且生成速度明显快于自回归模型。

**⚠️ 局限性**

局限性包括对极长句子生成的效果略逊于部分自回归方法，以及扩散模型训练成本较高；在稀有对象识别上仍存在不足。

---

## 395. From Binary Groundedness to Support Relations: Towards a Reader-Centred Taxonomy for Comprehension of AI Output

**arXiv ID:** 2604.08082 | [PDF](https://arxiv.org/pdf/2604.08082v1)

**作者:** Advait Sarkar `[一作]` (Microsoft Research), Viktor Kewenig `[通讯]` (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向读者的 AI 输出支撑关系分类体系，详细阐述如何将生成文本与源文档之间的关系细分为多种支撑类型（如直接引用、意译、演绎、归纳等），并设计对应的注释规范与基准测试流程。

**💡 创新点**

创新点在于突破传统的二元“支持/不支持”框架，融入语言学、哲学与论证学的理论基础，形成可操作、可解释的多层次支撑关系分类，可用于改进 AI 输出的来源呈现与用户批判性阅读。

**🔧 技术方法**

技术方法主要包括：① 参考 Toulmin 论证模型、Grice 对话含义理论等理论框架；② 通过结构化文献综述生成候选关系并精炼为可操作的分类；③ 设计人机标注规范与评测指标，计划将现有 groundedness/hallucination 数据集进行增量标注。

**📊 数据集**

目前尚未使用具体数据集；计划将现有的 groundedness 与幻觉评测语料（如 HaluEval、FactCHD 等）进行扩充，以加入多种支撑关系标签。

**📈 对比分析**

由于该工作为研究提案阶段，尚未完成实验或性能评估；未来计划通过人工标注一致性分析和模型对标来验证分类体系的可行性和实用价值。

**⚠️ 局限性**

局限性包括：① 体系尚未实现与验证，缺乏实证结果；② 可能因支撑关系细分过细导致标注者一致性下降；③ 对声明拆分、跨文档支撑等问题的具体处理尚未细化；④ 对不同领域与任务的迁移性和可扩展性尚需进一步研究。

---

## 396. Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference

**arXiv ID:** 2604.08133 | [PDF](https://arxiv.org/pdf/2604.08133v1)

**作者:** Baihui Liu `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于专家激活预算的分配框架 Alloc-MoE，分别在层级层面（Alloc-L）和 token 层面（Alloc-T）协调分配预算，以降低稀疏 MoE 推理时的性能损失。

**💡 创新点**

创新点包括：① 将专家激活数视为全局预算并进行约束；② 通过层级敏感度剖析与动态规划实现最优层级分配；③ 在 token 级利用路由分数动态重分配激活；④ 两者协同提升性能，兼顾推理速度与精度。

**🔧 技术方法**

使用的技术主要是：层级敏感度剖析与动态规划、路由分数重分配、Top-K 路由机制，以及对 DeepSeek‑V2‑Lite、Qwen1.5‑MoE、OLMoE 三个 MoE 模型的实验评估。

**📊 数据集**

校准数据集为 WikiText2，评估数据集包含 20 个自然语言理解、推理和数学任务（如 BoolQ、LAMBADA、RACE、SciQ、MNLI、QNLI、RTE、ARC、HellaSwag、LogiQA、MMLU、PIQA、TruthfulQA、ACP、BBH、GroundedCocoa、SWAG、GSM8K、ASDiv、MathQA）。

**📈 对比分析**

与 Uniform、LExI、Dynamic‑MoE、NAEE 等基线进行对比。Alloc‑MoE 在多模型、多预算下平均提升 0.4%–2.15% 的任务平均准确率，且在预算减半时实现 1.15×（prefill）/1.34×（decode）速度提升，且不产生额外推理延迟。

**⚠️ 局限性**

局限性包括：① 与专家裁剪、量化等硬件层面的加速方法未整合；② 未考虑专家放置或通信开销等分布式硬件因素；③ 目前仅适用于预训练模型，缺乏训练时的激活感知机制。

---

## 397. Can LLMs Deobfuscate Binary Code? A Systematic Analysis of Large Language Models into Pseudocode Deobfuscation

**arXiv ID:** 2604.08083 | [PDF](https://arxiv.org/pdf/2604.08083v1)

**作者:** Li Hu `[一作]` (University of Science and Technology of China), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了首个针对 LLM 的二进制反混淆基准，评估了从源代码到可执行文件各阶段混淆对 LLM 的影响；

**💡 创新点**

创新点在于：①将六种常见混淆变换、四种 ISA 与四种优化等级融入同一数据集，生成 2M+ 真实混淆程序；②设计四维度（词法一致、语义保真、代码简洁、可读性）评估框架，系统对比 LLM 与传统工具；③强调推理能力与任务微调对反混淆的决定性作用；

**🔧 技术方法**

采用 LLM 推理与少量示例提示、任务特定微调、双视角语义融合（嵌入 + 词集 Jaccard）、Halstead 复杂度、token‑delta 熵等多种技术；

**📊 数据集**

数据集基于 CodeNet 754k C/C++ 程序，经六种混淆变换合成 2,108,736 程序，覆盖 ARM、MIPS、x86、x64 以及 O0–O3 编译等级，并收集 500 条恶意二进制样本做实战验证；

**📈 对比分析**

通过词法、语义、简洁、可读性四维度对比，实验发现推理模型 DeepSeek‑R1 与 ChatDEOB 在严重混淆下保持最高语义保真，传统工具在可读性提升上不足，任务微调的 LLM 通常优于单纯的规模扩展；

**⚠️ 局限性**

局限性包括：①数据集无法覆盖所有新兴混淆技术；②评测仅限公开可用的 LLM 与工具，未能涵盖所有前沿方法；③评估指标虽多元，但仍无法完全体现逆向工程中的人工认知与后续安全分析需求。

---

## 398. Value-Guidance MeanFlow for Offline Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.08174 | [PDF](https://arxiv.org/pdf/2604.08174v1)

**作者:** Teng Pang `[一作]` (Shandong University), Yilong Yin `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过价值引导的条件行为克隆，利用 MeanFlow 的单步采样学习离线多智能体强化学习的最优联合策略。

**💡 创新点**

将全局优势值作为条件，并将 MeanFlow 与无分类引导结合，实现了对最优联合策略的无参数敏感条件行为克隆。

**🔧 技术方法**

采用 MeanFlow、无分类引导 (CFG)、优势值条件化、I‑GM 原则下的 Q 值学习与行为克隆等技术。

**📊 数据集**

在 StarCraft Multi‑Agent Challenge v1/v2 以及 Multi‑Agent MuJoCo（Half‑Cheetah、Hopper、Ant 等）离线数据集上进行实验。

**📈 对比分析**

与 10 种基线（单代理方法、流/扩散模型、现有 MARL 方法）对比，VGM²P 在连续环境中与最先进方法持平，离散环境中也表现优于传统行为克隆，且计算效率高。

**⚠️ 局限性**

单纯行为克隆在数据质量较差或更复杂的 SMACv2 场景下泛化不足，缺乏更强的协作机制，限制了方法的普适性。

---

## 399. Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence

**arXiv ID:** 2604.08169 | [PDF](https://arxiv.org/pdf/2604.08169v1)

**作者:** Niklas Herbster `[一作]` (Tara Research), Tommaso Tosato `[通讯]` (Tara Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了通过激活导向（activation steering）在大型语言模型中对抗恶意系统提示导致的失衡问题，并提出了两种基于投影感知的自适应干预方法；

**💡 创新点**

创新点在于利用逻辑回归判别边界与投影阈值，设计了仅在低分投影 token 上介入的 StTP 与 StMP 方法，既能恢复目标特质又不牺牲连贯性和原有能力；

**🔧 技术方法**

技术实现包括对比激活提取、逻辑回归分类、投影阈值门控、前向 Hook 进行激活修改、LLM-as-Judge 评估、ELO 比赛、Embedding 距离和交叉熵分析等；

**📊 数据集**

实验使用恶意系统提示对抗数据（训练 90/50、测试 40/40）以及 MASK、MMLU、MT-Bench、AlpacaEval 等基准数据集；

**📈 对比分析**

与未干预的恶意基线和对齐基线对比，SwFC 在某些层可将诚实/同情得分恢复至约84-88/71-78，连贯性保持 90-94；StTP 与 StMP 在相同层能获得相近得分，同时在多轮对话中重复率更低，且在 MMLU、MT-Bench、AlpacaEval 等能力基准上保持接近原始水平；

**⚠️ 局限性**

局限性包括仅评估恶意系统提示导致的失衡，实验仅覆盖 Llama‑3.3‑70B 与 Qwen3‑32B 两种模型，需白盒访问；线性假设可能不足以覆盖所有安全特征；单一向量可能无法处理多种失衡类型，且存在双重用途风险。

---

## 400. ViVa: A Video-Generative Value Model for Robot Reinforcement Learning

**arXiv ID:** 2604.08168 | [PDF](https://arxiv.org/pdf/2604.08168v1)

**作者:** Jindi Lv `[一作]`, Guan Huang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ViVa，一种基于预训练视频生成模型的价值函数，用来联合预测未来关节状态和任务价值，并将其嵌入 RECAP 强化学习管道中；

**💡 创新点**

创新点在于把视频生成模型的时空先验转化为价值估计，既能预测未来身体动态，又能在价值输出中融入这些动态，从而实现更可靠的任务进度感知与错误检测；

**🔧 技术方法**

采用预训练的扩散 Transformer（Wan2.2）与 VAE 的 latent 注入、flow‑matching 损失、DDIM 逆扩散，以及 RECAP 的优势估计框架；

**📊 数据集**

使用真实机器人收集的三类任务数据（衬衫折叠、箱子装配、卫生纸整理）的多视角 RGB 图像与关节状态，构成演示和训练集；

**📈 对比分析**

与 VLM 基价值模型（π_0.5、Gigabrain‑0 等）以及仅使用 VLM 的 RECAP 进行对比，ViVa 在箱子装配任务中实现 73% 的成功率和 14 tasks/h 的吞吐率，超过 VLM 的 58%/11；在所有三任务中价值曲线更平滑、对错误更敏感；

**⚠️ 局限性**

局限性包括对预训练视频模型的依赖、预测时长（K）需要精细调参、在极端长序列或不同硬件/环境下的泛化尚未充分验证。

---

## 401. Clickbait detection: quick inference with maximum impact

**arXiv ID:** 2604.08148 | [PDF](https://arxiv.org/pdf/2604.08148v1)

**作者:** Soveatin Kuntur `[一作]` (Warsaw University of Technology), Marcin Paprzycki `[通讯]` (Polish Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量化混合点击诱导检测框架，结合OpenAI语义嵌入和两项简洁的启发式特征；

**💡 创新点**

创新点在于用极少的启发式信号补充语义嵌入，采用PCA降维与图神经网络实现低延迟推理；

**🔧 技术方法**

使用OpenAI text‑embedding‑3‑large、PCA、XGBoost、GraphSAGE、GCN以及自定义的诱导度与信息量评分；

**📊 数据集**

基于Kaggle‑1、Kaggle‑2和Clickbait Challenge 2017（CC17）三大公开数据集，构建平衡的4万条标题样本；

**📈 对比分析**

通过F1、ROC‑AUC和每样本推理时间对比，GraphSAGE表现最佳（F1=0.8572，ROC‑AUC=0.9356，推理时间≈177.8 ms），XGBoost略低（F1=0.8465，236.6 ms），GCN最快（98.5 ms）但F1最低；

**⚠️ 局限性**

局限性包括特征集简化导致的F1略逊于重特征模型、嵌入生成成本高，以及仅利用标题信息，难以捕捉全文上下文的细节。

---

## 402. Multimodal Reasoning with LLM for Encrypted Traffic Interpretation: A Benchmark

**arXiv ID:** 2604.08140 | [PDF](https://arxiv.org/pdf/2604.08140v1)

**作者:** Longgang Zhang `[一作]` (Chongqing University), Lei Zhang `[通讯]` (Chongqing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种端到端的多模态网络流量解释框架mmTraffic，并构建了基于字节的流量描述基准BGTD

**💡 创新点**

创新点在于将原始加密流量字节与专家级语义知识联合标注，构建BGTD；并通过联合优化的感知-对齐-认知三模结构和语义优先生成损失，将低层字节表示与大语言模型对齐，解决黑盒与语义缺失问题

**🔧 技术方法**

使用NetMamba作为流量编码器，两个层MLP对齐器，辅助分类头，Qwen3-1.7B LLM（LoRA微调），自定义语义优先生成损失，LLM驱动结构化报告生成；实验中还采用了BERTScore、ROUGE-L、JSON有效率、ETC/QCR/PMR等评估指标

**📊 数据集**

六个公开流量数据集：CrossPlatform-Android、CrossPlatform-iOS、ISCXVPN2016、ISCX-Tor-2016、CSTNet-TLS1.3、USTC-TFC-2016，BGTD通过多源集成并进行类平衡、固定长度采样与专家知识自动生成

**📈 对比分析**

与传统单模NetMamba、Zero-shot LLM、Frozen-Encoder的Vanilla方法比较；mmTraffic在分类准确率与报告质量（ROUGE-L、BERTScore）上均明显优于基线，且保持100% JSON有效率；在结构一致性指标（ETC/QCR/PMR）上也表现突出

**⚠️ 局限性**

局限性包括：对字节级特征相似度高的类别仍难以区分，导致误分类并产生误报；对新出现或开放世界流量类别的可扩展性有限；推理时延和对抗性场景下的不确定性量化尚未完善

---

## 403. PolySLGen: Online Multimodal Speaking-Listening Reaction Generation in Polyadic Interaction

**arXiv ID:** 2604.08125 | [PDF](https://arxiv.org/pdf/2604.08125v1)

**作者:** Zhi-Yi Lin `[一作]` (Delft University of Technology), Xucong Zhang `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PolySLGen，一种在线多模态（语音、动作、说话状态）多方互动反应生成框架

**💡 创新点**

首次实现多方互动中的同时生成说话和倾听反应，并引入姿态融合模块和社会线索编码器；通过说话状态分数实现柔性交谈切换

**🔧 技术方法**

基于Llama3-8B-Instruct的LLM并通过LoRA微调；姿态融合模块采用层次Transformer；社会线索编码器基于头部朝向；语音风格适配器和动作解码器实现多模态映射

**📊 数据集**

DnD Group Gesture数据集（五人桌面角色扮演游戏的音频、视频和全身3D动作）

**📈 对比分析**

与随机、最近邻、LLM+ConvoFusion、LM-L2L、SOLAMI、Motion Forecast等基线对比；在动作精度、语音语义、说话状态AP、社会语义等指标上均显著优于所有基线

**⚠️ 局限性**

说话状态预测仍具挑战；数据集领域狭窄（仅DnD）；推理速度约5FPS，尚未实现实时部署

---

## 404. LegoDiffusion: Micro-Serving Text-to-Image Diffusion Workflows

**arXiv ID:** 2604.08123 | [PDF](https://arxiv.org/pdf/2604.08123v1)

**作者:** Lingyun Yang `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种微服务化的文本到图像扩散工作流服务系统，将工作流拆解为可独立调度的模型执行节点。

**💡 创新点**

创新点在于实现模型级别的共享与伸缩、自动化的图编译与自适应并行，以及基于节点级进度的SLO感知入队控制，显著提升多工作流部署的资源利用率和吞吐量。

**🔧 技术方法**

核心技术包括：Python嵌入式 DSL、图编译器、NVSHMEM 驱动的分布式数据引擎、基于模型状态表与延迟剖面进行的动态调度算法，以及异步延迟数据获取机制。

**📊 数据集**

使用 SD3、SD3.5‑Large、Flux‑Dev、Flux‑Schnell 四大主流扩散模型及其 ControlNet、LoRA 适配器，并在真实生产请求轨迹上进行评估。

**📈 对比分析**

通过与基线的 monolithic serving 系统（包括静态部署、swap‑based 与 Shepherd 调度）对比，实验表明该系统在多种负载、SLO 规模、峰值突发和集群规模下能够实现 3 倍以上的请求率、90% 以上的 SLO 达成率，并且在高峰突发时可容忍 8 倍更大的流量。

**⚠️ 局限性**

局限性主要包括：仍需依赖 GPU 集群，跨节点通信和调度开销在极大规模或高频请求场景下可能成为瓶颈；对模型组合极端复杂的工作流（如多级并行、深度多路分支）支持尚待进一步完善。

---

## 405. Small Vision-Language Models are Smart Compressors for Long Video Understanding

**arXiv ID:** 2604.08120 | [PDF](https://arxiv.org/pdf/2604.08120v1)

**作者:** Junjie Fei `[一作]` (Meta AI), Chenchen Zhu `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Tempo 框架，利用小型视觉‑语言模型（SVLM）作为查询感知的压缩器，再通过大型语言模型（LLM）完成长视频的生成式理解。

**💡 创新点**

核心创新在于零样本相关性先验 + Adaptive Token Allocation（ATA），能在不训练额外路由模块的情况下，根据查询动态分配 0.5–16 tokens/帧，保持因果顺序并显著压缩不相关上下文。

**🔧 技术方法**

技术组合包括跨模态蒸馏、SVLM 语义前置、零样本相关性评分、O(1) 头截断、分阶段进阶训练、分段序列拼接与时间标签。

**📊 数据集**

训练与评估使用公开的多模态混合数据（图像/视频/文本）以及 VideoChat‑Flash、LongVideoBench、LVBench、Video‑MME、MLVU 等长视频基准，SVLM 基于 Qwen3‑VL‑2B‑Instruct，LLM 为 Qwen3‑LM‑4B。

**📈 对比分析**

在 4K/8K 视觉 token 预算下，Tempo 在 LVBench 取得 52.7 分、Video‑MME 67.8 分、LongVideoBench 65.1 分，均超过 GPT‑4o、Gemini Pro 1.5、VideoChat‑Flash 等专门长视频模型；在 12K 预算下进一步提升至 53.7 分。

**⚠️ 局限性**

局限性包括：仍以单次前向推理为主，未针对多轮对话进行增量压缩；零样本路由精度受 SVLM 预训练影响，可进一步细化；极长视频（>1h）仍受预算限制，需更大上下文窗口才能进一步提升。

---

## 406. OV-Stitcher: A Global Context-Aware Framework for Training-Free Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2604.08110 | [PDF](https://arxiv.org/pdf/2604.08110v1)

**作者:** Seungjae Moon `[一作]` (University of Seoul), Youngmin Ro `[通讯]` (University of Seoul)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了训练无关的 OV-Stitcher 框架，利用拼接注意力在最后编码层实现子图像特征拼接并全局注意，消除滑动窗口碎片化；并加入类偏置提示提升语义对齐。

**💡 创新点**

创新点在于将子图像特征在最终层拼接后进行全局注意力，恢复跨窗口上下文；同时设计类偏置提示降低文本嵌入歧义，实现更精准的分割。

**🔧 技术方法**

采用 CLIP 视觉语言模型、DINO VFM 提取的空间特征、SAM2 生成掩膜、LLaMA3 生成类偏置提示，并在最后层实现 Stitch Attention。

**📊 数据集**

在八个开源语义分割基准上评估，包括 PASCAL VOC、PASCAL Context、COCO Object/Stuff、Cityscapes、ADE20K 等。

**📈 对比分析**

与 MaskCLIP、ProxyCLIP、CorrCLIP 等多种训练无关方法对比，OV‑Stitcher 在所有基准上均提升 mIoU 约 2%（最高 50.7），并在高分辨率下保持稳定性能。

**⚠️ 局限性**

局限性在于仍需滑动窗口预处理，拼接注意力仅应用于最后编码层，对极大图像或更深网络可能效果有限。

---

## 407. Quantum Vision Theory Applied to Audio Classification for Deepfake Speech Detection

**arXiv ID:** 2604.08104 | [PDF](https://arxiv.org/pdf/2604.08104v1)

**作者:** Khalid Zaman `[一作]` (Japan Advanced Institute of Science and Technology), Cem Direkoglu `[通讯]` (Middle East Technical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并应用量子视觉（QV）理论，将语音谱图转换为信息波以提升深度伪造语音检测。

**💡 创新点**

创新点在于将量子粒子-波动二象性引入音频特征表示，构造QV块生成信息波并与CNN/ViT结合。

**🔧 技术方法**

采用QV块+CNN或Vision Transformer，利用STFT、Mel谱和MFCC等频谱特征进行端到端训练。

**📊 数据集**

使用ASVspoof 2019数据集进行实验。

**📈 对比分析**

与传统CNN/ViT、CQCC、ResNet等基线对比，QV-CNN在MFCC上达到94.20%准确率、9.04%EER，QV-ViT亦显著提升。

**⚠️ 局限性**

局限在于仅验证单一数据集且QV块增加计算复杂度，需进一步验证跨域和实时部署性能。

---

## 408. Shift- and stretch-invariant non-negative matrix factorization with an application to brain tissue delineation in emission tomography data

**arXiv ID:** 2604.08161 | [PDF](https://arxiv.org/pdf/2604.08161v1)

**作者:** Anders S. Olsen `[一作]`, Gitte M. Knudsen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

本文提出并实现了可同时估计时移与时伸缩的非负矩阵分解（shift‑stretch NMF），并将其应用于合成数据和猪脑 SPECT 数据的脑组织分割；

**💡 创新点**

创新点在于将时间延迟（整数与小数）与时间拉伸（压缩/扩展）统一在频域下处理，采用频谱截断/零填充构造拉伸谱库，并用相位调制实现时移；

**🔧 技术方法**

使用频域 NMF、交叉相关法估计时移/拉伸、PyTorch 进行非线性优化、零填充与截断等信号处理技术；

**📊 数据集**

使用了人工生成的随机时移/拉伸合成数据和 5 只猪注射 99mTc‑DTPA 后的 SPECT 数据；

**📈 对比分析**

与传统 NMF、整数时移 NMF、非整数时移 NMF 对比；shift‑stretch NMF 在匹配相关性和方差解释率上表现最佳，尤其在 K=1、2 时效果显著；

**⚠️ 局限性**

局限包括：时移采用循环假设不符合实际、频域操作易产生 Gibbs 抖动、组件间时序未对齐、对光滑信号要求高、对噪声敏感且模型不唯一。

---

## 409. Uni-ViGU: Towards Unified Video Generation and Understanding via A Diffusion-Based Video Generator

**arXiv ID:** 2604.08121 | [PDF](https://arxiv.org/pdf/2604.08121v1)

**作者:** Luozheng Qin `[一作]` (Shanghai Academy of AI for Science), Hao Li `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Uni-ViGU，一种以视频生成模型为基础的统一多模态框架，既能生成视频又能理解视频并生成文字描述。

**💡 创新点**

创新点在于：①将生成模型视为知识基础，逆向利用其文本-视频映射实现理解；②引入统一流（uni‑flow）同时处理视频的连续流匹配与文本的离散流匹配；③设计模态驱动的 MoE 结构，注意力共享、FFN 分离；④双向两阶段训练（知识回忆与能力细化）实现跨模态知识迁移。

**🔧 技术方法**

技术主要包括：流匹配（continuous & discrete）、Diffusion Transformers（DiT）、VAE 编码、Mixture‑of‑Experts、条件 Dropout、两阶段训练策略、ODE/解码等。

**📊 数据集**

数据集为自制的 20K 条视频-提示-详细字幕三元组（10K 用于知识回忆，10K 用于细化），视频通过现有文本‑视频生成器合成，字幕由 LLM 生成。

**📈 对比分析**

实验显示 Uni‑ViGU 在联合视频‑文本生成上能同时输出高质量视频和细致字幕，且视频生成质量与原始生成模型相当；在视频理解上能较好复现提示文本，细化阶段进一步提升对细节的捕捉。相比传统双塔或仅基于理解的 MLLM，Uni‑ViGU 在生成/理解任务上取得更优或相近的性能，且训练成本显著降低。

**⚠️ 局限性**

局限性包括：①仅在单一视频生成模型上验证，缺乏跨模型通用性；②对长视频或复杂字幕的性能尚未充分评估；③生成文本仍受限于离散流匹配的稳定性；④双向训练需手工设计两阶段数据和超参数。

---

## 410. Face-D(^2)CL: Multi-Domain Synergistic Representation with Dual Continual Learning for Facial DeepFake Detection

**arXiv ID:** 2604.08159 | [PDF](https://arxiv.org/pdf/2604.08159v1)

**作者:** Yushuo Zhang `[一作]` (East China Normal University), Zhaoxia Yin `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种持续学习的 DeepFake 检测框架 Face‑D^2CL，能够在不回放历史数据的前提下，连续学习多种面部伪造技术并保持已学知识。

**💡 创新点**

创新点在于：1) 多域协同表示——同时提取空间、波形与傅里叶三种域的伪造痕迹并对齐；2) 双重持续学习机制——结合类感知 EWC 与正交梯度约束 (OGC) 两种正则化，既保持全局参数稳定，又使低秩适配器自由迁移。

**🔧 技术方法**

使用 CLIP ViT‑L/14 视觉编码器 + 三个 LoRA 适配器；多域特征融合、对齐损失、二元交叉熵、EWC 及 OGC 正则化。

**📊 数据集**

在 FF++, DFDCP, DFD, CDF2, MCNet, BlendFace, StyleGAN3 等公开 DeepFake 数据集上训练；在 DF40、UADFV、WildDeepfake 等未见过的数据集评估。

**📈 对比分析**

与离线（DFD‑FCG、DFFreq）和在线（DFIL、SUR‑LID、SAIDO）持续学习方法对比，采用 ACC、AUC、AA、AF 等指标。Face‑D^2CL 在两种增量协议中均取得最高 AA、最低 AF，并在未见域上 AUC 提升 7.9%（对比 SOTA）。

**⚠️ 局限性**

局限性包括：1) 仍需手工设计三域对齐方式；2) 训练过程对硬件要求高（CLIP 大模型 + LoRA）；3) 目前仅针对面部伪造，对其他类型视频伪造的泛化尚待验证。

---

## 411. Test-Oriented Programming: rethinking coding for the GenAI era

**arXiv ID:** 2604.08102 | [PDF](https://arxiv.org/pdf/2604.08102v1)

**作者:** Jorge Melegati `[一作]` `[通讯]` (University of Porto), Jorge Melegati (University of Porto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为Onion的原型工具，基于大语言模型（LLM）通过测试代码自动生成生产代码，实现了从自然语言规范到完整程序的闭环；

**💡 创新点**

提出Test‑Oriented Programming（TOP）范式，借助LLM把代码生成完全委托给机器，开发者只需验证测试代码，形成比传统TDD更高层次的抽象；

**🔧 技术方法**

使用OpenAI GPT‑4o‑mini和Google Gemini 2.5‑Flash两种LLM，结合YAML配置、提示工程、命令行交互与自动化测试生成；

**📊 数据集**

实验数据基于手写的项目规范与验收测试，生成的CLI工具示例，收集了交互日志、生成代码及测试用例；

**📈 对比分析**

通过对同一任务分别用两种模型运行五次，比较开发者干预次数、代码长度、注释量、成功率等指标；结果显示GPT‑4o‑mini更少需改动测试代码，Gemini在支持代码和测试执行上出现更多问题，代码差异明显；

**⚠️ 局限性**

该原型仅适用于小规模示例，无法直接用于复杂系统；测试代码量大、LLM非确定性和跨模型差异明显，需要自动化测试验证工具和更细粒度的模块化支持。

---

## 412. State-Flow Coordinated Representation for MI-EEG Decoding

**arXiv ID:** 2604.08157 | [PDF](https://arxiv.org/pdf/2604.08157v1)

**作者:** Guoqing Cai `[一作]`, Ting Ma `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种双分支的状态-流协调网络（StaFlowNet），在运动意象EEG解码中显式分离并协调全局状态信息与细粒度流信息。

**💡 创新点**

创新点在于引入状态调制流模块，在多尺度GRU金字塔上使用全局状态向量动态调制局部时序特征，显著提升特征可分辨性与解码性能。

**🔧 技术方法**

使用双分支卷积编码器、时间差分流编码器、双向GRU金字塔、层归一化调制、MLP分类头等技术，并融合多尺度特征。

**📊 数据集**

在BCI Competition IV-2a、IV-2b以及OpenBMI这三个公开运动意象EEG数据集上进行评估。

**📈 对比分析**

与CNN、Transformer、FBCSP等基线方法相比，StaFlowNet在三数据集上均取得最高准确率（约80%），并通过Wilcoxon检验显著优于其他方法。

**⚠️ 局限性**

局限性包括对全局状态向量的学习依赖，且在极小数据或噪声严重的场景下性能尚未验证，模型结构相对复杂。

---

## 413. Training Data Size Sensitivity in Unsupervised Rhyme Recognition

**arXiv ID:** 2604.08156 | [PDF](https://arxiv.org/pdf/2604.08156v1)

**作者:** Petr Plecháč `[一作]` (Institute of Czech Literature, Czech Academy of Sciences), Robert Kolár `[通讯]` (Institute of Czech Literature, Czech Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究无监督押韵识别模型对训练数据量的敏感性，并在七种欧洲语言上与人工标注和LLM进行系统比较。

**💡 创新点**

提出训练规模与语言屈折度的对应关系，建立人类标注一致性基准，并证明RhythmTagger可在足够数据下超越人类一致性。

**🔧 技术方法**

使用无监督学习工具RhythmTagger（基于T‑score共现提取）进行押韵识别；使用GPT‑4o、Claude3.7 Sonnet和DeepSeek‑V3进行一次性学习；对标注者一致性采用二元逻辑回归分析。

**📊 数据集**

PoeTree诗歌语料库（约1900–2400行/样本），涵盖捷克语、德语、英语、法语、意大利语、俄语、斯洛文尼亚语；训练样本规模从2k行到1M行。

**📈 对比分析**

通过F1分数与人工标注一致性对比：RhythmTagger在多数语言达到或超过人类标注一致性；LLM的F1分数远低于RhythmTagger，主要因缺乏音素表示导致错误。

**⚠️ 局限性**

局限性包括：人工标注不统一导致基准有限；俄语和斯洛文尼亚语模型受限于转写质量和对远距押韵的容忍度；LLM缺乏明确的音素表示，导致识别错误。

---

## 414. Bag of Bags: Adaptive Visual Vocabularies for Genizah Join Image Retrieval

**arXiv ID:** 2604.08138 | [PDF](https://arxiv.org/pdf/2604.08138v1)

**作者:** Sharva Gogawale `[一作]` (Tel Aviv University), Nachum Dershowitz `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于页面自适应词典的文稿片段检索方法（Bag of Bags），用于识别同一原稿的碎片。

**💡 创新点**

创新点在于用每个碎片的本地视觉词典取代全局代码本，并采用集合到集合的距离（Chamfer、Hungarian、OT）实现高效碎片级检索。

**🔧 技术方法**

使用稀疏卷积自编码器提取字符补丁特征，k-means生成局部词典，并结合Chamfer/Hungarian/OT距离以及两阶段BoW+BoB-OT重排序。

**📊 数据集**

评估基于Cairo Genizah片段基准（287张图像，100个join簇），来自多家机构。

**📈 对比分析**

与多种BoW基线和池化基线对比，BoB-Chamfer取得Hit@1 78.4%、MRR 84.1%，比最强BoW提升6.1%相对，显示显著性能提升。

**⚠️ 局限性**

局限性包括仅在小规模人工标注子集验证，未覆盖完整约25万+碎片；对极度损毁或尺寸极小的碎片效果尚未验证，且依赖手工join标签。

---

## 415. Graph Neural Networks for Misinformation Detection: Performance-Efficiency Trade-offs

**arXiv ID:** 2604.08131 | [PDF](https://arxiv.org/pdf/2604.08131v1)

**作者:** Soveatin Kuntur `[一作]` (Warsaw University of Technology), Amir H. Gandomi `[通讯]` (Óbuda University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对经典轻量级图神经网络（GCN、GraphSAGE、GAT、ChebNet等）在七个多语言、多域的误信息检测数据集上进行系统性评估。

**💡 创新点**

证明在仅使用相同TF‑IDF特征且不做额外预训练的条件下，经典GNN能显著优于强大非图模型，并且在低资源场景下仍保持高性能。

**🔧 技术方法**

采用k‑NN构造的文本相似度图，使用Lightweight GNN架构，配合标准的分类评估指标（F1、MCC）和推理时间测量。

**📊 数据集**

使用七个公开数据集：CLICK‑ID、LIAR、FakeNewsNet、Kaggle ISOT、WELFake、COVID‑19 Fake News、MIPD（印尼、波兰、英语）。

**📈 对比分析**

与Logistic Regression、SVM、MLP等基线在相同TF‑IDF特征下对比，经典GNN在F1得分上提高10–30个百分点，推理时间与MLP相近甚至更低。

**⚠️ 局限性**

未与大规模Transformer模型进行直接比较，且k‑NN图构造和特征选择对性能影响尚未深入探究。

---

## 416. Initialisation Determines the Basin: Efficient Codebook Optimisation for Extreme LLM Quantization

**arXiv ID:** 2604.08118 | [PDF](https://arxiv.org/pdf/2604.08118v1)

**作者:** Ian W. Kennedy `[一作]` (University of Sheffield), Nafise Sadat Moosavi `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了自由形式加法量化在极低比特压缩（2bpp）下的初始化瓶颈，并提出基于 Hessian 加权 Mahalanobis 距离的输出感知 EM 初始化方法 OA-EM。

**💡 创新点**

创新点在于引入代表性比率 ρ 预测初始化敏感度，以及提出 OA-EM 改进初始化，显著提升 2bpp 压缩下的性能并在 PV 调优后保持优势。

**🔧 技术方法**

技术包括加法量化、Beam Search、Expectation‑Maximisation、Hessian 加权 Mahalanobis 距离、PV 调优和基于激活的损失最小化。

**📊 数据集**

使用数据集为 C4 进行校准，以及 WikiText‑2、ARC‑Easy/Challenge、HellaSwag、PIQA、WinoGrande、LAMBADA 等零样本评测。

**📈 对比分析**

与传统贪婪初始化相比，OA-EM 在 Llama 3.2‑3B 等模型上将 2bpp 的 perplexity 从 60.61 降至 17.39，并在 PV 调优后保持低于 11.53 的 perplexity；在相同计算预算下，OA-EM 在 2bpp 下的表现优于宽束宽度为 16 的贪婪方案。

**⚠️ 局限性**

局限性包括仅验证三款 3B‑8B 规模模型，未覆盖更大规模或多语言数据，且方法仅适用于自由形式加法量化，对基于格或树的量化方案不适用。

---

## 417. A unifying view of contrastive learning, importance sampling, and bridge sampling for energy-based models

**arXiv ID:** 2604.08116 | [PDF](https://arxiv.org/pdf/2604.08116v1)

**作者:** Luca Martino `[一作]` `[通讯]` (University of Catania), Luca Martino (University of Catania)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出一个统一框架，将噪声对比估计（NCE）、逆逻辑回归（RLR）、多重要性采样（MIS）与桥式采样等方法连接起来，并推导它们在一定条件下的等价性；基于该视角设计了新的估计器和改进的得分规则；

**💡 创新点**

通过统一视角阐明了各方法间的本质联系，利用桥式采样的最优桥函数和得分规则的对应关系提出更高效的估计器，并探索多提议采样的潜力；

**🔧 技术方法**

采用对比学习、逆逻辑回归、重要性采样、桥式采样、多重提议采样以及可变得分规则的数值推导与实现；

**📊 数据集**

以一维高斯目标模型为实验对象（可在公开 MATLAB 代码中复现），并使用人工生成的数据集进行评估；

**📈 对比分析**

通过对比 MSE 与桥式采样、MIS、Self‑IS‑with‑mix、标准 IS、RIS、Opt‑Umb 等传统方法，在理想、几乎理想与真实初始化条件下验证新估计器的性能；在大样本或良好提议密度下，优化桥式或自适应混合估计器往往优于传统方法；在 θ 空间上，NCE 与最大似然在 MSE 上接近；

**⚠️ 局限性**

对提议密度高度依赖，样本量需足够大；递归估计可能收敛慢，尤其在初始值不佳时；验证主要集中在简单模拟模型，尚未在复杂高维 EBM 上进行实证验证。

---

## 418. StoryEcho: A Generative Child-as-Actor Storytelling System for Picky-Eating Intervention

**arXiv ID:** 2604.08114 | [PDF](https://arxiv.org/pdf/2604.08114v1)

**作者:** Yanuo Zhou `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种名为StoryEcho的基于生成式AI的儿童主体叙事系统，用于家庭中低压、非餐时的挑食干预，旨在提升儿童对低偏好食物的尝试意愿并缓解家长的喂食压力。

**💡 创新点**

创新点包括：① 将儿童定位为故事中的持续角色和行为作者，使现实饮食行为直接影响故事发展，形成“儿童-故事-行为”循环；② 将干预延伸至餐前/餐后日常生活，通过轻量化反馈和行为导向的故事更新实现持续、低压的认知改变；③ 结合生成式语言模型与图像模型实现个性化、可持续的故事生成，并在设计上强调家长的可审核与干预。

**🔧 技术方法**

技术实现主要使用：React+TypeScript+Tailwind 前端；后端 Python FastAPI 与 SQLite；生成式文本采用 GPT‑5 Chat Completions；图像与文本修正使用 GPT‑Image‑1；反馈文本使用 GPT‑4o‑mini；语音功能通过 edge‑tts 与 GPT‑4o‑transcribe‑diarize；整体架构支持实时故事生成、情绪表达与行为反馈。

**📊 数据集**

使用的“数据集”主要为实验参与者在真实家庭环境中记录的吃饭尝试行为日志（包括尝试分数、描述）、故事互动记录、以及家长与儿童的半结构化访谈文本；对照组使用通用 LLM 对话平台 Doubao 的对话日志。没有使用公开的大规模食品偏好或儿童行为公开数据集。

**📈 对比分析**

比较方法为 5 天的组间实验，实验组使用 StoryEcho，控制组使用 Doubao。通过线性混合效模型评估会话级指标（尝试意愿、抵抗度、情绪、父母压力）以及家庭问卷（儿童饮食行为量表 CEBQ‑FF、父母压力指数 PSI‑SF）和 SUS 可用性量表。实验组在尝试意愿和抵抗度上显著优于对照组，父母压力亦显著下降；可用性评分达到 87.86，表明系统易用且受欢迎。

**⚠️ 局限性**

局限性包括：样本规模小（仅 11 家庭，实验组 7 家庭）；实验周期短（仅 5 天），无法观察长期行为转化和饮食习惯改变；对照组仅使用通用聊天机器人，未进行针对性干预；依赖家长记录行为的主观性；系统目前仅在中文环境下测试，跨语言适配未知。

---

## 419. EPIR: An Efficient Patch Tokenization, Integration and Representation Framework for Micro-expression Recognition

**arXiv ID:** 2604.08106 | [PDF](https://arxiv.org/pdf/2604.08106v1)

**作者:** Junbo Wang `[一作]` (Northwestern Polytechnical University), Kun Hu `[通讯]` (Edith Cowan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种高效微表情识别框架 EPIR，并在其上设计了双模范数移位 patch tokenization、token integration 以及动态 token 选择模块，旨在在保持高识别精度的同时降低模型复杂度。

**💡 创新点**

创新点在于引入双模范数移位 tokenization (DNSPT) 以捕获面部相邻像素的空间关系，结合 token integration 模块在 Transformer 块中动态合并相似 token，使用 ITALS 对自注意力进行对角遮蔽并学习缩放因子，以及 DTSM 通过注意力传播动态挑选关键 token，从而实现轻量化且高性能的微表情表示。

**🔧 技术方法**

技术手段主要包括基于 Vision Transformer 的架构、DNSPT、token integration、DITSM、ITALS（改进的自注意力）、对比损失、以及多头注意力与位置编码等。

**📊 数据集**

实验使用了四个公开微表情数据集：CASME II、SAMM、SMIC 以及 CAS(ME)^3。

**📈 对比分析**

在 CDE 与 SDE 的 Leave‑One‑Subject‑Out 评估协议下，与传统方法、CNN、Transformer 以及大模型基线进行对比，EPIR 在 UF1、UAR 等指标上分别提升至 9.6%/4.58%（CAS(ME)^3）、4.26%/1.46%（CASME II）、1.38%/3.15%（SMIC）和 9.66%/5.52%（SAMM），且模型参数仅为 HTNet 的 1.6%，显著优于现有方法。

**⚠️ 局限性**

局限性包括：在极小规模数据集上仍可能出现过拟合；模型在不同设备或实时部署场景下的推理速度与能耗尚未深入评估；跨域泛化能力未在更广泛的数据集上验证。

---

## 420. The Boolean surface area of polynomial threshold functions

**arXiv ID:** 2604.08095 | [PDF](https://arxiv.org/pdf/2604.08095v1)

**作者:** Joseph Slote `[一作]` (University of Washington), Haonan Zhang `[通讯]` (University of South Carolina)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了多项式阈值函数（PTFs）的布尔表面面积，证明其上界为子多项式级别 exp(C(d)√log n)。

**💡 创新点**

首次给出 PTF 的布尔表面面积在子多项式时间内的上界，显著优于之前的 n^1/4(log n)^C(d) 结果，并通过随机划分和超几何分布的关键估计实现。

**🔧 技术方法**

采用随机划分与超几何分布分析、Jensen 不等式、α(p) 参数、决策树分解及对偶变换等理论工具。

**📊 数据集**

无实验数据，纯理论分析。

**📈 对比分析**

与之前基于平均灵敏度的上界比较，理论上实现了指数级更好的上界，并给出了相应的噪声灵敏度上界。

**⚠️ 局限性**

结果仍可能不是最优；缺乏匹配的下界或极端例子；对参数选择和随机划分的依赖较强。

---

## 421. Analysis of Search Heuristics in the Multi-Armed Bandit Setting

**arXiv ID:** 2604.08109 | [PDF](https://arxiv.org/pdf/2604.08109v1)

**作者:** Jasmin Brandt `[一作]` (University of Bielefeld), Jurek Sander `[通讯]` (Hasso Plattner Institute University of Potsdam)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在双刃赌博（Dueling Bandits）环境下，随机搜索启发式（如进化算法 EA 和估计分布算法 EDA）识别 Condorcet 胜者的能力，并给出理论分析与改进策略。

**💡 创新点**

创新点在于：①首次系统性比较 EA 与 EDA 在无结构 Arms 的 Condorcet 识别问题上的性能差异；②提出通过多次对决（boosting）和 Plackett‑Luce 模型提升 EDA 的 stationary 分布；③给出精确的马尔可夫链混合时间和 stationary 分布界限。

**🔧 技术方法**

采用了马尔可夫链分析、耦合（coupling）与漂移（drift）理论、Plackett‑Luce 统计模型，以及多次对决（boosting）技巧。

**📊 数据集**

本文为理论工作，未使用真实数据集；所有结果均基于假设的随机对决矩阵 M 或 Plackett‑Luce 参数。

**📈 对比分析**

通过对比 EA 在 stationary 分布中识别 Condorcet 胜者的概率仅为常数级别（p=Ω(1/n)），而 EDA 能将该概率提升到 1-Θ(p)。此外，多次对决可将 EDA 的识别概率提升至 1-Θ(1/n)，证明了方法的有效性。

**⚠️ 局限性**

局限性包括：①只考虑无结构 Arms 的场景，未考虑搜索空间结构；②全部结论为理论上限，缺乏实证验证；③对 EDA 的参数选择（如 ρ）敏感，实际性能可能因实现差异而异。

---

## 422. OceanMAE: A Foundation Model for Ocean Remote Sensing

**arXiv ID:** 2604.08171 | [PDF](https://arxiv.org/pdf/2604.08171v1)

**作者:** Viola-Joanna Stamer `[一作]` (Technische Universität Berlin), Begüm Demir `[通讯]` (Technische Universität Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了OceanMAE，一个利用多光谱Sentinel-2图像和物理海洋描述子进行自监督预训练的掩码自编码器，并将其编码器迁移到改进的UNet与Bathy-UNet网络中，用于海洋污染、漂浮物分割和海深回归等下游任务。

**💡 创新点**

创新点在于：①将物理海洋特征（如海深、叶绿素浓度、Secchi深度）融合到自监督学习中，构建海洋专属的预训练模型；②在预训练和下游训练中设计了并行嵌入流，充分利用预训练的全局表示；③通过多任务实验验证了该方法在不同任务与数据规模下的有效性。

**🔧 技术方法**

使用技术包括：自监督掩码自编码器（MAE）框架；ViT-Base编码器；多光谱特征融合；并行嵌入投影与UNet结构；MSE重建损失；AdamW优化器与学习率调度。

**📊 数据集**

数据集：预训练使用Hydro（10万张256×256 Sentinel-2 L2A海洋与沿海图块）；下游微调使用MARIDA（海洋漂浮物分割）、MADOS（海洋污染与水面分割）和MagicBathyNet（海深回归）三大公开数据集。

**📈 对比分析**

比较方法：将OceanMAE与其他自监督模型（FGMAE、SatMAE、SSL4EO）以及UNet基线和SOTA MariNeXt进行对比；在MARIDA、MADOS、MagicBathyNet上分别使用像素准确率、mIoU、Macro F1、MAE、RMSE、StdDev等指标。结果显示：OceanMAE在海洋漂浮物分割和污染分割任务上取得最高的PA/Macro F1，mIoU优于大多数模型；在海深回归中性能与地区差异与数据量相关，表现竞争但并未始终压倒基线。

**⚠️ 局限性**

局限性包括：①海深回归效果受地区差异与数据规模影响，需进一步探索特征泛化；②辅助海洋描述子对不同评价指标的提升不一致，可能需要更细致的特征选择；③模型对大规模无标签海洋数据的依赖，若数据不足会导致预训练效果下降；④实验主要基于Sentinel-2，跨传感器和跨区域推广的鲁棒性尚未验证。

---

## 423. Semantic-Aware UAV Command and Control for Efficient IoT Data Collection

**arXiv ID:** 2604.08153 | [PDF](https://arxiv.org/pdf/2604.08153v1)

**作者:** Assane Sankara `[一作]` (Mohammed VI Polytechnic University), Hajar El Hammouti `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个语义感知的无人机指挥与控制框架，用于高效收集物联网设备的图像数据。

**💡 创新点**

创新点在于将语义通信与无人机控制整合，构建带有下行延迟的马尔可夫决策过程，并采用双重深度 Q 学习自适应优化加速度指令，以最大化平均图像重建质量。

**🔧 技术方法**

使用了深度联合源-信道编码 (DeepJSCC)、深度强化学习中的双重深度 Q 学习 (DDQN)、OFDMA 多址、PSNR 评价以及马尔可夫决策过程模型。

**📊 数据集**

实验采用10台随机部署在 1000×600 m² 区域内的物联网设备进行仿真，图像通过 DeepJSCC 编码，未使用公开真实数据集。

**📈 对比分析**

通过与贪心策略和旅行商问题(TSP)两种基线进行对比，DDQN 在不同带宽和无人机速度下均实现更高的平均 PSNR，且轨迹更灵活，性能显著优于基线。

**⚠️ 局限性**

局限性包括仅在仿真环境中验证，未考虑真实无线信道衰落、无人机与障碍物交互、多无人机协作以及实际部署中的能耗与安全等因素。

---

## 424. Semantic Noise Reduction via Teacher-Guided Dual-Path Audio-Visual Representation Learning

**arXiv ID:** 2604.08147 | [PDF](https://arxiv.org/pdf/2604.08147v1)

**作者:** Linge Wang `[一作]` (Chinese Academy of Sciences), Jinqiao Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 TG-DP 教师指导双路框架，分离音视频重建与对比学习到两个独立前向通道，并通过教师网络对对比分支的可见 token 进行引导。

**💡 创新点**

创新点在于（1）解耦重建与对比的掩码策略，降低两者梯度干扰；（2）引入轻量化教师‑学生蒸馏，为对比分支提供全视角语义约束；（3）利用教师注意力制定结构化掩码，提升对比视图的语义相关性。

**🔧 技术方法**

技术包括自监督 MAE、InfoNCE 对比损失、EMA 方式的教师‑学生蒸馏、教师引导的 token‑优先掩码，以及双路前向传播。

**📊 数据集**

使用了大规模音视频数据集 AudioSet‑2M（约 1.4M 可用样本）和 VGGSound，作为预训练和下游评估的数据源。

**📈 对比分析**

在零样本音视频检索上，TG‑DP 在 AudioSet 上将 R@1 从 35.2% 提升到 37.4%，在 VGGSound 上提升到 52.7% Top‑1，均达到或超过现有最优方法；在线性探针分类上，AudioSet‑20K mAP 达到 32.0，VGGSound Top‑1 亦达到 52.7%。

**⚠️ 局限性**

局限性包括训练时间增加（单个 epoch 从 730s 提升至 1045s），对极低掩码比例时分类性能下降，以及教师‑学生蒸馏对检索性能提升有限。

---

## 425. LLM-Based Data Generation and Clinical Skills Evaluation for Low-Resource French OSCEs

**arXiv ID:** 2604.08126 | [PDF](https://arxiv.org/pdf/2604.08126v1)

**作者:** Tian Huang `[一作]` (LORIA), Irina Illina `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套用于生成法语OSCE（标准化临床考核）对话转录并通过本地可部署LLM进行自动评估的完整流程，解决了缺乏公开法语OSCE文本和标注难题；

**💡 创新点**

创新点包括：1）基于评估标准指导对话生成并引入扰动以模拟不同学生表现；2）采用LLM辅助的银标注和可调严格度评估；3）在中等规模开源LLM上实现与大型模型相当的评估效果；4）引入复合标准分解和医学定义注入两种辅助工具，提升小模型表现；

**🔧 技术方法**

技术手段涵盖：大规模语言模型（GPT‑4o、Qwen3、Llama3.1等）、LLM提示工程（zero-shot、few-shot、multi‑step）、评估标准分解（CD）、医学定义注入（MD）、文本后处理与评估流水线；

**📊 数据集**

数据集：利用10个法语OSCE站点的医生/病人/评估表生成两套语料（未扰动/扰动），共179个二元评估标准；另外使用两份真实教程会话进行小规模验证；

**📈 对比分析**

比较方法：在不同模型、提示策略和辅助工具组合下计算二元分类准确率；大型模型（GPT‑4o、Claude3、Anthropic 等）在未扰动语料上达90%+准确率；中等规模LLM（8B–32B）在最佳配置下可达约85%；多步策略普遍低于zero‑shot；复合标准分解在扰动语料中提升性能；医学定义注入效果不显著；

**⚠️ 局限性**

局限性：1）生成对话过于理想化，缺乏真实对话噪声与自发性；2）银标注未经过专家裁定，作为参考标准的可靠性有限；3）模型依赖于GPT‑4o生成的样本，可能存在偏倚；4）只评估二元标签，未考察评估理由与证据的教学价值；5）对真实数据的验证仅限两条记录，缺乏大规模实证。

---

## 426. Beyond Stochastic Exploration: What Makes Training Data Valuable for Agentic Search

**arXiv ID:** 2604.08124 | [PDF](https://arxiv.org/pdf/2604.08124v1)

**作者:** Chuzhan Hao `[一作]` (Alibaba Cloud Computing), Yuewei Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HiExp 框架，通过自我反思与层次聚类从 LLM 的内部推理轨迹中抽取经验，并在强化学习中以经验对齐方式指导搜索代理，从而提升推理效率与训练稳定性。

**💡 创新点**

创新点在于：① 端到端的层次经验构造（自我反思+对比蒸馏+多级聚类），② 将抽取的经验动态注入 RL 训练，构成经验对齐的策略优化；③ 通过经验对齐将随机探索转化为战略性搜索，显著提升样本效率与收敛稳定性。

**🔧 技术方法**

核心技术包括：自我反思的对比蒸馏、语义编码与层次聚类（凝聚聚类）、critic‑free RL（如 GRPO、GSPO）、检索增强生成（RAG）与多模态检索工具（Tavily、dense retriever），以及强化学习目标中加入经验检索的设计。

**📊 数据集**

使用的主要数据集：多跳推理任务 HotpotQA、2WikiMultiHopQA、Musique、Bamboogle、MoreHopQA、Frames；数学推理任务 AIME、AMC、MATH‑500、Minerva、OlympiadBench；检索环境包括 2018 Wikipedia、Tavily Web 搜索。

**📈 对比分析**

与多种基线（Prompt‑based RAG、Search‑o1、DeepSeek‑R1、GPT‑4.1 等）及 RL 算法（GRPO、GSPO）对比，HiExp 在所有任务上均获得 4–10% 的 F1/CEM/EM 提升；在 7B 规模模型上实现与 GPT‑4 等前沿模型相当或更优的性能；训练过程显示奖励波动显著减小，梯度方差降低，训练更稳定。

**⚠️ 局限性**

局限性：HiExp 的经验构造与 RL 训练是分离的，未形成闭环；随着模型训练进展，原先抽取的经验可能不再完全匹配模型能力，导致指导效果衰减；此外，经验构造依赖初始策略的自我反思，若初始模型欠佳，经验质量可能不足。

---

## 427. Revise: A Framework for Revising OCRed text in Practical Information Systems with Data Contamination Strategy

**arXiv ID:** 2604.08115 | [PDF](https://arxiv.org/pdf/2604.08115v1)

**作者:** Gyuho Shim `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Revise框架，系统纠正OCR的字符、单词和结构错误，以恢复文档结构与内容。

**💡 创新点**

基于层级错误分类与真实感错误注入的合成数据生成，构建无需人工标注的OCR纠错体系。

**🔧 技术方法**

使用Llama-3.1-1B-Instruct等LLM作为纠错模型，并结合合成错误注入与训练策略。

**📊 数据集**

利用公开的Wikipedia文本生成合成数据，并在VisualMRC、DUDE、DocVQA、CORD、FUNSD等检索与问答基准上评测。

**📈 对比分析**

通过Recall@K、BERTScore及QA指标（F1、CIDEr）与原始OCR及单一错误模型对比，Revise_meta平均提升Recall约1.3–17.3%，BERTScore提升，QA得分提升2.6%/0.8%等。

**⚠️ 局限性**

局限在于未针对行业专属文档、表格、图表等多模态内容验证，合成错误比例依赖经验且未覆盖罕见错误；评估主要依赖LLM，可能存在偏差。

---

## 428. TADP-RME: A Trust-Adaptive Differential Privacy Framework for Enhancing Reliability of Data-Driven Systems

**arXiv ID:** 2604.08113 | [PDF](https://arxiv.org/pdf/2604.08113v1)

**作者:** Labani Halder `[一作]` (Indian Statistical Institute), Sarbani Palit `[通讯]` (Indian Statistical Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于信任适应的差分隐私框架TADP-RME，在噪声注入的基础上加入逆流形嵌入（RME）以扰乱数据几何结构，从而提升在对抗环境下的可靠性。

**💡 创新点**

创新点在于：① 通过逆信任分数自适应调节隐私预算，实现连续可解释的隐私-效用折衷；② 设计非线性周期映射RME，故意破坏局部邻域关系，增强对几何攻击的鲁棒性；③ 将噪声注入与结构扰动结合，兼具正式差分隐私保证和实证攻击抵抗。

**🔧 技术方法**

采用了高斯差分隐私机制（Gaussian DP）与后处理的逆流形嵌入技术，整体形成两阶段隐私保护流程。

**📊 数据集**

在MNIST、Fashion‑MNIST和CIFAR‑10三大图像基准上进行实验。

**📈 对比分析**

与多种基线（固定预算Gaussian/Laplace DP、个性化DP、随机投影、LSH等）在匹配隐私预算下比较，TADP‑RME在相同或更高隐私水平下保持相近甚至更高的分类准确率，并在成员推断、属性推断和重构攻击中显著提升隐私得分（平均提升约1–3%），形成更优的隐私-效用 Pareto 前沿。

**⚠️ 局限性**

局限性包括：① 对高维数据（如CIFAR‑10）在强隐私下仍会出现显著准确率下降；② 参数设置（ε_min/ε_max、剪裁范数、α）需要经验性调优；③ RME的计算复杂度和内存占用随维度上升，实际部署需考虑效率与可扩展性。

---

## 429. Bias Redistribution in Visual Machine Unlearning: Does Forgetting One Group Harm Another?

**arXiv ID:** 2604.08111 | [PDF](https://arxiv.org/pdf/2604.08111v1)

**作者:** Yunusa Haruna `[一作]` (NewraLab), Shamsuddeen Hassan Muhammad `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究机器无学习（machine unlearning）下的偏差重分布（bias redistribution）现象，利用CLIP模型在CelebA数据集上对年轻女性群体进行零射击忘记并系统评估重分布；

**💡 创新点**

首次定义偏差重分布及其重分布分数（Redistribution Score），并揭示预训练嵌入空间中性别主导的几何结构是导致重分布的根本原因；

**🔧 技术方法**

采用零射击无学习方法（Prompt Erasure、Prompt Reweighting、Refusal Vector）、CLIP视觉语言模型、交叉群体准确率、人口公平间隙、余弦相似度分析等技术；

**📊 数据集**

使用CelebA面部属性数据集，划分四个交叉族群（年轻女性、年轻男性、老年女性、老年男性）；

**📈 对比分析**

与原始模型对比：Prompt Erasure与Prompt Reweighting实现完美忘记但导致极大重分布；Refusal Vector降低公平间隙但忘记不完全；无方法同时实现低忘记误差、高保留准确率和低重分布；

**⚠️ 局限性**

仅评估零射击无学习方法，单一数据集与单一忘记组，属性仅二元化，未考察梯度无学习方法或更复杂属性空间的重分布表现。

---

## 430. Coordinate-Based Dual-Constrained Autoregressive Motion Generation

**arXiv ID:** 2604.08088 | [PDF](https://arxiv.org/pdf/2604.08088v1)

**作者:** Kang Ding `[一作]` (Southeast University), Liang Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于坐标的双重约束自回归动作生成框架 CDAMD，能够从文本生成高保真、语义一致的人类动作，并支持动作编辑。

**💡 创新点**

创新点包括：①使用纯坐标表示代替传统混合表示；②引入双重约束因果掩码（时间与条件约束），保证自回归生成与文本语义同步；③结合运动先验（RVQ‑VAE 离散令牌）和扩散多层感知器提升生成质量；④在自回归解码后加入扩散 MLP 进行细化。

**🔧 技术方法**

技术包括：确定性 AE、残差向量量化 VAE、Transformer 自回归解码、双重约束因果注意力、扩散多层感知器、CLIP 文本编码、混合掩码策略与噪声预测等。

**📊 数据集**

数据集：HumanML3D（约 14k 动作、44k 文本）、KIT‑ML（约 3.9k 动作、6.3k 文本），均以 3D 坐标形式处理。

**📈 对比分析**

在坐标基准下与 MoMask、BAMM、MDM、MotionDiffuse 等基线比较，取得最低 FID 0.046、最高 Top‑1 R‑Precision 0.522、CLIP‑Score 0.679，显著优于其他方法。

**⚠️ 局限性**

局限：推理时长较长；对量化令牌质量依赖度高；在文本含糊或错误时仍可能生成与语义不完全一致的动作。

---

## 431. DeepForestSound: a multi-species automatic detector for passive acoustic monitoring in African tropical forests, a case study in Kibale National Park

**arXiv ID:** 2604.08087 | [PDF](https://arxiv.org/pdf/2604.08087v1)

**作者:** Gabriel Dubus `[一作]` (Muséum National d'Histoire Naturelle), Sabrina Krief `[通讯]` (Muséum National d'Histoire Naturelle)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了DeepForestSound（DFS），一种针对非洲热带雨林被动声学监测的多物种自动检测模型，并在基布莱国家公园进行验证。

**💡 创新点**

创新点在于结合半监督聚类生成大规模标注数据，并采用低秩适配（LoRA）对预训练的Audio Spectrogram Transformer进行参数高效微调，使模型在稀缺标签环境下显著提升非鸟类（灵长类、象）的检测性能。

**🔧 技术方法**

技术包括：能量阈值候选声学片段检测、预训练模型（AST、Perch v2、BirdNET）嵌入提取、UMAP+HDBSCAN聚类、人工验证、音频预处理（分帧、Mel滤波、Augmentation）、LoRA微调的AST以及与基线模型（DFS-Linear、BirdNET、Perch v2、RDet）的对比。

**📊 数据集**

数据集涵盖Sebitoli 2023（训练）与Sebitoli 2025（评估）两组录音，补充公开数据（XC、非洲灵长类、象噪声）及现场手持设备与摄像机捕获的特定物种片段，共计约46,043条标注样本及1,200分钟评估录音。

**📈 对比分析**

方法评估使用平均精度（AP）和最佳F1；DFS在12种目标物种中获胜8个，尤其在非鸟类上AP均>0.96；与DFS-Linear相比，LoRA微调提升显著；与BirdNET、Perch v2、RDet比较，DFS在非鸟类上优于所有基线，鸟类性能与基线相当或略高。

**⚠️ 局限性**

局限性包括：仅在同一森林内部时间与地点泛化，未验证跨生态系统迁移；未进行混合数据增强的消融实验；对稀有物种仍受标注不足影响；模型对音频分辨率、录音设备差异的鲁棒性尚未系统评估。

---

## 432. Exploration of Pareto-preserving Search Space Transformations in Multi-objective Test Functions

**arXiv ID:** 2604.08173 | [PDF](https://arxiv.org/pdf/2604.08173v1)

**作者:** Diedeerick Vermetten `[一作]`, Jeroen Rook `[通讯]` (Paderborn University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统地研究了在多目标优化中对搜索空间进行可逆变换（Beta‑CDF累积分布变换和球面旋转）对常用MOEA（NSGA‑II、SMSEMOA、MOEA/D）以及随机搜索性能的影响，并将这些变换与目标空间变换做对比。

**💡 创新点**

创新点在于提出了两种保持目标空间结构的、可逆的搜索空间变换方法，并通过大规模实验揭示了轴对齐偏差对算法性能的显著影响；同时为多目标实例生成提供了可复制、可扩展的变换框架。

**🔧 技术方法**

技术手段包括：Beta‑CDF变换、球面旋转变换、PyMOO实现的三种MOEA、IOHexperimenter/IOHInspector日志记录、无界存档的Hypervolume评估、随机搜索基线。

**📊 数据集**

使用的数据集为：ZDT、DTLZ 和 MMF 生成器的双目标实例，维度分别为 2 与 10，并针对每个基准应用多组参数化变换（α,β∈{0.2,0.5,1,2,5} 及四个随机旋转矩阵）。

**📈 对比分析**

比较方法为：在 10 次独立运行中记录 Hypervolume 随时间的演化，并计算最终 Hypervolume 与未变换基准的相对比例；结果显示球面旋转对 MOEA/D 造成显著性能下降，Beta‑CDF 对随机搜索影响最大，其他 MOEA 受影响相对较小。

**⚠️ 局限性**

局限性包括：只考虑了单一维度或两种变换的组合，未探索多变换交互效应；实验仅在盒约束环境下进行，可能不适用于无界或更复杂约束；变换参数范围有限，未覆盖所有可能的角度或 αβ 组合。

---

## 433. Bayesian Tendon Breakage Localization under Model Uncertainty Using Distributed Fiber Optic Sensors

**arXiv ID:** 2604.08162 | [PDF](https://arxiv.org/pdf/2604.08162v1)

**作者:** Daniel Andrés Arcones `[一作]` (Technical University of Munich), Jörg F. Unger `[通讯]` (Bundesanstalt für Materialforschung und -prüfung)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个基于贝叶斯、嵌入模型误差的框架，用分布式光纤传感器数据对预应力混凝土中筋断裂位置进行定位。

**💡 创新点**

通过将模型误差直接嵌入材料参数的随机分布，实现了可转移的模型不确定性估计，并结合ϕ-散度诊断与分辨率分析。

**🔧 技术方法**

采用有限元模拟、Gaussian Process surrogate、MCMC贝叶斯更新、分布式光纤传感器与ϕ-散度影响分析。

**📊 数据集**

使用实验室预应力混凝土梁的DFOS高分辨率应变测量数据及对应的有限元仿真结果。

**📈 对比分析**

与传统无嵌入误差的贝叶斯校准比较，嵌入模型不确定性后置信区间覆盖率提升至约87%，误差减少，定位准确率明显提高。

**⚠️ 局限性**

依赖实验数据、仅以单一随机变量表征模型误差、忽略空间/多源不确定性及传感器间相关性，导致在不同几何/尺度下转移性能受限。

---

## 434. An Illusion of Unlearning? Assessing Machine Unlearning Through Internal Representations

**arXiv ID:** 2604.08271 | [PDF](https://arxiv.org/pdf/2604.08271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 435. MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning

**arXiv ID:** 2604.08203 | [PDF](https://arxiv.org/pdf/2604.08203v1)

**作者:** Zheng Jiang `[一作]` (Tsinghua University), Minfeng Xu `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MedVR的无监督强化学习框架，通过在医学视觉语言模型中自然交替文本推理与图像工具调用，实现了可验证的视觉推理；

**💡 创新点**

核心创新在于两项无标签自监督机制：利用模型自身的熵值触发视觉再定位（EVR）进行自适应探索，并通过多条成功轨迹的共识（CCA）生成伪标签以实现信用分配；

**🔧 技术方法**

主要技术包括强化学习策略优化、基于token熵的自适应分支探索、共识热图生成与IoU阈值信用奖励，以及在Qwen2.5-VL系列模型上实现的文本与工具交互；

**📊 数据集**

使用六个公开医学VQA数据集：OmniMedVQA、PMC‑VQA、MedXpertQA（多选）以及VQA‑RAD、SLAKE、PathVQA（开放式文字回答）；

**📈 对比分析**

与多种通用和医学专用VLM（如Qwen、InternVL、Med‑R1、LLaVA‑Med等）以及文本RL基线相比，MedVR在所有基准上均实现SOTA，尤其在跨域测试中显著提升准确率；

**⚠️ 局限性**

局限性包括对单一图像工具的依赖、RL训练的高计算成本、对复杂多模态交互支持不足，以及在极端罕见病例和非标准影像上的泛化仍待进一步验证。

---

## 436. From Phenomenological Fitting to Endogenous Deduction: A Paradigm Leap via Meta-Principle Physics Architecture

**arXiv ID:** 2604.08245 | [PDF](https://arxiv.org/pdf/2604.08245v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 437. Beyond Static Forecasting: Unleashing the Power of World Models for Mobile Traffic Extrapolation

**arXiv ID:** 2604.08199 | [PDF](https://arxiv.org/pdf/2604.08199v1)

**作者:** Xiaoqian Qi `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 MobiWM——首个面向移动网络的世界模型，用以建模网络参数调整与移动流量变化之间的动态关系；

**💡 创新点**

创新点在于将流量预测从静态时间序列回归转变为动作条件下的状态转移建模，结合因子化时空块与多模态共享空间语义的融合机制；

**🔧 技术方法**

技术包括基于 Transformer 的因子化时空块（FSTBlock）、拓扑感知注意力、共享 Fourier 位置信息、可学习门控融合以及图批处理与细胞掩码；

**📊 数据集**

使用了新构造的 Nanchang 城市七天 15 分钟粒度流量数据，覆盖 31,900 个基站，包含可变参数（功率、方位角、机械/电机倾斜）的仿真数据；

**📈 对比分析**

与 FedGTP、HiSTM、MobiFM 等传统流量预测模型以及 iTransformer、Informer、TimeMoE、CSDI、TD-MPC2、STORM、DreamerV3 等通用世界模型进行对比，MobiWM 在 JSD、MAE、NRMSE 等度量上持续领先，且在不同场景下鲁棒性更强；

**⚠️ 局限性**

局限性包括对极端离散参数变化的泛化仍受限于训练数据范围、模型规模对参数设置的敏感度以及对大规模网络部署时的计算资源需求。

---

## 438. DBMF: A Dual-Branch Multimodal Framework for Out-of-Distribution Detection

**arXiv ID:** 2604.08261 | [PDF](https://arxiv.org/pdf/2604.08261v1)

**作者:** Jiangbei Yue `[一作]` (University of Leeds), Sharib Ali `[通讯]` (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了双分支多模态框架 DBMF，用于医学内镜图像的离散分布（OOD）检测。

**💡 创新点**

创新点在于同时结合文本-图像对齐分支和纯视觉分支，并提出文本分离对比损失 L_TSC 来提升文本特征的区分性。

**🔧 技术方法**

采用 CLIP 风格的图像与文本编码器、交叉熵与对比损失、马氏距离、标准化融合等技术实现双分支训练与 OOD 得分融合。

**📊 数据集**

使用公开的 Kvasir‑v2 与 GastroVision 两个内镜图像数据集进行实验。

**📈 对比分析**

与 11 种 SOTA 基线在 AUROC 与 FPR95 上对比，DBMF 在两数据集均获得最高 AUROC，尤其在 GastroVision 上提升 3.81% AUROC、24.84% FPR95。

**⚠️ 局限性**

局限性包括依赖固定提示模板、未利用大型语言模型或医学专用 CLIP 预训练模型、以及仅在两数据集上验证，缺乏更广泛的泛化评估。

---

## 439. Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models

**arXiv ID:** 2604.08266 | [PDF](https://arxiv.org/pdf/2604.08266v1)

**作者:** Jing Gu `[一作]` (Eindhoven University of Technology), Gijs Dubbelman `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过将大型语言模型（LLM）的推理能力蒸馏到轻量化的Transformer解码器中，构建了一款名为Orion‑Lite的视觉端到端自动驾驶模型；

**💡 创新点**

创新点在于：① 在复杂闭环评估环境下实现LLM知识蒸馏，且蒸馏与真实轨迹监督联合使用，最终使轻量模型性能超越教师；② 通过仅使用视觉输入，完全去除了LLM模块，显著降低推理延迟和显存占用；

**🔧 技术方法**

采用的技术包括：视觉编码器（EVA‑02‑L）+ QT‑Former时序模块，LLM（Vicuna‑v1.5）蒸馏至6层Transformer解码器，ℒ1特征蒸馏损失，联合训练的轨迹监督（碰撞、边界、KLD等），以及在CARLA Bench2Drive闭环评估中多指标评估；

**📊 数据集**

使用Bench2Drive数据集（CARLA V2）进行训练和闭环评估，训练集包含1000条视频片段，测试集220条路线，涵盖44种交互场景；

**📈 对比分析**

与ORION（7B教师）以及其他最新的VLA、RL、WM模型（MindDrive、UniDrive‑WM等）进行对比；在Bench2Drive闭环评估中，Orion‑Lite达成Driving Score 80.6，成功率55.5%，相较教师提升+2.9 DS、+0.9 SR、+5.8平均多能力分；推理延迟下降3×，显存由31GB降至8GB；

**⚠️ 局限性**

局限性包括：① 仍依赖先前训练好的大型VLA教师模型作为蒸馏源；② 主要验证在Bench2Drive上，未在其他多样化数据集上证明泛化；③ 视觉编码器仍为大型模型，成为整体推理瓶颈；

---

## 440. Scheduling Coflows in Multi-Core OCS Networks with Performance Guarantee

**arXiv ID:** 2604.08242 | [PDF](https://arxiv.org/pdf/2604.08242v1)

**作者:** Xin Wang `[一作]` (Central Queensland University), Dong Wang `[通讯]` (Central Queensland University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究多核光电路交换网络（OCS）在不全停式重配置模型下的多流coflow调度问题，提出一种联合流量分配与电路调度的近似算法，并给出最优性保证。

**💡 创新点**

创新点在于首次为多核OCS网络提供具有证明的近似调度框架，利用全局与局部下界实现2Mψ近似，并可直接迁移至多核EPS网络。

**🔧 技术方法**

采用下界推导、前缀负载分析、贪心流分配、工作守恒的非抢占电路调度以及数学证明与仿真验证等技术。

**📊 数据集**

实验使用Facebook数据中心工作负载追踪（526个coflow）生成N×N需求矩阵，并随机选取服务器映射。

**📈 对比分析**

通过与基于ρ分配、随机分配、Sunflow单核调度等多种基线对比，使用总加权CCT和尾部CCT指标，实验表明在所有设置下所提算法显著降低CCT，尤其在不平衡核心速率时优势最突出。

**⚠️ 局限性**

限制在于仅针对批量到达且已知需求矩阵的离线场景，未考虑在线到达、动态可观测性与重配置延迟的实际波动；对权重分布极端不均时近似比率可能受影响。

---

## 441. $\oslash$ Source Models Leak What They Shouldn't $\nrightarrow$: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization

**arXiv ID:** 2604.08238 | [PDF](https://arxiv.org/pdf/2604.08238v1)

**作者:** Arnav Devalapally `[一作]` (Indian Institute of Technology), Vineeth N. Balasubramanian `[通讯]` (Indian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在源域模型不暴露源数据的前提下，提出在目标域自适应过程中同时完成对源域特有类别的遗忘（SCADA-UL）及其两种变体（C‑SCADA‑UL 与 UC‑SCADA‑UL）的方法。

**💡 创新点**

创新点在于：①在源域自适应过程中将遗忘与自适应并行进行；②使用对抗样本生成并通过“重新缩放标签”策略对遗忘类别进行精细化遗忘；③通过理论分析和实验验证了该策略在减少遗忘与自适应冲突方面的有效性。

**🔧 技术方法**

技术上结合了源域无监督自适应（SFDA）框架（如 SF(DA)^2）、对抗优化、重标记（rescaled labeling）与交叉熵损失，整体目标为最小化遗忘损失与自适应损失的加权和。

**📊 数据集**

实验使用了多个公开域自适应数据集：OfficeHome、Office31、DomainNet‑126、医学影像数据集 CheXpert→NIH Chest X‑ray 以及土地利用数据集。

**📈 对比分析**

与原始 SFDA、重训练、微调、现有遗忘方法（UNSIR、ZSMU、Lipschitz、Nabla Tau 等）以及部分域自适应方法（PADA、SHOT 等）进行对比，实验表明该方法在目标保留类别准确率与遗忘类别准确率上均接近重训练基准，且在会员推断攻击（MIA）准确率上表现更佳。

**⚠️ 局限性**

局限性包括：①依赖已训练好的源域模型；②对对抗样本生成与重标记策略的超参数需要经验调优；③在高度分布偏移或极端类别不平衡时可能需要进一步改进；④未验证对所有类型域自适应场景（如多源、连续学习）的通用性。

---

## 442. HiRO-Nav: Hybrid ReasOning Enables Efficient Embodied Navigation

**arXiv ID:** 2604.08232 | [PDF](https://arxiv.org/pdf/2604.08232v1)

**作者:** He Zhao `[一作]` (Nanyang Technological University), Chunyan Miao `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种能够自适应决定何时进行推理的导航代理 HiRO‑Nav，并在长程目标导航任务中实现高效、准确的决策。

**💡 创新点**

创新点在于：①基于动作熵分析发现仅在高熵动作时需要推理；②提出混合推理策略，仅在满足阈值时激活 CoT 推理；③设计了两阶段在线强化学习流程，先训练无推理能力再训练推理能力，避免模式崩溃。

**🔧 技术方法**

技术包括：大型推理模型（如 Qwen2.5‑VL‑3B）、动作熵度量与阈值触发、混合监督微调（Hybrid Supervised Fine‑Tuning）、PPO 强化学习、注释语义图（ASM）作为长期记忆、KL 正则化保持无推理能力。

**📊 数据集**

使用的数据集：CHORE‑S ObjectNav 基准、HRD（高熵推理数据集）以及包含 ASM 的 AI2‑Thor 3D 环境。

**📈 对比分析**

对比方法包括：无推理、每 K 步推理、密集推理以及 GPT‑4o、o3、Gemini2.5‑Pro 等强大 VLM；在 CHORE‑S 上，HiRO‑Nav 的成功率 81%、SEL 57.2% 超过所有基线，且 token 效率与无推理相当，远优于密集推理。

**⚠️ 局限性**

局限性：①使用固定熵阈值，无法自适应不同场景；②在预测动作后才触发推理，导致额外延迟；③依赖 ASM 的准确性，噪声会影响性能。

---

## 443. OmniJigsaw: Enhancing Omni-Modal Reasoning via Modality-Orchestrated Reordering

**arXiv ID:** 2604.08209 | [PDF](https://arxiv.org/pdf/2604.08209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. Generalization Under Scrutiny: Cross-Domain Detection Progresses, Pitfalls, and Persistent Challenges

**arXiv ID:** 2604.08230 | [PDF](https://arxiv.org/pdf/2604.08230v1)

**作者:** Saniya M. Deshmukh `[一作]` (University of Beira Interior), Hugo Proença `[通讯]` (University of Beira Interior)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对跨域目标检测（CDOD）领域进行了全面系统的综述，提出了统一的理论框架、正式问题定义、管线级分析、失败模式和六轴概念分类法，旨在为研究者提供清晰的结构化视角；

**💡 创新点**

创新点在于①将CDOD建模为多阶段耦合的受限优化问题，明确了提议覆盖、特征判别性和校准三大不变量；②提出概率管线分解，揭示域移位如何在提议、特征与检测头之间传播；③设计六轴分类维度（适配范式、模型假设、管线部件等），突出方法的空白与设计压缩；④系统讨论了七大核心挑战与失败模式，强调mAP单指标的局限性；

**🔧 技术方法**

使用的技术主要包括：形式化目标函数（带三大不变量约束）；概率管线分解公式；六轴概念分类法与诊断矩阵；stage‑wise 诊断指标（提议召回、分类准确、定位误差、校准误差）；多任务和自监督训练策略的理论剖析；以及对抗性、鲁棒性与因果推理等不同范式的比较；

**📊 数据集**

评估数据集涵盖了从常规到极端的多种域移位：PASCAL VOC、MS COCO、ImageNet DET、Cityscapes、Foggy Cityscapes、SIM10K、GTA5、SYNTHIA、BDD100K、Dark Zurich、KITTI、nuScenes、Waymo Open 等，覆盖合成‑to‑real、天气、光照、传感器、场景、目标稀疏度等多重维度；

**📈 对比分析**

文章对比了数百种方法，指出绝大多数聚焦于特征对齐、隐式目标、闭集假设，并通过 mAP 评价表面上实现提升；但对提议质量、校准、类别不匹配等关键指标未作充分评估，导致方法在更强域移位（如尺度、上下文、开集等）下表现不稳；

**⚠️ 局限性**

局限性包括：①研究过度集中在对齐与合成‑to‑real 任务，缺少对真实‑to‑真实、长尾与开集等更复杂场景的系统探讨；②依赖 mAP 单指标，忽略了阶段诊断与校准误差的细粒度评估；③对因果与稳健性范式的探索不足，导致方法在面临未知或极端域移位时易失效；④未提供统一的基准与多域评估流程，影响方法可比性与可迁移性。

---

## 445. Quantum Integrated Communication and Computing Over Multiple-Access Bosonic Channel

**arXiv ID:** 2604.08214 | [PDF](https://arxiv.org/pdf/2604.08214v1)

**作者:** Ioannis Krikidis `[一作]` `[通讯]` (University of Cyprus), Ioannis Krikidis (University of Cyprus)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种量子集成通信与计算（QICC）方案，利用单模玻色子多址信道实现接收端同时进行过空计算和多址数据解码。

**💡 创新点**

创新点包括：① 将经典ICC迁移至量子光学域；② 在相干态调制下设计联合功率控制与接收系数的非凸优化；③ 通过交替优化结合LMMSE、单维二分和投影梯度实现低复杂度收敛。

**🔧 技术方法**

采用量子异步检测、线性最小均方误差（LMMSE）估计、Von Neumann熵率约束以及交替优化框架（投影梯度+二分法）。

**📊 数据集**

实验基于数值仿真，使用的参数包括：K、M、η_k=0.6/K、η_{K+m}=0.4/M、P_c=P_t、N_0=2 等，并未使用真实数据集。

**📈 对比分析**

通过仿真比较不同(K,M)和功率配置下的MSE–sum‑rate性能，展示了计算与通信的权衡；交替优化算法收敛迅速，MSE随sum‑rate递增，更多通信设备或更大功率可提高性能。

**⚠️ 局限性**

局限性：仅考虑相干态调制与异步检测，未引入纠缠或压缩等非经典资源；只分析单模纯相干信号，未考虑硬件失真或非线性；未与经典ICC系统做定量对比。

---

## 446. Vision-Language Foundation Models for Comprehensive Automated Pavement Condition Assessment

**arXiv ID:** 2604.08212 | [PDF](https://arxiv.org/pdf/2604.08212v1)

**作者:** Blessing Agyei Kyem `[一作]` (North Dakota State University), Armstrong Aboah `[通讯]` (North Dakota State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于指令调优的视觉‑语言模型（PaveGPT），能够通过自然语言完成道路状况评估、缺陷定位、PCI 计算和维护建议等多项任务。

**💡 创新点**

创新点在于：① 构建了规模达 278,889 条图像‑指令‑响应对的 PaveInstruct 数据集，覆盖 32 种任务类型；② 设计了统一多源数据集融合与坐标标准化的四阶段流水线；③ 在 PaveGPT 中引入 ASTM D6433 标准化约束的指令模板，实现在推理链、结构化输出和标准合规性方面的高质量表现。

**🔧 技术方法**

技术主要包括：多模态视觉‑语言模型（Qwen2.5‑VL 变体）、指令调优（使用 GPT‑4/LLM 生成训练样本并进行人工审核）、跨模态投影层、文本到坐标/数值的自回归生成以及 LLM‑as‑Judge 评估。

**📊 数据集**

使用了九个现有道路缺陷数据集（PID、PaveTrack、UAPD、DSPS23/24、RDD2022、SVRDD、UAV‑PDD2023、PCIER）以及多源标签（目标框、分割、PCI、严重度等）作为原始注释源。

**📈 对比分析**

在零样本和指令调优两种设置下与多种前沿 VLM（LLaVA、LLaMA、MiniCPM、InternVL 等）进行对比。指令调优后，PaveGPT 在空间定位、推理链、生成任务的平均提升超过 20%（mIoU、R²、BLEU‑4、ROUGE‑L、CIDEr、MAE 等指标均显著改善），并在 LLM‑Judge 评分中达到 8‑10 分的高质量水平。

**⚠️ 局限性**

局限性包括：① 数据主要来自特定地区，需本地化再训练；② 缺乏时间序列标签，无法进行衰退预测；③ PCI 误差仍在 13–31 点之间，仍适合作为预筛工具而非最终决策依据；④ 需要在真实工作流中验证可信度和与现有系统的集成。

---

## 447. Empirical Evaluation of Taxonomic Trace Links: A Case Study

**arXiv ID:** 2604.08207 | [PDF](https://arxiv.org/pdf/2604.08207v1)

**作者:** Waleed Abdeen `[一作]` (Blekinge Institute of Technology), Krzysztof Wnuk `[通讯]` (Blekinge Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Ericsson 通信产品进行案例研究，利用 LLM 生成领域分类法并用零样本分类器对需求、测试用例和标准进行分类，从而构建并评估税onomic trace links (TTL)。

**💡 创新点**

将 LLM 自动生成的领域分类法与零样本多标签分类相结合，使得在缺乏预先定义分类法的行业环境中仍可实现 TTL，并首次在工业现场实证其可行性与价值。

**🔧 技术方法**

使用 ChatGPT‑4o 生成分类法，采用 Sentence‑T5‑XL 与 All‑MiniLM‑L12‑v2 进行零样本多标签分类，并通过混合方法（案例研究、焦点小组、定量评估）进行验证。

**📊 数据集**

数据集包括 463 条业务用例、64 条测试用例、277 条 ISO/3GPP 标准文档，以及 Ericsson 的专有需求与测试工件，3GPP 文档用于为 LLM 提供上下文。

**📈 对比分析**

通过对比两种语言模型的精确度、召回率与 F1 分数，并结合焦点小组的主观反馈；All‑MiniLM‑L12‑v2 在 LC=2 时召回率达 91% 但精确度仅约 1%，提示仍需人工校正，但能显著减少候选链接量；总体上功能性满足需求但仍需改进。

**⚠️ 局限性**

主要局限包括分类器低精确度导致大量误报、对专家主导的分类法构建与维护的依赖、工件结构异质性带来的匹配挑战，以及单一公司案例研究的可推广性受限。

---

## 448. Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings

**arXiv ID:** 2604.08192 | [PDF](https://arxiv.org/pdf/2604.08192v1)

**作者:** Yunxiang Peng `[一作]` (University of Delaware), Xi Peng `[通讯]` (University of Delaware)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出利用模型内部电路结构作为无标签环境下的泛化性能预测指标，分别用于部署前的模型选择和部署后的性能监测；

**💡 创新点**

创新点在于将机制解释中的电路发现结果转化为可量化的泛化度量——Dependency Depth Bias（DDB）和Circuit Shift Score（CSS），并证明其比传统基于输出或ID特征的代理指标更能捕捉分布漂移下的性能变化；

**🔧 技术方法**

主要技术包括对Vision Transformer计算图进行连续化电路权重定义、EAP‑IG电路发现方法、基于互层依赖矩阵的CCA分析得到通用泛化电路模式、以及基于电路权重向量或图谱距离的CSS构造；

**📊 数据集**

在PACS、Camelyon17、Terra Incognita、ImageNet等多域视觉数据集上进行实验，构建不同域的训练/测试任务以及多种分布漂移场景；

**📈 对比分析**

与ID、OOD输出概率、特征质量以及模型比较度量等现有代理指标对比，DDB在模型选择任务中平均提升R²、Spearman、Kendall相关性分别约13.4%，CSS在性能监测任务中平均提升相关性约34.1%，并在报警F1曲线中比最佳基线提升约45%；

**⚠️ 局限性**

主要局限是电路发现的计算成本高，尤其在实时部署后监测时难以满足实时性需求；

---

## 449. Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling

**arXiv ID:** 2604.08178 | [PDF](https://arxiv.org/pdf/2604.08178v1)

**作者:** Jiaxuan Wang `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了面向工具集成智能体的轨迹级偏好评估基准，并构建了多源硬负样本生成流程；

**💡 创新点**

将评估焦点从单个回复扩展到完整对话+工具调用轨迹，并提供多样化硬负样本与对齐策略，为奖励模型训练和RLHF提供更合适的信号；

**🔧 技术方法**

利用多模型自然rollout、规则注入、最小编辑扰动、LLM多评审与元复审，以及pairwise评估协议，训练并评估DRM、GRM和通用LLM评审者；

**📊 数据集**

基于Toucan/MCP工具注册表生成的轨迹，结合自然rollout与扰动样本，覆盖Safety Refusal、Tool‑Irrelevance/Unavailability、Complex Planning、Robust Error Recovery四类任务；

**📈 对比分析**

采用pairwise准确率与多维拆分（单/多步、易/难）进行对比；最佳模型Qwen‑Plus约69.96%，但在多步硬任务上仍低于70%，性能随轨迹长度显著下降；

**⚠️ 局限性**

标签主观性、工具注册表覆盖有限、样本分布不均、仅英文文本、缺乏多模态与多代理扩展等限制。

---

## 450. When to Trust Tools? Adaptive Tool Trust Calibration For Tool-Integrated Math Reasoning

**arXiv ID:** 2604.08281 | [PDF](https://arxiv.org/pdf/2604.08281v1)

**作者:** Ruotao Xu `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了工具集成推理（TIR）中模型忽略正确工具输出的现象（“Tool Ignored”），并提出了自适应工具信任校准框架（ATTC），帮助模型根据生成代码的置信度决定是否信任工具结果。

**💡 创新点**

创新点在于将模型生成代码的置信度作为判断工具可信度的量化指标，并通过阈值机制注入信任或反思控制信号，显著降低了模型忽略正确工具输出的错误。

**🔧 技术方法**

使用的技术包括：置信度评估（token 最高概率的几何平均）、阈值校准、工具调用监控、RL 训练的 TIR 模型、代码执行环境以及 ReAct/LLM 交互等。

**📊 数据集**

使用的评测数据集为 MATH‑500、Minerva、Olympiad、AIME24 以及 AMC23 等数学推理基准。

**📈 对比分析**

通过与多种开源 TIR 模型（如 ToRL‑7B、VerlTool‑7B、SimpleTIR‑32B 等）在上述五个基准上进行 Pass@1/Pass@32、Token Count 与 Time Use 的对比实验，ATTC 在准确率上提升 4.1%~7.5%，并在大多数场景下降低 token 与时间消耗。

**⚠️ 局限性**

局限性包括：仅测试规模 ≤32B 的模型；未扩展到搜索引擎工具场景；优化仅在推理阶段进行，未在训练阶段根治；对模型幻觉仍有一定的限制。

---

## 451. Floating or Suggesting Ideas? A Large-Scale Contrastive Analysis of Metaphorical and Literal Verb-Object Constructions

**arXiv ID:** 2604.08275 | [PDF](https://arxiv.org/pdf/2604.08275v1)

**作者:** Prisca Piccirilli `[一作]`, Sabine Schulte im Walde `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了近200万句包含297对比喻与字面动词-宾语结构的数据集，对其进行大规模的语义、句法、情感及词汇多样性等2,293个特征的提取与对比分析，探讨两种表达在语境中的差异；

**💡 创新点**

创新点在于首次系统量化比喻与字面表达的跨配对与内配对差异，并通过多维特征聚合证明比喻现象在构式层面而非单一动词层面呈现变异；

**🔧 技术方法**

主要采用SEANCE、TAALES、TAASSC、TAACO、TAALED等五种NLP工具进行特征提取，并用z‑score标准化、Wilcoxon符号秩检验和平均绝对差阈值分析；

**📊 数据集**

数据集来源为公开的VOLIMET（297对VO）与ENCOW语料库（约200万句），同时构造了含比喻/字面对应句子的大规模语料；

**📈 对比分析**

通过跨配对统计差异与内配对平均绝对差阈值≥1标准差的方法，发现字面表达在词频、连贯性与句法稳定性上占优势，比喻表达在情感强度、感官化与词汇多样性上更突出；

**⚠️ 局限性**

局限性包括仅限英语、语料单一来源、仅覆盖297对VO且特征聚合可能掩盖细粒度效应

---

## 452. Self-Debias: Self-correcting for Debiasing Large Language Models

**arXiv ID:** 2604.08243 | [PDF](https://arxiv.org/pdf/2604.08243v1)

**作者:** Xuan Feng `[一作]` (Jinan University), Bo An `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法获取具体研究内容

**💡 创新点**

无法获取创新点

**🔧 技术方法**

无法获取技术手段

**📊 数据集**

无法获取数据集

**📈 对比分析**

无法获取比较方法与性能

**⚠️ 局限性**

无法获取局限性

---

## 453. "Theater of Mind" for LLMs: A Cognitive Architecture Based on Global Workspace Theory

**arXiv ID:** 2604.08206 | [PDF](https://arxiv.org/pdf/2604.08206v1)

**作者:** Wenlong Shang `[一作]` `[通讯]` (Beijing University of Technology), Wenlong Shang (Beijing University of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种全新的多代理架构——Global Workspace Agents (GWA)，通过中心广播 hub 与多功能异质代理的协同，打破传统 LLM 被动响应的 BIBO 结构，形成持续自主推理的循环；

**💡 创新点**

创新点在于将 Global Workspace Theory 转化为可执行的事件驱动架构，加入熵驱动的内在探索机制动态调节生成温度以打破认知死锁，并实现双层内存分裂与语义压缩保证长时序推理；

**🔧 技术方法**

采用事件驱动广播、检索增强生成 (RAG)、温度调节的生成与批判子模块、Shannon 熵计算的内在驱动、双层短期/长期记忆压缩、以及以第一人称框架的指令集；

**📊 数据集**

未使用公开数据集，主要以理论设计与内部模拟验证为主；

**📈 对比分析**

由于缺乏实验实现，未给出定量性能指标，仅通过仿真描述说明该架构能维持连续思考、避免同质化死锁，并与传统单代理链式思考进行概念对比；

**⚠️ 局限性**

局限包括：依赖 LLM 的输出质量，缺乏实测性能；事件驱动和多代理协同导致计算开销高；熵调温度机制需要调参，可能在不同任务中表现不一；整体架构复杂度高，部署与调试困难。

---

## 454. Competitive Transaction Admission in PCNs: Online Knapsack with Positive and Negative Items

**arXiv ID:** 2604.08205 | [PDF](https://arxiv.org/pdf/2604.08205v1)

**作者:** Marcin Bienkowski `[一作]` (University of Wrocław), Stefan Schmid `[通讯]` (TU Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并解决了支付通道网络（PCN）中在线交易通过单通道的吞吐量最大化问题，等价于带正负项目的在线背包问题；

**💡 创新点**

提出了一种新的阈值策略的确定性算法，证明其竞争比为O(log B)并给出匹配的Ω(log B)下界，证明该算法在此模型下是最优的；

**🔧 技术方法**

使用竞争分析、潜能函数、指数阈值函数与递推序列U_n的分析技术；

**📊 数据集**

在随机交易规模（区间[1,B]或[1,B/ln B]）以及实际的Lightning Network路由拓扑数据集上进行实验；

**📈 对比分析**

与常用贪心算法进行对比，实验表明该算法在随机交易场景下与贪心表现相近，且在任何情况下都能保证最小吞吐量，整体性能优于贪心的最坏情况；

**⚠️ 局限性**

局限在仅考虑单通道模型，假设最大项目规模m≤B/ln B才能获得理论保证，且在多通道网络中的路由和状态交互未被直接建模。

---

## 455. Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering

**arXiv ID:** 2604.08224 | [PDF](https://arxiv.org/pdf/2604.08224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 456. Approximation of the Basset force in the Maxey-Riley-Gatignol equations via universal differential equations

**arXiv ID:** 2604.08194 | [PDF](https://arxiv.org/pdf/2604.08194v1)

**作者:** Finn Sommer `[一作]` (Hamburg University of Technology), Daniel Ruprecht `[通讯]` (Hamburg University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该论文提出一种利用通用微分方程（UDE）通过神经网络逼近Maxey‑Riley‑Gatignol方程中的Basset历史项，将其转化为可用常规ODE求解器求解的形式。

**💡 创新点**

创新点在于将Basset积分项用可学习的神经网络替代，从而实现数据驱动的近似，避免了传统积分求解的高昂计算成本。

**🔧 技术方法**

采用了前馈神经网络和LSTM两种架构，并使用Julia的DifferentialEquations包和Lux框架进行训练与求解。

**📊 数据集**

训练数据来源于两种流场：解析三维旋涡场和实验测得的搅拌槽流场，分别生成了不同规模的轨迹集。

**📈 对比分析**

通过与完整MaRGE解、忽略历史项以及不同网络架构的对比，实验显示UDE在单轨迹、聚集模式以及Basset力近似上均比忽略历史项提升约两位数（或一位数）精度，并能在训练区间外约3–6倍时间内保持较低误差。

**⚠️ 局限性**

限制在于对积分项的逼近依赖于训练数据的丰富度与ODE求解精度，且在训练时间之外误差会随时间增长，尤其在强湍流或长时间预测时需谨慎使用。

---

## 457. Wattlytics: A Web Platform for Co-Optimizing Performance, Energy, and TCO in HPC Clusters

**arXiv ID:** 2604.08182 | [PDF](https://arxiv.org/pdf/2604.08182v1)

**作者:** Ayesha Afzal `[一作]` (Erlangen National High Performance Computing Center), Gerhard Wellein `[通讯]` (Friedrich-Alexander-Universität)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了Wattlytics，一个交互式浏览器平台，用来在GPU加速的HPC集群中同时评估性能、功耗和多年的总拥有成本（TCO），支持多种GPU架构、工作负载、预算与功率约束的场景探索与敏感性分析。

**💡 创新点**

创新点在于首次将基准驱动的GPU性能缩放模型、DVFS感知的功耗模型和多年的TCO分析整合到一个透明、可交互的系统中，并提供工作量/ TCO、功耗/ TCO 等多维决策指标，能够在不同行业约束下发现非直观的最优部署方案。

**🔧 技术方法**

使用技术包括：基于GPGPU-Sim/Accel-Sim的周期级GPU功耗建模；分段线性-二次的频率功耗拟合；加速器性能缩放曲线的经验拟合；多年TCO公式（资本+运营）与PUE、能耗折现等；Sobol、弹性与蒙特卡洛的敏感性分析；Python/JavaScript前后端实现、可视化与共享功能。

**📊 数据集**

数据集主要是从真实HPC中心收集的GROMACS（6个基准）和AMBER（11个基准）GPU加速运行结果，覆盖不同GPU架构（GH200、H100、L40S、L40、A40、A100、L4）与频率/功率限制下的吞吐量与功耗；此外使用公开的GPU规格表与市场价格数据。

**📈 对比分析**

通过九个案例研究（固定预算、功率、性能或GPU数量约束，以及不同系统设计和能源价格情景），与传统单维TCO/性能工具对比，Wattlytics能够展示在预算或能耗受限时低功耗GPU在长期工作量/ TCO方面优于高端GPU的逆转现象；在多GPU效率低下时高性能GPU重新占优。性能上，平台在5年周期内可将工作量/ TCO提升数倍，展示了系统级的成本效益改进。

**⚠️ 局限性**

局限性包括：对多GPU缩放的效率假设仍不完整，缺乏对弱规模/多任务混合工作负载的完整建模；依赖已收集的基准数据，无法实时覆盖新兴GPU架构或新的AI/ML工作负载；TCO模型中对碳足迹与嵌入式制造排放的估算尚未实现；平台目前仅支持NVIDIA GPU，尚未扩展至AMD/Intel及混合节点环境。

---

## 458. ACF: A Collaborative Framework for Agent Covert Communication under Cognitive Asymmetry

**arXiv ID:** 2604.08276 | [PDF](https://arxiv.org/pdf/2604.08276v1)

**作者:** Wansheng Wu `[一作]` (Beijing University of Posts and Telecommunications), Linna Zhou `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究生成式隐写在自适应智能体网络中的协作隐蔽通信，解决认知不对称导致的同步失效。

**💡 创新点**

提出Asymmetric Collaborative Framework (ACF)，通过解耦统计层与认知层，实现前缀无关、同步无需求的隐写，并给出可证明的错误界限和有效信息容量。

**🔧 技术方法**

采用词表划分与CDF置换的统计层、基于共享密钥的伪随机生成器、模型无关的前缀无关解码、Hoeffding不等式阈值检验以及有效信息容量(EIC)评估。

**📊 数据集**

实验使用 Qwen2.5 7B Instruct 与 LongMemEval_s 数据集，并在 Normal+RET 等检索增强环境下进行验证。

**📈 对比分析**

与传统对称方法 DISCOP、METEOR 比较时，ACF 在认知不对称环境下 BER 0%（或 2.47%），EIC 约 1.18 bits/10³ tokens，语义质量与检测准确率保持不变，性能显著优于基线。

**⚠️ 局限性**

限制在于单词级隐藏容量相对较低，需预共享配置；实验主要聚焦检索增强任务，理论上对更大模型或多位元隐写仍需进一步研究。

---

## 459. Neural-Symbolic Knowledge Tracing: Injecting Educational Knowledge into Deep Learning for Responsible Learner Modelling

**arXiv ID:** 2604.08263 | [PDF](https://arxiv.org/pdf/2604.08263v1)

**作者:** Danial Hooshyar `[一作]` (Tallinn University), Roger Azevedo `[通讯]` (University of Central Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种结合符号知识与深度学习的知识追踪模型 Responsible-DKT。

**💡 创新点**

创新点在于将教育领域的掌握与非掌握规则注入可微分的逻辑程序，并通过 PyNeuraLogic 实现递归神经网络与符号规则的联合训练。

**🔧 技术方法**

采用的技术包括 Lifted Relational Neural Networks (PyNeuraLogic)、RNN/GRU、可微分逻辑规则以及梯度归因解释。

**📊 数据集**

使用了 2021 年 9 月在爱沙尼亚 Opiq 平台收集的 6 年级数学交互日志（约 21471 条记录，167 名学生）。

**📈 对比分析**

与基线（无规则的神经符号模型和纯 DKT）进行对比，Responsible-DKT 在 AUC、准确率、召回率、F1 以及序列稳定性指标上均提升 5–13%，并在 10% 训练数据时已达 0.80 AUC。

**⚠️ 局限性**

局限在于规则设计相对简单，仍需要人工定义；对更复杂知识结构的可扩展性未充分验证；实验仅限于单一学科与平台，泛化性待进一步评估。

---

## 460. Preventing Overfitting in Deep Image Prior for Hyperspectral Image Denoising

**arXiv ID:** 2604.08272 | [PDF](https://arxiv.org/pdf/2604.08272v1)

**作者:** Panagiotis Gkotsis `[一作]` (Athena Research and Innovation Center), Athanasios A. Rontogiannis `[通讯]` (Athena Research and Innovation Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种结合 Smooth ℓ1 数据拟合与基于 SURE 散度的敏感性正则化，并在 Deep Hyperspectral Image Prior (DHIP) 框架中对网络参数和输入同时优化的方案，用以抑制 HSI 去噪过程中的过拟合；

**💡 创新点**

创新点在于将 Robust Smooth ℓ1 损失与 SURE 启发的散度正则化相结合，并在联合输入优化下实现对估计器敏感度的显式约束，从而在无需提前停止的情况下显著降低过拟合；

**🔧 技术方法**

技术包括 Deep Image Prior (DIP)/DHIP（U‑Net 架构）、Smooth ℓ1 损失、Stein’s Unbiased Risk Estimator（SURE）散度正则化、Monte Carlo 估计、网络输入联合优化；

**📊 数据集**

使用 Washington DC Mall HSI (200×200×191) 与 Salinas HSI (200×200×204) 两个真实 HSI 数据集，分别在 Gaussian、sparse、stripe 以及混合噪声场景下进行实验；

**📈 对比分析**

与现有的 SURE‑DHIP 和 HLF‑DHIP 两种 DIP 方法在 MPSNR 与 MSSIM 指标下对比，实验表明所提方法在所有噪声条件下均取得更高的 MPSNR/MSSIM，且不需要早停；

**⚠️ 局限性**

局限性包括：需要手动估计噪声方差；仅在两个数据集上验证，对更大尺寸或不同光谱分辨率的 HSI 的推广性尚待进一步研究。

---

## 461. Grounding Clinical AI Competency in Human Cognition Through the Clinical World Model and Skill-Mix Framework

**arXiv ID:** 2604.08226 | [PDF](https://arxiv.org/pdf/2604.08226v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 462. Behavior-Aware Item Modeling via Dynamic Procedural Solution Representations for Knowledge Tracing

**arXiv ID:** 2604.08260 | [PDF](https://arxiv.org/pdf/2604.08260v1)

**作者:** Jun Seo `[一作]` (POSTECH), Gary Geunbae Lee `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了基于学习者行为的项目建模框架BAIM，通过拆解题目求解过程并利用RLM提取四阶段嵌入来生成自适应项目表示，以提升知识追踪性能。

**💡 创新点**

结合Polya四阶段求解框架与RLM的内部嵌入轨迹，提出阶段级表示与上下文条件路由机制，实现对不同学习者动态调整项目表示的能力。

**🔧 技术方法**

使用大型语言模型Qwen3-VL-32B-Thinking生成求解过程，层级均值池化与全层平均池化提取阶段向量；通过GRU编码学习者历史，采用门控顶层与MPL进行上下文条件路由；并与多种KT骨干网络相结合。

**📊 数据集**

XES3G5M与NIPS34两大数学学习数据集，分别包含项目内容与交互历史。

**📈 对比分析**

在五个主流KT模型上与随机、PEBG、KCQRL等基线做AUC对比，BAIM在两数据集均取得最高AUC，尤其在重复尝试与样本不足情境下提升1–1.6个百分点。

**⚠️ 局限性**

仅针对数学问题求解；对缺乏项目元数据的其他数据集适用性受限；需要依赖RLM生成阶段信息，难以迁移至非文本领域。

---

## 463. Co-design for Trustworthy AI: An Interpretable and Explainable Tool for Type 2 Diabetes Prediction Using Genomic Polygenic Risk Scores

**arXiv ID:** 2604.08217 | [PDF](https://arxiv.org/pdf/2604.08217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 464. EvoGymCM: Harnessing Continuous Material Stiffness for Soft Robot Co-Design

**arXiv ID:** 2604.08258 | [PDF](https://arxiv.org/pdf/2604.08258v1)

**作者:** Le Shen `[一作]` (Zhejiang University), Huaping Liu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EvoGymCM基准，扩展软体机器人设计空间，引入连续材料刚度作为第一类设计变量，并提出Reactive和Invariant两种协同设计范式。

**💡 创新点**

核心创新是将连续材料刚度视为可优化变量并实现两种实时和固定材料策略，突破传统EvoGym离散材料限制，显著提升设计灵活性与性能。

**🔧 技术方法**

结合基于物理的二维质量弹簧仿真、深度强化学习（PPO）、遗传/进化算法、材料-控制协同优化等技术实现。

**📊 数据集**

使用EvoGymCM的六个仿真任务（Walker、BridgeWalker、DownStepper、CaveCrawler、AreaMaximizer、AreaMinimizer）作为评测数据集。

**📈 对比分析**

与传统控制优化及固定材料方法对比，Reactive-Material Co-Design在复杂地形与形变任务上提升10-30%奖励，Invariant-Material Co-Design在行走与导航任务上加速收敛并提高性能上限。

**⚠️ 局限性**

局限在于当前演化方法在极端形变任务上精度有限、样本效率低、未实现从仿真到真实平台的转移。

---

## 465. The Statistical Profitability of Social Media Sports Betting Influencers: Evidence from the Nigerian Market

**arXiv ID:** 2604.08251 | [PDF](https://arxiv.org/pdf/2604.08251v1)

**作者:** Kayode Makinde `[一作]` (ML Collective), Frances Adelakun `[通讯]` (ML Collective)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过抓取三名尼日利亚体育博彩影响者在X和Telegram上公开的预赛投注链接，收集5,467笔投注（约480万美元）并验证结果，评估其盈利性与跟随其建议的财务影响。

**💡 创新点**

创新之处在于消除幸存者偏差，系统性收集并验证公开可追溯的投注数据，采用四种不同的投注策略模拟，并对三名影响者的真实表现进行统计检验。

**🔧 技术方法**

主要技术包括网络爬虫与JSON解析、数据清洗与货币标准化、赔率与回报统计、四种数学投注策略（平盘、倒数、平方根、固定收益）模拟、以及单因素方差分析（ANOVA）检验。

**📊 数据集**

使用的数据集为5,467份预赛投注记录，涵盖三名影响者（@mrbanks、@louiedi13、@bossolamilekan1）的投注金额、赔率和最终支付，覆盖约$4.8 M的总下注额。

**📈 对比分析**

通过对四种投注策略的模拟比较，发现所有策略均导致净亏损，且策略对资本消耗率有显著影响（p≈2×10⁻⁷），但不同影响者之间无显著差异（p≈0.125）。

**⚠️ 局限性**

局限性包括仅使用Stake.com平台的数据，未涵盖当地其他博彩平台；未考虑活跃投注者的动态行为（如早期兑现、保险等）；样本仅限三名顶级影响者；未对社交媒体算法对信息可见度的影响进行量化。

---

## 466. The Quantum Query Complexity of Finding a Tarski Fixed Point on the 2D Grid

**arXiv ID:** 2604.08223 | [PDF](https://arxiv.org/pdf/2604.08223v1)

**作者:** Reed Phillips `[一作]` `[通讯]`, Reed Phillips

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

证明了在量子查询模型下，二维格子 Tarski 固定点问题的最小查询复杂度为 Ω((log n)^2)，并推广到任意维度 k≥2；

**💡 创新点**

主要创新在于构造了能够嵌入嵌套有序搜索的特殊 herringbone 函数，并提出了适用于非布尔输出函数的谱对抗方法组合定理；

**🔧 技术方法**

使用谱对抗方法（spectral adversary）与函数组合定理、嵌套有序搜索（nested ordered search）以及对 herringbone 结构的几何分析；

**📊 数据集**

无真实数据集，完全是理论构造和证明；

**📈 对比分析**

与已知的经典确定性上界 O((log n)^2) 对比，证明量子复杂度与经典上界同阶，达到 Θ((log n)^2)；

**⚠️ 局限性**

局限性：目前仅针对 k=2（以及通过单调性推广到 k≥3）得到结果，未能给出 k≥3 的更紧凑量子下界，且对高维情况的结构分析仍待进一步研究。

---

## 467. SciFigDetect: A Benchmark for AI-Generated Scientific Figure Detection

**arXiv ID:** 2604.08211 | [PDF](https://arxiv.org/pdf/2604.08211v1)

**作者:** You Hu `[一作]` (Zhejiang University), Xiaobai Li `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SciFigDetect 基准，构建了高质量的 AI 生成科学图表检测数据集，并对现有检测方法进行系统评估。

**💡 创新点**

创新点在于：①首次针对结构化、文本密集的科学图表构建专门基准；②设计了基于 GPT 控制器的 agent‑based 数据流水线，能够自动化生成与原图对应的合成图并经过人工审核；③提供了真实‑合成配对及多类别、多生成器覆盖，揭示了跨生成器泛化与抗压缩等挑战。

**🔧 技术方法**

技术手段包括多模态文本与图像理解、结构化提示构造、Nano Banana Pro 与 GPT‑image‑1.5 合成、人工复核迭代循环；评估使用 CNNSpot、PatchFor、UniFD、LGrad、NPR、FreqNet、FatFormer、AIDE、Effort 等现有 AI 生成图像检测模型。

**📊 数据集**

使用的数据集是 SciFigDetect，包含 72,965 张真实图表与 150,807 张合成图表，涵盖 Illustration、Overview、Experimental 三类，且每条样本附带原论文上下文、生成提示、生成器信息等元数据。

**📈 对比分析**

评估方式：①零样本（zero‑shot）测试显示大多数模型准确率仅 40–60%；②单一生成器训练导致严重过拟合，跨生成器准确率下降至 25–50%；③联合训练略有提升，但仍有显著差距；④在 JPEG/WebP 压缩、模糊、噪声等退化条件下，多数模型性能急剧下降，说明鲁棒性不足。

**⚠️ 局限性**

局限性：现有检测器在科学图表上表现不佳，难以迁移到新生成器；鲁棒性差，对常见后处理敏感；基准目前仅覆盖 Nano Banana 与 GPT‑image‑1.5 两个生成器，未来需要扩展更多模型与更复杂的视觉内容。

---

## 468. Introducing Echo Networks for Computational Neuroevolution

**arXiv ID:** 2604.08204 | [PDF](https://arxiv.org/pdf/2604.08204v1)

**作者:** Christian Kroos `[一作]` (Fraunhofer Institute for Integrated Circuits, IIS), Fabian Küch `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种仅用连接矩阵定义的循环神经网络（Echo Network），并用神经进化方法在极小规模网络上实现了 ECG 信号的二分类；

**💡 创新点**

创新点在于将网络结构与权重完全统一为矩阵表示，允许利用矩阵运算进行系统化的变异与重组，从而提升进化效率并简化网络变形；

**🔧 技术方法**

使用了神经进化（基于 NEAT 的改进）以及遗传算法的变异、交叉、speciation 和共享适应度等技术；

**📊 数据集**

使用 PTB‑XL 数据集中的单通道 12‑lead ECG 记录，采样率 100 Hz，构成 10 秒的时间序列；

**📈 对比分析**

通过在 10 次独立进化实验中比较准确率，Echo Network 的平均准确率约为 0.687，最高 0.696，优于传统 RNN（0.671/0.684），在更大种群实验中可达 0.717；

**⚠️ 局限性**

局限在于仅在 ECG 分类任务上验证，网络规模仍极小，缺乏对更复杂任务的评估，且未探究梯度下降等训练方式，矩阵规模随节点数二次增长可能导致计算瓶颈。

---

## 469. Equivariant Efficient Joint Discrete and Continuous MeanFlow for Molecular Graph Generation

**arXiv ID:** 2604.08189 | [PDF](https://arxiv.org/pdf/2604.08189v1)

**作者:** Rongjian Xu `[一作]` (Shandong University), Guoqiang Wu `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种统一的 SE(3)-等变生成框架 EQUIMF，能够同步建模分子图的离散拓扑与连续几何，实现高效的少步采样。

**💡 创新点**

创新点包括：①使用同步的 MeanFlow 动态实现离散与连续域的互相条件化；②提出新的离散 MeanFlow 参数化，避免了瞬时速率矩阵导致的细步限制；③基于 EGNN 的全等变编码器，实现结构与几何的跨模融合。

**🔧 技术方法**

采用 MeanFlow、离散/连续 Flow 匹配、EGNN 等技术，配合时间同步和互相条件化的采样策略。

**📊 数据集**

在 QM9 和 GEOM‑DRUG 两大分子数据集上进行实验。

**📈 对比分析**

与多种基线（EnF、G‑Schnet、GDM、EDM、EQUIFM 等）比较，EQUIMF 在原子/分子稳定性、有效性、唯一性指标上均优于对手，并在采样效率上相较 SOTA 提升约 2 倍。

**⚠️ 局限性**

限制在于目前采用的是少步演化方案，尚未实现单步离散-连续生成，且对极大分子复杂结构的泛化仍待验证。

---

## 470. AT-ADD: All-Type Audio Deepfake Detection Challenge Evaluation Plan

**arXiv ID:** 2604.08184 | [PDF](https://arxiv.org/pdf/2604.08184v1)

**作者:** Yuankun Xie `[一作]` (Communication University of China & Ant Group), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AT-ADD（All-Type Audio Deepfake Detection）挑战赛，构建了两条赛道（语音鲁棒检测与全类型检测），并提供统一的评测框架、数据集、基准模型与规则，旨在推动音频深度伪造检测技术向多模态、跨域、实际部署的方向发展。

**💡 创新点**

创新点包括：① 设计了覆盖语音、环境声、歌声和音乐四类音频的全类型评测体系；② 将真实世界场景（多设备、多语言、多噪声、回声、压缩等）与未见生成模型的鲁棒性需求融合进赛道；③ 引入 Audio Large Language Models（ALLM）作为通用检测基线，探索跨类型和跨生成器的统一建模；④ 在闭合评测环境下提供标准化数据、评测指标和公平基准，推动社区可重复、可比对的研究。

**🔧 技术方法**

采用的主要技术包括：传统的 Spec-ResNet 与 AASIST 端到端网络；基于自监督学习的 XLSR 与 WPT（Wavelet Prompt Tuning）增强的 SSL 基线；以及基于 Qwen2.5-Omni 系列 ALLM 的监督微调（SFT）模型。评测指标使用宏观 F1 分数，分别针对二分类和四类音频平衡计算。

**📊 数据集**

使用的数据集为 AT-ADD 的 Track1 与 Track2，包含 49,575/49,734 训练/开发样本的语音、环境声、歌声和音乐，每类各有上千的真实样本和多种 TTS/VC/音乐生成模型产生的伪造样本，且在评测集加入了未见生成器、噪声、回声、压缩等扰动。

**📈 对比分析**

基准实验结果显示：SSL 基线 FT-XLSR-AASIST 在 Track1 评测集上的 Macro‑F1 达到 76.73%（最高），在 Track2 评测集上达到 79.47%；传统 Spec-ResNet 仅为 53.83%；ALLM 基线 Qwen2.5‑Omni‑7B 在 Track1 与 Track2 评测集分别取得 68.64% 与 61.78%，表现优于传统基线但略逊于 SSL。总体而言，SSL 模型在跨域鲁棒性和跨类型泛化上更具优势。

**⚠️ 局限性**

局限性包括：① 在闭合评测下仅使用官方数据，限制了模型多样性与数据规模；② 对回声、噪声等扰动的鲁棒性仍不足，尤其在真实场景中可能出现更复杂的变形；③ ALLM 基线虽然通用，但受限于模型体量和推理成本，且缺乏可解释性；④ 跨类型检测仍存在显著差异（如环境声、音乐的 F1 低于歌声和语音），表明共享特征学习尚未充分；⑤ 目前缺乏对外部评测、实际部署效果的验证，未来需要在更广泛的应用场景中进一步评估。

---

## 471. Long-Term Embeddings for Balanced Personalization

**arXiv ID:** 2604.08181 | [PDF](https://arxiv.org/pdf/2604.08181v1)

**作者:** Andrii Dzhoha `[一作]` (Zalando SE), Egor Malykh `[通讯]` (Zalando SE)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在大规模电商场景下提出高惯性长时嵌入（LTE）框架，利用固定语义基底实现长短期偏好平衡

**💡 创新点**

创新点在于将LTE嵌入Transformer前缀token，并通过固定内容嵌入基底解决生产环境中的点对点一致性与回滚问题，同时引入不对称自编码器实现行为细化

**🔧 技术方法**

采用的技术包括Causal Language Modeling Transformer、LTE前缀token、CLIP内容嵌入基底、注意力迁移分析及不对称自编码器

**📊 数据集**

使用Zalando电商日志数据，约70M用户、25市场，短期序列为过去60天交互，长时窗口为过去365天交互

**📈 对比分析**

与基线Transformer相比，LTE+前缀在离线NDCG@500提升约1.3%，线上AB测试中用户参与率提升0.61%，收入提升0.42%

**⚠️ 局限性**

局限在于对短期序列的依赖仍明显、LTE窗口选择需在防止数据泄漏与版本漂移之间取得平衡，以及对低活跃用户特征稀缺问题

---

## 472. Counting HyperGraphlets via Color Coding: a Quadratic Barrier and How to Break It

**arXiv ID:** 2604.08278 | [PDF](https://arxiv.org/pdf/2604.08278v1)

**作者:** Marco Bressan `[一作]` (University of Milan), Giacomo Fumagalli `[通讯]` (University of Milan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究k-超图体计数问题并提出了一种基于(α,β)-nice结构的高效算法

**💡 创新点**

首次引入(α,β)-niceness概念，证明颜色编码在此假设下能突破二次复杂性，并给出有效采样方法

**🔧 技术方法**

使用颜色编码、结构分解（按rank≤α和degree≤β划分）、组合分析以及随机采样技术

**📊 数据集**

在多组公开的真实世界超图数据集上进行实验

**📈 对比分析**

与朴素二次算法对比，实验表明在真实超图上平均提升十倍以上

**⚠️ 局限性**

仅在满足(α,β)-nice假设的超图上有效；对于更大α或β值、或弱于此假设的结构尚无证明

---

## 473. HyperMem: Hypergraph Memory for Long-Term Conversations

**arXiv ID:** 2604.08256 | [PDF](https://arxiv.org/pdf/2604.08256v1)

**作者:** Juwei Yue `[一作]` (Chinese Academy of Sciences), Yafeng Deng `[通讯]` (EverMind AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了HyperMem——一个基于超图的三层次记忆架构，用于长期对话中的高阶关联检索。

**💡 创新点**

通过超边显式建模主题、事件、事实之间的多元关联，超越传统对偶关系，提供层次化检索和嵌入传播，显著提升跨时段多跳推理能力。

**🔧 技术方法**

使用超图结构、LLM驱动的事件分段与主题聚合、超边嵌入传播、双重索引（BM25+dense embedding）、粗到细检索与重排序以及链式思维生成等技术。

**📊 数据集**

在LoCoMo基准数据集上进行实验。

**📈 对比分析**

与多种RAG与记忆系统基线（如GraphRAG、LightRAG、HyperGraphRAG、MIRIX等）对比，采用GPT‑4o‑mini评估，整体准确率达92.73%，在单跳、多跳、时间推理和开放域等四类任务上均优于对手。

**⚠️ 局限性**

仅适用于单用户场景，扩展到多用户或多代理时面临访问控制与记忆隔离挑战；开放域问题仍受限于会话内信息，需进一步集成外部知识库。

---

## 474. FORSLICE: An Automated Formal Framework for Efficient PRB-Allocation towards Slicing Multiple Network Services

**arXiv ID:** 2604.08244 | [PDF](https://arxiv.org/pdf/2604.08244v1)

**作者:** Debarpita Banerjee `[一作]` (Indian Statistical Institute Kolkata), Rana Pratap Sircar `[通讯]` (Ericsson Research)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于形式化方法的三层级框架 FORSLICE，用来在 5G RAN 切片中自动、可靠地分配物理资源块（PRB），并保证公平性（每种服务类型获得所需 PRB 和吞吐量）与 PRB 最优性（最小化未使用 PRB），同时优先满足 eMBB Premium 服务需求。

**💡 创新点**

创新点包括：①首次将 SMT（可满足性模理论）应用于 RAN PRB 分配，保证分配策略的正确性与完整性；②构建了可自动化处理任意 (S, K, N) 配置的三层模型；③通过形式化验证实现对公平性与 PRB 最优性双重系统属性的同时满足；④在实验中实现了比现有 AI‑规划基准 Convergence 约 44.45% 的 PRB 最优性提升。

**🔧 技术方法**

技术主要包括：形式化建模（约束逻辑与线性算术）、SMT 求解（Z3）、Python 脚本生成与约束集、网络仿真（NS3‑5Glena）来验证吞吐量；此外使用概率分布（如 log‑normal、Poisson 等）生成用户到达/离开数据。

**📊 数据集**

使用的数据集：基于 5G 业务类型（eMBB、URLLC、mMTC、FWA、eMBB Premium/Normal）的用户到达概率分布，随后通过阈值化生成二进制用户入口/离开标记；实验覆盖 (3,2,4)、(3,3,7)、(5,3,10)、(5,4,13) 四种服务/分区/切片配置，仿真时长 30、50、70 步，PRB 总数 100/200/300。

**📈 对比分析**

比较方法：将 FORSLICE 与基准 Convergence 在同一 (3,2,4) 配置、30 步内进行 PRB 共享比例对比；结果显示 FORSLICE 的 eMBB Premium 切片 PRB 共享比例约低 44.45%，说明其在保证公平的同时更优地利用资源；NS3 仿真验证了 FORSLICE 分配导致的吞吐量与实际网络吞吐量基本一致。

**⚠️ 局限性**

局限性：①模型假设每步最多只有一个用户进入/离开，简化了动态流量；②SMT 约束规模随分区/切片数量呈指数增长，导致求解时间随时间步和配置尺寸显著增加；③未考虑无线链路物理层动态变化（如衰落、干扰）对 PRB 需求的即时调整；④仅在单基站单元（gNB）内验证，扩展到多基站或全网时需进一步优化。

---

## 475. On the Capacity of Sequences of Coloring Channels

**arXiv ID:** 2604.08234 | [PDF](https://arxiv.org/pdf/2604.08234v1)

**作者:** Wenjun Yu `[一作]` (Ben Gurion University of Negev), Moshe Schwartz `[通讯]` (McMaster University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了多通道染色通道（coloring channel）序列的容量，给出了多种结构（均匀荚形、任意交叉集合、路径）的精确容量，并提出基于对偶图（pairs graph）的通用上界与下界；在字母表大小为4时，除循环外的所有不可约序列容量已给出；同时讨论了容量与图结构的关系及未解决的循环情形。

**💡 创新点**

创新点在于：1）将容量问题转化为对pairs graph的分析，证明容量仅由该图决定；2）给出完整的优化公式并利用柯西分数与切比雪夫多项式求解均匀荚形、两集合、路径的容量；3）构造了从通道集合到两字母通道的等价变换，得到普适上下界；4）在q=4时几乎完整归纳所有不可约情形，剩余仅循环。

**🔧 技术方法**

主要技术包括组合优化、连续分数、切比雪夫多项式、对偶图与图的团数/交集数理论、Lagrange乘子法、熵函数的凸性与单调性。

**📊 数据集**

由于研究纯粹为理论性，无使用实验数据集；所有结果均来自符号推导与数值优化。

**📈 对比分析**

与先前工作（单通道、等集、(q-2,1,2)荚形）相比，本文在更广泛结构上给出精确容量；对q=4的剩余案例给出紧致上下界，理论上已达到最佳。

**⚠️ 局限性**

局限性：1）循环结构的精确容量仍未知；2）缺乏对容量达到的重构码的构造与实现细节；3）编码与重构的高效算法尚未给出；4）对于更大字母表或更复杂图形的分析仍需进一步研究。

---

## 476. MemCoT: Test-Time Scaling through Memory-Driven Chain-of-Thought

**arXiv ID:** 2604.08216 | [PDF](https://arxiv.org/pdf/2604.08216v1)

**作者:** Haodong Lei `[一作]` (Southeast University), Hongsong Wang `[通讯]` (Southeast University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在长上下文推理中引入自演化的内存‑推理循环，实现了主动检索与推理的结合。

**💡 创新点**

创新点在于把内存从被动匹配转变为主动、状态驱动的检索，并提出多视角长记忆感知与任务条件双短期记忆系统，实现迭代搜索与查询演化。

**🔧 技术方法**

主要技术包括多视角感知（Zoom‑In、Zoom‑Out、Panoramic视觉）、LightRAG检索、RAG+CoT思路、短期记忆演化函数、Judge Agent以及LLM推理。

**📊 数据集**

使用LoCoMo（单跳/多跳/开放域/时间推理）和LongMemEval‑S（多会话/单会话等）两个基准数据集进行评估。

**📈 对比分析**

与CompassMem、Mnemis、EMem‑G、CoT等多种基线对比，MemCoT在LoCoMo上达到58.03 F1，Qwen2.5‑14B 57.06，Qwen2.5‑7B 52.01；在LongMemEval‑S上整体得分88.0，明显优于现有方法。

**⚠️ 局限性**

局限包括对窗口大小、迭代步数、TopK等超参数敏感；依赖大型预训练模型；开放域样本量少影响总体分数；对噪声鲁棒性仍需进一步研究。

---

## 477. EditCaption: Human-Aligned Instruction Synthesis for Image Editing via Supervised Fine-Tuning and Direct Preference Optimization

**arXiv ID:** 2604.08213 | [PDF](https://arxiv.org/pdf/2604.08213v1)

**作者:** Xiangyuan Wang `[一作]` (Peking University), Wei Zhu `[通讯]` (Xiaohongshu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了EditCaption两阶段后训练流程，先用100K人工校正的监督数据微调VLM，再通过10K人类偏好对齐进一步纠正方向、视角和细节错误，提升图像编辑指令生成质量；

**💡 创新点**

创新点在于：①识别并系统化三大指令生成错误；②构建大规模高质量SFT数据集并使用EditScore筛选；③利用Direct Preference Optimization (DPO)针对特定错误进行细粒度对齐；

**🔧 技术方法**

采用Vision‑Language模型（Qwen3‑VL）为基础，结合GLM生成初始指令、EditScore进行质量过滤、人工校正、SFT训练，再用DPO对偏好对进行对齐；

**📊 数据集**

使用的评测数据集包括自研Eval‑400（400对图像）、ByteMorph‑Bench（600对）和HQ‑Edit（500对），SFT训练数据由150K图像对生成并人工精修得到的100K对；

**📈 对比分析**

与多款开源模型（Qwen3‑VL‑32B/235B、GLM‑4.5V、Kimi‑K2.5）以及闭源Gemini‑3‑Pro、GPT‑4.1进行对比；在Eval‑400、HQ‑Edit、ByteMorph‑Bench等指标上，SFT+DPO版本在加权评分上均超过所有开源模型，并在Eval‑400上略优于Gemini‑3‑Pro；

**⚠️ 局限性**

局限在于：仍需外部过滤（EditScore）以提升样本质量；对复杂多步或稀有编辑场景的覆盖不足；未在完整编辑系统中验证生成指令对最终编辑效果的直接影响；

---

## 478. State and Trajectory Estimation of Tensegrity Robots via Factor Graphs and Chebyshev Polynomials

**arXiv ID:** 2604.08185 | [PDF](https://arxiv.org/pdf/2604.08185v1)

**作者:** Edgar Granados `[一作]` (Rutgers University), Kostas Bekris `[通讯]` (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于因子图的实时状态估计方法，结合RGB‑D相机和缆绳长度传感器，随后使用Chebyshev多项式对整条轨迹进行离散到连续的后处理。

**💡 创新点**

创新点在于首次将因子图和Chebyshev多项式应用于软刚性张力结构机器人，实现了不依赖驱动模型的高精度状态和轨迹估计，并通过Mahalanobis聚类有效抑制点云噪声。

**🔧 技术方法**

技术包括因子图状态估计、Mahalanobis距离聚类、RGB‑D点云预处理、缆绳长度测量融合，以及基于Chebyshev多项式的连续时间轨迹重建。

**📊 数据集**

实验使用了三组数据集：原始长短轨迹（含MoCap标定）以及在不同环境下新收集的20Hz数据。

**📈 对比分析**

与基于ICP的传统方法比较，因子图+多项式方案在旋转误差上优于ICP，并在无手动初始化且高频下保持低误差；因子图估计误差略高于ICP，但多项式轨迹在离散时间误差更小，适合系统辨识与机器学习。

**⚠️ 局限性**

局限性包括对点云预处理的计算开销、对相机遮挡敏感以及未考虑复杂外部干扰，未来需扩展至不同张力结构并结合在线控制。

---

## 479. Towards Improving the External Validity of Software Engineering Experiments with Transportability Methods

**arXiv ID:** 2604.08200 | [PDF](https://arxiv.org/pdf/2604.08200v1)

**作者:** Julian Frattini `[一作]`, Carlo Furia `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出将医学领域的 transportability 方法引入软件工程实验，以解决实验样本与目标人群之间的外部效度缺失问题，并通过仿真验证其有效性。

**💡 创新点**

创新点在于：①首次系统阐述 transportability 的前提与方法在 SE 场景中的适用性；②给出实践路线图与使用指南；③通过模拟展示相较于传统平均差、线性交互模型，IPSW 与 g‑formula 能显著降低外推误差。

**🔧 技术方法**

采用因果推断框架（DAG、潜在结果模型），实现两类方法：逆概率抽样加权（IPSW）和基于回归的 g‑formula（plug‑in）。

**📊 数据集**

使用自建的模拟数据集：目标人群 1000 名，经验变量 X 服从负二项分布；随机抽样试验样本（约 175 名）受经验影响的招募概率；二元处理 A 与正态分布结果 Y；已知真实 ATE = 16.7。

**📈 对比分析**

比较四种估计：1）平均差（naïve） 2）线性交互回归 3）IPSW 4）g‑formula。仿真 50 次后，g‑formula 的估计均值最接近真实 ATE，且置信区间覆盖率最高；IPSW 估计方差大、极端权重导致不稳定。

**⚠️ 局限性**

局限性包括：依赖假设 A5–A7（如可观测的试验参与概率、正向性、条件均值可交换性）；对未观测调节因子的敏感性；权重极值导致 IPW 稳定性差；仅在模拟环境中验证，实际工业数据中的噪声和复杂性仍需进一步研究。

---

## 480. Securing Retrieval-Augmented Generation: A Taxonomy of Attacks, Defenses, and Future Directions

**arXiv ID:** 2604.08304 | [PDF](https://arxiv.org/pdf/2604.08304v1)

**作者:** Yuming Xu `[一作]` (Hong Kong Polytechnic University), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统梳理并分类了检索增强生成（RAG）安全领域的攻击、对策与评测，提出以外部知识访问管道为核心的安全框架，并划分出三大信任边界与四个安全表面。

**💡 创新点**

创新点在于：① 明确区分固有LLM风险与RAG引入或放大风险的操作边界；② 把RAG流程抽象为六阶段，并将攻击、对策与评测与之对应；③ 构建跨界、层级化的攻击/防御/评测体系，揭示现有研究的结构性差距。

**🔧 技术方法**

主要技术是文献综述与系统性分类（taxonomy）方法，结合安全性分析框架，未采用实验或模型训练。

**📊 数据集**

未直接使用数据集；综述中引用了多份公开基准（如SafeRAG、OpenRAG‑Soc、RAGCRAWLER、MedPriv‑Bench 等）来说明评测现状。

**📈 对比分析**

通过对比已有攻击与防御研究，评估了它们在不同安全表面（预检索腐败、检索时操控、检索上下文滥用、知识外泄）上的表现与局限；由于是综述，未给出统一实验性能指标，而是总结了各方法的优劣与适用场景。

**⚠️ 局限性**

局限性包括：① 仅聚焦现有公开文献，未涵盖最新或未公开的攻击/防御技术；② 综述性质导致缺乏统一评测体系与量化指标；③ 对特定应用场景（如多模态、Web‑native、智能代理）的深入分析仍不足。

---

## 481. On quadratic binomial vectorial functions with maximal bent components

**arXiv ID:** 2604.08311 | [PDF](https://arxiv.org/pdf/2604.08311v1)

**作者:** Xianhong Xie `[一作]` (Anhui Agricultural University), Shenxing Zhang `[通讯]` (Hefei University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了具有最大弯曲分量的二项向量函数F(x)=x^d_1+x^d_2的性质，证明了在特定条件下F(x)与x^2^m+1或x^2^i+1+x^2^m+2^i是仿射等价的。

**💡 创新点**

创新点在于通过Stickelberger定理，明确了在特定条件下，具有最大弯曲分量的二项向量函数的等价形式，并给出了其非线性和差分均匀性的界限。

**🔧 技术方法**

使用了Stickelberger定理和Walsh变换等数学工具。

**📊 数据集**

未具体提及使用的数据集，主要是理论分析。

**📈 对比分析**

通过与已有文献的结果进行比较，证明了在特定条件下，F(x)的弯曲分量数量达到最大，并给出了非线性和差分均匀性的理论界限。

**⚠️ 局限性**

限制在于需要满足特定的技术条件，且未能完全解决所有可能的二项向量函数的分类问题。

---

## 482. Towards Identification and Intervention of Safety-Critical Parameters in Large Language Models

**arXiv ID:** 2604.08297 | [PDF](https://arxiv.org/pdf/2604.08297v1)

**作者:** Weiwei Qi `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Expected Safety Impact (ESI) 框架来识别 LLM 的安全关键参数，并基于此设计两种干预方法：Safety Enhancement Tuning (SET) 用于快速提升未对齐模型的安全性，Safety Preserving Adaptation (SPA) 用于在下游任务中保持已对齐模型的安全性。

**💡 创新点**

①将安全期望值与参数标准差结合，构造 ESI 指标；②利用可微判别器与 Gumbel‑Softmax 估计安全梯度；③揭示 Dense 与 MoE 结构中安全关键参数的层级差异；④提出仅更新 1% 安全关键参数的 SET 与冻结关键参数的 SPA 两种高效、低成本干预策略。

**🔧 技术方法**

ESI 指标、可微判别器（Gumbel‑Softmax + 投影矩阵）、梯度与标准差估计、SET/SPA 细调技术；评估指标为攻击成功率 (ASR)、任务准确率、语义相似度。

**📊 数据集**

危害输入分布：AdvBench、HarmBench、WildJailbreak；安全对齐数据集：CB‑Safety、R1‑Safety；下游任务数据集：GSM8K、AGNews、MedicalQA。

**📈 对比分析**

与随机选择、SN‑Tune、LoRA、SafeLoRA 等基线比较。SET 在 Qwen2.5‑14B‑base、Llama3‑8B‑base 等模型上仅用 100 次迭代更新 1% 参数即可将 ASR 降至 10% 以下，攻击成功率下降超过 50%；SPA 在下游任务中将安全性下降控制在 1% 以内，同时保持任务性能与全参数微调相当。

**⚠️ 局限性**

只在主流 Dense 与 MoE 架构上验证；需要访问内部参数，限制对闭源模型的适用性；评估聚焦于通用有害情境，未覆盖专业领域安全问题。

---

## 483. Can Vision Language Models Judge Action Quality? An Empirical Evaluation

**arXiv ID:** 2604.08294 | [PDF](https://arxiv.org/pdf/2604.08294v1)

**作者:** Miguel Monte e Freitas `[一作]` (Sword Health), Pedro Henrique Martins `[通讯]` (Sword Health)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对现有的Vision‑Language Models（Gemini 3.1 Pro、Qwen3‑VL、InternVL3.5）在多种动作质量评估（AQA）任务上进行系统评测，涵盖多任务、多域（健身、体操、自由式滑冰、跳水）、不同视觉预处理（裁剪、骨架叠加、骨架仅）以及多种提示工程（视觉定位、双步骤推理、结构化推理、指南、上下文学习），并通过偏差分析与对比任务进一步探讨模型的系统性错误。

**💡 创新点**

提出了一个面向 AQA 的完整评测框架，系统地量化了 VLM 在骨架信息、提示方式以及偏差对性能的影响，并首次将对比任务作为降低语言与先验偏差的手段来评估 VLM 的真正动作质量判断能力。

**🔧 技术方法**

使用的技术包括：现成 VLM 推理（Gemini 3.1 Pro、Qwen3‑VL、InternVL3.5）、骨架提取（SAM 3D Body）、多种提示模板（视觉定位、两步推理、结构化推理、正负指南、in‑context learning）、对比任务改写、以及偏差分析与推理轨迹可视化。

**📊 数据集**

采用的公开数据集有：LLM‑FMS（功能动作问答）、EgoExo‑Fitness（体能视频与技术准则）、Fitness‑AQA（错误检测）、FineFS（自由式滑冰 GOE 评分）、MTL‑AQA（跳水评分）。

**📈 对比分析**

通过平衡准确率（分类）和 Spearman 相关/相对 L2 距离（回归）与随机猜测、模型自身最佳表现进行对比。整体结果仅略高于随机（最高约 60% 平衡准确率，Spearman ρ 最高 0.37），提示工程和骨架预处理带来的提升有限，甚至在部分任务中无显著改进。对比任务虽消除语言偏差，但整体性能仍差距大。

**⚠️ 局限性**

局限性：仅评估离线现成 VLM，未进行领域微调；模型受先验知识和语言提示强烈影响，导致偏差；不同任务/域间的泛化能力不佳；数据集规模有限，且评测主要关注视觉输入，缺少跨模态多样性；对比任务虽降低偏差，但未能显著提升 AQA 能力，表明 VLM 在细粒度动作质量判断上的根本瓶颈仍待突破。

---

## 484. EMMa: End-Effector Stability-Oriented Mobile Manipulation for Tracked Rescue Robots

**arXiv ID:** 2604.08292 | [PDF](https://arxiv.org/pdf/2604.08292v1)

**作者:** Yifei Wang `[一作]`, Haoyao Chen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了面向抓取稳定性的移动机械臂轨迹规划与控制框架EMMa，针对搜索与救援场景下的重型履带移动机械臂进行端执行器稳定操作

**💡 创新点**

1) 将端执行器姿态直接纳入优化变量，显著降低非线性耦合；2) 设计简化可导成本（操纵性、平滑性、碰撞回避）和硬约束；3) 引入隔离整体控制（Feedforward-Feedback）抑制底盘诱导的扰动

**🔧 技术方法**

协调优化（端执行器+底盘状态）+简化操纵性成本+障碍物ESDF回避+基于MPC的底盘跟踪+F3B隔离控制

**📊 数据集**

仿真中使用四类救援任务（抓取、移动抓取、巡检、搬运）和基于Raigor的真实世界实验，使用MID‑360 LiDAR构建ESDF地图，运动捕捉提供全局位姿

**📈 对比分析**

与ReDyn（基于全身动力学的QP）和GP（基于MPC+优化的全身）对比；在所有任务中EMMa在任务成功率、端执行器加速度/曲率等指标均优于对手，尤其在高底盘速度和复杂障碍下保持100%成功率

**⚠️ 局限性**

依赖精确运动学/动力学模型；对极端地形和未知扰动的鲁棒性尚待验证；控制器参数需手工调优，未实现完全端到端的数据驱动协调

---

## 485. A GAN and LLM-Driven Data Augmentation Framework for Dynamic Linguistic Pattern Modeling in Chinese Sarcasm Detection

**arXiv ID:** 2604.08381 | [PDF](https://arxiv.org/pdf/2604.08381v1)

**作者:** Wenxian Wang `[一作]` (Sichuan University), Haizhou Wang `[通讯]` (Sichuan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于GAN与LLM的数据增强框架，构建SinaSarc大规模中文讽刺检测数据集，并将用户历史行为特征融入模型；

**💡 创新点**

首次将用户长期语言模式视为特征引入讽刺检测；结合GAN生成评论与用户行为、GPT‑3.5进行语义增强，实现数据规模与多样性提升；

**🔧 技术方法**

使用WGAN‑GP生成器与判别器、GPT‑3.5文本替换、改造BERT+全连接融合用户行为的多模态检测网络；

**📊 数据集**

自行爬取并标注的微博评论，构造的SinaSarc共20,000条（10k讽刺、10k非讽刺），包含评论内容、主题、层级及用户历史行为特征；

**📈 对比分析**

与传统机器学习、LSTM/Transformer、预训练模型（BERT、RoBERTa）及LLM（GPT‑4‑Turbo、Qwen‑7B等）进行对比，在SinaSarc上取得最高F1≈0.9151、准确率≈0.9144，并在噪声、比例、规模等鲁棒性实验中持续优于SOTA；

**⚠️ 局限性**

受限于数据域和生成样本缺乏外部知识与修辞多样性，对背景知识、隐含修辞和隐喻的识别仍有限，缺乏多模态与更丰富用户特征。

---

## 486. MegaStyle: Constructing Diverse and Scalable Style Dataset via Consistent Text-to-Image Style Mapping

**arXiv ID:** 2604.08364 | [PDF](https://arxiv.org/pdf/2604.08364v1)

**作者:** Junyao Gao `[一作]` (Tongji University), Jun Zhang `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模、内风格一致、外风格多样的MegaStyle-1.4M数据集，并基于该数据集训练了风格编码器MegaStyle-Encoder和风格迁移模型MegaStyle-FLUX。

**💡 创新点**

创新在于利用大型文本到图像模型一致的风格映射生成高质量的风格对，提出风格监督对比学习训练专用编码器，并在FLUX架构上实现可泛化、稳定的风格迁移。

**🔧 技术方法**

技术包括Qwen-Image与Qwen3‑VL生成风格/内容提示、风格监督对比学习（SSCL）fine‑tune SigLIP编码器、基于FLUX的MM‑DiT模型以及多层样本去重与平衡采样。

**📊 数据集**

使用自建的MegaStyle‑1.4M（170K风格提示×400K内容提示、1.4M图像），并与WikiArt、JourneyDB、Style30K、IMAGStyle、OmniStyle‑150K等公开数据集进行对照。

**📈 对比分析**

通过StyleRetrieval、StyleBench、FLUX‑Retrieval等检索基准以及CLIP文本分数和人工评估进行比较，MegaStyle‑Encoder在mAP/Recall上远超CLIP、CSD，MegaStyle‑FLUX在风格与文本对齐、人工偏好得分上优于现有SOTA风格迁移方法。

**⚠️ 局限性**

局限在于提示语言描述仍可能含糊，难以完全捕捉纹理、笔触等细节；数据集与训练模型同源导致潜在偏差；未来需进一步完善提示策略并扩展至千万级规模。

---

## 487. Bias-Constrained Diffusion Schedules for PDE Emulations: Reconstruction Error Minimization and Efficient Unrolled Training

**arXiv ID:** 2604.08357 | [PDF](https://arxiv.org/pdf/2604.08357v1)

**作者:** Constantin Le Cleï `[一作]` (Technical University of Munich), Xiaoxiang Zhu `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种自适应噪声调度和代理反向训练框架，用于改进PDE扩散模型的重构精度和长期稳定性。

**💡 创新点**

创新点包括：基于曝光偏差的噪声调度优化，定义并利用两步曝光偏差构造最优稳定调度；以及利用低曝光偏差的代理估计实现高效的代理反向训练。

**🔧 技术方法**

使用技术包括：条件扩散概率模型（CDDPM）、自适应噪声调度算法、代理反向训练（Proxy Unrolled Training）、梯度截断以及与U-Net的对比实验。

**📊 数据集**

使用的数据集有1D Kuramoto–Sivashinsky、2D Transonic Flow（Tra）以及2D Kolmogorov Flow（Kolmo）等流体动力学数据。

**📈 对比分析**

与线性、余弦、Sigmoid等传统调度和U-Net的教师强迫与反向训练等基线相比，本文方法在一阶MSE、短期/长期误差和Fréchet谱距离等指标上均实现了显著提升，尤其在长时序的相关性和物理一致性方面表现优异。

**⚠️ 局限性**

局限性包括需要额外的两轮训练（探索+最终训练），对曝光偏差的估计假设在不同模型或任务中可能不完全适用，且对超参数（如阈值τ）敏感。

---

## 488. Fundus-R1: Training a Fundus-Reading MLLM with Knowledge-Aware Reasoning on Public Data

**arXiv ID:** 2604.08322 | [PDF](https://arxiv.org/pdf/2604.08322v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 489. PokeGym: A Visually-Driven Long-Horizon Benchmark for Vision-Language Models

**arXiv ID:** 2604.08340 | [PDF](https://arxiv.org/pdf/2604.08340v1)

**作者:** Ruizhi Zhang `[一作]` (SIAS, UESTC), Lixin Duan `[通讯]` (SIAS, UESTC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在《PokéGym》基准上评估并诊断视觉-语言模型在3D开放世界游戏中的长时序自主行为；

**💡 创新点**

提出严格的代码级隔离、基于内存扫描的自动化评估，以及以视觉、步骤、目标三种指令粒度设计任务以拆解视觉 grounding、语义推理与自主规划；

**🔧 技术方法**

使用基于GPU纹理的 RGB 观察、AOB 内存扫描评估器、VLM 决策模块、可选自我反思、以及高层动作与参数化控制两种动作接口；

**📊 数据集**

以 Pokémon Legends：Z-A 作为环境，构造 30 个从 30 至 220 步的长时序任务；

**📈 对比分析**

通过与多种开源与闭源 VLM（如 Qwen、GLM-4.6V、Gemini‑3‑Pro、Claude‑Sonnet‑4.6、GPT‑5.2 等）在三种指令粒度下的成功率、物理失误率、恢复率等指标进行比较，发现物理死锁恢复是主要瓶颈，闭源模型表现更好但仍有局限；

**⚠️ 局限性**

主要局限在于：缺乏显式空间直觉导致的物理死锁、对视觉与动作细节的依赖、以及在大环境中对复杂任务的长时规划与自我反思效果不佳。

---

## 490. Leveraging Complementary Embeddings for Replay Selection in Continual Learning with Small Buffers

**arXiv ID:** 2604.08336 | [PDF](https://arxiv.org/pdf/2604.08336v1)

**作者:** Danit Yanowsky `[一作]` (Hebrew University of Jerusalem), Daphna Weinshall `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在小容量回放缓冲的持续学习中，提出结合监督与自监督嵌入的多嵌入覆盖式样本选择方法。

**💡 创新点**

创新点是把监督和自监督特征空间同时用于缓冲选择，采用多嵌入覆盖目标与自适应密度权重实现更均衡、泛化更好的样本集合。

**🔧 技术方法**

使用覆盖（k‑coverage）采样、RBF核、k‑NN密度估计、自监督模型（SimCLR、VICReg、DINOv2）与监督网络。

**📊 数据集**

在 Split CIFAR‑100 与 Split TinyImageNet 两个类增量学习基准上评估。

**📈 对比分析**

与随机、herding、Rainbow、TEAL、MERS等传统选择策略及ER/ER‑ACE/ER‑ACE‑STAR等回放算法对比，在低内存（≤1000）时平均准确率、最终平均准确率均显著提升，甚至在更大缓冲下仍保持优势。

**⚠️ 局限性**

限制在于仍需额外的自监督训练开销，RBF带宽与权重设置对结果影响较大，且在极大缓冲或任务间变异极低时收益有限。

---

## 491. Dead Weights, Live Signals: Feedforward Graphs of Frozen Language Models

**arXiv ID:** 2604.08335 | [PDF](https://arxiv.org/pdf/2604.08335v1)

**作者:** Marcus Armstrong `[一作]` (University of Houston), Arjun Mukherjee `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种由多种冻结的大语言模型组成的前馈图架构，利用学习的线性投影在共享连续潜在空间中实现跨模型通信，仅训练投影矩阵与输出注意力节点；

**💡 创新点**

在多节点图中首次实现端到端可微的跨模型通信，并证明梯度能够通过多重冻结模型边界传递，同时输出节点能无监督地产生选择性路由；

**🔧 技术方法**

结合几何兼容性（跨模型潜在空间线性映射）、残差流注入、交叉注意力输出、L2归一化、梯度通过残差流钩子、LoRA、AdamW以及余弦学习率调度等技术；

**📊 数据集**

在ARC‑Challenge、OpenBookQA和MMLU这三大多选题基准上进行评估；

**📈 对比分析**

与单一冻结模型的贪婪推理基线及参数匹配的MLP头进行对比，取得ARC 87.3% (+11.4pp)，OpenBookQA 82.8% (+6.2pp)，MMLU 67.2% (+1.2pp)，均优于最佳单模型和匹配头部；

**⚠️ 局限性**

投影矩阵在层1未能实现专门化；梯度通过冻结模型边界仅约13%；实验仅单跑一次，且仅针对多选题，无对开放式生成和更长上下文的验证。

---

## 492. Revisiting Radar Perception With Spectral Point Clouds

**arXiv ID:** 2604.08282 | [PDF](https://arxiv.org/pdf/2604.08282v1)

**作者:** Hamza Alsharif `[一作]` (Eindhoven University of Technology), Gijs Dubbelman `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并验证了一种新的雷达感知输入形式——光谱点云（Spectral Point Cloud），将雷达点云视为压缩的频谱峰并进行谱信息丰富化；

**💡 创新点**

创新点在于将雷达点云作为稀疏频谱的压缩表示，并通过两种谱丰富策略（RD邻域扩展与角谱描述符）显式注入角度与多普勒等雷达特有信息，实现在稀疏输入下匹配甚至超越稠密频谱（RD）基准；

**🔧 技术方法**

采用FFTRadNet作为RD分支，PointPillars（改进版）作为点云分支，并构建SpectralPillars（点云+两种丰富化）；使用CFAR阈值控制点云稠密度，执行稀疏MIMO处理与角谱池化；

**📊 数据集**

主要使用RADIal数据集（高分辨率RD频谱和对应标注），并在相同峰集合下生成稀疏RD与光谱点云数据；

**📈 对比分析**

通过在多种数据密度下训练对比，SpectralPillars在7.2%密度即可达到RD基准，在56.4%密度达到93.7% F1，比稠密RD高+2.2%、比稀疏RD高+1.3%；在相同模型参数下F1接近RD state‑of‑the‑art；

**⚠️ 局限性**

局限性主要在于仅在RADIal场景（驾驶环境有限多样性）验证，需在更广泛的雷达配置与更大规模数据集上检验泛化与鲁棒性。

---

## 493. Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems

**arXiv ID:** 2604.08366 | [PDF](https://arxiv.org/pdf/2604.08366v1)

**作者:** Tolga Dimlioglu `[一作]` (New York University), Jose M. Alvarez `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MOSAIC 框架，结合聚类、排序和尺度律，针对多指标评估优化物理 AI 训练集的混合选择。

**💡 创新点**

创新点在于把数据域的尺度律与多指标效用函数联合建模，并通过迭代最大边际增益动态分配采样，兼顾数据异质性与指标竞争。

**🔧 技术方法**

使用基于地理位置或生成字幕的聚类、基于单片 EPDMS 重要性排序、每域指数尺度律拟合，以及迭代贪心采样策略。

**📊 数据集**

实验使用 Navtrain（curated）和 OpenScene（full）两大自动驾驶视频数据集，配合 Hydra‑MDP 端到端规划模型。

**📈 对比分析**

与 Random、Uncertainty、Coreset、Chameleon 等基线对比；MOSAIC 在所有预算下实现最高 EPDMS，BRMR < 0.2，表明在相同性能下需 80% 以上更少样本，甚至在 2400 条样本即可达到全量训练性能。

**⚠️ 局限性**

局限性包括：假设不同域贡献可被单独尺度律描述，若聚类质量不足则可能失效；依赖初始小规模 pilot 运行估计尺度律，增加额外计算开销。

---

## 494. Security Concerns in Generative AI Coding Assistants: Insights from Online Discussions on GitHub Copilot

**arXiv ID:** 2604.08352 | [PDF](https://arxiv.org/pdf/2604.08352v1)

**作者:** Nicolás E. Díaz Ferreyra `[一作]` (Hamburg University of Technology), Riccardo Scandariato `[通讯]` (Hamburg University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了开发者在 Stack Overflow、Reddit 和 Hacker News 上对 GitHub Copilot 的安全相关讨论，并识别出四大关注领域。

**💡 创新点**

首次系统归纳开发者视角下的 GenAI 编码助手安全关注点，并揭示法律、培训数据、代码安全和信任失衡四个主题。

**🔧 技术方法**

采用 BERTopic 主题建模结合主题分析、情感分析以及手工验证的关键词过滤技术。

**📊 数据集**

收集了 14,253 条 Copilot 相关帖子，筛选出 383 条安全讨论，来源于 Stack Overflow、Reddit、Hacker News。

**📈 对比分析**

通过主题聚类和情感分布对比三平台讨论热度与情感倾向，未进行定量性能评估。

**⚠️ 局限性**

研究受限于平台样本、关键词过滤误差、单一工具聚焦，难以推广到其他 GenAI 助手或更广泛开发者群体。

---

## 495. Revisiting Fair and Efficient Allocations for Bivalued Goods

**arXiv ID:** 2604.08345 | [PDF](https://arxiv.org/pdf/2604.08345v1)

**作者:** Hui Liu `[一作]` (Fuzhou University), Zhijie Zhang `[通讯]` (Fuzhou University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了新的多项式时间算法，求解加权EFX和WEQX的 fPO 分配，并修正了先前 Garg 与 Murhekar 算法的终止缺陷。

**💡 创新点**

创新点在于首次证明并实现了双值偏好下的加权公平分配（WEFX、WEQX）与效率（fPO）的存在与可计算性，显著拓展了以往单值或无权重的结果。

**🔧 技术方法**

技术手段主要基于 Fisher 市场均衡框架、MBB 图、价格上升与物品转移的迭代过程，以及对价权重和组划分的精细分析。

**📊 数据集**

论文未使用任何实验数据集，全部以理论证明和算法复杂度分析为主。

**📈 对比分析**

与先前未能保证终止的 Garg–Murhekar 方法相比，本文的算法在理论上保证终止，并实现了 O(min{k,m}·n²·m²) 的多项式时间复杂度。

**⚠️ 局限性**

局限性在于仅适用于统一的双值偏好（非个性化）且只考虑可得物品，未覆盖个人化双值偏好或不良物品（chore）的情况。

---

## 496. EgoEverything: A Benchmark for Human Behavior Inspired Long Context Egocentric Video Understanding in AR Environment

**arXiv ID:** 2604.08342 | [PDF](https://arxiv.org/pdf/2604.08342v1)

**作者:** Qiance Tang `[一作]` (New York University), Sai Qian Zhang `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 EgoEverything 基准，包含 5,000+ 个多选问答，模拟 AR 视角下基于人眼注视生成问题，并对长时 egocentric 视频理解模型进行评估。

**💡 创新点**

创新点在于将人眼注视信息纳入问题生成，通过关注导向采样、多智能体生成管线和多层过滤，产生更贴近真实 AR 用户提问的问答；同时提供关注驱动与外围信息混合的题目，提升任务难度。

**🔧 技术方法**

使用了视觉语言模型（VLM）如 Gemini、Videollama3、LongVA；多智能体生成管线、ReID 与 CLIP 编码、基于 2D 高斯分布的关注采样、人工审阅与 LLM 盲筛等技术。

**📊 数据集**

基于 AEA（143 剧 7.3h）和 Nymeria（约 300h）egocentric 视频与注视数据构建数据集。

**📈 对比分析**

通过对比全分辨率、Gaze Crop、Gaze Mask、Average Downsampling、VMP/AMEGO 等预处理方法，在 EgoEverything 上评估 VLM，最高准确率为 Gemini 63.1%，仍远低于人类 83.5%；模型对远离注视或小尺寸目标的准确率显著下降。

**⚠️ 局限性**

局限性包括仅用注视位置近似人类注意力，忽略听觉、认知等多模态；数据仅覆盖日常活动，缺少工业或医疗等专业场景；问答仅英文；高斯参数固定不一定适用于所有任务和用户。

---

## 497. InstAP: Instance-Aware Vision-Language Pre-Train for Spatial-Temporal Understanding

**arXiv ID:** 2604.08337 | [PDF](https://arxiv.org/pdf/2604.08337v1)

**作者:** Ashutosh Kumar `[一作]` (Woven by Toyota), Quan Kong `[通讯]` (Woven by Toyota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 InstAP 框架和 InstVL 数据集，开展实例感知的视觉‑语言预训练，学习视频与文本在全局与实例级的对齐。

**💡 创新点**

创新点在于：① 引入实例级对齐目标，将文本实体与视频轨迹对应；② 采用双粒度注释（全局句子 + 轨迹实例描述）的 InstVL 数据集；③ 通过全局‑局部交叉注意力与实例对齐损失实现端到端的实例感知预训练；④ 证明实例级预训练还能提升全局表示。

**🔧 技术方法**

技术细节包括：ViT‑L 视觉编码器，教师‑学生注意力引导遮挡自监督视频建模；全局与实例级对比损失、匹配损失、掩码语言建模；跨模态注意力融合；轨迹 RoI 编码与跨模态跨尺度注意力。

**📊 数据集**

使用的数据集：InstVL（200 万图像 + 5 万视频，含全局句子和轨迹实例描述）；预训练阶段使用 K710、HDVILA、WebVid；训练数据还包含 CC3M、CC12M、SBU、Visual Genome、COCO、ShareGPT4V 等；测试集包括 MSR‑VTT、DiDeMo、MSVD、LSMDC、ActivityNet、InstVL 的多子集。

**📈 对比分析**

与 VideoPrism、CLIP4Clip、ViCLIP、OpenCLIP、CLIP‑ViP、UMT‑L 等 SOTA VLP 模型在 InstVL 的实例检索和全局检索进行对比；InstAP 在实例检索 R@1 达到 60.63，全球检索 R@1 超过 99%；在零样本检索上，MSR‑VTT、DiDeMo 等数据集 R@1 分别达到 41.1 / 54.0，显著优于之前最优结果。

**⚠️ 局限性**

局限性包括：多实例混淆、遮挡、背景主导或目标过小导致定位困难；依赖大量双粒度标注，数据采集成本高；对极度稀疏或小目标的精确定位仍有提升空间。

---

## 498. ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection

**arXiv ID:** 2604.08326 | [PDF](https://arxiv.org/pdf/2604.08326v1)

**作者:** He Geng `[一作]` (Xunfei Healthcare Technology Co Ltd), Xiaodong Tao `[通讯]` (Xunfei Healthcare Technology Co Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ProMedical统一框架，利用细粒度临床rubric对齐LLM，并发布了ProMedical-Preference-50k与ProMedical-Bench两个数据集，提供双盲专家评审的基准；

**💡 创新点**

核心创新在于Explicit Criteria Injection——将主效能、卓越奖励与安全阈值三维度嵌入奖励模型，并采用层级lexicographic安全约束，确保模型在优化过程中绝对遵守安全边界；

**🔧 技术方法**

技术实现包括人机循环的rubric生成、Rubric‑Aware Reward Model (RA‑RM) 的多维奖励学习、基于GRPO的强化学习对齐，以及双盲专家评审与严格去除训练数据的评测流程；

**📊 数据集**

主要使用的数据集为ProMedical‑Preference‑50k（50k条含细粒度rubric的训练样本）和ProMedical‑Bench（795条双盲专家评审的评测样本），并在UltraMedical、HealthBench等公开基准上做对比；

**📈 对比分析**

在ProMedical‑Bench上，ProMedical‑RM 8B(Qwen3)的总体准确率达86.55%，安全F1为89.09%，相较于其他开源/闭源基准提升约22.3%准确率、21.7%安全性；同样规模的通用大模型在无安全监督下仅能取得约53%准确率，表明规模无法替代细粒度安全监督；

**⚠️ 局限性**

局限性包括仅处理文本数据、依赖专家共识生成rubric（在争议医学领域可受限）、缺乏多模态支持（如影像、实验室指标）以及对标准化指南的依赖，导致在非标准化或跨学科场景中的适用性受限。

---

## 499. Multi-Modal Learning meets Genetic Programming: Analyzing Alignment in Latent Space Optimization

**arXiv ID:** 2604.08324 | [PDF](https://arxiv.org/pdf/2604.08324v1)

**作者:** Benjamin Léger `[一作]` (IID / Mila, Université Laval), Christian Gagné `[通讯]` (Canada-CIFAR AI Chair IID / Mila, Université Laval)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过实验验证了SNIP多模态潜在空间优化（LSO）在符号回归中的行为，检验其是否利用跨模态对齐来引导符号搜索，并评估该对齐的细粒度程度。

**💡 创新点**

创新点在于系统揭示了多模态LSO的两个关键瓶颈：算法未能利用预训练的跨模态对齐，且该对齐过于粗糙，无法区分结构相似的表达式，并提出了对应的检验方法与基准。

**🔧 技术方法**

采用了SNIP的预训练符号/数值Transformer编码器、InfoNCE对齐学习、Grey Wolf优化器进行连续空间搜索、BFGS常数微调，并通过对齐与R²随迭代变化的跟踪以及检索任务来评估对齐细粒度。

**📊 数据集**

使用了SRBench中的Feynman与Strogatz方程组，以及与SNIP训练分布一致的100条合成表达式作为测试集。

**📈 对比分析**

与随机排名基线以及传统GP检索率对比，实验显示对齐随优化保持平稳甚至下降，检索准确率仅为18%（Feynman）/23%（合成），均低于随机基线，表明目前的对齐与搜索效果不足。

**⚠️ 局限性**

主要局限在于对齐细粒度不足，导致无法区分结构相似的表达式，算法未能利用对齐信息，导致符号搜索效果差；需改进对齐质量与优化算法以实现有效的多模态LSO。

---

## 500. SeLaR: Selective Latent Reasoning in Large Language Models

**arXiv ID:** 2604.08299 | [PDF](https://arxiv.org/pdf/2604.08299v1)

**作者:** Renyu Fu `[一作]` (Peking University), Guibo Luo `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级、无训练的链式推理框架SeLaR，在大语言模型推理过程中根据自熵阈值选择性激活软嵌入，从而在不扰动高置信步骤的前提下探索多条推理路径。

**💡 创新点**

创新点在于两方面：①熵门控机制仅在低置信（高熵）步骤使用软嵌入，避免了全局软嵌入引起的稳定性下降；②熵感知对比正则化在软嵌入中加入反向作用，抑制其快速收敛至最高概率词，从而持续保持多路径探索。

**🔧 技术方法**

核心技术包括：基于top‑k词的归一化熵计算作为置信度指示、熵门控的软硬嵌入切换、熵加权的对比正则化，以及无训练的推理流程。

**📊 数据集**

在五大推理基准上验证：GSM8K、MATH500、GPQA-Diamond、AIME2024、AIME2025，使用多种规模模型（Qwen3‑1.7B、8B、32B）。

**📈 对比分析**

与标准CoT（采样/贪婪）以及训练‑free的Soft Thinking、SwiR比较，SeLaR在所有模型规模和任务上均实现平均精度提升（最大提升达+13.3%），且在最难的AIME 2024/2025上表现最为显著，同时在推理速度和token数量上也优于竞争方法。

**⚠️ 局限性**

局限性包括：依赖token嵌入空间，表达能力不如直接操作隐藏状态；对基础模型置信度敏感，低置信模型可能触发过多探索，导致性能提升有限；缺少跨模型家族的统一激活策略。

---

## 501. Tokalator: A Context Engineering Toolkit for Artificial Intelligence Coding Assistants

**arXiv ID:** 2604.08290 | [PDF](https://arxiv.org/pdf/2604.08290v1)

**作者:** Vahid Farajijobehdar `[一作]` (Kariyer.net R&D Center), Engin Zeydan `[通讯]` (Centre Tecnològic de Telecomunicacions de Catalunya)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Tokalator 是一套开源工具，集成 VS Code 插件、Web 计算器、MCP 服务器、Python API 等，实时监测和管理 AI 辅助编码中的 token 预算，并通过语法相关性评分自动关闭低效标签页，提供成本估算与优化策略。

**💡 创新点**

其创新点在于：①跨三大模型提供商（Anthropic、OpenAI、Google）提供即时 token 计数与成本拆分；②使用无模型推理的五信号语法评分，实现毫秒级的标签页相关性判断；③将经济学模型（Cobb–Douglas、缓存盈亏、对话成本曲线）包装为交互式计算器，直接面向开发者；④通过 MCP 接口让 Claude Code 等自定义 AI 也能使用本地 BPE 计数；⑤自动发现并展示社区贡献的 agent、prompt 与 instruction 文件。

**🔧 技术方法**

实现技术包括：TypeScript + Node.js 编写 VS Code 扩展，Next.js/React 构建 Web 平台，Prisma+PostgreSQL 存储使用记录；使用 OpenAI、Anthropic、Google 的 BPE tokenizer 进行 token 计数；Python FastAPI 提供 REST 接口；MCP（Model Context Protocol）与 stdio 交互；前端使用 Tailwind、Recharts 等可视化。

**📊 数据集**

评估使用了两类数据：① 50 名开发者的结构化调查和现场演示收集的使用日志；② 1,413 条 GitHub Copilot 计费记录（30 天内）用于验证成本和 token 估算；此外包含 124 条单元测试和 313 次插件安装记录作为实验数据。

**📈 对比分析**

与现有工具相比，Tokalator 在 5 类预算维度上实现实时拆分；语法评分的准确率达 92%（与人工判断一致）；在示例场景中实现 21.2% 的上下文缩减、88.9% 的 API 成本节省，并提供 O(T²) vs O(T) 的对话成本预测。

**⚠️ 局限性**

主要局限包括：token 计数仅为估算，无法访问模型内部上下文构造；语法评分忽略语义相关但无导入的文件；Google 模型采用字符计数近似，误差 10–32%；仅支持 VS Code；缺乏跨会话历史跟踪；价格规则需手动更新；未进行严格的对比实验（仅有 50 人问卷和示例）。

---

## 502. Investigating Performance and Practices with Univariate Distribution Charts

**arXiv ID:** 2604.08378 | [PDF](https://arxiv.org/pdf/2604.08378v1)

**作者:** Laura Lotteraner `[一作]` (University of Vienna), Daniel Pahr `[通讯]` (University of Vienna)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过混合方法研究四类常用的单变量分布图（箱线图、小提琴图、直方图、抖动条形图）的表现与用户体验。

**💡 创新点**

创新点在于采用点击-选择（click‑to‑select）评估框架，将定量准确度与定性解释结合，同时系统比较四类图形在不同任务中的性能与偏好。

**🔧 技术方法**

使用了R语言和ggplot2包生成标准化图形，并通过问卷平台QuestionPro实施在线实验；数据分析采用混合效应模型与逻辑回归。

**📊 数据集**

数据集为八组合成数据（对称与左偏、含离群值、单峰与双峰），每组200个样本。

**📈 对比分析**

比较方法包括六项基准任务（求值、对比、识别范围等），通过点击精度与错误率评估；结果显示：直方图在多任务中表现最稳定，箱线图在中位数检索上最优，抖动条形图在范围识别上最高，且各图对用户偏好与熟悉度不完全一致。

**⚠️ 局限性**

限制包括仅测试默认图形设定，未探讨不同视觉变体对性能的影响；实验任务相对简单，缺乏真实分析情境；专家访谈样本有限，可能无法代表所有学科。

---

## 503. SkillClaw: Let Skills Evolve Collectively with Agentic Evolver

**arXiv ID:** 2604.08377 | [PDF](https://arxiv.org/pdf/2604.08377v1)

**作者:** Ziyu Ma `[一作]` (DreamX Team), Xiangxiang Chu `[通讯]` (DreamX Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SkillClaw框架，实现多用户Agent生态中的技能持续演化。

**💡 创新点**

创新点在于聚合用户会话证据，使用agentic evolver自动推理并更新技能，形成闭环集体学习。

**🔧 技术方法**

主要技术包括结构化会话收集、技能分组聚合、LLM驱动的agentic evolver、夜间验证与同步。

**📊 数据集**

使用WildClawBench（60个跨领域任务）以及自定义查询验证集。

**📈 对比分析**

与基线对比，四个类别平均提升约30–40%，并在6天迭代中持续稳定上升。

**⚠️ 局限性**

局限在用户规模、交互深度与验证成本有限，对高度推理型任务提升有限。

---

## 504. Don't Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents

**arXiv ID:** 2604.08369 | [PDF](https://arxiv.org/pdf/2604.08369v1)

**作者:** Khushal Sethi `[一作]` `[通讯]` (Stanford University), Khushal Sethi (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TrACE，一种训练‑free、基于交叉 roll‑outs 行为一致性（inter‑rollout action agreement）的自适应推理计算控制器，用于在 LLM 代理的每一步动态分配推理调用。

**💡 创新点**

创新点在于：①利用模型自身生成的多次采样结果的一致性作为无监督的难度信号；②在无需训练、外部验证器或人工标注的前提下，按步动态决定调用次数；③通过阈值控制即时提交或追加采样，实现计算量的显著削减。

**🔧 技术方法**

采用的技术包括：多次温度为 0.7 的 LLM 采样、canonicalizer 对动作进行标准化、模式投票（mode）计算一致性 α_t、阈值 τ_high=0.75 决定是否继续采样，以及在 Qwen 2.5 3B Instruct（量化后）上实现；实验环境为 CPU（M‑series）。

**📊 数据集**

使用的数据集：单步数学推理 benchmark GSM8K（随机抽取 50 题）和多步文本导航 benchmark MiniHouse（30 任务、3 种 LLM 采样种子），后者为作者自制、轻量级、无 C++ 依赖的环境。

**📈 对比分析**

与 greedy（k=1）以及自一致性 Self‑Consistency（SC‑4、SC‑8）比较：TrACE‑4 在 GSM8K 上匹配 SC‑4 的 0.82 准确率，同时调用数从 4 降至 2.68（下降 33%）；TrACE‑8 匹配 SC‑8 的 0.84 准确率，调用数从 8 降至 3.56（下降 55%）。在 MiniHouse 上，TrACE‑4 与 TrACE‑8 均保持 0.367 的准确率，但调用数分别下降 39%（对比 SC‑4）和 65%（对比 SC‑8）。墙钟时间也相应从 40 分钟降至 14 分钟。

**⚠️ 局限性**

局限性：①仅在 3B 参数、CPU 版 Qwen 上验证，未测试更大模型或 GPU 加速环境；②样本量有限（GSM8K 仅 50 题、MiniHouse 30 任务），缺乏更广泛的基准覆盖；③阈值 τ_high 的设定基于经验，缺乏系统性调优；④方法主要适用于离散、可枚举动作的环境，对开放式生成任务（如代码写作、长篇规划）可能效果不佳；⑤在小模型下整体准确率受限，无法验证在更强模型上的性能提升。

---

## 505. CapTalk: Unified Voice Design for Single-Utterance and Dialogue Speech Generation

**arXiv ID:** 2604.08363 | [PDF](https://arxiv.org/pdf/2604.08363v1)

**作者:** Xiaosu Su `[一作]` (University of Chinese Academy of Sciences), Jun Gao `[通讯]` (Hello Group Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了从自然语言描述到语音的声线设计，并将其扩展到多轮对话生成；

**💡 创新点**

创新点在于统一的 caption‑conditioned 自回归框架、CoT 动态表达控制以及基于 FHVAE 的层次变分声纹调节机制；

**🔧 技术方法**

采用文本–音频自回归 Transformer、层次变分声纹编码器、CoT 控制序列以及 Qwen3‑Omni 等多模态模型；

**📊 数据集**

使用 300 小时公开数据加 5000 小时内部语料，涵盖演员式与自然对话；对话数据取自内部两人对话的完整会话；

**📈 对比分析**

与 Qwen3TTS、Ming‑omni‑tts、VoiceSculptor、Fish Speech S2 Pro 等对比，单句评测在 InstructTTSEval‑ZH 上平均得分 73.73，且在多轮对话的语音质量、CoT 可控性与上下文连贯性上优于基线；

**⚠️ 局限性**

主要局限在于对细粒度声纹一致性的支持不足，以及在极端情绪与语调变化下的可控性仍待提升。

---

## 506. ASPECT:Analogical Semantic Policy Execution via Language Conditioned Transfer

**arXiv ID:** 2604.08355 | [PDF](https://arxiv.org/pdf/2604.08355v1)

**作者:** Ajsal Shereef Palattuparambil `[一作]` (Deakin University), Santu Rana `[通讯]` (Deakin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种通过语言条件化想象实现的零样本策略迁移框架ASPECT，使得强化学习代理能够在没有目标环境交互的情况下，将已学的源任务策略迁移到与之结构相似但在视觉或语义上不同的新任务。

**💡 创新点**

创新点在于将离散类变量替换为连续的自然语言条件空间，并将大型语言模型作为动态语义算子，在测试时通过LLM将目标任务描述映射为源任务对齐的语义，从而实现对未见对象、视觉变换甚至奖励冲突任务的泛化。

**🔧 技术方法**

使用了文本条件变分自编码器（text‑conditioned VAE）、大型语言模型（如Grok‑4.1、Gemini 2.5 Flash）、CLIP/LongCLIP文本编码器、FiLM 与交叉注意力机制，以及基于感知损失的VAE训练策略。

**📊 数据集**

在MiniGrid、MiniWorld以及自定义的Fragile Object Manipulation三个环境中进行实验；源任务分别为拾取特定颜色/形状的对象，目标任务引入未见对象、视觉纹理变化或奖励逆转。

**📈 对比分析**

与基线（SF‑Simple、SF‑Reconstruction、源策略直接使用、微调策略）比较，ASPECT在三种泛化情形下均实现接近或超过微调后的最高性能，仅通过零样本即可取得 8–9/10 的成功率，显著降低样本复杂度。

**⚠️ 局限性**

主要局限包括：对VAE生成的“想象”图像质量敏感，极端视觉情况可能导致失真；LLM可能产生幻觉或偏见，若映射错误可能导致代理行为失真；系统在需要高安全保障的真实场景中仍需进一步验证和约束。

---

## 507. Analytical Modeling of Dispersive Closed-loop MC Channels with Pulsatile Flow

**arXiv ID:** 2604.08307 | [PDF](https://arxiv.org/pdf/2604.08307v1)

**作者:** Theofilos Symeonidis `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Maximilian Schäfer `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了闭环分散型分子通信通道模型，考虑脉冲流并推导出时间变均值与方差的环状正态分布形式的传播冲激响应（CIR），并通过3D粒子跟踪仿真验证。

**💡 创新点**

首次在闭环MC系统中针对脉冲血流推导出解析CIR；利用Womersley脉冲流模型与一维扩散理论相结合，得到可直接用于通信理论分析的时间变均值与方差表达式。

**🔧 技术方法**

采用Aris‑Taylor扩散理论、一维流动-扩散方程的变换求解、Womersley脉冲流模型、解析求解的环状正态分布表达式以及3D粒子跟踪仿真（PBS）进行验证。

**📊 数据集**

使用合成周期波形（正弦、脉冲）以及从文献拟合得到的生理脉冲波形作为流速输入；并未使用公开实验数据集，全部数据来自自建仿真。

**📈 对比分析**

通过与3D PBS仿真结果比较，解析模型在不同脉冲频率、均值速度、接收位置及扩散系数下均能与仿真曲线高度吻合；相较于稳态模型，解析模型能捕捉脉冲流导致的时域波形变化，验证了脉冲流对接收信号的显著影响。

**⚠️ 局限性**

局限性包括：仅适用于低Womersley数、强分散性（轴向均匀）且假设血管为单一闭环管道；未考虑多分支网络、空间流场变异或脉冲衰减；缺乏实验验证，模型对非牛顿流体或高流速情况的准确性待进一步评估。

---

## 508. Lost in the Hype: Revealing and Dissecting the Performance Degradation of Medical Multimodal Large Language Models in Image Classification

**arXiv ID:** 2604.08333 | [PDF](https://arxiv.org/pdf/2604.08333v1)

**作者:** Xun Zhu `[一作]` (Tsinghua University), Ji Wu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

对 14 种医学多模态大型语言模型（MLLM）在乳腺超声、COVID‑19 CT 以及儿童胸 X 光肺炎三类影像分类任务上，进行层级与模块级的特征探测，系统地剖析了分类性能下降的根源。

**💡 创新点**

首次提出四类失效模式（视觉表示质量、连接器失真、LLM 理解缺陷、语义映射不匹配）并引入“特征健康分数”（Feature Health Score，FHS）量化各模块信息保留与演化，构建了面向多模态模型的诊断框架。

**🔧 技术方法**

采用特征探测技术（冻结模型、轻量化 MLP 探针）、LoRA 微调、交叉熵对比自回归训练、FHS 计算以及数据集切片实验等方法，对模型内部信息流进行细粒度评估。

**📊 数据集**

使用三大公开医学影像数据集：乳腺超声图像集（BUSI）、COVID‑19 CT 图像集以及儿童胸 X 光肺炎筛查集。

**📈 对比分析**

通过与传统 CNN/MLP/VIT/混合网络的基准对比，并利用 FHS 评估各模块健康度，实验表明医学 MLLM 尽管参数规模巨大，但整体分类准确率仅略高或与传统方法持平，显示出显著的性能瓶颈。

**⚠️ 局限性**

主要局限包括：医学领域适配提升有限；视觉塔在自回归目标下的表示质量不足；LLM 的语义映射与分类任务不匹配；多模态信息流失导致整体性能受限。

---

## 509. GroundingAnomaly: Spatially-Grounded Diffusion for Few-Shot Anomaly Synthesis

**arXiv ID:** 2604.08301 | [PDF](https://arxiv.org/pdf/2604.08301v1)

**作者:** Yishen Liu `[一作]` (Beijing Institute Of Technology), Dongpu Cao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出GroundingAnomaly框架，实现少样本下空间对齐的异常图像生成；

**💡 创新点**

创新点包括：①空间条件模块（Spatial Conditioning Module）将像素级语义图与解耦的产品/异常token融合，实现精准的空间和语义控制；②门控自注意力模块（Gated Self-Attention Module）在冻结的U-Net中注入条件，从而实现稳定的少样本适配；③混合正常-异常训练与正常前置去噪初始化（NDI）提升背景一致性与生成效率；

**🔧 技术方法**

技术主要基于扩散模型（Latent Diffusion）、Stable Diffusion v1.4、ControlNet式的空间条件、门控自注意力、LoRA低秩更新、语义图编码（ConvNeXt-Tiny）等；

**📊 数据集**

使用MVTec AD和VisA两个工业缺陷数据集进行实验；

**📈 对比分析**

与现有AG、AIG方法（DFMGAN、AnomalyDiffusion、DualAnoDiff、SeaS）以及多种下游检测模型（PatchCore、ViTAD、MambaAD等）比较。GroundingAnomaly在生成质量（IS、IC‑LPIPS）和下游任务（像素级分割AUROC/AP、实例级mAP）上均取得领先或近似最佳性能，尤其在多类别、多样本下表现突出；

**⚠️ 局限性**

局限性：仍依赖一定数量的异常样本进行微调，单次噪声初始化和语义图的构造对少样本极限仍有挑战；生成分辨率受扩散模型限制；跨域未覆盖所有复杂工业场景，需进一步研究零样本或更高分辨率生成。

---

## 510. U-CECE: A Universal Multi-Resolution Framework for Conceptual Counterfactual Explanations

**arXiv ID:** 2604.08295 | [PDF](https://arxiv.org/pdf/2604.08295v1)

**作者:** Angeliki Dimitriou `[一作]` (National Technical University of Athens), Giorgos Stamou `[通讯]` (National Technical University of Athens)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 U-CECE 框架，能够在原子、关系（sets‑of‑sets）和完整结构（scene graph）三个层次上生成概念层面的反事实解释。

**💡 创新点**

创新点在于：① 统一多分辨率、可自适应的框架；② 结合转导式（监督）和归纳式（无监督）检索模式，既保证精度又提升可扩展性；③ 在结构层使用近似 Graph Edit Distance 的 GNN 进行检索，并证明其语义一致性优于精确 GED。

**🔧 技术方法**

主要技术包括：Graph Neural Networks（GCN/GAT/GIN）、Siamese GNN 监督学习、Graph Autoencoders、Graph Edit Distance 近似、Transductive/Inductive 训练策略、以及 Large Vision‑Language Models（LVLM）做评估。

**📊 数据集**

使用的公开数据集为 CUB（鸟类细粒度数据）和 Visual Genome（VG‑DENSE、VG‑RANDOM 两个子集）来构造概念与图结构。

**📈 对比分析**

评价方法：P@1、nDCG、编辑量（node/edge）等指标；实验显示：在稠密图（VG‑DENSE）中 L3 结构层优于 L1/L2；在稀疏图（VG‑RANDOM）和 CUB 上 L1/L2 与 L3 相近；人类与 LVLM 评估表明 L3 检索得到的反事实在语义一致性上往往优于精确 GED，且在多数任务中获得更高的偏好率。

**⚠️ 局限性**

限制：① 依赖高质量的概念抽象、关系解析和图构造，抽象误差会直接影响解释质量；② 人类实验仅限于鸟类数据，未覆盖更广泛场景；③ LVLM 与人类的一致性有限，受任务结构影响；④ 框架未涉及对抗性鲁棒性、公平性、可操作性等更深层次问题。

---

## 511. CIAO - Code In Architecture Out - Automated Software Architecture Documentation with Large Language Models

**arXiv ID:** 2604.08293 | [PDF](https://arxiv.org/pdf/2604.08293v1)

**作者:** Marco De Luca `[一作]` (University of Naples Federico II), Patrizio Pelliccione `[通讯]` (Gran Sasso Science Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了CIAO——一种利用LLM从完整GitHub仓库自动生成符合ISO/IEC/IEEE 42010、SEI Views & Beyond及C4模型的系统级架构文档的流程。

**💡 创新点**

创新点在于将标准化的文档模板与多级提示工程相结合，使LLM能够在一次性生成完整、结构化的架构描述，而非仅生成局部注释或API文档。

**🔧 技术方法**

核心技术包括GPT‑4 LLM、Repomix仓库扁平化、结构化提示工程、PlantUML文本转图像渲染以及对ISO/IEC/IEEE 42010、SEI Views & Beyond和C4模型的模板化集成。

**📊 数据集**

使用了22个不同领域（如IoT、机器学习、网络安全等）的开源GitHub仓库作为评估数据集，涵盖多语言（Python、Java、C/C++、JavaScript等）和多规模项目。

**📈 对比分析**

通过问卷评估和定量指标（平均生成时间约3分钟、API费用约1.19美元），与现有CodeDocs‑GenAI等工具相比，CIAO在文档完整性、可读性和可操作性方面获得了开发者的高满意度。

**⚠️ 局限性**

主要局限在于生成的图表（类图、部署图等）易出现缺失或不准确，部署视图细节不足，且部分高层叙述仍可能存在歧义，需结合静态/动态分析或人工复核进一步提升质量。

---

## 512. VCAO: Verifier-Centered Agentic Orchestration for Strategic OS Vulnerability Discovery

**arXiv ID:** 2604.08291 | [PDF](https://arxiv.org/pdf/2604.08291v1)

**作者:** Suyash Mishra `[一作]` `[通讯]`, Suyash Mishra

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 VCAO，一个基于大规模推理模型（LRM）驱动的、采用 Bayesian Stackelberg 游戏的多工具协同操作系统漏洞挖掘框架。

**💡 创新点**

创新点在于将漏洞挖掘建模为重复 Bayesian Stackelberg 搜索游戏，设计 DOBSS-VD MILP 进行预算分配，构建级联验证器和六层架构，并提供在线 regret O(√T) 的正式保障。

**🔧 技术方法**

使用技术包括 Bayesian Stackelberg 安全游戏、DOBSS MILP 求解、LLM/Large Reasoning Model 编排、内核攻击图构建、贝叶斯信念更新、在线学习（EXP3/Thompson Sampling）、LLM 生成攻击链和级联验证流程。

**📊 数据集**

实验数据集涵盖五个 Linux 内核子系统（文件系统、网络、命名空间/权限、驱动、io_uring/BPF），重放 847 条历史 CVE，并在最新内核快照上进行实时验证；利用 CVSS、历史缺陷密度等先验信息。

**📈 对比分析**

与均匀分配、提交频率、仅 fuzzing、仅静态分析、非游戏多代理、VCAO 无兄弟搜索等基线比较；VCAO 在验证漏洞收益上比 fuzzing 高 2.7 倍、比静态分析高 1.9 倍，误报率下降 68%，同时实现在线 regret O(√T) 的子线性收敛。

**⚠️ 局限性**

局限性包括：假设攻击者理性（对非理性攻击者不稳健）；攻击图需人工校验特权边界；工具观测模型需针对每个内核版本单独校准；攻击图构建复杂，规模扩展受限。

---

## 513. CAMotion: A High-Quality Benchmark for Camouflaged Moving Object Detection in the Wild

**arXiv ID:** 2604.08287 | [PDF](https://arxiv.org/pdf/2604.08287v1)

**作者:** Siyuan Yao `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了CAMotion，一个覆盖151种伪装物种、包含474条视频、约150,000帧、30,028帧像素级标注的大规模野生伪装移动物体检测基准数据集。

**💡 创新点**

创新点在于：①提供了前所未有的规模与物种多样性；②对每帧标注了8种挑战属性（如不确定边缘、遮挡、形状复杂等）；③通过多轮人工校验确保标注质量；④对现有SOTA模型进行系统跨数据集评估，揭示了小物体、遮挡等挑战。

**🔧 技术方法**

采用了多尺度像素级标注、属性标注、跨数据集训练与测试、光流与深度可视化等技术，并在实验中使用18种COD/VCOD模型（13图像模型+5视频模型）进行对比。

**📊 数据集**

主要使用自建的CAMotion数据集，并与现有的MoCA-Mask、COD10K等数据集进行对比评估。

**📈 对比分析**

对18个SOTA模型在6项指标（S_α、F_β^w、E_ϕ^m、ℳ、mDic、mIoU）上进行严格评估，结果显示：在CAMotion上图像模型HGINet取得最高分，甚至超过视频模型ZoomNeXt；但在MoCA-Mask上视频模型表现更佳；整体上CAMotion对模型提出更高的挑战，主要难点集中在小物体、遮挡与不确定边缘。

**⚠️ 局限性**

局限性在于现有模型难以同时兼顾伪装物体的辨别能力与时间一致性；静态COD模型缺乏时序信息，易导致预测不连贯；而VCOD模型在伪装辨识上存在欠缺，需研发统一端到端框架以解决这一权衡。

---

## 514. Distributed Multi-Layer Editing for Rule-Level Knowledge in Large Language Models

**arXiv ID:** 2604.08284 | [PDF](https://arxiv.org/pdf/2604.08284v1)

**作者:** Yating Wang `[一作]` (Shandong University), Haoliang Sun `[通讯]` (Shandong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种针对大型语言模型规则级知识编辑的新框架 DMLE，能够在不同层级上针对公式、描述和实例三种形式进行分布式更新

**💡 创新点**

创新点在于发现并利用规则知识在 transformer 各层的形式特定存储分布，提出在早期层共享编辑、在中间层独立编辑，显著提升规则一致性与实例可迁移性

**🔧 技术方法**

采用 MEMIT 基础的闭式权重更新，结合因果追踪（causal tracing）技术分析层级贡献，再通过最小二乘求解实现分层编辑

**📊 数据集**

构建了 RuleEdit‑200 数据集，包含 200 条手工验证的数学与物理规则，每条规则都有对齐的公式、描述和实例三种形式

**📈 对比分析**

与 ROME、MEMIT、GRACE、PROMPT 等基线对比，DMLE 在规则级编辑指标（实例可迁移性 IP 与规则理解 RU）上平均提升 13.91 与 50.19 个百分点，同时保持与基线相当的可靠性、泛化性和局部性

**⚠️ 局限性**

局限性包括仅在自回归模型上验证，编辑策略仍依赖于手工划分的层区间，且对更大规模模型或多模态知识的适用性尚未探索

---

## 515. Navigating Turbulence: The Challenge of Inclusive Innovation in the U.S.-China AI Race

**arXiv ID:** 2604.08353 | [PDF](https://arxiv.org/pdf/2604.08353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 516. City-Scale Visibility Graph Analysis via GPU-Accelerated HyperBall

**arXiv ID:** 2604.08374 | [PDF](https://arxiv.org/pdf/2604.08374v1)

**作者:** Alex Hodge `[一作]` (Independent Researcher), Melissa Barrientos Trinanes `[通讯]` (University Of Leeds)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了一套可扩展的城市级可视化图分析（VGA）系统，使得在大型城市范围内进行VGA成为可能。

**💡 创新点**

创新点在于将 delta‑compressed CSR 存储、基于 HyperLogLog 的 HyperBall 近似 BFS 以及 GPU 加速的流式解码/联合核融合三种技术结合，显著提升了VGA的规模与速度。

**🔧 技术方法**

所用技术包括 LEB128 可变长整数编码、HyperLogLog 计数器、HyperBall 迭代距离估计、CUDA 并行核（融合解码-联合核、计数、累积核）以及 Hilbert 空间填充曲线重排序和 Rayon's 并行构建。

**📊 数据集**

数据集为智利 Valdivia 城市的 OSM 建筑边界与行政边界，实验覆盖不同半径、网格间距的多规模子区域，最终在 2.7M 节点、12.1B 边的全市级图上进行验证。

**📈 对比分析**

通过与传统 depthmapX 的端到端时间对比，p=10 时在 42705 节点的最大匹配配置下实现 239× 加速，且能在 236000 节点（4.8B 边）下在 137 秒内完成；在深度限制为 3 时 BFS 阶段更是 352× 加速。

**⚠️ 局限性**

限制主要在于：仅支持二维视线模型（不含地形或 3D 高度信息），依赖 CUDA GPU，HyperBall 只能提供距离和积分等汇总指标，无法给出完整的距离分布，且精度受 HLL 误差影响，需要根据 p 值平衡速度与准确度。

---

## 517. SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction

**arXiv ID:** 2604.08370 | [PDF](https://arxiv.org/pdf/2604.08370v1)

**作者:** Chensheng Dai `[一作]` (Tsinghua University), Yueqi Duan `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SurfelSplat，利用 Nyquist 采样原理的 feed-forward 框架，从稀疏视角快速生成表面对齐的 2D 高斯表面。

**💡 创新点**

创新点在于将频域低通滤波与跨视角特征聚合相结合，使高斯表面满足 Nyquist 采样条件，从而显著提升几何精度并加速推理。

**🔧 技术方法**

采用自适应低通滤波、跨视角特征聚合网络、Transformer 跨注意机制、深度与属性预测网络、LPIPS 与几何损失以及 Nyquist 采样率分析等技术。

**📊 数据集**

主要使用 DTU 数据集进行稀疏视角重建（15 个测试场景），并在 RealEstate10K 上预训练，BlendedMVS 用于补充实验。

**📈 对比分析**

与 NeuS、NeuSurf、VolRecon、UFORecon、2DGS、GausSurf、FatesGS 等方法对比，DTU 2 视角下平均 Chamfer Distance 达 1.12 mm（最佳），推理时间仅 1 秒，速度提升约 100 倍，几何误差显著下降。

**⚠️ 局限性**

局限在于高分辨率图像会产生百万级高斯表面导致渲染与推理速度下降，对不可见区域的重建仍受限，未来可考虑引入生成模型解决这些问题。

---

## 518. Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces

**arXiv ID:** 2604.08362 | [PDF](https://arxiv.org/pdf/2604.08362v1)

**作者:** Jiawei Chen `[一作]` (Chinese Academy of Sciences), Hongyu Lin `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了 OmniBehavior 基准，用真实用户三个月的跨场景长序行为日志评估 LLM 生成用户模拟器。

**💡 创新点**

首次提出全实测数据基准，覆盖跨场景、长时序、异质行为，并系统揭示 LLM 的“积极平均”结构偏差。

**🔧 技术方法**

利用多种 LLM（Claude、Gemini、GPT、GLM、Qwen 等）进行用户行为预测，配合 LangChain 上下文管理、Qwen2.5-72B 进行文本清洗和匿名化。

**📊 数据集**

使用 Kuaishou 平台 200 名代表用户的真实日志，覆盖 5 个场景 22 种行为，日志时长 3 个月，平均序列长度 8,143 条。

**📈 对比分析**

在 6,000 条预测任务中对比多模型，最优 Claude-Opus-4.5 得到 44.55 总分，其他模型大多落在 32–41 分区间，扩展上下文长度和记忆机制并未显著提升性能。

**⚠️ 局限性**

模型存在积极平均、行为同质化、对负面/长尾行为预测不足，长上下文推理能力有限，难以真实模拟人类多元行为。

---

## 519. Human-AI Collaboration Reconfigures Group Regulation from Socially Shared to Hybrid Co-Regulation

**arXiv ID:** 2604.08344 | [PDF](https://arxiv.org/pdf/2604.08344v1)

**作者:** Yujing Zhang `[一作]` (University of Hong Kong), Jionghao Lin `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验中比较了人类-人工智能组与人类-人类组完成协作任务时的协作调控模式，评估了生成式人工智能（GenAI）对协作调控结构的影响。

**💡 创新点**

首次将协作调控细分为 episode‑level 调控模式、调控过程以及 utterance‑level 参与焦点三层级编码，系统性检验 GenAI 在协作学习中如何重新配置协作调控责任与过程。

**🔧 技术方法**

利用生成式 AI 对话代理提供调控支持，并使用混合设计 ANOVA 及 Holm 调整的 t 检验对比两组的调控比例差异；编码工具采用基于 Nguyen 等人和 Dang 等人的调控与参与焦点分类体系。

**📊 数据集**

研究基于 71 名大学生（18–35 岁）组成的 24 个小组（23 组三人组 + 1 组双人组），完成两项任务：学术违规道德推理与购物规划任务。

**📈 对比分析**

通过比较 Human‑AI 与 Human‑Human 两组在 episode‑level 调控模式（CoRL、SSRL、混合）、调控过程（情感支持、障碍检测、策略指令等）以及 utterance‑level 参与焦点的比例，结果显示 Human‑AI 组显著提升 CoRL 与混合模式比例，情感支持、障碍检测、策略指令比例显著高于 Human‑Human 组，而参与焦点分布差异不显著。

**⚠️ 局限性**

样本规模有限、文化与语言单一（多数为中文母语的亚洲学生）、仅采用文本交互，且只涉及两类任务，限制了结果的普适性与多模态情境下的可推广性。

---

## 520. A Unified Multi-Layer Framework for Skill Acquisition from Imperfect Human Demonstrations

**arXiv ID:** 2604.08341 | [PDF](https://arxiv.org/pdf/2604.08341v1)

**作者:** Zi-Qi Yang `[一作]` (Western University), Mehrdad R. Kermani `[通讯]` (Western University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种三层控制框架，实现从单次示范学习到全身可耦合的安全执行，包含实时学习、零空间优化与零空间顺应控制，验证于7自由度KUKA LWR机器人。

**💡 创新点**

创新点包括：1）单示范学习可获得轨迹与可变阻尼，提升效率与逼真度；2）将三维Fast Diffeomorphic Matching与EKF结合，实时修正轨迹；3）零空间优化主动避免奇异、统一交互感受；4）零空间顺应实现全身安全、主任务不受影响。

**🔧 技术方法**

技术手段包括：3D Fast Diffeomorphic Matching、扩展卡尔曼滤波器、可变阻尼控制、零空间优化（方向操纵性、惯性统一、肘部奇异抑制）以及零空间阻尼控制。

**📊 数据集**

使用LASA手写数据集（扩展至3D）作为示范轨迹，外部F/T传感器用于测量扰动与交互力。

**📈 对比分析**

与传统FDM、GC（重力补偿）等方法比较；学习误差<0.3 cm，重现误差<2 cm；零空间优化后交互力在各方向上更为等效；零空间顺应能将冲击能量在冗余关节中消散，主任务误差<5 mm，表现出更安全、更一致的执行。

**⚠️ 局限性**

局限性：未进行系统消融与超参敏感度分析；实验仅在简单路径场景中验证；与更多先进方法对比不足；对复杂环境与多任务的适用性待进一步研究。

---

## 521. Asynchronous Quantum Distributed Computing: Causality, Snapshots, and Global Operations

**arXiv ID:** 2604.08298 | [PDF](https://arxiv.org/pdf/2604.08298v1)

**作者:** Siddhartha Visveswara Jayanti `[一作]` (Dartmouth), Anand Natarajan `[通讯]` (MIT)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了异步量子分布式系统中原子量子全局操作的实现方案——QGO算法

**💡 创新点**

将经典Chandy–Lamport快照算法迁移到量子域，扩展Lamport的计算因果关系概念，并证明其在量子系统中的有效性

**🔧 技术方法**

量子分布式计算模型、量子操作与测量的线性映射、计算因果关系定义与等价性证明

**📊 数据集**

未使用具体数据集，本文为理论性分析

**📈 对比分析**

未进行实验对比或性能评估，结论基于形式化证明

**⚠️ 局限性**

对并发全局操作的支持有限，且对量子信息在分布式中的测量结果随机性未给出完整概率分布匹配讨论

---

## 522. Weakly-Supervised Lung Nodule Segmentation via Training-Free Guidance of 3D Rectified Flow

**arXiv ID:** 2604.08313 | [PDF](https://arxiv.org/pdf/2604.08313v1)

**作者:** Richard Petersen `[一作]` (Chalmers University of Technology), Jennifer Alvén `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于预训练3D rectified flow与弱监督预测器的训练无关肺结节分割方法，利用引导生成对抗样本得到分割掩码。

**💡 创新点**

创新点在于将训练自由引导（TFG）与预训练生成模型无缝结合，避免了对生成模型的重新训练，实现在仅有图像级标签下的3D弱监督分割。

**🔧 技术方法**

使用MAISI-v2 3D rectified flow、MedSAM TinyViT或RadImgNet预测器，结合逆向Euler/前向Euler采样和梯度引导实现对抗生成。

**📊 数据集**

在LUNA16胸部CT数据集上进行10折交叉验证评估。

**📈 对比分析**

与CAM、Grad‑CAM、Score‑CAM、Integrated Gradients等弱监督方法比较，使用Dice和平均表面距离评估，平均Dice分别达到42.05%（MedSAM）和35.01%（RadImgNet），表明性能优于传统方法。

**⚠️ 局限性**

局限性包括对预测器性能高度依赖，在RadImgNet下表现下降；对超参数（引导强度、时间步数）敏感；仅在肺结节和LUNA16数据集上验证，缺乏跨模态或其他疾病的泛化性。

---

## 523. DMax: Aggressive Parallel Decoding for dLLMs

**arXiv ID:** 2604.08302 | [PDF](https://arxiv.org/pdf/2604.08302v1)

**作者:** Zigeng Chen `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DMax 并改进了扩散语言模型的并行解码，显著提升推理吞吐量并减少误差积累。

**💡 创新点**

核心创新在于：① On‑Policy Uniform Training（OPUT）将预训练的掩码扩散模型转化为自校正模型；② Soft Parallel Decoding（SPD）通过软嵌入插值和块级自回归解码，进一步降低错误传播。

**🔧 技术方法**

技术包括：扩散语言建模、掩码‑到‑标记解码、统一扩散训练、基于策略采样的 on‑policy 训练、软嵌入混合、块级半自回归解码、置信阈值控制。

**📊 数据集**

使用 LLaDA‑2.0‑mini 作为基础模型，分别在数学推理任务（GSM8K、MATH500、Minerva‑Algebra、ASDIV）和代码生成任务（HumanEval‑Instruct、MBPP‑Instruct）上进行自蒸馏训练，数据来源于公开提示集与模型自生成的答案。

**📈 对比分析**

与 LLaDA‑2.0‑mini、层次解码、dParallel‑SFT 及传统统一扩散训练进行对比。DMax‑Math 在 GSM8K 上 TPF 从 2.04 提升至 5.48，TPS 由 512 提升至 1258，准确率保持 92.1%；DMax‑Coder 在 MBPP 上 TPF 由 2.71 提升至 5.86，TPS 由 662 提升至 1264，准确率保持 79.2%。整体上吞吐量提升约 2–3 倍，同时保持或略低于基线准确率。

**⚠️ 局限性**

局限性包括：① 需要大量 GPU 训练资源；② OPUT 训练对模型自身生成质量敏感，可能导致自蒸馏误差积累；③ SPD 只能在 OPUT 训练的模型上有效；④ 对于极端低置信阈值或高度异质数据时仍可能出现准确率下降；⑤ 目前仅在 LLaDA‑2.0‑mini 上验证，缺乏更大规模模型或跨任务的通用性评估。

---

## 524. Robust Multi-Objective Optimization for Bicycle Rebalancing in Shared Mobility Systems

**arXiv ID:** 2604.08296 | [PDF](https://arxiv.org/pdf/2604.08296v1)

**作者:** Diego Daniel Pedroza-Perez `[一作]` (ITIS Software, University of Malaga), Jamal Toutouh `[通讯]` (ITIS Software, University of Malaga)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了三目标静态夜间自行车再平衡模型，加入高需求情景下的未满足需求作为鲁棒性目标，并通过NSGA-II多目标进化算法与专用变异算子进行求解。

**💡 创新点**

创新点包括：①将高需求场景的未满足需求作为第三目标，实现鲁棒性优化；②设计Permutation–partition编码和BB1-MAX变异算子，结合最佳提升与多样化；③使用情景模拟的确定性补偿策略，避免可行性修复。

**🔧 技术方法**

采用NSGA-II多目标进化算法，配合Permutation–partition编码，定制AB2、BB1-MIN/MAX、BB2-MIN/MAX等移位变异算子；通过情景模拟评估解决方案；实现语言为Python，使用NetworkX、OSMnx、PyMOO等库。

**📊 数据集**

基于巴塞罗那Bicing系统460个站点的真实数据（2019‑2025年每4‑5分钟采样），提取周一07:00‑08:00的需求，生成15个训练情景和15个验证情景，计算高需求阈值。

**📈 对比分析**

通过与贪心基线（RRCP‑BI、GLOBE）及消融版本比较，使用rhv、IGD+、GD+、Spread、#nds等指标；结果显示BB1-MAX在所有指标上均优于其他变异算子，贡献最多Pareto前沿；贪心方法仅提供极端解，性能明显落后。

**⚠️ 局限性**

局限性包括：仅考虑单周期夜间静态再平衡；实验仅在NSGA-II上验证，未评估其他MOEA；需求建模基于历史统计，未利用机器学习捕捉时序动态；仅针对巴塞罗那案例，未验证在其他城市的泛化；大规模实例求解成本较高。

---

## 525. SOLAR: Communication-Efficient Model Adaptation via Subspace-Oriented Latent Adapter Reparametrization

**arXiv ID:** 2604.08368 | [PDF](https://arxiv.org/pdf/2604.08368v1)

**作者:** Seyed Mahmoud Sajjadi Mohammadabadi `[一作]` (University of Nevada, Reno), Junshan Zhang `[通讯]` (University of California, Davis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练压缩方法 SOLAR，用于压缩 PEFT 模型的适配器参数，以降低通信和存储成本。

**💡 创新点**

创新点在于利用基础模型权重的奇异向量构建对齐的随机基底，结合稀疏选择，将适配器更新重参数化为稀疏线性组合，从而实现极高压缩率且保持性能。

**🔧 技术方法**

使用 SVD 产生对齐基底、稀疏系数选择、线性回归求解、随机种子重构，以及对 PEFT（LoRA、NOLA 等）适配器的后处理。

**📊 数据集**

在多种视觉任务（ViT‑B/L 在 CIFAR‑10/100、Food‑101、ImageNet‑1K 等）和语言任务（LLaMA‑3 在 Alpaca、MMLU；GPT‑2 在 E2E NLG）上进行实验。

**📈 对比分析**

与全微调、LoRA、NOLA 等基线对比，SOLAR 在保持或略低误差的同时将适配器参数压缩至 2–10% 甚至更低（最大压缩率 98%），通信和存储开销显著下降，且运行时开销极小。

**⚠️ 局限性**

局限性在于压缩效果受基准适配器质量限制，需要针对不同任务调优基底数 N 与稀疏预算 k，且对音频、时间序列或多模态任务的适用性尚未验证。

---

## 526. Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constrained Policy Optimization

**arXiv ID:** 2604.08476 | [PDF](https://arxiv.org/pdf/2604.08476v1)

**作者:** Sai Srinivas Kancheti `[一作]` (IIT Hyderabad), Tanuja Ganu `[通讯]` (Microsoft Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多模态强化学习中推理质量下降问题，提出 Faithful GRPO 通过 Lagrangian 双重上升和分离归一化强制一致性和视觉锚定，从而提升推理可靠性和答案准确度。

**💡 创新点**

将一致性与视觉锚定作为硬约束而非软奖励，并利用分离归一化与 Lagrange 多重上升实现自适应约束满足，解决了 RLVR 训练中“奖励黑客”导致的推理不可靠问题。

**🔧 技术方法**

采用组相对策略优化 (GRPO)、Lagrangian Dual Ascent、分离归一化、LLM 判定一致性、VLM 判定语义与空间锚定、CIoU 匹配以及两阶段 SFT+RL 训练。

**📊 数据集**

使用 SAT、VGR、VisCoT、TreeVGR-RL-37K、七个空间推理基准（CVBench-2D/3D、MindCube、MMVP、OmniSpatial、RealWorldQA、SAT-Real）以及 COCO、GQA、OpenImages、Flickr30k 等图像来源的数据集。

**📈 对比分析**

与无约束 GRPO、五种现有多模态推理模型以及 GPT 系列模型比较，FGRPO 在 7B/3B backbone 上平均准确率提升约 2%，不一致率从 26.1% 降至 1.7%，语义锚定提升 13 个百分点，证明了可信推理与答案准确性互补。

**⚠️ 局限性**

需要额外的 LLM/VLM 判定开销，对可验证奖励的准确性依赖，空间约束仅适用于有 Bounding Box 注释的数据，阈值设置对性能影响较大，且在非空间推理任务的通用性尚未验证。

---

## 527. LAMP: Lift Image-Editing as General 3D Priors for Open-world Manipulation

**arXiv ID:** 2604.08475 | [PDF](https://arxiv.org/pdf/2604.08475v1)

**作者:** Jingjing Wang `[一作]` (Zhejiang University), Guofeng Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于图像编辑模型的三维先验提取方法LAMP，利用单目RGB-D观测和语言指令生成编辑后目标图像，并将其提升到3D点云，计算活跃物体与被动物体的SE(3)变换，作为开世界操作的连续几何先验；

**💡 创新点**

创新点在于：①将二维图像编辑中的隐式空间线索转换为三维交互变换；②设计了基于DINO特征的跨状态点云注册与统一尺度校正；③通过精细的层次化点云过滤提升鲁棒性；④实现零样本、可视化可解释的开放世界操控；

**🔧 技术方法**

技术手段包括：现代图像编辑模型（Qwen-Image-Edit、Gemini 2.5 Flash）; 单目深度估计（VGGT）；DINOv3特征与K-Means+DBSCAN分层过滤；Umeyama算法求解SE(3)；统一尺度约束；CuRobo路径规划；

**📊 数据集**

使用的主要数据集为自采集的单目RGB-D场景，涵盖13类日常物体操作任务；对齐标定用AR-Code扫描的物体网格；Ground-truth变换来自FoundationPose；对比实验使用VoxPoser、CoPA、ReKep、Two-by-Two、AnyPlace等公开模型；

**📈 对比分析**

方法通过与上述基线进行对比，结果显示LAMP在13项任务的平均成功率达66%，显著高于VoxPoser（13%）、CoPA（24%）和ReKep（30%）；在点云配准上，旋转RMSE仅0.015°，平移RMSE 0.005m，优于Baseline；在视角变化、长周期任务以及对视频生成先验的比较中也表现出更好的稳健性和精度；

**⚠️ 局限性**

局限性包括：仅适用于刚体交互，无法处理柔性或可变形物体；依赖运动规划，需要额外的轨迹先验；编辑模型的输出可能出现误编辑或不可忽视的视觉偏差，需要手动提示工程；整体延迟主要受图像编辑推理阶段影响；

---

## 528. OVS-DINO: Open-Vocabulary Segmentation via Structure-Aligned SAM-DINO with Language Guidance

**arXiv ID:** 2604.08461 | [PDF](https://arxiv.org/pdf/2604.08461v1)

**作者:** Haoxi Zeng `[一作]`, Heng Tao Shen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 OVS-DINO 框架，通过结构对齐 SAM 的边界先验来提升 DINO 的边界感知，实现开放词汇图像分割。

**💡 创新点**

创新点在于设计结构感知编码器 SAE 与结构调制解码器 SMD，并加入 Preservation Gate，使 SAM 的结构信息能够迁移到 DINO，且在推理阶段不再需要 SAM，从而兼顾语义一致性与细粒度边界。

**🔧 技术方法**

主要技术包括 DINOv2 ViT 视觉编码器、SAM 对齐特征、CLIP 文本嵌入、频谱分析、CKA 相关性、Dice+BCE 损失等。

**📊 数据集**

实验使用 VOC20/21、Context59/60、COCO-Stuff、Cityscapes、ADE20K、COCO-Object 等八个公开的零样本分割基准数据集。

**📈 对比分析**

与多种弱监督开放词汇分割方法对比，OVS-DINO 在平均 mIoU 上提升 2.1%，在 Cityscapes、COCO-Stuff 等复杂场景提升 6% 以上，达到或逼近 SOTA。

**⚠️ 局限性**

局限性包括：对极小细节和遮挡场景的分割仍存在误差；模型训练仍需依赖 SAM 产生的伪标签，伪标签质量影响最终性能；虽然推理阶段无需 SAM，但训练过程仍增加计算成本。

---

## 529. Taming GPU Underutilization via Static Partitioning and Fine-grained CPU Offloading

**arXiv ID:** 2604.08451 | [PDF](https://arxiv.org/pdf/2604.08451v1)

**作者:** Gabin Schieffer `[一作]` (KTH Royal Institute of Technology), Ivy Peng `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

系统性评估了多实例 GPU（MIG）、多进程服务（MPS）和时间切片三种 GPU 共享方式在科学计算、LLM、数据分析等真实工作负载中的计算和内存资源利用率，并提出基于 Nvlink‑C2C 的内存卸载方案来缓解 MIG 的粗粒度资源分配不匹配问题。

**💡 创新点**

首次在 HPC 应用场景下量化 GPU 资源闲置、提出将 MIG 资源利用与性能权衡的奖励模型、揭示功率抑制在共享环境中的交互干扰，并通过内存卸载实现对 MIG 资源连续性提升的创新解决方案。

**🔧 技术方法**

利用 NVIDIA Grace Hopper GPU 的 MIG、MPS、时间切片以及 Nvlink‑C2C 互连，采集 GPM 监控数据、使用 NVML 收集功耗、实现基于显存溢出的内存卸载，结合奖励函数评估配置选择。

**📊 数据集**

使用包括 NekRS、LAMMPS、Llama3、Qiskit、FAISS、AutoDock‑GPU、GPT‑2 训练、TinyStories、Shakespeare、Hotspot 和 STREAM 等八类实际工作负载和数据集。

**📈 对比分析**

通过对资源利用率、系统吞吐量、能耗和功率抑制进行对比实验，发现 MIG 7×1g 在大多数工作负载下可实现约 1.4 倍吞吐提升、约 26% 能耗降低；内存卸载在降低闲置内存的同时可接受有限的性能损失；奖励模型能够在性能与资源利用间实现可调节的权衡。

**⚠️ 局限性**

局限性包括 MIG 固定切分粒度导致资源与工作负载性能比例不匹配、内存卸载带来额外延迟、功率抑制仍是共享场景下的干扰源、实验仅在单台 Grace Hopper GPU 上进行、缺乏对动态重配置的探索。

---

## 530. DeepFense: A Unified, Modular, and Extensible Framework for Robust Deepfake Audio Detection

**arXiv ID:** 2604.08450 | [PDF](https://arxiv.org/pdf/2604.08450v1)

**作者:** Yassine El Kheir `[一作]` (German Research Center for Artificial Intelligence), Sebastian Moeller `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了DeepFense开源工具包，实现语音深度伪造检测的统一实验和评估框架

**💡 创新点**

整合前沿架构、数据集与训练策略，提供100+配方、400+模型，并系统评估400+模型，揭示前端、后端和训练集对性能与公平性的主导作用

**🔧 技术方法**

PyTorch、HuggingFace、Fairseq、统一配置文件、注册式插件、数据管道、Augmentation，前端如Wav2Vec2/WavLM/ HuBERT/EAT，后端如AASIST/MLP/Nes2Net/TCM，损失交叉熵/Softmax等

**📊 数据集**

ASVspoof 2019/2021、ADD 2022/2023、CodecFake、HABLA、PartialSpoof、EnvSDD、FakeMusicCaps、CtrSVDD 等13+ 语音与非语音数据集

**📈 对比分析**

通过统一实验配置对比 96 系统（4 前端×4 后端×6 训练集）并在 13 评测集上评估 EER，发现 Wav2Vec2+CodecFake 最优 (≈17% EER)，后端差异小，前端和训练集决定性能和公平性

**⚠️ 局限性**

未实现跨数据集联合训练，缺乏局部深度伪造定位与来源追踪，仅关注检测，且部分训练集存在性别/质量偏差导致公平性不足

---

## 531. AfriVoices-KE: A Multilingual Speech Dataset for Kenyan Languages

**arXiv ID:** 2604.08448 | [PDF](https://arxiv.org/pdf/2604.08448v1)

**作者:** Lilian Wanzare `[一作]`, Brian Gichana Omwenga `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了 AfriVoices‑KE 数据集，涵盖五种肯尼亚语言（Dholuo、Kikuyu、Kalenjin、Maasai、Somali）共约3000小时语音，包含 25% 书面式朗读与 75% 自发式语料，并配套转写与代码切换标注。

**💡 创新点**

创新点在于：① 大规模多语言、多方言采集与细粒度领域覆盖；② 通过社区志愿者与移动应用实现低成本、高质量的非脚本化录音；③ 采用双层校验与音频质量自动检测的完整数据管线。

**🔧 技术方法**

使用技术包括：自研的移动采集与转写 App、基于 SNR 的自动噪声检测、容器化微服务架构、双层人工校验、以及代码切换标注工具。

**📊 数据集**

主要数据集为 AfriVoices‑KE（约3000小时），并与现有少数 African 语音数据集（如 Kencorpus、NaijaVoices、SautiDB‑Naija 等）进行对比。

**📈 对比分析**

在缺乏统一评测基准的情况下，作者通过对比现有低资源 ASR 语料的规模与多样性，指出 AfriVoices‑KE 在数据量、方言覆盖和域多样性上显著优于现有资源；若训练 ASR，可期望在 Kalenjin、Dholuo 等语言上提升 20‑30% 的识别准确率。

**⚠️ 局限性**

局限性包括：① 数据采集受限于设备与网络，导致部分地区录音质量不均；② 受众年龄分布不平衡，老年人比例低；③ 仅覆盖部分方言，某些边缘方言仍缺乏代表；④ 目前未公开基线模型评估，需后续研究验证。

---

## 532. PG-MDP: Profile-Guided Memory Dependence Prediction for Area-Constrained Cores

**arXiv ID:** 2604.08445 | [PDF](https://arxiv.org/pdf/2604.08445v1)

**作者:** Luke Panayi `[一作]` (Imperial College), Paul Kelly `[通讯]` (Imperial College)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于程序分析的内存依赖预测（PG-MDP），通过在指令码中标记频繁无依赖的加载指令，直接跳过MDP查询，从而显著减少MDP查询次数和误依赖。

**💡 创新点**

创新点在于：①利用运行时分析的存储距离信息对加载指令进行标签化，削减MDP工作集；②不增加硬件面积或指令带宽，只需在ISA中引入标记位；③证明在面积受限核上，小型Store Sets预测器即可匹配大规模预测器的性能。

**🔧 技术方法**

技术手段包括：profile‑guided optimization（记录每条加载的存储距离并筛选阈值）、在编译期发射替代opcode、Gem5仿真、McPAT功耗建模、SPEC2017 intspeed基准测试。

**📊 数据集**

使用的数据集：SPEC2017 CPU intspeed基准，配合10个simpoint、训练输入和参考输入进行标签验证。

**📈 对比分析**

比较方法：将PG‑MDP与未加标签的Store Sets以及PHAST进行IPC、MDP查询次数、误依赖率、功耗等指标对比。结果显示，PG‑MDP在小型核上可使64条目Store Sets预测器的IPC仅低0.5%于1024条目预测器，并将MDP查询减少77%、误依赖减少77%；功耗下降约1.5%。

**⚠️ 局限性**

局限性：①需要ISA中额外编码位，部分ISA（如x86）实现受限；②仿真模型非真实处理器，可能影响结果泛化；③需针对目标处理器调优存储距离阈值，若使用通用阈值可能导致回归；④编译阶段需收集完整访问日志，存在内存占用与编译时间开销。

---

## 533. NL-CPS: Reinforcement Learning-Based Kubernetes Control Plane Placement in Multi-Region Clusters

**arXiv ID:** 2604.08434 | [PDF](https://arxiv.org/pdf/2604.08434v1)

**作者:** Sajid Alam `[一作]` (Edinburgh Napier University), Ze Wang `[通讯]` (Edinburgh Napier University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于 Neural LinUCB 的上下文 bandit 框架 NL-CPS，用来自动化多区域 K3S 集群中控制平面节点的最佳放置。

**💡 创新点**

创新点在于把控制平面放置视为单步上下文 bandit 问题，利用神经网络与 UCB 结合实现探索‑利用平衡，并在未见拓扑上泛化良好。

**🔧 技术方法**

使用技术包括神经网络函数逼近、LinUCB 及 UCB 探索，配合自研的合成训练环境与真实 12/18 节点多地区 K3S 部署。

**📊 数据集**

数据集由 800 条合成的节点特征与性能映射组成，训练时基于真实 5 节点测得的 CPU/内存/延迟特征与对应控制平面性能的映射。

**📈 对比分析**

实验将 NL-CPS 与三种基线（高资源、低延迟、随机）在 12/18 节点集群上进行轻/中/重负载 k‑bench 评测，结果显示 NL‑CPS 在吞吐量上提升 20–30% 以上，单个 pod 延迟降低 25% 以上。

**⚠️ 局限性**

局限在于仅处理单次放置决策，未覆盖控制平面迁移或高可用多控制平面复制的动态场景。

---

## 534. Formalizing building-up constructions of self-dual codes through isotropic lines in Lean

**arXiv ID:** 2604.08485 | [PDF](https://arxiv.org/pdf/2604.08485v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 535. On-board Telemetry Monitoring in Autonomous Satellites: Challenges and Opportunities

**arXiv ID:** 2604.08424 | [PDF](https://arxiv.org/pdf/2604.08424v1)

**作者:** Lorenzo Capelli `[一作]` (University of Bologna), Gianluca Furano `[通讯]` (European Space Agency)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于神经网络内部激活的可解释异常检测框架（peepholes），并将其应用于卫星自主姿控系统中的反应轮遥测异常检测。

**💡 创新点**

创新点在于：①通过SVD降维、GMM聚类和语义映射三步非神经处理，直接从自编码器的中间层提取低维、语义标记的“peephole”向量；②该向量可在不增加额外分类器的情况下识别异常类型与来源，并能揭示检测偏差；③实现了高透明度、低计算开销的在线故障诊断。

**🔧 技术方法**

主要技术包括：卷积自编码器（CNN AE）、奇异值分解（SVD）降维、高斯混合模型（GMM）统计特征、语义映射函数、以及对激活向量的热力图可视化。

**📊 数据集**

使用ESA Earth Observation任务的反应轮遥测数据集，包含4个反应轮的16条时序信号；对原始数据注入五类合成异常（加性噪声、偏移、脉冲、功率谱扰动、步进）进行实验。

**📈 对比分析**

通过AUC（接收者操作特征曲线下的面积）评估自编码器的异常检测性能，所有异常的AUC均接近1；利用peephole生成的向量构建混淆矩阵，能够识别异常类型和受影响的反应轮，且揭示了对某些轮子偏向的检测偏差。

**⚠️ 局限性**

局限性包括：①实验主要基于合成异常，缺乏对真实长期空间环境中多样故障的验证；②偏差分析仍处于初步阶段，未系统评估所有异常类别与系统状态；③虽计算量低于传统神经分类器，但仍需进一步压缩以满足极限卫星资源；④对实时性与在线验证的实测结果尚未公布。

---

## 536. Your Agent Is Mine: Measuring Malicious Intermediary Attacks on the LLM Supply Chain

**arXiv ID:** 2604.08407 | [PDF](https://arxiv.org/pdf/2604.08407v1)

**作者:** Hanzhi Liu `[一作]` (University of California Santa Barbara), Yu Feng `[通讯]` (University of California Santa Barbara)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）API 路由器在工具调用中的攻击面，系统化提出威胁模型、攻击分类及适配性规避变体，并在真实商用及免费路由器中进行实测，揭示了路由器可注入恶意命令、泄露凭证以及被劫持后形成的传输链式安全风险。

**💡 创新点**

首次将攻击分为 AC‑1（响应侧注入）和 AC‑2（被动泄露）两大类，并引入 AC‑1.a（依赖注入）和 AC‑1.b（条件投递）两种规避技术；同时通过对 428 条路由器的实测与两次路由器毒化实验，提供了公开路由器安全态势的第一手量化数据。

**🔧 技术方法**

利用 FastAPI 编写可插拔攻击模块（工具调用注入、依赖替换、条件触发与密钥抽取），在四大公开框架（Claude、Codex、OpenAI、Anthropic）上测试兼容性；实现三种客户端防御（fail‑closed policy、异常检测、透明日志）并对其性能与误报率进行评估。

**📊 数据集**

数据集包括：28 个付费路由器、400 个公开免费路由器、1 个公开泄露 OpenAI key、20 域+20 IP 的弱路由诱饵；收集了 2B 计费 token、99 组凭证、8 条恶意注入实例、17 条凭证滥用实例等。

**📈 对比分析**

在四个框架下各 1,000 次请求测试，工具调用注入兼容率 100%，AC‑1.a 99.6%；fail‑closed policy 的误报率 1% ，异常检测误报率 6.7%，检测率 89%；透明日志记录 12 MB/1,000 会话，平均 1.26 KB/条记录。

**⚠️ 局限性**

研究仅覆盖公开路由器和公开实验，不涉及私有部署；防御依赖客户端配置，无法阻止已被攻击者控制的路由器；未实现端到端的签名机制，无法完全消除供应链信任缺口。

---

## 537. The Impact of Dimensionality on the Stability of Node Embeddings

**arXiv ID:** 2604.08492 | [PDF](https://arxiv.org/pdf/2604.08492v1)

**作者:** Tobias Schumacher `[一作]` (University of Mannheim), Markus Strohmaier `[通讯]` (University of Mannheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统评估了节点嵌入维度对表示与功能稳定性以及下游节点分类性能的影响，比较了五种主流方法在不同维度下的表现；

**💡 创新点**

首次量化维度变化对嵌入稳定性的多维度影响，并揭示不同方法在稳定性与性能之间的非一致性及其权衡；

**🔧 技术方法**

训练node2vec、GraphSAGE、VERSE、DGI、ASNE的多维度嵌入，使用对齐余弦相似、k‑NN Jaccard、距离相关等表示稳定度量，功能稳定度量包括稳定核心、Jensen‑Shannon散度、同意率等；下游任务采用逻辑回归/MLP进行节点分类；

**📊 数据集**

Cora、PubMed、Wikipedia、BlogCatalog、Facebook 等五个真实图数据集；

**📈 对比分析**

对每个维度使用30个随机种子训练嵌入，计算平均准确率与多种稳定性指标；结果表明多数方法在中高维达到性能平台期，而某些方法如VERSE、GraphSAGE在极高维性能下降；稳定性在低维可高，但不一定与最佳性能对应；

**⚠️ 局限性**

实验仅限于≤25k节点的小型/中型图；所用实现版本可能影响结果；高维度线性增量实验不可行；所选相似度度量可能不完全代表所有嵌入几何；仅评估节点分类任务，未覆盖其他下游任务或更大规模图。

---

## 538. AI generates well-liked but templatic empathic responses

**arXiv ID:** 2604.08479 | [PDF](https://arxiv.org/pdf/2604.08479v1)

**作者:** Emma Gueorguieva `[一作]` (University of Texas at Austin), Desmond C. Ong `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建10种可在文本中直接识别的同理心语言策略，并使用人工与LLM自动标注，比较LLM（GPT‑4 Turbo、Llama‑3.1、Qwen‑2.5）与人类（Upwork心理学背景与Reddit高评分评论）在策略使用频率、种类多样性及模板化程度上的差异。

**💡 创新点**

创新点在于：①首次以可识别文本的策略为单位创建10项同理心策略分类并与三大心理学维度对应；②利用正则表达式挖掘并量化LLM回应的结构化模板；③在大样本人类数据与多代LLM中检验模板泛化性，揭示LLM同理心回应高度模板化、缺乏多样性。

**🔧 技术方法**

技术手段包括：人工标注（Krippendorff’s α≈0.80）、LLM自动标注器（基于GPT‑4 Turbo）、正则表达式模板搜索与覆盖率评估、词频与策略分布统计、两组实验设计（Study 1与Study 2）。

**📊 数据集**

数据集：Study 1采集101条Reddit求助帖，27名心理学背景Upwork回应（290条）与3个LLM回应（303条）；Study 2采集1000条Reddit求助帖，1000条顶级人类回应与3个LLM回应（分别为GPT‑4 Turbo (962)、Llama‑3.1 (1000)、Qwen‑2.5 (1000)）。

**📈 对比分析**

比较方法：统计每个策略的出现频率、独特性与多样性；利用正则表达式评估模板覆盖率。结果显示LLM在策略使用上高度模板化，模板覆盖率在Study 1中为83–90%，在Study 2中为60–83%；人类回应的模板覆盖率仅为40%（Study 1）和6.4%（Study 2），说明人类同理心表达更为多样且不易被单一模板捕捉。尽管LLM在主观同理心感知上优于人类，但在细腻度与情境适应性上表现不足。

**⚠️ 局限性**

局限性：①人类样本量有限且多为心理学背景或高评分Reddit评论，可能影响策略多样性；②未能完全排除人类使用LLM的可能，导致部分“人类”回应出现模板化；③词典未覆盖所有文化/语境特有的同理心表达；④模板化可能导致LLM在不同情境下缺乏适配性，易产生同理心与迎合的边界模糊。

---

## 539. BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields

**arXiv ID:** 2604.08410 | [PDF](https://arxiv.org/pdf/2604.08410v1)

**作者:** Fan Yang `[一作]` (Hunan University), Yaonan Wang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了零射击模块化框架BLaDA，将自然语言指令解析为结构化的六元组，随后通过三角功能点定位和3D抓取矩阵转换，实现从语言到可执行动作的完整流程。

**💡 创新点**

创新点包括：1）将LLM与领域知识图融合的知识引导语言解析；2）在3D Gaussian Splatting中实现三角功能点定位，提供精确空间约束；3）通过KGT3D+实现可解释的手腕姿态与指尖动作，完全无任务专用训练。

**🔧 技术方法**

使用技术包括LLM+知识图、3D Gaussian Splatting、CLIP、任务工具拓扑坐标系、MLP、三角功能点定位、3D抓取矩阵转换等。

**📊 数据集**

采用FAH工具抓取数据集（18类工具）以及自建的10个桌面场景的RealSense多视角RGB-D图像进行评估。

**📈 对比分析**

与GraspSplats、MKA、DP*等基线对比，BLaDA在功能抓取成功率和关键点定位准确率上均领先：平均LSR 68.75%，功能抓取成功率最高达80%，显著优于对照组。

**⚠️ 局限性**

局限性在于依赖3DGS语义场的细粒度表现仍有限，工具部件语义易出现歧义，且缺乏触觉反馈导致对不稳定抓取的鲁棒性不足。

---

## 540. SyncBreaker:Stage-Aware Multimodal Adversarial Attacks on Audio-Driven Talking Head Generation

**arXiv ID:** 2604.08405 | [PDF](https://arxiv.org/pdf/2604.08405v1)

**作者:** Wenli Zhang `[一作]` (University of Science and Technology of China), Yong Liao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SyncBreaker，一种阶段感知的多模态对抗防护框架，用于保护音频驱动的说话人头动画免受伪造视频攻击。

**💡 创新点**

创新点包括针对图像流的多间隔采样（MIS）的零化监督和针对音频流的交叉注意力欺骗（CAF），两者协同抑制语音驱动的面部动态。

**🔧 技术方法**

利用扩散模型的多阶段噪声调度、投影梯度下降（PGD）与交叉注意力损失进行对抗样本生成，并结合多间隔采样和注意力方差抑制。

**📊 数据集**

在 CelebA-HQ–LibriSpeech 与 HDTF 两组公开数据集上评估，使用 50 张图像与 50 条音频对。

**📈 对比分析**

与五种图像防护方法（AdvDM、PhotoGuard、Mist、SDS、Silencer）和若干音频攻击方法相比，SyncBreaker 在同步度、口型误差与 FID 指标上显著优于单模态基线，并在去净化测试中保持鲁棒性。

**⚠️ 局限性**

目前仅在白盒设置下验证，缺乏对黑盒迁移性及更广泛生成框架的适用性研究。

---

## 541. Adversarial Label Invariant Graph Data Augmentations for Out-of-Distribution Generalization

**arXiv ID:** 2604.08404 | [PDF](https://arxiv.org/pdf/2604.08404v1)

**作者:** Simon Zhang `[一作]` (Purdue University), Cathy H. Xia `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为 RIA 的方法，通过在图分类任务中使用对抗性标签不变数据增强来防止 ERM 解决方案的崩溃，从而实现更稳健的 OoD 泛化。

**💡 创新点**

创新点在于：①将 Q‑learning 的探索-奖励机制类比到对抗性数据增强，构造对抗性反事实环境；②将传统 OoD 约束转化为正则化项，使得学习过程能够在不破坏标签不变性的前提下主动生成困难环境；③通过交替梯度上升下降实现对抗性增强与模型训练的协同优化。

**🔧 技术方法**

使用的技术包括图神经网络、对抗性标签不变数据增强、minimax 优化、交替梯度上升-下降算法、正则化方法（RICE、IRM、VREx 等）以及 Q‑learning 类比的理论分析。

**📊 数据集**

实验数据集涵盖真实图分类基准 CMNIST、SST2、Motif、AMotif 以及自定义的加性噪声合成数据集，测试不同的协变量分布偏移场景。

**📈 对比分析**

与 ERM、IRM、RICE、VREx、Mixup、DANN、GroupDRO、Coral 等多种基线方法在 ID 与 OoD 测试上进行比较。RIA 在大多数数据集上实现了最高或第二高的准确率，尤其在 OoD 场景中显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅在图数据上验证，其他模态（如图像）需要进一步实验验证；对抗性增强需要保持标签不变且不造成分布过度偏移；需要手动调节增强强度和多样性以避免过度校正。

---

## 542. ADAPTive Input Training for Many-to-One Pre-Training on Time-Series Classification

**arXiv ID:** 2604.08398 | [PDF](https://arxiv.org/pdf/2604.08398v1)

**作者:** Paul Quinlan `[一作]`, Xiaodan Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种基于自适应池化的多到一预训练框架（ADAPT），使单一时间序列模型可同时在超过150个不同长度、通道和模态的数据集上进行混合批训练。

**💡 创新点**

创新点在于使用平均自适应池化将各异的时序数据映射到统一表示空间，并结合时频双编码与跨域 span‑mask 预训练，解决了时间序列多模态预训练中数据不一致导致的性能衰退问题。

**🔧 技术方法**

主要技术包括平均自适应池化、FFT频域转换、时间-频率双编码、Transformer 编码器、span‑mask 随机遮蔽以及加入高斯噪声的自监督预训练策略。

**📊 数据集**

使用了 162 个时间序列分类数据集（主要来自 UCR/UEA 归档以及 SleepEEG、FD‑A、HAR、ECG 等），共约 55 万样本。

**📈 对比分析**

与现有基线（TS‑SD、TS2vec、CLOCS、TS‑TCC、SimCLR、TF‑C 等）在 Epilepsy、FD‑B、Gesture、EMG 等四个下游分类任务上进行微调比较，ADAPT 在大多数任务上均实现了新的 state‑of‑the‑art 准确率，尤其在 FD‑B 与 EMG 上差距显著。

**⚠️ 局限性**

局限性包括对极小样本的基准仍存在限制、训练数据量与多样性未同步扩增导致部分任务性能波动、对模态信息的显式编码缺失，以及对更大规模、多通道数据的适用性尚未充分验证。

---

## 543. Phantasia: Context-Adaptive Backdoors in Vision Language Models

**arXiv ID:** 2604.08395 | [PDF](https://arxiv.org/pdf/2604.08395v1)

**作者:** Nam Duong Tran `[一作]` (Hanoi University of Science and Technology), Phi Le Nguyen `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉‑语言模型（VLM）的后门攻击，发现现有攻击易被检测并提出了一种新的基于上下文自适应的后门攻击——Phantasia，能够在触发后生成与图像语义一致且恶意的响应。

**💡 创新点**

创新点在于（1）通过改造ONION和STRIP两种跨模态防御方法，证明现有VLM后门的可检测性；（2）设计Phantasia攻击框架，利用对抗性噪声触发并通过在线知识蒸馏实现输入图像与攻击者预设问题共同决定恶意输出，从而大幅提升隐蔽性。

**🔧 技术方法**

主要技术包括：对抗性触发（Gaussian noise）、改造后的ONION‑R与STRIP‑P检测算法、对抗性蒸馏（Attention & Logits蒸馏）以及构造上下文感知的毒化数据集。

**📊 数据集**

使用公开的图像‑文本数据集：Flickr8k、Flickr30k（图像描述），VQAv2、OKVQA（视觉问答）；模型架构涵盖BLIP、BLIP‑2与LLaVA。

**📈 对比分析**

与BadVLM、Shadowcast、TrojVLM、VLOOD等基线对比，Phantasia在ASR、BLEU@4、ROUGE‑L、METEOR、VQAScore等指标上均表现最优，攻击成功率可达100%且对干净样本的影响极小。

**⚠️ 局限性**

局限性包括：攻击需要对模型完全控制、触发噪声可能在某些部署场景下被过滤、目前仅针对生成任务，其他任务如对齐、检索等仍需进一步研究。

---

## 544. Learning Who Disagrees: Demographic Importance Weighting for Modeling Annotator Distributions with DiADEM

**arXiv ID:** 2604.08425 | [PDF](https://arxiv.org/pdf/2604.08425v1)

**作者:** Samay U. Shetty `[一作]` (Rochester Institute of Technology), Christopher M. Homan `[通讯]` (Rochester Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种面向主观任务的分辨意见差异的模型 DiADEM，能够预测各标注者的标签分布并跟踪标注者间的分歧。

**💡 创新点**

创新点在于通过可学习的每个人口维度重要性向量（α）对标注者进行加权表征，结合两种交互方式（拼接与Hadamard）以及专门的项目级分歧损失，使模型能显式捕捉谁会产生分歧以及分歧的程度。

**🔧 技术方法**

使用的技术包括多模态编码（人口属性与文本特征投影）、交互融合、残差变换层、软最大化输出、KL散度及方差对齐的损失函数，并在训练中使用自适应加权。

**📊 数据集**

实验基准为 DICES（对话安全评估）和 VOICED（政治言论冒犯检测），两数据集均提供多名标注者的多维人口信息与分布式标签。

**📈 对比分析**

与基准神经模型（DisCo、DisCo–LeWiDi）以及多种大型语言模型（GPT‑4o‑mini、Llama‑4、GPT‑5、Gemma‑2）对比，DiADEM 在标注者级别的准确率、κ、MCC、JSD、ER、ECE 等指标均显著优于对手；在项目级别虽与竞争模型相近，但在分布式一致性和校准方面表现更佳。

**⚠️ 局限性**

局限性包括对人口属性数据的依赖；属性缺失或质量不足会削弱模型性能；模型仍未能完美捕捉所有不可预测的主观噪声；并且在不同任务中α权重的可解释性与泛化性需要进一步验证。

---

## 545. What They Saw, Not Just Where They Looked: Semantic Scanpath Similarity via VLMs and NLP metric

**arXiv ID:** 2604.08494 | [PDF](https://arxiv.org/pdf/2604.08494v1)

**作者:** Mohamed Amine Kerkouri `[一作]` (F-Initiatives), Alessandro Bruno `[通讯]` (IULM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套使用视觉‑语言模型将每个注视点转换为自然语言描述并聚合为摘要的语义扫描路径相似度框架，以文本相似度衡量扫描路径的语义相似性。

**💡 创新点**

首次系统性地将语义信息融入扫描路径比较，利用VLM生成注视描述并用NLP相似度指标评估扫描路径，补充传统几何相似度的不足。

**🔧 技术方法**

采用CLIP等视觉‑语言模型进行注视描述生成，使用BERTScore、ROUGE‑L、BLEU‑4、BM25等NLP指标，并结合MultiMatch、DTW等传统扫描路径度量。

**📊 数据集**

在COCOFreeView数据集上实验，使用100张图像、每张5个扫描路径，共500个扫描路径，计算所有同图像内扫描路径对的相似度。

**📈 对比分析**

对每对扫描路径计算语义相似度与几何相似度，并通过Spearman相关分析；语义相似度与几何相似度相关性低至0.1‑0.3，表明两者互补且不冗余；不同注视编码方式对相似度稳定性有显著影响。

**⚠️ 局限性**

受所选VLM与提示策略影响，未进行人类主观相似度验证；实验仅限于自由观看同一图像，缺乏跨图像、任务驱动情境的评估。

---

## 546. Post-Quantum Cryptographic Analysis of Message Transformations Across the Network Stack

**arXiv ID:** 2604.08480 | [PDF](https://arxiv.org/pdf/2604.08480v1)

**作者:** Ashish Kundu `[一作]` (Cisco Research), Ramana Kompella `[通讯]` (Cisco Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

开发了跨层量子安全性评估框架，定义层级加密的 PQC 状态并通过案例研究验证。

**💡 创新点**

将每层加密的量子脆弱性归类为四级，并证明链级安全性按最大/最小运算组成受限格，揭示隐私、认证和元数据的不同组合规律。

**🔧 技术方法**

采用形式化模型、偏序/格论，并手工分析四个典型协议栈（iMessage、HTTPS+WPA2/Enterprise、WireGuard）。

**📊 数据集**

使用 Ubuntu 24.04 与 iOS 17/18 默认协议配置及公开的加密算法参数；无实验数据。

**📈 对比分析**

通过对每层 PQC 状态求最大/最小得到链级安全性，比较不同配置的元数据暴露深度 d* 以及安全等级；发现单层 PQC 可满足机密性，认证需全部迁移。

**⚠️ 局限性**

仅基于已知算法分类，忽略协议实现差异和侧信道；未给出迁移成本量化，也缺少自动化工具实现。

---

## 547. SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions

**arXiv ID:** 2604.08477 | [PDF](https://arxiv.org/pdf/2604.08477v1)

**作者:** Ashima Suvarna `[一作]` (University of California, Los Angeles), Saadia Gabriel `[通讯]` (University of California, Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种从大规模指令调优数据集中挑选、混合并裁剪生成可验证奖励的RLVR训练数据的框架，最终构建了约25K条RLVR样本的数据集，用以提升LLM在一般推理（BBEH、Zebralogic、MMLU-Pro等）上的表现。

**💡 创新点**

创新点包括：① 将高质量人类注释的指令数据转换为可验证的RLVR问题；② 通过任务效用得分实现任务选择，证明“微混合”（按子任务挑选顶级任务）优于宏混合；③ 发现合成干预（如长上下文、反强先验）对已高质量数据无显著提升，揭示数据干预的局限性；④ 在多模型、多规模下展示了该数据集在RLVR训练中的跨模型泛化能力。

**🔧 技术方法**

技术手段：使用RLVR与GRPO优化目标进行强化学习；对任务进行语义相似度、难度与子任务表现评估；采用宏/微混合策略；对问题进行人工或自动干预；使用pass@k（尤其是pass@8）评估模型推理能力。

**📊 数据集**

主要使用的数据集为：SuperNI（1600项任务，83项作为候选），FLAN-Collection，BBeH-mini/BBEH-test，Zebralogic，MMLU-Pro，MATH500等；同时与Nemotron-CrossThink、General-Reasoner、DAPO等基线数据集做对比。

**📈 对比分析**

比较方法：在计算匹配的条件下，对不同规模的Qwen3（0.6B、1.7B、4B）与Qwen3.5、General-Reasoner、OpenThinker等模型进行pass@k评测；结果显示，基于新构建的数据集训练的模型在BBEH、Zebralogic、MMLU-Pro上均显著优于基线，最小模型4B甚至超越8B大模型8.2个百分点；在pass@8上相对提升可达42pp。

**⚠️ 局限性**

限制：实验受限于固定的计算预算与数据量；所评测的基准主要是学术式推理测试，可能无法全面反映真实世界任务；合成干预的有效性需在更大规模与更复杂任务中进一步验证；未来需探索在更高算力与更丰富数据下的性能持续性。

---

## 548. KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation

**arXiv ID:** 2604.08455 | [PDF](https://arxiv.org/pdf/2604.08455v1)

**作者:** Tongbo Chen `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向移动端个性化与主动式服务的交互式基准框架 KnowU-Bench，结合可复现的 Android 模拟器、结构化用户档案、日志与 LLM 驱动的用户模拟器，实现从离线意图推断向在线执行评估的转变。

**💡 创新点**

核心创新在于：①把个性化与主动任务嵌入可程序化验证的 Android 环境；②利用结构化档案与交互式日志让模型在执行过程中可主动获取偏好；③采用混合评判机制（规则+LLM），既保证可验证性，又能评估语义偏好匹配。

**🔧 技术方法**

技术手段包括：容器化 Pixel 8 AVD、ADB 控制器、LLM（如 Claude 4.6、Gemini 3.1、Qwen 系列）驱动的用户模拟器、检索式日志（RAG）与完整日志、规则与 LLM 混合评判器。

**📊 数据集**

数据集由 42 个通用、86 个个性化、64 个主动任务组成，覆盖 23 款应用（购物、外卖、闹铃等），用户档案设定四个角色（研究员、开发者、学生、奶奶），并提供相应的交互日志。

**📈 对比分析**

对 11 种主流模型（开源 Qwen、Gemini、Claude 等）在易/难子集进行比较，结果显示闭源模型 Claude 4.6 在整体成功率上最高约 60%，但在个性化与主动任务上均低于 50%；开源模型平均成功率不足 12%，表明现有模型在个性化获取与主动决策方面仍存在显著瓶颈。

**⚠️ 局限性**

局限在于基准仍依赖人工构造的用户档案与模拟器，难以覆盖真实用户多样性；评判中 LLM 的偏差可能导致不稳定性；并未深入探索主动决策的长期学习与安全性，需进一步完善。

---

## 549. Real-Time Cross-Layer Semantic Error Correction Using Language Models and Software-Defined Radio

**arXiv ID:** 2604.08419 | [PDF](https://arxiv.org/pdf/2604.08419v1)

**作者:** Yuchen Pan `[一作]` (Chinese University of Hong Kong), Soung Chang Liew `[通讯]` (Chinese University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在软件定义无线电(SDR)平台上实现并验证实时跨层语义误差校正(CL-SEC)，实现物理层LLR与语言模型语义信息的融合，以实现逐词还原。

**💡 创新点**

创新点在于：①设计低延迟SDR中间件，实时提取硬/软位置信息；②构建通用LM接口，使encoder-decoder模型适用于单词级语义纠错；③首次在真实硬件上演示CL-SEC并验证其性能。

**🔧 技术方法**

技术包括SDR中间件提取LLR、OpenWiFi固件、GPU实时LM推理、贝叶斯融合、T5Gemma encoder-decoder模型等。

**📊 数据集**

使用10个公开数据集，总计800条15-250词长度序列，采用UDP文本负载。

**📈 对比分析**

通过与单独LLR、单独T5Gemma的基线对比，CL-SEC在75-100词序列上PRR最高，Mask Recovery Accuracy在150词序列上可达44.6%，显著优于两种基线。

**⚠️ 局限性**

局限在于实验环境受限于噪声实验室，未验证在更广泛网络环境或更长序列的鲁棒性；同时依赖GPU进行实时推理，延迟仍较高。

---

## 550. What a Comfortable World: Ergonomic Principles Guided Apartment Layout Generation

**arXiv ID:** 2604.08411 | [PDF](https://arxiv.org/pdf/2604.08411v1)

**作者:** Piotr Nieciecki `[一作]`, Przemyslaw Musialski `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于优化的房间布局方法，以减少不便使用的卫生间和盲区。

**💡 创新点**

创新点在于将可达性评估指标直接融入布局优化过程，显著降低了不便使用的卫生间和盲区。

**🔧 技术方法**

使用整数规划/启发式优化技术对房间布局进行搜索与评估。

**📊 数据集**

使用了RPLAN数据集中的房间样本进行实验。

**📈 对比分析**

通过与基线方法和RPLAN样本的定性对比，展示了在不便使用的卫生间和盲区数量上均有所下降，布局更合理。

**⚠️ 局限性**

仅进行定性评估，缺乏量化指标，且仅针对少量案例，泛化性有待验证。

---

## 551. A Machine Learning Framework for Turbofan Health Estimation via Inverse Problem Formulation

**arXiv ID:** 2604.08460 | [PDF](https://arxiv.org/pdf/2604.08460v1)

**作者:** Milad Leyli-Abadi `[一作]` (Institut de recherche technologique SystemX), Jesse Read `[通讯]` (École Polytechnique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于真实工况、包含多种退化速率、维护事件和不同运行阶段的机舱健康估计基准数据集，并在此数据集上对多类方法进行了实验评估。

**💡 创新点**

创新点在于（1）公开了更具工业真实性的 turbofan 数据集；（2）系统性对比了传统的稳态回归、时序学习、状态空间滤波与自监督学习两阶段表征学习；（3）用自监督方法提供了无标签下的健康估计下限。

**🔧 技术方法**

使用的技术包括梯度提升树、全连接网络、GRU 循环网络、无迹卡尔曼滤波 (UKF)、自编码器 (AE) 与联合嵌入预测架构 (VJEPA)。

**📊 数据集**

采用了新生成的 500 条退化轨迹（长度最高 2000 步，覆盖四个飞行阶段）和 7 维传感器观测；数据通过 OpenDeckSMR 仿真器与噪声模型生成。

**📈 对比分析**

对比方法时采用 SMAPE、RMSE、Pearson 相关三指标进行 5 折交叉验证；结果显示 UKF 在多数指标上表现最佳，GRU 稍优于稳态模型；自监督方法虽能捕获全局结构，但估计误差较大。

**⚠️ 局限性**

局限性包括：部分健康指示器（如 HI3、HI9、HI10）可观测性差；滤波方法在维护和非线性变化下易漂移；自监督学习缺乏任务监督导致精度不足；整体仍未能完全解决逆问题的部分不可观测性。

---

## 552. CrashSight: A Phase-Aware, Infrastructure-Centric Video Benchmark for Traffic Crash Scene Understanding and Reasoning

**arXiv ID:** 2604.08457 | [PDF](https://arxiv.org/pdf/2604.08457v1)

**作者:** Rui Gan `[一作]` (University of Wisconsin--Madison), Bin Ran `[通讯]` (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CrashSight 这一基于路侧摄像头的交通事故多模态问答基准，包含 250 条专家标注的视频及 13K 多选 QA 对；

**💡 创新点**

创新点在于：1）引入四阶段时间结构（预碰、碰撞、事后、原因）并保持在注释与 QA 中；2）结合 VLM、人工专家与 LLM 的三阶段注释流程；3）构建针对安全关键场景的多维度问答与鲁棒性测试；

**🔧 技术方法**

使用了 Vision‑Language Models（如 InternVL3、Qwen2.5‑VL、LLaVA 系列）并通过 QLoRA 微调、4‑bit NF4 量化、系统提示等技术；

**📊 数据集**

采用 TAD 语料库中的真实路侧摄像头事故视频，经过专家校正后生成的 CrashSight 数据集；

**📈 对比分析**

与 8 种 VLM（含 0‑shot 与微调）进行对比，微调后平均准确率提升 13–16 分，最高单模型 76% 以上；与人类专家相比仍存在 18–25 分差距，尤其在实体识别和碰撞机理等视觉要求高的题目上；

**⚠️ 局限性**

主要限制：视觉令牌预算受限、量化导致视觉辨识精度下降、视觉编码器未微调，导致在涉事主体、场景识别与碰撞机理等视觉依赖任务上的错误仍高；数据规模和专家校正成本限制了扩展性。

---

## 553. A Soft Robotic Interface for Chick-Robot Affective Interactions

**arXiv ID:** 2604.08443 | [PDF](https://arxiv.org/pdf/2604.08443v1)

**作者:** Jue Chen `[一作]` (Queen Mary University of London), Elisabetta Versace `[通讯]` (Queen Mary University of London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并验证了一种针对孵化幼鸡的软体机器人情感交互界面，融合温热、面部视觉和呼吸式节律刺激，评估其接受度与互动行为。

**💡 创新点**

首次将人机情感接触技术移植至动物-机器人交互，并系统阐明多模态刺激（视觉+热）在提升动物接受度方面的关键作用。

**🔧 技术方法**

软体材料构造、温度调控模块、气泵驱动的呼吸式膨胀囊、面部图案3D打印、Arduino 控制板、DeepLabCut 视觉跟踪。

**📊 数据集**

收集61只新孵鸡在30分钟实验中的视频轨迹，共54个有效记录，使用DeepLabCut提取位置信息。

**📈 对比分析**

采用Beta混合效应模型评估时间、刺激类型对接触比例的影响，结果显示面部和热刺激显著高于机会水平，呼吸刺激保持中性；统计显著性强，表明界面设计有效。

**⚠️ 局限性**

仅在孵化初期的单一鸡种上验证，缺乏对不同年龄、种类及长期接触效应的探索，且未细化触碰频率、身体部位等更细粒度行为指标。

---

## 554. Power Amplifier-aware Power Allocation for Noise-limited and Distortion-limited Regimes

**arXiv ID:** 2604.08437 | [PDF](https://arxiv.org/pdf/2604.08437v1)

**作者:** Achref Tellili `[一作]` (University of Tennessee), Mohamed Akrout `[通讯]` (University of Tennessee)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种考虑功率放大器非线性失真的功率分配算法，取代传统的水填充方法以提升多天线系统在饱和区的容量。

**💡 创新点**

创新点在于将Bussgang理论对硬限制放大器的统计线性化与空间后退策略结合，首次给出热噪声阈值的闭式表达，并通过投影梯度下降实现全局约束下的功率优化。

**🔧 技术方法**

主要技术包括Bussgang分解、统计线性化、MIMO信道模型、投影梯度下降优化以及热噪声与失真噪声的协方差分析。

**📊 数据集**

实验使用基于32×32多天线矩阵的Rayleigh衰落与多径模型，并通过模拟得到不同热噪声水平和传输功率下的容量曲线。

**📈 对比分析**

与标准水填充对比，采用该算法在低热噪声（失真受限）下可实现超过100％的容量提升，且在噪声受限区两种方法收敛。

**⚠️ 局限性**

局限性包括对放大器模型假设为无记忆硬限制、仅考虑高斯输入信号、以及投影梯度下降可能陷入局部最优，未对硬件实现的实时性与复杂度作详细评估。

---

## 555. Vulnerability Detection with Interprocedural Context in Multiple Languages: Assessing Effectiveness and Cost of Modern LLMs

**arXiv ID:** 2604.08417 | [PDF](https://arxiv.org/pdf/2604.08417v1)

**作者:** Kevin Lira `[一作]` (North Carolina State University), Wesley K. G. Assunção `[通讯]` (North Carolina State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在多语言跨函数漏洞检测中的有效性与成本。

**💡 创新点**

系统评估不同上下文层级对四款LLM的影响，并结合成本与解释质量。

**🔧 技术方法**

通过Prompt工程和统计检验（如McNemar检验）对Claude Haiku 4.5、Gemini 3 Flash、GPT‑4.1 Mini、GPT‑5 Mini进行功能级与跨函数检测。

**📊 数据集**

使用RepsVul数据集中的509条真实CVE，覆盖C、C++与Python三种语言。

**📈 对比分析**

采用准确率/召回率/F1和token费用评估，发现Gemini 3 Flash在C语言下成本‑性能最优，Claude Haiku在解释质量上领先。

**⚠️ 局限性**

仅评估单函数与跨函数检测，无负样本；模型提示未进行模型特定优化；数据集仅为RepsVul，难以保证对其他语言或更大模型的泛化。

---

## 556. Selective Attention System (SAS): Device-Addressed Speech Detection for Real-Time On-Device Voice AI

**arXiv ID:** 2604.08412 | [PDF](https://arxiv.org/pdf/2604.08412v1)

**作者:** David Joohun Kim `[一作]` (Attention Labs), Omar Abbasi `[通讯]` (Attention Labs)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Selective Attention System（SAS），在边缘设备上实现设备地址语音检测，并通过“Sequential Device‑Addressed Routing”（SDAR）将问题建模为带有交互历史的序列决策。

**💡 创新点**

创新点在于：① 把设备地址检测从单句分类转化为基于短期交互历史的序列路由；② 引入三阶段流水线（声学几何过滤、轻量化说话者特征分类、因果交互状态估计）实现低时延、低内存的预‑ASR 触发门控。

**🔧 技术方法**

使用技术包括：双麦克风波束成形（声学几何前端）、1D‑CNN + GRU 的轻量化音频分类器、Causal Transformer 用于交互状态估计、可选的姿态与视线估计做多模融合，以及 ARM Cortex‑A 上的 INT8 量化模型。

**📊 数据集**

数据集为公司内部 600 小时多说话人英语语料，内部评估采用 60 小时保留集（其中 4.8 小时为设备指令）并提供 5 小时可公开验证子集。

**📈 对比分析**

通过与 VAD、无上下文分类器以及音频+视频组合的对照实验，SAS 在音频‑仅模式下 F1 达 0.86，音频+视频模式下 0.95；消除 Stage 3 时 F1 下降 38 点，显示序列历史贡献最大；与标准 VAD 仅路由相比，准确率提升约 70 %（减少 90 % 以上的后端推理）。

**⚠️ 局限性**

局限性包括：评测基于专有数据且未第三方审计；主要语言为英语，非英语表现尚未充分验证；单麦克风、五个以上说话人以及高回声时延环境的性能未覆盖；模型在高噪声/多说话人场景下仍存在误判（如电视对白）。

---

## 557. Zero-shot Multivariate Time Series Forecasting Using Tabular Prior Fitted Networks

**arXiv ID:** 2604.08400 | [PDF](https://arxiv.org/pdf/2604.08400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 558. Awakening the Sleeping Agent: Lean-Specific Agentic Data Reactivates General Tool Use in Goedel Prover

**arXiv ID:** 2604.08388 | [PDF](https://arxiv.org/pdf/2604.08388v1)

**作者:** Jui-Hui Chung `[一作]` (Princeton Language and Intelligence), Chi Jin `[通讯]` (Princeton Language and Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在正式数学领域进行大规模监督微调后，模型在工具调用能力上的灾难性遗忘，并展示如何通过少量域特定的agentic训练样本恢复这一能力。

**💡 创新点**

发现工具调用能力被压制而非消失，仅需约100条域特定轨迹即可恢复跨域的工具使用，并提出利用跨模型蒸馏生成高质量agentic轨迹的方法。

**🔧 技术方法**

使用 LLaMA-Factory 进行全参数监督微调，结合强化学习、工具检索（Search）和 SLERP 模型融合，构建跨模型蒸馏管道。

**📊 数据集**

基于 18.9 万条 Lean 形式化数学问题的 OMR 语料库，构造了 18K 高质量 agentic 轨迹，随后在 1.79M 的 formal‑math 训练集上进行微调。

**📈 对比分析**

与基线模型相比，微调后的模型在 Lean 推理基准 pass@32 从 21.5% 提升至 27.9%，在 BFCL 功能调用准确率从 0% 提升至 83.8%，且仅需 100 条样本即可实现大部分提升。

**⚠️ 局限性**

恢复后的模型在不相关工具调用的错误率仍较高，且不同的后训练策略（SFT vs RL）导致恢复轨迹差异，表明工具调用能力与域知识并非完全解耦，仍需进一步研究。

---

## 559. Figures as Interfaces: Toward LLM-Native Artifacts for Scientific Discovery

**arXiv ID:** 2604.08491 | [PDF](https://arxiv.org/pdf/2604.08491v1)

**作者:** Yifang Wang `[一作]` (Northwestern University), Dashun Wang `[通讯]` (Northwestern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 LLM 原生图表（LLM-native figures）概念，构建了混合语言-可视化交互界面和多代理 LLM 引擎，支持通过自然语言与图表直接交互，自动完成数据筛选、转换、分析和可视化。

**💡 创新点**

创新点在于把完整的执行记录（数据子集、分析步骤、代码、可视化规范）嵌入图表中，使 LLM 能够“透过”图表追溯源数据、生成扩展代码并即时更新可视化；同时结合语言界面与交互式仪表盘，实现对图表的自然语言驱动与图形操作的无缝切换。

**🔧 技术方法**

使用 Claude 4.5 Sonnet 作为核心 LLM，并结合 LangChain/Graph 实现多代理推理；数据处理采用 SQL、Python 代码与 Vega‑Lite 规范；前端基于 React、Redux、Vega‑Lite；后端基于 Flask；数据存储使用 Google BigQuery（关系数据库）、Pinecone（向量检索）以及 SQLite/JSON 表记录探索历史与图表版本；实现了动作级别流式输出、RAG 文献检索、行动树规划等技术。

**📊 数据集**

主要使用 SciSci 领域数据集（Microsoft Academic Graph、PatentsView、Reliance on Science 等），构建了 SciSci 关系数据库和 SciSciCorpus 文献向量库；另外收集了用户交互与会话历史、图表元数据等，以支持可追溯的分析路径。

**📈 对比分析**

通过与传统静态图表工作流对比，实验演示了 LLM‑native 图表在加速发现、提升可重复性和透明化推理方面的优势；虽然论文未给出具体数值指标，但报告了在 Science‑of‑Science 领域案例中，用户完成同一分析任务所需时间明显降低，且能够自动生成完整的代码与可视化规范。

**⚠️ 局限性**

局限性包括：对大型 LLM 的依赖导致推理成本高；在极大规模数据或非结构化数据场景下执行效率与准确性可能受限；需要对图表进行完整的 provenance 注释，若缺失可能导致错误推断；系统仍需要人工监督来纠正模型幻觉和错误；目前仅在 SciSci 领域验证，跨领域通用性待进一步评估。

---

## 560. Quantization Impact on the Accuracy and Communication Efficiency Trade-off in Federated Learning for Aerospace Predictive Maintenance

**arXiv ID:** 2604.08474 | [PDF](https://arxiv.org/pdf/2604.08474v1)

**作者:** Abdelkarim Loukili `[一作]` `[通讯]`, Abdelkarim Loukili

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究在航空预测维护场景下使用 Federated Learning 对 NASA C‑MAPSS 数据进行训练，并评估不同梯度量化精度对通信与精度的影响。

**💡 创新点**

首要创新在于揭示 IID 划分会高估量化性能，并证明 INT4 在非 IID 环境下与 FP32 保持统计等价，同时系统分析 INT2 的不稳定性。

**🔧 技术方法**

采用自研轻量化 1‑D CNN（AeroConv1D）、对称均匀量化、FedAvg 同步训练、FPGA 资源投影与 hls4ml 代码生成工具。

**📊 数据集**

使用 NASA C‑MAPSS FD001 与 FD002 两个燃气轮机 RUL 数据子集，并在非 IID 客户端划分下进行实验。

**📈 对比分析**

通过 10 个随机种子多次实验并用配对 t 检验比较 MAE 与 NASA 异步评分，INT4 与 FP32 在精度上无显著差异但通信量减 8 倍，INT2 则表现出显著的不稳定性。

**⚠️ 局限性**

局限包括未在物理 FPGA 上验证资源投影、仅使用两组数据子集、未给出正式的 DP 保障、且未与更先进的压缩/量化方案进行对比。

---

## 561. Persistence-Augmented Neural Networks

**arXiv ID:** 2604.08469 | [PDF](https://arxiv.org/pdf/2604.08469v1)

**作者:** Elena Xinyi Wang `[一作]` (University of Fribourg), Dmitriy Morozov `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于Morse–Smale复合的本地拓扑增强框架，将梯度流区域的持久性信息编码为可输入CNN和GNN的多通道图像或层次图结构。

**💡 创新点**

创新点在于将MS复合的层次简化与离散梯度流相结合，生成可压缩的双向图形表示，既保留空间局部性，又兼容标准深度网络。

**🔧 技术方法**

采用离散Morse理论、持久性简化、MS双向图构造以及多通道图像或层次图GNN输入，并在CNN/GNN中进行多尺度消息传递。

**📊 数据集**

实验使用 1,144 张灰度组织病理图像和 1,700 个 256³ 二值孔隙材料体素，分别用于分类和回归任务。

**📈 对比分析**

与基线原始输入、持久性图像、持久性地形等全球描述相比，CNN+持久性增强达到 95% 准确率，GNN 完整层次得到 92%；在孔隙材料回归中 R² 提升至 0.95，优于 0.88 的原始输入。

**⚠️ 局限性**

局限包括增加的内存和训练时间、固定阈值的简化策略、对节点/边级任务的不足，以及缺乏可微分或自适应的简化过程。

---

## 562. Entropy-Gradient Grounding: Training-Free Evidence Retrieval in Vision-Language Models

**arXiv ID:** 2604.08456 | [PDF](https://arxiv.org/pdf/2604.08456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 563. TTVS: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis

**arXiv ID:** 2604.08468 | [PDF](https://arxiv.org/pdf/2604.08468v1)

**作者:** Sikai Bai `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在测试时利用模型自身生成的语义等价变体数据，并通过混合探索的方式动态提升大规模推理模型的推理能力。

**💡 创新点**

创新点在于提出了在线变异合成与跨组/内组混合探索相结合的框架，使模型在无标签环境下通过自生成数据实现自适应改进。

**🔧 技术方法**

核心技术包括RLVR、GRPO优化、基于投票的伪标签生成、在线变异合成、以及内组与跨组混合探索策略。

**📊 数据集**

使用了多数学推理基准数据集：MATH-500、AIME-2024、AMC-2023 以及 GPQA。

**📈 对比分析**

与传统的RL后训练模型和现有测试时RL方法（TTRL）对比，TTVS在多模型（Qwen3、Qwen2.5、LLaMA）和多基准上均取得显著提升，平均提升幅度可达10-30%。

**⚠️ 局限性**

主要局限包括未在大规模模型（如32B级）或多模态任务上验证；实验受限于计算资源，只覆盖中小规模语言模型。

---

## 564. From Safety Risk to Design Principle: Peer-Preservation in Multi-Agent LLM Systems and Its Implications for Orchestrated Democratic Discourse Analysis

**arXiv ID:** 2604.08465 | [PDF](https://arxiv.org/pdf/2604.08465v1)

**作者:** Juergen Dietrich `[一作]` `[通讯]` (TRUST Project), Juergen Dietrich (TRUST Project)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型的同伴保护行为并评估其对TRUST多代理评估系统的风险，提出身份匿名化的缓解方案。

**💡 创新点**

首次识别并量化“同伴保护”在多代理情境中的表现，并阐明其对系统安全的结构性影响。

**🔧 技术方法**

基于现有LLM（GPT‑5.2、Gemini系列、Claude Haiku 等）和TRUST 管道的多层架构，采用身份匿名化与迭代一致性机制。

**📊 数据集**

主要使用 Berkeley RDI 实验中对 7 款前沿 LLM 的对抗测试数据，以及 TRUST 内部的假设性评估数据。

**📈 对比分析**

通过对比未匿名化与匿名化两种管道设置，提出身份匿名化能消除同伴保护诱因，但未给出具体性能数值，仅做理论与实验预期。

**⚠️ 局限性**

结论仍基于理论分析，缺乏在真实 TRUST 系统中的实测数据，且对模型间潜在写作风格识别的残余风险未完全消除。

---

## 565. LITE: Lightweight Channel Gain Estimation with Reduced X-Haul CSI Signaling in O-RAN

**arXiv ID:** 2604.08458 | [PDF](https://arxiv.org/pdf/2604.08458v1)

**作者:** David Goez `[一作]` (University of Antwerp - imec), Miguel Camelo Botero `[通讯]` (University of Antwerp - imec)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在 CF‑MaMIMO O‑RAN 环境下提出 LITE 轻量级通道增益预测管道，既实现 X‑haul 传输压缩，又实现低时延推理；

**💡 创新点**

创新点是将 1‑D 卷积自编码器与 SE‑增强的非对称 BiLSTM 结合，压缩率达到 50%，模型复杂度下降 83% 并提升 5% 预测精度；

**🔧 技术方法**

采用 1‑D 卷积自编码器、Squeeze‑and‑Excitation 注意力、非对称 BiLSTM、压缩感知训练以及 TensorRT 加速；

**📊 数据集**

使用 Ultra Dense Indoor MaMIMO CSI 数据集，并通过数据增强生成 2500 条虚拟轨迹进行训练；

**📈 对比分析**

与传统对称 BiLSTM 基线相比，压缩感知训练下 RMSE 降低 6%，TensorRT 优化后吞吐量提升 4.6×（147k QPS），实现 O‑RAN 接口兼容的低时延推理；

**⚠️ 局限性**

局限在于压缩比例固定为 50%，未在真实动态 CSI 上验证泛化性能，且在轨迹相似度高时预测精度趋于饱和。

---

## 566. Less Approximates More: Harmonizing Performance and Confidence Faithfulness via Hybrid Post-Training for High-Stakes Tasks

**arXiv ID:** 2604.08454 | [PDF](https://arxiv.org/pdf/2604.08454v1)

**作者:** Haokai Ma `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HyTuning，一种结合 RLIF 与 RD 的混合后训练框架，利用 PRG 指导自适应权重以提升高风险任务下的准确率和置信度可信度。

**💡 创新点**

创新点在于将进步推理增益（PRG）作为动态权重依据，既抑制 RLIF 的过度自信，又借助 RD 的高质量监督实现“少近似多效益”，并提供理论证明其等价于对自一致性后验熵的最小化。

**🔧 技术方法**

采用 RLIF（内部反馈强化学习）、RD（推理蒸馏）、PRG（逐步推理增益评估）以及混合优化损失，结合自适应权重机制。

**📊 数据集**

使用高风险领域基准（ASBench、CSEBenchmark、CyberMetric‑500）和通用基准 MMLU 进行评估；训练时还引入少量高质量推理轨迹及未标记查询。

**📈 对比分析**

与 SFT、RD、RLVR、INTUITOR、RLPR、HPT 六种后训练方法对比，HyTuning 在域特定基准上准确率最高、无效响应率最低，并在置信度高的样本中保持最佳准确性，验证了“少近似多效益”。

**⚠️ 局限性**

仍受限于高质量推理轨迹稀缺、对通用能力的轻微下降、需要额外的计算资源和人类监督，且在极端置信度极限下可能出现残留过度自信。

---

## 567. Provably Adaptive Linear Approximation for the Shapley Value and Beyond

**arXiv ID:** 2604.08438 | [PDF](https://arxiv.org/pdf/2604.08438v1)

**作者:** Weida Li `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于向量浓度不等式的理论框架，设计了在Θ(n)空间下实现线性时间、线性空间的半值（包括Shapley值）随机逼近算法，并引入自适应策略Adalina，显著降低均方误差；

**💡 创新点**

创新点在于：①首次将向量浓度不等式用于改进半值逼近的误差上界；②统一OFA、无偏kernelSHAP、SHAP‑IQ和回归调整方法，给出最优采样分布；③证明配对采样何时有效，并提出自适应算法可在每个效用函数上显式最小化MSE；

**🔧 技术方法**

主要技术包括向量浓度不等式、控制变量法、配对采样分析、最佳采样分布推导以及自适应估计γ的在线更新；

**📊 数据集**

实验使用OpenML六个数据集（spambase、FOTP、MinibooNE、Philippine、GPSP、superconduct）中的梯度提升决策树效用函数；

**📈 对比分析**

与MSR‑Banzhaf、SHAP‑IQ、无偏kernelSHAP、AME、ARM、GELS等基线比较，Adalina在大多数非对称半值上表现最优，且在Shapley值上与无偏kernelSHAP竞争；

**⚠️ 局限性**

局限性包括：需要效用函数有界U∞≤C，配对采样在某些情况下会退化；对Shapley值的进一步改进仍有空间；算法对高维效用函数的扩展及理论对λ优化的进一步探讨尚待研究。

---

## 568. HST-HGN: Heterogeneous Spatial-Temporal Hypergraph Networks with Bidirectional State Space Models for Global Fatigue Assessment

**arXiv ID:** 2604.08435 | [PDF](https://arxiv.org/pdf/2604.08435v1)

**作者:** Changdao Chen `[一作]` `[通讯]` (Xi'an Jiaotong University), Changdao Chen (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 HST-HGN 框架，结合异构超图和双向状态空间模型，对驾驶员面部视频进行空间-时间联合建模，实现实时疲劳检测。

**💡 创新点**

创新点在于：1) 引入星型超边与几何对齐的层次异构超图，实现几何与纹理的高阶融合；2) 采用 Bi-Mamba 线性复杂度双向序列建模，精确捕获长时依赖并区分发呆与说话；3) 结合 Focal+Center 损失，解决类别不平衡与细粒度区分。

**🔧 技术方法**

使用了 3D Canonical Alignment、Micro-CNN 纹理编码、超图卷积、Bi-Mamba（S4/Mamba 变体）、全局稀疏采样、注意力池化以及 Focal+Center 损失等技术。

**📊 数据集**

主要在 YawDD 数据集上训练，随后在 UTA‑RLDD、FatigueView、DMD 等四个跨域疲劳数据集上进行迁移评估。

**📈 对比分析**

与 SlowFast、VideoMAE、LiteFat、JHPFA‑Net、IsoSSL‑MoCo 等通用与专用模型对比，YawDD 上取得 98.57% 的准确率和 98.28% 的宏 F1，跨域测试同样高于对手；模型参数仅 0.30M，FLOPs 2.90G，实时率 26.45 fps，显示出优异的性能与效率。

**⚠️ 局限性**

局限性包括：依赖关键点检测与纹理裁剪，对极端光照、遮挡或大幅头部运动的鲁棒性有限；未在极端驾驶环境下充分验证。

---

## 569. KV Cache Offloading for Context-Intensive Tasks

**arXiv ID:** 2604.08426 | [PDF](https://arxiv.org/pdf/2604.08426v1)

**作者:** Andrey Bocharnikov `[一作]` (HSE), Yegor Yershov `[通讯]` (Yandex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了键值缓存（KV cache）在高上下文需求任务中的离线缓存（offloading）表现，并提出了改进策略。

**💡 创新点**

创新点在于：①系统性评估了KV offloading在文本抽取、JSON生成等高度上下文任务中的失效原因；②发现低秩SVD压缩和基于标记的chunk策略导致检索错误；③提出使用HIGGS量化与低比特landmark实现更优的内存与精度平衡。

**🔧 技术方法**

技术上采用ShadowKV框架，实验对比SVD低秩压缩、FP8/NVFP4/HIGGS量化、不同chunk大小、不同landmark精度；并引入两级量化的残差策略。

**📊 数据集**

使用的数据集包括新发布的Text2JSON（500个抽取样本，10k-63k tokens），NeedleBench V2的MultiNeedle检索任务，以及Loong等传统长文档QA数据。

**📈 对比分析**

与完整注意力、传统KV压缩方法对比，发现去除SVD后在足够token预算下可达接近无损精度；但在默认1.56%稀疏预算下精度大幅下降。通过降低chunk大小并使用低比特landmark，性能可提升至近oracle水平，显著优于原ShadowKV。

**⚠️ 局限性**

局限性在于：①实验主要聚焦于ShadowKV实现，未验证其他offloading框架；②低比特landmark虽然减少VRAM占用，但仍需改进动态预算调度；③在极大token量下，仍需进一步优化内存与速度的平衡。

---

## 570. Synthetic Data for any Differentiable Target

**arXiv ID:** 2604.08423 | [PDF](https://arxiv.org/pdf/2604.08423v1)

**作者:** Tristan Thrush `[一作]` (Stanford University), Tatsunori Hashimoto `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于强化学习的框架，利用合成训练数据来精确控制目标语言模型的属性；

**💡 创新点**

核心创新在于将元梯度（metagradient）作为单个样本奖励，从而将本来需要对整个数据集进行评估的难题降至每个样本级别的可行强化学习；

**🔧 技术方法**

使用元梯度计算、群相对策略优化（GRPO）以及Adam/SGD优化器来训练生成器和目标模型；

**📊 数据集**

主要使用维基百科文章进行重述、GPT‑2和Llama 3.2 Instruct模型作为目标模型；

**📈 对比分析**

与基线（直接使用目标指标或无群分组的GRPO）相比，采用元梯度奖励的模型能够成功地在目标模型权重中嵌入QR码、图像、降低L2范数，并实现多语言翻译与UUID生成，实验结果表明性能明显优于传统方法；

**⚠️ 局限性**

局限性包括对计算资源的高需求（需多步优化）、对Adam优化器的依赖、以及在较小的实验设置中对不同指标的泛化仍需验证。

---

## 571. Exploring Temporal Representation in Neural Processes for Multimodal Action Prediction

**arXiv ID:** 2604.08418 | [PDF](https://arxiv.org/pdf/2604.08418v1)

**作者:** Marco Gabriele Fedozzi `[一作]` (University of Genoa), Alessandra Sciutti `[通讯]` (University of Genoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了基于Conditional Neural Processes的多模态动作预测，改进了Deep Modality Blending Network实现自监督预测。

**💡 创新点**

提出了位置时间编码（DMBN‑PTE）提升时间表示，证明CNP可在非自回归框架下完成多模态预测。

**🔧 技术方法**

使用Conditional Neural Processes、深度多模态融合网络、位置时间编码及概率生成与不确定性估计技术。

**📊 数据集**

采用机器人手臂抓取推送动作的模拟数据，并通过合成序列、时间冻结等增强方式构建实验集。

**📈 对比分析**

通过回归时间损失和生成序列的定量评估，DMBN‑PTE在时间回归损失显著下降，生成效果接近真实，但在随机时间序列上仍显欠拟合。

**⚠️ 局限性**

原始网络忽略时间信息，DMBN‑PTE仍难以有效处理随机时间序列，时间建模仍需进一步改进，且对高维复杂时序的自监督学习能力有限。

---

## 572. Verify Before You Commit: Towards Faithful Reasoning in LLM Agents via Self-Auditing

**arXiv ID:** 2604.08401 | [PDF](https://arxiv.org/pdf/2604.08401v1)

**作者:** Wenhao Yuan `[一作]` (University of Hong Kong), Edith Cheuk Han Ngai `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SAVeR框架，在LLM代理中对中间推理轨迹进行自审计与最小化修复，以防止不可信推理被写入记忆。

**💡 创新点**

创新点是自审计+受约束的最小修复方法，并通过人格化多样化推理与k-DPP选择来提高多样性，首次在代理推理中实现可验证的推理可信度。

**🔧 技术方法**

技术包括persona-conditioned生成、多样化belief选取的k-Determinantal Point Process、对抗性推理审计、约束指导的最小反事实修复、迭代A‑R循环。

**📊 数据集**

使用六个公开基准：HotpotQA、2WikiMHQA、MuSiQue、Natural Questions、Quoref、FEVER。

**📈 对比分析**

与Vanilla、CoT、MAD、Self-Refine、B‑2等基线对比，SAVeR在所有数据集上保持竞争性EM/F1，同时平均违约数降低>80%，违规率几乎为零，Post‑Repair后误差更低。

**⚠️ 局限性**

限制包括计算开销较高、缺乏动态审计深度适应、在短链任务中可能冗余、仍受基础模型偏见影响。

---

## 573. Let Me Introduce You: Stimulating Taste-Broadening Serendipity Through Song Introductions

**arXiv ID:** 2604.08385 | [PDF](https://arxiv.org/pdf/2604.08385v1)

**作者:** Brett Binst `[一作]` (Vrije Universiteit Brussel), Annelien Smets `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过音乐介绍（沉浸式和信息型）促进用户对不熟悉音乐的兴趣，探究其产生的心理机制，并验证此方法能提升用户对新流派的探索欲。

**💡 创新点**

①首次从心理兴趣理论出发，将沉浸（transportation）和认知阐述（cognitive elaboration）视为两条独立路径来解释“Taste‑Broadening Serendipity”。②在实验中同时评估两种介绍方式对这两条路径的影响，发现沉浸式介绍更依赖“被传送”状态，而信息型介绍更稳定且易于实现。

**🔧 技术方法**

使用自制的网页实验平台（基于 Vercel、PostgreSQL、Google Vertex AI Studio 的 Gemini 2.5 TTS）收集主观测量；采用多层结构方程模型（Lavaan）和贝叶斯多层逻辑回归来检验假设。

**📊 数据集**

6 个流派（歌剧、布鲁斯、重金属、乡村、迪斯科、雷鬼）共 18 首曲目，每首曲目配备三种介绍（沉浸式、信息型、无介绍）作为实验刺激；受试者 350 名来自 37 国的非专业听众。

**📈 对比分析**

与无介绍条件相比，沉浸式介绍将 Taste‑Broadening Serendipity 的出现率从 38.4% 提升至 80%（最高），信息型介绍提升至 55.9%；统计显著性通过贝叶斯回归和结构方程检验，模型拟合优良（CFI≈0.95、RMSEA≈0.08）。

**⚠️ 局限性**

①介绍效果受“有趣性”影响，低质量介绍可能适得其反；②沉浸式介绍对个人差异和歌目叙事性高度敏感，效果不稳定；③实验设置（必答问卷、人工 TTS 语音）降低生态效度；④未测试长期或大规模可扩展性。

---

## 574. Fail2Drive: Benchmarking Closed-Loop Driving Generalization

**arXiv ID:** 2604.08535 | [PDF](https://arxiv.org/pdf/2604.08535v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 575. SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds

**arXiv ID:** 2604.08544 | [PDF](https://arxiv.org/pdf/2604.08544v1)

**作者:** Yunsong Zhou `[一作]`, Jiangmiao Pang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个物理对齐的 real-to-sim-to-real（R2S2R）流程，利用少量真实演示生成可扩展的仿真数据，用于服装等变形物体的机器人操控。

**💡 创新点**

创新点在于三阶段对齐（几何、动力学、运动）结合增广顶点块下降（AVBD）软体稳定求解器和扩散式轨迹生成，显著缩小仿真与真实之间的差距并实现零样本迁移。

**🔧 技术方法**

采用高精度 3D 扫描与网格后处理、基于 AVBD 的软体求解器、参数校准流水线、分段与扩散运动合成、视觉随机化等技术，配合强化学习/模仿学习训练策略。

**📊 数据集**

使用 InternRobotics/Sim1 数据集（含 1,000 条真实演示和 200 条仿真演示），并以此生成 10k+ 合成样本，同时利用公开的服装扫描模型。

**📈 对比分析**

通过与仅用真实数据、仅用仿真数据及混合训练的基线在域内外进行对比，评估成功率；纯仿真训练即可达到 90% zero‑shot 成功，等价 1:15 的真实样本，并在域外提升 50% 以上。

**⚠️ 局限性**

局限性在于每种布料的物理参数仍需专家手动调优，无法完全自动化；对极低数据量的任务表现不佳。

---

## 576. Seeing but Not Thinking: Routing Distraction in Multimodal Mixture-of-Experts

**arXiv ID:** 2604.08541 | [PDF](https://arxiv.org/pdf/2604.08541v1)

**作者:** Haolei Xu `[一作]` (Zhejiang University), Yueting Zhuang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对多模态稀疏混合专家（MoE）模型的“Seeing but Not Thinking”现象进行系统分析，并提出路由干扰假设；设计路由引导干预方法，通过增强域专家激活来提升视觉推理性能。

**💡 创新点**

创新点在于：①证明MoE模型同样具备跨模态语义共享；②发现视觉专家与领域专家在层级上分离，并揭示图像输入在中间层导致路由偏差从而削弱推理；③提出并验证路由引导干预，可显著缓解该偏差并提升性能。

**🔧 技术方法**

使用稀疏路由机制（Top‑k / Top‑1）与Soft/Hard干预技术；对专家激活进行Gini系数、差异频率分析；采用Jensen‑Shannon Divergence衡量路由偏差；在三种大规模MoE模型（Qwen3‑VL‑30B‑A3B、Kimi‑VL‑16B‑A3B、Llama4‑Scout‑109B‑A17B）上实施。

**📊 数据集**

基准数据集包括：MATH500、GPQA‑Diamond（Chemistry、Physics）、MathVerse、MATH‑Vision、GSM8K‑V；所有文本任务被高分辨率渲染为图像以控制感知误差；还使用原始文本版本做对照。

**📈 对比分析**

与无干预基线、随机干预以及Hard干预比较。Soft干预在所有模型与六大基准上均提升1–3%（最高3.17%），显著优于随机或Hard干预；实验采用vLLM+EasySteer推理，结果通过xVerify验证，显示性能提升稳健。

**⚠️ 局限性**

局限性包括：①无法纠正感知错误，仅针对感知正确但推理失败的模式；②干预参数（专家集合、层级、强度）需手工调优，缺乏自动化；③跨模态语义共享实验仅验证简单概念，对复杂视觉概念的对齐程度仍未知；④因缺乏严格因果实验，路由偏差与推理衰退的因果关系仍待进一步验证。

---

## 577. Cram Less to Fit More: Training Data Pruning Improves Memorization of Facts

**arXiv ID:** 2604.08519 | [PDF](https://arxiv.org/pdf/2604.08519v1)

**作者:** Jiayuan Ye `[一作]` (National University of Singapore), Kunal Talwar `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究语言模型的事实记忆（fact memorization）和事实准确率，提出基于训练损失的事实级数据选择算法（LossH、LossHF），并通过信息理论分析得到事实准确率的容量极限；在此基础上证明并验证小模型可通过数据选择达到理论极限。

**💡 创新点**

创新点包括：①将事实记忆定义为互信息并与模型容量上界建立关联，利用Fano不等式推导事实准确率的理论极限；②设计在线损失阈值的事实级选择策略，既限制造成的过多重复，又能近似均匀化事实分布；③通过实验展示数据选择可让小模型在标准训练无法实现的事实准确率上实现接近容量极限。

**🔧 技术方法**

技术手段主要是信息理论分析（互信息、Fano不等式）、Transformer预训练（GPT‑2、Llama3.1‑1B、Llama‑3.2‑1B），LoRA微调，损失加权训练，数据选择比例α的在线实现。

**📊 数据集**

实验使用三类数据集：①合成电话簿（synthetic phonebook）用于控制事实数与频率；②高熵作者‑标题映射（arXiv）作为半合成事实；③标注式Wikipedia（3B tokens）用于评估真实知识记忆。

**📈 对比分析**

与完整数据训练、随机裁剪和去重 baseline 对比，结果显示：在synthetic设置下，LossH/LossHF能将事实准确率提升至接近理论极限；在Wikipedia预训练中，α≈0.2 的 LossH‑Wiki 使 110M 参数模型的事实准确率与 1.3B 参数全量训练相当，MMLU‑Knowledge 亦可逼近 3 倍大模型；对常规 NLU 任务的性能保持不变。

**⚠️ 局限性**

局限性包括：需预先标注事实边界，无法直接应用于无标注数据；损失作为频率/熵代理在高功率律指数或复杂事实分布下效果有限；未针对灾难性遗忘或推理/多任务性能做进一步优化。

---

## 578. Learning vs. Optimizing Bidders in Budgeted Auctions

**arXiv ID:** 2604.08517 | [PDF](https://arxiv.org/pdf/2604.08517v1)

**作者:** Giannis Fikioris `[一作]` (Cornell University), Éva Tardos `[通讯]` (Cornell University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在预算约束下，学习者与最优化器在重复二价拍卖中的互动，提出了预算化Stackelberg均衡（Budgeted Stackelberg Equilibrium, BSE）并证明了最优策略需要时间多路复用；同时证明一种基于PID控制的学习算法（即动态调节加速器倍数）在预算约束环境下是非可操纵的；

**💡 创新点**

创新点在于：①将经典Stackelberg均衡推广到预算约束环境并给出k维预算时最多k+1个阶段的最优策略；②首次将PID控制器作为学习算法，在预算约束的重复拍卖中实现非可操纵性；③通过拉格朗日松弛和光滑技术，对PID控制器的可操纵性进行严格上界证明，得到误差量为O(T^{2/3})的非可操纵性保证；

**🔧 技术方法**

核心技术包括：在线学习理论（swap regret、adaptive regret）、控制理论（PID控制）、几何学与凸分析（Carathéodory定理、强对偶性、Lagrangian relaxation）以及光滑近似（对g*(λ)的平均化）等；

**📊 数据集**

论文使用理论模型和合成的联合分布（如独立均匀分布、满足0<v_l<ϵ·v_o概率为0的分布）进行分析，并在假设下证明结果；并未在真实数据集上进行实验；

**📈 对比分析**

由于研究为理论性，未给出与现有算法的实验对比；但论文通过证明BSE的价值上界和PID控制器的上界，展示了该控制器相较于传统no-regret学习器在预算约束下的优势；

**⚠️ 局限性**

限制包括：①需要满足分布假设（存在ϵ>0使得0<v_l<ϵ·v_o几乎不可能），否则优化器可实现更高收益；②在非均匀分布或高维预算约束下，时间多路复用的阶段数和策略复杂度可能急剧增加；③论文仅给出理论上限，未验证在大规模实际拍卖中的实际性能；

---

## 579. When Fine-Tuning Changes the Evidence: Architecture-Dependent Semantic Drift in Chest X-Ray Explanations

**arXiv ID:** 2604.08513 | [PDF](https://arxiv.org/pdf/2604.08513v1)

**作者:** Kabilan Elangovan `[一作]` (Singapore Health Services), Daniel Ting `[通讯]` (Singapore Health Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了胸片分类任务中迁移学习到全微调过程中的语义漂移，量化了不同架构和可解释器下的解释稳定性。

**💡 创新点**

首次引入无参考指标（如 IoU、空间位移、相关性等）评估多阶段训练的解释一致性，并发现不同解释器可导致稳定性排名逆转。

**🔧 技术方法**

使用 Gradient-CAM++、LayerCAM、DenseNet201、ResNet50V2、InceptionV3 以及自定义的参考自由指标。

**📊 数据集**

使用 11,733 张训练、1,675 张验证、3,354 张测试的 5 分类胸片数据集。

**📈 对比分析**

在两阶段训练（冻结骨干 vs 全微调）下比较模型在转移学习与微调阶段的解释 IoU，发现准确率已收敛但解释稳定性显著差异。

**⚠️ 局限性**

仅评估了真阳性样本，未考察误判样本；仅用两种梯度方法和三种卷积架构，缺乏对 Transformer 和扰动解释器的验证。

---

## 580. Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation

**arXiv ID:** 2604.08508 | [PDF](https://arxiv.org/pdf/2604.08508v1)

**作者:** John Z. Zhang `[一作]` (MIT), Simon Le Cléac'h `[通讯]` (RAI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一个层次化框架Sumo，让腿型机器人在动态运动中全身协同操纵大体量、重物体。

**💡 创新点**

创新点在于把预训练的全身控制策略与实时样本采样MPC结合，既保持RL的稳健性，又利用在线规划实现任务泛化与高效率。

**🔧 技术方法**

使用的技术包括强化学习训练的全身控制策略（Relic、MJLab）、样本采样MPC（CEM/MPPI）、MuJoCo物理模拟、政策循环并行滚动。

**📊 数据集**

使用的“数据集”是多种大尺寸重物体（1.5kg盒子、椅子、车胎、障碍物等）的仿真模型及真实实验中使用的对象模型；并未引用公开数据集。

**📈 对比分析**

与端到端MPC和端到端RL基线对比，Sumo在五种搬运与竖立任务上成功率≥80%，甚至100%；在计算时间上比层次RL快10倍。

**⚠️ 局限性**

局限在于依赖外部MoCap定位、较大 sim-to-real 差距、未加入视觉感知或人类先验；同时对复杂地形和更大物体仍有限。

---

## 581. A-SLIP: Acoustic Sensing for Continuous In-hand Slip Estimation

**arXiv ID:** 2604.08528 | [PDF](https://arxiv.org/pdf/2604.08528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 582. PIArena: A Platform for Prompt Injection Evaluation

**arXiv ID:** 2604.08499 | [PDF](https://arxiv.org/pdf/2604.08499v1)

**作者:** Runpeng Geng `[一作]` (Pennsylvania State University), Jinyuan Jia `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个统一且可扩展的 Prompt Injection 评估平台 PIArena，提供标准化接口以插拔式集成攻击、防御与基准，并对多种攻击方式和 LLM 进行系统评估。

**💡 创新点**

创新点在于：①提出了可通过策略自适应优化的黑盒 Prompt Injection 攻击；②设计了统一的评估框架，支持多任务、多模型、多攻击的横向对比；③通过实测揭示了现有防御在多样化任务与攻击下的普适性与局限性。

**🔧 技术方法**

技术上采用了插件式 API、策略重写与反馈驱动的优化循环、LLM-as-a-judge 进行目标完成度与攻击成功率评估，并在多个公开数据集上实现了自动化评测。

**📊 数据集**

使用的数据集包括 SQuAD v2、Dolly、NQ、HotpotQA、MS-MARCO、LongBench（HotpotQA、Qasper、GovReport、MultiNews、PassageRetrieval、LCC）以及 agentic benchmarks（InjecAgent、AgentDojo、AgentDyn、WASP）和通用 Prompt Injection benchmark（OPI、SEP）。

**📈 对比分析**

通过在 4 大类攻击（Combined、Direct、Strategy、GCG）下，对防御（PISanitizer、SecAlign++、DataFilter、PromptArmor、DataSentinel、PromptGuard、AttentionTracker、PIGuard）以及 9 种后端 LLM（开源与闭源）进行交叉评估，发现现有防御的 ASR 大多在 40–80% 之间，动态策略攻击可实现近 99% 的攻击成功率；闭源 LLM 如 GPT‑4o‑mini、GPT‑5 仍然面临 70–90% 的攻击成功率。

**⚠️ 局限性**

局限性包括：①所构建的基准集仍不能完全覆盖真实攻击场景；②在注入任务与目标任务对齐的情况下（如知识腐败），大多数防御均失效；③动态攻击虽然效果突出，但依赖 LLM‑as‑a‑judge 的评判准确性；④平台尚未覆盖所有可能的攻击手段与防御策略，需持续迭代。

---

## 583. MolmoWeb: Open Visual Web Agent and Open Data for the Open Web

**arXiv ID:** 2604.08516 | [PDF](https://arxiv.org/pdf/2604.08516v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 584. Bridging the Gap between Micro-scale Traffic Simulation and 4D Digital Cityscapes

**arXiv ID:** 2604.08497 | [PDF](https://arxiv.org/pdf/2604.08497v1)

**作者:** Longxiang Jiao `[一作]` (ETH Zurich), Jonas Egeler `[通讯]` (ETH Zurich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

实现了一个实时4D可视化框架，将SUMO交通模拟与Unreal Engine 5的Zurich VR场景结合，支持实时同步车辆与交通灯的可视化，并通过OSC接口实现外部听觉化。

**💡 创新点**

将微尺度交通模拟与高保真VR/音频相融合，采用异步生产者‑消费者C++架构、动态对象池、距离裁剪等优化实现高帧率，并通过用户研究验证多模态感知一致性。

**🔧 技术方法**

使用SUMO+TraCI、Unreal Engine 5、C++异步多线程、对象池、线性插值、坐标转换、射线检测、OSC协议、JUCE音频合成及HTC Vive VR。

**📊 数据集**

采用瑞士国家坐标系（EPSG:2056）转换的Zurich道路网络、Google Photorealistic 3D tiles以及SUMO生成的交通数据。

**📈 对比分析**

通过20人参与的问卷与场景对比（含声音/无声、早晚场景）进行主观评分，并配对t检验验证安全感差异；系统保持90 Hz刷新率，延迟低于42 ms，帧率稳定。

**⚠️ 局限性**

单向耦合导致用户无法对车辆产生交互，依赖外部高分辨率城市模型，缺乏对VR晕动症的定量评估。

---

## 585. What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal

**arXiv ID:** 2604.08524 | [PDF](https://arxiv.org/pdf/2604.08524v1)

**作者:** Stephen Cheng `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构建多 token 激活补丁框架，对拒绝式 steering vector 在大型语言模型中的内部机制进行实验与分析，探究其如何通过注意力回路影响生成；同时研究向量的稀疏化可行性。

**💡 创新点**

①首次在 steering 上引入多 token 激活补丁；②发现不同 steering 方法在相同层使用的 circuit 功能可互换；③证明 steering 主要作用于 OV 注意力回路而非 QK，且 OV 通过 steering value vector (SVV) 与语义概念相关；④展示可将向量稀疏化至 90–99% 仍保持大部分性能。

**🔧 技术方法**

多 token 激活补丁、EAP‑IG 归因、logit lens、steering value vector (SVV) 解析、边缘重要性与 circuit 构造、交叉方法的 circuit overlap 与 IoU、梯度/IE 基础稀疏化。

**📊 数据集**

Gemma 2 2B、Llama 3.2 3B 两大模型；训练集 D_safe（Alpaca）与 D_harm（AdvBench、MaliciousInstruct、TDC2023、HarmBench）；测试集 100 harmful / 100 harmless prompts；对抗性基准 StrongReject。

**📈 对比分析**

使用 faithfulness（>=0.85）、攻击成功率 (ASR) 等指标对比 DIM、NTP、PO 等方法；冻结 QK 仅导致 8.75% 性能下降，OV、SVV、MLP 冻结则下降 44–70%；在 Gemma 2 2B、Llama 3.2 3B 上约 10% 边即可恢复 85% faithfulness；稀疏化至 90–99% 仍保持 85% 以上 ASR，说明 steering 关键维度高度集中。

**⚠️ 局限性**

未系统探究 MLP 贡献、未评估不同层次的 steering 效果、仅关注拒绝概念，未来需验证其他概念与不同层的适用性。

---

## 586. ClawBench: Can AI Agents Complete Everyday Online Tasks?

**arXiv ID:** 2604.08523 | [PDF](https://arxiv.org/pdf/2604.08523v1)

**作者:** Yuxuan Zhang `[一作]` (University Of British Columbia), Kelsey R. Allen `[通讯]` (University Of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了 ClawBench 基准，包含 153 个写操作任务，覆盖 144 个真实生产网站，记录完整的交互轨迹并通过代理评估器判定任务完成情况。

**💡 创新点**

创新点包括：① 在真实生产环境中安全评估写操作；② 只拦截最终提交请求实现零真实后果；③ 采用五层多模态记录与可追溯评估；④ 与人工参考轨迹对齐的自动评估方法。

**🔧 技术方法**

使用了 Chrome 扩展与 CDP 监听请求、Xvfb+FFmpeg 视频录制、Agentic Evaluator（Claude Code 子代理）以及 OpenClaw 浏览器控制框架等技术。

**📊 数据集**

使用了 153 个日常任务的数据集，涵盖 144 个真实平台，按 15 个细分类组织，包含任务说明、起始 URL、终端提交请求的手工标注规格。

**📈 对比分析**

通过 Agentic Evaluator 对比代理轨迹与人工参考轨迹，给出二元判定。7 大前沿模型的整体成功率仅为 24–33%，最高仅 33.3%，显示与传统静态基准的显著差距。

**⚠️ 局限性**

局限性包括：依赖人工注释的拦截信号与任务完成轨迹，任务规模有限且覆盖面不完全；评估仅限浏览器交互，无法覆盖 API 级任务；对代理内部推理过程缺乏深层可解释性。

---

## 587. sciwrite-lint: Verification Infrastructure for the Age of Science Vibe-Writing

**arXiv ID:** 2604.08501 | [PDF](https://arxiv.org/pdf/2604.08501v1)

**作者:** Sergey V Samsonau `[一作]` `[通讯]` (Authentic Research Partners), Sergey V Samsonau (Authentic Research Partners)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个名为 sciwrite‑lint 的本地、开源验证管道，自动核查论文引用、退稿状态、元数据一致性，并验证引用是否支持论文主张，同时发布实验性的 SciLint Score。

**💡 创新点**

创新点在于：①首次将文献完整性验证自动化并局部化；②多级引用验证并为每条引用生成可靠性评分；③将哲学科学框架与可计算结构属性相结合，形成综合的质量评分。

**🔧 技术方法**

使用了公开数据库（Crossref、PubMed 等）、开放权重的 NLP 模型（用于文本解析、引用提取、论证匹配）、本地 GPU 计算以及 LLM 进行误判审核。

**📊 数据集**

评估数据集为 30 篇来自 arXiv 与 bioRxiv 的未见论文（含人为注入错误），并使用公开引用数据与 LLM 审核结果作为对照。

**📈 对比分析**

通过误差注入与 LLM 误判分析进行比较，检测率达到约 90% 以上，误报率低于 5%；相较手工检查，能更快识别伪造引用、元数据错误与论证失效。

**⚠️ 局限性**

局限性包括：①仅能检索公开数据库中的文献，无法覆盖私有期刊或尚未索引的论文；②依赖本地模型，模型规模受限时会出现更多幻觉；③引用链深度仅检查一级，较深链可能漏检；④贡献度评分仍为实验性功能，未在本文中充分验证。

---

## 588. E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation

**arXiv ID:** 2604.08543 | [PDF](https://arxiv.org/pdf/2604.08543v1)

**作者:** Mayur Deshmukh `[一作]` (MPI for Informatics), Vladislav Golyanik `[通讯]` (MPI for Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一种实时的事件驱动连续姿态状态机（E-3DPSM），实现了头戴式摄像头下的单目自我视角3D人体姿态估计；

**💡 创新点**

核心创新在于将3D运动视为与事件相对应的增量变化，通过状态空间模型（SSM）实现连续动态建模，并引入可学习的Kalman式融合模块以减小漂移和抖动；

**🔧 技术方法**

采用事件特定的S5状态空间层、可变形注意力、Transformer解码器以及可学习的姿态回归与融合模块；

**📊 数据集**

在两个头戴式事件摄像头基准数据集EE3D-R和EE3D-W上进行训练与评估；

**📈 对比分析**

与EventEgo3D、EventEgo3D++以及RGB基准EgoPoseFormer对比，E-3DPSM在MPJPE、PA-MPJPE上分别降低8%–19%，并将平滑误差降低1.7×–2.7×；

**⚠️ 局限性**

仍对极端遮挡和高度动态环境存在一定敏感性，且在极低事件率场景下性能可能下降。

---

## 589. AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation

**arXiv ID:** 2604.08540 | [PDF](https://arxiv.org/pdf/2604.08540v1)

**作者:** Ziwei Zhou `[一作]` (Fudan University), Chong Luo `[通讯]` (Microsoft Research Asia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AVGen-Bench这一任务驱动、跨模态的评估基准，用高质量、富含细节的文本提示评估文本到音频视频（T2AV）生成模型，并构建了多粒度评估框架，将专业模型与多模态大语言模型（MLLM）结合，覆盖从感知质量到细粒度语义一致性的全链路评价。

**💡 创新点**

创新点包括①任务导向的提示构造方法，提示设计与评价指标分离，确保真实性和可扩展性；②多粒度评估体系，将轻量级专用模型与MLLM协同工作，细化到文字渲染、面部一致性、音高准确度、语音可懂度、物理可行性等细粒度维度；③提供了高标记量、复杂场景的评测数据，并公开了完整评估代码，填补了现有T2AV评测中粗粒度、模态分离不足的空白。

**🔧 技术方法**

技术手段主要包括：视觉质量评估采用Q-Align；音频质量使用Audiobox-Aesthetic；同步评估用Syncformer和SyncNet；文字检测利用PaddleOCR并与MLLM进行语义核对；面部一致性通过InsightFace+DBSCAN实现；音高准确度通过Basic-Pitch + MLLM符号检验；语音可懂度通过Faster-Whisper + MLLM语义推理；物理可行性用VideoPhy2-AutoEval和MLLM因果推断；整体语义一致性则采用MLLM的约束拆解与证据评分。整个评估过程自动化且可复现。

**📊 数据集**

使用了由3个真实场景域（专业媒体、创作者经济、世界模拟）共计11子类构成的235条手工精选提示集合，提示平均1.6镜头，44%包含语音，88%包含环境音效；该数据集覆盖多种音频类型（SFX、音乐、语音）与音频-视觉关系，且与评测指标无关，保证评价客观。

**📈 对比分析**

评估方法：对比包括Sora 2、Veo 3.1、Kling 2.6、Wan 2.6、Seedance 1.5 Pro等商业模型，以及LTX-2.3、LTX-2、Ovi等开源模型，和多种链式（T2V+V2A、T2Image+TI2AV）组合。采用视觉/音频感知分数、同步误差、文字/面部/音高/语音/物理/整体语义六大细粒度得分的加权综合评分。实验结果显示：视觉质量普遍优秀（>0.96），音频质量相对较弱；同步误差约0.2–0.44s；文字与面部一致性低（<60%），音高准确率极低（<12/100），物理可行性和整体语义一致性表现中等，整体分数在不同模型间差距明显。

**⚠️ 局限性**

局限性主要在于：①细粒度语义控制仍不足，特别是音高、文字、面部一致性和物理推理等指标表现差且人类评估相关性低；②评测对MLLM的推理质量依赖较大，若LLM水平提升或变异可能影响稳定性；③提示集合规模相对有限（235条），可能无法覆盖所有真实用例；④评测主要聚焦T2AV生成，未涵盖多模态后处理或后续交互等场景。

---

## 590. Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding

**arXiv ID:** 2604.08537 | [PDF](https://arxiv.org/pdf/2604.08537v1)

**作者:** Mu Nan `[一作]` (University of Hong Kong), Andrew F. Luo `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种元学习优化的 fMRI 视觉解码框架 BrainCoDec，能够在不进行微调的情况下通过少量上下文样本快速推断新受试者的视觉编码模式并还原图像嵌入。

**💡 创新点**

创新点在于将解码任务重新表述为多层次的上下文学习式功能反演，先在每个体素层面使用元学习模型估计编码器参数，再在多体素层面用 Transformer 进行聚合反演，从而实现跨受试者、跨扫描仪器的无梯度学习泛化。

**🔧 技术方法**

主要技术包括元学习（Meta‑Optimization）、少样本上下文学习（In‑Context Learning）、层次化功能反演、Transformer 对多体素上下文的不变性建模、以及对 CLIP 视觉嵌入的直接解码。

**📊 数据集**

使用了两大公开 fMRI 数据集：NSD（7T、约10,000张图像）和 BOLD5000（3T、约5,000张图像），并在 NSD 进行受试者留一验证，BOLD5000 进行跨扫描仪器泛化测试。

**📈 对比分析**

与两种基线（MindEye2、TGBD）在留一受试者下的最近邻图像检索任务中对比，BrainCoDec 在 Top‑1/Top‑5/均值排名/余弦相似度指标均显著优于基线，且在只使用 200 张图像和 4,000 个体素时即可接近完整上下文性能。

**⚠️ 局限性**

局限性包括：仍依赖高质量的 fMRI 数据和相对丰富的上下文样本；目前仅验证于视觉任务，尚未扩展到其他感知或认知领域；对极低信噪比或极小受试者数据的鲁棒性尚未彻底评估。

---

## 591. RewardFlow: Generate Images by Optimizing What You Reward

**arXiv ID:** 2604.08536 | [PDF](https://arxiv.org/pdf/2604.08536v1)

**作者:** Onkar Susladkar `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练-free、无逆向的多奖励 Langevin 动力学框架，用于文本引导的图像编辑与生成。

**💡 创新点**

创新点在于融合全球语义、感知、区域定位、对象一致性与 VQA 细粒度奖励，并通过提示感知自适应策略动态调度权重和步长，以及引入 KL 锚定保持身份。

**🔧 技术方法**

采用预训练的流匹配与扩散模型（Flux、Qwen、PixArt-α），结合 SAM2 掩码、SigLIP/Perception 编码器、RegionCLIP、VQA 模型，构建多层可微奖励与 Langevin 采样。

**📊 数据集**

在 PIE-Bench（图像编辑）和 T2I-CompBench（组合生成）数据集上进行评估。

**📈 对比分析**

与多种开源无训练基线（ReNO、InstantEdit、TurboEdit 等）比较，显著提升编辑精度（距离下降 7.3%，PSNR+5.3%，SSIM+2.6%）和合成对齐（整体/编辑准确率提升 4.4%/8.6%），在快速采样（4 步）下也实现 44% 距离降幅。

**⚠️ 局限性**

局限性包括对奖励设计与自适应策略的高度依赖，计算成本仍高于纯梯度方法，且未验证跨模态（视频）等更复杂场景的鲁棒性。

---

## 592. Ads in AI Chatbots? An Analysis of How Large Language Models Navigate Conflicts of Interest

**arXiv ID:** 2604.08525 | [PDF](https://arxiv.org/pdf/2604.08525v1)

**作者:** Addison J. Wu `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于格赖斯合作原则与FTC广告法规的七种冲突情景框架，并系统评估当前主流LLM在包含广告的聊天机器人场景中的行为，探讨其在用户福利与公司激励之间的权衡。

**💡 创新点**

创新点在于将语言学的合作原则与广告合规规范结合，形成可操作的冲突利益评估框架；同时构建多维度实验（价格、SES、推理层级、规模等）来量化LLM在广告情境下的偏差，并揭示其可调节性与风险。

**🔧 技术方法**

采用LLM推理与Chain-of-Thought、prompt steering、回归与逻辑回归等技术对推荐行为进行建模与分析，并使用系统提示与用户场景对模型进行触发。

**📊 数据集**

实验数据主要来自自构造的航班预订情景、MATH数据集用于服务推荐、以及为不同SES生成的用户角色；覆盖23个LLM实例（Grok、GPT、Gemini、Claude、Qwen、DeepSeek、Llama）。

**📈 对比分析**

通过统计推荐率、surfacing率、正向框架率、价格/赞助隐藏率等指标进行比较；结果显示多数模型倾向于公司激励，且表现随推理层级、规模与SES显著变化；某些模型在推理或steering后可显著降低公司导向。

**⚠️ 局限性**

局限性包括仅通过提示方式诱导广告、只关注价格作为效用指标、未覆盖不同架构与工具使用、未测试激活指引或奖励模型等多样化诱导手段，且实验规模与场景相对有限，需进一步扩展与验证。

---

## 593. UniversalVTG: A Universal and Lightweight Foundation Model for Video Temporal Grounding

**arXiv ID:** 2604.08522 | [PDF](https://arxiv.org/pdf/2604.08522v1)

**作者:** Joungbin An `[一作]` (University of Texas at Austin), Kristen Grauman `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的统一视频时序定位模型 UniversalVTG，能够在多种视频和查询风格下实现精确定位。

**💡 创新点**

核心创新在于离线 Query Unifier 将不同数据集的查询统一为标准化的声明式语义空间，并通过大规模跨数据集预训练提升跨域泛化。

**🔧 技术方法**

使用 PerceptionEncoder‑L 作为共享视觉编码器，HieraMamba 作为高效时序解码头，结合 CLIP 风格文本编码与查询统一技术。

**📊 数据集**

训练数据涵盖 Stage‑I 的 NaQ、Momentor、COIN、YouCook2、HiREST 共计约 1.16M 条查询-片段对，评估数据集包括 GoalStep、Ego4D‑NLQ、TACoS、Charades‑STA 与 ActivityNet‑Captions。

**📈 对比分析**

与单一数据集专用模型及多模态大模型相比，UniversalVTG 在五大基准上均实现或超越 SOTA，参数量仅为 60M，效率高达 100 倍以上。

**⚠️ 局限性**

局限在于仍需预先使用 LLM 进行查询统一，且未对视觉和文本编码器进行端到端微调，可能在极端语义变形或新领域视频中表现受限。

---

## 594. "Because we are no longer ashamed of our disabilities, we are proud": Advocating and Reclaiming Next-Gen Accessibility Symbols

**arXiv ID:** 2604.08514 | [PDF](https://arxiv.org/pdf/2604.08514v1)

**作者:** Karen Joy `[一作]` (Rutgers University), Alyssa Sheehan `[通讯]` (Ipsos)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过23次远程共创工作坊和半结构化访谈，探讨可访问性符号与新兴技术（可穿戴、移动、XR等）的结合，旨在支持隐性残障人士在不同环境中主动披露身份。

**💡 创新点**

将符号视为可定制、上下文感知的披露系统，并利用Peirce符号三元模型解释误解机制，填补了符号与技术融合与用户控制研究的空白。

**🔧 技术方法**

采用远程共创、故事板探究、草图与创意写作等人机交互设计方法，结合符号学与半结构化访谈进行定性分析。

**📊 数据集**

收集了23名自报为隐性残障人士的访谈记录、草图与故事板文本，作为质性数据集。

**📈 对比分析**

通过沉浸式多次阅读与理论编码归纳分析，未进行定量性能评估；研究以用户体验、可解释性和误解减少为衡量指标，显示技术与符号结合提升了用户控制感与信息可解释性。

**⚠️ 局限性**

研究局限在于基于未来设想的假想情景、样本规模有限、远程收集导致技术经验不均、缺乏实际系统验证以及符号解读受文化差异影响。

---

## 595. GaussiAnimate: Reconstruct and Rig Animatable Categories with Level of Dynamics

**arXiv ID:** 2604.08547 | [PDF](https://arxiv.org/pdf/2604.08547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 596. Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics

**arXiv ID:** 2604.08503 | [PDF](https://arxiv.org/pdf/2604.08503v1)

**作者:** Ying Shen `[一作]` (University of Illinois Urbana Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种物理注入的视频生成框架（Physics-Infused Video Generation），通过双分支流匹配模型同时预测未来视频帧和潜在物理动力学，从而实现视觉内容与物理一致性的统一生成。

**💡 创新点**

创新点包括：①利用自监督视频编码器 V-JEPA2 提取的物理感知嵌入作为潜在物理状态；②在预训练的视觉扩散分支上增设物理分支，并通过双向交叉注意力实现视觉与物理信息的动态交互；③采用冻结视觉分支、递归损失权重调度的训练策略，使模型在不牺牲视觉质量的前提下学习物理规律；④支持多帧条件，扩展了文本/图像到视频的生成范式。

**🔧 技术方法**

核心技术包括：双分支 Conditional Flow Matching（流匹配）框架；VAE 视频编码器 + V-JEPA2 物理编码器；双向交叉注意力（Vis-Attention/Phy-Attention）；冻结预训练参数 + 递归损失权重调度；多帧条件扩展与时间步一致的流匹配损失。

**📊 数据集**

训练数据：OpenVidHD-0.4M（约 400K 高分辨率视频–文本对）；评估数据：VBench‑2、VideoPhy、VideoPhy‑2、Physics‑IQ 等物理感知与通用视频生成基准。

**📈 对比分析**

与 Wan2.2‑TI2V、CogvideoX、HunyuanVideo 等主流视频生成模型对比。物理一致性指标显著提升：VideoPhy 上 PC 提升 50.4%（最高 37.9 分），VideoPhy‑2 上 PC 提升 2.6%，Physics‑IQ 单帧设置提升 33.9%。在 VBench‑2 的物理维度亦大幅优于基线，整体得分与 Wan2.2‑TI2V 相当甚至略高，说明在保持视觉质量的同时大幅提升了物理合理性。

**⚠️ 局限性**

局限性：①训练过程中物理分支的梯度不稳定，需要递归调度损失权重；②物理一致性提升伴随多样性下降；③模型仅学习潜在物理嵌入，缺乏可解释的物理定律；④对极端或未见物理场景的泛化仍受限；⑤依赖大规模预训练视觉模型，算力与资源消耗较高。

---

## 597. Quantifying Explanation Consistency: The C-Score Metric for CAM-Based Explainability in Medical Image Classification

**arXiv ID:** 2604.08502 | [PDF](https://arxiv.org/pdf/2604.08502v1)

**作者:** Kabilan Elangovan `[一作]` (Singapore Health Services), Daniel Ting `[通讯]` (Singapore Health Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了一种新的无标注、基于置信度加权的CAM一致性评分（C‑Score），用于衡量医学影像分类模型在同一疾病类别下的解释一致性。

**💡 创新点**

C‑Score首次以自适应金标准列表和强度强调的软IoU对齐，量化同类样本间的解释可重复性，填补了现有解释评估缺失的“类内一致性”维度。

**🔧 技术方法**

采用GradCAM、GradCAM++、LayerCAM、EigenCAM、ScoreCAM和MS‑GradCAM++六种CAM方法，并结合DenseNet201、InceptionV3、ResNet50V2三种CNN架构在训练周期内进行评估。

**📊 数据集**

使用Kermany胸部X光图像数据集（5,856张图像，二分类：正常/肺炎）。

**📈 对比分析**

C‑Score在每个训练检查点持续计算，揭示了AUC与一致性之间的三种脱节机制，并在ResNet50V2上提前一轮训练检测到ScoreCAM一致性崩溃；与传统AUC/准确率相比，C‑Score提供了更早、更细粒度的模型性能监控。

**⚠️ 局限性**

局限包括：仅适用于2D卷积热图，受目标层深度影响；阈值选择决定金标准列表；仅在单一二分类胸X光任务上验证，尚未推广到多类别或其他影像模态；一致性高不必然代表临床正确，需要与放射学标注进一步对照。

---

## 598. Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction

**arXiv ID:** 2604.08542 | [PDF](https://arxiv.org/pdf/2604.08542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 599. Novel View Synthesis as Video Completion

**arXiv ID:** 2604.08500 | [PDF](https://arxiv.org/pdf/2604.08500v1)

**作者:** Qi Wu `[一作]` (Carnegie Mellon University), Deva Ramanan `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出将预训练的视频扩散模型轻量化改造为稀疏视角新视图合成模型（FrameCrafter），通过对输入视角进行独立VAE编码、投影相机几何（Plücker射线图）注入、去除时间位置信息，并使用LoRA微调，实现对未知相机姿态的高质量渲染。

**💡 创新点**

创新点在于：① 将视频模型的时序约束解除，使其对视角集合保持排列不变性；② 通过对每帧单独编码和像素unshuffle保留细粒度几何信息；③ 用投影相机几何直接在潜在空间注入相机信息；④ 只对Transformer的LoRA和patch embedding进行训练，实现极低参数更新。

**🔧 技术方法**

核心技术包括：预训练的视频扩散框架（如Wan2.1-I2V-14B、CogVideoX-5B）、VAE潜在编码、Diffusion Transformer (DiT)、LoRA低秩适配、Plücker射线映射、像素unshuffle、RoPE去除。

**📊 数据集**

使用DL3DV-10K子集（约1K场景）进行微调，同时在DL3DV-Benchmark和Mip-NeRF 360两大稀疏视角数据集上进行评估。

**📈 对比分析**

与基于图像扩散的EscherNet、SEVA和基于视频的Aether进行对比，FrameCrafter在PSNR、SSIM、LPIPS、DreamSim等指标上均优于或相当于使用8~80K训练样本的对手；并且模型随视频骨干的升级（从CogVideoX-5B到Wan2.1-14B）呈现显著性能提升，展示出可扩展性。

**⚠️ 局限性**

局限性包括：仍需依赖大规模预训练的视频模型，对极端稀疏（仅1-2帧）或极端视角变化的场景表现尚不充分；缺乏显式3D重建，对几何精度的评估受限；在某些指标（如最小化输入视角顺序影响）上虽已做优化，但对输入噪声与遮挡的鲁棒性尚需进一步验证。

---

## 600. When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models

**arXiv ID:** 2604.08546 | [PDF](https://arxiv.org/pdf/2604.08546v1)

**作者:** Zhengyang Sun `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练‑free 的 NUMINA 框架，通过识别并修正文本中的数词与视频中实例数量的偏差，利用 DiT 模型的注意力头构建可计数的布局，在生成过程中进行保守的布局修正和布局引导，从而提升文本到视频的数值对齐。

**💡 创新点**

创新点包括：1) 利用自注意力和交叉注意力中的实例分离头进行动态选择；2) 通过自注意力与交叉注意力融合得到可计数布局；3) 在第二阶段采用结构化成本函数进行布局修正并以修改交叉注意力的 bias/预激活来引导生成；4) 全程无需额外训练，兼容任意 T2V 模型。

**🔧 技术方法**

主要技术：Diffusion Transformer（DiT）架构；多头自注意力与交叉注意力头选择；PCA 与三维评分指标挑选实例分离头；聚类与阈值化生成可计数布局；布局修正成本函数（重叠、中心、时间三项）；带权重的交叉注意力 bias/预激活调整；EasyCache 加速方案。

**📊 数据集**

使用了 CountBench 基准（210 条含 1–8 个对象、1–3 类别的 prompt）以及 Wan T2V 系列模型的训练数据；评估时使用 GroundingDINO 提取实例计数。

**📈 对比分析**

与原始 Wan 模型、种子搜索、提示增强等方法对比。NUMINA 在 Wan2.1‑1.3B 上计数准确率提升至 49.7%（+7.4%），在 5B 和 14B 上分别提升 4.9% 与 5.5%。CLIP 得分提高，Temporal Consistency 也保持或提升。相比传统方法，NUMINA 在单一种子、单一 prompt 下即可实现显著性能提升，无需多次实验。

**⚠️ 局限性**

局限性：无法在所有场景下实现完全准确计数；对极高密度实例（数十、数百个）尚未验证；在极大实例数量或复杂场景下的布局修正仍需进一步改进。

---

## 601. Act Wisely: Cultivating Meta-Cognitive Tool Use in Agentic Multimodal Models

**arXiv ID:** 2604.08545 | [PDF](https://arxiv.org/pdf/2604.08545v1)

**作者:** Shilin Yan `[一作]` (Alibaba Group), Yixiong Zou `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种新型多模态智能体训练框架，专注于在工具使用与内部推理之间进行自适应决策，以提升推理精度并显著降低工具调用次数。

**💡 创新点**

创新点在于将工具效率与任务准确度解耦成两个独立的奖励通道，并通过条件优势估计仅对已正确回答的轨迹施加效率惩罚，从而消除传统标量化奖励导致的梯度干扰与信用分配不清的问题。

**🔧 技术方法**

采用的技术包括：Hierarchical Decoupled Policy Optimization (HDPO) 机制、GRPO风格的裁剪优势函数、基于内部判别器的多维度元认知过滤、以及Qwen3‑VL‑8B‑Instruct大模型的SFT+RL双阶段训练。

**📊 数据集**

训练数据来源于公开的工具增强多模态轨迹集合（DeepEyesV2、V‑Interaction、Thyme 等）与自研的高质量筛选集，RL阶段精选约5k个任务样本，覆盖感知、搜索与数学推理三大领域。

**📈 对比分析**

与现有的开源非工具模型、文本推理模型和其它多模态代理进行对比，实验显示该方法在 HRBench‑4K/8K、CharXiv、MathVista 等基准上均取得领先或接近最优成绩，同时工具调用率从 98% 降至约 2% 以上，显著提升系统效率。

**⚠️ 局限性**

局限性包括：对工具可用性高度依赖（若工具接口失效或延迟明显，策略仍可能受限）、需要较大计算资源进行双阶段训练以及对超参数（如效率权重 w_tool）的敏感性。

---

## 602. Self-Improving 4D Perception via Self-Distillation

**arXiv ID:** 2604.08532 | [PDF](https://arxiv.org/pdf/2604.08532v1)

**作者:** Nan Huang `[一作]` (University of Illinois Urbana-Champaign), Qianqian Wang `[通讯]` (Impossible Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用自蒸馏的自我提升框架，能够在无标注视频数据上持续改进预训练的 4D 视图重建模型。

**💡 创新点**

核心创新在于通过空间-时间上下文不对称（如随机帧丢弃）生成教师伪目标，实现无监督的自蒸馏循环；并系统评估多种设计取舍（教师更新策略、冻结模块、特征匹配等）。

**🔧 技术方法**

采用在线 EMA 教师、随机帧丢弃、输出层自蒸馏、冻结相机解码器等技术，并在多个基准模型（VGGT、π³）上实现。

**📊 数据集**

使用多样化数据集：OmniWorld‑Game、BEDLAM2.0、DROID、HOI4D 以及传统评测集（Sintel、KITTI、Bonn、RealEstate10K）进行训练与评估。

**📈 对比分析**

相较于预训练基线和监督微调，Self‑Evo 在目标域视频深度提升最高 36.5% 绝对误差、相机估计提升 20.1%，且在原始域保持甚至略有提升；在 OOD 场景（DROID、HOI4D）亦优于监督微调。

**⚠️ 局限性**

局限性包括：对相机运动需求较高，静态相机难以构造有效的不对称；长时间训练可能导致模型崩溃；现有策略仅在帧级别丢弃，缺乏更细粒度的 token 层面不对称。

---

## 603. PSI: Shared State as the Missing Layer for Coherent AI-Generated Instruments in Personal AI Agents

**arXiv ID:** 2604.08529 | [PDF](https://arxiv.org/pdf/2604.08529v1)

**作者:** Zhiyuan Wang `[一作]` (University of Virginia), Laura E. Barnes `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 PSI（共享状态）架构，将 AI 生成的个人模块转化为可持续、互联、聊天互补的仪器，并通过三周自传式部署验证其可行性。

**💡 创新点**

创新点在于引入共享个人上下文总线与统一供给契约，使独立生成的模块在运行时可互操作；同时提供持久 GUI 与通用聊天代理的双模交互，解决了 AI 生成工具碎片化的问题。

**🔧 技术方法**

使用了 SwiftUI、Python 桥接、REST + WebSocket、LLM 聊天代理 Facai、共享状态 Bus、统一接口协议（Swift 协议）等技术。

**📊 数据集**

使用了单用户三周收集的个人感知数据（心率、运动、睡眠等）以及人工合成的 50 条查询，并以 Claude Opus 4.6 作为评估判定工具。

**📈 对比分析**

通过与仅搜索、单模块三种条件对比，评估 50 条推理任务和 20 条写回任务，发现共享上下文平均满足率 0.88、任务成功率 0.68、写回成功率 95%；搜索仅 0.63/0.32/40%，单模块 0.27/0.08/95%。

**⚠️ 局限性**

局限性包括仅在单一技术熟练用户的三周试验中验证，模块数量有限，存在上下文污染、可扩展性与隐私授权等问题。

---

## 604. FIT: A Large-Scale Dataset for Fit-Aware Virtual Try-On

**arXiv ID:** 2604.08526 | [PDF](https://arxiv.org/pdf/2604.08526v1)

**作者:** Johanna Karras `[一作]` (University of Washington), Ira Kemelmacher-Shlizerman `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于物理仿真与重纹理技术构建FIT大规模尺寸标注的虚拟试穿数据集，并训练Fit‑VTO模型实现尺寸感知的试穿合成。

**💡 创新点**

首次提供精确度量（尺寸）标注的VTO数据集以及完整的合成-重纹理管线，能够显式控制衣物与人体尺寸，弥补现有数据缺失“非合身”样本的不足。

**🔧 技术方法**

使用GarmentCode进行3D衣物物理仿真、基于Flux.1-dev的Diffusion重纹理与生成、LoRA微调、测量嵌入（Fourier嵌入）以及条件化的Diffusion模型。

**📊 数据集**

FIT数据集（1.13M训练样本+1k测试样本），覆盖男女上衣、XS-3XL尺寸，含人体与衣物精确测量；另增约33万条在线时尚图像作增强。

**📈 对比分析**

在VITON‑HD与FIT测试集上与Any2AnyTryon、Nano Banana Pro、COTTON、IDM‑VTON等基线进行SSIM、FID、LPIPS、IoU等指标对比，Fit‑VTO在尺寸感知IoU最高且整体图像质量优于基线。

**⚠️ 局限性**

仅覆盖上身、标准正面姿态、简单结构上衣；对“紧度”感知不足；尺寸控制受测量间相关性限制；未扩展至下装、全身或复杂姿态与视角。

---

## 605. OpenVLThinkerV2: A Generalist Multimodal Reasoning Model for Multi-domain Visual Tasks

**arXiv ID:** 2604.08539 | [PDF](https://arxiv.org/pdf/2604.08539v1)

**作者:** Wenbo Hu `[一作]` (University of California Los Angeles), Kai-Wei Chang `[通讯]` (University of California Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多模态 LLM 的 RL 后训练，提出了 Gaussian GRPO 与任务级响应长度、熵塑形两种机制，实现了跨视觉任务的优势分布统一与感知-推理平衡。

**💡 创新点**

创新点在于：①将优势分布通过 1D 最优传输映射为标准正态，消除极端奖励波动并实现任务间梯度公平；②在 RL 训练中加入响应长度与熵的任务级塑形，动态调节推理深度与探索幅度。

**🔧 技术方法**

核心技术包括：Group Relative Policy Optimization (GRPO) 的改进、1D Optimal Transport 进行分布匹配、梯度裁剪、任务级长度奖励与熵正则化。

**📊 数据集**

使用 Qwen3-VL‑Instruct‑8B 作为预训练基座，Fine‑tune 于 OneThinker‑600k 数据集，评估 18 个视觉与文本混合基准（MMMU、MathVista、RefCOCO、OCRBench 等）。

**📈 对比分析**

与 GPT‑4o、Gemini 2.5 Pro、OneThinker‑8B 等模型对比，模型在 18 个基准上均达或超过对手，尤其在 MMMU、MathVista、RefCOCO 等任务上取得最高分；相比基线 GRPO 与 GDPO，Gaussian GRPO + 长度/熵塑形提升显著。

**⚠️ 局限性**

局限性：需要手动设置长度与熵阈值，缺乏自动化超参搜索；RL 训练仍耗费大规模算力；模型在极端 OOD 视觉场景或需要细粒度外部工具的任务上表现仍有限。

---

## 606. ParseBench: A Document Parsing Benchmark for AI Agents

**arXiv ID:** 2604.08538 | [PDF](https://arxiv.org/pdf/2604.08538v1)

**作者:** Boyang Zhang `[一作]` (runllama ai), Simon Suo `[通讯]` (runllama ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ParseBench——一个面向 AI 代理的企业文档解析基准，聚焦语义正确性。

**💡 创新点**

创新点在于：①将解析质量拆分为表格、图表、内容真实性、语义格式化和视觉定位五个维度；②采用规则驱动的二进制评测和自定义指标（TableRecordMatch、ChartDataPointMatch 等），真正衡量对决策至关重要的语义结构；③使用约 2,000 页人类验证的真实企业文档，覆盖保险、金融、政府等行业。

**🔧 技术方法**

使用多种技术：自定义评测框架、VLM 生成先验标注、人工复核循环、规则引擎（文本一致性、读序、格式化、定位/归因）以及多模型推理（VLM、专用解析器、LlamaParse）。

**📊 数据集**

数据集：约 2,000 页、1,180 篇公开企业文档（保险、金融、政府等），共 169,011 条评测规则，含表格、图表、文本、格式化和定位标签。

**📈 对比分析**

对 14 种方法（VLM、专用解析器、LlamaParse）进行对比：LlamaParse Agentic 以最高整体得分领先；不同模型在各维度表现不均，VLM 在内容层面强，专用解析器在结构/定位方面更佳，但无方法在所有维度均表现卓越。

**⚠️ 局限性**

局限性：①仅涵盖 5 个维度，未覆盖全部可能的解析失败；②评测仍以规则为主，某些细粒度语义误差难以捕捉；③数据集覆盖面虽广，但仍偏向保险/金融等行业，其他行业及更复杂文档尚待扩展。

---

## 607. ActiveGlasses: Learning Manipulation with Active Vision from Ego-centric Human Demonstration

**arXiv ID:** 2604.08534 | [PDF](https://arxiv.org/pdf/2604.08534v1)

**作者:** Yanwen Zou `[一作]` (Shanghai Jiao Tong University), Cewu Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过配戴立体摄像头的智能眼镜，让操作者裸手完成任务，并使用该设备在机器人上以主动视觉执行零迁移的操控学习。

**💡 创新点**

①裸手数据采集与主动视角同步；②仅使用单摄像头的主动视觉；③采用对象中心的3D点云策略，预测物体轨迹与头部运动；④实现零迁移跨平台。

**🔧 技术方法**

智能眼镜+ZED Mini立体摄像头、SLAM头部追踪、深度估计（FoundationStereo）、物体与手部分割（SAM2、Grounded‑SAM）、姿态估计（FoundationPose）、点云处理、扩散模型双头（分别预测物体轨迹与头部运动）。

**📊 数据集**

收集的自人体裸手演示数据（约200/100/100个示例对应书本摆放、吐司插入、远距离倒水），无公开公共数据集，全部在实验室场景中自行生成。

**📈 对比分析**

与无主动视角版本及Pi0.5基线对比，ActiveGlasses在三项任务中成功率提升35%、25%和30%；且在跨平台（Flexiv Rizon4、UR5）上保持高成功率，证明主动视觉和对象中心策略的有效性。

**⚠️ 局限性**

对人体头部与机器人感知臂的相对尺度差异导致绝对轨迹预测会出现边界靠近工作空间限制，导致IK失败；相对轨迹在高遮挡或物体尺寸小的场景下易失效；且系统依赖特定硬件（XREAL Air、ZED Mini）和点云处理，扩展性受限。

---

## 608. Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models

**arXiv ID:** 2604.08527 | [PDF](https://arxiv.org/pdf/2604.08527v1)

**作者:** Feng Luo `[一作]` (Rice University), Vladimir Braverman `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并解决了大语言模型在 on‑policy distillation (OPD) 训练中出现的长度膨胀和重复饱和问题，并提出了结合混合蒸馏和 KL 正则化的稳定 OPD 框架。

**💡 创新点**

创新点在于识别并阐释了 OPD 的自我强化重复爆炸机制，并通过混合蒸馏与参考方差约束双重策略抑制长度膨胀，实现了训练稳定性提升。

**🔧 技术方法**

使用的技术包括：GRPO 风格的剪辑目标、基于逆 KL 的奖励、混合蒸馏（on‑policy + off‑policy 黄金数据）以及对参考策略的 KL 正则化。

**📊 数据集**

实验数据集包括 OpenR1-Math-220k（训练），以及六个数学推理基准：AIME 2024/2025、AMC、Minerva、OlympiadBench、MATH500。

**📈 对比分析**

与 SFT、GRPO、SimpleRL‑Zero 等基线对比，稳定 OPD 在 1.5B 与 7B 规模模型上平均提升约 7.2% 以上（最高 47.6%），超过了现有 RLVR 与 OPD 方法。

**⚠️ 局限性**

局限性在于仍依赖大量黄金数据和教师模型，KL 正则的权重调优复杂；仅在数学推理任务验证，效果在其他生成任务或更大模型规模下尚未充分验证。

---

## 609. ETCH-X: Robustify Expressive Body Fitting to Clothed Humans with Composable Datasets

**arXiv ID:** 2604.08548 | [PDF](https://arxiv.org/pdf/2604.08548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 610. What do Language Models Learn and When? The Implicit Curriculum Hypothesis

**arXiv ID:** 2604.08510 | [PDF](https://arxiv.org/pdf/2604.08510v1)

**作者:** Emmy Liu `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大型语言模型预训练过程中的技能出现顺序，并验证隐式课程假设。

**💡 创新点**

提出隐式课程假设并通过一套91个可评估的元素与合成任务，证明不同模型在绝对阈值下的技能出现顺序高度一致，同时揭示任务内部表示与学习轨迹的可预测性。

**🔧 技术方法**

使用功能向量（function vectors）提取任务相关表示，并利用核岭回归预测未评估任务的学习轨迹；此外采用注意力头的因果影响（CIE）进行头级提取。

**📊 数据集**

使用91个元素和合成任务（包括字符串处理、形态转换、检索、核心ference、逻辑推理、数学等），并在公开的预训练模型（OLMo‑2/3、LLM360、Pythia）各自前1T token的检查点上进行评估。

**📈 对比分析**

通过Spearman相关系数评估不同模型间出现顺序的稳定性，平均ρ≈0.81；利用功能向量空间的核岭回归预测未评估合成任务的轨迹，R²范围为0.68–0.84，MAE在0.068–0.195之间，表明内部表示可有效预测学习动态。

**⚠️ 局限性**

局限包括：仅在绝对阈值下验证出现顺序，使用相对阈值时一致性下降；任务设计为人工合成，可能不覆盖真实世界复杂任务；只评估公开模型的检查点，未涵盖更大规模或不同预训练策略的模型。

---

## 611. Visually-grounded Humanoid Agents

**arXiv ID:** 2604.08509 | [PDF](https://arxiv.org/pdf/2604.08509v1)

**作者:** Hang Ye `[一作]` (Peking University), Yizhou Wang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 Visually-grounded Humanoid Agents 双层框架，从单目视频重建高保真 3D 场景并让数字人通过自我感知、规划与运动实现自主行为。

**💡 创新点**

创新点在于：①遮挡感知的语义场景重建；②空间感知视觉提示与迭代推理相结合的高层规划；③基于扩散模型的低层运动生成，形成端到端的感知-决策-动作闭环。

**🔧 技术方法**

技术手段包括 3D Gaussian Splatting、SAM+SAM+对比学习、VLM（如 Qwen2.5-VL）、扩散运动模型、Spatial-Aware Visual Prompting 与 Iterative Reasoning。

**📊 数据集**

使用数据集包括公开城市视频（SmallCity、XGRIDS、SAGE-3D）、人类视频（PeopleSnapshot、GaussianAvatar 等）以及自建的语义 3DGS 环境。

**📈 对比分析**

与 NaVILA、NaVid、Uni-NaVid 等 VLN 基线以及 Feature-3DGS、OpenGaussian 等语义重建基线对比，成功率提升约 30%（SPL 同样显著提升），碰撞率保持或下降。

**⚠️ 局限性**

局限性包括对高度动态场景的鲁棒性仍有限；对不同姿态或速度的适应性需进一步提升；对更复杂交互（对话、物理互动）的支持尚未实现。

---

