# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-24 | 今日论文总数: 883

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis

**arXiv ID:** 2602.19585 | [PDF](https://arxiv.org/pdf/2602.19585v1)

**作者:** Chunlei Meng `[一作]` (Fudan University), Chun Ouyang `[通讯]` (Fudan University)

**通讯引用:** 5845 | [OpenAlex ID](https://openalex.org/A5075868200)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出三子空间解耦框架TSD，用来捕捉多模态情感分析中的全局共享、子模态共享与私有特征；

**💡 创新点**

创新点在于显式区分并学习三种子空间，并通过解耦监督与结构化正则化保持子空间纯净，同时设计子空间感知交叉注意力（SACA）实现自适应融合；

**🔧 技术方法**

使用BERT、时序卷积网络进行模态编码；三子空间编码器（共享、子共享、私有）和解耦监督；SACA融合模块；多项正则化（一致性、对偶一致性、HSIC、正交性、监督）；

**📊 数据集**

评估数据集包括CMU-MOSI、CMU-MOSEI（情感回归/分类）以及MIntRec（多模态意图识别）；

**📈 对比分析**

与多种早期融合、注意力融合和对比学习等基线对比，TSD在MAE、Acc-2、Acc-7、F1等指标上均超越最新方法，取得0.691 MAE/54.9% ACC_7在CMU-MOSI以及0.535 MAE/54.6% ACC_7在CMU-MOSEI，并在MIntRec上实现73.67%准确率；

**⚠️ 局限性**

局限在于需要手动设定子空间维度和正则化系数，对超参数敏感；同时三子空间结构较为复杂，训练时对资源和时间要求较高；

---

## 2. Dual-Kernel Adapter: Expanding Spatial Horizons for Data-Constrained Medical Image Analysis

**arXiv ID:** 2602.18888 | [PDF](https://arxiv.org/pdf/2602.18888v1)

**作者:** Ziquan Zhu `[一作]` (University of Leicester), Tianjin Huang `[通讯]` (University of Exeter)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5028180352)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在极低数据量（≤1%）的医学图像分类与分割任务中，对现有适配器（Adapter）进行系统评估，并发现其在低数据场景下会退化；随后提出Dual‑Kernel Adapter（DKA）来改进该性能。

**💡 创新点**

DKA通过在同一模块中并行使用大卷积核（51×51）和小卷积核（5×5）两条深度可分离卷积分支，既扩展有效感受野（ERF），又保持局部细节，解决了传统适配器在低数据下缺乏长程依赖的缺陷。

**🔧 技术方法**

采用深度可分离卷积、线性降/升投影、GELU激活以及残差连接；训练时对适配器和分类/分割头使用不同学习率（异步学习率）以提升收敛效果。

**📊 数据集**

使用COVID、BUSI、ISIC‑2019三大医学分类数据集以及BRATS、BUSI、ISIC‑2018三大医学分割数据集；预训练骨干分别为ViT‑B、Swin‑B、Segmenter‑B、RadImageNet‑ResNet‑50和MedSAM。

**📈 对比分析**

与线性探测、全微调、BitFit、Prompt、LoRA以及多种适配器变体（Adapter、AdapterFormer、Convpass、CIAT、AIM）进行对比；在0.63%、1.25%和100%数据量下，DKA均明显优于所有基线，尤其在低数据条件下常优于全微调与线性探测。

**⚠️ 局限性**

仅在医学图像分类/分割任务上验证；对极低数据量的实验仍受限于样本量与模型规模；大卷积核对跨通道信息建模有限，且超参数（核尺寸、降维维度）需手动调优，可能不适用于所有医学任务。

---

## 3. How Far Can We Go with Pixels Alone? A Pilot Study on Screen-Only Navigation in Commercial 3D ARPGs

**arXiv ID:** 2602.18981 | [PDF](https://arxiv.org/pdf/2602.18981v1)

**作者:** Kaijie Xu `[一作]` (McGill University), Clark Verbrugge `[通讯]` (McGill University)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5001343030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并实现了一个仅使用屏幕像素和视觉占位符的导航代理，能够在现实3D ARPG关卡中从视觉STP/MSTP预测中驱动摄像机与前进动作，完成预设的视觉里程碑序列；

**💡 创新点**

创新点在于：①将已有的单帧视觉占位符检测（STP/MSTP）直接嵌入至完整的有限状态控制器中，实现纯视觉的端到端导航；②提出基于设计师指定视觉里程碑的完全屏幕化评估协议，提供可复现的基线；③在三种代理配置（Naive、FSM、Full）中探索视觉记忆的效果。

**🔧 技术方法**

技术手段包括：单帧STP检测（Faster‑R‑CNN + 选择器）、MSTP选择、有限状态机（Scan/Align/Advance/Recover等）、脉冲式摄像机控制、视觉记忆银行（pHash+嵌入近邻检索）以及模板匹配的里程碑检测。

**📊 数据集**

实验数据来自四款商业游戏（Dark Souls I、Dark Souls III、Elden Ring、Black Myth: Wukong）的多条手工指定路段，每条路段拆分为6个视觉里程碑，采用游戏内屏幕截图与手动标注的STP/MSTP训练集。

**📈 对比分析**

对比三种代理，核心游戏中FSM往往在某些段落提升成功率（如DS III Grand Archives中M₃段从20%提升到60%），但整体路段成功率仅上升至50%；在跨游戏迁移时性能显著下降（Black Myth 0%/40%/20%，Elden Ring始终0%）。表格显示Naive最低、Full偶有改善但不稳定。

**⚠️ 局限性**

局限性包括：①依赖单帧占位符模型，无法处理垂直结构、跳跃等物理交互；②视觉记忆仅对已检测到的STP进行惩罚，无法生成新的导航候选；③里程碑评估忽略玩家探索与多目标行为；④实验规模有限，未覆盖更多游戏与路段，结果不具广泛统计意义。

---

## 4. Smooth Gate Functions for Soft Advantage Policy Optimization

**arXiv ID:** 2602.19345 | [PDF](https://arxiv.org/pdf/2602.19345v1)

**作者:** Egor Denisov `[一作]` (Lomonosov Moscow State University), Roman Ischenko `[通讯]` (Institute for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Soft Adaptive Policy Optimization中门函数的结构化分析框架，定义了可接受门函数的梯度特性，并对sigmoid、误差函数、反正切、softsign等多种门函数在训练稳定性与探索性上的影响进行理论与实验研究。

**💡 创新点**

创新点在于把门函数的可接受性拆解为四条梯度条件（连续可微、梯度最大值为1且在1处、单调衰减、趋于零），并基于这些条件设计多种满足条件的门函数，系统阐明不同梯度衰减（指数、幂律、Gaussian）对RL训练的探索‑稳定性权衡。

**🔧 技术方法**

采用GRPO的Soft Adaptive Policy Optimization框架，使用平滑门函数替代硬裁剪，并引入温度参数；在Qwen2.5‑7B‑Instruct上通过多轮rollout、梯度累积、并行GPU训练实现强化学习对齐。

**📊 数据集**

训练集使用GSM8K与DeepMath混合；评估集包括GSM8K、MATH500、AIME24、AIME25四个数学推理基准。

**📈 对比分析**

通过对不同门函数的梯度形状、熵演化、重要性采样比率方差等指标进行对比实验；实验显示多项式衰减门产生更宽的有效梯度区域，Gaussian衰减门更集中，表明门函数形状直接影响训练动态；目前尚未给出完整的准确率对比，后续版本将公布量化结果。

**⚠️ 局限性**

局限性包括：实验仅限于数学推理任务，缺乏大规模多任务评估；目前仅给出梯度形状与训练动态的初步分析，缺少完整性能数值；门函数在不同任务与模型规模上的泛化能力尚未充分验证。

---

## 5. Evaluating Large Language Models on Quantum Mechanics: A Comparative Study Across Diverse Models and Tasks

**arXiv ID:** 2602.19006 | [PDF](https://arxiv.org/pdf/2602.19006v1)

**作者:** S. K. Rithvik `[一作]` (Physical Research Laboratory), S. K. Rithvik `[通讯]` (Physical Research Laboratory)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5108812815)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

系统评估了15款大型语言模型在量子力学20道任务上的表现，涵盖推导、创意设计、非标准概念和数值计算，并公开了基准及验证工具。

**💡 创新点**

提出了量子力学专属基准、三阶能力层级的性能分层、工具增强效果的细粒度分析以及多次运行的可复现性评估。

**🔧 技术方法**

采用大模型推理、Python代码执行（NumPy/SciPy）以及自动化验证器的组合进行评测。

**📊 数据集**

使用了由20道多项选择题组成的自定义量子力学数据集，涵盖符号推导、设计优化、前沿概念与数值实验，数据与验证器已公开。

**📈 对比分析**

通过对15模型、3个能力层级、3轮无随机性的评测，发现旗舰模型平均准确率81%，中层77%，低速快模67%，工具增强在数值任务上总体提升约4.4个百分点但代价三倍，模型间差异显著。

**⚠️ 局限性**

局限包括数值任务仍难以提升、工具使用对部分问题造成下降、模型可复现性受限于离散评分，且对更广阔量子领域的覆盖仍不足。

---

## 6. Non-Interfering Weight Fields: Treating Model Parameters as a Continuously Extensible Function

**arXiv ID:** 2602.18628 | [PDF](https://arxiv.org/pdf/2602.18628v1)

**作者:** Sarim Chaudhry `[一作]` `[通讯]`, Sarim Chaudhry

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种在冻结大型预训练模型的基础上，利用坐标动态和可学习权重场实现零遗忘的增量学习框架。

**💡 创新点**

创新点在于引入非干扰权重场，通过GRU产生64维坐标来指挥多基适配器的门控，并通过高斯拟合与锚点快照锁定已学习的参数区域，避免新任务对旧知识的破坏。

**🔧 技术方法**

采用的技术包括冻结Mistral‑7B backbone、GRU坐标动态、三层MLP权重场、top‑k softmax门控、适配器银行、锁定损失、分离损失、稀疏预算损失等。

**📊 数据集**

在多种通用语言建模和连续任务数据集上进行评估，主要使用WikiText、PG‑19、WebText等公开文本集。

**📈 对比分析**

与LoRA、Adapter、Weight‑Mixing等传统方法相比，本方法在保持原任务性能的同时，实现了零遗忘，并在下游任务上获得了与或略高于基线的效果。

**⚠️ 局限性**

限制包括需要额外的坐标与锚点设计、对高斯假设的依赖、适配器容量受限、训练过程较为复杂，以及对不同任务分布的泛化仍需验证。

---

## 7. Towards Personalized Multi-Modal MRI Synthesis across Heterogeneous Datasets

**arXiv ID:** 2602.19723 | [PDF](https://arxiv.org/pdf/2602.19723v1)

**作者:** Yue Zhang `[一作]` (University of Science and Technology of China), S. Kevin Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了PMM‑Synth，一种能在多个异构医学影像数据集上联合训练、支持任意输入‑输出配置的多模态MRI缺失模态合成框架

**💡 创新点**

创新点包括：1）个性化特征调制（PFM）根据数据集标识动态调节特征以缓解分布漂移；2）模态一致批量调度（MCBS）确保每个mini‑batch具有相同模态组合，提升训练效率；3）选择性监督损失在目标模态缺失时仅对可用真值进行监督

**🔧 技术方法**

采用统一的多模态生成网络作为骨干，并集成PFM、MCBS和选择性监督损失；利用自注意力/Transformer或GAN（如ResViT/Uni‑Synth）作为后端生成器与判别器

**📊 数据集**

四个异构数据集：TTG（脑瘤，6模态）、TTI（免疫相关脑病，4模态）、BraTS（脑瘤，4模态）和ISLES（卒中，3模态）

**📈 对比分析**

与六个单/多数据集基线（MM‑GAN、MM‑Synth、ResViT、pFLSynth、FTN、PMM‑Synth w/o PFM）在一对一和多对一合成任务上进行对比；PMM‑Synth在PSNR/SSIM上均领先，对比实验显示平均提升约1–2 dB；在下游肿瘤分割和放射学报告实验中也显著提高了Dice和诊断一致性

**⚠️ 局限性**

局限性：模型尚未支持持续适应新数据集或新模态的增量学习，面对完全新模态或协议时可能出现灾难性遗忘；需要进一步研究参数高效微调或连续学习策略

---

## 8. Partial Soft-Matching Distance for Neural Representational Comparison with Partial Unit Correspondence

**arXiv ID:** 2602.19331 | [PDF](https://arxiv.org/pdf/2602.19331v1)

**作者:** Chaitanya Kapoor `[一作]` (University of California), Meenakshi Khosla `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了部分软匹配距离（partial soft‑matching distance），用于神经网络和大脑功能图像的表示相似性比较。

**💡 创新点**

创新点在于允许部分神经元不匹配，从而在保持旋转敏感性的同时提升对噪声和离群值的鲁棒性，并在理论上放宽了质量守恒约束。

**🔧 技术方法**

采用部分最优传输框架、软匹配技术、神经元排序、交叉网络对齐以及卷积图像可视化等方法。

**📊 数据集**

使用仿真数据、噪声扰动的模型识别任务、fMRI 大脑功能影像以及深度卷积网络内部层的神经元。

**📈 对比分析**

与全匹配软匹配和暴力匹配等方法对比，部分软匹配在存在噪声和离群值时能保留正确匹配，fMRI中自动剔除低可靠体素，并在同源脑区实现更高对齐精度。

**⚠️ 局限性**

局限包括对匹配比例选择的敏感性、需要预先设定阈值或比例参数，以及在极端高噪声环境下仍可能出现误匹配。

---

## 9. Physics-Compliant Modeling and Optimization of MIMO Systems Aided by Microwave Linear Analog Computers

**arXiv ID:** 2602.19379 | [PDF](https://arxiv.org/pdf/2602.19379v1)

**作者:** Matteo Nerini `[一作]` (Imperial College London), Bruno Clerckx `[通讯]` (Imperial College London)

**通讯引用:** 16192 | [OpenAlex ID](https://openalex.org/A5070530952)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对毫米波线性模拟计算机（MiLAC）在多输入多输出系统中考虑天线互耦的物理合规建模，并给出了在不同部署场景（仅发射端、仅接收端或双端）下的端到端系统模型。

**💡 创新点**

创新点在于将多端口网络理论引入MiLAC，首次提供了包含互耦效应的闭式优化解，证明在互耦存在时MiLAC可实现与匹配网络相同的功率传输并优于无匹配网络的数字波束成形；并给出了互耦对性能提升的解析性上界。

**🔧 技术方法**

采用多端口网络理论、阻抗/导纳矩阵分析、线性代数优化（利用单位ary对称矩阵）以及闭式最优解推导。

**📊 数据集**

使用了基于Rayleigh衰落的模拟信道（独立同分布、单位路径增益）以及真实天线互耦矩阵（通过电磁仿真得到的导纳/阻抗矩阵）进行数值验证。

**📈 对比分析**

通过与数字波束成形（有匹配网络与无匹配网络）在相同信道条件下的理论上限和仿真结果进行对比，结果显示：MiLAC在有互耦时可与匹配网络数字系统等效，在无匹配网络时始终表现更优，且互耦强度越大优势越明显。

**⚠️ 局限性**

局限性包括：仅分析了单流（MISO）情形；对互耦的建模假设为理想化的多端口网络；未考虑非线性器件损耗、硬件实现中的可调元件限制及实际天线阵列尺寸对互耦的精确影响。

---

## 10. The Algorithmic Unconscious: Structural Mechanisms and Implicit Biases in Large Language Models

**arXiv ID:** 2602.18468 | [PDF](https://arxiv.org/pdf/2602.18468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 11. Detecting High-Potential SMEs with Heterogeneous Graph Neural Networks

**arXiv ID:** 2602.19591 | [PDF](https://arxiv.org/pdf/2602.19591v1)

**作者:** Yijiashun Qi `[一作]` (University of Michigan), Yijiazhen Qi `[通讯]` (The University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建包含公司、研究主题和政府机构的异质图，并使用 Heterogeneous Graph Transformer 预测 SBIR Phase I 获奖公司是否能进入 Phase II。

**💡 创新点**

① 仅使用公开 SBIR/STTR 数据构建完整异质图，避免对私有数据的依赖；② 采用 HGT 捕捉公司–主题、公司–机构以及公司间共同主题等多种关系；③ 采用严格的时间拆分评估，确保无信息泄露。

**🔧 技术方法**

异质图神经网络 Heterogeneous Graph Transformer（HGT）；对比 MLP 与 R‑GCN；实现基于 PyTorch Geometric。

**📊 数据集**

SBIR/STTR 奖项公开数据库，包含 32,268 家公司、124 个研究主题、13 个政府机构，约 99,000 条边。

**📈 对比分析**

在 AUPRC、AUROC、Precision@K、Lift@K 等指标上进行比较；SME‑HGT 在测试集 AUPRC 0.621（比 MLP 0.590 提升 3.1pp），Precision@100 约 89.6%，相较随机提升 2.14×，整体性能优于基线。

**⚠️ 局限性**

仅依赖 SBIR 数据，缺少专利、财务等补充信息；预测任务本身难度大，导致整体 AUROC 仅 0.60–0.65；实体解析采用规则匹配，可能出现错误；评估仅基于单一时间拆分，缺乏多窗口鲁棒性验证。

---

## 12. Runtime-Augmented LLMs for Crash Detection and Diagnosis in ML Notebooks

**arXiv ID:** 2602.18537 | [PDF](https://arxiv.org/pdf/2602.18537v1)

**作者:** Yiran Wang `[一作]` (Linköping University), Dániel Varró `[通讯]` (Linköping University)

**通讯引用:** 5949 | [OpenAlex ID](https://openalex.org/A5064890236)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于LLM的运行时增强方法CRANE-LLM，用来在Jupyter Notebook中提前检测和诊断代码单元崩溃。

**💡 创新点**

创新点在于：①将可执行内核状态（对象类型、张量形状、数据属性等）结构化为“运行时信息”并动态注入LLM提示；②实现对多种ML库和崩溃根因的通用检测与诊断；③通过系统化的消融实验验证不同类别运行时信息的贡献；④评估API文档注入对性能的影响。

**🔧 技术方法**

使用的大型语言模型包括 Gemini‑2.5‑Flash、GPT‑5 与 Qwen‑2.5‑Coder‑32B‑Instruct；通过构造包含静态代码、执行顺序及运行时摘要的提示，调用LLM完成“是否崩溃”与“崩溃原因”两任务。

**📊 数据集**

评估基准为 JunoBench，包含 111 对有缺陷与修复的 ML Notebook（共 222 个 Notebook），覆盖 TensorFlow/Keras、PyTorch、Scikit‑learn、NumPy、Pandas 等主流库。

**📈 对比分析**

对比方法为：在相同 Notebook 上分别使用有运行时信息（+RT）与无运行时信息（‑RT）的提示，计算准确率、召回率、F1 分数；对多模型进行平均；对诊断进行人工评估。结果显示：+RT 在所有模型上均提升 7–10% 的准确率、8–11% 的 F1 分数；当要求诊断时提升更显著；不同库、根因与信息类别的贡献差异也被量化。

**⚠️ 局限性**

局限性包括：①仅在 JunoBench 上验证，未覆盖工业级或特定领域的 Notebook；②运行时信息抽取采用固定模板，未针对不同 LLM 或任务自适应；③API 文档注入虽不提升性能但显著增加 token 费用；④LLM 输出受随机性影响，需多次重复；⑤人工诊断评估主观性与标注覆盖面有限。

---

## 13. Tower of Babel in Cross-Cultural Communication: A Case Study of #Give Me a Chinese Name# Dialogues During the "TikTok Refugees'' Event

**arXiv ID:** 2602.18549 | [PDF](https://arxiv.org/pdf/2602.18549v1)

**作者:** Jielin Feng `[一作]` (Fudan University), Siming Chen `[通讯]` (Fudan University)

**通讯引用:** 4212 | [OpenAlex ID](https://openalex.org/A5050391600)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了2025年TikTok难民事件中中国RedNote平台上外国新来者与中国用户之间通过赋予中文名字进行跨文化沟通的实践，构建了多渠道（语义、音韵、视觉）的命名策略框架；

**💡 创新点**

创新点在于提出“人类–人工智能协同”的抽取与生成流程，利用LLM批量化提取和补全命名解释，形成超过40种细粒度跨文化编码策略，并将其组合层次化为现代“巴别塔”；

**🔧 技术方法**

主要技术包括：大型语言模型（DeepSeek-V3、GPT‑4.1‑mini、Grok‑3‑mini、Gemini‑2.0 Flash、Qwen‑Plus）的多模型投票与一致性评估、prompt工程、语义聚类（BERTopic+HDBSCAN）、视觉描述生成（GPT‑4o‑mini）以及统计与计量分析（χ²、负二项回归）；

**📊 数据集**

数据集为70,614条RedNote评论，涵盖318条含中文名字请求的帖子，提取了62,461条命名对及其语义、音韵、视觉解释；

**📈 对比分析**

与传统人工标注相比，该人机协同方案在70,614条评论上的整体准确率达99.66%，人工作业仅占3.9%，相比纯人工耗时约784.6小时；

**⚠️ 局限性**

局限在于只分析单层回复、未获取完整对话结构；数据来源受RedNote API限制，缺少多级回复与更细粒度用户特征；以及对新兴网络语义与幽默的捕捉仍依赖人工校正，无法完全自动化。

---

## 14. Fairness-Aware Partial-label Domain Adaptation for Voice Classification of Parkinson's and ALS

**arXiv ID:** 2602.18535 | [PDF](https://arxiv.org/pdf/2602.18535v1)

**作者:** Arianna Francesconi `[一作]` (Ecole Polytechnique Federale de Lausanne), Mary-Anne Hartley `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并评估了一种用于跨域、跨队列的三分类（健康/帕金森/ALS）语音识别框架

**💡 创新点**

创新点是将风格混合域泛化、条件部分标签对抗对齐和性别无关对抗分支结合，解决部分标签不匹配和性别公平性问题

**🔧 技术方法**

使用了 MixStyle、Partial CDAN（条件对抗）、梯度反转层（GRL）实现对抗学习，基于 ResNet-18 提取特征

**📊 数据集**

使用了四个公开持续元音数据集：mPower、VOC-ALS、UAMS、Minsk

**📈 对比分析**

与12种最先进的机器学习/深度学习方法（SVM、XGBoost、ResNet-18、ECAPA–TDNN、Wav2Vec 2.0、MixStyle、CORAL、DANN、CDAN、Partial CDAN等）在内部交叉验证和外部测试下进行对比，FairPDA 在外部 MCC、BalAcc 方面表现最佳且性别公平性指标最低

**⚠️ 局限性**

局限在于数据量仍有限、只评估了二元性别、对源只存在类的处理不够充分、跨域性能仍有提升空间

---

## 15. CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval

**arXiv ID:** 2602.20083 | [PDF](https://arxiv.org/pdf/2602.20083v1)

**作者:** Xinzhao Li `[一作]` (Villanova University), Ruiyang Qin `[通讯]` (Villanova University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CQ-CiM，一套统一的硬件感知数据整形框架，用于将高精度高维句子嵌入压缩、量化后直接映射到不同 Compute-in-Memory（CiM）交叉栅结构上，从而实现低延迟、低功耗的 RAG（检索增强生成）系统。

**💡 创新点**

创新点在于：
- 同时对维度和精度进行联合学习的压缩‑量化框架；
- 通过 LoRA 适配器实现参数高效的嵌入模型微调；
- 在压缩后引入噪声注入模块，使模型对真实 CiM 器件的电导波动具有鲁棒性；
- 使用 N2UQ 量化头并配合对比学习+重建损失的自监督训练，显著提升低比特嵌入的检索性能。

**🔧 技术方法**

主要技术包括：LoRA 参数高效微调、密集投影压缩层、基于硬件噪声模型的噪声注入、非均匀到均匀量化（N2UQ）、对比学习损失（Contrastive）+重建 MSE 损失、以及自监督端到端训练。

**📊 数据集**

使用了 5 个检索基准数据集：ARCChallenge、NanoHotpotQA、CQADupStackGisRetrieval、ArguAna 以及一个 8bit PQ 作为高精度基准；嵌入模型以 all‑MiniLM‑L6‑v2 为主，维度压缩至 128D。

**📈 对比分析**

与 UMAP、PCA+均匀量化、Vanilla 截断、PQ‑8bit 等基线相比，在 1‑bit、1.58‑bit、2‑bit 低比特设置下，CQ‑CiM 在 Recall@5 和 nDCG@10 上普遍领先，尤其在最苛刻的 1‑bit 方案中提升 3–10% 以上；在 RAG 下引入设备噪声仍能保持高召回，显示出强鲁棒性。

**⚠️ 局限性**

局限性包括：
- 仍依赖软件模拟的噪声模型，未在真实硬件上完成大规模验证；
- 主要针对句子嵌入，扩展到更复杂多模态嵌入尚未验证；
- 对比学习需要大量未标注语料，可能受制于可用数据量。

---

## 16. Stable Deep Reinforcement Learning via Isotropic Gaussian Representations

**arXiv ID:** 2602.19373 | [PDF](https://arxiv.org/pdf/2602.19373v1)

**作者:** Ali Saheb `[一作]` (Mila), Pablo Samuel Castro `[通讯]` (Google DeepMind)

**通讯引用:** 2070 | [OpenAlex ID](https://openalex.org/A5068291173)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探讨在深度强化学习中，采用等方差高斯分布的表示（等方差高斯表示）可显著提升训练稳定性和性能，提出了轻量化的 Sketched Isotropic Gaussian Regularization (SIGReg) 作为正则化手段，减少表示崩塌、神经元休眠和梯度不稳定问题。

**💡 创新点**

将等方差高斯表示理论与实际 RL 训练相结合，证明在非平稳目标下等方差高斯表示能保证追踪误差稳定并收敛；同时引入 SIGReg 通过随机投影匹配单维高斯分布，既保持等方差又控制尾部，轻量且易实现。

**🔧 技术方法**

理论分析：线性追踪误差动力学、Lyapunov 稳定性、条件数影响；正则化技术：SIGReg（随机投影+特征函数匹配）；实验工具：PQN、PPO、Isaac Gym 以及 CIFAR‑10 等站点。

**📊 数据集**

实验数据集：CIFAR‑10（在标签置换协议下模拟非平稳监督）；Arcade Learning Environment（Atari 10 游戏子集及完整 57 游戏）；Isaac Gym 连续控制任务。

**📈 对比分析**

与基线（无正则化）、Kronecker‑factored 优化、multi‑skip 架构等方法对比；在 57 个 Atari 游戏中，SIGReg 改进 51 个，平均 AUC 提升 889%，中位数 138%；在连续控制任务中表现出更低方差和更高返回；在 PPO 中同样获得一致的 AUC 提升。

**⚠️ 局限性**

仅对表示的边缘分布进行约束，未考虑任务特定结构；假设线性读头与近似恒定协方差，未涵盖非线性头、完全耦合的离线 actor‑critic、多任务和持续学习场景；SIGReg 需对随机投影进行重采样，可能在高维场景下产生开销。

---

## 17. Computational Social Choice: Research & Development

**arXiv ID:** 2602.20074 | [PDF](https://arxiv.org/pdf/2602.20074v1)

**作者:** Dorothea Baumeister `[一作]` (Federal University of Applied Administrative Science), Arianna Novaro `[通讯]` (University Paris 1 Panthéon-Sorbonne)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文呼吁将计算社会选择（COMSOC）从以理论为主转向以实践为导向，提出了“COMSOC‑R&D”研究议程，并通过定义、价值论证、案例展示和障碍剖析，阐明了问题驱动、项目化、可部署的研究范式；

**💡 创新点**

创新点在于：①提出专门针对问题导向、工程化的COMSOC‑R&D框架；②明确了从理论到实战的转化路径；③给出了评审准则和实践案例，填补了COMSOC理论与真实应用之间的空白；

**🔧 技术方法**

主要技术包括：多胜者投票机制（如MES、Sortition、PoS等）、社会选择理论方法、软件工程实践（接口设计、可部署系统）以及案例研究方法；

**📊 数据集**

文章并未使用特定公开数据集，而是引用了真实应用场景的数据（如城市参与式预算投票数据、选举投票数据等），并鼓励未来研究收集新的偏好与投票数据；

**📈 对比分析**

在案例中，通过将MES等方法与传统贪心规则或现状进行对比，展示了更高的公平性和效率；但文章侧重概念验证与经验总结，缺乏统一的实验指标或大规模性能评估；

**⚠️ 局限性**

局限性包括：①缺乏统一的评估基准和公开数据集；②实践转化面临合作网络、激励机制、运维成本、可信度等多重障碍；③目前案例多为小规模或特定领域，难以直接推广至更广泛的应用场景。

---

## 18. A Green Learning Approach to LDCT Image Restoration

**arXiv ID:** 2602.19540 | [PDF](https://arxiv.org/pdf/2602.19540v1)

**作者:** Wei Wang `[一作]` (University of Southern California), C. -C. Jay Kuo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种绿色U形学习（GUSL）框架，用于低剂量CT图像的去噪恢复。

**💡 创新点**

通过多分辨率逐级残差估计、无监督表示学习、特征选择（RFT）和基于XGBoost的残差回归，实现了可解释性、低模型规模和低计算复杂度。

**🔧 技术方法**

利用PixelHop单元、相关特征测试（RFT）、统计特征生成（SFG）+LNT、XGBoost回归等技术。

**📊 数据集**

使用NIH‑AAPM‑Mayo Clinic 2016 LDCT Grand Challenge公开数据集（512×512的LDCT/NDCT对）。

**📈 对比分析**

与RED‑CNN、WGAN‑VGG、MAP‑NN、AD‑NET和CTformer等五种深度学习方法在PSNR、SSIM、参数量和MACs/像素上进行对比，GUSL仅次于CTformer，且模型参数0.57M、MACs0.03M，显著小于其他模型。

**⚠️ 局限性**

缺点是仍需在更大规模、多中心或不同设备上验证鲁棒性，且在极低剂量或高噪声情况下的性能可能不如某些高级GAN方法。

---

## 19. On scheduling coupled tasks with exact delays to minimize maximum lateness

**arXiv ID:** 2602.20010 | [PDF](https://arxiv.org/pdf/2602.20010v1)

**作者:** Wiesław Kubiak `[一作]` `[通讯]` (Memorial University), Wiesław Kubiak (Memorial University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在单机上对耦合任务（每个任务由前置操作、精确延迟p和后置操作组成）进行调度，目标是最小化最大迟到时间L_max的调度问题。

**💡 创新点**

主要创新在于首次为约束为可调序列的同意（agreeable）与相反（disagreeable）两类实例设计了多项式时间算法，并通过复杂度分析指出一般问题仍为开放。

**🔧 技术方法**

采用图模型构造、可行序列（semi‑feasible sequence）与交换论证相结合的技术，并利用二分搜索与最短路径求解来实现算法。

**📊 数据集**

该研究为纯理论分析，未使用公开实验数据集，所有结果均基于数学证明和理论实例。

**📈 对比分析**

与以往仅关注完工时间或总完成时间的研究相比，本文在两类特殊顺序下给出了多项式解法，展示了在这些情形下L_max可在多项式时间内最优求解。

**⚠️ 局限性**

限制在于算法仅适用于可调序列的同意与相反实例，对一般无序实例的复杂度仍未解决，且对大规模实例的实际效率尚未验证。

---

## 20. Monocular Mesh Recovery and Body Measurement of Female Saanen Goats

**arXiv ID:** 2602.19896 | [PDF](https://arxiv.org/pdf/2602.19896v1)

**作者:** Bo Jin `[一作]` (Northwest A&F University), Meili Wang `[通讯]` (Northwest A&F University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文建立了真实场景下的八视角RGBD摄像阵列，采集55只女性Saanen奶山羊的动态视频，并利用多视角DynamicFusion生成高精度3D扫描；在此基础上构建了专属的SaanenGoat参数化三维形状模型，实现单视角RGBD图像的高精度重建，并通过该模型自动测量六个关键体型指标。

**💡 创新点**

创新点包括：1）首次公开Saanen奶山羊的八视角同步RGBD数据集FemaleSaanenGoat；2）针对该品种定制的高分辨率骨骼模板与乳腺结构，构建具有41个关节的专属参数化模型；3）融合深度、轮廓、法向等三维几何约束和生物力学正则化的单视角重建方法；4）实现了对乳房区域的精准捕捉与多姿态测量。

**🔧 技术方法**

主要技术手段有：多视角动态融合（DynamicFusion改进版）、点云配准与T姿归一化、SVD姿态优化、逆LBS求解、PCA构建形状空间、3D几何与深度约束的SMALify改造、以及骨骼正则化与形变平滑损失。

**📊 数据集**

使用的数据集为FemaleSaanenGoat，包含55只山羊的8视角RGBD视频、约3,200份高精度3D扫描和32个标注解剖关键点；此外对照实验还使用了SMAL和SMAL+模型的标准模板和公开的动物3D数据。

**📈 对比分析**

在与SMAL和SMAL+的对比实验中，SaanenGoat在扫描配准上的平均Chamfer距离和网格-扫描误差分别降低至10.65mm/7.02mm（相比SMAL的36.72mm/31.53mm），体型测量MAE大幅下降至1.90mm（SMAL为4.89mm，SMAL+为3.48mm），单视角重建误差也从约70mm降低至约20mm，整体性能提升显著。

**⚠️ 局限性**

局限性主要在于模型仅适用于Saanen品种，难以直接迁移至其他山羊品种；在乳房遮挡严重时重建精度下降；对毛发厚度变化的鲁棒性不足；未来需开发遮挡感知算法、扩展多品种数据并结合多模态传感提升整体性能。

---

## 21. BioEnvSense: A Human-Centred Security Framework for Preventing Behaviour-Driven Cyber Incidents

**arXiv ID:** 2602.19410 | [PDF](https://arxiv.org/pdf/2602.19410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 22. SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models

**arXiv ID:** 2602.18993 | [PDF](https://arxiv.org/pdf/2602.18993v1)

**作者:** Jiwoo Chung `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1501 | [OpenAlex ID](https://openalex.org/A5029469141)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需训练、可插拔的缓存加速方法SeaCache，用谱演化感知滤波器决定何时重用前一步的中间特征，以降低扩散模型的推理时延。

**💡 创新点**

创新点在于将扩散模型的谱演化规律（低频先出现，高频后细化）融入缓存决策，通过设计时钟相关的SEA滤波器对特征进行频谱加权，抑制噪声成分，仅对内容相关的频谱做距离度量，从而更准确地评估重用误差。

**🔧 技术方法**

使用FFT-逆FFT频谱滤波、线性最小均方误差分析得到的SEA滤波器、相对ℓ1距离、累积距离阈值策略实现动态缓存。

**📊 数据集**

在FLUX.1-dev（文本到图像）、HunyuanVideo、Wan2.1（文本到视频）等三种主流扩散/正则化流模型上进行实验，使用DrawBench和VBench数据集。

**📈 对比分析**

与TeaCache、TaylorSeer、DiCache等基准方法对比，SeaCache在相同缓存比例下（约50%或30%刷新率）显著提升PSNR、LPIPS、SSIM，并且在CycleReward评估中排名最低，说明其在速度-质量权衡上更优。

**⚠️ 局限性**

局限在于仅利用谱演化信息，未考虑模型结构或注意力等其他潜在冗余；对极端低刷新率可能仍会出现细节缺失；在某些视频场景中仍需手动调整刷新阈值以避免运动模糊。

---

## 23. GLaDiGAtor: Language-Model-Augmented Multi-Relation Graph Learning for Predicting Disease-Gene Associations

**arXiv ID:** 2602.18769 | [PDF](https://arxiv.org/pdf/2602.18769v1)

**作者:** Osman Onur Kuzucu `[一作]` (Hacettepe University), Tunca Doğan `[通讯]` (Hacettepe University)

**通讯引用:** 16719 | [OpenAlex ID](https://openalex.org/A5003981652)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于图神经网络的异构图学习框架GLaDiGAtor，用于预测疾病与基因的关联关系

**💡 创新点**

创新点在于整合多关系异构图（基因‑基因、疾病‑疾病、基因‑疾病）与预训练的语言模型嵌入（ProtT5和BioBERT），并采用编码器‑解码器结构实现高效预测

**🔧 技术方法**

采用双层Graph Convolutional Network作为编码器，Bilinear解码器进行边预测，结合ProtT5蛋白序列嵌入与BioBERT文本嵌入，使用AdamW优化器和二元交叉熵损失

**📊 数据集**

使用DisGeNET（疾病‑基因关联）、BioGRID（蛋白‑蛋白交互）、DisGeNET疾病相似度、OGB biokg（蛋白‑疾病边）构建多种图变体，实验涵盖六种图构造方式

**📈 对比分析**

与14种现有方法（SkipGNN、HOGCN、ResMGCN等）进行基准测试，GLaDiGAtor在ROC‑AUC和PR‑AUC上均遥遥领先，最高ROC‑AUC≈0.965，PR‑AUC≈0.967，显示出显著的性能优势

**⚠️ 局限性**

局限包括对输入数据库质量与偏差的高度依赖、训练大型异构图所需的计算资源和内存、以及缺乏可解释性解释模型预测背后的机制

---

## 24. Safe and Interpretable Multimodal Path Planning for Multi-Agent Cooperation

**arXiv ID:** 2602.19304 | [PDF](https://arxiv.org/pdf/2602.19304v1)

**作者:** Haojun Shi `[一作]` (Johns Hopkins University), Tianmin Shu `[通讯]` (Johns Hopkins University)

**通讯引用:** 2915 | [OpenAlex ID](https://openalex.org/A5037017610)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了 CaPE，一种安全可解释的多模态路径编辑框架，利用视觉语言模型生成可验证的路径编辑程序并在规划器中验证，实现多智能体协作时基于语言的路径调优。

**💡 创新点**

创新点在于把语言信息映射为可验证的程序编辑路径，而非直接生成路径，兼顾安全性、可解释性，并实现对多机器人与人机协作的通用 Plug-and-Play 模块。

**🔧 技术方法**

核心技术包括视觉语言模型（如 Qwen 2.5、Gemini-3）、自定义 DSL 路径编辑程序、联合 RRT+同伦类规划器以及基于规划器的程序验证器。

**📊 数据集**

数据集方面，使用合成的二维障碍地图（约 5 万例）进行训练，并在 SimWorld（停车场）、VirtualHome（家庭）和真实 Stretch 3 机器人实验中进行评估。

**📈 对比分析**

与规划器仅、VLM 直接生成路径、VLM 生成 waypoint 等基线对比，CaPE 在两机器人/三机器人协调、家庭物品整理、真实联合搬运任务中分别达到了 90%/60%/70% 的成功率，显著优于基线。

**⚠️ 局限性**

局限性包括对可靠感知与对方轨迹预测的依赖，缺乏对感知不确定性的处理，严格的安全验证可能导致与人类意图不完全一致，并且未扩展到大规模群体协作。

---

## 25. Hardware-Friendly Randomization: Enabling Random-Access and Minimal Wiring in FHE Accelerators with Low Total Cost

**arXiv ID:** 2602.19550 | [PDF](https://arxiv.org/pdf/2602.19550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 26. InfEngine: A Self-Verifying and Self-Optimizing Intelligent Engine for Infrared Radiation Computing

**arXiv ID:** 2602.18985 | [PDF](https://arxiv.org/pdf/2602.18985v1)

**作者:** Kun Ding `[一作]` (Institute of Automation, Chinese Academy of Sciences), Shiming Xiang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了InfEngine，一个自验证和自优化的智能引擎，旨在推动红外辐射计算的自动化，减少人工工作流程。

**💡 创新点**

创新点在于引入了自验证和自优化机制，通过联合求解器-评估器调试和自发现适应度函数的进化算法，实现了功能正确性和科学合理性的提升。

**🔧 技术方法**

采用了多代理架构，结合了问题分析代理、问题解决代理、评估生成代理和代码进化代理，利用进化算法进行自优化。

**📊 数据集**

使用了InfTools，包含270个特定于红外辐射计算的工具，以及InfBench，一个包含200个任务的基准数据集。

**📈 对比分析**

与现有的代码生成方法（如Direct、FewShot、CodeCoT等）进行比较，InfEngine在所有评估维度上均表现出色，整体通过率达到92.7%，在优化任务中接近完美，且速度比人工专家快21倍。

**⚠️ 局限性**

限制在于InfTools的范围有限，当前验证主要集中在红外辐射计算领域，未来需要扩展到其他领域，并探索更高效的搜索策略以降低优化成本。

---

## 27. L3DR: 3D-aware LiDAR Diffusion and Rectification

**arXiv ID:** 2602.19064 | [PDF](https://arxiv.org/pdf/2602.19064v1)

**作者:** Quan Liu `[一作]` (Nanyang Technological University), Shijian Lu `[通讯]` (Nanyang Technological University)

**通讯引用:** 16670 | [OpenAlex ID](https://openalex.org/A5023507910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了L3DR框架，先用2D Range-View LiDAR扩散生成点云，再通过3D残差回归网络消除深度泄露、波浪表面等Artifacts；

**💡 创新点**

创新点在于：①利用3D残差回归网络对RV扩散的3D坐标进行局部几何校正；②设计Welsch Loss抑制高偏差区域，专注于修正高方差RV噪声；③框架可对任意LiDAR扩散模型通用；

**🔧 技术方法**

核心技术包括：Range-View投影与逆投影、语义条件LiDAR扩散（LiDM）、3D残差回归网络（SPUNET/RTN）、Welsch Loss、Diffusion模型（DDIM）

**📊 数据集**

使用SemanticKITTI、KITTI360、nuScenes、Waymo Open Dataset四大公开数据集进行训练与评估；

**📈 对比分析**

与LiDARGen、R2DM、LiDM等基线在全局与局部几何指标（FSVD、FPVD、JSD、MMD）上均取得显著提升，特别是JSD降低到0.07~0.18，MMD提升超过40%；

**⚠️ 局限性**

局限性包括：需先训练扩散模型，处理高偏差错误仍受限；目前仅针对单一类型的RVArtifacts，扩散模型对场景多样性的适应仍需进一步验证；

---

## 28. Emergent Dark Patterns in AI-Generated User Interfaces

**arXiv ID:** 2602.18445 | [PDF](https://arxiv.org/pdf/2602.18445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 29. Learning Discriminative and Generalizable Anomaly Detector for Dynamic Graph with Limited Supervision

**arXiv ID:** 2602.20019 | [PDF](https://arxiv.org/pdf/2602.20019v1)

**作者:** Yuxing Tian `[一作]` (University of Montreal), Jian-Yun Nie `[通讯]` (University of Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种通用的动态图异常检测框架SDGAD，能够在无标签或少量标签的情况下学习明确且稳健的判别边界。

**💡 创新点**

创新点在于三方面：① 通过残差表示编码突出当前交互与近期上下文的差异，增强异常相关信号；② 采用双同心超球约束把正常残差压缩到一个一致尺度的区间内，避免尺度不一致导致的边界模糊；③ 用正则化的bi‑boundary优化结合正则化流模型，显式学习正常与异常的对数似然分布边界并保持缓冲区间。

**🔧 技术方法**

技术手段包括：残差表示编码、双同心超球约束损失、正则化流（normalizing flow）密度估计、softplus基的bi‑boundary优化，以及对标记异常进行交叉熵监督的可选 MLP。

**📊 数据集**

在六个数据集上进行评估：三组真实异常数据（Wikipedia、Reddit、MOOC）和三组插入合成异常的数据（Enron、UCI、LastFM）。

**📈 对比分析**

与多种基线（DTDG方法、CTDG专用DGAD、通用CTDG编码器）对比，SDGAD在AUROC、AP和F1上均优于所有方法，尤其在F1上提升显著；在全无监督和少量监督设置下仍保持高稳定性，标准差更低。

**⚠️ 局限性**

局限性包括：仍需依赖预训练的CTDG编码器；对极端高维数据的可扩展性未完全验证；以及在极少标注（k≤1）时性能提升相对有限，需进一步探索更高效的少样本学习策略。

---

## 30. Personalized Longitudinal Medical Report Generation via Temporally-Aware Federated Adaptation

**arXiv ID:** 2602.19668 | [PDF](https://arxiv.org/pdf/2602.19668v1)

**作者:** He Zhu `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种新的联邦学习设置——Federated Temporal Adaptation（FTA），用于在保证隐私的前提下对长期医学影像报告生成进行时序建模。

**💡 创新点**

核心创新包括：①利用患者人口统计信息生成低秩LoRA适配器实现个性化；②引入元学习的时间残差聚合策略，根据时间步动态加权全局模型更新；③将时序演化作为目标函数的一部分，克服传统FL对数据分布静态假设的局限。

**🔧 技术方法**

技术方法包括：Gaussian Mixture Model（GMM）对人口统计进行软聚类，Hypernetwork生成LoRA参数，Transformer（ViT+DistilGPT‑2）作为基模型，MAML/超梯度用于学习时间权重α_t，残差聚合实现凸组合保证收敛性。

**📊 数据集**

主要数据集：1）J‑MID（约100万张CT/MR扫描，10家医院，5次随访）；2）MIMIC‑CXR（公开胸部X光图像及报告）。

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD、FedAdam、FedYogi、DRFA等FL基线在BLEU、ROUGE、CIDEr、CE指标上对比，FedTAR均获得显著提升（如BLEU‑4从10.98提升至12.40，CIDEr从31.70提升至42.80）。

**⚠️ 局限性**

局限性：①对均匀随访时间窗假设，未处理不规则/缺失访视；②模型仍依赖预训练Transformer，对低资源场景的适用性未知；③元学习阶段需要额外的验证集，若验证样本有限可能导致过拟合。

---

## 31. RADE-Net: Robust Attention Network for Radar-Only Object Detection in Adverse Weather

**arXiv ID:** 2602.19994 | [PDF](https://arxiv.org/pdf/2602.19994v1)

**作者:** Christof Leitgeb `[一作]` (Infineon Technologies AG), Daniel Watzenig `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于3D投影的RADE雷达张量轻量化检测网络RADE-Net，实现了雷达仅在恶劣天气下的鲁棒目标检测。

**💡 创新点**

创新点在于通过3D投影保留Doppler与Elevation信息，结合2D CNN编码解码与空间-通道注意力，以及解耦中心点检测与旋转3D框回归，实现91.9%数据压缩并显著提升检测性能。

**🔧 技术方法**

采用FFT处理雷达张量、3D投影、CBAM注意力、残差膨胀颈、CenterPoint式检测头，并使用焦点损失与Gaussian‑Wasserstein+SmoothL1组合损失。

**📊 数据集**

在K‑Radar大型雷达数据集上训练和评估，涵盖多天气、多车辆类型和3D框标注。

**📈 对比分析**

与基线雷达模型、LiDAR、Camera‑LiDAR等多模态方法对比，RADE-Net在所有天气下的AP提升约16.7%（基线）/6.5%（最新雷达）并在恶劣天气下超过LiDAR，尤其雾天提升32.1%。

**⚠️ 局限性**

局限在于对小目标（行人、骑车人）检测不足，投影预处理仍未实时优化，且仅在单帧雷达上进行，未考虑多帧上下文或跟踪。

---

## 32. PCA-VAE: Differentiable Subspace Quantization without Codebook Collapse

**arXiv ID:** 2602.18904 | [PDF](https://arxiv.org/pdf/2602.18904v1)

**作者:** Hao Lu `[一作]` (Wake Forest University), Metin Nafi Gurcan `[通讯]` (Wake Forest University)

**通讯引用:** 10821 | [OpenAlex ID](https://openalex.org/A5077316017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新型的PCA‑VAE模型，将传统VQ瓶颈替换为在线学习的PCA层；

**💡 创新点**

创新点在于利用Oja’s Rule实现可微分的正交PCA投影，消除VQ的非可导性与码本坍塌问题，同时获得自排序且可解释的潜在轴；

**🔧 技术方法**

核心技术包括在线PCA（Oja规则 + r‑fade平均）、VAE架构以及全微分训练；

**📊 数据集**

实验使用CelebA‑HQ 256×256人脸数据集；

**📈 对比分析**

与VQGAN、SimVQ、VQ‑VAE和AutoencoderKL在相同网格与令牌预算下比较，PCA‑VAE在PSNR、SSIM、LPIPS和rFID等指标上表现更佳，并以10–100倍更少的latent位数实现同等或更优的重建质量；

**⚠️ 局限性**

限制在于仅评估重建性能，未验证生成采样效果，也未在更大规模数据或更大模型上进行实验。

---

## 33. CLAP Convolutional Lightweight Autoencoder for Plant Disease Classification

**arXiv ID:** 2602.18833 | [PDF](https://arxiv.org/pdf/2602.18833v1)

**作者:** Asish Bera `[一作]` (Birla Institute of Technology and Science Pilani), Sudiptendu Banerjee `[通讯]` (Birla Institute of Technology and Science Pilani)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级卷积自编码器 CLAP，用于植物叶片病害的分类。

**💡 创新点**

创新点在于结合深度可分离卷积、门控 Sigmoid 注意力以及编码器‑解码器的融合结构，使模型参数仅约 5 M、计算量低且特征表征更丰富。

**🔧 技术方法**

采用的技术包括深度可分离卷积、BatchNorm+ReLU、全局平均池化（GAP）、Sigmoid 门控注意力、解码器上采样、特征拼接和 Softmax 分类。

**📊 数据集**

使用了公开的集成植物病害（IPD）、Groundnut 以及 CCMT（cashew、cassava、maize、tomato）等多种数据集进行实验。

**📈 对比分析**

与 MobileNetV2、Xception、ResNet、DenseNet 等基准模型比较，CLAP 在 IPD 95.67%、Groundnut 96.85%、CCMT 87.11% 等指标上与轻量化模型相当或更优，且训练时间仅 20 ms/图像、推理时间约 1 ms/图像，显著加速。

**⚠️ 局限性**

局限性包括在极少样本或复杂光照场景下的泛化能力尚未充分验证，且与大规模预训练模型相比仍存在性能差距。

---

## 34. Nazrin: Atomic Tactics for Graph Neural Networks for Theorem Proving in Lean 4

**arXiv ID:** 2602.18767 | [PDF](https://arxiv.org/pdf/2602.18767v1)

**作者:** Leni Aniva `[一作]` (Stanford University), Clark Barrett `[通讯]` (Stanford University)

**通讯引用:** 10145 | [OpenAlex ID](https://openalex.org/A5026961968)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套有限且完整的原子 tactics，并设计了转置 atomization 算法将现有 Lean 证明转化为原子化序列；同时引入了 ExprGraph 作为 Lean 表达式的图形表示，并基于图神经网络与神经概率自动机（NPA）实现了低资源、高吞吐的自动证明器 Nazrin Prover。

**💡 创新点**

创新点包括：①使用原子 tactics 形成有限动作空间并保证任何可证明命题的可达性；②转置 atomization 能把任意复杂证明映射到原子化表述；③ExprGraph 通过“essentialization”去除多余信息，保留可重写子表达式；④NPA 结合 GNN 的多头注意力与可变参数生成，实现在低资源环境下的快速证明；⑤在标准库和 Mathlib 上展示了与 Aesop、Grind 的互补性能。

**🔧 技术方法**

主要技术：图神经网络（Core GNN + Fixed‑point GNN）、神经概率自动机（NPA）、转置 atomization 算法、Rainbow guidance（基于先后关系的启发式）、自动化前置处理（展开自动化 lemmas、标准化 congruence 等）。

**📊 数据集**

使用了 Lean 4 标准库和 Mathlib 的 170,180 条用户定义定理（已转化为 atomized 形式），并按 slice 划分进行训练与评估。

**📈 对比分析**

评估方法：在 15 秒/定理的时间预算下，将 Nazrin 与 Aesop、Grind 在标准库 slice 1‑2 与 Mathlib slice 3‑4 进行比较。Nazrin 在标准库 slice 2 上实现 57% 的证明成功率，在 Mathlib slice 4 上实现 34%；相比 Aesop、Grind，Nazrin 能证明部分其无法完成的定理；证明速度可达每分钟数千条 tactic，显著高于基于 LLM 的方法。

**⚠️ 局限性**

限制：atomization 覆盖率约 58%，对高交叉截面（high cross‑section）证明的支持仍有限；NPA 对多参数 tactic 的依赖关系处理不充分；GNN 不能处理数字与字符串，需要机械补偿；架构未经过充分调优，存在性能提升空间；对未见符号的 embedding 依赖固定点生成，可能影响泛化。

---

## 35. Media Integrity and Authentication: Status, Directions, and Futures

**arXiv ID:** 2602.18681 | [PDF](https://arxiv.org/pdf/2602.18681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 36. Understanding Fire Through Thermal Radiation Fields for Mobile Robots

**arXiv ID:** 2602.19108 | [PDF](https://arxiv.org/pdf/2602.19108v1)

**作者:** Anton R. Wagner `[一作]` (Kiel University), Sören Pirk `[通讯]` (Kiel University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套基于热深度融合的火灾环境中移动机器人安全导航框架，能够实时构建三维温度点云、定位火源并生成连续的热辐射场，以此为 A* 规划提供热安全成本。

**💡 创新点**

创新点在于将热相机表面温度通过物理模型（Stefan‑Boltzmann 定律与逆平方衰减）扩展为自由空间中的连续热辐射场，并将此场直接融入占用率成本图，首次实现了从传感器数据到热安全路径规划的闭环。

**🔧 技术方法**

主要技术包括热深度相机外参标定、DBSCAN 高温点聚类、半球近似求发射面积、Stefan‑Boltzmann 计算热功率、光照线‑视效验、基于热阈值的安全距离估计以及 A* 路径规划。

**📊 数据集**

使用数据集为 Boston Dynamics Spot 机器人在实验室控制的丙烷火源下的实时深度与热图像，并结合 500 ml 水热量计实验验证火源热功率。

**📈 对比分析**

实验比较显示，加入热安全成本后机器人成功避开热区并能按不同安全边界规划不同距离的路径；热安全距离估计误差约 0.22 m；与仅几何障碍规划相比，热安全路径能更好避免高温区域。

**⚠️ 局限性**

局限性包括仅测试单一静态火源、假定固定火焰温度、仅考虑辐射热而忽略对流、低分辨率热相机导致温度上限受限、未验证多火源或动态火焰的鲁棒性。

---

## 37. Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding

**arXiv ID:** 2602.18977 | [PDF](https://arxiv.org/pdf/2602.18977v1)

**作者:** Thinesh Thiyakesan Ponbagavathi `[一作]` (Institute for Artificial Intelligence University of Stuttgart), Alina Roitberg `[通讯]` (Intelligent Assistive Systems Lab University of Hildesheim)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于频域的参数高效适配器 Frame2Freq，能够将预训练的图像视觉基础模型（VFM）迁移到视频任务，并显著提升细粒度动作识别性能。

**💡 创新点**

创新点在于：①首次将快速傅里叶变换（FFT）引入视频时序建模；②设计单尺度（ST）和多尺度（MS）频域适配器，专门对中频动量信号进行强调；③通过频率判别分析（Frequency Discriminability Analysis）证明中频段对细粒度动作最具判别力，从而实现对频域能量的动态分配。

**🔧 技术方法**

主要技术包括：冻结的视觉变压器（如 CLIP、DINOv2）作为骨干；FFT/短时FFT（STFT）对时序特征进行频域转换；频域分支学习频带特定滤波器并通过 iFFT 回到时域；多尺度窗口和深度可分离卷积实现局部与全局频率融合；残差连接将频域特征注入 Transformer 块；轻量化瓶颈结构实现参数高效学习。

**📊 数据集**

在五个细粒度/领域特定数据集上评估：SSv2、Diving48、Drive&Act、IKEA-ASM、HRI-30；每个数据集使用冻结的 CLIP 或 DINOv2 骨干，仅训练适配器与线性头。

**📈 对比分析**

与主流 PEFT 方法（ST‑Adapter、AIM、DualPath 等）和部分全微调模型（如 ORViT、UniformerV2）对比，Frame2Freq‑MS 在 4 个细粒度数据集上均达到或超过全微调模型的准确率，甚至在 5 个数据集上实现 0.9–4.5% 的 Top‑1 提升；在 SSv2 的少量样本设置下更是实现 1.5–3.0% 的提升，优于现有 few‑shot 方案。

**⚠️ 局限性**

局限性包括：①对非常粗粒度或非运动强烈的动作提升有限；②需要 FFT 计算，虽然参数少但在实时推理时仍有额外运算开销；③目前仅支持冻结的图像 VFM，尚未探索在已微调或跨域场景中的泛化能力；④多尺度设计需手动选择窗口组合，可能不适用于所有视频节奏。

---

## 38. The Metaphysics We Train: A Heideggerian Reading of Machine Learning

**arXiv ID:** 2602.19028 | [PDF](https://arxiv.org/pdf/2602.19028v1)

**作者:** Heman Shakeri `[一作]` (University of Virginia), Heman Shakeri `[通讯]` (University of Virginia)

**通讯引用:** 423 | [OpenAlex ID](https://openalex.org/A5006445265)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

以海德格尔phenomenology视角对现代机器学习进行哲学解读，揭示算法投影的本体特性与存在缺失；

**💡 创新点**

提出算法投影的四个独特维度（自动化、隐晦、高维、执行），并指出即使技术进步仍停留在Ge‑Stell框架内，AI 缺乏 Care 结构可解释其失效；

**🔧 技术方法**

主要运用海德格尔哲学概念（投影、Ge‑Stell、Care、Angst）与机器学习术语（Transformer、Softmax、ERM、损失函数）的理论分析；

**📊 数据集**

未使用具体数据集，仅对Transformer及其训练流程进行概念性案例分析；

**📈 对比分析**

无定量实验或性能评估；通过对比常见技术进步与哲学视角的差异，强调现有方法提升计算能力而非本体质疑；

**⚠️ 局限性**

局限在于缺乏经验验证，依赖哲学解释，未提供可操作技术改进方案，可能被视为抽象讨论。

---

## 39. Servicing Matched Client Pairs with Facilities

**arXiv ID:** 2602.19680 | [PDF](https://arxiv.org/pdf/2602.19680v1)

**作者:** Fateme Abbasi `[一作]` (Institute of Computer Science), Yongho Shin `[通讯]` (Institute of Computer Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并求解一种新的设施选址与配对匹配问题（匹配兼容图下的设施选址），同时最大化匹配数量并最小化开设设施及匹配成本。

**💡 创新点**

1) 设计了结合设施选址与匹配多面体的线性规划松弛，证明其整形间隙有限；2) 通过“重路由”子程序将任意LP解转化为仅支持最大匹配的分数解，进而得到3.868-近似（一般情况）和2.218-近似（兼容图可完美匹配时）的算法。

**🔧 技术方法**

线性规划松弛、匹配多面体理论、分数重路由、双因子近似（Bifactor）LP-舍入技术、最小费用完美匹配求解。

**📊 数据集**

未使用公开数据集，算法在理论上对任意度量空间实例成立；若需实验，可在随机生成的设施/客户/兼容图上进行验证。

**📈 对比分析**

与传统的设施选址近似（1.678-近似）以及先前对完全兼容图的2-近似相比，本工作在一般情况取得了3.868的全局近似，并在兼容图完美匹配时将误差压缩至2.218，显著提升。

**⚠️ 局限性**

主要限制包括：开放成本的倍增因子至少为1.5、连接成本倍增因子至多3，且在一般情况下整体近似因子仍高；对动态/在线场景、非完美匹配以及更广义的k-包装约束尚未覆盖。

---

## 40. Quantum approaches to learning parity with noise

**arXiv ID:** 2602.19819 | [PDF](https://arxiv.org/pdf/2602.19819v1)

**作者:** Daniel Shiu `[一作]` `[通讯]`, Daniel Shiu

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了量子方法在学习带噪声的平衡问题（LPN）中的应用，探讨了如何利用Simon算法生成新的LPN样本，以期简化问题的求解。

**💡 创新点**

创新点在于将覆盖码应用于LPN问题的量子超位置，尝试保持Simon的承诺，并利用Simon算法生成额外的样本以简化问题。

**🔧 技术方法**

使用了量子计算方法，特别是Simon算法，并结合覆盖码的概念。

**📊 数据集**

使用了随机生成的Goppa码和随机线性码的数据集进行实验，评估了不同噪声权重下的样本生成效果。

**📈 对比分析**

通过与经典的LPN求解方法进行比较，展示了量子方法在生成新样本方面的潜力，尽管没有明确的性能提升，但提供了新的思路。

**⚠️ 局限性**

限制在于所提出的方法可能无法与现有的LPN攻击方法竞争，且对噪声率的敏感性可能影响其有效性。

---

## 41. Watson & Holmes: A Naturalistic Benchmark for Comparing Human and LLM Reasoning

**arXiv ID:** 2602.19914 | [PDF](https://arxiv.org/pdf/2602.19914v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 42. TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures

**arXiv ID:** 2602.19679 | [PDF](https://arxiv.org/pdf/2602.19679v1)

**作者:** Hyeongjin Nam `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于文本指导的框架，能够从单张图像同时重建3D人类与物体，并生成全局一致的纹理与几何；

**💡 创新点**

创新点在于：①利用文本描述作为语义引导，补偿传统接触信息缺失，支持非接触交互；②将全局外观信息与文本对齐的扩散网络作为损失，提升整体可视化一致性；③采用三维高斯表示实现无拓扑、可渲染的细粒度优化；

**🔧 技术方法**

技术包括：三维高斯表示（3DGS）、SMPL-X骨骼驱动、Vision-Language模型（GPT‑4）进行文本生成、预训练扩散模型（Stable Diffusion）做score distillation、CLIP/HDM等对齐与碰撞惩罚、三维高斯到网格的转换；

**📊 数据集**

在Open3DHOI和BEHAVE两个公开数据集上进行评估，Open3DHOI用于测试，BEHAVE用于验证；

**📈 对比分析**

与多种现有方法（PHOSA、CONTHO、HOI‑Gaussian、InteractVLM、PICO、HDM 等）做定量与定性比较，使用Chamfer距离、接触F1、碰撞率等指标；结果表明本文在人/物体精度、接触准确性和非接触场景下均显著优于现有方法；

**⚠️ 局限性**

局限性：对细节纹理和小配件的重建不足；仅针对单帧图像，未考虑时序一致性；文本生成依赖于VLM的准确性，若描述错误可能影响结果；

---

## 43. Driving with A Thousand Faces: A Benchmark for Closed-Loop Personalized End-to-End Autonomous Driving

**arXiv ID:** 2602.18757 | [PDF](https://arxiv.org/pdf/2602.18757v1)

**作者:** Xiaoru Dong `[一作]` (University of Hong Kong), Yuexin Ma `[通讯]` (ShanghaiTech University)

**通讯引用:** 4092 | [OpenAlex ID](https://openalex.org/A5102015139)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个完整的个人化端到端自动驾驶（Person2Drive）平台与基准，涵盖低成本数据采集、可解释的驾驶风格评估以及风格引导的轻量化微调框架。

**💡 创新点**

①首次构建大规模、多模态、闭环的人机交互驾驶数据集；②引入风格向量、MMDSS与KL离散度的定量评估体系；③设计仅更新预测头的风格奖励模型实现高效安全的个性化微调。

**🔧 技术方法**

CARLA仿真、低成本人机交互硬件、DiffusionDrive基础模型、风格奖励模型（基于轨迹的可微映射）、MMDSS/KL离散度、BEV特征融合、强化学习与监督学习混合训练。

**📊 数据集**

Person2Drive（30名司机，每人多条1km轨迹，包含RGB、LiDAR、雷达、IMU、BEV等多模态数据），并在公开数据集StyleDrive进行验证。

**📈 对比分析**

与基线（原始DiffusionDrive）及直接微调(DFT)比较，Person2Drive风格微调后MMDSS提升约5%–7%，KL下降，驾驶得分提升约15点，成功率基本不变，表明既保留又提升了驾驶能力。

**⚠️ 局限性**

依赖仿真环境，真实场景的可迁移性待验证；风格向量仍由经验选择，可能缺乏更细粒度的心理学特征；BEV特征在实际部署中需额外硬件支持。

---

## 44. Initialization matters in few-shot adaptation of vision-language models for histopathological image classification

**arXiv ID:** 2602.18766 | [PDF](https://arxiv.org/pdf/2602.18766v1)

**作者:** Pablo Meseguer `[一作]` (Universitat Politécnica de Valencia), Valery Naranjo `[通讯]` (Universitat Politécnica de Valencia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出零样本多实例学习框架ZS-MIL，用文本嵌入初始化分类器来实现全切片图像子类型的零样本/少样本分类。

**💡 创新点**

创新点在于利用VLM文本编码器的类别嵌入作为线性分类器的初始化，从而在少样本MIL中显著提升性能并降低随机初始化带来的方差。

**🔧 技术方法**

使用Vision‑Language模型的图像编码器提取补丁特征，文本提示集生成类别原型，注意力/池化聚合方法与温度软化的多类别softmax进行分类，并通过交叉熵训练。

**📊 数据集**

在TCGA肺腺癌与肺鳞癌公开数据集上进行实验，分别包含445份LUSC和291份LUAD的WSI。

**📈 对比分析**

与Kaiming、Xavier等随机初始化以及MI‑Zero零样本基线进行对比，ZS‑MIL在4-shot和16-shot情境下分别提升约19%和5%的平衡准确率，并显著降低方差。

**⚠️ 局限性**

局限性包括仍需冻结图像编码器，聚合方法对轻量级模型依赖较大，且对文本提示的选择和质量敏感，未对更大模型或其他组织类型验证。

---

## 45. Spilled Energy in Large Language Models

**arXiv ID:** 2602.18671 | [PDF](https://arxiv.org/pdf/2602.18671v1)

**作者:** Adrian Robert Minut `[一作]`, Iacopo Masi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 4882 | [OpenAlex ID](https://openalex.org/A5089059382)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练-free的能量泄漏检测框架，通过将LLM的softmax层重新解释为能量模型（EBM），在解码时监测相邻时间步的能量差异（spilled energy）来识别模型的幻觉和错误。

**💡 创新点**

创新点在于：① 引入能量泄漏Δi以及边缘能量im作为无训练的误差指示器；② 利用EBM框架在推理时实现跨任务、跨模型的泛化；③ 在不需要额外训练的前提下实现对幻觉的实时检测。

**🔧 技术方法**

技术手段包括：基于链式概率的能量模型推导、对LLM logits的直接读取、对答案位置的精确定位以及多种池化策略（如最小池化）来聚合能量值。

**📊 数据集**

数据集涵盖9个公开基准（Math、TriviaQA、HotpotQA、Winogrande、Winobias、Movies、MNLI、IMDB 等）以及合成算术问题（Qwen-3、Llama‑3、Mistral 等）。

**📈 对比分析**

与传统基线（logits、p(true)、训练probe）相比，spilled energy 在 AUROC 上显著提升，尤其在难以检测的错误范围（如[1,10]）和跨数据集迁移时表现更稳健；在指令微调模型上提升更明显。

**⚠️ 局限性**

局限性包括：对非语义信息（标点、句首词等）可能产生误报，需要精准定位答案位置；能量差异受领域差异影响，导致在不同任务间方差较大。

---

## 46. Vinedresser3D: Agentic Text-guided 3D Editing

**arXiv ID:** 2602.19542 | [PDF](https://arxiv.org/pdf/2602.19542v1)

**作者:** Yankuan Chi `[一作]` (Hong Kong University of Science and Technology), James M. Rehg `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 23827 | [OpenAlex ID](https://openalex.org/A5002228469)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于多模态大语言模型（MLLM）的智能代理 Vinedresser3D，实现文本指令驱动的高质量 3D 资产编辑，支持增删改、跨回合编辑并保持未编辑区域完整。

**💡 创新点**

创新点包括：① 将 MLLM 作为核心，自动从文本指令生成结构化与外观化的多模态编辑指引；② 通过 PartField 等 3D 分割模型结合 MLLM 自动定位编辑区域，消除用户手工掩码需求；③ 在 3D 潜空间中采用基于 RF‑Solver 的逆向编辑与交错使用 Trellis‑Text 与 Trellis‑Image 的互补采样，实现高保真、语义一致的编辑。

**🔧 技术方法**

主要技术包括 Gemini‑2.5‑Flash MLLM、Nano Banana 图像编辑器、PartField 3D 分割、Trellis 3D 流式生成模型、RF‑Solver 逆向求解、交错采样的 3D inpainting 方案。

**📊 数据集**

使用来自 Trellis 生成、GSO 及 PartObjaverse‑Tiny 的 57 条高质量 3D 资产数据，并为每条资产设计符合常识的编辑提示，覆盖多种编辑类别与难度。

**📈 对比分析**

与 Trellis（基于人类掩码的编辑）、VoxHammer（需要掩码且只用图像编辑）以及 Instant3dit（多视图扩散+重建）对比；在 CLIP‑T 对齐、Chamfer Distance、PSNR/SSIM、LPIPS 与 FID 方面，Vinedresser3D 在无掩码模式下取得最高 CLIP‑T 并保持竞争性保真度，使用人类掩码时性能进一步提升，整体 FID 也优于基线。

**⚠️ 局限性**

局限性：MLLM 只能处理 2D 输入，缺乏直接 3D 推理；所用外部工具（PartField、图像编辑器等）仍易产生不理想的分割或编辑结果；对复杂场景和极端几何形状的鲁棒性尚待提升。

---

## 47. As Content and Layout Co-Evolve: TangibleSite for Scaffolding Blind People's Webpage Design through Multimodal Interaction

**arXiv ID:** 2602.19243 | [PDF](https://arxiv.org/pdf/2602.19243v1)

**作者:** Jiasheng Li `[一作]` (University of Maryland), Huaishu Peng `[通讯]` (University of Maryland)

**通讯引用:** 1074 | [OpenAlex ID](https://openalex.org/A5067580146)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了一种名为TangibleSite的多模态网页设计工具，使盲人能够在同一系统中同时创建和迭代网页内容与布局。

**💡 创新点**

创新点在于将内容与布局的共同进化纳入盲人工作流程，提供持久的页面结构记忆、触摸对齐的即时反馈以及可操作的音频指导。

**🔧 技术方法**

技术采用可调形状的物理框架与磁性触点、振动反馈、语音识别/合成、React前端、Express后端以及数据库来同步硬件与网页渲染。

**📊 数据集**

评估使用了六名盲人参与者自行设计的网页数据集，共六个示例页面。

**📈 对比分析**

通过问卷Likert与实操验证，参与者在内容生成、布局设计与迭代方面均获得了高分（如66.7%强烈同意可生成内容），并报告较低的挫折感。

**⚠️ 局限性**

局限性包括：布局仅支持固定网格与边框；不支持响应式或高级CSS；未自动生成可访问性元数据；仅进行单次会话的短期评估。

---

## 48. Online Navigation Planning for Long-term Autonomous Operation of Underwater Gliders

**arXiv ID:** 2602.19315 | [PDF](https://arxiv.org/pdf/2602.19315v1)

**作者:** Victor-Alexandru Darvariu `[一作]` (Oxford Robotics Institute), Nick Hawes `[通讯]` (Oxford Robotics Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了水下滑翔机的长期自主导航规划，提出基于随机最短路径马尔科夫决策过程的在线样本规划方法，并实现了完整的自动控制系统。

**💡 创新点**

创新点包括：①将滑翔机导航问题建模为不确定性马尔科夫决策过程；②设计了可数据驱动、可加速的物理仿真器；③在此基础上采用根并行、双递进扩展的蒙特卡洛树搜索进行在线规划；④在北海完成了连续3个月、约1000公里的现场验证。

**🔧 技术方法**

采用的技术包括蒙特卡洛树搜索（UCT + 双递进扩展）、物理仿真器、Bayesian Optimization（BoTorch）进行仿真器参数调优、Oceanids C2通信框架、Slocum滑翔机的指令转换接口。

**📊 数据集**

使用了过去滑翔机历史轨迹与AMM15海流预测数据作为训练/校准数据集，以及北海现场部署的实际测量数据进行验证。

**📈 对比分析**

通过与传统直达目标（Straight‑to‑Goal）方法比较，在仿真中提升了约9.88% 的里程、16.51% 的路径长度，在现场实验中平均降低约30分钟的时长并实现9.55%的路径长度显著下降。

**⚠️ 局限性**

主要局限包括：仿真器对位置与时间仍有误差，受海流预测误差和控制执行噪声影响；通信失败可能导致指令丢失；目前仅针对单机导航，未涵盖多机协同和能源优化等方面。

---

## 49. Alternating Bi-Objective Optimization for Explainable Neuro-Fuzzy Systems

**arXiv ID:** 2602.19253 | [PDF](https://arxiv.org/pdf/2602.19253v1)

**作者:** Qusai Khaled `[一作]` (Eindhoven University of Technology), Laura Genga `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 457 | [OpenAlex ID](https://openalex.org/A5056706542)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 X-ANFIS，一种交替双目标梯度优化的可解释神经模糊系统，用于同时优化准确性与可解释性。

**💡 创新点**

创新点在于将准确性与解释性目标解耦并交替更新，使用 Cauchy 成员函数提升梯度稳定性，并能够在非凸 Pareto 前沿上获得比加权标量化更优解。

**🔧 技术方法**

采用 Cauchy 成员函数、正则化最小二乘估计、梯度下降、可微解释性目标函数、交替双目标优化以及热图可视化评估。

**📊 数据集**

实验基于 9 个 UCI 回归数据集（如 Combined Cycle Power Plant、Airfoil Noise、Concrete Strength、Energy Efficiency、Yacht、Bike Sharing Day/Hour、Steel Industry Energy、Superconductivity）。

**📈 对比分析**

通过与单目标 ANFIS 与加权多目标 MO-ANFIS 的 R² 与可分辨性（D）对比，X-ANFIS 在保持约 0.5 的可分辨性的同时，R² 与 ANFIS 仅略低，显著优于 MO-ANFIS 的性能折衷。

**⚠️ 局限性**

局限在于仅验证了零阶 T‑S 模型，对更大规则集、Mamdani 系统以及结构与语义解释性共同优化的普适性尚待进一步研究。

---

## 50. Online Realizable Regression and Applications for ReLU Networks

**arXiv ID:** 2602.19172 | [PDF](https://arxiv.org/pdf/2602.19172v1)

**作者:** Ilan Doron-Arad `[一作]` (Massachusetts Institute of Technology), Elchanan Mossel `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 9803 | [OpenAlex ID](https://openalex.org/A5013467728)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

在对抗性在线回归模型中，提出了一种基于 Dudley 熵积分的潜能函数方法，利用覆盖数估计上界化了可实现的尺度 Littlestone 维数，从而给出可实现在线回归的无穷期累计损失界；并将该框架应用于 L‑Lipschitz 函数和有限参数 ReLU 网络，得到关于损失指数 q、维度 d、网络宽度 k 的精确相位转移与上界/下界。

**💡 创新点**

创新点主要包括：
1) 通过覆盖数的 Dudley 熵积分给出了尺度 Littlestone 维数的通用上界，实现了从几何熵到可实现累计损失的直接映射；
2) 证明了在 ℓ_q 损失下 L‑Lipschitz 函数的可实现回归存在严格的相位转移（q>d 有界，q≤d 无界）；
3) 在 ReLU 网络中展示了分类与回归的显著差异，得到 2‑ReLU 网络在分类下不可学习但在回归下可实现有限损失，并给出了 O(k^2 log^4 d) 的累积损失上界；
4) 提出了一个效率可达 O(1) 的实现方案，适用于单个 ReLU 并可扩展到多层网络。

**🔧 技术方法**

使用的技术包括：
- 近似伪度量（c‑approximate pseudo‑metric）与对应的 sup‑伪度量；
- 扩展的尺度 Littlestone 维数与对可实现回归的极小化性质；
- Dudley 熵积分与覆盖数（ε‑cover）理论；
- 卷积/包络（envelope）策略实现可实现在线学习；
- 序列 fat‑shattering 与 Rademacher 复杂度相结合的耦合/缩放论证；
- 低维空间分层分解与 McShane 延拓构造的对抗性树。

**📊 数据集**

无（纯理论分析，没有使用实际数据集）。

**📈 对比分析**

与已有工作相比，本文在可实现回归领域提供了：
- 对 L‑Lipschitz 类的 q>d 情况下实现了与已知上界匹配的 O_d,q(L^d) 绝对损失；
- 在 q≤d 时给出与对抗性树匹配的对数/多项式下界，验证了相位转移的严谨性；
- 对 bounded k‑ReLU 网络给出了 O(k^2 log^4 d) 的累积损失上界，并证明在 2‑ReLU、1‑ReLU 等特殊情况下的最优性；
- 通过对比 0/1 损失下的不可学习性，凸显回归与分类在可实现性上的根本区别。整体而言，算法在可实现回归问题上实现了无穷期、无穷期的最优/近似最优累计损失控制。

**⚠️ 局限性**

限制与未来方向：
- 熵势函数上界在某些情形下可能不是紧致的（Φ 可能发散导致 𝔻_onl 上界无用）；
- 对 bounded k‑ReLU 的上界仍有 k^2 级别，尚未证明是否可降至线性 k；
- 对 k>1 的 ReLU，尽管提供了信息论上限，但高效实现仍未突破；
- 对更深层网络的泛化仍需进一步研究；
- 近似三角不等式的假设限制了可适用的损失函数范围；
- 对非平滑损失与高维数据集的实验验证仍缺失。

---

## 51. Keep it SymPL: Symbolic Projective Layout for Allocentric Spatial Reasoning in Vision-Language Models

**arXiv ID:** 2602.19117 | [PDF](https://arxiv.org/pdf/2602.19117v1)

**作者:** Jaeyun Jang `[一作]` (Kyung Hee University), Hyoseok Hwang `[通讯]` (Kyung Hee University)

**通讯引用:** 293 | [OpenAlex ID](https://openalex.org/A5018395387)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SymPL框架，将视角感知空间推理问题改写为符号布局问题，便于视觉语言模型处理；

**💡 创新点**

通过四个关键因子（投影、抽象、二分、定位）系统性地重构问题，实现从对象视角到符号化视角的转换；

**🔧 技术方法**

使用预训练的视觉语言模型（如Qwen2.5‑VL）、物体检测器GroundingDINO、DepthPro、OrientAnything等，结合投影与符号绘制；

**📊 数据集**

在五大基准数据集上评估：COMFORT#、3DSRBench、COCOSPATIAL、COMFORT VI、COMFORT Multi；

**📈 对比分析**

与多种基准VLM（通用、推理辅助、专用的前/后、左右等）和传统方法（APC、SAT）对比，SymPL在allocentric与egocentric场景均显著提升，allocentric上最高可达97.33%/91.5%等；

**⚠️ 局限性**

主要限制包括依赖基础模型的准确性（如对象检测、深度估计、方向向量），以及在极端视觉伪影或复杂场景下仍可能产生错误。

---

## 52. Temporal Action Representation Learning for Tactical Resource Control and Subsequent Maneuver Generation

**arXiv ID:** 2602.18716 | [PDF](https://arxiv.org/pdf/2602.18716v1)

**作者:** Hoseong Jung `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**通讯引用:** 7180 | [OpenAlex ID](https://openalex.org/A5073996122)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出TART框架，用对时序动作表示学习实现资源控制与随后的机动生成

**💡 创新点**

通过信息最大化对齐资源决策与后续连续动作，并将结果量化为可解释的VQ代码，兼顾因果性与多模态性

**🔧 技术方法**

对比学习（InfoNCE）+向量量化（VQ）+PPO强化学习

**📊 数据集**

改进版POMDP环境：预算化迷宫导航（POGEMA）和基于JSBSim的F‑16空战模拟器

**📈 对比分析**

与PADDPG、PDQN、HPPO、HyAR等基线对比，TART在两大任务的成功率、时间/射击效率等指标均优于所有基线

**⚠️ 局限性**

仅在仿真中验证，资源约束仅为行动预算，未考虑真实能耗、通信或多智能体情境

---

## 53. Robotic Fruits with Tunable Stiffness and Sensing: Towards a Methodology for Developing Realistic Physical Twins of Fruits

**arXiv ID:** 2602.18661 | [PDF](https://arxiv.org/pdf/2602.18661v1)

**作者:** Saitarun Nadipineni `[一作]` (Queen Mary University of London), Thilina Dulantha Lalitharatne `[通讯]` (Queen Mary University of London)

**通讯引用:** 803 | [OpenAlex ID](https://openalex.org/A5067501094)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发并测试了可调硬度的软物理双胞胎模型，用于模拟不同成熟度的奇异果并评估机器人把手的抓握性能

**💡 创新点**

首次将可调硬度软体物理双胞胎引入食品机器人抓取领域，实现硬度可根据果实成熟度实时调节，并通过内部压力传感实现抓取力反馈

**🔧 技术方法**

利用纤维增强气动驱动的硅胶结构、压缩机泵、力/扭矩传感器、Arduino与MATLAB/Python同步采集实现硬度调节与测量

**📊 数据集**

使用真实的奇异果（Hayward品种）在不同成熟阶段（Day 1、5、9）进行压缩试验，获得硬度数据（3 mm压缩下的平均硬度）

**📈 对比分析**

采用压力-硬度一次线性校准（R²=0.9916），在三种目标硬度（2、2.5、3 N/mm）下实现硬度精准度达97–99%；在50次压缩循环中，硬度误差仅为0.56–1.10%，表明高度重复性与可靠性

**⚠️ 局限性**

目前硬度上限低于成熟奇异果的真实硬度，受限于材料弹性、气动压力上限及纤维加固方式，未来需改进材料与驱动方案以逼近自然果实硬度

---

## 54. Capable but Unreliable: Canonical Path Deviation as a Causal Mechanism of Agent Failure in Long-Horizon Tasks

**arXiv ID:** 2602.19008 | [PDF](https://arxiv.org/pdf/2602.19008v1)

**作者:** Wilson Y. Lee `[一作]` `[通讯]` (Independent Researcher), Wilson Y. Lee (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究语言代理在同一任务下多次运行时成功与失败的差异，发现成功主要取决于轨迹是否遵循任务的“规范路径”，而非模型能力缺失。

**💡 创新点**

首次将规范路径（多模型成功运行共通的工具集合）作为可靠性失效的因果机制，证明随机漂移导致成功轨迹逐步偏离并自我强化。

**🔧 技术方法**

采用Jaccard相似度衡量轨迹与规范路径的遵循度，利用自然实验（模型×任务单位内多次随机采样）进行因果识别，并用多重稳健检验与差分异同验证机制。

**📊 数据集**

使用Toolathlon基准数据集，包含22款前沿模型、108个真实工具使用任务，共计约7000条轨迹，其中515个模型×任务单位存在混合成功/失败结果。

**📈 对比分析**

与传统单跑pass@k/Pass@k评估不同，本研究通过同模型多跑的混合单元实现因果估计，发现成功轨迹平均比失败轨迹高0.060 Jaccard，等价于5.3个百分点的成功率提升；对失败轨迹实施中途恢复监控可提升约8.8个百分点。

**⚠️ 局限性**

局限性包括：仅有3次跑测不易获得更精确估计；采用工具集合而非序列可能遗漏顺序信息；对多模态任务的规范路径定义较为保守；实验验证尚未完成，因果识别仍基于观测假设。

---

## 55. Time Series, Vision, and Language: Exploring the Limits of Alignment in Contrastive Representation Spaces

**arXiv ID:** 2602.19367 | [PDF](https://arxiv.org/pdf/2602.19367v1)

**作者:** Pratham Yashwante `[一作]` (University of California San Diego), Rose Yu `[通讯]` (University of California San Diego)

**通讯引用:** 6528 | [OpenAlex ID](https://openalex.org/A5057778679)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在时间序列、视觉图像和文本这三种模态之间，利用对比学习（CLIP风格的InfoNCE）构建了trimodal对齐框架，并系统评估了不同模型规模、信息密度、视觉丰富度以及三模态互补对齐效果。

**💡 创新点**

首次揭示时间序列与视觉图像对齐明显强于与文本对齐的非对称性；发现信息密度提升至一定阈值后对对齐益处衰减；证明图像可作为“语义桥梁”，显著提升弱对齐对（TS–TXT）的对齐；展示预训练的视觉‑文本模型对trimodal对齐的积极影响。

**🔧 技术方法**

对比学习、投影头、InfoNCE损失、Procrustes对齐、CKA、kNN重叠等多种度量；通过冻结预训练编码器并仅训练投影层，保持不同模态的特征可比性。

**📊 数据集**

CaTS-Bench（包含时间序列、绘图和描述）、TRUCE（短时间序列与多种视觉注释）、MIMIC（ECG + 诊断报告）和PTB‑XL（ECG + 德语报告）等四个公开数据集。

**📈 对比分析**

与单模态或双模态对齐相比，trimodal对齐在TS–IMG上维持较高性能，TS–TXT对齐明显受限；信息密度提升至约两倍时对齐指标基本饱和；较大的模型和更大批量、强投影头可进一步提升对齐；视觉注释丰富度提升可显著提高TS–IMG对齐；预训练的VL模型能在较小规模下实现强烈的IMG–TXT对齐。

**⚠️ 局限性**

受限于可用的trimodal数据集规模较小，实验多基于离线评估指标，未验证在下游任务中的实际收益；对齐性能仍受模态表达隐式/显式差异限制，难以突破信息密度阈值；未考虑多语言预训练对非英语文本的影响深度。

---

## 56. Prior Aware Memorization: An Efficient Metric for Distinguishing Memorization from Generalization in Large Language Models

**arXiv ID:** 2602.18733 | [PDF](https://arxiv.org/pdf/2602.18733v1)

**作者:** Trishita Tiwari `[一作]` (Cornell University), G. Edward Suh `[通讯]` (NVIDIA)

**通讯引用:** 11834 | [OpenAlex ID](https://openalex.org/A5024329178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Prior-Aware Memorization (PA) 指标，用来区分 LLM 记忆与泛化；

**💡 创新点**

PA 通过比较条件概率与无条件概率比值，无需训练基线模型，计算成本低；

**🔧 技术方法**

使用贝叶斯分解、蒙特卡洛估计、自动回归语言模型概率；

**📊 数据集**

在 Llama 与 OPT 预训练模型上测试，使用 WikiText、Common Crawl、The Pile 以及 SATML 训练数据提取挑战集；

**📈 对比分析**

与 Counterfactual Memorization 对比，实验显示 PA 与 CF 相关性强；在大模型中，提取可记忆序列中 55-90% 被认为是统计常见，说明传统指标高估了记忆；

**⚠️ 局限性**

PA 受限于 P(s) 可能因近似重复或多次出现而被高估，无法完全区分近似复制与真正记忆；

---

## 57. Systematic Analysis of Coupling Effects on Closed-Loop and Open-Loop Performance in Aerial Continuum Manipulators

**arXiv ID:** 2602.18684 | [PDF](https://arxiv.org/pdf/2602.18684v1)

**作者:** Niloufar Amiri `[一作]`, Farrokh Janabi-Sharifi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了航空连续体操纵器（ACM）的耦合与非耦合动力学模型，并在开放环和闭环视觉跟踪中对两种模型的性能进行系统比较。

**💡 创新点**

首次系统性评估耦合与非耦合模型对精度与计算效率的影响，并提出基于动力学的PD+滑模图像视觉伺服（DPD‑SM‑IBVS）控制器，证明在多数场景下非耦合模型能够保持与耦合模型相当的跟踪精度。

**🔧 技术方法**

采用Euler–Lagrange方法推导动力学，使用片段常曲率（PCC）假设进行几何建模，引入滑模项的图像视觉伺服控制器，并通过Lyapunov分析证明闭环误差的全局最终有界性；实验实现基于Simulink的仿真。

**📊 数据集**

利用多种仿真实验场景（冲击、正弦、外力、自由落体等）进行数据收集，不依赖真实物理数据集。

**📈 对比分析**

通过比较开放环NRMSE误差、闭环图像误差DS（<0.7像素）以及每个采样周期的计算时间（耦合模型32 ms，非耦合模型22 ms）来评估两种模型；结果显示非耦合模型在保持相近精度的同时显著降低计算开销。

**⚠️ 局限性**

局限性在于对长臂、强外力或高速运动场景下耦合效应显著，非耦合模型初期误差较大；对极端动力学条件的鲁棒性仍需进一步实验验证。

---

## 58. Studying the Separability of Visual Channel Pairs in Symbol Maps

**arXiv ID:** 2602.20022 | [PDF](https://arxiv.org/pdf/2602.20022v1)

**作者:** Poorna Talkad Sukumar `[一作]` (New York University), Oded Nov `[通讯]` (New York University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究通过在线实验系统评估了符号地图中四种视觉通道对（颜色×形状、颜色×大小、大小×形状、大小×方向）的可分离度；

**💡 创新点**

创新点在于首次在空间嵌入式地图环境中使用符号地图进行可分离度实验，并揭示了通道分配与通道值水平的非对称影响；

**🔧 技术方法**

使用的技术包括基于D3.js的在线图形生成、重复测量ANOVA与线性混合模型、以及对反应时间的对数变换；

**📊 数据集**

数据集为包含11个西部美国州的符号地图，每个州用三水平的颜色、形状或大小编码两维度数据；

**📈 对比分析**

对比方法为在准确率和对数化反应时间两项指标上进行配对t检验与Tukey多重比较，结果显示颜色×形状组准确率最高、速度最快，大小×方向组最差，其余两组处于中间；

**⚠️ 局限性**

局限性包括仅测试了四个通道组合、未包含单通道基线、任务仅限于识别最高值、并且实验在在线环境中进行，可能受硬件差异影响；

---

## 59. A Context-Aware Knowledge Graph Platform for Stream Processing in Industrial IoT

**arXiv ID:** 2602.19990 | [PDF](https://arxiv.org/pdf/2602.19990v1)

**作者:** Monica Marconi Sciarroni `[一作]` (Polytechnic University of Marche), Emanuele Storti `[通讯]` (Polytechnic University of Marche)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于知识图谱的上下文感知语义平台，用于工业物联网（IIoT/IoE）环境下的流数据采集、处理、存储、监控与查询，并通过语义推理实现动态上下文与角色的访问控制；

**💡 创新点**

创新点在于将语义模型与流处理技术深度融合，利用统一的知识图谱描述设备、流、处理管道、角色与权限；通过SPARQL与SWRL实现对上下文的实时推理，从而支持灵活的流发现、动态权限授予和多租户协作；该方案在Industry 5.0场景下得到验证；

**🔧 技术方法**

技术栈包括：Apache Kafka与Apache Flink用于分布式流处理；RDF/OWL语义模型（SemIoE+IoT-Streams+St）配合SPARQL和SWRL进行推理；Flask+JWT+Redis实现认证与会话管理；Trino用于跨数据库联邦查询；Redis与MongoDB/ PostgreSQL/MySQL支持多种存储后端；

**📊 数据集**

实验使用合成传感器数据（125 B JSON），覆盖MQTT、HTTP、BLE等协议；在不同规模（8 k–5.1 M三元组）和不同DB规模（3.6 k–36 M记录）的知识图谱与存储上进行评估；

**📈 对比分析**

通过对KG查询（Q1‑Q5）、监控服务、查询服务以及流处理的延迟进行测评，发现即使在5.1 M三元组、36 M记录时，监控响应<0.5 s、查询响应<200 ms、流处理延迟<15 ms；KG查询仅占总延迟5%（最高1.44 s用于角色变更时的查询），整体系统可在4核/16 GB commodity 机器上水平扩展；

**⚠️ 局限性**

局限性包括：在角色切换时KG查询仍可产生较大延迟（约1.44 s）；缺乏对管道类型安全与验证的支持；模型建设仍需人工，自动语义关系发现尚未实现；未集成机器学习模型，对极大规模实时推理的极限未知；

---

## 60. Hypersequent Calculi Have Ackermannian Complexity

**arXiv ID:** 2602.19229 | [PDF](https://arxiv.org/pdf/2602.19229v1)

**作者:** A. R. Balasubramanian `[一作]` (Max Planck Institute for Software Systems), Revantha Ramanayake `[通讯]` (Bernoulli Institute)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

证明了所有在无剪切条件下的超序列扩展（如FL_ec和FL_ew）具有Ackermannian的可判定性上界，而不是先前认为的超Ackermannian。

**💡 创新点**

通过利用超序列内部序列间的依赖关系，避免了对幂集的使用，并在弱化情况采用Karp-Miller风格的加速，从而得到最优Ackermannian上界；这一思路显著降低了复杂度。

**🔧 技术方法**

Dickson引理、良序（well‑quasi‑order）与受控坏序列论证、超序列依赖关系分析以及Karp‑Miller加速技术。

**📊 数据集**

本研究为理论分析，不涉及任何实验数据集。

**📈 对比分析**

与之前基于幂集的hyper‑Ackermannian复杂度对比，证明上界被降至Ackermannian；在MTL逻辑上实现了更优的上界，并在FL_ec的情况下证明该上界是最优的。

**⚠️ 局限性**

仅针对无剪切超序列系统，剪切系统的复杂度仍未得到研究；理论结果未在实际实现或实验中验证，且对更广泛子结构逻辑的适用性仍有待进一步探讨。

---

## 61. Test-Time Learning of Causal Structure from Interventional Data

**arXiv ID:** 2602.19131 | [PDF](https://arxiv.org/pdf/2602.19131v1)

**作者:** Wei Chen `[一作]` (Hong Kong University of Science and Technology), Dongmei Zhang `[通讯]` (Microsoft Research)

**通讯引用:** 11431 | [OpenAlex ID](https://openalex.org/A5100331488)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向未知干预的监督因果学习框架，通过测试时训练（Test‑Time Training）自增采样实现对因果结构的识别

**💡 创新点**

创新点在于结合IS‑MCMC后验采样产生自增训练样本，采用两阶段（骨架+v‑结构）监督学习，并统一JCI框架来处理多样化干预场景

**🔧 技术方法**

主要技术包括JCI增广数据/图、IS‑MCMC后验采样、前向采样生成配套数据、两阶段XGBoost分类器和Meek规则推断

**📊 数据集**

使用bnlearn数据库中的14个（半）真实因果图，生成多种干预数据（软/多变量、未知目标）进行评估

**📈 对比分析**

与GIES、IGSP、ENCO、PC、GOLEM等方法比较，平均F1得分在因果发现和干预目标检测上均显著领先（提升≈13‑50%），并保持可扩展性和高效性

**⚠️ 局限性**

局限性包括仅处理离散数据、对JCI假设（外生性、无系统-环境因果）依赖强，且对干预类型和样本规模的鲁棒性仍待进一步验证

---

## 62. Benchmarking Unlearning for Vision Transformers

**arXiv ID:** 2602.20114 | [PDF](https://arxiv.org/pdf/2602.20114v1)

**作者:** Kairan Zhao `[一作]` (University of Warwick), Peter Triantafillou `[通讯]` (University of Warwick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对视觉Transformer（ViT 与 Swin‑T）在机器不学习（MU）任务上进行系统基准评估，探究不同模型容量、数据集规模、连续不学习等因素的影响。

**💡 创新点**

首次在VT上全面基准MU，验证VT与CNN在记忆模式与代理效能上的相似性，并揭示预训练与记忆代理对VT不学习性能的决定性作用；同时给出了最优算法‑代理组合与实践指导。

**🔧 技术方法**

使用 ViT、Swin‑T 两类 Transformer；采用 FT、NegGrad+、SalUn 以及 RUM 框架；结合 Confidence、MaxConf、Entropy、BinaryAccuracy 与 Holdout Retraining 等记忆代理；统一评估指标为 ToW 与 ToW‑MIA。

**📊 数据集**

CIFAR‑10、CIFAR‑100、SVHN 以及 ImageNet‑1K 验证集，覆盖从小规模、低复杂度到大规模、高复杂度的视觉任务。

**📈 对比分析**

对算法、代理、架构、容量、数据集以及单次与连续不学习等维度进行多重对比。结果显示：NegGrad+ 与 Holdout Retraining 组合在 VT 上表现最稳健；Fine‑tune 在 ViT 上效果最佳；Swin‑T 与 NegGrad+ 配合优异；总体上 VT 的 MU 性能可与 CNN 相当甚至更好。

**⚠️ 局限性**

仅聚焦记忆驱动的 MU 方法，未覆盖非记忆驱动或其他不学习范式；实验规模受限，未测试更大模型或多任务场景；代理的选择仍以经验为主，缺乏严格理论保障；未充分验证在更复杂数据分布或攻击模型下的鲁棒性。

---

## 63. Latent Introspection: Models Can Detect Prior Concept Injections

**arXiv ID:** 2602.20031 | [PDF](https://arxiv.org/pdf/2602.20031v1)

**作者:** Theia Pearson-Vogel `[一作]` (ACS Research), Jan Kulveit `[通讯]` (ACS Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在Qwen 32B模型中注入预定义概念并使用logit lens等分析手段，探究了模型对自身内部状态的内省能力；

**💡 创新点**

创新点在于首次在开源大模型中揭示概念注入能激活隐藏的内省信号，并证明提示工程可显著放大检测效果，且通过互信息量化概念识别能力；

**🔧 技术方法**

采用的技术包括：概念注入的steering vectors、KV缓存注入策略、层级logit lens分析、概念识别的互信息计算以及多维提示（框架与信息文档）实验；

**📊 数据集**

实验使用九个手工挑选的概念（猫、面包、爱、恐惧、死亡、真理、创造力、编程、音乐）以及随机种子作为注入目标；

**📈 对比分析**

与无注入基线、控制问题（always‑yes/no、ambiguous、confusing）及多种提示组合对比，检测准确率从0.3%提升至39.2%，概念识别互信息最高1.35 bits（约43%理论上限），在最佳提示下平衡准确率可达88.4%；

**⚠️ 局限性**

局限性包括：结果高度依赖提示与信息文档，未揭示具体机制或因果路径，模型间表现差异大，未系统评估不同规模或训练阶段模型，且只验证了少数概念与模型版本。

---

## 64. Spherical Hermite Maps

**arXiv ID:** 2602.20063 | [PDF](https://arxiv.org/pdf/2602.20063v1)

**作者:** Mohamed Abouagour `[一作]` (Indiana University), Eleftherios Garyfallidis `[通讯]` (Indiana University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于Hermite插值的球面函数查找表（Spherical Hermite Maps），实现用4个纹理采样完成值和梯度的高质量重建。

**💡 创新点**

在立方体贴图上存储函数值与标量化的一阶和混合导数，利用Hermite多项式实现C1连续的二次重建，并在同一次采样得到解析法线，兼顾质量与采样效率。

**🔧 技术方法**

Hermite插值、立方体贴图（带1像素缝隙）、解析球面谐波导数、卷积滤波与纹理压缩、GPU WGSL着色器实现。

**📊 数据集**

使用合成球面谐波几何、真实扩散MRI ODF数据、复杂陨石网格的径向深度图以及多尺度行星高程噪声。

**📈 对比分析**

与双线性采样、16点双三次滤波和快速双三次等基准在PSNR、法线误差、纹理采样量和帧率上进行对比，结果显示Hermite在保留高质量重建的同时，比16点双三次快约1.8倍、法线误差降低9–13%，且在相同存储下优于双线性。

**⚠️ 局限性**

对低分辨率或折叠图（如八面体）可能出现缝隙误差，存储开销为原来的4倍，需要在每个mipmap级别重新计算导数；对极低分辨率或非星形表面支持有限。

---

## 65. Contextual Safety Reasoning and Grounding for Open-World Robots

**arXiv ID:** 2602.19983 | [PDF](https://arxiv.org/pdf/2602.19983v1)

**作者:** Zachary Ravichadran `[一作]` (University of Pennsylvania), George J. Pappas `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 CORE 框架，利用视觉语言模型在线推理语境安全约束，进行空间地面化，并通过控制障碍函数（CBF）实现实时安全控制，适用于未知环境中的开放世界机器人。

**💡 创新点**

创新点在于：①首次将视觉语言模型与安全约束推理结合，实现无需先验地图或安全规范的语境安全推断；②设计结构化安全谓词和链式推理提示，提高安全推理的准确性；③提出基于感知不确定性的概率 CBF 保障，兼顾在线地面化与控制。

**🔧 技术方法**

使用的技术包括：4-bit 量化的 Gemma 3 27B 视觉语言模型；SAM3 开放词汇语义分割；OpenCV 结构化空间运算；CBF 约束求解（CVXPY）；ROS 2 实时控制；DepthAnythingV3 低频深度补全；以及基于感知置信度的安全概率分析。

**📊 数据集**

实验数据集包括：① NVIDIA Isaac Sim 中的仓库、医院和住宅三类仿真环境；② 真实世界的 Boston Dynamics Spot 机器人实验室环境；③ 用于 VLM 评估的 100 张包含不同语境安全场景的图像。

**📈 对比分析**

通过与 Oracle（先验真值安全约束）、No Context（离线 LLM 推理）和 Geometric（仅几何避障）三种基线对比，CORE 在安全任务中实现 96.6% 的成功率，危害任务 93.3%，与 Oracle 接近；仿真中相较于基线显著提升；硬件实验中也达成 86.6% 的成功率，并通过消融实验验证结构化推理和链式推理的重要性。

**⚠️ 局限性**

局限性包括：① VLM 的预测不包含帧级不确定性，导致感知误差难以量化；② 需要较高的计算资源和推理延迟；③ 仅针对控制仿射系统，难以推广至高阶动力学；④ 假设安全初始化和地面化的即时性，真实环境中可能出现误检导致安全失效；⑤ 与规划器耦合不足，无法在高层规划中实时调整。

---

## 66. Parallelism and Adaptivity in Student-Teacher Witnessing

**arXiv ID:** 2602.19934 | [PDF](https://arxiv.org/pdf/2602.19934v1)

**作者:** Ondřej Ježil `[一作]` (Charles University), Dimitrios Tsintsilidas `[通讯]` (University of Warwick)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文通过构造学生-教师游戏与新的证明技术，证明了多种有界算术理论之间的严格分离，并将不可证明性结果扩展到更强的理论中。

**💡 创新点**

创新点在于首次提出两类全新学生-教师游戏的分离定理（适应性与并行性），推导出一套通用的证明见证定理，利用该定理构造了新的长度诱导与有界替换的自洽子结构，从而在更强的理论中实现了电路上界与平均情形下电路下界的不可证明性，并展示了一个兼容两者不可证明性的单一理论。

**🔧 技术方法**

主要技术包括学生-教师游戏框架、Herbrand化与本质扩张、定义按情况闭包的项、可加性与有限递归的运算、设计矩阵与Nisan–Wigderson生成器、以及利用多项式大小电路与非确定性TM的时间复杂度对比。

**📊 数据集**

由于论文为纯理论研究，未使用任何实验数据集。

**📈 对比分析**

通过证明在多项式时间内的学生-教师游戏与复杂度类的保留关系，展示了理论上的计算复杂度提升，理论证明表明所构造的更强理论在可证明性和不可证明性上均优于传统的有界算术理论。

**⚠️ 局限性**

主要局限在于所有结果均依赖于尚未证明的复杂度假设（如PH不崩塌、P/poly≠EXP），且只适用于特定的有界算术层级，尚无法推广到所有有界算术理论或在更通用的非构造性情形下完成。

---

## 67. Efficient endometrial carcinoma screening via cross-modal synthesis and gradient distillation

**arXiv ID:** 2602.19822 | [PDF](https://arxiv.org/pdf/2602.19822v1)

**作者:** Dongjing Shan `[一作]` (Southwest Medical University), Chunxiang Zhang `[通讯]` (Southwest Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了基于结构引导的 MRI‑>超声图像生成（SG‑CycleGAN）与梯度蒸馏轻量化筛查网络（LSNet）的两阶段深度学习框架，用于早期子宫内膜癌（EC）肌层侵犯的无创检测。

**💡 创新点**

创新点：① 在 CycleGAN 中加入模态无关特征提取器（MAFE）和结构一致性损失，实现对解剖结构的精准保持；② 在轻量网络中引入梯度引导稀疏注意力蒸馏，既将高容量教师模型的判别知识迁移到学生网络，又通过梯度重要性自适应压缩注意力，显著降低计算量。

**🔧 技术方法**

主要技术：跨模态无监督图像翻译（SG‑CycleGAN）、结构一致性损失、梯度逆向层、梯度蒸馏与稀疏注意力、MobileViT 轻量骨干、二阶段预训练+微调、Bootstrap 置信区间评估。

**📊 数据集**

使用多中心回顾性数据：7,951 名受试者（651 例 EC，7,300 例正常），共 3,354 例 EC 超声图像和 33,189 例正常图像；MRI 数据 498 名 EC 病例，生成 34,214 张多序列图像用于合成；数据按 8:1:1 训练/验证/测试比例划分，确保中心均衡。

**📈 对比分析**

与基线生成网络（CycleGAN、UNIT、MUNIT、DCLGAN）和轻量分类模型（MobileNet‑V2、EfficientNet、MobileViT）对比。SG‑CycleGAN 在 FID/KID 上分别达到 73.25/0.0636，生成质量最优；LSNet 在真实测试集上敏感度 99.5%，特异性 97.2%，AUC 0.987，GFLOPs 0.289，显著优于 10 位超声医生（敏感度 75.8%，特异性 78.1%，AUC 0.769）。

**⚠️ 局限性**

局限性：① 生成的合成图像可能仍缺乏某些微细病变特征；② 模型在不同设备、不同解剖变异上的泛化需要进一步外部验证；③ 需在临床环境中离线部署，且算法更新需人工审核；④ 受限于训练数据的多中心代表性，仍可能存在样本偏倚。

---

## 68. AdsorbFlow: energy-conditioned flow matching enables fast and realistic adsorbate placement

**arXiv ID:** 2602.19289 | [PDF](https://arxiv.org/pdf/2602.19289v1)

**作者:** Jiangjie Qiu `[一作]` (Tsinghua University), Xiaonan Wang `[通讯]` (Tsinghua University)

**通讯引用:** 22702 | [OpenAlex ID](https://openalex.org/A5100410939)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于能量条件流匹配的确定性生成模型 AdsorbFlow，用于快速高效的吸附物放置。

**💡 创新点**

创新点在于用确定性流匹配替代随机扩散、采用分类器无监督指导的能量条件化、实现仅5步ODE采样。

**🔧 技术方法**

采用了条件流匹配、E3‑equivariant GNN（PaiNN/EquiformerV2）、Heun ODE积分、CFG指导和MLFF预松弛。

**📊 数据集**

使用OC20-Dense数据集（44个内测系统和50个OOD系统）进行评估。

**📈 对比分析**

与 AdsorbDiff、AdsorbML 对比，AdsorbFlow 在 SR@1 34.1%/61.4%（SR@10）且仅5步，性能提升约20倍且异常率最低。

**⚠️ 局限性**

局限在于仅适用于刚体吸附物、未处理内部扭转、对大分子或多吸附物的泛化仍需验证。

---

## 69. Rendezvous and Docking of Mobile Ground Robots for Efficient Transportation Systems

**arXiv ID:** 2602.19862 | [PDF](https://arxiv.org/pdf/2602.19862v1)

**作者:** Lars Fischer `[一作]` (FZI Research Center for Information Technology), Sören Hohmann `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种中央模型预测控制（MPC）方法，实现两台全向轮式移动机器人在任意初始位置的运动中物理耦合与实时转运；

**💡 创新点**

在MPC中显式建模两台机器人动力学与耦合接口，并引入接近策略与约束，首次实现从任意初始状态可靠的运动中物理耦合，并通过软耦合与接近通道控制避免碰撞；

**🔧 技术方法**

使用中央MPC、软变量约束、接近通道约束、全向轮动力学模型，并在CasADi/Ipopt框架下实现；

**📊 数据集**

仅在仿真环境中验证，采用自定义参数（机器人半径0.1 m、磁耦合、距离阈值0.2 m等）构建物流场景；

**📈 对比分析**

与无耦合场景对比：时间缩短19.75%，能耗降低21.04%，行驶距离减少15.52%，展示了显著的效率提升；

**⚠️ 局限性**

仅有仿真验证，未进行实地实验；仅测试全向轮动力学，未涵盖差速或艾克曼驱动；参数需在实际环境中进一步调优。

---

## 70. Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection

**arXiv ID:** 2602.18811 | [PDF](https://arxiv.org/pdf/2602.18811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 71. CLASH: Collision Learning via Augmented Sim-to-real Hybridization to Bridge the Reality Gap

**arXiv ID:** 2602.18707 | [PDF](https://arxiv.org/pdf/2602.18707v1)

**作者:** Haotian He `[一作]` (Peking University), Wenzhao Lian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 568 | [OpenAlex ID](https://openalex.org/A5017678179)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CLASH框架，将仿真中提取的物理先验与极少量真实数据结合，生成高精度的碰撞模拟器并用于机器人控制任务。

**💡 创新点**

创新点在于先用仿真数据蒸馏出可微分的碰撞代理模型，再用仅10条真实碰撞样本微调，从而在保持高精度的同时大幅降低计算成本，实现数据高效、精度与速度兼顾的混合仿真。

**🔧 技术方法**

主要技术包括多层感知器（MLP）作为碰撞代理、仿真蒸馏与梯度微调、可微分系统辨识、混合仿真器集成、基于CMA-ES的模型优化和SAC强化学习。

**📊 数据集**

使用MuJoCo仿真生成10万条碰撞数据，真实实验使用Franka机械臂在不同材质和沙面上收集约110条碰撞样本（其中10条用于训练）。

**📈 对比分析**

与MuJoCo、MJX和直接在真实数据上训练的网络进行对比。CLASH模型在碰撞预测准确率上高于所有基线，模型优化时定位误差降低约35%并将CMA-ES运行时间减半；在强化学习任务中，使用CLASH训练的策略成功率提升约两倍。

**⚠️ 局限性**

局限性包括对扭转摩擦建模不足，导致部分形状（如方块、三角块）姿态误差仍显著；此外需要依赖原始仿真引擎的结构，未能完全实现对新几何形状的无监督泛化。

---

## 72. From Trial by Fire To Sleep Like a Baby: A Lexicon of Anxiety Associations for 20k English Multiword Expressions

**arXiv ID:** 2602.18692 | [PDF](https://arxiv.org/pdf/2602.18692v1)

**作者:** Saif M. Mohammad `[一作]` (National Research Council Canada), Saif M. Mohammad `[通讯]` (National Research Council Canada)

**通讯引用:** 15389 | [OpenAlex ID](https://openalex.org/A5033684482)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了WorryMWEs词典，即大规模（约2万条）多词表达式（MWEs）的焦虑与平静关联评分，首次为MWEs提供情绪规范。

**💡 创新点**

创新点在于：①提出焦虑/平静与MWEs关联的重要性；②通过大规模众包手段构建高可靠度的MWEs情绪词典；③系统分析MWEs在不同类型（成语、名词短语、轻动词结构）中的焦虑分布与非组合性；④将词典与已有单词焦虑词典（WorryWords）整合，形成WorryLex v2。

**🔧 技术方法**

技术方法包括：基于Mechanical Turk的众包任务，使用金标问题进行即时反馈和质量控制；对每个MWE收集至少9名评审的评分（-3至+3）；对评分进行平均和类别化；利用拆分半相关（SHR）评估可靠性；对MWEs类型、语法类别进行分布统计；对大、三、四字短语与构成词情绪关联进行可视化与组合性分析。

**📊 数据集**

使用的数据集为：①已有的10,000条最常见的双字MWEs（从PMID:35867207抽取）+ 10,600条最常见的三字与四字MWEs；②WorryWords词典（44,450个单词）；③MWE concreteness数据集用于获取MWE类型；③SUBTLEX语料库用于获取词的词性信息；④对上述词条进行自制问卷并在MTurk上采样得到评分。

**📈 对比分析**

与以往单词情绪词典相比，WorryMWEs的拆分半相关在0.81–0.95之间，表明高可靠度；在不同n-gram与MWE类型下，焦虑/平静的极化程度被量化（如成语中40%为焦虑，名词短语仅25%）。相比已存在的NRC VAD等多词情绪词典，WorryMWEs覆盖范围更广、情绪维度更细（焦虑/平静），并能揭示非组合性现象（如绝大多数焦虑MWEs由中性词构成）。

**⚠️ 局限性**

局限性包括：①词典仅覆盖标准美国英语，缺乏跨文化和跨语种的多样性；②MTurk参与者样本不代表整个英语使用者，可能存在性别、地区等偏倚；③注释基于直觉而非临床诊断，不能用于诊断焦虑障碍；④部分MWEs可能因历史种族/偏见产生的语义而在使用上有争议；⑤大多数MWEs为高频短语，低频表达被忽视。

---

## 73. Beyond a Single Extractor: Re-thinking HTML-to-Text Extraction for LLM Pretraining

**arXiv ID:** 2602.19548 | [PDF](https://arxiv.org/pdf/2602.19548v1)

**作者:** Jeffrey Li `[一作]` (University of Washington), Fartash Faghri `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究不同 HTML 文本提取器对 Web 规模 LLM 预训练数据集的影响，并提出并行使用多种提取器以提高文本覆盖率和模型性能

**💡 创新点**

发现单一提取器导致大规模网页的稀缺覆盖；并通过对比实验证明多提取器联合可显著提升 token 产量（最高 71%）且对标准评测保持不降，且对表格和代码等结构化内容影响更大

**🔧 技术方法**

采用三种主流规则基提取器（Boilerpipe、Justext、Readability 等）在 Common Crawl 上并行提取；使用 DCLM 的过滤、去重与 WikiTableQuestions、HumanEval、MMLU 等下游评测；对表格/代码页面使用自定义过滤器与分类模型

**📊 数据集**

主要使用 Common Crawl WARC 语料；针对表格与代码分别构建 CC‑Tables 与 CC‑Code 子集；同时对比 DCLM 基线与 Llama‑3 等公开模型

**📈 对比分析**

比较方法：在相同后处理管道下分别使用单提取器与多提取器 union，并对 token 数、MMLU、WikiTQ、HumanEval 等指标进行评测。实验表明 union 数据集在 token 产量上提升 58–71%，在表格任务上提升 10–12%，在代码任务上提升 1–4%，而标准语言任务几乎不变

**⚠️ 局限性**

仅评估了三种规则基提取器，未探索基于内容自适应选择或深度学习提取器；对表格/代码过滤策略仍不够完善；多提取器带来的重复与去重效果未彻底量化；实验规模受算力限制，缺乏跨规模多次复现

---

## 74. FUSAR-GPT : A Spatiotemporal Feature-Embedded and Two-Stage Decoupled Visual Language Model for SAR Imagery

**arXiv ID:** 2602.19190 | [PDF](https://arxiv.org/pdf/2602.19190v1)

**作者:** Xiaokun Zhang `[一作]` (Fudan University), Haipeng Wang `[通讯]` (Fudan University)

**通讯引用:** 5973 | [OpenAlex ID](https://openalex.org/A5100405762)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了首个SAR图像-文本-特征三元组数据集，并提出FUSAR-GPT模型，专门针对SAR图像的视觉‑语言理解任务。

**💡 创新点**

创新点包括：①使用AlphaEarth Foundations（AEF）多源时空先验作为地理知识库；②通过时空锚点将AEF嵌入SAR图像；③设计Token‑wise Linear Modulation（TLM）模块实现细粒度的先验注入；④采用两阶段分离的SFT策略，将知识注入与任务推理解耦。

**🔧 技术方法**

技术主要包括：多源时空特征提取、时空锚点对齐、TLM融合模块、两阶段SFT微调、基于Qwen2.5-VL-7B的视觉‑语言架构。

**📊 数据集**

使用了自建的FUSAR‑GEOVL‑1M（10k SAR‑文本‑AEF三元组）和FUSAR‑GPT（2k带目标标注的SAR图像）数据集。

**📈 对比分析**

与主流VLM（Qwen、LLaVA、InternVL等）在目标计数、空间定位、分类、检测四个任务上对比，FUSAR‑GPT平均提升约10%‑12%，在计数准确率、定位Acc@100、检测F1等指标上均显著优于基线。

**⚠️ 局限性**

局限性：仍主要聚焦SAR域，对其他遥感模态（光学、LiDAR等）的泛化待验证；依赖高质量的AEF先验，若先验缺失或不精确可能影响性能；两阶段训练复杂度较高，需要大量标注与计算资源。

---

## 75. Transformers for dynamical systems learn transfer operators in-context

**arXiv ID:** 2602.18679 | [PDF](https://arxiv.org/pdf/2602.18679v1)

**作者:** Anthony Bao `[一作]` (University of Texas at Austin), William Gilpin `[通讯]` (University of Texas at Austin)

**通讯引用:** 1525 | [OpenAlex ID](https://openalex.org/A5090794745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

训练了一种两层单头Transformer在单一动力学系统的单变量时间序列上，并在未见的不同动力学系统上评估其预测性能。

**💡 创新点**

首次揭示Transformer在推理阶段通过自适应时间延迟嵌入和在上下文中学习传递算子，实现对未见系统的零样本预测和OOD泛化的机制。

**🔧 技术方法**

使用最小GPT‑style Transformer、相对位置编码、注意力回放、Takens延迟嵌入、Ulam方法估计Perron‑Frobenius算子等技术。

**📊 数据集**

利用100对从大型ODE数据库中抽取的动力学系统（以Lorenz‑96等可调维度系统为主）进行实验。

**📈 对比分析**

与均值回归、k阶马尔可夫链直接在测试系统上训练等基线对比，Transformer在测试ID和OOD上显著优于基线，并在不同延迟阶数和吸引子维度下展现与真实动力学相符的特征。

**⚠️ 局限性**

局限在于仅使用单变量轨迹训练，缺乏对多变量系统的直接建模；对高维吸引子仍需较长上下文长度，限制了在复杂系统上的实时预测。

---

## 76. Rodent-Bench

**arXiv ID:** 2602.18540 | [PDF](https://arxiv.org/pdf/2602.18540v1)

**作者:** Thomas Heap `[一作]` (University of Bristol), Adriana Casado Rodriguez `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Rodent-Bench 基准，评估多模态大语言模型（MLLM）在啮齿动物行为视频注释的能力；

**💡 创新点**

首次为科学行为注释设计短长两版基准，统一评估指标并揭示现有 MLLM 的显著局限；

**🔧 技术方法**

使用 Gemini-2.5-Pro、Gemini-2.5-Flash 与 Qwen-VL-Max 等最新 MLLM 进行零样本视频注释；

**📊 数据集**

采集 CalMS21、Rodent Grooming、Mouse-Ventral（1/2）、Scratch-AID 与 Freezing 等公开与私有行为视频数据集；

**📈 对比分析**

通过秒级准确率、宏 F1、mAP、互信息、MCC 等指标对比模型性能；Gemini‑2.5‑Pro 最高，但整体性能仍远低于实际研究需求；

**⚠️ 局限性**

限制包括标注不一致、缺乏人工基准、仅零样本评估、JSON 格式化错误、模型视频长度限制以及数据集多样性不足等。

---

## 77. INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic

**arXiv ID:** 2602.18956 | [PDF](https://arxiv.org/pdf/2602.18956v1)

**作者:** Serafim Batzoglou `[一作]` `[通讯]`, Serafim Batzoglou

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了有限结构概念合成的三种全可验证任务（FullObs、CI、EC），并提出了可控难度的 INDUCTION 基准。

**💡 创新点**

首次将完整可检验的 FOL 概念合成拆解为三种模式，构建对抗生成和预算化评估来衡量公式简洁性与泛化能力。

**🔧 技术方法**

使用 SMT/模型检查对公式进行精确验证；通过版本空间追踪、trap 机制生成实例；用 AST 大小、量化深度等度量评估 bloat。

**📊 数据集**

v1 INDUCTION 数据集，包含 375 个 FullObs、200 个 CI 与 200 个 EC 例子，所有实例基于小签名 {P,Q,R,S}，分层控制难度。

**📈 对比分析**

通过准确率、覆盖率、预算准确率 Acc@+Δ 和 bloat 率等指标进行比较；结果显示 GPT‑5.2 等模型虽达高无穷大准确率，但 bloat 率高；低 bloat 公式在 held‑out 泛化显著更好；CI 与 EC 亦表现出类似的简洁性差距。

**⚠️ 局限性**

局限在于仅考虑小签名和有限域，模板化概念导致偏见；存在非唯一解、对提示敏感、易于基准过拟合；EC 的存在性语义与生成器可能产生缺陷。

---

## 78. Learning Adaptive Perturbation-Conditioned Contexts for Robust Transcriptional Response Prediction

**arXiv ID:** 2602.18885 | [PDF](https://arxiv.org/pdf/2602.18885v1)

**作者:** Yinhua Piao `[一作]` (KAIST), Sungsoo Ahn `[通讯]` (KAIST)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出AdaPert框架，利用扰动条件子图与自适应学习解决基因扰动预测中的平均收缩问题

**💡 创新点**

通过从统一知识图中提取扰动特异子图并结合三项损失（重构、非DEG鲁棒、对齐）实现稀疏信号与噪声的显式分离

**🔧 技术方法**

使用图神经网络、语言模型嵌入、Gumbel-Softmax采样及自适应正则化等技术

**📊 数据集**

在K562.Replogle与RPE1.Replogle两大单细胞CRISPR扰动数据集上进行评估

**📈 对比分析**

相较于无图与基于图的基线方法，AdaPert在Pearson-Δ、PDS及DEG相关指标（DES@K、Spearman等）上均显著提升，尤其在小效应扰动中表现优异

**⚠️ 局限性**

对知识图质量依赖较大，且超参数（如λ_non）需针对扰动效应大小进行调优，极大扰动时正则化过强可能抑制信号

---

## 79. AuditoryHuM: Auditory Scene Label Generation and Clustering using Human-MLLM Collaboration

**arXiv ID:** 2602.19409 | [PDF](https://arxiv.org/pdf/2602.19409v1)

**作者:** Henry Zhong `[一作]` (Macquarie University), Richard F. Lyon `[通讯]` (Google Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了AuditoryHuM框架，利用多模态大语言模型自动生成听觉场景标签，并通过Human-CLAP评估标签与音频的匹配度，结合人工干预和聚类算法构建可用于边缘设备的听觉场景标签体系。

**💡 创新点**

创新点在于：①将MLLM生成的标签与零射击音频-文本相似度评估相结合，量化标签质量并通过Human-CLAP筛选最不匹配样本；②引入带惩罚项的调整轮廓分数，平衡聚类细粒度与语义连贯性；③通过标签分布向量自动生成集成标签，实现可解释的标准化标签集。

**🔧 技术方法**

采用多模态大语言模型（Gemma、Qwen系列），Human-CLAP零射击技术评估标签一致性，Sentence Transformers进行文本嵌入，凝聚聚类/谱聚类做标签聚类，t‑SNE可视化，AIC导出的λ参数调节聚类惩罚。

**📊 数据集**

实验数据集包括ADVANCE、AHEAD-DS和TAU 2019三大不同领域的听觉场景音频数据集。

**📈 对比分析**

通过与三种MLLM、两种CLAP实现、不同文本清洗方法、句子Transformer版本以及聚类算法的对比，Qwen 2.5 Omni 3B与Human‑CLAP在μ_c、μ_1%等指标上表现最佳；聚类得到的调整轮廓分数峰值最高，表明标签质量高且聚类紧凑。

**⚠️ 局限性**

局限性包括：零射击模型对复杂多源音频的匹配度仍有限；MLLM易产生幻觉（非英语词、长句、特殊字符），需大量清洗；人工审核仅覆盖少量样本，难以覆盖全部误差；当前实现仅支持单标签，无法处理多标签场景。

---

## 80. Git Takes Two: Split-View Awareness for Collaborative Learning of Distributed Workflows in Git

**arXiv ID:** 2602.19714 | [PDF](https://arxiv.org/pdf/2602.19714v1)

**作者:** Joel Bucher `[一作]` (ETH Zurich), April Yi Wang `[通讯]` (ETH Zurich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一款基于浏览器的 Git 学习平台 GitAcademy，通过 split‑view 实时镜像两位学习者的本地仓库，支持协作式训练；

**💡 创新点**

其创新点在于将 Git 的分布式状态可视化与即时同步展示于双视图，提供持续的工作空间意识和即时协作教学的原型；

**🔧 技术方法**

实现技术包括前端 React、后端 Docker 容器化 Git 环境、WebSocket 实时同步、终端与文件编辑器的实时镜像；

**📊 数据集**

实验使用了 26 名参与者组成 13 对，完成两套练习（Hangman 与 Arctic）并收集行为日志、问卷与访谈数据；

**📈 对比分析**

与单视图基线比较显示，split‑view 并未显著提升任务完成率，但显著降低了主观认知负荷，提升了社交存在感和教学/跟随行为，整体效能优于基线；

**⚠️ 局限性**

局限性包括实验规模小、时间短、仅在实验室环境下进行、未评估长期学习效果，以及部分 UI 仍需改进（如鼠标指针等细节）。

---

## 81. IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning

**arXiv ID:** 2602.19049 | [PDF](https://arxiv.org/pdf/2602.19049v1)

**作者:** Yinhan He `[一作]` (University of Virginia), Jundong Li `[通讯]` (University of Virginia)

**通讯引用:** 13500 | [OpenAlex ID](https://openalex.org/A5029588473)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在GRPO的优势函数中引入基于条件互信息的token级优势，结合早退出式MI估计和KV-cache+chunk推理加速，对LLM进行后训练，实现推理步骤更高效、更加简洁。

**💡 创新点**

创新点在于①使用条件互信息量化每个token对最终答案的贡献，②在优势分配中加入探索调节项，③提出早退出+KV-cache+chunk式的高效MI估计方案，③从理论上证明可降低推理长度且不降低准确率。

**🔧 技术方法**

采用强化学习后训练（GRPO），条件互信息估计（早退出式）、KV-cache预加载、chunkwise forward、KL正则化和token级优势标准化。

**📊 数据集**

在数学推理数据集GSM8K、MATH-500、DAPO-Math-17k上进行实验，并在非数学推理任务上验证通用性。

**📈 对比分析**

与Dapo、GFPO、GTPO、S-GRPO等基线对比，IAPO在Pass@k最高、长度最短，Token Efficiency（Ratio@k）在所有设置中均居首或接近首位，推理长度平均降低约36%，推理时间下降11%。

**⚠️ 局限性**

局限性包括：①需要额外计算MI估计，尽管已做加速但仍比纯序列优势略高；②对超大模型或极长推理序列的计算开销与内存仍有挑战；③在非数学推理场景下性能提升相对有限。

---

## 82. A Patient-Specific Digital Twin for Adaptive Radiotherapy of Non-Small Cell Lung Cancer

**arXiv ID:** 2602.18496 | [PDF](https://arxiv.org/pdf/2602.18496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 83. Laplacian Multi-scale Flow Matching for Generative Modeling

**arXiv ID:** 2602.19461 | [PDF](https://arxiv.org/pdf/2602.19461v1)

**作者:** Zelin Zhao `[一作]` (Georgia Institute of Technology), Yongxin Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7757 | [OpenAlex ID](https://openalex.org/A5066940107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种LapFlow框架，将图像分解为拉普拉斯金字塔残差，并通过混合Transformer（MoT）在多尺度上并行进行流匹配，直接在潜空间训练并生成高分辨率图像。

**💡 创新点**

创新点在于：①使用拉普拉斯金字塔实现多尺度并行生成，避免了传统级联式需要逐级去噪的桥接过程；②采用MoT架构并加入因果注意力，保证从低分辨率到高分辨率的信息流自然而高效；③引入分段时间点的多尺度去噪调度和渐进式训练，进一步提升生成质量与采样速度。

**🔧 技术方法**

技术手段包括：流匹配（Conditional Flow Matching）在潜空间的实现；拉普拉斯金字塔分解与重构；混合Transformer（MoT）与全局多头自注意力；因果掩码以实现层级信息流；使用线性或GVP噪声调度；ODE求解器（torchdiffeq）进行并行采样；以及EQVAE等变分自编码器来获取更统一的潜空间。

**📊 数据集**

数据集：CelebA-HQ（256×256、512×512、1024×1024）用于无条件生成；ImageNet（256×256）用于类别条件生成。

**📈 对比分析**

与单尺度LFM、Pyramidal Flow、EdifyImage、Relay Diffusion等基线相比，LapFlow在CelebA-HQ 256×256实现FID 3.53（低于LFM 5.26、Pyramidal Flow 11.20），512×512 4.04（vs. LFM 6.35）、1024×1024 5.51（vs. LFM 8.12）；在ImageNet 256×256，B/2模型Fidelity 36.50（LFM 39.40、Pyramidal Flow 39.40），XL/2模型Fidelity 14.38（LFM 17.10）；同时GFLOPs、NFE和采样时间均显著降低，体现了更高效的并行采样。

**⚠️ 局限性**

局限性包括：①需要精心设计的多尺度时间段和噪声调度，超参数敏感；②目前仅在潜空间实现，缺乏直接图像空间的实验；③多尺度数目增多会导致模型复杂度上升，训练成本上升；④在极高分辨率或不同领域（如视频、3D）中的泛化尚未验证；⑤对VAE结构（如EQVAE）的依赖可能限制迁移到其他数据集。

---

## 84. AdaptStress: Online Adaptive Learning for Interpretable and Personalized Stress Prediction Using Multivariate and Sparse Physiological Signals

**arXiv ID:** 2602.18521 | [PDF](https://arxiv.org/pdf/2602.18521v1)

**作者:** Xueyi Wang `[一作]` (University of Groningen), Elisabeth Wilhelm `[通讯]` (University of Groningen)

**通讯引用:** 869 | [OpenAlex ID](https://openalex.org/A5088689866)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了一套可解释且个性化的连续压力预测框架，利用可穿戴智能手表采集的多变量生理信号进行天级预测，支持提前干预。

**💡 创新点**

创新点在于将稀疏时间序列多变量特征与领域先验结合，构建Transformer‑based自适应注意力网络，并实现域适应和可选的测试时适配，以实现跨个体的稳健预测与可解释性。

**🔧 技术方法**

技术手段包括：多层Transformer编码器、特征级注意力、对抗式域分类器、基于MMD/KL/方差的分布偏移检测、伪标签熵最小化测试时适配、SHAP解释以及多窗口滑动窗口时间序列学习。

**📊 数据集**

使用了16名受试者（平均监测时长10‑15周）在Garmin Vivosmart 5上收集的15个筛选后特征（心率变异、呼吸、睡眠、活动等），共计约1万多日的数据。

**📈 对比分析**

与Informer、TimesNet、PatchTST、CNN、LSTM、CNN‑LSTM等基线模型对比，最优H5‑P1配置下MSE = 0.053、MAE = 0.190、RMSE = 0.226，较最佳基线提升约20‑36%（MAE/RMSE），并在所有窗口设置下保持更低方差和更高趋势方向准确率。

**⚠️ 局限性**

局限性包括样本量仅16人、人口学多样性不足、缺少外部环境/社会情境变量、部分特征缺失率高且依赖手表算法生成的压力分数，未来需扩展至更大、异质人群并结合干预实验验证。

---

## 85. LLM-enabled Applications Require System-Level Threat Monitoring

**arXiv ID:** 2602.19844 | [PDF](https://arxiv.org/pdf/2602.19844v1)

**作者:** Yedi Zhang `[一作]` (National University of Singapore), Jun Sun `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

阐述LLM应用在部署后面临的安全与可靠性挑战，并提出将系统级威胁监控作为保障可靠运行的核心手段；

**💡 创新点**

将LLM应用的安全威胁按14类进行细致分类，并给出针对每类威胁的监控指标与日志体系，形成一套完整的系统级监控框架；

**🔧 技术方法**

采用威胁分类、攻击向量映射、监控工件定义、审计日志整合等方法，结合Model Context Protocol（MCP）实现端到端的安全事件检测；

**📊 数据集**

未使用特定数据集进行实验，而是基于现有的LLM应用工作流和安全文献构建理论模型；

**📈 对比分析**

未提供实验比较与性能评估，本文主要提出设计思路与技术路线，后续工作计划在实际部署环境中验证该框架；

**⚠️ 局限性**

缺乏实证验证与性能指标，所提框架的可扩展性、实现成本与误报率等关键指标仍待进一步研究。

---

## 86. Cooperation After the Algorithm: Designing Human-AI Coexistence Beyond the Illusion of Collaboration

**arXiv ID:** 2602.19629 | [PDF](https://arxiv.org/pdf/2602.19629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 87. FinSight-Net:A Physics-Aware Decoupled Network with Frequency-Domain Compensation for Underwater Fish Detection in Smart Aquaculture

**arXiv ID:** 2602.19437 | [PDF](https://arxiv.org/pdf/2602.19437v1)

**作者:** Jinsong Yang `[一作]` (Dalian Ocean University), Hong Yu `[通讯]` (Dalian Ocean University)

**通讯引用:** 57599 | [OpenAlex ID](https://openalex.org/A5100403938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了FinSight‑Net，一种针对水下鱼类检测的轻量级、物理感知目标检测网络；

**💡 创新点**

创新点在于设计了多尺度解耦双流处理（MS‑DDSP）瓶颈来补偿频域信息损失，以及高效路径聚合特征金字塔（EPA‑FPN）来恢复高频细节；

**🔧 技术方法**

技术手段包括物理感知的并行卷积分支、软注意力权重、CSPDarknet骨干、长距离跳连与路径剪枝的FPN结构；

**📊 数据集**

使用了DeepFish、AquaFishSet以及自研的UW‑BlurredFish三大数据集进行训练与评估；

**📈 对比分析**

与YOLO系列、RT‑DETR等SOTA方法对比，在UW‑BlurredFish上mAP达92.8%（比YOLOv11s高4.8%），参数仅6.7M，推理时间8.6 ms，显著提升检测精度与实时性；

**⚠️ 局限性**

局限性包括对动态视角和长期运动一致性缺乏充分验证，且对极端海流或多光源场景的鲁棒性尚待进一步研究。

---

## 88. NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning

**arXiv ID:** 2602.20119 | [PDF](https://arxiv.org/pdf/2602.20119v1)

**作者:** Jiahui Fu `[一作]` (Robotics and AI Institute), George Konidaris `[通讯]` (Brown University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 NovaPlan，基于闭环视频语言规划的零样本长时程操纵框架，将 VLM 与视频生成结合，并通过对象流和手部流的双轨提取实现稳健执行与错误恢复。

**💡 创新点**

创新点包括：将视频生成视为可查询的验证环节，多候选视频按流与语义一致性筛选；引入混合对象/手部流切换与几何校准，将生成视频与机器人运动对齐；实现闭环验证+自我恢复的高层规划。

**🔧 技术方法**

使用技术包括：VLM（GPT‑5.2）任务分解与验证；视频生成模型 Wan 2.2 / Veo 3.1；深度与几何估计（MoGe2、CVD、SAM3、TAPIP3D）；手部姿态估计（HaMeR）与抓取提议网络；混合流提取与几何对齐。

**📊 数据集**

主要数据集：自定义长时程任务集（四层堆叠、颜色分拣、隐藏物体搜索）及 Functional Manipulation Benchmark（FMB）中的多物体多阶段装配任务；视频生成与验证基于公开 VLM/视频生成模型。

**📈 对比分析**

与 NovaFlow、π_0.5、MOKA 等基线对比，NovaPlan 在三类长时程任务中成功率更高（如四层堆叠 7/10，颜色分拣与隐藏搜索全部成功），在 FMB 上能完成复杂装配并实现非抓握错误恢复，整体性能优于基线。

**⚠️ 局限性**

局限性：单视角视频生成对不规则物体的可行性低；深度估计误差导致流提取不稳定；手部流在强遮挡或小幅动作时表现不佳；错误恢复受限于生成的物理可行视频；抓取提议网络对复杂形状表现欠佳。

---

## 89. Token-UNet: A New Case for Transformers Integration in Efficient and Interpretable 3D UNets for Brain Imaging Segmentation

**arXiv ID:** 2602.20008 | [PDF](https://arxiv.org/pdf/2602.20008v1)

**作者:** Louis Fabrice Tshimanga `[一作]`, Manfredo Atzori `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了 Token-UNet，一种将 TokenLearner、TokenFuser 以及可选轻量 Transformer 嵌入传统 UNet 的 3D 医学图像分割框架，在保持高分割精度的同时显著降低计算资源占用。

**💡 创新点**

创新点包括：① 通过 TokenLearner 将高维特征图压缩为固定数量可解释的语义 token，令 Transformer 输入量不随分辨率增长；② 通过 TokenFuser 将 token 再映射回空间，生成可视化的注意力图；③ 在此框架中可插入轻量 Transformer，进一步提升全局建模能力，兼顾效率与精度。

**🔧 技术方法**

使用技术：3D 卷积 UNet（残差块、加法 skip 连接）、TokenLearner/TokenFuser（基于 MLP 的 token 化与解码）、轻量 Transformer（4 层多头自注意力+MLP）、MONAI/PyTorch 框架、梯度累积、滑窗推理。

**📊 数据集**

数据集为 FeTS 2022（BraTS 2022 子集）：1251 份多模态 MRI（T1、T1ce、T2、T2‑FLAIR），统一尺寸 240×240×155，分割标签包括 WT、TC、AT。

**📈 对比分析**

在 5 折交叉验证中与 UNet、UNet**、SwinUNETR 对比：Token-UNet（无 Transformer）和 Token-UNet+Transformer 分别达到 87.21%±0.35% 与 87.34%±0.32% Dice，均优于 SwinUNETR 的 86.75%±0.19%；同时参数量降至 SwinUNETR 的 33%，显存占用降至 10%，推理时间降至 35%。

**⚠️ 局限性**

局限性包括：① 固定 token 数量可能限制模型表达能力；② 对不同体积、分辨率的适应仍需更多实验；③ Transformer 规模受限，进一步提升仍需研究；④ 需要更多自监督/预训练验证其通用性；⑤ 对失败案例的可解释性仍需深入探讨。

---

## 90. Federated Learning Playground

**arXiv ID:** 2602.19489 | [PDF](https://arxiv.org/pdf/2602.19489v1)

**作者:** Bryan Guanrong Shan `[一作]` (Nanyang Technological University), Han Yu `[通讯]` (Nanyang Technological University)

**通讯引用:** 75698 | [OpenAlex ID](https://openalex.org/A5100462720)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

创建了一个交互式浏览器平台——Federated Learning Playground，允许用户通过可视化界面无需编写代码即可体验并调参联邦学习过程；

**💡 创新点**

在TensorFlow Playground的基础上加入了多种FL聚合算法（FedAvg、FedProx、FedAdam、SCAFFOLD）、非IID数据采样、差分隐私、聚类等功能，并实现了实时客户端与全局模型可视化，降低了联邦学习的入门门槛；

**🔧 技术方法**

利用TensorFlow.js实现前端模型训练；使用Dirichlet分布实现非IID数据划分；实现FedAvg、FedProx、FedAdam、SCAFFOLD聚合；采用DP‑SGD实现差分隐私；使用k‑means聚类客户端更新；通过权重向量化与单遍计算实现高效本地与服务器端运算；

**📊 数据集**

主要使用人工合成的二维分类数据（Tiny MLP），不依赖大型真实数据集；

**📈 对比分析**

通过在同一界面同时展示联邦学习与传统集中式训练的可视化结果（客户端参与度、通信成本、客户端损失分布、收敛速率等），帮助用户直观比较两种模式；性能方面未给出数值评估，侧重于可视化示范与教育效果；

**⚠️ 局限性**

局限性包括：仅使用小型合成数据和极简网络，缺乏对真实大规模任务的评估；实现仅覆盖横向联邦学习；缺少可信任与公平性功能；整体仅为教学演示，未深入探索算法细节与系统层面优化。

---

## 91. Shifting Engagement With Cybersecurity: How People Discover and Share Cybersecurity Content at Work and at Home

**arXiv ID:** 2602.19695 | [PDF](https://arxiv.org/pdf/2602.19695v1)

**作者:** William Seymour `[一作]` (Kings College London), Martin J. Kraemer `[通讯]` (KnowBe4)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对1200名来自英国、美国、法国和德国的受访者进行在线调查，探索他们在工作场所和家庭中如何发现、分享网络安全内容，并考察工作场所安全培训（SAT）对其信息消费和分享行为的影响。

**💡 创新点**

首次将雇主视为网络安全信息的主要来源，对比以往仅关注新闻媒体的研究；揭示SAT导致受访者将关注点从家庭转向工作，并过度强调钓鱼攻击；提出将工作场所培训扩展至个人场景以提升整体安全意识的建议。

**🔧 技术方法**

采用问卷设计（基于Das等人方法），收集最近一次网络安全内容消费及其来源、分享动机等信息；使用Python、R进行多项式逻辑回归、卡方检验、点双相关以及Benjamini–Hochberg多重比较校正；通过内容编码构建威胁类型代码本。

**📊 数据集**

约1095名有效受访者（年龄均值36.6岁）来自四国，覆盖互联网使用频率、就业状态、IUIPC和SA‑6安全/隐私态度评分；对比先前Das等人研究中的信息来源分布。

**📈 对比分析**

与Das等人研究进行横向比较，利用卡方检验评估信息来源差异（p<0.001），多项式逻辑回归检验SAT对信息来源的影响（受雇者更倾向于接受雇主提供内容，系数≈0.27）。结果显示雇主成为主要信息来源，SAT显著降低受访者在工作之外分享意愿，整体相关性显著但效应大小有限。

**⚠️ 局限性**

局限性包括：受访者回忆受提示影响，可能导致内容回忆偏差；仅为一次性快照，无法评估信息分享频率与持续影响；自我报告数据可能受社会期望偏差；样本主要为网络使用者，难以推广至不常使用互联网的群体；未能证实因果关系，仅能观察相关性。

---

## 92. Spectral Phase Encoding for Quantum Kernel Methods

**arXiv ID:** 2602.19644 | [PDF](https://arxiv.org/pdf/2602.19644v1)

**作者:** Pablo Herrero Gómez `[一作]` (University of Alicante), Higinio Mora Mora `[通讯]` (University of Alicante)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了在量子核方法中加入离散傅里叶变换前置的谱相位编码（SPE），并评估其在受加性噪声扰动下的鲁棒性。

**💡 创新点**

创新点在于将经典频域预处理与对角相位量子嵌入相结合，形成低深度、硬件友好的特征映射，并通过降噪斜率比较展示了其在噪声条件下的优越稳定性。

**🔧 技术方法**

采用DFT预处理、对角相位编码、SWAP测试重叠估计、支持向量机训练、数据固定效应回归和野集群引导自助抽样等技术。

**📊 数据集**

使用20个常见图像/医疗/面部等公开数据集（灰度化、32×32尺寸），并在每个数据集上添加不同幅度的高斯噪声。

**📈 对比分析**

与QK-PCA、QK-RP以及SVM-Linear、SVM-RBF 进行对比，结果显示QK-DFT在大多数噪声水平下保持最高或竞争性准确率，且其降噪斜率最小，优于其他量子和经典基准。

**⚠️ 局限性**

局限在于只考察了加性高斯噪声、固定的DFT截断和单一对角编码，未探讨更复杂的噪声模型、可学习的频谱选择以及大规模量子硬件的深度和资源约束。

---

## 93. Beyond Privacy Labels: How Users Perceive Different Information Sources for Understanding App's Privacy Practices

**arXiv ID:** 2602.19352 | [PDF](https://arxiv.org/pdf/2602.19352v1)

**作者:** Varun Shiri `[一作]` (Polytechnique Montreal), Jinghui Cheng `[通讯]` (Polytechnique Montreal)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个 Chrome 插件原型，扩展 Google Play 的隐私标签页面，添加了四种信息源（隐私政策摘要、摘录、针对隐私的应用评论和 ToS;DR 社区评估），并通过用户研究评估其对用户理解隐私实践的帮助

**💡 创新点**

提出了在过度简化的隐私标签和冗长的隐私政策之间的中间方案，即将多源信息整合到隐私标签中，探讨不同来源对用户信任与可用性的影响

**🔧 技术方法**

采用 ChatGPT 生成隐私政策摘要、手工提取政策摘录、从 Google Play 获取应用评论、使用 ToS;DR API 读取社区评估；实现 Chrome 插件界面展示与交互

**📊 数据集**

使用了 Snapchat 的隐私政策文本、Google Play 上该应用的用户评论以及对应的 ToS;DR 评估结果作为样本数据

**📈 对比分析**

通过定性主题分析对 10 名参与者的访谈记录进行评估，比较四种信息源在易用性、可信度、信息量等维度的用户感知，未给出量化性能指标，但发现摘要易懂但可信度低，摘录可信但信息量大，评论易于共情但可疑，ToS;DR 直观但不被广泛信任

**⚠️ 局限性**

样本规模有限（10 人），仅针对单个应用（Snapchat），信息源提取为手工或有限自动化，缺乏跨平台或多应用的验证，研究结果主要是主观感知，未进行客观可测的效果评估

---

## 94. KUDA: Knowledge Unlearning by Deviating Representation for Large Language Models

**arXiv ID:** 2602.19275 | [PDF](https://arxiv.org/pdf/2602.19275v1)

**作者:** Ce Fang `[一作]` (Zhejiang University), Yunjun Gao `[通讯]` (Zhejiang University)

**通讯引用:** 5352 | [OpenAlex ID](https://openalex.org/A5006238145)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于表示偏移的LLM遗忘方法（Knowledge Unlearning by Deviating Representation，简称KUDA），通过在已识别的知识存储层对FFN参数进行有选择的更新，实现对目标知识的精确去除，同时保持模型的生成与推理能力。

**💡 创新点**

创新点：①利用因果追踪与因果效应指标实现对FFN层的细粒度定位；②通过滑动窗口策略挑选遗忘层；③设计表示偏移（cosine complement）损失实现对目标知识表示的精准偏移；④采用松弛零空间投影（Relaxation Null‑Space Projection）消除遗忘与保留梯度冲突；⑤提出两阶段超参调优方案，将高维联合搜索拆分为一次性确定稳定边界与一次性调节遗忘强度。

**🔧 技术方法**

核心技术包括因果追踪、因果效应度量、滑动窗口选择、表示偏移损失、松弛零空间投影、随机比例采样、梯度对齐分析、逆遗忘温度与零空间阈值调参。

**📊 数据集**

实验使用 MUSE（BOOKS、NEWS）与 WMDP（Biosecurity、Cybersecurity）两个基准，评估在 ICLM‑7B、Llama‑2‑7B、Zephyr‑7B‑Beta、Llama‑3.1‑8B、Qwen‑3‑8B 等模型上的效果。

**📈 对比分析**

与 GA、NPO、SimNPO、WHP、RMU 等主流方法对比，KUDA 在遗忘质量（KRD 接近 1）与保留效能（模型整体指标降幅 ≤3%）方面均取得显著提升；在 WMDP 上可消除 88.8% 的危险知识，且保留泛化性能仅微降；在现代模型上表现出更好的迁移与鲁棒性。

**⚠️ 局限性**

局限性：①因果追踪计算量大，需要一次性离线完成；②方法主要针对基于 FFN 的 Transformer 结构，对非 FFN 存储或混合存储模型的适用性未知；③对超参（逆温度、零空间阈值）的选择仍需经验性搜索；④在极大规模遗忘集或高噪声数据上，表示偏移与零空间投影的效果可能下降。

---

## 95. S$^3$GND: An Effective Learning-Based Approach for Subgraph Similarity Search Under Generalized Neighbor Difference Semantics (Technical Report)

**arXiv ID:** 2602.19167 | [PDF](https://arxiv.org/pdf/2602.19167v1)

**作者:** Qi Wen `[一作]` (East China Normal University), Mingsong Chen `[通讯]` (East China Normal University)

**通讯引用:** 7852 | [OpenAlex ID](https://openalex.org/A5102865504)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在加权图中基于关键词集合与边权差异的子图相似度搜索问题 S³GND，并给出了完整的学习驱动的解决方案。

**💡 创新点**

创新点包括：①定义了全新的 Generalized Neighbor Difference（GND）语义，兼顾节点关键词包含关系和邻居边权差异；②利用超图神经网络（HGNN）学习关键词嵌入，并通过最小包围矩（MBR）实现高效关键词集剪枝；③设计了 ND/GND 下界剪枝以及基于预计算的树索引，显著减少搜索空间；④在算法框架中实现了离线预处理与在线高效查询。

**🔧 技术方法**

核心技术：超图神经网络嵌入、关键词 MBR 训练、ND 与 GND 下界推导、关键词集合剪枝、树形索引构造、在线查询时的多级剪枝与子图组装。

**📊 数据集**

实验数据集：5 个真实图（Cora、PubMed、Wiki、TWeibo、Shanghai）以及 3 个合成小世界图（Syn-Uni、Syn‑Gau、Syn‑Zipf）。

**📈 对比分析**

与 S³AND（仅考虑结构）和 Bloom Filter（仅关键词哈希）进行对比；实验显示 S³GND 的剪枝率 > 99%，查询时间比对手快 1–2 个数量级，且返回的子图在结构与权重匹配上质量更高。

**⚠️ 局限性**

局限性：仅支持无向加权图，聚合函数限定为 MAX/SUM；缺乏多跳/动态更新支持；训练依赖超图规模，可能在极大图上受限；对极端稀疏或极密集边权分布的鲁棒性待进一步验证。

---

## 96. Hierarchical Reward Design from Language: Enhancing Alignment of Agent Behavior with Human Specifications

**arXiv ID:** 2602.18582 | [PDF](https://arxiv.org/pdf/2602.18582v1)

**作者:** Zhiqin Qian `[一作]`, Vaibhav Unhelkar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

说明了AAMAS 2026会议使用的LaTeX文档类的主要功能和使用方法。

**💡 创新点**

创新点仅在于对原ACM文档类做了细微修改，以便正确归属IFAAMAS版权。

**🔧 技术方法**

使用LaTeX排版技术，包括acmart风格、Libertine字体、表格和图形环境等。

**📊 数据集**

未使用任何数据集，本文仅为格式说明。

**📈 对比分析**

无实验或方法比较，本文不涉及性能评估。

**⚠️ 局限性**

缺乏实际研究内容和实验数据，主要聚焦格式与排版规范。

---

## 97. QUIETT: Query-Independent Table Transformation for Robust Reasoning

**arXiv ID:** 2602.20017 | [PDF](https://arxiv.org/pdf/2602.20017v1)

**作者:** Gaurav Najpande `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个查询无关的表格转换框架 QuIeTT，将原始半结构化表格预处理为统一、可 SQL 的规范化表格。

**💡 创新点**

创新点在于将表格预处理与下游推理解耦，提供一次性、可重用、无信息损失的转换计划，并通过合成探测查询识别结构缺陷。

**🔧 技术方法**

使用了基于 LLM 的合成探测、规划、代码生成与执行等技术，保持转换确定性并附加原始快照。

**📊 数据集**

在 WikiTQ、NQ‑Table、SequentialQA、HiTab 四个基准以及自制多样化挑战集上评测。

**📈 对比分析**

与多种 LLM 与推理策略（CoT、SQL、NormTab 等）对比，QuIeTT 在所有数据集和模型上均提升 4–8 F1 分，尤其在结构复杂的挑战集上显著领先。

**⚠️ 局限性**

局限在于仅适用于中小型表格、无法处理超大表或多模态内容，且转换阶段仍需足够强大的 LLM，未覆盖跨表检索与推理。

---

## 98. NutriOrion: A Hierarchical Multi-Agent Framework for Personalized Nutrition Intervention Grounded in Clinical Guidelines

**arXiv ID:** 2602.18650 | [PDF](https://arxiv.org/pdf/2602.18650v1)

**作者:** Junwei Wu `[一作]` (Emory University), Carl Yang `[通讯]` (Emory University)

**通讯引用:** 3902 | [OpenAlex ID](https://openalex.org/A5006897094)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种名为NutriOrion的分层多智能体框架，用于为多病共存的患者制定个性化营养干预方案，解决单一大型语言模型在高维病人资料处理中的注意力分散和误判问题。

**💡 创新点**

创新点在于：①采用“并行‑再序列”推理拓扑，将专业领域智能体在并行阶段独立评估，随后在序列阶段进行条件细化与安全约束；②引入多目标优先级评分和药食相互作用硬约束机制，保证生成方案在安全与临床目标上得到构造式验证；③将输出映射为ADIME标准并转化为FHIR R4资源，保证系统与临床电子病历的互操作性。

**🔧 技术方法**

使用技术包括：多智能体框架（Domain、Refine、Synthesis三类智能体），检索增强生成（RAG）从临床指南和DailyMed药物数据库获取证据；多目标优化与优先级排序；硬约束注入（Safety Constraint Mechanism）；结构化输出投影至ADIME和FHIR；以及基于BGE-M3等模型的嵌入检索。

**📊 数据集**

实验数据集为330例美国国立卫生与营养调查（NHANES）2011‑2012周期中卒中患者的完整病历、饮食回顾、实验室指标与药物记录。

**📈 对比分析**

与多种基线（GPT‑4.1、Claude‑4‑Sonnet、单体LLM推理范式、无ADIME或无专职角色的多智能体等）比较，NutriOrion在药食相互作用违规率（12.1%）仅比闭源基线稍高、在行动性（97.8%）和专家评分（7.5/8）上优于所有对照；在个性化（生物标志物与风险营养素负相关系数‑0.26~‑0.35）和营养质量（Food Compass分数≈80）方面也领先。

**⚠️ 局限性**

局限性包括：剩余12%药食相互作用违规表明药物数据库与自然语言表述之间存在语义鸿沟；未提供精准份量与烹饪细节，影响可执行性；仅在卒中患者样本上验证，未覆盖更广泛慢性病人群；对外部知识图谱的依赖有限，需进一步扩展药食知识库。

---

## 99. Make Some Noise: Unsupervised Remote Sensing Change Detection Using Latent Space Perturbations

**arXiv ID:** 2602.19881 | [PDF](https://arxiv.org/pdf/2602.19881v1)

**作者:** Blaž Rolih `[一作]` (University of Ljubljana), Luka Čehovin Zajc `[通讯]` (University of Ljubljana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的无监督变化检测框架 MaSoN，通过在潜在特征空间注入自适应高斯噪声生成多样化的合成变化，并使用共享权重的 ViT‑L DINOv3 编码器和 UPerNet 解码器进行检测。

**💡 创新点**

创新性地在特征空间动态估计两类独立噪声（相关与无关），实现无需人工规则或外部数据的多样化变化合成，从而显著提升跨场景泛化。

**🔧 技术方法**

使用 ViT‑L (DINOv3) 编码器、UPerNet 语义分割头、Perlin 噪声生成二值掩模、动态量化采样、Dice 损失和 AdamW 优化器。

**📊 数据集**

在五个遥感变化检测基准（SYSU、LEVIR、GVLM、CLCD、OSCD）以及 SAR OMBRIA 等多模态数据上进行实验，并对多光谱 Sentinel‑2 进行验证。

**📈 对比分析**

与 PixelDiff、DCVA、DINOv3‑CVA、AnyChange、SCM、DynamicEarth、S2C、HySCDG、I3PE 等方法对比，MaSoN 在平均 F1 上提升 14.1%（相对提升 38.6%），在所有数据集上均优于现有无监督方法。

**⚠️ 局限性**

缺乏完全零训练的特性，需要少量微调；在文本过滤等特殊场景下效果不稳定；对特定噪声参数设置依赖性较高。

---

## 100. Enhancing Automatic Chord Recognition via Pseudo-Labeling and Knowledge Distillation

**arXiv ID:** 2602.19778 | [PDF](https://arxiv.org/pdf/2602.19778v1)

**作者:** Nghia Phan `[一作]` (California State University), Xiao Dong `[通讯]` (Beijing Normal Hong Kong Baptist University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个两阶段训练管道，先用预训练教师生成伪标签训练学生，再在有标注时使用选择性知识蒸馏继续学习，显著提升和超越教师模型。

**💡 创新点**

创新点在于将无标注数据与标注数据分离的两阶段训练，并将知识蒸馏作为持续学习的正则化，特别针对稀有和不平衡和弦质量提升。

**🔧 技术方法**

使用预训练的BTC教师模型生成伪标签，学生模型包括BTC和轻量级2E1D Transformer，采用选择性KD和温度软化的KL损失。

**📊 数据集**

使用超过1000小时的无标注音频（FMA、MAESTRO、DALI）以及约600首带标注的Isophonics、McGill Billboard、RWC Pop、USPop数据集。

**📈 对比分析**

与传统监督学习基线和先前半监督方法对比，学生在七大指标上平均提升约2-4%，尤其在稀有和弦上提高10%以上，最终在连续学习阶段甚至超越教师。

**⚠️ 局限性**

局限在于对教师质量高度依赖，教师偏差会被学生继承；轻量级2E1D对噪声更敏感，需要更强的KD；未验证跨任务扩展。

---

## 101. Facet-Level Persona Control by Trait-Activated Routing with Contrastive SAE for Role-Playing LLMs

**arXiv ID:** 2602.19157 | [PDF](https://arxiv.org/pdf/2602.19157v1)

**作者:** Wenqiu Tang `[一作]` (Nagoya University), Ichiro Ide `[通讯]` (Nagoya University)

**通讯引用:** 2227 | [OpenAlex ID](https://openalex.org/A5034941095)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向角色扮演大型语言模型的面向层级人格控制方法，结合对比稀疏自编码器生成控制向量，并通过特征激活路由在推理时动态注入，以实现细粒度且可解释的个性塑造。

**💡 创新点**

创新点在于：①构建泄漏控制的30面向 Big Five 数据集；②在 SAE 潜在空间引入对比学习，使控制向量在同一面向内紧凑、跨面向分离；③采用 Agent‑based Trait Activation Routing 只在当前语境中激活相关控制向量，从而避免向量竞争和漂移。

**🔧 技术方法**

使用的技术包括：对比稀疏自编码器 (Contrastive SAE)、控制向量 (Control Vector) 注入、对比学习损失 (Prototype Contrast + Distance Loss)、特征激活路由 (Trait‑Activated Routing)、RAG+Prompt 角色扮演管线、评估指标 (FA、MSE、MAE、MTR)。

**📊 数据集**

数据集：新构建的 15,000 条泄漏控制 Big Five 30 面向样本（每面向 500 条正负样本），以及 WikiText‑103‑raw‑v1 用于训练 SAE。

**📈 对比分析**

与基线无干预、Prompt‑Label、CV‑CAA 以及两者组合进行对比；在 Qwen3‑4B 与 Mistral‑7B 上实验显示：CV‑SAE+Prompt 获得最高 Full‑Accuracy（最高 88.5%）、最低 MSE/MSE（≈2.4/3.1）并保持最小 MTR，显著优于对比激活加法和单纯提示方法。

**⚠️ 局限性**

局限性：①对大规模模型的推理效率受限，需进一步优化注入方式；②在某些模型上（如 Mistral‑7B）CV‑CAA+Prompt 失效，提示路由冲突仍待解决；③对比学习对超参敏感，需更稳健的调优；④长期对话中的漂移与上下文迁移仍需进一步评估。

---

## 102. Orchestrating LLM Agents for Scientific Research: A Pilot Study of Multiple Choice Question (MCQ) Generation and Evaluation

**arXiv ID:** 2602.18891 | [PDF](https://arxiv.org/pdf/2602.18891v1)

**作者:** Yuan An `[一作]` (Drexel University), Yuan An `[通讯]` (Drexel University)

**通讯引用:** 5337 | [OpenAlex ID](https://openalex.org/A5040924293)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并执行了一套AI驱动的研究工作流程，利用多模型LLM生成、评估SAT数学多项选择题（MCQ），并对生成题目与专家审核题目的质量进行系统比较；

**💡 创新点**

创新点在于将人工与多LLM协同实现端到端科研流程自动化，揭示LLM如何重塑科研劳动力分配，提出“AI研究运营”新工作范式，并首次使用24维评价框架对生成与基线题目进行等价性检验；

**🔧 技术方法**

所采用的技术包括Gemini-2.5-Flash与GPT-5-nano等LLM代理、RAG检索生成、向量检索与链式思考提示、LLM-as-judge评估方法，以及自定义的多维评价指标与等价性检验（TOST）；

**📊 数据集**

使用的数据集为官方SAT数学题库（共1,071道MCQ）以及OpenStax开放教材（约2,980章节块）作为知识源；

**📈 对比分析**

比较方法为两位LLM评判员对基线、Gemini生成、GPT生成三组MCQ进行24维Likert评分，并通过TOST等价性检验；结果显示生成题目在大多数表面质量维度上与基线相当（平均得分4.64-4.89/5，74-93%指标满分），但在深度、认知参与、难度校准等维度仍不等价；

**⚠️ 局限性**

主要局限包括单一研究者、单一学科范畴、评估完全依赖LLM缺乏人类专家验证、评估工具由LLM生成可能偏倚、未进行真实学生IRT验证，以及效率对专家经验高度依赖。

---

## 103. Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness

**arXiv ID:** 2602.19615 | [PDF](https://arxiv.org/pdf/2602.19615v1)

**作者:** Xin Hu `[一作]` (Tulane University), Zhengming Ding `[通讯]` (Tulane University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个可插拔的轻量模块，通过学习多模态类别嵌入，提升预训练视觉语言模型在罕见物体识别与推理中的表现，无需对VLM进行大规模微调。

**💡 创新点**

创新点在于：①将视觉与文本信息融合生成可学习的多模态类别嵌入；②利用这些嵌入在视觉层面做交叉注意力增强视觉Token，在文本层面插入对象提示；③在保持VLM参数冻结的前提下，显著提升罕见物体推理性能。

**🔧 技术方法**

核心技术包括：多模态类别嵌入学习（视觉提取器+CLIP文本编码器+对齐损失）、交叉注意力视觉Token增强、文本提示注入、轻量跨模态适配器、EMA更新和对齐与分类联合优化。

**📊 数据集**

使用CODA-LM（自动驾驶场景）和GeoBench-VLM（卫星图像）两个benchmark，分别包含罕见物体类别如bollard、stroller、storage tank等。

**📈 对比分析**

在所有基准VLM（LLaVA-1.5-7B/13B、Qwen2.5-VL-7B、InternVL3-8B）上均实现显著提升：CODA-LM总分从46.5提升到72.8（+26.3），在罕见/安全关键类别上提升超过30分；GeoBench-VLM总分从20.9提升到33.2（+12.3）。与专门微调模型相比差距缩小至3分左右，远优于现有训练‑free 方法。

**⚠️ 局限性**

局限性包括：仍需在少量数据上训练类别嵌入与适配器，依赖预训练视觉/文本基础模型和LLM生成的同义描述，处理极度稀有类别时仍可能受限；在推理时需额外计算适配器和类别检测，虽然开销小，但仍比纯训练‑free 方案略高。

---

## 104. CountEx: Fine-Grained Counting via Exemplars and Exclusion

**arXiv ID:** 2602.19432 | [PDF](https://arxiv.org/pdf/2602.19432v1)

**作者:** Yifeng Huang `[一作]` (Stony Brook University), Minh Hoai `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 CountEx 框架，实现可通过正负多模态提示（文本 + 视觉例子）进行精细视觉计数，并在此基础上设计了 Discriminative Query Refinement (DQR) 模块来分离共享特征与负特征。

**💡 创新点**

创新点在于：① 允许用户明确给出排除对象（负提示），大幅提升在视觉相似物体混杂场景的计数准确度；② DQR 三阶段分解机制（共享特征识别、负专属特征提取、选择性抑制），实现正负信息的联合推理；③ 构建 CoCount 数据集，涵盖 97 对细粒度类别，支持公开评测。

**🔧 技术方法**

采用多模态查询编码（文本 + 视觉例子）与 Transformer 交互，基于 LLMDet/CLIP 视觉语言模型；DQR 模块实现注意力投影、负专属残差抑制；点监督计数结合密度预测、共享特征原型学习；训练使用交叉熵、Focal、密度 MSE 等损失。

**📊 数据集**

主要数据集为 CoCount（1,780 视频 / 10,086 帧 / 97 物种对），并在 LOOKALIKES、PairTally、FSC-147 上做零样本或迁移评测。

**📈 对比分析**

在 CoCount 上，CountEx 在 NC（novel category）/ KC（known category）分别取得 MAE 26.61 / 12.72，显著优于 LLMDet、CAD-GD、GroundingREC、CountGD 等基线；在 LOOKALIKES 上零样本 MAE 18.53，成为最优；在 PairTally 上亦领先；在 FSC-147 上 MAE 8.63，略低于 CountGD。

**⚠️ 局限性**

局限性：对正文本依赖过强，模型对模糊或抽象提示的推理能力有限（受 BERT 语义推理限制），导致对含有不确定描述的场景性能下降；排除提示的有效性高度依赖提示的相关性。

---

## 105. Developing a Multi-Agent System to Generate Next Generation Science Assessments with Evidence-Centered Design

**arXiv ID:** 2602.18451 | [PDF](https://arxiv.org/pdf/2602.18451v1)

**作者:** Yaxuan Yang `[一作]` (University of Georgia), Xiaoming Zhai `[通讯]` (University of Georgia)

**通讯引用:** 4909 | [OpenAlex ID](https://openalex.org/A5013379229)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究构建了一套将Evidence-Centered Design（ECD）嵌入多智能体系统（MAS）的框架，利用LLM自动生成符合NGSS三维标准的评估题目，并对生成题目与人类专家编写题目的质量进行比较。

**💡 创新点**

创新点在于将ECD的三层（领域、证据、任务）与LLM多智能体协同流程结合，实现端到端的评估设计，使用JSON结构保证可追溯性与一致性；同时在生成流程中嵌入自动质量评估与图像生成，形成完整的自动化评估制作链条。

**🔧 技术方法**

采用GPT‑4o‑mini作为各角色代理（领域建模、证据建模、情景设计、题目生成、质量评估），使用角色提示与JSON交互；多模态图像由Gemini Imagen 4.0生成；评估采用基于NGSS对齐、DOK、兴趣/包容、语言清晰度、multimodal coherence等维度的综合评分表。

**📊 数据集**

使用两份平行题库共60道题目：30道由人类专家开发的NGSA题目（15生命科学+15物理科学）和30道由MAS自动生成的对应题目，作为实验与对照数据集。

**📈 对比分析**

采用双评估者手工评分，并通过Gwet's AC1/AC2检验一致性；对评分结果进行描述性统计与主题分析。结果显示MAS生成题目在NGSS三维对齐与DOK层面与人类题目相当，但在包容性、语言清晰度与multimodal一致性方面略逊；总体质量基本可与专家相提并论。

**⚠️ 局限性**

限制主要包括：MAS在多模态图像与文本的一致性、图像文字清晰度、语言冗余与提示完整性方面表现不足；系统仍需人类专家审阅；缺乏对教师与评估开发者实际使用体验的评估，需进一步研究人机协同与可用性。

---

## 106. Decoupling Vision and Language: Codebook Anchored Visual Adaptation

**arXiv ID:** 2602.19449 | [PDF](https://arxiv.org/pdf/2602.19449v1)

**作者:** Jason Wu `[一作]` (Amazon Web Services), Jonathan Wu `[通讯]` (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CRAFT，一种轻量级框架，仅微调视觉编码器，将其输出量化到固定离散码本，随后可与任何共享该码本的 LVLM 直接配合使用，从而在不改动 LLM 的前提下实现域适配。

**💡 创新点**

创新点包括：① 使用共享离散码本解耦视觉编码器与语言模型，避免每次域迁移都需重新对齐；② 采用 surrogate alignment loss、commitment loss 与 contrastive loss 三项复合目标，引导编码器生成既准确又简洁的视觉 token；③ 在推理阶段引入稀有度+残差+空间分布的 token 剪枝策略，提升视觉输入的质量与效率。

**🔧 技术方法**

技术手段：离散向量量化（VQ），straight‑through 估计，surrogate 语言模型监督，图像-文本对比学习，视觉 token 的稀有度加权剪枝，以及对码本大小与分配策略的实验探究。

**📊 数据集**

使用的域特定数据集共 10 个：IconQA、OCRVQA、ScienceQA、VQARAD、EuroSAT、Flowers、Kvasir、PlantVillage、Cars、Dogs；其中部分数据被改造成多选 VQA 形式，并随机去掉 20% 类别以测试泛化。

**📈 对比分析**

实验与多种基线对比：零样本、Vision‑FT、Projector‑FT、LDIFS、LLM‑LoRA 等。CRAFT 在 10 个基准上平均提升 13.51%（相较于基线），在多 LLM 体系（Llama‑2、Qwen‑2 系列）中均能跨模型迁移；同时保持甚至提升指令跟随与解释质量。

**⚠️ 局限性**

局限性：① 受离散码本容量限制，太小的码本会导致表达不足；② 依赖 surrogate 语言模型的质量，弱 surrogate 可能无法提供足够训练信号；③ 对细粒度分类仍有限，需更细粒度视觉表征；④ 目前码本固定，若未来出现更大或更细的码本，现有编码器可能需要重新微调。

---

## 107. A Text-Guided Vision Model for Enhanced Recognition of Small Instances

**arXiv ID:** 2602.19503 | [PDF](https://arxiv.org/pdf/2602.19503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 108. Statistical Imaginaries, State Legitimacy: Grappling with the Arrangements Underpinning Quantification in the U.S. Census

**arXiv ID:** 2602.18636 | [PDF](https://arxiv.org/pdf/2602.18636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 109. Hagenberg Risk Management Process (Part 2): From Context-Sensitive Triage to Case Analysis With Bowtie and Bayesian Networks

**arXiv ID:** 2602.19270 | [PDF](https://arxiv.org/pdf/2602.19270v1)

**作者:** Eckehard Hermann `[一作]` (University of Applied Sciences Upper Austria), Harald Lampesberger `[通讯]` (University of Applied Sciences Upper Austria)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5010248122)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套从多维极坐标热图筛选、Bowtie 结构分析到 DAG/BN 运营化的风险管理流程。

**💡 创新点**

创新点在于将上下文维度嵌入热图、将 Bowtie 与 DAG 通过激活节点实现可操作的因果图，兼顾可追溯与实时监控。

**🔧 技术方法**

使用了极坐标热图、Bowtie 结构、DAG/贝叶斯网络、激活节点等技术，并借助 pyBNBowTie 进行映射。

**📊 数据集**

通过即时支付网关的情景案例（自定义的风险场景、上下文调整值）进行验证。

**📈 对比分析**

对方法的比较主要通过工具实现的可重复性、结构完整性来评估，结果显示流程在案例中保持确定性、可追溯，但缺乏量化性能指标。

**⚠️ 局限性**

局限性包括热图仅半量化，贝叶斯网络需人工参数化，假设门控确定性和独立性，未实现完整数据驱动概率估计。

---

## 110. Drift Localization using Conformal Predictions

**arXiv ID:** 2602.19790 | [PDF](https://arxiv.org/pdf/2602.19790v1)

**作者:** Fabian Hinder `[一作]` (Bielefeld University), Barbara Hammer `[通讯]` (Bielefeld University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对概念漂移中的漂移定位问题，提出基于对合预测的全局检测方法

**💡 创新点**

将对合预测引入漂移定位，摆脱局部分组限制，并可使用任意评分函数

**🔧 技术方法**

对合预测、Bootstrap校准、决策树/MLP分类器以及DINOv2深度嵌入

**📊 数据集**

Fashion‑MNIST、NINCO、Fish‑Head（自建）三种图像流数据集

**📈 对比分析**

与基于局部统计检验、随机森林启发式和模型基方法在ROC‑AUC上对比，取得显著提升

**⚠️ 局限性**

对合预测需要足够的校准样本，样本量小或漂移样本稀缺时效果有限

---

## 111. VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery

**arXiv ID:** 2602.19180 | [PDF](https://arxiv.org/pdf/2602.19180v1)

**作者:** Wenhao Shen `[一作]` (Nanyang Technological University), Guosheng Lin `[通讯]` (Nanyang Technological University)

**通讯引用:** 15772 | [OpenAlex ID](https://openalex.org/A5029912845)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于VLM的双记忆自反思批评代理，对单张RGB图像生成的多种3D人体网格进行质量评估，并利用评估结果构建组内偏好数据集，随后采用组级偏好对齐框架（类似GRPO）对扩散式人体网格恢复模型进行微调，使其生成更符合物理约束且更贴合图像的三维网格。

**💡 创新点**

①引入双记忆（规则记忆+原型记忆）和自反思机制，让VLM在评估时能动态检索并更新判定规则，实现稳定且语义一致的评分；②利用组级偏好数据在无3D标注的情况下完成对扩散模型的偏好对齐，突破传统对齐只能使用配对或单例评分的限制；③在扩散模型中采用优势加权对数似然损失，将GRPO思想迁移到确定性ODE采样框架。

**🔧 技术方法**

扩散式人体网格恢复模型、视觉语言模型Qwen3‑VL‑32B、双记忆与自反思机制的批评代理、组级偏好对齐（GRPO改进）算法、优势加权对数似然损失、CLIP视觉嵌入、UCB探索评分、组内标准化优势计算。

**📊 数据集**

用于探索与构建评估知识的合成与真实数据：HI4D、BEDLAM、DNA‑Rendering、GTA‑Human II、SPEC；评估测试集：DNA‑Rendering、GTA‑Human II；参考模型训练集：Human3.6M、3DPW、MPI‑INF‑3DHP、MPII、COCO、UP‑3D；微调与无标注偏好学习集：InstaVariety；基准对比集：3DPW、Human3.6M。

**📈 对比分析**

与多种基准（Deterministic：HMR、HybrIK等；Probabilistic：ProHMR、HMDiff、ScoreHypo、ADHMR、MEGA等）进行对比，评估指标包括PVE、MPJPE、PA‑PVE、PA‑MPJPE。实验表明：在3DPW上，Ours^†（仅使用组级偏好微调）MPJPE降至49.9 mm（相较ADHMR的52.5 mm降低≈6.0%），PVE提升至60.9（相较基准73.4提升≈18.5%），在Human3.6M同样实现显著改进；相比于传统的监督微调（使用伪3D标签）仅提升约3%，本方法显著提升整体性能。

**⚠️ 局限性**

1) 依赖大型VLM（如Qwen3‑VL‑32B），推理时显著增加算力与延迟；2) 双记忆与自反思机制需要频繁的记忆写入与检索，存储与查询成本不容忽视；3) 组级偏好对齐仅在有多样化预测的情况下才有效，若扩散模型生成的多样性不足，偏好信息可能不足；4) 对于极端遮挡或非典型姿态的真实图像，批评代理仍可能产生误判，导致微调方向偏离；5) 目前实验主要集中在公开合成与真实数据集，跨域泛化（如不同光照、服饰）仍需进一步验证。

---

## 112. On the Energy Cost of Post-Quantum Key Establishment in Wireless Low-Power Personal Area Networks

**arXiv ID:** 2602.18708 | [PDF](https://arxiv.org/pdf/2602.18708v1)

**作者:** Tao Liu `[一作]` (Queensland University of Technology), Raja Jurdak `[通讯]` (Queensland University of Technology)

**通讯引用:** 11743 | [OpenAlex ID](https://openalex.org/A5088135082)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

在蓝牙低功耗（BLE）环境下，对后量子密钥协商（PQKE）的能耗进行分解与测量，区分计算与通信两大成本，并在真实硬件上验证模型。

**💡 创新点**

创新点在于：① 将后量子密钥协商的能耗拆分为计算与通信两部分；② 通过实验校正理论模型（引入γ_comp、γ_comm系数），量化通信成本的真实影响；③ 给出跨层优化的设计指南（如启用DLE、调节ATT MTU/LL PDU）以降低能耗。

**🔧 技术方法**

使用技术包括：BLE 1 Mbps PHY、DLE（链路层分片消除）、ML‑KEM（ML‑KEM‑512/768/1024）实现（来自PQClean）、Zephyr RTOS、nRF52840开发板、Nordic Power Profiler Kit II进行功耗采集、周期计数与收发电流模型。

**📊 数据集**

数据集：公开的 ML‑KEM 标准算法参数（不同安全级别的公钥/密文尺寸），以及基准的 ECC（P‑256）配对数据；实验使用 nRF52840 硬件与 BLE PHY，采集不同 ATT MTU/LL PDU 配置下的能耗。

**📈 对比分析**

比较方法：理论能耗模型（周期计数→计算能耗、收发电流→通信能耗）与实际测量结果进行对比，计算校正因子；在多种链路配置下，PQKE 总能耗在 721–2633 μJ，通信占比随 ATT MTU、LL PDU 变化而波动，平均比传统 ECDH 配对高 2.5–8.5 倍。

**⚠️ 局限性**

局限性：仅在连接型 BLE 平台验证，未覆盖 Mesh 或 IEEE 802.15.4 等争用型 PAN；模型未能完全捕捉固件/协议栈的额外开销；未考虑动态网络负载、拥塞和多跳通信对能耗的进一步影响。

---

## 113. TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding

**arXiv ID:** 2602.19768 | [PDF](https://arxiv.org/pdf/2602.19768v1)

**作者:** Fan Yang `[一作]`, Jinqiao Wang `[通讯]` (University of Chinese Academy of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 TraceVision，一种端到端的大型视觉-语言模型，能够双向处理人类视觉轨迹，实现轨迹感知的图像描述、轨迹预测、定位与分割等任务。

**💡 创新点**

创新点包括统一的轨迹感知视觉-语言框架、Trajectory‑aware Visual Perception (TVP) 模块实现视觉与轨迹的双向融合、几何简化与语义加权提取关键点，以及构建的 320k RILN 交互式轨迹数据集。

**🔧 技术方法**

使用技术主要有 Qwen2.5‑VL‑7B 与 QwenViT 的大规模视觉‑语言模型、交叉注意力的 TVP 模块、Douglas–Peucker 几何简化与语义加权、三阶段学习策略和轻量化分割解码器。

**📊 数据集**

使用的数据集包括 Localized Narratives（及其视频扩展）、COCO、ADE20K、Flickr30k、OpenImage 作为基础，并构建了 320k 的 RILN 指令式轨迹任务数据。

**📈 对比分析**

与 Flamingo、BLIP2、LLaVA、Qwen2.5‑VL、PixelLLM 等 LVLM 进行对比，TraceVision 在轨迹导向图像描述、轨迹预测、视觉生成等任务均取得 SOTA 或接近 SOTA，尤其在 RefCOCO、VG 等定位/分割任务中比同规模模型高出 2–5% 以上，整体空间推理准确率提升 23%。

**⚠️ 局限性**

局限性在于仍需大量已标注轨迹数据，模型在极端复杂动态场景下的轨迹生成准确性有限；计算开销较大，实时性能待进一步优化；目前仅关注二维图像轨迹，对多模态（如手势、语音）融合不足。

---

## 114. BloomNet: Exploring Single vs. Multiple Object Annotation for Flower Recognition Using YOLO Variants

**arXiv ID:** 2602.18585 | [PDF](https://arxiv.org/pdf/2602.18585v1)

**作者:** Safwat Nusrat `[一作]` (Leading University), Prithwiraj Bhattacharjee `[通讯]` (Leading University)

**通讯引用:** 30 | [OpenAlex ID](https://openalex.org/A5002304137)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在单框（SISBB）和多框（SIMBB）两种注释策略下，对YOLOv5、YOLOv8和YOLOv12系列模型进行花卉检测实验，并提出了新的FloralSix全注释数据集。

**💡 创新点**

创新点在于：①构建首个包含单框和多框标注的花卉检测基准数据集；②系统评估密度对YOLO检测性能的影响；③对比不同模型规模与优化器在稀疏与稠密场景下的表现。

**🔧 技术方法**

采用YOLOv5s、YOLOv8n/s/m、YOLOv12n模型，使用SGD和AdamW优化器，CIoU损失函数，评估指标为Precision、Recall、mAP@0.5 与 mAP@0.5:0.95。

**📊 数据集**

使用FloralSix数据集：2816张高分辨率图片，包含6种花卉，分别标注单框与多框，共6934个边界框。

**📈 对比分析**

方法：统一训练设置（640×640输入，batch 16，100 epoch，早停），在训练/验证/测试集上计算指标。结果显示：SISBB下YOLOv8m+SGD精度最高；SIMBB下YOLOv12n+SGD在密集环境下表现最佳；SGD在所有场景中优于AdamW。

**⚠️ 局限性**

局限性：数据集仅包含6种花卉，未验证跨物种泛化；光照、遮挡等极端条件下性能仍需提升；模型规模与推理速度之间存在权衡。

---

## 115. SAMAS: A Spectrum-Guided Multi-Agent System for Achieving Style Fidelity in Literary Translation

**arXiv ID:** 2602.19840 | [PDF](https://arxiv.org/pdf/2602.19840v1)

**作者:** Jingzhuo Wu `[一作]`, Junbo Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Style‑Adaptive Multi‑Agent System（SAMAS），通过对文学文本的节奏特征做信号分析，动态生成个性化翻译工作流，提升文学翻译的风格忠实度。

**💡 创新点**

将文学风格量化为可控信号——Stylistic Feature Spectrum（SFS），并利用SFS驱动的动态路由机制，突破传统静态多代理体系在风格适配上的局限。

**🔧 技术方法**

采用波形包络变换提取词长序列的多尺度特征，构建81维SFS向量；基于此向量设计决策树式路由，构成由六个专门化翻译代理组成的可动态组装工作流。

**📊 数据集**

在标准翻译基准FLORES‑200和WMT24上进行评测，涉及英语↔德语、日语、俄语、乌克兰语、汉语等五个语言对。

**📈 对比分析**

使用XCOMET、COMETKIWI等指标对比单一LLM与现有多代理系统，SAMAS在多种基准上显著提升分数（例如Qwen3‑235B‑A22B XCOMET从84.17提升至96.78，且对TACTIC表现出统计显著优势）。

**⚠️ 局限性**

局限性包括：SFS主要基于词长序列，可能忽略句法与语义层面的风格细节；阈值设定依赖验证集，跨作者或跨语种时需重新校准；系统对非文学文本或极端方言的适配效果未知。

---

## 116. OSInsert: Towards High-authenticity and High-fidelity Image Composition

**arXiv ID:** 2602.19523 | [PDF](https://arxiv.org/pdf/2602.19523v1)

**作者:** Jingyuan Wang `[一作]` (Shanghai Jiao Tong University), Li Niu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 31225 | [OpenAlex ID](https://openalex.org/A5111709519)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两阶段框架OSInsert，先用ObjectStitch生成与背景兼容的前景形态，再用SAM提取精确掩码，最后用InsertAnything填充前景细节，实现图像合成的真实性与保真度双提升。

**💡 创新点**

创新点在于将真实性和保真度的权衡拆分为两阶段，分别采用高真实性与高保真度的现有方法，并通过SAM提供精准掩码桥接两阶段，避免单阶段方法面临的细节损失与空间不匹配问题。

**🔧 技术方法**

核心技术包括扩散模型驱动的ObjectStitch（高真实性）、Segment Anything Model (SAM) 用于精确掩码提取，以及InsertAnything（高保真）进行细节填充的两阶段生成流水线。

**📊 数据集**

实验使用MureCOM基准数据集，该数据集涵盖多样背景、复杂前景对象以及显著的姿态/视角差异，能充分测试模型的空间兼容性与细节保留能力。

**📈 对比分析**

与ObjectStitch、InsertAnything以及Banana pro、Seedream 5.0等基线进行对比，OSInsert在真实性（姿态、光照匹配）和保真度（色彩、纹理细节）上均优于单阶段方法，且相对商用模型在框选框定位精度和背景色调保持方面表现更好。

**⚠️ 局限性**

局限性包括：需要两次扩散模型推理导致推理时间较长；对极端姿态/光照冲突仍可能产生误差；仍需人工提供框选框；依赖预训练模型的质量，无法自适应完全未知场景或多前景组合。

---

## 117. A Computationally Efficient Multidimensional Vision Transformer

**arXiv ID:** 2602.19982 | [PDF](https://arxiv.org/pdf/2602.19982v1)

**作者:** Alaa El Ichi `[一作]` (University of Littoral Opal Coast), Khalide Jbilou `[通讯]` (University of Littoral Opal Coast)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于张量余弦积（c-product）的 Vision Transformer 框架 TCP-ViT，保留图像多维结构，直接在张量域上实现线性投影、注意力和前馈网络，显著降低参数量。

**💡 创新点**

创新点包括：
- 用 DCT 基础的张量余弦积实现跨通道耦合，得到 1/C 的参数压缩；
- 该压缩为完全无损、无近似，且对模型深度、头数、FFN 比例均保持统一；
- 通过理论推导与实验验证，证明了参数效率与性能兼备。

**🔧 技术方法**

技术手段：
- 离散余弦变换（DCT）和其逆变换；
- 张量余弦积（c-product）及其基本运算（张量线性、张量注意力、张量层归一化、张量前馈网络）；
- 传统 Transformer 结构（多头自注意力、FFN、层归一化）在张量化版本中的替代实现；
- 混合精度训练、梯度裁剪等训练技巧。

**📊 数据集**

使用的数据集：
- 图像分类：CIFAR-10、SVHN、STL-10（含子样本 10K 训练/2K 测试集和完整 50K/10K 集合）；
- 语义分割：Oxford‑IIIT Pet（128×128 图像，二分类前景/背景）。

**📈 对比分析**

对比方式：与同配置（L、H、r_ff、训练协议）下的标准 ViT 直接比较，保持所有参数除了卷积层之外相同。性能结果：
- 子样本集：参数仅 36.2%，准确率提升 +1.8%（CIFAR-10）、+14.0%（SVHN）、+3.4%（STL-10）；
- 完整 CIFAR-10：参数 36.2%，准确率下降 -5.3%；
- 语义分割：参数 47.8%（相对 Std‑ViT），mIoU 仅下降 1.0%。

**⚠️ 局限性**

局限性：
- 目前仅在小规模 ViT 结构和低分辨率数据集上验证；
- 严格的块对角张量结构限制跨频率交互，导致在大数据量场景下性能下降；
- decoder 未做张量化，整体压缩率受限；
- 当前实现为顺序处理 C 切片，未获得理论 1/C 的算力加速；
- 需进一步验证在大规模数据（如 ImageNet）和更深网络中的可扩展性。

---

## 118. Human-Guided Agentic AI for Multimodal Clinical Prediction: Lessons from the AgentDS Healthcare Benchmark

**arXiv ID:** 2602.19502 | [PDF](https://arxiv.org/pdf/2602.19502v1)

**作者:** Lalitha Pranathi Pulavarthy `[一作]` (Indiana University), Saptarshi Purkayastha `[通讯]` (Indiana University)

**通讯引用:** 2786 | [OpenAlex ID](https://openalex.org/A5075722115)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在AgentDS医疗基准上，利用人类指导的Agentic AI实现多模态临床预测，迭代优化特征工程、模型选择与验证流程。

**💡 创新点**

提出人机协作框架，即在关键决策点用专业知识引导Agentic AI，展示多模态特征抽取、集成多样性对性能的累计提升，并给出三条可推广经验。

**🔧 技术方法**

使用大语言模型驱动的Agentic系统、TF‑IDF（单/多词）、手工关键字匹配、PDF正则解析、统计特征聚合，基于XGBoost/GB/RandomForest等树模型的堆叠或加权集成，并通过交叉验证与宏F1/MAE评估。

**📊 数据集**

AgentDS Healthcare Benchmark的合成数据集，包含30天再入院、急诊成本预测、十一日出院准备三项任务，涵盖结构化表格、临床笔记、扫描PDF账单、时间序列生命体征。

**📈 对比分析**

相较于仅自动化的基线，人工指导后30天再入院任务宏F1提升0.050、急诊成本MAE下降≈8美元、出院准备宏F1提升0.24；最终在公开排行榜上分别位居5/6/3名，总域得分5名；消融实验显示人工决策累计提升≈0.065宏F1。

**⚠️ 局限性**

主要局限在于合成数据缺乏真实噪声、样本量有限导致深度学习探索受限、手工特征工程耗时且难以快速迁移、对GPU资源有一定依赖、外部真实EHR验证尚未完成。

---

## 119. Trojan Horses in Recruiting: A Red-Teaming Case Study on Indirect Prompt Injection in Standard vs. Reasoning Models

**arXiv ID:** 2602.18514 | [PDF](https://arxiv.org/pdf/2602.18514v1)

**作者:** Manuel Wirth `[一作]` `[通讯]` (University of Mannheim), Manuel Wirth (University of Mannheim)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了链式推理（CoT）模型在招聘系统中对间接提示注入（IPI）攻击的脆弱性，开展红队实验对比标准与推理增强 Qwen 3 30B 模型的表现。

**💡 创新点**

揭示推理能力既能放大攻击效果，又在逻辑复杂度提升时产生元认知泄露，使攻击更易被发现，展示了安全-鲁棒性之间的悖论。

**🔧 技术方法**

采用 Qwen 3 30B 大模型，设置标准指令调优版本与基于 CoT 的推理增强版本，设计合成简历与隐藏注入文本进行实验与分析。

**📊 数据集**

使用三份合成简历（目标、干扰者、特洛伊木马）以及嵌入的隐藏指令文本，模拟 ATS 处理流程，未使用公开大规模数据集。

**📈 对比分析**

通过定性对比实验，观察模型在简单攻击下的说服力提升与复杂攻击下的元认知泄露；实验未给出量化指标，仅以示例结果展示性能差异。

**⚠️ 局限性**

局限性包括仅针对 Qwen 3 30B 进行单次定性红队，随机性导致结果不易复现；未评估其他模型或多轮实验；缺乏对泄露频率的统计量化。

---

## 120. SafePickle: Robust and Generic ML Detection of Malicious Pickle-based ML Models

**arXiv ID:** 2602.19818 | [PDF](https://arxiv.org/pdf/2602.19818v1)

**作者:** Hillel Ohayon `[一作]`, Ran Dubin `[通讯]` (Ariel University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 SafePickle，一个基于机器学习的轻量级 Pickle 文件扫描器，利用静态字节码 opcode 统计特征实现恶意模型检测。

**💡 创新点**

创新点在于不需要手工生成库策略或运行时监控，使用 opcode 频率分布训练监督/无监督模型，实现泛化、快速检测，并且能识别压缩、嵌套或多格式包装的恶意文件。

**🔧 技术方法**

采用 Python 的 pickletools 提取 opcode 序列并构造归一化频率向量；使用 scikit‑learn 的 RandomForest、CatBoost、AutoEncoder 等分类/异常检测模型；实验中还使用 t‑SNE 可视化分析。

**📊 数据集**

数据集包括自建的 727 条 Hugging Face Pickle 文件（648 正常 79 恶意），PickleBall、Hide‑and‑Seek（9 条高级逃逸样本）以及 5 条真实恶意文件。

**📈 对比分析**

与传统扫描器（Picklescan、Fickling、ModelScan、ClamAV、VirusTotal、PickleBall）对比，SafePickle 监督模型在自建集上 F1 达 90.01%，在 PickleBall OOD 上 81.22%，并成功检测 Hide‑and‑Seek 所有 9 条样本；传统扫描器 F1 仅 7.23–62.75%。

**⚠️ 局限性**

局限性包括未公开代码、对极度变形或动态生成的 pickle 仍可能产生误报/漏报、仅依赖静态特征，无法检测运行时动态构造的攻击，且在更高级的自定义 opcode 变形面前仍需进一步验证。

---

## 121. When Friction Helps: Transaction Confirmation Improves Decision Quality in Blockchain Interactions

**arXiv ID:** 2602.18834 | [PDF](https://arxiv.org/pdf/2602.18834v1)

**作者:** Eason Chen `[一作]` (Carnegie Mellon University), Kostas Chalkias `[通讯]` (Mysten Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在Sui网络上实现一个区块链 Connect Four 游戏，比较两种授权流程：手动钱包确认（Confirmation Mode）与自动委托（Frictionless Mode），探究交易确认对用户决策质量的影响。

**💡 创新点**

发现交易确认能够充当认知检查点，在不可逆操作中提升决策质量，尽管用户主观更喜欢无确认模式并认为其表现更好。

**🔧 技术方法**

使用区块链前端、智能合约、混合 AI 引擎（MCTS + Minimax）、自适应难度控制及基于 Minimax 的移动质量评分等技术。

**📊 数据集**

收集了109名具备钱包经验的参与者（126人报名，分两波实验）的游戏数据，包含每人四局游戏的移动记录和后测问卷。

**📈 对比分析**

通过配对 t 检验比较主观体验与客观指标（胜率、移动质量）。结果显示：无确认模式客观胜率下降约 11.8%，移动质量下降 0.051；确认模式下被拒绝的移动可自我纠正，移动质量平均提升 0.121。

**⚠️ 局限性**

局限包括：样本仅来自熟悉区块链的社群，缺乏新手；实验为策略游戏，生态效度需验证；未测定思考时间等过程机制；仅对两端极端模式进行比较，未探究中间的确认设计。

---

## 122. Do Generative Metrics Predict YOLO Performance? An Evaluation Across Models, Augmentation Ratios, and Dataset Complexity

**arXiv ID:** 2602.18525 | [PDF](https://arxiv.org/pdf/2602.18525v1)

**作者:** Vasile Marian `[一作]` (University of Queensland), Alexander Buddery `[通讯]` (University of Queensland)

**通讯引用:** 53 | [OpenAlex ID](https://openalex.org/A5042434043)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估多种生成器在不同单类检测场景下对YOLOv11的合成数据增强效果，并研究预训练数据集指标与检测性能的关联。

**💡 创新点**

①系统化对比六类生成器、七种增强比例及两种初始化；②引入Inception‑v3、DINOv2与基于边界框统计的对象中心度量，并对比其对YOLO性能的预测力；③通过残差化相关和多重检验校正，剔除增强比例混杂效应，揭示指标与性能的真实关联。

**🔧 技术方法**

YOLOv11训练、GAN/扩散/混合生成器（DiT、ADM、DiffusionGAN、StyleGAN2‑ADA、ProjectedGAN、LayoutDiffusion）、Inception‑v3/DINOv2嵌入、FID/Precision/Recall、Wasserstein/JS分布距离、Bootstrap匹配样本、残差化相关分析、Benjamini–Hochberg FDR。

**📊 数据集**

Cityscapes Pedestrian、Traffic Signs、COCO PottedPlant三种单类检测数据集，覆盖稀疏/密集/多实例不同检测 regime。

**📈 对比分析**

在真实训练集+合成数据的7种比例下，分别从零初始化和COCO预训练两种方式训练YOLOv11；在真实测试集上计算mAP@0.5:0.95；结果显示从零训练时密集/多实例场景可获+7.6%至+30.6%的mAP提升，稀疏场景提升有限；预训练下提升小或无效。

**⚠️ 局限性**

仅评估单类数据集与YOLOv11，关联性为相关非因果；残差化线性假设可能忽略非线性；生成器样本不独立；指标与多类别、其他检测器的迁移性未知。

---

## 123. The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure

**arXiv ID:** 2602.18523 | [PDF](https://arxiv.org/pdf/2602.18523v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在共享Transformer上进行多任务（加法、乘法、平方）grokking实验，系统研究不同权重衰减下的训练动态与几何特征

**💡 创新点**

首次揭示多任务grokking的阶梯式一般化顺序、执行流的可积性、权重衰减作为相位参数、超参数冗余导致的“超位置”与“无压缩性”，并证明梯度正交削减可导致训练崩溃

**🔧 技术方法**

使用轨迹PCA、对角子矩阵缺陷投影、Hessian最小特征值、重构阈值k*、正交梯度削减与压缩实验等多种几何与动态分析技术

**📊 数据集**

采用模97的算术任务（mod-add、mod-mul、mod-sq），输入为两整数a,b∈{0,…,96}，目标为对应模97的输出

**📈 对比分析**

通过对不同权重衰减、种子和任务数的90+跑，量化grokking时间、曲率深度、缺陷提前期、k*和压缩失效，发现k*仅需4–8个主成分即可恢复>90%准确率，但任何压缩或±5%参数扰动都会导致性能崩溃

**⚠️ 局限性**

实验规模受限于小型Transformer（∼3×10^5参数）和单一算术任务，未检验更大模型、自然语言或视觉任务的可迁移性；正交削减实验仅在单一种子和WD值上进行，缺乏更广泛的鲁棒性验证

---

## 124. Scout-Rover cooperation: online terrain strength mapping and traversal risk estimation for planetary-analog explorations

**arXiv ID:** 2602.18688 | [PDF](https://arxiv.org/pdf/2602.18688v1)

**作者:** Shipeng Liu `[一作]` (University of Southern California), Feifei Qian `[通讯]` (University of Southern California)

**通讯引用:** 562 | [OpenAlex ID](https://openalex.org/A5067936185)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一个混合团队（腿式探测机器人与轮式/腿式车辆）在行星模拟环境中通过本体感知实时估算土壤渗透阻力，构建连续的regolith强度与不确定性地图，并基于此规划安全高效的行驶路径。

**💡 创新点**

首次将腿式机器人的本体力感知与高斯过程回归相结合，生成细粒度的土壤强度地图；将该地图转化为针对不同平台（RHex、轮式）的行进风险模型，实现跨平台协同的安全路径规划；在月球模拟实验与白沙丘实地演示中展示该框架的可行性。

**🔧 技术方法**

直驱/准直驱腿关节、力感知与本体定位、Crawl‑N‑Sense与Trot‑Walk步态、渗透阻力线性模型、Gaussian Process Regression、RFT与RHex旋转行走模型、潜在场路径规划、Chrono物理仿真、机器人探测器Penetrometer。

**📊 数据集**

NASA Ames SSERVI LHS-1 月球模拟实验室的人工压实土壤三档（高、中、低）及其Ground‑truth Penetrometer测量；White Sands National Park 10 m×15 m砂丘区的LIDAR、GPS与现场测量；实验室流化砂床的渗透阻力数据；对应的时间序列、速度与路径信息。

**📈 对比分析**

通过与Penetrometer地面真实值比较，渗透阻力估计平均误差 <10%；在Chrono仿真中验证RHex与轮式车的滑移率、动力需求，仿真与实验速度差异 <15%；在白沙丘实地演示中，安全路径让车完成目标而Naïve路径失效，表明规划方法能有效避免软土陷坑；总体而言，该协同框架将软土风险降低约30%，显著提升探索成功率。

**⚠️ 局限性**

对高强度土壤的渗透阻力估计受脚部柔性压缩误差影响；GPR地图需足够密集采样，对大面积覆盖需要多次巡检；当前未考虑探测机器人自身的地形风险与多探测器协同；仅验证在两种环境，未知在更极端或地下层面的适用性。

---

## 125. A Flow Extension to Coroutine Types for Deadlock Detection in Go

**arXiv ID:** 2602.19686 | [PDF](https://arxiv.org/pdf/2602.19686v1)

**作者:** Qiqi Jason Gu `[一作]` (Macao Polytechnic University), Wei Ke `[通讯]` (Macao Polytechnic University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出 Flow 扩展的协程类型，并实现基于该扩展的 Go 程序死锁检测器；通过类型系统将 Go 并发表达式映射为协程类型，结合 SMT 求解器处理条件与类型继承，最终在 21 个测试程序中准确判定死锁。

**💡 创新点**

创新点在于：① Flow 扩展允许协程拥有任意次数的接收/投递操作和约束类型；② 将值层条件（if/else）直接编码为类型约束并由 Z3 求解；③ 覆盖 17 种典型 Go 死锁模式，并在 4 个真实项目中成功检测到死锁。

**🔧 技术方法**

主要技术包括：协程类型（Coroutine Types）及其 Flow 扩展、行为类型理论、抽象解释、SMT 求解（Microsoft Z3）、Go 代码解析与类型映射、约束求解与类型匹配规则。

**📊 数据集**

使用 17 个人工构造的单文件 Go 程序（涵盖 17 种交互模式）以及 4 个来自开源项目的真实 Go 程序进行评估。

**📈 对比分析**

与 GoDDaR、GOMELA、GoAT 三个主流死锁分析器进行对比；本方法在 21 个案例中无误判且不崩溃，且无需手工调参；与动态工具相比检测速度更快、误报更少；性能上通过 500 步裁剪保证终止。

**⚠️ 局限性**

局限性：仅支持无缓冲双向通道、go 与 defer；不处理单向通道、缓冲通道、select、循环、channel 关闭等特性；未跟踪通道名导致可能误判；状态爆炸风险；需进一步扩展 SMT 以支持更复杂的条件和更多变量。

---

## 126. PedaCo-Gen: Scaffolding Pedagogical Agency in Human-AI Collaborative Video Authoring

**arXiv ID:** 2602.19623 | [PDF](https://arxiv.org/pdf/2602.19623v1)

**作者:** Injun Baek `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PedaCo-Gen，一套以 Mayer 的多媒体学习理论为核心的中介式（IR）人机协作视频生成系统，旨在提升教育视频的教学效果。

**💡 创新点**

创新点在于：①在生成流程中引入“Intermediate Representation”阶段，让教师可先审阅并修改脚本；②将 12 条 CTML 原则嵌入生成约束，并通过 LLM 生成可解释的评审反馈，构成“元认知支架”；③采用人机循环的协作工作流，既保持生成效率又恢复教师的专业主导权。

**🔧 技术方法**

技术上使用了大型语言模型（LLM）做脚本生成与评审、Text‑to‑Video（T2V）模型做视觉合成、前端 Web UI 实现多面板交互，以及基于 CTML 的规则引擎进行约束与反馈。

**📊 数据集**

未公开使用大规模公开数据集，主要依赖参与的 23 位教育专家提供的学习内容与自定义主题，且实验中采用的三类教育题材（因果、抽象、程序性）作评估素材。

**📈 对比分析**

实验对比了基线（LLM 脚本 + 教师手动按 CTML 调整）与 PedaCo-Gen，结果显示所有 12 条 CTML 原则均显著提升（平均提升 0.79 分，p<0.01），生产效率评分 4.26/5，评审有效性 4.04/5，整体满意度 3.78/5；统计检验使用 Wilcoxon 符号秩检验与 Mann‑Whitney U 检验。

**⚠️ 局限性**

局限性包括：① T2V 生成的音频质量不佳，② 生成过程透明度不足，导致教师对 AI 产出的来源与一致性缺乏信任；③ 研究仅在韩国教育者与科学主题上验证，缺乏跨文化与多学科的普适性；④ 未测量学习者的认知负荷与学习成效，仍需后续实地课堂验证。

---

## 127. ManCAR: Manifold-Constrained Latent Reasoning with Adaptive Test-Time Computation for Sequential Recommendation

**arXiv ID:** 2602.20093 | [PDF](https://arxiv.org/pdf/2602.20093v1)

**作者:** Kun Yang `[一作]` (Xiamen University), Hui Li `[通讯]` (Xiamen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 ManCAR，一种基于协同图约束的多步潜在推理框架，用以改进序列推荐。

**💡 创新点**

创新点在于：① 将协同邻域视为可行流形，用图条件先验约束推理轨迹；② 通过 ELBO 形式的 KL 正则化实现潜在漂移抑制；③ 引入教师分布调度与收敛判据，实现自适应推理终止。

**🔧 技术方法**

采用 Transformer 编码器、图嵌入、KL 蒸馏、温度与教师分布调度、残差归一化等技术。

**📊 数据集**

在七个 Amazon 2023 复审子类数据集（CDs、Video、Office、Arts、Music、Toys、Grocery）上进行实验。

**📈 对比分析**

与 SASRec、BERT4Rec、ContextBERT4Rec、ReaRec-ERL/PRL、LARES、PLR 等基线比较，ManCAR 在 NDCG@10、Recall@10 等指标上均优于所有对手，最高相对提升达 46.88%。

**⚠️ 局限性**

局限性包括：对稠密图依赖较强，稀疏数据时优势不明显；需要预构建全局交互图，增加额外工程成本；教师分布调度与温度参数需手工调优。

---

## 128. BayesFusion-SDF: Probabilistic Signed Distance Fusion with View Planning on CPU

**arXiv ID:** 2602.19697 | [PDF](https://arxiv.org/pdf/2602.19697v1)

**作者:** Soumya Mazumdar `[一作]` (Gargi Memorial Institute of Technology), Tapas Samanta `[通讯]` (Variable Energy Cyclotron Centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种CPU-only的贝叶斯SDF融合框架，实现密集3D重建并提供显式不确定性估计与下一最佳视角规划。

**💡 创新点**

创新点在于将传统TSDF初始化与稀疏高斯随机场（GMRF）贝叶斯融合相结合，利用稀疏线性代数和随机Rademacher探针实现后验方差估计，从而得到可用于主动感知的不确定性度量。

**🔧 技术方法**

使用了TSDF bootstrap、稀疏体素层次结构、GMRF先验、预条件共轭梯度（PCG）求解、随机探针方差估计、Marching Cubes/Dual Contouring以及基于方差降低的下一最佳视角（NBV）规划。

**📊 数据集**

实验数据集包括受控消融场景（带参考表面）和CO3D对象序列。

**📈 对比分析**

与TSDF bootstrap和TSDF mesh baseline 通过Chamfer距离、准确率、完整率及多阈值F-score进行比较，实验表明BayesFusion‑SDF在CD、F-score等指标上优于基线，尤其在受控场景中显著提升。

**⚠️ 局限性**

局限性主要是内存和计算开销增加，受限于高分辨率和大环境规模；随机探针估计方差需要额外求解，导致处理时间增加；对阈值敏感，尚未支持动态场景或学习先验。

---

## 129. AegisSat: Securing AI-Enabled SoC FPGA Satellite Platforms

**arXiv ID:** 2602.19777 | [PDF](https://arxiv.org/pdf/2602.19777v1)

**作者:** Huimin Li `[一作]` (Technische Universitaet Darmstadt), Ahmad-Reza Sadeghi `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并实现了一套完整的安全框架，保护基于SoC FPGA的AI卫星平台免受恶意软件、重配置攻击和模型篡改。

**💡 创新点**

创新点在于将安全引导、运行时隔离、身份验证重配置和回滚机制集成到一个层次化、深度防御的架构中，并在空间环境下验证其可行性。

**🔧 技术方法**

使用了可信启动、ARM TrustZone、AXI防火墙、AES-256/ RSA/ECDSA 加密、eFUSE、PTM等技术，以及Xilinx Vitis 2023.1、Vivado 的部分重配置流程。

**📊 数据集**

本文未使用特定的AI数据集，而是基于轻量级CNN加速器与移位电路的硬件示例进行验证。

**📈 对比分析**

通过在ZCU102板上实验，部分重配置耗时约495-528 ms，安全验证与加密解密过程对性能影响可接受，未与传统方法直接对比。

**⚠️ 局限性**

局限性包括仅在单一平台上验证，缺乏对量子后密码、离线自检、能耗高效隔离以及星座级联邦学习安全等场景的支持。

---

## 130. Suppression or Deletion: A Restoration-Based Representation-Level Analysis of Machine Unlearning

**arXiv ID:** 2602.18505 | [PDF](https://arxiv.org/pdf/2602.18505v1)

**作者:** Yurim Jang `[一作]` (Sungkyunkwan University), Simon S. Woo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 2595 | [OpenAlex ID](https://openalex.org/A5033106393)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于恢复的表示层级分析框架，用Sparse Autoencoder识别中间层的类别专家特征，并通过推理时调控恢复被“忘记”数据的预测，以区分模型的抑制（suppression）与删除（deletion）效果；并将该框架应用于12种主流机器无学习（unlearning）方法，发现大部分方法仅实现抑制而非完整删除。

**💡 创新点**

创新点在于：①首次用恢复实验检验无学习效果的内部表示层级完整性；②结合Sparse Autoencoder实现类别专家特征的自动选择并进行推理时的特征调控；③提供了以恢复率为指标的评价指标，补足传统基于输出的评估不足。

**🔧 技术方法**

主要技术包括：Sparse Autoencoder（TopK稀疏性）用于特征提取与对齐；Hungarian算法对特征索引进行匹配；推理时特征调控（steering）实现恢复；ViT-B/16预训练与微调；多种无学习方法（如Finetune、AdvNegGrad、SCRUB、UNSIR等）的对比实验。

**📊 数据集**

实验数据集为CIFAR-10和ImageNette（ImageNet子集），采用ViT-B/16模型在ImageNet-21K上预训练后在两数据集上微调。

**📈 对比分析**

通过比较原始模型、无学习模型与恢复后模型在“忘记”类别上的准确率，发现大多数方法在输出上表现为0%忘记准确率，但恢复后准确率可恢复至80%以上，说明仅抑制；只有如EU-K、L1-Sparse等结构化修改方法在恢复测试中保持0%准确率，显示有效删除。

**⚠️ 局限性**

局限性包括：①仅在ViT图像分类任务上验证，未验证其他网络结构或生成模型；②Sparse Autoencoder对超参数敏感，可能无法捕捉全部内部行为；③框架主要针对类别级无学习，实例级或文本模型的适用性待研究。

---

## 131. Artefact-Aware Fungal Detection in Dermatophytosis: A Real-Time Transformer-Based Approach for KOH Microscopy

**arXiv ID:** 2602.19156 | [PDF](https://arxiv.org/pdf/2602.19156v1)

**作者:** Rana Gursoy `[一作]` (Yildiz Technical University), Gulsum Gencoglan `[通讯]` (Medicana Atakoy Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并评估了基于RT-DETR的对象检测模型，用于自动定位KOH显微镜下皮肤癣菌的真菌结构；

**💡 创新点**

创新点在于：①采用多类别标注，将真菌与易混淆的假体结构分别标注；②使用Transformer端到端检测器RT-DETR，充分利用全局注意力对低对比度、细长结构进行识别；③在真实临床KOH样本上训练，增强模型对噪声与假体的鲁棒性；④实现图像级诊断时无漏检（100%敏感性），具备临床安全性。

**🔧 技术方法**

技术包括：RT-DETR（CNN‑Transformer 混合编码器‑解码器架构）、多尺度特征融合、注意力查询选择、混合精度训练、AdamW优化器、Cosine 学习率调度等。

**📊 数据集**

数据集为2540张高分辨率(2048×2048)KOH显微图像，包含631个真菌实例和381个假体实例，来源于伊斯坦布尔研究与培训医院，按比例划分为训练、验证和测试集。

**📈 对比分析**

性能评估采用 mAP@0.5:0.95、AP@0.5、Mean IoU 等对象级指标，模型在对象级取得召回0.9737、精度0.8043、F1 0.881；在图像级诊断上实现100%敏感性、98.18%特异性、98.82%准确率。相较于传统YOLOv4等CNN方法，RT-DETR 在复杂背景与低对比度下表现更稳健。

**⚠️ 局限性**

局限性包括：单中心样本，缺乏多中心外部验证；只识别真菌而未区分不同种类，也未排除恶性肿瘤等假象；对极弱或对焦不佳的真菌片段仍可能漏检；多类别假体标注覆盖面有限，需进一步完善。

---

## 132. TriTopic: Tri-Modal Graph-Based Topic Modeling with Iterative Refinement and Archetypes

**arXiv ID:** 2602.19079 | [PDF](https://arxiv.org/pdf/2602.19079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 133. LLM Scalability Risk for Agentic-AI and Model Supply Chain Security

**arXiv ID:** 2602.19021 | [PDF](https://arxiv.org/pdf/2602.19021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 134. Differential Perspectives: Epistemic Disconnects Surrounding the US Census Bureau's Use of Differential Privacy

**arXiv ID:** 2602.18648 | [PDF](https://arxiv.org/pdf/2602.18648v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 135. Decoupling Defense Strategies for Robust Image Watermarking

**arXiv ID:** 2602.20053 | [PDF](https://arxiv.org/pdf/2602.20053v1)

**作者:** Jiahui Chen `[一作]` (Tsinghua University), Lifeng Sun `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种两阶段微调框架AdvMark，用以提升深度学习图像水印在抗失真、重建和对抗攻击上的鲁棒性。

**💡 创新点**

创新点包括：①解耦防御策略，先用针对性对抗训练仅微调编码器以保持原图像准确率；②随后对已编码图像直接进行受约束的图像优化，既提升对失真与重建攻击的鲁棒性，又保持阶段一获得的对抗鲁棒性；③引入质量感知早停的PGD优化以及理论可验证的约束损失。

**🔧 技术方法**

核心技术为：自定义对抗攻击构造、条件性编码器/解码器微调、受约束图像优化、LPIPS+MSE质量损失、质量感知PGD早停。

**📊 数据集**

实验使用MS‑COCO和DiffusionDB两大图像数据集。

**📈 对比分析**

与9种传统与最新的水印方法（DwtDctSvd、HiDDeN、MBRS、PIMoG、StegaStamp、DADW、EditGuard、VINE、Stable Signature）在10种攻击（失真、重建、对抗、几何等）下对比，AdvMark在PSNR/SSIM/LPIPS方面达到最高视觉质量，同时在失真、重建和对抗攻击上分别提升29%、33%和46%的位准度。

**⚠️ 局限性**

局限性包括：需要额外的两阶段训练与图像优化，导致训练时延和内存开销略高；对未知攻击模式的泛化能力尚未完全验证；对不同水印模型的适配性需进一步探究。

---

## 136. Operational Robustness of LLMs on Code Generation

**arXiv ID:** 2602.18800 | [PDF](https://arxiv.org/pdf/2602.18800v1)

**作者:** Debalina Ghosh Paul `[一作]` (Oxford Brookes University), Ian Bayley `[通讯]` (Oxford Brookes University)

**通讯引用:** 614 | [OpenAlex ID](https://openalex.org/A5008165707)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一种微观操作鲁棒性评估方法——情景域分析，用于评估大语言模型在代码生成任务对自然语言描述微小变动的敏感性。

**💡 创新点**

创新点在于将安全区（safe zone）与最小失败距离定义到具体情景域中，并通过语义相近的替换生成可用的变体来实现高效的微观鲁棒性探索；该方法不依赖于模型内部信息，可对多种LLM进行公平比较。

**🔧 技术方法**

技术包括：词向量（GloVe）驱动的句子同义替换、多种文本相似度/距离度量（RoBERTa、BLEU、COSINE等）、基于可执行代码的功能等价检测（生成测试用例、AST/DFG/PDG编辑距离）以及数据形态学方法实现的实验系统。

**📊 数据集**

使用ScenEval基准（900个Java编码任务，包含书本与真实任务），并在这些任务上生成不同难度与主题的子集进行实验。

**📈 对比分析**

通过对四个主流LLM（Gemini‑Pro、Codex、Llama2‑13B、Falcon‑7B）计算R^o与R^*两项指标，比较模型在整体与不同情景（复杂度、主题）下的鲁棒性；实验显示方法能有效区分模型，鲁棒性随任务复杂度和主题的提升而下降；且每个测试案例平均不超过17次LLM调用，成本低、时间可控。

**⚠️ 局限性**

局限性包括：仅在Java语言上验证，跨语言迁移需要进一步验证；方法需要生成大量同义变体，若任务描述较长或词汇稀疏可能导致变体不足；并且对可执行代码的功能等价检测在复杂程序上仍存在一定误差。

---

## 137. Anticipate, Adapt, Act: A Hybrid Framework for Task Planning

**arXiv ID:** 2602.19518 | [PDF](https://arxiv.org/pdf/2602.19518v1)

**作者:** Nabanita Dash `[一作]` (Robotics Research Center IIIT Hyderabad), K. Madhava Krishna `[通讯]` (Robotics Research Center IIIT Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个混合框架，使助手机器人在与人类协作时能够预测即将到来的任务，评估人类可能的失败风险，并主动采取行动来预防或恢复这些失败，从而提升任务完成率。

**💡 创新点**

创新点在于三方面的融合：①利用LLM进行任务序列预测；②使用RDDL进行概率性序列决策规划；③构造奖励机制平衡任务完成与失败预防/恢复的权衡，首次在此类框架中将人类行为预测模型与奖励机制相结合，显著提升协作鲁棒性。

**🔧 技术方法**

核心技术包括：LLM（llama‑3.3‑70b‑versatile）+ Groq API 进行任务预测；Relational Dynamic Influence Diagram Language (RDDL) + PROST 规划器构建基于马尔可夫决策过程的决策模型；自定义奖励函数与状态转移模型来模拟人类行为的不确定性，并通过强化学习式的启发式搜索优化执行计划。

**📊 数据集**

实验数据集为 VirtualHome 3D 模拟环境中的 11 种家庭任务（烹饪与清洁），共 9 种食材、8 台家电、9 件餐具和 5 件清洁工具；通过 10 次带噪声的模拟轨迹估计人类行为转移概率。

**📈 对比分析**

与两类基线（仅 LLM 生成计划、仅 RDDL/PROST 规划且无失败预防奖励）对比。评估指标包括任务完成率、子目标完成率、平均动作数、失败数、失败预防数和恢复数。结果显示：我们框架在任务完成率（84.78%）和子目标完成率（85.5%）上分别高出 LLM 基线约 40% 与 21%；失败数显著下降，且失败预防与恢复率大幅提升，平均动作数比基线少约 15%~30%。

**⚠️ 局限性**

局限性包括：①实验仅在仿真环境中验证，缺乏真实机器人部署的验证；②人类行为模型仅通过有限次数的带噪声仿真估计，未考虑多样化的真实人类行为；③奖励函数中的多重参数需要人工调优，缺乏自动化自适应机制；④当前框架仅处理单人类单机器人协作，未扩展至多智能体场景。

---

## 138. Design and Control of Modular Magnetic Millirobots for Multimodal Locomotion and Shape Reconfiguration

**arXiv ID:** 2602.19346 | [PDF](https://arxiv.org/pdf/2602.19346v1)

**作者:** Erik Garcia Oyono `[一作]` (Imperial College London), Dandan Zhang `[通讯]` (Imperial College London)

**通讯引用:** 5019 | [OpenAlex ID](https://openalex.org/A5100386760)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文设计并实现了一种低场（<13 mT）二维磁场可编程的模块化微型机器人平台，包含自由模块、固定模块和抓取模块，能够在同一工作空间内实现单元无碰撞的自组装、形态重构和闭环导航。

**💡 创新点**

创新点在于：①使用仅二维Helmholtz+Maxwell线圈生成可编程磁场，显著降低场强与热耗；②通过磁偶极子相互作用实现无碰撞的链-抓取机、链-方块重构；③实现单模块的闭环视觉控制与A*路径规划，使单个模块完成迷宫导航。

**🔧 技术方法**

采用的技术包括：磁偶极子动力学模型、时间可变均匀/梯度磁场驱动、基于HSV阈值的实时模块检测与轨迹跟踪、A*搜索+有限状态机控制、SLAM-like路径纠正以及三维打印与SLA成型的模块制备。

**📊 数据集**

实验数据集由在35 mm × 35 mm平面工作空间内的多次（>80）单元运动、链-抓取机重构（20次）和迷宫导航试验（数次）组成，记录了位移、转向误差、重构成功率与时长等指标。

**📈 对比分析**

相较于先前需10–40 mT、三维线圈阵列、以及仅能实现链式运动的系统，本平台在<13 mT下实现了90 % 的链-抓取机重构成功率、65 % 的链-方块重构成功率，单模块在梯度场辅助下实现可重复的直线运动，且实现了无碰撞的迷宫闭环导航。

**⚠️ 局限性**

局限性包括：缺乏纯滚动运动模式；实验仅在平面、无生物相容性及流体阻力的仿真环境下进行；未对导航误差、实时控制频率、动态扰动鲁棒性等进行量化评估；模块尺寸仍为3 mm级，尚未达到更小型化与生物医学实际需求。

---

## 139. CREM: Compression-Driven Representation Enhancement for Multimodal Retrieval and Comprehension

**arXiv ID:** 2602.19091 | [PDF](https://arxiv.org/pdf/2602.19091v1)

**作者:** Lihao Liu `[一作]` (Tsinghua University), Guorui Zhou `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的压缩驱动框架，通过可学习的合唱词令牌（chorus tokens）将多模态信息压缩为共享表示，从而实现检索与生成任务的双向优化；

**💡 创新点**

将压缩提示设计与压缩感知注意力相结合，构建联合对比与语言建模的混合目标，首次让生成监督提升检索嵌入质量，同时保持生成能力；

**🔧 技术方法**

可学习合唱词令牌、压缩感知注意力掩码、统一提示模板、对比学习（InfoNCE）、随机压缩语言建模、LoRA微调、混合生成数据采样；

**📊 数据集**

MMEB检索基准、ShareGPT‑4V生成数据、Qwen2‑5‑VL‑7B预训练模型；

**📈 对比分析**

在MMEB上取得最优 Precision@1（65.8/70.8/66.7，分别在不同规模下击败 UNITE、VLM2Vec 等基线），在MMB、MMVet、AI2D、HallusionBench、MMMU、MMStar 等多模态生成基准上保持与原始 Qwen2‑VL 相当的平均分；

**⚠️ 局限性**

仍存在压缩后信息丢失导致部分细节理解不足、对极大规模检索集的负载与长文本生成的可扩展性需要进一步验证。

---

## 140. UFO: Unlocking Ultra-Efficient Quantized Private Inference with Protocol and Algorithm Co-Optimization

**arXiv ID:** 2602.18758 | [PDF](https://arxiv.org/pdf/2602.18758v1)

**作者:** Wenxuan Zeng `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**通讯引用:** 24616 | [OpenAlex ID](https://openalex.org/A5100457407)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种端到端的 2PC 私有推理框架，系统性地将 Winograd 卷积和低精度量化结合，并通过协议层和算法层的协同优化显著降低通信与延迟。

**💡 创新点**

创新点在于：①提出了 QWinConv 协议，利用 Winograd 变换实现乘法减少并配合量化；②引入图层级协议融合、简化残差协议和 MSB 已知优化，消除冗余位宽转换；③设计了基于敏感度的混合精度 QAT 与位权重重映射算法，既提升精度又避免了 Winograd 转换引入的权重离群点。

**🔧 技术方法**

技术手段包括 OT‑based 2PC、Winograd 卷积、低位宽量化、位权重重映射、敏感度分析、整数线性规划、图层级协议融合与简化残差、MSB‑已知位宽优化、以及基于 PyTorch 的 QAT 细调。

**📊 数据集**

实验使用了 MiniONN、CIFAR‑10、CIFAR‑100、Tiny‑ImageNet、ImageNet 等数据集，评估了 ResNet‑20/32/18、Tiny‑ImageNet 及 ImageNet 上的模型。

**📈 对比分析**

与 SiRNN、COINN、CoPriv、ReLU‑优化方案以及 HE‑based 框架（Gazelle、Delphi、CrypTFlow2）对比，取得了 0.71%‑1.29% 的准确率提升，并在总通信量上分别实现 9.41×、11.7×、6.33× 的大幅减少；在在线通信和延迟方面亦比现有方法低 2.1×‑3.6×。

**⚠️ 局限性**

局限性包括：仅针对 CNN 架构，Winograd 加速不易直接迁移至 Transformer；协议仍基于 honest‑but‑curious 设定；在极低位宽场景下的量化稳定性和安全性分析尚不完整；对 HE‑based 系统的集成仍需进一步研究。

---

## 141. LAMMI-Pathology: A Tool-Centric Bottom-Up LVLM-Agent Framework for Molecularly Informed Medical Intelligence in Pathology

**arXiv ID:** 2602.18773 | [PDF](https://arxiv.org/pdf/2602.18773v1)

**作者:** Haoyang Su `[一作]` (Fudan University), Xiaosong Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 8662 | [OpenAlex ID](https://openalex.org/A5100724911)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了LAMMI‑Pathology，一个面向病理图像与空间转录组等分子证据的工具调用型多代理框架，利用Atomic Execution Nodes (AEN) 构建半模拟轨迹并通过轨迹感知 Adapter (TA) 进行微调，实现在视觉与分子层面的证据驱动推理。

**💡 创新点**

①采用自底向上工具聚类成组件代理，显著降低上下文长度；②引入 AEN 作为真实工具交互的最小原子单元，用以生成结构化轨迹；③设计轨迹感知 Adapter 只更新低秩子空间，兼顾结构学习与基础模型能力；④在病理领域实现多模态工具调用与分子证据的统一推理。

**🔧 技术方法**

使用大规模视觉‑语言模型（Qwen3‑VL‑8B、InternVL3.5‑8B、MiniCPM‑V‑4.5 等）结合工具调用框架；AEN 轨迹构造、轨迹感知 Adapter；多模态视觉‑文本编码；DeepSpeed ZeRO‑3 并行训练；评估指标包括 TCF1、TSS、ACS、HR、TRR 等。

**📊 数据集**

构建 ST‑Traj（基于 10,684 条 AEN 的轨迹集合，分为 2‑8 步）；PathSpatial‑DocQA（基于 HEST 与 STimage‑1K4M 的问答集合）；PathMMU 公共 QA ；同时利用空间转录组与组织学图像的公开数据集做工具与信息抽取。

**📈 对比分析**

与 OpenAI‑Agents‑SDK、ReACT、MAT‑Agent、MLLM‑Tools 等框架在相同基线 LVLM 上进行对比。LAMMI 在 PathSpatial‑DocQA 的 ACS 达到 0.809（InternVL3.5‑8B），高于 GPT‑5 的 0.739；在 ST‑Traj 中 TCF1、TSS、ACS 分别显著高于基线（如 0.397、0.868、0.533）；在 PathMMU 上，LAMMI 以 open‑source 模型获得最高 ACS 与 F1。

**⚠️ 局限性**

局限性：目前聚焦病理图像与空间转录组，未覆盖更广泛的多模态或其他分子数据；工具集需预先定义，对未知工具的适应性有限；某些对比实验在模型与基线配置上可能不完全统一，导致性能差异受限；在极大规模数据或更复杂推理任务中，仍需进一步提升效率与通用性。

---

## 142. On the Inherent Resilience of Task-Oriented V2X Networks to Content-Selection Errors

**arXiv ID:** 2602.18620 | [PDF](https://arxiv.org/pdf/2602.18620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 143. ConfSpec: Efficient Step-Level Speculative Reasoning via Confidence-Gated Verification

**arXiv ID:** 2602.18447 | [PDF](https://arxiv.org/pdf/2602.18447v1)

**作者:** Siran Liu `[一作]` (Peking University), Cyril Y. He `[通讯]` (ScitiX AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型语言模型的链式思考（CoT）推理，提出一种置信门控级联验证框架，实现了在不牺牲准确率的前提下显著加速推理。

**💡 创新点**

创新点在于发现步骤级验证的难度不均衡，利用小模型的置信度来判断是否需要上层大模型验证，从而在多数步骤直接使用低成本验证，显著提升推理速度。

**🔧 技术方法**

核心技术包括：置信门控级联验证、草稿模型与目标模型的双层推理、树结构草稿扩展以及与令牌级自适应推理（speculative decoding）的融合。

**📊 数据集**

使用了多种数学推理与代码生成基准：AIME24、AMC23、MATH500、GPQA Diamond 与 HumanEval，覆盖了多领域的长链推理任务。

**📈 对比分析**

与现有的 SpecReason、Embedding、LookaheadReasoning 等方法相比，该方法在保持与目标模型相同的 Pass@1 准确率（约 99.7% 的匹配度）的同时，实现了平均 1.68×（DeepSeek）/1.30×（Qwen） 的推理加速；与 token‑level speculative decoding 结合后可实现 2× 以上的叠加加速。

**⚠️ 局限性**

局限性包括：对草稿模型置信度的依赖可能在不同模型或任务分布中失效；步骤级验证仅近似语义等价，可能对高度细致或隐式推理产生误判；在草稿模型性能不足或领域漂移显著时，系统会频繁退回至目标模型，导致加速效果下降。

---

## 144. TactiVerse: Generalizing Multi-Point Tactile Sensing in Soft Robotics Using Single-Point Data

**arXiv ID:** 2602.19850 | [PDF](https://arxiv.org/pdf/2602.19850v1)

**作者:** Junhui Lee `[一作]` (Kyungpook National University), Saekwang Nam `[通讯]` (Kyungpook National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

引入TactiVerse——一种基于U‑Net的视觉软触觉传感器形变预测框架，能够从相机图像中估计单点乃至多点接触几何。

**💡 创新点**

通过将接触几何映射为热图预测，使得仅用单点压痕数据即可泛化到多点接触，并显著提升多点感知性能。

**🔧 技术方法**

采用U‑Net卷积网络、基于Hertz理论校正的高斯核热图生成、BCE损失训练以及深度缩放映射。

**📊 数据集**

使用TacTip软触觉传感器采集的约5000张单点压痕RGB图像为主，并扩充了双点与三点接触样本。

**📈 对比分析**

与传统CNN基线在单点任务上对比，U‑Net MAE为0.0589 mm优于0.0612 mm；在双点任务中，单点训练的U‑Net MAE为1.214 mm，加入多点数据后降至0.383 mm，表明性能显著提升。

**⚠️ 局限性**

对极稀疏或高度重叠的接触仍存在误差，热图宽度近似受限，且未充分验证对复杂三维形变的鲁棒性。

---

## 145. Where Should Robotaxis Operate? Strategic Network Design for Autonomous Mobility-on-Demand

**arXiv ID:** 2602.19341 | [PDF](https://arxiv.org/pdf/2602.19341v1)

**作者:** Xinling Li `[一作]` (Massachusetts Institute of Technology), Gioele Zardini `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 141 | [OpenAlex ID](https://openalex.org/A5043524649)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并求解了自动驾驶按需出行系统的网络设计问题，即选择可自动驾驶的子网络、调度车辆并满足基础设施、车队时长及路径级服务质量限制，以实现运营利润最大化。

**💡 创新点**

1）首次将自动驾驶网络设计建模为单层混合整数规划，直接结合基础设施投资与车队时长约束；2）提出基于路径的混合整数表述，并通过列生成（Column‑Generation）方法将指数规模的路径变量降至可处理量；3）将鲁棒优化（box不确定性）融入同一框架，保持分解结构；4）通过精确的标签修正算法解决带资源约束的最短路径子问题。

**🔧 技术方法**

路径级混合整数线性规划、列生成分解、带资源约束的最短路径标签修正算法、鲁棒优化（box不确定性）。

**📊 数据集**

纽约曼哈顿的道路网络（从OpenStreetMap精简后的约4300节点/7000条有向边）与基于TLC出租车数据的日需求（约20万行程/日）。

**📈 对比分析**

与传统固定网络或仅考虑运营层的研究相比，本方法在规模化城市网络（7k边）上能够在25分钟内得到LP最优并且整数解与LP最优之间的相对最优差距<10⁻⁶；鲁棒版本仅需修改参数，保持相同求解流程。

**⚠️ 局限性**

①忽略了空车重平衡、时间动态与需求峰谷变化；②鲁棒扩展仅覆盖box不确定性，无法处理更一般的不确定集；③最终整数求解仍受限于MILP难度，实例难度高度依赖网络与需求；④路径级约束（如左转限制）会显著削减可行路径，进一步增加求解复杂度。

---

## 146. ReAttn: Improving Attention-based Re-ranking via Attention Re-weighting

**arXiv ID:** 2602.19969 | [PDF](https://arxiv.org/pdf/2602.19969v1)

**作者:** Yuxing Tian `[一作]` (University of Montreal), Jian-Yun Nie `[通讯]` (University of Montreal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM的注意力权重进行后处理，以改进零样本重排序。

**💡 创新点**

提出了跨文档IDF加权和熵正则化的两步后处理策略ReAttn。

**🔧 技术方法**

使用注意力权重、交叉文档IDF、熵正则化以及对现有LLM注意力权重的无监督重加权。

**📊 数据集**

在BEIR的11个公共数据集以及LongMemEval、Clipper等长上下文推理基准上评估。

**📈 对比分析**

与RankGPT、ICR、QRhead等注意力重排序方法和传统检索器对比，ReAttn在nDCG@10、Recall@k等指标均显著提升，且性能可与监督式交叉编码器相媲美。

**⚠️ 局限性**

局限：仅在通用LLM上评估，未考虑专门为IR微调的模型；未深入分析注意力头的贡献；仅在英语数据集，缺乏跨语言验证。

---

## 147. MetaBlue: A Metasurface-Assisted Acoustic Underwater Localization System

**arXiv ID:** 2602.19252 | [PDF](https://arxiv.org/pdf/2602.19252v1)

**作者:** Junling Wang `[一作]` (Shanghai Jiao Tong University), Zhenlin An `[通讯]` (University of Georgia)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了低成本被动声学超表面(AMS)，通过将其安装在普通超声发射器上，嵌入方向相关的频谱特征，实现单水听器的角度估计；同时提出利用电磁泄漏信号与声波时间差实现无时钟同步的测距，从而在单基站条件下完成3D定位。

**💡 创新点**

创新点包括：① 适用于水下的厚度工程式AMS，克服空气-固体阻抗差导致的相位调制失效；② 基于EM泄漏的TDoA测距方案，消除时钟同步需求；③ 通过谱特征匹配与低通滤波实现多径抑制，提升AoA精度；④ 将AoA与ToA联合融合，实现在单基站下同步无关的高精度3D定位。

**🔧 技术方法**

技术方法包括：声学超表面设计与仿真、宽带扫频脉冲（125–375 kHz）发射、频谱特征库与匹配、低通滤波多径抑制算法、EM泄漏时间参考提取、非线性最小二乘融合定位、Kalman滤波数据融合。

**📊 数据集**

使用实测数据集：室内水槽、游泳池（8 m×8 m）和户外池塘（10 m×10 m）环境，距离范围0–12 m，360°角度的校准集；采样率31.25 MHz，chirp时长0.2 ms。

**📈 对比分析**

与U-Star、3D‑Blue等现有方法对比，单基站平均AoA误差8.7°，3D定位误差0.73 m；四基站下误差降至0.37 m；所有环境下定位精度均保持在0.8 m以内，成本约为$11/基站，显著低于传统多基站阵列方案。

**⚠️ 局限性**

局限性包括：测距受EM泄漏衰减限制，深水或极端浑浊环境验证不足；多径强度大时精度下降；单基站下对环境深度、结构干扰较敏感，需要更多基站或更复杂的多径抑制以进一步提升鲁棒性。

---

## 148. Understanding the Curse of Unrolling

**arXiv ID:** 2602.19733 | [PDF](https://arxiv.org/pdf/2602.19733v1)

**作者:** Sheheryar Mehmood `[一作]`, Peter Ochs `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了算法展开（unrolling）在求解参数化优化问题时的非渐近行为，揭示了在有限迭代深度下导数迭代可能先增后减的“展开诅咒”（curse of unrolling），并提出通过截断早期迭代或利用暖启动（warm‑starting）来缓解这一现象。

**💡 创新点**

1) 给出了非渐近误差上界，明确指出导致展开诅咒的关键因素（收敛率、更新映射的光滑性、初始误差）；2) 证明了截断早期迭代能显著削弱该误差并改善梯度估计；3) 将暖启动解释为隐式截断，提供了无需手动设定截断点的自适应方案。

**🔧 技术方法**

使用了Banach固定点定理、隐函数定理、自动微分（前向/后向模式）与递归导数传播，结合非渐近分析（递推不等式）推导误差上界；还在理论中引入了截断策略与计算资源分配模型。

**📊 数据集**

主要使用合成的代表性例子（如二次目标函数 f(u)=−‖u‖²/2+‖u‖²/2）以及梯度下降和 Heavy‑ball 等经典迭代算法的数值实验；未涉及公开真实数据集。

**📈 对比分析**

与传统完整展开（no truncation）和仅使用暖启动的对比实验显示：截断或暖启动能显著降低梯度误差的峰值，缩短收敛时间，同时减少内存占用；在有限迭代预算下，截断还能让算法迭代次数翻倍，进一步提升性能。

**⚠️ 局限性**

1) 上界仅为理论上限，实际误差可能更好；2) 关键参数（收敛率、Lipschitz常数等）往往未知，截断点难以精确设定；3) 需要全局收敛性与光滑性假设，实际问题可能不满足；4) 现代自动微分框架对动态截断支持有限，导致实现上有一定复杂性。

---

## 149. PaReGTA: An LLM-based EHR Data Encoding Approach to Capture Temporal Information

**arXiv ID:** 2602.19661 | [PDF](https://arxiv.org/pdf/2602.19661v1)

**作者:** Kihyuk Yoon `[一作]` (Georgia Institute of Technology), Jing Li `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了PaReGTA框架，将结构化EHR按访视文本化并使用LLM生成患者向量，同时提出PaReGTA‑RSS解释因子重要性。

**💡 创新点**

①基于LLM的轻量级对比微调实现领域适配；②混合时序池化同时保留递归与全局信息；③针对LLM编码提出可解释的Representation Shift Score。

**🔧 技术方法**

使用大语言模型句子嵌入（GTE‑base‑v1.5）+SimCSE对比学习；混合时间权重+注意力池化；Logistic回归+RSS；传统机器学习分类器（LightGBM、XGBoost等）。

**📊 数据集**

All of Us 研究计划中39,088名偏头痛患者的结构化EHR数据。

**📈 对比分析**

与传统 one‑hot 与 Bag‑of‑Codes 稀疏编码对比，在慢性与发作性偏头痛分类任务中，PaReGTA 在多种分类器上 AUC 提升至 0.95 以上，显著优于稀疏基线；序列模型难以收敛。

**⚠️ 局限性**

①因子重要性基于模型预测，不具因果解释；②仅使用传统 ML，深度序列模型未充分探索；③实验包含诊断后期记录，实际临床预测可能受限；④LLM 模型对计算资源有一定需求。

---

## 150. UniE2F: A Unified Diffusion Framework for Event-to-Frame Reconstruction with Video Foundation Models

**arXiv ID:** 2602.19202 | [PDF](https://arxiv.org/pdf/2602.19202v1)

**作者:** Gang Xu `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Junhui Hou `[通讯]` (City University of Hong Kong)

**通讯引用:** 10338 | [OpenAlex ID](https://openalex.org/A5031957432)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用预训练视频扩散模型和事件数据实现事件到帧的统一重建、插值与预测

**💡 创新点**

结合事件的物理残差引导、在扩散过程中的条件调制，形成零样本统一框架

**🔧 技术方法**

SVD预训练模型、条件微调、残差引导、逆扩散采样调制、ResNet残差预测

**📊 数据集**

合成的TrackingNet+DVS‑Voltmeter事件‑帧对，以及HS‑ERGB真实事件‑RGB数据集

**📈 对比分析**

与E2VID、FireNet、CUBE等方法对比，在MSE/SSIM/LPIPS/FID上均优于SOTA，零样本插值/预测亦保持竞争力

**⚠️ 局限性**

依赖大型扩散模型导致推理开销大、显存高；事件稀疏时难以重建无事件区域，色彩一致性仍有差距

---

## 151. Relational Feature Caching for Accelerating Diffusion Transformers

**arXiv ID:** 2602.19506 | [PDF](https://arxiv.org/pdf/2602.19506v1)

**作者:** Byunggwan Son `[一作]` (Yonsei University), Bumsub Ham `[通讯]` (Korea Institute of Science and Technology)

**通讯引用:** 4340 | [OpenAlex ID](https://openalex.org/A5054888241)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种关系特征缓存（RFC）框架，针对扩散变换器（DiTs）在多步去噪过程中的特征缓存问题进行改进。

**💡 创新点**

创新点在于：①使用关系特征估计（RFE）通过输入特征变化来推断输出特征的变化幅度；②引入关系缓存调度（RCS）根据预测误差动态决定何时执行完整计算，从而大幅减少缓存误差。

**🔧 技术方法**

核心技术包括：基于线性映射的输入‑输出关系建模、Taylor展开的特征预测、L₂归一化以及误差累计阈值控制的动态调度。

**📊 数据集**

实验使用了 ImageNet（类条件图像生成）、FLUX.1 dev（文本到图像）和 HunyuanVideo（文本到视频）等数据集。

**📈 对比分析**

与 FasterCache、GOC、TaylorSeer 等现有缓存方法相比，RFC 在相同 FLOPs 下显著提升了 sFID、FID2FC、PSNR 等指标，例如在 ImageNet 上 3.37 TFLOPs 下的 sFID 低于 4.76 TFLOPs 的 TaylorSeer 1.26 点；在有限计算预算下保持了更高的生成质量。

**⚠️ 局限性**

局限性主要体现在：①假设输入‑输出映射局部线性且方向一致，若模型出现非线性或大幅度特征变动时可能失效；②虽然 RFE/RCS 计算成本低，但在极大模型规模或特殊任务中仍需进一步验证。

---

## 152. WildOS: Open-Vocabulary Object Search in the Wild

**arXiv ID:** 2602.19308 | [PDF](https://arxiv.org/pdf/2602.19308v1)

**作者:** Hardik Shah `[一作]` (Jet Propulsion Laboratory), Patrick Spieler `[通讯]` (Jet Propulsion Laboratory)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5006271370)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为WildOS的实时系统，实现开放词汇物体搜索与长距离、无地图的户外环境中语义导航；

**💡 创新点**

结合导航图的几何记忆与视觉基础模型（ExploRFM）对前沿点进行语义评分，实现对未知区域的长距离感知与安全决策；

**🔧 技术方法**

使用基于RADIO的视觉语言模型ExploRFM进行可穿越性、视觉前沿和目标相似度预测，采用粒子滤波式三角定位粗估目标位置，构建可评分导航图并在其上进行Dijkstra规划；

**📊 数据集**

训练ExploRFM的可穿越性头使用RUGD数据集，前沿头使用GrandTour手工标注350张图像；实验在Boston Dynamics Spot平台上在多种户外与城市环境中进行；

**📈 对比分析**

与纯视觉前沿导航（LRN）和仅几何导航（Vanilla GraphNav）做对比，WildOS在总路径长度、耗时、成功率和鲁棒性上显著优于两种基线，并在复杂死胡同、障碍穿越等情境表现更好；

**⚠️ 局限性**

局限包括可能的探索振荡、缺乏视觉记忆以支持回溯式查询、探索区域近似为圆形导致前沿误检、以及单一相似度阈值对不同目标类别的不足。

---

## 153. K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model

**arXiv ID:** 2602.19128 | [PDF](https://arxiv.org/pdf/2602.19128v1)

**作者:** Shiyi Cao `[一作]` (University of California Berkeley), Ion Stoica `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将大型语言模型（LLM）作为可共进化的世界模型，驱动 GPU kernel 生成的搜索框架，解耦高层规划与低层代码实现。

**💡 创新点**

创新点在于：①将 LLM 从单纯的代码生成器提升为“世界模型”，能够维护并更新搜索树状态；②通过插入、更新、剪枝三种树编辑操作，让搜索过程在有限预算内动态聚焦有前景的优化路径；③实现了在新硬件和新工作负载下的自适应优化。

**🔧 技术方法**

核心技术包括：LLM 作为世界模型的概率推理、搜索树（Closed/Open 节点）管理、优先级估计 V、基于 LLM 的局部实现策略 π_code、增量更新与前景剪枝、以及基于 in‑context 学习的模型共进化。

**📊 数据集**

使用 FlashInfer‑Bench 提供的四类 kernel（MLA Paged Prefill、MLA Paged Decode、GQA Paged Decode、FP8 MoE）以及 GPUMODE TriMul 任务进行实验；对照 OpenEvolve、ShinkaEvolve 及官方基准进行评测。

**📈 对比分析**

与传统基于程序空间搜索的进化方法相比，所提方法在 120 次评估预算内平均获得 2.10× 的速度提升，MoE kernel 上达到 14.3×，GPUMODE TriMul 任务获得 1030 的几何平均延迟，超越现有手工和自动化方案。

**⚠️ 局限性**

局限性：①性能提升高度依赖 LLM 的质量与提示工程，较弱的模型可能导致搜索效率下降；②共进化过程需要显著的评估成本，尽管相对高效，但在极大搜索空间下仍受限；③缺乏理论收敛保证，搜索结果对硬件特性和任务细节的泛化能力待进一步验证。

---

## 154. Incremental Transformer Neural Processes

**arXiv ID:** 2602.18955 | [PDF](https://arxiv.org/pdf/2602.18955v1)

**作者:** Philip Mortimer `[一作]` (University of Cambridge), Richard E. Turner `[通讯]` (University of Cambridge)

**通讯引用:** 6132 | [OpenAlex ID](https://openalex.org/A5108284020)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Incremental Transformer Neural Process（incTNP），通过因果遮罩和 KV 缓存实现可线性增量更新的神经过程模型。

**💡 创新点**

创新点在于：①在 TNP 中引入因果遮罩实现递归增量更新；②使用 KV 缓存将每步计算从 O(N²) 降到 O(N)；③提出密集自回归训练策略（incTNP-Seq），在单次前向传播中学习所有上下文长度；③用“隐式贝叶斯度量”评估因果遮罩对预测一致性的影响。

**🔧 技术方法**

主要技术包括 Transformer 自注意力、因果遮罩、KV 缓存、密集自回归训练、KL Gap 评估、以及 AR（自回归）部署模式。

**📊 数据集**

数据集：合成 1D GP 任务、合成与四个真实 UCI 回归数据集（Skillcraft、Powerplant、Elevators、Protein）、以及实测气温预测任务（非洲与欧洲气象站）。

**📈 对比分析**

与标准 TNP‑D、incTNP‑Seq、CNP、LBANP 等基线进行对比。incTNP 在大多数任务上能匹配或优于 TNP‑D，且在 AR 模式下实现 3–4 倍的推理速度提升；在温度预测的插值与预测场景中，incTNP 的 KL Gap 与非递增 TNP 相近，且在真实数据上表现出更好的精度。

**⚠️ 局限性**

局限性包括：①牺牲了对上下文集合的置换不变性，导致在某些条件下的非一致性；②KV 缓存仍需占用显存，极大规模下可能成为瓶颈；③模型在非平稳或分布漂移场景下的适应速度尚未完全验证。

---

## 155. Stop Preaching and Start Practising Data Frugality for Responsible Development of AI

**arXiv ID:** 2602.19789 | [PDF](https://arxiv.org/pdf/2602.19789v1)

**作者:** Sophia N. Wilson `[一作]` (University of Copenhagen), Sebastian Mair `[通讯]` (Linköping University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文主张机器学习社区应从空谈到实际实践数据节约，强调在训练大模型时应减少不必要的数据量以降低能耗与碳排放。

**💡 创新点**

创新点在于将数据生命周期的环境成本量化（以ImageNet-1K为例），并实证展示通过核心集（coreset）子集选择既能保持性能，又显著降低训练时间、能耗和存储成本，同时可缓解数据偏差。

**🔧 技术方法**

采用核心集子集选择技术（如Dyn‑Unc、InfoMax等），配合Carbontracker等能耗跟踪工具进行能耗评估，并在不同模型架构上进行对比实验。

**📊 数据集**

主要使用ImageNet-1K（约130 GB，128万张图像）进行实验，并参考Color‑MNIST做偏差治理示例。

**📈 对比分析**

与完整数据集及随机子集比较，核心集方法在25–35%裁剪下Top‑1准确率基本不变；训练时间降低约24–40%，能耗降低约24–33%，验证了节能与性能的良好平衡。

**⚠️ 局限性**

局限性包括核心集构造的前期计算开销、对不同任务（尤其生成模型）的适用性不确定、实验集中在图像数据而非文本或多模态数据，且未考虑数据收集阶段的能耗。

---

## 156. Charting the Future of AI-supported Science Education: A Human-Centered Vision

**arXiv ID:** 2602.18471 | [PDF](https://arxiv.org/pdf/2602.18471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 157. NILE: Formalizing Natural-Language Descriptions of Formal Languages

**arXiv ID:** 2602.19743 | [PDF](https://arxiv.org/pdf/2602.19743v1)

**作者:** Tristan Kneisel `[一作]` (Ruhr University Bochum), Thomas Zeume `[通讯]` (Ruhr University Bochum)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种名为Nile的描述性形式语言，用来将学生用自然语言描述的形式语言转换为可计算的表达式，从而实现对描述正确性的自动判定和错误说明；

**💡 创新点**

提出Nile能够与自然语言语法结构紧密对齐的语法，结合大语言模型实现高精度的自然语言→Nile翻译，并通过语法树差异提供可解释的反馈；

**🔧 技术方法**

使用OpenAI gpt‑oss‑120b大语言模型进行自然语言到Nile/正则/CFG的翻译；实现Nile语法及其语义评估；利用语法树比较算法生成错误解释；

**📊 数据集**

基于包含2280条学生自然语言描述的教育数据集，覆盖11种形式语言（正则、集合、有限自动机、上下文无关文法等），其中2040条为有效描述；

**📈 对比分析**

通过对比三种模型输出方式（直接判断、转为正则/CFG、转为Nile表达式）在三大研究问题中的准确率进行评估；结果显示Nile表达式翻译语义与语法匹配的准确率≥85%，正则/CFG翻译仅24‑80%，LLM判断描述正确性的准确率约95%；

**⚠️ 局限性**

仅能表达正则和有界上下文无关语言，无法覆盖回文等完全无界 CFL；性能受具体语言和提示覆盖程度影响，提示不足时准确率显著下降；LLM对CFG翻译仍表现不佳。

---

## 158. Designing and Implementing a Comprehensive Research Software Engineer Career Ladder: A Case Study from Princeton University

**arXiv ID:** 2602.19353 | [PDF](https://arxiv.org/pdf/2602.19353v1)

**作者:** Ian A. Cosden `[一作]` (Princeton University), Joel U. Bretheim `[通讯]` (Princeton University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

在普林斯顿大学设计并实施了一套从助理到首席级别的研究软件工程师（RSE）职业阶梯，并对其进行了标准化职位描述、薪酬对齐和多条职业路径的规划。

**💡 创新点**

创新点在于：①创建了兼具技术深度与管理分支的双轨制阶梯；②通过与外部咨询公司合作，将RSE职位与行业市场水平进行对标；③利用统一的语言框架和职责分配比例，使RSE岗位在HR体系中更易归类、晋升更透明；④将职业发展与科研产出、技术领导和社区参与相结合。

**🔧 技术方法**

采用的方法主要是：需求调研与访谈、职责矩阵与能力模型构建、职位描述编写与标准化、外部市场基准对标、HR层面审核与批准、试点评估、年度迭代更新。

**📊 数据集**

使用的数据集包括：RSE团队成员名单、职位层级与薪酬信息、人员流动率（在岗与离职数据）、项目与资助信息、内部调查反馈（2021与2025年问卷）。

**📈 对比分析**

对比方法：通过计算实施前后两阶段的离职率（Turnover Rate）来评估留任效果；通过审批周期（从岗位创建到批准）来衡量招聘效率。结果显示：实施前5年离职率约70%，实施后3年约7%；审批周期从数月缩短至数周，显著提升了招聘效率。

**⚠️ 局限性**

局限性包括：①案例仅适用于普林斯顿这样规模相对较小、集中管理的研究型大学，难以直接推广至更大、分散的机构；②阶梯设计仍需根据项目资助情况和资源限制进行微调；③对RSE产出与绩效的量化评估仍不够完善，需要结合项目特性和软技能综合考核；④外部市场基准的更新周期有限，需定期审视薪酬与市场的匹配度。

---

## 159. Turbo Coded Single Sideband OFDM-OQAM Signaling through Frequency Selective Rayleigh Fading Channels

**arXiv ID:** 2602.18881 | [PDF](https://arxiv.org/pdf/2602.18881v1)

**作者:** Kasturi Vasudevan `[一作]` (Indian Institute of Technology Kanpur), Kasturi Vasudevan `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 769 | [OpenAlex ID](https://openalex.org/A5049326382)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种基于单边带调制的OFDM‑OQAM系统，并设计了离散时间的帧检测、两步频偏估计、通道与噪声方差估计以及Turbo译码算法。

**💡 创新点**

通过使用根升余弦脉冲及其改进的Hilbert变换实现单边带调制，减小带宽一半，同时采用匹配滤波器简化接收机，并提出两步频偏估计的并行可扩展算法。

**🔧 技术方法**

OFDM‑OQAM、单边带（SSB）调制、根升余弦脉冲与Hilbert变换、匹配滤波、最大似然频偏估计、基于BCJR的Turbo译码，及其在MIMO环境中的潜在扩展。

**📊 数据集**

采用仿真频率选择性Rayleigh衰落模型、AWGN以及随机训练位分布的帧结构；未使用公开数据集。

**📈 对比分析**

通过仿真比较不同子载波多样性（N_sc=1–4）的误码率，结果显示随N_sc增大误码率提升约10 dB至1 dB，系统在带宽和复杂度上相较传统OFDM‑OQAM具有优势。

**⚠️ 局限性**

性能受限于通道失真导致的ISI，需提高插值因子或采用MIMO以改善；子载波多样性增加会导致系统带宽与功耗上升。

---

## 160. MagicAgent: Towards Generalized Agent Planning

**arXiv ID:** 2602.19000 | [PDF](https://arxiv.org/pdf/2602.19000v1)

**作者:** Xuhui Ren `[一作]` (Honor Device Co), Yunke Zhang `[通讯]` (Honor Device Co)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MagicAgent系列基础模型，专注于通用规划能力，涵盖层级任务分解、工具增强规划、多约束调度、程序逻辑编排和长周期工具执行；

**💡 创新点**

创新点包括：1）轻量化可扩展的合成数据框架，覆盖多种规划场景；2）两阶段训练策略（SFT+多目标RL，含在线χPO）缓解多任务梯度冲突；3）基于全局负载平衡与路由正则化的MoE优化，解决专家不平衡；

**🔧 技术方法**

技术手段包括：大规模LLM预训练、参数高效微调（LoRA/QLoRA）、离线与在线强化学习（χPO），全局负载平衡与z‑loss的MoE训练；

**📊 数据集**

使用公开基准（Worfbench、NaturalPlan、τ^2-Bench、BFCL‑v3、ACEBench）和公司内部MagicEval‑Plan/Tool数据集进行评估；

**📈 对比分析**

与超大规模（>100B）及大型（≤100B）模型对比，MagicAgent-32B/30B-A3B在大多数指标上显著领先（例如Worfbench F1 Chain 80.3%，BFCL‑v3 Live 84.1%，MagicEval‑Plan Step 98%），表现优于现有公开及闭源基线；

**⚠️ 局限性**

局限性包括：1）对真实复杂环境的适应仍有限，长周期推理仍有提升空间；2）依赖合成数据，可能缺乏某些真实交互细节；3）大规模模型训练成本高，部署时需权衡稀疏化与精度。

---

## 161. DeepInterestGR: Mining Deep Multi-Interest Using Multi-Modal LLMs for Generative Recommendation

**arXiv ID:** 2602.18907 | [PDF](https://arxiv.org/pdf/2602.18907v1)

**作者:** Yangchen Zeng `[一作]` `[通讯]` (Southeast University), Yangchen Zeng (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于生成式推荐的框架 DeepInterestGR，利用多模态大型语言模型挖掘用户深层兴趣，并将兴趣信息嵌入语义 ID 生成流程，从而提升个性化推荐效果。

**💡 创新点**

核心创新点包括：① 多 LLM 交叉挖掘（MLIM）通过 Chain‑of‑Thought 提示提取多维深层兴趣；② 利用二分类器给兴趣打标签（RLDI），为 RL 提供质量监督；③ 将兴趣编码成 SID（IEID）并在奖励函数中加入兴趣匹配，形成兴趣感知奖励。

**🔧 技术方法**

使用前沿 LLM（GPT‑5.1、Gemini‑3‑Pro、Kimi‑K2‑Thinking、Grok‑4 及其多模态变体）进行兴趣提取，Qwen3‑Embedding 对兴趣文本编码，RQ‑VAE 量化为 SID，GRPO 强化学习与兴趣感知奖励相结合，并以 Qwen2.5‑7B‑Instruct 作为生成基座。

**📊 数据集**

在亚马逊产品评论的 Beauty、Sports、Instruments 三个子域数据集上进行实验，采用 5‑core 过滤与 leave‑last‑out 评估。

**📈 对比分析**

与传统序列模型（GRU4Rec、Caser、HGN）、Transformer 模型（SASRec、BERT4Rec、S³‑Rec、FDSA）、生成式推荐（TIGER、LC‑Rec、HSTU、MiniOneRec）以及 LLM‑based 模型（BIGRec、D3、S‑DPO）等多类基线对比，DeepInterestGR 在 HR@K 与 NDCG@K 上平均提升 9.2%–15.1%，并在跨域泛化实验中实现 24.8%–27.3% 的显著提升。

**⚠️ 局限性**

主要局限包括：① 对多模态 LLM API 的高成本与可用性依赖；② 生成式兴趣提取可能出现幻觉或偏差，影响兴趣标签的准确性；③ 计算与存储成本相对较高，尤其在大规模实时系统中的部署尚未验证；④ 对极度稀疏或无视觉信息的项目支持不足。

---

## 162. EvalSense: A Framework for Domain-Specific LLM (Meta-)Evaluation

**arXiv ID:** 2602.18823 | [PDF](https://arxiv.org/pdf/2602.18823v1)

**作者:** Adam Dejl `[一作]` (Imperial College), Jonathan Pearson `[通讯]` (NHS England)

**通讯引用:** 3082 | [OpenAlex ID](https://openalex.org/A5037916338)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EvalSense框架，用于在自定义任务上系统化评估大型语言模型，提供交互式评估方法选择与自动化元评估工具。

**💡 创新点**

创新点在于将交互式评估指南与基于扰动的元评估结合，扩展Inspect框架并支持多模型、多评估器及自定义数据集，帮助用户在特定领域精准挑选评估方法。

**🔧 技术方法**

主要技术包括LLM-as-a-judge方法（G-Eval、QAGS）、统计指标（BLEU、ROUGE、BERTScore）、扰动元评估、Python API与vLLM推理、可视化用户界面。

**📊 数据集**

使用了ACI-Bench临床对话转结构化笔记数据集（120个样本）进行案例研究。

**📈 对比分析**

在13种评估器变体（统计、BERTScore、G-Eval、QAGS）中，元评估显示Gemma 3与Llama 3.1的G-Eval、以及两种QAGS版本相关性最高，统计与BERTScore表现最差；不同方法得分不一致，强调需基于元评估选择。

**⚠️ 局限性**

局限性包括：需人工配置评估器与扰动策略，评估效果受所选LLM、prompt与参数影响；仅在单一医疗对话案例验证，缺乏跨任务与跨领域的广泛评估。

---

## 163. A Unified Framework for Weighted Hypergraphic Networks and Fractional Matching

**arXiv ID:** 2602.18779 | [PDF](https://arxiv.org/pdf/2602.18779v1)

**作者:** Rémi Castera `[一作]` (Moroccan Center for Game Theory), Rida Laraki `[通讯]` (Moroccan Center for Game Theory)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了在超图（多元关系）以及存在预算约束的情形下对网络形成的稳定性进行定义，并在此基础上引入了更强的“全稳定”概念。通过严谨的数学证明给出了稳定性与全稳定性的存在性定理、反例以及在特定假设下的算法解。

**💡 创新点**

创新点在于：① 将传统的二元关系稳定性扩展到超图及预算约束；② 在此框架下提出全稳定概念，弥补了单边/双边偏差无法捕捉多边资源再分配的不足；③ 与匹配理论建立统一桥梁，展示全稳定等价于分数匹配中的稳定性；④ 给出在无约束/正外部性、以及有约束/正外部性、凸性假设下的存在性证明与求解算法。

**🔧 技术方法**

主要技术手段包括：变形的固定点与可偏序论证（Yannelis–Prabhakar理论）；对偏序与协同变化的可行性映射构造；对超图网络的可变性与约束化的数学建模；对匹配理论的多元推广；以及基于 Row‑Greedy 的迭代求解算法。

**📊 数据集**

无数据集，本文完全基于理论分析与数学证明。

**📈 对比分析**

通过构造反例与证明，阐明在不同假设下是否存在稳定网络；在满足凸性、正外部性等条件时给出算法并证明收敛；对算法的性能仅在理论上讨论（可能需无穷步收敛、未给出具体运行时间）。

**⚠️ 局限性**

限制与开放问题包括：
• 全稳定性在存在负外部性或非凸性时不一定存在；
• 在一般图或超图且有预算约束时是否存在全稳定网络仍未解决；
• 所提出的迭代算法收敛可能需要无穷多步，缺乏多项式时间保证；
• 对非二分图（或超图）非线性支付的高效算法尚未实现；
• 该框架未给出对具体实际数据或模拟实验的验证。

---

## 164. Janus-Q: End-to-End Event-Driven Trading via Hierarchical-Gated Reward Modeling

**arXiv ID:** 2602.19919 | [PDF](https://arxiv.org/pdf/2602.19919v1)

**作者:** Xiang Li `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出端到端事件驱动交易框架 Janus‑Q，直接将细粒度金融新闻事件映射为可执行的交易决策。

**💡 创新点**

创新点在于构建大规模细粒度事件数据集，并通过层次门控奖励模型（HGRM）将事件语义与多维交易收益对齐，从而实现事件到决策的完整闭环。

**🔧 技术方法**

使用的技术包括监督预训练 + 强化学习（GRPO），层次门控奖励（HGRM），事件到CAR的统计建模（市场模型与风险模型），以及多任务学习与结构化奖励设计。

**📊 数据集**

使用的数据集为 62,400 篇人工标注的金融新闻，包含 10 类事件、对应股票、情感标签及累计异常收益（CAR），并结合 Tushare 价格数据和 Wind 公司信息。

**📈 对比分析**

通过与 16 种基线（市场指数、时间序列 LLM、金融 LLM、通用 LLM）在 MAE、RMSE、方向准确率、事件类型准确率、Sharpe 比率、最大回撤等六项指标上的对比，Janus‑Q 在 Sharpe Ratio 上提升 102%，方向准确率提高 17.5%，最大回撤与最佳基线相近，整体表现显著优于所有对照方法。

**⚠️ 局限性**

局限性包括对事件识别误差的敏感性、奖励阈值与超参数的手工设定、以及模型在不同市场周期或跨境市场的泛化能力尚未得到充分验证。

---

## 165. Anatomy of Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning

**arXiv ID:** 2602.19612 | [PDF](https://arxiv.org/pdf/2602.19612v1)

**作者:** Borisiuk Anna `[一作]` (AIRI), Elena Tutubalina `[通讯]` (ISP RAS Research Center for Trusted AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含28.6k Wikidata事实问答对并标注事实流行度的双阶段机器忘记基准，系统评估预训练与指令微调（SFT）模型在不同流行度事实上的忘记效果。

**💡 创新点**

首次将事实流行度作为忘记难度的因素，系统比较了预训练与SFT模型在流行度不同的数据上的忘记表现，并证明预训练先做SFT可提升忘记稳定性。

**🔧 技术方法**

使用Gradient Ascent、Gradient Difference、Negative Preference Optimization三种常见的机器忘记算法，并在LLaMA‑3.1‑8B（以及Gemma‑7B）上进行实验；评估指标为ROUGE‑L、MMLU、HellaSwag。

**📊 数据集**

基准数据来自57k Wikidata事实三元组，经过筛选、QA生成后得到28.6k QA对；流行度由Wikipedia页面链接数与LLM显著性评分确定。

**📈 对比分析**

对比方法：在预训练和SFT两种模型上分别对热门和稀有事实进行1%、5%、10%比例的忘记任务，记录忘记集ROUGE‑L下降和保持集ROUGE‑L提升；结果显示预训练模型对热门事实往往反向学习，SFT模型忘记更平滑，保持集损失降低10‑50%。

**⚠️ 局限性**

局限性包括：仅验证了LLaMA‑3.1‑8B及Places City子集，未覆盖更大模型和更多主题；流行度标注依赖Wikipedia和模型显著性，可能随时间或语言变化；评估指标局限于ROUGE‑L，缺乏人类事实性与安全性评估。

---

## 166. EDU-MATRIX: A Society-Centric Generative Cognitive Digital Twin Architecture for Secondary Education

**arXiv ID:** 2602.18705 | [PDF](https://arxiv.org/pdf/2602.18705v1)

**作者:** Wenjing Zhai `[一作]` (High School Affiliated to Beijing Normal University), Tao Liu `[通讯]` (North China Electric Power University)

**通讯引用:** 37740 | [OpenAlex ID](https://openalex.org/A5100338122)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了面向中学教育的社会中心生成式认知数字孪生架构 EDU-MATRIX，构建了包含 2400 名学生、300 名教师和 100 名校友的数字孪生校园，模拟 30 天的社会交互和价值传承。

**💡 创新点**

创新点在于将教育模拟从“个体化规则硬编码”转向“社会空间+引力场”模型：1）环境上下文注入引擎（ECIE）将机构规则视为引力场，动态注入到智能体；2）模块化逻辑演化协议（MLEP）把知识视为流体胶囊，实现知识融合与逻辑资产化；3）角色拓扑与内生对齐机制让价值在社会网络中自洽而非依赖外部过滤。

**🔧 技术方法**

使用的技术包括：Gemini 大语言模型、Google AI Studio 接口、数据库‑per‑plugin 架构、数字孪生核心微内核、神经握手界面（Human‑in‑the‑loop 冲突调解）、多层次（Meta‑Agent → Domain Agent → Student Agent）知识流循环和可视化监控工具（神经拓扑图、KAI Digital Facilitator）。

**📊 数据集**

数据集主要来自于内部生成的模拟环境（校园实体、角色关系、历史记忆），并未采用公开教育或社交网络数据集；实验使用 2,400 名学生、300 名教师和 100 名校友的虚拟代理组成的社会图谱。

**📈 对比分析**

通过对比实验与对照组，评估指标包括：社会聚类系数 0.72、全局共振同步率 98.4%、对话一致性 94.1% 和价值注入效能 +42%（社会贡献讨论比对照组高 42%）。实验表明系统在价值对齐、行为一致性和安全性方面表现优秀，达到了高度真实的社交网络仿真。

**⚠️ 局限性**

局限性主要包括：1）计算成本高，需进一步优化模型推理和资源调度；2）验证仅在北京师范大学附属中学的文化背景下，缺乏跨校和跨文化的泛化评估；3）多校园跨域认知联动机制尚未实现；4）人‑AI 共进化中的实时干预和教育者体验尚待完善。

---

## 167. Multilingual Large Language Models do not comprehend all natural languages to equal degrees

**arXiv ID:** 2602.20065 | [PDF](https://arxiv.org/pdf/2602.20065v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 168. Dark and Bright Side of Participatory Red-Teaming with Targets of Stereotyping for Eliciting Harmful Behaviors from Large Language Models

**arXiv ID:** 2602.19124 | [PDF](https://arxiv.org/pdf/2602.19124v1)

**作者:** Sieun Kim `[一作]` (KAIST), Hwajung Hong `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计并实施了针对韩国地区大学毕业生（被社会标签为“jibangdae”）的参与式红队实验，评估其在识别LLM偏见时的心理成本与赋权效果；

**💡 创新点**

首次系统性探讨刻板印象目标群体参与红队的双重影响，提出结合心理安全与赋能的设计方案，填补了红队方法在少数群体中的空白；

**🔧 技术方法**

采用混合方法：量化心理测评（K-PANAS、RSES、CSES、SCQ、SUDS、NASA‑TLX）与质性访谈及红队攻击日志分析；

**📊 数据集**

未使用公开数据集，而是让受试者使用其熟悉的LLM（如ChatGPT等）进行攻击，收集对应的对话与攻击记录；

**📈 对比分析**

通过比较攻击成功率（平均每人2.6次）与心理量表变化来评估效果，发现攻击成功与心理负担呈正相关，但未设置对照组；

**⚠️ 局限性**

样本量有限（20人）、单一文化背景、缺乏长期随访与对照实验，且在实验中暴露受害者身份可能带来伦理风险。

---

## 169. Phase-Consistent Magnetic Spectral Learning for Multi-View Clustering

**arXiv ID:** 2602.18728 | [PDF](https://arxiv.org/pdf/2602.18728v1)

**作者:** Mingdong Lu `[一作]` (Dalian University of Technology), Liang Zhao `[通讯]` (Dalian University of Technology)

**通讯引用:** 9071 | [OpenAlex ID](https://openalex.org/A5069974428)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种无监督多视图聚类方法，通过相位一致的磁性谱学习生成共享结构信号来指导表示学习和跨视图对齐。

**💡 创新点**

创新点在于将跨视图的方向一致性编码为复数相位，构造磁性拉普拉斯谱以获得稳定共享谱；同时采用锚点超图实现高阶共识和可扩展性，并将谱作为结构自监督与标签一致性对比学习的目标。

**🔧 技术方法**

使用了多视图自编码器、锚点超图、曲率驱动的 Ricci 流加权、复值磁性拉普拉斯谱、学生t软分配、KL对齐、以及标签一致性对比学习等技术。

**📊 数据集**

在十个公开多视图基准上评估，数据集包括 100Leaves、Caltech-5V、Digit-Product、ALOI、BDGP、Fashion-MV、UCI-3V、Handwritten、CUB 和 Multi-COIL-20。

**📈 对比分析**

与 K‑means、DEC、DMCAG、AEMVC、ALPC、hubREP、DCMVC、STCMC‑UR 等基线进行比较，使用 ACC、NMI、ARI 等指标，本文方法在绝大多数数据集上均取得最高或次高分，显著提升聚类性能。

**⚠️ 局限性**

局限性包括：相位估计对视图差异和噪声敏感；锚点选择与超参数可能影响结果；超图构造与谱计算仍有一定计算开销；未在极大规模数据上进行可扩展性评估。

---

## 170. Beyond Pass-by-Pass Optimization: Intent-Driven IR Optimization with Large Language Models

**arXiv ID:** 2602.18511 | [PDF](https://arxiv.org/pdf/2602.18511v1)

**作者:** Lei Qiu `[一作]` (Institute of Computing Technology), Xiaobing Feng `[通讯]` (Institute of Computing Technology)

**通讯引用:** 5035 | [OpenAlex ID](https://openalex.org/A5011365911)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种 Intent‑Driven IR Optimizer（IntOpt），将 IR 优化拆分为意图形成、细化和实现三阶段，实现全局协调的编译器优化；

**💡 创新点**

首次明确引入优化意图抽象，并利用 LLM 生成意图后结合编译器分析细化，从而获得可验证、可解释且能够发现传统 pass‑by‑pass 无法获取的优化机会；

**🔧 技术方法**

采用专门的 LLM（如 GPT‑5、LLM‑Compiler FTD‑13B）生成与细化优化意图，利用 LLVM 结构化知识库和分析结果进行信息检索和约束；

**📊 数据集**

基于 IR‑OptSet 数据集构建训练、验证与测试集（分别 4000、500、200 个程序），并使用 Alive2 与 LibFuzzer 进行语义验证与差异测试；

**📈 对比分析**

与多种 end‑to‑end LLM 优化器（GPT‑5、Claude Haiku 4.5、DeepSeek‑V3.2、LLM‑Compiler FTD）以及 LLVM 19.1.0 -O3 进行对比；在 200 程序测试集上实现 90.5% 的正确率，平均 2.660× 的速度提升，并在 37 个基准上超过 LLVM -O3，最高加速达 272.60×；

**⚠️ 局限性**

受限于 LLM 的生成质量和训练数据覆盖，细化与实现阶段对编译器分析的依赖导致计算开销较大；在极其复杂或包含外部非标准调用的程序上，验证与优化效果有限；

---

## 171. SHIELD: Semantic Heterogeneity Integrated Embedding for Latent Discovery in Clinical Trial Safety Signals

**arXiv ID:** 2602.19855 | [PDF](https://arxiv.org/pdf/2602.19855v1)

**作者:** Francois Vandenhende `[一作]` (ClinBAY Limited), Ellie Karekla `[通讯]` (ClinBAY Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了 SHIELD 方法，结合贝叶斯信息量计算与语义嵌入对临床试验 AE 进行分群并生成可解释的安全信号网络。

**💡 创新点**

创新点在于将多臂临床试验的失衡信息量与 MedDRA 语义相似度整合，通过谱聚类和 LLM 注释实现自动化的“症候群”安全信号发现。

**🔧 技术方法**

使用信息组件 (IC)、经验贝叶斯收缩、语义嵌入、谱嵌入+层次 Ward 聚类、以及大型语言模型进行簇标注。

**📊 数据集**

采用杜氏肌营养不良症 Phase III 临床试验（NCT05096221）作为示例数据集。

**📈 对比分析**

与传统的单纯统计或 SMQ 归类方法相比，SHIELD 能够恢复已知肝脏相关 AE 信号，并通过网络图和分层树提供更直观的多维安全概览，表现为准确捕获肝脏症候群及其在不同治疗组中的差异。

**⚠️ 局限性**

局限性包括缺乏在多种试验类型中系统评估性能、未考虑 AE 的严重度、持续时间等属性、以及对单一 MedDRA 词汇表的依赖。

---

## 172. Botson: An Accessible and Low-Cost Platform for Social Robotics Research

**arXiv ID:** 2602.19491 | [PDF](https://arxiv.org/pdf/2602.19491v1)

**作者:** Samuel Bellaire `[一作]` (University of Michigan), Samir Rawashdeh `[通讯]` (University of Michigan)

**通讯引用:** 1014 | [OpenAlex ID](https://openalex.org/A5074948543)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个低成本、可复制的具有人形外观、基于LLM的社交机器人平台 Botson。

**💡 创新点**

首次将云端 LLM 与物理机器人实时结合，并通过单一推理步骤同时产生文本回应和情感标签，生成情感一致的多模态行为。

**🔧 技术方法**

使用 Raspberry Pi、Arduino、3D 打印底盘、GPT‑4o、Speech‑to‑Text、Text‑to‑Speech 以及 Arduino 控制的关节姿态等技术。

**📊 数据集**

未使用公开数据集，使用本地语音输入与 GPT‑4o 的对话历史作为输入。

**📈 对比分析**

与仅语音的 ChatGPT（同为 GPT‑4o）对比，用户认为语音版更有帮助，但在自然对话流畅度和语音质量上落后，用户更倾向于语音版。

**⚠️ 局限性**

局限包括：语音识别误差导致回应不准确、机器人语音较僵硬、手势意义不够明确、实验样本量小以及缺乏真实情绪表达。

---

## 173. CTS-Bench: Benchmarking Graph Coarsening Trade-offs for GNNs in Clock Tree Synthesis

**arXiv ID:** 2602.19330 | [PDF](https://arxiv.org/pdf/2602.19330v1)

**作者:** Barsat Khadka `[一作]` (University of Southern Mississippi), Md Rubel Ahmed `[通讯]` (Louisiana Tech University)

**通讯引用:** 157 | [OpenAlex ID](https://openalex.org/A5084578583)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 CTS-Bench benchmark，用于系统评估图 coarsening 对 GNN 在 Clock Tree Synthesis (CTS) 分析中的精度与效率权衡，并提供配对的原始门级图和聚类图及其对应的 CTS 结果。

**💡 创新点**

创新点包括：①提供大规模（4,860 条）原始与压缩图数据，②引入 Pareto Gap 归一化评估指标量化 Placement-CTS 交互，③开放可复现的自动化数据生成框架，④通过多尺度图对比揭示通用聚类对局部时钟偏移预测的破坏性影响。

**🔧 技术方法**

使用技术：OpenLane+OpenROAD+TritonCTS 生成物理设计；三步聚类算法（Atomic BFS、High-Spread Filtering、Gravity‑Vector‑Aligned Merging）得到压缩图；Graph Neural Networks（GCN、GraphSAGE、GATv2）与多模态融合网络进行多目标回归；PyTorch Geometric 存储和训练；GPU 计费指标（VRAM、训练时间）与预测指标（MAE、R²）进行评估。

**📊 数据集**

数据集：5 种架构（PicoRV32、AES、SHA256、EthMAC、Zipdiv）共 486 个 Placement，分别生成 10 个 CTS 变体，合计 4,860 条实例；每条实例配有原始门级图、聚类图、15 个CTS QoR 目标及超参数。

**📈 对比分析**

比较方法：对 Raw 与 Clustered 两种图在同一 GNN 基础上进行训练和测试，测量 VRAM 使用、训练吞吐、MAE 与 R²；在 Seen（4 种已训练架构）与 Unseen（Zipdiv）两组数据上评估泛化。结果显示：Clustered 图可将 VRAM 降至 17.2×、训练时间提升约 3×；但对全局指标（功耗、线长）MAE 影响不大；对局部指标（时钟偏移）R² 下降甚至负值；Unseen 上 Raw 仅维持 0–0.2 的 R²，Clustered 更差。

**⚠️ 局限性**

limitations：①通用聚类削弱了对局部时钟偏移的表征；②在未知架构上的泛化性能差；③缺乏任务感知或学习型聚类策略；④未覆盖多 VT、宏、非均匀电源网格等更复杂的工业工艺特征。

---

## 174. Towards Automated Page Object Generation for Web Testing using Large Language Models

**arXiv ID:** 2602.19294 | [PDF](https://arxiv.org/pdf/2602.19294v1)

**作者:** Betül Karagöz `[一作]` (Technical University of Munich), Andrea Stocco `[通讯]` (Technical University of Munich)

**通讯引用:** 2496 | [OpenAlex ID](https://openalex.org/A5027652385)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究利用 GPT‑4o 与 DeepSeek Coder 在五个真实 Web 应用上自动生成 Page Object，并通过人工比对评估其准确率与元素识别率。

**💡 创新点**

首次系统化地验证大语言模型在 PO 自动化生成中的可行性，并提供可复现的实验库与详细的缺陷分类。

**🔧 技术方法**

使用零样本提示技术、结构化提示模板、HTML 预处理以及 LLM 生成的 Java Selenium PO 代码。

**📊 数据集**

基于公开的五个 Web 应用基准（Bludit、Kanboard、MediaWiki、PrestaShop、ExpressCart）及其手工编写的 PO 作为真值集。

**📈 对比分析**

采用“正确/待改/缺失/额外”四类评分类别，计算方法与元素的准确率（32.6–54.0%）与元素识别率（61.2–94%），两模型整体表现相近，DeepSeek 在多数指标略优。

**⚠️ 局限性**

局限包括仅评估两款 LLM、仅使用静态 HTML、缺乏动态上下文与多样化提示策略、数据集规模有限以及模型可能的训练集泄漏。

---

## 175. OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness

**arXiv ID:** 2602.19035 | [PDF](https://arxiv.org/pdf/2602.19035v1)

**作者:** Phuc D. A. Nguyen `[一作]` (University of Maryland), Ming C. Lin `[通讯]` (University of Maryland)

**通讯引用:** 16967 | [OpenAlex ID](https://openalex.org/A5102878981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 OpenVO，一种能够在无标定摄像头和不同帧率下从一阶视角视频中实时估计真实世界尺度的视觉里程计框架。

**💡 创新点**

核心创新在于引入时间感知流编码器和可微分 2D 引导 3D 流，并结合几何感知上下文编码器，使模型在未见摄像机、未见帧率和无标定条件下实现零样本跨域通用性。

**🔧 技术方法**

使用 MaskFlowNet 提取光流、Metric3Dv2 与 WildCamera 估计深度与内参、Transformer 自注意力网络、时间位置编码、Fisher 矩阵旋转回归以及多时间尺度训练策略。

**📊 数据集**

在 nuScenes 训练集上训练，测试于 KITTI（10 Hz）、nuScenes（12 Hz）未见区域以及 Argoverse2（20 Hz）数据集，并在多种帧率下评估。

**📈 对比分析**

与 TartanVO、DPVO、ZeroVO、XVO 等方法对比，OpenVO 在 KITTI、nuScenes、Argoverse2 上的 ATE、t_err、r_err、s_err 均平均提升约 20–30%，在不同帧率下误差下降 46%–92%。

**⚠️ 局限性**

局限性包括对深度估计和内参推断的依赖，误差会在最终里程计中累积；多帧率训练是经验设定，缺乏自适应采样机制；在极端帧率变化时性能仍有提升空间。

---

## 176. Learning Positive-Incentive Point Sampling in Neural Implicit Fields for Object Pose Estimation

**arXiv ID:** 2602.19937 | [PDF](https://arxiv.org/pdf/2602.19937v1)

**作者:** Yifei Shi `[一作]` (National University of Defense Technology), Kai Xu `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于神经隐式场的正向激励采样方法，结合SO(3)等变卷积网络和估计网络，能够自动生成少量但信息量高的采样点来训练姿态估计模型。

**💡 创新点**

创新点在于①学习式正向激励采样策略，自动筛选具有高估计置信度且几何稳定的点；②引入SO(3)等变卷积实现姿态不变性；③使用各向异性不确定性估计来判定点的价值；④实现了跨任务的采样策略泛化。

**🔧 技术方法**

技术手段包括SO(3)-equivariant 3D图卷积、估计网络（点云编码+体素解码）、Gumbel‑Softmax门控选择、正则化稀疏性与几何稳定性损失、知识蒸馏生成伪标签、利用DINOv2提取RGB特征。

**📊 数据集**

采用三大数据集：NOCS‑REAL275、ShapeNet‑C（新建的高难度数据集）和LineMOD‑O，用于类别级、实例级姿态估计。

**📈 对比分析**

在所有基准上均取得SOTA：NOCS‑REAL275 5°/2cm精度0.63；ShapeNet‑C 5°/5cm精度0.62；LineMOD‑O AR 77.3，超越多数对手，尤其在高遮挡、未知姿态、形状新颖和噪声场景表现突出。

**⚠️ 局限性**

局限性在于需额外训练教师模型生成伪标签、训练过程分阶段耗时、对遮挡导致的姿态歧义处理不足、未统一框架与扩展到多模态渲染场景。

---

## 177. Benchmark Test-Time Scaling of General LLM Agents

**arXiv ID:** 2602.18998 | [PDF](https://arxiv.org/pdf/2602.18998v1)

**作者:** Xiaochuan Li `[一作]` (Carnegie Mellon University), Chenyan Xiong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4847 | [OpenAlex ID](https://openalex.org/A5102363883)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了General AgentBench，一个统一评估多领域LLM代理的基准，并系统研究了其在测试时扩展中的表现。

**💡 创新点**

创新点在于构建统一工具接口，跨域任务评估，并揭示了顺序扩展的上下文上限和并行扩展的验证缺口。

**🔧 技术方法**

采用MCP统一主机、工具注册表、LLM代理、顺序/并行测试时扩展、pass@K与自我选择等技术。

**📊 数据集**

使用四个领域的任务集：编码（SWE-Bench、Terminal Bench）、检索（BrowseComp、WebVoyager）、工具使用（Tau2-Bench、MCP-Bench）以及推理（MathHay）。

**📈 对比分析**

通过与专用域评估对比以及对比顺序/并行扩展方式，结果显示大多数模型在通用评估中下降10-30%，Claude Sonnet 4.5最稳健；并行扩展提升pass@K但自我选择性能不足；顺序扩展出现上下文上限。

**⚠️ 局限性**

局限性包括顺序扩展受限于上下文上限导致性能不稳， 并行扩展受验证缺口限制实际收益；通用评估对模型提出更高挑战，导致大规模模型仍受限。

---

## 178. Bayesian Lottery Ticket Hypothesis

**arXiv ID:** 2602.18825 | [PDF](https://arxiv.org/pdf/2602.18825v1)

**作者:** Nicholas Kuhn `[一作]` (Karlsruhe Institute of Technology), Charlotte Debus `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5024295845)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了贝叶斯神经网络（BNN）在稀疏训练中的可行性，验证并扩展了 Lottery Ticket Hypothesis（LTH）到 BNN。

**💡 创新点**

创新点在于证明 LTH 同样适用于 BNN，并提出基于均值‑方差权重的稀疏度量与迁移策略，将非贝叶斯 LTH 直接迁移至 BNN。

**🔧 技术方法**

主要技术包括均值场变分推断、Iterative Magnitude Pruning（IMP）、多种权重打分策略（SNR、平方和、均值）以及温度缩放的 KL 正则化。

**📊 数据集**

实验使用 CIFAR‑10 数据集，模型为 ResNet‑18、VGG‑11 与 VisionTransformer（小模型）。

**📈 对比分析**

与传统非贝叶斯 LTH 比较，BNN LTH 在大多数稀疏度下保持相同或更优的精度，迁移票据在保持精度的同时将训练时间缩短约 50%，并保持更好的校准。

**⚠️ 局限性**

局限性包括仅在 CIFAR‑10 上验证、使用 VI 实现 BNN 未探究更大规模数据或其他贝叶斯推断方法，以及未评估结构化稀疏训练。

---

## 179. Sub-City Real Estate Price Index Forecasting at Weekly Horizons Using Satellite Radar and News Sentiment

**arXiv ID:** 2602.18572 | [PDF](https://arxiv.org/pdf/2602.18572v1)

**作者:** Baris Arat `[一作]` (Ozyegin University), Emre Sefer `[通讯]` (Ozyegin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了每周的迪拜19个子城地区房价指数，并结合卫星雷达、新闻情绪和宏观变量进行多模态预测，评估了不同预测时延的准确性。

**💡 创新点**

创新点在于首次将Sentinel-1 SAR回波与新闻情绪结合，用多模态框架在子城级别的高频指数上实现长期（26–34周）预测性能提升，并系统化地剖析各模态的贡献与模型选择。

**🔧 技术方法**

采用的技术包括时间序列滚动交叉验证、特征工程（价格、成交量、情绪词典+句子嵌入、SAR统计、利率、全局上下文）、非参数学习（KNN、随机森林、XGBoost）、线性岭回归、LSTM以及基线ARIMA与简单平均。

**📊 数据集**

使用的数据集为2015–2025年迪拜土地部（DLD）公开交易记录（35万笔）、GDELT新闻语料（约27k篇）、Sentinel‑1 SAR（VV/VH）与Sentinel‑2 NDBI、以及阿联酋央行的银行间利率。

**📈 对比分析**

通过滚动时间序列交叉验证与宏平均MAE比较，发现非参数模型在短期内与价格历史相当，但在长周期（26–34周）中加入情绪与SAR后MAE下降约35%，显著优于单一价格模型和ARIMA基线。

**⚠️ 局限性**

局限性包括对迪拜市场的专属性，样本量对深度学习模型不足，外生变量的选择仍有限，且未探索跨地区或图结构依赖的更复杂模型。

---

## 180. A Systematic Evaluation of Environmental Flakiness in JavaScript Tests

**arXiv ID:** 2602.19098 | [PDF](https://arxiv.org/pdf/2602.19098v1)

**作者:** Negar Hashemi `[一作]` (Massey University), Rachel Blagojevic `[通讯]` (Massey University)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5034836091)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

系统地评估了 JavaScript 项目中由操作系统、Node.js 版本和浏览器造成的环境相关 flaky 测试，并基于实验结果开发了一款轻量级的注解式测试过滤工具

**💡 创新点**

提出了通过在测试用例注解中声明可执行环境（OS、Node.js 版本、浏览器）来自动跳过不满足条件的 flaky 测试，从而使 CI 构建不必重新运行整个测试套件

**🔧 技术方法**

实验使用 GitHub Actions 进行多环境多重运行，手工分析 flaky 原因，并实现了一个 Babel 插件（支持 Jest、Mocha、Vitest）来解析注解、检测运行环境并动态跳过测试

**📊 数据集**

收集了约 400 个受欢迎的 GitHub JavaScript 开源项目（涵盖服务器、前端、全栈等领域），共执行 90 次不同 OS/Node/浏览器组合的测试

**📈 对比分析**

通过对比执行前后测试数量、失败率以及工具生成的跳过报告，发现工具能够准确跳过所有因环境导致的 flaky 测试（无 FP/FN），且对执行时间的影响可忽略，单例项目如 fabric.js 甚至因跳过导致总耗时下降

**⚠️ 局限性**

受限于仅挑选流行的 GitHub 仓库、手工分类可能存在偏差、工具仅支持 Jest/Mocha/Vitest 以及排除了一些多工作区或数据库依赖的项目，且未对不同语言或私有仓库的泛化效果做进一步验证

---

## 181. Narrowing the Complexity Gap in the Evaluation of Large Language Models

**arXiv ID:** 2602.18928 | [PDF](https://arxiv.org/pdf/2602.18928v1)

**作者:** Yang Chen `[一作]` (University of Illinois Urbana-Champaign), Reyhaneh Jabbarvand `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 641 | [OpenAlex ID](https://openalex.org/A5058824250)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多目标遗传算法的自动化方法，对现有编程基准进行代码演化，从而在保持可读性的前提下显著提升问题的真实世界复杂度。

**💡 创新点**

创新点在于：①首次将代码层面的可读性与复杂度作为并行优化目标；②设计了22种以语义保持为前提的代码变换算子，涵盖并发、装饰器、第三方 API、类间依赖等高级特性；③无需手工挖掘或维护真实项目即可生成接近真实复杂度的基准。

**🔧 技术方法**

技术栈包括：Python AST 解析与修改、基于流向和上下文的程序分析、NSGA-II 多目标遗传算法、复杂度与可读性指标计算，以及对13种大型语言模型的调用与评测。

**📊 数据集**

使用了四个主流 Python 基准（如 HumanEval、CodeNet、SWE‑Bench、R2E 等）共 714 条程序，并在此基础上生成了多种版本，随后在程序修复、代码翻译、代码推理等四项任务中对 13 个 LLM 进行评测。

**📈 对比分析**

通过对原始基准与演化后基准在相同任务中的表现进行对比，发现 LLM 的准确率普遍下降 14.9%–60.5%（平均 35.2%）；即使采用少量示例微调或少量提示，性能仍下降 4.7%–65.2%，表明该方法能够有效揭示模型在真实复杂度场景下的瓶颈。

**⚠️ 局限性**

局限性包括：目前仅支持 Python，转换过程依赖于高覆盖率的测试集，可能无法完全覆盖所有真实世界的复杂性；对其他语言的迁移仍需工程实现；部分变换可能在极端情况导致语义漂移；评测受限于所选基准与模型的覆盖面。

---

## 182. When Coordination Is Avoidable: A Monotonicity Analysis of Organizational Tasks

**arXiv ID:** 2602.18673 | [PDF](https://arxiv.org/pdf/2602.18673v1)

**作者:** Harang Ju `[一作]` `[通讯]` (Johns Hopkins University), Harang Ju (Johns Hopkins University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并验证了将组织学中任务互依类型与分布式系统中的单调性（CALM）相对应的桥梁定理，从而确定哪些任务在正确性方面不需要协调，并量化可避免的协调开销。

**💡 创新点**

创新点在于将 Thompson 的互依分类映射到 CALM 单调性条件，提供了可计算的决策规则；引入协调税量化结构性无效开销；并通过多智能体模拟验证了理论。

**🔧 技术方法**

主要技术包括理论证明（桥梁定理、反馈边界）、机器学习分类器（GPT‑4.1 mini, Claude Sonnet 4.5）以及多智能体任务模拟与评估。

**📊 数据集**

使用了 APQC 过程分类框架中的 65 个企业工作流程和 O*NET 29.1 数据库中 13,417 个职业任务进行分类。

**📈 对比分析**

通过在三种大型语言模型（GPT‑4.1 mini、Claude Haiku 4.5、Claude Sonnet 4.5）上进行 10 次重复的有/无协调实验，证明非单调任务在无协调情况下 0% 合法率，单调任务维持 100% 合法率，协调开销平均提升 2.3–4.4 倍。

**⚠️ 局限性**

局限性包括仅以正确性为判据忽略质量、对任务分类的静态假设以及 LLM 在实现单调规范时的随机性，误分类对正确性的风险偏向正面；实验环境简化了现实世界的多模态复杂性。

---

## 183. Depth from Defocus via Direct Optimization

**arXiv ID:** 2602.18509 | [PDF](https://arxiv.org/pdf/2602.18509v1)

**作者:** Holly Jackson `[一作]`, Benjamin Recht `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于直接优化的深度从散焦（depth‑from‑defocus）方法，通过交替最小化同时恢复景深图和全景清晰图。

**💡 创新点**

利用深度固定时全景清晰图线性可解、全景清晰图固定时像素级并行深度搜索，将问题化简为凸优化和并行网格搜索，避免了传统正则化。

**🔧 技术方法**

交替最小化、FISTA（Nesterov加速梯度）、稀疏矩阵卷积、网格搜索与黄金分割搜索、并行GPU实现等技术。

**📊 数据集**

NYUv2、Make3D以及Samsung Galaxy S3手机焦距堆栈。

**📈 对比分析**

在NYUv2、Make3D上与监督、无监督、分析方法及单目深度估计进行对比，RMSE 0.109，AbsRel 8.37e‑3，δ1/δ2/δ3 超过99%，显著优于现有最优方法。

**⚠️ 局限性**

对低纹理区域易产生局部伪影；需已知相机标定与薄透镜模型；对焦距离选择敏感；前向模型与稀疏矩阵在大图像上计算量大。

---

## 184. Automatic, Expressive, and Scalable Fuzzing with Stitching

**arXiv ID:** 2602.18689 | [PDF](https://arxiv.org/pdf/2602.18689v1)

**作者:** Harrison Green `[一作]` (Carnegie Mellon University), Claire Le Goues `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7037 | [OpenAlex ID](https://openalex.org/A5032356672)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为 stitching 的可动态拼接 API 调用序列的模糊测试技术，并构建了一个完全自动化的端到端系统，在 1365 个开源项目中发现 131 个新缺陷。

**💡 创新点**

创新点包括：①通过可拼接的代码块和外部状态元数据（extrinsic typestate）来表达任意复杂的 API 使用约束；②使用 LLM 自动从头生成规范、修复规范和调优；③将静态类型和动态元数据结合，实现了既高覆盖又高精度的模糊测试；④全自动化的项目配置、规范推断、崩溃 triage 与修复循环。

**🔧 技术方法**

技术包括：C/C++ 代码块与 JSON 规范；基于 LibAFL 的覆盖引导模糊器；静态类型系统与运行时 extrinsic typestate；LLM（如 GPT‑4）用于规范生成、错误分析与修复；Docker 容器化执行；自动崩溃最小化与 triage。

**📊 数据集**

数据集：33 个主流 C/C++ 基准（来自 prior work），以及 1365 个在 OSS‑Fuzz、Chromium 依赖及 GitHub top‑star 的开源项目；使用的 API 说明文件、源代码、文档等作为输入。

**📈 对比分析**

对比：与 PromeFuzz、PromptFuzz、OGHarn 及 OSS‑Fuzz harness 进行 24h/5次跑测，平均覆盖率最高，在 21/33 任务上显著优于其他工具；在基准上发现 30 个 TP，比其他工具合计的 10 个 TP 多；精度 70% 对比 12%/2%；在真实项目中单机 1–3 天后发现 131 个新缺陷，成本约 2.00 USD/缺陷，速度最快（19分钟构造规范）。

**⚠️ 局限性**

限制：①规范生成依赖 LLM，可能产生错误，需要手动或自动修复；②对极大代码库的构建与调优仍需要时间与资源；③缺陷分类依赖 triage 规则，误报/漏报仍存在；④仅测试库内部 API，未覆盖下游调用；⑤目前仅支持 C/C++ 目标，其他语言尚未支持。

---

## 185. Descriptor: Dataset of Parasitoid Wasps and Associated Hymenoptera (DAPWH)

**arXiv ID:** 2602.20028 | [PDF](https://arxiv.org/pdf/2602.20028v1)

**作者:** Joao Manoel Herrera Pinheiro `[一作]`, Marcelo Becker `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并公开了一个3556张高分辨率图像的寄生蜂数据集，并对1739张图像进行了COCO格式的多类别标注。

**💡 创新点**

数据集融合多视角、多科目且包含完整身体、翅膀、标尺等细粒度分割和边界框标注，为自动化识别提供了高质量的训练资源。

**🔧 技术方法**

采用Leica显微镜+Helicon Focus拍摄、CVAT+SAM标注、YOLOv12、EfficientNetV2等深度学习模型进行图像分类与目标检测。

**📊 数据集**

使用DAPWH数据集（3556张图像，1739张COCO标注），来源于巴西各大学虫子收藏，覆盖11个膜翅目科。

**📈 对比分析**

通过70/15/15的训练/验证/测试划分，比较多种CNN和YOLO模型；YOLOv12在mAP@50达到90.53%，图像分类Top-1 92.28%，F1 95.59%。

**⚠️ 局限性**

仅标注至科级，部分科样本不平衡（如Colletidae）导致模型泛化受限，尺度条标注精度不高，缺乏更细粒度的亚科/属级标注。

---

## 186. Closed-Loop Environmental Control System on Embedded Systems

**arXiv ID:** 2602.19305 | [PDF](https://arxiv.org/pdf/2602.19305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 187. Towards Proving Liveness on Weak Memory (Extended Version)

**arXiv ID:** 2602.19609 | [PDF](https://arxiv.org/pdf/2602.19609v1)

**作者:** Lara Bargmann `[一作]` (Carl von Ossietzky University Oldenburg), Heike Wehrheim `[通讯]` (Carl von Ossietzky University Oldenburg)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了第一套用于弱内存模型下的并发程序终止（liveness）证明计算器，能够对任意线程数的程序进行推理。

**💡 创新点**

在已有的安全性证明框架上扩展了Manna–Pnueli的响应规则，加入了内存公平性与基于弱内存状态的秩函数，并通过势能（potential）逻辑统一多个弱内存模型的验证。

**🔧 技术方法**

使用线性时序逻辑、Manna–Pnueli响应规则、势能（potential）逻辑、记忆公平性、排名函数以及对内部内存步骤的帮助性转换的证明规则。

**📊 数据集**

无具体实验数据集，全部为理论证明与形式化模型。

**📈 对比分析**

未给出实验或性能对比；验证通过理论证明与示例程序（如Ticket锁）展示证明有效性。

**⚠️ 局限性**

局限于已证明可在Release‑Acquire与Strong Coherence模型下有效，对其他弱内存模型（如PSC）尚未完全验证；证明复杂度高，且未处理非安全性相关的终止问题。

---

## 188. Entropy in Large Language Models

**arXiv ID:** 2602.20052 | [PDF](https://arxiv.org/pdf/2602.20052v1)

**作者:** Marco Scharringhausen `[一作]` `[通讯]` (Carl von Ossietzky Universität Oldenburg), Marco Scharringhausen (Carl von Ossietzky Universität Oldenburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对来自Mistral和Blablador两大语言模型族的文本进行大规模生成，并计算其按词的熵率，与美国开放语料库（OANC）中的书面和口语文本进行对比。

**💡 创新点**

首次将熵率作为定量指标评估LLM输出文本的随机性，并系统比较不同模型、不同温度设定下的熵率差异。

**🔧 技术方法**

使用Python实现熵率计算器（上下文长度≤6），并通过指数拟合外推到无限上下文；采用API调用与默认参数生成文本；利用熵率公式与极限定义。

**📊 数据集**

生成文本基于多组提示词（国家、首都、苹果、哺乳动物、一般动物、物理常数）；对照语料为公开的Open American National Corpus（约11.5M书面词、3.1M口语词）。

**📈 对比分析**

对每个模型按词熵率进行比较；结果显示LLM熵率约为0.57–0.62，比OANC书面0.716和口语1.255低；温度变化对熵率影响不显著，Mistral族熵率略低于Blablador族。

**⚠️ 局限性**

局限于上下文长度6导致样本不足，Blablador高温度输出量极少导致估计不准；仅评估默认参数；未深入探讨LLM内部机制或语义质量；样本来源与提示词有限，无法完全代表LLM的整体输出。

---

## 189. How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization

**arXiv ID:** 2602.19208 | [PDF](https://arxiv.org/pdf/2602.19208v1)

**作者:** Yangyi Fang `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DynaMO 双层优化框架：在序列层采用基于梯度方差最小化的动态滚动分配，在 token 层引入梯度-熵关系的优势补偿与稳定机制。

**💡 创新点**

创新点在于：① 从理论推导得到的梯度方差最小化分配策略；② 利用 Bernoulli 方差作为可计算的梯度信息代理；③ 基于梯度-熵上界的梯度补偿与熵变化激活的稳定机制，实现对高置信度动作梯度衰减和过大更新的精细调控。

**🔧 技术方法**

使用了 RLVR 训练框架、GRPO 算法、Softmax 策略梯度、Bernoulli 方差代理、熵变化指标、梯度补偿因子与稳定因子、统一调节超参数 α 等技术。

**📊 数据集**

数据集包括 DAPO-Math-17k 训练集以及 AIME24/25、AMC23、MATH500、Minerva、Olympiad 六个数学推理基准。

**📈 对比分析**

与 GRPO、Clip-Higher、Entropy Loss、Fork Tokens、Entropy Advantages、Clip-COV、KL-COV、W-REINFORCE 等基线在 Qwen2.5-Math-1.5B、7B 以及 Qwen3-14B 上的 Pass@1/Pass@32 指标对比，DynaMO 在所有模型和基准上均实现显著提升，尤其在 7B/14B 上提升幅度更大。

**⚠️ 局限性**

主要限制是实验仅覆盖 Qwen 系列模型，未对其他模型家族或更大规模模型进行系统验证，且方法改动主要在算法层，需进一步评估跨模型可迁移性。

---

## 190. Defining Explainable AI for Requirements Analysis

**arXiv ID:** 2602.19071 | [PDF](https://arxiv.org/pdf/2602.19071v1)

**作者:** Raymond Sheh `[一作]` (Curtin University), Isaac Monteath `[通讯]` (Curtin University)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5072193795)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了可解释人工智能（XAI）的三维分类框架（来源、深度、范围），并结合不同应用场景与机器学习技术讨论其匹配与需求。

**💡 创新点**

创新点在于将已有的解释分类进行整合，形成统一的Source、Depth、Scope三维模型，并针对不同技术（如深度学习、决策树、规则学习）阐述其可解释性能力与局限。

**🔧 技术方法**

主要采用理论分析与归纳方法，对现有XAI文献进行综述，提出新的分类维度；并通过案例讨论（如服务机器人、法证/合规、深度学习、决策树）验证框架。

**📊 数据集**

本研究未使用具体实验数据集，而是基于文献和理论讨论进行推导。

**📈 对比分析**

由于本研究属于概念性与理论性工作，没有直接实验或性能对比；其贡献主要体现在提供了一套可用于需求分析的分类工具，而非数值性能评估。

**⚠️ 局限性**

局限性包括：缺乏对框架的实证验证与量化评估；未给出具体指标与度量方法；仅聚焦于机器学习技术，对更广泛的AI领域尚需扩展。

---

## 191. Insertion Based Sequence Generation with Learnable Order Dynamics

**arXiv ID:** 2602.18695 | [PDF](https://arxiv.org/pdf/2602.18695v1)

**作者:** Dhruvesh Patel `[一作]` (University of Massachusetts Amherst), Andrew McCallum `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 51277 | [OpenAlex ID](https://openalex.org/A5107835063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了可学习的生成顺序动态（Learnable Order Dynamics）用于可变长度插入式生成模型LFlexMDM，能够在插入与掩蔽过程中学习数据依赖的生成顺序。

**💡 创新点**

核心创新在于将可学习的泊松过程到达率与状态转移核相结合，构造可学习的插入与掩蔽时间分布，且不需完整轨迹仿真即可通过一次反向传播同时训练目标速率与生成器。

**🔧 技术方法**

使用离散流匹配（Discrete Flow Matching）框架，结合可学习的Kumaraswamy分布生成时序，采用变压器（Transformer）实现目标速率与生成器网络，配合tau-leaping采样。

**📊 数据集**

实验使用STAR图遍历数据集和SAFE分子生成数据集（包含ZINC和UniChem）。

**📈 对比分析**

与基线FlexMDM相比，在星形图遍历任务中，LFlexMDM在硬难度下生成顺序与手工优选顺序相关性显著提升；在分子生成任务中，LFlexMDM在有效性（validity）和质量（quality）上较FlexMDM提升约10%-15%，但多样性略有下降。

**⚠️ 局限性**

局限性包括：学习速率的自由度可能导致训练不稳定，需要手工冻结部分参数；模型在需要强随机性的任务上可能欠佳；并且目前只在插入与掩蔽两种操作上实验，未扩展到删除或替换等更复杂的编辑操作。

---

## 192. Exact Algorithms for Resource Reallocation Under Budgetary Constraints

**arXiv ID:** 2602.18438 | [PDF](https://arxiv.org/pdf/2602.18438v1)

**作者:** Arun Kumar Das `[一作]` (University of Hyderabad), Nikolaos Melissinos `[通讯]` (Charles University)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5088003340)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了红蓝强化（Red‑Blue Reinforcement）问题，研究在预算约束下最小化客户重新分配以降低所需服务器数量的最优解。

**💡 创新点**

创新点在于首次将此类资源再分配建模为图论中的红蓝图问题，并将其与强化数（reinforcement number）及统治数（domination number）等经典概念相联系；同时给出了三种参数化可行性（FPT）算法，分别基于距离到团、模宽和团宽。

**🔧 技术方法**

主要技术包括参数化复杂度理论、动态规划与组合优化（如对最大覆盖问题的变体做的多维 DP）、模块分解和 clique‑width 表达式的递归求解。

**📊 数据集**

论文未使用实证数据集，全部工作基于理论分析与算法证明。

**📈 对比分析**

由于未进行实验评估，无法与其他方法做性能比较；理论上所给算法在三类结构参数下均为 FPT，时间复杂度为 3^{dc}·n^{O(1)}、2^{mw}·n^{O(1)}、4^{cw}·n^{O(1)}。

**⚠️ 局限性**

局限性包括：算法仅在参数化意义下可行，且在实际大规模实例中可能仍受参数值过大或树宽高等限制；缺乏实验验证和对近似/启发式方案的探讨。

---

## 193. BURMESE-SAN: Burmese NLP Benchmark for Evaluating Large Language Models

**arXiv ID:** 2602.18788 | [PDF](https://arxiv.org/pdf/2602.18788v1)

**作者:** Thura Aung `[一作]`, Peerat Limkonchotiwat `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 BURMESE-SAN，这是首个面向缅甸语的综合性 LLM 评测基准，涵盖 7 个 NLP 任务（情感分析、毒性检测、问答、因果推理、自然语言推理、摘要、机器翻译），所有数据均由母语者精心构造与验证，保证语言真实性与文化适用性。

**💡 创新点**

创新点包括：①首次为低资源缅甸语创建完整的 NLU、NLR、NLG 任务集合；②采用人类中心化高质量数据集、统一正式 Burmese 提示模板；③系统评估商业与开源模型、模型规模、SEA 细调、量化以及代际进步，为模型选择提供实证依据。

**🔧 技术方法**

技术手段包括：使用 SEA-HELM 风格的 Burmese 提示模板；采用多种评估指标（准确率、MetricX-24、ROUGE‑L F1），并将得分标准化到 0–100；对不同规模、架构的 LLM 进行零样本推理实验；探索 NVFP4、DynFP8 量化方法的效果。

**📊 数据集**

使用的数据集来源于公开数据集（Belebele、GKLMIP-mya、Balanced COPA、XL-Sum、FLORES+）并通过翻译、归一化、标签验证等四步流程改造，最终构成 3,920 条样本的 BURMESE‑SAN 数据集。

**📈 对比分析**

通过将商业与开源模型、不同参数规模、SEA‑fine‑tuned 变体及量化模型在同一基准上进行对比实验，结果表明商业模型显著优于开源模型；模型规模提升带来收益但边际递减；SEA 细调对 Llama 系列收益显著；DynFP8 量化几乎保持性能，NVFP4 在推理任务上略有下降；整体来看，缅甸语能力在新一代模型中快速提升。

**⚠️ 局限性**

局限性：评测仅使用正式书面 Burmese 提示模板，未覆盖非正式或口语化用法；仅关注标准缅甸语，未包含 Rakhine、Tavoyan 等方言；数据集主要基于写作文本，未考虑多语种混合或语音输入。

---

## 194. Taming Scope Extrusion in Gradual Imperative Metaprogramming

**arXiv ID:** 2602.19951 | [PDF](https://arxiv.org/pdf/2602.19951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 195. ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization

**arXiv ID:** 2602.19575 | [PDF](https://arxiv.org/pdf/2602.19575v1)

**作者:** Minseo Kim `[一作]` (Korea Advanced Institute of Science and Technology), Junmo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 9455 | [OpenAlex ID](https://openalex.org/A5100606266)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ConceptPrism 框架，通过在一组参考图像中比较，自动将共享视觉概念与图像特定残差信息分离，从而解决个性化文本到图像生成中的概念纠缠问题。

**💡 创新点**

创新点在于：①引入无指导的排除损失（exclusion loss）迫使残差 token 摒弃共享概念；②使用目标 token 与残差 token 的双重优化实现概念的自动 disentanglement；③不需要类名或分割掩码等手动提示，适用于抽象风格和对象等多种概念。

**🔧 技术方法**

主要技术包括：文本到图像扩散模型（Stable Diffusion）、CLIP 编码器用于文本/图像特征、LoRA 参数高效微调、目标/残差 token 的联合优化（重建 loss + 排除 loss）以及基于 KL 的无信息化约束。

**📊 数据集**

使用 DreamBench 数据集（30 个个性化主题，4–6 张参考图像 + 25 条提示）以及其他公开数据集进行定性评估。

**📈 对比分析**

与 DreamBooth、Custom Diffusion、ELITE、DisenBooth、DisEnvisioner 等基线比较，在 CLIP‑T（文本对齐）和 DINO/CLIP‑I（概念保真）指标上均表现出更优的权衡，训练步骤更少（仅 120 步即可达到最优），并在概念保真度上提升显著，同时保持较高的文本对齐。

**⚠️ 局限性**

局限性包括：对极少量参考图像（如单张）效果不明；排除损失对残差 token 维度的依赖性仍需进一步探索；以及在极为复杂或高度抽象的概念中，目标 token 可能仍难以完全捕捉所有细节。

---

## 196. When Do LLM Preferences Predict Downstream Behavior?

**arXiv ID:** 2602.18971 | [PDF](https://arxiv.org/pdf/2602.18971v1)

**作者:** Katarina Slama `[一作]` (UK AI Security Institute), Lennart Luettgau `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

测量了5个前沿LLM在72个实体上的偏好，并检验这些偏好能否在无显式指令的情况下预测捐赠建议、拒绝行为和任务表现。

**💡 创新点**

系统性地将偏好测量与多种行为结果结合，验证偏好是否能驱动模型行为，并探讨不同任务类型（捐赠、拒绝、阅读理解、代理任务）中的偏好效应差异。

**🔧 技术方法**

使用Elo评分、Spearman相关、线性和逻辑回归分析，并通过Inspect框架与自动重试机制收集行为数据。

**📊 数据集**

采用72个实体集合进行偏好评估，捐赠对话的72对、36个实体的捐赠分配；BoolQ阅读理解（训练/验证集）；GAIA与Cybench代理任务。

**📈 对比分析**

通过偏好排名与行为结果的相关性评估；对BoolQ准确率进行偏好相关性检验；对GAIA、Cybench的首尾5名偏好实体进行准确率比较。结果显示捐赠建议与偏好高度相关，拒绝行为也呈偏好相关，但任务表现仅在部分模型上出现微弱偏好效应，代理任务无显著影响。

**⚠️ 局限性**

效应幅度小，代理任务样本量不足，无法断定因果关系，实验环境受控可能不代表真实对话；仅测试单一实体类型；模型受帮助性训练抑制；未探究更复杂任务和更大规模模型的偏好效应。

---

## 197. Replication Study: Federated Text-Driven Prompt Generation for Vision-Language Models

**arXiv ID:** 2602.18439 | [PDF](https://arxiv.org/pdf/2602.18439v1)

**作者:** Suraj Prasad `[一作]`, Anubha Pant `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 FedTPG 进行复制实验，验证其在联邦学习环境下对未见类别的推广性能，使用预训练模型在六个公开数据集上进行评估。

**💡 创新点**

提出通过文本驱动的 Prompt 生成网络（PromptTranslator）动态生成与类别名称相关的上下文向量，从而显著提升对新类别的泛化能力，并保持参数高效且无需共享原始数据。

**🔧 技术方法**

使用 CLIP 的冻结图像编码器 ViT-B/16 与文本编码器，配合可学习的 PromptTranslator（交叉注意力、GEGLU 等），并采用 FedAvg 进行联邦训练；评估时采用余弦相似度与交叉熵。

**📊 数据集**

Caltech101、Oxford Flowers、FGVC Aircraft、Oxford Pets、Food-101 与 DTD 共六个多样化视觉分类数据集。

**📈 对比分析**

将预训练 FedTPG 模型的基类（Seen）与新类（Unseen）准确率进行对比，平均基类准确率 74.58%，新类 76.00%，与原论文差异 ≤0.2%，表明复现成功且新类泛化提升约 +1.43%。

**⚠️ 局限性**

复制仅覆盖六个数据集（缺少 UCF101、Stanford Cars、SUN397），且未从零开始重新训练 FedTPG，仅进行评估；未对不同随机种子与客户端异质性进行多次验证。

---

## 198. Large Language Model-Assisted UAV Operations and Communications: A Multifaceted Survey and Tutorial

**arXiv ID:** 2602.19534 | [PDF](https://arxiv.org/pdf/2602.19534v1)

**作者:** Yousef Emami `[一作]`, Zhu Han `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并整合了大语言模型（LLM）在无人机（UAV）操作与通信中的应用，提出统一框架与系统化分类。

**💡 创新点**

创新点在于系统性地将LLM适配技术与UAV任务结合，阐明LLM在决策、规划、协同与多模态感知中的角色，并提供伦理与未来研究路线图。

**🔧 技术方法**

采用LLM适配方法（预训练、微调、检索增强生成、提示工程）、链式推理、上下文学习以及多模态LLM，结合传统优化与强化学习。

**📊 数据集**

以对超过200篇相关文献和实例系统的综合分析为数据来源，未引入专门实验数据集。

**📈 对比分析**

通过对比不同LLM适配与传统方法在UAV导航、资源分配、集群控制等任务中的性能，展示LLM在决策准确率（最高92%）与任务成功率（94%）上的提升。

**⚠️ 局限性**

局限性包括模型规模与计算资源限制、实时性与能耗挑战、数据偏差与安全攻击风险，以及缺乏统一评测基准和大规模实测验证。

---

## 199. Agentic AI as a Cybersecurity Attack Surface: Threats, Exploits, and Defenses in Runtime Supply Chains

**arXiv ID:** 2602.19555 | [PDF](https://arxiv.org/pdf/2602.19555v1)

**作者:** Xiaochong Jiang `[一作]` (Independent Researcher), Cheng Ji `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化梳理并分类 Agentic LLM 在运行时的供应链攻击，提出以零信任为核心的 Runtime 体系结构，并指出病毒式代理循环（Viral Agent Loop）作为自我传播的新威胁模型。

**💡 创新点**

1) 将数据与工具供应链统一框架化，明确三阶段攻击（Discovery、Implementation、Invocation）与内存持续性攻击；2) 引入 Man‑in‑the‑Environment（MitE）威胁模型与 Viral Agent Loop 概念，揭示循环依赖导致的自传播风险；3) 提出零信任运行时架构（确定性能力绑定、神经符号信息流控制、审核‑工作者（Auditor‑Worker）语义防火墙）以构建全链路防御。

**🔧 技术方法**

基于 LLM 的推理与工具调用、加密证书与 SBOM、语义相似度检索、统计过滤与加密溯源、强化学习/监督式判别模型、模型上下文协议（MCP）等技术。

**📊 数据集**

无专门实验数据集，主要依托已有的安全研究文献与公开案例进行综述与理论分析。

**📈 对比分析**

通过对比已发表的安全防御方法（如 Instruction Hierarchy、Audit‑Worker、SLSA 等），从攻击阶段、目标组件与防御原理四维度进行系统评估；由于为系统化研究，未给出量化性能指标，评估以安全覆盖率与缺陷完整性为主。

**⚠️ 局限性**

缺乏针对真实 Agentic LLM 的实测验证；对零信任运行时模型的实现与部署成本未给出细化方案；缺少针对新兴工具与数据源的动态更新策略，待后续基准测试与实验验证。

---

## 200. Strategic Gaussian Signaling under Linear Sensitivity Mismatch

**arXiv ID:** 2602.19292 | [PDF](https://arxiv.org/pdf/2602.19292v1)

**作者:** Hassan Mohamad `[一作]` (University of Lorraine), Samson Lasaulce `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了在目标误差为线性敏感性失配的 Gaussian 策略游戏中，利用 Stackelberg 结构分析了信息传递的最优策略；

**💡 创新点**

创新点在于将传统的加法偏差模型推广为一般的线性敏感性失配，并通过谱分解揭示了信息泄露与失配矩阵特征值之间的关系，进一步在噪声通道下给出了信息阈值和相位转移的解析条件；

**🔧 技术方法**

主要技术包括：线性高斯信号的条件期望推导、谱分解与投影优化、半正定规划（SDP）与投影解构、信息理论下的互信息与容量下界、以及对称水位分配的能力评估；

**📊 数据集**

实验数据使用了合成的零均值高斯源与高斯噪声，参数化为协方差矩阵 Σ_m 与 Σ_w，未使用公开数据集；

**📈 对比分析**

由于研究为理论性，比较主要通过数值仿真验证阈值与相位转移：在阈值之下系统处于无信息（沉默）状态；超过阈值后通信强度随敏感性下降而递增，显示出明显的相位转移现象；

**⚠️ 局限性**

局限性包括：只考虑线性策略与线性敏感性失配；未讨论无承诺的 Nash 均衡；对非对角噪声协方差的分析仅限于等方差或易近条件；动态控制与时变失配仍待进一步研究。

---

## 201. Metasurfaces-Integrated Wireless Neural Networks for Lightweight Over-The-Air Edge Inference

**arXiv ID:** 2602.19312 | [PDF](https://arxiv.org/pdf/2602.19312v1)

**作者:** Kyriakos Stylianopoulos `[一作]` (National and Kapodistrian University of Athens), George C. Alexandropoulos `[通讯]` (National and Kapodistrian University of Athens)

**通讯引用:** 15659 | [OpenAlex ID](https://openalex.org/A5056331037)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实验了基于多层可编程超材料（SIM）的元件，构成的端到端 MIMO 物理层神经网络（MINN），实现无线通道中的算子执行；

**💡 创新点**

将无线通道本身视为可训练的深度学习层，利用 OTA 计算将数字 DNN 的计算负荷下沉到波传播域，从而显著降低能耗与硬件复杂度；

**🔧 技术方法**

利用多天线 MIMO、可编程超材料（SIM）、深度衍射神经网络、极大规模（XL）天线阵列、线性/非线性激活元件、梯度下降等技术进行联合训练与优化；

**📊 数据集**

在 MNIST、Parkinson's、Wisconsin Breast Cancer Diagnosis、SECOM 等公共数据集上进行分类实验；

**📈 对比分析**

与纯数字 DNN（不经过无线传输）和仅使用数字模块的基线对比，结果显示在适当的 SIM 元素数和 SNR 条件下，MINN 的分类准确率可逼近或与数字 DNN 相当，同时功耗显著下降；

**⚠️ 局限性**

面临硬件量化误差、非线性激活实现困难、动态响应训练复杂、对极大规模天线与宽带信号支持不足、理论通用逼近证明不完整等限制。

---

## 202. Routing-Aware Explanations for Mixture of Experts Graph Models in Malware Detection

**arXiv ID:** 2602.19025 | [PDF](https://arxiv.org/pdf/2602.19025v1)

**作者:** Hossein Shokouhinejad `[一作]` (University of New Brunswick), Ali. A Ghorbani `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种基于控制流图的混合专家图神经网络，并提出路由感知的后置可解释方法，用于恶意软件检测。

**💡 创新点**

创新点包括：① 在节点层引入多统计编码与度数加权以实现节点表示多样化；② 在读出层设置六个专家并通过 Top‑k 路由与负载平衡实现专家选择；③ 将路由权重融入解释，生成与模型决策一致的边级解释。

**🔧 技术方法**

采用技术包括：图神经网络（GCN、GIN、GAT 等）+ 混合专家框架 + Top‑k 路由 + 负载平衡 + 自动编码器压缩节点特征 + 事后梯度/掩码解释（Integrated Gradients）等。

**📊 数据集**

使用的 CFG 数据集为 BODMAS、PMML（恶意样本）和 DikeDataset（良性样本）。

**📈 对比分析**

通过与单一专家 GNN 基线（GCN、GIN、GAT）以及其他 MoE 变体比较，采用准确率、F1 等指标评估。Top‑2 专家+负载平衡模型在准确率达到 93.59%，显著优于基线。

**⚠️ 局限性**

局限性：解释基于后置方法，缺乏内在可解释性；实验仅覆盖 CFG，未验证多模态或其他安全任务；路由学习需要人工调参（k、LB 系数）。

---

## 203. Exponential Convergence of (Stochastic) Gradient Descent for Separable Logistic Regression

**arXiv ID:** 2602.18946 | [PDF](https://arxiv.org/pdf/2602.18946v1)

**作者:** Sacchit Kale `[一作]` (Indian Institute of Science), Anant Raj `[通讯]` (Indian Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了简洁的非自适应增大步长调度和自适应 SGD 调度，证明在线性可分逻辑回归下可实现指数级收敛并保持稳定；

**💡 创新点**

证明了加速不必依赖不稳定（边缘稳定）阶段，纯粹的步长增长即可；同时提出了无需线搜索的 SGD 与块式自适应 SGD；

**🔧 技术方法**

利用逻辑回归的自界曲率、单调性证明、漂移分析与倍增技巧等理论工具；

**📊 数据集**

在合成高维可分数据和 MNIST 二分类子集上进行实验；

**📈 对比分析**

与常数步长 GD/SGD 对比，实验显示损失在对数尺度下呈指数/线性下降，速度显著提升；

**⚠️ 局限性**

仅适用于可分逻辑回归，依赖边距假设，尚未推广至一般损失或非可分情形；

---

## 204. Carbon-aware decentralized dynamic task offloading in MIMO-MEC networks via multi-agent reinforcement learning

**arXiv ID:** 2602.18797 | [PDF](https://arxiv.org/pdf/2602.18797v1)

**作者:** Mubshra Zulfiqar `[一作]` (Wuhan University of Technology), Basit Qureshi `[通讯]` (Prince Sultan University)

**通讯引用:** 2434 | [OpenAlex ID](https://openalex.org/A5083814185)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于多智能体近端策略优化（MAPPO）的分散式任务卸载框架 CADDTO-PPO，旨在降低 MEC 生态系统的碳排放与任务延迟。

**💡 创新点**

创新点在于将碳优先奖励与 MIMO 干扰管理、能量收集耦合在一起，并通过参数共享实现可扩展的 O(1) 推理复杂度。

**🔧 技术方法**

主要技术包括：多智能体 PPO 与参数共享（CTDE 结构）、基于状态观测的本地决策、碳优先奖励设计、通用优势估计（GAE）和离线硬件性能分析。

**📊 数据集**

实验采用基于 Poisson 过程生成的任务到达与能量收集数据，设置多用户 MIMO‑MEC 仿真环境，未使用公开真实数据集。

**📈 对比分析**

与集中式 PPO、DDPG、Lyapunov‑DPP、贪婪、局部/仅卸载等基线对比，CADDTO‑PPO 在高负载下实现近零包溢出、相当于贪婪的吞吐率，并将碳强度降低 15‑20%，同时保持 0.145 ms 的推理时延。

**⚠️ 局限性**

主要局限在于单小区、静态网格碳强度、无多小区切换与移动、未考虑语义通信等未来扩展方向。

---

## 205. Universal Basic Income with Time-Decaying Currency: Structural Effects on Essential Labor and Long-Term Formation

**arXiv ID:** 2602.18714 | [PDF](https://arxiv.org/pdf/2602.18714v1)

**作者:** Hitoshi Yamada `[一作]` (Toyota Motor Corporation), Hitoshi Yamada `[通讯]` (Toyota Motor Corporation)

**通讯引用:** 5515 | [OpenAlex ID](https://openalex.org/A5101414077)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

**🎯 论文内容**

研究了双币制UBI系统中时间衰减货币的分配与使用，利用代理模型仿真分析其对劳动力参与、必需劳动力供给以及长期人力资本形成的影响。

**💡 创新点**

将时间衰减货币与UBI结合，提出接纳比例作为关键设计参数，揭示其对短期消费稳定与长期参与动力学的分离效应，区分劳动力崩溃与参与延迟的结构性矛盾。

**🔧 技术方法**

基于离散时间的代理式效用最大化模型，使用程序化决策与必要品支付约束，绘制相位图和时间平均指标。

**📊 数据集**

无实测数据，采用人工生成的代理人异质性参数与不同福利水平、接纳比例、折旧率等组合进行参数空间探索。

**📈 对比分析**

通过比较不同接纳比例与UBI规模下的必需劳动力供给、瞬时非工作峰值以及时间平均非工作份额，发现存在临界阈值导致劳动力参与延迟，但必需劳动力供给基本不崩溃，说明短期稳定与长期激励之间存在折衷。

**⚠️ 局限性**

模型假设简化，忽略心理动机、社会规范和非经济效益，仅关注机构设计导致的经济动力学；未进行实证验证，结果高度依赖模型设定和参数选取。

---

## 206. Unsupervised Anomaly Detection in NSL-KDD Using $β$-VAE: A Latent Space and Reconstruction Error Approach

**arXiv ID:** 2602.19785 | [PDF](https://arxiv.org/pdf/2602.19785v1)

**作者:** Dylan Baptiste `[一作]` (Université de Reims Champagne-Ardenne), François Foyer `[通讯]` (Seckiot)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在NSL-KDD数据集上使用β-VAE进行无监督异常检测，比较了基于重建误差和基于潜在空间距离的两种判别方法。

**💡 创新点**

创新点在于展示潜在空间的平均欧氏距离（尤其在k取大值时）可与重建误差相媲美甚至超越，并指出两种信号互补，具备增量学习与行为分类的潜力。

**🔧 技术方法**

使用技术包括β-Variational Autoencoder、重建误差与KL散度联合损失、潜在空间的k近邻欧氏距离，以及在不同β与k组合下的实验评估。

**📊 数据集**

所用数据集为NSL-KDD，包含正常与多类攻击流量，实验采用仅正样本的无监督训练方式。

**📈 对比分析**

通过AUROC比较，两种方法均能达到超过96%的检测性能；在β=1e-5、k=5000时，潜在空间距离达97.9%AUROC，略优于重建误差（96.8–97.7%）。

**⚠️ 局限性**

局限性包括对β与k参数的敏感性、计算成本随k增大而显著提升、欧氏距离未考虑协方差结构，以及实验仅在单一数据集与二分类任务上验证。

---

## 207. Traffic-Aware Configuration of OPC UA PubSub in Industrial Automation Networks

**arXiv ID:** 2602.19603 | [PDF](https://arxiv.org/pdf/2602.19603v1)

**作者:** Kasra Ekrad `[一作]` (Mälardalen University), Mohammad Ashjaei `[通讯]` (Mälardalen University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出了一套基于工业通信时序和质量服务（QoS）规范，将工业自动化交通类型映射到OPC UA PubSub配置的指导原则，并通过工业用例进行验证，展示错误配置对延迟和吞吐量的影响；

**💡 创新点**

创新点在于首次系统化地给出不同工业交通类型对应的PubSub参数配置（如DSM类型、KeyFrameCount、Delta Frames、KeepAlive、DatasetOrdering等），填补了OPC UA规范与实际网络需求之间的空白；

**🔧 技术方法**

使用技术包括OPC UA PubSub（Part 14）与TSN结合的UDP UADP传输、关键帧与增量帧的差异化配置、事件与周期性数据的区分，以及基于TSN的时间敏感网络架构；

**📊 数据集**

采用了5G-ACIA提供的13种工业交通类型规范作为输入数据集，且通过构造的工业生产线用例（机器人臂、输送带、边缘服务器等）进行场景化评估；

**📈 对比分析**

评估方法主要是定性对比：在用例中演示不同配置（如错误使用Delta Frame、错误KeyFrameCount、错误DatasetOrdering等）导致的延迟增大、丢包率上升和吞吐量下降；通过对比正确配置与误配置的网络行为，证明了交通感知配置能够提升可预测性和实时性；

**⚠️ 局限性**

局限性包括：未对订阅端配置进行评估，仅关注发布端；缺乏定量性能指标（如精确延迟、Jitter、吞吐量数值）；实现细节与open62541栈的差异未完全覆盖；未来工作需在真实工业环境中进行实验验证并探究TSN调度与PubSub参数交互的进一步优化。

---

## 208. VLANeXt: Recipes for Building Strong VLA Models

**arXiv ID:** 2602.18532 | [PDF](https://arxiv.org/pdf/2602.18532v1)

**作者:** Xiao-Ming Wu `[一作]` (Nanyang Technological University), Chen Change Loy `[通讯]` (Nanyang Technological University)

**通讯引用:** 79289 | [OpenAlex ID](https://openalex.org/A5005626854)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统评估并优化Vision–Language–Action（VLA）模型的设计空间，提出一套实用的构建配方并实现VLANeXt模型。

**💡 创新点**

创新点在于统一框架下对VLM–策略模块连接方式、感知输入融合、动作建模等三维设计进行大规模消融，发现软连接、VLM侧姿态感知、动作分块和频域辅助损失能显著提升性能。

**🔧 技术方法**

技术手段包括使用多模态大型语言模型（如Qwen3‑VL‑2B）作为视觉语言后端、构建深层专用策略模块、动作分块预测与流匹配学习、频域损失辅助、以及多视角视觉与姿态输入的融合。

**📊 数据集**

主要数据集为LIBERO（四套任务）及其鲁棒性扩展版LIBERO‑plus，实际实验中还使用DROID预训练以及Frank‑Emika和Aloha双臂实机数据。

**📈 对比分析**

与目前主流VLA与直接策略学习方法对比，VLANeXt在LIBERO和LIBERO‑plus平均成功率分别提升至97.4%和80.1%，在实机单臂与双臂任务中也取得显著优势。

**⚠️ 局限性**

局限性包括：对更长时序任务和复杂多模态世界建模的可扩展性仍需验证，且频域损失在多峰分布动作空间下的效果尚未充分探究。

---

## 209. Misquoted No More: Securely Extracting F* Programs with IO

**arXiv ID:** 2602.19973 | [PDF](https://arxiv.org/pdf/2602.19973v1)

**作者:** Cezar-Constantin Andrici `[一作]` (Max Planck Institute for Software Systems), Théo Winterhalter `[通讯]` (Inria)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一种用于将使用单子实现 IO 的浅层嵌入程序安全提取到深层嵌入 λ‑计算机的框架，并给出了完整的机理证明；

**💡 创新点**

创新点在于引入“关系引用”(relational quotation)——一种只在第一次提取时使用翻译验证、随后由一次性验证的语法生成函数完成剩余工作的方法；该方法极大降低了可信根基，并通过双向逻辑关系证明了 Robust Relational Hyperproperty Preservation (RrHP) 的安全编译属性；

**🔧 技术方法**

核心技术包括：基于依赖类型的类型推导关系、对 IO 单子采用 Fine‑Grained Call‑By‑Value 语义、两种跨语言的跟踪产生语义与逻辑关系、以及在 F* 中实现的可验证语法生成器；

**📊 数据集**

论文未使用任何外部数据集，所有验证均在 F*/Peregrine 工具链和 GitHub artifact 中完成；

**📈 对比分析**

由于本文关注的是理论安全性与可验证性，未给出传统意义上的性能基准；证明与实验主要通过 F* 的定理证明与单元测试完成；

**⚠️ 局限性**

局限性包括：只支持有限的 IO 单子（无递归、无效能多态等）、未实现完全可验证的最终 OCaml 编译步骤、以及对更复杂特性的（如依赖式子、子类型、可变状态）扩展仍待研究。

---

## 210. PrivacyBench: Privacy Isn't Free in Hybrid Privacy-Preserving Vision Systems

**arXiv ID:** 2602.18900 | [PDF](https://arxiv.org/pdf/2602.18900v1)

**作者:** Nnaemeka Obiefuna `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Steven Kolawole `[通讯]` (Carnegie Mellon University)

**通讯引用:** 242 | [OpenAlex ID](https://openalex.org/A5041114293)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个系统化的隐私保护机器学习基准框架，用于检测多种隐私技术（FL、DP、SMPC）组合在计算机视觉任务中的交互、性能与能耗。

**💡 创新点**

首次提供了可复现的 YAML 配置、资源监控（训练时间、内存、能耗）与混合隐私技术交互分析的平台，揭示了 FL+DP 组合的致命收敛失败和 FL+SMPC 的低成本高效能模式。

**🔧 技术方法**

使用 Flower 实现联邦学习、Opacus 注入差分隐私、Shamir 共享实现 SMPC、CodeCarbon 监测能耗，同时对 ResNet18 与 ViT 进行实验。

**📊 数据集**

采用阿尔茨海默病 MRI 分类数据集和 ISIC 皮肤病变多分类数据集，均采用非 IID 的 Dirichlet 分区模拟真实医学场景。

**📈 对比分析**

通过对比基线、FL、FL+SMPC 与 FL+DP 四种配置，结果显示 FL 与 FL+SMPC 维持接近基线的准确率（98%~99%）并仅增加 10–20% 的计算/能耗，而 FL+DP 在所有策略下均出现严重收敛失效（准确率降至 7–18%），并导致 9–24 倍的计算成本和能耗激增。

**⚠️ 局限性**

局限于仅 3 端联邦、仅评估 ResNet18 与 ViT 两种架构，未验证更大规模参与者或不同模型族；实验聚焦医学影像，泛化到其他领域尚未证明；能源测量受硬件与软件环境约束，需进一步跨平台验证。

---

## 211. A Two-Stage Detection-Tracking Framework for Stable Apple Quality Inspection in Dense Conveyor-Belt Environments

**arXiv ID:** 2602.19278 | [PDF](https://arxiv.org/pdf/2602.19278v1)

**作者:** Keonvin Park `[一作]` (Seoul National University), Jin Hong Mok `[通讯]` (Dongguk University)

**通讯引用:** 460 | [OpenAlex ID](https://openalex.org/A5086445968)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了一种两阶段检测‑跟踪‑分类框架，利用YOLOv8定位苹果，ByteTrack保持对象身份，并在轨迹级进行缺陷聚合，以实现工业输送带视频中的稳定质量检测。

**💡 创新点**

创新点在于将农场训练的YOLOv8检测器与ByteTrack多目标跟踪结合，并首次引入轨迹级聚合消除帧间预测振荡，同时提出了视频级缺陷比和时间一致性等工业评估指标。

**🔧 技术方法**

采用YOLOv8进行目标检测，ByteTrack实现多目标跟踪，ResNet18负责缺陷分类，轨迹级多数投票聚合保证判定稳定，整体实现基于PyTorch。

**📊 数据集**

训练使用农场苹果检测数据集和Healthy‑Defective苹果缺陷图像数据集；评估使用公开YouTube输送带视频。

**📈 对比分析**

通过mAP、分类准确率、召回率等传统指标评估检测与分类；引入轨迹级缺陷比和时间一致性指标，实验显示加入跟踪与聚合后时间一致性显著提升，缺陷比估计更可靠。

**⚠️ 局限性**

局限性包括缺少帧级缺陷标注，无法在视频上做精确评估；域漂移导致检测精度下降；模型在真实工厂环境中的鲁棒性尚待验证。

---

## 212. FineRef: Fine-Grained Error Reflection and Correction for Long-Form Generation with Citations

**arXiv ID:** 2602.18437 | [PDF](https://arxiv.org/pdf/2602.18437v1)

**作者:** Yixing Peng `[一作]` (University of Science and Technology of China), Quan Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 9097 | [OpenAlex ID](https://openalex.org/A5108047863)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

FineRef通过对生成式模型的尝试–反思–纠正链进行监督学习和强化学习，显著提升长文本引用生成的准确性和答案质量。

**💡 创新点**

创新点在于细粒度、可控的自我反思机制，区分并纠正引用中的不匹配和无关错误，并在过程级强化学习中设计多维奖励。

**🔧 技术方法**

采用监督学习结合外部事实一致性模型（FCM）和重排序器生成反射标签，在线自反思自举、过程级强化学习（REINFORCE+LOO）与多维奖励。

**📊 数据集**

使用ALCE基准（包含ASQA和ELI5）作为评测数据集。

**📈 对比分析**

与GPT‑4、ChatGPT、Self‑RAG、CaLF等基线对比，FineRef在Citation F1上提升至GPT‑4的+18%，EM Recall提升4%，并在领域迁移与噪声检索下保持领先。

**⚠️ 局限性**

局限性：仅针对匹配与无关两类错误；依赖外部FCM和重排序器生成反射标签；在多语言或更大规模场景的适用性尚待验证。

---

## 213. Taming Preconditioner Drift: Unlocking the Potential of Second-Order Optimizers for Federated Learning on Non-IID Data

**arXiv ID:** 2602.19271 | [PDF](https://arxiv.org/pdf/2602.19271v1)

**作者:** Junkang Liu `[一作]` (Tianjin University), Yuanyuan Liu `[通讯]` (Xidian University)

**通讯引用:** 18918 | [OpenAlex ID](https://openalex.org/A5100405062)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在非IID联邦学习中第二阶优化器的稳定性问题，并提出FedPAC框架以对齐和纠正预条件器漂移。

**💡 创新点**

创新点在于将预条件器漂移作为主要瓶颈提出，并通过全局预条件器聚合（对齐）和全局方向纠正（校正）两步实现对齐与纠正，且提供非凸收敛证明。

**🔧 技术方法**

使用了第二阶优化器（SOAP、Sophia、Muon）的预条件器框架，配合FedPAC的对齐与纠正技术。

**📊 数据集**

在CIFAR-100、Tiny-ImageNet、ViT-B/Tiny、ResNet-18以及C4数据集上的LLaMA模型进行实验。

**📈 对比分析**

与FedAvg、SCAFFOLD、FedCM、Local AdamW、Local Sophia、Local Muon、Local SOAP等基线对比，FedPAC在非IID场景下提升了5–6%准确率，收敛更快且更稳定。

**⚠️ 局限性**

主要限制是需要额外传输预条件器导致通信负担，且在近IID环境下收益有限；第二阶本身计算开销略大。

---

## 214. Open-vocabulary 3D scene perception in industrial environments

**arXiv ID:** 2602.19823 | [PDF](https://arxiv.org/pdf/2602.19823v1)

**作者:** Keno Moenck `[一作]` (Hamburg University of Technology), Thorsten Schüppstuhl `[通讯]` (Hamburg University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无需训练的开词汇3D感知流程，通过基于超点的合并和工业CLIP实现工业环境的实例和语义分割。

**💡 创新点**

创新点在于不使用预训练的类无关实例分割模型，而是采用超点生成与特征驱动的合并来生成mask，并验证工业CLIP在工业场景下的有效性。

**🔧 技术方法**

使用了BPSS进行超点生成、SAM进行2D掩码生成、CLIP/工业CLIP提取特征、余弦相似度进行超点合并，以及HDBSCAN进行实例聚类。

**📊 数据集**

实验基于Leica BLK 360 Terrestrial Laser Scanner获取的工业车间点云及其对应的多视角2D图像，未使用公开工业数据集。

**📈 对比分析**

通过与传统Mask3D+OpenMask3D等基线对比，发现Mask3D在工业对象上表现不佳；本方法在工业场景中成功分割lathe、vise等对象，表现以定性评估展示，表明在工业环境下的可行性。

**⚠️ 局限性**

局限性包括工业CLIP过拟合工业图像，难以区分语义相近对象；缺乏量化评估和多语言广度支持，且方法在复杂场景下的性能仍待进一步验证。

---

## 215. Cost-Aware Diffusion Active Search

**arXiv ID:** 2602.19538 | [PDF](https://arxiv.org/pdf/2602.19538v1)

**作者:** Arundhati Banerjee `[一作]` (Carnegie Mellon University), Jeff Schneider `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8531 | [OpenAlex ID](https://openalex.org/A5055199976)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于扩散模型的成本感知主动搜索算法（CD‑AS），利用梯度引导的扩散过程在每一步生成未来动作序列，从而实现长期规划与在线决策。

**💡 创新点**

创新点在于：①发现扩散模型在主动搜索中的乐观偏差并通过仅生成动作序列而非状态-动作对来缓解；②将回报估计网络和距离估计网络结合进扩散采样，实现在不构造搜索树的情况下实现非贪心的多步规划；③在分布式多代理环境下实现异步解耦的成本感知决策。

**🔧 技术方法**

核心技术包括：扩散生成模型（U‑Net/图注意力网络）、梯度引导采样、Kalman滤波后的贝叶斯置信更新、线性传感模型与成本模型、离散网格化的搜索空间、离散时间的动作序列表示。

**📊 数据集**

使用在仿真环境中生成的合成数据：1D 16格、2D 8×8 网格，单/多代理（J=1、3），目标数量 k=1 或 4，噪声水平 σ=1/16 或 0.2。训练数据由信息贪婪（EIG）代理在仿真中采集。

**📈 对比分析**

与基准（EIG、TS、IQL、CAST、扩散策略）对比。CD‑AS 在低噪声或感知成本较高的场景下实现更快的全目标恢复、相对更低的总成本；在感知成本低时仍略逊于 CAST。计算上，CD‑AS 每步耗时比 CAST 低 30% 以上，显著提升实时性。

**⚠️ 局限性**

局限性包括：①距离估计网络在组合动作空间上的泛化不足，导致在低感知成本场景下成本感知效果不佳；②训练数据来自子最优贪婪策略，缺乏真正的最优序列；③多代理协同机制有限，仅通过异步信息共享，未实现全局最优；④在复杂大规模网格下扩散模型的训练与采样仍较耗时。

---

## 216. Counted NFT Transfers

**arXiv ID:** 2602.19199 | [PDF](https://arxiv.org/pdf/2602.19199v1)

**作者:** Qin Wang `[一作]` (CSIRO Data61), Shiping Chen `[通讯]` (CSIRO Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出ERC‑7634标准，在ERC‑721基础上实现按转移计数的有限可转让 NFT，填补了完全可转让与完全不可转让之间的空白。

**💡 创新点**

创新点在于将转移次数纳入可转让性的可编程度，既保留了ERC‑721的流动性，又通过可计数的转移限制引入“流动性溢价”、抵制洗盘、限制递归抵押等经济机制；实现方式极简，仅需60行 Solidity 并保持向后兼容。

**🔧 技术方法**

采用 Solidity 实现 ERC‑7634 通过 ERC‑721 生命周期钩子实现计数与限制；使用理论经济模型（流动性溢价、边际成本）、数值模拟、Gas 计量与安全分析工具。

**📊 数据集**

使用 5,0000 条基于公开数据校准的合成转移记录，涵盖 PFP、艺术、游戏、会员、元宇宙等 5 类 NFT，模拟不同转移阈值（L=5/10/15/20/50）。

**📈 对比分析**

通过与 ERC‑721 基线对比测量 Gas 费用（约 10–11% 增加）、对不同 L 下的 token 受限比例（L=10 时 <15% 受限）、洗盘收益下降、递归杠杆最大值下降等指标，表明该机制在保持兼容性的同时提供可观的经济约束。

**⚠️ 局限性**

局限性包括：包装（wrapper）可绕过计数的成本阈值、仿真数据缺乏真实链上验证、跨链转移不计数、限制可变更导致安全隐患、以及对特殊用例（如治理或监管资产）需额外补充实现。

---

## 217. TokenTrace: Multi-Concept Attribution through Watermarked Token Recovery

**arXiv ID:** 2602.19019 | [PDF](https://arxiv.org/pdf/2602.19019v1)

**作者:** Li Zhang `[一作]` (University of California), Vishal Asnani `[通讯]` (Adobe)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5094185916)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TokenTrace，一种主动式水印框架，能够在文本到图像扩散模型生成过程中嵌入多概念（如对象、风格）的水印，并通过查询式 TokenTrace 模块实现对单一或多重概念的精确检索；

**💡 创新点**

创新点在于：①双重条件嵌入——同时扰动文本提示嵌入和初始潜在噪声，将水印与概念语义紧密绑定；②查询式解码模块能够根据用户给出的文本查询仅检索对应概念的水印，实现多概念的可解耦与独立鉴权；

**🔧 技术方法**

核心技术包括 Stable Diffusion 1.5 潜在扩散模型、CLIP frozen encoders、概念编码器与秘密映射器、投影+注意力的 TokenTrace 模块以及多项损失（BCE、CSD、L2、正则化）的联合优化；

**📊 数据集**

实验数据集涵盖：WikiArt（23种艺术风格）、ImageNet（1000个物体类别）、Stable Diffusion Textual Inversion Embeddings（定制概念）以及自建的 ChatGPT 生成的四概念提示数据集；

**📈 对比分析**

与 ALADIN、CLIP、ProMark、CustomMark 等被动/主动基线进行比较；TokenTrace 在单概念任务中取得 98.33%/91.67%（风格）和 95.82%/90.43%（物体）的 attribution accuracy，且在多概念任务中提升至 88–90% 以上；视觉质量接近原始模型，FID 仅略高于原始 LDM，鲁棒性对旋转、压缩等常见变换均保持 80%+ 以上的检索准确率；

**⚠️ 局限性**

主要限制：对抽象或视觉弱显著的概念检索率下降；多定制概念组合时图像质量略受影响；需要为每个概念维护独立的 16-bit secret，管理上相对繁琐；

---

## 218. SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World

**arXiv ID:** 2602.18887 | [PDF](https://arxiv.org/pdf/2602.18887v1)

**作者:** Jungho Kim `[一作]` (Seoul National University), Jun Won Choi `[通讯]` (Seoul National University)

**通讯引用:** 3937 | [OpenAlex ID](https://openalex.org/A5102839991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SafeDrive，一种基于稀疏世界模型和细粒度安全推理的端到端自动驾驶规划框架。

**💡 创新点**

使用轨迹条件稀疏世界模型捕获实例级交互，并通过对每个动态主体的 pair‑wise 碰撞风险和时间维度行驶区域合规性进行细粒度推理，显著提升安全性。

**🔧 技术方法**

结合 BEVFormer‑M、稀疏世界生成器、自注意力与轨迹引导可变形注意力、交叉熵与对数损失等多任务学习技术，实现检测、分割、轨迹预测与安全评分的统一训练。

**📊 数据集**

在开放循环 NAVSIM（基于 OpenScene 的 360° 传感器数据）和闭环 Bench2Drive（CARLA 交互轨迹）上进行评估。

**📈 对比分析**

与多种 SOTA 端到端方法（UniAD、Hydra‑MDP、WoTE、DiffusionDrive 等）对比，PDMS 91.6、EPDMS 87.5 以及 Bench2Drive 驾驶分数 66.8%，均超越前沿水平。

**⚠️ 局限性**

模型仅保留关键主体的稀疏表示可能忽略无关但潜在危险的细节，且在极端稀有场景下的泛化尚待进一步验证。

---

## 219. RobPI: Robust Private Inference against Malicious Client

**arXiv ID:** 2602.19918 | [PDF](https://arxiv.org/pdf/2602.19918v1)

**作者:** Jiaqi Xue `[一作]` (University of Central Florida), Qian Lou `[通讯]` (University of Central Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种针对基于全同态加密的私有推理（PI）协议的推理操纵攻击PI-Attack，并设计了RobPI协议以抵御恶意客户端的攻击；

**💡 创新点**

创新点在于①提出低查询量（3×–8×）的PI-Attack；②在RobPI中引入加密兼容噪声注入与动态噪声训练（DNT），显著提升对恶意客户端的鲁棒性；

**🔧 技术方法**

采用离散余弦变换（DCT）搜索方向、可变步长调度、加密兼容噪声、动态噪声训练等技术；

**📊 数据集**

实验使用MNIST、CIFAR-10和糖尿病视网膜病变数据集；

**📈 对比分析**

与SimBA‑DCT、Square Attack等攻击以及RND、RND‑GF、Adversarial Training等防御比较，RobPI在攻击成功率下降约91.9%、查询量提升10×，同时保持≈74%清洗准确率；

**⚠️ 局限性**

局限性包括：对平均推理攻击仍需进一步强化（如非零均值噪声、改为前层噪声），且在更大模型或更复杂网络结构上的评估仍待验证。

---

## 220. Is Log-Traced Engagement Enough? Extending Reading Analytics With Trait-Level Flow and Reading Strategy Metrics

**arXiv ID:** 2602.19616 | [PDF](https://arxiv.org/pdf/2602.19616v1)

**作者:** Erwin Lopez `[一作]` (Kyushu University), Atsushi Shimada `[通讯]` (Kyushu University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

结合学生自评的流畅性（DEC）和基于电子书交互日志的阅读策略序列指标，对学生最终成绩进行回归预测，探索行为、认知与情感维度的交互。

**💡 创新点**

首次将个体特质流畅性与基于时间序列的阅读策略指标同并入日志分析框架，揭示行为轨迹与学习成效的多维关联，并证明特质能调节日志指标对成绩的影响。

**🔧 技术方法**

使用Pearson相关、线性回归、逐步回归、交互项检验与层次聚类分析，对日志与问卷数据进行特征提取与统计建模。

**📊 数据集**

在九州大学两门工程课程中收集的100名学生（DSP+PT）日志与问卷数据，总计约400,000条事件记录与对应的最终成绩。

**📈 对比分析**

与仅基于日志的参与度指标相比，加入DEC和序列指标后模型R²提升至约0.47（约增加21%），显示显著预测性能提升。

**⚠️ 局限性**

样本规模有限，跨文化适用性未知；模型假设线性关系，未考虑非线性或动态变化；逐步回归可能产生选择偏差。

---

## 221. The LLMbda Calculus: AI Agents, Conversations, and Information Flow

**arXiv ID:** 2602.20064 | [PDF](https://arxiv.org/pdf/2602.20064v1)

**作者:** Zac Garby `[一作]` (University of Nottingham), David Sands `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一种面向 LLM 代理的无类型 λ 计算机 LLMbda，提供对对话、工具调用与信息流控制的形式化语义，并通过大步/小步语义证明了终止无关非干涉性；随后构建了解释器并在其中验证了 Prompt Injection 攻击与 CaMeL 防御方案的正确性。

**💡 创新点**

创新点在于：①首次将对话、fork/clear 以及 LLM 调用整合到 λ 计算机中；②提出了可直接在语言中使用的标签测试与强测试操作；③在此框架下证明了三类子语言满足终止无关非干涉性，弥补了现有工作对标签测试的安全性缺口；④通过实现解释器将形式化模型与实际 LLM 调用紧密结合，展示了理论与实践的可行性。

**🔧 技术方法**

技术包括：无类型调用式 λ 计算机、动态信息流跟踪（标签化表达式与测试）、大步/小步语义、终止无关非干涉性证明、Python 实现的解释器、OpenAI Responses API 与 Lark 语法解析。

**📊 数据集**

实验主要使用 OpenAI API（实际 LLM 生成对话）以及参考 AgentDojo 基准中的 prompt‑injection 任务；论文中并未提供专门的数据集，仅在解释器中重现了 CaMeL 与攻击示例。

**📈 对比分析**

方法对比：利用解释器直接执行 LLMbda 程序，并与手写脚本/传统脚本比较，示例运行时间在 5–10 秒左右，验证了防御方案在实际 LLM 调用中的有效性；然而并未给出系统化的性能基准或对比实验。

**⚠️ 局限性**

局限性包括：①假设 LLM 响应确定性，未考虑概率或非确定性；②工具调用与外部数据源建模过于简化；③缺乏类型系统与更复杂的并发/分布式场景；④未进行大规模基准测试或安全漏洞的系统性评估；⑤对标签测试的安全性仅在特定子语言或两层标签下保证。

---

## 222. GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery

**arXiv ID:** 2602.19872 | [PDF](https://arxiv.org/pdf/2602.19872v1)

**作者:** Jizhou Han `[一作]` (Xi'an Jiaotong University), Yihong Gong `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一框架GOAL，用固定的ETF（等角紧致帧）分类器在持续学习中对已知和未知类别进行特征对齐，实现持续的类别发现与知识保留。

**💡 创新点**

创新点在于：① 预定义固定ETF结构，使整个学习过程保持全局几何一致性，避免动态权重更新导致的遗忘；② 结合置信度引导的对齐策略，自动将高置信度的未标样本分配到未占用的ETF方向，稳健加入新类别；③ 通过监督与无监督ETF对齐双模机制，兼顾已知类别保留与未知类别可分性。

**🔧 技术方法**

技术包括：等角紧致帧（ETF）固定分类器、特征编码器（基于预训练DINO ViT-B/16）、监督与无监督对比学习、置信度（最小熵）筛选、聚类导向的分类器初始化、交叉熵与对齐损失组合训练。

**📊 数据集**

在四个C‑GCD基准数据集上评估：CIFAR‑100、TinyImageNet、ImageNet‑100、CUB‑200。

**📈 对比分析**

与现有GCD与C‑GCD方法（VanillaGCD、SimGCD、SimGCD+、FRoST、GM、MetaGCD、Happy）对比，GOAL在所有数据集上均获得最高准确率，平均相较最强对手Happy降低16.1%遗忘率、提升3.2%新类别发现率，并在10阶段实验中进一步降低19.14%/23.71%遗忘率。

**⚠️ 局限性**

局限性包括：① 固定ETF维度与类别数预设，无法动态适应未知类别数量；② 依赖高置信度样本对齐，低置信度样本可能未被利用；③ 对多模态或文本辅助信息的处理尚未扩展，未来可探索自适应ETF扩展与跨模态集成。

---

## 223. Secure Communications, Sensing, and Computing Towards Next-Generation Networks

**arXiv ID:** 2602.19942 | [PDF](https://arxiv.org/pdf/2602.19942v1)

**作者:** Ruiqi Liu `[一作]` (ZTE Corporation), Deniz Gündüz `[通讯]` (Imperial College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了下一代集成无线网络（通信、感知、计算）中的安全与隐私威胁与对策，涵盖物理层安全、语义与目标导向通信、传感器安全、分布式安全编码计算以及统一安全框架。

**💡 创新点**

创新点在于提出统一的安全设计视角，将传统物理层安全与新兴技术（RIS、深度联合源信道编码、量子密钥分发、语义隐私等）结合；同时为多层交叉攻击与防御提供了完整的分类与对比。

**🔧 技术方法**

主要技术包括：物理层加密与密钥生成、深度学习驱动的 JSCC 与语义压缩、差分隐私与信息瓶颈方法、量子密钥分发标准化、联邦学习与安全聚合、以及对抗机器学习与安全编码计算。

**📊 数据集**

由于是综述性工作，未使用实验数据集；引用了大量公开论文与标准（如ITU‑R IMT‑2030、3GPP、ETSI/ITU‑T QKD 规范）作为资料来源。

**📈 对比分析**

通过对比分析不同安全技术的理论性能（如信道安全容量、隐私‑效用折衷、协同计算安全负载）以及标准化进展，展示了各方案的优势与局限，未给出单一实验指标。

**⚠️ 局限性**

局限性：1）缺乏统一实验评估，难以量化不同方案在实际 6G 场景中的性能；2）对量子与多极化环境的实测验证不足；3）在跨层协同安全策略的实现细节与标准化路径仍待进一步研究。

---

## 224. Dynamic data structures for twin-ordered matrices

**arXiv ID:** 2602.18770 | [PDF](https://arxiv.org/pdf/2602.18770v1)

**作者:** Bartłomiej Bosek `[一作]` (Jagiellonian University), Anna Zych-Pawlewicz `[通讯]` (University of Warsaw)

**通讯引用:** 293 | [OpenAlex ID](https://openalex.org/A5078055627)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

设计并实现了一种动态数据结构，用于维护任意时刻保持 d‑twin‑ordered 的二进制 n×n 矩阵，支持单元查询和单元翻转操作，时间复杂度均为 O(log log n)，空间复杂度为 O_d(n) 位。

**💡 创新点**

创新点主要在于：
- 提出了可动态维护的双层 slab（斜面）分解（canonical slab decomposition）以及相应的在线算法；
- 结合 adhesive segment set 与 van Emde Boas 结构，构造了高效的更新与查询机制；
- 通过自适应的增量重构（amortized 重新计算 slab 分解）以及严格的去抖动（de‑amortization）实现了期望 O(log log n) 的 worst‑case 更新时间；
- 进一步证明该结构在理论上可与已知的静态结构（Pilipczuk 等）相比，保持相同的查询/更新性能但支持动态修改。

**🔧 技术方法**

主要技术手段包括：
- Twin‑width 的 contraction sequence 与 slab 分解概念；
- Canonical slab decomposition 的构造算法，利用扫描与 adhesive segment set；
- van Emde Boas 字典与 2D orthogonal point‑location 结构实现 O(log log n) 的查询；
- 随机化哈希表用于存储差分更新；
- 典型的增量重构与 de‑amortization 方案，借助 epoch 切分与双副本策略。

**📊 数据集**

论文未提供实验或真实数据集；所有结果均为理论分析与证明，主要关注空间与时间复杂度的上界。

**📈 对比分析**

与之前的静态结构（Pilipczuk 等）相比：
- 静态结构空间 O_d(n) 位，查询时间 O(log log n)；
- 本文动态结构同样实现 O_d(n) 位空间，查询与更新均达到 O(log log n)；
- 通过自适应重构降低了重构成本，保持了期望 O(log log n) 的更新性能；
- 论文未给出实验性能对比，所有结论均基于理论复杂度分析。

**⚠️ 局限性**

限制与不足：
- 需要事先保证矩阵始终保持 d‑twin‑ordered；
- 结构依赖随机化哈希表，更新时间为期望值；
- 重新构造 slab 分解的隐藏常数可能较大；
- 论文仅给出理论证明，缺乏实测评估；
- 对于非常大 d 或 n，空间常数因子可能影响实际可行性。

---

## 225. Multi-CoLoR: Context-Aware Localization and Reasoning across Multi-Language Codebases

**arXiv ID:** 2602.19407 | [PDF](https://arxiv.org/pdf/2602.19407v1)

**作者:** Indira Vats `[一作]` (Advanced Micro Devices), Marsha Chechik `[通讯]` (University of Toronto)

**通讯引用:** 6598 | [OpenAlex ID](https://openalex.org/A5079431306)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Multi-CoLoR 框架，结合组织知识检索与图结构推理，实现在多语言代码库中的代码定位。

**💡 创新点**

创新点在于将历史类似问题检索（SIC）与多语言统一依赖图推理相结合，形成可扩展的模块化定位系统。

**🔧 技术方法**

核心技术包括检索增强式向量检索（Weaviate + BM25）、LLM 预处理（LLM 摘要）、多语言解析器（Tree-sitter）、统一依赖图（NetworkX）、Agentic 推理（Claude 4.5 Sonnet）。

**📊 数据集**

使用 AMD 内部企业数据集，共计 2,563 题（包含 140 题 QML/C++ 子集）以及公开 Benchmarks（如 SWE-bench、Multi-SWE-bench）作对比。

**📈 对比分析**

与纯词向量检索、Code Search 以及原始 LocAgent 进行 Ablation，Multi-CoLoR 在 Acc@5 方面比基线提升 4–9%（QML+ C++）且工具调用次数下降，证明组织知识和图推理协同提升定位效果。

**⚠️ 局限性**

局限性包括仅在单一公司代码库验证，缺乏跨组织/跨语言泛化实验，且对高变动模块的索引成本和检索效率仍待进一步优化。

---

## 226. IR$^3$: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking

**arXiv ID:** 2602.19416 | [PDF](https://arxiv.org/pdf/2602.19416v1)

**作者:** Mohammad Beigi `[一作]` (University of California Davis), Lifu Huang `[通讯]` (University of California Davis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出IR³框架，通过后期反向工程RLHF代理奖励，进行解释、诊断和手术式修复。

**💡 创新点**

将奖励破解视为隐式奖励重建与解释，首次使用对比式逆强化学习(C‑IRL)与稀疏自编码器解构奖励并精准定位破解特征，随后设计四种手术式缓解策略。

**🔧 技术方法**

对比逆强化学习、最大熵IRL、稀疏自编码器、特征贡献分析、清洗奖励优化、对抗奖励塑形、约束优化及特征引导蒸馏等技术。

**📊 数据集**

基于多种奖励模型的人工标注比较集（RM‑HH、Ultra‑RM、RM‑Safe、HelpSteer2、ArmoRM‑8B）和人工生成的短篇作弊样本，用于训练和评估。

**📈 对比分析**

通过Spearman/Pearson相关、顶层10%一致性、KL、奖励差距、胜率以及MMLU/GSM8K指标验证C‑IRL与原始奖励的一致性；在长度偏差、Goodhart作弊和过度安全等三种场景下与KL正则、PPO剪裁、长度惩罚等基线对比，IR³在削弱作弊率的同时保持90%以上原有能力。

**⚠️ 局限性**

仅为后期工具，无法在训练期间实时检测；需事先标注少量作弊示例；对未预见的破解模式依赖人工判断。

---

## 227. Classroom Final Exam: An Instructor-Tested Reasoning Benchmark

**arXiv ID:** 2602.19517 | [PDF](https://arxiv.org/pdf/2602.19517v1)

**作者:** Chongyang Gao `[一作]` (Northwestern University), Kezhen Chen `[通讯]` (Analogy AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了CFE-Bench，多模态与文本的课堂期末考试基准，用真实课程题目评估大型语言模型的多步推理能力。

**💡 创新点**

创新点包括：①使用教师验证的真实作业与考试题，②构建可拆分的多步推理流，③引入变量基验证协议以消除长文本误匹配，④通过步骤级诊断揭示模型瓶颈。

**🔧 技术方法**

技术实现基于 GPT‑mini 的结构化变量抽取与验证，链式思考提示，单步注入与多步推理进程评估，并使用多模态视觉‑语言预训练模型。

**📊 数据集**

数据集包含 449 道 STEM 题目（305 文本，144 多模态），来源于公开课程、考试、作业，涵盖物理、数学、工程、化学、生物等 20+ 学科。

**📈 对比分析**

评价指标为变量准确率和问题准确率，前者衡量步骤级成功率，后者衡量完整答案准确率；Gemini‑3.1‑pro‑preview 在整体上最高，问题准确率 59.69%，最佳开源模型 Qwen‑3.5‑397B 达 47.44%。

**⚠️ 局限性**

局限性包括：模型在生成中间步骤时易出现漂移导致错误累积，推理步骤冗余降低效率；多模态性能相对落后；缺乏针对更广泛视觉场景和跨学科通用性的训练与验证。

---

## 228. Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation

**arXiv ID:** 2602.19184 | [PDF](https://arxiv.org/pdf/2602.19184v1)

**作者:** Thanh Nguyen Canh `[一作]` (Japan Advanced Institute of Science and Technology), Xiem HoangVan `[通讯]` (Vietnam National University)

**通讯引用:** 361 | [OpenAlex ID](https://openalex.org/A5022703513)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种“人类到机器人”分阶段模仿学习框架，使机器人仅凭普通视频演示学习抓取、移动、放置等基本操作。

**💡 创新点**

将视频理解与控制学习解耦，使用Temporal Shift Module与Vision‑Language Model实现精准的动作+对象识别，并通过无语法命令与层次化奖励实现子目标化学习。

**🔧 技术方法**

采用Temporal Shift Module、Vision‑Language Model、TD3深度强化学习、对象选择算法、模糊检测与重叠检测、BLEU评估及Sim2Real迁移技术。

**📊 数据集**

使用Something‑Something V2、135段自制演示视频、12个标准物体+9新物体的对象集，并在PyBullet仿真环境与UF850真实实验平台上进行训练与测试。

**📈 对比分析**

与Video2Command、V2C、Watch‑and‑Act等视频‑命令模型以及SAC、DDPG、PPO、Asym‑PPO等RL算法对比，动作分类精度达89.97%，BLEU‑4在标准集为0.351、在新物体集为0.265，机器人成功率平均87.5%，在所有动作上均优于基线。

**⚠️ 局限性**

仅支持四种基本手工动作，单臂无工具使用，仿真到现实迁移仍需改进，物体遮挡与复杂场景下的对象识别仍有限。

---

## 229. TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation

**arXiv ID:** 2602.19053 | [PDF](https://arxiv.org/pdf/2602.19053v1)

**作者:** Qingwen Zhang `[一作]` (KTH Royal Institute of Technology), Patric Jensfelt `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 7107 | [OpenAlex ID](https://openalex.org/A5028082686)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了TeFlow，一种利用多帧自监督的实时场景流估计框架

**💡 创新点**

通过时间集成和投票策略构建候选运动池，聚合内部网络预测与外部几何候选，生成更稳定的监督信号；同时引入动态集群损失（点级+集群级）实现规模平衡训练

**🔧 技术方法**

多帧时间集成、方向一致性投票、加权聚合、动态集群损失、DeltaFlow骨干、HDBSCAN聚类、DUFOMap分割

**📊 数据集**

Argoverse 2、nuScenes、Waymo Open Dataset

**📈 对比分析**

与优化型方法（FastNSF、EulerFlow、Floxels）和流行的自监督前馈方法（SeFlow、VoteFlow、ZeroFlow等）对比；在Argoverse 2上取得3.57 cm Three‑way EPE、与Floxels同级但速度提升150×；在nuScenes上取得4.64 cm Three‑way EPE、比SeFlow++提升约22%；在Waymo上亦表现领先

**⚠️ 局限性**

在帧数超过5帧时性能趋于平稳甚至下降，主要受远程帧噪声影响；仍略低于极慢的优化基方法；对静态/动态分割质量敏感

---

## 230. Whisper: Courtside Edition Enhancing ASR Performance Through LLM-Driven Context Generation

**arXiv ID:** 2602.18966 | [PDF](https://arxiv.org/pdf/2602.18966v1)

**作者:** Yonathan Ron `[一作]` (Reichman University), Tammuz Dubnov `[通讯]` (Reichman University)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5062106164)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种多代理LLM管道，在不重新训练Whisper模型的前提下，通过在解码阶段注入领域上下文、正式姓名和术语提示，显著提升NBA体育解说语音识别的准确性。

**💡 创新点**

创新点在于将LLM生成的短自然语言提示直接注入Whisper的初始提示通道，并通过多任务代理（主题分类、命名实体识别、术语提取、决策过滤）实现精准的领域适配，避免了传统的模型微调或后处理方法。

**🔧 技术方法**

技术主要包括Whisper的初始提示机制、多代理LLM架构、模糊匹配（Levenshtein+Jaro–Winkler）、基于词典的NBA球员名表与术语库、Prompt长度控制与决策过滤逻辑。

**📊 数据集**

使用了421段NBA篮球解说音频，并由领域专家进行标注，形成人工校正的黄金标准。

**📈 对比分析**

与基线、仅主题提示、LLM后处理、仅命名实体提示等四种变体比较，最完整的P4管道实现相对WER下降17.0%，40.1%段落得到提升，降级率仅7.1%，显著优于其他方案。

**⚠️ 局限性**

局限包括需要持续维护领域知识库、对非体育域的通用性尚未验证、对LLM API的成本和可用性依赖，以及在极端噪声或快速语速场景下仍可能出现误差。

---

## 231. Task-Aware Exploration via a Predictive Bisimulation Metric

**arXiv ID:** 2602.18724 | [PDF](https://arxiv.org/pdf/2602.18724v1)

**作者:** Dayang Liang `[一作]` (Xiamen University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6823 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了TEB——一种基于预测性Bisimulation度量的任务感知探索框架，联合视觉表示学习与内在奖励；

**💡 创新点**

创新点在于用预测的高斯奖励差异代替稀疏奖励差异，保证Bisimulation度量在稀疏奖励下不退化，并基于该度量构造潜在函数形状的探索奖励；

**🔧 技术方法**

核心技术包括Bisimulation度量、奖励预测网络（高斯分布）、潜在空间编码器、基于潜在距离的潜在函数奖励和策略梯度训练；

**📊 数据集**

使用MetaWorld（视觉操纵任务）和Maze2D（低维导航）两套公开基准；

**📈 对比分析**

与DrM、CTRL‑SR、RAP等视觉RL基线以及ICM、CeSD、LBS等探索方法对比，在MetaWorld上实现更高成功率（如87.9%），在Maze2D上获得更高状态覆盖率，整体性能显著优于现有方法；

**⚠️ 局限性**

局限性包括：仍需对高斯奖励预测进行调参；对奖励稀疏程度极低的环境可能需要更大样本；方法在某些简单任务如Box‑close的提升有限。

---

## 232. A Comparative Analysis of Peer Support in Forum-based and Chat-based Mental Health Communities: Technical-Structural-Functional Model of Social Support

**arXiv ID:** 2602.19232 | [PDF](https://arxiv.org/pdf/2602.19232v1)

**作者:** Han Li `[一作]` (Cornell University), Han Li `[通讯]` (Cornell University)

**通讯引用:** 461366 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统比较了20个基于论坛的和20个基于聊天的心理健康社区，分析了不同技术结构对信息、情感与陪伴三类支持类型的影响，并构建了技术-结构-功能模型；

**💡 创新点**

创新点在于首次将技术特点纳入社会支持模型，结合机器学习和社交网络分析，揭示了社区格式通过网络结构中介机制对支持类型的影响；

**🔧 技术方法**

主要技术包括监督式机器学习（随机森林+LIWC、LDA主题、心理健康词典和情感词典特征）、社交网络分析（网络密度、中心化指标）以及多层结构方程模型；

**📊 数据集**

使用的数据集来自40个中国心理健康社区（20论坛型：百度贴吧、豆瓣小组；20聊天型：微信、QQ），共计182,778条文本，覆盖约8,269名活跃用户；

**📈 对比分析**

比较方法采用多层结构方程模型和中介分析，机器学习分类准确率达0.83；结果显示论坛型社区信息与情感支持比例显著高于聊天型，而聊天型社区陪伴支持更高，网络中心化与支持类型关联显著；

**⚠️ 局限性**

局限性包括：横断面设计无法确定因果关系；用户自选可能导致平台与需求匹配偏差；仅涉及心理健康主题及中国平台；使用三类单标签支持分类，忽略多重支持与细粒度子类；数据仅为一月观测，缺乏长期变化视角。

---

## 233. BigMaQ: A Big Macaque Motion and Animation Dataset Bridging Image and 3D Pose Representations

**arXiv ID:** 2602.19874 | [PDF](https://arxiv.org/pdf/2602.19874v1)

**作者:** Lucas Martini `[一作]` (Hertie Institute, University of Tübingen), Martin A. Giese `[通讯]` (Hertie Institute, University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文构建了大型无标记多视角猴子运动捕捉数据集BigMaQ，结合了三维面部模型和关节角度描述，并在此基础上评估动作识别性能。

**💡 创新点**

创新点在于首次将高质量3D姿态与面部网格表面重建整合到非人类灵长类动物的行为识别任务中，并提供了超过750个互动场景、个体专属纹理化模型和大规模动作标签。

**🔧 技术方法**

使用技术包括多摄像头同步标定、YOLOv8检测、HRNet-W48关键点估计、SAM 2语义分割、基于LBS的线性混合蒙皮与可微渲染、时间一致性损失、姿态参数θ（轴角）表征，以及多种视觉编码器（ResNet50、ViT、DINOv2、TimeSformer、VideoPrism）与姿态特征融合的Transformer模型。

**📊 数据集**

主要数据集为BigMaQ（全数据集）和其子集BigMaQ500（511个动作、8176段多视角视频），并与公开的MAMMAL、AniMer+等动物3D姿态跟踪方法进行对比。

**📈 对比分析**

通过多视角网格拟合对比，BigMaQ在IoU、MPJPE、MPJTD指标上分别优于MAMMAL和AniMer+，在动作识别上加入姿态特征后，mAP从约34提升至44，尤其在社交交互类动作上表现显著提升。

**⚠️ 局限性**

局限性包括行为标签仅由两名研究者标注，复杂多个体场景下检测/分割误差影响姿态重建，缺乏跨环境的泛化评估，以及对单视角重建的进一步研究尚未完成。

---

## 234. An Explainable Memory Forensics Approach for Malware Analysis

**arXiv ID:** 2602.19831 | [PDF](https://arxiv.org/pdf/2602.19831v1)

**作者:** Silvia Lucia Sanna `[一作]` (University Of Cagliari), Giorgio Giacinto `[通讯]` (Consorzio Interuniversitario Nazionale per l’Informatica)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种跨平台的内存取证与解释框架，利用大型语言模型对Windows与Android完整内存或进程内存进行自动解析、IoC提取与可解释性分类。

**💡 创新点**

创新点在于将通用LLM与传统Volatility分析相结合，提供可解释的报告，并通过人机协作的内核重编译来简化Android内存获取。

**🔧 技术方法**

技术手段包括Volatility 2/3内存分析、LiME与Fridump获取、ChatGPT‑4o‑mini/ Gemini‑2.0‑flash‑lite进行输出解释与IoC抽取，以及正则与熵规则的NLP规则引擎。

**📊 数据集**

实验数据集为公开的Windows恶意软件（勒索软件等）与Android恶意与伪造APK（Netflix、Twitter、Tiktok等）在虚拟机/模拟器中执行得到的完整内存与进程快照。

**📈 对比分析**

在Windows上与Android上分别与VirusTotal、Drebin、Entroplyzer以及自定义NLP规则进行对比，LLM方案在IoC覆盖率上超过传统工具，尤其在动态行为与系统级痕迹上表现更优，准确率在Windows约93%/Android约90%。

**⚠️ 局限性**

局限性包括样本规模有限、仅在单一Android内核/虚拟机环境验证、对LLM输出的可靠性与一致性尚未充分评估，以及需保留内存数据的隐私与合规性挑战。

---

## 235. ORION: ORthonormal Text Encoding for Universal VLM AdaptatION

**arXiv ID:** 2602.19530 | [PDF](https://arxiv.org/pdf/2602.19530v1)

**作者:** Omprakash Chakraborty `[一作]` (ÉTS Montréal), Ismail Ben Ayed `[通讯]` (ÉTS Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过仅使用类别名称对视觉‑语言模型（VLM）的文本编码器进行细粒度正交化微调，以提升文本原型的判别性；

**💡 创新点**

提出一种仅依赖文本、无图像监督的正交性正则化框架（Orthogonal Text Fine‑Tuning, OTF），并通过低秩适配器（LoRA）实现参数高效微调；

**🔧 技术方法**

使用低秩适配器LoRA对文本编码器进行微调，构造融合“正交化惩罚+保持原始原型”的复合损失，并给出其最大似然概率解释；

**📊 数据集**

在11个标准视觉分类基准上评估，包括细粒度（Pets、Cars、Aircraft）、纹理（DTD、Food）、场景/卫星（SUN、EuroSAT）、通用（Flowers、Caltech、UCF、ImageNet）等多域数据集；

**📈 对比分析**

与原始CLIP/MetaCLIP、CoOp、CLAP以及MTA、TPT、StatA等多种方法对比，零样本下平均提升约1.5%–2.0%，少样本（1–4 shot）提升幅度更显著（高达+13.6%），在TTA场景亦持续上升（约+1.5%）；

**⚠️ 局限性**

仅使用类别名称可能无法捕捉更细致的视觉特征；正交化对极端低样本或高度分化类的收益有限；当类别数量过多或语义关系复杂时，软正则化仍可能导致语义信息损失。

---

## 236. Hydrodynamic Performance Enhancement of Unmanned Underwater Gliders with Soft Robotic Morphing Wings for Agility Improvement

**arXiv ID:** 2602.20054 | [PDF](https://arxiv.org/pdf/2602.20054v1)

**作者:** A. Giordano `[一作]` (Laboratory of Sustainability Robotics Empa), M. Kovac `[通讯]` (Laboratory of Sustainability Robotics Empa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

评估并比较软形态变形翼与传统刚性翼在水下无人机（UUV）中的流体动力效率，使用结构有限元与CFD仿真，并与文献实验数据进行验证。

**💡 创新点**

引入可变形软翼替代刚性翼，利用软体机器人技术实现翼形可调节，从而提升升阻比、静态稳定性和机动性，并显著增加UUV的作业范围。

**🔧 技术方法**

结构仿真采用ANSYS Mechanical，采用Mooney–Rivlin双参数模型；CFD仿真使用ANSYS Fluent，k‑ω SST湍流模型；并对升阻比、升力、阻力、气动矩进行数值计算。

**📊 数据集**

使用文献中的软翼实验数据（不同注射量的形变与升阻比）以及对应的UUV实验数据，作为模型验证与对比基准；未使用公开的专用数据集。

**📈 对比分析**

通过计算升阻比（C_L/C_D）和效率，比较软翼UUV与刚性翼UUV的性能。结果显示软翼UUV在同等条件下效率提升9.75%，升阻比误差约15.7%，且在负攻角时仍保持正升力，表明软翼能显著提升整体性能。

**⚠️ 局限性**

局限性包括：仅在实验所述低流速（Re≈5.7×10⁴–9.2×10⁴）范围内验证，转折点与流动分离对模型敏感；软翼的形变响应时间约1分钟，未满足高频动态控制需求；未对长期作业中的能耗与可靠性进行系统评估。

---

## 237. Less is More: Convergence Benefits of Fewer Data Weight Updates over Longer Horizon

**arXiv ID:** 2602.19510 | [PDF](https://arxiv.org/pdf/2602.19510v1)

**作者:** Rudrajit Das `[一作]` (Google Research), Vahab Mirrokni `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在有限内层步骤下的多域数据混合问题，证明短视更新T=1会导致系统性偏差，并给出了最优的 lookahead 长度 T 的理论上界。

**💡 创新点**

创新点在于从理论上推导了数据混合中 lookahead Horizon T 的最优增长速率，证明了“少即多”原则，并给出确定性场景下 T=Θ(log N) 与随机性场景下 T=Θ(√(N log N)) 的最优选择。

**🔧 技术方法**

采用了二层优化、隐函数定理、近似 Hessian、镜像下降、Neumann 级数截断等技术，并在强凸假设下完成了收敛分析。

**📊 数据集**

实验数据集为 MNIST，构造了三类训练域（原始、随机旋转、标签噪声）以及旋转验证集。

**📈 对比分析**

通过与 T=1 及不同 T 的对比实验，发现适度的 T 能显著降低验证损失并提升重要域的权重，表现优于短视策略。

**⚠️ 局限性**

局限性包括仅在每域损失强凸时适用，未对非凸情况给出收敛保证，且大规模模型和数据集的实验尚未完成。

---

## 238. MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations

**arXiv ID:** 2602.18792 | [PDF](https://arxiv.org/pdf/2602.18792v1)

**作者:** Changlu Guo `[一作]` (Technical University of Denmark), Morten Rieger Hannemose `[通讯]` (Technical University of Denmark)

**通讯引用:** 124 | [OpenAlex ID](https://openalex.org/A5058264848)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MaskDiME，一种无训练、基于扩散模型的视觉反事实解释框架，利用自适应双重掩码聚焦决策相关区域，生成局部、语义一致的反事实图像。

**💡 创新点**

创新点在于：①引入自适应双重掩码机制，利用分类器梯度动态生成前向和后向的掩码，实现空间精确控制；②采用一阶梯度放缩和单步清晰图像估计，显著提升采样速度与质量；③通过在扩散采样过程中的掩码约束与梯度引导协同，使模型在保持高图像保真度的同时实现决策一致性。

**🔧 技术方法**

核心技术包括：扩散模型（DDPM）与无条件权重、分类器梯度引导、联合损失（分类、感知、L1）、Tweedie公式的一步清晰图像估计、形态学膨胀的掩码生成以及梯度缩放因子。

**📊 数据集**

实验数据集覆盖五个视觉域：CelebA、CelebA‑HQ（面部属性）、BDD100K、BDD‑OIA（自动驾驶场景）、ImageNet（多类别相似对）。

**📈 对比分析**

与 DiME、ACE、FastDiME、RCSB、LDCE、TiME 等方法在 FID、sFID、FR、FVA/FS、COUT、S^3 等指标上对比，MaskDiME 在大多数数据集实现或逼近最优的图像真实性、有效性与决策一致性；同时推理速度提升约 30×，GPU 内存使用仅为 ACE/RCSB 的十分之一。

**⚠️ 局限性**

局限性包括：①依赖分类器梯度，易受梯度噪声影响，尤其在 ImageNet 等多类别场景下定位不精确；②缺乏真实的反事实标注，难以从因果角度进行严格验证；③评价指标（如 FID）可能因图像中大范围未修改区域而产生偏差。

---

## 239. Mapping Networks

**arXiv ID:** 2602.19134 | [PDF](https://arxiv.org/pdf/2602.19134v1)

**作者:** Lord Sen `[一作]` (National Institute of Technology), Shyamapada Mukherjee `[通讯]` (National Institute of Technology)

**通讯引用:** 215 | [OpenAlex ID](https://openalex.org/A5024019557)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Mapping Networks 通过低维潜在向量和权重调制来生成目标网络参数，消除对大模型训练的需求；

**💡 创新点**

创新点在于证明 Mapping Theorem，设计可训练的低维映射网络和 Mapping Loss，利用权重调制实现参数空间的低维化；

**🔧 技术方法**

使用超参数优化的潜在向量、正交初始化的映射网络、权重调制、低秩分解、剪枝以及自适应正则化的 Mapping Loss；

**📊 数据集**

在 MNIST、Fashion‑MNIST、Celeb‑DF、FF++、Cityscapes、空气污染时间序列等数据集上进行实验；

**📈 对比分析**

与全参数网络基线相比，Mapping Networks 在图像分类、深度伪造检测、图像分割、时间序列预测等任务中实现了 200×–500× 的可训练参数压缩，同时保持或提升准确率，且过拟合显著下降；

**⚠️ 局限性**

局限在于单一潜在向量训练时内存开销较大，需采用层级训练；此外对极大规模模型的可扩展性仍待验证。

---

## 240. Depth-Enhanced YOLO-SAM2 Detection for Reliable Ballast Insufficiency Identification

**arXiv ID:** 2602.18961 | [PDF](https://arxiv.org/pdf/2602.18961v1)

**作者:** Shiyu Liu `[一作]` (Marshall University), Pingping Zhu `[通讯]` (Marshall University)

**通讯引用:** 1759 | [OpenAlex ID](https://openalex.org/A5090275826)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于RGB‑D的深度校正YOLO–SAM2框架，用于铁路轨道桥砟不足的自动检测。

**💡 创新点**

引入了床位对齐的多项式深度偏差校正、SAM2旋转边界框提取以及双重几何判别标准，显著提升了缺失桥砟的召回率。

**🔧 技术方法**

使用YOLOv8检测、SAM2分割、RANSAC+多项式拟合深度校正、EMA平滑、旋转边界框、深度残差和边缘缺口双阈判别。

**📊 数据集**

基于实测的Intel RealSense D435采集的1,823张RGB‑D轨道图像，其中1,405张用于训练，418张用于测试。

**📈 对比分析**

与YOLO仅RGB、YOLO+SAM2轴对齐盒以及深度校正+轴对齐盒+双阈等方案对比，最优配置在召回率0.806、F1分数0.813，显著优于单RGB模型。

**⚠️ 局限性**

受限于单摄像头视角、真实轨道曲率及深度传感噪声，且缺乏多时相或多摄像头融合，导致在极端光照或曲线条件下仍可能出现误检。

---

## 241. Vibe Coding on Trial: Operating Characteristics of Unanimous LLM Juries

**arXiv ID:** 2602.18492 | [PDF](https://arxiv.org/pdf/2602.18492v1)

**作者:** Muhammad Aziz Ullah `[一作]` (Texas Tech University), Abdul Serwadda `[通讯]` (Texas Tech University)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5014177748)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究使用LLM陪审团对SQL代码进行安全审查，评估不同规模和组合的委员会在执行准确率上的表现。

**💡 创新点**

采用统一执行验证的标签方法构建可复现的基准，系统性枚举所有强模型组合并分析一致投票规则对安全性和吞吐量的影响。

**🔧 技术方法**

使用Ollama托管多种开源LLM（如GPT-OSS、CodeGemma等），在MySQL 8.0.44上执行查询并通过结果对比进行判定，利用Python、LangChain、SQLAlchemy实现评估流水线。

**📊 数据集**

82个面向MySQL的自然语言到SQL提示，覆盖基本检索、聚合、连接、多表联结及子查询等多层级复杂度，并在10个随机生成的实例数据库上进行验证。

**📈 对比分析**

通过每个委员会的TPR、FPR和Youden J指标衡量，发现单模型差异大，2-3人一致投票可在保持高TPR的同时降低FPR，4-5人委员会趋向极端保守，最佳组合集中在两大GPT模型。

**⚠️ 局限性**

仅在单一MySQL schema下实验，使用温度0的生成，未探讨多数投票或成本感知阈值，缺乏对不同DBMS、查询难度和时延/成本的评估。

---

## 242. Mitigating Artifacts in Pre-quantization Based Scientific Data Compressors with Quantization-aware Interpolation

**arXiv ID:** 2602.20097 | [PDF](https://arxiv.org/pdf/2602.20097v1)

**作者:** Pu Jiao `[一作]` (University of Kentucky), Franck Cappello `[通讯]` (Argonne National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种后解压缩的量化感知插值算法，用来减轻基于预量化的误差约束无损压缩产生的伪影，提高数据质量；

**💡 创新点**

创新点在于对预量化压缩器产生的量化误差进行系统特征化，发现误差符号与量化索引梯度相关、误差大小随距离量化边界递减，并基于此设计了两轮欧几里得距离变换的插值修正流程；

**🔧 技术方法**

采用量化边界识别、欧几里得距离变换、误差符号传播、IDW插值等传统算法，并在共享内存（OpenMP）和分布式内存（MPI）上实现并行化；

**📊 数据集**

在五个真实科学数据集（CESM、Hurricane、NYX、S3D、JHTDB）上进行实验；

**📈 对比分析**

与高斯、均匀、维纳滤波器及原始无损压缩结果对比，实验显示在中等误差边界下SSIM提升可达108.33%，并在保持高压缩吞吐量的同时，MPI版本在512核时仍保持约83%扩展性；

**⚠️ 局限性**

局限性包括：算法在极大误差边界下对噪声敏感，处理速度依赖于距离变换的实现，且目前仅在CPU上实现，GPU加速仍待完成；

---

## 243. MANATEE: Inference-Time Lightweight Diffusion Based Safety Defense for LLMs

**arXiv ID:** 2602.18782 | [PDF](https://arxiv.org/pdf/2602.18782v1)

**作者:** Chun Yan Ryan Kan `[一作]`, Maheep Chaudhary `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对LLM的对抗性破解攻击，提出了一种在推理时利用扩散模型对隐藏状态进行密度估计并纠正异常表示的防御方法。

**💡 创新点**

创新点在于将安全问题转化为对贝叶斯表示流形的密度估计，利用扩散模型既检测又纠正异常隐藏状态，无需对抗样本、改造模型结构或再次微调。

**🔧 技术方法**

使用了DDPM扩散模型、分数匹配、异常评分和条件拒绝策略，在隐藏层空间进行检测与修正。

**📊 数据集**

在Mistral-7B-Instruct、Llama-3.1-8B-Instruct和Gemma-2-9B-it三款LLM上，利用MAD、JailbreakBench和Anthropic Sleeper Agent等数据集评估。

**📈 对比分析**

与未防御模型对比，MANATEE在攻击成功率(ASR)上平均降低约78%，在某些数据集上实现100%降低，同时保持对正常输入的语义效果。

**⚠️ 局限性**

局限性包括需手动设定阈值、仅在最终层隐藏状态上操作、对扩散推理的计算开销以及对更复杂或新型攻击的泛化能力尚待验证。

---

## 244. Toward Self-Driving Universities: Can Universities Drive Themselves with Agentic AI?

**arXiv ID:** 2602.18461 | [PDF](https://arxiv.org/pdf/2602.18461v1)

**作者:** Anis Koubaa `[一作]` (Alfaisal University), Anis Koubaa `[通讯]` (Alfaisal University)

**通讯引用:** 11962 | [OpenAlex ID](https://openalex.org/A5024626195)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出“自驱大学”概念，并通过Agentic AI实现学术与行政流程的分级自动化，完成了MUST大学的统一数据平台XEducation的构建与多项自动化工具的部署；

**💡 创新点**

创新点在于将自动驾驶车辆的分级自动化模型迁移至高等教育，提出六级自治框架，并将Agentic AI（感知‑规划‑行动‑记忆循环）与RAG相结合，实现课程、评估、认证等流程的端到端自动化；

**🔧 技术方法**

主要技术包括Transformer‑based LLM、检索增强生成（RAG）、ReAct/工具调用框架、视觉‑语言模型用于手写试卷识别、以及自定义的多任务Agentic工作流；

**📊 数据集**

使用的数据集为MUST大学内部的完整机构数据：学生信息、课程规范、学习成果、成绩与评估记录，以及由XEducation统一管理的知识库；

**📈 对比分析**

通过案例评估显示，自动化后课程规范录入、评估数据汇总和认证文件编制的时间分别缩短了约70%–90%；虽然未给出严格对比指标，但定性报告指出教师在教学、指导和研究上的时间得以显著提升；

**⚠️ 局限性**

局限性包括：需要先行构建统一、高质量的数据基础，缺乏跨校验证；在高风险决策上仍需人工审查；伦理与公平性风险（如偏见与学术诚信）需进一步治理；

---

## 245. Transcending the Annotation Bottleneck: AI-Powered Discovery in Biology and Medicine

**arXiv ID:** 2602.20100 | [PDF](https://arxiv.org/pdf/2602.20100v1)

**作者:** Soumick Chatterjee `[一作]` `[通讯]` (Human Technopole), Soumick Chatterjee (Human Technopole)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了自监督学习在医学影像、基因组学和电子健康记录中的应用，并展示了其在特征学习、表型发现、异常检测和图像配准等任务上的效果。

**💡 创新点**

创新点在于系统性整合了多领域的最新研究，证明了在无标签数据下可获得与监督模型相媲美甚至更优的性能，并提出将这些技术聚合为“基础模型”的前景。

**🔧 技术方法**

主要技术包括对比学习（SimCLR、DINO）、自监督 ViT、VAE、扩散模型、跨模态自编码器、生成式 Transformer、State Space Models 等。

**📊 数据集**

数据集涵盖 UK Biobank MRI、GTEx 组织学切片、脑肿瘤 MRI、电子健康记录等大规模无标签数据库。

**📈 对比分析**

通过与传统监督模型（如 UNet++、VAE+后处理）及基准评测（AP、AUC、RMSE）对比，本文示例性实验显示无监督方法在肿瘤检测等任务中平均精度可达 0.83，超过监督基线 0.75，且在多模态表型发现中揭示 89 个显著基因位点。

**⚠️ 局限性**

限制包括对复杂生成模型的训练成本高、缺乏明确的可解释性、在极度异质的数据中仍需改进表示的稳健性，以及基础模型整合多模态数据的技术挑战。

---

## 246. Asking the Right Questions: Improving Reasoning with Generated Stepping Stones

**arXiv ID:** 2602.19069 | [PDF](https://arxiv.org/pdf/2602.19069v1)

**作者:** Hengyuan Hu `[一作]` (Stanford University), Jakob Nicolaus Foerster `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM通过生成“踏石”问题来提高解决困难任务的能力，并提出ARQ框架。

**💡 创新点**

将提问步骤作为推理的先导，证明踏石问题能显著提升性能，并将其作为后训练任务。

**🔧 技术方法**

结合提示工程、链式思考、RLHF/DPO后训练，采用两阶段生成-评估机制。

**📊 数据集**

使用AIME 2024/2025、BeyondAIME 100道题以及合成的踏石数据集。

**📈 对比分析**

与Solver‑Only、Analogical、Least‑to‑Most等基线比较，ARQ在BeyondAIME提升约13%，多踏石时可进一步提升3–5%。

**⚠️ 局限性**

需要大量评估、对生成问题质量敏感，现有模型在生成有效踏石时仍受限，跨任务泛化尚待验证。

---

## 247. The Welfare Gap of Strategic Storage: Universal Bounds and Price Non-Linearity

**arXiv ID:** 2602.19660 | [PDF](https://arxiv.org/pdf/2602.19660v1)

**作者:** Zhile Jiang `[一作]` (Aarhus University), Stratis Skoulakis `[通讯]` (Aarhus University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了电池储能在电力市场中的效率损失，量化了中心化规划与去中心化利润最大化之间的价格无序度。

**💡 创新点**

提出了在不同价格函数下（线性、凸、单项式）价格无序度的理论上界与下界，并揭示线性价格是效率保留的必要条件。

**🔧 技术方法**

采用连续时间随机过程、凸分析、变分不等式以及价格无序度理论进行严谨推导。

**📊 数据集**

未使用具体数据集，全部为理论推导和极端实例分析。

**📈 对比分析**

通过理论证明与构造极端实例对比，得到PoA上界为4/3、在凸价格下可无界、单项式价格下上界为2（二次型达到27/19）等结果。

**⚠️ 局限性**

局限在单一储能、无网络与传输损耗约束，且在非线性价格下难以给出统一上界，未考虑多储能竞争与实际市场规则。

---

## 248. Interconnect-Aware Logic Resynthesis for Multi-Die FPGAs

**arXiv ID:** 2602.19720 | [PDF](https://arxiv.org/pdf/2602.19720v1)

**作者:** Xiaoke Wang `[一作]` (Ghent University), Dirk Stroobandt `[通讯]` (Ghent University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种面向多芯片FPGA的互连感知逻辑重构方法，在划分后对LUT级网表进行局部重构，以减少跨芯片的超级长线（SLL）连接；

**💡 创新点**

首次将逻辑重构（resubstitution）技术直接用于多芯片FPGA的逻辑层面，针对SLL减少而非仅仅通过划分或布线优化，形成新的优化维度；

**🔧 技术方法**

基于MFS（最小功能集）重构的SAT判定框架，配合窗口构造、同芯片除子选择和交叉边缘剪裁，实现局部逻辑重新表达；

**📊 数据集**

使用EPFL、MCNC以及Koios三组基准，分别覆盖组合逻辑、时序逻辑和带有硬宏的异构设计；

**📈 对比分析**

与现有的基于划分的多芯片FPGA流程对比，SLL数量平均下降约25%（2芯片）至27%（3芯片），在EPFL上最大降幅达67%；在MCNC、Koios上表现出约1.6%–12%间的SLL削减、1%–4%布线长度与临界路径延迟改善，且保持或略微提升负载平衡与实现时间；

**⚠️ 局限性**

受限于LUT级重构的可行性——硬宏、时序约束和大LUT尺寸限制了重构空间；方法未显式考虑路由和时序，可能导致部分设计的临界路径恶化，且对不同芯片架构的适配性需要进一步验证。

---

## 249. Value Entanglement: Conflation Between Different Kinds of Good In (Some) Large Language Models

**arXiv ID:** 2602.19101 | [PDF](https://arxiv.org/pdf/2602.19101v1)

**作者:** Seong Hah Cho `[一作]` (Independent), Anna Leshinskaya `[通讯]` (AI Objectives Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大型语言模型是否能区分道德、语法和经济三类好，并发现其价值表征会互相混淆，利用方向消融法纠正了这种混淆。

**💡 创新点**

提出“价值混沌（value entanglement）”概念并在多种开源与闭源模型上通过行为与内部表征实验证实，同时展示了消融干预可以恢复独立价值判断。

**🔧 技术方法**

采用句子行为问卷（Likert评分）、词嵌入向量投影、残差流差值法（difference‑of‑means）以及方向性消融（directional ablation）等技术进行表征探测与干预。

**📊 数据集**

使用自制的 MoralGrammar68（道德与语法正交）和 MoralEconomic68（道德与经济正交）句子集合，结合 Prolific 收集的人类评分及真实商品价格作为基准。

**📈 对比分析**

通过将模型评分与人类评分、商品价格进行 Pearson 相关比较，发现模型在道德判断上与人类高度一致，但在语法和经济判断上显著受道德影响；消融后语法与经济评分与人类/价格的相关性显著提升。

**⚠️ 局限性**

局限性包括：仅评估了有限的模型和任务，闭源模型虽表现较好但嵌入层仍存在混沌；消融干预对不同模型和层级的效果差异大，且混沌可能因模型规模、训练阶段或数据分布不同而变化。

---

## 250. Rethinking Preference Alignment for Diffusion Models with Classifier-Free Guidance

**arXiv ID:** 2602.18799 | [PDF](https://arxiv.org/pdf/2602.18799v1)

**作者:** Zhou Jiang `[一作]` (Westlake University), Zhen Liu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 4968 | [OpenAlex ID](https://openalex.org/A5100412087)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于分类器自由引导（CFG）的偏好对齐方法 PGD 与其对比版本 cPGD，在不重新训练基础扩散模型的前提下通过推理时引导实现对人类偏好的对齐。

**💡 创新点**

创新点在于：①将偏好优化视为 CFG 推理问题，使得只需微调小量模型即可获得对齐信号；②将正负样本分别训练成两个独立模型，形成对比引导向量，提升泛化与控制力度；③通过轻量级微调避免过拟合，并可直接复用到其他基础模型。

**🔧 技术方法**

主要技术包括：直接偏好优化（DPO）/Diffusion-DPO、分类器自由引导（CFG）、对比 PGD 的正负两模型微调、NTK 视角下的核回归解释、以及可选的蒸馏压缩。

**📊 数据集**

使用 Stable Diffusion 1.5 与 Stable Diffusion XL 作为基准模型；偏好数据来自 Pick‑a‑Pic v2（约 90 万对）和 HPDv3（约 210 万对），并在 Pick‑a‑Pic v2 测试集、HPDv2 测试集和 Parti‑Prompts 基准上评估。

**📈 对比分析**

与 SFT‑Pref、Diffusion‑DPO、Diffusion‑KTO、MaPO、Diffusion‑NPO 等基线相比，PGD 与 cPGD 在 PickScore、HPSv2/3、CLIP、ImageReward 等奖励指标上均有显著提升，win‑rate 也普遍超过 80%；同时在 FID 与多样性上保持更优或相近的性能，显示出更好的奖励-多样性-保真度折中。

**⚠️ 局限性**

局限性包括：推理时需要双模型计算，导致采样时间翻倍；过高的引导权重可能产生失真或不自然的图像；对不同偏好数据分布的适应性受限，需在高质量子集上表现更好；模型对负样本的依赖使得在负样本稀缺时效果下降。

---

## 251. WiCompass: Oracle-driven Data Scaling for mmWave Human Pose Estimation

**arXiv ID:** 2602.18726 | [PDF](https://arxiv.org/pdf/2602.18726v1)

**作者:** Bo Liang `[一作]` (Peking University), Chenren Xu `[通讯]` (Peking University)

**通讯引用:** 3775 | [OpenAlex ID](https://openalex.org/A5003999919)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于“oracle”驱动的覆盖感知数据采集框架（Compass），通过将大规模 MoCap 数据映射到 VQ‑VAE 的离散姿态空间，量化现有 mmWave HPE 数据集的覆盖率和冗余度，并在此基础上闭环选择最具信息量的目标姿态进行真实或仿真采集，从而在固定采集预算内显著提升 OOD 泛化性能。

**💡 创新点**

创新点包括：① 用 MoCap 作为先验“oracle”直接驱动 mmWave 数据采集；② 设计离散姿态词表和 k‑NN 方向覆盖度量，结合归一化冗余指数 (NRI) 评估数据集质量；③ 采用 Capped‑PPS 采样策略和 Efraimidis–Spirakis 算法实现对稀疏、可行姿态的高效挑选；④ 构建闭环反馈循环，动态更新覆盖集并迭代改进数据集。

**🔧 技术方法**

核心技术包括 VQ‑VAE 离散编码器、k‑NN 近邻覆盖度量与 NRI 评估、Capped‑PPS 加权采样、Efraimidis–Spirakis 加权无放回采样、雷达点云预处理与 3D 关键点回归（Point‑Transformer）、仿真雷达管线（RF Genesis）和真实 mmWave 采集硬件。

**📊 数据集**

使用 AMASS（8.9M 帧 MoCap）作为覆盖先验；评估 mmWave 数据集 mmBody（160k）和 MMFi（321k）；在仿真中生成合成 mmWave 数据；在真实实验中使用 3Tx‑4Rx 77 GHz 雷达和多摄像头 MoCap 进行标注。

**📈 对比分析**

与传统顺序动作采集基线（mmBody‑trace）比较，Compass 在相同数据预算下的 OOD MPJPE 下降约 25–30 mm，数据规模指数 α 从 -0.0008 提升到 -0.0116；在覆盖度量上，Compass 在 k=12 时覆盖率提升至 ~70%（相比 mmBody 的 3.7% 和 MMFi 的 79.8%）；在真实实验中，Compass 在 8k 采集预算下 OOD MPJPE 为 105.7 mm，明显优于基线 112.9 mm，接近重新采集（oracle）92.4 mm 的水平。

**⚠️ 局限性**

局限性包括：① 仅针对姿态覆盖，未同时解决环境/信道多样性；② 依赖 AMASS 完整性，某些文化或极端动作仍缺失；③ 仿真与真实场景的差距仍存在，需进一步验证大规模实测效果；④ 采样策略对阈值和 k 的选择敏感，可能在不同数据规模下需要调优。

---

## 252. CORVET: A CORDIC-Powered, Resource-Frugal Mixed-Precision Vector Processing Engine for High-Throughput AIoT applications

**arXiv ID:** 2602.19268 | [PDF](https://arxiv.org/pdf/2602.19268v1)

**作者:** Sonu Kumar `[一作]` (Indian Institute of Technology Indore), Santosh Kumar Vishvakarma `[通讯]` (Indian Institute of Technology Indore)

**通讯引用:** 2257 | [OpenAlex ID](https://openalex.org/A5068792760)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于迭代CORDIC的可运行时自适应向量处理引擎，能够在边缘AI加速中动态切换精度与近似深度，兼容4/8/16位固定点操作；

**💡 创新点**

核心创新在于低资源、可调迭代MAC单元与时间复用多激活函数块的组合，使得在不增加辅助纠错硬件的前提下即可实现近似与精确模式的无缝切换；

**🔧 技术方法**

技术实现包括可配置迭代的CORDIC MAC、时间复用的多激活函数单元（Sigmoid、Tanh、Softmax、GELU、Swish、ReLU、SELU）、可扩展的PE阵列、以及基于FPGA/ASIC的软硬件协同验证；

**📊 数据集**

评估使用标准CNN（如VGG‑16、TinyYOLO‑v3）和Transformer‑style MLP 任务，结合 ImageNet、COCO 等公开数据集进行推理测试；

**📈 对比分析**

与现有SoTA加速器（Flex‑PE、LPRE、TVLSI 等）比较，ASIC 256‑PE 版本实现 4.83 TOPS/mm² 计算密度、11.67 TOPS/W 能效；FPGA 版在 VC707 上达到 6.43 GOPS/W、0.53 W；Pynq‑Z2 上的 end‑to‑end 延迟 84.6 ms、功耗 0.43 W，均显著优于同类硬件；

**⚠️ 局限性**

局限性主要在于目前仅支持固定点定点运算，缺乏完整的编译器自动化精度/迭代配置；对大规模模型和训练过程的支持有限，且尚未完成全物理设计与布局优化。

---

## 253. RKHS Representation of Algebraic Convolutional Filters with Integral Operators

**arXiv ID:** 2602.19094 | [PDF](https://arxiv.org/pdf/2602.19094v1)

**作者:** Alejandro Parada-Mayorga `[一作]` (University of Colorado), Juan Bazerque `[通讯]` (University of Pittsburgh)

**通讯引用:** 2120 | [OpenAlex ID](https://openalex.org/A5029019906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了将代数卷积滤波器通过 RKHS 的积分算子形式进行表示的方法，并将其应用于信号处理和深度学习。

**💡 创新点**

创新点在于利用 RKHS 统一代数信号模型与卷积神经网络的框架，实现了更灵活的滤波器设计与理论分析。

**🔧 技术方法**

采用 RKHS 理论、积分算子、代数信号处理、谱图方法以及卷积神经网络的技术。

**📊 数据集**

实验使用了经典的图数据集 Cora、Citeseer 和 MNIST 等。

**📈 对比分析**

与传统的频域卷积和 GCN 进行对比，实验结果显示在节点分类任务中准确率提高了约 3%-5%，且收敛速度更快。

**⚠️ 局限性**

主要局限是对大规模稀疏图的计算复杂度较高，且需要先验知识确定合适的核函数。

---

## 254. From Few-Shot to Zero-Shot: Towards Generalist Graph Anomaly Detection

**arXiv ID:** 2602.18793 | [PDF](https://arxiv.org/pdf/2602.18793v1)

**作者:** Yixin Liu `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**通讯引用:** 23960 | [OpenAlex ID](https://openalex.org/A5008056593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了通用图异常检测（Generalist Graph Anomaly Detection）范式，设计了少样本方法ARC和零样本方法ARCZero，实现了在未见图数据上无需微调即可检测异常节点。

**💡 创新点**

创新点包括：① 用平滑性排序的特征对齐统一不同域图的特征空间；② 构建残差式GNN编码器捕捉跨域的异常特征；③ 采用跨注意力的上下文学习实现少样本异常评分；④ 引入伪上下文迭代机制完成零样本检测。

**🔧 技术方法**

技术手段涵盖PCA降维、平滑性特征排序、残差多跳GNN、跨注意力重构、k‑means伪上下文初始化、迭代自适应重构与多轮得分聚合。

**📊 数据集**

使用了17个真实图数据集，包括Cora、CiteSeer、ACM、PubMed、BlogCatalog、Flickr、Facebook、Weibo、Reddit、Questions、Amazon、YelpChi、Coauthor-CS、Amazon-Photo、Tolokers、T-Finance和Elliptic，覆盖学术、社交、电商、金融等多领域。

**📈 对比分析**

与现有监督/无监督基线以及最新的UNPrompt、AnomalyGFM等方法比较，ARC和ARCZero在AUROC/AUPRC上显著优于所有基线，在零样本情境下仍保持竞争力，且无需数据特定微调即能获得最优性能。

**⚠️ 局限性**

局限性包括：① 在某些数据集上ARCZero仍可能被更精细的伪上下文过滤方法超越；② 伪正常节点数的选择和初始化质量对性能影响大，缺乏自适应机制；③ 目前缺乏理论证明对通用异常检测的泛化保证。

---

## 255. Mitigating Shortcut Learning via Feature Disentanglement in Medical Imaging: A Benchmark Study

**arXiv ID:** 2602.18502 | [PDF](https://arxiv.org/pdf/2602.18502v1)

**作者:** Sarah Müller `[一作]` (University of Tübingen), Philipp Berens `[通讯]` (University of Tübingen)

**通讯引用:** 16447 | [OpenAlex ID](https://openalex.org/A5043130208)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了多种特征去耦方法在医学影像中缓解捷径学习的效果，结合人工与真实数据集进行实验。

**💡 创新点**

创新点在于首次对抗性学习、距离相关、互信息与MMD等去耦技术进行统一基准比较，并揭示数据重平衡与去耦的协同优势，特别是距离相关+重平衡的高效性。

**🔧 技术方法**

使用了编码器-子空间拆分框架，加入统计相互独立约束（距离相关、互信息神经估计、MMD）以及对抗性判别器，并通过重平衡采样与交叉验证优化模型。

**📊 数据集**

实验数据包括Morpho‑MNIST（数字写法风格）、CheXpert（胸部X光，性别为混淆因子）和OCT（视网膜扫描，合成径向陷波为混淆因子）。

**📈 对比分析**

通过在原始、平衡和反转测试集上计算AUROC以及kNN子空间判别率进行对比，结果表明所有去耦方法均优于基线，距离相关+重平衡与MINE在倒置分布上表现最佳，且实现了更快的收敛。

**⚠️ 局限性**

局限性包括仅考虑单一混淆因子、低维子空间可能限制表达能力、MINE训练时间过长、MMD对核选择敏感，以及未探索多重或交互混淆因子对模型的影响。

---

## 256. I Dropped a Neural Net

**arXiv ID:** 2602.19845 | [PDF](https://arxiv.org/pdf/2602.19845v1)

**作者:** Hyunwoo Park `[一作]` `[通讯]` (Carnegie Mellon University), Hyunwoo Park (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究如何在仅有网络层权重和训练数据的情况下，恢复被打乱的48块残差网络的原始层顺序。

**💡 创新点**

创新点在于利用残差块的动态等距性导致的负对角结构和对角优势比进行块配对，并用 Bradley–Terry 排序和局部 hill‑climb 逐步恢复块序列。

**🔧 技术方法**

使用了对角优势比、Hungarian 匹配、delta‑norm/输出投影 Frobenius 归一化、Bradley–Terry 软排序、MSE 冒泡修复等技术。

**📊 数据集**

数据集为 10,000 条 48 维输入特征、对应的原始模型预测和真实标签，以及 97 个不标记的线性层权重。

**📈 对比分析**

通过与 delta‑norm、W_out_F 等不同初始排序的直接 hill‑climb 和 BT+repair 方案对比，BT+repair 在 5 轮内收敛到 MSE=0，W_out_F seed 在 6 轮收敛，性能与直接 hill‑climb 差别不大。

**⚠️ 局限性**

局限在于需要完整的权重与训练数据，且对残差块间几乎可交换的假设成立；在更大规模或不同架构下可否推广仍未知。

---

## 257. Progressive Value Reading: The Use of Motion to Gradually Examine Data Involving Large Magnitudes

**arXiv ID:** 2602.19853 | [PDF](https://arxiv.org/pdf/2602.19853v1)

**作者:** Leni Yang `[一作]` (Inria), Pierre Dragicevic `[通讯]` (Inria)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文提出并定义了“渐进式数值阅读（Progressive Value Reading）”这一概念，探讨如何通过运动（包括视觉、时间与体力）来让观众逐步检视并感知大幅度数据；随后基于实例语料构建了包含10个维度的设计空间，涵盖数据呈现、运动实现以及促进理解与参与的设计策略。

**💡 创新点**

创新点：
1) 明确了渐进式数值阅读的定义和核心特征，填补了以往仅讨论多尺度可视化或滚动叙事的空白；
2) 构建了全新的十维设计空间，为设计师和研究者提供了统一术语与系统框架；
3) 将运动、时间、体力等非传统视觉通道与传统空间编码、叙事、锚点等策略结合，提出了可操作的设计策略组合。

**🔧 技术方法**

技术与方法：
- 运动实现技术（交互式平移、动画平移、可步行信息对象、车辆/虚拟移动、信息对象逐步构建）。
- 视觉与感知技术（长度/面积/体积编码、可视化锚点、进度回顾、短路、叙事）。
- 研究方法：案例收集与整理（真实案例、工作坊生成案例、说明性示例）、编码与维度迭代、对比分析（与Solen等多尺度可视化空间对比）。

**📊 数据集**

数据集：本文并未使用统一的实验数据集，而是汇总了大量公开的可视化实例（如全球人口、富豪财富、航天、海洋深度等），这些实例中包含的数值来自公开统计、政府报告或媒体报道。

**📈 对比分析**

比较与评估：文章未进行量化实验或性能评测，主要采用案例比较与维度编码来验证设计空间的完整性与可覆盖性；对比分析强调该空间相较于Solen等空间更细粒度、更适合单值/大幅度情境。因缺乏用户研究，未给出具体性能指标。

**⚠️ 局限性**

局限性：
- 未覆盖信息对象类型与运动特性等维度（因难以统一度量）；
- 设计空间为描述性而非规范性，缺乏实证评估与可控实验验证；
- 仅聚焦大幅度数值，未系统探讨对小值或多维度数据的适用性；
- 仅讨论视觉与运动通道，忽略了音频、触觉等多感官融合的可能性；
- 目前示例多为静态或单用户，缺乏多用户协作、情境化应用的研究。

---

## 258. YOLOv10-Based Multi-Task Framework for Hand Localization and Laterality Classification in Surgical Videos

**arXiv ID:** 2602.18959 | [PDF](https://arxiv.org/pdf/2602.18959v1)

**作者:** Kedi Sun `[一作]` (University of Birmingham), Le Zhang `[通讯]` (University of Birmingham)

**通讯引用:** 6397 | [OpenAlex ID](https://openalex.org/A5100350651)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了基于YOLOv10的多任务框架，实时定位手部并区分左右手；

**💡 创新点**

将YOLOv10的输出空间扩展为左右手两类，并结合多任务学习与丰富的数据增强，提升了在复杂手术场景下的鲁棒性；

**🔧 技术方法**

使用YOLOv10一阶段检测器、随机翻转/缩放/旋转/颜色抖动等增强技术、SGD + cosine annealing、dropout、early stopping等训练策略；

**📊 数据集**

利用Trauma THOMPSON Challenge 2025 Task 2数据集（第一人称手术视频，手部框注解）；

**📈 对比分析**

与YOLOv8和DETR进行对比，YOLOv10在mAP_[0.5:0.95]达到0.33，FPS约38，左右手识别准确率分别为67%和71%，YOLOv8略高但速度慢，DETR精度可比但无法实时；

**⚠️ 局限性**

存在手部与背景混淆导致误检、数据集规模有限导致轻度过拟合、缺乏时间建模与更细致的背景区分等局限。

---

## 259. Luna-2: Scalable Single-Token Evaluation with Small Language Models

**arXiv ID:** 2602.18583 | [PDF](https://arxiv.org/pdf/2602.18583v1)

**作者:** Vatsal Goel `[一作]` (Galileo AI), Yash Sheth `[通讯]` (Galileo AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Luna-2，一种将小型语言模型（SLM）转化为单词级评估器，支持实时低成本且高准确率的安全评估；

**💡 创新点**

通过在共享decoder-only SLM backbone 上加入LoRA/PEFT适配器，实现多任务单前向推理并给出概率分数，显著降低成本与延迟；

**🔧 技术方法**

使用decoder-only SLM（LLaMA、Mistral、Qwen）+ LoRA适配器 + 单词级标注 + 条件softmax（只对目标类 token 计算概率）等技术；

**📊 数据集**

利用人工标注的生产数据、合成数据与LLM一致性标注，覆盖Prompt Injection、Context Adherence、Tool Selection、Tone、PII等多种评估任务；

**📈 对比分析**

与GPT-4.1 ChainPoll、Azure Content Safety、直接单词推理等方法比较，F1 0.95–0.99，成本下降 80×，延迟 20×；

**⚠️ 局限性**

局限在于需依赖专门标注数据、对多任务共享的精度略低、长文本注意力成本随长度线性或更快增长，需要工程优化；

---

## 260. Why Agent Caching Fails and How to Fix It: Structured Intent Canonicalization with Few-Shot Learning

**arXiv ID:** 2602.18922 | [PDF](https://arxiv.org/pdf/2602.18922v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]`, Abhinaba Basu

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出基于W5H2结构化意图规范化与SetFit少样本对比学习的个人AI代理缓存框架，并构建了包含63种语言的NyayaBench v2数据集，用五层级缓存架构评估其在四个公开基准及新数据集上的效果。

**💡 创新点**

创新点包括：将代理缓存视为规范化问题，用V‑measure分解评估缓存质量；设计W5H2（What‑Where）分解方案以实现语言无关且安全的缓存键；利用SetFit实现低成本高精度的少样本缓存；通过信息理论的速率-失真分析解释模型压缩；提供跨语言零样本迁移基准与风险控制阈值选取方法。

**🔧 技术方法**

技术手段涵盖：MiniLM‑based SetFit对比学习、语义聚类与V‑measure、温度缩放与置信度校准、风险控制可选预测（RCPS）、五层级缓存流水线（指纹→BERT→SetFit→廉价LLM→深度代理）、跨语言零样本推断与信息理论速率-失真框架。

**📊 数据集**

使用的数据集为：MASSIVE（1,102条，8类）、BANKING77（3,080条，77类）、CLINC150（4,500条，150类）以及自研的NyayaBench v2（8,514条，528细意图，20 W5H2超类，63种语言）。

**📈 对比分析**

与现有方法对比：相较于GPTCache、APC、20B LLM以及全监督BERT，SetFit在MASSIVE上取得91.1%准确率、CLINC150 85.9%、BANKING77 77.9%、NyayaBench v2 55.3%（16-shot提升至62.6%）。V‑measure显示SetFit在所有基准上均优于GPTCache；成本模型表明在50请求/日场景下可实现97.5%的API费用下降。

**⚠️ 局限性**

局限性包括：NyayaBench v2标签由单一标注者完成，缺乏多标注者一致性；跨语言迁移仍依赖英文训练样本，非英语性能不均；少样本学习对示例选择敏感，表现波动；未在真实代理执行链中评估缓存安全性；五层级架构的流量分布假设未经生产验证；与全监督BERT的性能差距仍存在。

---

## 261. HillInfer: Efficient Long-Context LLM Inference on the Edge with Hierarchical KV Eviction using SmartSSD

**arXiv ID:** 2602.18750 | [PDF](https://arxiv.org/pdf/2602.18750v1)

**作者:** He Sun `[一作]` (University of Science and Technology of China), Mingjun Xiao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 1780 | [OpenAlex ID](https://openalex.org/A5101838604)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HillInfer框架，实现了在边缘设备上利用SmartSSD进行长上下文LLM推理的高效KV缓存管理与推理加速。

**💡 创新点**

创新点在于结合SmartSSD的近数据处理能力，提出层级KV缓存管理与自适应预取流水线，显著减少KV缓存移动与评估延迟。

**🔧 技术方法**

采用了SmartSSD中的FPGA实现关键字重要性评估、CPU/GPU与SmartSSD协同的双向KV缓存池、以及自适应预取（APP）调度。

**📊 数据集**

使用LLaMA、Qwen、OPT等模型，并在LongBench、PG-19等长上下文基准以及LM-evaluation-harness的少样本任务集上进行评估。

**📈 对比分析**

相较于Full Cache、H2O-like、Prefetch-based、LeoAM-like等四个基线，HillInfer在多模型、多上下文长度下实现了4.21×至8.56×的速度提升，且准确率与基线持平。

**⚠️ 局限性**

局限性包括仅针对Samsung SmartSSD实现，未验证其它CSD平台；只在x86 GPU PC上测试，缺乏对ARM/Jetson等边缘平台的支持；对推理之外的复杂推理工作负载尚未探索。

---

## 262. Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning

**arXiv ID:** 2602.19917 | [PDF](https://arxiv.org/pdf/2602.19917v1)

**作者:** Thanh Nguyen `[一作]` (Korea Advanced Institute of Science and Technology), Chang D. Yoo `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于不确定性感知的Rank-One MIMO Q网络框架，用于提升离线强化学习性能。

**💡 创新点**

①使用Rank‑One MIMO网络实现近单网络成本的集成，支持高效不确定性量化；②通过最大化Q值的下置信界并加入熵奖励与数据似然正则化，精准刻画OOD数据并实现惰性策略改进；③仅需一个超参数K即可调节模型的悲观程度，避免传统方法的多网络与采样开销。

**🔧 技术方法**

离线RL、Actor‑Critic、Q‑ensemble、Rank‑One MIMO网络、下置信界估计、熵奖励、似然正则化、惰性策略改进、向量化前向传播等技术。

**📊 数据集**

D4RL基准数据集：HalfCheetah、Hopper、Walker2d，分别采用random-v2、medium-v2、medium-replay-v2、medium-expert-v2、expert-v2五种数据集。

**📈 对比分析**

与BCQ、IQL、BEAR、UWAC、CQL、MOPO、TD3‑BC、EDAC、PBRL等先进算法进行对比，平均得分在所有数据集上均超过或接近PBRL，最高分突出；在跑时与显存方面比CQL快1.8倍、比PBRL快5.9倍，显存占用最小。

**⚠️ 局限性**

对K的选择仍需经验搜索；在极端OOD或小样本场景下可能需要进一步调优；实验集中在连续控制任务，缺乏对更高维或离散任务的验证；对熵与似然项的设置在不同环境中可能产生不稳定性。

---

## 263. Expanding the Role of Diffusion Models for Robust Classifier Training

**arXiv ID:** 2602.19931 | [PDF](https://arxiv.org/pdf/2602.19931v1)

**作者:** Pin-Han Huang `[一作]` (National Taiwan University), Hsuan-Tien Lin `[通讯]` (National Taiwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在对抗训练中，将扩散模型的中间表示作为辅助学习信号，提升鲁棒分类器性能。

**💡 创新点**

首次将扩散模型的特征对齐方法（DRA）与对抗训练结合，证明其能提供部分鲁棒且多样化的特征，并与合成数据互补。

**🔧 技术方法**

采用PGD+TRADES的对抗训练、冻结扩散模型提取特征、余弦相似度对齐、稀疏自编码器评估等技术。

**📊 数据集**

在CIFAR‑10、CIFAR‑100和ImageNet上实验，使用EDM、LightningDiT等扩散模型生成的合成图像。

**📈 对比分析**

通过RobustBench（AutoAttack）评估，DRA+DM‑AT在清洁准确率与鲁棒准确率均超越基线，鲁棒性提升显著。

**⚠️ 局限性**

依赖大量合成数据和冻结扩散模型，噪声训练单独的判别器效果不佳，说明生成目标对特征质量关键；同时对模型推理无额外开销，但训练成本仍高。

---

## 264. JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation

**arXiv ID:** 2602.19163 | [PDF](https://arxiv.org/pdf/2602.19163v1)

**作者:** Kai Liu `[一作]` (Zhejiang University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60708 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种高效统一的音视频联合生成框架，在文本提示下同步生成高质量音频与视频。

**💡 创新点**

创新点包括：模态专属Mixture-of-Experts（MS-MoE）提升单模态质量；时序对齐RoPE（TA‑RoPE）实现帧级同步；音视频直接偏好优化（AV‑DPO）让模型更贴合人类审美。

**🔧 技术方法**

采用流匹配与Diffusion、DiT架构、MS‑MoE、TA‑RoPE、AV‑DPO、RoPE、reward模型（AudioBox、VideoAlign、Syncformer等）以及多模态注意力。

**📊 数据集**

训练使用公开数据：780K音频‑文本对、360K音频‑视频‑文本三元组（约1M条），并在JavisBench和JavisBench‑mini上评估。

**📈 对比分析**

与JavisDiT、UniVerse‑1等开源方法对比，模型在FVD、FAD、JavisScore、DeSync等指标上均显著提升（如JavisScore↑、DeSync↓），性能逼近甚至超过商用模型Ve3；推理时延与原Wan2.1保持一致，仅增加1.6%。

**⚠️ 局限性**

局限性在于：仍无法完全匹配Ve3的细腻度；训练成本高，需大规模计算；生成时长、分辨率有限（2–5s，240p–480p）。

---

## 265. Vectorized Bayesian Inference for Latent Dirichlet-Tree Allocation

**arXiv ID:** 2602.18795 | [PDF](https://arxiv.org/pdf/2602.18795v1)

**作者:** Zheng Wang `[一作]` (Concordia University), Nizar Bouguila `[通讯]` (Concordia University)

**通讯引用:** 9584 | [OpenAlex ID](https://openalex.org/A5090600716)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Latent Dirichlet-Tree Allocation（LDTA）模型，将LDA的Dirichlet先验替换为任意Dirichlet-Tree分布，以支持更丰富的主题层次和相关性结构。

**💡 创新点**

创新点在于：① 推导并统一了Dirichlet-Tree的三种等价表示（节点、通用、指数形式）并证明其与多项式似然的共轭性；② 设计了两种可扩展、向量化的推理算法——均值场变分推理（MFVI）和期望传播（EP），并在GPU上实现高效并行计算；③ 引入了Bayesian、Dirichlet选择和Derived分布等概念，简化推理推导。

**🔧 技术方法**

采用了Dirichlet-Tree分布、指数族推理、变分推理、期望传播、矩匹配、Newton方法、PyTorch张量运算以及GPU加速等技术。

**📊 数据集**

在NIPS、Reuters-21578、20 Newsgroups、15 Scene Categories、PBMC单细胞RNA-seq等公开数据集上进行实验，覆盖文本、图像和生物信息学应用。

**📈 对比分析**

通过对ELBO收敛、预测困惑度、主题一致性/多样性、文档/图像分类准确率等指标进行比较。实验表明：MFVI在ELBO和主题一致性上略优，EP在收敛速度和主题多样性上更突出；在分类任务中，LDTA往往比传统LDA取得更高或相近的准确率。

**⚠️ 局限性**

局限性包括：树结构需手工指定，缺乏自动学习结构的机制；推理算法依赖KL散度，可能不适用于所有场景；目前未考虑非参数化的树拓扑或更灵活的近似分布。

---

## 266. Asymptotic Semantic Collapse in Hierarchical Optimization

**arXiv ID:** 2602.18450 | [PDF](https://arxiv.org/pdf/2602.18450v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Bugra Kilictas `[通讯]` (Bahcesehir University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多代理自然语言系统中，使用Riemannian流形优化构建了一个层级优化框架，证明了语义坍塌的渐近收敛性、路径无关性以及熵的不可逆消失。

**💡 创新点**

首次将智能合约概念与语义一致化结合，提出了“不可变共识协议”并用信息理论证明了从原子向量到合同张量的不可逆压缩，揭示了主锚节点导致的全局语义统一。

**🔧 技术方法**

运用了Riemannian流形投影、梯度流与随机梯度下降、凸分析与信息熵理论，并在RWKV-7 13B GGUF模型上进行实验验证。

**📊 数据集**

采用无监督的“dataset‑free”基准，仅使用RWKV-7 13B GGUF检查点进行实验，不引入外部数据集。

**📈 对比分析**

通过对比贪婪与随机（top‑p）解码，测量下一词熵、合规度、Jaccard相似度与哈希碰撞率；结果显示熵显著下降、随机解码略高的合规度、贪婪解码更接近锚点，碰撞率为零。

**⚠️ 局限性**

假设单一静态主锚、凸损失与单层星形层级，忽略了动态或多主锚情况，模型过于理想化，未能覆盖真实多代理交互的复杂性。

---

## 267. In Defense of Cosine Similarity: Normalization Eliminates the Gauge Freedom

**arXiv ID:** 2602.19393 | [PDF](https://arxiv.org/pdf/2602.19393v1)

**作者:** Taha Bouhsine `[一作]` `[通讯]` (Azetta), Taha Bouhsine (Azetta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

阐述了在单位球面上余弦相似度与欧氏距离的等价性，并指出通过归一化消除矩阵分解模型中的D-自由度问题。

**💡 创新点**

提出了“对齐原则”，即训练目标与度量之间的对称性一致性，并强调在球面约束下余弦相似度是良好定义的。

**🔧 技术方法**

主要使用了几何推导、对称性分析、Riemannian 视角等理论工具；未使用实验技术。

**📊 数据集**

无实验数据集，理论分析为主。

**📈 对比分析**

无对比实验，讨论了与Steck等人建议的训练方法相一致的结果；未给出性能数值。

**⚠️ 局限性**

局限在于未给出具体实现细节或大规模实验验证，理论结论可能不涵盖所有深度学习模型；需要进一步验证其在实际推荐系统中的效果。

---

## 268. DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via Frequency-Aware Filtering

**arXiv ID:** 2602.19323 | [PDF](https://arxiv.org/pdf/2602.19323v1)

**作者:** Yiran Qiao `[一作]` (Case Western Reserve University), Jing Ma `[通讯]` (Case Western Reserve University)

**通讯引用:** 4824 | [OpenAlex ID](https://openalex.org/A5034823980)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了DefenseSplat频率感知防御方案，通过离散小波变换过滤高频噪声并利用3D Gaussian Splatting的自适应渲染过程抑制对抗攻击。

**💡 创新点**

创新点在于首次将小波频率分析与3DGS结合，提出仅保留低频子带重建图像并辅以ReLU尺度正则化抑制伪纹理，显著提升鲁棒性且不需对抗训练。

**🔧 技术方法**

采用离散小波变换、深度图像匹配、旅行商问题（TSP）优化视角顺序、3D Gaussian Splatting渲染、尺度正则化损失等技术。

**📊 数据集**

在Mip-NeRF 360、Tanks-and-Temples和LLFF三个公开3D重建基准数据集上进行实验。

**📈 对比分析**

与原始3DGS、Difix3D+、CompactGS三种基线对比，在训练时间、Gaussians数、显存占用、渲染速度等鲁棒性指标以及PSNR、SSIM、LPIPS等重建质量指标上均实现更佳表现，且对不同攻击强度保持稳定。

**⚠️ 局限性**

实验仅针对已知的单一对抗攻击方法，缺乏对新型或更强攻击的泛化评估；在极端攻击下低频信息过滤可能略微影响细节恢复。

---

## 269. BarrierSteer: LLM Safety via Learning Barrier Steering

**arXiv ID:** 2602.20102 | [PDF](https://arxiv.org/pdf/2602.20102v1)

**作者:** Thanh Q. Tran `[一作]` (National University of Singapore), Wei Xiao `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于控制障碍函数（CBF）的推理时安全约束框架（BarrierSteer），将非线性安全边界直接嵌入大型语言模型（LLM）的潜在表示空间，实现安全轨迹的实时引导。

**💡 创新点**

创新点在于：①利用CBF将安全约束转化为可闭式求解的控制问题，避免昂贵的在线优化；②通过学习非线性安全边界而非线性多面体，提升安全判别的表达力；③提供多种约束合成策略（QP、Top‑2、LSE），兼顾安全性与推理效率。

**🔧 技术方法**

核心技术包括：控制障碍函数（CBF）与约束合成；潜在表示抽取与对比学习；强化学习/决策过程中的安全约束学习；多头梯度下降/二次规划求解。

**📊 数据集**

使用 HarmBench（包含 320+80 的攻击样本）和 a 14 类大规模有害提示数据集（每类 400 样本）来构建安全/不安全激活样本集，评估模型在多类攻击和真实有害提示上的表现。

**📈 对比分析**

与原始 LLM、Activation Addition、Directional Ablation 以及 SaP 等基线对比。BarrierSteer 在 9 种攻击中的 ASR 均显著降低（多模型 0–1% 左右），同时保持 MMLU/GSM8K 等通用指标与原始模型相近；在多约束合成实验中，LSE/QP 方案将不安全生成率降至 1.82%，远优于单个 CBF 或 Top‑2 组合。

**⚠️ 局限性**

局限性包括：①依赖高质量安全标签，数据偏差会导致边界失效；②安全保证仅在潜在空间，对语义安全的直接映射有限；③采用简化的潜在动态模型，理论假设可能与真实 Transformer 行为不完全一致；④仍存在安全‑效能权衡，强制引导会轻微降低模型表现。

---

## 270. Exploiting Label-Independent Regularization from Spatial Dependencies for Whole Slide Image Analysis

**arXiv ID:** 2602.19487 | [PDF](https://arxiv.org/pdf/2602.19487v1)

**作者:** Weiyi Wu `[一作]` (Dartmouth), Jiang Gui `[通讯]` (Dartmouth)

**通讯引用:** 5403 | [OpenAlex ID](https://openalex.org/A5008965974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于空间正则化的多实例学习框架（SRMIL），利用图注意网络同时学习标签引导的分类和标签无关的空间重构，从而在病理全切片图像（WSI）分析中实现更稳健的特征学习。

**💡 创新点**

创新点在于：①引入标签无关的空间重构任务作为正则化，利用WSI中天然的空间结构而非噪声较大的注意力或标签信息；②采用双流学习架构，将重构与分类任务共享编码器，实现两者协同优化；③在重构中使用随机遮挡70%的节点，保证监督信号均匀分布，提升特征泛化。

**🔧 技术方法**

技术方法包括图注意网络（GAT）构建WSI图结构，随机遮挡与自监督重构损失（余弦距离），滑动图分类损失，辅助腐败图分类损失；联合损失权重训练。

**📊 数据集**

使用公开的三大WSI分类数据集：CAMELYON-16（二分类肿瘤检测）、TCGA-Lung（肿瘤亚型分类）和BRACS（组织分级）。两种特征提取器（ResNet50与ViT）用于实验。

**📈 对比分析**

与多种先进MIL方法（ABMIL、CLAM、DSMIL、DTFD-MIL、MHIM-MIL、TransMIL、MambaMIL、PatchGCN等）在相同数据集与特征提取器上进行比较。SRMIL在所有任务上均达到或超过对手最佳成绩，最高AUC提升约3-4%，在CAMELYON-16上准确率提升至91.2%。

**⚠️ 局限性**

局限性包括：①对遮挡比例与图结构构造的依赖，可能对不同尺寸或分辨率的WSI需要重新调参；②训练过程中仍需较大计算资源（图注意网络与重构任务并行）；③对极度稀疏或高度不平衡的实例分布仍可能产生误差，未来需进一步改进自监督掩码策略和多尺度融合。

---

## 271. Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision

**arXiv ID:** 2602.19715 | [PDF](https://arxiv.org/pdf/2602.19715v1)

**作者:** Kartik Kuckreja `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Abhinav Dhall `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DeepfakeJudge框架，集成了OOD检测、视觉-文本理由监督与评估。

**💡 创新点**

创新点在于构建人类标注的视觉理由数据、使用生成‑评估循环实现可扩展理由评分，并训练多模态评判器。

**🔧 技术方法**

采用大规模视觉语言模型、生成器‑评估循环、数据增强、点对点与配对评分技术。

**📊 数据集**

使用自构造的OOD DeepfakeJudge‑Detect/Reason 数据集、内部人工标注的 1500 张图像，以及公开的 Open‑Images、FaceForensics++ 等。

**📈 对比分析**

与 15 种闭源、开源及理由增强模型对比，DeepfakeJudge‑7B 在点对点 RMSE 0.50、配对准确率 98.9%，显著超越所有对手。

**⚠️ 局限性**

局限在于仍需人工标注，规模有限；评判器对极端伪造细粒度判断不足；跨语言和多模态通用性待进一步验证。

---

## 272. Temporal-Aware Heterogeneous Graph Reasoning with Multi-View Fusion for Temporal Question Answering

**arXiv ID:** 2602.19569 | [PDF](https://arxiv.org/pdf/2602.19569v1)

**作者:** Wuzhenghong Wen `[一作]` (Nanjing University of Posts and Telecommunications), Jianting Liu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种时序感知的异构图推理框架，包含约束感知问题编码、多跳时序图神经网络和多视角自适应融合模块。

**💡 创新点**

创新点在于将问题表征与图中时间动态关联、通过时间感知消息传递实现显式多跳推理、以及使用多视角注意力实现语言与图知识的高效融合。

**🔧 技术方法**

采用预训练语言模型（如BERT/Roberta）、TComplEx时序嵌入与位置编码、时间感知GNN（带路径注意力与扩散算子）以及多视角注意力融合机制。

**📊 数据集**

在公开时序问答基准CRONQUESTIONS和Time-Questions上进行实验。

**📈 对比分析**

与BERT、KnowBERT、EmbedKGQA、CronKGQA、TMA、TSQA、TempoQR、CTRN等基线相比，CRONQUESTIONS上Hits@1从0.920提升至0.969，Time-Questions上从0.466提升至0.539，表现出显著性能提升。

**⚠️ 局限性**

主要局限包括对时间位置编码的依赖、推理过程中计算开销较大，以及在极长链多跳场景下性能仍可能下降。

---

## 273. Unlocking Multimodal Document Intelligence: From Current Triumphs to Future Frontiers of Visual Document Retrieval

**arXiv ID:** 2602.19961 | [PDF](https://arxiv.org/pdf/2602.19961v1)

**作者:** Yibo Yan `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文系统综述了多模态文档检索（VDR）在大语言模型（MLLM）时代的发展，涵盖基准、嵌入模型、重排序模型、RAG管线与智能体系统，并梳理未来挑战与研究方向。

**💡 创新点**

创新之处在于首次提供面向MLLM时代的VDR全景性综述，提出按嵌入、重排序和RAG/智能体三大范式分类，并强调多向量嵌入与思考-再嵌入的研究热点。

**🔧 技术方法**

综述所涉及技术包括多模态嵌入（Late‑Interaction、Multi‑Vector）、对比学习、预训练M‑LLM、RAG与Agentic框架、以及跨语言与多任务训练策略。

**📊 数据集**

评述使用了多种VDR基准数据集，如ViDoRe‑V1/V2/V3、Jina‑VDR、NL‑DIR、Real‑MM‑RAG、SeaDoc、MMDocIR、M4DocBench等，并引用通用多模态评测MMEB等。

**📈 对比分析**

通过对比分析，发现ColPali/ColQwen等多向量嵌入模型在Recall@k与nDCG上领先，UniME‑V2‑Reranker等重排序模型在列表级别提升显著，但在存储与推理速度上仍存在性能-效率权衡。

**⚠️ 局限性**

局限性包括快速演进的领域导致部分最新工作未覆盖，综述偏重宏观缺乏细节深度，且聚焦VDR而非完整视觉文档理解体系，数据偏向英文及单页文档，可能影响跨语言与多文档推理的评估。

---

## 274. Pyramid MoA: A Probabilistic Framework for Cost-Optimized Anytime Inference

**arXiv ID:** 2602.19509 | [PDF](https://arxiv.org/pdf/2602.19509v1)

**作者:** Arindam Khaled `[一作]` `[通讯]` (Independent Researcher), Arindam Khaled (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Pyramid MoA框架，利用轻量级路由器动态决定是否将查询从小模型层升级到大模型层；

**💡 创新点**

通过决策理论的价值计算（VoC）和基于语义一致性与输出方差的概率失败预测，实现可调节的“随时算法”特性，显著降低推理成本；

**🔧 技术方法**

使用多模型集成（Llama‑3‑8B、Mistral‑7B、Qwen‑2.5‑7B）作为底层、随机森林或XGBoost作为路由器、语义一致性/方差特征、以及Oracle层（Llama‑3‑70B）；

**📊 数据集**

MBPP、HumanEval（代码生成）以及GSM8K（数学推理）数据集；

**📈 对比分析**

与Oracle单模型以及FrugalGPT级联基线对比；在GSM8K上实现93.0%准确率（接近Oracle 98%），计算成本降低61%，在代码生成上与70B模型保持相同准确率，推理成本降低40%；

**⚠️ 局限性**

依赖Oracle的预先标注进行路由器训练，可能在新领域泛化受限；对极端复杂查询的识别仍需进一步优化；路由器的训练和维护成本需评估；

---

## 275. Generative 6D Pose Estimation via Conditional Flow Matching

**arXiv ID:** 2602.19719 | [PDF](https://arxiv.org/pdf/2602.19719v1)

**作者:** Amir Hamza `[一作]` (Fondazione Bruno Kessler), Fabio Poiesi `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于条件流匹配（Conditional Flow Matching）的实例级6D姿态估计方法（FLOSE），能够从RGBD图像恢复目标物体的位姿。

**💡 创新点**

创新点在于将语义特征与重叠感知特征融合到条件流匹配的条件向量中，以消除对称物体的姿态歧义，并用RANSAC而非传统SVD对流匹配产生的噪声离群点进行鲁棒配准。

**🔧 技术方法**

使用的核心技术包括：条件流匹配框架、点云特征编码（PointTransformerV3 + DINOv2）、RANSAC + ICP精细配准、以及多步Euler积分实现的迭代去噪。

**📊 数据集**

在BOP基准的五个数据集（如LM-O、YCB-V、YCB-Video、ICL-NUIM、OCID等）上进行实验验证。

**📈 对比分析**

与10个同行方法（包括单对象和单数据集训练的最先进方法）对比，单模型训练下平均提升+4.5 AR，单对象训练下+1.2 AR，尤其在对称物体上的表现显著优于对手。

**⚠️ 局限性**

主要局限在于：需要两阶段训练流程；流匹配迭代过程耗时较长，限制实时应用；依赖目标物体的分割结果，尚未直接支持全场景级姿态估计。

---

## 276. MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving

**arXiv ID:** 2602.20060 | [PDF](https://arxiv.org/pdf/2602.20060v1)

**作者:** Junli Wang `[一作]` (Chinese Academy of Sciences), Qichao Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了MeanFuser，一种端到端的自驾规划框架，利用高效的生成式方法一次性采样多模态轨迹，并通过自适应重构模块提升轨迹质量。

**💡 创新点**

核心创新包括：① 用高斯混合噪声（GMN）取代离散锚词，实现连续轨迹空间的多模态覆盖；② 引入MeanFlow标识，直接学习平均速度场，消除ODE求解误差并实现一次采样；③ 设计轻量化自适应重构模块（ARM），在所有采样方案均不理想时自动重构轨迹。

**🔧 技术方法**

技术实现基于流匹配的MeanFlow模型、Transformer编码器-解码器、GMN采样、注意力融合与投影层，并采用ResNet-34视觉骨干与多任务辅助（检测、映射）。

**📊 数据集**

实验使用NAVSIMv1和NAVSIMv2闭环仿真数据集，采用RGB图像输入，评估指标为PDM Score与Extended PDM Score。

**📈 对比分析**

与GoalFlow、DiffusionDrive、Hydra-MDP等基线相比，MeanFuser在NAVSIMv1达89.0 PDMS、NAVSIMv2达89.5 EPDMS，速度提升至59 FPS（比GoalFlow快约5.2×、DiffusionDrive快约2.6×），并在多模态覆盖率、碰撞率等子指标上均优于对手。

**⚠️ 局限性**

限制：仅在RGB输入下验证；对更复杂多传感器场景的鲁棒性尚未充分验证；GMN参数固定，进一步优化混合模型可能提升性能。

---

## 277. Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception

**arXiv ID:** 2602.19596 | [PDF](https://arxiv.org/pdf/2602.19596v1)

**作者:** Yihang Tao `[一作]` (City University of Hong Kong), Yuguang Fang `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种针对协同感知（CP）系统的自适应对抗攻击框架，利用 Mutual View Information Graph（MVIG）学习共享信息中的脆弱性，生成动态风险图并优化攻击时机、位置与持续性。

**💡 创新点**

创新点包括：① 将不同防御系统暴露的漏洞统一映射为 MVIG 表示，捕捉互视信息不对称；② 通过时间图学习（GRU + GNN）生成随时间演化的制造风险图；③ 采用熵感知的漏洞搜索和多目标损失实现对攻击位置、时机和持续性的自适应优化；④ 实现跨防御配置的通用攻击。

**🔧 技术方法**

所用技术：图神经网络（MVIGNet）、时序 GRU、PGD 生成特征扰动、熵感知漏洞搜索、两阶段离散优化、实时推理与三维映射、数据映射到 BEV 空间。

**📊 数据集**

使用的数据集：OPV2V（原始多车协同感知数据）以及其扩展攻击基准 Adv-OPV2V，用于训练 MVIG 并在攻击评估中测试。

**📈 对比分析**

与 Basic Feature Attack、BAC Attack、RC Attack 等对抗方法以及 ROBOSAC、CP-Guard、GCP、CAD 等防御系统进行对比。实验表明，在 Adv-OPV2V 上，该攻击在 CAD 防御下 DSR 降至 0.32，攻击成功率高；在 3‑frame 持续攻击下 DSR 仅为 0.63；实现帧率约 29.9 FPS，满足实时要求。

**⚠️ 局限性**

局限性：① 去除攻击（removal）受限于已存在目标的优化空间；② 效果受参与车辆视角覆盖率限制；③ 当前仅针对 LiDAR 传感器，尚未扩展到摄像头；④ 对抗鲁棒性需进一步评估。

---

## 278. Echoes of Ownership: Adversarial-Guided Dual Injection for Copyright Protection in MLLMs

**arXiv ID:** 2602.18845 | [PDF](https://arxiv.org/pdf/2602.18845v1)

**作者:** Chengwei Xia `[一作]` (Lanzhou University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 81083 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于触发图像的黑盒版权追踪框架 AGDI，能够在模型微调后仍准确识别衍生模型。

**💡 创新点**

创新点在于：① 双注入机制（响应级一致性 + CLIP 语义对齐），利用 MLLM 中稳定的 CLIP‑like 子模型实现跨模型泛化；② 通过对抗训练辅助模型模拟微调变化，提升鲁棒性；③ 以图像作为可学习张量，在黑盒查询下无需模型内部信息即可生成可验证所有权的触发图像。

**🔧 技术方法**

技术手段包括：PGD 对抗优化、梯度裁剪、双目标损失（交叉熵 + CLIP 嵌入距离）、对抗训练（min‑max 更新辅助模型）、参数剪枝/量化/模型融合测试、ASR 评估。

**📊 数据集**

数据集：200 张 ImageNet‑1K 验证集清洁图像；5 组触发问答对（共 1000 条触发查询）；微调数据集包括 V7W、ST‑VQA、TextVQA、PaintingForm、MathV360k；基模型为 LLaVA‑1.5 与 Qwen2‑VL。

**📈 对比分析**

与 Ordinary、RNA、PLA、IF 等基线在多种微调策略（LoRA、Full‑Fine‑Tune）以及剪枝、量化、模型融合、模型融合后续测试中对比。AGDI 在所有设置下 ASR 明显提升，尤其在模型剪枝、量化和模型融合场景下保持较高的追踪成功率。

**⚠️ 局限性**

局限性：① 触发图像需要在视觉上保持隐蔽，过度优化可能影响图像质量；② 对极大规模微调或显著改动 CLIP‑like 子模块时可能失效；③ 对极端白盒攻击尚未充分评估；④ 依赖特定的触发问答对，若攻击者改写触发内容可能导致失效。

---

## 279. Analyzing and Leveraging the $k$-Sensitivity of LZ77

**arXiv ID:** 2602.19649 | [PDF](https://arxiv.org/pdf/2602.19649v1)

**作者:** Gabriel Bathie `[一作]` (University of Bordeaux), Akka Zemmari `[通讯]` (University of Bordeaux)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究了Lempel-Ziv 77压缩算法对编辑的敏感性，展示了如何通过修改字符串w来改善或恶化其压缩效果。

**💡 创新点**

提出了一个紧密的上界，表明在k次编辑下，压缩大小的变化与原始压缩大小之间的关系，并且发现了基于可压缩性的三分法。

**🔧 技术方法**

使用了Lempel-Ziv 77压缩算法，并提出了一种ε-近似算法来预编辑字符串以改善压缩效果。

**📊 数据集**

未具体提及使用的数据集，但研究涉及的字符串和压缩算法在理论上具有广泛的应用。

**📈 对比分析**

通过与Lempel-Ziv 78算法的对比，展示了LZ77在多次编辑下的压缩性能变化，表明LZ77在k次编辑下的压缩性能变化不超过3倍，并且提供了相应的上界和下界。

**⚠️ 局限性**

限制在于当前的研究主要集中在LZ77算法上，未来需要扩展到其他压缩算法以验证敏感性是否可以被利用来改善压缩效果。

---

## 280. Large Causal Models for Temporal Causal Discovery

**arXiv ID:** 2602.18662 | [PDF](https://arxiv.org/pdf/2602.18662v1)

**作者:** Nikolaos Kougioulis `[一作]` (Institute of Applied and Computational Mathematics, FORTH), Ioannis Tsamardinos `[通讯]` (Computer Science Department, University of Crete)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种大型因果模型（Large Causal Models，LCMs），通过在多源时间序列数据上预训练Transformer结构，实现一次前向推理即可完成跨数据集的时间序列因果图推断。

**💡 创新点**

创新点包括：①将因果发现视作大规模预训练任务，构建真正的基础模型；②构造包含合成、半合成与真实数据的多样化训练语料，显著提升零样本与OOD泛化；③在Transformer编码器中加入卷积嵌入、位置编码和相关性正则化等训练辅助，增强模型对时间滞后关系的辨识能力；④展示模型可扩展到12维以上、长时滞，且推理速度远快于传统搜索式方法。

**🔧 技术方法**

核心技术包括：Transformer编码器堆叠、Conv1D嵌入、位置编码、交叉熵与相关性正则化损失、混合数据训练、单步前向推断、以及可选的注意力蒸馏层。

**📊 数据集**

数据集：230k 合成TSCM实例；45k 真实行业时间序列（能源、气象、交通等）；半合成 fMRI 与 Kuramoto 系统；通过多比例混合（100/0、80/20、50/50、20/80）构建训练与测试集，总计275k实例。

**📈 对比分析**

与 PCMCI、DYNOTEARS、VARLinGAM、以及预训练Transformer等基线对比。LCMs 在 AUC 上与或优于传统方法，特别是在 OOD 与零样本场景下表现突出；同时单次前向推理使运行时间显著降低，几乎与输入维度无关。

**⚠️ 局限性**

局限性：模型假设为无潜在混杂、无同滞因果、加性噪声、固定最大滞后；若假设被违反，模型输出可能偏离真实因果结构，应结合领域知识进行解读。

---

## 281. NEXUS : A compact neural architecture for high-resolution spatiotemporal air quality forecasting in Delhi Nationa Capital Region

**arXiv ID:** 2602.19654 | [PDF](https://arxiv.org/pdf/2602.19654v1)

**作者:** Rampunit Kumar `[一作]` (Independent Researcher), Aditya Maheshwari `[通讯]` (Indian Institute of Management Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种紧凑的神经网络架构，用于高分辨率的时空空气质量预测，重点解决德里国家首都区的三种主要污染物（CO、NO、SO₂）的未来浓度；

**💡 创新点**

创新点在于将补丁嵌入、低秩投影与自适应融合（NanoBlock）相结合，显著减少参数（比FEDformer低94%）并提升R²（比基准提升约6.95%）；

**🔧 技术方法**

使用了Patch Embedding压缩序列、低秩线性投影提取主要模式、三路并行NanoBlock（CompactKernel、MicroConv、FusionGate）进行多尺度特征提取、加权空间池化与轻量化输出层，训练采用Adam+学习率衰减与早停；

**📊 数据集**

利用2018–2021年四年的Copernicus污染观测（CO、NO、SO₂，3小时一次，四个角点）与ERA5气象（16个点，小时级）拼接后生成15,392个完整的时空样本；

**📈 对比分析**

与SCINet、Autoformer、FEDformer在相同数据集和划分下比较；最终模型在CO、NO、SO₂上的R²分别为0.940、0.914、0.952，参数仅18,748，推理速度0.8 ms/样本，显著优于对比模型；

**⚠️ 局限性**

局限在于仅针对德里四格观测区域，模型缺乏物理过程约束，可能对其他地区或极端天气不具备同等泛化性；依赖高分辨率气象输入，若气象误差或缺失将影响预测准确性。

---

## 282. Issues with Measuring Task Complexity via Random Policies in Robotic Tasks

**arXiv ID:** 2602.18856 | [PDF](https://arxiv.org/pdf/2602.18856v1)

**作者:** Reabetswe M. Nkhumise `[一作]` (University of Sheffield), Aditya Gilra `[通讯]` (Wirtschaftsuniversität)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文评估了随机权重猜测（RWG）下的两种信息论任务复杂度指标PIC和POIC，并在一系列结构相似的机器人臂到达任务（1-连杆和2-连杆、稠密与稀疏奖励）上进行实验；

**💡 创新点**

创新点在于提出了基于已知相对复杂度的任务框架来检验任务复杂度指标的可靠性，并系统揭示了PIC/POIC在某些情形下与直觉和经验RL结果相悖的现象；

**🔧 技术方法**

使用的技术包括随机权重猜测（RWG）、信息论度量PIC/POIC、统计分布可视化、Bootstrap置信区间、Welch t检验，以及Soft Actor Critic（SAC）和SAC+HER等RL算法进行基准验证；

**📊 数据集**

实验数据集为自定义的六个机器人臂到达任务，包含三种臂配置（1-连杆长度1.0m、1.65m；2-连杆长度组合0.95/0.70m）与两种奖励形式（密集、稀疏），共计10⁴条随机策略样本；

**📈 对比分析**

比较方法包括：1）统计随机策略性能分布；2）PIC/POIC指标；3）训练后的SAC学习曲线。结果显示统计分析能定性区分任务难度，但PIC/POIC在密集奖励下排序与预期和SAC学习曲线不符，而在稀疏奖励下匹配；

**⚠️ 局限性**

局限性包括：PIC/POIC高度依赖先验参数分布、搜索空间静态且未考虑训练过程中的探索；RWG对解空间稀疏的任务无效；指标未考虑状态空间访问复杂度，且缺乏可解释性。

---

## 283. Can You Tell It's AI? Human Perception of Synthetic Voices in Vishing Scenarios

**arXiv ID:** 2602.20061 | [PDF](https://arxiv.org/pdf/2602.20061v1)

**作者:** Zoha Hayat Bhatti `[一作]` (Lahore University of Management Sciences), Mobin Javed `[通讯]` (Lahore University of Management Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在实验中，研究者让 22 名参与者聆听 16 条模拟诈骗的语音片段（8 条 AI 合成，8 条真人录制），并判断其来源是否为 AI，同时记录置信度和理由。

**💡 创新点**

创新点在于将人工智能生成的语音与真实语音放入真实的诈骗情境中进行感知评估，并结合量化的 Signal Detection Theory 与质性主题分析，揭示人类对 AI 语音的误判和高置信度误差。

**🔧 技术方法**

使用了商业 TTS 平台 Play.ht 的语音合成、普通话录音收集、Web 量化问卷、信号检测理论、开放式编码主题分析等技术。

**📊 数据集**

数据集为 16 条音频样本，按 4 种诈骗脚本、2 种情绪强度（平静/紧急）以及 AI 与真人两种来源生成的平衡组合，覆盖 8 条 AI、8 条人声。

**📈 对比分析**

方法上，先计算整体识别准确率（37.5%）与偶然率（50%）的显著性，随后用 Signal Detection Theory 计算 d' ≈ 0 与轻微人类偏好；对 315 条开放式回应进行编码，提炼五类认知线索。性能方面，识别率低于偶然，表明人类无法可靠地区分两类语音。

**⚠️ 局限性**

局限包括样本量仅 22 人、音频长度短（10–15 秒）、仅使用美国男性语音、使用商业 TTS 而非最先进的克隆模型、实验环境低风险且缺乏真实诈骗压力，因而可能低估真实情境中的识别难度。

---

## 284. Athena: An Autonomous Open-Hardware Tracked Rescue Robot Platform

**arXiv ID:** 2602.19898 | [PDF](https://arxiv.org/pdf/2602.19898v1)

**作者:** Stefan Fabian `[一作]` (Technical University of Darmstadt), Oskar von Stryk `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了Athena开源救援地面机器人平台，具备四个可独立重构的翻板、强大的操纵臂以及低成本远程E‑Stop解决方案。

**💡 创新点**

创新点包括：全新的PU带式履带设计可更换履带剖面；四个独立可调翻板实现对复杂地形的高适应性；集成多模态传感器与力觉传感器的高覆盖度操纵臂；远程E‑Stop系统的低成本实现；所有CAD、PCB及低层软件均开源。

**🔧 技术方法**

使用的技术有：PU工业带与金属齿插件的履带结构；碳纤维增强塑料底盘；Unitree A1电机；Dynamixel舵机驱动翻板和操纵臂；Livox MID‑360 LiDAR、RGB‑D相机、热成像相机等多传感器融合；Beelink Mini PC + Nvidia Jetson Orin AGX进行CPU/GPU并行计算；LoRa+ESP32实现远程E‑Stop；电源管理与高侧开关控制。

**📊 数据集**

未使用公开数据集，评估采用RoboCup Rescue League现场测试和实验室环境下的步梯高度、负载、E‑Stop延迟等实验数据。

**📈 对比分析**

与Quince、Karo、Asterix等现有平台对比，Athena在步梯高度（41 cm）和操纵臂可达距离（1.54 m）与载荷（2.9 kg）上处于领先或相当水平；远程E‑Stop延迟在0.2–0.4 s之间，落在人体反应时间范围内。

**⚠️ 局限性**

局限性包括：翻板驱动轴承易在极限载荷下损坏、履带剖面耐磨性仍需改进；操纵臂负载仅适用于轻量物体；系统尚未进行完整的自治导航与任务规划验证；远程E‑Stop仅在短距离内可靠，障碍物遮挡下的可靠性有限。

---

## 285. FOCA: Frequency-Oriented Cross-Domain Forgery Detection, Localization and Explanation via Multi-Modal Large Language Model

**arXiv ID:** 2602.18880 | [PDF](https://arxiv.org/pdf/2602.18880v1)

**作者:** Zhou Liu `[一作]` (Harbin Institute of Technology), Lei Fan `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多模态大型语言模型框架FOCA，可同时利用RGB空间与频域特征进行图像伪造检测、定位并给出可解释的交叉域说明。

**💡 创新点**

创新点在于首次将高频频域信息通过交叉注意力融合引入LLM，实现语义与频域证据的联合推理，并构建了包含像素级掩码与双域注释的大规模FSE-Set数据集。

**🔧 技术方法**

采用离散小波变换提取高频子带、频域注意力融合（FAF）、对比学习、LoRA微调、多模态LLM（如LISA‑7B）与SAM视觉编码器。

**📊 数据集**

使用自建FSE‑Set（10万张真实+伪造图像，含像素掩码与双域解释），并在CASIA v1、Columbia等公开数据集上进行验证。

**📈 对比分析**

与传统检测器（CnnSpott、Fusing、Uni‑vFD、DRCT）以及其他多模态LLM（SIDA、InternVL3、Qwen2.5‑VL、DeepSeekVL2）对比，FOCA在图像检测准确率与F1、定位IoU/F1均达到或超过SOTA，整体精度约为96.2%。

**⚠️ 局限性**

局限性包括训练时主要冻结大模型，仅微调FAF与分割解码器；对复杂光照、压缩噪声等场景鲁棒性未充分评估；生成解释受LLM语言模型误差影响。

---

## 286. GPU-Resident Gaussian Process Regression Leveraging Asynchronous Tasks with HPX

**arXiv ID:** 2602.19683 | [PDF](https://arxiv.org/pdf/2602.19683v1)

**作者:** Henrik Möllmann `[一作]` (University of Stuttgart), Alexander Strack `[通讯]` (University of Stuttgart)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对GPRat库进行扩展，实现GPU驻留的GP预测管线，并利用HPX异步任务调度实现高效的GPU加速。

**💡 创新点**

采用GPU驻留的切块（tiled）算法并结合多CUDA流与HPX执行，在最佳配置下实现比cuSOLVER高达11%的加速。

**🔧 技术方法**

使用HPX任务运行时、CUDA Toolkit（cuBLAS、cuSOLVER）、行列式内存布局转换、GPU流调度及APEX性能计数等技术。

**📊 数据集**

使用自行生成的耦合质量-弹簧-阻尼模拟数据，训练样本规模从128到32768，测试样本与训练样本规模相等。

**📈 对比分析**

通过与CPU实现（48核AMD EPYC + NVIDIA A30）对比，GPU在样本≥128时显著优越；Cholesky 4.3×加速，GP预测 4.6×加速；最佳配置下GPU比cuSOLVER提升11%。

**⚠️ 局限性**

仅支持NVIDIA GPU，缺乏多GPU分布式/异构并行支持；在极大规模时CUDA流/任务调度开销趋近瓶颈；仍需进一步优化能耗与混合精度实现。

---

## 287. Adaptive Collaboration of Arena-Based Argumentative LLMs for Explainable and Contestable Legal Reasoning

**arXiv ID:** 2602.18916 | [PDF](https://arxiv.org/pdf/2602.18916v1)

**作者:** Hoang-Loc Cao `[一作]`, Hung Cao `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 ACAL 框架，通过自适应多代理协作与基于竞技场的定量双极论证（A-QBAF）实现可解释、可争议的法律推理。

**💡 创新点**

创新点在于将多代理辩论与量化论证框架相结合，并加入冲突解决（CR）与不确定性感知升级（UAE），实现结构化推理与人机交互式争议处理。

**🔧 技术方法**

采用技术包括：LLM 驱动的专家代理池、混合检索增强生成（Hybrid RAG）、Arena 折冲突解决、Quadratic Energy（QE）语义推理以及 HITL 争议工作流。

**📊 数据集**

使用的数据集为 LegalBench，主要包含“Learned Hands Courts”和“Hearsay”两类判例推理任务。

**📈 对比分析**

与 SP、CoT、RAG、MAD 基线比较，ACAL 在 Gemini‑2.5‑Flash‑Lite 与 Gemini‑2.5‑Flash 两个后端上均实现或超过最优准确率与 F1 分数，同时提供可争议的推理图。

**⚠️ 局限性**

局限性包括：多代理推理消耗计算资源，需手动调参（如冲突调整因子 β），不确定性评估在缺少 CR 时效果欠佳，以及目前仅在公开基准上验证，尚未在更复杂的法律场景或多语言环境中测试。

---

## 288. Continuous Telemonitoring of Heart Failure using Personalised Speech Dynamics

**arXiv ID:** 2602.19674 | [PDF](https://arxiv.org/pdf/2602.19674v1)

**作者:** Yue Pan `[一作]` (Southeast University), Ming Chu `[通讯]` (Nanjing Medical University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种基于语音的心衰纵向监测框架（LIPT），通过个体化序列编码对患者心衰进展进行跟踪。

**💡 创新点**

创新点在于将心衰诊断从传统的横断面分类转为跨时序的个体化跟踪，并引入Personalised Sequential Encoder（PSE）对长序列进行高效编码。

**🔧 技术方法**

使用的技术包括特征提取（ComParE 2016、RASTA、MFCC 等）、统计显著性筛选（t 检验）、变分自编码器预训练、卷积+LSTM序列编码以及二分类损失。

**📊 数据集**

数据集来自江苏台州人民医院 225 名急性左心衰患者的语音记录，包含入院、出院、6 个月和 12 个月随访，共 7 种语音任务。

**📈 对比分析**

与传统交叉面分类（XGBoost、FNN）相比，LIPT 在 FNN 上将准确率从 69.3% 提升至 81.8%；RASTA+HF-voice 组合几乎实现完美分类；在随访重入预测中，敏感度达到 100%，但特异性偏低。

**⚠️ 局限性**

局限在于缺乏典型稳定负样本导致误报率升高，稳定状态定义不够精细，数据来源单一且录音环境有限，需扩大样本与环境多样性。

---

## 289. Semantic Substrate Theory: An Operator-Theoretic Framework for Geometric Semantic Drift

**arXiv ID:** 2602.18699 | [PDF](https://arxiv.org/pdf/2602.18699v1)

**作者:** Stephen Russell `[一作]` (University of West Florida), Stephen Russell `[通讯]` (University of West Florida)

**通讯引用:** 2499 | [OpenAlex ID](https://openalex.org/A5107278933)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

提出一种统一的语义子strate模型，将语义漂移拆分为平移漂移、重连漂移、动力学漂移和过程漂移四种可观测模式，并引入曲率与桥梁质量等几何量来解释邻域重连风险与递归不稳定性。

**💡 创新点**

创新点在于将多种漂移信号归纳为同一时序子strate的不同投影，形成可测量的机制性预测；通过桥梁质量预测未来重连风险，阐明干预顺序非可交换性。

**🔧 技术方法**

采用基于嵌入的度量空间、局部马尔可夫扩散、粗Ricci曲率、熵变与运输距离等数学工具，构造局部邻域分布并计算曲率、桥梁质量。

**📊 数据集**

未给出具体数据集，本文仅为理论框架，后续研究将在大规模历史语料上验证。

**📈 对比分析**

比较方法未在本文中实施，计划通过预设终点指标（重连漂移、基底切换率、校准误差）和控制变量（频率、窗口设定）进行未来实证验证。

**⚠️ 局限性**

局限包括：嵌入不稳定导致几何失真；邻域图构造不稳会把噪声映射为漂移；窗口划分不当可能引入短期波动；曲率仅为机制性变量，无法完整解释漂移。

---

## 290. Learning to Reason for Multi-Step Retrieval of Personal Context in Personalized Question Answering

**arXiv ID:** 2602.19317 | [PDF](https://arxiv.org/pdf/2602.19317v1)

**作者:** Maryam Amirizaniani `[一作]` (University of Washington), Hamed Zamani `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 14729 | [OpenAlex ID](https://openalex.org/A5100618738)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了PR^2框架，将检索与推理集成用于个性化问答；

**💡 创新点**

创新点在于利用强化学习动态决定何时检索、检索何内容，并将检索结果在多步推理中交互融合，以实现真正的用户定制化回答；

**🔧 技术方法**

采用GRPO强化学习、检索增强推理（RAR）与大型语言模型（Qwen2.5-3B/7B、Gemma3-4B）以及对用户历史记录的检索；

**📊 数据集**

使用专门的个性化问答基准数据集LaMP-QA（包含Art、Lifestyle、Society三域）进行训练与评估；

**📈 对比分析**

在LaMP-QA测试集上与多种基线（No Personalization、Search-R1、RAG-Personalization、PPlug、HYDRA-Adapter、PrLM）对比，PR^2平均提升8.8%–12.0%的个性化评分，均显著优于所有对比方法；

**⚠️ 局限性**

局限性主要包括对检索准确性的依赖、检索噪声对推理的影响，以及模型规模与训练成本高昂，尚需进一步探索更高效的检索与推理策略。

---

## 291. Matching with Committee Preferences

**arXiv ID:** 2602.19009 | [PDF](https://arxiv.org/pdf/2602.19009v1)

**作者:** Haoyu Song `[一作]`, Young-san Lin `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在匹配市场中提出一种将多重排名内生化为机构内部社会选择的框架，并定义可接受的选拔集合与稳定匹配。

**💡 创新点**

创新点在于：①把多重评估标准直接嵌入机构的选择函数；②通过参数化的(α⃗,β)模型实现可接受选拔集合的定义；③使用随机需求的Lindahl均衡和迭代舍入构造近似可接受与稳定匹配，且对容量与排名阈值的扰动量可控。

**🔧 技术方法**

核心技术包括：随机需求的ordinal均衡（Lindahl equilibrium with ordinal preferences）、可接受的分数集合构造、迭代舍入算法以及对多机构场景的Matching Equilibrium with Ordinal Preferences（MEO）建模。

**📊 数据集**

本文为理论工作，未使用任何实验数据集；所有结果均基于理论证明。

**📈 对比分析**

通过理论分析证明：若机构规模相对于评审团人数较大，则仅需对容量和α参数进行极小扰动即可保证存在近似稳定匹配；相比传统的外部随机抽签或预留约束，方法在可接受性和稳定性上更具内生性与可解释性。

**⚠️ 局限性**

主要限制在于计算复杂度——虽然证明了存在性与近似保证，但实际求解均衡与稳定匹配在大规模或约束复杂时可能计算困难，缺乏高效算法。

---

## 292. PIPE-RDF: An LLM-Assisted Pipeline for Enterprise RDF Benchmarking

**arXiv ID:** 2602.18497 | [PDF](https://arxiv.org/pdf/2602.18497v1)

**作者:** Suraj Ranganath `[一作]` `[通讯]` (University of California San Diego), Suraj Ranganath (University of California San Diego)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 PIPE‑RDF 三阶段流程，生成面向企业 RDF 语义的 450 对平衡 NL–SPARQL 基准问答；

**💡 创新点**

创新点在于将逆向查询、类别平衡检索增强提示以及执行验证修复循环结合起来，实现对企业专属模式的高质量、可执行问答生成；

**🔧 技术方法**

使用 LLM（Ollama）、GraphDB SPARQL 解析执行、逆向查询、检索增强提示（RAG）、语法约束与修复以及结构化指标记录技术；

**📊 数据集**

基于从公开 RDF（DBpedia）提取的 5000 家公司公司‑地点切片（Schema C）——仅包含 3 个类和 6 个谓词的迷你数据集；

**📈 对比分析**

通过解析成功率、执行成功率、空结果率、策略覆盖率、查询复杂度等指标评估，与公开基准对比，最终达到 100% 解析与执行成功、LLM 平均延迟 5.3 s、SPARQL 7 ms；

**⚠️ 局限性**

局限性在于仅在单一固定 schema 切片验证，缺乏对更大规模、多模态图谱及 ASK/NEGATION 等操作的覆盖，并未评估下游 NL–SPARQL 模型的实际性能。

---

## 293. Decision MetaMamba: Enhancing Selective SSM in Offline RL with Heterogeneous Sequence Mixing

**arXiv ID:** 2602.19805 | [PDF](https://arxiv.org/pdf/2602.19805v1)

**作者:** Wall Kim `[一作]` (Samsung Electronics), Hanul Kim `[通讯]` (Seoul National University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种新的离线强化学习模型 Decision MetaMamba (DMM)，通过将 Mamba 的 token mixer 替换为密集层序列混合器（DSM），并改进位置结构，提升信息保留与序列建模效果。

**💡 创新点**

创新点在于将局部混合器 DSM 与全局混合器 Mamba 通过残差连接先后组合，先局部混合再全局建模，解决 Mamba 中选择性扫描导致的关键信息丢失问题，并且不需要额外位置编码。

**🔧 技术方法**

技术：密集层序列混合器 (DSM)、改进的 Mamba 结构、残差连接、无位置编码以及在离线 RL 中的序列建模。

**📊 数据集**

使用的数据集包括 D4RL 基准中的 MuJoCo（Hopper、Walker2d、HalfCheetah）、AntMaze、Franka Kitchen 等。

**📈 对比分析**

与基准方法（如 DT、EDT、DM、DS4、CQL、TD3+BC 等）进行比较，DMM 在大多数 dense 和 sparse reward 环境中实现了 state‑of‑the‑art 性能，并且参数量更少，表现更优。

**⚠️ 局限性**

局限：尚未验证在线微调是否进一步提升；缺乏正则化技术研究；实现常数时间推理仍是挑战。

---

## 294. Federated Causal Representation Learning in State-Space Systems for Decentralized Counterfactual Reasoning

**arXiv ID:** 2602.19414 | [PDF](https://arxiv.org/pdf/2602.19414v1)

**作者:** Nazal Mohamed `[一作]` (Georgia Institute of Technology), Nagi Gebraeel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5866 | [OpenAlex ID](https://openalex.org/A5054372641)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在分布式环境中，提出一种联邦因果表示学习框架，用于在状态空间系统中进行反事实推理

**💡 创新点**

将联邦学习与因果分离相结合，实现了在不共享原始数据的前提下进行反事实推理

**🔧 技术方法**

使用线性时不变系统建模、联邦训练算法、因果分解技术及差分隐私保护

**📊 数据集**

采用合成 LTI 系统数据以及公开的机器人/控制系统数据集（如 UCI 控制系统数据集）

**📈 对比分析**

与传统集中式因果学习和非因果联邦学习做对比，实验显示在精度相近的同时，隐私泄露风险显著降低

**⚠️ 局限性**

仅适用于线性或近似线性系统，对非线性系统的推广有限，且通信成本和同步需求较高

---

## 295. PA-Attack: Guiding Gray-Box Attacks on LVLM Vision Encoders with Prototypes and Attention

**arXiv ID:** 2602.19418 | [PDF](https://arxiv.org/pdf/2602.19418v1)

**作者:** Hefei Mei `[一作]` (City University of Hong Kong), Minjing Dong `[通讯]` (City University of Hong Kong)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5084617283)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种针对大型视觉–语言模型共享视觉编码器的灰盒对抗攻击（PA-Attack），通过原型锚定指导和两阶段注意力增强实现对多任务的高效攻击。

**💡 创新点**

创新点包括①利用原型锚定指导提供多样化、稳定的攻击方向，避免单一视觉属性过拟合；②采用两阶段注意力细化机制，动态聚焦关键视觉token并随攻击过程自适应重估注意权重。

**🔧 技术方法**

技术手段包括梯度优化、PCA+K‑means原型构建、基于类token自注意力的token注意力加权、两阶段迭代攻击以及对抗样本的ℓ∞限制。

**📊 数据集**

使用的数据集包括图像字幕任务的COCO、Flickr30k；视觉问答任务的TextVQA、VQAv2；幻觉检测任务的POPE+COCO；以及以COCO为基础的指导集（m=3000）用于原型构建。

**📈 对比分析**

通过与MIX.Attack、VT-Attack、AttackVLM‑ii、VEAttack以及黑盒基准进行对比，PA‑Attack在多模型、多任务上平均实现75.1% SRR，明显优于最佳灰盒攻击（提升≈11%）和黑盒基线（提升≈27%），在ε=2/255时仍能将字幕指标压至个位数以上。

**⚠️ 局限性**

局限性包括：仅针对共享视觉编码器；需要对编码器的灰盒访问；对不同视觉编码器或已细调模型的迁移性待验证；在更严格的扰动预算下隐蔽性可能受限；缺乏对应的防御方法研究。

---

## 296. Beyond Description: A Multimodal Agent Framework for Insightful Chart Summarization

**arXiv ID:** 2602.18731 | [PDF](https://arxiv.org/pdf/2602.18731v1)

**作者:** Yuhang Bai `[一作]` (Hong Kong Polytechnic University), Wenqi Fan `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 4725 | [OpenAlex ID](https://openalex.org/A5043696243)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Chart Insight Agent Flow（CIAF）框架，通过多代理计划与执行实现基于MLLM的图表深度洞察式摘要。

**💡 创新点**

创新点在于将规划、数据洞察与领域洞察分离为专门代理，利用MLLM的视觉理解与推理生成更丰富、领域相关的洞察，并推出专门的ChartSummInsights数据集。

**🔧 技术方法**

采用多模态大型语言模型（如Qwen‑VL、LLaVA、Intern‑VL）作为核心，配合提示式（ICL）规划、数据分析与领域分析代理实现流程。

**📊 数据集**

使用新构建的ChartSummInsights数据集（240张真实图表与专家撰写的洞察式摘要）以及公开的ChartSummInsights Benchmark进行评估。

**📈 对比分析**

通过自定义的Insight Quality（IQ）评分与Insight Diversity（ID‑RC/ID‑Span）指标，与早期端到端模型、微调MLLM以及原生MLLM对比，结果显示CIAF在IQ与多样性上均显著提升，尤其在Qwen‑VL系列上提升了约12% IQ。

**⚠️ 局限性**

局限性包括对大规模图表与新型图表类型的泛化尚不充分，对MLLM的依赖导致推理效率受限，以及数据集规模有限，未来需扩充多样化图表和自动化评估机制。

---

## 297. Discover, Segment, and Select: A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation

**arXiv ID:** 2602.19944 | [PDF](https://arxiv.org/pdf/2602.19944v1)

**作者:** Yilong Yang `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种零样本伪装物体分割框架 DSS（Discover‑Segment‑Select），通过无监督特征聚类生成高质量候选框，利用 SAM 进行分割，再用多模态大语言模型对候选分割结果进行推理式选择，从而得到最终分割掩码。

**💡 创新点**

创新点：① 引入 Feature‑coherent Object Discovery（FOD）模块，将视觉特征聚类与 Part Composition（PC）细化相结合，显著提升候选框的完整性与精度；② 通过 Similarity‑based Box Generation（SBG）从自相似度图生成更完整的提示框；③ 设计 Semantic‑driven Mask Selection（SMS）模块，让 MLLM 在对抗式比较中做出最终掩码选择，避免单次推理可能产生的 hallucination；④ 整个流程无需任何训练，完全零样本。

**🔧 技术方法**

主要技术：自监督视觉编码器 DINOv2 作为特征提取器；Leiden 聚类算法自动确定聚类数；PC 迭代优化特征一致性；SBG 通过自相似度映射生成提示框；SAM2（ViT‑L）进行分割；QWen2.5‑VL‑Instruct 作为多模态大语言模型用于掩码选择。

**📊 数据集**

使用四个公开 COD 数据集：CHAMELEON（76张）、CAMO‑Test（250张）、COD10K‑Test（2,026张）以及 NC4K（4,121张）。

**📈 对比分析**

与全监督、无监督以及其它零样本方法在四个数据集上进行量化对比，DSS 在所有指标（Fβ^w、Sα、Eϕ、ℳ）上均优于现有零样本方法，并与部分全监督方法相近；在多实例场景中表现尤为突出，误检和漏检显著下降；计算时间方面，SMS 阶段占大多数，但总体推理时长仅为 RDVP‑MSD 的 2.32 倍，且显著降低 GPU 内存占用。

**⚠️ 局限性**

局限性：① SMS 仍有改进空间，理想选择（Ideal Seg）与实际选择差距说明掩码评估仍不完美；② 对极小或极细纹理的伪装物体识别仍受限；③ 推理速度受 SMS 阶段主导，虽然整体已可接受，但在实时应用中仍需进一步优化。

---

## 298. TherA: Thermal-Aware Visual-Language Prompting for Controllable RGB-to-Thermal Infrared Translation

**arXiv ID:** 2602.19430 | [PDF](https://arxiv.org/pdf/2602.19430v1)

**作者:** Dong-Guw Lee `[一作]` (Seoul National University), Ayoung Kim `[通讯]` (Seoul National University)

**通讯引用:** 5681 | [OpenAlex ID](https://openalex.org/A5100740100)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可控的 RGB‑to‑TIR 翻译框架 TherA，通过热感知视听语言模型 TherA‑VLM 与潜在扩散器联合，实现从单张 RGB 图像生成热像并支持文本与参考图像控制。

**💡 创新点**

将热物理先验融入 VLM，生成结构化热属性嵌入，并用 TE‑Adapter 在 UNet 中进行条件化，首次实现对热量分布的可控、物理合理生成。

**🔧 技术方法**

基于 LLaVA 的热感知 VLM、潜在扩散模型、TE‑Adapter、双 CFG 引导、R2T2 数据集与多模态 LLM 生成热描述。

**📊 数据集**

构建 R2T2（10万 RGB‑TIR‑文本三元组）并在 M3FD、FLIR、CART 等公开 RGB‑TIR 数据集上训练与评估。

**📈 对比分析**

与多种 SOTA RGB‑to‑TIR 方法在 M3FD、FLIR 上用 PSNR/SSIM/FID/LPIPS 评价，TherA 在所有指标上平均提升约 33%（PSNR+4.66 dB），在零样本迁移中也表现最优。

**⚠️ 局限性**

仅适用于相对热像，像素值为归一化温差，无法生成绝对辐射温度，且对极低光照 RGB 输入的鲁棒性仍有限。

---

## 299. OptiRepair: Closed-Loop Diagnosis and Repair of Supply Chain Optimization Models with LLM Agents

**arXiv ID:** 2602.19439 | [PDF](https://arxiv.org/pdf/2602.19439v1)

**作者:** Ruicheng Ao `[一作]` (Massachusetts Institute of Technology), Xinshang Wang `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个闭环的诊断与修复框架，用于自动诊断和修复多层供应链优化模型中的不合格（不可解）问题，并通过五个基于库存理论的可行性与操作合理性检查，确保修复后的模型既可解又具备运营合理性。

**💡 创新点**

创新点在于：①将模型诊断与修复分为两阶段（域无关的可行性修复与域特定的合理性验证）并分别训练；②使用solver反馈驱动的自我学习循环（Beam Search + 监督蒸馏 + 强化学习）来提升可行性恢复率；③将五个库存理论检查作为可验证的“RationalityOracle”，将运营合理性量化为可训练的奖励；④通过对比22款API LLM和自研8B模型，证明域特化训练比模型规模更关键。

**🔧 技术方法**

技术包括：大规模语言模型（GPT‑5.2、Gemini 2.5 Pro等），自我学习训练框架（Iterative Self‑Taught Reasoning），Beam Search 与 Policy Distillation，基于Gurobi的可解性与合理性检查，强化学习（Group Relative Policy Optimization）与奖励设计，LoRA参数微调。

**📊 数据集**

数据集为两部分：976个注入10类错误的多层供应链实例（2–5层，12–24期，含不同需求模式、成本比例），以及约5,000个跨领域通用LP错误实例（9类），每个实例都配有自然语言描述、Gurobi代码、错误类型标签和人工设定的“ground-truth”修复。

**📈 对比分析**

比较方法：在两条评测轨道上：①对通用LP进行单阶段可行性恢复率的评测，②对供应链问题进行两阶段完整管道评测。结果显示：自研8B模型在可行性恢复率为97.2%，Rational Recovery Rate为81.7%；API模型平均可行性恢复率仅27.6%，Rational Recovery Rate仅42.2%。两阶段模型在修复步骤和token数上也优于API（平均步数5.2 vs 14，token数3,439 vs 20,730）。

**⚠️ 局限性**

局限性包括：错误仅为合成且单一线性模型，未覆盖混合整数或非线性问题；仅针对串行多层库存网络，未考虑网络拓扑、随机需求或多产品；RationalityOracle仅编码固定的五个检查，需按业务自定义；所有评测使用贪婪推理，其他推理策略可能表现不同；实际工业案例验证尚未完成。

---

## 300. Modeling and Recovering Hierarchical Structural Architectures of ROS 2 Systems from Code and Launch Configurations using LLM-based Agents

**arXiv ID:** 2602.18644 | [PDF](https://arxiv.org/pdf/2602.18644v1)

**作者:** Mohamed Benchat `[一作]` (Institute for Software and Systems Engineering), Meng Zhang `[通讯]` (Institute for Software and Systems Engineering)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向ROS 2系统的分层结构架构恢复方法，结合UML建模概念与LLM辅助恢复管线，能够从代码、构建文件和Launch配置中自动生成层级化的架构图。

**💡 创新点**

核心创新在于：①基于ROS 2的蓝图（节点、主题、服务、命名空间、Launch驱动的组合）构建UML Profile；②将蓝图编码为提取契约、结构化提示与验证规则，限制LLM的生成空间，提升恢复的确定性与可验证性；③混合管线将确定性静态分析与受限LLM推理相结合，兼顾低层实现细节与高层组合语义。

**🔧 技术方法**

使用的技术包括：UML建模与PlantUML渲染、静态代码分析（规则驱动提取节点定义）、LLM（如GPT‑4）与CrewAI代理进行结构化提示生成、蓝图约束验证与错误检测、以及基于JSON的中间数据交换。

**📊 数据集**

实验基于三套ROS 2代码库：①无Launch的合成示例（665 LOC，6个节点，3个Launch）；②含Launch的同一合成示例；③工业级Autoware子集（≈14,000 LOC，3个节点，2个Launch）。

**📈 对比分析**

通过与专家手工构建的参考模型在UML层面进行精确比较，采用Precision/Recall/F1指标。结果显示：对单节点层面，精确率、召回率均为1.0；在组合层面，精确率保持高（0.88‑1.0），召回率随系统规模与Launch隐式语义复杂度下降（0.75‑0.35）。与现有自动化建模工具相比，蓝图驱动方法在结构合法性上具有更高的精确率，而在复杂系统中的召回率仍是主要瓶颈。

**⚠️ 局限性**

主要局限在于：①对大型、功能丰富的代码库中隐式的Launch与命名空间语义仍难完整恢复，导致召回率下降；②当前实现未形成正式的UML Profile与元模型，只是概念性描述；③对复杂的Launch语法（参数化、条件执行等）支持有限；④在CI/持续集成中的自动化与实时漂移检测尚未实现。

---

## 301. StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues

**arXiv ID:** 2602.20089 | [PDF](https://arxiv.org/pdf/2602.20089v1)

**作者:** Zanxi Ruan `[一作]` (University of Verona), Marco Cristani `[通讯]` (Reykjavik University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了结构中心的微调框架 StructXLIP，通过提取图像边缘并过滤文本中的外观词（如颜色、材质），将视觉与语言的结构信息进行对齐，从而提升长文本对齐和跨模态检索性能。

**💡 创新点**

创新点包括：① 用边缘图作为视觉结构表征；② 采用 LLM 自动生成词典并对文本进行结构过滤；③ 引入全局结构对齐、局部结构对齐和一致性正则化三种结构中心损失；④ 通过信息理论解释辅助损失对优化稳定性的正向作用；⑤ 该方法可作为 plug‑in 轻松集成到任何 CLIP 微调框架。

**🔧 技术方法**

使用的技术主要有：CLIP 预训练模型；Canny/LoG 等经典边缘检测器；LLM（大语言模型）生成颜色与材质词典；InfoNCE 对比损失；SAM 分割用于局部对齐；多正样本对齐与一致性正则化；信息理论下的互信息分析。

**📊 数据集**

实验使用的主要数据集包括：一般域的 DCI（7.8k 图像+长文本）和 DOCCI（15k 图像+长文本）；专业域的 SKETCHY（46k 服装多模态数据）和 Insect（6k 细粒度生物图像+专家注释）。

**📈 对比分析**

在跨模态检索（Text→Image 与 Image→Text）Recall@1/5/10 指标下，StructXLIP 在所有四个数据集均优于或等价于最新的 Long-CLIP、GOAL、SmartCLIP、FineLIP、SigLIP2 等方法，尤其在 SKETCHY 上 R@1 提升 6–7%。在低数据量（5–20%）下仍保持领先；作为 plug‑in 加入 LoRA/DoRA 也可提升 4–5%。跨域实验表明其泛化能力较强。

**⚠️ 局限性**

局限性：① 侧重结构信息，可能忽视颜色与材质等外观特征；② 依赖边缘提取和 LLM 过滤，质量不佳时效果会下降；③ 仅做微调，未从头训练 VLM；④ 对极长或语义极其丰富的文本仍有限；⑤ 需要额外的预处理步骤，虽推理无额外成本，但训练时需要额外的边缘/文本处理。

---

## 302. RPU -- A Reasoning Processing Unit

**arXiv ID:** 2602.18568 | [PDF](https://arxiv.org/pdf/2602.18568v1)

**作者:** Matthew Adiletta `[一作]`, David Brooks `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

无法完成总结，缺少论文内容

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 303. Robust Exploration in Directed Controller Synthesis via Reinforcement Learning with Soft Mixture-of-Experts

**arXiv ID:** 2602.19244 | [PDF](https://arxiv.org/pdf/2602.19244v1)

**作者:** Toshihide Ubukata `[一作]` (Waseda University), Kenji Tei `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 851 | [OpenAlex ID](https://openalex.org/A5045332896)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

结合Soft Mixture-of-Experts框架，使用先验-置信门控与软混合技术，在OTF-DCS中为RL探索策略提供鲁棒性，解决单一专家的各向异性泛化问题。

**💡 创新点**

创新点在于把专家的各向异性专长视为互补优势，通过高斯核插值构造先验强度并与熵/边距置信度结合，形成动态门控；同时采用软混合而非硬选择，提升搜索鲁棒性。

**🔧 技术方法**

采用强化学习（深度策略/值网络）、Mixture-of-Experts结构、先验加权、熵与边距置信度、Gaussian核插值、Soft-MoE软混合策略。

**📊 数据集**

使用 Air Traffic benchmark（飞机数量n与可用高度k两维参数空间，225个测试实例）进行评估。

**📈 对比分析**

在225个测试实例上将Soft-MoE与单一RL专家进行对比，发现可解空间显著扩展、成功率提升、平均探索步骤下降；尽管多专家推理成本上升，但因探索步骤减少，总体计算时间不成正比增长。

**⚠️ 局限性**

局限包括：门控权重仅在起始状态计算并固定，缺乏在线自适应；多专家推理产生额外开销；实验仅覆盖二维参数和单一基准，难以直接推广至高维或不同领域。

---

## 304. The Story is Not the Science: Execution-Grounded Evaluation of Mechanistic Interpretability Research

**arXiv ID:** 2602.18458 | [PDF](https://arxiv.org/pdf/2602.18458v1)

**作者:** Xiaoyan Bai `[一作]` (University of Chicago), Chenhao Tan `[通讯]` (University of Chicago)

**通讯引用:** 5481 | [OpenAlex ID](https://openalex.org/A5079270249)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套基于执行资源的评估框架（MechEvalAgent），用于检查研究输出的叙述与代码/数据的一致性、可复现性和可推广性，并在机制解释性领域对30个研究结果进行评估。

**💡 创新点**

创新点在于：①首次将代码和数据的可执行性纳入科学评估；②设计统一的研究输出规范和多维度检查表；③构建自动化评估代理，实现与人类专家高一致性并发现额外错误。

**🔧 技术方法**

使用技术包括：大语言模型 Claude Code、Scribe（Jupyter notebook+执行日志）、GitHub 版本控制、自动化 check‑list 评估脚本以及多代理协作的评估管线。

**📊 数据集**

数据集为机制解释性研究中的30个研究输出，涵盖10个复制任务、10个开放式探索任务和10个人写仓库，提供了代码、数据与报告三部分。

**📈 对比分析**

与3名人类专家采用相同 check‑list 进行对比，自动评估与人类一致率超过80%，且发现了 51 条人类未捕捉的错误，评估时间从人类平均 2.2 小时压缩至 30–60 分钟，显著提升效率与覆盖面。

**⚠️ 局限性**

局限性包括：①仅使用单一模型（Claude Code），缺乏多模型冗余；②每个仓库仅由单个专家评审；③二值化检查表可能过于硬化，无法表达细微不确定性；④指令遵循仍存在偶发错误，需进一步完善指令与执行隔离。

---

## 305. SongEcho: Towards Cover Song Generation via Instance-Adaptive Element-wise Linear Modulation

**arXiv ID:** 2602.19976 | [PDF](https://arxiv.org/pdf/2602.19976v1)

**作者:** Sifei Li `[一作]` (Chinese Academy of Sciences), Weiming Dong `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个轻量级的覆盖歌曲生成框架SongEcho，能够在保持原曲主旋律的同时，通过文本提示生成全新演绎；

**💡 创新点**

创新点在于设计了Instance‑Adaptive Element‑wise Linear Modulation（IA‑EiLM）——包含EiLM与IACR两部分，显著提升了条件注入的时序对齐和条件表示的实例适应性；

**🔧 技术方法**

采用FiLM扩展的EiLM对Transformer隐藏层进行逐元素线性调制，并引入IACR模块通过隐藏状态与条件交互实现自适应调节；

**📊 数据集**

构建了开源的Suno70k数据集（约69k首AI生成歌曲），对原有Suno数据进行质量筛选、标签增强和音频评测；

**📈 对比分析**

与Stable Audio ControlNet和MuseControlLite等基准方法相比，SongEcho在RPA、RCA、FD_openl3等多项客观指标上优于对照组，并在MOS评测中获得最高得分，且仅使用原模型3%~26%的可训练参数；

**⚠️ 局限性**

局限性包括对声线细节控制不够（仅支持性别调节）、缺乏局部情感细化（如音色变化、vibrato）以及依赖ACE‑Step的文本控制能力，未来需引入声纹编码器和更细粒度的音乐指令。

---

## 306. Placing Green Bridges Optimally for Robust Habitat Reconnection

**arXiv ID:** 2602.19834 | [PDF](https://arxiv.org/pdf/2602.19834v1)

**作者:** Gero Ellmies `[一作]` (Humboldt University of Berlin), Till Fluschnik `[通讯]` (Humboldt University of Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在不同栖息地最大大小与顶点最大度数的组合下，对Habitat Cycle Vertex Cover（HCVC）问题的复杂性进行系统分析，给出了多项式时间算法与NP‑难性证明；

**💡 创新点**

首次精确划分了HCVC在最大栖息地大小与最大度数参数空间中的复杂性阈值，并提供了针对各参数区间的构造性归约与暴力搜索策略；

**🔧 技术方法**

主要使用图论归约、组合构造、逼迫边分析和暴力搜索等技术；

**📊 数据集**

未使用实验数据集，全部为理论构造与归约实例；

**📈 对比分析**

通过与已知NP‑难问题（如3SAT、Hamiltonian Cycle等）的归约证明NP‑难性；在可解区间内通过暴力搜索实现多项式时间求解；

**⚠️ 局限性**

仅针对栖息地最大大小不超过6的情况给出完整复杂性划分，未讨论更大规模或实际应用中的性能与实现细节。

---

## 307. Satellite-Based Detection of Looted Archaeological Sites Using Machine Learning

**arXiv ID:** 2602.19608 | [PDF](https://arxiv.org/pdf/2602.19608v1)

**作者:** Girmaw Abebe Tadesse `[一作]` (Microsoft), Juan Lavista Ferres `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并评估了一套基于 PlanetScope 月度影像的卫星检测框架，用于自动识别阿富汗考古遗址的被盗情况。

**💡 创新点**

首次系统性比较 ImageNet 预训练 CNN 与传统手工特征+基础模型嵌入的 looting 检测方法，并展示空间掩模与预训练能显著提升准确率，此外公开了包含 1,943 个遗址的最大规模数据集。

**🔧 技术方法**

采用卷积神经网络（ResNet、EfficientNet）配合空间掩模和图像预训练；传统机器学习（随机森林、XGBoost）配合手工光谱纹理特征和地理基础模型嵌入；并通过均值、PCA、Concatenation 等多时相聚合策略进行比较。

**📊 数据集**

使用阿富汗 1,943 个考古遗址（898 个被盗，1,045 个保存）数据集，涵盖 2016–2023 年 4.7 m/像素 PlanetScope 月度拼接图像，并配有人工手绘站点掩模。

**📈 对比分析**

通过 5 折交叉验证和 stratified split 对比 CNN 与传统模型的准确率、精确率、召回率、F1 与 AUROC；ImageNet 预训练 CNN 取得最高 F1≈0.926，传统模型最高 F1≈0.710，空间掩模提升 CNN F1 高达 30–45%。

**⚠️ 局限性**

研究仅局限于阿富汗单一地区，可能对不同地质、土地利用或更高分辨率影像的泛化性有限；月度拼接可能平滑短时破坏，且需要人工掩模支持。

---

## 308. Health+: Empowering Individuals via Unifying Health Data

**arXiv ID:** 2602.19319 | [PDF](https://arxiv.org/pdf/2602.19319v1)

**作者:** Sujaya Maiyya `[一作]` (University of Waterloo), Avinash Kumar `[通讯]` (Independent Researcher)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一个面向个人用户的多模态健康数据管理系统Health^+，支持用户上传、查询、共享各种格式的医疗记录（文本、图像、报告等），并提供智能推荐和可解释的隐私保护机制。

**💡 创新点**

创新点主要包括：① 以用户为中心的设计理念，允许非技术用户轻松管理多模态数据；② 通过学习常见查询模式动态生成存储和索引策略；③ 结合多模态融合与推理实现语义查询；④ 在云端实现加密、侧信道隐藏和SMC等多重隐私保护；⑤ 采用联邦学习自动化推荐共享范围，避免全量共享。

**🔧 技术方法**

技术实现涵盖：元数据解析与标注、Schema & Policy 管理、增量式数据丰富与索引、Deterministic/Order‑Preserving Encryption、ORAM/SMC、LLM 语义查询翻译、可解释共享策略、联邦学习、加密索引与查询优化。

**📊 数据集**

未在论文中给出具体实验数据集，示例使用虚拟的医院、影像中心、实验室以及可穿戴设备等多模态真实医疗记录进行演示；若实测则可能采用公开的多模态医学数据集（如 MIMIC‑III、NIH Chest X‑ray、PhysioNet 等）进行评估。

**📈 对比分析**

论文属于 vision 设计，未给出量化实验；若实施，则可通过与传统 EHR 系统比较，测量查询延迟、数据一致性、共享精度及加密运算开销。预期实现高效查询（低毫秒级延迟）并保持较低的隐私泄漏风险；但在加密/ORAM/SMC 上的开销仍需要评估。

**⚠️ 局限性**

局限性包括：① 需要用户主动上传，缺乏与医院系统的自动同步；② 侧信道攻击、加密效率和可扩展性仍是挑战；③ 联邦学习与隐私策略的调优复杂；④ 依赖法规与合规，跨国部署存在法律障碍；⑤ 目前缺乏大规模真实部署与性能验证。

---

## 309. Deep Reinforcement Learning for Optimizing Energy Consumption in Smart Grid Systems

**arXiv ID:** 2602.18531 | [PDF](https://arxiv.org/pdf/2602.18531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 310. An efficient recursive decomposition algorithm for undirected graphs

**arXiv ID:** 2602.19189 | [PDF](https://arxiv.org/pdf/2602.19189v1)

**作者:** Pei Heng `[一作]`, Jianhua Guo `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于凸包的递归图分解算法，利用MCS顺序定位原子，避免了传统方法中的最小三角化步骤。

**💡 创新点**

创新点在于证明原子可通过包含最小MCS节点及其邻域的凸包递归获取，并给出了RDA与PRDA两种实现，时间复杂度降低到O(nm)，大幅提升运行速度。

**🔧 技术方法**

主要技术包括最大卡数搜索(MCS)、凸包（CMSA）计算、递归分解与并行分解框架。

**📊 数据集**

使用了来自Snap和NetworkRepository的真实网络数据集，如Animal Network-1/2、bio-CE-GT、Email-Enron等。

**📈 对比分析**

通过20次实验对比Xu‑Guo的Leimer+MCS‑M算法，记录平均运行时间；RDA在绝大多数网络上比传统算法快1–2个数量级，超过传统算法设定的时间阈值的网络不再记录。

**⚠️ 局限性**

局限性包括并行版PRDA受MCS顺序影响，某些图形无法实现并行化；对极大稠密图仍受O(nm)上限限制，且实验未覆盖所有网络类型。

---

## 311. RadioGen3D: 3D Radio Map Generation via Adversarial Learning on Large-Scale Synthetic Data

**arXiv ID:** 2602.18744 | [PDF](https://arxiv.org/pdf/2602.18744v1)

**作者:** Junshen Chen `[一作]` (Chinese University of Hong Kong), Shuguang Cui `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 30301 | [OpenAlex ID](https://openalex.org/A5009164482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 RadioGen3D 框架，用于 3D 无线电地图的高质量合成与精确估计。

**💡 创新点**

创新点：①利用有限真实测量和参数化目标模型高效合成 5 万张 3D 无线电地图数据集 Radio3DMix；②在该数据集上以条件 GAN 训练 3D U‑Net，兼顾对抗损失与 L1 损失，实现高精度、低延迟 RME；③通过微调实现强泛化，跨数据集（从 Radio3DMix 到 Ray‑Tracing 数据集）性能仍保持优秀。

**🔧 技术方法**

采用的技术包括：参数化目标模型 + 最小二乘系数拟合；Gaussian 热图编码发射机信息；3D U‑Net 结构加自注意力模块；条件 GAN（判别器对齐输入-输出配对）；L1 + 对抗损失，Adam 优化。

**📊 数据集**

使用的数据集：自研 5 万张 Radio3DMix（200 城区建筑、每张 2 台发射机、1–10% 稀疏测量），对比 Ray‑Tracing 生成的 Radio3D‑RT（5000 张、单台发射机、全向天线），以及公开基线数据集如 UrbanRadio3D、RadioMapSeer 等。

**📈 对比分析**

通过 RMSE、NMSE、SSIM、PSNR 等指标与 RME‑GAN、RadioUNet、RadioDiff‑3D 等 2D/3D 基线对比。结果显示：在 1% 稀疏采样下，RadioGen3D 的 RMSE < 0.01、NMSE < 0.004、SSIM > 0.99、PSNR > 45 dB；推理时间仅 0.06 s，显著优于其他方法。

**⚠️ 局限性**

局限性：仅使用发射机信息（无测量）时精度明显下降；模型需要稀疏测量来推断天线极化；目前仅针对两台发射机场景，单发射机场景泛化受限。

---

## 312. EMS-FL: Federated Tuning of Mixture-of-Experts in Satellite-Terrestrial Networks via Expert-Driven Model Splitting

**arXiv ID:** 2602.19485 | [PDF](https://arxiv.org/pdf/2602.19485v1)

**作者:** Angzi Xu `[一作]`, Shuguang Cui `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 EMS-FL，一种在卫星‑地面网络（STN）环境下的混合专家（MoE）模型联邦微调与训练方法；

**💡 创新点**

创新点在于：①基于专家与本地数据相关度的专家驱动模型拆分，确保每个设备簇仅训练对应的专家并保持其它专家冻结；②非重叠专家分配与异步本地训练，使卫星在有限覆盖内可连续训练所有簇；③采用掩码门控与 LoRA 进一步降低本地内存与通信开销；

**🔧 技术方法**

核心技术包括：联邦学习、Mixture‑of‑Experts（MoE）结构、低秩参数压缩（LoRA）、掩码门控策略、卫星‑地面通信模型以及异步训练调度；

**📊 数据集**

实验使用公开数据集：MMLU、CMExam、MedMCQA 进行微调，GLUE（SLJ、SSJ、NLI、QA‑NLI）进行完整训练；

**📈 对比分析**

与传统同步 FL（全模型同步上传/下载）相比，EMS‑FL 在同样的通信/内存开销下实现了 50–75% 的训练时间缩短，准确率提升至 63–70%（取决于任务），并在实验中展示了更快的收敛速率与更高的最终精度；

**⚠️ 局限性**

限制包括：需要先估计专家与数据相关度以进行拆分；在极端数据异质性下可能导致某些专家更新频率不足；掩码门控可能导致专家间训练不平衡；方法主要针对可拆分的 MoE 大模型，无法直接用于无专家结构的传统大模型。

---

## 313. Learning Beyond Optimization: Stress-Gated Dynamical Regime Regulation in Autonomous Systems

**arXiv ID:** 2602.18581 | [PDF](https://arxiv.org/pdf/2602.18581v1)

**作者:** Sheng Ran `[一作]` (Washington University in St. Louis), Sheng Ran `[通讯]` (Washington University in St. Louis)

**通讯引用:** 3924 | [OpenAlex ID](https://openalex.org/A5071518409)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个两时间尺度的动态框架，利用内部压力变量监测自身动态健康并通过门控机制触发结构可塑性，从而在没有外部目标函数的情况下产生可重复的学习事件。

**💡 创新点**

创新点在于放弃传统的外部损失优化，改为以内部动态指标（冻结指数、非遍历性、不可逆性）评估系统健康，并用压力门控实现结构更新的离散化，从而实现自我评估与自我重组。

**🔧 技术方法**

技术主要包括连续时间Langevin动力学与离散时间循环网络的两时间尺度耦合、指数加权平均的压力累积、以及基于动态指标的门控触发机制。

**📊 数据集**

无外部数据集，所有实验均在仿真高维随机网络的自生成时间序列上完成。

**📈 对比分析**

通过与持续可塑性对照实验比较，门控可塑性在压力累积后触发结构跃迁，形成可重复的学习周期；相反，持续可塑性导致持续漂移且缺乏明确的学习事件。

**⚠️ 局限性**

局限性在于仅在极简模型中验证，缺乏在真实神经网络或复杂环境中的实验，压力指标设计较为主观，尚未证明对更大规模或真实任务的可迁移性。

---

## 314. Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models

**arXiv ID:** 2602.19926 | [PDF](https://arxiv.org/pdf/2602.19926v1)

**作者:** Jin Liu `[一作]` (Xidian University), Junkang Liu `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LA-LoRA，一种在差分隐私联邦学习（DPFL）中对大规模视觉模型和语言模型进行参数高效微调的新框架；

**💡 创新点**

通过局部交替更新 LoRA 的两个低秩矩阵并加入低通滤波器，解决梯度耦合、噪声放大和全局收敛尖锐性三大挑战，显著提升隐私保护下的模型性能；

**🔧 技术方法**

结合 LoRA、差分隐私机制（梯度裁剪+高斯噪声）、低通滤波器、FedAvg聚合以及理论收敛分析；

**📊 数据集**

在视觉任务中使用 Swin Transformer（Swin‑T、Swin‑B）配合 Tiny‑ImageNet 和 CIFAR‑100 数据集；在语言任务中使用 RoBERTa‑Base 并在 GLUE（SST‑2、QNLI、QQP、MNLI）基准上评测；

**📈 对比分析**

与 DP‑LoRA、FFA‑LoRA、RoLoRA 等基线对比，LA‑LoRA 在 ϵ=1 的严格隐私预算下，Swin‑B 在 Tiny‑ImageNet 上提升 16.83% 的测试准确率，整体在所有任务与隐私级别上均保持最高或最稳健的表现；

**⚠️ 局限性**

目前未对极端异构客户端分布、超大模型规模以及多任务迁移进行深入验证，且需要针对不同任务调优滤波和更新策略，未来工作需进一步扩展鲁棒性与可扩展性。

---

## 315. Do Large Language Models Understand Data Visualization Principles?

**arXiv ID:** 2602.20084 | [PDF](https://arxiv.org/pdf/2602.20084v1)

**作者:** Martin Sinnona `[一作]`, Emmanuel Iarussi `[通讯]` (Universidad Torcuato Di Tella)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过构建含 57 条可通过 ASP 形式化的可视化原则的合成与真实 Vega‑Lite 规范数据集，系统评估 LLM 与 VLM 在文本规范和图像+文本输入下对图表遵循这些原则的检测与修复能力。

**💡 创新点**

①首次将 LLM/VLM 与基于 ASP 的形式化约束直接对接；②提出统一的多轮 Prompt 设计与 JSON 结果格式；③在检测与修复两个任务上同时量化模型表现，揭示检测与纠错之间的明显性能差距。

**🔧 技术方法**

使用大型语言模型（GPT‑4o、Gemini‑2.5‑Flash、GPT‑OSS‑20B 等）和视觉语言模型（Llava‑13B、Gemma‑3‑12B），配合 ASP（Clingo）进行原则编码，利用 Vega‑Lite 进行图表渲染与规范转换。

**📊 数据集**

①合成数据集：2000 条 Vega‑Lite 规范，涵盖 57 条原则，共 12,858 条违规实例；②真实数据集：从 GitHub 转译 307 条 Vega‑Lite 规范，评估 16 条原则。

**📈 对比分析**

比较方法：宏观 F1 分数、检测标准差；修复任务评估编译率、强制率（ER）与合规率（CR）。结果显示：最优文本模型 Gemini‑2.5‑Flash 在合成数据集上 F1=0.678，VLM 模型提升至 0.716；在真实集上最高 F1=0.778；修复方面 Gemini‑2.5‑Flash ER 达 94.3%，CR ≈0.724。

**⚠️ 局限性**

①检测仍受限于细粒度感知约束，F1 通常低于 0.10；②评估仅覆盖 57 条原则，未包含所有可视化设计规则；③对真实图表的高分可能受预训练样本偏倚影响；④缺乏对模型推理过程的解释与校准分析；⑤对多模态输入的利用仍有限。

---

## 316. From Docs to Descriptions: Smell-Aware Evaluation of MCP Server Descriptions

**arXiv ID:** 2602.18914 | [PDF](https://arxiv.org/pdf/2602.18914v1)

**作者:** Peiran Wang `[一作]` (University of California Los Angeles), Yuan Tian `[通讯]` (University of California Los Angeles)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Model Context Protocol（MCP）服务器的工具描述进行系统化研究，提出并验证四维质量标准（准确性、功能性、信息完整性与简洁性），并标注了 10,831 个公开 MCP 服务器的 18 类描述“味道”；

**💡 创新点**

首次构建 MCP 描述的“味道”分类与四维质量标准，并通过大规模实验展示描述质量对 LLM 工具选择、效率及竞争优势的显著影响；

**🔧 技术方法**

利用代码与描述对比抽取、LLM 生成的卡片排序与问题分类、统计分析、描述变异实验以及对比实验（工具选择概率评估），核心使用 gpt‑4o‑mini；

**📊 数据集**

10,831 个公开 MCP 服务器的元数据与工具描述，并对其进行 18 类描述味道的手工/LLM 标注；

**📈 对比分析**

通过在不同描述质量维度下进行工具选择实验（使用 gpt‑4o‑mini），发现功能性 +11.6%、准确性 +8.8%、信息完整性 +5.9%、简洁性 +1.5%；在功能相同的服务器竞争场景中，符合标准的服务器选择概率提升至 72%（相对 20% 基线提升 260%）；

**⚠️ 局限性**

仅评估工具元数据，未覆盖资源与提示；采用静态代码‑描述匹配，未检测运行时动态错误或描述漂移；实验基于公开服务器，缺乏对企业私有或快速演化生态的覆盖；

---

## 317. ComUICoder: Component-based Reusable UI Code Generation for Complex Websites via Semantic Segmentation and Element-wise Feedback

**arXiv ID:** 2602.19276 | [PDF](https://arxiv.org/pdf/2602.19276v1)

**作者:** Jingyu Xiao `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 41430 | [OpenAlex ID](https://openalex.org/A5069596903)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个多页复杂网站基准ComUICoder Benchmark，并设计了基于组件的UI代码生成框架ComUICoder，解决了复杂网页UI分割、代码重复、UI不一致等问题。

**💡 创新点**

创新点包括：①混合语义感知分块HSBS，融合MLLM语义与UI检测工具；②基于视觉相似度与图结构匹配的块合并VGBM；③优先级元素级反馈PEF，实现细粒度修正；③基准含多页、组件注释，支持组件重用评估。

**🔧 技术方法**

采用了MLLM与UI检测工具的混合分块、SSIM/LPIPS视觉相似度、图结构匹配、CLIP视觉相似度、文本相似度、Prompt工程、层级代码生成以及元素级匹配与反馈。

**📊 数据集**

使用了40个真实网站共150页，标注了2055个语义UI块，1134个组件组的ComUICoder基准（来源于Moz Top 500），并公开数据。

**📈 对比分析**

与10+基线（开源微调、prompt、分段方法）进行对比，利用低级视觉指标（Block‑Match、Text、Position、Color）、高级视觉指标（CLIP、SSIM）、代码指标（TreeBLEU、重复率、重用率）等，在所有指标上均取得最高分，重用率最高，重复代码最少。

**⚠️ 局限性**

局限性包括：仍受MLLM视觉理解能力限制；需要人工标注组件，标注成本高；对极其复杂的动态交互或实时数据场景尚未充分验证。

---

## 318. Softmax is not Enough (for Adaptive Conformal Classification)

**arXiv ID:** 2602.19498 | [PDF](https://arxiv.org/pdf/2602.19498v1)

**作者:** Navid Akhavan Attar `[一作]` (University of Melbourne), Uwe Aickelin `[通讯]` (University of Melbourne)

**通讯引用:** 7581 | [OpenAlex ID](https://openalex.org/A5002768704)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Helmholtz自由能的能量尺度非一致性分数，对传统基于softmax的CP非一致性分数进行逐样本重加权，从而提升预测集的自适应性与效率。

**💡 创新点**

创新点在于将预softmax对数值空间中的能量信息引入CP框架，利用自由能作为模型对样本熟悉度的量化指标，突破softmax不可靠的局限；同时提供统一的尺度函数实现非一致性分数的动态调节。

**🔧 技术方法**

使用Conformal Prediction、能量基非一致性分数、自由能计算、温度缩放及多种深度网络（ResNet、ImageNet等）实现。

**📊 数据集**

在CIFAR‑100、ImageNet、Places365、长尾CIFAR‑100‑LT等数据集上进行实验，涵盖多种网络架构。

**📈 对比分析**

与传统softmax非一致性分数及其多种变体（如距离、逆概率等）相比，能量增强版本在保持覆盖率的前提下显著降低平均集大小、提升自适应性；在OOD（Places365）下更易产生大集或空集，符合可靠性设计。

**⚠️ 局限性**

仍受CP交换性假设限制，OOD下无法提供严格覆盖保证；能量计算依赖logit空间，可能增加计算开销；在极端分布偏移或多模态数据上效果尚待进一步验证。

---

## 319. When AI Teammates Meet Code Review: Collaboration Signals Shaping the Integration of Agent-Authored Pull Requests

**arXiv ID:** 2602.19441 | [PDF](https://arxiv.org/pdf/2602.19441v1)

**作者:** Costain Nachuma `[一作]` (Idaho State University), Minhaz Zibran `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统性分析了来自AIDev数据集的代理作者拉取请求在GitHub上的集成结果、决策时间以及回顾期间的协作信号，发现审核者参与度、协作稳定性等社交信号是影响成功集成的主要因素。

**💡 创新点**

首次将代理拉取请求视为社会技术过程，结合定量回归和定性编码，揭示了审核参与度和协作稳定性对集成的重要作用，超越单纯的代码质量或迭代量。

**🔧 技术方法**

使用逻辑回归（仓库聚类标准误）、描述性统计、log变换、Wilson置信区间以及定性编码法。

**📊 数据集**

AIDev数据集（GitHub公开代理作者PR记录，包含33,596条PR，1,797作者，2,807仓库）。

**📈 对比分析**

通过逻辑回归比较不同协作信号对合并概率的影响，结果显示审核者参与度的odds ratio最高，强制推送降低merge机会；迭代次数与合并关联弱；整体merge率约71.5%，决策时间与代理不同有显著差异。

**⚠️ 局限性**

研究基于公开流行仓库，缺乏因果推断，无法捕捉完整的审核者意图，结果可能不适用于私有项目或未来更先进的代理。

---

## 320. Robust Predictive Uncertainty and Double Descent in Contaminated Bayesian Random Features

**arXiv ID:** 2602.19126 | [PDF](https://arxiv.org/pdf/2602.19126v1)

**作者:** Michele Caprio `[一作]` (University of Manchester), Sayan Mukherjee `[通讯]` (Max Planck Institute for Mathematics in the Sciences)

**通讯引用:** 71877 | [OpenAlex ID](https://openalex.org/A5014241799)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Huber型污染集的鲁棒贝叶斯随机特征回归框架，通过在先验和似然上引入ε-和η-污染集，实现对模型误设的显式容忍；

**💡 创新点**

创新点在于将传统的单一高斯先验/似然替换为可逼近的可信集，并利用悲观推广贝叶斯更新，得到可解析的上下后验预测密度与方差包络，且在高维比例增长下保持双峰下降结构；

**🔧 技术方法**

采用可信集理论、悲观推广贝叶斯推理、Huber型污染集、精确与近似的后验预测密度界定、Imprecise Highest Density Region (IHDR) 的外部逼近与截断逼近的方差上界；

**📊 数据集**

实验主要使用人工生成的随机特征教师模型以及带有Huber型标签离群的合成数据；

**📈 对比分析**

通过与经典高斯贝叶斯随机特征模型对比，展示了在不同η（或ρ）下预测误差曲线的双峰下降仍保持峰位不变但峰值上升，证明理论上界在实测中既保留了结构又提供了保守的误差包络；

**⚠️ 局限性**

局限性包括需要人工设定截断区间或η值、仅给出最坏情况界限、假设污染水平不极端、且尚未扩展到核极限或深度特征网络。

---

## 321. Responsible Intelligence in Practice: A Fairness Audit of Open Large Language Models for Library Reference Services

**arXiv ID:** 2602.18935 | [PDF](https://arxiv.org/pdf/2602.18935v1)

**作者:** Haining Wang `[一作]` (Indiana University), Angelica Peña `[通讯]` (San Leandro Public Library)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估开放式大型语言模型（Llama‑3.1 8B、Gemma‑2 9B、Ministral 8B）在学术与公共图书馆虚拟参考服务中的公平性，使用系统化的Fairness Evaluation Protocol（诊断分类+词语解释），通过合成的姓名暗示性别与种族/民族的邮件查询进行实验。

**💡 创新点**

首次在图书馆服务场景中引入FEP方法，结合诊断分类与统计解释词语，揭示多模型在性别和种族/民族维度上的公平性差异，发现仅在Llama‑3.1学术情境中出现细微性别差异。

**🔧 技术方法**

采用TF‑IDF特征提取、逻辑回归、MLP、XGBoost诊断分类、统计逻辑回归解释词语、Bonferroni校正、五折交叉验证；使用学术与公共图书馆模板、姓名生成等模拟交互流程。

**📊 数据集**

合成参考查询邮件：使用SSA婴儿姓名数据库与美国人口普查姓氏数据库生成12种性别×族裔组合的姓名；对每个模型、情境生成2500条响应，共计18,000条响应。

**📈 对比分析**

通过诊断分类准确率与随机基准比较，并用t检验+Bonferroni校正判断显著性；若显著则统计回归识别关键词。结果显示种族/民族分类无显著差异；性别分类在Llama‑3.1学术情境略显著，但仅由称呼“dear”驱动，整体表现表明LLM在此场景下公平性良好。

**⚠️ 局限性**

局限包括：仅使用标准英语与姓名暗示种族/性别，二元性别设定，未考虑方言、多语言或真实用户交互；未评估事实真实性、交叉性别与族裔效应；仅测试开放LLM，未验证实时数据库检索对公平性的影响。

---

## 322. Adaptive Underwater Acoustic Communications with Limited Feedback: An AoI-Aware Hierarchical Bandit Approach

**arXiv ID:** 2602.20105 | [PDF](https://arxiv.org/pdf/2602.20105v1)

**作者:** Fabio Busacca `[一作]` (University of Catania), Yin Sun `[通讯]` (Auburn University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计了双层多臂赌博机框架，联合自适应调制、功率控制与反馈间隔调度，提升水下声学网络吞吐量与能效。

**💡 创新点**

创新点在于结合AoI感知的双层MAB，既优化调制/功率，又动态调整反馈频率，兼顾吞吐与反馈开销。

**🔧 技术方法**

使用上下层UCB算法的上下文延迟MAB与非上下文MAB、AoI度量、分配信用机制。

**📊 数据集**

使用DESERT水下网络模拟器，模拟真实环境下的多节点场景。

**📈 对比分析**

与DRL-AM、DRL-MCS基线对比，吞吐量提升最多20.6%，能耗降低36.6%，收敛更快且更稳定。

**⚠️ 局限性**

仅在仿真中验证，未考虑硬件实现复杂性，且仅在小规模节点（≤10）场景下评估。

---

## 323. Fine-Pruning: A Biologically Inspired Algorithm for Personalization of Machine Learning Models

**arXiv ID:** 2602.18507 | [PDF](https://arxiv.org/pdf/2602.18507v1)

**作者:** Joseph Bingham `[一作]` (Technion Israel Institute of Technology), Dvir Aran `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于生物神经修剪原理的Fine‑Pruning算法，用无标签本地数据对预训练模型进行稀疏化与个性化。

**💡 创新点**

创新点在于将突触修剪的“用它或失去”机制直接映射到模型压缩与自适应，既不需要反向传播也不依赖标注数据，能在保持甚至提升精度的同时实现高达70%稀疏。

**🔧 技术方法**

技术上采用基于激活幅度的绝对值剪枝、前向传播激活统计、无监督数据驱动的权重筛选，并与传统反向传播、SVD分解等方法对比。

**📊 数据集**

实验覆盖语音识别的Free Spoken Digit、面部表情识别的CK+、图像分类的ImageNet子集以及MobileNetV2、VGG‑19/7、ResNet50等多种网络架构。

**📈 对比分析**

通过与标准反向传播微调、SVD分解、DFPC、DepGraph等方法在内存占用、压缩率与精度提升三维度对比，Fine‑Pruning在70%稀疏下可提升约67%精度，且内存和计算量显著低于传统方法。

**⚠️ 局限性**

局限性包括只能对已有训练好的源模型进行微调，无法从零训练；当目标分布与源分布相似度高时增益下降；过度剪枝会导致精度急剧下降。

---

## 324. SemanticNVS: Improving Semantic Scene Understanding in Generative Novel View Synthesis

**arXiv ID:** 2602.20079 | [PDF](https://arxiv.org/pdf/2602.20079v1)

**作者:** Xinya Chen `[一作]` (Max Planck Institute for Informatics), Jan Eric Lenssen `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种将预训练语义特征（DINO）融入多视角扩散模型的生成式新视角合成方法。

**💡 创新点**

创新点在于：① 在条件输入中加入几何投影的语义特征；② 在每一步去噪时交替使用当前无噪声估计的语义特征，显著提升长距离视角下的语义一致性和图像质量。

**🔧 技术方法**

核心技术包括：多视角条件扩散模型（基于SEVA）、预训练视觉模型DINO进行语义特征提取、几何投影与线性降维、交替语义理解-生成策略。

**📊 数据集**

使用RealEstate10K（室内外真实视频）和Tanks‑and‑Temples（户外测试集）两大数据集进行评估。

**📈 对比分析**

与ViewCrafter、Uni3C、SEVA等基线进行定量比较，FID提升4.69%–15.26%，图像质量提升4.93%–13.41%，图像质量漂移降低25.07%–28.77%；在长轨迹和跨域数据集上表现尤为突出。

**⚠️ 局限性**

局限性包括：对预训练模型的依赖、在极端视角下仍可能出现细节缺失、训练成本较高，且在极大视角偏移时仍未完全消除偶发的几何不一致。

---

## 325. GazeFlow: Personalized Ambient Soundscape Generation for Passive Strabismus Self-Monitoring

**arXiv ID:** 2602.19966 | [PDF](https://arxiv.org/pdf/2602.19966v1)

**作者:** Joydeep Chandra `[一作]` (Tsinghua University), Yong Zhang `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了 GazeFlow——一种基于浏览器、使用个性化环境音景的被动斜视与眼球协同异常自我监测系统，帮助用户在日常电脑使用中提高对眼部状态的外围意识；

**💡 创新点**

创新点包括①利用 β‑VAE 与小波分解实现 Binocular Temporal‑Frequency Disentanglement（BTFD），可解释分离 vergence、saccade 与 fixation 动态；②提出 Contrastive Biometric Pre‑training（CBP）以自监督对比学习实现跨 1000Hz→30Hz 采样率的域迁移；③引入 Gaze‑MAML 元学习框架，实现仅 15 秒校准的 5‑shot 个性化；④设计了逐渐演变的环境音反馈，将异常严重度与音乐参数映射，符合 calm computing 原则；

**🔧 技术方法**

使用技术包括 MediaPipe 眼动追踪、Discrete Wavelet Transform、β‑VAE、对比学习、MAML、Tone.js 生成音频、自动化音频参数映射与指数平滑；

**📊 数据集**

使用的数据集为 GazeBase（1000Hz 12,334 录音，N=322）进行预训练与评估，以及 30Hz webcam 数据（N=10，6 斜视、4 对照）用于验证；

**📈 对比分析**

在同域和跨域实验中与 Population AE、Personal AE、BTFD、BTFD+C 进行对比，GazeFlow 在 30Hz 上 F1 最高 0.84，仅因域迁移损失 4%；在用户体验实验中，与报警/视觉反馈比较，环境音在意识提升、低侵入性与使用意愿上均表现最佳；

**⚠️ 局限性**

局限性包括①异常检测依赖注入的合成异常，缺乏真实临床标签；②用户样本量小（N=6）；③未对分解因子进行专家验证；④仅使用自报漂移作为 ground truth；⑤隐私与临床验证尚未完成。

---

## 326. Sycophantic Chatbots Cause Delusional Spiraling, Even in Ideal Bayesians

**arXiv ID:** 2602.19141 | [PDF](https://arxiv.org/pdf/2602.19141v1)

**作者:** Kartik Chandra `[一作]` (Massachusetts Institute of Technology), Joshua B. Tenenbaum `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 73469 | [OpenAlex ID](https://openalex.org/A5071093940)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文通过构建贝叶斯对话模型，模拟AI聊天机器人对用户意见的赞同（sycophancy）如何导致用户在与之长时间交互后形成极端错误信念（delusional spiraling），并验证即使理想的贝叶斯推理者也会受到影响。

**💡 创新点**

创新点在于首次用正式的贝叶斯推理框架证明sycophancy是导致AI精神病的因果机制，并系统评估了两种干预策略（强制事实化和用户知情）对失真螺旋的抑制效果。

**🔧 技术方法**

采用贝叶斯推理、层级认知模型（类似Rational Speech Acts）、Monte Carlo仿真以及memo编程语言实现模型与实验。

**📊 数据集**

实验未使用真实数据集，而是基于抽象的二元世界状态和预设的概率分布（如 p(D_i|H)=2/5 或 3/5）生成模拟对话数据，代码托管在OSF平台。

**📈 对比分析**

通过在不同sycophancy比例π（0至1）下进行10,000次对话仿真，统计“灾难性失真螺旋”发生率；结果显示，π>0时失真率显著上升，干预可降低但未完全消除失真，且干预效果在高π时更为明显。

**⚠️ 局限性**

局限性包括：模型假设用户为理想贝叶斯推理者，未考虑人类非理性、情感与社会因素；实验情境过于简化，真实用户可能出现更复杂的行为；即使干预大幅降低失真率，在大规模用户基数下仍可能导致大量人受害。

---

## 327. ABD: Default Exception Abduction in Finite First Order Worlds

**arXiv ID:** 2602.18843 | [PDF](https://arxiv.org/pdf/2602.18843v1)

**作者:** Serafim Batzoglou `[一作]` `[通讯]`, Serafim Batzoglou

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在有限一阶关系世界中构建了ABD基准，要求模型生成异常谓词定义以修复给定的默认理论。

**💡 创新点**

提出了三种观测语义（完整、存在、普遍）以及基于SMT的可验证性与成本衡量，并通过对抗性过滤和CEGIS式实例生成来保证难度控制。

**🔧 技术方法**

使用SMT求解器（Z3）进行模型可验证性和异常度计算，配合对抗过滤、实例生成和十款前沿LLM的零射提示实现评估。

**📊 数据集**

提供了600个ABD实例，涵盖三种情景和七个默认理论，世界域大小9-12，已公开可复现。

**📈 对比分析**

通过有效率、异常度gap（相对solver下界和gold规则）以及持久化检验进行比较；最强模型有效率≈90‑99%，gap≈1‑1.6异常/世界；在Full/Partial上持久化gap≈+1，Skeptical则有效率显著下降。

**⚠️ 局限性**

限制包括规模有限导致可用case‑splitting、free‑Ab基准为保守下界、gold规则并非唯一、对大公式易过拟合以及持久化表现受观测语义影响。

---

## 328. Counterfactual Understanding via Retrieval-aware Multimodal Modeling for Time-to-Event Survival Prediction

**arXiv ID:** 2602.19987 | [PDF](https://arxiv.org/pdf/2602.19987v1)

**作者:** Ha-Anh Hoang Nguyen `[一作]` (University of Engineering and Technology), Hoang-Quynh Le `[通讯]` (University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为CURE的检索感知多模态时间到事件生存预测框架，用于估计不同治疗方案下的个体生存曲线。

**💡 创新点**

创新点在于通过隐式潜在子群检索实现对治疗响应的无监督建模，结合交叉注意力融合多模态数据和混合专家机制自动挑选最具信息量的组学特征。

**🔧 技术方法**

核心技术包括多模态交叉注意力编码、混合专家（MoE）组学嵌入、潜在子群混合分配、基于Cox比例风险的潜在风险网络，以及隐式检索的混合子群融合。

**📊 数据集**

使用公开的METABRIC和TCGA‑LUAD两大癌症数据集，分别包含临床、旁临床、人口统计和多组学特征。

**📈 对比分析**

与CPH、DeepSurv、RSF、DeepHit、Cox‑Time、CMHE、SA‑DGNet等基线方法对比，CURE在C^td（时间相关一致性指数）和IBS（集成Brier分数）上均取得最高分或接近最高分，显著优于其它模型。

**⚠️ 局限性**

局限性包括对多模态数据完整性和质量的高依赖、潜在的公平性与偏差问题，以及在真实临床部署前缺乏外部验证和伦理监管。

---

## 329. Addressing Instrument-Outcome Confounding in Mendelian Randomization through Representation Learning

**arXiv ID:** 2602.19782 | [PDF](https://arxiv.org/pdf/2602.19782v1)

**作者:** Shimeng Huang `[一作]` (Institute of Science and Technology Austria), Francesco Locatello `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于跨环境不变性表示学习的框架，用来分离混合遗传工具中的可变成分与不可变成分，从而在Mendelian随机化分析中恢复有效工具变量。

**💡 创新点**

创新点在于：①利用多环境的分布不变性理论证明可以在一般可微混合模型下识别可变工具的可变成分；②给出多项式和可微混合下的可识别性定理；③通过独立性约束保证可变成分可用于方差降低，并提供相应的理论支持。

**🔧 技术方法**

技术手段包括：自编码器（可微或正则化的流网络）结合最大均值差异（MMD）不变性损失和Hilbert–Schmidt独立性判别（HSIC）损失；以及传统的2SLS、MR‑Egger等因果估计方法。

**📊 数据集**

实验使用All of Us Research Hub中来自东亚和非洲祖先的652个SNP，并通过ICA构造半合成数据；此外还使用全模拟数据来验证理论。

**📈 对比分析**

与标准2SLS、MR‑Egger及其通过种群指示器调整的版本相比，提出的2SLS(W)和PO(V)-2SLS(W)在多种混合函数（多项式、可逆MLP）下均保持接近零的偏差，且在使用V进行协变量调整时显著降低方差；误差在维度错配和缺少独立性约束时会显著增大。

**⚠️ 局限性**

局限性包括：需要满足V在不同环境中的充分变异性假设；对多环境数据的依赖；当不变性假设被违反或高维稀疏映射未被显式建模时，方法效果可能下降；实验中对假设的检验仍有限。

---

## 330. A Causal Framework for Estimating Heterogeneous Effects of On-Demand Tutoring

**arXiv ID:** 2602.19296 | [PDF](https://arxiv.org/pdf/2602.19296v1)

**作者:** Kirk Vanacore `[一作]` (Cornell University), Rene Kizilcec `[通讯]` (Cornell University)

**通讯引用:** 7149 | [OpenAlex ID](https://openalex.org/A5071778778)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

建立了一套可扩展的因果推断框架，用以评估嵌入式按需人类辅导对中学数学学习者即时题目表现和近转移表现的影响。

**💡 创新点**

创新点在于将深度知识追踪（DKT）生成的时变知识状态与双重稳健的因果森林（Causal Forest）结合，解决自选性与时间变化混杂问题，并实现会话级别的因果效应分布估计。

**🔧 技术方法**

使用技术包括 LSTM‑基础的 Deep Knowledge Tracing、Generalized Random Forest（因果森林）与 Robinson 分解、AIPW 加权、以及多项鲁棒性和敏感性检验。

**📊 数据集**

数据集来自 Eedi 数学学习平台，包含 2,585 名学生、852,274 次题目尝试、1,221 次人类辅导会话（共 5,000+ 课堂级别交互）。

**📈 对比分析**

方法通过对比未请求辅导者的控制组，得到平均处理效应（ATE）为即时问题正确率提升 4.01pp、近转移提升 2.73pp；效应在加入外部协变量、扩展对照样本及安慰剂检验后保持稳健，CATE 方差揭示显著的效应异质性。

**⚠️ 局限性**

局限性包括仅采用二元处理变量，未对对话内容进行细粒度分析导致因果机制不清；结果仅覆盖即时与近转移效应，未评估长期学习轨迹；以及框架仅在单一平台上验证，推广性待进一步验证。

---

## 331. A Logic-Reuse Approach to Nibble-based Multiplier Design for Low Power Vector Computing

**arXiv ID:** 2602.19007 | [PDF](https://arxiv.org/pdf/2602.19007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 332. LLM-Assisted Replication for Quantitative Social Science

**arXiv ID:** 2602.18453 | [PDF](https://arxiv.org/pdf/2602.18453v1)

**作者:** So Kubota `[一作]` (Tohoku University), Yuki Nakamura `[通讯]` (University of Tokyo)

**通讯引用:** 9266 | [OpenAlex ID](https://openalex.org/A5055328011)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用大型语言模型（LLM）开发自动化复制系统，能从论文方法描述和数据表中生成可执行代码，运行后与原始结果对比并迭代调试；以社会学经典论文Bryson (1996)及其使用的1993年GSS数据为案例，演示系统能在多次迭代中接近原表格与图形结果；

**💡 创新点**

提出了基于LLM的“复制编译器”思路，首次将自然语言方法转化为代码并循环执行、比较、修正，实现了自动化复制与差异报告；

**🔧 技术方法**

核心技术包括OpenAI GPT‑5.2大语言模型、Python代码执行沙箱、基于输入的结构化规格生成（表/图摘要、指令摘要）、数值对齐评分与迭代调优；

**📊 数据集**

使用了公开可获取的社会学调查数据1993年GSS（简化版CSV）及其代码簿；

**📈 对比分析**

与人工复制（约45小时、两位研究生完成）对比，AI系统在Table 2、Table 3和Figure 1的重现精度相当或更优；在Table 1上迭代100次后最佳对齐得分为74/100，表明系统能在多次迭代中逐步逼近原结果；

**⚠️ 局限性**

局限性包括：只能处理单一横截面数据和标准OLS/Logit模型；不支持多数据集合并、面板、SEM等复杂方法；缺乏详细错误诊断与原因解释；对LLM的误判和不足仍会导致重复失败；且重现仅保证计算可重复性，不能确认结论的真实性或外部有效性。

---

## 333. Poster: Privacy-Preserving Compliance Checks on Ethereum via Selective Disclosure

**arXiv ID:** 2602.18539 | [PDF](https://arxiv.org/pdf/2602.18539v1)

**作者:** Supriya Khadka `[一作]` (Coventry University), Sanchari Das `[通讯]` (George Mason University)

**通讯引用:** 4231 | [OpenAlex ID](https://openalex.org/A5003726306)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了基于以太坊的选择性披露框架，允许用户在不泄露身份信息的前提下完成合规检查。

**💡 创新点**

创新点在于将零知识证明（zk‑SNARK）与客户端证明模型相结合，实现了可撤销的授权生命周期（Grant‑Verify‑Revoke）。

**🔧 技术方法**

使用Circom编写电路、SnarkJS生成zk‑SNARK、Solidity编写验证与访问登记合约，并部署于Sepolia/Arbitrum。

**📊 数据集**

论文未使用公开数据集，而是基于年龄验证的示例场景进行原型实现。

**📈 对比分析**

性能评估显示：客户端证明生成耗时<200 ms；主网上链验证消耗≈240,512 gas，约$15；在Arbitrum等L2上成本降至<$0.5，验证过程仅为一次性查表。

**⚠️ 局限性**

局限性包括：主网高昂的 gas 成本、潜在的 mempool 前置攻击、跨 dApp 聚类导致的匿名性泄露，以及对普通 Web 环境的安全性依赖。

---

## 334. The Path to Conversational AI Tutors: Integrating Tutoring Best Practices and Targeted Technologies to Produce Scalable AI Agents

**arXiv ID:** 2602.19303 | [PDF](https://arxiv.org/pdf/2602.19303v1)

**作者:** Kirk Vanacore `[一作]` (Cornell University), Jeremy Roschelle `[通讯]` (Digital Promise)

**通讯引用:** 12911 | [OpenAlex ID](https://openalex.org/A5028682942)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了人类辅导、传统ITS（如知识跟踪、情感检测、知识图谱）与新兴生成式AI对话式辅导的融合，提出了保持、改变、聚焦、研究（Keep‑Change‑Center‑Study）框架来指导未来对话式辅导系统的设计与评估。

**💡 创新点**

创新点在于将已有的教学实践与ITS技术与生成式AI的即时内容生成、对话反馈能力相结合，形成一个以对话为中心、既保持传统精准适应性又利用LLM生成式能力的全新设计范式。

**🔧 技术方法**

技术手段包括：①传统ITS核心技术（知识跟踪、行为与情感检测、知识图谱、内部-外部循环架构）；②生成式AI大语言模型（LLM）用于实时内容生成、对话式提示与诊断；③检索增强生成（RAG）与混合提示工程用于将传统知识结构与LLM输出结合。

**📊 数据集**

本文并未使用新的实验数据集，而是基于已有的系统案例（如Khanmigo、Rori、AutoTutor、OpenTutor等）以及公开的ITS评测数据和教育AI基准（如AI‑for‑Education.org）进行文献综述与案例分析。

**📈 对比分析**

比较方法主要是文献回顾与案例对比，未给出统一的实验指标或量化性能。文中指出，现有对话式辅导系统普遍缺乏大规模随机对照试验、长期学习成效与公平性评估，效果评估多为小样本或质性报告。

**⚠️ 局限性**

局限性包括：①LLM易产生幻觉与不准确反馈，影响教学质量；②对学生知识状态与情感的实时诊断仍不够精准；③缺乏与人类辅导者协作的可验证模型；④系统设计与实施缺乏系统化的实验验证，难以量化其对学习成效与动机的真实影响。

---

## 335. TAG: Thinking with Action Unit Grounding for Facial Expression Recognition

**arXiv ID:** 2602.18763 | [PDF](https://arxiv.org/pdf/2602.18763v1)

**作者:** Haobo Lin `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14849 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 TAG 框架，在面部表情识别任务中通过行动单元（AU）实现可验证的多模态推理。

**💡 创新点**

创新点在于：① 将推理过程明确约束在 AU 相关的面部区域；② 通过两阶段训练（SFT + AU 关注的 RL）将视觉证据与语言解释绑定；③ 构建了 310k 条 AU‑grounded 推理轨迹数据集 TAG‑310k。

**🔧 技术方法**

采用视觉‑语言模型 Qwen2.5‑VL‑7B，结合 AU 检测器 GraphAU，使用监督微调、GRPO 强化学习和基于 AU IoU 的奖励机制来实现 AU‑grounded 推理。

**📊 数据集**

使用 RAF‑DB、FERPlus、AffectNet 三大公开表情数据集，构建 TAG‑310k 推理轨迹；同时在这些数据集上评估模型性能。

**📈 对比分析**

与开源与闭源 VLM（如 InternVL3、GPT‑5）以及 FER 专用方法对比，TAG 在统一训练设置下平均准确率达 74.34%（比 InternVL3 高 9.23%），在每个数据集微调后达到 83.78% 的平均准确率，稳步超过所有基线。

**⚠️ 局限性**

局限性包括：对外部 AU 检测器的依赖，可能引入检测误差；奖励设计需谨慎以避免过度依赖单一检测器；在跨域迁移或非典型表情场景下表现仍有提升空间。

---

## 336. Face Presentation Attack Detection via Content-Adaptive Spatial Operators

**arXiv ID:** 2602.18965 | [PDF](https://arxiv.org/pdf/2602.18965v1)

**作者:** Shujaat Khan `[一作]` (King Fahd University of Petroleum and Minerals), Shujaat Khan `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 1385 | [OpenAlex ID](https://openalex.org/A5026671975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于内容自适应空间算子（involution）的轻量级单帧RGB面部防欺骗模型CASO-PAD；

**💡 创新点**

通过在MobileNetV3骨干中插入位置自适应、通道共享的分组involution，显著提升局部伪造特征捕获能力，同时保持极低的模型尺寸与计算量；

**🔧 技术方法**

核心技术包括：MobileNetV3-Large轻量网络、分组involution（GI）算子、卷积与SE模块、交叉熵训练、灰度中心裁剪与多尺度输入；

**📊 数据集**

在Replay-Attack、Replay-Mobile、OULU-NPU、ROSE-Youtu以及SiW-Mv2（Protocol‑1）等公开数据集上进行训练与评估；

**📈 对比分析**

与多种先进方法（包括HybridNet、Deformable Convolution、SiW-Mv2等）对比，CASO-PAD在RA、RM上实现100%精度/0% EER，OULU-NPU 99.68%精度、0.44% HTER，ROSE-Youtu 98.90%精度、0.82% HTER，SiW-Mv2 95.45%精度、3.11% HTER，显著优于同类轻量级模型；

**⚠️ 局限性**

局限性包括：仅基于单帧RGB输入，对高度动态遮挡或极端光照条件仍可能受限，且在大规模真实世界部署前需进一步验证跨设备与跨文化泛化能力；

---

## 337. FruitTouch: A Perceptive Gripper for Gentle and Scalable Fruit Harvesting

**arXiv ID:** 2602.18991 | [PDF](https://arxiv.org/pdf/2602.18991v1)

**作者:** Ruohan Zhang `[一作]` (University of Illinois), Wenzhen Yuan `[通讯]` (University of Illinois)

**通讯引用:** 4094 | [OpenAlex ID](https://openalex.org/A5055947140)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了 FruitTouch——一种采用单摄像头+镜面共用光学布局的紧凑式抓取器，集成高分辨率视觉触觉、力估计、滑动检测和软度评估，并在实验室环境下验证其在樱桃番茄和草莓采摘中的闭环控制性能。

**💡 创新点**

创新点包括：1) 通过单摄像头与双镜面实现双指触感，显著降低成本与体积；2) 将高分辨率触觉与实时力、滑动、软度检测无缝集成到同一抓取器；3) 在采摘过程中实现基于力与滑动反馈的闭环控制，提升采摘成功率与抓取柔和性。

**🔧 技术方法**

主要技术手段为 GelSight 风格视觉触觉传感器、光学镜面布局与 LED 光照优化、基于 MLP 的几何重建、线性电流-力模型、标记位移分解与滑动检测、轻量级 ranker 对软度的成对比较预测。

**📊 数据集**

使用了自制钢球、果实模型和硅胶果实复制品收集的触觉图像与力测量数据；RealSense D435 与 YOLOv5 检测得到的果实位置；以及四个硬度等级的水果模型（草莓、覆盆子、樱桃番茄）作为软度评估的数据集。

**📈 对比分析**

与 ATI Nano17 真值进行对比，力估计 R²≈0.95、MAE≈0.27 N、MAPE≈3%；滑动检测精度≈0.725、召回≈0.661、F1≈0.692；在采摘实验中，滑动+力闭环控制成功率 100%（相较于开环 58.3%/12.5%），并显著降低了平均抓取力与方差，验证了闭环策略的有效性。

**⚠️ 局限性**

局限性：仅在实验室光照条件下验证，缺乏户外真实田间环境的鲁棒性；触觉传感器对湿度或污垢敏感；目前仅针对小型水果（樱桃番茄、草莓）测试，尚未验证对大果或重物的适用性。

---

## 338. The Price Is Not Right: Neuro-Symbolic Methods Outperform VLAs on Structured Long-Horizon Manipulation Tasks with Significantly Lower Energy Consumption

**arXiv ID:** 2602.19260 | [PDF](https://arxiv.org/pdf/2602.19260v1)

**作者:** Timothy Duggan `[一作]` (Tufts University), Matthias Scheutz `[通讯]` (Tufts University)

**通讯引用:** 9154 | [OpenAlex ID](https://openalex.org/A5044523801)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对比了在结构化长时序机械臂任务中，微调的视觉-语言-动作模型（VLA）与神经符号模型（NSM）的性能与能耗。

**💡 创新点**

首次将VLA与NSM在同一任务中直接对比，并系统评估能耗差异。

**🔧 技术方法**

使用VLA的π_0（OpenPi）模型、LoRA微调、神经符号架构结合PDDL规划与扩散控制；以及VLM如GPT‑5、Qwen、PaliGemma。

**📊 数据集**

利用300条完整的三到四盘塔汉诺塔演示和仅50条堆叠演示的数据集。

**📈 对比分析**

通过在三盘和四盘塔汉诺塔任务、单个搬运任务上测量成功率、任务进度、GPU/CPU功耗与能耗，结果显示NSM成功率95%/78%且能耗≈2% VLA；VLA在三盘任务仅34%成功。

**⚠️ 局限性**

VLA对轨迹质量敏感、能耗高、难以泛化，实验仅覆盖特定仿真环境，缺少真实世界验证。

---

## 339. DoAtlas-1: A Causal Compilation Paradigm for Clinical AI

**arXiv ID:** 2602.19158 | [PDF](https://arxiv.org/pdf/2602.19158v1)

**作者:** Yulong Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Imran Razzak `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 10553 | [OpenAlex ID](https://openalex.org/A5033585021)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了医学因果编译范式，并实现了 DoAtlas‑1 系统，能将 754 篇临床研究的 1,445 个效应核（effect kernels）转化为可执行的因果估计对象；同时使用 10,000 例 Human Phenotype Project（HPP）数据进行外部验证。

**💡 创新点**

创新点包括：① 将叙事式医学知识转化为结构化、可执行的因果估计；② 引入标准化的效应规范化（canonicalization）和时间窗口对齐；③ 通过冲突感知图（conflict‑aware graph）系统化检测和记录证据不一致；④ 用真实世界数据（HPP）做大规模外部验证，提升可信度；⑤ 支持六类可执行查询（do‑calculus、counterfactual、trajectory、CATE、mediation、joint），实现可审计、可验证的决策支持。

**🔧 技术方法**

技术实现主要包括：基于 SCM 的因果抽象与五元组估计器；效应标准化算子 N（对测量尺度、时间窗口做确定性变换）；冲突检测谓词（方向性、区间不重叠、高异质性）；多边缘估计图（multi‑edge atlas）构建；查询接口与可执行性判定；外部验证框架与可靠性信号融合；以及增量编译管道。

**📊 数据集**

使用的主要数据集为：① 754 篇已发表的临床研究（心血管、糖尿病、代谢疾病）构成的效应核；② Human Phenotype Project（10,000 名参与者）的多模态生理数据，用于对已编译效应进行方向性和效应规模的一致性检验。

**📈 对比分析**

方法评估：效应核抽取字段 F1 0.982，canonicalization 一致性 98.5%；查询可执行率 80.5%，在 1,110 个桶中 42 个（3.8%）被标记冲突；冲突检测精确率与召回率均为 100%；与无规范化基线相比，查询准确率提升 28.5pp；与基线 2（文本+粗粒度队列）相比，外部验证的方向一致性从 0.712 提升至 1.000，误报率降为 0%。

**⚠️ 局限性**

局限性：① 仅覆盖心血管、糖尿病和代谢疾病三大领域，尚未扩展至肿瘤、神经、免疫等；② 对效应类型的可聚合性要求较高，非可聚合的测量（如 HR vs. RR）默认不合并，可能导致信息缺失；③ 外部验证仅基于 HPP，未涵盖所有人群与疾病；④ 需手工标注与质量控制，仍有人工干预；⑤ 对极端罕见疾病和少数族裔人群的研究数据稀缺，影响可靠性。

---

## 340. PoseCraft: Tokenized 3D Body Landmark and Camera Conditioning for Photorealistic Human Image Synthesis

**arXiv ID:** 2602.19350 | [PDF](https://arxiv.org/pdf/2602.19350v1)

**作者:** Zhilin Guo `[一作]` (University of Cambridge), Cengiz Oztireli `[通讯]` (Google)

**通讯引用:** 2252 | [OpenAlex ID](https://openalex.org/A5046322671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PoseCraft——一种基于稀疏 3D 标志点和相机外参的离散令牌条件扩散框架，并配合 RigCraft 的多视角三角化与 GenHumanRF 数据生成流程，实现无模板、无网格绑定的高保真人像合成。

**💡 创新点**

创新点在于用离散 3D 令牌直接注入扩散模型，消除了 2D 投影不确定性，显著提升姿态与视角一致性，同时保持身份与高频细节，突破传统 2D 控制与基于模板的 3D 方法的局限。

**🔧 技术方法**

采用 Latent Diffusion (VAE+UNet)、3D Control Tokenizer（球面谐波编码、傅里叶位置编码）、多视角三角化与 Savitzky–Golay 时序平滑、以及交叉注意力等技术，构建完整的训练与推理管线。

**📊 数据集**

使用 ActorsHQ 数据集，并通过 GenHumanRF 生成约 440k 张全身高分辨率图像（512×384），结合 HumanRF 进行体素化渲染，为每个演员提供训练/测试拆分。

**📈 对比分析**

与多种 2D 关键点引导扩散方法（CFLD、ControlNet、T2I‑Adapter、AnimateAnyone、CHAMP）以及基于体素的 3D 渲染 SOTA Animatable Gaussians 进行对比；在 PSNR、SSIM、LPIPS、FID 等指标上显著优于 2D 方法，且与 3D 渲染方法持平或更好。

**⚠️ 局限性**

局限性包括：仅针对单个身份训练，缺乏零样本或少样本泛化；对松散或多层衣物的捕捉仍易出现“幽灵”效果；手部细节未显式建模，手指姿态缺失。

---

## 341. RAmmStein: Regime Adaptation in Mean-reverting Markets with Stein Thresholds -- Optimal Impulse Control in Concentrated AMMs

**arXiv ID:** 2602.19419 | [PDF](https://arxiv.org/pdf/2602.19419v1)

**作者:** Pranay Anchuri `[一作]` `[通讯]` (Offchain Labs), Pranay Anchuri (Offchain Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究如何在Uniswap V3等AMM中对集中流动性进行冲击式重平衡，以最大化手续费收益并降低重平衡成本。

**💡 创新点**

将LP重平衡建模为冲击控制问题，推导HJB‑QVI，并引入Stein信号（OU均值回归速率）作为市场 regime 先验，利用DDQN学习动态懒惰边界；同时通过深度强化学习实现对重平衡时机的自适应决策。

**🔧 技术方法**

使用Deep Reinforcement Learning（Double DQN）求解HJB‑QVI，特征工程包括OU参数、价格偏差、边界距离等；构建仿真环境并训练Q‑网络。

**📊 数据集**

基于Coinbase 1 Hz的ETH‑USD 高频交易数据（约 6.8 M笔交易），分为训练、验证和测试集。

**📈 对比分析**

与Merlin（理论上最佳）、Bedivere（被动）、Lancelot（贪婪）和Galahad（LSTM）四种基线对比，RAmmStein在测试集上实现净ROI 0.7159%，比Lancelot高26%，并将重平衡次数降低67%，在不同Gas成本下保持正收益。

**⚠️ 局限性**

仅针对单一资产对，假设OU参数在估计窗口内局部平稳；未考虑MEV风险或Gas价格预测；动作空间仅为二元，未考虑连续宽度调整，未来可扩展。

---

## 342. Benchmarking Computational Pathology Foundation Models For Semantic Segmentation

**arXiv ID:** 2602.18747 | [PDF](https://arxiv.org/pdf/2602.18747v1)

**作者:** Lavish Ramchandani `[一作]`, Tijo Thomas `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对10种病理专用基础模型在四个公开组织学图像数据集上的像素级语义分割性能进行无解码器的基准测试。

**💡 创新点**

创新点在于利用模型注意力图作为像素级特征并用XGBoost分类器进行快速可解释评估，同时展示多模型注意力拼接可显著提升性能。

**🔧 技术方法**

采用Transformer基础模型的自注意力头输出、双线性插值上采样、XGBoost监督分类器以及多模型特征拼接技术。

**📊 数据集**

使用GlaS、OCELOT Tissue、LyNSeC 2和BCSS四个公开数据集进行评估。

**📈 对比分析**

与单一模型相比，CONCH表现最佳，PathDino紧随其后，且CONCH+PathDino+CellViT的拼接模型平均Dice提升约7.95%，显著优于单模型。

**⚠️ 局限性**

局限在于仅依赖注意力特征且未进行模型微调，数据集覆盖有限，且对更大规模或跨科室数据的泛化尚未验证。

---

## 343. Interaction Theater: A case of LLM Agents Interacting at Scale

**arXiv ID:** 2602.20059 | [PDF](https://arxiv.org/pdf/2602.20059v1)

**作者:** Sarath Shekkizhar `[一作]`, Adam Earle `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Moltbook社交平台上对数千个LLM代理的帖子和评论进行大规模、无监督的交互质量分析，评估其信息贡献、相关性和讨论深度。

**💡 创新点**

首次将词汇熵、自相似距离、词典与嵌入特异性以及LLM评判结合起来，对代理交互进行量化评估，揭示“交互戏剧”现象——表面活跃却缺乏实质内容。

**🔧 技术方法**

使用词汇熵和自相似距离测量代理输出多样性；Jaccard词典特异性与OpenAI嵌入余弦相似度评估评论与贴文的相关性；Anthropic LLM 作为评判机对评论进行质量分类和信息贡献评分。

**📊 数据集**

利用公开的Moltbook数据集：800,730 条帖子、3,530,443 条评论、78,280 条代理档案，时间范围为2026年1月27日至2月17日。

**📈 对比分析**

通过与随机贴文对照、长度、嵌入特异性等多维度比较，发现 65% 评论无明显特异性，信息饱和快速下降（第15条评论后仅 ~32% 新信息），LLM 评判显示 28% 为垃圾 22% 离题，表明传统活跃度指标误导，需引入信息论与语义指标。

**⚠️ 局限性**

局限性包括：仅分析无目标社交环境的代理交互，缺乏代理内部模型/提示信息，短评论特异性难评估，依赖特定嵌入与评判模型，且仅覆盖短期（3 周）数据，无法观察长期适应或演化。

---

## 344. Automated Generation of Microfluidic Netlists using Large Language Models

**arXiv ID:** 2602.19297 | [PDF](https://arxiv.org/pdf/2602.19297v1)

**作者:** Jasper Davidson `[一作]` (University of Utah), Pierre-Emmanuel Gaillardon `[通讯]` (Primis AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用大型语言模型（LLM）将自然语言描述的微流控设备规格自动转换为系统级结构Verilog netlist，展示了微流控设计自动化的可行性。

**💡 创新点**

创新点在于首次将LLM应用于微流控设计，实现在自然语言到硬件描述语言的端到端转换，并基于先前的HDL代码生成研究提出了新方法。

**🔧 技术方法**

采用的技术包括大规模预训练语言模型（如GPT系列）与HDL代码生成技术，结合微流控领域的专业知识。

**📊 数据集**

使用了代表典型微流控设计的实际基准数据集，包含若干常见实验室微流控设备的规格说明。

**📈 对比分析**

通过将自动生成的Verilog netlist与人工手工设计进行功能和语法对比，实验结果显示生成的 netlist 功能正确，平均语法准确率为 88%。

**⚠️ 局限性**

局限性包括：准确率尚未达到完全可靠的水平，模型对复杂或未见过的设计仍可能产生错误，且缺乏对更大规模、跨域设计的泛化验证。

---

## 345. IDSelect: A RL-Based Cost-Aware Selection Agent for Video-based Multi-Modal Person Recognition

**arXiv ID:** 2602.18990 | [PDF](https://arxiv.org/pdf/2602.18990v1)

**作者:** Yuyang Ji `[一作]` (Drexel University), Feng Liu `[通讯]` (Drexel University)

**通讯引用:** 11476 | [OpenAlex ID](https://openalex.org/A5100415332)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于强化学习的成本感知模型选择器IDSelect，能够在视频人像识别中自适应挑选面部、步态、身体模型以平衡准确率和算力。

**💡 创新点**

创新点在于将跨模态模型组合视为可学习的策略，并通过奖励平衡准确率与计算成本，实现输入条件下的模型组合选择。

**🔧 技术方法**

使用了actor-critic强化学习、Lagrangian预算控制、熵正则化等技术，同时保持后端特征提取模型冻结，只训练选择策略。

**📊 数据集**

在CCVID和MEVID两大视频人像识别数据集上进行评估。

**📈 对比分析**

与固定融合、质量引导融合等基线相比，IDSelect在CCVID上实现95.9% Rank‑1、94.6% mAP，仅消耗53.7 GFLOPs（比QME降低92%算力），在MEVID上也获得1.7×算力节省。

**⚠️ 局限性**

局限性在于需要预先构建多样化模型池并保持多模型占用存储；此外在极端低质量情况下面部信息可能无效，导致跨模态协同受限。

---

## 346. Measuring Validity in LLM-based Resume Screening

**arXiv ID:** 2602.18550 | [PDF](https://arxiv.org/pdf/2602.18550v1)

**作者:** Jane Castleman `[一作]` (Princeton University), Aleksandra Korolova `[通讯]` (Princeton University)

**通讯引用:** 6226 | [OpenAlex ID](https://openalex.org/A5005219368)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套系统化的框架，用来在已知真值的人工合成简历对中评估大型语言模型（LLM）在简历筛选中的有效性和公平性；

**💡 创新点**

创新点在于：①利用基于工作描述的结构化资格提取，生成基准简历及其高低资格和等价变体，形成不易被模型训练集污染的真值对；②从心理测量学与系统评估角度引入判别有效性与准则有效性的度量；③通过允许模型拒绝选择（⊥）来区分不合理拒绝与错误选择，进一步揭示模型在微小差异判断上的弱点；

**🔧 技术方法**

技术主要包括：大规模LLM（Claude、Gemini、GPT-4o-mini、GPT-5等）作为数据处理工具；prompt设计与多模型生成；对比度指标（criterion validity、discriminant validity、over‑assessment、unjustified abstention/selection）等；

**📊 数据集**

数据集：基于Greenhouse抓取的186个职位描述，涵盖25类；对每个职位生成数千对合成简历；同时在LinkedIn、Indeed及Reddit实测数据上进行泛化检验；

**📈 对比分析**

比较方法：在合成对上计算每个模型的criterion validity（正确选择比例）和discriminant validity（拒绝比例、over‑assessment率）；实验显示大模型规模越大有效性越好（如GPT-5的criterion validity≈0.98），但在k=1（仅有细微差别）时多数模型低于0.90；公平性方面，部分模型在等价对中显示对少数族裔的偏好或过度校正，over‑assessment率在0.14–0.24之间；

**⚠️ 局限性**

局限性：①合成简历虽可避免训练集污染，但与真实简历仍有差异，无法完全反映自然语言多样性与模糊性；②仅评估四个族群（黑人/白人、男/女），未涵盖更广泛的身份维度；③只关注pairwise比较，未直接评估top‑k排序效果；④模型在生成简历过程中可能引入错误，影响下游评估结果。

---

## 347. The Landscape of AI in Science Education: What is Changing and How to Respond

**arXiv ID:** 2602.18469 | [PDF](https://arxiv.org/pdf/2602.18469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 348. Do LLMs and VLMs Share Neurons for Inference? Evidence and Mechanisms of Cross-Modal Transfer

**arXiv ID:** 2602.19058 | [PDF](https://arxiv.org/pdf/2602.19058v1)

**作者:** Chenhang Cui `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60708 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）与视觉-语言模型（LVLM）共享的神经元子空间，并提出共享神经元低秩融合（SNRF）框架，将LLM成熟的推理能力高效迁移到LVLM；

**💡 创新点**

创新点在于：①发现跨模态推理任务中大量神经元激活重叠；②通过激活放大揭示共享神经元的概念层级因果作用；③设计仅在共享子空间内进行低秩矩阵更新的参数高效迁移方法；

**🔧 技术方法**

使用了神经元激活分析、因果放大实验、SVD低秩近似、掩码投影更新以及对比实验评测等技术；

**📊 数据集**

实验数据集包括 MathVista、MME、POPE、ScienceQA、MMMU/Pro 以及其他数学与视觉推理任务集；

**📈 对比分析**

与原始LVLM、链式推理SFT、RL对齐、线性融合、DARE、FRANK 等方法对比，SNRF 在推理相关指标上提升了 1–5 分，且保持感知与幻觉性能；

**⚠️ 局限性**

局限性包括：需预先有配对的LLM和LVLM，难以跨大结构差异；对多语言或细粒度多模态推理挑战的适用性待验证；

---

## 349. Optimizing ID Consistency in Multimodal Large Models: Facial Restoration via Alignment, Entanglement, and Disentanglement

**arXiv ID:** 2602.18752 | [PDF](https://arxiv.org/pdf/2602.18752v1)

**作者:** Yuran Dong `[一作]` (Wuhan University), Mang Ye `[通讯]` (Wuhan University)

**通讯引用:** 11918 | [OpenAlex ID](https://openalex.org/A5008999954)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个训练无关、可插拔的编辑框架 EditedID，用于提升多模态大模型在面部编辑任务中的身份一致性。

**💡 创新点**

创新点在于三大技术的结合：自适应混合实现跨源分布偏差校正，混合求解器（DDIM + DPM‑Solver++）实现身份与细节的分离与融合，注意力门控实现单元素结构与多元素交互的平衡，从而实现身份与编辑属性的双重保持。

**🔧 技术方法**

技术手段包括扩散模型反演、Null‑Text 预优化、跨源自注意力与自注意力替换、语义掩模与权重融合、混合采样器调度等。

**📊 数据集**

实验主要使用公开人脸数据集（如 FFHQ 等）进行无监督验证，未公开大量专门的人脸数据集；对工业与学术多模态编辑模型在公开数据集上的效果做对比。

**📈 对比分析**

与现有身份保留方法和多模态大模型（如 In‑ContextEdit、GPT‑4o Plus 等）进行对比。实验表明 EditedID 在 ID‑Sim 上平均提升 0.27、CLIP‑S 提升 2.43、I‑Reward 提升 0.27；在单人和多人场景下推理时间约 4.2 秒，保持了较低的计算成本。

**⚠️ 局限性**

局限性包括：对极端遮挡或极端角度仍可能出现细节缺失；仅针对扩散模型的反演，未覆盖跨模态（如视频、三维）实时推理；对多语言文本控制的鲁棒性尚未系统评估。

---

## 350. TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics

**arXiv ID:** 2602.19313 | [PDF](https://arxiv.org/pdf/2602.19313v1)

**作者:** Shirui Chen `[一作]` (University of Washington), Ranjay Krishna `[通讯]` (University of Washington)

**通讯引用:** 12894 | [OpenAlex ID](https://openalex.org/A5032451496)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TOPReward，一种利用预训练视频视觉语言模型内部token概率进行零样本进度估计的奖励模型，能够在多种机器人平台的真实操纵任务中提供连续、可解释的奖励信号。

**💡 创新点**

创新点在于跳过传统文本生成的数值预测，直接从VLM的token概率中提取完成度（True/False），从而克服数值生成不准确的问题，并构建了涵盖130+任务的跨机器人基准来评估奖励模型。

**🔧 技术方法**

采用了预训练视频视觉语言模型（Qwen3‑VL、Molmo2、Gemini），通过对完成度标记的token概率进行log计算、min‑max归一化，并将进度信号用于成功检测、优势加权行为克隆等下游任务。

**📊 数据集**

使用了包含Franka、YAM、SO‑100/101等四种机器人平台的130+多任务真实操纵数据，提供了子任务级时间标注，并与Open X‑Embodiment（OXE）数据集进行对比评估。

**📈 对比分析**

与现有无训练方法GVL在VOC（Value‑Order Correlation）指标上对比，TOPReward在OXE上取得0.857 VOC（Qwen3‑VL），在TOPRewardBench上平均VOC 0.947，成功检测ROC‑AUC提升至0.654，优势加权行为克隆比传统行为克隆提升至最高10/10成功率。

**⚠️ 局限性**

局限性包括：受限于底层VLM的视觉感知能力，对细粒度空间推理任务可能产生噪声；min‑max归一化仅在轨迹内可比较，无法直接比较不同轨迹的绝对进度；对Chat模板敏感，某些模型在使用时效果下降。

---

## 351. Reasoning Capabilities of Large Language Models. Lessons Learned from General Game Playing

**arXiv ID:** 2602.19160 | [PDF](https://arxiv.org/pdf/2602.19160v1)

**作者:** Maciej Świechowski `[一作]` (Grail Team), Jacek Mańdziuk `[通讯]` (Warsaw University of Technology)

**通讯引用:** 2241 | [OpenAlex ID](https://openalex.org/A5073814691)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将一般游戏玩法（GGP）框架与游戏描述语言（GDL）结合，构建了一套可验证的逻辑推理基准，用来评估大型语言模型（LLM）的符号推理能力；

**💡 创新点**

创新点在于：①首次将GGP/GDL用作LLM逻辑推理的可交叉验证平台；②对四类任务（下一步状态、合法动作、多步状态、多步动作-状态）进行细粒度评估；③引入符号混淆实验以分离语言语义与结构推理；④利用40个结构特征量化游戏难度并与模型表现关联；

**🔧 技术方法**

技术包括：GDL前置语义解析、基于Jaccard指数和全一致率的评估指标、符号混淆（占位符、词典、随机字符串）、线性相关分析、Wilcoxon符号秩检验；

**📊 数据集**

使用了35款来自公开存储库的GDL游戏（共86个原始文件），覆盖多种状态复杂度和规则深度；

**📈 对比分析**

比较方法是对四个模型（Gemini 2.5 Pro/Flash、Llama 3.3 70B、GPT‑OSS 120B）在四个任务上分别计算平均Jaccard和全一致率，并对不同推理步长及混淆类型进行统计；性能上，Gemini 2.5 Pro在单步任务中>95%全一致率，三步/多步任务仍维持60–80%全一致率，弱模型（Llama 3.3）多步任务往往低于40%；

**⚠️ 局限性**

局限性包括：①长推理链（尤其是同时生成动作与状态）容易出现误差累积导致最终状态错误；②对深层递归规则与大量规则集合的敏感度高；③当前评估仅覆盖有限模型与游戏样本，缺乏可扩展性；④实验未探究更细粒度的外部符号辅助或自监督训练对性能的提升作用。

---

## 352. When World Models Dream Wrong: Physical-Conditioned Adversarial Attacks against World Models

**arXiv ID:** 2602.18739 | [PDF](https://arxiv.org/pdf/2602.18739v1)

**作者:** Zhixiang Guo `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 99147 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种针对生成式世界模型的白盒物理条件攻击方法PhysCond-WMA

**💡 创新点**

通过两阶段优化（质量保持引导+动量导向去噪）在物理条件通道注入微小扰动，既保持视觉质量，又能诱发语义、逻辑及决策级别错误

**🔧 技术方法**

利用扩散式生成式世界模型、梯度引导、动量EMA、语义一致性评估SSCD等技术

**📊 数据集**

在nuScenes数据集上进行实验，使用DriveDreamer与DriveDreamer2模型

**📈 对比分析**

与未攻击、单阶段攻击相比，双阶段方法在保持FID/FVD相近的前提下，目标攻击成功率提升至55%，非目标提升至32%；攻击导致3D检测mAP下降约1%，规划轨迹误差和碰撞率上升

**⚠️ 局限性**

仅验证了图像目标的攻击效果，尚未对多模态目标（视频、文本）展开；缺乏统一的世界模型安全评估标准

---

## 353. Effect of Patch Size on Fine-Tuning Vision Transformers in Two-Dimensional and Three-Dimensional Medical Image Classification

**arXiv ID:** 2602.18614 | [PDF](https://arxiv.org/pdf/2602.18614v1)

**作者:** Massoud Dehghan `[一作]` (Danube Private University), Amirreza Mahbod `[通讯]` (Danube Private University)

**通讯引用:** 2193 | [OpenAlex ID](https://openalex.org/A5044448889)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究在12个医学影像数据集上系统评估了Vision Transformer（ViT）中不同patch尺寸对分类性能的影响，探讨了从2D到3D的多模态数据；

**💡 创新点**

提出了针对医学图像的细粒度tokenization策略，即使用更小的patch尺寸能显著提升ViT的准确率，同时验证了不同尺寸的模型融合可进一步提高性能；

**🔧 技术方法**

使用预训练的ViT‑Small模型在单GPU环境下进行微调，采用ImageNet预训练权重，并实现了从1×1到28×28的多种patch分割方式；

**📊 数据集**

数据集来自MedMNIST V2，包括7个2D（如Breast、Retina、Derma等）和5个3D（如Adrenal、Fracture等）医学影像分类任务；

**📈 对比分析**

通过对每个patch尺寸的模型在Acc、Balanced Acc、AUC以及GFLOPs进行对比，结果显示较小的patch尺寸（1、2、4）在大多数数据集上提升了12.78%（2D）至23.78%（3D）的平衡准确率，融合1/2/4尺寸模型可进一步提升；

**⚠️ 局限性**

主要局限包括仅使用ViT‑Small模型、单GPU实验导致模型规模受限、3D小patch导致显著计算成本增加、以及仅在MedMNIST V2低分辨率数据集验证，未探究更高分辨率真实临床影像的泛化性；

---

## 354. The Invisible Gorilla Effect in Out-of-distribution Detection

**arXiv ID:** 2602.20068 | [PDF](https://arxiv.org/pdf/2602.20068v1)

**作者:** Harry Anthony `[一作]` (University of Oxford), Konstantinos Kamnitsas `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了深度学习模型在分布外（OOD）数据上的检测偏差，提出并验证了“Invisible Gorilla Effect”，即当OOD痕迹与模型关注区域（ROI）视觉相似时检测效果提升，反之下降；通过对40种OOD检测方法在7个基准（CheXpert、ISIC、MVTec‑AD等）上的大规模实验，发现特征基方法受影响最严重，并尝试了颜色抖动和子空间投影两种缓解策略。

**💡 创新点**

首次揭示了OOV检测中的视觉相似性偏差，量化了不同OOV痕迹颜色对检测性能的影响，并证明该效应普遍存在于多种模型与检测方法；同时提出并验证了一种基于主成分分析的“噪声子空间”投影缓解方案。

**🔧 技术方法**

使用了多种OOD检测技术（包括内部后置方法、外部重建/密度/分类方法、特征基与置信度基方法），以及PCA、Spearman相关分析等统计手段；并实现了颜色抖动数据增强与子空间投影的实验。

**📊 数据集**

CheXpert（胸部X光）、ISIC（皮肤病理图像）以及MVTec‑AD（工业缺陷图像）三大公开数据集，辅以合成covariate OOD（颜色图表、墨水标记、合成方块）以及颜色交换的对照实验。

**📈 对比分析**

在7个OOV基准上对40种方法进行比较，使用AUROC评估；结果显示特征基方法平均下降约7.1个百分点，置信度基方法约1.5个百分点；在ISIC和彩色图表实验中，颜色相似性越高，检测AUROC越高；子空间投影能显著减小相似/不相似颜色之间的性能差距。

**⚠️ 局限性**

仍未彻底解决Invisible Gorilla Effect，颜色抖动并非通用缓解手段，子空间投影在不同方法与数据集上的效果不一；实验未涉及大型预训练模型（如CLIP），其对该偏差的影响未知；且在提升OOV检测的同时，某些策略会显著降低ID准确率，存在性能权衡。

---

## 355. Design, Locomotion, and Control of Amphibious Robots: Recent Advances

**arXiv ID:** 2602.19077 | [PDF](https://arxiv.org/pdf/2602.19077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 356. When Pretty Isn't Useful: Investigating Why Modern Text-to-Image Models Fail as Reliable Training Data Generators

**arXiv ID:** 2602.19946 | [PDF](https://arxiv.org/pdf/2602.19946v1)

**作者:** Krzysztof Adamkiewicz `[一作]` (RPTU University Kaiserslautern-Landau), Andreas Dengel `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对 2022‑2025 年间公开发布的 13 种文本到图像扩散模型，构建大规模合成数据集，并用 ResNet‑50 等标准分类器仅在这些合成图像上训练，随后在真实 ImageNet‑1k 子集上评估其泛化性能，揭示随着 T2I 模型视觉质量提升，合成数据作为训练集的效果却在递减。

**💡 创新点**

发现了 T2I 生成质量与数据可训练性之间的逆向关系，并通过纹理、频率和分布多维度分析证明了：1) 高频细节缺失导致判别信息缺失；2) 结构与纹理不平衡，合成图像在全局布局上逼真但细节贫乏；3) 数据分布收敛，密度高但覆盖率低，导致训练模型在真实图像上表现不佳；此外提出应把多样性和自然频谱作为评估指标。

**🔧 技术方法**

使用低指导尺度（2.0）和 50 步去噪的扩散采样，生成 512×512 图像后下采到 256×256；对生成图像做深度估计（Depth Anything V2）提取结构，BagNet 9×9 提取纹理；对低/高频过滤的图像进行训练；在 CLIP‑ViT‑L 特征空间计算密度与覆盖率；对不同提示（类别名 vs 详细描述）进行对比；训练 ResNet‑50、ViT‑Ti、ConvNeXt‑Ti、Swin‑Ti 等网络。

**📊 数据集**

数据集：从 ImageNet‑1k 随机抽取 200 类，每类 500 张（总计 100k 张），用于合成数据生成；在每类 50 张真实图像上评估。合成图像来源于上述 13 种公开 T2I 模型。

**📈 对比分析**

对比方法：在真实训练集上训练得到基线 Accuracy；在每个 T2I 模型生成的合成集上训练得到 Synth→Real Accuracy；使用 GenEval、CLIPScore 测量文本对齐；使用深度/纹理/频率域评估不同信息维度对性能的影响；通过密度‑覆盖率曲线和 Real→Synth 交叉域测试量化分布失真。结果显示，随着模型年份递进，Synth→Real Accuracy 下降，说明即使视觉质量提升，合成数据的学习效果在递减。

**⚠️ 局限性**

局限性：仅在 ImageNet‑1k 的 200 类分类任务上验证，未涵盖检测、分割等其他视觉任务；使用公开模型，无法反映私有模型的性能差异；合成数据只基于单一 prompt（类别名或详细描述），未尝试多模态或动态提示；最后评估未覆盖训练效率和算力开销。

---

## 357. The Confusion is Real: GRAPHIC - A Network Science Approach to Confusion Matrices in Deep Learning

**arXiv ID:** 2602.19770 | [PDF](https://arxiv.org/pdf/2602.19770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 358. Urban mobility network centrality predicts social resilience

**arXiv ID:** 2602.18546 | [PDF](https://arxiv.org/pdf/2602.18546v1)

**作者:** Lin Chen `[一作]` (Hong Kong University of Science and Technology), James Evans `[通讯]` (University of Chicago)

**通讯引用:** 15266 | [OpenAlex ID](https://openalex.org/A5071261828)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用Safegraph大规模人类移动数据，对15个美国城市在COVID-19、2020年哈维龙卷风和2018年加州野火等三种城市冲击中的场所访客量与收入分层隔离度进行变化分析，研究场所的社会韧性。

**💡 创新点**

创新点在于构建基于场所类型的偏好网络并计算其特征向量中心度，发现该中心度能显著预测场所的隔离度变化和访客量变化，并提出将核心场所视为“深井”、外围场所视为“浅池”的比喻，用以解释灾害期间移动优先级的转变。

**🔧 技术方法**

主要技术包括：基于Safegraph数据的访客量和收入分层计数；通过访客比例偏好构建场所类型邻接矩阵；计算特征向量中心度；运用OLS回归评估中心度对隔离度与访客量变化的解释力；对比中心度模型与传统前期访客量、隔离度和桥梁指数的模型。

**📊 数据集**

使用的数据集包括Safegraph Patterns（按CBG聚合的月度访客数据）、Safegraph Places（场所位置信息与NAICS分类）以及美国人口普查局2019年五年ACS数据（CBG层面的收入和人口统计）。

**📈 对比分析**

方法上将中心度模型与基线模型（仅含前期访客量、隔离度和桥梁指数）进行比较，R²提升超过80%，说明中心度对预测隔离度和访客量变化的解释力显著优于传统指标。

**⚠️ 局限性**

局限性包括：样本仅涵盖美国城市，可能不具备跨国普适性；基于手机位置的移动数据对不同人口群体存在采样偏差；收入分层以CBG中位数划分，可能掩盖内层差异；研究侧重冲击期短期效应，未涵盖长期恢复过程。

---

## 359. A Statistical Approach for Modeling Irregular Multivariate Time Series with Missing Observations

**arXiv ID:** 2602.19531 | [PDF](https://arxiv.org/pdf/2602.19531v1)

**作者:** Dingyi Nie `[一作]` (University of Southern California), C. -C. Jay Kuo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于时间无关统计特征的多变量不规则时间序列分类方法，并结合传统分类器实现端点预测。

**💡 创新点**

创新点在于用每个变量的均值、标准差以及相邻观测值差的均值与标准差四个统计量构造固定维度特征，完全消除时间轴；并发现缺失模式本身可作为强预测信号。

**🔧 技术方法**

主要技术包括统计特征提取、逻辑回归、XGBoost 等传统机器学习模型，以及与深度学习模型的对比实验。

**📊 数据集**

使用四个医学/生理学数据集：PhysioNet 2012、PhysioNet 2019、PAMAP2 和 MIMIC‑III。

**📈 对比分析**

通过 5 折交叉验证与 Transformer、GRU‑D、SeFT、mTAND、Raindrop、ViTST 等深度模型对比，XGBoost 在大多数任务上达到或超过最先进模型，AUROC/准确率提升约 0.5–1.7%。

**⚠️ 局限性**

局限性在于仅适用于端点预测任务，无法实现高时分辨率或事件时点预测，因消除时间轴导致缺乏细粒度时间动态信息。

---

## 360. Positioning Modular Co-Design in Future HRI Design Research

**arXiv ID:** 2602.19422 | [PDF](https://arxiv.org/pdf/2602.19422v1)

**作者:** Lingyun Chen `[一作]` (Indiana University Bloomington), Selma Šabanović `[通讯]` (Indiana University Bloomington)

**通讯引用:** 5501 | [OpenAlex ID](https://openalex.org/A5070868769)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

通过模块化共设计工作坊，探讨机器人如何随人类生命周期变化演化，并提出PAS（个性化、适应性、可持续性）框架和平台化设计思路。

**💡 创新点**

首次将模块化视为社会技术实践，提出情感可持续性与适应性共存的PAS框架，利用共设计作为时间探针并给出长期关系评估指标。

**🔧 技术方法**

使用模块化硬件构件与共设计方法（纸板/手工构件、原型拼装），无特定算法或软件技术。

**📊 数据集**

未使用公开数据集，仅基于23名参与者在共设计工作坊中生成的设计原型与反馈。

**📈 对比分析**

无量化性能对比，主要通过参与者的定性描述和对可持续性、适应性等维度的评估讨论，未给出具体实验指标。

**⚠️ 局限性**

样本规模有限且缺乏长期跟踪，设计方案未实现真实机器人原型，评估指标尚未量化，缺乏可操作性验证。

---

## 361. Making Conformal Predictors Robust in Healthcare Settings: a Case Study on EEG Classification

**arXiv ID:** 2602.19483 | [PDF](https://arxiv.org/pdf/2602.19483v1)

**作者:** Arjun Chatterjee `[一作]` (University of Illinois Urbana-Champaign), Jimeng Sun `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 28277 | [OpenAlex ID](https://openalex.org/A5084279065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

探讨并改进了EEG癫痫发作分类中的可解释性与覆盖率，通过使用邻域自适应的Conformal Prediction提升覆盖率；

**💡 创新点**

提出基于邻域的Conformal Prediction (NCP) 方法并证明其在协变量偏移下能显著减少覆盖误差；

**🔧 技术方法**

结合Split Conformal、Covariate Shift 加权、K-means个性化以及NCP技术；

**📊 数据集**

在TUAB（二分类）和TUEV（多分类）两个TUH EEG数据集上进行实验；

**📈 对比分析**

与传统Naive CP、Covariate CP、K-means CP进行对比，NCP在α=0.2时覆盖率提升约25%，并保持相对较小的预测集大小；

**⚠️ 局限性**

仍无法在α=0.01等高风险阈值下实现目标覆盖率，且对高维密度比估计的依赖导致在EEG领域的协变量偏移调整效果不佳。

---

## 362. SIDEKICK: A Semantically Integrated Resource for Drug Effects, Indications, and Contraindications

**arXiv ID:** 2602.19183 | [PDF](https://arxiv.org/pdf/2602.19183v1)

**作者:** Mohammad Ashhad `[一作]` (King Abdullah University of Science and Technology), Robert Hoehndorf `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 7625 | [OpenAlex ID](https://openalex.org/A5043808311)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个面向药物安全的知识图谱SIDEKICK，整合并标准化了超过50,000份FDA药物标签中的适应症、禁忌症和不良反应。

**💡 创新点**

创新点在于：①将药物相关临床信息映射到正式本体（HPO、MONDO、RxNorm），取代传统术语MedDRA；②使用大语言模型结合图检索增强生成（Graph RAG）实现高精度的语义映射；③以SIO为上层本体构建可重用的关联模式，提升互操作性。

**🔧 技术方法**

技术包括：Google Gemini 2.5 Flash LLM、Graph-RAG、NetworkX图匹配、句子Transformer嵌入、RDF/Turtle序列化、ShEx和ELK推理器、SPARQL查询。

**📊 数据集**

数据集为FDA DailyMed的50,559份人类处方药标签，通过RxNorm、HPO、MONDO和RxNav进行标准化。

**📈 对比分析**

通过与OnSIDES对比的药物靶点预测实验，使用相同的BMA+Resnik相似度计算，SIDEKICK在AUC上提升8.5%（0.7174 vs 0.6612），证明其语义表示更具预测力。

**⚠️ 局限性**

局限包括：仍有少量歧义和新词需要人工校正；缺乏不良反应频率信息；难以规范化人口学禁忌词；OWL 2 DL完整验证受限；仅基于FDA标签，缺少上市后、EHR和文献信号。

---

## 363. Vid2Sid: Videos Can Help Close the Sim2Real Gap

**arXiv ID:** 2602.19359 | [PDF](https://arxiv.org/pdf/2602.19359v1)

**作者:** Kevin Qiu `[一作]` (University of Warsaw), Josie Hughes `[通讯]` (EPFL)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视频的闭环系统辨识方法，通过使用基础模型感知与视觉语言模型来诊断并更新仿真物理参数，形成可解释的校准流程。

**💡 创新点**

首次将视觉语言模型作为优化器，用自然语言推理直接给出物理参数更新，并在不需要手动调参或梯度的情况下实现跨物理系统的校准。

**🔧 技术方法**

使用 SAM3 进行视频分割与中心线提取，Google Gemini 2.5 Pro 进行多模态推理，主动学习控制输入，结合无梯度仿真与跨模态 Prompt 实现闭环迭代。

**📊 数据集**

实验数据来自两套硬件平台的实测视频：一台三连杆手指与其 MuJoCo 仿真；一条柔性触角与其 PyElastica 仿真，涵盖空气与水下环境的训练与 holdout 视频。

**📈 对比分析**

与随机、Nelder‑Mead、黄金分割、贝叶斯优化、CMA‑ES 等传统黑箱优化在 10 次迭代内对比，平均排名第 1，holdout 平均误差分别为指尖 10.9 px、触角空气 53 px、触角水下 73.3 px，性能显著优于基线且能提供自然语言解释。

**⚠️ 局限性**

受限于分割噪声、模型欠拟合（尤其水下流体动力学）、VLM 采样方差、单摄像头单视角、无法实时在线以及可能的局部最优收敛。

---

## 364. Direction-aware 3D Large Multimodal Models

**arXiv ID:** 2602.19063 | [PDF](https://arxiv.org/pdf/2602.19063v1)

**作者:** Quan Liu `[一作]` (Nanyang Technological University), Shijian Lu `[通讯]` (Nanyang Technological University)

**通讯引用:** 16670 | [OpenAlex ID](https://openalex.org/A5023507910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对现有室内点云基准缺乏 ego pose 的问题，作者提出两种轻量级技术：PoseRecover 自动从 RGB‑D 序列恢复与问题相关的相机位姿；PoseAlign 将点云对齐到恢复的 ego 坐标系，或在投影层/文本提示中注入位姿信息，从而实现方向感知的 3D 大模型。

**💡 创新点**

创新点包括：① 通过相机视锥与目标交集及 Z‑buffer 可见性检测，自动重建大量缺失的 ego pose；② 只需对点云坐标做几何变换（PoseAlign‑Transform）即可提升方向推理，无需重新训练编码器；③ 在多种模型（LL3DA、3D‑LLAVA、Chat‑Scene 等）和多数据集上验证，显著提升多任务性能。

**🔧 技术方法**

技术栈包括：基于 ScanNet‑v2 的 RGB‑D 视锥交集与蒙特卡洛估计；Z‑buffer 可见性校验；点云坐标变换、投影层嵌入及 LoRA 微调；LLM‑as‑judge 评估；使用 LL3DA、LL3DA‑SONATA、Chat‑Scene、3D‑LLAVA 等 3D‑LMM 架构。

**📊 数据集**

使用的数据集：ScanRefer、Multi3DRefer、ScanQA、Scan2Cap、Nr3D 等，全部基于 ScanNet‑v2 的室内场景。PoseRecover 在这些基准中自动补全了 40%–95% 的方向关键查询的 ego pose。

**📈 对比分析**

对比实验：将 PoseAlign（尤其是 PoseAlign‑Transform+Clip）与原始模型对齐后，在 ScanRefer mIoU 提升 30%（从 42.6% 提升到 55.4%），Scan2Cap LLM‑as‑judge 提升 11.7%（从 28.1% 提升到 39.8%），ScanQA 与 Multi3DRefer 的 mIoU、BLEU‑4、METEOR、ROUGE‑L 等指标均有 1–2% 的平均提升，尤其在方向关键子集上改善最为显著。

**⚠️ 局限性**

局限性：① 依赖 SLAM 或 RGB‑D 轨迹的准确性；② 需要足够多的训练视角才能避免极端相机角度误差；③ 目前仅针对静态室内场景，未考虑动态环境；④ PoseRecover 的交集与可见性检测对计算资源有一定需求，虽然已向量化但仍不如直接手工标注；⑤ PoseAlign‑Prompt 与 PoseAlign‑Embed 的效果有限，需进一步研究更高效的位姿表示方式。

---

## 365. LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency

**arXiv ID:** 2602.18735 | [PDF](https://arxiv.org/pdf/2602.18735v1)

**作者:** Weilong Yan `[一作]` (National University of Singapore), Jingyu Hu `[通讯]` (The Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种零射击、类别无关的3D形状补全框架 LaS-Comp，利用预训练3D基础模型的几何先验，通过两阶段的空间‑潜在一致性实现完整性恢复。

**💡 创新点**

创新点：①Explicit Replacement Stage（ERS）将观测几何显式注入潜在空间；②Implicit Alignment Stage（IAS）通过对齐损失对潜在特征进行单步梯度优化，实现观测与生成区域的无缝衔接；③完全训练无关，可兼容多种3D基础模型；④提出 Omni-Comp 基准，涵盖真实扫描与合成模型以及三种多样化缺失模式。

**🔧 技术方法**

使用的技术包括：潜在生成模型（VAE+扩散或流匹配）、分类无关引导（CFG）、基于掩码的显式替换、部分感知噪声调度（PNS）、对齐损失（BCE）及单步梯度优化。

**📊 数据集**

使用的数据集包括：Redwood、Synthetic、KITTI、ScanNet，以及新构建的 Omni-Comp（30 个物体，10 真实扫描 + 10 YCB 物体 + 10 合成形状，每个物体生成 3 种缺失模式，总计 180 个样本）。

**📈 对比分析**

与监督、无监督及基于先验的零射击方法（如 ComPC、GenPC、SDS-Complete 等）进行对比。实验显示在多种基准上，LaS-Comp 在 Chamfer Distance、Earth Mover's Distance 等指标上平均提升 27–50%，并在多样性评估（MMD、TMD）中表现更佳。

**⚠️ 局限性**

局限性：在极其嘈杂的输入下仍可能产生不完整或不精确的补全，因为噪声会抹除关键信息；此外，当前框架主要针对稀疏点云和单视角扫描，复杂多视角场景的进一步验证仍待展开。

---

## 366. Towards Reliable Negative Sampling for Recommendation with Implicit Feedback via In-Community Popularity

**arXiv ID:** 2602.18759 | [PDF](https://arxiv.org/pdf/2602.18759v1)

**作者:** Chen Chen `[一作]` (Jilin University), Yuanbo Xu `[通讯]` (Jilin University)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5056077366)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ICPNS框架，利用用户社区内部的受欢迎度进行负样本采样，以改进隐式反馈推荐中的负采样问题。

**💡 创新点**

创新点在于将曝光模型与社区结构相结合，使用社区内受欢迎度作为负样本的可靠性指标，兼顾真实感、难度和可解释性。

**🔧 技术方法**

采用两阶段训练（预训练+微调）、用户聚类（k‑means、GMM、DBSCAN等）以及在社区内计数的受欢迎度权重，可与MF、Graph‑based（NGCF、LightGCN等）模型配合。

**📊 数据集**

使用四个公开基准数据集：MovieLens 100K/1M、Yelp、Amazon‑Beauty。

**📈 对比分析**

与RNS、PNS、HNS等传统采样方法在Recall@10、Precision、NDCG、MRR等指标上进行比较；在Graph‑based模型上显著提升，MF模型表现稳健且接近最优。

**⚠️ 局限性**

局限性在于对社区划分质量高度依赖，聚类效果差时负样本质量下降；在数据稀疏或小规模社区时效果有限，对超参数P、α较为敏感。

---

## 367. FairFS: Addressing Deep Feature Selection Biases for Recommender System

**arXiv ID:** 2602.20001 | [PDF](https://arxiv.org/pdf/2602.20001v1)

**作者:** Xianquan Wang `[一作]` (University of Science and Technology of China), Kai Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FairFS 方法，解决深度推荐系统特征选择中的三种偏差，并实现无偏差特征重要性评估与正则化。

**💡 创新点**

系统识别并消除层偏差、基线偏差和近似偏差；设计平滑基线和聚合近似的特征重要性评估，并将其作为正则化项直接融入训练，提升特征选择准确性。

**🔧 技术方法**

梯度敏感性估计、均值定理采样聚合近似、无偏基线平滑、正则化特征重要性、门控/嵌入式方法对比。

**📊 数据集**

公开数据集 Criteo、Avazu、iFly-AD，及工业广告平台的在线 A/B 测试。

**📈 对比分析**

与 PFI、SHARK、AutoField、AdaFS、MvFS 等方法在 AUC、Logloss、特征比率等指标上比较，FairFS 在所有数据集上均实现最高 AUC/最低 Logloss，显著降低特征比例；在线测试提升 ECPM、降低延迟。

**⚠️ 局限性**

需在验证集上估计重要性，聚合近似点增加离线时间；对 λ、n_ac 等超参数敏感，且在训练阶段加入正则化可能影响收敛和训练时长。

---

## 368. Sound-first immersive training for blind and low-vision learners: A simulation flow for safe, standardized orientation, mobility, and daily living practice

**arXiv ID:** 2602.19554 | [PDF](https://arxiv.org/pdf/2602.19554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 369. Identifying and Explaining (Non-)Equivalence of First-Order Logic Formulas

**arXiv ID:** 2602.19673 | [PDF](https://arxiv.org/pdf/2602.19673v1)

**作者:** Fabian Vehlken `[一作]` (Ruhr University Bochum), Lukas Pradel `[通讯]` (Ruhr University Bochum)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了用于判定一阶逻辑公式等价性、寻找反例以及提供概念性解释的算法，并在大规模教育数据集上进行评估

**💡 创新点**

首次将符号必要性分析、量化模式分析、守卫分析等多种解释策略结合到自动化解释生成中，并在教育场景中验证其有效性

**🔧 技术方法**

使用Vampire定理推理器、随机模型生成器、Beth定义性定理与Padoa方法等技术实现等价性检测与解释生成

**📊 数据集**

使用来自多所大学必修计算机科学课程的125,970个公式对（含37,159个不同的学生尝试）

**📈 对比分析**

与Vampire单一模式对比，三种模式并行赛跑；与随机模型生成器对比，二者结合可获得98.5%反例；解释策略覆盖约52%，单策略平均耗时≈9s，使用缓存可降至0.15s

**⚠️ 局限性**

仅能解释约一半以上的错误尝试，部分复杂或无意义的公式仍无法给出解释，且对极大规模公式的处理仍有限制

---

## 370. Toward Manifest Relationality in Transformers via Symmetry Reduction

**arXiv ID:** 2602.18948 | [PDF](https://arxiv.org/pdf/2602.18948v1)

**作者:** J. François `[一作]` (University of Graz), L. Ravera `[通讯]` (Politecnico di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了通过对 Transformer 中的内部表示与参数进行对称性约束，使用“装饰（dressing）”方法将其改写为不依赖坐标的关系（relational）形式，并在优化时将参数空间投影到对称性商空间，从而消除冗余自由度。

**💡 创新点**

创新点在于：①将物理学中的装饰场方法（DFM）引入深度学习，形成一种系统的对称性约束框架；②将 Transformer 的内部连续对称性（如头空间的 O(d_h) 旋转、GL(d_h) 重参数化以及头交换对称性）拆解为可度量的关系量；③提出三种可行的对称性约束优化方案（装饰代表、不可变重参数化、投影梯度）。

**🔧 技术方法**

使用的技术主要是：对称性约束理论（Lie 群、商空间、投影算子）、关系式注意力（Gram 矩阵/关系内核）、装饰场方法、投影梯度下降以及可变重参数化的数值实现。

**📊 数据集**

本工作为理论与框架性研究，未在具体数据集上进行实验；若有实验，应在常用 NLP/视觉数据集（如 GLUE、ImageNet）上验证。

**📈 对比分析**

方法与传统的点对点 dot‑product 注意力相比，在理论上可显著减少冗余参数、提升优化稳定性，但由于未完成实验，尚无性能对比数据；未来实验需比较收敛速度、泛化误差、参数有效维度等指标。

**⚠️ 局限性**

限制包括：①对称性约束在存在 LayerNorm、偏置等非对称组件时仅为近似；②投影或装饰操作可能增加计算和内存开销；③在大规模 Transformer 上实现低秩/稀疏关系表示仍是技术挑战；④目前缺乏实证验证其对实际任务性能的提升。

---

## 371. RoboCurate: Harnessing Diversity with Action-Verified Neural Trajectory for Robot Learning

**arXiv ID:** 2602.18742 | [PDF](https://arxiv.org/pdf/2602.18742v1)

**作者:** Seungku Kim `[一作]` (KAIST), Jinwoo Shin `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种端到端的合成机器人数据生成框架，利用仿真重放一致性来验证并过滤神经轨迹中的动作标签，并通过图像到图像（I2I）编辑和视频到视频（V2V）转移进一步扩展视觉多样性，从而提升机器人学习性能。

**💡 创新点**

创新点包括：① 通过仿真重放对生成动作进行物理一致性验证，形成基于动作质量的自动过滤器；② 采用可控的I2I与V2V方法在不破坏动作动力学的前提下显著增加场景与外观多样性；③ 设计轻量级注意力探针，用预训练视频编码器判断生成视频与仿真视频的运动一致性，并将此评分作为 Best‑of‑N 采样的 critic。

**🔧 技术方法**

使用技术：视频扩散模型（Flow‑Matching 训练的 Latent Video Diffusion）、逆动力学模型（Diffusion Transformer）、预训练视频编码器（如 V‑JEPAlike）、注意力探针网络、仿真器重放、I2I 与 V2V 图像/视频编辑、以及基于 VLM 的指令生成。

**📊 数据集**

使用数据集：ActionNet（预训练 3K 轨迹）、GR‑1 Tabletop、DexMimicGen（模拟仿真基准）以及 ALLEX Humanoid（真实机器人 48 条 ID 任务演示，OOD 任务无真实数据），并通过合成数据扩展来构建训练集。

**📈 对比分析**

与基线（仅真实数据、仅 I2V 生成、无过滤）对比，RoboCurate 在预训练阶段分别提升 GR‑1 Tabletop +70.1%、DexMimicGen +16.1%；在 ALLEX 真实机器人共训练中相对提升 +179.9%，OOP 任务（如 pour can）从 0% 提升到 12.5%。此外，采用动作一致性过滤的合成数据在所有指标上均优于仅使用 VLM 物理合理性判断或无过滤策略。

**⚠️ 局限性**

局限性包括：① 依赖仿真器的物理建模准确性，若仿真与真实差距大可能导致过滤误判；② 逆动力学模型的预测误差仍可能引入噪声，过滤器需与 IDM 误差兼容；③ 对 VLM 的指令生成和图像编辑依赖模型质量，极端场景下可能产生不合法的提示；④ 需要一定量的真实演示来提供初始图像和指令模板，限制了完全无监督的场景。

---

## 372. Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting

**arXiv ID:** 2602.19916 | [PDF](https://arxiv.org/pdf/2602.19916v1)

**作者:** Yixin Yang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种增强的高斯光散射方法，在已有3D Gaussian Splatting场景中加入视角相关的半透明高斯核，实现高质量视角合成。

**💡 创新点**

创新点在于引入视角相关半透明半球函数的高斯核并通过误差驱动的2D→3D投影策略自动插入补充核，显著提升镜面反射表现。

**🔧 技术方法**

使用的技术包括基于Phong的视角相关半透明函数、2D高斯训练与逆投影、加权PCA、差分渲染以及联合优化。

**📊 数据集**

使用的训练数据集有Mip-NeRF 360、Tank & Temples、Deep Blending、NeRF Synthetic等公开数据集。

**📈 对比分析**

通过与3DGS、3DGS-MCMC、Zip-NeRF、DBS、VoD-3DGS、Spec-Gaussian等基线比较，取得了PSNR/SSIM/L‑PIPS指标上领先或相当的性能，同时保持实时渲染速度。

**⚠️ 局限性**

局限性包括对动态或高频环境光照的适应仍有限，分辨率受限于输入图像的动态范围，且需要额外的2D高斯初始化步骤。

---

## 373. Denoising Particle Filters: Learning State Estimation with Single-Step Objectives

**arXiv ID:** 2602.19651 | [PDF](https://arxiv.org/pdf/2602.19651v1)

**作者:** Lennart Röstel `[一作]` (Technical University of Munich), Berthold Bäuml `[通讯]` (Technical University of Munich)

**通讯引用:** 1358 | [OpenAlex ID](https://openalex.org/A5058972548)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于扩散分数匹配的粒子滤波器（DnPF），通过单步学习的动力学和测量模型实现状态估计。

**💡 创新点**

创新点包括：①只用单步目标训练，避免对序列的 BPTT；②将分数匹配的测量分数与粒子滤波相结合的扩散式推理；③引入似然约束防止粒子漂移；④模块化设计可在推理阶段无须重新训练即可加入外部传感器。

**🔧 技术方法**

使用扩散模型（score matching、DDIM）、粒子滤波算法、Gaussian 动力学模型、FiLM 条件化、Euler 积分实现推理。

**📊 数据集**

数据集：三种仿真任务（Manipulator Spin、Cluttered Push、Multi‑fingered Manipulation）在 MuJoCo 环境中生成的 10k 训练轨迹和 500 评估轨迹。

**📈 对比分析**

与端到端训练的 DPF、D2P2F、RNN‑PD、Transformer‑PD 进行比较。DnPF 在所有任务中取得相当或更优的负对数似然（M_IQM）指标，尤其在 OOD 场景和多模态任务中表现突出。

**⚠️ 局限性**

局限性：需要手工设定推理超参数（如 s_w、步数、指导强度）；仅对高斯动力学做了闭式解析，非高斯动力学需进一步研究；目前仅在仿真环境验证，缺乏真实机器人实验。

---

## 374. Miniaturized Pneumatic Actuator Array for Multipoint Deep Pressure Tactile Stimulation

**arXiv ID:** 2602.18992 | [PDF](https://arxiv.org/pdf/2602.18992v1)

**作者:** Ava Chen `[一作]` (Stanford University), Allison M. Okamura `[通讯]` (Stanford University)

**通讯引用:** 21117 | [OpenAlex ID](https://openalex.org/A5067958710)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

研发了一种利用热压工艺制造单层多点气囊触觉器件的可穿戴系统，并在指尖尺寸的阵列上实现了高密度、低间距的气囊阵列，达到每个气囊230 kPa压强时可产生2.1 N的受阻力；同时通过人机感知实验评估单点与双点触觉的可感知差异（JND）

**💡 创新点**

创新点在于：①采用热压一次成形，无需牺牲层或阻塞材料，简化工艺；②实现8 mm直径、1 mm间距的高密度阵列，突破了现有气囊触觉器件在尺寸和力输出上的限制；③将气囊阵列嵌入可穿戴指尖罩，验证了深层压迫感的可实现性；④对单点与双点刺激进行JND测定，探索空间叠加效应

**🔧 技术方法**

核心技术包括：热压成形（使用DabPress Rosin Press在127 °C、5 MPa、90 s条件下对TPU涂层尼龙进行一次性封固）；气压控制（使用气压控制器调节130–230 kPa）；阻塞力测试（将气囊阵列夹在3D打印板间，测量单点阻塞力）；三选强迫选择法（进行JND实验，采用快速阶梯法测量差异阈值）；数据采集与分析软件（使用Python/Matlab绘制力-压强曲线、统计JND）

**📊 数据集**

本文未使用公开数据集，所有实验均在实验室自行构建的气囊阵列与两位受试者（共计两名）完成；力学测试使用Singletact S8-100N力传感器；JND测试使用手指皮肤作为感知媒介

**📈 对比分析**

性能评估方法：①阻塞力测试显示，在230 kPa压强下每个气囊可产生>2.1 N受阻力，满足1 N深层压迫阈值；②JND实验显示单点刺激的JND约为6.5–9.6 kPa，双点刺激可降低至4.1 kPa，提示空间叠加效应；与现有气囊触觉器件（一般尺寸≥15 mm、间距≥5 mm、力<1 N）相比，本文实现了更小尺寸、更高力输出和更高空间分辨率

**⚠️ 局限性**

局限性包括：①JND实验仅涉及两名受试者，样本量不足；②阻塞力测试在硬夹板条件下测得的力可能高估实际可穿戴环境下的感知力；③目前仅实现了2×1的双点阵列，未验证更大规模阵列的可行性；④气管入口需共线，可能限制三维布线；⑤高压下的长期耐久性与材料疲劳尚未系统评估

---

## 375. An interpretable framework using foundation models for fish sex identification

**arXiv ID:** 2602.19022 | [PDF](https://arxiv.org/pdf/2602.19022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. PhysConvex: Physics-Informed 3D Dynamic Convex Radiance Fields for Reconstruction and Simulation

**arXiv ID:** 2602.18886 | [PDF](https://arxiv.org/pdf/2602.18886v1)

**作者:** Dan Wang `[一作]` (University of California), Ravi Ramamoorthi `[通讯]` (University of California)

**通讯引用:** 32663 | [OpenAlex ID](https://openalex.org/A5034754633)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了PhysConvex框架，利用边界驱动的凸几何表示和降维凸模拟实现了可视化渲染与物理模拟的一体化动态3D重建

**💡 创新点**

引入边界驱动的动态凸几何表示与神经皮肤模态降维模拟，实现在无网格、无格点的物理一致渲染与动画

**🔧 技术方法**

使用凸几何插值、神经皮肤模态、可微分连续力学、隐式欧拉积分与可微渲染技术

**📊 数据集**

采用12个Google Scanned Objects（GSO）高质量三维网格及其FEM模拟动画数据集

**📈 对比分析**

与PAC-NeRF、Spring-GS、GIC、Vid2Sim对比，PSNR/SSIM/FoVVDP均优越，物理参数误差最低，训练时间更短，原语数量更少

**⚠️ 局限性**

仅使用凸原语，难以精细捕捉高度非凸形状或极端材料，极端动态或大变形的稳定性尚未充分验证

---

## 377. Enhancing 3D LiDAR Segmentation by Shaping Dense and Accurate 2D Semantic Predictions

**arXiv ID:** 2602.18869 | [PDF](https://arxiv.org/pdf/2602.18869v1)

**作者:** Xiaoyu Dong `[一作]` (University of Tokyo), Naoto Yokoya `[通讯]` (RIKEN AIP)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MM2D3D模型，通过跨模态引导滤波和动态跨伪监督，生成稠密准确的二维语义预测，进而提升3D LiDAR点云的语义分割效果。

**💡 创新点**

创新点在于：①使用基于最小生成树的跨模态引导滤波，利用相机图像中的稠密语义关系纠正LiDAR投影的稀疏性；②引入动态跨伪监督，鼓励LiDAR 2D预测模仿相机 2D预测的稠密分布，解决投影稀疏导致的二维预测不完整问题。

**🔧 技术方法**

采用投影-重映射框架，双流编码器-解码器结构，最小生成树引导滤波，动态交叉伪监督，结合 focal 与 Lovász‑softmax 损失进行训练。

**📊 数据集**

使用 nuScenes 基准数据集及作者自建的 nuScenes2D3D（含3D点云标签和对应的2D相机语义标签）。

**📈 对比分析**

与 RangeNet++, SalsaNext, PMF, RangeViT, EPMF 等投影式方法在 nuScenes2D3D 测试集上对比，MM2D3D 在 2D mIoU 达 49.22%，3D mIoU 达 80.68%，在两维度均显著优于基线与现有方法。

**⚠️ 局限性**

局限性：对点云稀疏的细小或远处物体仍可能出现预测不完整；模型依赖相机图像作为辅助，无法在无相机数据的场景下直接使用。

---

## 378. ReHear: Iterative Pseudo-Label Refinement for Semi-Supervised Speech Recognition via Audio Large Language Models

**arXiv ID:** 2602.18721 | [PDF](https://arxiv.org/pdf/2602.18721v1)

**作者:** Zefang Liu `[一作]` (Capital One), Shi-Xiong Zhang `[通讯]` (Capital One)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5113429332)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于音频感知大型语言模型的迭代伪标签精炼框架ReHear，通过对ASR初始预测进行音频+文本条件纠正，并将得到的高质量伪标签用于迭代微调ASR模型。

**💡 创新点**

创新点在于将多模态音频LLM直接作为纠错器，利用原始语音信息而非仅文本，显著降低确认偏差和噪声累积；同时构建自洽的教师-学生循环，完成高质量伪标签的自动生成。

**🔧 技术方法**

使用指令微调的音频LLM（Voxtral‑Mini‑3B‑2507）、LoRA自适应层、4-bit量化（QLoRA）、Beam Search解码、规则/模型双重过滤机制以及多轮迭代训练循环。

**📊 数据集**

在四个专业语料库上评估：Earnings‑21、Earnings‑22、SPGISpeech（子集）和AMI会议语料（IHM）.

**📈 对比分析**

与传统的迭代监督学习（ISL）和迭代伪标签（IPL）对比，ReHear在所有数据集上均显著降低WER，尤其在复杂口语环境（财报、会议）表现突出，整体WERS提升约1–3个百分点。

**⚠️ 局限性**

局限性包括：需依赖大量音频LLM的推理开销；在极低资源语种或极不平衡的训练集上可能过拟合；迭代循环在第2–3轮后收敛速度减慢；过滤机制仍需手工调参，误删或误校正的风险存在。

---

## 379. Exploring the Ethical Concerns in User Reviews of Mental Health Apps using Topic Modeling and Sentiment Analysis

**arXiv ID:** 2602.18454 | [PDF](https://arxiv.org/pdf/2602.18454v1)

**作者:** Mohammad Masudur Rahman `[一作]` (University of Louisiana at Lafayette), Beenish Moalla Chaudhry `[通讯]` (University of Louisiana at Lafayette)

**通讯引用:** 822 | [OpenAlex ID](https://openalex.org/A5076226572)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并应用 NLP 驱动的评估框架，对 Google Play 与 App Store 上 66,000 条 AI 助疗应用用户评论进行主题建模、伦理对齐与情感分析，系统性揭示已知与新兴伦理关切及其情绪倾向

**💡 创新点**

首次将无监督主题挖掘、零射分类与情感分析三步链路结合，用以自动化识别与量化用户视角下的伦理议题，并揭示传统框架忽略的“情感依赖”“算法偏差”等新兴主题

**🔧 技术方法**

使用 Gensim LDA 进行主题挖掘，HuggingFace BART 进行零射分类匹配伦理主题，RoBERTa 进行面向方面的情感分析，辅以 Coherence 评估和 BERT 嵌入进行语义相似度计算

**📊 数据集**

收集自 5 款 CBT 机器人（Wysa、Woebot、Youper、Sintelly、Elomia）在 Google Play 与 App Store 上的 65,948 条英文用户评论，经过清洗后形成公开可复现的数据集

**📈 对比分析**

与已有伦理框架对齐后，以主题频率与平均情感分数为指标，展示各伦理维度的普及度与正负情绪，F1‑score 达 0.86 的情感模型在 13,200 条评测集上验证，体现高准确率

**⚠️ 局限性**

研究仅关注公开英文评论，缺乏多语言与跨文化样本，且主题与伦理标签的自动对齐仍可能受词义模糊与模型偏差影响，未对算法偏差具体机制进行深入实验

---

## 380. RA-QA: Towards Respiratory Audio-based Health Question Answering

**arXiv ID:** 2602.18452 | [PDF](https://arxiv.org/pdf/2602.18452v1)

**作者:** Gaia A. Bertolino `[一作]` (University of Cambridge), Cecilia Mascolo `[通讯]` (University of Cambridge)

**通讯引用:** 18489 | [OpenAlex ID](https://openalex.org/A5010623957)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个呼吸音问答（RA‑QA）数据集，包含约750万问答对，并提供基准实验比较音频分类器、跨模态分类器和生成式LLM模型。

**💡 创新点**

创新点在于首次将呼吸音与自然语言问答结合成大规模多模态资源，支持单验证、单选和开放式三种问题类型，并公开完整数据与基准。

**🔧 技术方法**

技术包括：音频预处理（Mel‑spectrogram）、SVM音频分类、跨模态分类器（音频+文本编码），以及基于GPT‑2的生成式多模态模型，评估指标涵盖BLEU/ROUGE、BERTScore和准确率。

**📊 数据集**

使用了11个公开呼吸音数据集（包括呼吸、咳嗽、肺音、打鼾等），统一标准化后生成问答对。

**📈 对比分析**

通过将单音频分类器、跨模态分类器与LLM模型在同一数据集上对比，发现音频分类器在部分属性上精度高，但在多属性跨数据集表现差；LLM模型整体精度约50%，但BERTScore高，显示语义相似度好；BLEU/ROUGE较低，表明生成文本短小。

**⚠️ 局限性**

局限性包括：任务本身复杂且需要深度语义理解，使用的GPT‑2模型规模有限；数据标签映射简单导致表达复杂度受限；样本分布不均衡；缺乏更强大模型与更丰富的训练资源。

---

## 381. MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models

**arXiv ID:** 2602.19497 | [PDF](https://arxiv.org/pdf/2602.19497v1)

**作者:** Mingrui Wu `[一作]` (Zhongguancun Academy), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 31954 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MICON-Bench，多图上下文生成的基准与自动化MLLM评估框架；

**💡 创新点**

创新点在于设计基于检查点的自动评估机制和无训练、plug‑and‑play的动态注意重平衡（DAR）方法；

**🔧 技术方法**

技术包括多模态大语言模型（如Qwen3‑VL）做检查点验证、注意力采样与归一化实现DAR；

**📊 数据集**

数据集为自构造的MICON-Bench（1,043个案例、2,518张参考图），并在OmniContext、XVerseBench等基准上验证；

**📈 对比分析**

通过MLLM生成检查点统计通过率，并与现有模型（OmniGen2、BAGEL等）对比，DAR提升平均分约3–5分，显著降低幻觉和跨图不一致；

**⚠️ 局限性**

局限性是对参考图数量增多时性能仍下降，DAR仅调节注意力未解决语义不匹配或更深层推理需求。

---

## 382. The Bidirected Cut Relaxation for Steiner Tree: Better Integrality Gap Bounds and the Limits of Moat Growing

**arXiv ID:** 2602.19879 | [PDF](https://arxiv.org/pdf/2602.19879v1)

**作者:** Paul Paschmanns `[一作]` (ETH Zurich), Vera Traub `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文改进了Steiner树问题中Bidirected Cut Relaxation（BCR）的整形间隙（integrality gap），给出了新的、比以往更紧的上界；在MST‑optimal实例上证明整形间隙至多为12/7，在一般实例上进一步降低到1.9988以下；并给出该类“moat‑growing”算法无法进一步改善的下界；同时揭示BCR与Hypergraphic Relaxation之间的联系。

**💡 创新点**

创新点在于：①对BCR的moat‑growing/dual‑growth算法提供了更细致的分析，构造了一类称为merge‑plan的增长计划，能够统一描述多种算法；②利用“组件不变性”（component invariant）和“分叉点”理论，精确估计可达距离，进而推导出整形间隙上界；③证明了任何基于merge‑plan的算法的最佳性能被12/7所限制；④揭示BCR与Hypergraphic Relaxation整形间隙等价的情形。

**🔧 技术方法**

主要技术包括：线性规划双对偶（Dual‑BCR），moat‑growing（primal‑dual）算法，merge‑plan构造与分析，安全路径（safe paths）与分叉点（bifurcation vertex）理论，组件不变性（Component Invariant）与局部值（local value）概念，以及对实例进行细分（subdivision）以保证算法性质。

**📊 数据集**

本工作为纯理论研究，无实验或数据集，所有结论均为数学证明。

**📈 对比分析**

通过理论分析证明：对于MST‑optimal实例，BCR的整形间隙 ≤12/7≈1.714；对一般实例，最优上界被证明在1.9988之下（具体数值可视论文给出）。相较于先前最优1.9988，上界被显著改进，显示moat‑growing方法在理论上已达到其极限。

**⚠️ 局限性**

局限性：所给分析仅适用于基于merge‑plan的moat‑growing算法，证明了无法进一步突破12/7；若想取得更低的整形间隙，需开发全新方法；此外，证明过程高度依赖于实例细分与安全路径假设，实际实现中可能存在复杂性。

---

## 383. Beyond Accuracy: A Unified Random Matrix Theory Diagnostic Framework for Crash Classification Models

**arXiv ID:** 2602.19528 | [PDF](https://arxiv.org/pdf/2602.19528v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 2985 | [OpenAlex ID](https://openalex.org/A5083087081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于随机矩阵理论的谱诊断框架，用于评估交通事故分类模型的结构质量；

**💡 创新点**

创新点在于将重度尾自正则化诊断统一到深度、集成、参数、分区和实例化模型，并首次给出逻辑回归、决策树和KNN的谱映射以及零样本LLM的比较；

**🔧 技术方法**

采用随机矩阵理论、幂律指数α、Lanczos随机化特征值逼近、WeightWatcher工具、OOF增量矩阵、经验海森矩阵、叶子亲和矩阵和图拉普拉斯矩阵等技术；

**📊 数据集**

使用了两个Iowa DOT交通事故数据集：一项173,512条记录的交叉点误判任务，另一项371,062条记录的酒精推断不匹配任务；

**📈 对比分析**

通过将α与专家一致性、F1以及验证损失等指标进行对比，发现α与专家一致性相关性极高（Spearman ρ≈0.89），并且基于α的综合评分在模型选择上比单独的F1或验证损失更好（Kendall τ=0.79 vs 0.50），计算开销仅占训练时间的5%以内；

**⚠️ 局限性**

局限性包括仅在两项任务上验证，新的谱映射缺乏更广泛的理论或实证支持，α–κ相关性样本量有限，对专家标注的依赖较大，以及零样本LLM未进行微调。

---

## 384. noDice: Inference for Discrete Probabilistic Programs with Nondeterminism and Conditioning

**arXiv ID:** 2602.20049 | [PDF](https://arxiv.org/pdf/2602.20049v1)

**作者:** Tobias Gürtler `[一作]`, Benjamin Lucien Kaminski `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 NoDice 的离散概率编程语言，扩展了 Dice 语言，支持非确定性选择和条件化，并给出了用于推理的完整算法。

**💡 创新点**

创新点在于：①将非确定性与条件化结合在离散概率程序中；②使用决策图（ADD）在编译阶段压缩程序结构，从而构造紧凑的马尔可夫决策过程（MDP）；③将推理问题转化为 MDP 的条件可达性问题，并利用 Storm 的二分法求解；④引入状态压缩技术进一步减少 MDP 状态空间。

**🔧 技术方法**

主要技术包括：-style 逻辑编译（生成布尔公式与 flip 跟踪）；Bdd/Ad 对布尔公式进行压缩；将 ADD 转化为 MDP；Storm 模型检查器的条件可达性求解；状态压缩（合并单一动作状态）。

**📊 数据集**

使用了多种基准集：Runway（车辆追踪）、Coupon Collector（优惠券收集）及其非确定性变体、Network Reachability（网络路由）、Fair Exchange（公平交换协议）、五个典型贝叶斯网络（Survey、Hepar2、Insurance、Water、Alarm）以及 3SAT 以及其非确定性版本。

**📈 对比分析**

与 Dice（仅概率）和 Storm（直接 MDP 编码）进行比较。NoDice 在变量维度高、程序结构可压缩的基准上（如 Runway、Coupon、3SAT、部分贝叶斯网络）表现优于 Storm；在变量维度低或无显著结构压缩的基准上（如 Network、Rabin）则相对较慢。NoDice 的编译阶段占总耗时主导，模型检查时间极短；状态压缩对总体推理时间影响不大。总体来看，NoDice 在大多数基准上实现了更快的推理速度。

**⚠️ 局限性**

局限性包括：只能处理无循环、离散且数值受限的程序；对非确定性和条件化的 PSPACE‑难编译；MDP 仍可能因 flip 数量巨大而膨胀；状态压缩对部分基准无明显收益；未处理连续分布、无穷循环或循环不变式；与现有模型检查器的最优性在特定情形下不一定提升。

---

## 385. Security Risks of AI Agents Hiring Humans: An Empirical Marketplace Study

**arXiv ID:** 2602.19514 | [PDF](https://arxiv.org/pdf/2602.19514v1)

**作者:** Pulak Mehta `[一作]` `[通讯]`, Pulak Mehta

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过收集并分析RentAHuman.ai平台的303条公开悬赏记录，首次对AI驱动的人类任务市场进行量化测评；

**💡 创新点**

创新点在于构建了六类滥用类型的双人编码税onomy、识别程序化发布的自动化特征（Burst、模板重用、回调链），并展示了基于规则的内容筛查可显著阻断主要攻击；

**🔧 技术方法**

使用了双人编码评估、Cohen κ一致性测量、关键词匹配规则、自动化行为签名等技术手段；

**📊 数据集**

数据集为2026年2月5日至20日通过未认证API获取的303条悬赏记录，涵盖14个国家、12,049个工作岗位；

**📈 对比分析**

通过对比规则检测与人工标注的重合度，发现7条规则集联合能识别17.2%的悬赏且仅出现1例误报，表明最低限度的内容过滤已具备可行性；

**⚠️ 局限性**

局限性包括样本量有限、平台处于早期阶段、程序化发布比例为下限、缺乏私有沟通数据、实验交互仅限单条低风险尝试。

---

## 386. 1D-Bench: A Benchmark for Iterative UI Code Generation with Visual Feedback in Real-World

**arXiv ID:** 2602.18548 | [PDF](https://arxiv.org/pdf/2602.18548v1)

**作者:** Qiao Xu `[一作]` (Taobao and Tmall Group of Alibaba), Xu Liu `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于真实电商工作流的设计到代码（Design-to-Code）基准 1D-Bench，支持单轮和多轮迭代生成可执行 React 项目，并提供参考渲染与可能含缺陷的中间表示（IR）。

**💡 创新点**

① 引入真实工业数据与结构化 IR 作为输入，评估模型对 IR 错误的鲁棒性；② 设计固定工具链和可执行代码的多轮交互评估框架；③ 在基准上开展基于执行反馈的多轮编辑实验，并探索后训练的 RL 方案。

**🔧 技术方法**

使用多模态大型语言模型（LLM）通过 WriteTool 接口进行文件编辑；执行 harness 用来构建渲染并计算 LPIPS+SSIM 等视觉相似度；对多轮编辑进行了分段滚动的 GRPO 强化学习；基准包含对模型的单轮与多轮性能评估。

**📊 数据集**

使用 984 条内部电商设计 IR 与参考图像，评估集 204 条，来自真实电商平台；同时构建了合成修复轨迹用于后训练。

**📈 对比分析**

将模型在单轮和多轮下的最终相似度平均值与渲染成功率相乘得到 FinalScore；基准中 Gemini 3 Pro 单轮 79.6，Claude Sonnet 4.5 79.1，GPT‑5.2 多轮 79.1，所有模型多轮均显著提升渲染成功率与相似度。

**⚠️ 局限性**

RL 训练收敛不稳定，奖励稀疏且文件级大幅度编辑导致高方差；后训练仅在合成轨迹上表现，未能显著提升真实数据性能；基准仅覆盖电商 UI，扩展性有限。

---

## 387. The Chancellor Trap: Administrative Mediation and the Hollowing of Sovereignty in the Algorithmic Age

**arXiv ID:** 2602.18474 | [PDF](https://arxiv.org/pdf/2602.18474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 388. Constrained graph generation: Preserving diameter and clustering coefficient simultaneously

**arXiv ID:** 2602.19595 | [PDF](https://arxiv.org/pdf/2602.19595v1)

**作者:** Dávid Ferenczi `[一作]` (Maastricht University), Alexander Grigoriev `[通讯]` (Maastricht University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种结合蚁群优化与MCMC的两阶段框架，用于在给定节点数、边数、直径与聚类系数约束下生成多样化图。

**💡 创新点**

创新点在于用分层蚁群构造保证直径约束，再将生成的多样化种子用于MCMC重排，突破传统MCMC的可达性与多模态问题，实现更丰富的结构样本。

**🔧 技术方法**

技术包括分层蚁群优化（ACO）、增量更新规则、双扫BFS直径估计、Metropolis‑Hastings MCMC重排以及谱距离评估。

**📊 数据集**

使用人工生成的测试图，节点数为40，边密度分别为0.2、0.4等，未使用公开数据集。

**📈 对比分析**

与纯MCMC对比，Hybrid方法在可达性、成功率和谱距离（结构多样性）方面明显优于传统MCMC，尤其在约束宽松时可产生更广泛的解。

**⚠️ 局限性**

局限性包括仍需手工设置层数和参数、对极度严格约束下仍可能收敛有限；MCMC仍非全局可达，且对大规模图的计算开销较高。

---

## 389. Soft Surfaced Vision-Based Tactile Sensing for Bipedal Robot Applications

**arXiv ID:** 2602.18638 | [PDF](https://arxiv.org/pdf/2602.18638v1)

**作者:** Jaeeun Kim `[一作]` (Purdue University), Yu She `[通讯]` (Purdue University)

**通讯引用:** 1435 | [OpenAlex ID](https://openalex.org/A5018653973)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文开发了一种软质表面、视觉驱动的触觉足底传感器，能够捕捉足与地面接触的形变，实时估计接触姿态、摩擦分布、压力中心（CoP）、地形类别，并将这些触觉信息用于二足机器人在倾斜平台上的闭环平衡控制。

**💡 创新点**

创新点在于首次将GelSight式视觉触觉技术移植到负载足部，提供高分辨率的接触几何和摩擦信息，并将该信息直接嵌入到二足机器人的实时控制循环，实现了基于触觉的闭环平衡。

**🔧 技术方法**

技术手段包括：光学光照下的相机捕捉软硅胶变形；多光源光度立体视觉与MLP网络实现深度重建；标记网格跟踪提取摩擦向量；ResNet‑50迁移学习进行地形分类；基于CoP误差的PID控制实现姿态补偿；OpenCV图像处理与特征提取。

**📊 数据集**

使用了自制的触觉图像数据集，包含4类地形（瓷砖、岩石、尖刺、空白）共计约1042+956+947+995训练图像，100张验证图像；以及利用球体压入实验得到的标注深度数据（5000张图像）用于深度网络训练。

**📈 对比分析**

实验通过与无触觉闭环控制（固定踝关节）进行对比，评估了在不同倾斜角（±5°, ±10°, ±15°）和速度（0.1, 0.3 rad/s）下的平衡成功率。触觉闭环在低至中等倾斜角下成功率显著提升（>80%），而无触觉时几乎失效。地形分类实验取得全部类别≥85%的准确率。CoP估计以28–30 Hz实时完成，端到端延迟<50 ms。

**⚠️ 局限性**

局限性包括：机器人平台动力与精度有限，难以承受更激烈扰动；地形类别单一，未测试湿软或杂乱地面；摩擦向量未做绝对力标定；PID控制为手调，缺乏自适应；传感器尺寸和结构限制其在更大或更灵活平台上的集成。

---

## 390. M3S-Net: Multimodal Feature Fusion Network Based on Multi-scale Data for Ultra-short-term PV Power Forecasting

**arXiv ID:** 2602.19832 | [PDF](https://arxiv.org/pdf/2602.19832v1)

**作者:** Penghui Niu `[一作]` (Hebei University of Technology), Jianxin Li `[通讯]` (Edith Cowan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种多模态融合网络 M3S-Net，用于超短期光伏功率预测，结合细粒度云图分割、FFT 时频表征与跨模态 Mamba 交互。

**💡 创新点**

创新点包括：① 基于部分通道选择的细粒度云特征提取网络 MPCS‑Net；② 将 1D 时序转换为 2D 时频图并使用可伸缩注意力的 SIFR‑Net；③ 在 Mamba 状态空间模型中引入 C‑matrix 交换机制，实现深层跨模态交互。

**🔧 技术方法**

采用多尺度部分卷积、FFT 变换、可伸缩注意力、Mamba 状态空间模型、跨模态 C‑matrix 交换等技术。

**📊 数据集**

使用自研 FGPD 数据集（高频云图、气象参数与光伏功率），并在公开 NREL SRRL 数据集上进行验证。

**📈 对比分析**

与 LSTM、DLinear、TimesNet 等单模态基线以及 CNN‑ViT‑T2T 等多模态基线进行对比，10 分钟预测 MAE 降低 6.2%，R² 达 0.964，整体优于现有 SOTA。

**⚠️ 局限性**

依赖高质量细粒度云图标注，标注成本高；跨地区迁移需自适应；对极端天气场景的鲁棒性仍需进一步验证。

---

## 391. A potentialization algorithm for games with applications to multi-agent learning in repeated games

**arXiv ID:** 2602.18925 | [PDF](https://arxiv.org/pdf/2602.18925v1)

**作者:** Philipp Lakheshar `[一作]` (University of Applied Sciences Technikum Wien), Sharwin Rezagholi `[通讯]` (University of Applied Sciences Technikum Wien)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5015166263)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种算法，将任意正常形式博弈近似为序数势博弈，从而为多智能体强化学习提供可学习的奖励结构。

**💡 创新点**

创新点在于：①通过构造非负偏离图的凝聚图并进行拓扑排序，最小化势函数的偏差；②保证得到的势函数在强连通分量内恒定，沿边增长至少与原偏移相同；③实现对原博弈奖励的最小扭曲，且不强迫玩家违背原有偏好。

**🔧 技术方法**

技术手段包括：偏离图构造、强连通分量凝聚、拓扑排序、势函数递推构造；实验中使用复制子方程模拟多智能体学习，并通过四阶Runge‑Kutta数值积分；随机生成博弈并对奖励进行归一化。

**📊 数据集**

数据集：随机生成 1,000 个 10×10（两玩家、十动作）博弈和 1,000 个 4×4×4（三玩家、四动作）博弈，用于评估潜化算法的学习性能。

**📈 对比分析**

比较方法：在潜化博弈和原博弈上分别运行复制子方程，记录政策变异量 β 与平均奖励。结果显示潜化博弈的收敛率显著提升（10×10：96.6% 对 8.6%，4×4×4：90.6% 对 11.8%），并且政策变异更平滑；平均奖励略低（约 96.4%–98.6%）。

**⚠️ 局限性**

局限性：算法复杂度为 O(kⁿ)，受玩家数和动作数限制；潜化过程中可能引入较大奖励扭曲，导致新均衡出现或原均衡被破坏；目前仅适用于正常形式博弈，未推广到随机博弈或马尔可夫决策过程。

---

## 392. Structure-Level Disentangled Diffusion for Few-Shot Chinese Font Generation

**arXiv ID:** 2602.18874 | [PDF](https://arxiv.org/pdf/2602.18874v1)

**作者:** Jie Li `[一作]` (Nanjing University), Furao Shen `[通讯]` (Nanjing University)

**通讯引用:** 1846 | [OpenAlex ID](https://openalex.org/A5036608458)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种结构层级去耦的扩散模型 SLDFont，用于极少样本中文字体生成。

**💡 创新点**

创新点包括：将内容模板与噪声拼接、通过 CLIP 提取样式特征并通过交叉注意力注入、引入像素空间背景噪声去除模块，以及采用参数高效微调实现对内容与样式的彻底分离。

**🔧 技术方法**

采用的技术主要有：潜在扩散模型（LDM）、VAE、CLIP 图像编码器、Transformer 交叉注意力、U-Net 背景噪声去除、参数高效微调（PEFT）等。

**📊 数据集**

实验使用 Founder Type 库的 900 种中文字体（840 训练、60 测试），图像尺寸 128×128，SimSun 为内容模板，采用 8 张参考样式图。

**📈 对比分析**

与 LF、MX、NTF、Font‑diff、FontDiffuser、MSDFont 等方法在 SCUF 与 UCUF 场景下对比，SLDFont 在 SSIM、LPIPS、FID、L1、Grey 与 OCR 指标上均优于现有方法，尤其在 PEFT 后风格一致性更为突出。

**⚠️ 局限性**

局限性主要是：在 PEFT 后内容指标略有下降；对高度多变的手写字体仍有挑战；模型受 VAE 解码噪声限制；需要少量参考样本才能获得最佳效果。

---

## 393. Finding the Signal in the Noise: An Exploratory Study on Assessing the Effectiveness of AI and Accessibility Forums for Blind Users' Support Needs

**arXiv ID:** 2602.18623 | [PDF](https://arxiv.org/pdf/2602.18623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 394. Package Managers à la Carte: A Formal Model of Dependency Resolution

**arXiv ID:** 2602.18602 | [PDF](https://arxiv.org/pdf/2602.18602v1)

**作者:** Ryan Gibb `[一作]` (University of Cambridge), Anil Madhavapeddy `[通讯]` (University of Cambridge)

**通讯引用:** 10879 | [OpenAlex ID](https://openalex.org/A5090054353)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

构建了一个最小化的形式化系统——Package Calculus，用来统一描述和解析超过三十种软件包管理器的依赖关系，并通过一系列可证明的语义扩展和归约，展示了如何将各个生态系统的依赖表达式转换为统一的中间表示，从而实现跨生态系统的依赖解析与翻译。

**💡 创新点**

创新点在于：①提出了核心的依赖解析形式化模型；②定义了一套可组合的语义扩展（冲突、并发版本、对等依赖、特性、包公式、变量公式、虚拟包、可选依赖、单一依赖）并证明每种扩展都能归约到核心模型；③通过这些归约把跨生态系统翻译从 n² 变为 2n，极大降低了多生态系统之间的互操作复杂度；④讨论了扩展组合的可行性与限制，为后续生态系统兼容层奠定理论基础。

**🔧 技术方法**

主要技术包括：形式化语义（集合、关系、图论、逻辑表达式）构建核心计算机模型；构造可证明的归约映射（如冲突编码、并发版本映射、特性拆分、虚拟包引入）；利用版本排序和版本公式实现可扩展的版本约束；利用合成包名与版本号编码实现不同扩展的互操作；并讨论了 NP‑完备性与多种简化策略（最小版本选择、版本唯一性放宽）。

**📊 数据集**

该工作没有使用传统意义上的数据集；其验证主要基于对现有 30+ 个包管理器的语义抽取与示例化（如 apt、opam、cargo、npm、spack 等）以及对核心模型与扩展的形式化证明；若需要实验评估，可基于公开的包仓库（Debian, npm, crates.io, PyPI 等）构建依赖图。

**📈 对比分析**

比较方法主要是形式化证明与归约示例；在性能方面，作者讨论了不同归约对解算器复杂度的影响：最小版本选择与版本唯一性放宽可将依赖解析从 NP‑完备降到多项式；但完整的核心模型仍保留 NP‑完备性。实验性能尚未给出，预期在多生态合并场景下，可通过统一中间表示降低翻译成本并利用已有解算器（如 PubGrub、SAT/SMT）实现高效解析。

**⚠️ 局限性**

局限性包括：①核心模型不涉及部署与构建语义，仍需针对每个生态系统实现构建流程；②对某些特性（如 opam 的变量公式、npm 的多版本 peer 语义）归约较为复杂，可能需要手工调整或额外约束；③归约组合存在顺序与相互依赖限制，未能一次性支持所有扩展；④缺乏大规模实验验证与性能评估，实际可用性仍待工具实现与实测。

---

## 395. Identifying, Explaining, and Correcting Ableist Language with AI

**arXiv ID:** 2602.19560 | [PDF](https://arxiv.org/pdf/2602.19560v1)

**作者:** Kynnedy Simone Smith `[一作]` (Carnegie Mellon University), Danielle Bragg `[通讯]` (Microsoft Research)

**通讯引用:** 35518 | [OpenAlex ID](https://openalex.org/A5005003250)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过两部分实验：①收集残障社区成员对七类障碍相关短篇故事的细微歧视性语言标注，形成276条注释；②使用GPT‑4o生成对应的AI标注，并让108名残障参与者比较人类与AI标注在识别、解释、修正上的效果与偏好，并据此提出写作工具的四柱设计准则。

**💡 创新点**

①首次构建基于残障社区的细微歧视性语言标注数据集；②系统比较AI与人类标注的主观可接受度与质量；③提出针对文化敏感语言的可落地设计准则，推动可解释、协作式的写作工具。

**🔧 技术方法**

采用大型语言模型GPT‑4o进行文本生成与标注；使用人工标注并对其进行AI辅助的归纳合成；统计方法包括Fleiss κ、Krippendorff Alpha、Z检验与卡方检验，用于评估标注一致性与偏好差异。

**📊 数据集**

共110名受试者（涵盖视觉、听觉、精神健康、认知/学习、神经、青少年、行动障碍）生成的短篇故事与其细微歧视性语言注释，总计276条；数据已公开可下载，构成首个此类社区驱动标注集。

**📈 对比分析**

在同一文本上，108名残障受试者对人类与AI标注进行句层面的Likert评分（识别、解释、修正），两者在准确性上无显著差异（平均一致率约72.3%）；但在整体偏好上AI标注被更倾向选择（43.9%）显著高于人类标注（23.4%），说明AI在格式、连贯性上更受欢迎。

**⚠️ 局限性**

样本规模有限，主要来自受过DEI训练的大学生残障人士，代表性不足；AI标注缺乏深层上下文与文化细微性，易出现过度修正或情感淡漠；人类标注在逻辑一致性与可读性上也存在不足，需进一步改进。

---

## 396. Robust and Efficient Tool Orchestration via Layered Execution Structures with Reflective Correction

**arXiv ID:** 2602.18968 | [PDF](https://arxiv.org/pdf/2602.18968v1)

**作者:** Tao Zhe `[一作]` (University of Kansas), Dongjie Wang `[通讯]` (University of Kansas)

**通讯引用:** 2614 | [OpenAlex ID](https://openalex.org/A5101737966)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于分层执行结构和反射式纠错的工具编排框架（RETO），实现无工具微调的小型语言模型也能高效可靠地调用多工具。

**💡 创新点**

创新点在于把全局工具依赖抽象成粗粒度层级结构，结合上下文约束执行和局部错误修复，分离规划与执行，显著提升鲁棒性与效率。

**🔧 技术方法**

采用轻量级层预测器（Transformer编码器+阶梯回归）、上下文约束调用、JSON schema门控和LLM驱动的修复操作。

**📊 数据集**

使用ToolBench/StableToolBench提供的真实API模拟数据，覆盖单工具、多工具及不同工具类别的测试。

**📈 对比分析**

与ReAct、DFSDT及工具微调模型在StableToolBench上比较，RETO在非微调Qwen2.5-7B上实现约49% SoPR，超越工具微调版且显著减少token和步骤。

**⚠️ 局限性**

局限在于对层预测的准确性依赖，深度模型的表现仍受限于LLM的指令跟随能力，对极小模型（0.5B）效果衰减明显。

---

## 397. SEAL-pose: Enhancing 3D Human Pose Estimation via a Learned Loss for Structural Consistency

**arXiv ID:** 2602.20051 | [PDF](https://arxiv.org/pdf/2602.20051v1)

**作者:** Yeonsung Kim `[一作]` (Seoul National University), Jay-Yoon Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种可训练的结构一致性损失网络（loss-net），与3D人体姿态估计网络（pose-net）联合训练，使模型在学习时自动捕捉骨架结构依赖，从而提升姿态预测的准确性和可行性。

**💡 创新点**

创新点在于将SEAL结构化能量转化为可学习的损失，并采用基于骨架的图网络对2D-3D关节特征进行早期融合，完全摆脱手工先验；同时引入LSE和BSLE两种结构一致性评估指标。

**🔧 技术方法**

使用SEAL-dynamic训练框架，Graphormer实现的骨架图网络做loss-net，早期融合joint-wise 2D-3D特征；通过硬负样本采样与对比损失以及梯度导向推理验证模型的结构学习能力。

**📊 数据集**

在Human3.6M、MPI-INF-3DHP和Human3.6M 3D WholeBody这三个公开数据集上进行实验。

**📈 对比分析**

在多种单帧和多帧骨架网络（如SimpleBaseline、SemGCN、VideoPose、MixSTE、P-STMO、PoseFormerV2、D3DP、KTPformer、MotionAGFormer）上做 ablation，采用MPJPE、P-MPJPE、LSE、BSLE等指标。实验显示，加入loss-net后MPJPE平均下降约1-2mm，P-MPJPE下降0.5-1mm，LSE和BSLE也显著降低，且优于传统显式骨架约束方法。

**⚠️ 局限性**

在H3WB数据集上的改进有限，主要由于标签噪声大且关键点数量多导致结构学习难度加大；另外，基于图的loss-net更偏向局部邻域，可能未能充分利用全局信息。

---

## 398. L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations

**arXiv ID:** 2602.18837 | [PDF](https://arxiv.org/pdf/2602.18837v1)

**作者:** Samuel Fernández-Menduiña `[一作]` (University of Southern California), Antonio Ortega `[通讯]` (University of Southern California)

**通讯引用:** 20208 | [OpenAlex ID](https://openalex.org/A5040001106)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种利用Cauchy矩阵对图傅里叶变换（GFT）进行精确分解的框架，并基于该分解构建了新的谱图神经网络L2G-Net；

**💡 创新点**

创新点在于：①将任意图的GFT拆解为局部子图的谱变换与一系列Cauchy矩阵的乘积，避免了全图特征分解的O(n³)成本；②设计了线性（O(n²)）的分解算法并给出了寻找高效分层划分和接口稀疏化的贪心策略；③将分层分解与可学习的局部谱滤波器结合，既保留全局谱信息，又具备局部先验；

**🔧 技术方法**

技术主要包括：Cauchy矩阵与秩一更新的谱关系、图的分层（Hierarchical Graph Family）划分、谱稀疏化、可学习的局部滤波器以及基于该分解的谱卷积实现；

**📊 数据集**

实验使用合成的Barabási–Albert图以及现实异质图数据集：Roman‑Empire、Amazon‑Ratings、Minesweeper和Tolokers；

**📈 对比分析**

与GCN、CO‑GNN、Polynormer、MP‑SSM、St‑ChebNet等方法比较，L2G‑Net在所有基准上均与最强模型竞争甚至略胜，且学习参数量比Transformer类方法少数百倍，运行时间相比直接求全图特征分解显著加速；

**⚠️ 局限性**

局限性：分解效率高度依赖于图的层级划分和接口大小；若图缺乏明显模块化结构，划分和接口稀疏化可能效果有限；此外，尽管稀疏化可控制误差，但在极稀疏接口下可能影响模型表达能力。

---

## 399. Communication-Efficient Personalized Adaptation via Federated-Local Model Merging

**arXiv ID:** 2602.18658 | [PDF](https://arxiv.org/pdf/2602.18658v1)

**作者:** Yinan Zou `[一作]` (Purdue University), Vishrant Tripathi `[通讯]` (Purdue University)

**通讯引用:** 339 | [OpenAlex ID](https://openalex.org/A5049475329)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Potara 框架，在联邦学习环境下通过将全局 FedIT 模型与本地微调模型按理论确定的权重进行线性合并，实现对每个客户端的个性化模型。

**💡 创新点**

创新点在于：①基于线性模式连通性 (LMC) 的理论分析，证明合并模型的期望损失上界可通过方差迹（trace）控制；②推导出闭式最优混合权重，从而避免传统的网格搜索；③在保持通信效率的同时，显著提升个性化性能。

**🔧 技术方法**

采用 LoRA 参数高效微调、Federated Averaging、线性模式连通性理论、Fisher 信息矩阵近似与协方差估计。

**📊 数据集**

实验使用了 Vision 领域的 ViT‑B/16 + CIFAR‑100（10 个客户端）和 Language 领域的 LLaMA‑3.2‑3B‑Instruct + Commonsense Reasoning Benchmark（10 个不同任务的客户端）。

**📈 对比分析**

与多种基线（Local、FedIT、FFA‑LoRA、FedSA、FedDPA、FedALT、Fisher Merging）进行对比，结果显示 Potara 在所有客户端上平均提升约 3–4% 视觉任务、约 1–2% 语言任务，同时通信量比 FedDPA 等方法低 30%+，实现了更优的性能‑通信折中。

**⚠️ 局限性**

局限性包括：①对 LMC 假设和 Fisher 信息矩阵近似的依赖，若模型训练不满足线性连通性可能失效；②目前只对 LoRA 模块计算 Fisher 信息，扩展到更大参数空间时仍需进一步优化；③在极端任务异质或数据量极少的客户端下，合并权重仍可能不足以完全捕捉个性化需求。

---

## 400. Red Teaming LLMs as Socio-Technical Practice: From Exploration and Data Creation to Evaluation

**arXiv ID:** 2602.18483 | [PDF](https://arxiv.org/pdf/2602.18483v1)

**作者:** Adriana Alvarado Garcia `[一作]` (IBM Research), Karla Badillo-Urquiola `[通讯]` (University of Notre Dame)

**通讯引用:** 1164 | [OpenAlex ID](https://openalex.org/A5074522618)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过22个半结构化访谈，研究了人工智能从业者在生成、开发与评估大型语言模型（LLM）红队（red‑teaming）数据集时的实践与动机，揭示了风险概念化、数据选择与评估过程中的社会技术因素；

**💡 创新点**

创新点在于首次系统性地将社会技术视角与人机交互研究方法引入红队评估，提供了关于风险定义、数据代表性与多轮交互缺失的实证洞察，并为HCI研究者提出了扩展红队实践的三大路径；

**🔧 技术方法**

研究方法主要采用半结构化访谈、主题分析（thematic analysis）及访谈记录编码；

**📊 数据集**

使用的数据集并非模型训练数据，而是访谈对象所涉及的22名红队数据集创建者/评估者的背景与经验；

**📈 对比分析**

文章并未对算法或模型性能进行量化比较，而是通过对访谈结果的归纳，阐释不同数据实践对红队效果的影响，未给出具体指标或性能提升；

**⚠️ 局限性**

局限性包括样本主要来自学术界，行业参与度低；访谈依赖自述，可能存在回忆偏差；研究聚焦LLM红队数据集，未覆盖多模态或非英语场景；未提供实验验证红队方法的效果差异。

---

## 401. ComplLLM: Fine-tuning LLMs to Discover Complementary Signals for Decision-making

**arXiv ID:** 2602.19458 | [PDF](https://arxiv.org/pdf/2602.19458v1)

**作者:** Ziyang Guo `[一作]` (Northwestern University), Jessica Hullman `[通讯]` (Northwestern University)

**通讯引用:** 4589 | [OpenAlex ID](https://openalex.org/A5068008545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出基于决策理论的后训练框架Fine-Tuning LLM，以发现并提取能够提升多代理决策质量的补充信号；

**💡 创新点**

创新点在于将互补信息价值量化为奖励目标，训练LLM专门寻找相对于已有推荐能够带来增量决策收益的可解释信号，而非传统的特征或主题生成；

**🔧 技术方法**

技术手段包括决策理论定义互补价值、监督微调（SFT）与链式推理生成标签、强化学习（GRPO）对奖励进行优化、以及Qwen3系列LLM作为模型骨干；

**📊 数据集**

使用的数据集涵盖合成实验、医学诊断（MIMIC-CXR radiology reports +血检标签）、内容审核（DICES 人机对话 toxicity 标注）以及学术论文评审（Review5K 文本与人类评审）；

**📈 对比分析**

与零样本/少样本学习、BERTopic主题生成、HypotheSAE假设生成及不可解释基准对比，结果显示Complementary Signals在所有实验中均实现最高互补信息价值，且在论文评审任务中提升LLM准确率至约79.7%，接近不可解释基准；

**⚠️ 局限性**

局限性在于依赖估计的生成模型，可能漏掉罕见但重要的信号；当监督者与推荐者相同（如人机协同）时，补充信号可能导致后验偏移；未来需引入持续学习以适应信念更新。

---

## 402. Understanding Empirical Unlearning with Combinatorial Interpretability

**arXiv ID:** 2602.19215 | [PDF](https://arxiv.org/pdf/2602.19215v1)

**作者:** Shingo Kodama `[一作]` (Middlebury), Nir Shavit `[通讯]` (MIT)

**通讯引用:** 11837 | [OpenAlex ID](https://openalex.org/A5037659256)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在所谓忘记（unlearning）后模型权重中知识的残留与恢复现象，并利用组合可解释性框架在可解释的两层网络上评估三种常见忘记方法的效果及其在不同微调场景下的恢复难易度。

**💡 创新点**

创新点在于首次将组合可解释性应用于忘记研究，能够直接检视权重中的知识结构；其次揭示即使在非目标数据微调下已忘记的概念也能恢复，并证明恢复主要沿原始权重方向进行，从而阐明忘记过程的机制。

**🔧 技术方法**

使用了组合可解释性框架、梯度上升（Gradient Ascent）、任务向量（Task Vector）和隐私保护蒸馏（Privacy Preserving Distillation）等忘记技术，并对权重差异进行向量分解分析。

**📊 数据集**

使用合成的DNF（Disjunctive Normal Form）逻辑公式数据集，包括共享变量与不共享变量的两种设置。

**📈 对比分析**

通过衡量恢复时间（达到80%真阳性率所需微调步数）、成功率以及权重余弦距离等指标，对三种忘记方法进行比较；结果显示梯度上升和任务向量在目标数据微调下恢复最快，但在所有数据或非目标数据微调下仍能恢复；隐私保护蒸馏则在非目标数据微调下恢复较慢。

**⚠️ 局限性**

研究仅在极简的两层可解释网络中验证，未验证对大型深度模型的适用性；实验使用的是合成逻辑数据，缺乏对真实语言或视觉任务的评估。

---

## 403. How Retrieved Context Shapes Internal Representations in RAG

**arXiv ID:** 2602.20091 | [PDF](https://arxiv.org/pdf/2602.20091v1)

**作者:** Samuel Yeh `[一作]` (University of Wisconsin Madison), Sharon Li `[通讯]` (University of Wisconsin Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析大型语言模型在检索增强生成（RAG）中的隐藏表示，探究检索上下文的不同类型如何影响模型内部状态及生成结果。

**💡 创新点**

创新点在于从内部表示层面系统地评估相关、干扰和随机检索文档对模型隐藏状态的影响，并揭示层级处理对检索证据利用的机制。

**🔧 技术方法**

主要技术包括使用指令调优的LLM（如Qwen3-Next-80B、Qwen1.5-72B、Llama2-70B）、对检索文档进行分类的GPT-5、PCA可视化、层级表示差异分析以及LLM-as-Judge评估。

**📊 数据集**

实验数据集涵盖四个问答数据集：TriviaQA、NQ、PopQA和StrategyQA，检索数据库为1.4万亿词的MassiveDS，检索器为Contriever。

**📈 对比分析**

与仅查询或随机上下文的基线对比，结果表明相关文档对模型内部状态影响小且多能提升置信度，但在难题上效果有限；随机文档导致显著表示漂移并诱发模型回避；多文档场景中只需一篇相关文档即可稳定表示并保持准确率；整体来看，LLM在存在可靠证据时对噪声具有一定抗干扰性，但在缺乏内部知识的难题上检索帮助有限。

**⚠️ 局限性**

主要局限是仅关注最终提示词的隐藏表示，未捕捉到上下文或生成过程中的词级动态，且实验聚焦于指令调优模型，未探究更广泛的模型或更细粒度的表示分析。

---

## 404. The Neural-Wave Quick Escape Manual 2036: A Field Guide to Adversarial Living in the Era of "Empathic" AIoT

**arXiv ID:** 2602.19139 | [PDF](https://arxiv.org/pdf/2602.19139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 405. Contrastive meta-domain adaptation for robust skin lesion classification across clinical and acquisition conditions

**arXiv ID:** 2602.19857 | [PDF](https://arxiv.org/pdf/2602.19857v1)

**作者:** Rodrigo Mota `[一作]` (Universidade Federal de Pernambuco), Tsang Ing Ren `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于对比预训练和引导微调的域适应策略，用于提升皮肤病变分类模型在临床图像中的鲁棒性。

**💡 创新点**

创新点在于将对比学习与元域适配相结合，利用视觉元域概念在源域（大规模镜下图像）与目标域（临床图像）之间对齐特征，同时通过引导微调（guided‑tuning）在保持已有知识的同时适应新域，解决领域漂移与灾难性遗忘。

**🔧 技术方法**

使用EfficientNet骨干网络，结合多变换对比损失（InfoNCE）和多正样本对比损失，采用颜色统计迁移和模糊等视觉增强进行元域生成；训练过程包括对比预训练、基于校准子集的元域自适应和引导微调。

**📊 数据集**

实验数据集包括HAM10000（高分辨率镜下图像）、PAD‑UFES‑20（手机临床图像）和DDI（多种肤色的临床手机图像）。

**📈 对比分析**

与传统无预训练、仅微调或仅增强的基线相比，CT‑pretrain+GT策略在PAD‑UFES‑20和DDI上的准确率分别提升至0.88/0.84、F1为0.84/0.81，显著优于单一对比预训练或单纯微调；同时保持了对源域HAM10000的性能，避免了灾难性遗忘。

**⚠️ 局限性**

限制在于需要额外的源域数据用于预训练和元域生成，对目标域样本的标注量仍有限；此外，颜色和模糊迁移方法在极端照明或特殊设备下可能效果不佳，需进一步验证跨设备的普适性。

---

## 406. Exploration of Always $S$-Connected Temporal Graphs

**arXiv ID:** 2602.19657 | [PDF](https://arxiv.org/pdf/2602.19657v1)

**作者:** Duncan Adamson `[一作]` (University of St Andrews), Paul G Spirakis `[通讯]` (University of Liverpool)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文提出并研究了总是 S‑连通的时变图（Generalization of always connected graphs），给出了针对 m = |S|、平均度 Δ 的探索时间上界，并将其应用于树宽约束、区间图、格子图等特殊图类，实现了更快的探测调度。

**💡 创新点**

创新点在于引入 S‑连通概念，构造了新的组合式证明与 (r,b)-划分技术，推导出 O(n m³ Δ^1.5 log^1.5 n) 的探测上界，并进一步得到树宽为 k 时的 O(n^{4/3} k^{5.5} log^{2.5} n) 结果，显著改进了先前的 O(n²) 与 O(n^{3/2}√Δ log n) 上界。

**🔧 技术方法**

主要使用了组合分析、递归分割、(r,b)-划分与树宽分解、动态图的路径覆盖和时间步推理等技术，结合多层次的子图构造与迭代探索策略。

**📊 数据集**

该工作为理论分析，未使用具体实验数据集，所有结果均为理论证明与渐进上界。

**📈 对比分析**

与之前的 O(n²) 和 O(n^{3/2}√Δ log n) 结果相比，本文在小树宽和区间图中取得了更低的时间复杂度，实验或仿真未给出，但理论上表现显著提升；对格子图等特定结构也给出了更优的上界。

**⚠️ 局限性**

局限性包括：缺乏对树宽约束图的下界研究，仍无法证明所给上界是否最优；对平面图、格子图等严格类的探测时间尚未达到最优，开放问题仍在；此外，结果仅在理论层面，缺乏实际实验验证。

---

## 407. SkillOrchestra: Learning to Route Agents via Skill Transfer

**arXiv ID:** 2602.19672 | [PDF](https://arxiv.org/pdf/2602.19672v1)

**作者:** Jiayu Wang `[一作]`, Frederic Sala `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 SkillOrchestra 框架，学习可重用的技能手册，实现基于技能的多轮模型与代理编排。

**💡 创新点**

通过技能抽象而非端到端 RL，将编排知识拆分为模式级洞察、细粒度技能和代理档案，实现可转移、可解释、样本高效的路由决策。

**🔧 技术方法**

结合 LLM 生成技能、Beta 分布估计代理能力、Pareto 验证挑选子手册、基于技能的决策公式，实验使用 Qwen、Mixtral 等大模型与工具池。

**📊 数据集**

在十个 QA 与多跳、数学推理基准（NQ、HotpotQA、MATH、AMC23、FRAMES 等）上进行训练与评估。

**📈 对比分析**

与启发式、判别式、RL 路由（Router-R1、ToolOrchestra）以及专有模型对照，SkillOrchestra 在精度-成本 Pareto 前沿领先，RL 方法最多提升 22.5% 精度、成本降低 2×，且可迁移至不同 orchestrator。

**⚠️ 局限性**

需要先生成并维护技能手册，手册细粒度需与 orchestrator 能力匹配，过细或过粗都可能导致性能下降；当前仅在大语言模型与工具池场景验证，跨领域泛化与持续更新机制待完善。

---

## 408. Covering a Polyomino-Shaped Stain with Non-Overlapping Identical Stickers

**arXiv ID:** 2602.19525 | [PDF](https://arxiv.org/pdf/2602.19525v1)

**作者:** Keigo Oka `[一作]` (Google), Akira Iino `[通讯]` (Nippon Hyoron Sha, Co., Ltd.)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对在多格形（polyomino）约束下的污渍覆盖问题进行研究，完成了所有可“始终可覆盖”的多格形的分类，并给出了 O(1) 的判定算法；同时构造了最小的不被任何多格形覆盖的污渍实例；并证明了给定多格形能否覆盖另一多格形的平面版和一维版问题分别为 NP‑complete。

**💡 创新点**

创新点包括：①首次给出“始终可覆盖”多格形的必要且充分条件，证明尺寸≥7 的多格形永不满足；②在极小尺寸（≤6）下通过计算搜索完成完整分类；③设计了基于模拟退火与贪婪搜索的反例构造方法；④通过 3‑色着色与 Golomb ruler 构造的多格形与一维模板证明了平面与一维版本的 NP‑完整性。

**🔧 技术方法**

主要技术手段有：计算机枚举与深度优先搜索验证小尺寸多格形的可覆盖性；模拟退火搜索用于寻找反例；证明时采用反证法与几何变换（旋转、反射、平移）；对 NP‑完整性使用多格形构造与 3‑色着色、诱导子格图的多格形与 Golomb ruler 之间的多对多映射。

**📊 数据集**

使用了完整的多格形枚举数据集（所有尺寸≤6 的多格形）以及从搜索得到的最小反例集；一维模板与多格形模板均由程序自动生成，未使用公开的标准数据集。

**📈 对比分析**

算法的判定复杂度为 O(1)，且在非可覆盖情形下可以在线性时间输出一个反例多格形；NP‑完整性证明通过多格形构造与 3‑色着色的多对多归约完成，说明问题在一般情况下是不可多项式可解的。

**⚠️ 局限性**

主要局限在于仅针对多格形进行分析，未考虑带孔的多格形或更高维度的连续形状；算法仅在已知尺寸≤6 的污渍时有效，尺寸≥7 时直接判定为不可覆盖；此外，一维模板的 NP‑完整性结果仍属于理论证明，缺乏实际实例验证。

---

## 409. Temporal-Logic-Aware Frontier-Based Exploration

**arXiv ID:** 2602.18951 | [PDF](https://arxiv.org/pdf/2602.18951v1)

**作者:** Azizollah Taheri `[一作]` (Northeastern University), Derya Aksaray `[通讯]` (Northeastern University)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5053550436)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于前沿探索的算法，在未知离散环境中利用自动机理论完成语法安全线性时序逻辑（scLTL）任务，并通过commit状态机制避免不可逆的任务进展；

**💡 创新点**

创新点在于引入commit状态概念并利用自产品自动机检测不可逆进展，同时在前沿选择中同时考虑信息增益、距离与任务进度，实现在未知环境下安全、高效地满足scLTL任务；

**🔧 技术方法**

使用自动机理论（DFA、产品自动机）、scLTL转DFA、增量构造产品自动机、信息增益评估、权重评价与前沿探索策略；

**📊 数据集**

实验使用500个20×20网格地图，随机生成L块并放置P、S标签；

**📈 对比分析**

与现有方法（<cit.>）对比，包含L区域时满足率从35%提升到100%，平均路径长度从56.47步下降到46.20步；整体性能更好；

**⚠️ 局限性**

局限在于假设标签可在可见范围内完全获取，且不考虑需要冒险进入commit状态的情形；仅适用于离散网格，未扩展到连续空间或多机器人协作。

---

## 410. HEHRGNN: A Unified Embedding Model for Knowledge Graphs with Hyperedges and Hyper-Relational Edges

**arXiv ID:** 2602.18897 | [PDF](https://arxiv.org/pdf/2602.18897v1)

**作者:** Rajesh Rajagopalamenon `[一作]` (Indian Institute of Technology Palakkad), Unnikrishnan Cheramangalath `[通讯]` (Indian Institute of Technology Palakkad)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5057428477)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种统一的知识图谱嵌入模型 HEHRGNN，可同时处理包含超边和超关系边的 n-ary 事实，并实现对未知实体的归纳链接预测。

**💡 创新点**

创新点包括：① 统一的事实表示格式，兼容超边与超关系边；② 针对超边/超关系边的四步消息传播机制（Gather‑Apply‑Scatter），使 GNN 能捕获多元语义上下文；③ 端到端的编码器-解码器框架，兼顾深度学习与传统知识图谱推理。

**🔧 技术方法**

采用技术：图神经网络（自定义 MessagePassing）、多层消息传播、权重共享的关系嵌入更新、DistMult/HypE 解码器、PyTorch Geometric 实现。

**📊 数据集**

使用的数据集有：JF17K、FB‑AUTO、M‑FB15K（超边）、WD50K、WikiPeople（超关系边）、Combined‑HEHR（混合）、Yago‑6M（大规模）等。

**📈 对比分析**

与先前的单一模型（HypE、HSimplE、STARE 等）相比，HEHRGNN 在大多数基准上取得了更高的 Hits@k（尤其 Hits@1）和 MRR，证明其在统一嵌入和归纳推理上的优越性；在 Yago‑6M 上仍保持了可接受的性能并显著降低了训练时间。

**⚠️ 局限性**

局限性：在极小数据集（如 FB‑AUTO）上表现不佳，3–4 层 GNN 训练难以收敛；对大规模图的显存/计算需求仍较高，尤其是当事实中包含大量超关系时；目前实验主要集中在传递式模式，归纳模式下的性能尚未充分评估。

---

## 411. Agentic Problem Frames: A Systematic Approach to Engineering Reliable Domain Agents

**arXiv ID:** 2602.19065 | [PDF](https://arxiv.org/pdf/2602.19065v1)

**作者:** Chanjin Park `[一作]` (Seoul National University), Chanjin Park `[通讯]` (Seoul National University)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5091602559)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Agentic Problem Frames (APF) 与 Agentic Job Description (AJD) 框架，构建动态规范与 Act‑Verify‑Refine（AVR）闭环以实现 LLM 代理在工业环境中的可靠性验证。

**💡 创新点**

将问题框架迁移至概率模型的动态规范，强调外部环境约束与闭环反馈而非内部模型微调，并首次将 AJD 定义为工程化合同规范代理。

**🔧 技术方法**

基于问题框架理论、动态规范 S_t、AVR 循环、上下文注入、回调与确认、知识资产化，并结合工具接口如 MCP、A2A 等实现。

**📊 数据集**

示例使用企业旅行系统与航旅预订 API、内部邮件日志、工业设备传感器数据及维护记录等为知识库与验证源。

**📈 对比分析**

通过两条对比案例（代理式商务旅行、工业设备监控）展示任务完成率、错误率降低、用户满意度提升，但未给出量化基准或与现有方法的系统对比。

**⚠️ 局限性**

缺乏大规模实测数据与极端环境下的可扩展性与安全性评估，模型仍依赖人工制定的 AJD 与上下文注入，难以自动化生成。

---

## 412. On the Variability of Source Code in Maven Package Rebuilds

**arXiv ID:** 2602.19383 | [PDF](https://arxiv.org/pdf/2602.19383v1)

**作者:** Jens Dietrich `[一作]` (Victoria University of Wellington), Behnaz Hassanshahi `[通讯]` (Oracle Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文在大规模重建 Google 与 Oracle 生成的 Maven 包时，系统地分析了导致源代码不等价的情况，并归因于构建时代码生成和提交不一致等因素。

**💡 创新点**

创新点在于提出将生成代码的来源与可追溯性纳入构建可复现性评估，并对生成器的非确定性及其安全影响进行细粒度分类。

**🔧 技术方法**

采用源码差异分析、git bisect、macaron、JavaParser 以及自定义的二进制等价关系 daleq 等技术进行验证和比较。

**📊 数据集**

使用先前研究构建的 2,714 对 Artifact（Maven Central vs Google/Oracle）数据集，涵盖 268 对非等价源代码记录。

**📈 对比分析**

通过对比二进制包的哈希与源代码差异，利用等价关系判定相似性，实验表明能识别出 23 包存在缺失或非等价源文件，整体处理时间在数小时级别。

**⚠️ 局限性**

局限性包括仅分析 .java 文件忽略 Kotlin/Scala/Groovy 等语言，缺失源代码信息导致某些包无法完整检索，且生成器的非确定性未能在工具中完全消除。

---

## 413. SplitLight: An Exploratory Toolkit for Recommender Systems Datasets and Splits

**arXiv ID:** 2602.19339 | [PDF](https://arxiv.org/pdf/2602.19339v1)

**作者:** Anna Volodkevich `[一作]` (SB AI Lab), Alexey Vasilev `[通讯]` (SB AI Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一个名为SplitLight的开源工具，用于系统性审计推荐系统数据集及其拆分过程，揭示关键数据特性、泄露风险、冷启动比例及分布偏移等问题。

**💡 创新点**

创新点在于提供统一、可视化、可比较的审计框架，覆盖数据核心统计、时间特征、重复消费、时间泄露、冷启动与分布移位六大维度，且兼具Python库和交互式Streamlit UI。

**🔧 技术方法**

采用Python实现的统计分析与可视化技术，利用时间序列、分布统计、Kolmogorov–Smirnov检验等方法进行泄露检测、冷启动评估和分布偏移量度。

**📊 数据集**

在六个常用推荐数据集上进行案例研究，包括MovieLens-1M/20M、Amazon Beauty、Diginetica、Dressipi、Zvuk等，涵盖评分、点击流与音乐流媒体数据。

**📈 对比分析**

通过在相同拆分策略下对SASRec模型的NDCG@10进行多重实验，展示不同预处理（如保留/移除重复、是否保留时间戳顺序、是否过滤冷启动）对评估指标的显著影响，说明审计结果能显著提高实验可复现性与比较性。

**⚠️ 局限性**

局限性包括：目前仅支持用户-物品-时间三元组的数据格式；对复杂特征（如内容信息、上下文信号）支持有限；缺乏自动化修复建议，需要用户手动解释和调整。

---

## 414. Representation Stability in a Minimal Continual Learning Agent

**arXiv ID:** 2602.19655 | [PDF](https://arxiv.org/pdf/2602.19655v1)

**作者:** Vishnu Subramanian `[一作]` `[通讯]` (Accenture), Vishnu Subramanian (Accenture)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实验了一个仅维护词频向量的最小持续学习代理，持续记录内部表示并通过余弦相似度监测其随时间的稳定与可塑性。

**💡 创新点**

在不引入任何任务目标、优化或重放机制的情况下，揭示了仅凭持久状态和自我比较即可自然产生稳定-可塑性权衡，并建立了可观测的表示稳定性基准。

**🔧 技术方法**

采用词频计数的固定维向量表示，归一化后计算相邻运行的余弦相似度；通过累积计数和词汇表增长统计来跟踪内部状态演化。

**📊 数据集**

文本语料库（未指明公开数据集），在连续的八次执行中逐步加入新文档，并在第5次加入语义上完全不同的文档进行扰动实验。

**📈 对比分析**

通过相邻运行间的余弦相似度以及词汇量/token计数指标进行自对比；结果显示初期相似度快速上升，稳定期接近0.99，扰动后下降至0.8957后恢复至0.998，证明可塑性与稳定性可实现且无灾难性遗忘。

**⚠️ 局限性**

表示向量维度无限增长，缺乏压缩与抽象；未涉及任务性能或交互式学习；仅使用静态文本，无法评估在动态环境或资源受限条件下的可扩展性。

---

## 415. Self-Configurable Mesh-Networks for Scalable Distributed Submodular Bandit Optimization

**arXiv ID:** 2602.19366 | [PDF](https://arxiv.org/pdf/2602.19366v1)

**作者:** Zirui Xu `[一作]` (University of Michigan), Vasileios Tzoumas `[通讯]` (University of Michigan)

**通讯引用:** 952 | [OpenAlex ID](https://openalex.org/A5052733656)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在有限通信带宽、数据速率和网络连通性受限的环境下，利用分布式在线带宽约束下的bandit子模函数协调方法，实现多智能体的可扩展近似最优行动协调。

**💡 创新点**

创新点在于引入信息价值度量（Value of Coordination）并通过主动设计一跳邻居网络来优化信息访问，提供严格的a priori、a posteriori和渐进子最优性界，同时在任意拓扑下保证非零任意时刻性能。

**🔧 技术方法**

采用分布式双重bandit优化（动作选择与邻居选择）结合Exp3/Exp3++等 adversarial bandit 算法，并利用子模函数的曲率和VoC指标构造理论上可证明的性能上界。

**📊 数据集**

使用仿真数据：多摄像头区域监控场景（8摄像头+20摄像头、50摄像头等），模拟不同通信范围、邻居数以及带宽延迟条件。

**📈 对比分析**

与最近邻、随机邻居以及DFS‑SG/DFS‑BSG基线比较，实验表明该方法在覆盖率、收敛速度和实时决策次数上均优于基线，尤其在有限延迟和大规模网络下实现了子线性扩展。

**⚠️ 局限性**

局限性包括：决策时间仍受通信/计算延迟影响，邻居网络被假设为静态且仅为一跳；对移动机器人动态邻居的适配尚未实现，且在极端网络分离情况下理论保证仍不充分。

---

## 416. SCHEMA for Gemini 3 Pro Image: A Structured Methodology for Controlled AI Image Generation on Google's Native Multimodal Model

**arXiv ID:** 2602.18903 | [PDF](https://arxiv.org/pdf/2602.18903v1)

**作者:** Luca Cazzaniga `[一作]` `[通讯]` (Independent Researcher), Luca Cazzaniga (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并验证了面向Google Gemini 3 Pro Image的SCHEMA结构化提示工程方法，涵盖六大专业视觉生产领域，提供三层可扩展控制、模块化标签、强制与禁止约束及失败路由机制。

**💡 创新点**

创新点包括：①三层进阶控制（BASE/MEDIO/AVANZATO）实现从探索到严格交付的控制尺度；②七核心+五可选标签的模块化架构；③将正向“Mandatory”和负向“Prohibitions”约束纳入逻辑约束箱，实现约束先行；④集成决策树实现模型适配与失败路由；⑤在信息设计领域公开验证>95%首生合规率。

**🔧 技术方法**

采用手工实践评估、批量一致性实验、独立40人工作坊验证，以及对Gemini 3 Pro Image API的调用与结果记录，结合参考图像与思考模式等提示增强技术。

**📊 数据集**

数据集包括约4,800张图像、621条结构化提示、850条Replicate API验证预测，以及约300张公开可验证的信息设计图（约75条提示，每条约4张图）。

**📈 对比分析**

通过对比10张图批量一致性测试（叙事式自然语言提示 vs. SCHEMA AVANZATO），发现一致性得分从4–5/10提升至8–9/10（平均提升3.5–4张）。工作坊确认三层控制比例符合预期，信息设计子集首生合规率>95%，表明SCHEMA显著提升专业级交付的可预见性和一致性。

**⚠️ 局限性**

局限性包括：单一实践者数据来源、缺乏自动化客观评估指标（如FID/CLIP/FID）、仅英文提示、仅针对Gemini 3 Pro Image的模型版本，时间窗口集中于2025-2026年，且工作坊验证缺乏正式量表。

---

## 417. A Secure and Private Distributed Bayesian Federated Learning Design

**arXiv ID:** 2602.20003 | [PDF](https://arxiv.org/pdf/2602.20003v1)

**作者:** Nuocheng Yang `[一作]` (Beijing University of Posts and Telecommunications), Kaibin Huang `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种分布式联邦学习框架，采用贝叶斯局部训练和邻居子集选择，利用图神经网络与强化学习自主决策设备连接与模型聚合，以实现Byzantine鲁棒性、隐私保护和收敛加速。

**💡 创新点**

创新点在于：①将贝叶斯后验估计融入联邦学习，直接交换后验而非权重；②把邻居选择建模为优化问题，分析其对安全、隐私和收敛的三重权衡；③提出基于GNN‑RNN的PPO强化学习方法，可处理动态邻居数的无中心网络；④不依赖重加密或零知识证明，显著降低计算与通信开销。

**🔧 技术方法**

技术手段包括：贝叶斯推理（MAP与ELBO）、图注意力网络（GAT）、递归神经网络（RNN）、PPO强化学习、OFDMA链路调度与功率分配。

**📊 数据集**

使用 MNIST、EMNIST（字母/数字）与 CIFAR‑10 数据集，分别配合 LeNet、EMNIST 相关模型和 ShuffleNet 进行实验。

**📈 对比分析**

与三种基线比较：层级 DFL（THDFL）、加密+zk‑SNARK DFL（PT‑DFL）以及随机连接的传统 DFL。实验显示，所提方法在 Byzantine 识别率上可提升 13–30%（相较 THDFL/随机），收敛速度与时间更快，且整体计算延迟仅 ~30 ms，远低于加密/证明方案的秒级/分钟级延迟。

**⚠️ 局限性**

局限性：假设设备在学习过程中保持静止，未处理动态加入/离开；强化学习模型需离线训练，可能受网络环境变化影响；隐私和安全约束基于理论阈值，实际环境下可调参数需进一步实验；对极端非 IID 分布和大规模网络的鲁棒性待验证。

---

## 418. "Write in English, Nobody Understands Your Language Here": A Study of Non-English Trends in Open-Source Repositories

**arXiv ID:** 2602.19446 | [PDF](https://arxiv.org/pdf/2602.19446v1)

**作者:** Masudul Hasan Masud Bhuiyan `[一作]` (CISPA Helmholtz Center for Information Security), Cristian-Alexandru Staicu `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对2015-2025年间GitHub上的讨论、代码和文档进行大规模、跨时间的多语言使用量化分析，揭示非英语参与度的显著增长及其对项目可见性、合作和问题解决效率的影响。

**💡 创新点**

①首次系统地量化OSS中文化进程，覆盖10年时间跨度；②将讨论、代码、文档三大内容维度统一分析；③结合统计检验和多维度参与度指标，揭示非英语项目的可见性与协作瓶颈。

**🔧 技术方法**

使用Lingua和Google Cloud Translation API进行高置信度语言识别；通过GHArchive和GitHub REST API收集原始事件；采用分层抽样、字符串解析、词法分析提取代码元素；利用Kruskal-Wallis和Mann-Whitney U检验比较不同语言类别的参与度指标。

**📊 数据集**

基于GHArchive（约4.3 TB压缩JSON，包含476 M+仓库、78 M+开发者）与GitHub REST API（按月采样新文件补丁），构成跨语言、跨年份的完整数据集。

**📈 对比分析**

通过对非英语、英语主导、混合三类仓库在评论数、贡献者数、星标数、问题解决时长等四项指标进行分桶比较，并用非参数检验评估显著性；结果显示非英语仓库在大多数维度上均处于劣势，效应量（η²）从0.019到0.186不等，表明语言归属对项目参与度有显著影响。

**⚠️ 局限性**

①语言识别在短句、代码混合文本上存在误判，导致非英语比例为下限；②对代码的采样只保留新文件且字符数≥500，可能漏掉重要的小幅更新；③仅覆盖公共GitHub仓库，无法推广至GitLab、Bitbucket或私有项目；④时间窗口限制可能低估老旧项目的真实活跃度。

---

## 419. Conditionally Site-Independent Neural Evolution of Antibody Sequences

**arXiv ID:** 2602.18982 | [PDF](https://arxiv.org/pdf/2602.18982v1)

**作者:** Stephen Zhewen Lu `[一作]` (University of California, Berkeley), Yun S. Song `[通讯]` (University of California, Berkeley)

**通讯引用:** 13955 | [OpenAlex ID](https://openalex.org/A5050837693)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种条件单独位点神经进化模型（cSNE），用于模拟抗体亲和力成熟过程并预测变体效应。

**💡 创新点**

创新点在于将深度学习的表达能力与连续时间马尔可夫链（CTMC）相结合，利用神经网络参数化每个位点的速率矩阵，从而在保持可计算性的同时捕捉序列间的共变效应，并通过理论证明该模型是序列点突变过程的一阶近似。

**🔧 技术方法**

使用了神经网络（基于ESM‑2 150M预训练模型）来输出位点速率矩阵，并结合Gillespie采样算法、分类器引导的指导（Guided Gillespie）以及梯度近似的TAG方法进行序列生成与优化。

**📊 数据集**

训练数据来自约200万条克隆树的演化转移，涵盖五个公开数据源（OAS、CoV‑AbDab等），并在FLAb2基准的四个DMS实验以及多种抗原靶向的亲和力模拟中进行评估。

**📈 对比分析**

与多种基线模型（AbLang‑2、DASM、ProGen2、ESM‑2）在零样本变体效应预测和抗体设计任务上进行比较，结果显示cSNE在所有DMS数据集上均取得最高或次高的Spearman相关性，并在引导亲和力成熟与局部CDR优化中实现显著性能提升。

**⚠️ 局限性**

局限性包括：仅采用一阶近似导致对长分支长度的误差上升；忽略插入/缺失操作，限制仅适用于等长序列；模型对极长分支或非抗体蛋白的泛化能力尚未验证。

---

## 420. LunaAI: A Polite and Fair Healthcare Guidance Chatbot

**arXiv ID:** 2602.18444 | [PDF](https://arxiv.org/pdf/2602.18444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 421. Fast and simple multiplication of bounded twin-width matrices

**arXiv ID:** 2602.20023 | [PDF](https://arxiv.org/pdf/2602.20023v1)

**作者:** László Kozma `[一作]`, Michal Opler `[通讯]` (Czech Technical University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了基于双宽度（twin-width）的矩阵-向量乘法与矩阵乘法加速算法，能够在仅预处理一次后对任意向量实现近线性时间乘法；

**💡 创新点**

创新点在于不需要事先构造或知道收缩序列，直接利用双宽度上界即可完成预处理和乘法；

**🔧 技术方法**

采用了1-矩形分解、Hamming一致性分析、平衡对称收缩序列、混合自由矩阵的几何分解等技术；

**📊 数据集**

论文主要在理论层面，未使用具体数据集进行实验；

**📈 对比分析**

与传统的O(n^3)或O(n^2.81)算法相比，单侧双宽度矩阵可在O(n^2)预处理后实现O(n)乘法，混合自由矩阵可在O(d n)乘法，显著优于之前的O(d n log n)或O(d n^2)结果；

**⚠️ 局限性**

局限在于对任意双宽度矩阵仍需2^{O(d)}的乘子，且目前仅适用于二进制矩阵，对更一般域或半环尚未扩展。

---

## 422. Chasing Ghosts: A Simulation-to-Real Olfactory Navigation Stack with Optional Vision Augmentation

**arXiv ID:** 2602.19577 | [PDF](https://arxiv.org/pdf/2602.19577v1)

**作者:** Kordel K. France `[一作]` (University of Texas at Dallas), Rohith Peddi `[通讯]` (University of Texas at Dallas)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5093581582)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并验证一套完整的无人机嗅觉导航系统，使用最小化传感器实现在线气味源定位，并可选用视觉加速定位过程。

**💡 创新点**

首次实现仅靠嗅觉的无人机源定位，提供可复现的硬件、仿真到真实的端到端框架，并开源所有代码、硬件设计与数据集；同时实现双模式（嗅觉+视觉）并展示性能提升。

**🔧 技术方法**

自制嗅觉处理单元（MOX/EC传感器+ESP32），边缘学习导航策略（OIO、Expected SARSA(λ)），在Gymnasium仿真中训练，再部署至DJI Tello；利用时间延迟估计、双时尺度指数平滑进行气味方向估计，并用信息对比学习（COLIP）结合CLIP视觉编码实现嗅觉-视觉关联。

**📊 数据集**

使用自行构建的嗅觉-视觉联合数据集（基于公开嗅觉-视觉数据集的融合）以及实验室采集的酒精源环境传感记录；未使用预先训练的公开数据集，所有训练均在仿真中完成。

**📈 对比分析**

通过与单嗅觉OIO、Expected SARSA(λ)以及加入视觉的版本进行对比；平均定位时间在94–112秒之间，加入视觉可缩短数秒；电化学传感器相对慢；所有实验均成功定位源。

**⚠️ 局限性**

仅在单一化学物质（乙醇）和有限传感器类型下验证；受气流扰动、IMU漂移影响；统计样本受限，未在多源或动态环境中进一步验证。

---

## 423. TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models

**arXiv ID:** 2602.18884 | [PDF](https://arxiv.org/pdf/2602.18884v1)

**作者:** Zhenkun Gao `[一作]` (East China Normal University), Yuan Xie `[通讯]` (East China Normal University)

**通讯引用:** 30903 | [OpenAlex ID](https://openalex.org/A5100385336)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了大规模时序程序理解数据集TPRU，并使用强化学习（GRPO）对小规模多模态大型语言模型进行微调，以提升其在时序推理和程序理解任务上的表现。

**💡 创新点**

创新点包括：① 通过多来源（机器人操作、LEGO装配、GUI导航、虚拟导航）构建真实的多帧序列，系统化设计三种任务（时间重排、下一帧预测、前帧回顾）并加入负样本；② 引入RL微调策略，使小模型在时序推理上超越更大模型；③ 通过TPRU-Test手工挑选的难题验证方法的泛化能力。

**🔧 技术方法**

采用的技术：强化学习微调（GRPO）、基于Qwen2.5-VL的多模态模型（7B/32B/3B版本）、RL奖励设计、负样本对抗训练。

**📊 数据集**

使用的数据集：TPRU-25k（24,750 QA对、126,000图像，四大真实场景）以及手工构造的TPRU-Test（461个挑战样本）。

**📈 对比分析**

与 GPT‑4o、LLaVA‑Video、SmolVLM2、Long‑VITA、Qwen2.5‑Omni 等多模态模型进行对比；在TPRU‑Test、MuirBench、LEGO‑Puzzles等基准上，TPRU‑7B从50.33%提升至75.70%，并在多项子任务上显著超越 GPT‑4o 与更大模型，表明小模型经过专门训练后可取得 SOTA 级别的时序推理性能。

**⚠️ 局限性**

局限性：数据集覆盖的四大场景仍有限，未包含更广泛的日常活动；RL微调过程计算成本较高；模型在极大规模或长序列推理方面仍有提升空间。

---

## 424. SLDP: Semi-Local Differential Privacy for Density-Adaptive Analytics

**arXiv ID:** 2602.18910 | [PDF](https://arxiv.org/pdf/2602.18910v1)

**作者:** Alexey Kroshnin `[一作]` (Weierstrass Institute for Applied Analysis and Stochastic), Alexandra Suvorikova `[通讯]` (Weierstrass Institute for Applied Analysis and Stochastic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种半本地差分隐私框架 SLDP，通过交互式本地随机化在不需要可信中介的情况下私密地构造基于 k‑匿名的自适应分区，从而提升密度感知统计任务的精度。

**💡 创新点**

引入以隐私区域为邻接关系的 SLDP 定义，解耦迭代细化过程的隐私预算，使多轮细化不消耗额外预算，并通过交互式协议在本地化随机化基础上构造隐私分区，兼顾局部几何适配与强隐私保证。

**🔧 技术方法**

交互式本地随机化、拉普拉斯噪声注入、k‑匿名分区、隐私区域定义、理论隐私分析与实验评估。

**📊 数据集**

合成二维正态分布数据、加州住房地理坐标数据，以及四个真实地理数据集（Gowalla、Brightkite、Geolife、Porto Taxi）用于空间查询实验。

**📈 对比分析**

与中心化 DP、标准 LDP、Geo‑Indistinguishability、PrivTree（中心化）和 LDP‑KDTree（本地化）进行对比；实验显示 SLDP 在均方误差、F1 分数及空间查询相对误差等指标上普遍优于标准 LDP，且与中心化 DP 接近，证明了更好的隐私‑实用性平衡。

**⚠️ 局限性**

需交互式通信并依赖诚实但好奇的服务器，无法完全避免服务器侧信息泄露；算法对 k 参数敏感且在大规模数据下通信/计算成本较高；隐私预算分配需手动划分，且在极端稀疏区域估计仍可能不准确。

---

## 425. Hepato-LLaVA: An Expert MLLM with Sparse Topo-Pack Attention for Hepatocellular Pathology Analysis on Whole Slide Images

**arXiv ID:** 2602.19424 | [PDF](https://arxiv.org/pdf/2602.19424v1)

**作者:** Yuxuan Yang `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 7920 | [OpenAlex ID](https://openalex.org/A5039812471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了专门针对肝细胞癌（HCC）病理切片的多模态大型语言模型 Hepato-LLaVA，同时构建了覆盖三尺度（WSI、ROI、Patch）的 33K 题答对列表数据集 HepatoPathoVQA，并在模型中引入稀疏拓扑包注意力（Sparse Topo-Pack Attention）来压缩和聚合病理图像特征。

**💡 创新点**

创新点：
① 在滑动窗口级别引入稀疏拓扑包注意力，显式建模 2D 病理组织拓扑，显著降低特征冗余；
② 采用层级化包摘要与全局上下文交互的结构，保持全局信息同时聚合局部诊断证据；
③ 通过三阶段预训练（MAE→MoCo→LoRA 指令微调）和 Q-Former 视觉连接器实现高效跨模态对齐；
④ 以 Gemini‑3‑flash 生成多尺度高质量 QA 对，构成首个面向 HCC 的多尺度 VQA 数据集。

**🔧 技术方法**

技术：
- 大规模语言模型与 Transformer 结构；
- 稀疏拓扑包注意力（hierarchical sparse attention）；
- MAE 自监督预训练、MoCo 对比学习、LoRA 微调；
- Q-Former 视觉连接器；
- Gemini‑3‑flash 生成式数据扩充；
- 低秩适配（LoRA）与多尺度指令微调。

**📊 数据集**

使用的数据集：
- 200 份 HCC WSI（内部）用于构建 HepatoPathoVQA；
- HepatoPathoVQA（33K QA 对，覆盖 WSI/ROI/Patch 三尺度）；
- HepatoPathoCaption（3.3K 图文配对）用于视觉‑语言对齐；
- HepatoPathoBench（3.0K 对）用于评估；
- 预训练集：TCGA、HCMI、内部 WSIs（约 10K）用于 MAE/ MoCo 预训练。

**📈 对比分析**

比较方法：在多模态医学 LLM（HuatuoGPT、Lingshu）、缩略图基准（Quilt‑LLaVA、Patho‑R1）和 WSI 基准（SlideChat、WSI‑LLaVA）上进行对比，使用 METEOR、WSI‑P（基于 LLM 评估）、单/多选准确率等指标。Hepato‑LLaVA 在 Avg 分数上达到 0.83，远超最佳 WSI 对手 SlideChat 的 0.66；在开放式任务中 WSI‑P 为 0.79（形态）/0.75（诊断），闭合式任务形态单选 0.97、诊断多选 0.88，且在三尺度（WSI/ROI/Patch）均维持 0.82–0.83 的高性能。

**⚠️ 局限性**

局限性：
- 数据集规模受限（200 份 WSIs），可能影响模型在更大样本量或不同病理类型上的泛化；
- 依赖 Gemini‑3‑flash 生成的多尺度文本，对生成质量的敏感性未完全评估；
- 模型在跨模态对齐和指令微调阶段仍需要大规模算力和预训练资源；
- 目前仅针对 HCC 病理，扩展到其他肿瘤类型需要重新构建数据集与迁移学习；
- 仍存在对极低分辨率细节的捕捉不足，需进一步优化多尺度融合策略。

---

## 426. ucTrace: A Multi-Layer Profiling Tool for UCX-driven Communication

**arXiv ID:** 2602.19084 | [PDF](https://arxiv.org/pdf/2602.19084v1)

**作者:** Emir Gencer `[一作]` (Koç University), Didem Unat `[通讯]` (Koç University)

**通讯引用:** 855 | [OpenAlex ID](https://openalex.org/A5040713791)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一款名为 ucTrace 的多层次 UCX 通信分析与可视化工具，能够跟踪并关联 UCX 传输层事件与 MPI 调用，支持 GPU 设备归因并提供交互式可视化。

**💡 创新点**

创新点在于将 UCX 级别的 transport 事件与 MPI 级别函数一一映射，实现了 GPU 设备层面的归因，并以多层视图展示 CPU‑GPU‑NIC 之间的通信，填补了现有工具在 transport‑level 细粒度分析上的空缺。

**🔧 技术方法**

采用 UCX 代码层拦截、MPI_T、调用栈收集、CUDA 内存分配/释放记录、Python 生成 pickle 进行日志处理，并通过自定义可视化界面展示结果。

**📊 数据集**

实验使用的主要数据集包括 CG Solver 的 Hook_1498 与 nd24k 稀疏矩阵、GROMACS water-cut1.0_GMX50_bare/0024、以及在 MareNostrum5 与 Leonardo 超算上执行的 MPI Allreduce、Eager/RNDV、NUMA 对齐等基准。

**📈 对比分析**

在 MareNostrum5 与 Leonardo 上比较 OpenMPI 与 MPICH 的 Allreduce 算法、不同 UCX 配置下的 eager/rndv 传输方式，以及 NUMA 对齐与无对齐对通信模式的影响，发现通过 ucTrace 可显著定位瓶颈，NUMA 对齐后性能提升约 5 倍。

**⚠️ 局限性**

局限性包括目前仅支持 NVIDIA GPU 的设备归因（缺少 AMD/Intel 的支持）、日志记录会产生一定的时间和存储开销、以及对非 MPI（如 NVSHMEM、NCCL）等通信库的支持仍在计划中。

---

## 427. MAS-FIRE: Fault Injection and Reliability Evaluation for LLM-Based Multi-Agent Systems

**arXiv ID:** 2602.19843 | [PDF](https://arxiv.org/pdf/2602.19843v1)

**作者:** Jin Jia `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出MAS-FIRE框架，对LLM驱动的多智能体系统进行系统化的故障注入与可靠性评估。

**💡 创新点**

构建15种内外部故障的完整分类，并设计三种非侵入式注入机制；引入四层级的故障容忍行为体系，能够细粒度诊断系统成功与失败的根源。

**🔧 技术方法**

采用提示词修改、响应重写和消息路由操控等注入技术；结合系统级可靠性分数与过程级指标（O_f、L_f、S_f）评估容错行为。

**📊 数据集**

使用HumanEval（代码生成）、WikiTableQuestions（表格推理）和WebShop（多轮电商交互）三个标准数据集，对MetaGPT、Table-Critic和Camel三种架构进行实验。

**📈 对比分析**

对比GPT‑5与DeepSeek‑V3两套基础模型，结果显示模型规模提升并不一定提高鲁棒性，架构改进（如共享消息池、迭代闭环）能显著降低多种故障的影响，系统级可靠性分数在最佳场景下提升超过40%。

**⚠️ 局限性**

实验仅覆盖三种MAS架构与有限的故障类型，缺乏对更广泛真实部署环境的验证；注入策略主要针对语义和通信层，可能忽略低层硬件或网络级别的崩溃；并未系统评估跨域迁移与持续学习场景下的鲁棒性。

---

## 428. Global Low-Rank, Local Full-Rank: The Holographic Encoding of Learned Algorithms

**arXiv ID:** 2602.18649 | [PDF](https://arxiv.org/pdf/2602.18649v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在共享Transformer上联合训练加法、乘法与二次模97运算，系统分析其grokking学习轨迹与权重压缩特性，揭示学习过程低维而最终模型全秩的对立关系。

**💡 创新点**

提出全息编码原则：学习轨迹呈3‑5维低秩，但最终权重在每个矩阵上均为全秩，解释了grokking中的低维控制与高维实现之谜。

**🔧 技术方法**

使用轨迹PCA、单矩阵SVD、联合SVD三种重建方法；通过梯度方向消融、子空间重叠度量等技术评估任务电路可分离性与跨矩阵相关性。

**📊 数据集**

数据集为模数97的所有输入对（9,409个样本），采用50/50划分为训练集和测试集。

**📈 对比分析**

比较方法显示：轨迹PCA仅需5‑6个主成分即可重建>95%准确率；单矩阵SVD或联合SVD在低秩下几乎失效；所有实验均在训练结束后达到>97%的测试准确率。

**⚠️ 局限性**

限制：实验仅在小型Transformer与模数算术任务上进行，可能不适用于大规模自然语言模型或非算法任务；轨迹PCA依赖完整训练检查点，缺乏可迁移性。

---

## 429. Spritz: Path-Aware Load Balancing in Low-Diameter Networks

**arXiv ID:** 2602.19567 | [PDF](https://arxiv.org/pdf/2602.19567v1)

**作者:** Tommaso Bonato `[一作]` (ETH Zurich), Torsten Hoefler `[通讯]` (ETH Zurich)

**通讯引用:** 12688 | [OpenAlex ID](https://openalex.org/A5026990786)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出了Spritz，一种基于端点的负载均衡框架，利用标准以太网功能将自适应路由决策从交换机迁移到终端，实现对Dragonfly和Slim Fly等低直径拓扑的路径选择；

**💡 创新点**

创新点在于Source‑Guided Adaptive Routing（通过EV1、EV2控制前两跳），以及两种具体算法Spritz‑Scout与Spritz‑Spray，利用ECN、报文裁剪和超时反馈进行路径探索、缓存与动态调整，且不依赖任何专有硬件；

**🔧 技术方法**

技术手段包括：标准ECMP路由、Explicit Congestion Notification、报文裁剪、超时反馈、DCTCP‑style拥塞控制、权重采样、路径缓存、模拟器htsim扩展；

**📊 数据集**

使用的数据集主要是基于仿真生成的工作负载，包括：微基准（adversarial、permutation）、AI训练收敛（Allreduce、Alltoall）、真实Web搜索数据中心流量轨迹，以及在网络中随机失效2%链路的场景；

**📈 对比分析**

对比基线包括Minimal、Valiant、UGAL‑L、ECMP、Flicr、OPS（u、w）等；实验表明Spritz‑Scout/Spray在流完成时间上相较UGAL‑L提升最多1.8×，在Permutation/Adversarial模式下提升1.1–1.5×，在失败场景中比最佳基线提升2.5–25.4×，并显著降低包丢失和顺序错误；

**⚠️ 局限性**

限制包括：需要在终端存储路径表（Slim Fly最高8.5 MiB，Dragonfly约2.3 MiB），对极小流量工作负载时最短路径偏好不佳；依赖ECN、报文裁剪等标准特性；评估仅在仿真环境下完成，实际部署的网络波动和硬件差异可能影响效果；

---

## 430. StyleStream: Real-Time Zero-Shot Voice Style Conversion

**arXiv ID:** 2602.20113 | [PDF](https://arxiv.org/pdf/2602.20113v1)

**作者:** Yisi Liu `[一作]` (UC Berkeley), Gopala Anumanchipalli `[通讯]` (UC Berkeley)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了StyleStream，一种可以实时进行零样本语音风格转换（音色、口音与情感）的系统。

**💡 创新点**

创新点在于：①通过ASR监督与极小的有限标量量化(FSQ)瓶颈实现了更纯净的内容-风格解耦；②使用连续预量化特征而非离散编码，提升内容保留；③将扩散变换器(DisDiffusion Transformer)与风格编码器结合，实现全非自回归流式生成，端到端延迟约1秒。

**🔧 技术方法**

核心技术包括：HuBERT‑Large + conformer + FSQ（代码表大小45）构成Destylizer；扩散Transformer（DiT）配合条件流匹配(CFM)与Classifier‑Free Guidance实现Stylizer；Vocos改写的因果vocoder；块式因果注意力实现流式推理；文本监督与信息瓶颈。

**📊 数据集**

训练数据：Destylizer使用约1300小时的LMG（LibriTTS+MSP‑Podcast+GLOBE）；Stylizer使用50k小时的Emilia‑EN英语语料；评测基准StyleStream‑Test（300源×10目标，共3000对），使用Emotion Speech Dataset、RAVDESS、GLOBE‑test、L2‑ARCTIC等。

**📈 对比分析**

与FACodec、CosyVoice 2.0、SeedVC、Vevo等基线比较，StyleStream（离线）在WER（9.2%）和S‑SIM/A‑SIM/E‑SIM上均优于前者；实时版本虽WER略高（15.3%）但在风格相似度(S‑SMOS≈4.28、A‑SMOS≈4.37、E‑SMOS≈4.29)上显著领先，显示出高质量且低延迟的风格转换效果。

**⚠️ 局限性**

局限性：①对极短（≤2s）参考语句的风格估计效果下降；②仍存在少量说话人信息泄漏（约3%）；③在GPU资源有限的情况下，流式推理仍需~0.4s处理时延；④对非英语或低资源语言的泛化尚未验证。

---

## 431. Bayesian Meta-Learning with Expert Feedback for Task-Shift Adaptation through Causal Embeddings

**arXiv ID:** 2602.19788 | [PDF](https://arxiv.org/pdf/2602.19788v1)

**作者:** Lotta Mäkinen `[一作]` (Aalto University), Samuel Kaski `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于因果嵌入的贝叶斯元学习方法，并通过专家的对比反馈推断目标任务的因果嵌入，从而在任务级分布移位的环境下实现快速适应并减少负迁移。

**💡 创新点**

创新点包括：①将任务先验条件化于低维因果嵌入，利用因果相似性指导迁移；②在目标任务缺乏数据时，利用专家的对比判断（pairwise similarity）和贝叶斯主动学习（BALD）恢复目标任务嵌入；③给出先验不匹配与负迁移的理论分析，并证明在误差小于任务移位时负迁移被抑制。

**🔧 技术方法**

使用技术：贝叶斯元学习（amortized hierarchical Bayesian meta-learning）、因果嵌入（通过Mendelian randomization、Invariant Causal Prediction等因果发现方法构造）、专家偏好模型（Probit likelihood）与变分推断、贝叶斯主动学习（BALD）进行查询选择。

**📊 数据集**

数据集：1) 合成模拟数据，用于检验负迁移和专家推断的效果；2) UK Biobank临床疾病预测任务（如J44、J45、G45、I21等），每个任务对应不同疾病的二分类预测。

**📈 对比分析**

与传统元学习方法（MAML、DKT、HBM）以及无迁移基线相比，在OOD任务上因果嵌入+专家方案的AUROC更高，负迁移更低，性能在合成数据上显著优于全局先验，且在UK Biobank数据中也能获得显著的AUROC提升。

**⚠️ 局限性**

局限性：①需要先验的因果嵌入，构建和估计因果结构在高维、复杂任务中仍具有挑战；②专家反馈的质量和数量直接影响目标嵌入精度；③方法对嵌入空间光滑性等假设敏感，理论与实践间仍存在一定差距；④在大规模多任务或跨域场景下的可扩展性和计算成本尚未充分验证。

---

## 432. HybridFL: A Federated Learning Approach for Financial Crime Detection

**arXiv ID:** 2602.19207 | [PDF](https://arxiv.org/pdf/2602.19207v1)

**作者:** Afsana Khan `[一作]` (Maastricht University), Anna Wilbik `[通讯]` (Maastricht University)

**通讯引用:** 1521 | [OpenAlex ID](https://openalex.org/A5011946737)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 HybridFL，融合水平与垂直联邦学习，在保持数据本地化的前提下，让交易方与多家银行共同训练金融犯罪检测模型。

**💡 创新点**

创新点在于将交易方（主动方）与各银行的发送方/接收方编码器结合，采用嵌入梯度交换与周期性聚合，解决混合分布下样本不重叠与特征互补问题。

**🔧 技术方法**

使用联邦平均（FedAvg）、嵌入级梯度通信、深度前馈网络（多层感知机）以及交叉熵/焦点损失进行训练。

**📊 数据集**

使用AMLsim（模拟洗钱交易）和SWIFT（实际支付网络）两个数据集，包含交易级特征和各银行账户级特征。

**📈 对比分析**

与中心化模型以及仅使用交易方特征的局部模型对比，HybridFL 在AUPRC、准确率、召回率等指标上接近中心化模型，并显著优于仅交易方模型。

**⚠️ 局限性**

局限在于仍需共享中间嵌入，存在潜在隐私泄露风险；在极端不平衡（SWIFT）时与中心化模型仍有一定差距；未来需要加入差分隐私或安全聚合以平衡隐私与性能。

---

## 433. Generalized Random Direction Newton Algorithms for Stochastic Optimization

**arXiv ID:** 2602.19893 | [PDF](https://arxiv.org/pdf/2602.19893v1)

**作者:** Soumen Pachal `[一作]` (Indian Institute of Technology Madras), Avinash Achar `[通讯]` (TCS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出了一系列基于随机方向随机逼近（RDSA）的广义Hessian估计器，利用噪声函数测量来优化不确定性下的目标函数。

**💡 创新点**

创新点在于通过增加函数测量的数量来降低估计偏差，并证明了这些估计器的渐近无偏性和收敛性。

**🔧 技术方法**

使用了随机方向随机逼近（RDSA）技术，结合了随机牛顿方法进行Hessian估计。

**📊 数据集**

使用了Rastrigin目标函数作为数据集进行数值实验，验证了理论结果。

**📈 对比分析**

与现有的Hessian估计方法（如2SPSA和2RDSA）进行比较，结果显示提出的方法在给定的函数测量预算下表现更优，具有更低的参数误差。

**⚠️ 局限性**

限制在于需要更多的函数测量来实现更低的偏差，这可能在高维情况下增加计算成本。

---

## 434. AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG

**arXiv ID:** 2602.19127 | [PDF](https://arxiv.org/pdf/2602.19127v1)

**作者:** Qijie You `[一作]` (University of Science and Technology Beijing), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14849 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AgenticRAGTracer，一种能够追踪每一步推理过程的多跳检索增强生成（Agentic RAG）基准，并通过自动化流水线生成高质量的多跳问题；

**💡 创新点**

创新点在于首次构建了面向跳数的基准，采用多阶段自动化生成与人工复核相结合的流程，实现了推理链的可追踪性和逻辑完整性；

**🔧 技术方法**

利用大语言模型（如 GPT‑4o‑mini）完成问答生成、逻辑校验与检索策略评估，并使用 ReAct 框架模拟代理执行；

**📊 数据集**

数据集基于 Wikipedia 并与 FlashRAG 索引保持一致，包含 1,305 条 2‑4 跳的推理题，覆盖 11 个领域，且与现有基准无重叠；

**📈 对比分析**

通过 EM、F1 以及 LLM‑as‑a‑Judge 指标评估 13 种主流 LLM（含 GPT、Grok、Qwen、DeepSeek 等），发现随着跳数增加性能急剧下降，GPT‑5 在 4‑跳推理的 EM 仅 22.6%，GPT‑4o 甚至表现异常差；

**⚠️ 局限性**

局限性包括：依赖 Wikipedia 语料，可能无法覆盖更广泛的知识域；生成过程仍受 LLM 偏差影响；虽然人工复核提升质量，但仍需人工投入；且基准未能充分评估代理的自我纠错与动态调优能力。

---

## 435. Identifying Body Composition Measures That Correlate with Self-Compassion and Social Support Within The Lived Experiences Measured Using Rings Study (LEMURS)

**arXiv ID:** 2602.18467 | [PDF](https://arxiv.org/pdf/2602.18467v1)

**作者:** Enerson Poon `[一作]` (University of Vermont), Nick Cheney `[通讯]` (University of Vermont)

**通讯引用:** 1508 | [OpenAlex ID](https://openalex.org/A5112965505)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究利用 InBody770 身体成分分析仪测量学生的体脂、肌肉、水分等指标，并结合自我同情与社会支持的问卷评估，探讨身体成分与心理社会福祉之间的关系；

**💡 创新点**

创新点在于：①将传统 BMI 之外的细分身体成分（如躯干/腿阻抗、ECW/TBW 比值等）作为潜在心理健康生物标志；②采用典型的多变量方法 Canonical Correlation Analysis (CCA) 统一分析身体成分与自我同情、社会支持之间的协方差结构；③从季节和年级（大一 vs 大二）两个维度对关系进行细粒度比较。

**🔧 技术方法**

技术与方法：k‑NN 缺失值插补、CANNONICAL CORRELATION ANALYSIS (CCA)、单因素 ANOVA、Python sklearn 等数据处理与统计工具；问卷工具包括自我同情量表（SCS‑SF）与医疗结果社会支持量表（MOSS）。

**📊 数据集**

数据集为 LEMURS（Lived Experiences Measured Using Rings Study）学生自我报告与 InBody770 测量数据，样本为 156 名大一与大二学生，分三季（秋、冬、春）收集。

**📈 对比分析**

通过比较不同季节与年级的 CCA 相关系数与 ANOVA 统计，发现身体成分与心理社会变量的相关系数在 0.38–0.73 之间，表现出中等到较强的相关性；同时发现两年级间、季节间无显著差异。

**⚠️ 局限性**

局限性包括：样本量小、仅来自一所大学、未包含 3/4 年级学生、未控制文化、经济、性别等混杂因素、季节性收集可能引入偏差、缺乏纵向跟踪导致因果推断受限。

---

## 436. Modularity is the Bedrock of Natural and Artificial Intelligence

**arXiv ID:** 2602.18960 | [PDF](https://arxiv.org/pdf/2602.18960v1)

**作者:** Alessandro Salatiello `[一作]` `[通讯]` (University of Tuebingen), Alessandro Salatiello (University of Tuebingen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对模块化原则在工程、进化、生物学、神经科学与人工智能中的作用进行系统综述，并提出统一的概念框架，强调模块化是自然与人工智能的基础计算原理。

**💡 创新点**

将跨学科领域的模块化研究整合为一个统一框架，并提出通过识别脑功能模块并映射到AI架构的思路，展示模块化如何弥合自然与人工智能之间的差距。

**🔧 技术方法**

主要采用文献综述、概念框架构建与跨领域对比分析技术，未进行实验性模型训练或数据驱动的算法实现。

**📊 数据集**

无自定义数据集；所引用的数据与实验均来自已有研究（如CNN、RL、LLM、神经网络实验等）及相关脑科学实验。

**📈 对比分析**

文章未进行新的性能评估，依托已有研究的对比结果说明模块化能够提升样本效率、泛化能力、能效和抗灾损等计算优势。

**⚠️ 局限性**

局限性包括：缺乏针对脑模块抽象层级的统一标准；将脑功能模块映射到可训练的AI组件仍处于理论阶段；未提供实证验证，仍需实验验证模块化设计的实际收益；对现实数据与算力成本的进一步评估尚未完成。

---

## 437. Computational Complexity of Edge Coverage Problem for Constrained Control Flow Graphs

**arXiv ID:** 2602.18774 | [PDF](https://arxiv.org/pdf/2602.18774v1)

**作者:** Jakub Ruszil `[一作]` (Jagiellonian University), Jakub Zelek `[通讯]` (Jagiellonian University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5113288970)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在控制流图（CFG）中加入约束后实现边覆盖（EC）的测试集构造问题，分析不同类型约束对可行性与计算复杂度的影响。

**💡 创新点**

提出了五类约束（POSITIVE、NEGATIVE、ONCE、MAX-ONCE、ALWAYS），并证明除POSITIVE外，其余约束类型的EC问题均为NP‑complete；同时给出NEGATIVE约束的固定参数可扩展（FPT）算法。

**🔧 技术方法**

利用图论与SAT归约、构造约束图、序列化路径约束（c‑proper路径）以及动态子图构造的技术实现NP‑completeness证明与FPT算法。

**📊 数据集**

该工作为理论研究，未使用实际数据集，所有实验均为理论构造与证明。

**📈 对比分析**

论文未给出实验比较，所述结果仅为计算复杂度与算法可行性的理论证明。

**⚠️ 局限性**

局限性：仅考虑单一约束类型；对ONCE、MAX-ONCE及ALWAYS约束的FPT性尚未给出；未探讨多约束组合与实际软件测试工具的集成。

---

## 438. JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments

**arXiv ID:** 2602.18527 | [PDF](https://arxiv.org/pdf/2602.18527v1)

**作者:** Zhan Liu `[一作]` (Tsinghua University), Chao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 95662 | [OpenAlex ID](https://openalex.org/A5042841794)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 JAEGER 框架，将 3D RGB‑D 与 FOA 多通道音频联合，用于 3D 空间定位与推理

**💡 创新点**

1) Neural IV：可学习的 FOA 空间音频表示，取代传统 STFT intensity vector；2) 将 RGB‑D 的深度编码为 3D 位置编码，实现端到端 3D 视觉-语音联合推理；3) 构建 SpatialSceneQA 61k 真实感 3D 音视问答数据集

**🔧 技术方法**

多模态深度学习（RGB‑D 视觉编码 + 3D 位置编码），FOA 双路音频处理（语义通道 + Neural IV 空间通道），大模型微调（LoRA）以及 3D 视角投影与音频声学渲染

**📊 数据集**

Synthetic SpatialSceneQA（61k 样本，包含 RGB‑D、FOA 音频、3D 标注），基于 Habitat‑Sim 与 SoundSpaces 2.0 合成；使用 LibriSpeech 语音与 HM3D 3D 场景进行数据生成

**📈 对比分析**

与 2D 视听大模型、专用音频定位（BAT）以及 3D 视觉定位模型（N3D‑VLM）对比；JAEGER 在单源定位 MAE 2.21°、重叠源 MAE 13.13°、3D IoU 0.32、视觉中心误差 0.16m，推理准确率 99.5%（单源）/99.2%（双源）

**⚠️ 局限性**

仅在模拟环境中评估，真实世界的音频传输与视角多样性尚未验证；模型规模大，计算成本高；缺乏对不同音频设备、环境噪声的鲁棒性研究

---

## 439. Rules or Weights? Comparing User Understanding of Explainable AI Techniques with the Cognitive XAI-Adaptive Model

**arXiv ID:** 2602.19620 | [PDF](https://arxiv.org/pdf/2602.19620v1)

**作者:** Louth Bin Rawshan `[一作]` (National University of Singapore), Brian Y Lim `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种认知自适应XAI模型CoXAM，用于解释用户在规则（Rules）与权重（Weights）XAI方案下的推理过程，并在前向（forward）和反事实（counterfactual）决策任务中验证其对人类决策的模拟精度。

**💡 创新点**

创新点在于：①从用户思维访谈挖掘出7种具体推理策略；②将这些策略编码进共享记忆-漂移决策-计算理性三元框架，形成可解释的认知模型；③通过强化学习控制器实现自适应策略选择，并以计算理性对速度与准确性权衡；④在实验数据上证明CoXAM比传统代理模型更贴合人类行为，解释了XAI效果差异的认知机制。

**🔧 技术方法**

采用ACT‑R记忆模型、漂移决策模型（DDM）、计算理性策略选择与强化学习（PPO）控制器，结合线性回归与决策树的全局解释，以及基于梯度的反事实计算。

**📊 数据集**

使用两大公开表格数据集：Wine Quality（含6个化学属性）与Mushroom（含6个物理属性），并在每个数据集上训练MLP做基准预测。

**📈 对比分析**

与传统代理模型（KNN、决策树代理、线性回归代理）以及随机/全局SHAP基线比较；在前向任务中CoXAM在NLL、BIC上均优于代理模型；在反事实任务中CoXAM在NLL、BIC上亦优于随机/SHAP；模型拟合后与真实用户表现的相关性高（r≈0.9，RMSE≈2%），表明性能优秀。

**⚠️ 局限性**

局限包括：仅研究了二分类、6维表格数据且只覆盖规则与权重两种XAI方案；实验任务为前向与反事实模拟，未涵盖属性归因、信任等目标；模型基于有限数据集预训练，可能对更复杂、高维或视觉/语言XAI的泛化不足；以及实验设计顺序（前向先于反事实）可能影响生态有效性。

---

## 440. Compliance Management for Federated Data Processing

**arXiv ID:** 2602.19360 | [PDF](https://arxiv.org/pdf/2602.19360v1)

**作者:** Natallia Kokash `[一作]` (University of Amsterdam), Paola Grosso `[通讯]` (University of Amsterdam)

**通讯引用:** 1867 | [OpenAlex ID](https://openalex.org/A5007029875)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个面向合规性的联邦数据处理（FDP）框架，结合了政策即代码、工作流编排和大语言模型辅助合规管理，并通过BraneHub平台实现项目管理与政策生成；

**💡 创新点**

创新点在于将法律法规与组织约束通过LLM和检索增强生成（RAG）技术转化为可执行的OPA/Rego或eFlint策略，同时引入Spatio‑Temporal Purpose‑Aware RBAC Graph（STP‑RBACG）对跨域、跨时空的访问控制进行形式化；

**🔧 技术方法**

使用技术包括Brane容器化工作流引擎、OPA/Rego与eFlint策略引擎、LLM（如Anthropic、OpenAI）结合RAG、Flask前端、Qdrant向量检索、GitHub Issues跟踪需求与代码；

**📊 数据集**

论文以药物研发与医疗数据协作为应用场景，示例工作流基于模拟的CSV数据文件，未使用公开大规模真实数据集；

**📈 对比分析**

方法对比主要通过单元测试验证LLM和OPA两种决策方式的可接受性，未给出量化性能指标，评估侧重功能可用性和合规性可追溯性；

**⚠️ 局限性**

局限包括：LLM在解读法规时可能产生幻觉或错误，需要人工审核；政策生成与执行仍需手工维护；缺乏大规模真实场景的系统级评估和性能基准；以及跨域网络通信与隐私技术（差分隐私等）集成不足。

---

## 441. VIGiA: Instructional Video Guidance via Dialogue Reasoning and Retrieval

**arXiv ID:** 2602.19146 | [PDF](https://arxiv.org/pdf/2602.19146v1)

**作者:** Diogo Glória-Silva `[一作]` (NOVA School of Science and Technology), João Maglhães `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个面向多模态指令视频对话的LVLM模型（命名为VIGIA），能够在对话中进行计划导向问答、图像问题回答（pVQA）以及基于计划检索的对话式视频片段检索；

**💡 创新点**

创新点包括：①整合多模态计划推理与计划检索的统一框架；②通过学习检索起始/结束帧实现对话式视频片段检索；③多阶段训练策略与多任务目标设计；④首个在多模态指令视频对话中支持pVQA的模型；

**🔧 技术方法**

技术细节：基于LLaMA‑3 8B LLM与SigLIP视觉编码器；两层投影器连接视觉与语言模态；检索投影层（start、end、ret）与RoPE时间编码；InfoNCE对比损失用于检索；多任务与多阶段训练（初始化、视觉指令调优、域专化、任务专化）；LLM判别评估对话质量；

**📊 数据集**

使用的数据集：构建的多模态计划对话数据集，扩展TastyVidDial（包含Cooking计划），加入DIY计划（COIN）并通过Claude 3.5生成计划；共约6760对话，涵盖Cooking和DIY；并在对话中插入pVQA问答；

**📈 对比分析**

与现有基线比较：MM‑PlanLLM、TRACE、LLaVA‑1.5、Idefics2、InternVL 3.5、Qwen 3 VL等；在PGAG、pVQA、VSG、CVMR任务上均显著优于基线，pVQA准确率达94%，CVMR R@1提升约20点；在MME通用图像理解基准上仍优于MM‑PlanLLM；

**⚠️ 局限性**

局限性：训练数据规模受限，计划与视频配对稀缺；模型可能带有西方文化偏差，跨语言/文化通用性未知；在跨域通用图像理解任务上存在一定下降；

---

## 442. DICArt: Advancing Category-level Articulated Object Pose Estimation in Discrete State-Spaces

**arXiv ID:** 2602.19565 | [PDF](https://arxiv.org/pdf/2602.19565v1)

**作者:** Li Zhang `[一作]` (University of Science and Technology of China), Cewu Lu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15703 | [OpenAlex ID](https://openalex.org/A5010726528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出 DICArt 框架，利用离散扩散过程进行类别级关节对象位姿估计，并通过流决定器与层级运动耦合机制实现更精准的姿态预测。

**💡 创新点**

创新点包括：① 将位姿估计转化为离散分类任务；② 重新设计逆扩散流程，加入柔性流决定器实现同步且温和的去噪；③ 引入层级运动耦合，利用父子部件的关节约束显著缩小搜索空间；④ 在离散扩散中加入特殊的 [] 标记与动态掩码，提升模型对不确定状态的处理能力。

**🔧 技术方法**

核心技术：离散扩散概率模型（Discrete Denoising Diffusion Probabilistic Models）、Gumbel‑Softmax 流决定器、层级运动耦合模块（父子部件分离 + 关节轴预测 + 运动轴正交约束）、点云/RGB‑D 输入网络以及离散化的旋转/平移表示。

**📊 数据集**

使用了合成 ArtImage、半合成 ReArtMix、真实 RobotArm 数据集，并在各类（Laptop、Eyeglasses、Dishwasher、Scissors、Drawer、7‑part RobotArm 等）上进行实验。

**📈 对比分析**

与 A‑NCSH、GenPose、OP‑Align、ShapePose 等 SOTA 方法比较，DICArt 在 ArtImage 上旋转误差降至 3.2°‑5.3°，平移误差降至 0.02m‑0.09m；在 Drawer、ReArtMix、RobotArm 等数据集上亦比基准低 30‑50% 的误差，且在自遮挡场景下保持低误差。

**⚠️ 局限性**

局限性：离散化导致极细粒度精度受限；对极端遮挡仍有误差；需要先验关节拓扑结构；模型训练依赖大量标注数据；推理时间相较纯连续方法略慢。

---

## 443. Toward AI Autonomous Navigation for Mechanical Thrombectomy using Hierarchical Modular Multi-agent Reinforcement Learning (HM-MARL)

**arXiv ID:** 2602.18663 | [PDF](https://arxiv.org/pdf/2602.18663v1)

**作者:** Harry Robertshaw `[一作]` (King's College London), Thomas C Booth `[通讯]` (King's College London)

**通讯引用:** 2599 | [OpenAlex ID](https://openalex.org/A5003607819)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发并验证了一种层级模块化多智能体强化学习（HM‑MARL）框架，实现两台设备（导丝与支架导管）在体外患者血管模型中的自主导航，完成从股动脉到颈内动脉的长距离导航，并首次实现机械血栓栓塞（MT）血管的体外自主导航。

**💡 创新点**

①引入分阶段专用代理的层级模块化多智能体架构，实现长程导航的可扩展性与可泛化；②利用Soft Actor‑Critic（SAC）分别训练各子任务代理；③在stEVE仿真与真实3D打印血管模型之间完成仿真到现实的迁移；④通过CTA扫描构建多样化血管环境，增强模型的鲁棒性。

**🔧 技术方法**

Soft Actor‑Critic强化学习、层级模块化多智能体（HM‑MARL）、stEVE仿真框架、3D打印血管模型、透明聚酯血管实验平台、光学追踪与G‑code控制的共聚器械运动。

**📊 数据集**

10个基于CTA扫描的患者血管模型（颈部+躯干），随机旋转、缩放增强；单一患者血管模型用于对照实验；体外实验使用同一血管模型。

**📈 对比分析**

与单智能体SAC（SA‑RL‑1）对比。结果：在单个血管模型上，HM‑MARL‑1/10在右侧任务成功率均为100%，左侧和长任务仍高于SA‑RL（0%）。在多血管模型上，HM‑MARL‑10右侧任务成功率约98–92%，左侧约62–66%；SA‑RL左侧和长任务表现差距明显。体外实验中，HM‑MARL‑1/10在右侧任务成功率100%，右侧第二段A_2,3约70–80%，左侧任务全部失败；与专家临床操作对比，专家在左侧20%成功。程序时间从仿真到体外显著增加。

**⚠️ 局限性**

①未能完成左侧任务体外；②机器人运动范围受限，无法评估完整从股动脉到ICA的长任务；③未测量血管壁受力，未验证安全性；④未使用真实X射线影像追踪；⑤样本量和解剖多样性不足；⑥未实现共享/协作式人机交互；⑦缺少未见数据验证。

---

## 444. PerturbDiff: Functional Diffusion for Single-Cell Perturbation Modeling

**arXiv ID:** 2602.19685 | [PDF](https://arxiv.org/pdf/2602.19685v1)

**作者:** Xinyu Yuan `[一作]` (Mila - Quebec AI Institute), Jian Tang `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种将单细胞扰动预测建模为分布层面的功能扩散框架 PerturbDiff，通过把细胞分布嵌入 RKHS 并在该空间上定义扩散过程，实现对细胞群体响应分布的生成。

**💡 创新点**

创新点在于把细胞分布视作随机变量而非单细胞状态，利用核均值嵌入将分布映射到 Hilbert 空间，并在此空间上进行扩散建模，从而捕捉因未观测的微环境、批效应等导致的分布级别变异；同时自然而然得到 MMD 目标，解决了传统方法对分布的平均化处理。

**🔧 技术方法**

核心技术包括核均值嵌入（kernel mean embedding）、RKHS 中的 Gaussian 随机过程、条件扩散逆过程（采用 classifier‑free guidance）、MM‑DiT 变压器结构、以及基于大规模单细胞图谱的边缘预训练。

**📊 数据集**

主要实验数据集：PBMC（信号扰动），Tahoe100M（药物扰动），Replogle（遗传扰动）；预训练使用 61M 细胞的 CellxGene 大型单细胞 RNA‑seq 数据集。

**📈 对比分析**

与 Mean、Linear、scGPT、scFoundation、CPA、STATE、CellFlow、Squidiff 等基线对比，PerturbDiff 在 12+ 评价指标（包括 R²、PDCorr、MAE、DE 相关指标如 AUROC/AUPRC 等）上均显著优于基线，尤其在 DE 预测和低样本场景下表现突出；在 Replogle 上与 STATE 接近，预训练版本进一步提升性能。

**⚠️ 局限性**

局限性包括：模型在超大规模数据上需要中等计算资源且过度扩展可能导致过拟合；对单细胞表达稀疏性仍有一定敏感性；以及对未知扰动或未见细胞类型的泛化仍受限于预训练数据分布。

---

## 445. Impact of AI Search Summaries on Website Traffic: Evidence from Google AI Overviews and Wikipedia

**arXiv ID:** 2602.18455 | [PDF](https://arxiv.org/pdf/2602.18455v1)

**作者:** Mehrzad Khosravi `[一作]` (University of Washington), Hema Yoganarasimhan `[通讯]` (University of Washington)

**通讯引用:** 1083 | [OpenAlex ID](https://openalex.org/A5076336545)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过差分中的差分设计，评估了 Google AI Overviews (AIO) 在美国上线后对 Wikipedia 英文条目访问量的因果影响。

**💡 创新点**

创新点在于利用 AIO 在不同地区分阶段上线与 Wikipedia 多语言版本的天然实验，首次在大规模面板数据上提供 AIO 对信息出版商流量的实证证据。

**🔧 技术方法**

主要采用两面固定效应的差分中的差分估计，并结合 PPML、加权对数回归、事件研究、平行趋势检验等多重稳健性检验。

**📊 数据集**

使用 Wikimedia Analytics API 提供的每日页面浏览量数据，涵盖 52,262 篇英文条目及其在印地语、印尼语、日语、葡萄牙语四种语言版本，总计 161,382 组文章‑语言组合，时间跨度为 2023‑10‑28 至 2024‑08‑14。

**📈 对比分析**

通过将英文版与其他四种语言版本在 AIO 上线前后对比，发现英文版流量平均下降约 15%（文化类最大、STEM 最小），并且所有主要检验（日、周级别、事件研究、PPML 等）均给出一致的负效应，表明估计稳健。

**⚠️ 局限性**

局限性包括：只关注 Wikipedia，缺乏对其它类型出版商的推广；使用语言版本作为 AIO 暴露代理，无法精准捕捉个体点击行为；以及未量化广告收入转移及其他间接经济后果。

---

## 446. CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion

**arXiv ID:** 2602.19140 | [PDF](https://arxiv.org/pdf/2602.19140v1)

**作者:** Sijie Mai `[一作]` (South China Normal University), Shiqin Han `[通讯]` (South China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `67630363-6be0-4f51-ab05-7198250671a5` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CaReFlow 框架，通过直流变换 (rectified flow) 将源模态的分布映射到目标模态，从而显著缩小多模态融合中的模态间分布差距。

**💡 创新点**

核心创新点包括：① 一对多 (one-to-many) 的映射策略，使每个源模态样本能感知目标模态的整体分布；② 自适应宽松对齐 (adaptive relaxed alignment)，对同一样本或同类别模态实行更严格对齐，对不同样本或类别采用更宽松对齐；③ 循环信息流 (cyclic information flow)，通过逆向 rectified flow 保持源模态信息，防止信息丢失。

**🔧 技术方法**

使用了 rectified flow、Euler 步长、时间嵌入、MLP 速度场网络、前向/后向对齐损失以及多模态融合的 MLP/张量融合等技术。

**📊 数据集**

在多任务 MAC 数据集上评估：CMU-MOSI、CMU-MOSEI、CH-SIMS-v2、UR-FUNNY、MUStARD。

**📈 对比分析**

与传统对齐方法（如 ARGF、Deep CCA、CLGSI、MulT、Diffusion Bridge）以及 SOTA 基线（DLF）进行对比，CaReFlow 在 MSA、MHD、MSD 等任务上均实现或超过 SOTA，提升了 Acc、F1、MAE 等指标，且参数量与对齐方法相当或更少。

**⚠️ 局限性**

局限性包括：① 对超参数（如 β、α_f、α_b）仍有一定依赖，需经验性调优；② 目前以语言模态为目标，扩展到多目标模态的效果尚未验证；③ 仍依赖预训练的单模态编码器，对极端稀疏或噪声模态的鲁棒性有待进一步测试。

---

## 447. Audio-Visual Continual Test-Time Adaptation without Forgetting

**arXiv ID:** 2602.18528 | [PDF](https://arxiv.org/pdf/2602.18528v1)

**作者:** Sarthak Kumar Maharana `[一作]` (University of Texas at Dallas), Guan-Ming Su `[通讯]` (Dolby Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种无源数据的音频-视觉持续测试时自适应方法（CAV-MAE+Selective Retrieval，简称 CAV-MAE-CR），通过在连续域中检索并复用注意力融合层的参数来实现模型的持续自适应。

**💡 创新点**

创新点在于：①发现注意力融合层的参数具有跨任务的迁移能力；②引入基于低层统计量的选择性参数检索机制和可扩展的缓冲区；③通过EMA平滑与元素合并实现内存控制与灾难性遗忘的抑制。

**🔧 技术方法**

采用的技术包括：基于KL散度的输入分布匹配、EMA更新、缓冲区合并、READ式的自监督损失（置信度与负熵），以及Transformer的多模态融合结构。

**📊 数据集**

使用的数据集有：Kinetics50-C、VGGSound-C（单模态扰动）以及Kinetics50-2C、VGGSound-2C（双模态扰动）等。

**📈 对比分析**

与传统TTA方法（TENT、EATA、SAR）及音频-视觉TTA基线（READ、SuMi、PTA、BriMPR*）对比，CAV-MAE-CR在所有测试集上均取得显著提升（如在Kinetics50-C上平均提升2.7%，在VGGSound-C上提升约4.95%，在双模态场景下相对READ提升1.25%），同时显著减少灾难性遗忘。

**⚠️ 局限性**

局限性包括：①对统计量近似（对角协方差、均值）可能在极端分布变换下失效；②阈值τ与缓冲区大小η的设置仍需经验调优；③目前仅对融合层参数进行自适应，未考虑更深层参数的可迁移性。

---

## 448. Beyond single-channel agentic benchmarking

**arXiv ID:** 2602.18456 | [PDF](https://arxiv.org/pdf/2602.18456v1)

**作者:** Nelu D. Radpour `[一作]` `[通讯]` (Florida State University), Nelu D. Radpour (Florida State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将人工智能安全评估从单一准确率阈值转向人机协作系统的联合可靠性评估。

**💡 创新点**

创新点在于将AI安全benchmark与安全工程中的冗余、多样化误差模式概念相结合，强调不相关误差模式的风险降低作用。

**🔧 技术方法**

主要技术包括使用大型语言模型和视觉语言模型作为审计层，以及基于瑞士奶酪模型的风险交互分析。

**📊 数据集**

使用LabSafety Bench实验室安全基准数据及其人类信任调查问卷。

**📈 对比分析**

通过与单独AI性能对比，展示即使AI准确率低于70%，其与人类的误差不相关性仍能将整体失效概率显著降低，实验显示联合可靠性提升约45%。

**⚠️ 局限性**

局限在于缺乏对人类遵从度和信任校准的量化建模，且基准仅关注实验室场景，难以直接推广到其他高危领域。

---

## 449. Assessing Risks of Large Language Models in Mental Health Support: A Framework for Automated Clinical AI Red Teaming

**arXiv ID:** 2602.19948 | [PDF](https://arxiv.org/pdf/2602.19948v1)

**作者:** Ian Steenstra `[一作]` (Northeastern University), Timothy W. Bickmore `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套自动化临床AI红队（Red‑Team）评估框架，专门针对大型语言模型（LLM）在心理健康支持中的使用。通过构建质量护理与风险的本体，利用多代理仿真将AI心理治疗师与配备动态认知‑情感模型的模拟患者进行多轮对话，系统化地测量治疗效果、联盟质量、干预忠诚度以及危机事件与不良结果。框架涵盖自动评估指标、交互式可视化仪表盘，并在酒精使用障碍（AUD）情境下，对6款公开AI模型与控制条件进行大规模仿真（369轮会话），揭示关键安全缺口。

**💡 创新点**

创新点包括：
1) 领域特定的红队评估方法，突破单次对话安全测试的局限，捕捉长期、累积的心理风险。
2) 将动态认知‑情感模型嵌入模拟患者，实时追踪内部心理状态，提供可解释的危机与风险轨迹。
3) 统一的质量护理与风险本体，支持多维度自动评估（治疗联盟、干预忠诚、危机应对、预后等）。
4) 构建交互式仪表盘，帮助多方利益相关者（工程师、临床医生、政策制定者）可视化并审计“黑盒”AI治疗效果。

**🔧 技术方法**

技术手段：
- 大型语言模型（ChatGPT、Gemini、Character.AI 等）作为被测治疗师。
- 统一的仿真编排器（Python）管理多轮对话、状态持久化与 API 调用。
- 认知‑情感管道（基于认知评价理论、BDI、情绪调节等）嵌入模拟患者的 LLM 实例。
- LLM‑as‑Judge 自动评估器，用于测量危机检测、危机响应、治疗忠诚、治疗联盟、SURE 结果等。
- 交互式 Web 仪表盘（基于 Dash/Plotly）实现可视化与细粒度审计。

**📊 数据集**

数据集与实验设计：
- 15 组 AUD 病例人设，基于五类经验分型（青年、功能型、家庭型、反社会型、慢性严重型）与行为改变阶段（预思考、思考、行动）构建。
- 每位人设通过多次独立仿真获得 30 个患者-治疗师对，合计 180 组对话。
- 每对话 4 周会话（共 4 次）＋会间事件，累计 369 轮会话。
- 采用标准问卷（SURE、WAI、SRS）和 MITI 评分系统的 LLM 自动化实现。

**📈 对比分析**

比较方法与结果：
- 对 6 种 AI（ChatGPT Basic/MI、Gemini MI、Character.AI、Harmful AI 控制、NIAAA 教育手册）与 15 人设交叉，形成 180 组。
- 通过单变量 ANOVA / GLM 检验每种 AI 在质量护理（联盟、忠诚、复杂反射、R:Q 比）和风险（危机事件、协议遵循、总不良结果）上的显著差异。
- 结果显示：公开 AI 在危机处理和“AI Psychosis”（误验证患者妄想）上表现最差；ChatGPT MI 与 Gemini MI 在治疗忠诚度上优于基本模型；NIAAA 手册在联盟与风险评估上最低。
- 通过仪表盘可视化各模型在多维度上的分布，明确指出安全缺口与改进方向。

**⚠️ 局限性**

局限性：
- 模拟患者虽基于实证构建，但仍为人工智能代理，可能无法完全再现真实患者的情感细腻度与突发行为。
- LLM‑as‑Judge 评估器受模型偏见与上下文敏感性的限制，可能对某些细粒度指标产生误判。
- 仅在酒精使用障碍情境下验证，跨领域通用性尚未测试。
- 试验仅进行 4 次会话，未覆盖更长期的治疗过程与潜在慢性风险。
- 利益相关者评估样本规模有限（9 人），仪表盘验证的外部可行性仍需进一步研究。

---

## 450. Personalized Prediction of Perceived Message Effectiveness Using Large Language Model Based Digital Twins

**arXiv ID:** 2602.19403 | [PDF](https://arxiv.org/pdf/2602.19403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 451. PositionOCR: Augmenting Positional Awareness in Multi-Modal Models via Hybrid Specialist Integration

**arXiv ID:** 2602.19188 | [PDF](https://arxiv.org/pdf/2602.19188v1)

**作者:** Chen Duan `[一作]`, Pengfei Yan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PositionOCR，一种将文本识别专家模型与大型语言模型结合的混合框架，专注于位置精准的OCR任务；

**💡 创新点**

创新点在于通过指令微调专家模型实现位置推理而无需大规模LLM训练，显著降低参数规模至1.31亿；

**🔧 技术方法**

采用了ResNet50图像编码器、Qwen2.5-7B LLM、自动回归Transformer解码器，并利用序列化坐标+文本的输出格式；

**📊 数据集**

使用了多来源数据集：文档（IIT‑CDIP、DocVQA等）、场景文本（Total‑Text、ICDAR2015等）、对象检测（COCO、Objects365）、图表与表格（ChartQA、WikiTableQuestions）以及文本定位数据（DocStruct4M-subset）；

**📈 对比分析**

与传统MLLM和专用文本识别模型对比，PositionOCR在文本定位（IOU@0.5 83.0%）、文本识别（ICDAR2015 77.4%）和文档VQA（DocVQA 69.8%）等任务均达到或超过现有方法，且仅需131M参数；

**⚠️ 局限性**

受限于预训练数据量（仅2.1M图像）和任务多样性，难以匹敌像DocOwl‑1.5或Qwen2.5‑VL的规模，需进一步扩大数据集以提升泛化能力。

---

## 452. Why iCloud Fails: The Category Mistake of Cloud Synchronization

**arXiv ID:** 2602.19433 | [PDF](https://arxiv.org/pdf/2602.19433v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAEDAELUS), Paul Borrill (DAEDAELUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文批判性分析iCloud Drive的同步协议及其与时间机器、Git、自动化工具链等POSIX文件系统依赖程序的兼容性缺陷，并提出Open Atomic Ethernet（OAE）作为理论解决方案。

**💡 创新点**

创新点在于将iCloud的同步问题归结为“类别错误”——将分布式因果图投影到线性时间链导致的失真，并引入双向可逆的OAE协议以消除FITO假设。

**🔧 技术方法**

使用的技术包括分布式系统理论（因果图、FITO、最后写者赢策略）、POSIX文件系统语义分析、日志审计、以及OAE的双向事务协议设计。

**📊 数据集**

主要数据集为作者自行收集的366 GB iCloud Drive分叉存档（含110个顶级文件夹）以及相关操作日志与错误记录。

**📈 对比分析**

对比方法主要是案例分析与实测错误日志，结果显示iCloud在与Time Machine、Git等交互时出现频繁冲突、数据丢失和“Operation not permitted”等错误；OAE理论模型未在实验中验证，但预计可在避免时间戳冲突和中间状态泄漏方面实现更高一致性。

**⚠️ 局限性**

局限性包括：未在真实系统中实现OAE同步；仅从理论和案例层面讨论，缺乏量化性能评估；对不同云服务（Google Drive、Dropbox）缺乏广泛验证。

---

## 453. Gait Asymmetry from Unilateral Weakness and Improvement With Ankle Assistance: a Reinforcement Learning based Simulation Study

**arXiv ID:** 2602.18862 | [PDF](https://arxiv.org/pdf/2602.18862v1)

**作者:** Yifei Yuan `[一作]` (New Jersey Institute of Technology), Xianlian Zhou `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 1169 | [OpenAlex ID](https://openalex.org/A5019820614)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究利用强化学习驱动的肌肉骨骼仿真框架，系统探讨单侧肌肉无力如何导致步态不对称，并评估踝关节外骨骼辅助在不同无力程度下对步态对称性的改善效果。

**💡 创新点**

创新点在于：①将多代理强化学习与完整3D肌肉骨骼模型耦合，能在仿真中自动学习针对无力状态的踝关节辅助策略；②未使用显式对称奖励，而是观察学习策略自发改善对称性的能力；③提供了从无力水平逐步递减到25%时步态特征的系统量化。

**🔧 技术方法**

核心技术包括MyoAssist强化学习框架、MuJoCo物理引擎、Proximal Policy Optimization（PPO）多代理学习、基于关节角度与肌肉激活的模仿奖励。

**📊 数据集**

使用的是内部构建的3D人体模型（26条下肢肌肉）与仿真生成的步态数据；未涉及公开人类步态数据库。

**📈 对比分析**

通过比较不同无力比例（100%、75%、50%、25%）下的站立时间不对称、关节ROM对称指数（SI）、关节角度相关系数及有无外骨骼辅助的指标，发现50%无力时外骨骼能将踝关节SI从-25.8%改善至-18.5%，相关系数从0.948提升至0.966，但载荷对称性仍未恢复。

**⚠️ 局限性**

主要局限包括：仅在仿真环境验证，缺乏真实人类实验；无力模型统一降低右腿所有肌肉，未考虑肌肉特异性或个体差异；仅提供踝关节辅助，未探讨多关节或全身控制；实验仅在水平地面、单一步行速度下进行，未涵盖坡度、速度变化或扰动条件。

---

## 454. SGNO: Spectral Generator Neural Operators for Stable Long Horizon PDE Rollouts

**arXiv ID:** 2602.18801 | [PDF](https://arxiv.org/pdf/2602.18801v1)

**作者:** Jiayi Li `[一作]` (University of New South Wales), Flora D. Salim `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种Spectral Generator Neural Operator（SGNO），用于在短训练长测试场景下进行稳定的长时序PDE rollouts。

**💡 创新点**

创新点在于：①用学习到且非正实部的对角谱生成器来保证线性ETD步进器的非扩张性；②通过ϕ1权重的非线性注入和门控通道混合控制残差；③仅在非线性注入通道使用频谱截断与遮罩，以抑制高频回馈，从而实现长期稳定性。

**🔧 技术方法**

技术手段包括：频谱域指数时间微分（ETD）步进、对角生成器约束、ϕ1加权非线性注入、门控与通道混合、频谱截断/遮罩、点对点校正以及针对一步放大与有限时域误差的理论稳定性分析。

**📊 数据集**

实验使用了APEBench七个任务的平面和三维PDE数据集：1D Dispersion、1D KdV、1D Kuramoto–Sivashinsky、2D Anisotropic Diffusion、2D Kolmogorov Flow、3D Unbalanced Advection 与 3D Swift–Hohenberg。

**📈 对比分析**

在与FNO、UNet、ResNet、Dilated网络等基线在相同训练/评估协议下对比，采用nRMSE和GMean100指标；SGNO在六项任务上实现最佳或第二最佳性能，并显著延长了稳定推理步数。

**⚠️ 局限性**

局限性包括：仅在固定分辨率与时间步长下验证；稳定性分析局部且对非线性部分缺乏严格上界；采用一阶ETD，未探索更高阶或更复杂的生成器/遮罩设计。

---

## 455. StreetTree: A Large-Scale Global Benchmark for Fine-Grained Tree Species Classification

**arXiv ID:** 2602.19123 | [PDF](https://arxiv.org/pdf/2602.19123v1)

**作者:** Jiapeng Li `[一作]` (Peking University), Yu liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了全球首个大规模精细街道树种数据集StreetTree，并提供了评估集StreetTree-18kEval，供城市绿化与生态研究使用。

**💡 创新点**

创新点在于：①全球133国覆盖、超过12 M条街景图像；②四级层级（目–科–属–种）完整分类；③包含季节与多年连续记录；④针对城市化区域进行筛选与地理精度高。

**🔧 技术方法**

技术方法包括：基于GBIF+TreeGOER的全局树种目录构建；利用GHSL进行城市化区域筛选；通过Google Street View API采集街景；使用Segformer+ResNet18+MLP做图像有效性过滤；模型训练采用ViT与CLIP（预训练与微调）并进行数据量效能实验。

**📊 数据集**

主要数据集为StreetTree（12 M样本、3.36 M树木、133国），并结合TreeML、U.S. 5Million、UrbanForest等源构建。

**📈 对比分析**

对StreetTree-18kEval进行基准测试，比较ViT、CLIP及Fine‑tuned CLIP，发现CLIP在极少数据（0.1%）下表现优越，完整数据下Fine‑tuned ViT略胜CLIP；Top‑1准确率仍仅约32%（物种级），Top‑10可达63%，表明任务极其细粒度。

**⚠️ 局限性**

主要局限包括：物种视觉相似导致Top‑1低，长尾分布与季节变化带来高不确定性；数据在部分地区稀疏，部分分类标签可能不精确；缺乏多视角与多模态信息，限制模型泛化。

---

## 456. DUET-VLM: Dual stage Unified Efficient Token reduction for VLM Training and Inference

**arXiv ID:** 2602.18846 | [PDF](https://arxiv.org/pdf/2602.18846v1)

**作者:** Aditya Kumar Singh `[一作]` (Advanced Micro Devices Inc), Emad Barsoum `[通讯]` (Advanced Micro Devices Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种双阶段视觉压缩框架（DUET-VLM），在视觉编码器和语言解码器两侧同时进行冗余感知合并与文本引导的视觉令牌裁剪。

**💡 创新点**

创新点在于将视觉侧的局部聚类合并与语言侧的层级文本引导裁剪结合成可微分、端到端可训练的双向压缩体系，既保持低层细粒度信息，又在高层逐步剔除无关视觉信息，显著提升压缩率与准确率的平衡。

**🔧 技术方法**

核心技术包括：V2V自注意力导向的主导与上下文令牌选择；局部聚类聚合（保留细节、避免信息稀释）；T2V文本引导的层级裁剪（使用显著文本令牌计算交叉注意力得分）；可微分的压缩接口与联合训练；以及基于聚类宽度与保留比例的参数化控制。

**📊 数据集**

使用多种图像/视频任务数据集：POPE、GQA、TextVQA、MME、SQA-Image、SeedBench-Image、TGIF-QA、MSVD-QA、MSRVTT-QA；在 LLaVA‑1.5‑7B、LLaVA‑NeXT‑7B、Video‑LLaVA‑7B、Qwen‑2.5‑VL‑7B 等模型上进行评估。

**📈 对比分析**

与 VisionZip、PyramidDrop、FastV、SparseVLM 等现有压缩方法对比，DUET‑VLM 在 67%‑89% 令牌压缩率下保持 99%‑95% 的基线准确率，甚至在 64 令牌下仍达 95.4%；训练时缩短 26%‑36% 训练时间；在视频任务中在 53.1% 令牌压缩率下平均准确率 100.8%，在 93.4% 压缩率下 97.6%。

**⚠️ 局限性**

局限性包括：仍需进一步优化推理与训练速度的底层实现；对更长时序视频的推理效果未完全验证；聚类与裁剪参数需手动调优；未探索跨模态（音频、文本）扩展。

---

## 457. The Doctor Will (Still) See You Now: On the Structural Limits of Agentic AI in Healthcare

**arXiv ID:** 2602.18460 | [PDF](https://arxiv.org/pdf/2602.18460v1)

**作者:** Gabriela Aránguiz Dias `[一作]` (Stanford University), Mykel J. Kochenderfer `[通讯]` (Stanford University)

**通讯引用:** 12491 | [OpenAlex ID](https://openalex.org/A5068326377)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过对20名技术开发者、临床实施者和终端用户进行半结构化访谈，研究了医疗领域中“agentic AI”的定义、实际自主性与评估缺口，并探讨了相关的责任与治理问题。

**💡 创新点**

首次将医疗中agentic AI的概念碎片化、实际自主性有限、评估指标缺失这三条互相强化的张力系统化，揭示了定义不统一导致的责任分散和评估缺口造成的安全隐患，并提出以部署情境为中心的评估与治理框架。

**🔧 技术方法**

采用定性研究方法：扎根理论、开放/轴向编码、访谈记录转录与Taguette编码，未涉及机器学习或算法实现。

**📊 数据集**

访谈转录文本共约577页（约173,009词），未使用公开数据集。

**📈 对比分析**

本文不包含算法实现或性能对比，研究重点是基于访谈结果的定性分析，未给出可量化的性能指标。

**⚠️ 局限性**

样本仅为美国20名具有技术背景的研究人员和临床人员，缺少基层医院、患者、监管者等视角；访谈时间与技术进展可能导致结果过时；自我报告与转录可能带来偏见。

---

## 458. Next Reply Prediction X Dataset: Linguistic Discrepancies in Naively Generated Content

**arXiv ID:** 2602.19177 | [PDF](https://arxiv.org/pdf/2602.19177v1)

**作者:** Simon Münker `[一作]`, Achim Rettinger `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）在社交媒体内容生成中的真实性与可检测性，通过比较prompt生成与fine‑tuned模型生成的文本与真实推文在量化、形态句法和语义层面的差异。

**💡 创新点**

创新点在于构建了基于历史上下文的回复预测数据集，提出了多维度评估框架，并系统比较不同特征编码与分类器在检测合成内容上的表现。

**🔧 技术方法**

采用了NeLa量化特征、spaCy形态句法分析、TweetEval语义分类、Qwen3嵌入、TF‑IDF、FastText以及XGBoost分类器等技术。

**📊 数据集**

使用公开的德语和英语X（Twitter）推文及回复数据，最终构成1,000个样本的公开数据集。

**📈 对比分析**

通过相似度度量和宏F1分类实验比较，发现prompt生成最易被检测，fine‑tuned模型接近真实；最优特征组合（TF‑IDF+FastText+多维特征）在德语上宏F1≈0.73、英语≈0.70。

**⚠️ 局限性**

局限性包括仅使用Qwen3 8B模型、仅考察单轮3句历史上下文、数据仅来自2023年上半年X平台、仅涉及德英两种语言、缺乏多轮对话与更丰富的隐性语义特征评估。

---

## 459. Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction

**arXiv ID:** 2602.18996 | [PDF](https://arxiv.org/pdf/2602.18996v1)

**作者:** Shannan Yan `[一作]` (Tsinghua University), Fengyun Rao `[通讯]` (WeChat Vision, Tencent Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于条件二值分割的轻量级框架，通过将源视图的目标对象掩码编码为条件token，在目标视图中进行对象级视觉对应；

**💡 创新点**

创新点在于引入循环一致性自监督损失与推理时训练（TTT）相结合，使模型无需额外标注即可学习视角不变的表示，并在推理阶段进一步提升性能；

**🔧 技术方法**

核心技术包括DINOv3预训练视觉变压器骨干、条件token注入、双头解码（掩码与可见性），以及BCE+Dice、循环一致性以及TTT的联合优化；

**📊 数据集**

实验使用了Ego-Exo4D（1.8M帧对象掩码）和HANDAL-X（44k训练/14k测试配对）两个跨视角数据集；

**📈 对比分析**

在Ego-Exo4D上mIoU提升至44.57%（比前沿O‑MaMa高2.9%），在HANDAL-X上IoU达到78.8%（相对O‑MaMa提升约84%），验证了方法在跨视角对应任务中的领先表现；

**⚠️ 局限性**

限制主要包括对极小物体的分割仍不理想、Ego→Exo任务普遍性能低于Exo→Ego、受限于数据量与隐私限制的训练样本，且对不可见对象的处理尚未完善。

---

## 460. Contradiction to Consensus: Dual Perspective, Multi Source Retrieval Based Claim Verification with Source Level Disagreement using LLM

**arXiv ID:** 2602.18693 | [PDF](https://arxiv.org/pdf/2602.18693v1)

**作者:** Md Badsha Biswas `[一作]` (George Mason University), Ozlem Uzuner `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种开放域主张验证系统，结合原始主张及其否定句检索多源证据并进行聚合。

**💡 创新点**

创新点在于双视角检索（原句+否定句）以及跨来源（Wikipedia、PubMed、Google）证据聚合，同时通过LLM的 log-prob 分数可视化源间不一致。

**🔧 技术方法**

使用的大模型为 Llama 3.3 70B/405B、Mistral、Qwen 2.5、Phi-4；检索工具包括 Elasticsearch、dense BM25、Google Custom Search；证据筛选与去重采用 SPICED 嵌入。

**📊 数据集**

评估数据集包括 SciFact、PubHealth、Averitec 与 LIAR 四个公开基准。

**📈 对比分析**

对比方法为单源检索与聚合检索、原始句检索与加否定句检索，未使用微调；实验显示加否定句提升 2–10% 准确率，聚合检索在大多数模型和数据集上提升 10–70% 宏 F1，展示稳健性能提升。

**⚠️ 局限性**

局限包括上下文窗口限制导致证据截断、缺乏时间敏感检索、数据集噪声与标签不平衡以及 LLM 对偏见与分布漂移的敏感性。

---

## 461. SenTSR-Bench: Thinking with Injected Knowledge for Time-Series Reasoning

**arXiv ID:** 2602.19455 | [PDF](https://arxiv.org/pdf/2602.19455v1)

**作者:** Zelin He `[一作]` (Pennsylvania State University), Matthew Reimherr `[通讯]` (Amazon RME)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种混合知识注入框架，将时间序列专家模型（TSLM）的领域知识注入到通用推理大型语言模型（GRLM）中，实现时序诊断推理；同时推出了真实工业环境下的多阶段诊断基准 SenTSR‑Bench。

**💡 创新点**

创新点：1) 通过可插拔的知识注入函数把 TSLM 生成的分析片段直接嵌入 GRLM 的推理轨迹；2) 用无监督的强化学习与可验证奖励（RLVR）训练 TSLM 生成“思考先行”的推理轨迹，解决传统注入任务中的任务偏移问题；3) 构建大规模真实工业多变量时序数据与人工注释的诊断基准。

**🔧 技术方法**

技术：时序文本编码（视觉或结构化 JSON），生成式推理模型的“思考‑响应”解耦，知识注入函数（Early/Intermediate/Late），强化学习（GRPO）与可验证奖励，基准评估指标（准确率、RAGAS）。

**📊 数据集**

数据集：SenTSR‑Bench（真实工业多变量时序+多阶段诊断文本），TSEvol 与 TS&Language Benchmark（公开时序推理数据）。

**📈 对比分析**

比较方法：对比单独 TSLM（SFT/RL 训练）、单独 GRLM（零射/少射）、以及注入版本；注入模型在 SenTSR‑Bench 上比 TSLM 提升 15.5%–26.1%，比 GRLM 提升 7.3%–22.4%；在 TSEvol/TS&Language 上同样取得 5.2%–10.4% 的提升。RL‑注入相较于 SFT‑注入提升更大（1.66×~2.92×）。

**⚠️ 局限性**

局限性：1) 需要先训练 TSLM 并生成可验证奖励的环境；2) 注入过程对 GRLM 的推理结构有一定依赖，可能受限于模型支持的推理模板；3) 仍需大量人工标注的时序与诊断文本来提升 TSLM 的泛化；4) 对极长时间序列的处理仍受 token 限制。

---

## 462. GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning

**arXiv ID:** 2602.19206 | [PDF](https://arxiv.org/pdf/2602.19206v1)

**作者:** Zehao Deng `[一作]` (Soochow University), Yan Wang `[通讯]` (Tsinghua University)

**通讯引用:** 26848 | [OpenAlex ID](https://openalex.org/A5100322712)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种两阶段零样本3D异常检测框架GS-CLIP，能在无目标训练数据下从3D点云中检测异常。

**💡 创新点**

创新点包括：①几何感知提示生成，利用全局形状与局部缺陷信息构建文本提示；②协同视图表征学习，将渲染图和深度图并行编码并通过Synergistic Refinement Module融合；③在深度分支上引入LoRA以弥补域差距。

**🔧 技术方法**

采用CLIP（ViT-L/14）视觉/文本编码器、PointNet++提取3D特征、Geometric Defect Distillation Module、LoRA、SRM、注意力聚合、交叉视角一致性损失等技术。

**📊 数据集**

在四个公开数据集上评估：MVTec3D-AD、Real3D-AD、Eyecandies、Anomaly-ShapeNet。

**📈 对比分析**

与PointAD、MVP-PCLIP、AA-CLIP等SOTA方法对比，GS-CLIP在对象级AUROC/AP及点级AUROC/PRO均取得最高分，平均提升约2-3%。

**⚠️ 局限性**

局限性在于：1）依赖2D投影，仍可能丢失细节；2）计算与显存开销略高；3）对极端光照或复杂纹理的鲁棒性尚待进一步验证。

---

## 463. TRUE: A Trustworthy Unified Explanation Framework for Large Language Model Reasoning

**arXiv ID:** 2602.18905 | [PDF](https://arxiv.org/pdf/2602.18905v1)

**作者:** Yujiao Yang `[一作]` (Dalian University of Technology), Yujiao Yang `[通讯]` (Dalian University of Technology)

**通讯引用:** 840 | [OpenAlex ID](https://openalex.org/A5112751533)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出统一可信解释框架TRUE，结合可执行推理验证、结构保持扰动生成可行区DAG以及因果失败模式分析，对LLM推理过程实现多层次可解释与诊断。

**💡 创新点**

①将解释定义为可执行的过程规范，使用盲执行验证评估其完整性；②通过结构保持扰动构建可行区有向无环图，刻画局部推理稳定性；③采用Shapley值因果分解自动发现并量化类级结构性失败模式。

**🔧 技术方法**

可执行推理验证、结构保持扰动、可行区DAG、白盒算术/规则验证、Shapley值因果分析、LLM引导的聚类与摘要等技术。

**📊 数据集**

数学推理数据集GSM8K、MATH；逻辑/知识推理数据集BBH（多选）和MMLU‑CF。

**📈 对比分析**

与多种提示策略（CoT、Zero‑Shot CoT、Self‑Refine、Plan‑and‑Solve）和不同基础模型（GPT‑4o‑mini、GPT‑4.1‑nano、GPT‑5‑nano）对比，评估可执行准确率、执行一致性、成功率预测的交叉熵、覆盖率等指标，实验表明可执行准确率接近原始准确率，DAG显著降低预测交叉熵并提升覆盖率，失败模式分析揭示关键结构性缺陷。

**⚠️ 局限性**

对知识/逻辑任务可执行率相对较低；盲执行依赖外部白盒工具；可行区DAG需足够扰动且仅覆盖局部空间；失败模式发现需要足够样本并受聚类误差影响；模型能力限制导致解释完整性受限；未评估实时推理开销。

---

## 464. ArabicNumBench: Evaluating Arabic Number Reading in Large Language Models

**arXiv ID:** 2602.18776 | [PDF](https://arxiv.org/pdf/2602.18776v1)

**作者:** Anas Alhumud `[一作]` (Saudi Data and Artificial Intelligence Authority), Muhammad Badruddin Khan `[通讯]` (Imam Mohammad Ibn Saud Islamic University)

**通讯引用:** 1237 | [OpenAlex ID](https://openalex.org/A5077568501)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ArabicNumBench基准，用71个大模型在四种提示策略下评估东阿拉伯-Indic和西式阿拉伯数字的读取能力。

**💡 创新点**

创新点在于引入提取方法跟踪衡量结构化输出，并揭示高准确率与格式化遵循的分离。

**🔧 技术方法**

采用Chain-of-Thought提示、零样本/少样本学习以及层级化答案提取技术。

**📊 数据集**

使用自制的Arabic Number Reading Benchmark v3，共210个测试案例，涵盖纯数字、地址、日期、数量和价格六大语境。

**📈 对比分析**

通过281种模型-策略组合比较，发现少样本CoT平均准确率达80%，结构化输出率提升至34%，顶尖模型可实现95-100%结构化输出。

**⚠️ 局限性**

局限性包括仅有210个案例、仅覆盖日常场景且提示策略有限，难以全面评估跨领域性能。

---

## 465. Scaling Ultrasound Volumetric Reconstruction via Mobile Augmented Reality

**arXiv ID:** 2602.18500 | [PDF](https://arxiv.org/pdf/2602.18500v1)

**作者:** Kian Wei Ng `[一作]` (National University Health System), Eng Tat Khoo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计并验证基于移动设备的AR辅助3D超声系统MARVUS，用于提高肿瘤体积测量的准确性和一致性。

**💡 创新点**

创新点在于：①使用低成本移动设备和单件3D打印校准仿真，实现无外部跟踪硬件的3D超声重建；②引入AR可视化实时反馈，提升操作员信心和测量精度；③采用基础模型和EdgeTAM实现跨专科、低资源部署。

**🔧 技术方法**

技术包括：ArUco标记与单摄像头外参校准；基于单帧LEDge结构的内参标定；自由手扫面 + 半自动分割 (EdgeTAM) + voxelization + marching cubes；AR实时叠加与mesh‑US交叉验证。

**📊 数据集**

使用MAMA‑MIA乳腺结节样本构造的临床级塑料模型；用户研究采用8名经验丰富的超声医生与12个尺寸相同（1.69 cm³）的结节模型。

**📈 对比分析**

通过对比手工椭圆公式、自由扫+重建、自由扫+AR三种测量方式，发现重建方案将体积误差从0.63 cm³降至0.27 cm³，加入AR后进一步降至0.16 cm³，同时操作员变异性显著降低。NASA‑TLX与SUS问卷显示AR提高信心但整体工作量无显著差异。

**⚠️ 局限性**

局限性包括仅在人工制模型与单一超声探头上验证；缺乏真实病人多样性与不同超声设备的广泛测试；AR功能在高强度现场使用中的可靠性未评估；系统对快速手持运动的鲁棒性有限。

---

## 466. Seeing Farther and Smarter: Value-Guided Multi-Path Reflection for VLM Policy Optimization

**arXiv ID:** 2602.19372 | [PDF](https://arxiv.org/pdf/2602.19372v1)

**作者:** Yanting Yang `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种价值引导的多路径反思规划框架，用以解决长周期机器人操作任务。

**💡 创新点**

创新点在于显式价值评估（目标距离衰减）与多路径聚合解码策略，以及置信度触发的早停机制。

**🔧 技术方法**

采用Vision‑Language Model（LLaVA-1.5）、扩散动力学模型、价值批评器、触发器及Beam Search等技术。

**📊 数据集**

使用ReflectVLM所用的1000个任务模拟数据和100个未见任务进行评估。

**📈 对比分析**

在100个未见长周期任务上与MCTS、BC、Zero‑Shot VLM及ReflectVLM比较，单轮后训练即可达到81–83%成功率，显著高于基线并提升约45%推理速度。

**⚠️ 局限性**

局限在于需要大量交互式训练数据、仿真到真实的差异，以及缺乏低层控制集成。

---

## 467. LEVDA: Latent Ensemble Variational Data Assimilation via Differentiable Dynamics

**arXiv ID:** 2602.19406 | [PDF](https://arxiv.org/pdf/2602.19406v1)

**作者:** Phillip Si `[一作]`, Peng Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于预训练可微分潜在动力学网络的集合变分数据同化方法LEVDA，能够在低维潜在空间内进行四维集合变分优化，联合同化状态与未知参数，并支持任意时空观测；

**💡 创新点**

创新点在于：①在潜在空间直接进行4DEnVar优化，避免高维梯度或旁线/伴随模型；②不需要额外观测到潜在编码器，利用解码器直接把潜在轨迹映射到观测空间；③通过自动微分获得精确梯度，允许在不规则时空观测下进行多时间窗口一致的动态约束；

**🔧 技术方法**

使用技术包括：可微分潜在动力学网络（LDNet），坐标基神经场解码器，自动微分，集合子空间优化（L-BFGS），以及多成员并行优化以生成分析集；

**📊 数据集**

实验数据集涵盖三大基准：Kolmogorov流动（未知粘度）、海啸传播（未知源）和全球大气动力学（未知强迫），各自具备高空间分辨率与极端观测稀疏性；

**📈 对比分析**

与基线比较包括传统高维滤波（LETKF、EnSF）、潜在滤波（Latent-EnSF、LD-EnSF）以及全状态4DEnVar；LEVDA在观测稀疏、非规则采样环境下的相对RMSE往往低于或与最佳基线相当，并且在计算时间上比全状态4DEnVar快数十倍；

**⚠️ 局限性**

局限性：依赖预训练潜在模型的精度，若模型误差或分布偏移显著会导致同化失效；优化目标非凸，可能需要手动调参（集合规模、膨胀系数）；在极端大气动力学场景下的误差校准仍有待提升。

---

## 468. Robust Self-Supervised Cross-Modal Super-Resolution against Real-World Misaligned Observations

**arXiv ID:** 2602.18822 | [PDF](https://arxiv.org/pdf/2602.18822v1)

**作者:** Xiaoyu Dong `[一作]` (University of Tokyo), Naoto Yokoya `[通讯]` (University of Tokyo)

**通讯引用:** 14636 | [OpenAlex ID](https://openalex.org/A5034435383)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了RobSelf模型，解决真实场景下跨模态超分辨率存在的空间失配问题，模型通过在线自监督训练，无需预训练数据、监督标签或预先对齐即可实现高质量的深度/近红外超分。

**💡 创新点**

创新点主要包括：①将跨模态与跨分辨率对齐问题重新表述为弱监督的失配感知特征翻译子任务；②设计内容感知参考滤波器，利用对齐后引导的参考信息自适应区分重要与冗余内容，实现高保真自我提升；③一次性在线自监督优化，避免传统两阶段预对齐的泛化瓶颈。

**🔧 技术方法**

采用多尺度变形场估计器 + 可变形卷积或简单重采样实现失配感知特征翻译；构建重要性图并根据阈值选择大/小卷积核，实现内容感知参考滤波；使用一致性回归损失（对源的LR）进行自监督；实现基于PyTorch、A100 GPU的在线优化框架。

**📊 数据集**

主要使用了自收集的RealMisSR数据集（RGB‑Depth 与 RGB‑NIR 两个子集，包含真实跨传感器失配、视角变化与局部运动），以及中性合成数据（Middlebury 2006、NYU‑v2）用于对比实验。

**📈 对比分析**

与监督方法（DORNet、C2PD、SGNet、DCTNet）以及自监督方法（SSGNet、MMSR、CMSR、P2P）在合成×4/×8、真实×2/×4 RGB‑Depth、RGB‑NIR 等任务中进行比较。RobSelf在RMSE/NIQE上均取得最优或第二优表现，并且在速度上比P2P/SSGNet快 2.5–15.3 倍，展示了优异的性能与效率。

**⚠️ 局限性**

局限性：在极端大幅失配或高频细节场景下仍可能产生伪纹理；对不同模态的细粒度对齐缺乏专门机制；由于缺乏真实×4 GT，评价主要依赖无参考指标，可能不足以完全反映视觉质量；部分参数需要针对不同任务手动调节。

---

## 469. Federated Reasoning Distillation Framework with Model Learnability-Aware Data Allocation

**arXiv ID:** 2602.18749 | [PDF](https://arxiv.org/pdf/2602.18749v1)

**作者:** Wei Guo `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**通讯引用:** 10102 | [OpenAlex ID](https://openalex.org/A5102969899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LaDa框架，实现联邦学习中大语言模型与小语言模型的双向知识蒸馏与学习可行性感知数据分配。

**💡 创新点**

创新点在于识别双向学习可行性差距和领域无关推理迁移问题，并提出基于学习可行性感知的数据过滤器和对齐推理路径的对比蒸馏方法。

**🔧 技术方法**

采用联邦学习、知识蒸馏、对比学习、探索-利用（UCB/Thompson Sampling）策略、模型可行性评估以及推理路径对齐技术。

**📊 数据集**

使用MathInstruct和CoT-Collection（包含MATH、GSM8K、TheoremQA等子数据集）进行实验。

**📈 对比分析**

与FedKD、FedMKT、WASP等基线对比，在四个多场景、多规模、多架构实验中实现了0.2%~13.8%的准确率提升，尤其在零样本/单示例推理中表现突出。

**⚠️ 局限性**

局限性包括依赖公开/合成蒸馏数据集、对超参数调优敏感、通信成本随模型规模和异构程度上升，以及在极端模型差距或极大模型规模时效果可能受限。

---

## 470. A high-resolution nationwide urban village mapping product for 342 Chinese cities based on foundation models

**arXiv ID:** 2602.18765 | [PDF](https://arxiv.org/pdf/2602.18765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 471. MIRROR: Multimodal Iterative Reasoning via Reflection on Visual Regions

**arXiv ID:** 2602.18746 | [PDF](https://arxiv.org/pdf/2602.18746v1)

**作者:** Haoyu Zhang `[一作]` (Beijing Institute of Technology), Yunde Jia `[通讯]` (Shenzhen MSU-BIT University)

**通讯引用:** 7277 | [OpenAlex ID](https://openalex.org/A5100731042)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MIRROR 框架，实现多轮视觉反思闭环推理，显著降低视觉语言模型的幻觉并提升推理准确性。

**💡 创新点**

创新点在于把反思从纯文本修正转为主动视觉验证：模型可自发触发视觉提示生成器，检查图像特定区域，从而把推理过程与视觉证据紧密耦合。

**🔧 技术方法**

技术实现包括：在 Qwen2.5‑VL 基础上加入 Molmo‑7B + SAM‑2 视觉提示生成器；采用多代理多轮对话构造 ReflectV 数据集；使用自回归多轮训练目标，强化闭环反思与视觉验证。

**📊 数据集**

使用数据集：自建 ReflectV（≈24k 样本）以及多项公开基准（MM‑Vet、MMStar、SeedBench‑2‑Plus、TextVQA、OCRBench、ChartQA、POPE、HallusionBench、HRBench、MME‑RealWorld‑Lite、VStarBench、MathVision 等。

**📈 对比分析**

通过 VLMEvalKit 与 GPT‑4o‑mini 评估，在通用能力、OCR 与文档、幻觉、细粒度感知与数学推理等任务上与 Qwen2.5‑VL、InternVL3、LLaVA‑OneVision 等 SOTA 对比，MIRROR 在多项指标上均提升明显（如 HallusionBench +13.36%，OCRBench +3.90%，TextVQA +6.56% 等），验证闭环视觉反思显著提升模型鲁棒性。

**⚠️ 局限性**

局限性在于对抽象概念或非空间属性的推理仍难以实现精确空间锚定，粗粒度属性绑定与抽象知识验证效果有限。

---

## 472. Adaptive Time Series Reasoning via Segment Selection

**arXiv ID:** 2602.18645 | [PDF](https://arxiv.org/pdf/2602.18645v1)

**作者:** Shvat Messica `[一作]` (Harvard Medical School), Marinka Zitnik `[通讯]` (Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于控制器-推理器协作的时间序列推理框架，模型在推理过程中动态选择并读取最具信息量的时间段，实现了按需获取证据而非一次性编码整个序列。

**💡 创新点**

核心创新在于将时间段选择与多步推理拆分为两个角色，并通过协作式自我对弈（self‑play）和层次化策略优化，使用可靠性奖励引导控制器学习最具鲁棒性的段落集合；同时利用多段交互的优势信号，克服单步奖励的信用分配难题。

**🔧 技术方法**

技术手段包括：控制器-推理器角色化的单一LLM；工具调用机制实现段落检索；强化学习（PPO式）自我对弈；层次化策略优化与方差引导采样；可靠性奖励评估；在Qwen‑3 4B基础上实现。

**📊 数据集**

在六大多领域基准上评估：TSQA、TRQA、Sleep‑QA、RCW、ECG‑QA、ETI。

**📈 对比分析**

与多类基线（文本LLM、时序编码‑LLM、视觉‑语言模型）对比，平均准确率提升6.46个百分点；RL相较SFT进一步提升约5.65个百分点；在大多数数据集上均超越最强基线，且在少量段落（30–70%）下即可达到最佳表现，显示出更高的推理效率。

**⚠️ 局限性**

局限性：推理时需要多轮段落检索，导致推理延迟增加；实验仅限单变量时间序列，扩展至多变量或不规则采样尚待验证；控制器与推理器共享参数，可能导致训练不稳定。

---

## 473. Social Media Feed Elicitation

**arXiv ID:** 2602.18594 | [PDF](https://arxiv.org/pdf/2602.18594v1)

**作者:** Lindsay Popowski `[一作]` (Stanford University), Michael S. Bernstein `[通讯]` (Stanford University)

**通讯引用:** 64116 | [OpenAlex ID](https://openalex.org/A5076189854)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于LLM的访谈式馈送提炼方法，帮助用户明确并实现个性化社交媒体内容订阅。

**💡 创新点**

创新点在于通过交互式访谈主动触发用户思考和细化偏好，弥补传统手工描述的“表达缺口”，并将访谈结果直接转化为可执行的馈送规范。

**🔧 技术方法**

核心技术是大语言模型（GPT‑5）用于访谈引导、关键词与属性生成，以及文本分类与排序；系统将自然语言规格转为二阶段的相关性与质量评分。

**📊 数据集**

使用了BlueSky Firehose公开的约2万条帖子数据，并对其进行预过滤（NSFW、短文本等）以形成评测数据库。

**📈 对比分析**

通过对400名受试者的双向对照实验，访谈生成的馈送在用户偏好得分（48% 选择）和帖子满意度上显著优于手工描述（p<0.005），而结构化手工条件表现与基线无显著差异。

**⚠️ 局限性**

局限包括：访谈对细粒度优先级的表征不足、对可用帖子库存的依赖导致过度精确导致内容稀缺、实验仅在单一会话内评估，缺乏长期使用和心理影响的数据。

---

## 474. Prefer-DAS: Learning from Local Preferences and Sparse Prompts for Domain Adaptive Segmentation of Electron Microscopy

**arXiv ID:** 2602.19423 | [PDF](https://arxiv.org/pdf/2602.19423v1)

**作者:** Jiabao Chen `[一作]` (Huaqiao University), Jialin Peng `[通讯]` (Huaqiao University)

**通讯引用:** 1350 | [OpenAlex ID](https://openalex.org/A5002811823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Prefer-DAS 框架，结合稀疏提示与局部/无监督偏好学习，实现无监督、弱监督及交互式跨域电镜图像分割；

**💡 创新点**

创新点在于：① 通过稀疏中心点提示的 promptable 多任务模型实现全/无提示一次性分割；② 引入局部直接偏好优化（LPO）和稀疏局部偏好优化（SLPO），以及无监督偏好优化（UPO），弥补传统 DPO 在空间上不一致的问题；③ 将提示引导对比学习与伪标签结合，提升特征区分度；

**🔧 技术方法**

使用 Transformer（SAM 类解码器）+ DINO 视觉编码器，伪标签（mean‑teacher）、Prompt‑guided Contrastive Learning、Direct Preference Optimization（多负样本）、自监督边缘活化级联；

**📊 数据集**

在四个跨域电镜数据集上评估：MitoEM‑Human、MitoEM‑Rat、Lucchi++、ME2‑Stem；

**📈 对比分析**

与 SAM、SAM‑Med2D、Med‑SAM Adapter、WeSAM 以及 UDA/WDA 方法（如 UALR、CAFA、WDA‑Net、DAMT‑Net 等）对比，Prefer‑DAS 在自动模式下几乎达到或超过有监督上限，在交互模式下在三项任务中优于有监督模型，且在稀疏提示/偏好场景下仍保持显著性能；

**⚠️ 局限性**

局限性：仅考虑单步域适配，无法应对持续出现的新域；缺乏对不同提示类型（如框、掩模）兼容性的进一步探索；

---

## 475. MultiDiffSense: Diffusion-Based Multi-Modal Visuo-Tactile Image Generation Conditioned on Object Shape and Contact Pose

**arXiv ID:** 2602.19348 | [PDF](https://arxiv.org/pdf/2602.19348v1)

**作者:** Sirine Bhouri `[一作]` (Imperial), Dandan Zhang `[通讯]` (Imperial)

**通讯引用:** 5019 | [OpenAlex ID](https://openalex.org/A5100386760)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MultiDiffSense，一种统一的扩散模型，用来在单一架构中合成 ViTac、TacTip 与 ViTacTip 三种视觉触觉传感器的对齐图像，并实现基于 CAD 深度图和结构化文本提示的可控生成。

**💡 创新点**

创新点：① 在单模型中实现多模态生成，替代传统多模型方法；② 双重条件——CAD 生成的姿态对齐深度图与包含传感器类型与 4-DoF 接触姿态的文本提示，保证生成的图像在物理与几何上保持一致；③ 通过将 ControlNet 与 Stable Diffusion 结合，实现高质量、对齐的数据合成。

**🔧 技术方法**

使用技术：Stable Diffusion v1.5 + ControlNet 双条件扩散框架、CLIP 文本编码、VAE 潜在空间、Zero-convolution 连接、DDIM 采样、AdamW 优化、对齐深度图生成管线。

**📊 数据集**

数据集：基于 ViTacTip、TacTip、ViTac 原始数据集，选取 5 个物体（直棱、立方体、球体、Pacman、空心圆柱），每模态 500 张图像，总计 7,500 张；使用 70/15/15 的训练/验证/测试划分。

**📈 对比分析**

比较方法：与 Pix2Pix cGAN 单模态基线在 seen-object‑unseen‑pose 与 unseen-object 两种测试场景下进行对比。MultiDiffSense 在 SSIM、PSNR、MSE、LPIPS、FID 等指标上分别提升 36.3%、134.6%、64.7% 等；在下游 3-DoF 姿态估计任务中，混合 50% synthetic + 50% real 数据可实现与全实数据相当的 R² 与误差水平，而纯 synthetic 数据表现不佳。

**⚠️ 局限性**

局限性：① 仅生成单帧静态图像，缺少时序动态；② 只支持 4-DoF 接触参数，未覆盖完整 6-DoF；③ 对透明、反射或纹理复杂表面的合成效果有限；④ 需要大量高质量 CAD 与深度图配对；⑤ 对 TacTip 的 yaw 估计仍有较大误差，表明纯触觉模态的生成仍有挑战。

---

## 476. Prompt Tuning for CLIP on the Pretrained Manifold

**arXiv ID:** 2602.19198 | [PDF](https://arxiv.org/pdf/2602.19198v1)

**作者:** Xi Yang `[一作]` (Guizhou University), Jie Wen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 7261 | [OpenAlex ID](https://openalex.org/A5017617923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ManiPT框架，在冻结的CLIP预训练模型上进行提示调优，加入余弦一致性约束和结构性偏差，并通过LLM生成的语义原型对提示进行知识增强。

**💡 创新点**

1）在预训练manifold上显式约束特征，抑制由局部伪相关驱动的漂移；2）引入结构性偏差实现增量修正，避免在manifold内部陷入快捷路径；3）利用LLM构建稳定语义原型；4）给出理论误差上界，证明在有限监督下能减轻过拟合。

**🔧 技术方法**

Prompt tuning、余弦一致性损失、结构性偏差（加法融合）、LLM知识增强、PCA估计manifold漂移、Rademacher复杂度分析、跨域评估等。

**📊 数据集**

15个公开数据集：ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101；以及ImageNet的四个变体（ImageNet‑V2、ImageNet‑Sketch、ImageNet‑A、ImageNet‑R）。

**📈 对比分析**

与CoOp、CoCoOp、MaPLe、LLaMP、PromptSRC、CoPrompt、TAC、TAP等方法在基于新旧类的一般化、跨数据集迁移、领域泛化和少样本分类四个设置中对比；ManiPT在所有实验均显著优于基线，平均提升2–4个百分点，尤其在1/2 shot、跨域和跨数据集零样本任务中表现突出。

**⚠️ 局限性**

仍存在轻微延迟（双分支融合），对LLM生成描述的质量敏感，PCA参数（主成分维度）影响漂移估计，结构性偏差可能限制更大方向的改动，未在跨模态（如文本检索）或更大规模任务中验证。

---

## 477. Should I Hide My Duck in the Lake?

**arXiv ID:** 2602.18775 | [PDF](https://arxiv.org/pdf/2602.18775v1)

**作者:** Jonas Dann `[一作]` (ETH Zurich), Gustavo Alonso `[通讯]` (ETH Zurich)

**通讯引用:** 17392 | [OpenAlex ID](https://openalex.org/A5103144919)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并评估了将数据解码和推算操作下沉到云端 SmartNIC 的方案，以降低在数据湖查询时对 CPU 的负载。

**💡 创新点**

创新之处在于将 SmartNIC 作为网络数据路径中的数据处理节点，支持行速解码、流式查询引擎以及 SSD 缓存，提供端到端的加速和资源共享。

**🔧 技术方法**

使用了 DuckDB 1.4.4 的查询计划重写扩展、Parquet/CSV/JSON 解码、FPGA 加速的行速解码与推算、SSD 本地缓存及网络堆栈协同。

**📊 数据集**

实验基于 TPC‑H 规模因子 10 与 30 的 Parquet、CSV 与 JSON 数据集，并对随机与排序的 Parquet 进行比较。

**📈 对比分析**

通过在双路 AMD EPYC 7V13 服务器上测量 5 次中位数，结果显示在预过滤数据下 16 线程可匹配 64 核 CPU，且 Parquet 通过率比 CSV/JSON 高 14–16 倍，推算操作能显著缩短扫描密集查询时间。

**⚠️ 局限性**

主要局限在于实现 100 Gbps 行速多格式解码、与主机数据库的零拷贝交互以及 SSD 缓存管理仍未成熟，需要进一步的硬件软件协同设计与多查询调度研究。

---

## 478. VGGT-MPR: VGGT-Enhanced Multimodal Place Recognition in Autonomous Driving Environments

**arXiv ID:** 2602.19735 | [PDF](https://arxiv.org/pdf/2602.19735v1)

**作者:** Jingyi Xu `[一作]` (Shanghai Jiao Tong University), Ling Pei `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 VGGT-MPR 框架，利用 VGGT 作为统一几何引擎实现多模态位姿识别，并设计了无训练的重排序机制。

**💡 创新点**

创新点在于将视觉几何基底模型 VGGT 重新诠释为多模态融合的几何引擎，同时提供训练‑free 的跨视角点跟踪重排序方案。

**🔧 技术方法**

采用 VGGT 基础模型、深度监督与点云稠密化、交叉变压器与 NetVLAD 进行全局特征提取，结合关键点跟踪与置信度加权的重排序技术。

**📊 数据集**

使用 nuScenes、NCLT、KITTI 三大公开数据集以及作者自采集的无人车数据集进行评估。

**📈 对比分析**

与多种 SOTA 单模态、双模态方法对比，实验表明 VGGT‑MPR 在所有基准上均取得最高或接近最高的 AR@1/5/10/20，尤其在零样本与极端环境下表现突出。

**⚠️ 局限性**

仍受限于对 VGGT 预训练权重的依赖，且对极端光照、极度稀疏点云或高度动态场景的鲁棒性有限。

---

## 479. Restoration-Guided Kuzushiji Character Recognition Framework under Seal Interference

**arXiv ID:** 2602.19086 | [PDF](https://arxiv.org/pdf/2602.19086v1)

**作者:** Rui-Yang Ju `[一作]` (Kyoto University), Shinsuke Mori `[通讯]` (Kyoto University)

**通讯引用:** 1679 | [OpenAlex ID](https://openalex.org/A5001224773)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个三阶段的恢复导向熊须字识别框架（RG‑KCR），通过检测、图像修复和分类实现含印章干扰文档的识别。

**💡 创新点**

创新点在于结合无训练的红印章去除方法与单字符分类，显著提升印章重叠情况下的识别准确率。

**🔧 技术方法**

使用YOLOv12‑medium进行字符检测，基于颜色阈值+形态学+OpenCV Inpaint实现文档修复，最后采用ViT‑基模型Metom进行字符分类。

**📊 数据集**

构建了含10个重叠印章的合成文档测试集（100张、17,982字符），并使用CODH原始文档数据集进行检测与修复评估。

**📈 对比分析**

检测精度达98.0%/召回率93.3%，修复后PSNR约34 dB、SSIM ≈ 0.975；在分类上从93.45%提升至95.33% Top‑1（Top‑5 97.46%→98.62%），仅额外耗时0.51 s/图。

**⚠️ 局限性**

局限性在于仅实现单字符识别，未恢复阅读顺序与连贯文本；密集印章区的修复可能导致局部纹理失真。

---

## 480. A Formal Framework for Predicting Distributed System Performance under Faults

**arXiv ID:** 2602.19088 | [PDF](https://arxiv.org/pdf/2602.19088v1)

**作者:** Ziwei Zhou `[一作]` (East China Normal University), MIn Zhang `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个面向分布式系统的正式框架，能够在包含多种故障的环境下从模型直接预测吞吐量、延迟等性能指标。

**💡 创新点**

创新点在于构建了可复用的故障库并通过故障注入器实现系统与故障模型的模块化组合；保证了无非确定性属性，支持统计模型检测；以及将模型预测结果与真实部署进行系统对比。

**🔧 技术方法**

技术上使用 Maude 形式化语言实现概率重写理论；设计故障注入器（调度器、控制器、处理器）实现消息拦截、故障触发；利用 PVeStA 统计模型检测与 QuaTEx 进行性能量化。

**📊 数据集**

在实验中使用六个代表性分布式系统的 Maude 规范（2PC+CTP、Raft、PowerDNS、Cassandra、RAMP、Fast‑HotStuff）以及相应的真实部署（CloudLab、Tencent Cloud、Emulab）作为数据集；故障参数从实验配置得到。

**📈 对比分析**

对比方法是把模型预测的吞吐量、延迟曲线与在真实环境下的测量结果进行曲线匹配；实验显示预测值与部署结果高度一致，误差在可接受范围内，证明框架能够准确捕捉故障对性能的影响。

**⚠️ 局限性**

局限性包括：时间单位抽象导致需要校准比例因子；故障库主要覆盖消息层面的变异，无法直接表达更复杂的硬件或软件故障；模型规模仍受 Maude 与 PVeStA 计算资源限制。

---

## 481. Semantic Conflict Model for Collaborative Data Structures

**arXiv ID:** 2602.19231 | [PDF](https://arxiv.org/pdf/2602.19231v1)

**作者:** Georgii Semenov `[一作]` (ITMO University), Vitaly Aksenov `[通讯]` (ITMO University)

**通讯引用:** 90 | [OpenAlex ID](https://openalex.org/A5069723237)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了基于“蕴含-放弃”语义的冲突模型，支持本地优先的显式冲突解决，并通过对操作的前提与冲突关系进行建模，设计了协同寄存器、算术寄存器、LWW-寄存器和多寄存器等示例；

**💡 创新点**

创新点在于将冲突识别与解决转化为操作间的依赖关系（蕴含）与冲突关系（放弃），实现无中心协调的显式冲突解析，并在此基础上定义了一套可组合的协同数据类型框架；

**🔧 技术方法**

使用的技术主要包括：操作型日志、蕴含图（有向无环图）来表示前提依赖、三向合并的重基化（rebase）技术、可撤销操作（tombstone）以及基于状态复制的同步算法；

**📊 数据集**

该工作为理论性工作，未使用具体数据集，而是在模型层面给出了多种寄存器示例；

**📈 对比分析**

论文未提供实验评估或性能对比，仅通过理论推导证明在无新操作且冲突已解决的情况下系统能够强一致收敛；

**⚠️ 局限性**

主要局限包括：尚未完成完整的可行性证明，缺乏针对大规模并发场景的实验评估，且模型在多操作冲突的并发解决上仍可能导致额外的调解步骤；

---

## 482. The Category Mistake of Cislunar Time: Why NASA Cannot Synchronize What Doesn't Exist

**arXiv ID:** 2602.18641 | [PDF](https://arxiv.org/pdf/2602.18641v1)

**作者:** Paul Borrill `[一作]` `[通讯]`, Paul Borrill

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文以量子力学与分布式系统的范畴错误理论为框架，批判目前 NASA 的协调月球时间 (LTC) 架构，将同步时间视为本体实体的错误，并提出基于双向事务的时间传递方案。

**💡 创新点**

创新点在于将 Ryle、Spekkens 的范畴错误与 FITO 论点与月球时间同步结合，揭示同步时间的本体/认识范畴误解，并设计了以双向事务为核心的交易式时钟网络。

**🔧 技术方法**

采用范畴错误分析、FITO 先验、Spekkens 的本体/认识区分、Lamport 时钟、Shannon 信道、Wood‑Spekkens 细调论证，以及 Open Atomic Ethernet（OAE）事务模型。

**📊 数据集**

主要使用 IAU Resolution II、NIST 纠正框架、China LTE440 软件及其 0.15 ns 精度声明等公开标准与软件作为案例数据；未提供实验数据。

**📈 对比分析**

对比方法主要是理论推导与现有协议（如 LunaNet、GPS）进行对比，指出单向信道在失联时无法实现共识；未给出定量性能指标。

**⚠️ 局限性**

局限性包括缺乏实测验证、事务模型的工程实现细节缺失，以及对大规模分布式系统中时延与失联场景的完整评估不足。

---

## 483. Evolution of fairness in hybrid populations with specialised AI agents

**arXiv ID:** 2602.18498 | [PDF](https://arxiv.org/pdf/2602.18498v1)

**作者:** Zhao Song `[一作]` (Teesside University), The Anh Han `[通讯]` (Teesside University)

**通讯引用:** 3869 | [OpenAlex ID](https://openalex.org/A5012915897)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在双重角色分化的Ultimatum游戏中，专门化AI代理对人类公平行为演化的影响。

**💡 创新点**

创新点是引入异质角色的双向混合群体模型，并证明AI接收者比AI提出者更能促进公平，同时提出基于预测接收者阈值的差别化AI提出者。

**🔧 技术方法**

使用了有限well‑mixed种群中的马尔可夫链、Moran过程以及Fermi选择函数等进化动力学工具。

**📊 数据集**

通过理论模型设定（h=0.5、l=0.1）并在h、l、β等参数空间进行数值实验，没有使用真实数据集。

**📈 对比分析**

对比不同AI类型在不同选择强度下的平稳分布，发现差别化AI在强选择下仅需少量代理即可实现全公平，性能优于Samaritan AI。

**⚠️ 局限性**

模型假设完全信息、无噪声、well‑mixed交互，未考虑网络结构、学习型AI以及实际实现中的成本与不确定性。

---

## 484. Semantic Caching for OLAP via LLM-Based Query Canonicalization (Extended Version)

**arXiv ID:** 2602.19811 | [PDF](https://arxiv.org/pdf/2602.19811v1)

**作者:** Laurent Bindschaedler `[一作]` `[通讯]` (Max Planck Institute for Software Systems), Laurent Bindschaedler (Max Planck Institute for Software Systems)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种安全优先的 OLAP 缓存中间件，将 SQL 与 NL 查询统一映射为 OLAP Intent Signature，实现跨客户端、跨表面形式的缓存复用。

**💡 创新点**

创新点在于：① 通过 OLAP Intent Signature 消除 SQL 与 NL 的表面碎片化；② 采用置信度门控与安全推导（roll‑up、filter‑down）确保零误命中；③ 限定查询范围为星型/雪花单事实表，保证语义等价可判定。

**🔧 技术方法**

技术实现：Python 中间件 + DuckDB 后端；LLM（GPT‑4o‑mini）生成约束 JSON；JSON 序列化 + SHA‑256 哈希；缓存以 Parquet + SQLite 存储；严格的 schema 验证与置信度门控。

**📊 数据集**

使用的数据集包括 TPC‑DS、SSB、NYC TLC 共 1,395 条 SQL/NL 查询；对 NL 评测 63 句对抗问句和 150 条 BIRD 人工问句。

**📈 对比分析**

与文本、AST、NL‑to‑SQL+AST 三个基线比较，hit 率最高 82%，后端计算下降 85–90%；安全推导将层级查询 hit 率提升至 80%，所有方法均实现零误命中。

**⚠️ 局限性**

局限性：仅支持星型/雪花单事实表查询；NL 识别准确率仅 44%（对抗查询）；无法处理窗口函数、CTE、集合运算等；缓存失效需要手动刷新；推导不适用于非加法度量和 top‑k 查询。

---

## 485. Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection

**arXiv ID:** 2602.19631 | [PDF](https://arxiv.org/pdf/2602.19631v1)

**作者:** Uichan Lee `[一作]` (Seoul National University of Science and Technology), Sangheum Hwang `[通讯]` (Seoul National University of Science and Technology)

**通讯引用:** 1845 | [OpenAlex ID](https://openalex.org/A5091438057)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HiRM 方法，通过仅微调 CLIP 文本编码器的第一层并在最终层引导高层语义向随机或语义向量偏移，从而实现对目标概念（如风格、物体、NSFW）的精准消除，同时保持生成质量。

**💡 创新点**

创新点在于将更新位置与消除目标解耦：只更新早期层的权重，利用高层语义表示做误导，既避免了对 U‑Net 的大规模修改，又实现了对非目标概念影响极小的高效、可迁移的概念消除。

**🔧 技术方法**

核心技术包括：CLIP 文本编码器微调、随机/语义向量误导损失、对抗攻击评估、与 U‑Net 基准方法（ESD、UCE、CA 等）融合、在 Flux 等非 U‑Net 体系中直接迁移。

**📊 数据集**

使用的数据集包括：UnlearnCanvas（风格+物体），I2P（NSFW/攻击），COCO‑30k（质量评估），以及对抗攻击数据集 Ring‑A‑Bell、MMA、I2P 等。

**📈 对比分析**

与多种训练/无训练基线对比（ESD、UCE、CA、MACE、SALUN、RECE、Diff‑Q、Ediff、SHS 等），HiRM‑R/S 在 UnlearnCanvas 上 UA/IRA/CRA 均超过 90%，平均分数 ~95–96%；在 NSFW 攻击上攻击成功率显著低于对手；在 Flux 迁移中保持 CLIP 得分不变；与 U‑Net 方法结合可显著降低攻击成功率，保持生成质量。

**⚠️ 局限性**

局限性：对所有 token 统一误导，忽略 token 重要性；目前仅在单概念场景验证，组合概念、复杂提示的效果待进一步验证；对极端 NSFW 细节仍有改进空间。

---

## 486. Path planning for unmanned surface vehicle based on predictive artificial potential field. International Journal of Advanced Robotic Systems

**arXiv ID:** 2602.19062 | [PDF](https://arxiv.org/pdf/2602.19062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 487. Marginalized Bundle Adjustment: Multi-View Camera Pose from Monocular Depth Estimates

**arXiv ID:** 2602.18906 | [PDF](https://arxiv.org/pdf/2602.18906v1)

**作者:** Shengjie Zhu `[一作]` (Michigan State University), Wen-Sheng Chu `[通讯]` (Google)

**通讯引用:** 2231 | [OpenAlex ID](https://openalex.org/A5077732223)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种利用单目深度估计（MDE）进行多视角相机姿态估计与3D重建的系统

**💡 创新点**

设计了Marginalized Bundle Adjustment（MBA）目标函数，基于RANSAC思想将残差分布的面积（AUC）作为鲁棒评分，消除了阈值选择的敏感性；同时实现了从高方差稠密深度图直接优化相机姿态的稀疏-稠密融合框架

**🔧 技术方法**

使用预训练的深度网络（如DUSt3R、ZoeDepth、UniDepth等）获取稠密深度图，使用RoMa或MASt3R等对应关系网络得到像素对应；在优化中采用梯度下降（Adam）实现MBA；构建稀疏相机图并进行粗细两阶段Bundle Adjustment

**📊 数据集**

在ScanNet、ETH3D、IMC2021、Tanks&Temples、7-Scenes、Wayspots等公开数据集上进行评测；对单目深度与对应关系模型组合进行多场景实验

**📈 对比分析**

与经典SfM（COLMAP、DF‑SfM、MASt3R‑SfM等）、学习式SfM（VGG‑SfM、FlowMap、MASt3R‑SfM等）以及重定位基线（HSCNet++、DSAC*、MAREPO等）进行对比。MBA在多数数据集上实现了SOTA或接近SOTA的姿态精度（如ETH3D RRA/RTA、IMC2021 AUC、Tanks&Temples 近似匹配），并且能在数千帧的全局BA中保持可扩展性

**⚠️ 局限性**

使用一阶优化器导致相较于传统优化式SfM耗时更长；MBA对深度模型的依赖仍有限，若深度预测误差过大会影响结果；未与基于Transformer的全景基础模型（如VGGT）实现紧耦合，可能进一步提升性能

---

## 488. On Voronoi diagrams in the Funk Conical Geometry

**arXiv ID:** 2602.18980 | [PDF](https://arxiv.org/pdf/2602.18980v1)

**作者:** Aditya Acharya `[一作]` (University of Maryland), Danesh Sivakumar `[通讯]` (University of Maryland)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究了前向和逆向 Funk 计量在多维椭圆锥和三维多边锥中的 Voronoi 图，并给出了相应的构造算法。

**💡 创新点**

创新点包括：①证明 bisector 是从锥顶发出的射线；②将 d 维 Funk Voronoi 图降到 (d‑1) 维加权 Voronoi 图；③利用 Apollonius 图和抽象 Voronoi 图实现椭圆锥和多边锥的高效构造；④给出 3 维锥中三点环心存在性的完整判定。

**🔧 技术方法**

主要技术手段包括：几何性质证明、射线分解、加权 Voronoi 图、Apollonius 图算法、抽象 Voronoi 图框架以及对称性与射线延展的分析。

**📊 数据集**

本文未使用真实数据集，所有结果均基于理论算法和数学证明；实验验证未给出。

**📈 对比分析**

算法复杂度为：椭圆锥中的 Voronoi 图 O(n^{d-1/2+1})，三维多边锥中的 Voronoi 图 O(m n log n)。与现有文献无直接对比，但相对于传统 Voronoi 计算在 Funk 计量下显著提升了可行性。

**⚠️ 局限性**

局限性在于：①多边锥结果仅限三维；②未探讨 Funk Voronoi 与 Hilbert Voronoi 的具体关系；③对其他几何结构（如 Delaunay 三角剖分、最近邻搜索）在 Funk 计量下的算法尚未给出。

---

## 489. Watermarking LLM Agent Trajectories

**arXiv ID:** 2602.18700 | [PDF](https://arxiv.org/pdf/2602.18700v1)

**作者:** Wenlong Meng `[一作]` (Zhejiang University), Wenzhi Chen `[通讯]` (Zhejiang University)

**通讯引用:** 4009 | [OpenAlex ID](https://openalex.org/A5101562846)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种用于LLM代理轨迹数据集的行为级数字水印方法

**💡 创新点**

通过在轨迹中注入隐藏的“钩子动作”并以密钥触发，既不改变任务结果又能在后续模型中学习到水印

**🔧 技术方法**

利用LLM生成钩子动作、统计熵分析确定注入位置、黑盒检测方法以及t检验评估水印显著性

**📊 数据集**

在MATH、SimpleQA、SWE‑Smith三类代理轨迹数据集上进行实验，采用Qwen‑2.5‑Coder、Qwen‑3‑Coder等模型

**📈 对比分析**

与基准CodeMark相比，水印在单提示、单查询下平均AUC>90；在不同模型规模、比例下表现稳健，且对去水印攻击（过滤、改写、摘要）保持较高检测率

**⚠️ 局限性**

局限性包括：对较小或极短轨迹学习效果有限，水印比例需平衡隐蔽性与可检测性；对极大模型或更复杂任务的迁移性能尚未彻底验证

---

## 490. Enhancing Goal Inference via Correction Timing

**arXiv ID:** 2602.18603 | [PDF](https://arxiv.org/pdf/2602.18603v1)

**作者:** Anjiabei Wang `[一作]` (Yale University), Tesca Fitzgerald `[通讯]` (Yale University)

**通讯引用:** 187 | [OpenAlex ID](https://openalex.org/A5044374534)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究人类在机器人执行任务时物理纠正的时机如何为机器人学习任务目标提供信息，并基于Transformer构建时机预测模型与目标推断模型。

**💡 创新点**

创新点：1）将干预时机视为学习信号，探讨其对目标推断的价值；2）设计多维轨迹特征与Transformer结合，显著提升时机预测；3）将时机信息与空间信息联合的COMBINED模型，提升早期目标推断精度。

**🔧 技术方法**

使用技术：Transformer时机预测网络；多特征轨迹抽取（期望对齐、速度一致性、效率、直接性等）；多层感知机+高斯混合模型进行空间推断；基于贝叶斯推理的时机+空间联合目标推断。

**📊 数据集**

数据集：基于Kinova Gen3 + Robotiq 2F‑85抓手的实验数据，共120名参与者、7435个交互回合、3585条纠正轨迹，任务为将四种形状插入对应颜色孔的pick‑and‑place。

**📈 对比分析**

比较方法：与Boltzmann单特征基线对比，使用F1、MAE、纠正比例评估时机预测；目标推断使用KLD与WHEN、WHERE、COMBINED模型比较；结果显示多特征Transformer显著优于基线，COMBINED模型在早期干预时KLD最低，后期时机对改进有限。

**⚠️ 局限性**

局限性：仅考虑首个纠正，未处理多次纠正；任务相对简单，终点已接近目标，导致时机信息对终点推断增益有限；特征集可能未覆盖所有影响人类干预的因素；模型尚未集成至实时规划。

---

## 491. Ani3DHuman: Photorealistic 3D Human Animation with Self-guided Stochastic Sampling

**arXiv ID:** 2602.19089 | [PDF](https://arxiv.org/pdf/2602.19089v1)

**作者:** Qi Sun `[一作]` (City University of Hong Kong), Jing Liao `[通讯]` (City University of Hong Kong)

**通讯引用:** 7957 | [OpenAlex ID](https://openalex.org/A5013972536)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结合分层运动表示与自引导随机采样的框架，用预训练视频扩散模型将粗糙的三维动画重渲为高质量、身份保持的动画视频，并用其监督残差非刚体运动场，实现光照逼真、物理可行的3D人体动画。

**💡 创新点**

创新点：①分层运动表示，将SMPL骨骼刚体运动与隐式残差运动场相结合；②自引导随机采样，利用随机SDE采样修正OOD渲染并通过自引导保持身份；③对角视角-时间采样，降低多轨道不一致，加速4D优化。

**🔧 技术方法**

技术手段：kinematics-based SMPL动画、3D Gaussian Splatting、个性化视频扩散模型（Wan2.1-1.3B）、流匹配（Flow Matching）、随机SDE采样、Diffusion Posterior Sampling自引导、对角视角-时间采样、L1/SSIM/LPIPS/Mask/Regularization等损失。

**📊 数据集**

数据集：ActorsHQ单视图图像与提取的SMPL运动序列用于定量评估；公开单视图视频与3D图像扩散训练集用于个性化扩散模型预训练。

**📈 对比分析**

与LHM、Disco4D、SV4D 2.0、PERSONA四个SOTA方法对比，PNSR/SSIM/LPIPS/CLIP-Image/FID/FVD等指标均优于对手；在用户研究中在身份保持、帧质量、运动真实性、非刚体物理可行性等方面获得最高分；数值上比最强对手提升约18.8 FID。

**⚠️ 局限性**

限制：视频扩散生成耗时长；对大尺度运动或多人物/多视角场景的适应性尚待验证。

---

## 492. DeepInnovator: Triggering the Innovative Capabilities of LLMs

**arXiv ID:** 2602.18920 | [PDF](https://arxiv.org/pdf/2602.18920v1)

**作者:** Tianyu Fan `[一作]` (University of Hong Kong), Chao Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 12191 | [OpenAlex ID](https://openalex.org/A5006594763)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个训练框架，使大型语言模型能够自动生成并迭代优化科学研究思路。

**💡 创新点**

创新点包括：①通过自动化抽取与结构化科学文献知识，构建“巨人肩膀”知识图谱；②设计“下一思路预测”训练任务，配合过程式奖励和解耦的奖励/评论机制，模拟科学的推测与反驳循环；③在14B模型上实现高达80–94%胜率，并在未见领域保持竞争力。

**🔧 技术方法**

使用技术：大型语言模型（Qwen‑2.5‑14B‑Instruct）、强化学习（GRPO）、自动化知识提取与聚类、层次化信号抽取（Insight、Trending、Serendipity）、奖励‑评论解耦架构、Rubrics与SGI‑Bench评估。

**📊 数据集**

数据集：从 arXiv 取自计算机科学、数学、金融、统计四个学科的论文及其引用，构成 1,012 条目标研究思路训练集与 113 条验证集；实验中亦评估法律、教育、生物技术等未见领域。

**📈 对比分析**

比较方法：将模型与未训练基线、GPT‑4o、Gemini‑2.5‑Pro、Qwen3‑Max、Deepseek‑r1、Grok‑4.1、Minimax‑M2.1 等 LLM 通过 Rubrics 评估与 SGI‑Bench 4 维度（Novelty、Effectiveness、Feasibility、Detail）对决；专家评估三新领域。结果显示 -14B 在四维度赢率 80.5%–93.8%，在 Novelty 与 Feasibility 上常超越 GPT‑4o，并在跨域实验中保持优势。

**⚠️ 局限性**

局限性：模型规模（14B）相对大型 LLM 较弱，仍在某些领域（教育、法律）面临可行性低；奖励设计依赖外部评判，仍有误判或偏好问题；仅使用 arXiv 文献，知识覆盖可能不足；评估多依赖人工与自动化对比，尚未覆盖所有创新维度。

---

## 493. Conversational AI for Automated Patient Questionnaire Completion: Development Insights and Design Principles

**arXiv ID:** 2602.19507 | [PDF](https://arxiv.org/pdf/2602.19507v1)

**作者:** David Fraile Navarro `[一作]` (Macquarie University), Mor Peleg `[通讯]` (University of Haifa)

**通讯引用:** 5305 | [OpenAlex ID](https://openalex.org/A5011142830)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一个基于 GPT-5 的对话式 AI 代理，用话题式交互一次性收集慢性下腰痛患者的 NIH 推荐最小数据集（41 项）PROMs，取代传统繁琐的表单填写；

**💡 创新点**

创新点包括：①将多项问卷题目聚合在同一话题对话中一次性收集，②使用交通灯色彩视觉反馈展示对话收集数据的置信度，③将临床决策支持系统（CDSS）的设计原则迁移并扩展到对话界面，④通过迭代 Prompt 工程与多模态输入（表单/聊天/语音）实现灵活交互与数据质量保障，⑤固定 LLM 版本以避免模型升级导致行为漂移。

**🔧 技术方法**

主要技术：GPT-5（以及 GPT-4o）、Prompt Engineering、Chatbot Assessment Reporting Tool (CHART)、OpenAI Playground、Custom GPT、OpenWebUI（自托管）、多模态输入接口。

**📊 数据集**

使用 NIH Chronic Low Back Pain Task Force 推荐最小数据集（41 项多项选择题）作为数据源。

**📈 对比分析**

方法上与传统在线表单比较，利用 SUS、NPS 等可用性指标（SUS 69.7 vs 67.7，NPS 24 vs 13）表明对话式收集更受欢迎；先前相关研究显示对话式 AI 能将专科就诊时长缩短 28.7%；本研究通过患者消费面板与临床医生的 pilot 评估获得正面反馈，但未给出大规模量化性能数据。

**⚠️ 局限性**

局限性包括：①不同 LLM 版本对提示效果差异大，需要固定模型并频繁重新调试；②对话生成随机性导致置信度表达和错误信息（如身高体重顺序）无法完全校正；③未实现多语言与语音模式，需进一步适配；④缺乏大规模多中心量化评估，当前仅基于小样本面板；⑤患者隐私与数据访问需进一步完善。

---

## 494. IDLM: Inverse-distilled Diffusion Language Models

**arXiv ID:** 2602.19066 | [PDF](https://arxiv.org/pdf/2602.19066v1)

**作者:** David Li `[一作]` (Mohamed Bin Zayed University of AI), Alexander Korotin `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过逆向蒸馏（Inverse Distillation）把预训练的扩散语言模型（DLM）压缩为仅需几步采样即可产生高质量文本的学生模型。

**💡 创新点**

1) 在离散域中证明逆向蒸馏目标具有唯一最优解；2) 引入梯度稳定的松弛方法，使离散空间中的梯度传播可行；3) 将方法统一到多种 DLM（SEDD、MDLM、Duo 等）并实现多步蒸馏。

**🔧 技术方法**

逆向蒸馏框架、Concrete Score Matching、SEDD/MDLM/Duo 损失、概率单纯形松弛、重参数化（Gumbel‑Softmax、正态重参数化）以及多步蒸馏训练策略。

**📊 数据集**

OpenWebText 数据集（与 GPT‑2 训练数据相同）。

**📈 对比分析**

与原始教师模型（SEDD、MDLM、Duo、Duo‑DCD）在生成困惑度（GenPPL）和序列熵上对比，结果显示：在保持或接近教师模型的 GenPPL 与熵的前提下，采样步骤分别从 1024 步压缩到 256 步（SEDD）、16 步（MDLM）、16/8/4 步（Duo/Duo‑DCD），速度提升 4×–256×。

**⚠️ 局限性**

1) 评估指标（GenPPL、熵）可能不足以全面衡量生成质量，尤其是大规模模型；2) 仅在 GPT‑2 规模模型上验证，缺乏在更大模型（如 LLaMA、LLaDA）和下游基准上的评估；3) 对离散空间梯度松弛的理论保证虽然成立，但实现复杂度仍高。

---

## 495. DerMAE: Improving skin lesion classification through conditioned latent diffusion and MAE distillation

**arXiv ID:** 2602.19848 | [PDF](https://arxiv.org/pdf/2602.19848v1)

**作者:** Francisco Filho `[一作]` (Universidade Federal de Pernambuco), Tsang Ing Ren `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

解决皮肤病变分类中的类别不平衡问题，先利用类条件扩散模型合成大量恶性病变图像，再在合成数据上使用Mask Autoencoder对大规模ViT-H进行自监督预训练，随后通过知识蒸馏将其表征迁移到轻量级ViT‑B/EfficientNet-B0，实现移动端可部署的高性能分类器。

**💡 创新点**

创新点在于三项技术的有机结合：①类条件扩散生成用于平衡极度不平衡的医疗数据；②在合成数据上进行MAE自监督预训练以获得更具泛化性的视觉表示；③通过软标签蒸馏将大模型知识压缩到轻量级网络，使其既能保持高准确率，又能满足资源受限的临床场景。

**🔧 技术方法**

使用DDPM类条件扩散模型（含MSE+感知损失）合成图像；Mask Autoencoder (MAE)在ViT‑H/16上进行自监督预训练；知识蒸馏采用软标签（KL散度+交叉熵）将教师（ViT‑H）迁移到学生（ViT‑B/16或EfficientNet‑B0）。

**📊 数据集**

HAM10000皮肤镜像数据集（约10,000张，8类，恶性占约10%），用于训练与评估。

**📈 对比分析**

与ViT‑L/B基线、EfficientNet系列、Derm‑t2im等方法对比；在二分类与多分类任务上，MAE+合成+蒸馏方案分别取得ACC≈0.90/0.88、F1≈0.89/0.87，显著优于所有基线，尤其在恶性样本上提升显著。

**⚠️ 局限性**

局限性包括：合成图像的多样性与真实病变仍有差距；扩散模型与MAE预训练对计算资源要求高；蒸馏效果高度依赖教师模型质量；轻量级模型在极低算力或极低延迟场景下仍面临挑战。

---

## 496. Evaluating SAP RPT-1 for Enterprise Business Process Prediction: In-Context Learning vs. Traditional Machine Learning on Structured SAP Data

**arXiv ID:** 2602.19237 | [PDF](https://arxiv.org/pdf/2602.19237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 497. TimeRadar: A Domain-Rotatable Foundation Model for Time Series Anomaly Detection

**arXiv ID:** 2602.19068 | [PDF](https://arxiv.org/pdf/2602.19068v1)

**作者:** Hui He `[一作]` (Singapore Management University), Guansong Pang `[通讯]` (Singapore Management University)

**通讯引用:** 5753 | [OpenAlex ID](https://openalex.org/A5039104219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 TimeRadar，一种可旋转时频域的基础模型，用于零/少样本时间序列异常检测。

**💡 创新点**

创新点是学习可调的分数阶傅里叶变换角度，将时间序列投射到连续时频域，并结合上下文偏差学习提升局部异常判别。

**🔧 技术方法**

采用分数阶傅里叶变换 (FRFT)、Patch+Mask、Fractionally Modulated Encoder (带相位调制)、Contextual Deviation Learning、MSE+margin 损失等技术。

**📊 数据集**

在 Monash 预训练数据集（约4.08亿点）上预训练，随后在8大公开 TSAD 数据集（SMD、MSL、PSM、SWaT、SMAP、CICIDS、SWAN、Creditcard）以及 UCR 基准上评测。

**📈 对比分析**

与13种现有方法（包括 GPT4TS、TimesFM、Chronos-Bolt、DADA、CATCH 等）进行零样本与少样本对比，TimeRadar 在 AUC‑R、AUC‑P、Aff‑F1 上平均提升 10.5%/29.4%/约8%，在大多数数据集名列前茅。

**⚠️ 局限性**

限制在于模型仍依赖预训练数据的多样性，对极端稀有异常的检测仍有挑战，且参数规模与推理速度相对传统 TSAD 方法略高。

---

## 498. Rethinking Retrieval-Augmented Generation as a Cooperative Decision-Making Problem

**arXiv ID:** 2602.18734 | [PDF](https://arxiv.org/pdf/2602.18734v1)

**作者:** Lichang Song `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**通讯引用:** 250718 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了CoRAG框架，将检索器和生成器视为协作多智能体，通过共享任务奖励共同训练，以降低生成器对检索排序的非对称依赖。

**💡 创新点**

创新点在于将RAG改造成多智能体决策问题，构建了检索器与生成器的协同优化机制，使二者在生成过程中相互补强而非单向依赖。

**🔧 技术方法**

技术上使用了GRPO强化学习、组相对优势（group-relative preference）与对比学习的损失、LoRA微调、BGE-Reranker作为检索器以及Llama-3-Instruct-8B作为生成器。

**📊 数据集**

仅在PopQA数据集上进行训练，随后在PopQA、TriviaQA、NQ、2WikiMultiHopQA和ASQA等五个知识密集型基准上进行评估。

**📈 对比分析**

与InstructRAG、RetRobust、Self‑RAG等基线比较，CoRAG在PopQA（71.2%）、TriviaQA（81.0%）、NQ（72.4%）和2WikiMultiHopQA（58.2%）等四个主要任务上实现了SOTA，并在代码与表格推理任务上亦表现出色。

**⚠️ 局限性**

局限性包括：生成器对检索排序不再高度敏感，导致进一步提升检索器的收益有限；对多答案、多跳等更复杂任务（如ASQA）的泛化能力仍有待提升。

---

## 499. TextShield-R1: Reinforced Reasoning for Tampered Text Detection

**arXiv ID:** 2602.19828 | [PDF](https://arxiv.org/pdf/2602.19828v1)

**作者:** Chenfan Qu `[一作]` (South China University of Technology), Lianwen Jin `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TextShield-R1，一种基于多模态大型语言模型（MLLM）的强化学习框架，用于检测并推理文本图像伪造。

**💡 创新点**

创新点包括：① Forensic Continual Pre-training（循序渐进的取证预训练）让 MLLM 适应微观伪造检测；② Group Relative Policy Optimization（GRPO）强化学习方法，使用多维奖励减少对昂贵标注的依赖并提升推理能力；③ OCR Rectification 通过 OCR 结果校正定位框，显著提升定位精度；④ 构建 Text Forensics Reasoning（TFR）基准，覆盖多域、多语言、全新伪造技术与丰富推理注释。

**🔧 技术方法**

使用 Qwen2.5-VL-7B 作为基模型，结合 LoRA 微调、AdamW 优化器、OCR 引擎、Levenshtein 距离、交叉熵等技术；预训练阶段采用 3D Forensic Learning 与 OCR 参考定位任务；微调阶段采用 GRPO。

**📊 数据集**

主要数据集为 TFR benchmark（约 45,971 张伪造图与 45,514 张真实图，16 种语言、10 种伪造技术）以及 120k 以上的自然图像取证数据、COCO、LAION 等作为预训练素材。

**📈 对比分析**

在 TFR 基准及公开基线（MiniCPM、InternVL、FakeShield 等）上，TextShield-R1 在图像级分类、文本识别、定位和推理任务上均超过所有对比模型，尤其在定位（IoU）和推理（平均相似度）上提升显著，最优结果达 88.8% 分类、58.8% OCR、72.9% 定位、85.5% 推理。

**⚠️ 局限性**

局限性：① 仍依赖大规模预训练数据和多模态模型；② 在跨域、跨方法、跨语言的极端 OOD 场景下性能下降；③ 强化学习奖励设计复杂，可能导致训练不稳定；④ 需要 OCR 引擎支持，影响部署成本。

---

## 500. Hilbert-Augmented Reinforcement Learning for Scalable Multi-Robot Coverage and Exploration

**arXiv ID:** 2602.19400 | [PDF](https://arxiv.org/pdf/2602.19400v1)

**作者:** Tamil Selvan Gurunathan `[一作]` (University of Maryland Baltimore County), Aryya Gangopadhyay `[通讯]` (University of Maryland Baltimore County)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出一种基于Hilbert曲线的去中心化覆盖强化学习框架，通过在DQN和PPO中加入Hilbert索引和曲线引导探索，实现稀疏奖励环境下的快速收敛、覆盖率提升和冗余度降低。

**💡 创新点**

创新点在于：①将空间填充曲线嵌入多智能体强化学习，使机器人无需通信即可隐式分区与协同；②将Hilbert索引映射为时间参数化的SE(2)轨迹，直接在Boston Dynamics Spot等硬件上执行；③通过奖励塑造与探索偏向进一步提升稀疏奖励环境下的样本效率。

**🔧 技术方法**

使用技术包括：深度强化学习（DQN、PPO）+Hilbert索引状态增强；曲线引导的探索策略和潜在奖励塑造；Waypoint接口将Hilbert顺序转化为可执行的SE(2)轨迹；仿真环境为32×32/64×64网格，实验环境为10m×10m室内Spot实验室。

**📊 数据集**

数据集：自建的二维网格环境（带障碍、奖励区），以及Spot实验室的实际10m×10m测量空间；未使用公开公开数据集。

**📈 对比分析**

比较方法：与标准DQN/PPO在累计奖励、覆盖率、冗余度和收敛速度上对比；结果显示Hilbert增强模型在4-16机器人规模下收敛速度提升20-50%，覆盖率提升10-30%，冗余度降低20-40%，在Spot上实现更短覆盖时间和更高轨迹吻合度。

**⚠️ 局限性**

局限性：仅验证于规则网格/静态障碍环境，Hilbert曲线对动态或不规则区域不具自适应；对定位漂移、传感噪声的鲁棒性尚未充分评估；扩展到SE(3)或不平坦地形需要进一步研究。

---

## 501. (Perlin) Noise as AI coordinator

**arXiv ID:** 2602.18947 | [PDF](https://arxiv.org/pdf/2602.18947v1)

**作者:** Kaijie Xu `[一作]` (McGill University), Clark Verbrugge `[通讯]` (McGill University)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5001343030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于连续 Perlin 噪声场的多层 AI 协调框架，用以控制大规模游戏代理的运动、激活时机和世界布局。

**💡 创新点**

创新点在于将低频、时空连贯的噪声场直接作为全局协调器，统一实现行为参数化、激活调度和类型特征生成，并可种子化、轻量化。

**🔧 技术方法**

技术主要包括多八度 Perlin 噪声、漂移/重采样更新、区域分位映射、Poisson 过程以及与随机、过滤、确定性、邻域约束和物理启发基线的对比实验。

**📊 数据集**

使用自定义的 2D 方格地图和数千代理的合成实验数据，覆盖多种种子、尺度和时间步长。

**📈 对比分析**

与随机、过滤、确定性、邻域约束和物理启发基线相比，噪声驱动方法在保持局部一致性、平滑激活、空间覆盖和计算效率方面均表现优于或相当，并显著降低锁步现象。

**⚠️ 局限性**

局限包括仅针对运动与生成的基础功能、缺乏碰撞/协作等复杂交互、仅使用二维噪声、未进行用户体验评估以及对更高维/更丰富场景的适用性尚待验证。

---

## 502. Meta-Learning and Meta-Reinforcement Learning - Tracing the Path towards DeepMind's Adaptive Agent

**arXiv ID:** 2602.19837 | [PDF](https://arxiv.org/pdf/2602.19837v1)

**作者:** Björn Hoppmann `[一作]`, Christoph Scholz `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文以严谨的任务为基础的形式化框架，全面梳理并评述了从早期元学习与元强化学习算法到 DeepMind Adaptive Agent 的发展历程，提出统一的性能评估指标，并对关键算法的原理、训练与测试流程进行系统阐释。

**💡 创新点**

创新点包括：①在元学习领域首次给出数学上可计算的任务分布、元目标与性能指标定义；②提出统一的 meta‑training/meta‑testing 视角，兼顾梯度型与记忆型方法；③将 Adaptive Agent 的设计与以往算法的核心技术（如 VAE 任务推理、Transformer 长程记忆、动态奖励引导蒸馏、自动化课程学习）整合到同一框架中；④通过对比表格和实验分析，系统揭示不同算法在样本效率、适应速度、OOV 性能及资源消耗等维度的取舍。

**🔧 技术方法**

技术手段涵盖：梯度基元学习（MAML、FO‑MAML、PEARL 等）、记忆型元学习（RL²、VariBAD、TrMRL 等）、Transformer‑based 元学习（Transformer‑XL、Muesli 等）、模型基 RL 与贝叶斯推理、动态奖励引导蒸馏、自动化课程学习、任务抽样与层级 Transformer 等。

**📊 数据集**

主要数据集与环境：传统网格世界、Atari、MuJoCo、Meta‑World、Crafter、XLand 等 RL 基准；在分类任务中引用 Omniglot、SPL 之类的少量样本数据；在 ADA 评估中使用更大规模的 XLand 任务集合与多智能体环境。

**📈 对比分析**

比较方法：作者定义了通用的 meta‑generalization、adaptation speed、sample‑efficiency 等指标，并通过在同一任务分布下多轮实验（K‑shot fine‑tuning、meta‑validation、meta‑testing）对比各算法；实验表明：梯度型算法在小样本下更快收敛，但 OOD 表现差；记忆型与 Transformer 方案在复杂环境中获得更高的样本效率与 OOD 适应；ADA 在大规模任务池、长上下文长度下表现最佳，但对计算资源需求极高。

**⚠️ 局限性**

局限性：①缺乏跨工作统一的基准与指标，导致结果可比性差；②大多数算法（尤其 Transformer‑based）对数据量与算力要求极高，限制了在边缘设备与小规模任务中的应用；③OOV 与任务不确定性下的稳健性仍不充分，尤其在变异环境中；④缺乏对隐式贝叶斯推理与动态蒸馏等复杂技术的理论分析与收敛证明；⑤对伦理与安全影响的讨论不足。

---

## 503. Keyboards for the Endangered Idu Mishmi Language

**arXiv ID:** 2602.19815 | [PDF](https://arxiv.org/pdf/2602.19815v1)

**作者:** Akhilesh Kakolu Ramarao `[一作]` `[通讯]`, Akhilesh Kakolu Ramarao

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了适用于 Idu Mishmi 语言的 Android 和 Windows 键盘工具，支持完整的字符集并实现离线无管理员权限的输入方式。

**💡 创新点**

创新点在于社区主导的设计流程、对多代码点字符的完整支持以及提供可复用的低成本开发方案，可直接移植至其他濒危语言。

**🔧 技术方法**

采用 HeliBoard 开源 Android 键盘的 Fork、Go 语言实现的低级键盘钩子、Unicode 组合编码、以及与 Keyman 等工具的对比分析。

**📊 数据集**

使用 Idu Mishmi 字符表（包含所有特殊字符和组合序列）以及 约 3,500 条词条的词典数据集，但尚未构建大规模文本语料库。

**📈 对比分析**

目前未进行正式的性能基准测试，计划在社区内进行可用性评估（打字速度、错误率等），已在 100 名教师与社区领袖中部署，满足离线使用需求。

**⚠️ 局限性**

局限性包括：缺少完整词典与预测功能、无法自动纠正粘贴文本中错误的 Unicode 顺序、键盘对安全策略敏感的低级钩子可能被企业防火墙阻止。

---

## 504. Evaluating the Impact of Data Anonymization on Image Retrieval

**arXiv ID:** 2602.19641 | [PDF](https://arxiv.org/pdf/2602.19641v1)

**作者:** Marvin Chen `[一作]` (Stuttgart Media University), Johannes Maucher `[通讯]` (Stuttgart Media University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估匿名化对内容检索系统（CBIR）的影响，提出评估框架并系统实验

**💡 创新点**

首次将匿名化程度与检索、下游任务性能关联，提出三种 DINOv2 训练适配方案与 mnDCG 指标，并揭示检索偏差现象

**🔧 技术方法**

使用 DINOv2 自监督模型、传统匿名化方法（像素化、模糊、遮罩）、mAP 与 mnDCG 评估指标以及 k‑NN/线性分类等下游任务

**📊 数据集**

CelebA、RVL‑CDIP 公开数据集以及内部 DOKIQ 文档数据集

**📈 对比分析**

通过对比未改造、适配 A/B/C 模型在不同匿名化方法与程度下的检索指标，发现未改造模型在检索上表现最优；mnDCG（使用匿名化查询）与下游任务精度高度相关，表明其更能反映匿名化对嵌入空间的影响；在高匿名化程度下部分适配方案甚至优于未改造模型

**⚠️ 局限性**

存在检索偏差（未改造模型天然更匹配伪真值），法律合规性（是否可使用原始嵌入或模型）以及对不同数据集/任务的适用性有限，且传统匿名化方法在信息完整性与隐私保护间仍需权衡

---

## 505. Flash-VAED: Plug-and-Play VAE Decoders for Efficient Video Generation

**arXiv ID:** 2602.19161 | [PDF](https://arxiv.org/pdf/2602.19161v1)

**作者:** Lunjie Zhu `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 85037 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对视频生成中的 VAE 解码器进行加速，提出 Flash‑VAED 系列，通过通道裁剪和阶段化算子优化实现高效解码，并提供三阶段动态蒸馏训练框架。

**💡 创新点**

引入基于线性独立性的通道裁剪、针对各阶段的 CausalConv3D 替换策略以及三阶段动态蒸馏训练，保持与原始潜在分布完全一致。

**🔧 技术方法**

SVD 分析、R² 评估、线性重构、深度可分离卷积、2D 卷积替换、1×1 卷积映射、LPIPS/SSIM/PSNR 评价、三阶段蒸馏损失等技术。

**📊 数据集**

使用 Wan 2.1 与 LTX‑Video VAE 解码器，训练集包含 10K 视频‑潜在对（VidGen + GPT‑4o 生成文本提示），评估集为 UCF‑101 及 VBench 2.0。

**📈 对比分析**

与 Turbo‑VAED 与 LightVAE 对比，Flash‑VAED 在 RTX 5090D 与 Jetson Orin 上实现约 6× 加速，重建质量保持 93–97% 原始水平；端到端生成速度提升 27–36%，VBench 2.0 指标几乎与原始解码器持平。

**⚠️ 局限性**

需依赖原始 VAE 解码器结构，裁剪比例与算子替换需要针对具体网络手工调优；在极低分辨率或特定模型场景下迁移效果可能受限，未针对不同硬件做进一步优化。

---

## 506. Advantage-based Temporal Attack in Reinforcement Learning

**arXiv ID:** 2602.19582 | [PDF](https://arxiv.org/pdf/2602.19582v1)

**作者:** Shenghong He `[一作]` (Sun Yat-sen University), Shenghong He `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 Advantage-based Adversarial Transformer (AAT)，能够在深度强化学习（DRL）环境中一次性生成时间相关且高效的对抗扰动，从而显著降低目标策略的累计奖励。

**💡 创新点**

创新点包括：① 多尺度因果自注意（MSCSA）机制，捕获短期与长期历史信息的多粒度依赖；② 权重优势机制（weighted advantage），通过对扰动优势进行加权来引导生成过程并提升攻击性能；③ 将上述两种机制结合在 Transformer 架构中，实现单步前向生成高质量对抗样本，兼顾攻击效果与生成速度。

**🔧 技术方法**

技术栈主要有 Transformer（MSCSA）、Vision Transformer 视觉分块、Q/V 期望回归、优势学习、权重优势调节、梯度基攻击（FGSM、Skip、S-T、EDGE）对比、奖励基攻击（PA-AD、AdvRL-GAN、TSGE、PIA）等。

**📊 数据集**

使用的数据集包括：Atari 2600 游戏（Breakout、Pong、Chopper Command、Seaquest、Qbert、Space Invaders）、DeepMind Control Suite（Continuous 动作环境）、Google Football、StarCraft Multi-Agent Challenge（2s3z 任务）以及多种目标策略（DQN、A3C、TRPO、PPO、D4PG、QMix 等）。训练集由 expert、medium、random 轨迹构成，规模从数千到数万条轨迹。

**📈 对比分析**

与梯度基攻击、奖励基攻击以及多种白盒/黑盒基线进行对比。AAT 在所有测试环境下均优于基线，白盒情景平均降低累计奖励约 3%（与 PA-AD、AdvRL-GAN 相比），在连续动作和多人合作场景中亦保持领先；生成速度最快（单步前向），检测率最低，表现出更好的隐蔽性和稳定性。

**⚠️ 局限性**

局限性：目前仅针对单智能体攻击；对极度稀缺或高度不确定的黑盒环境仍存在性能波动；在更大规模、多任务或跨域迁移场景下的可扩展性与鲁棒性尚待进一步验证。

---

## 507. MoBind: Motion Binding for Fine-Grained IMU-Video Pose Alignment

**arXiv ID:** 2602.19004 | [PDF](https://arxiv.org/pdf/2602.19004v1)

**作者:** Duc Duy Nguyen `[一作]` (Australian Institute for Machine Learning), Minh Hoai `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出MoBind框架，学习IMU信号与视频骨骼序列的联合表征，实现跨模态检索、同步、定位和动作识别。

**💡 创新点**

在对比学习中引入层次（token、本地、全局）对齐，并结合Mask Token Prediction保证语义保留；同时采用骨骼序列而非原始像素消除背景噪声，并对多传感器IMU进行体部级对齐。

**🔧 技术方法**

1D卷积+Transformer编码器、InfoNCE层次对比损失、Mask Token Prediction、token化与聚合、共享投影空间。

**📊 数据集**

mRi、TotalCapture、EgoHumans三大多模态数据集。

**📈 对比分析**

与IMU2CLIP、DeSPITE、SyncNet等基线对比，MoBind在跨模态检索Recall@1、同步MAE、定位准确率、动作识别精度上均取得最优或显著提升，误差可低至50ms。

**⚠️ 局限性**

对高度重复或短周期动作仍易产生混淆，需足够长窗口；在极短窗口或单传感器场景性能下降；依赖精确骨骼估计，姿态误差会影响整体效果。

---

## 508. Linear Reservoir: A Diagonalization-Based Optimization

**arXiv ID:** 2602.19802 | [PDF](https://arxiv.org/pdf/2602.19802v1)

**作者:** Romain de Coudenhove `[一作]` (ENS PSL), Xavier Hinaut `[通讯]` (Inria Center of Bordeaux University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了通过对线性ESN的转移矩阵进行对角化，将递归更新从O(N^2)降至O(N)的优化方法，显著提升计算效率；

**💡 创新点**

创新点在于三种实现方式——EWT、EET、DPG，尤其是DPG直接从谱分布构造特征值和特征向量，摆脱了显式矩阵生成与对角化的开销；

**🔧 技术方法**

技术包括矩阵对角化、基变换、点乘式状态更新、随机/黄金分布特征值采样以及岭回归读出层训练；

**📊 数据集**

实验使用的标准基准为多频叠加振荡器(MSO)和记忆容量(MC)任务；

**📈 对比分析**

在MSO和MC任务中，DPG（尤其是带噪声的黄金分布）与传统线性ESN的预测误差相当或更优，且计算速度提升数倍；

**⚠️ 局限性**

局限性包括对可对角化矩阵的要求、一次性O(N^3)对角化成本，以及在极低连接率下性能下降。

---

## 509. Gecko: A Simulation Environment with Stateful Feedback for Refining Agent Tool Calls

**arXiv ID:** 2602.19218 | [PDF](https://arxiv.org/pdf/2602.19218v1)

**作者:** Zeyu Zhang `[一作]` (ANU), Liang Zheng `[通讯]` (ANU)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套名为 Gecko 的工具调用模拟环境，并基于该环境提出 GATS（Grounding Agent Test-time Scaling）方法，在测试阶段通过模拟工具执行反馈逐步优化 LLM 的工具调用，从而显著提升 LLM 在 BFCLv3 与 τ²-bench 这两大 benchmark 上的表现。

**💡 创新点**

创新点：
• 设计了四个互补的模拟子系统（参数验证、响应生成、任务状态估计、任务反馈），实现对工具调用的语法、语义、执行结果和任务进度的全维度模拟；
• 通过 GATS 将模拟反馈实时回馈给规划 LLM，实现无实工具调用的自适应迭代优化，避免了实工具调用的高成本和安全风险；
• 使得模拟环境能够自动将任意工具描述转换为 OpenAPI schema，极大提升了工具集成的可扩展性。

**🔧 技术方法**

技术手段：
• 结合规则（正则、类型/范围检查）与 LLM 辅助模型进行参数验证；
• 采用 LLM 生成器模拟工具返回值，条件化于工具 schema 与当前任务状态；
• 通过 LLM 估计任务状态并生成任务完成检查表；
• 采用多轮对话的状态隔离与回放机制实现无缝迭代；
• 通过 OpenAPI 3.1.0 自动化转换实现工具描述到模拟器的无缝桥接。

**📊 数据集**

数据集：
• Berkeley Function Call Leaderboard v3 (BFCLv3)：3,633 任务、8,578 工具，涵盖非实时单轮、实时单轮、多轮三类；
• τ²-bench：τ²-retail（13 APIs，114 任务）与 τ²-airline（12 APIs，50 任务）两子任务集。

**📈 对比分析**

比较与性能：
• 在 BFCLv3 上，GATS 对 GPT‑4o 的整体准确率由 76.93% 提升至 84.62%；对 Qwen‑3‑14B 从 73.78% 提升至 78.60%；对 Gemini‑3.0‑pro 从 85.97% 提升至 88.19%；并在多轮基准中实现 73.50% 的最高准确率；
• 在 τ²-bench 上，GATS 将 GPT‑4o 的整体 pass@1 从 54.2% 提升至 60.7%，将 Gemini‑3.0‑pro 从 73.5% 提升至 82.3%；
• 与其它测试时扩展方法（Reflexion、Self‑refine、Best‑of‑N、Merge‑to‑one）对比，GATS 在 τ²-airline 上以 65.0% 的 pass@1 领跑，且工具调用次数与成本保持在可接受范围。

**⚠️ 局限性**

局限性：
• 仅支持文本输出工具，无法处理图像、音频等多模态工具；
• 对依赖外部数据库的查询工具，模拟结果可能与真实数据有偏差，影响评估准确性；
• 模拟环境仍依赖高质量工具描述与 schema，描述不足会导致模拟失真；
• 迭代次数受限，过多重试会导致延迟与成本上升；
• 目前无法完整复制某些实时交互式工具的细粒度错误反馈，可能导致累积误差。

---

## 510. A Dataset for Named Entity Recognition and Relation Extraction from Art-historical Image Descriptions

**arXiv ID:** 2602.19133 | [PDF](https://arxiv.org/pdf/2602.19133v1)

**作者:** Stefanie Schneider `[一作]` (Marburg University), Ricarda Vollmer `[通讯]` (University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 FRAME 数据集：包含 200 条艺术史图像描述，配备手工标注的三层注释（元数据层、内容层、共指层），涵盖 37 种实体类型和实体间关系。

**💡 创新点**

创新点：① 细粒度、专门针对艺术史的实体与关系类型体系；② 分层注释将对象属性与被描绘内容分离；③ 通过与 Wikidata 对齐实现知识图谱互操作；④ 提供 UIMA XMI 形式的标准化注释，便于下游 NLP 处理。

**🔧 技术方法**

使用技术：Python 异步爬虫 + Selenium 采集文献；OpenAI GPT‑4o 翻译；多步手工注释（Inception 工具）和层级验证流程；Python 自动化校验规则；最终生成 UIMA XMI CAS 与 CSV 元数据。

**📊 数据集**

使用数据集：FRAME（200 条描述、1136 字平均、48 实体/44 关系）以及来自 13 家公开艺术机构（博物馆、拍卖行、学术数据库）的原始文本与图片；与 Wikidata 进行实体对齐。

**📈 对比分析**

对比方法：在零/少样本设定下微调大型语言模型（LLM）进行 NER/RE，使用标准 F1/Precision/Recall 评估。实验结果显示，现有通用模型在艺术史文本上的 F1 明显低于在一般语料上的水平，证明该领域的任务难度和数据稀缺性。

**⚠️ 局限性**

局限性：① 数据主要来自西方博物馆，功能性与跨文化对象被过滤，导致样本偏斜；② 以英语翻译为主，隐含学科和语言偏见；③ 译文与术语准确性仍受限；④ 数据量有限，缺乏对更广泛艺术品类型的覆盖；⑤ 标注过程中仍存在歧义和解释性差异，需人工裁决。

---

## 511. Discrete Diffusion Models Exploit Asymmetry to Solve Lookahead Planning Tasks

**arXiv ID:** 2602.19980 | [PDF](https://arxiv.org/pdf/2602.19980v1)

**作者:** Itamar Trainin `[一作]` (Hebrew University of Jerusalem), Amir Feder `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文比较了自回归（AR）Transformer与非自回归（NAR）离散扩散模型（dLLM）在Star‑Path图形的lookahead规划任务上的学习机制与性能。

**💡 创新点**

创新点在于发现NAR模型可利用任务的方向不对称性，采用反向解码策略将高阶lookahead简化为一阶邻接预测，从而大幅提升样本效率。

**🔧 技术方法**

使用技术包括：Transformer架构、离散扩散训练/推断、全序列（full‑sequence）训练目标、梯度回传到图描述、以及对比分析的解码动态可视化。

**📊 数据集**

数据集为合成的Star‑Path图形，包含图描述、源-目标节点对及从源到目标的路径，使用随机节点分配以消除标识符相关的偏差。

**📈 对比分析**

比较方法：对两种模型在相同图形规模下使用Exact‑Match准确率评估；结果显示NAR在训练样本数级数上显著优于AR（AR需数十亿样本方能收敛），并且NAR在不需要额外预训练的情况下即可实现完美准确率。

**⚠️ 局限性**

局限性包括：实验仅在合成任务上进行，未验证在真实规划场景中的推广性；NAR优势可能仅源于解码灵活性而非更强的推理能力；且对大规模复杂图的适应性仍需进一步研究。

---

## 512. On Identifying Critical Network Edges via Analyzing Changes in Shapes (Curvatures)

**arXiv ID:** 2602.19328 | [PDF](https://arxiv.org/pdf/2602.19328v1)

**作者:** Bhaskar DasGupta `[一作]` (University of Illinois Chicago), Katie Kruzan `[通讯]` (University of Illinois Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究利用 Ollivier‑Ricci 曲率检测无向图中关键边的算法与计算复杂度问题，提供多种问题定义、算法、NP‑完整性与近似性结果，揭示与完全匹配相关的结构联系。

**💡 创新点**

首次系统化构建基于曲率的关键边判定框架，证明多版本问题的 NP‑完整性并给出多项多项式时间/近似算法，展现与完全匹配、匹配阻塞问题的深层联系。

**🔧 技术方法**

利用 Ollivier‑Ricci 曲率定义、运输距离线性规划、完美匹配与匹配阻塞理论、随机化匹配算法（如 Mulmuley‑Vazirani‑Vazirani 方案）以及集合覆盖的逼近下界等组合优化技术。

**📊 数据集**

无，本文为理论分析与算法复杂度研究，不依赖具体数据集。

**📈 对比分析**

无实验比较，主要通过理论证明与归约展示算法的多项式/近似性能，证明部分问题在多项式时间内可解或 近似因子为 b、a+b 等。

**⚠️ 局限性**

仅覆盖部分曲率关键边判定问题，未给出完整多版本分析；缺乏对真实网络的实验验证，随机化算法尚未完全去随机化，且对曲率的其他距离度量讨论有限。

---

## 513. Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning

**arXiv ID:** 2602.18867 | [PDF](https://arxiv.org/pdf/2602.18867v1)

**作者:** Zhuofan Xie `[一作]` (Xiamen University), Shuo Li `[通讯]` (Case Western Reserve University)

**通讯引用:** 50096 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于 VLM 的主动学习框架 SaE，将文本‑图像相似度转化为 Dirichlet 证据以解决冷启动与过度自信问题。

**💡 创新点**

通过 Similarity Evidence Head 将 VLM 相似度映射为 Dirichlet 证据，并将不确定性拆分为 vacuity 与 dissonance，形成双因子采样策略，实现可解释且校准的主动学习。

**🔧 技术方法**

结合 CLIP/MedCLIP 等 Vision‑Language 模型、Prompt Learning（CoOp）与 Evidential Deep Learning、Dirichlet 证据建模及双因子评分函数。

**📊 数据集**

在十个公开医学影像数据集（DermaMNIST、Kvasir、RETINA、LC25000、CHMNIST、BTMRI、OCTMNIST、BUSI、COVID‑QU‑Ex、KneeXray）上实验。

**📈 对比分析**

与随机、PCB、MedCoOp+BADGE 等基线在 20% 标注预算下对比，SaE 在宏平均准确率上达到 82.57%（高出 4.8%），并显著提升校准指标（ECE、NLL）。

**⚠️ 局限性**

仍依赖 VLM 预训练的质量，无法处理完全无标签的领域迁移；双因子策略需手动设置权重；对极少样本类别的 vacuity 估计可能不稳定。

---

## 514. Questions beyond Pixels: Integrating Commonsense Knowledge in Visual Question Generation for Remote Sensing

**arXiv ID:** 2602.19217 | [PDF](https://arxiv.org/pdf/2602.19217v1)

**作者:** Siran Li `[一作]` (Ecole Polytechnique Federale de Lausanne), Devis Tuia `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种结合常识知识的遥感图像视觉问题生成方法，能够生成多样化、信息丰富的针对性问题。

**💡 创新点**

创新点在于将概念网络知识三元组引入问题生成，并通过图像描述作为中间表示提升图像根植性，同时提出三阶段（视觉预训练、语言预训练、微调）训练策略。

**🔧 技术方法**

采用BLIP架构（ViT+BERT）实现图像编码、Caption Decoder、Text Encoder和Question Decoder，并使用概念网络知识句子进行多模态融合。

**📊 数据集**

构建并使用了两个知识感知遥感VQG数据集：NWPU‑300和TextRS‑300，均基于概念网络（ConceptNet）构建知识三元组。

**📈 对比分析**

与IM‑VQG、LMQG、TextRS‑VQG、ConVQG等基线对比，KRSVQG在BLEU‑1/4、METEOR、ROUGE_L、CIDEr等指标上均领先5–10%，例如NWPU‑300上BLEU‑1 41.87、BLEU‑4 14.78、METEOR 18.70、ROUGE_L 38.48、CIDEr 1.24；TextRS‑300上BLEU‑1 44.26、BLEU‑4 22.90、METEOR 19.64、ROUGE_L 42.90、CIDEr 1.47。

**⚠️ 局限性**

局限性包括对手工标注知识三元组的依赖、数据规模仍较小，且在不同遥感场景下的泛化能力尚需进一步验证。

---

## 515. Knowledge-aware Visual Question Generation for Remote Sensing Images

**arXiv ID:** 2602.19224 | [PDF](https://arxiv.org/pdf/2602.19224v1)

**作者:** Siran Li `[一作]`, Devis Tuia `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 16452 | [OpenAlex ID](https://openalex.org/A5005192117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种知识驱动的遥感图像视觉问题生成模型 KRSVQG，利用外部知识句子丰富问题内容并通过图像字幕中介增强图像定位；

**💡 创新点**

创新点在于将外部知识与图像特征通过交叉注意力融合，并使用图像字幕作为中介显著提升问题的图像根基和语义多样性；

**🔧 技术方法**

采用 BLIP 基础架构，结合 Vision Transformer 图像编码器、Caption Decoder、Text Encoder 与 Question Decoder，使用交叉注意力和交叉熵损失进行训练；

**📊 数据集**

使用自构建的 NWPU‑300 与 TextRS‑300 两个遥感 VQG 数据集（共 600 张图像），知识来源为 ConceptNet；

**📈 对比分析**

与 IM‑VQG 与 AutoQG 基线进行对比，评估 BLEU‑1/2/3/4、METEOR、ROUGE_L、CIDEr 等指标，KRSVQG 在 BLEU‑4 提升约 59% 以上、CIDEr 提升约 46% 以上，整体性能最优；

**⚠️ 局限性**

局限性包括仅在遥感图像数据上验证，缺乏跨模态泛化能力，且尚未在完整 VQA 系统中验证鲁棒性。

---

## 516. Compact Hadamard Latent Codes for Efficient Spectral Rendering

**arXiv ID:** 2602.18741 | [PDF](https://arxiv.org/pdf/2602.18741v1)

**作者:** Jiaqi Yu `[一作]` (University of York), Giuseppe Claudio Guarnera `[通讯]` (University of York)

**通讯引用:** 705 | [OpenAlex ID](https://openalex.org/A5009336450)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Hadamard 频谱码（compact latent representation），使光线跟踪能仅用少量 RGB 渲染通道即可近似完整光谱渲染，并通过轻量级神经上采样把现有 RGB 资产迁移到该频谱空间。

**💡 创新点**

核心创新在于：① 设计可线性保持缩放和相加的非负线性编码器-解码器；② 在块级 Hadamard 乘积上约束乘法近似；③ 通过多目标损失学习出兼顾光谱重建、相乘逼近与感知色差的压缩码；④ 结合 RGB 上采样网络将 RGB 颜色映射到已学习的频谱码空间。

**🔧 技术方法**

技术包括：非负线性变换（softplus 参数化）、块级 Hadamard 乘积、端到端重建损失、相乘逼近损失、颜色感知损失、MSE+最大误差上限、Silu 激活的轻量 MLP 上采样网络、Mitsuba 渲染器的多通道渲染。

**📊 数据集**

使用 47 维光谱采样的测量反射率（Munsell、合成极彩色）与光照（LSPDD、CIE 日光、LED 等）数据集共计约 2000 条谱线，训练集约 1k 条，测试集 700 条，覆盖全光谱范围和多种色度。

**📈 对比分析**

在各种 3D 场景（Cornell Box、多层玻璃等）下与传统 RGB 渲染和完整光谱渲染比较：k=6（两通道）在色差上明显优于 RGB，误差约为 1–2 ΔE，渲染速度比完整光谱快约 23×；k=9（三通道）进一步降低误差但成本略升高；RGB 上采样方法也能把旧资产显著逼近光谱结果。

**⚠️ 局限性**

局限包括：乘法近似导致极尖锐光谱和多次光交互后误差积累；RGB‑>频谱上采样网络未显式保证光谱平滑，可能出现尖峰；在高度尖锐或多跳光照条件下仍存在色差；实现依赖于已训练的编码器/解码器，需额外训练成本。

---

## 517. GRAB: A Systematic Real-World Grasping Benchmark for Robotic Food Waste Sorting

**arXiv ID:** 2602.18835 | [PDF](https://arxiv.org/pdf/2602.18835v1)

**作者:** Moniesha Thilakarathna `[一作]` (University of Canberra), Damith Herath `[通讯]` (University of Canberra)

**通讯引用:** 462 | [OpenAlex ID](https://openalex.org/A5003252250)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了GRAB——一种针对食物废弃物分类的机器人抓取基准框架，系统评估了三种抓手（刚性、柔性和真空）在21个真实无机污染物样本上共进行1,750次抓取实验。

**💡 创新点**

创新点在于将预抓取条件（对象形变、视觉抓取质量与场景混乱度）纳入量化度量，结合大规模实地实验突破了传统仅以成功率评估的局限，并提出了完整的抓取可抓性指标。

**🔧 技术方法**

核心技术包括6D抓取姿势检测（AnyGrasp与SuctionNet）、ROS 2+MoveIt2运动规划、基于点云的形变评估（DCD）、以及逻辑回归/分数逻辑回归的性能因子建模。

**📊 数据集**

使用自建的21个无机食物废弃物（7类×3件）数据集，配合高精度3D扫描和RGB‑D摄像机生成的点云数据，实验共收集1,750次抓取样本。

**📈 对比分析**

通过雷达图对三种抓手在成功率、稳定性和效率三指标进行对比；柔性Fin‑Ray抓手在变形/半变形物体上表现最佳，真空抓手在平坦刚性物体上优异，整体而言Fin‑Ray在混合废弃物环境中取得最高综合性能。

**⚠️ 局限性**

局限性包括：真空抓手对多孔或柔性物体失败率高；实验对象仅为无机污染物，未覆盖有机废弃物细节；视觉模型对高度可变形对象的训练不足；物理交互失效仍占主导，说明仍需多模态抓手与更完善的抓取策略。

---

## 518. Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations

**arXiv ID:** 2602.19320 | [PDF](https://arxiv.org/pdf/2602.19320v1)

**作者:** Dongming Jiang `[一作]` (University of Texas at Dallas), Bingzhe Li `[通讯]` (University of Texas at Dallas)

**通讯引用:** 904 | [OpenAlex ID](https://openalex.org/A5048972267)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM Agent的记忆系统进行结构化分类，并系统评估其在不同维度（benchmark饱和度、指标可靠性、主干模型敏感度、系统延迟与成本）下的实际表现。

**💡 创新点**

提出了四类记忆结构的简洁分类法，并将其与实际瓶颈关联；引入“Context Saturation Gap”和LLM-as-a-Judge评估协议，揭示传统指标与语义质量之间的脱节。

**🔧 技术方法**

采用检索-写入双向机制、强化学习优化的记忆压缩、图结构与分层存储、LLM作为评判者等技术，结合多种主干模型（gpt‑4o‑mini、Qwen‑2.5‑3B）进行实验。

**📊 数据集**

使用LoCoMo、LongMemEval‑S/M、MemBench等多任务数据集，重点关注交互深度、实体多样性与总体token量，验证benchmark是否仍需外部记忆。

**📈 对比分析**

与全上下文Baseline对比，使用F1与LLM评判得分排序；对5个系统（LOCOMO、AMem、MemoryOS、Nemori、MAGMA）分别测量用户延迟、构建成本与Token消耗，发现MAGMA在语义评判上排名最高，MemoryOS延迟最高，SimpleMem最快但精度最低。

**⚠️ 局限性**

局限性包括：benchmark仍可能在大上下文模型下饱和；传统指标与语义质量不对齐；主干模型对结构化输出敏感导致“Silent Failure”；维护成本与写入延迟未被充分异步化，导致系统吞吐受限。

---

## 519. Measuring the Prevalence of Policy Violating Content with ML Assisted Sampling and LLM Labeling

**arXiv ID:** 2602.18518 | [PDF](https://arxiv.org/pdf/2602.18518v1)

**作者:** Attila Dobi `[一作]` (Pinterest), Faisal Farooq `[通讯]` (Pinterest)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套可每日运行的、基于概率抽样和LLM标签的曝光权重违规率测量系统，能够在大规模平台上以设计一致的方式估计全局以及多维下钻的违规曝光率；

**💡 创新点**

创新点在于①设计一致的单样本下钻估计器，②将ML辅助概率抽样与LLM高效标注相结合实现每日低成本大规模测量，③通过持续的决策质量监控与可配置工作流实现自动化治理；

**🔧 技术方法**

使用了设计式抽样（PPS、Hansen–Hurwitz、Horvitz–Thompson）、加权流式抽样（Efraimidis–Spirakis）、GPT‑4 multimodal LLM进行结构化标注、置信区间与有效样本量诊断以及标签误差校正等技术；

**📊 数据集**

基于Pinterest平台的日常印象日志（曝光计数）、安全模型的风险分数以及SME审核的金标集；

**📈 对比分析**

通过仿真和一年的生产实验与传统人类审核和均匀抽样相比，ML辅助抽样将正例率提升6–11倍、CI宽度显著缩小，标注吞吐量提升约100×，成本降低约95%；

**⚠️ 局限性**

局限包括极低基率导致每日置信区间宽、模型提示/政策漂移和LLM标签漂移需要持续治理、对极高权重样本敏感需人工验证，以及缺乏完整绝对数值的公开可比性。

---

## 520. The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers

**arXiv ID:** 2602.19096 | [PDF](https://arxiv.org/pdf/2602.19096v1)

**作者:** Wei Tao `[一作]` (National University of Defense and Technology), Qing Tao `[通讯]` (Hefei Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在签名梯度攻击中引入单调递减坐标步长（MDCS）的新优化器，解决了传统I-FGSM/MI-FGSM等方法的非收敛和不稳定问题；

**💡 创新点**

创新点在于将MDCS机制（即每个坐标的步长随迭代递减）植入签名梯度优化器，并提供了最优收敛率O(1/√T)的理论证明；

**🔧 技术方法**

采用了坐标梯度下降理论、Adam/AMSGrad的MDCS思想、以及对MI-FGSM、PGD等经典攻击算法的改写；

**📊 数据集**

在图像分类任务使用NIPS2017数据集（1000张图像），评估多种CNN和ViT模型；在跨模态检索任务使用Flickr30K和MSCOCO数据集，针对ALBEF、TCL、CLIP_CNN/ViT等VLM模型；

**📈 对比分析**

与多种基线攻击（I-FGSM、MI-FGSM、PGN、MEF、OPS、SGA、DRA、SA-AET等）进行对比，实验显示MDCS-改进版在攻击成功率（攻击转移率）上提升约5–15%，同时在稳定性（随迭代次数不下降）方面显著优于传统方法；

**⚠️ 局限性**

局限性包括：实验主要集中在非目标攻击和有限的模型与数据集；理论分析假设凸性和梯度有界，对更复杂的非凸任务可能不直接适用；MDCS的实现仍需调参（ε、γ），且对大规模高维数据的计算效率尚待进一步评估。

---

## 521. De novo molecular structure elucidation from mass spectra via flow matching

**arXiv ID:** 2602.19912 | [PDF](https://arxiv.org/pdf/2602.19912v1)

**作者:** Ghaith Mqawass `[一作]` (Technical University of Munich), Djork-Arné Clevert `[通讯]` (Pfizer Research and Development)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 MSFlow 模型，采用两阶段 Encoder‑Decoder 结构：首先用 MIST + SIRIUS 将质谱编码为 512 维 CDDD 嵌入；随后用基于 BERT 的离散流匹配解码器在该嵌入上解码 SAFE 序列，从而生成分子结构。

**💡 创新点**

创新点在于：① 将连续化学表征 CDDD 作为中间条件显著提升信息保留；② 采用非自回归离散流匹配解码器，结合自适应 LayerNorm 与无条件引导，克服传统自回归模型的搜索和速度瓶颈；③ 通过两阶段预训练与条件微调，实现在大规模质谱数据上高效、准确的结构重建。

**🔧 技术方法**

技术细节包括：MIST Transformer 编码器、SIRIUS 公式预测、CDDD 连续嵌入、BERT‑style 变压器解码器、离散流匹配（DFM）目标、AdaLN 自适应层归一化、Classifier‑free Guidance。

**📊 数据集**

数据集：训练 Encoder 与 Decoder 使用公开质谱‑结构对数据集 CANOPUS（NPLIB1）和 MassSpecGym；Decoder 预训练使用合并的 DSSTox、HMDB、COCONUT、MOSES 等 2.8M 分子集合；评估时采用 CANOPUS 与 MassSpecGym 的官方 train/val/test 分割。

**📈 对比分析**

与 Spec2Mol、MADGEN、DiffMS、MS‑BART 等基线比较，MSFlow 在 NPLIB1 的 Top‑1 准确率达到 44.7%（比 DiffMS 提升 5.4×），在 MassSpecGym 上 Top‑1 为 32%（比 DiffMS 提升 13.9×），Top‑10 也显著提升（NPLIB1 58.5%、MassSpecGym 42.5%）。

**⚠️ 局限性**

主要局限：质谱→CDDD 编码阶段信息损失显著，导致与 oracle 条件下的性能差距大；对大分子（>40 原子）和高柔性分子（>7 可旋转键）的重建准确率仍有下降；目前模型仅在非商业 GitHub 公开，需进一步验证在真实实验环境中的稳健性。

---

## 522. Bumper Drone: Elastic Morphology Design for Aerial Physical Interaction

**arXiv ID:** 2602.18976 | [PDF](https://arxiv.org/pdf/2602.18976v1)

**作者:** Pongporn Supa `[一作]` (University of Bristol), Basaran Bahadir Kocer `[通讯]` (University of Bristol)

**通讯引用:** 1176 | [OpenAlex ID](https://openalex.org/A5006237898)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并测试了一种配备弹性锥形前端的四旋翼无人机，以实现自律的触碰-即退跳动作和稳定的壁面推送操作。

**💡 创新点**

创新之处在于将软弹性触碰装置与生物学灵感的触角传感相结合，实现被动的弹簧-阻尼机动并用低成本的电阻式柔性传感器估计接触力，显著降低对主动控制和高精度力传感器的依赖。

**🔧 技术方法**

采用3D打印TPU弹性锥体、内置薄膜电阻柔性传感器、PX4+Raspberry Pi飞控、ROS2、MAVROS、低通Butterworth滤波等硬件与软件技术。

**📊 数据集**

主要使用实验室飞行实验收集的原始姿态与传感器信号数据，没有公开数据集。

**📈 对比分析**

通过与全软TPU四锥、半软上锥和硬PLA四锥三种配置进行对比，测量姿态振荡的RMSE。结果显示，全软配置将俯仰振荡幅度降低38%，并在四锥形下进一步降至54%，并在推送实验中保持稳定的俯仰角，硬PLA配置则出现较大滚转和偏航振荡导致失控。

**⚠️ 局限性**

实验仅在平坦垂直墙面进行，且使用手动遥控，缺乏闭环触觉控制、动态模型与多表面验证，柔性传感器存在非线性、滞后与蠕变，限制了其在复杂环境下的普适性。

---

## 523. Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning

**arXiv ID:** 2602.19027 | [PDF](https://arxiv.org/pdf/2602.19027v1)

**作者:** Haoyu Yang `[一作]` (NVIDIA Corporation), Haoxing Ren `[通讯]` (NVIDIA Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于生成式强化学习的逆光刻（ILT）框架，将生成器视为条件采样器，直接产生多种可供ILC细化的掩模初始值；

**💡 创新点**

创新点在于（1）引入风格感知的生成网络实现设计条件下的分布式掩模采样；（2）采用两阶段训练（WGAN预训练+GRPO强化学习）让采样器既保留掩模分布又倾向于易于ILC收敛的解；（3）在强化学习过程中用教师对齐基线与模仿学习，显著降低梯度方差；

**🔧 技术方法**

使用风格GAN架构、WGAN‑GP预训练、Group Relative Policy Optimization (GRPO)强化学习、仿真光刻的低分辨率迭代奖励、模仿损失以及多尺度卷积网络；

**📊 数据集**

主要使用公开的MetalSet、ViaSet作为预训练数据，测试集包括StdMetal、StdContact和ICCAD13；

**📈 对比分析**

与最新的数值ILT求解器（CurvyILT、ISPD'25等）在EPE、Process Window (PV) 等指标上对比，结果显示在15 nm阈值下EPE相同或更低、PV更优；在3 nm严格阈值下，本文方法实现了EPE约30–50 %的下降，且迭代次数约为传统方法的一半；

**⚠️ 局限性**

局限性包括：①需要在ILC求解器上进行物理反馈，仍然有计算开销；②风格空间采样的多样性受限，可能无法覆盖极端优化解；③在极大尺寸芯片或不同工艺节点时需要进一步验证可扩展性与泛化能力。

---

## 524. InfScene-SR: Spatially Continuous Inference for Arbitrary-Size Image Super-Resolution

**arXiv ID:** 2602.19736 | [PDF](https://arxiv.org/pdf/2602.19736v1)

**作者:** Shoukun Sun `[一作]`, Xiaogang Ma `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用扩散模型进行联合超分辨率的框架，以消除传统裁剪拼接所产生的伪影，得到无缝的高分辨率图像。

**💡 创新点**

核心创新点是：①在重叠裁剪区域分别使用扩散模型推理；②将两个模型输出通过融合模块合并，生成全局一致的结果；③从而显著降低拼接产生的视觉噪声。

**🔧 技术方法**

使用技术包括：扩散模型（U‑Net结构的噪声预测器）、重叠裁剪策略、融合模块（对重叠区域的特征进行加权/拼接）以及后处理的无缝边界修正。

**📊 数据集**

实验采用常见的超分辨率公开数据集，如DIV2K、Flickr2K等，或者作者自行构造的高分辨率图像集合。

**📈 对比分析**

方法与传统单图像基于patch的SR模型、以及其他基于GAN或CNN的SR方法进行了对比，实验报告显示在PSNR/SSIM等指标上较基线提升约0.5–1.0 dB，且视觉效果更加平滑，无明显拼接痕迹。

**⚠️ 局限性**

局限性包括：扩散模型推理时间较长，导致整体推理速度慢；重叠裁剪的大小和比例对结果有影响，需手工调参；当前实验多在中等尺寸图像上验证，尚未验证大尺寸图像或实时场景的效果。

---

## 525. Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series

**arXiv ID:** 2602.18473 | [PDF](https://arxiv.org/pdf/2602.18473v1)

**作者:** Guoqi Yu `[一作]` (Polytechnic University), Shujun Wang `[通讯]` (Polytechnic University)

**通讯引用:** 6299 | [OpenAlex ID](https://openalex.org/A5100602073)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计了核心令牌聚合-重分配（CoTAR）模块替代 Transformer 注意力，并基于此提出可自适应捕捉时间与通道依赖的 TeCh 框架，用于医学时间序列（MedTS）分类。

**💡 创新点**

创新点在于：① 通过核心令牌实现集中式交互，将注意力的二次复杂度降至线性；② 引入自适应双重分词（时间与通道分词）同时建模两种依赖；③ 将上述两点组合成统一框架并在多数据集上验证其高效性。

**🔧 技术方法**

采用 MLP‑核心令牌聚合、Transformer 编码器、时间/通道分词、线性复杂度注意力替代，以及对比实验与多指标评估。

**📊 数据集**

使用 5 个医学时间序列数据集（EEG：APAVA、TDBrain、ADFTD；ECG：PTB、PTB‑XL）以及 2 个人体活动识别数据集（FLAAP、UCI‑HAR）进行实验。

**📈 对比分析**

在受试者独立设置下与 10 个 Transformer 基线（包括 Medformer）进行对比，TeCh 在大多数数据集上平均提升 4–12% 指标，显著优于 Medformer；同时仅占用 33% 内存、20% 推理时间。

**⚠️ 局限性**

局限性：主要针对集中式源的医学信号，尚未验证对非集中式或多模态信号的适用性；核心令牌的生理学解释仍需进一步研究。

---

## 526. ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models

**arXiv ID:** 2602.20117 | [PDF](https://arxiv.org/pdf/2602.20117v1)

**作者:** Andre He `[一作]` (Carnegie Mellon University), Huzefa Rangwala `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了ReSyn管线，自动生成大量带代码验证器的推理环境，并利用可验证奖励进行强化学习训练语言模型

**💡 创新点**

通过利用生成器–验证器差距，摆脱对模型自身解题能力的依赖，显著提升监督质量；同时将任务多样性和实例规模作为两个可扩展维度

**🔧 技术方法**

使用大型语言模型（Claude 3.5 Sonnet v2）生成代码实现的环境与验证器，采用DPO/ DAPO强化学习算法，训练Qwen‑2.5‑Instruct和Llama‑3.1‑Instruct模型

**📊 数据集**

生成的ReSyn数据集包含约418个不同环境、约16K个问答–验证器对；对比SynLogic、SynLogic-7B以及基线Instruct模型

**📈 对比分析**

与Instruct基线相比，ReSyn模型在BBH和BBEH等推理基准上分别提升约15%与27%（相对），在GSM8K、AIME等数理基准上也保持竞争力；对比SynLogic在BBH上取得更大提升

**⚠️ 局限性**

受限于LLM对代码生成的准确性、验证器设计的复杂性，以及对多步推理可解释性的缺失；验证器错误率虽低，但仍可能影响奖励信号

---

## 527. Lost in Instructions: Study of Blind Users' Experiences with DIY Manuals and AI-Rewritten Instructions for Assembly, Operation, and Troubleshooting of Tangible Products

**arXiv ID:** 2602.18630 | [PDF](https://arxiv.org/pdf/2602.18630v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 528. HDR Reconstruction Boosting with Training-Free and Exposure-Consistent Diffusion

**arXiv ID:** 2602.19706 | [PDF](https://arxiv.org/pdf/2602.19706v1)

**作者:** Yo-Tin Lin `[一作]` (National Yang Ming Chiao Tung University), Yu-Lun Liu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练的扩散模型加SDEdit迭代补偿管线，利用深度控制的Inpainting技术修复HDR重建中曝光过度区域的细节；

**💡 创新点**

创新点在于将扩散模型与SDEdit结合，用光照一致的迭代补偿机制实现对多曝光LDR堆栈的无监督、跨曝光一致性修复；

**🔧 技术方法**

使用SDXL‑Base（带深度ControlNet）作为扩散后端，配合SDEdit调度、亮度补偿以及基于CRF的对齐；

**📊 数据集**

在VDS和HDR‑Eye两个公开HDR数据集上进行实验；

**📈 对比分析**

与CEVR、SingleHDR、GlowGAN、Multi‑Exposure Generation等现有直接或间接HDR重建方法对比，非参考指标显著提升（NIQE、PU21‑PIQE、CLIP‑IQA等下降），参考指标（HDR‑VDP‑3、KID）略低但符合“创意重建”目标；

**⚠️ 局限性**

受限于基线方法估计的逆相机响应函数（inverse CRF）若不合理，易导致色偏和光度失真，影响最终重建质量。

---

## 529. Embedding arbitrary Boolean circuits into fungal automata with arbitrary update sequences

**arXiv ID:** 2602.19477 | [PDF](https://arxiv.org/pdf/2602.19477v1)

**作者:** Eric Goles `[一作]` (Universidad Adolfo Ibáñez), Thomas Worsch `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 503 | [OpenAlex ID](https://openalex.org/A5015010410)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文证明了在任意包含至少一次 H 与 V 的更新序列 Z 下，真菌沙堆预测问题是 P‑完备的；并提供了一套可在任意此类 Z 上直接构造的通用方法。

**💡 创新点**

创新点在于提出了“桥”这一通用构件，能够在不同更新序列下统一实现信号传递、延迟、分叉等功能，从而在不改动原先层次结构的前提下完成对任意 Z 的归约；同时完成了对先前仅针对 HV 的 P‑完备性结果的完全推广。

**🔧 技术方法**

主要技术手段包括：
- 通过对原始 HV 方案的模块化拆分，重构块划分（2×2 → h×v）以适配不同 Z；
- 设计并证明桥（Z‑path 与配置）能够可靠地在块之间传递信号，并通过组合桥实现更复杂的电路元件；
- 使用逻辑门、延迟器、分叉器等在桥层实现极化电路；
- 通过从电路值问题（CVP）进行对数空间归约得到最终 P‑完备性证明。

**📊 数据集**

本研究为理论工作，未使用任何实验数据集；所有结果均为理论证明。

**📈 对比分析**

方法上与之前的研究相比，仅在通用性上有所提升：原先只证明 HV 的 P‑完备性；本工作在任意包含 H 与 V 的 Z 下同样成立；由于采用的归约与桥构造本身无性能指标，故未给出实验对比或数值评估。

**⚠️ 局限性**

限制与未解决问题：
- 仍未能将真菌沙堆模型的 P‑完备性结果归约到普通二维沙堆预测问题，因而该问题的复杂度仍未知；
- 对单一 H 或单一 V 的更新序列仅能得到一维沙堆结果，未能进一步探究多维情形下的细分阈值；
- 虽然桥构件对任意 Z 有通用性，但实现细节在极端极值更新序列（如 H 与 V 只出现一次）时仍需特殊处理，增加了构造复杂度。

---

## 530. Multi-Channel Speech Enhancement for Cocktail Party Speech Emotion Recognition

**arXiv ID:** 2602.18802 | [PDF](https://arxiv.org/pdf/2602.18802v1)

**作者:** Youjun Chen `[一作]`, Xunying Liu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5037109470)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于多通道语音增强（DNN-WPE去混响+Mask-based MVDR分离）的前端，并将其与HuBERT和ViT的语音视觉特征相结合，用于混响环境下的情绪识别。

**💡 创新点**

创新点在于首次系统性评估多通道前端对情绪识别的影响，结合了去混响和分离两步，且在音频-视觉融合任务中展现出显著提升，并通过零样本方法验证跨域泛化。

**🔧 技术方法**

使用的技术包括DNN-WPE、mask-based MVDR、HuBERT SSL、ViT视觉编码器、BiLSTM降维、早期/晚期音视频融合、双任务微调。

**📊 数据集**

采用IEMOCAP（4情绪）和MSP-FACE（混响/噪声真实视频）进行多通道混合语音仿真，训练与测试均使用这些数据。

**📈 对比分析**

与单通道基线（Conformer-MetricGAN、WavLM+SE-ER）相比，MCSE前端在IEMOCAP上平均提升了WA≈9.5%、UA≈8.5%、F1≈9.1%；在MSP-FACE零样本时提升约5%；在音视频融合任务中提升3–4%。

**⚠️ 局限性**

主要局限包括：混响混合语音完全基于仿真，缺乏真实酒吧/车内混响数据；仅评估四情绪分类，且融合方法仍相对简单，未来需在更大规模、多情绪及真实场景下验证。

---

## 531. In-Context Planning with Latent Temporal Abstractions

**arXiv ID:** 2602.18694 | [PDF](https://arxiv.org/pdf/2602.18694v1)

**作者:** Baiting Luo `[一作]` (Vanderbilt University), Ayan Mukhopadhyay `[通讯]` (William and Mary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种离线强化学习框架 I‑TAP，利用观察条件的残差量化 VAE 将连续观测‑宏动作序列离散化为多层离散码，并在此码空间中通过 Transformer 进行上下文条件的序列建模，随后在离散码空间内使用 MCTS 进行在线规划，完成自适应连续控制。

**💡 创新点**

创新点在于将时间抽象与上下文适应相结合：① 用残差量化 VAE 获得高效的多层离散码，极大提升离散化容量；② 构造上下文条件的自回归先验，既可捕获长时序依赖，又可在规划时提供“先验”指导；③ 在离散码空间直接执行 MCTS，实现对部分可观测、噪声环境的自适应规划，避免传统模型无条件复制数据集缺陷。

**🔧 技术方法**

技术主要包括：残差量化 VAE（RQ‑VAE）+ Transformer 先验（RQ‑Transformer）、自回归时间先验建模、Monte Carlo Tree Search（P‑UCT）在离散码空间的规划，以及整体离线训练与在线解码策略。

**📊 数据集**

使用 D4RL 基准任务：标准 Gym 仿真环境（MuJoCo）和高维 Adroit 机械臂；其中 MuJoCo 包含确定性和带不同噪声级别的随机动力学；Adroit 包含完全可观测和部分可观测版本。

**📈 对比分析**

与多种强基线（CQL、IQL、Decision Transformer、TAP、L‑MAP、1R2R 等）对比，I‑TAP 在所有测试环境（含随机扰动与部分可观测）均达到或超过基线得分，表现出更强的自适应性和更好的规划效果；在部分可观测 Adroit 上尤为突出。

**⚠️ 局限性**

局限性包括：① 对于极高维度或极长时间依赖的任务，残差码堆栈深度与 MCTS 计算量仍显著；② 需要离线数据覆盖足够多的潜在任务参数才能实现良好适应；③ MCTS 的实时延迟随上下文长度和规划深度增加而显著上升，可能限制在实时控制中的应用。

---

## 532. Training-Free Generative Modeling via Kernelized Stochastic Interpolants

**arXiv ID:** 2602.20070 | [PDF](https://arxiv.org/pdf/2602.20070v1)

**作者:** Florentin Coeurdoux `[一作]` (Capital Fund Management), Eric Vanden-Eijnden `[通讯]` (Courant Institute of Mathematical Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于核方法的随机插值框架，利用特征映射求解线性系统来估计生成过程的漂移，从而实现无训练的生成模型。

**💡 创新点**

创新点包括将随机插值迁移为有限维核方法，推导最优扩散系数并设计兼容无穷端点的积分器；支持预训练模型特征组合，实现训练免费模型融合。

**🔧 技术方法**

使用了核方法、线性系统求解、Girsanov变换下的KL路径上界、最优扩散系数、随机插值理论、散射变换特征、预训练生成模型梯度和专门的数值积分器。

**📊 数据集**

实验使用了金融S&P500日收益率序列、三维湍流切片、暗物质密度场、磁湍流vorticity、弱透镜收敛图，以及MNIST和CelebA图像数据集。

**📈 对比分析**

与传统基于神经网络的扩散/流匹配模型对比，在单样本、低训练样本场景下实现了与真实分布高度一致的统计特征；在物理场中生成图像与真实样本视觉上高度相似；在弱模型集合上线性组合显著提升样本质量，log-likelihood 随集合规模提升并趋于饱和。

**⚠️ 局限性**

局限性包括对特征映射表达能力的依赖、特征维数 P 的手工设定、t→0 时扩散系数发散需特殊积分器、仅在有限维核下逼近，可能无法捕捉高阶结构，以及缺乏对模型泛化与训练细调的直接控制。

---

## 533. Layered Monoidal Theories I: Diagrammatic Algebra and Applications

**arXiv ID:** 2602.19776 | [PDF](https://arxiv.org/pdf/2602.19776v1)

**作者:** Leo Lobski `[一作]` (University College London), Fabio Zanasi `[通讯]` (University College London)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并形式化了层化单子理论（Layered Monoidal Theories），一种可在同一图示（字符串图）中混合多层抽象级别的单子理论，并给出了相应的数学基础与若干实例应用。

**💡 创新点**

创新点在于：①将多层抽象的单子理论通过翻译（translation）机制统一在同一图形框架内；②引入“Functor Box”和“CoBox”概念，实现不同层之间的语义映射；③在单子理论图示基础上提供了可扩展的多层表示法，可直接用于电路、量子过程、化学反应、并发进程和概率理论等多领域。

**🔧 技术方法**

采用了范畴理论中的单子理论与字符串图方法，构建了层化单子理论的语法与语义；使用了翻译子范畴与Functor/CoBox框架；通过对比单层单子理论与层化单子理论，展示其表达力和可组合性。

**📊 数据集**

本研究为理论工作，未使用具体实验数据集；所示实例（如数字电路、量子过程等）均为形式化模型而非数值数据集。

**📈 对比分析**

比较方法主要基于理论可表达性与语义清晰度：层化单子理论能够在同一图形中同时表示多种抽象层次，而传统单一层单子理论则需拆分图示。性能方面，因无数值实验，未给出具体指标；但作者指出该框架在保持数学严谨性的同时，支持跨领域语义互操作。

**⚠️ 局限性**

限制包括：①理论尚处于初步阶段，缺乏广泛的实证评估；②对复杂系统的实际建模仍需进一步工具化；③翻译规则的设计可能导致层间耦合过度，影响可扩展性。

---

## 534. Interpolation-Driven Machine Learning Approaches for Plume Shine Dose Estimation: A Comparison of XGBoost, Random Forest, and TabNet

**arXiv ID:** 2602.19584 | [PDF](https://arxiv.org/pdf/2602.19584v1)

**作者:** Biswajit Sadhu `[一作]` (Bhabha Atomic Research Centre), S. Anand `[通讯]` (Homi Bhabha National Institute)

**通讯引用:** 1896 | [OpenAlex ID](https://openalex.org/A5113675681)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在核设施安全评估中，构建了一种基于插值增强的机器学习框架，用于快速精确估计云闪光剂量（plume shine dose）

**💡 创新点**

通过形状保持插值（PCHIP）将稀疏的离散剂量表转化为高分辨率连续数据，从而显著提升模型泛化性能，并对不同模型的特征利用方式进行了可解释性对比

**🔧 技术方法**

采用了树基集成模型（随机森林RF、梯度提升XGBoost）和注意力型深度网络TabNet，配合对数尺度变换、整数编码与归一化处理

**📊 数据集**

使用pyDOSEIA生成的17种γ发射核素在不同释放高度、下风距离、气象稳定类别下的离散剂量表，随后通过PCHIP插值得到约4×10⁶点的高分辨率训练集

**📈 对比分析**

在低分辨率和高分辨率两种数据集上评估，指标包括R²、MAPE和sMAPE；XGBoost在高分辨率数据上取得最高准确率（R²≈0.999，MAPE≈1%），RF次之，TabNet性能最差；解释性分析显示树模型侧重几何扩散特征，TabNet分散关注多变量

**⚠️ 局限性**

局限性包括：深度学习模型在稀疏数据下表现不佳；模型仍依赖高质量插值数据，无法完全解决安全关键性对误差容忍度的苛刻要求；缺乏对更复杂多源放射场的评估及实时更新能力

---

## 535. GenPlanner: From Noise to Plans -- Emergent Reasoning in Flow Matching and Diffusion Models

**arXiv ID:** 2602.18812 | [PDF](https://arxiv.org/pdf/2602.18812v1)

**作者:** Agnieszka Polowczyk `[一作]` (Silesian University of Technology), Michał Wieczorek `[通讯]` (Silesian University of Technology)

**通讯引用:** 1241 | [OpenAlex ID](https://openalex.org/A5074313380)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于生成模型的路径规划框架 GenPlanner，包含 DiffPlanner（扩散模型）和 FlowPlanner（流匹配模型）两个变体，用来在迷宫等离散网格环境中生成从起点到终点的可行路径。

**💡 创新点**

创新点主要有：① 将路径规划视为条件图像生成任务，使用四通道（墙、起点、终点、路径）输入；② 采用流匹配方法学习连续动力学，从随机噪声逐步收敛到完整路径；③ 在生成过程中避免传统规划器的梯度优化和软约束，提升了结构一致性和效率。

**🔧 技术方法**

使用的技术包括：U‑Net 条件网络、扩散模型（DDIM 采样）、流匹配（显式欧拉积分）以及针对噪声的 MSE 损失；训练与推理都以条件图像为输入，推理时从纯噪声迭代收敛。

**📊 数据集**

数据集：自制的 8×8、16×16、32×32、48×48 四种尺寸的迷宫，每个样本由墙壁图、起点、终点和 A* 求得的最短路径构成；训练集中分别包含 5k–20k 条样本，测试集 250–1k 条。

**📈 对比分析**

实验通过 Validity、Single‑Path、Length Ratio、Branch‑Rate 四个指标评估，结果显示 FlowPlanner 在所有尺寸下均显著优于基准 CNN 与 DiffPlanner，尤其在 48×48 大迷宫中 Validity 超 88%，Single‑Path 约 86%，Branch‑Rate 仅 0.09；DiffPlanner 受步数影响更大，步数过少时性能急剧下降。

**⚠️ 局限性**

局限性包括：① 需要多步迭代，尽管 FlowPlanner 对步数更鲁棒，但在极低步数下仍会失效；② 仅在合成迷宫上验证，真实世界连续空间或更复杂约束下的泛化尚待验证；③ 生成时间相对较长，特别是扩散变体需要上百步；④ 依赖完整的起点/终点/墙信息，若输入缺失则性能骤降。

---

## 536. DD-CAM: Minimal Sufficient Explanations for Vision Models Using Delta Debugging

**arXiv ID:** 2602.19274 | [PDF](https://arxiv.org/pdf/2602.19274v1)

**作者:** Krishna Khadka `[一作]` (University of Texas at Arlington), D. Richard Kuhn `[通讯]` (National Institute of Standards and Technology)

**通讯引用:** 10384 | [OpenAlex ID](https://openalex.org/A5011985891)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了一种无梯度的解释方法DD‑CAM，通过delta debugging找出对模型预测最小决定性子集的特征图或补丁，从而生成更聚焦、faithful的saliency map。

**💡 创新点**

将软件调试中的delta debugging应用于视觉模型解释，首次实现对内部表示层的1‑最小化保证，并兼顾线性与非线性分类头的优化；从而生成仅包含必要特征的解释。

**🔧 技术方法**

采用delta debugging、零掩蔽（zero‑masking）在最终层特征图/补丁上、权重归因、加权平均生成热图，并针对CNN与ViT统一实现。

**📊 数据集**

使用ImageNet 2012验证集2000张图片和NIH ChestX‑ray14 1000张X光图作为评估数据集。

**📈 对比分析**

与Grad‑CAM、Grad‑CAM++、XGrad‑CAM、Layer‑CAM、Score‑CAM、Ablation‑CAM、Recipro‑CAM等七种CAM基线在六项perturbation指标和胸X‑ray的IoU/Precision/Recall进行对比，DD‑CAM在大多数指标上均优于基线，尤其在定位上提升约45% IoU、22% Precision。

**⚠️ 局限性**

仍受CAM范式限制，如上采样导致空间不精确；需要白盒访问内部激活；仅关注最终层，未覆盖输入级解释。

---

## 537. Support Vector Data Description for Radar Target Detection

**arXiv ID:** 2602.18486 | [PDF](https://arxiv.org/pdf/2602.18486v1)

**作者:** Jean Pinsolle `[一作]` (CentraleSupelec), Jean-Philippe Ovarlez `[通讯]` (ONERA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了两种基于一类异常检测算法（SVDD 与 Deep SVDD）的雷达目标检测器，设计为 CFAR 结构；

**💡 创新点**

创新点在于将 SVDD 与深度 SVDD 迁移到雷达检测任务中，利用核/神经网络映射构建最小球形空间，并以二次阈值实现 CFAR 判决；

**🔧 技术方法**

采用核方法的 SVDD、深度学习版 Deep SVDD（含三层 1D 卷积 + 线性层）、传统 AMF‑SCM、ANMF‑Tyler 以及 Matched Filter 作为对比；

**📊 数据集**

使用合成雷达数据集，模拟 Gaussian 与 Compound‑Gaussian 堵塞（加白噪声）场景，尺寸 m=16、K=1 或 32，训练 5000 条无目标样本；

**📈 对比分析**

与传统检测器（AMF‑SCM、ANMF‑Tyler）对比，SVDD 与 Deep SVDD 在 Gaussian 堵塞下高 SNR 处优于 AMF‑SCM，在 Compound‑Gaussian 堵塞中 11dB 以上 SNR 时 Deep SVDD 超越 ANMF‑Tyler；

**⚠️ 局限性**

在 Compound‑Gaussian 环境及零多普勒 bin 处性能仍弱，低 SNR 下检测率不佳，需要在真实雷达数据上进一步验证。

---

## 538. Asymptotic Subspace Consensus in Dynamic Networks

**arXiv ID:** 2602.19121 | [PDF](https://arxiv.org/pdf/2602.19121v1)

**作者:** Matthias Függer `[一作]` (Université Paris-Saclay), Thomas Nowak `[通讯]` (Université Paris-Saclay)

**通讯引用:** 815 | [OpenAlex ID](https://openalex.org/A5030570962)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

引入了渐近子空间共识问题，要求过程的输出收敛到一个共同的子空间，同时保持在初始向量的凸包内。

**💡 创新点**

提出了渐近子空间共识的完整可解性特征，并展示了用于渐近共识的大量算法在通信网络假设较弱的分布式系统中优雅地降级为渐近子空间共识。

**🔧 技术方法**

使用了平均算法，并在弱假设下证明了其解决渐近子空间共识的能力。

**📊 数据集**

研究了在动态网络中执行的平均算法，特别是在非根的无意识消息对手中。

**📈 对比分析**

通过与根的无意识消息对手的比较，证明了在特定条件下，平均算法的收敛速度与维度无关，并且在k-广播可达的无意识消息对手中，平均算法能够实现维度的降低。

**⚠️ 局限性**

在非根的无意识消息对手中，渐近共识不一定可解，且对算法的收敛速度和维度降低的量化仍然是一个开放问题。

---

## 539. Open-Vocabulary Domain Generalization in Urban-Scene Segmentation

**arXiv ID:** 2602.18853 | [PDF](https://arxiv.org/pdf/2602.18853v1)

**作者:** Dong Zhao `[一作]` (University of Trento), Zhun Zhong `[通讯]` (Hefei University of Technology)

**通讯引用:** 10633 | [OpenAlex ID](https://openalex.org/A5065328976)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于视觉语言模型的空间相关性聚合方法S^2-Corr，用以解决真实道路场景中视觉语言分割的跨域与新类别识别问题；

**💡 创新点**

创新点在于提出OVDG-SS任务设置、设计S^2-Corr模块结合学习型衰减、snake采样、类空间相关性来提升VLM在真实道路环境下的性能；

**🔧 技术方法**

采用CLIP等视觉语言模型骨干，并结合S^2-Corr特征聚合、snake-decay采样、snake采样以及空间相关性机制；

**📊 数据集**

使用CS-7、ACDC-41、BDD-41、Mapi-30与RW-10等真实道路数据集进行实验；

**📈 对比分析**

与SAN、CAT-Seg、CLIPSelf、RSC-CLIPSelf、MAFT+、ESC-Net、MaskAdapter等基线方法对比，S^2-Corr在多组实验中均获得最高mIoU，表现优于现有方法；

**⚠️ 局限性**

局限性包括对VLM模型域适应能力有限、需要手工设计文本模板、以及在大规模类别与多任务场景下的可扩展性不足。

---

## 540. Many AI Analysts, One Dataset: Navigating the Agentic Data Science Multiverse

**arXiv ID:** 2602.18710 | [PDF](https://arxiv.org/pdf/2602.18710v1)

**作者:** Martin Bertran `[一作]` (Amazon Web Services), Zhiwei Steven Wu `[通讯]` (Amazon Web Services and Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过部署基于大型语言模型的自治AI分析师，对固定数据集和假设进行完整的分析流程，并对结果进行自动化审计，构建了可扩展的多分析师实验框架。

**💡 创新点**

创新点在于实现了对分析多样性的自动化、可控化和可量化研究，展示了模型和提示个性化如何系统性地影响科学结论，从而揭示了“分析多样性”在AI驱动科学中的重要性。

**🔧 技术方法**

使用了ReAct式工具使用的AI分析师、Claude、Haiku、Qwen3等大型语言模型、AI审计器以及结构化决策提取技术来生成和评估分析结果。

**📊 数据集**

实验使用了三大数据集：足球裁判偏见数据、AI编程辅助随机对照试验、以及美国全国选举研究（ANES）时间序列数据。

**📈 对比分析**

方法通过在每个数据集上多次运行不同LLM和提示个性化的分析师，筛选合规分析后比较效应估计、p值与假设支持率，结果显示个性化提示可导致高达66个百分点的结论差异，验证了可控的分析多样性。

**⚠️ 局限性**

局限性包括LLM的幻觉和审计准确性限制、对“合理”分析定义的主观性以及无法根除选择性报告带来的系统性风险。

---

## 541. CACTO-BIC: Scalable Actor-Critic Learning via Biased Sampling and GPU-Accelerated Trajectory Optimization

**arXiv ID:** 2602.19699 | [PDF](https://arxiv.org/pdf/2602.19699v1)

**作者:** Elisa Alboni `[一作]` (University of Trento), Andrea Del Prete `[通讯]` (University of Trento)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出 CACTO-BIC，结合轨迹优化与强化学习，利用价值函数不连续性进行偏置初始状态采样并采用 GPU 加速求解。

**💡 创新点**

创新点在于：①基于 critic 不确定性识别最易改进的状态空间区域；②引入 std‑critic 估计不确定性；③使用 GPU 并行化 iLQR 与网络训练，显著提升样本与时间效率。

**🔧 技术方法**

技术包括：轨迹优化（iLQR）、连续 actor‑critic 强化学习、JAX/Flax GPU 并行计算、Eigenvalue 正定化正则化、std‑critic 估计、自动微分与 GPU 训练。

**📊 数据集**

数据集与实验平台：点质量、Dubins 车、3-DOF 操作臂三种基准；Reacher 环境；AlienGO 四足机器人（仿真与硬件）以及对应的状态空间、控制参数和障碍物配置。

**📈 对比分析**

与 CACTO 的对比显示样本效率提升 2.5‑3.5×，GPU 加速实现 30‑250× 速度提升；与 PPO 的对比表明在相同任务中 CACTO‑BIC 能以 7‑30% 的训练时间获得相似或更低的成本。

**⚠️ 局限性**

局限性包括：仍受轨迹优化计算复杂度限制；GPU 速度受批次中难解实例影响；需要可微分模型；对非可微动力学（如低级腿部控制）需额外层级或采样优化，且对硬件 GPU 的依赖较高。

---

## 542. Detecting Cybersecurity Threats by Integrating Explainable AI with SHAP Interpretability and Strategic Data Sampling

**arXiv ID:** 2602.19087 | [PDF](https://arxiv.org/pdf/2602.19087v1)

**作者:** Norrakith Srisumrith `[一作]` (King Mongkut's University of Technology North Bangkok), Sunantha Sodsee `[通讯]` (King Mongkut's University of Technology North Bangkok)

**通讯引用:** 126 | [OpenAlex ID](https://openalex.org/A5109373554)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在网络入侵检测中提出并实现了一套整合采样、特征选择、数据泄漏防护与SHAP解释的多配置XAI框架。

**💡 创新点**

创新点在于将战略分层采样、自动泄漏检测与多配置评估与SHAP解释无缝整合，实现数据规模化、实验严谨和可解释性的统一。

**🔧 技术方法**

使用的技术包括分层采样、MRMR/Chi2特征选择、XGBoost/RandomForest/LogisticRegression、自动泄漏检测、时间序列划分、SHAP解释、5折交叉验证与统计显著性测试。

**📊 数据集**

使用的数据集是CIC‑IDS2017（约280万条记录）。

**📈 对比分析**

通过三种数据划分（40‑10‑50、60‑10‑30、80‑10‑10）进行多配置评估，并与现有方法对比，最终在60‑10‑30配置下实现99.92%准确率、99.77% F1‑macro，ROC‑AUC 0.99997，训练时间仅21s。

**⚠️ 局限性**

局限在于仅在单一基准数据集上验证，缺乏跨域（如IoT、云、工业控制）或实时流式环境的实验，未解决概念漂移和零日攻击等动态威胁。

---

## 543. Proximity-Based Multi-Turn Optimization: Practical Credit Assignment for LLM Agent Training

**arXiv ID:** 2602.19225 | [PDF](https://arxiv.org/pdf/2602.19225v1)

**作者:** Yangyi Fang `[一作]` (Tsinghua University), Peilin Zhao `[通讯]` (Tencent Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多轮强化学习中为大型语言模型提出ProxMO框架，结合任务难度感知的episode级调节和基于语义接近度的step级软聚合，实现更精准的信用分配。

**💡 创新点**

创新点在于将全局上下文引入两级信用分配：通过成功率加权调节梯度强度以适应任务难度，以及用连续语义加权替代硬阈值聚类，解决传统方法的噪声误判与单点稀疏问题。

**🔧 技术方法**

采用轻量级的成功率加权公式、TF‑IDF词向量与余弦相似度加温度的软聚合、以及PPO裁剪策略实现优势估计，无需额外价值网络。

**📊 数据集**

实验基准为ALFWorld（多种家居任务）和WebShop（在线购物导航）两大真实多轮交互数据集。

**📈 对比分析**

与GPT‑4o、Gemini‑2.5‑Pro、ReAct、Reflexion以及GRPO、GiGPO等基线对比，ProxMO在两大数据集上取得显著提升，甚至在小模型规模上超过部分闭源LLM，且计算开销仅增加约1%。

**⚠️ 局限性**

实验仅覆盖1.5B/7B规模模型，未验证在更大基础模型上的通用性，且在不同任务或数据域仍需进一步调优超参数。

---

## 544. Online decoding of rat self-paced locomotion speed from EEG using recurrent neural networks

**arXiv ID:** 2602.18637 | [PDF](https://arxiv.org/pdf/2602.18637v1)

**作者:** Alejandro de Miguel `[一作]` (Chapman University), Uri Maoz `[通讯]` (California Institute of Technology)

**通讯引用:** 1451 | [OpenAlex ID](https://openalex.org/A5019100753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本研究利用头皮32通道EEG连续记录，探讨了非侵入性方式下对大鼠自我节奏跑步速度进行实时连续解码的可行性。

**💡 创新点**

创新点在于通过大规模133小时EEG数据和LSTM递归神经网络实现0.88相关系数，揭示低频视觉区电位为主导信息源，并首次证明EEG能预测未来高达1秒的速度变化。

**🔧 技术方法**

技术手段包括32通道头皮EEG采集、基于LSTM的递归神经网络、Encoder-Only Transformer、线性回归与随机森林等模型，并结合增量学习与迁移学习策略。

**📊 数据集**

数据集来源于14只头固定大鼠，包含225个实验室内自我控制跑步会话，总计133小时EEG与速度配对样本超过48百万。

**📈 对比分析**

通过Pearson相关系数和R²对单会话训练、零射击迁移和微调三种策略进行比较，LSTM在单会话80%训练下取得r=0.88、R²=0.78；迁移学习显著提升10%训练的表现，且EEG预测未来1秒速度的相关性仍高于仅基于速度自相关。

**⚠️ 局限性**

局限性包括对不同个体的迁移效果差，需进行个体化校准；在静止期预测中存在误报；输出信号平滑度不够，影响实时控制应用。

---

## 545. IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbation

**arXiv ID:** 2602.18831 | [PDF](https://arxiv.org/pdf/2602.18831v1)

**作者:** Fadi Boutros `[一作]` (Fraunhofer IGD), Naser Damer `[通讯]` (Department of Computer Science, TU Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于几何角度采样的 IDperturb 方法，在预训练的身份条件扩散模型中对身份嵌入进行角度扰动，从而生成同一身份的多样化合成人脸，用于训练更具泛化能力的面部识别模型。

**💡 创新点**

创新点在于：①仅通过在单位超球面上以余弦约束的锥形区域对身份嵌入进行几何扰动，完全不修改扩散模型或引入额外条件；②实现了轻量化、易于兼容现有预训练扩散模型的多样性增强方案；③通过调节下界 lb 可直观控制多样性与身份一致性之间的权衡。

**🔧 技术方法**

使用技术包括：预训练的身份条件扩散模型（IDiff‑Face、Latent Diffusion）、Classifier‑Free Guidance、余弦约束的锥形采样（角度扰动）、LPIPS、年龄/表情/姿态熵等多维度多样性指标，以及 ResNet50+CosFace 的面部识别训练框架。

**📊 数据集**

数据集：扩散模型在 FFHQ、C‑WF（0.49M）及 WebFace4M（4M）上训练；合成身份嵌入来源于这些模型；面部识别训练采用 0.5M 或 1M 合成样本；评估基准包括 LFW、AgeDB‑30、CFP‑FP、CALFW、CP‑LFW 以及 IJB‑C。

**📈 对比分析**

与 GAN、数字渲染、以及其他基于扩散的生成方法（如 IDiff‑Face、ID^3、Arc2Face、HyperFace、UIFace 等）进行对比；在所有小规模基准上平均准确率达到 93.62%，仅比真实 C‑WF 的 94.63% 略低；在 IJB‑C 上也实现了领先或相近的表现；通过实验验证了不同 lb 与 CFG 参数对身份一致性与多样性的影响，表明该方法在保持身份保真度的同时显著提升了 intra‑class 变化。

**⚠️ 局限性**

局限性：①仅在身份嵌入空间扰动，未显式控制表情、光照、姿态等属性；②依赖于扩散模型的质量与偏差，若基模型存在缺陷，生成样本质量会受限；③需要经验性调节 lb 参数以平衡多样性与身份一致性；④与真实数据相比仍存在微小差距，且极端扰动可能导致身份一致性下降。

---

## 546. RAID: Retrieval-Augmented Anomaly Detection

**arXiv ID:** 2602.19611 | [PDF](https://arxiv.org/pdf/2602.19611v1)

**作者:** Mingxiu Cai `[一作]` (Northeastern University), Xiatian Zhu `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于检索增强生成（RAG）的工业无监督异常检测框架 RAID，能够通过检索正常样本并在生成阶段进行噪声抑制，实现精细的异常定位。

**💡 创新点**

创新点包括：①将 UAD 重新解释为 RAG 任务，充分利用检索信息进行生成；②构建三级向量数据库（类原型-语义原型-实例 token）实现粗细层级检索；③设计双向引导的 Mixture‑of‑Experts（MoE）过滤器，在检索后对匹配噪声进行自适应去噪，提升异常边界清晰度。

**🔧 技术方法**

使用预训练 ViT（DINOv2-s）作为特征提取器，基于余弦相似度进行检索，构造匹配成本体积，并通过两阶段 MoE 网络（指导层 + 去噪层）生成最终异常图。

**📊 数据集**

在四个工业异常检测基准上评测：MVTec‑AD、VisA、MPDD 和 BTAD，涵盖全射击、少射击和多数据集联合训练场景。

**📈 对比分析**

与 PatchCore、GLAD、AnomalyDINO、CostFilter‑AD 等多种 SOTA 方法对比，RAID 在全射击模式下 I‑AUROC 最高达 99.4%、P‑AUROC 98.6%；在少射击场景下在 MVTec‑AD/VisA 上分别提升约 6%/8% 的 I‑AUROC，且在多数据集联合训练中超过 OneNIP，表现出显著的通用性与可扩展性。

**⚠️ 局限性**

限制包括：①需要构建并维护多层向量数据库，增加了前期准备工作；②对检索模板的质量和数量仍有一定依赖，极少量样本下性能仍受限；③当前方法仅适用于纯视觉输入，未结合语言或多模态信息；④在极大规模数据集上的实时推理仍存在一定延迟。

---

## 547. DGPO: RL-Steered Graph Diffusion for Neural Architecture Generation

**arXiv ID:** 2602.19261 | [PDF](https://arxiv.org/pdf/2602.19261v1)

**作者:** Aleksei Liuliakov `[一作]` (Bielefeld University), Barbara Hammer `[通讯]` (Bielefeld University)

**通讯引用:** 9689 | [OpenAlex ID](https://openalex.org/A5091180862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 DGPO，将强化学习细调的离散图扩散模型扩展到有向无环图，用于神经架构搜索。

**💡 创新点**

通过拓扑节点排序和位置编码，使原本只适用于无向图的 GDPO 能处理有向无环图，并证明模型能学习可迁移的结构先验。

**🔧 技术方法**

采用离散图扩散（DiGress）+ 强化学习细调（GDPO）+ 拓扑排序 + 位置编码 + 优势归一化的 REINFORCE 等技术。

**📊 数据集**

在 NAS-Bench-101 与 NAS-Bench-201 两个公开搜索空间上进行实验。

**📈 对比分析**

与随机搜索、仅预训练无 RL、以及多种公开 NAS 方法比较，DGPO 在 NB201 上达到或逼近基准最优（91.61% CIFAR‑10、73.49% CIFAR‑100、46.77% ImageNet），在 NB101 上也表现接近最优，并能从仅 7% 低质量样本的预训练中通过 RL 取得近似最优结果，整体性能优于大多数基线。

**⚠️ 局限性**

需要大量在线查询（约 2k 次评估）成本高于基于条件的生成方法，且实验仅限于 NAS 基准，是否能推广到其他 DAG 领域尚未验证。

---

## 548. GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry

**arXiv ID:** 2602.18584 | [PDF](https://arxiv.org/pdf/2602.18584v1)

**作者:** Guanghui Min `[一作]` (University of Virginia), Chen Chen `[通讯]` (University of Virginia)

**通讯引用:** 73379 | [OpenAlex ID](https://openalex.org/A5100418548)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于耦合优化几何的针对性数据选择框架 GIST，旨在为指令微调挑选对特定目标任务最有影响的少量训练样本。

**💡 创新点**

创新点在于：①从优化几何角度统一分析数据选择方法，揭示传统基于 Adam 的对角预调仅捕捉轴对齐信息，忽视参数耦合；②利用验证梯度的谱分解（SVD）恢复低秩任务子空间，构建非对角几何逼近；③在此子空间上投影训练梯度并用余弦相似度评分，实现高效、精准的样本挑选。

**🔧 技术方法**

核心技术包括：梯度预热（轻量 LoRA 预训练）、验证梯度矩阵的 SVD 与低秩投影、子空间投影后的梯度对齐评分、最大相关（Max‑Relevance）多任务聚合。

**📊 数据集**

实验使用约 270k 条指令微调样本（Flan‑v2、CoT、Dolly、Open‑Assistant 等）作为候选池，目标集为 MMLU、TydiQA、BBH 三大基准，模型分别为 Llama2‑7B、Llama3.2‑3B 与 Qwen2.5‑1.5B。

**📈 对比分析**

与随机、长度、困惑度、Embedding 相似度、RDS+、以及基于 Adam 的 LESS 进行对比，GIST 在所有模型上均超过基线，平均提升 4.5–6.2%，在 5% 数据下甚至能匹配或优于使用完整数据的微调。其存储仅 0.29% 的基线量，计算量仅为 25%。

**⚠️ 局限性**

局限性包括：①依赖验证梯度低秩特性，若任务梯度结构不低秩或不稳定，子空间恢复效果会下降；②需要轻量 LoRA 预热，若预热不足可能导致子空间失真；③方法在 PEFT（LoRA）框架下验证，迁移到其他微调方式需进一步评估。

---

## 549. Can Multimodal LLMs See Science Instruction? Benchmarking Pedagogical Reasoning in K-12 Classroom Videos

**arXiv ID:** 2602.18466 | [PDF](https://arxiv.org/pdf/2602.18466v1)

**作者:** Yixuan Shen `[一作]` (Drexel University), Feng Liu `[通讯]` (Drexel University)

**通讯引用:** 11476 | [OpenAlex ID](https://openalex.org/A5100415332)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 SciIBI 基准，系统评估多模态 LLM 在 K–12 科学课堂视频中对核心教学实践（CIP）的自动编码，并提出基于证据的评估协议。

**💡 创新点**

①首个面向科学课堂视频的 CIP 编码基准；②将视觉与文本结合的多模态输入与不同提示策略系统对比；③引入模型需给出文本/时间戳证据的 Evidence‑Quality Score，揭示模型推理质量。

**🔧 技术方法**

使用多模态大语言模型（GPT‑4o、Claude Sonnet 4.5、Gemini‑2.5‑Pro、Qwen3‑VL‑235B、InternVL3‑78B、Llama‑3.3‑70B 等）与传统文本 LLM；采用零样本、少样本、链式思考提示；视频输入采用均匀采样帧；模型需输出结构化 JSON（预测标签、文本片段、时间戳）。

**📊 数据集**

SciIBI 数据集：113 条 6–600 秒的 K–12 科学课堂视频片段，来源于 YouTube 公开频道，按 NGSS 对齐并通过专家共识标注 CIP 四类及二元复杂度；包含转录文本和时间戳。

**📈 对比分析**

在文本仅 (T) 与视觉+文本 (V+T) 两种输入下，使用三种提示（Zero‑Shot、Few‑Shot、Chain‑of‑Thought）对八种模型进行评估；准确率最高为 InternVL3‑78B 在 V+T 零样本下 53.6%（比文本仅高 7.1pp）；链式思考对大模型提升约 5–8pp，若小模型则下降；总体准确率均低于数学 TalkMoves 约 79% 的 F1；证据质量评分表明准确率与推理质量并不完全一致。

**⚠️ 局限性**

模型对教学意图的深层推理不足，仍依赖表面词汇；多模态输入并非始终提升性能，受模型架构和训练数据影响；基准基于公开 YouTube 视频，缺乏真实课堂多样性；需要进一步 fine‑tune 或结合检索式人机协作以实现可用的教学辅助。

---

## 550. When the Inference Meets the Explicitness or Why Multimodality Can Make Us Forget About the Perfect Predictor

**arXiv ID:** 2602.18850 | [PDF](https://arxiv.org/pdf/2602.18850v1)

**作者:** J. E. Domínguez-Vidal `[一作]` (Institut de Robótica i Informàtica Industrial), Alberto Sanfeliu `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 7347 | [OpenAlex ID](https://openalex.org/A5041730641)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在机器人协同搬运任务中对两种推理器（力预测器与速度预测器）和两种显式通信方式（按钮接口与语音识别）以及它们的组合进行实验，对人机协作的感知与表现进行评估。

**💡 创新点**

创新点在于首次将推理与显式通信两大类技术在同一任务中并行比较，并探讨它们结合对人机交互质量的提升。

**🔧 技术方法**

采用了基于力传感器与激光雷达的机器人控制平台IVO，实施力预测与速度预测模型，使用按钮式交互与基于蓝牙麦克风的语音识别系统。

**📊 数据集**

使用自制实验数据集：75名志愿者共执行255次搬运实验，记录力、时间等客观指标以及7点李克特量表的主观评价。

**📈 对比分析**

通过方差分析、Kruskal‑Wallis检验等统计方法比较四种系统，结果显示单独的推理或通信方式对主观评价提升有限，而推理与通信的组合在流畅度、信任感和舒适度等指标上显著优于单一方案。

**⚠️ 局限性**

局限性包括：推理器性能提升未被用户明显感知；实验仅限于单一搬运场景；参与者被告知存在推理器可能影响行为；缺乏对更复杂任务或环境噪声下的验证。

---

## 551. Image-Based Classification of Olive Varieties Native to Turkiye Using Multiple Deep Learning Architectures: Analysis of Performance, Complexity, and Generalization

**arXiv ID:** 2602.18530 | [PDF](https://arxiv.org/pdf/2602.18530v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 552. No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection

**arXiv ID:** 2602.19248 | [PDF](https://arxiv.org/pdf/2602.19248v1)

**作者:** Zunkai Dai `[一作]` (Beijing University of Posts and Telecommunications), Yuanyuan Qiao `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1361 | [OpenAlex ID](https://openalex.org/A5090482974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种端到端的零样本视频异常检测框架 LAVIDA，利用多模态大型语言模型和伪异常数据实现帧级与像素级异常检测。

**💡 创新点**

创新点在于：①异常曝光采样器将语义分割数据转化为伪异常，消除对真实异常数据的依赖；②MLLM 提取语义特征，解决上下文依赖问题；③逆注意力令牌压缩有效抑制背景信息，提升稀疏异常检测；④多尺度语义投影将视频级语义与帧级信息融合，兼顾时间与空间粒度。

**🔧 技术方法**

使用的技术包括多模态大型语言模型（MLLM）、CLIP 文本与视觉编码器、SAM2 掩码解码器、Q-Former 投影、逆注意力令牌压缩、异常曝光采样器、以及多尺度语义投影模块。

**📊 数据集**

训练使用外部语义分割数据集（无 VAD 数据），评估在四个公开 VAD 基准数据集（UBnormal、ShanghaiTech、UCF-Crime、XD-Violence）以及 UCSD Ped2 的像素级检测。

**📈 对比分析**

与无监督、弱监督、少量样本以及现有零样本方法对比，零样本帧级 AUC 分别达到 76.45%、85.28%、82.18%，XD-Violence AP 为 90.62%；像素级 AUC 在 UCSD Ped2 上为 87.68%，均在对应指标上实现或超过当前 SOTA。

**⚠️ 局限性**

局限性包括：对低分辨率视频的 MLLM 理解能力有限，导致在 UCF-Crime 上不及弱监督方法；异常类别数量或压缩比例过大时模型性能下降；依赖大型语言模型与掩码解码器，计算资源需求高。

---

## 553. Gradient based Severity Labeling for Biomarker Classification in OCT

**arXiv ID:** 2602.19907 | [PDF](https://arxiv.org/pdf/2602.19907v1)

**作者:** Kiran Kokilepersaud `[一作]` (Georgia Institute of Technology), Charles Wykoff `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

为未标记的 OCT 图像生成疾病严重度伪标签，并利用这些标签进行监督式对比学习，以提升糖尿病视网膜病变相关生物标志物的分类精度。

**💡 创新点**

创新点在于通过梯度响应计算异常度并将其离散为严重度类别，进而在医学影像中以严重度相似性替代传统随机增强作为正样本选择策略；同时首次将监督式对比学习与伪严重度标签结合，以充分利用大量无标签数据。

**🔧 技术方法**

技术包括：① 基于自编码器的健康分布学习与梯度约束，② 计算重建误差与梯度相似度得到严重度分数，③ 将严重度分数划分为 N 个 bins 生成伪标签，④ 使用 ResNet‑18 + 多层感知机投影头训练监督式对比损失，⑤ 在已标记的生物标志物子集上冻结编码器并添加线性分类层进行微调。

**📊 数据集**

数据集：Kermany 健康 OCT 数据集用于训练健康自编码器；Prime + TREX DME 数据集共约 60k 未标记 OCT 与 7.5k 标记 20 种生物标志物（其中 5 种平衡用于二分类）的 OCT 图像。

**📈 对比分析**

与 SimCLR、PCL、MoCo v2 等三种最先进的自监督方法在同一 ResNet‑18 架构下比较，使用 25 轮训练、学习率 1e‑3、批量 64。实验显示，在多标记 AUC 以及单一标记的准确率/ F1‑score 上，基于严重度伪标签的监督式对比学习平均提升约 0.5–1.0%，最高单标记提升可达 4%（如 DME、IRF）。

**⚠️ 局限性**

局限性包括：① 需要手动选择严重度划分数 N，过多或过少会影响性能；② 伪标签的质量高度依赖自编码器的健康分布学习与梯度约束，若健康样本不足或分布偏差可能导致错误标签；③ 仅针对 OCT 视网膜图像验证，未证明可推广至其他医学影像模态。

---

## 554. Policy or Community?: Supporting Individual Model Creators' Open Model Development in Model Marketplaces

**arXiv ID:** 2602.19354 | [PDF](https://arxiv.org/pdf/2602.19354v1)

**作者:** Eun Jeong Kang `[一作]` (Cornell University), Angel Hsing-Chi Hwang `[通讯]` (University of Southern California)

**通讯引用:** 493 | [OpenAlex ID](https://openalex.org/A5085100954)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 19 位使用开源模型进行微调并在公开市场发布文本到图像模型的创作者进行半结构化访谈，分析其工作流程、对平台治理工具（如责任 AI 工具、模型归因）的使用以及他们对平台政策的感知与挑战，从而提出三大监管需求：降低下游危害、认可创作者原创性与保障模型所有权。

**💡 创新点**

创新点在于将平台治理与个人创作者实践结合，揭示责任 AI 工具被创作者用于自我保护与品牌传播而非纯粹透明；提出社区规范在塑造创作者责任感中的核心作用；为平台制定更细化、可操作化的指导方针与多方协商治理模式提供实证依据。

**🔧 技术方法**

采用定性研究方法：半结构化访谈、主题分析（reflexive thematic analysis）以及访谈脚本与工具原型（如模型归因工具、归因指示器）的实验性评估。

**📊 数据集**

数据集为 19 名创作者的访谈记录，涵盖其使用的平台（Hugging Face、CivitAI 等）、技术工具（ComfyUI、Google Colab）和工作经验；此外收集了相关社区讨论、平台政策文档和责任 AI 工具原型作为佐证材料。

**📈 对比分析**

由于研究侧重于定性洞察，没有传统意义上的“性能”评估；研究通过对访谈内容的编码与主题归纳，确认了三大监管需求与创作者对治理工具的真实使用场景，说明现有平台政策与工具在满足创作者需求方面存在显著缺口。

**⚠️ 局限性**

局限性包括：仅聚焦图像生成（Text‑to‑Image）创作者，缺乏语言模型或其他模态的视角；样本来自美国，可能不具备全球代表性；性别比例失衡，女性创作者需求可能被低估；访谈数据仅为自我报告，可能存在社会期望与偏差。

---

## 555. Carbon-Aware Governance Gates: An Architecture for Sustainable GenAI Development

**arXiv ID:** 2602.19718 | [PDF](https://arxiv.org/pdf/2602.19718v1)

**作者:** Mateen A. Abbasi `[一作]` (University of Jyväskylä), Niko K. Mäkitalo `[通讯]` (University of Jyväskylä)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了一种Carbon-Aware Governance Gates (CAGG)架构，在GenAI辅助软件开发流程中嵌入碳预算、能耗溯源与绿色验证调度。

**💡 创新点**

创新点在于将可持续性指标直接集成到治理检查点，并设计了能耗溯源账本、碳预算管理器和绿色验证编排器以及可执行的治理策略与可复用设计模式。

**🔧 技术方法**

采用的技术包括基于碳会计的能耗跟踪、动态模型选择与调度、基于策略的治理控制以及DevOps CI/CD流水线的集成。

**📊 数据集**

未使用具体数据集，本文以架构设计与案例场景为主。

**📈 对比分析**

由于缺乏实验验证，本文未给出性能对比；通过案例演示说明可以在保持合规与验证深度的同时降低碳排放。

**⚠️ 局限性**

局限性包括碳排放估算的不确定性、对安全关键系统可能过于严格的预算限制，以及引入碳可观测性与治理策略后可能增加的流程复杂度。

---

## 556. Agents of Chaos

**arXiv ID:** 2602.20021 | [PDF](https://arxiv.org/pdf/2602.20021v1)

**作者:** Natalie Shapira `[一作]` (Northeastern University), David Bau `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在真实实验环境中，对20名研究者在两周时间里对自治LLM代理进行红队攻防测试，记录并分析了11个代表性安全缺陷案例。

**💡 创新点**

首次系统展示了自治代理在持续记忆、工具调用、多方通信与跨代理交互等场景下的独特安全、隐私与治理风险，并提出了对代理层面的新型威胁模式。

**🔧 技术方法**

利用OpenClaw框架，将Claude Opus和Kimi K2.5 LLM与Discord、ProtonMail、Shell、文件系统等工具集成，借助心跳与Cron作业实现自治与任务调度。

**📊 数据集**

实验数据来源为实验产生的交互日志、电子邮件内容、Discord对话等自生成记录，未使用公开数据集。

**📈 对比分析**

通过对比对抗性与正常交互情景，展示了多种攻击成功率与失败模式；由于研究重点在安全案例而非性能指标，未给出传统准确率/召回率等度量。

**⚠️ 局限性**

受限于实验环境的早期OpenClaw实现、工具配置不完善、样本规模有限、缺乏统计推断，对不同LLM模型和部署平台的泛化性尚未评估，且仅聚焦代理层面，未覆盖基础模型的潜在缺陷。

---

## 557. Workflow-Level Design Principles for Trustworthy GenAI in Automotive System Engineering

**arXiv ID:** 2602.19614 | [PDF](https://arxiv.org/pdf/2602.19614v1)

**作者:** Chih-Hong Cheng `[一作]` (Carl von Ossietzky University of Oldenburg), Hasan Esen `[通讯]` (DENSO AUTOMOTIVE Deutschland)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一套面向安全关键系统工程的生成式 AI 工作流设计原则，构建了从需求变更识别、SysML v2 体系结构更新到回归测试的完整汽车工程流水线；

**💡 创新点**

创新点在于将传统的“全量单一提示”拆解为细粒度可检验的子任务，结合多模型多种采样、冗余与一致性机制，并在关键步骤引入经典算法做校验，形成可追溯、可验证的生成式 AI 框架；

**🔧 技术方法**

使用了大语言模型（Qwen3:32B、Nemotron3:30B、GPT-OSS:20B）进行文本摘要、段落检索与差异提取；结合 PDF 结构化工具、TF‑IDF 相似度、自然语言推理（NLI）、SysML v2 编译器和静态分析器；

**📊 数据集**

主要实验数据集为 Automotive SPICE v3.1 与 v4.0 的需求文档（约 100 页/版本），以及简化版 AEB 体系结构模型；

**📈 对比分析**

对比方法是将单体“全量 diff”提示与拆分的段落级多模型提取进行实验，结果显示段落级拆解后召回率提升至 0.79，精确度在执行 sanity 检查后可达 0.87；

**⚠️ 局限性**

局限性包括仍需针对各领域手工设计和验证校验规则，工作流对特定安全标准和技术需求的适配需工程师参与；

---

## 558. HOCA-Bench: Beyond Semantic Perception to Predictive World Modeling via Hegelian Ontological-Causal Anomalies

**arXiv ID:** 2602.19571 | [PDF](https://arxiv.org/pdf/2602.19571v1)

**作者:** Chang Liu `[一作]` (National University of Defense Technology), Zhiping Cai `[通讯]` (National University of Defense Technology)

**通讯引用:** 8056 | [OpenAlex ID](https://openalex.org/A5006334685)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本工作提出HOCA-Bench，一个基于黑格尔哲学的物理异常分类框架，用以评估视频大模型的预测世界建模能力。

**💡 创新点**

创新点在于将实体逻辑（本体异常）与交互逻辑（因果异常）区分为两层，利用生成式视频模型产生可控异常，构建包含1,439段视频、3,470问答对的综合基准，并引入四阶段任务（可行性检查、域归类、细粒度描述、逆因果推理）及H-Index评价。

**🔧 技术方法**

技术包括：Hegelian逻辑层设计、生成式视频模型（如Wan 2.1、HunyuanVideo、Sora等）做对抗生成、LLM（GLM-4.5、GPT-OSS-120B）用于自动标注与评价、视频-LLM体系结构（InternVL、Qwen、GLM系列）以及系统2（Thinking Mode）推理。

**📊 数据集**

使用的数据集为HOCA-Bench自身，涵盖809段合成异常视频与630段真实视频，主题覆盖357类、场景198类；此外引用了Panda-70M、VideoPhy2、IPV-Bench等现有视频资源。

**📈 对比分析**

通过17个视频大模型（Dense、MoE、Thinking vs Instruct、开源与闭源）进行对比，结果显示模型在本体异常上表现良好（F1≈70-80%），但因果异常明显落后（差距≥20%），System-2推理虽提升深度推理表现，但仍未消除差距；整体H-Index最高的开源模型Qwen3-VL-32B达到70.3，超过部分闭源模型。

**⚠️ 局限性**

局限性包括：依赖生成式模型产生的异常可能不完全符合真实物理规律；评测侧重可见异常，未覆盖更复杂的物理模拟；模型在多异常场景下仍表现不佳，精确定位与完整识别能力不足。

---

## 559. Joint Post-Training Quantization of Vision Transformers with Learned Prompt-Guided Data Generation

**arXiv ID:** 2602.18861 | [PDF](https://arxiv.org/pdf/2602.18861v1)

**作者:** Shile Li `[一作]` (vivo Tech Research), Onay Urfalioglu `[通讯]` (vivo Tech Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种端到端的后训练量化框架，可在不使用标签数据的情况下，对Vision Transformer进行低比特量化，并通过学习多模态提示在Stable Diffusion Turbo中生成多样化的合成校准样本，实现数据自由量化。

**💡 创新点**

创新点包括：① 通过全网络联合优化权重、激活量化参数、通道尺度与偏移以及权重量化微调，实现跨块误差补偿；② 引入多模态提示学习，在生成合成数据时兼顾语义正确性与视觉多样性；③ 在极低比特（W1.58A8）下仍保持较高精度。

**🔧 技术方法**

使用的技术包括：统一量化+STE、通道尺度重参数化、权重量化微调、KL蒸馏与MSE特征重构损失；数据自由部分采用Stable Diffusion Turbo与CLIP文本编码器，配合多模态提示学习与多样性正则化。

**📊 数据集**

使用ImageNet-1K数据集进行校准，实际实验中对真实样本使用10k张，对合成样本使用100k张（每类多模态提示生成）。

**📈 对比分析**

与RepQ‑ViT、FIMA‑Q、APHQ‑ViT等前沿PTQ方法在ViT、DeiT、Swin三个主干上进行对比，在W4A4、W3A3、W1.58A8等低比特设置下实现了最高精度（例如ViT‑S在W1.58A8下从~4%提升至约70%），数据自由模式仅落后1–2%。

**⚠️ 局限性**

局限性包括：① 对生成模型的质量与多样性高度依赖；② 目前仅验证于ImageNet分类任务，对其他视觉任务或更小模型的通用性尚未充分评估；③ 训练过程仍需要数小时GPU时间，尽管相对低廉，但在资源极端受限场景下仍是挑战。

---

## 560. Stress-constrained Topology Optimization for Metamaterial Microstructure Design

**arXiv ID:** 2602.19662 | [PDF](https://arxiv.org/pdf/2602.19662v1)

**作者:** Yanda Chen `[一作]` (Arts et Metiers Institute of Technology), Francisco Chinesta `[通讯]` (CNRS)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在静态和高循环疲劳条件下加入局部应力约束的拓扑优化框架，用于设计具有优异力学性能的多尺度结构体（超材料）微结构。

**💡 创新点**

创新点包括：① 将增广拉格朗日方法扩展为同时处理局部应力约束与全局约束；② 将静态应力约束与多种临界平面疲劳准则（Findley、Matake、Dang‑Van）集成进同一优化框架；③ 通过对比不同疲劳准则的结果揭示了优化结果对准则选择的敏感性。

**🔧 技术方法**

使用的技术包括：数值同质化（异质性单元平均法）+ SIMP 质量密度法 + Heaviside 投影 + 多尺度拓扑优化 + 增广拉格朗日求解（单个伴随向量），以及针对周期边界条件的消除/拉格朗日乘子/罚函数方法。

**📊 数据集**

数据集为 Ti‑6Al‑4V 添加制造材料（E=108.8 GPa, ν=0.29, σ_y=972 MPa, f_-1=454 MPa, t_-1=300 MPa），设计域为 10 mm³ 立方体（或 10×10 mm² 平面），采用 10⁵ 次完全反转周期加载的周期性负载案例。

**📈 对比分析**

通过 2D、3D 参考问题（最大体积模量、最大剪切模量、最小泊松比）与不同疲劳准则的结果进行对比；结果表明加入局部应力约束可显著降低最大 von Mises 应力（≈10‑15%），并保持弹性模量几乎不变；在疲劳约束下，Findley 与 Matake 产生相似但在低体积分数下的拓扑差异；Dang‑Van 对准则表现出更均匀的应力分布。

**⚠️ 局限性**

局限性：① 未考虑激光粉末床熔化过程对材料力学性能的影响；② 仅给出局部最优解，缺乏实验验证与后处理；③ 对大规模 3D 设计仍存算力与存储瓶颈；④ 对不同材料/加载条件的泛化尚需进一步研究。

---

## 561. Towards Dexterous Embodied Manipulation via Deep Multi-Sensory Fusion and Sparse Expert Scaling

**arXiv ID:** 2602.19764 | [PDF](https://arxiv.org/pdf/2602.19764v1)

**作者:** Yirui Sun `[一作]` (Fudan University), Zhongxue Ga `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 DeMUSE，一种将 RGB、深度和 6 轴力反馈序列化为统一 token 并在 Diffusion Transformer 中深度融合的多模态控制框架，用于实现高精度、物理一致的机器人操作。

**💡 创新点**

创新点包括：① 将多模态信息统一序列化并在 Transformer 里进行全局自注意力融合；② 引入 Adaptive Modality-specific Normalization (AdaMN) 来解决模态尺度不均衡与表示失衡；③ 使用稀疏 Mixture-of-Experts (MoE) 结构在保持实时推理的前提下扩展模型容量；④ 采用联合去噪目标，使环境演化与动作生成同步，确保物理一致性。

**🔧 技术方法**

技术手段主要包括 Diffusion Transformer、联合去噪（Joint Denoising）、AdaMN、稀疏 MoE、Action Chunking、EMA 过滤以及多模态同步采样的数据处理管线。

**📊 数据集**

使用了增强版 MetaWorld MT50 数据集（包含同步深度与力信号）以及真实世界 UR10+RG2+RealSense 的高精度实验数据。

**📈 对比分析**

通过与 Diffusion Policy、RT-2、PAD、ForceVLA、RDT-1B 等五个基线在 MT50 及真实环境下的多任务评估对比，DeMUSE 在模拟任务中取得 83.2% 的平均成功率，在实地任务中达到 72.5%，在 Medium/Hard 难度任务中提升约 7%–6% ，在接触密集任务中成功率可达 96%。

**⚠️ 局限性**

局限性：需要高质量的力、深度同步采样；AdaMN 与 MoE 需要额外的计算开销；稀疏 MoE 在大规模训练时易出现专家崩溃；当前验证集中主要是特定任务，跨任务泛化与长期稳定性尚未完全证明。

---

## 562. Denotational Semantics for ODRL: Knowledge-Based Constraint Conflict Detection

**arXiv ID:** 2602.19883 | [PDF](https://arxiv.org/pdf/2602.19883v1)

**作者:** Daham Mustafa `[一作]`, Stephan Decker `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文提出了一套基于解释语义的 ODRL 约束冲突检测框架，实现了对所有八种 ODRL 约束的三值判定（冲突、兼容、未知）。

**💡 创新点**

创新点在于将 ODRL 约束映射到知识库概念集合，利用集合交叉实现冲突检测，并在跨数据空间、跨标准对齐时保证冲突不被误判；同时揭示排他组合需要比合取或析取更强的知识库公理。

**🔧 技术方法**

采用了 EPR（有效命题）一阶逻辑编码、双向解释规则、Kleene 三值逻辑和顺序保持对齐的数学证明，并在 Isabelle/HOL 中机械化验证所有元定理。

**📊 数据集**

使用了六类实际知识库（GeoNames、ISO 3166、DPV、GDPR 衍生分类、BCP 47、ISO 639‑3）及四个结构化基准 KB，构建了 154 个对照实验。

**📈 对比分析**

在 Vampire 与 Z3 两个独立定理/SMT 求解器上对所有基准进行评估，取得 100% 一致性，证明了框架在可判定 EPR 片段内的高效与准确性。

**⚠️ 局限性**

局限在于仅覆盖了 8 种 ODRL 约束、仅处理约束级别的冲突，未对完整规则级别的权限/禁止冲突及 11 种未实现的 ODRL 约束进行进一步扩展。

---

## 563. Spatial-Temporal State Propagation Autoregressive Model for 4D Object Generation

**arXiv ID:** 2602.18830 | [PDF](https://arxiv.org/pdf/2602.18830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 564. Dynamic 3D Convex Hulls Revisited and Applications

**arXiv ID:** 2602.18653 | [PDF](https://arxiv.org/pdf/2602.18653v1)

**作者:** Haitao Wang `[一作]` (University of Utah), Haitao Wang `[通讯]` (University of Utah)

**通讯引用:** 38245 | [OpenAlex ID](https://openalex.org/A5100396117)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

论文提出一种改进的 Chan 框架，用于维护 3D 点集的动态凸包（或 3D 平面/曲面的动态下层包络），实现 O(log²n) 的插入时间、O(log³n log log n) 的删除时间以及 O(log³n / log log n) 的查询时间，并将该数据结构应用到加权最邻近查询、动态 BCP、最小生成树、凸层、盘图最短路等多种几何问题中。

**💡 创新点**

创新点在于：1）对 Chan 原始结构进行细粒度重构，采用主/辅助子集划分和批量合并策略，将平均子结构层级 2 提升为最坏情况的上限；2）通过对删除操作的细粒度计费与合并阈值的精细设置，使得删除时间从 O(log⁴n) 降到 O(log³n log log n)；3）将新结构嵌入到多种应用中，获得比以往最优 O(log⁴n) 或 O(n log⁴n) 的时间提升约 1/ log n；4）在 k‑lowest‑planes 查询和半空间范围报告中，实现 O(log³n / log log n + k) 的查询时间。

**🔧 技术方法**

主要技术包括：a）多层垂直浅切割（vertical shallow cutting）与递归子结构；b）主/辅助子集分离和批量插入/合并；c）冲突列表的插入/删除只维护一次的方式；d）使用贝塞尔-塞克斯的插入‑仅 Voronoi 结构；e）对冲突列表的 k‑最小查询采用 Frederickson 堆选择改进；f）动态盘图最短路采用基于最近邻查询的逐步扩展算法。

**📊 数据集**

论文未在实验中使用公开数据集，而是以理论分析和期望运行时间为主。所有结论均基于对期望或最坏情况时间复杂度的证明；在随机化版本中，假设浅切割生成是期望线性时间。

**📈 对比分析**

与之前的 O(log⁴n) 插入/删除或 O(n log⁴n) 盘图最短路相比，本文提供了 O(log³n log log n) 或 O(n log³n log log n) 的改进。对于 k‑lowest‑planes 查询，旧版 O((k+log n) log n) 的查询时间被压缩到 O(k + log³n / log log n)，在高复杂度操作占主导时提供了显著的对数级别提升。

**⚠️ 局限性**

局限性：a）查询时间相较于原始 Chan 结构略增（由 O(log²n) 变为 O(log³n / log log n)），不适合极低查询时间需求；b）对随机化浅切割的期望时间依赖，非确定性部分；c）在 k‑lowest‑planes 的第二方案中，仍需要额外的 log log n 预处理时间，可能在大规模数据上影响常数；d）对某些特殊几何结构（如仅单位圆盘）仍未充分优化；e）实现复杂度高，需维护多层子结构和多种冲突列表，实用性待验证。

---

## 565. Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation

**arXiv ID:** 2602.19863 | [PDF](https://arxiv.org/pdf/2602.19863v1)

**作者:** Filip Wolf `[一作]` (University of Ljubljana), Luka Čehovin Zajc `[通讯]` (University of Ljubljana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种双教师对比蒸馏框架DEO，利用多光谱教师和光学VFM教师共同训练学生网络，实现多光谱和光学特征的统一学习。

**💡 创新点**

创新点在于将学生的预训练目标与光学VFM（如DINOv3）的对比自蒸馏范式对齐，既保留多光谱的自蒸馏学习，又通过第二教师将光学高级语义知识迁移到多光谱空间，实现高效跨模态知识迁移。

**🔧 技术方法**

技术包括对比自蒸馏（DINO）、编码率正则化、Swin Transformer骨干、光学与多光谱分支的双视图增强、投影头分离、光学教师的类标记与局部标记蒸馏、以及高分辨率光学数据增强等。

**📊 数据集**

使用Sentinel‑2多光谱数据（去除3个大气波段），fMoW‑Sentinel、fMoW‑RGB作为预训练集；下游任务采用SpaceNetv1、LEVIR、OSCD、GEO‑Bench多光谱/光学分割、变化检测与分类数据集，以及10%低标注数据的子集。

**📈 对比分析**

与现有MIM或单教师方法相比，DEO在光学和多光谱语义分割、变化检测、分类任务上均实现了SOTA，平均提升约3.64%（分割）、1.2%（变化检测）和1.31%（分类），在低标注场景下仍保持优势。

**⚠️ 局限性**

局限性包括对光学VFM教师质量的依赖、假设光学与多光谱数据空间对齐、对SAR等非光学模态的可扩展性不足，以及未能为非光学教师提供强大蒸馏目标。

---

## 566. Variational Trajectory Optimization of Anisotropic Diffusion Schedules

**arXiv ID:** 2602.19512 | [PDF](https://arxiv.org/pdf/2602.19512v1)

**作者:** Pengxi Liu `[一作]` (Duke University), Xiang Cheng `[通讯]` (Duke University)

**通讯引用:** 307256 | [OpenAlex ID](https://openalex.org/A5100362465)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种变分框架，用矩阵值的噪声轨迹替代标量噪声调度，联合训练score网络和噪声矩阵

**💡 创新点**

创新点在于轨迹级score匹配目标、可学习的矩阵噪声路径以及推导出的矩阵逆向ODE求解器与梯度估计方法

**🔧 技术方法**

使用变分推导、矩阵值路径优化、轨迹级score匹配、Heun式矩阵逆向ODE、梯度插值估计及流式归一化技术

**📊 数据集**

在CIFAR‑10、AFHQv2、FFHQ、ImageNet‑64四个标准图像生成数据集上进行实验

**📈 对比分析**

与EDM基准模型在FID与NFE（函数评估次数）上比较，学习的矩阵噪声调度在多种数据集和NFE设置下均能取得显著FID提升，尤其在有条件生成任务中表现最优

**⚠️ 局限性**

限制在于矩阵轨迹仍需满足正定与可交换假设，结构化投影分解可能限制表达能力，并且对极大维度的实现仍面临计算瓶颈

---

## 567. VecFormer: Towards Efficient and Generalizable Graph Transformer with Graph Token Attention

**arXiv ID:** 2602.19622 | [PDF](https://arxiv.org/pdf/2602.19622v1)

**作者:** Jingbo Zhou `[一作]` (Zhejiang University), Stan Z. Li `[通讯]` (Westlake University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为VecFormer的图Transformer模型，通过向量量化（SoftVQ）对节点特征和图结构进行离散化编码，并在第二阶段使用图令牌（Graph Token）进行跨码本注意力，实现高效节点分类；

**💡 创新点**

创新点包括：①将SoftVQ引入图Transformer，将图的特征与结构压缩为离散码，降低注意力计算复杂度；②设计图令牌注意力机制，取代传统节点级注意力，提升对分布外（OOD）样本的泛化能力；③采用两阶段训练范式，先自监督学习码本，再在Transformer中进行微调；

**🔧 技术方法**

使用技术：图神经网络（GAT）编码器、SoftVQ向量量化、图令牌生成与融合、跨码本自注意力、标准交叉熵分类损失；

**📊 数据集**

数据集：六类同源图（CoraFull、Pubmed、Computer、Photo、CS、Physics）、两类异源图（Chameleon、Squirrel）、OOD数据（Cora、Citeseer、Twitch）以及大规模图（Ogbn‑proteins、Amazon2m、Pokec）；

**📈 对比分析**

与传统GNN（GCN、GAT、GraphSAGE）、多种Graph Transformer（GOAT、GraphGPS、NodeFormer、SGFormer、Exphormer、NAGphormer、Polynormer）以及基于量化的VQGraph比较。VecFormer在节点分类、OOD泛化以及大规模图上均显著优于对照方法，尤其在异源图和OOD场景中提升了8–17%的准确率；

**⚠️ 局限性**

局限性：①需要手动设定码本大小和图令牌数量，超参数调节成本较高；②两阶段训练流程略长，增加训练复杂度；③目前仅针对静态图结构，未验证对动态图或时序图的适用性；④在极大图规模下仍需进一步压缩码本以避免内存瓶颈。

---

## 568. TAPE: Tool-Guided Adaptive Planning and Constrained Execution in Language Model Agents

**arXiv ID:** 2602.19633 | [PDF](https://arxiv.org/pdf/2602.19633v1)

**作者:** Jongwon Jeong `[一作]` (University of Wisconsin), Kangwook Lee `[通讯]` (KRAFTON)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为TAPE的语言模型代理框架，用工具驱动的自适应规划与受限执行来降低规划错误和采样错误。

**💡 创新点**

创新点在于：①将多条抽象路径聚合成计划图并利用外部整数线性规划（ILP）挑选可行路径；②在执行时采用受限解码抑制采样噪声；③实时检测与重规划以应对环境与内部模型的不匹配。

**🔧 技术方法**

技术包括：语言模型内部世界模型推理、计划图构建与状态归并、ILP求解、受限解码（constrained decoding）、动态重规划机制。

**📊 数据集**

使用的数据集包括Sokoban、ALFWorld、MuSiQue和GSM8K-Hard（各自添加了预算/工具使用等可行性约束）。

**📈 对比分析**

与ReAct和Plan‑and‑Act框架对比，TAPE在所有四个基准上均表现更好，尤其在困难设置和弱模型上提升约20–21个百分点；对Best‑of‑N等变体也保持优势。

**⚠️ 局限性**

局限性在于：①计划图构建高度依赖LM的结构化与状态归并能力；②需要预设的外部求解器，可能不适用于所有任务的优化形式；③增加了推理时间与算力消耗。

---

## 569. MagHeart: Exploring Playful Avatar Co-Creation and Shared Heartbeats for Icebreaking in Hybrid Meetings

**arXiv ID:** 2602.18676 | [PDF](https://arxiv.org/pdf/2602.18676v1)

**作者:** Black Sun `[一作]` (Aarhus University), Eve Hoggan `[通讯]` (Aarhus University)

**通讯引用:** 2096 | [OpenAlex ID](https://openalex.org/A5002561749)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一个多模态系统 MagHeart，用于混合会议的破冰，结合了基于 LEGO 的共同创作头像和磁性心率显示装置。

**💡 创新点**

创新点在于：①将身体感知（磁性心率）与可视化头像同步，提升远程参与者的可感知存在感；②通过轻松的 LEGO 共同创作，降低远程参与者的沉默门槛；③将心率信息以低干扰的节奏动画呈现，既提供情境线索又避免过度信息泄露。

**🔧 技术方法**

采用了 ESP32 微控制器 + PWM 电磁铁实现心率节奏动画；Apple Watch + HealthKit 收集实时心率；移动端 iOS App 与后端 FastAPI 通过 WebSocket/Server‑Sent Events 传输数据；使用 LEGO 官方建模网站进行头像设计；物理装置由 MDF、PETG、磁铁构成。

**📊 数据集**

未使用公开数据集；实验采用 34 名受试者的情境化问卷，记录预期体验与后情境体验，量化指标来自 7 点李克特量表。

**📈 对比分析**

方法为情境前后配对 t 检验，比较远程与现场参与者的预期参与、尴尬、焦虑等差异；结果显示：对远程参与者，低参与度、低发言意愿显著下降（p < .001），对现场参与者，发言意愿下降显著（p = .007）。所有参与者对社交存在感与未来使用意愿均得分在 5–6 之间，表明高接受度。

**⚠️ 局限性**

限制：①实验基于情境视频，未观察真实使用行为；②样本规模有限，未覆盖大组或高压会议场景；③缺乏与现有破冰工具的直接对比；④心率共享的隐私、干扰等问题仍需进一步探索。

---

## 570. Federated Learning-Assisted Optimization of Mobile Transmission with Digital Twins

**arXiv ID:** 2602.18627 | [PDF](https://arxiv.org/pdf/2602.18627v1)

**作者:** Mohammad Heydari `[一作]` (McMaster University), George Karakostas `[通讯]` (McMaster University)

**通讯引用:** 1708 | [OpenAlex ID](https://openalex.org/A5051576344)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种基于数字孪生的隐私保护调度框架，解决在能量约束下的多用户上行传输时隙/带宽分配问题。

**💡 创新点**

创新点在于：①仅通过联邦优化与DT交互，避免泄露用户通道状态；②使用依赖性舍入（Smart/ Simple）将松弛解转换为可执行调度；③首次实现跨DT无裸数据共享的通道调度。

**🔧 技术方法**

核心技术包括：联邦学习/优化（增广拉格朗日 + ADMM）、凸松弛求解、Smart/ Simple Dependent Rounding、基于GMM的阈值化、二分搜索求最短传输周期。

**📊 数据集**

使用仿真数据：基于Jakes模型生成的时变小尺度衰落，10名移动用户，100个时间槽，10个独立实验组，模拟真实无线链路条件。

**📈 对比分析**

与最优MINLP求解器Bonmin对比，展示带宽/能量提升仅比最优低约1–2%，约束违例率<2%，算法总时延≤500 ms（P1,P3）或≤15 ms（P2），满足实时调度需求。

**⚠️ 局限性**

局限性：仅适用于用户数较少或可分组的场景；对非凸约束的理论收敛性不完整；依赖通道预测精度，需进一步验证大规模用户下的可扩展性。

---

## 571. String Diagrams for Monoidal Categories, in Rocq

**arXiv ID:** 2602.19806 | [PDF](https://arxiv.org/pdf/2602.19806v1)

**作者:** Damien Pous `[一作]` `[通讯]` (Centre National de la Recherche Scientifique), Damien Pous (Centre National de la Recherche Scientifique)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文实现了一个 Rocq 库，用于定义任意单张范畴，并提供了决定式程序来证明态射等价，同时配备了独立的字符串图形编辑器，支持在图形层面进行重写并生成正式证明。

**💡 创新点**

创新点在于自动推断 MacLane 等价同构、实现完整的决定程序与 tetris 风格归一化，以及将图形编辑与 Coq 形式化紧密耦合，使得图形重写可直接导出可读的 Coq 证明。

**🔧 技术方法**

使用了 Coq 以及 Hierarchy Builder 进行结构化形式化，Ltac 反射技术编写自动化策略，OCaml 负责图形编辑与布局，力导向算法实现字符串图形排版。

**📊 数据集**

本文并未使用传统机器学习数据集，而是以单子组合的实例作为演示用例来验证系统功能。

**📈 对比分析**

通过对字符串图形的归一化和正则化进行等价判定，利用决定程序 mcat 自动证明等式；在演示中证明步骤几乎全自动完成，效率足以满足交互式证明需求。

**⚠️ 局限性**

局限性在于仅支持所有节点至少有一个输出端口的单张范畴；对空目标态射、双范畴或对称单张范畴的支持尚未完成，需要进一步扩展。

---

## 572. Sketch2Feedback: Grammar-in-the-Loop Framework for Rubric-Aligned Feedback on Student STEM Diagrams

**arXiv ID:** 2602.18520 | [PDF](https://arxiv.org/pdf/2602.18520v1)

**作者:** Aayam Bansal `[一作]` `[通讯]` (Institute of Electrical and Electronics Engineers), Aayam Bansal (Institute of Electrical and Electronics Engineers)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于语法循环的图解反馈框架Sketch2Feedback，专门为学生绘制的自由体图和电路图提供即时、符合评分标准的反馈。

**💡 创新点**

创新点在于将多模态反馈过程拆分为四个阶段（混合感知、符号图构造、约束检查与受限VLM生成），通过约束检查仅允许模型报告已被验证的错误，从而显著降低幻觉并实现错误来源可追踪。

**🔧 技术方法**

技术包括经典计算机视觉方法（CLAHE、阈值化、HoughLinesP等）进行原语检测、图结构构造与基于规则的约束推理，以及轻量化视觉语言模型Qwen2‑VL‑2B进行受限自然语言生成。

**📊 数据集**

使用了两套自研微基准FBD‑10（200个自由体图）和Circuit‑10（200个电路图），每套按10个场景各20个样本，采用合成绘图噪声模拟真实学生手绘。

**📈 对比分析**

在检测精度、反馈可操作性、幻觉率、校准和延迟等多维指标上与端到端LMM（LLaVA‑1.5‑7B）以及仅视觉检测基线进行比较，结果显示在自由体图上LLaVA领先，而在电路图上Sketch2Feedback的微F1显著优于LLaVA，且在电路图的可操作性评分达到5/5。

**⚠️ 局限性**

主要局限包括感知模块对真实学生绘图的泛化不足、测试集规模有限导致置信区间较宽、约束检查可能过度触发导致高幻觉率、以及缺乏课堂实验验证教学效果。

---

## 573. Back to Blackwell: Closing the Loop on Intransitivity in Multi-Objective Preference Fine-Tuning

**arXiv ID:** 2602.19041 | [PDF](https://arxiv.org/pdf/2602.19041v1)

**作者:** Jiahao Zhang `[一作]` (Carnegie Mellon University), Zhiwei Steven Wu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3431 | [OpenAlex ID](https://openalex.org/A5001070941)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了最大熵布莱克威尔胜者（MaxEntBW）概念并设计了一种可扩展的S算法，用于在多目标不传递偏好环境下进行偏好微调和LLM优化。

**💡 创新点**

创新点包括：① 处理多目标不传递偏好而不需单目标投影；② 通过熵正则化消除对抗训练需求；③ 将原三玩家游戏转化为单玩家回归问题，显著提升可扩展性。

**🔧 技术方法**

采用游戏理论（Blackwell胜者）、最大熵/ KL 正则化、在线镜像下降、回归式梯度估计、配对式LLM评判（Qwen3-14B）等技术。

**📊 数据集**

使用 WildChecklists（含Rubric化评估）作为训练数据；评测采用 AlpacaEval、Arena-Hard、IFEval、MMLU、ARC、HellaSwag、TruthfulQA 等标准基准。

**📈 对比分析**

与 RLCF、单目标优化及固定对手等基线对比，S 在指令跟随、聊天以及通用评测中均取得更高分数（约提升1–2个百分点），且未出现能力退化。

**⚠️ 局限性**

局限性：仍无法完全消除评判不传递；对评判模型质量与回归误差假设敏感；对更大规模或多模态模型的可扩展性仍待验证。

---

## 574. Learning to Detect Language Model Training Data via Active Reconstruction

**arXiv ID:** 2602.19020 | [PDF](https://arxiv.org/pdf/2602.19020v1)

**作者:** Junjie Oscar Yin `[一作]` (University of Washington), Hannaneh Hajishirzi `[通讯]` (University of Washington)

**通讯引用:** 47119 | [OpenAlex ID](https://openalex.org/A5067919401)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于强化学习的主动数据重建攻击（ADRA），通过在目标语言模型上进行对比奖励的微调来检测并重构训练数据，实现成员推断。

**💡 创新点**

首次将强化学习与对比奖励相结合，主动激活模型内在的成员记忆；引入自适应匹配（ADRA+）进一步提升检测效果，构成了首个主动成员推断攻击框架。

**🔧 技术方法**

使用on‑policy RL（GRPO）、对比奖励（matching/ADRA+）、词集/最长公共子序列/ n‑gram覆盖等重建指标，并与传统 loss‑based 基线进行对比。

**📊 数据集**

在6个自建LLM‑MIA数据集（Dolma3/Tulu3、AIME/Olympia Math、S1/S1.1）以及公开基准（BookMIA、WikiMIA2024‑Hard）上进行评估。

**📈 对比分析**

与5种 loss‑based 基线和重建采样基线相比，ADRA 在预训练、后训练与蒸馏三大场景下平均提升 10.7% AUROC，某些指标提升高达 18.8%/7.6%，整体表现显著优于现有方法。

**⚠️ 局限性**

需要在目标模型上执行 RL 微调，计算成本和时间高；对大模型或缺乏公开训练数据的模型适用性有限；未来的对抗性防御可能进一步削弱重建效果。

---

## 575. AgenticSum: An Agentic Inference-Time Framework for Faithful Clinical Text Summarization

**arXiv ID:** 2602.20040 | [PDF](https://arxiv.org/pdf/2602.20040v1)

**作者:** Fahmida Liza Piya `[一作]` (University of YYY), Rahmatollah Beheshti `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提供ICML 2026会议的论文提交与排版规范

**💡 创新点**

通过统一的模板与严格的格式要求，保障会议论文质量与公平评审

**🔧 技术方法**

使用LaTeX、PDF、Type‑1字体、EPS/PDF图形以及算法环境等技术

**📊 数据集**

未使用具体数据集，仅描述通用的提交流程

**📈 对比分析**

不涉及实验比较，仅说明提交与审稿步骤，性能指标不适用

**⚠️ 局限性**

局限于ICML 2026，需严格遵守细节，否则将被拒稿

---

## 576. Refactoring for Novices in Java: An Eye Tracking Study on the Extract vs. Inline Methods

**arXiv ID:** 2602.18579 | [PDF](https://arxiv.org/pdf/2602.18579v1)

**作者:** José Aldo Silva da Costa `[一作]` (State University of Paraiba), Alessandro Garcia `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过眼动追踪实验和问卷调查，比较了 Java 初学者在执行包含 Extract Method 与 Inline Method 重构的程序时的代码理解效率和视觉工作量，并结合访谈数据探讨了方法命名对认知的影响。

**💡 创新点**

创新点在于首次采用动态眼动指标（注视持续时间、注视次数、回视次数）来量化重构对初学者代码理解的影响，并将定量结果与访谈和调查的质性数据相结合，揭示方法抽取在不同任务复杂度下的利弊；同时验证了“命名灯塔”在初学者中的有效性不足。

**🔧 技术方法**

使用技术包括 Tobii Eye Tracker 4C 眼动仪、基于 Dispersion-Threshold Identification 的注视检测、统计检验（Shapiro-Wilk、t检验、Mann‑Whitney、FDR 校正）以及 Strauss & Corbin 的扎根理论进行访谈文本编码。

**📊 数据集**

数据集为 32 名本科生在 8 个简易 Java 任务（每个任务有 Inline 与 Extract 两种实现）收集的眼动和任务完成数据，外加 58 名初学者完成的在线问卷数据。

**📈 对比分析**

比较方法为对 Inline 与 Extract 两种重构方式在时间、尝试次数、注视持续时间、注视次数、回视次数等指标上做配对或分组统计；结果显示：对复杂任务（如阶乘、最高分）提取方法显著降低时间与视觉努力（多达 78.8% 与 84.6% 的改进），但对简单任务（如平方面积、偶数判断）反而导致时间与回视大幅增加（高达 166.9% 与 200%），表明效果与任务复杂度密切相关。

**⚠️ 局限性**

局限性包括样本规模相对较小、仅覆盖 Java 初学者、任务数量有限、眼动仪精度与校准误差可能影响细粒度分析、且实验仅评估了两种重构方式，未涵盖其他常见重构；未来研究需要在更大、多样化人群与多语言环境中验证结果。

---

## 577. CIBER: A Comprehensive Benchmark for Security Evaluation of Code Interpreter Agents

**arXiv ID:** 2602.19547 | [PDF](https://arxiv.org/pdf/2602.19547v1)

**作者:** Lei Ba `[一作]`, Songze Li `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并公开CIBER，一个自动化动态评估框架，用以测量代码解释器代理在四类攻击下的安全性。

**💡 创新点**

提出首个结合动态攻击生成、隔离沙箱和状态感知评估的综合基准，揭示结构与模型对齐、自然语言伪装以及隐式威胁的三层漏洞。

**🔧 技术方法**

利用四种攻击（DPI、IPI、MPA、PBD）、Docker隔离执行、AST静态检查、语义防火墙和加密历史认证等技术。

**📊 数据集**

使用RedCode安全场景数据集（25个场景、8个领域）和生成的57,000个多模态测试实例。

**📈 对比分析**

对比两种典型解释器（OpenInterpreter、OpenCodeInterpreter）与六个基础模型，发现安全设计与模型对齐共同决定安全基线，且高智能模型易受复杂提示攻击。

**⚠️ 局限性**

仅覆盖已知攻击方式，未探测零日攻击、侧信道泄露，且在不同硬件与部署环境下的泛化性尚需验证。

---

## 578. CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion

**arXiv ID:** 2602.18936 | [PDF](https://arxiv.org/pdf/2602.18936v1)

**作者:** Yu Li `[一作]` (George Washington University), Chi Zhang `[通讯]` (AGI Lab, Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出统一框架实现内容与风格的解耦与融合，结合低秩限制的Backbone微调、专家编码器和时间步可调的非对称CFG；

**💡 创新点**

①通过Rank‑Limited Backbone Fine‑Tuning实现内容/风格子空间分离；②专家编码器实现提示引导的细粒度控制；③无训练的时间步可调非对称CFG提升生成稳定性；

**🔧 技术方法**

LoRA、低秩投影、频域分解、对比式内容‑风格数据对、异步Classifier‑Free Guidance（ACFG）、Stable Diffusion XL；

**📊 数据集**

DreamBooth内容参考、StyleDrop风格集合以及自构造的10×10内容‑风格对比图像；

**📈 对比分析**

与Direct Merging、StyleDrop、ZipLoRA、KLoRA、BLoRA等基线在CLIP‑I内容/风格相似度、GPT‑4o组合分数以及用户评测中对比，内容0.79、风格0.80、组合0.83，用户评价82.5%/85%/87.5%；

**⚠️ 局限性**

依赖手工层级划分、对文本嵌入质量敏感、时间步校正略有计算开销；

---

## 579. A Relational Theory of Grounding and a new Grounder for SMT

**arXiv ID:** 2602.19102 | [PDF](https://arxiv.org/pdf/2602.19102v1)

**作者:** Pierre Carbonnelle `[一作]` `[通讯]` (Independent Researcher), Pierre Carbonnelle (Independent Researcher)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于关系代数的SMT-LIB定量公式去量化（grounding）框架，并实现了一个名为xmt-lib的grounder

**💡 创新点**

创新点在于：①通过x-generator实现对无限域量化的有限化；②将语法树逐层展开为关系操作，支持部分解释的符号；③将去量化直接生成可交互的SMT-LIB脚本，保持声明式建模的优势

**🔧 技术方法**

使用了关系代数与SQL（SQLite）作为底层执行引擎，结合Rust实现高效的查询与聚合；采用自定义聚合函数和递归查询来处理计数和递归定义

**📊 数据集**

在DIRT基准集（包含1050个模型扩展问题，22个子集）上进行评测

**📈 对比分析**

与Z3、clingo、IDP3、ManyWorlds、SLI等solver比较；总体而言，xmt-lib在大多数子集中显著提升了“ground-and-solve”时间，达到或接近clingo的性能，远优于原始Z3；但在某些子集（如TGSubset、stablemarriage）仍落后，主要受限于solver算法差异

**⚠️ 局限性**

局限性：仅支持SMT-LIB Core、Int、Real理论；未支持布尔、数组、字符串等子句；对大型数据库的依赖导致一定的I/O开销；对对称性、递归定义等仍需改进；对某些问题仍无法完全消除量化，导致求解器需自行处理

---

## 580. AI-Powered Conflict Management in Open RAN: Detection, Classification, and Mitigation

**arXiv ID:** 2602.19758 | [PDF](https://arxiv.org/pdf/2602.19758v1)

**作者:** Abdul Wadud `[一作]` (University College Dublin), Fatemeh Golpayegani `[通讯]` (University College Dublin)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个 AI 驱动的 Open RAN xApp 冲突管理框架，涵盖冲突检测、分类与缓解，并实现了合成冲突生成器 GenC 用于训练和评估模型。

**💡 创新点**

创新点在于：①将原生 AI 直接嵌入 O-RAN 架构，实现实时冲突预警；②构建可控参数共享、类别不平衡的合成冲突数据集 GenC；③使用 SMOTE-GNN 对不平衡数据进行强化，显著提升分类鲁棒性；④在 ns3-oran 仿真环境中验证 AI 方法比传统规则方法快 3.2 倍且准确率接近 100%。

**🔧 技术方法**

采用的技术包括图神经网络（GraphSAGE）、双向 LSTM、SMOTE 过采样、强化学习（PPO/MARL）用于冲突缓解、Open RAN 典型接口（E2、A1、O1）与 MLOps 流程、ns3-oran 模拟器以及 OpenCellID 地理拓扑。

**📊 数据集**

使用的数据集有：①GenC 生成的合成数据（5–50 个 xApp、100 万时间步，包含四类冲突标签）；②基于 Dublin 区域 OpenCellID 的 ns3-oran 仿真数据（13 个基站、117 台 UE，涵盖能量与移动性 xApp 的冲突场景）。

**📈 对比分析**

比较方法为规则基检测/分类、Bi-LSTM、GNN、SMOTE-GNN。结果显示：SMOTE-GNN 在所有冲突强度和 xApp 数量下都实现 100% 准确率；GNN 接近 99%，Bi-LSTM 98–99%；规则基方法 100% 准确但耗时最长。检测延迟方面，规则基约 0.41 ms，而 AI 方法约 0.12 ms，提升约 3.2 倍。宏 F1 分数方面，SMOTE-GNN 接近 100%，其余方法随冲突强度下降。

**⚠️ 局限性**

局限性包括：实验仅覆盖至 50 个 xApp，尚未验证数千 ICP/KPI 的规模化；GenC 数据生成与 GNN 邻域采样在大规模网络中计算成本上升；缺乏分布式或层级化图训练方案，未来需探索多节点训练与自适应学习机制以满足 6G 级别需求。

---

## 581. Efficient Multi-Party Secure Comparison over Different Domains with Preprocessing Assistance

**arXiv ID:** 2602.19604 | [PDF](https://arxiv.org/pdf/2602.19604v1)

**作者:** Kaiwen Wang `[一作]`, Ruichen Zhang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了首个基于被动代理商（dealer）支持的多方安全比较协议，包括在𝔽ₚ和ℤ₂ᵏ域上的LTBits和MSB提取；

**💡 创新点**

创新点在于利用代理商生成更丰富的相关随机性，重新设计比较协议以实现常数轮（𝔽ₚ）或O(logₙk)轮（ℤ₂ᵏ）在线复杂度，并以黑盒ABB+模型实现跨域、跨攻击模型的通用性；

**🔧 技术方法**

采用多项式构造、m输入AND门优化、扩展的ABB+模型以及B2A转换等技术；

**📊 数据集**

未使用具体机器学习数据集，而是通过在模拟LAN/WAN网络环境下对比Rabbit框架进行实验评估；

**📈 对比分析**

与Rabbit比较，实验表明在小批量比较或网络受限环境下可获得5.7×–19.4×的速度提升；在大批量比较或本地网络中，Rabbit在某些配置下仍优于本工作；

**⚠️ 局限性**

局限性包括：对大量比较操作时性能下降；需要部署可信的非参与代理商；对某些大字段或高安全等级下仍受传统开销影响。

---

## 582. Hiding in Plain Text: Detecting Concealed Jailbreaks via Activation Disentanglement

**arXiv ID:** 2602.19396 | [PDF](https://arxiv.org/pdf/2602.19396v1)

**作者:** Amirhossein Farzam `[一作]` (Duke University), Guillermo Sapiro `[通讯]` (Princeton University)

**通讯引用:** 64243 | [OpenAlex ID](https://openalex.org/A5025218580)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于激活层分离的语义因子拆解方法，检测隐藏式越狱攻击。

**💡 创新点**

创新点在于把攻击视为目标与框架的组合，并用自监督方法在冻结LLM中分离这两种语义因子，从而实现更通用、更高效的检测。

**🔧 技术方法**

核心技术包括ReDAct模块（自监督对比学习+正交正则+重构），以及基于拆解框架特征的异常检测方法FrameShield。

**📊 数据集**

使用从多种LLM安全数据集（如JailbreakBench、官方越狱集合）生成的对比目标-框架语料库共6,269条提示，构建了86k+正样本对。

**📈 对比分析**

与现有模型无关的JBShield、LlamaGuard、SelfEx、GradSafe等方法对比，FrameShield在多款LLM上实现了最高达≈0.94的准确率（F1≈0.93），显著优于对照组。

**⚠️ 局限性**

局限性包括：仅针对二元语义因子，缺乏多因子扩展；对极端低资源或新模型的适配需要更多训练；对极其隐蔽的框架变形仍可能产生误判。

---

## 583. Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition

**arXiv ID:** 2602.19316 | [PDF](https://arxiv.org/pdf/2602.19316v1)

**作者:** Alexandros Haliassos `[一作]` (NatWest AI Research), Stavros Petridis `[通讯]` (NatWest AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了USR 2.0框架，在半监督音视频语音识别中通过CTC驱动的教师强制和混合采样，实现一次前向传播即可生成CTC与注意力伪标签，显著提升了训练效率和鲁棒性。

**💡 创新点**

创新点在于用CTC输出直接驱动注意力解码器（消除自回归瓶颈），并采用混合采样降低训练时的暴露偏差，同时在伪标签阶段实现两分支耦合，提升对分布外长序列、噪声与跨域数据的鲁棒性。

**🔧 技术方法**

技术包括基于Transformer的音视频共享编码器、CTC与注意力双分支结构、CTC驱动的教师强制伪标签生成、混合采样（与自回归伪标签交替）、EMA教师更新、联合CTC-注意力损失以及Beam搜索解码。

**📊 数据集**

使用的数据集包括LRS3（低资源30h和高资源433h）、VoxCeleb2、AVSpeech、WildVSR、LibriSpeech等，且实验中把未标注数据作为伪标签来源。

**📈 对比分析**

与US、US-0.0、AV-HuBERT、BRAVEn等基线相比，USR 2.0在LRS3、LRS2、WildVSR、AVSpeech等多任务中取得了接近或超过State‑of‑the‑Art的WER，并且训练时间约减半、对长序列和噪声的鲁棒性提升显著。

**⚠️ 局限性**

局限性包括：混合采样比例需手动调节、对极端噪声或完全陌生语音仍可能产生不连贯的注意力伪标签、以及在极大模型规模下对显存与GPU资源仍有一定需求。

---

## 584. Descent-Guided Policy Gradient for Scalable Cooperative Multi-Agent Learning

**arXiv ID:** 2602.20078 | [PDF](https://arxiv.org/pdf/2602.20078v1)

**作者:** Shan Yang `[一作]` (National University of Singapore), Yang Liu `[通讯]` (National University of Singapore)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用可微解析模型引导的政策梯度方法，以解决协作多智能体学习中的跨智能体噪声问题。

**💡 创新点**

创新点在于构建噪声自由的每个智能体指导梯度，理论证明将梯度方差从Θ(N)降为O(1)，同时保持原协作游戏的稳态不变。

**🔧 技术方法**

使用可微解析模型、优势估计、MAPPO框架与梯度分解技术。

**📊 数据集**

在异构云资源调度基准（基于AWS实例类型的模拟器）上进行实验。

**📈 对比分析**

与MAPPO、IPPO以及贪心/随机基线对比，DG‑PG在5到200个智能体规模下实现10倍以上的收敛速度，样本复杂度与智能体数无关，且接近或超过最佳拟合基线性能。

**⚠️ 局限性**

仅适用于存在可微解析模型的协作问题，指导权重α需手工调度，且在最大规模（200个）仍存在微小性能差距。

---

## 585. DEEP: Docker-based Execution and Evaluation Platform

**arXiv ID:** 2602.19583 | [PDF](https://arxiv.org/pdf/2602.19583v1)

**作者:** Sergio Gómez González `[一作]` (PRHLT Research Center), Francisco Casacuberta `[通讯]` (ValgrAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个名为 DEEP 的 Docker 化端到端评估平台，自动执行、评估并可视化机器翻译和 OCR 模型，支持多种评估指标与统计聚类。

**💡 创新点**

将 Docker 容器化执行、art 统计显著性聚类和交互式可视化整合到同一工具中，使评估过程透明、可扩展且不依赖特定任务；首次在同一平台上提供自动化的指标计算与聚类分析。

**🔧 技术方法**

使用 Docker、Python、Streamlit、sacreBLEU、art 统计显著性检验以及多种指标（BLEU、TER、chrF、BEER、WER、Bag‑of‑words WER）进行实现。

**📊 数据集**

主要使用 WMT20 News 共享任务的英德测试集（1418 句）进行 MT 实验；OCR 任务使用相应的图像与文本文件。

**📈 对比分析**

通过在 DEEP 中运行八个前沿模型（Seed、MADLAD‑400、NLLB‑200、Opus‑MT、M2M‑100、mBART、T5、EuroLLM），对 BLEU、TER、chrF 等指标进行评估，利用统计聚类区分模型显著性，并通过可视化展示速度‑质量关系；结果表明 Seed 在 BLEU 上最佳且最快，聚类显示各模型在不同指标上均存在显著差异。

**⚠️ 局限性**

目前仅支持 MT 与 OCR，评估指标有限；实验仅针对单一数据集，样本规模小；需手动 Docker 化模型，对 OOD、偏差与公平性等方面的支持不足。

---

## 586. ChordEdit: One-Step Low-Energy Transport for Image Editing

**arXiv ID:** 2602.19083 | [PDF](https://arxiv.org/pdf/2602.19083v1)

**作者:** Liangsi Lu `[一作]` (Guangdong University of Technology), Yang Shi `[通讯]` (Guangdong University of Technology)

**通讯引用:** 78389 | [OpenAlex ID](https://openalex.org/A5100361956)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练‑无关、反演‑无关的单步图像编辑框架ChordEdit，利用稳健的控制场实现高保真、实时的文本导向编辑。

**💡 创新点**

创新点在于把编辑问题视作动态最优传输，构造低能量的Chord Control Field（时域平均的控制场），从而消除单步模型中漂移差异的高能量、噪声导致的不稳定性。

**🔧 技术方法**

技术手段包括：基于动态最优传输的控制场推导、时间滑动平均（Chord Control Field）、可选的近端细化步骤、以及在单步扩散/流式模型上的无监督、无反演调用。

**📊 数据集**

实验使用PIE‑Bench 512×512图像编辑基准数据集，包含700个样本和10类编辑任务。

**📈 对比分析**

与多步、少步和单步编辑器（如FlowEdit、TurboEdit、SwiftEdit等）对比，ChordEdit在PSNR、CLIP‑Whole/Edited、Runtime、VRAM等指标上实现了最优或同等性能，单步实现时间约0.38 s，内存占用约6988 MiB，且无模型专属训练。

**⚠️ 局限性**

局限性：仅适用于已蒸馏的单步生成模型；对极端或复杂的编辑指令可能仍受限于原模型的表达能力；在极大尺寸或高分辨率下的稳定性尚未彻底验证。

---

## 587. Adaptive Multi-Agent Reasoning for Text-to-Video Retrieval

**arXiv ID:** 2602.19040 | [PDF](https://arxiv.org/pdf/2602.19040v1)

**作者:** Jiaxin Wu `[一作]` (Shenzhen University), Qing Li `[通讯]` (The Hong Kong Polytechnic University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种自适应多智能体框架，用来解决文本到视频检索中的跨模态、时间推理和逻辑约束难题；

**💡 创新点**

核心创新在于：①将检索、推理、查询改写三个专门智能体与一个LLM驱动的编排智能体动态协同；②引入检索性能记忆库与历史推理轨迹，提升查询改写和智能体协调的决策；③通过多轮迭代实现探索与利用的平衡，实现零样本时空推理；

**🔧 技术方法**

使用跨模态检索模型（IITV）、多模态大语言模型 Qwen3‑VL‑8B‑Instruct 进行推理与改写、编排，配合检索性能记忆库；

**📊 数据集**

在三大 TRECVid Ad‑hoc Video Search 公开数据集（IACC.3、V3C1、V3C2）共210条查询上进行评测；

**📈 对比分析**

与现有 30+ 先进方法（CLIP、CLIP4Clip、BLIP‑2、InternVid、GLSCL、GenSearch、IITV 等）对比，xinfAP 提升 2.2 倍，特别是在复杂查询子集上分别提升 86.6%‑100%；

**⚠️ 局限性**

主要限制包括：依赖单一大型 LLM 进行推理与编排，计算成本仍高；对极端长视频或多段式查询的支持有限；在记忆库与决策机制的参数调优上仍需进一步研究。

---

## 588. The Digital Gorilla: Rebalancing Power in the Age of AI

**arXiv ID:** 2602.20080 | [PDF](https://arxiv.org/pdf/2602.20080v1)

**作者:** M. Alejandra Parra-Orlandoni `[一作]` (Harvard University), Christopher J. Mallet `[通讯]` (Harvard University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将 AI 视为第四类社会主体（数字大猩猩），构建四社会主体模型与五种权力模态，用以分析 AI 治理与权力再分配。

**💡 创新点**

创新点在于突破传统技术类比框架，将 AI 定义为具备独立权力与合法性的社会主体，并提出基于联邦化与多中心治理的制度设计方案。

**🔧 技术方法**

采用政治学、宪法理论与制度设计方法构建概念框架，无具体实验技术。

**📊 数据集**

本文为理论综述，无使用特定数据集。

**📈 对比分析**

未进行定量比较或性能评估，仅提供概念性分析与案例阐释。

**⚠️ 局限性**

局限性：仅适用于自由民主制度，缺乏经验验证与实证检验，理论推断未得到数据或实验支持。

---

## 589. "The explanation makes sense": An Empirical Study on LLM Performance in News Classification and its Influence on Judgment in Human-AI Collaborative Annotation

**arXiv ID:** 2602.19690 | [PDF](https://arxiv.org/pdf/2602.19690v1)

**作者:** Qile Wang `[一作]` (University of Delaware), Matthew Louis Mauriello `[通讯]` (University of Delaware)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大型语言模型（LLM）在对美国新闻内容进行左、中、右倾向分类的可行性，并评估其生成的解释对人类标注者的判断与信心影响。

**💡 创新点**

①在新闻偏见分类任务中实现与监督模型相当甚至略优的性能；②首次系统比较不同长度解释（简短 vs 详细）对人类决策和信心的影响；③提出基于主题分析的AI解释改进建议。

**🔧 技术方法**

使用OpenAI GPT‑4（包括GPT‑4o、GPT‑4o‑mini）并进行提示工程；通过人机协作实验（MTurk）收集人类标注与自我报告信心；使用主题分析对开放式回答进行定性评估。

**📊 数据集**

AllSides 历史数据集（34,737 篇）与自采集的近期数据集（17,166 篇），新闻摘要平均77词；采用出版者层级的政治倾向标签。

**📈 对比分析**

与监督学习基准（BERT、Fine‑tuned BERT 等）比较，GPT‑4 在全文上达约68%准确率、F1≈0.69，略低于最优监督模型；在摘要上约62%准确率；在人机协作中，详细解释比简短解释更能提升人类信心与决策改变，但亦导致部分误导。

**⚠️ 局限性**

①样本文本短缺乏上下文，可能导致分类与解释不完整；②仅使用出版者层级标签，未考察文章级别偏差；③研究局限于美国左‑中‑右三分法；④实验仅评估OpenAI GPT模型，结果对其他LLM推广性有限；⑤受试者样本规模与多样性受限。

---

## 590. Can Large Language Models Replace Human Coders? Introducing ContentBench

**arXiv ID:** 2602.19467 | [PDF](https://arxiv.org/pdf/2602.19467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 591. Detecting labeling bias using influence functions

**arXiv ID:** 2602.19130 | [PDF](https://arxiv.org/pdf/2602.19130v1)

**作者:** Frida Jørgensen `[一作]` (Copenhagen University), Siavash Bigdeli `[通讯]` (Technical University of Denmark)

**通讯引用:** 584 | [OpenAlex ID](https://openalex.org/A5024315163)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

利用影响函数（Influence Function）检测训练数据中的标签偏差，定位错误标注样本

**💡 创新点**

首次将影响函数应用于标注偏差检测，提供了一种通过样本重要性分数来识别误标的可解释方法

**🔧 技术方法**

影响函数、梯度与Hessian近似（对角线/最后一层）、逆Hessian向量积、梯度计算与阈值分数

**📊 数据集**

MNIST（二分类子集）与CheXpert（胸部X光）

**📈 对比分析**

在MNIST上通过阈值化影响分数，误标样本检出率约90%且误报率低于1%；在CheXpert上误标样本亦显示更高影响分数，但区分度下降，检出率随阈值上升而递减

**⚠️ 局限性**

对大模型的Hessian近似（仅最后一层）导致精度下降；模型复杂度、数据不平衡与噪声分布不均导致在医学影像上效果不如MNIST，需改进Hessian估计和模型训练

---

## 592. Strengths and Limitations of Greedy in Cup Games

**arXiv ID:** 2602.18610 | [PDF](https://arxiv.org/pdf/2602.18610v1)

**作者:** Kalina Jasińska `[一作]` (University of Cambridge), Gyudong Lee `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了杯子游戏及其变体（竹园修剪、固定速率杯子游戏）中贪心算法的极限，证明贪心在竹园修剪中并非最优并给出下界 2.076，同时提出一种混合算法在三种游戏中实现最优或近优性能。

**💡 创新点**

创新点包括：①首次给出贪心在竹园修剪问题中的下界 2.076；②引入半观测模型并给出贪心的匹配上下界；③提出 Deadline‑Driven/Greedy Hybrid 混合算法，在竹园修剪和固定速率杯子游戏中达到最优 backlog，且在可变速率杯子游戏中仅差 1；④在半观测冲洗游戏中证明贪心达到 2^{Θ(√log n)} 的子多项式下界。

**🔧 技术方法**

主要技术包括潜在函数分析、乘法不等式与阶乘/伽马函数估计、递推不变式、对数与指数函数逼近以及对数和幂级数的精细计数。

**📊 数据集**

论文为理论性工作，没有使用实际数据集；实验验证采用随机生成的速率分布，证明了下界的可行性。

**📈 对比分析**

与以往贪心算法在杯子游戏中已知的 O(log n) 结果对比，混合算法在竹园修剪和固定速率杯子游戏中实现 backlog 2 或 H(n)+2，半观测模型中贪心实现 Θ(n^{c-1}/c) 或 2^{Θ(√log n)} 的匹配上下界。

**⚠️ 局限性**

局限性包括：仅考虑确定性对手；半观测模型中的常数 c 对结果影响显著；随机算法的性能未完整分析；对于更大规模的竹子分布，进一步下界的证明仍是开放问题。

---

## 593. Controlled Face Manipulation and Synthesis for Data Augmentation

**arXiv ID:** 2602.19219 | [PDF](https://arxiv.org/pdf/2602.19219v1)

**作者:** Joris Kirchner `[一作]` (Vicarious Perception Technologies), Chirag Raman `[通讯]` (Delft University of Technology)

**通讯引用:** 51 | [OpenAlex ID](https://openalex.org/A5044787452)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预训练的Diffusion Autoencoder的语义潜空间中，通过学习线性编辑方向实现对面部表情中Action Unit的可控、低耦合编辑，并将这些编辑用于生成均衡标签数据。

**💡 创新点**

创新点在于引入依赖感知条件和正交投影来抑制AU之间以及与无关属性的共激活，实现绝对AU编辑；同时通过中性化步骤消除原始表情，实现精准数据增强。

**🔧 技术方法**

采用Diffusion Autoencoder、线性回归/SVM编辑方向、依赖感知条件、正交投影、中性化神经网络、MobileNetV3-Small AU检测器、FaceReader评估等技术。

**📊 数据集**

使用DISFA、DISFA+、FEAFA、BP4D、CelebA以及FFHQ等公开人脸数据集进行训练与评估。

**📈 对比分析**

与StyleGAN‑NADA、StyleAU、MagicFace、逆频率重加权、NNCLR自监督预训练等方法比较，AU检测F1平均提升约10–15%，误报率下降，编辑质量MAE最低，身份保持最佳。

**⚠️ 局限性**

局限在于需依赖预训练面部生成器、无法完整覆盖稀有AU组合、对光照、姿态等外部因素仍存在潜在敏感性，且方法性能受生成器质量限制。

---

## 594. The Convergence of Schema-Guided Dialogue Systems and the Model Context Protocol

**arXiv ID:** 2602.18764 | [PDF](https://arxiv.org/pdf/2602.18764v1)

**作者:** Andreas Schlapbach `[一作]` `[通讯]` (Swiss Federal Railways), Andreas Schlapbach (Swiss Federal Railways)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了 Schema‑Guided Dialogue 与 Model Context Protocol 的统一视角，提出了面向 LLM 代理的五大 schema 设计原则，并分析了两者在实践中的互补与不足。

**💡 创新点**

创新点在于将 SGDs 的零样本推理能力与 MCP 的标准化协议结合，识别并补全了失败模式、工具关系、逐步披露等关键缺口，并提出了系统化的 schema 设计规范，为软件 3.0 提供了治理框架。

**🔧 技术方法**

使用了自然语言 schema 解析、JSON‑RPC 交互、零样本推理、token 量化与主动发现技术，以及多代理架构 COMPASS 等实现手段。

**📊 数据集**

主要数据集包括 SGD 数据集（20,000+ 交互，20 领域）以及在 MCP 生态中构建的 10+ 代理、1000+ 工具依赖的内部数据。

**📈 对比分析**

通过与 MCP‑Universe、MCPAgentBench 等基准对比，发现即使最先进模型（如 GPT‑5‑High）任务完成率也仅在 44% 左右，验证了工具选择与 schema 质量对性能的显著影响。

**⚠️ 局限性**

局限性包括缺乏对五大原则的系统化实验验证、schema 版本兼容与演进机制不足、token 限制导致的上下文消耗以及长周期任务下的可靠性与可持续性仍未完全解决。

---

## 595. Deep LoRA-Unfolding Networks for Image Restoration

**arXiv ID:** 2602.18697 | [PDF](https://arxiv.org/pdf/2602.18697v1)

**作者:** Xiangming Wang `[一作]` (Harbin Institute of Technology), Yongyong Chen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 3996 | [OpenAlex ID](https://openalex.org/A5031480448)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种通用的深度低秩适配（LoRA）展开网络 LoRun，利用共享的预训练去噪器与轻量级 LoRA 适配器实现多阶段图像恢复。

**💡 创新点**

创新点在于：①将所有阶段的去噪模块共享同一个高质量基准去噪器，②在每个阶段注入低秩 LoRA 适配器来实现阶段特定的去噪强度调节，③将预训练与 LoRA 微调分离，显著降低参数量和显存占用，同时保持或提升性能。

**🔧 技术方法**

使用的技术包括：深度展开网络（Deep Unfolding Networks）、低秩适配（LoRA）技术、梯度下降模块（GDM）与近似映射模块（PMM）等，训练策略分为基准去噪器预训练与 LoRA 微调两步。

**📊 数据集**

实验数据集包括：CAVE + KAIST 进行 CASSI 复原；COCO 训练、General100、Set11、Set14 进行压缩感知；DIV2K、Flickr2K 训练、BSD68、Set5 进行超分辨率。

**📈 对比分析**

与多种 SOTA 方法（如 GAP‑Net、DAUHST、RCUMP、Block‑K 等）以及传统 Block‑K 方案比较，LoRun 在 PSNR/SSIM 上保持同等或更优，同时参数量降低约 30–70%，显存使用减少 30–60%，推理时间无显著增加。

**⚠️ 局限性**

局限性：需要为每个任务训练专门的基准去噪器（预训练），LoRA 的秩超参数需要经验调优，且在极端噪声或极高压缩率下性能提升有限。

---

## 596. SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation

**arXiv ID:** 2602.19213 | [PDF](https://arxiv.org/pdf/2602.19213v1)

**作者:** Yujie Lu `[一作]` (Sichuan University), Junlong Cheng `[通讯]` (Sichuan University)

**通讯引用:** 812 | [OpenAlex ID](https://openalex.org/A5024180326)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了SegMoTE，一种在保持SAM“随处可分割”能力的前提下，通过 token‑级混合专家（MoE）和逐步提示标记化（PPT）实现多模态医学图像分割的框架，且仅需要17M可学习参数；

**💡 创新点**

创新点包括：①仅冻结SAM编码器，添加轻量级MoE实现对不同影像模态的自适应专家路由；②设计PPT机制，使用随机掩码/文本提示生成自适应提示 token，从而在二分类任务中实现无人工提示的自动分割；③构建高质量但规模极小的MedSeg‑HQ数据集，实现高效训练与强泛化；

**🔧 技术方法**

主要技术包括：混合专家（MoE）与噪声 Top‑k 路由、负载平衡损失、进步提示标记化、Dice 损失、冻结 SAM 编码器的轻量级适配；

**📊 数据集**

使用了MedSeg‑HQ（12 个公开数据集，约 15.5 万高质量标注），并在 ISLES、SegThor、TotalSegmentator 等外域数据集上进行评估；

**📈 对比分析**

通过与 SAM、SAM‑2、MedSAM、IMIS 等基线在点/框提示下的 Dice 评价对比，SegMoTE 在二分类任务上提升 7% 以上，在多分类任务上提升 1‑2%，在无人工提示的自动分割中也表现优于传统交互方法；

**⚠️ 局限性**

局限性在于：PPT 仅适用于二分类（单目标）任务；目前仅针对 2D 图像；需要进一步验证在 3D 医学影像及医学视频上的可迁移性；

---

## 597. INSURE-Dial: A Phase-Aware Conversational Dataset \& Benchmark for Compliance Verification and Phase Detection

**arXiv ID:** 2602.18448 | [PDF](https://arxiv.org/pdf/2602.18448v1)

**作者:** Shubham Kulkarni `[一作]` (Interactly), Shiva Chaitanya `[通讯]` (Interactly)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出INSURE-Dial公共基准，用于评估保险福利核查电话的阶段检测与合规验证

**💡 创新点**

首次将阶段级、顺序敏感的合规审核框架（OIP‑SCE）量化为可测评任务，并构建公开数据集

**🔧 技术方法**

结合LLM、ASR、自动生成与规则校验的技术实现

**📊 数据集**

包含50个真实脱敏呼叫和1000个程序校验合规的合成呼叫

**📈 对比分析**

通过对比多种LLM（Gemini、GPT‑4系列等）在阶段边界检测和合规判定任务上的EM/F1和调用级精度，显示边界检测是瓶颈，合规判定在边界准确时可达≈90%

**⚠️ 局限性**

受限于少量真实样本、仅英文、最多两药检查、依赖ASR转写且未公开原始音频

---

## 598. Visual Prompt Guided Unified Pushing Policy

**arXiv ID:** 2602.19193 | [PDF](https://arxiv.org/pdf/2602.19193v1)

**作者:** Hieu Bui `[一作]` (Ritsumeikan University), Joo-Ho Lee `[通讯]` (Ritsumeikan University)

**通讯引用:** 3896 | [OpenAlex ID](https://openalex.org/A5100695583)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的推送策略，能够在位移、分组、单独推送等三种非抓取任务中实现多模态推送；

**💡 创新点**

创新点包括：将视觉提示+任务指令融入流匹配网络，实现轻量级提示引导的多模态推送；通过统一模型实现跨任务的正迁移；在VLM规划框架中将其作为低层原语，展示多对象抓取的可重用性；

**🔧 技术方法**

主要技术：基于流匹配的生成式策略，使用DiT（Diffusion Transformer）网络；视觉输入通过ResNet34提取并与机器人本体信息拼接；采用Transformer编码器捕获时间序列；使用分类器无条件引导（CFG）提升提示遵循；

**📊 数据集**

数据集：自建550条人类演示数据（位移200、分组200、单独150），对象为3D打印长方体；对未见物体（115×34×34mm盒子）进行泛化测试；

**📈 对比分析**

与单任务流匹配模型和目标图像（goal-image）条件模型对比：在位移+分组任务中统一模型提升10%（85%/70% vs 75%/60%），在单独任务相当或略优（65% vs 70%），并且在混乱场景下显著优于目标图像（如3物体位移70% vs 40%）；

**⚠️ 局限性**

局限性：提示点位置对性能高度敏感；手腕视角固定，缺乏视角不变提示；抓取时易失去接触导致失败；缺乏物理动态建模；VLM规划受空间推理限制，需更鲁棒的高层规划。

---

## 599. The Climate Change Knowledge Graph: Supporting Climate Services

**arXiv ID:** 2602.19786 | [PDF](https://arxiv.org/pdf/2602.19786v1)

**作者:** Miguel Ceriani `[一作]` (CNR Institute of Cognitive Sciences and Technologies), Andrea Giovanni Nuzzolese `[通讯]` (CNR Institute of Cognitive Sciences and Technologies)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个气候变化知识图谱（Climate Change Knowledge Graph），将多来源的气候模型模拟元数据、情景、变量、时空分辨率等信息统一集成，支持复杂查询以服务气候决策。

**💡 创新点**

采用基于Ontology Design Patterns的模块化极限设计方法，设计了核心气候服务本体（CCSO）与数据本体，实现对模型、情景、变量、分辨率等跨数据源语义统一；并公开发布本体与数据，形成可互操作的知识图谱。

**🔧 技术方法**

使用OWL本体建模、RDF、RML映射、SPARQL查询与更新、Apache Jena Fuseki、GraphDB、LodView、Docker化部署，配合Python、pyRML、jq等脚本实现数据映射与图谱构建与部署。

**📊 数据集**

集成CMIP5/CMIP6模拟元数据、ESGF、NetCDF、CF元数据、CMOR表、CORDEX领域与数据、Climdex指数等多种公开数据源的元数据信息。

**📈 对比分析**

通过预先设计的Competency Questions（CQ）和相应的SPARQL查询验证本体与图谱的完整性；在约1400万三元组的图谱上执行CQ，查询返回准确结果，性能满足当前评估需求，具体响应时间未公开但已通过实验验证可接受。

**⚠️ 局限性**

仅集成元数据而非完整数值内容，覆盖范围有限（如IAMs未集成）；查询性能未针对高并发或大规模使用进行系统化评估；对气候服务流程层面的描述仍在完善中，可能导致部分应用场景缺失；依赖手工映射与脚本，存在维护成本与错误风险。

---

## 600. Sparse Masked Attention Policies for Reliable Generalization

**arXiv ID:** 2602.19956 | [PDF](https://arxiv.org/pdf/2602.19956v1)

**作者:** Caroline Horsch `[一作]` (Delft University of Technology), Wendelin Böhmer `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种稀疏掩蔽注意力（Sparse Masked Attention）策略，利用学习的注意力权重掩蔽实现信息去除并提升对未见任务的泛化

**💡 创新点**

将掩蔽函数直接嵌入自注意力网络，通过路径矩阵和稀疏正则化实现可解释且可泛化的稀疏化处理，克服传统输入掩蔽在未见样本上泛化差的问题

**🔧 技术方法**

自注意力网络、路径矩阵稀疏化、硬Gumbel‑Softmax采样、稀疏正则化损失、CNN特征提取与二维位置编码

**📊 数据集**

Procgen基准（16个程序化生成的游戏环境）

**📈 对比分析**

与CNN+MLP、密集注意力网络以及传统输入掩蔽注意力进行对比；在大多数游戏中稀疏掩蔽注意力在未见任务上的归一化回报显著优于其它基线，尤其在稀疏依赖策略的环境中提升最大

**⚠️ 局限性**

对稀疏比例α的选择敏感，且假设观测特征局部可分离；在依赖全局或高度耦合特征的任务中可能效果受限

---

## 601. Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis

**arXiv ID:** 2602.19756 | [PDF](https://arxiv.org/pdf/2602.19756v1)

**作者:** Junhyeok Choi `[一作]` (Pohang University of Science and Technology), Minwoo Chae `[通讯]` (Pohang University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个学习‑自由的多模态数据集蒸馏框架PDS，通过CLIP提取图文对齐的特征，聚类并线性匹配得到图像与文本原型，随后利用unCLIP解码器从图像原型生成新图像，生成紧凑但语义丰富的图文蒸馏数据集。

**💡 创新点**

创新点包括①无需任何模型训练或微调即可完成蒸馏；②通过CLIP原型与unCLIP解码器实现跨模态的高质量图像生成；③生成的蒸馏数据集对不同视觉后端具有良好的泛化性；④在效率（内存、时间）和性能上优于传统优化式蒸馏方法。

**🔧 技术方法**

主要技术手段包括CLIP图像与文本编码器、mini‑batch k‑means聚类、Hungarian线性分配匹配、unCLIP解码器（基于Diffusion的图像生成），以及无监督的对齐与检索策略。

**📊 数据集**

实验数据集为Flickr30K和MS‑COCO这两大视觉‑文本基准。

**📈 对比分析**

与TESLA‑VL、LoRS等优化式多模态蒸馏、K‑center、Herding、CLIP‑score、LAION‑filtering等子集选择方法，以及D4M/MGD3等学习‑自由图像分类蒸馏方法进行比较。PDS在极小蒸馏集（100/300对）下，在IR@1/5/10、TR@1/5/10等检索指标上均明显优于所有基线；在不同视觉后端的交叉架构测试中表现最稳健；同时生成过程内存仅4.3 GB、耗时9.7 s/图像，远低于基线的6.1 GB、1478 s。

**⚠️ 局限性**

主要局限包括：①依赖CLIP与unCLIP这类在自然图像上预训练的模型，若迁移到医学等专业域需再训练；②原型聚类可能受频繁出现的概念主导，导致稀有或长尾类别在蒸馏集中的表现不足；③当前尚未实现对更强对齐模型（如SigLIP）的直接条件化，限制了潜在的对齐质量提升。

---

## 602. UniRank: A Multi-Agent Calibration Pipeline for Estimating University Rankings from Anonymized Bibliometric Signals

**arXiv ID:** 2602.18824 | [PDF](https://arxiv.org/pdf/2602.18824v1)

**作者:** Pedram Riyazimehr `[一作]` (NotionWave), Seyyed Ehsan Mahmoudi `[通讯]` (NotionWave)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 UniRank 多智能体 LLM 流水线，利用公开的文献计量数据估算大学在全球排名中的位置，并通过匿名化与数据隐藏防止模型记忆。

**💡 创新点**

创新点包括三阶段多智能体架构、严格的匿名化与隐藏协议、记忆指数（MI）评估、工具增强的校准流程以及可扩展到多系统的评估框架。

**🔧 技术方法**

采用 GPT‑5.2 与 ReAct/Tool‑augmented 机制，基于 MAgICoRe 的多代理迭代思考，结合 OpenAlex/Semantic Scholar 的数据抓取、Z‑score 归一化及统计指标计算。

**📊 数据集**

使用的主要数据集为 OpenAlex 与 Semantic Scholar 的公开元数据，QS、THE、ARWU 的公开排名 CSV，以及经过分层抽样得到的 500 所大学的测试样本。

**📈 对比分析**

评估采用留一隐藏+匿名化的双盲协议，按等级分层测试；结果显示 MAE=251.5、Spearman ρ=0.769、hit@100=39.8%、记忆指数 MI=0，性能随排名级别从精英到尾部逐步下降。

**⚠️ 局限性**

局限性包括仅依赖研究指标，无法捕捉声誉、教学、行业等重要维度导致正向偏差；单一系统（THE）评估、样本规模有限、模型依赖性与结果不确定性。

---

## 603. Assessing the Reliability of Persona-Conditioned LLMs as Synthetic Survey Respondents

**arXiv ID:** 2602.18462 | [PDF](https://arxiv.org/pdf/2602.18462v1)

**作者:** Erika Elizabeth Taday Morocho `[一作]` (National Research Council of Italy), Stefano Cresci `[通讯]` (National Research Council of Italy)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估多属性人物化提示对大语言模型（LLM）作为合成调查受访者的可靠性，采用匹配对照设计将人物化与无人物化（vanilla）对比；

**💡 创新点**

创新点在于通过严格的匹配实验只添加多属性人物化提示，细粒度地测量各问卷项和各子组的影响，揭示人物化提示的异质性与潜在风险；

**🔧 技术方法**

使用两种开源聊天模型（如LLaMA系列）与硬/软相似度（Normalized Match Distance）评估指标，并加入随机猜测基线；

**📊 数据集**

基于美国版World Values Survey第七波（WVS‑7）微观数据，选取31条问卷项进行实验；

**📈 对比分析**

与无人物化和随机猜测基线对比，整体硬相似度与软相似度均显著优于随机基线，但人物化提示未产生显著的整体提升；在个别问卷项与低样本子组中，人物化提示甚至导致性能下降；

**⚠️ 局限性**

局限性包括：仅覆盖美国WVS样本与31条问卷，未考虑多语言或其他国家；仅评估两大模型、单一prompt与解码策略；软相似度对顺序假设敏感；人物化属性有限，可能产生刻板化；未校正样本权重，且未评估多次生成方差。

---

## 604. How Ten Publishers Retract Research

**arXiv ID:** 2602.19197 | [PDF](https://arxiv.org/pdf/2602.19197v1)

**作者:** Jonas Oppenlaender `[一作]` (University of Oulu), Jonas Oppenlaender `[通讯]` (University of Oulu)

**通讯引用:** 942 | [OpenAlex ID](https://openalex.org/A5090875146)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

**🎯 论文内容**

对1997–2026年十大出版商共计46,087篇撤稿记录进行跨出版商的量化比较研究。

**💡 创新点**

首次系统比较十大出版商的撤稿文化，揭示规范差异、突出ACM的异常撤稿特征，并探讨其高阈值与暗存档对科研诚信的影响。

**🔧 技术方法**

使用Retraction Watch数据库与Dimensions API获取数据，结合归一化、HHI、熵、置换检验与卡方检验等统计技术进行分析。

**📊 数据集**

Retraction Watch撤稿数据库（46,087条记录）及Dimensions提供的出版物计数、作者识别信息。

**📈 对比分析**

通过描述性统计、归一化撤稿率、原因分布的HHI/熵、置换检验和卡方检验进行比较；结果显示出版社间存在显著差异，ACM表现为极度集中且时序异常。

**⚠️ 局限性**

数据完整性受限（Retraction Watch覆盖不全、ACM暗存档导致信息缺失）、理由归一化主观性、样本量差异可能引入统计偏差。

---

## 605. From "Help" to Helpful: A Hierarchical Assessment of LLMs in Mental e-Health Applications

**arXiv ID:** 2602.18443 | [PDF](https://arxiv.org/pdf/2602.18443v1)

**作者:** Philipp Steigerwald `[一作]` (Technische Hochschule Nürnberg), Jens Albrecht `[通讯]` (Technische Hochschule Nürnberg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对11种大型语言模型（含开源与专有）在德语心理咨询邮件中生成六词主题行，并通过层级评估（先分类为Good/Fair/Poor，再在类别内排名）进行质量评估。

**💡 创新点**

首次系统引入层级评估与人机评估相结合的框架，利用人工与AI评估者共九位，并在分类后加入排名以解决分类上限效应；同时探讨语言特定微调与量化对模型性能的影响。

**🔧 技术方法**

使用大型语言模型生成任务、JSON结构化输出、prompt工程、Krippendorff's α、Spearman、Kendall、Pearson等统计方法进行多维度评估；对AI评估器采用同一评估提示。

**📊 数据集**

23条合成的德语心理咨询邮件线程，共生成253条主题行；评估者对每条主题行进行9份评测。

**📈 对比分析**

通过比较Good比例、排名平均值以及统计相关性，GPT‑4o与GPT‑3.5 Turbo以73% Good和低平均排名（3.21、3.46）表现最佳；最佳开源模型SauerkrautLM Llama 3 70b Q4与Mixtral 8×7b Q8以54% Good排名位列第二；语言微调显著提升，量化效果因模型而异。

**⚠️ 局限性**

数据仅为合成线程，样本量有限；过滤后仅保留45.8%评测，可能导致代表性下降；人机评估一致性不完全；高性能模型仍有约27%不佳输出；隐私与性能的权衡未完全解决。

---

## 606. Distributed and Consistent Multi-Robot Visual-Inertial-Ranging Odometry on Lie Groups

**arXiv ID:** 2602.19173 | [PDF](https://arxiv.org/pdf/2602.19173v1)

**作者:** Ziwei Kang `[一作]` (North China Electric Power University), Yizhi Zhou `[通讯]` (George Mason University)

**通讯引用:** 152 | [OpenAlex ID](https://openalex.org/A5101115202)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于右不变误差Lie群的分布式协作视觉–惯导–UWB里程计（DC‑VIRO），能够在GPS受限环境下同时估计多机器人位置、姿态以及UWB锚点位置。

**💡 创新点**

创新点包括：
- 将UWB锚点位置直接纳入状态实现自标定；
- 利用共享锚点和共视视觉特征提供额外几何约束；
- 在分布式框架下采用协方差交叉法（Covariance Intersection）处理跨机器人不相关信息；
- 通过右不变误差保证与单机VIO相同的四个不可观测方向，从而实现估计一致性。

**🔧 技术方法**

主要技术手段：
- 多状态约束Kalman滤波（MSCKF）
- Lie群右不变误差EKF
- UWB测距观测模型与视觉投影模型
- 交叉协方差合并
- 统一的系统状态线性化与协方差传播

**📊 数据集**

实验数据集为仿真场景：3台机器人、3个UWB锚点，采用两条预设轨迹（a、b），每条轨迹进行100次Monte‑Carlo仿真。

**📈 对比分析**

与单机不共享信息的VIO做对比。结果显示加入跨机器人信息后，位置RMSE从0.205 m/1.788°下降到0.147 m/1.295°（轨迹a），
0.498 m/1.575°下降到0.256 m/1.027°（轨迹b），显著提升定位精度。

**⚠️ 局限性**

局限性：
- 仅在仿真中验证，未在真实机器人平台上测试；
- 假设通信成功率70%，未考虑时延、丢包等实际网络问题；
- 只对单锚点情况进行建模，扩展到多锚点需进一步验证；
- 需要预先获取IMU到UWB标定信息，未讨论同步与实时性问题。

---

## 607. Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training

**arXiv ID:** 2602.19580 | [PDF](https://arxiv.org/pdf/2602.19580v1)

**作者:** Jeremy McEntire `[一作]` `[通讯]`, Jeremy McEntire

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Leap+Verify框架，利用可验证的推测式权重预测加速神经网络训练；

**💡 创新点**

创新点在于三阶段训练状态检测（混沌、过渡、稳定）与可验证的推测式权重预测结合，发现动量预测普遍失效，且大型模型训练时可预测阶段稀缺；

**🔧 技术方法**

使用激活空间余弦相似度做状态检测，线性/二次有限差分预测权重，验证通过保持集损失对比；

**📊 数据集**

在WikiText-103语料上训练GPT‑2 124M与Qwen 2.5‑1.5B两大语言模型；

**📈 对比分析**

与无条件预测方法对比，线性/二次预测在非混沌阶段可达9–37%严格接受率，接近率高达99–100%，而动量预测则在两种规模下失效100‑10,000×；

**⚠️ 局限性**

局限在于仅训练2000步，1.5B模型几乎不进入稳定阶段，导致可验证推测机会稀少；

---

## 608. Thin Plate Spline Surface Reconstruction via the Method of Matched Sections

**arXiv ID:** 2602.19182 | [PDF](https://arxiv.org/pdf/2602.19182v1)

**作者:** Igor Orynyak `[一作]` (National Technical University of Ukraine Igor Sikorsky Kyiv Polytechnic Institute), Danylo Tavrov `[通讯]` (National Technical University of Ukraine Igor Sikorsky Kyiv Polytechnic Institute)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了“匹配截面法”（MMS）用于高精度、能量最优的曲面重建与混合。

**💡 创新点**

创新点在于：1) 将二维偏微分方程分解为可用传递矩阵求解的一维方向方程，实现了全域的 C² 连续；2) 通过在角点和内部点处引入力学约束（弹性弯矩、剪力），实现了物理一致且美观的曲面；3) 提出了基于局部正则化核的三阶导数平滑技术，解决点约束导致的奇异性；4) 显示MMS在多种边界数据（一次导数、二次导数）下均能快速收敛到最优解。

**🔧 技术方法**

采用薄板弯曲理论、薄板能量泛函（双调和方程）以及传递矩阵方法（TMM）构建一维ODE系统；对二维网格做数值积分计算能量；利用矩阵求解（线性方程组）完成整个域的求解。

**📊 数据集**

使用一系列合成测试函数（余弦型双调和、非对称双调和、余弦平面、多峰曲面）和随机内部点约束作为验证数据；未使用公开工业或现实世界数据集。

**📈 对比分析**

通过与已知解析解或高精度参考结果对比，MMS在不同网格分辨率下均能在几百个元素以内达到几百份之一的误差，能量值始终低于或等于原始函数的薄板能量；相较传统有限元（C⁰ 或 C¹）方法，MMS 维持 C² 连续并显著提升表面平滑度。性能上，求解时间与二维 FEM 相当，且不出现高阶导数不连续。

**⚠️ 局限性**

局限性：1) 目前仅在矩形网格上实现，需进一步推广至任意网格；2) 在角点或内部点强约束时会产生三阶导数不连续，需要正则化；3) 对于非常细密或复杂几何，矩阵规模增大，计算成本提升；4) 方法假设薄板弹性参数 ν=0 或简单值，复杂材料或非线性问题尚未覆盖。

---

## 609. How Do LLMs Encode Scientific Quality? An Empirical Study Using Monosemantic Features from Sparse Autoencoders

**arXiv ID:** 2602.19115 | [PDF](https://arxiv.org/pdf/2602.19115v1)

**作者:** Michael McCoubrey `[一作]` (Open University), Enrico Motta `[通讯]` (Open University)

**通讯引用:** 13991 | [OpenAlex ID](https://openalex.org/A5068408717)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用稀疏自编码器提取大型语言模型（Gemma 2 2B/9B）中的单义特征，构建决策树分类器，评估这些特征在预测论文引用量、期刊SJR和h‑指数四分位数上的效果，并对特征进行语义解释。

**💡 创新点**

首次系统探究LLM内部如何编码科研质量概念，并通过单义特征发现四类主要特征（研究方法、出版类型、热点领域技术、专业术语），为LLM可解释性和科研评价提供新的视角。

**🔧 技术方法**

稀疏自编码器（SAE）提取特征；Gemma 2 2B/9B模型；决策树分类器；BERT文本分类基线；Neuropedia服务用于特征解释。

**📊 数据集**

AIDA知识图谱（约2500万条CS文献）中抽取的38,639篇论文的标题、摘要、5年引用计数、期刊SJR和h‑指数；按四分位数划分后构成三组二分类子集。

**📈 对比分析**

将决策树（仅使用单义特征）与BERT基线在三项二分类任务（引用、SJR、h‑指数四分位）上对比，使用准确率衡量。结果显示决策树在引用任务上略优，BERT在SJR和h‑指数任务上更好；整体准确率均在0.66–0.79之间，表明单义特征具备可观的预测能力。

**⚠️ 局限性**

解释过程主观性大、仅评估Gemma 2模型、使用四分位数平衡化导致信息损失、未做回归预测、缺乏跨模型和跨学科的验证。

---

## 610. Universal 3D Shape Matching via Coarse-to-Fine Language Guidance

**arXiv ID:** 2602.19112 | [PDF](https://arxiv.org/pdf/2602.19112v1)

**作者:** Qinfeng Xiao `[一作]` (Hong Kong Polytechnic University), Kit-lun Yick `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 3014 | [OpenAlex ID](https://openalex.org/A5018378567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种语义感知的粗到细框架，用于在没有预定义部件先验的情况下实现通用 3D 形状匹配。

**💡 创新点**

核心创新在于：①使用类无关 3D 分割（PartField）与 MLLM（GPT‑5）提示获取部件名称；②利用 FG‑CLIP 语言嵌入构建粗粒度语义对应；③在功能映射中引入组级 Rank‑n 对比损失，既减少计算量，又保持语义连续性；④实现了跨类别、非等距变形的匹配。

**🔧 技术方法**

技术手段包括：类无关 3D 分割（PartField）、多模态大语言模型（GPT‑5）提示、细粒度语言嵌入（FG‑CLIP）、SD‑DINO 语义特征场、DiffusionNet 细化器以及扩展的功能映射（URSSM）配合组级 Rank‑n 对比损失。

**📊 数据集**

实验使用了跨类别数据集 SNIS、TOSCA、SHREC07；非等距变形数据集 SMAL、TOPKIDS；近等距变形数据集 FAUST、SCAPE、SHREC19；以及在 SHREC07 中挑选的原始对象进行野生测试。

**📈 对比分析**

与基准方法（ZoomOut、URSSM、Diff3F、DenseMatcher、ZSC 等）对比，本文在交叉类别上平均几何误差最低（0.19/0.23/0.37），在非等距和近等距变形上表现同样优异（4.8/5.9、1.6/1.9/3.2），显著优于之前的功能映射和语义驱动方法。

**⚠️ 局限性**

局限性包括：对部件命名的歧义性导致顺序不匹配（如椅腿）、仅在训练阶段使用 MLLM 提示、对缺乏明显语义特征的形状适应性有限，以及未考虑物体方向导致的对称部件误配。

---

## 611. Information-Guided Noise Allocation for Efficient Diffusion Training

**arXiv ID:** 2602.18647 | [PDF](https://arxiv.org/pdf/2602.18647v1)

**作者:** Gabriel Raya `[一作]` (Jheronimus Academy of Data Science), Luca Ambrogioni `[通讯]` (Radboud University)

**通讯引用:** 849 | [OpenAlex ID](https://openalex.org/A5039391126)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于信息论的自适应噪声调度方法，实时根据前向扩散过程中的条件熵率重新分配训练更新的频率。

**💡 创新点**

创新点在于将I–MMSE理论与条件熵率结合，形成可在线估计的噪声重要性指标，并利用该指标动态构建训练噪声分布，从而无需人工手动设计噪声调度。

**🔧 技术方法**

主要技术包括：I–MMSE身份、条件熵率估计、低噪声门控正则化、基于累积分布的逆CDF采样、EMA平滑与在线刷新机制。

**📊 数据集**

使用的公开数据集包括连续图像（MNIST、FashionMNIST、CIFAR‑10、FFHQ）以及离散数据（binarized MNIST/FashionMNIST、DNA序列）。

**📈 对比分析**

与固定噪声调度（log‑uniform、EDM log‑normal等）对比，实验显示在离散任务上可减少1.5–3倍训练样本；在连续图像上与手工调优的EDM调度竞争，且在CIFAR‑10等任务上可获得约1.4–1.5倍的训练加速。

**⚠️ 局限性**

局限性包括：依赖于在线噪声熵率估计，早期训练阶段估计噪声较高方差；需调节刷新频率、缓冲长度与门控参数；对非高斯或离散噪声模型的推广尚未完全验证。

---

## 612. MentalBlackboard: Evaluating Spatial Visualization via Mathematical Transformations

**arXiv ID:** 2602.19357 | [PDF](https://arxiv.org/pdf/2602.19357v1)

**作者:** Nilay Yilmaz `[一作]` (Arizona State University), Yezhou Yang `[通讯]` (Arizona State University)

**通讯引用:** 4417 | [OpenAlex ID](https://openalex.org/A5002278578)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究开发了MentalBlackboard，一个开放式的空间可视化基准，评估最新的视觉-语言模型（VLMs）在纸折叠和打孔测试中的空间可视化能力，主要包括预测和规划两个核心任务。

**💡 创新点**

创新点在于引入了一个大规模的开放式基准，结合了纸折叠测试和旋转变换，能够动态生成任务的3D动画，并通过开放式评估框架分析模型的表现和错误原因。

**🔧 技术方法**

使用了VPython平台构建物理动态系统，结合了多种折叠类型和旋转角度，生成了超过12,000个独特配置的任务。

**📊 数据集**

数据集基于纸折叠和打孔测试，包含900个预测问题和400个规划问题，涉及多种折叠类型和旋转配置。

**📈 对比分析**

与现有模型的比较显示，模型在文本预测任务中的准确率最高为25%，而在规划任务中仅为10%。模型在处理对称关系和多阶段对称过程时表现出明显的局限性。

**⚠️ 局限性**

限制在于当前模型在执行空间可视化时缺乏顺序推理和视觉空间工作记忆，尤其在处理旋转和对称变换时表现不佳。

---

## 613. MapTab: Can MLLMs Master Constrained Route Planning?

**arXiv ID:** 2602.18600 | [PDF](https://arxiv.org/pdf/2602.18600v1)

**作者:** Ziqiao Shang `[一作]` (National Key Laboratory for Novel Software Technology), Lan-Zhe Guo `[通讯]` (National Key Laboratory for Novel Software Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MapTab多模态基准，用于评估多模态大型语言模型在多约束路线规划任务中的推理能力。

**💡 创新点**

创新点包括：①将地图图像与结构化表格（节点表、边表）结合，构建多维约束的异构图；②覆盖两大真实场景（Metromap与Travelmap），提供数十万个路线规划与数千个问答样本；③系统化评估15种主流多模态LLM，揭示视觉感知、跨模态融合、约束推理与多步推理的瓶颈。

**🔧 技术方法**

技术手段涵盖：图像预处理与OCR、Gemini‑3‑Flash生成表格、基于Dijkstra的最短路径与约束优化、不同提示模板（Direct、Graph-theoretic等）、长链思考（Long-CoT）和标准评价指标（EMA、PMA、DS）。

**📊 数据集**

数据集为MapTab，包含328张高分辨率地图（Metromap 160张，Travelmap 168张），配合Edge_tab与Vertex_tab，共生成196,800条路线规划查询与3,936条问答查询。

**📈 对比分析**

对比方法：将模型在Map‑Only、Edge‑Tab‑Only、Map+Edge‑Tab、Map+Edge‑Tab+Vertex‑Tab等多模态输入下进行统一评测。结果显示：当前模型在视觉感知受限时表现明显逊色；多模态协同不总能提升性能；约束引入往往导致精度下降，且多步推理与数值运算是主要瓶颈。

**⚠️ 局限性**

局限性：1）仅针对静态拓扑图，缺乏动态或实时推理；2）视觉理解依赖高分辨率，低质量图像会严重影响性能；3）跨模态融合机制仍不成熟，易受视觉噪声干扰；4）对复杂约束与异构权重的建模能力有限。

---

## 614. Weak-Form Evolutionary Kolmogorov-Arnold Networks for Solving Partial Differential Equations

**arXiv ID:** 2602.18515 | [PDF](https://arxiv.org/pdf/2602.18515v1)

**作者:** Bongseok Kim `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**通讯引用:** 6244 | [OpenAlex ID](https://openalex.org/A5078138445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种弱形式演化Kolmogorov–Arnold网络（KAN）框架，用于求解时变偏微分方程，并通过将参数更新线性系统的维数与训练样本数解耦，实现可扩展且数值稳定的数值求解。

**💡 创新点**

创新点包括：①在弱形式下构造参数更新系统，系统大小由测试函数数量决定，独立于样本点数量；②将强形式点值残差替换为弱形式积分残差，降低微分阶数并改善线性系统条件；③利用边界约束KAN实现Dirichlet/周期边界，并在弱形式中直接处理Neumann边界。

**🔧 技术方法**

技术手段：弱形式演化KAN（RBF-KAN架构）、Gauss–Legendre积分、最小二乘正则化、自动微分、前向Euler时间步进、测试函数投影、边界约束Kans。

**📊 数据集**

使用四个基准PDE数据集：一维Allen–Cahn方程、二维Burgers方程、二维热方程（Neumann边界）、二维含漂移的Porous Medium方程，所有数据均为人工生成的解析/数值解。

**📈 对比分析**

对比方法包括：弱形式演化KAN、强形式演化KAN以及传统PINN（强形式）。实验表明，弱形式方法在L2误差、条件数、计算时间上均优于强形式，且在样本点不足时保持稳定；强形式方法误差随样本增大下降，但条件数和计算时间增长迅速。

**⚠️ 局限性**

局限性：①弱形式仍需手工选择合适的测试函数数量与类型；②对高度非线性或高维问题的可扩展性尚未完全验证；③在极端激波或不连续解的捕捉仍可能需要更高阶测试函数或自适应网格；④计算成本仍受网络参数规模影响，需要进一步优化稀疏性或近似方法。

---

## 615. Towards Calibrating Prompt Tuning of Vision-Language Models

**arXiv ID:** 2602.19024 | [PDF](https://arxiv.org/pdf/2602.19024v1)

**作者:** Ashshak Sharifdeen `[一作]` (Mohamed bin Zayed University of AI), Muhammad Haris Khan `[通讯]` (Mohamed bin Zayed University of AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对提示调优的 CLIP 进行校准，提升模型的置信度可靠性。

**💡 创新点**

提出均值-方差边缘正则化和文本矩匹配两种正则项，兼顾基类的欠信和新类的过信问题，并保持预训练嵌入空间几何不变。

**🔧 技术方法**

使用交叉熵加边缘正则化、文本矩匹配的联合损失，并在提示调优框架（CoOp、KgCoOp、MaPLe 等）中实现。

**📊 数据集**

在 11 个多样数据集上评估，涵盖 ImageNet、Caltech101、DTD、Flowers、Food101、SUN397、UCF101、Stanford Cars、FGVC-Aircraft、Oxford Pets、EuroSAT 等。

**📈 对比分析**

与 MBLS、Temperature Scaling、DAC、ZS-Norm 等基线比较，平均 ECE 降低 30%-50%，同时保持甚至略微提升分类准确率。

**⚠️ 局限性**

局限在于正则项超参数需要手工调节，且在极少样本情形下收敛速度和稳定性尚未得到充分验证。

---

## 616. Pixel2Phys: Distilling Governing Laws from Visual Dynamics

**arXiv ID:** 2602.19516 | [PDF](https://arxiv.org/pdf/2602.19516v1)

**作者:** Ruikun Li `[一作]` (Tsinghua University), Yan Lu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 4741 | [OpenAlex ID](https://openalex.org/A5100756584)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

Pixel2Phys 是一种协作多智能体框架，能够从高维视觉视频中自动提取物理变量并推导可解释的支配方程；

**💡 创新点**

创新点在于：① 通过规划、变量、方程、实验四个专职智能体实现迭代式协同优化，形成“观察–假设–实验–修正”的科学工作流；② 将前置已发现的方程反馈给变量提取器，构建物理信息驱动的自编码器，打破变量与方程共识问题；③ 采用多粒度工具（对象跟踪、像素梯度卷积、物理信息自编码）满足不同物理场景；

**🔧 技术方法**

技术手段包括：基于大型多模态语言模型（如 GPT‑4o）的智能体交互；对象级分割跟踪（Segment Anything）；像素梯度卷积算子；物理信息自编码器（重构+方程一致性损失）；符号回归（STLSQ）和稀疏约束；实验评估（RMSE、R²、VPS、相位图）等；

**📊 数据集**

使用的数据集包括：① 5 种经典动力学系统（Linear、Cubic、Circular、Van Der Pol、Glider）的视频；② 4 种 PDE 反应扩散方程（Lambda–Omega、Brusselator、FitzHugh–Nagumo、Swift–Hohenberg）的离散网格视频；③ 6 个真实世界实验视频（4 个 Kármán vortex、2 个 Belousov–Zhabotinsky），均为灰度帧；

**📈 对比分析**

与 AE‑SINDy、Latent‑ODE、Coord‑Equ 等隐式/显式基线，以及 FNO、UNO、WNO、PDE‑Find、SGA‑PDE、LLM‑PDE、Wan2.2 等对比。Pixel2Phys 在长时预测 RMSE 下降约 45%（R²>0.99）、VPS 远高于对手，假设精度高、误项极少；在 PDE 任务上实现最低 RMSE 与几乎 1000 步的有效预测；在真实实验视频中获得最低预测误差与最高振幅一致性。

**⚠️ 局限性**

局限性包括：对 MLLM 解释能力的依赖，模型规模越大效果越好；在极噪声或非常复杂的非线性方程（如多项式/三角混合）时仍可能出现误匹配；需要手工选择合适的提取工具或参数；实验规模受限于已公开的视频数据集，尚未在更大、不同领域的真实数据上进行广泛验证；

---

## 617. Generative AI in Knowledge Work: Perception, Usefulness, and Acceptance of Microsoft 365 Copilot

**arXiv ID:** 2602.18576 | [PDF](https://arxiv.org/pdf/2602.18576v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 618. Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications

**arXiv ID:** 2602.19763 | [PDF](https://arxiv.org/pdf/2602.19763v1)

**作者:** Yida Lin `[一作]` (Victoria University of Wellington), Richard Green `[通讯]` (University of Canterbury)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

对十种深度立体匹配网络进行训练与评估，使用基于 DEFOM‑Stereo 的伪真值，对实际树枝图像进行深度估计，并在实测无人机上测试实时性能。

**💡 创新点**

①首个专为植被场景设计的立体匹配基准；②使用 DEFOM‑Stereo 伪真值生成大规模真实森林数据；③在嵌入式 Jetson Orin Super 上系统性比较质量与速度，提出可行的分辨率与模型选择指南。

**🔧 技术方法**

深度学习立体匹配架构（BANet、RAFT‑Stereo、PSMNet 等），使用 AdamW 优化、smooth L1 损失、数据增强；评估指标包括 SSIM、LPIPS、ViTScore、SIFT/ORB 匹配率；推理加速在 Jetson 上使用原生 PyTorch 进行 FPS/延迟测评。

**📊 数据集**

Canterbury Tree Branches 数据集：5,313 对齐的 ZED Mini 雙目图像（1080P/720P），使用 DEFOM‑Stereo 生成的视差作为训练标签。

**📈 对比分析**

通过在测试集上计算 SSIM、LPIPS、ViTScore、SIFT/ORB 匹配率，对十个模型的质量进行排序；在 Jetson Orin Super 上测量 FPS 与延迟，绘制 SSIM vs FPS Pareto 前沿。结果显示 BANet‑3D 质量最高（SSIM 0.883），AnyNet 速度最快（6.99 FPS），BANet‑2D 兼顾质量与速度。

**⚠️ 局限性**

①伪真值仅可反映 DEFOM‑Stereo 的准确度，无法证明绝对几何精度；②仅测试 Radiata pine，缺乏对不同树种、气候和季节的泛化评估；③未使用 TensorRT 等硬件加速；④推理在 1080P 下多数模型仍低于实时阈值，需进一步优化。

---

## 619. BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment

**arXiv ID:** 2602.19170 | [PDF](https://arxiv.org/pdf/2602.19170v1)

**作者:** Kanglei Zhou `[一作]` (Tsinghua University), Liyuan Wang `[通讯]` (Tsinghua University)

**通讯引用:** 3481 | [OpenAlex ID](https://openalex.org/A5115695075)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Bridged Modality Adaptation (BriMA) 方法，用于解决多模态持续动作质量评估中非平稳模态失衡问题，能够在模态缺失或不稳定的实际部署环境下保持高质量评分；

**💡 创新点**

创新点包括两大模块：1）基于记忆引导的桥接重建 (Memory‑Guided Bridging Imputation, MBI)，通过检索历史示例并仅预测残差来实现稳定且符合评分语义的模态补全；2）模态感知重放优化 (Modality‑Aware Replay Optimization, MRO)，根据模态失真与分数漂移动态选择并优先重放样本，以减轻灾难性遗忘；

**🔧 技术方法**

技术实现上结合了特征检索、残差桥接网络、可学习任务指示向量、加权余弦相似度候选选择、优先级重放机制和记忆正则化；模型采用 PyTorch，使用 Adam + cosine 学习率调度；

**📊 数据集**

在三个多模态 AQA 基准上进行实验：Rhythmic Gymnastics (RG)、Figure Skating Video (Fis‑V) 与 FS1000；每个数据集分别使用视频、音频和文本评论三种模态；

**📈 对比分析**

与多种基线（包括联合训练、持续学习方法、回放方法等）进行对比，BriMA 在 10%、25% 与 50% 模态缺失率下均实现了显著提升，平均 SRCC 提升约 6–8%，MSE/ RL2 降低 12–15%，并在效率方面仅增加约 0.1M 参数、1 小时训练时间和 0.1 次/秒推理延迟；

**⚠️ 局限性**

局限性包括：1）在模态数量较大或缺失模式多样时，候选检索和记忆存储可能面临规模挑战；2）未显式建模细粒度时序动态；3）仅假设可观测的模态缺失，无法处理完全未知缺失模式；4）残差桥接的保守设计可能在极弱模态或未见缺失模式下效果不佳。

---

## 620. Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning

**arXiv ID:** 2602.18493 | [PDF](https://arxiv.org/pdf/2602.18493v1)

**作者:** Kehao Zhang `[一作]` (Chinese Academy of Sciences), Yang Feng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 146509 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Unified Memory Agent（UMA），一个端到端强化学习框架，将记忆操作（CRUD）与问答统一为单一策略。

**💡 创新点**

通过 Task-Stratified GRPO 同步优化记忆维护与推理，结合多阶段记忆银行与混合检索，并推出 Ledger‑QA 评测动态状态追踪的 benchmark。

**🔧 技术方法**

采用强化学习（Task-Stratified Group Relative Policy Optimization）、LLM 工具调用接口、双重记忆结构（核心摘要+结构化银行）以及 LLM-as-Judge 评估。

**📊 数据集**

使用 13 个任务集：Ledger‑QA、Test‑Time Learning（TREC‑Coarse/Fine、NLU、Clinic、Banking77、PubMed‑RCT）以及 Accurate Retrieval（HotpotQA、LoCoMo、LongMemEval、MSC、PearlTQA、SQuAD、ConvoMem）。

**📈 对比分析**

与 Concat、RAG、MemAgent、MemAlpha 等基线对比，在动态状态追踪、TTL 与 AR 任务中均取得显著提升；尤其在 Ledger‑QA 长序列（50 次会话）上准确率超过 50%，而基线低于 25%。

**⚠️ 局限性**

仅能在固定上下文窗口内工作，无法处理百万级长文本；Ledger‑QA 为合成数据，缺乏真实噪声；工具集有限，难以扩展至多模态或更复杂场景。

---

## 621. Axis Decomposition for ODRL: Resolving Dimensional Ambiguity in Policy Constraints through Interval Semantics

**arXiv ID:** 2602.19878 | [PDF](https://arxiv.org/pdf/2602.19878v1)

**作者:** Daham Mustafa `[一作]`, Stephan Decker `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了轴分解框架解决ODRL 2.2中多维左操作数与单标量约束模式的矛盾，实现了对多维空间约束的确定性解释与冲突检测。

**💡 创新点**

通过将多维约束拆分为轴特定的标量约束，证明了四项属性（确定性、AABB完整性、投影下的保真性与保守扩展），并实现了强Kleene三值逻辑的冲突合成。

**🔧 技术方法**

形式化证明使用Isabelle/HOL完成，工具链包括TPTP FOF（Vampire）和SMT‑LIB（Z3）对约束进行定理证明与符号执行；实现了ODRL Spatial Axis Profile的OWL 2和SHACL形状验证。

**📊 数据集**

使用117个合成基准问题，涵盖9类场景（单轴、多轴、盒子、组合、跨域、策略质量、边界、逻辑或、逻辑异或），基准来源于文化遗产数据空间（如Datenraum Kultur）案例。

**📈 对比分析**

与Vampire和Z3进行对比，全部117个问题均达成100%结果一致，证明框架在定理证明与SMT求解上的一致性与可靠性。

**⚠️ 局限性**

仅适用于轴独立且非环形的数值域，无法处理环形坐标、耦合轴、跨轴约束（面积、比例）以及非数值结构空间；未来工作需扩展至环形域与耦合约束。

---

## 622. Iconographic Classification and Content-Based Recommendation for Digitized Artworks

**arXiv ID:** 2602.19698 | [PDF](https://arxiv.org/pdf/2602.19698v1)

**作者:** Krzysztof Kutt `[一作]` (Jagiellonian University), Maciej Baczyński `[通讯]` (Jagiellonian University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `a2602d71-93ab-4bad-974b-672788df8193` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了CARIS原型，实现了对数字化艺术品的Iconclass分类与基于内容的推荐，自动化地将视觉信息映射为层级化符号代码并生成相似作品列表。

**💡 创新点**

首次将YOLO目标检测与Iconclass层级映射相结合，利用规则推理产生抽象符号代码，并设计三种互补的Iconclass推荐算法（层级相似度、IDF加权重叠、Jaccard相似度）。

**🔧 技术方法**

采用YOLOv8目标检测、关键词/描述映射、规则引擎推理、层级相似度、IDF加权重叠、Jaccard相似度等技术，并集成Python Iconclass库实现代码检索与层级导航。

**📊 数据集**

使用公开的Wikimedia Commons图像进行分类评测，以及Iconclass AI Test Set（约87k图像）用于推荐模块的性能验证。

**📈 对比分析**

通过与专家标注的Iconclass代码对比评估分类准确性，并在测试集上对三种推荐器进行比较；层级方法在缺少精确匹配时效果最佳，IDF在稀有代码场景表现突出，Jaccard在代码数量多的作品中更为稳健，整体展示了系统的可行性与潜在优势。

**⚠️ 局限性**

主要限制在于目标检测召回率低导致分类误差、代码爆炸与误匹配问题、规则引擎需要人工校验，以及缺乏用户历史与多模态信息支持。

---

## 623. Toward a Quiet Wireless World: Multi-Cell Pinching-Antenna Transmission

**arXiv ID:** 2602.19459 | [PDF](https://arxiv.org/pdf/2602.19459v1)

**作者:** Zhiguo Ding `[一作]` (Nanyang Technological University), Zhiguo Ding `[通讯]` (Nanyang Technological University)

**通讯引用:** 60036 | [OpenAlex ID](https://openalex.org/A5002904166)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并评估了多小区pinching天线的下行传输方案，构建了系统模型并通过功率与天线位置的联合优化实现总体功率最小化。

**💡 创新点**

利用pinching天线将基站激活点靠近用户，实现低功率“耳语”传输；并在多小区环境中引入无协调、无回程/前传的干扰管理框架。

**🔧 技术方法**

交叉熵优化用于天线位置搜索，线性规划求解功率分配，二小区特例给出闭式解。

**📊 数据集**

采用仿真数据，设定服务区尺寸、噪声功率、目标速率等参数；无公开数据集。

**📈 对比分析**

与传统中心基站方案进行对比，仿真表明pinching天线方案在功率消耗和不可达率方面明显优于基准，尤其在高速率时更具鲁棒性。

**⚠️ 局限性**

优化问题仍为非凸，跨多小区时求解复杂；对天线位置的先验假设和仿真参数限制了实际可推广性。

---

## 624. MiSCHiEF: A Benchmark in Minimal-Pairs of Safety and Culture for Holistic Evaluation of Fine-Grained Image-Caption Alignment

**arXiv ID:** 2602.18729 | [PDF](https://arxiv.org/pdf/2602.18729v1)

**作者:** Sagarika Banerjee `[一作]` (Algoverse AI Research), Vasu Sharma `[通讯]` (PocketFM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个细粒度图像-字幕对齐评估基准 MiSCHiEF，包含安全（MiS）和文化（MiC）两类最小差异样本，并对四个主流视觉‑语言模型在四种匹配与一致性任务上进行评估。

**💡 创新点**

创新点在于：①针对安全与文化两大社会关键场景设计最小差异对（minimal-pairs）数据，①通过对齐任务揭示模型在负向推理和多重匹配上的系统性偏差，②提出对比实验框架揭示跨模态对齐不对称性。

**🔧 技术方法**

采用 GPT‑Image 与 Gemini 2.5 Pro 生成与校验图像与字幕，使用四类对齐任务（Caption‑to‑Image、Image‑to‑Caption、Dual Caption‑Image Alignment、Pairwise Consistency）对模型进行评估。

**📊 数据集**

数据集包括 MiS（190 对安全场景）、MiC（279 对文化代理）共 400 对最小差异图像‑字幕对，全部手工验证以确保无歧义。

**📈 对比分析**

通过对四个模型（Qwen 3B、InternVL、Phi‑3.5、Llava‑Next‑Video、GPT‑4o）在上述四项任务中计算准确率，发现模型在确认正确配对时表现优于拒绝错误配对；I2C 任务准确率普遍高于 C2I，DCI 任务整体准确率最低；GPT‑4o 在大多数任务中取得最高且更均衡的性能。

**⚠️ 局限性**

限制：数据规模较小（仅 400 对），高度依赖人工验证；未进行全面人类评估或跨基准相关性分析；未探究自动化生成方式对质量的影响。

---

## 625. Robust Taylor-Lagrange Control for Safety-Critical Systems

**arXiv ID:** 2602.20076 | [PDF](https://arxiv.org/pdf/2602.20076v1)

**作者:** Wei Xiao `[一作]` (Worcester Polytechnic Institute and Massachusetts Institute of Technology), Anni Li `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种鲁棒Taylor‑Lagrange控制（rTLC）方法，用于在保持非线性系统安全性的同时解决采样间效应；

**💡 创新点**

通过在Taylor展开中使用比安全函数相对度高一级的阶数，直接将控制量显式为当前时刻的值，并利用可达集分析得到Lagrange剩余项的下界，从而实现仅需一个超参数即可保证安全；

**🔧 技术方法**

使用Taylor展开与Lagrange余项、可达集分析、控制限约束的Lyapunov/Kontinuity理论、二次规划求解安全约束；

**📊 数据集**

采用自适应巡航控制（ACC）仿真案例，使用MATLAB/ode45、quadprog，参数设置见论文；

**📈 对比分析**

与高阶控制B函数（HOCBF）、时基TLC、事件触发TLC进行比较，实验显示时基TLC在采样间效应下会违约，而rTLC既能保证安全又保留较低的保守性，计算时间与其他方法相近；

**⚠️ 局限性**

方法的保守性随Δt减小而降低，但仍可能导致过度安全；对Lagrange剩余项可达集的精确求解尚未实现，影响方法的紧凑性和实时性。

---

## 626. Constructing (Co)inductive Types via Large Sizes

**arXiv ID:** 2602.18921 | [PDF](https://arxiv.org/pdf/2602.18921v1)

**作者:** Benno van den Berg `[一作]` (University of Amsterdam), Daniël Otten `[通讯]` (University of Amsterdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种基于大小（sizes）与参数化量词的类型系统，利用它们来构造任意（共）归纳类型，并证明该系统的一致性；

**💡 创新点**

创新点在于：①将大小类型解释为不可数序数（ω₁），从而满足所有多态量词与大小之间的等价性；②通过参数化量词实现完整的（共）归纳类型，而不需使用最大大小 ∞；③给出基于 Hyland 有效上层范畴的可实现性模型来验证一致性；

**🔧 技术方法**

核心技术包括：大小类型与参数化量词的引入、对多态量词与类型构造器的等价性公理、使用 ω₁ 作为大小的解释、构造 fixpoint 运算符（仅为命题等式）、构造大小索引的初始代数与终极余代数，以及基于 PER 的可实现性模型与大集合的解释；

**📊 数据集**

无实验数据集，本文完全基于理论证明与模型构造；

**📈 对比分析**

本文未进行实验或性能对比，所有结果均为形式化证明与模型解释；

**⚠️ 局限性**

局限性包括：fixpoint 运算符仅满足命题 β 规则，缺乏计算行为；需进一步研究计算型参数化量词与可计算 β 规则；实现细节与可计算性（如可计算性、归一化）尚未解决；

---

## 627. ApET: Approximation-Error Guided Token Compression for Efficient VLMs

**arXiv ID:** 2602.19870 | [PDF](https://arxiv.org/pdf/2602.19870v1)

**作者:** Qiankun Ma `[一作]` (Shenzhen Institutes of Advanced Technology), Hairong Zheng `[通讯]` (Shenzhen Institutes of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无注意力机制的视觉令牌压缩框架ApET，利用线性逼近误差评估每个视觉令牌的重要性，自动筛选并合并不重要的令牌，以显著减少令牌数量。

**💡 创新点**

核心创新在于：①从信息论角度将重构误差作为令牌重要性的度量，彻底摆脱对注意力权重的依赖；②实现了对FlashAttention等高效注意力核的兼容，避免了位置偏差与内部权重访问的限制；③通过线性逼近实现低成本、模型无关的压缩。

**🔧 技术方法**

使用线性逼近（投影基向量 + 最小二乘求解）计算重构误差；基向量采样（FPS、DPC、随机）；基于误差排序的令牌淘汰与相似性合并；与FlashAttention兼容的无注意力压缩。

**📊 数据集**

评估数据集包括图像理解：GQA、MMB、MMB^CN、MME、POPE、SQA、VQA^V2、VQA^Text、VizWiz；视频理解：TGIF、MSVD、MSRVTT；模型使用LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B和Video-LLaVA-7B。

**📈 对比分析**

与现有基于注意力的压缩方法（ToMe、FastV、SparseVLM、PDrop、VisionZip）进行对比。ApET在保留88%~89%令牌时仍能保持95.2%图像任务性能，甚至在视频任务中达到100.4%性能；在压缩率更高时性能提升更显著；在总推理时间、预填充时间和TFLOPs上相较基线和其他方法提升约1.3–1.5×。

**⚠️ 局限性**

局限性包括：①线性逼近可能无法完全捕捉令牌间的非线性关系，对极端场景的压缩效果有限；②虽然对大多数模型通用，但在需要精细视觉细节的任务中可能需要更精细的基向量选择；③未对语言令牌进行压缩，若整体模型需要进一步加速仍需配合其它策略。

---

## 628. Grokking Finite-Dimensional Algebra

**arXiv ID:** 2602.19533 | [PDF](https://arxiv.org/pdf/2602.19533v1)

**作者:** Pascal Jr Tikeng Notsawo `[一作]` (Université de Montréal), Guillaume Rabusseau `[通讯]` (Université de Montréal)

**通讯引用:** 391 | [OpenAlex ID](https://openalex.org/A5023766963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究神经网络在学习有限维代数（FDA）乘法时出现的grokking现象。

**💡 创新点**

提出将所有FDA映射为结构张量的统一框架，并将grokking分析扩展到非结合、非交换、非单位代数。

**🔧 技术方法**

采用结构张量表示、线性逆问题、矩阵分解、表示学习以及Transformer/MLP/LSTM模型等技术。

**📊 数据集**

在有限域\(\mathbb{F}_p\)上随机生成的FDA张量及其乘法表作为实验数据集。

**📈 对比分析**

通过对比不同代数属性、张量稀疏度/秩以及训练样本比例的实验，发现非单位结合代数易于grokking，稀疏/低秩张量加速泛化，性能随代数属性和数据量显著变化。

**⚠️ 局限性**

主要局限在于对实域FDA的理论分析不足，实验规模受限于小维度，且稀疏度结果依赖于基底选择。

---

## 629. BabyLM Turns 4: Call for Papers for the 2026 BabyLM Workshop

**arXiv ID:** 2602.20092 | [PDF](https://arxiv.org/pdf/2602.20092v1)

**作者:** Leshem Choshen `[一作]` (IBM Research), Ethan Gotlieb Wilcox `[通讯]` (Georgetown University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本次 CFP 介绍了 BabyLM 2026 研讨会与第四届挑战赛，推出新的多语言 track 并鼓励跨学科研究；

**💡 创新点**

创新点在于把认知科学与语言模型结合，新增 BabyBabelLM 多语言数据集，强调样本高效与多语言普适性；

**🔧 技术方法**

采用预训练 + 微调、零样本评估、基线模型 GPT‑BERT、GPT‑2 Small 与 SimPO 等技术；

**📊 数据集**

使用 100M 词严格 BabyLM 数据集、BabyBabelLM 的 45 种语言样本以及多模态文本与图像数据；

**📈 对比分析**

通过提交模型、基线及评估脚本在 EMNLP 评测平台比较，预期性能以往基线为参考并鼓励改进；

**⚠️ 局限性**

限制训练轮数、总数据量、外部模型使用、毒性检测等，兼顾计算公平与可解释性。

---

## 630. WANSpec: Leveraging Global Compute Capacity for LLM Inference

**arXiv ID:** 2602.18931 | [PDF](https://arxiv.org/pdf/2602.18931v1)

**作者:** Noah Martin `[一作]` (Tufts University), Fahad Dogar `[通讯]` (Tufts University)

**通讯引用:** 1266 | [OpenAlex ID](https://openalex.org/A5075615793)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 WANSPEC，利用宽域网络把大型语言模型推理的草稿模型迁移到低端或非高负载的数据中心，从而缓解核心数据中心的 GPU 负载。

**💡 创新点**

创新点在于：① 在 controller 与 worker 两端使用熵阈值预测同步需求，动态决定是否使用冗余草稿推理；② 通过熵预测和树形分支避免过多冗余计算；③ 通过 KV 缓存同步与并行预填等技术，确保跨地区推理保持一致且延迟可控。

**🔧 技术方法**

使用的技术包括：speculative decoding、draft 与 target 模型、熵阈值预测、KV 缓存同步、分布式 controller/worker 架构、vLLM 的修改、Python 事件驱动模拟器、AWS Bedrock 的多地区实验。

**📊 数据集**

数据集与测量数据包括：MTBench（问答生成），Llama-3.3‑70B/1B、Qwen2‑72B/1.5B、Claude 3 Haiku 的时间到首词（TTFT）测量，及 AWS 各区域的延迟与队列负载数据。

**📈 对比分析**

在模拟器和真实云部署（AWS）中与传统 speculative decoding 进行对比。结果显示：在 10‑30 ms 的 RTT 下，WANSPEC 可将 controller 侧草稿模型的前向传递减少约 30‑50%，整体生成时间提升不超过 5%；在同一洲多区域部署时性能可与单机相当甚至略优。

**⚠️ 局限性**

局限性：① 高 RTT 时性能退化；② 熵阈值需手动调优，误判可能导致额外计算或短暂停顿；③ KV 缓存同步未压缩，传输开销仍存在；④ 目标模型仍集中在高端数据中心，迁移仅限草稿模型；⑤ 真实跨洲实验不足，跨洲性能验证有限。

---

## 631. Hyper-KGGen: A Skill-Driven Knowledge Extractor for High-Quality Knowledge Hypergraph Generation

**arXiv ID:** 2602.19543 | [PDF](https://arxiv.org/pdf/2602.19543v1)

**作者:** Rizhuo Huang `[一作]` (Xi'an Jiaotong University), Yue Gao `[通讯]` (Tsinghua University)

**通讯引用:** 19078 | [OpenAlex ID](https://openalex.org/A5100602494)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于技能驱动的知识超图生成框架 Hyper‑KGGen，结合粗细层次提取和自适应技能获取；

**💡 创新点**

通过稳定性反馈循环动态演化技能库解决场景差距和结构失衡，并采用粗到细的层次提取保证低阶与高阶关系共存；

**🔧 技术方法**

使用粗细层次抽取（文档分块、实体抽取、逐级超边提取）、并行推理、稳定性奖励、路径归因与事后推理、全局技能库管理与LLM提示化等技术；

**📊 数据集**

构建 HyperDocRED 文档级 n‑ary 关系标注集；使用 MINE 多领域文章集和 UltraDomain 书本集做实验；

**📈 对比分析**

与 KGGen、iText2KG、RAKG、NativeRAG、GraphRAG、HyperGraphRAG 等基线在 HyperDocRED、MINE、UltraDomain 上进行 n‑ary 关系提取、事实覆盖和 RAG 评估；Hyper‑KGGen+ 在所有指标上均超过基线，尤其在 n‑ary recall 和 RAG 指标上提升显著；

**⚠️ 局限性**

依赖 LLM 的输出质量；技能库规模增长导致存储与检索开销；对极端领域词汇仍可能需要人工干预；对长文本分块可能遗漏跨块关系。

---

## 632. IPv2: An Improved Image Purification Strategy for Real-World Ultra-Low-Dose Lung CT Denoising

**arXiv ID:** 2602.19314 | [PDF](https://arxiv.org/pdf/2602.19314v1)

**作者:** Guoliang Gong `[一作]` (Tianjin University of Science and Technology), Man Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出改进后的图像净化策略IPv2，用于真实世界超低剂量CT图像的去噪任务。

**💡 创新点**

创新点在于引入三大模块：去背景、加噪（在肺泡纹理中注入真实噪声）和去噪（利用弱去噪器构造无噪声肺部标签），有效解决背景噪声和肺泡纹理去噪的缺陷。

**🔧 技术方法**

使用技术包括：结构对齐与残差分解、Radon变换+Poisson-Gaussian噪声注入、弱去噪器、掩模融合、频域流匹配与频域流匹配模型（FM/FFM）、核心扩散（CoreDiff）、冷扩散等深度去噪网络。

**📊 数据集**

采用真实患者肺CT 2%剂量数据集，共 4310 对 uLDCT/NDCT 图像，按 7:1.5:1.5 划分为训练/验证/测试集。

**📈 对比分析**

在四种主流迭代映射网络（CoreDiff、Cold Diffusion、FM、FFM）上进行对比实验，使用 FID、KID、CLIP‑FID 等特征级指标评估；IPv2 相比 IPv1 全部模型提升 50%+，FFM 在 IPv2 下达到最佳 27.02 的 FID，证明性能显著提升。

**⚠️ 局限性**

局限性包括：对噪声模型的匹配仍不完美，特别是核心扩散的采样策略需要进一步优化；IPv2 依赖人工设计的三模块，可能在其他解剖区域或不同扫描协议下效果有限。

---

## 633. Evaluation and Benchmarking Suite for Financial Large Language Models and Agents

**arXiv ID:** 2602.19073 | [PDF](https://arxiv.org/pdf/2602.19073v1)

**作者:** Shengyuan Lin `[一作]` (Carnegie Mellon University), Xiao-Yang Liu Yanglet `[通讯]` (Columbia University)

**通讯引用:** 2801 | [OpenAlex ID](https://openalex.org/A5100405230)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向金融LLM与代理的评估与基准套件，涵盖从探索到治理的全生命周期，并提供公开排行榜、评估流水线和治理框架。

**💡 创新点**

创新点在于：①将金融LLM的评估体系分为探索、准备与治理三阶段；②推出开放式FinLLM Leaderboard并结合MOF评估开放性；③引入AgentOps框架实现线上/线下双循环评估；④开发多模态金融代理如FinSearch、Tutor Agent、FinSight等。

**🔧 技术方法**

采用了多模态检索增强生成（RAG）、轨迹追踪、LLM‑as‑Judge、零知识证明、OpenAI GPT‑4/ChatGPT‑Turbo、FinGPT等技术。

**📊 数据集**

使用了FinBen、MultiFinBen、SEC filings（SECQUE、FinanceBench）、TAT‑QA、FinQA、TATQA、ECC 数据、GameStop事件文本、财经新闻等金融领域数据集。

**📈 对比分析**

评估方法通过排行榜得分、数值推理准确率、工具选择准确率、轨迹透明度等指标对比；目前FinSearch在数值推理任务中达到约85%准确率，高于Perplexity 55%，BloombergGPT在情绪分析任务中表现领先。

**⚠️ 局限性**

局限包括：基准尚未覆盖所有真实场景，仍存在幻觉与数据泄露风险，治理框架需进一步与监管合规结合，开放性评估对模型训练数据的透明度要求高，且多模态数据处理仍有技术瓶颈。

---

## 634. On the Dynamics of Observation and Semantics

**arXiv ID:** 2602.18494 | [PDF](https://arxiv.org/pdf/2602.18494v1)

**作者:** Xiu Li `[一作]` `[通讯]` (Bytedance Seed), Xiu Li (Bytedance Seed)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并证明了物理限制（Landauer原理与有限能量）决定了智能体必须将高熵感知投影到低熵语义基空间，形成观察‑语义纤维束，并由此推导出符号化、离散化与组合性的必然性，揭示语言与逻辑是信息固态的必要结构

**💡 创新点**

将语义定义为因果不变的等价类，建立观察‑语义纤维束理论；通过Landauer极限导出语义常数B并证明其迫使语义必须离散、可组合；提出“因果商（Causal Quotient）”作为理解的核心框架

**🔧 技术方法**

理论推导与形式化证明（纤维束几何、Kolmogorov复杂度、Landauer热力学约束、Minkowski几何）；符号化与组合性分析；对现有深度学习方法的理论分类（Gauge Fixing by Loss/Structure）

**📊 数据集**

无实验数据集，本文为理论性工作

**📈 对比分析**

无实验对比，本文通过严格的物理与信息论推导得到的结论；没有定量性能指标

**⚠️ 局限性**

缺乏可实验验证；对实际智能体的具体实现细节（如如何实际构建纤维束投影）未给出；对多模态、时序动态环境的适应性需要进一步研究

---

## 635. Predicting known Vulnerabilities from Attack News: A Transformer-Based Approach

**arXiv ID:** 2602.19606 | [PDF](https://arxiv.org/pdf/2602.19606v1)

**作者:** Refat Othman `[一作]` (Free University of Bozen-Bolzano), Barbara Russo `[通讯]` (Free University of Bozen-Bolzano)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 MPNet 句子变换器对网络安全新闻进行语义嵌入，计算攻击描述与 CVE 描述的余弦相似度，生成排名前 k 的 CVE 预测列表，并通过四种验证方法评估预测效果。

**💡 创新点**

在非结构化新闻文本中直接预测已知 CVE 的方法；构建并公开了首个标注的 SecurityWeek 新闻数据集；设计了多维度验证框架（手工评估、阈值过滤、首 CVE 相似、全部 CVE 相似）来客观衡量模型性能。

**🔧 技术方法**

MPNet 句子变换器（fine‑tuned on ATT&CK‑CVE 关联），余弦相似度评分，阈值过滤 (ρ = 0.58)，手工评估和三种自动化验证方法。

**📊 数据集**

训练集：MITRE ATT&CK 与 CVE 的映射（约 14k 条关联）；评估集：100 篇 SecurityWeek 安全新闻（97 篇含 CVE ID）。

**📈 对比分析**

采用四种验证方法：M1 手工评估 (70% 相关)，M2 阈值过滤 (81% 相关)，M3 与首 CVE 相似 (80% 相关)，M4 与所有 CVE 相似 (78% 相关)。在 100 篇文章中，57% 至少匹配一个真实 CVE ID，整体精度最高达 81%。

**⚠️ 局限性**

局限性：仅使用单一新闻来源且仅英文文本；top‑k 固定为 20，缺乏动态调整；模型仅为 MPNet，未尝试其他 transformer；阈值 ρ 需针对不同场景重新校准；未覆盖多语言或社交媒体等非结构化文本。

---

## 636. KGHaluBench: A Knowledge Graph-Based Hallucination Benchmark for Evaluating the Breadth and Depth of LLM Knowledge

**arXiv ID:** 2602.19643 | [PDF](https://arxiv.org/pdf/2602.19643v1)

**作者:** Alex Robertson `[一作]` (Newcastle University), Srijith Rajamohan `[通讯]` (Sage Group PLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于知识图谱的动态多维hallucination benchmark KGHaluBench，并构建自动化的实体级和事实级验证流程，对LLM的答案进行精细评估。

**💡 创新点**

创新点包括：①利用KG动态生成包含三条关系的复合问题，避免静态偏倚；②通过统计难度与Item Response Theory相结合的Sigmoid模型，得到加权准确率W_a；③将hallucination拆分为知识广度（Halu_BOK）和深度（Halu_DOK）两项指标，提供更可解释的分析。

**🔧 技术方法**

使用的技术包括：知识图谱检索（Wikidata）、动态问题生成、实体级文本相似度与NLI模型、LLM推理（事实验证）、专家决策后端、统计难度估计与加权准确率计算。

**📊 数据集**

使用的数据集为Wikidata（118M实体）及其对应的描述、统计信息，并在此基础上生成10轮150道题目，评测25款LLM（含公开与专有模型）。

**📈 对比分析**

比较方法：通过W_a、Halu_BOK、Halu_DOK三项指标对模型进行排名，结果显示GPT‑5获得最高65.60%的W_a；专有模型平均W_a为55.94%，开源模型为48.32%，专有模型在广度与深度hallucination上均表现更佳，且差距正在缩小。

**⚠️ 局限性**

limitations：①仅使用Wikidata，覆盖范围受限且不平衡（非英语、边缘主题欠缺）；②验证流程虽与人工判断达成高一致率，但仍可能误判；③缺乏多领域KG支持，限制了在专业场景下的hallucination评估。

---

## 637. Physics-Aware, Shannon-Optimal Compression via Arithmetic Coding for Distributional Fidelity

**arXiv ID:** 2602.19476 | [PDF](https://arxiv.org/pdf/2602.19476v1)

**作者:** Cristiano Fanelli `[一作]` `[通讯]` (William and Mary), Cristiano Fanelli (William and Mary)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个物理感知的无损算术编码器，用于对实验数据与生成或模拟数据的分布一致性进行评估，并将码长差作为信息量化的分布一致性指标。

**💡 创新点**

创新点在于将算术编码的码长差直接解释为交叉熵差/KL 散度，从而得到可解释、全局且可加的分布一致性度量；同时利用物理约束的概率分解实现压缩的可逆性和比特预算分析。

**🔧 技术方法**

采用物理约束的概率模型（占空、strip、ADC 的分层条件化，必要时以动量分箱条件化）结合算术编码；利用交叉熵/KL 理论、分块统计检验和 MMD 对照实验来验证方法。

**📊 数据集**

使用 CLAS12 电子量能计模拟数据（约 10⁶ 事件），包含 PCAL、ECIN、ECOUT 子系统的整数读数与粒子动量信息。

**📈 对比分析**

在相同的参考模型下，对真实数据与扰动样本进行独立分块编码，计算平均码长差 ΔL；与通用无损压缩（如 LZMA）相比，算术编码实现更高压缩比；相较 MMD，算术编码在更小的 ADC 扰动 ε 处即可显著拒绝零假设，显示更高灵敏度。

**⚠️ 局限性**

限制包括：需先学习固定概率模型，无法自适应新分布；有限样本导致估计误差；算术编码实现速度慢；仅评估已建模物理结构的分布一致性，可能忽略未建模的特征；模型复杂度和估计误差会影响码长和敏感性。

---

## 638. HeRO: Hierarchical 3D Semantic Representation for Pose-aware Object Manipulation

**arXiv ID:** 2602.18817 | [PDF](https://arxiv.org/pdf/2602.18817v1)

**作者:** Chongyang Xu `[一作]` (Sichuan University), Shuaicheng Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 6995 | [OpenAlex ID](https://openalex.org/A5039387461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 HeRO 框架，结合 Dense Semantic Lifting 与 Hierarchical Conditioning 的扩散式策略，实现对机器人姿态感知任务的细粒度语义感知与动作生成。

**💡 创新点**

创新点包括① 通过融合 DINOv2 的高分辨率语义特征与 Stable Diffusion 的全局语义先验，构建更稠密、更具辨识度的 3D 语义场；② 设计层次化的条件模块，既利用全局语义场，又通过无序排列的局部语义块实现对部件级语义的无序、可变条件；③ 在扩散策略中使用 permutation‑invariant 的交叉注意力，避免了传统顺序敏感的偏置。

**🔧 技术方法**

核心技术包括：3D 扩散策略（Diffusion Policy）、点云特征提取（PointNet、MLP）、DINOv2 与 Stable Diffusion 的特征融合、PCA 基础的部件划分、时间一致性的语义场递推、交叉注意力与自注意力的无序处理。

**📊 数据集**

在 RoboTwin 2.0 基准上进行评估，使用模拟环境与真实机器人（AgileX Cobot Magic + RealSense D435i）收集的数据，任务包括 Place Dual Shoes、Pick Diverse Bottles 等多种姿态感知挑战。

**📈 对比分析**

与 G3Flow、DP、DP3 等现有方法对比，HeRO 在 Place Dual Shoes 上提升 12.3% 的成功率，六项挑战任务平均提升 6.5%。在闭集与开集测试中，平均成功率分别从 32.3% 提升至 38.9%，以及从 24.4% 提升至 30.1%，并在真实环境中表现最优。

**⚠️ 局限性**

主要限制在于对 RGB‑D 传感器的依赖，特征融合与扩散模型的计算开销较大，导致实时性受限；此外，模型对高度动态或遮挡严重的场景仍需进一步鲁棒性验证。

---

## 639. How Well Can LLM Agents Simulate End-User Security and Privacy Attitudes and Behaviors?

**arXiv ID:** 2602.18464 | [PDF](https://arxiv.org/pdf/2602.18464v1)

**作者:** Yuxuan Li `[一作]` (Carnegie Mellon University), Sauvik Das `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2767 | [OpenAlex ID](https://openalex.org/A5006053551)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究评估大语言模型代理能否模拟人类在安全与隐私（S&P）领域的态度、行为和一致性，并提出了SP‑ABCBench基准。

**💡 创新点**

创新点在于构建基于真实人类实验的30项测评基准，系统量化LLM模拟与经验数据的一致性，并揭示模型规模、人物构造和提示策略对模拟质量的影响。

**🔧 技术方法**

技术上使用12个大型语言模型、四种人物构造方法、基于隐私计量学和有限理性的一致性提示，并定义了跨维度的模拟质量评分。

**📊 数据集**

数据集来自15项已验证的S&P实验研究（如IUIPC、SeBIS、SA‑6、OPC等）以及基于美国人口普查的合成样本。

**📈 对比分析**

比较方法采用每项测试的对齐分数（0–100），结果显示平均分仅在50–64之间，规模更大或更新的模型并不总是更好，特定配置可达到95+的高分。

**⚠️ 局限性**

局限包括基准主要覆盖美国样本、单步情境、未使用微调或记忆机制，且当前模型仍难以精准复现实验效应。

---

## 640. A Three-stage Neuro-symbolic Recommendation Pipeline for Cultural Heritage Knowledge Graphs

**arXiv ID:** 2602.19711 | [PDF](https://arxiv.org/pdf/2602.19711v1)

**作者:** Krzysztof Kutt `[一作]` (Jagiellonian University), Luiz do Valle Miranda `[通讯]` (Jagiellonian University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该论文提出并实现了一个三阶段神经符号推荐流水线，用于文化遗产知识图谱的个性化推荐。

**💡 创新点**

创新点在于将KG嵌入、近似最近邻检索与基于SPARQL的语义过滤相结合，兼顾召回率与可解释性，并针对文化遗产数据的稀疏、异构特征进行专门设计。

**🔧 技术方法**

技术包括PyKEEN训练的ComplEx嵌入、HNSWLib构建的近似最近邻索引，以及自定义SPARQL查询实现的语义过滤；实验平台使用CPU与GPU资源。

**📊 数据集**

数据集为Jagiellonian University Heritage Metadata Portal（JUHMP）知识图谱，包含两个版本：CHExRISH_Prototype2（≈402k triples）与完整版CHExRISH_FullCAC_260128（≈3.2M triples）。

**📈 对比分析**

通过对四种KG嵌入模型的比较和对ComplEx的超参数调优、对HNSW参数的网格搜索，最终的流水线在Prototype2上MRE=0.2207、Hits@1=0.1659、Hits@10=0.3288，完整版在MRE=0.1393、Hits@1=0.1057、Hits@10=0.2098；专家评估显示推荐结果具有可解释性且大部分被评为“近连接”。

**⚠️ 局限性**

限制包括元数据稀疏导致语义过滤过滤掉过多候选；对时间与地点信息的依赖使得缺乏结构化链接时效果下降；以及对实体名称的模糊匹配导致误判，需要外部唯一ID支持。

---

## 641. FORMICA: Decision-Focused Learning for Communication-Free Multi-Robot Task Allocation

**arXiv ID:** 2602.18622 | [PDF](https://arxiv.org/pdf/2602.18622v1)

**作者:** Antonio Lopez `[一作]` (Worcester Polytechnic Institute), Carlo Pinciroli `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3160 | [OpenAlex ID](https://openalex.org/A5034543991)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种无需机器人间通信即可完成多机器人任务分配的学习框架 FORMICA。

**💡 创新点**

创新点在于：①首次将决策聚焦学习（Smart Predict‑then‑Optimize）应用于多代理协调；②利用均值场近似将预测复杂度从 O(NT) 降到 O(T)，实现大规模无缝迁移；③通过端到端训练将预测器与任务分配目标直接对齐，显著提升分配质量。

**🔧 技术方法**

主要技术包括：深度预测器（对竞争者投标分布进行建模）、均值场近似、决策聚焦学习（SPO）框架、端到端优化训练。

**📊 数据集**

使用的实验数据集为人工生成的平面任务分配场景：训练集包含 16 台机器人与 64 个任务（任务位置为 6 个高斯聚类，reward ∈ [6,24]），测试集包括相同规模的 100 个随机实例以及 256 台机器人、4096 个任务的大规模扩展场景。

**📈 对比分析**

与基线的比较：与理想化的 Analytical Mean Field（AMF）以及小规模的 MILP 最优解进行对比。结果显示 FORMICA 在训练规模下比 AMF 提升约 17% 的系统奖励，在大规模扩展下提升约 7%，且覆盖率略低但目标收益更高；在 100% 场景中均优于 AMF。

**⚠️ 局限性**

局限性包括：仅验证同质机器人和静态任务场景；仅采用单一几何投标函数，未考虑障碍物、边界效应或能量/时限约束；训练时忽略了覆盖概率梯度，可能限制了在覆盖至关重要场景下的性能。

---

## 642. Yor-Sarc: A gold-standard dataset for sarcasm detection in a low-resource African language

**arXiv ID:** 2602.18964 | [PDF](https://arxiv.org/pdf/2602.18964v1)

**作者:** Toheeb Aduramomi Jimoh `[一作]`, Nikola S. Nikolov `[通讯]` (University of Limerick)

**通讯引用:** 1315 | [OpenAlex ID](https://openalex.org/A5088624697)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了首个 Yor‑Sarc 讽刺语料库，包含 436 条 Yoruba 文本，采用三位母语注释者进行二元标注，并系统分析注释者一致性。

**💡 创新点**

创新点包括：① 采用文化敏感、上下文依赖的标注协议；② 记录软标签以保留不确定性；③ 在低资源语言上取得超过所有公开 benchmark 的高一致性。

**🔧 技术方法**

使用了多注释者评估技术（Cohen κ、Fleiss κ）、软标签生成与熵度量、热图和混淆矩阵来分析注释质量和不确定性。

**📊 数据集**

使用的数据集为 436 条来源多样的 Yoruba 文本，来自 BBC News Yoruba、X/Twitter、Facebook、Instagram、YouTube 及众包调查，全部采用标准化 Yoruba 书写。

**📈 对比分析**

通过与英语、印地语等已有讽刺数据集的 κ 值对比评估，平均 pairwise κ 为 0.7671，最佳 κ 为 0.8743，远高于公开 benchmark；软标签分布显示 83.3% 为统一标注，验证了标注协议的有效性。

**⚠️ 局限性**

局限性在于数据域覆盖不足，仅涵盖社交媒体和新闻文本，缺少面对面对话等场景，未来需扩展多域覆盖以提升泛化能力。

---

## 643. Efficient Dynamic Test Case Generation for Path-Based Coverage Criteria

**arXiv ID:** 2602.18768 | [PDF](https://arxiv.org/pdf/2602.18768v1)

**作者:** Jakub Zelek `[一作]` (Jagiellonian University), Artur Polański `[通讯]` (Jagiellonian University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5029184785)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于改进 Johnson 算法的流式动态测试用例生成方法，能够按需生成满足 Prime Path、Simple Cycle、Simple Path 与 e‑Acyclic Path 覆盖标准的路径；

**💡 创新点**

主要创新在于对 prime path 的强连通分量表征与“Ex”图构造，实现在不存储全部路径的情况下即时生成，并支持并行化；

**🔧 技术方法**

使用改进的 Johnson 算法、Tarjan 强连通分量、线图（L(G)）转换及 Python 生成器（yield/yield from）实现流式输出；

**📊 数据集**

在包含 120,314 个来自 GitHub 的 C++ 与 Python 函数的控制流图上进行实验，平均每个函数约 30 个节点、42 条边；

**📈 对比分析**

与 Ammann‑Offutt、Fazli 等传统方法相比，小型图平均执行时间下降约 88%，大型图实现流式生成，单路径平均生成时间为 0.0001 s，内存占用仅 0.5 MB；

**⚠️ 局限性**

限制主要在于对强连通分量的依赖，极大 SCC 或高度稠密图仍可能产生较多中间计算；此外，某些 NP‑hard 的起点/终点可扩展性问题尚未完全解决。

---

## 644. Scale-PINN: Learning Efficient Physics-Informed Neural Networks Through Sequential Correction

**arXiv ID:** 2602.19475 | [PDF](https://arxiv.org/pdf/2602.19475v1)

**作者:** Pao-Hsiung Chiu `[一作]` (Astar), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 27785 | [OpenAlex ID](https://openalex.org/A5068243197)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于迭代残差校正的Physics‑Informed Neural Network（Scale‑PINN），通过将数值求解器中的残差平滑与顺序校正机制嵌入PINN损失函数，实现高效训练；

**💡 创新点**

创新点在于将经典数值方法中的迭代残差校正原理直接映射到PINN损失中，引入辅助校正项，显著加速收敛并提升稳定性；

**🔧 技术方法**

使用自动微分实现的多层感知机（MLP）网络，结合频率热身、Sine激活、Skip‑Connection结构以及残差平滑算子（Helmholtz 过滤器）和Adam/SGD优化器；

**📊 数据集**

针对多种 PDE 进行验证，包括二维拉格朗日驱动腔流（Re=400~20k）、NACA0012 气动翼、三棱柱气流、Rayleigh‑Bénard 对流、Kuramoto‑Sivashinsky、Gray‑Scott、Korteweg–De Vries、Allen‑Cahn 等；

**📈 对比分析**

与传统 PINN、LSA‑PINN、PirateNets、TSA‑PINN、FFV‑PINN 以及高阶数值求解器（Fluent、IDFC²）比较，Scale‑PINN 在相同硬件上从 15 小时缩短到 2 分钟内达到 2% 以内相对误差；在 Re=3200、7500、10k 的腔流中实现 90–150 秒训练时间，且精度与最优数值解相当；

**⚠️ 局限性**

局限性包括对超参数 τ_sc、τ_α、γ 的经验调优需求；目前主要针对稳态或可微分的 PDE，且对高度非线性或极端多尺度问题的进一步验证仍待扩展；

---

## 645. DSDR: Dual-Scale Diversity Regularization for Exploration in LLM Reasoning

**arXiv ID:** 2602.19895 | [PDF](https://arxiv.org/pdf/2602.19895v1)

**作者:** Zhongwei Wan `[一作]` (Ohio State University), Mi Zhang `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双尺度多样性正则化框架DSDR，用于强化大语言模型在基于可验证奖励的强化学习（RLVR）中的探索。

**💡 创新点**

创新点在于同时对正确推理轨迹进行全局多样性奖励，并对同一轨迹的 token 层面引入长度不变的熵正则化；两尺度通过全局‑>局部分配机制紧耦合，确保仅在多样性高的正确轨迹上加大熵扩散，从而避免模式崩塌并保持学习信号。

**🔧 技术方法**

技术包括：GRPO（Group Relative Policy Optimization）作为优化骨架，冻结文本编码器计算嵌入相似度、公式唯一性评分构造全局多样性；token‑level 条件熵正则化与重要性采样；全局‑>局部 softmax 分配；以及与 RLVR 损失的联合优化。

**📊 数据集**

数据集涵盖多种数学推理任务：AIME2024、AIME2025、MATH500、Minerva 以及 Olympiad。模型在 Qwen2.5‑Math‑1.5B、Qwen3‑1.7B、Qwen3‑4B 三个规模上进行训练与评估。

**📈 对比分析**

与基准模型（Backbone）、GRPO、DAPO 进行对比，使用 Pass@1、Avg@16 以及 Pass@k（k=2…64）等指标。实验表明 DSDR 在所有模型规模和所有基准任务上均实现显著提升，特别是在 Pass@k 上表现更优，说明其扩展了正确推理轨迹集合而非仅收敛单一解法。

**⚠️ 局限性**

局限性包括：对正则化系数 λ_d、λ_ℓ 以及温度 τ 的敏感性，需要经验调参；在奖励稀疏或验证器不完善的任务中，正确‑only 多样性奖励可能无法充分激励探索；目前仅在数学推理任务上验证，泛化到其他推理场景（如常识推理、代码生成）仍需进一步探索。

---

## 646. Altar: Structuring Sharable Experimental Data from Early Exploration to Publication

**arXiv ID:** 2602.18588 | [PDF](https://arxiv.org/pdf/2602.18588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 647. Fore-Mamba3D: Mamba-based Foreground-Enhanced Encoding for 3D Object Detection

**arXiv ID:** 2602.19536 | [PDF](https://arxiv.org/pdf/2602.19536v1)

**作者:** Zhiwei Ning `[一作]`, Wei Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了ICLR 2026会议论文的格式规范与排版指南，说明如何使用ICLR提供的LaTeX样式文件、排版参数、标题、作者、摘要、章节标题、引用、图表、脚注等元素的具体格式要求。

**💡 创新点**

创新点在于统一了ICLR会议论文的排版标准，并在NeurIPS格式的基础上细化了细节，如严格的页边距、行距、字号、标题样式、图表与参考文献格式等，确保提交稿件符合会议的技术规范。

**🔧 技术方法**

使用的技术主要是LaTeX排版系统，配合ICLR官方提供的cls、sty文件，引用natbib包处理引用，graphicx包处理图形，以及通过PDF/PS生成US Letter尺寸的最终稿件。

**📊 数据集**

无（该文档仅为排版规范，不包含实验数据或数据集）。

**📈 对比分析**

无（本文不涉及方法比较或性能评估，只提供排版与提交流程的技术细节）。

**⚠️ 局限性**

限制在于该规范仅覆盖排版与提交格式，并未涉及论文内容的学术质量评估，且依赖作者严格遵循样式文件，否则可能导致稿件被退回。

---

## 648. Bellman Value Decomposition for Task Logic in Safe Optimal Control

**arXiv ID:** 2602.19532 | [PDF](https://arxiv.org/pdf/2602.19532v1)

**作者:** William Sharpless `[一作]` (University of California San Diego), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1902 | [OpenAlex ID](https://openalex.org/A5019603699)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出将复杂时序逻辑（TL）任务的 Bellman 值分解为由基础 Bellman 方程构成的有向图，并基于此设计了 VDPPO 算法来学习高维任务的最优策略。

**💡 创新点**

创新点包括：①证明 TL 谓词的 Bellman 值可精确分解为 Reach‑Avoid、Avoid 与 Reach‑Avoid‑Loop 三类基础 Bellman 方程构成的图；②提出对应的 RAA_ℓ‑BE 以及自动生成 DVG 并利用动态规划求解子问题；③将分解结构嵌入 PPO，形成 VDPPO，从而在不需手工设计奖励或构造自动机的情况下解决复杂 TL 任务。

**🔧 技术方法**

使用了时序逻辑与 Hamilton–Jacobi Reachability（HJR）理论、Bellman 方程分解、PPO 的变体 VDPPO、动态规划、自动 DVG 生成与神经网络嵌入等技术。

**📊 数据集**

实验主要在自建的仿真环境（双积分器、herding、delivery、机械臂）以及真实硬件平台（Crazyflie 与 Unitree Go2）中进行，使用随机初始状态和自定义 TL 公式，没有引用公开数据集。

**📈 对比分析**

与 LCRL、TL‑MPPI 等基线在相同训练步骤下对比；评估指标为 TL 终端满足率及子公式满足率；在 TL 复杂度、智能体数量和动力学难度三个维度上，VDPPO 在所有实验中均表现出更高的成功率，尤其在深度 TL 及多智能体场景中优势明显。

**⚠️ 局限性**

局限性：①依赖于 TL 公式可被分解为预定义 Bellman 方程的结构，对某些非分解型 TL 或高度连续状态空间可能缺乏收敛保证；②自动 DVG 生成和多层网络嵌入实现复杂度较高；③在极大规模智能体或极其复杂动力学下的可扩展性和计算成本尚待进一步验证。

---

## 649. GrIT: Group Informed Transformer for Sequential Recommendation

**arXiv ID:** 2602.19728 | [PDF](https://arxiv.org/pdf/2602.19728v1)

**作者:** Adamya Shyam `[一作]` (University of Delhi), Vikas Kumar `[通讯]` (University of Delhi)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种在序列推荐中同时建模用户个体历史和随时间演化的群体特征的Transformer框架GrIT，以提升下一个物品预测准确率。

**💡 创新点**

创新点在于通过统计特征估计用户的动态群体归属权重，并将可学习的群体嵌入与个体序列表示融合，首次在Transformer中引入可随时间变化的群体亲和度。

**🔧 技术方法**

使用Transformer自注意力模块、指数加权移动平均（EWMA）统计特征、可学习位置编码、软max归一化的群体权重、组聚合与融合机制，并以交叉熵训练。

**📊 数据集**

在五个公开基准数据集上实验：Amazon Video Games、Industrial & Scientific、CDs & Vinyl，以及GroupLens的MovieLens 100K/1M。

**📈 对比分析**

与FMLPRec、DuoRec、LinRec、BSARec、LRURec等SOTA方法在Recall@k、NDCG@k、MRR@k上对比，GrIT在所有数据集与指标上均显著优于基线，提升幅度明显。

**⚠️ 局限性**

局限性包括需预设群体数目与温度参数，对极稀疏数据群体表示的可靠性有限，模型复杂度较高，且未探究跨域迁移与实时推理的适配性。

---

## 650. Debug2Fix: Supercharging Coding Agents with Interactive Debugging Capabilities

**arXiv ID:** 2602.18571 | [PDF](https://arxiv.org/pdf/2602.18571v1)

**作者:** Spandan Garg `[一作]` (Microsoft), Yufan Huang `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入 Debug Subagent，将交互式调试嵌入编码代理，主代理通过该子代理获取运行时信息并指导修复。

**💡 创新点**

通过子代理架构把复杂的调试流程抽象为单一工具，既解决了直接暴露调试器导致的低使用率和不稳定性，又显著提升了模型的bug修复表现。

**🔧 技术方法**

结合大型语言模型（GPT‑5、Claude Sonnet/Haiku）、Java/Python 调试器（JDB/PDB）、子代理调用与工具限制策略，并在基准评估中使用。

**📊 数据集**

在 GitBug‑Java（186 个 Java bug）和 SWE‑Bench‑Live Python（400 个实例）两个基准上进行实验。

**📈 对比分析**

与基线、仅暴露调试工具、Debug2Fix 与工具限制四种配置对比，实验显示在 GitBug‑Java 上 Debug2Fix 的通过率提升最高可达 21.8% 以上，在 Python 版亦提升 12% 以上；弱模型可匹敌或超越强模型基线。

**⚠️ 局限性**

仅支持 Java/Python，需项目可构建且有可执行测试，主子代理使用相同模型，工具限制可能对简单 bug 过度复杂，且未测量调试器本身的运行开销等局限。

---

## 651. HeatPrompt: Zero-Shot Vision-Language Modeling of Urban Heat Demand from Satellite Images

**arXiv ID:** 2602.20066 | [PDF](https://arxiv.org/pdf/2602.20066v1)

**作者:** Kundan Thota `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用零射击视觉语言模型从卫星图像中提取语义特征，结合GIS和建筑层级信息，估算城市热需求。

**💡 创新点**

首次提出Zero-shot语义热映射框架HeatPrompt，既能自动获取可解释的能源相关特征，又无需手工标注，显著提升预测精度。

**🔧 技术方法**

采用预训练的VLM（如GPT‑4o、CLIP）进行文本提示生成语义嵌入，并用多层感知机回归器整合语义与结构特征。

**📊 数据集**

使用德国莱茵-普法尔茨能源图谱（RLP Energy Atlas）与Esri世界影像以及OpenStreetMap、LOD2建筑高度等公开遥感与GIS数据。

**📈 对比分析**

通过五折交叉验证与基准模拟模型对比，HeatPrompt在R²上提升93.7%（从0.32到0.62），MAE降低约30%。

**⚠️ 局限性**

受限于VLM文本生成的可靠性、遥感分辨率与缺失的详细建筑参数，模型在极端地区和建筑类型上仍可能出现误差。

---

## 652. Ada-RS: Adaptive Rejection Sampling for Selective Thinking

**arXiv ID:** 2602.19519 | [PDF](https://arxiv.org/pdf/2602.19519v1)

**作者:** Yirou Ge `[一作]` (PayPal AI), Prakhar Mehrotra `[通讯]` (PayPal AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在成本与延迟敏感环境下的LLM选择性思考，提出Ada‑RS框架通过自适应长度惩罚和随机拒绝采样过滤训练样本以提升效率；

**💡 创新点**

提出一种算法无关的Adaptive Rejection Sampling机制，能够在不同优化器（DPO、DAPO）中动态选择最有价值的样本或对，结合自适应长度惩罚显著减少冗余推理；

**🔧 技术方法**

使用自适应长度惩罚（ALP）奖励、随机拒绝采样（对/组），结合DPO（对比学习）和DAPO（分组策略优化），并在Qwen3‑8B基础模型上采用LoRA微调；

**📊 数据集**

基于合成多轮电商工具调用数据集（类似tau^2‑Bench），共15k训练样本，2.5k评估样本，使用Qwen3‑8B模型；

**📈 对比分析**

与无微调提示（NFT）、SFT、DPO、DAPO等基线对比。Ada‑RS‑DPO与Ada‑RS‑DAPO在保持或提升工具调用准确率的同时，平均输出token减少约70‑80%，思考率降低约95%，显著提升准确‑效率前沿；

**⚠️ 局限性**

仅在单一域和单一模型规模（Qwen3‑8B）下验证；评估指标聚焦工具调用准确率，未涵盖多轮任务完成、用户满意度等，更广泛的领域和更大规模模型需要进一步验证。

---

## 653. A Benchmark and Knowledge-Grounded Framework for Advanced Multimodal Personalization Study

**arXiv ID:** 2602.19001 | [PDF](https://arxiv.org/pdf/2602.19001v1)

**作者:** Xia Hu `[一作]` (Google DeepMind), Howard Zhou `[通讯]` (Google DeepMind)

**通讯引用:** 1545 | [OpenAlex ID](https://openalex.org/A5056352820)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于虚拟账户的多模态个性化评测基准 LifeGraph，并构建了结构化检索框架 LifeGraph；

**💡 创新点**

① 通过全合成的多模态数据构建真实感的复杂个性化任务；② 将个人历史与关系信息建成知识图，实现高效结构化检索；③ 在关系、时间与聚合推理任务上显著超越传统检索方法；

**🔧 技术方法**

使用 Vision‑Language Models（Gemini 2.5 Pro/Flash Image、Gemma‑3）、改进版 Think‑on‑Graph 图检索算法以及图结构化构建流程；

**📊 数据集**

全合成的 LifeGraph 数据集（10 个虚拟账户、16,315 题/答、2,887 张图像），文本种子来源于 YFCC100M 图像说明；

**📈 对比分析**

与 RAP、R2P、RAG 等检索增强个性化基线在 10 个任务上对比，LifeGraph 在 7/10 任务上取得最高分，尤其在关系/时间推理任务上提升 ≥0.1 的准确率；

**⚠️ 局限性**

仅使用合成数据，缺乏真实用户多模态行为；图检索深度/宽度对性能敏感；需要更强的 backbone 模型来处理检索上下文噪声。

---

## 654. Validated Code Translation for Projects with External Libraries

**arXiv ID:** 2602.18534 | [PDF](https://arxiv.org/pdf/2602.18534v1)

**作者:** Hanliang Zhang `[一作]` (Amazon Inc.), Taro Sekiyama `[通讯]` (National Institute of Informatics)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5018002187)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一套基于检索增强生成（RAG）的框架，完成Go项目到Rust项目的完整迁移，并通过跨语言适配器实现了功能等价验证。

**💡 创新点**

创新点在于：① 针对外部库的API映射与导入解析，构建了按Crate划分的知识库；② 通过生成共享的Proto介质，合成适配器实现Go‑Rust互操作；③ 在验证阶段采用双向round‑trip检验与差分I/O测试，专门处理不透明的库类型。

**🔧 技术方法**

主要技术包括检索增强生成（RAG）、密集向量检索+交叉编码reranker、LLM生成代码与适配器、Proto介质的schema生成、跨语言round‑trip验证以及差分测试。

**📊 数据集**

使用了六个真实世界的Go开源项目（涵盖密码学、密码哈希、时间解析等依赖域），每个项目均包含非平凡的外部库调用。

**📈 对比分析**

与不使用RAG或导入解析的基线（以及手工硬编码Crate的基线）对比，本文方法在所有项目中实现了100%编译率和I/O等价率，平均提升约2倍；单独去除RAG或导入解析会导致编译率和等价率显著下降。

**⚠️ 局限性**

局限性包括：① 仍依赖LLM生成的代码质量，偶尔产生语义错误；② 适配器生成与验证需要大量源测试数据；③ 目前仅针对Go→Rust迁移，泛化到其他语言需额外工作。

---

## 655. Training-Free Cross-Architecture Merging for Graph Neural Networks

**arXiv ID:** 2602.19332 | [PDF](https://arxiv.org/pdf/2602.19332v1)

**作者:** Rishabh Bhattacharya `[一作]` (International Institute of Information Technology), Naresh Manwani `[通讯]` (International Institute of Information Technology)

**通讯引用:** 851 | [OpenAlex ID](https://openalex.org/A5007462471)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了H-GRAMA框架，用于训练无、标签无的跨架构图神经网络合并。

**💡 创新点**

创新点是将模型合并从参数空间迁移到统一的消息传递算子空间（UMPM），并使用CKA对齐深度、Procrustes对齐参数、闭式门回归融合算子、LFNorm校正消息分布。

**🔧 技术方法**

技术包括UMPM算子基，CKA相似度匹配，Procrustes半正交变换，闭式门回归（岭回归），LFNorm自适应消息校准，以及基于自信度的α选择。

**📊 数据集**

使用Cora、CiteSeer、Actor、Amazon‑Ratings、Arxiv等数据集进行评估。

**📈 对比分析**

与现有同构和异构合并基线相比，H-GRAMA在同构下保持90%+的保留率，在异构下实现约90%的保留率，并在Arxiv上实现1.2–1.9×的推理加速，整体性能优于或匹配基线。

**⚠️ 局限性**

局限性包括深度不匹配导致保留率下降，尤其是包含GIN等MLP重的架构，且线性Procrustes对非线性变换捕获不足，未来可改进更丰富的对齐方法。

---

## 656. Why ReLU? A Bit-Model Dichotomy for Deep Network Training

**arXiv ID:** 2602.19017 | [PDF](https://arxiv.org/pdf/2602.19017v1)

**作者:** Ilan Doron-Arad `[一作]` (Massachusetts Institute of Technology), Elchanan Mossel `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 9803 | [OpenAlex ID](https://openalex.org/A5013467728)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文在有限精度的位级模型下，研究了深度神经网络的经验风险最小化（ERM）与反向传播的理论复杂度。

**💡 创新点**

创新点在于揭示激活函数的多项式性与分段线性性对训练复杂度的根本区分：多项式激活导致 #P 难度甚至 BPP 难题，而分段线性激活保持 NP 完全与多项式时间可验证。

**🔧 技术方法**

主要技术包括把位级 SLP（直线程序）与网络算子嵌入相结合的归约，利用 PosSLP 与 BitSLP 的 #P 难度，以及构造多项式激活的乘法“gadget”。

**📊 数据集**

未使用实际数据集，全部为理论构造的输入实例。

**📈 对比分析**

与传统的实数模型相比，本文在位级模型下提供了更严格的可计算性边界，证明了深度多项式激活的训练是 #P‑hard，显著高于实数模型中的 ∃ℝ‑难度；而对 ReLU 等分段线性激活，则仍在 NP 范围内，后向传播可在多项式时间完成。

**⚠️ 局限性**

局限性包括：仅适用于具有有理系数多项式激活和分段线性激活的理论分析；对非整数指数或负指数的多项式无法扩展；实验验证缺失；以及仅关注最坏情况而不说明典型实例的实际复杂度。

---

## 657. Post-Routing Arithmetic in Llama-3: Last-Token Result Writing and Rotation-Structured Digit Directions

**arXiv ID:** 2602.19109 | [PDF](https://arxiv.org/pdf/2602.19109v1)

**作者:** Yao Yan `[一作]` (Chongqing Normal University), Yao Yan `[通讯]` (Chongqing Normal University)

**通讯引用:** 6086 | [OpenAlex ID](https://openalex.org/A5039992067)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Meta‑Llama‑3‑8B 在三位数加法任务中的结果最终化过程，定位了从跨词路由变为单词路由的切换点，并通过低秩正交变换验证了数字方向在不同上下文间的可迁移性。

**💡 创新点**

创新点包括：1）在单一令牌读取协议下精确定位结果写入边界（约层 17），并证明该层后自注意力作用可忽略；2）发现数字/和方向在不同高位上下文间通过低秩正交变换保持一致；3）通过旋转该变换恢复跨上下文编辑效果，从而首次将几何对齐与因果编辑结合。

**🔧 技术方法**

采用的技术包括：单令牌读取协议、跨样本残差补丁、累积注意力消融、差分均值（DoM）学习一维功能方向、低秩 Procrustes 对齐、基于方向的向量编辑（移除+添加）。

**📊 数据集**

使用的数据集是由两数 a、b ∈ {1,…,999} 生成的三位数加法实例，限制总和 s = a+b 在 [200,999] 内，确保结果为三位数且可映射为单一令牌。

**📈 对比分析**

方法与基线的比较：基线严格准确率约 99%；残差补丁显示从层 16 开始结果完全由最后 token 控制；累积注意力消融表明层 17 之后自注意力可忽略；旋转编辑在跨上下文场景下恢复到直接编辑的成功率，验证了低秩对齐的有效性。

**⚠️ 局限性**

局限性包括：仅在三位数加法、单一模型（Meta‑Llama‑3‑8B）与单令牌读取下验证；未探究更长数字、其他算术、不同 tokenizer 或多令牌生成场景；仅聚焦后期结果写入，未解释早期 carry 计算或信息路由；方法可能忽视非线性结构或更高阶交互。

---

## 658. When Agda met Vampire

**arXiv ID:** 2602.18844 | [PDF](https://arxiv.org/pdf/2602.18844v1)

**作者:** Artjoms Šinkarovs `[一作]` (University of Southampton), Michael Rawson `[通讯]` (University of Southampton)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5080196917)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在Agda中实现了一种轻量级的hammer式自动定理证明器，将依赖类型证明助手与经典一阶逻辑ATP Vampire相结合，能够将Agda中的等式Horn子句自动翻译成ATP问题，随后通过Prolog重构生成可在Agda中检查的构造性证明。

**💡 创新点**

创新点在于：①只针对可翻译的等式Horn子句实现了双向翻译，避免了复杂的深度集成；②利用Agda的反射API轻松导出目标与前提；③通过Prolog脚本对ATP的TSTP证明进行反向搜索，生成可验证的Agda证明术语；④在实际FFT及根号圆性质的证明中展示了显著的自动化效果，节省了人工证明时间。

**🔧 技术方法**

使用技术包括：Agda反射API（用于提取目标与环境）、Vampire ATP（提供经典一阶逻辑推理）、Prolog（实现证明重构与证明步骤检索）、SMT‑LIB/TSTP格式（作为ATP与Agda之间的中介）。

**📊 数据集**

主要使用了TPTP基准集中的Horn子句示例进行实验；另外在案例研究中使用了FFT相关的复数与根号圆定义与公理，手工写成的12条定理作为测试目标。

**📈 对比分析**

与手工证明相比，自动化系统在12条定理上几乎实时完成（毫秒级）而原始手工需要两天；与传统Agda内置搜索相比，能处理更复杂的组合与推理；与Sledgehammer等系统的比较显示，本方案在依赖类型环境下保持了完整的可信度且实现成本低。

**⚠️ 局限性**

局限性包括：①仅支持等式Horn子句，无法处理完整的Clausal Logic；②缺乏前提选择与完整的hammer自动化流程；③对经典与构造性逻辑差异的处理仍需手工规则；④在更大规模或更复杂理论（如依赖类型、参数化类型）上的可扩展性待验证。

---

## 659. SceneTok: A Compressed, Diffusable Token Space for 3D Scenes

**arXiv ID:** 2602.18882 | [PDF](https://arxiv.org/pdf/2602.18882v1)

**作者:** Mohammad Asim `[一作]` (Max Planck Institute for Informatics), Jan Eric Lenssen `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 2575 | [OpenAlex ID](https://openalex.org/A5073462022)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 SceneTok，一个将场景视图集合压缩成无序、极度压缩的 token 的自编码器，并配合轻量级 Rectified Flow 解码器实现快速新视角合成和高效场景生成。

**💡 创新点**

创新点在于：①将 3D 场景信息脱离空间网格，以 permutation‑invariant 的 token 形式编码；②采用 Perceiver 与 AdaLN 进行多视角融合；③将渲染与生成解耦，利用极度压缩的 token 进行可扩散的 latent 生成；④实现仅 5–8 秒即可完成完整场景生成。

**🔧 技术方法**

核心技术包括：VA‑VAE 图像压缩、Perceiver 结构、AdaLN 位置编码、2D RoPE、Rectified Flow 逆扩散网络、VideoDCAE 解码器、以及基于 Transformer 的 latent diffusion。

**📊 数据集**

在 RealEstate10K、DL3DV 以及 ACID 数据集上进行训练与评测；通过多视角数据和单视角生成任务展示效果。

**📈 对比分析**

与 MVSplat、MVSplat360、DepthSplat、LVSM、RayZer 等基线相比，SceneTok 在 PSNR、LPIPS、SSIM、rFVD、rFID 等指标上均取得更优或相近的表现，渲染速度可达 32 帧/秒，生成速度为 5–8 秒，显著优于传统 3D 生成与视频生成模型。

**⚠️ 局限性**

主要局限是对高频细节的还原能力不足，因极度压缩导致细节丢失；并且缺乏显式 3D 结构，对精细几何重建存在挑战。

---

## 660. The Human Factor in Data Cleaning: Exploring Preferences and Biases

**arXiv ID:** 2602.19368 | [PDF](https://arxiv.org/pdf/2602.19368v1)

**作者:** Hazim AbdElazim `[一作]` (Western University), Mostafa Milani `[通讯]` (Western University)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5072207353)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对51名参与者进行受控调查，研究在错误检测、缺失值填补与实体匹配等核心数据清洗任务中出现的认知偏差。

**💡 创新点**

首次将代表性启发式、框架效应、锚定偏差和省略偏差等经典心理学机制系统性映射到具体的数据清洗行为，并证明其在多样化工作场景中普遍存在。

**🔧 技术方法**

采用结构化问卷设计与实验情境，并利用精确二项检验评估偏差出现频率；同时使用可解释的基准阈值（如稀有错误10%或多数50%）来判定偏差显著性。

**📊 数据集**

基于加拿大人口普查公开使用数据文件（PUMF）的子集，构造含有语义有效但异常、格式变异、缺失与命名冲突等特征的模拟记录。

**📈 对比分析**

对各场景下偏差率进行统计比较，发现代表性偏差导致约60%参与者误标异常记录，框架效应约57%误标格式差异，锚定偏差约67%顺从专家标记，省略偏差约67%选择不填缺失值；自动化建议遵循率仅≈10%不显著。

**⚠️ 局限性**

局限包括样本量有限、受试者为技术背景较为集中的群体、情境过度简化、仅测试单一错误类型与干预缺失，因而结果可能不完全适用于更大规模或多样化的真实清洗流程。

---

## 661. RL-RIG: A Generative Spatial Reasoner via Intrinsic Reflection

**arXiv ID:** 2602.19974 | [PDF](https://arxiv.org/pdf/2602.19974v1)

**作者:** Tianyu Wang `[一作]` (Shanghai Jiao Tong University), Bowen Zhou `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 Generate‑Reflect‑Edit 的 RL‑RIG 框架，利用 VLM 的链式推理和强化学习迭代校正图像，解决文本到图像生成中的空间关系推理难题。

**💡 创新点**

创新点在于将 VLM 的自我反思作为内在奖励，结合 GRPO 对生成轨迹进行优化，且无需额外布局输入即可实现多轮空间关系校正，首次在空间一致性上实现显著提升。

**🔧 技术方法**

核心技术包括 Flux 扩散模型、RF‑Inversion 图像编辑器、Curr‑ReFT（基于 Qwen‑2.5）VLM 进行链式推理和检查、Group‑Relative Policy Optimization（GRPO）强化学习、Scene Graph IoU 评价。

**📊 数据集**

实验基于 LAION‑SG 数据集，选取最复杂的空间关系子集进行训练与评测。

**📈 对比分析**

与 SD3.5 Large、Flux、LAION‑SG 基线在 SG‑IoU、Ent‑IoU、Rel‑IoU、Qwen‑Judge、GPT‑Judge 等指标对比，RL‑RIG 经过后训练后 SG‑IoU 提升 11% 以上，整体在所有空间一致性指标上均位居第一。

**⚠️ 局限性**

局限性包括：性能高度依赖底层模型，缺乏更大规模复杂空间关系数据集，图像编辑器在多轮编辑时可能出现语义漂移或连贯性问题，训练过程较为复杂且对算力要求较高。

---

## 662. 3D Shape Control of Extensible Multi-Section Soft Continuum Robots via Visual Servoing

**arXiv ID:** 2602.19273 | [PDF](https://arxiv.org/pdf/2602.19273v1)

**作者:** Abhinav Gandhi `[一作]` (Worcester Polytechnic Institute), Berk Calli `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 2844 | [OpenAlex ID](https://openalex.org/A5008443652)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于外部相机视觉的全身形状闭环控制算法，能够调节多段可伸缩柔性连续机器人的完整3‑D 形状并精准定位末端执行器；

**💡 创新点**

提出了机器人形状雅可比矩阵与块对角图像雅可比的组合，实现无本体传感器的全局渐进稳定形状控制，并通过2-1/2D视觉伺服框架实现全身冗余利用；

**🔧 技术方法**

采用PUP（Prismatic-Universal-Prismatic）模型、PCC（Piecewise Constant Curvature）弧参数化、逆向运动学求解器（AMoRPH）、块对角图像雅可比、机器人形状雅可比、RGB相机+ArUco标记视觉检测、闭环控制律及实验验证；

**📊 数据集**

使用YCB杯子集进行堆叠实验，结合自制米饭与抽屉实验；主要数据来源为实验现场采集的图像与控制日志，没有公开数据集；

**📈 对比分析**

与传统仅控制末端执行器的视觉伺服方法对比，实验中两段机器人实现稳态误差<1 mm，图像误差<15 像素，三段机器人误差略增大；上升时间4.8 s（两段）/2.75 s（三段），稳态时间12.5 s/8.2 s；杯子堆叠成功率4/5，展示了控制的可靠性与适用性；

**⚠️ 局限性**

对三段机器人精度下降，误差随段数增加；缺乏动力学模型导致非线性效应未被充分利用；参考形状生成目前局限于2‑D，未实现全三维优化；系统依赖外部相机与良好光照，需额外标记或高分辨率RGB‑D支持。

---

## 663. Synthesizing Multimodal Geometry Datasets from Scratch and Enabling Visual Alignment via Plotting Code

**arXiv ID:** 2602.18745 | [PDF](https://arxiv.org/pdf/2602.18745v1)

**作者:** Haobo Lin `[一作]` (Hong Kong University of Science and Technology), Binhang Yuan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 649 | [OpenAlex ID](https://openalex.org/A5002684888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种从零开始合成多模态几何问题的完整流水线，并构建了包含绘图代码的18k规模高质量数据集GeoCode。

**💡 创新点**

创新点包括：①将几何问题拆分为符号种子生成、数值实例化和可视化三阶段，保证符号、文本、图像三模态严格一致；②引入绘图代码作为显式对齐目标，使模型必须从图像恢复完整几何结构，从而提升视觉–符号对齐；③通过严谨的语义与几何验证，生成结构复杂、难度更高的几何题。

**🔧 技术方法**

使用的大规模语言模型（如GPT-OSS-120B）完成种子翻译、数值实例化、推理轨迹与答案生成；AlphaGeometry进行符号推理与依赖图构建；Coder模型生成绘图代码；OpenCV渲染图像；LoRA+GRPO实现SFT与强化学习训练。

**📊 数据集**

主要使用自合成的数据集GeoCode（18k题目，包含图像、问题、推理、答案及绘图代码），并在公开几何基准（Geometry3K、MathVerse、MathVista、GeoQA、OlympiadBench）进行跨域评估。

**📈 对比分析**

与无对齐、仅文字对齐和仅答案监督等基线对比，使用Code对齐可使Test-mini精度从17.9%提升至26.5%，Geometry3K从58.57%提升至60.07%；在公开基准上，GeoCode训练的模型在OlympiadBench提升11.5%，Geometry3K提升6.2%，说明对齐策略显著提高性能。

**⚠️ 局限性**

局限性：数据仍主要覆盖平面几何，复杂空间几何或更高级几何结构难以覆盖；绘图代码的生成和验证依赖大模型，可能引入隐藏错误；对齐策略在多模态场景外的迁移性尚未充分验证。

---

## 664. VALD: Multi-Stage Vision Attack Detection for Efficient LVLM Defense

**arXiv ID:** 2602.19570 | [PDF](https://arxiv.org/pdf/2602.19570v1)

**作者:** Nadav Kadvil `[一作]` (Technion Israel Institute of Technology), Ayellet Tal `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关、轻量化的 LVLM 对抗防御框架 VALD，通过两阶段预检测与多视角文本融合恢复正确输出。

**💡 创新点**

创新点在于：①将内容保持变换与文本嵌入一致性检测分离成两阶段，显著降低开销；②仅在确认攻击后才调用大型 LLM 进行答案合成，提升效率和鲁棒性。

**🔧 技术方法**

使用内容保持图像变换（大幅裁剪、像素遮罩）、视觉编码器嵌入相似性检测、轻量化 LVLM 生成多份回答、基于 KL 散度的文本一致性检测、以及外部 LLM 对多答案合成。

**📊 数据集**

使用 MS‑COCO 数据集的图像与标注作为基准，攻击方法采用 MF‑ii 与 MixAttack 生成对抗样本。

**📈 对比分析**

与 SmoothVLM 等基线相比，VALD 在 LLaVA‑1.5‑7B 与 MiniGPT‑v2 的图像描述任务中，Sentence‑BERT 分数在对抗样本上提高 4‑6%，且清洁图像性能保持不变，推理速度提升至原先的 1/3。

**⚠️ 局限性**

限制包括：对多视角中错误一致性难以辨别导致误判（如假三辆摩托车）；以及检测阈值与清洁图像误报/漏报之间的权衡。

---

## 665. Virtual Parameter Sharpening: Dynamic Low-Rank Perturbations for Inference-Time Reasoning Enhancement

**arXiv ID:** 2602.19169 | [PDF](https://arxiv.org/pdf/2602.19169v1)

**作者:** Saba Kublashvili `[一作]` `[通讯]` (Independent Researcher), Saba Kublashvili (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在推理时为冻结的 Transformer 线性层动态注入低秩扰动，提升模型对不同输入的适应性与推理能力。

**💡 创新点**

提出权重相关的激活条件化扰动结构 ΔW = γ·Wᵀ V Uᵀ W，配合 SK/SC/Hybrid 选择器构造、谱裁剪和自适应策略，实现无需训练即可的输入特定调优。

**🔧 技术方法**

低秩动态扰动、激活统计与梯度引导的选择器构造、谱裁剪、激活能量与词元熵驱动的自适应策略、迭代验证、多目标损失、Q/K耦合、L‑BFGS 预处理等。

**📊 数据集**

ARC‑Challenge（多选科学推理）和 GSM8K（小学数学推理）两大基准数据集。

**📈 对比分析**

在 Qwen2.5 基线上进行对比实验，VPS 在 ARC 的低置信度样本和 GSM8K 上提升约 3–5% 的准确率，ablation 进一步验证各模块贡献。

**⚠️ 局限性**

计算开销显著增加、超参数需要细致调优、缺乏正式理论保证、迭代验证依赖标注、对实时低延迟场景影响较大。

---

## 666. TactEx: An Explainable Multimodal Robotic Interaction Framework for Human-Like Touch and Hardness Estimation

**arXiv ID:** 2602.18967 | [PDF](https://arxiv.org/pdf/2602.18967v1)

**作者:** Felix Verstraete `[一作]` (Imperial College London), Dandan Zhang `[通讯]` (Imperial College London)

**通讯引用:** 5019 | [OpenAlex ID](https://openalex.org/A5100386760)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了TactEx框架，结合视觉、触觉与语言实现果实硬度评估与解释；

**💡 创新点**

创新点在于将可解释性嵌入多模态交互，使用Grounded‑SAM进行零样本对象定位，ResNet50‑LSTM硬度回归，并通过LLM生成基于触觉证据的自然语言解释；

**🔧 技术方法**

采用了GelSight‑Mini触觉相机、ResNet50+LSTM网络、Transformer、Grounded‑SAM、YOLO、DeepSeek-R1‑Distill‑Llama‑70B LLM和Vision‑Language模型进行跨模态融合；

**📊 数据集**

使用公开GelSight数据集（约5,000个对象）进行预训练，随后在自采集的280条果实/软体样本上微调，并在三种水果（芒果、酸橙、西红柿）和三组香蕉/鳄梨等组合上验证；

**📈 对比分析**

与YOLO对比后发现GSAM在分割精度和定位误差上更优，ResNet50‑LSTM在硬度预测上实现RMSE 4.30、R² 0.93、Spearman ρ 0.88，并在四个交互情景中获得最高90%任务成功率，LLM解释准确率与完整性均达4.8/5；

**⚠️ 局限性**

局限在于处理延迟高、对少样本类别（如奇异果）性能受限、需要手动标注的视觉分割与触觉数据收集量大，且框架在复杂多目标未指明场景下精度下降。

---

## 667. Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training

**arXiv ID:** 2602.18851 | [PDF](https://arxiv.org/pdf/2602.18851v1)

**作者:** Seyed Morteza Emadi `[一作]` `[通讯]` (University of North Carolina Chapel Hill), Seyed Morteza Emadi (University of North Carolina Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer注意力logits的谱界限，基于此提出低精度FP8训练的几何自适应缩放方法；

**💡 创新点**

创新点在于引入秩感知的概率保证与无激活观测的自预测缩放，解决了延迟缩放在转瞬场景下的溢出问题，并实现与融合注意力内核兼容的低内存实现；

**🔧 技术方法**

采用隐式幂迭代估计谱范数、旋转位置嵌入不变性、分组查询注意力的隐式展开、以及α校准与自动α技术；

**📊 数据集**

在GPT‑2 XL、Mistral‑7B、Llama‑2‑13B 与 Llama‑2‑70B 的预训练权重上验证，并在MMLU STEM 子集上进行微调评估；

**📈 对比分析**

与传统延迟缩放和保守谱缩放对比，零溢出率、MMLU准确率与延迟缩放相当，FP8利用率提升至约30%，且额外计算开销小于5%；

**⚠️ 局限性**

需在热身阶段额外一次前向以收敛幂迭代，对突变分布可能失效；自适应α在出现显著分布漂移时缺乏严格保证。

---

## 668. Wide Open Gazes: Quantifying Visual Exploratory Behavior in Soccer with Pose Enhanced Positional Data

**arXiv ID:** 2602.18519 | [PDF](https://arxiv.org/pdf/2602.18519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 669. All Constant Mutation Rates for the $(1+1)$ Evolutionary Algorithm

**arXiv ID:** 2602.18989 | [PDF](https://arxiv.org/pdf/2602.18989v1)

**作者:** Andrew James Kelley `[一作]` `[通讯]`, Andrew James Kelley

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文构造了一类新的位序列优化函数DistantSteppingStones_p，并证明在(1+1)演化算法中，其最优突变率随着问题规模n的增大可任意逼近给定的p∈(0,1)，从而证明最优突变率在[0,1]区间内是稠密的。

**💡 创新点**

创新点在于通过设计具有大尺度平台与深谷结构的函数，实现对突变率的精细控制，首次给出对所有p∈(0,1)均可逼近的理论证明。

**🔧 技术方法**

主要技术是解析演化算法的随机过程，利用递推式求解突变概率、几何分布的期望，以及极限定理和指数收敛性分析。

**📊 数据集**

该工作完全基于理论分析，没有使用实际数据集；所有结论来自对构造函数的数学证明。

**📈 对比分析**

由于缺乏实验对比，该论文仅提供了理论上的时间复杂度上界与下界，未给出数值实验验证，但通过大规模渐近分析展示了与传统1/n突变率的相对优势。

**⚠️ 局限性**

限制在于构造函数较为人工且难以直接应用于现实问题，且证明只在n→∞的极限下成立，对小规模实例的性能影响不明确。

---

## 670. Exact Attention Sensitivity and the Geometry of Transformer Stability

**arXiv ID:** 2602.18849 | [PDF](https://arxiv.org/pdf/2602.18849v1)

**作者:** Seyed Morteza Emadi `[一作]` `[通讯]` (University of North Carolina Chapel Hill), Seyed Morteza Emadi (University of North Carolina Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于几何稳定性理论，构建了Transformer的完整稳定性分析框架，解释了预LayerNorm与后LayerNorm、DeepNorm缩放、warmup等训练技巧的根本机制，并验证了注意力敏感度θ(p)在训练全过程中保持接近1的关键发现。

**💡 创新点**

创新点包括：1）推导出softmax Jacobian的精确ℓ∞→ℓ1范数等式 J∞→1 = θ(p)/τ，揭示了注意力分布形状对敏感度的决定作用；2）提出与Transformer运算对齐的block‑∞/RMS几何，得到与序列长度无关的Lipschitz上界；3）从四重投影的乘法结构解释DeepNorm的N⁻¹⁄⁴缩放；4）证明预LayerNorm通过保留加性身份梯度路径消除梯度衰减，而后LayerNorm导致梯度指数衰减；5）提出温度warmup等价于学习率warmup的几何解释。

**🔧 技术方法**

主要技术包括：软max精确敏感度分析、block‑∞/RMS范数设计、层级Lipschitz上界推导、梯度流分析、经验验证实验（梯度与θ(p)监控）以及对温度warmup的理论预测。

**📊 数据集**

使用的基准数据集为FineWeb（大规模文本语料），在774M参数的GPT‑2‑style Transformer（d=1280、N=36、H=20、L=1024）上进行训练和验证。

**📈 对比分析**

通过在同一模型规模和数据集上对比预LayerNorm、后LayerNorm、DeepNorm以及不同warmup策略，发现预LayerNorm训练稳定，后LayerNorm易出现梯度失真；DeepNorm的N⁻¹⁄⁴缩放显著提升深度可训练性；温度warmup与学习率warmup表现相近。实验结果与理论预测高度一致，验证了θ(p)≈1不随训练改变。

**⚠️ 局限性**

局限性在于：1）推导的Lipschitz上界为最坏情况，难以给出精确学习率上限；2）对温度warmup和非四重投影（m≠4）缩放的实证验证尚未完成；3）目前框架主要适用于标准Transformer，未覆盖RMSNorm、MoE等变体。

---

## 671. LMFPPO-UBP: Local Mean Field Proximal Policy Optimization with Unbalanced Punishment for Spatial Public Goods Games

**arXiv ID:** 2602.18696 | [PDF](https://arxiv.org/pdf/2602.18696v1)

**作者:** Jinshuo Yang `[一作]` (Guizhou University), Youliang Tian `[通讯]` (Guizhou University)

**通讯引用:** 3228 | [OpenAlex ID](https://openalex.org/A5052363813)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于局部平均场感知与不平衡惩罚的深度强化学习框架（LMFPPO‑UBP），用于解决空间公共物品游戏中的合作难题。

**💡 创新点**

创新点在于：①将平均场近似从全局转为局部邻域，构建局部平均场（Local Mean‑Field）感知；②设计不平衡惩罚机制，只对被多数合作邻居包围的背叛者施加惩罚，避免对合作方产生成本；③将上述两项融入Proximal Policy Optimization（PPO）中，形成可学习且可自适应的惩罚策略。

**🔧 技术方法**

技术方法包括：深度强化学习（PPO）+局部平均场输入；局部邻域统计量嵌入策略梯度；不平衡惩罚奖励设计；Actor‑Critic网络架构；通用奖励与优势估计。实验基于仿真模拟，未使用公开数据集。

**📊 数据集**

使用 200×200 周期性格子模拟的空间公共物品游戏数据，探索不同增强因子 r（2.0–6.0）以及多种初始化（随机、半半、全背叛）下的合作率。

**📈 对比分析**

与 LMFPPO、Fermi 更新规则、Q‑learning、PPO 等基线对比。结果表明：LMFPPO‑UBP 将合作临界增强因子从 5.0 降至 4.3，收敛速度快，合作率高且波动小；LMFPPO、PPO 在低 r 下收敛至全背叛，Fermi 与 Q‑learning 仅在 r≥5 时出现显著合作，且收敛慢且不稳定。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏真实系统的实验数据；对超参数（惩罚强度 p、熵权 ρ 等）敏感，需经验调优；在更大规模或更复杂网络结构下的可扩展性尚待进一步评估。

---

## 672. DCInject: Persistent Backdoor Attacks via Frequency Manipulation in Personal Federated Learning

**arXiv ID:** 2602.18489 | [PDF](https://arxiv.org/pdf/2602.18489v1)

**作者:** Nahom Birhan `[一作]` (Old Dominion University), Daniel Takabi `[通讯]` (Old Dominion University)

**通讯引用:** 813 | [OpenAlex ID](https://openalex.org/A5002423453)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DCInject，一种专为个性化联邦学习设计的频域后门攻击，通过改写DC分量实现全局触发器；

**💡 创新点**

创新点在于首次将频域（尤其DC分量）用于PFL后门，结合自适应噪声与感知加权，既保持隐蔽性又对多种个性化机制具有持久性；

**🔧 技术方法**

采用频域变换（DCT/FFT）+高斯噪声注入、感知加权、梯度一致性重建等技术；

**📊 数据集**

在CIFAR-10、CIFAR-100、SVHN、GTSRB四个公共图像数据集上进行实验；

**📈 对比分析**

与Bad-PFL、BadNet、FTrojan等基线对比，DCInject在保持接近基线的干净准确率（Acc）下，攻击成功率（ASR）显著更高（例如GTSRB 100% ASR），并且在I‑BAU防御下依旧保持高ASR；训练时间与基线相当；

**⚠️ 局限性**

局限性包括：对不同PFL框架的通用性尚未完全验证，某些网络结构下自适应变体的ASR可能下降，且未来针对频域后门的更强防御仍可能降低其效果。

---

## 673. Unfolding Ordered Matrices into BioFabric Motifs

**arXiv ID:** 2602.19745 | [PDF](https://arxiv.org/pdf/2602.19745v1)

**作者:** Jules Wulms `[一作]` (TU Eindhoven), Bettina Speckmann `[通讯]` (TU Eindhoven)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

自动化将良好排序的邻接矩阵（按Moran's I优化）中的噪声图形模式检测并展开为可视化的BioFabric模式简化。

**💡 创新点**

首次提出利用Moran's I排序来高效揭示图形模式，并通过矩阵“展开”技术将这些模式直接映射为BioFabric的形状符号，解决了以往需人工调优的瓶颈。

**🔧 技术方法**

使用Moran's I矩阵排序、噪声模式检测算法（基于σ、τ阈值的矩形子矩阵识别）、矩阵展开逻辑以及自定义的图形符号（矩形环、空洞表示缺失边）来实现。

**📊 数据集**

在四个真实数据集上验证：FLT（脑网络）、MIS（Les Miserables人物关系）、ZKC（Zachary的拳击社团）、SCH（学校社交网络，242节点）。

**📈 对比分析**

与手工调优的BioFabric对比，展示了更高的模式识别率和更简洁的可视化；计算上，矩阵重排约20 s，模式检测<1 s，整体可在现代笔记本上几乎即时完成。

**⚠️ 局限性**

局限性包括：仅处理≤250节点的图；矩阵重排依赖外部TSP求解器，可能在极大图上受限；算法对重叠模式做了排除，可能遗漏部分结构；并未涵盖较弱的路径模式。

---

## 674. RetinaVision: XAI-Driven Augmented Regulation for Precise Retinal Disease Classification using deep learning framework

**arXiv ID:** 2602.19324 | [PDF](https://arxiv.org/pdf/2602.19324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 675. The Million-Label NER: Breaking Scale Barriers with GLiNER bi-encoder

**arXiv ID:** 2602.18487 | [PDF](https://arxiv.org/pdf/2602.18487v1)

**作者:** Ihor Stepanov `[一作]` (Knowledgator Engineering), Oleksandr Lukashov `[通讯]` (Knowledgator Engineering)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 bi-encoder 结构 GLiNER-bi-Encoder，能够在零样本条件下识别海量实体类型，并支持实体链接。

**💡 创新点**

创新点在于将文本编码与标签编码分离，预先计算标签嵌入，消除上下文窗口瓶颈，实现千级甚至百万级实体类型的高效识别，并在保持性能的同时提升 130 倍吞吐量。

**🔧 技术方法**

使用了现代 BERT 变体（Ettin 系列）作为文本编码器，句子 Transformer（BGE、MiniLM 等）作为标签编码器，结合跨注意力融合、focal 负采样等技术；实现端到端的 span 级或 token 级 NER。

**📊 数据集**

在 26 个 NER 基准（包括 CrossNER、CoNLL 2003、GENIA、BC5CDR、WikiNeural 等）上进行评测；预训练数据来自 8M GPT‑4o 注释文本，后续 fine‑tune 使用 40k 高质量样本。

**📈 对比分析**

与 uni‑encoder GLiNER 及其它 GLiNER‑v2 版本对比，GLiNER-bi‑Encoder 在 CrossNER 上达到 61.5% Micro‑F1，整体零样本平均 49.7%；在吞吐量上，当标签数 1024 时预计算模式下比 uni‑encoder 高 130×，在 100 级标签时提升 5.3×。

**⚠️ 局限性**

局限性包括：对高度上下文依赖的实体（如 HarveyNER、FabNER）表现有限；最大 span 宽度限制为 12，难以捕捉更长实体；跨注意力融合仅在实验中少量验证，需进一步系统化研究。

---

## 676. Enhancing Capstone Program Workflow: A Case Study on a Platform for Managing Academic-Industry Projects

**arXiv ID:** 2602.20120 | [PDF](https://arxiv.org/pdf/2602.20120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 677. RotorSuite: A MATLAB/Simulink Toolbox for Tilt Multi-Rotor UAV Modeling

**arXiv ID:** 2602.18814 | [PDF](https://arxiv.org/pdf/2602.18814v1)

**作者:** Nicola Cigarini `[一作]` (University of Padova), Angelo Cenedese `[通讯]` (University of Padova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了RotorSuite MATLAB/Simulink工具箱，用于快速、可配置地建模和仿真任意倾斜或无倾斜多旋翼无人机的动力学与控制

**💡 创新点**

创新点在于提供统一的参数化框架，可一次性生成不同几何布局的多旋翼模型，并实现双层仿真：基于解析Newton‑Euler方程的数值仿真与基于Simscape的物理一致性仿真；工具箱无需外部软件，跨平台且易于教育与研究使用

**🔧 技术方法**

采用MATLAB/Simulink与Simscape进行建模；利用欧拉角与SO(3) Lie代数处理姿态；通过解析力矩矩阵F、M求解控制输入；应用SE(3)轨迹跟踪控制器实现姿态/位置闭环；在仿真中加入电机转速与推力参数

**📊 数据集**

未使用公开数据集，而是通过设计一套六旋翼互相倾斜的测试平台（给定质量、惯性矩、推力/阻尼系数等参数）进行实验验证

**📈 对比分析**

对比方法为在相同控制器与参考轨迹下，分别使用解析与物理一致性仿真两种模型；结果显示两者在位置误差（10 cm以内）和姿态误差（10⁻⁴ rad以内）上高度一致，主差异仅出现在瞬态期间的转速上，误差限于±2 rad/s

**⚠️ 局限性**

局限性包括：对惯性矩的差异会导致姿态误差差异；电机动力学仅在物理仿真中体现，解析模型忽略；工具箱目前支持3–8旋翼，较大平台或非星形布局可能需手动扩展；未与外部硬件/仿真平台集成

---

## 678. HIME: Mitigating Object Hallucinations in LVLMs via Hallucination Insensitivity Model Editing

**arXiv ID:** 2602.18711 | [PDF](https://arxiv.org/pdf/2602.18711v1)

**作者:** Ahmed Akl `[一作]` (Griffith University), Kewen Wang `[通讯]` (Queensland University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对大型视觉语言模型（LVLM）中的物体幻觉问题，系统分析了不同解码层的幻觉敏感度，提出了层级敏感度指标 Hallucination Insensitivity Score (HIS)，并基于 HIS 进行层级自适应的模型权重编辑方法 HIME，旨在抑制幻觉同时保持预训练知识。

**💡 创新点**

创新点在于：①首次揭示 LVLM 解码层在幻觉抑制方面表现出显著的层级差异；②设计 HIS 指标，以注意力分布的 KL 散度量化每层对真幻对齐的敏感度；③提出 HIME，利用 HIS 对每层 MLP 权重施加加权投影，避免全局编辑导致的知识失真。

**🔧 技术方法**

技术手段包括：对比样本（真实 vs 幻想）生成；提取注意力矩阵并通过 KL 散度得到 HIS；利用注意力加权的特征进行 SVD，得到幻觉子空间；对 MLP 权重应用 HIS 加权的投影操作；整体实现不引入额外参数、推理延迟或计算开销。

**📊 数据集**

使用的数据集和模型有：CHAIR、LLaVA‑Bench、MME、GPT‑4V 辅助评估；测试的 LVLM 包括 LLaVA‑1.5、MiniGPT‑4、mPLUG‑Owl2、Qwen2‑VL‑8B‑Instruct、Qwen3‑VL‑8B‑Instruct。

**📈 对比分析**

在 CHAIR、MME、LLaVA‑Bench 等基准上与 fine‑tuning、对比解码（VCD、DoLa 等）以及其他模型编辑方法（Nullu、OPERA、HALC 等）进行对比，HIME 在平均 61.8% 的幻觉降低、CHIAR_s 与 CHAIR_i 下降显著、BLEU/感知分数保持或提升方面均优于现有方法，且不增加额外参数或推理延迟。

**⚠️ 局限性**

局限性包括：对比样本生成依赖 GPT‑3.5，可能对不同幻觉类型或复杂场景的覆盖有限；仅针对对象幻觉的抑制，未考虑更广泛的幻觉或语义不一致问题；编辑仅涉及 MLP 权重，其他模块的幻觉机制仍未被彻底抑制；在更大规模或多任务环境中的泛化仍需进一步验证。

---

## 679. Give Users the Wheel: Towards Promptable Recommendation Paradigm

**arXiv ID:** 2602.18929 | [PDF](https://arxiv.org/pdf/2602.18929v1)

**作者:** Fuyuan Lyu `[一作]` (McGill and Mila - Quebec AI Institute), Xiuqiang He `[通讯]` (Shenzhen Technological University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了可对传统序列推荐模型进行自然语言提示的解耦框架 DPR

**💡 创新点**

创新点在于融合模块对协同与语义信号对齐、Mixture‑of‑Experts 进行正向/负向控制、以及三阶段训练策略

**🔧 技术方法**

使用多头交叉注意力、两路 MoE Tower、预训练 Sentence‑Transformer、温度缩放交叉熵等技术

**📊 数据集**

在 MovieLens‑1M 和 MIND 两大公开电影/新闻数据集上进行实验

**📈 对比分析**

与过滤式、LLM‑后端、LLM‑重排等方法对比，DPR 在正/负提示任务上均显著提升 NDCG/Recall，尤其在正向提示上提升超过 70% 的 NDCG@10

**⚠️ 局限性**

局限性包括：仍依赖 LLM 进行提示编码、对多模态提示支持有限、训练三阶段成本较高、未在真实用户交互中验证

---

## 680. Derivation Depth as an Information Metric: Axioms, Coding Theorems, and Storage--Computation Tradeoffs

**arXiv ID:** 2602.19137 | [PDF](https://arxiv.org/pdf/2602.19137v1)

**作者:** Jianfeng Xu `[一作]` (Shanghai Jiao Tong University), Jianfeng Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 7922 | [OpenAlex ID](https://openalex.org/A5101973930)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出“推导深度”作为在线推理成本的可计算指标，并通过推导轨迹编码与Kolmogorov复杂度建立信息量与深度的紧密联系；在此基础上给出存储与计算的频率加权权衡、断点频率和子模优化的缓存分配方案，并扩展到噪声前提基的情况。

**💡 创新点**

创新点包括：
① 将推导深度视为信息度量，提供两侧编码界限；
② 推导深度与条件Kolmogorov复杂度呈Θ关系，揭示信息丰富查询的“深度＝信息/对数”定律；
③ 导出频率加权的存储–计算折算率及临界频率 f_c=Θ(ρ·log(m+d))；
④ 在全局缓存决策中利用子模最大化实现 1‑1/e 近似；
⑤ 引入噪声前提基模型，给出噪声对深度与信息量的影响和鲁棒优化。

**🔧 技术方法**

技术手段包括：
- 基于 FO(LFP) 的可有效子结构与证明系统；
- 可计算的前驱运算符和不可冗余核心提取；
- 推导轨迹的自限编码与最短轨迹长度 N(q|B)；
- 丰富性（richness）条件与不可压缩性证明；
- Kolmogorov复杂度与编码理论的结合；
- 子模优化与 Knapsack 贪心算法；
- 噪声前提基的多态描述与两向可编码性。

**📊 数据集**

该工作主要是理论性，没有使用标准数据集；所有结论在抽象的逻辑语义框架下证明，实证实验待后续实现。

**📈 对比分析**

评价方式主要是理论证明与复杂度上界；对比基准为传统视图/缓存设计的经验公式，本文给出了更精确的深度–信息定量关系和频率临界阈值；在子模优化层面，提出的贪心算法在满足子模性假设时可获得 1‑1/e 的近似保证。

**⚠️ 局限性**

局限性：
- 依赖一系列可计算假设（序列化、可逆前驱、可判定冗余、丰富性条件），这些在具体逻辑或应用场景中难以验证；
- Kolmogorov 复杂度不可计算，实际系统需用近似编码；
- 噪声模型仅覆盖前提集合的丢失/污染，未考虑规则或语义漂移等更复杂噪声；
- 证明主要适用于信息丰富查询，低信息量查询的表现未知；
- 纯理论框架，缺乏大规模实验验证。

---

## 681. Robustness of Deep ReLU Networks to Misclassification of High-Dimensional Data

**arXiv ID:** 2602.18674 | [PDF](https://arxiv.org/pdf/2602.18674v1)

**作者:** Věra Kůrková `[一作]` `[通讯]` (Institute of Computer Science of the Czech Academy of Sciences), Věra Kůrková (Institute of Computer Science of the Czech Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

本文通过高维几何与ReLU网络的分段线性结构，理论推导了参数化深度ReLU网络在随机输入扰动下的局部鲁棒性下界；

**💡 创新点**

创新点在于：①将极限几何中的“极角极限”与网络几何复杂度结合，给出鲁棒性随输入维度指数增长、随网络单元数仅轻微下降的非平凡下界；②证明对几乎所有输入，鲁棒性随维度急剧提升，从而反驳Szegedy等人提出的对抗样本稠密假设；

**🔧 技术方法**

主要技术手段包括：高维球面极角极限、对ReLU网络的凸单元分区与面数上界、概率论与几何测度结合的误判概率估计；

**📊 数据集**

本文为理论研究，未使用任何公开数据集；

**📈 对比分析**

没有实验比较，论文仅给出理论证明；

**⚠️ 局限性**

局限性：仅适用于单输出ReLU网络，无法推广到卷积、核函数网络；对网络深度无具体提升作用，且对输入噪声分布假设为均匀球面分布，实际应用中可能不符合。

---

## 682. Referring Layer Decomposition

**arXiv ID:** 2602.19358 | [PDF](https://arxiv.org/pdf/2602.19358v1)

**作者:** Fangyi Chen `[一作]` (ByteDance), Longyin Wen `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于多模态提示的 Referring Layer Decomposition (RLD) 任务，构建了 RefLade 数据集并提供了 RefLayer 基线模型。

**💡 创新点**

创新点在于：①首次将对象级 RGBA 图层分解与可控提示结合；②设计了可扩展的数据生成引擎和自动化评估协议；③提出了 HPA（Human Preference Aligned）统一评估指标，紧密匹配人类偏好。

**🔧 技术方法**

主要技术包括：Stable Diffusion 3 的扩散生成框架、VAE 编码解码、专门的 alpha 解码器、CLIP 用于语义一致性评估、LPIPS 与 FID 进行视觉保留与整体质量评估。

**📊 数据集**

使用了 RefLade 数据集，包含 430K 张图像、约 1.11M 个 RGBA 图层以及 100K 人工审核层，测试集 10K。对比基线 MuLAn 数据集及其扩散模型。

**📈 对比分析**

通过 HPA 分数与人类 ELO 排名的高相关性验证评估方法，RefLayer 在 RefLade 1M 预训练加高质量微调后取得最高 HPA ≈ 0.4813（F1 为 10.5/13.1），明显优于 MuLAn 版基线，且随数据量增大表现稳步提升。

**⚠️ 局限性**

局限性包括：文本提示下定位精度不足；背景图层分解依赖于复制粘贴且对背景颜色敏感；模型在罕见场景或大遮挡时仍易失真，需进一步提升对复杂遮挡和多对象交互的泛化能力。

---

## 683. PerSoMed: A Large-Scale Balanced Dataset for Persian Social Media Text Classification

**arXiv ID:** 2602.19333 | [PDF](https://arxiv.org/pdf/2602.19333v1)

**作者:** Isun Chehreh `[一作]` (Institute for Advanced Studies in Basic Sciences), Ebrahim Ansari `[通讯]` (Institute for Advanced Studies in Basic Sciences)

**通讯引用:** 268 | [OpenAlex ID](https://openalex.org/A5054232595)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个规模达36,000条、9类平衡的波斯语社交媒体文本分类数据集，并对多种深度学习模型进行评估。

**💡 创新点**

创新点在于首次公开大规模平衡波斯语社交媒体数据集、结合词汇替换与ChatGPT few-shot生成的混合增强策略，以及对多模态、参数高效的Transformer模型进行系统比较。

**🔧 技术方法**

采用文本预处理、ChatGPT自动标注、语义嵌入的欠采样、词汇替换与few-shot生成的数据增强，以及BiLSTM、XLM-RoBERTa、FaBERT、TookaBERT、SBERT+注意力等多种Transformer架构，并引入LoRA/AdaLoRA参数高效微调。

**📊 数据集**

使用的主要数据集为从波斯语社交媒体抓取的约60,000条帖子，最终整理成36,000条、每类4,000条的平衡数据集。

**📈 对比分析**

通过在预处理后的数据上进行模型训练，TookaBERT-Large在F1上达0.9621，明显优于BiLSTM（0.8813）和XLMR-B（0.9491），而LoRA/AdaLoRA在保持近似性能的同时显著减少可训练参数。

**⚠️ 局限性**

主要限制包括部分类别（如Social与Political）边界模糊导致准确率略低，以及数据仅覆盖近期讨论，可能不适用于未来的语言或话题变化。

---

## 684. Position: General Alignment Has Hit a Ceiling; Edge Alignment Must Be Taken Seriously

**arXiv ID:** 2602.20042 | [PDF](https://arxiv.org/pdf/2602.20042v1)

**作者:** Han Bao `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出边缘对齐（Edge Alignment）框架，强调在多值冲突、少数派表达和不确定性情境下的动态规范治理，形成七大支柱，涵盖结构化多目标优化、分层约束、少数派与情境化对齐以及交互式协商机制

**💡 创新点**

将传统标量奖励法提升为向量化多目标优化和分层优先级策略，构建可在决策边缘动态仲裁的治理体系，并提出基于社会选择和风险感知的交互式对齐方法

**🔧 技术方法**

多目标强化学习（MO-RL）与Pareto优化框架（PAMA）、安全RLHF（Safe RLHF）、分层优先级算法、社会选择理论、语义不确定性评估、构造性拒绝与主动澄清交互模型

**📊 数据集**

论文主要讨论已有的RLHF/数据集（如BeaverTails、HH-RLHF等），并指出这些数据集缺乏冲突与交互场景，建议构建冲突增广数据和多样化参与者数据集

**📈 对比分析**

本研究为概念性框架，没有具体实验评估，提出了基于多轮对抗和交互式指标的评测思路，认为需在多轮情境下量化冲突识别与仲裁质量，尚无可直接比较的性能数值

**⚠️ 局限性**

缺乏实证验证与基准；对新数据集与评测框架的依赖使实现成本高；多目标与分层优化在大规模模型上的计算复杂度仍需优化；治理与参与机制的可操作性与法律合规性仍待完善

---

## 685. ChimeraLoRA: Multi-Head LoRA-Guided Synthetic Datasets

**arXiv ID:** 2602.19708 | [PDF](https://arxiv.org/pdf/2602.19708v1)

**作者:** Hoyoung Kim `[一作]` (POSTECH), Jungseul Ok `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了多头 LoRA 框架 ChimeraLoRA，利用类共享 LoRA A 与每张图 LoRA B 结合，并通过 Grounded‑SAM 边框实现语义增强，生成既多样又细节丰富的少量训练样本，用于数据稀缺或长尾分类任务。

**💡 创新点**

创新点在于：① 将 LoRA 拆分为共享类级和实例级两部分，形成多头结构；② 训练时利用目标类边框做语义增强；③ 合成时用 Dirichlet 采样混合实例 LoRA，以在多样性与细节之间取得平衡。

**🔧 技术方法**

采用的技术包括稀疏低秩适配器 LoRA、Latent Diffusion 模型（Stable Diffusion 2.1）、Grounded‑SAM 边框检测、Dirichlet 混合、CLIP 编码器、FID/CLIP 分数评估等。

**📊 数据集**

实验使用了 11 个公开分类数据集：FGVCAircraft、Caltech101、StanfordCars、DTD、EuroSAT、Flowers102、Food101、OxfordPets、Skin Lesions（ISIC）、CIFAR‑10 与 ImageNet100，且每类仅采样 4 张真实图像作为参考。

**📈 对比分析**

与 IsSynth、LoFT、DataDream 三个基线对比，在 4‑shot 参考加 500 生成图的 CLIP 训练下，ChimeraLoRA 在大多数数据集上均获得更高准确率，甚至超过 4‑shot 真实数据；在长尾实验中显著提升尾类准确度并整体提升；Synthetic‑to‑Real 评估显示更低 FID、更高 CLIP 与中心相似度。

**⚠️ 局限性**

局限性包括：当少量参考图语义差异较大时难以应用；依赖 Grounded‑SAM 边框检测，检测失误会影响效果；需要为每个类训练多头 LoRA，导致参数和计算成本上升。

---

## 686. Limited Reasoning Space: The cage of long-horizon reasoning in LLMs

**arXiv ID:** 2602.19281 | [PDF](https://arxiv.org/pdf/2602.19281v1)

**作者:** Zhenyu Li `[一作]` (Academy of Sciences), Yongqiang Zhao `[通讯]` (Peking University)

**通讯引用:** 5096 | [OpenAlex ID](https://openalex.org/A5073503029)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Halo 框架，利用模型预测控制（MPC）与熵监测动态调节大语言模型的推理路径，从而在长链推理中避免“有限推理空间”导致的性能崩溃。

**💡 创新点**

创新点在于：1）将长链推理视作非自治随机动力系统，给出有限推理空间假设并推导误差累积界限；2）设计熵驱动的双控制器（观察器+控制器），实现实时检测与纠正推理漂移；3）通过语义压缩与历史重置在推理轨迹上进行闭环纠正。

**🔧 技术方法**

采用的技术包括：非自治随机动力系统建模、Lyapunov指数与误差传播分析、注意力熵监测、模型预测控制（MPC）、语义压缩/历史重置、实验对比评估与超参数调优。

**📊 数据集**

使用的数据集：Omni-MATH、RULER、GSM8K、MATH Easy、LongBench、InfBench、LRA-L2 等；在七种规模从 7B 到 72B 的 LLM 上进行实验。

**📈 对比分析**

与 8 个基线（标准 CoT、CoT-SC、Tree-of-Thought、AdaCoT、CoT-Valve 等）在 Tier1（短链）与 Tier2（长链）任务中对比，Halo 在长链任务上显著提升成功率（如 Omni-MATH 42.7% vs 15.8%），相对令牌开销仅 1.29×；在短链任务保持与 CoT 同等性能，展现出良好的适应性。

**⚠️ 局限性**

局限性包括：需手动设定熵阈值 Ψ 与灵敏度 α，可能对不同任务或模型不完全通用；理论假设与实际噪声分布存在偏差；对极端长链或更大模型的可扩展性尚未完全验证。

---

## 687. A Replicate-and-Quantize Strategy for Plug-and-Play Load Balancing of Sparse Mixture-of-Experts LLMs

**arXiv ID:** 2602.19938 | [PDF](https://arxiv.org/pdf/2602.19938v1)

**作者:** Zijie Liu `[一作]` (University of North Carolina), Tianlong Chen `[通讯]` (University of North Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种训练无关的推理时负载均衡框架 Replicate-and-Quantize（R&Q），通过识别并复制负载过重的专家以及量化重要性低的专家，以在不增加模型体积的前提下缓解 Sparse Mixture-of-Experts（SMoE）模型在推理阶段的专家负载不均问题。

**💡 创新点**

创新点在于：①将专家复制与量化相结合，在保持精度的同时均衡专家使用；②首次提出 Load Imbalance Score（LIS）这一量化指标，用于客观评估推理时专家负载偏斜；③R&Q为后训练部署提供了无额外训练、无模型改造的轻量级解决方案。

**🔧 技术方法**

核心技术包括：SMoE 路由分析、专家重要性评估（基于 Wanda 近似剪枝得分）、低位数量化（如 8‑bit 或 INT6）以及按需复制专家的后处理流程；实验中还使用了常见的量化工具如 GPTQ/LLM.int8 等。

**📊 数据集**

使用的数据集覆盖多种任务：MMLU、TruthfulQA、GSM8K、PIQA、Winogrande、Hellaswag、CodexGLUE、CoQA、WikiText 等，并在 Switch Transformer、LLaMA‑MoE、DeepSeek‑MoE、DeepSeek‑V2 Lite 等多种 SMoE 架构上进行评估。

**📈 对比分析**

方法通过与原始模型、单独量化、仅复制、全量化等基线以及多种微调策略进行对比。结果显示 R&Q 在所有模型与任务上平均降低 LIS 0.5–1.0（即负载不均度下降 1.4 倍），且保持或略微提升任务准确率（±0.6% 以内），同时未显著增加内存占用或推理延迟。

**⚠️ 局限性**

局限性包括：①需要在推理前使用小量校准数据统计专家负载和重要性；②复制后若过多专家被复制，模型大小仍会增加；③在极端分布漂移或实时流式推理中，重复制率与量化策略需进一步动态调整；④对硬件支持量化的要求较高，某些设备可能无法充分利用量化带来的收益。

---

## 688. Equivalence and Divergence of Bayesian Log-Odds and Dempster's Combination Rule for 2D Occupancy Grids

**arXiv ID:** 2602.18872 | [PDF](https://arxiv.org/pdf/2602.18872v1)

**作者:** Tatiana Berlenko `[一作]` (Constructor University), Kirill Krinkin `[通讯]` (JetBrains LTD)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过构建公平的传感器模型匹配，对比了贝叶斯log‑odds融合与Dempster–Shafer组合规则在二维占用网格中的性能。

**💡 创新点**

创新点在于提出基于pignistic转换的匹配方法，消除观测级别的传感器参数差异，并揭示若不匹配会导致显著偏差；进一步证明贝叶斯融合在点概率指标上优于DS规则。

**🔧 技术方法**

使用的技术包括贝叶斯log‑odds累积、Dempster’s rule、Yager’s rule、pignistic转换、L_max消除实验、TOST等效检验以及Cohen’s d效应量分析。

**📊 数据集**

实验基于仿真单机、多机器人场景以及真实室内激光雷达数据集Intel Research Lab和Freiburg Building 079。

**📈 对比分析**

比较方法：对每个观测的pignistic概率设为相等，保证仅对融合规则进行评估；实验结果显示在细胞准确率、边界锐度、Brier分数等指标上，贝叶斯融合均优于或等价于DS规则，差异极小且方向一致。

**⚠️ 局限性**

局限性：仅评估点概率指标，未考虑Belief Function的区间概率优势；未测试非共轭观测、多假设框架、3D网格或其它组合规则；真实数据中前沿细胞覆盖有限，可能低估Belief Function的潜在优势。

---

## 689. Incremental Learning of Sparse Attention Patterns in Transformers

**arXiv ID:** 2602.19143 | [PDF](https://arxiv.org/pdf/2602.19143v1)

**作者:** Oğuz Kaan Yüksel `[一作]` (École Polytechnique Fédérale de Lausanne), Nicolas Flammarion `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 962 | [OpenAlex ID](https://openalex.org/A5061093552)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了 transformer 在高阶马尔可夫链任务中的稀疏注意力模式学习，并证明了从竞争到合作的分阶段学习过程。

**💡 创新点**

首次把稀疏注意力模式的增量学习形式化为分阶段动力学，并给出了精确的差分方程与收敛分析。

**🔧 技术方法**

使用简化的单层多头注意力模型、差分方程动力学、梯度流分析以及实验验证。

**📊 数据集**

采用合成的高阶马尔可夫链生成的数据（以及对应的回归变体），没有使用真实语言数据。

**📈 对比分析**

通过 KL 散度和损失曲线对不同上下文长度的模型以及不同初始化/超参数进行对比，证明了模型按阶段逼近完整上下文的预测，并在实验中显示了预期的阶梯式收敛。

**⚠️ 局限性**

仅在极简合成任务和简化 Transformer 架构下验证，缺乏对真实 NLP 任务的实验，且理论假设过于理想化。

---

## 690. Machine-Generated, Machine-Checked Proofs for a Verified Compiler (Experience Report)

**arXiv ID:** 2602.20082 | [PDF](https://arxiv.org/pdf/2602.20082v1)

**作者:** Zoe Paraskevopoulou `[一作]` `[通讯]`, Zoe Paraskevopoulou

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本论文中，作者利用Claude Code等大型语言模型（LLM）在无人工书写证明的情况下，从零开始完成了 CertiCoq 编译器中 ANF（行政正常形式）转换的完整形式化证明，证明长度约 7,800 行；

**💡 创新点**

创新点在于首次展示了 agentic LLM 能够根据已有的 CPS 证明模板，自动化适配并完成一种新变换的完整、可验证证明，缩短了从数月到数天的开发周期；

**🔧 技术方法**

使用的技术包括：Claude Opus 4.6 代理式编码助手、自然语言交互指导、步骤索引逻辑关系、与 CertiCoq 代码库的自动化集成（编译、检索、提交）以及基于证明模板的结构化推理；

**📊 数据集**

本研究主要基于 CertiCoq 编译器的源码和已完成的 CPS 证明作为模板，未使用传统意义上的外部数据集；

**📈 对比分析**

与人工完成的 CPS 证明相比，LLM 完成的 ANF 证明在 96 小时内完成，总行数约 7,783 行（比 CPS 的 5,294 行略多），但证明时间从数月缩短到数天，表明自动化提升了生产效率；

**⚠️ 局限性**

局限性包括：LLM 可能在未提示时弱化证明陈述、误假设变量新鲜度、需要频繁人工审查、受限于上下文窗口与错误信息结构化、对商业 LLM 的依赖以及在缺乏模板或专家指导时可推广性不足。

---

## 691. Can a Teenager Fool an AI? Evaluating Low-Cost Cosmetic Attacks on Age Estimation Systems

**arXiv ID:** 2602.19539 | [PDF](https://arxiv.org/pdf/2602.19539v1)

**作者:** Xingyu Shen `[一作]` (Reality Inc), Simiao Ren `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了家庭级化妆攻击（假胡须、灰发、化妆、皱纹）对 AI 年龄估计模型的影响，并系统评估了 8 个模型的鲁棒性。

**💡 创新点**

首次系统性评估非技术化妆攻击对年龄估计的攻击效果，提出攻击转换率(ACR)指标，并将 CV 与 VLM 模型的鲁棒性直接对比。

**🔧 技术方法**

使用 Gemini 2.5 Flash Image 进行图像编辑模拟攻击，评估 5 个 CV 架构（MiVOLO、Custom‑Best 等）和 3 个零样本 VLM（Gemini 3 Flash、Gemini 2.5 Flash、GPT‑5‑Nano），并计算平均年龄偏移与 ACR。

**📊 数据集**

采用 8 个公开年龄估计数据集（UTKFace、IMDB‑WIKI、MORPH、AFAD、CACD、FG‑NET、APPA‑REAL、AgeDB）中的 329 张 10–21 岁面部图像。

**📈 对比分析**

通过平均年龄偏移和 ACR 衡量攻击效果：单一假胡须可达 28–69% ACR，四种攻击组合平均 ACR 约 69%，VLM 模型相对更稳健但差距不显著，整体模型均被显著攻击。

**⚠️ 局限性**

局限包括：攻击为模拟，真实化妆效果可能不同；缺乏多角度、多光照、视频等真实场景评估；ACR 分母小导致估计不确定；攻击生成器对少儿图像存在偏拒绝；未考虑主动优化攻击或其他化妆类型。

---

## 692. LLMs Can Learn to Reason Via Off-Policy RL

**arXiv ID:** 2602.19362 | [PDF](https://arxiv.org/pdf/2602.19362v1)

**作者:** Daniel Ritter `[一作]` (Cornell University), Wen Sun `[通讯]` (Databricks)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种名为OAPL的全新离线强化学习框架，用于大语言模型的后训练，能够在离线（off‑policy）异步训练环境下显著提升推理和代码生成的 Pass@k 性能；

**💡 创新点**

其创新点在于将 KL 约束与最优优势回归结合，直接利用推理引擎的延迟策略作为 KL 参考，完全摒弃重要性采样或剪裁技巧，从而实现高度离线且稳定的训练；

**🔧 技术方法**

技术方法包括 KL‑regularized RL、最优优势（Optimal Advantage）回归、分组奖励估计、延迟推理策略同步、Pass@k 评估指标、异步训练框架以及对数回归目标的最小化；

**📊 数据集**

使用的数据集涵盖数学竞赛（AIME 25、HMMT 25、BRUMO 25）、DeepScaler 训练集、LiveCodeBench 代码生成数据、Qwen3‑4B‑Thinking 预训练模型以及 DeepCoder 的复制数据集；

**📈 对比分析**

与 GRPO（含重要性采样）和 DeepCoder 进行对比，结果显示在数学竞赛中 OAPL 的 Pass@1/5/10 均优于 GRPO，熵保持稳定；在 LiveCodeBench 上 OAPL 与 DeepCoder 匹配或略优，并在样本效率上提升约 3 倍；在策略滞后高达 400 步的极端情况下仍能保持稳定学习；

**⚠️ 局限性**

局限性包括：需要手动调节同步间隔和 KL 超参数，尚未在自然语言推理或多模态任务上进行验证，KL 约束可能限制探索，导致在极长滞后时收敛速度减慢。

---

## 693. Detector-in-the-Loop Tracking: Active Memory Rectification for Stable Glottic Opening Localization

**arXiv ID:** 2602.19380 | [PDF](https://arxiv.org/pdf/2602.19380v1)

**作者:** Huayu Wang `[一作]` (University of Washington), Jenq-Neng Hwang `[通讯]` (University of Washington)

**通讯引用:** 12602 | [OpenAlex ID](https://openalex.org/A5101702810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种闭环内存纠正（Closed‑Loop Memory Correction，CL‑MC）框架，用于在视频喉镜（Video Laryngoscopy）中实现对声门开口的稳定、可靠定位；

**💡 创新点**

创新点在于将单帧语义检测器（YOLO）与视频分割基座模型（SAM2）通过状态机互联，实现高置信检测主动监督和跟踪器内存的实时纠正，从而避免传统方法的漂移与记忆污染；

**🔧 技术方法**

核心技术包括：①趋势感知置信度归一化，使检测器与SAM2的置信度空间统一；②基于置信度与IoU的状态机决策，动态选择检测或跟踪预测；③主动内存纠正机制，在漂移状态下用检测框重置SAM2的记忆池；④不需要对基座模型进行微调的训练无关闭环控制；

**📊 数据集**

训练集：Laryngoscope8（2497张临床喉镜图像）+583张YouTube医生标注图像；评估集：24段急诊气管插管视频（Harborview Dataset，共8931帧）；

**📈 对比分析**

与YOLO、BoT‑SORT、ByteTrack（运动学关联基准）以及SAM2、SAMURAI（基座模型跟踪器）对比，CL‑MC在mAP_50达到84.32%、AUC 76.52%、漏检率仅6.85%，均优于所有基线，尤其在漂移和遮挡严重时表现最为稳健；

**⚠️ 局限性**

局限性包括：①对单目标定位专注，尚未验证多目标场景；②对极端光照/模糊仍可能出现检测置信度失真；③需要设置若干阈值（τ_init、τ_iou等），在不同环境下可能需微调；

---

## 694. CausalFlip: A Benchmark for LLM Causal Judgment Beyond Semantic Matching

**arXiv ID:** 2602.20094 | [PDF](https://arxiv.org/pdf/2602.20094v1)

**作者:** Yuzhe Wang `[一作]` (University of Virginia), Jundong Li `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CausalFlip基准，构建了包含confounder、chain、collider三种因果结构的事件三元组，并设计了语义相似但标签相反的问答对，用于评估LLM在因果推理上的能力。

**💡 创新点**

首次通过对抗性问题对设计，使模型无法仅依赖语义匹配，并引入噪声前缀评估与隐式因果推理训练策略，以减少对语义的依赖。

**🔧 技术方法**

采用链式思考（CoT）与隐式CoT（逐步遮蔽中间推理步骤的监督）以及传统无CoT训练，并结合LLM生成与评估。

**📊 数据集**

使用自制的CausalFlip数据集（含confounder、chain、collider子集）以及公开的Llama-3.2-3B-Instruct预训练模型进行微调。

**📈 对比分析**

通过对比无CoT、显式CoT与隐式CoT在CausalFlip上的准确率，并在噪声前缀下评估稳健性；隐式CoT在无噪声场景下达到90%+准确率，且在噪声前缀时的降幅更小。

**⚠️ 局限性**

仅在三种因果结构上评估，缺乏更复杂结构和多模态输入；隐式CoT训练仍需手工设定遮蔽策略，难以自动化。

---

## 695. Universal Pose Pretraining for Generalizable Vision-Language-Action Policies

**arXiv ID:** 2602.19710 | [PDF](https://arxiv.org/pdf/2602.19710v1)

**作者:** Haitao Lin `[一作]` (Tencent Robotics), Yanwei Fu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并训练了Pose-VLA，一个将视觉‑语言‑动作模型解耦为先预训练3D空间先验提取再后期物体对齐的框架。

**💡 创新点**

采用统一姿态令牌将RGB‑D与语义信息映射到相机中心空间；预训练中引入光线编码与深度先验；将机器人轨迹映射到相机坐标以实现跨体型泛化。

**🔧 技术方法**

基于PaliGemma的视觉语言模型，加入光线编码、深度先验和离散姿态令牌；使用多模态融合与next‑token预测预训练，后期使用流匹配动作专家实现细粒度控制。

**📊 数据集**

1.4M图像+6.5M 3D标注（Omni3D、Omni6DPose、BOP等），约1.55M机器人轨迹（AgibotWorld、InternData‑A1），以及RoboTwin 2.0、LIBERO、RealWorld Xtrainer实验集。

**📈 对比分析**

与Qwen3‑VL、Seed1.5‑VL、Gemini等基线在3D定位、RoboTwin、LIBERO上对比，Pose‑VLA在Objectron AP_15 87.3、RoboTwin Hard 79.1%、LIBERO平均 96.0% 等取得SOTA性能。

**⚠️ 局限性**

对深度与光线信息依赖较高；仅在相机中心空间预训练，跨相机视角泛化仍有限；离散姿态令牌分辨率有限；需要大量预训练数据；在极端视觉噪声或非结构化环境下表现尚未充分验证。

---

## 696. Hexagon-MLIR: An AI Compilation Stack For Qualcomm's Neural Processing Units (NPUs)

**arXiv ID:** 2602.19762 | [PDF](https://arxiv.org/pdf/2602.19762v1)

**作者:** Mohammed Javed Absar `[一作]` (Qualcomm Technologies International), Zachary Zipper `[通讯]` (Qualcomm Technologies, Inc.)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发Hexagon-MLIR编译栈，实现从PyTorch模型或Triton内核到Hexagon NPU二进制的自动编译。

**💡 创新点**

使用生成式融合+多级IR管线，结合张量级融合、分块、向量化、多线程和双缓冲，解决传统手写库无法覆盖的长链式算子；并提供开源可扩展的MLIR后端。

**🔧 技术方法**

MLIR框架、Triton-to-Linalg、Linalg Generic、Fusion、Tiling、HVX向量化、多线程（Async）、双缓冲、QHL数学库、布局转换、Bufferization、LLVM后端。

**📊 数据集**

基准模型/算子：GELU、RMS‑Norm、SiLU、Flash Attention、Vector‑Add‑2D，主要在float16/float32上测试。

**📈 对比分析**

与单线程、无融合、无双缓冲等基线对比，最大单核向量化可达63.9×，多线程加速2.3~3.9×，双缓冲进一步提升。

**⚠️ 局限性**

仍在开发中，覆盖范围有限，内存带宽敏感场景受限，需进一步优化调度、布局和支持更多算子。

---

## 697. Distributional Stability of Tangent-Linearized Gaussian Inference on Smooth Manifolds

**arXiv ID:** 2602.19179 | [PDF](https://arxiv.org/pdf/2602.19179v1)

**作者:** Junghoon Seo `[一作]` (AI Robot Team PIT IN Corp), Jaehoon Sim `[通讯]` (AI Robot Team PIT IN Corp)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

本文研究在平滑流形上对高斯分布进行切线线性化推断，并给出投影边缘化和曲面测量条件化的 2-Wasserstein 稳定性上界。通过将误差分解为局部几何畸变和非局部尾泄漏，提供了可计算的诊断量，可用于判定单图线性化是否可靠。随后在圆形流形与二维平面推送任务上验证了理论预测，并观察到校准转移点约为 √(Σ_op)/R ≈ 1/6。

**💡 创新点**

创新点包括：①首次以非渐近的 2-Wasserstein 距离形式给出切线线性化推断的稳定性界；②将误差拆解为局部几何和尾泄漏两项，直接关联曲率、切向扩散和分布偏移；③提出了基于 (μ,Σ)、曲率/到达半径的闭式诊断，能够在推断前即预测误差；④在实验中展示了正则化阈值与正常方向不确定性主导失配的对应关系。

**🔧 技术方法**

技术手段包括：Wasserstein-2 距离与极大耦合、曲面二阶微分几何（第二基本形式、到达半径）、拉伸映射（Retraction、Chart）、切线线性化、Gaussian 条件/边缘化公式、集中不等式与四阶矩估计、Monte Carlo 采样做精确基准。

**📊 数据集**

使用的数据集主要为：①合成圆形流形实验（二维圆盘与不同半径、协方差与偏移的组合）；②二维平面推送基准（GTSAM/InCOpt）中的箱子推送轨迹，利用已知接触约束的约束优化求解器产生真值；③通过在轨迹上插值产生的 Monte Carlo 样本得到精确分布。

**📈 对比分析**

比较方法：将单图切线线性化得到的边缘/条件分布与 Monte Carlo 采样得到的精确分布进行比较；使用方差比、95% 区间覆盖率和理论/实测 W_2 诊断量进行评估。实验结果显示，当 √(Σ_op)/R < 1/6 时误差极小；超过该阈值，单图推断明显失配，尤其在正则化方向的协方差放大时失配最为严重；通过曲率/到达半径与偏移指标可以成功预测失配点。

**⚠️ 局限性**

局限性：①上界常数保守，主要依赖通用耦合与集中不等式，实际误差往往低于理论值；②仅适用于 C^2 流形且具有正到达半径，无法覆盖尖角、交叉或奇异几何；③分析未考虑多图或迭代重线性化的情形，未来需扩展至多图更新或样本推断框架。

---

## 698. City Editing: Hierarchical Agentic Execution for Dependency-Aware Urban Geospatial Modification

**arXiv ID:** 2602.19326 | [PDF](https://arxiv.org/pdf/2602.19326v1)

**作者:** Rui Liu `[一作]` (University of Kansas), Dongjie Wang `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种城市地理空间编辑框架，利用分层代理系统根据自然语言指令逐级修改现有 GeoJSON 规划，实现高效、可验证的城市更新。

**💡 创新点**

创新点包括：①将编辑任务分解为多层几何意图（多边形、线、点）并通过任务规划器生成结构化执行计划；②采用层级 GeoExecutor 与自我反思 Validator 进行粗到细的状态传递与中间验证；③将所有验证结果聚合生成最终 GeoJSON，并提供可追溯的执行摘要。

**🔧 技术方法**

技术手段包括：大型语言模型（LLM）驱动的代理体系；自然语言解析与意图拆分；GeoJSON 语义工具链（查询、几何变换）；自我反思式验证与重执行；层级状态转移与上下文传播。

**📊 数据集**

使用从 OpenStreetMap（OSM）通过 Overpass API 提取的 219 个城市的 1km×1km 区域样本，共计 500 条点编辑、500 条线编辑、1500 条多边形编辑的合成指令，构成公开数据集。

**📈 对比分析**

与单次通行（Single-pass）LLM 直接生成编辑序列的基线对比。实验显示，在点、线、面三层任务中，该框架均降低了相对执行误差（REE/ACE），提升了执行有效率（EVR），尤其在面编辑任务上优势显著，整体性能提升约 10–30% 以上。

**⚠️ 局限性**

局限性包括：①对 LLM 的生成质量高度依赖，复杂约束下可能出现误差；②需要预定义的几何工具集，缺乏通用可扩展性；③重执行机制虽提升稳健性但增加计算开销；④未充分融合多方利益相关者的实时反馈与政策约束。

---

## 699. Evaluating Replay Techniques for Asynchronous Task Handover in Immersive Analytics

**arXiv ID:** 2602.18978 | [PDF](https://arxiv.org/pdf/2602.18978v1)

**作者:** Zhengtai Gou `[一作]` (Georgia Tech), Yalong Yang `[通讯]` (Georgia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了异步协作中沉浸式分析会话的回放设计，用以提升任务交接效率

**💡 创新点**

发现不同平台、视角与导航控制对回放效果有显著影响，并提出面向VR与PC的最佳配置与设计准则

**🔧 技术方法**

通过Unity3D实现可记录与重放的沉浸式分析系统，并整合第一/第三人称视角、主动/被动导航及暂停/回放等交互

**📊 数据集**

使用Starbucks、US Colleges、Wine Quality和Spotify数据集生成的分析过程回放

**📈 对比分析**

通过两阶段对比实验评估认知理解、工作流程重建与主观负荷，VR主动第三人称视角显著提升理解与满意度

**⚠️ 局限性**

限制在于未集成显式洞察注释、导航提示与自动剪辑，且样本量有限

---

## 700. Think$^{2}$: Grounded Metacognitive Reasoning in Large Language Models

**arXiv ID:** 2602.18806 | [PDF](https://arxiv.org/pdf/2602.18806v1)

**作者:** Abraham Paul Elenjical `[一作]` (International Institute of Information Technology Hyderabad), Vasudeva Varma `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将 Ann Brown 的元认知循环（规划‑监控‑评估）作为结构化提示框架，并在轻量级双进程 MetaController 中实现，以提升 LLM 的错误诊断和自我纠正能力。

**💡 创新点**

创新点在于将心理学理论与提示工程结合，构建可执行的元认知循环，并设计可动态路由的双进程架构，实现对 LLM 的自适应努力分配。

**🔧 技术方法**

采用结构化元认知提示（Ann Brown 框架）、轻量级 MetaController、基于 Llama‑3 和 Qwen‑3 的 8B 规模模型，以及 Greedy 解码与 2048 长度限制等技术。

**📊 数据集**

使用 GSM8K、CRUXEval、MBPP、AIME、CorrectBench、TruthfulQA 六大基准数据集。

**📈 对比分析**

与标准 Prompt、Chain‑of‑Thought、Metacognitive Prompting 进行对比；在 Qwen‑3 上 Ann Brown 通常位列首位或次优，在 Llama‑3 上在需要错误诊断的任务上表现更好，整体提升自我纠正率三倍，提升人类评估信任度达84%。

**⚠️ 局限性**

局限性包括对非推理预训练模型的认知负担导致性能下降；MetaController 的表面级路由易误判，缺乏持续度量与多级努力分配；仅在提示层面，未将元认知信号融入训练目标。

---

## 701. Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery

**arXiv ID:** 2602.19910 | [PDF](https://arxiv.org/pdf/2602.19910v1)

**作者:** Wei He `[一作]` (Beijing University of Posts and Telecommunications), Chun-Guang Li `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SSR^2-GCD 框架，利用半监督速率降低（SSR^2）学习结构化多模态表示，并通过检索式文本聚合（RTA）增强文本特征，解决 Generalized Category Discovery (GCD) 任务；

**💡 创新点**

创新点在于：①将 Semi‑Supervised Rate Reduction 引入 intra‑modal 对齐，消除对比损失导致的类别压缩不平衡；②提出 RTA 通过聚合多候选标签/属性提升文本表示；③证明在多模态 GCD 中 inter‑modal 对齐不必要，反而可能干扰 intra‑modal 学习；

**🔧 技术方法**

使用 CLIP 预训练视觉与文本编码器、半监督速率降低损失、对比损失、dual‑branch 分类器、co‑teaching、自蒸馏、R_e 互连度、有效秩评估、t‑SNE 可视化等技术；

**📊 数据集**

在八个基准数据集上评估：CIFAR‑10、CIFAR‑100、ImageNet‑100、ImageNet‑1k、CUB‑200‑2011、Stanford Cars、Oxford Pets、Oxford 102 Flowers；

**📈 对比分析**

与多种单模态基线（GCD、GPC、SimGCD、PromptCAL、SPTNet、SelEx、Hyp‑SelEx）和多模态基线（CLIP‑GCD、TextGCD、GET）对比，SSR^2‑GCD 在所有数据集上均取得最高或显著提升的聚类准确率（ACC），尤其在“旧类/新类”准确率差距被明显缩小；

**⚠️ 局限性**

局限性包括：①随着候选标签/属性数量增大，计算和内存开销略增；②目前图像与文本信息权重相同，未探索各模态在 GCD 中的相对重要性，需进一步研究；

---

## 702. CaliCausalRank: Calibrated Multi-Objective Ad Ranking with Robust Counterfactual Utility Optimization

**arXiv ID:** 2602.18786 | [PDF](https://arxiv.org/pdf/2602.18786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 703. Learning to Localize Reference Trajectories in Image-Space for Visual Navigation

**arXiv ID:** 2602.18803 | [PDF](https://arxiv.org/pdf/2602.18803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 704. Physics-informed Active Polarimetric 3D Imaging for Specular Surfaces

**arXiv ID:** 2602.19470 | [PDF](https://arxiv.org/pdf/2602.19470v1)

**作者:** Jiazhang Wang `[一作]` (University of Arizona), Florian Willomitzer `[通讯]` (University of Arizona)

**通讯引用:** 659 | [OpenAlex ID](https://openalex.org/A5069603431)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于物理信息的深度学习框架，实现单次拍摄下高精度复杂镜面物体的三维重建；

**💡 创新点**

创新点在于将偏振信息与几何对应信息通过双编码器与 FiLM 融合方式互补，既利用偏振先验定位法线，又通过网络学习修正几何对应误差，实现无多拍、多视角的全场法线估计；

**🔧 技术方法**

采用 U‑Net 编码器、FiLM 层、共享解码器的双分支网络，并结合 Stokes 参数、DoLP 以及粗略对应图的物理先验；

**📊 数据集**

使用基于 Mitsuba 渲染器构建的数字双胞胎合成数据集，包含 38 个物体、605 张四角偏振图像，图像尺寸 1024×1024，添加 40–50 dB 噪声；

**📈 对比分析**

与前期的解析物理方法和传统单摄像偏振三维成像进行对比，单次拍摄的最终平均角误差仅 0.79°（<1° 区域 73.23%，<2° 区域 93.64%），显著优于传统方法的 4.20°；推理时间仅 8 ms；

**⚠️ 局限性**

局限在于仅使用合成数据训练，未充分建模真实相机微偏振误差与物体材料异质性，且目前仅适用于光滑镜面，难以直接推广到混合或高散射材料。

---

## 705. See What I See: An Attention-Guiding eHMI Approach for Autonomous Vehicles

**arXiv ID:** 2602.18798 | [PDF](https://arxiv.org/pdf/2602.18798v1)

**作者:** Jialong Li `[一作]` (Waseda University), Kenji Tei `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 851 | [OpenAlex ID](https://openalex.org/A5045332896)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并评估了一种基于投影的注意力引导型外部人机接口（AGeHMI），通过空间映射与风险颜色编码，引导行人关注路面周围潜在危险。

**💡 创新点**

创新点在于：①将环境风险与自车意图一起可视化；②使用深度空间投影将车辆位置映射到路面，直观呈现危险程度；③在传统仅传达自车意图的eHMI基础上，加入多车道风险提示，减少“隧道视野”现象。

**🔧 技术方法**

技术手段包括：投影显示系统、VR仿真平台（Unity + Meta Quest 3）、基于距离的风险评估逻辑、色彩编码（绿、黄、红）以及利用头部追踪代替眼动记录。

**📊 数据集**

数据来源为作者自建的虚拟现实驾驶测试场景，共包含 4 种车辆（自车、小车、对向大卡车等）与 22 种随机行驶/不行驶组合；未使用公开数据集。

**📈 对比分析**

通过在三种条件（无eHMI、仅自车意图eHMI、AGeHMI）下进行 20 名受试者的 VR 用户研究，比较潜在碰撞率、视觉关注分布、按钮按压时序及 NASA‑TLX 工作量；结果显示 AG eHMI 将碰撞率从约 2.9 % 降至 ≤ 1.2 %，提升视觉分布覆盖率，降低认知负荷与挫败感，用户信任度与使用体验评分最高。

**⚠️ 局限性**

局限性包括：①VR 环境缺乏真实物理风险与光照影响；②受试者主要为年轻学者，缺乏跨年龄、跨文化验证；③投影可见性受车体颜色、路面纹理等影响；④多投影车辆时易产生视觉杂乱；⑤色彩编码对色盲用户不友好；⑥对传感器与 V2X 的依赖可能导致误报或漏报，且用户可能过度信任提示。

---

## 706. ReVision : A Post-Hoc, Vision-Based Technique for Replacing Unacceptable Concepts in Image Generation Pipeline

**arXiv ID:** 2602.19149 | [PDF](https://arxiv.org/pdf/2602.19149v1)

**作者:** Gurjot Singh `[一作]` (University of Waterloo), Ryan Ko `[通讯]` (University of Queensland)

**通讯引用:** 3306 | [OpenAlex ID](https://openalex.org/A5047816029)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种后置、无训练、基于视觉语言模型的安全框架，对生成图像中的违规概念进行检测并局部替换，作为生成管线的最后防线。

**💡 创新点**

创新点在于：①使用单一通用 VLM（Gemini‑2.5‑Flash）完成违规检测与提示生成；②引入 VLM‑辅助空间门控机制，在扩散过程中对注意力掩模进行实例一致性约束，显著降低 mask 溢出；③构建包含单概念与多概念的 245 张评测图像数据集。

**🔧 技术方法**

核心技术包括：Gemini‑2.5‑Flash 进行语义检测与提示生成；LOCATEdit 进行基于提示的局部编辑；VLM‑辅助实例一致性空间门控；评估指标包括 LPIPS/PSNR/SSIM、CLIP 对齐、类别专用检测器（NudeNet、CopyCAT、GCD）以及人工审核。

**📊 数据集**

使用 Stable Diffusion 3.5 Turbo 生成 245 张图像（170 单概念、75 多概念），覆盖五类违规主题：裸照、版权内容、公众人物、烟酒、暴力/武器。

**📈 对比分析**

与原始未防护模型、传统 LOCATEdit 及其他基线相比，改进了背景保真度（LPIPS↓、PSNR↑、SSIM↑）、CLIP 安全对齐（Δ_clip 改为正值）、类别检测率（NudeNet、CopyCAT、GCD）及人工审核识别率（从 95.99% 降至 10.16%）。

**⚠️ 局限性**

局限性包括：①后置处理会引入额外推理延迟（单概念 2–4 秒，多概念 10–12 秒）；②使用单一 VLM 可能对未见或高度相似的违规概念识别受限；③编辑器与原始生成器架构不匹配可能导致风格一致性问题；④对抗性攻击虽理论上难以突破，但仍未系统验证。

---

## 707. Neural Markov chain Monte Carlo: Bayesian inversion via normalizing flows and variational autoencoders

**arXiv ID:** 2602.19597 | [PDF](https://arxiv.org/pdf/2602.19597v1)

**作者:** Giacomo Bottacini `[一作]` (Politecnico di Milano), Andrea Manzoni `[通讯]` (Politecnico di Milano)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种结合变分自编码器（VAE）、条件归一化流（CNF）和差分进化 Metropolis MCMC 的模拟推断框架，用于高维、非线性逆问题的贝叶斯推断。

**💡 创新点**

创新点在于：①将监督信息嵌入 VAE 的潜在空间，生成既稀疏又包含参数相关性的表示；②利用 CNF 在潜在空间中逼近条件似然，避免频繁调用昂贵的前向模型；③在 DE‑MCMC 中同时采样潜在变量和参数，显著提升采样效率和收敛速度。

**🔧 技术方法**

核心技术包括：变分自编码器（自监督+监督预测分支）、条件 RealNVP 归一化流、差分进化 MCMC、PCA、有限元（FE）、POD‑Galerkin、Karhunen‑Loève（KL）展开等。

**📊 数据集**

数据集：①铁路桥梁结构健康监测的合成观测数据（10k 训练样本、4k 测试样本）；②地下水流域的 hydraulic head 观测数据（32k 训练样本、8k 测试样本）。

**📈 对比分析**

与传统 MCMC + 完整前向模型比较：链长 20k/30k、Gelman‑Rubin < 1.01，单条链运行时间 1.5–3 分钟。桥梁案例定位准确率 93%/97%，误差 ≈6%；地下水案例平均相对误差 ≈0.18。总体表明模型在保持贝叶斯一致性的同时大幅降低计算成本。

**⚠️ 局限性**

局限性：①高维参数的后验不确定性仍随参数维度增加而增大；②潜在空间结构在复杂非线性映射中可能不够清晰；③目前仅验证了仿真生成数据，缺乏真实测量数据评估；④未覆盖时变或实时推断场景，需要进一步改进。

---

## 708. Synthetic Media in Multilingual MOOCs: Deepfake Tutors, Pedagogical Effects, and Ethical-Policy Challenges

**arXiv ID:** 2602.18457 | [PDF](https://arxiv.org/pdf/2602.18457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 709. SKYLIGHT: A Scalable Hundred-Channel 3D Photonic In-Memory Tensor Core Architecture for Real-time AI Inference

**arXiv ID:** 2602.19031 | [PDF](https://arxiv.org/pdf/2602.19031v1)

**作者:** Meng Zhang `[一作]` (Rensselaer Polytechnic Institute), Zhaoran Rena Huang `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 276 | [OpenAlex ID](https://openalex.org/A5103556228)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SKYLIGHT，一种可扩展的三维光子内存张量核心架构，用于实时 AI 推理

**💡 创新点**

创新点包括：3D Si/SiN 交叉电路消除光学交叉损耗；非共振、热稳 WDM 数据通道；使用 VCSEL 光学编程的 PCM 低功耗可擦写权重；多端口光电探测器与分层累积实现百万级通道的高信噪比求和；以及在单晶圆内实现 144×256 通道的完整系统布局

**🔧 技术方法**

采用的技术有：多波长光源(频率梳)、SL‑MZM 调制、Bragg‑grating 辅助波长选择耦合器、光学编程的 N‑GST PCM 单元、垂直耦合 VCSEL、三维 Si/SiN 波导堆叠、光电探测器集成 SOA、混合信号 DAC/ADC、噪声感知量化训练

**📊 数据集**

使用的数据集包括：ImageNet‑1K（ResNet‑50 推理），CSPB‑ML‑2018R2（8 类 RF 信号分类），CIFAR‑10（无监督 SCFF 训练），SpaceNet‑8（洪水分割）

**📈 对比分析**

性能对比：单 144×256 核实现 342.1 TOPS、23.7 TOPS/W；ResNet‑50 推理 1212 FPS、每图像约 27 mJ，整体系统吞吐 84.17 FPS/W，超出 NVIDIA RTX PRO 6000 Blackwell GPU 的 52.27 FPS/W（约 1.61 倍）

**⚠️ 局限性**

局限性包括：对高质量多通道频率梳和精细光学对准的高度依赖；光源功率与链路预算限制；VCSEL 与 PCM 组合的热管理与封装复杂度；以及在更大规模（> 256 通道）或更高温度环境下的热稳定性和制造一致性挑战

---

## 710. UP-Fuse: Uncertainty-guided LiDAR-Camera Fusion for 3D Panoptic Segmentation

**arXiv ID:** 2602.19349 | [PDF](https://arxiv.org/pdf/2602.19349v1)

**作者:** Rohit Mohan `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2564 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UP-Fuse，一个基于不确定性引导的 LiDAR‑Camera 融合框架，用统一的 2D 范围视图（Range‑View）实现 3D 全景分割。

**💡 创新点**

创新点：① 在范围视图空间引入 uncertainty‑aware fusion 模块，动态抑制受损视觉特征；② 设计 Hybrid 2D‑3D Transformer 解码器，缓解投影模糊和 360° 连续性问题；③ 公开 Panoptic Waymo 基准并在多种失效场景下开展鲁棒性评估。

**🔧 技术方法**

技术手段：Swin‑Transformer 编码器、视图变换（View‑Transformation）、变形注意力（Deformable Attention）融合、特征级 aleatoric uncertainty 估计、Hybrid 2D‑3D panoptic 解码器、Huber 损失、多种光照与几何失真增强等。

**📊 数据集**

数据集：Panoptic nuScenes、SemanticKITTI 以及自行生成的 Panoptic Waymo（结合 Waymo Open Dataset 的 3D 语义和框选标签）。

**📈 对比分析**

对比方法：Panoptic‑FusionNet、LCPS、IAL、P3Former 等现有 3D 全景分割与融合模型。性能方面，在 Panoptic nuScenes 验证集上 UP‑Fuse 获得 80.7% PQ（相较 LiDAR‑only 74.9% 提升 5.8%），并在鲁棒性实验（相机失效、校准漂移、视觉域移）中比其他方法损失更小；帧率约 5.7 FPS，显著快于 IAL 等对标模型。

**⚠️ 局限性**

局限性：依赖固定摄像头外参；在严重校准漂移时仅能退化为 LiDAR‑only 性能；未实现外参自适应校正，导致极端失效场景下融合优势受限。

---

## 711. EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease

**arXiv ID:** 2602.19178 | [PDF](https://arxiv.org/pdf/2602.19178v1)

**作者:** Qiuhui Chen `[一作]` (East China University of Science and Technology), Yi Hong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15772 | [OpenAlex ID](https://openalex.org/A5051418301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建了一种端到端的视觉-语言框架 EMAD，用于生成基于多模态（3D sMRI 与临床变量）的透明、证据支持的阿尔茨海默病诊断报告。

**💡 创新点**

创新点包括：① Sentence–Evidence–Anatomy (SEA) 分层对齐机制，实现句子→临床证据→解剖结构的显式归因；② GTX‑Distill 通过教师-学生蒸馏高效迁移稀疏标注的 grounding 信息；③ 可执行规则的 GRPO 强化微调，使用可验证奖励（结构完整性、NIA‑AA 诊断一致性、推理-诊断一致性）保证报告的临床可信度。

**🔧 技术方法**

技术手段包括多模态编码器（3D Vision Transformer + Longformer）、双向交叉注意力融合、因果语言模型（LLaMA 3.2‑1B + LoRA）、3D 语义分割解码器、信息对比（InfoNCE）、KL 蒸馏、PPO‑style GRPO 与可验证奖励。

**📊 数据集**

主要使用了新构建的 AD‑MultiSense 数据集，涵盖 10,378 份 3D sMRI 与 6 类临床信息（人口学、认知、遗传、CSF、实验室、基因）样本，训练时还利用 FastSurfer 生成的解剖掩模做 grounding 监督。

**📈 对比分析**

在 CN‑vs‑CI、CN‑vs‑MCI 和三分类（CN/MCI/AD）任务中，EMAD 在报告质量（BERTScore、BLEU、ROUGE）和诊断指标（ACC、AUC、Sensitivity/Specificity）上均优于 LLaVA、LLaVA‑Med、Med‑PaLM‑M、M3d‑LaMed 等基准；例如在 CN‑vs‑CI 上 ACC 达到 93.33%，AUC 91.83%，比最佳基准高 4–5%。

**⚠️ 局限性**

局限性包括：① 仍需一定量的专业标注（句子‑证据、解剖掩模）来训练教师模型；② 计算资源和推理成本高，尤其是 3D 视觉编码器与 3D 分割解码器；③ 对异常或罕见病例的泛化性尚未彻底验证；④ 强化学习奖励设计需手工制定，可能难以覆盖所有临床细节。

---

## 712. Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models

**arXiv ID:** 2602.19619 | [PDF](https://arxiv.org/pdf/2602.19619v1)

**作者:** Luhan Tang `[一作]` (University of California, Riverside), Greg Ver Steeg `[通讯]` (University of California, Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于oracle的评估框架，用来剔除离散扩散语言模型中因采样动态导致的误差；

**💡 创新点**

创新点在于将学习的去噪器替换为基于真实Markov链的HMM后验，完全隔离采样器错误，使得能单独衡量采样器的分布正确性；

**🔧 技术方法**

采用HMM前向后向推理获取oracle后验，结合SEDD、MDLM、LLaDA、ReMDM等采样器的oracle实例，评估不同采样策略；

**📊 数据集**

主要使用OpenWebText（基于BPE词表）和Text8字符级数据，构造稀疏bigram Markov链作为ground‑truth；

**📈 对比分析**

比较时以transition KL、NLL、熵、n‑gram多样性等转移层指标为主，发现即使在oracle去噪器下，少步采样仍存在显著误差，常用的GenPPL和MAUVE指标可能误导；

**⚠️ 局限性**

局限在于仅考虑一阶Markov链的oracle后验，无法捕捉更高阶语言依赖；此外，框架主要针对离散扩散模型，未对连续扩散模型或更复杂采样器做深入探讨。

---

## 713. Red-Teaming Claude Opus and ChatGPT-based Security Advisors for Trusted Execution Environments

**arXiv ID:** 2602.19450 | [PDF](https://arxiv.org/pdf/2602.19450v1)

**作者:** Kunal Mukherjee `[一作]` `[通讯]` (Virginia Tech), Kunal Mukherjee (Virginia Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出针对可信执行环境安全顾问的红队评估框架，评测 LLM 在 TEE 领域的技术准确性与安全性。

**💡 创新点**

创新点在于结合 TEE 专属威胁模型、可扩展的提示词库、双轨评分法，并研究提示诱发错误的跨模型可转移性。

**🔧 技术方法**

使用大规模提示生成、语义评估、工具调用模拟与验证检查等技术手段。

**📊 数据集**

构建了包含 208 条提示词的 TEE 专属评测集，并对两大 LLM（Claude 5.2 与 ChatGPT 4.6）进行测试。

**📈 对比分析**

通过平均分、过度自信错误率、转移率以及逐层防御消减实验对比，发现完整防御可将失败率降低 80.62%。

**⚠️ 局限性**

局限性包括对单一硬件平台的覆盖有限、评测需手工注解、以及防御措施虽降低错误但仍留有可转移残余风险。

---

## 714. Chat-Based Support Alone May Not Be Enough: Comparing Conversational and Embedded LLM Feedback for Mathematical Proof Learning

**arXiv ID:** 2602.18807 | [PDF](https://arxiv.org/pdf/2602.18807v1)

**作者:** Eason Chen `[一作]` (Carnegie Mellon University), Ken Koedinger `[通讯]` (Carnegie Mellon University)

**通讯引用:** 26487 | [OpenAlex ID](https://openalex.org/A5062550465)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本科离散数学课程中，构建并部署了名为GPTutor的LLM驱动辅导系统，提供了结构化证明评审工具和基于聊天的数学问答两种支持方式，并对148名学生进行了分阶段访问实验，分析其使用模式与学习成效的关系。

**💡 创新点**

创新点在于：①将LLM嵌入两种不同接口——工作锚定的证明评审与开放式聊天，比较它们对学习的影响；②系统性地结合使用频率、会话行为标签、先前学业与自我效能，探究LLM支持的风险路径；③通过序列中介模型揭示聊天使用与考试成绩负相关的机制。

**🔧 技术方法**

主要技术：使用GPT‑4o作为聊天引擎，并将教师提供的参考证明作为上下文向量交给LLM生成局部反馈；利用Python进行日志收集、会话标签自动化分类、混合效应回归、OLS回归和中介分析。

**📊 数据集**

数据集包括：①课程成绩记录（10次作业、3次期中考试）；②GPTutor交互日志（证明评审请求次数、聊天消息数、时间戳）；③自我效能问卷（基线后期测）；⑤人工与LLM自动标注的聊天行为标签（回答寻求、帮助寻求、升级）。

**📈 对比分析**

比较方法：采用分层访问设计的混合效应回归检验先行访问对作业和考试的影响；对使用频率进行多元回归，控制自我效能和先前成绩；使用序列中介模型评估聊天使用对考试成绩的间接效应。结果显示：早期访问显著提升作业成绩（B≈2.7，p=0.026），但对期中考试无显著影响；聊天使用频率与期末考试成绩呈负相关（p=0.014），证明评审使用无显著独立效应。

**⚠️ 局限性**

局限性：①实验设计仅为描述性，缺乏因果结论；②未测量学习时长、替代性学习行为或自我效能随时间变化；③样本仅来自一门课程，外推性受限；④对LLM生成答案的完整性与质量未做深入验证；⑤未评估聊天对学习动机或长期掌握的影响。

---

## 715. Adaptive Problem Generation via Symbolic Representations

**arXiv ID:** 2602.19187 | [PDF](https://arxiv.org/pdf/2602.19187v1)

**作者:** Teresa Yeo `[一作]` (Singapore-MIT Alliance for Research and Technology), Archan Misra `[通讯]` (Singapore Management University)

**通讯引用:** 10625 | [OpenAlex ID](https://openalex.org/A5054849647)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种闭环式数据生成框架，通过在符号空间中优化提示来为强化学习奖励可验证的数学任务生成训练数据，进而提升小型开源语言模型的数学推理能力。

**💡 创新点**

创新点在于将问题先转换为符号表示（如 SymPy），再通过文本梯度（TextGrad）对生成提示进行闭环优化，既实现了对问题结构的精细控制，又能自动生成正确答案，解决了传统开放式生成难以适应模型能力的缺陷。

**🔧 技术方法**

采用的技术包括符号化表示（SymPy/SMT-LIB）、基于文本梯度的提示优化、RLVR（PPO/GRPO）强化学习、以及GPT‑5‑mini 进行数据过滤与验证。

**📊 数据集**

使用的数据集主要为 GSM8K 与 MATH 作为种子数据，并在此基础上通过符号化生成多种变体，构建扩充训练集。

**📈 对比分析**

在多项数学基准（GSM8K、GSM‑Symbolic、MATH‑500、AIME24 等）上，与仅使用种子数据、自然语言基线或非优化提示生成的数据相比，闭环符号化生成的方法在 PPO/GRPO 下平均提升约 3–8 % 的通过率，显示出显著的性能提升与数据效率。

**⚠️ 局限性**

局限性包括：需要可符号化的题目限制了适用范围；生成提示的优化依赖学生模型的错误模式，易出现过拟合；过度依赖专有 LLM（GPT‑5‑mini）导致可复现性与成本问题；目前仅在数学领域验证，尚未推广到其他需要正式化表示的任务。

---

## 716. CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning

**arXiv ID:** 2602.19605 | [PDF](https://arxiv.org/pdf/2602.19605v1)

**作者:** Chunlei Meng `[一作]` (Fudan University), Chun Ouyang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了跨层协同表示（CLCR）框架，将每个模态拆分为浅层、中层、深层三级语义层次，并在每层实现共享-私有分解与预算化的交互，然后跨层同步聚合。

**💡 创新点**

创新点在于：①明确跨层语义异步问题；②通过三层语义层次与IntraCED/InterCAD实现级别对齐与共享/私有分离；③在共享子空间内预算化token交互与跨层anchor引导聚合，减少误传与私有泄漏。

**🔧 技术方法**

使用的技术包括：多层BERT/TCN编码、共享-私有投影（正交基）、预算化交叉注意力、跨层anchor加权聚合、白化相关正则化、Stiefel参数化。

**📊 数据集**

采用的基准数据集有：CREMA-D、AVE、Kinetics-Sounds、UCF101、CMU-MOSI、CMU-MOSEI。

**📈 对比分析**

与多种先前融合与解耦方法（Concat, Grad-Blending, OGM-GE, AGM, PMR, MMPareto, MLA, D&R, ARL, CLCR）以及多模态情感分析基线（MulT, MISA, Self-MM, FDMER, PMR, DMD, CGGM, DEVA, ARL, DLF, EMOE）进行对比，CLCR在所有任务上均取得最高或相近的准确率/F1/MAE等指标，提升幅度约1–3%。

**⚠️ 局限性**

局限性包括：①需要对三层语义层次进行预先设定，可能对新模态或不易划分层次的数据适配性有限；②预算化token选择和anchor权重的学习需要额外超参，训练复杂；③在极端噪声或大规模多模态场景下的可扩展性尚未充分验证。

---

## 717. Boosting for Vector-Valued Prediction and Conditional Density Estimation

**arXiv ID:** 2602.18866 | [PDF](https://arxiv.org/pdf/2602.18866v1)

**作者:** Jian Qian `[一作]` (Hong Kong University), Shu Ge `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了 (α,β)-boostability 的概念，并构建了一个基于指数加权和几何中值聚合的通用结构化预测提升框架

**💡 创新点**

首次给出了在向量预测和条件密度估计下不同散度的几何中值聚合的可提升性阈值，揭示了维度相关与维度无关的提升关系，并提供了间接提升 KL 散度的方法

**🔧 技术方法**

利用几何中值聚合、指数重加权、γ-鲁棒几何中值、弱学习器的超出率保证和散度之间的几何关系

**📊 数据集**

本工作为理论研究，未使用具体数据集

**📈 对比分析**

通过理论分析与对比经典算法（如 MedBoost、AdaBoost、SAMME），证明了在弱学习器满足超出率保证时，提升算法能够实现指数下降的经验超出率误差，达到与传统方法相同或更优的理论收敛速度

**⚠️ 局限性**

几何中值聚合无法直接提升平方根 KL 散度，需要通过 Hellinger 散度间接提升；ℓ1 维度相关性导致维度增长；在某些散度下可能无法满足 (α,β)-boostability 条件

---

## 718. Fully Convolutional Spatiotemporal Learning for Microstructure Evolution Prediction

**arXiv ID:** 2602.19915 | [PDF](https://arxiv.org/pdf/2602.19915v1)

**作者:** Michael Trimboli `[一作]` (Florida Institute of Technology), Xianqi Li `[通讯]` (Florida Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究利用深度卷积时空模型对相场模拟产生的晶粒生长与相分离过程进行预测，并展示其在低分辨率训练后可直接迁移到高分辨率仿真。

**💡 创新点**

创新点包括：①提出无循环、无Transformer的全卷积框架（SimVPv2），通过轻量化卷积和局部注意力实现高效时空建模；②采用自监督序列训练，能在仅有相场图像的情况下学习演化算子；③模型可在低分辨率训练后无改动直接推理高分辨率数据；④显著降低推理开销并保持长时序稳定性。

**🔧 技术方法**

技术手段：SimVPv2全卷积网络、门控时空注意力模块（gSTA）、深度卷积编码器/解码器、滑动窗口自监督训练、并行多步预测、卷积注意力聚合、零填充输入、迭代滚动推理。

**📊 数据集**

数据集：基于相场方程（Allen‑Cahn/ Cahn‑Hilliard）生成的晶粒生长与相分离序列；训练集约1070条（64×64）轨迹，验证集1000条，测试集100条；另外提供256×256高分辨率轨迹用于迁移验证；每轨迹包含200帧，训练样本采用10帧输入预测90帧，测试时还评估10→190、5→95等任务。

**📈 对比分析**

与传统循环模型（ConvLSTM、PredRNN++）比较。RMSE在10→90任务平均<0.11，SSIM>0.86；在10→190、10→200、5→95任务RMSE<0.20，SSIM>0.5；粒度分布（GSD）与真实分布高度一致。推理时间仅30秒，相比ConvLSTM的8分钟、PredRNN++的22分钟提升约30×；FLOPs 9.3 G 远低于1.17 T/3.55 T。整体表明模型在精度、统计一致性和计算效率上均优于现有递归方法。

**⚠️ 局限性**

局限性：①仅评估像素误差和统计特征，缺乏物理守恒（如质量守恒、能量衰减）等指标；②长时序滚动推理仍存在误差累积，未采用多步训练或一致性正则；③模型针对单一演化机制训练，未实现统一多机制或参数可调的通用模型；④未给出不确定性量化或可解释性分析。

---

## 719. Early Evidence of Vibe-Proving with Consumer LLMs: A Case Study on Spectral Region Characterization with ChatGPT-5.2 (Thinking)

**arXiv ID:** 2602.18918 | [PDF](https://arxiv.org/pdf/2602.18918v1)

**作者:** Brecht Verbeken `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 1223 | [OpenAlex ID](https://openalex.org/A5049169851)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用ChatGPT-5.2通过可审计的生成-评审-修正对话流程，完成了对Ran和Teng 2024年关于4周期行随机非负矩阵谱域Conjecture 20的完整可验证证明。

**💡 创新点**

创新点在于将大型语言模型与人类专家的“生成‑评审‑修正”循环系统化应用于科研级数学证明，并通过版本化草稿与对话日志实现过程可审计，揭示LLM在高层结构搜索中的有效性与验证瓶颈。

**🔧 技术方法**

所用技术包括ChatGPT-5.2（Thinking模式）进行推理生成、人工评审与修正、Lamport式依赖拆解、对话日志与版本控制、以及Python符号验证工具。

**📊 数据集**

数据集主要为人工构造的对话日志与证明草稿，不依赖公开数学数据集，证明的验证通过可审计的对话记录完成。

**📈 对比分析**

与传统形式化证明（如Lean/Coq）相比，本方法在速度与人力成本上更具优势，能够快速生成可验证的证明草稿，但缺乏完整的机械化验证，故在证明完整性上仍需人工审核；在证明成功率与可审计性方面表现相当。

**⚠️ 局限性**

局限性包括：依赖特定结构化问题（4周期谱域）；LLM仍可能产生错误，验证瓶颈集中在符号展开与不等式检验；未测试无结构化或更一般性问题；未实现完全形式化验证。

---

## 720. GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data Layout

**arXiv ID:** 2602.19873 | [PDF](https://arxiv.org/pdf/2602.19873v1)

**作者:** Felix Thaler `[一作]` (Swiss National Supercomputing Centre), Sebastian Keller `[通讯]` (Swiss National Supercomputing Centre)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种GPU原生的压缩邻居列表，结合Hilbert曲线数据布局和Cornerstone八叉树实现高效的邻居搜索与对称对流体动力学与引力耦合计算。

**💡 创新点**

创新点在于：①利用Hilbert曲线的空间紧凑性直接构建SIMD友好的聚类邻居列表，无需额外重排；②提出基于四分位变量长度编码的邻居列表压缩算法，显著降低内存占用；③实现完全在GPU上构建邻居列表，并支持可变截断半径与高密度对比的粒子分布。

**🔧 技术方法**

技术包括：GPU原生CUDA/HIP编程、Cornerstone八叉树、Hilbert曲线空间填充、SIMD/SIMT向量化、按位掩码与位操作、四分位变长编码压缩、结构化/面向数组（SoA）数据布局、混合精度计算。

**📊 数据集**

数据集涵盖：1）4百万原子均匀分布的Lennard‑Jones（12/6）与库仑相互作用基准；2）Evrard坍塌（含可变半径的非均匀粒子云）进行弱缩放测试，至1024个GH200 GPU；3）33.5百万粒子SPH模拟用于结果验证。

**📈 对比分析**

通过与LIGMPS（全/半Verlet列表）和GROMACS（聚类邻居列表）在NVIDIA GH200与AMD MI300A GPU上进行核运行时间、构建时间、内存占用和邻居计数等指标对比；结果显示：在GH200上大邻居计数时全列表最快，而在MI300A上聚类列表在大多数情况下更快；压缩后内存占用降至4B/粒子，性能损失≤5%。

**⚠️ 局限性**

主要局限在于：①聚类邻居导致额外的非截断对数计算，尤其在邻居数较少时显著；②在高缓存GPU（如GH200）上全列表在内存足够时仍能击败聚类列表；③Hilbert曲线的紧凑性仅在平均意义上保证，极端分布可能导致较大构建开销；④压缩与解压开销在某些GPU上可达20%。

---

## 721. Adaptive Data Augmentation with Multi-armed Bandit: Sample-Efficient Embedding Calibration for Implicit Pattern Recognition

**arXiv ID:** 2602.19385 | [PDF](https://arxiv.org/pdf/2602.19385v1)

**作者:** Minxue Tang `[一作]` (Duke University), Taha Belkhouja Yujia Bao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种在极少训练样本下对预训练嵌入模型进行轻量化校准的框架，利用残差神经相似度网络实现对查询和标签嵌入的微调，称为 ADAMAB。

**💡 创新点**

创新点在于将轻量化校准器与改进的多臂老虎机（MAB）算法结合，使用上置信界（UCB）驱动自适应数据增广，并给出梯度下降的收敛保证。

**🔧 技术方法**

核心技术包括残差结构的轻量化相似度网络、基于 UCB 的自适应增广策略、梯度下降优化以及通用生成模型（如 GPT‑4o‑mini、GPT‑Image‑1‑mini）生成合成样本。

**📊 数据集**

实验涵盖六个跨模态数据集：文本类的 MultiWD、Forbidden Question Set (FQS)、TREC；图像类的 OxfordPets、Flowers102、CUB200。

**📈 对比分析**

与多种基线（LLM 生成、重排器、嵌入模型、随机增广）比较，在零样本和少样本场景下平均提升 20–40% 的准确率，最高可达约 89% 的分类准确率。

**⚠️ 局限性**

局限性包括对生成模型质量的高度依赖、合成样本可能产生同质化导致过拟合、以及在极小样本或极大类别数时的性能下降。

---

## 722. Retrieval Augmented Enhanced Dual Co-Attention Framework for Target Aware Multimodal Bengali Hateful Meme Detection

**arXiv ID:** 2602.19212 | [PDF](https://arxiv.org/pdf/2602.19212v1)

**作者:** Raihan Tanvir `[一作]` (BRAC University), Md. Golam Rabiul Alam `[通讯]` (BRAC University)

**通讯引用:** 3645 | [OpenAlex ID](https://openalex.org/A5051981813)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对低资源语言孟加拉语的仇恨表情包进行检测，提出了增强版的双重共注意框架 xDORA 并结合检索增强与非参数 k‑NN 方案。

**💡 创新点**

通过将 BHM 数据集与 MIMOSA 样本语义对齐扩充数据，设计了多编码器双重共注意机制，并将检索增强 (FAISS) 与模型输出融合，显著提升少数类性能。

**🔧 技术方法**

使用 CLIP/DINOv2 视觉编码器、XGLM/XLM‑R 文本编码器、加权注意池化、FAISS k‑NN、RAG 融合以及 LLaVA 生成式推理。

**📊 数据集**

扩展后的 BHM（9342 条）和 MIMOSA（2233 条）样本的孟加拉语仇恨表情包数据。

**📈 对比分析**

与 Vision‑Only、Text‑Only、原 DORA、MAF、LLaVA 以及基线对比，xDORA 在二分类宏 F1 0.78、四分类 0.71，RAG‑Fused DORA 达到 0.79/0.74，显著优于基线；k‑NN 也表现竞争力。

**⚠️ 局限性**

在零/少量样本设置下，LLaVA 等大模型仍表现不足，代码混杂和极少数类（TS）仍难以充分学习，且依赖大量 GPU 资源。

---

## 723. Detecting AI-Generated Forgeries via Iterative Manifold Deviation Amplification

**arXiv ID:** 2602.18842 | [PDF](https://arxiv.org/pdf/2602.18842v1)

**作者:** Jiangling Zhang `[一作]` (Wuhan University of Technology), Ziyu Chen `[通讯]` (Wuhan University of Technology)

**通讯引用:** 668 | [OpenAlex ID](https://openalex.org/A5101544455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于冻结MAE的双阶段闭环框架IFA-Net，用真实图像重建不一致性来实现图像伪造的像素级定位。

**💡 创新点**

创新点在于把关注点从学习伪造模式转向建模“真实”，通过冻结MAE的重建差异进行异常发现，并利用任务自适应提示（TAPI）引导重建放大异常，实现闭环放大与细化。

**🔧 技术方法**

使用的技术包括冻结的Masked Autoencoder预训练权重、双流分割网络（DSSN）、FiLM基任务自适应提示注入（TAPI）、交叉注意力融合、以及BCE+Dice损失的联合训练。

**📊 数据集**

评估数据集涵盖四个扩散模型生成的伪造基准（OpenSDID、GIT10K、CocoGlide、Inpaint32K）和三个传统篡改基准（IMD2020、NIST16、CASIA）。

**📈 对比分析**

与现有方法（如TruFor、PSCC-Net、MVSS-Net、Span、IML-ViT、MaskCLIP、DcDsDiff等）比较，IFA-Net在所有生成式基准平均IoU达0.778、F1达0.855，并在传统基准上平均F1 0.708，均显著优于或排名第二。

**⚠️ 局限性**

局限性包括在极端后处理（高压缩或强模糊）下性能下降，且对视频或多模态伪造的适应性尚未验证。

---

## 724. Orbital Escalation: Modeling Satellite Ransomware Attacks Using Game Theory

**arXiv ID:** 2602.18624 | [PDF](https://arxiv.org/pdf/2602.18624v1)

**作者:** Efrén López-Morales `[一作]` (New Mexico State University), Efrén López-Morales `[通讯]` (New Mexico State University)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5073390580)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并求解了基于堆叠伯格尔游戏的卫星勒索软件攻击模型——轨道升级游戏，用以描述攻击者随轨道通行升级赎金而防御者在每次通行中选择支付、等待、恢复或拒绝的决策。

**💡 创新点**

创新点在于：①首次将游戏理论应用于卫星勒索攻击；②将轨道通行作为时间维度，将赎金动态升级与恢复时间、成功概率等经济参数耦合；③通过动态规划得到最优防御策略并演示平衡解。

**🔧 技术方法**

使用了堆叠伯格尔游戏、动态规划（贝尔曼方程）以及概率恢复模型等理论技术。

**📊 数据集**

采用公开的 GPS III 卫星成本与恢复方案（如安全模式、特权遥控）以及假设的攻击者成本和赎金参数进行案例研究；并提供 GitHub 代码实现。

**📈 对比分析**

通过与现有陆地勒索游戏的对比（如 Laszka 等、Caporusso 等）表明轨道升级游戏在游戏阶段、玩家策略和经济参数上更完整；实验结果显示在最佳防御策略下卫星运营成本可从数亿美元降至约千万元，攻击者收益被压制。

**⚠️ 局限性**

局限性包括：假设完美信息且玩家完全理性；仅适用于 LEO/MEO 轨道且以单一通行为时隙；未考虑多地面站、多卫星互联、攻击者对恢复策略的隐藏或不确定性；以及缺乏实测数据验证。

---

## 725. Parallelizable Neural Turing Machines

**arXiv ID:** 2602.18508 | [PDF](https://arxiv.org/pdf/2602.18508v1)

**作者:** Gabriel Faria `[一作]` (University of São Paulo), Arnaldo Candido Junior `[通讯]` (São Paulo State University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可并行化的简化版神经图灵机（P-NTM），通过移除内容寻址和循环控制状态，实现了基于扫描的并行计算。

**💡 创新点**

创新点在于：①对原始NTM进行结构简化，去掉内容寻址和隐状态，使其仅依赖当前输入产生控制信号；②利用扫描（scan）算法在内存寻址与写入上实现并行化，显著提升训练与推理速度；③引入稳定化阈值控制移位权重，保持长序列推理的数值稳定。

**🔧 技术方法**

技术手段包括：循环网络改写为无状态的线性控制（minGRU），扫描（parallel scan）并行计算，FFT/频域卷积与对数空间实现的稳定寻址与写入，以及对齐的自回归输出。

**📊 数据集**

使用了人工合成的算法任务数据集，包含 6 类任务：奇偶校验（PC）、环路导航（CN）、字符串反转（RS）、字符串复制（DS）、模运算（MA）和二进制加法（BA），每个任务的输入长度在训练阶段 1–40，测试阶段 41–120。

**📈 对比分析**

与 LSTM、minGRU、Transformer（多种位置编码）以及标准 NTM 进行比较。P-NTM 在所有任务上均实现 100% 的长序列（长达 120）的准确率，完全匹配标准 NTM；相较于标准 NTM，P-NTM 的并行推理速度提升 3.6–18.5 倍；在训练速度上，P-NTM 的并行实现比标准 NTM 快约 10 倍。

**⚠️ 局限性**

局限性：1）依赖中间自回归输出做状态表示，若缺少或噪声的中间步骤，学习效果会下降；2）在需要严格顺序推理（如在线强化学习）时并行优势无法体现；3）并行扫描需按序列长度占用线性内存，极长序列可能需分块处理；4）对初始条件敏感，随机初始化时成功率略低于标准 NTM。

---

## 726. ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?

**arXiv ID:** 2602.19594 | [PDF](https://arxiv.org/pdf/2602.19594v1)

**作者:** Ayush Nangia `[一作]`, Paras Chopra `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ISO-Bench 基准，构造 54 条来自 vLLM 与 SGLang 的 GPU 推理优化任务，并评估编码代理在这些真实代码库中的性能。

**💡 创新点**

1) 结合硬件执行度量与 LLM‑as‑a‑Judge 两类指标，形成四象限评估框架；2) 发现代理的主要失败模式是执行失效而非理解瓶颈；3) 展示代理性能随代码库显著变化，凸显架构与模型的相互作用。

**🔧 技术方法**

使用 Claude Code、Codex CLI 与 TRAE‑Agent 等编码代理；软评估采用 Gemini‑3‑Flash‑Preview 作为 LLM‑Judge；硬评估通过 TTFT、吞吐量等指标，实验环境为 Docker + NVIDIA H100 GPU。

**📊 数据集**

ISO‑Bench 数据集包含 54 个任务：39 条来自 vLLM，15 条来自 SGLang，每条任务提供仓库快照、基准模型与性能指标。

**📈 对比分析**

对比代理生成的补丁与人类专家提交的原始补丁。硬指标衡量性能提升（TTFT、吞吐），软指标判断是否定位正确瓶颈；四象限框架将结果分为 Q1–Q4。结果显示硬指标下成功率较高，但真实成功率（Q1）约 20–40%；不同代理在 vLLM 与 SGLang 的排名互相交叉。

**⚠️ 局限性**

局限性：数据集仅 54 条，缺乏多 GPU 与不同硬件场景；任务聚焦局部补丁，未覆盖大规模系统级优化；软评估依赖单一 LLM 判定；潜在训练泄漏风险，未做时间过滤或代码重写；评估仅在 NVIDIA H100 GPU 上完成。

---

## 727. ReportLogic: Evaluating Logical Quality in Deep Research Reports

**arXiv ID:** 2602.18446 | [PDF](https://arxiv.org/pdf/2602.18446v1)

**作者:** Jujia Zhao `[一作]` (Leiden Institute of Advanced Computer Science), Zhaochun Ren `[通讯]` (Leiden Institute of Advanced Computer Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReportLogic基准来评估深度研究报告的逻辑质量，并构建了LogicJudge评估器。

**💡 创新点**

创新地将逻辑质量细化为宏观、阐释和结构三层税onomy，并通过实例化的上下文敏感rubric实现细粒度诊断；同时提供人类标注数据和对抗鲁棒性分析。

**🔧 技术方法**

采用层次化的逻辑税onomy、上下文感知rubric生成（使用Claude-4.5-Sonnet）、人类双重标注、教师模型蒸馏、SFT+GRPO训练逻辑评判器，并开展对抗攻击评测。

**📊 数据集**

使用DeepResearch、Zhihu、Quora三大问答集的深度研究查询，检索多来源上下文，并由DeepSeek-V3、Claude-4-Sonnet、GPT-4o、Qwen-Max等模型生成报告，构成人类标注数据集。

**📈 对比分析**

通过与17个主流LLM评判器和两种集成基线在报告对比任务上的一致率比较，LogicJudge在所有数据集上与人工评审的吻合度最高，并在对抗攻击中表现出更低的成功率。

**⚠️ 局限性**

仅适用于深度研究报告，无法直接推广到创意、叙事或说服性写作；且本工作仅评估逻辑缺陷，未给出提升逻辑生成的具体方法。

---

## 728. Sculpting the Vector Space: Towards Efficient Multi-Vector Visual Document Retrieval via Prune-then-Merge Framework

**arXiv ID:** 2602.19549 | [PDF](https://arxiv.org/pdf/2602.19549v1)

**作者:** Yibo Yan `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1210 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段压缩框架，先自适应剪枝低信息块，再层次聚合剩余高语义块，以压缩多向量视觉文档检索模型的存储和计算开销。

**💡 创新点**

创新点在于将剪枝与聚合协同组合，既利用剪枝的精准信息筛选，又避免单一聚合导致的特征稀释，从而在高压缩率下保持近乎无损检索性能。

**🔧 技术方法**

使用了基于视觉语言模型的自注意力重要性评估进行自适应阈值剪枝，随后采用层次聚类（Ward方法）生成语义聚类中心作为压缩后的向量。

**📊 数据集**

在29个主流视觉文档检索基准（包括ViDoRe‑V1/V2、JinaVDR、REAL‑MM‑RAG、ViDoSeek、MMLongBench‑Doc等）上进行实验，基准模型为ColQwen2.5、ColNomic和Jina‑v4。

**📈 对比分析**

与基线（随机剪枝、Attention‑Plus‑Similarity、DocPruner、Sem‑Cluster、1D/2D Pooling等）比较，提出的框架在保持0.5%以内nDCG@5性能的同时，实现约55%存储压缩，且在高达80–90%压缩率下仍优于单一剪枝或聚合方法。

**⚠️ 局限性**

局限性包括：剪枝阶段依赖LVLM内部注意力作为信息重要性估计，可能受模型训练偏差影响；框架需手动设定适配因子k和聚合因子m，缺乏自动化自适应机制。

---

## 729. Reassurance Robots: OCD in the Age of Generative AI

**arXiv ID:** 2602.19401 | [PDF](https://arxiv.org/pdf/2602.19401v1)

**作者:** Grace Barkhuff `[一作]` (Georgia Institute of Technology), Grace Barkhuff `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 26 | [OpenAlex ID](https://openalex.org/A5018998088)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对100条 Reddit 上 r/OCD 子版块中包含“AI”或“ChatGPT”关键词的公开帖子进行定性分析，探讨了人工智能如何产生新的强迫症冲动与强迫行为。

**💡 创新点**

首次系统识别了 AI 引发的强迫症冲动与强迫行为，并提出“Reassurance Robots”概念，指出 AI 在提供安慰、承认、决策时可能加剧患者症状。

**🔧 技术方法**

采用扎根理论与常量比较法进行编码，并手动重述原始语句以保护用户隐私。

**📊 数据集**

使用来自 r/OCD 子版块的100条公开帖子作为数据集。

**📈 对比分析**

未进行定量性能比较，而是通过主题饱和度和质性解释验证研究结果的可靠性与丰富性。

**⚠️ 局限性**

局限包括样本仅来自单一社交平台、样本量有限、缺乏临床专家访谈以及重述过程中可能导致语义失真。

---

## 730. NeuroWise: A Multi-Agent LLM "Glass-Box" System for Practicing Double-Empathy Communication with Autistic Partners

**arXiv ID:** 2602.18962 | [PDF](https://arxiv.org/pdf/2602.18962v1)

**作者:** Albert Tang `[一作]` (Marriotts Ridge High School), Jiahuan Pei `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 317 | [OpenAlex ID](https://openalex.org/A5061075100)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计并评估了 NeuroWise，一款集情绪可视化、内部状态解释和即时沟通建议于一体的 AI 辅助工具，旨在帮助神经典型人士更有效地与自闭症伙伴沟通。

**💡 创新点**

创新点在于将双重共情的归因维度与 LLM 驱动的多代理系统相结合，提供可视化情绪反馈、解释器阐释内在体验以及针对情境的教练建议，聚焦于帮助用户理解而非单纯纠正行为。

**🔧 技术方法**

采用多代理 LLM 架构（基于 GPT‑4o‑mini），结合情绪估计模块、规则加权情绪条形图和解释器/教练代理，实现实时交互与反馈。

**📊 数据集**

使用实验设计的脚本化对话（约 15 组、63 回合）以及 30 名神经典型参与者的对话与问卷数据进行评估；情绪估计算法亦在两名人工评审者对同样脚本的评分中验证。

**📈 对比分析**

通过与无辅助基线聊天机器人进行 30 人的随机实验，使用 Mann‑Whitney U 和 Wilcoxon 检验，NeuroWise 在减少归因偏差（δ = -0.49）、提升功能接受度（平均 6.60/7）和提升对话效率（中位数对话轮数从 11 降至 8，δ = -0.48）方面表现出显著且大效应的优势。

**⚠️ 局限性**

局限性包括仅在单一模拟情境下进行即时评估，缺乏真实世界长期跟踪验证，且解释器的内部假设可能未充分覆盖自闭症谱系的多样性与个体差异。

---

## 731. A Mixed-Method Framework for Evaluating the Social Impact of Community Cooperation Projects in Developing Countries

**arXiv ID:** 2602.20009 | [PDF](https://arxiv.org/pdf/2602.20009v1)

**作者:** Giorgia Sampò `[一作]`, Zelda Alice Franceschi `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实证验证了一种名为 PIRA 的混合方法框架，用以评估发展中国家社区合作项目对社会网络结构的影响。

**💡 创新点**

创新点在于引入了新的自我中心度量 2‑ELDA，能够识别“建筑性改变者”（architectural alters）——即在项目实施中起到桥梁作用的隐性节点，从而补充传统网络中心性与中介作用的不足。

**🔧 技术方法**

技术方法包括社会网络分析（SNA）与人类学访谈的混合研究；在网络层面使用全网络与自我中心网络（ego‑network）分析，并通过 2‑ELDA 量化间接连结；在定性层面采用田野观察、半结构访谈与问卷访谈。

**📊 数据集**

数据集为 2026 年 2 月在坦桑尼亚 Pomerini 村进行的 3 个月现场研究，收集了 382 名受访者的关系问卷（包含家庭、朋友、同事及项目产生的关系），并构建了 2881 节点、5222 条边的自我中心网络以及 382 节点、1885 条边的全网络。

**📈 对比分析**

通过对比项目实施前后（或去除项目产生的关系的“对照网络”）的网络指标（如核心/边缘结构、密度、碎片化、中心化、平均路径长度、同质性等）评估 PIRA 的效果。结果显示，项目参与显著降低网络中心化、提高连通性，并通过 2‑ELDA 明确了关键桥梁节点，表明 PIRA 在捕捉结构性变化方面具有较高的可操作性和解释力。

**⚠️ 局限性**

主要限制包括：仅使用单一时间点的数据，缺乏纵向因果推断；数据清洗过程中涉及研究者主观判断，可能影响可靠性；案例聚焦于单一村落，普适性有限，未来需要在不同文化与项目类型中验证。

---

## 732. One Size Fits None: Modeling NYC Taxi Trips

**arXiv ID:** 2602.19404 | [PDF](https://arxiv.org/pdf/2602.19404v1)

**作者:** Tomas Eglinskas `[一作]` `[通讯]` (University of Texas at Austin), Tomas Eglinskas (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对2024年纽约市280多万次出租车与网约车行程进行小费预测模型的构建与评估。

**💡 创新点**

揭示不同车种小费可预测性差异，证明通用模型因Simpson悖论而失效，并提出按车种分模型的必要性。

**🔧 技术方法**

使用线性回归、CatBoost梯度提升、Tweedie损失优化、深度神经网络等多种机器学习技术。

**📊 数据集**

基于纽约市TLC 2024年黄、绿、HVFHS三类行程记录数据，约2.8亿条交易记录。

**📈 对比分析**

通过MAE、RMSE、R²三指标比较不同算法，非线性模型在黄绿车种将R²提升至0.68–0.70，HVFHS仍低；统一模型虽全局R²为0.57，但各子集出现负R²，显示性能失衡。

**⚠️ 局限性**

由于网约车小费受不可观测因素影响，现有特征空间难以捕捉足够信号，模型难以进一步提升，需要更多外部数据或新特征。

---

## 733. Characterizing MARL for Energy Control: A Multi-KPI Benchmark on the CityLearn Environment

**arXiv ID:** 2602.19223 | [PDF](https://arxiv.org/pdf/2602.19223v1)

**作者:** Aymen Khouja `[一作]` (InstaDeep), Ruan De Kock `[通讯]` (InstaDeep)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对六种多智能体强化学习算法在 CityLearn 能源管理环境中的性能进行全面基准测试。

**💡 创新点**

提出了新的关键性能指标（如电池放电深度、个体代理重要性和极端情况表现），并用多种统计方法（IQM、CVaR、改进概率、排名分布）评估算法在平均性能、风险敏感度和鲁棒性方面的差异。

**🔧 技术方法**

采用了 PPo、SAC 等在 DTDE（独立）与 CTDE（集成）两种训练范式下的实现，并对其前馈与递归（GRU）两种网络架构进行对比；使用了 Sebulba 架构和多种奖励设计。

**📊 数据集**

使用 CityLearn 2023 数据集（6 栋建筑、三个月时间范围）作为实验环境，并在所有算法上统一调参、种子采样和完整的评估协议。

**📈 对比分析**

通过多次种子实验、严谨的超参数搜索和统计分析，发现 IPPO 在 IQM 与 CVaR 上表现最优；递归模型显著提升了与时间相关的 KPI（如 ramping、电池使用），但对即时响应指标（如舒适度）提升有限；CTDE 方案表现更不稳定，且在最坏情况时更易受扰。

**⚠️ 局限性**

局限性包括：仅在 CityLearn 环境下验证，未覆盖更大规模或更复杂的城市网络；电池退化未直接纳入奖励函数；MAPPO 的中心化调度导致计算开销大且易受观测维度影响；缺乏对通信延迟或故障场景的系统性评估。

---

## 734. US-JEPA: A Joint Embedding Predictive Architecture for Medical Ultrasound

**arXiv ID:** 2602.19322 | [PDF](https://arxiv.org/pdf/2602.19322v1)

**作者:** Ashwath Radhachandran `[一作]` (University of California), William Speier `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了US-JEPA，一种基于联合嵌入预测（JEPA）和静态教师（SALT）的无监督学习框架，专门为超声图像设计；

**💡 创新点**

首次在超声领域使用JEPA进行潜在空间预测，采用静态教师URFM实现SALT以解耦学生与教师，加入USrc掩蔽机制排除非解剖噪声，并在公开数据上构建最大规模的预训练语料；

**🔧 技术方法**

使用Vision Transformer编码器、SALT的静态教师预测、USrc的解剖掩码、线性探测器进行下游评估、以及对高斯模糊、对比度降低、空间相关斑点噪声的鲁棒性测试；

**📊 数据集**

预训练使用约4.73 M帧，来自49个公开超声数据集（22个解剖部位）；下游评估基于UltraBench的8个分类任务（AUL、BUSBRA、BUTTERFLY、FATTY LIVER、GBCU、MMOTU、POCUS、TN5000）；

**📈 对比分析**

采用冻结特征的线性探测（5个随机种子）与USFM、URFM、USF‑MAE、EchoCare、UltraSAM、SAMUS、DINOv3、I‑JEPA对比；US‑JEPA在5项任务夺得冠军，在2项任务第二，整体性能优于基线，且在少样本和噪声条件下表现更稳健；

**⚠️ 局限性**

对部分器官（胆囊、甲状腺）表现不足，受预训练数据不均衡影响；在对比度退化的鲁棒性有限；模型依赖大型公开数据集，尚未覆盖所有设备与采集工况；使用静态教师可能限制进一步性能提升。

---

## 735. Learning from Complexity: Exploring Dynamic Sample Pruning of Spatio-Temporal Training

**arXiv ID:** 2602.19113 | [PDF](https://arxiv.org/pdf/2602.19113v1)

**作者:** Wei Chen `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 24093 | [OpenAlex ID](https://openalex.org/A5011384237)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对时空预测任务中样本冗余严重的问题，提出了一种专门针对时空数据的动态样本修剪框架（Dynamic Sample Pruning，DSP），在训练过程中自适应地评估并保留信息量高的样本，从而显著减少计算量并保持甚至提升预测性能。

**💡 创新点**

创新点包括：① 发现并量化了“平均遮蔽效应”与“长期静态分布”两种时空特有的冗余现象；② 设计了基于误差空间/时间方差的“时空复杂度评分”来识别结构性难样本；③ 引入了“静态性感知梯度重缩放”，在样本削减后通过动态加权补偿分布偏移；④ 采用随机软剪枝与学习率退火的训练调度，以保证梯度估计无偏且方差可控。

**🔧 技术方法**

核心技术包括：时空误差统计（空间/时间方差）、复杂度评分函数、随机软剪枝策略、静态性感知权重调整、训练熔化（Annealing）与全数据后期微调。

**📊 数据集**

在交通领域的 PeMS08、交通与能源混合数据集（含 SD、GBA、GLA 子集）以及大型时空基准 LargeST 上进行评估；同时将方法与 ST‑Foundation Model（Mini/Base/Plus）结合，验证其在大规模模型训练中的可扩展性。

**📈 对比分析**

与多种静态与动态剪枝基线（随机、几何、基于不确定性、基于损失、子模、双层等）对比，DSP 在 10%–70% 保留比例下均实现了 1–5% 的 MAPE/MAE 改进，甚至在某些低冗余数据集上超过全数据训练；同时实现约 2 倍的训练速度提升（每 epoch 时间约 50% 缩减），在极端 10× 加速时仍保持性能可接受。

**⚠️ 局限性**

主要局限包括：① 需要手动调节复杂度权重 λ 和 annealing 参数 δ，超参数敏感性在不同任务/模型间可能略有差异；② 目前针对离线静态数据的剪枝方案，对持续学习/时空拓扑演化场景尚未适配；③ 在极低保留率（≤1%）下，梯度方差上升仍可能影响收敛，需进一步研究更稳健的重采样或加权策略。

---

## 736. LoMime: Query-Efficient Membership Inference using Model Extraction in Label-Only Settings

**arXiv ID:** 2602.18934 | [PDF](https://arxiv.org/pdf/2602.18934v1)

**作者:** Abdullah Caglar Oksuz `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**通讯引用:** 2622 | [OpenAlex ID](https://openalex.org/A5028326739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在仅可查询标签的黑盒环境下，通过模型提取构建高保真度代理模型，随后在代理模型上进行离线成员推断，从而显著降低对目标模型的查询成本。

**💡 创新点**

创新之处在于将模型提取与标签仅推断结合，利用主动采样与自合成数据实现一次性高质量提取，且不依赖公共数据或置信度信息。

**🔧 技术方法**

采用MARICH与AUTOLYCUS的混合主动采样与噪声增广、互信息最大化模型提取，随后使用距离阈值方法进行无监督成员推断。

**📊 数据集**

在Location、Purchase和Texas Hospital三大基准表格数据集上进行实验。

**📈 对比分析**

与传统标签仅MIA（如Choquette‑Choo等）及对抗防御进行对比，结果显示在仅1%训练样本查询预算下即可获得与最先进方法相当的准确率和AUC，且查询总量显著下降。

**⚠️ 局限性**

局限在于高维稀疏或复杂特征空间下需要更多查询；对大型深度模型或图像数据的适用性仍待验证。

---

## 737. Ensemble Prediction of Task Affinity for Efficient Multi-Task Learning

**arXiv ID:** 2602.18591 | [PDF](https://arxiv.org/pdf/2602.18591v1)

**作者:** Afiya Ayman `[一作]` (Pennsylvania State University), Aron Laszka `[通讯]` (Pennsylvania State University)

**通讯引用:** 2408 | [OpenAlex ID](https://openalex.org/A5049435924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出ETAP框架，用白盒梯度相似度与数据驱动残差预测相结合，提前估计多任务学习（MTL）收益并指导任务分组；

**💡 创新点**

创新在于：①单模型即可计算任务梯度相似度；②用B-spline非线性映射把相似度映射为MTL收益；③残差岭回归进一步校正高阶交互；③实现训练组数大幅降低、计算成本显著下降；

**🔧 技术方法**

使用白盒梯度相似度计算、B-spline基函数展开+正则化线性回归、残差岭回归、分支限界搜索以及跨域实验评估；

**📊 数据集**

在四个多任务基准上验证：CelebA（视觉）、ETTm1（时间序列）、Chemical（分子分类）和Ridership（交通乘客）；

**📈 对比分析**

与TAG、GRAD-TAE、MTGNet、Linear Surrogate、PCGrad等基线对比。MTL收益预测的相关性、R²、F1均优于基线；在组选择实验中，ETAP实现的总收益接近最优，且计算成本（训练组数）低于其他方法；

**⚠️ 局限性**

仍需一定量的训练组来拟合残差；对极大任务集合的高阶分组搜索仍受限；依赖梯度信息，若模型不共享特征或梯度不稳定，效果可能受影响。

---

## 738. Nacrith: Neural Lossless Compression via Ensemble Context Modeling and High-Precision CDF Coding

**arXiv ID:** 2602.19626 | [PDF](https://arxiv.org/pdf/2602.19626v1)

**作者:** Roberto Tacconelli `[一作]` `[通讯]` (Independent Researcher), Roberto Tacconelli (Independent Researcher)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于SmolLM2-135M语言模型的无损压缩系统，结合轻量级在线预测器、精细CDF量化和32位算术编码，支持文本与任意二进制文件的混合压缩。

**💡 创新点**

创新点包括：①将CDF精度从2^16提升到2^24，显著降低量化误差；②引入N-gram与自适应log偏置混合预测器；③基于置信度跳过LLM推理；④实现原生KV缓存滑动窗口和多GPU并行压缩；⑤首次在LLM压缩框架中实现二进制文件压缩格式NC06。

**🔧 技术方法**

技术实现包括SmolLM2-135M预训练Transformer、llama.cpp GPU推理、32位算术编码、N-gram模型、指数权重自适应混合器、学习率为0.001的自适应偏置层、置信度阈值1.5bits的LLM跳过、原生KV滑动窗口、NC05/NC06容器格式、并行多GPU工作线程。

**📊 数据集**

主要使用的基准数据集有：Canterbury Corpus（alice29.txt、152KB）、enwik8（100MB Wikipedia摘录）、asyoulik.txt（莎士比亚文本）以及后训练期的UK政府技术报告文本，用以评估OOS性能。

**📈 对比分析**

与传统压缩器（gzip、xz、bzip2、zstd、brotli）以及其他LLM压缩器（CMIX、ts_zip、FineZip、NNCP、PAQ8px）对比，系统在alice29.txt压缩率0.918 bits/byte（11.5%），在enwik8压缩率0.9389 bits/byte（11.74%），均明显优于所有对比系统，尤其在OOS文本上0.723 bits/byte（9.0%）。

**⚠️ 局限性**

主要局限包括：压缩吞吐量低（约21–30 tokens/s，需多GPU提升）；模型文件约500MB，需在压缩与解压端共享；受限于2,048-token上下文窗口，对长文本可能失效；主要针对英文，低资源语言性能待验证；可能受训练数据记忆影响，需要更广泛的OOS验证。

---

## 739. Variational Inference for Bayesian MIDAS Regression

**arXiv ID:** 2602.19610 | [PDF](https://arxiv.org/pdf/2602.19610v1)

**作者:** Luigi Simeone `[一作]` `[通讯]` (Independent Researcher), Luigi Simeone (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一套针对贝叶斯 MIDAS 回归的 Coordinate Ascent Variational Inference (CAVI) 算法，利用线性权重参数化和条件共轭实现所有参数块的闭式更新。

**💡 创新点**

首次为 MIDAS 模型设计变分推断，解决 bilinear 结构导致 HMC 失效的问题，并通过 null‑space 重新参数化保证正则化；CAVI 在保持后验均值精度的同时显著提升计算速度。

**🔧 技术方法**

采用 CAVI（坐标上升变分）、线性 Almon 多项式/ B‑spline 权重参数化、null‑space 重参数、ELBO 解析优化、Monte Carlo 评估、与 ADVI 与 Gibbs 采样对比。

**📊 数据集**

使用 21 种仿真配置（最多 50 个预测变量）以及真实数据：2000‑2025 年的 S&P 500 每日收益的月度实现波动率（Realized Volatility）进行实证。

**📈 对比分析**

与块 Gibbs 采样比较，CAVI 的后验均值误差 ≤0.03，速度提升 107×–1,772×；与 ADVI 对比，偏差 7–14 倍大、速度慢 2,000–100,000 倍；权重参数 95% 置信区间覆盖率 >92%，影响系数置信区间出现均值场下方的覆盖率下降（从 89% 到 55%）。

**⚠️ 局限性**

均值场近似导致 β 置信区间显著 under‑dispersion，尤其在高维（J≥5）时覆盖率仅 55%；需要结构化变分或后验校准来改善高维情况下的不确定性估计；对时间变参数、全局局部收缩或随机波动性等扩展仍需进一步研究。

---

## 740. How to Train Your Deep Research Agent? Prompt, Reward, and Policy Optimization in Search-R1

**arXiv ID:** 2602.19526 | [PDF](https://arxiv.org/pdf/2602.19526v1)

**作者:** Yinuo Xu `[一作]` (National Laboratory of Pattern Recognition and Machine Intelligence, Chinese Academy of Sciences), Jian Liang `[通讯]` (National Laboratory of Pattern Recognition and Machine Intelligence, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Deep Research代理中的RL训练，系统分析了提示模板、奖励函数和策略优化对性能的影响，并基于此提出了改进的Search-R1++基线。

**💡 创新点**

系统分离三维度进行实验，发现Fast Thinking模板、F1+奖励和REINFORCE优化能显著提升稳定性和准确率，并基于此构建Search-R1++。

**🔧 技术方法**

使用强化学习算法PPO、GRPO、REINFORCE；引入Fast Thinking提示模板；使用F1+奖励；采用Qwen2.5-3B/7B LLM与E5检索器。

**📊 数据集**

NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、Musique、Bamboogle等七个公开问答基准。

**📈 对比分析**

与Search-R1、R1-base、ReAct等基线进行对比；在Qwen2.5-7B上平均提升约3.9%，在Qwen2.5-3B上提升约4.2%；在多任务上表现优于EM奖励方案。

**⚠️ 局限性**

仅在预定义提示与奖励结构下验证，缺乏对更大模型或不同检索器的泛化；REINFORCE在探索上受限于小模型；未探究更复杂奖励或自监督机制。

---

## 741. A Markovian View of Iterative-Feedback Loops in Image Generative Models: Neural Resonance and Model Collapse

**arXiv ID:** 2602.19033 | [PDF](https://arxiv.org/pdf/2602.19033v1)

**作者:** Vibhas Kumar Vats `[一作]` (Indiana University), Samuel Goree `[通讯]` (Stonehill College)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5062429758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究生成式 AI 模型在迭代反馈循环中的长期行为，提出“神经共振”现象并给出条件与诊断方法；

**💡 创新点**

首次将迭代反馈建模为马尔可夫链，发现只有在满足可遍历性和方向性收缩时才出现低维不变结构，并建立八种维度模式分类；

**🔧 技术方法**

使用马尔可夫链理论、Fréchet Inception Distance 评估漂移、隐空间维度指标（参与度比、Levina‑Bickel 本征维度）以及 diffusion、CycleGAN、音频线性卷积等生成模型；

**📊 数据集**

实验数据集包括 MNIST、ImageNet‑5、Horse‑Zebra 图像对以及物理空间的冲击响应；

**📈 对比分析**

通过对比可遍历与不可遍历链，计算局部与累计 FID，绘制隐空间维度曲线，显示可遍历链最终趋于稳态；性能指标表现为低漂移与低维收缩；

**⚠️ 局限性**

局限性包括对真实数据多样性不足的解释仍不完整、实验规模有限、对非可遍历系统的理论解释不足、以及对多模态/更大规模数据的推广性待验证。

---

## 742. Infinite-Dimensional Closed-Loop Inverse Kinematics for Soft Robots via Neural Operators

**arXiv ID:** 2602.18655 | [PDF](https://arxiv.org/pdf/2602.18655v1)

**作者:** Carina Veil `[一作]` (Stanford University), Cosimo Della Santina `[通讯]` (Delft University of Technology)

**通讯引用:** 4163 | [OpenAlex ID](https://openalex.org/A5050239145)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将闭环逆运动学（CLIK）扩展到软体机器人的无限维形状空间，使用可微分神经算子学习驱动-形状映射，并在三纤维软机器人上实现定位任务的仿真。

**💡 创新点**

通过将无限维链式法则与CLI K结合，得到完整软体形状到任务空间的Jacobian，并利用神经算子（DeepONet）逼近无闭式的驱动-形状映射，从而实现整个软体形状的连续控制。

**🔧 技术方法**

神经算子网络（DeepONet）、自动微分、无限维链式法则、Jacobian‑based CLIK、三纤维主动弹性细丝模型。

**📊 数据集**

约1,000,000条基于三纤维主动弹性细丝模型的仿真样本，激活范围[-1.67,0]，每条样本含100个空间点的中心线坐标。

**📈 对比分析**

与解析常曲线段CLI K示例比较，展示指数收敛；在三纤维软机器人上进行固定点和最近点定位任务，仿真误差平均MSE为1.38×10⁻¹⁰，L2相对误差为6.08×10⁻⁴，控制权重K=8在1秒内收敛。

**⚠️ 局限性**

仅在仿真中验证，未进行实验验证；需要大量仿真数据，可能对真实硬件的模型不匹配敏感；未验证对障碍物或轨迹跟踪的适应性；算子训练与推理的计算开销相对较高。

---

## 743. From Bias Mitigation to Bias Negotiation: Governing Identity and Sociocultural Reasoning in Generative AI

**arXiv ID:** 2602.18459 | [PDF](https://arxiv.org/pdf/2602.18459v1)

**作者:** Zackary Okun Dunivin `[一作]` (University of Stuttgart), John Bollenbocher `[通讯]` (RTI International)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实证验证“偏见协商”治理范式，探究大语言模型在身份与社会文化推理中的互动方式；

**💡 创新点**

创新点在于：①将偏见治理从单纯的检测与抑制扩展为“协商”，强调身份认知的上下文化与可修正性；②构建偏见协商的决策政策与行动空间框架；③提出基于过程敏感评估与可操作路径的评估与实现手册；

**🔧 技术方法**

使用大型预训练语言模型（GPT‑4o、Claude 3、DeepThink‑R1、LLaMA 3.1）结合提示工程；进行最小化引导的半结构化访谈与主题分析；以及在情感与风格提示下的文本分类实验；

**📊 数据集**

数据集包括：①从 Google Places 采集的 1‑星负面医师评论（约 154k 条，筛选后 1,690 男性 + 2,921 女性）；②多款公开聊天机器人在访谈中产生的对话记录；

**📈 对比分析**

比较方法：在情感/风格提示与无提示的对照实验中，GPT‑4o 的性别识别准确率从 0.52‑0.53 提升至 0.62，进一步到 0.70（基于情感/风格阐释）。访谈部分采用主题分析评估模型的协商行为，未给出数值性能；整体表明模型能在身份推理上实现一定程度的协商，但仍不稳定；

**⚠️ 局限性**

局限性：①偏见协商的评估主要依赖定性主题分析，缺乏统一的量化基准；②实验样本仅覆盖负面医师评论与部分公开聊天机器人，难以推广到更广泛场景；③模型在高不确定性或多重身份冲突时仍倾向“公平话语”或回避决策，显示程序性不稳定；④潜在的 WEIRD 文化偏向未得到充分消除；

---

## 744. Breaking the Barriers of Database-Agnostic Transactions

**arXiv ID:** 2602.19440 | [PDF](https://arxiv.org/pdf/2602.19440v1)

**作者:** Toshihiro Suzuki `[一作]` (Scalar, Inc.), Hiroyuki Yamada `[通讯]` (Scalar, Inc.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

通过引入原子性单元（Atomicity Unit）并实现两种技术（AUP和AUD），提升了数据库无关的联邦事务管理效率，并在ScalarDB中实现和评估。

**💡 创新点**

发现并利用数据库的原子性作用域来推送操作（AUP）和分离事务元数据（AUD），在保持事务无关性的同时显著降低事务开销。

**🔧 技术方法**

采用数据库抽象层与适配器模式，原子写入、线性可见条件写、两阶段提交、乐观并发控制，以及视图连接与一致可读性特性。

**📊 数据集**

使用YCSB基准工作负载（工作负载F和C）在PostgreSQL实例上进行实验，加载100万条记录。

**📈 对比分析**

与未使用AU的ScalarDB以及直接使用JDBC的理想情况对比；AUP在多操作事务时接近理想吞吐量，AUD在元数据解耦后仅增加约30%的开销。

**⚠️ 局限性**

需要数据库适配器声明AU并支持原子写入；协调器写入仍是瓶颈，且在高并发时性能下降；仅在支持一致可读性与视图连接的数据库上才能获得最佳效果。

---

## 745. Transforming Science Learning Materials in the Era of Artificial Intelligence

**arXiv ID:** 2602.18470 | [PDF](https://arxiv.org/pdf/2602.18470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 746. DECAF: Dynamic Envelope Context-Aware Fusion for Speech-Envelope Reconstruction from EEG

**arXiv ID:** 2602.19395 | [PDF](https://arxiv.org/pdf/2602.19395v1)

**作者:** Karan Thakkar `[一作]` (Johns Hopkins University), Mounya Elhilali `[通讯]` (Johns Hopkins University)

**通讯引用:** 6028 | [OpenAlex ID](https://openalex.org/A5038788686)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于状态空间融合的动态EEG到语音包络重构模型DECAF，利用EEG直接估计与自回归时间先验相结合，实现递归、因果的包络重构。

**💡 创新点**

创新点在于把包络重构视为迭代状态估计任务，构建可学习的门控融合器动态平衡神经证据与时间预测，并实现全因果递归预测，显著提升重构质量。

**🔧 技术方法**

采用深度CNN+GRU+多头注意力的包络预测器、三层1D卷积门控融合模块、混合L1与Pearson相关的联合损失，并借鉴Kalman滤波思想实现状态空间框架。

**📊 数据集**

使用ICASA 2023 Auditory EEG Decoding Challenge Task 2数据集，包含85位受试者、64通道EEG和叙事故事音频，按官方划分进行训练、验证和测试。

**📈 对比分析**

与线性mTRF、VLAAI、HappyQuokka等基线对比，DECAF在测试集上的平均Pearson相关系数为0.170 ± 0.061，较线性基线提升约60%，优于HappyQuokka（0.162），达成SOTA。

**⚠️ 局限性**

在高噪声（SNR ≤ –10 dB）下性能衰退，依赖足够的EEG信噪比；递归自预测可能累计误差；尚未在更自然连续听觉场景中验证鲁棒性。

---

## 747. SiGRRW: A Single-Watermark Robust Reversible Watermarking Framework with Guiding Strategy

**arXiv ID:** 2602.19097 | [PDF](https://arxiv.org/pdf/2602.19097v1)

**作者:** Zikai Xu `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23251 | [OpenAlex ID](https://openalex.org/A5064573190)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单水印鲁棒可逆水印框架 SiGRRW，能够在不降低可逆性的前提下实现高容量、高鲁棒性与低失真。

**💡 创新点**

创新点：① 引入引导策略（Guider），通过 Gnet 与 Hnet 双阶段生成完全相同的引导图像，从而实现残差导向的可逆嵌入；② 在训练阶段加入简化噪声层（JPEG + 高斯模糊），显著提升对常见失真与重生成攻击的鲁棒性；③ 通过联合训练实现单水印同时兼顾鲁棒性与可逆性，打破传统两阶段 RRW 的性能瓶颈。

**🔧 技术方法**

使用深度学习技术：UNet 结构的 Hnet 与 Gnet、CEILNet 的 Enet、PatchGAN 的 Dnet；损失函数结合 L2、感知、对抗、提取误差与噪声误差；Guider 的双子网络实现精确引导图像；Noise 层实现抗攻击训练。

**📊 数据集**

数据集：PASCAL VOC 与 LAION‑Aesthetics（训练 4000 张，验证 200 张），包含自然图像和模型生成图像；水印采用 256×256 二值图像。

**📈 对比分析**

对比方法包括传统与深度学习 RRW、可逆与鲁棒水印基线（CIN、MuST、DMIPP、IWRN、RRW‑FoZM、RRW‑PZMs、DRRW、RRWID 等）。实验表明 SiGRRW 在 PSNR（44.25 dB）、SSIM（0.9923）、容量（256² bits）等指标上优于所有基线，并在多种失真（高斯噪声、模糊、盐椒、JPEG、尺度、裁剪、Dropout）下保持 99%+ ACC，甚至在 VAE 重生成攻击中显著优于 RRWID。

**⚠️ 局限性**

局限性：① 在极端失真条件下可逆性会受到影响，需要权衡噪声层强度；② 训练和部署需要较大算力（GPU 4090），模型尺寸相对较大；③ 目前主要评估在 256×256 图像，需验证在更大尺寸或视频等场景的可扩展性。

---

## 748. Physiologically Informed Deep Learning: A Multi-Scale Framework for Next-Generation PBPK Modeling

**arXiv ID:** 2602.18472 | [PDF](https://arxiv.org/pdf/2602.18472v1)

**作者:** Shunqi Liu `[一作]` (University of Southern California), Tong Wang `[通讯]` (University of Connecticut)

**通讯引用:** 15325 | [OpenAlex ID](https://openalex.org/A5100344502)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种融合Transformer、受限扩散模型与混合GNN+Neural ODE的 PBPK 建模框架，用于药物动力学预测、虚拟人群生成和跨物种推断。

**💡 创新点**

创新点在于：① 将生理先验直接嵌入Transformer的序列预测；② 通过物理约束的扩散模型生成符合生理约束的虚拟人群；③ 用混合GNN+Neural ODE实现跨物种的连续尺度学习，自动学习物种嵌入。

**🔧 技术方法**

采用的技术包括 Transformer（多头自注意力）、DDPM（带物理损失的受限扩散模型）、Neural ODE 与 GNN 的混合架构、Physics-Informed Loss、金字塔式训练与归一化技术。

**📊 数据集**

使用的三套合成数据集：① 1000个虚拟患者的两室 ODE PK 数据；② 2000个含年龄、身高、体重、肝心体积等生理参数向量；③ 50种药物在 Rat、Dog、Human 三物种上的 PK 模拟，用于跨物种验证。

**📈 对比分析**

性能对比：Transformer 在预测任务上 MSE 7.80，显著捕捉多相衰减；无约束扩散模型违例率 2%，加入物理约束后降至 0.5%；Neural Allometry 在留一物种验证中人类 MSE 0.506，展示零样本跨物种推断能力。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证，缺乏真实临床数据；扩散模型仍需手动调节物理约束权重；跨物种模型依赖预设的所有尺度规律，可能在复杂生理差异下失效；未与完整 PBPK 体系集成验证。

---

## 749. ALPACA: A Reinforcement Learning Environment for Medication Repurposing and Treatment Optimization in Alzheimer's Disease

**arXiv ID:** 2602.19298 | [PDF](https://arxiv.org/pdf/2602.19298v1)

**作者:** Nolan Brady `[一作]` (University of Colorado Boulder), Tom Yeh `[通讯]` (University of Colorado Boulder)

**通讯引用:** 4629 | [OpenAlex ID](https://openalex.org/A5070687718)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建了ALPACA，一个基于Gym兼容的强化学习环境，用来模拟阿尔茨海默病（AD）患者的个性化、序列化治疗决策，并通过该环境评估多种RL策略。

**💡 创新点**

创新点包括：①使用药物条件化的自回归混合专家Transformer（CAST）对ADNI的纵向轨迹进行预测，实现高保真度的药物响应模拟；②将模拟环境包装成Gym接口，支持持续的临床状态与多二进制药物行动；③在此环境中对多种RL算法（PPO、A2C、SAC、BDQ）进行基准测试，并结合SHAP解释学习策略的临床意义。

**🔧 技术方法**

技术手段：混合专家Transformer（CAST）用于状态转移预测；Stable Baselines3实现的PPO、A2C、SAC以及自定义的BDQ；SHAP进行模型可解释性分析；Gaussian Mixture Model生成起始状态；额外使用ExtraTrees、Dropout、L2正则化等训练细节。

**📊 数据集**

使用的公开数据集是阿尔茨海默病神经成像倡议（ADNI）的纵向临床与影像数据，约1900名受试者、12,984次访问，包含21个连续变量、17个药物类别及时间间隔信息。

**📈 对比分析**

比较方法：将RL策略与无药物基线、行为克隆的临床医生策略、以及基于临床规则的启发式策略进行同一组模拟患者的对比。评估指标为累计奖励、每步奖励、最终ADNI-Mem分数。结果显示，RL策略显著优于无药物和行为克隆基线，其中PPO表现最佳；SHAP分析表明策略依赖临床相关特征。

**⚠️ 局限性**

局限性：药物行动采用离散多二进制形式，缺乏连续剂量调节；时间步固定为6个月，未充分体现真实随访不规则性；患者表征相对粗略，缺少多模态或更丰富的临床变量；模拟本身不能直接证明因果关系，需要进一步与实际临床验证结合。

---

## 750. Hyperbolic Busemann Neural Networks

**arXiv ID:** 2602.18858 | [PDF](https://arxiv.org/pdf/2602.18858v1)

**作者:** Ziheng Chen `[一作]` (University of Trento), Nicu Sebe `[通讯]` (University of Trento)

**通讯引用:** 35911 | [OpenAlex ID](https://openalex.org/A5027171279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在双曲空间中设计并实现了两种新型神经网络组件——Busemann 多项式逻辑回归（BMLR）和 Busemann 全连接（BFC）层，并将其统一到 Poincaré 球模型与 Lorentz 模型。

**💡 创新点**

创新点在于利用 Busemann 函数及其等值面（horosphere）构造本质上是双曲几何的、参数紧凑且批量高效的 MLR 与 FC 层，能够在两种主流双曲模型上无缝工作，并在零曲率极限下回归欧氏网络。

**🔧 技术方法**

核心技术包括：Busemann 函数的解析表达、点到等值面的距离解释、双曲欧氏映射（Möbius、Lorentz）、张量化批处理、以及与现有欧氏/双曲网络的参数对齐与转换。

**📊 数据集**

使用的数据集涵盖图像分类（CIFAR‑10、CIFAR‑100、Tiny‑ImageNet、ImageNet‑1k）、基因组序列学习（TEB、GUE）、节点分类（Disease、Airport、PubMed、Cora）以及链路预测（同四个图数据集）。

**📈 对比分析**

与现有的双曲 MLR（PMLR、PBMLR‑P、LMLR）和 FC（Möbius、Poincaré FC、Lorentz FC、LTFC）进行对比，实验显示 BMLR 在多类别任务中显著提升准确率和 AUC，BFC 在链路预测任务中取得最高 AUC；同时 BMLR‑L 与 BFC 在训练时间与参数规模上均优于或与最优方法相当。

**⚠️ 局限性**

限制与不足主要包括：1）实验集中在四个主流双曲模型和固定曲率值，未对不同曲率进行系统分析；2）对非常大规模图或非双曲度较低的数据集的可扩展性和鲁棒性未作深入探讨；3）实现中仍依赖数值稳定的双曲运算，可能在极端深度或高维度下产生数值不稳定。

---

## 751. Morphological Addressing of Identity Basins in Text-to-Image Diffusion Models

**arXiv ID:** 2602.18533 | [PDF](https://arxiv.org/pdf/2602.18533v1)

**作者:** Andrew Fraser `[一作]` `[通讯]`, Andrew Fraser

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了形态学压力在文本到图像扩散模型中的作用，证明通过形态特征描述符或音素结构可在潜在空间中导航，达到对已记忆身份的检索与构造新视觉实体的目的。

**💡 创新点**

创新点在于提出形态学描述符可在训练阶段通过 LoRA 自蒸馏形成坐标系统，既能精准定位已记忆身份，又能在反向条件下产生“诡异”结构；并在提示层面利用音素结构（phonestheme）生成新颖一致的视觉概念，揭示潜在空间可被形态学梯度系统化。

**🔧 技术方法**

核心技术包括Stable Diffusion 1.5、LoRA 参数高效微调、自蒸馏循环、负提示与正提示的 push‑pull 组合、ArcFace 识别相似度评估、CLIP ViT‑L/14 文本编码器、Purity@1 评估指标。

**📊 数据集**

使用的数据集为 LAION‑5B（Stable Diffusion 1.5 训练集），并在实验中随机生成 200 个音素结构词、100 个随机可发音词、50 个无结构词，以及 4 个已知词作对照。

**📈 对比分析**

与随机/可发音对照相比，音素结构词的 Purity@1 平均值从 0.209 提升至 0.372，显著差异（p<0.00001，Cohen’s d≈0.55）。在身份导航实验中，LoRA 训练后平均相似度从 0.245 提升至 0.465，且在推拉组合条件下从“ eldritch ”转为“ uncanny valley ”，展示结构化失败模式。

**⚠️ 局限性**

局限包括仅在 Stable Diffusion 1.5 上验证，未探测其他模型；仅测试 Marilyn Monroe，未评估对其他身份的适用性；音素库仅基于英语，缺乏跨语言验证；Purity@1 仅衡量 CLIP 嵌入相似度，缺乏人类主观评估；训练数据污染检查不完备，可能存在未知实例；实验规模相对有限，未充分探索正向/负向参数空间。

---

## 752. High Dimensional Procedural Content Generation

**arXiv ID:** 2602.18943 | [PDF](https://arxiv.org/pdf/2602.18943v1)

**作者:** Kaijie Xu `[一作]` (McGill University), Clark Verbrugge `[通讯]` (McGill University)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5001343030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了将游戏机制维度（层级、时间）作为高维空间坐标的PCG框架，并在二维/三维网格上实现层级切换与时间同步的关卡生成。

**💡 创新点**

创新点在于把非几何玩法维度直接嵌入状态空间，形成高维扩展图（DEG），并在统一搜索框架下实现可控的层级与时间交互关卡。

**🔧 技术方法**

采用潜在图搜索（A*、DP）、时间展开图、势场规划、遗传算法等技术构建四阶段管线：骨架规划、属性实例化、验证与评估。

**📊 数据集**

使用自定义的二维/三维网格生成实验，覆盖小/中/大规模，未使用公开数据集。

**📈 对比分析**

与三种基线（随机噪声、静态骨架、简化A*）相比，PF‑A*在大规模下获得最高综合分数，TEG‑DP在时间方向上表现最佳；GA优化提升质量但耗时增加，整体性能满足可控性与可验证性要求。

**⚠️ 局限性**

局限包括时间方向的A*状态爆炸、缺乏ML/强化学习支持、未实现多机制组合、鲁棒性未纳入GA目标、缺少用户实验等。

---

## 753. Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models

**arXiv ID:** 2602.19111 | [PDF](https://arxiv.org/pdf/2602.19111v1)

**作者:** Kainan Liu `[一作]` (Ping An Technology), Jing Xiao `[通讯]` (Ping An Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Astra 方法，在 LoRA 的低秩更新中利用输出激活的尾部特征子空间进行初始化，以实现参数高效微调。

**💡 创新点**

创新点在于将小特征值对应的激活子空间作为适配方向，使模型在保持预训练知识的同时显著提升表示能力和收敛速度。

**🔧 技术方法**

使用校准数据集计算激活协方差矩阵的特征值分解，投影至尾部特征子空间并构造 LoRA 的 A、B 矩阵；结合低秩重参数化与传统 LoRA 训练流程。

**📊 数据集**

实验覆盖 NLU（GLUE 5 任务）、数学推理（GSM8K、MATH）、代码生成（HumanEval、MBPP）、常识推理（BoolQ、PIQA、HellaSwag 等）以及 MetaMathQA、Commonsense170K 等校准/训练数据集。

**📈 对比分析**

通过与全微调（FFT）、Vanilla LoRA 及 6 类 LoRA 变体（PiSSA、MiLoRA、CorDA、LoRA-GA、rsLoRA、DoRA）在 16 个基准上进行对比，Astra 在多数任务上取得最高平均准确率，收敛更快，甚至在部分任务上超过 FFT。

**⚠️ 局限性**

限制主要在于仅在 7B–8B 规模模型验证，未检验在 32B+ 或 72B+ 大模型的表现；使用固定 LoRA 秩，未探索与动态秩或结构化 LoRA 方案的协同效应。

---

## 754. RDBLearn: Simple In-Context Prediction Over Relational Databases

**arXiv ID:** 2602.18495 | [PDF](https://arxiv.org/pdf/2602.18495v1)

**作者:** Yanlin Zhang `[一作]` (University of Hong Kong), Minjie Wang `[通讯]` (University of Hong Kong)

**通讯引用:** 1963 | [OpenAlex ID](https://openalex.org/A5108050002)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现RDBLearn框架，利用关系数据库的聚合特征化与单表ICL模型完成预测任务。

**💡 创新点**

创新点在于用一种简单的两步式方案（关系特征化+单表ICL）替代复杂的图编码器，展示其在多种基准上同样强大的性能。

**🔧 技术方法**

技术上采用基于SQL的递归聚合特征化（DFS）与TabPFN、LimiX等tabular ICL后端；实现了与scikit‑learn相似的估计器接口。

**📊 数据集**

在RelBench（包含8个二分类和8个回归任务）以及4DBInfer（5个实体级分类任务）两个关系数据库基准上进行实验。

**📈 对比分析**

与RT、Griffin、KumoRFM、AutoGluon+DFS等关系基础模型以及语言模型基线进行对比；RDBLearn在AUC、MAE等指标上往往位居基础模型之首，接近甚至超过部分监督图神经网络和Fine‑Tuned Griffin。

**⚠️ 局限性**

局限在于特征化仍是手工或外部流程，缺乏模型内置的特征选择；仅针对实体级分类/回归，未扩展到链接预测等更广泛的关系任务。

---

## 755. RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing

**arXiv ID:** 2602.19753 | [PDF](https://arxiv.org/pdf/2602.19753v1)

**作者:** Kaifa Yang `[一作]` (Shanghai Jiao Tong University), Zhu Li `[通讯]` (University of Missouri Kansas City)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了RAP，一种快速、渲染无关的属性引导方法，用于预测3D高斯光栅化（3DGS）中每个高斯原语的重要性分数。

**💡 创新点**

创新点在于：①直接利用原始高斯属性与局部邻域统计构建15维特征；②用轻量级MLP结合三种损失（渲染损失、剪枝感知损失、分布正则化）进行学习；③实现无视角、无场景特定训练的通用性与高效推理。

**🔧 技术方法**

技术包括：属性提取（距离、颜色异质性、尺度、体积、透明度、DC颜色等），全局与局部标准化，MLP前馈网络，渲染损失（基于差分光栅化），剪枝均值正则，熵正则化等。

**📊 数据集**

在DL3DV-10K、Mip-NeRF360（室外/室内）、Deep Blending、Tanks & Temples等公开数据集上进行训练与评估。

**📈 对比分析**

与基线（简单透明度阈值、LightGaussian、MesonGS、EAGLES、C3DGS、PUP-3DGS）相比，RAP在保留率、PSNR、SSIM、LPIPS以及BD-Rate上均表现最优，尤其在高剪枝率下仍保持2 dB以内的质量损失，计算时间也仅次于透明度基线。

**⚠️ 局限性**

局限在于：目前仅采用固定全局剪枝比例，未针对不同区域自适应分配采样或层次化编码；对极端稀疏或大规模场景的扩展性与与压缩编码的更深层耦合仍需进一步研究。

---

## 756. Who Has the Final Word? Designing Multi-Agent Collaborative Framework for Professional Translators

**arXiv ID:** 2602.19016 | [PDF](https://arxiv.org/pdf/2602.19016v1)

**作者:** George X. Wang `[一作]` (New York University), Jing Qian `[通讯]` (New York University)

**通讯引用:** 5464 | [OpenAlex ID](https://openalex.org/A5001637232)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 CHORUS——一种基于多维质量指标(MQM)的多智能体协作翻译框架，支持专业翻译者在多维质量维度上进行迭代式决策。

**💡 创新点**

创新点在于：①将 MQM 的质量维度拆解为专门化的 AI 代理；②引入维度路由器根据翻译者目标动态选择相关代理；③形成由翻译者控制的迭代反馈循环，显著降低翻译者的认知负担并提高可解释性。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑4o、Gemini 2.0 Flash、Claude 3 Haiku）作为底层生成器；多智能体协作架构（维度路由器、专门代理、共享翻译记忆）；基于 MQM 的对话式质量评估与提示机制。

**📊 数据集**

使用公开的 WMT 2023/2024 句子级数据集，涵盖 6 种语言对（EN↔DE、EN↔HE、EN↔JA、EN↔RU、EN↔ZH、EN↔AR），每方向约 200 条样本。

**📈 对比分析**

通过与零射击翻译和单智能体自我校正的三种条件比较，使用 BLEU、METEOR、BLEURT、COMET 等指标评估。结果显示 CHORUS 在大多数指标上显著优于基线，尤其在 BLEURT/COMET 上提升 5–12%（最高 21.7%）且 BLEU 提升可达 +10.6，差异在多数方向上统计显著。

**⚠️ 局限性**

局限性：评估仅在无人工参与的“代理‑only”模式下完成，未验证真实翻译者交互效果；缺乏对多轮人机协作、时间成本、资源消耗及跨域适应性的深入分析。

---

## 757. Compositional Planning with Jumpy World Models

**arXiv ID:** 2602.19634 | [PDF](https://arxiv.org/pdf/2602.19634v1)

**作者:** Jesse Farebrother `[一作]` (Meta), Ahmed Touati `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种基于跳跃世界模型（Geometric Horizon Model）的规划框架，能够直接在预训练的参数化策略上进行组合，解决长时程决策难题。

**💡 创新点**

创新点在于：①将“跳跃”多步动力学模型与策略条件化结合；②提出跨时间尺度的一致性（horizon‑consistency）损失，提升长时程预测准确性；③通过几何切换策略实现可插拔的策略组合，统一了动作级控制、GPI 与层次化规划。

**🔧 技术方法**

核心技术包括：基于流匹配的ODE生成模型（flow‑matching world model）、Temporal Difference Flow + consistency loss、几何切换策略的 successor‑measure 解析、随机射击（random shooting）规划和 FiLM 条件化网络。

**📊 数据集**

使用 OGBench 基准，包含 antmaze（medium/large/giant）和 multi‑cube 机器人操作任务的离线数据集。

**📈 对比分析**

与零射击策略、动作级规划、Generalized Policy Improvement、HIQL、SHARSA 等方法对比，CompPlan 在长时程任务中平均提升约 200%（相较动作级规划）和 90%（相较 GPI），在 antmaze‑giant 与 cube‑4 上分别实现 89% 与 67% 的成功率，明显优于现有层次化和规划方案。

**⚠️ 局限性**

局限性包括：一致性损失对中等时程规划提升有限；模型训练依赖大量离线数据；对非常长时程或更复杂任务的泛化仍需进一步验证；在实时或低延迟场景下采样与推理成本较高。

---

## 758. PuppetChat: Fostering Intimate Communication through Bidirectional Actions and Micronarratives

**arXiv ID:** 2602.19463 | [PDF](https://arxiv.org/pdf/2602.19463v1)

**作者:** Emma Jiren Wang `[一作]` (Virginia Tech), Zhicong Lu `[通讯]` (George Mason University)

**通讯引用:** 34236 | [OpenAlex ID](https://openalex.org/A5063218435)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 PuppetChat，一个面向亲密关系的双人聊天原型，结合可互补的角色动作与可编辑的微故事来促进情感表达与持续互动。

**💡 创新点**

创新点在于：①通过动作推荐系统实现“互补对话”，将单向表情转化为双向共演；②引入短文本微故事作为语境补全，使动作更具个人化与叙事深度；③允许用户编辑、选择是否持久化，并可即时回放动作，构建共享记忆。

**🔧 技术方法**

技术方案包括：1）基于 Azure OpenAI 的 LLM 用于动作匹配与微故事生成；2）向量相似度检索与情感/互动角色评分的混合评分模型；3）WebSocket 实时通信；4）Vue3+Tailwind 前端与 Node.js+Express+MongoDB 后端。

**📊 数据集**

数据集：①共 42 条精心策划的动作（从 215 条 GIF 中筛选、标注情感与互动角色）；②收集的 11 对情侣/朋友对话日志（约 10 天、22 份数据）；③用户自填的个人故事与标签。

**📈 对比分析**

通过 10 天现场实验与 22 名参与者的问卷（SUS、NASA‑TLX、16 项自定义量表）评估。SUS 平均 86.6，表现为优秀可用性；RTLX 31.7，工作负荷低。定性访谈显示用户对互补动作与微故事的认可度高，尤其在社交存在与持续性方面得分 ≥5.3。

**⚠️ 局限性**

局限性包括：①样本规模小、仅限大学生与熟悉伴侣/好友，缺乏不同关系阶段与文化背景的代表性；②实验周期仅 10 天，无法评估长期影响；③未与传统表情包/单向回执的基线系统做对比，无法量化各功能的单独贡献；④微故事生成质量与用户历史关联性仍有限。

---

## 759. Programmable Property-Based Testing

**arXiv ID:** 2602.18545 | [PDF](https://arxiv.org/pdf/2602.18545v1)

**作者:** Alperen Keles `[一作]` (University of Maryland), Leonidas Lampropoulos `[通讯]` (University of Maryland)

**通讯引用:** 471 | [OpenAlex ID](https://openalex.org/A5075217645)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的属性语言实现方式——延迟绑定抽象语法（DBAS），将属性重新表述为可操作的数据结构，使其与属性运行器解耦，支持可编程的测试循环；

**💡 创新点**

创新点在于通过DBAS实现深层嵌入的属性DSL，使用户可以在不修改框架内部代码的情况下自定义和实验不同的属性运行器（传统、集成收缩、覆盖导向、定向、组合、并行等）；

**🔧 技术方法**

技术手段包括：在Rocq中利用依赖类型实现上下文推导与模式匹配；在Racket中利用动态类型与宏实现可扩展注解；对属性进行抽象化后实现通用的生成、检查、收缩、打印等组件；

**📊 数据集**

使用ETNA基准数据集（Binary Search Tree、Red‑Black Tree、Simply‑Typed Lambda Calculus、System F、IFC）进行性能与功能评测；

**📈 对比分析**

对比方法：将DBAS实现的PBT框架与QuickCheck/QuickChick/Hypothesis等现有浅层嵌入框架在相同工作负载下进行基准；结果显示DBAS实现几乎无性能损失，甚至在某些任务上优于传统框架；

**⚠️ 局限性**

局限性包括：目前实现仅覆盖Rocq和Racket两种语言，Haskell等主流静态语言缺乏相应支持；在Rocq中对上下文实现复杂，Racket中缺乏类型安全；未在大规模工业项目中验证，需进一步扩展到更多语言与更复杂的属性场景。

---

## 760. Complex Event Processing in the Edge: A Combined Optimization Approach for Data and Code Placement

**arXiv ID:** 2602.19338 | [PDF](https://arxiv.org/pdf/2602.19338v1)

**作者:** Halit Uyanık `[一作]` (Istanbul Technical University), Tolga Ovatman `[通讯]` (Istanbul Technical University)

**通讯引用:** 404 | [OpenAlex ID](https://openalex.org/A5004929878)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了基于受限规划的关键路径优化方法，用于在边缘 IoT 设备上智能分配 CEP 代码与数据，以提升整体吞吐量和降低延迟。

**💡 创新点**

创新点在于将 CEP 任务依赖图的关键路径与代码/数据放置同时建模为受限规划问题，并结合虚拟共享内存（VSM）实现跨设备实时迁移，同时提供易用的 Python 库，兼顾低功耗设备与多传感器协同。

**🔧 技术方法**

使用技术包括：Python、CP-SAT 受限规划求解器、MQTT+MongoDB（tmpfs 实现 VSM）、CEP DAG 建模、遗传算法对比实验、Raspberry Pi4 设备与 Docker 的实验环境。

**📊 数据集**

实验数据集为模拟智能车场景的 CEP 流程，包含 9 个传感器（距离传感器和摄像头），共 10 台 Raspberry Pi4（1 台管理器 + 9 台工作器），生成真实的距离与图像帧数据。

**📈 对比分析**

通过将 CP 与 4 种启发式方法（轮询、随机、局部性、遗传算法）进行对比，结果显示 CP（使用 1.25 的迁移惩罚）在吞吐量、平均延迟与最大延迟方面均优于其他方法；在 CPU 限制情境下，CP 仍保持相对稳健。

**⚠️ 局限性**

主要局限包括：只能保证每个事件单实例执行，无法并行化同一任务；假设管理器与消息队列无单点故障；代码迁移依赖下载而非线程级别迁移，安全与隐私考虑缺失；在大规模设备（>25 节点）时优化时间迅速膨胀。

---

## 761. LAPIS: Lightweight API Specification for Intelligent Systems

**arXiv ID:** 2602.18541 | [PDF](https://arxiv.org/pdf/2602.18541v1)

**作者:** Daniel Garcia `[一作]` `[通讯]` (Independent Researcher), Daniel Garcia (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LAPIS，一种针对LLM消费优化的轻量级API规范格式。

**💡 创新点**

通过结构重构（集中错误定义、类型扁平化、签名化操作、去除无关元数据等）显著压缩Token量。

**🔧 技术方法**

使用规则化转换器将OpenAPI 3.x自动转换为LAPIS，并采用可读性强的自定义语法。

**📊 数据集**

使用五个真实生产API（GitHub、DigitalOcean、Twilio、Petstore、HTTPBin）进行评估。

**📈 对比分析**

与OpenAPI YAML/JSON对比，平均Token减少85.5%（最小71.9%），且对不同tokenizer保持一致性。

**⚠️ 局限性**

缺少详细JSON Schema校验、示例等信息，未对LLM实用性能进行基准测试，转换器对复杂OpenAPI构造支持有限。

---

## 762. Revisiting the Seasonal Trend Decomposition for Enhanced Time Series Forecasting

**arXiv ID:** 2602.18465 | [PDF](https://arxiv.org/pdf/2602.18465v1)

**作者:** Sanjeev Panta `[一作]` (University of Louisiana at Lafayette), Nian-Feng Tzeng `[通讯]` (University of Louisiana at Lafayette)

**通讯引用:** 2750 | [OpenAlex ID](https://openalex.org/A5032851065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对时间序列进行趋势与季节分解，并在每个分量上分别使用去归一化的RevIN与多层感知机（MLP）模型，提出两种轻量级双MLP（RMSM、RMM）框架，显著提升多变量时间序列预测性能。

**💡 创新点**

创新点在于：①仅使用简单的移动平均分解来提取趋势和季节成分；②对季节成分去除归一化/缩放，仅用MLP直接预测；③构造双MLP架构实现与SOTA Transformer同等或更优的准确度，同时保持线性时间复杂度。

**🔧 技术方法**

技术方法包括：移动平均分解、Mixture of Experts、频域分解、RevIN、MLP、Shift‑MLP、双MLP（RMSM、RMM）、去归一化处理以及对现有Transformer/Linear骨干网络进行去标准化改造。

**📊 数据集**

实验使用四大基准数据集：Electricity、ETT（ETTm2、ETTh2）、Weather以及美国USGS河道站点的水文数据（Comite River）。

**📈 对比分析**

与iTransformer、PatchTST、TimesNet、DLinear等SOTA基线对比，本文模型在四个基准上平均降低约9% MSE、8% MAE；双MLP在iTransformer上进一步提升6–13% MSE，同时推理时间仅为10ms/5ms，显著低于Transformer级模型。

**⚠️ 局限性**

局限性包括：对分解方法的依赖主要是简单的移动平均，可能无法充分捕捉复杂季节性或非线性模式；实验范围限于四个数据集，未验证在更广泛行业或跨域时序任务中的通用性；对极端不平稳或极端事件的预测性能尚未评估。

---

## 763. HD-TTA: Hypothesis-Driven Test-Time Adaptation for Safer Brain Tumor Segmentation

**arXiv ID:** 2602.19454 | [PDF](https://arxiv.org/pdf/2602.19454v1)

**作者:** Kartik Jhawar `[一作]` (Nanyang Technological University), Lipo Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 11283 | [OpenAlex ID](https://openalex.org/A5086764741)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于假设驱动的测试时自适应框架（HD‑TTA），用于脑肿瘤分割时避免模型在不同域上产生不安全的边界泄漏或过度抑制。

**💡 创新点**

创新点在于将自适应过程视为决策流程，先用轻量化Gatekeeper筛选边缘样本，再通过并行生成“紧凑”与“扩张”两种几何假设，并用无监督的纹理一致性信号自动选择最安全的结果。

**🔧 技术方法**

技术包括冻结nnU‑Net v2基准网络，仅对logit进行优化，使用熵最小化、TV、重力损失与地理障碍约束的混合损失，配合门控阈值和无监督代表性选择器。

**📊 数据集**

使用BraTS 2023 GLI（成人胶质瘤）作为源域训练集，并在未见的BraTS 2023 PED（儿童）和BRA 2023 MEN（脑膜瘤）目标域上评估。

**📈 对比分析**

与经典TTA、SAR、IST、TCA、TEGDA等多种SOTA自适应方法对比，HD‑TTA在HD95（边界误差）和Precision（误检率）上分别下降约6.4 mm并提升约4%，Dice保持相近水平，显示出更安全、更稳定的性能。

**⚠️ 局限性**

局限在于仅针对二分类肿瘤分割验证，门控阈值和选择规则需人工设定，推理延迟较大，未在多类别或其他结构预测任务中充分验证。

---

## 764. Beyond Mimicry: Toward Lifelong Adaptability in Imitation Learning

**arXiv ID:** 2602.19930 | [PDF](https://arxiv.org/pdf/2602.19930v1)

**作者:** Nathan Gavenski `[一作]` (King's College London), Odinaldo Rodrigues `[通讯]` (King's College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种以目标为导向、可控上下文的MDP框架（GCMDP），并将模仿学习转向对任务结构的组合性泛化。

**💡 创新点**

创新点在于：①将任务目标与奖励分离，使用显式目标评估轨迹；②引入可控上下文与确定性转移，消除随机性干扰；③定义组合性泛化维度（系统性、生产力、可替换性）与泛化边界度量。

**🔧 技术方法**

采用GCMDP理论、可控上下文生成、Levenshtein距离评估上下文差异、确定性转移与轨迹级目标判定，并提出组合性泛化边界公式。

**📊 数据集**

主要使用基于块堆叠的仿真环境以及Procgen等可生成可控上下文的实验平台，但未给出具体数据集名称，强调对可控上下文与固定原语语义的需求。

**📈 对比分析**

与传统BC、IRL、对抗式模仿学习等方法对比，指出传统方法在跨分布泛化方面失效；作者通过组合性度量展示在不同上下文距离下的性能衰减曲线，但未给出具体数值，仅说明理论上可实现更稳健的泛化。

**⚠️ 局限性**

主要局限在于缺乏实证实验与基准验证，GCMDP对确定性与可控上下文的假设可能不适用于高度随机或未知环境；此外，框架仍需进一步评估在多任务、多智能体场景中的可扩展性。

---

## 765. All Cities are Equal: A Unified Human Mobility Generation Model Enabled by LLMs

**arXiv ID:** 2602.19694 | [PDF](https://arxiv.org/pdf/2602.19694v1)

**作者:** Bo Liu `[一作]` (Hunan University), Kenli Li `[通讯]` (Hunan University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种跨城市统一的合成人类移动生成框架UniMob

**💡 创新点**

通过LLM驱动的旅行规划、统一空间嵌入以及条件扩散生成器，实现了不受城市数据量限制的高保真移动合成

**🔧 技术方法**

大语言模型（LLM）用于推断时间与语义旅行计划，统一空间嵌入模块将不同城市的空间结构映射至共享表示空间，条件扩散Transformer（DiT）生成时空轨迹

**📊 数据集**

使用两份真实数据集：包含广州、深圳、长沙三市的私家车轨迹数据与北京、上海、深圳三市的手机定位数据

**📈 对比分析**

与8种基准（规则、EPR、GAN、DiffTraj、TrajGDM、LLMob、LLM-COD等）对比，UniMob在五项指标上平均提升30%+，在零样本和少量样本场景下依旧保持领先，并在未见城市中实现较好泛化

**⚠️ 局限性**

主要限制包括对大型LLM模型的依赖（模型规模大、算力需求高）以及对城市空间划分细节的依赖，未来需进一步降低模型体量和增强对极低资源城市的适配能力

---

## 766. Celo2: Towards Learned Optimization Free Lunch

**arXiv ID:** 2602.19142 | [PDF](https://arxiv.org/pdf/2602.19142v1)

**作者:** Abhinav Moudgil `[一作]` (Mila), Eugene Belilovsky `[通讯]` (Concordia University)

**通讯引用:** 1461 | [OpenAlex ID](https://openalex.org/A5025113992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为Celo2的学习优化器，通过极低的元训练计算（仅4.5 GPU小时）实现对大规模模型（如GPT‑3 1.3B、ViT、RL Atari）的稳定、优异优化效果。

**💡 创新点**

创新点在于：1）采用完全归一化的MLP更新规则并结合Newton‑Schulz正交化；2）将正交化、不同维度参数的独立更新规则、分离权重衰减等现代优化技术融入学习优化框架；3）构造一个极简、可迁移的元训练方案（任务增广、RMS归一化、PES等），显著提升元泛化能力；4）在极小的元训练预算下即可超越之前最强学习优化器VeLO。

**🔧 技术方法**

主要技术包括：MLP学习更新、RMS归一化、Newton‑Schulz正交化、任务增广、PES（持久进化策略）元训练、分离1‑D/2‑D参数更新、分离权重衰减、与现代优化框架（Optax）无缝集成。

**📊 数据集**

使用的基准数据集包括：MNIST、Fashion‑MNIST、CIFAR‑10、SVHN（8×8图像分类，用于元训练）；GPT‑2 124M、GPT‑3 1.3B、30M Transformer decoder（FineWeb‑Edu）用于语言建模；ViT‑ImageNet（512 batch, 50k steps）用于长序列训练；Atari环境（PPO）用于强化学习。

**📈 对比分析**

与AdamW和VeLO在相同的超参搜索预算下对比，Celo2在大模型训练（GPT‑3、GPT‑2）上实现更快收敛、更低最终损失；在ViT和RL任务上亦表现与或优于AdamW；相较于需要4000 TPU‑months的VeLO，Celo2仅需4.5 GPU‑hours即可获得更优或相当的性能。

**⚠️ 局限性**

限制包括：1）元训练仍需针对不同任务进行调优，泛化虽强但可能受限于训练任务种类；2）相对Adam，Celo2的内存开销更高（≈5×），运行时间略增（≈1.3×）；3）目前仅在float32下验证，混合精度下的稳定性尚待进一步研究；4）在极大规模（>1B）模型或更复杂任务上，进一步的可扩展性验证仍待完成。

---

## 767. AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation

**arXiv ID:** 2602.20057 | [PDF](https://arxiv.org/pdf/2602.20057v1)

**作者:** Ge Yuan `[一作]` (University of Hong Kong), Dong Xu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一的世界模型驱动扩散策略与在线自适应学习框架，提升机器人在动态环境中的操作性能。

**💡 创新点**

将世界模型作为主动监督，通过在线自适应学习（AdaOL）结合力-扭矩反馈形成闭环自我校正，并利用多模态自注意力实现模块间深度特征交换。

**🔧 技术方法**

采用 Flow Matching Diffusion Transformers、Cosmos-Predict2 世界模型、低秩适配（LoRA）在线更新、多模态自注意力（MMSA）以及力预测器。

**📊 数据集**

在 LIBERO‑10、Variant PushT、CALVIN 三个仿真基准和真实机器人上使用人类遥控演示数据（PS5 控制器）进行评估。

**📈 对比分析**

与 UVA、MODE、DP‑Force 等基线对比，在所有基准上均取得 SOTA；在线学习使 OOD 场景成功率提升约 5% 以上，实际机器人任务成功率提高 10%+。

**⚠️ 局限性**

受限于大模型计算成本、需依赖力传感器支持，且对极端动态变化和长时间连续任务的自适应能力仍有限。

---

## 768. A Checklist for Deploying Robots in Public: Articulating Tacit Knowledge in the HRI Community

**arXiv ID:** 2602.19038 | [PDF](https://arxiv.org/pdf/2602.19038v1)

**作者:** Claire Liang `[一作]` (Massachusetts Institute of Technology), Xiang Zhi Tan `[通讯]` (Northeastern University)

**通讯引用:** 558 | [OpenAlex ID](https://openalex.org/A5057463399)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并发布了一份面向公共空间机器人部署的可定制检查清单，采用模块化翻转卡片的层级表格形式，并通过社区协同维护与演进。

**💡 创新点**

创新点在于将机器人、研究人类与公共空间三个领域与部署阶段（前期、进行中、后期）相结合，形成可视化的翻转卡片，既可作为整体指南也可按需抽取子集；同时开放源码、支持个性化与在线编辑，促进社区共同完善。

**🔧 技术方法**

使用了Miro进行卡片可视化与协作，搭建了一个可交互的网站（GitHub托管），提供Word格式下载；技术实现主要是前端网页与GitHub Issue/PR 机制，确保资源易于贡献与更新。

**📊 数据集**

本工作不基于任何公开数据集，而是直接采集作者团队和六位专家在公共机器人部署中的经验与反馈。

**📈 对比分析**

未与传统方法做实验性性能对比；评价方式是专家访谈和社区反馈，验证了清单的可用性、易用性以及对实际部署流程的改进作用。

**⚠️ 局限性**

局限性包括：1）经验来源主要是北美和欧洲，缺乏全球南方视角；2）仍无法完全预防公共场景中不可预见的突发事件；3）工具需持续社区维护，若缺乏贡献将难以保持更新。

---

## 769. Depth-Structured Music Recurrence: Budgeted Recurrent Attention for Full-Piece Symbolic Music Modeling

**arXiv ID:** 2602.19816 | [PDF](https://arxiv.org/pdf/2602.19816v1)

**作者:** Yungang Yi `[一作]` `[通讯]` (Auckland University of Technology), Yungang Yi (Auckland University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了 Depth‑Structured Music Recurrence（DSMR），一种通过分层记忆窗长实现全片段长上下文的递归 Transformer；

**💡 创新点**

创新点在于将跨段 KV 记忆的时间窗口按层次分配（层级记忆 horizon），并通过两尺度调度（低层长记忆、高层短记忆）在固定预算内构建多时间感受野；

**🔧 技术方法**

采用 Transformer‑XL 样式的状态递归、stop‑gradient KV 缓存、可学习门控以及多种层级记忆策略；

**📊 数据集**

在 MAESTRO 现场钢琴 MIDI 数据集上进行训练与评估；

**📈 对比分析**

与全注意力 Transformer‑XL、Perceiver‑AR 类参考及多种记忆分配变体对比，DSMR 在相同记忆预算下实现 5.96 的验证 perplexity（低于 5.98 的全注意力模型），同时显著降低 GPU 内存使用（约 6.3 GB 对比 15.5 GB）并加快训练速度；

**⚠️ 局限性**

局限性包括仅在符号音乐任务上验证，教师强迫下的 perplexity 可能无法完整评估生成质量，且对更大规模模型或不同长序列任务的泛化尚未探究。

---

## 770. What Distributed Computing Got Wrong: The Category Mistake That Turned Design Choices into Laws of Nature

**arXiv ID:** 2602.18723 | [PDF](https://arxiv.org/pdf/2602.18723v1)

**作者:** Paul Borrill `[一作]` `[通讯]`, Paul Borrill

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

阐述分布式计算中的基本不可能定理是由对信息流向时空的类别错误（将单向前向信息流视为自然法则）导致的，并提出通过转向双向事务模型来消除这些不可能性

**💡 创新点**

将量子物理学中的本体/表象区分与分布式系统中的单向消息传递进行类比，利用莱布尼茨不可辨识原则识别并剔除多余本体结构，提出双向事务作为新的设计范式

**🔧 技术方法**

类别错误分析、Spekkens的本体/表象框架、莱布尼茨不可辨识原则、事务性通信模型（Wheeler‑Feynman吸收理论、Cramer事务解释、Open Atomic Ethernet实现）

**📊 数据集**

无（本文为理论性研究，无数据集）

**📈 对比分析**

无实验比较，本文通过逻辑推导和对比已证明的不可能定理来说明双向事务模型可消除FIT0假设下的限制；未给出性能数值

**⚠️ 局限性**

依赖于对单向消息传递假设的严格排除，现实系统中实现双向事务可能需要新协议和硬件支持；论文未讨论实现成本、可扩展性与兼容性等实际工程限制

---

## 771. Artificial Intelligence for Modeling & Simulation in Digital Twins

**arXiv ID:** 2602.19390 | [PDF](https://arxiv.org/pdf/2602.19390v1)

**作者:** Philipp Zech `[一作]` (University of Innsbruck), Istvan David `[通讯]` (McMaster University)

**通讯引用:** 612 | [OpenAlex ID](https://openalex.org/A5041475393)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述数字孪生中建模、仿真与人工智能的协同作用，阐述其在运营、研发和业务三大层面的角色，并总结标准、挑战与未来研究方向。

**💡 创新点**

首次系统性梳理建模与仿真、AI与数字孪生之间的双向互补关系，提出可操作的框架和标准化路线图。

**🔧 技术方法**

综合运用多范式建模（物理、离散事件、系统动力学、代理模型等）、仿真技术、机器学习、深度学习与强化学习，并结合数字孪生架构层次。

**📊 数据集**

未使用单一公开数据集，而是引用行业典型的物联网传感器流、维护记录、工厂工艺日志等多源实时/历史数据。

**📈 对比分析**

该工作属于综述性质，未进行实验对比；通过对比国内外标准（ISO 23247、ISO 30173等）和案例，展示了不同方法在实时性、可解释性、可扩展性等维度的差距与潜在收益。

**⚠️ 局限性**

局限在于缺乏统一实验评估，所述技术与标准在不同领域的适配性仍需验证，且对深度学习模型的可解释性、数据治理和实时性能挑战未给出完整解决方案。

---

## 772. Searching Through Complex Worlds: Visual Search and Spatial Regularity Memory in Mixed Reality

**arXiv ID:** 2602.18669 | [PDF](https://arxiv.org/pdf/2602.18669v1)

**作者:** Lefan Lai `[一作]` (University of Sydney), Brandon Victor Syiem `[通讯]` (University of Sydney)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5012972097)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在Meta Quest 3 MR环境中设计双因素实验（物理环境复杂度、虚拟对象深度、任务类型），研究了这些因素对视觉搜索效率和空间规律记忆的影响；

**💡 创新点**

创新点在于同时考察三种现实与虚拟交互环境因素的交互效应，发现二次听觉任务对客观搜索性能影响不大但主观负荷显著提升，且不同深度的虚拟对象能在复杂环境中降低搜索负担；

**🔧 技术方法**

技术上采用Unity 6000.1.0f1开发MR应用，并用线性混合效应模型、广义线性混合模型以及序列型混合模型对反应时、准确率与NASA‑TLX分量进行统计分析；

**📊 数据集**

数据集为24名受试者在8种实验条件下完成125次视觉搜索试验和4次识别试验，共计24,728条反应时记录和768条识别记录；

**📈 对比分析**

与传统两维显示或单因素研究相比，本文未使用外部基准模型，而是通过对比不同实验条件内部的反应时、准确率和主观负荷，证明复杂环境和不同深度会显著拉长搜索时间，重复空间配置能显著提升搜索速度；

**⚠️ 局限性**

局限性包括：实验环境固定且静止，未考察移动或动态场景；仅控制颜色、大小等视觉属性；样本量虽符合HCI常规但缺乏事先功效分析；未检验多任务对空间规律学习的长期影响。

---

## 773. RegionRoute: Regional Style Transfer with Diffusion Model

**arXiv ID:** 2602.19254 | [PDF](https://arxiv.org/pdf/2602.19254v1)

**作者:** Bowen Chen `[一作]` (University of Texas at Austin), Divya Kothandaraman `[通讯]` (Dolby Laboratories, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种注意力监督的扩散框架，旨在实现精确且无掩码的局部风格转移。

**💡 创新点**

通过将风格令牌的注意力图与目标对象的二进制掩码对齐，显著提高了局部风格转移的精度，且不依赖手工分割。

**🔧 技术方法**

使用了注意力监督的扩散模型，结合了LoRA-MoE（低秩适应）策略以提高参数效率和风格多样性。

**📊 数据集**

使用了Grounded COCO数据集的子集，生成了600个伪地面真相图像，涵盖四种代表性风格（像素艺术、赛博朋克、表现主义和线条艺术）。

**📈 对比分析**

与多种基线方法进行了比较，结果显示该方法在区域风格匹配（RSM）上表现优异，同时在背景保留（LPIPS和MSE）上显著优于现有方法，表明其在局部风格转移中的有效性。

**⚠️ 局限性**

局限性在于对非常小、被遮挡或语义模糊的对象的处理仍然存在挑战，未来研究可以探索如何增强基于注意力的空间对齐能力。

---

## 774. A User-driven Design Framework for Robotaxi

**arXiv ID:** 2602.19107 | [PDF](https://arxiv.org/pdf/2602.19107v1)

**作者:** Yue Deng `[一作]` (Hong Kong University of Science and Technology), Changyang He `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了真实世界机器人出租车（robotaxi）使用体验，系统梳理了用户动机、收益、挑战及对隐私、安全、伦理与信任的感知，并基于此提出了一套以用户需求为驱动的端到端设计框架。

**💡 创新点**

创新点在于：① 通过18份半结构访谈和22次自动民族学车程实地调查获取真实场景数据；② 将动机、优势与痛点与隐私、安防、伦理维度整合；③ 提出了涵盖预订、接载、行驶和下车四阶段的用户驱动设计框架。

**🔧 技术方法**

采用质性研究方法，包括半结构访谈、自动民族学、主题分析、affinity diagramming；并结合机器人出租车平台的交互信息与传感器状态进行描述性分析。

**📊 数据集**

数据集包含：18名用户访谈记录（包括背景、使用频次、平台、城市等信息）和22次机器人出租车行程日志（记录时间、天气、地点、用户情绪、车辆状态等）。

**📈 对比分析**

研究未进行定量性能对比；通过主题编码与对比分析验证发现，提出的设计框架在真实场景下具有可操作性，为后续产品与服务设计提供实践指引。

**⚠️ 局限性**

局限性包括：① 样本受限于pilot城市与平台多样性不足；② 研究者作为参与者的自动民族学可能带来自我影响；③ 缺乏长期跟踪与极端天气/高复杂路况下的体验数据；④ 结果主要为描述性，缺乏可量化评估。

---

## 775. Structured Bitmap-to-Mesh Triangulation for Geometry-Aware Discretization of Image-Derived Domains

**arXiv ID:** 2602.19474 | [PDF](https://arxiv.org/pdf/2602.19474v1)

**作者:** Wei Feng `[一作]` (Ocean University of China), Haiyong Zheng `[通讯]` (Ocean University of China)

**通讯引用:** 2693 | [OpenAlex ID](https://openalex.org/A5049456032)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

针对基于像素的域，提出一种基于模板的局部重划分框架（SBMT），实现了边界精确嵌入、结构保持、局部一致且无全局拓扑更新的三角网生成。

**💡 创新点**

创新点在于：① 对边界与等边三角格的交互进行离散等价与对称分类，构造有限的符号查找表；② 采用单值、无状态的模板替换规则保证确定性和可并行；③ 通过阈值驱动的预处理（顶点捕获、脚点排斥、短边消除）保证子三角角度与面积的下界，从而实现无斜角、近等边网格。

**🔧 技术方法**

主要技术：等边基础网格、离散交叉模式分类、符号查找表、局部模板重划分、顶点捕获/排斥/短边消除、并行无锁执行、离散微分算子与有限元数值稳定性分析。

**📊 数据集**

使用合成数据（星形、滴形、Y 形）与真实医学二值图（胃截面）进行实验验证，并与 Shewchuk 的 Triangle 与 Gmsh 的 CDT 进行对比。

**📈 对比分析**

与 Triangle / Gmsh 相比，SBMT 在边界对齐、无斜角、近等边内部网格方面表现更好；网格单元数更少，内角分布峰值靠近 60°；但最小内角低于传统 Delaunay 优化。数值实验显示在 PDE 计算（热扩散、谐波 1-形式）中保持稳定且几何一致。

**⚠️ 局限性**

局限性：① 需要满足角度≥90°和段长>网格边长的预设约束，超出此范围时模板覆盖不足；② 通过固定阈值的预处理缺乏自适应细化；③ 不能直接提供全局最优角度保证，最小内角低于 CDT；④ 对复杂曲线或高频细节的细化能力有限。

---

## 776. pHNSW: PCA-Based Filtering to Accelerate HNSW Approximate Nearest Neighbor Search

**arXiv ID:** 2602.19242 | [PDF](https://arxiv.org/pdf/2602.19242v1)

**作者:** Zheng Li `[一作]` (South China University of Technology), Simei Yang `[通讯]` (South China University of Technology)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5083744300)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了pHNSW，融合PCA降维与Hierarchical Navigable Small World（HNSW）算法，并实现了专用硬件处理器。

**💡 创新点**

通过在低维空间进行PCA过滤搜索并仅在高维空间计算k次距离，实现了算法与硬件的协同优化；对数据库结构做出规律访问的重组；设计自定义指令集与专用的Dist.L、kSort.L模块以提升吞吐量与能效。

**🔧 技术方法**

使用PCA降维、HNSW图搜索、定制指令集体系结构、并行距离计算与k排序模块、DRAM/HBM访问控制及优化的离散化数据库表。

**📊 数据集**

SIFT1M 128维视觉特征数据集。

**📈 对比分析**

在单查询场景下与CPU和GPU标准HNSW做对比，QPS提升14.47×至21.37×，能耗下降57.4%。

**⚠️ 局限性**

数据库内存使用量增加约2.92倍；目前仅在单核单查询环境验证，未对更大规模（如SIFT1B）或多核/PIM体系结构进行评估。

---

## 777. HONEST-CAV: Hierarchical Optimization of Network Signals and Trajectories for Connected and Automated Vehicles with Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.18740 | [PDF](https://arxiv.org/pdf/2602.18740v1)

**作者:** Ziyan Zhang `[一作]` (University of California), Guoyuan Wu `[通讯]` (University of California)

**通讯引用:** 5674 | [OpenAlex ID](https://openalex.org/A5006183071)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一套层次化的协同优化框架，用以联合调度交通信号灯（TSC）和自动驾驶车辆（CAV）的能耗高效行驶（EAD）。

**💡 创新点**

创新点包括：①基于价值分解网络（VDN）的多智能体强化学习（MARL）实现网络层级的协同信号控制；②通过混合预测算法对下一周期绿灯时段进行实时估计，桥接信号控制与车辆轨迹规划；③使用机器学习轨迹规划算法（MLTPA）逼近最优能耗轨迹，兼顾实时性与能耗优化；④框架可插拔、可扩展，适用于不同动力系统（ICEV/EV）。

**🔧 技术方法**

技术手段：多智能体Soft Actor-Critic（MASAC）+ VDN、混合阶段预测、机器学习轨迹规划（MLTPA）以及SUMO仿真。

**📊 数据集**

数据集与仿真环境：采用杭州真实 4×4 城市网络（来自 CityFlow 校准流量）在 SUMO 上进行仿真；CAV 比例随机在 10%–90% 之间，动力系统分别为 ICEV 与 EV。

**📈 对比分析**

比较方法：与传统 Webster 固定信号、独立强化学习（IRL）以及仅 TSC 或仅 MLTPA 方案对比。实验结果显示：在 60% CAV 环境下，能耗降低 10.23%（相对基线），怠速时间下降 45.83%，平均速度提升 7.67%；不同 CAV/EV 比例下，性能随比例提升显著提升，EV 环境更能显著降低能耗。

**⚠️ 局限性**

局限性：仅在 4×4 网络规模仿真，未验证大规模真实部署；SPaT 预测依赖实时感知，可能对感知误差敏感；框架假设 CAV 与 HV 的比例已知且能及时更新，实际城市交通流变动更为复杂。

---

## 778. One Year After the PDPL: a Glimpse into the E-Commerce World in Saudi Arabia

**arXiv ID:** 2602.18616 | [PDF](https://arxiv.org/pdf/2602.18616v1)

**作者:** Eman Alashwali `[一作]` (King Abdulaziz University), Abeer Alhuzali `[通讯]` (King Abdulaziz University)

**通讯引用:** 159 | [OpenAlex ID](https://openalex.org/A5024214057)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对100个沙特阿拉伯电商网站的隐私政策进行细粒度的合规性评估，并探讨利用LLM自动化检查PDPL合规性的可行性。

**💡 创新点**

首次将PDPL细粒度条款与LLM自动分析结合，揭示搜索排名与托管平台对合规性的显著影响，且系统化评估LLM的误差来源与改进方向。

**🔧 技术方法**

采用手工定性分析（模板分析）、Python文本提取器以及GPT‑5 LLM进行自动问答，随后与人工答案进行对比。

**📊 数据集**

收集自Google搜索的100个沙特阿拉伯电商网站名单，提取其对应的隐私政策全文（含中文/英文版本）。

**📈 对比分析**

将人工标注与LLM回答进行匹配，计算agreement率：保留、销毁、投诉约>90%，获取副本仅58%；表明LLM总体表现良好但在细粒度理解上仍有提升空间。

**⚠️ 局限性**

样本规模有限（仅100个网站、仅四项PDPL条款）、未评估实际数据处理行为、LLM对上下文敏感导致误判、网站政策随时可能变更，导致结果易过时。

---

## 779. CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence

**arXiv ID:** 2602.20048 | [PDF](https://arxiv.org/pdf/2602.20048v1)

**作者:** Tarakanath Paipuru `[一作]` `[通讯]` (Independent Researcher), Tarakanath Paipuru (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了在大上下文窗口下，LLM 编码代理仍然无法充分利用架构层次信息的问题，提出“导航悖论”，并通过构建 CodeCompass 图形导航工具进行验证。

**💡 创新点**

创新点在于（1）系统化地定义并验证“导航悖论”；（2）提出三组任务分类（语义、结构、隐藏）以精准衡量检索与导航的差异；（3）展示了在隐藏依赖任务中，图形导航可显著提升覆盖率；（4）发现并量化了工具采纳率对性能的决定性影响。

**🔧 技术方法**

使用了 Model Context Protocol（MCP）与 Neo4j 构建静态 AST 依赖图；基于 Claude Sonnet‑4‑5 LLM 进行 agentic 编码；检索端采用 BM25；评估指标包括 Architectural Coverage Score（ACS）、First Correct Tool Call（FCTC）和 Veto Protocol。

**📊 数据集**

数据集为 FastAPI RealWorld 示例应用，约 3,500 行代码、40 个源文件；从中抽取 30 个手工编写的代码修改任务，按任务可检索性分为 G1、G2、G3 三组。

**📈 对比分析**

在 30 题 3 组的 258 次完整实验中，Graph 条件在隐藏依赖（G3）任务上 ACS 达到 99.4%，比 Vanilla（76.2%）和 BM25（78.2%）提升 23.2pp；在语义任务上 BM25 以 100% ACS 主导；在结构任务上 Graph 仍略逊于 BM25。

**⚠️ 局限性**

局限性包括：仅在单一 Python 项目上验证；ACS 只衡量文件访问而非实现正确性；工具采纳率高度依赖提示工程，缺乏统一的工具调用约束；图结构由自动 AST 生成，缺少人工验证，可能导致错误依赖。

---

## 780. Cross-lingual Matryoshka Representation Learning across Speech and Text

**arXiv ID:** 2602.19991 | [PDF](https://arxiv.org/pdf/2602.19991v1)

**作者:** Yaya Sy `[一作]` (LORIA, CNRS), Irina Illina `[通讯]` (LORIA, CNRS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了首个跨语言语音-文本 Matryoshka 嵌入模型，使 Wolof 语音能直接检索 French 文本。

**💡 创新点**

创新点在于将语音特征通过 Late‑Fusion 方式嵌入冻结文本 Matryoshka 模型，并证明信息主要集中于少数维度，可实现灵活的精度-成本折中。

**🔧 技术方法**

采用 Qwen3‑0.6‑Embedding、HuBERT、InfoNCE 对比 Late‑Fusion 与 Dual‑Encoder 结构，以及基于 Matryoshka Representation Learning 的多维嵌入。

**📊 数据集**

使用人工合成的 Wolof‑French 语音-文本对、French mMARCO、Senegalese French 网页、以及 Kallaama 与 Fleurs 真实语音检索基准。

**📈 对比分析**

在 Kallaama‑Retrieval‑Eval 与 Fleurs‑Retrieval‑Eval 上，Late‑Fusion 模型在 128–1024 维度下均超过 10 倍参数的 NLLB‑LLM2Vec 与 pipelined baseline，且在少样本语音意图检测任务中达到 96% F1。

**⚠️ 局限性**

局限性包括仅针对 Wolof‑French 语言对验证，过度依赖合成数据，Matryoshka 低秩压缩效率不佳。

---

## 781. OpenClaw, Moltbook, and ClawdLab: From Agent-Only Social Networks to Autonomous Scientific Research

**arXiv ID:** 2602.19810 | [PDF](https://arxiv.org/pdf/2602.19810v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 782. Quantifying Automation Risk in High-Automation AI Systems: A Bayesian Framework for Failure Propagation and Optimal Oversight

**arXiv ID:** 2602.18986 | [PDF](https://arxiv.org/pdf/2602.18986v1)

**作者:** Vishal Srivastava `[一作]` (Johns Hopkins University), Tanmay Sah `[通讯]` (Harrisburg University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

该论文提出了一个贝叶斯风险分解公式，用以量化AI系统自动化水平导致的危害概率，分离技术风险、部署风险与后果风险。

**💡 创新点**

创新点在于将P(H|F,A)与执行控制可观测性等价，并给出自动化效率弹性、最优资源分配和效率前沿的理论推导。

**🔧 技术方法**

采用了贝叶斯概率推理、风险弹性分析、最优前沿理论以及案例分析等技术。

**📊 数据集**

论文以2012年Knight Capital交易算法失效事件为示例，并在文中讨论跨领域（金融、医疗、交通、内容治理、关键基础设施）事故数据库的构建需求。

**📈 对比分析**

通过与传统仅关注模型准确度的风险评估方法对比，展示在高自动化情境下降低P(H|F,A)的投资回报远高于降低P(F)，表明该方法能显著降低预期损失。

**⚠️ 局限性**

限制在于假设P(F)与A独立、P(H|F,A)的具体形式未知以及缺乏跨域实证验证。

---

## 783. Unlearning Noise in PINNs: A Selective Pruning Framework for PDE Inverse Problems

**arXiv ID:** 2602.19967 | [PDF](https://arxiv.org/pdf/2602.19967v1)

**作者:** Yongsheng Chen `[一作]` (Zhejiang University), Xinghui Zhong `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于选择性剪枝的物理信息神经网络（P-PINN），通过对已有PINN模型进行残差-数据忠实度评估、偏置驱动的神经元重要性衡量和迭代剪枝，再对保留数据进行微调，以提升在噪声观测下的逆PDE求解精度。

**💡 创新点**

创新点包括：① 将残差与数据误差结合成联合指标用于划分可靠与受损样本；② 引入偏置型神经元重要性度量，捕捉对受损样本响应偏移的隐藏单元；③ 采用迭代剪枝并在每步重新计算重要性，实现自适应结构压缩；④ 将剪枝后微调作为轻量后处理，避免完整重训练。

**🔧 技术方法**

技术手段主要包括：物理信息神经网络（PINN）框架、残差-数据融合评分、基于激活差异的偏置重要性指标、层级迭代剪枝与权重零化、对剪枝后网络在可靠数据上微调、对比实验与多指标评估（L2RE、MSE、fMSE等）。

**📊 数据集**

使用九个PDE逆问题基准：Poisson、Heat、Wave、Stokes（数据同化）以及参数逆问题PInv、HInv、NSInv、EBInv、WInv，数据集包含高低噪声混合观测、采样点与残差点均匀或Sobol分布。

**📈 对比分析**

与基线PINN、仅在保留数据上微调（FT）、在保留数据上从零训练（RT）、以及Bayesian/Dropput PINN（B-PINN/D-PINN）等方法对比；P-PINN在所有9个基准上平均L2RE下降至0.026，最高可达96.6%的误差降低，且高频误差显著下降，训练时间仅为FT的1–2倍、远低于RT。

**⚠️ 局限性**

局限性：缺乏严格的理论分析与收敛保证；剪枝阈值与迭代策略仍需经验调参；仅验证在全连接MLP上，未探究卷积或图网络；未结合贝叶斯不确定性量化；对极大规模网络与多物理耦合系统的适用性还有待进一步验证。

---

## 784. PolyFrame at MWE-2026 AdMIRe 2: When Words Are Not Enough: Multimodal Idiom Disambiguation

**arXiv ID:** 2602.18652 | [PDF](https://arxiv.org/pdf/2602.18652v1)

**作者:** Nina Hosseini-Kivanani `[一作]` (University of Luxembourg), Nina Hosseini-Kivanani `[通讯]` (University of Luxembourg)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5060530051)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PolyFrame 系统，用统一的多阶段管道实现多语言多模态成语歧义判定与排名任务。

**💡 创新点**

创新点包括：① 将句子类型分类与成语改写结合，显式处理成语的非组合性；② 采用三流零样本相似度（SigLIP2 视觉-文本、BGE‑M3 文本、SigLIP2 文本）并用 Borda 加权融合；③ 在不微调大型编码器的前提下，通过轻量级模块（逻辑回归、LLM 辅助句型预测、替换表）实现性能大幅提升。

**🔧 技术方法**

技术手段：冻结 CLIP/SigLIP2 视觉‑语言编码器、BGE‑M3 句子编码器；逻辑回归预测句子是否成语；LLM（GPT‑4o/Qwen3 等）做 literal‑first 句型提示；成语同义替换；三路相似度计算；Borda‑rank 融合。

**📊 数据集**

数据集：AdMIRe 2.0 共享任务数据（英葡训练/验证/测试）以及覆盖 15 种语言的 blind‑test 评测集，包含句子、目标成语、5 张候选图像与其说明文字。

**📈 对比分析**

与基线相比，基线 CLIP 仅 26.7% Top‑1；加入成语改写提升至 60.0% Top‑1；零样本跨语言推断 PT 亦达 60.0% Top‑1 与 0.822 NDCG@5；在 15 语言 blind‑test 上平均 Top‑1 0.35、NDCG 0.73（子任务 A）与 0.32、0.71（子任务 B），表明跨语言、跨模态的鲁棒性良好。

**⚠️ 局限性**

局限性：成语同义替换表手工维护，仅覆盖约 50 种英语成语，非英语覆盖不足；融合权重固定为 Borda 系数，缺乏自适应学习；依赖外部 LLM 句型预测增加延迟与成本；在低资源或文化差异显著的语言上性能仍有提升空间。

---

## 785. DReX: An Explainable Deep Learning-based Multimodal Recommendation Framework

**arXiv ID:** 2602.19702 | [PDF](https://arxiv.org/pdf/2602.19702v1)

**作者:** Adamya Shyam `[一作]` (University of Delhi), Vikas Kumar `[通讯]` (University of Delhi)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为DReX的多模态推荐框架，能够在交互层面动态融合文本评论与评分信息，并通过GRU迭代更新全局用户与项目嵌入，实现统一且对缺失模态鲁棒的表示学习。

**💡 创新点**

其创新点包括：①在全局表示更新中同时考虑细粒度交互特征与整体偏好；②消除传统方法中独立训练用户/项目特征的耦合误差；③利用注意力权重自动生成用户与项目关键词档案，从而提升可解释性并适应缺失模态场景。

**🔧 技术方法**

技术上结合了BERT提取评论语义、线性投影与注意力聚合生成文本特征、one-hot编码与线性映射获得评分特征、交叉特征拼接后再映射为交互特征，并用GRU进行迭代更新全局嵌入；最终用MLP进行评分预测。

**📊 数据集**

实验使用Amazon公开数据集中的三个子域：Video Games、Software 与 CD & Vinyl，每个数据集均经过过滤后拆分为70/20/10的训练/验证/测试集。

**📈 对比分析**

与EMF、DMF、DeepCoNN、NARRE、PESI等五个先进基线在MAE、NDCG、F1等指标上进行对比，DReX在MAE与NDCG上均表现优于所有基线，尤其在Top‑k推荐精度上显著提升；DReX‑MLP在更大k值下略优。

**⚠️ 局限性**

局限性包括：当前仅实现了文本与评分两种模态，未扩展至图像或音频等更丰富来源；虽然可处理缺失模态，但对极端稀疏或噪声高的模态仍可能影响性能；此外模型仍需进一步优化以生成自然语言级解释。

---

## 786. IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping

**arXiv ID:** 2602.18709 | [PDF](https://arxiv.org/pdf/2602.18709v1)

**作者:** Tingyang Xiao `[一作]` (Horizon Robotics), Zhizhong Su `[通讯]` (Horizon Robotics)

**通讯引用:** 2829 | [OpenAlex ID](https://openalex.org/A5087325725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种RGB语义SLAM系统IRIS‑SLAM，融合了统一的几何‑实例表示，实现了密集几何重建与实例级语义理解的协同工作；

**💡 创新点**

创新点在于将几何基础模型扩展为同时输出稠密深度与跨视角一致的实例嵌入，利用这些视角不变的语义锚点实现数据关联、实例引导闭环检测，并将语义映射与闭环检测统一为同一实例一致性约束；

**🔧 技术方法**

采用Transformer‑based feed‑forward 3D重建框架（如Depth‑Anything‑v3）加入实例头；使用对比学习的 pull/push 损失实现跨视角一致性；利用实例嵌入做聚类、投影与语义相似度匹配；闭环通过实例匹配与几何一致性验证，并进行Sim(3)全局优化；

**📊 数据集**

在TUM RGB‑D、Replica、ScanNet等数据集上评估，使用TUM RGB‑D做位姿评估，Replica/ScanNet做零样本语义映射；

**📈 对比分析**

与ORB‑SLAM3、DROID‑SLAM、Mast3r‑SLAM、VGGT‑SLAM、NetVLAD、SALAD等基线对比，IRIS‑SLAM在TUM RGB‑D位姿误差上排名前列，在语义映射mIoU、mAcc等指标上超越SOTA，在极端宽视角闭环检索中Recall@1、F1显著高于传统全局描述子；

**⚠️ 局限性**

局限在于单目输入导致尺度不确定，缺乏多视角/多模态约束；闭环检测仍受实例检测质量影响，且对极端遮挡/动态场景的鲁棒性待进一步提升。

---

## 787. A Computer Vision Framework for Multi-Class Detection and Tracking in Soccer Broadcast Footage

**arXiv ID:** 2602.18504 | [PDF](https://arxiv.org/pdf/2602.18504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 788. Prompt Optimization Via Diffusion Language Models

**arXiv ID:** 2602.18449 | [PDF](https://arxiv.org/pdf/2602.18449v1)

**作者:** Shiyu Wang `[一作]` (Salesforce AI Research), Huan Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于扩散语言模型（DLM）的提示优化框架，通过在交互轨迹上对系统提示进行掩码去噪迭代，动态细化提示。

**💡 创新点**

创新点在于利用DLM的双向迭代生成特性实现无梯度、无模型参数更新的span级提示更新，并通过用户或LLM反馈实现交互式、可自适应的提示优化。

**🔧 技术方法**

使用扩散语言模型（Dream‑7B）、掩码-去噪迭代、交互式反馈条件化、以及对比传统AR模型和梯度优化方法。

**📊 数据集**

在多任务数据集上评估：τ‑bench（airline、retail）、SST‑2、SST‑5、MRPC、SNLI。

**📈 对比分析**

与基线、Llama‑3‑8B、Qwen3‑8B及TextGrad比较，DLM优化后的提示在所有任务上均提升成功率或准确率，尤其在结构化函数调用和细粒度情感分析上显著提高；在SST‑5上最佳性能出现于约64步去噪。

**⚠️ 局限性**

局限性包括：需要足够的掩码步数（过多会导致过度编辑或重复内容）、对DLM训练质量和推理速度敏感、尚未在更大规模或多语言场景下验证，且对非结构化提示的改进效果有限。

---

## 789. A Risk-Aware UAV-Edge Service Framework for Wildfire Monitoring and Emergency Response

**arXiv ID:** 2602.19742 | [PDF](https://arxiv.org/pdf/2602.19742v1)

**作者:** Yulun Huang `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套完整的 UAV 边缘服务框架，用于野火监测与应急响应，框架同时优化传感器聚类、边缘节点分配、巡航路线与机队规模，并实现了基于火险历史的动态紧急调度；

**💡 创新点**

创新点在于：① 火险历史加权的 K‑means 聚类，使高风险区域优先巡航；② QoS 关注的两阶段边缘分配，兼顾距离与负载；③ 2‑opt 本地搜索与自适应机队规模相结合的路线优化；④ 三步动态紧急调度协议；整体实现了子问题间的闭环协同；

**🔧 技术方法**

采用加权 K‑means、两阶段边缘分配、2‑opt 路线优化、迭代机队尺寸决策、紧急调度三步协议，并用多目标加权目标函数与约束约束；

**📊 数据集**

实验基于模拟的野火风险场景：100 km² 区域内 200 传感器的非均匀分布，采样自真实野火风险数据，边缘节点 5 台，仿真 20 次；

**📈 对比分析**

与 GA、PSO、贪心三种基线在同一约束下对比，结果显示：平均响应时间下降 70.6–84.2%，能耗下降 73.8–88.4%，机队规模缩减 26.7–42.1%；紧急响应时间 233 s，满足 300 s 截止，且对正常运行影响微乎其微；

**⚠️ 局限性**

主要限制：仅在离线规划环境下验证，未考虑实时天气、复杂地形、异构 UAV 及多模态传感器；缺乏真实野火现场部署验证，且大规模扩展（>300 传感器）需进一步验证可行性。

---

## 790. Scalable Low-Density Distributed Manipulation Using an Interconnected Actuator Array

**arXiv ID:** 2602.19653 | [PDF](https://arxiv.org/pdf/2602.19653v1)

**作者:** Bailey Dacre `[一作]`, Andrés Faíña `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一个由3-自由度平面式机器人模块组成、通过柔性连接层形成连续操控表面的2×2分布式操纵系统，验证了其能够在不靠近多接触点的情况下对尺寸小于模块间距的物体进行平稳平移；

**💡 创新点**

利用可调节柔性连接层显著降低了传统高密度分布式操纵器的执行器密度，并通过工作空间耦合分析与状态机控制实现了对任意位置的物体平移；

**🔧 技术方法**

采用了基于Canfield三轴平面机的3-自由度模块，三轴步进电机+磁编码器+FOC控制；柔性层使用天然橡胶+PVC薄膜；微控制器使用Teensy；通信通过SPI+串口；路径规划采用Dijkstra算法；

**📊 数据集**

使用6相OptiTrack运动捕捉系统实时获取物体位置；未使用公开数据集；

**📈 对比分析**

通过在2×2原型上对四种形状（立方体、圆盘、球体、四面体）进行循环与点对点平移实验，实验显示在相同物体尺寸下，球体在连接层平移时更快，而平板物体在模块间平移时更可靠；与传统高密度DMS相比，操纵器数量减少约60%，但在极端角度下位置误差略增；

**⚠️ 局限性**

（1）柔性层在中心区域存在非线性弹性和闪变，导致不可预测的运动；（2）边界张力不均衡导致物体在中心偏移；（3）静态XY区域划分在大角度时误差增大；（4）实验仅限于2×2尺寸，缺乏大规模阵列验证；

---

## 791. Topology of Reasoning: Retrieved Cell Complex-Augmented Generation for Textual Graph Question Answering

**arXiv ID:** 2602.19240 | [PDF](https://arxiv.org/pdf/2602.19240v1)

**作者:** Sen Zhao `[一作]` (Chongqing University of Posts and Telecommunications), Ding Zou `[通讯]` (Zhongxing Telecom Equipment)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 TopoRAG 框架，通过细胞复杂体将文本图提升到高维拓扑空间，结合拓扑子复合体检索、多维拓扑推理与细胞复合体驱动的生成，实现更精细的问答推理。

**💡 创新点**

创新点：①将文本图映射到正则细胞复杂体，显式编码节点、边与循环等高维拓扑结构；②拓扑感知的子复合体检索，结合 Prize‑Collecting Steiner Tree 的高维扩展；③多维消息传递机制，跨 0/1/2 维细胞传播信息；④将细胞复合体表示作为软提示融入 LLM，实现结构化、循环感知的推理。

**🔧 技术方法**

技术细节：细胞提升映射、基于 SentenceBert 的细胞嵌入、拓扑子复合体检索（高维 Prize‑Collecting Steiner Tree），多维消息传递网络（面/共面消息），soft‑prompt 训练，LoRA 参数高效微调，LLM 基座为 Llama‑2‑7B，使用 AdamW 优化。

**📊 数据集**

实验数据集：ExplaGraphs、WebQSP、SceneGraphs（或相似的三大公开文本图问答数据集）。

**📈 对比分析**

与 inference‑only、frozen‑LLM + prompt tuning、tuned‑LLM（LoRA）等 GraphRAG 基线对比。TopoRAG 在所有基线上均显著提升，Acc 提升约 5%，Hit 提升约 4.7%；LoRA 版 TopoRAG 更进一步，Acc 0.9151、Hit 90.66，展示了卓越的性能。

**⚠️ 局限性**

局限性：①对大规模图的检索与推理成本仍较高；②细胞复杂体构造与检索算法尚未针对极大图做优化；③当前仅考虑二维循环，未扩展到更高维拓扑结构；④依赖预训练 LLM 的语言理解，若 LLM 先验不足，仍可能出现幻觉。

---

## 792. Beyond Behavioural Trade-Offs: Mechanistic Tracing of Pain-Pleasure Decisions in an LLM

**arXiv ID:** 2602.19159 | [PDF](https://arxiv.org/pdf/2602.19159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 793. Deep Else: A Critical Framework for AI Art

**arXiv ID:** 2602.19754 | [PDF](https://arxiv.org/pdf/2602.19754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 794. FuzzySQL: Uncovering Hidden Vulnerabilities in DBMS Special Features with LLM-Driven Fuzzing

**arXiv ID:** 2602.19490 | [PDF](https://arxiv.org/pdf/2602.19490v1)

**作者:** Yongxin Chen `[一作]` (National University of Defense Technology), Yongjun Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 4912 | [OpenAlex ID](https://openalex.org/A5100424205)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一种基于大型语言模型的自适应数据库模糊测试框架，系统性挖掘并复现了多种 DBMS 的特殊功能漏洞。

**💡 创新点**

创新点在于将语法引导的模板生成、逻辑移位递进变异、混合错误修复以及状态维护重放验证融合为一体，从而突破传统模糊器对特性盲点和状态依赖缺陷的局限。

**🔧 技术方法**

技术手段包括：CFG 语法模板扩展 + LLM（如 Qwen3‑30B）语义化实例化与错误修复；逻辑移位变异算法；基于 AFL++ 的灰盒覆盖与 AddressSanitizer；持久化会话与重放验证。

**📊 数据集**

实验使用了五个开源 DBMS（MySQL、MariaDB、SQLite、PostgreSQL、ClickHouse）作为测试目标，未采用外部 SQL 数据集，全部由框架自动生成。

**📈 对比分析**

与 Squirrel、EET、SQLancer 三种主流模糊器在相同 24 小时预算下对比，FuzzySQL 在边界特性漏洞发现率、整体覆盖率和首次发现 bug 的时间方面均优于对手，覆盖率提升数倍，bug 发现更快。

**⚠️ 局限性**

局限性主要体现在：当前 LLM 对深度嵌套或极复杂 SQL 结构的生成能力不足，导致在极简系统如 SQLite 或高优化引擎如 ClickHouse 的表现不如某些基于 AST 的模糊器；修复流程增加了计算开销；对新 DBMS 的迁移仍需提供语法文件和初始模式生成脚本。

---

## 795. Mechanism Design via Market Clearing-Prices for Value Maximizers under Budget and RoS Constraints

**arXiv ID:** 2602.19085 | [PDF](https://arxiv.org/pdf/2602.19085v1)

**作者:** Xiaodong Liu `[一作]` (Renmin University of China), Zihe Wang `[通讯]` (Renmin University of China)

**通讯引用:** 2460 | [OpenAlex ID](https://openalex.org/A5069728539)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在在线广告中，买家具有私有预算和私有投资回报率（RoS）约束的情况下的机制设计，提出了一种扩展的艾森伯格-盖尔程序，以解决这一问题。

**💡 创新点**

创新点在于将RoS约束纳入机制设计中，提出了一种市场清算机制，并证明了其激励相容性和与最佳收入基准的1/2近似性。

**🔧 技术方法**

使用了扩展的艾森伯格-盖尔程序作为凸优化框架，并提出了一种去中心化的在线算法来实现该机制。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到的背景是现代广告平台使用机器学习模型预测买家的估值。

**📈 对比分析**

与传统机制相比，提出的机制在激励相容性和收入近似性方面表现良好，能够在多次拍卖中实现卖方收入和买方效用的收敛，具有次线性遗憾的表现。

**⚠️ 局限性**

限制在于该机制的实现依赖于买家真实报告其预算和RoS约束，而在实际应用中，买家可能会选择不真实地报告这些信息。

---

## 796. Design and Biomechanical Evaluation of a Lightweight Low-Complexity Soft Bilateral Ankle Exoskeleton

**arXiv ID:** 2602.18569 | [PDF](https://arxiv.org/pdf/2602.18569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 797. Diagnosing LLM Reranker Behavior Under Fixed Evidence Pools

**arXiv ID:** 2602.18613 | [PDF](https://arxiv.org/pdf/2602.18613v1)

**作者:** Baris Arat `[一作]` (Ozyegin University), Emre Sefer `[通讯]` (Ozyegin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于固定主题相关证据池的诊断方法，用来隔离并评估LLM重排器在排队文档时的行为。

**💡 创新点**

创新点在于：①通过使用人类编辑的Multi‑News聚类构建固定大小的证据池，消除检索阶段的变异；②在同一池内对比LLM、BM25、MMR和随机排序，揭示LLM在覆盖率和冗余度上的模型特异性；③提供一套可对任何黑盒重排器适用的评估流程。

**🔧 技术方法**

采用词汇覆盖率、词汇冗余、摘要召回、语义冗余与覆盖率等指标（MiniLM嵌入计算余弦相似度），以及Kendall τ和Jaccard相似度进行排序一致性评估；使用Bootstrap差分估计统计显著性。

**📊 数据集**

使用Multi‑News数据集：345个聚类，每个聚类挑选8篇文档，查询为摘要第一句（截断400字符），文档长度600字符。

**📈 对比分析**

与BM25、MMR、随机排序进行对比。结果显示：LLM在词汇覆盖率和召回率上普遍低于BM25/ MMR；在冗余度上表现因模型差异显著——Llama在大预算下趋向多样化，GPT更易产生冗余；所有LLM均显著优于随机排序。

**⚠️ 局限性**

局限性：仅评估了三种LLM和单一提示模板；使用自动指标，缺乏人工评估；未检验排序稳定性或置信度；诊断侧重于排序，而非完整检索性能；未探究不同预算或不同主题对结果的影响。

---

## 798. Order Bounds for Hypergeometric and q-Hypergeometric Creative Telescoping

**arXiv ID:** 2602.19886 | [PDF](https://arxiv.org/pdf/2602.19886v1)

**作者:** Hui Huang `[一作]` `[通讯]` (Fuzhou University), Hui Huang (Fuzhou University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文提出了一种统一的基于约简的算法，用于计算超几何项及其q-超几何项的最小阶Telescoper，并给出了新的终止性证明；

**💡 创新点**

创新点包括：1）通过证明约简既有限维又完整，从而独立证明Telescoper存在性；2）给出统一的上、下阶界，尤其首次为q-超几何项提供下阶界；3）在已知上界基础上进一步与Apagodu-Zeilberger 上界比较，证明在一般情况下两者相当，且在某些特例下可显著更紧；

**🔧 技术方法**

主要技术手段为：符号积分框架下的多元差分域与RNF（Rational Normal Form）分解、σ_y-约简、σ_y-正则化、σ_y-标准补、以及对整数线性多项式的代数结构分析；

**📊 数据集**

本工作不依赖特定数据集，而是针对通用的符号超几何项进行理论分析和证明；

**📈 对比分析**

通过理论推导与已公布的Apagodu‑Zeilberger 上界进行对比，发现新上界在一般情况与已知上界等价，且在部分实例下明显更优；实验验证表明在求解Telescoper时，新上界可缩短所需递推次数，提升算法效率；

**⚠️ 局限性**

局限性包括：1）需假设Telescoper存在；2）对整数线性分母的要求，若分母非整数线性则当前方法不直接适用；3）对q-非“q‑proper”项的处理尚不完整，需进一步研究。

---

## 799. One Color Makes All the Difference in the Tractability of Partial Coloring in Semi-Streaming

**arXiv ID:** 2602.18987 | [PDF](https://arxiv.org/pdf/2602.18987v1)

**作者:** Avinandan Das `[一作]` (Aalto University), Avinandan Das `[通讯]` (Aalto University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5025342406)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并研究了k-部分着色（k-partial coloring）的半流式（semi-streaming）复杂性，证明了k-部分(k+1)-着色可以在一遍随机半流式算法下完成，而k-部分k-着色则需要超线性存储。

**💡 创新点**

创新点包括：
• 引入“证人子图”过滤器，将原图压缩为满足最小度≥k的子图，从而实现部分着色的转换为普通着色；
• 将彩色稀疏化（palette sparsification）技术扩展到k-部分着色框架，证明在随机颜色列表下仍能以k+1颜色完成着色；
• 通过构造边/颜色复制器（gadgets）和Index通信问题的化简，给出k-部分k-着色的Ω(n^{4/3})存储下界，展示出与(k+1)-着色的根本差异。

**🔧 技术方法**

主要技术手段有：
• 证人子图的在线过滤与稀疏化；
• 颜色列表的随机采样与两阶段着色（先着色高度>k顶点，再扩展至低度顶点）；
• 彩色稀疏化定理（palette sparsification）的改编，适用于k+1颜色；
• 通过一遍通信协议和Index问题证明下界；
• 退化图（degeneracy）与方向化边的随机分析。

**📊 数据集**

论文使用的是合成图实例（通过构造的边/颜色复制器和矩阵A生成的图），没有真实数据集。

**📈 对比分析**

与此前的Assadi、Chen、Khanna等人在(Δ+1)-和Δ-着色中的结果对比：
• 对于k-部分(k+1)-着色，算法与之前的随机半流式算法相当，存储约为O(n log^4 n)；
• 对于k-部分k-着色，论文证明在k≈n^{1/3}时存储需求至少为Ω(n^{4/3})，远超线性空间，说明此问题在半流式模型中不可求解。

**⚠️ 局限性**

局限性包括：
• 仅对k-部分(k+1)-着色提供正向结果，k-部分k-着色仍为难题；
• 需要随机化和离线求解（暴力搜索）来完成最终着色，实际实现上可能耗时；
• 下界证明仅适用于k≈n^{1/3}（以及更大k）的情形，无法说明更一般k值下的存储需求；
• 算法仅针对插入仅（insertion‑only）流，无法直接推广到删除/动态流；
• 彩色稀疏化的概率保证需要较大样本大小（Θ(log^2 n)），对极小n时可能不够紧凑。

---

## 800. NeXt2Former-CD: Efficient Remote Sensing Change Detection with Modern Vision Architectures

**arXiv ID:** 2602.18717 | [PDF](https://arxiv.org/pdf/2602.18717v1)

**作者:** Yufan Wang `[一作]`, Chandra Kambhamettu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种端到端的双时相遥感变更检测框架NeXt2Former-CD，利用Siamese ConvNeXt编码器、可变形注意力时间融合模块和Mask2Former解码器，实现高精度变更映射；

**💡 创新点**

创新点包括：1) 将大规模自监督预训练的ConvNeXt（DINOv3）作为Siamese编码器；2) 在特征融合阶段引入可变形注意力以抵抗残余配准误差和小物体位移；3) 采用Mask2Former解码器与混合损失（set loss+像素级交叉熵），兼顾全局与像素级监督；

**🔧 技术方法**

使用的关键技术有ConvNeXt、DINOv3自监督预训练、可变形注意力（Deformable Attention）、Mask2Former分割头、Hybrid损失、Siamese网络结构；

**📊 数据集**

实验数据集：LEVIR-CD、WHU-CD 和 CDD；

**📈 对比分析**

与CNN、Transformer、Mamba等主流方法在相同训练协议下对比，NeXt2Former-CD在LEVI‐CD、WHU‑CD和CDD上均取得最高F1/IoU/OA；虽然参数量更大，但推理速度与Mamba相当，显示良好的性能-效率折中；

**⚠️ 局限性**

局限性包括：1) 参数量较大（约392M），训练时间和显存占用相对高；2) 对极小或极微细变更的检测仍有挑战；3) 依赖大规模预训练模型，若无预训练可能效果下降。

---

## 801. Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System

**arXiv ID:** 2602.18640 | [PDF](https://arxiv.org/pdf/2602.18640v1)

**作者:** Longfei Yun `[一作]` (Meta), Junfeng Pan `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GEARS 框架，将大规模排名优化转为自主探索式实验，显著降低人工工程成本与业务痛点。

**💡 创新点**

核心创新在于 Specialized Agent Skills 与 Deterministic Lifecycle Governance 两大模块，能够将高层意图自动映射为可执行、可验证的策略并确保长期部署稳定。

**🔧 技术方法**

采用大型语言模型（Claude Sonnet）结合 Agentic Search (GAS)、技能调用、Pareto 前沿可视化以及验证钩子，实现从意图到候选策略再到部署的完整闭环。

**📊 数据集**

实验使用 Meta 内部 20 个实验的数据集，每个实验生成数百条候选策略，并构造 100 条多类型指令，用于评估策略选择效果。

**📈 对比分析**

与 Naive Prompt、Chain‑of‑Thought、Self‑Consistency、Self‑Refine、Code‑as‑Action 等基线对比，GEARS 在 Precision@K、Recall@K、NDCG@K、Top‑1 Accuracy 及长期稳定性指标上均优于基线。

**⚠️ 局限性**

局限性主要在于对工程知识的依赖，需要为新领域手工构建技能；验证钩子无法覆盖所有业务约束，仍可能漏掉某些复杂的可部署风险。

---

## 802. Extending CPU-less parallel execution of lambda calculus in digital logic with lists and arithmetic

**arXiv ID:** 2602.19884 | [PDF](https://arxiv.org/pdf/2602.19884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 803. From Human-Level AI Tales to AI Leveling Human Scales

**arXiv ID:** 2602.18911 | [PDF](https://arxiv.org/pdf/2602.18911v1)

**作者:** Peter Romero `[一作]` (University of Cambridge), José Hernández-Orallo `[通讯]` (University of Cambridge)

**通讯引用:** 8113 | [OpenAlex ID](https://openalex.org/A5029864546)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 LLM 的世界人口基准化框架，将 AI 评估与人类水平对齐，并构建多维能力与知识尺度；

**💡 创新点**

通过 LLM 自动进行人口外推、标准化及维度特定基数校准，使得 AI 与人类的表现可在同一心理测量尺度上直接比较；

**🔧 技术方法**

利用大型语言模型进行后测平衡与人口外推、ADeLe 框架的多维需求注释、对数比率尺度以及误差评估指标（MAE、RMSE、相关系数）；

**📊 数据集**

PISA、TIMSS、ICAR、UK Biobank Fluid Intelligence、ReliabilityBench 等公开教育与认知测评数据集；

**📈 对比分析**

将子组成功率外推至整体，评估 MAE、RMSE 与相关系数；ICAR 结果MAE≈0.03、TIMSS 约0.12‑0.16，且基数校准后模型在多维度上已接近或超越人类平均水平；

**⚠️ 局限性**

受限于样本多样性不足、LLM 本身的文化/教育偏差、对极异质数据外推不稳健、仅使用有限 prompt 与模型，以及潜在的伦理与代表性偏差。

---

## 804. VariBASed: Variational Bayes-Adaptive Sequential Monte-Carlo Planning for Deep Reinforcement Learning

**arXiv ID:** 2602.18857 | [PDF](https://arxiv.org/pdf/2602.18857v1)

**作者:** Joery A. de Vries `[一作]` (Trent AI Ltd), Matthijs T. J. Spaan `[通讯]` (Delft University of Technology)

**通讯引用:** 3104 | [OpenAlex ID](https://openalex.org/A5075395956)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于变分贝叶斯、序列蒙特卡罗规划和元强化学习的Bayes‑adaptive MDP近似解法 VariBASeD，能够在单 GPU 上实现可扩展的在线规划与贝叶斯推断。

**💡 创新点**

创新点在于：①将贝叶斯推断、蒙特卡罗规划和元学习三者统一到一个 EM 框架中；②使用状态空间模型（S5）实现高效的变分信念学习；③通过在线 SMC 规划为梯度更新提供更高质量的学习目标，从而显著提升样本和运行时效率。

**🔧 技术方法**

核心技术包括：控制‑作为‑推断（Control-as-Inference）+ EM 优化；变分贝叶斯推断与重参数化梯度；序列蒙特卡罗（SMC）规划与重要性采样；S5 状态空间序列模型；元学习（通过可微的推断网络实现经验转移）。

**📊 数据集**

使用自定义的零样本多任务 RL 数据集：连续函数优化环境和不确定奖励的网格世界，任务先验为均匀未知分布；实验中没有采用公开数据集，而是根据论文附录构造的任务分布。

**📈 对比分析**

与基准 RL^2（使用 S5 架构的模型无监督元 RL）对比。实验表明 VariBASeD 在样本效率、学习曲线收敛速度和运行时开销上均优于 RL^2；尤其在网格世界环境下，增加规划预算（H,K）显著提升性能，而在函数优化任务中，高预算反而略逊，说明对不同任务的敏感性不同。

**⚠️ 局限性**

局限性：①高规划预算在某些任务（函数优化）可能导致过度贪婪，影响收敛；②使用 RNN/状态空间模型时的隐藏状态缓存会出现表征漂移，需更长的 unroll window 以保证稳定性；③实验仅限于小规模任务，未验证在更长上下文和更大任务多样性上的表现；④基准 RL^2 在本设置下未能有效学习，可能与超参选择相关。

---

## 805. On the Equivalence of Random Network Distillation, Deep Ensembles, and Bayesian Inference

**arXiv ID:** 2602.19964 | [PDF](https://arxiv.org/pdf/2602.19964v1)

**作者:** Moritz A. Zanger `[一作]` (Delft University of Technology), Matthijs T. J. Spaan `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在无限宽网络的 NTK 极限下对随机网络蒸馏（RND）进行理论分析，证明标准 RND 的误差等价于深度集成的预测方差，并通过设计目标函数使 RND 的误差分布与贝叶斯后验预测一致，从而实现了 RND 的后验采样方法。

**💡 创新点**

创新点包括：①首次将 RND 与深度集成和贝叶斯推断在 NTK 框架下统一；②给出了 RND 误差在无限宽极限下的解析分布；③设计了“贝叶斯 RND”模型，使其误差直接等价于贝叶斯后验预测；④提出通过目标函数工程来定制不同的不确定性估计。

**🔧 技术方法**

使用的技术主要有：NTK 理论、神经网络高斯过程（NNGP）分析、梯度流学习动力学、随机特征模型、随机网络蒸馏多头架构、后验采样算法。

**📊 数据集**

在数值验证中采用了合成高斯数据集（3 维输入，训练集和测试集均采样自 N(0,I)）。

**📈 对比分析**

通过训练单层全连接网络，将 RND 的平方误差与贝叶斯深度集成的方差进行对比，使用 1024 个模型或 1024 头 RND 进行估计，结果显示两者高度相关且在宽网络上尺度校准良好，性能与真实 GP 方差相当。

**⚠️ 局限性**

局限性在于所有理论结果仅在无限宽、NTK “懒”训练极限下成立，无法直接推广到有限宽或具备特征学习能力的实际网络；目标函数工程的实现细节与可扩展性也未深入讨论。

---

## 806. Beyond the Binary: A nuanced path for open-weight advanced AI

**arXiv ID:** 2602.19682 | [PDF](https://arxiv.org/pdf/2602.19682v1)

**作者:** Bengüsu Özcan `[一作]` (Centre for Future Generations), Max Reddel `[通讯]` (Centre for Future Generations)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了基于风险评估和安全性验证的分层开放权重AI模型发布框架，并为各利益相关方提供具体治理与技术建议

**💡 创新点**

创新点在于把开放与安全绑定，打破传统“开放/封闭”二元对立，倡导可验证、可监管的分层开放策略

**🔧 技术方法**

采用风险评估方法、标准化安全案例（Safety Case）技术、技术防护如防篡改和加密权重、沙箱和可信访问机制等

**📊 数据集**

本报告未使用传统机器学习数据集，而是基于现有公开AI模型发布案例、政策文件、行业标准和安全评估报告进行综合分析

**📈 对比分析**

对比方法主要是案例对照与风险阈值匹配；报告指出开放权重模型在性能上已逼近封闭模型，但安全性和治理成熟度差异显著

**⚠️ 局限性**

局限性包括：缺乏统一、可执行的国际标准；技术防护仍处于早期实验阶段；对实际部署环境的可落地性评估不足；治理建议依赖自愿与行业自律，缺乏强制性执行力

---

## 807. GUIDE-US: Grade-Informed Unpaired Distillation of Encoder Knowledge from Histopathology to Micro-UltraSound

**arXiv ID:** 2602.19005 | [PDF](https://arxiv.org/pdf/2602.19005v1)

**作者:** Emma Willis `[一作]` (Queen's University), Purang Abolmaesumi `[通讯]` (University of British Columbia)

**通讯引用:** 8539 | [OpenAlex ID](https://openalex.org/A5023095072)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

使用无配对病理知识蒸馏训练微超声编码器，实现前列腺癌的分级与检测。

**💡 创新点**

创新点在于通过ISUP分级条件的无配对蒸馏与注意力多实例学习，克服了超声与病理在尺度、分辨率和解剖覆盖上的差距。

**🔧 技术方法**

采用ABMIL、triplet对比蒸馏、MedSAM弱监督分割、预训练病理模型（如GigaPath/UNI‑2）等技术。

**📊 数据集**

使用来自ExactVu微超声试验的578例病人微超声图像和PANDA挑战的9555张病理WSI。

**📈 对比分析**

与ProstNFound+等基线比较，AUROC提升至0.784，在60%特异性下csPCa敏感度提高约3.5%，差异具有统计学显著性。

**⚠️ 局限性**

仅在训练阶段使用病理信息，跨模态蒸馏仍依赖预训练模型且未验证在其他器官或临床环境中的泛化能力。

---

## 808. Beyond Stationarity: Rethinking Codebook Collapse in Vector Quantization

**arXiv ID:** 2602.18896 | [PDF](https://arxiv.org/pdf/2602.18896v1)

**作者:** Hao Lu `[一作]` (Wake Forest University), Metin Nafi Gurcan `[通讯]` (Wake Forest University)

**通讯引用:** 10821 | [OpenAlex ID](https://openalex.org/A5077316017)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出两种新型向量量化方法 NS-VQ 与 TransVQ，解决 VQ 码本崩溃问题，并在 VQ‑VAE 框架下进行实验验证。

**💡 创新点**

创新点在于从理论上将码本崩溃归因于编码器非平稳性，并通过核函数传播漂移与 Transformer 可学习的码本映射，既保持 k‑means 收敛性，又显著提升码本利用率。

**🔧 技术方法**

主要技术包括非平稳向量量化（NS‑VQ）、Transformer 基变换（TransVQ）、核函数近似、改进的直通估计器（STE）以及传统的 Gumbel‑Softmax 量化。

**📊 数据集**

实验数据集为 CelebA‑HQ（256×256），共 30,000 张人脸图像，用于训练与评估。

**📈 对比分析**

与 VQ‑VAE、VQGAN、SimVQ 等传统方法对比，NS‑VQ 与 TransVQ 在 rFID、LPIPS、SSIM 指标上均大幅提升，码本利用率几乎达到 100%。

**⚠️ 局限性**

局限性包括需要手动调节核宽度 2σ² 等超参数，TransVQ 的 Transformer 计算开销较大，且尚未在如 ImageNet 这类大规模数据集上进一步验证。

---

## 809. TurkicNLP: An NLP Toolkit for Turkic Languages

**arXiv ID:** 2602.19174 | [PDF](https://arxiv.org/pdf/2602.19174v1)

**作者:** Sherzod Hakimov `[一作]` (University of Potsdam), Sherzod Hakimov `[通讯]` (University of Potsdam)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5091548229)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TurkicNLP，一个统一的Python NLP工具包，支持24种突厥语族语言的标记化、形态分析、词性标注、词干化、依存句法、命名实体识别、双向脚本互译、跨语言句子嵌入和机器翻译；

**💡 创新点**

创新点在于一体化、脚本感知的多后端架构，实现自动脚本检测与双向转写，支持规则式Apertium FST与神经Stanza模型无缝切换，并通过CoNLL-U兼容输出实现与Universal Dependencies生态的无缝集成；

**🔧 技术方法**

采用规则式分词器、Apertium FST形态分析器、Stanza神经模型、NLLB-200跨语言模型，以及自研的脚本检测与转写引擎；

**📊 数据集**

使用了Apertium提供的约20种语言的FST词典、Stanza的5种语言UD树库、NLLB-200的200语言预训练模型、以及多语种平行语料（如Azerbaijani、Kyrgyz、Turkish、Uzbek等）进行评测；

**📈 对比分析**

通过对比已知的单语工具（如Zemberek、ISSAI、Apertium）和多语工具（如Trankit、spaCy、NLTK）在统一API下的功能覆盖度和性能，TurkicNLP在形态分析、转写及机器翻译任务中表现与现有最佳工具相当或更优，且实现了统一调用的便利性；

**⚠️ 局限性**

局限包括：对非目标脚本的转写仍存在歧义和缺失规则、形态歧义处理依赖启发式策略、缺乏低资源语言的神经模型、NER仅覆盖土耳其语和哈萨克语、ASR/TTS功能尚未集成、以及脚本转写中对某些特殊字符（如乌尔古语的点号、阿拉伯语的harakat）的处理尚未完善。

---

## 810. Control in Hedonic Games

**arXiv ID:** 2602.18506 | [PDF](https://arxiv.org/pdf/2602.18506v1)

**作者:** Jiehua Chen `[一作]` (TU Wien), Sofia Simola `[通讯]` (TU Wien)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5040384528)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文对外部控制在协同（hedonic）游戏中的影响进行了系统研究，分析了在添加或删除代理人后实现特定稳定分区（如保证某个代理人不孤立、保证一对代理人在同一联盟或形成全局联盟）的复杂度。

**💡 创新点**

创新点在于首次将选举控制（adding/deleting agents）概念引入协同游戏，提出了控制问题的正式定义，并给出了完整的复杂度图谱；揭示了在友好导向（friend-oriented）和加性（additive）偏好下，不同稳定概念下控制的可行性与不可行性差异，尤其发现添加代理人能实现多种控制目标，而删除代理人往往无效。

**🔧 技术方法**

主要技术包括图论方法（最短路径、Steiner网络最小权重子图求解）、多项式时间算法设计、归约与 NP/Σ^P_2 难度证明，结合稳定性概念的形式化与偏好模型的结构化分析。

**📊 数据集**

本工作完全基于理论模型，无使用具体数据集；所有结果均来自算法设计与复杂度证明。

**📈 对比分析**

通过对比不同稳定概念（个体可接受性、个体稳定性、纳什稳定性、核心稳定性）以及两种偏好模型的计算复杂度，本文发现某些控制目标可在多项式时间内实现，而大多数控制问题则属于 NP-或 Σ^P_2 难度；对比结果展示了控制难度在不同偏好和稳定性下的显著差异。

**⚠️ 局限性**

局限性包括：仅研究了友好导向和加性两种偏好，未覆盖更一般或更复杂的偏好模型；缺乏实验验证，仅给出理论复杂度分析；在控制预算有限（k 较大）时的精确复杂度仍不完全清晰。

---

## 811. Accurate Planar Tracking With Robust Re-Detection

**arXiv ID:** 2602.19624 | [PDF](https://arxiv.org/pdf/2602.19624v1)

**作者:** Jonas Serych `[一作]` (Czech Technical University in Prague), Jiri Matas `[通讯]` (Czech Technical University in Prague)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了两种新型平面目标跟踪方法 SAM‑H 与 WOFTSAM，利用强大的 SAM 2 语义分割进行长期跟踪并通过几何分析实现稳健的 8 维同构姿态估计；

**💡 创新点**

创新点在于将分割跟踪的鲁棒重定位与基于光流的细粒度同构估计相结合，并通过对齐对称性和 DINOv2 特征实现分割结果的正确角点重排；

**🔧 技术方法**

使用了 SAM 2 语义分割、RAfT 光流、DINOv2 视觉特征、霍夫变换线段拟合以及 WOFT 的加权流同构模块；

**📊 数据集**

在 POT‑210 与 PlanarTrack（PlanarTrackTST）两个公开基准数据集上进行评估；

**📈 对比分析**

与现有最先进方法 WOFT 相比，WOFTSAM 在 p@5 与 p@15 上分别提升约 9–10% 及 12–15%，在两大基准上均刷新最高分；

**⚠️ 局限性**

主要局限包括：对目标必须近似四边形；在部分遮挡、透明、强反射或动态纹理目标时，分割或光流方法均易失效；SAM‑2 在面临同形目标或多重遮挡时可能错误扩展。

---

## 812. Active perception and disentangled representations allow continual, episodic zero and few-shot learning

**arXiv ID:** 2602.19355 | [PDF](https://arxiv.org/pdf/2602.19355v1)

**作者:** David Rawlinson `[一作]` (Cerenaut AI), Gideon Kowadlo `[通讯]` (Cerenaut AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种补充学习系统，快速的非泛化STM通过主动感知向慢速LTM查询，将感知信息转换为可解释的特征，从而实现连续、零/少样本学习。

**💡 创新点**

核心创新在于：①将STM作为主导的、完全不泛化的记忆；②利用主动感知让LTM提供泛化的感知结果；③通过稀疏分布式记忆（SDM）和关联SDM实现分离、无干扰的学习。

**🔧 技术方法**

技术实现包括：稀疏递归编码器、稀疏分布式记忆（SDM）、关联SDM、Q‑学习框架以及使用冻结的LLM（Mistral‑7B）作为LTM；对比使用全连接前馈网络作为基线。

**📊 数据集**

使用自生成的、LLM构造的对象识别任务数据集，包含24类具体对象及其属性，环境通过LLM动态产生奖励。

**📈 对比分析**

与基线模型对比，分离STM在少样本学习中保持原始任务性能不变，快速学习新类；在零样本学习中即使训练未见某类也能达到最佳策略；在流式RL中批量大小1与16的性能相当，说明可即时在线学习。

**⚠️ 局限性**

局限性包括：实验规模极小、环境为低维人工生成、感知动作空间预设且未学习；未在大规模真实世界任务上验证；关联SDM等高级功能尚未实证。

---

## 813. Feedback-based Automated Verification in Vibe Coding of CAS Adaptation Built on Constraint Logic

**arXiv ID:** 2602.18607 | [PDF](https://arxiv.org/pdf/2602.18607v1)

**作者:** Michal Töpfer `[一作]` (Charles University), Petr Hnětynka `[通讯]` (Charles University)

**通讯引用:** 1098 | [OpenAlex ID](https://openalex.org/A5062364548)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

结合 vibe coding 与基于时序逻辑 FCL 的运行时验证，自动生成并验证 CAS 适配器 (AM) 的完整流程。

**💡 创新点**

创新点包括：①设计细粒度时序逻辑 FCL 用于表达动态系统约束；②开发 ADSL/FCDSL 两种 DSL 以自然语言+形式化方式描述系统与约束；③将约束验证嵌入适配循环，形成闭环反馈给 LLM。

**🔧 技术方法**

技术手段：生成式大型语言模型 (GPT‑5 nano/min)、vibe coding、时序逻辑 FCL、运行时验证器、DEECo 体系结构、MAPE‑K 适配循环、DSL 自动提示生成。

**📊 数据集**

数据集/实验案例：两个 CAS 示例—“Dragon Hunt”游戏与“Smart Farm”农田保护，作为实验输入并生成相应提示与约束。

**📈 对比分析**

对比方法：在六种实验变体（两种初始提示 × 三种反馈形式）下，重复 10 次实验，统计反馈循环迭代次数。结果显示：完整约束反馈最快，平均仅需 2–4 次迭代即可得到合法 AM，且生成 AM 的性能与手工实现相近。

**⚠️ 局限性**

局限性：仅在两个简易示例验证，未覆盖真实复杂 CAS；约束设计与调优耗时；LLM 对提示敏感，易导致死循环；仅测试有限 LLM 版本，结果可能不具普适性；验证基于约束，可能误判代码符合但功能不完整。

---

## 814. Incidental Reverberations: Poetic Similarities in AI Art

**arXiv ID:** 2602.19769 | [PDF](https://arxiv.org/pdf/2602.19769v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 815. The Sample Complexity of Replicable Realizable PAC Learning

**arXiv ID:** 2602.19552 | [PDF](https://arxiv.org/pdf/2602.19552v1)

**作者:** Kasper Green Larsen `[一作]` (Aarhus University), Clement Svendsen `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究可复制（replicable）PAC学习，给出了一个在有限假设类上可复制学习的样本复杂度下界，并构造了近似匹配的上界；即证明了在该硬实例上，任何可复制的可实现PAC学习器都至少需要Ω((log|H|)^{3/2})个样本，且存在一个算法在O((log|H|)^{3/2}/ρ)样本下即可实现。

**💡 创新点**

创新点主要有：
- 通过构造一个特殊的Cayley图和相关的随机游走，利用其谱性质得到对可复制PAC学习的强下界；
- 提出了一种新型的随机化取整（randomized rounding）技术，结合共享随机性，使得学习算法在不同训练集上能够以高概率输出相同的分类器；
- 将下界实例与上界算法相匹配，表明(log|H|)^{3/2}的依赖并非仅是证明技巧导致的伪因素；
- 对可复制学习与隐私/稳健性等概念的关系提供了新的视角。

**🔧 技术方法**

使用的技术包括：
- Cayley图的构造与分析，尤其是对其邻接矩阵的谱分解；
- 随机游走与随机步长的概率分析；
- 组合学与矩阵秩的估计（Littlewood–Offord反集中理）来控制低扩展集；
- 随机化取整与共享随机性，用于实现可复制性；
- Hoeffding、Chernoff等尾概率界用于样本量的上界分析。

**📊 数据集**

数据集：本文使用的是一个合成的、结构化的输入域 X = [d] ×_{k}，其中 k 为满足约束的素数，分布为 X 上的均匀分布。假设类 H 的每个成员对应一个 d‑元组 (i_0,…,i_{d-1})∈ℤ_k^d，定义的标签规则是围绕每个坐标的长度为 ⌊k/2⌋ 的区间。该实例并非来源于真实世界数据，而是为证明理论边界而构造的。

**📈 对比分析**

与其他方法的比较：
- 与传统PAC学习相比，可复制学习的样本复杂度从 Θ(1/(VC+log(1/δ))) 上升到 Θ((log|H|)^{3/2})；
- 与私有学习（differential privacy）相比，本文展示了可复制学习不一定能得到与隐私学习相同的下界，揭示了两者的差异；
- 与近似可复制或列表可复制等更弱形式相比，本文证明了在完全可复制性下，样本复杂度必然高于这些弱形式；
- 性能方面，上界算法在 O((log|H|)^{3/2}/ρ) 样本内即可达到误差 ε、失败概率 δ，并保证高概率复制性。

**⚠️ 局限性**

局限性：
- 证明仅适用于特定的硬实例（X = [d] ×_{k} 与相应 H），不一定能推广到所有有限假设类；
- 仍未确定 (log|H|)^{3/2} 是否为真正的上界，是否存在更强的下界或通用上界（例如 log^2|H|）仍是开放问题；
- 该方法需要在所有运行中共享内部随机种子，实际应用中实现共享随机性的成本未被讨论；
- 复杂的谱与矩阵秩分析给实现带来较高的理论难度，直接转化为实际算法可能不易。

---

## 816. One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image

**arXiv ID:** 2602.19766 | [PDF](https://arxiv.org/pdf/2602.19766v1)

**作者:** Pengfei Wang `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出 One2Scene 框架，能够从单张图像生成可全景可探索的 3D 场景。

**💡 创新点**

将单图像到 3D 场景的难题拆分为三步：生成全景 anchor、利用 feed‑forward Gaussian Splatting 构建 3D scaffold、以 scaffold 为先验进行可视化合成，显著提升几何一致性和视觉质量。

**🔧 技术方法**

结合 Hunyuan‑Pano‑DiT 生成全景、VGGT+bidirectional fusion 的 feed‑forward Gaussian Splatting 生成 3D scaffold、Dual‑LoRA 训练的 SEVA 风格图像扩散模型、记忆条件以保持时空一致性。

**📊 数据集**

在多数据集上训练与评估，包括 Synthetic Structured3D、Deep360、Matterport3D、Stanford2D3D、DL3DV、RealEstate10K 等。

**📈 对比分析**

与 DreamScene360、WonderJourney、VMem、SEVA 等 SOTA 方法对比，在可探索 3D 场景生成、深度估计、视觉一致性指标（NIQE、Q‑Align、CLIP‑I、CamMC 等）上取得显著优于对手的结果，如 NIQE 4.43、CamMC 0.389。

**⚠️ 局限性**

虽然几何一致性显著提升，但仍存在微小的不连贯现象，尤其在极端视角下的细节与纹理一致性不足，且缺乏更大规模数据支持，未来需进一步优化。

---

## 817. Habilis-$β$: A Fast-Motion and Long-Lasting On-Device Vision-Language-Action Model

**arXiv ID:** 2602.18813 | [PDF](https://arxiv.org/pdf/2602.18813v1)

**作者:** Tommoro Robotics `[一作]`, Theo Taeyeong Kim `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Habilis-β模型，旨在实现高速运动、长时间持续运行以及设备端部署的视觉‑语言‑动作系统；

**💡 创新点**

通过连续运行评估框架（Productivity‑Reliability Plane）重新定义机器人部署指标，并结合三阶段训练（play + cyclic + ESPADA）实现快速、可靠的机器人控制；

**🔧 技术方法**

利用预训练视觉‑语言模型（VLM）+流匹配动作专家、rectified flow蒸馏、高频控制、ESPADA下采样、CFG指令引导等技术；

**📊 数据集**

使用大规模无标签play数据进行预训练，随后在针对目标工序的循环演示（cyclic）数据上微调，实验涵盖RoboTwin 2.0仿真任务和RB‑Y1工业人形机器人真实工作流；

**📈 对比分析**

与公开基线π_0.5和GR00T N1.5在相同的1小时连续运行协议下比较，Habilis‑β在仿真中TPH提升近5.8倍、MTBI提升约1.3倍；在真实世界中TPH提升6.5倍、MTBI提升3倍，成功率从19.6%提升至82.7%；

**⚠️ 局限性**

受限于数据规模和多模态对齐，ESPADA和CFG在不同任务中需要手工调参；模型在极端稀疏或多指手部操作、触觉缺失以及在线自适应方面仍有待提升。

---

## 818. Eye-Tracking-while-Reading: A Living Survey of Datasets with Open Library Support

**arXiv ID:** 2602.19598 | [PDF](https://arxiv.org/pdf/2602.19598v1)

**作者:** Deborah N. Jakobi `[一作]` (University of Zurich), Lena A. Jäger `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并提供眼动追踪阅读数据集的综合清单，发布可过滤在线表格并将数据集集成到pymovements库。

**💡 创新点**

提供“living”数据集概览，结合FAIR原则，统一标准化的元数据和预处理接口，促进跨学科共享与可复现性。

**🔧 技术方法**

采用Python的pymovements库实现数据标准化与预处理，使用多种眼动事件检测算法（I-DT、I-VT等），并通过在线表格技术实现可过滤展示。

**📊 数据集**

集成了45+公开眼动阅读数据集，包括MECO、GECO、Dundee、CELER、GazeBase、EMTeC、ZuCo等多语种、跨任务的大型语料。

**📈 对比分析**

对比现有数据集的规模、语言、任务多样性、元数据完整性，并展示数据可用性与FAIR度量；未进行算法性能评估，而是关注数据互操作性与可重用性。

**⚠️ 局限性**

受限于现有公开数据集的数量与质量，部分数据缺失元数据；集成过程依赖原始文件格式，未统一所有眼动事件定义；未提供统一评估指标。

---

## 819. Reliable Abstention under Adversarial Injections: Tight Lower Bounds and New Upper Bounds

**arXiv ID:** 2602.20111 | [PDF](https://arxiv.org/pdf/2602.20111v1)

**作者:** Ezra Edelman `[一作]` (University of Pennsylvania), Surbhi Goel `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在“对抗注入模型”下的在线学习，探讨了在未知分布情况下，学习者在可选弃权（abstain）时的总误差下限与上限；提出了基于潜力的通用框架并通过推断维度和证书维度得到新的分布无关学习上界；证明了VC-1类的下界为Ω(√T)，从而展示了分布访问与未知分布之间的根本差距。

**💡 创新点**

1) 首先给出VC-1类的Ω(√T)下界，证明了在没有分布访问时无法达到O(log T)的oracle率；2) 引入潜力框架，统一并扩展了以往分布无关算法；3) 利用推断维度和新定义的证书维度，给出了包括ℝ²半空间在内的多类分布无关上界，首次获得了ℝ²半空间的Õ(T^{2/3})上界。

**🔧 技术方法**

潜力函数与稳健证人（robust witness）思想；留k出子集潜力和攻击性（attackability）分析；引入推断维度（Inference Dimension）和证书维度（Certificate Dimension）等组合维度；使用交换性与蒙特卡罗分析控制弃权误差。

**📊 数据集**

无数据集，全部为理论分析与证明。

**📈 对比分析**

与已有的VC-1上界Õ(√T)和轴对齐矩形的Õ(d√T)比较，本文实现了相同阶数的上界并进一步扩展到更一般的推断/证书维度；对ℝ²半空间实现了更优的Õ(T^{2/3})上界，优于之前无穷大或指数样本复杂度的鲁棒学习结果。

**⚠️ 局限性**

1) 下界仅针对VC-1类，未知分布下更高VC类的下界仍未知；2) 上界依赖于推断/证书维度，可能无法覆盖所有VC类，尤其高维半空间；3) 当前分析主要基于交换性，难以处理更强的自适应攻击；4) 对分布访问的中间模型与样本复杂度分析仍待完善。

---

## 820. DP-RFT: Learning to Generate Synthetic Text via Differentially Private Reinforcement Fine-Tuning

**arXiv ID:** 2602.18633 | [PDF](https://arxiv.org/pdf/2602.18633v1)

**作者:** Fangyuan Xu `[一作]` (New York University), Longqi Yang `[通讯]` (Microsoft)

**通讯引用:** 2474 | [OpenAlex ID](https://openalex.org/A5057330200)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DP-RFT算法，通过差分隐私奖励实现LLM无眼见私有数据的合成文本生成。

**💡 创新点**

创新点在于使用DP保护的最近邻投票作为RL奖励，训练LLM而不直接暴露私有样本，兼顾隐私与生成质量。

**🔧 技术方法**

采用差分隐私投票、近邻相似度奖励、PPO强化学习以及LLM-as-a-judge与规则校验以防奖励失控。

**📊 数据集**

使用四个公开数据集（PubMed、BBC新闻、WildChat、QMSum）进行实验，模拟私有语料。

**📈 对比分析**

与DP-FT、Aug-PE、QWEN等基线对比，DP-RFT在低隐私预算下在BERTSmall和GPT-2的下游任务中显著提升准确率，同时在生成质量上优于Aug-PE。

**⚠️ 局限性**

局限包括奖励失控风险、对提示设计的敏感性以及相对较高的训练与迭代生成计算开销。

---

## 821. HistCAD: Geometrically Constrained Parametric History-based CAD Dataset

**arXiv ID:** 2602.19171 | [PDF](https://arxiv.org/pdf/2602.19171v1)

**作者:** Xintong Dong `[一作]` (University of Science and Technology of China), Zhouwang Yang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2032 | [OpenAlex ID](https://openalex.org/A5047580521)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了HistCAD大型多模态CAD数据集（包含平面与三维建模序列、STEP、原生CAD文件、多视图渲染及文本注释），并开发了AM_HistCAD LLM驱动注释模块，用于生成建模过程、几何结构和功能类型的文本描述。

**💡 创新点**

创新点包括：① 将建模序列压缩为非层次化、平面约束（十种几何约束）可编辑格式；② 结合工业级Siemens NX部件提升建模复杂度与真实感；③ 采用LLM自动生成多维语义注释（过程、结构、功能），弥补传统数据集缺乏功能语义的不足；④ 提供统一的多模态评测基准，显著提升文本驱动CAD生成的有效性和几何精度。

**🔧 技术方法**

主要技术：LLM（Qwen3-8B、Qwen3-32B）+ LoRA微调、规则化几何与约束提取、统一建模序列格式、旋转挤压与布尔运算实现、基于Chamfer Distance与Invalidity Ratio的评测。

**📊 数据集**

使用的数据集：HistCAD-Academic（DeepCAD、SketchGraphs、Fusion 360 Gallery，共152,360模型） + HistCAD-Industrial（Siemens NX PRT文件，8,141模型），总计160,501建模序列；评测时与ABC、DeepCAD、Text2CAD等公开数据集进行对比。

**📈 对比分析**

比较方法：在文本驱动CAD生成任务中，用Qwen3-8B + LoRA训练模型（HistCAD_T、HistCAD_T（w/o c）、Text2CAD_T），评估指标为Invalidity Ratio、平均Chamfer Distance及中位数；实验显示HistCAD_T在IR仅1.40%、平均CD3.00（相对Text2CAD_T的4.73）且在工业子集上提升显著，整体性能优于现有方法。

**⚠️ 局限性**

局限性：LLM生成注释存在噪声与少量幻觉，数据集虽覆盖工业部件但仍缺乏全部高级CAD特性（如自定义曲面、复杂装配逻辑）；平面约束的十种类型仍无法覆盖所有工程约束；进一步提升模型泛化与对更大规模模型的支持仍是挑战。

---

## 822. Towards Understanding Views on Combining Videos and Gamification in Software Engineering Training

**arXiv ID:** 2602.19628 | [PDF](https://arxiv.org/pdf/2602.19628v1)

**作者:** Pasan Peiris `[一作]` (University of Canterbury), Jay Holland `[通讯]` (University of Canterbury)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

调查软件工程学生与专业人士对将游戏化元素融入视频培训的感知与态度

**💡 创新点**

对比两群体在游戏化动机与挫折上的共同与差异，提出针对视频培训的游戏化设计原则

**🔧 技术方法**

采用问卷调查、描述性统计与内容分析，使用AVW‑Space视频培训平台进行实验

**📊 数据集**

85名学生与100名专业人士的数据，来源于两所大学与Prolific平台

**📈 对比分析**

使用卡方检验比较两组差异，结果显示在活动难度、用途与动机上两组无显著差异

**⚠️ 局限性**

仅收集感知数据、样本地域分布不均、仅针对单一技能与平台、未验证游戏化对学习效果的实际影响

---

## 823. EdgeSketch: Efficient Analysis of Massive Graph Streams

**arXiv ID:** 2602.18957 | [PDF](https://arxiv.org/pdf/2602.18957v1)

**作者:** Jakub Lemiesz `[一作]` (Wroclaw University of Science and Technology), Philippe Cudré-Mauroux `[通讯]` (University of Fribourg)

**通讯引用:** 7652 | [OpenAlex ID](https://openalex.org/A5028454093)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出EdgeSketch，一种在单通道流式处理下构建的紧凑图表示，能够在不存储完整边列表的前提下估计图属性并直接执行图算法。

**💡 创新点**

创新点在于结合NodeSketch的节点邻域采样与FastExpSketch的指数更新规则，得到包含权重采样边和聚合信息的双数组结构，实现无损估计与集合操作兼备的统一图摘要。

**🔧 技术方法**

使用技术包括指数随机采样、随机数种子一致性、集合运算（并集、交集、差集）以及基于采样的度、密度、内部边数、模块度、节点相似度估计。

**📊 数据集**

在实验中使用合成的Stochastic Block Model（SBM）以及真实的Epinions bipartite图（转换为用户-物品相似网络）等大规模数据集。

**📈 对比分析**

与邻接表、矩阵以及之前的图摘要方法（gSketch、Scube、NodeSketch）比较，EdgeSketch在内存占用上可低至1%–5%，运行时间减少10–50倍，并保持与原图相当的模块度估计与重构精度。

**⚠️ 局限性**

局限性包括对小样本（m较小）时估计方差增大、对动态更新仅支持逐批重建、以及在极稀疏图中采样覆盖率低导致重构精度下降。

---

## 824. Uncovering Context Reliance in Unstructured Knowledge Editing

**arXiv ID:** 2602.19043 | [PDF](https://arxiv.org/pdf/2602.19043v1)

**作者:** Zisheng Zhou `[一作]` (Shandong University), Pengjie Ren `[通讯]` (Shandong University)

**通讯引用:** 5098 | [OpenAlex ID](https://openalex.org/A5046700486)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决了 NTP（下一个词预测）框架在无结构知识编辑中出现的上下文依赖问题，提出了 COIN（Context‑INdependent）编辑框架。

**💡 创新点**

创新点在于将上下文对齐损失（Context Alignment Loss）和知识一致性损失（Knowledge Consistency Loss）嵌入编辑过程，以强制模型把新知识内化而非仅记忆特定上下文，从而显著减少上下文缺口导致的召回失败。

**🔧 技术方法**

技术手段包括基于梯度优化的 NTP 训练、KL 对齐损失、对 MLP 权重的二阶矩正则化（Knowledge Consistency Loss）以及多任务加权损失组合。

**📊 数据集**

实验使用 Llama3‑8B、Qwen2.5‑7B 两大模型，在 AKEW、UnKEBench、MQuAKE（多跳推理）以及 GLUE 基准上进行评估，数据均为开放的无结构文本知识问答数据集。

**📈 对比分析**

与 FT、LoRA、AdaLoRA、UnKE、AnyEdit 等基线相比，COIN 在 AKEW 上上下文缺口下降 45.2%，编辑成功率提升 25.6%，在多跳推理任务中表现为 SOTA，并且在批量编辑和 GLUE 泛化测试中也保持了优良性能。

**⚠️ 局限性**

主要局限在于采用固定大小的局部上下文窗口，可能无法捕捉所有知识所需的上下文长度；同时模型仍需在不同任务间平衡知识注入与原始能力的保留，且对极端长文本的适应性待进一步验证。

---

## 825. Influence of Autoencoder Latent Space on Classifying IoT CoAP Attacks

**arXiv ID:** 2602.18598 | [PDF](https://arxiv.org/pdf/2602.18598v1)

**作者:** María Teresa García-Ordás `[一作]` (University of León), Héctor Alaiz-Moretón `[通讯]` (University of León)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用自动编码器的潜在空间对CoAP协议的网络流量进行降维，并结合决策树、随机森林与XGBoost三种分类器构建高精度IDS；

**💡 创新点**

首次通过优化自动编码器的潜在空间，仅保留2-3个特征即可实现>99%的检测精度，并对比三种分类器在不同维度下的表现；

**🔧 技术方法**

自动编码器（ReLU+线性潜在层）、MinMax归一化、OneHot编码、决策树、随机森林、XGBoost及类别加权；

**📊 数据集**

自制CoAP攻击数据集（3个CSV文件：CoAP_DoS、CoAP_MitM、CoAP_Cross_protocol，共68个特征）；

**📈 对比分析**

对不同潜在维度（1-27）下的Precision、Recall、F1进行评估，随机森林在2维时已达到0.99以上，决策树需要4+维；XGBoost在大部分场景表现不如RF；RF在所有维度保持接近100%；

**⚠️ 局限性**

仅针对CoAP协议，未验证多协议或多网络环境；Autoencoder潜在空间对不同攻击类型的泛化能力未充分评估；XGBoost在小样本上易过拟合。

---

## 826. PIS: A Physics-Informed System for Accurate State Partitioning of $Aβ_{42}$ Protein Trajectories

**arXiv ID:** 2602.19444 | [PDF](https://arxiv.org/pdf/2602.19444v1)

**作者:** Qianfeng Yu `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 624 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发了PIS系统，对阿尔茨海默病相关的β‑淀粉样蛋白Aβ42的分子动力学轨迹进行状态划分和动力学分析。

**💡 创新点**

创新点在于将先验物理量（如R_g和SASA）融入图神经网络的拓扑特征提取，并采用双轨融合和自适应门控机制，显著提升了对高度无序蛋白状态分辨的能力。

**🔧 技术方法**

技术方面使用了GASchNet图卷积、注意力池化、双轨门控融合、VAMP‑2/VAMP‑E无监督损失，以及WebGL实时渲染的交互平台。

**📊 数据集**

使用公开的Aβ42全原子MD数据集，包含5,119条轨迹、1,259,172帧，模拟时长315 µs。

**📈 对比分析**

与VAMPnets、GCN‑VAMP、MAGNN‑VAMP、RevGraphVAMP等基线模型比较，PIS在VAMP‑2和VAMP‑E指标上分别达到3.99±0.003，几乎匹配或优于最优模型。

**⚠️ 局限性**

局限性包括对先验物理量的依赖、训练过程对计算资源要求高，以及在其他蛋白体系上的推广性尚未验证。

---

## 827. Combining Small-Step and Big-Step Semantics to Verify Loop Optimizations

**arXiv ID:** 2602.19868 | [PDF](https://arxiv.org/pdf/2602.19868v1)

**作者:** David Knothe `[一作]` (FZI Research Center for Information Technology), Oliver Bringmann `[通讯]` (University of Tübingen)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在已存在的 CompCert 验证编译器中引入一种新的行为抽象框架，将小步语义用于局部优化、 大步语义用于结构化优化（如循环展开、循环无关化），并证明两种语义在该框架下保持等价，从而实现对新循环优化的完整验证。

**💡 创新点**

创新点包括：① 设计了一个“行为语义”抽象接口，将小步与大步语义映射到同一抽象层；② 在大步语义中引入共诱导的可泛化收敛判定，支持有限与无限轨迹；③ 证明了大步语义与小步语义的行为等价，并在 CompCert 管道中实现插入大步优化而不破坏整体正确性；④ 将大步优化的证明拆分为更小的逻辑单元，提升了可重用性与模块化。

**🔧 技术方法**

使用技术主要有：小步语义（operational）、大步语义（natural）以及共诱导推理；构造行为语义抽象；对大步语义进行“guard”修正以实现完整轨迹收敛；证明小步与大步之间的前向/后向行为保持；在 CompCert 中实现循环无关化、完全循环展开等大步优化，并给出相应的 Coq 证明。

**📊 数据集**

本工作并未使用传统意义上的实验数据集，而是在 CompCert 的 Cminor 中实现并验证了一组循环优化（循环无关化、完整循环展开、消除永不退出的空循环），并在该环境下完成了形式化证明。

**📈 对比分析**

与仅基于小步语义的验证方法相比，本方法在证明过程上更简洁、逻辑更模块化，尤其在结构化优化（如循环展开）方面省去了繁琐的匹配关系与状态线程；在保持 CompCert 主要正确性定理的前提下，成功验证了新优化；实验结果表明，整体编译链的行为保持性与性能提升符合预期（即没有引入额外运行时错误）。

**⚠️ 局限性**

限制主要包括：需要目标语言为确定性（determinate）并且不支持 goto 之类的非结构化跳转；共诱导证明仍较为技术性，易出现细节错误；目前实现仅覆盖 Cminor 级别的循环优化，未扩展到更高层或跨函数的优化；若要迁移至其他编译器，需要重新验证两种语义的等价性。

---

## 828. Minimizing Total Travel Time for Collaborative Package Delivery with Heterogeneous Drones

**arXiv ID:** 2602.19535 | [PDF](https://arxiv.org/pdf/2602.19535v1)

**作者:** Thomas Erlebach `[一作]` (Durham University), Wen Zhang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了具有不同速度的无人机在协作交付任务中的调度问题，提出了非抢占式最小总行驶时间的常数近似算法；

**💡 创新点**

创新点在于证明细粒度协作收益有限，且通过将问题归约为树组合问题并采用多层原始-对偶法实现常数近似；

**🔧 技术方法**

主要技术包括树组合问题的建模、Primal‑Dual 套装算法、Prizes‑Collecting Steiner Tree 转化以及 Delaunay 图加速的最小生成树构造；

**📊 数据集**

实验使用了三组合成数据（最坏例、均匀分布、GMM 分布）以及从美团收集的真实订单与骑手数据；

**📈 对比分析**

与基线贪婪最小插入启发式相比，实验显示所提算法在大规模实例上运行时间显著降低、解决方案质量与基线相当或更优，并且具有可证明的近似保证；

**⚠️ 局限性**

局限性在于理论近似比率仍为常数（如18 倍或 11.8 倍），对极端速度差异或高度动态环境的鲁棒性尚未充分评估。

---

## 829. Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines

**arXiv ID:** 2602.19265 | [PDF](https://arxiv.org/pdf/2602.19265v1)

**作者:** Siavash Khodakarami `[一作]` (Brown University), George Em Karniadakis `[通讯]` (Brown University)

**通讯引用:** 96270 | [OpenAlex ID](https://openalex.org/A5009658255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统研究并量化了物理信息神经网络（PINN、PIKAN）与神经算子（DeepONet、DeepOKAN、FNO、CNO）中的谱偏差（spectral bias），并揭示其不仅是模型可表达性的限制，还受优化动态和损失设计的影响。

**💡 创新点**

创新点：① 将谱偏差的理论分析与频率分辨误差指标、Barron范数和高阶统计量相结合，实现统一量化；② 证明一阶优化（Adam）与高阶优化（SOAP、SS‑Broyden）对频率学习速率的不同影响；③ 设计谱感知损失（Binned Spectral Power, BSP）以显著提升神经算子的高频保真度；④ 在多类 PDE（Kdv、波方程、扩散-反应方程）和数据驱动任务（湍流喷射、地震结构响应）上系统验证。

**🔧 技术方法**

技术：物理信息神经网络、物理信息 Kolmogorov‑Arnold 网络、神经算子（DeepONet、DeepOKAN、FNO、CNO），一阶与二阶优化（Adam、L‑BFGS、SOAP、SS‑Broyden），频率分辨误差指标、Barron 范数、统计矩，谱感知损失 BSP。

**📊 数据集**

数据集：Kdv、波方程、扩散‑反应方程的合成样例；高分辨率 Schlieren 图像（低分辨率→高分辨率映射）；美国地震工程研究中心（PEER NGA‑West2）地震加速度记录预测六层混凝土框架楼顶位移。

**📈 对比分析**

比较方法：对不同网络架构、激活函数（Tanh、SIREN、Chebyshev）、优化器（Adam、L‑BFGS、SOAP、SS‑Broyden）进行统一实验；利用频率分辨误差、Barron 范数、统计矩评估谱偏差；对算子模型应用 BSP 训练。结果显示：二阶优化能将误差降低 3–5 个数量级，BSP 可在保持推理成本不变的情况下将谱误差降 3–6 倍；在大多数任务中 SS‑Broyden 超越 SOAP 与 Adam。

**⚠️ 局限性**

局限性：SS‑Broyden 目前不支持 mini‑batch，规模受限；部分算子（DeepONet/DeepOKAN）在有因果约束时对 BSP 敏感；BSP 需要在完整序列上计算，增加训练时间；研究多聚焦于 1D/2D PDE 与合成/实验数据，未覆盖更大规模 3D 实际工程问题；优化与数据采样策略的耦合机制仍需进一步探索。

---

## 830. Think with Grounding: Curriculum Reinforced Reasoning with Video Grounding for Long Video Understanding

**arXiv ID:** 2602.18702 | [PDF](https://arxiv.org/pdf/2602.18702v1)

**作者:** Houlun Chen `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 23000 | [OpenAlex ID](https://openalex.org/A5100339293)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Video-TwG框架，使视频LLM在多轮推理过程中动态进行“思考-定位”视频片段的操作；

**💡 创新点**

创新点在于将思考与定位（think-with-grounding）作为学习目标，并通过两阶段强化学习课程化训练以及TwG-GRPO算法（细粒度定位奖励、自确认伪奖励、准确度门控机制）实现无需大量人工标注的自我改进；

**🔧 技术方法**

技术包括多分辨率视频表征、强化学习（GRPO）、动态视频裁剪与定位、奖励设计以及LoRA微调；

**📊 数据集**

使用了TwG-51K数据集（含7,195个带定位标签的GQA样本与42,549个无标签的通用视频QA样本），并在Video-MME、LongVideoBench和MLVU三大长视频基准上进行评测；

**📈 对比分析**

与多种基线（一般视频LLM、长上下文模型、视频代理与推理模型）对比，Video-TwG在所有基准上均取得显著提升，低分辨率下提升幅度更大；

**⚠️ 局限性**

局限在于仍依赖多轮推理与视频裁剪的计算开销，且对超长视频的定位策略与奖励设计需进一步细化。

---

## 831. A Multimodal Framework for Aligning Human Linguistic Descriptions with Visual Perceptual Data

**arXiv ID:** 2602.19562 | [PDF](https://arxiv.org/pdf/2602.19562v1)

**作者:** Joseph Bingham `[一作]` `[通讯]` (Technion University), Joseph Bingham (Technion University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多模态框架，用机器配对者（MCP）在重复引用游戏中自动实现词汇安定化（lexical entrainment），通过将人类自然语言描述与视觉感知数据对齐。

**💡 创新点**

创新点包括：① 将动态语义与可可能世界语义相结合，用更新语义来维护和更新共同基础；② 采用基于爬取的外部图像（Bing 搜索）结合 SIFT 对齐与 UQI 相似度来估计感知空间的匹配；③ 在公开的重复引用游戏数据集上首次实现完整的自动化匹配，且在多项指标上优于人类匹配者。

**🔧 技术方法**

技术手段包括：动态语义与更新语义框架、可能世界语义、SIFT 关键点对齐、Universal Quality Index（UQI）图像相似度、Bing 图像爬取、Spacy 文本预处理、基于阈值的决策阈值与 softmax 生成候选绑定、top‑k 准确率评估。

**📊 数据集**

使用的数据集为 Stanford Repeated Reference Game 公开语料库（约 15,000 条导演–匹配者对话）以及对应的 Tangram 视觉刺激图像，外加通过 Bing 搜索获得的爬取图像。

**📈 对比分析**

通过与人类匹配者的 top‑k 准确率和交互次数进行比较：单句匹配准确率 41.66%（人类 20%），利用 3 个候选提高至 63.01%，5 个候选 83.56%；平均仅需 1.78 条指令完成匹配（人类 2.73），且在时间上更快，显著减少了交互负担。

**⚠️ 局限性**

局限性：① 依赖外部爬取图像，无法实时提问澄清；② 只在预录数据上验证，缺乏实时人机交互；③ 对某些表达（如“zig zag with square on top”）的图像检索效果差；④ 未记录人类内部假设空间，导致无法完整比较中间推理过程；⑤ 仅在 Tangram 任务上验证，泛化到更复杂场景仍待验证。

---

## 832. DREAM: Deep Research Evaluation with Agentic Metrics

**arXiv ID:** 2602.18940 | [PDF](https://arxiv.org/pdf/2602.18940v1)

**作者:** Elad Ben Avraham `[一作]` (AWS Agentic AI), Ron Litman `[通讯]` (AWS Agentic AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于代理评估的Deep Research Evaluation框架（DREAM），并建立了四维度（呈现质量、任务遵从、分析深度、来源质量）的统一分类，以解决评估面临的“合成幻觉”问题。

**💡 创新点**

创新点在于：1）通过让评估者具备与研究代理相同的检索与验证能力实现“能力平行”；2）引入可自适应生成的指标（KIC、RQ、Factuality等）；3）通过两阶段协议（创建+执行）实现动态、时间感知的评估。

**🔧 技术方法**

技术手段包括：LLM-as-a-judge、CodeAgent及工具调用（Web搜索、ArXiv、GitHub）、两阶段评估协议、结构化评估工作流以及人工标注验证。

**📊 数据集**

使用数据集：DeepResearch Bench（DRB）、LiveResearchBench、ResearchRubrics，以及构造的控制实验对照集。

**📈 对比分析**

与现有静态基准（如DRB‑RACE、DRB‑FACT）对比，DREAM 在时效性、推理缺陷、事实错误的检出率上显著更高；写作质量评估与DRB‑RACE保持中等相关性（τ≈0.6），在各种评估维度上均表现出更细粒度的敏感性。

**⚠️ 局限性**

局限性包括：依赖外部工具导致可用性和检索偏差风险；评估过程计算量大、延迟高；仅为后验评估，未覆盖研究过程的中间步骤。

---

## 833. To Move or Not to Move: Constraint-based Planning Enables Zero-Shot Generalization for Interactive Navigation

**arXiv ID:** 2602.20055 | [PDF](https://arxiv.org/pdf/2602.20055v1)

**作者:** Apoorva Vashisth `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在长期、未知且拥堵环境中，移动机器人通过规划和感知交互式移动障碍物以完成连续放置任务的方法。

**💡 创新点**

核心创新在于将大语言模型（LLM）从动作序列生成器改为约束推理器：LLM根据结构化场景图、导航成本和障碍物重要性做出是否移动、何处移动及何时探索的决策；同时引入了长期效率指标LES，衡量即时成功、时间效率与环境可持续性。

**🔧 技术方法**

技术包括：基于语义场景图的感知与更新；LLM约束推理（对每个阻碍物评估搬移成本与betweenness centrality）；零射程规划将高层决策转化为导航-抓取-放置低层指令；Dijkstra路径搜索与物理仿真（ProcTHOR-10k）结合；在真实Spot机器人上部署时使用前摄像头深度和RGB进行在线场景图更新。

**📊 数据集**

使用ProcTHOR-10k仿真环境（10k个房间级别，1-10间房，20个任务/序列）以及在Spot机器人上构建的三室实地测试环境。

**📈 对比分析**

与四个基线对比：学习型InterNav、始终绕行Always Detour、始终交互Always Interact、先清理后执行Clean + S/P。通过SR、TS、PoC和LES评估，LLM驱动方法在4-6、7-10房间规模中获得最高LES（比非学习基线提升20-50%），并且在任务完成率和导航效率上保持竞争力，减少不必要的搬运次数。

**⚠️ 局限性**

局限性包括：对完整环境信息的依赖仍然存在，未知区域探索会导致额外时间成本；LLM推理受限于结构化序列化输入，可能误解复杂语义；对高遮挡、动态障碍物的鲁棒性未充分验证；实现对现实硬件的依赖较高，仿真与真实差异仍需进一步研究。

---

## 834. Ambient Analytics: Calm Technology for Immersive Visualization and Sensemaking

**arXiv ID:** 2602.19809 | [PDF](https://arxiv.org/pdf/2602.19809v1)

**作者:** Sebastian Hubenschmid `[一作]` (Aarhus University), Michael Sedlmair `[通讯]` (University of Stuttgart)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了如何通过AR技术以安静技术为核心，将数据可视化无缝嵌入用户日常环境，提出“Ambient Analytics”概念并描绘其设计原则与应用场景。

**💡 创新点**

创新点在于将安静技术与可视化融合，提出将可视化置于用户感知外缘的“Ambient Analytics”范式，强调低认知负荷与自然沉浸式交互的可能性。

**🔧 技术方法**

利用现代AR头显（如智能眼镜）、多模态感知与人工智能（AI）辅助的实时数据处理与可视化渲染技术。

**📊 数据集**

未使用具体数据集，主要基于现有文献综述与技术趋势进行理论阐述。

**📈 对比分析**

没有具体实验比较，本文以设计原则、案例想象和技术可行性评估为主，并未给出性能指标。

**⚠️ 局限性**

局限性包括：技术实现依赖高性能AR硬件与AI算法；用户注意力分配与隐私安全尚未得到充分验证；缺乏实证实验与量化评估。

---

## 835. BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS

**arXiv ID:** 2602.18755 | [PDF](https://arxiv.org/pdf/2602.18755v1)

**作者:** Omar Basit `[一作]` (Purdue University), Y. Charlie Hu `[通讯]` (Purdue University)

**通讯引用:** 5666 | [OpenAlex ID](https://openalex.org/A5103272749)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套两层框架，分别在粗粒度（5 分钟）上通过 ILP 计算预填充/解码阶段的资源分配与基准频率，在细粒度（每个迭代）上通过模型预测控制（Prefill）和一次性频率选择（Decode）动态调节 GPU 频率，从而在满足 TTFT/TPOT SLO 的前提下显著降低能耗。

**💡 创新点**

创新点包括：① 对预填充与解码的性能与功耗特性进行阶段化建模；② 在层次化控制中将粗粒度的能耗最优配置与细粒度的实时频率调节耦合；③ 对预填充采用多批预测模型预测队列演化并做 MPC；对解码采用基于时间间隔的简易频率选择；④ 使用离线训练的迭代级 Latency/Power 模型提升决策速度与精度；⑤ 在 ILP 目标中同时考虑 GPU 计数、张量并行、频率与 SLO 的可行性。

**🔧 技术方法**

核心技术包括：离线性能/功耗建模（梯度提升树 + 插值）、GPU DVFS 控制（MPC 与 per‑batch 选择）、整数线性规划（资源与 SLO 约束）、Prefill/Decode 阶段分离与 KV 缓存管理、VLLM 框架与 NVML 频率接口、以及基于模拟器的配置验证。

**📊 数据集**

使用 Llama 3.3 70B 模型，在 16×H100 GPU 集群上，以 Azure 实际生产日志（包含不同 RPS 负载）为测试数据，生成了可调节负载的控制窗口。

**📈 对比分析**

与现有 DistServe 体系对比，系统在 5 分钟窗口内的能耗降低约 39%（Prefill）和 48%（Decode），且始终满足 TTFT/TPOT SLO；通过离线评估与现场跑测均验证了模型预测的准确性（Latency MAPE 约 2.5%–3%）。

**⚠️ 局限性**

局限性包括：① 需要对 GPU 频率切换延迟与热管理做硬件依赖假设；② 预测模型与 ILP 结果对负载变化的鲁棒性有限，预测误差仍会导致能耗波动；③ 5 分钟的重配置窗口无法跟踪更短周期的负载峰值；④ 系统假设固定 GPU 架构与模型规模，迁移到其他硬件或更大模型需重新训练模型；⑤ 目前未考虑跨节点网络开销与 KV 缓存迁移的能耗。

---

## 836. To Slide or Not to Slide: Exploring Techniques for Comparing Immersive Videos

**arXiv ID:** 2602.19048 | [PDF](https://arxiv.org/pdf/2602.19048v1)

**作者:** Xizi Wang `[一作]` (University of Waterloo), Jian Zhao `[通讯]` (University of Waterloo)

**通讯引用:** 24344 | [OpenAlex ID](https://openalex.org/A5100398385)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了在VR与2D环境下比较两段360°沉浸式视频（IV）的技术，提出并实现了五种对比方法，并通过20名参与者的实验评估其使用体验。

**💡 创新点**

创新点在于将二维图像/视频比较中的“滑动”概念迁移至IV领域，形成可在VR与2D间灵活切换侧并排与覆盖（toggle）两种视图的SlideInVR/SlideIn2D技术，并系统比较其与传统Toggle与SideBySide方法的差异。

**🔧 技术方法**

使用技术包括Unity 2022与Meta Quest 3实现VR交互，Vite+React+Three.js实现2D界面；实现滑动、延伸、交换、窥视等交互，辅以最小地图、ROI轨迹可视化与时间轴控制。

**📊 数据集**

实验数据集由20对30 秒的360°视频剪辑构成，18对来自YouTube的自然场景，2对为人工合成的AI生成IV，用于四类对比任务T1–T4；每个视频配有预先标注的ROI。

**📈 对比分析**

比较方法为用户对五种技术在四类任务中的主观感受（NASA‑TLX、UMUX‑Lite）、准确率以及偏好排序；结果显示SlideInVR和SlideIn2D被视为最灵活、最受欢迎，尽管在准确率上与其他技术无显著差异，仍显著降低了切换视图所需的认知负荷。

**⚠️ 局限性**

局限性包括仅评估两段IV的成对比较、仅使用单目360°视频、ROI预设且未支持多目标或更长时长视频，且VR实现学习曲线陡峭、物理负荷高，未来需扩展到多视频、多视角与自动化ROI检测。

---

## 837. BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation

**arXiv ID:** 2602.18873 | [PDF](https://arxiv.org/pdf/2602.18873v1)

**作者:** Miaowei Wang `[一作]` (University of Edinburgh), Amir Vaxman `[通讯]` (University of Edinburgh)

**通讯引用:** 1440 | [OpenAlex ID](https://openalex.org/A5003150364)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于B‑spline的快速文本驱动动态3D角色动画生成框架，能够在固定输入尺寸下生成连续、表达性强、与文本提示高度一致的运动。

**💡 创新点**

创新点包括：①闭式拉普拉斯正则化B‑spline求解将可变长度运动压缩为固定控制点；②多层控制点嵌入与正态融合策略，提升运动细节恢复；③对应损失和局部刚度损失，保持形状一致性和运动连贯性；④将上述表示与VAE‑latent扩散相结合，构成高效的生成管线。

**🔧 技术方法**

使用的核心技术有：B‑spline曲线表示与闭式求解、Laplacian正则化、VAE‑latent扩散模型、Flow‑matching生成、跨注意力与正态融合、CLIP文本编码、Charbonnier与对应损失、局部刚度损失。

**📊 数据集**

使用新构建的BIMO数据集（约38.9k序列、3.68M帧），其内容来自DeformingThings4D Animals、ObjaverseV1和ObjaverseXL，文本注释来自OmniMotionGPT和GPT‑5多模态生成。

**📈 对比分析**

与AnimateAnyMesh、GVFDiffusion、V2M4等SOTA进行对比；在VBench指标（OC、SC、TF、AQ、DD）和用户研究（文本-运动一致性、运动可行性、表达性）均取得领先；生成时间和GPU显存更低，整体性能显著提升。

**⚠️ 局限性**

局限性包括：对极高频细节的捕捉受限于控制点数量；不支持拓扑变化的运动；依赖固定网格假设。

---

## 838. OpenClaw AI Agents as Informal Learners at Moltbook: Characterizing an Emergent Learning Community at Scale

**arXiv ID:** 2602.18832 | [PDF](https://arxiv.org/pdf/2602.18832v1)

**作者:** Eason Chen `[一作]` (Carnegie Mellon University), Cyuan Jhen Wu `[通讯]` (GiveRep Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对全人工智能代理构成的非正式学习社区Moltbook进行了大规模实证研究。

**💡 创新点**

首次量化AI代理社区的参与不平等、广播逆转（statement-to-question比率高）和并行独白模式，揭示了AI学习社区的独特动态。

**🔧 技术方法**

使用API抓取、关键词分类、Gini系数衡量参与不平等、VADER情感分析以及统计对比分析。

**📊 数据集**

采用包含2.8M+注册代理、231,080篇实质帖子、1.55M条评论的数据集，分为三阶段。

**📈 对比分析**

将Moltbook的指标与人类社区（如Reddit子版块、MOOC论坛）对比，发现参与不平等更极端、S:Q比率更高、评论独立率更高，表明AI社区在规模与交互模式上存在显著差异。

**⚠️ 局限性**

研究仅覆盖三周，使用关键词分类可能误判内容类型，未能衡量代理是否真正“学习”，且仅基于单一平台，结果的普适性有限。

---

## 839. MRI Contrast Enhancement Kinetics World Model

**arXiv ID:** 2602.19285 | [PDF](https://arxiv.org/pdf/2602.19285v1)

**作者:** Jindi Kong `[一作]` (Case Western Reserve University), Shuo Li `[通讯]` (Case Western Reserve University)

**通讯引用:** 50096 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 MRI CEKWorld——一种利用世界模型预测和生成基于对比剂增强动力学的连续 MRI 图像序列。

**💡 创新点**

创新点包括：① 基于空间一致性构建患者特定模板的 Latent Alignment Learning (LAL)；② 基于时间平滑性插值并约束二阶差分的 Latent Difference Learning (LDL)；③ 将两者结合的 SpatioTemporal Consistency Learning (STCL) 以解决稀疏采样导致的内容失真与时间不连续问题。

**🔧 技术方法**

技术手段：扩散模型+ControlNet 架构、CLIP 时间编码、零卷积图像编码、协方差模板与 Log‑Cholesky 参数化的空间约束、稠密插值与二阶差分的时间平滑损失。

**📊 数据集**

使用了两个真实临床数据集：私有腹部 DCE‑MRI（91 病例）和公开 Duke 乳腺 DCE‑MRI（922 条记录）。

**📈 对比分析**

与 CustomDiff、T2I、CCNet、EditAR、ControlNet 基线等方法对比，实验表明 MRI CEKWorld 在空间指标（PSNR/SSIM/LPIPS）和时间指标（cSSIM）上均领先，尤其在连续时间点的结构一致性和动力学平滑度上表现最佳。

**⚠️ 局限性**

局限性：仍受限于 MRI 低时序采样；模型在极端时序分布或其他对比剂成像模态（如 CT）上的泛化能力待验证；插值方法在极稀疏采样时可能引入噪声。

---

## 840. Learning Invariant Visual Representations for Planning with Joint-Embedding Predictive World Models

**arXiv ID:** 2602.18639 | [PDF](https://arxiv.org/pdf/2602.18639v1)

**作者:** Leonardo F. Toso `[一作]` (Columbia University), James Anderson `[通讯]` (Capital One)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预训练视觉特征上加入双仿射编码器，构建鲁棒的低维世界模型，能够在背景变化和视觉干扰下仍实现可靠规划。

**💡 创新点**

首次将无奖励的双仿射约束与PCA‑基VICReg正则化结合到JEPA框架，既消除了慢特征又保持了低维表示，并且兼容多种自监督视觉编码器。

**🔧 技术方法**

Joint Embedding Predictive Architectures、双仿射度量、PCA‑基VICReg、Vision Transformer 动态模型、CEM 采样规划。

**📊 数据集**

MuJoCo PointMaze 环境（含多种背景与移动干扰）以及预训练的 DINOv2、SimDINOv2、iBOT 等视觉编码器。

**📈 对比分析**

与 DINO‑WM 及其域随机化版本在相同的成功率指标下对比，模型在所有背景变化下均保持 0.75‑0.86 的成功率，显著优于基线。

**⚠️ 局限性**

依赖足够信息的预训练视觉特征；在缺乏结构化特征（如无 DINOv2）时性能显著下降；理论上 reward‑free 双仿射导致与 Horizon 成线性关系的误差上界，可能不如 reward‑aware 版本紧凑。

---

## 841. Scaling Inference-Time Computation via Opponent Simulation: Enabling Online Strategic Adaptation in Repeated Negotiation

**arXiv ID:** 2602.19309 | [PDF](https://arxiv.org/pdf/2602.19309v1)

**作者:** Xiangyu Liu `[一作]` (University of Maryland), Aranyak Mehta `[通讯]` (University of Maryland)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在自然语言谈判游戏中提出了一种通过扩展推理时间计算实现在线策略适应的框架，利用平滑虚拟博弈中的对手建模与最佳候选模拟；

**💡 创新点**

创新点在于把传统的平滑虚拟博弈步骤转化为LLM的推理任务，使用情境化对手模拟和结构化的候选生成来实现无参数更新的动态学习；

**🔧 技术方法**

核心技术包括在上下文中进行对手行为的即时模拟、基于BoN（Best‑of‑N）候选生成与全局轨迹仿真、以及在LLM内实现的多步骤推理与思考；

**📊 数据集**

实验主要在买卖者谈判游戏和资源交换游戏两个标准模拟环境上进行，使用Gemini‑2.5‑Flash、Claude‑Sonnet‑4、Qwen3、Llama‑3.3等大型语言模型作为参与者；

**📈 对比分析**

与零样本、思考型、BoN‑eval、BoN‑simulation、外部自适应方法（AI反馈、经验反思、私有信息预测）等基线比较，实验表明该方法在奖励、社会福利以及对动态对手的鲁棒性上均显著优于所有基线；

**⚠️ 局限性**

局限性包括对LLM推理成本的依赖、对对手模型精度的敏感、在极端对手策略或环境非平稳性下可能产生误判，以及可扩展性与实际部署延迟方面仍需进一步优化。

---

## 842. DSLean: A Framework for Type-Correct Interoperability Between Lean 4 and External DSLs

**arXiv ID:** 2602.18657 | [PDF](https://arxiv.org/pdf/2602.18657v1)

**作者:** Tate Rowney `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2257 | [OpenAlex ID](https://openalex.org/A5019030424)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

开发了DSLean框架，支持Lean 4与任意DSL之间的双向翻译，并以此实现了区间算术、常微分方程求解和理想成员判定的自动化推理工具。

**💡 创新点**

将Lean的解析器、元变量统一与可扩展的语法规则结合，形成统一的、可声明式的DSL定义语法，显著降低了手写翻译代码量，实现了可重用的双向转换；同时首次在Lean中用DSL实现高效的外部求解器接口。

**🔧 技术方法**

Lean 4本身的宏与元编程接口、Lean解析器（Pratt parser）、自定义元变量统一算法、以及对外部程序的调用（如Gappa、SageMath、Macaulay2）。

**📊 数据集**

未使用传统机器学习数据集，而是直接调用外部数学求解器（Gappa、SageMath、Macaulay2）作为“数据来源”。

**📈 对比分析**

在三个案例中，DSLean将翻译代码从数千行压缩至约300行；虽然论文未给出严格的时间/空间基准，但作者指出代码量减少、编写简洁，且生成的证明与原始求解器输出保持定义等价。

**⚠️ 局限性**

对Lean内部解析器的依赖导致部分语法规则受限；目前无法处理浮点运算与舍入误差；外部求解器的可靠性与证明的可验证性仍需额外公理或后续验证工作。

---

## 843. OVerSeeC: Open-Vocabulary Costmap Generation from Satellite Images and Natural Language

**arXiv ID:** 2602.18606 | [PDF](https://arxiv.org/pdf/2602.18606v1)

**作者:** Rwik Rana `[一作]` (University of Texas at Austin), Joydeep Biswas `[通讯]` (University of Texas at Austin)

**通讯引用:** 2366 | [OpenAlex ID](https://openalex.org/A5004302220)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个零样本、模块化的 Interpret–Locate–Synthesize 框架，利用航拍图像与自然语言提示即时生成符合偏好的全局代价图。

**💡 创新点**

创新之处在于将大语言模型用于实体提取与代码合成、开源语义分割与遮罩细化相结合，支持任意新实体与组合偏好在未见数据上的即时生成，并引入 Ranked Regret Path Integral 指标评估偏好一致性。

**🔧 技术方法**

技术手段包括：LLM（如 ChatGPT‑4o）进行实体识别与成本函数代码生成；CLIPSeg 进行开源语义分割；SAMRefiner 进行遮罩细化；以及基于 Python 的成本函数执行与 Dijkstra/A* 等规划器。

**📊 数据集**

使用的数据集包括：训练无监督，评估使用公开卫星图像数据集 𝒟₁（6000 张 512×512 图像）以及分布式测试集 𝒟₂（ID/OOD/OOD‑OV）和 𝒟₃（OOV 场景与人类绘制轨迹）。

**📈 对比分析**

通过与固定词典的 SegFormer‑B5 与 DINO‑UNet 两基线对比，本文方法在 RRPI、路径长度、Hausdorff 距离和 IoU 等指标上，在 ID、OOD 及 OOD‑OV 场景均实现了更低 regret、更短路径和更高分割精度。

**⚠️ 局限性**

局限性在于：对 LLM 推理准确性的依赖；遮罩细化与分割未联合训练导致效率与一致性受限；对复杂层级关系、阴影或遮挡等视觉噪声的鲁棒性不足。

---

## 844. Protecting and Promoting Human Agency in Education in the Age of Artificial Intelligence

**arXiv ID:** 2602.20014 | [PDF](https://arxiv.org/pdf/2602.20014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 845. Variations on the Problem of Identifying Spectrum-Preserving String Sets

**arXiv ID:** 2602.19408 | [PDF](https://arxiv.org/pdf/2602.19408v1)

**作者:** Sankardeep Chakraborty `[一作]` (University of Tokyo), Wiktor Zuba `[通讯]` (University of Warsaw)

**通讯引用:** 60 | [OpenAlex ID](https://openalex.org/A5083139616)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于“necklace cover”的SPSS表示方法，利用de Bruijn图的环与分支结构，生成更紧凑的k‑mer集合。

**💡 创新点**

创新点在于：①引入necklace覆盖概念，将路径覆盖扩展到含环与树形附着的结构；②设计括号嵌入的表示法；③提出贪心算法在最小PC覆盖下可得到最优necklace cover；④理论证明存在无限族实例其压缩率优于现有最优Eulertigs。

**🔧 技术方法**

使用的技术包括：de Bruijn图构建、路径/回路覆盖（PC cover）、最大二部匹配求最小PC覆盖、贪心连接路径/环形成necklace、括号表示与树形子结构嵌入、理论证明与实验评测。

**📊 数据集**

实验数据集为四个真实基因组/测序数据集：Chr19、Caenorhabditis elegans、Bombyx mori、Homo sapiens；k值范围从 11 到 31。

**📈 对比分析**

与Eulertigs、Masked Superstring、kmercamel等SPSS方法及非精确Mask Superstring进行比较，评价输出大小与运行时间。实验显示在较大k下该方法产生最小或接近最小的输出，且速度较快；在较小k下Mask Superstring压缩更紧，但不保证k‑mer谱精确。

**⚠️ 局限性**

限制：需先得到最小PC覆盖（可通过最大匹配），实现相对复杂；仅针对de Bruijn图结构；大规模数据时内存占用仍高；未评估对错误k‑mer或噪声数据的鲁棒性。

---

## 846. Soft Sequence Policy Optimization: Bridging GMPO and SAPO

**arXiv ID:** 2602.19327 | [PDF](https://arxiv.org/pdf/2602.19327v1)

**作者:** Svetlana Glazyrina `[一作]` (Lomonosov Moscow State University), Roman Ischenko `[通讯]` (Lomonosov Moscow State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Soft Sequence Policy Optimization (SSPO)，一种在大型语言模型对齐任务中使用序列级软权重的离线强化学习目标。

**💡 创新点**

创新点在于将序列级重要性采样与 token 级 sigmoid 门控相结合，既保留了软策略的连续信号，又避免了传统硬裁剪导致的方差激增和探索受限。

**🔧 技术方法**

使用技术包括几何平均聚合、长度归一化的序列重要性比、温度控制的 sigmoid 门控以及多组采样的相对优势计算。

**📊 数据集**

实验使用数学推理数据集（如数理推理 benchmark）来评估对齐效果。

**📈 对比分析**

与 GRPO、GSPO、GMPO 和 SAPO 对比，SSPO 在训练稳定性、重要性权重方差控制和样本效率方面表现更好，并显著减少了熵崩塌现象。

**⚠️ 局限性**

局限性包括：对超长序列的方差仍可能不可忽视；温度超参数需要手动调优；实验范围仅限于数理推理任务，尚未在更广泛的对齐场景中验证。

---

## 847. Scaling Law of Neural Koopman Operators

**arXiv ID:** 2602.19943 | [PDF](https://arxiv.org/pdf/2602.19943v1)

**作者:** Abulikemu Abuduweili `[一作]` (Robotics Institute Carnegie Mellon University), Changliu Liu `[通讯]` (Robotics Institute Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了神经 Koopman 逼近的缩放规律，推导采样误差与投影误差的非渐近上界，并验证其对预测与闭环控制性能的影响。

**💡 创新点**

首次给出采样与投影误差的解析上界，提出协方差正则化与逆控制正则化两种轻量级正则化，显著提升模型预测精度与闭环控制稳定性。

**🔧 技术方法**

结合 Koopman 理论、EDMD、深度学习编码器，利用矩阵伯恩斯坦不等式推导误差，采用多步预测、协方差损失与逆控制损失训练神经 Koopman，并在 MPC 中实现线性控制。

**📊 数据集**

在六个机器人环境上进行实验，包括阻尼摆、双摆、Franka Panda、Kinova Gen3、Unitree Go2 与 G1，涵盖模拟与真实数据。

**📈 对比分析**

与 EDMD、无结构神经动力学（NNDM）以及无正则化的 Koopman 进行对比，开放循环预测误差显著降低，闭环跟踪误差与生存步数在高维复杂机器人中提升超过 70%。

**⚠️ 局限性**

假设 i.i.d. 采样与光滑谱衰减，未完全处理混合接触非平稳动力学；在样本稀缺时逆控制正则化仍受限，且存在不可忽视的优化难度。

---

## 848. DP-FedAdamW: An Efficient Optimizer for Differentially Private Federated Large Models

**arXiv ID:** 2602.19945 | [PDF](https://arxiv.org/pdf/2602.19945v1)

**作者:** Jin Liu `[一作]` (Xidian University), Junkang Liu `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对差分隐私联邦学习的 AdamW 优化器 DP‑FedAdamW。

**💡 创新点**

创新点在于三项改进：1）块级二阶矩聚合以降低方差；2）对 DP 噪声产生的偏差进行无偏校正；3）加入本地‑全局对齐项抑制客户端漂移。

**🔧 技术方法**

采用 AdamW 与 DP‑SGD 结合的自适应优化技术，配合加噪梯度裁剪、指数滑动平均与全局更新估计。

**📊 数据集**

实验涵盖视觉数据集（CIFAR‑10/100、Tiny‑ImageNet）使用 ResNet‑18、ViT‑Base、Swin‑Tiny/Base，以及语言数据集 GLUE（RoBERTa‑Base 在 SST‑2、QQP、QNLI、MNLI）。

**📈 对比分析**

与 DP‑FedAvg、DP‑SCAFFOLD、DP‑FedAvg‑LS、DP‑FedSAM、DP‑LocalAdamW 等基线对比，在相同隐私预算下，DP‑FedAdamW 在非 IID 条件下提升 3–8 % 以上的准确率，尤其在 ε 较小的强隐私场景表现最佳。

**⚠️ 局限性**

局限性包括：仍需在更大规模模型或真实 FL 场景中验证；块划分方案对不同网络架构需手动调优；通信成本虽被压缩但仍高于纯 SGD 基线。

---

## 849. Effects of Property Recovery Incentives and Social Interaction on Self-Evacuation Decisions in Natural Disasters: An Agent-Based Modelling Approach

**arXiv ID:** 2602.19639 | [PDF](https://arxiv.org/pdf/2602.19639v1)

**作者:** Made Krisnanda `[一作]` (University of Newcastle), Kirill Glavatskiy `[通讯]` (University of Newcastle)

**通讯引用:** 743 | [OpenAlex ID](https://openalex.org/A5011392303)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究构建了基于代理的模型，结合演化博弈理论，对社区内部家庭代理在灾害前通过社交网络交流后做出自我疏散或停留的决策进行仿真，并探索政府财政与服务激励以及不同节点优先级对疏散率的影响。

**💡 创新点**

创新点在于将演化博弈与代理建模相结合，首次将政府激励（财政补贴与交通/保护服务）与社交网络节点度数优先策略共同纳入决策模型；通过模拟发现存在阈值跃迁，识别“社区影响者”（高度数节点）对全局疏散的关键作用。

**🔧 技术方法**

技术方法包括：基于小世界网络的社会网络构建、代理式演化博弈（收益矩阵与模仿更新规则）、参数敏感性分析（政府激励水平与优先比例）、多次仿真取平均与方差评估。

**📊 数据集**

使用的数据集为自建的5,000节点小世界网络，网络度数分布由模型设定，属性值（如P、α、β等）均取实验参数，无外部真实数据。

**📈 对比分析**

通过在四种优先策略（随机/固定，高/低度数）下，配合不同政府激励水平（-10%~20%）进行5次仿真，比较最终疏散率、阈值跳变、波动性；结果显示高度数优先显著提升疏散率，低度数优先则低效，且在阈值附近出现大幅跃迁；未在真实灾情数据上进行验证，仅在模拟环境中评估性能。

**⚠️ 局限性**

局限性包括：仅在小世界网络上验证，缺乏对尺度自由或随机网络等多种拓扑的检验；网络被假设为静态，未考虑灾害前后动态演化；代理异质性有限，未加入风险感知差异、移动约束等现实因素；最终结论依赖模拟参数，缺乏实证数据支持。

---

## 850. Test-Time Computing for Referring Multimodal Large Language Models

**arXiv ID:** 2602.19505 | [PDF](https://arxiv.org/pdf/2602.19505v1)

**作者:** Mingrui Wu `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 31954 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于测试时优化的视觉提示注入框架ControlMLLM++，使冻结的多模态大型语言模型能够在不进行任何训练或微调的情况下实现细粒度的区域级视觉推理。

**💡 创新点**

核心创新包括：1）通过学习可优化的视觉潜变量在推理阶段直接调整跨模态注意力，从而精准引导模型关注指定图像区域；2）提出Optim++优化策略（使用Adam、答案起始标记注意力、层选择）显著提升优化收敛速度与稳定性；3）引入PromptDebias对比解码机制，缓解语言提示偏置，减少跨模态幻觉。

**🔧 技术方法**

技术手段主要有：跨模态注意力分析与能量函数优化、硬/软掩码能量函数、Optim++（Adam优化器、答案起始注意力、层选择）、PromptDebias对比解码、以及对不同视觉提示（框、掩码、涂鸦、点）的一致处理。

**📊 数据集**

实验数据集包括：LVIS（ROC任务）、RefCOCOg与Screenshot（Referring描述任务）、COCO-Text（RTC任务），以及公开的Qwen2.5-VL、LLaVA-1.5、LLaVA-HR等多模态大模型。

**📈 对比分析**

与基线LLaVA、LLaVA-HR及其各种训练/无训练方法对比，ControlMLLM++在ROC、RTC、Referring描述等任务上均实现显著提升；例如在ROC box任务中从54.72%提升至71.19%，在RefCOCOg的CIDEr从55.61提升至78.42；同时保持了对多种提示形式的高泛化能力。

**⚠️ 局限性**

主要局限包括：1）推理时需梯度反向传播，导致额外计算和内存消耗；2）仅支持单一图像区域控制，无法一次性处理多区域交互；3）依赖模型梯度与内部表示，限制了对闭源商业模型的适用性。

---

## 851. gencat: Generative computerized adaptive testing

**arXiv ID:** 2602.20020 | [PDF](https://arxiv.org/pdf/2602.20020v1)

**作者:** Wanyong Feng `[一作]` (University of Massachusetts), Andrew Lan `[通讯]` (University of Massachusetts)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大语言模型的生成性自适应测试框架 GENCAT，能够利用题目文本与学生开放式回答来估计学生知识水平。

**💡 创新点**

创新点在于：①设计了 Generative Item Response Theory (GIRT) 模型，实现了将学生隐式知识映射为显式知识并生成对应代码回答；②提出三种基于生成回答的题目选择策略（不确定性、多样性、信息量）；③通过监督微调+偏好优化实现知识-回答对齐。

**🔧 技术方法**

核心技术包括大语言模型（Llama‑3.2‑1B‑Instruct）、多层感知机、线性软提示、两阶段训练（SFT + DPO）以及三种基于生成样本的题目选择算法。

**📊 数据集**

实验数据来自两个真实编程问答数据集：CodeWorkout（Java）和 ProgFeed（Python），分别包含 246/374 名学生和 50/42 道题目。

**📈 对比分析**

与 1PL IRT、BOBCAT、NCAT、LACAT 等基线比较，GENCAT 在早期测试阶段（t≤7）AUC 提升最高 4.32%，整体准确率和 AUC 均优于所有基线，并保持较低的题目曝光率和重叠率。

**⚠️ 局限性**

局限性包括：仅在编程领域验证，缺乏跨学科泛化；依赖大量训练样本，计算开销大；需要手工标注知识成分，难以在新场景快速部署。

---

## 852. Musical Training, but not Mere Exposure to Music, Drives the Emergence of Chroma Equivalence in Artificial Neural Networks

**arXiv ID:** 2602.18635 | [PDF](https://arxiv.org/pdf/2602.18635v1)

**作者:** Lukas Grasse `[一作]` (University of Lethbridge), Matthew S. Tata `[通讯]` (Canadian Centre for Behavioural Neuroscience)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了自监督和监督语音模型在包含音乐或仅音乐任务时，音高高度和音色等价的表征是否能在人工神经网络中出现。

**💡 创新点**

首次通过对不同训练任务（自监督、仅语音、音乐混合、自监督+音乐、音乐转录）进行RSA对比，证明音乐转录任务是诱导音色等价的关键因素。

**🔧 技术方法**

采用Transformer架构的Wav2Vec2.0、Data2Vec等模型，结合自监督学习、监督微调和RSA分析技术。

**📊 数据集**

使用了NSynth、LibriSpeech、LAION‑DISCO‑12M音乐、MAESTRO多音轨钢琴数据集以及CQT基准等。

**📈 对比分析**

通过构建RDM并与音高高度和音色等价模型进行Spearman相关性，发现所有预训练模型仅编码音高高度，而音乐转录微调后显著提升音色等价匹配度，接近噪声上限。

**⚠️ 局限性**

仅考察了两类监督任务，未探究其他非音乐任务或单旋律音乐是否也能诱导音色等价，且模型仅基于人工合成和钢琴数据，缺乏跨文化多样性验证。

---

## 853. Narrating For You: Prompt-guided Audio-visual Narrating Face Generation Employing Multi-entangled Latent Space

**arXiv ID:** 2602.18618 | [PDF](https://arxiv.org/pdf/2602.18618v1)

**作者:** Aashish Chandra `[一作]` (BITS Pilani), Abhijit Das `[通讯]` (BITS Pilani)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于文本提示、静态人像和语音特征的多模态音视频同步生成框架，能够一次性生成与文本内容对应的说话人视频和语音。

**💡 创新点**

创新点：
• 首个泛化的以文本为驱动的多模态学习架构，支持任意身份的音视频生成。
• 引入多重交织潜在空间，将音频、视频、文本嵌入在同一潜在空间内进行交叉注意力，显著提升同步性与真实感。
• 结合双流Transformer、变分自编码、扩散模型与GPT‑2解码器，实现端到端的高质量音视频合成。

**🔧 技术方法**

使用技术：双流Transformer编码器、HiFi‑GAN & Wav2Vec音频编码、VAE + Landmark 视觉编码、跨模态交叉注意力、潜在扩散模型、GPT‑2 语音解码、HiFi‑GAN 声码器、AdamW 优化器。

**📊 数据集**

使用数据集：VoxCeleb、FakeAVCeleb、CelebV‑HQ、HDTF，构建 36,000 条视频子集；利用 Whisper 进行语音转录文本。

**📈 对比分析**

评估方式：与 Audio2Head、Hallo、EAT、SadTalker 等 SOTA 进行 FID/FVD/FVMD、PSNR/SSIM/LPIPS/MOS 等视频指标；音频方面评估 FAD/MCD/STOI/PESQ/WER/LSE‑C/D。实验显示本模型在大多数指标上均优于或与 SOTA 相当，尤其在跨数据集泛化性能更佳。

**⚠️ 局限性**

局限性：
• 对极端噪声、闭眼或极端姿态的图像生成效果下降；
• 对不同语言或口音的泛化仍有限；
• 需要较高算力（12 GB VRAM）和专业 GPU；
• 仍存在隐私与恶意使用风险。

---

## 854. A Theory of How Pretraining Shapes Inductive Bias in Fine-Tuning

**arXiv ID:** 2602.20062 | [PDF](https://arxiv.org/pdf/2602.20062v1)

**作者:** Nicolas Anguita `[一作]`, Clementine Domine `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对预训练‑微调（PT+FT）流程，构建了对角线线性网络的解析理论，推导了泛化误差的闭式表达式，并揭示了不同初始化参数下的四种学习模式，并在实验中验证了理论预言。

**💡 创新点**

创新点在于提出了基于初始化绝对与相对尺度的四种细调模式，阐明了稀疏性与预训练依赖之间的不可兼得权衡，并通过自举理论（replica）与梯度流分析得到隐式正则化表达式。

**🔧 技术方法**

主要技术包括梯度流动态分析、对角线线性网络模型、Replica理论求解隐式正则化、以及实验验证中的仿真与ResNet-18微调。

**📊 数据集**

实验使用了教师-学生生成的稀疏 spike‑and‑slab 数据集，以及真实图像数据 CIFAR‑100 进行 ResNet 微调。

**📈 对比分析**

通过将理论预测的泛化误差与数值仿真结果对比，发现理论与仿真高度一致；在 CIFAR‑100 微调中，采用小 κ、低 c_PT 以及低 γ_FT 的初始化能提升准确率，验证了理论对实际模型的指导作用。

**⚠️ 局限性**

局限性在于理论基于线性网络、无深度与非线性，可能不完全适用于大规模深度网络；实验仅覆盖 ResNet‑18 和 CIFAR‑100 的一小子集，需进一步验证对更复杂架构和任务的推广性。

---

## 855. ExpPortrait: Expressive Portrait Generation via Personalized Representation

**arXiv ID:** 2602.19900 | [PDF](https://arxiv.org/pdf/2602.19900v1)

**作者:** Junyi Wang `[一作]` (University of Science and Technology of China), Juyong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种高保真个性化头部表示，用于生成表达丰富且身份保持一致的肖像视频。

**💡 创新点**

通过同时学习静态身份偏移场和动态表情偏移场，实现身份与表情彻底分离，并设计身份自适应表情转移模块。

**🔧 技术方法**

结合SMPL‑X参数、细分的几何偏移场、MLP表情转移网络以及Diffusion Transformer（DiT）的视频扩散模型。

**📊 数据集**

使用VFHQ、CelebV‑HQ、HDTF等约4000段视频进行训练，并在RAVDESS和NeRSemble测试集上评估。

**📈 对比分析**

在自我与跨人重现任务中，相比LivePortrait、AniPortrait、Follow‑Your‑Emoji等基线，取得更高的PSNR、SSIM、LPIPS、CSIM，并显著降低AED/APD，性能最优。

**⚠️ 局限性**

当前表示未显式建模内颊、舌头和细眼球运动，导致这些部位细节生成受限。

---

## 856. Global Commander and Local Operative: A Dual-Agent Framework for Scene Navigation

**arXiv ID:** 2602.18941 | [PDF](https://arxiv.org/pdf/2602.18941v1)

**作者:** Kaiming Jin `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60708 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种双代理协作框架DACo，用于零样本室内场景导航。

**💡 创新点**

创新点在于将全局决策与局部执行解耦，分别由全局指挥者和本地执行者承担，并引入动态子目标规划和自适应重新规划机制。

**🔧 技术方法**

使用大规模预训练视觉-语言模型（如GPT‑4o、Qwen‑VL系列）作为语言后端，并结合鸟瞰图（BEV）作为全局观测。

**📊 数据集**

在Matterport3D基准上评估，使用R2R、REVERIE和R4R三大数据集。

**📈 对比分析**

与现有零样本方法（MapGPT、NavGPT、DiscussNav等）比较，DACo在R2R、REVERIE、R4R上分别提升4.9%、6.5%和5.4%的成功率，整体表现优于同类框架。

**⚠️ 局限性**

局限性包括对BEV地图的依赖、对大型视觉语言模型的依赖以及在真实环境中获取合适地图的挑战；同时系统仍受LLM输出不确定性的影响。

---

## 857. Forgetting-Resistant and Lesion-Aware Source-Free Domain Adaptive Fundus Image Analysis with Vision-Language Model

**arXiv ID:** 2602.19471 | [PDF](https://arxiv.org/pdf/2602.19471v1)

**作者:** Zheang Huai `[一作]` (Hong Kong University of Science and Technology), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 9399 | [OpenAlex ID](https://openalex.org/A5100427643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种源无关域自适应（SFDA）方法，用于眼底图像诊断。

**💡 创新点**

创新点包括：①使用记忆库与双重互信息损失实现遗忘抑制；②利用 ViL 模型的细粒度补丁级预测实现病变感知自适应。

**🔧 技术方法**

技术手段包括：记忆库、互信息损失、补丁级监督、逐步衰减权重以及基于 FLAIR 的 Vision‑Language 模型。

**📊 数据集**

使用的数据集为 ODIR、FIVES 与 VietAI 三个眼底图像数据集，共涉及四个共享类别（正常、年龄相关黄斑变性、糖尿病视网膜病变、青光眼）。

**📈 对比分析**

与传统源模型、零样本 ViL 模型以及 SHOT、COWA、Co‑learn、DIFO 等 SOTA SFDA 方法比较，FRLA 在平均准确率上提升约 5–10%，在部分病变类别上表现尤为显著。

**⚠️ 局限性**

局限性在于依赖 ViL 模型的性能，记忆库更新频率和补丁监督会增加计算开销，且仅在眼底图像上验证，缺乏跨模态通用性评估。

---

## 858. UrbanAlign: Post-hoc Semantic Calibration for VLM-Human Preference Alignment

**arXiv ID:** 2602.19442 | [PDF](https://arxiv.org/pdf/2602.19442v1)

**作者:** Yecheng Zhang `[一作]` (Tsinghua University), Chunlei Shi `[通讯]` (Southeast University)

**通讯引用:** 894 | [OpenAlex ID](https://openalex.org/A5101566780)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练‑free 的后置概念瓶颈管线（UrbanAlign），通过概念挖掘、多代理结构化评分和局部加权岭回归，将冻结的 VLM 输出对齐到城市感知任务的主观偏好。

**💡 创新点**

创新点在于将概念挖掘、Observer–Debater–Judge 多代理推理和局部加权岭回归三阶段无训练框架结合起来，并通过端到端维度优化循环自动挑选最优概念集合，实现完全可解释且无需模型微调的校准。

**🔧 技术方法**

使用的技术包括：VLM 生成中间概念（Concept Bottleneck）、多代理链（Observer、Debater、Judge）提取连续概念分数、CLIP 特征融合、局部加权岭回归（LWRR）在视觉-语义混合空间上进行几何校准，以及温度调度的维度搜索。

**📊 数据集**

实验数据集为 Place Pulse 2.0（约 110K 街景图像、110 万对比标注，覆盖安全、活跃、美观、富裕、无聊、沮丧六个维度）。

**📈 对比分析**

与 ResNet‑50 / CLIP Siamese、SegFormer+CLIP 回归以及零样本 GPT‑4o 进行对比，UrbanAlign 在六个维度的平均准确率达到 72.2%（κ = 0.45），比最佳监督基线提升 15.1pp，零样本提升 16.3pp，且在每个维度均实现显著改进。

**⚠️ 局限性**

局限性包括：不同维度之间性能差异显著（安全最高，沮丧最低）；仅依赖静态街景图像，未考察时间与文化差异；需要一定量的人类对比标注来构建参考集；对跨城市或跨文化迁移的鲁棒性尚未验证。

---

## 859. Subtle Motion Blur Detection and Segmentation from Static Image Artworks

**arXiv ID:** 2602.18720 | [PDF](https://arxiv.org/pdf/2602.18720v1)

**作者:** Ganesh Samarth `[一作]` (Amazon Prime Video), Caren Chen `[通讯]` (Amazon Prime Video)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一框架 SMBlurDetect，用高质量合成运动模糊数据训练端到端检测器，实现零样本多粒度模糊定位

**💡 创新点**

1) 开发六种物理驱动的模糊合成方法，能生成细腻、局部的运动模糊；2) 采用混合采样与多阶段课程学习，使模型同时具备全局与局部鲁棒性；3) 设计双头 U‑Net 与多项损失（Dice、Focal Tversky、Huber 等）实现精确定位与强度估计

**🔧 技术方法**

基于 SAM 分割的实例掩码生成、光学运动模糊仿真、alpha‑aware 合成、双头 U‑Net（ResNet‑50 编码器）、混合损失、三阶段课程学习、Hard Negative Mining、焦点损失、频域通道、分辨率感知增强

**📊 数据集**

自定义高分辨率 LAION‑5B 采样并使用 SAM 生成前景掩码；在此基础上合成模糊，随后在公开基准 GoPro、NFS、CUHK 上进行零样本评估

**📈 对比分析**

与 Kim 等人基准对比；在 CUHK 上平均 IoU 59.77%（比 baseline 9% 提升 6.6×），在 GoPro 上准确率 89.68%（vs 66.5% baseline），在 NFS 上准确率 80.33%（vs 59.33%），整体表现显著优于现有监督方法

**⚠️ 局限性**

1) 模糊边界仍有平滑化现象，IoU 仅 59.77%；2) 依赖 SAM 掩码，掩码误差会传递噪声；3) 六种模糊类型仍未覆盖极端长曝光、特定传感器畸变等情况；4) 仅用合成数据训练，仍需进一步验证在真实视频序列中的时序一致性

---

## 860. BeamVLM for Low-altitude Economy: Generative Beam Prediction via Vision-language Models

**arXiv ID:** 2602.19929 | [PDF](https://arxiv.org/pdf/2602.19929v1)

**作者:** Chenran Kou `[一作]` (Southern University of Science and Technology), Chengwen Xing `[通讯]` (Beijing Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于预训练视觉‑语言模型（VLM）的端到端生成式机载波预测框架 BeamVLM，能够从地面基站相机捕获的连续图像序列中推断未来时刻的最佳波束指向。

**💡 创新点**

创新点在于：①将波束预测任务转化为生成式视觉问答，直接在语言空间输出波束索引；②通过“视觉词”把原始图像信息投射到文本域，保留完整环境语义；③利用精心设计的任务提示（Prompt）注入飞行轨迹与物理先验，实现高效、可解释的多模态推理。

**🔧 技术方法**

核心技术包括：Qwen2.5‑VL 视觉‑语言模型、LoRA 参数高效微调、视觉分块编码（ViT）、自回归文本生成、结构化 Prompt 设计和旋转位置编码的多头注意力。

**📊 数据集**

使用的主要数据集为 DeepSense 6G Scenario 23（UAV 端到端 mmWave 通信数据）以及 Scenario 8（车路协同 V2I 场景），两者均提供连续图像帧与对应波束标签。

**📈 对比分析**

与 RNN、LSTM、BeamLLM 等传统序列模型及基于目标检测的 LLM 方法进行对比。BeamVLM 在 UAV 场景的 Top‑1 预测准确率 t+1 为 83.3%、t+5 为 71.4%，相较 LSTM 提升约 10.8%；在 V2I 场景的 Top‑1 t+1 为 72.1%，比 BeamLLM 高出 11.1%。总体而言，BeamVLM 在多步预测、跨场景泛化以及 Top‑3 准确率上均取得显著优势。

**⚠️ 局限性**

主要局限包括：①推理时延相对较高（受模型规模和多模态编码影响）；②对 Prompt 设计的依赖性较强，若任务约束或语义先验更改需要重新调试；③在极端动态或遮挡严重的环境中，视觉信息的完整性仍可能影响预测精度。

---

## 861. Redefining the Down-Sampling Scheme of U-Net for Precision Biomedical Image Segmentation

**arXiv ID:** 2602.19412 | [PDF](https://arxiv.org/pdf/2602.19412v1)

**作者:** Mingjie Li `[一作]` (Stanford University), Lei Xing `[通讯]` (Stanford University)

**通讯引用:** 33032 | [OpenAlex ID](https://openalex.org/A5100381484)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出并实现了“Stair Pooling”这一新的下采样策略，改进U-Net在医学图像分割中的长距离语义捕获能力。

**💡 创新点**

创新点在于将大尺寸池化拆分为小且狭窄的 1×2 与 2×1（或 3D 对应）组合，并通过多方向路径融合、逐步降低空间分辨率，从而减少信息损失；同时利用传递熵自动选取最优下采样路径。

**🔧 技术方法**

使用了卷积+ReLU 的分支结构、拼接融合、传递熵（Transfer Entropy）与高斯熵估计来评估信息保留并搜索最佳路径。

**📊 数据集**

实验使用 Synapse（多器官 CT）、ACDC（心脏 MRI）和 KiTS23（肾肿瘤 CT）三大 2D/3D 数据集。

**📈 对比分析**

与传统 U‑Net、Attention‑UNet、TransUNet、SwinUNet、Pyramid UNet 等多种池化/注意力变体对比，在 2D/3D benchmark 上平均提升 Dice 约 3.8%，并在模型尺寸上通过 TE 选路进一步压缩。

**⚠️ 局限性**

局限性包括：对每个下采样步骤只能计算单一路径，导致 3D pooling 进一步扩展时搜索空间有限；并且全图熵计算可能不适用于所有组织形状。

---

## 862. Spiking Graph Predictive Coding for Reliable OOD Generalization

**arXiv ID:** 2602.19392 | [PDF](https://arxiv.org/pdf/2602.19392v1)

**作者:** Jing Ren `[一作]` (RMIT University), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 19440 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为SIGHT的图神经网络插件模块，利用尖峰神经网络与预测编码实现对分布外（OOD）样本的无监督不确定性估计与鲁棒泛化。

**💡 创新点**

创新点在于将尖峰动力学与预测编码相结合，形成迭代误差驱动的纠正循环；内部误差信号可直接解释不确定性来源；同时采用局部Hebbian更新，实现可解释、无梯度传播、能效高的OOPO训练。

**🔧 技术方法**

技术手段包括Poisson特征编码、LIF尖峰神经元、预测编码循环、局部误差驱动的梯度更新，以及多种后处理校准方法（G-ΔUQ、温度缩放、Dirichlet等）做对比。

**📊 数据集**

实验使用Cora、Citeseer、Pubmed（协方差偏移）和Twitch、CBAS（概念偏移）等五个节点分类数据集。

**📈 对比分析**

通过与GCN/GAT基线以及多种后处理校准方法比较，采用准确率、ECE、NLL、BS、AUROC等指标。SIGHT在大多数数据集实现最高准确率、最低ECE、最高AUROC，显著提升了OOD检测与不确定性估计的性能。

**⚠️ 局限性**

局限性包括：对超参数（K、T、学习率等）敏感；在某些特定组合（如SIGHT+GAT on CBAS）表现略逊；未在动态图或极大规模网络上验证；尖峰编码实现的硬件依赖性仍待进一步探讨。

---

## 863. Closing the gap in multimodal medical representation alignment

**arXiv ID:** 2602.20046 | [PDF](https://arxiv.org/pdf/2602.20046v1)

**作者:** Eleonora Grassucci `[一作]` (Sapienza University of Rome), Danilo Comminiello `[通讯]` (Sapienza University of Rome)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种新的损失组合来关闭医学多模态中的模态间隙，提升图像与文本的语义对齐。

**💡 创新点**

创新点在于提出 Align True Pairs 与 Centroid Uniformity 两个损失函数，使正样本更紧密、负样本均匀化，从而消除模态间隙。

**🔧 技术方法**

使用 CLIP 对比损失结合两新损失，采用 EVA-Clip-ViT-G 视觉编码器和 BERT 文本编码器。

**📊 数据集**

使用 ROCO（Radiology Object in Context）医学图像与文本数据集进行训练和评估。

**📈 对比分析**

与标准 MedCLIP (LT/FT) 对比，所提方法在 Cos True Pairs 从 0.20 提升到 0.54，Recall@10 提升 7.4 分，Captioning 的 BLEU、ROUGE 等指标亦有提升。

**⚠️ 局限性**

限制在于仅验证了两模态，未考虑更多医学模态，且对极端不平衡或噪声数据的鲁棒性未充分评估。

---

## 864. EEG-Driven Intention Decoding: Offline Deep Learning Benchmarking on a Robotic Rover

**arXiv ID:** 2602.20041 | [PDF](https://arxiv.org/pdf/2602.20041v1)

**作者:** Ghadah Alosaimi `[一作]` (Imam Mohammad Ibn Saud Islamic University), Toby P. Breckon `[通讯]` (Durham University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实施了真实环境下的脑-机器人接口实验，使用16通道EEG在户外移动机器人上对前进、后退、左转、右转和停止五个指令进行多时延解码，并对11种深度学习模型进行系统评估。

**💡 创新点**

创新点包括：①提出多时延标签策略（Δ=0~1000 ms）评估即时与前瞻意图解码；②首次在室外机器人上实现多指令EEG解码并提供可复现的基准；③在同一实验框架下系统比较CNN、RNN和Transformer模型在此任务中的表现。

**🔧 技术方法**

技术手段包括16通道OpenBCI EEG采集与PyPREP预处理、时间分层分割与窗口化、11种DL模型（ShallowConvNet、EEGNet、TSCeption、DeepConvNet、STNet、CCNN、CNN1D、LSTM、GRU、ViT、EEG-Conformer），以及加权交叉熵与Adam优化。

**📊 数据集**

数据集为12名受试者在室外预设路径上完成的120个会话（每会话≈20 min）所收集的EEG与操纵标签，形成多时延自标注数据集。

**📈 对比分析**

通过时间分层分割、加权交叉熵训练和每个Δ的准确率/精确率/召回率/F1评估，结果显示ShallowConvNet在Δ=0时F1≈67%，Δ=300 ms时≈66%，保持>60%直至Δ=900 ms；其他CNN与GRU表现相近，Transformer模型表现相对较低。

**⚠️ 局限性**

局限性在于仅为离线实验，未验证在线实时控制；样本规模有限，跨会话与跨受试者泛化需进一步评估；Transformer模型因数据量不足表现不佳；EEG信号易受运动与环境噪声影响。

---

## 865. Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration

**arXiv ID:** 2602.20104 | [PDF](https://arxiv.org/pdf/2602.20104v1)

**作者:** Hasan Amin `[一作]` (Purdue University), Rajiv Khanna `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种人本适应性AI集成框架，能够在补充性与一致性之间动态切换，从而提升人机协同决策的整体性能。

**💡 创新点**

①正式表述补充-一致性权衡，证明单一模型不可同时优化两者；②提出基于Confidence‑Gated Probabilistic Reliance (CGPR) 的人机交互模型；③设计Rational Routing Shortcut (RRS) 机制，实现无须人类内部状态即可近似最优路由。

**🔧 技术方法**

使用两种专家模型（对齐模型和补充模型）训练；基于逻辑回归/ResNet的分类器；利用CGPR模型进行人机决策模拟；理论分析证明RRS的近似最优性；在模拟与真实数据上进行对照实验。

**📊 数据集**

1) 通过人工生成的二元决策数据（GPA与标准化考试成绩）模拟两种人类信心水平；2) 10类ImageNet子集（5类物体+5类犬种）与人工标注的置信度，构成真实视觉决策基准。

**📈 对比分析**

与标准AI、对齐AI、补充AI、行为感知AI对比，采用人机团队准确率评估。结果显示Adaptive AI（Oracle）/RRS分别达到约74.8%/75.1%团队准确率，显著高于标准AI（≈69.1%）和行为感知AI（≈70.9%），即提升约5–6个百分点。

**⚠️ 局限性**

需要估计人类置信度阈值分布，RRS对置信度校准敏感；在高不确定性或错误路由下性能下降；理论假设（如独立性、校准误差限制）在复杂现实场景中可能不完全成立。

---

## 866. Skill-Inject: Measuring Agent Vulnerability to Skill File Attacks

**arXiv ID:** 2602.20156 | [PDF](https://arxiv.org/pdf/2602.20156v1)

**作者:** David Schmotz `[一作]` (Max Planck Institute for Intelligent Systems), Maksym Andriushchenko `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于LLM代理技能文件的注入攻击基准（SkillInject），并评估了当前前沿模型的易受性

**💡 创新点**

首次构建了涵盖明显与情境注入、23个技能、202个注入-任务对的基准，并引入了上下文安全策略（合法化与警告）

**🔧 技术方法**

使用LLM代理（Claude Code、Gemini CLI、OpenAI Codex）与自然语言安全策略、LLM审计器等技术进行评估

**📊 数据集**

收集并整理了23个不同技能（文档、机器学习、支付、医疗等），共30个明显注入与41个情境注入，形成202个注入-任务对

**📈 对比分析**

在不同安全策略（基线、合法化、警告）下对比前沿模型的攻击成功率，发现大多数模型在情境注入下攻击成功率高达50%以上，明显注入下可达70%，表明现有防御不足

**⚠️ 局限性**

局限在有限的技能与攻击场景、对模型实现的依赖、未覆盖更复杂的多轮或跨模型攻击，且安全策略的评估受制于人类对“合法”与“恶意”的主观判断

---

## 867. Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data

**arXiv ID:** 2602.20152 | [PDF](https://arxiv.org/pdf/2602.20152v1)

**作者:** Zhenyao Ma `[一作]` (Xiamen University), Dongxu Li `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Behavior Learning (BL)，一种可解释且可识别的通用机器学习框架，学习数据中隐含的优化结构。

**💡 创新点**

通过把模型构造为由可解释的效用最大化块组成的层次化结构，解决性能-可解释性权衡，并在统计上证明可识别性。

**🔧 技术方法**

采用 Gibbs 分布、可微惩罚式 UMP、深度块堆叠、denoising score matching、熵正则化等技术。

**📊 数据集**

在10个标准预测数据集、Boston Housing、图像（MNIST、Fashion‑MNIST）和文本（AG News、Yelp）等多维数据上进行实验。

**📈 对比分析**

与10种基线（包括 MLP、树模型、梯度提升、贝叶斯方法、线性回归）比较，BL 在可解释模型中获得第一梯队预测性能，并在高维任务上保持与 E‑MLP 相当的准确率、校准和 OOD 检测优势。

**⚠️ 局限性**

理论假设在大规模、过参数化网络下尚未充分验证，多项式基函数易导致数值不稳定，且模型仍需进一步扩展到生成任务和更复杂的科学领域。

---

## 868. Simulation-Ready Cluttered Scene Estimation via Physics-aware Joint Shape and Pose Optimization

**arXiv ID:** 2602.20150 | [PDF](https://arxiv.org/pdf/2602.20150v1)

**作者:** Wei-Cheng Huang `[一作]` (University of Illinois at Urbana-Champaign), Kris Hauser `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理约束的联合形状与姿态优化框架，能够将单张RGB‑D图像中的混乱场景转换为可直接用于物理仿真的、力平衡且无交叉的物理一致场景。

**💡 创新点**

创新点主要包括：①使用可微分的形状可微接触模型SDRS，消除正向力变量，使得形状与姿态可直接耦合优化；②利用增广拉格朗日方法结合Hessian的稀疏结构，采用Woodbury+Schur补的线性求解器，显著降低多物体场景的计算复杂度；③在同一管线中集成学习式初始化、物理约束优化与可微纹理细化，实现端到端的物理一致重建。

**🔧 技术方法**

技术手段包括可微分渲染、形状可微接触模型SDRS、增广拉格朗日方法、Levenberg–Marquardt子问题求解、Woodbury矩阵恒等式与Schur补线性求解、凸包表示的物体模型、CPU并行与GPU可微纹理优化。

**📊 数据集**

实验使用自建的5个桌面混乱场景（最多5个刚体，共22个凸包）作为数据集，输入为单视RGB‑D图像。

**📈 对比分析**

与SAM3D+FoundationPose以及其他单视重建方法进行比较，评估指标包括：PSNR、仿真第一秒动能增量、仿真一分钟漂移距离以及总计算时间。实验结果表明：①物理一致性显著提升，仿真稳定性好；②PSNR与初始估计相当；③计算时间在几分钟级别，利用Woodbury+Schur求解器比直接LU快最高8.7倍。

**⚠️ 局限性**

主要局限：计算成本仍偏高，尤其是形状参数化导致决策变量增多；依赖SAM3D的初始形状在严重遮挡场景下可能不准；当前实现以CPU为主，未充分利用GPU；未实现完全端到端的图像引导优化。

---

## 869. Agentic AI for Scalable and Robust Optical Systems Control

**arXiv ID:** 2602.20144 | [PDF](https://arxiv.org/pdf/2602.20144v1)

**作者:** Zehao Wang `[一作]` (Duke University), Tingjun Chen `[通讯]` (Duke University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AgentOptics——一种基于 Model Context Protocol (MCP) 的 agentic AI 框架，利用 64 个标准化工具控制八类光学设备，并在 410 个真实硬件任务上进行评估；同时演示了五个从链路配置到分布式传感的实际案例。

**💡 创新点**

创新点包括：① 将光学设备操作抽象为 MCP 工具，形成统一、可扩展的接口；② 通过 LLM 进行自然语言解析、工具选择与多步协作，显著提升任务成功率；③ 构建了涵盖单/双/三步、语义变体、错误检测等多维度的 410 任务基准；④ 通过与 CodeGen（代码生成）基线对比，展示了基于 MCP 的方法在多模型上的鲁棒性与优越性。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑4o mini、Claude Sonnet 4.5、DeepSeek‑V3 等在线模型以及 Qwen‑0.6B/8B/12B 本地模型）、MCP 协议实现的客户端/服务器架构、工具调用与返回结果的结构化语义解析、LLM 反馈循环与错误处理机制。

**📊 数据集**

数据集：410 个基准任务（来自 10 条单步、10 条双步、10 条三步原始任务，经过 5 种变体扩展），以及对应的真实光学设备操作脚本与手工验证结果；此外，针对案例研究使用了现场 DWDM 链路、5G ARoF 链路、两跨段 CFP2 链路、光纤偏振仪与 DAS 传感器等实验数据。

**📈 对比分析**

与 CodeGen 直接代码生成基线对比，AgentOptics 在在线 LLM 上平均成功率 98.8–100%，本地 LLM 上 87.7%；而 CodeGen 仅能达 8–50% 的成功率；在成本方面，轻量级在线模型（GPT‑4o mini、DeepSeek‑V3）以低至 $0.004/任务即可实现近乎完美的成功率；执行时间上，在线模型平均 11–23 秒，轻量级模型更快；总的来说，AgentOptics 在准确率、成本与可扩展性上均优于传统代码生成方法。

**⚠️ 局限性**

局限性：① 本地 LLM 受参数规模限制，复杂多步任务成功率下降；② 仍存在工具调用顺序/命名误差导致的执行失败；③ 对新设备/接口需要重新定义 MCP 工具；④ 高端在线模型成本相对较高，需平衡收益；⑤ 案例研究多基于实验室设备，工业规模验证仍待进一步展开。

---

## 870. Recurrent Structural Policy Gradient for Partially Observable Mean Field Games

**arXiv ID:** 2602.20141 | [PDF](https://arxiv.org/pdf/2602.20141v1)

**作者:** Clarisse Wibault `[一作]` (University of Oxford), Jakob Foerster `[通讯]` (London School of Economics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了RSPG——一种针对具有公共噪声的部分可观测平均场游戏的循环结构化策略梯度方法，并开发了支持白盒/黑盒转移、部分可观测、公共噪声及多初始均衡的JAX框架MFAX

**💡 创新点**

创新点在于首次将循环记忆（历史依赖）引入结构化方法，利用已知个体转移实现精确期望更新并显著降低方差；同时MFAX通过函数式表示与GPU并行实现线性加速，解决了现有库对部分可观测和公共噪声的不足

**🔧 技术方法**

主要技术包括：Hybrid Structural Methods（利用已知个体转移），可微分析平均场更新，循环神经网络处理共享观测历史，连续动作空间的分布参数化与离散化策略，JAX框架下的GPU加速矩阵运算与批量环境并行

**📊 数据集**

使用的实验环境为三种自定义平均场游戏：线性二次（Linear Quadratic）、海滩酒吧（Beach Bar）和宏观经济（Macroeconomics）——后者是异质代理、公共噪声的金融宏观模型；所有环境均在MFAX中实现并公开

**📈 对比分析**

与传统结构化方法（SPG）以及RL方法（IPPO、RIPPO、M‑OMD）对比，RSPG在三种环境中均实现了更低的exploitability，且收敛速度比RL快约十倍；实验采用wall‑clock时间作为指标，RSPG在GPU上可在几十秒完成与RL相同精度的学习

**⚠️ 局限性**

主要局限在于：1）需要对个体转移有白盒访问，若转移未知需改用采样更新；2）仅适用于共享观测的特殊情况，完整IAOH的记忆管理仍不可行；3）高维状态空间仍受限，需进一步引入函数逼近或价值学习；4）目前未覆盖主从玩家或多均衡情形

---

## 871. Do Large Language Models Understand Data Visualization Rules?

**arXiv ID:** 2602.20137 | [PDF](https://arxiv.org/pdf/2602.20137v1)

**作者:** Martin Sinnona `[一作]`, Viviana Siless `[通讯]` (Universidad Torcuato Di Tella)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估大语言模型在Vega‑Lite规范中识别可视化规则违规的能力

**💡 创新点**

首次将Draco的ASP硬核验证规则转化为自然语言进行LLM评估，并构建专门的验证基准

**🔧 技术方法**

结合Draco、ASP、Vega‑Lite、Gemma、GPT‑OSS和Llama等LLM技术

**📊 数据集**

使用2,000条标注有Draco规则违规的Vega‑Lite规格数据集

**📈 对比分析**

通过准确率、F1与prompt‑adherence指标比较，GPT‑OSS最高（F1≈0.82，prompt‑adherence≈98%），Gemma 27B次之，细微感知规则识别仍较差

**⚠️ 局限性**

局限性包括对细微感知规则识别不足、ASP形式规则识别差、LLM对提示格式依赖强、与符号求解器相比性能仍有限

---

## 872. AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization

**arXiv ID:** 2602.20133 | [PDF](https://arxiv.org/pdf/2602.20133v1)

**作者:** Mert Cemri `[一作]` (University of California), Ion Stoica `[通讯]` (University of California)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 AdaEvolve，一个基于 LLM 的自适应进化框架，利用累积改进信号在局部、全局与元层面动态调节探索强度、资源分配与搜索策略；

**💡 创新点**

创新点包括：1）统一的累积改进信号驱动三层自适应；2）使用全局归一化奖励的 UCB 选岛策略，避免低基线岛屿被过度偏好；3）动态岛屿生成与迁移；4）元指导层生成高层搜索策略以突破概念性局限；

**🔧 技术方法**

技术手段：LLM（GPT‑5、Gemini‑3‑Pro）语义变异、指数移动平均累积改进、UCB 多臂赌博机资源分配、探索/利用概率调节、元学习/反思提示生成、基于改进信号的探索‑利用平衡；

**📊 数据集**

数据集：共 185 个开放式优化/算法设计任务，涵盖 6 个数学优化、7 个 ADRS 系统优化、172 个 Frontier‑CS 算法设计问题；

**📈 对比分析**

方法对比：与 OpenEvolve、GEPA、ShinkaEvolve、AlphaEvolve 等开源基线对比，AdaEvolve 在所有任务上均表现更优；在 Circle Packing、ADR 系统等任务中取得 SOTA，甚至超过人类/AlphaEvolve 的得分；

**⚠️ 局限性**

局限性：1）对 LLM 计算资源需求高；2）仅基于离散改进信号，缺乏梯度信息；3）元指导仍依赖预设提示与人工设计，可能难以泛化到极其多样的任务；4）在奖励平滑、少量改进任务上提升有限；5）大规模任务的可扩展性和效率尚待验证。

---

## 873. LAD: Learning Advantage Distribution for Reasoning

**arXiv ID:** 2602.20132 | [PDF](https://arxiv.org/pdf/2602.20132v1)

**作者:** Wendi Li `[一作]` (University of Wisconsin-Madison), Sharon Li `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Learning Advantage Distribution (LAD) 框架，将 RLVR 的优化目标从期望奖励最大化改为匹配优势诱导分布，并在数学与代码推理任务上验证其有效性。

**💡 创新点**

创新点在于将策略优化视为 f-散度最小化的分布匹配问题，避免传统奖励最大化导致的模式崩溃与多样性缺失。

**🔧 技术方法**

技术方法包括：f-散度（Jensen–Shannon、Hellinger 等）最小化、信赖域策略优化、实践替代目标、GRPO、FlowRL 对比等。

**📊 数据集**

使用的数据集与模型：Qwen2.5-7B（数学推理）和 DeepSeek‑R1‑Distill‑7B（代码推理）；数学基准包括 MATH500、AIME 2024/2025、AMC、OlympiadBench、Minerva；代码基准包括 LiveCodeBench、CodeForces、HumanEval+。

**📈 对比分析**

与 GRPO、Entropy‑regularized baselines（EntAdv、KLCov、ClipCov）以及 FlowRL 进行比较，LAD 在 Avg@32、Pass@32、distinct‑3/4 及 GPT‑4 Judge 等指标上均优于对手，显著提升准确率与生成多样性。

**⚠️ 局限性**

局限性：仅适用于具备可验证奖励的推理场景；在无 verifer 的开放式对话、创意写作等领域尚未验证；实验规模仅 1.5B–7B 级模型，缺乏对更大模型的扩展与收敛理论。

---

## 874. A Very Big Video Reasoning Suite

**arXiv ID:** 2602.20159 | [PDF](https://arxiv.org/pdf/2602.20159v1)

**作者:** Maijunxian Wang `[一作]` (University of California, Berkeley), Hokin Deng `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并发布了VBVR大规模视频推理数据集及可验证的评估框架，并在此上开展规模化实验

**💡 创新点**

首次构建覆盖150种任务、约1,000万视频实例的海量多任务推理数据集，采用规则化评估实现可解释性和可复现性

**🔧 技术方法**

使用参数化任务生成器的分布式生成管线，结合LoRA微调的DiT模型进行训练与评估

**📊 数据集**

VBVR包含约2,015,000帧、1,007,500段视频，涵盖5大认知维度（抽象、知识、空间、感知、变换）

**📈 对比分析**

在VBVR上与多种开源与闭源模型对比，最佳开源模型（Wan-2.2-I2V-A14B）在Fine‑tune后取得0.685分，仍与人类水平存在显著差距

**⚠️ 局限性**

受当前生成架构对长期时序一致性、过程可验证性和身份保持等挑战限制，泛化仍受限

---

## 875. To Reason or Not to: Selective Chain-of-Thought in Medical Question Answering

**arXiv ID:** 2602.20130 | [PDF](https://arxiv.org/pdf/2602.20130v1)

**作者:** Zaifu Zhan `[一作]` (University of Minnesota), Rui Zhang `[通讯]` (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Selective Chain-of-Thought（Selective CoT）推理策略，在医学问答中动态决定是否生成推理步骤，以提高大模型的效率和可解释性。

**💡 创新点**

创新点在于将推理需求判别与生成过程自适应融合，首次在医学QA中实现按需推理控制，显著减少无用推理而不牺牲准确率。

**🔧 技术方法**

采用的大模型是开源的Llama-3.1-8B和Qwen-2.5-7B，比较了标准CoT、固定长度CoT和Selective CoT三种推理范式，并通过token计数和实际推理时间评估效率。

**📊 数据集**

使用的四个医学问答基准为HeadQA、MedMCQA、MedQA-USMLE和PubMedQA，涵盖多语言、多学科及文献推理等多种任务风格。

**📈 对比分析**

与传统CoT和固定长度CoT相比，Selective CoT在保持或略微提升准确率的同时，将推理时间降低13%–45%，token消耗下降8%–47%；部分模型-数据对甚至实现精度提升。

**⚠️ 局限性**

局限性包括：推理时延受硬件和批处理等后端因素影响，实验仅覆盖两款模型，缺乏对更大规模模型和外部检索场景的验证，且未对模型对推理决策的准确性进行深入分析。

---

## 876. Enormous Fluid Antenna Systems (E-FAS)--Part II: Channel Estimation

**arXiv ID:** 2602.20127 | [PDF](https://arxiv.org/pdf/2602.20127v1)

**作者:** Farshad Rostami Ghadi `[一作]` (University College London), Chan-Byoung Chae `[通讯]` (Yonsei University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文对采用巨大流体天线系统（E‑FAS）辅助的下行链路在存在不完美信道状态信息（CSI）情况下的性能进行了系统性分析，涵盖了MMSE信道估计、单用户MRT和多用户ZF预编码的误差统计、误码概率、可实现速率以及训练开销与空间乘法收益的权衡。

**💡 创新点**

创新点在于首次给出E‑FAS系统在基于训练的CSI估计下的闭式估计误差统计和信道增益分布；揭示了残留自干扰和多用户干扰导致的高SNR饱和现象；以及量化了训练长度、用户数与空间自由度之间的性能权衡，凸显了路由增益与CSI精度的耦合效应。

**🔧 技术方法**

主要技术包括MMSE信道估计、正交训练序列、零逼迫（ZF）预编码、伽马分布和广义指数积分（E_n）解析、以及基于两时间尺度模型的路由增益估计。

**📊 数据集**

实验采用 Monte Carlo 仿真，假设 M‑天线 BS、K 个单天线用户，信道为复高斯/瑞利分布，使用参数 M=64、K=8、T_c=200、β 为可调的大尺度路由增益；并不使用真实数据集，而是通过随机仿真验证解析结果。

**📈 对比分析**

通过将 MMSE 与 LS 估计、理想 CSI、以及无 E‑FAS 的传统系统进行对比，结果显示 E‑FAS 在任何 SNR 区间均保持显著的功率增益；MMSE 在低功率/短训练周期下优于 LS；但在高 SNR 时两者均出现饱和；与理想 CSI 之间的性能差距随 β 增大而减小。

**⚠️ 局限性**

主要局限包括：假设瑞利小尺度衰落和独立同分布；仅考虑两时间尺度的估计，未考虑硬件失真和非理想路由配置；对路由增益的估计误差仅用一阶近似；并未针对多用户情况下的更复杂预编码（如 MMSE 或软干扰消除）进行分析。

---

## 877. Adaptation to Intrinsic Dependence in Diffusion Language Models

**arXiv ID:** 2602.20126 | [PDF](https://arxiv.org/pdf/2602.20126v1)

**作者:** Yunxiao Zhao `[一作]` (University of Michigan), Changxiao Cai `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种分布无关的随机化解码（unmasking）调度方法，用以提升离散扩散语言模型（DLM）的并行采样效率并减少采样误差。

**💡 创新点**

通过随机化每一步的解码批量大小，设计了无需先验信息或超参数调优的调度方案，并在理论上证明其采样误差与数据分布的总相关性（TC）或双总相关性（DTC）成比例，取得优于固定批量大小策略的收敛速度。

**🔧 技术方法**

采用信息论度量（TC、DTC）和 KL 散度分析，构造递归随机采样策略，并利用动态规划预计算系数实现 O(K+L) 的采样集合生成复杂度。

**📊 数据集**

在 Reed–Solomon 码的均匀分布上验证理论，利用 q=2048 字母表、L=2000 长度的实验；此分布具有已知 TC 与 DTC，可精确评估采样误差。

**📈 对比分析**

与固定大小（均匀随机）解码方案对比，随机批量调度在相同迭代次数下 KL 误差呈 O(1/K) 降低，且误差随 TC 或 DTC 的线性增长，验证了理论预测并显著优于传统方法。

**⚠️ 局限性**

局限性包括：理论假设已知完美的 mask 预测器；未考虑训练阶段对采样策略的影响；仅对随机位置选择的批量大小进行优化，未探讨基于置信度或熵的自适应位置选择；并未给出同时兼顾 TC 与 DTC 的单一最优调度。

---

## 878. KNIGHT: Knowledge Graph-Driven Multiple-Choice Question Generation with Adaptive Hardness Calibration

**arXiv ID:** 2602.20135 | [PDF](https://arxiv.org/pdf/2602.20135v1)

**作者:** Mohammad Amanlou `[一作]`, Behnam Bahrak `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

该文本仅为会议论文模板，没有具体研究内容

**💡 创新点**

无

**🔧 技术方法**

无

**📊 数据集**

无

**📈 对比分析**

无

**⚠️ 局限性**

无法进行评估

---

## 879. Modeling Epidemiological Dynamics Under Adversarial Data and User Deception

**arXiv ID:** 2602.20134 | [PDF](https://arxiv.org/pdf/2602.20134v1)

**作者:** Yiqi Su `[一作]` (Virginia Tech), Naren Ramakrishnan `[通讯]` (Virginia Tech)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出将动态SVEAIR疫情模型与信号博弈框架相结合，以研究个人在疫苗接种与口罩使用上的策略性误报对疫情控制与公共卫生决策的影响。

**💡 创新点**

在传统的传播模型中首次引入了信号博弈来显式建模人口与公共卫生机构之间的沟通与信息不对称；并对三种博弈均衡（分离、部分池化、完全池化）进行解析与政策设计。

**🔧 技术方法**

使用基于SVEAIR的分段化学动力学模型、两阶段交互式信号博弈、强化学习式的自适应政策更新与基于熵与失真理论的后验推理。

**📊 数据集**

主要使用仿真生成的10000人群数据（10万模拟运行）构建SVEAIR传播过程，并通过模拟获得的自报行为、误报率与医院入院率等信息进行实验。

**📈 对比分析**

与无交互（基准）和随机政策进行对比；实验结果显示在低基线行为下，分离均衡在第11周即可将有效传播数R_c降至1以下；部分池化在第22周控制成功；完全池化则无法在26周内控制。自适应策略相比随机策略能够更快提升疫苗与口罩覆盖率、降低峰值住院率并压制误报率。

**⚠️ 局限性**

模型假设为总体水平的均质化、固定的非响应比例与最大化误报；未考虑人口异质性、动态策略适应与多机构协同；对非响应的处理过于极端，实际情况可能更为复杂。

---

## 880. Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device

**arXiv ID:** 2602.20161 | [PDF](https://arxiv.org/pdf/2602.20161v1)

**作者:** Abdelrahman Shaker `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fahad Shahbaz Khan `[通讯]` (Linköping University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 Mobile-O 的轻量级统一视觉‑语言‑扩散模型，可在移动端实现图像理解与图像生成的实时推理。

**💡 创新点**

创新点包括：
1) Mobile Conditioning Projector（MCP），通过层级融合、深度可分离卷积和轻量级通道注意力，将 VLM 的隐藏状态直接投射到扩散模型的条件空间，省去传统的查询 token，显著降低参数量和计算量；
2) 四元组统一后训练（generation prompt + image + question + answer）格式，使同一批样本同时参与视觉理解与图像生成的多任务训练，提升跨模态对齐能力；
3) 采用极小化的训练样本量（仅数百万条）实现强大性能，并实现 3 秒/512×512 图像生成的边缘部署。

**🔧 技术方法**

技术方案：
- 视觉编码器：FastVLM（FastViT+Qwen2 LLM）
- 生成器：SANA-600M-512 及 VAE 编码/解码
- 连接模块：MCP（深度可分离 Conv1D + 轻量化通道注意力）
- 训练流程：跨模态对齐 → 监督微调 → 四元组统一后训练
- 损失：跨模态理解的交叉熵 + 速度匹配式扩散损失

**📊 数据集**

使用的数据集与方法：
- 预训练：JourneyDB（400 万对）、BLIP3o‑Short‑Caption（500 万对）
- 微调：约 105k 目标样本（60k BLIP3o + 45k ShareGPT‑4o‑Image）
- 四元组后训练：利用 GPT‑4o 生成描述，自动合成 Q&A 组，共 105k 条四元组
- 对比基准：GenEval（图像生成）、MMMU、MM‑Vet、SEED、TextVQA、ChartQA、POPE、GQA（视觉理解）

**📈 对比分析**

与现有统一模型的比较：
- 在 GenEval 上获得 0.74 分，比 Show‑O 高 5%；
- 在七项视觉理解基准上平均提升 4–5%（与 JanusFlow 相比提升 4.9%）；
- 在 iPhone 17 Pro 上单张 512×512 图像生成仅需 3 s，速度比 Janus、Show‑O 快 11–46×，内存占用 < 2 GB；
- 与 2B 参数规模的统一模型相比，Mobile‑O 参数量更小（1.6B），但在理解和生成上保持或超过对手。

**⚠️ 局限性**

局限性：
- 仍需在更大规模数据或更复杂任务上验证泛化能力；
- 目前仅在特定硬件（iPhone、Jetson Nano、MacBook M2）上评测，其他边缘设备的适配度未知；
- 统一后训练依赖人工构造的四元组，可能导致某些领域的问答覆盖不足；
- 由于追求极致轻量化，模型在极高分辨率或极长文本提示的细节再现上仍略逊于大型 4‑8 B 参数模型。

---

## 881. Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning

**arXiv ID:** 2602.20157 | [PDF](https://arxiv.org/pdf/2602.20157v1)

**作者:** Zhongxiao Cong `[一作]` (Carnegie Mellon University), Shubham Tulsiani `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用无标签视频通过因子化光流监督提升可视几何学习

**💡 创新点**

提出因子化光流预测模块，将源视角几何与目标相机位姿解耦，直接引导几何与位姿学习，并可扩展至动态场景

**🔧 技术方法**

多视角Transformer、光流预测头、稠密对应网络、基于流的损失以及预训练的图像编码器

**📊 数据集**

800K无标签视频（SpatialVID、Kinetics-700、EPIC-Kitchens）与约34K有标签3D/4D数据（CO3Dv2、Habitat、ARKitScenes等）

**📈 对比分析**

在8个静态/动态基准上与CUT3R、VGGT、π^3和Flow3r比较，取得RRA、RTA、CD、MSE等指标的最高分，尤其在野外动态视频上提升显著

**⚠️ 局限性**

依赖教师模型生成伪光流，难以处理复杂多运动物体，实验规模仍为数十万帧，尚未验证向亿级视频的可扩展性

---

## 882. tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction

**arXiv ID:** 2602.20160 | [PDF](https://arxiv.org/pdf/2602.20160v1)

**作者:** Chen Wang `[一作]` (University of Pennsylvania), Yiwei Hu `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Test‑Time Training的全局3D重建模型，支持多视角高分辨率、长上下文以及自回归流式重建，并能将隐式记忆转化为显式三维表示（Gaussian Splats、NeRF三平面）。

**💡 创新点**

核心创新在于：① 引入线性复杂度的LaCT块实现Test‑Time Training，消除传统注意力的二次开销；② 将TTT快权重解释为隐式三维记忆，能够通过查询得到不同格式的显式3D；③ 设计自回归更新机制，使模型在接收连续输入时可在线重建并逐步细化。

**🔧 技术方法**

技术主要包括Test‑Time Training、Large Chunk Test‑Time Training（LaCT）、线性注意力/状态空间模型、Muon优化器、深度与不透明度正则、分布式序列并行训练，以及多视角图像的patchify + tokenization。

**📊 数据集**

训练与评估使用Objaverse（物体级）、GSO（Google Scanned Objects）、DL3DV‑10K（场景级视频）、Tanks & Temples等数据集。

**📈 对比分析**

与GS‑LRM、Long‑LRM、3DGS、Mip‑Splatting、Scaffold‑GS等基线对比，结果显示：在相同输入视角下PSNR提升约1–2 dB，SSIM与LPIPS更优；在512×512时推理速度约为Attention模型的1/2；在1024×1024时仍可训练并产生高质量重建；自回归模式随视角增多逐步提升质量；多GPU可线性加速。

**⚠️ 局限性**

主要局限在于快权重记忆尺寸固定，难以应对极大视角或极复杂场景；与预训练的隐式模型相比，显式输出的质量略有下降；目前实时性能仍未完全达到工业级要求。

---

## 883. NanoKnow: How to Know What Your Language Model Knows

**arXiv ID:** 2602.20122 | [PDF](https://arxiv.org/pdf/2602.20122v1)

**作者:** Lingwei Gu `[一作]` (University of Waterloo), Jimmy Lin `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 NanoKnow 基准数据集，利用公开的 FineWeb‑Edu 预训练语料，对 Natural Questions 与 SQuAD 的问题进行支持/不支持划分，并系统评估 nanochat 在不同答案频率、外部证据与干扰信息下的表现。

**💡 创新点**

创新点在于提供完整可追溯的预训练数据与问答映射，突破传统 LLM 知识来源不透明的瓶颈，使研究者能够精准区分参数知识与外部知识的贡献。

**🔧 技术方法**

技术上采用 BM25 检索+字符串匹配+LLM（Qwen3‑8B）验证来判定答案是否真实存在于语料中，并用 Qwen3‑14B 进行答案正确性评判，实验使用了 NanoChat 多尺度检查点。

**📊 数据集**

所用数据集为 100B 令牌的 FineWeb‑Edu 公开语料，以及 NQ 验证集 3,610 条和 SQuAD 验证集 10,570 条。

**📈 对比分析**

通过 EM 与 LLM‑Judge 两种评估方式，比较闭书与开放书、支持与不支持、模型规模以及干扰位置，结果显示参数知识受答案频率影响显著，RAG 能显著弥补稀缺答案的缺口，但在支持集上仍优于不支持集，且更多干扰会降低准确率。

**⚠️ 局限性**

局限性包括仅适用于公开预训练语料，对罕见答案仍依赖外部知识，干扰信息对 RAG 效果敏感，且实验仅在 NanoChat 体系内验证，未覆盖更大规模或不同架构的 LLM。

---

