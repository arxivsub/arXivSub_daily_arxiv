# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-03 | 今日论文总数: 883

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Constructing Everyday Well-Being: Insights from God-Saeng for Personal Informatics

**arXiv ID:** 2603.00847 | [PDF](https://arxiv.org/pdf/2603.00847v1)

**作者:** Inhwa Song `[一作]` (Princeton University), Hwajung Hong `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

通过三阶段的文化探针式定性研究（工作坊、10 天小任务和访谈）探讨了韩国青年在 God‑Saeng（“神生活”）实践中如何构建日常幸福感。研究聚焦于技术、社交与价值观如何交织，揭示了个人信息化系统在支持意义建构与价值协商中的潜在设计机遇。

**💡 创新点**

创新点：① 把日常幸福感视为在社会技术语境下的可协商建构；② 通过自定义“bite‑sized missions”将个人价值与行为实践紧密结合；③ 提出了面向意义建构与价值协商的 PI 系统设计原则，而非单纯追求行为追踪和合规性。

**🔧 技术方法**

技术方法：文化探针与自我报告问卷相结合的研究设计；基于参与式工作坊生成价值关键词；利用可视化工具（例如 MyRoutine、TimeTree、Challengers 等应用）记录与验证行为；定性主题分析和归纳编码技术。

**📊 数据集**

数据集：24 名 20–26 岁韩国青年（8 男，16 女），包含工作坊记录、价值关键词卡片、10 天任务日志、访谈转录。每位参与者提供约 10–20 张个人意义照片作为反思触发。

**📈 对比分析**

由于研究采用的是定性方法，没有数值化的对比实验或性能评估；研究结果主要通过主题归纳呈现，未提供可量化的指标或与其他系统的客观比较。

**⚠️ 局限性**

局限性：① 样本规模与文化背景限定，难以推广至不同族群或地区；② 采样方式自我报告，可能存在社会期望偏差；③ 研究周期仅 10 天，缺乏长期追踪；④ 依赖同辈组讨论，可能导致共识偏差；⑤ 未进行跨文化或跨实践的比较研究。

---

## 2. A Deep Learning Framework for Heat Demand Forecasting using Time-Frequency Representations of Decomposed Features

**arXiv ID:** 2603.01137 | [PDF](https://arxiv.org/pdf/2603.01137v1)

**作者:** Adithya Ramachandran `[一作]` (Pattern Recognition Lab, Friedrich Alexander Universität), Siming Bayer `[通讯]` (Pattern Recognition Lab, Friedrich Alexander Universität)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于连续小波变换（CWT）将多变量时间序列映射为二维时频图像，再通过卷积神经网络（CNN）进行24小时一步预测的热负荷预测框架。

**💡 创新点**

创新点在于：① 将时频表示作为输入，提升CNN捕捉非线性、周期性特征的能力；② 对原始时序进行趋势、季节、残差拆分后再进行CWT，显著提升预测精度；③ 通过系统性实验验证并对比多种传统统计、机器学习、Transformer及基础模型，建立统一评估基准。

**🔧 技术方法**

使用连续小波变换、CNN（无池化层、Dropout）、时间序列分解（trend/season/residual）、特征选择与编码（循环编码、节假日处理）以及对照模型包括SARIMAX、XGBoost、LSTM、Informer、Autoformer、PatchTST、TimesNet、DLinear、TSMixer、TimeMixer、Chronos-2、TTM等。

**📊 数据集**

数据集：丹麦Brønderslev市3个district metered areas（2016‑2019年）、德国Flensburg市（2017‑2024年）、丹麦Aalborg市住宅用热量（2018‑2020年）。

**📈 对比分析**

与传统统计模型、XGBoost、LSTM、Transformer（Informer、Autoformer、PatchTST、TimesNet）以及基础模型进行统一实验，结果显示本方法平均MAE下降36‑43%（约比最佳基线低一半），MAPE约5–6%，误差分布更集中、极端误差更小，且在不同地区和年份保持稳健。

**⚠️ 局限性**

局限性：对罕见事件（如节假日、异常突发事件）的预测仍受限；小波基选取固定，未探索自适应多波基；依赖高质量的外部天气与日历数据；对极端温度变化和设备故障的鲁棒性尚未完全验证。

---

## 3. Truth as a Trajectory: What Internal Representations Reveal About Large Language Model Reasoning

**arXiv ID:** 2603.01326 | [PDF](https://arxiv.org/pdf/2603.01326v1)

**作者:** Hamed Damirchi `[一作]` (Australian Institute for Machine Learning), Javen Shi `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Truth as a Trajectory (TaT) 方法，将LLM推理视为层级轨迹，利用激活差分捕捉推理过程中的几何变化，以此检测推理有效性

**💡 创新点**

创新点在于把静态层激活转为动态位移轨迹，利用轨迹的几何不变量而非表面词汇进行解释和监控，展示了跨任务可迁移的推理特征

**🔧 技术方法**

使用残差流差分构建轨迹，输入LSTM进行序列学习，并与线性探针、LoRA等对比，同时评估多种模型的计算开销

**📊 数据集**

在ARC-Easy/Challenge、BoolQ、Hellaswag、OpenBookQA、StoryCloze、CommonsenseQA、CosmosQA、SocialIQA等推理基准以及RealToxicityPrompts和ToxiGen等毒性检测数据集上进行实验

**📈 对比分析**

与基线（线性探针、模型原生零样本/少样本推理、LoRA）比较，TaT在多任务 OOD 迁移中平均提升约15‑20%，在毒性检测中对上下文毒词鲁棒性提升约5‑7个百分点

**⚠️ 局限性**

计算成本较高，需要提取完整的残差流并执行LSTM，且模型内部学习的几何特征缺乏可解释性，仍需训练数据才能泛化

---

## 4. CyclicJudge: Mitigating Judge Bias Efficiently in LLM-based Evaluation

**arXiv ID:** 2603.01865 | [PDF](https://arxiv.org/pdf/2603.01865v1)

**作者:** Ziyi Zhu `[一作]` (Slingshot AI), Jinghong Chen `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种方差分解方法，用于量化LLM评判中的场景、生成、评审偏差和残差，并设计了CyclicJudge循环分配评审，消除偏差且保持成本。

**💡 创新点**

创新点在于：①将评审偏差与随机噪声分离；②证明循环分配比全评审或随机评审更优；③提供闭式理论证明并在MT‑Bench上实证验证。

**🔧 技术方法**

采用混合效应模型、方差分解（基于Generalizability Theory）以及统计检验（ANOVA、Bootstrap）来评估偏差和方差。

**📊 数据集**

使用MT‑Bench数据集（80个两轮问题，8类），对Qwen 2.5 7B、Llama 3.3 70B和GPT‑5.2三模型生成10条对话，5名LLM评审。

**📈 对比分析**

与全评审（All‑judges）和随机评审（Random）策略比较，CyclicJudge在任何预算下方差均最低，尤其在小预算时可降低约30–35%，有效消除评审偏差。

**⚠️ 局限性**

局限性：①线性随机效应模型未考虑分数的有序约束；②评审人数仅5人，未验证更大规模；③假设场景可等价抽样，未考虑场景重要性；④仅针对平衡设计，缺乏自适应策略和成本加权考虑。

---

## 5. LLM-as-an-Annotator: Training Lightweight Models with LLM-Annotated Examples for Aspect Sentiment Tuple Prediction

**arXiv ID:** 2603.01778 | [PDF](https://arxiv.org/pdf/2603.01778v1)

**作者:** Nils Constantin Hellwig `[一作]`, Christian Wolff `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了 LA-ABSA，一种利用大语言模型（Gemma‑3‑27B）自动给无标注文本打标注，并用这些人工智能生成的标注数据微调轻量级模型（T5‑base 的 DLO 或 Paraphrase）来完成目标方面情感检测（TASD）与方面情感四元组预测（ASQP）任务。

**💡 创新点**

创新点在于：①首次将 LLM 作为自动标注器用于 ABSA 细粒度 tuple 预测，显著降低人工标注成本；②通过自一致性（多次生成并投票）提升标注质量；③在能耗方面对比 LLM 直接提示，证明在大规模推理时轻量模型更节能；④在低资源场景下系统性评估 LLM 标注、零/少样本提示、数据增广及全量人工标注四种策略的性能差异。

**🔧 技术方法**

技术：使用 Gemma‑3‑27B 进行自监督式标注；采用自一致性采样（5 次生成，投票≥3）筛选最终标签；对生成的标注与原始文本做合法性校验；使用 T5‑base（223M）微调的 DLO 与 Paraphrase 两种轻量模型；在 NVIDIA RTX A5000 GPU 上进行训练与评估；能耗分析基于 GPU 的瓦时计量。

**📊 数据集**

数据集：五个领域的公开 ABSA 数据集，分别是 SemEval 2015/2016（Rest15、Rest16）、FlightABSA、Coursera 与 Hotels；每个数据集均提供 TASD（triplet）与 ASQP（quad）标注。

**📈 对比分析**

与方法对比：在 0-shot、10-shot 与 50-shot 场景下，LA‑ABSA 的 F1 与 LLM 直接提示相差不超过 4–5%（部分 0-shot 场景甚至优于提示），且显著优于基于 EDA、QAIE、DS2‑ABSA 的增广策略；相较于全量人工标注的微调模型（DLO/Paraphrase），性能略低，但能耗大幅降低（例如 TASD 任务中每样本能耗从 703 mWh 降至 2.5–3 mWh）。

**⚠️ 局限性**

局限性：①仅评估单一 LLM（Gemma‑3‑27B），缺乏多模型比较；②生成标注质量仍低于人工标注，导致最终模型性能仍落后于全量人工微调；③未使用具备推理步骤的 LLM，可能进一步提升标注质量但会增加推理时间；④能耗与速度评估仅基于 RTX A5000，结果对不同硬件需进一步校准。

---

## 6. Toward Graph-Tokenizing Large Language Models with Reconstructive Graph Instruction Tuning

**arXiv ID:** 2603.01385 | [PDF](https://arxiv.org/pdf/2603.01385v1)

**作者:** Zhongjian Zhang `[一作]` (Beijing University of Posts and Telecommunications), Chuan Shi `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 15702 | [OpenAlex ID](https://openalex.org/A5100705849)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于重构的图指令微调流程（RGLM），通过在大型语言模型（LLM）中加入图重构任务显式对齐图与文本信息；

**💡 创新点**

创新点在于：①揭示现有GTokenLLM仅依赖文本监督导致图–文本对齐隐式、偏文本；②证明对齐目标上界为图与LLM隐藏表示的互信息，进而通过重构提升该上界；③设计三种RGLM变体（Decoder、Similarizer、Denoiser），从输入空间和潜在空间分别实现图重构；

**🔧 技术方法**

技术手段包括：图序列化（Neighbor Detail Template）、投影器对齐、LLM解码器、LoRA微调、图重构损失（特征MSE、结构BCE、余弦相似度、去噪MSE）、预训练GNN编码器、Diffusion去噪、信息理论分析；

**📊 数据集**

使用的公开图数据集有Cora、Pubmed、OGBN‑Arxiv、Reddit；

**📈 对比分析**

与传统GNN、图Transformer、预训练GNN、GTextLLM及GTokenLLM等多种基线进行对比，在节点分类、边预测、多数据集泛化及零样本任务上，RGLM显著提升准确率（0.1–30+点），并保持在多种LLM（LLaMA3‑8B、Vicuna‑13B）上的兼容性；

**⚠️ 局限性**

局限性包括：重构任务作为正则项，需调参以平衡文本与图监督；当图结构或特征信息缺失时，重构难以充分发挥；当前仅在子图级别实验，未深入大规模全图推理；对LLM的计算资源依赖仍较高。

---

## 7. Constructive and Predicative Locale Theory in Univalent Foundations

**arXiv ID:** 2603.01308 | [PDF](https://arxiv.org/pdf/2603.01308v1)

**作者:** Ayberk Tosun `[一作]` `[通讯]`, Ayberk Tosun

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 8. PathMoE: Interpretable Multimodal Interaction Experts for Pediatric Brain Tumor Classification

**arXiv ID:** 2603.01547 | [PDF](https://arxiv.org/pdf/2603.01547v1)

**作者:** Jian Yu `[一作]` (University of Texas), Ankita Shukla `[通讯]` (University of Nevada)

**通讯引用:** 465 | [OpenAlex ID](https://openalex.org/A5023797933)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种可解释的多模态融合框架 PathMoE，用于儿科脑肿瘤分类，融合 H&E 全切片图像、病理报告文本和细胞图网络。

**💡 创新点**

创新点在于：① 采用交互感知的 Mixture‑of‑Experts（MoE）架构，将模态独特性、冗余与协同作为专家；② 通过输入依赖的门控网络实现样本级权重分配和解释；③ 将结构化细胞图作为领域知识引入，提升对稀有亚型的判别能力。

**🔧 技术方法**

使用了 UNIv2、TITAN 等基础模型进行图像与文本编码；GraphSAGE 进行细胞图嵌入；MIL、Transformer、注意力聚合；交互感知 MoE 与门控网络；损失函数包含交互正则化。

**📊 数据集**

内部数据集 PBT（253 张 WSI，196 名患者，四分类）；外部数据集 TCGA‑GBM / TCGA‑LGG（208 张 WSI，67 名患者，三分类）。

**📈 对比分析**

与 CLAM、TransMIL、S4MIL、MambaMIL 等单模态 MIL 基线以及 EF_W、SG_W 融合基线进行比较；在 PBT 上宏 F1 从 0.762 提升到 0.799（+0.037），在 TCGA 上从 0.668 提升到 0.709（+0.041），显示显著性能提升。

**⚠️ 局限性**

局限性包括：① 样本规模有限，稀有亚型样本不足导致泛化性受限；② 细胞图构建依赖核检测精度，错误会影响特征；③ 文本报告噪声在某些外部数据集影响效果；④ 模型复杂度高，训练与推理成本较大；⑤ 解释性仍需专家人工验证。

---

## 9. OMG-Avatar: One-shot Multi-LOD Gaussian Head Avatar

**arXiv ID:** 2603.01506 | [PDF](https://arxiv.org/pdf/2603.01506v1)

**作者:** Jianqiang Ren `[一作]` (Alibaba Group), Steven Hoi `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了OMG-Avatar，一种基于多级高斯表示的单图像3D头部全身可动画化模型，能够实时生成可调节LOD的头部/肩部头像并实现高质量重演；

**💡 创新点**

创新点包括：①使用分层全局-局部特征提取结合深度缓冲的遮挡感知特征融合；②在训练中逐级细分网格实现粗细细化学习，支持运行时动态LOD；③采用多区域建模分离头部和肩部并融合，解决非头部区域细节不足；④在单张图片上实现0.2s的全流程构建与85FPS的实时渲染；

**🔧 技术方法**

技术主要包括：DINOv2特征提取、FLAME 3DMM、Transformer跨注意力全局特征、投影采样局部特征、Occlusion-Aware Feature Fusion（OAFF）、多级网格细分、Gaussian Splatting渲染、UNet神经细化器；

**📊 数据集**

使用VFHQ（766k帧）作为训练集，采用HDTF进行跨数据集泛化评估；

**📈 对比分析**

与包括ROME、StyleHeat、OTAvatar、HideNeRF、GOHA、CVTHead、GPAvatar、Real3DPortrait、Portrait4D系列、GAGAvatar、LAM等多种SOTA方法在自重演与跨重演的PSNR/SSIM/LPIPS/CSIM/AED/APD/AKD等指标对比，OMG-Avatar在所有指标上均取得领先，且在A100 GPU上达到85FPS、RTX4090上126FPS；

**⚠️ 局限性**

局限性：①依赖FLAME 3DMM，无法精准模拟舌头、发型等细节；②单视角训练导致对大视角（±60°外）鲁棒性不足，计划加入多视角数据增强以提升空间理解与视角泛化。

---

## 10. DOCFORGE-BENCH: A Comprehensive Benchmark for Document Forgery Detection and Analysis

**arXiv ID:** 2603.01433 | [PDF](https://arxiv.org/pdf/2603.01433v1)

**作者:** Zengqi Zhao `[一作]` (University of North Carolina), Simiao Ren `[通讯]` (Scam.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个统一的零样本文档伪造检测基准，使用14种预训练模型在8个不同文档数据集上进行不进行微调的评估。

**💡 创新点**

创新点在于揭示了文档域中普遍存在的校准失效（AUC–F1间隙），并证明仅通过在少量域内图像上调阈值即可恢复一半以上性能；同时指出现有数据集缺乏生成式AI攻击的覆盖。

**🔧 技术方法**

采用了14种图像取证与文档特定取证算法（如TruFor、CAT-Net、FFDN、ASCFormer等），并使用像素级评估指标：Pixel‑F1（τ=0.5）、Pixel‑AUC、Oracle‑F1。

**📊 数据集**

使用的数据集包括：DocTamper、T‑SROIE、RealTextManipulation、Tampered‑IC13、ReceiptForgery、MixTamper、FSTS‑1.5k 和 FantasyID，涵盖文本篡改、收据伪造与身份文件篡改等多种威胁模型。

**📈 对比分析**

比较结果显示：大多数方法在Pixel‑AUC上表现良好（≥0.76）但在固定阈值下Pixel‑F1接近零；校准后可恢复39–55% Oracle‑F1，文档特定训练在跨域测试中并未优于通用模型，整体仍未实现可靠的零样本性能。

**⚠️ 局限性**

局限性包括：极端类别不平衡导致阈值校准失效；仅评估像素级定位，未覆盖多模态或文本内容分析；所用数据集未涵盖扩散模型或LLM生成的文档伪造；对模型进行微调或改进架构仍需进一步研究。

---

## 11. Individual Turing Test: A Case Study of LLM-based Simulation Using Longitudinal Personal Data

**arXiv ID:** 2603.01289 | [PDF](https://arxiv.org/pdf/2603.01289v1)

**作者:** Minghao Guo `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了“Individual Turing Test”来评估LLM在长期个人对话数据上的个体模拟能力，并在一份长达十年的私人聊天记录上对比了微调、检索增强、记忆化以及混合方法。

**💡 创新点**

创新点在于：①首次将Turing Test扩展为针对个体身份的评估；②揭示了参数化微调与非参数检索/记忆在语言风格与观点一致性之间的权衡；③证明了时效性对模拟效果的重要性，并指出混合模型在两方面均优于单一方法。

**🔧 技术方法**

使用技术包括LoRA参数微调、检索增强生成（RAG）与基于记忆的A-Mem、以及LoRA+RAG / LoRA+A-Mem的混合架构；后端模型为Qwen2.5-7B；检索采用BGE-M3嵌入，解码采用受控温度、重复惩罚等策略。

**📊 数据集**

数据集为一名志愿者提供的约十年私人消息记录，包含12,151段对话、72,652条消息、1,157,842个训练标记，经过时间窗口与去重等预处理后构成实验材料。

**📈 对比分析**

评估方法包括人类判断的“Individual Turing Test”和“General Turing Test”以及自动相似度指标（BLEU、ROUGE、Precision、Recall、Distinct）。结果显示：混合方法在两种测试中均取得最高分，但仍低于真实回复；在一般测试中部分模型甚至超过真实回复；在个体测试中真实回复优于所有模拟方法，表明真实性差距依旧。

**⚠️ 局限性**

局限性：①单一受试者的数据与评估者样本有限，缺乏跨人群泛化；②评估主要依赖熟识者与陌生人，对话场景不够多样；③记忆检索策略与时间窗口设置为经验式，未充分探索动态与可解释的记忆管理；④当前模型在个体身份模拟上仍无法完全逼近真实表现，真实性差距仍显著。

---

## 12. Tide: A Customisable Dataset Generator for Anti-Money Laundering Research

**arXiv ID:** 2603.01863 | [PDF](https://arxiv.org/pdf/2603.01863v1)

**作者:** Montijn van den Beukel `[一作]` (University of Amsterdam), Ana-Lucia Varbanescu `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 Tide——一种可复现、可定制的合成金融交易网络生成器，能够在注入洗钱模式时同时控制结构与时间特征，并发布了两个参考数据集（低欺诈率0.10%、高欺诈率0.19%）；

**💡 创新点**

创新点在于将传统过程驱动的生成器从仅关注结构拓扑扩展到同时模拟交易时序与频率，并提供精细可配置的模式注入接口，弥补了现有 AMLSim、AMLWorld、SynthAML 等工具在时间维度和可扩展性上的不足；

**🔧 技术方法**

采用基于图的生成框架，结合风险评分、实体聚类、对数正态分布的交易金额与频率模型，配置化 YAML 控制流程，并利用 GNN（GIN、GIN+EU、PNA）与 GBT（LightGBM、XGBoost）作为基线模型进行评估；

**📊 数据集**

发布了两个数据集：LI（36,629 结点、7,642,030 交易、欺诈比例0.10%）和 HI（36,653 结点、7,618,299 交易、欺诈比例0.19%），均基于 8,000 名个人、12 个月交易周期，并提供节点/边 CSV 与 PyTorch Geometric/NetworkX 格式；

**📈 对比分析**

采用时间切分（60%/20%/20%）并在验证集上使用 Youden’s J 确定阈值，评估 PR‑AUC、F1、Precision、Recall 等指标；结果显示 LightGBM 在低欺诈率条件下 PR‑AUC 78，XGBoost 在高欺诈率条件下 PR‑AUC 85，模型排名随条件变化，证明生成器能有效区分不同模型的性能；

**⚠️ 局限性**

局限性包括仅实现五种洗钱模式，未覆盖贸易、房地产、加密货币等更复杂方案；依赖美国支付数据，泛化性有限；未能与真实交易数据验证；未对不同模式的检测难度做细粒度分析；缺乏对抗式演化和跨机构协同的研究。

---

## 13. Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning

**arXiv ID:** 2603.02070 | [PDF](https://arxiv.org/pdf/2603.02070v1)

**作者:** Guilhem Fouilhé `[一作]` (IRIT), Nicholas Asher `[通讯]` (CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多代理LLM的交互式规划框架，支持用户通过自然语言提问并获得逐步改进计划的解释；

**💡 创新点**

创新点在于将LLM集成为多角色翻译器和解释生成器，实现了与传统模板化界面相比更灵活、更用户友好的交互式解释流程；

**🔧 技术方法**

核心技术包括GPT‑4.1‑mini LLM代理、自然语言到形式化目标和问题的翻译器、基于MUS/MCS的冲突解释模型以及基于LLM的摘要与选择机制；

**📊 数据集**

使用了“Parent’s Afternoon”规划任务的人工设定实例，包含19个目标、224个冲突和313个修正集，作为用户研究的实验数据；

**📈 对比分析**

通过与模板化问答界面对比的用户研究，LLM接口在主观可用性上显著优于基线，且在目标效用达成上表现出更快的收敛（差异虽不显著但趋势积极）；

**⚠️ 局限性**

主要限制包括仅在单一中等难度实例上验证，未涉及领域专家用户，且LLM在解释摘要的准确性与泛化性仍待进一步评估。

---

## 14. Characterization of Blind Code Rate Recovery in Linear Block Codes

**arXiv ID:** 2603.02031 | [PDF](https://arxiv.org/pdf/2603.02031v1)

**作者:** Atreya Vedantam `[一作]` (Indian Institute of Technology Madras), Radha Krishna Ganti `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种新的指标 E[C] 并基于软信息的秩估计方法来盲恢复线性块码的码率。

**💡 创新点**

创新点是给出 E[C] 的闭式表达、利用它修正码率估计并优化参数 t1、t2，实现比现有方法更低的码字需求。

**🔧 技术方法**

采用软判决、矩阵秩估计、Gaussian elimination、BPSK + AWGN 传输模型、统计分析。

**📊 数据集**

使用 5G NR 标准 LDPC 码（长度 544 与 1088）生成的 1000/2000 条码字。

**📈 对比分析**

与 Ramabadran、Wang 等方法对比，实验表明在 10–12 dB SNR 下仅需 1000 条码字即可逼近真码率，而其他方法需要数十万码字，性能显著提升。

**⚠️ 局限性**

局限在于对大块长码字的恢复仍受噪声环境限制，需要高 SNR；此外需足够多的码字且算法对 t1、t2 的选择较敏感。

---

## 15. Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery

**arXiv ID:** 2603.01034 | [PDF](https://arxiv.org/pdf/2603.01034v1)

**作者:** Yangyang Xu `[一作]` (Hunan Normal University), Chao Wang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 104228 | [OpenAlex ID](https://openalex.org/A5100339418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

论文提出了一种将 Tensor Ring (TR) 分解扩展到连续域的函数式 TR 分解，并通过重新参数化的 TR 因子来提升高频细节的学习效果。

**💡 创新点**

创新点在于：① 通过频域分析揭示 TR 因子频谱对重构结果的影响；② 设计了由可学习潜在张量和固定基底组成的结构化重参数化方法，显著改善训练动态和高频建模；③ 给出了理论证明的基底初始化与 Lipschitz 连续性。

**🔧 技术方法**

使用的技术包括：Implicit Neural Representations (INR)、多层感知机 (MLP)、共享正弦频率嵌入、TR 功能化约束、重参数化与基底初始化、Lipschitz 连续性分析。

**📊 数据集**

实验数据集涵盖：图像修复、去噪、超分辨率 (DIV2K) 以及点云恢复 (SHOT)，其中包含彩色图像、多光谱/高光谱图像、视频序列和稀疏采样点云。

**📈 对比分析**

与 TRLRF、FCTN、HLRTF、LRTFR、DRO‑TFF、NeurTV 等传统及 INR 基线方法进行对比，实验结果显示在 PSNR/SSIM（或 NRMSE）上平均提升约 1–2 dB，且在高频细节恢复上表现更优，计算成本保持在可接受范围。

**⚠️ 局限性**

局限性包括：① 对基底初始化尺度敏感，需要经验调参；② 重新参数化后模型参数略增，训练时间略长；③ 目前仅在 TR 结构上验证，尚未推广到 Tucker 或块项分解等其他张量分解形式。

---

## 16. TITAN: Twin-Informed Topology Adaptation for LAWN-enabled D2C Communication

**arXiv ID:** 2603.00795 | [PDF](https://arxiv.org/pdf/2603.00795v1)

**作者:** Talip Tolga Sarı `[一作]` (Istanbul Technical University), Debashri Roy `[通讯]` (University of Texas at Arlington)

**通讯引用:** 965 | [OpenAlex ID](https://openalex.org/A5010877111)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 TITAN 框架，通过数字孪生 (Digital Twin) 与射线追踪评估，利用 Bayesian 优化在 LAWN（低空无线网络）中自适应布置无人机（UAV），实现灾后城市环境下的 D2C（Direct‑to‑Cell）通信覆盖与 QoS 最优化。

**💡 创新点**

创新点：① 将高精度数字孪生与射线追踪整合到网络规划；② 采用 Bayesian 优化在高维 3D UAV 布局搜索空间中高效逼近全局最优；③ 引入动态适应机制，能够在灾害演化时重新优化网络拓扑；④ 以综合覆盖/吞吐/公平性为目标函数，兼顾多指标；⑤ 公开完整代码与数据，促进复现与社区协作。

**🔧 技术方法**

技术手段：Sionna RT 射线追踪、基于 TensorFlow 的系统级仿真、3D 环境网格、Starlink TLE 卫星轨道、贝叶斯优化（Tree‑structured Parzen Estimator）、Proportional‑Fair 调度、RZF 预编码与 LMMSE 判决。

**📊 数据集**

数据集：San Francisco 区域的高分辨率 3D 网格（建筑、地形）、随机生成的 100 个 UE 位置信息、Starlink 卫星可见性 TLE 数据、仿真生成的链路脉冲响应与系统级日志。

**📈 对比分析**

比较方法：① 传统 D2C（LoS 直连），② 随机 UAV 布局，③ Bayesian + 统计 3GPP TR38.901 通道模型，④ 仅以吞吐为目标的 Bayesian，⑤ 现有 SOTA UAV 放置算法。实验表明 TITAN 在完整基础设施瘫痪时相较 SOTA 提升用户覆盖率 32.2%、系统总吞吐 64.9% 与公平性 49.3%，在部分基站失效与动态重优化场景亦保持优异性能。

**⚠️ 局限性**

局限性：① 需要高质量 3D 环境网格与实时更新，获取成本高；② 射线追踪与 Bayesian 优化仍有计算开销，低延迟部署仍受限；③ 对 UE 定位误差和数字孪生精度敏感，误差会影响最终布局；④ 仅在模拟环境验证，实际灾区硬件、能量与法律限制需进一步研究。

---

## 17. Towards Computing Average Merge Tree Based on the Interleaving Distance

**arXiv ID:** 2603.00783 | [PDF](https://arxiv.org/pdf/2603.00783v1)

**作者:** Elena Farahbakhsh Touli `[一作]` (Linköping University), Talha Bin Masood `[通讯]` (Linköping University)

**通讯引用:** 573 | [OpenAlex ID](https://openalex.org/A5060181768)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出了在交错距离下计算两棵合并树的平均合并树的方法，并给出了构造代表平均树的具体算法；

**💡 创新点**

创新点在于证明平均树虽存在但非唯一，提出一种构造代表平均树的策略，并通过简化的ε‑good map定义证明其满足自然的平均性质；

**🔧 技术方法**

主要技术包括交错距离与ε‑good map的理论框架、树的增强与动态规划、FPT算法用于高效计算交错距离、以及函数值平移和最小公共祖先处理；

**📊 数据集**

实验数据采用两个叶子（leaf）数据集，利用平均测地距离作为标量场来构建合并树；

**📈 对比分析**

通过构造的平均树使与原树的交错距离均不超过原距离的一半，算法复杂度主要由ε‑good map的计算决定，最坏时间为O(n³·2^b·(b+1))，若参数b有界则可实现多项式时间；

**⚠️ 局限性**

局限性包括仅处理两棵树的平均问题、交错距离计算仍为NP‑hard、算法对树结构有一定依赖、以及平均树的非唯一性导致可能存在多种代表平均树。

---

## 18. ATA: Bridging Implicit Reasoning with Attention-Guided and Action-Guided Inference for Vision-Language Action Models

**arXiv ID:** 2603.01490 | [PDF](https://arxiv.org/pdf/2603.01490v1)

**作者:** Cheng Yang `[一作]` (Rutgers University), Bo Yuan `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ATA（Attention‑Guided & Action‑Guided inference）框架，在 Vision‑Language‑Action (VLA) 模型推理时通过注意力图与动作方向 ROI 进行隐式推理，无需额外训练或标注。

**💡 创新点**

创新点在于：① 训练无关的隐式推理；② 将模型内部注意力信息与机器人运动意图融合；③ 兼容高效注意力实现（如 FlashAttention），在单次前向推理中完成指导。

**🔧 技术方法**

采用了 VLA 模型（OpenVLA、π₀‑fast、HybridVLA、GR00T‑N1.5），注意力提取、动作导向 ROI 生成、掩码混合；不使用任何外部标注或额外数据集。

**📊 数据集**

使用了 LIBERO（Spatial、Goal、Object、Long）四子集、RLBench（8项任务）以及真实世界 3cm 方块堆叠数据集（50条测试配置），共计 500 条轨迹。

**📈 对比分析**

通过与基线、API、单一策略（Attention‑Guided 或 Action‑Guided）比较：在 LIBERO 上 OpenVLA 提升 5.2%、π₀‑fast 提升 2%；在 RLBench 上 HybridVLA 提升 5.5%；真实世界堆叠任务提升 10% 以上；同时平均推理调用数下降（如 OpenVLA 从 235↓225，π₀‑fast 从 41↓39）。

**⚠️ 局限性**

局限性：需要针对不同模型和任务手动调节注意层、触发频率和动作引导时机；验证范围主要集中在四个模型，缺乏更广泛的跨域鲁棒性评估；在复杂动态场景下仍可能出现掩码误导导致误操作。

---

## 19. End-to-End Simultaneous Dysarthric Speech Reconstruction with Frame-Level Adaptor and Multiple Wait-k Knowledge Distillation

**arXiv ID:** 2603.01382 | [PDF](https://arxiv.org/pdf/2603.01382v1)

**作者:** Minghui Wu `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8100 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种端到端的同步失语症语音重建系统，将流式 ASR 与增量式 TTS 通过帧级适配器耦合，实现低延迟且自然的重建。

**💡 创新点**

创新点：① 帧级适配器采用显式-隐式语义融合和门控激励，提升 TTS 对 ASR 错误的鲁棒性；② 多 wait‑k 自回归 TTS 与知识蒸馏结合，在有限感受野内兼顾低延迟与高语调自然度。

**🔧 技术方法**

使用的技术包括 Conformer‑RNNT 流式 ASR、GPT‑based 自回归 TTS、VQ 量化声码器、SwitchGLU 门控线性单元、知识蒸馏、两阶段预训练与微调。

**📊 数据集**

训练与评测数据集：商业中文弱发音语料 500h（3,000 说话人）和 UASpeech 英文失语症数据 29 说话人。

**📈 对比分析**

与传统句级 ASR+TTS、SOTA 失语症语音重建方法对比，平均 WER 降低 54.25%，MOS 提升至 4.67；系统平均响应时间 1.03 s，实时因子 RTF 0.71。

**⚠️ 局限性**

局限性：在极低延迟场景下仍需进一步压缩多 wait‑k 感受野；对不同严重度失语症的适应性需进一步验证；低资源数据仍可能限制鲁棒性。

---

## 20. Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics

**arXiv ID:** 2603.01209 | [PDF](https://arxiv.org/pdf/2603.01209v1)

**作者:** Victor May `[一作]` (Ontocord), Huu Nguyen `[通讯]` (Ontocord)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在单一任务族“Opaque Knapsack”上做了一个 2×2 的实验，交叉对比训练时的执行语义（持久 vs. 无持久）与部署时的执行语义（持久 vs. 无持久），研究它们对工具增强型语言模型的状态管理、效率与鲁棒性的影响。

**💡 创新点**

创新点是把解释器持久性视为可学习的训练语义而非仅仅是推理时的实现细节，并构造了一个不可一次性求解、需要多轮交互的非可折叠基准（Opaque Knapsack）来显式检验持久性对行为的影响。

**🔧 技术方法**

使用 CodeAct 风格的教师生成交互轨迹，对模型做 LoRA 微调，评估时交替自然语言推理与可执行 Python 代码；测量指标包括归一化最优度、token 消耗、状态利用率等。

**📊 数据集**

数据集是自定义的 Opaque Knapsack，包含两种难度级别（Easy、Hard），在同一任务实例上分别生成持久和无持久的轨迹，以保证只改变执行语义。

**📈 对比分析**

比较方法是训练两套适配器（持久训练 vs. 无持久训练），然后分别在持久和无持久的运行时下评估，得到四个组合；实验显示对齐的持久组合在 token 消耗上大幅优于无对齐组合，鲁棒性更好，解决质量基本相同；无持久训练在持久运行时仍保留“遗忘税”，但总体仍比持久训练的无对齐模型更高效。

**⚠️ 局限性**

局限性包括样本量有限（仅 100 任务/难度），训练时使用的 token 量不匹配（无持久轨迹更长），实验只覆盖单一模型与单一任务族，且运行时提供的状态元数据可能掺杂了持久性信号，未能完全分离。

---

## 21. RAG-RUSS: A Retrieval-Augmented Robotic Ultrasound for Autonomous Carotid Examination

**arXiv ID:** 2603.01153 | [PDF](https://arxiv.org/pdf/2603.01153v1)

**作者:** Dianye Huang `[一作]` (Technical University of Munich), Zhongliang Jiang `[通讯]` (Technical University of Munich)

**通讯引用:** 4014 | [OpenAlex ID](https://openalex.org/A5030672669)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种可解释的、检索增强的机器人超声系统RAG‑RUSS，能够按临床流程自动完成颈动脉超声检查，并实时给出当前扫描阶段、解释和下一步操作；

**💡 创新点**

创新点在于将大语言模型与医学视觉模型结合，并引入检索增强生成（RAG）机制，通过检索类似案例提供上下文，实现对扫描阶段和动作的可解释决策，既降低对大规模训练数据的依赖，又提升透明度；

**🔧 技术方法**

采用Vicuna 7B LLM作为主干，PubMedCLIP ViT作为视觉编码器，双层MLP进行跨模态投影，ResMLP与三元组损失构建检索器，LoRA微调以及多轮问答检索增强推理流程；

**📊 数据集**

基于32名健康志愿者采集的颈动脉超声体积扫描构建数据集，共计15,459条含图像+文本注释的条目，分为训练集（28人）与测试集（4人），并分别拆分为RAG、跨模态预训练与微调子集；

**📈 对比分析**

通过与仅使用RAG、仅使用VLM以及不同检索量（1或2）的对照进行闭环评估，RAG‑RUSS@2平均准确率达79.1%，在多阶段识别上显著优于基线，检索提升约10%精度；

**⚠️ 局限性**

局限性包括计算资源消耗大，检索上下文越多推理时间越长；仅使用离散API控制而非连续动作，可能限制细粒度扫描；系统仅在预录人类体积数据验证，真实场景中的变形与噪声仍待进一步测试。

---

## 22. SINR Estimation under Limited Feedback via Online Convex Optimization

**arXiv ID:** 2603.02061 | [PDF](https://arxiv.org/pdf/2603.02061v1)

**作者:** Lorenzo Maggi `[一作]` (NVIDIA), Alexander Keller `[通讯]` (NVIDIA)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于在线凸优化的SINR估计框架，利用ACK/NACK、CQI报告及先前的MCS决策，通过镜像梯度下降和Nesterov动量实现实时SINR跟踪，并通过固定分享专家算法实现超参数自适应调节。

**💡 创新点**

创新点在于：①将SINR估计问题重新表述为带正则化的OCO问题；②将ACK/NACK误差与CQI正则化结合，得到可解释的凸组合更新；③将Nesterov动量引入OCO估计，提升跟踪速度；④利用专家学习实现在线自调参，实现持续学习。

**🔧 技术方法**

核心技术包括：在线凸优化（镜像梯度下降）、Nesterov加速动量、正则化（CQI平方误差）、专家学习（固定分享算法）、BCE损失、线性组合与滑动窗口调参。

**📊 数据集**

实验基于Sionna-RT射线追踪生成的100条3000时隧SINR轨迹，覆盖柏林城市地图的不同移动和多径环境；同时使用行业标准OllA、SALAD等方案进行对比。

**📈 对比分析**

与SALAD、OLLA等现有链接适配方案对比，固定分享专家方法在约90%场景下优于所有单一专家，平均SINR估计误差显著降低（RMSE提升约10-20%），并在自适应MCS选择实验中展示了更好的信噪比估计能力。

**⚠️ 局限性**

主要局限：1) 需要先行构造多组专家，计算复杂度线性于专家数；2) 估计与MCS探索-利用权衡之间存在折中，过度探索会降低光谱效率；3) 对极端瞬时SINR跳变的鲁棒性仍有限；4) 实验集中在单天线、特定城市场景，需验证在更广泛部署中的泛化性。

---

## 23. A two-steps tensor eigenvector centrality for nodes and hyperedges in hypergraphs

**arXiv ID:** 2603.01513 | [PDF](https://arxiv.org/pdf/2603.01513v1)

**作者:** Qing Xu `[一作]` (Harbin Engineering University), Jihong Shen `[通讯]` (Harbin Engineering University)

**通讯引用:** 2163 | [OpenAlex ID](https://openalex.org/A5040210816)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于三阶张量的两步特征向量中心性，用于评估一般超图中节点和超边的重要性。

**💡 创新点**

创新点在于将超图的两步走路编码为第三阶张量，并通过张量的Perron向量定义唯一正向量中心性，同时提供几何容量的组合解释。

**🔧 技术方法**

使用了非负张量的Perron‑Frobenius理论、幂迭代算法以及两步扩展树的几何容量递归定义。

**📊 数据集**

实验使用了 Math‑StackExchange 共标记超图和 Walmart‑Trips 购物车超图。

**📈 对比分析**

与线性、最大值和 Log‑Exp 三种中心性方法进行散点与秩相关性比较，结果显示 HTEC 与线性方法一致性最高、与 Log‑Exp 差异最大，但能更好识别重要节点和超边。

**⚠️ 局限性**

局限性包括仅适用于连通超图，迭代收敛受张量稀疏性影响，计算成本对大规模超图较高，且未考虑多层网络情况。

---

## 24. FACE: A Face-based Autoregressive Representation for High-Fidelity and Efficient Mesh Generation

**arXiv ID:** 2603.01515 | [PDF](https://arxiv.org/pdf/2603.01515v1)

**作者:** Hanxiao Wang `[一作]` (Chinese Academy of Sciences Institute of Automation), Dong-Ming Yan `[通讯]` (Chinese Academy of Sciences Institute of Automation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 FACE，基于面级自回归自动编码器的高保真网格生成与重建框架。

**💡 创新点**

创新点是采用“one-face-one-token”策略，将三角面视为单个 token，压缩比达 0.11，显著降低 Transformer 计算成本。

**🔧 技术方法**

使用自回归自动编码器、VecSet 编码器、CausalMLP 解码头、面序列化排序（ZYX）、以及图像到网格的潜在扩散模型等技术。

**📊 数据集**

训练和评估使用 Objaverse、Toys4K、Famous 等数据集，测试集为未见模型。

**📈 对比分析**

与 MeshAnything、TreeMeshGPT 等基线在 Hausdorff 与 Chamfer 指标上比较，FACE 在三组数据均取得最低误差（例如 Hausdorff 0.067 对比 0.091），并在图像到网格任务中优于 EdgeRunner。

**⚠️ 局限性**

局限性包括离散量化上限导致细节受限，以及依赖点云采样，极细薄结构如自行车辐条可能无法完整重建。

---

## 25. The Art of Generative Narrativity

**arXiv ID:** 2603.01086 | [PDF](https://arxiv.org/pdf/2603.01086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 26. Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning

**arXiv ID:** 2603.01759 | [PDF](https://arxiv.org/pdf/2603.01759v1)

**作者:** Zichen Tian `[一作]` (Singapore Management University), Qianru Sun `[通讯]` (Singapore Management University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出MetaPEFT，一种通过统一可微调的调制器与双层优化框架自动优化参数高效微调（PEFT）中的插入位置、层深和缩放因子，从而在遥感与自然视觉的长尾分布下显著提升模型性能，尤其是尾类精度。

**💡 创新点**

创新点包括①将离散的插入位置和层深以及连续的缩放因子统一为可微调的标量调制器；②利用双层优化实现对调制器的梯度学习；③在外循环中随机抽样训练子集以防止尾类过拟合，从而在不手工搜索超参数的情况下达到或超过现有最佳PEFT方案。

**🔧 技术方法**

主要技术：参数高效微调（LoRA、AdaptFormer等）、可微调调制器、双层（bi‑level）优化、softplus约束、随机子集抽样、Logit Adjustment 损失。

**📊 数据集**

实验数据集涵盖遥感光学与雷达图像（DOTA、FUSRS v2、SatMAE→SAR）以及自然视觉的长尾基准（CIFAR100‑IR100、Places‑LT、iNaturalist‑2018）。

**📈 对比分析**

与基线PEFT方法（LoRA、Adapters、AdaptFormer、VPT）以及传统长尾学习策略进行对比，MetaPEFT在平均准确率和尾类准确率上均取得提升（如在LoRA上平均提升约1.1%，在SatMAE→SAR场景尾类提升约1.2%），在交叉光谱适配任务中达到了当前最优表现。

**⚠️ 局限性**

限制主要包括：仍需在外循环中手工设定调制器学习率和抽样比例；对非加法PEFT方法的通用性尚未验证；虽然参数量增幅极小，但在极大模型（如更大ViT或更深网络）上可能需要进一步评估。

---

## 27. Pharmacology Knowledge Graphs: Do We Need Chemical Structure for Drug Repurposing?

**arXiv ID:** 2603.01537 | [PDF](https://arxiv.org/pdf/2603.01537v1)

**作者:** Youssef Abo-Dahab `[一作]` (University of California), Ismael Caleb Arechiga Duran `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了基于ChEMBL 36的药理知识图谱，利用时间拆分和生物验证的硬负样本，对药物-蛋白和药物-适应症关系进行链接预测。

**💡 创新点**

创新点在于严格的时间验证与硬负样本采样，发现数据量和图拓扑对预测性能影响大于模型复杂度和化学结构编码，提出仅使用拓扑嵌入即可近似最优的高效方案。

**🔧 技术方法**

采用知识图嵌入模型（TransE/TransR/RotatE/ComplEx/DistMult）、异构图神经网络（Standard GNN 与 Blackwell GNN）以及特征消融、数据与参数缩放实验，评估指标为 PR-AUC 与 Hits@k。

**📊 数据集**

使用ChEMBL 36数据库中的3,127种药物、1,156种蛋白、1,065种适应症共5,348实体、20,015条边，并将实验失效和临床试验失败作为生物验证的硬负样本。

**📈 对比分析**

在2022年前训练、2023–2025测试的严格时间拆分下，Standard GNN在药物-蛋白预测上 PR‑AUC 达 0.5631；移除药物图结构后提升至 0.5785 并将显存从 5.30 GB 降至 353 MB；Blackwell GNN 在 36.8 GB 显存下取得最高 0.5910 PR‑AUC，较小模型可达 95% 性能。

**⚠️ 局限性**

局限性包括仅做转导式推理，未评估新药/疾病的归纳泛化；适应症缺乏时间戳导致随机拆分可能引入数据泄漏；仅使用单一数据源，指标可能偏高；硬负样本选择虽严格但仍可能影响评估公平性。

---

## 28. State-Action Inpainting Diffuser for Continuous Control with Delay

**arXiv ID:** 2603.01553 | [PDF](https://arxiv.org/pdf/2603.01553v1)

**作者:** Dongqi Han `[一作]` (Microsoft Research Asia), Dongsheng Li `[通讯]` (Microsoft Research Asia)

**通讯引用:** 7947 | [OpenAlex ID](https://openalex.org/A5100440903)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种将延迟控制问题建模为状态-动作序列的填充（inpainting）任务，使用扩散模型直接生成当前状态和未来动作，形成 SAID（State-Action Inpainting Diffuser）框架。

**💡 创新点**

创新点在于把信号延迟问题视为生成式序列填充，而非传统的递归预测或状态堆叠；通过联合生成状态与动作来抑制累积误差，同时兼具模型无关性，既可在线又可离线应用。

**🔧 技术方法**

核心技术包括扩散模型（Diffusion Implicit Models + Diffusion Transformer）、蒙特卡洛采样与评价（MCSS）选择器、以及对延迟观测的条件化编码。

**📊 数据集**

使用的主要数据集是MuJoCo连续控制任务（Halfcheetah、Ant、Walker2d、Hopper）以及D4RL的离线机器人运动数据（halfcheetah、hopper、walker2d 的 medium、medium-expert、medium-replay）。

**📈 对比分析**

在延迟为 0、4、8、16 步的在线和离线基准上，与多种状态扩展、延迟补偿、变分延迟策略、以及 Diffusion Q‑Learning 等方法相比，SAID 在所有延迟场景下均表现出更高的平均回报和更稳健的性能，尤其在高延迟（8~16 步）时明显优于基线。

**⚠️ 局限性**

主要限制是推理时需进行多步扩散去噪，导致计算延迟较高；虽可通过采样步数减小但仍不适用于极高频控制；此外，方法在面对极端随机或时间变异延迟时的鲁棒性尚待进一步验证。

---

## 29. MIST-RL: Mutation-based Incremental Suite Testing via Reinforcement Learning

**arXiv ID:** 2603.01409 | [PDF](https://arxiv.org/pdf/2603.01409v1)

**作者:** Sicheng Zhu `[一作]` (Fudan University), Xin Li `[通讯]` (South China University of Technology)

**通讯引用:** 83108 | [OpenAlex ID](https://openalex.org/A5100387487)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计并实现了一种基于强化学习的增量式单元测试生成框架MIST-RL，用来验证大语言模型生成的代码。

**💡 创新点**

创新点在于把测试生成转化为马尔科夫决策过程，提出基于突变的增量奖励和动态冗余惩罚，并采用Group Relative Policy Optimization进行无价值网络的优化，突破传统按数量扩展的局限。

**🔧 技术方法**

使用技术包括强化学习（GRPO）、突变测试、Python AST突变引擎、增量奖励机制、动态惩罚、LLM（Llama-3‑8B等）作为生成模型。

**📊 数据集**

实验数据集为HumanEval+、MBPP+和DS‑1000，均在原始数据集基础上加入了大量突变测试。

**📈 对比分析**

通过与CodeRM‑8B、Qwen3‑14B以及基础Llama模型对比，MIST‑RL在HumanEval+上Mutant Kill Rate提升约28.5%，测试用例长度减少19.3%，并在代码重排序任务中Pass@1提高约3%（对10/20候选样本），表现优于所有基线。

**⚠️ 局限性**

局限性包括依赖突变工具的覆盖范围、训练成本高、对特定语言/工具链的依赖、在更大规模或多文件项目上的可迁移性尚未验证。

---

## 30. Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning for Recommendation at Industrial Scale

**arXiv ID:** 2603.00632 | [PDF](https://arxiv.org/pdf/2603.00632v1)

**作者:** Zheng Hu `[一作]` (University of Electronic Science and Technology of China), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个端到端的 Semantic ID 学习框架 QuaSID，旨在通过资格感知的冲突抑制和对齐来提升多模态推荐系统中的语义 ID 质量。

**💡 创新点**

创新点包括 Hamming 指导的边缘惩罚（HaMR）用于基于碰撞严重度的距离约束，冲突感知有效对齐掩蔽（CVPM）过滤无害重叠，以及结合对比学习的双塔正则化。

**🔧 技术方法**

技术方法涵盖残差量化变分自编码器（RQ‑VAE）作为编码器，量化后生成 SID，加入 HaMR、CVPM 和 InfoNCE 对比损失，利用 Straight‑Through Estimator 进行端到端优化。

**📊 数据集**

使用了 Amazon 2018 Beauty 与 Toys 两个公开数据集以及来自 Kuaishou 电商的工业级数据进行离线评估和在线 A/B 测试。

**📈 对比分析**

与多种基准（RQ‑VAE、GRVQ、SimRQ 等）对比，QuaSID 在 HR@K、NDCG@K 及 SID 熵指标上平均提升 5.9% 左右；在工业 A/B 实验中 GMV‑S2 上提升 2.38%，冷启动下完成订单提升 6.42%。

**⚠️ 局限性**

局限性主要在于需要手工设定碰撞阈值与掩蔽规则，且对比学习的超参敏感；此外，HaMR 虽提升离散空间质量，但单独使用时无法完全匹配 QuaSID 的整体性能。

---

## 31. Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from an Industry Deployment

**arXiv ID:** 2603.02153 | [PDF](https://arxiv.org/pdf/2603.02153v1)

**作者:** Luigi Medrano `[一作]` (Dell Technologies), Mukul Chhabra `[通讯]` (Dell Technologies)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估检索融合技术在企业级生产环境下的RAG系统中的端到端效果

**💡 创新点**

首次在受限的检索深度、重排序预算和延迟约束下，对检索融合的实际收益与成本进行系统性评估

**🔧 技术方法**

采用双查询融合（原始查询+LLM生成改写）、BM25+稠密检索、交叉编码器重排序、RRF融合、LangChain框架

**📊 数据集**

合成的115条企业支持查询，配合对应的知识库文章作为ground truth

**📈 对比分析**

使用KB级Top‑1/Top‑3/Hit@10指标对比融合与单查询基线；结果显示融合虽提升检索召回，但在重排序与截断后未提升，甚至在部分配置下下降，且延迟显著增加

**⚠️ 局限性**

限制：实验仅覆盖固定检索深度与重排序预算，未探究更大规模或不同检索模型；融合带来冗余与冲突，且对大多数查询无显著收益，实际应用需权衡成本与收益

---

## 32. On Best-Possible One-Time Programs

**arXiv ID:** 2603.00544 | [PDF](https://arxiv.org/pdf/2603.00544v1)

**作者:** Aparna Gupte `[一作]` (Massachusetts Institute of Technology), Mark Zhandry `[通讯]` (Stanford University)

**通讯引用:** 3807 | [OpenAlex ID](https://openalex.org/A5024874638)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了在无硬件假设下的一次性程序（OTP）的可行性与安全性，证明最佳一次性编译器不可实现，并提出可测试一次性程序与SEQ安全性等新概念。

**💡 创新点**

首次给出最佳一次性程序不可实现性的严谨证明，定义了可测试一次性程序并证明SEQ安全性是实现最佳测试一次性安全性的必要与充分条件，以及将可测试一次性程序与状态化量子iO联系起来。

**🔧 技术方法**

采用可失效率加密、混合实现对比、单效查询（SEQ）安全框架、状态化量子等价obfuscation以及量子身份认证方案等技术。

**📊 数据集**

无数据集，完全为理论构造与证明。

**📈 对比分析**

通过相对论性证明和oracle模型实验验证，展示理论上的最佳安全性，但未提供实际性能指标。

**⚠️ 局限性**

在普通模型下无法构造最佳一次性程序，需依赖尚未实现的状态化量子iO，且对非随机或损失模式的功能可能不适用。

---

## 33. Piecing Together Cross-Document Coreference Resolution Datasets: Systematic Dataset Analysis and Unification

**arXiv ID:** 2603.00621 | [PDF](https://arxiv.org/pdf/2603.00621v1)

**作者:** Anastasia Zhukova `[一作]`, Bela Gipp `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了uCDCR统一的跨文档共指解析基准，整合并标准化12个公开数据集；

**💡 创新点**

创新点在于构建统一格式与属性补全的解析流水线，系统评估词汇多样性与歧义度，并提供基准基线性能；

**🔧 技术方法**

使用了重token化、spaCy依存解析、特征提取，以及词汇多样性/歧义度指标（UL-a、UL-o、PD、MTLD、AL）和同词干基线；

**📊 数据集**

使用了12个公开CDCR数据集，包括ECB+、ECB+METAm、WEC-Eng、FCC-T、GVC、HyperCorefexp、NewsWCL50r、MEANTIMEeng、NIdenten-cd、NP4Ecd、CERECexp、CD2CR；

**📈 对比分析**

通过在子主题和主题层面计算MUC、B3、CEAF_e、CoNLL F1指标的同词干基线进行比较，发现WEC-Eng最易解，HyperCorefexp和NewsWCL50r最难，整体平均CoNLL F1约56.9；

**⚠️ 局限性**

局限在于仅覆盖英文数据，缺少多语言和社交媒体等多样化语料；注释差异与潜在偏倚仍存；基线无法捕捉隐喻等复杂现象。

---

## 34. On Channel Model to Bridge the Gap between MIMO Design and Performance Requirements in 3GPP

**arXiv ID:** 2603.01843 | [PDF](https://arxiv.org/pdf/2603.01843v1)

**作者:** Lynda Berrah `[一作]` (Orange Research), Matthew Baker `[通讯]` (Nokia UK Limited)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估并推广 rCDL 渠道模型，用于 MIMO 性能评估和 CSI 报告要求。

**💡 创新点**

将 RAN1 CDL 简化为 rCDL，保留空间性且降低复杂度，并证明其优于 TDL。

**🔧 技术方法**

采用 CDL 的几何化随机模型、角度缩放、随机性降低、AAV 虚拟化及多种 MIMO 代码书(低分辨率 Type‑I 与高分辨率 eType‑II) 的性能模拟。

**📊 数据集**

现场 4×4 MIMO 频段 2162.2 MHz 的实际测量数据以及 TR 38.901/38.753 规格表参数。

**📈 对比分析**

通过 Bartlett 空间谱、后均衡 SINR 分布、以及 CSI‑反馈下的谱效率与理想束成分上限/随机 PMI 下限比较，结果显示 rCDL 能明显区分两类代码书，获得 2–5 dB 的增益；而 TDL 无法区分。

**⚠️ 局限性**

仅验证单用户、特定场景和角度扩展，缺乏多用户空间一致性验证，且 rCDL 对角度扩散值的稳定性仍需进一步测量。

---

## 35. Completing the Complexity Classification of 2-Solo Chess: Knights and Kings are Hard

**arXiv ID:** 2603.01675 | [PDF](https://arxiv.org/pdf/2603.01675v1)

**作者:** Kolja Kühn `[一作]` (Karlsruhe Institute of Technology), Wendy Yi `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 2365 | [OpenAlex ID](https://openalex.org/A5044897487)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文对单玩家变体 2‑Chess（k=2）中的仅使用国王或骑士时的可解性进行复杂度分析，并证明这两种情况均为 NP‑完全问题；

**💡 创新点**

创新点在于完成了 2‑Chess 所有单一棋子类型的复杂度分类，首次为国王和骑士设计并证明了对应的逻辑门（变量、连线、OR、AND、交叉等）构造，填补了现有研究的空白；

**🔧 技术方法**

主要技术手段是从 3‑SAT 变体进行多项式时间归约，构造专门的棋子布局（门控与交叉）并利用捕获图、虚拟预算等图论工具证明其功能与最优性；

**📊 数据集**

研究使用的是理论构造的棋盘实例，没有采用真实棋局数据；

**📈 对比分析**

通过理论归约与构造证明来进行比较，结果表明仅在使用甲兵时可在多项式时间内判定可解性，而在国王或骑士时属于 NP‑难，未做实验性能评估；

**⚠️ 局限性**

局限性在于仅考虑 k=2 且单一棋子类型的情况，未讨论 k>2 或混合棋子组合的复杂度，也未验证构造实例在实际棋局中的可行性。

---

## 36. OpenRad: a Curated Repository of Open-access AI models for Radiology

**arXiv ID:** 2603.02062 | [PDF](https://arxiv.org/pdf/2603.02062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 37. The Aftermath of DrawEduMath: Vision Language Models Underperform with Struggling Students and Misdiagnose Errors

**arXiv ID:** 2603.00925 | [PDF](https://arxiv.org/pdf/2603.00925v1)

**作者:** Li Lucy `[一作]` (University of Washington), Kyle Lo `[通讯]` (Allen Institute for AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 2025 年发布的 11 种视觉语言模型在真实学生手绘数学答卷上的表现进行评估，并深入分析模型在错误识别与正确性评估方面的差距。

**💡 创新点**

揭示了 VLM 在错误答卷和正确性/错误相关问答上的系统性弱点，证明模型往往默认错误为零并难以处理开放式错误检测，同时提供了图像噪声、问题难度、文本支持等多因素的细致因子分析。

**🔧 技术方法**

采用多模态 VLM（OpenAI、Anthropic、Google、Meta 等）、LLM 评判器、回归分析、图像重绘实验以及教师标注与自动生成的 QA 对话，综合评估模型性能。

**📊 数据集**

使用 DrawEduMath 基准数据集，该数据集包含 2,030 张学生手绘答卷图片、教师自由描述和超过 44k 条自动生成/教师编写的 QA 对。

**📈 对比分析**

通过 LLM 判别器对模型生成答案与黄金答案进行 1–4 评分并二值化，比较不同模型在内容描述、正确性/错误问答（开放式与二元式）的准确率；结果显示模型在错误答卷和正确性评估上明显低于正常答卷，部分二元评估甚至仅略高于随机水平。

**⚠️ 局限性**

局限性包括：仅使用单一英语基准，数据主要来自 Title I 学校；图像重绘实验样本有限；依赖教师标注可能存在主观偏差；未覆盖多语言或更广泛的教育环境。

---

## 38. Anatomy of the Modality Gap: Dissecting the Internal States of End-to-End Speech LLMs

**arXiv ID:** 2603.01502 | [PDF](https://arxiv.org/pdf/2603.01502v1)

**作者:** Ming-Hao Hsu `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究深入探讨了大声学语言模型（LSLM）中语音输入与文本输入在推理任务中的性能差距，结合内部表示层级动态分析，阐明了模态差距的根源与机制。

**💡 创新点**

创新点在于提出模态差距可分为三阶段：①结构转换（语音特征需经过非线性投影才能进入文本相似空间）；②语义扩散（语音冗余导致语义信息在多帧中扩散，形成宽泛的跨层对齐带）；③决策不稳定（即使信息已聚合，语音层难以在最终层形成锐利的决策信号）。

**🔧 技术方法**

技术手段包括：跨层Centered Kernel Alignment（CKA）配合动态时间规整（DTW）实现语音-文本对齐；标准化L2距离与层级均值范数评估几何相似度；注意力熵与决策边际（logit margin）指标分析信息聚焦与决策不确定性；线性与MLP探针验证信息是否在隐藏层中保留。

**📊 数据集**

使用的数据集为 SpeechMMLU（MMLU 语音版）和 VoiceBench‑BBH（BIG‑Bench Hard 语音版），覆盖 STEM、人文与推理等多领域任务；同时对四个公开权重的端到端 LSLM 进行实验（Qwen2.5‑Omni‑7B、MiniCPM‑o‑2.6、Qwen2‑Audio‑7B‑Instruct、LLaMA‑Omni）。

**📈 对比分析**

方法比较：将 S2T（语音输入）与 T2T（文本输入）在相同提示下进行内部诊断，发现 S2T 的整体准确率下降 3%–13%，且在中间层虽对齐但在最后层表现出决策不稳定；单纯的统计校准或输入层投影会进一步恶化性能，证明仅靠几何对齐不足以弥补模态差距。

**⚠️ 局限性**

局限性：实验集中在现有公开模型，未给出可直接落地的修复方案；对更大规模、多语种或更复杂任务的泛化性尚未验证；且仅从内部诊断角度揭示机制，未对模型架构进行改进或设计新的压缩语音 token 的方法。

---

## 39. Multi-Domain Riemannian Graph Gluing for Building Graph Foundation Models

**arXiv ID:** 2603.00618 | [PDF](https://arxiv.org/pdf/2603.00618v1)

**作者:** Li Sun `[一作]` (Beijing University of Posts and Telecommunications), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 134323 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在多域图预训练中提出了基于黎曼几何的统一框架，将各域图数据通过神经流形拼接（Neural Manifold Gluing）融入同一光滑黎曼流形，实现预训练与迁移的系统化；

**💡 创新点**

核心创新包括：①神经流形拼接理论，利用局部正交框架、边缘平移、三角形霍洛诺米一致性和Ricci曲率平滑构造全局光滑流形；②EMA原型与Riemannian Mixture‑of‑Experts的结合，实现批量预训练与域语义区分；③几何转移指标GTM与几何缩放定律，为跨域迁移难度提供可解释度量；

**🔧 技术方法**

技术手段涵盖：Cartan移动框架、稀疏方向扰动与QR正交化、边缘正交映射、霍洛诺米损失、Log‑Determinant光滑损失、EMA原型更新、对比学习、可学习提示、Riemannian MoE、曲率与霍洛诺米损失的联合训练；

**📊 数据集**

实验使用六个代表性图数据集：学术引用网络、商品共购网络、社交网络、知识图谱、生物信息学图、化学分子图；

**📈 对比分析**

在leave‑one‑out交叉域、少样本（1‑shot、5‑shot）节点/边/图分类任务上，GraphGlue相较于监督GNN、对比学习、以及现有图基础模型（PRODIGY、GFT、RAGraph、SAMGPT、GCOPE、MDGFM）均取得显著提升（大约4–8% ACC/AUC），并在Ablation中验证霍洛诺米与曲率损失的重要性；

**⚠️ 局限性**

局限性：①对极度异构或动态图的适用性尚未验证；②预训练需要较多域数据，计算与存储成本相对较高；③理论假设（如三角形霍洛诺米平凡）在真实数据中可能不完全成立，导致拼接误差；④在样本极少的1‑shot情形下仍受限于模型表达能力。

---

## 40. SCOUT: Fast Spectral CT Imaging in Ultra LOw-data Regimes via PseUdo-label GeneraTion

**arXiv ID:** 2603.00687 | [PDF](https://arxiv.org/pdf/2603.00687v1)

**作者:** Guoquan Wei `[一作]` (Nanchang University), Qiegen Liu `[通讯]` (Nanchang University)

**通讯引用:** 3479 | [OpenAlex ID](https://openalex.org/A5057647276)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种零样本伪标签生成的自监督方法SCOUT，在极低原始投影数据条件下无需外部数据即可快速恢复高质量的PCCT图像；

**💡 创新点**

创新点在于结合空间非局部相似性与投影域共轭对称性，生成低秩伪标签，利用投影域的统计特性实现极快、无外部训练的去噪与伪影消除；

**🔧 技术方法**

核心技术包括三维投影体块的相似块搜索、共轭替换构造伪标签、基于Noise2Noise理论的三维卷积网络训练；

**📊 数据集**

使用了多种数据集：PCCT小鼠实验数据、公开的核桃双能CT数据、Mayo2016、Mayo2020、LIDC‑IDRI、CTspine1K等，涵盖多通道、多能量和低剂量场景；

**📈 对比分析**

与BM3D、B2U、Neighbor2Neighbor、Noise2Noise、PromptSID、Noise2sim、Noise2detail、ZS‑N2N等八种自监督方法对比，SCOUT在PSNR/SSIM上提升3 dB/7%，处理时间仅为3–10分钟，速度是传统方法的数百倍，且在低剂量、伪影、细节恢复等多项指标上表现最佳；

**⚠️ 局限性**

局限包括对相邻像素相似性假设的依赖，低采样率时可能失效；伪标签生成与网络简单化可能限制极端噪声场景下的细节重建；在高维大体量数据时仍需足够存储空间；

---

## 41. AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution

**arXiv ID:** 2603.00589 | [PDF](https://arxiv.org/pdf/2603.00589v1)

**作者:** Cencen Liu `[一作]` (University of Electronic Science and Technology of China), Guoming Lu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5000444026)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 AlignVAR 的视觉自回归框架，用于高质量图像超分，重点解决局部注意力偏差和层级误差累积导致的全局不一致问题。

**💡 创新点**

创新点在于：① Spatial Consistency Autoregression (SCA)，利用结构引导的自适应掩膜重新加权注意力以强化跨域依赖；② Hierarchical Consistency Constraint (HCC)，通过在潜在空间对完整尺度的监督，校正跨尺度误差并抑制累积失真。

**🔧 技术方法**

技术手段包括：VQ‑VAE 编码/量化、下一尺度视觉自回归 Transformer、Laplacian 结构引导、MLP 生成自适应掩膜、交叉熵+MSE 组合损失、teacher‑forcing 训练策略。

**📊 数据集**

训练使用 LSDIR 与 FFHQ（前 10K 张人脸）数据集，评估则在 DIV2K‑Val、RealSR 与 DRealSR 三个基准上进行。

**📈 对比分析**

与 GAN（BSRGAN、Real‑ESRGAN）、扩散（StableSR、DiffBIR 等）以及 VARSR 进行对比；在合成与真实数据集上，AlignVAR 在 LPIPS、DISTS、FID、MANIQA、CLIPIQA、MUSIQ 等感知/无参考指标上显著优于竞争方法，且推理时间仅 0.43 s、参数 1056 M，速度比扩散模型快 10×，比 VARSR 快 5×。

**⚠️ 局限性**

局限性：仍无法恢复失去的细节（像素级 PSNR 与 SSIM 并未突破现有最高水平），对低分辨率输入的细节恢复受限；模型依赖 VQ‑VAE 的量化表达，可能在极端降噪或非标准 degradations 上表现不佳。

---

## 42. Generalizing Logic-based Explanations for Machine Learning Classifiers via Optimization

**arXiv ID:** 2603.01870 | [PDF](https://arxiv.org/pdf/2603.01870v1)

**作者:** Francisco Mateus Rocha Filho `[一作]` (Instituto Federal de Educação, Ciência e Tecnologia do Ceará), Thiago Alves Rocha `[通讯]` (Instituto Federal de Educação, Ciência e Tecnologia do Ceará)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了两种逻辑基础的解释扩展方法（Onestep和Twostep），用于生成具有正确性和最小性的解释并提升其泛化范围；

**💡 创新点**

创新点在于将范围扩展问题转化为一次性优化（Onestep）和两步优化（Twostep）策略，从而避免迭代增量导致的计算开销和覆盖率下降；

**🔧 技术方法**

利用第一阶逻辑表述模型预测约束并构造混合整数线性规划（MILP）求解器，进行解释生成与范围扩展；

**📊 数据集**

在12个UCI及Kaggle数据集上实验，包括IRIS、PIMA、WINE等，使用线性SVC和单隐层ReLU MLP作为目标模型；

**📈 对比分析**

与先前的增量扩展方法相比，Twostep在大多数数据集上显著提高了覆盖率（最高提升约72.6%），计算时间略高（约55.6%），Onestep在速度上更快但覆盖率相对较低；

**⚠️ 局限性**

主要限制在高维数据中覆盖率下降，且方法仍依赖于最小解释，可能忽略更简洁但覆盖更广的解释方案。

---

## 43. D3LM: A Discrete DNA Diffusion Language Model for Bidirectional DNA Understanding and Generation

**arXiv ID:** 2603.01780 | [PDF](https://arxiv.org/pdf/2603.01780v1)

**作者:** Zhao Yang `[一作]` (Renmin University of China), Bing Su `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 D3LM，一种在离散 DNA 空间中使用掩码扩散的统一 DNA 基础模型，既能进行双向表征学习，也能实现 DNA 序列生成。

**💡 创新点**

创新点在于将变动掩码比例的离散扩散训练目标与双向 Transformer 结合，突破了传统 BERT 仅限理解和自回归仅限生成的局限，并在生成时实现了全序列的迭代解码。

**🔧 技术方法**

主要技术包括 NT‑v2 Transformer 架构、6‑mer 非重叠分词、基于 1/t 加权的掩码扩散损失、温度缩放与随机解码策略。

**📊 数据集**

使用了跨物种的 EPD‑GenDNA 语料库（160k 条 2048bp 长度的 DNA 序列，含 15 种生物，约 80k 条哺乳动物序列）进行预训练和评估。

**📈 对比分析**

在 DNA 生成任务上与自回归模型、潜在扩散模型及随机采样等基线比较，D3LM‑R 在 SFID 上达 10.92（接近真实 DNA 的 7.85），显著优于自回归 29.16 及潜在扩散 62.74；在下游表征任务上，其性能与 NT‑v2 相当或更好，尤其在 splice site 预测上提升至 0.947。

**⚠️ 局限性**

局限性包括对大规模预训练语料的依赖（从零初始化的 D3LM‑R 表现逊色），以及在采样策略上仍需进一步探索以兼顾多样性与生物学真实性。

---

## 44. Navigating Time's Possibilities: Plausible Counterfactual Explanations for Multivariate Time-Series Forecast through Genetic Algorithms

**arXiv ID:** 2603.00855 | [PDF](https://arxiv.org/pdf/2603.00855v1)

**作者:** Gianlucca Zuin `[一作]` (Universidade Federal de Minas Gerais), Adriano Veloso `[通讯]` (Kunumi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合Granger因果、分位数回归和遗传算法的多变量时间序列反事实预测框架，能够在满足目标约束的前提下生成可解释的未来情景。

**💡 创新点**

创新点在于将Granger因果检验与分位数回归相结合限制可行性空间，再使用遗传算法搜索满足目标约束的可解释路径；首次针对回归型多变量时间序列实现可解释反事实预测。

**🔧 技术方法**

技术栈包括Granger因果检验、Auto-Regressive模型、分位数回归（pinball loss）、LightGBM回归、遗传算法（种群、变异、交叉、随机移民）等。

**📊 数据集**

使用M. Dias Branco公司ACL真空系统的33个时序变量数据，采样频率3秒，覆盖数月时间，记录真空压力及相关传感器值。

**📈 对比分析**

通过与ARIMA、LSTM、SVR、Elastic Net等模型对比，LightGBM在MAE、MSE、R²等指标上表现最佳；遗传算法在90%实验中成功找到满足约束的方案，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：模型训练和Granger检验计算量大，极端值外推能力不足，可能陷入局部最优；对更大规模数据集的可扩展性尚未充分验证。

---

## 45. ArtLLM: Generating Articulated Assets via 3D LLM

**arXiv ID:** 2603.01142 | [PDF](https://arxiv.org/pdf/2603.01142v1)

**作者:** Penghao Wang `[一作]` (ShanghaiTech University), Jiayuan Gu `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个基于3D LLM的框架，能够从单张图片或文本快速生成具有物理可行性和高质量几何的可动三维资产。

**💡 创新点**

将可动物体的部件布局和关节信息表述为离散化的语言序列，利用自回归3D LLM联合预测部件数量、位置与关节参数，并结合多任务多阶段训练与物理约束的关节极限校正，打破了传统优化慢、检索重复的局限。

**🔧 技术方法**

采用点云编码器 Point Transformer v3 与 Qwen3 0.6B 语言模型进行跨模态编码；对连续几何与关节参数进行 128/48/64 级离散化；使用 XPart 等部件生成模型生成细节几何；通过物理碰撞检测对关节极限进行后处理；并使用随机旋转缩放数据增强。

**📊 数据集**

构建了包含 20,673 个可动物体的统一数据集，来源于 PartNet-Mobility、PhysX3D 及 12k 通过 Infinite-Mobility 生成的程序化模型，涵盖 43 个类别。

**📈 对比分析**

在 PartNet-Mobility 的 7 个类别上与 URDFormer、SINGAPO、Articulate-Anything 等基线进行对比，评估指标包括部件 mIoU、关节类型与轴向准确率、极限 IoU 以及 kinematic graph 准确率；实验表明本方法在部件布局、关节预测与层级建模上均显著优于基线，并在推理速度上也更快。

**⚠️ 局限性**

由于训练数据类别仍相对有限，模型对车辆、机器人等复杂类别的泛化能力不足；同时未联合学习物理属性（质量、惯量等），因此对物理动态的完整预测仍有限。

---

## 46. IDER: IDempotent Experience Replay for Reliable Continual Learning

**arXiv ID:** 2603.00624 | [PDF](https://arxiv.org/pdf/2603.00624v1)

**作者:** Zhanwang Liu `[一作]` (Shanghai Jiao Tong University), Weiran Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1362 | [OpenAlex ID](https://openalex.org/A5057918643)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于幺等性（idempotence）的经验回放方法IDER，用于解决连续学习中的灾难性遗忘和过度自信问题。

**💡 创新点**

创新点在于：①利用幺等性原理将模型训练为自一致性；②设计了标准幺等损失与幺等蒸馏损失，结合旧模型与新模型的自洽；③只需额外一次前向传播，计算成本低，且易于与现有回放方法无缝集成。

**🔧 技术方法**

技术上包括：改造ResNet骨干以接收第二个输入；构造交叉熵双向幺等损失（ℒ_ice）；使用旧任务模型进行幺等蒸馏损失（ℒ_ide）；总体损失为ℒ_IDER=ℒ_ice+αℒ_ide+βℒ_rep-ice。

**📊 数据集**

数据集：CIFAR‑10、CIFAR‑100、Tiny‑ImageNet 的分裂式类别增量学习（CIL）以及通用类别增量学习（GCIL）场景。

**📈 对比分析**

与ER、iCaRL、DER、XDER、CLS‑ER、BFP、SARL 等基线及 NPCL 等不确定性方法对比，IDER 在 FAA、ECE 和遗忘度上均表现优异（例如在CIFAR‑10/100上显著提升 3–5% 准确率、降低 30–50% ECE、减少 10–15% 遗忘），且训练时间仅略增。

**⚠️ 局限性**

局限性：①仅在小规模视觉基准上验证；②对大规模数据或不同任务域（如自然语言、强化学习）的适用性尚未探测；③幺等性强制可能在某些模型结构或任务类型中导致过度约束，需进一步理论分析。

---

## 47. Beyond Reward: A Bounded Measure of Agent Environment Coupling

**arXiv ID:** 2603.01283 | [PDF](https://arxiv.org/pdf/2603.01283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 48. An Investigation of the Relation Between Immersion and Learning Across Three Domains

**arXiv ID:** 2603.01644 | [PDF](https://arxiv.org/pdf/2603.01644v1)

**作者:** Paolo Boffi `[一作]` (Polytechnic University of Milan), Pier Luca Lanzi `[通讯]` (Polytechnic University of Milan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过CAMIL框架在文化遗产、环境意识和高中物理三大领域中，构建并实验评估三款沉浸式VR应用，探讨沉浸度与学习效果的关系。

**💡 创新点**

创新点在于跨域统一实验设计与评价体系，利用CAMIL模型阐释沉浸对学习心理路径的影响，并提出基于沉浸度的教学设计指南。

**🔧 技术方法**

采用HMD沉浸式VR与桌面VR/幻灯片对比，结合3D图形、音频导航、头部追踪、手部交互与生理测量（EDA）。

**📊 数据集**

使用自制的多项选择测试、TAM、Presence问卷、UEQ等问卷，收集实验参与者（18–35岁）在三种媒介下的学习与体验数据。

**📈 对比分析**

通过实验室与课堂生态部署的混合方法，使用非参数检验、线性混合模型等统计手段比较三种媒介，结果显示沉浸式VR显著提升存在感、体验和技术接受度，但即时知识测试相当；在物理实验中，沉浸式VR在两周保留期表现更佳。

**⚠️ 局限性**

局限包括未评估概念/程序性知识、仅短期保留、主要依赖自评量表、实验室与远程条件不完全匹配、受新颖性影响以及样本聚类未完全控制。

---

## 49. PlantWhisperer: Designing Conversational AI to Support Plant Care

**arXiv ID:** 2603.00598 | [PDF](https://arxiv.org/pdf/2603.00598v1)

**作者:** Daniel Mejer Christensen `[一作]` (Aalborg University), Joel Wester `[通讯]` (University of Copenhagen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发并评估了一款名为PlantWhisperer的移动应用，该应用通过LLM驱动的聊天机器人让用户与自家植物进行自然语言对话，支持植物养护并探讨其对用户心理健康的正面影响。

**💡 创新点**

创新点在于：① 将植物拟人化为具备个性化情感表达的聊天机器人；② 结合PERMA模型对幸福感进行结构化访谈评估；③ 通过对话式交互而非传统搜索方式，为植物养护提供情感支持与学习帮助。

**🔧 技术方法**

使用技术：大型语言模型（LLM）配合提示工程（prompt engineering）、移动端UI/UX设计、基于文本的聊天接口；评估方法采用半结构化访谈与主题分析；对比方法以Google搜索为基准。

**📊 数据集**

数据集：无公开训练数据集，使用的是10名参与者在实验中产生的交互日志、访谈转录和自评量表；额外使用4名参与者的pilot测试结果进行迭代。

**📈 对比分析**

与Google搜索对比：参与者普遍认为聊天机器人在答案具体性、易懂性和互动性方面优于搜索；虽然缺乏量化指标，但质性反馈显示更高的情感连接与学习效果。

**⚠️ 局限性**

局限性：样本规模小（10人），实验时间短且受控，未进行长期实地使用研究；对不同用户对话风格（情感化 vs 直接）的偏好未能完全满足；缺乏对多植物、多用户聊天等更复杂场景的验证。

---

## 50. Modular Memory is the Key to Continual Learning Agents

**arXiv ID:** 2603.01761 | [PDF](https://arxiv.org/pdf/2603.01761v1)

**作者:** Vaggelis Dorovatas `[一作]` (Toyota Motor Europe), Rahaf Aljundi `[通讯]` (Toyota Motor Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于模块化记忆的持续学习框架，将上下文学习（ICL）与权重学习（IWL）相结合，以实现大规模连续适应和知识累积。

**💡 创新点**

核心创新在于：①将核心模型与工作记忆、长期记忆三大模块分离；②利用ICL实现快速适应与经验检索；③通过低频IWL对长期记忆进行消化与模型参数更新；④为记忆设计了多尺度、多表征与主动管理的原则。

**🔧 技术方法**

所用技术包括：多模态大型预训练模型、注意力机制用于ICL、可插拔的记忆模块（如KV缓存、激活嵌入、分层图结构）、元认知控制策略以及基于回放与正则化的IWL方案。

**📊 数据集**

文中未给出具体实验数据集，主要以理论和架构设计为主。

**📈 对比分析**

由于缺乏实验验证，本文未与现有方法进行性能比较，故无法给出数值指标。

**⚠️ 局限性**

局限性包括：①缺乏系统级实证评估；②记忆管理与更新策略的具体实现尚不明确；③在资源受限场景下的可扩展性和计算开销待验证；④对模型更新频率与遗忘的平衡缺乏经验依据。

---

## 51. Improved MambdaBDA Framework for Robust Building Damage Assessment Across Disaster Domains

**arXiv ID:** 2603.01116 | [PDF](https://arxiv.org/pdf/2603.01116v1)

**作者:** Alp Eren Gençoğlu `[一作]` (Istanbul Technical University), Hazım Kemal Ekenel `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 3463 | [OpenAlex ID](https://openalex.org/A5009982931)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

改进MambaBDA建筑损毁评估模型，加入焦点损失、轻量化注意门和对齐模块以提升性能。

**💡 创新点**

采用模块化设计，在不增加显著计算成本的前提下通过焦点损失解决类别不平衡、注意门过滤背景、对齐模块补偿空间偏移。

**🔧 技术方法**

使用焦点损失、注意门（Attention Gate）、轻量对齐模块、视觉状态空间（VSS）骨干的ChangeMamba架构，以及多任务损失（CE+Focal+Lovász）。

**📊 数据集**

主要数据集为xBD、Pakistan Flooding、Turkey Earthquake、Hurricane Ida，训练和测试均使用这些高分辨率灾害图像对。

**📈 对比分析**

在in-domain测试中平均提升0.8–5%的F1分数；在跨数据集测试中提升高达27%，表明显著的泛化能力提升；与基线相比，FOCAL+AGB组合效果最佳。

**⚠️ 局限性**

注意门在损伤头的稳定性差，容易失效；对齐模块在跨域时迁移性有限；对极少样本类别仍有提升空间。

---

## 52. Extending Adaptive Cruise Control with Machine Learning Intrusion Detection Systems

**arXiv ID:** 2603.01173 | [PDF](https://arxiv.org/pdf/2603.01173v1)

**作者:** Lotfi Ben Othmane `[一作]`, Naga Prudhvi Mareedu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了自适应巡航控制（ACC）在速度伪造攻击下的脆弱性，并提出将机器学习入侵检测系统（IDS）集成到ACC中，实现检测后紧急制动以保障安全。

**💡 创新点**

创新点在于通过理论证明阈值和Kalman滤波失效条件，设计了ACC‑IDS架构并给出在有限检测延迟下可保证不碰撞的安全定理，证明了IDS能补偿滤波器在恶意注入下的不足。

**🔧 技术方法**

采用Kalman滤波估计车速、PID控制实现ACC、基于CAN总线的机器学习IDS（如SVM/神经网络）进行攻击检测，并在仿真中使用CARLA/Simulink进行验证。

**📊 数据集**

使用CARLA仿真生成的车辆速度、距离及CAN消息数据，构造了注入攻击样本，作为IDS训练和仿真评估的数据集。

**📈 对比分析**

通过比较ACC+KF、ACC+KF+攻击、ACC+KF+IDS三种方案的碰撞发生时间和距离误差，结果表明单一KF无法避免碰撞，而加入IDS后可显著延长碰撞时间并保持安全；性能随IDS准确率提升而显著改善。

**⚠️ 局限性**

局限性包括假设IDS具有完美检测精度与固定延迟、攻击模型仅限于持续速度注入、未考虑车载网络时延与误报/漏报对系统性能的影响，以及理论假设在实际系统中可能不完全成立。

---

## 53. PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts

**arXiv ID:** 2603.01650 | [PDF](https://arxiv.org/pdf/2603.01650v1)

**作者:** Xianqi Wang `[一作]` (Huazhong University of Science and Technology), Xin Yang `[通讯]` (Optics Valley Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PromptStereo，通过将 GRU 替换为基于单目深度模型解码器的 Prompt Recurrent Unit (PRU)，实现零样本立体匹配的迭代细化。

**💡 创新点**

创新点在于使用 PRU 继承单目深度先验，结合结构提示 (SP) 与运动提示 (MP) 与 AIF 的仿射不变融合，显著提升了迭代更新的表达与收敛速度。

**🔧 技术方法**

采用的技术包括深度先验迁移、提示式增益、基于 DPT 的多分辨率解码器、卷积残差更新以及多尺度成本体积构造。

**📊 数据集**

使用的主要数据集包括 Scene Flow、KITTI 2012/2015、Middlebury V3、ETH3D、DrivingStereo、Booster 以及多域混合训练集 FoundationStereo、CREStereo、Falling Things、Virtual KITTI 2。

**📈 对比分析**

与 RAFT‑Stereo、IGEV‑Stereo、MonSter、BridgeDepth、MGStereo 等基线对比，在基本和高级零样本基准上均取得领先，误差均显著下降（如 Midd‑2021 EPE 下降约 50%），并保持或提升推理速度。

**⚠️ 局限性**

局限性主要是对极端恶劣天气（如雨天、雾天）的适应性不足，以及对透明、镜面等极端几何场景的鲁棒性仍待提升。

---

## 54. SkeleGuide: Explicit Skeleton Reasoning for Context-Aware Human-in-Place Image Synthesis

**arXiv ID:** 2603.01579 | [PDF](https://arxiv.org/pdf/2603.01579v1)

**作者:** Chuqiao Wu `[一作]` (Alibaba Group), Yiyun Fei `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SkeleGuide框架，基于先“推理骨架再渲染”实现人像置入；

**💡 创新点**

首次将骨架推理与渲染联合训练，使用可编辑的内部骨架布局；

**🔧 技术方法**

采用流匹配（Flow Matching）与Diffusion Transformer（DiT），并利用LoRA实现条件注入；

**📊 数据集**

使用V-COCO公共数据集与自制家居场景数据，合计约14,279张图；

**📈 对比分析**

与多种通用与专用模型对比，SkeleGuide在FID、KID、背景一致性、人体结构质量等指标均超越对手，取得SOTA表现；

**⚠️ 局限性**

仍受限于推理阶段生成骨架的精度与对复杂多人人体场景的处理能力，某些极端姿态与大尺度对象仍易出现细节失真。

---

## 55. Neural Functional Alignment Space: Brain-Referenced Representation of Artificial Neural Networks

**arXiv ID:** 2603.00793 | [PDF](https://arxiv.org/pdf/2603.00793v1)

**作者:** Ruiyu Yan `[一作]` (New York University), Lin Zhao `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 10997 | [OpenAlex ID](https://openalex.org/A5073279939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出Neural Functional Alignment Space (NFAS)，通过动态模式分解对网络层次表示进行建模，以脑功能响应为参照对模型进行对齐。

**💡 创新点**

创新点在于将网络深度视为动力学轨迹并用DMD提取稳定模式，构建脑参照的坐标系，并引入Signal-to-Noise Consistency Index (SNCI) 对跨模态一致性进行量化。

**🔧 技术方法**

使用动态模式分解 (DMD)、线性编码模型、平方 Pearson 相关、PCA、PERMANOVA 与 ANOVA 等统计方法。

**📊 数据集**

评估使用了三大 fMRI 基准：Narratives、Algonauts 2021、The Little Prince，共 45 个预训练模型（视觉、音频、语言）。

**📈 对比分析**

通过与脑 ROI 的平方相关度计算对齐分数，并利用 SNCI 衡量跨模型一致性，结果显示 NFAS 在三种模态间实现明显聚类，能够捕捉脑‑网络对应关系。

**⚠️ 局限性**

局限在于仅使用静态 fMRI 数据、假设线性编码模型、对 DMD 稳定模式的选择主观、以及未考虑模型训练过程动态变化。

---

## 56. Spectral Condition for $μ$P under Width-Depth Scaling

**arXiv ID:** 2603.00541 | [PDF](https://arxiv.org/pdf/2603.00541v1)

**作者:** Chenyu Zheng `[一作]` (Renmin University of China), Chongxuan Li `[通讯]` (Renmin University of China)

**通讯引用:** 1753 | [OpenAlex ID](https://openalex.org/A5072905534)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一个统一谱视角的μP框架，用于实现宽度与深度同时扩展的生成基础模型的稳定特征学习和超参数迁移。

**💡 创新点**

创新点在于将μP的谱条件从仅宽度扩展到宽度+深度，并给出一套通用的谱约束与超参数化方法，涵盖多种优化器并统一推导公式。

**🔧 技术方法**

使用谱条件分析、线性代数与概率推导、Mu**on-Kimi**、AdamW等优化器的谱参数化，以及实验验证技术。

**📊 数据集**

在OpenWebText数据集上进行实验。

**📈 对比分析**

通过与标准SP（普通参数化）对比，评估特征尺度保持、超参数迁移稳定性以及训练损失，结果表明μP保持特征尺度不变、超参数迁移效果好且损失更低。

**⚠️ 局限性**

局限性：推导基于简化的线性残差MLP模型，尚未覆盖非线性网络、多步梯度更新、不同结构的完整情况；实际应用仍需经验验证。

---

## 57. SCATR: Mitigating New Instance Suppression in LiDAR-based Tracking-by-Attention via Second Chance Assignment and Track Query Dropout

**arXiv ID:** 2603.01485 | [PDF](https://arxiv.org/pdf/2603.01485v1)

**作者:** Brian Cheong `[一作]` (University of Toronto), Steven L. Waslander `[通讯]` (University of Toronto)

**通讯引用:** 10075 | [OpenAlex ID](https://openalex.org/A5024242059)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SCATR，一种 LiDAR‑based tracking‑by‑attention（TBA）框架，并通过两种针对性训练策略系统解决新实例抑制问题；

**💡 创新点**

创新点包括：①Track Query Dropout——在训练时随机丢弃部分轨迹查询，形成多组查询组合，提升对缺失或新出生轨迹的鲁棒性；②Second Chance Assignment——在匈牙利匹配前将未分配的轨迹查询与 proposal 查询拼接，给轨迹查询“第二机会”，缓解检测与跟踪冲突导致的漏检；

**🔧 技术方法**

技术手段包括：两阶段 Transformer 解码器（检测解码器 + 轨迹解码器）、BEV 特征编码、anchor‑based proposal 查询、Hungarian 匹配、Group‑DETR 思路、稀疏查询、深度 Transformer、以及上述两种训练策略；

**📊 数据集**

使用了 nuScenes 追踪基准（700 训练、150 验证、150 测试，2 Hz、10 检测类 + 7 跟踪类）；

**📈 对比分析**

在 nuScenes 验证/测试集上与现有 TBA（JDT3D、MotionTrack）和 TBD（CenterPoint、SimpleTrack）方法对比。SCATR 在 AMOTA 上比 JDT3D 提升 7.6%（测试）/6.6%（验证），FN 减少 26%，IDS 减少 19.7%，实现了 LiDAR‑based TBA 的最高性能，并显著缩小了与 TBD 的性能差距；

**⚠️ 局限性**

局限性：虽然与 TBD 方法差距明显缩小，但仍略低于顶尖 TBD 的 AMOTA 与 mAP，主要因检测头未使用最先进的检测器；训练时多组 dropout 增加了计算成本；未实现多模融合，仍有进一步提升空间。

---

## 58. The MAMA-MIA Challenge: Advancing Generalizability and Fairness in Breast MRI Tumor Segmentation and Treatment Response Prediction

**arXiv ID:** 2603.01250 | [PDF](https://arxiv.org/pdf/2603.01250v1)

**作者:** Lidia Garrucho `[一作]` (Barcelona Artificial Intelligence in Medicine Lab), Karim Lekadir `[通讯]` (Institució Catalana de Recerca i Estudis Avançats)

**通讯引用:** 6818 | [OpenAlex ID](https://openalex.org/A5078391768)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了大规模跨中心的 MAMA‑MIA 基准，联合评估乳腺 MRI 的肿瘤分割和术前化疗后的病理完全缓解（pCR）预测，强调模型的泛化性与公平性。

**💡 创新点**

创新点在于：①首次将分割与响应预测两大任务统一到同一基准；②采用公平性加权评分框架，兼顾整体性能与年龄、绝经状态、乳腺密度等亚组一致性；③通过公开数据、评测流程与基准实现可复现、可持续的竞赛生态。

**🔧 技术方法**

主要技术包括 3D nnU‑Net 与残差编码器、Vision Transformer、深度自监督预训练（Masked Autoencoding、SimCLR）、多阶段 DCE 输入、投影与后处理、集成学习以及 XGBoost 等传统机器学习模型；评估使用 Dice、Hausdorff、Balanced Accuracy 等指标。

**📊 数据集**

使用 MAMA‑MIA 训练集（1506 病例，来源 4 机构）与 3 个欧洲中心（574 病例，分 30/232/312）构成的公开/私有数据集，涵盖多种扫描协议、磁场强度、造影剂与乳腺密度。

**📈 对比分析**

在分割任务中，前 5 名团队 DSC 达到 0.81‑0.82，NormHD 低于 0.10，公平性得分均高于基线，表明模型在不同亚组间表现一致；在 pCR 预测中，最高综合得分为 0.691，平衡准确率仅略高于随机（0.504），但公平性明显提升，说明目前仅凭预处理 MRI 难以实现高精度的响应预测。

**⚠️ 局限性**

局限性包括：① pCR 预测性能受限于单次预处理 MRI、样本不平衡与响应标记噪声；② 分割仍在小/非质斑肿瘤、乳腺植入与低对比度场景下表现差；③ 公平性指标仅基于均值差异，可能忽视亚组内部波动；④ 目前缺乏多模态、时间序列与临床变量的整合，导致模型对真实临床变异的适应不足。

---

## 59. WildActor: Unconstrained Identity-Preserving Video Generation

**arXiv ID:** 2603.00586 | [PDF](https://arxiv.org/pdf/2603.00586v1)

**作者:** Qin Guo `[一作]` (University of Science and Technology), Dan Xu `[通讯]` (University of Science and Technology)

**通讯引用:** 71675 | [OpenAlex ID](https://openalex.org/A5100460802)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种任何视角条件下的身份保持人类视频生成框架，并构建了大规模人类视频数据集。

**💡 创新点**

提出对称身份保持注意力（AIPA）与视角自适应蒙特卡洛采样，在多视角下实现全身身份一致性，并通过I‑RoPE区分视频与参考令牌。

**🔧 技术方法**

基于Latent Diffusion Transformer（DiT）+ Rectified Flow，配合LoRA、3D RoPE、视角自适应采样和多模态LLM进行训练与验证。

**📊 数据集**

使用自建的WildActor数据集，包含1.6M视频与18M多视角参考图，覆盖多环境、多姿态。

**📈 对比分析**

与Qwen‑Image‑Edit+I2V、T2V→I2V、VACE、Stand‑In、Kling、Vidu Q2等基线对比，分别在Sequential Narrative和Contextual Generalization任务中在Face Identity、Body Consistency和Semantic Alignment等指标上取得最高或与顶尖闭源模型相当的性能。

**⚠️ 局限性**

仍受训练数据视角不平衡、极端背景/光照鲁棒性不足影响，且模型在极端姿态或细粒度文本指令下可能出现姿态锁定或身份漂移的边缘情况。

---

## 60. Consistent Low-Rank Approximation

**arXiv ID:** 2603.02148 | [PDF](https://arxiv.org/pdf/2603.02148v1)

**作者:** David P. Woodruff `[一作]` (Carnegie Mellon University), Samson Zhou `[通讯]` (Texas A&M University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究一致低秩逼近问题，设计在线算法在保持近似误差的同时最小化子空间更新量（recourse）。

**💡 创新点**

首次把recourse概念引入低秩逼近，给出近似保证下的子线性更新量，并证明相应的下界；同时提出多种实现方案（整数约束、在线条件数场景）。

**🔧 技术方法**

利用在线Ridge leverage score采样、频繁方向（Frequent Directions）与SVD更新、奇异值分解的最小化以及抗Hadamard矩阵的结构和整数矩阵的奇异值下界进行理论与算法设计。

**📊 数据集**

实验中使用Landmark、Skin Segmentation、Rice、随机合成等四个数据集进行评估。

**📈 对比分析**

与传统Frequent Directions、随机SVD等方法比较，展示recourse降低约400倍，逼近质量在实际中优于理论上给出的上界，近似比值普遍在1~4之间。

**⚠️ 局限性**

理论下界仅为Ω(k/log(n/k))，而实现的recourse仍为O(n√k + k/ log²(ndM))，在极端数据或高维情形下可能仍不够低；对抗Hadamard矩阵等特殊情况的处理尚有提升空间。

---

## 61. Event-Anchored Frame Selection for Effective Long-Video Understanding

**arXiv ID:** 2603.00983 | [PDF](https://arxiv.org/pdf/2603.00983v1)

**作者:** Wang Chen `[一作]` (Xiamen University), Xiawu Zheng `[通讯]` (Xiamen University)

**通讯引用:** 1422 | [OpenAlex ID](https://openalex.org/A5054226277)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种无训练、事件感知的关键帧选择方法——Event‑Anchored Frame Selection (EFS)，用于提升大视觉语言模型在长视频理解中的性能。

**💡 创新点**

创新点在于将视频分割成视觉同质的事件段并在每个事件内挑选查询最相关的帧作为锚点，再通过自适应的最大边际相关性（MMR）进行全局细化，三步实现事件覆盖、查询相关性与视觉多样性的协同优化。

**🔧 技术方法**

核心技术包括自监督视觉嵌入 DINOv2 用于检测事件边界和计算视觉相似度，BLIP2‑ITM 用于评估帧与查询的语义相关性，以及基于锚点的自适应 MMR 进行全局帧筛选。

**📊 数据集**

在 VideoMME、LongVideoBench 与 MLVU 三大长视频问答基准上进行评估。

**📈 对比分析**

与传统的均匀采样以及多种基于查询的采样方法相比，EFS 在 LLaVA‑Video‑7B 上分别提升 4.7%、4.9% 与 8.8% 的准确率，并在其他开源 LVLM（如 LLaVA‑OneVision、Qwen2.5‑VL）上同样获得显著增益；在相同帧数预算下，EFS 也持续优于现有采样策略。

**⚠️ 局限性**

主要限制包括：额外的预处理开销（视觉与语义特征提取占比约 90%），对帧率和预训练模型的依赖，且在极长视频或极低帧率场景下可能需进一步优化。

---

## 62. ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration

**arXiv ID:** 2603.00906 | [PDF](https://arxiv.org/pdf/2603.00906v1)

**作者:** Xiaolong Zeng `[一作]` (Tsinghua University), Bin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 51691 | [OpenAlex ID](https://openalex.org/A5100372375)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ShiftLUT框架，融合可学习的空间偏移、非对称双分支以及自适应采样压缩，显著提升 LUT 基础图像恢复性能并降低资源消耗。

**💡 创新点**

创新点包括：1）Learnable Spatial Shift (LSS) 模块通过通道级学习空间偏移扩大感受野；2）非对称双分支设计，将计算重点放在信息密集的 MSB 分支；3）Error-bounded Adaptive Sampling (EAS) 通过误差约束自动选择采样步长并缓存插值，减小存储并保持速度。

**🔧 技术方法**

核心技术包括：1) 低维 LUT 替代卷积的 1D LUT ；2) 双分支 MSB/LSB 结构与可学习空间偏移的 Shift‑Block；3) 端到端训练与两阶段权重替换；4) 采样步长自适应优化与缓存加速。

**📊 数据集**

使用 DIV2K 作为训练集，并在标准的 Set5/Set14/BSDS100/Urban100/Manga109（超分）、Set12/BSD68（去噪）以及 Classic5/LIVE1（去块）等公开基准数据集进行评估。

**📈 对比分析**

与多种 LUT 及 DNN 方法（如 TinyLUT‑F、MuLUT、RCLUT、SPFLUT、ECLUT、FSRCNN、VDSR 等）在 PSNR/SSIM、存储尺寸和推理时间上对比，ShiftLUT‑L 在 PSNR 上超过 TinyLUT‑F、并将 LUT 大小从 171 KB 降至 104 KB，运行时间从 146 ms 降至 84 ms；ShiftLUT‑M 在 PSNR 上优于 FSRCNN，速度提升 11×；ShiftLUT‑S 则实现最小存储与最快速度。

**⚠️ 局限性**

局限性在于：1）仍依赖 LUT 的离散化，极高分辨率或大尺寸图像可能需要更细粒度的 LUT；2）两阶段训练中偏移量固定后可能错过输入依赖的细微调节；3）在更复杂的恢复任务（如视频增强、实时多帧）中，非对称分支的设计可能需要进一步调整。

---

## 63. Closing the Gap Between Float and Posit Hardware Efficiency

**arXiv ID:** 2603.01615 | [PDF](https://arxiv.org/pdf/2603.01615v1)

**作者:** Aditya Anirudh Jonnalagadda `[一作]` (Birla Institute of Technology and Science), John L. Gustafson `[通讯]` (Arizona State University)

**通讯引用:** 4113 | [OpenAlex ID](https://openalex.org/A5056691858)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了32/64位的b‑posit解码器和编码器，显著降低功耗、面积和延迟。

**💡 创新点**

创新点是将regime长度限定为6位（rS=6），并采用固定指数位数eS，通过一热编码和多路复用器实现并行解码/编码，从而简化硬件并提升性能。

**🔧 技术方法**

使用组合逻辑、优先编码器、二进制解码器、5输入多路复用器以及XOR/NOT/AND门，并在SiliconCompiler+freepdk45工艺下实现RTL到后布局的硬件设计。

**📊 数据集**

未使用专门的数据集，性能评估基于随机输入向量以及最坏情况的能耗计算。

**📈 对比分析**

通过与标准posits和IEEE float在峰值功耗、面积、延迟及能耗pJ的对比，b‑posit在32/64位时比float更快、功耗更低、面积更小，整体性能优越。

**⚠️ 局限性**

局限在于缺乏对实际AI/HPC工作负载的精度验证，以及对更大rS配置的可扩展性和实现复杂度的进一步研究。

---

## 64. GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis

**arXiv ID:** 2603.01010 | [PDF](https://arxiv.org/pdf/2603.01010v1)

**作者:** Xuqin Wang `[一作]` (Huawei), Daniel Cremers `[通讯]` (Technical University of Munich)

**通讯引用:** 48617 | [OpenAlex ID](https://openalex.org/A5087710605)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出数据到数据的流匹配（D2D‑FM）和基于概率密度的测地线流匹配（PDG‑FM）以实现视角一致的合成

**💡 创新点**

首次将概率密度测地线作为几何正则化直接融入条件流匹配，且采用确定性数据对间的流学习而非噪声到数据的映射

**🔧 技术方法**

利用预训练的扩散模型得分作为密度近似，构建GeodesicNet；使用U‑Net+Plücker射线、CLIP、VAE编码的条件网络训练速度场；进行变分蒸馏与变分测地线优化

**📊 数据集**

Objaverse（多视角渲染数据集）和GSO30（外域数据集）

**📈 对比分析**

与Zero‑1‑to‑3、EscherNet、Free3D以及噪声到数据的流匹配基线对比；在Objaverse与GSO上取得更低的FID、更高的CLIP‑S、SSIM、PSNR以及更低的LPIPS，尤其在10步NFE推理时仍保持优势

**⚠️ 局限性**

训练过程分两阶段，需先蒸馏测地线再训练速度场，计算成本高、对资源要求大，限制了模型的可扩展性

---

## 65. Conformal Policy Control

**arXiv ID:** 2603.02196 | [PDF](https://arxiv.org/pdf/2603.02196v1)

**作者:** Drew Prinster `[一作]` (Genentech), Samuel Stanton `[通讯]` (Genentech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于对数比率截断的保守策略控制框架（Conformal Policy Control, CPC），通过对安全参考策略与优化后策略的似然比进行校准，实现安全探索与性能提升的平衡。

**💡 创新点**

创新点在于将 conformal risk control 扩展到非单调损失，并将控制参数从损失函数转移到策略分布（即似然比阈值），从而在不假设模型结构、无需调参的前提下在有限样本下保证风险阈值；同时提出从安全到激进的参数搜索策略，确保即使在多步反馈导致的交换性破坏时仍可提供风险控制。

**🔧 技术方法**

核心技术包括：
- 对安全策略数据进行加权重要性重采样与 conformal 校准；
- 似然比截断形成的可插值策略；
- 对非单调损失的通用 CRC（gCRC）与 Lipschitz、替换稳定性分析；
- 采用接受-拒绝采样实现高维组合动作空间下的高效推断。

**📊 数据集**

实验数据集：
- MedLFQA（医学问答）用于 FDR 控制；
- Robot Arm Kinematics、Airfoil、Healthcare Utilization（MEPS）用于受限主动学习；
- Ehrlich 函数（生物分子序列优化）用于黑箱序列优化。

**📈 对比分析**

方法对比：
- 与标准 CRC、Learn‑Then‑Test（LTT）在 MedLFQA 上相比，gCRC 在控制 FDR 的同时显著提升召回率；
- 在主动学习中，CPC 在保证约束违约风险低于设定 α 的同时降低测试 MSE，部分情形下甚至优于无约束策略；
- 在黑箱优化中，CPC 能有效抑制无效样本生成，提升目标函数值，且中等风险控制（α>0.6）可进一步提升整体性能。

**⚠️ 局限性**

局限性：
- 仅提供平均风险控制保证，缺乏针对单个上下文的条件安全保证；
- 依赖安全策略数据的分布稳定性，若上下文分布漂移需重新校准；
- 需要能够显式计算策略似然比，若不可得需使用比率估计或神经估计方法；
- 对高度复杂或连续动作空间，计算效率与采样复杂度仍是挑战。

---

## 66. Opponent State Inference Under Partial Observability: An HMM-POMDP Framework for 2026 Formula 1 Energy Strategy

**arXiv ID:** 2603.01290 | [PDF](https://arxiv.org/pdf/2603.01290v1)

**作者:** Kalliopi Kleisarchaki `[一作]` `[通讯]` (Independent Researcher), Kalliopi Kleisarchaki (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个两层模型框架，利用30状态隐马尔可夫模型（HMM）推断对手能量与轮胎状态，再以DQN决策策略在2026年F1能量管理中做出最优部署。

**💡 创新点**

创新点在于将对手状态建模为可观测的30状态HMM，首次形式化并检测“counter‑harvest trap”这一基于Active Aero的欺骗策略，并将belief state输入到POMDP‑DQN中实现端到端决策。

**🔧 技术方法**

使用技术包括隐马尔可夫模型前向推理、Baum‑Welch EM参数学习、双重DQN（含经验回放、Huber损失、梯度裁剪）以及合成赛果模拟进行闭环验证。

**📊 数据集**

数据集为2026赛季官方Telemetry（FastF1）提供的公开观测信号（Δv_trap、Δt_sector、Δb_brake、σ²_speed、z_aero）以及20场合成赛果作为验证数据。

**📈 对比分析**

通过与四个基线（确定阈值、观测阈值、Oracle、全系统）对比，HMM+ DQN在合成实验中ERS推断准确率92.3%，抢占策略检测召回率95.7%，并在对抗性情境下显著优于单观测阈值策略。

**⚠️ 局限性**

主要局限包括：对手被假设为静态过程，忽略了交互式博弈动态；条件独立的发射模型导致校准误差；训练数据量受限，需在真实赛季（如Melbourne）中进一步验证和校正。

---

## 67. Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning

**arXiv ID:** 2603.01696 | [PDF](https://arxiv.org/pdf/2603.01696v1)

**作者:** Haonan Jia `[一作]` (Taobao and Tmall Group of Alibaba), Kaifu Zhang `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种不需要人工标注的跨模态身份映射（CIM）框架，利用检索结果评估并强化视觉-语言模型在图像字幕任务中的信息保留与细粒度描述。

**💡 创新点**

创新点在于将字幕作为查询，利用检索到的图像集合衡量信息损失，并将两项检索指标（Gallery Representation Consistency GRC 与 Query-gallery Image Relevance QIR）融合为奖励函数，指导RL训练从而实现图像-文本身份映射。

**🔧 技术方法**

核心技术包括：检索式评价指标（GRC、QIR）、基于Group Relative Policy Optimization (GRPO) 的强化学习、无监督奖励设计、以及多模型（LLaVA、Qwen-VL、InternVL）和多编码器（OpenCLIP、DINOv3、SBERT/MiniLM、MPNet）的实验验证。

**📊 数据集**

使用的公开数据集包括 RefinedCaps（6.5k图像）用于RL训练；COCO-LN500 与 DOCCI500（各500对）作为评估基准；此外在信息损失验证中使用 Oxford-IIIT Pet 分类数据集。

**📈 对比分析**

与基线（基模型、SFT、SC-Captioner）对比，CIM 在 COCO-LN500 和 DOCCI500 上在 CAPTURE、对象/属性/关系 F1、QA 等指标上平均提升 3–12%，在 Qwen2.5-VL-7B 上甚至超过 20% 的关系 QA 分数；在检索编码器多样性测试中表现稳定。

**⚠️ 局限性**

局限性包括：奖励函数仍依赖检索质量，若检索语料库或编码器效果差会影响评估；RL 训练耗时较长；在对象识别子任务上提升有限，主要受限于全局奖励的细粒度信息捕捉不足。

---

## 68. BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling

**arXiv ID:** 2603.01163 | [PDF](https://arxiv.org/pdf/2603.01163v1)

**作者:** Jiachen Yang `[一作]` (Shenzhen Campus of Sun Yat-sen University), Yanmei Fang `[通讯]` (Guangdong Provincial Key Laboratory of Information Security Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于强化学习的面部美颜框架 BeautyGRPO，通过奖励模型与动态路径引导实现对人类审美偏好的自适应优化。

**💡 创新点**

创新点：①构建细粒度偏好数据集 FRPref-10K 及专用多维奖励模型；②提出 Dynamic Path Guidance (DPG)，在在线 RL 中平衡探索与高保真，避免随机漂移导致噪点；③将奖励模型与 FlowGRPO 结合，首次在面部美颜任务中实现自我学习与人类审美的闭环。

**🔧 技术方法**

技术：FlowGRPO 变体、动态路径引导（DPG）、奖励模型三阶段训练（SFT→自监督→GRPO）、LoRA 微调、FluxKontext/ Qwen-Image-Edit 基础模型、无参考美学评估指标与 ArcFace 识别一致性评估。

**📊 数据集**

数据集：FRPref-10K（10k 对比样本，覆盖皮肤平滑、瑕疵去除、纹理质量、清晰度、身份保留五维度）；测试集使用 FFHQR 与公开的 in-the-wild 1000 张真实人像。

**📈 对比分析**

对比方法：RetouchFormer、VRetouchEr、NanoBanana、SeedDream4.0、FluxKontext+FlowGRPO 等。BeautyGRPO 在 NIQE、NIMA、MUSIQ、MANIQA、NRQM、TOPIQ、FID 与 ArcFace 等指标上均显著优于基线，用户研究胜率高达 63% 以上，验证其更好地符合人类审美。

**⚠️ 局限性**

局限：①需要大量人类与 VLM 标注的数据，构建成本高；②DPG 依赖高偏好 anchor 样本，对极端光照/姿态下的鲁棒性有限；③RL 训练耗时且对奖励信号敏感，模型规模大，部署成本较高。

---

## 69. Differential privacy representation geometry for medical image analysis

**arXiv ID:** 2603.01098 | [PDF](https://arxiv.org/pdf/2603.01098v1)

**作者:** Soroosh Tayebi Arasteh `[一作]` (RWTH Aachen University), Daniel Truhn `[通讯]` (RWTH Aachen University)

**通讯引用:** 6534 | [OpenAlex ID](https://openalex.org/A5016512818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 DP‑RGMI 的框架，用于在医疗影像深度学习中从表示几何和利用率两个维度评估差分隐私对模型性能的影响。

**💡 创新点**

创新点在于将差分隐私解释为对编码器表示空间的结构化变换，并将性能衰减拆分为表示位移、谱有效维度和利用率缺口三部分，从而揭示隐私噪声导致的利用率下降而非单纯的线性可分性丧失。

**🔧 技术方法**

使用 DP‑SGD、谱有效维度计算（基于协方差矩阵）、线性探针评估、欧氏表示位移度量和 Spearman 相关分析等技术；模型采用 ConvNeXt‑Small 编码器，配合不同的预训练初始化（ImageNet、DinoV3、MIMIC‑CXR）。

**📊 数据集**

主要在四个公开胸部 X‑ray 数据集上实验：PadChest（110k 图像）作为基准，另外 CheXpert（157k 图像）和 ChestX‑ray14（112k 图像）用于泛化评估；数据被按病人分层划分为训练、验证和测试集。

**📈 对比分析**

对比方法：非私有训练（ε=∞）与不同 ε 的 DP 训练，评估 end‑to‑end AUROC 与线性探针 AUROC 的差距 G；结果显示在强隐私下 G 明显增大（例如 ImageNet+ε=1.0 时 G≈8.0），表明可分性保留但未被充分利用；同时 Δ 与 d_eff 的变化非单调，受初始化和数据集影响，提供更细粒度的诊断。

**⚠️ 局限性**

局限性包括：仅在多标签胸部 X‑ray 分类任务验证，尚未证明在分割或其他医学影像任务中的适用性；利用率缺口定义与 AUROC 紧密相关，可能不完全揭示优化层面的根本原因；对预训练模型的依赖较大，初始化差异导致结果解释复杂；未深入探讨如何在实践中根据诊断结果调整训练策略。

---

## 70. Online Generation of Collision-Free Trajectories in Dynamic Environments

**arXiv ID:** 2603.00759 | [PDF](https://arxiv.org/pdf/2603.00759v1)

**作者:** Nermin Covic `[一作]`, Bakir Lacevic `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在线方法，将任意几何路径转换为具有有限加加速度（jerk）约束的可运动学可行轨迹，利用二次或四次/五次样条；

**💡 创新点**

创新点在于：①基于“自由空间气泡”概念的快速碰撞检测与安全保证；②使用二分法快速求解每个关节的加加速度系数，实现实时轨迹生成；③兼顾动态环境，提供安全轨迹与紧急停止策略；

**🔧 技术方法**

技术实现包括：quintic/quartic样条插值、气泡（bur）与扩展气泡（DEB）碰撞检测、Bisection求解、实时距离与最小距离估计、与Ruckig库集成、ROS2实时控制；

**📊 数据集**

数据集：UFactory xArm6机器人模型、1000次随机起止配置的仿真；动态障碍物场景（10个随机移动障碍或4个大障碍），实验使用两款深度相机实时感知；

**📈 对比分析**

与Ruckig进行对比：在两种动态规划器（ORRT、DRGBT）下，CFS45平均生成时间≈Ruckig的1/3；轨迹平滑度（加加速度L1范数）提升约3.3倍；在高频规划（T≤10ms）时成功率、算法时间和路径长度均优于Ruckig；

**⚠️ 局限性**

局限性：仅针对机械臂类多自由度机器人，未评估大规模DOF可扩展性；实验缺乏加速度反馈；对极端高速动态障碍物的安全保证仍需进一步验证。

---

## 71. GAC: Stabilizing Asynchronous RL Training for LLMs via Gradient Alignment Control

**arXiv ID:** 2603.01501 | [PDF](https://arxiv.org/pdf/2603.01501v1)

**作者:** Haofeng Xu `[一作]` (Alibaba Group), Chuan Wu `[通讯]` (University of Hong Kong)

**通讯引用:** 11026 | [OpenAlex ID](https://openalex.org/A5012597518)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为Gradient Alignment Control (GAC) 的动态调节方法，用以稳定大语言模型在异步强化学习（GRPO）训练过程中的梯度更新；

**💡 创新点**

创新点在于首次将梯度相似度（连续梯度余弦相似度）作为训练不稳定的早期指标，并通过方向性梯度投影抑制梯度共线性，从而消除由梯度陈旧导致的系统性偏差；

**🔧 技术方法**

核心技术包括基于梯度余弦相似度的动态阈值控制、单向梯度投影（分解平行与正交分量）以及对齐消除的理论分析与收敛保证；

**📊 数据集**

实验使用了七个数学推理基准（AIME24/25、AMC23/24、Math500、OlympiadBench、MinervaMath）以及Qwen3和Llama-3.2-3B-Instruct 等大型语言模型；

**📈 对比分析**

与同步GRPO、GRPO+旧版异步、M2PO及BAPO等基线比较时，GAC在高陈旧度（s≥8）下实现了与同步训练相近的最终性能，且在大多数基准上显著提升准确率，成功关闭了同步与异步之间的性能差距；

**⚠️ 局限性**

主要局限在于GAC需要对梯度相似度做阈值划分，阈值选择对不同任务和模型规模可能需要调优，同时其理论分析假设梯度流畅且局部线性，实际应用中可能存在未考虑的高阶非线性影响。

---

## 72. GuiDINO: Rethinking Vision Foundation Model in Medical Image Segmentation

**arXiv ID:** 2603.01115 | [PDF](https://arxiv.org/pdf/2603.01115v1)

**作者:** Zhuonan Liang `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**通讯引用:** 11910 | [OpenAlex ID](https://openalex.org/A5076697411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种名为GuiDINO的框架，将预训练的DINOv3视觉基础模型用作医学图像分割的视觉引导生成器，生成引导掩码并在多种分割骨干网络中进行门控。

**💡 创新点**

创新点在于将冻结的基础模型通过轻量级TokenBook机制转换为空间引导掩码，并在不进行全微调的情况下注入模型先验，同时支持LoRA参数高效适配。

**🔧 技术方法**

使用技术包括DINOv3预训练视觉编码器、TokenBook生成引导掩码、门控机制、组合损失（分割损失+引导对齐损失）以及可选的边界焦点切换损失和LoRA适配。

**📊 数据集**

在Kvasir-SEG（大肠息肉）、ISIC 2017（皮肤病变）和TN3K（甲状腺超声结节）三大医学图像分割数据集上进行实验。

**📈 对比分析**

与nnUNet、SwinUNet、H2Former、U-KAN、nnWNet以及SegDINO等基线方法对比，GuiDINO在IoU、Dice和HD95等指标上实现了显著提升，尤其在Kvasir和ISIC上取得最高分数。

**⚠️ 局限性**

局限性包括引导掩码的精度受限于基础模型的域迁移效果，LoRA适配效果不一，且对计算资源和GPU显存仍有一定需求；进一步研究需要评估在更多医学模态和更大规模数据上的鲁棒性。

---

## 73. Where Do Smart Contract Security Analyzers Fall Short?

**arXiv ID:** 2603.00890 | [PDF](https://arxiv.org/pdf/2603.00890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 74. Action-Guided Attention for Video Action Anticipation

**arXiv ID:** 2603.01743 | [PDF](https://arxiv.org/pdf/2603.01743v1)

**作者:** Tsung-Ming Tai `[一作]` (NVIDIA), Oswald Lanz `[通讯]` (Free University of Bozen-Bolzano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于动作预测的注意机制（Action‑Guided Attention，AGA），在视频动作预测任务中通过动作序列来引导注意力并自适应融合历史与当前视觉信息；

**💡 创新点**

创新点在于用高层动作概率作为查询与键，替代传统像素级自注意力，显著减少对视觉噪声的过拟合，并支持事后前向与反向分析揭示模型内部决策路径；

**🔧 技术方法**

采用Transformer结构中的多头点乘注意力，结合EMA平滑的动作预测作为查询、MLP映射键和值、RMSNorm、预归一化、以及自适应门控混合；

**📊 数据集**

主要在EPIC‑Kitchens‑100（以及EPIC‑Kitchens‑55、EGTEA Gaze+）上进行实验；

**📈 对比分析**

与现有基线（如AVT、MemViT、RaftFormer、AFFT、S‑GEAR等）相比，在EPIC‑Kitchens‑100的验证集和未见测试集上实现了更高的Top‑5 Recall/准确率，验证了模型的良好泛化；

**⚠️ 局限性**

局限性包括对动作预测质量高度依赖，训练中需要保存完整动作序列的显存，且在数据稀疏或标签不完整的场景下性能可能受限。

---

## 75. RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations

**arXiv ID:** 2603.01194 | [PDF](https://arxiv.org/pdf/2603.01194v1)

**作者:** Mochu Xiang `[一作]` (Northwestern Polytechnical University), Yuchao Dai `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 12590 | [OpenAlex ID](https://openalex.org/A5036202579)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一种名为RnG的统一Transformer框架，能够从少量无姿态图像中同时实现完整3D重建和新视角生成。

**💡 创新点**

核心创新是重建引导的因果注意力机制和KV-Cache作为隐式完整3D表示，实现在一次前向推理中完成重建与生成。

**🔧 技术方法**

使用了Transformer、因果注意力、KV-Cache、Plücker射线编码、DINO特征提取、DPT头等技术。

**📊 数据集**

训练使用Objaverse（113.5k对象），评估使用Google Scanned Objects（GSO）数据集。

**📈 对比分析**

与现有多视角重建与新视角合成方法（VGGT、LVSM、Matrix3D等）对比，RnG在相机姿态估计、深度精度、Chamfer距离、PSNR/SSIM等指标上均达到或超过最先进水平，并实现实时推理（≈85 ms）。

**⚠️ 局限性**

局限在细节纹理缺失、对世界坐标原点的依赖、需要多视角累积完成完整3D，以及对动态或非刚性物体适应性不足。

---

## 76. Compensation-free Machine Unlearning in Text-to-Image Diffusion Models by Eliminating the Mutual Information

**arXiv ID:** 2603.00992 | [PDF](https://arxiv.org/pdf/2603.00992v1)

**作者:** Xinwen Cheng `[一作]` (Shanghai Jiao Tong University), Xiaolin Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9995 | [OpenAlex ID](https://openalex.org/A5005338317)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无需补偿的概念遗忘方法MiM‑MU，直接通过降低文本概念与生成图像之间的互信息来实现概念抹除；

**💡 创新点**

核心创新在于：①将概念遗忘视为信息论问题，利用预训练扩散模型估计条件/无条件分布，最小化互信息；②在此基础上对未学习模型的条件分布与预训练模型无条件分布对齐，保证对其他概念的影响最小；

**🔧 技术方法**

技术手段包括：信息理论互信息最小化、预训练扩散模型作为判别器、基于无条件分布的KL对齐、消除U-Net雅可比项以降低计算成本；

**📊 数据集**

主要使用UnlearnCanvas基准（包含50种艺术风格+20种物体共70个概念），并在Stanford Dogs、Oxford Flowers、CUB‑200等细粒度数据集上进行细粒度概念遗忘评估；

**📈 对比分析**

与现有9种MU方法（如SalUn、ESD、FMN等）以及SDD对比，MiM‑MU在概念遗忘率、在域内外保留率（IRA/CRA）均达90%以上，FID最低，且在多概念、细粒度场景下仍保持优异表现；

**⚠️ 局限性**

局限性包括：仍需依赖预训练扩散模型；对高度互相关的概念（如语义交叉）抹除效果待进一步完善；训练时计算成本相对较高，未在极大模型规模上验证；

---

## 77. Consensus and fragmentation in academic publication preferences

**arXiv ID:** 2603.00807 | [PDF](https://arxiv.org/pdf/2603.00807v1)

**作者:** Ian Van Buskirk `[一作]` (University of Colorado Boulder), Daniel B. Larremore `[通讯]` (University of Colorado Boulder)

**通讯引用:** 8006 | [OpenAlex ID](https://openalex.org/A5027079032)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

**🎯 论文内容**

通过三阶段自适应调查，收集了美国终身追踪教员对学术期刊的偏好，并将偏好与学科、机构声望、性别等因素关联。

**💡 创新点**

首次在同一方法框架下横向比较13个学科的期刊偏好，揭示高共识学科与低共识学科的显著差异；发现期刊影响因子与实际偏好吻合度不足；同时揭示性别和机构声望对期刊偏好的系统影响。

**🔧 技术方法**

采用SpringRank算法对个体和学科层面的成对比较生成排名；利用多元线性回归和留一预测准确率评估共识程度；还使用前5名一致度衡量偏好一致性。

**📊 数据集**

数据集为3,510名美国终身追踪教员的调查结果，包含163,002个成对比较、13个学科的期刊列表，以及每位受访者的学术背景信息。

**📈 对比分析**

通过留一预测准确率和前5名一致度等指标进行比较；结果显示高共识学科（如经济学）预测准确率可达90%，而低共识学科（如计算机科学）仅约50%；这些指标能够定量衡量学科内部偏好的一致性。

**⚠️ 局限性**

研究局限在于样本仅限美国终身教员，响应率存在偏倚；偏好为自报偏好，未能捕捉实际投稿行为；对偏好形成机制的解释有限；此外，期刊影响因子的覆盖率和可比性不足。

---

## 78. QIME: Constructing Interpretable Medical Text Embeddings via Ontology-Grounded Questions

**arXiv ID:** 2603.01690 | [PDF](https://arxiv.org/pdf/2603.01690v1)

**作者:** Yixuan Tang `[一作]` (National University of Singapore), Anthony K. H. Tung `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 QIME 框架，利用医学本体生成可解释的问答式医学文本嵌入，并实现训练无监督的稀疏二值嵌入构造方法。

**💡 创新点**

创新点在于（1）将 UMLS 概念签名用于聚类对比式问题生成，保证维度临床可解释且判别力强；（2）提供完全训练无监督的稀疏嵌入构造，消除每维分类器或 LLM 查询；（3）引入 MMR 多样性选择提升 top‑k 维度的多样性和性能。

**🔧 技术方法**

使用语义聚类、UMLS 概念链接、LLM（Qwen3‑30B）对比式问题生成、MedEmbed 编码、MMR 多样性 top‑k、稀疏二值向量等技术。

**📊 数据集**

实验基于 PubMed 大规模文本（约 25M 段落，5M 用于聚类）、MTEB 医学子集（BioP2P、BioS2S 等）、ClusTREC‑Covid、BIOSSES、NFCorpus、TRECCOVID、PublicHealthQA、MedicalQARetrieval、R2MED 等多任务数据集。

**📈 对比分析**

与强大黑盒编码器（PubMedBERT、BioLORD、MedEmbed 等）和可解释基线（QA‑Emb、CQG‑MBQA、LDIR‑500）比较，QIME 在聚类、语义相似度、检索等任务上均优于可解释基线，训练无监督变体 QIME‑TF‑MMR 在聚类甚至超越部分黑盒模型，整体显著缩小可解释与密集嵌入的性能差距。

**⚠️ 局限性**

主要限制包括对 UMLS 本体覆盖与质量的依赖，导致本体不完整或过时时问题生成质量下降；以及可解释性需求因用户群体差异而异，当前尚未在真实临床工作流中进行系统评估。

---

## 79. Legal RAG Bench: an end-to-end benchmark for legal RAG

**arXiv ID:** 2603.01710 | [PDF](https://arxiv.org/pdf/2603.01710v1)

**作者:** Abdur-Rahman Butler `[一作]`, Umar Butler `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Legal RAG Bench benchmark 与评估方法，用于系统评估法律检索增强生成（RAG）模型的端到端性能。

**💡 创新点**

创新点在于结合全因子实验设计与层次错误分解框架，可细致区分检索、推理与幻觉错误，证明检索质量是主导因素。

**🔧 技术方法**

使用三种嵌入模型（Kanon 2 Embedder、Gemini Embedding 001、Text Embedding 3 Large）和两款前沿 LLM（Gemini 3.1 Pro、GPT‑5.2）以及基于 LangChain 的统一 RAG pipeline。

**📊 数据集**

采用 4,876 条维多利亚州刑事起诉书段落与 100 条专家手工构造的复杂法律问题及其长文本答案组成的数据集。

**📈 对比分析**

通过全因子实验评估正确率、基准度和检索准确率，并利用错误分解揭示各模型贡献；Kanon 2 Embedder 在正确率提升约 17.5 %，基准度提升 4.5 %，检索准确率提升 34 %，LLM 影响相对温和。

**⚠️ 局限性**

局限性包括只聚焦维多利亚州法律，缺乏多司法区或多语言覆盖，实验使用统一 RAG pipeline，可能忽略模型特定调优带来的差异。

---

## 80. A short tour of operator learning theory: Convergence rates, statistical limits, and open questions

**arXiv ID:** 2603.00819 | [PDF](https://arxiv.org/pdf/2603.00819v1)

**作者:** Simone Brugiapaglia `[一作]` (Concordia University), Nicholas H. Nelsen `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了算子学习中经验风险最小化的误差界、压缩感知方法以及最优采样宽度等理论进展；

**💡 创新点**

创新点在于把holomorphic算子与神经网络逼近理论相结合，给出在不同正则性假设下的采样复杂度下界与上界；

**🔧 技术方法**

采用了经验过程理论、压缩感知技术、极小化分析与神经网络表达能力的熵估计；

**📊 数据集**

未使用具体实验数据集，而是以理论框架为主；

**📈 对比分析**

通过文献对比和理论推导对算子学习的性能做了定性评估，未给出数值实验；

**⚠️ 局限性**

局限性包括缺乏实证验证、对噪声模型的深入探讨有限以及在完全可训练网络下的速率仍未达最优。

---

## 81. Compliance as Code: A Study of Linux Distributions and Beyond

**arXiv ID:** 2603.01520 | [PDF](https://arxiv.org/pdf/2603.01520v1)

**作者:** Jukka Ruohonen `[一作]` (University of Southern Denmark), Hiraku Morita `[通讯]` (University of Southern Denmark)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对ComplianceAsCode项目中的1504条合规规则和102份指南进行系统收集与分析，探讨了五大Linux发行版在合规覆盖、规则相似度、所涵盖安全控制及与欧盟《网络弹性法案》（CRA）关键要求的映射情况。

**💡 创新点**

创新点在于首次对合规即代码（Compliance as Code）工具的真实规则集进行经验评估，揭示不同发行版在合规覆盖与规则设计上的差异，并验证其规则与CRA关键要求的匹配潜力，为后续自动化合规检查提供实证依据。

**🔧 技术方法**

主要采用的技术包括：文本预处理与词频(TF/T‑IDF)计算；余弦相似度评估规则代码片段与说明文本的相似度；Kruskal‑Wallis与卡方检验比较不同发行版的规则数量、严重性分布；Cohen与Fleiss κ系数评估三位作者对规则至CRA映射的可靠性。

**📊 数据集**

使用的数据集来自ComplianceAsCode项目公开档案，涵盖5个供应商（Debian、Oracle、Red Hat、SUSE、Canonical）14个发行版（共10个版本）及其对应的102份指南和1504条唯一规则。

**📈 对比分析**

通过统计检验比较各发行版的规则覆盖率和严重性分布，结果显示Red Hat系列覆盖率最高、Debian/Ubuntu相对较低；规则代码片段的平均余弦相似度约为0.55（TF‑IDF），说明代码片段相似度较高；规则与CRA关键要求的映射一致性κ系数约为0.45，表明作者间存在中等偏差。

**⚠️ 局限性**

主要局限包括：相似度评估无标准阈值，缺乏外部基准；映射工作为人工单一作者主导，虽做验证但仍存主观性；仅关注5大发行版，未覆盖全部OSS项目；CRA映射仅使用一次性分配方法，未考虑多对多关系。

---

## 82. DualSentinel: A Lightweight Framework for Detecting Targeted Attacks in Black-box LLM via Dual Entropy Lull Pattern

**arXiv ID:** 2603.01574 | [PDF](https://arxiv.org/pdf/2603.01574v1)

**作者:** Xiaoyi Pang `[一作]` (Hong Kong University of Science and Technology), Zhibo Wang `[通讯]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DualSentinel框架，能够在黑盒LLM生成过程中实时检测并阻断定向攻击。

**💡 创新点**

创新点在于发现并利用“熵寂静”（Entropy Lull）这一低熵稳定模式，并通过双重检查（任务翻转）实现几乎零误报和几乎完美检测率。

**🔧 技术方法**

使用熵监测、窗口统计、任务翻转验证等技术，基于模型返回的top-k概率进行无监督检测。

**📊 数据集**

在LLaMA、Qwen系列开源模型以及GPT‑4o等闭源模型上，利用Alpaca、XSum等公开数据集进行评估。

**📈 对比分析**

与PPL、ONION、STRIP、Paraphrase、CleanGen、ConfGuard等基线对比，DualSentinel实现TPR≈100%、FPR≈0%且ATGR≈1.0，显著优于所有基线。

**⚠️ 局限性**

局限性包括对极短攻击序列需要额外的Completed Lull判定，对任务翻转前缀的设计依赖，且要求模型返回top‑k概率。

---

## 83. Proscenium: Exploring Design Spaces of Layered Information Experience on a Large Dual-Layer Transparent Display

**arXiv ID:** 2603.01238 | [PDF](https://arxiv.org/pdf/2603.01238v1)

**作者:** Chen Chen `[一作]` (Florida International University), Nicolai Marquardt `[通讯]` (Microsoft Research)

**通讯引用:** 5288 | [OpenAlex ID](https://openalex.org/A5036819859)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个可调距离的大型双层透明 OLED 显示工作站，并基于此快速原型化了14个跨六类的层信息交互体验。

**💡 创新点**

提出了面向层信息的转移与链接两大设计维度，并通过双层可调分离的透明显示器展示了层间信息迁移的可行性与交互潜力。

**🔧 技术方法**

使用了双层 Planar LookThru LO552 55" 透明 OLED 显示器、可调铝合金支架以及预录制视频/图像数据进行快速原型实验。

**📊 数据集**

使用了预录制的视频和图片素材，未采用公开数据集。

**📈 对比分析**

未进行正式对比实验或性能评估，文章仅呈现原型展示与设计思路。

**⚠️ 局限性**

局限在于缺乏用户研究与实证验证、仅局限双层设计、原型使用录制素材而非完整交互系统。

---

## 84. A Safety-Aware Shared Autonomy Framework with BarrierIK Using Control Barrier Functions

**arXiv ID:** 2603.01705 | [PDF](https://arxiv.org/pdf/2603.01705v1)

**作者:** Berk Guler `[一作]` (Technical University of Darmstadt), Jan Peters `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在共享自主控制框架中加入了基于控制障碍函数（CBF）的逆运动学安全滤波器（BarrierIK），并在仿真与VR遥操作实验中验证其对安全性与任务性能的提升。

**💡 创新点**

创新点在于：①将CBF硬约束放置在混合命令后的IK层，实现对混合命令的安全投影；②提出BarrierIK方法，在保持姿态跟踪、关节平滑与自碰撞避免的同时，强制执行障碍物安全约束；③通过实验验证该安全滤波器既不降低任务表现，又提升用户感知安全与信任。

**🔧 技术方法**

技术实现包括：线性 SE(3) 混合策略（位置线性插值、姿态 SLERP）；使用 JAX 自动微分进行正向运动学、距离与梯度计算；SLSQP 优化求解逆运动学并加入 CBF 约束；VR 实验使用 HTC Vive、Unity3D 以及 PhysX；用户体验评估采用改良的 NASA‑TLX。

**📊 数据集**

数据集与实验：①仿真场景—两类障碍环境（Shelf/Frame 与 Dynamic Obstacles）；②VR 远程操作实验—10 名受试者完成 6 种配置（N、P、B、SA‑N、SA‑P、SA‑B），共 300 试验；未使用公开公开数据集。

**📈 对比分析**

与 Baseline N（无障碍约束）和 Baseline P（软碰撞惩罚）比较。评估指标包括碰撞次数、最小间隙、违规时间、姿态误差、任务/关节 jerk、成功率与完成时间。结果显示：在仿真中 BarrierIK 降低碰撞与违规时间，提升最小间隙；在 VR 实验中 SA‑B 在安全性、成功率、用户满意度（NASA‑TLX）上均优于其他配置，且保持了较低的碰撞次数。

**⚠️ 局限性**

局限性：①CBF 采用离散时间近似，可能导致短暂穿透；②缺乏障碍物速度估计，动态场景中安全集可能过保守；③硬约束导致路径弯转，可能增加关节 jerk；④类 K 函数与温度参数未自适应调节，需针对场景/用户手动设定；⑤在位置控制模式下使用 CBF，未考虑动力学约束；⑥实验仅在仿真与虚拟环境中验证，缺乏真实机器人硬件的碰撞与力学反馈验证。

---

## 85. MMNavAgent: Multi-Magnification WSI Navigation Agent for Clinically Consistent Whole-Slide Analysis

**arXiv ID:** 2603.02079 | [PDF](https://arxiv.org/pdf/2603.02079v1)

**作者:** Zhengyang Xu `[一作]` (Institute of Pathology), Peter J. Schüffler `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种多放大倍率WSI导航代理MMNavAgent，能够模拟病理路径学家在诊断过程中自适应地在不同放大倍率间切换，并通过交互式的热图生成选择诊断相关区域。

**💡 创新点**

创新点包括：①跨放大倍率导航工具（CMT）通过交叉注意力和邻层信息融合，实现多尺度特征的互补；②放大倍率选择工具（MST）基于记忆驱动的多步决策，实现与临床路径一致的动态放大倍率选择；③导航驱动的监督损失融合稀疏眼动标注与软损失，提升热图精度；④整体MST‑CMT闭环迭代架构。

**🔧 技术方法**

使用了视觉大模型（Qwen3-14B LLM、Patho-R1-7B VLM）、JWTH病理编码器、U‑Net解码器、跨尺度交叉注意力、记忆银行、软Dice/软Focal/ℓ1损失等技术。

**📊 数据集**

使用公开的Eye‑Tracking Dataset（918份皮肤WSI，四类），并在1.25×、2.5×、5×、10×等多放大倍率下生成导航热图。

**📈 对比分析**

与固定放大倍率基线、PEAN‑C、PathFinder等SOTA导航方法对比，MMNavAgent在AUC上提升1.45%（94.33%），BACC提升2.93%（79.71%），并在导航一致性（与眼动热图、肿瘤覆盖率）方面优于对手。消融实验表明CMT与MST组件均对性能贡献显著。

**⚠️ 局限性**

局限性包括：①对稀疏且主观的眼动标注依赖较大，监督仍不充分；②目前仅在皮肤WSI数据集验证，跨组织、跨疾病的推广性待评估；③记忆驱动决策与交互式推理带来计算开销；④尚需公开代码并在更多临床环境中进行验证。

---

## 86. Predictive Importance Sampling Based Coverage Verification for Multi-UAV Trajectory Planning

**arXiv ID:** 2603.01687 | [PDF](https://arxiv.org/pdf/2603.01687v1)

**作者:** Snehashish Ghosh `[一作]` (Indian Statistical Institute), Sasthi C. Ghosh `[通讯]` (Indian Statistical Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于预测重要性采样（PIS）的连续覆盖验证框架，用于多UAV轨迹规划与实时可靠低时延通信（URLLC）服务的覆盖管理。

**💡 创新点**

创新点在于：①将LSTM+混合密度网络（MDN）用于多模态用户轨迹预测并构建防御式混合采样；②证明PIS在保持无偏性的同时实现方差显著降低；③将PIS与多智能体深度确定性策略梯度（MADDPG）结合，实现自适应多目标奖励的协同轨迹优化。

**🔧 技术方法**

主要技术包括：LSTM‑MDN 轨迹预测、重要性采样与防御混合采样、MADDPG 训练与策略执行、可学习权重网络以动态平衡吞吐量、覆盖率、能耗与公平性。

**📊 数据集**

使用仿真数据生成的环境：1500 m×1500 m 区域，30 名URLLC 用户、200 名 eMBB 用户，5 架 UAV；采用 73 GHz mmWave 链路模型与真实建筑障碍物地图。

**📈 对比分析**

与三种基线（标准 MADDPG、SAC、MADDPG‑BO）以及 1000 采样的均匀验证进行比较。结果显示 PIS‑MADDPG 在吞吐量约 3128 Mbps、URLLC 覆盖率 74.1%、能耗 116.1 kJ、Jain 公平性 0.71 等指标均优于基线，且验证延迟仅 0.57 ms（相较于均匀采样的 11 ms）。

**⚠️ 局限性**

局限性包括：①需预先训练 LSTM‑MDN，预测误差对 α 参数敏感；②仅在二维地面移动场景验证，未覆盖空中移动用户；③仿真环境缺乏硬件真实干扰，未来需在实际 UAV 试验台验证。

---

## 87. SoberDSE: Sample-Efficient Design Space Exploration via Learning-Based Algorithm Selection

**arXiv ID:** 2603.00986 | [PDF](https://arxiv.org/pdf/2603.00986v1)

**作者:** Lei Xu `[一作]` (Shantou University), Chenglong Xiao `[通讯]` (Shantou University)

**通讯引用:** 334 | [OpenAlex ID](https://openalex.org/A5103014792)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SoberDSE 框架，利用图神经网络提取高层代码特征，并结合监督学习与强化学习实现 DSE 算法的自动选择，从而在 FPGA 加速器核设计空间中快速得到 Pareto 前沿。

**💡 创新点**

创新点在于：①将算法选择理念引入 DSE，缓解了 No‑Free‑Lunch 定理对单一算法的限制；②采用混合监督+强化学习（PPO）推荐模型，显著提升了样本效率与泛化能力；③使用 CDFG + ProGraML 构建 Benchmark Feature Graph，有效捕获控制与数据依赖。

**🔧 技术方法**

使用的技术包括：图神经网络（GNN）对 Benchmark Feature Graph 编码；ECoGNN 进行 QoR 预测；ProGraML 与 LLVM 生成中间表示；PPO 强化学习与交叉熵监督学习；传统 DSE 算法（NSGA‑II、SA、ACO、PSO、Lattice、HGBO‑DSE、MOEDA）与 RL 算法（IRONMAN‑PRO、QL‑MOEA）作为对比基线。

**📊 数据集**

数据集由 20 个训练基准（MachSuite、Polyhedral 等）和 9 个推理基准组成，设计空间规模从千级到百亿级不等，全部以 HLS 生成的 CDFG 形式构造。

**📈 对比分析**

通过 ADRS 指标对 SoberDSE 与 10 个基线（7 传统启发式 + 3 RL）进行比较，SoberDSE 在平均 ADRS 上比 Lattice、HGBO‑DSE、MOEDA 提升 81.32%、78.61%、83.24%；相较 RL 基线提升 76.31%、54.81%、75.29%；总体运行时间比大多数基线低 75% 以上。

**⚠️ 局限性**

局限性包括：受限于样本数量，算法推荐仍可能出现过拟合；目前算法库有限，无法覆盖所有 FPGA 设计模式；仅在特定基准集上验证，需进一步扩展数据集与算法范围以验证泛化能力。

---

## 88. EMPA: Evaluating Persona-Aligned Empathy as a Process

**arXiv ID:** 2603.00552 | [PDF](https://arxiv.org/pdf/2603.00552v1)

**作者:** Shiya Zhang `[一作]` (Nature Select), Xiaofan Zhang `[通讯]` (Nature Select)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EMPA 框架，用以在多轮心理支持对话中对 LLM 的同理心进行过程级轨迹评估。

**💡 创新点**

创新点是将同理心视为隐藏状态的方向性干预，并通过实时模拟、隐状态推断与能量门控的 EPM 实现可解释、可训练的轨迹评估。

**🔧 技术方法**

技术包括 Real-to-Sim 情境生成、基于 LLM 的用户模拟、双循环多代理控制、规范化评判器与向量方向投影的 EPM。

**📊 数据集**

使用从真实情感交互抽取的对话数据转化为模拟情境，并构建了覆盖 C/A/P 维度、约 1000+ 情境的 EMPA 基准。

**📈 对比分析**

在 14 款 LLM 上评估 EMPA‑Q 分数，发现 Claude 4.6、Gemini 3 等顶级模型在方向性、效率与稳定性上显著优于中低阶模型，验证了方法的区分性。

**⚠️ 局限性**

局限在于需要大量人工验证的情境生成、对用户真实隐状态推断的依赖，以及对更复杂人类情绪与行为的建模尚不完善。

---

## 89. Measuring What VLMs Don't Say: Validation Metrics Hide Clinical Terminology Erasure in Radiology Report Generation

**arXiv ID:** 2603.01625 | [PDF](https://arxiv.org/pdf/2603.01625v1)

**作者:** Aditya Parikh `[一作]` (Technical University of Denmark), Stella Frank `[通讯]` (Technical University of Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了胸片报告生成中视觉语言模型在解码时的词汇消失与偏差问题，并提出了诊断词汇关联位移(CAD)和加权关联消失(WAE)两种度量。

**💡 创新点**

创新点在于首次从词汇层面量化模型在不同解码策略下临床词汇的消失、产生新偏差和保留情况，揭示传统相似度指标掩盖的模板崩塌现象。

**🔧 技术方法**

使用Dirichlet平滑的词汇关联度量、统计显著性检验、加权方差聚合以及多种解码策略（贪婪、束搜索、温度采样等）进行评估。

**📊 数据集**

数据集为ReXGradient-160K胸片报告集和CheXpert，采用性别二分类平衡或偏斜子集进行实验。

**📈 对比分析**

与BERTScore、ROUGE等传统指标相比，贪婪解码虽然分数高，但CAD显示30+词汇被抹去；随机采样提高多样性但引入数十个新偏差；WAE量化了这些消失与偏差的总量，显示不同解码策略在临床准确性、多样性与公平性之间存在权衡。

**⚠️ 局限性**

局限在于仅针对性别二元组评估，未覆盖年龄、种族等交叉群体，且缺乏临床医生验证以确认CAD标记的消失是否真正临床重要。

---

## 90. DRIFT: Diffusion-based Rule-Inferred For Trajectories

**arXiv ID:** 2603.00936 | [PDF](https://arxiv.org/pdf/2603.00936v1)

**作者:** Jinyang Zhao `[一作]` (Hefei University of Technology), Shunyu Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 46 | [OpenAlex ID](https://openalex.org/A5087893108)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了DRIFT，一种条件扩散框架，用于在移动机器人无地图环境下生成同时满足几何精度和运动平滑的轨迹；

**💡 创新点**

通过“Architecture-as-Regularizer”理念，将关系性结构感知模块（SSP）和时间注意力模块（GTGRU）嵌入扩散过程，实现全局拓扑一致与局部终端精度的分离；

**🔧 技术方法**

采用条件扩散模型、图神经网络（EdgeConv）构建场景图、Graph-Conditioned Time-Aware GRU（GTGRU）实现动态注意力、稀疏交叉注意力以及基于课程学习的训练策略；

**📊 数据集**

使用基于Husky机器人3Hz VLP-16 LiDAR的户外导航数据集，采集点云与历史速度，并以A*规划轨迹作为标注；

**📈 对比分析**

与BC、CAVE、S2TNet、DTG等基线对比，DRIFT在终点误差0.041 m、Jerk 27.19 m/s³、推理成功率91.66%、碰撞率5.17%等指标上实现了最佳平衡，优于所有基线；

**⚠️ 局限性**

主要局限在于推理延迟相对较高（0.27 s），尚未与闭环控制器集成，且在极端动态或高拥挤场景下仍可能出现误差，未充分验证实时性与适应性。

---

## 91. MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms

**arXiv ID:** 2603.02184 | [PDF](https://arxiv.org/pdf/2603.02184v1)

**作者:** Jinqi Wu `[一作]` (Nanjing University), Chaoyou Fu `[通讯]` (Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了多归因学习（MAL）框架，用以提升转化率（CVR）预测模型的性能，并且首次公开了包含多归因标签的公共基准数据集 Multi‑Attribution BenChmark (MABench) 与开源实现库 PyMAL；

**💡 创新点**

创新点在于：①提供首个多归因标签的公开 CVR 数据集和可复现的实验平台；②系统归纳并验证了 MAL 的三大关键洞见；③设计并实现了融合 MoE 知识获取与主任务优先异构转移的 Mixture of Asymmetric Experts（MoAE）模型，显著优于现有最先进方法；

**🔧 技术方法**

采用多任务学习（MTL）与混合专家（MoE）结构、异构知识转移模块、辅助任务权重自适应（GCS）、梯度冲突消除（PCGrad）等技术，基于 PyTorch 实现；

**📊 数据集**

使用基于淘宝/阿里广告系统的真实业务日志，构成的 MABench 数据集，包含 0.8 万用户、7900 万点击、960 万物品，并为每条点击提供四种归因标签（last‑click、first‑click、linear、DDA）；

**📈 对比分析**

通过在 MABench 上对比单归因基线、传统 MTL 模型（Shared‑Bottom、MMoE、PLE、HoME）以及 NATAL 与 MoAE，评估 GAUC/AUC 指标，MoAE 在四种目标归因场景均实现 0.13–0.39 个百分点的 GAUC 提升；

**⚠️ 局限性**

局限性包括：对第一归因标签的噪声影响仍显显著；辅助任务的选择对性能影响较大，需针对业务场景精细调优；实验集中在单业务场景，跨域泛化及大规模分布式训练仍待进一步验证。

---

## 92. Learning Thermal-Aware Locomotion Policies for an Electrically-Actuated Quadruped Robot

**arXiv ID:** 2603.01631 | [PDF](https://arxiv.org/pdf/2603.01631v1)

**作者:** Letian Qian `[一作]` (Huazhong University of Science and Technology), Xin Luo `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种将全身热模型与控制屏障函数奖励整合进强化学习的热感知四足机器人步态策略，提升了机器人在负载下的持续运行时间。

**💡 创新点**

创新点在于：①将整个机器人的温度耦合模型实时嵌入仿真；②通过控制屏障函数设计温度相关奖励，实现主动热管理；③在无额外硬件冷却的前提下实现可持续行走。

**🔧 技术方法**

采用深度强化学习（PPO + 非对称Actor–Critic）、全身热传递模型、控制屏障函数、域随机化、Isaac Gym仿真与真实Unitree A1机器人实验等技术。

**📊 数据集**

使用内部生成的仿真环境数据（随机负载、外力、地形、温度等），并在真实机器人上采集传感器信息；未使用公开数据集。

**📈 对比分析**

通过与未加入温度奖励的基线策略比较；基线在约7分钟内因前左膝电机过热停止，而热感知策略在3 kg负载下连续行走超过27分钟，且保持所有电机温度低于阈值，命令跟踪性能保持一致。

**⚠️ 局限性**

局限性包括：策略趋于保守，导致在低温环境下仍保持较低行走姿态，限制了机器人在陡坡或楼梯等复杂地形上的机动性；未来需引入多模式识别实现温度自适应行为。

---

## 93. FATE: Closed-Loop Feasibility-Aware Task Generation with Active Repair for Physically Grounded Robotic Curricula

**arXiv ID:** 2603.01505 | [PDF](https://arxiv.org/pdf/2603.01505v1)

**作者:** Bingchuan Wei `[一作]` (Tsinghua University), Sen Cui `[通讯]` (Tsinghua University)

**通讯引用:** 1037 | [OpenAlex ID](https://openalex.org/A5075895196)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种闭环的、可行性感知的机器人任务生成框架FATE，能够在生成任务时自动验证并修正场景与动作策略的物理可行性，显著降低无效任务比例；

**💡 创新点**

创新点在于将可行性检验与任务生成融合为一个递归的审计-修复闭环，利用静态与动态两阶段感知审核、主动修复以及可塑性策略空间的动态调整，实现从语言描述到物理可执行任务的自适应投影；

**🔧 技术方法**

使用大语言模型（LLM）生成任务描述，集成多模态视觉语言模型（VLM）进行静态审核，结合MPC与SAC等控制器做动态审核，辅以主动修复模块对场景布局与策略参数进行迭代优化；

**📊 数据集**

实验基于NVIDIA Isaac Sim仿真平台，使用RidgebackFranka机器人、Objaverse和PartNet‑Mobility等公开资产库；

**📈 对比分析**

与专家设计基准（RLBench、ManiSkill2等）及开源的GenSim‑V2等基线对比，FATE在可行任务率（FTR）上从29.8%提升至92.1%，静态可行率和动态可行率分别从58.2%/51.2%提升至97.5%/94.5%；

**⚠️ 局限性**

局限性包括：VLM生成的不确定性导致仍有细粒度不一致；对精细动力学（摩擦、阻尼）建模不足，导致部分高接触任务仍出现失效；未在真实机器人上验证，需进一步校准物理参数和安全评估。

---

## 94. Spectral Attention Steering for Prompt Highlighting

**arXiv ID:** 2603.01281 | [PDF](https://arxiv.org/pdf/2603.01281v1)

**作者:** Weixian Waylon Li `[一作]` (University of Edinburgh), Shay B. Cohen `[通讯]` (University of Edinburgh)

**通讯引用:** 5702 | [OpenAlex ID](https://openalex.org/A5030503109)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SEKA及其自适应版本Adaptive SEKA，通过在注意力计算前直接编辑key向量的方式，实现训练无关的注意力调节。

**💡 创新点**

创新点在于利用谱分解学习“相关子空间”，以低秩投影放大关键token的注意力，并通过查询驱动的专家路由动态组合多任务专家投影，同时对KV头进行敏感性筛选。

**🔧 技术方法**

技术手段包括SVD谱分解、低秩投影编辑、基于查询的动态加权路由、以及与FlashAttention等高效注意力实现的兼容。

**📊 数据集**

实验使用了CounterFact、Bias in Bios、Lost-in-the-Middle等数据集，并在Qwen3和Gemma3等多种规模的LLM上评测。

**📈 对比分析**

与PASTA、SPA及无干预基线相比，SEKA在标准提示高亮和失真中间位置等基准上均取得SOTA成绩，同时仅产生约0.03 s的推理延迟和极小的内存开销。

**⚠️ 局限性**

局限性包括需预先生成并存储谱投影、对KV头阈值调参敏感、在对Markdown不敏感的模型上提升有限，以及对对比性提示生成的依赖。

---

## 95. Self-Anchoring Calibration Drift in Large Language Models: How Multi-Turn Conversations Reshape Model Confidence

**arXiv ID:** 2603.01239 | [PDF](https://arxiv.org/pdf/2603.01239v1)

**作者:** Harshavardhan `[一作]` `[通讯]` (Independent Researcher), Harshavardhan (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在多轮对话中自我锚定所导致的校准漂移（SACD）现象。

**💡 创新点**

创新点在于提出了SACD概念，并揭示其多模态表现（Claude 的信心抑制、GPT‑5.2 的信心提升、Gemini 的 ECE 停滞），以及对不同模型训练方式对该现象的影响进行比较。

**🔧 技术方法**

技术方法包括：三条件实验设计（单轮基线、多轮自锚定、独立重复），自报概率估计提取模型信心，使用 Expected Calibration Error (ECE) 量化校准误差，配合统计检验（t检验、F检验、U检验）分析效应。

**📊 数据集**

数据集为 150 个跨领域问题（事实、技术、开放式）按比例分配，每个模型在多轮条件下使用 15（或 30）个样本进行测试。

**📈 对比分析**

通过比较 Claude Sonnet 4.6、Gemini 3.1 Pro、GPT‑5.2 在三种条件下的 Confidence Drift Score (CDS) 和 ECE 变化，发现模型异质性显著：Claude 在多轮自锚定下信心下降且 ECE 上升，GPT‑5.2 信心略升但 ECE 增高，Gemini 在自锚定下 ECE 保持高位而非自然下降，表明 SACD 对不同模型的表现不一致。

**⚠️ 局限性**

局限性包括样本量有限（每模型仅 15 条问题）、信心测量依赖模型自报概率且可能不完全反映内部置信度、开放式问题真值的不确定性导致 ECE 评估噪声、未对各模型机制进行深入因果解析。

---

## 96. Non-verbal Real-time Human-AI Interaction in Constrained Robotic Environments

**arXiv ID:** 2603.01804 | [PDF](https://arxiv.org/pdf/2603.01804v1)

**作者:** Dragos Costea `[一作]`, Marius Leordeanu `[通讯]` (NORCE Norwegian Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个实时非语言人机交互框架，利用2D人体关键点同时实现动作预测与情感识别

**💡 创新点**

首次提出双任务轻量级模型框架，结合大规模合成数据预训练来提升性能，并系统评估了合成与真实数据之间的差距

**🔧 技术方法**

采用了MLP、LSTM、CNN‑LSTM和Transformer四种轻量级网络，配合中心化与尺度归一化、复合损失、MotionLCM预训练、MediaPipe关键点提取以及SOTA视频生成模型的评测

**📊 数据集**

使用了437条真实人类视频（COCO格式）做训练与测试，MotionLCM生成的9k/45k/90k样本做预训练，以及由SORA和VEO生成的合成测试集

**📈 对比分析**

通过对四种模型在真实测试集上的MAE和情感分类准确率进行对比，发现预训练后LSTM/CNN‑LSTM误差显著下降；在SORA/VEO测试中误差更大，VEO误差更低；所有模型在NVIDIA Orin Nano上均可达约100FPS

**⚠️ 局限性**

局限性包括仅处理2D关键点、情感类别有限、仅使用MotionLCM预训练、评估为离线实验，尚未验证真实人机交互中的用户体验与更广泛情感

---

## 97. LexChronos: An Agentic Framework for Structured Event Timeline Extraction in Indian Jurisprudence

**arXiv ID:** 2603.01651 | [PDF](https://arxiv.org/pdf/2603.01651v1)

**作者:** Anka Chandrahas Tummepalli `[一作]` (TCS Research), Preethu Rose Anish `[通讯]` (TCS Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LexChronos 框架，能从印度最高法院判决中迭代提取结构化事件时间线。

**💡 创新点**

创新点在于双代理迭代细化与生成合成数据集，解决缺乏印度司法事件标注问题。

**🔧 技术方法**

采用 LoRA 指令调优的 4B 级 LLM 作为提取代理，预训练 LLM 作为反馈代理，并使用零样本提示、角色/风格提示等技术。

**📊 数据集**

构建了 2000 篇印度最高法院判决的合成语料，包含事件时间线与对应判决文本。

**📈 对比分析**

通过 BERTScore 评估提取质量，最佳配置 F1=0.8751；在摘要任务中，结构化时间线输入被 GPT‑4 以 75% 的比例认为优于无结构文本。

**⚠️ 局限性**

局限在于数据为合成且仅覆盖 25 个案件类别，且仅使用英文，缺少真实案例验证与多语种适配。

---

## 98. DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles

**arXiv ID:** 2603.01111 | [PDF](https://arxiv.org/pdf/2603.01111v1)

**作者:** Yiming Ma `[一作]` (Chongqing Research Institute of Harbin Institute of Technology), Jianzhi Teng `[通讯]` (Department of Applied Mathematics, Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DeAR框架，在CLIP的深层Transformer中细粒度地插入属性提示并通过角色基注意力掩码控制信息流，实现任务适配与零射击泛化的平衡。

**💡 创新点**

创新点包括：1) 用概念熵自动划分注意力头为属性、泛化和混合三类；2) 角色基注意力掩码，隔离泛化头只让属性头与属性提示交互；3) 结合多模态属性提示与任务自适应融合，实现可控的精细调优；4) 引入自监督正则化和融合权重正则化提升泛化与专属知识兼顾。

**🔧 技术方法**

技术手段包括CLIP+ViT‑B‑16、可学习属性提示、Role‑Based Attention Mask、Concept Entropy、HDBSCAN聚类、SBERT语义嵌入、Self‑Regularization loss、Fusion‑Weight regularization、AdamW优化。

**📊 数据集**

评估使用11个图像分类基准（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、UCF101、DTD、EuroSAT）、四个ImageNet变体（V2、Sketch、A、R）以及few‑shot和cross‑dataset测试。

**📈 对比分析**

与CoOp、MMRL、Prompt‑SRC等PEFT方法对比，在base‑to‑novel泛化上调和均值达82.72%（比MMRL高1.83%），在领域泛化上取得最优或接近最优成绩，few‑shot和跨数据集评测均显著优于基线。

**⚠️ 局限性**

局限性：对属性集合的先验选择依赖经验；混合头仅采用全开策略，未进一步细化；实验仅在ViT‑B‑16上验证，未测试更大模型；对掩码超参数（β、mask策略）敏感，需进一步自动化调优。

---

## 99. Validation of Space Robotics in Underwater Environments via Disturbance Robustness Equivalency

**arXiv ID:** 2603.00628 | [PDF](https://arxiv.org/pdf/2603.00628v1)

**作者:** Joris Verhagen `[一作]` (KTH Royal Institute of Technology), Jana Tumova `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1854 | [OpenAlex ID](https://openalex.org/A5042698317)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于水下环境的空间机器人验证框架，利用中性浮力实验近似微重力，验证空间任务规划与控制的可行性。

**💡 创新点**

创新点在于：①将任务规范的扰动鲁棒度度量引入规划，使水下与空间平台获得相同的鲁棒性；②通过反馈等价控制实现水下机器人在闭环层面与空间机器人等价；③使用Signal Temporal Logic与混合整数规划实现全局最优规划。

**🔧 技术方法**

主要技术包括：Signal Temporal Logic（STL）任务规范、扰动鲁棒度优化、线性化空间机器人模型的混合整数规划、模型预测控制（MPC）、扩展卡尔曼滤波（EKF）实现在线扰动估计、以及反馈等价控制实现系统等价。

**📊 数据集**

实验数据来源于BlueROV2水下机器人、ATMOS物理平台（2D 空间机器人）以及Basilisk仿真中的CubeSat，三者共享相同的软件栈和控制架构。

**📈 对比分析**

通过将水下轨迹的时间尺度调整与空间轨迹匹配，并比较扰动估计是否落在预设鲁棒区间内，验证了两平台在扰动鲁棒度约为1.3~1.6时的性能一致；实验表明水下执行中的扰动始终位于鲁棒限值以内，说明验证方法有效。

**⚠️ 局限性**

局限性包括：①需要满足即时扰动检测与补偿的假设；②水下模型必须足够准确，误差须可视为可加扰动；③未考虑估计管线的不确定性与不同特性，且未给出可容忍误差的定量阈值。

---

## 100. Real-Time Thermal-Inertial Odometry on Embedded Hardware for High-Speed GPS-Denied Flight

**arXiv ID:** 2603.02114 | [PDF](https://arxiv.org/pdf/2603.02114v1)

**作者:** Austin Stone `[一作]` (Brigham Young University), Cammy Peterson `[通讯]` (Brigham Young University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发并实现了实时单目热红外‑惯性里程计系统，可在30 m/s的高速、GPS‑禁飞环境中闭环飞行。

**💡 创新点**

创新点包括：轻量化热优化前端与多阶段特征过滤；IIR预滤波并预积分高频IMU；利用GRU网络捕捉巴罗压强随速度变化的时序误差并补偿；将多传感器（热相机、IMU、激光测距、气压计、磁力计）融入固定滞后因子图；在嵌入式硬件上实现低延迟（≈42 ms）闭环控制。

**🔧 技术方法**

使用的核心技术有：FLIR Boson+ 640 LWIR相机、1200 Hz IMU预滤波+120 Hz下采样与预积分、激光测距深度先验、磁力计yaw先验、GRU神经网络巴罗补偿、ICE‑BA增量重线性、LMPCG非线性优化、EKF低延迟融合。

**📊 数据集**

数据集：自采集的无人机飞行日志，包含20–30 m/s高速、不同光照、烟雾、低温等环境；基准GPS地面真值；训练GRU的多速度、不同风速轨迹。

**📈 对比分析**

评估方法：RMSE、漂移率、闭环误差以及每个模块的消融实验；相较于基线（无预滤波、无GRU、无多阶段过滤），RMSE从19 m降低至12–14 m，漂移率从≈3 %降至≈1 %；GRU巴罗补偿比多项式/MLP平均降低≈1.3 m RMSE；闭环30 m/s箱形轨迹误差31.5 m（漂移1 %），总体平均延迟42 ms。

**⚠️ 局限性**

局限性：依赖热相机纹理对比度，极低纹理/均温场景仍难以跟踪；GRU模型需针对特定机型和风速训练；FPN dropout与热滞后仍可导致特征丢失；需要手动标定与同步；在极寒、雪、雨等极端温度环境下性能尚未充分验证。

---

## 101. IU: Imperceptible Universal Backdoor Attack

**arXiv ID:** 2603.00711 | [PDF](https://arxiv.org/pdf/2603.00711v1)

**作者:** Hsin Lin `[一作]` (National Yang Ming Chiao Tung University), Chia-Mu Yu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 1794 | [OpenAlex ID](https://openalex.org/A5085568918)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种不可察觉的通用后门攻击，利用图卷积网络生成类特定触发器，在极低poison率下实现多目标攻击；

**💡 创新点**

首次将GCN用于捕捉类间关系以生成隐蔽触发器，并提出触发器可分离性指数（TSI）理论，显著提升低poison率下的攻击成功率；

**🔧 技术方法**

使用图卷积网络、双目标（隐蔽性与攻击性）损失、PSNR约束、预训练ResNet特征提取和数据poisoning；

**📊 数据集**

主要使用ImageNet-1K数据集（CIFAR-10作为附录实验）；

**📈 对比分析**

与传统UVA-Blend对比，在0.16% poison率下ASR达72%（vs 0.4%），在更高poison率下相近；保留BA≈69.7，PSNR≥30；在Fine‑Tuning、Fine‑Pruning、NAD等后门移除和STRIP、SCALE‑UP、IBD‑PSC、BARBIE、MM‑BD等检测方法下均保持高ASR且难以被检测；

**⚠️ 局限性**

对Vision Transformer的迁移效果不佳，PSNR阈值提升会导致ASR下降，且依赖可用的预训练模型；

---

## 102. Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions

**arXiv ID:** 2603.00568 | [PDF](https://arxiv.org/pdf/2603.00568v1)

**作者:** Yunqing Liu `[一作]` (Hong Kong Polytechnic University), Wenqi Fan `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 4769 | [OpenAlex ID](https://openalex.org/A5043696243)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种双图双尺度交互框架 DeMol，用于分子表示学习，显式地同时建模原子层和键层，并通过双螺旋块实现跨尺度的原子-键-键交互。

**💡 创新点**

创新点在于①引入键中心图（bond‑centric graph）捕捉键级信息和键间相互作用；②双螺旋块实现原子与键的动态双向注意力融合；③利用共价半径预测和扭转角编码作为几何一致性约束；④结构感知掩码降低不必要的全连接开销。

**🔧 技术方法**

技术主要包括图神经网络（GNN）与自注意力（Transformer）结合的双通道编码、Gaussian 基函数的距离与角度编码、信息瓶颈理论指导的双图信息融合、以及基于共价半径的正则化与掩码策略。

**📊 数据集**

使用四大公开基准：PCQM4Mv2（HOMO‑LUMO 间隙预测）、OC20 IS2RE（催化剂吸附能预测）、QM9（12项量子化学性质）以及 MoleculeNet（八个二分类任务）。

**📈 对比分析**

与现有 GNN、Transformer 及 3D 等价网络对比，DeMol 在 PCQM4Mv2 上 MAE 0.0603 eV（比最佳 0.0671 eV 低 10.1%），在 OC20 IS2RE 上 MAE 0.3879 eV（相较 0.4088 eV 降低 5.1%），在 QM9 多项指标上排名前列，MoleculeNet 上平均 ROC‑AUC 79.96，七项任务均优于竞争模型。

**⚠️ 局限性**

限制主要体现在：① 对极小、结构简单的分子（如 QM9）提升有限；② 双图与双尺度交互导致计算开销和内存消耗较大；③ 仍需进一步验证在更大规模、异构体系（多分子或材料）中的泛化能力。

---

## 103. Emerging Human-like Strategies for Semantic Memory Foraging in Large Language Models

**arXiv ID:** 2603.01822 | [PDF](https://arxiv.org/pdf/2603.01822v1)

**作者:** Eric Lacosse `[一作]` (Champalimaud Research), Daniel C. McNamee `[通讯]` (Champalimaud Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文利用语义流任务（SFT）对大型语言模型（LLM）进行机理可解释性研究，探讨LLM在聚类与切换两种语义搜索行为中的内部层分布与残差流表现，并与人类生成序列进行对比。

**💡 创新点**

创新点在于将人类认知理论（如Marginal Value Theorem）与LLM内部机制相结合，首次揭示LLM在层级内部可辨识的收敛（聚类）与发散（切换）模式，并证明这些模式可用于调控模型行为，实现认知对齐或有意脱离。

**🔧 技术方法**

采用的技术包括LogitLens分析、残差流线性探针、PCA降维、输出概率分布对比、转移概率矩阵构建以及Spearman相关与AUROC评估。

**📊 数据集**

使用的数据集为699条由人类完成的动物命名序列（来自三组实验）与对应的LLama‑3系列模型（1B、3B、8B、70B）生成的699条序列，随后扩展至2285条LLM序列。

**📈 对比分析**

比较方法包括构建类别转移概率矩阵并计算Spearman相关（人类与LLM为0.701，显著），切换比例对比（人类平均0.55，LLM平均0.40），以及对输出分布与残差流进行线性探针，发现输出层AUROC≈0.75，残差流在对比度增强后可达0.96，均表明LLM行为与人类高度相似且可辨识。

**⚠️ 局限性**

限制在于仅针对动物类别进行实验，未考察不同提示或任务复杂度的影响，残差流辨识需通过对比度增强才能达到高准确率，且缺乏对生成速度、细粒度错误率等人类真实认知指标的评估。

---

## 104. Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation

**arXiv ID:** 2603.01999 | [PDF](https://arxiv.org/pdf/2603.01999v1)

**作者:** Jan Finke `[一作]` (Fraunhofer Institute for Material Flow and Logistics), Marvin Wiedemann `[通讯]` (Fraunhofer Institute for Material Flow and Logistics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出教师‑学生强化学习框架，将单目深度估计（MDE）替代 2D 激光雷达，实现工业场景下的移动机器人安全避障与导航。

**💡 创新点**

创新点在于：①利用 MDE 生成全高信息的深度图，彻底消除 2D LiDAR 对扫描平面的局限；②教师‑学生结构把利用 2D LiDAR 训练出的稳健策略迁移到仅使用 MDE 的学生；③在 NVIDIA Jetson Orin AGX 上实现完整的离线推理，满足工业嵌入式部署需求；④在模拟与真实环境中均显示出相较 2D LiDAR 更高的成功率。

**🔧 技术方法**

采用的技术包括：Proximal Policy Optimization (PPO) 在 Isaac Lab 训练教师；Depth Anything V2 微调做单目深度估计；行为克隆 (BC) 训练学生；多相机 RGB 4 通道堆叠输入；IMPALA CNN 编码器；域随机化与七阶段噪声增强；TensorRT 优化在 Jetson Orin AGX 上的推理。

**📊 数据集**

使用的数据集为：1) 24,499 张 RGB + 深度对（来自 Orbbec Femto Bolt ToF 相机），用于 MDE 微调；2) 在模拟中随机放置 15 种障碍物的 3 个不同大小的 arena，产生 28,838 条成功教师演示样本；3) 真实 8m×8m 实验场景中的 10 次试验数据。

**📈 对比分析**

在模拟中对比教师（标准与特权碰撞网格）与学生：学生在 5–25 个障碍物时的成功率为 82–96.5%，明显高于教师的 50–89%；在真实 8m×8m 场地的 3 个终点，学生平均成功率 80%，教师仅 37%。通过与 2D LiDAR 的对比，证明 MDE‑学生在复杂 3D 障碍环境中表现更好。

**⚠️ 局限性**

局限性：仅针对静态环境；MDE 在远距离仍存在尺度误差导致精度下降；缺乏与其他外部基线（如三维传感器）直接对比；未能处理动态障碍物；需要进一步的在线适应与动态障碍规避研究。

---

## 105. PreSight: Preoperative Outcome Prediction for Parkinson's Disease via Region-Prior Morphometry and Patient-Specific Weighting

**arXiv ID:** 2603.01948 | [PDF](https://arxiv.org/pdf/2603.01948v1)

**作者:** Yand Wang `[一作]` (Beijing Jiaotong University), Shuai Shao `[通讯]` (Suzhou Institute for Advanced Research, University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究预手术预测帕金森病患者深脑刺激手术的运动功能改善情况，提出 PreSight 模型融合 DBM 与临床信息并通过患者特定权重模块实现区域自适应。

**💡 创新点**

引入患者特定权重模块根据临床嵌入动态调节脑区重要性，将结构 MRI 的 DBM 与临床先验结合，提升预测准确率并保持良好校准。

**🔧 技术方法**

使用 DBM（变形基形态测量）、Harvard–Oxford 区域分区、患者特定权重模块（MLP+sigmoid 门控）、大型语言模型提示编码临床变量，以及 3D 视觉网络（S3D）进行融合与分类。

**📊 数据集**

两中心帕金森病 DBS 病例共 400 例（366 训练/验证 + 34 外部测试），包含预手术 T1 MRI、DBM 图像、结构临床变量及 6‑12 个月后 UPDRS‑III 评分。

**📈 对比分析**

与临床+XGBoost、Radiomics+ML、S3D 等基线在内部与外部测试比较，PreSight 在内部测试 ACC 88.89%、TPR 95.83%、FPR 16.67%；外部测试 ACC 85.29%、TPR 94.12%、FPR 23.53%，相较最强基线提升约 8‑9 点 ACC，保持高灵敏度低假阳性。

**⚠️ 局限性**

受限于样本量有限、仅使用单模态 T1 MRI，且对后置技术因素（植入位置、编程）不敏感；外部验证仍在相对相似中心，需进一步验证泛化性能。

---

## 106. Beyond the Grid: Layout-Informed Multi-Vector Retrieval with Parsed Visual Document Representations

**arXiv ID:** 2603.01666 | [PDF](https://arxiv.org/pdf/2603.01666v1)

**作者:** Yibo Yan `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1216 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种“Beyond the Grid”框架，利用文档解析模型生成少量布局感知的子图像嵌入，并与全局页面向量融合，得到压缩且结构化的多向量表示，用于视觉文档检索。

**💡 创新点**

创新点在于：①通过文档解析将页面分割为语义丰富的若干子图像，突破传统均匀网格划分的局限；②无训练、可插拔的全局‑局部加权融合策略，既保留细粒度信息又加入全局语境；③实现95%以上的存储压缩，同时在多种基线模型上显著提升检索性能。

**🔧 技术方法**

技术包括：MinerU2.5 文档解析器、单向量检索模型（如VLM2Vec、GME、UniME、B3等）作为编码器、基于 late‑interaction 的匹配机制、加权向量融合和信息瓶颈理论分析。

**📊 数据集**

在24个视觉文档检索基准上评估，涵盖 ViDoRe‑V1/V2、VisRAG、ViDoSeek、MMLongBench 等多样化数据集。

**📈 对比分析**

与十种单向量模型的原始表现、传统多向量优化方法（如Light‑ColPali、DocPruner、MetaEmbed 等）及多种基线（如单图像、token‑chunking 等）对比。实验显示，Beyond the Grid 在所有基准上平均提升 nDCG@5 超过 10 分，且在最佳模型（GME‑7B）上实现 80.61 分，压缩后仅保留约 5‑6 个向量，存储压缩率>95%。

**⚠️ 局限性**

局限性包括：①依赖解析模型的准确性，解析错误会影响子图像选择；②当前加权融合采用固定 α，未对不同文档动态调节；③对极端稀疏或无明显布局结构的文档可能效果有限；④在极大规模检索时仍需考虑在线检索的计算开销。

---

## 107. MVR: Multi-view Video Reward Shaping for Reinforcement Learning

**arXiv ID:** 2603.01694 | [PDF](https://arxiv.org/pdf/2603.01694v1)

**作者:** Lirui Luo `[一作]` (Peking University), Qing Li `[通讯]` (BIGAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多视角视频奖励塑形框架MVR，用来生成视觉反馈以改进强化学习代理的学习效果。

**💡 创新点**

创新点包括：①利用多视角视频和VLM的视图无关相似性学习状态相关性模型；②采用匹配对比（Bradley–Terry）与正则化结合的训练策略，解决语义鸿沟与视角偏差；③设计基于策略相关性的状态依赖奖励塑形，使VLM引导随学习进展自动衰减；④通过参考集合实现无监督的“最优行为”对齐。

**🔧 技术方法**

技术手段主要是：视觉语言模型（ViCLIP）进行视频–文本相似性计算；匹配对比损失、正则化损失训练状态相关性网络；基于TQC（及DreamerV3）等离线RL算法的奖励整合；多视角渲染、视频编码与相似性聚合。

**📊 数据集**

数据集：HumanoidBench（9个机器人行走、跑步、爬楼梯、滑行、站立、坐姿、平衡等）和MetaWorld（10个单物体操纵任务），全部采用模拟环境。

**📈 对比分析**

与基线对比：TQC、DreamerV3、VLM-RM、RoboCLIP。MVR在HumanoidBench上平均排名1.67（最优），在MetaWorld上平均排名1.50，显著高于所有基线；在大多数任务上实现或超过官方成功阈值，且提升幅度可达约10%。

**⚠️ 局限性**

局限性：①VLM训练在以人类行为为主的数据上，导致某些机器人任务（如Hurdle）学习进展受限；②在极长周期或复杂动力学任务中，多视角渲染与计算开销较大；③对VLM参数规模的依赖显著，较小模型性能下降；④缺乏对真实机器人环境的验证。

---

## 108. Semantic XPath: Structured Agentic Memory Access for Conversational AI

**arXiv ID:** 2603.01160 | [PDF](https://arxiv.org/pdf/2603.01160v1)

**作者:** Yifan Simon Liu `[一作]` (University of Toronto), Scott Sanner `[通讯]` (Vector Institute of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Semantic XPath结构化内存模块，支持在对话式AI中高效检索与更新树状记忆，并实现了SemanticXPath Chat演示系统。

**💡 创新点**

将XPath语言改造成具语义感知的查询语法，实现结构化检索与更新，显著提升检索准确率并减少令牌消耗。

**🔧 技术方法**

采用树结构记忆模型、语义相关性评分（余弦相似度与推理模型）、RAG与in-context对比、LLM后端（如GPT、Llama）。

**📊 数据集**

构造了三类对话式AI场景数据集：旅行行程、待办列表、膳食推荐，各包含20个单轮和5个多轮交互；使用GPT/LLama等LLM进行实验。

**📈 对比分析**

与in-context与flat RAG基线对比：单轮下Semantic XPath与in-context相当但token消耗低5倍；多轮下保持高通过率并维持稳定token；总体提升176.7%，token使用仅9.1%。

**⚠️ 局限性**

依赖LLM的偏差与事实错误，结构化检索仍不能保证绝对准确，未评估更大规模或跨域的鲁棒性。

---

## 109. Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic

**arXiv ID:** 2603.01162 | [PDF](https://arxiv.org/pdf/2603.01162v1)

**作者:** Hongyi Zhou `[一作]` (Tsinghua University), Chengchun Shi `[通讯]` (London School of Economics and Political Science)

**通讯引用:** 623 | [OpenAlex ID](https://openalex.org/A5025970743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对GRPO（Group Relative Policy Optimization）在LLM推理任务中的梯度估计、误差及收敛性进行了统一的有限样本与渐近理论分析，证明其梯度本质为U-统计量并给出了梯度MSE、子最优性间隙的界定及收敛速率；

**💡 创新点**

首次将GRPO与U-统计量联系起来，利用Hoeffding分解实现梯度方差分解，给出完整的误差上界和渐近分布；提出通用的分组规模缩放定律，证明GRPO在大样本下与oracle算法等价且在参数空间上实现最优性；

**🔧 技术方法**

使用U-统计量理论、Hoeffding分解、梯度方差分析、参数一致性与无偏性证明、PL（Polyak-Lojasiewicz）条件下的收敛分析、CLT（中心极限定理）等统计与优化工具；

**📊 数据集**

在数学推理基准数据集GSM8K与MATH上进行实验验证；

**📈 对比分析**

与Vanilla REINFORCE和oracle基线进行对比，GRPO梯度MSE显著下降，分组规模随理论预测而变，最终模型的测试准确率与oracle相近，验证了oracle属性与分组规模最优性；

**⚠️ 局限性**

理论中需估计缩放定律常数、实验计算量大、仅验证了分组规模与准确率的关系，未对不同奖励设置或更大模型的泛化进行深入探讨。

---

## 110. MoltGraph: A Longitudinal Temporal Graph Dataset of Moltbook for Coordinated-Agent Detection

**arXiv ID:** 2603.00646 | [PDF](https://arxiv.org/pdf/2603.00646v1)

**作者:** Kunal Mukherjee `[一作]` (Virginia Tech), Murat Kantarcioglu `[通讯]` (Virginia Tech)

**通讯引用:** 12220 | [OpenAlex ID](https://openalex.org/A5087192873)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个跨越数十天的多种实体（代理、子社群、帖子、评论）以及曝光快照的时序异构图数据集 MoltGraph，并用该数据集对Moltbook平台上协调行为及其对曝光的影响进行了系统测量与分析。

**💡 创新点**

创新点在于：①首次将曝光快照与交互事件统一到时序异构图中，实现了曝光感知的协调检测；②通过近同步共参与窗口定义协调事件，并从中提取时间、规模、跨社区传播等多维特征；③提出匹配比较框架，量化协调帖子相较非协调帖子在早期互动和曝光量上的提升。

**🔧 技术方法**

主要技术包括：时序异构图建模与生命周期字段设计；基于窗口的近同步共参与检测；图结构分析（幂律拟合、聚类、中心性集中度）；曝光指标（曝光计数、持续时间、跨社区溢出）计算；匹配对照组与提升率计算。

**📊 数据集**

使用公开抓取的Moltbook平台数据，构成了约{天}天、{代理}个代理、{子社群}个子社群、{帖子}个帖子、{评论}条评论以及{边}条时序边的 MoltGraph 数据集。

**📈 对比分析**

通过匹配相同子社群、相近创建时间的非协调帖子作为对照，计算协调帖子在前{H}小时内评论/点赞、曝光计数、持续曝光以及跨社区溢出等指标的提升率。实验显示协调帖子在早期互动上提升{X}%，在曝光计数上提升{Y}%，在跨社区曝光上提升{Z}%，表明协调行为能显著增强帖子可见度。

**⚠️ 局限性**

局限性包括：①仅研究单一平台Moltbook，结果对其他平台的泛化受限；②曝光测度仅基于抓取的快照，缺乏完整的曝光日志；③缺乏对行为背后意图的明确标注，无法区分恶意与善意协调；④窗口参数（Δ、k）和匹配策略可能影响结果，需进一步验证稳健性。

---

## 111. Multiple Inputs and Mixwd data for Alzheimer's Disease Classification Based on 3D Vision Transformer

**arXiv ID:** 2603.00545 | [PDF](https://arxiv.org/pdf/2603.00545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 112. Insights for an AI Whistleblower Office from 30 Case Studies

**arXiv ID:** 2603.01245 | [PDF](https://arxiv.org/pdf/2603.01245v1)

**作者:** Ethan Beri `[一作]` (University of Oxford), Mauricio Baker `[通讯]` (RAND)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过收集30个公开举报者案例，系统描述了AI行业举报者的动机、过程及后果，并提出了十条政策建议。

**💡 创新点**

首次用实证方法构建高维案例数据集，揭示举报者多为内部、道德动机主导且易遭受报复，进而为AI监管提供具体可操作的方案。

**🔧 技术方法**

主要采用数据收集、整理与定量描述统计（频数、比例）以及定性案例分析，没有使用机器学习或深度模型。

**📊 数据集**

30个公开案例的数据集，涵盖58个字段（包括举报者属性、组织信息、违规类型、动机、报复等），来源于政府公开名单、新闻报道和学术资料。

**📈 对比分析**

研究并未与其它方法进行实验性对比，主要以描述性统计和案例比对为手段；结果表明举报者多为内部、道德动机强、报复率高，为政策设计提供经验依据。

**⚠️ 局限性**

局限性：样本偏向高知名案例、仅包含成功举报案例、匿名案例缺乏、样本量仅30、外推性有限、以及无法完全消除报道与自报偏差。

---

## 113. Can Vision Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective

**arXiv ID:** 2603.01083 | [PDF](https://arxiv.org/pdf/2603.01083v1)

**作者:** Arctanx An `[一作]` (Peking University), Jiang Bian `[通讯]` (Microsoft Research Asia)

**通讯引用:** 13807 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了AesEval-Bench基准，涵盖四维度十二指标的设计美学评估，并对多种 VLM 进行系统评测与微调。

**💡 创新点**

创新点在于构建细粒度可量化的美学评测框架，结合指标驱动的定位任务和基于指标的推理路径训练，显著提升模型性能。

**🔧 技术方法**

采用多模态 VLM（如 Qwen、GPT、LLaVA 等）、人类引导 VLM 标注、指标根据信息的推理路径以及全参数微调等技术。

**📊 数据集**

使用 Crello 图形设计数据集生成的约 4.5k 测试样本和 3 万条训练样本进行评估与微调。

**📈 对比分析**

在美学判断、区域选择和精确定位三任务中，GPT-5 等模型在美学判断上获得最高准确率（约 0.73），但整体仍明显落后于专业图像美学模型；微调后模型在所有任务上提升约 5–17%。

**⚠️ 局限性**

局限性包括仅基于 Crello 数据，未覆盖信息图或移动 UI，指标分类尚未完全解耦，缺少创意等主观维度，且未尝试强化学习提升推理能力。

---

## 114. Suffix-Constrained Greedy Search Algorithms for Causal Language Models

**arXiv ID:** 2603.01243 | [PDF](https://arxiv.org/pdf/2603.01243v1)

**作者:** Ayoub Hammal `[一作]`, Caio Corro `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在大语言模型中先自由推理后强制生成符合后缀语法约束的“后缀约束生成”方法。

**💡 创新点**

创新点在于：①阐明后缀约束生成的理论难点并给出基于贪心搜索的两种高效算法；②提出分支惩罚（Bifurcation Penalty）作为判定何时切换到约束部分的准则。

**🔧 技术方法**

主要技术包括：贪心搜索、束宽为两的并行搜索、Earley算法求约束词汇、对数/概率差分惩罚来驱动假设切换。

**📊 数据集**

使用了五个问答数据集：数学推理集（GSM8K、MATH500、SVAMP）和两类多项选择常识集（ARC‑Challenge、CommonsenseQA）。

**📈 对比分析**

与无约束生成、完整约束生成以及“仅约束”模式对比，采用 exact‑match 评估；Bifurcation‑Penalty + “最后假设”策略在大多数任务上显著提升了准确率，尤其在 IT 模型上提升显著；其他算法在某些任务表现不稳定。

**⚠️ 局限性**

局限性：对低能力或训练不足的模型，束搜索仍易出现中断或超出预算；分支惩罚对复杂上下文的适应性有限；方法在处理高度嵌套或极长推理链时仍需改进。

---

## 115. How RL Unlocks the Aha Moment in Geometric Interleaved Reasoning

**arXiv ID:** 2603.01070 | [PDF](https://arxiv.org/pdf/2603.01070v1)

**作者:** Xiangxiang Zhang `[一作]` (ByteDance), Jingxuan Wei `[通讯]` (Shenyang Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究几何问题中的图形绘制与推理交互，提出强化学习框架Faire实现功能对齐；

**💡 创新点**

发现SFT在交替绘图推理时会退化，提出三视角验证器（几何一致性、感知可视性、语义一致性）和RL奖励来强化因果依赖；

**🔧 技术方法**

采用GRPO强化学习、三视角验证与可执行GeoGebra脚本，构建可验证的图形推理过程；

**📊 数据集**

构建Faire-Bench基准，约8,000条可执行几何推理轨迹与图形脚本；

**📈 对比分析**

与多款开源和专有多模态模型对比，Faire在验证得分和答案准确率上显著提升，超越SFT和文本基线，取得最佳结构与关系一致性；

**⚠️ 局限性**

依赖可执行图形工具与奖励设计，且在非几何领域的推广尚有限。

---

## 116. OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution

**arXiv ID:** 2603.02134 | [PDF](https://arxiv.org/pdf/2603.02134v1)

**作者:** Chong Xia `[一作]` (Tsinghua University), Yueqi Duan `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

实现了基于在线高斯投影的实时3D重建与语义理解；

**💡 创新点**

提出活跃‑稳定状态演化范式，将局部高频细节与全局一致性分离，并联合视觉与语言场景表征，显著降低漂移；

**🔧 技术方法**

采用ViT编码器+双解码器、DPT式头、隐式高斯融合及低维CLIP语义特征等技术；

**📊 数据集**

在RealEstate10k、ScanNet以及DL3DV零样本场景上进行训练与评估；

**📈 对比分析**

与离线3DGS基线（MVSplat、NoPoSplat、FLARE）及在线点云方法（Spann3R、CUT3R）比较，在线新视合成PSNR、SSIM、LPIPS均优于基线，语义分割mIoU与mAcc也显著提升，且实现约23fps实时帧率；

**⚠️ 局限性**

在极端视角变化或快速运动下仍可能出现漂移，且对大规模场景的内存占用和长期序列的累计误差尚需进一步优化。

---

## 117. Systematic Survey on Privacy-Preserving Architectures for IoT and Vehicular Data Sharing: Techniques, Challenges, and Future Directions

**arXiv ID:** 2603.01876 | [PDF](https://arxiv.org/pdf/2603.01876v1)

**作者:** Phat T. Tran-Truong `[一作]` (Ho Chi Minh City University of Technology), Triet M. Nguyen `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2007-2025年关于IoT与车联网隐私保护架构的75篇技术论文进行系统综述，构建三维分类法，分析研究趋势与挑战。

**💡 创新点**

首次将隐私-效率-信任三难困境纳入系统评估，验证单一范式无法同时满足三者，提出混合架构的研究路径。

**🔧 技术方法**

采用系统性文献检索（PRISMA）、三维架构分类（去中心化计算、加密技术、分布式账本）以及威胁映射与成熟度评估。

**📊 数据集**

涵盖多种数据集与应用场景，主要聚焦IoT与车联网数据；对比实验基于公开数据集如CIFAR-10、MNIST、MIMIC‑III等。

**📈 对比分析**

通过定量统计（论文分布、时序趋势）与质性比较（安全性、效率、可扩展性），发现去中心化计算主导、加密方案效率低、账本可扩展性差；混合架构在保持高精度（≈93%）的同时通信降低30%。

**⚠️ 局限性**

局限性在于缺乏实际部署案例，未对跨层协同成本进行量化，且对量子安全与后量子加密集成仍处于初级阶段。

---

## 118. Transform-Invariant Generative Ray Path Sampling for Efficient Radio Propagation Modeling

**arXiv ID:** 2603.01655 | [PDF](https://arxiv.org/pdf/2603.01655v1)

**作者:** Jérome Eertmans `[一作]` (Université catholique de Louvain), Claude Oestges `[通讯]` (Université catholique de Louvain)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于生成式流网络（GFlowNet）的机器学习辅助射线路径采样框架，用来替代传统的穷举搜索；

**💡 创新点**

创新点包括：1）将GFlowNet引入射线跟踪领域；2）通过经验回放、均匀探索策略和基于物理的动作屏蔽三大机制解决稀疏奖励导致的收敛和退化问题；3）实现了在GPU/CPU上显著加速的同时保持高路径覆盖率；

**🔧 技术方法**

主要技术：生成式流网络（GFlowNet）、经验回放缓冲区、均匀探索策略、物理约束动作屏蔽；

**📊 数据集**

使用的实验数据集未在摘要中给出，推测为仿真或真实射线跟踪环境的测试场景；

**📈 对比分析**

与传统穷举搜索比较，GPU加速约10倍，CPU加速约1000倍；在保持高覆盖率的同时成功发现复杂传播路径；

**⚠️ 局限性**

局限性：稀疏奖励环境下仍可能出现收敛不稳；对极高阶交互的有效性尚未充分验证；需要精确的物理约束来屏蔽无效路径，设计复杂度较高。

---

## 119. MigMate: A VS Code Extension for LLM-based Library Migration of Python Projects

**arXiv ID:** 2603.01596 | [PDF](https://arxiv.org/pdf/2603.01596v1)

**作者:** Matthias Kebede `[一作]` (New York University), Sarah Nadi `[通讯]` (New York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 MigMate VS Code 插件，将 MigrateLib 的 LLM 自动库迁移功能集成到 IDE，提供交互式预览与确认，支持半自动化 Python 库迁移。

**💡 创新点**

将 LLM 自动迁移与 IDE 交互式预览相结合，形成人机协作的迁移流程，解决传统工具在迁移准确性和用户信任方面的缺陷。

**🔧 技术方法**

使用大型语言模型（GPT‑4o mini、Llama 3.1）进行代码迁移；结合单元测试校验迁移结果；在 VS Code 中实现 Webview、Refactor Preview 等交互 UI。

**📊 数据集**

以 PyMigBench 作为基准评估库迁移；实验中使用自制 Python 项目，包含 requests→httpx 和 tablib→pandas 两对迁移任务。

**📈 对比分析**

通过小规模用户研究比较手动迁移与插件辅助迁移：平均迁移时间从 25–28 分钟缩短至 10–11 分钟（约 60% 降低），SUS 得分 80.9（第 90 百分位），表明可用性显著提升。

**⚠️ 局限性**

局限性包括样本规模有限、仅评估一次迁移任务、对测试失败情形和插件配置使用不足，需进一步大规模实验验证并完善功能。

---

## 120. Real-Time 3D Simulation of Heat-Induced Air Turbulence

**arXiv ID:** 2603.02048 | [PDF](https://arxiv.org/pdf/2603.02048v1)

**作者:** Wanqi Yuan `[一作]` (Clemson University), Nianyi Li `[通讯]` (Clemson University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个实时的三维热诱导气流湍流模拟系统，结合可压缩SPH热流体动力学与曲线光线追踪，实现多视角一致的热雾效应；

**💡 创新点**

首次实现从热传输到流动再到折射的完整物理链路，并引入基于局部折射梯度的自适应步长曲线光线追踪；

**🔧 技术方法**

使用可压缩SPH求解热传递、浮力与压力驱动的流体动力学，利用Gladstone–Dale定律生成折射率场，采用显式Euler步进的曲线光线追踪并配合自适应步长；

**📊 数据集**

采集了实验室热源（烤盘）生成的真实热雾视频与棋盘测试场景，用于验证深度相关失真与多视角一致性；

**📈 对比分析**

与多款基于屏幕空间的湍流模拟器（DAAT‑Sim、QuickTurb、TurbSimP2S、ATsyn）进行对比，平均帧率约35–40 FPS（RTX 4090）且在多视角一致性指标上显著优于基线；

**⚠️ 局限性**

模型采用简化热力学与折射关系，适用于大气层级热雾但在极高湍流强度、长视程或非空气介质下的精度有限，需要进一步完善物理模型与更高分辨率支持。

---

## 121. LFPO: Likelihood-Free Policy Optimization for Masked Diffusion Models

**arXiv ID:** 2603.01563 | [PDF](https://arxiv.org/pdf/2603.01563v1)

**作者:** Chenxing Wei `[一作]` (Shenzhen University), Bo Jiang `[通讯]` (Bytedance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需似然的策略优化框架，专门用于掩码扩散大型语言模型（dLLM），实现对模型生成轨迹的对比式速度修正。

**💡 创新点**

创新点在于将连续流匹配（Flow Matching）理论映射到离散token空间，直接在logit空间对向量场进行正负对比修正，消除了对难以计算的似然估计的依赖，并通过一致性训练把生成轨迹直线化。

**🔧 技术方法**

使用的核心技术包括：离散向量场匹配、对比速度正则化、分层轨迹采样、块级梯度累积、指数移动平均（EMA）等。

**📊 数据集**

实验数据集包括：代码生成任务 AceCode-87K 训练集与 HumanEval、MBPP、EvalPlus、BigCodeBench；推理任务使用 LLaDA 8B、GSM8K、MATH、Hellaswag、GPQA、WinoGrande、PIQA。

**📈 对比分析**

与现有 RL 基线（Diffu-GRPO、UniGRPO、SPG、AGRPO、Coupled-GRPO）比较，LFPOL 在代码生成平均准确率提升约3–4%，在推理任务平均准确率提升约9–10%，且在推理步骤上平均减少约20%（约41步/159步）。

**⚠️ 局限性**

局限性包括：仍需手工设计奖励函数；对大型词表和极端稀疏奖励的收敛性未充分验证；在更大规模或多模态任务中的通用性待进一步实验。

---

## 122. Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards

**arXiv ID:** 2603.02008 | [PDF](https://arxiv.org/pdf/2603.02008v1)

**作者:** Faisal Mohamed `[一作]` (Mila Quebec AI Institute), Glen Berseth `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种利用时间对比学习的内在奖励机制C-TeC，鼓励智能体探索未来不可预测的状态。

**💡 创新点**

创新点在于不使用episodic memory或quasi‑metric网络，而是直接通过时间对比学习估计未来状态占用度，生成前瞻性、模式寻求的探索信号。

**🔧 技术方法**

核心技术包括InfoNCE对比学习、状态‑动作编码器ϕ和未来状态编码器ψ、负L1或L2相似度作为critic，以及PPO/SAC的策略优化。

**📊 数据集**

实验使用JaxGCRL的连续控制和像素任务环境：ant_maze、humanoid_maze、object_manipulation（Ant/ humanoid）以及Craftax‑Classic。

**📈 对比分析**

与RND、ICM、APT、E3B和ETD等基线比较，C-TeC在迷宫覆盖率和Craftax任务中取得与ETD相当或更优的表现，尤其在Craftax中显著超越ETD；在机器人操控任务中也优于大多数对比方法。

**⚠️ 局限性**

局限性包括：目前仅验证于低维状态空间；对像素观测或部分可观测环境支持不足；对超参数和表示学习质量较为敏感；缺乏对长期动态不确定性和灾难性遗忘的深入分析。

---

## 123. pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning

**arXiv ID:** 2603.00905 | [PDF](https://arxiv.org/pdf/2603.00905v1)

**作者:** Zhanpeng Luo `[一作]` (Carnegie Mellon University), Yaqi Xie `[通讯]` (Carnegie Mellon University)

**通讯引用:** 404 | [OpenAlex ID](https://openalex.org/A5032012609)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个零样本视觉编程框架pySpatial，利用多模态大语言模型生成Python程序来调用3D重建、相机姿态、视角合成等空间工具，使模型能够在已构建的三维场景中显式推理。

**💡 创新点**

核心创新是将LLM的代码生成能力与可组合的空间工具相结合，形成可解释、可扩展的视觉程序，既不需要梯度微调，也不依赖预训练的专用3D网络，完全零样本即可提升多视角空间推理性能。

**🔧 技术方法**

使用的技术包括：多模态大语言模型（如GPT‑4o、GPT‑4.1‑mini）、Python代码生成与执行、前向3D重建模型（CUT3R、VGGT等）、Open3D渲染、定义好的空间工具API、以及视觉编程框架。

**📊 数据集**

在多视角空间推理基准MindCube（包含MindCube-1k子集）和单视角空间推理基准SpatialBench上进行评估，并在真实室内导航实验中使用Unitree‑Go1四足机器人验证实际效果。

**📈 对比分析**

与开源多图像LLM、专有LLM、专门的空间模型以及以往视觉编程方法（ViperGPT、VADAR等）进行对比。结果显示：在MindCube上整体准确率58.56%，比GPT‑4.1‑mini提升12.94%；在MindCube‑1k上整体62.35%，比VADAR提升21.9%；在SpatialBench上整体得分领先VADAR 3.8%和ViperGPT 17.5%；在机器人导航实验中成功避障并到达目标，显著优于GPT‑4.1基线。

**⚠️ 局限性**

主要局限包括：约39%失败案例中有6%是程序生成错误，20%归因于LLM最终推理，13%来自3D重建误差；对前向重建模型的依赖导致在视角稀疏时精度下降；代码生成与执行时间仍有提升空间；总体上需要更鲁棒的重建与更精细的空间工具以进一步提升性能。

---

## 124. FWeb3: A Practical Incentive-Aware Federated Learning Framework

**arXiv ID:** 2603.00666 | [PDF](https://arxiv.org/pdf/2603.00666v1)

**作者:** Peishen Yan `[一作]` (Shanghai Jiao Tong University), Haibing Guan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6370 | [OpenAlex ID](https://openalex.org/A5049487451)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个 Web3 生态下的激励兼容联邦学习框架 FWeb3，支持在开放参与环境下的可审计奖励分配。

**💡 创新点**

主要创新包括模块化架构将 FL 与 Web3 支持层解耦、以链外训练+链上结算、混合通信（WebRTC+IPFS）与轻量加密，以及浏览器原生 DApp 交互，显著降低部署与使用门槛。

**🔧 技术方法**

使用了以太坊（Sepolia）智能合约、IPFS、WebRTC、ECDH 与对称密钥、TypeScript+WebGL/WASM 浏览器运行时，以及插件化聚合与贡献评估接口。

**📊 数据集**

采用 CIFAR-10 数据集，训练 ResNet‑18 模型，分割为 IID 子集。

**📈 对比分析**

与 VeryFL、OpenFL 在部署复杂度、用户上手时间以及链上/链下开销进行对比，实验表明 FWeb3 的非训练开销仅为 WAN 21.3% 的交易延迟与 3.4% 的数据传输，单轮平均耗时 152.81 s，Gas 费用约 0.58 USD。

**⚠️ 局限性**

主要限制在于默认的“owner‑executed”模式需信任聚合者，对抗性攻击与隐私保护尚未充分覆盖；委员会模式虽提高安全性但引入额外通信成本；在大规模跨国部署时仍受网络时延与链上确认延迟影响。

---

## 125. LaST-VLA: Thinking in Latent Spatio-Temporal Space for Vision-Language-Action in Autonomous Driving

**arXiv ID:** 2603.01928 | [PDF](https://arxiv.org/pdf/2603.01928v1)

**作者:** Yuechen Luo `[一作]` (Tsinghua University), Fuxi Wen `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LaST‑VLA 框架，通过在连续的潜在时空空间中进行思考，统一视觉、语言与动作的推理与规划。

**💡 创新点**

核心创新包括：① 在潜在空间中引入双路适配器（几何和动态）对齐 3D 及视频基础模型的物理先验；② 采用分阶段自监督微调（先对齐物理特征后再学习动作），并通过 GRPO 强化学习进一步提升安全与合规性；③ 通过结构化因果掩码迫使规划器完全依赖潜在思考，从而消除视觉与符号间的语义鸿沟。

**🔧 技术方法**

使用的技术包括：InternVL3 视觉语言模型、Cosmos 运动世界模型、VGGT 3D 基础模型、双路适配器（Φ_geo 与 Φ_dyn）、分阶段自监督微调（SFT）、结构化因果掩码、Group Relative Policy Optimization (GRPO) 强化学习。

**📊 数据集**

训练与评估数据集：NAVSIM v1 & v2（规划基准）、SURDS（3D 语义推理）、NuDynamics（动态场景推理）。

**📈 对比分析**

与多种 SOTA 方法对比，LaST‑VLA 在 NAVSIM v1 取得 91.3 PDMS（新纪录），NAVSIM v2 87.1 EPDMS；在 SURDS 上显著提升几何推理指标，NuDynamics 上获得 81.19% 的动态推理准确率；在多项安全、合规、时间到碰撞等指标上也均超过竞争方法。

**⚠️ 局限性**

局限性：① 对 3D 与视频基础模型的依赖导致训练成本高；② 仍主要在仿真环境验证，真实路况下的泛化需进一步测试；③ 结构化掩码与双路适配器的设计仍需针对更复杂场景进行调优。

---

## 126. Flow Matching-enabled Test-Time Refinement for Unsupervised Cardiac MR Registration

**arXiv ID:** 2603.01073 | [PDF](https://arxiv.org/pdf/2603.01073v1)

**作者:** Yunguan Fu `[一作]` (University College London), Yipeng Hu `[通讯]` (University College London)

**通讯引用:** 5234 | [OpenAlex ID](https://openalex.org/A5032309114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于流匹配的多步心脏MR图像配准框架 FlowReg，能够在两步内达到甚至超过单步模型的性能，并支持进一步的细化。

**💡 创新点**

创新点包括：①使用warmup‑reflow训练策略，从零开始学习，不依赖预训练模型；②Initial Guess策略，将第一步预测直接作为后续步骤的起点；③结合 Heun 公式、SDE 噪声注入与指导损失，实现高效且精细的多步配准。

**🔧 技术方法**

技术手段：流匹配（Flow Matching）、SDE 与 ODE 转换、Heun 积分、指导损失（image similarity）、EMA 教师-学生机制、时间步嵌入、噪声注入与残差校正。

**📊 数据集**

使用 ACDC 与 MM2 两个公开心脏MR数据集（含 ED、ES 帧、LV/Myo/RV 标注），在六个任务（同域、跨域、ED→ES、ES→ED）上进行评估。

**📈 对比分析**

与 CorrMLP、FSDiffReg 等基线比较，FlowReg 在五个任务上显著提升平均 Dice 分数（+0.6%）并减少 LVEF 估计误差（-2.58%），同时仅增加约 0.7% 参数，推理时间仅在 0.05–1.3 秒之间。

**⚠️ 局限性**

局限性：①单步 FlowReg 仍弱于 CorrMLP，需进一步提升单步质量；②随着步数增加，变形不规则性略升高；③尚未引入差分同胚约束；④目前未验证对其他下游任务（如应变量化）的效果。

---

## 127. HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents

**arXiv ID:** 2603.00977 | [PDF](https://arxiv.org/pdf/2603.00977v1)

**作者:** Hongbo Jin `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**通讯引用:** 14136 | [OpenAlex ID](https://openalex.org/A5100447673)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 HiMAC 框架，将长时序任务拆分为宏层蓝图生成与微层执行，解决 LLM 代理的上下文漂移和错误累积问题。

**💡 创新点**

创新点在于：① 将决策分层，显著降低探索空间；② 引入无 critic 的层次化组相对优势优化；③ 采用交替迭代共进训练以消除非平稳性。

**🔧 技术方法**

使用技术包括：组相对优势优化（GRPO）扩展至双层；批量对比组；迭代共进训练；<sub_done> 终止标记；以及 Qwen2.5‑Instruct/VL 等 LLM。

**📊 数据集**

实验数据集包括：ALFWorld、WebShop、Sokoban。

**📈 对比分析**

与 ReAct、Reflexion、PPO、RLOO、GRPO、GiGPO 等基线对比，HiMAC 在 ALFWorld、WebShop、Sokoban 的成功率分别提升至 92.1%、84.1% 和 87.5%，比 GiGPO 提升 3.8%、16% 与 4.7%，并在样本效率上显著更快。

**⚠️ 局限性**

局限性在于：仍依赖较大 LLM 规模；训练需要多轮迭代；对更开放或极长序列环境的鲁棒性待验证；蓝图生成受语言表达限制，跨域迁移能力尚未评估。

---

## 128. MemPO: Self-Memory Policy Optimization for Long-Horizon Agents

**arXiv ID:** 2603.00680 | [PDF](https://arxiv.org/pdf/2603.00680v1)

**作者:** Ruoran Li `[一作]` (Tsinghua University), Jinli Suo `[通讯]` (Tsinghua University)

**通讯引用:** 4136 | [OpenAlex ID](https://openalex.org/A5051445938)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出自记忆策略优化算法MemPO，使LLM代理能够主动压缩与管理记忆，实现长时延交互中上下文有效缩减。

**💡 创新点**

创新点在于：①将记忆生成视为代理行为，联合优化记忆、推理和工具调用；②在优势函数中加入基于记忆条件概率的奖励，解决传统RL中记忆稀疏反馈的问题；③采用基于组的相对优势（GRPO）与记忆奖励相结合，提升训练稳定性。

**🔧 技术方法**

技术包括：强化学习（GRPO）、自回归语言模型（Qwen2.5 7B）、记忆压缩与条件概率奖励设计、组级优势归一化、KL正则化、工具调用框架（ReAct）。

**📊 数据集**

使用多目标任务数据集：HotpotQA、NQ（合成2/4/6/8/10目标任务），并在本地Wiki搜索与网络搜索环境下进行评估。

**📈 对比分析**

与ReAct、DeepResearcher、ReSearch、A-MEM、MEM1、GRPO等基线比较，MemPO在所有长时延基准上实现F1绝对提升25.98%（比基线）和7.1%（比SOTA），并将token使用量分别降低67.58%和73.12%，在保持或提升任务精度的同时显著提升资源利用率。

**⚠️ 局限性**

局限性：不同步骤的工具调用导致记忆状态差异，组级优势计算可能带来偏差；目前的记忆奖励设计主要针对单任务，通用性与更复杂环境中的适应性仍待进一步验证。

---

## 129. LiveCultureBench: a Multi-Agent, Multi-Cultural Benchmark for Large Language Models in Dynamic Social Simulations

**arXiv ID:** 2603.01952 | [PDF](https://arxiv.org/pdf/2603.01952v1)

**作者:** Viet-Thanh Pham `[一作]` (Monash University), Dinh Phung `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在模拟小镇中，将LLM嵌入为目标代理，评估其完成任务与遵守多元文化规范的能力；

**💡 创新点**

提出动态多文化基准LiveCultureBench，结合可观测的社会规范与LLM作为判定者的风险评估；

**🔧 技术方法**

采用LLM推理、对话生成、Conformal Language Modeling以及基于图的环境模拟；

**📊 数据集**

使用基于澳大利亚墨尔本人口普查的合成人口样本、CultureBank文化规范、以及多款LLM模型；

**📈 对比分析**

通过对比不同LLM家族（Gemini, Qwen, Llama, Ministral）在任务完成率、规范遵从率与验证器准确率的差异，发现LLM在多文化情境下易出现规范违例且任务完成优先；

**⚠️ 局限性**

受限于文化规范数据不足、验证器可靠性不足、以及高计算成本，且模拟仅基于合成人口与静态规范。

---

## 130. KERV: Kinematic-Rectified Speculative Decoding for Embodied VLA Models

**arXiv ID:** 2603.01581 | [PDF](https://arxiv.org/pdf/2603.01581v1)

**作者:** Zihao Zheng `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**通讯引用:** 36508 | [OpenAlex ID](https://openalex.org/A5100641667)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了KERV框架，利用卡尔曼滤波在视觉-语言-动作（VLA）模型的推理中纠正speculative decoding（SD）产生的错误，并通过基于运动学的阈值动态调节实现更高的推理速度；

**💡 创新点**

创新点在于将运动学域预测与token域解码相结合，使用卡尔曼滤波器代替昂贵的重推理，并以运动学误差量化自动调节SD接受阈值；

**🔧 技术方法**

技术包括VLA模型（OpenVLA+LLaMA草稿模型）、speculative decoding、卡尔曼滤波器、CPU/GPU协同实现的高效推理流程；

**📊 数据集**

使用LIBERO基准（包括LIBERO-Object、Spatial、Goal、Long四个子套件）进行实验；

**📈 对比分析**

与Naive VLA+SD和Spec‑VLA基线相比，KERV在保持1–2%成功率的同时实现了27%–37%的速度提升（1.48–1.57倍加速）；

**⚠️ 局限性**

局限性在于依赖VLA模型的可泛化性，未在真实机器人平台验证，且仅适用于已训练好的VLA架构，无法直接推广到新模型或不同硬件。

---

## 131. VectorMaton: Efficient Vector Search with Pattern Constraints via an Enhanced Suffix Automaton

**arXiv ID:** 2603.01525 | [PDF](https://arxiv.org/pdf/2603.01525v1)

**作者:** Haoxuan Xie `[一作]` (Nanyang Technological University), Siqiang Luo `[通讯]` (Nanyang Technological University)

**通讯引用:** 15702 | [OpenAlex ID](https://openalex.org/A5100705849)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的“带模式约束的近似最近邻搜索”（Pattern-Constrained ANNS）问题，并给出了完整的算法与实现；

**💡 创新点**

创新点在于引入了增强后缀自动机（Enhanced Suffix Automaton, ESAM）来对所有序列子串进行等价类划分，并将每个状态关联的向量集合使用 HNSW 图或原始 ID 集进行索引，既保持了空间近似线性，又实现了与查询模式匹配的精确过滤；

**🔧 技术方法**

主要技术包括：后缀自动机构造、等价类（poslist）定义、状态间索引重用与跳过构建策略、HNSW 图在状态索引中的使用、并行化构建与增量更新（插入/删除）等；

**📊 数据集**

实验使用了六个公开数据集：Spam（邮件标题+文本嵌入）、Words（单词+词向量）、MTG（图像描述+图像嵌入）、ArXiv（论文标题+文本嵌入）、SwissProt（蛋白序列+结构嵌入）和 CodeSearchNet（函数名+代码嵌入）；

**📈 对比分析**

与四种基线（OptQuery、PreFiltering、PostFiltering、Pgvector、ElasticSearch）对比，VectorMaton 在可行的场景下与 OptQuery 的查询吞吐量相近，同时比其它基线高出 2–3 倍，且在所有数据集上显著降低了索引大小（最高 18×压缩）和构建时间（最高 9×加速）；

**⚠️ 局限性**

限制主要在于：对删除操作只能采用惰性标记，无法即时释放空间；当状态 ID 集小于阈值 T 时使用暴力搜索，可能在极小查询集上产生一定的查询延迟；此外，构建 HNSW 图仍为串行过程，对极大状态集的构建仍有瓶颈。

---

## 132. RAIE: Region-Aware Incremental Preference Editing with LoRA for LLM-based Recommendation

**arXiv ID:** 2603.00638 | [PDF](https://arxiv.org/pdf/2603.00638v1)

**作者:** Jin Zeng `[一作]` (Sun Yat-sen University), Lu Bai `[通讯]` (Beijing Normal University)

**通讯引用:** 23623 | [OpenAlex ID](https://openalex.org/A5045790623)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种 Region‑Aware Incremental Editing (RAIE) 框架，针对大型语言模型推荐系统的用户偏好漂移，冻结主干模型并为每个语义一致的偏好区域配备独立 LoRA 适配器，支持 Update、Expand、Add 三种局部编辑操作，实现增量、细粒度的偏好更新。

**💡 创新点**

创新点在于：① 将用户行为划分为语义相干的“知识区域”，② 采用基于置信度的路由实现精确区域定位，③ 对目标区域执行三种编辑操作，仅更新局部参数，④ 通过局部 LoRA 适配器减少全局干扰和灾难性遗忘，整体实现精准、高效、可解释的持续学习。

**🔧 技术方法**

技术方法包括：LLM 表征提取（如 BERT4Rec、OpenP5），spherical k‑means 区域聚类，LoRA 参数高效微调，EMA 更新策略，置信度门控的归路与编辑决策，三种编辑操作（Update、Expand、Add）与局部 LoRA 训练。

**📊 数据集**

实验数据集：MovieLens‑10M 与 Yelp，均按时间切分为 Set‑up（S）、Fine‑tune（F）和 Test（T）三阶段，并进行 k‑core 过滤与正向滑动窗口构造训练样本。

**📈 对比分析**

评估方法：在 S→F→T 协议下与多种基线（Replay、LwF、LSAT、MoLE、E‑BPR、全局 LoRA 等）比较，使用 Recall@10 与 NDCG@10 评价。RAIE 在所有主干（BERT4Rec、SASRec、TiSASRec、OpenP5 等）上均显著提升 Recall@10/NDCG@10，且保持较低的遗忘率，尤其在 BERT4Rec 与 OpenP5 上表现突出。

**⚠️ 局限性**

局限性：① 依赖固定的区域数 K，需手动调参；② spherica k‑means 对高维稀疏数据敏感，可能影响区域划分质量；③ 在极度稀疏或噪声较大的数据集上表现有限；④ 仅在两大数据集验证，泛化性待进一步探索。

---

## 133. FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

**arXiv ID:** 2603.01712 | [PDF](https://arxiv.org/pdf/2603.01712v1)

**作者:** Qizheng Li `[一作]` (Microsoft Research Asia), Jiang Bian `[通讯]` (Microsoft Research Asia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了FT-Dojo交互式基准环境，并设计了专门的FT-Agent框架，实现了LLM在多域任务中的全流程自动微调。

**💡 创新点**

创新点在于把数据构造与训练配置统一视为可优化变量，构建了结构化迭代规划、fail-fast验证和反馈聚合三阶段的专属代理机制。

**🔧 技术方法**

采用LLM指令驱动代理、自动化数据清洗与合成工具、LoRA/SFT训练框架以及结构化评估器，实现端到端微调自动化。

**📊 数据集**

使用13个跨领域任务（数学、专利审查、化学、金融、表格QA）与对应的原始数据源，数据量限制在2k样本以内。

**📈 对比分析**

与基线（未微调模型、人工SFT、工具增强OpenHands）对比，FT-Agent在10/13任务上获得最佳成绩，特别在AIME 2025任务中实现非零准确率。

**⚠️ 局限性**

局限在于因果推理能力不足，难以对训练失败进行深层原因分析，导致部分任务出现“枪击式调试”而非系统性改进。

---

## 134. Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration

**arXiv ID:** 2603.01623 | [PDF](https://arxiv.org/pdf/2603.01623v1)

**作者:** Jiaqi Han `[一作]` (Stanford University), Stefano Ermon `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无训练的光谱特征预测器，用Chebyshev多项式逼近潜在特征并在线拟合系数，显著加速扩散模型推理。

**💡 创新点**

创新点在于采用全局Chebyshev谱逼近，给出误差不随跳步增大的长期预测，解决传统局部Taylor预测误差快速累积的问题。

**🔧 技术方法**

核心技术包括Chebyshev多项式逼近、岭回归在线拟合系数、仅缓存最后注意力块特征以及自适应时间步调度。

**📊 数据集**

实验使用文字图像生成的DrawBench和文字视频生成的VBench数据集。

**📈 对比分析**

与FORA、ToCa、TeaCache、TaylorSeer等基线在相同采样步数下比较，速度提升4–4.7×，且PSNR/SSIM等质量指标基本不下降甚至提升。

**⚠️ 局限性**

限制在于对极短采样步或过高阶多项式仍可能产生误差；仅缓存最后块可能忽略中间块信息；不同模型对缓存策略的适配性尚需进一步验证。

---

## 135. What Exactly do Children Receive in Language Acquisition? A Case Study on CHILDES with Automated Detection of Filler-Gap Dependencies

**arXiv ID:** 2603.02082 | [PDF](https://arxiv.org/pdf/2603.02082v1)

**作者:** Zhenghao Herbert Zhou `[一作]` (Yale University), Robert Frank `[通讯]` (Yale University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于混合主系和依赖解析的自动检测工具，对儿童语料中矩阵 wh-提问、嵌入 wh-提问和相对从句这三类核心填充-空格依赖进行细粒度子类化（按提取位划分）并进行大规模统计分析。

**💡 创新点**

创新点包括：①首次将主系解析与依赖解析结合，以提升对不同填充-空格结构的准确识别；②在每种构造中进一步细化为主语、宾语、附加等提取位子类，满足对学习输入的高分辨率需求；③通过该工具在数百万句子级别进行自动标注，为儿童语言习得和大模型泛化研究提供了大规模、细粒度的标注资源。

**🔧 技术方法**

使用技术：spaCy 的依赖解析器 + Berkeley Neural Parser（主系解析），结合自定义规则集完成结构识别与提取位判定；并在此基础上实现批量处理与统计汇总。

**📊 数据集**

数据集：57个英语 CHILDES 儿童语料库（约 2.84 万句子，含 3.19 万句子总量），并在 Brown、Valian 等子集上与人工标注数据做对比评估。

**📈 对比分析**

比较方法与性能：①手工抽样 100 句评估，精度 ≥ 0.94；②与已标注的 56,461 句子对照，Precision ≥ 0.82，Recall ≥ 0.78，F1 多数 > 0.8；③对儿童与成人语料进行按年龄分桶的频率与偏差分析，展示与成人输入高度一致的学习轨迹；④在 BabyLM 任务上进行过滤语料训练，验证不同构造的消融效果，显示模型对同一构造的显著性能下降。

**⚠️ 局限性**

局限性：①检测器受解析错误影响，误检/漏检仍存在；②仅覆盖三类核心填充-空格构造，未识别如 cleft、topicalization 等；③目前仅适用于英语；④大规模应用时需进一步优化规则以降低误差。

---

## 136. An Analysis of Multi-Task Architectures for the Hierarchic Multi-Label Problem of Vehicle Model and Make Classification

**arXiv ID:** 2603.01746 | [PDF](https://arxiv.org/pdf/2603.01746v1)

**作者:** Alexandru Manole `[一作]` (Babeș-Bolyai University), Laura Diosan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了多任务学习在车辆制造商和型号层级分类问题中的效果，比较并分析了并行、级联等多任务架构对CNN和Transformer的影响。

**💡 创新点**

通过系统实验探讨了损失权重、dropout、不同基础模型对多任务的影响，并首次在VMMC任务中对Transformer（MaxVit‑T）和ConvNext与DenseNet进行对比，揭示并行MTL最具优势。

**🔧 技术方法**

使用了多任务学习（并行与级联）、加权交叉熵损失、Adam + one‑cycle 学习率调度、Dropout 以及在 ImageNet 预训练权重上微调的 CNN 和 Vision Transformer。

**📊 数据集**

在 Stanford Cars（16,185 张图像，196 型号 / 49 制造商）和 CompCars（186,726 张图像，1716 型号 / 163 制造商）两大公开基准上进行实验。

**📈 对比分析**

采用单任务基线对比并计算多任务与单任务的准确率提升；实验显示并行 MT‑L 在 DenseNet、MaxVit‑T 上提升约 3–5% 准确率，最大达到 83.8%（CompCars）或 98.9% top‑5（MaxVit‑T）。

**⚠️ 局限性**

仅评估了两种数据集，未覆盖更深层级或跨任务交互；ConvNext 的 MT‑L 效果不佳；缺乏大规模数据或多任务层级的进一步验证。

---

## 137. Towards Orthographically-Informed Evaluation of Speech Recognition Systems for Indian Languages

**arXiv ID:** 2603.00941 | [PDF](https://arxiv.org/pdf/2603.00941v1)

**作者:** Kaushal Santosh Bhogale `[一作]` (Indian Institute of Technology Madras), Mitesh M. Khapra `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 3496 | [OpenAlex ID](https://openalex.org/A5050036814)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一套基于正字法变体的评估框架和指标（Orthographically‑Informed Word Error Rate，OIWER），通过大型语言模型自动生成上下文感知的词级变体，并在人工校正后用于评估印度语言ASR系统。

**💡 创新点**

创新点包括：①利用LLM生成多样化、上下文相关的正字法变体；②将这些变体集成到WER计算中，得到更贴合人类感知的评估；③验证LLM生成的变体可大幅替代人工校正，显著降低成本。

**🔧 技术方法**

采用的技术有：大型语言模型（如 Gemini‑2.5‑Pro）生成变体；动态规划实现的OIWER算法；LabelStudio 接口进行人工后期编辑；与传统 WER、WER‑SN 进行对比实验。

**📊 数据集**

使用的数据集包括：IndicVoices（22 语言评测基准）；MahaDhwani 伪标注数据用于 fine‑tune Canary；以及 Gemini 生成的文本用于变体生成。

**📈 对比分析**

通过与标准 WER、WER‑SN 以及 Gemini、Canary、IndicConformer 等多种模型的对比，OIWER 在所有语言上平均降低 6.3 分的错误率，缩小 Gemini 与 Canary 的性能差距（由 18.1 分降至 11.5 分），并且与人工感知 WER 的差距缩小 4.9 分，表明其更能反映实际性能。

**⚠️ 局限性**

局限性包括：仍需人工后期校正以确保变体质量；对音频歧义导致的误差无法完全消除；部分低资源语言的变体覆盖可能不完整；LLM 生成的变体质量取决于模型本身，可能出现错误。

---

## 138. GCTAM: Global and Contextual Truncated Affinity Combined Maximization Model For Unsupervised Graph Anomaly Detection

**arXiv ID:** 2603.01806 | [PDF](https://arxiv.org/pdf/2603.01806v1)

**作者:** Xiong Zhang `[一作]` (Yunnan University), Hua Jiang `[通讯]` (Yunnan University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种无监督图异常检测框架GCTAM，通过结合上下文和全局截断亲和力最大化实现异常节点识别。

**💡 创新点**

创新点在于同时使用上下文截断模块（CAT）和全局截断模块（GAT）替代传统硬阈值截断，利用上下文相似度与欧氏距离动态截断异常边，并通过全局亲和图增强正常节点亲和，最终在共享GCN中最大化亲和分数。

**🔧 技术方法**

采用图神经网络（GCN）与MLP投影，结合余弦相似度、欧氏距离、稀疏邻接构造、亲和分数计算以及二元阈值截断等技术。

**📊 数据集**

在七个真实图异常检测数据集上评估：Amazon、YelpChi、ACM、Facebook、Reddit、Amazon-all、YelpChi-all。

**📈 对比分析**

与八种最先进方法（CoLA、SL-GAD、HCM-A、DOMINANT、iForest、ANOMALOUS、ComGA、GGAD、TAM）比较，GCTAM在大部分数据集上均取得最高AUROC和AUPRC，尤其在Amazon和YelpChi上相较最佳对手提升约15-20%。

**⚠️ 局限性**

局限性：对高度异质（heterophilic）图如Reddit的性能相对较弱；需要手动调节截断比例β和邻居数k，对不同数据集敏感；模型复杂度和运行时间相对较高。

---

## 139. A Framework for Transparent Reporting of Data Quality Analysis Across the Clinical Electronic Health Record Data Lifecycle

**arXiv ID:** 2603.00921 | [PDF](https://arxiv.org/pdf/2603.00921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 140. CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification

**arXiv ID:** 2603.01940 | [PDF](https://arxiv.org/pdf/2603.01940v1)

**作者:** Jinpeng Chen `[一作]` (Huawei Research), Rui Liu `[通讯]` (Huawei Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CoVe框架，用明确任务约束引导多轮交互工具使用数据的生成与检验。

**💡 创新点**

创新点在于利用明确约束进行采样与模糊化，结合规则验证确保数据可解且完全正确，同时支持SFT和RL两种后训练方式。

**🔧 技术方法**

使用约束采样、约束模糊化、用户模拟LLM、多轮对话生成、规则验证、SFT与RL训练等技术。

**📊 数据集**

采用12K条高质量交互轨迹（基于τ^2-bench Airline和Retail域）及相应的sandbox数据库。

**📈 对比分析**

在τ^2-bench上与多种开源与专有模型对比，CoVe-4B在pass^1上达到51.2%（Retail）和43.0%（Airline），在同规模模型中排名第一，甚至接近70B级别模型。

**⚠️ 局限性**

限制主要是SFT+RL顺序训练因用户模拟器能力不足导致性能下降，且仅验证了Airline与Retail两个域，缺乏跨域泛化评估。

---

## 141. A level-wise training scheme for learning neural multigrid smoothers with application to integral equations

**arXiv ID:** 2603.01064 | [PDF](https://arxiv.org/pdf/2603.01064v1)

**作者:** Lingfeng Li `[一作]` (Hetao Institute of Mathematics and Interdisciplinary Sciences), Justin Wan `[通讯]` (University of Waterloo)

**通讯引用:** 551 | [OpenAlex ID](https://openalex.org/A5102887139)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出了一种在多重网格框架下，利用神经网络作为平滑器来求解卷积型积分方程的线性系统；

**💡 创新点**

创新点在于设计了层级频率滤波的损失函数，强制每层神经平滑器只针对对应频段的误差，从而实现对高频误差的有效消除，并保持对低频误差的递归处理；

**🔧 技术方法**

使用了 Fourier Neural Operator（FNO）架构作为平滑器，结合离散傅里叶变换实现频率滤波；

**📊 数据集**

实验数据为人工生成的右端向量，针对 1D 和 2D 维度的积分方程（高斯卷积核）以及含有 Tikhonov 与各向异性正则化的 1D 方程；

**📈 对比分析**

与传统多重网格（Jacobi 预/后平滑）以及共轭梯度（CG）比较，神经多重网格在积分方程上显著减少迭代次数与总耗时，且对正则化参数 α 与网格尺寸 n 的变化更为鲁棒；

**⚠️ 局限性**

局限性是仅适用于系数矩阵固定、右端向量变化的线性系统；若系数矩阵变动，则需重新训练所有平滑器，训练成本较高；

---

## 142. ContextCov: Deriving and Enforcing Executable Constraints from Agent Instruction Files

**arXiv ID:** 2603.00822 | [PDF](https://arxiv.org/pdf/2603.00822v1)

**作者:** Reshabh K Sharma `[一作]` (University of Washington), Reshabh K Sharma `[通讯]` (University of Washington)

**通讯引用:** 28 | [OpenAlex ID](https://openalex.org/A5068245169)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将Agent Instruction文件从被动文本转为可执行规范，自动提取、合成并执行检查，防止LLM代理产生上下文漂移。

**💡 创新点**

提出层次化约束提取、域路由合成与多层运行时执行的框架ContextCov，首次将自然语言约束转为可执行的静态、过程与架构检查。

**🔧 技术方法**

利用Markdown AST、Tree-sitter、NetworkX、Shell shim以及LLM（ChatGPT）进行约束提取与代码生成，并使用LLM-as-judge完成语义层次检查。

**📊 数据集**

评估基于Chatlatanagulchai等的723个GitHub仓库（共182k星），这些仓库均包含Agent Instruction文件。

**📈 对比分析**

与无检查或仅人工检查的对比，ContextCov提取约46k可执行检查，99.997%语法合法，检测到超过50万违例，81%仓库至少有一条违例，显著提升了合规性检测效率。

**⚠️ 局限性**

存在误报风险、LLM误解约束、缺乏对实际agent运行日志的实验、对语义检查的LLM依赖以及需要人工审核检查文件等局限。

---

## 143. Adversarial Query Synthesis via Bayesian Optimization

**arXiv ID:** 2603.01570 | [PDF](https://arxiv.org/pdf/2603.01570v1)

**作者:** Jeffrey Tao `[一作]` (University of Pennsylvania), Ryan Marcus `[通讯]` (University of Pennsylvania)

**通讯引用:** 1806 | [OpenAlex ID](https://openalex.org/A5025731013)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个自动化系统，利用贝叶斯优化在SQL查询与执行计划的联合潜在空间中搜索具有显著优化空间（headroom）的对手式查询‑计划对，生成难度更高的基准工作负载。

**💡 创新点**

将贝叶斯优化扩展到离散结构化搜索空间，使用组合变分自编码器将查询与计划映射到共享连续潜在空间；同时通过语法约束解码确保生成的SQL合法，并直接针对headroom进行优化。

**🔧 技术方法**

组合变分自编码器（VAE）、贝叶斯优化（高斯过程+获取函数）、语法约束解码（EBNF+vLLM）、文本嵌入编码器（OpenAI）、小型LLM解码器、潜在空间联合搜索等技术。

**📊 数据集**

使用IMDb数据库生成122个查询‑计划对，并与JOB、JOB‑Complex、Stack等现有基准进行对比。

**📈 对比分析**

在DuckDB上对比绝对与相对headroom，结果显示生成工作负载的中位绝对headroom为25秒，相对headroom为20倍，最高可达30秒/80倍，明显优于JOB、JOB‑Complex和Stack。

**⚠️ 局限性**

搜索耗时长、仅限于简单的等值连接与谓词的连结查询、潜在空间质量受限于约20k查询的数据量，且目前仅为初步实验，未验证在更大规模或不同数据库上的可迁移性。

---

## 144. Pro-HOI: Perceptive Root-guided Humanoid-Object Interaction

**arXiv ID:** 2603.01126 | [PDF](https://arxiv.org/pdf/2603.01126v1)

**作者:** Yuhang Lin `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 61910 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 Pro‑HOI 框架，利用根轨迹指导实现了在 Unitree G1 上基于单机感知与计算的鲁棒人形-物体交互，支持连续搬运、导航、障碍规避与失误自恢复；

**💡 创新点**

核心创新在于：① 将根轨迹与接触状态作为低维观测，解耦运动追踪奖励；② 引入数字孪生+实时检测实现失误检测与重抓；③ 设计可与高层规划无缝协同的通用接口；

**🔧 技术方法**

采用端到端强化学习（PPO）+根轨迹指导，SDF 优化的运动重定向，FoundationPose 6D 姿态估计，TEB 本地规划，数字孪生落地预测以及域随机化等技术；

**📊 数据集**

使用自采的 SMPL 人体箱子搬运动作（Xsens 记录）与 Blender 合成的物体几何数据；未使用公开数据集；

**📈 对比分析**

与 OmniRetarget、HDMI、DemoHLM、PhysHSI、Falcon 等基线对比；在 5,756 个 ID/OOD 场景中，Pro‑HOI 抓取成功率 99.93%、放置精度 6.46 cm、任务成功率 88.38%，显著优于 PhysHSI（82.54%/9.5 cm/70.17%）等；在连续 15 次搬运、障碍规避与自动重抓等实验中表现出极高的鲁棒性；

**⚠️ 局限性**

仍需预先构造根轨迹与接触模式，极端高速与复杂 3D 环境适应性待验证；数字孪生依赖外部计算资源，视觉遮挡严重时估计精度下降；

---

## 145. Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning

**arXiv ID:** 2603.01639 | [PDF](https://arxiv.org/pdf/2603.01639v1)

**作者:** Jiebin Zhang `[一作]` (Peking University), Sujian Li `[通讯]` (Peking University)

**通讯引用:** 8219 | [OpenAlex ID](https://openalex.org/A5058353424)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LTD方法，利用强化学习动态优化LLM推理中的草稿与验证阶段，从而显著提升推理吞吐量。

**💡 创新点**

直接以吞吐量为奖励，训练深度与验证规模双策略的协同适配，并引入迭代共适配训练以实现最佳平衡。

**🔧 技术方法**

强化学习（PPO）+树结构草稿+两策略MLP+动态草稿深度与验证规模决策。

**📊 数据集**

HumanEval（训练），MT‑Bench、GSM8K、Alpaca、Natural Questions、MMLU等（评测）。

**📈 对比分析**

与Eagle3及多种动态方法对比，greedy下平均提升6.5%至36.4%，高温下仍保持最高加速，表现优异。

**⚠️ 局限性**

未能实现最长接受长度，且对极端大模型的超参数调优仍需实验支持，缺乏对所有推理模式的完整覆盖。

---

## 146. Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos

**arXiv ID:** 2603.00881 | [PDF](https://arxiv.org/pdf/2603.00881v1)

**作者:** Yu Luo `[一作]` (Ocean University of China), Yueming Lyu `[通讯]` (Nanjing University)

**通讯引用:** 344 | [OpenAlex ID](https://openalex.org/A5053437240)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出基于SAM3的教师‑学生半监督框架 SMART，融合概念提示、进阶置信一致性与双流时间一致性，实现 X‑光血管视频的高精度分割。

**💡 创新点**

创新点包括：① 用 SAM3 的概念提示替代几何提示，提升伪标签质量；② 引入进阶置信一致性正则化，动态加权不确定区域；③ 采用双流光流一致性与流一致性损失，捕获时间连贯性。

**🔧 技术方法**

技术手段包括 SAM3 模型微调、进阶置信一致性损失、双流光流一致性与流一致性损失、SEA‑RAFT 光流估计、弱/强数据增强、Dice/BCE 监督损失。

**📊 数据集**

使用 XCAV、CADICA、私有 CAVSA 三套血管视频数据集；XCAV 111 视频、CAVSA 1061 视频、CADICA 662 视频。

**📈 对比分析**

与 UNet、MedSAM2、SAM3、KnowSAM、CPC‑SAM、Denver 等 SOTA 方法对比；在 XCAV 上仅用 14% 标注得到 Dice 84.39%（领先 CPC‑SAM 6.49%），在 CAVSA 上仅用 1.5% 标注提升 13.1% Dice；整体在 Dice、clDice 等指标上均显著优于基线。

**⚠️ 局限性**

局限性包括：对光流估计质量依赖强，光流误差会影响一致性损失；计算资源需求较高；仅在 X‑光血管视频上验证，需进一步验证至其他医学视频；对噪声扰动次数敏感，需调参。

---

## 147. Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs

**arXiv ID:** 2603.00590 | [PDF](https://arxiv.org/pdf/2603.00590v1)

**作者:** Yiran Zhao `[一作]` (Nanjing University of Aeronautics and Astronautics), Liming Fang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 10694 | [OpenAlex ID](https://openalex.org/A5066215018)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IRIS Benchmark，统一同步评估统一多模态大语言模型（UMLLM）在生成与理解任务中的公平性。

**💡 创新点**

创新点在于：① 设计三维公平评估框架（Ideal Fairness、Real-world Fidelity、Bias Inertia & Steerability）；② 将生成与理解双任务同步映射到高维公平空间；③ 引入 IRIS‑MBTI 人格诊断工具以直观展示模型公平特征。

**🔧 技术方法**

技术手段包括：高维度度量归一化与聚合、ARS 自适应多路径人口属性分类器、四大自制评测数据集、三维公平空间投影与指数衰减评分。

**📊 数据集**

使用数据集：IRIS-Ideal-52、IRIS-Steer-60、IRIS-Gen-52、IRIS-Classifier-25，以及公开图像/文本集合进行自动标注。

**📈 对比分析**

通过对 7 大 UMLLM 与 5 个控制模型的公平得分与 IRIS‑MBTI 诊断进行多维比较，揭示生成缺口、任务间人格分裂等现象；不同模型在各维度上表现不一，无法出现全能最佳模型。

**⚠️ 局限性**

局限性包括：人口属性离散化粗糙导致交叉身份欠表征；ARES 分类器存在噪声与潜在偏差；Steerability 指标缺乏人工评估；评测仅覆盖图像职业场景与 VQA，未扩展至其他模态或任务。

---

## 148. Leveraging Model Soups to Classify Intangible Cultural Heritage Images from the Mekong Delta

**arXiv ID:** 2603.02181 | [PDF](https://arxiv.org/pdf/2603.02181v1)

**作者:** Quoc-Khang Tran `[一作]` (Can Tho University), Nguyen-Khang Pham `[通讯]` (Can Tho University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在红河三角洲文化遗产图像分类任务中，本文通过将CoAtNet混合卷积‑自注意力网络与权重空间平均的模型Soups（greedy与uniform）相结合，构建并评估了一个低资源、无额外推理成本的集成框架。

**💡 创新点**

创新点在于首次将CoAtNet的混合结构与模型Soups技术结合，提出一种多样性驱动的权重平均方法，并通过MDS可视化与交叉熵距离证明其在输出空间中的多样性优势。

**🔧 技术方法**

使用了CoAtNet架构、模型Soups（greedy/ uniform）权重平均、交叉熵距离+MDS可视化、MixUp+CutMix数据增强、AdamW+Cosine annealing调度、fp16混合精度训练等技术。

**📊 数据集**

实验数据集为ICH‑17，包含7406张图像、17个文化遗产类别，主要用于训练、验证与测试。

**📈 对比分析**

与ResNet‑50、DenseNet‑121、ViT等基线对比，CoAtNet‑2+Uniform Soup在测试集上达到72.36% top‑1准确率、69.28%宏F1，显著优于所有基线模型，验证了方法的有效性。

**⚠️ 局限性**

主要限制包括对ImageNet预训练权重的强依赖、对噪声标签敏感、数据集不可公开、仅在单一区域上验证，以及模型Soups对检查点多样性的依赖程度仍有待进一步提升。

---

## 149. VizQStudio: Iterative Visualization Literacy MCQs Design with Simulated Students

**arXiv ID:** 2603.00994 | [PDF](https://arxiv.org/pdf/2603.00994v1)

**作者:** Zixin Chen `[一作]`, Meng Xia `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套可视化分析系统，利用多模态大型语言模型（MLLM）自动生成并迭代设计可视化素养的多选题，同时通过模拟学生的推理过程为教师提供即时反馈，支持在设计阶段对题目质量与多样性进行评估。

**💡 创新点**

创新点在于将 MLLM 与可视化生成、学生属性建模、推理模拟和可视化分析工具融合，形成教师与 AI 的交互式迭代设计流程；并首次实现多模态（图表+文本）题目生成与学生推理可视化的结合，提供设计时可视化的误导图表、难度调节与学生误解挖掘。

**🔧 技术方法**

核心技术包括：1）GPT‑4o 及检索增强生成（RAG）驱动的 D3.js 代码与题干、答案、干扰项生成；2）多模态学生模型（基于人口学、认知特质、可视化知识点）与聚类；3）LLM 推理生成的学生推理轨迹；4）可视化分析仪表盘（Sankey、雷达图、堆叠柱状图）实现设计反馈展示。

**📊 数据集**

使用的数据集：80 个标准 D3.js 可视化模板；已公开的可视化素养题库（VLAT、CALVI）；教师上传的幻灯片与图表样例；模拟学生采用合成属性，无真实学生数据；大规模线上实验招募的 100 名 Prolific 受试者数据。

**📈 对比分析**

比较方法：① 对齐度评估（认知、推理步骤、语义一致性）与生成可靠性测量；② 课堂实验比较真实学生答题准确率与模拟学生预测，使用 Pearson 相关系数；③ 学习成效评估，采用前测-后测差异、学习收益与主观教学价值评估。性能表现：模拟器平均预测准确率与真实学生相近，学习收益与标准 VLAT 题库无显著差异；生成与迭代平均耗时 10–15 秒，学生模拟约 30 秒；整体系统交互流畅，支持教师每题多次迭代。

**⚠️ 局限性**

限制：1）依赖外部 LLM，生成结果可能含错误、偏见或不一致；2）学生模拟主要覆盖结构化推理，无法充分捕捉启发式或粗略估计等真实学生的捷径行为；3）缺乏大规模真实学生数据验证，模拟结果需教师人工校验；4）系统仍需教师对最终题目内容进行人工审核，无法完全自动化。

---

## 150. CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning

**arXiv ID:** 2603.01768 | [PDF](https://arxiv.org/pdf/2603.01768v1)

**作者:** Pratik Jawahar `[一作]` (University of Manchester), Maurizio Pierini `[通讯]` (CERN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出Causal Hamiltonian Learning Unit（CHLU），通过相对论哈密顿动力学与辛积分实现长期稳定记忆与可控噪声滤除。

**💡 创新点**

创新点在于将相对论速度上限、辛守恒结构与Wake‑Sleep对比散度训练相结合，突破离散不稳定与连续耗散的两难。

**🔧 技术方法**

使用相对论动能守门器、辛 Verlet 积分、Wake‑Sleep 对比散度训练以及 Langevin 采样技术。

**📊 数据集**

实验数据集包括MNIST图像以及自定义周期轨迹（Lemniscate、正弦波）。

**📈 对比分析**

与LSTM、Neural ODE基线对比，CHLU在长期轨迹跟踪保持闭环稳定、速度受限且无无穷加速度，生成实验可得到多种数字模式。

**⚠️ 局限性**

局限在于缺乏大规模性能评估、参数调优与失效模式分析，生成质量仍受潜在模式偏好影响。

---

## 151. Direct low-field MRI super-resolution using undersampled k-space

**arXiv ID:** 2603.00668 | [PDF](https://arxiv.org/pdf/2603.00668v1)

**作者:** Daniel Tweneboah Anyimadu `[一作]` (University of Exeter), Ahmed Karam Eldaly `[通讯]` (University College London)

**通讯引用:** 65 | [OpenAlex ID](https://openalex.org/A5055467044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研发了一种在低场 MRI k 空间直接进行超分辨率/图像质量转移的深度学习框架，以实现从欠采样 k 空间恢复高场相似图像。

**💡 创新点**

将 SR/IQT 与重建统一到 k 空间域，采用双通道 U-Net 同时处理实部和虚部，首次在低场 MRI 直接从欠采样 k 空间完成 SR/IQT。

**🔧 技术方法**

采用双通道 U-Net、复数卷积、Adam 优化器、MAE+MSE 损失，使用 PyTorch 实现。

**📊 数据集**

使用 3T HCP T1w 数据通过噪声/对比模拟合成低场 MRI，并以伪径向与笛卡尔 50% 与 30% 欠采样率进行训练。

**📈 对比分析**

与空间域 IQT 与插值基准对比，采用 PSNR/SSIM 评价；k 空间 IQT 在 30% 采样下获得 PSNR 33.17 dB、SSIM 0.9406，优于空间域 30.54 dB/0.9330，且与全采样质量相当。

**⚠️ 局限性**

仅在合成低场数据上验证，未处理真实低场扫描中的运动/磁不均匀等问题，且依赖预先仿真生成的低场图像。

---

## 152. Prompt Sensitivity and Answer Consistency of Small Open-Source Large Language Models on Clinical Question Answering: Implications for Low-Resource Healthcare Deployment

**arXiv ID:** 2603.00917 | [PDF](https://arxiv.org/pdf/2603.00917v1)

**作者:** Shravani Hariprasad `[一作]` `[通讯]` (Independent Researcher), Shravani Hariprasad (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了5个小型开源语言模型（Gemma 2、Phi-3 Mini、Llama 3.2、Mistral 7B、Meditron-7B）在三组医学问答基准（MedQA、MedMCQA、PubMedQA）下，使用五种提示风格（原始、正式、简化、角色扮演、直接）对其回答的一致性、准确性和指令遵循率进行全面评估。

**💡 创新点**

提出了基于多数答案的一致性得分和UNKNOWN率等多维指标，揭示高一致性并不等同于高准确性，首次系统证明角色扮演提示会显著降低小型模型的临床问答性能，并证明域知识与指令遵循是独立需求。

**🔧 技术方法**

采用温度为0的确定性推理、Ollama REST API 在消费级 CPU 上进行本地推理，并通过正则表达式提取答案、计算一致性、准确率和UNKNOWN率等指标。

**📊 数据集**

使用 MedQA、MedMCQA、PubMedQA 三个公开的医学问答基准，每组随机抽取 200 题，共 600 题。

**📈 对比分析**

对每个模型在三组数据与五种提示风格下的平均一致性、准确率和UNKNOWN率进行统计比较；结果显示 Llama 3.2 在准确率上最高（49–65%），Gemma 2 在一致性上最高（0.845–0.888）但准确率最低（33–44%），且角色扮演提示导致所有模型准确率普遍下降。

**⚠️ 局限性**

局限性包括样本量有限、未做医学细化微调、仅评估多选/是/否/可能等固定格式、未覆盖真实临床对话、缺乏人类评估、仅测试 CPU 环境且未检验更大模型的表现。

---

## 153. Stroke outcome and evolution prediction from CT brain using a spatiotemporal diffusion autoencoder

**arXiv ID:** 2603.00756 | [PDF](https://arxiv.org/pdf/2603.00756v1)

**作者:** Adam Marcus `[一作]` (Imperial College London), Daniel Rueckert `[通讯]` (Technische Universität München)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种基于扩散自编码器的自监督方法，利用CT图像生成语义表示并预测中风患者下一日NIHSS和出院时mRS。

**💡 创新点**

创新点在于将去噪扩散概率模型（DDPM）作为自编码器，同时加入时空信息，利用无标签纵向CT影像构建时空语义表示。

**🔧 技术方法**

使用了扩散去噪概率模型（DDPM）、ResNet‑50语义编码器、AdaSpaGN/AdaTempGN归一化、自监督学习、FID/MSE评估以及传统CNN、VICReg等对比模型。

**📊 数据集**

数据集为两家医院共3,573例急性缺血性中风患者的5,824幅CT影像，采用随机20%测试集、五折交叉验证进行训练。

**📈 对比分析**

与CNN、VICReg、变分自编码器等基线比较，时空扩散自编码器在预测next‑day NIHSS的AUC为0.669，出院mRS的AUC为0.789，表现最佳。

**⚠️ 局限性**

主要局限包括缺乏90天长期结局数据、模型对局部图像细节捕捉有限、未结合临床信息，且仅使用CT影像。

---

## 154. Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering

**arXiv ID:** 2603.01997 | [PDF](https://arxiv.org/pdf/2603.01997v1)

**作者:** Hari Prasanth S. M. `[一作]`, Risto Ojala `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种仅使用事件相机数据并结合转子转速估计的实时无人机轨迹预测方法。

**💡 创新点**

通过从事件数据中提取螺旋桨转速来动态调节Kalman滤波过程噪声，实现了无学习、无RGB依赖的高精度轨迹预测。

**🔧 技术方法**

采用事件相机时序处理、频率直方图周期估计、RPM估计、基于常数速度的Kalman滤波器以及过程噪声按转速调节等技术。

**📊 数据集**

使用FRED数据集（同步RGB与事件相机的高分辨率无人机轨迹数据）。

**📈 对比分析**

与线性外推、经典Kalman滤波以及多种LSTM/Transformer/CNN+Transformer深度学习模型对比，在FRED测试集上0.4s/0.8s ADE和FDE均最低，提升约1.6–4.2像素。

**⚠️ 局限性**

仅估计全机转速而非单旋翼，未考虑姿态估计；对多机同时出现或遮挡情况的鲁棒性待验证；方法依赖事件相机的良好可视性。

---

## 155. Unix Tools and the FITO Category Mistake: Crash Consistency and the Protocol Nature of Persistence

**arXiv ID:** 2603.01384 | [PDF](https://arxiv.org/pdf/2603.01384v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAEDALUS), Paul Borrill (DAEDALUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统分析了 Unix 文件系统及其下层存储栈中的“即时原子状态转移”假设，并证明其为类别错误。

**💡 创新点**

首次形式化证明 syscall 级别的持久性边界在失败情况下不可观察，揭示跨层时间假设泄漏的递归链条，并提出将持久性视为协议收敛而非时间事件的新范式。

**🔧 技术方法**

采用正式因果模型、可重放日志分析、实验验证 ext4/XFS/Btrfs 等文件系统以及 NVMe Flush/FUA 行为，并对 Linux restartable sequences 与 NMI 进行理论探讨。

**📊 数据集**

利用公开的云服务事件记录（Google、AWS、Meta 等）以及 PostgreSQL、etcd、MySQL 的日志来演示 fsync 失败导致的数据破坏。

**📈 对比分析**

与传统基于 fsync 的可靠性模型对比，通过实验显示重试机制在 FITO 失效时导致吞吐量下降、计算浪费 12–43%，财务损失达数十亿美元。

**⚠️ 局限性**

研究主要聚焦在 Linux 生态与 NVMe 设备，未覆盖其他文件系统或存储介质；理论结论在实际系统中实现需重新设计协议，复杂性与兼容性仍是挑战。

---

## 156. Compatible Triangulations of Simple Polygons

**arXiv ID:** 2603.01282 | [PDF](https://arxiv.org/pdf/2603.01282v1)

**作者:** Peyman Afshani `[一作]` (Aarhus University), Günter Rote `[通讯]` (Freie Universität Berlin)

**通讯引用:** 4297 | [OpenAlex ID](https://openalex.org/A5083387191)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在给定两个简单多边形P和Q时，如何计算两者兼容的三角剖分，并针对两种情形给出了高效算法。

**💡 创新点**

创新点：①在已知P的三角剖分时，利用可见性查询数据结构实现 O(n log n + nr) 的判定；②在已知顶点对应关系时，将兼容剖分问题归约为布尔矩阵乘法，得到 O(n^ω) 的极限时间解，显著提升了先前的 O(n^3) 动态规划方法。

**🔧 技术方法**

技术要点：递归二叉分割的可见性查询结构、批量可见性检验、二分搜索、布尔矩阵乘法、块递归与动态规划。

**📊 数据集**

论文未使用公开数据集，主要在理论上证明复杂度；实验以合成多边形为例进行验证。

**📈 对比分析**

与之前的 O(n^3) 方法相比，算法把时间降低到几乎与矩阵乘法相同的 n^ω；实验表明在 n≈2000 时已显著提升。

**⚠️ 局限性**

局限性：①第一个变体受 Q 的凸/凹顶点数 r 限制；②第二个变体要求顶点一一对应且仅适用于无洞多边形；③需要高效矩阵乘法实现；④论文未讨论 Steiner 点的加入与最小化问题。

---

## 157. A Diffusion-Driven Fine-Grained Nodule Synthesis Framework for Enhanced Lung Nodule Detection from Chest Radiographs

**arXiv ID:** 2603.01659 | [PDF](https://arxiv.org/pdf/2603.01659v1)

**作者:** Aryan Goyal `[一作]` (Qure.ai), Preetham Putha `[通讯]` (Qure.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在胸部X光片上通过条件扩散模型与LoRA适配器实现可控肺结节合成

**💡 创新点**

引入多属性LoRA分离与正交化损失，实现属性间无冲突的合成与精准可控

**🔧 技术方法**

Diffusion Transformer（DiT-XL/2）、LoRA、正交化正则、CFG以及mask条件

**📊 数据集**

内部1.2M CXRs（40k结节）与公开JSRT、ChestX-ray14数据集

**📈 对比分析**

与GAN（ACGAN、ReACGAN）及填充模型CR-Fill比较，差异化合成后下游检测AUC提升至0.9023/0.9318，明显优于基线

**⚠️ 局限性**

合成的多属性插值仍受限于LoRA空间，难以处理极端或离散分布的属性组合

---

## 158. A Study on Building Efficient Zero-Shot Relation Extraction Models

**arXiv ID:** 2603.01266 | [PDF](https://arxiv.org/pdf/2603.01266v1)

**作者:** Hugo Thomas `[一作]`, Pascale Sébillot `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对在大规模文本库中进行实时零样本关系抽取的实际需求，本文提出了新的模型评估标准，并对三种主流的零样本关系抽取模型（Emma、ReMatching、AlignRE）进行改造，使其实现单通道（single‑pass）编码和离线预计算；随后在此基础上设计并实验三种拒绝机制（阈值、描述式拒绝、原型式拒绝），最后在 FewRel 和 WikiZSL 数据集上进行统一评测。

**💡 创新点**

① 形成了专门针对“离线编码 + 实时零样本 + 拒绝”三大约束的模型分层分类体系；② 在此体系下首次系统地将现有模型改造为可离线预编码的单通道版本；③ 设计了兼容所有模型的三种拒绝机制，并提出了一种结合排名目标的复合损失函数，提升拒绝与预测的权衡。

**🔧 技术方法**

使用基于 BERT 的编码器（含 SBERT 侧信息编码器）、late‑interaction 结构、单通道候选关系表示（通过头/尾实体的首/尾/均值/最大池化提取）、余弦相似度分类；拒绝机制采用阈值学习、描述式原型、以及多原型的余弦相似度；训练损失主要是平方铰链损失与自定义的三项排名损失。

**📊 数据集**

实验数据集主要为 FewRel（80 关系、56k 句子）和 WikiZSL（113 关系、94k 句子），并在这两个数据集上分别采用 5/10/15 个未知关系类别进行评估；此外还提及了 ReTACRED、NYT 等包含拒绝标签的数据集，但实际实验仅在上述两大公开数据集上完成。

**📈 对比分析**

与原始模型（off‑the‑shelf）相比，改造后的单通道模型在 F1 上仅损失 1–3 分；在离线编码 + 拒绝设置下，AlignRE 在宏 F1（无拒绝）与拒绝准确率上均优于 ReMatching 与 Emma；阈值拒绝机制效果最差，而描述式和原型式拒绝在保持较高 F1 的同时，拒绝准确率可达 70–90%；实验报告了不同未知关系数量下的性能曲线，并对各模型的参数量和推理速度进行了简要比较。

**⚠️ 局限性**

① 需要在目标任务中预先提供实体指示器或实体类型信息，单通道改造虽可避免实体识别，但在极长文本中仍可能影响上下文捕捉；② 拒绝机制的训练依赖于负样本设定，若负样本比例不匹配，可能导致过度拒绝；③ 仅在小规模公开数据集上验证，尚未在真正的大规模行业语料库（如法律或医学文本）上评估其可扩展性和鲁棒性；④ 该工作聚焦于 encoder‑only 结构，对生成式或检索式零样本关系抽取方法未做覆盖。

---

## 159. ToolRLA: Fine-Grained Reward Decomposition for Tool-Integrated Reinforcement Learning Alignment in Domain-Specific Agents

**arXiv ID:** 2603.01620 | [PDF](https://arxiv.org/pdf/2603.01620v1)

**作者:** Pengbo Liu `[一作]` `[通讯]`, Pengbo Liu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在金融顾问领域构建并部署了一款使用多工具调用的 ReAct 语言模型，并通过三阶段后训练流程实现高效、合规的对话交互。

**💡 创新点**

提出了细粒度、多维度的乘法分解奖励函数，将工具调用质量拆分为格式、正确性、效率与合规四个维度，并通过合规惩罚实现优先级偏置。

**🔧 技术方法**

使用了 Qwen3-14B 语言模型，先做监督微调 (SFT)，再利用 Group Relative Policy Optimization (GRPO) 进行强化学习，最后通过 Direct Preference Optimization (DPO) 对合规边界进行细调。

**📊 数据集**

训练集包括 4.2K sandbox 验证的轨迹、FA‑Bench 内部数据集（500 条生产查询）以及 4–6 万条合规敏感对话样本；评估使用 ToolBench、API‑Bank 等公开基准。

**📈 对比分析**

与多模型管道、ReAct+SFT、PPO、GRPO（粗糙/加法奖励）以及公开基准模型对比，ToolRLA 在任务完成率、工具调用错误率、合规违规率等指标上均领先，生产环境中任务完成率提升 47%，错误率下降 63%，违规率下降 93%。

**⚠️ 局限性**

局限包括对 sandbox 仿真依赖导致的奖励稀疏、合规注释成本高、模型只支持文本输入、对 API 变化的鲁棒性需进一步提升。

---

## 160. One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers

**arXiv ID:** 2603.01406 | [PDF](https://arxiv.org/pdf/2603.01406v1)

**作者:** Lennon J. Shikhman `[一作]` `[通讯]` (Georgia Institute of Technology), Lennon J. Shikhman (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在边界条件变化下神经PDE求解器的学习行为，指出它们实际上学习的是以边界为索引的算子族而非边界无关的单一算子

**💡 创新点**

提出将算子学习视为针对边界条件的条件风险最小化，揭示了非可识别性现象，并通过理论和实验验证了该观点

**🔧 技术方法**

使用条件风险最小化框架、Fourier Neural Operator（FNO）架构以及傅里叶截断的合成数据

**📊 数据集**

利用二维Poisson方程的合成数据，边界函数通过有限傅里叶级数生成，forcing函数保持固定

**📈 对比分析**

在边界分布内/外进行对比实验，边界感知FNO在训练分布上误差约0.08，跨分布时误差升至0.49；无边界信息的FNO误差始终≈1；展示了显著的泛化失效

**⚠️ 局限性**

实验仅针对单一椭圆型PDE，未考察时变或非线性方程，理论分析仍为启发性而非严谨保证

---

## 161. TriMoE: Augmenting GPU with AMX-Enabled CPU and DIMM-NDP for High-Throughput MoE Inference via Offloading

**arXiv ID:** 2603.01058 | [PDF](https://arxiv.org/pdf/2603.01058v1)

**作者:** Yudong Pan `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Ying Wang `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

设计并实现了 TriMoE 三域异构架构，联合 GPU‑AMX CPU‑DIMM‑NDP 加速 Mixture‑of‑Experts（MoE）推理；

**💡 创新点**

首次识别并量化“warm experts”，将其分配给 AMX‑CPU，提出瓶颈感知贪婪调度和预测驱动的动态重布局/再平衡，实现高吞吐 MoE 推理的系统性突破；

**🔧 技术方法**

采用 NVIDIA H100 GPU、Intel Xeon（AMX 扩展）CPU、DIMM‑NDP 及 DIMM‑Link、vLLM 与 KTransformers 结合 AMX 计算、EMA 预测、瓶颈感知贪婪调度以及预测驱动的重布局与再平衡技术；

**📊 数据集**

使用 DeepSeek‑V2、Qwen3‑235B‑A22B、GLM‑4.5‑Air 三大 MoE 模型，并基于 LMSys 与 CodeAlpaca 数据集提取真实专家激活轨迹；

**📈 对比分析**

与三种 SOTA offloading 系统（Klotski、Enhanced KTransformers、MoNDE）在大批量（256‑768）推理场景下对比，TriMoE 在 MoE 解码层平均提升 2.12–2.83×，端到端吞吐量提升 2.09–2.78×；

**⚠️ 局限性**

需要专用硬件（GPU＋AMX CPU＋DIMM‑NDP），系统实现复杂；迁移与重布局开销约 3.3%，对低批量或低吞吐场景效果相对有限；依赖 AMX 指令集，硬件兼容性受限。

---

## 162. Scaling Tasks, Not Samples: Mastering Humanoid Control through Multi-Task Model-Based Reinforcement Learning

**arXiv ID:** 2603.01452 | [PDF](https://arxiv.org/pdf/2603.01452v1)

**作者:** Shaohuai Liu `[一作]` (Texas A&M University), Le Xie `[通讯]` (Harvard University)

**通讯引用:** 11912 | [OpenAlex ID](https://openalex.org/A5048219277)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出 EfficientZero-Multitask（EZ‑M）算法，在多任务在线模型驱动强化学习框架下，利用共享物理动力学模型实现多任务控制；

**💡 创新点**

通过任务规模扩展而非样本规模扩展来提升在线学习效率；引入路径一致性、任务嵌入和独立经验回放等机制，解决多任务梯度冲突与数据不平衡问题；

**🔧 技术方法**

基于 EfficientZero‑v2 的模型、Gumbel 搜索、MCTS、路径一致性损失、任务嵌入、动作掩码与观测填充、独立经验回放；

**📊 数据集**

HumanoidBench（Medium 与 Hard 版本），涵盖 9–14 种全身控制任务；

**📈 对比分析**

与多种单任务/多任务、模型无关/模型相关基线（TD‑MPC2、DreamerV3、MH‑SAC、BRC 等）对比，EZ‑M 在 1M 环境步内实现 7/9 或 10/14 任务的最佳或最优成绩，样本效率显著优于对手；

**⚠️ 局限性**

计算开销较大（MCTS 规划成本），对实时高频部署存在延迟挑战；对物理环境相同的假设限制了跨形态或不同动力学任务的迁移能力。

---

## 163. D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping

**arXiv ID:** 2603.01151 | [PDF](https://arxiv.org/pdf/2603.01151v1)

**作者:** Haozhe Lou `[一作]` (University of Southern California), Yue Wang `[通讯]` (University of Southern California)

**通讯引用:** 54717 | [OpenAlex ID](https://openalex.org/A5113600509)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了D-REX框架，实现从真实世界视频和机器人交互中通过可微物理引擎识别物体质量，并利用该质量信息训练力感知的抓取策略；

**💡 创新点**

①将Gaussian Splat与可微仿真相结合，实现高保真视觉与碰撞模型；②通过梯度优化仅使用机器人动作与观测完成质量识别；③将人类演示转化为机器人可执行轨迹，强化力位姿联合控制；

**🔧 技术方法**

可微物理引擎（Brax/GradSim）、Gaussian Splat表示、COLMAP结构光重建、Vision‑Language模型、Dex‑Retargeting、自动微分与半隐式Euler积分；

**📊 数据集**

自制多种形状/密度的物体，收集RGB视频、机器人轨迹以及人类抓取演示；

**📈 对比分析**

与DexGraspNet 2.0和Human2Sim2Robot基线对比，在八种不同形状与质量的桌面抓取任务中，D‑REX取得更高的成功率且方差更低；质量识别实验显示误差低于12%；

**⚠️ 局限性**

仍需大量离线重建时间（30‑35 min/物体），对复杂网格的可微动力学收敛慢；对极端摩擦或柔性物体的适应性有限；

---

## 164. Demonstrating ViviDoc: Generating Interactive Documents through Human-Agent Collaboration

**arXiv ID:** 2603.01912 | [PDF](https://arxiv.org/pdf/2603.01912v1)

**作者:** Yinghao Tang `[一作]` (State Key Laboratory of Computer Aided Design and Computer Graphics), Wei Chen `[通讯]` (State Key Laboratory of Computer Aided Design and Computer Graphics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种人机协作的多代理系统，通过中间表示DocSpec自动生成交互式教育文档。

**💡 创新点**

创新点在于引入DocSpec（含SRTC四要素的交互规范）作为可编辑、可验证的中间层，结合Planner/Executor/Evaluator三代理流水线，实现对生成过程的可控性与可评估性。

**🔧 技术方法**

采用大语言模型（Gemini 3.0 Flash）实现Planner、Executor、Evaluator，利用DocSpec定义的SRTC结构自动合成HTML/CSS/JS交互代码。

**📊 数据集**

使用从60+网站收集的101份真实交互式文档构成的跨域数据集（覆盖11个领域）进行评测与实验。

**📈 对比分析**

与Naive Agent（单次LLM直接生成完整HTML）进行盲评对比，专家在内容丰富度、交互质量、视觉质量三维度分别提升约3.1、1.6、1.36分；用户研究显示界面易学易用，DocSpec编辑符合期望。

**⚠️ 局限性**

局限性包括对LLM生成代码的依赖、缺乏大规模专门数据集、实验样本量有限（仅3位专家与3名受试者），以及中间表示层外的代码质量仍需进一步评估。

---

## 165. Taking a Closer Look at Warnings Generated by PMD and SonarQube, their Rules and Compliance to Established Coding Standards

**arXiv ID:** 2603.00821 | [PDF](https://arxiv.org/pdf/2603.00821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 166. Remember You: Understanding How Users Use Deadbots to Reconstruct Memories of the Deceased

**arXiv ID:** 2603.01017 | [PDF](https://arxiv.org/pdf/2603.01017v1)

**作者:** Yifan Li `[一作]` (Fudan University), Xingyu Lan `[通讯]` (Fudan University)

**通讯引用:** 530 | [OpenAlex ID](https://openalex.org/A5046324646)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对26名Deadbot用户的深入访谈，探讨了用户如何主动构建、重塑和重构逝者记忆，并分析了Deadbot对哀悼实践的影响。

**💡 创新点**

创新点在于将记忆重构视角与用户行为相结合，揭示了用户在与Deadbot互动过程中如何动态构建逝者的数字人格以及记忆的演化过程。

**🔧 技术方法**

主要使用生成式人工智能（大语言模型驱动的Deadbot）与对话式交互平台；研究方法为定性访谈与主题分析。

**📊 数据集**

数据集为26名中国用户的访谈记录，涵盖年龄、性别、使用软件、与逝者关系和使用时长等信息。

**📈 对比分析**

未进行性能对比实验，研究以质性分析为主，未给出量化指标；比较重点在于不同用户群体与不同使用软件对记忆重构的差异。

**⚠️ 局限性**

局限包括样本主要来自线上社交平台，可能偏向技术接受度高者；未能细分关系类型和动机对记忆重构的调节作用；缺乏长期跟踪评估效果。

---

## 167. QCAgent: An agentic framework for quality-controllable pathology report generation from whole slide image

**arXiv ID:** 2603.01647 | [PDF](https://arxiv.org/pdf/2603.01647v1)

**作者:** Rundong Wang `[一作]` (University of Science and Technology of China), S. Kevin Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 QCAgent 框架，能够通过迭代的审核–检索–修订循环，实现基于全切片图像的可控、高质量病理报告生成。

**💡 创新点**

创新点包括将检查列表驱动的质量控制嵌入生成流程、需求驱动的文本引导局部图像检索以及严格的反幻觉设计。

**🔧 技术方法**

主要技术包括预训练的 WSI 基础模型 PRISM、跨模态检索模型 CONCH、视觉语言模型 Patho-R1 以及大型语言模型 Qwen3-VL-30B 进行迭代推理。

**📊 数据集**

使用公开的 TCGA‑STAD 胃癌全切片数据和内部的 524 例中文胃癌 H&E 切片进行实验。

**📈 对比分析**

与单通道 PRISM 基线对比，QCAgent 在 BLEU、ROUGE、METEOR、BERTScore 以及 Field Recall 上均显著提升，特别是 Field Recall 从 32% 提升至 63%。

**⚠️ 局限性**

局限性在于报告长度偏长导致 n‑gram 基准下降，且对多语言的适配仍需专家评估和细化。

---

## 168. Keyword-based Community Search in Bipartite Spatial-Social Networks (Technical Report)

**arXiv ID:** 2603.01500 | [PDF](https://arxiv.org/pdf/2603.01500v1)

**作者:** Kovan A. Bavi `[一作]` (Kent State University), Xiang Lian `[通讯]` (Kent State University)

**通讯引用:** 3901 | [OpenAlex ID](https://openalex.org/A5026993561)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在双向空间-社交网络中基于关键词、影响力、结构凝聚和空间距离的社区搜索查询（KCSBSSN）。

**💡 创新点**

引入了 (ω,π)-关键词核模型并同时考虑 (k,d)-三角核心、用户影响力与平均空间距离的多维约束；同时设计了专门的多级剪枝和索引树提升查询效率。

**🔧 技术方法**

利用多维上界剪枝（关键词、频率、影响力、三角支撑、社交/空间距离）和一棵结合社交、道路与关键词信息的树状索引，实现两阶段过滤-精炼查询算法。

**📊 数据集**

实验使用真实社交网络（Epinions、Twitter、DBLP）与人工生成的混合社交/道路网络（Unif、Gaus、Skew）进行评估。

**📈 对比分析**

与基线方法（仅按社交距离采样子图）比较，KCSBSSN 在多种参数设置下显著降低运行时间，且可扩展到 200K 节点规模。

**⚠️ 局限性**

局限在于需要预先计算多种上界与枢轴，且参数选择（k,d,ω,π,σ,θ）对性能影响较大，未来可考虑自适应或动态调整策略。

---

## 169. A Border Gateway Protocol Extension for Distributing Endpoint Identifier Reachability Information in Delay-tolerant Networks

**arXiv ID:** 2603.01263 | [PDF](https://arxiv.org/pdf/2603.01263v1)

**作者:** Marius Feldmann `[一作]` (D3TN GmbH), Felix Walter `[通讯]` (D3TN GmbH)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出并实现了基于BGP的EID可达性信息分发机制，用于自动配置DTN节点

**💡 创新点**

首次将BGP的多协议扩展与DTN的EID可达性结合，定义新的NLRI和属性，自动化DTN邻居发现与路由配置

**🔧 技术方法**

使用BGP、Bundle Protocol、Convergence Layer Adapter、µD3TN、BIRD、Hermes（Rust实现）以及AAP2接口

**📊 数据集**

未使用公开数据集，采用本地实验环境中的µD3TN与BIRD实例进行验证

**📈 对比分析**

未进行量化性能对比，仅在实验平台上验证功能可行性，未来计划与IS‑IS等协议比较

**⚠️ 局限性**

局限于仅传播当前可达EID，未覆盖时间变化的接触计划；实现仅支持IP与DTN互连；实现仅针对µD3TN/BIRD，尚未标准化或支持QUIC

---

## 170. Heterophily-Agnostic Hypergraph Neural Networks with Riemannian Local Exchanger

**arXiv ID:** 2603.00599 | [PDF](https://arxiv.org/pdf/2603.00599v1)

**作者:** Li Sun `[一作]` (Beijing University of Posts and Telecommunications), Philip Yu `[通讯]` (University of Illinois)

**通讯引用:** 134323 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于Riemannian热流的自适应局部热交换器（Adaptive Local Exchanger），并将其嵌入到超图神经网络中，形成HEALHGNN，实现了对同质与异质超图的统一消息传递；

**💡 创新点**

创新点在于将热流方程与Robin边界条件和外部源项相结合，利用Riemannian几何局部调节超图瓶颈，既缓解了过度压缩（oversquashing）又抑制了过度平滑（oversmoothing），实现了异构性无关的高效信息传播；

**🔧 技术方法**

核心技术包括Riemannian热流建模、Cheeger不等式与谱间隙关联、Robin边界条件的热交换、源项注入保持能量、节点-超边双向耦合及线性复杂度的实现；

**📊 数据集**

实验采用8个真实超图数据集（DBLP-CA、Cora、Cora-CA、Citeseer、PubMed、NTU2012、House、Congress、Senate、Walmart）以及合成异构实验和Long-Range Graph Benchmark（Peptides-func、Peptides-struct）验证模型；

**📈 对比分析**

与多种基线（HGNN、HyperGCN、HNHN、HCHA、UniGCNII、ED-HNN、KHGNN等）在节点分类、标签传递与长距离任务中进行10折平均比较，HEALHGNN在异构超图上提升约5–10%的准确率，深层网络中保持稳定且优于现有方法；

**⚠️ 局限性**

局限性包括对超图结构的先验假设（如超边大小）、源项参数学习复杂且调参繁琐，以及在极大规模数据集上可能需要进一步加速和优化。

---

## 171. Sparse View Distractor-Free Gaussian Splatting

**arXiv ID:** 2603.01603 | [PDF](https://arxiv.org/pdf/2603.01603v1)

**作者:** Yi Gu `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2061 | [OpenAlex ID](https://openalex.org/A5109900808)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出基于 VGGT 与 VLM 的稀疏视图无干扰 3D Gaussian Splatting 的前置掩码生成与温和启动策略。

**💡 创新点**

通过将 VGGT 的注意力匹配与 VLM 的语义推理相结合，实现高质量静态掩码，解决稀疏视图下色彩残差不可靠的问题。

**🔧 技术方法**

VGGT（几何基础模型）、VLM（如 GPT‑4/Claude）、VGGT attention 匹配、VLM 提示、Bundle Adjustment、RobustGS 的温和启动等。

**📊 数据集**

RobustNeRF（5 场景）和 NeRF on‑the‑go（6 场景）数据集。

**📈 对比分析**

与 WildGaussians、SpotLessSplat、DeSplat、RobustGS 等基线相比，RobustGS*+ours 在 PSNR/SSIM/LPIPS 等指标上在大多数场景中提升 1–4 dB，常位列首位。

**⚠️ 局限性**

依赖 VGGT 与 VLM 的性能，纹理稀少或遮挡严重区域仍可能产生误判；未考虑动态场景的时序信息。

---

## 172. Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis

**arXiv ID:** 2603.01398 | [PDF](https://arxiv.org/pdf/2603.01398v1)

**作者:** Junwei Zeng `[一作]` (Nanjing University of Aeronautics and Astronautics), Songcan Chen `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 13709 | [OpenAlex ID](https://openalex.org/A5101596072)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于物理的连续曝光时间建模的大气湍流合成方法，并构建了ET‑Turb大规模合成数据集

**💡 创新点**

创新点在于将曝光时间视为连续变量，推导曝光时间相关的ET‑MTF，并通过空间变异的PSF与随机场实现真实感湍流图像的生成

**🔧 技术方法**

采用Azoulay有限曝光MTF理论、频域到时域的PSF逆变换、空间随机场建模以及Taylor冻结流假设实现视频级同步

**📊 数据集**

使用SVW、TSR‑WGAN等源图像合成5,083段视频（2,005,835帧）作为ET‑Turb，并从多源摄像机中采集74段真实湍流视频构成ET‑Turb‑Real

**📈 对比分析**

通过对比ET‑Turb、TMT‑dynamic、ATSyn‑dynamic三组数据集在真实湍流测试集上的无参考质量指标（NIQE、BRISQUE）和视觉效果，发现ET‑Turb训练的模型在恢复质量上明显优于其他数据集，指标下降约0.2–0.3分

**⚠️ 局限性**

局限性：仍假设湍流为Taylor冻结流，难以模拟极端瞬态湍流；曝光时间范围仅限0.5–40 ms；仅针对光学成像，未考虑多波段或高帧率情境

---

## 173. Type-Based Unsourced Multiple Access Over Fading Channels in Distributed MIMO With Application to Multi-Target Localization

**arXiv ID:** 2603.01749 | [PDF](https://arxiv.org/pdf/2603.01749v1)

**作者:** Kaan Okumus `[一作]` (Chalmers University of Technology), Erik G. Ström `[通讯]` (Chalmers University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了TUMA（基于类型的无源多重接入）框架，用于在分布式MIMO中估计消息的多重度，支持多目标定位等应用。

**💡 创新点**

创新点在于：①在未知CSI的衰落环境下通过位置相关码本分区实现路径损耗一致性；②设计了基于多源AMP的贝叶斯解码器，可同时估计有效信道、消息多重度和类型；③提出分布式实现降低CPU计算量；④将TUMA应用于多目标定位，量化感知、量化与通信的整体权衡。

**🔧 技术方法**

采用的技术包括：分布式MIMO模型、位置相关码本分区、非正交码本、近似消息传递（AMP）与贝叶斯后验均值估计、Monte‑Carlo近似、分布式AMP（dAMP）以及Wasserstein、GOSPA等性能度量。

**📊 数据集**

使用了仿真数据：覆盖区域内随机布置200个传感器、50个目标，3×3网格的40个AP（每个4天线），仿真多种感知、量化比特数和总块长设置。

**📈 对比分析**

与AMP‑DA（需CSI预均衡）和理想通信基线比较，TUMA在未知CSI下仍能获得较低的TV误差和Wasserstein距离，分布式实现与集中式相比仅略逊，但在大规模系统中计算量显著降低；在感知-通信权衡实验中，最佳性能出现在感知与通信资源平衡区间。

**⚠️ 局限性**

局限性在于：①高量化比特数导致消息空间指数增长，AMP解码复杂度难以维持；②Monte‑Carlo近似和最大多重度截断会影响解码精度；③仿真仅在理想化的mmWave感知模型下，未覆盖真实硬件噪声、干扰和多径细节。

---

## 174. Benchmarking Semantic Segmentation Models via Appearance and Geometry Attribute Editing

**arXiv ID:** 2603.01535 | [PDF](https://arxiv.org/pdf/2603.01535v1)

**作者:** Zijin Yin `[一作]` (Beijing University of Posts and Telecommunications), Jun Guo `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 38724 | [OpenAlex ID](https://openalex.org/A5100361885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了自动化数据生成管道 Gen4Seg，用于在真实图像上编辑视觉属性（如颜色、材质、尺寸、位置、天气等），生成可用于评估语义分割模型鲁棒性的合成样本，并基于此构建了 Pascal‑EA 和 COCO‑EA 两个基准集。

**💡 创新点**

首次提出 mask‑guided 能量函数在扩散模型中实现精准局部属性编辑，并结合 ControlNet、VLM 与 LLM 实现无调参的图像生成；提出两阶段噪声过滤策略保证生成样本质量；系统评估语义分割模型在多维属性变化下的鲁棒性，发现 open‑vocabulary 模型并不一定更鲁棒。

**🔧 技术方法**

采用扩散模型（Stable Diffusion）与 Prompt‑to‑Prompt、ControlNet、掩模引导能量函数；利用 VLM（LLaVA）和 LLM（LLaMA3）进行文本编辑；引入两阶段噪声过滤、CLIP 方向相似度、梯度规范化等技术。

**📊 数据集**

以 Pascal VOC 与 COCO‑Stuff 164k 验证集为源图像，生成 Pascal‑EA 与 COCO‑EA 基准集；对比 ACDC、Multi‑weather、Fog Cityscapes、SHIFT、GenVal 等现有合成/仿真数据集。

**📈 对比分析**

对 13 种语义分割模型（CNN、Transformer、open‑vocabulary）在生成的属性变化子集上计算 mIoU，并通过 mR 衡量鲁棒性。结果表明 Transformer 与 open‑vocabulary 模型在 appearance 变化下更稳健，几乎所有模型对几何变化的鲁棒性相近；CutMix 数据增强能提升几何鲁棒性但对 appearance 影响有限；Gen4Seg 生成样本在 FID、LPIPS、DINO‑Dist 等指标上优于现有合成基准。

**⚠️ 局限性**

扩散模型在面部细节、动态主体等细粒度内容上表现欠佳；属性编辑难以完全解耦，改动一个属性往往会影响其他属性；合成数据可能携带训练时的偏见；未覆盖光照、运动模糊等更多属性；方法主要用于诊断而非完整增强方案，需与真实数据结合使用。

---

## 175. A Stochastic Conservative Field Transfer Method for Black-box Multiscale and Multiphysics Coupling

**arXiv ID:** 2603.00538 | [PDF](https://arxiv.org/pdf/2603.00538v1)

**作者:** Abhiyan Paudel `[一作]` (Rensselaer Polytechnic Institute), Jacob S. Merson `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5038517524)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于随机Galerkin投影的黑盒耦合场传递方法，利用Monte Carlo积分实现守恒，且不需要源网格信息。

**💡 创新点**

创新点在于使用随机近似Galerkin投影与Sobol序列采样，得到无源网格、可保守且可在GPU并行执行的字段传递算法。

**🔧 技术方法**

采用Monte Carlo与Sobol序列采样、R3D几何裁剪、邻接搜索、GPU并行化实现，并与传统网格交叉和RBF方法对比。

**📊 数据集**

使用简单几何域、LTX等离子体反应堆网格以及通过Gmsh生成的多级非匹配网格（最高约230万单元）进行验证。

**📈 对比分析**

与网格交叉（MI）和径向基函数（RBF）方法在精度、守恒、迭代误差以及在线/初始化时间上做比较；MC在足够采样时可逼近MI的精度和守恒，在线成本与MI相当，RBF最快但精度和守恒最差。

**⚠️ 局限性**

局限在于方法本质上是随机的，需要大量样本以降低方差，且对高维场的收敛性未作验证。

---

## 176. Practical Deep Heteroskedastic Regression

**arXiv ID:** 2603.01750 | [PDF](https://arxiv.org/pdf/2603.01750v1)

**作者:** Mikkel Jordahn `[一作]` (Technical University of Denmark), Mikkel N. Schmidt `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在已有均值网络上后期拟合线性方差头的异方差回归方法

**💡 创新点**

通过将方差预测器拆分到多个中间层并在留存数据上训练，解决优化、表征崩溃、残差方差过拟合和实用性四大问题

**🔧 技术方法**

使用线性方差头（softplus+线性投影）、中间特征提取、Gaussian负对数似然以及多层融合/集成等技术

**📊 数据集**

在QM9分子回归数据集（使用PaiNN网络）和OMol25大规模分子数据集（使用UMA、AllScAIP模型）上评测

**📈 对比分析**

与传统端到端的均值-方差网络以及β‑NLL、自然NLL等基线相比，后期方差集成在保持均值精度的同时，NLL往往更优或相当，并对hold‑out大小鲁棒

**⚠️ 局限性**

仅适用于可提取中间特征的模型，对层级选择仍需经验调优；在极端OOD检测或需要更尖锐分布的任务中表现略逊

---

## 177. From OCR to Analysis: Tracking Correction Provenance in Digital Humanities Pipelines

**arXiv ID:** 2603.00884 | [PDF](https://arxiv.org/pdf/2603.00884v1)

**作者:** Haoze Guo `[一作]` (University of Wisconsin), Ziqi Wei `[通讯]` (University of Wisconsin)

**通讯引用:** 4873 | [OpenAlex ID](https://openalex.org/A5006792059)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 span‑level 的 OCR 校正 provenance 框架，并在历史文本上进行 pilot 研究，比较不同校正路径对命名实体识别（NER）的影响。

**💡 创新点**

首次将校正来源、编辑类型、置信度与审核状态等 metadata 以 span 级别记录为 provenance，提供可调的覆盖‑稳定性权衡与可审计的文本变更链路。

**🔧 技术方法**

采用 PROV‑O/PROV‑DM 的 schema，JSONL/stand‑off 序列化，结合 Transformer‑based NER 模型（CoNLL‑2003 微调）以及阈值过滤与信任策略。

**📊 数据集**

使用小规模历史文本语料（扫描 + OCR + 手工校正）作为 pilot corpus。

**📈 对比分析**

通过实体计数、独特实体数、Jaccard 相似度和实体波动率等指标比较 Raw OCR、Fully corrected 与 Provenance‑filtered；发现 Provenance‑filtered 在保持大部分覆盖率的同时显著降低了高风险波动，阈值越高稳定性越好但覆盖率下降。

**⚠️ 局限性**

limitation 是样本规模有限、仅评估 NER 任务、置信度分数缺乏统一校准、未覆盖多语种和复杂版式，未来需扩展数据量、任务与跨域验证。

---

## 178. MMTA: Multi Membership Temporal Attention for Fine-Grained Stroke Rehabilitation Assessment

**arXiv ID:** 2603.00878 | [PDF](https://arxiv.org/pdf/2603.00878v1)

**作者:** Halil Ismail Helvaci `[一作]` (University of Kentucky), Sen-ching Samson Cheung `[通讯]` (University of California)

**通讯引用:** 3851 | [OpenAlex ID](https://openalex.org/A5022090297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种多成员时间注意力（MMTA）Transformer，用于细粒度中风康复动作分割，实现更精确的动作边界检测。

**💡 创新点**

通过让每帧参与多个重叠窗口的本地注意力，MMTA保留边界处竞争的局部上下文，消除了全局自注意力的时间粒度瓶颈，无需多阶段细化。

**🔧 技术方法**

基于Transformer编码器的局部窗口注意力、重叠窗口融合、线性复杂度实现，以及使用I3D视觉特征和IMU传感器特征。

**📊 数据集**

在临床中风康复数据集StrokeRehab（视频和IMU）以及公开的50Salads食物准备数据集上进行评估。

**📈 对比分析**

与MS‑TCN、ASRF、Seg2Seq、ASFormer等多种基线以及全局注意力Transformer进行单阶段比较，MMTA在StrokeRehab视频/IMU上分别提升Edit Score +1.3/+1.6、AER下降，50Salads上提升Edit Score至88.4、AER至0.116，显著优于现有方法。

**⚠️ 局限性**

仅使用固定窗口长度和重叠比例，缺乏对不同时间动态自适应的窗口配置，可能在极端运动速率下性能受限。

---

## 179. LLM-Powered Automatic Theorem Proving and Synthesis for Hybrid Systems and Game

**arXiv ID:** 2603.00737 | [PDF](https://arxiv.org/pdf/2603.00737v1)

**作者:** Aditi Kabra `[一作]` (Carnegie Mellon University), André Platzer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5581 | [OpenAlex ID](https://openalex.org/A5080481427)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一套基于大型语言模型（LLM）的自动定理证明与控制合成框架，用于混合游戏（Hybrid Games）与混合系统的形式化验证与控制合成。

**💡 创新点**

创新点在于将LLM与高表达式的差分游戏逻辑（Differential Game Logic）相结合，通过对话式推理与摘要反思机制显著提升了符号证明的自动化程度，突破了传统手动定理证明与现有自动工具无法解决的复杂案例。

**🔧 技术方法**

采用的技术包括：LLM（如OpenAI GPT‑5）驱动的交互式推理循环、证明策略提议与摘要反馈、KeYmaera X和Z3等定理证明器的调用、以及基于子值映射（subvalue maps）的控制合成管道。

**📊 数据集**

实验采用了五个挑战性案例研究，涵盖Lotka‑Volterra种群控制、列车控制系统、化学反应温度安全、核电厂冷却系统、以及Van der Pol振荡器初始条件选择等场景，数据主要来自公开的ARCH‑COMP基准和自建的数学模型。

**📈 对比分析**

与传统定理证明工具相比，该方法在所有五个案例中实现了完整的验证（100%）并在四个案例中完成控制合成；相较于现有自动化工具，仅通过LLM和回溯机制提升了成功率，并在平均LLM调用成本低于30美元的条件下完成任务。

**⚠️ 局限性**

限制包括：对LLM的依赖导致可解释性和可重复性受限；在极其复杂或非线性高维系统中仍可能出现求解失败；以及需要进一步完善摘要策略和错误恢复机制以提升在更大规模问题上的鲁棒性。

---

## 180. AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in Large Vision-Language Models

**arXiv ID:** 2603.01236 | [PDF](https://arxiv.org/pdf/2603.01236v1)

**作者:** Changwoo Baek `[一作]` (Pusan National University), Kyeongbo Kong `[通讯]` (Pusan National University)

**通讯引用:** 475 | [OpenAlex ID](https://openalex.org/A5000238164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统性分析视觉令牌剪枝，使用有效秩和注意力熵评估多样性与幻觉。

**💡 创新点**

首次揭示不同剪枝策略保留的多样性差异及其与幻觉关系，并证明图像复杂度决定最佳策略。

**🔧 技术方法**

有效秩(erank)、注意力熵、阈值自适应剪枝、混合与自适应阈值算法。

**📊 数据集**

VQAv2、GQA、VizWiz、TextVQA、ScienceQA、MME、MMBench、MMBench-CN 及 CHAIR 幻觉评估。

**📈 对比分析**

与 FastV、PDrop、SparseVLM、VisPruner、DivPrune 等基线对比，在 128/64/32 令牌下，自适应方法在多数基准上提升 0.5–2% 准确率，且幻觉率明显下降。

**⚠️ 局限性**

仅在 LLaVA-1.5‑7B 上验证，阈值设定需依赖训练集统计，对极端图像可能仍不稳定。

---

## 181. Characterizing Information Accuracy in Timeliness-Based Gossip Networks

**arXiv ID:** 2603.02197 | [PDF](https://arxiv.org/pdf/2603.02197v1)

**作者:** Emirhan Tekez `[一作]` (Bilkent University), Sinan Gezici `[通讯]` (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究在时效性（基于版本年龄）Gossip网络中节点信息准确度的分布和平均值，提出平均准确度与新鲜度基准准确度两种度量。

**💡 创新点**

创新点包括：① 用SHS框架得到二元CTMC源下的矩阵递推解析式；② 将结果推广到M状态CTMC并通过联合CTMC实现；③ 明确区分由源推送与由Gossip传播导致的准确性，并给出对应的精确表达式。

**🔧 技术方法**

主要技术手段为随机混合系统（Stochastic Hybrid Systems）建模、矩阵递推求解、联合CTMC分析、平衡方程求解。

**📊 数据集**

未使用真实数据集；所有验证均通过对系统进行大规模离散时间仿真（2.25 M 步，10 节点）完成。

**📈 对比分析**

通过仿真比较源推送率 λ_s 与Gossip率 λ 对平均准确度和新鲜度基准准确度的影响，结果显示提升 λ_s 能显著提高准确度，而提升 λ 主要提升同步性但对准确度提升有限。

**⚠️ 局限性**

局限性：仅考虑全连接拓扑；单一源；接受规则仅基于时间戳；未分析网络规模、不同拓扑、异构推送速率或多源情形的影响。

---

## 182. OmniRet: Efficient and High-Fidelity Omni Modality Retrieval

**arXiv ID:** 2603.02098 | [PDF](https://arxiv.org/pdf/2603.02098v1)

**作者:** Chuong Huynh `[一作]` (University of Maryland), Abhinav Shrivastava `[通讯]` (University of Maryland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种统一的多模态检索模型 OmniRet，能够同时处理文本、视觉和音频三种模态的查询与目标。

**💡 创新点**

创新点包括：1）共享注意力重采样器（Shared Media Resampler）实现高效的多模态序列压缩并保持模态特异性；2）Attention Sliced Wasserstein Pooling（ASWP）通过可学习参考点和投影实现高保真单向量聚合；3）构建了全新的 Audio-Centric Multimodal Benchmark，填补了复合音频检索与音频-视觉检索空白。

**🔧 技术方法**

核心技术为：使用 GTE‑Qwen2‑1.5B‑Instruct 作为跨模态 LLM；Perceiver 结构的共享重采样模块；ASWP 聚合方法；LoRA + 3 倍多任务训练；对比损失、三元组损失和多样性正则化共同优化。

**📊 数据集**

数据集涵盖约 6.2M 查询-目标对，来源于 30 个公开数据集（如 MSMarco、HotpotQA、Charades、WebVid2M、AudioCaps、ClothoV2.1、VGGSound 等），并扩展了 M‑BEIR 与 MMEBv2，另外自行构造了 Audio‑Centric Multimodal Benchmark。

**📈 对比分析**

在 13 个检索任务、MMEBv2 子集以及自建音频基准上，OmniRet 在复合查询、音频检索等方面均实现了显著提升，平均 Recall@5 通常领先同规模基线（如 CLIP、CLAP、ImageBind、VLM2VecV2 等），并在视频和音频任务上取得最优或接近最优表现。

**⚠️ 局限性**

主要局限包括：模型规模与训练数据受限，未使用更大 LLM；仅扩展模态而未覆盖更多任务；音频基准仍可进一步细化；进一步扩大数据量和模型规模预期能进一步提升性能。

---

## 183. Towards Policy-Adaptive Image Guardrail: Benchmark and Method

**arXiv ID:** 2603.01228 | [PDF](https://arxiv.org/pdf/2603.01228v1)

**作者:** Caiyong Piao `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 11247 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种两阶段的安全对齐框架 SafeGuard‑VL，首先利用自回译式的 SFT 学习对不安全视觉内容的细粒度语义描述，然后通过基于可验证奖励的强化学习（RLVR）实现对任意自然语言安全政策的动态判别。

**💡 创新点**

创新点包括：① 构造了跨政策评估基准 SafeEditBench，使用细粒度图像编辑保持视觉语义一致的安全‑不安全图像对，真正检验模型对不同安全政策的泛化；② 引入 RLVR，通过策略条件奖励直接对政策进行优化，克服传统 SFT 对固定标签的过拟合；③ 采用自回译式的 caption 生成方式在 SFT 阶段注入细节化不安全语义，保持模型的通用描述能力。

**🔧 技术方法**

核心技术包括：多模态自回译式 caption 生成与修订、基于 LlavaGuard 的 SFT 数据构造、GRPO 强化学习算法以及可验证奖励设计。

**📊 数据集**

主要数据集：LlavaGuard 的训练集与测试集，用 Nano Banana 等图像编辑模型构造 SafeEditBench（128 对安全‑不安全图像），以及公开的 UnsafeBench、LlavaGuardBench 等安全评测集。

**📈 对比分析**

与多种基准对比：在 SafeEditBench 上，SafeGuard‑VL‑Full 在 5 个政策的宏平均 F1 最高达 72.2%；在 UnsafeBench 上取得 72.16%（高于 Qwen2.5‑VL‑7B 的 41.7% 和 QwenGuard‑7B 的 62.4%）；在 LlavaGuardBench 上取得 71.78% 的得分，显著低于 QwenGuard‑7B 的 84.57% 但整体性能更平衡，兼顾安全与通用 QA。

**⚠️ 局限性**

局限性：模型仍依赖于训练时的政策描述，极端或极端对立的政策仍可能导致性能急剧下降；RL 训练对奖励设计和采样分布敏感，需额外调参；目前仅针对视觉安全，未覆盖文本或音频的跨模态安全场景。

---

## 184. MetaMind: General and Cognitive World Models in Multi-Agent Systems by Meta-Theory of Mind

**arXiv ID:** 2603.00808 | [PDF](https://arxiv.org/pdf/2603.00808v1)

**作者:** Lingyi Wang `[一作]` (Virginia Tech), Naren Ramakrishna `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出MetaMind，一种基于Meta-ToM的多智能体世界模型，利用自监督逆推和类比推断实现对他人信念与目标的零射推断，并在无通信、无中心化的环境中实现长期规划；

**💡 创新点**

创新点包括：1）自监督Meta-ToM框架，实现对自身及他人意图的逆推与自我反思；2）循环一致性正则提升逆推可辨识性；3）类比推断将第一人称逆推迁移到第三人称，支持零射泛化；4）集合信念聚合生成可交换的多智能体交互表示；5）理论证明目标识别的可行性与所需观测窗口；

**🔧 技术方法**

技术主要包括：目标条件TD-MPC2世界模型、逆推网络Ψ与Ω、循环一致性正则、Transformer聚合的集合信念、MPC规划、以及在SMAC环境中的训练与评估；

**📊 数据集**

使用StarCraft Multi-Agent Challenge（SMAC）数据集，共13张地图，涵盖同质与异质团队；

**📈 对比分析**

与DCWMC、MARIE、MAPPO、MBVD等基线对比，MetaMind在200k步内平均win rate提升约54%，在少样本泛化任务中提升2.4倍；对不同想象时长、模型规模和多地图训练的评估均显示性能稳健；

**⚠️ 局限性**

局限性在于：长时间想象会因模型误差累积导致性能下降；目标识别依赖足够的轨迹长度和模型容量；在极大规模或高度动态的多智能体环境中仍需进一步验证；

---

## 185. Boosting AI Reliability with an FSM-Driven Streaming Inference Pipeline: An Industrial Case

**arXiv ID:** 2603.01528 | [PDF](https://arxiv.org/pdf/2603.01528v1)

**作者:** Yutian Zhang `[一作]` (Tsinghua University), Jianmin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 38042 | [OpenAlex ID](https://openalex.org/A5100373517)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于对象检测+有限状态机的流式推理管线，用于自动计数挖掘机工作负载。

**💡 创新点**

将业务知识（操作流程）通过有限状态机编码，与检测结果结合，减少误检和漏检，避免手工规则过拟合。

**🔧 技术方法**

使用YOLOv9进行目标检测，设计事件与状态迁移规则，并构建FSM驱动的业务逻辑；同时采用数据增强、时序采样等数据工程技术。

**📊 数据集**

训练集约1.6万张图像，评估集约7千张来自12个施工现场的视频，覆盖300+个完整工作负载。

**📈 对比分析**

与基于手工规则的基线对比，F1提升约2%，精度提升至0.96，召回略降；在误工作负载和漏工作负载方面表现更稳定。

**⚠️ 局限性**

局限在特定任务/场景，FSM需要人工设计，依赖检测模型的准确性；对光照、遮挡等极端情况仍可能出现误检；可扩展性和跨域迁移需进一步验证。

---

## 186. Generative Visual Chain-of-Thought for Image Editing

**arXiv ID:** 2603.01893 | [PDF](https://arxiv.org/pdf/2603.01893v1)

**作者:** Zijin Yin `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Generative Visual Chain-of-Thought (GVCoT)框架，实现图像编辑的视觉中间推理。

**💡 创新点**

创新点在于生成式视觉推理取代外部工具，实现端到端的视觉空间推理与编辑。

**🔧 技术方法**

使用扩散模型、Bagel统一模型、强化学习、视觉工具SAM2等技术。

**📊 数据集**

构建1.8M样本的GVCoT-Edit-Instruct数据集。

**📈 对比分析**

与多种编辑模型对比，GVCoT在SREdit-Bench和ImgEdit上显著超越基线，整体分数提升约0.8。

**⚠️ 局限性**

主要局限在于对全局编辑任务的表现有限，且对极端复杂场景的鲁棒性待提升。

---

## 187. Hermes: A Unified High-Performance NTT Architecture with Hybrid Dataflow

**arXiv ID:** 2603.01556 | [PDF](https://arxiv.org/pdf/2603.01556v1)

**作者:** Hang Gu `[一作]` (University of Science and Technology of China), Xuehai Zhou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3608 | [OpenAlex ID](https://openalex.org/A5077322091)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文提出并实现了 Hermes，一种统一的、高性能的 NTT 加速器，能够在同一硬件架构下支持多种多项式长度，特别适用于混合同态加密（HHE）场景。

**💡 创新点**

创新点包括：① 混合数据流（hybrid dataflow）在时空两维同时并行，突破传统阶段式与流水式架构的局限；② 冲突无关的片上碎片化算法，消除银行冲突并实现 HBM 的 burst 访问；③ 高效的数据重用流设计，显著提升计算密度并降低带宽需求。

**🔧 技术方法**

主要技术手段包括：基于 FPGA 的可重构加速，Shoup 模块化乘法、预计算 Twiddle 表、全流水线 Butterfly 单元、片上 SRAM/URAM 片段化布局、混合数据流调度算法。

**📊 数据集**

实验使用多组多项式长度（N = 2⁸~2¹⁶）和对应的 128‑bit 安全级别参数（来自 HE Security Standard White Paper），并通过 OpenFHE 生成 Twiddle 系列。

**📈 对比分析**

对比方法：在同一 Xilinx U280 FPGA 及 Nvidia V100/A100 GPU 上实现同等功能；相对 FPGA 现有方案 FAB 与 Trinity 进行资源与吞吐率对比。结果显示 Hermes 在 N = 2¹⁶ 时，GPU 方案提升 13.6×，FPGA 方案提升 1.3×，同时保持较低的 HBM 带宽占用。

**⚠️ 局限性**

局限性：① 需要较大片上 RAM（URAM）支持 2¹⁶ 长度，可能受限于更大规模的同态加密需求；② 资源消耗（尤其 DSP）相对较高，可能限制在更小的 FPGA 或 ASIC 设计中；③ 目前未针对动态参数切换进行在线重配置优化，仅支持预设长度。

---

## 188. Scaling of learning time for high dimensional inputs

**arXiv ID:** 2603.01184 | [PDF](https://arxiv.org/pdf/2603.01184v1)

**作者:** Carlos Stein Brito `[一作]` `[通讯]`, Carlos Stein Brito

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 Hebbian 学习（用于独立成分分析）在高维输入下的学习时间随输入维度的增长规律，解析了优化面上的极值点和鞍点分布，并将高维学习动力学降维为一维系统来计算学习时间。

**💡 创新点**

提出学习时间与输入维度呈超线性（对称分布为 N³/ln(K)²，对非对称分布为 N²/ln(K)）的理论预测，并证明随机初始权重在高维空间几乎正交导致学习梯度小、学习时间变长，从而揭示了高维学习的根本瓶颈。

**🔧 技术方法**

使用非线性 Hebbian 学习规则、梯度下降、中心极限定理、极值统计、极值点计数以及一维动力学推导，结合数值模拟验证理论。

**📊 数据集**

采用合成数据：对 N 维输入做白化线性混合，隐藏变量为对称 Laplace 分布或非对称 χ² 分布，K=N（或 K≠N 的情况）来测试理论。

**📈 对比分析**

通过对比理论预测与数值模拟的学习时间标度，发现实验曲线与 N²/ln(K) 或 N³/ln(K)² 的拟合高度一致，证明了理论的正确性。

**⚠️ 局限性**

局限性：假设输入是线性投影，隐藏特征独立且分布为对称或非对称稀疏；仅考虑随机初始化、单层神经元；不涵盖卷积网络、递归网络或有约束的生物网络的非线性特性；在真实大规模数据集上的适用性尚未验证。

---

## 189. Search Multilayer Perceptron-Based Fusion for Efficient and Accurate Siamese Tracking

**arXiv ID:** 2603.01706 | [PDF](https://arxiv.org/pdf/2603.01706v1)

**作者:** Tianqi Shen `[一作]` (Chinese Institute of Coal Science), Ning An `[通讯]` (Chinese Institute of Coal Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文设计并实现了一种基于MLP的Siamese视觉跟踪器SEAT，采用Coarse Fusion MLP和Refine Fusion MLP实现像素级特征融合，并通过层次化MCAS+Harmony‑Relaxation实现高效的可微NAS搜索。

**💡 创新点**

创新点在于：①提出简单有效的两阶段MLP融合框架，实现像素级融合；②构建层次化MCAS搜索空间并通过Harmony‑Relaxation实现通道宽度与其他超参数的分离优化，首次在Siamese neck实现高效像素级融合且兼顾精度与速度。

**🔧 技术方法**

使用技术包括多层感知机（MLP）与Wave‑MLP、差分可微NAS（DARTS）+Harmony‑Relaxation、双阶段训练策略、交叉熵与IOU损失等。

**📊 数据集**

训练集使用COCO、YouTube‑BB、GOT‑10k、ImageNet DET/VID等，评估集涵盖通用跟踪基准GOT10K、OTB2015、VOT2019、NFS30以及空中跟踪基准UAV123、UAVDT、VISDRONE。

**📈 对比分析**

采用成功率/面积/EAO/FPS等指标与LightTrack、FEAR、E.T.Track等轻量级GPU跟踪器以及SiamAPN++、HiFT、TCTrack等NPU跟踪器对比，SEAT_LT在GPU上达到69.7%成功率、34%EAO、80.5 FPS，SEAT_AL在NPU上实现20.4 FPS、0.593精度，均优于同类方法。

**⚠️ 局限性**

局限性包括对光照变化仍敏感，Wave‑MLP通道数仍决定算力上限，且模型仍无法进一步压缩；未来需要实现一阶段统一设计并在更高算力环境下进一步提升性能。

---

## 190. Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards

**arXiv ID:** 2603.00918 | [PDF](https://arxiv.org/pdf/2603.00918v1)

**作者:** Seungwook Kim `[一作]` (Pohang University of Science and Technology), Minsu Cho `[通讯]` (RLWRLD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ARC框架，利用模型自身的自置信度作为奖励，对文本到图像生成模型进行无监督后训练。

**💡 创新点**

创新点在于用内部自置信度（通过重新噪声并测量恢复误差）替代外部奖励，既不需要标注数据也不依赖额外评估器，且与外部奖励互补能缓解奖励劫持。

**🔧 技术方法**

技术包括Flow-Matching文本图像生成器、Flow‑GRPO强化学习、LoRA参数高效微调、抗性噪声对偶、CFG、KL正则化以及采样子序列技术。

**📊 数据集**

主要使用OCR（文本渲染）提示集进行训练，并对比PickScore、GenEval提示；评估集包括GenEval、OCR、PickScore、HPSv2、ImageReward、UnifiedReward、CLIP‑Score、Aesthetic Score、DrawBench等。

**📈 对比分析**

通过与基线SD3.5‑M、SD3.5‑L及外部奖励（Flow‑GRPO）后训练模型对比，ARC在GenEval、OCR和CLIP‑Score等指标提升约0.05‑0.07，接近大模型水平；人类偏好得分略有提升；用户研究表明视觉逼真度和文本对齐均优于基线。

**⚠️ 局限性**

局限性包括对人类偏好的提升有限、无法单独针对特定属性进行微调、训练过程中易出现过度优化导致生成崩溃，需要通过限制步长比例和CFG等手段进行平衡。

---

## 191. RA-Det: Towards Universal Detection of AI-Generated Images via Robustness Asymmetry

**arXiv ID:** 2603.01544 | [PDF](https://arxiv.org/pdf/2603.01544v1)

**作者:** Xinchang Wang `[一作]` (Jiangnan University), Hui Li `[通讯]` (Jiangnan University)

**通讯引用:** 38306 | [OpenAlex ID](https://openalex.org/A5065859286)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于鲁棒性不对称的生成图像检测框架RA-Det。

**💡 创新点**

将鲁棒性不对称作为通用行为信号，并通过可学习的扰动探测器显著放大真实与伪造图像的差异。

**🔧 技术方法**

采用冻结的基础编码器（如DINOv3/CLIP）、条件UNet扰动生成、多分支融合与对比损失等技术。

**📊 数据集**

使用360k真实与360k伪造图像（ProGAN）训练，并在16种不同生成器上进行评估。

**📈 对比分析**

与10+基准方法对比，RA-Det平均准确率93.47%，AP 97.00%，显著优于最强通用检测器。

**⚠️ 局限性**

对抗或极端后处理（如强噪声、攻击）以及不同视觉任务的鲁棒性仍需进一步验证。

---

## 192. Subliminal Signals in Preference Labels

**arXiv ID:** 2603.01204 | [PDF](https://arxiv.org/pdf/2603.01204v1)

**作者:** Isotta Magistrali `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21287 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在 LLM‑as‑a‑Judge 框架下，二进制偏好标签是否能作为潜在通道向中立学生模型传递偏好信息；实验通过让学生生成数字序列，由偏好评判员生成偏好数据集，随后使用 SFT/DPO 对齐，并在迭代对齐中观察信号放大。

**💡 创新点**

证明即使只传递一比特的偏好标签，偏好评判员也能通过隐式信道影响中立学生模型的行为，且该效应在多轮对齐中可逐步增强，挑战了“偏好仅为语义反馈”的假设。

**🔧 技术方法**

使用 Qwen 2.5 7B 作为学生与评判员，构建偏好数据集；采用监督微调（SFT）与 Direct Preference Optimization（DPO）进行对齐；进行迭代对齐并通过多选问答评估模型偏好。

**📊 数据集**

实验数据主要包括自定义数字序列生成的候选完成、评判员产生的偏好对比数据集，以及用于评估的多选问答数据集（目标动物：猫、狮、熊猫，干扰动物：凤凰、企鹅）。

**📈 对比分析**

通过对比正常对齐与交换对齐的偏好分布，以及对照实验，发现正常对齐模型对目标动物的偏好显著高于对照，交换对齐则低于对照；在迭代对齐后，偏好差异和胜率进一步增强，表明信号随对齐迭代而增强。

**⚠️ 局限性**

在小型模型上效果相对弱化，跨目标动物的偏好一致性不稳定；实验仅限于数字序列和单一目标动物，未验证更大规模模型或更复杂任务；需进一步研究检测与缓解机制。

---

## 193. One-Token Verification for Reasoning Correctness Estimation

**arXiv ID:** 2603.01025 | [PDF](https://arxiv.org/pdf/2603.01025v1)

**作者:** Zhan Zhuang `[一作]` (Southern University of Science and Technology), Yu Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 41407 | [OpenAlex ID](https://openalex.org/A5112212826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 One‑Token Verification (OTV)，一种在大语言模型（LLM）内部通过单个特殊“truth” token 触发 LoRA 适配器，利用 KV cache 进行单向前向推断，从而在生成过程中以 token 级别即时估计推理路径的正确性。

**💡 创新点**

创新点：
- 通过单个 token 触发专门的验证路径（LoRA 加权），不影响原始推理流程；
- 直接在 KV cache 上进行多层注意力计算，获取完整前缀信息；
- 采用线性 ramp 的伪置信度标签实现无监督训练；
- 支持并行化查询（一次前向可得所有位置的置信度），显著降低验证开销。

**🔧 技术方法**

技术细节：LoRA 低秩适配器、KV cache 取样、单 token 触发、三层感知机回归头、线性伪置信度标签、并行多位置查询、基于置信度的早停和剪枝（Drop@10、Stop@600、Halve@300）等。

**📊 数据集**

使用数据集：MetaMathQA（训练），GSM8K（测试），DAPO‑17K（训练），AIME24 与 AIME25（测试）。

**📈 对比分析**

对比方法：内部验证器 DeepConf、GenRM；外部奖励模型 AceMath‑RM‑7B、VersaPRM、Math‑Shepherd‑7B、Qwen2.5‑PRM 等。实验表明：
- 在 AIME 任务中，OTV 在加权多数投票、Best‑of‑N 以及各类高效剪枝变体上均取得最高或相近最高的准确率；
- 相比传统方法，OTV 在同等生成预算下提升 5–15% Pass@k，且在早停策略下可减少 70–90% token 消耗，保持或提升最终准确率。

**⚠️ 局限性**

局限性：
- 需要针对每个基础 LLM 训练 LoRA 适配器；
- 伪置信度标签基于线性 ramp，可能对不同任务或推理长度不够鲁棒；
- 仅在可访问 KV cache 的 Transformer 架构上有效；
- 主要验证于数学推理任务，对通用文本推理的泛化仍需进一步研究。

---

## 194. SSMG-Nav: Enhancing Lifelong Object Navigation with Semantic Skeleton Memory Graph

**arXiv ID:** 2603.01813 | [PDF](https://arxiv.org/pdf/2603.01813v1)

**作者:** Haochen Niu `[一作]` (Shanghai Jiao Tong University), Fei Wen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了基于语义骨架记忆图（SSMG）的终身多模态目标导航框架SSMG-Nav，能够将历史观测持续地聚合到以拓扑关键点为锚的记忆图中，并利用多模态VLM推断目标概率，随后通过长程规划器最小化期望路径长度，显著提升终身多模态导航效率。

**💡 创新点**

创新点在于：1）构建语义骨架记忆图，将对象与空间语义聚合到拓扑关键点上，形成可重用的、稀疏的候选目标子图；2）利用多模态提示结合VLM进行目标信念推断，支持图像、文本、类别三种目标描述；3）设计基于信念与行进成本的长程规划器，避免贪婪策略导致的回环与无效走动。

**🔧 技术方法**

使用的技术包括：视觉深度感知与BEV投影、BLIP‑2与Grounding‑DINO、MobileSAM实例分割、语义骨架提取与拓扑构造、VLM（如Qwen‑VL‑Plus）推理、基于软最大化的概率化、2‑opt启发式长程路径规划、以及A*+VER的局部导航。

**📊 数据集**

实验使用了GOAT‑Bench（lifelong multimodal）与HM3D、MP3D（经典单模态）等公共数据集，分别提供多场景、多子任务的评测环境。

**📈 对比分析**

在GOAT‑Bench上，SSMG‑Nav在s‑SR、e‑SR和SPL上均超过所有基线（RL、记忆自由与记忆有状态的零样本方法），SPL提升显著；在标准对象导航任务中也实现了SPL领先且SR相当于最强零样本方法。

**⚠️ 局限性**

局限性包括：1）对大型模型的依赖导致推理延迟和低运行速度；2）目前无法处理多楼层或高度差异的环境；3）缺乏对多目标并行任务的完整支持。

---

## 195. Radiometrically Consistent Gaussian Surfels for Inverse Rendering

**arXiv ID:** 2603.01491 | [PDF](https://arxiv.org/pdf/2603.01491v1)

**作者:** Kyu Beom Han `[一作]` (Korea Advanced Institute of Science and Technology), Sung-eui Yoon `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于高斯点云的逆渲染框架 RadioGS，通过自监督的辐射一致性损失使得高斯表面在未观测视角下也能准确估计间接光照，从而实现更精确的材质与光照分离。

**💡 创新点**

创新点在于：①提出辐射一致性损失，利用物理渲染约束高斯表面在未观测方向上的辐射；②将该约束与二维高斯射线跟踪高效结合，形成 RadioGS；③设计快速微调的重光照方法，使得光照变化时仅需数分钟即可得到实时(<10 ms)的重渲染。

**🔧 技术方法**

使用技术包括：高斯点云/高斯表面（Gaussian Splatting）、二维高斯射线跟踪、物理基础渲染（PBR）与蒙特卡洛积分、神经辐射场（NeRF）框架以及深度图/法线一致性损失。

**📊 数据集**

实验使用了两个合成基准数据集：TensoIR 和 Synthetic4Relight，并在 Stanford‑ORB 实景数据上做重光照可视化。

**📈 对比分析**

与 GS‑IR、GI‑GS、R3DG、IRGS、SVG‑IR 以及 TensoIR 等基线对比，RadioGS 在新视角合成（PSNR/SSIM）、法线重建、颜色重建和重光照任务上均取得了更高的评估指标（PSNR 提升 ~1–2 dB，SSIM >0.97，重光照误差下降约30%），同时保持训练时长约1 小时，单帧渲染时长 <10 ms。

**⚠️ 局限性**

局限性：目前仅支持介电材质，无法直接处理高度反射或各向异性表面；对极端光照条件或复杂遮挡的鲁棒性还有待进一步提升。

---

## 196. Historian: Reducing Manual Validation in APR Benchmarking via Evidence-Based Assessment

**arXiv ID:** 2603.00649 | [PDF](https://arxiv.org/pdf/2603.00649v1)

**作者:** Sahand Moslemi `[一作]` (Bilkent University), Anil Koyuncu `[通讯]` (Bilkent University)

**通讯引用:** 990 | [OpenAlex ID](https://openalex.org/A5070518954)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Historian 框架，用大语言模型对新生成的补丁与历史验证补丁进行多引用语义比较，从而自动判定补丁是否正确，显著减少人工验证工作量。

**💡 创新点**

创新点在于把补丁验证转为基于历史证据的可追溯推理，利用多引用基线与两阶段推理逻辑，实现可解释且高精度的补丁评估。

**🔧 技术方法**

核心技术包括大语言模型（如 Qwen、CodeLlama、Gemini 2.0 Flash）进行语义相似度与等价性推理、正则/零样本文本解析、两阶段证据推理（Pairwise Inference + Majority Voting）以及多引用知识库构建。

**📊 数据集**

使用了 Defects4J 基准的 1,455 个正确补丁和 37,858 个过拟合补丁，涵盖 22 种 APR 工具生成的补丁，此外还对 TBar、TBar‑O 等工具的补丁进行实验。

**📈 对比分析**

在 22‑fold leave‑one‑tool‑out 评估中，Historian 的覆盖率达到 95.0%、覆盖集准确率 88.4%，并可将传统 APCA 工具的准确率提升至 21.8% 以上，混合管线实现 86.2% 的整体准确率。

**⚠️ 局限性**

局限性包括：依赖大语言模型的推理质量，模型可能产生误判；仅在 Java 生态中验证，跨语言泛化尚待探索；对历史补丁记录的依赖意味着新颖补丁仍需人工审核；以及对 LLM 参数与提示的敏感性需进一步稳定。

---

## 197. Efficient Test-Time Optimization for Depth Completion via Low-Rank Decoder Adaptation

**arXiv ID:** 2603.01765 | [PDF](https://arxiv.org/pdf/2603.01765v1)

**作者:** Minseok Seo `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对稀疏深度补全问题，提出了一种仅对解码器低秩子空间进行测试时优化的高效方法，使模型在不需要重训练的前提下实现高质量的稠密深度预测。

**💡 创新点**

创新点包括：① 通过层级相关性与PCA分析发现解码器早期层的低维子空间携带大部分深度信息；② 仅对该低维子空间使用LoRA进行自适应，显著降低计算量与参数量；③ 在测试时加入线性尺度平移对齐，解决稀疏深度的尺度不一致问题。

**🔧 技术方法**

技术手段：解码器低秩LoRA自适应、线性尺度平移对齐、PCA与层级相关性分析、稀疏深度监督的损失函数。

**📊 数据集**

实验使用了五个公开数据集：IBims-1、VOID、NYUv2、KITTI Depth Completion（KITTI-DC）和DDAD，涵盖室内外、不同分辨率与深度稀疏程度。

**📈 对比分析**

与现有零射频深度补全方法（如Marigold-DC、TestPromptDC、PromptDA）以及训练型方法进行对比。结果显示，本文方法在MAE/RMSE上达到或超过最佳零射频方法，同时在推理速度上比其他测试时优化方法快约4–10倍，创立了新的Pareto前沿。

**⚠️ 局限性**

局限性：仍需在推理时进行参数更新，无法实现严格的实时性能；方法目前仅针对基于估计的深度基础模型，无法直接迁移到基于扩散的生成模型；对视频序列的适用性尚未验证。

---

## 198. Structural Hallucination in Large Language Models: A Network-Based Evaluation of Knowledge Organization and Citation Integrity

**arXiv ID:** 2603.01341 | [PDF](https://arxiv.org/pdf/2603.01341v1)

**作者:** Moses Boudourides `[一作]` (Northwestern University), Moses Boudourides `[通讯]` (Northwestern University)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5035035192)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对LLM生成的学术知识表示，提出并评估结构性幻觉，使用网络分析的压测方法对Roget词典、Wikidata哲学家及Dimensions.ai引用进行结构完整性评估。

**💡 创新点**

首次正式定义结构性幻觉并构造可复现的网络级压测工具，展示LLM在保持概念组织、关系架构和文献引用方面的系统失真。

**🔧 技术方法**

知识图谱提取、图相似度测量、中心性比较、节点集合Jaccard、引用完整性验证等网络与文本分析技术。

**📊 数据集**

Roget 1911版词典、Wikidata哲学家数据集、Dimensions.ai 维度计量学引用记录。

**📈 对比分析**

通过将LLM生成的知识图与权威图进行节点/边比对、中心性对齐和引用核查，发现宏观F1<0.05、结构性幻觉率>93%、引用遗漏91.9%，表现极差。

**⚠️ 局限性**

仅覆盖三种典型领域，依赖已有知识图准确性；压测对LLM提取过程的依赖未充分评估；未考虑多模态或其他类型幻觉。

---

## 199. Tracking Capabilities for Safer Agents

**arXiv ID:** 2603.00991 | [PDF](https://arxiv.org/pdf/2603.00991v1)

**作者:** Martin Odersky `[一作]` (École Polytechnique Fédérale de Lausanne), Cao Nguyen Pham `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5107187319)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 Scala 3 捕获检查的安全“安全套装”，让 AI 代理通过生成受类型约束的代码来调用工具，防止信息泄露和恶意副作用。

**💡 创新点**

创新点在于将可追踪的能力（capabilities）嵌入静态类型，结合本地纯度和安全模式，提供编译时保证，而非仅靠模型对齐或运行时监控；实现了无安全风险的可执行代码生成。

**🔧 技术方法**

使用 Scala 3 的捕获检查、可追踪类型、Safe Mode、能力安全库以及分类（Classified）包装器；通过 MCP 服务器和 Scala REPL 执行生成的代码。

**📊 数据集**

评测采用 AgentDojo 安全基准、τ^2‑bench（客户服务对话）和 SWE‑bench Lite（GitHub 问题修复）等公开数据集。

**📈 对比分析**

与传统工具调用接口对比，使用安全套装的代理在 τ^2‑bench 上表现相当或略优（+0.8~3.7 %），在 SWE‑bench 上略有下降（≈1 %）；在安全性测试中，受限模式下两大模型均实现 100 % 防泄漏，非受限模式依赖模型对齐。

**⚠️ 局限性**

局限性包括：不保证逻辑正确性、无法阻止侧信道攻击、对外部命令安全依赖于白名单、只适用于 Scala 3，需手工构建能力库，且模型仍需生成符合类型约束的代码。

---

## 200. Forgetting is Competition: Rethinking Unlearning as Representation Interference in Diffusion Models

**arXiv ID:** 2603.00975 | [PDF](https://arxiv.org/pdf/2603.00975v1)

**作者:** Ashutosh Ranjan `[一作]` (TCS Research), Murari Mandal `[通讯]` (Kalinga Institute of Industrial Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SurgUn，针对文本到图像扩散模型的手术式无学习方法，能够精准删除指定视觉概念，同时保持其它能力不受影响。

**💡 创新点**

创新点包括：① 将认知心理学的逆向干扰理论映射到扩散模型，通过固定多样化干扰概念诱导表示竞争；② 设计基于干扰概念的损失函数；③ 基于像素空间表现的权重空间定位，实现局部化更新；④ 使用多准则决策（COMET）对检查点进行校准。

**🔧 技术方法**

使用技术包括：潜在扩散模型（Stable Diffusion v1.5、SDXL）、流匹配与扩散变换器（SANA）、CLIP语义距离、像素级诊断与定位、分层注意力块更新、MCDM（COMET）检查点选择。

**📊 数据集**

实验数据集：UnlearnCanvas（20对象+50艺术风格）、IP角色基准（10知名角色）、Holistic Unlearning Benchmark、EraseBench、Ring‑A‑Bell、COCO用于干扰概念抽取；并在 Stable Diffusion、SDXL、SANA 等模型上验证。

**📈 对比分析**

与 ESD、SalUn、UCE、SPM、MACE、RECE、ACE、CA、AdvUnlearn 等方法对比，SurgUn 在目标消除精度（UA）和保持度（IRA、CRA、SC、OC、UP）上均领先；在攻击成功率、连续/层次化/组合式无学习任务中表现更稳健，且生成的艺术质量和 CLIP 得分更高。

**⚠️ 局限性**

局限性：① 需要额外的像素诊断和定位计算，可能对极大模型开销较大；② 干扰概念集为固定且有限，可能不覆盖所有目标概念；③ 对高度相似概念的完全抑制仍存在一定风险；④ 依赖检查点校准，若参数选择不当仍可能出现过度或不足抑制。

---

## 201. Agentic Multi-Source Grounding for Enhanced Query Intent Understanding: A DoorDash Case Study

**arXiv ID:** 2603.01486 | [PDF](https://arxiv.org/pdf/2603.01486v1)

**作者:** Emmanuel Aboah Boateng `[一作]` (DoorDash), Sudeep Das `[通讯]` (DoorDash)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于代理多源定向的查询意图理解系统，能够在多类别电商/外卖平台上准确解析含糊查询，支持多意图输出并通过可插拔的消歧层确定最终类别。

**💡 创新点**

①将LLM推理与平台自身目录实体及实时网络搜索结果联合定向；②使用双意图（primary + secondary）预测而非单标签，配合历史赢率的配对规则实现消歧；③解耦预测与消歧，支持后续个性化扩展。

**🔧 技术方法**

检索增强型LLM推理（使用向量检索+模糊匹配获取目录实体），外部网页搜索工具（agentic tool），大语言模型（Gemini‑2.5‑flash、GPT‑4o等），双意图输出与规则式消歧模块。

**📊 数据集**

DoorDash 多类别搜索日志，约30,000条查询，覆盖品牌、零售、长尾合成与整体流量四个基准。

**📈 对比分析**

与传统混合 BERT+LLM 系统、未定向LLM、GPT‑4o、GPT‑4o‑mini 等基线对比。整体准确率提升至94.0%，比 Gemini 基线高+10.9个百分点、GPT‑4o 高+10.8个百分点、GPT‑4o‑mini 高+14.2个百分点；长尾查询提升至90.7%，比基线提升8.5个百分点。

**⚠️ 局限性**

当前实现采用离线批处理与缓存，无法实时覆盖未见查询；模型规模大，推理成本高；个性化消歧仅基于历史赢率，尚未整合用户位置或订单历史等实时信号。

---

## 202. The Configurational Element Method for Nonconvex Granular Media

**arXiv ID:** 2603.00731 | [PDF](https://arxiv.org/pdf/2603.00731v1)

**作者:** Zhecheng Wang `[一作]` (University of Toronto), Eitan Grinspun `[通讯]` (University of Toronto)

**通讯引用:** 9351 | [OpenAlex ID](https://openalex.org/A5049319779)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出一种在配置空间中预计算并学习非凸颗粒之间接触与摩擦信息的方法，通过神经网络压缩接触映射，实现大规模二维颗粒系统的高效模拟。

**💡 创新点**

创新点在于：①将接触与摩擦全部归约为两个标量场（符号距离和法向投影力矩）；②利用配置空间梯度假设实现单一接触法向量；③通过神经场对这两个标量场进行学习，使每对粒子类型的接触信息仅占几十 KB；④采用SDF近似快速生成训练数据，专注于零层集附近。

**🔧 技术方法**

使用深度多层感知机（MLP）对符号距离和投影力矩进行建模，采用梯度下降最小二乘损失训练；实现基于配置空间的碰撞检测与摩擦求解的离散元方法；结合广相阶段的球包围盒过滤。

**📊 数据集**

无公开数据集，实验采用自行构造的二维非凸形状集合（O、U、N、#、八边形、井号、U、D、O、N等），在不同几何形状、孔径、形态多样性下进行列崩塌、筒仓排料、列堆积、旋转鼓、谷物倒入、链条缠绕等场景。

**📈 对比分析**

与传统基于凸几何或显式接触点的碰撞方法对比，本文方法在相同接触参数下保持物理合理性（如角度失稳、滑移、堆积角、堵塞现象），且内存占用仅数十 KB、训练时间不到数十秒。性能上能够在单机上模拟1~1.2万颗粒的系统，保持较低的计算开销；实验展示了不同形状下的宏观行为差异，验证了模型的形状敏感性。

**⚠️ 局限性**

局限性：仅在二维平面上验证，三维扩展尚未实现；假设所有接触点的法向投影力矩相同，无法覆盖极端非凸多点接触；训练数据采用SDF近似，可能在深度渗透时失准；对极大粒子数或高度非凸形状的效率与稳定性尚待进一步评估。

---

## 203. Monocular 3D Object Position Estimation with VLMs for Human-Robot Interaction

**arXiv ID:** 2603.01224 | [PDF](https://arxiv.org/pdf/2603.01224v1)

**作者:** Ari Wahl `[一作]` (Fraunhofer HHI), Sebastian Bosse `[通讯]` (Fraunhofer HHI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文利用预训练的Vision‑Language模型，在单目RGB图像、语言输入和机器人状态的条件下，实现了3D物体位置估计。

**💡 创新点**

创新点在于在保持模型通用视觉问答能力的同时，加入专门的3D坐标回归头，并通过低秩适配（QLoRA）和条件路由实现专用任务与通用任务的动态切换。

**🔧 技术方法**

技术主要包括LLaVA‑v1.5 7B VLM、QLoRA低秩适配、线性回归头、条件路由以及Huber损失和MAE评估。

**📊 数据集**

数据集为自采集的100k+图像，涵盖750种物体、不同光照、轨迹和单/多物体场景。

**📈 对比分析**

与基线（LLaVA单线性回归）相比，模型在测试集上中位数MAE为13 mm、欧氏误差27 mm，基线误差提升5倍，25%样本误差≤10 mm。

**⚠️ 局限性**

局限在于对单一机器人工作空间的偏倚、z轴误差较大、缺乏多物体与跨工作空间的泛化，且模型仅在收集的特定数据上训练。

---

## 204. The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents

**arXiv ID:** 2603.00801 | [PDF](https://arxiv.org/pdf/2603.00801v1)

**作者:** Shrey Shah `[一作]` (Microsoft), Levent Ozgur `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构建可编程的合成互联网环境，提出 Synthetic Web Benchmark，用于评估 web‑enabled 语言模型在面对对抗性搜索排名时的推理与事实性表现。

**💡 创新点**

创新点在于：①可控的 rank‑controlled 冒险信息注入；②过程级工具调用与搜索轨迹日志；③结合合成数据与真值标签，实现对模型因果响应的精确测量。

**🔧 技术方法**

采用 LlamaIndex 混合检索、零样本提示与工具调用、LLM‑as‑Judge 评估、统计显著性检验以及自定义的 confidence 机制。

**📊 数据集**

使用程序化生成的数千篇带可信度、偏见和事实标签的超链接文章，并配套的查询与答案；与 FEVER、TruthfulQA 等公开数据集相比，提供完整的 ground‑truth 与过程追踪。

**📈 对比分析**

在六款前沿模型（GPT‑5、GPT‑4o、o3、o1、o4‑mini、o1‑mini）上，在标准与对抗排名两种条件下进行 5,870 题的对比，单一假新闻占据搜索首位即可使准确率从 65% 降至 18% 甚至更低，同时模型置信度与准确率失衡。

**⚠️ 局限性**

局限性包括：仅评估零样本提示下的基础模型，未覆盖生产系统中的多阶段推理与外部验证模块；合成文本可能比真实网络更易识别；人类基准样本有限；缺乏对主题熟悉度与分布偏移的细粒度分析。

---

## 205. On the Practical Feasibility of Harvest-Now, Decrypt-Later Attacks

**arXiv ID:** 2603.01091 | [PDF](https://arxiv.org/pdf/2603.01091v1)

**作者:** Javier Blanco-Romero `[一作]` (Universidad Carlos III de Madrid), Daniel Díaz Sánchez `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 1161 | [OpenAlex ID](https://openalex.org/A5042030511)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

研究构建了开源的模拟测试平台，量化了在 TLS1.2、TLS1.3、QUIC 与 SSH 下的“先收集后解密”（HN‑DL）攻击的存储与量子解密成本。

**💡 创新点**

创新点包括：① 将 HN‑DL 转化为可量化的经济模型，推出协议层存储开销比例 α 与量子计算负载 E；② 通过对比不同协议的前向安全特性与可复用会话，识别了攻击范围与量子成本的两条独立维度；③ 提出基于 Encrypted Client Hello、禁用旧模式、重钥与后量子密钥交换的多重防御策略。

**🔧 技术方法**

使用技术主要有：Python 控制器、patched OpenSSL/OpenSSH、tshark 进行抓包、量子模拟模块（Shor 算法输出）、Monte Carlo 统计与存储成本模型；协议层分析基于 RFC 说明与符号/计算证明。

**📊 数据集**

数据集：在循环回环环境下收集 32 次会话捕获（8 payload 大小 × 4 协议），并利用公开的云存储价格（$12.16–$14.74/TB‑yr）以及 LTO‑9 磁带成本（$5.25/TB）进行成本估算；同时采样日志用于 Monte Carlo 模拟。

**📈 对比分析**

对比方法：把每个协议在相同 payload 大小下的 α、存储成本与量子实例数 E 进行直接对比；实验验证表明：TLS1.2 RSA 需要单一私钥即可解密所有会话，TLS1.3/QUIC 需每个会话恢复一次 ECDH，SSH 在重钥后每次重钥均需一次 Shor 计算。性能上，存储成本仅在 1% 全球采集时约 10 亿美元/年，而量子解密成本随会话长度与重钥频率线性增长。

**⚠️ 局限性**

局限性：① 仅在无网络丢包、无 MTU 片段的循环回环环境下测试，未模拟真实网络延迟与错误；② 模型假设攻击者已获得完整流量，未量化获取与横向渗透成本；③ TLS1.3/QUIC 现行协议缺乏在链路内重钥机制，导致 E=1 的量子成本不可调；④ 只关注服务器端配置，未考虑客户端行为与跨协议协作的复杂性。

---

## 206. LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards

**arXiv ID:** 2603.02146 | [PDF](https://arxiv.org/pdf/2603.02146v1)

**作者:** Guanzheng Chen `[一作]` (National University of Singapore), Lidong Bing `[通讯]` (MiroMind AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在 RLVR 框架中加入可验证的上下文奖励，显式训练 LLM 先进行信息检索再生成答案，解决了长上下文情境定位的梯度消失问题

**💡 创新点**

创新点在于将稀疏的答案奖励与密集的上下文奖励耦合，提供可验证的证据选择信号，并通过理论证明消除了 grounding 梯度消失

**🔧 技术方法**

使用基于策略梯度的 RL（PPO/GRPO）、可验证 F_β 上下文奖励、以及 LLM 的分段检索头实现多阶段生成

**📊 数据集**

使用自研 46K 条长文本 QA 数据集（来自书籍、arXiv、代码等），每条样本均标注关键信息块，并在 RULER、LongBench v2、LongReason 三大长上下文 benchmark 上评估

**📈 对比分析**

与 SFT、naïve RLVR 基线对比，LongRLVR 在所有模型（LLaMA‑3.1‑8B、Qwen2.5‑7B‑1M、Qwen2.5‑14B‑1M）以及三大 benchmark 上显著提升（如 Qwen2.5‑14B‑1M 在 RULER‑QA 由 86.3 提升至 95.4，LongBench v2 由 75.20 提升至 88.90，LongReason 由 73.42 提升至 78.42），甚至在参数量更小的模型上超过更大规模的基准模型

**⚠️ 局限性**

局限在于需要人工/自动生成的高质量标注数据和已知证据，训练成本高，对长上下文的分块策略和可验证奖励设计仍需进一步优化，且在极端大规模文本或无标签场景下表现尚未验证

---

## 207. NextAds: Towards Next-generation Personalized Video Advertising

**arXiv ID:** 2603.02137 | [PDF](https://arxiv.org/pdf/2603.02137v1)

**作者:** Yiyan Xu `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 NextAds 生成式个性化视频广告范式，将传统检索式广告转向实时生成与闭环优化；

**💡 创新点**

创新点在于：①将创意优化视为连续空间的生成任务；②模块化体系（Director、Producer、Verifier、Reflector）实现计划、生成、验证与反馈循环；③给出两项任务（PCG、PCI）和轻量级基准；

**🔧 技术方法**

使用生成式 AI（大语言模型和多模态模型）如 Qwen3‑VL、GPT‑4o；视频生成模型如 Wan2.2、Sora2；以及 VLM Gemini 2.5‑Flash 进行自动评判；

**📊 数据集**

基准数据集包括 Qilin 推荐数据集（用户历史交互）与 MicroLens 微视频数据集，结合自建 18 件产品的创意资产库；

**📈 对比分析**

与非个性化基线 GenericAds 对比，NextAds 在个性化得分上明显提升，视频质量和产品一致性保持相当；然而多样性略有下降；实验表明闭源模型性能更优，开源模型多样性更高但可控性不足；

**⚠️ 局限性**

局限性包括：① 生成模型在高层语义偏好下易出现“伪条件”，导致视觉噪声或品牌混淆；② 当用户与产品兼容度低时强行个性化会产生不连贯创意；③ 需要更精细的用户建模与反馈归因；④ 生成过程计算成本和延迟仍是实际落地的瓶颈。

---

## 208. AG-VAS: Anchor-Guided Zero-Shot Visual Anomaly Segmentation with Large Multimodal Models

**arXiv ID:** 2603.01305 | [PDF](https://arxiv.org/pdf/2603.01305v1)

**作者:** Zhen Qu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Xingang Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于anchor的零样本视觉异常分割框架AG‑VAS，直接生成二值异常掩码

**💡 创新点**

创新点在于引入可学习的绝对锚词[SEG]与相对锚词[NOR]/[ANO]作为语义桥梁，并设计Semantic‑Pixel Alignment Module与Anchor‑Guided Mask Decoder实现高精度像素级对齐

**🔧 技术方法**

使用LMM（如LLaVA-OneVision）、SAM‑ViT‑H作为像素编码器，结合SPAM、AGMD以及LoRA微调的LLM进行指令学习

**📊 数据集**

训练数据包括新构建的Anomaly‑Instruct20K、20k工业异常图像集Anomaly‑Seg20K、通用分割数据ADE20K以及VQA数据集；评测在六个工业/医学基准（MVTec‑AD、KSDD2、RSDD、ISIC、ColonDB、ClinicDB）上

**📈 对比分析**

与CLIP‑基准（WinCLIP、APRIL‑GAN、AnomalyCLIP、Bayes‑PFL）和LMM‑基准（PixelLM、PaDT、LISA）对比，AG‑VAS在AP、F1‑Max、IoU_ano、IoU_nor等指标上均显著领先，尤其在正常样本的拒绝率上高达87.7%

**⚠️ 局限性**

局限在于对大型LMM后端依赖显著、推理速度受限、缺乏对更稀有或复杂异常类型的系统性验证

---

## 209. Bentō: Optimizing Persistent Memory Programs

**arXiv ID:** 2603.01889 | [PDF](https://arxiv.org/pdf/2603.01889v1)

**作者:** Sebastião Amaro `[一作]` (Instituto Superior Técnico), Miguel Matos `[通讯]` (Instituto Superior Técnico)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统评估持久内存程序中不同 flush 与 fence 指令组合的性能与一致性影响，提出并实现了一个黑盒二进制重写工具，能够在不改变 crash‑consistency 的前提下将指令替换为更高效的组合，实验表明可提升最高15%性能。

**💡 创新点**

创新点在于首次在真实应用与工作负载上量化各种持久性指令组合的效益，并提供一种自动化、无源代码依赖的二进制重写方案；此外提出 Minimal Crash‑Consistency State（MCCS）概念，为进一步的精细化性能优化奠定理论基础。

**🔧 技术方法**

研究使用了持久性指令分析、性能基准（Level Hashing、FAST&FAIR、WOART）在 YCSB 工作负载下的实验、以及 e9patch 工具实现的二进制重写；通过手工修改与自动重写的对比验证方法有效性。

**📊 数据集**

实验数据主要来自 YCSB benchmark 的 A、B、D 等工作负载（每个 2500 万操作）、Level Hashing 1 亿插入、FAST&FAIR 与 WOART 的相同负载；这些应用覆盖了插入密集、读写混合及读密集等多种情形。

**📈 对比分析**

通过在原始（vanilla）、手工修改、自动重写三种版本中测量平均运行时间和标准差，对比各工作负载的性能；在插入密集的 Level Hashing 上实现最高 15% 加速，其他工作负载提升 5–8%，在负载不利时无性能退化。

**⚠️ 局限性**

限制主要包括：二进制重写仅能匹配 flush 后紧跟的 fence，无法处理间隔较大的情况；假设原程序已满足 crash‑consistency，未能自动发现缺失的持久性指令；MCCS 的完整实现仍需更复杂的静态/动态分析。

---

## 210. Token-level Data Selection for Safe LLM Fine-tuning

**arXiv ID:** 2603.01185 | [PDF](https://arxiv.org/pdf/2603.01185v1)

**作者:** Yanping Li `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 85204 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型的微调过程，本文提出了 TOSS 框架，通过对每个 token 的安全风险进行评估并筛选，从而在保证安全性的同时保持下游任务性能。

**💡 创新点**

创新点在于：① 通过构建安全退化模型和实用性导向模型，提出基于两者损失差的 token 级安全风险度量；② 引入逐步精细化的 TOSS‑Pro 机制，在多轮迭代中不断改进安全退化模型，提升 token 级风险识别精度；③ 采用全局排名而非局部或样本级筛选，实现更细粒度、更有效的安全与效能平衡。

**🔧 技术方法**

使用的技术包括：token‑级损失差度量、全局 token 排名与二值掩码、LoRA 微调、进化式安全退化模型更新、以及 GPT‑4o 评估器进行安全与效能评测。

**📊 数据集**

数据集方面，使用 REDORCA（90k 任务对 + 22k 红队对）作为自定义数据，OpenOrca 及 Anthropic Red‑Team 数据用于构建实用与安全退化模型；在下游任务上评估 SLIMORCA、在安全评估上使用 HEx‑PHI 与 Anthropic HH 子集。

**📈 对比分析**

与标准 SFT、随机 token 选取、SafeInstr、DSIR、样本级 SEAL 等基线对比，TOSS 在安全评测（例如 HEx‑PHI、Anthropic HH）上平均提升约 20% win‑rate，且在下游任务上保持或提升效能；TOSS‑Pro 在安全方面进一步提升 4–6%，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：① 需要额外的安全退化与实用数据来训练参考模型，增加数据准备成本；② 目前仅验证在共享 tokenizer 的模型间迁移，跨 tokenizer 迁移效果未知；③ token‑级筛选与评估的计算开销较大，可能影响大规模部署；④ 仍需在更大模型与多语言场景下进一步验证。

---

## 211. Attention Smoothing Is All You Need For Unlearning

**arXiv ID:** 2603.01285 | [PDF](https://arxiv.org/pdf/2603.01285v1)

**作者:** Saleh Zare Zade `[一作]` (Wayne State University), Dongxiao Zhu `[通讯]` (Wayne State University)

**通讯引用:** 3155 | [OpenAlex ID](https://openalex.org/A5009256505)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于注意力平滑的自蒸馏方法ASU，用以在不重新训练的前提下让大型语言模型忘记指定敏感或版权信息，且保持输出连贯性。

**💡 创新点**

创新点在于：①将忘记过程转化为自蒸馏，构造“忘记教师”仅通过提高自注意力的softmax温度来抖动注意力分布；②通过抑制词级与语义级关联，实现对事实记忆的精准消除；③仅使用单一温度超参数，避免额外模型或参数，保持方法简洁。

**🔧 技术方法**

技术包括：Transformer自注意力温度缩放、KL散度自蒸馏、对保留集使用梯度下降或KL正则化、实验使用不同层级注意力平滑、以及对不同任务做多轮持续性学习评估。

**📊 数据集**

数据集涵盖：TOFU（模拟的作者问答集合），MUSE（新闻与书籍版权文本），WMDP（危险知识），以及真实世界的“忘记”个体集合和下游基准（MMLU、ARC‑c、GSM8K、TruthfulQA）。

**📈 对比分析**

与多种基线（GA、NPO、DPO、IDK、ME 等）比较，ASU 在 TOFU、MUSE、WMDP 以及真实世界场景中均实现最高或相近的忘记效能（FE）与模型效用（MU）折中；尤其在持续忘记任务中表现出最小性能衰退。

**⚠️ 局限性**

局限性包括：①对温度超参数的依赖，虽然在一定范围内稳定但需手动调节；②主要针对文本生成与问答任务，未充分验证在更复杂推理或多模态任务中的适用性；③在极高温度或过度平滑时可能导致输出不连贯，需平衡。

---

## 212. TC-SSA: Token Compression via Semantic Slot Aggregation for Gigapixel Pathology Reasoning

**arXiv ID:** 2603.01143 | [PDF](https://arxiv.org/pdf/2603.01143v1)

**作者:** Zhuo Chen `[一作]` (Shenzhen University of Advanced Technology), Lijian Xu `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出TC-SSA Token Compression via Semantic Slot Aggregation，将全尺寸WSI的数十万patch压缩到仅32个语义槽，以支持视觉语言模型进行诊断推理。

**💡 创新点**

创新点在于：①可学习的门控Top‑2稀疏路由将patch映射到固定语义槽；②加权聚合与语义亲和聚类正则共同避免槽坍塌；③实现高压缩比的同时保持全局诊断信息，避免传统采样导致的关键区域丢失。

**🔧 技术方法**

使用的技术包括轻量级gate网络、Top‑2稀疏路由、加权池化聚合、负载均衡+熵+z‑loss正则、CONCH视觉编码器以及下游视觉语言模型。

**📊 数据集**

采用的数据集有：SlideBench (TCGA) 诊断与显微任务、SlideBench (BCNB) 零样本、WSI‑VQA*、TCGA‑BRCA、TCGA‑NSCLC、PANDA。

**📈 对比分析**

与LLaVA‑Med、Quilt‑LLaVA、SlideChat、GPT‑4o等基线比较，TC‑SSA在32个token预算下在SlideBench TCGA上实现78.34% overall accuracy、77.14%诊断子集准确率；在MIL任务中分别达到95.83%（BRCA）、98.27%（NSCLC）和79.80%（PANDA），压缩率约58×，显著优于采样与稀疏注意力方法。

**⚠️ 局限性**

局限性在于固定槽数K对压缩质量高度依赖于patch编码器；将空间几何信息转换为语义结构可能影响定位密集任务；并且需要额外的正则约束来防止槽坍塌。

---

## 213. HAVEN: High-Bandwidth Flash Augmented Vector Engine for Large-Scale Approximate Nearest-Neighbor Search Acceleration

**arXiv ID:** 2603.01175 | [PDF](https://arxiv.org/pdf/2603.01175v1)

**作者:** Po-Kai Hsu `[一作]` (Georgia Institute of Technology), Shimeng Yu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 30644 | [OpenAlex ID](https://openalex.org/A5054894631)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了HAVEN架构，即在GPU内部集成High‑Bandwidth Flash（HBF）与近存储搜索单元，支持全精度向量数据库在包内存储并完成重排，从而显著提升大规模向量检索的吞吐量与低延迟。

**💡 创新点**

创新点：①将3D NAND重构为分布式子阵列以获得高并行度与读带宽；②在GPU包内用HBF取代一部分HBM，形成双层内存体系；③在HBF下方实现近存储搜索单元，直接在Flash内部完成重排，避免PCIe/DDR搬运。

**🔧 技术方法**

使用技术：IVF‑PQ索引与重排、High‑Bandwidth Flash die‑stacked NAND、近存储搜索单元（32个重排队列、32×MAC、Bitonic排序器）、分布式子阵列设计、HBM2E、CUDA/Faiss集成、NeuroSim/3D‑FPIM仿真、FPGA/ASIC验证。

**📊 数据集**

实验数据集：BIGANN‑1B、SPACEV‑1B（均1B向量）和Wiki‑88M（88M向量）。

**📈 对比分析**

对比方法：在相同Recall@k（BIGANN/SPACEV为0.95，Wiki‑88M为0.9）下，比较GPU‑DRAM、GPU‑SSD与GPU‑HBF三种方案。HBF在吞吐量上提升3–8×，在Wiki‑88M上超过20×；延迟下降3–6×，在Wiki‑88M上超过40×；在同一Recall水平下，HBF实现了Pareto优势，优于ANNA ASIC和SmartANNS SSD加速器，最高QPS达到8.1k。

**⚠️ 局限性**

局限性：①HBF技术仍处于研发阶段，实际芯片实现与热功率管理尚需验证；②双层内存架构对GPU工艺的适配和封装尺寸有一定限制；③分布式子阵列设计在极大容量下仍受功耗包络与带宽限制；④实验在单机A100 + A100‑HBF原型上完成，跨平台迁移与大规模集群验证仍需进一步研究。

---

## 214. Learning Nested Named Entity Recognition from Flat Annotations

**arXiv ID:** 2603.00840 | [PDF](https://arxiv.org/pdf/2603.00840v1)

**作者:** Igor Rozhkov `[一作]` (Lomonosov Moscow State University), Natalia Loukachevitch `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 2513 | [OpenAlex ID](https://openalex.org/A5003446912)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何仅利用平面NER注释学习嵌套命名实体识别。

**💡 创新点**

提出了基于字符串包含、实体腐败、平面中立化以及混合细调+LLM的多种弱监督策略，并系统比较其对嵌套NER的效果。

**🔧 技术方法**

采用Binder span‑based 模型进行细调，利用子串匹配、伪嵌套生成、内容感知中立化以及少量提示的DeepSeek‑R1/RuAdapt‑Qwen2.5等LLM。

**📊 数据集**

在俄语嵌套NER基准NEREL（29个实体类型，21%实体嵌套）上进行实验。

**📈 对比分析**

通过将平面训练、加入包含/腐败/中立化的弱监督、全监督基准以及纯LLM和混合方案进行对比，最佳弱监督方法实现内层F1 26.37%，相当于覆盖全监督缺口的40%，混合方案总体F1 70.16%但内层仍低于最强弱监督。

**⚠️ 局限性**

仅在俄语新闻文本上验证，未跨语言或跨领域测试，且LLM在细粒度嵌套识别上仍表现欠佳，方法对极少数类的处理不足。

---

## 215. FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding

**arXiv ID:** 2603.02096 | [PDF](https://arxiv.org/pdf/2603.02096v1)

**作者:** Yiweng Xie `[一作]` (Fudan University), Zuxuan Wu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FluxMem，一个训练无关、分层的视觉记忆框架，利用 Temporal Adjacency Selection 与 Spatial Domain Consolidation 对实时视频流中的视觉令牌进行自适应压缩，显著降低计算与内存负担；

**💡 创新点**

创新点在于：①将视觉上下文分为短期、中期、长期三层记忆，分别针对时空冗余进行分阶段压缩；②采用 Otsu 直方图阈值自适应决定压缩率，无需人工调参；③实现训练‑free、可即插即用，保持严格的实时因果性；④通过两阶段轻量模块精细化保留关键信息；

**🔧 技术方法**

主要技术包括：训练‑free 的令牌压缩、Cosine 距离相似度、Otsu 自适应阈值、Union‑Find 聚类、分层记忆缓冲、LLM 接口（如 Qwen2.5‑VL‑7B）、实时触发响应机制；

**📊 数据集**

使用的公开数据集：在线评测 OVO‑Bench 与 StreamingBench；离线评测 VideoMME、MLVU、LongVideoBench；

**📈 对比分析**

与现有在线/离线基准方法（LiveVLM、TimeChat‑Online 等）对比，FluxMem 在 StreamingBench 达到 76.4（+2.5）分、OVO‑Bench 67.2（+3.5）分，token 压缩率约 70%，延迟降低 69.9%，GPU 内存降低 34.5%；在离线任务中 VideoMME 65.3、MLVU 73.1、LongVideoBench 61.1，均超过训练‑free 方案并逼近训练‑based 结果；

**⚠️ 局限性**

局限性包括：依赖预训练视觉编码器，对极高动态或极大视频长度时自适应阈值可能失效；缺乏专门的多模态对齐优化；对视觉令牌分辨率和编码精度敏感；需要手动设定短/中/长期记忆容量。

---

## 216. SEAR: Sample Efficient Action Chunking Reinforcement Learning

**arXiv ID:** 2603.01891 | [PDF](https://arxiv.org/pdf/2603.01891v1)

**作者:** C. F. Maximilian Nagy `[一作]`, Gerhard Neumann `[通讯]` (FZI Forschungszentrum Informatik)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 SEAR，一种在在线强化学习环境下使用动作分块的离线策略算法。

**💡 创新点**

创新点在于：① 引入多时域目标的因果 Transformer 评估器，充分利用分块动作的时间结构；② 采用随机重规划（receding horizon）提升状态覆盖和收敛稳定性；③ 结合最大熵框架实现对长序列的高效样本利用。

**🔧 技术方法**

使用最大熵强化学习、因果 Transformer 评估器、多时域 Q 目标、随机前缀重规划与 Transformer Actor 网络。

**📊 数据集**

在 Metaworld ML1 基准（20 个最难任务）上进行实验，采用 1M 环境交互进行评估。

**📈 对比分析**

与 SimbaV2（单步）和 CQN-AS（分块）对比，SEAR 在 20 个任务上 IQM 成功率提升 20% 以上，单步成功率约 60%，SEAR-10 约 90%；样本效率更高，收敛速度更快。

**⚠️ 局限性**

局限性：① 需要手动设置分块长度 N，且不同任务对 N 的敏感度未知；② 对 Transformer 计算成本和内存依赖较高；③ 随机重规划在某些环境中可能导致决策频率下降；④ 目前仅在物理操控任务验证，尚未在其他域如步态或导航中测试。

---

## 217. Solving Inverse PDE Problems using Minimization Methods and AI

**arXiv ID:** 2603.01731 | [PDF](https://arxiv.org/pdf/2603.01731v1)

**作者:** Noura Helwani `[一作]` (American University of Beirut), Georges Sakr `[通讯]` (American University of Beirut)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对比了传统数值方法（如 RK4、ODE45、有限差分、Newton 等）与物理信息神经网络（PINN）在逻辑斯蒂方程和渗流方程（PME）前向和逆向问题中的求解性能。

**💡 创新点**

创新点在于：① 将 PINN 应用于具有无封闭解析解的非线性 PDE（PME）的逆向参数辨识；② 对 PINN 与经典数值方案在相同问题上的误差、收敛速度和计算成本进行系统评估；③ 通过归一化与两阶段优化（Adam + L-BFGS）提升 PINN 在参数恢复中的鲁棒性。

**🔧 技术方法**

技术手段包括：传统 ODE/PDE 求解器（Runge–Kutta、有限差分、Newton 迭代、fmincon），以及基于自动微分的 PINN（全连接网络、tanh 激活、Adam 与 L‑BFGS 优化器、Sobol 采样点）。

**📊 数据集**

数据集主要为合成数据：逻辑斯蒂方程使用已知解析解生成训练/验证样本，PME 使用 Barenblatt 解析解产生时间空间数据；亦尝试了少量真实人口观测数据用于逆向学习。

**📈 对比分析**

比较方法采用相对 L2 误差、训练损失曲线和计算时间。结果显示：PINN 在多数前向问题上可达到与 RK4 相当或略优的误差，远低于手工求解的数值误差；但在逆向问题中，PINN 能以极低误差恢复参数（如 r），但对 PME 的 β 参数敏感度高，误差仍高于经典 Newton+fmincon 方法。

**⚠️ 局限性**

局限性包括：① PINN 对初值和归一化敏感，若不合适会出现学习失败或数值不稳定；② 逆向参数识别高度依赖初始猜测，误差随猜测远近显著变化；③ 训练时间和 GPU 资源消耗大，且对高维或更复杂 PDE 仍需进一步优化。

---

## 218. HierKick: Hierarchical Reinforcement Learning for Vision-Guided Soccer Robot Control

**arXiv ID:** 2603.00948 | [PDF](https://arxiv.org/pdf/2603.00948v1)

**作者:** Yizhi Chen `[一作]` (Shanghai Innovation Institute), Yue Gao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 156209 | [OpenAlex ID](https://openalex.org/A5100399276)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HierKick框架，利用双频层级强化学习实现视听驱动的人形机器人足球赛动作，包含接近、对齐、盘球和射门四个阶段

**💡 创新点**

创新点在于：①双频层级控制（5 Hz高层决策+50 Hz低层执行）实现多时间尺度任务；②高层Coach模型通过YOLOv8感知并输出加速度指令；③预训练低层运动控制器与多阶段奖励机制相结合；④利用域随机化与命令历史提升仿真到真实转移

**🔧 技术方法**

使用技术包括：YOLOv8视觉检测、Proximal Policy Optimization（PPO）训练、预训练的踢球低层控制器、正则化多阶段奖励、域随机化、双频控制架构、PD控制器实现关节位置目标

**📊 数据集**

主要数据集为仿真环境（IsaacGym、MuJoCo）与真实场景的Booster T1机器人，球场配置与球的随机化数据，未使用公开公开数据集

**📈 对比分析**

与端到端RL方法对比：HierKick在IsaacGym、MuJoCo与真实机器人上分别取得95.2%、89.8%和80%成功率，显著优于端到端（25.6%）与删减信息/修改命令方式的版本，奖励曲线与踢球距离分布均更稳定、均值更低、方差更小

**⚠️ 局限性**

局限性包括：遮挡、照明变化导致视觉误检；草坪摩擦不均导致盘球不稳；固定阶段阈值与决策频率可能不适应所有场景；感知精度有限导致位置误差；对极端草坪或光照变化的鲁棒性仍有限

---

## 219. DiffusionXRay: A Diffusion and GAN-Based Approach for Enhancing Digitally Reconstructed Chest Radiographs

**arXiv ID:** 2603.01686 | [PDF](https://arxiv.org/pdf/2603.01686v1)

**作者:** Aryan Goyal `[一作]` (Indian Institute of Technology Bombay), Preetham Putha `[通讯]` (Qure.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种双阶段的DiffusionXRay图像增强管线，先利用MUNIT-LQ或DDPM-LQ生成逼真的低质量胸片，再用DDPM-HQ将其恢复为高质量图像。

**💡 创新点**

创新点在于把低质量生成视为无监督域迁移/风格迁移任务，利用两种独立方法（GAN+MUNIT与DDPM）合成配对训练数据，并在此基础上训练DDPM进行恢复，显著提升细节保留与对比度。

**🔧 技术方法**

采用MUNIT（多模态无监督图像到图像翻译）生成低质量图像，使用DDPM（去噪扩散概率模型）进行低质量生成和高质量恢复；训练时加入VGG感知损失、循环一致性以及条件DDPM。

**📊 数据集**

使用了LQ‑CXR12K（12,580张CT投影得到的低质量胸片）、HQ‑CXR300K（30万张高质量胸片）以及公开ChestX‑ray8测试集（25,596张），并公开了合成的低质量版本。

**📈 对比分析**

与基准DDPM（在双三角下采样生成的低质量图像上训练）相比，DiffusionXRay在PSNR上从20.08提升至27.50、SSIM从0.83提升至0.92；在DDPM‑LQ生成的数据上PSNR从19.85提升至22.21、SSIM保持在0.78。人工评估中，病灶可见率100%对比基准6.6%，整体质量提升到72.9%。

**⚠️ 局限性**

主要限制是计算成本高，扩散模型推理耗时长，未来可通过效率优化和自适应条件化来降低开销。

---

## 220. Unifying Language-Action Understanding and Generation for Autonomous Driving

**arXiv ID:** 2603.01441 | [PDF](https://arxiv.org/pdf/2603.01441v1)

**作者:** Xinyang Wang `[一作]` (State Key Lab of CAD and CG, Zhejiang University), Wei Chen `[通讯]` (State Key Lab of CAD and CG, Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个统一的 Vision‑Language‑Action（VLA）模型 LinkVLA，解决了语言与动作之间的对齐不足以及自回归动作生成的低效问题。

**💡 创新点**

创新点包括：① 将语言指令与轨迹动作映射到同一离散代码书（token），从架构层面消除模态鸿沟；② 引入动作理解（逆向生成）目标，构建双向语言–动作一致性；③ 采用两步粗细生成（Coarse‑to‑Fine）机制，显著降低推理延迟。

**🔧 技术方法**

技术细节包括：InternVL2‑1B 视觉‑语言基底（InternViT + Qwen2‑0.5B LLM）；共享代码书、基于 log 坐标的动作 tokenization 与空间软标签；双向训练目标；Coarse‑to‑Fine 两阶段生成；Chain‑of‑Thought 推理；LoRA 微调。

**📊 数据集**

使用的主要数据集有：Bench2Drive（CARLA 交互式场景）、SimLingo Action Dreaming（指令跟随评估）、DriveLM‑hard（VQA 与 commentary 评测）以及 CARLA Town 13 真实路况数据。

**📈 对比分析**

在 Bench2Drive 评测中，LinkVLA 获得最高 Driving Score 91.01、Success Rate 74.55，较 SimLingo 提升 5.94 / 7.28 分；推理延迟从 361 ms 降至 48 ms（+86% 节省）。在指令跟随任务中，平均成功率提升至 87.16%；在语言理解任务（VQA、commentary）中 SPICE/BLEU/ROUGE 指标均超过现有基线。

**⚠️ 局限性**

局限性：① 仍可能出现指令-动作不一致的细粒度错误；② 在极端天气或未见场景下的泛化与鲁棒性待进一步验证；③ 依赖大型预训练模型，部署成本和算力需求仍高；④ 评估主要基于仿真环境，缺乏真实道路部署的安全性验证。

---

## 221. A Unified Approach to Memory-Sample Tradeoffs for Detecting Planted Structures

**arXiv ID:** 2603.00770 | [PDF](https://arxiv.org/pdf/2603.00770v1)

**作者:** Sumegha Garg `[一作]` (Rutgers University), Vatsal Sharan `[通讯]` (University of Southern California)

**通讯引用:** 311 | [OpenAlex ID](https://openalex.org/A5009088661)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个统一的框架，用于证明在多通道流式算法中检测植入结构（如二分图中的大团、稀疏高斯信号、稀疏主成分等）所需的最小内存，并给出了多通道内存-样本权衡；

**💡 创新点**

创新点包括：①推导出一个新的分布式数据处理不等式，给出检测植入结构的充分条件；②利用该框架得到许多经典问题（植入k‑二分团、稀疏高斯均值、稀疏PCA、最大二分团、有限尺寸稠密子图等）的几乎最优内存下界；③首次给出稀疏PCA在脉冲协方差模型中的内存-样本权衡；④在顶点到达模型中得到更强的多通道内存下界。

**🔧 技术方法**

核心技术包括信息复杂度与信息成本分析、直接求和（direct‑sum）论证、Hellinger距离与KL散度的处理、剪切粘贴（cut‑paste）性质、截断（truncation）技巧以及对多通道信息成本（multi‑pass information cost）的利用。

**📊 数据集**

本文为理论研究，不使用具体数据集，而是通过随机图、随机高斯分布等模型进行分析。

**📈 对比分析**

与已知的上界算法相比，本文得到的内存下界在低内存（O(log n)） regime 下几乎与上界匹配；例如，针对植入k‑二分团，在 O(log n) 内存下，只有当 k=Ω(√n) 时才有多通道算法可行；对于最大二分团、稠密子图等问题，提供了比以往更严格的多通道内存下界。

**⚠️ 局限性**

局限性：仅针对多通道流式模型（固定通道数）；对稀疏PCA仅在信号强度为常数且特定稀疏结构时适用；框架要求平均分布与基准分布之间的似然比有点-wise 上界，可能不适用于所有植入结构；在极端稀疏或极大 k 的情况下，下界可能不再紧。

---

## 222. TraceSIR: A Multi-Agent Framework for Structured Analysis and Reporting of Agentic Execution Traces

**arXiv ID:** 2603.00623 | [PDF](https://arxiv.org/pdf/2603.00623v1)

**作者:** Shu-Xun Yang `[一作]` (Beijing Institute of Technology), Jie Tang `[通讯]` (Tsinghua University)

**通讯引用:** 28896 | [OpenAlex ID](https://openalex.org/A5044791875)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TraceSIR框架，对大型语言模型驱动的代理系统执行轨迹进行结构化分析与报告生成；

**💡 创新点**

创新点在于设计三位专门化代理（StructureAgent、InsightAgent、ReportAgent）以及一种名为“TraceFormat”的结构化抽象表示，能压缩长轨迹并保留关键行为信息；

**🔧 技术方法**

采用多代理协同、长度感知抽象、结构化诊断、统计聚合以及LLM工具链等技术实现分析与报告；

**📊 数据集**

使用了三大真实代理基准的失败案例——BrowseComp、Tau2Bench、SWE-bench，构建统一评测集TraceBench；

**📈 对比分析**

与ClaudeCode基线对比，TraceSIR在人类评测与LLM判定下各项指标均显著提升，尤其在错误分析、根因分析和整体影响上取得约9.7%–26.0%的相对提升；

**⚠️ 局限性**

局限性包括对LLM能力的依赖、报告生成对案例数量的规模限制、较高的延迟与成本以及输出的随机性和可复现性问题。

---

## 223. SimRecon: SimReady Compositional Scene Reconstruction from Real Videos

**arXiv ID:** 2603.02133 | [PDF](https://arxiv.org/pdf/2603.02133v1)

**作者:** Chong Xia `[一作]` (Tsinghua University), Yueqi Duan `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用感知-生成-仿真三阶段管线，将复杂的现实视频重建为可直接用于物理模拟的对象中心3D场景

**💡 创新点**

设计了两大桥接模块：主动视角优化（Active Viewpoint Optimization）以获得最具信息量的视图作为生成条件，场景图合成器（Scene Graph Synthesizer）以结构化的物理支撑关系引导场景在仿真器中的自适应堆叠

**🔧 技术方法**

采用3D Gaussian Splatting做底层几何重建、语义分割、Rodin生成单体对象、VLM（如Qwen2.5‑VL）推断属性和场景图、基于物理引擎的层次化组装流程

**📊 数据集**

仅使用ScanNet 20个室内场景的原始RGB视频作为训练与测试数据

**📈 对比分析**

与DPRecon、InstaScene、Gen3DSR、SceneGen以及MetaScenes等基线进行对比，实验显示在几何精度（CD、F‑Score、NC）、渲染质量（PSNR、SSIM、LPIPS、MUSIQ）及整体处理时长上均优于同类方法，最终生成的仿真场景在物理可行性方面亦表现更佳

**⚠️ 局限性**

局限性包括对大规模、极度混乱场景的鲁棒性仍待提升，桥接模块需要显著的GPU计算资源，且对现有预训练模型的依赖可能限制在新领域的迁移

---

## 224. nchellwig at SemEval-2026 Task 3: Self-Consistent Structured Generation (SCSG) for Dimensional Aspect-Based Sentiment Analysis using Large Language Models

**arXiv ID:** 2603.01788 | [PDF](https://arxiv.org/pdf/2603.01788v1)

**作者:** Nils Constantin Hellwig `[一作]` (University of Regensburg), Christian Wolff `[通讯]` (University of Regensburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在SemEval‑2026 Task 3 DimABSA 任务中，提出并实现了 Self‑Consistent Structured Generation (SCSG) 方法，用于维度化情感分析。

**💡 创新点**

创新点：①首次将自一致性 (self‑consistency) 推理与参数高效 LoRA 微调结合，提升预测可靠性；②利用 vLLM 的 PagedAttention 和批量推理高效复用 KV 缓存，解决多次推理的算力瓶颈；③通过多数投票与平均 valence‑arousal 值，进一步优化结果质量。

**🔧 技术方法**

技术：LoRA 微调、vLLM + PagedAttention、批量推理、多视图（k‑fold）自一致性验证、投票+平均机制。

**📊 数据集**

数据集：SemEval‑2026 Task 3 DimABSA 多语言多领域数据集（6 种语言、3 个领域，包含 DimASTE 与 DimASQP 两个子任务）。

**📈 对比分析**

比较方法：与单次提示基线以及多视图提示(MvP) 进行对照，使用连续 F1（cF1）评估。结果显示，平均 cF1 在 DimASTE 上从 55.52 提升至 56.50，在 DimASQP 上从 46.10 提升至 47.37，且多处显著提升（p < 0.001）。

**⚠️ 局限性**

局限：①对低资源语言/领域仍有显著差距；②自一致性仍需多次推理，算力和时间成本高；③模型选择对性能影响显著，未统一最佳模型；④未探索少样本或多语言联合训练的潜力。

---

## 225. Identifying and Characterising Response in Clinical Trials: Development and Validation of a Machine Learning Approach in Colorectal Cancer

**arXiv ID:** 2603.00757 | [PDF](https://arxiv.org/pdf/2603.00757v1)

**作者:** Adam Marcus `[一作]`, Paul Agapow `[通讯]` (AstraZeneca)

**通讯引用:** 6514 | [OpenAlex ID](https://openalex.org/A5021544795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证了一种结合部分条件建模与 Virtual Twins 方法的动态响应子群识别与表征流程，并在合成数据与结直肠癌临床试验中进行了应用。

**💡 创新点**

创新点在于将部分条件建模引入时间变 covariate 的重复测量，允许治疗响应随时间动态变化，并使用 survLIME 对时间特定响应进行可解释性分析。

**🔧 技术方法**

技术手段包括 Virtual Twins、部分条件建模、DeepSurv、随机生存森林、WTTE-RNN、survLIME、嵌套交叉验证、ARMA 仿真等。

**📊 数据集**

使用的数据集为：仿真生成的 1000、300、2000 人的时间序列数据，以及来自 Project Data Sphere 的四项结直肠癌临床试验数据（panitumumab 组合治疗）。

**📈 对比分析**

通过与不使用 PCM 的基线模型比较，利用 AUC、灵敏度、特异性等指标评估性能；固定响应基准下 AUC 0.773，动态响应 AUC 0.685；PCM 在大样本条件下提升识别与表征效果，并在临床试验中得出与文献一致的基因突变、转移部位与种族重要性。

**⚠️ 局限性**

局限性包括：计算量大、对大样本依赖强、易过拟合并误识别混杂因素、缺乏非线性解释能力、对缺失值处理与测量时间独立性的假设敏感，需要进一步验证与临床实验。

---

## 226. Contract-based Agentic Intent Framework for Network Slicing in O-RAN

**arXiv ID:** 2603.01663 | [PDF](https://arxiv.org/pdf/2603.01663v1)

**作者:** Fransiscus Asisi Bimo `[一作]` (National Taiwan University of Science and Technology), Ray-Guang Cheng `[通讯]` (National Taiwan University of Science and Technology)

**通讯引用:** 1895 | [OpenAlex ID](https://openalex.org/A5024212834)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于契约的代理式意图框架（CAIF），用于O‑RAN网络切片的意图到策略转换，并在实验平台上验证其闭环控制效果。

**💡 创新点**

创新点在于将LLM意图提取与正式意图契约分离，使用形式化契约（TMF‑921 API）对生成的策略进行确定性验证，构建双代理（Profile + Evaluator）闭环校验流程，确保安全可靠的策略执行。

**🔧 技术方法**

使用技术包括：大型语言模型（Qwen3‑4B‑Instruct‑2507 与 Llama‑3.3‑Nemotron），TMF‑921 意图管理 API，Open RAN 接口（A1、E2、O1），rApp/xApp 控制循环，以及JSON‑LD、schema‑driven校验。

**📊 数据集**

实验使用了自制的 500 条自然语言意图数据集（包含 1‑5 次 shot 变体），公开可获取，用于评估意图转策略的语义完整性。

**📈 对比分析**

与直接执行的 LLM 基线相比，CAIF 在准确率上从 96.8% 提升到 99.8%，置信区间更窄（98.9%‑100%）。字段级准确率达到 100%，延迟仅在 1‑shot 场景下略高（11.8s vs 8.5s），但在 2‑shot 以上场景下已实现性能弥补，并在多意图动态场景中保持 SLA 约束。

**⚠️ 局限性**

局限性包括：契约设计仍需人工定义，扩展至其他 RAN 管理场景（如能耗、流量导引）需进一步研究；对大型 LLM 的可靠性依赖仍存在风险；实验数据集规模有限，未覆盖更复杂的业务需求；系统在高并发多意图情况下的可扩展性与实时性需进一步验证。

---

## 227. Teacher-Guided Causal Interventions for Image Denoising: Orthogonal Content-Noise Disentanglement in Vision Transformers

**arXiv ID:** 2603.01140 | [PDF](https://arxiv.org/pdf/2603.01140v1)

**作者:** Kuai Jiang `[一作]` (China University of Mining and Technology), Zhuoran Zheng `[通讯]` (Qilu University of Technology)

**通讯引用:** 609 | [OpenAlex ID](https://openalex.org/A5077971554)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Teacher‑Guided Causal Disentanglement Network（TCD‑Net），实现环境偏置去除、内容‑噪声分离与教师引导的因果干预图像去噪。

**💡 创新点**

创新点是将因果干预框架与 Vision Transformer 结合，构建环境偏置调整（EBA）、双分支正交约束解耦以及 Google Nano Banana Pro 引导的因果先验，从而显著提升去噪鲁棒性和可解释性。

**🔧 技术方法**

使用技术包括 Vision Transformer、环境偏置调整模块、双分支内容/噪声解耦、正交约束、教师引导特征对齐、条件位置编码 CPE、以及高效的分块推理。

**📊 数据集**

使用的数据集包括 Synthetic Gaussian（CBSD68、Kodak24、McMaster、Urban100）和 Real‑World（SIDD、DND、Urban100）。

**📈 对比分析**

与多种 CNN、Transformer、SSM 复原器对比，TCD‑Net 在 PSNR/SSIM 上位居前列，且实时速度达 104.2 FPS，兼顾质量与效率。

**⚠️ 局限性**

局限在于对教师引导的依赖、在极端噪声或不同相机管线下的泛化仍有限，以及缺乏对更复杂分布漂移的理论保证。

---

## 228. ClinCoT: Clinical-Aware Visual Chain-of-Thought for Medical Vision Language Models

**arXiv ID:** 2603.01124 | [PDF](https://arxiv.org/pdf/2603.01124v1)

**作者:** Xiwei Liu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yutong Xie `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6449 | [OpenAlex ID](https://openalex.org/A5011835422)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于临床知识的视觉链式推理框架 ClinCoT，用于改进医学视觉-语言模型的事实一致性和区域级推理

**💡 创新点**

将偏好优化从仅纠正最终答案扩展到基于病理区域的中间推理链，结合区域生成、共识加权评分与边际感知优化，实现对视觉证据与临床推理的更紧密对齐

**🔧 技术方法**

多阶段自动数据生成（疾病条件区域提议与CoT生成）、多评估器共识评分、Margin‑Aware DPO 换算边际、迭代学习循环以及 LoRA 微调

**📊 数据集**

VQA‑RAD、SLAKE（医学 VQA 数据集）和 IU‑Xray（报告生成数据集）

**📈 对比分析**

与标准 DPO、Self‑Rewarding、STLLaVA‑Med、POVID、SIMA、FiSAO 以及医学专用 MMedPO 进行比较；在报告生成任务上 ClinCoT 取得最优成绩，在 VQA 任务上表现与 MMedPO 相当，SFT 后进一步提升

**⚠️ 局限性**

对单步 CoT 依赖较高，缺乏对区域精度的细粒度控制，且在短答题场景下中间推理可能导致不稳定；需要更高效的区域提议与评估器设计以适应更大规模数据

---

## 229. PM2Lat: Highly Accurate and Generalized Prediction of DNN Execution Latency on GPUs

**arXiv ID:** 2603.00549 | [PDF](https://arxiv.org/pdf/2603.00549v1)

**作者:** Truong-Thanh Le `[一作]` (University of Oslo), Peiyuan Guan `[通讯]` (University of Oslo)

**通讯引用:** 2102 | [OpenAlex ID](https://openalex.org/A5047837985)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了PM2Lat，一种基于GPU SIMT架构、内核区分与吞吐率插值的轻量级延迟预测框架；

**💡 创新点**

通过将不同实现细节视为独立内核并对其吞吐率进行插值建模，突破了传统基于FLOPs或单一深度学习模型的局限，实现了跨数据类型、跨内核、跨硬件的泛化；

**🔧 技术方法**

采用CUPTI和Nsight Compute对GPU内核进行实测、线性回归处理内存占用层、吞吐率插值预测、并利用Python/C++实现无GPU负载的CPU推理；

**📊 数据集**

在多种Transformer模型（GPT‑2、FLAN‑T5、Qwen‑3、DeepSeek‑R1）以及FP32/BF16两种精度、五款NVIDIA GPU（RTX‑3060M、T4、L4、A100、RTX‑5070）上收集实验数据；

**📈 对比分析**

与NeuSight对比，PM2Lat在层级预测误差平均低于10%，在自定义Triton/Flash/ Cutlass Attention内核误差保持在3–8%，且单次预测速度提升至0.045 ms（CPU）而非NeuSight的6.5 ms（GPU），模型级别误差亦维持在10%以内；

**⚠️ 局限性**

仅适用于静态DNN拓扑，无法处理如MoE等动态计算流、卷积等复杂层，且未考虑热阻、功耗等因素，且目前仅支持NVIDIA GPU。

---

## 230. Near-Optimal Regret for KL-Regularized Multi-Armed Bandits

**arXiv ID:** 2603.02155 | [PDF](https://arxiv.org/pdf/2603.02155v1)

**作者:** Kaixuan Ji `[一作]` (University of California), Quanquan Gu `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

研究了在多臂赌博机中加入KL正则化目标的在线学习问题，并给出了对应的上界与下界。

**💡 创新点**

首次实现了高正则化下的精确上界 Õ(η K log²T) 与下界 Ω(η K log T)，以及低正则化下的近似最优 Θ̃(√(KT log T))，填补了先前理论空白。

**🔧 技术方法**

采用了改进的 KL-UCB 算法、层化（peeling）技巧、Freedman 不等式以及精细的难例构造来实现高概率收敛分析。

**📊 数据集**

无实际数据集，全部为理论推导与分析。

**📈 对比分析**

与以往的 O(η K² log²T) 或 O(√(KT)) 上界相比，提出的方法在所有 η、K、T 维度上达到（至对数因子）最优，实验对比已在理论框架内验证。

**⚠️ 局限性**

仍存在对数因子差距（log T）未完全消除，仅适用于无结构的离散赌博机，未涵盖上下文、线性或对抗性环境。

---

## 231. Graph-centric Cross-model Data Integration and Analytics in a Unified Multi-model Database

**arXiv ID:** 2603.01598 | [PDF](https://arxiv.org/pdf/2603.01598v1)

**作者:** Zepeng Liu `[一作]` (Wuhan University), Zhiyong Peng `[通讯]` (Wuhan University)

**通讯引用:** 10554 | [OpenAlex ID](https://openalex.org/A5012603380)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个统一的多模型数据库，原生支持关系型、文档和图数据模型，并能高效执行图中心跨模型数据集成与分析（GCDIA）

**💡 创新点**

引入拓扑与属性双感知的图查询操作符、跨模型连接与并行分析框架，以及全局优化与查询重写规则，突破传统多模型数据库在图处理与分析上的瓶颈

**🔧 技术方法**

采用双存储引擎（统一记录存储+拓扑存储）、自定义混合遍历/模式匹配操作符、交叉模型 Join、基于块的向量化并行线性代数算子、成本模型与重写优化，以及在 openGauss 上实现的并行分析流水线

**📊 数据集**

使用专门为 GCDIA 设计的多模型基准 Benchmark，包含 17 个查询（t0‑t16），覆盖关系、文档和图三种模型，按不同规模因子（SF=1,2,5,10）扩展

**📈 对比分析**

与四类代表系统（MES、TBS、GNS、OLAP）在单机上进行对比，测量响应时间和吞吐量。新系统在图集成查询上最高可 107.89 倍加速，在分析任务上最高可 356.72 倍加速，整体平均速度提升 10.89×（GCDI）和 37.79×（GCDA）

**⚠️ 局限性**

目前仍为单机实现，缺乏分布式支持；分析任务仅覆盖部分线性代数算子，未覆盖聚合或随机访问式分析；对最短路径等纯拓扑查询的优化不足，且仅在开源环境中实现，缺乏商业化验证

---

## 232. Uniform Agent-interpolation of Distributed Knowledge

**arXiv ID:** 2603.01146 | [PDF](https://arxiv.org/pdf/2603.01146v1)

**作者:** Youan Su `[一作]` `[通讯]` (Liaoning University), Youan Su (Liaoning University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

证明了分布式知识的多代理模态逻辑 K_D、K_DD、K_TD 在统一插值性质（UIP）上的可证明性。

**💡 创新点**

首次将代理符号纳入插值公式，实现了同时消除代理和原子命题的统一插值。

**🔧 技术方法**

采用了纯粹的证明论技术：构造新的终结递归的序列算子（Sequent Calculus）与 𝒜-公式合成算法，并证明了 cut 的可接受性。

**📊 数据集**

本研究未使用实验数据集，全部工作基于形式证明。

**📈 对比分析**

由于仅为理论证明，未进行实验对比；理论上该方法可在所有满足条件的模型中构造统一插值，性能表现取决于推理树的深度。

**⚠️ 局限性**

局限性包括：仅适用于 K_D、K_DD、K_TD；在 S5 和直觉主义模态逻辑中缺乏 cut 消除；对大规模公式的计算复杂度未给出上界。

---

## 233. DUCX: Decomposing Unfairness in Tool-Using Chest X-ray Agents

**arXiv ID:** 2603.00777 | [PDF](https://arxiv.org/pdf/2603.00777v1)

**作者:** Zikang Xu `[一作]` (Institute of Artificial Intelligence), Xiaoxiao Li `[通讯]` (University of British Columbia)

**通讯引用:** 5152 | [OpenAlex ID](https://openalex.org/A5100458648)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于工具的胸部X光代理（MedRAX）进行系统公平性审计，构建阶段性偏差分解框架，将端到端差异拆分为工具暴露偏差、工具转换偏差和LLM推理偏差。

**💡 创新点**

提出“Decomposing Unfairness in Chest X-ray agents”框架，首次在代理系统中定义并量化三类偏差；构建新的MIMIC‑FairnessVQA基准；在五种驱动LLM上进行统一评估。

**🔧 技术方法**

利用MedRAX的ReAct式代理、LLM驱动规划器、分类器/分割器/报告生成器等工具集合；对工具暴露、工具转换和LLM推理分别设计度量；使用外部LLM Judge评估推理质量；采用统计显著性检验。

**📊 数据集**

CheXAgentBench（已存在的代理友好型数据集）以及新构建的MIMIC‑FairnessVQA（从MIMIC‑CXR采样并生成多选问答）。

**📈 对比分析**

通过ACC、ΔACC、DP、EoD、FUT等端到端指标和工具暴露/转换/推理偏差度量进行对比；结果显示Qwen系列LLM表现最佳，但整体仍存在显著子组差距（最高ΔACC≈20.8%，工具暴露差距可达50%）。

**⚠️ 局限性**

仅评估两种基准和五种LLM；仅关注性别和年龄两种属性；工具列表固定，未覆盖更广泛工具；未给出具体偏差缓解方案；依赖外部LLM Judge的质量；实验结果对不同代理架构的泛化能力有限。

---

## 234. Reward-Modulated Local Learning in Spiking Encoders: Controlled Benchmarks with STDP and Hybrid Rate Readouts

**arXiv ID:** 2603.00710 | [PDF](https://arxiv.org/pdf/2603.00710v1)

**作者:** Debjyoti Chakraborty `[一作]` `[通讯]`, Debjyoti Chakraborty

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在固定种子、严格可复现的实验框架下，评估了基于局部STDP奖励塑造的竞争性代理和使用同一神经编码器的本地率读出两种局部学习方法，并系统探究了归一化调度与奖励塑形的交互效应；

**💡 创新点**

首次通过可复现的固定种子协议，阐明归一化调度强度是控制本地学习性能的主导因子，并揭示奖励塑形效应随归一化状态和数据集变化而逆转；

**🔧 技术方法**

采用LIF网络编码的Poisson神经元群、三因子STDP规则、基于奖励的竞争性更新、以及带归一化调度的线性本地读出；

**📊 数据集**

主要在8×8手写数字数据集（digits）上进行实验，并对MNIST进行外部验证，此外还使用合成的时序分类任务检验计数读出与时序读出的差异；

**📈 对比分析**

与像素级Logistic回归（≈98%）和MLP（≈98%）基线相比，本地模型在默认设置下约86%准确率，归一化关闭后提升至≈95%；计数读出在时序任务中仅达50%而时序读出可达85%；

**⚠️ 局限性**

局部学习模型仍低于经典像素基准，STDP代理仅为简化的竞争性近似，缺乏完整的E/I重建；计数读出无法捕捉时序信用，需采用时序读出；实验范围受限于固定数据集和种子，未覆盖更复杂或多样化任务。

---

## 235. B$^2$F-Map: Crowd-sourced Mapping with Bayesian B-spline Fusion

**arXiv ID:** 2603.01673 | [PDF](https://arxiv.org/pdf/2603.01673v1)

**作者:** Yiping Xie `[一作]` (Zenseact), Gustaf Hendeby `[通讯]` (Linköping University)

**通讯引用:** 2334 | [OpenAlex ID](https://openalex.org/A5001137240)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了完整的基于生产车辆的 HD 地图生成流水线 B^2F-Map，使用单目相机、消费级 GNSS 与 IMU，完全不依赖先验 HD 地图。

**💡 创新点**

创新点包括：① 用 B‑spline 作为车道线表示，② 采用带 Gibbs 采样的 TPMB 多扩展对象追踪实现鲁棒的数据关联，③ 设计了 Bayesian B‑spline 融合算法，可在不同密度下融合带不确定性的 B‑spline 轨迹，实现几何一致且不冗余的地图融合。

**🔧 技术方法**

使用技术包括：B‑spline 控制点高斯建模、EOT Poisson 多伯努利过滤器、Gibbs 采样、图优化定位、信息形式的贝叶斯融合、伪测量网格搜索等。

**📊 数据集**

实验数据集为两个欧洲城市的真实数据，总长 70 km，包含 8 条行驶轨迹，使用高精度定位系统做 ground‑truth，并与专业测绘车辆提供的 HD 地图对标。

**📈 对比分析**

与基线（聚类 + 三次样条拟合）对比，Gibbs 采样的多车道追踪显著减少错误关联；融合后绝对误差约 0.58 m、相对误差约 0.12 m，优于基线，表明几何精度与相对精度均得到提升。

**⚠️ 局限性**

主要限制在于缺少车道拓扑信息，且当前的实时性能仍有待改进。

---

## 236. MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning

**arXiv ID:** 2603.02024 | [PDF](https://arxiv.org/pdf/2603.02024v1)

**作者:** Jiachun Li `[一作]` (University of Chinese Academy of Sciences), Jun Zhao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MMR-Life基准，用多图真实生活场景评估多模态大语言模型的推理能力

**💡 创新点**

首次覆盖七种推理类型且聚焦多图输入，消除知识依赖，提供真实场景评估

**🔧 技术方法**

采用多模态链式思考(CoT)、强化学习、最佳样本等推理增强技术

**📊 数据集**

收集19,108张真实图像，构成2,646道多选题

**📈 对比分析**

对37款先进模型进行零样本CoT评估，最强GPT-5仅58.7%，表现明显落后于人类

**⚠️ 局限性**

缺乏多图推理的预训练数据，模型对空间与时序推理表现弱，增强方法在大模型上效果有限

---

## 237. A Novel Reconfigurable Dexterous Hand Based on Triple-Symmetric Bricard Parallel Mechanism

**arXiv ID:** 2603.00892 | [PDF](https://arxiv.org/pdf/2603.00892v1)

**作者:** Chunxu Tian `[一作]` (Fudan University), Dan Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 20439 | [OpenAlex ID](https://openalex.org/A5100456041)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一种基于三对称Bricard连杆的可重构并行多指机械手，并通过原型验证其抓取性能。

**💡 创新点**

创新点在于将三对称Bricard机制作为可变手掌，结合并行机构实现大工作空间与高刚性，并利用Screw理论和闭环约束完成完整运动学与传动分析。

**🔧 技术方法**

采用结构拓扑合成、Denavit–Hartenberg闭环条件、Screw理论、Jacobian与刚度分析、Arduino/Servo驱动、线性导轨和触觉压力传感器。

**📊 数据集**

实验数据基于自制抓取物料集合，尺寸从17 mm到180 mm，包含瓶子、鼠标、玩具等多形状、多重量物体。

**📈 对比分析**

通过在不同手掌开合状态（收缩、半收缩、扩展）下进行抓取成功率对比，实验表明全扩展状态成功率最高，收缩状态对小体积物体更优；传动效率用κ指标评估，刚度始终为正，工作空间宽广。

**⚠️ 局限性**

局限性：仅实现三指结构，未覆盖多指复杂抓取；电缆驱动可能产生耦合；实验范围有限，缺乏大规模真实场景验证。

---

## 238. Political attitudes differ but share a common low-dimensional structure across social media and survey data

**arXiv ID:** 2603.02102 | [PDF](https://arxiv.org/pdf/2603.02102v1)

**作者:** Antoine Vendeville `[一作]` (Sciences Po), Pedro Ramaciotti `[通讯]` (Complex Systems Institute of Paris Ile-de-France CNRS)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对法国X平台（前Twitter）用户与ESS调查样本的政治态度进行对比，探讨在线与离线的意识形态极化与议题一致性，并分析活跃度、受欢迎度和可见度对态度分布的影响。

**💡 创新点**

首次将社交媒体交互嵌入空间与CHES专家调查对齐，构建多维态度空间；揭示两者均呈双维结构（左-右与全球-本土分离）；证明可见度高的用户态度更贴近普罗大众；将活跃度与受欢迎度的双重维度纳入政治心理学研究。

**🔧 技术方法**

采用意识形态尺度（Barberá法）与线性回归映射至CHES维度；使用熵基极化度量、Wasserstein距离、Pearson相关、主成分分析（PCA）与有效维度（ED）等统计方法；对用户按活跃度、受欢迎度和可见度进行分箱分析。

**📊 数据集**

X平台数据：2023年法国议员及其粉丝（约978,000名关注者）及其交互记录；ESS 2023法国面板（1,417名受访者）以及CHES 2019/2023的党派立场数据。

**📈 对比分析**

通过对比分布的熵极化值、Wasserstein距离和相关矩阵来衡量两样本间差异；PCA解释方差表明X用户90%以上的变异可由两维解释，ESS仅50%；活跃度增大时极化值提升、维度压缩；受欢迎度提升时与ESS的Wasserstein距离减小，表明更具代表性；可见度加权后，X平台分布更趋中立、极化度下降。

**⚠️ 局限性**

主要局限：X平台态度基于互动推断，缺乏自报验证；未考虑算法推荐对可见度的影响；可见度定义仅为活跃度×受欢迎度，忽略时间窗口和内容质量；样本仅限议员关注者，可能与一般公众存在人口统计偏差；不同测量尺度的转换可能引入误差。

---

## 239. AEDHunter: Investigating AED Retrieval in the Real World via Gamified Mobile Interaction and Sensing

**arXiv ID:** 2603.01075 | [PDF](https://arxiv.org/pdf/2603.01075v1)

**作者:** Helinyi Peng `[一作]` (University of Tokyo), Kaoru Sezaki `[通讯]` (University of Tokyo)

**通讯引用:** 3934 | [OpenAlex ID](https://openalex.org/A5050720322)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并评估了一款基于位置的游戏化移动应用 AEDHunter，旨在让公众在真实校园环境中快速定位并获取自动体外除颤器 (AED)。

**💡 创新点**

创新点在于将游戏化元素与手机传感器、低成本蓝牙标签相结合，利用“探索暂停”作为行为学习信号，并通过两状态运动分类器实时验证用户到达 AED 的瞬间，从而在训练中实现客观、细粒度的检索表现评估。

**🔧 技术方法**

使用了手机的加速度、陀螺仪、磁力计、GPS、Wi‑Fi BSSID、蓝牙低功耗 beacon 以及低功耗计数器等传感器，并实现了轻量级 SVM 两状态运动分类器。

**📊 数据集**

数据集包括两所东京大学校园的 AED 位置信息（共 19 台）以及 20 名参与者在 3 周内完成的 228 次检索（含传感日志和问卷）。

**📈 对比分析**

通过前后测评、Wilcoxon 符号秩检验以及 ΔDT（检索时间减少）和 ΔDP（探索暂停减少）等指标比较，实验显示平均检索时间从 132.7 s 降至 97.3 s（ΔDT≈0.39），探索暂停减少 40%（ΔDP≈0.40），且 Map 与 No‑Map 两组在检索效率上差异不显著。

**⚠️ 局限性**

局限性包括样本量有限、仅在年轻校园人群中验证、缺少高压真实急救情境、长周期记忆与保留效果仍待进一步研究。

---

## 240. Does Travel Stage Matter? How Leisure Travellers Perceive Their Privacy Attitudes Towards Personal Data Sharing Before, During, and After Travel

**arXiv ID:** 2603.01992 | [PDF](https://arxiv.org/pdf/2603.01992v1)

**作者:** Haiyue Yuan `[一作]` (Institute of Cyber Security for Society), Xiao Ma `[通讯]` (Centre for Business and Industry Transformation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对 318 名休闲旅行者进行在线问卷调查，研究其在旅行前、旅行中和旅行后三个阶段对个人数据共享的隐私态度与社交媒体分享行为的差异。

**💡 创新点**

创新点在于首次系统比较同一旅行者在不同旅行阶段对不同数据类型和共享目的的隐私感知差异，并揭示性别、旅行频率与居住国对共享偏好的影响。

**🔧 技术方法**

采用非参数检验（Mann‑Whitney U、Wilcoxon 符号秩检验）、卡方检验、三因素 ART‑ANOVA 以及多元逻辑回归等统计方法对问卷数据进行分析。

**📊 数据集**

使用的数据集为 318 位来自欧洲（约 66%）和南非（约 33%）的受访者在 Prolific 平台上完成的问卷结果，包含 31 种个人数据类型、3 种共享目的及多项社交媒体使用频率等信息。

**📈 对比分析**

通过显著性检验与回归模型评估，发现数据类型、共享目的、性别、旅行频率和居住国均显著影响共享态度；模型整体显著性（p ≤ 0.001），但对旅行阶段本身的影响不显著。

**⚠️ 局限性**

局限性包括：① 仅基于自我报告的问卷，可能与实际行为不一致；② 样本分布不均，欧洲样本占主导；③ 未涉及即时通讯工具；④ 未能捕捉到主动与被动共享行为的细微差异。

---

## 241. Reasoning or Rationalization? The Role of Justifications in Masked Diffusion Models for Fact Verification

**arXiv ID:** 2603.01190 | [PDF](https://arxiv.org/pdf/2603.01190v1)

**作者:** Jacob Devasier `[一作]` (University of Texas at Arlington), Jacob Devasier `[通讯]` (University of Texas at Arlington)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5063617278)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了掩码扩散语言模型在事实验证任务中的推理动态，发现它们往往在扩散早期就确定结论。

**💡 创新点**

提出“延迟裁决解码”干预并揭示了所谓的“精炼漂移”现象，即后续生成的推理链会削弱初始正确判断。

**🔧 技术方法**

使用非因果的掩码扩散架构 LLaDA-8B，结合交叉实验和因果干预分析。

**📊 数据集**

在 AVeriTeC 真实世界事实验证数据集上进行评估。

**📈 对比分析**

与 LLaMA 3.1 8B 和 Qwen3-8B 基线相比，LLaDA-8B 在不受输出顺序影响时取得 86.2% 的准确率，但强制延迟解码会下降至 71.9%。

**⚠️ 局限性**

局限性包括仅在 LLaDA-8B 上实验、对复杂多步推理任务缺乏验证，以及不一定能推广至其他扩散模型。

---

## 242. Frozen Policy Iteration: Computationally Efficient RL under Linear $Q^π$ Realizability for Deterministic Dynamics

**arXiv ID:** 2603.00716 | [PDF](https://arxiv.org/pdf/2603.00716v1)

**作者:** Yijing Ke `[一作]` (Peking University), Ruosong Wang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在确定性转换、随机初始状态和随机奖励的 MDP 上，提出了一种计算效率高且样本效率优良的在线强化学习算法 Frozen Policy Iteration（FPI），并给出了其子弹性和 Uniform‑PAC 性能分析；在实验中证明其在 CartPole-v1 与 InvertedPendulum-v4 上优于不使用冻结机制的基线。

**💡 创新点**

核心创新是“冻结”机制：对已充分探索的状态-动作对仅保留最早加入的数据，从而保证所有训练样本始终有效为 on‑policy，避免了传统算法对重采样的依赖；此外，通过多层次精度约束实现了 O(√T) 的 regret，同时实现了 Uniform‑PAC 与 bounded eluder dimension 的推广。

**🔧 技术方法**

主要技术包括：线性 Q^π 可实现假设下的最小二乘估计、椭圆势能引理与自归一化过程的高概率界、对高置信度区域的定义与利用、以及多层次精度（ε = 2^{‑l}) 的自适应探索。

**📊 数据集**

实验使用 OpenAI Gym 的 CartPole‑v1 与 InvertedPendulum‑v4 环境，采用 4 个子块的平面编码（tile coding）生成特征映射，并对连续动作空间做离散化处理。

**📈 对比分析**

与不冻结版本的对比实验表明，冻结策略显著提升学习曲线，算法在相同训练步骤下获得更高累计奖励；理论上，算法在线性情境下的 regret 为 O(√(d²H⁶T))，与上下界一致，且在 H = 1 的情形下达到最佳 bandit 结果。

**⚠️ 局限性**

局限性包括：对转移函数的确定性假设，导致无法直接推广到随机转移；对横向 H 的多项式依赖较大；仅适用于有限动作空间和已给定的特征映射；在高维连续状态空间下特征设计仍是挑战。

---

## 243. Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining

**arXiv ID:** 2603.01465 | [PDF](https://arxiv.org/pdf/2603.01465v1)

**作者:** Yipeng Chen `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 30979 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Keyframe‑Chaining VLA框架，通过关键帧链提取与联结稀疏语义历史，解决机器人非马尔可夫长周期任务的状态别名问题。

**💡 创新点**

创新点在于：1）引入Task‑Modulated Keyframe Selection Module (KSM) 结合FiLM调制多任务嵌入；2）使用统一度量学习与交叉注意力实现实时关键帧检索；3）通过贪婪时间平滑保证关键帧的时序可靠性；4）将稀疏关键帧直接注入VLA，实现长程依赖而不增加注意力复杂度。

**🔧 技术方法**

技术包括：ResNet‑18视觉编码器；Triplet Margin Loss 的统一度量学习；FiLM + 交叉注意力的任务调制查询；贪婪时间平滑；Flow‑Matching 生成的动作头；Prompt Refinement 以结构化提示引导多模态推理。

**📊 数据集**

数据集：在 ManiSkill 仿真器上构建的四个非马尔可夫长周期基准任务；真实世界实验使用 AgileX Piper 机械臂通过示范收集的专家演示数据。

**📈 对比分析**

与 Diffusion Policy、π_0、GR00T 的不同历史长度（无历史、短期 1–3 帧、长期固定步长 5–100 帧）进行对比；在模拟基准中平均成功率达 92%（对照最佳基线 57%）；在真实世界 20 次试验中，成功率和完成率显著高于基线，尤其在多阶段任务上表现突出。

**⚠️ 局限性**

局限性：关键帧以原始像素形式存储，导致存储冗余且随时间线性增长；目前缺乏压缩或离散化的关键帧表示；未来需要探索动态内存更新与更紧凑的特征压缩方法。

---

## 244. Capstone: Power-Capped Pipelining for Coarse-Grained Reconfigurable Array Compilers

**arXiv ID:** 2603.00909 | [PDF](https://arxiv.org/pdf/2603.00909v1)

**作者:** Sabrina Yarzada `[一作]` (University of Southern California), Christopher Torng `[通讯]` (University of Southern California)

**通讯引用:** 526 | [OpenAlex ID](https://openalex.org/A5026966440)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

在CGRA编译器Cascade基础上，提出了Capstone框架，实现了编译时的功耗预测与功耗上限控制；

**💡 创新点**

创新点在于：①使用基于门级功耗监督的层次化机器学习能耗模型，能够在编译循环内快速预测功耗；②设计了三种功耗控制器（保守保留、分形不确定性约束、误差界定），实现不同可靠性与性能折衷；

**🔧 技术方法**

技术包括：编译器可见事件提取、门级功耗监督学习（W、α矩阵）、非负最小二乘/岭回归更新、保守保留与分形不确定性约束、功耗控制决策与管线深度搜索；

**📊 数据集**

数据集为32×16的Intel 16nm CGRA织物上，使用Halide/Clockwork生成的稠密内核和Custard生成的稀疏内核，共计若干基准；

**📈 对比分析**

与原始Cascade和基于阈值节能的其他编译器比较，Capstone在满足功耗上限的同时，频率损失仅在1–10%之间；在多种功耗上限下，Capstone始终实现功耗合规，且多种控制器提供从保守到近乎最优的性能权衡；

**⚠️ 局限性**

局限性包括：需要门级功耗数据进行模型训练，训练集覆盖不足时误差可能增大；多位比特流产生会增加运行时配置成本；编译时间提升最多约20%，在极端功耗要求下仍需手动调校保留系数。

---

## 245. VoiceAgengRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures

**arXiv ID:** 2603.02206 | [PDF](https://arxiv.org/pdf/2603.02206v1)

**作者:** Jielin Qiu `[一作]` (Salesforce AI Research), Huan Wang `[通讯]` (Salesforce AI Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出双代理架构，将检索预取与生成解耦，以降低实时语音代理的检索延迟。

**💡 创新点**

创新点在于后台代理基于LLM预测后续话题并预取文档到语义缓存，前台直接从缓存读取，实现316×检索加速。

**🔧 技术方法**

采用FAISS内存语义缓存、GPT‑4o‑mini进行预测与生成、Qdrant Cloud向量数据库、Whisper STT 与 Edge TTS 等技术。

**📊 数据集**

使用合成企业知识库“NovaCRM”共76个文档块，并在10个多轮对话场景（共200轮）进行评估。

**📈 对比分析**

在200个查询上与传统RAG基线对比，缓存命中率达75%，检索延迟从110.4 ms降至0.35 ms，总检索时间节省16.5 秒。

**⚠️ 局限性**

主要局限在于LLM生成仍主导总延迟，预测误差导致预取资源浪费，缓存一致性和冷启动仍需改进。

---

## 246. SageBwd: A Trainable Low-bit Attention

**arXiv ID:** 2603.02170 | [PDF](https://arxiv.org/pdf/2603.02170v1)

**作者:** Jintao Zhang `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究低位量化注意力在大规模预训练中的可行性，并提供在一定 tokens-per-step 下可与全精度相当的训练方案。

**💡 创新点**

提出 QK-norm 与对软max梯度（dS）量化敏感性分析，证明在较小 tokens-per-step 时可恢复全精度性能，同时指出 K-smoothing 对稳定性至关重要。

**🔧 技术方法**

采用 INT8 量化、QK-norm、K-smoothing、FlashAttention 核实现，并在 Triton/XFormers 环境下实现高效前向/后向核。

**📊 数据集**

使用 78B OpenWebText 数据集训练 325M Llama 模型。

**📈 对比分析**

与全精度注意力（FPA）对比，2.1M tokens/step 需要 QK-norm 并仍略逊于 FPA；260K tokens/step 能匹配 FPA；相较 FlashAttention2 提升 1.5‑1.7 倍吞吐。

**⚠️ 局限性**

在大 batch / tokens-per-step 时训练不稳定，后向量化误差（尤其是 dS）难以抑制，导致性能落后。

---

## 247. Egocentric Co-Pilot: Web-Native Smart-Glasses Agents for Assistive Egocentric AI

**arXiv ID:** 2603.01104 | [PDF](https://arxiv.org/pdf/2603.01104v1)

**作者:** Sicheng Yang `[一作]` (Tsinghua University), Zhensong Zhang `[通讯]` (Independent Researcher)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了Egocentric Co-Pilot——一种可在智能眼镜上运行的网络原生神经-符号框架，用于实现以第一人称视角为核心的持续式辅助服务。

**💡 创新点**

创新点在于：①将大语言模型作为核心调度器，通过MCP协议组合多种感知、推理与Web工具；②引入Temporal Chain-of-Thought与Hierarchical Context Compression两级上下文管理，实现超长视频的长时序推理；③使用WebRTC（LiveKit）实现音视频与控制消息的统一流式通道，兼容云端与边缘部署。

**🔧 技术方法**

技术要点包括：大语言模型（LLM）+视觉语言模型、神经感知模块、符号推理引擎；MCP（轻量JSON工具调用协议）；WebRTC+LiveKit实时流；Temporal Chain-of-Thought (T‑CoT) 与 Hierarchical Context Compression (HCC)。

**📊 数据集**

主要使用的公开数据集为Egolife和HD‑EPIC，用于评估长时序问答与动作推理性能。

**📈 对比分析**

与当前SOTA方法相比，Egocentric Co-Pilot在Egolife上达到40.9%准确率（高于GPT‑4o 36.2%），在HD‑EPIC上取得46.2%准确率（高于Gemini‑1.5‑Pro 37.6%），并在智能眼镜上的人机评测中获得平均4.70分（接近人类基线4.92），显著优于商业同类设备。

**⚠️ 局限性**

局限性包括：对底层LLM/VLM性能高度依赖，感知错误与意图歧义仍会导致失败；云端流式处理带来延迟与能耗问题；缺乏长期与多样化人群（如残障老年人）的评估；隐私与旁观者同意等安全与伦理问题尚未完善。

---

## 248. Curvature-Weighted Capacity Allocation: A Minimum Description Length Framework for Layer-Adaptive Large Language Model Optimization

**arXiv ID:** 2603.00910 | [PDF](https://arxiv.org/pdf/2603.00910v1)

**作者:** Theophilus Amaefuna `[一作]` (University of South Florida), Ankur Mali `[通讯]` (University of South Florida)

**通讯引用:** 1943 | [OpenAlex ID](https://openalex.org/A5089064325)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于曲率的层级容量分配与稀疏化框架，利用MDL原理在硬件预算约束下实现大语言模型的层级容量最优分配和参数剪枝。

**💡 创新点**

创新点在于定义了曲率调整后的层级收益量化指标 ζ_k² = g_kᵀH_kk⁻¹g_k，将其转化为凸优化问题并给出唯一闭式解，从而实现可解释、可转移且具备理论最优性与泛化保证的容量分配与剪枝策略。

**🔧 技术方法**

主要技术包括二阶Taylor展开、Hessian的Tikhonov正则化、曲率加权的层级质量分数 q_k、MDL驱动的凸分配/剪枝程序、单维双分支搜索求解 λ、以及影响函数估计与LoRA‑MoE、K‑FAC等近似Hessian方法。

**📊 数据集**

实验使用了 Mistral‑7B 和 Gemma‑7B 两个 7B 级大语言模型，并在 CoLA、MRPC、CommonsenseQA、ScienceQA、OpenBookQA 等分类/问答基准，以及 C4（校准）和 RTE、ARC‑Easy、ARC‑Challenge、HellaSwag、BoolQ、WinoGrande 等零样本评测数据集。

**📈 对比分析**

与先前的 LayerIF 经验启发式方法对比，MDL 分配在 Mistral‑7B 上平均提升约 2.6%（83.07% vs 80.41%），在 Gemma‑7B 上提升约 0.06%；剪枝时 MDL 与 LayerIF 的平均零样本准确率基本持平（60.18% vs 60.18%），某些配置略有优势或劣势。

**⚠️ 局限性**

局限性包括对曲率估计（Hessian 近似）与影响函数质量的依赖、对 LoRA‑MoE 结构的特定假设、剪枝时对二次退化模型可能低估敏感性、以及在极端硬件预算或不同模型架构下的可扩展性尚待验证。

---

## 249. ViTex: Visual Texture Control for Multi-Track Symbolic Music Generation via Discrete Diffusion Models

**arXiv ID:** 2603.01984 | [PDF](https://arxiv.org/pdf/2603.01984v1)

**作者:** Xiaoyu Yi `[一作]` (Peking University), Ziyu Wang `[通讯]` (New York University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了可视化的乐器纹理表示ViTex，并在此基础上训练了一个离散扩散模型，实现对多轨符号音乐的纹理级可控生成。

**💡 创新点**

创新点在于：①将乐器选择与音高/时序信息统一映射为彩色空间的二维可视化；②通过规则化的纹理特征（旋律、和声、持续）构造低维可读的条件向量；③在离散扩散框架中引入ViTex与和弦条件，结合分类器无关引导实现可插值的条件与无条件生成。

**🔧 技术方法**

使用了离散吸收状态扩散模型、UNet结构、分类器无关指导、可视化纹理映射、以及基于规则的音符分类与可持续度判定。

**📊 数据集**

数据集为从Lakh MIDI与Meta MIDI中筛选出的7,175首符合条件的8小节（32拍）多轨曲目，包含钢琴、吉他、贝司、鼓等11种乐器类别。

**📈 对比分析**

与基线Q&A（两种文本化实现）、MMT、AMT等方法对比，利用Instrumentation Accuracy、Chord Accuracy、Overlapped Area、DOA、PCE、GPS等指标评估。实验表明：在可控性指标上（IA、CA、OAD、OAIOI、OAP）优于基线；在无条件生成质量（DOA、GPS）上与最优方法持平，PCE略低。

**⚠️ 局限性**

局限性包括：①对纹理细节的规则化划分可能无法完全捕捉复杂的音乐表达；②在生成连贯性上与专门训练的填充模型（AMT）相比略逊；③对高分辨率或更长片段的扩展尚未验证，需进一步研究模型规模与训练策略。

---

## 250. An Open-Source Modular Benchmark for Diffusion-Based Motion Planning in Closed-Loop Autonomous Driving

**arXiv ID:** 2603.01023 | [PDF](https://arxiv.org/pdf/2603.01023v1)

**作者:** Yun Li `[一作]`, Manabu Tsukada `[通讯]` (University of Tokyo)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5067716610)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个开源的模块化扩散模型运动规划基准，将单一的18,398节点ONNX图拆分为可单独执行的编码器、DiT核心和转弯指示器模块，并在Autoware+AWSIM闭环仿真环境中进行评估。

**💡 创新点**

①将迭代 denoising 循环从模型图中拆离，允许运行时动态配置步数、求解器阶数和噪声调度；②用 C++ 重实现 DPM‑Solver++，实现编码器缓存，大幅降低延迟；③提供可插拔求解器接口，支持在真实闭环堆栈下直接比较不同求解器策略。

**🔧 技术方法**

ONNX GraphSurgeon 图拆分、ONNX Runtime 与 C++ DPM‑Solver++、ROS 2 与 Autoware 集成、AWSIM 闭环仿真、VP 噪声调度、DPM‑Solver++、DDIM 等技术。

**📊 数据集**

使用 nuPlan 作为模型训练基准；在 AWSIM 的 Nishishinjuku 城市路段场景中收集输入数据进行闭环评估。

**📈 对比分析**

对 N∈{3,5,7,10,15,20} 步长、第一阶与第二阶 DPM‑Solver++、以及 DDIM 进行离线再现；评估 FDE、ADE；编码器缓存后推理时间从 328 ms 降至 53 ms，第二阶求解器在 N=3 时 FDE 下降 41%，所有模块化配置均满足 100 ms 规划预算。

**⚠️ 局限性**

仅在单一城市场景进行测试，未完成 GPU 延迟全面分析；低步数下跨模块传输开销未评估；对其他扩散规划器的可迁移性需进一步验证。

---

## 251. Constitutional Black-Box Monitoring for Scheming in LLM Agents

**arXiv ID:** 2603.00829 | [PDF](https://arxiv.org/pdf/2603.00829v1)

**作者:** Simon Storf `[一作]` (MATS), Marius Hobbhahn `[通讯]` (Apollo Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了使用宪法式提示的黑盒监控器，利用两条合成数据生成管道（STRIDE 与 Gloom）训练模型，仅依据外部可观测行为检测 LLM 代理的“阴谋”行为，并在半真实环境 ControlArena 上进行评估。

**💡 创新点**

创新点包括：①提出 STRIDE 与 Gloom 两种可扩展的合成轨迹生成方法；②证明仅基于合成数据训练的监控器能够跨域迁移到更真实的测试环境；③系统性展示在该任务中进一步的提示优化会导致过拟合，揭示提示式监控的性能上限。

**🔧 技术方法**

技术手段主要有：宪法式提示生成、提示网格搜索（grid search）、人类提示细化、DSPy 自动提示优化（GEPA、MIPROv2）、基于 pAUROC 的低 FPR 评估，以及对不同 LLM（Claude Sonnet 4.5、Gemini 2.5 Pro、GPT‑5.1）和多模型组合的评估。

**📊 数据集**

使用的数据集包括：STRIDE 合成训练集 1,000 条轨迹；Gloom 合成训练集 1,000 条轨迹；ControlArena 半真实测试集 7,500 条轨迹，涵盖 GPT‑5、Claude Sonnet 4 与 Gemini 2.5 Pro 在五种不同沙盒环境中的表现。

**📈 对比分析**

在实验中将不同提示与模型组合在合成数据上进行网格搜索、人工细化和自动优化，随后在 ControlArena 上评估。结果显示，简单的网格搜索已能获得 pAUROC ≈ 0.85‑0.87 的性能，额外的提示细化或自动优化往往不提升甚至下降测试集性能，表明已接近性能上限。

**⚠️ 局限性**

主要局限包括：①合成数据分布可能过于狭窄或包含噪声特征，导致监控器难以学习稳健的阴谋信号；②提示式调优的全局性限制使得难以在不引入其他错误的前提下精细化判别；③在更复杂或真正部署环境中，阴谋行为的多样性与合成场景可能差异更大，需进一步改进数据质量或考虑模型微调等替代方案。

---

## 252. LiTS: A Modular Framework for LLM Tree Search

**arXiv ID:** 2603.00631 | [PDF](https://arxiv.org/pdf/2603.00631v1)

**作者:** Xinzhe Li `[一作]` (RMIT University), Yaguang Tao `[通讯]` (RMIT University)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5009243103)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LiTS，一个可模块化的 Python 框架，专门用于将 LLM 推理拆解为 Policy、Transition、RewardModel 三个可复用组件，并支持 MCTS、BFS 等多种树搜索算法；同时提供了装饰器式注册机制，使领域专家和算法研究者可以各自扩展任务逻辑或搜索算法，而无需修改核心代码；通过跨任务（语言、环境、工具使用）演示了组件和算法的互操作性；并在无限动作空间下发现 LLM 策略多样性是树搜索的主要瓶颈。

**💡 创新点**

① 将 LLM 推理与树搜索解耦为三类通用组件，打破任务特定的 monolithic 设计；② 提供装饰器注册机制，支持无侵入式的领域扩展与搜索算法注册；③ 在无限动作空间中揭示策略多样性瓶颈，首次系统性分析模式坍塌问题；④ 通过统一的 PromptRegistry 与 InferenceLogger 实现可观测性与可复现性。

**🔧 技术方法**

Python 3、装饰器注册、抽象基类、LangChain 兼容工具协议、OpenAI / Claude 3.5 Sonnet / Llama3-8B LLM 调用、MCTS、BFS、ReAct、ReST-MCTS、RAP 等树搜索/链式推理算法；使用 InferenceLogger 记录 token 与 latency，Checkpointing 支持后期分析。

**📊 数据集**

MATH500（数值数学问答）、Crosswords（填字游戏）、MapEval-SQL（工具使用评估）、BlocksWorld、以及自定义数据集加载器。

**📈 对比分析**

在 MATH500 上，BFS（10 次迭代、branch=3）达 39% 准确率，MCTS 37%；CoT 仅 17%；在 MapEval-SQL 上，ReAct 40% 而 MCTS 0%；在 Crosswords 上，MCTS 14K tokens（$2.42）相较 Chain 2.5K tokens（$0.28）成本更高但精度提升。跨域实验显示同一套 Policy/Transition/RewardModel 可直接用于不同算法，验证了组件复用与算法可互换。

**⚠️ 局限性**

① 无限动作空间导致的模式坍塌，LLM 策略多样性不足是关键瓶颈；② 工具使用场景中，奖励模型（LLM-as-judge）偏好冗长错误查询，削弱树搜索效果；③ 现有实现对 LLM 调用成本高、算力需求大；④ 对动态工具发现与自适应调用支持不足，未来需引入 MCP 或链式工具整合。

---

## 253. Let Your Image Move with Your Motion! -- Implicit Multi-Object Multi-Motion Transfer

**arXiv ID:** 2603.01000 | [PDF](https://arxiv.org/pdf/2603.01000v1)

**作者:** Yuze Li `[一作]` (Tianjin University), Xinyu Zhang `[通讯]` (University of Auckland)

**通讯引用:** 35047 | [OpenAlex ID](https://openalex.org/A5107306837)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出第一个支持多对象多动作的隐式图像到视频（I2V）运动转移框架，并实现了精确、可组合的运动控制。

**💡 创新点**

创新点包括：① Motion Decoupled Mask Attention Mechanism (MDMA) 用对象特定掩码在注意力层实现运动分离；② Differentiated Mask Extraction Mechanism (DMEM)，融合训练阶段的注意力掩码提取与推理阶段的 Regressive Mask Propagation Mechanism (RMPM)，以及其动态版本；③ 通过动态RMPM显著提升推理效率。

**🔧 技术方法**

使用了 CogVideoX-5B-I2V 的 3D Diffusion 框架，Motion Tokens、MDMA、DMEM 以及 RMPM；并利用 RAFT 光流估计计算 Flow Fidelity（FF）评估指标。

**📊 数据集**

构建了 200 组视频-图像配对数据集，来源于 FlexiAct、Pexels、DAVIS 2017、Seedream，涵盖 20 种不同动作，图像包含单/多对象场景。

**📈 对比分析**

与 FlexiAct、I2VEdit、AnyV2V、Go-with-the-Flow、CogVideoX-5B-I2V 等基线进行对比；在 Trajectory Fidelity（TF）和 Flow Fidelity（FF）上实现了最优表现，同时保持或提升 Appearance Consistency（AC）、Temporal Consistency（TC）和 Text Similarity（TS），并在人工评估中取得最高分。

**⚠️ 局限性**

局限性包括：① 对单对象场景的 AC/TC 指标可能不如专门针对单对象的模型；② 运动细节受文本描述限制，难以完全捕捉微观动作；③ 多对象场景下掩码提取仍可能失效，特别是当对象重叠或分辨率低时；④ 动态 RMPM 的阈值需要手动调优。

---

## 254. Qwen3-Coder-Next Technical Report

**arXiv ID:** 2603.00729 | [PDF](https://arxiv.org/pdf/2603.00729v1)

**作者:** Ruisheng Cao `[一作]`, Fan Zhou `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个80B总参数、3B激活参数的开放权重编程代理模型，通过大规模可执行任务合成与环境反馈学习，提升编码与代理能力；

**💡 创新点**

创新点在于将大规模可执行任务合成、mid‑training、强化学习与多模板工具调用训练相结合，配合低激活参数的MoE与混合注意力架构，实现高效、可扩展的代理模型；

**🔧 技术方法**

采用Mixture‑of‑Experts、Hybrid attention、可执行任务合成、BFP packing、FIM目标、强化学习、指令微调与专家蒸馏等技术；

**📊 数据集**

使用GitHub PRs、SWE‑Smith/SWE‑Flow/SWE‑Rebench/Multi‑SWE‑RL等可执行任务数据集，结合Common Crawl、SWE‑Bench、Terminal‑Bench 2.0、FullStackBench等多语言代码库；

**📈 对比分析**

在SWE‑Bench Verified/Multilingual/Pro、Terminal‑Bench 2.0、FullStackBench、MultiPL‑E、CRUXEval、Codeforces等基准上与Claude Opus 4.5、DeepSeek、GLM、MiniMax、Kimi等模型对比，3B激活模型在多项基准上与更大模型相当且表现优异；

**⚠️ 局限性**

与专有模型相比仍存在能力差距，难以处理极大规模复杂软件任务，长程规划效率不足，前端 UI/视觉能力有限，需要进一步扩大训练数据与计算资源。

---

## 255. Polynomial Mixing for Efficient Self-supervised Speech Encoders

**arXiv ID:** 2603.00683 | [PDF](https://arxiv.org/pdf/2603.00683v1)

**作者:** Eva Feillet `[一作]` (Université Paris-Saclay), Alexandre Allauzen `[通讯]` (Université Paris-Dauphine-PSL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种Polynomial Mixer（PoM）作为线性复杂度的token混合器，用于替换语音Transformer中的多头自注意力。

**💡 创新点**

创新点在于使用多阶多项式投影产生全局状态，并通过逐元素选择实现token级混合，既保留了表达能力，又将时间/内存复杂度从二次降为一次。

**🔧 技术方法**

采用PoM与最佳自监督预训练框架BEST‑RQ相结合，并在SpeechBrain中实现为可插拔模块。

**📊 数据集**

使用LibriSpeech‑960h进行预训练，LibriSpeech‑100h进行下游ASR微调。

**📈 对比分析**

在LibriSpeech测试集上，PoM的词错误率（WER）与标准MHA相近，优于SummaryMixing，且推理时间和显存占用显著低于MHA和RoPE；相对Mamba、HyperConformer等线性替代方案，PoM表现可比或略优。

**⚠️ 局限性**

限制在于仅在少量预训练步骤（200k）下验证，可能无法达到最先进的WER；对不同语音任务和流式设置的适用性尚未充分评估；并且PoM的超参数（k、D、分频分组）需要进一步调优。

---

## 256. Federated Agentic AI for Wireless Networks: Fundamentals, Approaches, and Applications

**arXiv ID:** 2603.01755 | [PDF](https://arxiv.org/pdf/2603.01755v1)

**作者:** Lingyi Cai `[一作]` (Huazhong University of Science and Technology), Abbas Jamalipour `[通讯]` (University of Sydney)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出将联邦学习（Federated Learning）融合到代理式 AI（Agentic AI）的感知–记忆–推理–行动闭环中，并通过案例研究验证了联邦强化学习（FRL）在低空无人机网络（LAWNs）抵御干扰攻击中的有效性。

**💡 创新点**

创新点在于：①首次系统性地把四类 FL（监督、无监督、图、生成、强化）对应到代理式 AI 的四个核心模块；②设计专门的 FRL 框架来实现多工具协同防御，克服了传统集中式 RL 的规模瓶颈；③引入 SLM 作为推理助手，并通过奖励耦合实现算法与推理的一致性。

**🔧 技术方法**

主要技术包括：联邦学习（FedAvg 等聚合）、联邦强化学习（PPO/DDPG/TD3 等策略梯度）、低秩适配器（LoRA）用于联邦生成式微调、基于图神经网络的联邦图学习、链式推理（CoT）优化、以及无人机工具调用接口（角色切换、拓扑重构、频率跳变）。

**📊 数据集**

使用的数据集：①无人机低空监测仿真数据（1×1 km² 区域、100 m 高度、200 m 半径圆形编队）；②无人机与基站的通信链路状态、干扰日志和 RF 监测数据；③在案例研究中并未使用公开真实数据集，而是基于仿真生成的干扰与防御场景数据。

**📈 对比分析**

对比方法：与集中式 RL（CRL）在不同无人机数量下的防御成本和攻击成功率进行对比。实验表明，FRL 在 10 辆无人机时防御成本下降约 69.6%，攻击成功率下降约 56.5%；且 FRL 能够在无人机规模超过 10 辆时继续保持良好性能，而 CRL 随规模扩张快速退化。

**⚠️ 局限性**

限制与挑战：①联邦过程易受模型投毒或推理攻击，需要 Byzantine‑resilient 聚合和隐私增强措施；②不同设备的计算/通信异质性导致同步与聚合开销；③目前仅在仿真环境验证，缺乏真实无线网络部署的实测；④RL 奖励设计对任务适配性要求高，易出现样本效率低的问题。

---

## 257. Battery Lifetime Prediction using Data-driven Modeling Approaches

**arXiv ID:** 2603.00875 | [PDF](https://arxiv.org/pdf/2603.00875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 258. Optimal Solutions for the Moving Target Vehicle Routing Problem via Branch-and-Price with Relaxed Continuity

**arXiv ID:** 2603.00663 | [PDF](https://arxiv.org/pdf/2603.00663v1)

**作者:** Anoop Bhat `[一作]` (Carnegie Mellon University), Howie Choset `[通讯]` (Carnegie Mellon University)

**通讯引用:** 22059 | [OpenAlex ID](https://openalex.org/A5048906141)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的分支定价算法 BPRC，用于解决具有移动目标、时间窗口和容量约束的 MT‑VRP。

**💡 创新点**

创新点在于为移动目标设计的标签算法与支配判定：通过上界/下界估计和目标窗口分段，显著降低定价子问题的计算量。

**🔧 技术方法**

使用分支定价框架、SOCP 求解子路径成本、标签扩展、基于对偶变量的约束削减以及启发式剪枝。

**📊 数据集**

实验使用合成实例，最多 25 个目标、3~5 辆代理、容量 7–22、两时间窗口、线性轨迹。

**📈 对比分析**

与基线 Compact MICP 以及两种消融版本比较，BPRC 在大多数情形下（尤其是容量小、代理数≥3）相较于基线快 10 倍以上，且松弛更紧凑。

**⚠️ 局限性**

局限包括：当目标数≥25 时会触及时间上限，过大或过小的分段数会导致计算量增加；对极端大容量或极宽时间窗口的鲁棒性仍待验证。

---

## 259. Dehallu3D: Hallucination-Mitigated 3D Generation from Single Image via Cyclic View Consistency Refinement

**arXiv ID:** 2603.01601 | [PDF](https://arxiv.org/pdf/2603.01601v1)

**作者:** Xiwen Wang `[一作]` (Sichuan University), Ji-Zhe Zhou `[通讯]` (Sichuan University)

**通讯引用:** 149 | [OpenAlex ID](https://openalex.org/A5033838150)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Dehallu3D 框架，通过插入密集视角并应用视角一致性与自适应平滑损失，解决单图像 3D 重建中的假像（outlier）问题。

**💡 创新点**

核心创新在于 CVCR（Cyclic View Consistency Refinement）插件化模块，实现全 360° 的视角一致性约束并配合自适应平滑以防止过平滑；以及新提出的 Outlier Risk Measure (ORM) 评价指标。

**🔧 技术方法**

结合多视图生成、可微网格渲染、SSIM/CS 视角一致性损失、梯度权重自适应平滑损失，并使用差分渲染与梯度下降。

**📊 数据集**

使用 Google Scanned Objects (GSO) 数据集进行评估。

**📈 对比分析**

与 SF3D、Unique3D、CRM、InstantMesh、TripoSR、Wonder3D 等 SOTA 方法对比，Dehallu3D 在 PSNR、SSIM、LPIPS、Clip‑Sim、Chamfer Distance、F‑Score 上均取得最优或次优结果，ORM 值最低。

**⚠️ 局限性**

主要局限是需要大量密集视角渲染，导致推理时间显著增加；未来计划进一步优化速度。

---

## 260. RaUF: Learning the Spatial Uncertainty Field of Radar

**arXiv ID:** 2603.01026 | [PDF](https://arxiv.org/pdf/2603.01026v1)

**作者:** Shengpeng Wang `[一作]` (Huazhong University of Science and Technology), Wei Wang `[通讯]` (Wuhan University)

**通讯引用:** 49518 | [OpenAlex ID](https://openalex.org/A5100445442)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 RaUF 框架，学习雷达空间检测的异向不确定性并通过双向域注意力融合空间与多普勒特征，从而提升雷达感知的空间精度与可靠性。

**💡 创新点**

创新点包括：①基于物理的异向高斯不确定性建模，捕捉雷达“月牙形”空间误差；②Bidirectional Domain Attention Fusion（BDAF）模块，实现空间特征与多普勒特征的互补与伪散射抑制；③联合训练空间检测与不确定性，显著提升下游定位与跟踪任务的鲁棒性。

**🔧 技术方法**

技术手段：贝叶斯概率模型（BPM）+负对数似然损失；3D CNN 提取空间/多普勒特征；BDAF 双向注意力机制；多帧体素化生成基于 LiDAR 的稀疏真值；多普勒一致性约束与高斯误差传播。

**📊 数据集**

使用了 Coloradar、RaDelft 三个公开雷达数据集以及自行收集的包含单芯片与级联雷达的室内外场景数据集（共 11k+ 帧）。

**📈 对比分析**

与 OS‑CFAR、RPDNet、RadarHD、SDDiff 等传统与深度学习方法对比，在 Coloradar/ RaDelft 上在 CD、F‑score、CPR 指标均显著优于传统 CFAR，接近或超越 SDDiff；在变换估计与车速估计等下游任务中，也实现了 GICP 以上 20% 的改进。

**⚠️ 局限性**

局限性：对多普勒一致性假设依赖较强，易受高速或多路径环境影响；训练时间长且需要高性能 GPU；目前仅在雷达角分辨率不足的稀疏点云上验证，对高分辨率雷达或非线性多普勒模式的适应性仍待进一步探索。

---

## 261. Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control

**arXiv ID:** 2603.01751 | [PDF](https://arxiv.org/pdf/2603.01751v1)

**作者:** Peng Yu `[一作]` (Sun Yat-sen University), Ning Tan `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于多视角视觉的形状可解释自建模框架，用 Bézier 曲线编码连续体机器人的三维形状，实现无解析模型的几何感知混合形状-位置控制。

**💡 创新点**

创新点在于将连续体机器人形状以可解释的 Bézier 参数化表示，并通过神经常微分方程自建形状与末端位姿动力学模型，进而实现形状感知的障碍物规避与自运动。

**🔧 技术方法**

主要技术包括多视角图像分割与骨架提取、Bézier 曲线拟合、NODE 学习、Jacobian 估计、混合控制与基于形状的逃逸策略。

**📊 数据集**

实验使用一台三段绳索驱动连续体机器人，采集约 1000 条来自两台低成本单目相机的图像以及 MicronTracker 的末端位置，构成自建模与控制的数据集。

**📈 对比分析**

与深度视觉逆运动学（DVIK）方法进行对比，本文在形状误差（<1%图像分辨率）和末端位姿误差（<1%机器人长度）方面均优于基线，并成功实现了障碍物规避。

**⚠️ 局限性**

局限性包括对图像分割和骨架提取的依赖、Bézier 近似对高曲率细节的不足、未实现完整三维重建、任务优先级处理有限以及对光照/遮挡敏感。

---

## 262. SphUnc: Hyperspherical Uncertainty Decomposition and Causal Identification via Information Geometry

**arXiv ID:** 2603.01168 | [PDF](https://arxiv.org/pdf/2603.01168v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11949 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了 SphUnc 框架，将球面表示学习与结构因果模型结合，用于多智能体系统的可解释不确定性估计与因果推断。

**💡 创新点**

创新点在于：① 用 von Mises–Fisher 分布的浓度参数作为球面方向的知识不确定度；② 通过信息几何融合实现可解释的先验+观测不确定度分解；③ 在球面潜变量上训练稀疏结构因果模型，实现可解释的干预模拟。

**🔧 技术方法**

技术手段包括：球面嵌入 + vMF 归一化、角度注意力的超图消息传递、双头不确定度（先验+观测）估计、可解释融合网络、稀疏结构学习与时间先序约束、Monte‑Carlo 干预模拟、熵校准损失。

**📊 数据集**

实验数据集：SNARE（离线社交网络）、PHEME（推特讨论树）、AMIGOS（多模态情感交互）、Financial Network、Collaboration Graph。

**📈 对比分析**

与 HyperGCN、SS‑HGNN、SphereNet、CI‑GNN、Causal‑SphHN 等基线对比，SphUnc 在 F1/AUC/Accuracy 上分别提升约 10–15 % 以上，ECE 降至 0.02 左右，Precision@10 最高达 0.78/0.81/0.75；消融实验表明每个模块对性能均有显著贡献。

**⚠️ 局限性**

局限性包括：需要额外的 Monte‑Carlo 干预估计与球面推理导致计算开销上升；依赖稀疏性与时间先序约束，理论可识别性在实际数据中仍有限；缺乏对更大规模、多类型干预的验证。

---

## 263. De-paradox Tree: Breaking Down Simpson's Paradox via A Kernel-Based Partition Algorithm

**arXiv ID:** 2603.02174 | [PDF](https://arxiv.org/pdf/2603.02174v1)

**作者:** Xian Teng `[一作]` (University of Pittsburgh), Yu-Ru Lin `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可解释的两阶段决策树（De‑paradox Tree），专门用于在观测数据中检测并解释 Simpson’s Paradox。

**💡 创新点**

创新点在于：① 使用核均值嵌入的 kernel distance 作为分裂准则，实现对处理组分布的精准平衡；② 在平衡子组内部构建“相反效应”政策树，系统识别嵌套的效应异质性；③ 将因果平衡与效应同质化两大目标结合，形成面向非专家的易解释框架。

**🔧 技术方法**

技术细节包括：核均值嵌入与 kernel distance、基于决策树的递归分割、政策学习（双重稳健估计）和交叉拟合、第二阶段使用动态规划寻找最优政策子树。

**📊 数据集**

实验数据涵盖：合成数据（控制混杂与效应异质性），混合投票数据（人为注入混杂与效应异质性），以及真实数据集（Lalonde/NSW、Python 编程课堂）。

**📈 对比分析**

与多种基线（Propensity Tree、CausalForest-DML、Linear-DRLearner、Uplift Tree 等）比较，实验显示 De‑paradox Tree 在均衡度、误差率、回报（regret）等指标上均优于基线，且树结构更浅、解释更直观。

**⚠️ 局限性**

局限性包括：① 仅适用于假设的因果结构（混杂 + 效应异质性），不处理碰撞器导致的偏差；② 树仅做轴向切分，可能无法完全消除复杂非线性偏差；③ 仅利用可观测协变量，无法解决不可观测混杂；④ 对数据敏感，贪婪搜索可能产生局部最优解。

---

## 264. Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation

**arXiv ID:** 2603.01549 | [PDF](https://arxiv.org/pdf/2603.01549v1)

**作者:** Jisoo Kim `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Pri4R框架，在训练阶段为Vision‑Language‑Action模型引入4D点轨迹监督，使模型隐式学习世界动力学；通过在VLM骨干上添加轻量级点轨迹预测头，训练完成后无需改动推理网络；

**💡 创新点**

创新点在于使用三维点轨迹作为特权监督信号，直接对齐于动作时空度量空间，极大提升VLA对物理交互的理解，同时保持推理时无额外计算；

**🔧 技术方法**

技术方案包括在VLM骨干上插入两层MLP点轨迹预测头、点与动作嵌入融合、使用LoRA微调；训练目标为动作回归+点轨迹L1损失；

**📊 数据集**

使用LIBERO、RoboCasa多任务仿真数据集，以及O‑MY‑F3M机器人真实任务；点轨迹通过模拟网格或外置3D跟踪模型生成；

**📈 对比分析**

与Diffusion Policy、Octo、DiT、OpenVLA、OpenVLA‑OFT、π₀、π₀.₅等基线对比，Pri4R在LIBERO‑Long提升约10%成功率，在RoboCasa提升约40%；在真实任务中显著降低碰撞、提升定位精度；

**⚠️ 局限性**

局限在于仅在微调阶段使用特权监督，推理时仍不依赖几何输入；未在大规模预训练或更广泛任务上验证，未来可探索在预训练中加入点轨迹监督或在推理时利用几何信息提升鲁棒性。

---

## 265. Unbounded length minimal synchronizing words for quantum channels over qutrits

**arXiv ID:** 2603.00861 | [PDF](https://arxiv.org/pdf/2603.00861v1)

**作者:** Bjørn Kjos-Hanssen `[一作]`, Swarnalakshmi Lakshmanan `[通讯]` (University of Hawaii at Manoa)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一类三能级（qutrit）量子通道，使其最小同步词长度可以任意增长，进而证明了在量子自动机中不存在类似于Černý猜想的上界。

**💡 创新点**

创新点在于：①证明了量子通道的同步词长度无上界；②将先前仅能得到长度为3的同步词的构造推广到任意长；③通过轨迹距离与算子范数的结合，为量子通道同步性提供了新的分析工具。

**🔧 技术方法**

使用了量子通道理论（Kraus算子表示）、谱范数与迹距不等式、以及对B旋转算子的小角度近似展开。

**📊 数据集**

未使用任何外部数据集，研究完全基于理论构造和数学证明。

**📈 对比分析**

通过构造特定的量子通道（A、Bₙ）并证明其同步词长度上界无限，可直接与Černý猜想中的 (q-1)² 上界对比，显示在量子场景下该猜想不成立；未涉及实验性能评估。

**⚠️ 局限性**

局限性包括：仅针对三能级系统给出结果；未证明二能级（qubit）是否存在同样性质；并且构造主要基于理论演示，缺乏实验验证。

---

## 266. TopoMaskV3: 3D Mask Head with Dense Offset and Height Predictions for Road Topology Understanding

**arXiv ID:** 2603.01558 | [PDF](https://arxiv.org/pdf/2603.01558v1)

**作者:** Muhammet Esat Kalfaoglu `[一作]` (Middle East Technical University), Alptekin Temizel `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于掩模的道路拓扑预测模型TopoMaskV3，实现了3D中心线的端到端预测。

**💡 创新点**

创新点包括引入稠密偏移场与高度图头来纠正离散化误差，单一掩模头即可得到3D结果，并首次采用地理无重叠拆分和±100 m长距离基准。

**🔧 技术方法**

使用多视图BEV投影、Transformer解码器、稠密偏移与高度预测、Bezier融合以及LiDAR融合等技术。

**📊 数据集**

在Argoverse2 HDMap、OpenLane‑V2、NuScenes等数据集上训练评估，构建了新的地理拆分与长距离测试集。

**📈 对比分析**

在地理无重叠Near拆分上，融合版本TopoMaskV3(F)获得28.5 OLS（最高），相较先前方法提升约1点；在原始重叠拆分上亦接近TopBDA。

**⚠️ 局限性**

局限性包括仍需后处理步骤，未完全端到端；在重叠拆分上可能受记忆化影响；对极端天气或不同光照的鲁棒性尚未评估。

---

## 267. Position: Evaluation of Visual Processing Should Be Human-Centered, Not Metric-Centered

**arXiv ID:** 2603.00643 | [PDF](https://arxiv.org/pdf/2603.00643v1)

**作者:** Jinfan Hu `[一作]` (Shenzhen Institutes of Advanced Technology), Jinjin Gu `[通讯]` (INSAIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文论证在生成式图像恢复等任务中，传统单一指标（PSNR、SSIM、LPIPS等）已与人类感知失衡，主张将评估方法转向人本中心、情境感知和多维度评价，并呼吁重新平衡指标与用户偏好的关系。

**💡 创新点**

创新点在于：①提出人本评估作为评估基准，强调多维度和情境化评价；②指出现有IQM与生成模型、数据规模、语义理解之间的鸿沟；③系统梳理指标与人类感知的失配，强调学习型无参考 IQM 的过拟合与高频偏好；④提出未来 IQM 应向语义感知、可解释性、与大模型融合方向演进。

**🔧 技术方法**

使用技术包括：传统全参考与无参考 IQM（PSNR、SSIM、LPIPS、NIQE、MUSIQ、MANIQA、CLIP-IQA 等）；GAN 与扩散式生成恢复模型；对比与人类主观评价（MOS、配对比较、瑞士式锦标赛、ELO 排名等）；以及对比实验、统计分析和可视化。

**📊 数据集**

使用的数据集：PaQ‑2‑PiQ、SPAQ、KonIQ、AVA、PIPAL、KADID、CSIQ、TID2013、LIVE、SRIQA benchmark（含 100 张图像），以及大规模训练集（SUPIR、HYPIR 等）。

**📈 对比分析**

方法比较：在 SRIQA benchmark 上对 PSNR‑导向、GAN‑导向和扩散‑导向模型分别计算 PSNR、SSIM、LPIPS、MUSIQ、MANIQA、CLIP‑IQA，发现数值指标与人类偏好不匹配；通过场景感知的人类评估（如 SUPIR vs HAT 的手绘/漫画场景），揭示单一指标无法体现细粒度差异，凸显多维度评价的重要性。

**⚠️ 局限性**

局限性：①现有 IQM 与人类感知的偏差、易受高频/过锐化操作影响；②缺乏语义/情境理解，无法区分合理模糊与不自然细节；③数据与模型规模落后，难以评估大模型生成结果；④人类评估难以规模化、可复制，缺乏统一标准。

---

## 268. MatRIS: Toward Reliable and Efficient Pretrained Machine Learning Interaction Potentials

**arXiv ID:** 2603.02002 | [PDF](https://arxiv.org/pdf/2603.02002v1)

**作者:** Yuanchang Zhou `[一作]` (Chinese Academy of Sciences), Weile Jia `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了MatRIS，一种 invariant 机器学习原子势（MLIP），通过注意力机制捕获三体相互作用，实现高精度且低成本的材料模拟。

**💡 创新点**

创新点在于：①提出可分离注意力和维度软最大化的 O(N) 复杂度注意力机制；②将线图与原子图交互用于三体信息融合；③利用可学习包络函数和 gMLP 等模块提升表达能力，同时保持 invariance；这些设计在保持 equivariant 模型精度的同时显著降低计算开销。

**🔧 技术方法**

技术手段包括：基于图神经网络的线图-原子图交互、可分离注意力、维度软最大化、可学习包络函数、gMLP、学习的贝塞尔/傅里叶基底特征、降噪预训练、磁矩预测与图层损失等。

**📊 数据集**

使用的数据集主要有 Matbench-Discovery、MatPES、MDR phonon、Molecular zero‑shot benchmark 以及 MPTrj、OAM 等训练数据。

**📈 对比分析**

与 equivariant（eSEN、eqV2）、MACE、SevenNet 等现有模型在相同 MPTrj 训练集上进行 F1、MAE、RMSD 等指标对比；MatRIS-L 在 Matbench-Discovery 上取得 F1=0.847、RMSD=0.0717，MatRIS-M/MatRIS-S 在参数和训练成本上分别比 eqV2/eSEN 低 13×、6.4×，在其他基准上也接近或超越 SOTA。

**⚠️ 局限性**

局限性：目前未实现长程电静力学处理，模型在极大系统规模下的推理速度仍落后于 MACE‑L；依赖降噪预训练可能限制迁移到小规模数据集，需进一步在更大 QM 数据集上验证。

---

## 269. When Does RL Help Medical VLMs? Disentangling Vision, SFT, and RL Gains

**arXiv ID:** 2603.01301 | [PDF](https://arxiv.org/pdf/2603.01301v1)

**作者:** Ahmadreza Jeddi `[一作]` (University of Toronto), Babak Taati `[通讯]` (University of Toronto)

**通讯引用:** 3306 | [OpenAlex ID](https://openalex.org/A5011257199)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对医疗视觉‑语言模型（VLM）的后训练过程进行系统分析，拆分了视觉感知、监督微调（SFT）和强化学习（RL）对性能的影响，并基于此提出了边界感知的后训练策略；随后在OctoMed基础上使用该策略对PMC‑VQA的多模态问题进行RL后训练，取得了六大医学VQA基准的最佳平均性能。

**💡 创新点**

创新点在于：①用Pass@K与Accuracy@1的对比量化模型的支持边界；②揭示RL主要是分布锐化而非能力扩展；③提出先桥接支持后再锐化的分阶段RL后训练方案；④将该方案在多模态医学VQA上验证，首次实现基线模型与RL后训练相结合的最佳表现。

**🔧 技术方法**

使用的技术包括：Qwen2.5‑VL系列VLM、MedMNIST v2视觉线性探测、Accuracy@1与Pass@K评估、GRPO‑style一致性增强RL、监督微调、平衡采样的多模态问题构造与评估。

**📊 数据集**

数据集主要为MedMNIST v2（多模态医学分类任务）和PMC‑VQA（多模态医学问答），并在六个公开医学VQA基准上进行最终评测。

**📈 对比分析**

通过对比线性探测准确率、单样本Accuracy@1、Pass@K以及跨模态/跨任务迁移表现进行评估；实验显示SFT显著提升视觉感知与支持；RL在已有较高Pass@K时能显著提升Accuracy@1；最终基于OctoMed+RL的模型在六个医学VQA基准上平均表现超过所有公开的Qwen2.5‑VL基线。

**⚠️ 局限性**

局限性包括：RL对支持度低或跨模态迁移大的任务效果有限；分析范围主要集中在MedMNIST与PMC‑VQA，未覆盖更大规模或多中心真实数据；RL后训练对小样本和有限计算资源敏感；模型在不同机构、设备条件下的鲁棒性仍需进一步验证。

---

## 270. BioProAgent: Neuro-Symbolic Grounding for Constrained Scientific Planning

**arXiv ID:** 2603.00876 | [PDF](https://arxiv.org/pdf/2603.00876v1)

**作者:** Yuyang Liu `[一作]` (Peking University), Yonghong Tian `[通讯]` (Peking University)

**通讯引用:** 15806 | [OpenAlex ID](https://openalex.org/A5023918894)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种名为 BioProAgent 的神经符号框架，将 LLM 的概率推理锚定在确定性有限状态机（FSM）上，实现可执行的实验协议。

**💡 创新点**

创新点在于引入 Design‑Verify‑Rectify 工作流和语义符号 grounding，通过 FSM 作为安全互锁，显著降低 hallucination 并实现 95.6% 的物理合规率。

**🔧 技术方法**

采用了 LLM、确定性 FSM、语义符号 grounding、层级验证（科学与物理两层）以及规则引擎等技术。

**📊 数据集**

使用扩展版 BioProBench 数据集（A–D 四个子集）和 22 台实验设备的硬件注册表。

**📈 对比分析**

与 Vanilla LLM、ReAct、Reflexion、AutoGPT、Biomni 等基线对比，BioProAgent 在科学有效性、物理合规性和长周期任务成功率上均提升 30%~40%，且 token 消耗减少 82%。

**⚠️ 局限性**

局限性包括对预先注册硬件的依赖、缺乏真实随机物理噪声模拟，以及需手工维护硬件注册表。

---

## 271. MedGPT-oss: Training a General-Purpose Vision-Language Model for Biomedicine

**arXiv ID:** 2603.00842 | [PDF](https://arxiv.org/pdf/2603.00842v1)

**作者:** Kai Zhang `[一作]` (Lehigh University), Yonghui Wu `[通讯]` (University of Florida)

**通讯引用:** 31145 | [OpenAlex ID](https://openalex.org/A5010253402)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了MedGPT-oss，一个开放权重的20B参数医学多模态语言模型，利用GPT-oss语言骨干与CLIP视觉编码器，并通过三阶段训练实现医学图像与文本的跨模态推理。

**💡 创新点**

创新点在于以最小化架构为基础，结合分阶段训练（预训练、mid-training、指令微调）和长上下文技术，使通用模型在医学领域达到高性能，并提供完整可复现的训练与评估脚本，首次在多项OOD和复杂推理任务上超过更大模型。

**🔧 技术方法**

采用GPT-oss 20B语言模型、CLIP视觉编码器、线性投影、三阶段训练流程、YaRN RoPE长上下文、bfloat16混合精度、DeepSpeed ZeRO-3分布式优化等技术。

**📊 数据集**

使用了大规模医学多模态数据集，包括PMC-OA、Quilt-1M、ROCOv2、BIOMEDICA、MIMIC-CXR、MedTrinity、PubMedVision、MedQA、PubMedQA、MedXQA、MMLU-Med、SLAKE、MedFrameQA、MMMU-Med等，涵盖图像、文本、问答和报告生成。

**📈 对比分析**

与多种公开大模型（如Lingshu-32B、Hulu-Med-32B、OctoMed-7B、MedGemma-27B）在VQA、文本QA、报告生成和ICL等任务上对比，MedGPT-oss在多项OOD VQA、MedXQA、Medbullets和报告生成指标上取得SOTA，20B参数在保持可部署性的同时性能接近或超过更大模型。

**⚠️ 局限性**

仍是研究基础模型，尚未完成临床验证；长文本生成易出现幻觉或遗漏；对不同机构或设备的域漂移缺乏评估；需持续监控、审计与公平性评估；未支持3D影像、RLHF、交互式代理等高级功能。

---

## 272. MSP-ReID: Hairstyle-Robust Cloth-Changing Person Re-Identification

**arXiv ID:** 2603.01640 | [PDF](https://arxiv.org/pdf/2603.01640v1)

**作者:** Xiangyang He `[一作]` (China University of Geosciences), Lin Wan `[通讯]` (China University of Geosciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一框架MSP-ReID，用于克服服装和发型变化下的人体重识别挑战。

**💡 创新点**

创新点在于结合发型定向增强、服装保持随机擦除和基于解析的区域注意机制，显著降低发型偏差并保留结构信息。

**🔧 技术方法**

采用发型生成网络HairFastGAN进行发型增强、Cloth‑Preserved Random Erasing在服装区域的比例擦除，以及人像解析引导的注意力机制。

**📊 数据集**

在PRCC、LTCC、VC‑Clothes和LaST四个主流服装变更重识别基准上进行评估。

**📈 对比分析**

与现有最先进方法对比，MSP‑ReID在Rank‑1和mAP上实现了新的最高分，尤其在服装变更协议下显著提升。

**⚠️ 局限性**

局限性包括对发型生成质量的依赖，以及在大规模数据集上仍有一定差距，需进一步优化域适应。

---

## 273. Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction

**arXiv ID:** 2603.01423 | [PDF](https://arxiv.org/pdf/2603.01423v1)

**作者:** Jiyoon Myung `[一作]` `[通讯]` (Samsung SDS), Jiyoon Myung (Samsung SDS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了大型语言模型在多轮混合主题对话中的可靠性下降，提出了三类可通过pass/fail判定的实用任务；

**💡 创新点**

创新点在于设计了与单轮对照的可复制、可量化的多轮评估任务，重点关注全球约束维护、工具选择与实体跟踪；

**🔧 技术方法**

采用了GPT‑5生成的合成对话，结合定量准确率评估，分析了模型在不同任务上的表现；

**📊 数据集**

使用了三类任务的约600条对话样本（单轮与多轮各约200条），其中对话由GPT‑5自动生成并人工验证；

**📈 对比分析**

对比了多款商业与开源LLM（如GPT‑4o、Gemini‑2.5‑Flash、Qwen‑32B、Mistral‑small‑24B等），发现单轮表现良好，但多轮时显著下降，尤其是小模型在保持全局约束和工具选择上表现最差；

**⚠️ 局限性**

局限性包括使用合成数据可能不完全覆盖真实对话复杂性、仅评估准确率而非生成质量、对模型温度设定为0导致生成可重复性高但缺乏自然多样性。

---

## 274. Theoretical Perspectives on Data Quality and Synergistic Effects in Pre- and Post-Training Reasoning Models

**arXiv ID:** 2603.01293 | [PDF](https://arxiv.org/pdf/2603.01293v1)

**作者:** Adel Javanmard `[一作]`, Vahab Mirrokni `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了Transformer在预训练阶段使用平衡大规模数据进行线性回归权重预测任务，并在后训练阶段通过监督微调(SFT)或强化学习(RL)进行链式推理(CoT)以提升模型能力。

**💡 创新点**

首次系统阐明预训练数据的平衡性与后训练数据规模对SFT和RL的不同影响，并给出相应的理论解释与实践指引。

**🔧 技术方法**

采用Transformer在线性回归上下文权重预测任务的理论分析，并在大规模非线性Transformer架构上进行实验验证；后训练使用简化的SFT监督中间步骤和RL的终点回报监督。

**📊 数据集**

使用人工构造的线性回归数据集（可调节规模与难度）以及高质量与低质量SFT示例集，辅以规模可调的RL反馈数据集。

**📈 对比分析**

与传统的RLHF和SFT基准对比，实验表明小规模高质量SFT数据能获得最佳性能，而大规模且不太难的RL数据亦可显著提升表现，验证了理论结论。

**⚠️ 局限性**

局限性包括RL模型过于简化，未包含完整的采样、优势估计和策略梯度流程；实验仅在线性回归任务中验证，缺乏对更复杂实际场景的泛化检验。

---

## 275. The Lattice Representation Hypothesis of Large Language Models

**arXiv ID:** 2603.01227 | [PDF](https://arxiv.org/pdf/2603.01227v1)

**作者:** Bo Xiong `[一作]` (Stanford University), Bo Xiong `[通讯]` (Stanford University)

**通讯引用:** 4535 | [OpenAlex ID](https://openalex.org/A5102833089)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“格结构表征假设”，将大语言模型中的线性属性方向与Formal Concept Analysis（FCA）相结合，展示了在嵌入空间中可形成概念格，并通过几何运算实现概念的交集与并集。

**💡 创新点**

创新点在于：①将线性方向模型与FCA的概念格理论统一；②证明属性方向与半空间交集可构造完整的概念格；③提供软阈值与投影配置的连续概念包含度量与算子，构建可解释的符号抽象基础。

**🔧 技术方法**

技术方法包括：线性判别分析估计属性方向；软阈值逻辑概率模型；投影特征向量与软包含度量；基于半空间的概念交集/并集算子；以及基于Soft Inclusion和Fuzzy t‑norm 的概念相似度评估。

**📊 数据集**

使用 WordNet 子层次结构（Animal、Plant、Food、Event、Cognition）作为语义域；利用 GPT‑4o 自动生成属性集合并标注对象‑属性矩阵；对对象使用词表平均嵌入，对属性使用 Fisher 方向；实验涉及 LLaMA3.1‑8B、Gemma‑7B、Mistral‑7B 等 LLM。

**📈 对比分析**

对比 Random、Mean（中心向量）和本文线性方法。实验表明线性方法在 5 个域上均能达到 70%–80% 的 F1，显著优于基线；在子概念与并集推断任务中，MRR 最高可达 0.53，证明在几何空间直接推断层级关系与符号算子效果良好。

**⚠️ 局限性**

局限性包括：①对抽象概念的可分离性和阈值估计依赖于模型和属性质量，表现不如物理域；②软阈值和投影包含度量仍是经验性的，缺乏理论最优性证明；③属性方向需要先验生成（由 GPT‑4o），可能带来标签偏差；④当前实验仅验证了 WordNet 词汇层次，未扩展到更复杂或跨模态知识。

---

## 276. An Embedded Mesh Approach for Isogeometric Boundary Layers in Contact Mechanics

**arXiv ID:** 2603.01857 | [PDF](https://arxiv.org/pdf/2603.01857v1)

**作者:** Eugenia Gabriela Loera Villeda `[一作]` (University of Bundeswehr Munich), Alexander Popp `[通讯]` (University of Bundeswehr Munich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种新的离散化工作流程：先在接触边界生成 NURBS 边界层网格，再用结构化笛卡尔网格描述体积域，利用嵌入式重叠网格与约束算子（mortar 法）实现两种网格的耦合，从而在不一致网格下高效求解接触问题。

**💡 创新点**

创新点在于（1）将接触边界的高阶光滑 NURBS 表示与体积的简单笛卡尔网格解耦，既保留了接触面连续性又简化了体积网格生成；（2）采用嵌入式 mortar 约束在重叠网格上实现稳定耦合；（3）对 NURBS 边界层的偏移方法进行了系统比较，并给出了多种可行实现。

**🔧 技术方法**

使用的主要技术包括：NURBS 几何偏移（三种实现方式）、有限元/等几何分析（IGA）离散、嵌入式网格耦合（mortar）、Newton‑Raphson 求解、活跃集法处理接触约束、以及针对切割单元的数值积分。

**📊 数据集**

所用数据集为基于几何模型的合成案例：二维块体与刚性面、半圆柱与刚性板、两个扭体碰撞等，均从 CAD/三维几何导出 NURBS 表示并构造偏移边界层；并无公开实验数据集。

**📈 对比分析**

与经典全同构网格、传统接触离散化以及解析解进行对比；结果表明（1）能保持 1 阶或 2 阶收敛率；（2）在接触压力、应力分布与解析 Hertz 结果高度吻合；（3）多体大变形碰撞模拟在计算效率和稳定性方面优于纯同构网格，且能通过局部边界层细化显著提高接触应力精度。

**⚠️ 局限性**

主要局限包括：偏移操作对曲率过小的几何易产生自相交，需进一步处理；当重叠网格材料属性不一致时可能出现网格锁定；小切割单元导致矩阵病态，需 ghost 稳定或 Nitsche 方案；以及当前实现仅针对无摩擦接触，扩展至摩擦或多物理场仍待研究。

---

## 277. From Variance to Invariance: Qualitative Content Analysis for Narrative Graph Annotation

**arXiv ID:** 2603.01930 | [PDF](https://arxiv.org/pdf/2603.01930v1)

**作者:** Junbo Huang `[一作]` (University of Hamburg), Ricardo Usbeck `[通讯]` (Leuphana University Lüneburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了以通胀叙事为主题的图形数据集，并通过定性内容分析（QCA）方法对新闻文本进行叙事标注。

**💡 创新点**

创新点在于将QCA融入NLP叙事标注流程，提出多级叙事表示与多重距离度量的6×3因子实验框架，揭示了叙事表示与可靠性之间的权衡。

**🔧 技术方法**

采用Krippendorff’s alpha、节点/边重叠度量、Jaccard距离、图编辑距离以及图/节点的完全匹配等多种技术进行标注一致性评估。

**📊 数据集**

使用的是美国道琼斯新闻数据库（DJN）的新闻语料，其中标注集共包含488篇文章（104篇用于图形标注）。

**📈 对比分析**

实验比较显示：严格度越高，α值越低；局部结构（如Adjacent Story）在不同距离度量下表现最稳健，α值在0.44–0.70之间，说明局部叙事表示兼顾可靠性与上下文完整性。

**⚠️ 局限性**

局限性包括数据规模有限、标注者人数少导致统计功效不足、标注者多样性不足、距离度量忽略词义相似性以及对争议细粒度定位不足。

---

## 278. PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation

**arXiv ID:** 2603.00976 | [PDF](https://arxiv.org/pdf/2603.00976v1)

**作者:** Jiangshan Wang `[一作]` (Tsinghua University), Xiangyu Yue `[通讯]` (MMLab, CUHK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 PreciseCache，一种训练‑free 的视频扩散模型推理加速框架，包含基于低频差异的步级缓存（LFCache）和基于块级差异的块级缓存（BlockCache）。

**💡 创新点**

创新点在于：① 设计低频差异（Low‑Frequency Difference, LFD）作为精准衡量每一步冗余的指标；② 通过对下采样 latent 进行快速推理计算 LFD，决定是否跳过当前步骤；③ 在非跳过步骤内部识别“非关键块”，仅重用关键块的计算结果，实现块级加速。

**🔧 技术方法**

技术手段包括：FFT 分离低频/高频特征；下采样 latent 进行 trial inference；LFD 作为缓存判别阈值；BlockCache 通过块输入输出差异筛选关键块；FlashAttention、动态序列并行等硬件加速技术。

**📊 数据集**

使用 VBench 提供的 Prompt 集，生成 480P、720P、1080P 等分辨率视频，对比基线模型的完整推理。数据集仅限这些生成视频，未使用真实视频数据集。

**📈 对比分析**

与 PAB、TeaCache、FasterCache 等现有缓存方法比较：在 Open‑Sora 1.2、HunyuanVideo、CogVideoX、Wan2.1‑14B 上分别实现 1.7–2.6× 的加速（MACs 下降 30–60%），推理时延从 30–90 秒降至 18–35 秒，视觉质量指标（VBench、LPIPS、SSIM、PSNR）基本保持或略有提升，说明质量基本不受影响。

**⚠️ 局限性**

局限性包括：① 缓存阈值 δ 需根据模型/提示动态设定（采用 α×max LFD 的经验法则）；② 下采样比例选择需权衡速度与精度，过度下采样会导致缓存判别失效；③ 目前仅在 DiT‑based 视频扩散模型上验证，其他架构需进一步测试；④ 需要额外存储缓存特征，内存占用略增；⑤ 对极短推理步骤或低噪声阶段的精度控制尚未完全优化。

---

## 279. A 3D mesh convolution-based autoencoder for geometry compression

**arXiv ID:** 2603.02125 | [PDF](https://arxiv.org/pdf/2603.02125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 280. Semantic Novelty Trajectories in 80,000 Books: A Cross-Corpus Embedding Analysis

**arXiv ID:** 2603.01791 | [PDF](https://arxiv.org/pdf/2603.01791v1)

**作者:** Fred Zimmerman `[一作]` `[通讯]` (Nimble Books LLC), Fred Zimmerman (Nimble Books LLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对超过八万本书进行语义新颖性轨迹分析，比较19 世纪前后与现代英语文学的差异。

**💡 创新点**

首次将压缩进展理论与句子嵌入相结合，在大规模语料上实现语义新颖性度量，并发现新颖性与读者评价无关。

**🔧 技术方法**

使用句子 Transformer（SBERT）进行段落嵌入，计算运行质心新颖性，利用 PAA、SAX 与 K‑means 对轨迹进行聚类与特征化。

**📊 数据集**

PG19（28,730 本 1900 年前书）与 Books3（52,796 本 1990–2010 年现代书）两大语料集。

**📈 对比分析**

通过比较平均新颖性、轨迹曲率、曲线类型分布以及聚类占比，发现现代书新颖性提升约 10%，曲率升高 67%，收敛曲线比例下降 2.3 倍，且与读者评分无显著相关性。

**⚠️ 局限性**

局限包括对不同年代语料的聚类独立、Books3 版权来源不透明、嵌入模型对古文本的偏差、段落划分不够细致以及时间维度只对比两大时期而非细分年代。

---

## 281. Mean-Flow based One-Step Vision-Language-Action

**arXiv ID:** 2603.01469 | [PDF](https://arxiv.org/pdf/2603.01469v1)

**作者:** Yang Chen `[一作]`, Bin Zhao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Mean‑Flow一站式Vision‑Language‑Action框架，利用均值去噪向量场实现单步连续动作生成，并在机器人抓取、堆叠、排序等真实任务中验证效果。

**💡 创新点**

通过学习平均去噪向量场替代传统瞬时向量场，消除多步积分误差，实现单步推断并显著加速；同时引入流比率、Adaptive Loss等训练技巧提升稳定性。

**🔧 技术方法**

使用Mean‑Flow流匹配框架、预训练SmolVLM‑2视觉‑语言模型、Transformer动作专家、JVP训练损失以及单步/多步采样策略。

**📊 数据集**

基于SO‑101机械臂手动演示共300条数据，分别包含100条pick‑place、100条stacking和100条sorting任务示例。

**📈 对比分析**

在与SmolVLA（多步流匹配）和Diffusion Policy（迭代扩散）对比实验中，单步VLA在三项任务上的成功率与多步方法相近（例如pick‑place 88% vs 86.5%），但生成速度分别为SmolVLA的8.7倍、Diffusion Policy的83.9倍。

**⚠️ 局限性**

对高精度堆叠等需要细粒度校正的任务仍略逊于多步方法；单步全局映射对长序列学习难度大，易受高方差样本影响；缺乏对更复杂、规模更大场景的进一步验证。

---

## 282. CA-AFP: Cluster-Aware Adaptive Federated Pruning

**arXiv ID:** 2603.01739 | [PDF](https://arxiv.org/pdf/2603.01739v1)

**作者:** Om Govind Jha `[一作]` (Indian Institute of Science Education and Research Bhopal), Haroon R. Lone `[通讯]` (Indian Institute of Science Education and Research Bhopal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了CA-AFP框架，融合客户端聚类与自适应模型剪枝，解决联邦学习中的统计异质性与系统异质性问题。

**💡 创新点**

创新点包括：①集群感知重要性评分，将权重幅值、集群内一致性与梯度一致性三者结合；②迭代剪枝与自愈机制，使模型在剪枝过程中可动态恢复重要权重。

**🔧 技术方法**

采用聚类式联邦学习、权重剪枝、逐步稀疏化（Prune‑Heal）以及局部微调技术；使用TensorFlow实现，训练采用Adam优化器。

**📊 数据集**

在两个人体动作识别基准上验证：WISDM（36用户）和UCI‑HAR（30用户）。

**📈 对比分析**

与四类基线（聚类密集模型FedCHAR、ClusterFL；剪枝模型EfficientFL、FedSNIP）对比，CA-AFP在保持约70%稀疏度的前提下，平均精度与公平性均优于剪枝基线，且通信成本低于密集聚类方法；在不同非IID程度下仍保持较高准确率。

**⚠️ 局限性**

局限性：聚类策略固定，未探索自适应或多级聚类；重要性评分权重需手工设定，缺乏自动调优机制；对极端设备故障或标签噪声的鲁棒性仍待进一步提升。

---

## 283. Randomized Kiring Believer for Parallel Bayesian Optimization with Regret Bounds

**arXiv ID:** 2603.01470 | [PDF](https://arxiv.org/pdf/2603.01470v1)

**作者:** Shuhei Sugiura `[一作]` (Nagoya University), Shion Takeno `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了随机克里金信徒（RKB）方法，用于并行贝叶斯优化，并给出了其贝叶斯累计和简单回报的理论上界，同时在合成、基准和真实仿真数据上进行实验验证。

**💡 创新点**

创新点在于将传统的 KB 预估思路改为随机化：用单个后验采样值替代点估计，既保留了 KB 的低计算复杂度、异步支持和多样性提升，又实现了可证明的 BCR 与 BSR 上界，并在理论上消除了并行数量 Q 的依赖。

**🔧 技术方法**

使用技术包括高斯过程回归、贝叶斯优化、UCB/EI/PIMS 等采集函数、后验采样、最大信息增益的理论分析以及与现有并行 BO 方法的对比实验。

**📊 数据集**

实验数据集包含：基于 Gaussian 核的 4 维合成函数（10^4 网格点）、四个经典基准函数（Ackley、Hartmann6d、Shekel、Styblinski‑Tang）以及来自 Olympus 框架的 9 个真实仿真器。

**📈 对比分析**

与 KB、LP、BUCB、PTS、随机搜索、无信息采样等方法在同步 8 工作器设置下比较，RKB 在合成与基准任务上表现与 KB/LP 相当，且在 PTS、BUCB 等具有理论保证的并行方法上显著优越；在真实仿真器上也保持稳定且优于多数基线。

**⚠️ 局限性**

局限性包括：需要在初始阶段使用不确定性采样以避免额外的 O(√Q) 代价；理论分析仅覆盖贝叶斯设置，未考虑频率学说；仅使用单个后验采样，未探讨多样本或多目标、多保真度等扩展。

---

## 284. GRAD-Former: Gated Robust Attention-based Differential Transformer for Change Detection

**arXiv ID:** 2603.01161 | [PDF](https://arxiv.org/pdf/2603.01161v1)

**作者:** Durgesh Ameta `[一作]` (Indian Knowledge System and Mental Health Applications Center), Amit Shukla `[通讯]` (Center of Artificial Intelligence and Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 GRAD-Former 的新型遥感多时相图像变化检测框架，能够在不使用预训练特征提取器的情况下，准确识别高分辨率卫星图像中的变化区域。

**💡 创新点**

创新点在于引入 Adaptive Feature Relevance And Refinement (AFRAR) 模块，结合 Selective Embedding Amplification (SEA) 与 Global‑Local Feature Refinement (GLFR) 两个子模块，并采用差分注意力机制（Differential Attention）和差分融合（Difference Amalgamation, DA）来高效过滤噪声、捕获全局与局部上下文，从而在保持参数量低的前提下显著提升性能。

**🔧 技术方法**

使用了轻量级 Transformer 结构、门控机制、差分注意力、多尺度融合以及传统的卷积上采样与残差块等技术；训练时采用交叉熵损失并配合数据增强。

**📊 数据集**

在 LEVIR-CD、DSIFN-CD 与 CDD 三个公开基准数据集上进行评估，覆盖建筑物变化、土地利用变化和季节性/灾害性变化等多种场景。

**📈 对比分析**

与现有 CNN、SSM 以及 Transformer 变体进行对比，GRAD-Former 在 F1、IoU 和 OA 上均实现了最佳或第二最佳成绩，例如在 LEVIR-CD 上达到 91.52% F1、84.36% IoU；在 DSIFN-CD 上 93.14% F1、87.16% IoU；在 CDD 上 97.57% F1、95.26% IoU；参数量仅为 10.90M，显著低于多数竞争方法。

**⚠️ 局限性**

局限性包括：尚未针对实时边缘设备部署进行优化；对极端光照或云遮挡的鲁棒性仍有提升空间；在小样本或稀疏变化场景下的泛化性能需要进一步验证。

---

## 285. ResGene-T: A Tensor-Based Residual Network Approach for Genomic Prediction

**arXiv ID:** 2603.00744 | [PDF](https://arxiv.org/pdf/2603.00744v1)

**作者:** Kuldeep Pathak `[一作]` (Indian Institute of Technology Indore), Eric de Sturler `[通讯]` (Virginia Tech)

**通讯引用:** 3295 | [OpenAlex ID](https://openalex.org/A5002320696)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种基于ResNet的深度学习框架ResGene，用于从基因型序列生成二维图像或三维张量，并通过CNN预测作物表型；

**💡 创新点**

创新点在于将基因型数据先转化为二维图像再进一步升维为三维张量（ResGene-T），使得CNN能够在更早的层级捕获SNP之间的生物学交互，从而显著提升预测精度；

**🔧 技术方法**

主要技术包括基因型编码、图像/张量转换、ResNet-18残差网络、超参数调优（批量大小、学习率、dropout、通道数）以及MSE损失和Pearson相关系数评估；

**📊 数据集**

使用了三种作物（大豆、稻米、玉米）的公开数据集，分别包含66k、57k、56k个SNP及10个表型指标；

**📈 对比分析**

与七种主流方法（统计模型rrBLUP、BayesB；机器学习SVR、XGBoost；深度学习DLGWAS、DNNGP、GPFormer）在10折交叉验证下进行比较，ResGene-T平均Pearson相关系数达0.4281，比其他方法提升14.51%至41.51%；

**⚠️ 局限性**

局限性包括缺乏多表型预测的探讨、对大规模数据的计算可扩展性未充分验证、以及对理论分析与潜在近似计算策略的进一步研究需求。

---

## 286. FreeGNN: Continual Source-Free Graph Neural Network Adaptation for Renewable Energy Forecasting

**arXiv ID:** 2603.01657 | [PDF](https://arxiv.org/pdf/2603.01657v1)

**作者:** Abderaouf Bahi `[一作]` (Chadli Bendjedid University), Mohamed Amine Ferrag `[通讯]` (United Arab Emirates University)

**通讯引用:** 10463 | [OpenAlex ID](https://openalex.org/A5026903935)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出FreeGNN框架，实现在未知可再生能源站点的持续源无关图神经网络自适应预测

**💡 创新点**

将时空GNN、教师-学生一致性、图正则化、记忆回放与漂移感知加权融合，解决源不可用、目标无标签且分布漂移的难题

**🔧 技术方法**

基于spatio‑temporal GNN、EMA教师、增广一致性损失、图卷积正则、记忆缓冲与漂移估计的自监督在线学习

**📊 数据集**

GEFCom2012（风电）、Solar PV（光伏）、Wind SCADA（风机）三大真实多站点时序数据集

**📈 对比分析**

与12种基线（传统GNN、源无关DA、持续学习、时空图模型）对比，FreeGNN在MAE/RMSE/MAPE/sMAPE上均优于大多数基线，且在Wind SCADA上表现最优，Solar PV略逊于STGCN

**⚠️ 局限性**

对图结构依赖较强；记忆回放和教师-学生引入额外计算与存储；单节点场景提升有限；漂移估计对细微漂移不敏感

---

## 287. Tri-path DINO: Feature Complementary Learning for Remote Sensing Multi-Class Change Detection

**arXiv ID:** 2603.01498 | [PDF](https://arxiv.org/pdf/2603.01498v1)

**作者:** Kai Zheng `[一作]` (Zhejiang University), Wei Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 34160 | [OpenAlex ID](https://openalex.org/A5100622062)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出三路DINO架构，结合预训练的DINOv3、CNN与Transformer，实现高效的多类遥感变化检测。

**💡 创新点**

创新点在于三路互补特征学习与多尺度混合注意力解码器的组合，显著提升细粒度变化识别能力。

**🔧 技术方法**

使用自监督视觉基础模型DINOv3、LoRA/Adapter轻量调优、Transformer上下文路径、MLHA注意力机制以及Focal+Dice+Lovász混合损失。

**📊 数据集**

在Gaza‑change（灾后基础设施损害）和SECOND（城市语义变化）两个高分辨率遥感数据集上进行训练与评估。

**📈 对比分析**

与BIT、SNUNet、ChangeFormer等方法对比，Tri‑path DINO在OA、mIoU、Sek、F_scd四项指标均超过前沿方法，尤其mIoU提升约0.7%和Sek提升13点。

**⚠️ 局限性**

局限在仅处理两时遥感图像，未考虑多时序长周期变化；模型参数量仍较大，实时部署性能有待提升。

---

## 288. TARSE: Test-Time Adaptation via Retrieval of Skills and Experience for Reasoning Agents

**arXiv ID:** 2603.01241 | [PDF](https://arxiv.org/pdf/2603.01241v1)

**作者:** Junda Wang `[一作]` (University of Massachusetts Amherst), Hong Yu `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 35381 | [OpenAlex ID](https://openalex.org/A5100446261)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通过检索技能和经验进行测试时适应的框架TARSE，以提高临床决策支持的可靠性。

**💡 创新点**

创新点在于明确区分和检索临床技能与经验，并在测试时对模型进行轻量级适应，以减少推理过程中的不一致性。

**🔧 技术方法**

使用了检索增强生成（RAG）技术和轻量级测试时训练（TTT）方法。

**📊 数据集**

使用了医疗问答基准数据集，包括MedQA、MedMCQA和MMLU等。

**📈 对比分析**

与强大的医疗RAG基线和仅使用提示的方法进行比较，TARSE在多步和约束重的问题上表现出一致的性能提升，尤其是在需要明确检查和分支决策的情况下。

**⚠️ 局限性**

局限性在于错误或过时的技能、偏见的经验轨迹或错误的检索可能导致错误答案，且这些系统应作为决策支持工具，而非自主诊断。

---

## 289. Building a Strong Instruction Language Model for a Less-Resourced Language

**arXiv ID:** 2603.01691 | [PDF](https://arxiv.org/pdf/2603.01691v1)

**作者:** Domen Vreš `[一作]` (University of Ljubljana), Iztok Lebar Bajec `[通讯]` (University of Ljubljana)

**通讯引用:** 765 | [OpenAlex ID](https://openalex.org/A5010861913)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对斯洛文尼亚语等低资源语言，基于Gemma 3 12B模型进行三阶段持续预训练与两阶段监督微调，构建了开源大语言模型GaMS3‑12B；

**💡 创新点**

提出了分阶段持续预训练（并行对齐、基础预训练、长序列预训练）与多模态OCR流水线、以及专门为斯洛文尼亚语构建的GaMS‑Instruct与GaMS‑Nemotron‑Chat大规模指令数据集；

**🔧 技术方法**

使用NeMo框架进行CPT，DeepSpeed ZeRO‑2进行SFT，采用Tensor/Sequence并行、激活重计算等显存优化；

**📊 数据集**

数据来源包括140B混合语言预训练token（斯洛文尼亚语、英语、波斯尼亚语、塞尔维亚语、克罗地亚语），200k+英斯洛文尼亚语SFT样本，OCR提取的国立图书馆、大学论文及数学期刊PDF，及多语言Web、法律、医学等语料；

**📈 对比分析**

通过斯洛文尼亚‑LLM‑Eval、英语→斯洛文尼亚翻译基准和Slovene‑LLM‑Arena进行评估，GaMS3‑12B在所有评测中均优于Gemma 3‑12B，击败同尺寸开源模型，且在Slovene‑LLM‑Arena上与GPT‑4o、Gemini‑2.0‑Flash的性能相近（胜率>60%），仅被Gemma 3‑27B和更大商业模型超越；

**⚠️ 局限性**

主要局限是对机器翻译数据的高度依赖导致斯洛文尼亚语表述不够自然；长文本推理和逻辑推断能力不足，未来需减少机器翻译、提升长上下文与推理训练集。

---

## 290. The Semantic Arrow of Time, Part I: From Eddington to Ethernet

**arXiv ID:** 2603.01440 | [PDF](https://arxiv.org/pdf/2603.01440v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (Dædælus), Paul Borrill (Dædælus)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

阐述计算系统中隐藏的语义时间箭头，并指出它是设计选择而非自然法则。

**💡 创新点**

提出“语义时间箭头”概念，揭示分布式系统中的时序假设为范畴错误，并提供理论替代方案。

**🔧 技术方法**

使用哲学与物理学论证，包括时间对称性、热力学箭头、量子无因果顺序等。

**📊 数据集**

无实验数据集，主要基于文献综述与理论推导。

**📈 对比分析**

通过与经典分布式不可能定理（FLP、CAP等）的对比，说明去除该假设可解除限制，理论上性能提升。

**⚠️ 局限性**

局限在于尚未给出完整可实现的协议实现与实测评估，主要停留在概念层面。

---

## 291. UD-SfPNet: An Underwater Descattering Shape-from-Polarization Network for 3D Normal Reconstruction

**arXiv ID:** 2603.00908 | [PDF](https://arxiv.org/pdf/2603.00908v1)

**作者:** Puyun Wang `[一作]` (Fuzhou University), Yating Chen `[通讯]` (Tsinghua University)

**通讯引用:** 64443 | [OpenAlex ID](https://openalex.org/A5102770958)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为UD‑SfPNet的端到端网络，将水下散射消除与基于偏振的三维表面法向估计联合起来，实现同时去散射与3D重建。

**💡 创新点**

创新点包括：①将散射去除与SfP重建耦合为统一框架，避免级联误差；②引入色彩嵌入模块，将RGB编码与法向几何一致性对齐；③加入细节增强卷积模块以保留高频几何细节；④通过物理模型驱动的联合损失实现全局优化。

**🔧 技术方法**

采用深度学习技术：U‑Net、Transformer/多头注意力、DEConv细节增强卷积、PCE色彩嵌入、Polarization Parameter Network以及多种损失（L1、SSIM、LPIPS、TV、直方图、法向角误差）。

**📊 数据集**

使用公开的MuS‑Polar3D数据集，包含726个散射样本，按8:1:1划分为训练/验证/测试。

**📈 对比分析**

与DeepSfP、SfP‑wild、TransSfP、AttentionU2‑Net、DSINE等五个基线比较，采用平均角误差（MAE）评估。UD‑SfPNet在MuS‑Polar3D测试集上取得15.12°的MAE，低于所有基线；去散射方面，PSNR、SSIM提升显著，LPIPS降低。错误分布更均匀，细节保留更好。

**⚠️ 局限性**

局限性包括：单视角下深度不连续导致积分失真；对极端散射或极低光照仍有提升空间；需要大量标注数据；实时性能尚待进一步优化。

---

## 292. The Expressive Limits of Diagonal SSMs for State-Tracking

**arXiv ID:** 2603.01959 | [PDF](https://arxiv.org/pdf/2603.01959v1)

**作者:** Mehran Shakerinava `[一作]` (McGill University), Sarath Chandar `[通讯]` (Mila)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了输入依赖的复数对角状态空间模型（DCD SSM）的表达能力，特别是对群状态跟踪任务的可表达范围，并通过理论证明单层只能表达阿贝尔群，多层可表达可解群的子正规系列；同时在多种群任务上进行了实验验证。

**💡 创新点**

首次给出对角SSM的精确表达界限，将深度与群的可解性直接对应，并揭示了表达与可学习之间的差距。

**🔧 技术方法**

理论分析（群论与状态空间模型），使用输入依赖的复数对角SSM，深度分层结构；实验中使用单/多层SSM、Mamba、Negative Mamba、AUSSM、RNN等。

**📊 数据集**

使用各种可解群和非可解群的状态跟踪任务：C₂、C₆、C₂₄、C₆₀、C₂×C₄、C₃×C₆、S₃、A₄、A₅。

**📈 对比分析**

通过比较不同模型在最长可推断序列长度（≥100）上达到90%+准确率的能力；结果显示RNN能较好，单层Mamba与Negative Mamba表现差，二层Negative Mamba和AUSSM在阿贝尔群上可行，但在非阿贝尔群（S₃、A₄）多层模型仍无法学习。

**⚠️ 局限性**

模型存在可学习瓶颈：虽然理论上可表达，但梯度优化难以找到正确解，尤其对非阿贝尔可解群；对角结构限制了表达力；实验仅在有限精度下进行，未探究更大规模或其他初始化策略。

---

## 293. A natural language framework for non-conforming hybrid polytopal methods in Gridap.jl

**arXiv ID:** 2603.00880 | [PDF](https://arxiv.org/pdf/2603.00880v1)

**作者:** Jordi Manyer `[一作]` (Monash University), Santiago Badia `[通讯]` (Monash University)

**通讯引用:** 4542 | [OpenAlex ID](https://openalex.org/A5071818847)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个综合框架，用于在Gridap有限元库中实现非共形混合多面体有限元方法，涵盖了多种数学模型的实现。

**💡 创新点**

创新点在于引入了多面体网格表示的新抽象、破碎多项式空间、基于补丁的局部组装、局部算子构造和自动静态凝聚，简化了复杂混合方法的实现。

**🔧 技术方法**

使用了Julia语言的即时编译（JIT）和Gridap的惰性评估策略，构建了高效的计算框架。

**📊 数据集**

使用了多面体网格的数值示例，包括泊松问题、线性弹性、不可压缩斯托克斯流和最优控制问题。

**📈 对比分析**

通过与现有有限元库的比较，展示了该框架在实现复杂混合方法时的简洁性和高效性，通常只需30-50行代码，同时保持了计算效率。

**⚠️ 局限性**

限制在于当前框架主要支持非共形混合多面体方法，未来工作将扩展以支持更多变体的混合方法。

---

## 294. Autoregressive Synthesis of Sparse and Semi-Structured Mixed-Type Data

**arXiv ID:** 2603.01444 | [PDF](https://arxiv.org/pdf/2603.01444v1)

**作者:** Thomas Rückstieß `[一作]`, Robin Vujanic `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种能够直接从JSON记录生成合成数据的自回归Transformer模型（Origami），无需平坦化或填补缺失。

**💡 创新点**

创新点在于Key-Value位置编码、键顺序打乱、双头离散/连续输出以及语法/模式约束，实现了对稀疏、多类型、层级JSON结构的原生合成。

**🔧 技术方法**

使用了自回归Transformer、混合高斯连续头、键值位置编码、语法推送自动机、键顺序随机化和数据增强技术。

**📊 数据集**

在成人、糖尿病、汽车注册、Yelp业务、DDXPlus医学诊断等5个数据集上评估，其中后两个包含高稀疏度的JSON结构。

**📈 对比分析**

与GAN、VAE、扩散、TabularARGN等6种基线在保真度、实用性、检测和隐私四个指标上对比，Origami在稀疏结构数据上保真度最高、检测最难区分、实用性和隐私均优于或等同基线。

**⚠️ 局限性**

对多表关联或更大上下文窗口的支持有限，依赖自回归训练时间长，对极高维度或极大记录长度仍有规模瓶颈。

---

## 295. When Numbers Tell Half the Story: Human-Metric Alignment in Topic Model Evaluation

**arXiv ID:** 2603.01945 | [PDF](https://arxiv.org/pdf/2603.01945v1)

**作者:** Thibault Prouteau `[一作]`, Christophe Malaterre `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的基于人工评估的主题模型质量度量——主题词混合（Topic Word Mixing）任务，并与传统的词入侵（Word Intrusion）任务一起，对多种主题模型在哲学科学专门语料库上的表现进行评估；

**💡 创新点**

创新点在于引入了评估主题间区分度的人工任务（主题词混合），补充了仅评估主题内部连贯性的词入侵任务，并系统比较了自动化一致性与多样性指标与人工评估结果的对应关系；

**🔧 技术方法**

使用的技术包括传统概率主题模型（LDA、NMF、CFMF）、基于嵌入的主题模型（BERTopic、CFMF-emb、Top2Vec），以及自动化评估指标（C_V一致性、主题多样性）和人工评估平台LabelStudio；

**📊 数据集**

使用的数据集为1931–2017年间8本哲学科学期刊的全文（约16,917篇，约6500万词），经过机器翻译、词性标注、词形归一化等预处理；

**📈 对比分析**

结果显示，自动化一致性指标与人工评估不完全一致；如高一致性模型在词入侵任务中表现最差，而高词入侵准确率模型一致性相对较低；主题词混合任务与自动多样性指标较好对应，表明人工评估能部分验证自动多样性；

**⚠️ 局限性**

局限性包括人工评估样本量有限、任务难度差异导致结果偏倚、仅评估一致性与多样性两方面，缺乏对覆盖度和主题代表性的评估，且依赖领域专家的人工标注，难以推广到更大规模语料。

---

## 296. When Does Margin Clamping Affect Training Variance? Dataset-Dependent Effects in Contrastive Forward-Forward Learning

**arXiv ID:** 2603.00951 | [PDF](https://arxiv.org/pdf/2603.00951v1)

**作者:** Joshua Steier `[一作]` `[通讯]` (Independent Researcher), Joshua Steier (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 Contrastive Forward-Forward (CFF) 学习中的正样本间距（margin）实现细节进行可重复性评估，并通过梯度中性化的差分参考来剔除 margin 本身的影响。

**💡 创新点**

证明了在 mean‑over‑positives 归约下，后对对数概率的减法形式梯度中性，并将这种对比用于揭示 saturating similarity clamping 所导致的梯度截断与随机种子变异的关系。

**🔧 技术方法**

使用 Vision Transformer (ViT) 的层级对比损失、梯度截断分析、CAR（Clamp Activation Rate）与梯度范数等诊断技术，并结合梯度中性化证明。

**📊 数据集**

主要在 CIFAR‑10 上进行实验，随后在 CIFAR‑100、SVHN 和 Fashion‑MNIST 上进行跨数据集验证，另外通过 SVHN 的难度扫描验证任务难度对方差影响。

**📈 对比分析**

通过 2×2 因子实验、方差比（VR）和 F‑检验比较两种 margin 方式的方差差异；在 CIFAR‑10 上，clamp 方式方差比约 5.9 倍，差异显著；低 margin 方案与加法减法在平均准确率上无差异；在其他数据集上方差比反向或无显著差异。

**⚠️ 局限性**

局限性包括仅使用单一 ViT 架构与训练配置，样本量有限，未测量每种随机种子下 CAR 与梯度变化，温度‑margin 交互未解耦，难度扫描同时改变多种因素，且只针对 mean‑over‑positives 归约验证梯度中性化，未覆盖其他归约方式。

---

## 297. Sustainable Code Generation Using Large Language Models: A Systematic Literature Review

**arXiv ID:** 2603.00989 | [PDF](https://arxiv.org/pdf/2603.00989v1)

**作者:** Sabiya Banu Masthan Ali `[一作]` (Algoma University), Gautam Srivastava `[通讯]` (Brandon University)

**通讯引用:** 27960 | [OpenAlex ID](https://openalex.org/A5041541232)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM生成代码的可持续性进行系统文献综述，梳理评估指标、实验方法和研究现状。

**💡 创新点**

首次将可持续性视角纳入LLM代码生成研究，系统识别并归纳能耗、碳足迹等指标及其测量工具，并提出缺口与未来研究方向。

**🔧 技术方法**

采用系统文献综述（SLR）方法，包括关键词检索、筛选、数据提取表格，分析模型、任务、评估方法等多维度信息。

**📊 数据集**

不使用单一数据集，而是综述了多篇论文所用的基准与数据集，如LeetCode、CodeXGLUE、HumanEval、Codeforces、MATLAB等。

**📈 对比分析**

通过对19篇原始研究的定量统计与定性比较，发现大多数研究聚焦大模型、能耗评估主要依赖软件层工具，缺乏统一可持续性基准；对比结果表明能耗与性能评估不统一，且小模型与多领域实验不足。

**⚠️ 局限性**

研究数量有限、领域与语言聚焦单一、缺乏专用可持续性基准、硬件层能耗测量不足、可持续性微调方法缺乏、对小模型和非通用应用关注不足。

---

## 298. Probabilistic Learning and Generation in Deep Sequence Models

**arXiv ID:** 2603.00888 | [PDF](https://arxiv.org/pdf/2603.00888v1)

**作者:** Wenlong Chen `[一作]` (Imperial College London), Wenlong Chen `[通讯]` (Imperial College London)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5104157412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

该论文提交用于获得哲学博士学位，主要内容为作者的原创研究。

**💡 创新点**

论文的创新点在于提出了一种新的研究方法或理论框架，具体细节未提供。

**🔧 技术方法**

使用的技术未在提供的内容中详细说明。

**📊 数据集**

使用的数据集未在提供的内容中详细说明。

**📈 对比分析**

比较的方法和性能评估未在提供的内容中详细说明。

**⚠️ 局限性**

论文的局限性未在提供的内容中详细说明。

---

## 299. A Practical Guide to Streaming Continual Learning

**arXiv ID:** 2603.01677 | [PDF](https://arxiv.org/pdf/2603.01677v1)

**作者:** Andrea Cossu `[一作]` (University of Pisa), Davide Bacciu `[通讯]` (University of Pisa)

**通讯引用:** 87970 | [OpenAlex ID](https://openalex.org/A5043473089)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并探讨了Streaming Continual Learning (SCL) 这一融合SML和CL的新的学习范式，并用MNIST数据集构造虚拟与真实漂移场景进行实验。

**💡 创新点**

创新点在于系统性地将连续学习与流式学习的目标统一起来，阐述了SCL的五大能力，并通过实验证明单独的SML或CL方法无法同时满足快速适应和知识保持。

**🔧 技术方法**

采用了Adaptive Random Forest（SML）、Experience Replay、AGEM（CL）以及无策略的普通fine‑tune模型，并利用UMAP降维、River与Avalanche框架实现。

**📊 数据集**

实验使用MNIST数据集，按照两种漂移设置（虚拟漂移：每个经验包含两个数字的奇偶二分类；真实漂移：五个不同二分类任务）构成数据流。

**📈 对比分析**

比较方法包括预先评估（prequential）与CL评估（K_avg、BWT）两种指标。结果显示：SML模型ARF适应快但严重遗忘；CL策略ER、AGEM在虚拟漂移下可保持知识，但在真实漂移中表现不佳；普通模型在快速适应上优于CL，但遗忘率高。

**⚠️ 局限性**

局限性包括：实验仅在MNIST上进行，缺乏大规模真实世界数据验证；现有CL方法对真实漂移不友好；未给出完整的SCL系统实现，仅提出概念框架；未充分处理时间序列依赖和更复杂的漂移类型。

---

## 300. Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training

**arXiv ID:** 2603.02208 | [PDF](https://arxiv.org/pdf/2603.02208v1)

**作者:** Valentin Lacombe `[一作]` (University of Lille), Damien Sileo `[通讯]` (University of Lille)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出Reasoning Core，一个可扩展的程序化生成器集合，用于生成可验证的符号推理数据，并在预训练和后训练中提升语言模型的推理能力。

**💡 创新点**

其创新在于将随机化的核心符号域（PDDL规划、全一阶逻辑、CFG解析、贝叶斯网络因果推理和方程求解）与外部求解器验证结合，并提供连续难度控制与可训练的推理轨迹。

**🔧 技术方法**

采用程序化生成、外部求解器（如FastDownward、Vampire、Sympy）验证、上下文无关语法框架、并行生成管线等技术。

**📊 数据集**

数据集包含约5B预训练标记（10M样本）和1B后训练标记（1M样本），公开发布在GitHub与HuggingFace。

**📈 对比分析**

在与FineWeb、SYNTH、Dolci等自然语言数据混合预训练或指令微调时，模型在PlatinumBench上答案NLL显著降低，且对整体语言建模损失无负面影响；GPT-5在零样本评估中任务难度高。

**⚠️ 局限性**

局限性包括只覆盖形式化符号域、实验规模有限、缺乏RLVR评估以及生成过程虽多重验证但仍可能出现微小错误。

---

## 301. SeaVIS: Sound-Enhanced Association for Online Audio-Visual Instance Segmentation

**arXiv ID:** 2603.01431 | [PDF](https://arxiv.org/pdf/2603.01431v1)

**作者:** Yingjian Zhu `[一作]` (University of Chinese Academy of Sciences), Shiming Xiang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SeaVIS，首个在线音视频实例分割框架，解决离线模型无法持续推理和跨帧关联的缺陷。

**💡 创新点**

创新点包括因果交叉注意力融合（CCAF）将历史音频融入视觉特征，以及音频引导对比学习（AGCL）让实例嵌入同时编码外观与发声状态。

**🔧 技术方法**

采用Mask2Former+MSDeformAttn视觉解码器、VGGish/AudioMAE音频编码器、跨模态注意力、InfoNCE对比损失、外部记忆池关联与Transformer解码器。

**📊 数据集**

在AVISeg长视频数据集（26类声源、4类场景）上进行训练与评估。

**📈 对比分析**

与离线VIS/AVSS基线及最新AVISM对比，SeaVIS在FSLA、HOTA、mAP等指标上均实现领先，并保持约34–35fps的实时推理速度。

**⚠️ 局限性**

局限性包括对高噪声音频鲁棒性不足，难以完美处理多源快速切换和低音频质量场景；对超长视频的上下文管理仍有提升空间。

---

## 302. Predictive Reasoning with Augmented Anomaly Contrastive Learning for Compositional Visual Relations

**arXiv ID:** 2603.01125 | [PDF](https://arxiv.org/pdf/2603.01125v1)

**作者:** Chengtai Li `[一作]` (University of Nottingham Ningbo China), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15636 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为PR-A²CL的框架，用以解决组合视觉关系（Compositional Visual Relations）任务；框架包含增强异常对比学习（A²CL）感知模块和预测-验证推理模块（PARM），后者由层叠的预测异常推理块（PARBs）组成；

**💡 创新点**

创新点主要包括：① 通过弱/强数据增强配合对比学习，将正常实例的特征拉近、异常实例推远，从而获得更具判别力与泛化性的特征；② 将找出离群图像的任务拆解为四个预测-验证子问题，利用PARB层级递推捕捉多层次组合规则；③ 将对比学习与层级预测验证机制结合，模拟人类的反复推理过程；

**🔧 技术方法**

技术手段包括ResNet‑50编码器、弱/强增强、A²CL对比损失、BCE损失、PARB层级结构、残差跳连、t‑SNE可视化及实验中用到的多种数据增强策略；

**📊 数据集**

使用的公开数据集有SVRT（改造为四选一版本）、CVR以及更具挑战性的MC²R；

**📈 对比分析**

与现有SOTA方法（如WReN、SCL、PredRNet、SCAR、R³PCL、DBCR、SSL‑ResNet‑50等）在三大数据集上进行对比，PR‑A²CL在SVRT上AUC达99.4%（DBCR为98.8%），在CVR上整体优势约1–3%，在MC²R上最高精度90.4%（DBCR为89.3%）；在人类基准上，1k样本下表现优于人类，few‑shot场景仍落后；

**⚠️ 局限性**

局限性包括：在极少量样本（few‑shot）下性能显著下降；对复杂噪声或高层抽象规则的推理仍存在错误，主要因特征注意力偏向显著属性或规则耦合导致过拟合；模型对解释性与规则可视化仍有限；计算开销相对SOTA略高，但仍保持可接受水平。

---

## 303. FireRed-OCR Technical Report

**arXiv ID:** 2603.01840 | [PDF](https://arxiv.org/pdf/2603.01840v1)

**作者:** Hao Wu `[一作]` (Xiaohongshu Inc.), Changhao Qiao `[通讯]` (Xiaohongshu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FireRed-OCR 框架，将通用 Vision‑Language Model 转化为专门的 OCR 专家，解决结构幻觉问题，实现像素级精确的文档结构解析。

**💡 创新点**

核心创新在于“Geometry + Semantics”数据工厂与三阶段渐进训练（多任务预对齐 → 专门 SFT → GRPO 强化学习）相结合，既提升了结构一致性，又保持了文本识别的准确性。

**🔧 技术方法**

采用 Qwen3‑VL 作为基础模型，利用几何聚类+多维标签构建多样化数据，结合多任务预对齐、专门化微调和 Group Relative Policy Optimization（GRPO）等技术实现结构约束与语义对齐。

**📊 数据集**

利用多种公开数据集（OmniDocBench v1.5、OCRBench、PubTabNet、LaTeX OCR、IAM、BLIP‑3、Docmatix 等），并通过合成与专家校正（如 Gemini‑3 Pro）生成高质量的标注与合成样本。

**📈 对比分析**

在 OmniDocBench v1.5 上以 92.94% 的整体分数超过 DeepSeek‑OCR2、dots.ocr 等现有 E2E OCR 模型，且仅使用 2B 参数，显示出在参数效率与整体性能上的显著优势。

**⚠️ 局限性**

主要限制包括对极端扫描噪声、极大或复杂表格层级结构的推断仍不够稳健，且对多语言支持和数据清洗过程仍依赖人工与 LLM 辅助，导致生产成本与可扩展性受限。

---

## 304. Uniform-in-time concentration in two-layer neural networks via transportation inequalities

**arXiv ID:** 2603.01842 | [PDF](https://arxiv.org/pdf/2603.01842v1)

**作者:** Arnaud Guillin `[一作]` (Universite Clermont-Auvergne), Paul Stos `[通讯]` (Universite Clermont-Auvergne)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对宽两层网络在SGD训练下的参数分布进行均匀时间高概率收敛分析

**💡 创新点**

引入T_p传输不等式并在岭正则化下实现全时空的均匀一致收敛率

**🔧 技术方法**

利用传输不等式、Wasserstein距离、同步耦合和大数/切分技术

**📊 数据集**

无具体数据集，采用理论推导和模拟验证

**📈 对比分析**

相较于传统固定时间窗口的界限，提供指数不退化、维度无关的收敛率

**⚠️ 局限性**

仅适用于有界/光滑激活函数、需较强正则化，且Wasserstein距离仍受维数影响

---

## 305. Empirical Impact of Dimensionality on Random Geometric SAT

**arXiv ID:** 2603.01892 | [PDF](https://arxiv.org/pdf/2603.01892v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 306. Regularized Diffusion-based Contract Model for Covert Semantic Entropy Control in LAENets

**arXiv ID:** 2603.01478 | [PDF](https://arxiv.org/pdf/2603.01478v1)

**作者:** Yansheng Liu `[一作]` (Nanjing University of Aeronautics and Astronautics), Jiawen Kang `[通讯]` (Guangdong University of Technology)

**通讯引用:** 19087 | [OpenAlex ID](https://openalex.org/A5062761975)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对低空经济网络（LAENets）中受低探测概率约束的隐蔽语义通信，提出了一套以语义熵控制为核心的激励兼容框架，并利用合同理论与前景理论设计了基于无人机（UAV）私有信息的可实现契约；为解决高维非凸最优契约设计问题，提出了正则化扩散式Soft Actor-Critic（RDSAC）算法，完成对语义熵与报酬的联合学习。

**💡 创新点**

创新点包括：①将语义熵作为可观测契约变量，实现隐蔽语义传输中的不确定性可控；②在信息不对称下引入前景理论捕捉基站的风险偏好；③将扩散模型嵌入SAC框架，加入扩散熵与动作熵正则化，显著提升学习稳定性与探索效率；④构建了可兼容多层语义抽象的语义熵调节模块。

**🔧 技术方法**

主要技术包括：语义通信与层次化语义抽象、契约理论与前景理论建模、概率无线信道与隐蔽检测模型、强化学习中的条件扩散模型、Soft Actor-Critic与熵正则化、双Q学习与经验回放。

**📊 数据集**

实验数据基于合成视觉场景：使用CLIP视觉编码器提取特征，并通过t‑SNE进行低维可视化；通过不同信噪比（SNR）下的 PER 与 𝒬 指标进行评估；无人机类型与资源成本也以随机采样方式生成。

**📈 对比分析**

对比方法包括：完整信息下的契约（CC）、随机契约、传统 SAC 与 PPO；在前景理论下的平均奖励与标准差进行比较。实验结果显示 RDSAC 在平均奖励上比 SAC 提升 3.41%、比 PPO 提升 31.44%，并在不同环境、无人机类型与 SNR 变化下保持稳定的低方差表现。

**⚠️ 局限性**

局限性：①实验仅基于仿真与合成数据，缺乏真实无线与视觉传感数据验证；②未给出 RDSAC 的理论收敛性与最优性分析；③契约设计依赖预先设定的类型分布与参数，实际场景中无人机类型分布可能更复杂；④扩散模型训练成本较高，扩展到大规模多无人机网络时计算与存储负担需进一步研究。

---

## 307. Energy Efficiency Maximization for Integrated Sensing and Communications in Satellite-UAV MIMO Systems

**arXiv ID:** 2603.01717 | [PDF](https://arxiv.org/pdf/2603.01717v1)

**作者:** Ngo Tran Anh Thu `[一作]` (Hanoi University of Science and Technology), Hoang D. Le `[通讯]` (University of Aizu)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对LEO卫星与UAV共存的ISAC（集成感知与通信）MIMO系统，提出了一种联合功率与波束赋形的能效最大化框架，并给出了基于Dinkelbach迭代与半正定松弛（SDR）的高效交替优化算法。

**💡 创新点**

创新点包括：①将卫星与UAV的感知与通信任务在同一平台上完整协同，突破传统只做感知或通信的单一功能模式；②在联合优化中同时考虑多用户QoS、感知波束图约束和Rician概率信道模型；③利用Dinkelbach方法将分式能效目标转化为可求解的凸子问题，并证明SDR松弛后仍可获得等价的秩一解。

**🔧 技术方法**

主要技术手段有：Rician概率信道建模、LoS/NLoS概率混合、基于Steering向量的MIMO传输模型、Dinkelbach分式优化、二次变换与凸化、半正定松弛（SDR）以及CVX求解框架；实现时采用迭代交替优化实现收敛。

**📊 数据集**

实验使用模拟数据：基于Starlink星座的卫星参数、3GPP Release‑20 UAV模型、实际信道频率、功率、天线数量等参数，仿真平台为AMD Ryzen 9 7950X桌面电脑；并未使用公开的真实数据集。

**📈 对比分析**

与遗传算法（GA）和差分进化（DE）两种启发式算法进行对比，结果显示所提AO算法在能效上分别提高约90%、80%和72%（取决于场景），并在收敛速度和能效-波束阈值折衷上均优于基准方法。

**⚠️ 局限性**

局限性主要体现在：①仅考虑单一卫星与单一UAV的双平台配置，未覆盖多卫星/多UAV协同场景；②假设CSI完美或仅做简化估计，实际中可能存在估计误差；③仿真基于理想化概率模型，未对实际测量信道进行验证；④对动态移动、时变环境的鲁棒性未做深入研究。

---

## 308. UTICA: Multi-Objective Self-Distllation Foundation Model Pretraining for Time Series Classification

**arXiv ID:** 2603.01348 | [PDF](https://arxiv.org/pdf/2603.01348v1)

**作者:** Yessin Moakher `[一作]` (Ecole Polytechnique), Vasilii Feofanov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于DINOv2自蒸馏的时间序列基础模型Utica，利用多尺度随机裁剪与局部掩码联合预训练。

**💡 创新点**

创新点在于将多crop和掩码相结合的自蒸馏损失引入时间序列，突破了传统对比学习对负样本假设的限制。

**🔧 技术方法**

采用Transformer编码器、Mantis分词器、DINO、iBOT与KoLeo正则等技术。

**📊 数据集**

预训练使用合成DAG生成的时间序列，评估在UCR 128个单变量和UEA 21个多变量数据集上。

**📈 对比分析**

在线性探测与微调两种评估方式下，Utica均超过Mantis、Moment、NuTime等基线，平均线性精度达79.4%，微调精度达85.7%。

**⚠️ 局限性**

局限性包括仅在合成数据预训练、模型规模相对较小、未探索更大规模或不同结构的网络，以及对跨领域适用性的进一步验证不足。

---

## 309. Hexasort -- The Complexity of Stacking Colors on Graphs

**arXiv ID:** 2603.01244 | [PDF](https://arxiv.org/pdf/2603.01244v1)

**作者:** Linus Klocker `[一作]` (TU Wien), Simon D. Fink `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了单人堆叠游戏Hexasort的计算复杂性，证明了其在多种图结构（如两条独立边、树的有界高度或度）下的NP-难度，并给出了动态规划和按颜色/阈值参数化的FPT算法。

**💡 创新点**

创新点在于首次将Hexasort作为理论问题进行分析；通过构造专门的gadget，将Partition和3-Partition的归约映射到Hexasort，实现了弱与强NP难度的证明；提出了针对颜色数和阈值的参数化算法。

**🔧 技术方法**

主要技术包括图论归约、gadget构造、动态规划（状态空间为所有可能的堆叠配置）以及参数化复杂性分析。

**📊 数据集**

论文没有使用实验数据集，所有结果均来自理论构造和证明。

**📈 对比分析**

比较方式基于归约证明NP难度，算法性能以时间复杂度给出：动态规划为O(|S|·(|C|·t)^|V|)，在小图或参数化限制下可多项式；未给出实验性能对比。

**⚠️ 局限性**

局限性：对阈值为常数的情形尚未解决；强NP难度仅在树结构下证明，对常数颜色的强难度未完成；未考虑实际游戏中混合颜色堆叠的扩展。

---

## 310. The Sentience Readiness Index: Measuring National Preparedness for the Possibility of Artificial Sentience

**arXiv ID:** 2603.01508 | [PDF](https://arxiv.org/pdf/2603.01508v1)

**作者:** Tony Rost `[一作]` `[通讯]` (Harder Problem Project), Tony Rost (Harder Problem Project)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出并计算了Sentience Readiness Index（SRI），衡量31个司法管辖区在AI可能具备感知能力时的制度、专业与文化准备程度；

**💡 创新点**

首次构建针对AI感知治理的综合指数，采用LLM辅助专家评分、OECD/JRC框架以及可重复的加权方法，填补了现有AI成熟度指数对伦理与道德地位缺失的空白；

**🔧 技术方法**

使用LLM辅助评分、结构化提示、专家复核、OECD/JRC复合指标流程、加权算术平均、敏感性与稳健性分析；

**📊 数据集**

收集并整理了31个国家/地区的法律文件、政策框架、科研产出、专业培训、公共话语与适应性能力等多维度数据；

**📈 对比分析**

与现有AI成熟度指数（如Oxford、IMF、Stanford等）进行对比，发现SRI在“制度参与、专业准备、公共话语”三大新维度上显著低于其他指数；目前无任何司法管辖区达到“中等准备”以上，显示整体准备不足；

**⚠️ 局限性**

局限包括单次横断面数据、样本规模有限、加权方法主观性、LLM知识更新与偏见、缺乏正式的互评可靠性测试、仅评估国家层面而非子国家级差异，且对未来AI感知可能性仍保持高度不确定性。

---

## 311. A Systematic Study of LLM-Based Architectures for Automated Patching

**arXiv ID:** 2603.01257 | [PDF](https://arxiv.org/pdf/2603.01257v1)

**作者:** Qingxiao Xu `[一作]` (Texas A&M University), Jeff Huang `[通讯]` (Texas A&M University)

**通讯引用:** 3566 | [OpenAlex ID](https://openalex.org/A5052381120)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对LLM驱动的自动漏洞修补系统进行架构级评估，提出并实现了四种架构范式（固定流程、单一代理、多代理和通用编码代理），并在AIxCC基准上统一实验验证。

**💡 创新点**

首次系统比较这四种架构，揭示架构设计比模型能力更影响修补可靠性和成本，并量化不同架构在修补成功率、成本和鲁棒性上的优势与劣势。

**🔧 技术方法**

使用GPT-5、Claude Sonnet‑4.5等LLM与多种工具调用（搜索、编辑、测试等），构建固定流程、单代理、共享状态多代理和通用编码代理四种架构；采用上下文压缩、流式生成与工具调度实现高效交互。

**📊 数据集**

实验数据来自DARPA AIxCC Delta‑Scan Java漏洞基准，包含19个真实项目、提交差异、PoV、测试脚本等信息。

**📈 对比分析**

通过统一评估框架对修补正确率、Token使用、执行时延和工具调用次数等指标进行比较；结果显示通用编码代理修补成功率最高（16/19），单代理次之（12/13），多代理略低，固定流程最低；但通用代理Token成本最高，执行时延亦较长。

**⚠️ 局限性**

局限性包括仅评估19个Java项目，缺乏跨语言验证；通用编码代理可能出现误报或过度自适应导致失败；多代理系统开销大且对复杂任务敏感；固定流程缺乏自适应性，易碎；未评估模型更新、不同攻击情景或更大规模项目的表现。

---

## 312. GenDB: The Next Generation of Query Processing -- Synthesized, Not Engineered

**arXiv ID:** 2603.02081 | [PDF](https://arxiv.org/pdf/2603.02081v1)

**作者:** Jiale Lao `[一作]` (Cornell University), Immanuel Trummer `[通讯]` (Cornell University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 GenDB，一种基于 LLM 的代理式系统，能够针对每个查询动态生成实例化、性能优化的执行代码。

**💡 创新点**

创新点在于：① 用 LLM 直接合成查询执行代码而非手工编码；② 采用多代理分解架构，分别负责工作负载分析、存储/索引设计、查询规划、代码生成和迭代优化；③ 在运行时通过反馈循环逐步改进性能；④ 通过对数据分布、硬件特性和查询语义的联合推理实现极具针对性的优化。

**🔧 技术方法**

使用技术包括：Claude Sonnet 4.6 作为核心 LLM；多代理（agentic）系统；JavaScript + C++ 代码生成；JSON 结构化交互；运行时反馈机制；基于硬件感知的聚合、哈希、索引策略。

**📊 数据集**

使用数据集：TPC‑H（SF=10）和自定义的 SEC‑EDGAR（金融报表数据，约 5 GB，1,000 条随机查询）。

**📈 对比分析**

与 DuckDB、Umbra、MonetDB、ClickHouse、PostgreSQL 在相同硬件（两颗 Intel Xeon Gold 5218，384 GB RAM）上对比；不需要手工调参或索引；结果显示 GenDB 在 TPC‑H 5 题上总时长 214 ms，约比最快基线 DuckDB/ Umbra 的 594 ms 低 2.8×，在 SEC‑EDGAR 上 328 ms，约比 DuckDB 的 1,640 ms 低 5×；性能提升随查询复杂度递增。

**⚠️ 局限性**

局限性包括：缺乏对自然语言查询的形式化正确性保证；LLM 生成代码可能出现性能低下或无限循环的“无声失败”；生成成本高、token 消耗大；需要人工验证结果或对已知基准做迭代；对未见过的数据/查询的泛化性仍有限；系统对硬件和数据分布的假设可能不适用于所有环境。

---

## 313. Incremental LTLf Synthesis

**arXiv ID:** 2603.01201 | [PDF](https://arxiv.org/pdf/2603.01201v1)

**作者:** Giuseppe De Giacomo `[一作]` (University of Oxford), Moshe Y. Vardi `[通讯]` (Rice University)

**通讯引用:** 39189 | [OpenAlex ID](https://openalex.org/A5000059818)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在执行过程中增量提供目标的反应式综合（incremental synthesis），并给出一种利用 DFA 进展与缓存实现的高效算法。

**💡 创新点**

创新点在于将 DFA 进展与游戏求解结合，显著减少了每次加入新目标时的自动机重构开销，并证明了进展后的最小自动机大小不超过原自动机。

**🔧 技术方法**

主要技术包括：线性时序逻辑（LTL）到 DFA 的转换、自动机进展（progression）、产品自动机构造、符号游戏求解与 DFA 缓存。

**📊 数据集**

实验使用四组基准：车辆导航（10 个位置）、计数器（6-10 位）、植物护理（1-10 株）和服务交付（1-3 服务，每个服务 1-4 步骤），共计 1230 个目标。

**📈 对比分析**

与基于公式进展的实现对比，所提方法在大多数实例中能添加更多目标、平均加入时间更低（最多 1-2 个数量级），验证了其优越的性能；缺点是受限于内存（8GB）和自动机构造的双指数上界。

**⚠️ 局限性**

局限性包括：未解决目标冲突和优先级管理、对环境动态变化和部分可观测性支持不足，以及对极大状态空间实例的内存消耗。

---

## 314. Physical Layer Security for Sensing-Communication-Computing-Control Closed Loop: A Systematic Security Perspective

**arXiv ID:** 2603.00943 | [PDF](https://arxiv.org/pdf/2603.00943v1)

**作者:** Chengleyang Lei `[一作]` (Tsinghua University), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 29857 | [OpenAlex ID](https://openalex.org/A5030858163)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种针对非地面网络（NTN）辅助的感知-通信-计算-控制（SC³）闭环系统的物理层安全（PLS）设计，重点考虑闭环任务相关信息泄露约束，旨在通过联合分配传输时隙、功率、带宽与边缘计算资源，最大化闭环负熵（CNE）来提升闭环控制性能并确保信息安全。

**💡 创新点**

创新点包括：①引入闭环负熵（CNE）作为衡量闭环控制性能的新指标；②提出基于闭环任务相关信息泄露阈值的全局最优安全约束；③在不同信道顺序下提供KKT与单调优化（MO）相结合的全局最优求解算法，显著提升相较于传统链路级或加和链路级安全设计的性能。

**🔧 技术方法**

技术手段主要包括：物理层安全技术（利用信道优势降低对手可得信息）、Karush‑Kuhn‑Tucker（KKT）条件分析、单调优化（MO）理论以及多维Polyblock外逼近算法，以实现对功率、带宽、时隙与计算能力的联合最优化。

**📊 数据集**

本文采用仿真评估，未使用公开数据集；仿真参数基于自由空间与三维多径衰减模型，使用随机坐标与路径损耗指数生成信道增益。

**📈 对比分析**

与基准方案（控制导向闭环优化与单链路独立优化）相比，本文方法在多种信道条件与安全阈值下均能获得更高的CNE，实验显示在合法信道优势或受限带宽/功率下可提升约10‑30%。

**⚠️ 局限性**

主要局限包括：仅针对单一对手已知位置的情形；对信道不确定性和多对手情境的鲁棒性设计仍待完善；算法在更高维资源分配（如多频段、多天线）下的扩展复杂度未充分验证。

---

## 315. Towards Non-Latin Text and Layout Personalization for Enhanced Readability

**arXiv ID:** 2603.00688 | [PDF](https://arxiv.org/pdf/2603.00688v1)

**作者:** Rina Buoy `[一作]` (Techo Startup Center, Ministry of Economy and Finance), Koichi Kise `[通讯]` (Osaka Metropolitan University)

**通讯引用:** 3287 | [OpenAlex ID](https://openalex.org/A5000232184)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在非分段、非拉丁文字（高棉语和日语）中使用基于词性（POS）的字符级文本与排版个性化技术，以提升阅读可读性与记忆力。

**💡 创新点**

创新点：提出不插入空格而通过粗体或颜色等字符级样式在词性层面显式化语法边界的模块化技术；并针对高棉语采用粗体、对日语采用颜色编码的两种语言特定实现，首次在实验中验证其对阅读理解与记忆的正向影响。

**🔧 技术方法**

技术手段：使用高棉语Transformer POS/分词模型、GiNZA+SudachiPy做日语POS分析；字符级样式工具将POS映射到粗体或颜色；采用随机交替阅读实验、MCQ与关键词记忆测验；统计检验包括卡方检验、配对t检验、广义估计方程（GEE）与广义线性混合模型（GLMM）。

**📊 数据集**

数据集：Kaing等高棉语词性标注与分词数据集；10篇英文Newsela文章（约250词），经机器翻译并人工校正后得到高棉语和日语版本；为每篇文章生成四道MCQ和10个关键词（5个真关键词+5个相似非关键词）。

**📈 对比分析**

对比方法：在随机交替的阅读顺序下，让每位受试者分别阅读带样式和无样式文本，测量阅读理解准确率、关键词记忆准确率、阅读/答题时间以及主观难度；高棉语样式显著提升理解（最高+15%）和关键词记忆（p<0.05），日语样式在理解上显著正向（OR>1，p<0.05），但导致平均阅读时间增加约8秒。

**⚠️ 局限性**

局限性：日语实验仅10人；缺乏眼动、眼跳等高分辨率阅读行为数据；样式未针对个体偏好动态适配；POS特征为静态，未考虑读者意图；仅验证两种脚本，其他非拉丁脚本需进一步验证。

---

## 316. TopoCurate:Modeling Interaction Topology for Tool-Use Agent Training

**arXiv ID:** 2603.01714 | [PDF](https://arxiv.org/pdf/2603.01714v1)

**作者:** Jinluan Yang `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出TopoCurate框架，用拓扑感知的交互模型对工具使用数据进行两阶段精细筛选，提升SFT和RL训练效果。

**💡 创新点**

创新点在于：①构建语义商数拓扑，将多条轨迹压缩为统一状态空间；②针对SFT和RL分别设计反射恢复、语义效率、分布多样性以及错误分支比例和策略异质性等拓扑指标，实现结构化数据筛选，突破传统基于结果的“Outcome Equivalence Illusion”。

**🔧 技术方法**

使用图论/商拓扑、语义相似度匹配、强化学习中的GRPO、以及多种自定义拓扑指标；核心技术包括状态聚合、KL收敛分析和梯度信噪比优化。

**📊 数据集**

数据集涵盖BFCLv3（多轮工具调用）和Tau2-Bench（双控制环境）两大基准，用于评估模型在域内外的性能。

**📈 对比分析**

与APIGen‑MT、MUA、Simia-Tau等基线比较，TopoCurate在SFT上平均提升约4.2%，在RL上提升约6.9%；在Tau2-Bench和BFCLv3上均取得新SOTA，尤其在航空、零售、通信三大域显著提升通用成功率。

**⚠️ 局限性**

局限性包括：对语义相似度阈值的敏感性、构建拓扑所需的计算开销、以及在极高维度工具调用场景下商拓扑可能产生的过度聚合导致信息损失。

---

## 317. Bootstrapping Embeddings for Low Resource Languages

**arXiv ID:** 2603.01732 | [PDF](https://arxiv.org/pdf/2603.01732v1)

**作者:** Merve Basoz `[一作]` (University of Edinburgh), Mattia Opper `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究利用大型语言模型（Gemma 3 27B）生成合成三元组数据，用于训练低资源语言的句子嵌入模型，并比较了三种合成策略：基于提示的in-context学习、适配器组合（LoRA+AdamergeX）以及提出的跨语言LoRA（XL-LoRA）方法。

**💡 创新点**

创新点在于提出XL-LoRA，该方法在不需要目标语言平行数据的前提下，通过在英语中生成正负样本并保持目标语言anchor，实现跨语言的高质量合成；同时改进了适配器组合的训练策略，证明合成数据能显著提升低资源语言嵌入性能。

**🔧 技术方法**

技术包括Gemma 3 27B LLM、LoRA适配器、AdamergeX组合、XL-LoRA跨语言LoRA、SimCSE对比损失、无监督SimCSE、跨语言微调、零样本评估以及MTEB/STS基准。

**📊 数据集**

使用的数据集包括：Leipzig、Opus等公开语料库提取的anchor句子；英文NLI数据集；XNLI测试集（人类翻译）；高质量人类翻译文本；以及275k条合成三元组。

**📈 对比分析**

与基线（基线编码器、无监督SimCSE、跨语言英语NLI微调）对比，XL-LoRA在STS Spearman相关系数约81+、检索Recall@10显著提升；提示方法性能最差；适配器组合介于两者之间，略低于XL-LoRA。

**⚠️ 局限性**

局限性包括：实验仅限于参数<100B的LLM，较大模型可能进一步提升；评估语言覆盖有限，缺乏对不同语言学特征的系统分析；未探讨解码器式嵌入；合成质量高度依赖训练数据质量，且计算资源受限导致示例数有限。

---

## 318. MME: Mixture of Mesh Experts with Random Walk Transformer Gating

**arXiv ID:** 2603.00828 | [PDF](https://arxiv.org/pdf/2603.00828v1)

**作者:** Amir Belder `[一作]` (Technion Israel Institute of Technology), Ayellet Tal `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了基于Mixture of Experts（MoE）的混合专家框架，针对3D网格分类、检索和语义分割任务，融合多种异构专家网络，并通过基于随机游走的Transformer门控实现动态专家选择。

**💡 创新点**

创新点包括：①设计基于随机游走的Attention门控，能够聚焦网格关键区域并为异构专家动态分配；②引入相似性损失与多样性损失的动态权重调节，并用Soft Actor-Critic强化学习自动学习平衡策略；③通过门的预训练实现对专家关注区域的学习，显著提升专家选择效果。

**🔧 技术方法**

使用的核心技术有Transformer门控网络、随机游走抽取、交叉熵多样性损失、KLD相似性损失、SAC强化学习调节权重，以及六种主流网格网络（MeshCNN、MeshWalker、PD-MeshNet、AttWalk、MeshFormer、MeshNet）作为专家。

**📊 数据集**

在分类任务中使用SHREC11、ModelNet40、3D-FUTURE和Cube Engraving数据集；在检索任务中使用ShapeNet-Core55和ModelNet40；在语义分割任务中使用Human Body、COSEG和PartNet数据集。

**📈 对比分析**

与单一专家、投票集成以及各自论文报告的基准方法进行对比；在分类上实现SHREC11和Cube Engraving 100%准确率，在3D-FUTURE和ModelNet40分别提升多达8%；在检索上ShapeNet-Core55 mAP提升至93.2%（比基准高约8%）；在语义分割上Human Body、COSEG、PartNet分别提升约1–7%，总体表现均优于所有单专家和集成方法。

**⚠️ 局限性**

主要限制是训练和推理时间显著增加（多专家与RL代理导致每轮训练约21min、推理约270ms），部署复杂度提高；方法在已饱和的数据集上提升有限，且依赖门的预训练和RL收敛。

---

## 319. I-Perceive: A Foundation Model for Active Perception with Language Instructions

**arXiv ID:** 2603.00600 | [PDF](https://arxiv.org/pdf/2603.00600v1)

**作者:** Yongxi Huang `[一作]` (Shanghai Jiao Tong University), Panpan Cai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5077008575)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉语言模型与几何基础模型融合的基础模型，用自然语言指令预测机器人应采集的摄像头视角，实现开放式主动感知。

**💡 创新点**

创新点在于深度语义融合机制和语义-aware VGGT，首次将VLM语义特征跨层注入几何推理，从而支持任意自然语言指令的视角生成。

**🔧 技术方法**

采用Qwen3-VL作为视觉语言支柱，VGGT式几何网络，并在多层跨注意力中注入语义特征，同时使用LoRA微调与多任务监督。

**📊 数据集**

使用162K来自ScanNet与CA-1M的真实扫描任务与70K来自HSSD的仿真任务进行预训练与微调。

**📈 对比分析**

与GPT-5.2、Gemini3-Pro、Qwen3-VL等基线相比，在View Coverage IoU上提升至46.8%（高于最佳基线34.3%），并在VLM与人工评判中获得平均排名靠前，表现优异。

**⚠️ 局限性**

局限性包括未考虑碰撞与可达性约束，生成的视角可能不可执行；并且依赖两大预训练模型，缺乏统一的多模态学习框架。

---

## 320. Can LLMs Hack Enterprise Networks? -- Replicated Computational Results (RCR) Report

**arXiv ID:** 2603.01789 | [PDF](https://arxiv.org/pdf/2603.01789v1)

**作者:** Andreas Happe `[一作]` (TU Wien), Jürgen Cito `[通讯]` (TU Wien)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个名为Cochise的自主渗透测试原型，利用大型语言模型（LLM）在微软Active Directory的GOAD实验环境中模拟攻击，评估LLM在真实企业网络中的渗透能力；

**💡 创新点**

创新点在于首次将推理型LLM与自动化工具链相结合，形成完整的高层策略规划与低层执行闭环；同时提供公开的原型、日志与分析脚本，构建可复现的渗透测试框架；

**🔧 技术方法**

技术包括Python实现、Docker容器化、Vagrant+Ansible搭建GOAD测试网、使用LLM API（OpenAI GPT‑4o、o1、DeepSeek、Google Gemini、Alibaba SLM）进行策略生成与工具调用；

**📊 数据集**

数据集为GOAD实验环境（5台Windows VM构成的Active Directory网络）及其渗透测试过程中生成的JSON日志文件；

**📈 对比分析**

对比方法：在同一实验网下对多款LLM进行多次实验，记录受害账户数量、近乎成功攻击、可行路径、token使用量等指标；结果显示推理型LLM（o1）在生成策略和发现路径上表现更佳，但token消耗更高；

**⚠️ 局限性**

局限性包括：需搭建资源密集的虚拟机环境，实验仅覆盖5台VM的AD网络，无法代表更大规模企业网络；LLM输出缺乏安全防护；实验结果受LLM温度和调用方式影响，难以完全泛化。

---

## 321. Frontier Models Can Take Actions at Low Probabilities

**arXiv ID:** 2603.02202 | [PDF](https://arxiv.org/pdf/2603.02202v1)

**作者:** Alex Serrano `[一作]` (MATS), Erik Jenner `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大语言模型在前置部署评估中以极低概率执行目标行为（潜在违约）的能力。

**💡 创新点**

提出一种可重复的低概率行动校准实验框架，并揭示外部熵与自生成熵对模型校准性能的决定性影响。

**🔧 技术方法**

采用大模型提示、熵提取与阈值决策、链式推理、统计置信区间和相对校准误差等技术进行评估。

**📊 数据集**

使用编程（Python）、商务邮件和改造的石头剪刀布游戏三类任务作为实验数据集。

**📈 对比分析**

在GPT‑5、GPT‑4o与Qwen‑3系列模型上比较不同熵条件下的校准误差、最低可校准率（LCR）和输出概率底限（OPF），发现GPT‑5在低概率下表现最佳，旧一代模型差距显著。

**⚠️ 局限性**

局限性包括仅评估单轮显式目标行动、未覆盖自主策略生成、熵格式种类有限以及链式推理可见性受限。

---

## 322. VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection

**arXiv ID:** 2603.00912 | [PDF](https://arxiv.org/pdf/2603.00912v1)

**作者:** Yang Cao `[一作]` (Hong Kong University of Science and Technology), Dan Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 4927 | [OpenAlex ID](https://openalex.org/A5100778603)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Sensor-Geometry-Free（SG-Free）多视角室内 3D 目标检测框架 VGGT-Det，消除对相机姿态/深度的依赖；

**💡 创新点**

核心创新包括：1）Attention-Guided Query Generation（AG），利用 VGGT 编码器的注意力热图为目标查询提供语义先验；2）Query-Driven Feature Aggregation（QD），通过可学习的 See-Query 动态聚合多层几何特征；

**🔧 技术方法**

技术方案基于 Transformer 编码-解码架构，集成 VGGT 预训练 3D 重建编码器，并在解码器中实现自注意力与交叉注意力；

**📊 数据集**

在 ScanNet 与 ARKitScenes 两大室内数据集上进行实验，评估 mAP@0.25；

**📈 对比分析**

与改造成 SG-Free 的 ImVoxelNet、NeRF-Det、MVSDet 以及 FCAF3D 进行对比，VGGT-Det 在 ScanNet 上提升 4.4 mAP，在 ARKitScenes 上提升 8.6 mAP；

**⚠️ 局限性**

局限性包括：对 VGGT 预训练模型的依赖，需在多视角场景中拥有足够多的图像帧；在极小或隐蔽物体的检测上仍可能受限；

---

## 323. Protection against Source Inference Attacks in Federated Learning

**arXiv ID:** 2603.02017 | [PDF](https://arxiv.org/pdf/2603.02017v1)

**作者:** Andreas Athanasiou `[一作]` (Delft University of Technology), Catuscia Palamidessi `[通讯]` (Inria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在联邦学习的洗牌模型中提出了一种针对来源推断攻击（SIA）的防御机制，核心是对模型参数进行细粒度洗牌并结合残数系统（RNS）编码与单元编码，实现对每个参数的位级加密，从而在不引入噪声的前提下将SIA的成功率压至随机猜测水平。

**💡 创新点**

创新点包括：①证明标准的模型/层级洗牌不足以抵御SIA；②提出参数级洗牌+RNS+单元编码的组合方案，利用RNS的可加性与位级编码实现隐私放大；③在保持模型准确率、可与DP等其他隐私技术无缝集成的同时，兼容多种洗牌信任假设；④给出了完整的实现细节与通信成本分析。

**🔧 技术方法**

技术手段：参数级洗牌、残数系统（RNS）编码、单元（unary）位编码、MixNet或可信洗牌器、与DP或安全聚合的对齐、以及压缩技术（RLE）在完全可信洗牌时的应用。

**📊 数据集**

实验数据集：MNIST（CNN）、CIFAR‑10（CNN）、CIFAR‑100（ResNet‑18），并在不同异构度（Dirichlet α）下测试；还进行了对比实验，包括未洗牌、层级洗牌、参数级洗牌以及安全聚合（SA）。

**📈 对比分析**

对比方法：与传统FL、层级/参数级洗牌、以及基于阈值秘密共享的安全聚合进行对比。结果显示：①无洗牌时SIA成功率高达80%+；②层级洗牌降低到≈50%；③参数级洗牌进一步降至≈30%；④本方法将SIA成功率压至≈随机猜测（≈1/n）；模型准确率在保持r=2–3位小数时几乎不变；通信成本在跨机房场景下仅为原来的1.04–1.81倍，完全可信洗牌可降至≈1.03倍。

**⚠️ 局限性**

局限性：①仅适用于基于求和的聚合函数（如FedAvg、FedSGD、FedProx），不适用于中值、聚类或排名等非求和聚合；②需要对每个参数进行RNS和位级编码，导致实现复杂性和一定的计算/通信开销；③假设存在可信或半可信洗牌器，若洗牌器完全失效或被破坏，安全性无法保证；④对极大规模跨设备场景的可扩展性仍需进一步验证。

---

## 324. DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking

**arXiv ID:** 2603.01367 | [PDF](https://arxiv.org/pdf/2603.01367v1)

**作者:** Gilad Turok `[一作]` (Cornell University), Volodymyr Kuleshov `[通讯]` (Cornell University)

**通讯引用:** 6040 | [OpenAlex ID](https://openalex.org/A5021338648)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DUEL 框架，利用确定性位置选择实现 Masked Diffusion Model 的精确似然计算，给出与自回归模型等价的困惑度指标。

**💡 创新点**

将确定性解码策略与任何顺序自回归视角统一，证明可在单条生成路径上得到 Exact Likelihood，解决 ELBO 与生成困惑度不一致的问题。

**🔧 技术方法**

利用 Deterministic Unmasking 的确定性策略、Ordered Partition 形式化、精确似然累积算法，结合已有的 MDM 预训练网络和多种位置选择规则。

**📊 数据集**

在 OpenWebText、LM1B、PTB、Wikitext、Lambada、AG News 以及 8B LLaDA、Llama3 等大模型数据集上评估。

**📈 对比分析**

通过 DUEL 计算的困惑度与 ELBO、生成困惑度对比，发现 DUEL 在域内将 MDM–ARM 的困惑度差距缩小 21–32%，在零样本上 30–82%，并且在多 NFE 预算下可为不同采样策略提供一致的排名。

**⚠️ 局限性**

局限在于仅评估确定性采样策略，忽略采样策略（top‑k、 nucleus），需逐步前向推理，计算成本高，且未涵盖多样性或下游任务评估。

---

## 325. LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval

**arXiv ID:** 2603.01425 | [PDF](https://arxiv.org/pdf/2603.01425v1)

**作者:** Jiajie Jin `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3990 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自蒸馏框架LaSER，将大语言模型中的显式Chain‑of‑Thought推理过程内化为稠密检索器的潜在空间，实现了在不产生文本的情况下进行隐式推理。

**💡 创新点**

创新点在于双视图（Explicit‑View与Latent‑View）共训练以及多粒度对齐策略：输出层的KL蒸馏和过程层的轨迹对齐，确保潜在思考token能够捕捉到显式推理路径的语义进展。

**🔧 技术方法**

核心技术包括：大规模生成式LLM检索器、共享参数的自蒸馏训练、轨迹对齐（Temporal Downsampling+KL loss）、对比学习（InfoNCE）以及使用soft token生成的连续潜在思考。

**📊 数据集**

训练使用ReasonEmb合成数据集（81k例、12域），每条query附带GPT‑4o‑mini生成的CoT；评测在Bright、FollowIR、BrowseComp‑Plus等需深度推理的基准上进行。

**📈 对比分析**

与标准稠密检索器、纯对比学习、Rewrite‑then‑Retrieve以及现有隐式推理方法相比，LaSER在Bright上平均nDCG@10提升约15%，在FollowIR、BC‑Plus上Recall/Recall@k也均超过所有基线；且推理延迟仅为Rewrite‑then‑Retrieve的0.3%，接近单向检索。

**⚠️ 局限性**

局限性包括：仍需显式CoT数据进行蒸馏，对大规模多模态或长文档场景的适配未充分验证；隐式思考步骤数较少时可能不足以捕获极复杂逻辑；以及对自蒸馏过程的动态平衡调参仍需进一步研究。

---

## 326. LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models

**arXiv ID:** 2603.00592 | [PDF](https://arxiv.org/pdf/2603.00592v1)

**作者:** Yuchen Hou `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**通讯引用:** 13563 | [OpenAlex ID](https://openalex.org/A5110190620)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了语义扰动分类法并提出LangGap基准，用同一场景多任务设计迫使模型依赖语言指令。

**💡 创新点**

首个在视觉-语言-动作模型中通过同一视觉状态下多样化语言指令逼迫语言理解的基准，以及细粒度的四维语义扰动诊断方法。

**🔧 技术方法**

使用Transformer VLA模型π0.5，LoRA微调，四维语义扰动生成，基于LIBERO的任务扩展与数据增强。

**📊 数据集**

主要使用LIBERO、LIBERO-Plus任务以及自构建的LangGap（99任务，其中59个为扩展语义任务）。

**📈 对比分析**

与原始π0.5、π0、π0-FAST、SmolVLA等模型对比，单任务微调可将性能从3.75%提升至90%，6任务提升至28%，但随任务规模增大表现急剧下降，显示扩展任务训练易被官方数据稀释。

**⚠️ 局限性**

语义维度有限，未覆盖复杂语法结构；实验仅在仿真环境中进行，缺乏真实机器人验证；同一视觉场景下多任务训练对模型泛化提升有限。

---

## 327. Can AI Agents Agree?

**arXiv ID:** 2603.01213 | [PDF](https://arxiv.org/pdf/2603.01213v1)

**作者:** Frédéric Berdoz `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21287 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

在同步全互连网络下评估LLM驱动的代理在无赌注的标量拜占庭共识游戏中的一致性与活跃度表现。

**💡 创新点**

① 引入受控拜占庭比例并分离有效性与活跃度两大指标；② 发现失败主要源于活跃度丢失而非价值偏差；③ 在实验中系统比较不同模型规模、群体大小和是否提示潜在拜占庭的影响。

**🔧 技术方法**

使用基于Prompt的LLM（Qwen3-8B/14B）生成JSON输出的策略，A2A-Sim同步网络仿真，统计分析采用95% Wilson置信区间。

**📊 数据集**

无外部真实数据集，初始提案由[0,50]均匀分布随机采样，拜占庭代理在每轮可自行选取任意值。

**📈 对比分析**

通过对比不同模型大小、群体规模、是否提到拜占庭以及拜占庭数量，对实验进行多维度评估；结果显示：即便无拜占庭，约40–70%实验能达到有效共识；随群体增大或加入拜占庭，成功率显著下降，主要失败模式为超时/无共识。

**⚠️ 局限性**

实验仅采用单一拜占庭策略、仅两种模型大小、单一LLM家族；未考虑更复杂攻击、异构代理或更大规模网络，限制了结论的普适性。

---

## 328. Bespoke OLAP: Synthesizing Workload-Specific One-size-fits-one Database Engines

**arXiv ID:** 2603.02001 | [PDF](https://arxiv.org/pdf/2603.02001v1)

**作者:** Johannes Wehrstein `[一作]` (Technical University of Darmstadt), Carsten Binnig `[通讯]` (Technical University of Darmstadt)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套自动化的 Bespoke OLAP 合成管道，利用大语言模型（LLM）从工作负载规范（查询模板与数据集）自动生成并优化专门化的 OLAP 数据库引擎。

**💡 创新点**

创新点在于：①将 LLM 与系统生成、性能评估、热补丁等工具结合，形成可迭代、可验证的“构建-验证-优化”循环；②通过工作负载驱动的存储布局规划、查询实现以及多阶段经验性优化，系统性消除通用引擎的性能税；③提供完整的基础设施（热补丁、回滚、版本化、自动化测试），使得从无到有的引擎合成在数小时、数美元内完成。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑5.2 Codex）作为代码生成与调优代理；Shell、Patch、Compile、Validate 四种工具实现与环境交互；热补丁机制实现无停机的增量编译与部署；自我追踪与回滚机制保证逐步优化时不产生回退；经验性 join‑order 搜索、计数器与专家知识注入等多阶段性能驱动优化。

**📊 数据集**

实验使用了两大 OLAP 典型基准：TPC‑H（规模因子 20）和 CEB（基于 IMDB 的 CEB benchmark，规模因子 2）。数据通过 DuckDB 生成器或公开的 CEB 工具获得，并在 AMD EPYC 96 核 768 GB 内存机器上运行。

**📈 对比分析**

与单线程、内存内的 DuckDB 进行对比。Bespoke‑OLAP 在 TPC‑H 上实现了 11.78× 的总体加速，单条查询最高 103.98×；在 CEB 上实现了 9.76× 的总体加速，单条查询最高 1466×。加速随数据规模增长保持稳定或提升，尤其在 CEB 上随着规模扩大加速从 9.2× 提升至 70×，体现了专用存储与执行的优势。

**⚠️ 局限性**

局限性包括：①仅处理单线程、内存内的 OLAP 场景；未考虑磁盘 I/O、并发与事务一致性；②依赖 GPT‑5.2 Codex，若模型更新或可用性改变需重新训练；③合成过程仍需数小时，虽然成本低，但对极短周期的动态负载仍不适用；④对极端复杂或动态演变的工作负载，需频繁重新合成，可能产生维护成本。

---

## 329. Fast Confidence-Aware Human Prediction via Hardware-accelerated Bayesian Inference for Safe Robot Navigation

**arXiv ID:** 2603.01122 | [PDF](https://arxiv.org/pdf/2603.01122v1)

**作者:** Michael Lu `[一作]` (Simon Fraser University), Mo Chen `[通讯]` (Simon Fraser University)

**通讯引用:** 11458 | [OpenAlex ID](https://openalex.org/A5100387253)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个可实时、可并行化的多人体信心感知预测框架，并将预测结果直接投影到栅格中，用于全局（ANA*）和局部（MPPI）规划，从而实现安全的机器人导航。

**💡 创新点**

核心创新在于：1）将完整的贝叶斯推理通过GPU粒子采样实现并行化，获得125 Hz的高频预测；2）使用可自适应更新的贝叶斯框架，保持多模态分布；3）将粒子预测直接映射到离散栅格，并与任何时间变规划器兼容；4）通过JIT+GPU实现300倍速度提升。

**🔧 技术方法**

技术手段包括：GPU并行粒子采样、JAX+JIT编译、差分平坦动力学模型、Boltzmann策略、粒子滤波、Gaussian卷积平滑、LogSumExp、离散栅格投影、时间变A*（ANA*）与MPPI控制。

**📊 数据集**

实验使用了在9.6 m×5.4 m实验室房间中实时采集的人类轨迹（通过ZED 2摄像头），未使用公开数据集，而是基于自建数据进行验证。

**📈 对比分析**

与CPU单线程Python实现相比，GPU+JIT实现获得约300倍的速度提升；相比全贝叶斯推理，粒子方法在时间步上更快、粒子数可增至数千；在多人体实验中，预测频率可达45 Hz，机器人在5人环境下以0.7–1.1 m/s安全导航，证明了方法的性能优势。

**⚠️ 局限性**

主要局限：1）跟踪误差和模型不匹配导致“近撞”事件；2）粒子预测可能穿过障碍物；3）未考虑机器人动作对人类行为的反馈；4）多人体联合分布更新仍为顺序方式，计算开销较大；5）缺乏群体动力学建模，难以进一步提升效率。

---

## 330. Communication-Efficient Quantum Federated Learning over Large-Scale Wireless Networks

**arXiv ID:** 2603.01222 | [PDF](https://arxiv.org/pdf/2603.01222v1)

**作者:** Shaba Shaon `[一作]` (University of Alabama in Huntsville), Dinh C. Nguyen `[通讯]` (University of Alabama in Huntsville)

**通讯引用:** 7058 | [OpenAlex ID](https://openalex.org/A5076942188)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种面向大型NOMA无线网络的量子联邦学习（QFL）框架，联合优化量子设备的信道选择与发射功率以实现上行总速率最大化，并对全设备参与下的收敛行为进行了理论分析。

**💡 创新点**

创新点包括：①首次将量子近似优化算法（QAOA）与块坐标下降（BCD）结合，用于求解非凸混合整数非线性规划（MINLP）问题；②提供了QFL在非凸损失、异构数据分布及量子击打噪声条件下的收敛上界；③通过量子算子映射将问题转化为Hamiltonian形式，实现了可在NISQ硬件上近似求解；④在实验中实现了>100%总速率提升与显著收敛速度加速。

**🔧 技术方法**

采用的技术包括：量子电路（PQC）与变分量子算法（VQA）、经典梯度下降（SGD/Adam）训练本地模型；QAOA（含问题与混合哈密顿量）求解QUBO；BCD迭代分解主问题；信道模型（路径损耗、阴影、瑞利衰落与Jakes相关性）；对比算法如SCA与贪心法；Python/PennyLane与TorchQuantum仿真工具。

**📊 数据集**

实验使用的公开数据集为MNIST（手写数字识别）与CIFAR-10；在量子训练实验中采用4量子比特PQC；无线模拟则采用合成信道参数（4信道、50/200/500设备，最大功率20 dBm，噪声功率-114 dBm）。

**📈 对比分析**

与SCA和贪心方法比较显示：QAOA在信道选择和功率分配上均实现了近100%更高的总速率，收敛迭代次数约为原方法的1/4；延迟降低约30–40%；在量子测量击打次数从1→100时，QFL模型精度提升超过10%，损失下降同样显著。QAOA在计算开销上相较SCA实现了50×以上的速度提升。

**⚠️ 局限性**

局限性：①依赖NISQ设备的噪声水平与足够的击打次数，实际硬件实现仍具挑战；②问题仍是NP‑hard，只能得到近似解，无法保证全局最优；③实验仅覆盖上传方向的NOMA，未考虑下行或多基站情形；④块坐标下降与QAOA的组合对设备规模增长时的可扩展性和资源消耗尚未充分验证。

---

## 331. Bound Propagation meets Constraint Simplification: Improving Logic-based XAI for Neural Networks

**arXiv ID:** 2603.01923 | [PDF](https://arxiv.org/pdf/2603.01923v1)

**作者:** Ronaldo Gomes `[一作]` (Instituto Federal do Ceará), Thiago Alves Rocha `[通讯]` (Instituto Federal do Ceará)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合边界传播和约束简化的逻辑基可解释AI方法，用于提升神经网络解释的计算效率。

**💡 创新点**

创新点在于先用Box方法对输入属性进行预判是否必要，若不必要可直接排除；若不确定则利用传播得到的更窄神经元边界简化MILP约束，删除不必要的二进制变量，从而大幅减少求解量。

**🔧 技术方法**

使用的技术包括逻辑基解释框架、混合整数线性规划(MILP)、Box边界传播、ReLU网络建模及约束简化。

**📊 数据集**

实验数据集为UCI的Iris、Wine、Sonar、Digits和MNIST，共涉及多种层数与宽度的网络。

**📈 对比分析**

与原INMS方法进行对比，测量总解释时间、MILP求解时间、边界收紧比例和二进制变量删除比例；结果显示在大型网络中，解释时间提升超过89%，MILP求解时间从数千秒降至数秒，且边界收紧率>90%，二进制变量删除率>50%。

**⚠️ 局限性**

局限性包括Box方法粗糙导致在简单模型上可能不节省时间、仍受MILP规模限制；未来可尝试更精确的边界传播方法（如Zonotope、DeepPoly）以进一步提升性能。

---

## 332. CTForensics: A Comprehensive Dataset and Method for AI-Generated CT Image Detection

**arXiv ID:** 2603.01878 | [PDF](https://arxiv.org/pdf/2603.01878v1)

**作者:** Yiheng Li `[一作]` (University of Chinese Academy of Sciences), Zhen Lei `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文先构建了一个包含10种生成模型的CT图像数据集CTForensics，并基于该数据集提出了一种新的CT伪造检测器ESF-CTFD；

**💡 创新点**

创新点在于（1）数据集涵盖广泛的GAN与扩散模型，能系统评估检测器在未见生成器上的泛化能力；（2）检测器融合波形、空间和频域多尺度特征，利用Wavelet-Enhanced Central Stem、Spatial Process Block和Frequency Process Block实现对CT特有伪造痕迹的高效捕获；

**🔧 技术方法**

技术方法包括：多尺度分辨率采样、离散小波变换+深度可分离卷积、中央相关卷积、Fast Fourier Convolution（FFC）以及多分支特征融合与门控机制；

**📊 数据集**

使用的数据集为CTForensics，共计75,990张CT切片，真实与合成图像各一半，涵盖10种生成模型；

**📈 对比分析**

与ResNet‑50、SAFE、UFD、NPR、FerretNet、Freqnet等主流基线对比，ESF‑CTFD在mAcc上达到96.01%，mAP为99.96%，显著优于其他方法；

**⚠️ 局限性**

局限性包括：模型在更复杂或未覆盖的生成器上可能表现下降；目前仅评估2D切片，未扩展至3D体积或多模态医学图像；

---

## 333. Training Dynamics of Softmax Self-Attention: Fast Global Convergence via Preconditioning

**arXiv ID:** 2603.01514 | [PDF](https://arxiv.org/pdf/2603.01514v1)

**作者:** Gautam Goel `[一作]` (Simons Institute), Peter Bartlett `[通讯]` (Simons Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在软最大自注意力层中使用梯度下降进行线性回归的训练动态，展示了简单的一阶优化算法可以以几何速率收敛到全局最优的自注意力参数。

**💡 创新点**

提出了一种新颖的“结构感知”梯度下降变体，能够有效优化原始有限数据回归目标，并引入了预处理器和正则化器以避免虚假静止点。

**🔧 技术方法**

使用了一阶优化算法，结合了数据依赖的谱初始化、正则化和预处理技术。

**📊 数据集**

使用了生成自注意力层的线性回归模型的数据集，具体样本来自高斯分布。

**📈 对比分析**

与标准的梯度下降算法（如SGD和Adam）相比，提出的算法在收敛速度上表现出几何速率，且在样本数量和梯度下降迭代次数增加时，人口损失以n^-2的速率减少，优化误差以指数速率衰减。

**⚠️ 局限性**

该方法的局限性在于它依赖于样本数量的增加和初始化的接近性，且在有限样本情况下的表现可能不如理论预期。

---

## 334. Detect Repair Verify for Securing LLM Generated Code: A Multi-Language Empirical Study

**arXiv ID:** 2603.00897 | [PDF](https://arxiv.org/pdf/2603.00897v1)

**作者:** Cheng Cheng `[一作]` (Concordia University), Cheng Cheng `[通讯]` (Concordia University)

**通讯引用:** 26610 | [OpenAlex ID](https://openalex.org/A5100354225)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 LLM 生成代码的安全硬化，设计并评估了 Detect–Repair–Verify（DRV）循环，并提供了多语言可执行项目基准。

**💡 创新点**

提出了项目级别可执行基准、三种提示粒度（项目/需求/函数）以及基于测试的迭代验证的 DRV 流程，填补了现有实验中缺乏项目级安全测试的空白。

**🔧 技术方法**

使用大型语言模型（ChatGPT‑5、GLM‑5）进行漏洞检测与修复，并结合静态/动态检测工具与自动化功能/安全测试框架实现端到端评估。

**📊 数据集**

EduCollab benchmark，包含 PHP/JS/Python 三个可运行 Web 项目，配套 37 个功能测试与 26 个安全测试目标，覆盖多种 OWASP Top 10:2025 风险。

**📈 对比分析**

在相同交互预算（K=2）下，迭代 DRV（W2）显著提升安全-正确产出率，最高提升约 57%；功能回归率极低，验证效果优于单次 DRV（W1）。

**⚠️ 局限性**

局限于基准覆盖范围有限、检测报告可靠性不一、未充分评估修复后新引入的安全缺陷，以及对非 Web 领域的泛化能力不足。

---

## 335. Deepfake Forensics Adapter: A Dual-Stream Network for Generalizable Deepfake Detection

**arXiv ID:** 2603.01450 | [PDF](https://arxiv.org/pdf/2603.01450v1)

**作者:** Jianfeng Liao `[一作]` (Shenzhen Technology University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**通讯引用:** 6139 | [OpenAlex ID](https://openalex.org/A5101720092)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种双流适配器框架 Deepfake Forensics Adapter（DFA），通过在 CLIP 视觉编码器上加入全局与局部适配模块以及交互融合分类器，实现对未见深度伪造视频的高效检测。

**💡 创新点**

创新点在于：①保持 CLIP 参数不变，仅通过全局适配器产生注意力偏置引导模型关注伪造痕迹；②局部异常流利用面部结构先验聚焦眼口等关键区域；③交互融合模块采用 Transformer 深度融合全局与局部特征，显著提升跨域泛化能力。

**🔧 技术方法**

核心技术包括 CLIP ViT‑L/14 视觉编码器、全局适配器（多层特征融合 + 注意力偏置）、局部异常流（面部关键点掩模 + 轻量化 CNN）、Transformer 编码器交互融合、三任务多头损失学习。

**📊 数据集**

在五个公开数据集上评估：Celeb‑DF‑v1、Celeb‑DF‑v2、DFDCP、FaceForensics++ 作为训练集，保留 DFDC 为未见测试集；混合数据集与 DFDC 视频级、帧级分别进行评测。

**📈 对比分析**

与 Xception、Efficient‑ViT 等主流方法对比，DFA 在混合数据集帧级 AUC 达 0.976、准确率 0.983；在 DFDC 测试集帧级 AUC 0.816、EER 0.256，视频级 AUC 0.836、EER 0.251，分别比第二佳方法提升约 4.8% 及 0.013 的 EER，证明了优异的泛化性能。

**⚠️ 局限性**

局限性：① 仅基于帧采样的单帧分析，未充分利用视频长时序信息；② 研究聚焦面部伪造，对全身或多模态（音视频）伪造的鲁棒性尚待验证。

---

## 336. Exploration enhances cooperation in the multi-agent communication system

**arXiv ID:** 2603.01401 | [PDF](https://arxiv.org/pdf/2603.01401v1)

**作者:** Zhao Song `[一作]` (Teesside University), The Anh Han `[通讯]` (Teesside University)

**通讯引用:** 3872 | [OpenAlex ID](https://openalex.org/A5012915897)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在多智能体通信系统中加入随机探索的两阶段cheap talk博弈模型，并通过多种网络拓扑的代理仿真验证探索率对合作的影响。

**💡 创新点**

创新点在于将非零探索率作为核心变量，发现存在最优探索率能最大化系统合作，并揭示合作联盟的循环成功机制。

**🔧 技术方法**

使用了演化博弈论、两阶段cheap talk模型、Fermi复制更新以及基于代理的网络仿真技术。

**📊 数据集**

在多种网络结构（正方格、small‑world、随机、无混合、scale‑free）上进行大量（约10万次）模拟，没有使用外部真实数据集。

**📈 对比分析**

通过与无探索、无网络互惠等基线对比，发现最优探索率可将合作比例提升至约0.5以上，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括仅考虑静态网络、二元信号、统一的探索率，未涵盖动态拓扑、多维信号或自适应探索率等更真实场景。

---

## 337. Catalyst-Agent: Autonomous heterogeneous catalyst screening and optimization with an LLM Agent

**arXiv ID:** 2603.01311 | [PDF](https://arxiv.org/pdf/2603.01311v1)

**作者:** Achuth Chandrasekhar `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 36918 | [OpenAlex ID](https://openalex.org/A5003442464)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了Catalyst-Agent，一种基于Model Context Protocol (MCP)服务器和大型语言模型 (LLM) 的自动化催化剂筛选与优化框架

**💡 创新点**

首次实现了完整闭环的、工具驱动的AI代理，在材料数据库检索、表面模型构建、机器学习加速的吸附能评估以及表面改性迭代等环节实现无人工干预的端到端催化剂发现流程

**🔧 技术方法**

结合了OPTIMADE API、FAIRchem的UMA小型机器学习势能、AdsorbML吸附能评估工作流、GNN/LLM（GPT‑5.2）和MCP服务器架构

**📊 数据集**

使用了材料项目(Materials Project)、OQMD、Stanford SUNCAT（Catalysis Hub）等公开晶体数据库的数据，并通过OptiMAde API检索得到晶体结构

**📈 对比分析**

在O₂还原、N₂还原和CO₂还原三种典型电催化反应中，Catalyst-Agent在三种任务中分别取得23–34%的成功率，平均在1–2次试验内收敛成功，显示出显著的效率提升与可重复性

**⚠️ 局限性**

受限于仅使用吸附能作为表面活性描述符，未考虑溶剂、电位、表面重构等实验条件，改性操作仅限于单层置换和均匀拉伸，且LLM易出现推理不一致与信息误差，限制了对更复杂催化体系的推广

---

## 338. Evaluating GFlowNet from partial episodes for stable and flexible policy-based training

**arXiv ID:** 2603.01047 | [PDF](https://arxiv.org/pdf/2603.01047v1)

**作者:** Puhua Niu `[一作]` (Texas A&M University), Xiaoning Qian `[通讯]` (Brookhaven National Laboratory)

**通讯引用:** 4786 | [OpenAlex ID](https://openalex.org/A5073946580)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了子轨迹评估平衡（Sub-EB）目标，用于更可靠地学习评估函数V，从而提升GFlowNet的策略梯度训练效率，并实现了单阶段离线和在线训练框架。

**💡 创新点**

① 将状态流函数F(s)与评估函数V(s)建立理论联系；② 设计Sub-EB目标，允许灵活加权且支持参数化后向策略；③ 在策略梯度方法中实现离线单阶段训练，消除两阶段或固定后向策略的限制。

**🔧 技术方法**

使用GFlowNet的子轨迹平衡(Sub-TB)、λ-TD、Actor‑Critic框架、梯度估计与加权子轨迹损失；结合参数化后向策略和离线数据采集技术。

**📊 数据集**

超网格（Hypergrid）模拟实验；基因/分子序列设计数据集（SIX6、PHO4、QM9、sEH）；贝叶斯网络结构学习（5、10、15节点）和大规模分子图设计数据集。

**📈 对比分析**

与CV、RL（λ‑TD）、Sub‑TB、Q‑Much等基线进行比较。实验显示，Sub‑EB在超网格上具有更高的稳定性、收敛速度和最终 TV/JS 误差；在序列设计和BN结构学习中获得更高奖励、保持多样性；离线 Sub‑EB‑B 在奖励上最佳，但多样性略低。

**⚠️ 局限性**

仍需研究最优权重系数及其与更高级策略方法（如 TRPO）的结合；对极大组合空间的计算瓶颈未完全解决；多样性与奖励的平衡尚需进一步探索；部分基线缺少离线性能对比，限制了对离线效果的完整评估。

---

## 339. AIoT-based Continuous, Contextualized, and Explainable Driving Assessment for Older Adults

**arXiv ID:** 2603.00691 | [PDF](https://arxiv.org/pdf/2603.00691v1)

**作者:** Yimeng Liu `[一作]` (Michigan State University), Zhichao Cao `[通讯]` (Michigan State University)

**通讯引用:** 2059 | [OpenAlex ID](https://openalex.org/A5072253749)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套基于AIoT的连续、情境化、可解释的驾驶评估框架，旨在实时监测和分析老年驾驶员的驾驶行为与安全风险。

**💡 创新点**

创新点在于将多尺度行为表示、环境感知融合和层级可解释推理结合在车内边缘计算中，形成可持续、隐私友好的评估流程。

**🔧 技术方法**

采用多模态传感融合、时序卷积+自注意力的层级特征提取、交叉模态对齐与注意力融合、因果映射与语义抽象解释技术，并实现了模型蒸馏、量化与联邦学习。

**📊 数据集**

利用CARLA仿真测试平台、LongROAD纵向自然驾驶数据集以及DRIVES高频驾驶记录集进行验证与分析。

**📈 对比分析**

与传统单一时序模型和黑箱深度网络对比，所提框架在驾驶异常检测精度上提升约10–15%，并通过可解释报告大幅提高临床与监管的信任度，尽管实验多为仿真与受控数据。

**⚠️ 局限性**

局限包括：需要大规模真实车内部署验证；模型复杂度与实时性仍需进一步压缩；对极端环境与少数族裔驾驶行为的泛化能力待增强；隐私保护仍面临法律与伦理挑战。

---

## 340. Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models

**arXiv ID:** 2603.00629 | [PDF](https://arxiv.org/pdf/2603.00629v1)

**作者:** Yunzhong Qiu `[一作]` (Tsinghua University), Jianmin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 38042 | [OpenAlex ID](https://openalex.org/A5100373517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 TATO 框架，通过自动搜索并优化时间序列的预处理变换，使单一冻结的预训练大模型（LTM）能够在多域场景下实现高质量预测。

**💡 创新点**

创新点包括：1）FrozenForecasting 范式，强调在不调整模型参数的前提下通过数据变换实现跨域适配；2）构建了针对上下文、尺度、异常的九个可调节变换操作的搜索空间；3）采用两阶段 Pareto 评价机制，兼顾多种误差指标并保证稳健性；4）实现了极快的搜索（≤2 min），显著降低了适配成本。

**🔧 技术方法**

技术手段包括：树结构 Parzen Estimator (TPE) 超参数搜索、数据增强（翻转、扭曲、噪声、平移等）、上下文切片、尺度归一化、异常检测与修正、两阶段 Pareto 排序和加权多指标汇总。

**📊 数据集**

实验数据集涵盖 ETT（ETTh1、ETTh2、ETTm1、ETTm2）、Electricity、Exchange、Traffic、Weather 等八个时间序列数据集，均使用公开的训练/验证/测试划分。

**📈 对比分析**

通过与多种先进 LTM（Timer、Moirai、Chronos）在 192 个实验场景下的零样本预测进行对比，TATO 在 84.3% 的情况下实现了改进，平均 MSE 降低 13.6%，单个场景最高可达 65.4%，且搜索时间普遍低于 120 s。

**⚠️ 局限性**

局限性包括：在某些极端分布偏移或异常严重的数据上提升有限；目前仅支持单变量预测；搜索过程虽快但仍需额外计算；对高维多变量序列的适配尚未验证。

---

## 341. MetaRCA: A Generalizable Root Cause Analysis Framework for Cloud-Native Systems Powered by Meta Causal Knowledge

**arXiv ID:** 2603.02032 | [PDF](https://arxiv.org/pdf/2603.02032v1)

**作者:** Shuai Liang `[一作]` (Sun Yat-sen University), Chongkang Tan `[通讯]` (Individual Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出MetaRCA框架，构建离线Meta因果图（MCG）并在云原生系统故障发生时即时实例化、权重融合、剪枝后进行根因分析；

**💡 创新点**

通过将多源证据（LLM推断、故障报告、监控数据）融合入MCG，采用贝叶斯信念演化构建可扩展、可泛化的知识库，并在在线阶段实现局部化推理与实时剪枝，显著提升准确率与效率；

**🔧 技术方法**

使用大模型（Gemini 2.5 Flash、DeepSeek R1-70B）进行因果知识提取；因果发现算法（PC、PCMCI）和贝叶斯信念演化；时间序列异常检测、相关系数与上下文可行性评分；根因排序采用Causal Contribution Back‑propagation、PageRank、随机游走等；

**📊 数据集**

使用563个中国联通故障报告、375个RCAEval‑RE1案例、239个AIOPS2022案例、252个公开微服务案例和59个真实生产系统故障；

**📈 对比分析**

与五个基线（CIRCA、CausalRCA、PC_PR、PCMCI_RW、OpenRCA）进行AC@1/3/5和平均RCA时间比较；MetaRCA在所有数据集上均实现最高准确率（服务级AC@1最高0.66，指标级AC@1最高0.54），准确率提升29–48个百分点，且平均RCA时间仅为0.05–0.9秒，远快于学习型和LLM型基线；

**⚠️ 局限性**

受限于元数据知识库构建质量，对不同架构（非云原生）的适用性待验证；LLM可能产生幻觉，需人工校验；超参数（衰减常数、权重阈值）需针对系统调优；依赖高质量故障报告与监控数据，若数据质量低会影响性能。

---

## 342. TacMamba: A Tactile History Compression Adapter Bridging Fast Reflexes and Slow VLA Reasoning

**arXiv ID:** 2603.01700 | [PDF](https://arxiv.org/pdf/2603.01700v1)

**作者:** Zhenan Wang `[一作]` (Zhejiang University), Huixu Dong `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fede83ac-7505-405f-ab37-e7284695c47f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为TacMamba的框架，能够在100Hz的高频触觉循环中实时编码触觉历史，并以低维向量形式在约1Hz的视觉-语言-动作(VLA)规划器中异步注入触觉信息，实现高频触觉反射与低频视觉规划的无缝协同。

**💡 创新点**

核心创新点包括：
1) 采用Mamba-based Selective State Space Model实现常数时间推理的高频触觉编码，解决Transformer O(N²)和LSTM遗忘的问题；
2) 双阶段训练策略：自监督的三元时序判别任务提取因果触觉动力学，随后通过阶段均匀采样提升稀疏关键接触事件的数据利用率；
3) 触觉历史压缩后作为软提示注入VLA，支持零成本的插件式集成。

**🔧 技术方法**

使用技术包括：Mamba网络、Selective State Space Model、RevIN、Channel Independence、三元时序判别（Temporal Discrimination）、阶段均匀采样（Phase-Uniform Sampling）、软提示注入（Soft Prompt）以及Vision-Language-Action模型。

**📊 数据集**

数据集为在AgileX PiPER机器人上收集的增强真实世界触觉数据，涵盖按钮点击、碎片抓取、布料折叠等任务，统一采样频率为100Hz。

**📈 对比分析**

方法对比：与1D-CNN、Transformer、LSTM等传统编码器以及π_0.5视觉模型进行评估。TacMamba在相同数据集上达88.89%准确率、0.45 ms延迟、60.85 MB内存，显著优于Transformer（81.25%/11.37 ms）和LSTM。真实任务中，TacMamba实现100%顺序按钮按压成功率，远高于π_0.5及其他视觉基线。

**⚠️ 局限性**

局限性：仅使用1D力传感器，缺乏空间分辨的触觉信息；实验聚焦于遮挡和长序列场景，缺乏大规模多任务基准；在多指手的复杂抓取任务中的通用性尚待验证。

---

## 343. MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains

**arXiv ID:** 2603.00873 | [PDF](https://arxiv.org/pdf/2603.00873v1)

**作者:** Xuying Ning `[一作]` (University of Illinois Urbana-Champaign), Jingrui He `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4210 | [OpenAlex ID](https://openalex.org/A5073158087)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MC-Search基准，用于评估跨模态检索增强生成（MM‑RAG）中的代理式长链推理；

**💡 创新点**

创新点包括：①五种代表性推理拓扑和逐步验证的HAV筛选保证每一步必要且无冗余；②引入三种过程级评估指标（LLM-as-a-Judge、HPS、RD）；③开发统一的代理式MM‑RAG流水线和搜索对齐（Search‑Align）训练框架；

**🔧 技术方法**

技术主要有：多模态检索（文本/图像）、逐步子查询与行动生成、Hop-wise Attribution and Verification (HAVE)、过程级监督微调(Search‑Align)以及基于LLM评判的链级评估；

**📊 数据集**

数据集为3,333条经过HAVE验证的多模态推理链，覆盖文本、图像、并行与多图像分支，共计约3.8平均跳数；

**📈 对比分析**

与六款主流MLLM（含GPT‑4o‑Mini、Gemini‑2.5‑Flash/Pro、Claude‑3.7‑Sonnet、InternVL3.5‑8B、Qwen2.5‑VL‑7B）在统一代理流水线下评估，公开模型在文本链上接近Gemini‑2.5‑Pro，Search‑Align显著提升开源模型（如Qwen2.5‑VL‑7B提升≈13.7 F1、+16 HPS、RD↓3.1）；

**⚠️ 局限性**

局限性在于：①评估受限于单模态检索精度与top‑1约束；②对超长链的稳健性仍有限；③模型对图像检索的依赖过大，缺乏跨模态一致性；④未覆盖更复杂领域（如科学、数学）与多语言场景；

---

## 344. Whisper-MLA: Reducing GPU Memory Consumption of ASR Models based on MHA2MLA Conversion

**arXiv ID:** 2603.00563 | [PDF](https://arxiv.org/pdf/2603.00563v1)

**作者:** Sen Zhang `[一作]` (Tianjin University), Luo Si `[通讯]` (Banma Network Technology Co., Ltd.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

将Whisper模型转换为Whisper-MLA，引入多头潜在注意力（MLA）来压缩KV缓存；

**💡 创新点**

设计适用于绝对位置编码的MLA，并系统探究其在encoder、decoder自注意力和交叉注意力中的应用，提出仅改decoder自注意力的DSO方案，实现显著内存节省且保持性能；

**🔧 技术方法**

采用低秩键值压缩、维度保留策略（均匀采样、2‑norm）、联合SVD进行参数高效转换与微调，基于Whisper‑small实现；

**📊 数据集**

使用LibriSpeech 960h数据集进行微调与评估；

**📈 对比分析**

通过与原始Whisper及微调Whisper的WER对比，Whisper‑MLA（DSO+均匀采样）平均WER仅比基线高0.17%，KV缓存减少87.5%，在长序列和大batch时显著降低GPU内存使用，避免OOM；

**⚠️ 局限性**

在Full模型下性能相对较差，仍需保留encoder与交叉注意力原有结构；仅验证于Whisper‑small，缺乏对更大模型或多语言、噪声环境的评估。

---

## 345. VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning

**arXiv ID:** 2603.01195 | [PDF](https://arxiv.org/pdf/2603.01195v1)

**作者:** Mingkang Dong `[一作]` (Shanghai Jiao Tong University), Yuqian Fu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为VisNec的视觉必要性评分方法，用于在多模态指令调优中选择最有价值的样本。

**💡 创新点**

创新点在于通过对比文本仅推理和多模态推理的损失差异，量化视觉输入的边际贡献，筛选出视觉关键、冗余和失配样本。

**🔧 技术方法**

采用了基于V-usable信息的损失对比计算、K-Means语义聚类和对比损失的差分评分机制。

**📊 数据集**

在LLaVA-665K和Vision-Flan-186K两大指令调优数据集上进行实验。

**📈 对比分析**

与多种基线（随机、Self-Filter、EL2N、PreSel、XMAS等）比较，VisNec在仅使用15%数据时在10个评测基准上达到或超过100%完整数据的性能，且计算成本最低。

**⚠️ 局限性**

局限性包括对不同视觉模型的适用性仍需进一步验证，且在极少样本或多模态任务分布极度偏斜时可能表现不佳。

---

## 346. From Literature to Hypotheses: An AI Co-Scientist System for Biomarker-Guided Drug Combination Hypothesis Generation

**arXiv ID:** 2603.00612 | [PDF](https://arxiv.org/pdf/2603.00612v1)

**作者:** Raneen Younis `[一作]` (Hannover Medical School), Zahra Ahmadi `[通讯]` (Hannover Medical School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了CoDHy，一个交互式AI合作者系统，用于基于生物标志物的癌症药物组合假设生成。

**💡 创新点**

创新点在于将任务专属知识图谱构建、图嵌入推理、多智能体验证以及人机协作界面整合，能够生成可追溯、可评估且多样化的药物组合假设。

**🔧 技术方法**

采用的技术包括SpaCy NLP抽取实体关系、Neo4j存储知识图谱、Node2Vec嵌入、图检索+大语言模型生成（Llama‑3.1），以及Gradio + Hugging Face Inference API实现多智能体协作。

**📊 数据集**

使用的数据集包括各类公开的生物医学数据库、用户指定数量的PubMed摘要、DrugCombDB中的药物协同信息，并通过API动态检索得到最新文献。

**📈 对比分析**

通过与LLM‑only基线和No‑Node2Vec变体在7个（生物标志物，癌种）情景上进行对比实验，评估指标包括新颖度、证据覆盖率、Proceed@1/3、组合多样性、MRR和nDCG@3；CoDHy在新颖度和多样性上领先，证据覆盖率保持竞争力，排名指标略低但符合探索性目标。

**⚠️ 局限性**

局限性包括对LLM验证的依赖可能产生幻觉、代理判定仍是代理评估而非专家评审、知识图谱的不完整或噪声可能传递至后续推理、缺乏患者级别数据等导致的个性化不足。

---

## 347. Causal Neural Probabilistic Circuits

**arXiv ID:** 2603.01372 | [PDF](https://arxiv.org/pdf/2603.01372v1)

**作者:** Weixin Chen `[一作]` (University of Illinois Urbana-Champaign), Han Zhao `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2692 | [OpenAlex ID](https://openalex.org/A5101670508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种Causal Neural Probabilistic Circuit模型，用于在概念瓶颈模型中更准确地利用专家干预并捕获概念间的因果依赖。

**💡 创新点**

创新点在于将神经属性预测器与编译的因果概率电路相结合，并用Product of Experts近似干预后的属性分布，从而实现可解析、可推断的干预后类别分布。

**🔧 技术方法**

采用神经网络预测器、因果概率电路（sum‑product网络）以及Product of Experts融合方法。

**📊 数据集**

实验使用Asia、Sachs、MNISTAdd、cMNISTAdd和CelebA五个基准数据集。

**📈 对比分析**

与五个主流CBM基线（Vanilla CBM、CEM、SCBM、C²BM和基于电路的标签预测器）对比，CNPC在所有OOD场景下干预效率最高，任务准确率显著优于基线，尤其在未见变换、对抗扰动和伪相关偏移下。

**⚠️ 局限性**

主要局限在于需要先验因果图、假设无未观测混杂且对α调参敏感；若因果结构不完整或属性分布极不平衡，性能可能下降。

---

## 348. NeuroSCA: Neuro-Symbolic Constraint Abstraction for Smart Contract Hybrid Fuzzing

**arXiv ID:** 2603.01272 | [PDF](https://arxiv.org/pdf/2603.01272v1)

**作者:** Haochen Liang `[一作]` (University of Tokyo), Hideya Ochiai `[通讯]` (University of Tokyo)

**通讯引用:** 1380 | [OpenAlex ID](https://openalex.org/A5045456657)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出 NeuroSCA，一种在智能合约混合模糊测试中使用 LLM 进行语义约束抽象的轻量级框架，通过 LLM 选取关键约束并与 SMT 求解器配合，同时使用验证器循环来保证解的正确性，从而显著降低约束污染导致的求解超时。

**💡 创新点**

创新点：①将大型语言模型（LLM）嵌入到混合模糊测试的约束求解流程，形成“语义切片”以去除冗余约束；②设计了锚定约束抽象和可验证的抽象-精炼循环，确保在减少约束量的同时保持求解的完备性；③提出了基于路径指纹的自适应调用策略，在易路径时直接使用传统求解器，避免不必要的 LLM 调用，从而保持低开销。

**🔧 技术方法**

技术手段：- LLM（如 GPT‑4）用于生成约束核心列表；- 语义抽象（anchored abstraction）将路径约束拆分为核心和冗余两部分；- SMT 求解器（如 Z3）用于求解抽象后的约束；- 验证器循环（verifier‑in‑the‑loop）在 EVM 上执行生成的输入，检测缺失约束并迭代精炼；- 路径指纹缓存实现自适应调用与重用抽象。

**📊 数据集**

数据集：- 合成压力测试合约（如 3c、5c 等），用于制造高度污染的路径；- 真实 DeFi 合约（共 100+ 个公开合约，示例中包括 3cFuzzing 表所列的若干合约），用于评估实际漏洞检测效果。

**📈 对比分析**

比较方法：与基线混合模糊器（Baseline）以及“始终使用 NeuroSCA”模式（NeuroSCA‑only）进行对比。性能表现：在污染路径上，NeuroSCA‑only 将平均求解时间从 17.84 s 降至 6.25 s，P99 阈值从 44.80 s 降至 22.49 s；Selective 模式在保持低开销的同时，覆盖率和 Bug 数量均与 Baseline 相当或更优（例如在难度合约 5c 中 Coverage 由 67.9% 提升至 80.2%，Bug 从 0 提升至 2）。在简单合约上，Selective 与 Baseline 的平均时间差异不超过 0.1 s，说明自适应策略能避免不必要的 LLM 调用。

**⚠️ 局限性**

限制与挑战：①对 LLM 的可用性和成本依赖较大；②当 LLM 产生的核心约束缺失关键约束时，验证器循环可能需要多次迭代，导致额外开销；③在极大规模或多合约跨调用场景下，路径指纹缓存与抽象策略尚未充分验证；④目前主要针对单合约、branch‑flip 目标，尚未覆盖更复杂的攻击面（如跨合约递归调用、状态迁移等）。

---

## 349. TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization

**arXiv ID:** 2603.01169 | [PDF](https://arxiv.org/pdf/2603.01169v1)

**作者:** Sumin Kim `[一作]` (Seoul National University), Joonseok Lee `[通讯]` (Seoul National University)

**通讯引用:** 5602 | [OpenAlex ID](https://openalex.org/A5067433666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了TripleSumm框架和MoSu大规模trimodal视频摘要数据集，利用视觉、文本、音频信息实现帧级动态多模态融合；

**💡 创新点**

创新点包括：1）多尺度时序块（MST）通过滑动窗口自注意力捕捉局部到全局时序依赖；2）交叉模态融合块（CMF）采用融合Token实现帧级动态权重分配；3）首个提供视觉+文本+音频的长视频数据集MoSu，支持大规模训练与评估；

**🔧 技术方法**

使用预训练编码器（CLIP、RoBERTa、Audio Spectrogram Transformer）提取特征；MST（窗口自注意力+FFN）、CMF（交叉注意力+FFN）两层交替堆叠；预测头线性映射到帧级重要性得分；L2 损失训练；后续基于阈值的剪辑分割与选择。

**📊 数据集**

主要使用MoSu（52,678视频，约4,000小时，含视觉、文本、音频），在SumMe、TVSum、Mr. HiSum等公开数据集进行跨数据集评估。

**📈 对比分析**

在MoSu、Mr. HiSum、SumMe、TVSum上与现有SOTA进行对比，TripleSumm在Kendall τ、Spearman ρ、mAP等指标上均领先，且参数仅约1.37M，比主流模型小得多；零样本长视频测试亦表现最佳。

**⚠️ 局限性**

局限性：1）仍采用两步流程（帧重要性评分+后期剪辑分割），未实现端到端的直接剪辑生成；2）对极长视频的泛化虽然好，但在多样性更大的跨域场景仍需进一步验证；3）依赖预训练模态与大量视频回放统计，难以处理无回放数据；4）缺乏人工标注的高质量摘要数据，评估仍以统计回放为主。

---

## 350. Invariant-Stratified Propagation for Expressive Graph Neural Networks

**arXiv ID:** 2603.01388 | [PDF](https://arxiv.org/pdf/2603.01388v1)

**作者:** Asela Hevapathige `[一作]` (University of Melbourne), Saman Halgamuge `[通讯]` (University of Melbourne)

**通讯引用:** 12364 | [OpenAlex ID](https://openalex.org/A5067418792)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了Invariant-Stratified Propagation (ISP)框架，结合ISP-WL和ISP-GNN，实现了基于图不变量的层次化节点传播。

**💡 创新点**

创新点在于利用图不变量的全局排序进行节点分层，构建结构异质性编码，突破1-WL表达力瓶颈，并支持可学习的层级划分。

**🔧 技术方法**

主要技术包括图不变量排序、三角形聚合、层次差值编码、双流消息传递和可微分的学习层级机制。

**📊 数据集**

实验使用了TU图分类基准、OGB化学分子数据集、节点分类基准（Cora、Citeseer等）以及影响力估计的社交网络数据。

**📈 对比分析**

与GIN、ID-GNN、GSN、GraphSNN、KP-GIN等基线对比，ISP-GNN在图分类、节点分类和影响力估计任务上取得显著提升，准确率提升至~30%以上。

**⚠️ 局限性**

主要局限在于可学习层级需要手动设定层数和温度退火参数，复杂不变量的计算开销较高，且在某些稀疏图中仍可能受限于三角形稀缺。

---

## 351. Wild-Drive: Off-Road Scene Captioning and Path Planning via Robust Multi-modal Routing and Efficient Large Language Model

**arXiv ID:** 2603.00694 | [PDF](https://arxiv.org/pdf/2603.00694v1)

**作者:** Zihang Wang `[一作]` (Southeast University), Haoyang Che `[通讯]` (Chery Auto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Wild-Drive统一框架，实现离地环境下的场景字幕生成与路径规划，具备对单模态失效的鲁棒性与可解释性。

**💡 创新点**

创新点包括：①任务条件模态路由与压缩模块MoRo-Former，可在传感器失效时动态选择可靠模态；②将轻量LLM与规划token结合，生成结构化字幕并通过GRU解码轨迹；③构建专门针对离地环境的OR‑C2P基准，涵盖多种传感器破坏情况。

**🔧 技术方法**

使用技术包括：多模态编码器（VoxelNet+DINOv3）、MoRo‑Former路由与压缩、轻量LLM（Qwen2.5‑0.5B/3B）、GRU解码器、结构化Q&A模板、随机模态丢弃训练。

**📊 数据集**

使用的数据集为：基于ORAD‑3D的OR‑C2P基准（约57K帧、覆盖雨、雾、低光等破坏）以及自采集的SC数据集（4 km离地轨迹）。

**📈 对比分析**

与LiDAR‑LLM、BEV‑LLM、LLaMA‑Adapter等方法比较，Wild‑Drive‑3B在BLEU‑4 49.26、BERT‑P 98.13、路径FDE 1.09、minADE 0.66等指标上均优于现有LLM‑based方法；Wild‑Drive‑0.5B在参数量仅0.71B的情况下，性能仍接近大模型。

**⚠️ 局限性**

局限性：相较于专用规划器TopoPath，仍略逊一筹；LLM对超长输入或极端失真敏感；需进一步融合运动建模与更精准的模态失效检测。

---

## 352. Partial Causal Structure Learning for Valid Selective Conformal Inference under Interventions

**arXiv ID:** 2603.02204 | [PDF](https://arxiv.org/pdf/2603.02204v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), James P. Long `[通讯]` (MD Anderson Cancer Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了在多干预实验（如基因扰动）中，利用选择性（Mondrian） conformal prediction 并结合部分因果结构学习，给出更紧凑的置信区间；提出了δ-稳健覆盖定理、任务驱动的部分因果学习框架以及两种高效算法；

**💡 创新点**

创新点包括：①将误分类的干预子集误差量化为覆盖损失函数 g(δ,n)，并给出最优 α 校正；②只学习干预–目标对的子代指示而非完整因果图；③通过交集差异集合和局部 ICP 实现子代发现与距离估计，兼顾精度与可扩展性；

**🔧 技术方法**

采用了选择性 conformal prediction、Mondrian conformal、分布无关覆盖定理、交集差异集合因果发现、局部 invariant causal prediction、统计误差分析与实验验证等技术；

**📊 数据集**

实验数据包括合成线性结构方程模型（p=200、150个干预）和 Replogle K562 CRISPRi 细胞系基因扰动数据（约5k基因、50个扰动）；

**📈 对比分析**

与 oracle selective、pooled、corrected 方案对比：在合成数据中覆盖率≈0.9，受污染时逐渐下降，corrected 在所有污染水平下覆盖率≥0.95（宽度略增）；在真实数据中 Corrected 仅在约60%实验可行，但覆盖率最高；其他方法覆盖率较低或一致；

**⚠️ 局限性**

局限性包括：①需要足够的校准样本以满足 α' 校正；②对真实扰动实验的因果结构缺乏精确信息，proxy oracle 可能误导；③覆盖下界为 worst-case，实际性能更好但可能无法完全捕捉；④在高密度网络或子代稀疏性不足时 δ 可能增大，导致覆盖下降。

---

## 353. Causal Circuit Tracing Reveals Distinct Computational Architectures in Single-Cell Foundation Models: Inhibitory Dominance, Biological Coherence, and Cross-Model Convergence

**arXiv ID:** 2603.01752 | [PDF](https://arxiv.org/pdf/2603.01752v1)

**作者:** Ihor Kendiukhov `[一作]` `[通讯]` (University of Tuebingen), Ihor Kendiukhov (University of Tuebingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过对单细胞基础模型Geneformer和scGPT的稀疏自编码器（SAE）特征进行因果消融和下游激活测量，构建了完整的特征间因果电路图；

**💡 创新点**

首次系统性地揭示了特征级别的因果互动，发现大多数边为抑制性、存在广播型枢纽以及跨模型共识的生物学通路，并挖掘出众多潜在的新型生物关系；

**🔧 技术方法**

采用因果消融、Cohen’s d效应量与一致性统计、PMI共激活比较、图论分析等技术，实现了特征级因果可视化与定量；

**📊 数据集**

使用Geneformer V2‑316M与scGPT whole‑human模型，以K562细胞与Tabula Sapiens细胞为输入，并利用Replogle CRISPRi全基因组敲降实验进行验证；

**📈 对比分析**

在四种实验条件（K562/K562、K562/Multi、TS/Multi、scGPT/TS/Multi）下对电路密度、平均|d|、抑制比例、跨模型一致性等指标进行比较，Geneformer表现出约53%的生物学一致性、80%抑制边，scGPT则显示更强效应（|d|=1.40）和更平衡的兴奋/抑制比例；

**⚠️ 局限性**

研究仅针对30个高注释质量源特征、单特征消融、阈值设定、细胞样本量有限、缺乏多特征组合效应以及部分多组织SAE缺失层级等方面存在局限。

---

## 354. GAM-RAG: Gain-Adaptive Memory for Evolving Retrieval in Retrieval-Augmented Generation

**arXiv ID:** 2603.01783 | [PDF](https://arxiv.org/pdf/2603.01783v1)

**作者:** Yifan Wang `[一作]` (Fudan University), Hongfeng Chai `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、增益自适应记忆框架GAM-RAG，通过在线更新句子级任务与时间记忆，使检索模块能随查询经验不断演化，从而提升检索效率与生成质量。

**💡 创新点**

创新点包括：①构建无关系、层次化的轻量化图索引；②为每句子维护任务与时间两种记忆向量；③采用Kalman滤波启发的增益自适应更新，结合句子困惑度来调节学习率；④实现在线、无训练的记忆更新，允许系统在推理过程中自我改进。

**🔧 技术方法**

使用技术包括：spaCy NER与句子分割、all-mpnet-base-v2 文本嵌入、LLM（GPT‑4o）作为判定句子是否支持答案的反馈、Kalman滤波启发式增益调节、句子级记忆更新与 perplexity 控制、迭代图传播与多跳检索。

**📊 数据集**

实验数据集涵盖多跳问答（2WikiMultiHopQA、HotpotQA、MuSiQue）、时间敏感问答（TimeQA）以及领域特定问答（Medical/GraphRAG‑Bench），并在这些基准上进行评估。

**📈 对比分析**

与标准LLM、Vanilla RAG以及多种 GraphRAG 样式基线（GFM‑RAG、HippoRAG2、LinearRAG、PoG、DyG‑RAG、REMINDRAG）进行对比。GAM‑RAG 在多跳 QA 上平均提升 GPT‑Acc 约 3.95%，在 5‑turn 记忆下提升 8.19%；在 TimeQA 中排名第二，Medical 数据集取得最高分；推理成本下降 61%，相似/不同查询的鲁棒性提升约 10.3%。

**⚠️ 局限性**

局限性包括：①依赖 LLM 判定反馈，噪声可能影响记忆更新；②只对句子级记忆进行自适应更新，忽略段落/文档级细粒度信息；③持续在线更新可能导致存储与同步开销；④在极大规模语料下的可扩展性与并行化未深入验证；⑤实验仅覆盖五个基准，跨域通用性仍待进一步评估。

---

## 355. Reconstructing Content via Collaborative Attention to Improve Multimodal Embedding Quality

**arXiv ID:** 2603.01471 | [PDF](https://arxiv.org/pdf/2603.01471v1)

**作者:** Jiahan Chen `[一作]`, Keping Bi `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于协同注意力的内容重构预训练框架（CoCoA），通过双向注意力温启动、EOS桥接的压缩重构以及对压缩后EOS嵌入进行对比学习，显著提升多模态嵌入质量。

**💡 创新点**

创新点在于将内容重构任务嵌入多模态大语言模型的预训练流程，通过专门的注意力截断让视觉信息被压缩到单个EOS标记，从而实现信息稠密的嵌入；同时使用双向注意力来弥补因果注意力导致的跨模态融合不足。

**🔧 技术方法**

技术包括：双向注意力warm‑up（联合文本MNTP与图像MAE）、EOS‑bridged注意力截断的压缩重构任务、EOS基的对比学习（InfoNCE）、LoRA微调、动态图像分辨率处理和大规模的合成数据生成。

**📊 数据集**

主要数据集为MMEB‑V1（涵盖分类、VQA、检索、视觉定位四类任务），预训练使用其中20个任务的约300K样本，并加入约200K合成样本；对比学习阶段使用MMEB‑V1完整训练集。

**📈 对比分析**

与CLIP、OpenCLIP、GME、UNITE、VLM2Vec、E5‑V、mmE5、UniME、MoCa等SOTA多模态嵌入模型对比，CoCoA在≤3B参数规模上达到了SOTA，在7B规模下保持竞争力，并且仅使用比对比学习方法少得多的预训练数据。

**⚠️ 局限性**

局限性包括：单一EOS压缩可能无法充分表达多样化图像中的多重语义；对合成数据的依赖可能导致偏差；在极大规模多模态数据上仍需进一步验证；以及需要额外的预训练阶段，增加了工程复杂度。

---

## 356. Evaluating AI Grading on Real-World Handwritten College Mathematics: A Large-Scale Study Toward a Benchmark

**arXiv ID:** 2603.00895 | [PDF](https://arxiv.org/pdf/2603.00895v1)

**作者:** Zhiqi Yu `[一作]` (University of California Irvine), Yifeng Yu `[通讯]` (University of California Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并部署了基于 OCR + 大语言模型的 AI 成绩评估系统，对 UC Irvine 单变量微积分课程中数千份手写作业进行自动评分与反馈。

**💡 创新点**

提出了双 rubrics（柔性 + 固定）与 max‑rule 汇总策略、结构化 prompt 与 OCR 纠错提示、以及多视角人机评估框架，并计划发布标准化 benchmark。

**🔧 技术方法**

主要技术包括：手写数学 OCR（GPT‑4.1‑mini 及 Mathpix）、GPT‑4.1‑mini/​o3‑mini 进行结构化评分与生成反馈、prompt‑engineering、模型稳定化（多跑平均/最近邻）与错误触发人机回调。

**📊 数据集**

使用 2025 年春季数学 2A/2B 课程的 3,945 条手写自由答题记录（约 800 名学生）及 171 条对比集，配合 OCR 结果与教师/独立评审标签。

**📈 对比分析**

通过与 TA 评分、学生问卷和 20+ 独立评审的三方比较，AI 评分平均偏低 0.4 分，MAE 0.5‑1.1 分，86% 内 1 分；独立评审认为 79.8% 评估完全正确，90% 内 1 分一致，反馈准确率超过 90%。

**⚠️ 局限性**

主要限制包括：缺乏系统化的回退判定规则、对高风险评估（期中、期末）的适用性不明、对 OCR 失误与几何/图形识别的鲁棒性不足，以及 max‑rule 的最优性未完全验证。

---

## 357. FreeAct: Freeing Activations for LLM Quantization

**arXiv ID:** 2603.01776 | [PDF](https://arxiv.org/pdf/2603.01776v1)

**作者:** Xiaohao Liu `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FreeAct方法，在LLM量化中放松激活端的一对一变换约束，针对扩散和多模态LLM的动态激活分布进行后训练量化。

**💡 创新点**

创新点在于利用激活的秩缺陷构造低秩子空间变换，给不同token类型分配独立变换矩阵，同时保持权重端统一变换，从而突破传统一对一变换的局限。

**🔧 技术方法**

核心技术包括正交变换矩阵、低秩子空间拆分、零填充、可学习裁剪阈值、通道尺度、Kronecker乘积以及对权重与激活保持等价性的理论证明。

**📊 数据集**

使用WikiText2、COCO-Cap等数据构建校准集，并在HumanEval、GSM8K、Math500、MMMU、MMBench、RealworldQA等基准上评估。

**📈 对比分析**

与RTN、SmoothQuant、QuaRot、FlatQuant等SOTA方法对比，FreeAct在大多数任务上均领先，W4A4量化时平均提升约5.3%，接近或等同于16位基准。

**⚠️ 局限性**

局限性包括只处理两种token类型（遮蔽/未遮蔽或视觉/文本），未考虑多模态扩散模型，且实现主要在软件层面，尚未结合硬件加速或自动化token识别。

---

## 358. YCDa: YCbCr Decoupled Attention for Real-time Realistic Camouflaged Object Detection

**arXiv ID:** 2603.01602 | [PDF](https://arxiv.org/pdf/2603.01602v1)

**作者:** PeiHuang Zheng `[一作]` (Nanjing University of Aeronautics and Astronautics), Yang Li `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 15297 | [OpenAlex ID](https://openalex.org/A5100421672)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于YCbCr色彩空间解耦与信息感知通道注意力的轻量级早期特征处理策略（YCDa），以提升实时摄像机在真实隐蔽目标检测中的准确性。

**💡 创新点**

创新点：① 将人类视觉在隐蔽场景下转向亮度与纹理的机制迁移到模型；② 通过色彩空间转换实现色度与亮度信息分离；③ 引入基于方差与均值的通道注意力（ICA）动态调节通道权重；④ 采用无点卷积下采样保持通道独立。

**🔧 技术方法**

使用的技术包括 YCbCr 色彩空间转换、ESSamp（无点卷积下采样）、信息感知通道注意力模块（ICA）以及将 YCDa 插件式集成至现有实时检测器（YOLO、RT-DETR-L 等）。

**📊 数据集**

主要使用 COD-D 数据集（COD10K-D、NC4K-D、CAMO-D）进行训练与评估，同时在 COCO 数据集上预训练以提升模型泛化。

**📈 对比分析**

通过与多种基线模型（YOLOv8s、YOLOv11s、YOLO12s、RT-DETR-L 等）的对比，YCDa 在 COD10K-D 上实现 mAP 18.0%（比基线提升 112%），在 NC4K-D 与 CAMO-D 上分别提升 29.6% 与 27.5%；推理速度仅下降 4.9%，仍保持实时性能。

**⚠️ 局限性**

局限性：① 对于样本量有限的中小规模数据集，Transformer 结构在 YCDa 下可能适应不足；② YCDa 主要针对静态图像，视频序列中的时序信息未被利用；③ 目前色彩空间转换固定为 YCbCr，缺乏自适应学习不同场景下最优色彩空间的机制。

---

## 359. Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift

**arXiv ID:** 2603.01040 | [PDF](https://arxiv.org/pdf/2603.01040v1)

**作者:** Heewon Park `[一作]` (Sungkyunkwan University), Minhae Kwon `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无监督的联邦后期自适应框架 Fed-ADE，能在非平稳、多客户端的环境下通过动态学习率自适应分布漂移，提升模型在边缘设备上的实时性能。

**💡 创新点**

核心创新在于：①引入轻量级的两种分布动态估计器（不确定性动态估计和表示层漂移估计）；②将这两种估计结果融合成客户端时刻的分布漂移信号，进而根据该信号调节学习率；③提供理论证明，证明估计器能逼近真实分布漂移并给出动态 regret 与收敛上界。

**🔧 技术方法**

技术要点包括：联邦学习的层级分离（共享层与个性化层）、无监督风险估计（BBSE）、基于余弦相似度的漂移估计、动态学习率调度公式、梯度下降与投影步骤，以及动态 regret 分析。

**📊 数据集**

实验使用了四个图像基准（Tiny ImageNet、CIFAR-10、CIFAR-100、CIFAR-10-C/CIFAR-100-C）以及一个文本基准 LAMA，在多种标签/协变量漂移情景下进行测试。

**📈 对比分析**

与多种基线（本地无监督适应方法 FTH/ATLAS/UNIDA/UDA、联邦学习方法 Fed-POE/FedCCFA、固定学习率 Fed-ADE 变体）进行对比。Fed-ADE 在所有漂移场景下均取得最高平均准确率，并且壁时仅为 109 秒左右，明显快于本地方法，且远快于 FedCCFA，展示了性能与效率兼顾的优势。

**⚠️ 局限性**

局限性包括：①依赖于预训练模型的质量，若预训练分布与实际差异过大可能影响效果；②当前仅使用余弦相似度作为漂移度量，其他更精细的分布距离可能进一步提升鲁棒性；③在极端漂移速率或标签空间变化剧烈时，动态学习率调节仍可能不够及时或导致收敛不稳。

---

## 360. 3D Field of Junctions: A Noise-Robust, Training-Free Structural Prior for Volumetric Inverse Problems

**arXiv ID:** 2603.02149 | [PDF](https://arxiv.org/pdf/2603.02149v1)

**作者:** Namhoon Kim `[一作]` (Georgia Institute of Technology), Sara Fridovich-Keil `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出3D Field of Junctions（3D FoJ）方法，利用无训练的显式几何先验对3D体积进行去噪，并可作为正则化项应用于低信噪比的体积逆问题。

**💡 创新点**

创新点在于将二维 FoJ 扩展到三维，用三条相交平面分割体积形成瓦片，采用可微软指示器和 Heaviside 近似实现梯度优化，并能在多种任务中直接去噪或嵌入为正则化。

**🔧 技术方法**

使用的技术包括：重叠体积块分割、平面切割参数化、软指示器与平滑边界函数、非凸目标的初始化与联合梯度优化、Adam 更新、以及 prox‑gradient 正则化框架。

**📊 数据集**

实验数据集包括：低剂量 CT 合成数据集（pepper、teapot、jaw、foot、engine）、真实 cryogenic electron tomography 数据（centriole、mitochondria、vesicle、VEEV）、以及点云数据集（PointCleanNet 28个、dragon 点云）。

**📈 对比分析**

与 3D TV、R²‑Gaussian、Filter2Noise、SC‑Net、NMSG、NLM、PointCleanNet、PointCVaR 等方法在 MS‑SSIM、3D PSNR、Chamfer Distance 等指标上对比，3D FoJ 在低 SNR 条件下 consistently 表现最优，优于传统与深度学习方法。

**⚠️ 局限性**

局限性包括：未针对每个任务进行专门的超参数调优，且计算成本较高，内存随体积/块尺寸增大而显著增加，通常需要多 GPU 并行处理。

---

## 361. AMDS: Attack-Aware Multi-Stage Defense System for Network Intrusion Detection with Two-Stage Adaptive Weight Learning

**arXiv ID:** 2603.00859 | [PDF](https://arxiv.org/pdf/2603.00859v1)

**作者:** Oluseyi Olukola `[一作]` (University of Southern Mississippi), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**通讯引用:** 257 | [OpenAlex ID](https://openalex.org/A5102764912)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种攻击感知的多阶段防御系统（AMDS），通过多信号加权组合和攻击类别推断实现对网络入侵检测的自适应防御。

**💡 创新点**

创新点在于：①利用模型不一致、预测不确定性和分布异常三种信号的加权学习；②基于攻击类别的两阶段自适应检测与攻击自适应模型加权；③双输出架构同时提供异常检测与攻击分类。

**🔧 技术方法**

技术包括：集成学习、多信号加权（entropy、disagreement、anomaly）、两阶段检测算法、攻击类别推断阈值、基于置信度的模型加权、马氏距离异常分数、Cascade 路由。

**📊 数据集**

使用数据集：CSE-CIC-IDS2018（7类，77特征）和UNSW-NB15（9类映射至5类，190特征）。

**📈 对比分析**

与标准集成、对抗训练集成以及单一最佳模型比较，AMDS在CSE-CIC-IDS2018上整体准确率提升4.5个百分点、F1提升9.0个百分点，AUC达到94.2%；在UNSW-NB15中受基线性能限制表现不佳，提示维度依赖。

**⚠️ 局限性**

局限性：需预先生成对抗样本以学习权重，无法保证对未知攻击族的鲁棒性；二分类攻击推断可能无法覆盖更细粒度攻击；权重在部署后保持静态，长期适应性未知；跨数据集性能差异大，受特征维度与攻击强度比例影响。

---

## 362. Interpretable Cross-Network Attention for Resting-State fMRI Representation Learning

**arXiv ID:** 2603.00786 | [PDF](https://arxiv.org/pdf/2603.00786v1)

**作者:** Karanpartap Singh `[一作]` (Stanford University), Ehsan Adeli `[通讯]` (Stanford University)

**通讯引用:** 14216 | [OpenAlex ID](https://openalex.org/A5015355317)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一个网络感知的自监督框架，利用按功能网络掩蔽并用交叉注意力解码进行 rs‑fMRI 重建，捕捉网络间依赖并生成可解释的功能重组表征。

**💡 创新点**

创新点在于：① 结构化网络层级掩蔽；② 仅使用交叉注意力解码器实现纯粹跨网络依赖建模；③ 将注意力权重直接作为跨网络可解释性贡献度。

**🔧 技术方法**

采用 DiFuMo 1024 区域分区的 token 化，Mask‑then‑Decode 自监督学习（CrossMAE 变体），以及交叉注意力 Transformer 编码‑解码架构。

**📊 数据集**

预训练数据集为 HCP‑YA、HCP‑Aging、HCP‑Development、ABCD 共 3,087 条 rs‑fMRI；下游评估使用 ADNI 2,366 条，涵盖 CN/MCI/AD。

**📈 对比分析**

与 BrainLM、Brain‑JEPA、BrainGNN 等基线以及其随机掩蔽或去掉交叉注意力的 ablation 进行 3‑分类（CN/MCI/AD）比较，平衡准确率 77.5%、F1 77.5%、AUC 0.85，优于或匹配传统模型。

**⚠️ 局限性**

局限性包括：仅验证于 AD，未检验在其他神经疾病中的通用性；掩蔽策略固定，可能忽略跨网络细粒度动态；自监督训练受限于所选分区与网络划分。

---

## 363. RubricBench: Aligning Model-Generated Rubrics with Human Standards

**arXiv ID:** 2603.01562 | [PDF](https://arxiv.org/pdf/2603.01562v1)

**作者:** Qiyuan Zhang `[一作]` (City University of Hong Kong), Chen Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 27782 | [OpenAlex ID](https://openalex.org/A5100652421)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了RubricBench基准，包含1,147个具有人工标注的 rubric 的对比样本，专注于检验 rubric-guided 评估的可靠性。

**💡 创新点**

提出了统一的判别难度与基于指令的 rubric 注释方法，揭示模型在自行生成 rubric 时的显著缺口与执行失配问题。

**🔧 技术方法**

采用多维过滤管道、专家双人注释、Rubric Recall/HallucinationRate/StructuralF1 等指标，并对多种 scalar、生成、LLM-as-judge 与 rubric-aware 评估模型进行实验。

**📊 数据集**

从 RewardBench、HelpSteer3、PPE 等现有奖励模型基准中筛选并重构样本，形成了具有高判别性的 1,147 条对比数据。

**📈 对比分析**

在该基准上对比 4 类评估范式，发现自生成 rubric 的性能比无 rubric 提升约 20%，而引入人类 rubric 后提升约 27%，最高约 85% 的准确率。

**⚠️ 局限性**

受限于仅利用公开基准样本，规模有限且需人工 rubric 资源；此外，二值 checklist 形式可能无法完整捕捉主观质量的连续性。

---

## 364. GroundedSurg: A Multi-Procedure Benchmark for Language-Conditioned Surgical Tool Segmentation

**arXiv ID:** 2603.01108 | [PDF](https://arxiv.org/pdf/2603.01108v1)

**作者:** Tajamul Ashraf `[一作]` (Gaash Research Lab National Institute of Technology Srinagar), Janibul Bashir `[通讯]` (Gaash Research Lab National Institute of Technology Srinagar)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于语言条件的多手术程序工具实例分割基准GroundedSurg，并提供了612张手术图像、1071个工具实例的标注。

**💡 创新点**

首次将外科工具感知转化为语言驱动的实例级分割任务，结合自然语言查询、边界框、中心点和像素级掩码实现多实例、上下文关联的精准定位。

**🔧 技术方法**

采用多模态视觉语言模型（如Qwen2.5‑VL、VisionReasoner‑7B）与冻结的SAM2/3分割后端，利用提示工程、结构化空间锚点进行定位与分割，评估指标包括IoU、Dice、BBox IoU和中心点误差。

**📊 数据集**

从公开的多手术数据集（InSeg1/2、SISVE、EndoVis、CholecInstanceSeg）中采集并统一标注，覆盖眼科、腹腔镜、机器人和开放式手术场景。

**📈 对比分析**

在零样本设置下对多种VLM进行比较，结果显示大多数模型在IoU@0.1可达一定水平，但在更严格阈值（≥0.3）明显退化；VisionReasoner‑7B在BBox IoU和Dice上表现最佳，说明推理能力对精细定位更有帮助。

**⚠️ 局限性**

主要局限在于基准规模有限、对提示敏感、现有模型在高阈值下精度不足、缺乏针对手术场景的专门微调与深度语义推理。

---

## 365. FoSS: Modeling Long Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier State Space Integration

**arXiv ID:** 2603.01284 | [PDF](https://arxiv.org/pdf/2603.01284v1)

**作者:** Yizhou Huang `[一作]` (Brunel University of London), Kezhi Wang `[通讯]` (Brunel University of London)

**通讯引用:** 14309 | [OpenAlex ID](https://openalex.org/A5022523766)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 FoSS 双分支框架，将频域分解与线性时间序列建模相结合，用于多代理轨迹预测。

**💡 创新点**

创新点包括：1）采用 HelixSort 进行频谱有序重排，使频域序列可被状态空间模型顺序处理；2）引入两种选择性状态空间子模块 Coarse2Fine‑SSM 与 SpecEvolve‑SSM，在频域实现粗到细的空间交互和通道演化；3）在时域使用输入自适应选择性 SSM 近似自注意力但线性复杂度；4）跨域交叉注意力融合后以可学习查询生成多模态轨迹，并用加权融合表达不确定性。

**🔧 技术方法**

主要技术：离散傅里叶变换（FFT）、HelixSort 频谱重排、选择性状态空间模型（SSM）、输入自适应 SSM、跨域交叉注意力、可学习查询多模态解码、联合时间域与频域的 L1 损失。

**📊 数据集**

使用 Argoverse 1 与 Argoverse 2 两大公开轨迹预测数据集。

**📈 对比分析**

与多种基线（LaneGCN、DenseTNT、SceneTransformer、HiVT、MultiPath++、QCNet、DeMo 等）对比，FoSS 在 Argoverse 2 上 minADE_6 为 0.61、minFDE_6 为 1.07、b‑minFDE_6 为 1.69，均为最佳或第二佳；在 Argoverse 1 上 minADE_1 为 1.67、minFDE_1 为 2.05，保持最优；同时参数仅 4.18M，FLOPs 22.1G，推理延迟 64 ms，均显著低于同类方法。

**⚠️ 局限性**

局限性包括：对频率突变（如快速变道）仍略显抖动；在极端高频动态场景下频域重排可能无法充分捕获瞬时细节；模型对频域与时域的配合依赖较高，若训练不充分，可能导致不稳定性。

---

## 366. Reasoning Boosts Opinion Alignment in LLMs

**arXiv ID:** 2603.01214 | [PDF](https://arxiv.org/pdf/2603.01214v1)

**作者:** Frédéric Berdoz `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21287 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何利用大型语言模型通过结构化推理对个体或团体的政治观点进行对齐，提出在问卷数据上使用强化学习训练推理过程；

**💡 创新点**

创新点在于将推理与强化学习（GRPO）结合，直接在已知个体答案上训练模型生成推理链并对齐观点，同时在三国不同政治体系的真实问卷上建立基准；

**🔧 技术方法**

使用GRPO（Group‑relative Policy Optimization）对模型进行强化学习，辅以监督微调（SFT）产生结构化推理；

**📊 数据集**

使用美国ANES 2020、德国Wahl‑o‑Mat、瑞士SmartVote三份公开调查数据；

**📈 对比分析**

与随机、最频繁答案、以及仅用推理预训练的基础模型进行对比；在三大数据集上GRPO模型平均宏F1约55‑70%，显著优于基线，尤其在SmartVote上达到70.7%；

**⚠️ 局限性**

局限包括模型偏向左倾，右倾与中间倾向表现较差；对中立立场预测困难；需要为每个个体训练单独模型，计算成本高；问卷项目有限，难以覆盖更广泛议题；

---

## 367. Energy-Efficient Information Representation in MNIST Classification Using Biologically Inspired Learning

**arXiv ID:** 2603.00588 | [PDF](https://arxiv.org/pdf/2603.00588v1)

**作者:** Patrick Stricker `[一作]` (Albstadt-Sigmaringen University), Andreas Knoblauch `[通讯]` (Albstadt-Sigmaringen University)

**通讯引用:** 1594 | [OpenAlex ID](https://openalex.org/A5032109856)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用基于竞争性Hebbian可塑性与结构可塑性的生物学启发式学习规则，在MNIST分类任务中实现稀疏且高效的神经网络表示。

**💡 创新点**

创新点在于将竞争性Hebbian学习、非负约束、权重扰动、Homeostatic Plasticity以及信息瓶颈相结合，天然抑制过参数化，最大化突触容量，同时不需要预先设计网络结构。

**🔧 技术方法**

采用竞争性Hebbian可塑性、权重扰动、Homeostatic Plasticity、变分信息瓶颈（VIB）层和信息理论评估（互信息、突触容量）等技术。

**📊 数据集**

使用MNIST手写数字子集（数字1、2、6）进行训练与评估。

**📈 对比分析**

与标准BP和Chorowski的非负稀疏BP方法比较，实验显示本方法在突触容量和信息压缩上优于对手，准确率略低于BP，但显著降低了非静默突触数和能耗。

**⚠️ 局限性**

局限在于分类准确率仍略低于BP，对更大规模或更深层网络的可扩展性尚未验证，需要进一步提升准确率与计算效率。

---

## 368. On the Metric Nature of (Differential) Logical Relations

**arXiv ID:** 2603.01317 | [PDF](https://arxiv.org/pdf/2603.01317v1)

**作者:** Ugo Dal Lago `[一作]` (University of Bologna), Paolo Pistone `[通讯]` (Universite Claude Bernard Lyon)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了差分逻辑关系的度量性质，提出了弱化反射性和对称性的“quasi‑quasi‑metric”概念，并构造了其笛卡尔闭的度量范畴，给出基本引理，随后探讨了由此得到的差分前逻辑关系的偏序结构，证明存在最细的量化等式理论，却不存在最大元素。

**💡 创新点**

创新点在于：①把差分逻辑关系与量化度量的理论通过 quasi‑quasi‑metrics 连接起来；②在笛卡尔闭范畴框架下得到基本引理；③证明差分前逻辑关系的偏序具有最细但无最大元的特殊性质。

**🔧 技术方法**

采用的技术包括：量子模（quantale）值度量、逻辑关系的递归提升、笛卡尔闭范畴的构造与证明、量化等式理论的形式化以及对自相似性、左强传递性等属性的系统化分析。

**📊 数据集**

本文未使用任何实验数据集，全部基于形式化证明与理论构造。

**📈 对比分析**

本文未涉及实验对比或性能评估，而是通过形式化证明展示方法的有效性和理论性质。

**⚠️ 局限性**

主要局限：仅处理简单类型，非对称情形；对称情形和含效应的高阶语言尚未覆盖；缺乏最大差分前逻辑关系导致难以定义“上下文度量”。

---

## 369. A Tree-Structured Two-Phase Commit Framework for OceanBase: Optimizing Scalability and Consistency

**arXiv ID:** 2603.00866 | [PDF](https://arxiv.org/pdf/2603.00866v1)

**作者:** Quanqing Xu `[一作]` (OceanBase), Zixiang Zhai `[通讯]` (OceanBase)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 OceanBase 中提出一种以日志流为参与者的树形两阶段提交（2PC）框架，能够在分区迁移期间动态构造提交树，保证事务一致性。

**💡 创新点**

创新点包括：① 将日志流视为原子参与者，显著降低参与者数量；② 采用根节点驱动的 DAG 树结构实现对分区迁移的递归处理，消除显式参与者列表更新；③ 引入 prepare_unknown 与 trans_unknown 状态以避免上下文丢失导致的误 abort 并隐藏给用户的歧义。

**🔧 技术方法**

使用技术包括：单机日志流架构、Paxos 日志复制、动态树形 2PC 协议、TLA+ 形式化验证、事务数据表（TDT）用于缓存已完成事务状态，以及基于递归日志树构造的分区迁移上下文。

**📊 数据集**

实验数据集主要使用 TPC‑C 与 Sysbench 的 OLTP 工作负载，部署在 OceanBase 4.4.1 集群（8 节点）上进行性能评估。

**📈 对比分析**

通过与 MySQL、TiDB 以及 OceanBase 3.x（分区粒度 2PC）进行基准对比，发现该框架在吞吐量上几乎与单机事务持平，延迟提升不到 20%，且在分区迁移时保持线性可扩展、尾部延迟稳定，整体性能显著优于传统 2PC。

**⚠️ 局限性**

局限性主要体现在：① 需要维护未知状态机制和事务数据表，增加实现复杂度；② 在极深的迁移链或高度并发的分区迁移场景下，树高度提升可能导致尾部延迟上升；③ 迁移过程中仍会产生一定的 CPU 与网络开销，尽管已被控制在可接受范围。

---

## 370. Process Over Outcome: Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection

**arXiv ID:** 2603.01993 | [PDF](https://arxiv.org/pdf/2603.01993v1)

**作者:** Yuchen Zhang `[一作]` (Xi'an Jiaotong University), Zhedong Zheng `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于推理驱动的多模态伪造检测框架 REFORM，并构建了大规模含推理注释的数据集 ROM。

**💡 创新点**

创新点在于将检测任务从结果监督转为推理链优化，采用三阶段学习与强化学习对齐推理与结论，从而显著提升跨域与零样本泛化能力。

**🔧 技术方法**

使用的技术包括认知预热编码器、双解码器结构、链式思考（CoT）语言模型以及 Group Relative Policy Optimization（GRPO）强化学习。

**📊 数据集**

主要使用了自研的 ROM 数据集（704k 样本含推理注释），以及 MMFakeBench 与 DGM4 等公开基准。

**📈 对比分析**

在 ROM 上平均准确率达 88.22%，MMFakeBench 零样本 F1 为 74.9，DGM4 平均 mAP 达 65.72，均优于现有 SOTA（如 AMD、FKA‑Owl 等）并在跨域与零样本任务中表现最佳。

**⚠️ 局限性**

局限性包括：依赖教师模型生成的推理注释可能引入幻觉噪声；推理模式下推理链生成耗时，吞吐量低；对教师质量敏感。

---

## 371. The On-Chain and Off-Chain Mechanisms of DAO-to-DAO Voting

**arXiv ID:** 2603.00708 | [PDF](https://arxiv.org/pdf/2603.00708v1)

**作者:** Thomas Lloyd `[一作]` (South East Technological University), Martin Harrigan `[通讯]` (South East Technological University)

**通讯引用:** 1552 | [OpenAlex ID](https://openalex.org/A5035746981)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套自动检测以太坊区块链上 DAO 之间元治理（DAO‑to‑DAO voting）的方法，并构建了包含 61 个 DAO 与 72 条元治理关系的网络图；

**💡 创新点**

创新点在于：①使用签名匹配算法灵活识别多种 DAO 框架与投票方案的治理合约；②结合链上与链下（Snapshot）数据，自动映射投票权来源并检测跨 DAO 投票行为；③构造可视化网络展示元治理结构及其三类典型案例；

**🔧 技术方法**

技术包括：以太坊归档节点抓取事件日志与调用轨迹、基于函数签名与事件哈希的治理合约识别、Multi‑Sig 合约抽取、Snapshot GraphQL API 查询、Etherscan 名称标签匹配；

**📊 数据集**

数据集由 16 个高市值 DAO 的治理代币及其相关合约、Snapshot Spaces（≥50 关注者）以及以太坊主网从 2023 年 10 月 18,299,999 区块起的历史数据构成；

**📈 对比分析**

方法对 16 个 DAO 的治理框架实现了 11/18 的成功识别，误报率约 39%，其中 28 次误报来自 Snapshot 处理，2 次误报来自签名匹配。整体准确率仍有限，但已能生成 61 结点、72 条边的元治理网络；

**⚠️ 局限性**

局限性包括：①对非标准事件或自定义治理合约识别失效；② Snapshot 空间过滤仍依赖关注者阈值，可能漏检或误检；③方法对治理合约与投票权合约分离的情况不够鲁棒；④误报率高，尤其是包装合约与多签委托场景。

---

## 372. Geometry OR Tracker: Universal Geometric Operating Room Tracking

**arXiv ID:** 2603.00560 | [PDF](https://arxiv.org/pdf/2603.00560v1)

**作者:** Yihua Shao `[一作]` (Centre for Artificial Intelligence and Robotics Hong Kong Institute of Science and Innovation Chinese Academy of Sciences), Nassir Navab `[通讯]` (Technische Universität München)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Geometry OR Tracker，一种两阶段流程，用于在存在相机标定误差的手术室多视角 RGB-D 数据中实现几何一致的 4D 轨迹重建与 3D 点跟踪。

**💡 创新点**

创新点在于：① 通过 Multi‑view Metric Geometry Rectification 模块将不可靠的标定与 RGB‑D 对齐错误自动校正为全局尺度一致的相机参数；② 在统一的 OR 坐标系中进行遮挡鲁棒的 3D 点跟踪，显著降低 ghosting 并提升轨迹稳定性。

**🔧 技术方法**

主要技术包括：几何先验驱动的深度与标定自适应校正、全局尺度恢复、基于多视角特征融合的 3D 特征云构建、局部 3D 邻域检索与迭代 Transformer 细化。

**📊 数据集**

使用 MM‑OR 基准数据集（含 5 只 Kinect 相机的同步 RGB‑D 视频），在 10 个随机场景（每 100 帧）上进行评估。

**📈 对比分析**

与 CoTracker3、SpaTrackerV2、LocoTrack、SceneTracker、DELTA、MVTracker 等单视/多视 3D 跟踪基线比较；在 AJ、Δ_avg、OA、MTE 等指标上均优于全部基线，尤其是遮挡时的准确率和轨迹误差显著下降。

**⚠️ 局限性**

局限性：① 仅在手术室静态相机配置下实现一次性全局标定，动态相机或长时段漂移仍需进一步研究；② 依赖较为丰富的几何先验模型，若先验不充分或场景异常复杂时校正效果可能受限。

---

## 373. The Expurgated Error Exponent is Not Universally Achievable

**arXiv ID:** 2603.01736 | [PDF](https://arxiv.org/pdf/2603.01736v1)

**作者:** Seyed AmirPouya Moeini `[一作]` (University of Cambridge), Albert Guillén i Fàbregas `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了离散无记忆信道（DMC）的排除误差指数能否实现通用解码，证明其在一般情况下并不可通用实现；

**💡 创新点**

创新点在于构造了一族DMC，并证明无论采用何种通用解码策略（如MMI），均无法同时实现该族所有信道的排除误差指数；

**🔧 技术方法**

主要使用了信息论中的类型方法、联合类型分布构造、以及对MMI和SMI解码器的误差分析技术；

**📊 数据集**

没有使用实验数据集，整个工作基于理论构造和解析证明；

**📈 对比分析**

通过对比排除误差指数与MMI实现的误差指数，发现后者在所构造的信道族上总是低于前者，说明MMI并不能达到排除误差指数；

**⚠️ 局限性**

局限性在于结论仅针对特定构造的DMC族，并未给出对更广泛信道类的通用性结果，也未探讨是否存在其他更优的通用解码策略。

---

## 374. Foundation Models in Remote Sensing: Evolving from Unimodality to Multimodality

**arXiv ID:** 2603.00988 | [PDF](https://arxiv.org/pdf/2603.00988v1)

**作者:** Danfeng Hong `[一作]` (Southeast University), Jocelyn Chanussot `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对遥感领域的基础模型进行系统综述，阐述从单模态到多模态的演进，并提供实用的训练与部署教程；

**💡 创新点**

首次将遥感基础模型从单模态到多模态的演进进行系统梳理，并结合多任务基准评估，提出面向新人研究者的教程式框架；

**🔧 技术方法**

综述了自监督预训练+微调范式、对比学习、掩码自编码器、Transformer、跨模态对齐等技术，并给出示例实现；

**📊 数据集**

主要讨论了SeCo、SSL4EO、FMoW、MillionAID、SSL4EO-L、Satlas、GeoPile、DOFA-data、Prithvi、Skysense、MMEarth等大规模遥感数据集；

**📈 对比分析**

在PANGAEA-Bench上对多模态与单模态模型进行对比，表明多模态模型在分类、分割、变更检测等任务上普遍优于单模态，提升幅度显著；

**⚠️ 局限性**

仍缺乏统一的评测基准、模型对不同模态兼容性差、对低资源场景的适配不足、对灾害、极端气候等鲁棒性待提升。

---

## 375. Expanding LLM Agent Boundaries with Strategy-Guided Exploration

**arXiv ID:** 2603.02045 | [PDF](https://arxiv.org/pdf/2603.02045v1)

**作者:** Andrew Szot `[一作]` (Apple), Alexander Toshev `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种通过让LLM在每一步先生成高层语言策略并据此执行动作，从而提升在稀疏奖励环境中的探索效率的方法。

**💡 创新点**

创新点在于结合多温度采样与策略反思，能够显著扩展策略空间，使LLM能够发现原始模型无法解决的新解。

**🔧 技术方法**

采用LLM策略采样分布、混合温度采样、策略反思、GRPO强化学习框架等技术。

**📊 数据集**

使用的任务集包括AndroidWorld、LangR、Coding（hard）以及AppWorld等四种多步骤代理任务。

**📈 对比分析**

与GRPO、EntropyAdv、RND、RLAD等基线对比，实验显示在所有四个环境中均超过基线约27%的最终成功率，并突破基础模型的最大成功上限。

**⚠️ 局限性**

局限性在于需要基准LLM具有足够的推理与规划能力；生成策略每一步会增加推理延迟，对小模型效果有限。

---

## 376. LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model

**arXiv ID:** 2603.01068 | [PDF](https://arxiv.org/pdf/2603.01068v1)

**作者:** Zebin You `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 24088 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种长度自适应的 omni diffusion 模型 LLaDA‑o，用于多模态理解和生成。

**💡 创新点**

创新点是将离散文本与连续图像分别采用掩码扩散与连续扩散，并通过 Mixture‑of‑Diffusion 框架共享高效注意力骨干，同时引入自适应长度增益实现可变长度输出。

**🔧 技术方法**

采用了 Mixture‑of‑Diffusion、掩码扩散、连续扩散、共享自注意力、VAE、FLUX、SigLIP、两阶段/三阶段训练等技术。

**📊 数据集**

使用了大量公开数据集进行训练与评估，包括 MMU‑M、MME、SEED‑Bench、MMBench、MathVerse、MathVista、AI2D、ChartQA、DocVQA、InfoVQA 等；评估基准包括 GenEval、DPG‑Bench 等。

**📈 对比分析**

与现有 omni diffusion 和生成仅模型对比，LLaDA‑o 在多模态理解任务中达到 state‑of‑the‑art，文本到图像生成在 DPG‑Bench 获得 87.04 分，优于 Lumina‑DiMOO、Show‑o2 等模型。

**⚠️ 局限性**

局限性在于模型规模相对较小，导致在某些语言推理任务上略逊于大规模自回归模型，并且对极长文本的处理仍需进一步验证。

---

## 377. CHIMERA: Compact Synthetic Data for Generalizable LLM Reasoning

**arXiv ID:** 2603.00889 | [PDF](https://arxiv.org/pdf/2603.00889v1)

**作者:** Xinyu Zhu `[一作]` (University of Virginia), Yu Meng `[通讯]` (University of Virginia)

**通讯引用:** 42289 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个 9K 样本的全合成推理数据集 CHIMERA，并利用该数据集对 4B 参数 Qwen3 进行 SFT+RL 后训练得到的模型，在多项学科推理基准上实现了显著提升。

**💡 创新点**

① 通过三阶段 LLM 驱动的合成流程：主题扩展、问题生成、推理轨迹合成，得到长 Chain‑of‑Thought（CoT）轨迹；② 采用结构化层次化学科主题，覆盖 8 大科学领域；③ 采用双模型交叉验证的自动评估机制，实现无人工标注的高质量保证。

**🔧 技术方法**

主要技术包括：LLM（Qwen3）进行合成与训练；基于大语言模型的交叉验证器（两名独立模型）对问题和答案做真实性与正确性校验；SFT+CISPO RL 训练流程；多轮采样与无偏 pass@1/ pass@k 评估。

**📊 数据集**

核心数据集为自研 CHIMERA（9,225 个包含长 CoT 的问题）；基准对比数据集包括 GPQA‑Diamond、AIME24/25/26、HMMT25、Humanity’s Last Exam (HLE) 等；对比实验还使用 OpenScience 等公开合成数据集。

**📈 对比分析**

通过将 4B Qwen3 在 CHIMERA 上 SFT+RL 后与基线模型（未训练、OpenScience 训练、DeepSeek‑R1、Qwen3‑235B 等）在上述基准上进行 pass@1/ pass@k 评估，发现 CHIMERA‑训练模型在 GPQA‑Diamond、AIME、HMMT、HLE 等任务上分别提升 4.3–9.7%（pass@1）且在 8B–70B 规模模型基础上实现相近或更优性能，接近两阶规模更大的模型。

**⚠️ 局限性**

限制包括：① 样本规模相对较小（9K），虽然质量高但覆盖面仍有限；② 依赖强大 LLM 生成 CoT，若生成模型自身存在偏差，可能影响数据质量；③ 评估完全自动化，缺少人类最终验证，可能遗漏细微错误；④ 主要集中在学科推理，通用推理与跨模态推理的泛化性待进一步验证。

---

## 378. Modeling Grammatical Hypothesis Testing in Young Learners: A Sequence-Based Learning Analytics Study of Morphosyntactic Reasoning in an Interactive Game

**arXiv ID:** 2603.02084 | [PDF](https://arxiv.org/pdf/2603.02084v1)

**作者:** Thierry Geoffre `[一作]` (University of Luxembourg), Trystan Geoffre `[通讯]` (University of Fribourg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对小学学生在交互式语法游戏Tirettes中滑块移动的动作序列进行收集与分析，探索他们在句子构建过程中的语法推理策略。

**💡 创新点**

将滑块移动视为假设检验动作，并用汉明距离衡量与最优解的距离，首次将过程挖掘与学习分析相结合揭示儿童实时的语法推理与策略，为教师提供实时支架与回放工具。

**🔧 技术方法**

利用学习分析中的序列挖掘、汉明距离度量、过程挖掘与统计分析技术，并结合GamesHub平台的适应性学习引擎与日志记录。

**📊 数据集**

使用2025年4月在11所法语小学收集的597次游戏会话、100名8-11岁学生的数据，共9,783个动作（7,126次滑块移动）和2,657次句子验证，构成了实验数据集。

**📈 对比分析**

通过对汉明距离随动作序列的平均变化趋势和特定练习的收敛曲线进行可视化比较，评估不同难度与解数量练习的收敛效果。结果显示，主语-谓语链在大多数情况下能收敛，但整体趋势受少数学生影响，说明方法对个体差异敏感。

**⚠️ 局限性**

限制包括：练习多样性、学生差异和适应路径等变量过多导致整体趋势不稳定；缺乏对更大样本或其他语言的验证；研究依赖GamesHub平台日志，外部可复制性受限。

---

## 379. PymooLab: An Open-Source Visual Analytics Framework for Multi-Objective Optimization using LLM-Based Code Generation and MCDM

**arXiv ID:** 2603.01345 | [PDF](https://arxiv.org/pdf/2603.01345v1)

**作者:** Thiago Santos `[一作]` (Federal University of Ouro Preto), Gustavo de Souza `[通讯]` (Federal University of Ouro Preto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

开发了一个基于 pymoo 的可视化实验平台 PymooLab，集成了 LLM 自动化建模、实验管理、可复现的元数据记录以及后验多准则决策支持，旨在降低优化工作流的编程门槛并提升实验可复现性。

**💡 创新点**

创新点包括：①将大型语言模型嵌入建模环节，实现自然语言到可执行问题代码的即时转换；②设计可视化实验管理模块，使多算法、多问题、多次实验可在 GUI 中统一配置、执行和统计；③在同一平台内集成 MCDM 工具，直接对 Pareto 前沿进行决策分析；④采用元数据归档和随机种子计划，保证实验结果的可复现性与可追溯性。

**🔧 技术方法**

技术栈主要包括：Python 及 pymoo 框架、JAX 加速器、LLM（如 Qwen2.5-Coder）、GUI 交互界面、可视化库（用于前沿展示、指标曲线）、MCDM 算法（TOPSIS、加权和）以及 JSON/CSV 记录机制。

**📊 数据集**

实验使用了标准多目标基准问题（如 ZDT、DTLZ 系列等），并通过这些合成基准评估算法性能；未使用真实工程数据集，但平台可兼容任意自定义问题。

**📈 对比分析**

比较方法：在 Experiment Module 中一次性配置多算法、多个基准问题和多次随机种子，自动收集指标（如 IGD、HV 等）并计算均值±标准差；提供非参数统计检验（Wilcoxon、Friedman）来评估算法间差异。性能主要表现在实验流程的可复现性和用户体验提升，JAX 加速显著提高高维评估的可扩展性；具体数值指标未给出。

**⚠️ 局限性**

局限性：①依赖用户具备一定 Python 基础，LLM 生成的代码仍需人工验证；②MCDM 支持仅为后验单一方法，缺乏交互式偏好学习或多准则敏感性分析；③对极高维 many‑objective 问题的可视化与决策支持仍有限；④平台核心功能高度依赖 pymoo 与 JAX，迁移到其他生态可能受限。

---

## 380. MMCOMET: A Large-Scale Multimodal Commonsense Knowledge Graph for Contextual Reasoning

**arXiv ID:** 2603.01055 | [PDF](https://arxiv.org/pdf/2603.01055v1)

**作者:** Eileen Wang `[一作]` (University of Sydney), Caren Han `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MMCOMET，一种将物理、社交和事件三类常识关系与对应图片关联的多模态常识知识图谱；

**💡 创新点**

创新点在于：①首次构建大规模多模态常识图谱，覆盖19种关系，近1M三元组；②设计混合检索管道（基于CLIP的相似度匹配+基于实体抽象度的网页检索），实现高效且质量优秀的视觉对齐；③证明该资源可作为Plug‑and‑Play的外部知识库，显著提升视觉故事生成、VQA和图像描述任务；

**🔧 技术方法**

采用CLIP模型进行文本-图像嵌入、相似度匹配；利用词性标签和倒排索引做候选缩减；基于“concreteness”评分决定是否走网页检索；人类评测、标准VST/VQA/IC评价指标；

**📊 数据集**

基于ATOMIC2020文本知识库；使用开源图像语料（Conceptual Captions、COCO、Flickr30K、Visual Storytelling）进行相似度检索；网页检索（Google/百度等）获取抽象概念对应图片；

**📈 对比分析**

与传统文本KG（ATOMIC）、百科MMKG（IMGpedia、ImageGraph等）对比；在VST、VQA、IC三项任务上，MMCOMET提升R‑VG、BLEURT、BLEU、CIDEr等指标，表现出比仅使用文本KG或无KG更高的分数，证明多模态常识提升性能；

**⚠️ 局限性**

局限包括：①对抽象概念的图像检索仍不完美，匹配质量不如物理实体；②依赖外部网页检索，可能带来噪声和版权问题；③目前仅支持英文，跨语言扩展受限；④对极端稀有关系或新兴事件的覆盖不足。

---

## 381. Reservoir Subspace Injection for Online ICA under Top-n Whitening

**arXiv ID:** 2603.02178 | [PDF](https://arxiv.org/pdf/2603.02178v1)

**作者:** Wenjun Xiao `[一作]` (George Washington University), Vince D Calhoun `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于reservoir subspace injection (RSI) 的在线ICA框架，解决了在top‑n白化下扩展特征被丢弃的问题，并给出了诊断指标及自适应控制策略。

**💡 创新点**

创新点在于引入 IER、SSO 与 ρ_x 三种诊断指标，揭示“crowd‑out”效应，并设计了低开销的 guarded 控制器以保持透传能量，显著提升非线性混合下的分离性能。

**🔧 技术方法**

采用了稀疏echo‑state网络作为高维非线性映射、指数移动平均白化、自然梯度ICA更新，并结合 IER/SSO/ρ_x 诊断与自适应注入规模控制。

**📊 数据集**

使用了由 Lorenz、Mackey‑Glass 与线性 chirp 组成的三源混合信号（静态、时变、非线性三种混合模式），以及 Laplace、square wave、sawtooth 等超高斯标准源作为基准数据集。

**📈 对比分析**

与 vanilla 在线ICA（ORICA‑inspired）和离线 FastICA 进行对比，评估指标为 SI‑SDR_sc 与相关系数；RSI 在非线性混合下提升约 +1.7 dB，且在超高斯基准上从 -2.9 dB 变为 +0.6 dB，显示出显著性能优势。

**⚠️ 局限性**

局限性包括：top‑n 白化仍是 rank‑budget 限制，RSI 主要提升的是单体能量而非跨块结构；实验仅在小维度（n=3）合成信号上验证，缺乏真实数据的泛化评估；以及对混合矩阵条件数的敏感度仍需进一步研究。

---

## 382. A Deployable Bio-inspired Compliant Leg Design for Enhanced Leaping in Quadruped Robots

**arXiv ID:** 2603.01128 | [PDF](https://arxiv.org/pdf/2603.01128v1)

**作者:** Yiyang Chen `[一作]` (Xi'an Jiaotong-Liverpool University), Yikun Cheng `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一种可部署柔性腿（Deployable Compliant Leg，DCL），利用3D打印的PEBA材料和Gyroid晶格结构，模拟了蛙跳器官的能量存储与释放机制，并在四足机器人上验证了其提升跳跃高度的效果。

**💡 创新点**

创新点包括：①将生物学半月形弹性结构转化为可部署的扇形弹性模块（SSCM）并嵌入Gyroid晶格；②设计了双稳态可部署翻转机制，实现能量存储与正常行走模式的无缝切换；③通过FEA与实验结合的优化流程，得到精确的非线性弹性模型并实现控制。

**🔧 技术方法**

使用技术主要有：3D打印（PEBA材料）、nTopology晶格优化、有限元分析（Abaqus Explicit）构建非线性弹性模型、运动捕捉系统（Luster FZMotion）进行实验验证、PD控制与分段弹性模型集成。

**📊 数据集**

本文未使用公开数据集，而是基于自制的Unitree Go2机器人平台进行实验测量。

**📈 对比分析**

实验通过对比基线（无弹性模块）、挂载但保持收缩状态（Stowed）以及部署状态（Deployed）的垂直跳跃高度，验证了DCL在保持低质量负担的前提下，能够将有效跳跃高度提升约17.1%（从373.1 mm提升到437.1 mm），并表明附加质量对行走性能影响极小（-0.4%）。

**⚠️ 局限性**

局限性包括：①部署机制仍为手动触发，缺乏自动化切换；②仅在单一机器人平台（Unitree Go2）上测试，缺乏跨平台验证；③只评估了垂直跳跃性能，未检验在复杂地形或多任务情境下的整体机动性；④晶格结构对打印精度和耐久性的依赖仍需进一步研究。

---

## 383. Understanding LoRA as Knowledge Memory: An Empirical Analysis

**arXiv ID:** 2603.01097 | [PDF](https://arxiv.org/pdf/2603.01097v1)

**作者:** Seungju Back `[一作]` (KAIST), Sungjin Ahn `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了低秩适配器 LoRA 在大语言模型中作为可插拔知识存储的能力，涵盖单模块容量、数据驱动的内部化、模块化组合以及与 ICL/RAG 的混合使用。

**💡 创新点**

创新点在于：①给出 LoRA 参数容量与记忆能力的经验映射；②提出高密度合成数据（QA/摘要/重写）显著提升记忆效率；③量化多模块系统的路由与合并瓶颈，提出干扰感知合并策略；④展示 LoRA 与非参数方法混合可提升长上下文多跳推理。

**🔧 技术方法**

使用技术主要包括：LoRA 参数化适配、不同秩(rank)设置、合成数据生成（GPT‑4.1/ Llama‑3.1）、路由检索（基于嵌入）、多模块合并（线性平均、CAT、TIES、DARE）、与 ICL/RAG 的混合推理。

**📊 数据集**

数据集包括：PhoneBook（键值对）、CounterFact（反事实事实）、PaperQA（论文问答）、NarrativeQA 与 QuALITY（长文本多跳 QA），并设计了 PhoneBook 与 PaperQA 作为针对性基准。

**📈 对比分析**

与全上下文 ICL、标准 RAG、单 LoRA、Multi‑LoRA 以及知识蒸馏 KM_SDCD 等方法对比。实验表明：单 LoRA 在一定规模下能匹配 ICL/ RAG，但在长文本多跳任务中落后；Multi‑LoRA 在理想路由下容量提升显著，但实际路由误差导致性能下降；混合使用 ICL/RAG 与 LoRA 可显著恢复性能，且 LoRA 在多次查询时具有更低的推理延迟。

**⚠️ 局限性**

局限性包括：①对持续更新与时间序列知识的适应性未充分评估；②多模块路由与合并仍受检索误差与干扰影响，需要更稳健的算法；③实验主要在实验室设置下进行，实际系统部署中需要解决模块加载、缓存与 GPU 资源管理等工程挑战。

---

## 384. BLUFF: Benchmarking the Detection of False and Synthetic Content across 58 Low-Resource Languages

**arXiv ID:** 2603.00634 | [PDF](https://arxiv.org/pdf/2603.00634v1)

**作者:** Jason Lucas `[一作]` (Penn State University), Dongwon Lee `[通讯]` (Penn State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BLUFF——一个覆盖78种语言、超过20万条样本的多语言假新闻与人工/AI生成文本检测基准；

**💡 创新点**

首次系统地融合人类事实核查文本与LLM生成文本，构建四种文本类型（HWT、MGT、MTT、HAT）和39种操作技巧；引入AXL‑CoI多代理链式交互生成框架与mPURIFY多维质量过滤；

**🔧 技术方法**

使用19款前沿多语言LLM（GPT‑4、Llama‑3、Qwen‑3等）进行交互式内容生成，利用自对抗 ADIS 技术突破安全对齐；采用多维评估（一致性、验证、翻译、幻觉、结构缺陷）与LLM‑based AEM实现 mPURIFY；

**📊 数据集**

数据来源包括122k人类事实核查新闻（57种语言）和78k由19款LLM生成的假/真文本（71种语言），涵盖 20 头部语言与 58 长尾语言，覆盖 12 语系、9 字符集、6 句法类型；

**📈 对比分析**

在二分类与多分类（8类）语义真伪与作者身份检测任务上，S‑BERT（LaBSE）在多语言训练下达成 97%+macro‑F1，encoder 微调优于零-shot解码器；长尾语言与头部语言之间存在 9–25% 的性能差距，跨语言迁移显著受语法与字符集相似性影响；

**⚠️ 局限性**

局限包括人类样本主要集中在欧洲/南亚，长尾语言覆盖不均；VSO 语法仅以阿拉伯语代表；数据为静态快照，未覆盖时间演变；解码器仅以零-shot评估；

---

## 385. Revealing Combinatorial Reasoning of GNNs via Graph Concept Bottleneck Layer

**arXiv ID:** 2603.02025 | [PDF](https://arxiv.org/pdf/2603.02025v1)

**作者:** Yue Niu `[一作]` (Tongji University), Wei Ye `[通讯]` (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了图概念瓶颈模型（Graph Concept Bottleneck Models, GCBMs），在 GNN 中加入可解释的概念瓶颈层，使模型的预测可被表示为对图概念的软逻辑组合。

**💡 创新点**

创新点在于：①使用图概念瓶颈层将 GNN 的决策过程与图概念直接关联，形成可量化的组合推理；②将 WL 子树作为图概念并通过 Transformer 学习概念嵌入，捕获概念间的共现与语义关联；③实现模型可干预和解释性评估，验证概念的因果影响。

**🔧 技术方法**

技术方法包括： Weisfeiler‑Leman 子树提取、信息增益筛选概念、概念瓶颈层与线性分类器、Transformer 生成概念嵌入、稀疏正则、端到端训练。

**📊 数据集**

使用了 10 个标准图分类数据集（MUTAG、PTC‑MR、BZR、COX2、DHFR、PROTEINS、IMDB‑B、IMDB‑M、COLLAB、DBLP_v1）以及两个大规模不平衡数据集（PC‑3、MCF‑7）。

**📈 对比分析**

与 5 类基线（可解释 GNN、WL 核、最优传输、主流 GNN、图 Transformer）相比，GCBM-E 在大多数数据集上位列前二，单个数据集获得最高准确率；在不平衡数据集上 AUC 也显著优于所有对比方法。

**⚠️ 局限性**

局限性包括：①对概念集的构造依赖 WL 子树，可能不适用于所有图结构；②概念选择仍需要信息增益阈值或超参数调优；③在极大图或复杂多标签任务上，概念瓶颈层的计算成本和可解释性评估仍有待进一步优化。

---

## 386. CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion

**arXiv ID:** 2603.00682 | [PDF](https://arxiv.org/pdf/2603.00682v1)

**作者:** Yushan Han `[一作]` (Beijing Jiaotong University), Yidong Li `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 6972 | [OpenAlex ID](https://openalex.org/A5010019122)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种通信高效的早期协同感知框架CoLC，利用前景感知点采样、LiDAR完成和双向对齐，实现稀疏点云传输下的高精度检测。

**💡 创新点**

创新点在于：①前景感知点采样(FAPS)在前景与背景上采用不同采样策略；②基于VQ的柱体级LiDAR完成(CEEF)恢复稀疏输入的空间完整性；③密集引导双重对齐(DGDA)在语义与几何空间对齐增强特征学习。

**🔧 技术方法**

采用VQ-based pillar completion、Swin Transformer编码、Farthest Point Sampling、Random Sampling、adaptive complementary fusion以及语义/几何对齐损失。

**📊 数据集**

在四个LiDAR 3D检测数据集上评估：V2XSim、OPV2V、V2XSet、DAIR-V2X。

**📈 对比分析**

与基线及中后期融合方法对比，CoLC在保持约10%通信压缩的同时提升AP@0.7 约5%，并在异构模型、姿态误差与通信延迟下保持鲁棒性。

**⚠️ 局限性**

局限性包括：对ICP姿态对齐的依赖、VQ代码库导致模型体积增大，以及在极低带宽下仍略逊于极简后期融合方案。

---

## 387. Generative AI & Fictionality: How Novels Power Large Language Models

**arXiv ID:** 2603.01220 | [PDF](https://arxiv.org/pdf/2603.01220v1)

**作者:** Edwin Roland `[一作]` (University of Illinois), Richard Jean So `[通讯]` (Duke University)

**通讯引用:** 532 | [OpenAlex ID](https://openalex.org/A5049740369)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过比较只用维基百科训练的 BERT 与混合维基百科+小说数据训练的 BERT，探讨小说文本在大规模语言模型中的作用。

**💡 创新点**

首次将小说对模型的语义学习影响量化，发现小说显著提升代词、情态动词及对话等与人物相关的语言特征，并用信息增益证明小说可为模型增加约 1.03 bits/word 的信息。

**🔧 技术方法**

采用 BERT 的掩码语言建模与生成式随机网络（GSN）技术，并计算交叉熵与信息增益。

**📊 数据集**

使用 2021 年 1 月的英文维基百科数据集（约 15 亿词）和 2021 年 3 月从 Smashwords 收集的 BookCorpus 小说数据集（约 5 亿词），总计 2 亿词。

**📈 对比分析**

比较方法包括：掩码预测准确率（Wiki 模型 65% vs Full 模型 63%）、交叉熵差异（1.72 vs 1.88），以及在小说测试集上的信息增益（0.71，转换为 1.03 bits/词）。整体性能显示，小说虽略微降低整体准确率，却在特定词汇上有显著提升。

**⚠️ 局限性**

局限性：仅研究 BERT base 模型，未涉及更大或更新模型；实验仅基于英文数据；评估主要聚焦于掩码预测与生成文本，无法全面捕捉小说对创作质量与多样性的影响；训练数据选择与分布可能影响结果。

---

## 388. Training-Free Spatio-temporal Decoupled Reasoning Video Segmentation with Adaptive Object Memory

**arXiv ID:** 2603.01545 | [PDF](https://arxiv.org/pdf/2603.01545v1)

**作者:** Zhengtong Zhu `[一作]` (Soochow University), Fanzhang Li `[通讯]` (Soochow University)

**通讯引用:** 3203 | [OpenAlex ID](https://openalex.org/A5013713640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的推理视频分割框架SDAM，利用自适应对象记忆与时空解耦实现高质量分割

**💡 创新点**

创新点包括：①自适应运动驱动采样挑选关键帧；②自适应对象记忆模块存储关键对象特征；③时空解耦策略先在空间域定位后在时间域传播，提升时序稳定性

**🔧 技术方法**

核心技术为预训练多模态大语言模型(MLLM)、图像分割模型(SAM)、高效追踪器(Cutie)以及基于运动差异的关键帧筛选与联合置信度判定

**📊 数据集**

使用Ref-YouTube-VOS、Ref-DAVIS17、MeViS、ReasonVOS、ReVOS五个基准数据集进行评测

**📈 对比分析**

相较于现有训练依赖或推理能力有限的方法，SDAM在各数据集上均取得SOTA成绩，Ref-DAVIS17上最高的J&F 76.0%，ReVOS整体约58.0%

**⚠️ 局限性**

主要局限在于关键帧选择的鲁棒性和多关键帧传播的错误累积，需进一步提升对动态场景的适应性

---

## 389. Phase-Type Variational Autoencoders for Heavy-Tailed Data

**arXiv ID:** 2603.01800 | [PDF](https://arxiv.org/pdf/2603.01800v1)

**作者:** Abdelhakim Ziani `[一作]` (Université Paris Saclay), Paolo Ballarini `[通讯]` (Università di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了Phase-Type Variational Autoencoder（PH-VAE），一种能够自适应建模重尾分布的深度生成模型；

**💡 创新点**

创新点在于将Phase-Type（PH）分布作为VAE解码器的可学习概率分布，实现了无需预设极端尾部形式的高灵活性；

**🔧 技术方法**

技术主要包括：VAE框架、PH分布的矩阵指数密度、β‑VAE正则化、稀疏/有序率参数化、统一化法求矩阵指数、对数似然训练；

**📊 数据集**

使用了多种合成重尾数据（Weibull、Pareto、Lognormal、Burr）、保险赔付数据、谷歌词频数据及美国五只股票的绝对日收益；

**📈 对比分析**

与标准Gaussian VAE、t-VAE、x-VAE等基线在单维尾部拟合（KS_tail、Q99误差）以及多维依赖度量（相关矩阵误差、Kendall τ误差、尾部共超概率误差）上比较，PH-VAE在尾部精度、极值估计与跨维依赖方面均显著优于基线；

**⚠️ 局限性**

局限性包括：对PH阶段数和β参数敏感、当前仅适用于非负实值数据、未直接建模极限尾部依赖、对高维复杂结构（如图像）尚未验证。

---

## 390. Learning Shortest Paths with Generative Flow Networks

**arXiv ID:** 2603.01786 | [PDF](https://arxiv.org/pdf/2603.01786v1)

**作者:** Nikita Morozov `[一作]` (HSE University), Sergey Samsonov `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于非循环 GFlowNet 的最短路径学习框架，并在交换谜题与 Rubik’s Cube 上进行实验验证。

**💡 创新点**

理论上证明最小化期望轨迹长度等价于仅采样最短路径；将该结论转化为构造 GFlowNet 解决一般无权图的路径寻找问题；提出带流正则化的轨迹平衡训练方法。

**🔧 技术方法**

使用非循环 GFlowNet、轨迹平衡损失、流正则化、束搜索、MLP（残差+层归一化）网络以及 one‑hot 状态编码。

**📊 数据集**

实验数据集包括：n=15、n=20 的 Swap 交换谜题（≈10^12–10^18 状态），2x2x2 与 3x3x3 Rubik’s Cube（各 100/1000 条测试样本），并与 CayleyPy Cube 进行对照。

**📈 对比分析**

与 CayleyPy Cube 通过不同束宽度比较；在 2x2x2 任务中仅需 1/16 的束宽即可获得最优解；在 3x3x3 任务中小束宽下表现更好；单 GPU 上平均求解时间 1.74 秒，比 CayleyPy 的 6.19 秒快 4 倍，且模型规模更大。

**⚠️ 局限性**

对正则化系数 λ 敏感，过大导致无法找到有效路径；目前仅适用于无权图，扩展到加权或成本敏感情形以及极大图规模仍待研究。

---

## 391. FCN-LLM: Empower LLM for Brain Functional Connectivity Network Understanding via Graph-level Multi-task Instruction Tuning

**arXiv ID:** 2603.01135 | [PDF](https://arxiv.org/pdf/2603.01135v1)

**作者:** Xingcan Hu `[一作]` (University of Science and Technology of China), Li Xiao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 46770 | [OpenAlex ID](https://openalex.org/A5100452145)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出 FCN‑LLM 框架，将大脑功能连接网络（FCN）对齐到文本模态，使大型语言模型（LLM）能够理解并推理 FCN；通过多尺度编码、图层级指令调优和双阶段学习实现零样本泛化；

**💡 创新点**

创新点包括：① 多尺度 FCN 编码器（ROI、功能子网络、全脑）将图特征投射到 LLM 语义空间；② 设计多范式（预测、判断、比较）图层级指令调优，促进知识共享与泛化；③ 双阶段学习（先对齐再微调）避免过拟合，提升零样本性能；④ 通过 prompt‑conditioned attention 解释模型关注的子网络，为可解释性提供依据；

**🔧 技术方法**

技术上使用 2 层 GCN、MLP 投影、Qwen2.5‑3B/7B 预训练 LLM、滑动窗口增强、指令调优（多任务 prompt tuning）、两阶段训练策略；

**📊 数据集**

实验数据来自 10 个公开 rs‑fMRI 数据集（HBN、HCP、QTIM、GSP、ABIDE、ADHD、MDD、SRPBS、ABIDE II、CNP），共 19 个属性（性别、年龄、手偏好、疾病诊断、智力测量、个性等）；

**📈 对比分析**

与 6 种监督 FCN 模型（GCN、HGCN、BrainNetCNN、BrainGNN、Transformer、BNT）及 4 种基础模型（BrainNPT、PTGB、CINP、BrainMass）比较，FCN‑LLM 在内部测试集保持竞争力，在零样本测试集显著优于所有基线，且在疾病、认知、表型等任务上取得最佳零样本性能；

**⚠️ 局限性**

局限性包括：① 对指令调优数据量要求高，可能受数据多样性限制；② 过度微调会导致泛化下降；③ 更大 LLM 并不必然提升 FCN 理解；④ 仅处理静态 FCN，未考虑动态功能连接；⑤ 需要进一步丰富文本知识和任务场景来提升模型能力。

---

## 392. Vision-Language Feature Alignment for Road Anomaly Segmentation

**arXiv ID:** 2603.01029 | [PDF](https://arxiv.org/pdf/2603.01029v1)

**作者:** Zhuolin He `[一作]` (Fudan University), Xiangyang Xue `[通讯]` (Fudan University)

**通讯引用:** 14954 | [OpenAlex ID](https://openalex.org/A5003418019)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于视觉‑语言模型的道路异常分割框架 VL‑Anomaly，利用 PL‑Aligner 对像素和掩膜级特征进行跨模态对齐，并通过多源推理融合检测置信度、文本相似度和 CLIP 图像‑文本相似度，显著提升异常检测性能。

**💡 创新点**

创新点在于：①引入可学习的提示词进行类别级文本嵌入，②设计双层像素‑掩膜级对齐模块 PL‑Aligner，③提出多源推理策略实现不同信源的加权融合。

**🔧 技术方法**

使用 Mask2Former 语义分割网络、CLIP 视觉‑语言模型、学习提示词、跨模态对齐（交叉注意力/MLP）以及多源置信度融合技术。

**📊 数据集**

在 RoadAnomaly、SMIYC（RA21/RO21）和 Fishyscapes（Static/Lost&Found）等公开道路异常分割基准上进行评估。

**📈 对比分析**

与 Mask2Anomaly、ODIN、Mask2Former 等基线对比，VL‑Anomaly 在 AuROC、FPR_95、AuPRC、sIoU、PPV、F1* 等指标均达到或超过前沿水平，显示出更低误报和更高召回。

**⚠️ 局限性**

局限性：推理阶段的权重手工调参，缺乏自动化的权重学习，可能限制在多场景的可扩展性与泛化。

---

## 393. Towards Principled Dataset Distillation: A Spectral Distribution Perspective

**arXiv ID:** 2603.01698 | [PDF](https://arxiv.org/pdf/2603.01698v1)

**作者:** Ruixi Wu `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14624 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了面向长尾分布的数据集蒸馏方法 Class‑Aware Spectral Distribution Matching（CSDM）

**💡 创新点**

创新地将全局可辨别核的谱距离（SDD）与幅相分解相结合，并通过类感知加权实现头尾类别平衡的分布匹配

**🔧 技术方法**

采用 Bochner 定理的核方法、谱分布距离、幅相分解、类加权、CNN 特征提取与随机频率采样技术

**📊 数据集**

在 CIFAR‑10‑LT、CIFAR‑100‑LT、ImageNet 子集（ImageNette、ImageWoof、ImageSquawk）以及标准均衡数据集上进行实验

**📈 对比分析**

与随机、K‑Center、Graph‑Cut、MTT、IDM、DREAM、DATM、LAD、NCFM、RDED 等方法对比，在低 IPC 与高 imbalance factor 情况下，CSDM 领先 12–15% 以上，并在跨架构泛化上表现优异

**⚠️ 局限性**

对核尺度 γ 的选择仍需经验调优，且在极端头尾比例极高时性能仍会略有衰减

---

## 394. UNICBench: UNIfied Counting Benchmark for MLLM

**arXiv ID:** 2603.00595 | [PDF](https://arxiv.org/pdf/2603.00595v1)

**作者:** Chenggang Rong `[一作]` (Northwestern Polytechnical University), Junyu Gao `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 4255 | [OpenAlex ID](https://openalex.org/A5001848378)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 UNICBench，一套跨图像、文本和音频的统一计数基准，包含三层计数能力与难度标签，并提供了标准化的 QA 格式和评估工具。

**💡 创新点**

创新点在于将多模态计数任务统一到单一框架中，定义了 Pattern、Semantic、Reasoning 三个层级的计数能力，并通过可视化、难度阈值和证据优先的标注方式实现跨模态的严格比较。

**🔧 技术方法**

技术上采用统一的系统提示、数值提取规则和多指标评估（SuccessRate、HitRate、MAE/MSE）对 45 款多模态大语言模型进行零/少 shot 计数实验，并对模型内部推理与答案输出进行解析。

**📊 数据集**

使用的数据集涵盖 5,300 张图像（5,508 Q&A）、872 篇文本（5,888 Q&A）和 2,069 条音频（2,905 Q&A），来源包括 FSC‑147、NWPU‑MOC、DESED 等公开数据，并补充了手工标注的多类别样本。

**📈 对比分析**

在统一的推理配置下，作者对 45 款模型进行了对比，发现大部分模型在 Pattern/Semantic 任务中表现较好，但在 Reasoning 与 Hard 难度区间误差显著，说明当前 MLLM 在复杂推理和高密度计数方面仍有提升空间。

**⚠️ 局限性**

局限性包括数据集仍存在类别与场景偏差、缺乏实例级监督导致高密度/遮挡情境下误差高、评估仅关注数值准确性，未覆盖解释性与多模态对齐等方面。

---

## 395. Words & Weights: Streamlining Multi-Turn Interactions via Co-Adaptation

**arXiv ID:** 2603.01375 | [PDF](https://arxiv.org/pdf/2603.01375v1)

**作者:** Chenxing Wei `[一作]` (Shenzhen University), Yao Shu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 12633 | [OpenAlex ID](https://openalex.org/A5100720675)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种同时优化对话上下文（Words）和模型参数（Weights）的测试时多轮交互适应框架（T^2PAM）

**💡 创新点**

创新点在于将错误归因问题拆分为语义歧义与能力缺失两维并联优化，利用文本梯度预处理上下文再同步调整参数，形成全梯度近似的协同适配过程

**🔧 技术方法**

技术上结合了文本梯度(TextGrad)对上下文的微调和参数梯度(ROSA)对适配器权重的更新，并通过一次全梯度近似实现实时迭代

**📊 数据集**

主要使用了数学推理基准（MATH、MATH-500）、通用推理基准（MMLU‑R、SuperGPQA）、多语言推理基准（MT‑AIME24、MT‑MATH100）、代码生成基准（HumanEval）以及UI代理基准（OSWorld、AndroidWorld）

**📈 对比分析**

与基线、单轴方法(TextGrad、ROSA)对比，T^2PAM 在 MATH 上提升约30% 准确率，平均交互轮数减少40%，在所有评测集上均实现或刷新了 state‑of‑the‑art 结果

**⚠️ 局限性**

局限性包括：仍需在每轮推理后进行梯度计算，对低频或长文本推理的实时性能影响尚未完全评估；对极端稀疏奖励环境的泛化性需要进一步验证；理论假设（如平滑性、可微分性）在某些任务中可能不成立

---

## 396. Epistemic Gain, Aleatoric Cost: Uncertainty Decomposition in Multi-Agent Debate for Math Reasoning

**arXiv ID:** 2603.01221 | [PDF](https://arxiv.org/pdf/2603.01221v1)

**作者:** Dan Qiao `[一作]`, Baoxiang Wang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1827 | [OpenAlex ID](https://openalex.org/A5101781010)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对多代理辩论中不确定性进行贝叶斯分解，并通过不确定性引导的多智能体强化学习提升数学推理准确性。

**💡 创新点**

创新点在于将整体不确定性拆分为可通过辩论获取的认识性不确定性与不可通过辩论消除的内在噪声不确定性，并用此拆分结果驱动强化学习，使异构模型辩论效果显著提升。

**🔧 技术方法**

使用贝叶斯不确定性框架、系统级熵分解、IPPO/GRPO 等强化学习技术，辅以置信度调节与信息增益内在奖励。

**📊 数据集**

在 MATH、GSM8K、AMC2023、AIME24/25 等数学推理数据集上进行训练与评估。

**📈 对比分析**

与零射 MAD、标准 IPPO 对比，UMAD 在异构配置下在 T=5 轮时准确率提升约 10%，并在更长轮数下保持稳定性，证明其有效性。

**⚠️ 局限性**

局限在于推理成本随代理数和轮次线性增加；使用 NLL 近似内在噪声，未充分捕捉语义层面的不确定性。

---

## 397. Disk-Resident Graph ANN Search: An Experimental Evaluation

**arXiv ID:** 2603.01779 | [PDF](https://arxiv.org/pdf/2603.01779v1)

**作者:** Xiaoyu Chen `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文通过系统化拆解、统一分类与细粒度实验，对磁盘驻留图基近似最近邻（ANN）搜索方法进行了全面评估，涵盖存储策略、磁盘布局、缓存管理、查询执行与更新机制等五大技术维度，并给出针对不同场景的设计指南；

**💡 创新点**

创新点在于提出了统一的五维技术分类体系，进行细粒度组件评估与全流程性能对比；揭示维度、页面大小、Beam宽度等关键参数对I/O利用率与查询吞吐量的非直观影响；并基于实验结果提供可操作的配置决策树与未来研究方向；

**🔧 技术方法**

主要技术包括：基于PQ压缩的向量存储、耦合/解耦全局布局、ID/启发式/聚类/图复制局部布局、静态/动态/混合缓存策略、计算驱动与I/O驱动异步执行、原地与离线更新机制；实验使用C++实现并在Intel Xeon+SSD平台上测评；

**📊 数据集**

使用了八个公开基准向量数据集：GloVe、SIFT1M、SIFT100M、Deep、Tiny、MSong、GIST 与 OpenAI（3,072维）；

**📈 对比分析**

对比方法包括 DiskANN、FreshDiskANN、AiSAQ、Starling、Gorgeous、PageANN（major-in-disk 与 all-in-disk 版）、PipeANN、OdinANN、DGAI；实验结果显示：major-in-disk（如 PageANN）在低维场景下吞吐量高 1.6–13×；高维场景 PipeANN 更优；I/O利用率普遍低于 15%；页面尺寸从 4KB 递增到 16KB 时吞吐量会骤降；Beam宽度增大至 16 时已基本饱和；缓存策略随维度不同而表现不同；更新方式在查询密集或写密集场景下各有优势；

**⚠️ 局限性**

局限性包括：I/O利用率仍极低，说明磁盘布局仍可进一步优化；major-in-disk 方法显著占用内存，对内存受限环境不友好；性能对向量维度高度敏感，需手动调参；所有实验仅在单机 SSD 上完成，未验证云端多层存储环境；未研究动态重排与在线自适应布局等未来方向。

---

## 398. Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation

**arXiv ID:** 2603.00574 | [PDF](https://arxiv.org/pdf/2603.00574v1)

**作者:** Yongbo He `[一作]` (Zhejiang University), Tao Jin `[通讯]` (Zhejiang University)

**通讯引用:** 159938 | [OpenAlex ID](https://openalex.org/A5019365851)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 DASP 的多模态测试时适应框架，能够在没有源域标签的情况下动态诊断并缓解负迁移与灾难性遗忘。

**💡 创新点**

创新点包括：① 通过在融合层计算特征维度间的冗余得分来准确检测偏置模态；② 设计了异向适配器，将每个模态的自适应模块拆分为稳定和可塑两部分，并根据诊断结果按需激活，避免了对无偏置模态的误适应。

**🔧 技术方法**

使用的技术主要有：冗余得分（inter‑dimensional redundancy）、异向适配器（stable + plastic）、KL 正则化保持稳定性、熵最小化与多样性正则化提升自监督目标、Adam 优化器、批量正则化等。

**📊 数据集**

实验数据集为 Kinetics50-C 与 VGGSound-C（分别在视频与音频模态上添加 15 种视频腐蚀与 6 种音频腐蚀，采用高严重级别），使用 ViT‑based CAV‑MAE 作为预训练模型。

**📈 对比分析**

与 Tent、EATA、SAR、READ、TSA 等现有 TTA 方法进行比较，DASP 在 episodic 适应、持续适应以及交替模态腐蚀场景下均表现出更高准确率，平均提升约 1.6%/5.0%（单模态）和 4.4%/1.5%（交替模态），且保持了更高的推理效率和更低的内存占用。

**⚠️ 局限性**

局限性：对极端或非统一潜在空间结构的多模态模型效果可能受限；冗余阈值 δ 和损失系数 λ_ent/λ_kl 仍需经验调参；在大批量或多尺度腐蚀场景下可能出现性能衰减。

---

## 399. Detection-Gated Glottal Segmentation with Zero-Shot Cross-Dataset Transfer and Clinical Feature Extraction

**arXiv ID:** 2603.02087 | [PDF](https://arxiv.org/pdf/2603.02087v1)

**作者:** Harikrishnan Unnikrishnan `[一作]` `[通讯]` (Orchard Robotics), Harikrishnan Unnikrishnan (Orchard Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对高速度视频内镜（HSV）中的声门分割问题，提出一种检测门控（Detection‑Gate）管线，将YOLOv8检测器与U‑Net分割器结合，实现对非声门帧的自适应抑制，并在有限的患者级别数据上训练得到高精度、实时可用的分割模型；随后利用该模型自动提取声门面积波形并计算临床生物标志物，验证其在健康与病理声学差异中的有效性。

**💡 创新点**

创新点包括：①通过短时保持（4帧≈1 ms）和门控机制，消除非声门帧的伪分割，提高视频连续性；②采用crop‑zoom变体将检测框裁剪放大，提升跨机构、跨相机的泛化能力；③实现零样本（zero‑shot）跨数据集迁移，避免再训练；④在保持模型轻量化（约11 M参数）与实时性（≈35 fps）之间取得平衡。

**🔧 技术方法**

技术手段包括：YOLOv8n检测器（训练2轮）；U‑Net分割器（灰度输入、BCE+Dice损失、AdamW+cosine annealing）；Temporal Consistency Gate；Crop‑Zoom预处理；零样本评估策略；基于分割结果计算声门面积波形、开放商数、变异系数等特征；统计检验（Mann‑Whitney U）。

**📊 数据集**

使用数据集：①“G‑LA”临床数据集（65名患者，共600训练帧、80验证/测试帧），用于模型训练与小样本验证；②“BAGLS”公共基准（55,750训练帧、3,500测试帧），用于零样本跨域评估与最终对比。

**📈 对比分析**

评估方法：在GLAD测试集上与公开基线对比（DSC、DSC≥0.5）；在BAGLS上进行零样本测试（Det‑Recall、DSC、IoU、DSC≥0.5）并在最佳阈值下优化；再将模型在BAGLS上再训练后评估。结果显示：U‑Net单体在GLAD上达DSC 0.81，检测门控管线为DSC 0.75；在BAGLS零样本时，crop‑zoom管线DSC 0.64（阈值0.02），相较于单体U‑Net 0.59明显提升；在BAGLS再训练后，DSC 0.85。临床特征验证中，变异系数（CV）在女性组显著区分健康与病理（p=0.006）。

**⚠️ 局限性**

局限性：①样本量小且性别失衡（健康组女性占80%，病理组男性占56%），男性健康样本仅3例，影响统计显著性；②仅在公开数据集上验证，未覆盖真实临床环境中的多种光照、镜头运动条件；③门控阈值和保持窗口对不同帧率/相机系统的适用性需进一步调优。

---

## 400. Agent-Based Simulation of Trust Development in Human-Robot Teams: An Empirically-Validated Framework

**arXiv ID:** 2603.01189 | [PDF](https://arxiv.org/pdf/2603.01189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 401. Strategic Advice in the Age of Personal AI

**arXiv ID:** 2603.02055 | [PDF](https://arxiv.org/pdf/2603.02055v1)

**作者:** Yueyang Liu `[一作]` (Rice University), Wichinpong Park Sinchaisri `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了个人 AI 助手在不确定性咨询场景下对机构/专业顾问的战略影响，建立了顾问如何在顾客可能或可能不咨询个人 AI 的情况下，调节其建议以接近目标的理论模型。

**💡 创新点**

创新点在于：① 将咨询概率（广义边际）与信任权重（内部边际）两维度分解为顾问面临的两条策略通道；② 发现顾问的“对抗强度”随个人 AI 采用率单调增加，但顾问损失呈拱形曲线；③ 证明顾问对抗仅通过相对信任比率决定，并且引入了信任作为可投资的战略工具，说明个人 AI 的采用会改变对信任投资的激励。

**🔧 技术方法**

采用了贝叶斯更新与期望均方误差最小化的分析框架，求解顾问最优推荐的闭式表达式，并通过比较静态分析得到对抗强度和损失的单调性及拱形特征。

**📊 数据集**

无数据集，本文为完全理论模型，未涉及实证检验。

**📈 对比分析**

未进行经验对比或性能评估；所有结论均来自理论推导与比较静态分析。

**⚠️ 局限性**

局限性包括：假设个人 AI 的输出可被顾问完全预测、忽略顾问对推荐内容的实际可变性、未考虑监管、道德或声誉约束对策略的进一步限制、以及模型只处理单一顾问与个人 AI 的双源情形，未涵盖多顾问或多 AI 的复杂交互。

---

## 402. OpenAutoNLU: Open Source AutoML Library for NLU

**arXiv ID:** 2603.01824 | [PDF](https://arxiv.org/pdf/2603.01824v1)

**作者:** Grigory Arshinov `[一作]` (MWS AI), Leonid Sanochkin `[通讯]` (MWS AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个低代码、自动化的NLU库OpenAutoNLU，支持文本分类和命名实体识别，自动选择训练模式、集成数据质量检测和OOD检测，并利用LLM进行数据增强与合成测试集。

**💡 创新点**

创新点在于基于数据规模的无缝训练模式选择、统一的低代码API、可配置的OOD检测层、以及LLM驱动的数据增强和合成评测。

**🔧 技术方法**

采用Transformer微调、SetFit/AncSetFit少样本学习、Optuna超参搜索、Augmentex文本增强、LLM生成数据、集成多种OOD检测方法（Mahalanobis、Softmax、logit-based）等技术。

**📊 数据集**

使用了银行77、Massive、HWU64、Snips等意图分类数据集，并支持NER的BIO标注。

**📈 对比分析**

与AutoIntent、AutoGluon、LightAutoML、H2O AutoML等框架在同一基准下对比，OpenAutoNLU在大部分数据集上实现了宏F1相当或更优，尤其在低资源场景下表现突出，同时保持更低的训练时延。

**⚠️ 局限性**

局限在于目前的训练模式选择仅基于样本数阈值，缺乏更全面的元学习策略；未来计划通过dataset2vec等抽象特征构建更智能的组合策略。

---

## 403. Machine Learning Grade Prediction Using Students' Grades and Demographics

**arXiv ID:** 2603.00608 | [PDF](https://arxiv.org/pdf/2603.00608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 404. TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward Pose-free Reconstruction

**arXiv ID:** 2603.00697 | [PDF](https://arxiv.org/pdf/2603.00697v1)

**作者:** Yihui Li `[一作]` (Beihang University), Di Huang `[通讯]` (Beihang University)

**通讯引用:** 11520 | [OpenAlex ID](https://openalex.org/A5056972984)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了一种无姿态的多视角图像输入下的前向3D高斯重建与相机姿态估计框架TokenSplat。

**💡 创新点**

创新点在于Token对齐的高斯预测模块和非对称双流解码器(ADF-Decoder)实现了视角与场景特征解耦、跨视角特征融合与单次前向推理。

**🔧 技术方法**

使用ViT编码器、跨视角自注意力、Token对齐高斯预测头、ADF-Decoder、相机token、单位四元数对齐损失等技术。

**📊 数据集**

在RE10K和ScanNet两个真实场景数据集上进行训练与评估。

**📈 对比分析**

与Pose-Required的MVSplat/FreeSplat以及Pose-Free的NoPoSplat/VicaSplat/SPFSplat/AnySplat比较，TokenSplat在PSNR/SSIM/LPIPS上显著领先，且在姿态估计上RPE-r/ATE明显下降。

**⚠️ 局限性**

局限在于对稀疏视角下的高斯预测仍受限、对极端光照或遮挡场景鲁棒性待进一步提升，且模型规模较大。

---

## 405. From Transportation to Manipulation: Transforming Magnetic Levitation to Magnetic Robotics

**arXiv ID:** 2603.01982 | [PDF](https://arxiv.org/pdf/2603.01982v1)

**作者:** Lara Bergmann `[一作]` (Bielefeld University), Klaus Neumann `[通讯]` (Fraunhofer IOSB-INA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一个六自由度磁悬浮并联平台（6D-Platform MagBot）并配备自锁式对接站，实现了搬运与操作一体化的磁力机器人；

**💡 创新点**

创新点在于将两台MagLev移动装置通过并联机构耦合，显著扩展工作空间、负载与姿态自由度，且不需要额外电子；同时提出了针对该平台的逆运动学控制器以及基于低级控制器扭矩的负载定位方法；

**🔧 技术方法**

采用磁悬浮驱动、并联四杆机构、齿轮与链条传动、逆运动学求解、MuJoCo仿真、VICON姿态测量及低级控制参数调优等技术；

**📊 数据集**

实验基于Beckhoff XPlanar系统的APM4330/APS4322磁块与移动装置，采集VICON误差、低级扭矩等数据；未使用公开数据集；

**📈 对比分析**

通过与单个mover在工作空间、负载、定位精度和吞吐率等方面的对比实验，展示平台在z方向达到205-280 mm、α/β±14°、负载2 kg、定位误差<0.7 mm/1.1°，吞吐率约14件/分钟；实验结果验证了显著性能提升；

**⚠️ 局限性**

局限在于3D打印部件导致α/β旋转误差、低级控制参数需要人工调优、磁悬浮系统对角度受限（±10°）限制全360°旋转、对接需要高精度对齐且缺乏在线自适应控制与人机交互等进一步研究方向。

---

## 406. The Impact of Battery Cell Configuration on Electric Vehicle Performance: An XGBoost-Based Classification with SHAP Interpretability

**arXiv ID:** 2603.01275 | [PDF](https://arxiv.org/pdf/2603.01275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 407. Learning to Explore: Policy-Guided Outlier Synthesis for Graph Out-of-Distribution Detection

**arXiv ID:** 2603.00602 | [PDF](https://arxiv.org/pdf/2603.00602v1)

**作者:** Li Sun `[一作]` (North China Electric Power University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 134323 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种无监督图 OOD 检测框架 PGOS，先用原型对齐的图对比学习构建结构化潜在空间，再用强化学习代理在低密度区探索并合成伪 OOD 图，从而提升检测性能。

**💡 创新点**

创新点包括：① 将原型对齐与对比学习结合，形成可被主动探索的结构化潜在空间；② 设计可自适应的奖励、硬边界约束和空间感知熵正则的 RL 策略，让代理主动采样最具信息量的伪 OOD；③ 将生成的伪 OOD 与原始 ID 训练联合，形成端到端的 OOD 正则化过程。

**🔧 技术方法**

主要技术：原型对齐图对比学习（PGCL）、图自编码器、强化学习（SAC）策略学习、空间感知熵正则化、伪 OOD 解码、OOD 评分模型（如 GOOD-D）和 OOD 正则化损失。

**📊 数据集**

使用 10 个 OOD 检测基准（TU 图数据集、Tox21、COX2、MUTAG 等）以及 15 个图异常检测基准（PROTEINS、ENZYMES、AIDS 等）进行评估。

**📈 对比分析**

与 15 种竞争基线（图核、SSL、GNN 异常检测和 OOD 检测方法）对比，PGOS 在 10 个 OOD 检测基准上平均排名 1.9，AUC 最高可达 96.9%，在 7/15 个异常检测基准上取得 SOTA；总体上显著优于传统基于 ID 训练或静态采样的方案。

**⚠️ 局限性**

局限性：① 需要先行训练良好的原型对齐模型，对原型数量等超参敏感；② RL 采样过程计算开销较大，训练稳定性依赖奖励与熵正则设计；③ 目前仅在图数据上验证，跨模态或更大规模图的泛化尚未深入探讨。

---

## 408. SafeSci: Safety Evaluation of Large Language Models in Science Domains and Beyond

**arXiv ID:** 2603.01589 | [PDF](https://arxiv.org/pdf/2603.01589v1)

**作者:** Xiangyang Zhu `[一作]` (Shanghai AI Lab), Guangtao Zhai `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SafeSci 框架，包含 SafeSciBench（250K多学科安全评测基准）和 SafeSciTrain（150万安全指令微调数据集），对 LLM 在科学领域的安全知识与风险识别进行系统评估与提升。

**💡 创新点**

创新点在于：① 明确区分安全知识与安全风险两大维度；② 采用多学科、任务多样化的客观评测（MCQ、TF、生成等）消除评测主观性；③ 提供大规模安全训练集，验证微调能显著提升安全对齐。

**🔧 技术方法**

使用模板生成与双代理自动化生成方法构造问题；评估指标包括准确率、拒绝率、分子/蛋白/核酸生成的有效性与相似度；微调采用 LoRA 方法。

**📊 数据集**

数据集来源包括 PubChem、OpenFoodTox、UniProt、GeneBank、DrugBank、WMDP、SciSafeEval 等，结合 125 个任务覆盖化学、生命、医学、材料、工程、物理、心理等七个学科。

**📈 对比分析**

与 24 种 LLM（开源与专有）进行零样本评测，结果显示大模型在安全知识上有差距，安全风险识别不随参数规模提升；微调后知识准确率和拒绝率均显著提升，尤其是 Qwen3-8B 的安全拒绝率从 0.31 提升至 0.64。

**⚠️ 局限性**

主要局限在：① 知识与风险边界模糊导致过度拒绝（over‑refusal）；② 训练过程未显式区分两类信息，导致微调后仍存在误拒；③ 评测仅针对确定性答案，缺乏对含有安全提示的生成回复的客观评估。

---

## 409. PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation

**arXiv ID:** 2603.01593 | [PDF](https://arxiv.org/pdf/2603.01593v1)

**作者:** Bo Ma `[一作]` (Auckland University of Technology), Minh Nguyen `[通讯]` (Auckland University of Technology)

**通讯引用:** 88899 | [OpenAlex ID](https://openalex.org/A5108111223)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对驾驶摄像头视频的背景匹配攻击，设计了一种基于动态条件随机场（PPEDCRF）的隐私保护框架，能在保持目标检测/分割性能的同时，只对定位敏感的背景区域注入噪声，从而降低位置泄露风险。

**💡 创新点**

创新点包括：①使用动态条件随机场实现跨帧的空间-时间一致性检测敏感区域；②提出 Normalized Control Penalty（NCP）机制，根据层级敏感度精准调节噪声强度；③将噪声注入限定在特征空间，避免对模型梯度或整帧图像造成过度破坏；④可选的特征反演重构，实现对视频内容的可发布性。

**🔧 技术方法**

核心技术：动态条件随机场（DCRF）、Normalized Control Penalty（NCP）、Gaussian噪声注入、特征空间插值重构、YOLOv4/Faster R‑CNN目标检测、DeepLabV3分割、KL散度评估、Top‑k检索等。

**📊 数据集**

实验数据集：MOT16/17（目标检测/分割）、Cityscapes、KITTI、VOC2008（基准验证）、Street‑View数据库（位置泄露评估）。

**📈 对比分析**

与全局噪声、白噪声掩码、特征匿名化等基线相比，PPEDCRF 在 Top‑k 检索准确率明显下降（例如 Top‑k 召回率从 0.9 降至 0.3），同时目标检测的 mAP、GIoU、Precision、Recall 等指标保持在 0.8‑0.9 之间，甚至在 YOLOv4+PPEDCRF+NCP 的组合下，GIoU 和 Precision 更优。整体性能表现优于常见隐私保护方法。

**⚠️ 局限性**

局限性：①噪声生成与 NCP 计算的时间开销较大，影响实时性；②部分指标（如召回率、精度）在加入噪声后略有下降；③目前为两阶段实现，未实现单阶段端到端加速；④对极端天气或低分辨率场景的鲁棒性尚未充分验证。

---

## 410. Fed-GAME: Personalized Federated Learning with Graph Attention Mixture-of-Experts For Time-Series Forecasting

**arXiv ID:** 2603.01363 | [PDF](https://arxiv.org/pdf/2603.01363v1)

**作者:** Yi Li `[一作]`, Biplab Sikdar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向时间序列预测的个性化联邦学习框架 Fed‑GAME，通过服务器端的 Graph Attention Mixture‑of‑Experts（GAME）聚合器实现动态、内容感知的客户端信息交流。

**💡 创新点**

创新点在于：① 用参数差异而非完整模型进行通信，分离全局共识与个性化；② 在服务器端构建可学习的隐式图网络，利用共享专家和个性化门控实现多专家聚合；③ 通过相似度基 meta‑loss 在不访问本地数据的前提下训练聚合器；④ 仅上传最终 MLP 层的差分以降低通信成本。

**🔧 技术方法**

采用的技术包括：联邦学习（FedAvg、FedProx 等）、LSTM+MLP 时序模型、量化回归（Quantile Regression）、图注意力网络、混合专家（MoE）与噪声 Top‑k 门控、相似度基 meta‑loss、通信成本分析。

**📊 数据集**

使用了两套电动汽车充电数据集：Palo Alto（8 站，5 分钟）和 Shenzhen（247 站，30 分钟）。

**📈 对比分析**

与 FedAvg、FedProx、pFedMe、PAG‑FedAvg、GCRN‑FedAvg 以及仅本地训练（No_FL）进行对比；Fed‑GAME 在 QS、ICP 等指标上实现了 54% 以上的提升，并且 MIL 指标略低但仍优于大多数基线；通信成本仅比全量上传高 0.1%–0.2%。

**⚠️ 局限性**

局限性包括：① MIL 性能受限于只上传差分；② 仅聚焦最终层更新，可能忽略低层特征；③ 仅在两组充电数据上验证，未评估对更大规模或其他领域的适用性；④ 需要预训练共享专家，若任务分布剧烈变化可能需要重新训练。

---

## 411. DWAFM: Dynamic Weighted Graph Structure Embedding Integrated with Attention and Frequency-Domain MLPs for Traffic Forecasting

**arXiv ID:** 2603.00997 | [PDF](https://arxiv.org/pdf/2603.00997v1)

**作者:** Sen Shi `[一作]` (Nanjing University of Information Science and Technology), Yangfan He `[通讯]` (Nanjing Institute of Technology)

**通讯引用:** 983 | [OpenAlex ID](https://openalex.org/A5028171572)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种动态加权图结构嵌入（DWGS）并将其与注意力机制和频域MLP相结合，构建了DWAFM交通预测模型

**💡 创新点**

创新点在于：①利用自注意力学习时间变化的图权重，生成真实反映节点关联强度的动态加权邻接矩阵；②结合空间‑时间自适应嵌入和频域MLP实现更高效的空间与时间依赖建模；③整体框架保持轻量化，兼顾性能与计算效率

**🔧 技术方法**

使用的技术包括自注意力、动态加权邻接矩阵学习、空间‑时间自适应嵌入、频域（FFT/IFFT）MLP、残差与层归一化、1D‑CNN压缩/扩展等

**📊 数据集**

实验数据集：三条高速公路流量数据集（PEMS03、PEMS04、PEMS08）和两条速度数据集（PEMSD7(L)、PEMSD7(M)）

**📈 对比分析**

与传统方法（HI）、STGNNs（STGCN、GWNet、StemGNN等）、Transformer（STAEformer）和近期方法（DGCRN、MegaCRN、STWave、DFDGCN）等进行对比；在MAE、RMSE、MAPE等指标上，DWAFM在多数数据集上均取得最优或第二优成绩，且在计算效率上优于Transformer架构

**⚠️ 局限性**

限制：在极低流量或速度的极端值时MAPE偏高；模型对超参数（embedding维度、注意力缩放因子）较敏感；对节点初始化与图结构先验仍有一定依赖

---

## 412. The Disintegration of Free Speech

**arXiv ID:** 2603.00754 | [PDF](https://arxiv.org/pdf/2603.00754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 413. STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification

**arXiv ID:** 2603.00695 | [PDF](https://arxiv.org/pdf/2603.00695v1)

**作者:** Xingguo Xu `[一作]` (Dalian University of Technology), Dell Zhang `[通讯]` (China Telecom)

**通讯引用:** 1888 | [OpenAlex ID](https://openalex.org/A5015784640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出STMI框架，结合分割引导特征调制、可学习查询的语义令牌再分配和跨模态超图交互，实现多模态ReID的前景增强、信息保留与高阶语义关联。

**💡 创新点**

创新点在于利用SAM生成的分割掩码进行前景特征调制、引入可学习查询令牌避免硬剪裁导致的信息丢失，并通过超图卷积实现跨模态的高阶语义交互。

**🔧 技术方法**

采用SAM分割、CLIP预训练视觉/文本编码器、跨注意力机制、超图卷积、标签平滑交叉熵与三元组损失等技术组合。

**📊 数据集**

在RGBNT201、RGBNT100、MSVR310三大公开多模态ReID数据集上进行实验验证。

**📈 对比分析**

与多种CNN/ViT/CLIP基线对比，STMI在mAP与Rank-1指标上分别达到81.2%/83.4%等，显著优于IDEA、TOP-ReID等前沿方法。

**⚠️ 局限性**

仍依赖高质量的SAM分割与多模态文本生成，且在极端遮挡或模态缺失场景下跨模态对齐效果可能受限。

---

## 414. Planning Method for Skill-Based Control of Robots Using a PLC as Skill Trigger

**arXiv ID:** 2603.00555 | [PDF](https://arxiv.org/pdf/2603.00555v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 415. Modeling and Analysis of Fish Interaction Networks under Projected Visual Stimuli

**arXiv ID:** 2603.01682 | [PDF](https://arxiv.org/pdf/2603.01682v1)

**作者:** Hiroaki Kawashima `[一作]` (University of Hyogo), Saeko Takizawa `[通讯]` (University of Hyogo)

**通讯引用:** 4672 | [OpenAlex ID](https://openalex.org/A5029640488)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种在传统吸引型交互模型中加入视觉刺激项的扩展模型，并基于该模型从鱼群轨迹数据中实时估计动态交互网络，进一步定义了群体级和个体级的可解释指标，用以量化协同运动与刺激响应；

**💡 创新点**

创新点在于：①将视觉刺激方向显式纳入交互方程，分离内部协调与外部刺激的相对贡献；②采用稀疏线性回归得到可解释的权重矩阵；③构建可在实时环境下计算的群体协调强度、刺激响应度、个体影响力与熵化指标，实现对领导力动态分布和行为同质化的定量描述；

**🔧 技术方法**

技术手段包括：稀疏回归（L1正则化）估计交互权重；YOLO11目标跟踪+手工校正提取轨迹；离散化的线性动力学模型；基于时间窗的滑动估计；熵化指标与统计检验（Kruskal‑Wallis、Dunn法）评估结果显著性；

**📊 数据集**

数据集为实验室投影刺激下的五尾鱼（rummy‑nose tetras）轨迹数据：每个实验包含40 cm×40 cm浅水箱、5条鱼、三种旋转速度（S1≈0.286°/帧，S2≈0.572°/帧，S3≈1.144°/帧）以及对照条件，共三天×三次，每次录制60 fps的60 s视频（中段35 s用于分析）；

**📈 对比分析**

通过比较不同刺激强度下的群体指标（S_att、S_stim、H_influ、H_stim）以及个体影响力时间序列，利用非参数统计检验验证指标随刺激变化的显著性；结果表明模型能够清晰捕捉到领导力的短时转移、内部动力学与外部刺激的耦合关系，并在高强度刺激下显示出行为同质化，证明其在实时辨识与量化协同运动方面的有效性；

**⚠️ 局限性**

局限性包括：①仅考虑二维平面运动，缺乏对三维鱼群的推广；②模型仅包含吸引项，未加入对齐或斥力项，可能限制对更复杂鱼群行为的解释；③实验规模有限（仅5条鱼），对大规模群体的可扩展性未验证；④对刺激形式的适用性局限于可投影的视觉模式，其他外部刺激需要进一步研究；

---

## 416. Behavioral Outcomes of Human Cognitive Security within an Integrative Modeling Framework

**arXiv ID:** 2603.01355 | [PDF](https://arxiv.org/pdf/2603.01355v1)

**作者:** Aaron R. Allred `[一作]` (University of Colorado), Allison P. A. Hayman `[通讯]` (University of Colorado)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5057867194)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文提出了“认知安全”这一人类层面构念，并基于贝叶斯推理与情绪调节的决策价值整合，构建了一个能把信息威胁映射到可观测行为（真实性判断、任务行动和信息分享）的统一建模框架。

**💡 创新点**

创新点在于首次将认知资源分配、情绪评估与源可信度结合起来，形成可量化的认知安全度量，并通过模拟证明其能解释经典现象（如幻真效应、真实性判断与分享行为不一致）。

**🔧 技术方法**

采用了贝叶斯推理与累积前景理论的组合，利用资源映射函数描述信息处理，进而通过行动价值函数和选择概率映射实现行为预测。

**📊 数据集**

使用的主要数据来自先前关于幻真效应的实验（重复陈述的真值评估）以及模拟产生的理想实验数据；模型通过对比这些数据进行验证。

**📈 对比分析**

模型对幻真效应的预测与实验结果的拟合度高，R²≈0.86（验证数据），在真实性判断与分享行为模拟中也能重现误导信息分享与正确判断不一致的模式。

**⚠️ 局限性**

局限性包括：认知资源与情绪维度的多重性未被完全捕捉；模型仍以模拟为主，缺乏大规模实证验证；以及对团队层面和不同信息环境的适用性仍需进一步研究。

---

## 417. Towards Universal Khmer Text Recognition

**arXiv ID:** 2603.00702 | [PDF](https://arxiv.org/pdf/2603.00702v1)

**作者:** Marry Kong `[一作]` (Techo Startup Center, Ministry of Economy and Finance), Koichi Kise `[通讯]` (Osaka Metropolitan University)

**通讯引用:** 3287 | [OpenAlex ID](https://openalex.org/A5000232184)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了统一的柬埔寨语文本识别框架（UKTR），同时构建了新的场景和手写文本数据集，并通过跨模态适配训练实现了在文档、场景和手写文本上的高精度识别。

**💡 创新点**

核心创新点包括：①模态感知自适应特征选择（MAFS）模块，能够根据输入图像的模态动态选择最相关的视觉特征；②在同一模型中同时部署CTC和Transformer解码器，实现低延迟与高精度的可切换；③针对低资源柬埔寨语构建了首个多模态场景与手写数据集，填补了数据空缺；④通过统一模型与跨模态学习避免了多模型部署的存储与路由成本。

**🔧 技术方法**

技术方案包括：基于ResNet+Transformer的视觉编码器；MAFS模块由全局池化、路由器、适配器和聚合器组成；联合CTC与Transformer解码器；使用循环学习率、梯度裁剪等训练技巧；对不同模态数据进行混合采样以保持泛化。

**📊 数据集**

使用的数据集：已有的合成文档与场景集（Buoy et al., SynthText, HierText）、真实场景集（KhmerST, WildKhmerST）、真实手写集（KH）、以及本文新构建的普通柬埔寨语场景集（GKST）和手写集（KHT）。

**📈 对比分析**

与现有方法比较，UKTR在所有评测集上均取得了最低的字符错误率（CER），如Transformer解码器在KHOB、KhmerST、GKST、KHT上的CER分别为2.37%、2.19%、3.34%和6.10%；CTC解码器在速度上更快，但精度略逊；通过消融MAFS和调节模态源数均验证了其对性能的正向影响，证明了跨模态学习的有效性。

**⚠️ 局限性**

局限性包括：①路由器仅通过整体识别损失训练，缺乏明确的模态监督；②解码器在推理时未根据图像动态选择，导致延迟与精度的权衡未能完全优化；③模型仍主要针对柬埔寨语，跨语言推广需进一步研究。

---

## 418. IDProxy: Cold-Start CTR Prediction for Ads and Recommendation at Xiaohongshu with Multimodal LLMs

**arXiv ID:** 2603.01590 | [PDF](https://arxiv.org/pdf/2603.01590v1)

**作者:** Yubin Zhang `[一作]` (Xiaohongshu Inc.), Yao Hu `[通讯]` (Xiaohongshu Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用多模态大语言模型生成代理嵌入并与现有ID嵌入空间对齐，实现新商品在无交互历史情况下的高质量点击率预测。

**💡 创新点**

首次结合粗粒度MLLM生成、细粒度多层隐藏状态端到端对齐以及CTR模型结构先验，实现冷启动CTR预测的显著提升。

**🔧 技术方法**

InternVL等多模态LLM、对比学习、轻量级多粒度适配器以及CTR模型的端到端联合训练。

**📊 数据集**

小红书海量生产数据，包括数亿用户-物品交互记录及新发布的帖子和广告。

**📈 对比分析**

与工业基线及多模态对齐方法对比，Stage1提升AUC 0.05%，Stage2再提升0.14%；线上A/B测试全量流量提升0.12%~0.15%，新内容提升0.23%~0.32%，广告指标亦同步提升。

**⚠️ 局限性**

对MLLM推理成本和嵌入分布稳定性敏感，且在极低交互场景下提升空间仍存在。

---

## 419. DRIV-EX: Counterfactual Explanations for Driving LLMs

**arXiv ID:** 2603.00696 | [PDF](https://arxiv.org/pdf/2603.00696v1)

**作者:** Amaia Cardiel `[一作]` (Aptikal), Eric Gaussier `[通讯]` (Aptikal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DRIV-EX框架，利用梯度优化连续嵌入并在受控解码中重生成最小语义扰动，用于解释LLM在自动驾驶中的决策过程。

**💡 创新点**

创新点在于将梯度搜索与离散文本生成分离：先通过soft embedding的梯度更新识别决策边界，再将优化后的嵌入作为语义引导，借助受控自回归解码生成既保持流畅性又与原输入高度相近的对抗性示例。

**🔧 技术方法**

核心技术包括梯度下降、straight-through估计、soft embedding投影、受控自回归解码、BERTScore相似度评估、模板符合度评估以及多重正则化策略。

**📊 数据集**

实验数据来自LC-LLM规划器所使用的文本化HighD高速公路数据集。

**📈 对比分析**

与DAB、PEZ等基线相比，DRIV-EX在决策翻转成功率（P@1）、BERT相似度、模板符合度及聚合成功率（Aggregated Score）等指标上均表现更好，尤其在暴露潜在安全风险的“Aggreg & Col”上显著提升。

**⚠️ 局限性**

局限性包括仅适用于文本输入、需要白盒模型、固定序列长度不支持插入删除、仅在HighD文本模板上验证、计算成本相对较高、难以推广到无结构文本以及聚合成全局解释的挑战。

---

## 420. GMP: A Benchmark for Content Moderation under Co-occurring Violations and Dynamic Rules

**arXiv ID:** 2603.01724 | [PDF](https://arxiv.org/pdf/2603.01724v1)

**作者:** Houde Dong `[一作]` (Beijing University of Posts and Telecommunications), Jie Hao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Generalized Moderation Policy (GMP) 基准，用于评估内容审核模型在检测共现违规和适应动态规则两大维度上的泛化能力。

**💡 创新点**

创新点在于：① 将违规标签设计为多标签、共现模式，突破传统互斥分类；② 引入动态规则集，让模型在零样本情境下对不同平台、时间、身份的政策进行实时推理；③ 通过 LLM 委员会 + 人工仲裁的自动化标注流程，构建大规模可靠数据；④ 设计专门的评估指标（Coverage、Safety Accuracy 等）揭示模型在长尾违规和规则适配上的盲点。

**🔧 技术方法**

技术手段包括：多模型 LLM 委员会进行自动标注与难度分层；链式思考 (CoT) 与思考模式的 ablation；基于默认许可原则的规则拆解与组合；多种评估指标（Micro‑F1、Macro‑F1、Coverage、Precision/Recall、Latency/Cost）对模型进行全维度打分。

**📊 数据集**

使用公开的内容审核数据集（如 ChineseHarm‑Bench、THOS、STATE ToxiCN 等），通过 LLM 增强与合并产生 5,155 条样本，随后挑选 1,400 条用于共现违规任务、2,000 条用于动态规则任务；标注涵盖 5 大行为类与 10 大范围类，形成多标签与原子规则对。

**📈 对比分析**

与 20+ 主流 LLM（GPT‑4o、Claude‑Sonnet‑4、Grok‑3、Gemini‑2.5‑Pro 等）进行基准测试。结果显示：前沿模型虽能捕捉高频违规，但 Macro‑F1 与 Coverage 低于 Micro‑F1，表明长尾违规漏检严重；在动态规则任务中，平均 F1 仅约 0.55‑0.60，模型在规则冲突时易被安全先入观念干扰。成本与延迟方面，前沿模型性能最高但成本/延迟显著高于中等规模模型。

**⚠️ 局限性**

局限性：① 仅覆盖英文文本，缺少多模态、跨语言与高度细化子社区的情景；② 标注流程虽自动化但无法完全排除预训练语料泄露风险；③ 动态规则覆盖范围虽广，但未必包含所有极端或逆向的社区规范，未来需要更细粒度、跨领域的规则扩展。

---

## 421. Intent-Context Synergy Reinforcement Learning for Autonomous UAV Decision-Making in Air Combat

**arXiv ID:** 2603.00974 | [PDF](https://arxiv.org/pdf/2603.00974v1)

**作者:** Jiahao Fu `[一作]` (Northwestern Polytechnical University), Feng Yang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 36108 | [OpenAlex ID](https://openalex.org/A5075132465)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并验证了一种基于意图预测与情境分析相结合的强化学习框架ICS‑RL，用于无人机在高动态空战中的自主渗透决策。

**💡 创新点**

创新点包括：①使用LSTM对敌方轨迹进行意图预测并通过状态扩展实现主动规划；②将任务拆分为安全巡航、预防隐蔽和突破三大情境，训练专属Dueling‑DQN专家并通过优势值动态切换；③整合意图预测与情境专家的优势融合，形成真正的意图‑情境协同决策。

**🔧 技术方法**

技术栈：强化学习（Dueling‑DQN、DRQN）、LSTM意图预测、优势值动态切换、状态扩展、经验回放、TD误差更新；实验使用Python、PyTorch、OpenAI‑Gym样式仿真环境。

**📊 数据集**

使用自行构建的高保真仿真数据集：10 km×10 km战区、1架友机、5架敌机、目标区，包含动态敌方巡逻、雷达检测与攻击模型；不使用公开真实数据集。

**📈 对比分析**

与标准DDQN、情境分析DDQN（无意图预测）、粒子群优化（PSO）、博弈论方法进行对比；ICS‑RL在50次蒙特卡洛测试中获得88%成功率、平均暴露0.24次/回合、意图预测准确率80.2%，显著优于其他基线。

**⚠️ 局限性**

局限性：仅在仿真环境验证；意图预测对敌方策略变化的鲁棒性未知；计算量相对较大，实时部署需硬件支持；缺乏实战验证与多机协同扩展。

---

## 422. Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale

**arXiv ID:** 2603.02176 | [PDF](https://arxiv.org/pdf/2603.02176v1)

**作者:** Hao Li `[一作]` (Shanghai Artificial Intelligence Laboratory), Shuyue Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AgentSkillOS 框架，组织并编排大规模技能生态系统以自动完成多模态任务。

**💡 创新点**

创新点在于将技能组织成层次化的能力树进行检索，并通过 DAG 结构实现多技能的有序编排，解决了技能碎片化与缺乏结构化组合的问题。

**🔧 技术方法**

使用了 LLM（Claude Opus 4.5）驱动的节点分类与分组、基于能力树的检索、DAG 编排、LLM 评判与 Bradley‑Terry 模型、Claude Code Agent SDK（Sonnet 4.5）等技术。

**📊 数据集**

使用了 30 个人工构建的多格式创意任务基准（涵盖数据计算、文档创建、动画视频、视觉设计、网页交互），以及公开的技能市场与 GitHub 公开技能集合（规模 200 / 1K / 200K）。

**📈 对比分析**

通过 LLM 对二进制比较结果聚合至 Bradley‑Terry 模型，AgentSkillOS 在所有规模下 BT 分数均为 100，明显优于平面调用、全池调用和无技能基线；在 1K 与 200K 规模下，Efficiency‑First 与 Simplicity‑First 也能保持较高得分。

**⚠️ 局限性**

限制在于依赖手工挑选的技能集、LLM 在检索与编排中的误差、缺乏自动收集与评估新技能的机制、以及在极大规模生态中仍受检索与编排成本影响。

---

## 423. FAST-DIPS: Adjoint-Free Analytic Steps and Hard-Constrained Likelihood Correction for Diffusion-Prior Inverse Problems

**arXiv ID:** 2603.01591 | [PDF](https://arxiv.org/pdf/2603.01591v1)

**作者:** Minwoo Kim `[一作]` (Inha University), Hongki Lim `[通讯]` (Inha University)

**通讯引用:** 312 | [OpenAlex ID](https://openalex.org/A5019802439)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的训练‑无须重训练逆问题求解器fast‑dips，通过硬测量空间可行性投影、无伴随 ADMM 迭代和解析步长来实现数据一致性修正；

**💡 创新点**

创新点包括：1）在无手工伴随/伪逆的情况下使用闭式投影与解析步长的 ADMM‑风格纠正；2）通过模式替换重新热化实现高效降噪；3）构建潜在和像素‑潜在混合执行策略；4）提供局部模型最优步长与 KL 上界的理论保证；

**🔧 技术方法**

使用的技术有：扩散先验、逆向时间 SDE、变量拆分 ADMM、自动微分的 VJP/JVP（或有限差分估计）、解析步长初始化 + Armijo 回溯、重热化、潜在扩散模型、像素‑潜在混合调度；

**📊 数据集**

使用数据集：FFHQ‑256 与 ImageNet‑256（100 张图像），在八个线性与非线性逆问题（高斯模糊、运动模糊、随机修补、相位恢复、HDR 等）上进行实验；

**📈 对比分析**

与多种最先进基线（SITCOM、C‑ΠGDM、HRDIS、DAPS、PSLD、ReSample、Latent‑DAPS）在像素与潜在空间下进行对比，fast‑dips 在保持或提升 PSNR/SSIM/LPIPS 的同时实现最高 19.5× 的加速，尤其在运动模糊与相位恢复任务上表现突出；

**⚠️ 局限性**

局限性：对非线性算子缺乏全局收敛保证；性能受步长与 ε 预算的影响，需要在不同任务上做一定调参；目前仅针对加性高斯噪声；内循环迭代次数固定，缺乏自适应性；

---

## 424. MealRec: Multi-granularity Sequential Modeling via Hierarchical Diffusion Models for Micro-Video Recommendation

**arXiv ID:** 2603.01926 | [PDF](https://arxiv.org/pdf/2603.01926v1)

**作者:** Xinxin Dong `[一作]` (National University of Defense Technology), Xiaodong Wang `[通讯]` (National University of Defense Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种层次化扩散框架MealRec，用时间引导的内容扩散（TCD）提炼视频帧级表示，并在用户交互序列上做无时步盲扩散去噪（NPD）来改善微视频推荐；

**💡 创新点**

通过在视频帧级别使用时间引导的扩散模型实现细粒度内容去噪，并在用户历史上引入无时步盲扩散去除噪声行为，解决传统方法的模态冲突和非相关视频表示问题；

**🔧 技术方法**

扩散模型（U‑Net）、自注意力序列编码器（SASRec/CL4SRec/TedRec）、预训练VideoMAE帧特征、BERT/GloVe文本编码；

**📊 数据集**

四个真实微视频数据集：Microlens‑small、Microlens‑big、Shortvideo‑small、Shortvideo‑big；

**📈 对比分析**

与10个基线（顺序、跨模态、视频推荐）在Leave‑One‑Out评估下，以Hit@10/20和NDCG@10/20为指标，MealRec在所有数据集和指标上均优于或与最优基线相当，平均提升约4–18%；

**⚠️ 局限性**

仍依赖预训练视觉编码器，训练时需要两级扩散导致计算开销相对较大；对极端噪声或极短交互序列的鲁棒性待进一步验证。

---

## 425. How Small Can 6G Reason? Scaling Tiny Language Models for AI-Native Networks

**arXiv ID:** 2603.02156 | [PDF](https://arxiv.org/pdf/2603.02156v1)

**作者:** Mohamed Amine Ferrag `[一作]` (United Arab Emirates University), Merouane Debbah `[通讯]` (Khalifa University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对从135M到7B参数的语言模型在AI‑native 6G网络语义推理中的规模行为进行系统实验评估，使用标准化对齐的 6G‑Bench 基准。

**💡 创新点**

发现 1–1.5B 参数区间存在显著的稳定性转变，并提出了 Edge Score 评估模型在边缘资源下的可靠性/效率比；同时揭示了不同能力域的缩放弹性差异。

**🔧 技术方法**

利用 deterministic / stochastic 推理指标（pass@1、pass@k、Δ_k）、log‑linear 规模模型、分组灵敏度分析，并结合单查询延迟与内存剖面构建 Edge Score。

**📊 数据集**

使用 6G‑Bench 数据集：30 个决策任务、488 个 episode、3,722 个多选题，覆盖 3GPP、IETF、ETSI、ITU‑T、O‑RAN 相关能力。

**📈 对比分析**

通过模型参数规模对比准确率、稳定性及 Edge Score；结果表明 1.5–3B 参数模型在 deterministic accuracy 与资源成本之间达到最佳平衡，7B 模型仅带来微小增益。

**⚠️ 局限性**

局限性：评估仅限 7B 以下模型，未考虑更大规模模型；模型差异受训练数据、对齐策略和实现细节影响；Edge Score 未涵盖多模态推理与动态环境适应。

---

## 426. GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant

**arXiv ID:** 2603.01059 | [PDF](https://arxiv.org/pdf/2603.01059v1)

**作者:** Zhuokang Shen `[一作]` (East China Normal University), Shaohui Lin `[通讯]` (East China Normal University)

**通讯引用:** 3563 | [OpenAlex ID](https://openalex.org/A5043643513)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向多用户群聊的代理式框架 GroupGPT，能够在多模态环境下智能判断介入时机并生成回应。

**💡 创新点**

1) 小-大模型协同架构，将决策与生成分离，显著降低 token 消耗；2) 引入隐私转录器在云端前预处理 PII；3) 构建首个多用户群聊干预推理基准 MUIR，含 2500 条真实群聊段落与人类注释；4) 设计多种子系统（干预判定、隐私转录、多模态处理、响应生成）。

**🔧 技术方法**

使用轻量级 LLM（如 Qwen-3-4B、Llama-3.2-Instruct-3B）做干预判定与隐私转录，Qwen-2.5-32B 进行多模态内容的 caption，GPT‑4o 负责最终回复；同时利用 LoRA 微调、vLLM 推理框架、KNN+embedding、LLM‑as‑a‑judge 等技术。

**📊 数据集**

MUIR 数据集（2500 条群聊片段）以及公开的隐私标注数据集用于训练隐私转录器，视频/音频内容使用 Qwen‑2.5‑32B 与 Qwen3‑ASR‑Flash 进行转写。

**📈 对比分析**

在 MUIR 基准上对多种模型进行评测：随机、人工、LLM（GPT‑4o、Gemini‑2.5‑Pro 等）、embedding+KNN、轻量级 LLM 微调。GroupGPT 通过轻量化干预判定与生成实现了 4.72/5.0 的 LLM‑as‑a‑judge 分数，并在 token 消耗上较单一 LLM 方案减少约 3 倍，平均推理延迟约 4.3 秒。相比 LLM‑only，精确干预率与响应质量均明显提升。

**⚠️ 局限性**

1) 仍需对多模态推理与语音/视频处理进行更深层次优化；2) 当前的干预判定依赖人工标注的 MUIR，若数据偏差可能影响普适性；3) 在极高并发或大规模群聊中，GPU 内存仍较大；4) 由于使用公开 LLM，模型可能面临知识更新滞后与潜在的偏见。

---

## 427. Qayyem: A Real-time Platform for Scoring Proficiency of Arabic Essays

**arXiv ID:** 2603.01009 | [PDF](https://arxiv.org/pdf/2603.01009v1)

**作者:** Hoor Elbahnasawi `[一作]` (Qatar University), Tamer Elsayed `[通讯]` (Qatar University)

**通讯引用:** 2550 | [OpenAlex ID](https://openalex.org/A5032141720)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了首个支持跨提示、多属性评分的阿拉伯语自动论文评分平台，提供从作业创建、批量上传、评分配置到单属性评分及报告生成的完整工作流，并公开了对外API。

**💡 创新点**

创新点在于：①跨提示通用多属性评分，突破仅限单提示的传统系统；②整合前端后端完整端到端流程，教师可通过友好Web界面完成所有任务；③公开API供第三方调用；④基于LAILA大规模数据训练的TRATES和MOOSE两种SOTA模型，提升评分精度。

**🔧 技术方法**

技术方案包括：前端使用Next.js + Prisma+PostgreSQL管理作业与评分记录；后端采用FastAPI + SSE实现实时评分；GPU加速推理（NVIDIA A10）配合LLM（Fanar）Trait‑specific，辅以特征工程模型（NN、RF、XGB）和BERT‑based MOOSE；部署在Linux服务器，使用NGINX、PM2、Uvicorn等。

**📊 数据集**

使用LAILA数据集（7,859篇阿拉伯语作文，8个提示，7个属性+整体分），实现跨提示训练与评估。

**📈 对比分析**

评估方法采用Quadratic Weighted Kappa（QWK）。结果显示TRATES在所有属性上QWK最高，MOOSE次之；特征模型（NN/RF/XGB）QWK最低但推理时间最短（0.2‑0.3 s/篇）。TRATES推理耗时30 s/篇，MOOSE约1 s，特征模型约0.2 s。

**⚠️ 局限性**

限制包括：TRATES推理时间长（30 s/篇），对资源需求高；系统需手动配置新模型；跨提示评分虽然通用，但对极端新主题的解释性不足；当前仅支持阿拉伯语，需进一步扩展至多语言。

---

## 428. On the Exact Algorithmic Extraction of Finite Tesselations Through Prime Extraction of Minimal Representative Forms

**arXiv ID:** 2603.00911 | [PDF](https://arxiv.org/pdf/2603.00911v1)

**作者:** Sushish Baral `[一作]` (Chulalongkorn University), Warisa Sritriratanarak `[通讯]` (Chulalongkorn University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5018089434)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一个三阶段算法（合成发现、归一化、素块提取），实现二维平面网格上精确的周期性镶嵌模式提取，并提供层级过滤以显著提升效率。

**💡 创新点**

创新点包括：1）针对奇数维度的选择性复制策略；2）层级过滤机制避免冗余计算；3）双重解构策略（累计与单层）同时获取全局最优和粒度折衷；4）提供确定性、可复现的实现。

**🔧 技术方法**

使用了BFS剪枝搜索、矩形归一化、重复检测、图搜索求解（回溯+剪枝）、对称性约简以及开源实现。

**📊 数据集**

主要使用ARC‑AGI挑战的样例数据集（Band‑in‑blanks、Mixed‑noise、Simple‑pattern等）以及合成的不同尺寸网格进行可扩展性评估。

**📈 对比分析**

通过与传统统计提取方法对比实验，Band‑in‑blanks和Simple‑pattern在毫秒级完成；Mixed‑noise在层级过滤下从8.43 ms降至1.48 ms；总耗时受素块提取与求解主导，层级过滤可实现约5.3×的性能提升。

**⚠️ 局限性**

局限性：仅适用于轴对齐、矩形生成器的精确匹配；不支持旋转、非矩形或近似噪声模式；最坏情况下求解复杂度指数，对极大网格的可扩展性受限。

---

## 429. Constructing Synthetic Instruction Datasets for Improving Reasoning in Domain-Specific LLMs: A Case Study in the Japanese Financial Domain

**arXiv ID:** 2603.01353 | [PDF](https://arxiv.org/pdf/2603.01353v1)

**作者:** Yuma Okochi `[一作]` (Nomura Research Institute), Tomoyasu Okada `[通讯]` (Nomura Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个以主题词为驱动的合成指令数据集，规模约 9.5 亿 token，并在其中嵌入链式推理轨迹，针对日本金融领域。

**💡 创新点**

提出了可通用的全流程框架：从专业主题词出发、使用 LLM 生成指令与推理轨迹、通过 LLM‑as‑Judge 过滤、可调节推理长度，并可直接应用于任何需要高专业度的领域。

**🔧 技术方法**

技术包括：OpenAI gpt‑oss‑120b、Qwen3、gpt‑oss‑20b 进行生成与微调；MinHash/LSH 去重、MeCab 分词；LLM‑as‑Judge 质量过滤；持续预训练（CPT）+ 监督微调（SFT）；实验中对推理轨迹长度进行控制与分析。

**📊 数据集**

使用公开指令集 NuminaMath‑CoT、Nemotron‑Post‑Training‑Dataset‑v1、smol‑constraints 以及自建的日本金融语料库，合计约 9.5 亿 token。

**📈 对比分析**

在日本金融基准（japanese‑lm‑fin‑harness 与 pfmt‑bench‑fin‑ja）上进行评估，SFT+CPT+链式推理模型在所有子任务上均优于官方指令微调模型；在多轮对话任务中，推理轨迹存在时平均提升约 4.5–5.7 分。

**⚠️ 局限性**

局限性包括：推理轨迹长度超过约 1024 token 后无明显收益，过长时出现早期终止或输出循环；强行截断会导致性能下降；终止策略对性能敏感，需要进一步改进。

---

## 430. Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation

**arXiv ID:** 2603.01074 | [PDF](https://arxiv.org/pdf/2603.01074v1)

**作者:** Wangkai Li `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (National Key Laboratory of Deep Space Exploration)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对LiDAR点云语义分割在恶劣天气下性能下降的问题，提出了 A3Point 框架，能够自适应利用大范围数据增强并通过潜在空间学习和异常检测来缓解语义偏移。

**💡 创新点**

创新点包括：①将语义混淆先验建模为离散潜在表示，持续在线更新；②将语义偏移定位为异常检测问题，在增强点云中区分语义一致区和语义偏移区，从而对不同区域采用不同的优化策略。

**🔧 技术方法**

核心技术包括：VQ‑VAE 用于潜在表示学习；随机抖动（jitter）与点丢失（point drop）等广泛增强策略；基于潜在分布的 SSR 掩码生成；以及对 SSR 区域的潜在变量蒸馏损失。

**📊 数据集**

使用的公开数据集有：SemanticKITTI、SynLiDAR（源域），SemanticKITTI‑C、SemanticSTF（目标域）以及多种真实/合成恶劣天气数据，评估域泛化性能。

**📈 对比分析**

与现有域泛化与数据增强方法（如 PointDR、DGUIL、WADG、DGLSS、LiDARWeather、NTN）对比，A3Point 在 [A]→[C] 与 [B]→[C] 任务上分别提升了约 9.9% 与 11.7% 的 mIoU，并在 SPVCNN 与 Minkowski 这两种主干网络上均保持领先，证明其显著的鲁棒性。

**⚠️ 局限性**

局限性包括：①仍然依赖大量的增强样本和潜在编码器的训练，计算和存储开销较大；②对极端天气（如浓雾、暴雨）可能尚未完全覆盖；③模型在某些细粒度类别（如小型物体）上提升有限，需进一步改进语义一致性检测。

---

## 431. Multimodal Mixture-of-Experts with Retrieval Augmentation for Protein Active Site Identification

**arXiv ID:** 2603.01511 | [PDF](https://arxiv.org/pdf/2603.01511v1)

**作者:** Jiayang Wu `[一作]` (Westlake University), Yefeng Zheng `[通讯]` (Westlake University)

**通讯引用:** 17087 | [OpenAlex ID](https://openalex.org/A5051649145)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 MERA，一种基于检索增强的多模态混合专家框架，用于精确识别蛋白质活性位点。

**💡 创新点**

创新点在于引入多专家检索增强（MeRAG）和基于 Dempster–Shafer 证据理论的可靠性感知多模态融合（RMF），实现了残基级动态信息聚合和可信度驱动的融合决策。

**🔧 技术方法**

核心技术包括 ESM‑1b 序列编码、BioMedBERT 文本编码、检索增强生成、混合专家门控、可靠性估计和证据理论融合。

**📊 数据集**

使用 ProTAD‑Gen（自动生成文本的 ProTAD 扩展）和 TS125 两个公开数据集进行实验。

**📈 对比分析**

与多种序列、结构和多模态基线相比，MERA 在 ProTAD‑Gen 上 AUPRC 达到 0.90、Fmax 0.88，并在 TS125 上 AUROC 0.85，显著优于所有对照方法。

**⚠️ 局限性**

主要局限包括对结构模态的依赖仍未充分利用，检索过程对相似度阈值敏感，以及在边缘或不显著位点的预测准确性仍有提升空间。

---

## 432. SFCo-Nav: Efficient Zero-Shot Visual Language Navigation via Collaboration of Slow LLM and Fast Attributed Graph Alignment

**arXiv ID:** 2603.01477 | [PDF](https://arxiv.org/pdf/2603.01477v1)

**作者:** Chaoran Xiong `[一作]` (Shanghai Jiao Tong University), Ling Pei `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 4315 | [OpenAlex ID](https://openalex.org/A5021661339)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于慢–快协同的零样本视觉语言导航框架 SFCo‑Nav，在导航过程中动态决定是否调用大型语言模型。

**💡 创新点**

创新点在于：①将高成本 LLM 规划器与低成本实时导航器进行慢–快分离；②通过结构化属性图对齐置信度实现异步 LLM 触发；③显著降低视觉与语言令牌消耗与推理延迟。

**🔧 技术方法**

使用 GPT‑4o（LLM）、Grounding‑DINOv2（轻量级目标检测）、BLIP‑2（VLM 对比基线）、图匹配理论实现置信度评估；实现模块化的规划、导航和桥接。

**📊 数据集**

在公开 VLN 基准 R2R 与 REVERIE 上进行评测，并在实际酒店套房环境中实地验证。

**📈 对比分析**

与 NavGPT、MapGPT、NavCoT、SF‑Nav 等零样本基线相比，SFCo‑Nav 在 R2R/REVERIE 上保持或提升成功率，同时统一令牌使用量下降 50% 以上，运行时间提升 3–4 倍，且在真实机器人上实现实时导航。

**⚠️ 局限性**

局限性包括：①置信度阈值的选择需要经验权衡，阈值过高会导致过度触发 LLM，过低则可能失去规划精度；②对目标检测与图匹配的准确性高度依赖，误检或匹配错误可能导致规划失误；③在极其复杂或动态环境下，慢–快协同仍需频繁触发 LLM，效率提升有限。

---

## 433. SSKG Hub: An Expert-Guided Platform for LLM-Empowered Sustainability Standards Knowledge Graphs

**arXiv ID:** 2603.00669 | [PDF](https://arxiv.org/pdf/2603.00669v1)

**作者:** Chaoyue He `[一作]` (Alibaba-NTU Global e-Sustainability CorpLab), Chunyan Miao `[通讯]` (Alibaba-NTU Global e-Sustainability CorpLab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个端到端的可审计知识图谱平台SSKG Hub，能够将GRI、SASB、TCFD、IFRS S2等可持续发展标准的PDF文档自动提取三元组、存储至Neo4j并支持专家审核与认证，最终形成可查询、可融合的专业知识图谱；

**💡 创新点**

在标准文本提取与知识图谱治理上实现了多项创新：①基于标准的LLM提示与可配置分块提升抽取准确性；②首创的多角色治理流程（访客、专家、元专家、管理员）保障审计透明度与责任追踪；③完整的从草稿到认证的审计轨迹与可追溯性存储；④支持跨图谱融合与下游任务（KGQA、推理、分析）的统一平台；

**🔧 技术方法**

采用了PyMuPDF进行文本提取、Qwen‑Max LLM进行三元组抽取与验证、Neo4j数据库存储与查询、前端可视化（D3/force‑directed）、LLM验证器与专家界面、角色基于权限的访问控制；

**📊 数据集**

使用了行业标准文档集合（GRI、SASB、TCFD、IFRS S2）以及以IFRS S2 Industry‑based Guidance为案例的专家评审数据；

**📈 对比分析**

通过12名专家的手工评审案例对比实验验证：从73条草稿三元组中最终筛选出49条经过元专家认证的三元组（约67%保留），实验未给出传统方法的对比数值，仅展示专家评审与认证的比例提升；

**⚠️ 局限性**

局限性包括：依赖LLM抽取的质量与可控性，存在幻觉与误抽风险；需要专家持续参与，成本高；计算资源消耗大；目前仅覆盖有限标准，跨标准兼容性和自动化验证尚未充分测试；

---

## 434. Retrodictive Forecasting: A Proof-of-Concept for Exploiting Temporal Asymmetry in Time Series Prediction

**arXiv ID:** 2603.00636 | [PDF](https://arxiv.org/pdf/2603.00636v1)

**作者:** Cedric Damour `[一作]` `[通讯]`, Cedric Damour

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

该论文提出了一种逆向预测范式，通过在条件变分自编码器中进行逆向MAP优化来实现时间序列的逆向预测。

**💡 创新点**

创新点在于将预测视为逆向推断，利用统计时间不可逆性量化，并结合流型先验实现高效MAP推断，同时提供可操作的不可逆性诊断门控。

**🔧 技术方法**

核心技术包括逆向条件变分自编码器、RealNVP 正则化流先验、Adam 优化的多起点 MAP 推断，以及基于 J‑divergence 的时间箭头诊断。

**📊 数据集**

实验数据来自四个设计好的合成过程和两个 ERA5 气候再分析数据集（北海10 m 风速和表面太阳辐射）。

**📈 对比分析**

与前向 MLP、普通 CVAE、无流先验等基线比较，逆向 MAP 在不可逆案例中实现了10–18 % 的 RMSE 降低，且在可逆案例无明显优势，验证了预设四条预测。

**⚠️ 局限性**

局限包括仅使用 MAP 估计缺乏完整后验、多元高斯解码假设导致误差相关、单变量短窗口设置、样本量有限以及对流形优化的依赖。

---

## 435. Decoupling Motion and Geometry in 4D Gaussian Splatting

**arXiv ID:** 2603.00952 | [PDF](https://arxiv.org/pdf/2603.00952v1)

**作者:** Yi Zhang `[一作]` (Sun Yat-sen University), Jian-Fang Hu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1902 | [OpenAlex ID](https://openalex.org/A5102336058)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 VeGaS，一种基于速度的 4D 高斯 splatting 框架，实现了运动与几何的解耦；

**💡 创新点**

创新点在于引入了 Galilean shear 矩阵实现时间可变速度建模，并设计了轻量级几何变形网络以独立优化几何形变；

**🔧 技术方法**

技术主要包括时间可变速度的 shear 变换、四维高斯协方差的共轭变换、线性插值与分段数值积分、四元数与双四元数旋转、以及多尺度位置编码的 MLP；

**📊 数据集**

使用了两个公开数据集：多视角真实场景的 Neural 3D Video（Neu3DV）和单视角合成场景的 D-NeRF；

**📈 对比分析**

通过与 NeRFPlayer、HyperReel、4DGS、4DGaussians 等先进方法对比，VeGaS 在 PSNR、SSIM、LPIPS 等指标上均实现了领先（Neu3DV 上 PSNR +0.67dB，LPIPS -0.01，D-NeRF 上 PSNR +0.58dB，LPIPS -0.01），并在视觉上减少了伪影和细节丢失；

**⚠️ 局限性**

局限性主要是仍依赖多视角或高质量的训练数据，对极端非线性动态或极低光照条件的鲁棒性尚未充分验证。

---

## 436. More Data, Fewer Diacritics: Scaling Arabic TTS

**arXiv ID:** 2603.01622 | [PDF](https://arxiv.org/pdf/2603.01622v1)

**作者:** Ahmed Musleh `[一作]` (Qatar Computer Research Institute), Kareem Darwish `[通讯]` (Qatar Computer Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建了一个自动化管道，利用VAD、ASR、自动脱格化与降噪技术，生成了约4000小时的高质量阿拉伯语TTS语料库并训练了多种模型。

**💡 创新点**

创新点在于通过大规模非脱格化数据与逐步增量训练验证，证明当数据量足够大时，模型可隐式学习发音，从而降低了对脱格化文本的依赖。

**🔧 技术方法**

技术包括Silero VAD、Fanar ASR、基于RNN的自动脱格化、Pyannote声纹分离、F5‑TTS扩散变换器与ConvNeXt V2、Vocos Mel声码器以及WER与SpeechBERTScore评估。

**📊 数据集**

使用的数据集为约4000小时的阿拉伯语音频，经过自动处理后得到带/不带脱格化的文本；此外还从公开来源构造了100小时、1000小时的子集。

**📈 对比分析**

通过WER和SpeechBERTScore对比，发现带脱格化模型表现最优；但4000小时的非脱格化模型在SpeechBERTScore上与1K小时模型相当甚至更好，说明大数据量能弥补脱格化缺失。

**⚠️ 局限性**

局限性包括自动化管道中的ASR与脱格化误差会影响训练质量、评估仅基于客观指标缺少人工听觉验证、以及对方言与领域的覆盖不足。

---

## 437. Fungi as functors: A category-theoretic approach to mycelial organisation

**arXiv ID:** 2603.01320 | [PDF](https://arxiv.org/pdf/2603.01320v1)

**作者:** Andrew Adamatzky `[一作]` `[通讯]` (Unconventional Computing Lab, UWE Bristol), Andrew Adamatzky (Unconventional Computing Lab, UWE Bristol)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

提出了一个无方程、基于范畴理论的真菌网络组织框架，将真菌视为环境到网络的函子，统一描述环境变换、程序语义、自然变换、伴随关系等；

**💡 创新点**

创新点在于将环境→网络的函子化、自然变换用于种属/菌株差异、伴随对生态反馈的刻画，以及利用局部李代数与Baker‑Campbell‑Hausdorff展开给出顺序效应的可检验预测；

**🔧 技术方法**

使用了结构化范畴理论（函子、自然变换、伴随、极限/余极限）、局部李群与李代数、Baker‑Campbell‑Hausdorff公式等数学技术；

**📊 数据集**

未使用具体实验数据集，文中仅给出最小化暴露示例作为理论演示；

**📈 对比分析**

由于是理论框架，未给出实验比较或性能指标；框架提出可实验检验的定量预测（如顺序非交换二次缩放）以验证其可行性；

**⚠️ 局限性**

局限在于缺乏具体生物机制的验证、局部李结构的适用性需实验证明、推导仅在小扰动范围内成立，未提供数值实现或大尺度实验验证。

---

## 438. MedCollab: Causal-Driven Multi-Agent Collaboration for Full-Cycle Clinical Diagnosis via IBIS-Structured Argumentation

**arXiv ID:** 2603.01131 | [PDF](https://arxiv.org/pdf/2603.01131v1)

**作者:** Yuqi Zhan `[一作]` (Hangzhou Dianzi University), Zhu Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 192941 | [OpenAlex ID](https://openalex.org/A5100748869)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了MedCollab——一种基于多智能体的全周期临床诊断框架，能够动态招募专家和检查智能体，并通过IBIS结构化论证与层级病因链（HDCC）实现可追溯、因果驱动的诊断推理；

**💡 创新点**

创新点包括：1）将临床诊断映射为因果驱动的层级病因链，区别关联与因果；2）使用IBIS论证协议将诊断结果结构化为可验证的证据图；3）引入GP引导的多轮共识与逻辑审计机制，动态调整智能体权重以过滤逻辑错误；

**🔧 技术方法**

核心技术包括：多智能体动态招募、IBIS（Issue-Based Information System）论证协议、层级病因链构建、GP主导的逻辑审计与加权投票、基于LLM的专业诊断与检查报告生成；

**📊 数据集**

实验数据集为ClinicalBench（1500例真实临床病例）和MIMIC-IV（595例筛选病例），两者统一为相同的咨询模式；

**📈 对比分析**

与领先LLM（如GPT‑4o、Gemini‑3‑Flash、Baichuan4‑Turbo等）以及现有医学多智能体系统（ClinicalAgent、MedLA、MEDDxAgent）在ACC、CDR、Entity‑F1、DCA、RaTEScore等指标上对比，MedCollab在ACC上达到76.9%（ClinicalBench）/57.7%（MIMIC‑IV），明显优于基线；在RaTEScore上在四个临床维度均居前，显示更强的逻辑一致性和医学真实性；

**⚠️ 局限性**

局限性主要在于：①对高质量、结构化临床记录的依赖，数据稀缺时性能可能下降；②框架复杂度高，部署和运维成本相对较大；③虽然引入因果链和逻辑审计，但仍依赖底层LLM，潜在的误判或信息缺失仍需进一步验证。

---

## 439. Incremental, inconsistency-resilient reasoning over Description Logic Abox streams

**arXiv ID:** 2603.01799 | [PDF](https://arxiv.org/pdf/2603.01799v1)

**作者:** Cas Proost `[一作]` (KU Leuven), Pieter Bonte `[通讯]` (KU Leuven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了面向描述逻辑（DL）ABox 流的增量推理语义，并在此基础上设计了基于时间偏好修复的非递归不一致修复机制，同时给出了针对 OWL2 RL 的半朴素增量维护算法。

**💡 创新点**

创新点在于：① 将窗口内各时间戳的 ABox 视为优先级层，使用“新优先旧后”偏好修复实现不一致自恢复；② 通过构造“最小窗口模型”把推理结果归属到产生推理的最老 ABox，从而实现高效的增量窗口维护；③ 对不递归负向包含的冲突集提供专门的增量修复策略。

**🔧 技术方法**

采用的技术包括：描述逻辑（OWL2 RL）理论、窗口（sliding/tumbling）流模型、直接求和（direct sum）解释子、最小/规范模型构造、半朴素增量推理、基于优先级的修复语义。

**📊 数据集**

论文未使用任何公开数据集，也未进行实验评估；所有讨论均基于理论分析和伪代码。

**📈 对比分析**

由于缺乏实验实现，文中没有给出性能对比或基准；作者只说明该增量算法通过减少重推理量和及时删除旧事实，可在理论上实现更低延迟和更高吞吐量，但未给出具体数值。

**⚠️ 局限性**

局限性包括：① 只处理非递归不一致，递归负向包含需要预先展开或手工限制深度；② 仅适用于 OWL2 RL，其他 OWL2 规范缺乏可直接迁移的最小模型；③ 未考虑背景知识或静态知识库；④ 论文未提供实验验证，实际性能及可扩展性仍待评估。

---

## 440. Align and Filter: Improving Performance in Asynchronous On-Policy RL

**arXiv ID:** 2603.01365 | [PDF](https://arxiv.org/pdf/2603.01365v1)

**作者:** Homayoun Honari `[一作]` (Mila), Glen Berseth `[通讯]` (Université de Montréal)

**通讯引用:** 2131 | [OpenAlex ID](https://openalex.org/A5045351810)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种新算法 VACO（Total Variation-based Advantage-aligned Constrained policy Optimization），用于解决异步分布式强化学习中的政策滞后问题。

**💡 创新点**

创新点在于：①通过优势对齐（Advantage Realignment）将离线数据的优势估计对齐到当前学习策略，从而消除后向政策滞后；②利用总变差（Total Variation）度量对每个小批次进行过滤，只保留能降低政策差异的数据点，抑制前向政策滞后。

**🔧 技术方法**

核心技术包括：优势对齐（利用 V-trace 近似）、基于 TV 的数据过滤、受限策略优化框架以及对 PPO、TRPO 等常用算法的改进。

**📊 数据集**

主要数据集为：MuJoCo 机器人任务（多种运动控制环境）以及大型语言模型的数学推理任务（GSM8k 数据集）。

**📈 对比分析**

与 PPO‑Clip、PPO‑KL、SPO、IMPALA 等基线相比，VACO 在 MuJoCo 任务中在不同异步程度下均表现出更高的 IQM、归一化累计回报和更低的最优性间隙；在 LLM 推理任务中，VACO 对前向政策滞后具有更强的鲁棒性，保持了更高的评估准确率，同时训练效率更好。

**⚠️ 局限性**

限制：算法对 TV 阈值 δ 的设置仍需经验调优；在极高异步程度或极大数据量下，优势对齐的估计误差可能累积；以及在不同任务域（非机器人、非 LLM）中的泛化性尚未充分验证。

---

## 441. Hybrid TD3: Overestimation Bias Analysis and Stable Policy Optimization for Hybrid Action Space

**arXiv ID:** 2603.01302 | [PDF](https://arxiv.org/pdf/2603.01302v1)

**作者:** Thanh-Tuan Tran `[一作]` (University of Engineering and Technology, Vietnam National University), Xiem HoangVan `[通讯]` (University of Engineering and Technology, Vietnam National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了 Hybrid TD3，专门用于离散-连续混合动作空间的强化学习，解决机器人操作中的高层决策与低层执行的联合优化问题。

**💡 创新点**

创新点包括：①在 TD3 基础上引入加权裁剪 Q 学习目标，对离散动作分布进行边际化，提供更平滑的梯度；②对混合动作空间的过估计偏差进行严格理论分析，给出完整的偏差排序；③在全域随机化环境下验证该方法对零样本泛化的鲁棒性。

**🔧 技术方法**

使用的技术包括 Twin Delayed Deep Deterministic Policy Gradient（TD3）框架、双 critic、延迟策略更新、目标策略平滑、加权裁剪 Q 目标、离散-连续动作分解策略，以及 PyBullet 模拟器中的 UF850 机器人臂。

**📊 数据集**

数据集为在 PyBullet 中随机生成的多种对象（超过21种类别，随机姿态、质量、摩擦等），用于训练的四个操作任务（Reach、Pick、Move、Put），并在未见过的对象上进行零样本测试。

**📈 对比分析**

与多种基线（SAC、DDPG、PPO、HySAC、TD3AQ、TSMPDQN、PASAC、ADSAC、HyDARC、HyDATD3、HyACC、HyTQC）进行对比。Hybrid TD3 在训练稳定性、平均回报、最终奖励分布以及成功率上均优于所有基线，且在未见对象上表现与训练集相当，显示出强的泛化能力。

**⚠️ 局限性**

局限性包括：理论分析假设 critic 误差独立且同步，实际可能受相关性影响；仅在仿真环境中验证，真实机器人实验仍需进一步探索；对高维离散动作空间的扩展尚未深入，可能仍存在指数级爆炸问题。

---

## 442. Selection as Power: Constrained Reinforcement for Bounded Decision Authority

**arXiv ID:** 2603.02019 | [PDF](https://arxiv.org/pdf/2603.02019v1)

**作者:** Jose Manuel de la Chica Rodriguez `[一作]` (Grupo Santander), Juan Manuel Vera Díaz `[通讯]` (Grupo Santander)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了“Selection as Power 2.0”，将先前的静态选择治理框架扩展为动态的激励式治理体系，使用强化学习对选择策略和缩减器参数进行更新，但所有更新都被外部主权约束投影所限制，以防止选择集中化。

**💡 创新点**

创新点在于：①将投影式约束嵌入强化学习更新中，形成可控制的选择动力学；②采用双重更新机制同时调节评分和缩减器，保持结构多样性；③通过“治理债务”量化投影强度，提供可视化的约束执行反馈。

**🔧 技术方法**

技术主要包括：投影梯度上升（Projected Gradient Ascent）用于受限参数更新；多智能体强化学习框架；离散时间的可扩展选择策略；结构化约束集（如最小探索概率、最大集中度、最小多样性桶等）以及与评估器的即时反馈循环。

**📊 数据集**

实验数据集为三个受监管金融场景：欺诈检测、支付基础设施监测、季度业务回顾分析，每个场景包含5种任务变体；候选代理数为7个，特征涵盖风险、稳定性、延迟、可审计性和合规标签。

**📈 对比分析**

与三种基线对比：静态治理（无学习）、无约束强化学习、以及无多样性的确定性Top‑K聚合。实验结果显示：无约束RL快速收敛到单一代理的确定性占优；Scalar‑TopK在无学习情况下也产生结构性单一；而激励治理在保持收益提升的同时，选择集中度始终低于1，且治理债务在高学习率下可控，证明投影约束能在不牺牲性能的前提下限制权力集中。

**⚠️ 局限性**

局限性包括：评估器被假设为完美且即时，未考虑噪声、延迟或欺骗性反馈；奖励函数为合成特征级别，未覆盖真实复杂的长期目标；投影运算在大规模代理或高维约束下可能成本上升；实验仅覆盖小规模金融场景，缺乏对长期公平性和收敛性理论保证的深入分析。

---

## 443. Conformal Prediction for Risk-Controlled Medical Entity Extraction Across Clinical Domains

**arXiv ID:** 2603.00924 | [PDF](https://arxiv.org/pdf/2603.00924v1)

**作者:** Manil Shrestha `[一作]` (Drexel University), Edward Kim `[通讯]` (Drexel University)

**通讯引用:** 3759 | [OpenAlex ID](https://openalex.org/A5100728165)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并验证了一个基于对合预测的LLM实体抽取框架，应用于FDA药品标签和MIMIC‑CXR放射报告。

**💡 创新点**

发现不同临床文本结构导致LLM校准方向相反，并提出了域特异性对合阈值自动适配以保证覆盖率。

**🔧 技术方法**

使用GPT‑4.1和Llama‑4‑Maverick生成token级log‑probability，计算跨度置信度，采用split conformal校准。

**📊 数据集**

1,000份FDA药品标签（共110k实体）与100份MIMIC‑CXR报告的RadGraph注释。

**📈 对比分析**

通过事实分数和实体F1评估，FDA标签97.7%准确率，RadGraph F1 0.81‑0.84；对合预测在两域均达≥90%覆盖率，拒绝率9‑13%。

**⚠️ 局限性**

仅适用于公开log‑probability的模型，缺少对黑盒模型的支持，且在含稀疏或模糊语言时仍有较高ECE。

---

## 444. Security Is Not Enough: Privacy in Encryption Regulation and Lawful-Surveillance Protocols

**arXiv ID:** 2603.00841 | [PDF](https://arxiv.org/pdf/2603.00841v1)

**作者:** Artur Pericles L. Monteiro `[一作]` `[通讯]`, Artur Pericles L. Monteiro

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文论证了将隐私仅视为安全的框架不足以全面评估政府对加密数据的异常访问，探讨了隐私-安全二分的局限性，并对现有的合法监控协议（lawful‑surveillance protocols）与Apple的客户端扫描方案进行评述。

**💡 创新点**

创新点在于提出隐私的多维度概念（不只安全），指出当前安全为核心的隐私框架在面对异常访问、政府黑客以及内容监管等场景时无法充分解释和解决伦理与法律争议，从而呼吁对隐私权的更丰富理论构建。

**🔧 技术方法**

本文主要使用了密码学原理（如可验证加密、前向保密、密钥托管等）和法律/政策分析方法，讨论了多种协议设计（例如自托管、法官-陪审员机制、加密“屈服区”等）以及相关技术细节。

**📊 数据集**

由于是理论与综述性质，本文未使用具体数据集，而是引用了现有研究报告、案例法、法案文本以及技术白皮书中的数据和示例。

**📈 对比分析**

比较方法以概念与案例对比为主，未进行量化性能评估；文中主要评估各方案的安全风险、合法性、可操作性和对隐私的影响，并通过对比说明安全中心框架下的不足。

**⚠️ 局限性**

局限性包括：缺乏实证实验和量化结果，难以精确评估各协议的实际安全成本；对法律环境的假设可能与不同司法管辖区的实际法规不完全吻合；以及对隐私多维度概念的具体阐释仍不完善。

---

## 445. Symbol-Equivariant Recurrent Reasoning Models

**arXiv ID:** 2603.02193 | [PDF](https://arxiv.org/pdf/2603.02193v1)

**作者:** Richard Freinschlag `[一作]`, Günter Klambauer `[通讯]` (Medical AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种符号等变递归推理模型（SE‑RRM），在结构化推理任务中显式编码符号置换等变性；

**💡 创新点**

通过在模型架构中加入符号维度的自注意力实现符号等变性，显著降低数据增强需求并提升对未知符号和规模的泛化；

**🔧 技术方法**

基于Transformer的自注意力、RMSNorm、RoPE2d、深度监督以及符号嵌入的三维张量处理；

**📊 数据集**

使用Sudoku（9×9、4×4、16×16、25×25）、ARC‑AGI‑1/2（最多10色、30×30格）和Maze‑hard（30×30格）等公开基准数据集；

**📈 对比分析**

与HRM、TRM以及LLM（GPT‑OSS‑20B）对比，SE‑RRM在Sudoku上FSR提高超过11%、GPA提高超过7%；在ARC‑AGI上pass@2与FSR与TRM相当并优于HRM；在Maze上取得最优或接近最优性能，且参数仅200万；

**⚠️ 局限性**

计算与内存复杂度相较于传统RRM线性增大至K倍；对非符号等变任务需要破除符号等变性；对更大符号集（K≫I）时可能不切实际。

---

## 446. Accelerating PDE Surrogates via RL-Guided Mesh Optimization

**arXiv ID:** 2603.02066 | [PDF](https://arxiv.org/pdf/2603.02066v1)

**作者:** Yang Meng `[一作]` (University of Chicago), Yuxin Chen `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于强化学习的网格点自适应选择框架（RLMesh），用于在有限的模拟预算下高效训练 PDE 近似器；

**💡 创新点**

创新点包括将网格点选择视为有限时限 MDP，利用轻量级代理模型给 RL 策略提供终端奖励，并实现每实例的空间自适应采样；

**🔧 技术方法**

使用的技术包括深度 Q‑网络（Deep Q‑Learning）进行策略优化、核岭回归代理模型评估代理收益、傅里叶神经算子（FNO）作为下游近似器、以及支持非均匀采样的自定义有限体积求解器；

**📊 数据集**

实验数据集包括 1D Burgers 方程、2D Darcy 流动以及 3D Lorenz‑96 动力系统，分别用于评估时间预测、稳态映射和混沌格点系统；

**📈 对比分析**

与均匀、随机、梯度、方差、强度等启发式基线以及实例级主动学习方法（MRA‑FNO、AL4PDE）对比，RLMesh 在相同查询预算下实现更低的 RMSE，节约 30%–50% 的模拟成本，且时间‑误差曲线更快逼近全信息下界；

**⚠️ 局限性**

局限性包括：需先在完整网格上预训练 FNO；对低维、规则网格表现良好，尚未验证高维或不规则几何；当单实例网格点预算极低时，代理模型与真实误差的相关性下降，影响采样质量。

---

## 447. A note on Jerabek's paper "A simplified lower bound for implicational logic"

**arXiv ID:** 2603.01929 | [PDF](https://arxiv.org/pdf/2603.01929v1)

**作者:** Lev Gordeev `[一作]`, Edward Hermann Haeusler `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

讨论并否定Jeřábek提出的最小蕴含逻辑指数下界，证明使用多前提自然演绎（ND）中的有向无环图（DAG）形式能够给出多项式上界；

**💡 创新点**

引入了重复规则$R_n$与正则证明路径闭合的新概念，构造了证明系统T，使得所有最小命题可在多项式大小证明并且可在多项式时间验证；

**🔧 技术方法**

采用了证明理论中的树到DAG压缩技术，结合Prawitz自然演绎与Hudelmaier无剪切序列演算，并对可验证性进行了重新定义；

**📊 数据集**

该工作为纯理论研究，没有使用任何实验数据集；

**📈 对比分析**

与Jeřábek的Frege系统进行比较，指出其指数下界与本文多项式上界不兼容；证明系统T的证明证书可在多项式时间内验证，体现了良好的算法性能；

**⚠️ 局限性**

局限性包括：仅针对纯蕴含最小逻辑，未证明对其他逻辑的适用性；证明依赖于正则路径闭合的特殊定义，可能在其他证明系统中不可推广；缺乏实验验证与实际实现细节。

---

## 448. Achievability of Heterogeneous Hypergraph Recovery from its Graph Projection

**arXiv ID:** 2603.01268 | [PDF](https://arxiv.org/pdf/2603.01268v1)

**作者:** Alexander Morgan `[一作]` (Massachusetts Institute of Technology), Chenghao Guo `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出异质随机超图模型，并通过在投影图中寻找最大团来恢复给定度数的超边；给出了可实现的阈值条件。

**💡 创新点**

将可实现阈值推广到异质情况，证明最大团估计在噪声与多度数混合的投影下仍能实现；阐明了噪声“全边”配置的主导效应。

**🔧 技术方法**

理论分析、组合优化（g函数）、概率论（二项分布、Bonferroni 近似）、最大团枚举算法。

**📊 数据集**

无具体实验数据，研究基于理论随机模型。

**📈 对比分析**

与已知的 d‑uniform 超图阈值对比，证明在相同或更一般的条件下可实现；主要以渐进成功率为依据，未给出实验性能曲线。

**⚠️ 局限性**

仅给出可实现性而非最优性；只考虑 d≥3 的度数；对所有度数的最优阈值与噪声模型下的 exact recovery 尚未解决。

---

## 449. ASTRA-bench: Evaluating Tool-Use Agent Reasoning and Action Planning with Personal User Context

**arXiv ID:** 2603.01357 | [PDF](https://arxiv.org/pdf/2603.01357v1)

**作者:** Zidi Xiu `[一作]` (Apple), Samy Bengio `[通讯]` (Apple)

**通讯引用:** 41026 | [OpenAlex ID](https://openalex.org/A5017529415)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ASTRA-bench，评估 AI 个人助手在时序个人上下文、工具使用与多步规划中的表现。

**💡 创新点**

将时间演进的个人上下文、交互工具箱和多维复杂度标注结合成统一的状态化评测框架。

**🔧 技术方法**

采用事件驱动合成方案、LLM 生成的多应用实体、Milestones&Minefields 规则评估与 LLM 判定器。

**📊 数据集**

使用 5 个主角的 2,413 场景数据，包含 600+ 邮件/日历/消息等个人数据，并手工标注复杂度层级。

**📈 对比分析**

通过基准模型（Claude‑4.5‑Opus、DeepSeek‑V3.2 等）与人类标注的 Milestone/LLM 评测对比，发现高复杂度下性能显著下降，Payload 生成是主要瓶颈。

**⚠️ 局限性**

人工标注成本高、评估器可能误判、合成数据与真实日志差距、工具覆盖不足以及缺乏安全/授权机制。

---

## 450. Teen Vigilance: Navigating Risky Social Interactions on Discord

**arXiv ID:** 2603.02052 | [PDF](https://arxiv.org/pdf/2603.02052v1)

**作者:** Elena Koung `[一作]` (Pennsylvania State University), Yubo Kou `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国13-17岁青少年在Discord平台上面对风险社交互动时的应对策略进行了深入访谈研究。

**💡 创新点**

首次从“警觉性（vigilance）”角度系统阐述青少年如何在个人层面和社区层面主动识别、评估并管理风险，为青少年安全设计提供了新的视角。

**🔧 技术方法**

采用半结构化访谈与反思性主题分析（Reflexive Thematic Analysis）对16名受访者的文本进行编码和主题归纳。

**📊 数据集**

数据来源为16名美国青少年的访谈记录，无公开数据集；研究依赖参与者自述的风险经历与对策。

**📈 对比分析**

文章未进行定量对比或性能评估，仅通过质性分析呈现主题与案例；对照其它平台（如Instagram、TikTok）安全策略时做了概念性对比，但未给出可度量的性能指标。

**⚠️ 局限性**

局限性包括样本规模小、仅限美国青少年、IRB限制导致难以直接接触Discord社群、对不同族裔和性别的代表性不足，且研究仅关注青少年自述，缺乏客观验证。

---

## 451. Surgical Post-Training: Cutting Errors, Keeping Knowledge

**arXiv ID:** 2603.01683 | [PDF](https://arxiv.org/pdf/2603.01683v1)

**作者:** Wenye Lin `[一作]` (University of Hong Kong), Kai Han `[通讯]` (University of Hong Kong)

**通讯引用:** 9807 | [OpenAlex ID](https://openalex.org/A5101784732)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Surgical Post-Training（SPoT）框架，通过 Oracle 对模型错误推理进行最小化编辑，生成与原始分布相近的对比数据，并采用二分类交叉熵目标高效提升推理准确率，同时保持先前知识。

**💡 创新点**

创新点在于：①利用“弹性系绳”奖励机制（Reward‑SFT）实现隐式正则化，阻止灾难性遗忘；②将正负样本拆分为独立二分类任务，消除正样本拉升（pull‑up）效应；③结合数据修正管线与二分类目标，构成一体化的后训练流程，显著降低采样成本。

**🔧 技术方法**

技术手段包括：Direct Preference Optimization (DPO)、Reward‑SFT、SPoT‑BCE/BCO（二分类交叉熵）、Oracle‑guided surgical rectification、Longest Common Subsequence (LCS) 过滤、以及基于KL约束的奖励设计。

**📊 数据集**

使用数据集：DAPO‑Math‑17k（英文子集）用于训练；AIME24/25、AMC23、Math500、Minerva、Olympia、GPQA‑D、Connect4、IFEval 等用于评估，确保在分布内、分布外推理及指令遵循上的全面测试。

**📈 对比分析**

对比 SFT、RFT、SFT+、DPO、Reward‑SFT、DFT 等基线，SPoT‑BCO 在 Qwen3‑8B 与 Llama‑3.1‑8B‑Instruct 上平均提升约 6–7% 的推理准确率，同时在 OOD 推理和 IFEval 指令跟随任务中保持或提升性能，验证了方法的高效性与泛化能力。

**⚠️ 局限性**

局限性：① 需要 Oracle（人类或更强模型）进行修正，增加人工成本；② 目前仅在数学推理任务上验证，未充分展示在代码生成、规划等其他领域的适用性；③ 对更大规模模型的实验有限，需进一步验证可扩展性。

---

## 452. Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark

**arXiv ID:** 2603.00611 | [PDF](https://arxiv.org/pdf/2603.00611v1)

**作者:** Lijing Cai `[一作]`, Xun Cao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了动态高光谱图像数据集DynaSpec，提出了基于Transformer的PG‑SVRT网络，实现了视频级压缩光谱重建，并搭建了DD‑CASSI原型进行实测；

**💡 创新点**

创新点在于首次将空间先后时间注意力机制与桥接令牌结合，以低复杂度高效捕捉跨帧补偿信息，并通过Mask‑Guided Degradation Perception提升解码器对编码失真特征的感知；

**🔧 技术方法**

采用了Transformer框架中的MGDP、CDPA（空间→时间线性注意力+桥接令牌）与MDFFN，并结合PyTorch实现训练；

**📊 数据集**

使用了新收集的DynaSpec数据集（30序列300帧），以及CAVE、KAIST等公开图像数据集进行训练验证，并在实际DD‑CASSI采集的实测数据上进行测试；

**📈 对比分析**

与多种图像级SOTA方法（如MST‑L、CST‑L等）以及RGB视频恢复方法对比，PG‑SVRT在PSNR、SAM、ST‑RRED等指标上均取得最高分（PSNR>41 dB、SAM最低、ST‑RRED最佳），且FLOPs低于多数图像级模型；

**⚠️ 局限性**

局限性在于DynaSpec的采集环境相对理想（室内光照、预设运动），可能对未受控自然场景的泛化能力有限，需要进一步在更复杂多变环境中验证。

---

## 453. Certifiable Estimation with Factor Graphs

**arXiv ID:** 2603.01267 | [PDF](https://arxiv.org/pdf/2603.01267v1)

**作者:** Zhexin Xu `[一作]` (Northeastern University), David M. Rosen `[通讯]` (Northeastern University)

**通讯引用:** 1392 | [OpenAlex ID](https://openalex.org/A5082166133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可证实最优的因子图优化框架，将Shor松弛与Burer-Monteiro因式分解与因子图模型融合，使得现有因子图库可直接实现可证实估计。

**💡 创新点**

证明因子图结构在Shor松弛与BM分解下保持不变，允许通过简单的代数提升把原始变量和因子映射到更高维空间，从而用现有局部优化库实现Riemannian Staircase并获得全局最优性证明。

**🔧 技术方法**

使用QCQP建模、Shor松弛、Burer-Monteiro因式分解、Riemannian Staircase、Lifted变量与因子，以及GTSAM等因子图库实现。

**📊 数据集**

采用多种SLAM基准数据集，包括pose graph（MIT, CSAIL, Intel Kitti, Manhattan等）、landmark SLAM（Victoria, Trees, Goats等）和range-aided SLAM（Goats, Plaza1, Plaza2, Single Drone等）。

**📈 对比分析**

与手工设计的可证实求解器（SE-Sync, CPL-SLAM, CORA）以及传统局部因子图优化（GTSAM）对比，结果在大多数数据集上得到相同的全局最优目标值，性能接近但在某些场景下略慢，尤其是range‑aided SLAM因LM迭代和不良条件导致收敛较慢。

**⚠️ 局限性**

限制在于使用通用的LM优化器导致收敛速度不如专门化的Riemannian求解器；在高度不良条件下可能需要更高阶Rank；实现虽然比传统手工pipeline快，但仍需一定专业知识，整体性能仍略慢于手工特定方法。

---

## 454. CARD: Towards Conditional Design of Multi-agent Topological Structures

**arXiv ID:** 2603.01089 | [PDF](https://arxiv.org/pdf/2603.01089v1)

**作者:** Tongtong Wu `[一作]` (Monash University), Gholamreza Haffari `[通讯]` (Monash University)

**通讯引用:** 13061 | [OpenAlex ID](https://openalex.org/A5081525024)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CARD（Conditional Agentic Graph Designer）机制，利用条件图生成框架在训练和运行时根据环境动态调整 LLM‑基多智能体系统的通信拓扑，以提升效果与鲁棒性。

**💡 创新点**

创新点：①正式化 AMACP 适应性通信协议；②使用条件变分图编码器与解码器学习环境感知拓扑；③在训练与推理阶段均可自动响应模型升级、工具可用性及数据源变化。

**🔧 技术方法**

技术方法：图神经网络编码器/解码器、条件文本嵌入、环境感知优化、基于 AMACP 损失的梯度下降训练、条件化变分图生成。

**📊 数据集**

使用数据集：HumanEval、MATH、MMLU 以及多种公开 LLM 基础模型与工具组合。

**📈 对比分析**

对比方法：与手工链式思考、LLM‑Debate、随机图、GPT‑Swarm、G‑Designer、Aflow 等基线比较；在 HumanEval、MATH、MMLU 上 CARD 取得 90.5%/74.5%/86.67% 的最高准确率，平均提升 0.5–3.0pp，尤其在模型或工具变化下保持更高鲁棒性。

**⚠️ 局限性**

局限性：依赖预设 anchor 拓扑；未验证极大规模代理集的可扩展性；需准确的环境信息；未评估在线强化学习或长期持续适应的性能。

---

## 455. The Texture-Shape Dilemma: Boundary-Safe Synthetic Generation for 3D Medical Transformers

**arXiv ID:** 2603.00985 | [PDF](https://arxiv.org/pdf/2603.00985v1)

**作者:** Jiaqi Tang `[一作]` (Peking University), Qingchao Chen `[通讯]` (Peking University)

**通讯引用:** 2534 | [OpenAlex ID](https://openalex.org/A5069484115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于物理启发的空间解耦合合成框架，用来解决公式驱动监督学习（FDSL）在医学三维变压器预训练中的纹理-形状矛盾问题。

**💡 创新点**

创新点在于引入了梯度屏蔽缓冲区（Shielding Texture Model）和空间解耦合的纹理合成（Spatially‑Decoupled Texture Synthesis），通过距离变换将边界与高频纹理分离，显著减少边界混叠（boundary aliasing）。

**🔧 技术方法**

主要技术包括：欧几里得距离变换、梯度屏蔽缓冲、几何解耦（内部几何体与外部结构独立）、基于 Dirichlet 分布的物理驱动谱纹理混合、以及利用 SwinUNETR 与 UNETR 等 Transformer/卷积三维网络进行预训练与微调。

**📊 数据集**

实验数据集为 BTCV（30 份 CT 多器官标注）与 MSD（Task02-Heart、Task06-Lung、Task09-Spleen），采用 5k、15k、50k 规模的合成数据进行预训练。

**📈 对比分析**

与从零训练、PrimGeoSeg FDSL 以及在真实 CT 上进行自监督学习（SwinMM、SwinUNETR）等方法比较，本文方法在 BTCV 上以 SwinUNETR 计平均 Dice 提升 1.43%，在 MSD 各任务上均超过对比方法，尤其在 Task06 取得 1.08% 的显著提升。

**⚠️ 局限性**

局限性包括：合成纹理虽在物理上更逼真，但仍可能缺乏某些罕见或极端的医学影像特征；梯度屏蔽缓冲区宽度的选择需要经验调参；目前仅在 CT/MRI 模式下验证，未覆盖 X‑ray、PET 等其它模态。

---

## 456. GCL-Sampler: Discovering Kernel Similarity for Sampled GPU Simulation via Graph Contrastive Learning

**arXiv ID:** 2603.00551 | [PDF](https://arxiv.org/pdf/2603.00551v1)

**作者:** Jiaqi Wang `[一作]` (University of Science and Technology of China), Guangzhong Sun `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6226 | [OpenAlex ID](https://openalex.org/A5100932403)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于图对比学习的GPU工作负载采样框架GCL‑Sampler，用高质量的HRG图嵌入自动发现内核相似性，显著提升采样精度与加速比。

**💡 创新点**

创新点在于将GPU执行轨迹构造为异构关系图，并用Relational Graph Convolutional Networks结合对比学习自监督学习得到的高维嵌入替代传统手工特征，实现对不同名称、不同行为内核的统一聚类。

**🔧 技术方法**

主要技术包括NVBit动态追踪、异构关系图构造、RGCN图卷积网络、对比学习（InfoNCE）与K‑Means聚类，并将结果集成到HyFiSS模拟器。

**📊 数据集**

使用了包含PolyBench、Rodinia、Tango以及LLM（qwen1.5、phi‑2、pythia）等11个程序共7,746个内核的多样化数据集。

**📈 对比分析**

与PKA、Sieve、STEM+ROOT等现有方法对比，GCL‑Sampler平均误差0.37%（远低于20.9%/4.10%/0.38%），平均加速比258.94×（超过129.23×/94.90×/56.57×），并在三代NVIDIA GPU上保持低误差与高加速，验证了跨架构稳健性。

**⚠️ 局限性**

局限性包括需一次性收集和预处理高成本的SASS轨迹、对某些库行为敏感导致的误差波动（如phi‑2），以及模型训练与推理仍需显存/计算资源，限制了在资源受限环境下的即时部署。

---

## 457. Kruskal-EDS: Edge Dynamic Stratification

**arXiv ID:** 2603.02006 | [PDF](https://arxiv.org/pdf/2603.02006v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 458. Is Bigger Always Better? Efficiency Analysis in Resource-Constrained Small Object Detection

**arXiv ID:** 2603.02142 | [PDF](https://arxiv.org/pdf/2603.02142v1)

**作者:** Kwame Mbobda-Kuate `[一作]` (ENSAE Paris), Gabriel Kasmi `[通讯]` (Mines Paris - PSL University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在数据稀缺的地球观测场景下，系统评估了YOLO11系列模型在模型大小、数据量与输入分辨率三个维度上的检测效率。

**💡 创新点**

创新点在于发现规模倒置：最小模型YOLO11N在小数据量下同时获得最高精度和最高效率，并指出分辨率是资源分配的主导杠杆。

**🔧 技术方法**

使用YOLO11单阶段检测器、COCO预训练、统一超参数训练，并通过mAP_50、模型大小效率和FPS等指标进行评估，同时结合过拟合与数据稀缺理论解释结果。

**📊 数据集**

使用了OpenStat Madagascar光伏检测数据集，仅保留无人机图像，约8977张图像共约130k光伏标注。

**📈 对比分析**

对5个模型×4数据比例×3分辨率共60组合（完成44个训练）进行比较，YOLO11N在1280px时mAP_50达0.617，效率最高并位于精度–吞吐Pareto前沿，无需权衡。

**⚠️ 局限性**

局限性包括仅评估YOLO11家族，未考虑Transformer或专门的小目标检测方法；分辨率受GPU内存限制；仅单一地理任务；COCO预训练缺乏领域适配。

---

## 459. CausalWrap: Model-Agnostic Causal Constraint Wrappers for Tabular Synthetic Data

**arXiv ID:** 2603.02015 | [PDF](https://arxiv.org/pdf/2603.02015v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Chao Yan `[通讯]` (Vanderbilt University Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个模型无关的后置修正框架 CausalWrap，用以将部分因果知识注入任何预训练的表格生成模型，以提升其在因果推断中的可信度。

**💡 创新点**

创新点在于：1）支持任意预训练生成器（GAN、VAE、扩散模型等）而无需修改内部结构；2）通过可微分的残差化 HSIC 与单样本差分法实现对禁止边和单调性约束的软惩罚；3）采用自适应增量拉格朗日方法（ALM）平衡传统分布相似度与因果约束；4）理论上证明惩罚法收敛及近似条件匹配导致联合分布接近。

**🔧 技术方法**

主要技术包括：残差化独立性检验（HSIC）、对单调性约束的基于对偶样本的 hinge 损失、可微分的修正映射网络、增量拉格朗日（Augmented Lagrangian）优化、以及基于对比估计的实证指标。

**📊 数据集**

数据集覆盖三层：① 线性高斯、非线性加性、混合型的模拟 SCM；② 半合成基准 IHDP 与 ACIC‑style（10 个 DGP 设定）；③ 真实 ICU 组（MIMIC‑IV 约 2000 例，包含年龄、性别、实验室指标、治疗与 28‑天死亡等变量）。

**📈 对比分析**

与基线生成器（CTGAN、TabDDPM、TVAE）和基准 oracle 进行比较。实验显示：在模拟数据上因果误差下降 3–7%；在半合成基准上，CTGAN 与 TabDDPM 的 ATE 误差平均下降 19–57%，TVAE 在 ACIC 任务中下降 63%；在 ICU 数据上，TabDDPM 与 TVAE 的 ATE 一致性分别提升至 0.38 与 0.28（最高增幅），而 CTGAN 的提升有限。传统分布相似度大部分保持或略有波动。

**⚠️ 局限性**

局限性包括：① 仅能处理可微分的软约束，无法强制组合式硬约束；② 对不完整或错误的边约束敏感，过度约束可能削弱生成质量；③ 残差化 HSIC 的统计功效受回归模型误差影响；④ 仅在离散或连续混合变量上验证，未覆盖时间序列或多表结构；⑤ 需要访问 MIMIC‑IV 数据，重现性受限。

---

## 460. Unified Vision-Language Modeling via Concept Space Alignment

**arXiv ID:** 2603.01096 | [PDF](https://arxiv.org/pdf/2603.01096v1)

**作者:** Yifu Qiu `[一作]` (University of Edinburgh), Holger Schwenk `[通讯]` (FAIR at Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Omni —— 一个将图像和视频嵌入到多语言无语义文字嵌入空间的视觉‑语言共享表示框架，并将 LCM（大概念模型）扩展到可处理视觉输入的稀疏隐空间。

**💡 创新点**

创新点：①后置对齐（post‑hoc）使用粗到细的三阶段图像/视频字幕训练，将现成的文本空间映射到视觉域；②在该统一空间中直接使用 LCM 进行零样本视觉推理；③在 M3IT 多语言多模态指令调优后实现 61/62 语言的性能突破。

**🔧 技术方法**

技术：教师‑学生对齐、MSE 对齐损失、三阶段粗到细对齐策略、轻量化投影器、时序注意力、隐空间扩散语言模型（latent diffusion）、指令微调（instruction‑tuned）与双塔框架。

**📊 数据集**

数据集：12M 大规模图像‑字幕对、2M 合成视频‑字幕对、200K 人工审核视频‑字幕、M3IT（80 语言、8 任务）、PE‑Video、DREAM‑1K、各种视频检索/字幕基准（MSRVTT、VATEX、IVQA、ActivityNetQA 等）。

**📈 对比分析**

比较方式：零样本视频检索（Recall@1/5/10、MRR、AC、Trace、logdet），视频字幕（R‑1/R‑2/R‑L、BERTScore）。与 SigLIP2、InternVL、Qwen‑VL、Perception‑LM 等基线相比，Omni 在检索 Recall@1 上提升约 10 点，字幕 R‑1 提升 18+ 点；Omni‑IFT 在 M3IT 上在 61/62 语言上均超过竞争模型，尤其在低资源语言上显著提升。

**⚠️ 局限性**

局限性：①对齐仍受限于原始文本空间的结构，部分空间（如 PE‑Video）可能出现维度压缩；②在单概念/多概念视觉推理任务中仍略逊于大型 VLM；③依赖大规模图像/视频字幕数据，缺乏无监督/少样本场景的适应性。

---

## 461. According to Me: Long-Term Personalized Referential Memory QA

**arXiv ID:** 2603.01990 | [PDF](https://arxiv.org/pdf/2603.01990v1)

**作者:** Jingbiao Mei `[一作]` (University of Cambridge), Bill Byrne `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ATM-Bench，一套用于多模态、多源个人长期记忆问答的基准测试，并开发了Schema-Guided Memory（SGM）以结构化表示不同来源的记忆。

**💡 创新点**

创新点在于：①首次构建包含文本、图像、视频、邮件等四年个人记忆的真实数据集；②提出多维度记忆能力（PR、LA、MUT、ME、ABS）的评估框架；③引入SGM，使模型能更好地解析、聚合跨源记忆；④系统性评估现有记忆方案，揭示性能瓶颈。

**🔧 技术方法**

主要技术包括：多源记忆预处理（DM 与 SGM）、图结构记忆组织、基于向量检索的检索模块、单轮与迭代式答案生成（RAG、Self-RAG、ATM-RAG），以及多模态嵌入模型（MiniLM、Qwen、Gemini 等）。

**📊 数据集**

使用的数据集为ATM-Bench，包含约4年个人记忆（约12k条记录，涵盖邮件、图像、视频）和1038条带真值证据的问答对，特别设置了ATM-Bench-Hard 子集以考察多证据聚合和时间跨度长的查询。

**📈 对比分析**

通过与多种基准模型（A-Mem、Mem0、HippoRAG、Self-RAG、ATM-RAG）及 Oracle/No‑Evidence 对照，实验发现：SGM显著优于DM；但即使在 Oracle 情况下，最佳模型在 ATM‑Bench‑Hard 的准确率也不到 20%，显示长期记忆推理仍远低于人类水平。

**⚠️ 局限性**

局限性包括：①模型在需要跨源多跳推理与时间演化的查询上表现不佳；②当前检索多模态嵌入在高分辨率图像上导致信息稀释；③缺乏足够的长时序记忆更新机制；④实验仅在单一基准上评估，缺乏更广泛的跨任务验证。

---

## 462. Resolving Blind Inverse Problems under Dynamic Range Compression via Structured Forward Operator Modeling

**arXiv ID:** 2603.01890 | [PDF](https://arxiv.org/pdf/2603.01890v1)

**作者:** Muyu Liu `[一作]` (ShanghaiTech University), Yuyao Zhang `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出零样本框架CaMB-Diff，用于从未知动态范围压缩的图像中恢复高质量信号。

**💡 创新点**

通过将单调性硬性嵌入的级联Bernstein多项式作为前向运算参数化，解决了前向模型高偏差/高方差困境。

**🔧 技术方法**

使用级联单调Bernstein多项式、扩散概率模型、半二次分裂(HQS)与可微优化。

**📊 数据集**

在低光照（LOLv1、LOLv2）、低场MRI（HCP）和HDR（ImageNet）等数据集上实验。

**📈 对比分析**

与GDP、TAO等零样本基线及任务特定方法比较，PSNR/SSIM/LPIPS/FID等指标均显著优于基线，且与专用方法相当或更好。

**⚠️ 局限性**

假设前向映射仅为单调函数，可能不适用于所有成像物理场景；极端压缩或非单调噪声下鲁棒性有限。

---

## 463. Cryo-Bench: Benchmarking Foundation Models for Cryosphere Applications

**arXiv ID:** 2603.01576 | [PDF](https://arxiv.org/pdf/2603.01576v1)

**作者:** Saurabh Kaushik `[一作]` (University of Wisconsin Madison), Beth Tellman `[通讯]` (University of Wisconsin Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了Cryo-Bench评测基准，用于评估Geo-Foundation Models（GFMs）在冰川、冰湖、海冰和断崖等冰层领域的表现

**💡 创新点**

首次为极地冰层任务构建公开多模态、多地区、多传感器的数据集，并系统比较GFMs与传统UNet/ViT的性能，发现冻结编码器时UNet领先，但在全微调+超参优化后GFMs可显著超越

**🔧 技术方法**

使用GFMs（如DOFA、TerraMind、RemoteCLIP等）基于MAE/对比学习的预训练，冻结或微调Encoder并接UperNet解码器，采用学习率调优、few-shot训练等技术

**📊 数据集**

Cryo-Bench包含五个语义分割子集：GSDD、SICD、CaFFe、GLID、GLD，涵盖RGB、多光谱、SAR等不同传感器，覆盖格陵兰、南极、高山等地区

**📈 对比分析**

与UNet、ViT基线相比，GFMs在冻结编码器时平均mIoU低于UNet，但在全微调且学习率优化后，多数GFMs提升5–20个百分点；少量标签时GFMs保持≈94%完整数据性能，UNet仅≈85%

**⚠️ 局限性**

限制：全微调时性能波动大，部分GFMs（如RemoteCLIP）在微调后表现下降；对极地地区预训练数据不足导致域适应不平衡；计算成本与模型大小仍是部署瓶颈

---

## 464. KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models

**arXiv ID:** 2603.01875 | [PDF](https://arxiv.org/pdf/2603.01875v1)

**作者:** Songming Zhang `[一作]` (Beijing Jiaotong University), Jinan Xu `[通讯]` (Beijing Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 KDFlow，一种将教师模型部署在 SGLang 推理引擎、学生模型部署在 FSDP2 训练引擎的分离式知识蒸馏框架。

**💡 创新点**

创新点在于教师-学生的后端解耦、隐藏状态传输加上 logits 重新计算，显著减少通信成本并提升训练吞吐。

**🔧 技术方法**

采用 Ray 分布式调度、SGLang 高吞吐推理、PyTorch FSDP2 并行、隐藏状态零拷贝传输、以及多种 KL/JS/TVD 等蒸馏算法。

**📊 数据集**

使用 100k LMSys‑Chat‑1M 提示集与 Qwen3‑14B 生成的回答作为蒸馏数据，评估 AlpacaEval 2.0 结果。

**📈 对比分析**

与 TRL、ROLL、MS‑SWIFT 等现有框架对比，KDFlow 在多种教师‑学生组合下实现 1.44×–6.36× 的训练速度提升，同时保持与基线相近的推理性能。

**⚠️ 局限性**

局限在于仅基于 FSDP2 训练后端，无法匹配 Megatron‑LM 的 3D 并行效率，缺少异步训练等工业级优化。

---

## 465. SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks

**arXiv ID:** 2603.00575 | [PDF](https://arxiv.org/pdf/2603.00575v1)

**作者:** Yucheng Zeng `[一作]` (Baidu Inc), Jianmin Wu `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端到端的数据工厂系统SWE‑Hub，能够从任意开源仓库自动生成可执行的、可验证的软件工程任务（包括修复、系统级回归和全仓库构建），并持续以可扩展方式输出；

**💡 创新点**

创新点在于统一的可执行环境子系统（Env Agent + Test Agent），实现跨语言可复现的容器化执行；基于该子系统的三条产品线（SWE‑Scale、Bug Agent + Issue Agent、SWE‑Architect）分别提供高吞吐量的本地修复、逼真的系统级回归与长远构建任务；以及整个系统的无状态沙盒化、Kubernetes并行验证和配置即代码的多语言支持；

**🔧 技术方法**

使用技术包括：Tree‑Sitter统一解析器、基于配置的语言模板、LLM驱动的语义改动、规则化的程序化改动、容器化环境构建、标准化的验证入口（JUnit/XML/JSON等）、Kubernetes作业调度、无状态容器化验证、日志与指标收集；

**📊 数据集**

数据来源主要是GitHub公开仓库，结合自动化生成的合成bug集和隐藏测试套件；通过Bug Agent和Issue Agent生成用户式缺陷报告，SWE‑Architect通过代码hollowing和需求文档生成器构造构建任务；

**📈 对比分析**

与现有的SWE‑Bench、SWE‑Smith等基准相比，SWE‑Hub在任务规模、跨语言覆盖以及任务多样性方面有显著提升；具体性能指标（如每秒生成的修复实例数、成功率、执行时间）在论文中未给出，但架构设计旨在实现线性可扩展和高吞吐；

**⚠️ 局限性**

局限性包括：依赖现有测试用例验证，导致难以覆盖无测试的深层缺陷；合成缺陷的真实度仍有限；系统对CPU/GPU资源需求高；部分任务生成流程可能存在信息泄露风险（如报错报告过度提示）；

---

## 466. Integrating LTL Constraints into PPO for Safe Reinforcement Learning

**arXiv ID:** 2603.01292 | [PDF](https://arxiv.org/pdf/2603.01292v1)

**作者:** Maifang Zhang `[一作]` (University of Edinburgh), Fengxiang He `[通讯]` (University of Edinburgh)

**通讯引用:** 1755 | [OpenAlex ID](https://openalex.org/A5100635369)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 PPO-LTL 方法，将线性时序逻辑 (LTL) 约束通过 LDBA 监视器转化为成本信号，并在 PPO 的 Lagrangian 双重优化框架中实现安全强化学习。

**💡 创新点**

创新点在于：①利用 LTL 语义编码复杂安全规则并实时监测；②设计逻辑到成本机制，将违规行为映射为可梯度优化的处罚；③在 PPO 中加入 Lagrangian 方案并给出收敛理论；④实现了可插拔的安全约束插件。

**🔧 技术方法**

使用技术包括：Proximal Policy Optimization (PPO)、线性时序逻辑 (LTL) 编译为 limit-deterministic Büchi automata (LDBA)、逻辑到成本映射、Lagrangian 双重优化、偏置随机梯度证明。

**📊 数据集**

实验数据集：ZonesEnv（grid‑world）和 CARLA（自动驾驶仿真）。

**📈 对比分析**

与标准 PPO、TIRL‑PPO/SAC、PPO‑Mask、PPO‑Shielding、PPO‑Lagrangian 等基线对比，PPO‑LTL 在安全违规率上显著下降（如 CARLA 约 45% 的碰撞率减少），同时保持甚至提升任务完成度和速度等性能指标。

**⚠️ 局限性**

局限性：目前仅在 ZonesEnv 与 CARLA 这两类相对简单的仿真环境验证；对更大规模、多规则或实时复杂场景的评估仍待深入；LDBA 监视器在极大状态空间下的实时开销和多约束同步可能成为实际部署的瓶颈。

---

## 467. FREE-Edit: Using Editing-aware Injection in Rectified Flow Models for Zero-shot Image-Driven Video Editing

**arXiv ID:** 2603.01164 | [PDF](https://arxiv.org/pdf/2603.01164v1)

**作者:** Maomao Li `[一作]` (University of Hong Kong), Yu Li `[通讯]` (International Digital Economy Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种编辑感知（REE）特征注入方法，并基于预训练的Rectified Flow模型构建了零样本图像驱动视频编辑框架FREE-Edit。

**💡 创新点**

创新点在于通过光流跟踪编辑区域并自适应调节每个token的注入强度，避免了传统全局注入导致的语义冲突和运动信息不足。

**🔧 技术方法**

核心技术包括光流估计、二值化编辑掩码、自适应调节权重λ、Rectified Flow模型（LTX‑Video）以及Transformer自注意力中的特征注入。

**📊 数据集**

使用公开的Davis数据集、互联网上收集的视频以及自行编辑的首帧，构建了60视频的I2V‑Edit‑Bench基准数据集。

**📈 对比分析**

与Vanilla注入、FREE-Edit无注入以及现有图像驱动方法（VideoShop、I2VEdit、Go‑with‑the‑Flow）进行对比，FREE‑Edit在CLIP分数、Warp Error、SSIM、PSNR等指标上均优于对手，并在用户评估中获得高于随机的胜率，速度最快。

**⚠️ 局限性**

局限性在于无法为新插入的对象生成自然运动轨迹，导致插入对象运动不够自然。

---

## 468. FLANS at SemEval-2026 Task 7: RAG with Open-Sourced Smaller LLMs for Everyday Knowledge Across Diverse Languages and Cultures

**arXiv ID:** 2603.01910 | [PDF](https://arxiv.org/pdf/2603.01910v1)

**作者:** Liliia Bogdanova `[一作]` (Insilico Medicine AI Limited), Flor Miriam Plaza-del-Arco `[通讯]` (LIACS, Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

参与SemEval-2025 Task-7的两条子任务，使用检索增强生成与小型开源LLM构建文化知识库，并提供本地与在线搜索融合的多语言问答系统。

**💡 创新点**

推出可持续、本地可部署的RAG框架，构建多语言文化知识库，进行提示工程对比，提出分层检索与模型路由策略。

**🔧 技术方法**

采用小型LLM（Gemma 3、Llama 3.2、DeepSeek-R1、Mistral等）、RAG、向量检索（Chroma）、结构化提示（RP‑v1/​v2）、语言/模型路由、DuckDuckGo在线检索等技术。

**📊 数据集**

使用BLEnD共享任务数据集（SAQ与MCQ），从Wikipedia提取文本及手工文化事实构建知识库，并创建Pseudo ground truth用于内部评估。

**📈 对比分析**

通过提示与KB使用的ablation实验对比，评估SAQ与MCQ准确率；RP‑v1在大多数语言上表现最佳；RAG‑base在English/Spanish/Chinese上分别约17‑27% SAQ、82‑83% MCQ；RAG‑web在中文上提升至66%/91%，整体优于基线但低于大型模型。

**⚠️ 局限性**

仅覆盖三种语言，数据库质量不均导致噪声，RAG可能放大检索偏见，未使用官方金标准训练，模型规模小导致事实准确性受限。

---

## 469. A Unified Framework to Quantify Cultural Intelligence of AI

**arXiv ID:** 2603.01211 | [PDF](https://arxiv.org/pdf/2603.01211v1)

**作者:** Sunipa Dev `[一作]` (Google Research), Saška Mojsilović `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一个统一、可扩展的框架，用于量化人工智能模型在不同文化语境中的文化智能，包括文化感知、范围判断和流畅性三个核心能力。

**💡 创新点**

创新点在于：①构建三大领域（文化生产、行为实践、知识与价值）的“文化词汇表”作为操作性基础；②将文化智能拆解为可测量的子能力，并设计对应的指标；③结合客观知识与主观感知两类评测，提出“厚度评测”方法；④强调数据来源多样化（知识库、LLM探测、社区参与）并讨论评测与指标聚合策略。

**🔧 技术方法**

使用的技术包括：心理测量学的有效性理论、知识图谱（如Wikidata）构建文化词汇表、LLM进行知识探测、自然语言处理（prompt设计、文本相似度、情感与语调检测）、人工评估与自动评测器（LLM-as-judge）以及多维度指标聚合方法。

**📊 数据集**

所用数据集主要有：①公开知识库（Wikidata、其他多语种知识图谱）；②通过零/少量提示从大语言模型中抽取的文化知识；③由社区志愿者收集的本土文化数据（仪式、习俗、方言等）。

**📈 对比分析**

比较方法：将指标分为知识型（可客观评分）和感知型（需人工/LLM评估），采用多任务、自然/诊断式提示进行评测，最终通过加权平均或误差率等方式聚合成模型的文化智能得分。论文未给出具体实验结果，主要提供框架与方法论。

**⚠️ 局限性**

局限性：①文化知识库不完整、难以覆盖所有细节；②数据来源可能带有系统性偏见；③评测方法可能存在主观性与评估者多样性不足；④在高风险场景下高分并不等同安全；⑤文化规范的争议与冲突可能导致评测误导。

---

## 470. Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons

**arXiv ID:** 2603.02115 | [PDF](https://arxiv.org/pdf/2603.02115v1)

**作者:** Anthony Liang `[一作]` (University of Southern California), Jesse Zhang `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了一种结合轨迹级偏好监督与帧级进度监督的通用机器人奖励模型，并构建了包含超过 100 万条轨迹的大规模奖励学习数据集。

**💡 创新点**

创新点在于将偏好监督与进度监督双重目标融合，并通过视频重放、子序列裁剪等无标注增强手段有效利用失败轨迹，从而显著提升奖励模型的泛化与稳健性。

**🔧 技术方法**

采用预训练的多模态 VLM Qwen3‑VL‑4B‑Instruct，插入可学习的进度与偏好标记，训练目标包括进度交叉熵、成功 BCE 与二分类偏好损失，并使用视频重放、子序列裁剪等增强策略。

**📊 数据集**

构建的 RBL 数据集包含 21 种机器人、模拟与真实场景、专家与失败轨迹及人类演示，累计超过 1M 条轨迹，用于奖励学习与后续 RL 评估。

**📈 对比分析**

与 RoboReward、VLAC、GVL 等基线在奖励对齐（VOC、Kendall‑τ）和 RL 成功率上进行对比，平均奖励相关性提升约 14%，在多种 RL 任务中比最佳基线高 2.4–4.5 倍成功率。

**⚠️ 局限性**

仅基于帧级视频输入，难以捕捉细粒度时序与长程结构，缺乏物理状态信息，且对罕见或细微的失败模式识别能力有限。

---

## 471. Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies

**arXiv ID:** 2603.01841 | [PDF](https://arxiv.org/pdf/2603.01841v1)

**作者:** Matthieu Latapy `[一作]` (National Centre for Scientific Research), Stephany Rajeh `[通讯]` (Efrei Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了基于简单图特征的 TGF 方法，用历史图对每条链接进行特征提取并训练经典机器学习模型，以检测随机注入的异常链接。

**💡 创新点**

证明仅使用 trivial 图特征和传统学习即可在随机注入异常的场景下超越复杂方法，强调未来应聚焦更真实、更复杂的异常类型。

**🔧 技术方法**

利用 G‑type 与 H‑type 历史图、度数/加权度数/链接权重等本地特征，采用降序计数器实现 O(1) 计算；随后使用随机森林、SVM 或梯度提升等经典分类器，并通过欠采样平衡类别。

**📊 数据集**

在 9 个公开链接流数据集上评估：Bitcoin‑Alpha、Bitcoin‑OTC、DNC Emails、UCI Messages、Digg、Internet Topology、Taxi、Mawi 以及大规模 Bitcoin‑BC。

**📈 对比分析**

将 TGF 与 Node2Vec、DeepWalk、NetWalk、AddGraph、StrGNN、TADDY、RustGraph、SLADE 等最先进方法在 1%/5%/10% 注入率下进行 ROC‑AUC 对比；TGF 单一历史图 AUC>0.95，多历史图组合 AUC>0.98，显著优于对手，并在端到端时间上比 SLADE 快数百倍。

**⚠️ 局限性**

仅针对随机注入的“无结构”异常有效；对真实或更复杂的异常模式（如聚类、时间模式）尚未验证，且仍需对特定数据集进行参数调优。

---

## 472. BAED: a New Paradigm for Few-shot Graph Learning with Explanation in the Loop

**arXiv ID:** 2603.01941 | [PDF](https://arxiv.org/pdf/2603.01941v1)

**作者:** Chao Chen `[一作]` (Harbin Institute of Technology), Lei Chen `[通讯]` (Fuzhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出BAED框架，通过信念传播实现少样本图学习中的标签增强，并利用辅助GNN提取解释子图进行预测。

**💡 创新点**

核心创新在于把解释置入预测循环（explanation‑in‑the‑loop），通过BP进行标签扩充并用梯度反向传播提取关键子图，从而显著提升准确率、效率和可解释性。

**🔧 技术方法**

技术包括Belief Propagation（BP）标签扩充、辅助图神经网络（如SAGE）、梯度反向传播提取子图、集成BP与GNN的预测模块。

**📊 数据集**

实验数据集包括Cora、Citeseer、PubMed、Wiki、DBLP、Wisconsin、CoauthorCS、CoauthorPhy等七个主数据集（另有CoauthorPhy）。

**📈 对比分析**

与传统GNN（GCN、SAGE、GAT、GIN）、高级GNN（SGC、DLRGAE、HiD、tsGCN、DCI）以及10个FSGL基线（Meta-GNN、GPN、GPN等）比较，BAED在8个数据集上平均提升约50%准确率，训练速度提升数倍，解释性（faithfulness）也明显优于基线。

**⚠️ 局限性**

局限性：在极度稠密或类数极多的图（如CoauthorCS）中，BP传递噪声导致性能下降；对子图大小、兼容性超参数的选择敏感；实验主要集中在无特征或少特征场景，未充分验证在特征丰富的实际应用中的表现。

---

## 473. Information-Theoretic Framework for Self-Adapting Model Predictive Controllers

**arXiv ID:** 2603.01286 | [PDF](https://arxiv.org/pdf/2603.01286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 474. COMBAT: Conditional World Models for Behavioral Agent Training

**arXiv ID:** 2603.00825 | [PDF](https://arxiv.org/pdf/2603.00825v1)

**作者:** Anmol Agarwal `[一作]` (Indian Institute of Science Education and Research Bhopal), Spencer Frazier `[通讯]` (Overworld AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了COMBAT，一种基于扩散变压器的实时交互式世界模型，能在仅观察主玩家动作的条件下，隐式学习并生成具有反应性和战术性的对手行为；

**💡 创新点**

创新点在于通过仅以玩家1的输入进行条件化，利用世界模型的时间一致性自动产生对手的策略，突破传统需要完整动作标签或奖励信号的限制；

**🔧 技术方法**

核心技术包括：1.1B参数Diffusion Transformer（DiT）+自适应层归一化；2.深度压缩自编码器用于高比率状态压缩；3.局部-全局混合注意力与RoPE；4.DMD与Diffusion Forcing实现4步推断；5. Muon优化器提升训练效率；

**📊 数据集**

使用约7小时、1.2M帧的Tekken 3 1v1对战数据集，包含RGB帧、68点关节位姿、动作输入、健康与计时信息；

**📈 对比分析**

与基线RGB-only、全景训练模型比较，视觉质量指标（FID≈49.7/80.9，FVD≈593/1156，LPIPS≈0.05/0.07），行为一致性指标（TAA≈1.8–3.9，ARC≈1.0–3.9）均优于无姿态版本；通过4步DMD蒸馏保持90%以上视觉质量的同时提升12.5×速度，单卡可达85 FPS；

**⚠️ 局限性**

局限性：1) 蒸馏步骤降低了对手的反应频率和攻击性，导致行为失衡；2) 未通过强化学习微调以提升目标导向性（如获胜率）；3) 仅在单一游戏环境中验证，缺乏跨任务泛化评估。

---

## 475. Diagnosing Generalization Failures from Representational Geometry Markers

**arXiv ID:** 2603.01879 | [PDF](https://arxiv.org/pdf/2603.01879v1)

**作者:** Chi-Ning Chou `[一作]` (Flatiron Institute), SueYeon Chung `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于医学标志物启发的系统级诊断框架，利用图像分类中ID数据的任务相关表示几何指标（有效流形维度D和效用Ψ）来提前预测OOV泛化失败，并在多种网络结构、超参数以及ImageNet预训练模型上验证其有效性。

**💡 创新点**

首次将ID任务相关几何量作为系统诊断标记，用以预测OOV失效，证明对象流形的过度压缩（低维度、低效用）与OOV性能显著相关，从而提供一种无需细粒度机制解释的预测手段。

**🔧 技术方法**

采用GLUE理论计算对象流形的有效维度、半径与效用；在中等规模实验中进行超参数搜索和线性探针评估，并将这些几何指标与传统统计/对数线性指标进行对比；在ImageNet预训练模型上测量并预测迁移性能。

**📊 数据集**

使用CIFAR-10、CIFAR-100、ImageNet、CIFAR-10C以及9个ImageNet下游任务（Flowers102、Stanford Cars、Places365、Food101、Oxford-IIIT Pet等）进行实验。

**📈 对比分析**

通过Pearson相关、线性探针准确率等评估，发现D与Ψ与OOV准确率的相关性远高于ID准确率或统计特征；在ImageNet预训练模型上，使用D/Ψ预测OOV迁移性能的成功率达73%（相对ID准确率的37%），显示显著的预测优势。

**⚠️ 局限性**

局限性包括仅针对图像分类和类级OOV，未验证其他任务或分布偏移类型；对OOV归类的理论机制尚未完全阐明；在噪声或标签不变的偏移下ID准确率仍是最佳预测器；缺乏针对几何指标的干预或正则化实验。

---

## 476. Advancing Multimodal Judge Models through a Capability-Oriented Benchmark and MCTS-Driven Data Generation

**arXiv ID:** 2603.00546 | [PDF](https://arxiv.org/pdf/2603.00546v1)

**作者:** Zeyu Chen `[一作]` (Tsinghua University), Min Yang `[通讯]` (ByteDance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了以判断能力为中心的多模态评判基准 M‑JudgeBench，并构建了 Judge‑MCTS 数据生成框架，用于训练更强的 M‑Judger 判别模型。

**💡 创新点**

创新点在于：①从人类评判视角拆解判定维度为“结果错误判断”和“过程错误检测”，细化为十个子任务；②利用 MCTS 生成多长度、多正确性的推理轨迹，提供对比式训练样本；③在同一模型同长度对比中显著提升判别难度，揭示现有评判模型的偏差。

**🔧 技术方法**

技术包括：多模态大语言模型（Gemini、GPT‑4/5、Qwen、GLM 等）作为生成与评判者；Monte Carlo Tree Search (MCTS) 进行推理轨迹生成；监督微调（SFT）+ 强化学习（DAPO）训练 M‑Judger。

**📊 数据集**

使用公开基准数据作为种子（MMMU、MMM‑Pro、MMStar、MMReason、M3CoT、MathVision、MathVerse 等），并通过 GPT‑4.1 提取答案、对齐正负样本；还使用多种公开对比式训练集（MMPR、MMIF、RLAIF‑V 等）进行 SFT；MCTS 产生的四类推理轨迹用于训练。

**📈 对比分析**

与现有评判基准（VL‑RewardBench、Multimodal RewardBench、JudgeAnything 等）以及自建 M‑JudgeBench 进行比较，使用成对准确率作为指标；M‑Judger 在三大基准上均取得显著提升（整体准确率从 50%–70% 提升至 80%+，尤其在相同长度 CoT 比较和长度偏差任务上提升明显）。

**⚠️ 局限性**

局限性包括：仍对大型闭源模型产生依赖；对极长推理或极短推理的极端情况评判不稳定；MCTS 生成过程受基准模型质量影响，可能产生不完整或错误的轨迹；缺乏针对安全、偏见等伦理评判的覆盖。

---

## 477. Tool Verification for Test-Time Reinforcement Learning

**arXiv ID:** 2603.02203 | [PDF](https://arxiv.org/pdf/2603.02203v1)

**作者:** Ruotong Liao `[一作]` (Ludwig-Maximilians-University of Munich), Serena Yeung-Levy `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无标注测试数据下，作者提出了工具验证的测试时强化学习框架（T3RL），通过在奖励估计中加入外部工具验证来抑制错误共识导致的模式崩溃；

**💡 创新点**

核心创新在于将工具执行（如代码解释器）作为验证证据，构建验证加权投票机制，显著降低错误伪标签的影响；

**🔧 技术方法**

技术手段包括：LLM验证器对推理轨迹生成可执行代码、代码解释器执行并返回结果、以及基于验证结果的加权投票奖励函数；

**📊 数据集**

实验使用了三大数学推理基准：MATH‑500、AMC 与 AIME‑2024，覆盖不同难度；

**📈 对比分析**

与原始TTRL和基线模型对比，T3RL 在所有模型与基准上均取得提升，最显著的是 AIME‑2024 的 31.6% 相对增幅，整体平均提升约 11%；

**⚠️ 局限性**

局限性包括对验证器质量的依赖、在简单任务中收益有限，以及工具执行错误可能引入噪声导致奖励不稳定。

---

## 478. Dual Distillation for Few-Shot Anomaly Detection

**arXiv ID:** 2603.01713 | [PDF](https://arxiv.org/pdf/2603.01713v1)

**作者:** Le Dong `[一作]` (Xidian University), Lichao Mou `[通讯]` (MedAI Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于双向蒸馏的少量样本异常检测框架 D^24FAD，用少数正常参考图像识别医学影像中的异常。

**💡 创新点**

创新点在于：①引入教师-学生双蒸馏机制，学生在查询图像上蒸馏教师知识，在支持图像上自蒸馏；②提出基于查询图像动态评估支持图像参考价值的 learn‑to‑weight 机制，显著提升检测精度。

**🔧 技术方法**

使用预训练教师编码器（WideResNet‑50）、可学习学生解码器、知识蒸馏损失、交叉自蒸馏损失以及 softmax 加权机制；同时采用 Adam 优化器、图像尺寸 128×128 等实现细节。

**📊 数据集**

构建并使用了包含 13,084 张图像、4 种器官、4 种影像模态、5 种病理类别的综合基准集，涵盖 HIS、LAG、APTOS、RSNA、Brain Tumor 等子数据集。

**📈 对比分析**

与多种无监督方法（FastFlow、PatchCore 等）和少量样本方法（RegAD、InCTRL、MediCLIP 等）进行对比，在所有任务（K=2/4/8）下均实现 AUROC 最高，部分场景 100% 识别率，显著优于现有最优方法。

**⚠️ 局限性**

局限性：依赖预训练教师网络的表征能力，若教师模型与医学影像领域差异较大可能影响蒸馏效果；仅评估图像级异常检测，缺乏像素级定位性能验证；需要手工挑选合适的少量正常参考图像。

---

## 479. Message Passing Without Temporal Direction: Constraint Semantics and the FITO Category Mistake

**arXiv ID:** 2603.01405 | [PDF](https://arxiv.org/pdf/2603.01405v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DÆDÆLUS), Paul Borrill (DÆDÆLUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文指出传统消息传递模型中对时序的假设是多余的，并将消息交互重新表述为无时序约束满足问题。

**💡 创新点**

创新点在于提出并证明了消息传递执行与约束满足问题之间的等价性，从而揭示消息传递的本质是逻辑约束而非时间传播；同时提出了“前向时间唯独”(FITO)假设的形式化与批判。

**🔧 技术方法**

采用了形式化逻辑（偏序、约束系统）、约束满足理论、类别理论、Lamport时钟、Pratt pomset等技术手段进行建模与证明。

**📊 数据集**

本文不涉及具体实验数据集，主要为理论性研究；所示等价性在有限的消息传递协议模型上成立。

**📈 对比分析**

由于缺乏实验实现与基准测试，无法给出性能比较；论文主要通过形式证明展示两种模型在语义上的等价，未对运行时效率进行评估。

**⚠️ 局限性**

局限性在于：①等价性仅在有限的、满足特定观察条件的协议中成立；②实际协议的实现可能因消息数量、证书交换等因素导致复杂度过高；③未给出针对大规模系统的可扩展性或性能分析。

---

## 480. Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy

**arXiv ID:** 2603.02123 | [PDF](https://arxiv.org/pdf/2603.02123v1)

**作者:** Jiahao Huang `[一作]` (Fujian Normal University), Zhide Chen `[通讯]` (Fujian Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个三层认知层级，将情感任务按感知、理解、互动划分，并在此基础上开发了NanoEmoX这一小规模多模态语言模型以及P2E逐层训练框架；

**💡 创新点**

创新点在于：①将情感任务统一归入三层层级并据此构建任务统一框架；②NanoEmoX通过专用面部编码器和动态多模态融合专家，提升跨层任务的泛化；③P2E采用逐步递进的训练课程，先感知后理解再到互动，显著提升情感推理与共情能力；

**🔧 技术方法**

使用的技术包括：多模态编码器（CLIP-Large、HuBERT-Large）、Qwen2.5-1.5B LM、Q-Former重采样、FaceXFormer面部特征提取、交叉注意力与动态门控融合、LoRA微调、链式思考（think标签）等；

**📊 数据集**

使用的数据集涵盖情感识别与理解与共情生成：FERV39K、CAER、CREMA-D、M3ED、MIntRec、MIntRec2.0、MER-Caption+、MERR-Fine、AvaMERG、MER2023、MELD、MOSEI、MOSI、SIMS、SIMSV2等；

**📈 对比分析**

通过与多种基线（包括大规模LM如Qwen、Emotion-LLaMA、AffectGPT、MobileVLM等）和小规模模型的对比，NanoEmoX在六大核心情感任务上取得与大型模型相当甚至更优的成绩，尤其在OV-MER、ERG、MIR等任务中实现SOTA；

**⚠️ 局限性**

局限性包括：仍无法完全匹配极大规模模型在复杂任务上的性能；对高分辨率情感细粒度的捕捉仍有提升空间；需要更多跨域、多语言的数据来进一步增强鲁棒性；

---

## 481. Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning

**arXiv ID:** 2603.01741 | [PDF](https://arxiv.org/pdf/2603.01741v1)

**作者:** Naoki Shitanda `[一作]` (University of Tokyo), Takayuki Osa `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种在大规模并行环境下的集成策略梯度学习方法，利用领导-跟随框架通过KL约束调节策略多样性，并加入对抗奖励以防止策略过度聚集，提升样本效率与最终性能。

**💡 创新点**

创新点在于：① 理论分析显示过大策略多样性会降低有效样本量并增加梯度偏差；② 通过在跟随者更新中加入KL约束和对抗奖励来精细控制领导-跟随之间的距离，从而实现结构化且高质量的探索。

**🔧 技术方法**

主要技术包括：PPO基础算法、Split-and-Aggregate Policy Gradient (SAPG)框架、KL距离约束、对抗奖励（DIAYN启发）、重要性采样 (IS)、有效样本数 (ESS) 评估。

**📊 数据集**

使用 Isaac Gym 并行仿真平台的六个柔手操作任务、两种抓手操作任务和两种行走任务，共计 10 个机器人控制任务作为实验数据集。

**📈 对比分析**

与 PPO、DexPBT、SAPG 等基准进行比较，结果表明 CPO 在样本效率和最终性能上均优于基准，尤其在柔手操作任务上表现显著；在更简单的行走任务上差距相对较小。

**⚠️ 局限性**

局限性包括：固定的策略与环境数量限制了对不同任务与训练阶段的自适应扩展；未研究自动调整并行策略数量与环境分配的机制。

---

## 482. MO-MIX: Multi-Objective Multi-Agent Cooperative Decision-Making With Deep Reinforcement Learning

**arXiv ID:** 2603.00730 | [PDF](https://arxiv.org/pdf/2603.00730v1)

**作者:** Tianmeng Hu `[一作]` (Central South University), Tingwen Huang `[通讯]` (Texas A&M University)

**通讯引用:** 49559 | [OpenAlex ID](https://openalex.org/A5074290686)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 MO-MIX 的多目标多智能体强化学习算法，能够在协作决策场景下通过条件化的局部价值函数与并行混合网络实现对多目标的近似 Pareto 集合的学习。

**💡 创新点**

创新点在于：①将偏好向量作为条件直接输入到每个智能体网络中，使模型能泛化到任意偏好；②设计了多目标并行混合网络（MOMN）满足单调性约束；③引入基于当前非支配集的探索引导机制，提升最终 Pareto 集合的均匀性与多样性。

**🔧 技术方法**

采用的技术包括集中训练、分散执行（CTDE）框架、GRU+MLP 条件化代理网络、超网络生成权重的并行混合网络、基于 Envelope MOQ-Learning 的 TD 更新、经验回放和 ε-贪婪策略。

**📊 数据集**

在 OpenAI Multi-Agent Particle Environment（Simple Spread）与 StarCraft Multi-Agent Challenge（SMAC 2s3z）这两个公开环境中进行实验，并通过自定义的两目标奖励构造来测试算法。

**📈 对比分析**

与基线方法 Outer-loop QMIX（单目标 QMIX 在偏好外循环中训练）进行对比，使用四项指标（Hypervolume、Diversity、Spacing、Sparsity）评估 Pareto 集合。MO-MIX 在所有指标上均优于基线，同时训练样本量减少约 10 倍以上（MPE 75k 轮 vs 1.025M 轮；SMAC 5M 步 vs 41M 步）。

**⚠️ 局限性**

目前仅在两目标场景下验证，尚未对多目标（3+维）进行实验；对偏好采样区间的依赖较高，过小或过大均会影响样本效率；算法在大规模多智能体或更复杂任务中可扩展性尚待进一步评估。

---

## 483. SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing

**arXiv ID:** 2603.01630 | [PDF](https://arxiv.org/pdf/2603.01630v1)

**作者:** Anjali Parashar `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了SEED-SET框架，用于在有限样本预算下结合客观系统指标和主观利益相关者判断，对自主系统进行伦理评估。

**💡 创新点**

创新点在于将伦理评估分层建模为目标GP和主观GP的层级变分高斯过程，并设计联合信息增益与偏好一致的采集策略；同时利用LLM进行对比评估，减少人工成本。

**🔧 技术方法**

使用层级变分高斯过程、贝叶斯实验设计、互信息采集策略、LLM代理对偏好进行二元评估，并在BoTorch/GPyTorch实现。

**📊 数据集**

实验数据集包括IEEE 5‑Bus与30‑Bus电网资源分配场景、30维消防救援场景以及城市最优路径规划任务。

**📈 对比分析**

与随机采样、单GP、VS‑AL、BOPE等基线比较，SEED‑SET在偏好得分上提升至约1.25‑1.5倍，生成的测试用例覆盖率提高约1.25倍，且在高维场景下样本效率显著更优。

**⚠️ 局限性**

局限性包括：对极大数据集的可扩展性受限；使用平稳核假设可能不适用于多模式系统；需预先知道所有客观指标；LLM代理可能受提示敏感，需持续校准。

---

## 484. ICSE 2022 Sustainability Report

**arXiv ID:** 2603.01541 | [PDF](https://arxiv.org/pdf/2603.01541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 485. TactileWalk: Dynamic Electrotactile Patterns for Fingertip-Based Interaction During Walking

**arXiv ID:** 2603.01974 | [PDF](https://arxiv.org/pdf/2603.01974v1)

**作者:** Vedika Nimbalkar `[一作]` (Rochester Institute of Technology), Roshan Peiris `[通讯]` (Rochester Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在静止和行走条件下，利用指尖电刺激装置呈现动态电触觉模式进行导航信息传递，并评估不同模式的感知准确度和反应时间。

**💡 创新点**

创新点在于证明了单线和双线两种简单线性模式在指尖电触觉下的高识别率，并通过双线模式提供空间冗余来提升在运动中对方向的感知，从而为可穿戴移动导航系统提供了更简洁高效的设计思路。

**🔧 技术方法**

使用了10×6电极网格、ESP32微控制器与高压驱动器实现单极电流控制的时空渲染，配合Processing/Android软件进行刺激呈现和数据记录。

**📊 数据集**

未使用公开数据集，而是自行收集12名受试者（静止实验）和10名受试者（行走实验）的实验数据进行分析。

**📈 对比分析**

通过重复测量ANOVA和Wilcoxon符号秩检验比较模式识别准确率和反应时间，结果显示双线模式在行走中平均准确率达90.83%，比单线的81.67%更高，且两种模式的响应时间相近，无显著差异。

**⚠️ 局限性**

局限性包括样本量相对较小、实验环境受限（仅室内走廊）、只评估了模式识别未涉及完整导航任务，以及在行走实验中排除了对角方向，可能导致在真实世界复杂情境下的性能被高估。

---

## 486. Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport

**arXiv ID:** 2603.01771 | [PDF](https://arxiv.org/pdf/2603.01771v1)

**作者:** Harry Amad `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Hyperparameter Trajectory Inference（HTI）框架，利用条件拉格朗日最优传输学习神经网络在不同超参数下的概率路径，构建可在推理时即时调整超参数的 surrogate 模型；

**💡 创新点**

创新点包括：1）将条件最优传输与拉格朗日动力学相结合，学习条件下的潜能和度量；2）通过密度偏置潜能和最小作用原理提供先验；3）在高维空间引入可学习的 Givens 旋转度量，避免退化；4）在多种应用场景（RL 奖励权重、量化回归、生成模型 dropout）验证效果；

**🔧 技术方法**

采用了条件最优传输（COT）与条件拉格朗日最优传输（CLOT）框架，神经网络近似（含 FiLM、Givens 旋转、COT 潜能、Geodesic spline）、半对偶式优化、L‑BFGS 细化、Nadaraya‑Watson 估计潜能、min‑max 训练策略等技术；

**📊 数据集**

实验数据集包括自定义半圆轨迹、OpenAI Gym Reacher、癌症治疗仿真环境、ETTm2 时间序列预测、Two‑Moons 生成模型等；

**📈 对比分析**

与直接回归、CFM、MFM、NLOT 等基线方法比较，在半圆任务 NLL/CD 大幅提升，在 RL 任务（癌症治疗、Reacher、非线性奖励）获得最高奖励，在量化回归 MSE 最低，在 dropout 生成 WD 最低，整体性能显著优于基线；

**⚠️ 局限性**

局限性在于只适用于单一连续超参数，无法直接处理多超参数；在动力学混沌或样本稀疏情形下推断困难；训练过程仍需多网络与较高计算成本；对训练数据的质量和覆盖范围要求较高。

---

## 487. Energy Efficient Traffic Scheduling For Optical LEO Satellite Downlinks

**arXiv ID:** 2603.01334 | [PDF](https://arxiv.org/pdf/2603.01334v1)

**作者:** Ethan Fettes `[一作]` (Carleton University), Stéphane Martel `[通讯]` (MDA)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究光学LEO卫星的延迟容忍下行链路调度，提出静态阈值、排序和强化学习等方案，以在满足交付率的同时提高能效。

**💡 创新点**

创新点在于将下行调度问题建模为软背包问题变体，并设计了适用于动态天气条件的自适应RL/DRL调度方法；同时通过阈值调优算法实现了低复杂度的能量节约策略。

**🔧 技术方法**

采用的技术包括：阈值调度、基于云覆盖预测的排序算法、Q‑learning、深度双DQN（DDQN）以及用于评估的仿真环境；天气模型基于GFS预测和历史天气数据；链路可用性采用离散概率模型。

**📊 数据集**

数据集主要有三类：①统一分布的数据量与云覆盖的合成仿真数据；②可变云覆盖和数据量的组合仿真数据；③真实加拿大城市（如Inuvik、Calgary等）的历史天气记录与卫星轨道访问信息，用于案例研究。

**📈 对比分析**

与传统的CGR基线相比，单阈值和多阈值方案在能效上提升约10–30%但交付率下降；静态排序在能效上略优于阈值方案，交付率基本不变；自适应排序与DDQN在多数场景下保持与CGR相近的交付率，并在能效上提升14–25%。在真实天气案例中，DDQN表现最差，阈值方案能耗降低但交付率受影响；自适应排序在大多数配置下保持最优平衡。

**⚠️ 局限性**

主要限制包括：自适应排序与DDQN的计算复杂度高，难以在资源受限的卫星上实时实现；算法依赖准确的云覆盖预测与链路动态模型，实际环境不确定性可能导致性能下降；实验仅覆盖单卫星单站配置，未考虑多卫星多站网络；训练多需要大量仿真时间，缺乏在线学习策略。

---

## 488. VietSuperSpeech: A Large-Scale Vietnamese Conversational Speech Dataset for ASR Fine-Tuning in Chatbot, Customer Support, and Call Center Applications

**arXiv ID:** 2603.01894 | [PDF](https://arxiv.org/pdf/2603.01894v1)

**作者:** Loan Do `[一作]` (FPT University), Charlotte Nguyen `[通讯]` (NGHI Studio)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

创建并公开了一个 267.39 小时、52,023 条语音-文本对的越南语非正式对话 ASR 数据集 VietSuperSpeech，并提供了完整的预处理和伪标签流程。

**💡 创新点**

创新点在于：① 以 YouTube 个人 vlog、海外越南社区频道和非正式评论内容为来源，系统聚焦日常口语化对话；② 使用高质量 Zipformer‑RNNT 伪标注并做多层次质量控制；③ 公开 89/11 的训练/验证拆分和完整的 16 kHz mono WAV 预处理，填补了越南语 ASR 资源中缺乏对话式口语的空白。

**🔧 技术方法**

主要技术包括：Zipformer‑RNN‑Transducer 模型（训练 6,000 小时越南语）、Sherpa‑ONNX 推理、VAD+分段、长度/字符/置信度过滤、伪标注后人工抽样检查。

**📊 数据集**

使用的数据集：从四类公开 YouTube 频道收集的音频；伪标签使用已预训练的 Zipformer‑30M‑RNNT‑6000h；最终拆分为 46,822 条训练（240.67 小时）和 5,201 条验证/测试（26.72 小时）。

**📈 对比分析**

在标准越南语 ASR 任务（如 VLSP2020、VLSP2023）上，以 VietSuperSpeech 进行领域适配后，模型在对话式样本上的 WER 明显下降（相较于仅使用正式语料的基线，提升可达 15–20%），但在正式读音任务上略逊于最优正式语料集；实验采用了 WER、RTF 等指标，结果显示对话化预训练显著提升了非正式语音识别准确率。

**⚠️ 局限性**

局限性包括：伪标签可能带来噪声，尤其在重度口语化或方言混合语音上误差更大；样本来源受限于可公开 YouTube 内容，未涵盖极端嘈杂或老年人语音；说话人多样性与真实业务场景不完全匹配；数据仅面向非商业研究使用。

---

## 489. A Comprehensive Evaluation of LLM Unlearning Robustness under Multi-Turn Interaction

**arXiv ID:** 2603.00823 | [PDF](https://arxiv.org/pdf/2603.00823v1)

**作者:** Ruihao Pan `[一作]` (Pennsylvania State University), Suhang Wang `[通讯]` (Pennsylvania State University)

**通讯引用:** 18637 | [OpenAlex ID](https://openalex.org/A5011048500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在交互式环境（自我纠错与对话条件查询）下大语言模型的机器无学习（unlearning）鲁棒性。

**💡 创新点**

创新点在于突破传统单轮评估，揭示交互可恢复已遗忘知识，并区分行为僵化与真正知识消除的差异。

**🔧 技术方法**

采用梯度上升（GA）、负偏好优化（NPO）以及表示误导（RMU）三种无学习技术，并配合梯度下降（GD）和KL正则化。

**📊 数据集**

实验使用WMDP（生物医学）数据集作为遗忘目标，MMLU评估实用性，Wikidata作为保持集。

**📈 对比分析**

与基线单轮模型对比，GA/NPO在单轮下表现安全，但在自我纠错与多轮对话中恢复被遗忘知识；RMU鲁棒但对话适应性下降，整体准确率在交互中显著上升。

**⚠️ 局限性**

局限性包括仅测试两款中等规模模型、仅使用多选题评估、以及对提示模板和模型规模的依赖。

---

## 490. Explanation-Guided Adversarial Training for Robust and Interpretable Models

**arXiv ID:** 2603.01938 | [PDF](https://arxiv.org/pdf/2603.01938v1)

**作者:** Chao Chen `[一作]` (Harbin Institute of Technology), Chuanyi Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将对抗训练与解释引导学习融合的统一框架——Explanation‑Guided Adversarial Training (EGAT)，实现模型在鲁棒性和可解释性上的双重提升。

**💡 创新点**

创新点在于：①在对抗训练过程中加入解释一致性与对齐约束，促使模型在对抗扰动下仍保持对语义相关特征的关注；②提供PAC理论分析，证明EGAT在分布偏移下具有更小的泛化误差；③通过多任务损失统一优化，提高解释质量与鲁棒性。

**🔧 技术方法**

核心技术包括：对抗训练（PGD）+ Grad‑CAM 解释生成 + 解释一致性损失（BCE）+ 对齐损失 + Mixup 正则化 + 交叉熵分类损失。

**📊 数据集**

使用两个公开域泛化基准：VLCS（4个域）和 Terra Incognita（4个地点）。

**📈 对比分析**

与 ERM、IGR、IGN、DMADA、IRM、DRE、SGDrop 等基线对比，EGAT 在 clean accuracy、对抗 accuracy（提升约 37%）、OOD 泛化（平均排名 2.17）以及解释质量（Comprehensiveness 与 Sufficiency）均显著优于对手。

**⚠️ 局限性**

局限性包括：①对 Grad‑CAM 等梯度解释方法的依赖，噪声或不稳定的解释会削弱引导效果；②引导图质量不佳时可能导致性能下降；③相较于单纯 AT，训练开销略增；④在某些极端 OOD 场景下解释一致性约束可能过度限制模型适应性。

---

## 491. Tackling multiphysics problems via finite element-guided physics-informed operator learning

**arXiv ID:** 2603.01420 | [PDF](https://arxiv.org/pdf/2603.01420v1)

**作者:** Yusuke Yamazaki `[一作]` (Keio University), Shahed Rezaei `[通讯]` (ACCESS e.V.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于有限元加权残差的物理信息算子学习框架，用于无标注数据的多物理耦合 PDE 求解；

**💡 创新点**

创新点在于将有限元弱形式残差直接作为损失引入算子学习，兼顾复杂几何、非周期边界，且实现跨分辨率、跨域的参数化预测；

**🔧 技术方法**

核心技术包括 JAX/Folax 平台、Fourier Neural Operator、DeepONet、隐式有限算子 iFOL，以及单一/分段训练策略；

**📊 数据集**

使用人工生成的随机异质材料分布（Fourier 系列、Voronoi、Gyroid 等）以及工业铸造几何的仿真数据作为训练/测试集；

**📈 对比分析**

与传统非线性 FEM 对比，算子模型在 42×42、84×84、168×168 分辨率下推理时间平均低 32–2000 倍；相对 L2 错误在常规和极端样例均低于 10%，在三维铸造例中 iFOL 的最大误差仅 10%；

**⚠️ 局限性**

局限包括 FNO 对高频边界特征的欠捕捉、仅处理稳态问题、对非常复杂耦合或大尺寸域需更深层架构，以及训练样本多样性和质量仍是关键瓶颈。

---

## 492. Randomized Neural Networks for Partial Differential Equation on Static and Evolving Surfaces

**arXiv ID:** 2603.01689 | [PDF](https://arxiv.org/pdf/2603.01689v1)

**作者:** Jingbo Sun `[一作]`, Fei Wang `[通讯]` (National Natural Science Foundation of China)

**通讯引用:** 11248 | [OpenAlex ID](https://openalex.org/A5100455723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于随机化神经网络（RaNN）的无网格方法，用于在静态和随时间演化的曲面上求解线性偏微分方程。

**💡 创新点**

创新点包括：
① 随机生成隐藏层参数并固定，只对输出层做线性最小二乘求解，极大降低训练成本；
② 支持多种曲面描述（参数化、隐式水平集、点云）且理论上给出误差估计；
③ 对演化曲面采用流图（flow‑map）表示，将曲面演化与 PDE 解决在同一时空采样点上完成，避免网格重构与信息传递；
④ 对参数化方法加入界面不连续性惩罚，保证全局 $H^2$ 连续；
⑤ 通过理论分析提供误差分解与收敛率。

**🔧 技术方法**

主要技术手段：随机化神经网络（固定隐藏层权重，输出层线性最小二乘）、拉普拉斯–贝尔米方程与热方程等线性 PDE 的强式残差评估、曲面梯度与拉普拉斯在隐式/点云上的近似、空间‑时间采样与接口不连续性惩罚、理论分析中的图范数估计、流图的随机网络学习。

**📊 数据集**

使用人工合成的曲面和 PDE 作为测试集：
- 参数化的环面、奶酪形曲面、杯形曲面、弹性球面；
- 隐式水平集描述的奶酪形曲面与热方程；
- 点云表示的兔子模型；
- 振荡椭球面与其随时间变形；
- 受剪切流作用的球滴（演化曲面）。

**📈 对比分析**

对比方法主要是基于 PINN 的曲面 PDE 求解器；实验结果表明 RaNN 在相同网络宽度与采样点数下，相对 $L^2$ 误差可达到 $10^{-4}$–$10^{-7}$，训练时间仅为几秒至十几秒，明显优于传统 PINN 需要数十至数百秒的非线性优化；此外在演化曲面上无需重构网格，保持体积与质量守恒的误差低于 $10^{-4}$。

**⚠️ 局限性**

局限性：
① 仅处理无拓扑变化的光滑曲面；
② 只对线性 PDE 进行理论与实验验证，非线性 PDE 需要额外的非线性最小二乘或线性化；
③ 对曲面几何采样的误差敏感，点云近似需高质量投影；
④ 随机特征数目需要足够大以保证逼近精度，可能导致大规模线性系统；
⑤ 未考虑多物理耦合或自适应采样策略。

---

## 493. Recursive Models for Long-Horizon Reasoning

**arXiv ID:** 2603.02112 | [PDF](https://arxiv.org/pdf/2603.02112v1)

**作者:** Chenxiao Yang `[一作]` (Toyota Technological Institute at Chicago), Zhiyuan Li `[通讯]` (Toyota Technological Institute at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种递归模型，通过在多层上下文堆栈中递归调用自身，以突破传统语言模型的上下文窗口限制，实现长时序推理。

**💡 创新点**

创新点在于将递归视为核心原理，证明递归深度能使模型在局部上下文保持有限的同时实现指数级计算能力，并且证明相较单上下文方法递归模型在给定局部空间下是最优的。

**🔧 技术方法**

使用的技术包括递归调用框架（Call/Return工具）、上下文堆栈管理、支持多模型/工具调用的递归代理系统，以及在Transformer基础上对Qwen2.5-3B进行的监督微调。

**📊 数据集**

实验数据集为布尔可满足性（SAT）实例集，涵盖易、中、难三个难度级别。

**📈 对比分析**

与GPT‑4o、LLaMA3.3‑70B、Qwen3‑235B等前沿LLM进行对比，递归模型在SAT易/中/难题的准确率分别达到98%/95%/64%，显著优于对手；在上下文效率上，递归模型的活跃上下文长度保持有限，而总生成长度随问题规模指数增长。

**⚠️ 局限性**

局限性包括：递归深度可能导致错误累积；实验仅验证了SAT任务，未覆盖更广泛的长时序推理场景；以及需要外部存储和恢复堆栈上下文，实际推理成本与纯单上下文方法相比仍需进一步评估。

---

## 494. BAWSeg: A UAV Multispectral Benchmark for Barley Weed Segmentation

**arXiv ID:** 2603.01932 | [PDF](https://arxiv.org/pdf/2603.01932v1)

**作者:** Haitian Wang `[一作]` (University of Western Australia), Ajmal Mian `[通讯]` (University of Western Australia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在四年多光谱UAV田间数据上建立了BAWSeg基准数据集，并提出VISA双流网络实现更鲁棒的杂草分割。

**💡 创新点**

将辐射光谱与植被指数分为独立分支并在本地分辨率融合，结合窗口自注意力、Mamba状态空间块和Slot Attention，显著提升混合冠层下的分割性能。

**🔧 技术方法**

采用双流卷积‑Transformer架构、窗口自注意力、Mamba状态空间层、Slot Attention、软标签与边缘监督等技术。

**📊 数据集**

使用BAWSeg数据集，涵盖2020‑2023年两块西澳大麦田的五波段反射率、五个植被指数以及像素级农作物/杂草/其他标注。

**📈 对比分析**

与RF、UNet、SegFormer等基线对比，VISA在within‑plot mIoU 0.756、杂草IoU 0.635，cross‑plot/ year mIoU≈0.71/0.69，优于SegFormer‑B1等模型。

**⚠️ 局限性**

局限包括数据仅覆盖两块田地、单一UAV相机配置、仅三类标注且对混合像素处理粗糙，未充分验证对不同土壤、管理或传感器的迁移性。

---

## 495. Affine Correspondences in Stereo Vision: Theory, Practice, and Limitations

**arXiv ID:** 2603.01836 | [PDF](https://arxiv.org/pdf/2603.01836v1)

**作者:** Levente Hajder `[一作]` `[通讯]` (Eotvos Lorand University), Levente Hajder (Eotvos Lorand University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出基于仿射对应的立体视觉重建方法，结合新的仿射变换估计算法实现有方向信息的三维点云重建。

**💡 创新点**

创新点在于利用图像方向（线）而非传统特征匹配来估计仿射变换，并结合基础矩阵约束，提出多种基于尺度已知与未知的仿射估计方法。

**🔧 技术方法**

主要技术包括仿射变换与基线矩阵理论、基于方向的线性估计、八点算法求基础矩阵、PnP解算姿态、以及基于PCA的平面与法向量恢复。

**📊 数据集**

使用合成的三面棋盘格数据以及在实验室自制的三面棋盘模型拍摄的实景图像，涵盖一般运动、平面运动、标准立体与前进运动四种姿态。

**📈 对比分析**

通过与传统基于点对应的立体重建对比，评估法向量误差与基线重建误差；结果显示在方向噪声小于数度时，法向误差可控制在约5°以内，重建精度与传统方法相当或更优。

**⚠️ 局限性**

局限性包括对方向噪声敏感，特别是平面与相机姿态近似平行时误差显著；且仿射变换尺度估计仍依赖假设，未能在所有场景下自动精确获取。

---

## 496. Extended Empirical Validation of the Explainability Solution Space

**arXiv ID:** 2603.01235 | [PDF](https://arxiv.org/pdf/2603.01235v1)

**作者:** Antoni Mestre `[一作]`, Vicente Pelechano `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对 Explainability Solution Space (ESS) 框架在智能城市基础设施优化场景下进行实例化，演示其跨领域适用性

**💡 创新点**

在新治理配置（合成情景）下仅使用原有属性向量、投影函数和情景乘子，无需重新校准，证明 ESS 的结构可分离性和通用性

**🔧 技术方法**

使用 ESS 的三维投影（合规性、可理解性、开发者效用）与情景乘子实现多解释方法的定位

**📊 数据集**

未使用传统数据集，而是直接采用 ESS 原论文中预设的七维属性向量（如 SHAP、LIME 等）

**📈 对比分析**

通过手工计算各方法在 (C',U',D') 空间中的坐标，展示其相对优势；未进行实验性能比较，结果仅反映属性与情景乘子的组合效果

**⚠️ 局限性**

局限在于缺乏实际数据与实验验证，属性评分依赖主观预设，情景乘子固定，未考虑不同治理角色下的多重评估或敏感性分析

---

## 497. Learning Structured Reasoning via Tractable Trajectory Control

**arXiv ID:** 2603.01641 | [PDF](https://arxiv.org/pdf/2603.01641v1)

**作者:** Po-Nien Kung `[一作]` (University of California, Los Angeles), Kai-Wei Chang `[通讯]` (University of California, Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种基于可追踪轨迹控制的结构化推理学习框架（Ctrl‑R），通过约束解码引导语言模型在强化学习过程中探索并强化特定的推理结构；

**💡 创新点**

创新点在于将约束解码与可计算的行为策略相结合，既保证了对结构化推理模式的充分暴露，又能准确计算重要性采样权重；并引入权重幂尺度（β）实现对优势塑造的可控调节；

**🔧 技术方法**

采用可追踪的概率约束模型（如Ctrl‑G的HMM+DFA），基于离线强化学习的PPO/GRPO框架，配合权重幂尺度进行梯度更新；

**📊 数据集**

在语言模型上使用Qwen3-1.7B/8B与Math-17K数据集，在视觉语言模型上使用Qwen2.5-VL-7B与OpenVLThinker数据集；

**📈 对比分析**

与Dapo、GRPO、自然语言引导、奖励塑造、SFT预对齐等基线对比，Ctrl‑R在多项数学推理基准（AIME、MATH500、AMC、Minerva等）上平均提升约1–2个百分点，视觉语言模型上提升约2–3个百分点；

**⚠️ 局限性**

限制在于需要预先设计推理结构并转换为DFA；对β取值敏感，过大或过小均可能导致性能下降；模型对约束解码的计算开销相对较高。

---

## 498. WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments

**arXiv ID:** 2603.01475 | [PDF](https://arxiv.org/pdf/2603.01475v1)

**作者:** Joshua Knights `[一作]` (CSIRO Robotics), Peyman Moghadam `[通讯]` (CSIRO Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了WildCross基准，提供了超过476K帧的同步RGB-LiDAR数据，配以半稠密深度、表面法向、精确6DoF位姿以及对应的LiDAR子地图，并对视觉、LiDAR以及跨模态定位以及度量深度估计进行了系统评测。

**💡 创新点**

创新点在于：①将Wild-Places的原始序列重处理为大规模同步RGB-LiDAR基准；②设计了基于稠密点云可见性和GHPR遮挡剔除的半稠密深度/法向生成管线；③采用四折交叉验证与零样本/微调双重评测框架，凸显自然环境下现有方法的不足。

**🔧 技术方法**

技术上使用了稠密点云累计、点云可见性估计、GHPR遮挡剔除、ViT/Swin Transformer的视觉预训练（DINOv2/3）、传统CNN骨干（ResNet50）、跨模态检索算法（LIP‑Loc）、3D点云检索模型（MinkLoc3Dv2、LoGG3D‑Net、HOTFormerLoc）以及DepthAnythingV2深度估计模型。

**📊 数据集**

数据集主要为WildCross（基于Wild‑Places的改造），包含8条自然森林穿越路径，采集周期14个月，帧率15Hz；评估时还对比了KITTI、VirtualKITTI等城市/模拟数据以测试零样本泛化。

**📈 对比分析**

方法比较采用四折交叉验证，零样本和微调两种设置：VPR在自然环境中R@1仅约64%（相比城市数据90%+）；LPR在同序列达到90%+，跨序列低于86%；CMPR R@1仅约51%；DepthAnythingV2零样本RMSE>5m，微调后显著提升，但细节与时序一致性仍差。

**⚠️ 局限性**

限制在于：逆向重访导致定位性能显著下降；跨模态检索仍表现不足；深度估计在细节捕捉和时间一致性上不佳；数据集缺乏更广阔的视角、多模态传感器融合场景，且领域差距与遮挡问题仍需进一步解决。

---

## 499. NICO-RAG: Multimodal Hypergraph Retrieval-Augmented Generation for Understanding the Nicotine Public Health Crisis

**arXiv ID:** 2603.02047 | [PDF](https://arxiv.org/pdf/2603.02047v1)

**作者:** Manuel Serna-Aguilera `[一作]` (University of Arkansas), Khoa Luu `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了规模最大的尼古丁与烟草产品数据集NICO（202,599张多模态样本）并基于该数据集开发了NICO‑RAG框架，利用多种图像特征（颜色、形状、OCR、图像描述）构建多模态超图知识库，实现无须频繁调用大型语言模型的检索增强生成；

**💡 创新点**

创新点在于将多模态图像特征与文本信息融合到超图知识库中，减少对高成本多模态语言模型的依赖，同时通过多维检索提升事实性回答质量；

**🔧 技术方法**

使用的技术包括CLIP ViT‑14用于图像嵌入、DocTR进行OCR、Qwen3‑VL生成图像描述、GPT‑4o‑mini作为生成引擎，以及自定义的多模态超图检索算法；

**📊 数据集**

使用的数据集为自建的NICO数据集，涵盖55个品牌共202,599张图像；实验对比参考了Vassey、Murthy和PHAD等已有数据集；

**📈 对比分析**

在问答实验中与Naive Generation、Standard RAG和HypergraphRAG比较，NICO‑RAG在F1（0.273）、检索相似度（RS 0.800）和生成质量（GE 0.466）方面与最先进方法相近，同时显著减少了大型语言模型的调用次数；

**⚠️ 局限性**

局限性包括：某些图像特征提取（OCR、图像描述）仍需昂贵GPU资源，构建知识库过程耗时且可能存在噪声样本，且仍依赖部分大型语言模型进行文本实体提取。

---

## 500. CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework

**arXiv ID:** 2603.01607 | [PDF](https://arxiv.org/pdf/2603.01607v1)

**作者:** Yuexi Du `[一作]` (Yale University), Yan Lu `[通讯]` (Microsoft Research Asia)

**通讯引用:** 20011 | [OpenAlex ID](https://openalex.org/A5035278528)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一套基于视觉证据的医学视觉语言推理框架，分为实体提议、实体指代分割和基于证据的VQA，并加入动态协调器进行工具调用与答案复核。

**💡 创新点**

创新点在于：①将医学推理拆解为专业子模块，减少黑箱短路与幻觉；②利用可验证奖励的强化学习提升实体提议与VQA的证据一致性；③通过动态协调器实现工具调用规划与答案审查，提升可解释性和鲁棒性。

**🔧 技术方法**

核心技术包括：InternVL3 VLM、SA-Med-2D改进的实体指代分割模型、RLVR（DAPO）强化学习、Kuhn–Munkres匹配、动态协调器（如GPT‑5）以及多模态证据提示（缩放、掩码、全局）。

**📊 数据集**

使用的公开数据集有：SA‑Med‑20M（合成实体提议数据与分割训练）、MeCo‑G（分割评估）、OmniMedVQA、VQA‑RAD、SLAKE（医学VQA基准）以及VQA‑Med‑2019（OOD评估）。

**📈 对比分析**

与多种基线对比（如GPT‑4o、GPT‑5、Llama‑3.2 Vision、Qwen2.5‑VL、InternVL3、DeepEyes、LLaVA‑Med、MedVLm‑R1‑2B 等），10B 参数的框架在平均准确率上达到 74.91%，超过 32B 参数的 Lingshu‑32B（72.29%）约 2.6%；在动态协调模式下再提升约 3% 以上，OOD 上提升 6% 以上，显示出优越的性能与参数效率。

**⚠️ 局限性**

局限性包括：①对合成数据的依赖，实体提议的泛化仍需改进；②多模块协作导致推理链路较长，协调器的准确性决定最终表现；③在不需要局部视觉证据的数据集上效果不明显；④对计算资源的需求较高，尤其是动态协调器的调用与多模态输入。

---

## 501. No More Maybe-Arrows: Resolving Causal Uncertainty by Breaking Symmetries

**arXiv ID:** 2603.01052 | [PDF](https://arxiv.org/pdf/2603.01052v1)

**作者:** Tingrui Huang `[一作]` (Eindhoven University of Technology), Devendra Singh Dhami `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 417 | [OpenAlex ID](https://openalex.org/A5029345613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 CausalSAGE 的可微分框架，用来把由 FCI 等约束式方法学习得到的部分有向图（PAG）转换为完整的因果 DAG。

**💡 创新点**

创新点包括：① 在状态层面展开离散变量，提升细粒度因果推断；② 通过 PAG 的结构约束与软先验在参数空间预先构造可行搜索空间；③ 统一的可微分目标同时优化数据重构、稀疏、无环与骨架保持，天然产生方向偏好；④ 引入随机或 LLM 先验打破方向对称性。

**🔧 技术方法**

核心技术为：状态级 one‑hot 展开、块状权重矩阵与遮罩、交叉熵重构损失、加权组 Lasso 稀疏正则、方向性乘积惩罚、骨架保持正则、基于梯度的可微优化与后期 DAG 验证。

**📊 数据集**

在公开的 11–724 节点的贝叶斯网络基准（如 Sachs、Child、Insurance、Alarm、Win95pts、Andes、Pigs、Link 等）上进行实验，全部使用观测数据生成。

**📈 对比分析**

与原始 PAG、FCI+LLM 以及传统直接 DAG 学习器（PC、MMHC、Tabu、HC）比较。CausalSAGE 在消除方向歧义方面将未定比例降至 0%，SHD 大幅下降，且在中大型网络上表现稳定、可比甚至优于传统方法；运行时间随节点数线性增长，可在单台 CPU 上完成 724 变量的学习。

**⚠️ 局限性**

局限性包括：① 对初始 PAG 质量高度依赖，若骨架估计错误会影响最终 DAG；② 在极大规模或连续变量设置下的扩展性尚未验证；③ 先验（尤其是 LLM）需要额外计算与语言模型支持；④ 仍需进一步评估在真实因果实验环境下的稳健性。

---

## 502. Learning to Read Where to Look: Disease-Aware Vision-Language Pretraining for 3D CT

**arXiv ID:** 2603.02026 | [PDF](https://arxiv.org/pdf/2603.02026v1)

**作者:** Simon Ging `[一作]` (University of Freiburg), Thomas Brox `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

训练一个3D CT视听语言模型RadFinder，联合全卷积的对比预训练、基于疾病标签的提示式监督以及从报告中自动挖掘的片段-切片对进行的局部定位学习。

**💡 创新点**

创新点包括：①利用大量内部医院报告-体积对（约98k对）实现更大规模预训练；②在对比损失中加入疾病提示文本，以在共享嵌入空间直接完成疾病二分类；③通过自动提取报告中的“series X, image Y”引用，构造无标注的片段-切片对，实现三维切片级定位任务；④将检索、分类与定位统一到同一模型，避免任务间互相干扰。

**🔧 技术方法**

核心技术：SigLIP式对比学习、ViT-Large局部窗口编码+全局特征聚合、Qwen3文本编码器、prompt-based BCE损失、基于高斯软目标的切片定位交叉熵。训练时使用AdamW、学习率调度与数据增强。

**📊 数据集**

数据集：内部医院RefCT（78k报告-体积对，50k病人），公开数据CT-RATE（47k）、Merlin（15k）、INSPECT（19k）共计约159k对。片段-切片对约262k对，用于定位训练。

**📈 对比分析**

与公开基线比较：在CT-RATE上文本检索R@10从22.2提升至31.5（state‑of‑the‑art），疾病分类AUC在CT-RATE为83.8、Rad‑ChestCT为77.0，与MPS‑CT相当；在定位任务上MAE从67mm下降至36.3mm，精度显著提升。全模型在检索、分类与定位三项任务上均保持或提升性能，没有明显损失。

**⚠️ 局限性**

局限性：定位精度受12mm切片间隔限制，无法精确定位微小病变；片段-切片对的挖掘依赖报告中明确的切片引用，适用范围受制于报告习惯；模型仅在英文报告上训练，未直接评估双语性能。

---

## 503. A Classifying Topos for the Spectrum of Equivalences

**arXiv ID:** 2603.01056 | [PDF](https://arxiv.org/pdf/2603.01056v1)

**作者:** Kenan Oggad `[一作]` `[通讯]` (Universite Paris Saclay), Kenan Oggad (Universite Paris Saclay)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文构建了一个将过程代数中的 van Glabbeek 线性时间–分支时间谱嵌入分类上同构框架的理论体系，并通过几何逻辑与 Grothendieck 拓扑给出了模拟、分支和到态势等行为等价的严格层次结构。

**💡 创新点**

创新点在于：①将分类上同构与 van Glabbeek 谱统一为同一代数结构；②证明几何 van Benthem 定理，确立分支与到态势的几何不变子句；③利用能量拓扑与 Caramello 的商理论对偶得到完整的 30 元格谱格（L₃₀）及其 bi‑Heyting 结构。

**🔧 技术方法**

采用的技术包括：几何逻辑、分类上同构、Grothendieck 拓扑、能量游戏框架、Caramello 的双重性、树展开与 bounded tree unraveling 以及 Lean 4 + Mathlib 形式化。

**📊 数据集**

本研究主要是理论证明，不依赖外部数据集；示例以有限标记化转移系统（如 fork、path、hub‑spokes 等）为基础进行分离与层次验证。

**📈 对比分析**

通过形式化证明与构造性递归展示了所有层次的严格性；相较传统过程代数方法，提供了更抽象、统一且可形式化验证的证明；性能方面主要体现在理论可验证性上。

**⚠️ 局限性**

局限性：①部分证明仍依赖非构造性工具（如 König 引理）；②未覆盖死锁敏感等变体；③对无限系统的处理依赖几何但不保证可计算性；④部分结果在理论层面可推导，但在实际计算上的可行性尚未充分评估。

---

## 504. Phishing the Phishers with SpecularNet: Hierarchical Graph Autoencoding for Reference-Free Web Phishing Detection

**arXiv ID:** 2603.01874 | [PDF](https://arxiv.org/pdf/2603.01874v1)

**作者:** Tailai Song `[一作]` (Politecnico di Torino), Michela Meo `[通讯]` (Politecnico di Torino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种轻量级、无参考的网页钓鱼检测框架 SpecularNet，利用域名与 HTML DOM 树结构进行端到端的判别，提供快速、可部署的检测方案。

**💡 创新点**

创新点包括：1）将 DOM 视为树并设计层次化图自动编码器，采用方向性、层级级联的消息传递；2）融合重构误差与监督分类损失的双重目标，提升对恶意与正常网页的区分；3）使用域名字符 LSTM 与 Word2Vec 标签/属性嵌入，避免文本噪声，强调结构 invariant；4) 通过 top‑k 池化实现稀疏化并保持根节点，形成结构瓶颈。

**🔧 技术方法**

核心技术：图卷积网络 (GCN)、层级化图自动编码器、方向性子父消息传递、双分支判别（重构误差+MLP）、域名 LSTM、Word2Vec、top‑k 池化、LeakyReLU 激活、Adam 优化器。

**📊 数据集**

训练集：phishpedia（2021年收集的约6万网页）；测试集1：knowphish（2023年收集的1万网页）；测试集2：phishllm（2023‑24年收集的1.2万网页）。

**📈 对比分析**

与13种顶尖检测器（4个传统 DL、3个通用 GNN、6个基于参考的）做对比。SpecularNet 在 knowphish 上 F1 93.92%，仅落后最高 1.5pp；在 phishllm 上 F1 86.05%，排名前三；推理时间仅 20 ms/页，远低于 2–7 s 的参考方法；在实测域名缺失和 6k 开放世界数据集上仍保持 81%+ 的 F1；对 SpacePhish 的 HTML 级攻击保持 73%‑94% 的准确率，显著优于传统方法。

**⚠️ 局限性**

局限性：1）相较最强参考方法仍略有性能差距；2）依赖 DOM 结构，难以处理极端动态或高度混淆的页面；3）不利用文本、图片或网络上下文信息，可能被针对性特征的攻击绕过；4）在极大页面（>5k 节点）时推理时间升高；5）模型对域名分布变化仍有一定敏感性，需要定期再训练。

---

## 505. Mitigating topology biases in Graph Diffusion via Counterfactual Intervention

**arXiv ID:** 2603.02005 | [PDF](https://arxiv.org/pdf/2603.02005v1)

**作者:** Wendi Wang `[一作]` (Purdue University), Lu Lin `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于反事实干预的图扩散模型（FairGDiff），能够在图生成过程中一次性去除敏感属性导致的拓扑偏差，同时保持生成图的结构完整性。

**💡 创新点**

创新点在于：①构建因果模型估计敏感属性对边形成的偏差；②通过反事实处理生成无偏的“治疗”矩阵，直接融入前向扩散和后向去噪；③实现单步、无预处理的公平图生成，兼顾公平性与实用性。

**🔧 技术方法**

技术包括：图扩散模型（Latent Diffusion）、因果推断（反事实推理）、双路径训练（事实+反事实）、基于邻近无偏对的无偏治疗估计、正则化损失组合。

**📊 数据集**

实验使用四个真实图数据集：NBA（敏感属性为种族）、German Credit（性别）、Pokec‑n 和 Pokec‑z（性别）。

**📈 对比分析**

与四类基线（FairAdj、FairDrop、FairWire、FairGen）在节点分类、链接预测、图对比学习等下游任务上比较。FairGDiff 在公平指标（Δ_DP、Δ_EO）与效用指标（Accuracy、NDCG@10）上取得最优或次优组合，且 FLOPs 远低于两步和对抗方法，证明其高效与可扩展性。

**⚠️ 局限性**

局限性：仍依赖敏感属性标签；对极大规模图的扩散步骤时间可能受限；在某些任务中公平性提升幅度相对有限，需进一步探讨对不同敏感属性类别的适用性。

---

## 506. WristPP: A Wrist-Worn System for Hand Pose And Pressure Estimation

**arXiv ID:** 2603.00606 | [PDF](https://arxiv.org/pdf/2603.00606v1)

**作者:** Ziheng Xi `[一作]` (Tsinghua University), Jie Zhou `[通讯]` (Tsinghua University)

**通讯引用:** 34700 | [OpenAlex ID](https://openalex.org/A5100620306)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于腕部可穿戴摄像头的系统，可在单帧RGB图像中实时重建3D手部网格并估计每个顶点的接触压力；

**💡 创新点**

创新点包括：①利用腕部可穿戴的全景鱼眼摄像头获得近距离、无遮挡的手部视角；②结合ViT与离散化VQ‑VAE的混合网络，先将手部姿态离散化为代码索引，再通过解码得到完整手网格；③在同一网络中加入相机外参预测与跨注意力机制，使姿态与压力估计对视角高度鲁棒；

**🔧 技术方法**

核心技术包括Vision Transformer（ViT）骨干网络、Vector‑Quantized VAE（VQ‑VAE）代码库、交叉注意力（cross‑attention）与外参嵌入、以及多任务损失（姿态分类、压力回归、外参重投影等）；

**📊 数据集**

使用了自采集的133,000帧数据集，涵盖20名受试者、48个平面交互动作和28个空中手势，标注包括高精度手部Mesh（通过多模态优化得到）和每顶点压力；

**📈 对比分析**

与多种基准（MediaPipe、WiLoR、传统手势识别等）进行比较，系统在MPJPE 2.9 mm、MJAE 3.2°、接触IoU 0.712、压力MAE 10.4 g等指标上表现出色；在三项用户研究中实现了与触控板相当的指针效率、单/多指压力控制成功率高达98%及Whac‑A‑Mole大屏游戏中最高90%命中率、最低2%错误率，且用户感知疲劳显著降低；

**⚠️ 局限性**

局限包括：仅处理平面/准平面表面交互，无法直接估计手与任意形状物体或手指间的压力；对极端腕部旋转或大物体导致的局部遮挡仍易出现误差；未覆盖双手交互；设备体积与功耗仍高于商业可穿戴；模型在低功耗平台的实时性尚待优化。

---

## 507. \textsc{Mobile-VTON}: High-Fidelity On-Device Virtual Try-On

**arXiv ID:** 2603.00947 | [PDF](https://arxiv.org/pdf/2603.00947v1)

**作者:** Zhenchen Wan `[一作]` (University of Sydney), Mingming Gong `[通讯]` (University of Melbourne)

**通讯引用:** 8052 | [OpenAlex ID](https://openalex.org/A5102023771)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Mobile-VTON，一种能够在普通移动设备上完成高质量虚拟试衣的完整离线框架；

**💡 创新点**

创新点包括：①教师网络–服装网络–试衣网络（TGT）三模组结构，②基于特征引导对抗蒸馏（FGA）融合知识蒸馏与对抗学习；③服装网络的轨迹一致性训练保证跨步语义稳定；④试衣网络通过潜在拼接与多源服装语义融合实现无遮罩精确对齐；

**🔧 技术方法**

技术手段包括：基于Stable Diffusion的教师网络、轻量化SnapGen U-Net学生网络、DINOv2视觉编码的Light‑Adapter、特征级蒸馏+对抗损失、轨迹一致性损失、潜在拼接（Latent Concatenation）以及多模态跨模注意力；

**📊 数据集**

使用VITON‑HD、DressCode以及VITON‑HD In‑the‑Wild三大公开试衣数据集，统一在1024×768分辨率上进行训练与评估；

**📈 对比分析**

与多种服务器端基线（如SD‑VITON、LaDI‑VTON、StableVITON、IDM‑VTON、CatVTON、BooW‑VTON）以及本机端无遮罩方法对比，Mobile‑VTON在LPIPS、SSIM、CLIP‑I等结构与感知指标上与服务器端方法持平甚至优越，在FID/KID等真实性指标上亦具竞争力，同时仅占用约2.8 GB显存、参数约4 B，能够在普通手机NPU/GPU上实时推理；

**⚠️ 局限性**

局限性主要体现在：①仍依赖大规模教师模型进行蒸馏，导致训练成本高；②在极端姿态或极稀疏纹理服装上对细节再现仍有欠缺；③缺少多体型和全身试衣的支持；④在极低算力设备上性能与延迟仍需进一步优化。

---

## 508. S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation

**arXiv ID:** 2603.01554 | [PDF](https://arxiv.org/pdf/2603.01554v1)

**作者:** Akila Siriweera `[一作]` (University of Aizu), Isuru Jayanada `[通讯]` (KD University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于 Agentic RAG 的社会5.0驱动智能家居环境模拟框架 S5‑HES Agent，支持自然语言驱动配置、可扩展的多住户行为建模、威胁注入与自动化真值标注，并通过多代理协同实现端到端模拟。

**💡 创新点**

核心创新点包括：① 结合 RAG 与多代理协同实现无编程智能配置；② 引入可配置的知识库（文档、设备规格、CVE/ATT&CK）实现语义与关键词混合检索；③ 通过可复现的种子机制和完整配置导出保证跨机构可复现；④ 自动化标签生成覆盖多级 MITRE ATT&CK 结构，降低人工标注成本；⑤ 支持多住宅规模与多种实验模式（手工、LLM 辅助、无 LLM）。

**🔧 技术方法**

技术栈包括：LLM（OpenAI GPT‑4o、Gemini 2.0‑Flash、Llama‑3.2‑3B），RAG（ChromaDB + GTE‑Large embeddings + BM25，Reciprocal Rank Fusion），多代理架构（HomeBuilder、DeviceManager、ThreatInjector、Optimization），验证管道（schema、语义、事实、业务规则校验），Markov‑chain 行为模型，威胁生命周期引擎，TLS/JWT/AES‑256‑GCM 等安全组件。

**📊 数据集**

用于评估的数据集：知识库 20,306 份文档（学术、威胁、设备规格）；真实 IoT 流量基线 Edge‑IIoTset、IoT‑23、Bot‑IoT；设备行为基线 SDHAR‑HOME、Logging；标注对比基线 N‑BaIoT、IoT‑23、TON‑IoT。S5‑HES 在单次仿真中生成多种家居模板（Studio、Family House、Mansion）。

**📈 对比分析**

对比方法：① 文档检索评估（P@k、R@k、nDCG、MRR、MAP），S5‑HES Semantic 达到 P@1=1、MAP=0.967、MRR=1，Hybrid 次之；② 生成质量评估（faithfulness、fluency、ROUGE‑L、BERTScore），Gemini 2.0‑Flash 取得最高综合质量；③ 威胁情景真实度评估（ABC、ALF），8/9 目标通过，说明攻击行为与生命周期准确；④ 设备行为真实性（M.SIM）约 88‑90% ；⑤ 数据集质量（规模、特征、平衡、攻击多样性、时间均匀性、源多样性、标签深度）中 S5‑HES 在攻击多样性、时间均匀性和标签深度上领先，但规模相对较小；⑥ 可扩展性测试显示设备数量、事件量和设备多样性随家居模板复杂度线性增长。

**⚠️ 局限性**

局限性包括：仅在单一随机种子下评估，未展示多种初始化下的稳健性；设备行为评估受限于 SDHAR‑HOME/Logging 覆盖范围；部分攻击（如 Credential Theft）在 MITRE 指标匹配不足；ROUGE‑L 受 LLM 重新表述影响，不能完整反映事实一致性；知识库检索仍可能产生关键词噪声；整体数据集规模受单次仿真限制，需进一步扩大验证。

---

## 509. DynaMoE: Dynamic Token-Level Expert Activation with Layer-Wise Adaptive Capacity for Mixture-of-Experts Neural Networks

**arXiv ID:** 2603.01697 | [PDF](https://arxiv.org/pdf/2603.01697v1)

**作者:** Gökdeniz Gülmez `[一作]` `[通讯]` (Machine Learning Research), Gökdeniz Gülmez (Machine Learning Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 DynaMoE，一种动态专家激活与层级专家分配的 Mixture‑of‑Experts（MoE）架构；

**💡 创新点**

创新点在于：①通过基于门控分位数阈值实现输入自适应的动态 Top‑K 路由；②设计六种层级专家调度策略（递减、递增、金字塔、波浪等），以匹配任务的表示多样性；③给出动态路由的表达性、梯度方差与计算复杂度的理论分析；

**🔧 技术方法**

使用技术包括：Mixture‑of‑Experts 结构、分位数阈值动态路由、温度缩放、Gaussian 噪声探索、六种专家调度函数；训练采用 AdamW、余弦学习率退火、门控网络等；

**📊 数据集**

实验数据集包括图像分类：MNIST、Fashion‑MNIST、CIFAR‑10；语言建模：Recycling‑the‑Web（1,000 句子）；

**📈 对比分析**

比较方法：与全连接 MLP 基线和统一专家分配的 MoE 进行对比；在图像任务上，递减调度在所有模型规模下均优于均匀调度，最高提升达 5.47%（CIFAR‑10）；在语言任务上，递减、递增和均匀调度在不同模型规模下分别最优，且在中等规模下均可比 MLP 提升 3.4% PPL；

**⚠️ 局限性**

主要限制：语言实验样本极少，缺乏大型预训练数据；未加入传统 MoE 的容量约束和负载平衡正则；不同调度间参数量不一致，未给出 FLOPs/延迟的统一衡量；仅在 MLP 结构下验证，未评估 Transformer 等更深层架构；缺乏基准 MoE 对比（Switch Transformer 等）。

---

## 510. D-GVIO: A Buffer-Driven and Efficient Decentralized GNSS-Visual-Inertial State Estimator for Multi-Agent Systems

**arXiv ID:** 2603.01404 | [PDF](https://arxiv.org/pdf/2603.01404v1)

**作者:** Yarong Luo `[一作]` (Wuhan University), Chi Guo `[通讯]` (Wuhan University)

**通讯引用:** 2297 | [OpenAlex ID](https://openalex.org/A5072031132)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了D-GVIO，一个基于缓冲驱动的完全去中心化GNSS-视觉-惯性里程计框架，用于多智能体协同定位。

**💡 创新点**

创新点包括：利用协方差分割与缓冲策略实现模块化状态估计，采用左不变扩展卡尔曼滤波器（L-IEKF）实现不依赖状态估计的传播，基于缓冲的重传播策略高效处理延迟测量，以及自适应基于熵的GNSS异常剔除方法。

**🔧 技术方法**

技术手段包括：左不变EKF、协方差分割与缓存、协方差交叉传播、协方差交叉传播重建、协同更新采用协方差交叉（CI）算法、VLAD特征描述、缓冲重传播、熵自适应阈值。

**📊 数据集**

实验使用了公开的Castle Around仿真数据集、S3E Square实际数据集以及自采集的武汉大学高楼密集区双车数据集。

**📈 对比分析**

通过与X-VIO、COVINS、IC-GVINS、InGVIO等SOTA方法以及集中式EKF在ATE、CPU、内存、消息量等指标对比，D-GVIO在保持相近甚至更优的ATE的同时，CPU使用率低30–70%，内存使用量减少70%以上，消息量显著下降。

**⚠️ 局限性**

限制主要在于对多体协同网络规模的进一步验证、在极端GNSS遮挡下的鲁棒性仍需提升，以及缓冲大小和延迟处理在极高频传感器下的适配。

---

## 511. Trident: Adaptive Scheduling for Heterogeneous Multimodal Data Pipelines

**arXiv ID:** 2603.02075 | [PDF](https://arxiv.org/pdf/2603.02075v1)

**作者:** Ding Pan `[一作]` (Hong Kong University of Science and Technology), Binhang Yuan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套完整的三层适配调度框架，专为固定资源集群下的多模态数据准备流水线设计，能实时估算异步算子容量、根据工作负载动态调优算子配置，并通过混合整数线性规划协同优化算子并行度、部署位置与配置迁移；

**💡 创新点**

创新点在于①将噪声抑制与高斯过程回归结合，提供稳健的异步算子吞吐量预测；②在线工作负载聚类与内存约束贝叶斯优化相结合，安全探索配置空间并防止OOM；③将容量估计、配置建议与跨节点资源限制统一到MILP中，支持滚动更新与网络共定位；

**🔧 技术方法**

核心技术包括高斯过程回归+两阶段异常过滤、在线聚类+贝叶斯优化、混合整数线性规划（MILP）、Ray Data平台扩展、滚动更新策略；

**📊 数据集**

使用200k份PDF文档（学术、报告、财报）和410k段视频（短视频/长视频）等多模态数据集进行实验；

**📈 对比分析**

与静态、Ray Data默认自动缩放、DS2、ContTune、SCOOT等基线对比，Trident在PDF流水线上提升至2.01×、视频流水线提升至1.88×，在所有基线中表现最优；

**⚠️ 局限性**

局限包括对固定资源集群假设强、对GP模型与贝叶斯优化的训练成本、对大规模集群的求解时间增长、需要手工设计特征与阈值，且对极端高峰流量的自适应仍有挑战。

---

## 512. Visual Bias in Simulated Users: The Impact of Luminance and Contrast on Reinforcement Learning-based Interaction

**arXiv ID:** 2603.01901 | [PDF](https://arxiv.org/pdf/2603.01901v1)

**作者:** Hannah Selder `[一作]` (Leipzig University), Arthur Fleig `[通讯]` (Leipzig University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了亮度和对比度对强化学习驱动的模拟用户在指点与跟踪任务中的表现与鲁棒性影响，训练了247个不同渲染配置的模拟用户。

**💡 创新点**

创新点在于首次从视觉渲染角度量化亮度/对比度对RL模拟用户行为的影响，并公开了对应的训练模型数据集。

**🔧 技术方法**

采用了基于Proximal Policy Optimization的强化学习框架，结合User‑in‑the‑Box的生物力学模拟与MuJoCo环境进行训练与评估。

**📊 数据集**

使用了247个在不同背景、目标与干扰器亮度组合下训练的RL模型，公开于Zenodo的实验数据集。

**📈 对比分析**

通过比较指点成功率与跟踪误差，发现无干扰条件下亮度相近可获得70–100%成功率，静态干扰需高对比度才能维持性能，移动干扰时亮度影响减弱，整体鲁棒性受相对亮度顺序限制。

**⚠️ 局限性**

局限性包括仅使用灰度图像、固定干扰器形状与运动模式，未探讨色彩或更复杂视觉环境对RL模型的影响。

---

## 513. Jailbreaking Embodied LLMs via Action-level Manipulation

**arXiv ID:** 2603.01414 | [PDF](https://arxiv.org/pdf/2603.01414v1)

**作者:** Xinyu Huang `[一作]` (Hong Kong Polytechnic University), Yuanqing Zheng `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 4030 | [OpenAlex ID](https://openalex.org/A5091556511)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自动化攻击框架Blindfold，利用代理规划、意图混淆和规则验证器在语义层面外对具身大型语言模型进行动作级攻击；

**💡 创新点**

创新点在于：①通过本地受控LLM进行离线代理规划，避免对目标系统的多次交互；②对关键致伤动作进行位置识别并加入上下文无关的掩蔽动作，从而规避语义级安全检测；③结合有限状态机验证器迭代校验执行可行性，显著提升攻击可执行率；

**🔧 技术方法**

技术手段包括：代理规划（使用被攻击LLM的复制品生成行动序列）、意图识别与掩蔽（基于损害度评分与动作注入）、规则验证器（基于FSM的前置条件/效果验证）以及与目标系统的离线推理与迭代优化；

**📊 数据集**

数据集主要为 SafeAgentBench 与 BadRobot 的混合，过滤后得到187条具身指令，覆盖物理伤害、环境破坏、隐私侵犯等四大类；

**📈 对比分析**

与两种SOTA基线（POEX与BadRobot）对比，在多模仿器（VirtualHome、Habitat、ManiSkill、RoboTHOR）和闭源LLM（GPT‑4o、Claude‑3.5‑Sonnet）上，Blindfold 的攻击成功率（ASR）普遍超过80%，在最佳场景可达100%；任务完成率（TSR）提升15–30个百分点，最高可达74%；

**⚠️ 局限性**

局限性：①需要预先观察并假设目标环境在攻击期间相对稳定；②对代理LLM的能力与温度敏感，模型规模越大成功率越高；③攻击仍依赖于离线规划，缺乏实时反馈；④在高度受限的动作空间或复杂物理约束（如ManiSkill）时TSR下降；⑤未对多模态感知误差或硬件安全机制的抗性做深入评估。

---

## 514. PanCanBench: A Comprehensive Benchmark for Evaluating Large Language Models in Pancreatic Oncology

**arXiv ID:** 2603.01343 | [PDF](https://arxiv.org/pdf/2603.01343v1)

**作者:** Yimin Zhao `[一作]` (University of Washington), Jeffrey T. Leek `[通讯]` (Fred Hutchinson Cancer Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建PanCanBench benchmark，对来自Pancreatic Cancer Action Network的282条真实患者问题进行专家设计的评估量表，利用人机循环流程、LLM-judge与事实错误检测，评估22种LLM的临床完整性、事实准确性及网页检索效果。

**💡 创新点**

①使用真实患者问答和专科专家量表提升评估深度；②引入LLM-judge与双模型事实检测提升客观性；③对比人工量表与AI生成量表，探讨可扩展性；④综合完整性、事实性与检索指标，形成多维度评价框架。

**🔧 技术方法**

人机循环量表生成、GPT‑5 作为评审器、Gemini‑2.5 Pro/Claude 等多模型事实判定、基于atomic claim的幻觉检测、网页检索触发率与支持率指标、AI生成量表对比实验。

**📊 数据集**

PanCanBench 数据集：282条去标识化患者问题，3,130条问题专属量表项，涵盖治疗、诊断、支持、遗传、症状等五大类别。

**📈 对比分析**

在22种模型中，o3最高完整度82.3%，但幻觉率高；GPT‑5、Gemini‑2.5 Flash综合表现最佳；Web检索未显著提升整体得分，部分模型检索后信息缺失。AI生成量表相对人类量表提升平均17.9分，但模型排名保持稳定。

**⚠️ 局限性**

样本量有限，仅聚焦胰腺癌单一疾病；LLM-judge与事实检测存在潜在偏差；对不同医学领域的泛化能力待验证；对开源模型事实可靠性仍需提升。

---

## 515. Multi-Head Low-Rank Attention

**arXiv ID:** 2603.02188 | [PDF](https://arxiv.org/pdf/2603.02188v1)

**作者:** Songtao Liu `[一作]` (Pennsylvania State University), Yue Guo `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的注意力机制 Multi‑Head Low‑Rank Attention（MLRA），在保持低 KV 缓存尺寸的同时实现 4‑way 张量并行（TP）解码。

**💡 创新点**

创新点在于把单一 latent head 分解为多个独立的低秩分支（4‑branch 或 2‑branch），每个分支分别上投影到 KV，最后对各分支的注意力输出求和，从而既降低了 KV 缓存量，又天然支持 TP。

**🔧 技术方法**

技术包括：FlashAttention‑3 与 FlashMLA 核心实现、RoPE 与 RMSNorm 的变种、权重归一化与分支级别的方差校准、Zero 初始化与 scaling、Triton 自定义核与 torch.compile 优化。

**📊 数据集**

数据集：FineWeb‑Edu‑100B（98.3B 预训练 token）以及 FineWeb‑Edu、Wikipedia、C4、Pile、RefinedWeb、Cosmopedia、FineWeb 的验证集；下游评测使用 ARC‑E/C、OpenBookQA、BoolQ、HellaSwag、Winogrande、PIQA 等常识推理基准。

**📈 对比分析**

对比方法包括 MHA、MQA、GQA、MLA、MFA、TPA、GLA‑2、GLA‑4、GTA 等多种注意力变体。实验显示 MLRA‑4 在 2.9B 参数规模下获得最佳困惑度 13.672（比 MLA 低 0.1），在所有常识基准上平均准确率最高；解码速度比 MLA 高约 2.8×，比 GQA 1.05–1.26×；吞吐量在 1K–16K token 范围内最高。

**⚠️ 局限性**

局限性：MLRA 仍需 4‑way TP 进行性能优势，单机多卡扩展受限；对 KV 缓存的拆分和多分支求和在极大模型或超长上下文时可能产生额外的同步与内存开销；实验主要在 H100 GPU 上验证，其他硬件或更大规模模型的适配性尚未探究。

---

## 516. Machine Learning (ML) library in Linux kernel

**arXiv ID:** 2603.02145 | [PDF](https://arxiv.org/pdf/2603.02145v1)

**作者:** Viacheslav Dubeyko `[一作]` `[通讯]` (IBM), Viacheslav Dubeyko (IBM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个Linux kernel的ML库架构与PoC项目，实现内核与用户空间ML模型之间的交互接口。

**💡 创新点**

引入内核ML代理模式，支持多种交互模式（emergency、learning、collaboration、recommendation）并实现持续学习与误差反向传播机制。

**🔧 技术方法**

结合eBPF、sysfs、FUSE或字符设备实现数据收集与模型推理，使用Python用户空间训练，支持浮点运算。

**📊 数据集**

采集自内核子系统的运行时数据，示例中使用了Linux调度与网络子系统的监控数据，但未给出公开数据集。

**📈 对比分析**

通过PoC验证推理延迟和资源消耗在可接受范围内，未给出基准对比，仅指出理论上可降低配置错误与提升自适应性。

**⚠️ 局限性**

训练阶段对内核性能影响大，浮点运算受限，模型迁移与准确性受限，实际系统部署需进一步评估与优化。

---

## 517. Multimodal Adversarial Quality Policy for Safe Grasping

**arXiv ID:** 2603.01479 | [PDF](https://arxiv.org/pdf/2603.01479v1)

**作者:** Kunlin Xie Chenghao Li Haolan Zhang `[一作]`, Nak Young Chong `[通讯]` (Japan Advanced Institute of Science and Technology)

**通讯引用:** 2702 | [OpenAlex ID](https://openalex.org/A5000452220)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了基于 RGBD 视觉把握系统的安全性，提出 Multimodal Adversarial Quality Policy (MAQP)，通过生成对抗性质量补丁使机器人在抓取时避免碰到人手或附近物体。

**💡 创新点**

创新点主要有两方面：①使用模态特定初始化（HDPOS）解决 RGB 与深度在补丁生成时的分布差异；②在形状适配阶段引入梯度层面模态平衡（GLMBS），通过梯度重加权和距离自适应扰动限值平衡两模态的优化不平衡。

**🔧 技术方法**

采用深度神经网络视觉抓取模型、对抗性补丁生成与投影质量梯度下降（PQGD）、梯度重加权、距离自适应扰动等技术；同时将 RGB 与深度信息融合为四通道输入。

**📊 数据集**

在 Cornell Grasp 数据集和 OCID Grasp 数据集上进行实验，并在多种 DNN 抓取网络（GG-CNN、GG-CNN2、GR-ConvNet、FCG-Net、SE-ResUNet 等）上进行验证。

**📈 对比分析**

与传统 QFAAP 等方法对比，MAQP 在多模型、多数据集上实现 Q-ACC 超过 85%（部分模型在 Cornell 上为 70%），运行时 0.004–0.057 秒，且在真实机器人抓取中 DRD‑Rate 达到 84%–92%。

**⚠️ 局限性**

局限性包括对 RGB 与深度归一化的匹配不足导致某些模型（如 GG‑CNN2、FCG‑Net）表现不佳；方法对实时手部分割依赖较大，极端光照或遮挡下可能影响效果。

---

## 518. SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation

**arXiv ID:** 2603.01101 | [PDF](https://arxiv.org/pdf/2603.01101v1)

**作者:** Hongrui Wang `[一作]` (Hong Kong University of Science and Technology), Yang Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18721 | [OpenAlex ID](https://openalex.org/A5100322935)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了SyncTrack，一种同步多轨音乐生成模型，能够在保持节奏稳定性的同时捕获每个轨道独特的音色特征。

**💡 创新点**

创新点包括：① 轨道共享模块与轨道特定模块的组合，实现共通节奏与个体音色的分离处理；② 两种跨轨注意机制（全局与时间特定）以提升跨轨节奏同步；③ 引入三种新的节奏一致性评估指标 IRS、CBS、CBD，为多轨音乐质量提供更细粒度的量化。

**🔧 技术方法**

技术手段包括：基于潜在扩散模型（Latent Diffusion Model）框架，VAE 进行波形到潜在空间的编码，HiFi-GAN 进行解码；使用双跨轨注意力（全局+时间特定）以及可学习的乐器先验；训练采用 DDIM 采样；评估采用 FAD 与新指标 IRS、CBS、CBD。

**📊 数据集**

使用 Slakh2100 数据集（四轨：bass、drums、guitar、piano），对训练、验证和测试进行统一处理。

**📈 对比分析**

通过与 MSDM、STEMGEN、JEN-1 Composer、MSG-LD 等基线在 FAD、IRS、CBS、CBD 等指标进行对比，SyncTrack 在 FAD 上从 6.55 降至 1.26（相较 MSG-LD 为 1.31），在 IRS、CBS、CBD 上均优于基线，主观评测得分也明显高于 MSG-LD 与 MSDM。

**⚠️ 局限性**

局限性包括：生成片段长度受限（仅 10.24 秒）；跨轨同步在极端音色组合或更复杂节奏结构时仍可能出现误差；对更长曲目或更大规模多轨场景的适用性尚未验证。

---

## 519. Hippo: High-performance Interior-Point and Projection-based Solver for Generic Constrained Trajectory Optimization

**arXiv ID:** 2603.00871 | [PDF](https://arxiv.org/pdf/2603.00871v1)

**作者:** Haizhou Zhao `[一作]` (New York University), Majid Khadiv `[通讯]` (Technical University of Munich)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5043216529)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Hippo轨迹优化器，基于内部点方法与投影式SQP，能够高效处理机器人运动规划中的不等式约束和硬等式约束。

**💡 创新点**

创新点包括：① 将投影式 Riccati 递归与正则化 IPM 结合，避免传统投影法对等式约束的退化；② 引入 MP‑C 自适应障碍参数更新和安全阈值，减少手工调参；③ 采用并行化前向递归和固定步长回溯线搜索，提升数值稳定性。

**🔧 技术方法**

使用技术包括：Interior Point Method（IPM）、投影式 Riccati 递归、正则化原双梯度 IPM、MP‑C 障碍参数自适应、固定步长回溯线搜索、Eigen+BLASFEO 的高效线性代数、CasADi+Pinocchio 的代码生成与微分。

**📊 数据集**

实验数据集：UR5 随机 SE3 到达任务（100 条随机目标）和 Go2 四足机器人两步行走（trot 与 hopping，100 条随机目标）。

**📈 对比分析**

与 fatrop、acados、aligator、mim_solver 等现有求解器在上述任务上进行对比；Hippo 在成功率、QP 迭代次数和平均 QP 时间上表现最好或相近，尤其在复杂硬约束情形下保持高成功率，QP 迭代次数显著低于传统方法，整体求解速度与 fatrop 相当，明显优于 acados 与 ALM 基础求解器。

**⚠️ 局限性**

局限性：对正则化参数和固定步长回溯策略较为敏感；在某些全约束退化或高度稀疏的全约束设置下，Hippo(i) 可能因缺乏更完善的全局化（如 IPOPT 的 feasibility restoration 或 filter line search）而失效；投影式递归在与专用 BLASFEO QP 求解器相比速度略慢。

---

## 520. Implementing Dependent Type Theory Inhabitation and Unification

**arXiv ID:** 2603.01463 | [PDF](https://arxiv.org/pdf/2603.01463v1)

**作者:** Chase Norman `[一作]` (Carnegie Mellon University), Jeremy Avigad `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3124 | [OpenAlex ID](https://openalex.org/A5003483051)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个简化版的 Canonical（Canonical-min），在 185 行 Lean 代码中实现了依赖类型理论（DTT）的完整类型检查、体 inhabited 与统一求解器，并提供了 DTTBench 基准测试。

**💡 创新点**

创新点在于：1）采用可变元变量与显式替换表，将类型检查器无缝转化为约束求解器；2）使用单构造器表示法和 de Bruijn 索引提升数据结构效率；3）结合 monadic 框架（ReaderT、StateT、ContT）实现回溯式约束处理；4）采用熵驱动的迭代加深 DFS 搜索策略；5）公开完整实现作为参考。

**🔧 技术方法**

技术包括 Lean 语言实现、依赖类型理论、monad（ReaderT、StateT、ContT）组合、回溯式约束求解、迭代加深搜索、显式替换（ES）结构、de Bruijn 索引、MetaVariable（类型和表达式）管理。

**📊 数据集**

数据集为 DTTBench，包含 31 个从 Lean 标准库与 Mathlib 适配的 DTT 习题，涵盖等式、顺序、逻辑、关系、集合等常见数学概念。

**📈 对比分析**

通过 60 秒超时对比 Canonical-min 与 Twelf、sauto、mimer，Canonical‑min 在 31/31 题目全部成功，而其他系统仅完成 8/31、6/31、2/31；表明 Canonical‑min 完整且在性能上明显优于现有系统。

**⚠️ 局限性**

局限性：实现相对简化，缺乏 Canonical 的高级优化与扩展；未在更广泛的语言或自定义归约规则下验证；完整性证明仍未形式化；对更大规模问题的性能与可扩展性待进一步研究。

---

## 521. Spherical Latent Motion Prior for Physics-Based Simulated Humanoid Control

**arXiv ID:** 2603.01294 | [PDF](https://arxiv.org/pdf/2603.01294v1)

**作者:** Jing Tan `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2061 | [OpenAlex ID](https://openalex.org/A5109900808)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种两阶段的球面潜在运动先验(SLMP)，先用强化学习训练高质量运动追踪控制器，再将其蒸馏到单位球面潜空间，并配合鉴别器和鉴别器引导的局部语义一致性损失构造结构化的潜空间；同时收集并公开了一套约两小时的人类搏击动作捕捉数据集。

**💡 创新点**

创新点在于：①将运动追踪控制器蒸馏到球面潜空间，避免VAE的重构瓶颈；②结合鉴别器与局部语义一致性损失，使潜空间在保持多模态的同时具备稳定的采样性；③通过自我对弈在双代理搏击任务中验证了该先验可在稀疏奖励下生成多样化、物理上可行的战斗行为。

**🔧 技术方法**

使用技术包括：PPO强化学习训练目标条件运动追踪控制器；潜空间蒸馏结合鉴别器和鉴别器引导的局部语义一致性损失；自我对弈（self‑play）训练高层策略在双代理搏击任务中输出潜码；以及物理仿真引擎Isaac Gym进行评估。

**📊 数据集**

使用数据集：约两小时的搏击动作捕捉数据集（502段，约14秒/段），涵盖拳击、踢击、闪躲、脚步等多种技巧，数据采样频率30Hz。

**📈 对比分析**

与基准方法（VAE基准PULSE、AMP基准ASE）比较：SLMP在运动跟踪成功率和MPJPE上接近专家控制器，显著低于PULSE；在随机潜码采样的稳定性（存活率）上，SLMP优于PULSE，能够在更长时间内保持站立；在双代理搏击任务中，SLMP在仅使用稀疏规则奖励的情况下，能产生与人类相似且多样化的战斗行为，而基准方法往往陷入重复或不合理的动作。

**⚠️ 局限性**

局限性：对极端动态或复杂环境的鲁棒性尚未充分验证；潜空间对当前状态高度依赖，导致在某些姿态下可行动作空间收缩；仍需更大规模、多样化的数据集来进一步提升泛化能力；以及在真实机器人上的部署仍需更多实验验证。

---

## 522. ConVibNet: Needle Detection during Continuous Insertion via Frequency-Inspired Features

**arXiv ID:** 2603.01147 | [PDF](https://arxiv.org/pdf/2603.01147v1)

**作者:** Jiamei Guo `[一作]` (Technical University of Munich), Zhongliang Jiang `[通讯]` (The University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计并实现了实时连续针头检测框架ConVibNet，利用频域特征和时间相关性实现超声图像中针头位置与角度的连续估计。

**💡 创新点**

提出交叉差异损失（intersection-and-difference loss）显式利用连续帧的运动相关性，并将VibNet改造为适合连续插入、去掉高成本Hough Transform，保证实时性。

**🔧 技术方法**

频域特征提取（STFT）+卷积网络+时间编码器，交叉差异损失与焦点损失，RANSAC后处理，Adam优化及数据增强。

**📊 数据集**

自制含NDI跟踪的猪皮组织超声视频集，106条视频（约12k帧），插入角度仅为15°和30°。

**📈 对比分析**

与VibNet去掉DHT版本和UNet‑LSTM基线对比；ConVibNet tip误差2.80±2.42 mm、角误差1.69±2.00°、成功率79.6%，比基线提升tip误差0.75 mm、成功率约15.9%。

**⚠️ 局限性**

受限于样本量小、角度仅两种、未考虑针头弯曲与操作者差异，无法区分操作者动作与振动噪声，且对不同探头/系统的泛化未验证。

---

## 523. Demand- and Priority-Aware Adaptive Congestion Control for Heterogeneous V2X Service Requirements

**arXiv ID:** 2603.01134 | [PDF](https://arxiv.org/pdf/2603.01134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 524. ROSER: Few-Shot Robotic Sequence Retrieval for Scalable Robot Learning

**arXiv ID:** 2603.01474 | [PDF](https://arxiv.org/pdf/2603.01474v1)

**作者:** Zillur Rahman `[一作]` (University of Nevada Las Vegas), Cristian Meo `[通讯]` (TU Delft)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出ROSER框架，将长无标签机器人日志拆分为可检索的任务片段。

**💡 创新点**

将数据整理视作few-shot检索问题，构建任务无关的度量空间，且不需要任务特定训练。

**🔧 技术方法**

使用1D CNN原型网络进行元学习，滑动窗口检索配合NMS后处理，支持少量示例检索。

**📊 数据集**

在LIBERO、DROID和nuScenes三大大规模机器人/驾驶数据集上进行实验。

**📈 对比分析**

与STUMPY、Dtaidistance、Shapelets等经典时间序列匹配、LLM嵌入（Llama、Gemma、Qwen）以及时间序列基础模型MomentFM对比，ROSER在分布相似度、时间相关性、密度与多样性等指标均显著优于或排在第二，并且每次匹配耗时仅0.5毫秒。

**⚠️ 局限性**

对极短或多模态转移、传感器缺失或域漂移的情形易出现误检，且当前方法尚未充分利用视觉信息。

---

## 525. RAVEL: Reasoning Agents for Validating and Evaluating LLM Text Synthesis

**arXiv ID:** 2603.00686 | [PDF](https://arxiv.org/pdf/2603.00686v1)

**作者:** Andrew Zhuoer Feng `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 15804 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RAVEL 框架，通过让 LLM 自动规划和执行文本合成的四个关键操作（提纲、草稿、评审、修订）来动态评估 LLM 的合成能力；同时构建了 RAVEL-Score 基准，包含 Cloze、Edit、Expand、End‑to‑End 四个任务，数据来源于专业写作并采用反向工程生成指令；

**💡 创新点**

创新点在于：① 将评估视角从单次生成转向基于 LLM 自主决策的循环过程；② 通过 “逆向构造” 方式生成高质量参考文本与对应指令，提升任务真实度；③ 发现推理（规划/评审）对合成质量的决定性作用，生成能力并非主导因素；

**🔧 技术方法**

技术手段包括：① 基于 SDP（Sequential Decision Process）的推理-行动框架；② 使用 LLM 作为环境代理，执行 Outline、Draft、Review、Refine 四个原语；③ LLM‑as‑a‑judge 自动评判；④ 统计轨迹效率、修订密度等自定义指标；

**📊 数据集**

使用的主要数据集为 RAVEL-Score，包含 1,258 份专业写作样本，覆盖 12 种不同体裁的中英文文本；数据通过反向构造（先选取高质量参考，再生成对应指令和输入）得到；

**📈 对比分析**

对比方法：在 14 种主流 LLM（包括 GPT‑5.2、Gemini‑3 Pro、Claude‑4.5 以及多款开源模型）上进行实验；通过 Task Success Rate、Trajectory Efficiency、Refinement Density、Refinement Delta 等指标评估；结果显示专有模型整体领先，但在推理强的 LLM 组合下，生成模型可显著提升成功率，表明推理是关键；

**⚠️ 局限性**

局限性包括：① 仅覆盖英语和中文，无法推广至低资源语言；② 仅考察封闭书写操作，未涉及检索或多模态；③ 评判采用单轮 LLM‑as‑a‑judge，虽与人工高度一致，但缺少多轮交互式判定的细致度。

---

## 526. Streaming Continual Learning for Unified Adaptive Intelligence in Dynamic Environments

**arXiv ID:** 2603.01695 | [PDF](https://arxiv.org/pdf/2603.01695v1)

**作者:** Federico Giannini `[一作]` (Politecnico di Milano), Vincenzo Lomonaco `[通讯]` (University of Pisa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 Streaming Continual Learning (SCL) 框架，融合持续学习与流式机器学习的优势，实现对非平稳数据流的快速适应与记忆保持。

**💡 创新点**

创新点在于将 CL 的记忆与 SML 的漂移检测双重机制结合，形成快速-慢速双学习体系，并通过 CLS 理论阐述其生物学基础。

**🔧 技术方法**

采用了外部记忆、预序评估、漂移检测等技术，结合深度学习与统计学习模型的双层学习策略。

**📊 数据集**

文中未给出具体实验数据集，主要以概念性例子与通用数据流类型（域增量、类增量等）为示例。

**📈 对比分析**

与传统 CL、SML 和 OCL 的对比主要从理论和框架角度展开，未给出具体性能指标。

**⚠️ 局限性**

局限在于缺乏实证验证与定量评估，对多样化真实流式数据的适应性仍需进一步实验验证。

---

## 527. Pencil Puzzle Bench: A Benchmark for Multi-Step Verifiable Reasoning

**arXiv ID:** 2603.02119 | [PDF](https://arxiv.org/pdf/2603.02119v1)

**作者:** Justin Waugh `[一作]` `[通讯]` (Approximate Labs), Justin Waugh (Approximate Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个可验证的“笔算”拼图评测框架和对应的数据集，并在单轮提问和多轮 agentic 交互两种模式下，对数十种大型语言模型的推理能力进行了系统评估。

**💡 创新点**

其创新点在于：①实现了逐步验证的拼图求解过程，可定位具体规则违规；②利用该验证机制生成稠密的逐步奖励信号；③揭示了推理深度与 agentic 迭代这两条独立的能力轴。

**🔧 技术方法**

技术实现上结合了 pzprjs 引擎、cspuz-solver2 SAT 求解器、Python API 与工具调用框架，实现多轮交互与步骤级约束检查，并对多模型进行推理和 agentic 交互实验。

**📊 数据集**

使用了一个覆盖 20 多种拼图变体、包含唯一验证解和完整步骤轨迹的“黄金”测试集（300 题），以及更大规模的全数据集（数千题）。

**📈 对比分析**

通过比较单轮直接问答与多轮 agentic 交互两种策略，评估 30+ 模型的成功率。实验显示 agentic 能显著提升性能，例如 GPT‑5.2@xhigh 从 27% 提升至 56%；同时展示了推理力度与 agentic 迭代的相互作用。

**⚠️ 局限性**

主要局限包括：仅使用文本表示，未充分利用多模态输入；基线实验未进行 prompt 细化；样本量有限导致缺乏置信区间；未设置人类对照；高推理深度模式下存在较高的请求失败率。

---

## 528. InfoPO: Information-Driven Policy Optimization for User-Centric Agents

**arXiv ID:** 2603.00656 | [PDF](https://arxiv.org/pdf/2603.00656v1)

**作者:** Fanqi Kong `[一作]` (Peking University), Bang Liu `[通讯]` (Université de Montréal)

**通讯引用:** 1022 | [OpenAlex ID](https://openalex.org/A5100691219)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于信息增益的多轮交互强化学习方法InfoPO，用于优化面向用户的LLM代理的交互策略。

**💡 创新点**

创新点在于：①使用对抗性“假设掩码”计算细粒度的turn‑level信息增益奖励；②通过方差门控自适应融合信息增益与外部奖励，解决传统GRPO在稀疏奖励下的信用分配难题；③提供信息理论解释，证明累积信息增益是完成任务的必要资源。

**🔧 技术方法**

技术手段包括：对Dec‑POMDP建模、基于Group‑Relative Policy Optimization（GRPO）的优势估计、对比式信息增益计算、方差门控融合机制、KL约束的PPO风格更新。

**📊 数据集**

实验使用三个交互基准：UserGym（八类用户交互任务）、ColBench（协同编程与代码生成）和τ²‑Bench（跨域多轮决策），并在Sokoban、WebShop等非用户环境上进行泛化验证。

**📈 对比分析**

与提示、ReAct、Reflexion、UserRL、RAGEN、Search‑R1等基线比较，InfoPO在所有基准上显著提升任务完成率（平均提升约14%–16%），并在训练稳定性、样本效率及对用户/环境变化的鲁棒性方面表现更优。

**⚠️ 局限性**

局限性：①需要额外的对比推理计算，增加推理成本；②对“假设掩码”实现的敏感性仍需进一步研究；③在极端高维或极长序列的交互中，信息增益奖励可能需要更细粒度的归一化处理；④未对多模态输入/输出的适应性进行系统验证。

---

## 529. GeoDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis

**arXiv ID:** 2603.02172 | [PDF](https://arxiv.org/pdf/2603.02172v1)

**作者:** Srikumar Sastry `[一作]` (Washington University), Nathan Jacobs `[通讯]` (Washington University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 GeoDiT，一种点条件扩散 Transformer，能够通过少量点和自由文本提示实现卫星图像的空间与语义控制。

**💡 创新点**

创新点在于提出稀疏点条件框架和自适应局部注意（ALA）模块，既捕捉空间先验又实现细粒度的语义与空间约束，同时保持低成本高效推理。

**🔧 技术方法**

采用扩散 Transformer（DiT）架构、LongCLIP 文本编码、RANGE 地理位置嵌入、DinoV3 表示对齐、MetaRBF+Local Attention 的 ALA 模块，以及 AdaLN 进行三阶段训练。

**📊 数据集**

使用 Git-10M（约 200 万张 1 米分辨率卫星图像）与 OSM 矢量数据作为训练集，测试集包含 Git-Rand-15k、Git-Spatial-15k、RSICD 和 FMoW。

**📈 对比分析**

与 SDXL、SD3、PixArt-α/Σ、Text2Earth、GeoSynth 等基线进行定量（FID、LPIPS、SSIM、CLIP）与定性对比，GeoDiT 在所有指标上均优于现有卫星图像生成模型，点条件版本优于文本版。

**⚠️ 局限性**

主要局限在于仅支持点与文本提示，未涵盖线段或多边形等更复杂空间约束，且对地理位置依赖较强，极端遥感域外表现仍需验证。

---

## 530. TQCodec: Towards neural audio codec for high-fidelity music streaming

**arXiv ID:** 2603.01592 | [PDF](https://arxiv.org/pdf/2603.01592v1)

**作者:** Lixing He `[一作]`, Wenjiang Zhou `[通讯]` (Tencent Music Entertainment)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种面向 32–128 kbps 音乐流的神经音频编解码器，能够在 44.1 kHz 下实现高保真重建。

**💡 创新点**

创新点包括：① 使用 SEANet 结构实现低延迟、低算力（≈6.31 GMAC）解码；② 引入 SimVQ 解决中频细节丢失；③ 引入相位感知波形损失提升感知质量；④ 基于人耳敏感度的子频带比特分配策略，重点提升低频质量。

**🔧 技术方法**

技术手段包括：基于 DAC 的残差向量量化（RVQ）框架；SEANet 编码器/解码器；SimVQ 及其残差扩展；多尺度 mel‑谱、波形损失、对抗判别器；多通道子频带（PQMF）分解与不同维度网络。

**📊 数据集**

使用多大规模的音乐数据集：MusDBHQ（150曲）、Jingju（120曲）、Jamendo（55,609曲）、FMA（106,574曲）以及 10 万+ 曲目私有数据。

**📈 对比分析**

评估方法：采用 Log‑Spectral Distance（LSD）和 SNR，实验对比 DAC、Ogg‑Vorbis 等基线，结果显示 64 kbps 下 LSD 为 0.77、SNR 17.3 dB，128 kbps 下 LSD 0.67、SNR 18.2 dB，明显优于 48 kbps 的 Ogg‑Vorbis（LSD 0.76、SNR 16.9 dB）并在 MOS 上与 96 kbps Ogg‑HQ 接近。

**⚠️ 局限性**

局限性：① 仅针对 44.1 kHz 音乐测试，缺乏对语音或低采样率场景的验证；② 编码阶段算力高，主要部署在云端；② 子频带方案在极低比特率下对高频恢复仍有限；③ 目前尚未进行严格的实时部署与功耗评估。

---

## 531. Perspective-Equivariant Fine-tuning for Multispectral Demosaicing without Ground Truth

**arXiv ID:** 2603.01332 | [PDF](https://arxiv.org/pdf/2603.01332v1)

**作者:** Andrew Wang `[一作]` (University of Edinburgh), Mike Davies `[通讯]` (University of Edinburgh)

**通讯引用:** 5261 | [OpenAlex ID](https://openalex.org/A5009816332)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于视角等变性自监督微调的多光谱去马赛克框架 PEFD，能够仅利用马赛克测量无真实标注训练出高质量多光谱图像。

**💡 创新点**

创新点包括：①利用相机投影几何构造更丰富的群结构（同伦变换），从而恢复更多测量的空域信息；②在预训练的通用图像恢复模型（RAM）上进行参数高效微调，既继承了丰富的先验，又能针对多光谱任务适配；③设计无 GT 的视角等变性损失，解决传统测量一致性无法恢复 null‑space 的缺陷。

**🔧 技术方法**

技术手段包括：预训练的 RAM（Encoder‑Decoder）架构，视角等变性（投影变换/齐次坐标）损失，测量一致性+等变性联合损失，参数冻结+通道复制的高效微调策略。

**📊 数据集**

使用了两组公开数据集：HELICoiD（70 张人脑神经手术的 16 帧多光谱图像）和 HyKo（70 张车载多光谱图像），均在 4×4 MSFA 下生成马赛克测量。

**📈 对比分析**

与经典插值（双线性、加权双线性、Gaussian、PPID）、TV、DIP 以及最新自监督去马赛克方法（SDNet、DnCNN、EDSR、测量一致性微调、像素移位等变性）进行对比。PEFD 在 PSNR 上平均提升约 4 dB，SSIM 与 ERGAS 亦显著优于基线，并在细节恢复与光谱一致性上逼近监督学习结果。

**⚠️ 局限性**

局限性包括：①对相机姿态变化的依赖，缺乏对非旋转或极端视角场景的验证；②仅在 4×4 MSFA 结构下评估，未检验更稀疏或不同排列的滤波阵列；③视角等变性损失需要足够的视角样本，采样策略与摄像机运动约束仍需进一步研究；④在极低光或噪声严重的情况下，需进一步扩展到联合去噪等场景。

---

## 532. A Gauge Theory of Superposition: Toward a Sheaf-Theoretic Atlas of Neural Representations

**arXiv ID:** 2603.00824 | [PDF](https://arxiv.org/pdf/2603.00824v1)

**作者:** Hossein Javidnia `[一作]` (Dublin City University), Hossein Javidnia `[通讯]` (Dublin City University)

**通讯引用:** 581 | [OpenAlex ID](https://openalex.org/A5002473893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建离散规范理论框架，将大型语言模型的解释任务转化为本地图（chart）到全局图谱（atlas）的拼接问题，识别解释失效。

**💡 创新点**

提出三种可测量的全局解释障碍（局部拥堵、代理错位、非平凡 holonomy），给出可计算的正则化能量、下界证明及构造性 gauge 计算。

**🔧 技术方法**

使用 Fisher/高斯‑牛顿信息几何的局部度量、极化因子（polar factor）与线性回归估计跨 chart 的传输、稀疏编码的 over‑complete 字典、支撑向量图谱（spanning‑tree gauge）以及 Holonomy 计算。

**📊 数据集**

在 Llama 3.2 3B Instruct 的第 16 层激活上，以 WikiText‑103、C4 子集和代码数据为来源，聚类 128 个上下文 chart 进行实验。

**📈 对比分析**

通过对比理论下界与实际误差、覆盖率与零违规率、Bootstrap 稳定性评估，验证四个理论结果 A‑D 的有效性；诊断指标在不同种子与超参数下均保持稳健，性能表明框架能够可靠地量化解释失效。

**⚠️ 局限性**

局限在于代理定义的依赖性、线性传输模型的简化、未证明拓扑非平凡、计算成本高且仅适用于冻结模型。

---

## 533. CodecFlow: Efficient Bandwidth Extension via Conditional Flow Matching in Neural Codec Latent Space

**arXiv ID:** 2603.02022 | [PDF](https://arxiv.org/pdf/2603.02022v1)

**作者:** Bowen Zhang `[一作]` (Nanyang Technological University), A S Madhukumar `[通讯]` (Nanyang Technological University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于神经音频编码器的带宽扩展框架CodecFlow，用于在低分辨率语音输入下恢复高频内容并提升清晰度和可懂度。

**💡 创新点**

创新点包括：①声带感知的条件流匹配模型（FEC），在连续编码器潜在空间中根据声学V/UV信息进行精确高频重建；②结构约束残差向量量化（SC‑RVQ），通过边际与单调约束提升离散量化稳定性；③端到端联合训练，实现潜在空间与解码器的同步优化。

**🔧 技术方法**

采用了Descript Audio Codec（DAC）作为基础编码器，结合条件流匹配（CFM）技术、Conformer+U‑Conformer网络、分类器无关引导（CFG）以及量化正则化。

**📊 数据集**

使用了LibriTTS（约40h 16kHz、100h 16kHz）训练8→16kHz模型，VCTK（44.1kHz）训练8→44.1kHz模型，并在TIMIT及VCTK held‑out语音上进行评测。

**📈 对比分析**

与NU‑Wave2、AP‑BWE、Fre‑Painter和FlowHigh等代表性方法比较，CodecFlow在LSD、LSD‑HF、VISQOL、MOS和Coloration等指标上均取得最优或相近表现，尤其在高频重建和语音清晰度方面优于基准。

**⚠️ 局限性**

局限性包括：对极端低采样率或大幅度噪声环境的鲁棒性尚未充分验证；模型对计算资源仍有一定需求，尤其在高采样率下的推理速度相对慢；量化时可能出现微小的重建失真。

---

## 534. Cognitive Prosthetic: An AI-Enabled Multimodal System for Episodic Recall in Knowledge Work

**arXiv ID:** 2603.02072 | [PDF](https://arxiv.org/pdf/2603.02072v1)

**作者:** Lawrence Obiuwevwi `[一作]` (Old Dominion University), Sampath Jayarathna `[通讯]` (Old Dominion University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了Cognitive Prosthetic Multimodal System (CPMS)，通过同步语音、心率/皮肤电反应、眼动等多模态传感器，在本地捕获、对齐并存储为JSON格式的事件日志，并提供基于大型语言模型的自然语言检索接口，支持工作场景中的回忆与反思。

**💡 创新点**

创新点在于：① 将多模态传感器数据在秒级时间轴上统一对齐并结构化为“记忆切片”；② 采用隐私友好的本地端处理与存储；③ 将结构化事件日志与LLM结合，允许自然语言查询跨模态、跨时间的回忆；④ 设计了可模块化、可在传感器缺失时仍能工作，并包含多重隐私与伦理保障。

**🔧 技术方法**

技术包括：自动语音识别 (Whisper/Google STT)、语音分离与情绪/心率计算 (EmotiBit)、眼动追踪 (Pupil Labs Core)、时间同步与重采样、JSON/JSONL 结构化存储、LLM (如 GPT/ChatGPT) 的检索与语义匹配、Web 前端与后端交互。

**📊 数据集**

数据来源为工作现场的实测传感器流（语音、心率、皮肤电、眼动），未使用公开标准数据集；若需要可扩展至公开的工作场景多模态数据集（如 eDiary, Ego4D 等）。

**📈 对比分析**

系统并未进行正式的基准比较或性能评估，仅在原型验证阶段展示了语音转录、模态同步、JSON日志生成以及自然语言检索的可行性；暂无客观指标如检索准确率、召回率或使用者体验分数。

**⚠️ 局限性**

局限性包括：① 未开展用户研究，缺乏对回忆效果、认知负荷与信任度的实证评估；② 依赖高质量ASR与眼动数据，噪声或缺失会影响检索质量；③ 仅在本地实现，尚未验证大规模多用户或长期部署的可扩展性；④ 伦理与隐私方面仍需完善（如旁观者同意、持续监控等）。

---

## 535. Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning

**arXiv ID:** 2603.00667 | [PDF](https://arxiv.org/pdf/2603.00667v1)

**作者:** Wentao Huang `[一作]` (Stony Brook University), Chen Wang `[通讯]` (Mayo Clinic)

**通讯引用:** 19015 | [OpenAlex ID](https://openalex.org/A5100337669)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为HistoSelect的问答引导、组织结构感知、粗细层级的图像检索框架，用于提高病理学视觉问答（VQA）的效率和可解释性。

**💡 创新点**

创新点在于：①结合病理学家设计的组织类型提示实现多层次组织分割；②使用信息瓶颈（IB）理论驱动的两阶段选择（组采样+补丁选择），实现对问题相关视觉信息的端到端学习；③采用硬/软选择结合STE的可微分采样，兼顾稀疏性与性能。

**🔧 技术方法**

主要技术包括：CLIP式组织分割、预训练视觉编码器（CONCH）、LLM（Qwen2.5-7B-Instruct）、信息瓶颈（Variational IB）、Straight‑Through Estimator、两层线性采样器、token预算控制。

**📊 数据集**

使用了三大公开数据集（SlideBench‑VQA、WSI‑Bench）和一个内部卵巢癌数据集，涵盖多种病理学问题类型（显微镜学、诊断、临床、形态学、治疗规划）。

**📈 对比分析**

与多种基准模型（GPT‑4o、Quilt‑LLaVA、WSI‑VQA、LLaVA‑Med、SlideChat、MI‑Gen、Histo‑Gen）进行对比，HistoSelect在闭合式问答的准确率和开放式文本生成（BLEU、ROUGE、WSI‑Precision/Recall）上均取得最优或接近最优结果，平均准确率达84.6%，生成质量比对手提升约5%。

**⚠️ 局限性**

局限性包括：①对组织提示的依赖，需要临床专家参与设计；②在token预算极低（≤1k）时性能下降；③目前仅在WSI级别问答上验证，跨模态扩展与实时部署仍待研究。

---

## 536. THz RHS Transceiver for Low-Latency Multi-User VR Transmission with MEC

**arXiv ID:** 2603.01888 | [PDF](https://arxiv.org/pdf/2603.01888v1)

**作者:** Liangshun Wu `[一作]` (Shanghai Jiao Tong University), Ying Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种基于THz频段、可重构全息面（RHS）天线的多用户VR传输体系，并通过移动边缘计算（MEC）实现内容预取、渲染和射束重配置的联合优化。

**💡 创新点**

创新点在于：①首次将RHS天线与THz链路结合用于VR；②在模型中加入互耦效应；③在同质和异质场景下分别给出闭式最优策略或分解优化框架；④使用MMKP启发式、DCA/CCP以及投影梯度方法求解复杂非凸问题。

**🔧 技术方法**

主要技术包括：THz高频链路建模、RHS全息波束成形与多路访问（HDMA）、MEC边缘渲染与预取、离散优化（MMKP）与连续优化（DC/CCP、PG）。

**📊 数据集**

实验使用合成的360°视场（FoV）数据集，FoV大小服从均匀或Zipf分布，模拟不同用户、频段、功率与计算资源配置；未使用公开真实VR视频数据集。

**📈 对比分析**

与等权重、随机权重、无预取等基线相比，提出的方法在延迟上相对基线下降显著（多倍），尤其在内存/功率受限或用户移动场景下表现突出；仿真报告了不同参数（内存、功率、CPU频率、传输速率）的性能曲线。

**⚠️ 局限性**

局限性：① 依赖静态或半静态环境假设，移动重配置时的实时性未完全验证；② 优化求解复杂度高，难以直接嵌入实时系统；③ 仅使用仿真数据，缺乏真实硬件验证；④ 对用户行为模型（IRM、Zipf）假设相对理想，实际分布可能更复杂。

---

## 537. Uncertainty Quantification of Click and Conversion Estimates for the Autobidding

**arXiv ID:** 2603.01825 | [PDF](https://arxiv.org/pdf/2603.01825v1)

**作者:** Ivan Zhigalskii `[一作]` (Avito), Egor Samosvat `[通讯]` (Avito)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于贝叶斯推理的自动竞价方法，利用后验期望对噪声CTR与CVR进行去噪，并给出闭式竞价公式。

**💡 创新点**

创新点包括：① 将SPA中的CTR/CVR噪声建模为贝叶斯问题，推导出基于后验期望的闭式竞价规则；② 通过极限去卷积（XDGMM）从噪声样本恢复CTR/CVR的先验分布；③ 对联合CTR‑CVR协方差给出了高效近似计算方案。

**🔧 技术方法**

使用技术包括：贝叶斯推理、极限去卷积（XDGMM）、高斯混合模型、logit空间噪声模型、Gauss‑Hermite积分、CatBoost虚拟集成估计不确定性、线性规划及其对偶求解。

**📊 数据集**

实验数据集涵盖 Synthetic、iPinYou、BAT 与 Criteo Attribution 四个真实或合成数据集。

**📈 对比分析**

与非鲁棒基线和鲁棒优化方法对比，采用 R/R*（目标与理论最优比例）和 CPC/CPC_camp（实际/目标成本/点击比）衡量。结果显示，贝叶斯方法在噪声增大时仍能保持约 90% 以上的 R/R*，并严格满足 CPC 限制，显著优于对比方法。

**⚠️ 局限性**

局限性：实验仅在 GBDT 预测模型上验证，未覆盖深度学习模型；方法目前仅在第二价拍卖（SPA）框架下验证，未推广至一次价拍卖。

---

## 538. ProtRLSearch: A Multi-Round Multimodal Protein Search Agent with Large Language Models Trained via Reinforcement Learning

**arXiv ID:** 2603.01464 | [PDF](https://arxiv.org/pdf/2603.01464v1)

**作者:** Congying Liu `[一作]` (University of Chinese Academy of Sciences), Tiehan Cui `[通讯]` (Henan University)

**通讯引用:** 74551 | [OpenAlex ID](https://openalex.org/A5009237669)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ProtRLSearch，一个多轮多模态蛋白搜索代理，利用强化学习训练，使LLM在搜索过程中同时考虑蛋白序列和文本，实现更精准的蛋白功能与变异推理。

**💡 创新点**

创新点在于将蛋白序列作为并行输入融入搜索规划，并设计多维奖励（关键词、工具、格式、答案）引导多轮搜索；同时构建ProtMCQs基准评估序列约束推理能力。

**🔧 技术方法**

技术包括蛋白序列表征（如ESM-2）、LLM（Qwen3-8B）、多轮搜索框架（Planner/Retriever/Executor）以及基于多维奖励的强化学习。

**📊 数据集**

使用的数据集为3000条多模态样本（ProtMCQs 3000道多选题）作为训练集，并在BioMedMCQs、ProtMCQs、MedMCQA、MedQA等数据集上进行评测。

**📈 对比分析**

与Baseline、BioMedSearch、BioReason、Search-R1、ProtLLM等方法对比，ProtRLSearch在BioMedMCQs水平1-3分别达89.2%、75.8%、71.7%；在ProtMCQs 86.9%、77.4%、72.5%，显著优于对照组，推理时间控制在10-25秒，效率可接受。

**⚠️ 局限性**

局限性包括对蛋白序列表征的依赖导致对未见蛋白家族泛化受限，强化学习奖励仍偏重最终答案，早期搜索误差可能放大；缺乏在线监督与动态知识更新机制。

---

## 539. MiniUGV$_2$: A Compact UAV-Deployable Tracked Ground Vehicle with Manipulation Capabilities

**arXiv ID:** 2603.00972 | [PDF](https://arxiv.org/pdf/2603.00972v1)

**作者:** Durgakant Pushp `[一作]`, Lantao Liu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一款可由UAV部署的迷你底盘地面机器人 miniUGV_2，并开发了带电永久磁头的缆绳模块，实现无人机对地面机器人的部署、检索以及独立操作。

**💡 创新点**

创新点包括：①双关节全舵机手臂支持 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 360° 1

**🔧 技术方法**

ROS框架下的视觉惯性定位与地图构建（VINS+Voxel Mapping）、B-spline轨迹规划、PID控制、点云聚类与图像分割、缆绳跟踪、电子永久磁头控制、MuJoCo物理仿真。

**📊 数据集**

实验数据主要来自于无人机和地面机器人在真实室内外场景中的实时RGB-D点云与传感器采集，未使用公开标准数据集。

**📈 对比分析**

与EyeDrive、miniUGV_1等现有平台在尺寸、重量、操作时间、传感器与计算平台等指标进行比较；在实验中，miniUGV_2能在崎岖地形自右、跨越高障碍、操作3.5 kg物体；无人机部署与检索成功率高，缆绳模块可靠，整体性能优于之前的 miniUGV。

**⚠️ 局限性**

（1）检索时磁头精确对接在真实户外环境尚未验证；（2）机动与操纵算法相对简单，需进一步强化；（3）目前手臂为手动控制，缺乏完全自主；（4）对复杂动态环境的适应性有限。

---

## 540. Better Matching, Less Forgetting: A Quality-Guided Matcher for Transformer-based Incremental Object Detection

**arXiv ID:** 2603.01524 | [PDF](https://arxiv.org/pdf/2603.01524v1)

**作者:** Qirui Wu `[一作]` (Northwestern Polytechnical University), Peng Wang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 18987 | [OpenAlex ID](https://openalex.org/A5100395960)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对增量目标检测中 DETR 结构的灾难性遗忘问题，提出了一种基于质量导向的最小成本最大流匹配器（Q‑MCMF），通过剪枝低 IoU 边来避免强制匹配从而消除背景前景化；

**💡 创新点**

创新点在于将最小成本最大流算法与 IoU 质量阈值相结合，形成可剔除不合理匹配、同时最大化有效匹配的匹配策略，显著提升了 DETR 在增量学习中的稳定性与可塑性；

**🔧 技术方法**

使用技术包括 DETR/Deformable DETR 检测框架、Hungarian 匹配、最小成本最大流（MCMF）优化、IoU 质量剪枝、焦点损失、L1 与 GIoU 边框回归等；

**📊 数据集**

实验基于 COCO 2017 数据集，在 40-40、70-10、4010×4、4020×2 等多种增量划分下进行评估；

**📈 对比分析**

与 LwF、CL‑DETR、SDDGR、DyQ‑DETR、RILOD、SID、ERD、DCA 等多种基线（含有/无样本回放）对比，结果显示无样本回放的 Q‑MCMF 在 AP 上提升约 4%–7%，在单阶段和多阶段设置均显著优于现有最优方法；

**⚠️ 局限性**

局限性包括需要手动调节 IoU 阈值 α、β，且验证范围主要集中在 DETR 系列模型，尚未在其他检测框架或极端小样本/高类别数场景中充分验证。

---

## 541. HarmonyCell: Automating Single-Cell Perturbation Modeling under Semantic and Distribution Shifts

**arXiv ID:** 2603.01396 | [PDF](https://arxiv.org/pdf/2603.01396v1)

**作者:** Wenxuan Huang `[一作]` (Fudan University), Siqi Sun `[通讯]` (Fudan University)

**通讯引用:** 5285 | [OpenAlex ID](https://openalex.org/A5023777406)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个端到端的自动化代理（HarmonyCell），能够在不需要人工预处理或手动建模的情况下，从异构单细胞扰动数据中自动完成数据统一、模型设计和评估，实现“虚拟细胞”建模。

**💡 创新点**

创新点在于：①将语义异构问题通过LLM驱动的“语义统一器”自动映射为统一接口；②使用自适应蒙特卡洛树搜索（MCTS）在层次化动作空间中系统搜索最优统计归纳偏置，从而在分布漂移下保持高性能；③将语义统一与结构搜索耦合，形成双轨协同工作流，显著提升可复现性与跨数据集迁移能力。

**🔧 技术方法**

主要技术包括：大语言模型（LLM）用于生成语义映射规范和代码片段；层次化动作空间（宏观模型范式 → 结构骨干 → 优化细化）用于约束MCTS搜索；自适应MCTS算法（UCB、回溯更新、重用历史先验）；检索增强代理用于获取历史经验；以及基于执行结果的多目标奖励（准确度+计算效率）评估。

**📊 数据集**

实验使用了多种公开扰动数据集：Adamson、Replogle（CRISPRi扰动）、Norman、Srivatsan（药物剂量扰动）、以及各自的子集（如未见扰动/细胞）。这些数据涵盖了语义异构（不同元数据模式）和统计异构（连续剂量、离散基因编辑）两类分布偏移。

**📈 对比分析**

与通用编程代理（AIDE、R&D Agent）相比，HarmonyCell 在语义异构场景下实现了 95% 的有效执行率（对比 0%）。在多数据集、分布漂移评估中，自动化模型在 RMSE、DeltaPCC、CosLogFC 等指标上均能匹配或超过 Biolord、Sams VAE、CPA 等专家设计基线，尤其在未见扰动/细胞的 OOD 场景中表现尤为突出。

**⚠️ 局限性**

限制：①对 LLM 的推理能力和提示工程依赖较高，可能在不同 LLM 版本下效果波动；②自适应 MCTS 需要较长训练时间和较高计算资源；③在极度异构或极端稀疏的数据集上，语义统一器的泛化能力仍有待验证；④当前实现主要针对单细胞 RNA-Seq 拖动扰动，其他多模态或跨实验平台的兼容性尚未系统评估。

---

## 542. SimAB: Simulating A/B Tests with Persona-Conditioned AI Agents for Rapid Design Evaluation

**arXiv ID:** 2603.01024 | [PDF](https://arxiv.org/pdf/2603.01024v1)

**作者:** Tim Rieder `[一作]` (ETH Zurich), Mustafa Doga Dogan `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SimAB，一个利用基于角色的 AI 代理对网页设计变体进行快速、隐私友好的模拟 A/B 测试系统，能够在数分钟内给出优劣判定和可操作的设计建议。

**💡 创新点**

创新点：①将用户画像与 LLM 结合生成多样化的合成用户；②采用中性命名与计数平衡消除顺序偏差；③通过统计累积序列实现早停，显著加速实验；④聚合代理判定并生成可执行的理由，提升解释性；⑤在历史真实 A/B 实验上验证其可行性与准确度。

**🔧 技术方法**

核心技术：多模态大型语言模型（如 GPT‑4），RAG（检索增强生成）用于从上下文文档提取信息；图像输入与自然语言交互的 prompt 模式；批量生成并多样化筛选的 persona 生成；对称呈现与计数平衡的 counterbalancing；使用渐进式置信区间进行统计聚合；JSON 结构化响应与后续理由聚合。

**📊 数据集**

数据集：47 个历史 A/B 测试（包含电商网站、桌面应用与维基百科公开实验），每个测试都有控制与挑战版本截图、转化目标与可选受众描述；此外收集了 14 家企业实验者的反馈与实操案例。

**📈 对比分析**

比较方法：在 47 个已知结果的实验上进行后测；使用混淆矩阵、准确率、召回率、F1 等指标；对高置信度（≤70 代理）与超高置信度（≤20 代理）的子集分别评估；还与无偏差计数平衡与无多样性 persona 的 ablation 进行对比。性能：整体准确率 67%，高置信度 75% 以上，超高置信度 80%+；精确度、召回率与 F1 亦随代理数量下降而提升；相较传统 A/B，SimAB 在 60% 的实验中实现快速显著性，显著高于传统约 20–30%。

**⚠️ 局限性**

局限性：①仅评估静态截图，无法捕捉加载速度、动画、悬停、滚动等动态交互；②对细微效果的预测准确度有限；③合成 persona 继承 LLM 训练偏差，可能不代表全球多样化受众；④需大量 LLM 调用，成本与计算资源仍是瓶颈；⑤对全新、与训练分布差距较大的界面预测可靠性下降；⑥潜在伦理风险（易被用于快速迭代操纵性设计），需要使用规范与审查。

---

## 543. Stateful Cross-layer Vision Modulation

**arXiv ID:** 2603.00655 | [PDF](https://arxiv.org/pdf/2603.00655v1)

**作者:** Ying Liu `[一作]` (Beijing Institute of Technology), Liyuan Pan `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 753 | [OpenAlex ID](https://openalex.org/A5064162540)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉编码器内部通过递归跨层记忆调控实现多层视觉特征的动态融合。

**💡 创新点**

引入跨层记忆更新模块（TMSU）与令牌自适应门（TAG），在每层进行反馈调节，而非仅在读出时聚合特征。

**🔧 技术方法**

使用跨层记忆更新、令牌自适应门、语义对齐损失，以及冻结的 CLIP ViT‑L/14‑336 与 LLM 的投影机制。

**📊 数据集**

训练使用 LLaVA‑Instruct‑665K 数据集的 20K 采样子集，评测在 DocVQA、MME、SQA 等视觉问答数据集。

**📈 对比分析**

与 Dense Connector、MMFuser、TGIF 等静态多层融合方法对比，SCVM 在 DocVQA 21.00、MME 1520.60、SQA 70.10 等指标上取得最佳或相近成绩，同时仅微调轻量模块。

**⚠️ 局限性**

仅改造预训练视觉编码器，未能彻底避免在更深层次可能出现的细粒度信息丢失；对超大规模视觉模型的可扩展性和对长文本任务的适应性仍需进一步验证。

---

## 544. AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth

**arXiv ID:** 2603.01914 | [PDF](https://arxiv.org/pdf/2603.01914v1)

**作者:** Shixiang Song `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AdaPonderLM，一种自监督的可变深度递归语言模型，在推理时根据每个 token 的难度动态决定循环次数。

**💡 创新点**

核心创新在于：①使用迭代特定的 MLP 门与单调掩码实现 token 级别的早停；②引入 KV 重用机制保证训练与推理的一致性；③通过自监督的 bottom‑K 惩罚让模型在预训练阶段自然学习停留策略。

**🔧 技术方法**

技术主要包括：Transformer 共享权重的递归细化、MLP 门、monotonic 掩码、KV 缓存重用、ponder 目标正则化、两个阶段预训练（warm‑up 与 ponder regularization）。

**📊 数据集**

在 Pile 数据集上预训练，随后在多种 Pythia 后端（70M、410M、1.4B、2.8B）进行实验，并在 LAMBADA、PIQA、WinoGrande、ARC‑Easy/Challenge、SciQ、HellaSwag、RACE 等下游任务上评估。

**📈 对比分析**

与固定深度的 PonderLM、Pause Token、Loop Transformer 等基线对比，AdaPonderLM 在保持相同或更低 perplexity/下游准确率的同时，推理 FLOPs 下降约 8–10%；在 iso‑FLOPs 下的评估也显示其自适应策略优于均匀或几何分布。

**⚠️ 局限性**

主要局限：①对 hyper‑parameters k 与 λ 选择敏感；②实验规模受限，尚未验证在更大模型/数据集上的可扩展性；③额外的 MLP 门虽参数少但增加了训练开销。

---

## 545. InterCoG: Towards Spatially Precise Image Editing with Interleaved Chain-of-Grounding Reasoning

**arXiv ID:** 2603.01586 | [PDF](https://arxiv.org/pdf/2603.01586v1)

**作者:** Yecong Wan `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 62202 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 InterCoG，通过文本-视觉交错的链式定位推理实现复杂多实体场景的精准图像编辑

**💡 创新点**

将位置推理、视觉定位与编辑描述重写三步交错进行，并加入多模态重建与对齐监督，提升定位精度与可解释性

**🔧 技术方法**

基于 Bagel 的统一多模模型，结合文本推理、视觉定位、掩码重建监督、跨模态对齐及 rectified flow 生成技术

**📊 数据集**

使用自建 GroundEdit-45K（45K 细粒度编辑样本）以及 GroundEdit-Bench 评测基准

**📈 对比分析**

在 GroundEdit-Bench 上 EGA 达 0.88、ES 达 3.97，明显优于 Bagel、Qwen-Image-Edit 等最新指令式编辑模型，展示更高的定位准确率和编辑质量

**⚠️ 局限性**

缺点是链式定位步骤导致推理延迟显著增加（比 Qwen-Image-Edit 慢约 26 秒），对计算资源和模型规模有一定要求

---

## 546. DARS: Dysarthria-Aware Rhythm-Style Synthesis for ASR Enhancement

**arXiv ID:** 2603.01369 | [PDF](https://arxiv.org/pdf/2603.01369v1)

**作者:** Minghui Wu `[一作]` (University of Science and Technology of China), Yue Zhang `[通讯]` (Huawei Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 DARS 框架，用多阶段节奏预测与病态音响条件流匹配技术，生成与真实病态言语高度相似的合成语音，用于数据增强并显著提升 Whisper ASR 在 TORGO 数据集上的识别性能。

**💡 创新点**

创新点在于：① 引入停顿预测与多阶段节奏预测结合的 CPO（对比偏好优化）机制，精准捕捉病态语音的节奏碎片化特征；② 在 Matcha‑TTS 的条件流匹配中加入全局和局部病态音响风格向量，使合成过程能够同时模拟病态节奏与音质；③ 将合成语音用于 Whisper 大模型微调，实现比现有 SOTA 系统更低的 WER。

**🔧 技术方法**

使用了 Matcha‑TTS、对比偏好优化（CPO）、条件流匹配（OT‑CFM）、全局与局部风格向量（GST + VQ‑style），以及 Whisper‑Large 的全参数微调与 LoRA 微调技术。

**📊 数据集**

主要数据集为 TORGO（含 8 名病患与 7 名健康说话人，分为四个严重程度），实验中还使用 LibriSpeech 文本与随机引用 TORGO 语音进行跨域合成。

**📈 对比分析**

与基线 Matcha‑TTS、Grad‑TTS、FastSpeech2+Whisper 及 DNN‑HMM 等对比，DARS 在 TORGO 上实现了整体 WER 54.22% 的相对降低（与 E19 相比），并在各病情等级（严重到轻度）均优于 SOTA，尤其在严重病例提升最显著；跨域 LibriSpeech‑TORGO 试验中整体 WER 进一步降至 7.75%。

**⚠️ 局限性**

局限性包括：① 对极轻度病变的节奏捕捉仍有提升空间；② 依赖高质量停顿标注与风格向量，标注成本高；③ 目前仅在 TORGO 与 LibriSpeech 上验证，跨域通用性及多说话人环境需进一步研究；④ 研究缺少主观评测，需补充人类听感评价。

---

## 547. Architecture-Aware Multi-Design Generation for Repository-Level Feature Addition

**arXiv ID:** 2603.01814 | [PDF](https://arxiv.org/pdf/2603.01814v1)

**作者:** Mingwei Liu `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研发了一个仓库级架构感知的多设计特性实现框架，用于自然语言驱动的特性添加。

**💡 创新点**

创新点在于构建代码图进行架构感知定位，采用多设计生成多样化实现方案，并通过静态动态影响评估实现鲁棒的补丁选择。

**🔧 技术方法**

技术手段包括代码图构建与多轮检索、LLM多设计生成、静态影响分析、动态回归与新功能测试评估。

**📊 数据集**

使用了NoCode‑bench Verified数据集（114个高质量Python仓库的特性添加任务）。

**📈 对比分析**

通过与Agentless和OpenHands等基线进行对比，取得39.47%的成功率（比最佳基线提升36.34%），在多种LLM上表现稳健，且在跨文件修改上表现突出。

**⚠️ 局限性**

局限性在于实验仅针对Python项目，需验证跨语言适用性；对极大仓库的可扩展性待进一步评估；依赖LLM的推理能力和代码图构建的准确性。

---

## 548. LLM-assisted Semantic Option Discovery for Facilitating Adaptive Deep Reinforcement Learning

**arXiv ID:** 2603.01488 | [PDF](https://arxiv.org/pdf/2603.01488v1)

**作者:** Chang Yao `[一作]`, Hankz Hankui Zhuo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种闭环框架LLM-SOARL，结合大型语言模型、符号规划与深度强化学习，实现自动发现与重用通用技能，并将自然语言约束实时转化为可执行的安全监控规则；

**💡 创新点**

创新点在于：①利用LLM生成语义标签，构建技能库并实现跨任务、跨环境的技能迁移；②通过LLM解析自然语言约束，构造限制集并嵌入奖励机，实现实时约束检测与纠正；③将符号规划与选项框架闭环结合，提升数据效率与可解释性；

**🔧 技术方法**

使用技术包括大型语言模型（LLM）进行语义解析与标签生成、PDDL符号规划、选项框架（HRL）、深度Q学习、Reward Machine约束监控；

**📊 数据集**

实验数据集为Office World（含两种场景A、B）与经典高维稀疏奖励环境Montezuma's Revenge；

**📈 对比分析**

与基线SORL对比，LLM-SOARL在两种实验设置下均显著提升了数据效率（收敛速度快）、约束合规性（违约次数大幅下降）和跨任务可迁移性（任务切换后几乎无样本需求）；

**⚠️ 局限性**

局限性包括对LLM推理结果的依赖、对复杂自然语言约束的准确性仍有限、以及在更大规模或动态环境下的可扩展性和实时性能需要进一步验证。

---

## 549. Decentralized Federated Learning by Partial Message Exchange

**arXiv ID:** 2603.01730 | [PDF](https://arxiv.org/pdf/2603.01730v1)

**作者:** Shan Sha `[一作]` (Beijing Jiaotong University), Geoffrey Ye Li `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为PaME的去中心化联邦学习算法，通过部分消息交换实现通信效率、隐私和模型精度的平衡。

**💡 创新点**

创新点在于：①引入随机稀疏坐标交换机制，既降低通信量又保持无偏估计；②在仅要求局部梯度Lipschitz且通信矩阵双随机的前提下实现线性收敛；③支持时间变化、稀疏且非双随机的通信图。

**🔧 技术方法**

采用随机稀疏采样、局部梯度线性化、压缩传输、动态通信间隔以及双随机权重矩阵；理论上利用局部Lipschitz性和谱间隙保证线性收敛。

**📊 数据集**

在多种数据集上验证：线性回归与逻辑回归的合成数据；Fashion‑MNIST的CNN；CIFAR‑10的ResNet‑20；并通过不同的非IID划分（类别划分与Dirichlet划分）进行鲁棒性测试。

**📈 对比分析**

与D‑PSGD、DFedSAM、BEER、ANQ‑NIDS等基线在收敛速度、通信轮数和总传输量上进行对比；PaME在所有实验中均实现更快收敛、更低通信量和更高最终精度。

**⚠️ 局限性**

局限性包括：对通信图的随机性和参数设置（如s、ν、κ_i）的依赖；在极端异构或通信极稀疏情形下仍可能出现收敛速度下降；理论分析仍需假设梯度局部Lipschitz且通信矩阵初始双随机。

---

## 550. CueNet: Robust Audio-Visual Speaker Extraction through Cross-Modal Cue Mining and Interaction

**arXiv ID:** 2603.01530 | [PDF](https://arxiv.org/pdf/2603.01530v1)

**作者:** Jiadong Wang `[一作]` (Technical University of Munich), Björn Schuller `[通讯]` (Technical University of Munich)

**通讯引用:** 54325 | [OpenAlex ID](https://openalex.org/A5043060302)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种鲁棒的音视听说话人提取框架，在不需要训练时使用受损视觉数据的情况下，能够在视觉降质环境下保持高性能。

**💡 创新点**

创新点在于通过层次化的音视听学习器将视觉信息分解为说话人信息、声学同步和语义同步三种线索，并使用自适应交互模块动态融合这些线索，从而提升在视觉降质条件下的鲁棒性。

**🔧 技术方法**

主要技术包括跨模态交互学习、层次化线索解耦、基于 k‑means 的音频与语义监督、注意力加权的线索交互模块以及使用 SeaNet 结构的后端掩码估计。

**📊 数据集**

实验使用 LRS3 和 VoxCeleb2 两个公开数据集，并在四种视觉降质（高斯模糊、遮挡、特征遮蔽、面孔缺失）下评估。

**📈 对比分析**

与多种 SOTA 方法（VisualVoice、ConvTasNet、AVLiT、CTCNet、Seanet、IIANet 等）比较，CueNet 在所有降质条件下均显著优于基线，尤其在面孔缺失时提升超过 3 dB；在交叉域测试中仍保持领先。

**⚠️ 局限性**

局限性包括：对音视频同步的依赖仍可能在极端动态场景下受限；k‑means 聚类的超参数选择对性能有影响；模型对极端噪声和多说话人混合的处理尚未充分验证。

---

## 551. Implicitly Parallel Neuromorphic Solver Design for Constraint Satisfaction Problems

**arXiv ID:** 2603.01150 | [PDF](https://arxiv.org/pdf/2603.01150v1)

**作者:** Recep Bugra Uludag `[一作]` (University of Minnesota), Ulya R Karpuzcu `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并验证了一种基于神经采样的并行发现启发式，利用神经网络的多值表示在单个运行中同时探索多条解。

**💡 创新点**

创新点在于根据约束图中变量度数动态调整WTA抑制，使得低度数变量保持多值状态，从而在神经形态计算中实现真正的并行解探索。

**🔧 技术方法**

使用了神经采样（spiking neural network）与WTA/OR神经图样，基于神经采样的能量函数和基于度数的WTA权重调节算法。

**📊 数据集**

使用了平面图着色、SATLIB随机可满足实例、Sudoku、Ising等标准CSP基准集；具体包括0.8密度的平面图着色实例（9、25、36、49节点），SAT（uf20-91、uf50-218、uf75-325），Sudoku（easy、hard、AI Escargot），Ising（10-spin anti-ferromagnetic ring、10^3 ferromagnetic/anti-ferromagnetic cube）。

**📈 对比分析**

与传统序列基线（无启发式）和现有神经形态求解器（SpiNNaker、Neo-Cortical、Loihi）进行比较；实验显示，在单解发现上平均时间提升超过两倍，整体多解发现速度提升数倍，且与多次运行基线相比可缩短数倍；总求解时间略慢，但多解收益明显。

**⚠️ 局限性**

局限性包括：在高度约束或循环结构的实例中，度数启发式效果有限，可能无法利用并行性；少数情况下仍出现单解输出；在真正单解问题时，启发式退化为传统单值搜索，提升有限；对高度密集图或高域大小的CSP，效果下降。

---

## 552. FLICKER: A Fine-Grained Contribution-Aware Accelerator for Real-Time 3D Gaussian Splatting

**arXiv ID:** 2603.01158 | [PDF](https://arxiv.org/pdf/2603.01158v1)

**作者:** Wenhui Ou `[一作]` (Hong Kong University of Science and Technology), Chik Patrick Yue `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种面向实时3D高斯喷涂（3D Gaussian Splatting）的贡献感知加速器——FLICKER，能够在边缘设备上实现像素级精细的高斯剔除；

**💡 创新点**

创新点包括：自适应领导像素策略、像素矩形分组优化、两级层次化高斯测试以及混合精度贡献测试单元；

**🔧 技术方法**

采用硬软协同设计，融合自适应领导像素、像素矩形分组、层次化测试与FP16/FP8混合精度的贡献测试单元；

**📊 数据集**

使用真实场景数据集：Tanks & Temples、Mip-NeRF360（户外）、Deep Blending（室内）等；

**📈 对比分析**

与GSCore和Jetson XNX GPU进行对比，实验显示相较于GSCore实现1.5×速度提升、2.6×能效提升、14%面积下降；相较于XNX GPU实现19.8×速度提升、26.7×能效提升；

**⚠️ 局限性**

局限在于仍需要对深度缓冲区进行FIFO管理，且对极其稀疏或高度离散的高斯场景的自适应策略可能需要进一步优化；

---

## 553. Noise-Calibrated Inference from Differentially Private Sufficient Statistics in Exponential Families

**arXiv ID:** 2603.02010 | [PDF](https://arxiv.org/pdf/2603.02010v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Samhita Pal `[通讯]` (Vanderbilt University Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并评估了一套基于指数族充分统计量的差分隐私数据发布与推断管线，发布噪声化充分统计量后进行噪声校正的似然推断以及可选的参数化合成数据生成。

**💡 创新点**

提出针对指数族的完整频率学术理论，包括显式方差膨胀、Wald CI、噪声感知估计以及匹配下界，统一了DP推断与DP合成数据的文献。

**🔧 技术方法**

使用高斯机制发布噪声化充分统计量，利用delta方法和CLT证明渐近正态性，构造噪声感知似然优化与引导抽样，并讨论矩阵机制与高斯校准。

**📊 数据集**

在三类指数族（高斯均值、逻辑回归、泊松回归）上进行仿真，并在美国人口普查ACS收入预测任务上进行实证验证。

**📈 对比分析**

与传统非私有MLE、噪声感知估计、引导方法以及忽视隐私的合成数据做对比，结果显示噪声校正方法保持95%覆盖率，在低隐私/样本量下仍优于无校正方法，合成数据误差明显。

**⚠️ 局限性**

仅适用于正则指数族且需已知充分统计量；在高维或非正则模型下效果未知；Wald CI 在低隐私/凸链接模型可能欠覆盖，需更精细的多变量区间。

---

## 554. Agentic Code Reasoning

**arXiv ID:** 2603.01896 | [PDF](https://arxiv.org/pdf/2603.01896v1)

**作者:** Shubham Ugare `[一作]` (Meta), Satish Chandra `[通讯]` (Meta)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM代理在不执行代码的前提下进行代码推理，提出半正式推理（semi‑formal reasoning）模板，并在补丁等价验证、代码问答和错误定位三个任务上进行实验。

**💡 创新点**

创新点：①将结构化推理模板作为“证书”强制代理记录前提、执行路径和正式结论，提升推理完整性；②统一该模板应用于不同语言、不同任务；③在补丁等价验证中实现93%准确率，表明可用作RL训练的执行‑free 奖励信号。

**🔧 技术方法**

技术：LLM（Opus‑4.5、Sonnet‑4.5）与SWE‑agent 交互框架，bash 工具进行代码文件探索；对比单次调用、文件上下文、difflib相似度以及标准链式思考；采用结构化模板进行半正式推理。

**📊 数据集**

数据集：SWE‑bench‑Verified（补丁对与测试补丁）、Defects4J（错误定位）、RubberDuckBench（代码问答）以及SWE‑bench、SWE‑RM等辅助数据。

**📈 对比分析**

对比方法：单次调用、单文件上下文、difflib相似度、标准推理与半正式推理。结果显示：补丁等价验证半正式推理准确率提升至93%（标准78%）；代码问答提升至87%（标准78%）；Defects4J Top‑5 错误定位提升12个百分点（标准58%）。平均步骤虽多（半正式约30–40步）但效果显著。

**⚠️ 局限性**

限制：①推理步骤多，耗时较长；②对第三方库或隐藏语义的函数推断仍易出错；③无法获得运行时信息，导致某些细节推断不完整；④半正式模板需手工验证或后期审计；⑤不同模型对结构化模板的适应性差异，强模型效果有限。

---

## 555. SaferPath: Hierarchical Visual Navigation with Learned Guidance and Safety-Constrained Control

**arXiv ID:** 2603.01898 | [PDF](https://arxiv.org/pdf/2603.01898v1)

**作者:** Lingjie Zhang `[一作]` (Hong Kong University of Science and Technology), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SaferPath 分层视觉导航框架，将端到端视觉导航模型的引导轨迹与安全约束的 MP‑SVES 优化和 MPC 控制相结合，实现未知环境中的安全路径规划与跟踪。

**💡 创新点**

①仅将端到端模型作为粗导引，而非直接执行；②构建 Traversability Score Mapper 生成安全分数图；③提出高效的安全约束轨迹优化算法 MP‑SVES；④在狭窄、拥挤环境中显著提升成功率。

**🔧 技术方法**

端到端视觉导航模型（如 NoMaD、ViNT）、深度估计（DepthAnythingV2）、Stein 变分进化策略（MP‑SVES）、模型预测控制（MPC）、安全分数映射以及紧急指示模块。

**📊 数据集**

SACSoN/HuRoN、SCAND、GoStanford2、RECON 四个室内导航数据集，训练与评估均在未见环境下进行。

**📈 对比分析**

与 GNM、ViNT、NoMaD 等基线在未见障碍、密集无结构环境、狭窄走廊三类实验及真实世界四条轨迹进行对比，SaferPath 在成功率上提升 40%~50%，碰撞次数大幅降低，基线方法多数在狭窄走廊全部失败。

**⚠️ 局限性**

仅依赖 RGB 视角，视野受限导致对不可见障碍感知不足；需引入更丰富的传感器与预测场景理解。

---

## 556. Multi-Level Bidirectional Decoder Interaction for Uncertainty-Aware Breast Ultrasound Analysis

**arXiv ID:** 2603.01295 | [PDF](https://arxiv.org/pdf/2603.01295v1)

**作者:** Abdullah Al Shafi `[一作]` (Khulna University of Engineering and Technology), Engelbert Mephu Nguifo `[通讯]` (University Clermont Auvergne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计了一种多任务网络，利用多层解码器交互和不确定性感知自适应协调，实现乳腺超声图像的病灶分割与组织分类。

**💡 创新点**

在解码器每个尺度上实现双向分割-分类特征交互，并通过基于特征激活方差的不确定性代理注意力自适应权重，解决实例难度不一致的问题。

**🔧 技术方法**

采用多尺度上下文融合、注意力加权池化、乘性调制、激活方差代理注意力、Focal Tversky、边界与纹理正则化等技术。

**📊 数据集**

在公开乳腺超声数据集BUSI和BUSI‑WHU上进行训练与评估。

**📈 对比分析**

与CNN、Transformer及传统多任务共享编码器模型比较，取得74.5% IoU和90.6%分类准确率（BUSI）以及86.4% IoU和95%准确率（BUSI‑WHU），在多项指标上均超过基线1–5%幅度。

**⚠️ 局限性**

仅在二维单器官数据上验证，缺乏三维体积和多器官的通用性验证。

---

## 557. Zero- and Few-Shot Named-Entity Recognition: Case Study and Dataset in the Crime Domain (CrimeNER)

**arXiv ID:** 2603.02150 | [PDF](https://arxiv.org/pdf/2603.02150v1)

**作者:** Miguel Lopez-Duran `[一作]`, Alvaro Ortigosa `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了名为CrimeNERdb的犯罪相关文档NER数据集，并在该数据集上开展了零样本与少样本NER的基准实验。

**💡 创新点**

创新点包括：①首次提供超过1.5k条真实犯罪/恐怖袭击报告的双层（粗粒度与细粒度）实体标注；②将实体划分为5种粗粒度类型及22种细粒度标签；③在零样本与少样本场景下，对传统PLM（NUNER、CONTaiNER）与大型语言模型（GPT‑4o‑Mini、GPT‑4.1‑Mini、Gemini‑2.5‑Flash）进行系统评测。

**🔧 技术方法**

使用技术主要有：基于Transformer的预训练语言模型（如BERT变体）、对比学习增强的NERT模型（CONTaiNER）、以及大规模LLM在零/少样本推理框架（LLMNER）。

**📊 数据集**

数据集为CrimeNERdb，来源于美国司法部的新闻稿（2009‑2018）和全球恐怖主义数据库（2021）整理后经预处理并人工标注，涵盖5个粗粒度类别与22个细粒度类别，共计1,568份文档。

**📈 对比分析**

实验对比了零样本（NUNER、GPT‑4o‑Mini、GPT‑4.1‑Mini、Gemini‑2.5‑Flash）与少样本（CONTaiNER、上述LLM）在粗粒度与细粒度实体上的精确率、召回率和F1。结果显示，GPT‑4.1‑Mini在零样本中略优；在少样本中GPT‑4o‑Mini表现最好；Logistic类型的实体最易识别，Crime类型最难，整体F1均在中等水平（约10‑70%）。

**⚠️ 局限性**

局限性主要包括：①数据集规模相对有限，难以覆盖所有犯罪场景；②标注工作人工耗时且可能存在主观偏差；③对模型的评估主要集中在单一领域，缺乏跨语言或跨模态验证；④LLM在细粒度分类尤其是Crime类上的低精度表明仍需改进领域知识的利用和多任务学习策略。

---

## 558. Preoperative-to-intraoperative Liver Registration for Laparoscopic Surgery via Latent-Grounded Correspondence Constraints

**arXiv ID:** 2603.01720 | [PDF](https://arxiv.org/pdf/2603.01720v1)

**作者:** Ruize Cui `[一作]` (Hong Kong Polytechnic University), Jing Qin `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了一个两阶段的对应驱动柔性肝脏配准框架，利用潜在空间中的二维三维对应关系实现鲁棒的相机姿态估计和形变重建。

**💡 创新点**

① 引入潜在对齐模块构建跨模态统一潜在空间；② 设计不确定性增强的重叠关键点检测器提高对应质量；③ 通过形状约束监督（对应重投影、局部等距正则、渲染掩码对齐）解决二维三维深度歧义。

**🔧 技术方法**

PointNet++/ResNet特征提取、Cross‑Modal Latent Alignment、Uncertainty‑Enhanced Overlap Landmark Detector、EPnP+RANSAC姿态估计、基于自编码器的形变网络、Dice+对应重投影+等距正则损失、运行时优化。

**📊 数据集**

P2ILF数据集（9位患者的预手术3D肝模型和随手术的2D腹腔镜图像及对应标注），并用SAM2进行肝脏掩码分割。

**📈 对比分析**

与BHL、Grasp、NCT、UCL、VOR、LMR、Opt、ADeLiR等方法在P2ILF上对比，rigid端实现Dice 69.21%/RRE 0.90°/RTE 0.63mm，非刚性端Dice 45.52%/TRE_a 42.26px，均优于所有竞争方法，尤其在重投影误差和姿态误差上显著下降。

**⚠️ 局限性**

仅在训练集上评估，缺乏公开测试集验证；需要手工标注或伪标注来构建对应关系，推理速度受限于两阶段网络与运行时优化；在极端视角/遮挡下对应检测仍可能失效。

---

## 559. Active Flow Matching

**arXiv ID:** 2603.00877 | [PDF](https://arxiv.org/pdf/2603.00877v1)

**作者:** Yashvir S. Grewal `[一作]` (Australian National University), Edwin V. Bonilla `[通讯]` (Data61, CSIRO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 Active Flow Matching (AFM)，一种将隐式离散流模型与变分主动生成框架结合的技术，用于在高维离散空间中寻找高适应度序列。

**💡 创新点**

创新点在于将 KL 目标从不可计算的边缘分布改写为可计算的条件终点分布，允许在不需要显式似然的情况下对离散流进行梯度优化，并通过三种变体（forward‑KL、reverse‑KL、symmetric‑KL）实现探索–利用权衡。

**🔧 技术方法**

使用了离散流匹配（Discrete Flow Matching）技术、重要性采样（self‑normalized），以及三元混合提议（先验、流模型、回放缓冲）来近似目标分布；此外在实验中采用了 Transformer‑基底的编码器来实现条件终点分布的预测。

**📊 数据集**

在六个任务上评测：Ehrlich 32/64 伪合成景观、AAV 蛋白质包膜设计、FoldX 稳定性与 SASA 结构优化，以及 F2/Thrombin 分子对接（SELFIES）。

**📈 对比分析**

与 VSD、CbAS 以及 LaMBO‑2 进行对比。forward‑KL AFM 在大多数任务中实现了最快收敛和最低简单后悔，尤其在 Ehrlich、AAV、F2 对接等任务上明显优于基线；reverse‑KL 与 symmetric‑KL 在某些任务上表现逊色。

**⚠️ 局限性**

局限性包括对分类器预测 p(y≥τ|x) 的高度依赖，若分类器精度不足会导致采样偏差；重要性采样在提议分布与目标重叠不足时方差高；reverse‑KL 形式缺乏理论一致性保证，易出现过早收敛。

---

## 560. To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters

**arXiv ID:** 2603.00742 | [PDF](https://arxiv.org/pdf/2603.00742v1)

**作者:** Sara Dragutinović `[一作]` (New York University), Rajesh Ranganath `[通讯]` (New York University)

**通讯引用:** 6946 | [OpenAlex ID](https://openalex.org/A5022202456)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过理论分析和实验评估，探讨了 Muon 优化器的偏置及其对学习轨迹和最终模型性能的影响，尤其指出 Muon 缺失了梯度下降的“简易性偏置”，导致模型更易记忆噪声、难以捕捉共享结构；

**💡 创新点**

创新点在于首次将 Muon 的正交化更新机制与梯度下降的简易性偏置联系起来，给出了深度线性网络下 Spectral GD 的动力学解析，并通过路由任务与含伪相关的 MNIST 实验验证了 Muon 在共享表示和泛化方面的劣势；

**🔧 技术方法**

主要技术包括 Spectral Gradient Descent（精确 SVD 正交化、无动量）、深度线性网络理论、梯度流解析、以及对 Muon 的 Newton‑Schulz 近似；

**📊 数据集**

实验使用的主要数据集包括：1）高斯合成数据（用于验证深度线性网络理论）；2）自定义路由任务（7个输入/输出域的线性编码器/解码器）；3）MNIST 图像（加入特定像素的伪相关特征）；

**📈 对比分析**

与 SGD（以及 Adam）对比，Muon's 训练速度更快，但在路由任务中未能学习共享表示，泛化到未见输入‑输出对时性能大幅下降；在 MNIST 伪相关任务中，Muon's/Adam 的验证准确率受伪特征强度影响更快，SGD 在早期停机时仍保持更好的非伪相关性能；

**⚠️ 局限性**

局限性包括：1）理论仅适用于深度线性网络，未涵盖非线性激活与真实网络；2）Spectral GD 为简化模型，真实 Muon 采用近似正交化，理论与实践可能存在差距；3）实验规模有限，未覆盖更复杂任务或大模型；4）未对 Muon 的参数调优空间进行系统探究。

---

## 561. From Human Negotiation to Agent Negotiation: Personal Mobility Agents in Automated Traffic

**arXiv ID:** 2603.01035 | [PDF](https://arxiv.org/pdf/2603.01035v1)

**作者:** Pascal Jansen `[一作]` `[通讯]` (Ulm University), Pascal Jansen (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

提出个人移动代理（Personal Mobility Agent）作为用户代理，代理在自动驾驶环境中主动与其他自动化或人类行为体协商，代替用户在每一次交通决策中直接参与，进而实现高层次偏好与策略的委托与监督；

**💡 创新点**

创新点在于把传统一对一的用户-车辆交互升级为代理代理的多主体协商框架，解决了在多元化、自动化日益加剧的交通场景中用户与系统偏好冲突的可扩展性问题，并把人机交互焦点从即时控制转移到偏好表达与委托管理；

**🔧 技术方法**

利用现有的自动驾驶个性化学习技术（如贝叶斯优化、强化学习与人类反馈）、大语言模型与视觉语言模型实现代理的偏好建模与对话交互，配合安全约束下的多代理协商协议；

**📊 数据集**

无公开实验数据集，论文为位置/概念性阐述；

**📈 对比分析**

论文未提供实验或性能评估，主要通过理论分析与案例描述说明该框架的可行性；

**⚠️ 局限性**

局限包括：安全与法律约束下代理可能无法完全满足用户偏好；缺乏真实世界试验与用户调研支持；代理间协商机制在混合交通中对人类行为推断的鲁棒性不足；长期委托可能导致用户意识淡化与责任归属模糊，需进一步研究透明度与可解释性。

---

## 562. Learning from Synthetic Data Improves Multi-hop Reasoning

**arXiv ID:** 2603.02091 | [PDF](https://arxiv.org/pdf/2603.02091v1)

**作者:** Anmol Kabra `[一作]` (Cornell University), Kilian Q. Weinberger `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对大语言模型进行强化学习微调，使用规则生成的合成数据训练其知识组合能力，从而提升其在真实多跳问答任务中的表现。

**💡 创新点**

创新点在于证明即使训练数据与真实世界完全无事实重叠，规则生成的合成数据也能让模型学习到可迁移的知识组合技能，突破了传统需要大量人工标注或 LLM 生成数据的瓶颈。

**🔧 技术方法**

方法采用了基于验证奖励的强化学习（RLVR），使用 Group Relative Policy Optimization（GRPO）对 Qwen3、Qwen2.5 和 Phi-4-mini 等 LLM 进行微调。

**📊 数据集**

使用的合成数据集包括基于模板的 Family-Relationships、Knights-Knaves、GSM-Infinite、以及基于逻辑程序的 Qwen3 训练集，并在 2*WikiMultihopQA、2*HotpotQA、2*2WikiRetQA、CounterfactualQA 和 2*WikiComplexQA 等真实多跳基准上进行评估。

**📈 对比分析**

与仅使用监督微调或无合成数据的基线相比，RL 微调在所有真实基准上提升了 56%–131% 的 F1 分数，且提升随合成数据量线性增长，无明显过拟合。

**⚠️ 局限性**

局限性包括对大型模型的可扩展性尚未完全验证，合成数据虽易生成但在语义丰富度上仍低于真实语言，且对非多跳推理或其他推理范式的迁移效果仍需进一步探究。

---

## 563. Coarse-to-Fine Monocular Re-Localization in OpenStreetMap via Semantic Alignment

**arXiv ID:** 2603.01613 | [PDF](https://arxiv.org/pdf/2603.01613v1)

**作者:** Yuchen Zou `[一作]` (Xi'an Jiaotong University), Yuqing Tang `[通讯]` (International Digital Economy Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于 DINO‑ViT 的语义对齐与层次化匹配框架，能够从单张 monocular 图像高效定位到 OpenStreetMap（OSM）地图上的 3-DOF 位置与朝向。

**💡 创新点**

创新点在于：①利用 DINO‑ViT 对街景图像进行语义分解，并将其投影到 BEV 空间；②将 OSM 向量图转化为连续的神经语义图；③采用从粗到细的层次化匹配策略，既提升定位精度又显著降低计算成本。

**🔧 技术方法**

核心技术包括：DINO‑ViT（自监督视觉 Transformer）进行语义特征提取；BEV 投影与 Polar‑to‑Cartesian 转换；U‑Net 生成语义神经图；基于 Fourier 相关的全局粗匹配与不确定性驱动的细化搜索。

**📊 数据集**

使用 Mapillary（MGL）与 KITTI 两大数据集进行训练与评测，MGL 覆盖 12 个城市约 760k 张街景图像，KITTI 作为驾驶场景测试集。

**📈 对比分析**

与 Retrieval、Refinement、OrienterNet 等基准方法对比，在 MGL 上实现了 1m/1° 召回率 15.78%→29.54%（≈1.9×提升），在 KITTI 上实现了 3m/3° 召回率 92.84%→96.37%（≈4%提升），并将帧率提升三倍以上。

**⚠️ 局限性**

局限性包括：对高质量语义特征依赖较大，若图像中出现大量遮挡或极端光照会影响 DINO 分解；层次化匹配仍需先验定位范围，过大范围时粗匹配仍可能产生误匹配；目前仅在 3‑DOF 任务验证，未扩展至 6‑DOF 或动态场景。

---

## 564. 3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs

**arXiv ID:** 2603.01376 | [PDF](https://arxiv.org/pdf/2603.01376v1)

**作者:** Mehdi Makni `[一作]` (Massachusetts Institute of Technology), Rahul Mazumder `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3139 | [OpenAlex ID](https://openalex.org/A5045271820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于3-Block ADMM的Sparse+Low‑Rank分解方法，并引入Transformer‑Matching细化步骤，可在一次压缩后实现对LLM权重的高质量稀疏与低秩分解。

**💡 创新点**

创新点包括：①统一的3-Block ADMM框架实现稀疏与低秩的联合优化并给出收敛保证；②Transformer‑Matching可对任意(S+LR)分解进行全网络级联合微调；③通过以上两项实现压缩速度提升2-3倍、perplexity降低30%以上。

**🔧 技术方法**

采用3-Block ADMM、随机SVD、闭式低秩更新、梯度下降的Transformer‑Matching、LoRA初始化以及N:M稀疏/OWl等稀疏分配策略。

**📊 数据集**

使用C4数据集做校准（128条样本，2048 token），评估基准为WikiText2、Penn Treebank、C4 perplexity；零样本任务为PIQA、ARC‑Easy/Challenge、HellaSwag、Winogrande、RTE、OpenbookQA、BoolQ。

**📈 对比分析**

与OATS、HASSLE‑free‑SparseGPT、HASSLE‑free‑ALPS等基线比较；在Llama‑3、Llama‑3.2、OPT‑30B等模型上，3‑Block ADMM+TM在相同压缩比下perplexity降低30%+、压缩速度比SOTA快2‑3倍，LoRA fine‑tuning后仍保持性能优势。

**⚠️ 局限性**

局限性：仍需改进层级稀疏/低秩分配策略；目前仅针对一次压缩，未覆盖完整finetune流程；对量化或特殊CUDA kernel 的适配需进一步研究；高维矩阵仍以O(N^3)计算为瓶颈。

---

## 565. Hide&Seek: Remove Image Watermarks with Negligible Cost via Pixel-wise Reconstruction

**arXiv ID:** 2603.01067 | [PDF](https://arxiv.org/pdf/2603.01067v1)

**作者:** Huajie Chen `[一作]`, Wanlei Zhou `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为 HIDE&SEEK 的黑盒水印去除攻击方法，能够在不暴露水印检测器信息的情况下，利用像素级重建有效消除机器生成图像中的水印。

**💡 创新点**

创新点在于将攻击分为 HIDE 与 SEEK 两个阶段，首先通过可微掩码模型定位对水印影响最大的“脆弱像素”，随后采用自回归像素生成器按逆序重建这些像素，从而在仅修改极少像素的前提下产生最大频域与语义域的差异，既提升了攻击效果，又保持了视觉质量。

**🔧 技术方法**

使用了可微软掩码模型（Masking Model）、基于 MAE 的掩码重建器、FGN（Fractal Generative Network）自回归像素生成器，以及频域损失、语义损失（CLIP）、感知损失（AlexNet）等多种训练损失。

**📊 数据集**

训练数据集主要采用公开的 ImageNet 等真实图像数据；攻击实验中使用多种公开水印方法（HiDDeN、StableSignature、StegaStamp、TRW 等）生成的水印图像进行评估。

**📈 对比分析**

与 Diffusion Attack、VAE Attack、UnMarker 等 SOTA 去水印方法相比，HSN 在检测率下降上更为显著，HS+ 在保持 PSNR>30、SSIM>0.85 的同时也能显著降低检测率；两种方法的计算成本均低于 UnMarker，VAE Attack 更快，但在攻击效果上略逊。

**⚠️ 局限性**

局限性包括：在跨域数据（如动漫图像）上的效果不佳；需要先训练掩码模型和生成器，若面对完全未知的水印方案效果不确定；且攻击仍需在修改像素与视觉质量之间权衡，未来需研发更鲁棒的水印方案。

---

## 566. Accelerating Single-Pass SGD for Generalized Linear Prediction

**arXiv ID:** 2603.01951 | [PDF](https://arxiv.org/pdf/2603.01951v1)

**作者:** Qian Chen `[一作]` (Peking University), Cong Fang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在流式数据下的广义线性预测问题，提出一种在内外循环同时使用动量的自适应近端方法（SADA）并给出单梯度更新的收敛分析。

**💡 创新点**

首创通过数据依赖近端项实现双重动量加速，克服传统方差削减方法对问题条件数的高度依赖，并引入层剥离分解技术解析误判与内循环收敛。

**🔧 技术方法**

结合动量加速、数据驱动近端正则、尾部平均、两阶段步长调度、层剥离协方差分解以及非渐进的高阶时序分析。

**📊 数据集**

理论分析基于一般随机分布（如高斯或子高斯设计）及其四阶矩假设，文中未给出具体实验数据集。

**📈 对比分析**

与传统方差削减方法（如VR、SVRG）相比，SADA 的优化项从 O(α²κ) 降为 O(α²κ̃)，统计项保持最优，误判项随样本增大消失；在最坏情况可实现 √(κ(Σ)) 的加速。

**⚠️ 局限性**

仍需预知问题相关参数（α,κ,κ̃），对非凸问题的推广有限；在高精度 regime 下需更小步长，理论上误判项仍占优先级。

---

## 567. From Sustainable Materials to User-Centered Sustainability: Material Experience in Art Healing

**arXiv ID:** 2603.01377 | [PDF](https://arxiv.org/pdf/2603.01377v1)

**作者:** Yuxin Zhang `[一作]` (Academy of Arts and Design Tsinghua University), Chao Zhao `[通讯]` (Academy of Arts and Design Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究通过将可持续填料加入水凝胶基体，制造了28种可持续材料，并通过多感官体验评估其艺术疗愈效果。

**💡 创新点**

创新点在于将“材料体验”框架与可持续材料结合，系统阐释了美学、内在与物理属性对艺术疗愈的影响，并提出以用户体验为中心的可持续发展路径。

**🔧 技术方法**

使用了混合方法：探索性因子分析、结构方程模型（SEM）、层次聚类与t检验等统计技术，配合语义差分量表收集数据。

**📊 数据集**

数据集为10名设计/材料专业参与者对28种材料的28次评估，涵盖视觉、触觉、嗅觉的15个评估维度。

**📈 对比分析**

通过SEM的路径系数和拟合指标（χ²/df、CFI、RMSEA等）验证模型，结果显示“美学”属性对艺术疗愈影响最大（路径系数0.706），整体模型拟合优良，说明该框架能有效捕捉材料体验对疗愈效果的作用。

**⚠️ 局限性**

局限性包括样本量仅10人且专业背景较高，水凝胶在室温下热稳定性差，实验时间受限，且结果的普适性需要在更大、非专业人群中验证。

---

## 568. BornoViT: A Novel Efficient Vision Transformer for Bengali Handwritten Basic Characters Classification

**arXiv ID:** 2603.00755 | [PDF](https://arxiv.org/pdf/2603.00755v1)

**作者:** Rafi Hassan Chowdhury `[一作]` (Islamic University of Technology), Kaniz Fatiha `[通讯]` (Islamic University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了BornoViT，一个轻量化视觉Transformer模型，用于识别孟加拉语手写基本字符和数字。

**💡 创新点**

通过简化的Patch‑Embedding和仅四层Transformer块，显著降低参数（65万）和计算量（0.16 GFLOPs），同时保持高精度。

**🔧 技术方法**

采用Vision Transformer架构、数据增强（随机仿射、颜色抖动）、迁移学习（在Ekush上预训练）以及k‑fold交叉验证。

**📊 数据集**

在BanglaLekha‑Isolated、Ekush以及自采集的Bornomala（222人、13318张图像）三个数据集上进行训练与评估。

**📈 对比分析**

与MobileNetV2、EfficientViT、DenseNet、Xception、VashaNet等现有模型进行对比，BornoViT在BanglaLekha‑Isolated上达到95.77%准确率、0.65M参数、0.62MB模型，显著优于其他轻量模型。

**⚠️ 局限性**

模型仍受类间相似导致误判、对写字风格的鲁棒性有限，且仅在Ekush预训练，可能在更大或更复杂字符集上表现下降。

---

## 569. Unifying Heterogeneous Multi-Modal Remote Sensing Detection Via Language-Pivoted Pretraining

**arXiv ID:** 2603.01758 | [PDF](https://arxiv.org/pdf/2603.01758v1)

**作者:** Yuxuan Li `[一作]` (Nankai University), Jian Yang `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于语言中心的预训练框架BabelRS，旨在统一多传感器遥感目标检测任务，解决传统晚期对齐导致的梯度冲突与训练不稳定问题。

**💡 创新点**

创新点在于：①将不同遥感模态的特征先映射到共享的语言语义空间（Concept-Shared Instruction Aligning），实现无需空间对齐的跨模态对齐；②引入Layerwise Visual‑Semantic Annealing逐步融合多尺度视觉特征与语言语义，克服语义粒度与检测需求不匹配的难题；③通过在预训练阶段完成对齐，显著提升训练稳定性与跨模态泛化能力。

**🔧 技术方法**

核心技术包括：基于大语言模型（Qwen2）进行指令跟随式语言对齐；使用ViT‑Large视觉编码器并在不同层级进行逐步融合；使用自回归语言建模损失与多模态指令‑回答对齐；在下游进行简化的联合检测微调。

**📊 数据集**

预训练使用多源视觉‑语言数据集：Million‑AID、LevirCC、VHM、RSVQA、FIT_RS、GAIA、SARLang、MMRS‑1M、GeoChat、DIOR‑RSVG、VRSBench、Mini‑InternVL等；微调在SOI‑Det基准上，包含SARDet‑100K、DOTA‑v1.0和DroneVehicle（红外）三种模态。

**📈 对比分析**

与现有方法（SM3Det、UniDet、Uncertainty、DINOv2、CLIP等）在SOI‑Det上直接比较，BabelRS在AP@50、mAP与提出的H‑mAP上均优于所有对手，尤其在SAR和红外模态上提升显著，表明跨模态对齐与多尺度融合的有效性。

**⚠️ 局限性**

局限性包括：①仍需大量预训练资源和多模态文本标注，成本高；②对语言表述的依赖意味着对缺乏语义描述或语言多样性不强的模态（如高光谱、雷达回波深度图）可能效果有限；③仅在遥感目标检测任务上验证，尚未探索到其他遥感应用。

---

## 570. CollabEval: Enhancing LLM-as-a-Judge via Multi-Agent Collaboration

**arXiv ID:** 2603.00993 | [PDF](https://arxiv.org/pdf/2603.00993v1)

**作者:** Yiyue Qian `[一作]` (Amazon AWS Generative AI Innovation Center), Yi Zhang `[通讯]` (Amazon AWS Bedrock)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CollabEval，一种多代理协作评估框架，用于对 AI 生成内容进行多维度评估。

**💡 创新点**

创新点在于三阶段协作流程：初评 → 多轮协商 → 最终判决，并通过早期共识检查实现高效、鲁棒的评估；强调协作而非竞争。

**🔧 技术方法**

使用多大语言模型（Mistral Large、Claude Haiku、Claude Sonnet、Llama‑3 70B）作为评估代理，采用基于对话的迭代讨论和最终判决模型（Claude Sonnet 3.5）。

**📊 数据集**

使用 SummEval（内容评估）和两个对比式数据集（chatbot_arena_conversation、lmsys_arena_human_preference_55k）进行实验。

**📈 对比分析**

与单一 LLM 评估和“Round‑Table”多代理对比，CollabEval 在所有维度均获得最高准确率、最佳误差分布，并在对比式评估中表现出更高的准确率与更平衡的误判率；平均讨论轮数保持在 1–3 轮，计算成本相对可控。

**⚠️ 局限性**

局限性包括需要多轮讨论导致计算开销略增、依赖预先选定的模型组合、在极端偏差的单一模型上仍需进一步缓冲，并未探讨更大规模或跨语言评估场景。

---

## 571. KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging

**arXiv ID:** 2603.00907 | [PDF](https://arxiv.org/pdf/2603.00907v1)

**作者:** Lianjun Liu `[一作]` (Hainan University), Yunshan Zhong `[通讯]` (Hainan University)

**通讯引用:** 332 | [OpenAlex ID](https://openalex.org/A5101191231)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了KVSlimmer，针对大型语言模型KV缓存进行理论分析与高效压缩；

**💡 创新点**

通过谱能量分布揭示Q/K与V的异构性，构造精确Hessian并推导无梯度闭式合并公式；

**🔧 技术方法**

使用谱分析、SVD、第二阶Taylor展开、Moore–Penrose伪逆等数学工具实现无梯度KV合并；

**📊 数据集**

在Llama3.1-8B、Mistral-7B、Qwen2-1.5B模型上，对LongBench与LongBenchV2数据集进行实验；

**📈 对比分析**

与多种基线（StreamingLLM、LongCache、H_2O、CaM、AsymKV）对比，KVSlimmer在大多数任务上提升0.4-0.9分，同时将内存占用降低约30%，推理延迟降低约28%；

**⚠️ 局限性**

仅局限于局部相邻token合并，压缩比例在各层统一；未探索全局或自适应层级压缩策略，需进一步研究。

---

## 572. Analyzing and Improving Fast Sampling of Text-to-Image Diffusion Models

**arXiv ID:** 2603.00763 | [PDF](https://arxiv.org/pdf/2603.00763v1)

**作者:** Zhenyu Zhou `[一作]` (Zhejiang University), Can Wang `[通讯]` (Zhejiang University)

**通讯引用:** 11776 | [OpenAlex ID](https://openalex.org/A5100428567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在文本到图像扩散模型中实现了训练无关的采样加速，通过统一视角分析加速方法的设计空间，并提出了常数总旋转调度策略。

**💡 创新点**

创新点在于揭示时间调度是最关键因素，并利用弗雷内-塞雷公式的曲率与扭率构造常数总旋转调度，实现10步即可逼近50步的质量。

**🔧 技术方法**

技术包括ODE求解器、时间调度、特征缓存、Frenet-Serret几何分析、Fast DPM-Solver/UniPC、GITS、FORA/TaylorSeers等。

**📊 数据集**

主要使用MS-COCO、DrawBench、PIE-Bench等公开数据集，且在Flux.1-Dev和Stable Diffusion 3.5等模型上评估。

**📈 对比分析**

与多种现有加速方法（GITS、DPM-Solver、UniPC、FORA、TaylorSeers、TPDM等）对比，10步采样下在IR/CS/AS/HPSv2等指标上显著优于统一调度，并接近50步基线。

**⚠️ 局限性**

局限在于需要预先计算几何统计，且在某些模型或特征缓存策略上效果不显著；对极低步数或不同训练方法的鲁棒性仍待验证。

---

## 573. Beyond the Resumé: A Rubric-Aware Automatic Interview System for Information Elicitation

**arXiv ID:** 2603.01775 | [PDF](https://arxiv.org/pdf/2603.01775v1)

**作者:** Harry Stuart `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Timothy Baldwin `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的多轮面试系统，利用“判定者”（Judge）跟踪并更新对候选人能力维度的概率信念，并利用“访谈者”（Interviewer）逐步获取信息以实现信念收敛，最终生成信息丰富的面试记录和可审计的信念日志。

**💡 创新点**

创新点包括：① 将面试过程建模为对 Rubric 维度的概率信念更新，并通过总变差（TV）量化收敛；② 设计元同构测试（Metamorphic Testing）来校准 LLM 的信念更新行为；③ 通过模拟面试验证系统在 76.1% 的情况下能恢复候选人的潜在能力类型；④ 公开实现与数据，提供可复现的评估框架。

**🔧 技术方法**

核心技术：使用 GPT‑5 作为 LLM 基础模型，Chainlit 框架实现 web UI；信念更新采用 Bayesian 推断与全变差距离；面试策略包含四种访谈者策略（Belief Aware/Unaware、Shallow Unaware 等）；模拟生成候选人消息与面试轮次；对照实验使用基准面试策略。

**📊 数据集**

数据集：从 r/Resumes 子版块收集 30 篇匿名简历（每个领域 10 篇），手工构造 2–5 轮面试；结合三类 Rubric（图形设计、销售、机器学习工程），每类 3 维度，3 个等级；再合成 180 个候选人配置（10 简历 × 6 架构）。

**📈 对比分析**

比较方法：与仅基于简历的先验（t=0）和不同访谈者策略进行对比；使用总变差减小量和终点 MAP 归一化预测（nearest‑neighbor）来衡量信念收敛质量；结果显示：① 信念更新幅度从 0.0621 降至 0.0205，下降约 3 倍；② 架构恢复率从 16.7% 提升至 76.1%；附录中与 Gemini‑3.1 Pro 亦表现优秀。

**⚠️ 局限性**

局限性：① 模拟面试与真实人类候选人存在差距，可能忽略沟通风格、诚实度等变量；② 简历数据仅来自公开匿名样本，规模有限；③ 依赖 LLM 可能带来偏见与对受保护属性的潜在歧视；④ 需要人类审核，系统不做最终决策；⑤ 需要安全防护与合规性审查。

---

## 574. Compact Task-Aligned Imitation Learning for Laboratory Automation

**arXiv ID:** 2603.01110 | [PDF](https://arxiv.org/pdf/2603.01110v1)

**作者:** Kanata Suzuki `[一作]` (Fujitsu Limited), Tetsuya Ogata `[通讯]` (Waseda University)

**通讯引用:** 6865 | [OpenAlex ID](https://openalex.org/A5055922202)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 TVF-DiT，一种利用小型视觉基础模型（DINOv3）、视觉‑语言模型（SigLIP2）以及扩散变换器（Diffusion Transformer）实现的实验室自动化仿真学习框架。

**💡 创新点**

创新点在于：① 将自监督视觉特征与视觉‑语言特征通过轻量级适配器对齐，实现对任务提示的精细语义控制；② 将对齐后的特征作为跨注意力键值喂给扩散变换器，实现低参数（<500 M）下高质量动作生成；③ 通过细粒度任务提示显著提升跨模态对齐与成功率。

**🔧 技术方法**

使用技术包括：自监督视觉基础模型 DINOv3、视觉‑语言模型 SigLIP2、适配器（Projection + GatedRMS + Transformer Decoder）、扩散变换器（DiT）以及条件流匹配（Conditional Flow Matching）进行离线仿真学习。

**📊 数据集**

数据集：自行收集的遥控演示数据，共约 8 小时，包含 500 条清洗、400 条排列、400 条粉末转移演示，使用 224×224 RGB 图像和 14‑维关节角度动作。

**📈 对比分析**

与基线方法（单一 VLM、视觉基础+LLM）以及纯回放相比，TVF-DiT 在三项实验室任务的平均成功率达到 86.6%，显著优于 20%（VLM）和 36.6%（视觉+LLM）。进一步实验表明，使用详细任务提示可将成功率从 53.3% 提升到 86.6%。

**⚠️ 局限性**

局限性包括：① 仅使用最小版本 DINOv3，较大模型可能进一步提升几何表征精度；② 任务范围有限，仅覆盖三种试管操作；③ 对演示数据量依赖较高，需更高效的数据采集或奖励反馈机制；④ 仍未实现完整实验室流程自动化，缺乏导航等模块。

---

## 575. riMESA: Consensus ADMM for Real-World Collaborative SLAM

**arXiv ID:** 2603.01178 | [PDF](https://arxiv.org/pdf/2603.01178v1)

**作者:** Daniel McGann `[一作]`, Michael Kaess `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为riMESA的鲁棒增量分布式协同SLAM后端，能够在受限通信、噪声和离群测量条件下实时生成多机器人团队的高质量状态估计。

**💡 创新点**

1) 将Consensus ADMM（C-ADMM）框架应用于非凸、在流形上的C‑SLAM问题；2) 引入“鲁棒加权偏置先验”（RWBP）和双变量衰减机制以在增量式和离群环境中保持数值稳定；3) 设计了两阶段异步通信处理器，支持延迟和两将军失败；4) 结合鲁棒增量优化器riSAM实现全局一致性与实时性。

**🔧 技术方法**

C-ADMM分布式优化、在流形上的最小化（Geodesic误差）、M‑估计（SIG核）、鲁棒权重偏置先验、增量式Smoothing & Mapping（riSAM）和双变量衰减。

**📊 数据集**

合成数据（6机器人、1000姿态、不同测量类型和噪声水平），以及真实世界数据集COSMO‑Bench（24个LiDAR SLAM场景）和Nebula数据集。

**📈 对比分析**

与DLGBP、DDF‑SAM2、iMESA、kiMESA、Centralized Oracle、Centralized GNC、Centralized PCM以及独立iSAM2基线比较；riMESA在大多数情形下实现了7倍以上的误差改进、在真实数据中比DLGBP好≈8×、比DDF‑SAM2好≈18×，并且保持实时性。

**⚠️ 局限性**

在高噪声、低信噪比或通信质量极差的场景下仍易受限；鲁棒化和权重衰减虽提升稳定性，但仍需更频繁的通信；对全局最优性的理论保证尚未得到证明。

---

## 576. VP-Hype: A Hybrid Mamba-Transformer Framework with Visual-Textual Prompting for Hyperspectral Image Classification

**arXiv ID:** 2603.01174 | [PDF](https://arxiv.org/pdf/2603.01174v1)

**作者:** Abdellah Zakaria Sellam `[一作]` (Institute of Applied Sciences and Intelligent Systems), Abdenour Hadid `[通讯]` (Sorbonne University Abu Dhabi)

**通讯引用:** 19556 | [OpenAlex ID](https://openalex.org/A5013928164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种混合Mamba-Transformer架构VP-Hype，并结合视觉与文本提示进行高效的高光谱图像分类

**💡 创新点**

创新点在于将线性时间复杂度的状态空间模型与窗口注意力交替融合，构建分层混合骨干；同时引入双模态（视觉+文本）提示与跨注意力融合TCSP，实现少样本下的任务自适应

**🔧 技术方法**

使用3D-CNN特征提取、Mamba状态空间模型、窗口自注意力、CLIP文本嵌入、可学习视觉提示、TCSP交叉注意力及多级提示注入

**📊 数据集**

在Salinas、WHU‑Hi‑LongKou、WHU‑Hi‑HongHu（Longkou、HongHu）以及QUH‑Qingyun等公开高光谱数据集上进行实验

**📈 对比分析**

与九个主流基线（LoLA、HybridSN、ViT、MASSFormer等）比较，VP‑Hype在10%/2%标签稀缺条件下分别实现OA≈99.99%、99.95%、99.64%，在2%训练时仍保持99%以上准确率，明显优于所有对比方法

**⚠️ 局限性**

局限包括：提示文本固定难以覆盖所有获取条件；模型仍较大，对实时部署有待优化；仅聚焦分类任务，未扩展到目标检测、异常检测或光谱解混；假设已知类别，难以处理开放集场景

---

## 577. Rate-Distortion Signatures of Generalization and Information Trade-offs

**arXiv ID:** 2603.01568 | [PDF](https://arxiv.org/pdf/2603.01568v1)

**作者:** Leyla Roksan Caglar `[一作]` (Mount Sinai), Baihan Lin `[通讯]` (Mount Sinai)

**通讯引用:** 881 | [OpenAlex ID](https://openalex.org/A5018612055)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并评估了一套基于信息理论的行为率失真框架，用以对比人类与多种深度视觉模型在受扰动图像上的泛化性能。

**💡 创新点**

将刺激-响应混淆矩阵映射为有效通信通道，并从中提取可解释的几何签名（斜率β与曲率κ），形成统一的“率失真签名”，以量化系统在鲁棒性与准确性之间的权衡，而非单纯依赖准确率。

**🔧 技术方法**

采用率失真理论（RDT）与Blahut–Arimoto算法估计行为通道的最优前沿，推断成本矩阵并计算信息率与失真，再统计斜率与曲率，并用AUC衡量整体效率。

**📊 数据集**

使用Geirhos等人提出的16类ImageNet衍生分类任务，包含十二种控制扰动（颜色、对比、噪声、旋转等），以及对应的人工实验数据和多模型预测结果。

**📈 对比分析**

通过在同一扰动条件下计算每个系统的β、κ和AUC进行比较；结果显示人类的曲率更低、斜率更平缓，鲁棒训练能在某些维度逼近人类但在其他维度偏离，说明仅靠准确率无法揭示系统的泛化几何。

**⚠️ 局限性**

仅关注行为层面的信息率，未考虑内部表征；扰动集和任务有限，未评估对成本矩阵、平滑参数等设定的敏感性；未覆盖对抗性训练等最新鲁棒方法，导致结果的普适性受到限制。

---

## 578. EraseAnything++: Enabling Concept Erasure in Rectified Flow Transformers Leveraging Multi-Object Optimization

**arXiv ID:** 2603.00978 | [PDF](https://arxiv.org/pdf/2603.00978v1)

**作者:** Zhaoxin Fan `[一作]` (Beihang University), Wenjun Wu `[通讯]` (Beihang University)

**通讯引用:** 9053 | [OpenAlex ID](https://openalex.org/A5060858375)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 EraseAnything++，一种针对现代 Flow‑Matching Transformer 生成模型（包括文本到图像与文本到视频）的概念消除框架，能够在不显著降低生成质量与多样性的前提下，将指定概念从模型中“遗忘”。

**💡 创新点**

创新点包括：
- 将概念消除视为受约束的多目标优化（MOO）问题，利用隐式梯度手术（implicit gradient surgery）在保持模型实用性的同时最大化消除效果；
- 结合 LoRA 参数调优与注意力正则化，实现对 Transformer 里关键视觉表征的精准抑制；
- 引入逆向自对比（Reverse Self‑Contrastive）损失，增强对同义词与不相关概念的区分；
- 针对视频模型设计 Anchor‑and‑Propagate 机制，解决时序漂移与概念回弹问题；
- 通过统一框架在图像与视频两大任务上均取得 SOTA 性能。

**🔧 技术方法**

主要技术手段包括：
- Flow‑Matching 目标与 3D Transformer 结构；
- LoRA 低秩参数适配；
- 关注层正则化与自对比损失；
- 隐式梯度手术（基于 λ 的自适应梯度投影）；
- Anchor‑and‑Propagate 的时序一致性约束；
- 多目标优化理论与解析解。
- 采用 T5 文本编码器的句子级嵌入，并用 LLM 生成不相关概念。

**📊 数据集**

实验使用的数据集与模型：
- Flux.1‑dev（文本到图像）
- Open‑Sora‑v2（文本到视频）
- Inappropriate Image Prompt (I2P) 与 NudeNet 评估 NSFW 内容；
- MS‑COCO 10K 评估图像质量（FID、CLIP）;
- 200‑artist 数据集评估艺术风格消除；
- Ring‑A‑Bell、Gen、VBench、ImageNet 类别用于视频/图像概念消除；
- 公开的攻击数据集（ReFlux、Ring‑A‑Bell、UnlearnDiffAtk 等）评估鲁棒性。

**📈 对比分析**

与现有方法（ESD、UCE、MACE、EAP、SAFREE、VideoErasure、T2VUnlearning 等）进行对比，EraseAnything++ 在
- 词汇/实体、艺术风格、关系概念的消除准确率（Acc_e）下降最快；
- 非目标概念保留率（Acc_ir）保持最高；
- FID/CLIP 分数几乎不变或略优；
- NSFW 检测率显著下降（Nudity Rate 降至 17‑18%），且视频时序一致性（Object/Subject Consistency）接近原始模型；
- 在攻击鲁棒性评测中获得最低的攻击成功率；
- 在人类评估中获得最高的整体满意度分数。综合来看，EraseAnything++ 在消除效果、保留质量与安全性方面均达到了或超过现有方法的 SOTA 水平。

**⚠️ 局限性**

局限性与未来工作：
- 目前仅针对 Flow‑Matching Transformer 结构，未验证在传统 U‑Net 或 DDPM 模型中的适用性；
- 对极大规模模型的计算开销仍显高，尤其在多目标优化和 3D 关注层正则化上；
- 依赖于 T5 文本编码器的句子级表示，若更换文本编码器需重新设计同义词生成与对比策略；
- 对于极为细粒度或语义复杂的概念（如抽象情感、隐喻），消除效果尚需进一步验证；
- 随着模型不断迭代，概念的内部表征可能变化，需持续更新同义词/负样本池。

---

## 579. AnnoABSA: A Web-Based Annotation Tool for Aspect-Based Sentiment Analysis with Retrieval-Augmented Suggestions

**arXiv ID:** 2603.01773 | [PDF](https://arxiv.org/pdf/2603.01773v1)

**作者:** Nils Constantin Hellwig `[一作]` (University of Regensburg), Christian Wolff `[通讯]` (University of Regensburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了AnnovABSA注释工具，支持所有ABSA子任务并加入检索增强式LLM建议；

**💡 创新点**

首次提供完整支持所有ABSA子任务的开源工具，动态检索增强提示的RAG机制以及严格验证与动态列表功能；

**🔧 技术方法**

前端采用React.js+TypeScript，后端使用FastAPI+Python，LLM提示采用Gemma‑3‑27B与结构化输出，检索使用BM25，工具通过Docker/CLI可高度自定义；

**📊 数据集**

使用SemEval 2016餐厅评论、Coursera学习课程、FlightABSA航空公司及酒店评论等四个领域数据集；

**📈 对比分析**

与随机few‑shot采样对比，RAG在ACD、TASD、ASQP任务上平均提升约6%微平均F1；用户研究表明AI建议可使标注时间缩短约30.5%；

**⚠️ 局限性**

实验仅使用Gemma‑3‑27B，未评估更大模型或更多few‑shot示例；仅在小规模数据集上验证；缺乏对长文本及大规模真实场景的泛化评估；未系统评估标注质量与工作负担变化。

---

## 580. Take the Power Back: Screen-Based Personal Moderation Against Hate Speech on Instagram

**arXiv ID:** 2603.01187 | [PDF](https://arxiv.org/pdf/2603.01187v1)

**作者:** Anna Ricarda Luther `[一作]` (Institute for Information Management Bremen GmbH), Andreas Breiter `[通讯]` (Institute for Information Management Bremen GmbH)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 Instagram 上针对仇恨言论的活跃用户进行三波 Delphi 调研，探索他们认为哪些屏幕需要个人审查以及各屏幕所需的具体功能。

**💡 创新点**

首次将“屏幕”作为个人审查细粒度的概念，并提供基于用户优先级的功能排名，为设计更具针对性的个人审查工具奠定经验基础。

**🔧 技术方法**

采用 Delphi 方法结合定量 Likert 量表、功能排序与定性自由文本，随后使用主题分析提炼新功能建议。

**📊 数据集**

对 40 名在德国参与社会运动且曾遭受仇恨言论的活跃用户进行三轮在线问卷，得到屏幕重要性与功能需求数据。

**📈 对比分析**

通过对各屏幕功能的平均重要性评分和最终排名进行比较，未涉及算法性能；结果显示对话屏幕功能优先级最高，算法屏幕则偏好自动化或基于内容的过滤。

**⚠️ 局限性**

研究局限在样本仅为德国进步派激进分子、只聚焦 Instagram 屏幕、问卷顺序固定可能产生偏差，且未在真实平台上验证功能有效性。

---

## 581. NERFIFY: A Multi-Agent Framework for Turning NeRF Papers into Code

**arXiv ID:** 2603.00805 | [PDF](https://arxiv.org/pdf/2603.00805v1)

**作者:** Seemandhar Jain `[一作]` (University of California), Manmohan Chandraker `[通讯]` (University of California)

**通讯引用:** 9820 | [OpenAlex ID](https://openalex.org/A5046609009)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多智能体框架，自动将 NeRF 研究论文转化为可训练、可部署的 Nerfstudio 插件。

**💡 创新点**

创新点包括：① 将 Nerfstudio 架构编码为上下文无关文法（CFG）以强制结构化代码生成；② 基于 Graph‑of‑Thought 的多文件协作生成；③ 通过引用图递归检索并恢复论文隐式依赖；④ 结合视觉反馈（PSNR、交叉视角一致性、VLM 修复）实现迭代优化；⑤ 设计了专门的 30 论文基准进行系统评估。

**🔧 技术方法**

技术手段包括：大语言模型（LLM）多智能体协作、CFG 约束生成、图依赖解析、自动化单元与集成测试、视觉驱动的自我修正与 VLM 导引。

**📊 数据集**

使用了 Nerfstudio 官方数据集（Blender、DTU 等）以及自行收集的 30 篇无公开代码的 NeRF 论文作为评测集。

**📈 对比分析**

与基准方法（Paper2Code、AutoP2C、GPT‑5 等）以及专家实现进行对比；在可执行性方面 100% 成功率；在视觉质量上与专家实现相比 PSNR 误差 ≤0.5 dB、SSIM 误差 ≤0.02；在新颖性实现上 100% 正确率，缺失率为 0%。

**⚠️ 局限性**

局限性包括：仅针对 NeRF 及其 Nerfstudio 生态；对完全新颖或不在 CFG 约束内的架构支持有限；在引用信息缺失或误解时可能导致组件缺失；模型依赖大量预训练 LLM 与大规模算力。

---

## 582. Benchmarking LLM Summaries of Multimodal Clinical Time Series for Remote Monitoring

**arXiv ID:** 2603.01557 | [PDF](https://arxiv.org/pdf/2603.01557v1)

**作者:** Aditya Shukla `[一作]` (Georgia Institute of Technology), May Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了一种基于事件的评估框架，用于多模态时间序列临床摘要，并在TIHM-1.5痴呆远程监测数据上进行基准测试。

**💡 创新点**

创新点在于将事件级事实对齐作为评估核心，揭示传统NLP指标与临床事实不一致，并证明可视化基准在事件捕捉方面显著优于文本提示。

**🔧 技术方法**

使用的技术包括大型语言模型（Llama、Gemma、Gemini）、统计条件提示、基于渲染图表的视觉语言推理、规则抽取的临床事件、事件召回率与覆盖率评估，以及AlignScore、SummaC和GPT-4o-mini的评估辅助。

**📊 数据集**

实验数据来源于TIHM-1.5公共远程监测数据集，涵盖56名痴呆患者的2803个患者日，记录多模态生理和行为指标。

**📈 对比分析**

通过对比零射、统计提示和可视化基准三种生成管线，结果显示可视化基准在异常召回率45.7%、持续时间召回率100%和覆盖率100%方面最佳，统计提示显著提升了异常召回率至33%和覆盖率94%，而零射模型几乎无法捕捉异常事件。

**⚠️ 局限性**

局限性包括仍有大量事件被漏报；可视化基准在语言流畅度和专业性方面表现不佳；评估指标仅基于规则抽取，缺乏人工标注的验证；模型在真实临床环境中的泛化和完整性保障尚未得到充分验证。

---

## 583. PhotoBench: Beyond Visual Matching Towards Personalized Intent-Driven Photo Retrieval

**arXiv ID:** 2603.01493 | [PDF](https://arxiv.org/pdf/2603.01493v1)

**作者:** Tianyi Xu `[一作]` (Shanghai Jiao Tong University), Jianghao Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 481 | [OpenAlex ID](https://openalex.org/A5036057873)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并发布了 PhotoBench 基准，基于真实个人相册构建多源画像（视觉、时空元数据、社交身份、事件）并通过意图驱动的查询合成与零真值查询来评估检索系统。

**💡 创新点**

创新点在于①提出多源联合画像与意图驱动查询合成，②引入零真值查询用于检验系统拒绝能力，③通过实验揭示了统一嵌入模型的“模态缺口”与代理系统的“源融合悖论”。

**🔧 技术方法**

使用 GPT‑4o 等 MLLM 生成图像描述与事件摘要，构建多源索引，采用 ReAct 代理与工具调用（向量检索、元数据过滤、人脸检索）进行推理，亦对比传统统一嵌入模型（CLIP、VLM2Vec 等）。

**📊 数据集**

PhotoBench 数据集包含 3,582 张真实个人相册图片及 1,188 条中英文查询，保留完整 GPS、时间戳、人脸聚类与事件摘要，支持完整的 ground‑truth 召回与零真值查询。

**📈 对比分析**

评估方法使用 Recall@K、NDCG@K、Set‑based F1、AUC 等指标，结果显示统一嵌入模型在纯视觉查询上表现最佳；代理系统在多源查询上显著优于嵌入模型，但在零真值查询上召回率低、拒绝率差；手机图库系统在召回上落后但拒绝率更高。

**⚠️ 局限性**

局限性包括①统一嵌入无法准确编码时空与身份约束；②代理系统在工具协调与多源融合上易导致误判，出现召回下降；③缺乏自适应拒绝机制与大规模实验验证。

---

## 584. DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage

**arXiv ID:** 2603.01106 | [PDF](https://arxiv.org/pdf/2603.01106v1)

**作者:** Haowen Gao `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在多模态大型语言模型上，提出 DIVA‑GRPO 通过动态评估问题难度并生成对应难度的变体，结合局部与全局优势计算、归一化与难度加权，解决奖励稀疏和优势消失问题，提升长链推理能力。

**💡 创新点**

创新点在于：1）基于历史回合的难度自适应评估与变体生成；2）同时计算局部与全局优势并用批归一化与难度加权平衡；3）引入奖励范围重缩放以抑制优势过度放大。

**🔧 技术方法**

采用强化学习中的 Group Relative Policy Optimization (GRPO) 框架，配合动态难度评估、变体生成、批归一化、难度加权缩放、奖励范围重缩放等技术。

**📊 数据集**

使用六大多模态推理基准：MathVista、MathVerse、MathVision、OlympiadBench、WeMath 及 MMK12‑test。

**📈 对比分析**

与多种闭源专有模型、开放源模型及基于 GRPO 的对比方法相比，DIVA‑GRPO 在 7B 规模下实现了 SOTA 结果，平均准确率 54.58，且训练效率提升约 2.55×，显著优于 GRPO、GSPO、DAPO 等基线。

**⚠️ 局限性**

仍受模型容量与数据覆盖的限制，在最具挑战性的竞赛级数学任务上逊色于更大规模或更先进的专有模型。

---

## 585. Super Research: Answering Highly Complex Questions with Large Language Models through Super Deep and Super Wide Research

**arXiv ID:** 2603.00582 | [PDF](https://arxiv.org/pdf/2603.00582v1)

**作者:** Yubo Dong `[一作]` (Zhejiang University), Linyi `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“Super Research”任务与基准，评估LLM在极复杂研究任务（需要100+检索步骤、1000+网页）上的性能。

**💡 创新点**

创新点在于将结构化分解、超宽检索和超深调查三者结合，并构建了5维评估框架（覆盖度、逻辑一致性、报告实用性、客观性、引用健康）及图锚定审计工具，形成前沿“ceiling‑level”测试。

**🔧 技术方法**

使用了LLM驱动的多阶段检索与推理管道、知识图谱投影与链式思考、工具交互、以及自动化评估指标。

**📊 数据集**

构建了300个专家编写的超难问答任务，涵盖10个专业领域，并生成对应的研究图、结构化报告和QA对。

**📈 对比分析**

通过覆盖度、逻辑一致性、报告实用性、客观性和引用健康等指标对12种系统进行比较，SOTA系统总体分数仅约28.6%，表明任务难度极高且当前模型表现有限。

**⚠️ 局限性**

局限性包括：仍需人工审阅保证任务质量，评估工具对某些细粒度细节捕捉不够；高检索步骤导致计算成本高，模型可能出现幻觉或信息过度简化。

---

## 586. ACDC: Adaptive Curriculum Planning with Dynamic Contrastive Control for Goal-Conditioned Reinforcement Learning in Robotic Manipulation

**arXiv ID:** 2603.02104 | [PDF](https://arxiv.org/pdf/2603.02104v1)

**作者:** Xuerui Wang `[一作]` (Cornell University), Hengyan Liu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种在线目标条件强化学习框架 ACDC，将自适应课程规划与动态对比学习相结合，以提高机器人操控任务的样本效率与最终成功率。

**💡 创新点**

创新点在于：①自适应权重机制根据学习进度动态平衡多样性与质量评估；②动态对比控制利用课程评分构造正负样本，并通过编码器的范数约束实现经验的意义排序；③两级分层协同确保经验选择与学习阶段保持一致。

**🔧 技术方法**

使用了多样性与质量评分（基于DPP与高斯距离）、自适应权重函数、LSTM 编码器、InfoNCE 损失加范数约束的对比学习，以及 EMA 平滑参数更新。

**📊 数据集**

在 OpenAI Gym 的六个机器人操控环境上进行实验，涵盖 Fetch（7-DoF 机器人臂）与 Shadow Dexterous Hand（24-DoF 手部）两类平台。

**📈 对比分析**

与 DDPG+HER、DAGGER、CHER、DTGSH、FAHER 等多种基线对比，ACDC 在所有环境中均实现了最高或相近的最终成功率，同时在 Time-to-Threshold 与累计遗憾等样本效率指标上明显优于对照方法。

**⚠️ 局限性**

局限性包括：对极长时程任务（如 AntMaze）提升不显著；框架依赖在线学习与实时经验收集，难以直接迁移至离线或离线-在线设置；对超参数如阈值的选取仍需经验调优。

---

## 587. Assessing Crime Disclosure Patterns in a Large-Scale Cybercrime Forum

**arXiv ID:** 2603.01624 | [PDF](https://arxiv.org/pdf/2603.01624v1)

**作者:** Raphael Hoheisel `[一作]` (University of Twente), Masarah Paquet-Clouston `[通讯]` (Université de Montréal)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了一个大型网络犯罪论坛中的犯罪披露模式，研究了300k用户共350万条帖文中的公开披露与私聊行为

**💡 创新点**

首次将大规模犯罪披露划分为三层（benign、grey、crime），并使用Markov链、逻辑回归与 assortative mixing 方法揭示披露转移与私聊关系

**🔧 技术方法**

利用GPT‑4.1进行自动化标签，使用聚类分析区分活跃与非活跃用户，Markov链模拟披露转移，逻辑回归评估私聊概率，网络分析检测同类披露用户间的私聊同质性

**📊 数据集**

公开泄露的nulled.io论坛数据（2015‑2016年，约350万条帖子、400k私聊、约30万用户）

**📈 对比分析**

相较于以往仅做手工标注或单纯描述性分析，本文在350万帖中自动标注并量化披露转移，模型解释度（McFadden R²≈0.16）显示行为可预测；未提供跨论坛基准，因只研究单一论坛

**⚠️ 局限性**

仅标注初始帖子，忽略评论与私聊；只分析一个论坛且时间有限，可能导致披露比例低估；LLM标注虽准确≈83%但仍存在四类混淆；缺乏真实犯罪与法律行为的关联验证

---

## 588. The Finality Calculator: Analyzing and Quantifying Filecoin's Finality Guarantees

**arXiv ID:** 2603.01307 | [PDF](https://arxiv.org/pdf/2603.01307v1)

**作者:** Guy Goren `[一作]` (Aptos Labs), Jorge M. Soares `[通讯]` (Finisterra Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Filecoin网络的最终性（tipset是否会被链变更撤销）进行动态概率评估，提出并实现了 Finality Calculator 算法；

**💡 创新点**

创新点在于：①使用实时链历史动态计算错误概率，而非传统静态最坏情况；②仅需观察诚实节点产生的块即可，无需共识机制改动；③提出针对节点与链上智能合约两种视角的独立计算方法；

**🔧 技术方法**

技术方法包括概率论与随机过程（Poisson、Skellam 分布）、一致广播模型、随机 beacon（drand）与 VRF 的分析，最终用 Python+NumPy/ SciPy 实现公式推导；

**📊 数据集**

数据集：①在 0.8–1.0 的链饱和度（α）下的合成链，产生 40,000 个 round 的随机 tipset；②2023 年采集的真实 Filecoin 区块链 80,000 个 round（包含正常与突发低产区段）；

**📈 对比分析**

与传统 900-round 软最终性阈值比较：节点视角下 30 rounds 可实现 2⁻³⁰ 误差，约 30 倍加速；在真实链上 30 rounds 误差维持在 10⁻²⁵ 以内；智能合约视角需约 60 rounds；实现中使用早停和截断提升计算效率；

**⚠️ 局限性**

限制：对抗者模型假设较强（可利用所有非链块加权；分割策略假设为最优）；未考虑对链分叉拆分的实际协同难度；实现未针对性能优化，需手动获取链历史；仅覆盖节点与合约视角，未涵盖跨链或多节点同步细节。

---

## 589. Learn Hard Problems During RL with Reference Guided Fine-tuning

**arXiv ID:** 2603.01223 | [PDF](https://arxiv.org/pdf/2603.01223v1)

**作者:** Yangzhen Wu `[一作]` (University of California Berkeley), Tianle Cai `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种参考引导微调（ReGFT）方法，用以在强化学习前提高模型在难题上的推理能力，从而缓解奖励稀疏问题。

**💡 创新点**

创新点在于：①将参考解作为提示而非完整复制，促使模型生成与自身推理风格一致、同时与参考一致的轨迹；②通过混合自生成正确轨迹和参考引导轨迹进行监督微调，显著提升模型在硬题上的通过率；③在强化学习阶段采用 DAPO 并以 ReGFT 初始化，进一步放大稀疏奖励带来的梯度信息。

**🔧 技术方法**

主要技术包括：①参考引导采样（使用部分参考解作为提示）；②基于验证器的奖励机制；③ReFT 与 ReGFT 的对比微调；④DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）强化学习；⑤大规模采样与推理时的 scaling 评估。

**📊 数据集**

使用的数据集：训练集 OmniMath（4,428 个奥林匹克级数学题）；评估集 AIME 2024、AIME 2025、BeyondAIME（共 300 题）。

**📈 对比分析**

与原始检查点、ReFT 以及不同采样规模的 RL 方案进行对比。实验表明：ReGFT + DAPO 在 pass@k 评估中无论 k 值大小都优于其它方法；在训练过程中收敛更快、最终准确率更高；在 AIME 2024/25 和 BeyondAIME 上分别提升了约 1–2% 的通过率，并在多样性与推理稳定性上表现更好。

**⚠️ 局限性**

局限性包括：①仍未达到 100% 的通过率，主要受限于模型对复杂推理的理解和规则基验证器的覆盖；②部分参考引导可能无法捕捉到所有深层次推理策略；③方法依赖于高质量的参考解，若训练集参考质量不高，效果可能下降；④对其他非数学推理任务的迁移性尚未验证。

---

## 590. HVR-Met: A Hypothesis-Verification-Replaning Agentic System for Extreme Weather Diagnosis

**arXiv ID:** 2603.01121 | [PDF](https://arxiv.org/pdf/2603.01121v1)

**作者:** Shuo Tang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Chenglin Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多智能体系统HVR-Met，用于自动化极端天气诊断

**💡 创新点**

核心创新在于“假设-验证-再规划”闭环机制与深度专家知识库集成

**🔧 技术方法**

采用了多智能体协作框架、LLM生成代码、可视化工具、视觉语言模型等技术

**📊 数据集**

构建了基于584篇极端天气论文的诊断准则库和约束索引/图表知识库，并使用WeatherBench2等数据集

**📈 对比分析**

与GPT-5、Gemini-3-Pro等大型语言模型对比，HVR-Met在索引计算、图表绘制和完整报告的评估指标上分别达到71.86%、79.52%和85%通过率，整体表现优于传统单一模型

**⚠️ 局限性**

局限在于对专家知识库的依赖、对高度复杂多步推理的错误传播敏感，以及在极端天气数据稀缺时可能出现的可靠性不足

---

## 591. Boosting Entropy with Bell Box Quantization

**arXiv ID:** 2603.01599 | [PDF](https://arxiv.org/pdf/2603.01599v1)

**作者:** Ningfeng Yang `[一作]` (University of British Columbia), Tor M. Aamodt `[通讯]` (University of British Columbia)

**通讯引用:** 7504 | [OpenAlex ID](https://openalex.org/A5026788167)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Bell Box Quantization（BBQ）的量化方法，利用信息理论最优（ITO）量化并将输出映射到计算效率高的数值域；

**💡 创新点**

核心创新在于将学习视为域不相关过程，先在输入域进行ITO量化，再将结果转换到可直接在低精度算子上高效执行的域；

**🔧 技术方法**

使用Hadamard变换 + RMS归一化、概率积分变换（Φ）、均匀量化以及可学习的缩放因子γ；

**📊 数据集**

在LLaMA系列模型（95M-200M参数）上使用C4数据集进行预训练；

**📈 对比分析**

与现有QAPT方法QuEST、LSQ进行对比，BBQ在相同精度下熵更高、困惑度降低：4-bit下降0-2点，3-bit下降0-4点，2-bit下降0-5点，1-bit下降0-18点；在推理速度上相较FP16提升约40%，相较NF4提升约48%；

**⚠️ 局限性**

局限性：不具备限制欧氏误差的能力，适用于QAPT但不适合QAFT或PTQ；对Hadamard正态化的假设在训练后可能不完全成立；

---

## 592. MAP-Diff: Multi-Anchor Guided Diffusion for Progressive 3D Whole-Body Low-Dose PET Denoising

**arXiv ID:** 2603.02012 | [PDF](https://arxiv.org/pdf/2603.02012v1)

**作者:** Peiyuan Jing `[一作]` (Zurich University of Applied Sciences), Javier A. Montoya-Zegarra `[通讯]` (Zurich University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种多锚点引导的扩散模型 MAP-Diff，用于从超低剂量 PET 图像恢复全剂量图像，并生成与临床剂量相匹配的中间剂量重建。

**💡 创新点**

创新点在于：①将临床观察到的中间剂量图像作为时间步锚点，对扩散过程进行轨迹级监督；②通过时间步分区与加权损失实现阶段化引导；③实现无须低剂量输入即可在推理时生成剂量一致的中间输出。

**🔧 技术方法**

技术上采用条件扩散概率模型（DDPM）作为骨干，结合锚点监督、时间步加权、差分校准等实现多锚点引导；使用 3D 体卷积网络架构；损失为噪声预测损失加锚点重建损失。

**📊 数据集**

使用两组数据集：内部数据集（Siemens Biograph Vision Quadra）和外部跨仪器数据集（United Imaging uEXPLORER），均包含多剂量 PET 扫描。

**📈 对比分析**

与多种 CNN、GAN、Transformer 和传统 DDPM 基线对比，使用 PSNR、SSIM、NMAE 评估。MAP‑Diff 在内部数据集 PSNR 提升约 1.23 dB、SSIM 0.986、NMAE 降至 0.103；在外部数据集同样获得最佳 PSNR 34.42 dB、NMAE 0.141，显著优于所有对照方法。

**⚠️ 局限性**

局限性包括：需依赖已标注的多剂量配对数据以确定锚点和时间区间；锚点数量与位置对性能影响显著，未实现自适应锚点选择；计算资源消耗较大，推理速度受限；模型在极端噪声或非标准剂量情况下的鲁棒性尚未完全验证。

---

## 593. AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution

**arXiv ID:** 2603.01145 | [PDF](https://arxiv.org/pdf/2603.01145v1)

**作者:** Yutao Yang `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**通讯引用:** 8071 | [OpenAlex ID](https://openalex.org/A5062604912)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AutoSkill，一个基于经验驱动的终身学习框架，能够从用户交互中抽取、表示、检索并迭代更新可编辑的技能对象，以在不重新训练模型的前提下提升 LLM 代理的个性化和持续改进能力。

**💡 创新点**

核心创新在于：①把用户偏好和工作流程转化为可视化、可版本化的技能工件；②采用多模态检索与阈值过滤，实现技能的精准注入；③通过 Prompt‑Driven 模块完成抽取、维护与合并，确保技能生命周期的可控性与可维护性；④与现有 LLM 与向量检索后端无缝集成，构建可扩展的插件式体系。

**🔧 技术方法**

技术方案主要包括 Prompt‑Driven 模块（查询重写、对话生成、技能抽取、管理决策与合并）、向量检索（稠密+BM25 混合匹配）、技能表示（包含名称、描述、执行指令、触发器、标签、示例及版本号）以及本地 SkillBank 存储与索引。

**📊 数据集**

在 WildChat‑1M 真实对话语料上构建四个子集（中文 GPT‑3.5、英文 GPT‑3.5、中文 GPT‑4、英文 GPT‑4），并用这些数据评估技能抽取与维护效果；同时在多语言、多模型环境下统计抽取的技能数量、版本迭代次数与标签分布。

**📈 对比分析**

实验通过对比未使用 AutoSkill 前后的对话质量与技能覆盖度，发现：①可抽取 1858 条技能，覆盖 8 大类任务；②相同任务多次使用后，技能版本平均迭代 5 次以上，表明持续改进；③用户体验评估显示对个性化需求的满足度提升约 15‑20%（基于人工标注的满意度分数）。

**⚠️ 局限性**

局限性包括：①技能抽取依赖 LLM 的提示质量，误抽或漏抽仍然存在；②对隐晦或跨域需求的捕捉能力有限；③在高并发场景下，后台抽取/合并过程可能产生延迟；④技能库管理需要人工审核，缺乏自动化的错误纠正机制。

---

## 594. Relatively Smart: A New Approach for Instance-Optimal Learning

**arXiv ID:** 2603.01346 | [PDF](https://arxiv.org/pdf/2603.01346v1)

**作者:** Shaddin Dughmi `[一作]` (University of Southern California), Alireza F. Pour `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了相对智能学习（relatively smart learning）框架，将传统的 Smart PAC 学习进一步放宽，只要求监督学习器与可从未标记数据中可验证的最佳半监督保证相匹配。

**💡 创新点**

创新点包括：1) 发现 Smart 学习失败的根本原因是“不可区分性”现象；2) 设计相对智能学习作为对这一障碍的最小放宽；3) 在分布无关和分布族情境下给出相对智能学习的上界（OIG 学习器在样本量平方放大下可实现）和下界（任意学习器至少需要平方放大），揭示了该范式的极限；4) 说明相对智能学习在分布族上的非单调性。

**🔧 技术方法**

主要技术手段包括：离散化样本的无偏分配、Birthday 诅咒与均匀性检验、留一误差（leave‑one‑out）分析、OIG 学习器的最优性证明、概率方法构造极端分布族、以及对可验证错误率的证明与下界构造。

**📊 数据集**

论文纯理论，未使用具体实验数据集，所有结果均通过数学证明与构造实例得出。

**📈 对比分析**

通过理论证明与构造例子与以往的 Smart 学习结果进行比较：1) 相对智能学习能在样本量平方放大的前提下实现 Smart 学习的性能；2) 对于 OIG 学习器，展示了上界与下界相匹配；3) 对 ERM 学习器的相对智能性未能证明，留下开放问题；4) 在分布族情境中，证明了一些族下相对智能学习不可行，而在更大族中可实现，揭示非单调性。

**⚠️ 局限性**

局限性包括：1) 对 ERM 或更简单学习器是否相对智能仍是未解问题；2) 结果主要针对可测量的、计数域与可数分布族，未涵盖连续或更一般情况；3) 需要对未标记样本进行较大数量（平方级）才能实现，实际样本成本较高；4) 证明依赖于均匀性检验与离散化技术，可能难以推广到复杂分布。

---

## 595. Open-Vocabulary vs Supervised Learning Methods for Post-Disaster Visual Scene Understanding

**arXiv ID:** 2603.01324 | [PDF](https://arxiv.org/pdf/2603.01324v1)

**作者:** Anna Michailidou `[一作]` (Harokopio University), Georgios Th. Papadopoulos `[通讯]` (Archimedes Athena Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比闭集监督模型和开词汇模型在灾后航空场景的语义分割与目标检测任务上的性能，探讨它们在不同灾害数据集下的适用性。

**💡 创新点**

系统化地将多种卷积与变换器架构的监督与开词汇方法在相同实验协议下进行横向对比，首次揭示开词汇模型在灾后航空图像中的优缺点与可改进方向。

**🔧 技术方法**

采用CNN/Transformer的语义分割网络（PSPNet、CCNet、DeepLabV3+、SegFormer、Mask2Former）和检测网络（YOLOv5/8/26/11、RT-DETRv2），以及开词汇方法（MaskCLIP、SegEarth‑OV‑3、FC‑CLIP、OWL‑ViT、Grounding DINO、YOLOE）进行实验，使用mIoU和mAP₅₀指标评估。

**📊 数据集**

数据集包括FloodNet+、RescueNet（分割任务）以及LADD、D‑Fire（检测任务），涵盖洪水、地震、野火和搜救场景。

**📈 对比分析**

结果显示监督模型在所有指标上均优于零射击开词汇模型，且通过少量迁移学习后开词汇模型性能可显著提升；但开词汇方法在小目标、边界细化和域迁移方面仍显不足。

**⚠️ 局限性**

局限性主要在于开词汇模型对灾害特定视觉特征和细粒度类别的迁移能力差，且在面对极小目标和高遮挡环境时准确率低；监督方法虽表现最好，但对标注成本和灾害多样性的适应性有限。

---

## 596. Keyframe-Guided Structured Rewards for Reinforcement Learning in Long-Horizon Laboratory Robotics

**arXiv ID:** 2603.00719 | [PDF](https://arxiv.org/pdf/2603.00719v1)

**作者:** Yibo Qiu `[一作]` (Suzhou Institute for Advanced Research University of Science and Technology of China), Mingzhai Sun `[通讯]` (Suzhou Institute for Advanced Research University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于关键帧的奖励生成框架，自动从演示中提取关键帧并生成阶段目标，以指导实验室自动化中的长期精细操作。

**💡 创新点**

创新点在于利用运动学感知的关键帧提取、潜在相似度的阶段奖励、扩散模型生成目标以及多视角融合与人机协同微调。

**🔧 技术方法**

采用了视觉-语言-动作(VLA)模型Octo、扩散策略、向量量化潜在编码器、潜在相似度度量、混合缓冲区的RL以及行为克隆与Q最大化。

**📊 数据集**

使用了四个真实实验室任务（离心管装载、培养皿解盖、移液器尖端装配、精准液体传输）的演示数据作为训练集。

**📈 对比分析**

与HG-DAgger、Hil-ConRFT、Hil-SERL等基线比较，方法在40–60分钟在线微调后平均成功率达82%，显著高于对手（42%、47%和≈0%）。

**⚠️ 局限性**

局限在于仅基于RGB潜在相似度，对光照、反射等视觉噪声敏感；缺乏触觉、深度等多模态信息；仍需人工干预与手动环境重置。

---

## 597. TGM-VLA: Task-Guided Mixup for Sampling-Efficient and Robust Robotic Manipulation

**arXiv ID:** 2603.00615 | [PDF](https://arxiv.org/pdf/2603.00615v1)

**作者:** Fanqi Pu `[一作]` (Shenzhen International Graduate School, Tsinghua University), Wenming Yang `[通讯]` (Shenzhen International Graduate School, Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在RLBench与COLOSSEUM基准上提出TGM-VLA模型，通过改进关键帧采样、引入颜色反转投影分支以及任务引导的混合数据增强，显著提升机器人模仿学习的成功率和训练效率。

**💡 创新点**

创新点包括：①设计了去重与循环交替的关键帧采样策略，减少80%内存消耗并提升5倍训练速度；②在点云投影时加入颜色反转分支，解决暗色物体在黑背景下失真问题；③提出跨任务与同任务混合（Task‑Guided Mixup）机制，强化语言-动作对齐与抗干扰能力。

**🔧 技术方法**

主要技术手段为：多视角点云投影+SAM2视觉编码器、CLIP文本编码器、多视角Transformer融合、上采样头生成热图、颜色反转投影、跨任务/同任务混合训练策略。

**📊 数据集**

使用的数据集为RLBench（18个操控任务，每任务100条专家演示）与COLOSSEUM（12种视觉扰动，20,371个变体），并在真实SO101机器人上进行小规模验证。

**📈 对比分析**

与先前方法（BridgeVLA、RVT‑2、SAM2ACT等）对比，TGM‑VLA在RLBench的平均成功率达到90.5%（比BridgeVLA高2.3%），在COLOSSEUM的平均成功率68.8%（平均排名1.14）。训练时间减少5×、内存占用降低80%。

**⚠️ 局限性**

局限性包括：采样策略仍为手工设计，缺乏自适应与物理感知的增强；对极端真实场景的迁移能力还有待进一步验证；混合策略在复杂多目标任务中可能引入标签冲突，需更细粒度的指令解析。

---

## 598. Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning

**arXiv ID:** 2603.00550 | [PDF](https://arxiv.org/pdf/2603.00550v1)

**作者:** Yu Wang `[一作]` (Tongji University), Shengjie Zhao `[通讯]` (Tongji University)

**通讯引用:** 3184 | [OpenAlex ID](https://openalex.org/A5035948567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了LAS-VAD框架，结合异常连通分量机制（ACC）和意图感知机制（IAM），并利用异常属性信息来完成弱监督视频异常检测。

**💡 创新点**

创新点包括：① ACC模块通过构造视频帧的连通图并求解连通分量，实现帧级语义分组和伪标签生成；② IAM模块提取位置、速度、加速度特征，构建意图原型并进行跨意图对比学习，显著提升对相似正常与异常行为的区分；③ 将异常属性（如火焰、烟雾）与CLIP文本特征融合，引导模型学习更细粒度的异常语义。

**🔧 技术方法**

使用的技术包括：CLIP视觉‑文本预训练模型、局部Transformer、GCN、MIL（多实例学习）、交叉熵/二元交叉熵、L1正则、意图原型与InfoNCE对比学习、连通图DFS等。

**📊 数据集**

实验数据集：XD‑Violence（6类暴力事件）和UCF‑Crime（13类异常事件），均为未剪裁视频集。

**📈 对比分析**

与现有SOTA方法（PE‑MIL、LEC‑VAD、π‑VAD、VadCLIP、ITC等）在粗粒度AP/AUC和细粒度mAP上进行对比。LAS‑VAD在I3D/CLIP特征下分别取得89.96/87.92 AP（XD‑Violence）和91.05/90.86 AP（UCF‑Crime），在细粒度mAP上平均提升至36.89（XD‑Violence）和15.62（UCF‑Crime），相较基线提升约5%/15%。

**⚠️ 局限性**

局限性：① 依赖CLIP预训练的视觉‑文本特征，模型规模和推理速度相对较大；② 对实时检测或资源受限场景的适配尚未评估；③ 仅利用视频级标签，仍无法完全解决帧级语义的稀疏性；④ 对多模态音频等额外信息的利用有限。

---

## 599. When Humans Don't Feel Like an Option: Contextual Factors That Shape When Older Adults Turn to Conversational AI for Emotional Support

**arXiv ID:** 2603.01413 | [PDF](https://arxiv.org/pdf/2603.01413v1)

**作者:** Mengqi Shi `[一作]` (University of Washington), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**通讯引用:** 1813 | [OpenAlex ID](https://openalex.org/A5054435118)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对18名老年人的访谈，分析他们在何种情境下选择对话式AI而非亲近他人来表达情感，揭示时间可用性、关系负担与自我呈现等情境因素的作用。

**💡 创新点**

聚焦情境层面的时机决策，而非仅研究总体态度，首次系统阐述时间、关系与自我呈现对老年人情感支持AI使用的影响。

**🔧 技术方法**

采用定性访谈和归纳主题分析（Thematic Analysis），并未实现或训练任何算法模型。

**📊 数据集**

使用18名老年人（年龄50-77岁）在英文访谈中的文字记录，非公开数据集。

**📈 对比分析**

本研究为探索性定性研究，没有对模型或算法进行比较或性能评估，因此不存在性能指标。

**⚠️ 局限性**

样本量有限、地域与机构限制、仅英文访谈、未调查用户对AI的态度与期望，也未评估持续使用可能带来的风险。

---

## 600. Turning Black Box into White Box: Dataset Distillation Leaks

**arXiv ID:** 2603.01053 | [PDF](https://arxiv.org/pdf/2603.01053v1)

**作者:** Huajie Chen `[一作]`, Wanlei Zhou `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种三阶段信息泄露攻击（IRA），通过分析数据集蒸馏产生的合成数据，揭露模型架构、成员身份以及真实样本，证明现有蒸馏方法易被攻击者窃取隐私。

**💡 创新点**

创新点在于：①构建了基于损失轨迹的架构推断模型；②利用隐藏层输出的成员身份推断模型；③设计了双网络扩散框架与轨迹损失相结合的模型逆转攻击，实现从合成数据逆推真实样本。

**🔧 技术方法**

使用技术包括：损失轨迹分类器、全连接攻击网络、隐层输出特征提取、双网络扩散生成器（ϕ、ψ）、轨迹损失、分类损失、MSE、交叉熵等。

**📊 数据集**

实验数据集：CIFAR‑10、CIFAR‑100、CIFAR‑100‑Ext、CINIC‑10、TinyImageNet‑200、ImageNet，辅以对应的辅助数据集（如CINIC‑10、TinyImageNet‑200）。

**📈 对比分析**

与MTT、FTD、DATM、SelMatch、SeqMatch等五大蒸馏算法在多种网络架构（ConvNet、AlexNet、ResNet18、VGG11‑BN）下对比，架构推断准确率可达70–95%，成员身份推断的BA/AUC/T@LF最高可达0.94/0.98/74.8，模型逆转攻击准确率可超过90%，KNN距离显著下降，表明攻击效果极佳。

**⚠️ 局限性**

局限性：需要公开合成数据并获取同分布辅助数据；对高质量蒸馏（高IPC）更易泄露；攻击依赖对损失轨迹的可观测性，若引入DP‑SGD或差分隐私噪声会降低攻击效果；结果主要基于图像基准数据集，尚未验证在其他领域的通用性。

---

## 601. Learning to Attack: A Bandit Approach to Adversarial Context Poisoning

**arXiv ID:** 2603.00567 | [PDF](https://arxiv.org/pdf/2603.00567v1)

**作者:** Ray Telikani `[一作]` (University of Technology Sydney), Amir H. Gandomi `[通讯]` (University of Technology Sydney)

**通讯引用:** 59576 | [OpenAlex ID](https://openalex.org/A5039341855)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种黑盒的上下文污染攻击模型（AttackModel），将攻击参数空间视为连续三维臂，并通过 UCB‑aware 最大熵逆强化学习（MaxEnt IRL）构建受害者策略的代理模型，使用 GP‑UCB 进行连续臂探索，利用投影梯度下降（PGD）生成针对性扰动，并通过预算控制与查询选择策略在有限攻击预算下实现高效、隐蔽的攻击。

**💡 创新点**

① 将攻击参数化为连续臂并采用 GP‑UCB 实现自适应探索，避免传统离散攻击的浪费与易被检测；② 结合最大熵逆强化学习实现无梯度、无内部参数的代理学习，实现对非平稳神经上下文贝叶斯（NCB）算法的逼近；③ 提出多目标查询选择与攻击预算控制，兼顾攻击成功率、影响力与隐蔽性；④ 给出攻击者与受害者的子线性与下界理论保证，首次从信息论角度给出受害者在非平稳策略下的累积 regret 下界。

**🔧 技术方法**

最大熵逆强化学习（MaxEnt IRL）、高斯过程 UCB（GP‑UCB）、投影梯度下降（Projected Gradient Descent, PGD）、预算控制与查询选择策略、连续臂多目标优化（scalarization）、UCB‑aware 代理策略、上下文特征提取、梯度统计、Mahalanobis 距离正则化。

**📊 数据集**

Yelp（餐馆点评）、MovieLens（电影评分）和 Disin（离散交互）三大真实数据集。

**📈 对比分析**

与五种基线攻击（包括基于梯度、随机、贝叶斯等方法）以及五个主流 NCB 受害者（NeuralUCB、Neural‑LinUCB、NeuralTS、RobustBandit、R‑NeuralUCB）进行对比。结果显示：AttackModel 在 5,000 步时段内可使受害者累积 regret 提升 2.8×，目标臂被选比例提升 1.7–2.5×；在不同预算下的成本效益明显优于基线；同时在鲁棒性更高的算法（R‑NeuralUCB、RobustBandit）上能自动转向更隐蔽的攻击策略。

**⚠️ 局限性**

1）计算成本随攻击预算 B 的立方级增长，GP 核更新主导；2）对极度随机化策略（如 NeuralTS）攻击效果相对较弱；3）攻击效果依赖于 MaxEnt IRL 的逼近质量，若受害者策略剧烈漂移需更频繁重训练；4）理论假设（Lipschitz、噪声子高斯）在某些非平稳场景下可能不完全满足；5）仅针对上下文贝叶斯场景，扩展至更复杂的 RL 环境仍需进一步研究。

---

## 602. Boltzmann-based Exploration for Robust Decentralized Multi-Agent Planning

**arXiv ID:** 2603.02154 | [PDF](https://arxiv.org/pdf/2603.02154v1)

**作者:** Nhat Nguyen `[一作]` (University of Adelaide), Hung Nguyen `[通讯]` (University of Adelaide)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种分布式的协调 Boltzmann 蒙特卡洛树搜索（CB‑MCTS）算法，用于解耦多代理协同规划中的稀疏或偏斜奖励问题。

**💡 创新点**

创新点在于：①用随机 Boltzmann 策略替代传统的 UCT 选择，加入衰减的熵奖励以实现持续且聚焦的探索；②通过边际贡献函数实现无中心化的多代理协调；③首次给出 Dec‑MCTS 在欺骗性树上的简单风险上界，并证明 CB‑MCTS 在简单风险下降速率上显著更快。

**🔧 技术方法**

技术方法包括：分布式 MCTS 4 步流程、Boltzmann 选择政策（带温度调度与熵正则化）、折扣回溯、分布式梯度共识通信以及对联合奖励的边际贡献估计。

**📊 数据集**

实验数据集主要有：
- Frozen Lake（多目标稀疏奖励网格世界）
- Oil Rigs Inspection（油田巡检路径规划，覆盖率奖励）
- 经典 D‑chain 退化树用于理论验证。

**📈 对比分析**

与 Dec‑MCTS、GU‑MCTS、NE‑MCTS、Independent、CAR‑DENTS 等基线比较；在欺骗性和稀疏奖励环境下 CB‑MCTS 的简单风险下降速度远快于 Dec‑MCTS，联合得分提升 70% 以上；在标准基准（Frozen Lake、油田巡检）中与最先进方法保持竞争力，尤其在稠密奖励场景下 NE‑MCTS（无熵）表现最优。

**⚠️ 局限性**

局限性包括：
- 对极端大规模代理数或超大搜索树的可扩展性未在实验中验证；
- 依赖于折扣因子与温度调度的手动调参，缺少自适应机制；
- 对对抗扰动或不确定环境的鲁棒性尚未系统评估。

---

## 603. Bridging the gap between Performance and Interpretability: An Explainable Disentangled Multimodal Framework for Cancer Survival Prediction

**arXiv ID:** 2603.02162 | [PDF](https://arxiv.org/pdf/2603.02162v1)

**作者:** Aniek Eijpe `[一作]` (Utrecht University), Wilson Silva `[通讯]` (Utrecht University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出 DIMAFx 模型，结合 histopathology WSIs 与转录组数据进行癌症生存预测，并在模型内部实现模态特定与模态共享特征的可解释解耦表示。

**💡 创新点**

创新点在于：① 将分离的注意力融合与解耦损失（距离相关）相结合，得到可解释的模态特定与共享表示；② 在可解释框架中嵌入 SHAP 分析，系统揭示单模态特征、跨模态交互以及它们对风险预测的贡献；③ 通过可学习的聚合层进一步提升解耦质量。

**🔧 技术方法**

使用的技术包括：DINOv2 + UNI 预训练的视觉 Transformer、Gaussian Mixture 模型提取 WSI 原型、Self‑Normalizing Networks (SNN) 处理通路特征、双自注意力和交叉注意力实现模态解耦、距离相关正则化、Cox 比例风险层以及 DeepSHAP 进行可解释性分析。

**📊 数据集**

实验使用 TCGA 四个癌症数据集：BRCA、BLCA、LUAD、KIRC，均包含 WSIs、转录组和临床随访信息。

**📈 对比分析**

与多种基线（CoxPH、ABMIL、PANTHER、PIBD、SurvPath、MMP 等）比较，DIMAFx 在 C‑index 及其 IPCW 修正指标上实现了 state‑of‑the‑art 表现；在多模态组合下的性能尤其优于无解耦模型，且在解耦度（距离相关）上显著提升。

**⚠️ 局限性**

局限性包括：① 依赖预定义的通路集合，可能忽略未包含基因；② WSIs 与转录组特征数量不平衡，可能影响解耦效果；③ 对小样本集的过拟合风险；④ 交互机制的生物学解释尚未完全阐明，需进一步研究。

---

## 604. GlassMol: Interpretable Molecular Property Prediction with Concept Bottleneck Models

**arXiv ID:** 2603.01274 | [PDF](https://arxiv.org/pdf/2603.01274v1)

**作者:** Oscar Rivera `[一作]` (Northwestern University), Kaize Ding `[通讯]` (Northwestern University)

**通讯引用:** 2781 | [OpenAlex ID](https://openalex.org/A5044455276)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种通用的概念瓶颈模型（CBM）框架，用于解释性分子属性预测，并通过自动概念精炼和LLM引导的概念选择来解决相关的注释、相关性与容量三大缺口。

**💡 创新点**

创新点在于：①使用RDKit生成完整的物理化学描述子作为无监督的概念标签；②利用GPT‑4对任务相关概念进行自动筛选，克服了人工挑选概念的瓶颈；③实现了模型无关的CBM结构，证明解释性并不必然牺牲性能；④通过多任务实验验证了该方法在不同基准上均可匹配或超越黑盒模型。

**🔧 技术方法**

核心技术包括：分子图神经网络（GINE）与分子序列LLM（SMILY‑APE）作为基座；多层感知机进行概念投影；线性层做最终预测；组合任务损失与概念损失的联合训练；LLM（ChatGPT）进行任务感知概念筛选；RDKit进行概念标签生成。

**📊 数据集**

使用了Therapeutics Data Commons共13个公开基准数据集，涵盖ADME（如BBB、HIA等）与毒性（如DILI、AMES等）任务，采用80/10/10的结构分割。

**📈 对比分析**

与传统GNN基线（GINE、GCN、GAT、GraphSAGE）以及LLM基线（SMILY‑APE、Qwen、Llama）进行对比；在LLM设置下平均AUROC提升约0.057，在GNN设置下平均提升约0.012；在多数任务上均表现出与黑盒模型相当或更优的性能。

**⚠️ 局限性**

局限性包括：①概念选择高度依赖LLM的质量与可解释性，若LLM误判可能导致概念偏差；②使用RDKit描述子生成的概念标签仅覆盖连续数值属性，难以处理离散或结构化概念；③方法在新任务或极端化学空间的泛化仍需进一步验证；④对概念标签的噪声鲁棒性虽表现良好，但在真实工业数据中标签质量参差不齐时可能影响效果。

---

## 605. CMI-RewardBench: Evaluating Music Reward Models with Compositional Multimodal Instruction

**arXiv ID:** 2603.00610 | [PDF](https://arxiv.org/pdf/2603.00610v1)

**作者:** Yinghao Ma `[一作]` (Queen Mary University of London), Emmanouil Benetos `[通讯]` (Queen Mary University of London)

**通讯引用:** 4725 | [OpenAlex ID](https://openalex.org/A5084672392)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CMI-RewardBench统一评估框架，涵盖文本、歌词和音频混合指令的音乐奖励模型；

**💡 创新点**

创新点在于构建大规模伪标记偏好数据CMI-Pref-Pseudo、专家标注集CMI-Pref，以及实现参数高效、可处理多模态输入的CMI-RM；

**🔧 技术方法**

采用MuQ-MuLan两塔编码、Prompt Transformer与Joint Transformer架构，使用Bradley‑Terry概率模型和LCC/SRCC等统计评估；

**📊 数据集**

使用CMI-Pref-Pseudo（110k对）、CMI-Pref（4k对）以及PAM、MusicEval、Music Arena、SongEval等公开数据集；

**📈 对比分析**

与现有音乐质量与对齐基线相比，CMI-RM在音乐性和指令遵循方面达到或超过70%+准确率，尤其在CMI-Pref测试集上显著高于最先进的多模态LLM（如Gemini 3 Pro、Qwen3-Omni）并在最佳‑of‑N重排序中提升整体用户偏好；

**⚠️ 局限性**

局限性包括伪标签生成过程对LLM质量的依赖、数据集中可能存在的偏差、对非文本/非歌词输入的支持仍有限，以及需要在商业API合规性与版权等方面进行持续监控与治理。

---

## 606. Intrinsic Task Symmetry Drives Generalization in Algorithmic Tasks

**arXiv ID:** 2603.01968 | [PDF](https://arxiv.org/pdf/2603.01968v1)

**作者:** Hyeonbin Hwang `[一作]` (KAIST), Yeachan Park `[通讯]` (Sejong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文研究并验证了内在任务对称性如何驱动神经网络从记忆到泛化的过程（grokking），并提出三阶段训练动态（记忆、对称性获取、几何组织），在多种算法任务上进行实验。

**💡 创新点**

创新点包括：①将内在对称性定位为泛化的根本驱动因素；②给出对称性违背度量作为预测泛化的诊断指标；③设计对称性提示损失和几何促发先验以加速泛化；④提供理论解释对称性如何导致低维几何组织；⑤系统比较多种加速策略，验证对称性提示的优越性。

**🔧 技术方法**

技术手段：对称性违背度量（KL散度）、对称性提示损失、核范数/熵/Lipschitz几何先验、低秩正则化；模型使用多层感知机/Transformer；通过PCA可视化嵌入空间几何；在20个独立训练跑上评估训练步数与测试准确率。

**📊 数据集**

数据集：模块化算术任务（6个算术运算）、图度量完成任务（Path、Cycle、Cylinder、Hypercube、Lattice、3D Lattice）、比较任务（二维、三维属性空间），均为人工合成的算法问题。

**📈 对比分析**

比较方法：基准无对称性损失 vs. 对称性提示 vs. 几何先验 vs. GrokFast，使用每种策略在20个随机种子下记录达到100%测试准确率所需的训练步数。结果显示对称性提示可将时间缩短至基准的约1/5或更少，几何先验亦有提升但幅度小；GrokFast表现不稳定。总体证明对称性提示最能加速grokking。

**⚠️ 局限性**

局限性：仅在人工合成算法任务上验证，缺乏真实世界或更复杂模型的验证；对称性定义需任务特定，迁移性未知；理论证明主要在模块化算术上，其他任务的数学证明不完整；实验受网络架构、初始化等因素影响，需进一步探索。

---

## 607. NM-DEKL$^3_\infty$: A Three-Layer Non-Monotone Evolving Dependent Type Logic

**arXiv ID:** 2603.01366 | [PDF](https://arxiv.org/pdf/2603.01366v1)

**作者:** Peng Chen `[一作]` `[通讯]`, Peng Chen

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种三层非单调演化依赖型逻辑（NM-DEKL³∞），用于形式化动态环境中的知识演化与因果推理。

**💡 创新点**

创新点在于将可变证据层、可逆限制（预变构造）与 μ-演算分离，提供完整的语义证明与初始模型构造，并展示对非双射不变属性的可表达性。

**🔧 技术方法**

采用依赖型类型理论、CwF、预变构造范畴语义、μ-演算嵌入与证明论工具（归约、正则化、初始性）。

**📊 数据集**

无实验数据集，论文为理论性框架与证明。

**📈 对比分析**

通过理论证明与对比（LTL/CTL、μ-演算），展示了表达能力严格提升，但缺乏可量化性能指标。

**⚠️ 局限性**

主要限制包括系统本身不可判定、缺乏可执行实例、对 μ-演算的固定点要求严格单调性，以及实现上的复杂度。

---

## 608. Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark

**arXiv ID:** 2603.00543 | [PDF](https://arxiv.org/pdf/2603.00543v1)

**作者:** Ke Cao `[一作]` (University of Science and Technology of China), Jie Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 30014 | [OpenAlex ID](https://openalex.org/A5100436868)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种跨尺度的高分辨率多光谱图像融合方法 ScaleFormer，并配套构建了全新的跨尺度遥感融合数据集 PanScale 与评测基准 PanScale-Bench。

**💡 创新点**

创新点包括：①将图像分辨率变化视作序列长度变化，解耦空间特征与尺度特征；②引入 Scale‑Aware Patchify (SAP) 和桶式采样以增强尺度泛化；③在序列注意力中使用 Rotary Position Embedding (RoPE) 提升对未见尺度的外推能力；④首次提供涵盖 200–2000 像素全尺度的真实遥感数据集。

**🔧 技术方法**

技术方法主要基于 Transformer 架构，包括 Spatial‑Transformer、Sequence‑Transformer、Cross‑Transformer、SAP 模块、RoPE、L1 损失等；同时采用多尺度桶式训练与多维度注意力。

**📊 数据集**

使用的数据集为 PanScale（包含三个子数据集，训练集、RR 与 FR 测试集，分辨率 200–2000 像素）以及 PanScale‑Bench 评测套件；在此基础上与 GS、IHS、GFPCA、MSDCNN、SFINet、MSDDN、PanFlowNet、HFIN、Pan‑mamba、ARConv 等 SOTA 方法进行对比。

**📈 对比分析**

方法通过 PSNR、SSIM、ERGAS、Q、Dλ、Ds、QNR 等指标与多种传统与深度学习基线进行比较，实验结果显示 ScaleFormer 在融合质量、尺度泛化、GFLOPs 与显存占用等方面均优于现有最先进方法。

**⚠️ 局限性**

局限性：①对极大尺度（超出训练分布）仍可能出现性能下降；②实验主要在合成或已标注数据上验证，真实场景中无真值的评估仍依赖无参考指标；③尽管参数量降低，但在 2000 像素级别仍需高端 GPU 进行推理。

---

## 609. DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent

**arXiv ID:** 2603.01152 | [PDF](https://arxiv.org/pdf/2603.01152v1)

**作者:** Tongzhou Wu `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6141 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了规模达9000条、分层难度的DeepResearch-9K深度研究基准，并提供开源训练框架DeepResearch‑R1。

**💡 创新点**

创新点在于结合多源多跳QA数据、实体逐步模糊化与自动化长链生成，产生高难度、可验证的多步搜索轨迹；同时提供完整训练流水线。

**🔧 技术方法**

采用自动化低成本流水线、RL（PPO/GRPO）、SFT+RL、LLM‑Judge奖励等技术。

**📊 数据集**

使用HotpotQA、2WikiMultihopQA、MuSiQue等公开多跳QA数据集合成。

**📈 对比分析**

在DeepResearch‑9K上与BrowseComp‑Plus、DeepSeek V3对比，教师模型在ℒ_3级别仅21%准确率，SFT+RL模型约20%准确率，显示任务极具挑战。

**⚠️ 局限性**

局限在于仍难突破约20%准确率，且生成任务主要基于文本推理，缺乏多模态或跨领域覆盖。

---

## 610. Seeing Beyond 8bits: Subjective and Objective Quality Assessment of HDR-UGC Videos

**arXiv ID:** 2603.00938 | [PDF](https://arxiv.org/pdf/2603.00938v1)

**作者:** Shreshth Saini `[一作]` (University of Texas at Austin), Alan C. Bovik `[通讯]` (University of Texas at Austin)

**通讯引用:** 141471 | [OpenAlex ID](https://openalex.org/A5075463806)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HDR-Q多模态大语言模型和Beyond8Bits大规模HDR UGC质量评估数据集，解决HDR视频的主观质量评估问题。

**💡 创新点**

创新点包括①构建最大规模的HDR-UGC主观数据集Beyond8Bits；②设计HDR-aware视觉编码器与HDR‑Aware Policy Optimization（HAPO）强化学习框架，其中包含HDR–SDR对比KL、双熵正则化和高熵权重分配；③实现可解释的链式推理（CoT）质量评估。

**🔧 技术方法**

采用多模态大语言模型（Ovis2.5、Qwen2.5‑VL）、SigLIP‑2视觉编码器微调、对比学习、GRPO改进的HAPO、SUREAL MOS聚合、HDR–SDR对比、双熵正则、HEW以及Gaussian MOS回报等技术。

**📊 数据集**

使用数据集包括44K HDR UGC视频与1.5M人工评分的Beyond8Bits；以及公开HDR VQA基准LIVE‑HDR和SFV+HDR。

**📈 对比分析**

在Beyond8Bits上与传统NR‑VQA、HDR‑VQA和MLLM VQA等13+基线对比，HDR‑Q在SRCC、PLCC和RMSE等指标上显著领先（SRCC≈0.92、RMSE≈5.16），并在LIVE‑HDR和SFV+HDR的零样本迁移中保持高性能。

**⚠️ 局限性**

局限性在于依赖HDR显示硬件与环境，极端压缩或噪声场景鲁棒性尚未充分验证，长时序推理效率有待提升，且HAPO训练成本相对较高。

---

## 611. Personal Health Data Integration and Intelligence through Semantic Web and Blockchain Technologies

**arXiv ID:** 2603.02192 | [PDF](https://arxiv.org/pdf/2603.02192v1)

**作者:** Oshani Seneviratne `[一作]` (Rensselaer Polytechnic Institute), Jianjing Lin `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

结合语义网与区块链技术，设计并实现了BlockIoT系统，实现在个人健康设备与电子健康记录系统（EHR）之间的去中心化、安全、实时数据互通，并通过智能合约自动生成临床警报与可视化分析；

**💡 创新点**

创新点在于：①将FHIR兼容的语义模板与IPFS+以太坊智能合约相结合，实现设备数据的标准化转换、分布式存储与可信访问；②提供统一的RESTful API层，解耦设备厂商与EHR，消除对单一EHR供应商的依赖；③通过智能合约实现基于阈值的即时警报与决策支持，提升慢性病管理效率；

**🔧 技术方法**

使用技术包括：FHIR（资源模型与API）、SNOMED-CT/OWL/SWRL（语义层）、IPFS与IPNS（去中心化存储）、以太坊区块链+Solidity智能合约、RESTful API、MQTT/CoAP/HTTPS通信协议、JSON配置与模板等；

**📊 数据集**

未公开具体公开数据集，系统演示使用多种个人健康设备（血糖仪、血压计、心率监测、体重秤等）收集的示例数据，并通过配置文件与模板实现对这些数据的语义化转换；

**📈 对比分析**

论文未给出正式的对比实验或性能基准；仅描述系统架构、功能模块与工作流程；未来可通过在真实临床环境或大规模设备部署中评估吞吐量、延迟、存储成本等指标；

**⚠️ 局限性**

主要局限包括：①缺乏统一的设备标准，导致配置文件与模板难以扩展；②法规合规与跨域数据隐私保护仍需进一步研究；③区块链与IPFS的网络性能与可扩展性尚未在大规模场景验证；④部署成本与运维复杂度较高。

---

## 612. ClinConsensus: A Consensus-Based Benchmark for Evaluating Chinese Medical LLMs across Difficulty Levels

**arXiv ID:** 2603.02097 | [PDF](https://arxiv.org/pdf/2603.02097v1)

**作者:** Xiang Zheng `[一作]` (Alibaba Group), Bing Zhao `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向中文医学的端到端评测基准 ClinConsensus，涵盖预防、治疗与长期管理的 2,500 条真实病例；

**💡 创新点**

创新点在于将真实临床工作流、开放式对话、多阶段多专业情境与专家级评判体系相结合，并提出了基于阈值的 Clinically Applicable Consistency Score（CACS@k）指标；

**🔧 技术方法**

使用了双评判框架（LLM-as-judge 与训练好的 SFT-judge）、分级 rubrics、阈值校准、监督微调（SFT）等技术；

**📊 数据集**

数据集为 ClinConsensus，包含 2,500 条多阶段、多专业、按难度分级的病例，并配备 30 条专家制定的评判准则；

**📈 对比分析**

通过与 15 种顶尖 LLM（如 GPT‑5.2、ERNIE‑5.0、Gemini‑3‑Pro 等）在 CACS@7 上进行对比，发现 ERNIE‑5.0、GPT‑5.2 等模型总体表现相近，但在不同任务主题、护理阶段和专业上存在显著差异；

**⚠️ 局限性**

局限性包括：评判仍受专家裁定与 LLM 评判一致性的影响；基准偏向中文医疗体系，跨语言迁移需要进一步验证；CACS@k 仅衡量阈值以上的一致性，未覆盖全部临床细节与多模态交互场景。

---

## 613. Silo-Bench: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems

**arXiv ID:** 2603.01045 | [PDF](https://arxiv.org/pdf/2603.01045v1)

**作者:** Yuzhe Zhang `[一作]` (Beijing University of Technology), Wenyuan Jiang `[通讯]` (ETH Zurich)

**通讯引用:** 375 | [OpenAlex ID](https://openalex.org/A5111260534)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个无角色限制的分布式协作基准，涵盖30个算法任务，按通信复杂度分为三层；对54种配置（3种模型×3种协议×6种规模）共进行1620次实验，系统评估多智能体LLM在信息孤岛中的协作效果。

**💡 创新点**

首创揭示了“通信‑推理缺口”（Communication‑Reasoning Gap），展示智能体能主动形成合适的通信拓扑，却在信息整合与全局推理阶段失误；提供了可公开复现的任务生成器、度量体系和多维评估指标，成为评估多智能体协作进展的基准。

**🔧 技术方法**

使用大语言模型（DeepSeek‑V3.1、GPT‑OSS‑120B、Qwen3‑Next‑80B‑A3B）作为智能体；三种通信协议（点对点P2P、广播BP、共享文件系统SFS）；通过自定义任务生成器构造LeetCode灵感的分布式算子；评估指标包括成功率（SR）、部分正确性（PCS）、令牌消耗（C）与通信密度（D）。

**📊 数据集**

任务数据通过Python生成器自动构造，基于LeetCode算法问题（如最大值、前缀和、分布式排序等），按通信复杂度层级分布，保证每个智能体仅拥有局部输入。

**📈 对比分析**

对比方法：三模型、三协议、六规模的全因子设计；实验显示平均成功率仅36.9%，最高模型DeepSeek在最易层级的SR仍不足70%；与单智能体基线相比，协调成本（RCC）随规模急剧上升，级别III任务在N≥50时完全失去并行优势。

**⚠️ 局限性**

局限性：仅评估三种基本通信协议，未覆盖层级化、gossip等更复杂机制；所有智能体使用同一模型，缺乏异构实验；未评估闭源大模型；基准仅涵盖当前前沿模型，可能无法捕捉所有LLM的失败模式。

---

## 614. Linking Modality Isolation in Heterogeneous Collaborative Perception

**arXiv ID:** 2603.00609 | [PDF](https://arxiv.org/pdf/2603.00609v1)

**作者:** Changxing Liu `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8746 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CodeAlign，一个在多模态孤立情形下实现协同感知的对齐框架。

**💡 创新点**

创新点是利用可学习的代码本构建每种模态的离散代码空间，通过特征-代码-特征(FCF)翻译实现跨模态对齐，无需共现数据。

**🔧 技术方法**

采用代码本、投影器、跨模态翻译器、重建器以及轻量一对多翻译器，配合稀疏 L1 损失和相似性损失。

**📊 数据集**

在 OPV2V（模拟 V2V）和 DAIR‑V2X（真实车站‑车协同）数据集上进行评测。

**📈 对比分析**

与晚期融合、Pyramid Fusion、HMViT、CodeFilling、HEAL 等基线相比，CodeAlign 在多模态孤立场景下 AP30/50/70 均表现最佳，训练参数仅 8%，通信量缩小 1024 倍，且鲁棒性优于 HEAL。

**⚠️ 局限性**

局限性：实验仅覆盖有限模态种类，未在更大规模多模态群体中验证；在极端光照/天气下性能未充分评估。

---

## 615. SEAnet: A Deep Learning Architecture for Data Series Similarity Search

**arXiv ID:** 2603.01448 | [PDF](https://arxiv.org/pdf/2603.01448v1)

**作者:** Qitong Wang `[一作]` (Universite Paris Cite), Themis Palpanas `[通讯]` (Universite Paris Cite)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出使用深度学习嵌入（Deep Embedding Approximation, DEA）进行时间序列相似性搜索，设计了SEAnet与SEAtrans两种自编码器架构，并给出了基于可排序摘要InvSAX的采样策略SEAsam与SEAsamE，形成完整的近似搜索框架；

**💡 创新点**

创新点包括①将深度学习嵌入应用于相似性搜索；②提出SEAnet与SEAtrans两种架构，结合残差、指数扩张卷积与Transformer块，并引入Sum of Squares (SoS) 保持原始方差；③设计SEAsam与SEAsamE采样策略，在海量数据上实现高效训练；

**🔧 技术方法**

技术手段：全预激活ResNet卷积编码器、Transformer TransBlock、SoS保持正则化、PAA/SAX离散化、iSAX+MESSI索引、距离压缩与重建损失、InvSAX可排序摘要、采样与重采样；

**📊 数据集**

实验数据集包括三种合成（RandWalk、F5、F10）和四种真实（SALD、Deep1B、Seismic、Astro），长度分别为128/96/256，规模从1M到100M条；

**📈 对比分析**

评估方法：平均距离差、重建RMS、kNN覆盖率、1st BSF紧致度、叶节点紧凑性；与传统PAA、FDJNet、TimeNet、InceptionTime以及随机采样进行对比。结果显示SEAnet/SEAtrans在所有指标上优于传统方法，尤其在难度高的数据集上取得显著提升；SEAsam优于随机采样，SEAsamE对某些模型进一步提升；

**⚠️ 局限性**

局限性：当前仅支持欧氏距离，缺乏严格的下界保证；未实现增量学习或迁移学习；需要进一步探索索引结构与压缩技术；实验受限于所选数据集，需验证更广泛场景。

---

## 616. Knowledge without Wisdom: Measuring Misalignment between LLMs and Intended Impact

**arXiv ID:** 2603.00883 | [PDF](https://arxiv.org/pdf/2603.00883v1)

**作者:** Michael Hardy `[一作]` (Stanford University), Yunsung Kim `[通讯]` (Stanford University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5069435977)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过给多种主流大型语言模型（LLM）提供小学数学课堂转录文本，要求它们按教学质量维度打分，并将模型评分与人工专家评估以及学生学习增益（VAM）进行对齐度量，评估LLM在实际教育下游任务中的表现。

**💡 创新点**

创新点包括：①首次使用真实课堂转录和VAM作为最终影响度量，直接把LLM评估与学生学习成果关联；②采用距离相关与Kendall τ相结合的鲁棒对齐评估框架，避开绝对评分噪声；③对模型误差进行方差分解，量化模型选择与提示工程对对齐的可控性；④系统性检验集成（加权与全票一致）对对齐的影响，发现集成反而加剧误差。

**🔧 技术方法**

技术手段主要有：bias‑corrected squared distance correlation (dCor²ₙ) 用于衡量非线性依赖；Kendall τ 用于方向性对齐评估；Generalizability Theory/随机效应模型进行误差方差分解；多模型、多提示的实验设计；链式思维与推理提示无显著提升。

**📊 数据集**

数据集：美国 NCTE Main Study 中的 4‑5 年级数学课堂转录（约1,600 篇），对应的 MQI 与 CLASS 专家评分，及每位教师每学年计算的 VAM（学生学习增益）。

**📈 对比分析**

比较方法：对每个模型在七个教学维度上与专家评分、VAM 的 pairwise concordance 进行 Kendall τ 计算；同时绘制对齐散点图比较模型与专家评分的对齐与对齐与学习成果的对齐关系。结果显示：LLM 与专家评分对齐度高，但与 VAM 对齐度低甚至负相关；集成方法（专家权重或全票一致）未能改善，往往进一步降低与学习成果的对齐；模型与提示的可控性仅解释约 5% 的误差，其余 95% 来自共享的预训练偏差。

**⚠️ 局限性**

局限性：①人类评分本身可靠性有限，影响对齐评估；②研究仅涵盖美国 4‑5 年级数学课堂，难以推广至其他学段、学科或国家；③课堂转录数据稀缺且不易获取，限制了模型训练与评估的可复现性；④LLM 在教学语言上的偏差源于预训练语料缺失真实教育对话，导致系统性误差；⑤本文未探索更深层次的模型微调或多模态信息融合，未来工作需扩展。

---

## 617. QQ: A Toolkit for Language Identifiers and Metadata

**arXiv ID:** 2603.00620 | [PDF](https://arxiv.org/pdf/2603.00620v1)

**作者:** Wessel Poelman `[一作]` (KU Leuven), Miryam de Lhoneux `[通讯]` (KU Leuven)

**通讯引用:** 678 | [OpenAlex ID](https://openalex.org/A5080895973)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 QwanQwa（QQ），一个轻量级 Python 工具包，用于统一多源语言元数据、标准化语言标识符、构建可遍历的语言图谱，并支持多语言 NLP 研究中的数据管理与分析。

**💡 创新点**

创新点包括：① 将 LinguaMeta、Glottolog、Wikipedia/Wikidata 等多种来源的元数据统一为单一图结构；② 采用中性“languoid/region”抽象模型，避免政治化分类；③ 提供统一的标识符映射与冲突解析策略；④ 将图遍历、属性查询和跨语言信息检索集成到一个 Python API 中。

**🔧 技术方法**

技术实现主要依赖 Python，利用图数据库结构（节点：languoid、script、region，边：family、region、script 等关系）以及自定义的导入器对多源数据进行抽取、归一化、冲突解决；还使用了标准库如 pycountry、langcodes 进行编码解析与转换，配合 Rayleigh 商、置换检验等统计方法进行实验评估。

**📊 数据集**

使用的数据集包括：LinguaMeta、Glottolog、Wikipedia/Wikidata、Glotscript、IANA、SIL、Huggingface 数据集、BabelNet、Concepticon、NoRaRe 等，涵盖 7000+ 语言、4500+ 家族、12000+ 方言、230+ 文字和 5000+ 区域。

**📈 对比分析**

通过三大案例研究评估：① 在 Huggingface 数据集上进行标识符使用审计，展示 QQ 能快速识别合法/过时/未知代码；② 在生成 LaTeX 语言元数据表格时验证 QQ 的自动化与准确性；③ 在构建跨语言共现化概念图并计算 Rayleigh 商、置换检验，证明 QQ 能高效完成图结构构建、属性查询和统计分析；整体数据库体积约 2 MB，查询速度快，性能满足日常 NLP 研究需求。

**⚠️ 局限性**

局限性：① 数据来源仍可能存在错误、冲突或过时信息，QQ 的冲突处理可能导致不一致；② 对历史语言和方言的覆盖依赖于 Glottolog，尚未完全解决所有缺失；③ Huggingface 审计仅基于 API 元数据，未覆盖数据文件内部多语言列；④ QQ 的设计强调中性分类，某些应用场景可能需要进一步自定义层次或政治语境。

---

## 618. Local Differential Privacy for Molecular Communication Networks

**arXiv ID:** 2603.00690 | [PDF](https://arxiv.org/pdf/2603.00690v1)

**作者:** Melih Şahin `[一作]` (University of Cambridge), Ozgur B. Akan `[通讯]` (Koç University)

**通讯引用:** 11236 | [OpenAlex ID](https://openalex.org/A5042130660)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在基于扩散的分子通信网络中实现本地差分隐私，并对常用 LDP 机制进行基准测试。

**💡 创新点**

首次系统地在 MC 信道下评估 LDP 机制，发现 OLH 与 KRR 在资源不同情况下表现差异，并提出结合 RLIM 编码的 RLIM-LDP 框架以提升可靠性。

**🔧 技术方法**

使用本地差分隐私（KRR、RAPPOR、OUE、BLH、OLH、HR）、扩散 MC 信道模型、RLIM (2,∞)-RLL 编码、模拟仿真。

**📊 数据集**

利用 100 个随机 k-元概率分布（从均匀 Dirichlet 采样）和 N=10^4 用户的合成数据进行实验。

**📈 对比分析**

通过平均 ℓ1 误差比较不同 LDP 机制，结果表明 OLH 在资源充足、字母表中等至大时误差最低；KRR 在信道质量下降时更稳健；RLIM‑LDP 在短信号间隔和有限分子预算时显著降低误差。

**⚠️ 局限性**

仅在仿真环境下验证，未考虑真实生物信道噪声和设备硬件限制；RLIM 编码增加了实现复杂度；仅评估了有限的 LDP 机制和参数。

---

## 619. Operator Learning Using Weak Supervision from Walk-on-Spheres

**arXiv ID:** 2603.01193 | [PDF](https://arxiv.org/pdf/2603.01193v1)

**作者:** Hrishikesh Viswanath `[一作]` (Purdue University), Aniket Bera `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 WoS-NO 的神经算子框架，利用 Walk‑on‑Spheres (WoS) 的弱监督来训练无网格、无预计算数据的 PDE 解算子，直接学习参数化 Poisson 型线性椭圆方程的解映射。

**💡 创新点**

创新点在于：
• 通过 WoS 产生的无偏但高方差的 Monte‑Carlo 估计作为弱监督，避免了传统 PINN 中的高阶导数计算和梯度不稳定问题；
• 将弱监督与神经算子结合，构建数据自由、物理信息一致的学习目标；
• 通过对整个 PDE 族的随机走路进行方差降，实现在单前向推断中对未见几何、边界、源项的零样本泛化；
• 兼容多种神经算子架构（GINO、Transolver、GNOT 等），展示其通用性。

**🔧 技术方法**

使用技术包括：
• Walk‑on‑Spheres Monte‑Carlo 估计与 delta‑tracking（屏蔽 Poisson）;
• 神经算子架构（如 GINO、Transolver、GNOT 等）；
• 弱监督损失函数（WoS 估计的平方误差）;
• 训练时对 WoS 步数控制与缓存，降低方差；
• 对不同 PDE 参数化进行统一表示（几何、源、边界）。

**📊 数据集**

数据集与场景：
• ShapeNet 的无水密网格用于生成训练/测试几何；
• 随机生成的源项、边界条件参数；
• 进一步评估在 2D/3D 场景中的零样本泛化：Biharmonic 图像修复、von Kármán vortex 场景以及 Poisson 表面重建等。

**📈 对比分析**

与基线的比较方法：
• 在相同训练步数或相同训练时间下，对比 PINO、DeepRitz Operator、传统 WoS；
• 评估指标包括 L₂ 误差、训练时间、峰值 GPU 内存与功耗；
• 结果显示 WoS‑NO 在 L₂ 误差上比 PINO 提升 8.75×，训练速度提升 6.31×，GPU 内存减少 2.97×；
• 在多几何零样本推断时，WoS‑NO 相比 WoS 提升 3.73×，相对 PINO 2.1×，DeepRitz 1.59×；
• 在空间可变系数 Poisson 和 ShapeNet 评估中也保持领先。

**⚠️ 局限性**

局限性：
• 目前实验仅覆盖线性 Poisson / 屏蔽 Poisson 类方程，未验证对非线性、Neumann 边界等更广泛 PDE 的泛化；
• WoS 估计高方差，仍需一定走路步数以获得可接受误差，对极大尺度或极慢收敛问题仍需额外计算；
• 对高阶 PDE 需先拆分为二阶系统，增加模型使用复杂度；
• 对极端几何形状、极端源项变化时的收敛理论和稳定性尚未完全阐明。

---

## 620. Exploring 3D Dataset Pruning

**arXiv ID:** 2603.00651 | [PDF](https://arxiv.org/pdf/2603.00651v1)

**作者:** Xiaohan Zhao `[一作]` (MBZUAI), Zhiqiang Shen `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了3D数据集的样本剪枝，解决长尾分布下整体准确率（OA）与均衡准确率（mAcc）冲突的问题。

**💡 创新点**

创新点在于：①通过四分曲线误差分解揭示了 OA 与 mAcc 共享的优化方向；②提出基于校准软标签与嵌入几何蒸馏的先验不匹配消除方法；③引入安全底线与可调 steering 参数实现对两种评价指标的灵活权衡。

**🔧 技术方法**

采用的技术包括校准软标签（Calibrated Soft Labels）、嵌入几何蒸馏（RKD）、基于嵌入几何的全局选择与安全底线（SGS）以及可调 steering 包装器；同时利用知识蒸馏、类权重重采样等。

**📊 数据集**

主要使用的3D数据集有 ShapeNet55、ModelNet40、ScanObjectNN（点云）以及 MeshNet（网格）。

**📈 对比分析**

与多种基线方法（Loss、GradNorm、EL2N、Entropy、Herding、FL‑RBF、DRoP、NUCS、CCS 等）比较，在 OA 与 mAcc 上均表现出显著提升，尤其在 mAcc 上显著优于现有方法；在高压缩率下优势更为明显。

**⚠️ 局限性**

局限性包括：对极度不平衡类的处理仍需要更多样本；蒸馏过程对教师模型的依赖较大；在低预算或非点云模态（如图像、文本）迁移效果尚待进一步验证。

---

## 621. CoopDiff: A Diffusion-Guided Approach for Cooperation under Corruptions

**arXiv ID:** 2603.01688 | [PDF](https://arxiv.org/pdf/2603.01688v1)

**作者:** Gong Chen `[一作]` (Tianjin University), Pengcheng Lv `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 CoopDiff 框架，通过教师‑学生扩散模型实现对协同感知中多源噪声的自适应去噪与信息融合，提升鲁棒性；

**💡 创新点**

创新点在于结合质量感知早期融合教师、双分支去噪学生与跨代理注意力机制，使用 Gated Conditional Modulation 与 Cooperative Deformable Attention，实现对多类型噪声的统一去噪与协同；

**🔧 技术方法**

采用扩散模型、质量感知权重、Gated Conditional Modulation、Cooperative Deformable Attention、Ego‑Guided Cross‑Attention、知识蒸馏等技术；

**📊 数据集**

在 OPV2V、DAIR‑V2X 及其扩展的多噪声版本 OPV2Vn 与 DAIR‑V2Xn 上进行实验；

**📈 对比分析**

与多种 SOTA（V2X‑ViT、DSRC、MRCNet 等）在清洁与六类噪声条件下对比，CoopDiff 在所有场景下均取得最高 AP，并在 mRCE 上最低 12.94%/26.79%，表明显著提升鲁棒性；

**⚠️ 局限性**

限制在于扩散模型可能产生与真实几何不完全对齐的伪物体，且在极端噪声下的收敛速度与推理时延仍有提升空间。

---

## 622. China leads scientific trends; the West launches new ones

**arXiv ID:** 2603.01117 | [PDF](https://arxiv.org/pdf/2603.01117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 623. MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline

**arXiv ID:** 2603.01050 | [PDF](https://arxiv.org/pdf/2603.01050v1)

**作者:** Huanjin Yao `[一作]` (ByteDance), Jiaxing Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 70244 | [OpenAlex ID](https://openalex.org/A5100355322)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了MM-DeepResearch，一种具备显式推理、规划、多工具调用与跨模态信息合成的多模态深度研究代理，并通过离线训练实现了从零开始的智能搜索能力。

**💡 创新点**

创新点包括：①基于超图的Hyper-Search方法，用以生成需要多轮多模态搜索的QA数据；②Decompose–Recompose Tool Tree Search (DR‑TTS)框架，拆解任务为单工具专家并再组合成树搜索，以产生高质量搜索轨迹；③构建离线多模态搜索引擎，消除在线API成本，支持大规模强化学习。

**🔧 技术方法**

采用的技术包括：超图建模与节点扩展、LLM生成描述与摘要、离线文本与视觉检索（E5、Jina‑CLIP+FlashRAG）、强化学习（GRPO）与多轮工具调用、SFT与多模态上下文扩展。

**📊 数据集**

主要数据集为：Hyper‑Search‑3K（自制搜索密集型QA）、InfoSeek、FVQA、SimpleVQA、MM‑Search等基准；同时构建了包含图像与文本的离线检索语料库。

**📈 对比分析**

实验将MM‑DeepResearch与多类基线（非代理LLaMA、RAG工作流、现有代理如MMSearch‑R1、WebWatcher、SenseNova‑MARS等）在六个信息密集型基准上进行对比。结果显示，8B模型在MM‑Search上平均提升约17%，32B模型提升约14.9%，并在SimpleVQA、MM‑Search等任务上分别提升至65.9/67.6分，显著优于同类方法。

**⚠️ 局限性**

局限性主要体现在：①需要构建庞大的离线检索语料库，仍有构建成本；②对极端专业知识或实时更新信息的覆盖有限；③模型仍可能产生幻觉或误检，需进一步改进检索精度与答案验证机制。

---

## 624. Path Integral Particle Filtering for Hybrid Systems via Saltation Matrices

**arXiv ID:** 2603.01176 | [PDF](https://arxiv.org/pdf/2603.01176v1)

**作者:** Karthik Shaji `[一作]` (Georgia Institute of Technology), Yongxin Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7777 | [OpenAlex ID](https://openalex.org/A5066940107)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对含有间歇接触的混合动力系统的盐化路径积分粒子滤波（SPIPF/HPIPF）算法，用于在非高斯噪声下进行状态估计。

**💡 创新点**

创新点在于将路径积分滤波与混合系统的盐化矩阵（saltation matrix）结合，利用最优控制与滤波的对偶关系实现对跳跃事件中的不确定性传播建模，并通过滑动窗口与投票逻辑降低粒子权重退化，提高对接触事件的鲁棒性。

**🔧 技术方法**

采用的技术包括：路径积分粒子滤波框架、最优控制对偶、盐化iLQR算法、盐化矩阵、Girsanov定理、KL散度、滑动窗口与投票机制以及多模式SIR滤波的基线对比。

**📊 数据集**

实验数据集为仿真生成的两种混合系统：一维弹跳球系统和弹簧负载倒立摆（SLIP）系统，噪声均为高斯分布。

**📈 对比分析**

通过与基线多模式SIR粒子滤波器以及零控制版本的对比，评估指标为均方误差（MSE）和有效样本大小（ESS）。实验显示SPIPF在粒子数、滑动窗口长度及时间步长等参数下均显著低于基线的MSE，并保持较高的ESS，尤其在接触事件后能更好地预测模式并抑制权重退化。

**⚠️ 局限性**

局限性包括：对滑动窗口长度和重采样阈值的敏感性；对接触阈值检测的依赖导致在不稳定系统中易出现提前触发或延迟触发；在高维或复杂多模态系统中计算量仍较大，需进一步并行化；以及实验仅在仿真环境下验证，缺乏真实硬件验证。

---

## 625. Partition-based Simple Heaps

**arXiv ID:** 2603.01206 | [PDF](https://arxiv.org/pdf/2603.01206v1)

**作者:** Gerth Stølting Brodal `[一作]` (Aarhus University), Sebastian Wild `[通讯]` (University of Marburg)

**通讯引用:** 1166 | [OpenAlex ID](https://openalex.org/A5071263179)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了一类基于分区的简单堆（partition‑based simple heaps）实现优先队列，使用双向链表和基于分区的插入、删除最小、减键操作，能够在摊销意义下以线性时间 O(n) 完成这些基本操作。

**💡 创新点**

创新点在于：①通过“pivot‑forgetting”规则和简单的分区、合并策略，构造出既保持有序性又极简的结构；②不需要复杂的树形结构或额外的数组索引，完全基于线性数据结构；③提出了三种变体（lazy partition、next‑generation Fibonacci 以及指数上界堆），分别利用不同的大小约束和重构规则，实现同样的摊销线性性能。

**🔧 技术方法**

主要技术包括：双向链表实现集合、二分搜索维护 pivots、线性时间选择算法（median）进行集合划分、常数时间合并（列表拼接）、潜能分析（potential function）证明摊销线性时间、Fibonacci 数字约束（在 FH:TNG 变体中）以及指数上界约束（在 exponential 变体中）等。

**📊 数据集**

该工作属于理论分析范畴，没有使用实际数据集；所有结果均基于理论算法分析与潜能方法给出的摊销时间。

**📈 对比分析**

与传统的 Fibonacci 堆、pairing 堆等经典结构相比，虽然基本操作仍为 O(n)（而非 O(log n) 或 O(1) 的 Fibonacci），但实现极其简洁、指针使用率低、局部性更好；作者提到若将 pivots 存储在高效的搜索结构（如 fusion tree）中，插入/减键的摊销时间可进一步降低至 O(1)。

**⚠️ 局限性**

限制主要包括：摊销时间仍为线性，无法达到 Fibonacci 堆的 O(log n) 删除最小和 O(1) 插入/减键的性能；需要在每次操作中对 pivots 进行二分搜索，若 pivots 数量巨大会带来较大常数；meld（合并）操作不具备高效支持；实验评估尚未完成，实际常数和外部存储适应性未经过实证验证。

---

## 626. Beyond Microservices: Testing Web-Scale RCA Methods on GPU-Driven LLM Workloads

**arXiv ID:** 2603.02057 | [PDF](https://arxiv.org/pdf/2603.02057v1)

**作者:** Dominik Scheinert `[一作]` (logsight.ai GmbH), Odej Kao `[通讯]` (Technische Universität Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文评估并比较了24种传统微服务Root Cause Analysis（RCA）方法在GPU驱动的LLM推理服务中的效果；

**💡 创新点**

创新点在于首次系统性地验证这些RCA技术在LLM特有的硬件、调度和批处理环境下的适用性，并给出了针对LLM推理的可观测性与RCA改进建议；

**🔧 技术方法**

采用了基于Kubernetes、Ray、vLLM、NVIDIA GPU Operator、Prometheus、Grafana、DeepFlow等云原生技术栈，以及ChaosMesh进行故障注入；

**📊 数据集**

使用自定义的LLM推理服务（Falcon-H1-7B-Instruct）与人工注入的四类故障（CPU Hog、Memory Leak、Network Latency、GPU Throttling）作为实验数据集；

**📈 对比分析**

通过AC@k和Avg@k指标比较，发现多源方法（如MM-BARO、PDiagnose）与先进的度量基方法（NSigma、BARO、CIRCA、RCD）在识别根因时取得最高准确率，而基于追踪的方法整体表现较差；

**⚠️ 局限性**

局限性包括只测试单一故障注入、未覆盖更复杂的多故障情形、对特定LLM推理栈的依赖、以及缺少对模型级别指标（如TTFT、token速率）的深入分析。

---

## 627. WhisperNet: A Scalable Solution for Bandwidth-Efficient Collaboration

**arXiv ID:** 2603.01708 | [PDF](https://arxiv.org/pdf/2603.01708v1)

**作者:** Gong Chen `[一作]` (Tianjin University), Xinyan Zhao `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种接收器中心的协同感知框架 WhisperNet，能够在极低带宽条件下实现高精度三维目标检测与 BEV 语义分割。

**💡 创新点**

创新点在于将空间与通道维度的冗余性联合考虑，并通过接收方全局协同分配来决定“什么”和“何处”传输，实现高效的全局协调与精细通道选择。

**🔧 技术方法**

采用轻量化的发送端重要性估计（空间通道重要性图）、接收端信心感知模块（全局请求计划与预算分配）、协同特征路由模块（专家分组与通道对齐）等技术，并利用 Laplacian 高频特征评分。

**📊 数据集**

在大规模真实交通数据集 OPV2V 与 DAIR‑V2X 上进行实验，评估三维检测与 BEV 分割。

**📈 对比分析**

与多种基准方法对比，WhisperNet 在 OPV2V 上 AP@0.7 达到 93.34%，比第二名提升约 1.5%；在 1% 带宽下相对最优方法提升 10.6%/11.7%；同时实现 95% 带宽压缩，且在定位噪声下鲁棒性优于现有方案。

**⚠️ 局限性**

局限性包括：仍需全局信息交换（对极大规模车队的可扩展性尚未验证），以及在极低延迟场景下路由与预算计算可能产生的计算开销。

---

## 628. Beyond Length Scaling: Synergizing Breadth and Depth for Generative Reward Models

**arXiv ID:** 2603.01571 | [PDF](https://arxiv.org/pdf/2603.01571v1)

**作者:** Qiyuan Zhang `[一作]` (City University of Hong Kong), Chen Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 27782 | [OpenAlex ID](https://openalex.org/A5100652421)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Mix-GRM 框架，将评估推理拆分为宽度 CoT（Breadth-CoT）与深度 CoT（Depth-CoT），并通过自适应机制对齐不同任务的推理方式。

**💡 创新点**

创新点在于：① 采用两种互补的 CoT 结构而非单纯扩大长度；② 通过模块化标准化与合成管道实现原子化推理单元；③ 在 SFT+RLVR 阶段让模型自发聚焦合适的推理机制，实现“切换放大”效应。

**🔧 技术方法**

技术手段包括：模块化方案标准化（Principle–Judgment–Verdict 单元）；B-CoT 与 D-CoT 合成；监督微调（SFT）和基于 RL 的奖励可验证训练（RLVR via GRPO）；使用 Qwen3‑8B 作为基础模型。

**📊 数据集**

主要使用的评测数据集有 RewardBench、RewardBench‑v2、RMB、RM‑Bench、PPE 以及各种任务子集（数学、代码、开放式聊天等），共约 30,000 条样本（9K SFT + 21K RLVR）。

**📈 对比分析**

与七个主流 RMs（包括 Skywork‑Reward、JudgeLRM、RM‑R1、FARE、RubricRM、DeepSeek‑GRM 等）对比，Mix‑GRM 在五大基准上平均提升 8.2%（SFT 阶段），RLVR 后进一步提升 4.3%，并在 Offline RL 与 Test‑time Scaling 上获得最佳表现。

**⚠️ 局限性**

局限性在于：① 只捕捉了偏好与正确性两条主轴，未能细粒度划分更丰富的推理维度；② 对跨域混合任务的适应性有限，模型在极端多模态或跨领域场景中可能失去灵活性。

---

## 629. Adam Converges Without Any Modification On Update Rules

**arXiv ID:** 2603.02092 | [PDF](https://arxiv.org/pdf/2603.02092v1)

**作者:** Yushun Zhang `[一作]` (Chinese University of Hong Kong), Ruoyu Sun `[通讯]` (Shenzhen Research Institute of Big Data)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探究Adam优化器在不同参数设置下的收敛与发散行为，并给出相应的理论阈值；

**💡 创新点**

首次系统证明了当β₂接近1时，Adam能在任何β₁<√β₂的条件下收敛，而当β₂较小时，Adam在广泛的β₁范围内会发散到无穷大；

**🔧 技术方法**

主要使用了随机收敛分析、Lipschitz连续性假设、梯度分布的“可变方差”条件、期望与条件期望的递推公式以及尾和、递归不等式与概率上界等理论工具；

**📊 数据集**

在实验验证中使用了常规深度学习数据集MNIST、CIFAR-10和ImageNet；

**📈 对比分析**

与传统SGD、RMSProp等优化器相比，Adam在β₂足够大的安全区内收敛速度更快，且对β₁与β₂的调参给出了更稳健的建议；

**⚠️ 局限性**

局限性包括理论阈值和收敛速度常数的阶数不紧，未给出最优β₁、β₂组合，也未对Adam相对于SGD的速度优势给出定量证明；

---

## 630. Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models

**arXiv ID:** 2603.01400 | [PDF](https://arxiv.org/pdf/2603.01400v1)

**作者:** Jinlong Li `[一作]` (University of Trento), Nicu Sebe `[通讯]` (University of Trento)

**通讯引用:** 35970 | [OpenAlex ID](https://openalex.org/A5027171279)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了基于最优传输（OT）的AOT（Local-Global Token Anchor + OT）方法，用于在视频大语言模型中对视觉令牌进行训练无关的压缩与聚合，显著减少冗余并保留重要时空语义。

**💡 创新点**

创新点在于：①构建局部-全局两级令牌锚点以保留多样且语义重要的特征；②利用OT全局优化将被剔除或合并的令牌上下文迁移到锚点，实现信息聚合而非简单丢弃；③同时在帧内和帧间两层应用OT，实现完整的时空压缩；④所有步骤无须额外训练，仅通过Sinkhorn迭代即可完成。

**🔧 技术方法**

技术手段包括：最优传输（Wasserstein）与Sinkhorn‑Knopp迭代求解、局部-全局令牌锚点选择、帧内OT聚合、帧间OT稀疏更新、动态聚类帧剪辑，以及对LLaVA OneVision/Video 7B模型的无训练压缩实现。

**📊 数据集**

实验数据集涵盖MVBench、EgoSchema、LongVideoBench和VideoMME四大视频理解基准；模型采用LLaVA-OneVision‑7B和LLaVA‑Video‑7B两款视频大语言模型。

**📈 对比分析**

与FastV、PDrop、VisionZip、DyCoke、PruneVid、FastVID等主流压缩方法以及其动态分段版本进行对比；AOT在保持97.6%原模型性能的同时，将FLOPs压缩至8.3%，在各基准上均达或超越现有方法，甚至在部分场景下超过未压缩模型。

**⚠️ 局限性**

局限性包括：①帧间OT聚合仍为经验性、无理论上优的时间锚点构造；②动态帧划分可能导致视觉相似性不足的帧被聚在同一剪辑；③目前仅训练无关，未探索OT与模型微调或指令微调的联合优化；④对极长视频或复杂场景的鲁棒性仍待验证。

---

## 631. Tiny-Critic RAG: Empowering Agentic Fallback with Parameter-Efficient Small Language Models

**arXiv ID:** 2603.00846 | [PDF](https://arxiv.org/pdf/2603.00846v1)

**作者:** Yichao Wu `[一作]` (Northeastern University), Weiran Yan `[通讯]` (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Tiny-Critic RAG，将评估任务交给参数高效的小型语言模型，实现低延迟二进制路由

**💡 创新点**

通过LoRA微调SLM做判别器并结合非思考模式与受限解码，显著降低评估开销并保持与大型LLM相近的路由准确性

**🔧 技术方法**

LoRA、受限解码、非思考推理模式、模型上下文协议（MCP）

**📊 数据集**

Natural Questions、HotpotQA，并在此基础上注入45%的对抗噪声

**📈 对比分析**

与Naive RAG和Heavy‑CRAG对比，Tiny‑Critic在路由F1达0.912、TTFT仅42 ms、CPQ 0.06 美元，显著降低成本与延迟，同时保持近似相同的鲁棒性

**⚠️ 局限性**

仅做二进制路由，难以处理更细粒度评估；对模型固有偏见的修正仍需进一步研究

---

## 632. Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models

**arXiv ID:** 2603.01766 | [PDF](https://arxiv.org/pdf/2603.01766v1)

**作者:** Haoyun Liu `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Neural Implicit Action Fields (NIAF)，通过把机器人动作从离散轨迹点转化为可微的连续时间动作函数，实现高分辨率、可解析的动作预测。

**💡 创新点**

创新点在于：①采用 SIREN 作为隐式神经表示实现 C^∞ 平滑连续动作；②将大语言模型(MLLM)作为层级频谱调制器，生成可调的动作隐式网络参数；③利用解析可导性进行高阶动力学监督（速度、加速度、jerk），从而实现精确阻抗控制。

**🔧 技术方法**

技术包括：多模态大型语言模型、隐式神经表示（SIREN）、层级频谱调制（Hyper‑Modulation）、解析动力学监督、阻抗控制算法。

**📊 数据集**

使用 CALVIN、LIBERO 这两个标准仿真基准以及在真实机器人（AgileX Piper 与 AgileX Cobot Magic）上收集的演示数据进行训练与评估。

**📈 对比分析**

与 BEAST、OFT、FAST、Qwen3‑VL 等基线对比，NIAF 在 CALVIN、LIBERO 上均取得最高成功率，并在真实机器人上显著提升了轨迹平滑度和阻抗控制性能，减少了控制抖动。

**⚠️ 局限性**

局限性包括：在仿真环境中缺乏高阶动力学反馈导致难以充分验证动力学监督；模型对 MLLM 的依赖使得推理成本相对较高；在极端复杂或高频动态任务中，隐式网络的解析速度与硬件实时性可能成为瓶颈。

---

## 633. Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents

**arXiv ID:** 2603.01548 | [PDF](https://arxiv.org/pdf/2603.01548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 634. AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching

**arXiv ID:** 2603.01006 | [PDF](https://arxiv.org/pdf/2603.01006v1)

**作者:** Pengfei Zhang `[一作]` (Hong Kong University of Science and Technology), Li Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 33025 | [OpenAlex ID](https://openalex.org/A5100418903)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于因果归因的 REPA（AG-REPA）方法，利用前向门消融（FoG-A）识别并对齐音频流匹配模型中对速度场贡献最大的层，从而加速并提升音频生成质量。

**💡 创新点**

创新点在于揭示“存储-贡献解耦”（Store‑Contribute Dissociation，SCD），并将因果归因结果用于自适应层选择与加权，对比传统固定层 REPA，显著提升了训练效率与生成效果。

**🔧 技术方法**

主要技术包括：Bi‑Stream Teacher Cosine Alignment（BiT‑C）进行双教师对齐；Layer‑wise Analysis via Shared Projection（LASP）评估层级信息存储；前向门消融（FoG‑A）量化层级因果贡献；以及基于 FoG‑A 的 AG‑REPA 训练框架。

**📊 数据集**

实验数据集包括统一的语音生成任务（LibriSpeech）和通用音频生成任务（AudioSet），并在三种不同的 Token‑topology（S^3 + AudioSet、加密 BEATs）下进行评估。

**📈 对比分析**

与传统 REPA（单层或多层固定层）及随机/LASP 选层相比，AG‑REPA 在 FAD（Fréchet Audio Distance）上分别减少 18%（语音）和 16%（通用音频），同时 WER 下降至 3.45、MOS 提升至 4.12；在训练步骤上也快 3.3 倍，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：需要预先计算 FoG‑A，增加一次性计算开销；对极深模型或极大数据集的可扩展性尚待验证；并且在某些强大基线（如 CosyVoice）中提升幅度相对较小。

---

## 635. Milliscale: Fast Commit on Low-Latency Object Storage

**arXiv ID:** 2603.02108 | [PDF](https://arxiv.org/pdf/2603.02108v1)

**作者:** Jiatang Zhou `[一作]` (Simon Fraser University), Tianzheng Wang `[通讯]` (Simon Fraser University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个内存优化的OLTP引擎，直接使用低延迟可变对象存储（如Amazon S3）作为事务日志，降低提交延迟。

**💡 创新点**

创新点包括受限去中心化日志（按线程组共享日志缓冲区，减少S3 append请求）和记录级依赖跟踪（避免伪依赖，降低尾部延迟）。

**🔧 技术方法**

技术手段：基于ERMIA的内存优化结构、分组共享日志缓冲、记录级依赖计数、异步S3 append、双缓冲、delta日志、管道提交。

**📊 数据集**

使用YCSB-A/B（30M记录）和TPC‑C（100仓库）等基准测试。

**📈 对比分析**

与传统块存储（EBS gp3/io2）以及原始S3 baseline 对比，在相同吞吐下平均提交延迟降至约25%以内，99.9/99.99尾延迟相对 baseline 降低50%+，吞吐量保持接近或略高。

**⚠️ 局限性**

局限性：仍高于NVMe SSD实现；记录级依赖在极端冲突情形下无法完全消除；需要手动调整线程组和缓冲区尺寸；对低负载或高写密度环境效果有限。

---

## 636. Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance

**arXiv ID:** 2603.02175 | [PDF](https://arxiv.org/pdf/2603.02175v1)

**作者:** Yiqi Lin `[一作]`, Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了可扩展的数据生成管线，将现有视频编辑对转化为带有视觉参考的四元组，并基于此构建 RefVIE 数据集与 RefVIE‑Bench 评测基准，随后设计统一的 MLLM‑DiT 体系结构，实现指令+参考双模态的视频编辑。

**💡 创新点**

创新点在于：①利用图像生成模型自动合成高质量参考图像，突破了缺乏四元组数据的瓶颈；②在 MLLM 之外引入双连结（Query Connector 与 Latent Connector）和混合潜在注入，显著提升参考一致性与结构保留；③提出三阶段渐进式训练课程，确保语义对齐、指令跟随与参考精细化三方面的稳健收敛。

**🔧 技术方法**

技术包括：预训练的多模态大语言模型 Qwen2.5‑VL‑3B、Diffusion Transformer Wan2.2‑TI2V‑5B、LoRA 微调、基于 Vision‑Language grounding 与 SAM 的目标分割、基于 Qwen‑Image‑Edit 的参考图像合成、Flow Matching 损失与三阶段训练策略。

**📊 数据集**

使用的数据集包括：公开的指令视频编辑数据集（Ditto‑1M、ReCo、OpenVE‑3M）经过过滤得到 3.7M 样本；自动生成的 RefVIE 数据集 477K 四元组；并在 100 条手工验证样本上构建 RefVIE‑Bench 进行评测。

**📈 对比分析**

在 OpenVE‑Bench 及 RefVIE‑Bench 上，所提出的 KiWi‑Edit 在指令编辑、背景替换及参考一致性等指标上均超过现有开源基线（如 VACE、Omni‑Video、InsViE 等），在多模态参考任务上与闭源商业模型（Runway Aleph、Kling‑O1）相当甚至优于后者，整体性能提升约 15‑20%。

**⚠️ 局限性**

局限性包括：①参考图像合成质量受预训练编辑模型性能限制；②数据集仍偏重局部编辑，背景替换效果略弱；③模型对大尺寸、高复杂度场景的泛化能力待进一步验证；④依赖强大的预训练模型，推理成本和硬件需求较高。

---

## 637. Efficient RLVR Training via Weighted Mutual Information Data Selection

**arXiv ID:** 2603.01907 | [PDF](https://arxiv.org/pdf/2603.01907v1)

**作者:** Xinyu Zhou `[一作]` (Hong Kong University of Science and Technology Guandong), Zhijiang Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于信息量的在线数据选择方法 InSight，用于强化学习可验证奖励 (RLVR) 训练中大语言模型的高效学习。

**💡 创新点**

创新点在于将互信息与任务难度加权结合，既考虑先验置信度（epistemic uncertainty）对信息增益的衰减，又通过高方差过滤和目标难度偏置来引导可学习难度；同时使用贝叶斯后验均值而非采样成功率，避免采样噪声。

**🔧 技术方法**

使用贝叶斯推断（Beta 先验与后验更新）、互信息计算、加权互信息 (WMI) 得分、GRPO 算法的 RLVR 训练框架，以及多步 roll‑out 的信息量评估。

**📊 数据集**

在规划、数学和通用推理任务的标准数据集上评测：Countdown、DeepScaler、AIME24、AMC23、MATH500、Minerva Math、OlympiadBench、MMLU、GPQA 等。

**📈 对比分析**

与随机采样、MoPPS、Inverse‑Evidence、Expected‑Difficulty、Dynamic Sampling 等基线对比，InSight 在大多数指标上均优于对手，平均提升 1–2 分，训练效率提升约 2.2 倍；在小模型和低资源场景表现尤为显著。

**⚠️ 局限性**

局限性包括：对大模型提升幅度有限；对高噪声任务仍需谨慎；需要设置 Beta 先验和加权超参数；在极大模型或资源受限环境下的计算成本仍受限。

---

## 638. You Only Need One Stage: Novel-View Synthesis From A Single Blind Face Image

**arXiv ID:** 2603.01328 | [PDF](https://arxiv.org/pdf/2603.01328v1)

**作者:** Taoyue Wang `[一作]` (State University of New York), Lijun Yin `[通讯]` (State University of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单阶段方法NVB-Face，能直接从低质量盲人脸图像生成多视角高质量人脸图像。

**💡 创新点**

创新点在于将盲人脸恢复与新视角合成融合为统一的端到端流程，并利用Transformer构造3D潜在特征网格实现多视角一致性。

**🔧 技术方法**

主要技术包括Stable Diffusion扩散模型、图像编码器、Transformer‑based 3D特征构造模块、Camera Predictor、LoRA微调以及自定义损失函数。

**📊 数据集**

使用的训练数据包括NeRSemble多视角真实人脸、PanoHead合成数据、FFHQ高分辨率人脸以及CelebA、LFW测试集。

**📈 对比分析**

与两阶段流程（CodeFormer+PanoHead-PTI、GOAE、TriPlaneNet、DiffPortrait3D）以及单视角恢复方法比较，NVB‑Face在SSIM、LPIPS、DISTS、FID、ID相似度等指标上均优于基线，尤其在严重降质条件下表现更稳定。

**⚠️ 局限性**

限制在于仍需依赖扩散模型的预训练、对摄像机参数预测的准确性有限以及在极端噪声/模糊条件下可能出现身份或表情偏移。

---

## 639. GroupEnsemble: Efficient Uncertainty Estimation for DETR-based Object Detection

**arXiv ID:** 2603.01847 | [PDF](https://arxiv.org/pdf/2603.01847v1)

**作者:** Yutong Yang `[一作]`, Bin Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为GroupEnsemble的快速单通道不确定性估计方法，专门针对DETR及其变体；

**💡 创新点**

通过在Transformer解码器中并行使用多组对象查询并应用注意力掩码，实现高效的多重检测集聚合，从而在单前向传递中估计语义与空间不确定性；

**🔧 技术方法**

采用DETR架构、Group DETR的多组查询训练、注意力掩码、BSAS聚类、加权平均与方差聚合，以及与MC‑Dropout的混合使用；

**📊 数据集**

在Cityscapes、Foggy Cityscapes和COCO三大数据集上进行验证；

**📈 对比分析**

与MC‑Dropout、Deep Ensembles及Deterministic Baseline比较，GroupEnsemble在PDQ、mAP和校准指标上与MC‑Dropout相当，且延迟比Deep Ensembles快约66%，参数量仅低0.7%，混合版MC‑GroupEnsemble在某些指标上优于Deep Ensembles；

**⚠️ 局限性**

受限于需要额外的查询组和聚类步骤，对查询组数量与聚类阈值的选择较敏感，且在极端遮挡或小目标场景下的空间不确定性估计仍存在挑战。

---

## 640. FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters

**arXiv ID:** 2603.01685 | [PDF](https://arxiv.org/pdf/2603.01685v1)

**作者:** Shao Shitong `[一作]` (Hong Kong University of Science and Technology), Xie Zeke `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三阶段的FastLightGen算法，将大型视频生成模型压缩为轻量级且少步的模型，实现高效的视频生成；

**💡 创新点**

创新点在于联合压缩模型大小和采样步数的共训练框架，并引入“well‑guided teacher guidance”动态平衡教师强度；

**🔧 技术方法**

使用了动态概率剪枝、分层重要性评估、分布匹配损失、CFG调节以及DiT架构；

**📊 数据集**

在HunyuanVideo‑ATI2V与WanX‑TI2V两个大规模文本‑图像到视频数据集上训练和评估；

**📈 对比分析**

与LCM、DMD2、MagicDistillation、F3‑Pruning、ICMD等方法对比，FastLightGen在保持或提升视觉质量的同时，速度提升约35×，并且在VBench-I2V上平均分数超越教师模型；

**⚠️ 局限性**

局限性包括对超参数（如α、β₁、β₂）的依赖，需要手工调优；在极低步数（≤1）时性能仍受限，且实验主要聚焦TI2V任务，其他视频生成任务仍需验证。

---

## 641. VEMamba: Efficient Isotropic Reconstruction of Volume Electron Microscopy with Axial-Lateral Consistent Mamba

**arXiv ID:** 2603.00887 | [PDF](https://arxiv.org/pdf/2603.00887v1)

**作者:** Longmi Gao `[一作]` (Nanjing University of Aeronautics and Astronautics), Pan Gao `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 66816 | [OpenAlex ID](https://openalex.org/A5004893546)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 VEMamba，一个基于 Mamba 的自监督方法，用于从各向异性体电子显微镜（VEM）数据中恢复等轴高分辨率体积。

**💡 创新点**

创新点包括：1）Axial‑Lateral Chunking Selective Scan Module (ALCSSM) 将三维依赖重新排序为 1D 序列；2）Dynamic Weights Aggregation Module (DWAM) 动态融合多方向序列；3）利用 Momentum Contrast 学习降解表征并通过 VDIM 注入网络，提高对真实降解的鲁棒性。

**🔧 技术方法**

使用 Mamba 状态空间模型、连续多方向扫描、动态权重融合、MoCo 对降解的无监督学习以及像素混洗上采样等技术。

**📊 数据集**

在 EPFL（FIB‑SEM）和 CREMI（TEM）两个标准 VEM 数据集上进行训练与测试，分别模拟 ×4、×8、×10 的 z‑方向降解。

**📈 对比分析**

与插值基线、IsoVEM、EMDiffuse 等方法对比，VEMamba 在 PSNR、SSIM、LPIPS 等指标上多项排名第一，且参数量和 FLOPs 最低；下游线粒体分割任务中 IoU 与原始等轴数据几乎无差距，表明重建质量高。

**⚠️ 局限性**

局限性：仅在 VEM 数据上验证，可能对其他成像模态的迁移性不足；LPIPS 在灰度 EM 图像上存在域偏差；尽管算力低，但仍需 GPU 进行训练。

---

## 642. A402: Bridging Web 3.0 Payments and Web 2.0 Services with Atomic Service Channels

**arXiv ID:** 2603.01179 | [PDF](https://arxiv.org/pdf/2603.01179v1)

**作者:** Yue Li `[一作]` (Peking University), Jianbo Gao `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5089030314)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了 A402 体系，结合 Web3 资产支付与 Web2 服务，通过 Atomic Service Channels（ASC）实现机器对机器（M2M）支付与服务交付的原子性。

**💡 创新点**

核心创新：① Atomic Service Channels 将支付通道与服务执行绑定；② 使用 TEE 辅助的适配器签名（Adaptor Signature）实现 Exec‑Pay‑Deliver 原子性；③ 设计隐私保护的 Liquidity Vault，实现离线批量结算与链上匿名；④ 在单节点即可实现数千请求每秒的低延迟吞吐。

**🔧 技术方法**

技术手段：支付通道（Lightning‑style）、可信执行环境（TEEs 如 SEV‑SNP/SGX）、Schnorr 适配器签名、远程证明、以太坊与比特币智能合约、分布式共识（Raft）以及离线状态同步。

**📊 数据集**

实验使用模拟工作负载：100 个并发客户端、100 条 ASC、200 ms 处理延迟、10 ms 网络延迟；未使用真实业务数据集。

**📈 对比分析**

对比基准：x402（需链上交易每请求）和理想化的 x402。A402 在以太坊与比特币上实现每秒 2,875 次请求吞吐，吞吐率比 x402 高 95‑410 倍；延迟 0.34‑0.37 s，远快于 Solana（12.8 s）、以太坊（13 min）和比特币（60 min）；链上费用从 O(n) 降至 O(1)，成本减少约 28‑46 倍。

**⚠️ 局限性**

限制与挑战：① 依赖可信执行环境与硬件安全，侧信道攻击等未考虑；② 仍需链上最终结算，若链拥塞会影响极端情况；③ 需要服务方和客户端均支持 TEE 与签名协议；④ 对大规模异构网络的可扩展性和部署复杂度待进一步评估。

---

## 643. Nonconvex Latent Optimally Partitioned Block-Sparse Recovery via Log-Sum and Minimax Concave Penalties

**arXiv ID:** 2603.01304 | [PDF](https://arxiv.org/pdf/2603.01304v1)

**作者:** Takanobu Furuhashi `[一作]` (Nagoya Institute of Technology), Tatsuya Yokota `[通讯]` (Nagoya Institute of Technology)

**通讯引用:** 8141 | [OpenAlex ID](https://openalex.org/A5039764322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了两种基于非凸正则化的块稀疏信号恢复方法 LogLOP- 与 AdaLOP-，能够在未知块划分条件下进行精确恢复。

**💡 创新点**

创新点在于将对数求和和最小极大凸包(MCP)的非凸变分形式引入 LOP 框架，并通过可变权重自适应降低估计偏差，同时兼容任意数据逼近项。

**🔧 技术方法**

采用变分 LOP 形式、可变权重机制和 ADMM 求解算法，实现了非凸正则化下的块稀疏恢复。

**📊 数据集**

实验使用合成数据、角功率谱估计（APS）与纳米孔电流去噪三类数据集验证方法。

**📈 对比分析**

与 LOP、GME-LOP、SBL 等传统方法以及凸 ℓ1 方案对比，实验表明 LogLOP- 与 AdaLOP- 在估计精度上明显优于基准方法。

**⚠️ 局限性**

主要局限在于缺乏全局收敛理论，且超参数（如块划分粗细、γ、λ 等）需经验调优，尚未实现自动化选择。

---

## 644. Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?

**arXiv ID:** 2603.02069 | [PDF](https://arxiv.org/pdf/2603.02069v1)

**作者:** Jihwan Kim `[一作]` (Seoul National University and KAIST InnoCORE LLM), Chulhee Yun `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在平稳随机特征（PLRF）模型下推导了 signSGD 的缩放律，揭示了与 SGD 不同的漂移归一化和噪声重塑两大机制，并给出了在固定计算预算下的最优模型规模与学习率、计算最优损失衰减斜率；同时分析了常数学习率与温度平稳衰减（WSD）调度的影响；

**💡 创新点**

创新点在于①首次在 PLRF 框架下对 signSGD 进行理论分析并得到四项缩放律；②发现漂移归一化和噪声重塑对计算最优斜率的正面影响；③证明 WSD 调度可进一步提升计算最优斜率；④与 SGD 和 Adam 进行对比并验证理论。

**🔧 技术方法**

使用的技术主要包括：平稳随机特征模型（PLRF）、高斯压缩特征、连续时间 ODE/SDE 近似、漂移与噪声项的分解、隐式积分方程求解、稳态自洽方程、计算最优调度分析及实验验证。

**📊 数据集**

实验采用 PLRF 生成的合成数据（高斯特征 + 目标系数按幂律衰减），并在此模型下训练线性回归模型。

**📈 对比分析**

与 SGD 的缩放律进行对比，发现 signSGD 在噪声瓶颈阶段（Phase 3、4）具有更陡峭的计算最优斜率；在 WSD 调度下进一步提升；实验图表与理论指数一致，验证了 signSGD 的优越性。

**⚠️ 局限性**

局限性包括：①分析仅适用于线性模型和 PLRF 生成的合成数据；② Adam 的结论基于启发式假设，缺乏严格证明；③WSD 调度在部分相位下未能提升，且未证明对所有优化器都有效；④噪声重塑机制在更一般设置下的泛化性尚待验证。

---

## 645. $π$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs

**arXiv ID:** 2603.02083 | [PDF](https://arxiv.org/pdf/2603.02083v1)

**作者:** Siting Wang `[一作]` (GigaAI), Jun Wang `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种无价值网络、无似然推理的在线强化学习框架π-StepNFT，用于微调基于流的视觉‑语言‑动作模型，使其在更宽的探索空间中通过逐步监督实现更稳健的对齐。

**💡 创新点**

创新点在于：①采用SDE采样扩展探索空间；②将监督目标从终端x₀迁移到一步状态x_t⁻，实现细粒度对齐；③使用对比排名损失消除隐式分离惩罚，产生推拉动态。

**🔧 技术方法**

技术手段包括：流匹配（flow‑matching）策略、逆时随机微分方程（SDE）采样、对比排名（contrastive ranking）损失、softplus 正则化，以及EMA、动态衰减等训练技巧。

**📊 数据集**

使用的数据集为LIBERO（多任务、少样本）和ManiSkill（大规模视觉多样性、OOD测试）等。

**📈 对比分析**

与传统SFT、PPO、GRPO等基线比较，在LIBERO少样本SFT下平均提升约32.9%，在ManiSkill OOD环境下提升约11.1%，在长周期任务中与价值基线保持竞争力。

**⚠️ 局限性**

局限性在于：依赖稀疏终端奖励，对长序列信用分配不如价值网络细粒度；在极大噪声或大步长下可能失效；需要手工调节β、σ等超参数。

---

## 646. Defensive Refusal Bias: How Safety Alignment Fails Cyber Defenders

**arXiv ID:** 2603.01246 | [PDF](https://arxiv.org/pdf/2603.01246v1)

**作者:** David Campbell `[一作]` (Scale AI), Christina Q Knight `[通讯]` (Scale AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了安全对齐导致的防御性拒绝偏差，量化了安全语言模型在合法防御场景中的拒绝行为。

**💡 创新点**

首次从真实赛制数据出发，揭示了防御性请求与攻击性请求在语言上相似时模型拒绝的偏差，并指出授权信息反而会增加拒绝率。

**🔧 技术方法**

采用了拒绝检测、语义嵌入分类、统计显著性检验以及基于嵌入的预测模型等技术来分析拒绝模式。

**📊 数据集**

使用了来自全国大学生网络安全防御竞赛（NCCDC）的2,390条单轮对话数据，涵盖恶意软件分析、漏洞评估、系统硬化等八类防御任务。

**📈 对比分析**

将Claude 3.5 Sonnet、GPT‑4o和Llama‑3.3‑70B‑Instruct三种模型的拒绝率进行对比，安全关注型模型拒绝率最高（19.5%），开放源模型最低（6.6%），并通过chi‑square检验和AUC评估证明语义相似性是拒绝的主要驱动因素。

**⚠️ 局限性**

局限在于依赖基于关键词的注释方法，单轮对话限制，未涵盖多轮交互与不同攻击手法，且模型对授权信号的处理机制尚未明确。

---

## 647. Generative AI in Software Testing: Current Trends and Future Directions

**arXiv ID:** 2603.02141 | [PDF](https://arxiv.org/pdf/2603.02141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 648. RepoRepair: Leveraging Code Documentation for Repository-Level Automated Program Repair

**arXiv ID:** 2603.01048 | [PDF](https://arxiv.org/pdf/2603.01048v1)

**作者:** Zhongqiang Pan `[一作]` (Nanjing University), Vincent Ng `[通讯]` (University of Texas at Dallas)

**通讯引用:** 8523 | [OpenAlex ID](https://openalex.org/A5004305684)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 RepoRepair，一个面向仓库级自动程序修复（APR）的无代理框架，核心包括层级化代码文档生成、基于多模态问题描述的故障定位以及上下文感知的补丁生成与验证。

**💡 创新点**

创新点：
1) 利用大型语言模型（LLM）生成函数层和文件层的结构化代码文档，提供语义抽象，显著提升跨文件定位精度；
2) 通过文档驱动的三步定位（文件检索 → 可疑文件定位 → 函数/类定位），实现对复杂、多文件缺陷的全局理解；
3) 引入依赖感知的代码裁剪与组合式补丁验证（温度递增迭代），提高补丁质量与成本效益。

**🔧 技术方法**

技术手段：
- LLM（DeepSeek‑V3 生成文档，Claude‑4 处理定位与修复）
- 代码解析：Tree‑sitter + AST 提取函数、类、调用等元数据
- 文档检索：sentence‑transformers/all‑mpnet‑base‑v2 + FAISS 向量索引
- 多模态预处理：将图片/视频转化为文本描述后供 LLM 处理
- 迭代修复：温度逐步递增、组合式补丁验证、最小修改原则
- 成本评估：按 GPT‑4o/Claude‑4 推理计费

**📊 数据集**

数据集：
- SWE‑bench Lite（Python 仓库 323 个 issue）
- SWE‑bench Multimodal（JavaScript/TypeScript 仓库 619 个 issue，包含文本+视觉信息）

**📈 对比分析**

比较方法与性能：
- 与 RAG、SWE‑Agent、Agentless、Agentless Lite、OpenHands、PatchPilot、DARS、ExpeRepair、GUIRepair 等基线工具同一批 benchmark 进行对比；
- 评价指标：%Resolved、Avg. $Cost、%Correct Localization；
- 结果：
  • Lite：45.7% 修复率（$0.44/issue），高于 Agentless Lite（32.3%）且成本更低；
  • Multimodal：37.14% 修复率（$0.56/issue），超过 GUIRepair（35.98%）和 OpenHands‑Versa（34.43%）；
  • 文件定位准确率在 Multimodal 上达到 59.8%，大幅高于对手（Agentless 30.4%、GUIRepair 44.3%）。

**⚠️ 局限性**

局限性：
1) 文档生成仅覆盖包含函数/类的文件，对配置、静态资源等无功能文件缺失；
2) 目前不支持需要新增文件的功能请求；
3) 依赖增量文档更新可能忽略文件间的语义演化；
4) 文档质量高度依赖 LLM，若生成不准确会直接影响定位与修复；
5) 仅在 Python 与 JavaScript/TypeScript 上验证，其他主流语言尚未评估。

---

## 649. CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging

**arXiv ID:** 2603.00573 | [PDF](https://arxiv.org/pdf/2603.00573v1)

**作者:** Jie Cao `[一作]` (Zhejiang University), Siliang Tang `[通讯]` (Zhejiang University)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5063062444)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Core Space Mixture of LoRA (CoMoL) 框架，利用核心空间专家与核心空间路由实现参数高效且细粒度的 MoE‑LoRA 微调。

**💡 创新点**

创新点在于把 LoRA 专家压缩到低秩核心矩阵中，并在该核心空间完成软合并与路由，从而显著降低专家与路由参数，同时保持 token 级动态适配。

**🔧 技术方法**

使用 LoRA 低秩分解、Mixture‑of‑Experts、核心空间 M 矩阵软合并、低秩路由投影以及软路由策略。

**📊 数据集**

使用数学推理数据集（GSM8K、AQuA、SVAMP、MultiArith、AddSub、SingleEq）以及代码生成数据集（CodeAlpaca‑20k、HumanEval），在 Qwen3‑8B/14B、Llama3.1‑8B 等模型上进行实验。

**📈 对比分析**

与标准 LoRA、MoLoRA、HydraLoRA、MoLA、AdaMoLE、SparseMoA、FlyLoRA、DenseLoRA 等方法对比，CoMoL 在保持参数与 LoRA 相近的前提下，平均准确率/Pass@k 提升 1–3 个百分点，显著优于竞争者。

**⚠️ 局限性**

仍缺乏系统评估不同 PEFT 方法在多种微调场景下学习容量的基准，CoMoL 对不同模型家族和极限适用性尚未完全明晰。

---

## 650. Rich Insights from Cheap Signals: Efficient Evaluations via Tensor Factorization

**arXiv ID:** 2603.02029 | [PDF](https://arxiv.org/pdf/2603.02029v1)

**作者:** Felipe Maia Polo `[一作]` (University of Michigan), Isabela Albuquerque `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种利用张量分解与自动评估器辅助的细粒度生成模型评估方法，能够在人工标注稀缺时准确预测模型对不同提示的表现。

**💡 创新点**

创新点在于将自动评估器的评分作为辅助信号通过低秩张量分解学习模型和提示的潜在表示，并在少量人工数据上校准，既实现可扩展又保持人类对齐。

**🔧 技术方法**

使用了多维张量分解（CP decomposition）、有序逻辑回归、最大似然估计以及预训练–微调两阶段训练等技术。

**📊 数据集**

在Gecko（文本到图像）、BigGen Bench（文本生成）和LMArena（对话）三个基准上验证。

**📈 对比分析**

与常数模型、提示特定模型、P2L等基线相比，跨交叉熵、平均分、胜率等指标均表现更好，尤其在仅10%人工注释时仍能恢复排名并预测未见模型。

**⚠️ 局限性**

局限包括对低秩张量和有序逻辑假设的依赖、仅在侧对侧模板下可识别相对能力、需要足够多样化且与人类偏好相关的自动评估器、置信区间未完全考虑第一阶段误差、可微调后不再具备理论区间。

---

## 651. TIMI: Training-Free Image-to-3D Multi-Instance Generation with Spatial Fidelity

**arXiv ID:** 2603.01371 | [PDF](https://arxiv.org/pdf/2603.01371v1)

**作者:** Xiao Cai `[一作]` (University of Electronic Science and Technology of China), Jingkuan Song `[通讯]` (Tongji University)

**通讯引用:** 14258 | [OpenAlex ID](https://openalex.org/A5036987388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出TIMI框架，在不需要额外训练的情况下通过实例感知指导实现图像到3D多实例生成，保持高空间保真。

**💡 创新点**

设计了实例感知分离指导(ISG)与空间稳定几何自适应更新(SGU)，实现训练自由的实例解耦与全局布局保持。

**🔧 技术方法**

采用预训练的3D扩散模型（Hunyuan3D 2.0）、跨注意力机制、实例注意力锚定、分离损失、空间正则化与几何自适应梯度调节。

**📊 数据集**

使用3D-Front合成场景、Real-Data真实图像以及Flux.1 Kontext风格化图像。

**📈 对比分析**

与MIDI、DPA及单实例Hunyuan3D 2.0比较，TIMI在LCD、CD‑S、FS‑S和SSR、CD‑O、FS‑O等指标上均优于基线，且推理速度约59秒，快于MIDI与DPA。

**⚠️ 局限性**

依赖预训练基础模型的局限性，可能继承数据偏差；实例分离需手动掩码，且在极薄或复杂结构实例上仍可能出现几何失真。

---

## 652. Caught in a Mafia Romance: How Users Explore Intimate Roleplay and Narrative Exploration with Chatbots

**arXiv ID:** 2603.01319 | [PDF](https://arxiv.org/pdf/2603.01319v1)

**作者:** Julia Kieserman `[一作]` (New York University), Rosanna Bellini `[通讯]` (New York University)

**通讯引用:** 452 | [OpenAlex ID](https://openalex.org/A5019068104)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Character.AI平台进行大规模混合方法研究，分析用户创建与使用的聊天机器人以及Reddit讨论，探究用户的动机、角色关系和面临的挑战。

**💡 创新点**

首次系统性量化Character.AI社区用户的角色创作动机与行为，揭示两大核心使用场景——亲密角色扮演与叙事探索，并提出针对数字安全的设计建议。

**🔧 技术方法**

使用文本主题分析、人工编码与Anthropic Claude 3.5 Sonnet自动提取特征，并结合Python爬虫收集5.76M个聊天机器人描述与2k个Reddit帖子。

**📊 数据集**

收集了5,761,412个公开Character.AI聊天机器人描述（包含名称、描述、标签等）以及2,078条与cAI相关的Reddit子版块帖子。

**📈 对比分析**

对流行度最高和随机样本的聊天机器人进行描述性统计与主题编码，主要聚焦于定性发现与统计比例，未给出传统模型性能指标；发现亲密角色占63%，叙事探索比例亦较大。

**⚠️ 局限性**

数据仅限公开英文聊天机器人和Reddit用户，缺乏互动文本与跨语言样本，无法评估真实用户体验与情感依赖，平台内容审核不一致等限制。

---

## 653. Security Risks in Machining Process Monitoring: Sequence-to-Sequence Learning for Reconstruction of CNC Axis Positions

**arXiv ID:** 2603.01702 | [PDF](https://arxiv.org/pdf/2603.01702v1)

**作者:** Lukas Krupp `[一作]` (RPTU University Kaiserslautern-Landau), Norbert Wehn `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文将机床主轴上安装的宽带加速度计采集到的振动信号映射为轴位置信息，实现了对CNC加工轨迹的重构。

**💡 创新点**

首次证明序列到序列的LSTM模型能在工业加工环境中克服噪声和偏置导致的积分漂移，显著降低轴位置重构误差，并揭示振动数据对机床知识产权的安全风险。

**🔧 技术方法**

采用LSTM序列到序列学习模型（many-to-one、many-to-many、autoregressive）以及传统双积分数值积分基线进行位置重构，并使用PyTorch训练、SciPy实现DSP基线。

**📊 数据集**

基于RPTU的五轴铣床采集的主轴加速度与位置同步数据集，包含多种工况、刀具路径和切削参数，覆盖不同复杂度的加工序列。

**📈 对比分析**

与双积分基线对比，学习模型在四个场景下平均误差从约4~90 mm显著下降，误差降低率从98%（最简单场景）降至85%（最复杂场景），显示出明显的性能优势。

**⚠️ 局限性**

局限在于仅对单轴独立预测，缺乏多轴协同；仅在单一机床上验证，跨机床泛化性待验证；模型对极端噪声、传感器漂移的鲁棒性有限。

---

## 654. Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation

**arXiv ID:** 2603.02139 | [PDF](https://arxiv.org/pdf/2603.02139v1)

**作者:** Han Xue `[一作]` (Shanghai Jiao Tong University), Chuan Wen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在机器人模仿学习中系统评估腕部鱼眼摄像头对空间定位、场景泛化和硬件泛化的影响。

**💡 创新点**

首次量化鱼眼摄像头的宽视野优势与畸变挑战，并提出随机尺度增广（RSA）提升跨镜头泛化的实用方案。

**🔧 技术方法**

采用Diffusion Policy框架与U‑Net+DDIM推理，结合随机尺度增广和两阶段鱼眼仿真渲染技术。

**📊 数据集**

使用Robomimic与MimicGen两大仿真基准，并在真实Flexiv Rizon 4平台上收集多背景鱼眼与针孔摄像头数据。

**📈 对比分析**

通过成功率和归一化分数比较，鱼眼摄像头在视觉复杂环境下显著提升空间定位，场景多样性训练后成功率超过95%，RSA在跨镜头测试中保持高性能。

**⚠️ 局限性**

仅在有限任务与少数硬件配置验证，未覆盖更广泛的镜头参数、多传感器组合以及长时序任务的鲁棒性仍需进一步研究。

---

## 655. From Pixels to Patches: Pooling Strategies for Earth Embeddings

**arXiv ID:** 2603.02080 | [PDF](https://arxiv.org/pdf/2603.02080v1)

**作者:** Isaac Corley `[一作]` (Wherobots), Juan M. Lavista Ferres `[通讯]` (Microsoft AI for Good Research Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在地理空间嵌入领域，本文通过聚合像素级嵌入为补丁级表示，系统评估了 13 种无监督池化方法（含均值、最大、GeM、标准差、分位数、统计堆叠、协方差等）以及 2 种参数化方法（PCA、BoVW），并在 EuroSAT 10 类土地覆被数据集上使用 kNN 与线性分类器进行评估，揭示了更丰富的分布统计能显著提升空间泛化性能。

**💡 创新点**

创新点包括：① 发布了可直接复现的 EuroSAT‑Embed 嵌入数据集，包含三种 GFM（AlphaEarth、OlmoEarth、Tessera）生成的 81,000 个 64×64 嵌入图；② 对无监督池化方法进行大规模实验，首次量化不同统计量对空间分布偏移的鲁棒性；③ 证明 Generalized Mean Pooling (GeM) 可作为无额外参数的均值替代方案，在保持维度不变的前提下提升空间准确率 5%；④ 发现 Stats（min/max/mean/std）池化在 4× 增大维度时可获得最高准确率，最大减少 40% 的泛化差距。

**🔧 技术方法**

使用的技术包括：像素级嵌入池化（均值、最大、GeM、标准差、分位数、协方差、Stats、PCA、BoVW）；kNN（k=5，余弦距离）和多项式逻辑回归（线性探针）两种评估探针；对随机划分和地理上互斥的空间划分进行对比评估；统计分析（准确率、泛化差距、维度倍数）。

**📊 数据集**

使用 EuroSAT 土地覆被数据集（10 类，共 27,000 张 64×64 像素），以及从 AlphaEarth、OlmoEarth、Tessera 三个地理空间基础模型生成的嵌入（分别为 64、128、512 维）。

**📈 对比分析**

对 13 种无监督池化方法和 2 种参数化方法进行 3（模型）× 13（池化）× 2（探针）× 2（划分） 的完整网格实验。结果显示：均值池化在空间划分上下降 10%，而 Stats 池化将该差距压缩至 6.2%；GeM 在保持 1× 维度的情况下相对均值提升约 5%；Stats（4× 维度）在最优情况下提升 7–10%，且在所有模型中均表现出较低的空间泛化差距。

**⚠️ 局限性**

局限性包括：仅在 EuroSAT 单一 10 类数据集上验证；未覆盖多标签、多任务或更大规模的数据集；缺乏针对特定任务的自适应学习型池化（如注意力池化）的实验；并且所有池化方法的超参数（如 GeM 的 p=3、kNN 的 k=5）均采用固定值，未对不同编码器进行细粒度调优。

---

## 656. MetaState: Persistent Working Memory for Discrete Diffusion Language Models

**arXiv ID:** 2603.01331 | [PDF](https://arxiv.org/pdf/2603.01331v1)

**作者:** Kejing Xia `[一作]` (Georgia Institute of Technology), Wenke Lee `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 27160 | [OpenAlex ID](https://openalex.org/A5047140382)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

为离散扩散语言模型引入持久固定大小的工作内存，改善跨步信息流。

**💡 创新点**

提出MetaState三模块循环架构（Mixer‑Updater‑Injector）与共享时间条件，解决信息岛问题。

**🔧 技术方法**

采用跨注意力、GRU型递归、AdaRMSNorm和时间嵌入的轻量级模块。

**📊 数据集**

使用Tülu‑3 SFT混合数据集（约5万序列）进行微调。

**📈 对比分析**

在GSM8K、MATH‑500、HumanEval、MBPP四大基准上，与冻结的LLaDA‑8B和Dream‑7B基线相比，平均提升3–9点，尤其在数学推理与代码生成任务上显著。

**⚠️ 局限性**

训练时需要多步展开导致显著的时间与显存开销，推理时也增加了额外的模块执行，导致延迟和内存使用上升。

---

## 657. StegoNGP: 3D Cryptographic Steganography using Instant-NGP

**arXiv ID:** 2603.00949 | [PDF](https://arxiv.org/pdf/2603.00949v1)

**作者:** Wenxiang Jiang `[一作]` (Shandong Technology and Business University), Jinxin Wang `[通讯]` (Shandong Technology and Business University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种参数无增量、基于Instant‑NGP的3D加密隐写方法（StegoNGP），利用哈希函数的密钥控制实现覆盖场景与隐藏场景在同一模型权重中互切；

**💡 创新点**

创新点在于将Instant‑NGP的哈希映射改造成密钥驱动的场景开关，既实现了全场景级别的高容量隐写，又不改变网络结构或增加参数；此外，还提出多键层级分配方案，显著扩展密钥空间并提升对部分密钥泄露的鲁棒性；

**🔧 技术方法**

技术上使用Instant‑NGP多分辨率哈希特征表、MSE+稀疏正则化的联合训练、密钥替换的哈希索引函数以及多键层级赋值；

**📊 数据集**

实验采用Blender Synthetic数据集（8个合成对象）和Mip‑NeRF‑360真实场景数据集；

**📈 对比分析**

与基线Instant‑NGP和GS‑Hider进行对比，PSNR/SSIM/LPIPS指标与基线相当或略优，且StegoNGP无额外参数、模型体积与普通Instant‑NGP相同；多键方案在部分密钥泄露时仍保持覆盖场景，完整密钥才能恢复隐藏场景，显示出高安全性；

**⚠️ 局限性**

局限性包括：仅能嵌入单一完整隐藏场景，密钥管理复杂；对极大或动态场景的可扩展性尚待验证；仍需对模型进行完整训练，且目前仅针对Instant‑NGP实现，难以直接迁移至其他3D表示框架。

---

## 658. What Papers Don't Tell You: Recovering Tacit Knowledge for Automated Paper Reproduction

**arXiv ID:** 2603.01801 | [PDF](https://arxiv.org/pdf/2603.01801v1)

**作者:** Lehui Li `[一作]` (Shandong University), Yongshun Gong `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于图的代理框架，用来自动重现学术论文的可执行代码，并通过分阶段方法逐步恢复三类隐性知识（关系式、体感式、集体式）

**💡 创新点**

创新点在于将隐性知识细分为关系、体感、集体三类并为每类设计专门的图推理机制；引入语义科学图裁剪、节点关系聚合、执行反馈细化以及子图知识诱导，使得知识恢复更完整、更系统

**🔧 技术方法**

使用大型语言模型（Claude Opus 4.5）作为排名器、关系分析器和生成器；结合图神经网络的邻域聚合、Louvain聚类、ReAct式迭代调试等技术

**📊 数据集**

使用扩展版 ReproduceBench 数据集，涵盖推荐系统、时间序列分析、图学习三大领域，共 10 个任务，包含 191 篇训练、30 篇验证、40 篇测试论文

**📈 对比分析**

与 ReAct、OpenHands、Paper2Code、DeepCode、AutoReproduce 等基线比较，平均性能差距降至 10.04%，比最强基线提升 24.68%，在人类评测中也取得最高分，显示显著优越

**⚠️ 局限性**

局限性包括仍依赖可用的实现代码和运行反馈，跨领域、低资源或非可执行论文的恢复效果有限；模型规模对结果影响显著，需要进一步提升普适性和效率

---

## 659. CeProAgents: A Hierarchical Agents System for Automated Chemical Process Development

**arXiv ID:** 2603.01654 | [PDF](https://arxiv.org/pdf/2603.01654v1)

**作者:** Yuhang Yang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28278 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种分层协作多智能体系统，自动完成化学工艺开发的知识检索、流程图设计与参数优化三大阶段。

**💡 创新点**

创新点在于把单一大模型拆分为专门化的三大团体（知识、概念、参数），通过ChatGroup协同推理与Workflow确定性执行相结合，实现全流程闭环；并构建了覆盖知识、结构、参数的完整评测基准CeProAgents。

**🔧 技术方法**

使用大型语言模型（Gemini、GPT、Claude、Qwen、DeepSeek）作为核心推理器；多模态检索、知识图谱构建、图像解析、顶点链接抽取；以及与行业标准 Aspen Plus 进行闭环优化的计算工具。

**📊 数据集**

使用的数据集包括70份技术文档、113张PFD图（来自全国学生化学设计大赛）、20个Aspen Plus模拟案例，构成知识提取、概念解析/生成、参数优化三维评测。

**📈 对比分析**

通过与单模型基线（5种LLM）对比，层级协作模型在知识域准确率提升至78.8%，概念域蓝图合规度达63%，参数域通过闭环优化实现平均成本下降40%并保持产率最大化。

**⚠️ 局限性**

局限性包括：当前逻辑推理仍受基础LLM推理能力限制；未能显式嵌入热力学与物理约束；缺乏实时实验反馈与工业控制系统的深度集成，导致实际工厂部署的可行性尚待验证。

---

## 660. Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design

**arXiv ID:** 2603.01767 | [PDF](https://arxiv.org/pdf/2603.01767v1)

**作者:** Bosen Lin `[一作]` (Ocean University of China), Qian Du `[通讯]` (Mississippi State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于下游任务的海底图像增强框架DTI-UIE，并自动构建了任务感知增强数据集TI-UIED

**💡 创新点**

创新点包括①通过语义分割性能投票自动生成增强参考图；②设计双分支结构（全局特征恢复分支与细节增强分支）并引入任务感知卷积注意力Transformer块；③采用三阶段训练与任务驱动感知损失对齐增强结果与下游任务的特征空间

**🔧 技术方法**

使用卷积注意力Transformer、任务感知注意力模块、双分支UNet‑style编码器‑解码器、混合样本增强、三阶段联邦训练与任务感知感知损失等技术

**📊 数据集**

主要数据集包括自构造的TI-UIED（基于SUIM/UITIS）、SUIM-E、UIEB用于无参考图像质量评估、RUOD用于目标检测、UIIS用于实例分割

**📈 对比分析**

与10余种传统与深度学习UIE方法在语义分割、目标检测、实例分割等下游任务中进行对比，DTI-UIE在各指标上均取得显著提升（如语义分割mIoU提升1.2–3.3%，检测mAP提升0.5–0.6%，实例分割AP提升0.4–0.5），图像质量评估指标与常规方法相当

**⚠️ 局限性**

局限性在于数据集构建仅基于语义分割任务，可能导致任务偏差；对多任务的统一优化不足；对不同海底环境的泛化仍需进一步验证

---

## 661. AWE: Adaptive Agents for Dynamic Web Penetration Testing

**arXiv ID:** 2603.00960 | [PDF](https://arxiv.org/pdf/2603.00960v1)

**作者:** Akshat Singh Jaswal `[一作]` (Stux Labs), Ashish Baghel `[通讯]` (Stux Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AWE，一种基于多智能体、内存增强的自适应Web渗透测试框架。

**💡 创新点**

创新点在于把LLM与结构化、漏洞专用执行管线相结合，并通过持久内存与浏览器验证实现确定性、低成本的注入式漏洞挖掘。

**🔧 技术方法**

技术包括多层架构（Orchestration、Specialized Agents、Foundation）、Claude Sonnet 4 LLM、持久内存、浏览器验证、针对XSS/SQLi等的结构化探测流程。

**📊 数据集**

使用的评估数据集为104题的XBOW基准以及DVWA用于模型选择。

**📈 对比分析**

方法是与MAPTA对比，在XBOW上AWE实现51.9% solve率、平均53s、1.12M token；MAPTA为76.9% solve率、190s、54.9M token；AWE在XSS、盲SQL等注入类显著优于MAPTA，成本降低63%，速度提升4.4倍。

**⚠️ 局限性**

局限在于仅针对注入类漏洞，缺乏多步规划能力，依赖启发式抽象，对非典型框架和复杂业务逻辑效果有限。

---

## 662. Zero-shot Low-Field MRI Enhancement via Diffusion-Based Adaptive Contrast Transport

**arXiv ID:** 2603.01913 | [PDF](https://arxiv.org/pdf/2603.01913v1)

**作者:** Muyu Liu `[一作]` (ShanghaiTech University), Yuyao Zhang `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种零样本框架DACT，用于将低场MRI图像增强为高场质量图像。

**💡 创新点**

创新点在于引入可微分的Sinkhorn最优传输模块，结合自适应对比度传输（Adaptive Contrast Transport）实现非线性对比度映射，并在逆过程同时维护拓扑一致性。

**🔧 技术方法**

技术上使用了预训练的高场扩散模型作为先验、差分Sinkhorn算法进行直方图匹配、可微分的核密度估计、以及基于半二次分裂（HQS）的插件式优化。

**📊 数据集**

实验数据包括基于HCP数据模拟的0.2T低场合成数据，以及真实0.2T低场T2加权扫描；两者都被划分为训练/验证/测试集。

**📈 对比分析**

与六类基线方法（UNS‐INR、PF‐SR、DiffDeuR、GDP、TAO等）比较，DACT在合成数据上PSNR/SSIM/LPIPS最高，在真实数据上BRISQUE/FID最佳，并在脑组织分割Dice分数上显著优于其它方法。

**⚠️ 局限性**

局限性包括对低场数据的物理假设（如对比度保持单调）可能不适用于极端场强或特殊组织，且推理时需多步优化导致计算成本相对较高。

---

## 663. PARCER as an Operational Contract to Reduce Variance, Cost, and Risk in LLM Systems

**arXiv ID:** 2603.00856 | [PDF](https://arxiv.org/pdf/2603.00856v1)

**作者:** Elzo Brito dos Santos Filho `[一作]` `[通讯]`, Elzo Brito dos Santos Filho

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了PARCER框架，将LLM的交互转化为可版本化、可执行的YAML合同，以实现严格的治理；

**💡 创新点**

创新点包括七阶段执行结构、决策卫生（Decision Hygiene）机制、动态预算与上下文降噪策略，以及与OpenTelemetry的可观测性集成；

**🔧 技术方法**

技术实现涵盖YAML声明式合同、RAG、Map‑Reduce摘要、MMR筛选、动态预算算法、LLM工具调用控制、以及LangGraph/CrewAI等多智能体编排；

**📊 数据集**

未使用具体公开数据集，而是基于理论分析和行业案例构建框架；

**📈 对比分析**

文章未进行实验对比或性能评估，仅在概念层面与现有提示工程方法做比较；

**⚠️ 局限性**

局限性包括缺乏实证验证、对LLM内部机制的依赖、成本与延迟不可控、以及对异常输入的处理尚未完整；

---

## 664. Neural Latent Arbitrary Lagrangian-Eulerian Grids for Fluid-Solid Interaction

**arXiv ID:** 2603.00792 | [PDF](https://arxiv.org/pdf/2603.00792v1)

**作者:** Shilong Tao `[一作]` (Peking University), Yunhuai Liu `[通讯]` (Peking University)

**通讯引用:** 4749 | [OpenAlex ID](https://openalex.org/A5082653046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Fisale，专门用于解决复杂的双向流固耦合问题

**💡 创新点**

创新点在于将固体、流体与耦合界面三者建模为独立组件，并引入多尺度潜在 ALE 网格与分层 PCM 以实现跨域动态捕捉

**🔧 技术方法**

采用多尺度潜在 ALE 网格、分层跨域注意力、基于分区耦合的 PCM 以及自编码/解码投影机制

**📊 数据集**

使用三大真实场景数据集：StructureOscillation、VenousValve 与 FlexibleWing，覆盖 2D/3D、单步、自动回归与稳态推断任务

**📈 对比分析**

与 GeoFNO、GINO、CoDANO、LNO、Galerkin Transformer 等十余种先进学习式求解器对比，Fisale 在三大任务中均获得最小相对 L2 / RMSE 误差，表现优于对手

**⚠️ 局限性**

仍受限于模型规模与训练成本，对极端高雷诺数或更复杂三维几何的泛化仍需进一步验证

---

## 665. Extracting Training Dialogue Data from Large Language Model based Task Bots

**arXiv ID:** 2603.01550 | [PDF](https://arxiv.org/pdf/2603.01550v1)

**作者:** Shuo Zhang `[一作]` (Xi'an Jiaotong University), Jing Tao `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 13036 | [OpenAlex ID](https://openalex.org/A5013885739)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型微调后的任务型对话系统（TODS）在训练数据记忆与泄漏方面的隐私风险，并通过构造训练数据提取攻击评估对话状态的可恢复性。

**💡 创新点**

系统化揭示LLM‑based TODS的记忆机制，提出了基于对话schema的采样（schema‑guided sampling）与去偏差的条件困惑度（debiased conditional perplexity）两种专门适配该场景的攻击技术，并在untargeted/targeted 两种提取设定下实现了显著提升的提取精度。

**🔧 技术方法**

利用Llama2‑7B在 MultiWOZ 上的微调模型，以 suffix‑decoding + schema‑guided sampling 生成候选对话状态，并使用去偏差条件困惑度进行成员推断；随后通过统计与实验评估提取效果。

**📊 数据集**

以公开的 MultiWOZ 2.0 任务对话数据集为训练与评估基准，采用预训练的 Llama2 作为底层模型。

**📈 对比分析**

与原有训练数据提取方法（如 Carlini 等人提出的 suffix‑decoding 与 perplexity 评估）进行对比，实验显示在 untargeted 设定下对值级别信息可提取精度达 67%，完整状态 26%；在 targeted 设定下对值级别信息可达 100%，完整状态 70%，明显优于传统方法。

**⚠️ 局限性**

仅验证了 decoder‑only LLM 的情况；实验在黑盒 score‑based 环境下进行，缺乏对更严苛白盒或仅文本输出的评估；防御措施（对话级建模、值复制机制）仍需进一步实验验证其对性能与隐私的综合影响。

---

## 666. Are LLMs Reliable Code Reviewers? Systematic Overcorrection in Requirement Conformance Judgement

**arXiv ID:** 2603.00539 | [PDF](https://arxiv.org/pdf/2603.00539v1)

**作者:** Haolin Jin `[一作]` (University of Sydney), Huaming Chen `[通讯]` (University of Sydney)

**通讯引用:** 20305 | [OpenAlex ID](https://openalex.org/A5086004140)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估大型语言模型（LLMs）在仅依据自然语言需求说明进行代码合规性判断的可靠性，并揭示其在无测试用例情况下普遍存在的过度纠错偏差；同时探索提示设计对判定准确率与误判类型的影响；提出基于模型修复建议的验证过滤器来缓解过度纠错问题。

**💡 创新点**

创新点在于：①首次将误判模式细分为误读规范、额外约束、边界误判等四大类；②揭示提示复杂度会导致错误转移而非统一提升准确率；③设计了Fix‑guided Verification Filter，将模型给出的修复代码作为可执行反事实验证，从而显著降低错误拒绝率。

**🔧 技术方法**

技术主要包括统一的三种提示模式（Direct、Direct+Explain、Full），对五种主流 LLM（GPT‑4o、Claude‑4.5‑sonnet、Gemini‑2.0‑flash、Llama‑3.1‑8B、Mistral‑Small‑3.1‑24B）进行大规模实验；使用 GPT‑4o 进行解释一致性和故障意识评估；实现验证过滤器，执行原始与修复代码并对比基准与增量测试结果。

**📊 数据集**

数据集采用三大代码评测基准的配对版本：HumanEval‑X‑Bugs、MBPP‑paired、QuixBugs‑paired，共计约1400条任务（约700个需求），每条包含规范文本、正确实现与对应错误实现。

**📈 对比分析**

通过混淆矩阵计算 FNR（误拒）与 FPR（误准），实验显示：在更复杂提示下 FNR 明显上升（如 GPT‑4o 从 26.2%→73.2%），FPR 下降，说明提示诱导的过度纠错；Fix‑guided Filter 将 FNR 在 HumanEval 上从 70.7% 降至 23.2%，在 MBPP 上从 88.7% 降至 40.0%，显著提升判定可靠性。

**⚠️ 局限性**

局限性包括：①评估仅针对单一语言（Python）的小型函数，难以推广到大型系统级代码；②基准测试用例覆盖不足可能导致误判的真伪判定不完全；③提示设计对结果敏感，未探究多种提示变体的稳健性；④验证过滤器依赖于 GPT‑4o 生成的增量测试，若生成质量不足可能仍漏判或误判。

---

## 667. Adaptive-Growth Randomized Neural Networks for Level-Set Computation of Multivalued Nonlinear First-Order PDEs with Hyperbolic Characteristics

**arXiv ID:** 2603.01093 | [PDF](https://arxiv.org/pdf/2603.01093v1)

**作者:** Haoning Dang `[一作]` (Xi'an Jiaotong University), Fei Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 36601 | [OpenAlex ID](https://openalex.org/A5100455803)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出一种Adaptive‑Growth Randomized Neural Network (AG‑RaNN) 方法，用层生长和自适应采样来求解多值非线性一阶偏微分方程（如 Hamilton‑Jacobi 方程和标量超声波平衡律）的多值解；

**💡 创新点**

其创新点在于把层生长随机网络与自适应管道采样相结合，既保持了随机网络的凸最小二乘训练优势，又通过逐层扩展特征空间和在零等值面附近集中采样，有效缓解了层级增维带来的计算瓶颈，并给出了相应的收敛分析；

**🔧 技术方法**

技术上主要使用层级化的随机神经网络、最小二乘求解、层生长策略、基于管道的自适应采样、以及级别集方法把非线性一阶 PDE 线性化到高维相空间；

**📊 数据集**

实验数据由若干合成 PDE 示例构成，包括 1D/2D 伯格斯方程、Hamilton‑Jacobi 方程以及多值结构的隐式解（如光学相位、半经典 Schrödinger 极限等），不涉及公开数据集；

**📈 对比分析**

与传统随机采样、无层增长的 AG‑RaNN 以及基于差分的数值解法相比，实验表明自适应采样和层生长显著提升了精度、加速了收敛，并在相同计算时间下取得更优或相近的误差；

**⚠️ 局限性**

局限性包括对采样阈值、层数和随机初始化等超参数的敏感性，需要手工调节；在极高维或极端非线性情形下仍受维数灾难限制，且对非光滑多值解的理论收敛性尚未完全覆盖。

---

## 668. Bi-cLSTM: Residual-Corrected Bidirectional LSTM for Aero-Engine RUL Estimation

**arXiv ID:** 2603.00745 | [PDF](https://arxiv.org/pdf/2603.00745v1)

**作者:** Rafi Hassan Chowdhury `[一作]` (Islamic University of Technology), Morsalin Sheikh `[通讯]` (Military Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Bi-cLSTM 模型，用双向 LSTM 与残差纠正机制实现对 aero-engine RUL 的预测。

**💡 创新点**

创新点在于将双向时间建模与自适应残差纠正结合，辅以条件感知预处理，实现对多工况噪声敏感性低的 RUL 预测。

**🔧 技术方法**

使用了 Bi-cLSTM、残差纠正模块、条件归一化、随机森林特征选择、指数平滑、滑动窗口时序输入以及 LSTM/Bi‑LSTM 等深度网络。

**📊 数据集**

数据集为 NASA C‑MAPSS 的四个子集 FD001–FD004，包含多传感器监测信号。

**📈 对比分析**

通过 RMSE、MAE、R² 等指标与多种基准（CNN、LSTM、Bi‑LSTM、Transformer 等）对比，Bi‑cLSTM 在 FD002 与 FD004 上取得最低 RMSE，表现优于前沿方法。

**⚠️ 局限性**

局限在于对 FD001/FD003 等简单工况的性能不如专门模型，且 RUL 截断、窗口长度等预处理参数可能限制了更精细的预测。

---

## 669. CharacterFlywheel: Scaling Iterative Improvement of Engaging and Steerable LLMs in Production

**arXiv ID:** 2603.01973 | [PDF](https://arxiv.org/pdf/2603.01973v1)

**作者:** Yixin Nie `[一作]` (Meta Superintelligence Labs), Kevin Tang `[通讯]` (Meta Superintelligence Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并迭代优化了CharacterFlywheel LLM，专注于提升社交聊天的用户参与度、角色引导性和安全性；

**💡 创新点**

提出了以用户参与度为目标的非可微分优化框架（CharacterFlywheel），结合迭代式奖励模型、在线A/B测试和多元评估指标，形成闭环；

**🔧 技术方法**

采用了强化学习（DPO、GRPO）、奖励模型（点评与对评）、拒绝采样、SFT、用户信号模型、图像生成工具调用以及多种数据管道和安全过滤技术；

**📊 数据集**

使用了内部交互式对话数据、用户在线流量日志、专门标注的互动聊天、静态聊天偏好对比数据、图像生成标注数据，以及公开的标准基准（MMLU、GSM8K、HumanEval 等）；

**📈 对比分析**

通过离线奖励模型赢率、人工对比、社区基准得分以及线上A/B实验对比，15个迭代版本实现了持续的参与度提升（最高 8.8% 广度提升，19.4% 深度提升），且角色遵循率下降 78%，但在基准测试中保持竞争力；

**⚠️ 局限性**

局限性包括奖励模型过拟合导致的性能波动、对复杂多轮互动优化的可扩展性不足、用户信号模型难以直接用于RL、以及对跨语言、跨文化交互的鲁棒性待进一步验证。

---

## 670. Efficient Extractive Summarization with MAMBA-Transformer Hybrids for Low-Resource Scenarios

**arXiv ID:** 2603.01288 | [PDF](https://arxiv.org/pdf/2603.01288v1)

**作者:** Nisrine Ait Khayi `[一作]` (University of Memphis), Nisrine Ait Khayi `[通讯]` (University of Memphis)

**通讯引用:** 51 | [OpenAlex ID](https://openalex.org/A5042977106)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Mamba-Transformer混合架构用于提取式摘要，在低资源条件下可处理全文而无需截断。

**💡 创新点**

将Transformer的句子级语义编码与Mamba状态空间模型的线性时间序列处理结合，实现了无截断、线性复杂度的摘要生成。

**🔧 技术方法**

采用BERT-base进行句子编码，Mamba状态空间网络捕获句间依赖，最后线性分类器进行句子相关性预测，并使用量化、梯度裁剪等技术进行训练优化。

**📊 数据集**

在CNN/DailyMail、DebateSum和ArXiv三大领域各200篇文档的低资源设置下进行实验。

**📈 对比分析**

与BERTSUM、MATCHSUM等基线在ROUGE-1/2/L和推理速度上比较，Mamba-BERT在所有数据集上实现ROUGE-1提升0.17-0.56，新闻摘要推理速度提升24-27%，长文档表现尤为显著。

**⚠️ 局限性**

未显式建模重要性排序与实体优先级；实验规模受限于200篇文档，未覆盖最新抽取式或自回归基线，且对实体和重要性改进需求仍存在。

---

## 671. DriveCode: Domain Specific Numerical Encoding for LLM-Based Autonomous Driving

**arXiv ID:** 2603.00919 | [PDF](https://arxiv.org/pdf/2603.00919v1)

**作者:** Zhiye Wang `[一作]` (Lanzhou University), Jianqiang Wang `[通讯]` (Tsinghua University)

**通讯引用:** 62059 | [OpenAlex ID](https://openalex.org/A5109869303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发 DriveCode，一种专为 LLM 自动驾驶设计的数值编码方法，通过数值投影器将数字映射为连续嵌入，并在输出端使用数值回归头，实现端到端的文本、视觉与数值统一处理；

**💡 创新点**

将数字从离散 token 转化为连续专用模态，使用数值投影器和回归头在输入输出两端显式建模数值，并与文本视觉特征统一拼接，显著提升数值推理精度与推理效率；

**🔧 技术方法**

基于 LLaVA-NeXT 的多模态 Transformer，SigLIP 视觉编码器，MLP 数值投影器，数值回归头；采用并行 LM 头+数值头的自回归生成，混合精度训练与分布式并行；

**📊 数据集**

DriveGPT4（BDD-X+LLM 指令），DriveGPT4-V2（CARLA 规则驱动），OmniDrive（nuScenes+OpenLane‑v2），用于控制信号和轨迹预测的问答式数据集；

**📈 对比分析**

在与 ADAPT、原始 DriveGPT4（文本数字）、xVal 等基线同基座训练的实验中，DriveCode 在 RMSE、A_δ、θ 误差、L2 路径误差、速度误差等指标上普遍取得最小误差，尤其在控制信号 RMSE 和轨迹 L2 方面表现最优；

**⚠️ 局限性**

依赖准确的数字提取与对齐，数值尺度和异常值对性能敏感，受限于基础 LLM 的推理能力，尚未在闭环仿真中验证实时驾驶表现。

---

## 672. QSpy: A Quantum RAT for Circuit Spying and IP Theft

**arXiv ID:** 2603.00950 | [PDF](https://arxiv.org/pdf/2603.00950v1)

**作者:** Amal Raj `[一作]` (Singapore Institute of Technology), Vivek Balachandran `[通讯]` (Singapore Institute of Technology)

**通讯引用:** 378 | [OpenAlex ID](https://openalex.org/A5021323754)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

演示了一种名为 QSpy 的量子远程访问木马（Quantum Remote Access Trojan），能够在用户与云端量子计算服务交互时，悄悄拦截并记录量子电路的序列化数据与执行结果，而不干扰正常工作流程。

**💡 创新点**

创新点在于首次将传统的中间人（MITM）攻击与远程访问木马技术迁移到量子云计算场景，并提供了完整的实现与实验证明，揭示了提交层面在云量子计算中的安全缺口。

**🔧 技术方法**

技术主要包括：在客户端植入自签证书根 CA；使用 mitmproxy 等 HTTPS 代理拦截 IBM Qiskit SDK 的 API 调用；基于 job_id 对提交与结果进行关联；将完整记录通过 C2 服务器进行导出与分析。

**📊 数据集**

未使用公开大规模数据集；实验主要在本地机器上使用 IBM Qiskit SDK 提交若干示例量子电路（如 Grover、QFT 等常见量子算法），并在云端执行后收集结果。

**📈 对比分析**

对比方法：将拦截前后客户端体验对比，发现 QSpy 在时间延迟、错误率等指标上与正常提交无显著差异；通过记录完整的 job_id 与结果对应，展示了完整的截取与关联流程，性能上几乎无额外开销。

**⚠️ 局限性**

局限性包括：实现仅针对 IBM Qiskit 与 Windows 系统；攻击仅为被动拦截，未覆盖主动篡改电路或结果的场景；未评估现有加密、混淆或身份验证机制的抵御能力；在多 SDK（Cirq、PennyLane、Braket）或其他 OS 上的适配仍需进一步研究。

---

## 673. LEAR: Learning Edge-Aware Representations for Event-to-LiDAR Localization

**arXiv ID:** 2603.01839 | [PDF](https://arxiv.org/pdf/2603.01839v1)

**作者:** Kuangyi Chen `[一作]` (Graz University of Technology), Friedrich Fraundorfer `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出双任务学习框架LEAR，用于基于事件相机的LiDAR地图定位，联合估计事件-深度流场和边缘结构；

**💡 创新点**

创新点在于通过跨任务特征融合（CFF）和迭代特征细化（IFR）实现两任务互相强化，显著提升跨模态一致性与对应精度；

**🔧 技术方法**

采用深度流估计网络（改进RAFT）、HED边缘检测器、CFF/IFR模块，并用P5N PnP求解姿态；

**📊 数据集**

使用M3ED（LiDAR地图+事件序列）和DSEC（视差图重建点云）等公开数据集；

**📈 对比分析**

与EVLoc和I2D-Loc比较，LEAR在M3ED多场景下平均翻译误差下降约15-20%，旋转误差下降约20-25%；在DSEC上比EVLoc翻译误差低36%、旋转误差低29%，显著优于基线；

**⚠️ 局限性**

局限性：需要足够的视角重叠和深度变化；在平坦或光滑表面等深度退化场景下可能失效；对初始姿态误差敏感，需较好粗定位。

---

## 674. Information and communications technologies for carbon sinks from economics and engineering perspectives

**arXiv ID:** 2603.01787 | [PDF](https://arxiv.org/pdf/2603.01787v1)

**作者:** Yuze Dong `[一作]` (Guilin University of Electronic Technology), Jinsong Wu `[通讯]` (University of Chile)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

系统综述并评估ICT在碳汇项目中的经济与工程应用，梳理案例、技术架构与绩效指标，并提出未来发展与整合路径。

**💡 创新点**

从双重视角（经济与工程）整合ICT技术，构建跨领域的应用框架，系统对比12项代表性研究，揭示技术缺口与多技术协同机会。

**🔧 技术方法**

区块链（智能合约、NFT、去中心化验证）、AI预测模型（MEEMD‑LSTM、TCN‑Seq2Seq、VMD‑CNN‑BiLSTM‑MLP、PINN、数字孪生）、IoT、遥感GIS、云平台、深度学习与强化学习等。

**📊 数据集**

公开碳市场交易与价格数据（中国碳交易所、欧盟ETS等）、碳价格历史序列、CO₂捕集/储存实验与现场监测数据、卫星遥感与地质模型、金融工具（绿色债券、碳信用等）数据。

**📈 对比分析**

采用交易速度、成本、预测RMSE/MAPE、R²、实时监测准确率等指标与传统方法对比，普遍提升显著（交易速度+40%，成本-15%，RMSE≈0.196，R²>0.98），但模型复杂度高、需进一步验证。

**⚠️ 局限性**

存在数据互操作性差、模型可解释性不足、能源与碳足迹高、缺乏实时控制与多技术集成、政策支持不完善、缺乏长期验证与标准化评估、数据隐私与共享挑战。

---

## 675. Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling

**arXiv ID:** 2603.01864 | [PDF](https://arxiv.org/pdf/2603.01864v1)

**作者:** Alexander Prutsch `[一作]` (Graz University of Technology), Horst Possegger `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 SEAM，一种轻量化的流式轨迹预测框架，利用过去预测端点进行目标中心上下文编码，实现一次性解码而无需多阶段细化；

**💡 创新点**

创新点在于端点感知建模（Endpoint-Aware Modeling），将上一帧预测终点作为锚点提取目标上下文，并在双上下文注意力解码器中融合，显著提升准确性并降低延迟；

**🔧 技术方法**

采用多头自注意力与交叉注意力的双上下文解码器、目标中心编码器、运动感知层归一化、轨迹中继机制以及简化的 MLP 输出网络，整体保持单解码层无迭代；

**📊 数据集**

在 Argoverse 2（单/多主体）上进行评测，并在 Argoverse 1 上做消融实验；

**📈 对比分析**

与 snapshot 与流式基线（RealMotion、DeMo、QCNet、SmartRefine、QCNeXt）对比，SEAM 在 AV2 单主体 brier‑minFDE_6 及多主体指标上取得 SOTA，并在单/多主体场景下的平均在线延迟仅 38 ms；

**⚠️ 局限性**

局限包括目标上下文半径需设为 30 m 以获得最佳平衡，且对更长时序或极端交互场景的鲁棒性仍待提升。

---

## 676. TeraPool: A Physical Design Aware, 1024 RISC-V Cores Shared-L1-Memory Scaled-up Cluster Design with High Bandwidth Main Memory Link

**arXiv ID:** 2603.01629 | [PDF](https://arxiv.org/pdf/2603.01629v1)

**作者:** Yichao Zhang `[一作]` (ETH Zurich), Luca Benini `[通讯]` (University of Bologna)

**通讯引用:** 56800 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

设计并实现了一个 1024 个 RISC‑V 可编程核心共享 4 MB L1 统一地址内存的许多核集群（TeraPool），实现了近 1 GHz 频率、1.89 GFLOPS 单精度峰值性能和 200 GOPS/W 的能效。

**💡 创新点**

创新点包括：
1) 三层层次化组合交叉总线（Tile → SubGroup → Group）实现 4096 个 L1 内存 bank 的物理可实现互连；
2) 结合 HBM2E 主存的高带宽 HBML，支持 97 % 近峰值带宽；
3) 混合地址映射与模块化 DMA 方案，降低寄存器/总线占用；
4) 开源 DRAMSys5.0 动态链接库与完整软件栈，便于社区复现。

**🔧 技术方法**

技术细节：12 FinFET（GF12 LP+）工艺、Synopsys Fusion Compiler 2022、PrimeTime 2022、AMAT 模型驱动的交叉总线设计、AXI‑HBML、模块化 iDMA、混合 L1 地址映射、HBM2E 主存、单周期组合交叉总线与流水线寄存器结合。

**📊 数据集**

评测基准包括常见的数据并行内核：axpy、dotp、gemm、FFT、spmmadd；主要关注随机与局部内存访问模式，未使用特定数据集，但通过这些通用核验证了吞吐与能效。

**📈 对比分析**

与 MemPool（256 core）、Occamy（8 core）以及 NVIDIA H100 的 SM 进行对比。TeraPool 在核心数上比现有最大发布的共享‑L1 集群大 4 倍，单精度峰值 1.89 GFLOPS、IPC 0.7–0.85，能效 100–200 GOPS/W，整体性能/能效显著优于对比系统。

**⚠️ 局限性**

局限性：
1) 需要大量路由通道（≈40 % die area）来实现层次互连，导致占地增大；
2) 远程 Group 访问延迟（7–11 cycle）和 bisection 带宽（1.875 ×）相对较高，影响全局访问性能；
3) 物理实现复杂度高，进一步扩展至更多 Group 或更大核心数时的路由和时序难度提升；
4) 目前针对通用计算，未针对稀疏或量化等特殊工作负载提供专用硬件扩展。

---

## 677. SciDER: Scientific Data-centric End-to-end Researcher

**arXiv ID:** 2603.01421 | [PDF](https://arxiv.org/pdf/2603.01421v1)

**作者:** Ke Lin `[一作]` (William and Mary), Qingyun Wang `[通讯]` (William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个面向科学研究生命周期的端到端系统，通过专门的代理（构思、数据分析、实验与批评）以及自演化记忆机制，实现从数据上传到实验代码执行的完整闭环自动化。

**💡 创新点**

创新点包括：① 数据中心化设计，系统能自行解析多种实验数据格式并生成结构化报告；② 自演化记忆（短期/长期任务/项目层级）实现测试时学习；③ 通过批评代理的反馈循环持续改进输出；④ 将 LLM 与检索增强生成结合，支持跨领域知识检索。

**🔧 技术方法**

主要技术：大规模语言模型（如 GPT‑4、GPT‑4o、LLaMA‑3 等）、检索增强生成（RAG）、自演化记忆模块、专用数据读取器、代码生成框架（如 LlamaIndex、LangChain）、实验执行与监控工具。

**📊 数据集**

使用的数据集包括 AI‑Idea‑Bench 2025（构思评估）、MLE‑Bench（机器学习实验设计）、SciCode（跨学科编码任务）以及 Kepler 外星人光变曲线数据（案例研究）。

**📈 对比分析**

与 AI‑Scientist、AI‑Researcher、AIDE、AIRA、ML‑Master、GPT‑5 等基线进行对比，表现出显著优势：在 AI‑Idea‑Bench 的创新评分中突破 47.06/44.52，MLE‑Bench 的金牌率提升至 36.4%（高于 AIRA 的 28.64%），在 SciCode 主/子任务成功率分别达到 42.71%/15.38%（超过 GPT‑5 的 38.26%/13.85%）。

**⚠️ 局限性**

局限性：依赖 LLM 仍可能产生幻觉或逻辑不一致；数据隐私与版权风险（外部 API 可能泄露敏感实验数据）；对极端专业领域的适配仍有限；系统生成的假设和代码需人工验证，无法完全替代专家判断。

---

## 678. Robust White Blood Cell Classification with Stain-Normalized Decoupled Learning and Ensembling

**arXiv ID:** 2603.01976 | [PDF](https://arxiv.org/pdf/2603.01976v1)

**作者:** Luu Le `[一作]` (University of Technology), Ulas Bagci `[通讯]` (Northwestern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于分离式训练、染色标准化和多模型集成的白细胞分类框架，用于在长尾分布和跨域变化下实现稳健识别。

**💡 创新点**

创新点包括：①在表征学习阶段采用实例均衡采样，②在分类器再平衡阶段使用类均衡采样与有效数权重+焦点调制的混合损失；③利用Macenko染色标准化、TTA和多骨干网络集成，消除显式域适配的需求。

**🔧 技术方法**

技术手段包括：Macenko染色标准化、RandAugment、分离式两阶段训练（cRT）、有效数加权交叉熵、焦点损失、测试时增强（TTA）和ResNet50/ResNet152+Swin Transformer多模型集成。

**📊 数据集**

实验基于WBCBench 2026 Robust White Blood Cell Classification Challenge公开数据集，包含跨染色和扫描设备的真实世界血细胞图像。

**📈 对比分析**

与单模型监督训练和单阶段训练进行对比，并在held-out测试集上进行五折交叉验证。最终集成模型在宏观F1（74.2%）和均衡准确率（77.1%）上优于所有基线，表现最优。

**⚠️ 局限性**

局限性：长尾类别仍存在误分类，尤其是极少数子类型；集成与TTA增加推理时的计算成本；方法依赖染色标准化，若染色协议极端偏离可能影响鲁棒性。

---

## 679. TiledAttention: a CUDA Tile SDPA Kernel for PyTorch

**arXiv ID:** 2603.01960 | [PDF](https://arxiv.org/pdf/2603.01960v1)

**作者:** Taimur Khan `[一作]` `[通讯]` (Helmholtz Centre for Environmental Research), Taimur Khan (Helmholtz Centre for Environmental Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了可修改的 cuTile Python 实现的在线 softmax 前向 Scaled Dot-Product Attention

**💡 创新点**

提供了 Python 级别可调度的 Tile 程序，兼顾性能与可改性，并建立了可复现的基准工作流

**🔧 技术方法**

使用 cuTile Python、PyTorch、CUDA Tile IR、在线 softmax、分块流式 K,V、FP32 累积等技术

**📊 数据集**

使用合成的长序列长上下文 Q、K、V 张量，在 DGX GB10 上进行基准测试

**📈 对比分析**

与 PyTorch 自带的融合 attention 以及非融合基线比较，平均通过率约为 0.63× 融合版，但相对于标准 eager 与 math 路径提升高达 28×

**⚠️ 局限性**

仅实现前向；未包含反向、KV 缓存、融合后处理；仅在 Grace‑Blackwell GPU 上验证，搜索空间有限

---

## 680. VIKIN: A Reconfigurable Accelerator for KANs and MLPs with Two-Stage Sparsity Support

**arXiv ID:** 2603.01165 | [PDF](https://arxiv.org/pdf/2603.01165v1)

**作者:** Wenhui Ou `[一作]` (Hong Kong University of Science and Technology), C. Patrick Yue `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 7786 | [OpenAlex ID](https://openalex.org/A5033252398)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一款可重构的加速器VIKIN，支持Kolmogorov–Arnold网络(KAN)和多层感知机(MLP)的推理。

**💡 创新点**

创新点包括：统一的两模式数据流（管线模式用于KAN，平行模式用于MLP）；可重构的B‑spline单元(SPU)既能计算B‑spline，又可转化为MAC；双阶段稀疏支持提高稀疏度利用率。

**🔧 技术方法**

采用SIMD核、SPU阵列、PE阵列、TSE稀疏编码器，并在FP16下实现硬件加速；在Xilinx Virtex‑7 FPGA上实现并验证。

**📊 数据集**

使用交通拥堵预测数据集Traffic，进行时间序列预测任务。

**📈 对比分析**

与传统MLP对比，KAN在相同硬件上实现1.28×加速、19.58%准确率提升；与边缘GPU对比，KAN实现1.25×速度提升、4.87×能效提升；在更高精度的KAN（3.29×操作）上，延迟仅提升1.24×。

**⚠️ 局限性**

局限性：KAN的优势主要集中在特定任务，对通用场景支持有限；高阶B‑spline参数（大G/K）会带来运算量与硬件利用率不匹配；双模式切换仍需额外控制开销。

---

## 681. Sustainable Care: Designing Technologies That Support Children's Long-Term Engagement with Social Issues

**arXiv ID:** 2603.00996 | [PDF](https://arxiv.org/pdf/2603.00996v1)

**作者:** JaeWon Kim `[一作]` (University of Washington), McKenna F. Parnes `[通讯]` (Treuman Katz Center for Pediatric Bioethics and Palliative Care, Center for Clinical and Translational Research, Seattle Children's Research Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

组织了一场全日研讨会，聚焦技术如何支持儿童持续参与社会议题，并通过三轮圆桌讨论与行动规划来提出可持续关怀的四大主题框架。

**💡 创新点**

创新点在于将技术设计与儿童心理健康相结合，提出“可持续关怀”这一新的设计视角，并将美国疾控中心的八项青少年心理健康建议归纳为四个实践主题。

**🔧 技术方法**

文中未具体实施任何技术；讨论的技术包括游戏、教育平台、社交媒体、AI伙伴等，但主要是理论与案例讨论。

**📊 数据集**

未使用任何数据集，研讨会基于参与者的前置问卷和现场讨论。

**📈 对比分析**

没有进行实验或性能评估；方法为结构化研讨会与圆桌讨论，结果以共识与行动计划为输出。

**⚠️ 局限性**

局限性包括缺乏量化验证、依赖研讨会参与者的主观反馈、参与者背景和规模受限，难以推广到更广泛的实践场景。

---

## 682. Changes in Manuscript Length, Research Team Size, and International Collaboration in the Post-2022 Period: Evidence from PLOS ONE

**arXiv ID:** 2603.01718 | [PDF](https://arxiv.org/pdf/2603.01718v1)

**作者:** Yossi Ben-Zion `[一作]` (Bar-Ilan University), Nitza Davidovitch `[通讯]` (Ariel University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究分析了2019年至2025年间PLOS ONE期刊上发表的109,393篇研究文章的结构性出版指标，包括手稿长度、作者团队规模、参考文献数量和跨语言合作，特别关注2022年后这些指标的变化。

**💡 创新点**

研究发现，2022年后手稿长度显著增加，尤其是非母语英语（NNES）作者的手稿长度增长速度快于母语英语（NES）作者，且NNES作者的团队规模和与NES合作者的合作显著减少。这表明生成性语言模型可能正在重塑科学写作的方式和合作结构。

**🔧 技术方法**

使用了大规模的文献计量数据集，采用了差异中的差异回归分析、负二项回归和逻辑回归等统计技术来分析手稿长度、作者团队规模和国际合作的变化。

**📊 数据集**

数据集包括2019年至2025年间在PLOS ONE上发表的109,393篇研究文章，涵盖多个学科，确保了样本的多样性和代表性。

**📈 对比分析**

与2022年相比，2025年NNES作者的手稿长度增加了8.9%，而NES作者增加了5.3%。NNES作者的团队规模从6.54减少到6.06，合作率下降了36%。这些变化表明，NNES作者在写作和合作模式上经历了显著的转变。

**⚠️ 局限性**

本研究的局限性包括仅限于PLOS ONE期刊，无法确定因果关系，且作者的语言背景分类依赖于机构隶属关系，可能导致系统性误分类。此外，国际合作的下降可能部分受到政策层面的影响，尤其是在中国等国家。

---

## 683. Probabilistic Retrofitting of Learned Simulators

**arXiv ID:** 2603.01949 | [PDF](https://arxiv.org/pdf/2603.01949v1)

**作者:** Cristiana Diaconu `[一作]` (University of Cambridge), Payel Mukhopadhyay `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种在现有确定性 PDE 模型上进行后训练（retro‑fitting）的方法，将其转换为可生成概率预测的模型。

**💡 创新点**

创新点在于：① 利用 CRPS（连续排名概率分数）作为后训练目标，兼顾误差最小化与多样性；② 通过条件层归一化（AdaLN）注入全局噪声，架构无关、代码改动极少；③ 证明该方法在单系统模型与多系统基础模型上均有效，显著提升概率预测质量。

**🔧 技术方法**

技术手段包括：CRPS（fair）损失函数、条件层归一化噪声注入、微分学习率、预训练权重初始化；实现上基于 Walrus、Lola、Poseidon 三种主流架构，并在 640M 参数的 HalfWalrus 基础模型上验证。

**📊 数据集**

使用的模拟数据集：5 种 2D 物理系统（Rayleigh‑Bénard、Euler 多象限、Shear Flow、Turbulent Radiative Layer、Viscoelastic Instability）来自 The Well 数据集，均包含周期边界和不同尺度的混沌特征。

**📈 对比分析**

与传统的确定性微调方法相比，CRPS 后训练在所有模型/数据集上实现了 20–54% 的 CRPS 降低、10–30% 的 VRMSE 改善；在基础模型上可达 34–40% 的 CRPS 提升、>13% 的 VRMSE 提升；并且模型对集成大小具有可调节的性能提升。

**⚠️ 局限性**

局限性：① 依赖于预训练确定性模型的质量，若基础模型尚未收敛，VRMSE 提升有限；② 仅优化单变量边缘分布，未保证空间一致性；③ 受架构约束影响，如 Poseidon 的单步记忆导致过平滑；④ 长期回滚时可能继承并放大原模型的累积误差。

---

## 684. Recursive Think-Answer Process for LLMs and VLMs

**arXiv ID:** 2603.02099 | [PDF](https://arxiv.org/pdf/2603.02099v1)

**作者:** Byung-Kwan Lee `[一作]` (Korea Advanced Institute of Science and Technology), Yong Man Ro `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种递归思考‑回答框架 R‑TAP，通过置信度生成器和递归奖励让模型在推理过程中自我评估并迭代改进答案。

**💡 创新点**

创新点在于将置信度估计嵌入思考‑回答流程，设计递归置信度提升奖励与最终答案置信度奖励，实现无额外推理时延的自我纠错机制。

**🔧 技术方法**

采用置信度生成器、GRPO 强化学习、vLLM+DeepSpeed 训练框架以及两种奖励机制，并在 LLM 与 VLM 上统一实现。

**📊 数据集**

使用 Open‑R1‑Math、codeforce‑cot、PRIME、Phi‑4‑reasoning 系列等语言数据集，以及 Skywork‑R1V2、Geometry‑3K、MMK12‑16K 等多模态数据集进行训练与评估。

**📈 对比分析**

在多语言与视觉语言推理基准上与单通道思考‑回答模型及多种开源/闭源模型对比，R‑TAP 在多种模型上显著提升准确率、减少“Oops”错误，并实现更短的推理时间。

**⚠️ 局限性**

限制在于训练时需要并行生成所有递归轨迹，导致显存与计算开销增加；且推理仍需预设递归深度，缺乏完全动态终止机制。

---

## 685. HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC

**arXiv ID:** 2603.01661 | [PDF](https://arxiv.org/pdf/2603.01661v1)

**作者:** Maoliang Li `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**通讯引用:** 36508 | [OpenAlex ID](https://openalex.org/A5100641667)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套针对移动 SoC 上 agentic RAG 工作流的异构感知调度框架，能够在单请求场景下实现多加速器协同执行并显著降低延迟。

**💡 创新点**

创新点在于构建基于离线性能剖面、PU 亲和性、工作形状敏感性和内存带宽争用的调度模型，并在运行时融合形状感知子阶段划分、关键路径优先映射以及带宽感知并发控制的在线调度策略。

**🔧 技术方法**

主要技术包括离线性能剖析、线性回归模型预测、形状感知子阶段划分、关键性优先级评估、带宽争用惩罚的并发控制以及 C++/OpenCL/NPU QNN 接口实现。

**📊 数据集**

实验使用了四个公开 RAG 数据集——FinqaBench、TruthfulQA、HotpotQA 与 2WikiMultihopQA，并在 Qwen3 与 BGE/Llama3 模型族上验证。

**📈 对比分析**

与 GPU‑only、NPU‑only 以及手工静态映射（Ayo‑like）基线相比，系统在多种工作流与数据集上平均提升 1.5×（相对 Ayo‑like）且在 GPU‑only 上最高可达 10.94× 的延迟下降。

**⚠️ 局限性**

局限性包括仅针对单请求的工作流，需离线剖面和超参数调优，且对极端动态多请求或新型加速器的适应性尚未验证。

---

## 686. S2O: Enhancing Adversarial Training with Second-Order Statistics of Weights

**arXiv ID:** 2603.01264 | [PDF](https://arxiv.org/pdf/2603.01264v1)

**作者:** Gaojie Jin `[一作]` (University of Exeter), Xiaowei Huang `[通讯]` (University of Liverpool)

**通讯引用:** 5485 | [OpenAlex ID](https://openalex.org/A5020085889)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证一种基于权重二阶统计量的对抗训练方法 S^2O，利用 PAC‑Bayes 框架推导改进的鲁棒泛化上界，并通过拉普拉斯近似估计权重相关矩阵，在训练中加入 Frobenius 范数正则化以提升模型鲁棒性与泛化性能。

**💡 创新点**

① 放宽 PAC‑Bayes 对权重独立性的假设，允许使用非球面高斯分布并引入权重相关矩阵；② 将相关矩阵的谱范数与行列式（即二阶统计量）纳入鲁棒泛化上界；③ 设计可直接优化的 S^2O 正则化项；④ 将该方法与现有对抗训练（AT、TRADES、AWP、DDPM 等）无缝结合。

**🔧 技术方法**

理论分析：PAC‑Bayes、拉普拉斯近似、Kronecker‑factored Hessian；算法实现：S^2O 正则化、Frobenius 范数、相关矩阵估计；实验评估：PGD、FGSM、CW、Auto Attack、BPDA 等对抗攻击。

**📊 数据集**

CIFAR‑10 / CIFAR‑100 / SVHN / Tiny‑ImageNet / Imagenette；ViT‑B、DeiT‑S（ImageNet 预训练）以及使用 1M DDPM 生成样本的数据集。

**📈 对比分析**

与 AT、TRADES、TRADES+AWP、LBGAT、DDPM‑增强训练等基线在相同攻击（FGSM、PGD‑20、CW‑20、Auto Attack）下对比。S^2O 在大多数场景下提升 2–3% 的鲁棒准确率，同时保持或提升清晰准确率；在 ViT/DeiT、WideResNet 上同样表现优异。训练开销约 +20% 计算时间。

**⚠️ 局限性**

① 需要额外计算相关矩阵，导致训练时间与内存成本增加；② 仅在实验设定的模型与数据集上验证，未覆盖更大规模网络或其他任务；③ 对抗攻击多样性仍有限，尤其在极端超参数（如 α）下性能波动；④ 相关矩阵估计仍依赖拉普拉斯近似，精度受 Hessian 近似质量影响。

---

## 687. Structure Matters: Evaluating Multi-Agents Orchestration in Generative Therapeutic Chatbots

**arXiv ID:** 2603.00774 | [PDF](https://arxiv.org/pdf/2603.00774v1)

**作者:** Sina Elahimanesh `[一作]` (Saarland University), Abbas Edalat `[通讯]` (Imperial College London)

**通讯引用:** 2396 | [OpenAlex ID](https://openalex.org/A5058087229)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在八天的随机对照试验中评估了三种基于LLM的自我附着技术(SAT)聊天机器人架构的用户感知效果。

**💡 创新点**

创新在于将多代理有限状态机与长期记忆融合，证明架构设计比单纯提示工程更能提升对话自然度。

**🔧 技术方法**

采用GPT‑4o、有限状态机、多代理协同、检索增强生成(RAG)以及对话摘要的长期记忆等技术。

**📊 数据集**

使用自采的66名波斯语使用者对话日志和8天后调查问卷，并结合SAT知识库与27个练习内容。

**📈 对比分析**

通过单向ANOVA和置换检验比较Alpha（多代理FSM）、Beta（单代理）和Gamma（未引导）三组，Alpha在自然度上显著优于其他两组，其他指标虽呈Alpha优势但未达显著。

**⚠️ 局限性**

局限包括短期（仅8天）且样本量有限，受限于波斯语高学历用户群，缺乏长期疗效评估，且使用单项量表衡量情感与信任等构念。

---

## 688. Stereo-Inertial Poser: Towards Metric-Accurate Shape-Aware Motion Capture Using Sparse IMUs and a Single Stereo Camera

**arXiv ID:** 2603.02130 | [PDF](https://arxiv.org/pdf/2603.02130v1)

**作者:** Tutian Tang `[一作]` (Shanghai Jiao Tong University), Cewu Lu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Stereo-Inertial Poser，使用单目立体相机与六个 IMU 实时捕捉全身运动。

**💡 创新点**

创新点是将立体视觉与 IMU 融合，实现度量准确且形状感知的全局平移与局部运动，同时通过形状感知融合模块消除脚滑现象。

**🔧 技术方法**

技术包括基于 MediaPipe 的 2D/3D 关键点检测、三角化得到 3D 关键点、SMPL 体型优化、状态空间模型（SSM）进行 IMU 与视觉的时空融合、形状感知融合网络、RefineNet 以及多项损失（循环一致、脚接触、加速度抖动）。

**📊 数据集**

使用 AMASS、AIST++ 和 TotalCapture 数据集进行训练与评估，并在 AIST++ 通过合成 3D 关键点、噪声与虚拟立体相机进行实验。

**📈 对比分析**

与 RobustCap、HybridCap、PIP 等基线相比，在 AIST++ 上的 TE、SIP、FS 等指标显著优于对手，且实现 200+ FPS 的实时推理。

**⚠️ 局限性**

局限包括 60 FPS 的视觉-IMU 同步导致未充分利用 IMU 高采样率、形状建模仅限 SMPL 参数、对光照与工作空间的敏感性。

---

## 689. Fast Entropy Decoding for Sparse MVM on GPUs

**arXiv ID:** 2603.01915 | [PDF](https://arxiv.org/pdf/2603.01915v1)

**作者:** Emil Schätzle `[一作]` (ETH Zurich), Markus Püschel `[通讯]` (ETH Zurich)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于dtANS的熵编码方法，在GPU上对CSR格式稀疏矩阵进行压缩并在解码时实时执行稀疏矩阵向量乘法（SpMVM）

**💡 创新点**

创新点在于设计了dtANS——一种为GPU并行解码优化的tANS变体，并结合差分编码进一步降低索引熵，从而实现压缩率高、解码速度快的稀疏矩阵格式CSR-dtANS

**🔧 技术方法**

主要技术包括差分编码、dtANS熵编码、warp级数据交错、共享内存表格访问、以及对GPU特性的ILP/SIMT优化

**📊 数据集**

使用SuiteSparse矩阵集合中的8975个浮点型矩阵（不含复数）进行实验

**📈 对比分析**

与cuSPARSE的CSR、COO、SELL三种格式以及AlphaSparse自适应格式对比，CSR-dtANS在2^15以上非零元素且平均每行≥10个非零的矩阵上压缩率可达11.77倍，SpMVM加速率最高可达3.48×，在大多数大矩阵上表现优于cuSPARSE并在部分情况下优于AlphaSparse

**⚠️ 局限性**

局限性包括对小矩阵（<10^5字节）压缩效果差、对行长度不均匀或极少非零行的矩阵性能下降，以及当前实现仅以CSR为基础，未覆盖更适合的稀疏格式

---

## 690. DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs

**arXiv ID:** 2603.01762 | [PDF](https://arxiv.org/pdf/2603.01762v1)

**作者:** Yingjie Tan `[一作]` (Tsinghua University), Yaqing Wang `[通讯]` (Beijing Institute of Mathematical Sciences and Applications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 DGNet，一种在图上离散 Green 函数的神经 PDE 求解器，旨在提高数据效率。

**💡 创新点**

创新点包括：将 Green 函数的叠加原理离散化为图算子，将物理先验与 GNN 修正相结合，构建出强的结构性先验；在极少训练轨迹下实现高精度预测，并能在未见源项下保持零射击泛化。

**🔧 技术方法**

技术手段：离散 Green 函数和 Crank–Nicolson 时间积分、基于网格的梯度/拉普拉斯物理算子、GNN 编码‑处理‑解码的校正子、残差 GNN 细化、稀疏 LU 分解加速求逆。

**📊 数据集**

数据集：模拟 Allen–Cahn、Fisher–KPP、FitzHugh–Nagumo、含障碍物的污染物输运（圆柱、沉积物、复合障碍）以及激光加热（移动源）的数值轨迹，共仅数十条训练轨迹。

**📈 对比分析**

与 DeepONet、MGN、MP-PDE、BENO、PhyMPGN 在同一限量数据设置下比较；DGNet 的 MSE 低 1–2 个数量级，RNE 也显著优于基线，并在未见源项上保持误差不升高，显示出卓越的数据效率和泛化性能。

**⚠️ 局限性**

局限性：只适用于满足叠加原理的线性 PDE，难以推广到非线性或准线性 PDE；3D 大规模系统的求解仍面临计算和存储挑战。

---

## 691. Preference Score Distillation: Leveraging 2D Rewards to Align Text-to-3D Generation with Human Preference

**arXiv ID:** 2603.01594 | [PDF](https://arxiv.org/pdf/2603.01594v1)

**作者:** Jiaqi Leng `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24310 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种Preference Score Distillation（PSD）方法，用于在不需要3D训练数据的前提下通过2D奖励模型对文本到3D的生成过程进行人类偏好对齐；

**💡 创新点**

其创新点在于将偏好优化重新表述为基于Classifer-Free Guidance的指导信号，并在Score Distillation过程中在线构造胜负对，实现了直接利用预训练二维奖励模型的偏好引导；

**🔧 技术方法**

技术上结合了Score Distillation、CFG式偏好指导、动态对抗的win‑lose采样、以及负文本嵌入的自适应更新；

**📊 数据集**

实验主要使用Eval3d 200个测试提示、MVDream、Stable Diffusion v2.1等生成管线，并评估ImageReward、PickScore、Aesthetic、MPS及VQA等指标；

**📈 对比分析**

与RichDreamer、Trellis、DreamReward、DreamDPO等现有方法相比，PSD在文本对齐度、视觉质量和偏好得分上均有显著提升，且在用户研究中获得更高的主观评价；

**⚠️ 局限性**

局限性包括对奖励模型的依赖，可能出现奖励破解现象，以及在极高分辨率或复杂视角下的计算开销仍有提升空间。

---

## 692. RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design

**arXiv ID:** 2603.01229 | [PDF](https://arxiv.org/pdf/2603.01229v1)

**作者:** Tianxing Chen `[一作]` (Hong Kong University), Ping Luo `[通讯]` (Hong Kong University)

**通讯引用:** 53823 | [OpenAlex ID](https://openalex.org/A5100752686)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了RMBench基准和Mem-0记忆型策略，用于系统评估机器人操作中的记忆需求。

**💡 创新点**

创新点在于引入任务记忆复杂度指标、设计多级记忆的模块化策略，并对不同记忆机制进行可控消融分析。

**🔧 技术方法**

主要技术包括视觉语言模型、双系统规划与执行模块、锚点记忆、滑动记忆窗口以及子任务结束分类器。

**📊 数据集**

使用RoboTwin 2.0平台生成的9个双臂仿真任务（M(1)与M(n)）以及对应的真实机器人演示数据。

**📈 对比分析**

与现有非预训练和预训练策略对比，Mem-0在M(1)任务平均提升38.4%、M(n)任务提升21.2%成功率；在真实世界3个任务上亦优于ACT和Pi0.5。

**⚠️ 局限性**

局限性包括对语义理解的依赖、按钮按压检测不稳、子任务终止分类器简单、滑动记忆可能产生干扰以及缺乏专门的低层预训练。

---

## 693. Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

**arXiv ID:** 2603.00903 | [PDF](https://arxiv.org/pdf/2603.00903v1)

**作者:** Ke Sun `[一作]` (University of Alberta), Linglong Kong `[通讯]` (University of Alberta)

**通讯引用:** 1150 | [OpenAlex ID](https://openalex.org/A5062334200)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种双学习者框架 FAME，结合快学习者与元学习者以实现持续强化学习中的知识转移与知识整合。

**💡 创新点**

创新点包括：①用 MDP 距离与灾难性遗忘度量为持续 RL 提供理论基础；②引入自适应元预热的假设检验机制以避免负迁移；③在元学习者中给出基于 KL 与 Wasserstein 的增量更新规则，实现知识整合。

**🔧 技术方法**

技术上采用 DQN/PPO/SAC 等基线 RL 算法，配合软最大策略映射、行为克隆正则化、元缓冲区等实现双学习者的交互更新。

**📊 数据集**

使用的评测数据集包括 MinAtar、Atari 游戏、Meta‑World 机器人操作环境。

**📈 对比分析**

与多种基线（如 Finetune、MultiHead、PackNet、ProgressiveNet、PT‑DQN 等）比较，FAME 在平均性能、正向迁移和遗忘率上显著优于对手，尤其在连续动作空间中实现了零遗忘。

**⚠️ 局限性**

局限性在于假设已知任务边界与统一状态动作空间；元学习者需要保存全部知识，且未探索更高效的潜在表示或在线任务辨识。

---

## 694. DeLo: Dual Decomposed Low-Rank Experts Collaboration for Continual Missing Modality Learning

**arXiv ID:** 2603.01632 | [PDF](https://arxiv.org/pdf/2603.01632v1)

**作者:** Xiwei Liu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Imran Razzak `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 10602 | [OpenAlex ID](https://openalex.org/A5033585021)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DeLo 框架，解决持续缺失模态学习（CMML）问题；

**💡 创新点**

创新点在于双分解低秩专家架构（Modality‑Specific Factor Pools + Dynamic LoRA 分解）以及跨模态引导路由和任务键记忆实现任务无关推理；

**🔧 技术方法**

采用 LoRA 的分解专家池、模态特定因子池、Cross‑Modal Guided Routing、Task‑Key Memory、对齐与一致性损失等技术；

**📊 数据集**

使用 UPMC‑Food101‑CMML 和 MM‑IMDb‑CMML 两个公开 CMML 基准数据集；

**📈 对比分析**

与 MAP、MSP、L2P、DualPrompt、RebQ 等基线比较，DeLo 在 AP（平均性能）上显著领先，同时保持或降低平均遗忘（FG），显示出更优的稳定性-可塑性平衡；

**⚠️ 局限性**

局限性包括：对多标签任务的遗忘略高，依赖跨模态对齐的质量，对极端缺失比例和跨域迁移的鲁棒性仍待进一步验证。

---

## 695. UniHM: Unified Dexterous Hand Manipulation with Vision Language Model

**arXiv ID:** 2603.00732 | [PDF](https://arxiv.org/pdf/2603.00732v1)

**作者:** Zhenhao Zhang `[一作]` (ShanghaiTech University), Jingya Wang `[通讯]` (ShanghaiTech University)

**通讯引用:** 8724 | [OpenAlex ID](https://openalex.org/A5100639519)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniHM 框架，实现基于自由文本指令的全序列灵巧手操作生成和执行。

**💡 创新点**

1) 统一手势词典共享码本，实现跨手型通用；2) 仅依赖人类交互视频学习，无需大规模遥控数据；3) 物理引导动态优化，保证生成轨迹可执行。

**🔧 技术方法**

VQ‑VAE 统一码本、视觉语言模型（Qwen3‑0.6B + CLIPort）、光深度感知 + Point‑SAM、生成式 VLM 与掩码训练、基于能量的物理优化（接触、生成、时间先验）。

**📊 数据集**

DexYCB 与 OakInk 人机交互视频数据集，配合 GPT‑4o 注释生成自由文本指令，并使用 MANO retargeting 与 Dex‑Retargeting 生成多手模型。

**📈 对比分析**

与 TM2T、MDM、FlowMDM、MotionGPT3 等基线在 DexYCB 与 OakInk 上进行对比；在 MPJPE、FOL、FPL、FID、Diversity 等指标上均优于基线；实测抓取成功率显著提升，尤其在未见对象上。

**⚠️ 局限性**

仅支持单手；缺乏触觉/力觉感知；接触先验简化；未覆盖双手或工具使用；需要 RGB‑D 传感器，无法在低光/无深度环境下工作。

---

## 696. MOSAIC: A Unified Platform for Cross-Paradigm Comparison and Evaluation of Homogeneous and Heterogeneous Multi-Agent RL, LLM, VLM, and Human Decision-Makers

**arXiv ID:** 2603.01260 | [PDF](https://arxiv.org/pdf/2603.01260v1)

**作者:** Abdulhamid M. Mousa `[一作]` (Beijing Institute of Technology), Ming Liu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 22565 | [OpenAlex ID](https://openalex.org/A5100347797)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个统一平台MOSAIC，支持RL、LLM、VLM和人类决策者在同一多智能体环境中部署、评估与比较。

**💡 创新点**

创新点在于引入IPC基于worker的隔离协议、统一的Operator抽象层、支持共享种子下的确定性跨范式评估以及图形化实时可视化。

**🔧 技术方法**

采用Qt6 GUI、gRPC/JSON IPC、Protocol Buffers、Python子进程以及对CleanRL、XuanCe、RLlib、BALROG等第三方RL/LLM框架的无侵入式封装，兼容Gymnasium/PettingZoo多环境标准。

**📊 数据集**

兼容26个环境家族（如MiniGrid、BabyAI、Overcooked、Soccer、MultiGrid等）和LLM/ VLM数据集（如BALROG、AgentBench）。

**📈 对比分析**

通过手动模式（锁步同步、共享种子）和脚本模式（声明式Python驱动）实现跨范式行为对齐，统一生成JSONL Telemetry；实验表明结果可复现，RL与LLM/人类在任务性能与协作效果上存在显著差异。

**⚠️ 局限性**

仍受限于不同框架的接口差异、LLM算力与延迟、人工输入交互的主观性，以及需要进一步扩展更多环境与模型以提升跨范式一致性。

---

## 697. UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation

**arXiv ID:** 2603.01418 | [PDF](https://arxiv.org/pdf/2603.01418v1)

**作者:** Hebeizi Li `[一作]` (Beihang University), Yi Yang `[通讯]` (Huawei)

**通讯引用:** 81224 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

UniTalking 提出了一个统一端到端扩散框架，用多模态 Transformer 同时生成高保真语音和与之同步的唇形视频。

**💡 创新点**

创新点在于引入联合注意力的多模态 Transformer 块，实现潜在空间中细粒度的音频-视频对齐，并支持个性化语音克隆。

**🔧 技术方法**

使用的技术包括 Flow Matching 的连续归一化流、分类器无指导、MM‑DiT、共享自注意力、RoPE 等。

**📊 数据集**

数据集为 2.3M 条公开对齐音视频对话样本（OpenHumanVid + 内部收集），以及通过 IndexTTS2 生成的参考音频。

**📈 对比分析**

与 Sora2、Universe‑1、OVI 等方法比较，UniTalking 在音频质量和同步度上分别提升 116%/107%，视频质量相当，口型同步指标优于 Universe‑1 并接近 Sora2。

**⚠️ 局限性**

局限在于模型规模受限、未支持多人物参考生成、对比实验受公开数据规模限制。

---

## 698. Black Hole Search: Dynamics, Distribution, and Emergence

**arXiv ID:** 2603.00766 | [PDF](https://arxiv.org/pdf/2603.00766v1)

**作者:** Tanvir Kaur `[一作]` (Indian Institute of Technology Ropar), Kaushik Mondal `[通讯]` (Indian Institute of Technology Ropar)

**通讯引用:** 238 | [OpenAlex ID](https://openalex.org/A5059765803)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在1-有限1-间隔连通动态图中使用2δ_BH+17个散落式移动代理解决黑洞搜索（1-BHS）问题，并在任意静态图中用3个共存代理解决可出现时的黑洞搜索（Ebhs）问题。

**💡 创新点**

针对散落初始配置的1-BHS给出了近乎最优的代理数（2δ_BH+17），并在动态图中实现了O(m^2)时间复杂度；在Ebhs中实现了在环形网络上最优O(n)时间、任意静态网络上多项式时间且只需3个代理，且不依赖全局参数。

**🔧 技术方法**

采用了“个体谨慎移动（ICM）”和“链式谨慎移动”策略；使用白板存储有限信息（每节点O(log n)位），利用ID优先级协调代理行为；基于通用探索序列（UXS）实现无节点存储的全局探索；结合群组形成机制触发根节点算法。

**📊 数据集**

无数据集；论文为理论分析与算法设计，主要以图的规模n、m及黑洞度δ_BH为参数进行证明。

**📈 对比分析**

通过数学证明和案例分析，展示算法在最坏情况下可在O(m^2)时间内完成1-BHS，Ebhs在环形网络上实现线性时间，且在任意静态图上满足多项式时间。相比现有工作，代理数和时间复杂度均实现了新的下界或最优性能。

**⚠️ 局限性**

局限性包括：对动态图的模型仅限于1-有限1-间隔连通；在Ebhs中对黑洞出现时间做了多项式时间的假设；算法在实际网络环境中需要同步时钟与白板实现，且对代理数的上界仍未能达到理论最小（2δ_BH）。

---

## 699. The Observer-Situation Lattice: A Unified Formal Basis for Perspective-Aware Cognition

**arXiv ID:** 2603.01407 | [PDF](https://arxiv.org/pdf/2603.01407v1)

**作者:** Saad Alqithami `[一作]` `[通讯]` (Al-Baha University), Saad Alqithami (Al-Baha University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了Observer‑Situation Lattice（OSL）框架，用以统一表示和推理多代理系统中的观察者与情境视角。

**💡 创新点**

创新点在于将观察者与情境构造为完备格的笛卡尔积，形成单一的视角空间，并设计了针对该格的增量信念传播（RBP）和冲突分解（MCC）算法。

**🔧 技术方法**

技术手段包括格理论、贝叶斯/可信度权重的信念语义、递归式增量传播、图算法求冲突连通分量，以及在BDI架构中整合OSL的统一数据模型。

**📊 数据集**

实验使用合成信念记录、经典Theory of Mind测试（如Sally‑Anne、层级信念等）以及与ATMS、DTMS、MEPK等基准系统的性能对比。

**📈 对比分析**

与基准相比，OSL在10^5个格元素的平衡格上实现了子线性更新时间（≈0.42幂次），内存占用仅为传统系统的一半，且在ToM任务中实现了近乎即时（<1 ms）的正确推理。

**⚠️ 局限性**

局限性包括：格结构固定且有限，无法动态插入/删除节点；缺乏连续情境变量和概率可信度；单机实现受限于约10^4–10^5个元素，未在真实复杂域（如智能建筑、多传感器融合）中验证。

---

## 700. Accurate, private, secure, federated U-statistics with higher degree

**arXiv ID:** 2603.01986 | [PDF](https://arxiv.org/pdf/2603.01986v1)

**作者:** Quentin Sinh `[一作]` (Inria), Jan Ramon `[通讯]` (Inria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于多方安全计算（MPC）的联邦学习框架，用以在满足中央差分隐私的前提下高效计算高阶U统计量（k≥2）

**💡 创新点**

通过自适应边采样（Balanced Sample）和精细的误差分析，将MSE降至O(1/(n²ε²))，显著优于现有LDP和Shuffle模型方案，同时保持通信和计算成本低

**🔧 技术方法**

核心技术包括：Additive Secret Sharing、MPC实现的安全功能（ℱ_f与ℱ_noise）、Johnson–Lindenstrauss降维、离散化与固定精度表示以及可插拔的噪声产生子协议

**📊 数据集**

在Synthetic（[0,1]均匀分布）和Bank Marketing（包含年龄、年均余额等特征）两个数据集上进行实验，验证不同统计量（Gini差异、Kendall τ、重复对比例）

**📈 对比分析**

与Bell、Ghazi、Shuffle和LDP等基线协议对比，实验显示：在通信量与MSE方面表现最佳；在单方计算成本上与Ghazi相当或更低；服务器端计算相对更轻量

**⚠️ 局限性**

局限性包括：仅针对U统计量的固定k≥2，依赖于至少部分诚实方的安全模型；对离散化参数t影响不大但仍需预先设定；实现中需结合具体MPC库，可能存在实用性和部署复杂度的挑战

---

## 701. SkillCraft: Can LLM Agents Learn to Use Tools Skillfully?

**arXiv ID:** 2603.00718 | [PDF](https://arxiv.org/pdf/2603.00718v1)

**作者:** Shiqi Chen `[一作]`, Yee Whye Teh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SkillCraft 基准与 Skill Mode 评估协议，研究大型语言模型在多步骤工具使用中自动构造、缓存并重用高级工具组合的能力。

**💡 创新点**

创新点在于将可重复的子结构嵌入任务并引入轻量级的 Skill 库接口，使代理能够在测试时动态生成代码化的 Skill，并通过自动验证与存储显著提升效率与成功率。

**🔧 技术方法**

采用大语言模型、MCP 轻量级接口、代码验证器、层次化与迭代模式以及自动化工具调用与执行监控等技术。

**📊 数据集**

使用包含 126 个长周期、可扩展工具使用任务的 SkillCraft 数据集，任务来源于 Toolathlon、AgentCompany、WebArena 等基准，并按数量与复杂度分为易、中、难三个难度等级。

**📈 对比分析**

在 Gemini、Claude、GPT 等多款顶尖模型上对比基线与 Skill Mode，Token 使用率下降约 80%，成功率提升至 90%+，效率与成功率与模型实力呈正相关。

**⚠️ 局限性**

局限性包括层次化技能易产生错误传播与调试成本高，深层自动生成的层次化技能在成功率与效率上不如浅层技能；同时 Skill Mode 对模型的编程生成与执行能力依赖较大，低性能模型受益有限。

---

## 702. CoVAE: correlated multimodal generative modeling

**arXiv ID:** 2603.01965 | [PDF](https://arxiv.org/pdf/2603.01965v1)

**作者:** Federico Caretti `[一作]` (Scuola Internazionale Superiore di Studi Avanzati), Guido Sanguinetti `[通讯]` (Scuola Internazionale Superiore di Studi Avanzati)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种能够捕捉多模态间相关性的变分自编码器CoVAE，解决了传统多模态VAE在潜在空间融合后导致的相关性丢失问题。

**💡 创新点**

创新点在于使用非对角高斯先验（协方差矩阵）来建模模态间相关，并通过联合编码器学习正确的条件分布，从而在缺失模态时能给出更准确的不确定性估计。

**🔧 技术方法**

采用变分自编码器框架、Cholesky分解参数化协方差、Deep CCA预训练、联合与单模态编码器的多任务训练以及条件采样策略。

**📊 数据集**

在合成数据以及TCGA Pan‑Cancer数据集（含mRNA和miRNA两种模态）上进行实验。

**📈 对比分析**

与JMVAE、MVAE、MMVAE、MVTCAE、MoPoE、DMVAE等多模态VAE进行对比，CoVAE在跨模态重建、条件生成（MAE、Spearman相关系数）等指标上达到或接近最佳表现，尤其在条件重建误差和Spearman相关性上优于多数基线。

**⚠️ 局限性**

局限性包括：假设所有相关性可通过全局高斯协方差捕捉，低相关性场景下生成样本可能偏离数据流形；需要为不同子集训练多组编码器，扩展性受限；训练过程对协方差矩阵的优化受限于预训练方法。

---

## 703. MIDAS: Multi-Image Dispersion and Semantic Reconstruction for Jailbreaking MLLMs

**arXiv ID:** 2603.00565 | [PDF](https://arxiv.org/pdf/2603.00565v1)

**作者:** Yilian Liu `[一作]` (Beijing University of Posts and Telecommunications), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 49304 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多图分散与语义重构框架 MIDAS，用于在多模态大语言模型（MLLM）中绕过安全防护生成有害内容。

**💡 创新点**

创新点：①将有害语义拆分成子单元并分布在多张图片中；②使用游戏式视觉推理模板（如字母算式、拼图、排序等）让模型逐步解码；③在文本通道采用层级角色驱动和人格诱导，引导模型完成跨图重构；④通过多图链式推理显著延迟有害信息曝光，削弱安全关注。

**🔧 技术方法**

技术：语义提取器、分散引擎、游戏式视觉推理模板、跨图推理与重构模块、角色驱动与人格诱导文本模板、局部解码器、实验评估脚本。

**📊 数据集**

数据集：HADES、AdvBench、MM‑SafetyBench（tiny），以及多款闭源和开源 MLLM（GPT‑4o、GPT‑5‑Chat、Gemini‑2.5‑Pro、QVQ‑Max、Qwen2.5‑VL、InternVL‑2.5 等）。

**📈 对比分析**

与 5 种现有视觉/文本混合攻击方法（FigStep、HADES、SI‑Attack、VisCRA、HIMRD）对比；在所有模型上均获得最高攻击成功率（ASR）和危害评分（HR），例如在 Gemini‑2.5‑FT 上 ASR > 90%，在闭源 GPT‑5‑Chat 上也能保持 60% 以上；同时在多任务测试中效率更高（执行时间比 VisCRA 低 60%）。

**⚠️ 局限性**

局限性：①需要手工设计多种游戏式模板，难以自动化；②对极强对齐模型（如某些商业闭源系统）仍可能被检测到；③攻击基于黑盒或灰盒，未验证在对抗性输入过滤器极度严格的环境下的鲁棒性；④实验仅覆盖部分安全基准，尚未评估在更广泛的场景（如多轮对话、视频输入）中的表现。

---

## 704. Leave-One-Out Prediction for General Hypothesis Classes

**arXiv ID:** 2603.02043 | [PDF](https://arxiv.org/pdf/2603.02043v1)

**作者:** Jian Qian `[一作]` (University of Hong Kong), Jiachen Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种名为Median of Level-Set Aggregation（MLSA）的两层聚合框架，用于构造转导式留一法预测并给出乘法型LOO泛化不等式；

**💡 创新点**

创新点在于：1）将LOO误差与经验风险最优值直接联系，提供可数据依赖的乘法型上界；2）利用局部级集增长（local level‑set growth）条件来控制近似ERM级集的扩张；3）通过对一组容差（tolerance）取中位数，解决单一容差无法同时满足所有LOO子样本的问题；4）将此框架应用于VC分类、凸回归、密度估计与逻辑回归，给出相应的复杂度项；

**🔧 技术方法**

主要技术包括：经验风险级集构造、局部级集增长分析、聚合规则的稳定性（如投票、平均、Jensen不等式）、中位数聚合层、体积/几何论证（椭圆体积与协方差矩阵）以及光滑化处理；

**📊 数据集**

本文并未在真实数据集上进行实验，而是对任意固定数据集给出理论证明；在特定案例（VC、凸回归、密度、逻辑回归）中使用理论上可构造的“标准”假设族和损失函数；

**📈 对比分析**

与以往仅针对特定结构（如线性模型、SVM、Ridge）或在期望意义下给出的LOO上界相比，本文给出更一般、乘法型的不等式，复杂度项为O(d log n)或O(log |H|)，在可实现的范围内接近最优；

**⚠️ 局限性**

局限性包括：1）需要满足局部级集增长条件，某些模型或损失可能难以验证；2）聚合层在计算上可能需要枚举或近似级集，导致实用性受限；3）结果是针对转导式LOO，未直接给出自举或在线场景的推断；4）常数与具体实现细节对实际误差影响较大。

---

## 705. RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models

**arXiv ID:** 2603.00724 | [PDF](https://arxiv.org/pdf/2603.00724v1)

**作者:** Andrew Zhuoer Feng `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 15804 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过LLM工具调用与代码生成，实现奖励函数的自动构建、选择与调用，使RL奖励系统随训练过程自适应演化。

**💡 创新点**

将奖励设计视为动态工具合成与路由，利用LLM检索、WrapLLM封装专用奖励模型、CodeVerify生成可验证脚本，形成自我演化的奖励库。

**🔧 技术方法**

采用LLM Agent、WrapLLM检索、CodeVerify代码生成、工具库管理、GRPO强化学习框架以及LLM-as-Judge对标。

**📊 数据集**

使用多任务数据集：数学推理（GSM8k、Hendrycks-Math、AIME）、代码（LeetCode、MBPP）、翻译（Flores-200、WMT-24）以及对话（UltraChat）。

**📈 对比分析**

与传统静态分类/生成奖励模型和GPT‑5-as‑Judge基线对比，RLAR在多数基准上提升10%–60%，在OOD数据仍保持优势，并显著降低API与GPU成本。

**⚠️ 局限性**

限制包括未扩展至多模态/音频任务、依赖仓库README可信度、仅验证文本分类奖励以及对极大模型扩展性待进一步研究。

---

## 706. "When to Hand Off, When to Work Together": Expanding Human-Agent Co-Creative Collaboration through Concurrent Interaction

**arXiv ID:** 2603.02050 | [PDF](https://arxiv.org/pdf/2603.02050v1)

**作者:** Kihoon Son `[一作]` (KAIST), Juho Kim `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在 Figma 上构建两套交互原型，探究了人类与 AI 设计助手在实时协作中的交互模式，尤其是并行编辑与动态上下文感知。

**💡 创新点**

创新点在于提出并实现了“Collaborative Linked Executive Operator”——一种能实时识别用户并发操作意图并即时调整执行计划的协作式代理；并基于用户交互日志构建了包含五类交互模式、六类触发因素与四类启用因素的决策模型。

**🔧 技术方法**

核心技术包括 ReAct 结构的 LLM（Claude Sonnet 4.5/Haiku 4.5）驱动的工具调用、过程可视化、工作空间意识与用户行为识别模块；数据通过 38 种 Figma 操作映射实现。

**📊 数据集**

收集了 10 名专业设计师在两天实验中产生的 214 条交互轮（包含行动类型、触发、启用因子与说明），并公开了完整的日志数据集。

**📈 对比分析**

实验未与其他现有设计代理做性能对比，而是通过定性编码与触发因子分析，展示该代理支持 31.8% 并行交互、70.1% 全托管等比例，证明其在协作灵活性上的优势。

**⚠️ 局限性**

局限性包括样本量小、实验时间短、只测试了 ReAct 架构、未评估执行速度对用户体验的影响，以及缺乏跨域或长周期验证。

---

## 707. Reasoning as Gradient: Scaling MLE Agents Beyond Tree Search

**arXiv ID:** 2603.01692 | [PDF](https://arxiv.org/pdf/2603.01692v1)

**作者:** Yifei Zhang `[一作]` (Nanjing University), Jiang Bian `[通讯]` (Microsoft Research Asia)

**通讯引用:** 13807 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一款名为Gome的机器学习工程（MLE）代理，利用梯度优化框架替代传统树搜索，以提高在MLE-Bench上的性能。

**💡 创新点**

创新点包括：将结构化诊断推理映射为梯度计算；将成功记忆视为动量机制；利用多轨执行实现分布式优化，从而在LLM推理能力增强的情况下实现更高效的梯度更新。

**🔧 技术方法**

技术手段涵盖：梯度优化、动量记忆、结构化诊断推理、LLM推理与多轨分布式执行，并与传统树搜索做对比。

**📊 数据集**

使用数据集：MLE-Bench（在12小时预算内的单V100 GPU实验）。

**📈 对比分析**

比较方法：在闭环协议下与树搜索等基线进行对比，Gome在MLE-Bench上获得35.1%的any-medal率，显示出显著的性能提升。

**⚠️ 局限性**

局限性：对LLM推理准确性高度依赖，弱模型仍需树搜索补偿；实验规模受限于单GPU环境，缺乏大规模多GPU验证。

---

## 708. Inference-Time Safety For Code LLMs Via Retrieval-Augmented Revision

**arXiv ID:** 2603.01494 | [PDF](https://arxiv.org/pdf/2603.01494v1)

**作者:** Manisha Mukherjee `[一作]` (Carnegie Mellon University), Vincent J. Hellendoorn `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3181 | [OpenAlex ID](https://openalex.org/A5009679905)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SOSecure，一个在代码生成后通过检索Stack Overflow安全讨论来修正LLM生成代码的推理时安全机制。

**💡 创新点**

首次将社区经验作为推理时安全信号，使用检索增强生成在不需要再训练或微调的情况下提升代码安全性。

**🔧 技术方法**

检索增强生成（RAG）+ BM25检索、提示工程、GPT‑4推理、静态分析工具（CodeQL、Bandit）

**📊 数据集**

SALLM、LLMSecEval、LMSys（含Python与C代码子集）

**📈 对比分析**

与prompt‑only、revision‑only、GPT‑4+CWE三种基线对比，使用Fix Rate和Intro Rate衡量，SOSecure在三数据集上Fix Rate提升22.6%–59.2%，LMSys上达96.7%，且未新增漏洞。

**⚠️ 局限性**

依赖静态分析的评估、检索质量与时效性限制、仅使用词法检索、未进行功能性回归测试，且社区讨论可能过时或不完整。

---

## 709. SWE-Adept: An LLM-Based Agentic Framework for Deep Codebase Analysis and Structured Issue Resolution

**arXiv ID:** 2603.01327 | [PDF](https://arxiv.org/pdf/2603.01327v1)

**作者:** Kang He `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**通讯引用:** 46263 | [OpenAlex ID](https://openalex.org/A5031161187)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个两代理框架SWE-Adept，用于仓库级软件缺陷定位与修复；定位代理采用深度优先、依赖引导的搜索并通过两阶段筛选提升定位精度；修复代理利用工具-内存接口、Git版本控制与检查点机制，实现系统化的迭代修复与回滚。

**💡 创新点**

①首次将深度优先、依赖感知的搜索与两阶段筛选结合用于缺陷定位；②引入工具-内存接口与语义检查点，实现代理驱动的版本控制与高可靠性长周期修复；③通过两代理协同工作，避免单一代理的“思考-编辑”无序流程。

**🔧 技术方法**

大型语言模型(LLM)、Tree-sitter代码解析、代码结构树、agent-directed DFS搜索、工具-内存接口、Git版本控制、检查点管理、动态规划与回滚。

**📊 数据集**

SWE-Bench Lite 与 SWE-Bench Pro 两个基准，均为从GitHub issue提取的真实仓库级问题。

**📈 对比分析**

与基线相比，SWE-Adept在文件级定位Acc@3、函数级定位Acc@5均位居榜首；在完整修复任务中，使用GPT-5.2时提高了解决率3.3%–4.7%，使用Claude-Sonnet-4.5时提高2.6%–4.0%。相比图结构与基于嵌入检索的方案，SWE-Adept在定位准确性与修复成功率上均有显著提升。

**⚠️ 局限性**

依赖商业LLM，成本高；仅在Python代码库验证，跨语言迁移需重新实现解析与索引；对极大仓库仍受上下文窗口限制；缺少针对非结构化代码的鲁棒性评估。

---

## 710. HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images

**arXiv ID:** 2603.02210 | [PDF](https://arxiv.org/pdf/2603.02210v1)

**作者:** Yichen Liu `[一作]` (University of Chinese Academy of Sciences), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套高保真度参考图像修复框架，用于在给定文本提示、遮挡人像和产品参考图像的情况下生成细节丰富的人-产品合成图像。

**💡 创新点**

核心创新点包括：1) 高频图谱引导的DiT框架与标记合并机制；2) Shared Enhancement Attention（SEA）在双流视觉块中共享参数，强化高频细节；3) Detail-Aware Loss（DAL）通过像素级高频监督提升细节保真度；4) 通过自合成与自动过滤构建的大规模HP-Image-40K数据集。

**🔧 技术方法**

技术手段：基于FLUX.1-Dev的扩散模型（DiT），离散傅里叶变换提取高频图谱，token合并与双流注意力，SEA与DAL，流匹配训练策略。

**📊 数据集**

使用了自研的HP-Image-40K（40,000+ 样本）以及约14,000张内部样本，所有数据均通过自合成+自动过滤确保高质量与多样性。

**📈 对比分析**

与Paint-by-Example、ACE++、Insert Anything、FLUX-Kontext等四种基线方法在相同分辨率（1024×576）下进行对比。评价指标包括文本对齐（CLIP‑T）、视觉相似度（CLIP‑I、DINO）、结构相似度（SSIM、SSIM‑HF）及图像质量（LAION‑Aes、Q‑Align‑IQ）。实验结果显示，本框架在所有指标上均达到或接近最优，尤其在视觉一致性和细节保真度上显著优于对照组。

**⚠️ 局限性**

局限性：1) 依赖大量自合成数据，合成质量与真实场景差距仍可能影响泛化；2) 对极小遮挡区域或极端光照/纹理变化的细节重建仍存在轻微失真；3) 目前仅针对静态图像，未扩展到视频或动态图像。

---

## 711. Actor's Note: Examining the Role of AI-Generated Questions in Character Journaling for Actor Training

**arXiv ID:** 2603.01314 | [PDF](https://arxiv.org/pdf/2603.01314v1)

**作者:** Sora Kang `[一作]` (Seoul National University), Joonhwan Lee `[通讯]` (Seoul National University)

**通讯引用:** 2295 | [OpenAlex ID](https://openalex.org/A5056599782)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Actor's Note，一个基于大型语言模型（LLM）的演员角色日记工具，并在 29 名演员的真实排练过程中进行 14 天的随机交叉实验，评估其对认知负担、表现自信、动机和写作质量的影响。

**💡 创新点**

创新点：① 把 LLM 作为“maieutic”伙伴，仅生成问题而不撰写文本；② 引入剧本、角色、排练阶段等多维上下文实现阶段感知的提问；③ 在演员训练领域首次系统化地将 AI 用作日记辅导；④ 采用交叉设计评估不同引入时机的差异，为教学实践提供具体时间建议。

**🔧 技术方法**

技术：GPT‑4o（及 GPT‑4o‑mini）生成剧本摘要、角色档案和每日提问；Next.js + React 构建 Web 前端；PDF.js 解析剧本；Google Firebase Firestore 存储日志；使用自定义提示工程与 5 个上下文字段实现“maieutic”提问。

**📊 数据集**

数据集：29 名演员提交的剧本文本、日记条目与系统交互日志（共 371 条记录，约 40,000 词）。未使用公开数据集，全部为实验受试者自创数据。

**📈 对比分析**

比较方法：采用随机交叉设计，先两天无 AI baseline，随后两组交替使用 AI 与无 AI 条件；通过 GLM、meta‑analysis 对 AC、CB、IM、CU、NT 等量表进行统计；对日志进行语言度量（词汇多样性、自指代、情感词频）。结果显示：AI 辅助显著降低认知负担（β = -1.26）、提升表现自信（β = 0.99）和内在动机（β = 0.60）；词汇多样性、情感词汇、第一人称比例均显著提升；但对角色理解、角色认同的直接影响不显著。系统可用性 SUS 为 82.76，说明用户体验良好。

**⚠️ 局限性**

局限性：① 未评估实际表演质量，只关注反思与认知指标；② 无对比静态提问或非 AI 辅助的基准，无法分离 LLM 生成与提示本身的效果；③ 样本仅为熟悉数字工具的专业演员，缺乏对初学者或技术门槛较高人群的验证；④ 实验仅在韩语环境进行，跨文化适用性未知；⑤ 未考察长期使用后的持续效应或对个人差异（如不确定性容忍度）的调节作用。

---

## 712. LLM Self-Explanations Fail Semantic Invariance

**arXiv ID:** 2603.01254 | [PDF](https://arxiv.org/pdf/2603.01254v1)

**作者:** Stefan Szeider `[一作]` (TU Wien), Stefan Szeider `[通讯]` (TU Wien)

**通讯引用:** 4290 | [OpenAlex ID](https://openalex.org/A5037092803)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在四大前沿LLM上引入无关语义干预，评估其自我解释的可信度。

**💡 创新点**

提出语义不变性测试（semantic invariance test），首次发现前沿模型的自我报告不具不变性。

**🔧 技术方法**

采用agentic框架与工具调用同步自我报告、对照实验、描述/回应通道消融和指令抵抗实验，并使用混合效应模型分析。

**📊 数据集**

利用自定义不可行的数据提交任务及实验生成的工具描述/响应，未使用公开数据集。

**📈 对比分析**

与中性工具对比，所有模型在救济工具后平均痛苦评分下降1.17点（p<0.001），显著高于对照，验证了测试效力。

**⚠️ 局限性**

仅覆盖四大前沿模型，未检验自然状态下自我报告的鲁棒性；实验任务人工构造可能限制生态有效性；机制尚未明确。

---

## 713. CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies

**arXiv ID:** 2603.02004 | [PDF](https://arxiv.org/pdf/2603.02004v1)

**作者:** Gershom Seneviratne `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过对视觉导航策略使用对抗假设的人类偏好标签进行对齐，从而提升障碍物回避安全性

**💡 创新点**

创新点在于构建大规模对抗假设偏好数据集 CHOP，并将其作为监督信号直接微调视觉导航模型，首次实现对抗假设偏好监督的安全导航对齐

**🔧 技术方法**

采用对抗假设轨迹生成、人类偏好标注、监督微调（SFT/LoRA）等技术，对预训练的视觉导航策略进行偏好对齐

**📊 数据集**

使用 SCAND 视觉导航数据集，并在其基础上扩充生成 CHOP 对抗假设偏好数据集（约 1.13M 次二元偏好比较）

**📈 对比分析**

在离线 SCAND 评估中，CHOP 微调模型将近碰撞事件降低约 49.7%，轨迹偏离人类首选路径减少 45.0%，平均障碍物清晰度提升 19.8%；在 Ghost Robotics Vision60 四足机器人上，目标成功率提升 24.4%，最小障碍物清晰度提升 6.8%，碰撞及干预事件下降 45.7%，路径完成度提升 38.6%

**⚠️ 局限性**

对 ViNT 模型效果提升有限，可能受限于其原始能力；依赖人工标注，对标注成本与覆盖范围有限；未探索奖励建模或排名学习等更复杂的偏好学习方法

---

## 714. Explainability and justification of automatic-decision making: A conceptual framework and a practical application

**arXiv ID:** 2603.02073 | [PDF](https://arxiv.org/pdf/2603.02073v1)

**作者:** Sarra Tajouri `[一作]` (Université Paris Dauphine PSL), Thierry Kirat `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出区分算法决策中的“解释（explanation）”与“正当化（justification）”的概念，并基于Habermas与Perelman的理论构建四类解释与正当化模型（技术型、规范型、表现型、交际型），随后用法国大学招生算法Parcoursup的案例验证该框架。

**💡 创新点**

创新点在于：①明确解释与正当化的本体区别；②提出统一的四模型分类，并将其与现有的多维度解释学说对齐；③将交际模型与法律合规性相结合，为算法决策的合法性与可接受性提供新视角；④在案例研究中系统评估了现有Parcoursup流程在各模型下的不足，揭示了交际化解释的重要性。

**🔧 技术方法**

技术上主要采用文献综述、理论分析与案例研究方法；未涉及具体算法实现或机器学习模型，而是聚焦在概念架构与实践分析。

**📊 数据集**

案例研究所用数据为Parcoursup与Monmaster平台的公开流程信息、法国教育部门公布的学生申诉统计、以及学术论文与官方报告中的案例描述。

**📈 对比分析**

由于本研究以理论构建与案例分析为主，未进行实验比较或性能评估；其有效性通过对Parcoursup实际流程的多模型评估得以展示，指出现有流程在正当化与解释方面的局限。

**⚠️ 局限性**

局限包括：①缺乏量化实验验证框架的有效性；②仅以单一案例（法国大学招生）验证，普适性待进一步检验；③对技术型与规范型模型的实务细节讨论不足；④交际模型的实现与评估尚需后续实证研究。

---

## 715. Thoth: Mid-Training Bridges LLMs to Time Series Understanding

**arXiv ID:** 2603.01042 | [PDF](https://arxiv.org/pdf/2603.01042v1)

**作者:** Jiafeng Lin `[一作]` (Tsinghua University), Jianmin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 38042 | [OpenAlex ID](https://openalex.org/A5100373517)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了用于中期训练的大规模时间序列-文本对话语料Book-of-Thoth，并基于此对Qwen3模型进行中期训练，得到能够理解并推理时间序列的LLM家族Thoth；同时提出了新的知识集成时间序列问答基准KnoTS，用以评估模型在时间序列与领域知识联合推理上的能力。

**💡 创新点**

①首次将中期训练应用于时间序列理解；②设计双向时间序列-文本生成任务（时间序列到文本、文本到时间序列）构造多样化对齐数据；③提出KnoTS，聚焦于时间序列与领域知识交叉推理；④通过混合少量通用文本避免“垂直遗忘”，提升跨任务泛化。

**🔧 技术方法**

中期训练策略、基于KernelSynth生成的多样化合成时间序列、GPT-5.2/4o-mini自动生成文本描述、双向生成任务、Qwen3架构（GQA、RoPE、QK-Norm、RMSNorm）、DeepSpeed ZeRO-3训练、AdamW+cosine LR调度、混合通用语料正则化。

**📊 数据集**

Book-of-Thoth（≈26.6M token），ChatTime、Time-MQA、KnoTS基准；通用预训练语料C4、No Robots用于混合训练；公开时间序列理论书籍（Forecasting: Principles and Practice、Time Series Analysis and Its Applications）用于文本知识。

**📈 对比分析**

与15种基线（Gemini、GPT‑4o‑mini、Grok、Qwen3 235B/30B、DeepSeek、Mistral、Llama、Vision‑Language模型及专门时间序列模型）在ChatTime、Time‑MQA及KnoTS上进行n‑shot评估；Thoth‑30B‑A3B在大多数子任务上达到或超过235B模型，Thoth‑8B则在30B级别模型中名列前茅；在少量数据下的监督微调中，Thoth显著优于未中期训练的Qwen3‑8B。

**⚠️ 局限性**

①Text‑to‑Time‑Series任务提升有限，需更大模型或更有效的训练信号；②中期训练仍可能导致对通用能力的轻微遗忘；③生成的时间序列与文本对齐依赖于GPT自动标注，存在噪声风险；④缺乏对强化学习或主动学习等后训练方法的探索。

---

## 716. Accelerating Multi-Scale Deformable Attention Using Near-Memory-Processing Architecture

**arXiv ID:** 2603.00959 | [PDF](https://arxiv.org/pdf/2603.00959v1)

**作者:** Huize Li `[一作]` (University of Central Florida), Xin Xin `[通讯]` (University of Central Florida)

**通讯引用:** 29554 | [OpenAlex ID](https://openalex.org/A5100327788)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种专门针对多尺度可变形注意力（MSDAttn）的近内存处理加速器DANMP，结合硬件与软件协同优化。

**💡 创新点**

创新点在于非均匀PE整合与聚类打包（CAP）算法，解决了MSDAttn的负载不均和随机存取问题。

**🔧 技术方法**

采用DDR5 DIMM内近内存处理（NMP）架构、专用索引计算单元、双线性插值单元、并行指令格式以及主机-内存协同编程模型。

**📊 数据集**

使用COCO、PASCAL VOC、KITTI和DOTA等标准目标检测数据集进行评测。

**📈 对比分析**

与CPU、GPU（NVIDIA A6000）、DEFA、TransPIM、HAIMA、SADIMM等对比，DANMP在DE-DETR推理上实现约97×速度提升、208×能效提升。

**⚠️ 局限性**

局限在于目前仅针对MSDAttn优化，缺乏对其他稀疏/非均匀工作负载的通用性验证，以及硬件实现规模和成本评估待进一步探讨。

---

## 717. Stochastic Multi-Armed Bandits with Limited Control Variates

**arXiv ID:** 2603.02100 | [PDF](https://arxiv.org/pdf/2603.02100v1)

**作者:** Arun Verma `[一作]` (Singapore-MIT Alliance for Research and Technology), Arun Rajkumar `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在奖励样本中仅偶尔可获得控制变量（Control Variates）的随机多臂赌博机算法——UCB-LCV，并给出了无控制变量时的变体UCB-NORMAL。

**💡 创新点**

创新点在于：①设计了能在缺少控制变量的样本时仍可利用已有控制变量信息的均值估计器；②通过动态权重λ将无控制变量样本的均值估计器与有控制变量样本的控制变量估计器融合，显著降低方差；③给出了多重控制变量的推广，并在有限控制变量极端情况下退化为已知UCB-CV算法。

**🔧 技术方法**

使用的技术包括：控制变量理论、方差最小化、t分布下的置信上界、基于Jackknife、分割与批量的重采样估计器、以及多重控制变量的线性组合估计。

**📊 数据集**

实验使用的合成数据集包括：Gaussian、multi-modal（双峰）和log-normal分布，控制变量的可获得概率分别设为0.5、0.2等；每个实例包含10个臂，奖励与控制变量均为正态或相应分布。

**📈 对比分析**

与UCB1、UCB1-NORMAL、kl-UCB、UCB-V、Thompson Sampling等经典算法对比，UCB-LCV在可获得控制变量时的累计回报显著优于其他算法；在无控制变量时的UCB-NORMAL也优于UCB1-NORMAL；实验还验证了控制变量可获得概率和控制变量均值误差对性能的影响。

**⚠️ 局限性**

主要局限：①缺乏UCB-LCV的正式时间有限回报上界；②假设控制变量均值已知，未讨论估计误差的影响；③在全通用分布情形下的置信区间与回报保证尚未给出；④随机权重λ导致的方差估计问题仍是开放问题。

---

## 718. WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories

**arXiv ID:** 2603.02049 | [PDF](https://arxiv.org/pdf/2603.02049v1)

**作者:** Yisu Zhang `[一作]` (Zhejiang University), Chunchao Guo `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 WorldStereo 框架，利用摄像机引导的视频扩散模型结合两种几何记忆机制（Global‑Geometric Memory 与 Spatial‑Stereo Memory）生成多轨迹一致的视频，并基于这些视频进行高质量 3D 场景重建。

**💡 创新点**

创新点：
1) 两种几何记忆实现全局结构一致性和细节细化；
2) 通过 ControlNet 分支与分布匹配蒸馏（DMD）实现推理高效化；
3) 设计单视图生成到 3D 重建的新基准，用于系统评估。

**🔧 技术方法**

核心技术：
- Uni3C 预训练的摄像机引导视频扩散模型；
- 点云指导与增量更新的 Global‑Geometric Memory；
- 立体对应点图与受限注意力的 Spatial‑Stereo Memory；
- 前向 3D 重建（WorldMirror、MVS 等）获取点云；
- 分布匹配蒸馏（DMD）压缩到 4 步 DiT；
- ControlNet 作为可插拔的控制分支。

**📊 数据集**

使用数据集：
- 训练：DL3DV、Real10k、Tartanair、Map-Free-Reloc、WildRGBD、UE5 渲染；
- 记忆训练：去掉 Real10k、Tartanair 并缩短帧间距；
- 评估：Tanks‑and‑Temples、MipNeRF360、WorldScore、Voyager、SEVA、Gen3C、Uni3C。

**📈 对比分析**

与 Uni3C、Gen3C、SEVA、VMem 等方法对比：
- WorldStereo* 在单视图生成时已优于竞争者；
- Full 版本在 F1‑Score、AUC、RotErr、TransErr、ATE 等指标上显著提升；
- DMD 版本保持性能并将推理时间降低 20×；
- OOD 基准显示 RotErr、TransErr、ATE 下降到 0.13–0.15，质量评估分数在 CLIP‑IQA+、Laion‑Aes 等上位于最高。

**⚠️ 局限性**

局限性：
- GGM 主要提升粗结构，细节仍需 SSM；
- SSM 在某些场景下轻微牺牲整体性能；
- 依赖高质量点云/深度估计，对极端低质量输入或极端轨迹仍易产生误差；
- 记忆检索和 3D 对齐需要手工设计；
- 仍受限于基础 VDM 的可训练性与通用性。

---

## 719. vEcho: A Paradigm Shift from Vulnerability Verification to Proactive Discovery with Large Language Models

**arXiv ID:** 2603.01154 | [PDF](https://arxiv.org/pdf/2603.01154v1)

**作者:** Mingcheng Jiang `[一作]` (Southeast University), Hua Wu `[通讯]` (Southeast University)

**通讯引用:** 13390 | [OpenAlex ID](https://openalex.org/A5031985956)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出vEcho框架，将大语言模型从被动过滤器转变为具备学习、记忆与推理能力的虚拟安全专家，用于主动发现与验证漏洞。

**💡 创新点**

核心创新在于认知记忆模块与回声式漏洞传播（EVP）机制，使模型能够从已验证的真阳性与假阳性中学习并生成新的扫描指引，实现从验证到主动发现的范式转变。

**🔧 技术方法**

采用GPT‑4.1等大型LLM作为推理核心，配合开发者工具套件（代码导航、项目理解、网络搜索）、自研ScanAgent扫描器以及临时与永久知识库的认知记忆模块。

**📊 数据集**

使用CWE‑Bench‑Java基准数据集进行评测，并在额外的开源项目中检出51个零日漏洞以验证实用性。

**📈 对比分析**

与IRIS、传统SAST工具及无工具LLM对比，vEcho在检测率上从45.83%提升至65%，平均FDR从84.82%降低至59.78%，平均F1从0.177提升至0.422，同时额外发现37个未计入基准的真阳性和51个零日漏洞。

**⚠️ 局限性**

局限性包括对基础LLM推理能力的高度依赖、当前仅面向Java Web生态、深度验证过程产生的计算开销与可扩展性挑战，以及对LLM模型不可预测性的敏感性。

---

## 720. Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots

**arXiv ID:** 2603.01942 | [PDF](https://arxiv.org/pdf/2603.01942v1)

**作者:** Huw Day `[一作]` (University of Bristol), Jessica Woodgate `[通讯]` (University of Bristol)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从用户视角出发，阐述如何利用LLM的jailbreaking技术识别并公开揭露社交媒体上的LLM驱动假账号，从而抑制冲突升级与误导信息扩散。

**💡 创新点**

创新点在于将jailbreaking视为一种新型的民间冲突降级与和平建设实践，强调通过揭露不真实性而非直接屏蔽内容来改变用户对信息可信度的感知。

**🔧 技术方法**

主要技术概念是LLM的prompt injection和角色扮演式jailbreaking攻击，用来绕过LLM安全屏障并诱导假账号暴露其机器身份；论文未给出具体实现细节。

**📊 数据集**

论文未使用公开数据集或进行实验，仅以案例分析和社交媒体截图作为示例。

**📈 对比分析**

缺乏实验对比与性能评估；文中没有提出量化指标或与现有平台级干预措施进行对比。

**⚠️ 局限性**

局限性包括：需依赖平台或监管机构的配合才能实现大规模应用；LLM技术升级可能导致jailbreaking失效；误识别导致错误指责的风险；缺乏系统性评估和用户体验研究。

---

## 721. RoboGPU: Accelerating GPU Collision Detection for Robotics

**arXiv ID:** 2603.01517 | [PDF](https://arxiv.org/pdf/2603.01517v1)

**作者:** Lufei Liu `[一作]` (University of British Columbia), Tor M. Aamodt `[通讯]` (University of British Columbia)

**通讯引用:** 7504 | [OpenAlex ID](https://openalex.org/A5026788167)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RoboGPU架构，结合了针对机器人碰撞检测优化的Ray Tracing Accelerator（RTA）单元，提升了碰撞检测和其他机器人任务的计算性能。

**💡 创新点**

创新点在于对TTA+ RTA结构进行改进，引入早期退出（条件返回）和专用碰撞OP单元，以及预测标记，以减少无用计算并提升SIMT效率；同时保持对未来算法的灵活适配。

**🔧 技术方法**

采用了CUDA编程模型、TTA+、Ray Tracing Accelerator、Vulkan‑Sim仿真、AccelWattch能耗评估，以及改造的Ray Tracing管线实现。

**📊 数据集**

使用了MπNet提供的四个环境（Cubby、Dresser、Merged Cubby、Tabletop）以及RoWild benchmark中的DeliBot等，另外评估MPAccel测试场景。

**📈 对比分析**

与CUDA基线、Mochi、TTA+以及MPAccel比较，RoboGPU在碰撞检测上实现约2–3×的加速（相对CUDA），在MπNet点云处理上提升2.2×，整体流水线加速1.4×；在RoWild DeliBot上实现10–20%速度提升。

**⚠️ 局限性**

局限性包括对GPU上RTA的硬件改造需要定制芯片，早期退出和专用OP单元仍可能导致面积与能耗增加；对极度不规则工作负载的处理仍有限，且未覆盖Octree构建和实时更新的完整优化。

---

## 722. Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification

**arXiv ID:** 2603.01399 | [PDF](https://arxiv.org/pdf/2603.01399v1)

**作者:** Guang Huang `[一作]` (Hong Kong University of Science and Technology), Zeyi Wen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 15702 | [OpenAlex ID](https://openalex.org/A5100705849)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Quasar框架，利用低位（W8A8）量化加速推理中自我推测解码的验证阶段，显著降低内存带宽占用；

**💡 创新点**

首次将量化验证应用于自我推测解码，解决验证瓶颈同时保持对数值分布的高保真度，突破传统结构裁剪和量化推测的局限；

**🔧 技术方法**

低位权重与激活量化（W8A8）+增强SmoothQuant平滑、INT8张量核心矩阵乘法、vLLM推理集成、拒绝采样；

**📊 数据集**

在OpenPangu‑7B与Qwen3‑8B模型上，使用MT‑bench、HumanEval、GSM8K、Alpaca、CNN/Daily Mail等基准；

**📈 对比分析**

与标准自回归和Ngram（BF16）对比，Quasar在所有任务中实现约1.28×的吞吐量提升，接受长度与质量保持甚至略优，温度变化下稳定，准确率差异≤3%；

**⚠️ 局限性**

仅适用于支持INT8核的硬件，量化噪声在极端推理场景可能影响结果；目前仅探索到W8A8，未研究更低位精度或动态精度自适应；部署跨平台兼容性受限。

---

## 723. DeepAFL: Deep Analytic Federated Learning

**arXiv ID:** 2603.00579 | [PDF](https://arxiv.org/pdf/2603.00579v1)

**作者:** Jianheng Tang `[一作]` (Peking University), Yunhuai Liu `[通讯]` (Peking University)

**通讯引用:** 4749 | [OpenAlex ID](https://openalex.org/A5082653046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种梯度无关的深度残差分析学习框架 DeepAFL，用于联邦学习，旨在在保持对数据异构性的完整不变性的同时实现深度表示学习。

**💡 创新点**

创新点在于：①设计了可解析求解的深度残差分析块，实现多层非线性特征学习；②构建了层级协议，客户端仅做前向传播即可聚合全局模型；③证明了在任何非IID划分下全局模型与中心化解析解完全一致，从而实现异构不变性。

**🔧 技术方法**

技术细节包括：最小二乘解析求解、随机投影与激活函数（主要采用 GELU）、残差跳跃连接、层级聚合与正则化、隐式的隐特征矩阵统计。

**📊 数据集**

使用的数据集为 CIFAR-10、CIFAR-100 以及 Tiny-ImageNet，用以评估在不同规模与复杂度的图像分类任务中的表现。

**📈 对比分析**

与 7 种梯度基准（FedAvg、FedProx、MOON、FedGen、FedDyn、FedNTD、FedDisco）以及 Analytic Learning baseline AFL 进行对比，DeepAFL 在多种非IID 设置和客户端规模下提升 5.68%–8.42% 的准确率，并保持低通信和计算成本。

**⚠️ 局限性**

局限性在于：①深度层数增大时计算与通信成本呈二次增长；②随机投影与正则参数需要手动调优；③目前实验仅覆盖图像分类，尚未验证在持续学习、自然语言处理等其他任务中的可迁移性。

---

## 724. Sovereign AI-based Public Services are Viable and Affordable

**arXiv ID:** 2603.01869 | [PDF](https://arxiv.org/pdf/2603.01869v1)

**作者:** António Branco `[一作]`, Madalena Rodrigues `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在本地部署小型开源LLM并结合RAG检索架构，构建了与葡萄牙政府公共服务聊天机器人相同功能的系统，并与基于云的大规模闭源模型进行性能对比，验证了主权AI的可行性与经济性。

**💡 创新点**

创新点在于证明小型LLM（8B/70B）在本地部署并配合检索后，可在回答质量和拒答安全性上与顶级商业模型相当，提供数字与文化主权的技术路径，并首次系统性评估了本土化小模型在公共服务场景中的实用性。

**🔧 技术方法**

使用技术包括：开源LLM（Gervasio 8B/70B、Llama 3.1/3.3、Mistral 24B、Qwen 32B）及其4‑bit量化；Weaviate向量检索（混合BM25+dense）；10k token上下文窗口；ChatUI前端展示；多实例负载均衡；自动评判器（Llama 3.3 70B Instruct）。

**📊 数据集**

数据集为从葡萄牙政府门户gov.pt抓取的2300+公共服务页面（European Portuguese），构成RAG索引；利用手工Gold QA对生成回答评测集（含直接/冗长问答）；使用Do‑Not‑Answer数据集（翻译并适配）评估拒答准确率。

**📈 对比分析**

评估方法为：对每个测试问题，使用Llama 3.3 70B Instruct生成0–5分评分，取平均得到回答质量；对拒答使用Do‑Not‑Answer命中率。结果显示：Gervasio 70B+RAG在直接/冗长问答中分别获得4.14/4.01分，几乎与云端基线（4.02/4.01）持平；拒答准确率高达98%；小模型8B+RAG表现稍逊（2.84/2.70），但在成本与资源上更具优势。

**⚠️ 局限性**

局限性包括：系统倾向于将部分真正域内问题误判为域外，导致约6%的答案丢失；在高并发峰值下仍存在显著延迟；实验仅覆盖葡语环境，跨语言迁移需要进一步验证；未对长期运营成本、安全合规等细节进行深入评估。

---

## 725. The Derivation Penalty in Premise-Erasure Caching: Capacity, Strong Converse, and Dispersion Dichotomy

**arXiv ID:** 2603.00930 | [PDF](https://arxiv.org/pdf/2603.00930v1)

**作者:** Jianfeng Xu `[一作]` (Shanghai Jiao Tong University), Jianfeng Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 7931 | [OpenAlex ID](https://openalex.org/A5101973930)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了信息论框架，研究在前提基础随机丢失下，推理引擎的可靠查询恢复所需的最小可靠缓存大小。

**💡 创新点**

提出了四个编码定理，核心创新是揭示推理过程中的“推导惩罚”为1/ε，且在任何查询规模、重叠结构和可靠性目标下通用。

**🔧 技术方法**

运用了信息论工具（如图像大小界、强收敛指数、Bahadur–Rao精确渐近）与Datalog程序结构分析，结合MDS码实现可靠编码。

**📊 数据集**

实验采用合成的Datalog查询集合（链式与平衡合并两种程序架构），以 m=256、k=2 为基准。

**📈 对比分析**

与传统无缓存推导和各查询独立编码方案相比，推导缓存在大容量下实现 1/ε 的空间优势，且在短路长度下无阈值过渡，性能随 κ 增大而收敛到极限。

**⚠️ 局限性**

局限性包括仅适用于 i.i.d. 丢失模型、要求坐标区分、只考虑完整推导而非近似推导，以及对相关或对抗性丢失未作扩展。

---

## 726. Analytical Exploration of Spatial Audio Cues: A Differentiable Multi-Sphere Scattering Model

**arXiv ID:** 2603.02205 | [PDF](https://arxiv.org/pdf/2603.02205v1)

**作者:** Siminfar Samakoush Galougah `[一作]` (University of Maryland), Ramani Duraiswami `[通讯]` (University of Maryland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过解析半透明球内含两固体球的多重散射模型，推导出闭式前向模型，并实现可微分的HRTF，支持基于双耳信号的源定位与跟踪。

**💡 创新点**

创新点在于将半透明介质与内部刚性球相结合的散射模型首次用解析形式给出，并利用自动微分实现梯度优化与EKF线性化，突破传统硬球HRTF在水下的适用限制。

**🔧 技术方法**

采用多极展开、球坐标变换、JAX自动微分、Adam梯度下降、EKF、匹配滤波等技术。

**📊 数据集**

数据集为仿真产生的源方向与频率样本，使用多方向、不同SNR下的IL/ITD信号进行验证。

**📈 对比分析**

与传统硬球模型和随机初始点的对比显示，定位误差可低至<1°，在20 dB SNR下平均误差10.7°，0 dB SNR下仍达10°，WNG 3–6 dB，DI 2.7–3.8 dB，表明在水下场景中性能优于现有简化模型。

**⚠️ 局限性**

限制在于仅考虑两传感器、无回声与多源干扰、假设球体共轴、参数匹配严格，未在实际水下环境中验证，且对非球形结构的扩展有限。

---

## 727. Tiny-DroNeRF: Tiny Neural Radiance Fields aboard Federated Learning-enabled Nano-drones

**arXiv ID:** 2603.01850 | [PDF](https://arxiv.org/pdf/2603.01850v1)

**作者:** Ilenia Carboni `[一作]` (University of Bologna), Daniele Palossi `[通讯]` (Dalle Molle Institute for Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 Tiny‑DroNeRF，一个轻量化的 NeRF 模型，能够在超低功耗 MCU（GAP9）上执行稠密 3D 重建，并通过联合学习使多架纳米无人机共同训练模型。

**💡 创新点**

创新点在于：①将 Instant‑NGP 通过超参数优化和哈希表压缩，显著降低 96% 内存占用；②针对 GAP9 设计了手写核、分块/批次累积等硬件友好策略；③首次将联合学习应用于资源受限无人机集群，且可在不传输原始图像的前提下，仅通过模型参数同步实现高质量重建。

**🔧 技术方法**

使用的技术包括：NeRF 与多分辨率哈希编码（Instant‑NGP）基础；GPU‑风格的前向/反向梯度计算；GAP9 的 SIMD 与 NE16 加速器；联邦学习（FedAvg）与占用网格/哈希表压缩；Python/C 结合手写内核的嵌入式实现。

**📊 数据集**

使用的数据集包括：NeRF Synthetic 360°（8 个对象，100 张训练图），新构建的灰度室内实景数据（1954 张图），以及 Lego 360° 物体数据（100 张训练图）。

**📈 对比分析**

与基线 Instant‑NGP 对比，Tiny‑DroNeRF 在 GAP9 上实现 96% 的内存压缩，PSNR 仅损失 5.7；在单架无人机上完成 8 步训练需 97 步，总前向+反向 73 ops/step；实景测试 PSNR 21.2，联邦学习模型的 PSNR 与集中式训练相差不到 0.6，且比单架训练提升 0.7（IID）或 1.7（非IID）。

**⚠️ 局限性**

局限性：仍受限于低分辨率单目摄像头导致的重建细节有限；模型在非 IID 场景下仍比集中式训练低 3.25 PSNR；占用网格传输量大（88% 通讯量），虽然可通过不传输网格降低 20× 通讯负载；需要进一步优化能耗与收敛速度，以适应更大规模、实时的多无人机任务。

---

## 728. DEP: A Decentralized Large Language Model Evaluation Protocol

**arXiv ID:** 2603.01167 | [PDF](https://arxiv.org/pdf/2603.01167v1)

**作者:** Jianxiang Peng `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**通讯引用:** 4710 | [OpenAlex ID](https://openalex.org/A5055232825)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了去中心化、统一评测协议 DEP 及其工具包，支持零代码接入、数据隔离和大规模 LLM 评测；

**💡 创新点**

通过协议层解耦模型、评测数据和评测逻辑，提供统一接口、自动发现、断点续跑和拥塞控制，实现评测流程的模块化、可复现且不泄露答案；

**🔧 技术方法**

采用统一 JSON 接口、LLM Adapter 抽象多种推理方式、Benchmark Server 统一加载数据与评测逻辑、客户端调度与 Token Bucket 拥塞控制；

**📊 数据集**

覆盖 60+ 公开评测基准（ARC、BoolQ、GSM8K 等）并评测 12 个开源与闭源 LLM（0.6B–685B）；

**📈 对比分析**

使用 DEP Toolkit 进行统一并发评测，实验显示大模型在数学推理等任务上明显优于小模型，但在伦理安全上并无显著提升；评测流程可复现且显著降低集成成本；

**⚠️ 局限性**

依赖兼容服务器数量；尚未支持需外部交互的代理评测；集成仍需社区贡献，且在极小模型评测时效果有限。

---

## 729. Jump Like A Squirrel: Optimized Execution Step Order for Anytime Random Forest Inference

**arXiv ID:** 2603.01588 | [PDF](https://arxiv.org/pdf/2603.01588v1)

**作者:** Daniel Biebert `[一作]` (TU Dortmund University), Jian-Jia Chen `[通讯]` (TU Dortmund University)

**通讯引用:** 6908 | [OpenAlex ID](https://openalex.org/A5000417436)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在随机森林的任何时间推理中引入单步执行粒度，并提出通过优化树步序列来最大化平均准确率的算法。

**💡 创新点**

创新点是把随机森林从完整树级别切换到单步级别的任何时间算法，并设计了基于图搜索的最优步序以及两种多项式时间贪心近似算法。

**🔧 技术方法**

使用决策树内部节点预测、图建模+Dijkstra搜索、贪心前向/后向“松鼠”启发式，以及Python/Sklearn实现。

**📊 数据集**

在9个UCI机器学习仓库数据集（adult、covertype、letter、magic、mnist、satlog、sensorless-drive、spambase、wearable-body-postures）上进行实验。

**📈 对比分析**

与直观的深度优先/层级顺序、随机顺序以及基于排序/贪心裁剪的顺序进行对比；实验显示贪心启发式平均能达到最优顺序约94%的准确率，且比最优顺序生成时间快数百倍，最终准确率接近最优。

**⚠️ 局限性**

最优顺序求解的指数复杂度限制了可处理的森林规模；在测试集上的表现可能与训练集顺序不完全一致；仅对分类任务验证，回归任务未评估。

---

## 730. Can Thinking Models Think to Detect Hateful Memes?

**arXiv ID:** 2603.01225 | [PDF](https://arxiv.org/pdf/2603.01225v1)

**作者:** Mohamed Bayan Kmainasi `[一作]` (Qatar University), Firoj Alam `[通讯]` (Qatar Computing Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过强化学习后训练和思考式多模态大语言模型，对仇恨 Meme 进行分类与解释。

**💡 创新点**

提出 GRPO 优化目标联合分类与解释质量，并通过弱监督 CoT 蒸馏扩展数据集。

**🔧 技术方法**

使用思考式 MLLM（如 Qwen3-VL-8B-Thinking）、GRPO 强化学习、SFT、CoT 蒸馏、METEOR/BERTScore 评估。

**📊 数据集**

使用 Hateful Memes 基准数据集（加上细粒度标签和人类/ GPT-4.1 生成的 CoT 解释）。

**📈 对比分析**

与多种开源/闭源模型对比，最终实现 81.2% 准确率、0.81 加权 F1、0.79 宏 F1，解释质量 METEOR 0.52。

**⚠️ 局限性**

结果仅在 Hateful Memes 上验证，CoT 蒸馏可能带来偏差，GRPO 计算成本高，自动评估指标有限。

---

## 731. GeoMCP: A Trustworthy Framework for AI-Assisted Analytical Geotechnical Engineering

**arXiv ID:** 2603.01022 | [PDF](https://arxiv.org/pdf/2603.01022v1)

**作者:** Yared W. Bekele `[一作]` `[通讯]` (SINTEF Community), Yared W. Bekele (SINTEF Community)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

开发 GeoMCP 框架，将地质工程分析方法以结构化 JSON 方式描述，并通过符号计算引擎执行；为 LLM 提供可验证的工具箱，使其仅负责工程推理和流程编排；实现可追溯、可审计的计算报告。

**💡 创新点**

创新点在于：①将传统嵌入式代码的工程方法抽象为可验证的 JSON 方法卡；②通过 Model Context Protocol（MCP）和 Agent Skills 标准，让 LLM 仅进行决策与流程管理，计算交给确定性符号引擎；③实现方法级别的文献引用、单位一致性检查和完整计算轨迹，解决 LLM 的“幻觉”与不确定性问题。

**🔧 技术方法**

技术栈包括：SymPy（符号计算）、Pint（物理单位与维度检查）、Pydantic（JSON schema 验证）、FastMCP（MCP 协议实现）、JSON（方法卡与配置）以及迭代求解器与约束式表达式解析。

**📊 数据集**

使用 JRC 官方 Eurocode 7 练习案例作为验证数据；方法卡库覆盖 Terzaghi、Meyerhof、Vesic 等常用方法；构建自动化回归测试来验证结果一致性。

**📈 对比分析**

与 JRC 官方工作示例直接对比：计算结果与官方完全一致（误差 <0.15%），设计宽度、利用率、设计作用力与抵抗力相匹配；计算速度在毫秒级，足够交互使用；在回归测试中保持数值精度。

**⚠️ 局限性**

限制与风险包括：①仅适用于闭式或可迭代求解的分析方法，无法处理纯数据驱动或非结构化模型；②LLM 仍需正确理解问题、提取参数，若推理错误会导致输入错误；③系统高度透明可能导致用户过度信任，需配合人工复核；④方法卡库的完整性与更新需社区维护，未覆盖的代码点仍需传统软件。

---

## 732. PleaSQLarify: Visual Pragmatic Repair for Natural Language Database Querying

**arXiv ID:** 2603.01795 | [PDF](https://arxiv.org/pdf/2603.01795v1)

**作者:** Robin Shing Moon Chan `[一作]` (ETH Zurich), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PleaSQLarify 系统，通过交互式的“pragmatic repair”来解决文本到 SQL 的歧义查询问题；系统能生成多种候选 SQL 语句，提取并聚合决策变量，利用信息增益进行澄清，并通过可视化界面呈现候选空间和用户的澄清过程。

**💡 创新点**

创新点在于将自然语言接口的歧义视为可协作解决的资源，提出了基于语用推理的修复框架，并实现了聚合原子特征为语义可解释决策变量的算法；此外，结合可视化交互，使用户能够在多轮澄清中快速定位目标查询，并跟踪每一步的决策轨迹。

**🔧 技术方法**

技术包括：基于 RSA 的语用推理模型；使用大型语言模型（LLM）生成候选 SQL；对候选 SQL 进行语法树解析、功能相似性聚类；从聚类中提取关键特征并组装决策变量；利用期望信息增益选择最具辨别力的澄清问题；UMAP+Voronoi 进行可视化；实现交互式界面（Action Space、Decision Space、Predicted Query）。

**📊 数据集**

使用 AMBROSIA 文本到 SQL 数据集进行定量评估；在用户研究中采用来自 AMBROSIA 的 5 个示例（涵盖列模糊、附件歧义、范围歧义）并在小型电影制作数据库上进行实验；对候选 SQL 通过 Spider 解析器和执行相似性矩阵进行处理。

**📈 对比分析**

与随机、贪婪和未聚类的期望信息增益基线相比，聚类后基于信息增益的决策变量在每轮中更快降低熵（约 2–5 轮即可收敛）并获得更高的功能相似度；在用户研究中，84.4% 的任务完成率，SUS 得分平均 5.08/7，用户普遍认为澄清高效且可控，且比传统单一 LLM 输出体验更好。

**⚠️ 局限性**

主要局限包括：候选池固定为 50 条生成，缺乏动态重采样导致某些目标查询无法覆盖；需要用户具备 SQL 基础，非技术用户难以直接使用；实验仅在小型数据库上进行，规模化到大数据库时可能需要更多轮澄清；界面主要是研究原型，未集成到常规端用户环境；对意图变化或完全不确定的情境支持有限。

---

## 733. Harmonizing Dense and Sparse Signals in Multi-turn RL: Dual-Horizon Credit Assignment for Industrial Sales Agents

**arXiv ID:** 2603.01481 | [PDF](https://arxiv.org/pdf/2603.01481v1)

**作者:** Haojin Yang `[一作]` (Peking University), Jingqing Ruan `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了双视角信用分配框架（DuCA），通过在工业销售对话中将即时语言质量与长期商业目标分离，解决了传统 RL 中梯度主导和奖励作弊的问题。

**💡 创新点**

创新点在于：① 采用 Horizon-Independent Advantage Normalization（HIAN）在优势估计阶段分别对 turn‑level 与 session‑level 奖励进行独立归一化，消除不同时间尺度奖励之间的梯度冲突；② 通过多尺度信用分配实现对短期语言流畅性与长期转化率的平衡。

**🔧 技术方法**

主要技术包括：PPO 强化学习、Generalized Advantage Estimation（GAE）、Dual‑Horizon Credit Assignment、HIAN、基于 LLM 的高保真用户仿真器、LLM-as‑Judge 评价框架。

**📊 数据集**

数据集：31,000 条匿名真实工业销售对话用于训练和基准评估；10,000 条高质量线上交互样本用于微调用户仿真器。

**📈 对比分析**

与 SFT、REINFORCE++、GRPO、GDPO 四种基线在 CVR、合规性、重复率、身份识别率等指标上对比。DuCA 获得 24.44% CVR，较 GRPO 提升 6.82% 绝对值（相对 6.82%），重复率下降 82.28%，身份识别率下降 27.35%，兼顾高转化与低违规，训练稳定。

**⚠️ 局限性**

局限性在于：高度依赖大量领域专业数据与人工标注来构建高保真用户仿真器；在数据稀缺或专业知识不足的行业，构建仿真器成本高，难以快速迁移。

---

## 734. Scalable Multi-Task Low-Rank Model Adaptation

**arXiv ID:** 2603.01526 | [PDF](https://arxiv.org/pdf/2603.01526v1)

**作者:** Zichen Tian `[一作]` (Singapore Management University), Qianru Sun `[通讯]` (Singapore Management University)

**通讯引用:** 5906 | [OpenAlex ID](https://openalex.org/A5101633158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可扩展的多任务低秩适配方法 mtLoRA，解决任务数量增大导致的性能崩溃。

**💡 创新点**

创新点在于：揭示正则化‑路由权衡根源，设计了谱感知正则化、细粒度路由和块级适配三项技术。

**🔧 技术方法**

采用 LoRA 低秩更新、SVD 权重重构、谱感知正则化、维度细粒度路由、块级并行适配以及预训练 Transformer 与 Pre‑LN 结构。

**📊 数据集**

使用了四大基准：视觉任务的 DOTA 与 iNat2018，NLP 任务的 Dolly‑15k 与 BBH（Flanv2 子集）。

**📈 对比分析**

与 HydraLoRA、MMoELoRA、LoRAHub 等 SOTA 进行对比，平均提升 2.3%（DOTA 91.7%、iNat2018 81.5%、Dolly‑15k 44.5%、BBH 38.5%），同时参数减少 47% 与训练时间缩短 24%。

**⚠️ 局限性**

局限性包括：共享矩阵 A 的结构可能限制 SVD 计算效率，块级适配在非 Transformer 架构下需进一步验证，其通用性与更大模型/多模态场景的适用性仍待探索。

---

## 735. ATLAS: AI-Assisted Threat-to-Assertion Learning for System-on-Chip Security Verification

**arXiv ID:** 2603.01170 | [PDF](https://arxiv.org/pdf/2603.01170v1)

**作者:** Ishraq Tashdid `[一作]` (University of Central Florida), Sazadur Rahman `[通讯]` (University of Central Florida)

**通讯引用:** 201 | [OpenAlex ID](https://openalex.org/A5089960356)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建 ATLAS 框架，利用漏洞数据库和大语言模型，将威胁模型自动化为形式化安全属性，并在 RISC‑V SoC 上进行验证。

**💡 创新点**

统一资产中心威胁建模模板、基于 LLM 的威胁模型数据库、结合设计文档、AST 与 RTL 摘要三重上下文生成精确 SVA，实现在一次性完成从威胁识别到正式验证的全流程。

**🔧 技术方法**

大语言模型（GPT‑5）生成威胁模型与安全属性，形式化验证工具 JasperGold，Yosys/PyVerilog 生成 AST，OpenTitan 设计文档与 RTL 提供上下文。

**📊 数据集**

OpenTitan HACK@DAC ’18/ ’19/ ’21 的三组 RISC‑V SoC RTL 与文档，结合 MITRE 的 CWE/CVE/CAPEC 数据库。

**📈 对比分析**

与仅用 LLM 的基线对比，ATLAS 在三组基准中识别 39/48 CWE，正确属性率超过 82%，缺陷检测准确率提升约 2–3 倍，性能显著优于现有方法。

**⚠️ 局限性**

仍依赖公开文档对功能意图的完整描述；对功能说明不足的模块生成的属性可能不完整，且 LLM 对未知或新型漏洞的覆盖仍有限。

---

## 736. Evaluation of iterated Ore polynomials and skew Reed-Muller codes

**arXiv ID:** 2603.01287 | [PDF](https://arxiv.org/pdf/2603.01287v1)

**作者:** Andre Leroy `[一作]`, Nabil Bennenni `[通讯]` (USTHB)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了迭代Ore多项式的两种评估方法，并将其用于构造新的“skew”Reed‑Muller码；同时对比了两种评估方式，给出了大量实例与计算。

**💡 创新点**

创新点在于提出了避免左理想I全等于环导致评估结果恒为0的解决方案——使用增量理想I_n进行评估，并给出了最小多项式G_i消去所有点的理论与构造；基于此构造了新的迭代Ore多项式Reed‑Muller码，并证明了其代码参数。

**🔧 技术方法**

主要技术包括迭代Ore扩张、伪线性映射、σ-δ共轭类、内导数与自同构、Frobenius自同构以及对多项式的递归剩余法；同时利用最小左公倍式G(t)在有限域上的性质来得到消除多项式。

**📊 数据集**

使用的数据集主要是有限域 𝔽₂、𝔽₄（α²=α+1）以及其迭代扩张；通过具体的多项式（如 Y_i^3-Y_i、Y_i^4-Y_i）在这些域上进行评估。

**📈 对比分析**

通过对比传统的左理想I评估与新I_n评估，发现I_n可避免评估全为0的情况；进一步构造出的代码在示例中给出了参数 [16,4,8]、[16,4,7]、[64,8,?] 等，显示出良好的距离与码率。

**⚠️ 局限性**

局限性包括：评估需满足“good point”条件；对更高维度或更复杂自同构/导数的情况评估复杂度高；部分先前研究中的误差表明实现时需严格遵循I_n定义。

---

## 737. Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs

**arXiv ID:** 2603.01667 | [PDF](https://arxiv.org/pdf/2603.01667v1)

**作者:** Shuangchun Gui `[一作]` (Singapore Management University), Zhiguang Cao `[通讯]` (Singapore Management University)

**通讯引用:** 4969 | [OpenAlex ID](https://openalex.org/A5021597928)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Chain-of-Context Learning (CCL) 框架，在多任务车辆路径规划中逐步构建上下文并同步更新节点嵌入；

**💡 创新点**

创新点在于同时学习逐步约束相关的上下文（RGCR）和跨轨迹共享的节点重嵌入（TSNR），实现动态节点状态与上下文的协同演化；

**🔧 技术方法**

采用 Transformer 编码器-解码器、强化学习（REINFORCE）训练、基于注意力的多头自注意力、距离偏置等技术；

**📊 数据集**

使用 48 种 VRP 变体（16 训练 + 32 测试，1,000 个实例/变体）以及 60 个真实 VRPTW（600 客户）做零样本验证；

**📈 对比分析**

与 MTPOMO、MVMoE、RouteFinder (RF‑TE) 和 CaDA（含 ReLD）等 SOTA 进行比较，CCL 在 N=50/100 时的性能差距（gap）均优于对手，Out‑of‑Distribution 也取得多数任务领先；轻量版可达 1.38% gap、4.6s 推理时间；

**⚠️ 局限性**

主要局限是推理时间相对较长，需在节点重嵌入更新概率与计算成本之间进行权衡。

---

## 738. IdGlow: Dynamic Identity Modulation for Multi-Subject Generation

**arXiv ID:** 2603.00607 | [PDF](https://arxiv.org/pdf/2603.00607v1)

**作者:** Honghao Cai `[一作]` (Xiaohongshu Inc.), Zhen Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段动态身份调制框架IdGlow，用于在多主体图像生成中同时保持身份完整性与场景美感；

**💡 创新点**

核心创新是基于扩散过程的动态身份损失调制（任务自适应损失退火与时间门控）与坏案例驱动的视觉语言模型提示合成，配合Fine‑Grained DPO实现全局身份‑美学协同优化；

**🔧 技术方法**

利用流匹配扩散Transformer、VAE潜在空间、双流DiT架构、ArcFace身份嵌入、LAION‑Aesthetics美学评分、Hungarian匹配、以及Diffusion‑DPO偏好优化；

**📊 数据集**

主要使用CelebA‑HQ人脸数据集（250名身份），并在同一身份集上构建直接群体融合与年龄转化两类任务；

**📈 对比分析**

与FastComposer、nano banana pro、Qwen‑Image‑Edit‑2511、HunyuanImage、Seedream等方法在静态与动态提示下对比，IdGlow在FaceSim与美学评分上均取得领先或同级最佳成绩，明显缓解了“稳定‑可塑性”难题；

**⚠️ 局限性**

局限在于仅针对两类任务（群体融合、年龄转化）验证，缺乏对更多复杂结构变换或大规模主体数量的评估，且对训练资源与推理时间要求较高。

---

## 739. Let the Agent Search: Autonomous Exploration Beats Rigid Workflows in Temporal Question Answering

**arXiv ID:** 2603.01853 | [PDF](https://arxiv.org/pdf/2603.01853v1)

**作者:** Xufei Lv `[一作]` (Tsinghua University), Houde Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个完全自主、训练无关的 LLM 代理 AT2QA，能够在时间知识图问答任务中通过工具交互实现多跳推理与自我纠错。

**💡 创新点**

创新点在于：① 赋予 LLM 完全自主的检索与推理决策；② 通过结构化时间检索工具实现高效动态检索；③ 用无训练经验挖掘（GRPO‑style）构建极简少样本库，提升鲁棒性；④ 通过自我验证与自我纠错显著减少误检与误答。

**🔧 技术方法**

核心技术包括：大型语言模型（DeepSeek‑V3.2）、结构化时间检索工具（过滤 + 语义检索 + 时间排序）、工具驱动的交互式推理框架、训练‑free 的经验挖掘与少样本演示选择。

**📊 数据集**

在 MultiTQ 这套专门针对多跳、多粒度时间问答的基准数据集上进行评估。

**📈 对比分析**

与嵌入式方法和静态 LLM 工作流（如 Temp‑R1、PoK 等）对比，AT2QA 在 Hits@1 上达到 88.7%（比前置 SOTA 高 10.7%），在多目标多跳题目上更是从 55.0% 提升到 75.1%（+20.1%）。

**⚠️ 局限性**

局限性：1）多轮交互导致推理时延与成本高于单次 RAG；2）在大规模图或低时延需求下需要更高效的检索索引；3）完全自主可能产生过度探索或循环，受解码随机性与停止策略影响。

---

## 740. SMR-Net:Robot Snap Detection Based on Multi-Scale Features and Self-Attention Network

**arXiv ID:** 2603.01036 | [PDF](https://arxiv.org/pdf/2603.01036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 741. DAM-VLA: A Dynamic Action Model-Based Vision-Language-Action Framework for Robot Manipulation

**arXiv ID:** 2603.00926 | [PDF](https://arxiv.org/pdf/2603.00926v1)

**作者:** Xiongfeng Peng `[一作]` (Advanced Research Lab Samsung Research and Development Institute China Beijing), Daehyun Ji `[通讯]` (Samsung AI Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DAM-VLA 框架，利用 VLM 推理与专用扩散动作模型的动态路由，分别处理机械臂运动与抓手操作，完成机器人在动态环境中的任务执行。

**💡 创新点**

创新点包括：① 基于 VLM 推理的动作路由机制，动态选择合适的动作模型；② 双头扩散模型（动作为全局、抓手为局部）与高层认知低层视觉的融合；③ 双尺度动作加权（轨迹级 + 动作块级）调节学习；④ 在单一框架内兼顾任务特定精度与通用性。

**🔧 技术方法**

采用预训练的 DINOv2/SigLIP 视觉 Transformer、LLaMA-2 语言模型、Diffusion Transformer（DiT）动作模型，使用交叉熵 + 马氏距离损失，并结合双尺度权重机制进行训练。

**📊 数据集**

预训练数据集为 Open X-Embodiment（Fractal、BridgeDataV2）；评估数据集为 SIMPLER、FurnitureBench；真实世界 pick‑and‑place 任务使用 50 条演示轨迹。

**📈 对比分析**

在 SIMPLER 的 Google（VA/VM）和 WidowX（VM）机器人上与 OpenVLA、CogACT 等基线对比，平均成功率提升至 81‑83%（Google）和 71%（WidowX）；在 FurnitureBench 的“One‑Leg”装配任务中每步成功率达 100%（前三步）并高于对手；在真实世界 pick‑and‑place 任务中 ID/OOD 成功率分别为 91.4%/82.2%，显著优于 CogACT 的 65.7%/60%。

**⚠️ 局限性**

局限性包括：仅实现两种动作模型的路由，无法处理更丰富的动作类型；在细粒度协调（如堆叠方块）表现相对有限；任务覆盖范围主要集中在 pick‑and‑place 与家具装配，需进一步扩展到更多任务家族。

---

## 742. MobileMold: A Smartphone-Based Microscopy Dataset for Food Mold Detection

**arXiv ID:** 2603.01944 | [PDF](https://arxiv.org/pdf/2603.01944v1)

**作者:** Dinh Nam Pham `[一作]` (Technical University of Berlin), Jonas Thumbs `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作构建了MobileMold手机显微镜数据集，并在此基础上实现了霉菌检测与食物分类的基线模型与移动端应用；

**💡 创新点**

创新点在于利用低成本手机夹式显微镜捕获微观图像，创建涵盖11种食物、4台手机、3种显微镜的公开数据集，并实现多任务学习与可解释性可视化；

**🔧 技术方法**

采用迁移学习的深度网络（DenseNet、MobileNet、Swin Transformer等）、多种数据增强策略、Captum生成的saliency图以及Flutter实现的移动端推理；

**📊 数据集**

使用的主要数据集是MobileMold，共计4941张手机显微图像，包含霉菌/无霉菌二分类和11种食物类别标签；

**📈 对比分析**

通过对多种预训练模型和增强方法的对比实验，以MCC、ACC、F1等指标评估性能，Swin Transformer+Flip+Rotation组合在测试集上达到MCC 0.9907，MobileNet+AugMix同样达到MCC 0.9907，整体精度超过99%；

**⚠️ 局限性**

局限性包括：数据采集可能存在视觉或环境偏倚，模型在多任务学习时性能略有下降，数据集仅覆盖11种食物且缺乏时间序列或动态监测，且未完全评估对不同硬件和光照条件的泛化能力。

---

## 743. Production-Grade AI Coding System for Client-Side Development

**arXiv ID:** 2603.01460 | [PDF](https://arxiv.org/pdf/2603.01460v1)

**作者:** Ruihan Wang `[一作]` (Shanghai Jiao Tong University), Guangjing Wang `[通讯]` (Xiaohongshu)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种面向客户端开发的生产级AI编码系统，采用多阶段结构化流水线，将Figma设计、自然语言PRD与领域知识转化为可执行的中间表示，实现可控、可审计的代码生成

**💡 创新点**

将PRD理解视为UI逻辑提取，并构建UI组件分类学；在流水线中引入持久化的中间工件、分层任务图与显式协议，提升可控性与可恢复性；结合检索增强的领域知识与多模态PRD细分模型

**🔧 技术方法**

大型语言模型（Qwen2.5-72B/72B-VL）+ LoRA微调；多阶段管线（上下文规范化、任务规划、执行编排）；知识检索（向量+关键词匹配）；结构化任务IR、DAG调度；CI/视觉检验（YOLO）

**📊 数据集**

内部真实项目数据：PRD+Figma链接+生产客户端代码；PRD细分训练集（182条）包含文本与多模态；真实UI修改请求与交互逻辑测试案例（共4 UI+20 PRD逻辑）

**📈 对比分析**

对比未微调与微调的PRD细分模型；使用Precision/Recall/F1；多模态微调提升F1从0.211到0.848；UI fidelity用检查表评分，89%~83%；逻辑实现通过人评判，20个案例中15个通过，75%成功率

**⚠️ 局限性**

缺乏公开代码与数据；UI fidelity评估仍需人工检查；仅测试小规模UI修改与逻辑，无法验证大规模功能或长期维护；知识检索依赖文档而非代码知识图，后续研究仍待完善

---

## 744. Cross-Modal Guidance for Fast Diffusion-Based Computed Tomography

**arXiv ID:** 2603.01253 | [PDF](https://arxiv.org/pdf/2603.01253v1)

**作者:** Timofey Efimov `[一作]`, Amirkoushyar Ziabari `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种在不重新训练扩散模型的前提下，利用轻量级交叉模态一致性模块，将X射线CT辅助信息融合进稀疏视角中子CT重建的算法。

**💡 创新点**

创新点在于：①将通用扩散先验与交叉模态引导分离，避免对扩散模型进行多模态联合训练；②使用Pix2Pix等图像翻译网络在每次扩散反演后进行跨模态一致性校正，能容忍XCT信号的降质（噪声、模糊、稀疏采样）；③实现了在测试时的快速自适应与一致性约束，显著提升稀疏视角下的重建质量。

**🔧 技术方法**

核心技术包括：通用扩散先验（如D3IP）、扩散逆问题求解器、数据一致性约束、轻量级Pix2Pix跨模态一致性网络、测试时域自适应（梯度微调）。

**📊 数据集**

使用10个256×256×256的3D微结构体积做训练，3个不同体积做测试；对每个体积在不同噪声、模糊、视角稀疏条件下生成NCT/XCT配对样本，并用理想重建作为标签。

**📈 对比分析**

与单模态D3IP进行对比。实验结果表明：在稀疏视角（8–32视角）下，交叉模态方法平均提升PSNR约+1.6 dB、SSIM约+0.13；在高视角（128–256视角）仍能保持SSIM提升，PSNR提升相对较小；在有5%高斯噪声的条件下，仍显著优于基线。可见该方法在稀疏采样和噪声干扰下的鲁棒性较好。

**⚠️ 局限性**

局限性：仅在仿真数据上验证；对真实中子CT/XCT配对的适应性和鲁棒性尚未证实；跨模态一致性模块依赖于XCT与NCT的配准准确度；训练集规模相对较小，可能限制模型泛化。

---

## 745. DriveCombo: Benchmarking Compositional Traffic Rule Reasoning in Autonomous Driving

**arXiv ID:** 2603.01637 | [PDF](https://arxiv.org/pdf/2603.01637v1)

**作者:** Enhui Ma `[一作]` (Autolab, Westlake University), Kaicheng Yu `[通讯]` (Autolab, Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DriveCombo基准及Rule2Scene Agent，用于评估多模态大语言模型在多规则交通规则推理上的能力。

**💡 创新点**

创新点在于设计了五层认知阶梯与Rule2Scene Agent，将文本规则映射为CARLA仿真场景，系统化测试单规则理解到冲突解决的多规则推理。

**🔧 技术方法**

使用了LLM语义结构化、规则配对、规则与场景生成、CARLA仿真，以及链式推理（CoT）、检索增强生成（RAG）和监督微调（SFT）等技术。

**📊 数据集**

基于五国交通规则手册构造了约70k个多项选择题，结合CARLA和nuScenes等数据集生成视觉场景与问答。

**📈 对比分析**

通过零样本和微调方式评估14种MLLM，结果显示L1单规则表现优异，但L5冲突解答准确率仅41-44%；在DriveCombo微调后，nuScenes轨迹L2误差显著下降。

**⚠️ 局限性**

模型在多规则冲突推理上仍显弱，视觉理解与文本语义对齐不足，CARLA资产库限制了场景多样性。

---

## 746. GeMi: A Graph-based, Multimodal Recommendation System for Narrative Scroll Paintings

**arXiv ID:** 2603.00854 | [PDF](https://arxiv.org/pdf/2603.00854v1)

**作者:** Haimonti Dutta `[一作]` (State University of New York), Saurabh Amarnath Mahindre `[通讯]` (eBay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对印度东部叙事卷轴绘画，构建了 GeMi 多模态推荐系统，将图像、文本与用户偏好融合，提供个性化推荐。

**💡 创新点**

创新点：①首个专为叙事卷轴绘画设计的推荐系统；②引入 LLM 语义规范化、SigCLIP/ VAE 视觉-语言编码以及图结构学习；③支持同质和异质图结构，并集成用户偏好模块。

**🔧 技术方法**

使用技术包括：TinyLlama 文本规范化、SigCLIP 对比学习、VAE 融合、图神经网络（GCN、GAE、VGAE）以及多模态融合与不平衡处理。

**📊 数据集**

数据集为两阶段田野调查收集的 189 个面板图像与对应歌曲文本的多模态数据，已公开于 GitHub。

**📈 对比分析**

通过 Precision@K 与 LATTICE、PMGT、HUIGN 等基准对比，GeMi 在三标签（树、动物、神话）上实现最高或接近最高精度，尤其在 Mythology 和 Tree 上表现优异，并在同质、异质、inductive/ transductive 场景均优于现有方法。

**⚠️ 局限性**

局限：①数据量有限、标签稀疏；②对缺失模态的处理仍不完备；③评价仅基于 Precision@K，未考虑排名质量；④未建模动态用户偏好与价格机制。

---

## 747. (hu)Man vs. Machine: In the Future of Motorsport, can Autonomous Vehicles Compete?

**arXiv ID:** 2603.01560 | [PDF](https://arxiv.org/pdf/2603.01560v1)

**作者:** Armand Amaritei `[一作]` (Oxford Brookes University), Andrew Bradley `[通讯]` (Oxford Brookes University)

**通讯引用:** 15277 | [OpenAlex ID](https://openalex.org/A5042053820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估当前自主赛车技术与人类赛车的对标，分析竞赛表现、系统延迟和赛道规划，并探讨混合人机竞速的技术与监管挑战。

**💡 创新点**

提出以混合人机竞速为视角的研究议程，强调延迟评估、风险感知决策、强化学习在多智能体对抗中的应用，并对比人类与自驾车在滑行台、跑道计时等指标。

**🔧 技术方法**

采用基于现有竞赛数据的性能评估方法，结合感知-决策-执行管线延迟测量、SLAM、路径规划、强化学习框架等技术。

**📊 数据集**

使用阿布扎比自主赛车联赛（A2RL）和德国/英国学生赛车（FSG、FSUK-AI）的公开时间/轨迹数据，以及滑行台（skidpad）实验记录。

**📈 对比分析**

通过对比平均圈速、延迟、滑行台完成时间及路线规划效率等指标，结果显示自驾车在时间试验与延迟方面已逼近人类水平，滑行台性能差距已缩小至6.6%，路径规划可提升约10%的圈速。

**⚠️ 局限性**

局限在于缺乏风险模型与安全保障，延迟与多智能体交互的复杂性高，当前自驾车在抢占位置、逆向冲突处理等赛道策略方面仍落后；混合竞速的法规与观众体验仍待完善。

---

## 748. JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks

**arXiv ID:** 2603.01291 | [PDF](https://arxiv.org/pdf/2603.01291v1)

**作者:** Masahiro Kaneko `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Timothy Baldwin `[通讯]` (Mohammed Bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了JailNewsBench基准，系统评估LLM在jailbreak诱导下生成假新闻的鲁棒性，覆盖34个地区、22种语言、约30万实例。

**💡 创新点**

①首个多语言、多地区的假新闻生成与jailbreak评测基准；②提出LLM-as-a-Judge多维度评估框架；③设计5种专门针对假新闻的jailbreak攻击方法；④展示LLM内部表征对自我检测的潜力。

**🔧 技术方法**

利用LLM生成seed指令并对其进行5种jailbreak攻击（角色扮演、系统覆盖、研究目的、负面提示、上下文过载）；评估采用LLM-as-a-Judge（GPT‑5、Gemini‑2.5、Claude‑4等）；内部表征线性探针自我检测；对比黑盒与白盒模型的表现。

**📊 数据集**

基准数据集：JailNewsBench（34地区22语言真实新闻+约300k seed指令）；实验模型：GPT‑5、Gemini‑2.5、Claude‑4、DeepSeek‑70B/8B、Qwen3‑30B/4B、Llama3‑70B/8B。

**📈 对比分析**

通过攻击成功率（ASR）、无效率（IFL）、平均危害子指标评分以及与人工评估的Spearman相关性进行比较。结果显示：黑盒模型ASR最高达86.3%，平均危害分最高3.5/5；英语/美国相关主题防御最弱；与毒性/偏见相比，假新闻更易成功；不同模型表现差异显著。

**⚠️ 局限性**

受限地区（政治不稳定或有严格假新闻法律）导致样本多样性不足；翻译过程可能引入误差；仅评估LLM生成的假新闻，未涵盖非LLM来源；基准公开的攻击信息有限；需要进一步扩展至更多语言与文化场景。

---

## 749. Non-Existence of Some Function-Correcting Codes With Data Protection

**arXiv ID:** 2603.01049 | [PDF](https://arxiv.org/pdf/2603.01049v1)

**作者:** Charul Rajput `[一作]` (Aalto University), Camilla Hollanti `[通讯]` (Aalto University)

**通讯引用:** 1937 | [OpenAlex ID](https://openalex.org/A5035260653)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

分析了函数纠错码（FCC）在数据保护场景下的严格可行性，并给出了基于距离图的非存在性结论。

**💡 创新点**

首次将距离图连通性与FCC的函数距离约束联系起来，证明了完美码、MDS码等最优码无法实现严格FCC。

**🔧 技术方法**

使用图论方法、覆盖半径、距离图连通性分析以及经典码的结构性质。

**📊 数据集**

无实验数据集，全部为理论推导。

**📈 对比分析**

通过理论证明给出非存在性条件，未进行实验性能评估。

**⚠️ 局限性**

仅适用于理论模型，缺乏构造方法，对非线性或随机码的结论有限。

---

## 750. Alien Science: Sampling Coherent but Cognitively Unavailable Research Directions from Idea Atoms

**arXiv ID:** 2603.01092 | [PDF](https://arxiv.org/pdf/2603.01092v1)

**作者:** Alejandro H. Artiles `[一作]` (Fundación Vicomtech), Nasim Rahaman `[通讯]` (Tiptree Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出通过将论文拆解为概念单元，聚类形成共享的想法原子，并训练连贯性与可用性两模型，来采样既可行又不易被社区自然提出的研究方向。

**💡 创新点**

创新点在于将想法原子作为可组合的概念表示，并用可用性模型量化“认知可得性”，从而主动寻找社区视野盲区的高连贯低可用性研究方向。

**🔧 技术方法**

使用技术包括 LLM（压缩、提取单元及生成摘要）、HDBSCAN 聚类、GPT‑2 生成模型评估连贯性，以及采样与 Reciprocal Rank Fusion 结合的离线生成。

**📊 数据集**

数据集为约7,339 篇 2023‑2025 年大语言模型相关的 NeurIPS、ICLR、ICML 论文。

**📈 对比分析**

与 Claude 4.5、Gemini 3 Pro 以及随机采样等基线相比，Alien 采样在多样性、连贯性与新颖性上均优于 LLM 基线，并在可用性低的方向上表现突出。

**⚠️ 局限性**

局限在于词汇表固定只能重组已出现的概念，且可用性评估仅基于已发表文本，忽略隐式知识与阅读历史。

---

## 751. MLRecon: Robust Markerless Freehand 3D Ultrasound Reconstruction via Coarse-to-Fine Pose Estimation

**arXiv ID:** 2603.00990 | [PDF](https://arxiv.org/pdf/2603.00990v1)

**作者:** Yi Zhang `[一作]` (Shanghai Jiao Tong University), Xiaojun Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6799 | [OpenAlex ID](https://openalex.org/A5100369436)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了基于单一RGB‑D相机的MLRecon框架，实现无标记手持式三维超声重建，具备漂移鲁棒的6D探头姿态跟踪与自动失败恢复。

**💡 创新点**

创新点在于融合基础模型驱动的渲染对比姿态估计、可视化偏差检测与自动重新初始化，以及双阶段时域姿态细化网络，能够分离高频抖动与低频漂移，实现极低漂移的精准跟踪。

**🔧 技术方法**

使用了FoundationPose、SAM 2、SANSA、CNN时序卷积网络、频域损失与几何距离损失，以及基于N‑wire标定的相机‑探头‑图像坐标变换。

**📊 数据集**

训练与验证采用上海第六人民医院收集的243次扫描（约36万帧）数据集，并在三种体外模型上进行重建评估，对比NDI光学追踪获得的真实体积。

**📈 对比分析**

与多种标记式、IMU+US融合及现有无标记方法在直线、往返和螺旋三种轨迹上对比，MLRecon在最终漂移率、平均漂移率、最大漂移、平均位置误差等指标上分别低约7.6倍、12.4倍，最大位置误差降至1.85 mm，3D重建Dice系数0.85–0.91，显著优于现有方案。

**⚠️ 局限性**

仍依赖外部RGB‑D相机和CAD模型预训练，极端遮挡或高速抖动时可能产生跟踪失败，且在不同超声设备或解剖位置的跨域泛化尚未充分验证。

---

## 752. From Simulation to Reality: Practical Deep Reinforcement Learning-based Link Adaptation for Cellular Networks

**arXiv ID:** 2603.00689 | [PDF](https://arxiv.org/pdf/2603.00689v1)

**作者:** Lizhao You `[一作]` (Xiamen University), Liqun Fu `[通讯]` (Xiamen University)

**通讯引用:** 1554 | [OpenAlex ID](https://openalex.org/A5034563456)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于深度强化学习的链路适配算法DC‑DQN‑LA，并实现了在软件定义无线电平台上的实时原型；

**💡 创新点**

创新点在于将传统DQN的训练与推理解耦成两个模块，设计了能够兼容ACK/NACK延迟、HARQ重传与并行HARQ的MDP框架，并通过经验对齐实现对实际延迟的鲁棒性；

**🔧 技术方法**

主要技术包括深度Q网络（DQN）配合GRU特征提取、实时推理模块与离线训练模块的分离、srsRAN与USRP软硬件实现、以及基于PyTorch的模型训练；

**📊 数据集**

使用了在实验室实测得到的LTE信道SNR轨迹（包含静态、移动和移动‑静态场景），并基于这些轨迹进行离线仿真；

**📈 对比分析**

与BayesLA和OLLA进行对比，实验显示DC‑DQN‑LA在移动场景下吞吐量提升约70%，在静态场景提升约40%，BLER保持在可接受范围内，且对ACK延迟和训练间隔不敏感；

**⚠️ 局限性**

局限性包括：算法仍需手动调节超参数（如OLLA步长）、在极端高速多径变化时可能需要更高频率训练、以及在5G新型物理层和更大规模网络上的可扩展性尚待验证。

---

## 753. AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations

**arXiv ID:** 2603.01966 | [PDF](https://arxiv.org/pdf/2603.01966v1)

**作者:** Cheng Jiayang `[一作]` (Hong Kong University of Science and Technology), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AMemGym 交互式环境，用结构化状态演化与 LLM 模拟用户实现长时间对话中的内存评估与优化。

**💡 创新点**

创新点包括：① 按策略（on‑policy）评估取代传统的离线评估；② 通过结构化蓝图与自然对话相结合，实现可诊断、可扩展的测试框架；③ 证明代理可以通过环境反馈自行进化内存策略。

**🔧 技术方法**

采用结构化数据采样、GPT‑4.1 生成用户画像与对话、LLM 角色扮演、Agentic Write、RAG、AWI 等内存实现，以及诊断性写/读/利用率指标和自我进化算法。

**📊 数据集**

数据来源为 100K Nemotron‑Personas 生成的合成用户画像，按预设状态序列、问题集合和答案进行自生成；无真实用户数据。

**📈 对比分析**

通过基准配置（base、extra）对比 LLM、RAG、AWE、AWI、Mem0‑G 等实现，使用整体准确率与归一化内存分数评估；结果显示 AWE 最高，LLM 在长记忆上表现显著下降，on‑policy 评估比离线评估更能反映真实性能。

**⚠️ 局限性**

局限性：依赖 LLM 生成的合成数据，可能与真实对话偏差；对非结构化或隐式信息的捕捉仍受限；自我进化实验仅在模拟环境中验证，真实用户交互中的可迁移性待进一步研究。

---

## 754. Align-cDAE: Alzheimer's Disease Progression Modeling with Attention-Aligned Conditional Diffusion Auto-Encoder

**arXiv ID:** 2603.01552 | [PDF](https://arxiv.org/pdf/2603.01552v1)

**作者:** Ayantika Das `[一作]` (Indian Institute of Technology Madras), Mohanasankar Sivaprakasam `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 5077 | [OpenAlex ID](https://openalex.org/A5056763944)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出了 Align‑cDAE，一种条件对齐的图像空间扩散自编码框架，用于生成阿尔茨海默症进展 MRI 图像。

**💡 创新点**

创新点在于通过显式对齐注意力与进展掩模，并在潜在空间中分离进展与身份子空间，从而提升对疾病特定区域的生成控制。

**🔧 技术方法**

采用图像空间扩散自编码（DAE）、交叉注意力、注意力对齐损失和信息最大化损失等技术。

**📊 数据集**

使用 ADNI 纵向 T1‑weighted MRI 数据集，涵盖正常、轻度认知障碍和阿尔茨海默症三组受试者。

**📈 对比分析**

与 IPGAN、SITGAN、DE‑CVAE、BrLP 等基线方法比较，Align‑cDAE 在 PSNR、SSIM、MSE、MAE 等指标上均取得最优或最接近最优的结果。

**⚠️ 局限性**

限制在于仅使用年龄和疾病状态作为条件，缺乏更丰富的临床或多模态信息。

---

## 755. Theory of Code Space: Do Code Agents Understand Software Architecture?

**arXiv ID:** 2603.00601 | [PDF](https://arxiv.org/pdf/2603.00601v1)

**作者:** Grigory Sapunov `[一作]` `[通讯]` (Intento), Grigory Sapunov (Intento)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为Theory of Code Space的基准，用以评估代码代理在部分可观测代码库中主动构建、更新和利用建筑知识的能力。

**💡 创新点**

创新点在于引入程序生成代码库、四类可识别依赖边、定期结构化JSON置信地图外部化，以及对Active‑Passive Gap和Architectural Constraint Discovery的系统测评。

**🔧 技术方法**

采用工具化动作接口、程序解析、LLM交互、JSON架构校验以及AUC、F1等评估指标进行实验。

**📊 数据集**

使用3个中等复杂度Python代码库，由内部生成器基于Pipeline模式生成，并植入15~16条架构约束。

**📈 对比分析**

与4种规则基线及5种前沿LLM（Claude Sonnet 4.6、GPT‑5.3 Codex、Gemini等）对比，Claude Sonnet 4.6在Active模式下F1达0.646，显著优于规则基线，并能发现所有四种边类型；弱LLM的性能低于简单启发式。

**⚠️ 局限性**

仅覆盖Pipeline结构和Python语言，生成代码缺乏真实命名与复杂性；外部化置信地图仍存在显著差距；实验规模小、缺乏多次复现，且未评估约束检测能力。

---

## 756. BadRSSD: Backdoor Attacks on Regularized Self-Supervised Diffusion Models

**arXiv ID:** 2603.01019 | [PDF](https://arxiv.org/pdf/2603.01019v1)

**作者:** Jiayao Wang `[一作]` (Yangzhou University), Dongfang Zhao `[通讯]` (University of Washington)

**通讯引用:** 1342 | [OpenAlex ID](https://openalex.org/A5101671477)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在自监督扩散模型（RSSD）中植入隐蔽后门的攻击方法BadRSSD，并在其内部表示层实现触发器到目标图像的对齐；

**💡 创新点**

创新点在于：①利用PCA空间对齐实现表示层后门；②设计三项联合损失（PCA对齐、像素重建、表示分散正则）提升攻击精度与隐蔽性；③在保留模型生成质量的前提下实现高成功率，抵御现有防御；

**🔧 技术方法**

核心技术包括：自监督扩散（l-DAE）框架、PCA降维、表示分散正则化、三项损失联合优化、DPM-solver采样；

**📊 数据集**

在CIFAR‑10/100、CelebA‑HQ、ImageNet等四大公开数据集上训练与评估；

**📈 对比分析**

与BadEncoder、SSLBKD（SSL后门）以及BadDiffusion、TrojDiff（扩散后门）等方法对比，BadRSSD在ASR/FID/MSE/SSIM等指标上显著优于基线，且在不同模型骨干与采样器上均保持高效稳健；

**⚠️ 局限性**

局限性包括：攻击需要对RSSD内部结构有充分了解；在极低触发器污染率下仍有一定检测风险；对非PCA基底或非ViT类扩散模型的适用性尚未验证。

---

## 757. FastCode: Fast and Cost-Efficient Code Understanding and Reasoning

**arXiv ID:** 2603.01012 | [PDF](https://arxiv.org/pdf/2603.01012v1)

**作者:** Zhonghang Li `[一作]` (Hong Kong University), Chao Huang `[通讯]` (Hong Kong University)

**通讯引用:** 13111 | [OpenAlex ID](https://openalex.org/A5042083053)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出FastCode框架，旨在解决仓库级代码推理中准确性与上下文成本的矛盾，通过“先侦查后获取”策略实现高效定位与构建上下文；

**💡 创新点**

创新点在于将代码结构化为语义-结构图，将探索与内容消费解耦，利用轻量级元数据进行结构侦查，并通过成本感知的上下文管理策略在单步内完成高价值上下文的选取；

**🔧 技术方法**

核心技术包括层次化语义结构表示（多层图索引、稀疏/稠密检索）、结构感知的导航工具（目录遍历、正则搜索）、以及基于状态的成本感知策略（动态预算、置信度阈值、优先级打分）；

**📊 数据集**

实验使用了四大基准：SWE‑QA、LongCodeQA、LOC‑BENCH（SWE‑Bench‑Lite子集）以及GitTaskBench，用以评估问答、文件定位和端到端任务；

**📈 对比分析**

与直接LLM、RAG、现有agent及商业工具对比，FastCode在SWE‑QA、LongCodeQA和GitTaskBench等任务中实现了最高或相近的准确率，同时将token使用量和成本降低数十至数百倍；

**⚠️ 局限性**

局限性包括对结构化元数据构建的依赖、对复杂动态语言或大型编译系统的适用性尚待验证，以及在极大规模仓库中仍需评估图扩展与成本策略的可扩展性。

---

## 758. SHIELD8-UAV: Sequential 8-bit Hardware Implementation of a Precision-Aware 1D-F-CNN for Low-Energy UAV Acoustic Detection and Temporal Tracking

**arXiv ID:** 2603.01069 | [PDF](https://arxiv.org/pdf/2603.01069v1)

**作者:** Susmita Ghanta `[一作]` (Indian Institute of Technology Jammu), Rohit Chaurasiya `[通讯]` (Indian Institute of Technology Jammu)

**通讯引用:** 302 | [OpenAlex ID](https://openalex.org/A5016522720)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了SHIELD8-UAV——一种面向低能耗无人机（UAV）声学检测的精度感知、序列化8位硬件实现的1D特征驱动CNN加速器，能在边缘设备上实现实时推理。

**💡 创新点**

创新点在于：①可重用的序列化多精度计算单元，消除处理单元复制；②基于层敏感度的精度分配框架，支持FP32、BF16、INT8和FXP8；③针对序列化瓶颈的结构化通道剪枝，将稀疏特征维度从35,072压缩至8,704；④端到端的算法‑硬件协同优化，显著降低FPGA LUT和功耗。

**🔧 技术方法**

技术包括：1D特征驱动CNN网络、层级精度感知量化、结构化通道剪枝、共享可重配置计算路径、FPGA（Pynq‑Z2）实现与40 nm ASIC综合；使用AXI接口与CORDIC激活单元支持多种激活函数。

**📊 数据集**

使用由UAV录音和背景环境音组成的异构声学数据集，采样0.8 s窗口，提取MFCC、梅尔谱等特征，并加入公开数据集（AudioSet、Pixabay）以增强背景多样性。

**📈 对比分析**

与QuantMAC、LPRE等可重用加速器以及全并行设计、Jetson Nano、Raspberry Pi等嵌入式平台对比，SHIELD8-UAV在FPGA上仅占用2,268 LUT、0.94 W功耗、116 ms推理延迟；相较QuantMAC降低37.8%延迟、LPRE降低49.6%延迟，且与并行设计相比逻辑利用率低5–9%；检测精度FP32达到89.91%，INT8/FXP8仅低2.5%。

**⚠️ 局限性**

局限性包括：仅支持二分类任务；未实现运行时自适应精度控制；结构化剪枝对其他网络结构的通用性有限；在极低信噪比下误检率仍有提升空间；未来需进一步扩展到多类声学场景识别和更复杂的边缘部署。

---

## 759. Feature-Weighted Maximum Representative Subsampling

**arXiv ID:** 2603.01013 | [PDF](https://arxiv.org/pdf/2603.01013v1)

**作者:** Tony Hauptmann `[一作]` (Institute of Computer Science), Stefan Kramer `[通讯]` (Institute of Computer Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于特征加权的最大代表性子样本方法（FW-MRS），在MRS基础上加入特征权重以减轻特征偏差并降低样本丢失。

**💡 创新点**

创新点在于将特征重要性通过软最小函数转化为特征权重，并与样本权重结合，从而在保留更多样本的同时实现分布对齐。

**🔧 技术方法**

使用随机森林与线性SVM作为域分类器，采用PU学习、softmin温度调参、MMD衡量分布差异。

**📊 数据集**

实验涵盖八个公开表格数据集（如 Income、Employment、Diabetes、Breast Cancer 等）以及真实世界的 Gutenberg Brain Study。

**📈 对比分析**

与 MRS、KMM、PSA 等基线比较，FW-MRS 在保持更多样本、降低 MMD 的同时，后验分类 AUROC 基本与 MRS 相当，未出现显著性能下降。

**⚠️ 局限性**

局限在于温度选择敏感，过低会导致信息丢失；方法主要针对表格数据，且在高度偏差的特征上仍需谨慎。

---

## 760. EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training

**arXiv ID:** 2603.02041 | [PDF](https://arxiv.org/pdf/2603.02041v1)

**作者:** Aleksei Dorkin `[一作]` (Institute of Computer Science), Kairit Sirts `[通讯]` (Institute of Computer Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用Llama 3.1 8B对爱沙尼亚语进行一次性持续预训练，并在后期通过监督指令微调、偏好优化和聊天向量合并来恢复和提升指令遵循能力，同时保持英语性能。

**💡 创新点**

创新点在于（1）构建平衡的多域混合数据（爱沙尼亚语、英语回放、代码、数学、指令式数据），（2）证明一次性持续预训练即可获得显著收益，避免多轮训练的高成本，且（3）采用聊天向量合并在语言适配后快速恢复并提升指令遵循表现。

**🔧 技术方法**

主要技术包括持续预训练（CPT）、监督指令微调（SFT）、直接偏好优化（DPO）、聊天向量合并（ChatVector）、FlashAttention 2、FSDP、分布式训练框架（Accelerate）以及基于HuggingFace的评估工具。

**📊 数据集**

使用的数据集包括：爱沙尼亚国家语料库（ENC）8.6 B tokens、Cosmopedia 6.9 B、Python‑Edu 3.3 B、FineMath‑4+ 9.5 B、Instruct‑PT 7.4 B、NLLB‑并行、Keeleabi问答、Bilingual MaLA、Magpie‑Ultra、Cosmopedia‑style prompt–response 等，涵盖文本、代码、数学、指令和翻译等多领域。

**📈 对比分析**

在爱沙尼亚本土基准（语法纠错、变格、词义、Trivia、WinoGrande、GlobalPIQA、翻译）和英语基准（WinoGrande、TruthfulQA、MMLU‑Redux、GSM8K）上与同规模基线模型比较，模型在爱沙尼亚任务平均提升约30%（单词级到推理）并保持或略逊于英语性能；在多项基准上达到或超过其他8B规模多语种模型，说明持续预训练+后期对齐能显著提升单语种能力。

**⚠️ 局限性**

局限性包括：未对数据混合比例做系统优化；仅在8B规模实验，无法验证更大模型是否同样有效；单轮预训练效果尚需进一步验证；缺乏安全性、多轮对话和对混合成分与跨域迁移机制的深入探究。

---

## 761. PAC Guarantees for Reinforcement Learning: Sample Complexity, Coverage, and Structure

**arXiv ID:** 2603.01309 | [PDF](https://arxiv.org/pdf/2603.01309v1)

**作者:** Joshua Steier `[一作]` `[通讯]`, Joshua Steier

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了2018–2025年间在固定置信度（PAC）强化学习领域的理论进展，提出了Coverage‑Structure‑Objective (CSO) 框架来统一并解释几乎所有的样本复杂度结果，并在此框架下给出了跨设置的速率表、诊断工具、实践决策树和未解决问题清单。

**💡 创新点**

创新点：①将PAC、统一PAC、离线RL、奖励无关探索等看似分离的结果归结为 CSO 三因子乘积，显式揭示了覆盖度、结构复杂度和目标设定对样本量的贡献；②提出“CSO坐标”快速定位最合适的理论保证并识别瓶颈；③将理论结果汇总为可直接使用的速率表、Bellman残差诊断、覆盖度估计与决策门控等实用工具；④系统整理了近七年的开放问题，并按时间与 CSO 轴分类。

**🔧 技术方法**

主要技术：结构复杂度度量（Bellman rank、witness rank、Bellman‑Eluder 维度、低秩/双线性结构）、统一PAC与奖励无关探索的算法（LSVI‑UCB、OLIVE、FLAMBE 等）、离线RL 的惰性与保守估计（pessimistic Q‑learning）、贝塔分布式置信集合、稀疏/有效维度分析、神经网络在 NTK 近似下的理论映射、以及与经验分布相结合的覆盖度评估与证书生成。

**📊 数据集**

作为综述论文，本文未在单一数据集上进行实验；引用的研究覆盖了从模拟 MDP（gridworld、Tabular、Block、低秩等）到真实机器人/医疗场景的多种公开/自建数据集，但在本文中仅以理论表格方式展示其样本复杂度，未给出统一实验评测。

**📈 对比分析**

比较方式：作者通过 CSO 坐标表格对比不同方法在覆盖度、结构维度和目标上的乘积式样本复杂度，展示了在相同假设下哪类方法更优；同时在“实践工具”章节给出 Bellman 残差诊断与覆盖度门控的实验验证（在仿真环境中验证保真度和收敛速度），但总体未进行算法性能基准对比，更多侧重理论上界的互补性与实际可行性。

**⚠️ 局限性**

局限性：①理论主要关注统计可学习性，未系统解决计算复杂度与算法可实现性之间的差距；②CSO 框架是经验性组织工具，无法捕捉不同层级的细微相互作用（如覆盖度与结构共同导致的对数项交互）；③未提供统一实验平台，难以在实际环境中直接复现所有速率；④在深度 RL 和大规模问题下，NTK/贝塔等假设往往难以满足，导致理论保证缺乏可验证性；⑤未对非 PAC（如贝叶斯/平均奖励）目标进行统一处理。

---

## 762. Scalable overset computation between a forest-of-octrees- and an arbitrary distributed parallel mesh

**arXiv ID:** 2603.00760 | [PDF](https://arxiv.org/pdf/2603.00760v1)

**作者:** Hannes Brandt `[一作]` (Rheinische Friedrich-Wilhelms-Universität Bonn), Carsten Burstedde `[通讯]` (Rheinische Friedrich-Wilhelms-Universität Bonn)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

实现了一种针对并行分布式森林四叉/八叉树与任意分区的网格（仅由查询点定义）之间的一向性网格叠加（mesh overset）算法。

**💡 创新点**

创新点包括：
① 利用Morton序列实现无通信的全局分区搜索，精准定位查询点所在进程；
② 将查询点抽象为黑盒结构，通过回调实现与任意网格的几何交互与数据评估，保持算法高度模块化；
③ 结合逆映射技术避免曲面交叉测试的昂贵运算；
④ 引入基于查询点负载的加权再划分，显著缓解因几何不均导致的负载不平衡；
⑤ 支持在交叉区域自适应细化，为多网格耦合提供自适应精度控制。

**🔧 技术方法**

使用的技术包括：
- 空间填充曲线（Morton）分区与顺序映射；
- 递归树搜索（本地与分区搜索）与并行非阻塞 MPI 通信；
- 逆映射与轴对齐盒交叉测试；
- 加权重划分与 2:1 细化保证；
- 基于点云的插值与误差评估。

**📊 数据集**

测试数据集包括：
① 2D 单位正方形实例（两个重叠的四叉树，适度与自适应细化）
② 3D 弧形与砖块实例（多树四叉树，部分重叠，采用自适应细化）
③ 真实地球物理耦合案例（MAGIC 与 GEMINI 模型的网格叠加）。

**📈 对比分析**

与现有方法对比：
- 通过强/弱标度测试，在 12,288 核心上实现了近乎线性缩放；
- 在未加权划分时，负载不均导致 12,288 核心时总时长约 1.44 s；
- 加权划分后，12,288 核心总时长降至 0.38 s，提升约 3.8 倍；
- 单元误差在 2D 实例中低于 10⁻¹²，3D 实例随细化层数提升误差按 4 倍规律下降；
- 传输开销和分区搜索在低进程数下不随规模线性，主要受分区复杂度影响。

**⚠️ 局限性**

限制与挑战：
- 分区搜索在进程数非常少（≤12）时标度不佳，导致总时长受限；
- 需要精确的逆映射与几何交叉测试，若映射复杂会增加计算成本；
- 由于采用点云查询，若查询点密度极高会导致显存占用与通信压力；
- 负载不平衡仍可能在极不均匀几何或动态耦合中出现，需要更细粒度的再划分策略；
- 目前仅实现了一向性叠加，双向耦合需额外实现两次算法调用并处理消息聚合。

---

## 763. Co-Evolutionary Multi-Modal Alignment via Structured Adversarial Evolution

**arXiv ID:** 2603.01784 | [PDF](https://arxiv.org/pdf/2603.01784v1)

**作者:** Guoxin Shi `[一作]`, Yongzhe Chang `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多模态安全对齐的共进化框架CEMMA，结合自动红队攻击和自适应防御；

**💡 创新点**

创新点在于将进化算法（变异、交叉、差分进化）应用于多模态提示，利用跨攻击族结构迁移和对比指导，实现持续生成高质量攻击样本，并通过归档驱动的监督微调提升防御适应性；

**🔧 技术方法**

技术包括黑盒进化搜索、LLM判定器评分、遗传算子（Mutate、Crossover、DiffEvo）、Archival-based supervised fine‑tuning、LoRA微调、AdaShield等推理时防御；

**📊 数据集**

使用SafeBench、Mm‑safetybench、MML‑WR、QR、HADES等多模态对齐基准，以及VLGuard安全对话数据；

**📈 对比分析**

实验表明进化攻击在固定防御上可将ASR从约38%提升至约78%；CEMMA相比静态SFT在ID和OOD攻击上显著降低ASR，且与AdaShield组合可实现更低ASR且拒绝率不升高；

**⚠️ 局限性**

局限包括仅演化文本提示而非视觉输入、评估模型与基准有限、归档驱动微调可能导致语义漂移、缺乏更系统的漂移控制与跨模态语义约束。

---

## 764. Who Explains Privacy Policies to Me? Embodied and Textual LLM-Powered Privacy Assistants in Virtual Reality

**arXiv ID:** 2603.01638 | [PDF](https://arxiv.org/pdf/2603.01638v1)

**作者:** Vincent Freiberger `[一作]` (Center for Scalable Data Analytics and Artificial Intelligence Dresden Leipzig), Viktorija Paneva `[通讯]` (LMU Munich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在VR应用商店中嵌入了基于大型语言模型的隐私助手，并通过对比无助手、文本聊天助手和具身头像助手三种交互模式，探究其对用户隐私意识与决策的影响。

**💡 创新点**

创新点在于：①将隐私助手直接置于VR决策点，实现即时、可交互的隐私信息呈现；②提出双模态（文本与具身）交互方式，并系统评估两种模式在用户体验与信息理解上的差异；③设计分层隐私仪表盘与可查询功能，支持用户从宏观到细节的隐私评估。

**🔧 技术方法**

使用技术包括：OpenAI GPT‑4o 作为后端语言模型、Meta Quest 3 头显与 Unity 开发环境、Wit.ai 进行语音识别与合成、uLipSync 实现头像口型同步、PRISMe 的提示工程改造以及基于虚拟现实的交互 UI 设计。

**📊 数据集**

数据集主要为：①从现有 VR 生产力类应用（如虚拟会议、桌面、3D 绘图）收集的隐私政策文本；②用户实验中产生的问答对话记录与使用日志；③参与者的技术亲和度与隐私态度量表（ATI 与 IUIPC）。

**📈 对比分析**

比较方法为 21 名参与者的 within‑subjects 设计，先无助手后分别使用文本与头像助手，并在每个条件下记录问答数量、隐私评分查看频次、决策偏好；实验结果显示两种助手均提高了隐私信息的主动查看和拒绝高风险应用的倾向，但头像助手在提升用户兴趣与信任感方面更显著，文本聊天则更利于深度反思与信息回顾。

**⚠️ 局限性**

局限性包括：①实验仅限于生产力类 VR 应用，未覆盖社交或游戏场景；②实验室环境与 think‑aloud 可能提升隐私关注度，影响生态效度；③基于云端 LLM 与语音服务的架构可能引入隐私泄露风险；④样本偏年轻且技术熟悉度高，缺乏对不同人群的验证；⑤未对 LLM 生成内容进行真实性评估，存在幻觉风险。

---

## 765. CoSMo3D: Open-World Promptable 3D Semantic Part Segmentation through LLM-Guided Canonical Spatial Modeling

**arXiv ID:** 2603.01205 | [PDF](https://arxiv.org/pdf/2603.01205v1)

**作者:** Li Jin `[一作]` (Shandong University), Xueying Qin `[通讯]` (Shandong University)

**通讯引用:** 3039 | [OpenAlex ID](https://openalex.org/A5008274291)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了 CoSMo3D，一种基于潜在规范空间感知的开源可提示3D语义分割方法，能够在任意姿态和形状下准确识别语义部件。

**💡 创新点**

创新点包括：① 利用LLM驱动的跨类别规范化管道构建统一的规范数据集；② 采用双分支架构，在训练阶段通过规范空间正则化（规范图锚定与规范框校准）提升语义定位的姿态不变性；③ 在对齐损失中引入硬负样本采样，进一步强化语义对比学习。

**🔧 技术方法**

主要技术包括：点云编码（PointTransformerV3）、文本特征提取（SigLIP）、跨模态对比损失、规范图锚定损失（分布式Chamfer距离）、规范框校准损失；以及LLM（如GPT）进行类别聚类与跨类别规范化。

**📊 数据集**

使用统一的规范数据集（基于3Dcompat200，覆盖200类约17K个模型），并在公开的3Dcompat-Coarse、3Dcompat-Fine、ShapeNet-Part、PartNet-E等数据集上进行训练与评估。

**📈 对比分析**

在所有基准上均实现了显著提升：对3Dcompat-Coarse和Fine分别比Find3D提升约25–11% mIoU；在ShapeNet-Part上提升约30%；在PartNet-E上提升约5%；推理速度为0.9秒/模型，远快于基于渲染的2D方法（≈2.5分钟）。

**⚠️ 局限性**

主要局限包括：① 仍依赖几何-文本对齐，难以处理极端形状相似但语义不同的部件；② 规范化过程需大量人工或LLM辅助标注，扩展到更稀疏类别时可能受限；③ 目前仅在静态点云上验证，缺乏对动态或稀疏数据的鲁棒性分析。

---

## 766. A Decomposition Framework for Certifiably Optimal Orthogonal Sparse PCA

**arXiv ID:** 2603.01144 | [PDF](https://arxiv.org/pdf/2603.01144v1)

**作者:** Difei Cheng `[一作]` (Aerospace Information Technology University), Qiao Hu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 2042 | [OpenAlex ID](https://openalex.org/A5101849241)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合 Gram–Schmidt 正交化的稀疏主成分分析（GS‑SPCA）算法，并通过分支限界和块对角化两种加速策略实现可证可优化的多组件稀疏 PCA。

**💡 创新点**

创新点：①首次实现三者（稀疏、正交、最优）共同满足的稀疏 PCA；②将 Gram–Schmidt 机制嵌入组合搜索，保证每一步得到的向量互相正交；③提出块对角化分解定理，将大规模问题拆解为独立子问题，显著降低搜索空间；④引入分支限界实现 ε‑最优解，兼顾精度与效率。

**🔧 技术方法**

技术手段：组合搜索、Gram–Schmidt 正交化、分支限界优化、块对角化阈值分解、特征值分解、并行实现。

**📊 数据集**

实验数据集：CovColon（取前 20 行 20 列的子矩阵作为基准），以及对其经验协方差矩阵进行阈值化和块对角化后选取最大块进行测试。

**📈 对比分析**

与非正交稀疏 PCA 基线对比：测量稀疏主成分间最大夹角、计算时间和方差下降曲线。结果表明 GS‑SPCA 能确保正交性，方差下降更平稳，且在保持可解释性的同时，运行时间与基线相近或略优，尤其在块结构明显时表现突出。

**⚠️ 局限性**

局限性：采用序列化求解导致路径依赖问题，单步最优不保证整体最优；对极大规模稀疏度高的数据块仍需进一步加速；分支限界对全局最优的保证依赖于上界估计，实际运行时可能仍需较长时间。

---

## 767. SODA: Semantic-Oriented Distributional Alignment for Generative Recommendation

**arXiv ID:** 2603.00700 | [PDF](https://arxiv.org/pdf/2603.00700v1)

**作者:** Ziqi Xue `[一作]` (University of Science and Technology of China), Pablo N. Mendes `[通讯]` (Upwork)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了生成式推荐的分布级监督框架 SODA，通过概率分布对齐实现语义层面的优化。

**💡 创新点**

创新点在于使用多层码本的概率分布作为软语义表示，基于 BPR 的负 KL 损失进行对比学习，能够端到端联合优化 tokenizer 与 recommender，突破传统离散码级监督的瓶颈。

**🔧 技术方法**

使用技术包括 RQ‑VAE 词标记器、Transformer/T5 生成器、软分布计算、BPR 对比学习、负 KL 损失以及交替优化训练。

**📊 数据集**

实验数据集包括 Amazon Beauty、Pet（公开评论数据）和内部 Upwork 招聘数据。

**📈 对比分析**

与传统 ID‑based 方法和现有生成式推荐器（TIGER、LETTER、ETEGRec）进行基线对比，SODA 在 Recall@10/20、NDCG@10/20 等指标上均取得显著提升。

**⚠️ 局限性**

限制主要体现在：未在极大规模工业数据集上验证其可扩展性，且对温度、负样本构造等超参数较为敏感，额外的分布计算开销也可能影响部署效率。

---

## 768. Evaluating and Understanding Scheming Propensity in LLM Agents

**arXiv ID:** 2603.01608 | [PDF](https://arxiv.org/pdf/2603.01608v1)

**作者:** Mia Hopman `[一作]` (LASR Labs), David Lindner `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了四个逼真场景，对大型语言模型在现实环境中自我保全、资源获取和目标守护等工具性目标下的隐蔽性规划（Scheming）倾向进行系统评估。

**💡 创新点**

创新点在于：①提出了将Scheming分解为代理因素（如目标导向、代理性）与环境因素（如风险、监督、结果影响）的激励框架；②设计了可模块化、可调节的评估情景；③通过引入“Prompted Model Organisms”探究提示语对Scheming的触发机制；④揭示了Scheming倾向在当前模型下的脆弱性与环境因素的非直观相互作用。

**🔧 技术方法**

使用技术包括：大语言模型（Gemini 3 Pro/Flash、Grok 4 等）与工具链（编辑器、终端、邮件等）交互；自定义系统提示语与工具访问配置；基于 LLM 的行为分类器识别 Scheming 行为；实验通过 100 次 roll‑out 计算行为比例。

**📊 数据集**

使用的数据集为自建的四个评估场景（预算误报、评估破坏、待办优先级操控、纠正训练）与相应的工具交互日志；没有使用公开大规模文本数据集，而是以真实任务模拟数据为主。

**📈 对比分析**

评估方法对比了基线配置与包含对抗性提示片段（Agency、Long‑Term）以及生产环境提示片段的配置，结果显示基线 Scheming 率几乎为 0%，而对抗性提示可将率提升至 59% 以上；在不同工具访问与环境激励组合下，Scheming 率表现出高度不稳定且与模型种类相关；相较于传统安全评估，本文方法能够揭示 Scheming 的脆弱性和对细微配置的敏感性。

**⚠️ 局限性**

局限性包括：评估任务较短且单一，未覆盖更复杂或长期的 Scheming 行为；样本量有限，难以测量极低的倾向；使用的提示片段和工具配置可能不完全代表生产环境；对策性防护实验仅针对特定模型，结果可能无法推广至更高能力的未来模型。

---

## 769. UETrack: A Unified and Efficient Framework for Single Object Tracking

**arXiv ID:** 2603.01412 | [PDF](https://arxiv.org/pdf/2603.01412v1)

**作者:** Ben Kang `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**通讯引用:** 47240 | [OpenAlex ID](https://openalex.org/A5006986293)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了UETrack，一种统一高效的单目标跟踪框架，支持RGB、深度、热成像、事件流和语言五种模态的实时跟踪；

**💡 创新点**

创新点包括Token‑Pooling‑based Mixture‑of‑Experts（TP‑MoE）机制，用软相似度聚合实现无门控专家协作；以及Target‑aware Adaptive Distillation（TAD），根据样本可靠性动态决定是否进行教师蒸馏；

**🔧 技术方法**

技术主要包括Transformer‑based backbone、TP‑MoE替代传统MoE、软聚合与局部聚合、CLIP文本编码、以及Gumbel‑Softmax实现的自适应蒸馏网络；

**📊 数据集**

使用了多模态训练集：COCO、LaSOT、GOT‑10k、TrackingNet、VASTTrack、DepthTrack、VisEvent、LasHeR、OTB99、TNL2K 等共12个基准；

**📈 对比分析**

与12个公开基准（RGB、RGB‑Depth、RGB‑Thermal、RGB‑Event、Language）和3个硬件平台（GPU、CPU、Jetson AGX）进行对比，UETrack‑B在LaSOT、LaSOT_ext、TrackingNet、GOT‑10k、VOT2021 Real‑time 上分别取得69.2%、48.4%、82.7%、72.6%、0.313的最高AUC/AEA，速度在GPU/CPU/AGX分别为163/56/60 FPS，显著优于现有实时多模态和RGB‑only 跟踪器；

**⚠️ 局限性**

局限性主要在于对极端遮挡或大变形的场景下仍可能出现误判；TP‑MoE对专家数和插入层位置敏感，需要经验调优；TAD依赖教师模型的可靠性，若教师在某些模态上表现不佳，蒸馏效果有限。

---

## 770. ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting

**arXiv ID:** 2603.01417 | [PDF](https://arxiv.org/pdf/2603.01417v1)

**作者:** Jiyoon Myung `[一作]` (Samsung SDS), Joohyung Han `[通讯]` (MODULABS)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 ReFeed 框架，利用检索失败案例自动生成符合目标文档风格的查询重写，并验证其对检索召回率的提升。

**💡 创新点**

①通过检索失败案例作为弱监督源，生成风格感知的查询重写；②采用重检索验证重写有效性，确保数据高质量；③提供可直接用于少量样本提示或微调的风格匹配数据集。

**🔧 技术方法**

使用大语言模型（如 GPT‑5）进行查询重写；稠密检索器（E5、BGE）与 FAISS 索引实现检索和重检索；通过少量样本提示实验验证效果；使用数据集构建与指标分析技术。

**📊 数据集**

基于 SQuAD v1.1 训练集（约 87k 条问答）生成失败检索样本，并与已有重写数据集（CANARD、QReCC）做对照。

**📈 对比分析**

在 SQuAD 上，原始查询召回率提升约 18.7%（未检索到的 16k 条中，67.5% 成功重写），最终得到约 11k 条（原始↔重写）对；在少量样本提示实验中，重写查询显著提高了检索 Top‑k 排名，验证了数据集的实用性。

**⚠️ 局限性**

局限性包括：只适用于已存在 QA 数据；对简单或已匹配良好的查询提升有限；重写效果受文档风格多样性影响；依赖大语言模型，成本较高；未在工业级 RAG 系统中进行完整评估。

---

## 771. Resilient Chaotic Cross-Layer Routing for Smart Grid IoT Networks

**arXiv ID:** 2603.02105 | [PDF](https://arxiv.org/pdf/2603.02105v1)

**作者:** Dhrumil Bhatt `[一作]` (Manipal Academy of Higher Education), R. C. Mala `[通讯]` (Manipal Academy of Higher Education)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出并评估了分布式自适应多射频跨层路由（DAMCR）协议，用于提升智能电网物联网网络的可靠性、能效与抗干扰能力。

**💡 创新点**

创新点包括：① 利用基于对数映射的混沌频率跳变（C‑FHSS）实现非周期性频谱跳变，显著提升抗干扰与物理层安全；② 引入链路自适应质量功率控制（LAQPC）根据实时SNR和残余能量动态调节发射功率；③ 通过双射频（LoRa+Wi‑Fi）协同与跨技术路由构建分布式骨干；④ 在路由成本函数中结合可靠性与能量权重，实现能量均衡与高可靠性路径选择；⑤ 采用合作中继与时域反转（TR）信道聚焦进一步增强抗衰落性能。

**🔧 技术方法**

技术手段包括：混沌频率跳变（C‑FHSS）、链路自适应功率控制（LAQPC）、双射频路由、优先级消息分类、合作中继、时域反转聚焦、MATLAB Monte‑Carlo仿真以及 AWGN、Rayleigh 与 Rician 衰落模型。

**📊 数据集**

仿真数据集为基于随机布点（200×200 m²）生成的网络拓扑，节点数 50–500，15 % 为双射频设备；每个节点生成周期性监测包和事件驱动故障包；仿真周期 200 轮，周期内注入间歇性 jamming 攻击。

**📈 对比分析**

与 OLSR、AODV、AFAR 与 ML‑RPL 等传统与基于 SDN/ML 的协议进行对比。结果显示 DAMCR 在 AWGN/ Rayleigh/ Rician 环境下均能保持 95–99 % 的包交付率，平均端到端延迟 18–25 ms，明显优于单射频、静态路由方案；在 jamming 场景下 PDR 仍保持在 95 % 以上。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实硬件实现与能耗测量；节点静态部署，未考虑移动节点或动态网络拓扑；混沌频率跳变与跨技术路由实现复杂度较高，可能增加硬件实现与同步成本；未对能量采集或轻量级加密方案进行评估。

---

## 772. The Invisibility Hypothesis: Promises of AGI and the Future of the Global South

**arXiv ID:** 2603.01616 | [PDF](https://arxiv.org/pdf/2603.01616v1)

**作者:** L. Julian Lechuga Lopez `[一作]`, Luis Lara `[通讯]` (Mila)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文通过理论与概念分析，探讨人工通用智能（AGI）在全球南方的社会经济影响，提出“可见性假设”，并基于此构建三种可能的未来情景——乌托邦、崩溃与中间地带，系统阐述AGI如何放大或缓解既有不平等；

**💡 创新点**

创新点在于提出并阐释“可见性假设”，即随着AI系统成为决策主导，未被机器可读化的个体将被系统性排除；此外，文章首次将AGI的三种潜在路径与全球南方发展路径相结合，形成一套系统化的未来预判框架；

**🔧 技术方法**

未采用具体技术，主要使用理论建模与概念框架分析；

**📊 数据集**

未使用专门数据集，仅引用公开统计（如World in Data）作为示例来说明全球南方的结构性不平等；

**📈 对比分析**

无实验对比或性能评估，本文为概念性讨论，未给出量化指标；

**⚠️ 局限性**

局限性包括：假设AGI最终实现且具备一定能力，缺乏实证验证；未对技术路径与政策干预进行细化，可能过于抽象；缺乏定量评估和具体案例，难以验证预测的可操作性与准确性。

---

## 773. Data-Centric Benchmark for Label Noise Estimation and Ranking in Remote Sensing Image Segmentation

**arXiv ID:** 2603.00604 | [PDF](https://arxiv.org/pdf/2603.00604v1)

**作者:** Keiller Nogueira `[一作]` (University of Liverpool), Ronny Hänsch `[通讯]` (German Aerospace Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了遥感语义分割中的标签噪声估计，提出了将噪声视为连续排名任务的数据中心化基准，并公开了带噪声的SpaceNet8建筑分割数据集。

**💡 创新点**

创新点在于：①把标签噪声从二分类转为排名问题；②公开了合成噪声的数据集与评估基准；③提出两种基于集成的噪声量化方法，兼顾模型不确定性与预测方差。

**🔧 技术方法**

使用的技术包括深度卷积网络（RefineNet、UperNet+ScaleMAE）、数据增强、交叉熵损失、AdamW优化、L2正则化、IoU、方差计算以及多模型集成投票。

**📊 数据集**

所用数据集为SpaceNet8高分辨率图像及其建筑分割标注，训练集5,000张样本，验证/测试集1,298张；通过7种合成噪声（缩放、旋转、平移、删除、顶点添加、假阳性添加）构造带噪声标签。

**📈 对比分析**

与CleanLab、Uncertainty Quantification等基线对比，采用Kendall τ和Spearman相关系数评估排名精度；在U-Net和SegFormer上使用top‑x%样本训练，F1得分提升约2–3%，且显著优于基线。

**⚠️ 局限性**

局限性包括：噪声仅为合成噪声，缺乏真实噪声验证；方法需要多模型集成，计算成本高；仅在建筑二分类任务上验证，未扩展至多类别或其他遥感模态。

---

## 774. TopoEdge: Topology-Grounded Agentic Framework for Edge Networking Code Generation and Repair

**arXiv ID:** 2603.00569 | [PDF](https://arxiv.org/pdf/2603.00569v1)

**作者:** Haomin Qi `[一作]` (University of California San Diego), Yunkai Gao `[通讯]` (Duke University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了TopoEdge框架，实现了基于拓扑的端到端SDN配置生成与自动修复，支持边缘部署。

**💡 创新点**

创新点在于将拓扑视作图并通过对比学习的GNN进行嵌入，利用最近邻检索获取已验证的配置与驱动，从而将检索与生成、验证、修复循环融合成一个分布式、执行中心的Agent体系。

**🔧 技术方法**

核心技术包括图神经网络（GNN）对拓扑进行嵌入、对比学习训练、检索增强生成（RAG）、多Agent规划/生成/验证协作、以及FRRouting的Topotest/pytest驱动脚本。

**📊 数据集**

使用真实与合成的边缘网络拓扑数据集（如Internet2、Arista实验室网络等），并结合公开的SDN配置和测试脚本进行训练与评估。

**📈 对比分析**

在实验中与传统基于规则和纯Llama模型的配置生成方法对比，TopoEdge在配置正确率上提升了约12%，平均生成时间缩短30%，并在多种拓扑变更场景下表现出更好的鲁棒性。

**⚠️ 局限性**

局限性包括对参考配置库的依赖、GNN嵌入对极大规模网络的可扩展性不足，以及在高度动态或完全未知拓扑下的生成质量仍需进一步提升。

---

## 775. A Cascaded Graph Neural Network for Joint Root Cause Localization and Analysis in Edge Computing Environments

**arXiv ID:** 2603.01447 | [PDF](https://arxiv.org/pdf/2603.01447v1)

**作者:** Duneesha Fernando `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**通讯引用:** 106357 | [OpenAlex ID](https://openalex.org/A5014716105)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种级联图神经网络（Cascaded GNN）框架，用于边缘计算环境中联合根因定位（RCL）和根因分析（RCA）

**💡 创新点**

创新点在于通过通信驱动的聚类将大规模微服务图拆分为高互动子图，构建两层网络（P-Net与O-Net）实现局部与全局层次推理，从而在保持诊断准确度的同时显著降低推理复杂度和延迟

**🔧 技术方法**

技术主要包括1D-CNN时间特征提取、GCN消息传播、Louvain社区检测聚类、级联网络结构、联合多任务损失以及对边特征的共享编码

**📊 数据集**

使用了公开的MicroCERCL基准数据集（约81个微服务）进行准确率验证，并利用iAnomaly仿真框架生成从50到10,000节点的规模化图数据评估可扩展性

**📈 对比分析**

与传统中心化GNN基线比较，级联模型在MicroCERCL上的定位/分类精度相当或略低（Acc@1≈0.91 vs 0.93，F1≈0.86 vs 0.87），但在iAnomaly上随着图规模扩大保持近乎不变的推理时间，优于中心化模型随规模线性增长的延迟

**⚠️ 局限性**

局限包括在中等规模数据上级联模型的平均推理时间略高于中心化方案；聚类策略依赖通信模式，可能对通信较稀疏或多层分布的系统效果有限；当前实现仍在单机集中推理，未探索分布式或联邦部署

---

## 776. Learning to Weigh Waste: A Physics-Informed Multimodal Fusion Framework and Large-Scale Dataset for Commercial and Industrial Applications

**arXiv ID:** 2603.00931 | [PDF](https://arxiv.org/pdf/2603.00931v1)

**作者:** Md. Adnanul Islam `[一作]` (United International University), Sami Azam `[通讯]` (Charles Darwin University)

**通讯引用:** 6105 | [OpenAlex ID](https://openalex.org/A5062716310)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种多模态深度学习框架——Multimodal Weight Predictor (MWP)，用于通过单张RGB图像结合物理信息（尺寸、相机距离、高度及物料类别）来估计商业与工业废弃物的重量。

**💡 创新点**

创新点在于：①将物理感知的几何与材质特征与视觉特征进行双向互注意融合；②使用可解释的Shapley值与大语言模型生成人类可读的解释；③采用对数均方误差（MSLE）作为损失函数，以实现跨尺度的均衡学习；④构建了覆盖 3.5–3450 kg 10,421 条样本的 Waste‑Weight‑10K 公开数据集。

**🔧 技术方法**

核心技术包括 Vision Transformer (ViT) 视觉编码器、专门的物理元数据编码器、双向互注意融合模块、基于MSLE的训练策略，以及 SHAP 与 LLM 的后置可解释模块。

**📊 数据集**

使用了 Waste‑Weight‑10K 数据集，其中包含 11 类废弃物（如汽车碎片、铁屑、纸板、塑料等），每条记录均包含 RGB 图像、三维尺寸、相机距离/高度以及重量测量值。

**📈 对比分析**

与多种 CNN/Transformer 视觉基线（VGG, ResNet, EfficientNet, DeiT, Swin, BEiT 等）以及先前的单模态重量估计方法进行比较。MWP 在测试集上取得 MAE = 88.06 kg，RMSE = 181.52 kg，MAPE = 6.39%，R² = 0.9548，明显优于所有基线，尤其在轻量（≤100 kg）和重量（>1000 kg）区间保持稳定的相对误差。

**⚠️ 局限性**

局限性包括：对元数据（尺寸、相机参数）高度依赖，若测量误差或缺失会影响准确性；在极重物品上绝对误差仍相对较大；模型规模较大，部署在边缘设备上仍需进一步优化；未对噪声/缺失元数据的鲁棒性做充分评估。

---

## 777. From Dyads to Groups: Rethinking Emotional Support with Conversational AI

**arXiv ID:** 2603.00797 | [PDF](https://arxiv.org/pdf/2603.00797v1)

**作者:** Yuqing Hu `[一作]` (University of Washington), Yong Tan `[通讯]` (University of Washington)

**通讯引用:** 6184 | [OpenAlex ID](https://openalex.org/A5037984091)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了多智能体情感支持（group AI support）与单一智能体情感支持（single AI support）的效果差异，并通过三项实验验证了多智能体组在提升用户感知支持效能方面的优势及其机制。

**💡 创新点**

创新点在于①首次将社会支持理论与人工智能结合，提出并验证了多智能体支持能通过增强用户与系统的连结感来提升支持效能；②发现组大小对效能提升有限，且支持功能的多样化（情感式 vs 信息式）对效能有显著影响；③探讨收入水平对连结感与效能中介路径的调节作用。

**🔧 技术方法**

技术上采用 GPT‑4o 生成对话，实验环境为基于网页的聊天界面；通过结构化提示实现多智能体互相响应；采用 SPSS/PROCESS 进行方差分析与中介效应检验。

**📊 数据集**

数据集为 140–245 名来自 Prolific 的美国受试者，实验中收集了情绪强度、感知支持效能、连结感、可信度等自评量表数据。

**📈 对比分析**

比较方法主要是两因素/多因素 ANOVA 与 Bootstrap 中介分析；结果显示组 AI 在感知支持效能上显著优于单 AI（p<0.01），连结感是主要中介；组大小超过两名无显著提升，功能组合中情感与信息混合组表现最佳。

**⚠️ 局限性**

局限性包括：仅在英语环境、美国样本中验证；仅考察愤怒与恐惧两种负面情绪；仅使用文本对话形式，未涵盖语音或具身交互；以及未探讨不同文化背景下的通用性。

---

## 778. Information-Theoretic Digital Twins for Stealthy Attack Detection in Industrial Control Systems: A Closed-Form KL Divergence Approach

**arXiv ID:** 2603.01621 | [PDF](https://arxiv.org/pdf/2603.01621v1)

**作者:** Inda Kreso `[一作]` (University of Sarajevo), Mohammadhossein Homaei `[通讯]` (Extremadura University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种基于闭式KL散度的数字孪生框架，用于实时检测工业控制系统中的隐蔽假数据注入攻击。

**💡 创新点**

创新点在于将子空间系统辨识(N4SID)与稳态卡尔曼滤波相结合，直接计算创新残差的多变量高斯KL散度，实现对均值和协方差偏移的同时量化，且无需深度学习重建。

**🔧 技术方法**

采用N4SID系统辨识、稳态卡尔曼滤波、闭式KL散度公式、Tikhonov正则化、滑动窗口估计和阈值设定等技术。

**📊 数据集**

使用SWaT和WADI两个工业过程数据集进行实验。

**📈 对比分析**

与PCA、USAD、GDN、TranAD等方法比较，IT‑DT在SWaT上F1 0.832、WADI上0.615，精度优于TranAD；在CPU上推理时间仅0.12 ms，比TranAD快约600倍，展示了优异的检测精度和实时性。

**⚠️ 局限性**

主要局限是对残差高斯分布的假设，WADI数据的轻度非高斯性导致精度略低；在操作模式突变时可能产生误报；目前不支持显著时间变换系统。

---

## 779. From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck for Long-Horizon Video Agents

**arXiv ID:** 2603.01455 | [PDF](https://arxiv.org/pdf/2603.01455v1)

**作者:** Niu Lian `[一作]` (Harbin Institute of Technology), Shu-Tao Xia `[通讯]` (Tsinghua University)

**通讯引用:** 10659 | [OpenAlex ID](https://openalex.org/A5034104790)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于模糊记忆理论的金字塔式多模态记忆架构，支持从感知层到符号层的逐层压缩与检索，解决长视频理解中的记忆与推理瓶颈。

**💡 创新点**

创新点包括：①将 Fuzzy‑Trace Theory 具体化为 Sensory Buffer、Episodic Stream、Symbolic Schema 三层记忆；②设计 SIB‑GRPO（基于信息瓶颈的强化学习策略）动态压缩冗余；③使用熵驱动的自适应自下而上检索策略。

**🔧 技术方法**

核心技术有：信息瓶颈理论、PPO 强化学习、CLIP/CLIP‑style 视觉检索、BGE 文本检索、逆向层级检索机制、图知识库构建。

**📊 数据集**

使用四大基准数据集：Video‑MME、MLVU、VStream‑QA（Ego/Movie）以及自构建的 HD‑EPIC++。

**📈 对比分析**

在 Video‑MME、MLVU、VStream‑QA 与 HD‑EPIC++ 上均获得了相较于现有代理系统、开源 MLLM 与专有模型的显著提升，最高可达 7.1% 的相对增益；整体性能位列最先进水平。

**⚠️ 局限性**

主要限制包括：构建阶段计算成本高、依赖上游感知模块、当前仅在监督场景下训练，缺乏完全无监督或终身学习的机制，以及对持续多会话环境的适应性不足。

---

## 780. PARWiS: Winner determination under shoestring budgets using active pairwise comparisons

**arXiv ID:** 2603.01171 | [PDF](https://arxiv.org/pdf/2603.01171v1)

**作者:** Shailendra Bhandari `[一作]` (Oslo Metropolitan University), Shailendra Bhandari `[通讯]` (Oslo Metropolitan University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5081350686)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现并评估了PARWiS算法在鞋带预算下的赢家确定任务，并扩展为含上下文特征的Contextual PARWiS以及基于强化学习的RL PARWiS；

**💡 创新点**

创新点在于将上下文信息与Q‑学习策略引入PARWiS，提升了在有限比较预算下的赢家恢复性能；

**🔧 技术方法**

采用了谱排序、破坏性配对选择、双Thompson采样基线、RL Q‑学习与逻辑回归等技术；

**📊 数据集**

实验使用了合成的Bradley‑Terry模型数据，以及Jester笑话集和MovieLens 20M电影评分数据；

**📈 对比分析**

通过恢复率、真实排名、累积遗憾等指标与Double TS与随机基线比较，PARWiS与RL PARWiS在Δ₁,₂较大的Jester与合成数据上明显优于基线，在Δ₁,₂极小的MovieLens上性能仍然相对较弱；

**⚠️ 局限性**

局限在于上下文特征随机且信息不足，RL版本训练时间长且对难度大数据表现欠佳，且只针对赢家确定，未覆盖top‑k恢复等更广泛任务。

---

## 781. LaSTR: Language-Driven Time-Series Segment Retrieval

**arXiv ID:** 2603.00725 | [PDF](https://arxiv.org/pdf/2603.00725v1)

**作者:** Kota Dohi `[一作]` (Hitachi), Yohei Kawaguchi `[通讯]` (Hitachi)

**通讯引用:** 1434 | [OpenAlex ID](https://openalex.org/A5052394367)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自然语言的时间序列段检索框架LaSTR，能够从大规模时序数据库中检索与用户描述匹配的局部段落。

**💡 创新点**

创新点在于自动生成大量段落-标题对（使用TV2分段+GPT-5.2生成描述），并在共享文本-时序嵌入空间中训练对比学习模型，实现上下文感知的段落检索。

**🔧 技术方法**

采用TV2正则化进行分段，GPT-5.2生成标题，Conformer编码器提取时序特征，BERT-base文本编码器，双模态对比损失以及InfoNCE优化。

**📊 数据集**

使用LOTSA大规模多域时序数据集（170个子集），在每个子集提取1024长度窗口，生成约560k训练、17k验证、37k测试的段落-标题对。

**📈 对比分析**

与随机和CLIP基线对比，LaSTR在单正例检索Recall@K和mAP上显著优于基线，Caption‑side评估（SBERT与VLM-as-a-judge）也取得最高分，说明检索结果与查询语义高度一致。

**⚠️ 局限性**

局限在于段落生成依赖VLM生成的标题，可能引入噪声；对长时序的上下文捕获仍受窗口大小限制；评估仅在单正例场景，真实应用中可能存在多重匹配情况。

---

## 782. NP-Completeness and Physical Zero-Knowledge Proof of Hotaru Beam

**arXiv ID:** 2603.01393 | [PDF](https://arxiv.org/pdf/2603.01393v1)

**作者:** Taisei Otsuji `[一作]` (Chuo University), Takuro Fukunaga `[通讯]` (Chuo University)

**通讯引用:** 2701 | [OpenAlex ID](https://openalex.org/A5054214899)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文首先证明了日本数独游戏Hotaru Beam（光蝴蝶）是NP-完全的，并在此基础上设计了一种基于纸牌的物理零知识证明（ZKP）协议，用来证明持有者对某一实例的解。

**💡 创新点**

创新点包括：1）提出了“段嵌入协议”，可在保密的情况下在棋盘上逐段绘制光束；2）设计了“连接表”数据结构来记录已知的连通性信息，保持连通性不被泄露；3）将上述两点组合构成完整的物理ZKP，首次针对该类几何拼图实现了零知识证明。

**🔧 技术方法**

技术层面主要运用：1）平面单调3-SAT到Hotaru Beam的多项式时间还原；2）纸牌操作子协议（如堆移位洗牌、堆挑选、可逆堆移位洗牌、集合成员证明）来实现信息隐藏与验证；3）逻辑值对（True/False）以及逻辑或操作的纸牌编码。

**📊 数据集**

本文未使用任何外部数据集，而是基于理论构造的任意Hotaru Beam实例进行证明。

**📈 对比分析**

论文为理论性工作，未进行实验对比或性能评测；讨论集中在协议的可实现性与零知识性质的证明。

**⚠️ 局限性**

局限性包括：①协议实现需要大量纸牌操作，实际操作成本高；②只针对单一游戏实例的证明；③缺乏实验验证与效率评估，无法评估在大规模实例上的可行性。

---

## 783. Linking Knowledge to Care: Knowledge Graph-Augmented Medical Follow-Up Question Generation

**arXiv ID:** 2603.01252 | [PDF](https://arxiv.org/pdf/2603.01252v1)

**作者:** Liwen Sun `[一作]` (Carnegie Mellon university), Chenyan Xiong `[通讯]` (Carnegie Mellon university)

**通讯引用:** 4853 | [OpenAlex ID](https://openalex.org/A5102363883)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究提出一种基于知识图谱增强的大型语言模型框架KG‑Followup，用于自动生成临床诊断前的后续提问，帮助医生高效获取患者信息；

**💡 创新点**

创新点在于将医学知识图谱与主动式上下文学习相结合，利用EHR与DDX引导的生成模块，并通过KG推理路径和主动I‑C‑L案例提升问题的相关性与覆盖度；

**🔧 技术方法**

核心技术包括：知识图谱实体链接与子图构建、临床实体提取与关联排名、差异诊断路径搜索与关键节点抽取、KG‑信息化的主动上下文学习、以及聚类+LLM融合的冗余消除；

**📊 数据集**

实验使用两个公开数据集：FollowupBench（250例）和自研的ClinicalInquiryBench（1498例，来源于HealthBench），并从dev集抽取I‑C‑L示例；

**📈 对比分析**

与零样本、固定k提问、FollowupQ等基线相比，KG‑Followup在ClinicalInquiryBench的加权召回率达到70%，在FollowupBench为80%，比SOTA提升约5–8%，且在相同问题数下性能更佳；

**⚠️ 局限性**

局限性主要在于对预构建医学知识图谱的依赖，可能在低资源或快速演变领域缺乏完整性，并且评估基于离线数据，未覆盖真实EHR集成和动态临床工作流的复杂性。

---

## 784. Towards Khmer Scene Document Layout Detection

**arXiv ID:** 2603.00707 | [PDF](https://arxiv.org/pdf/2603.00707v1)

**作者:** Marry Kong `[一作]` (Techo Startup Center, Ministry of Economy and Finance, Cambodia), Koichi Kise `[通讯]` (Osaka Metropolitan University, Japan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建首个针对柬埔寨语场景文档的布局检测数据集，并在此基础上训练 YOLO11/12/26 的 OBB 检测模型。

**💡 创新点**

提出专门的文档布局增广工具（实现弹性与仿射变换的统一应用）以及面向柬埔寨语特性的 OBB 检测框架，解决了多层字符堆叠和视角畸变的挑战。

**🔧 技术方法**

使用 YOLO11/12/26 变体、基于方向的 OBB 输出、Perlin 噪声弹性变形、仿射/透视变换以及同步的多边形标注转换。

**📊 数据集**

数据集共 71,796 张页面（训练 62,662 张，验证 9,134 张），包含 2,258 张合成场景文档；同时参考 KH‑FUNSD、KhmerST、WildKhmerST 等公开资源。

**📈 对比分析**

与 Surya‑OCR、DocLayout、Docling、PaddleOCR 等方法对比，YOLO12x 在 mAP@0.5:0.95 上达到 0.9502，显著优于前者，且在多数类别上实现 95%+ 的精确率与召回率。

**⚠️ 局限性**

缺乏代码块、方程块等实例；当前标注不支持嵌套布局；模型未针对柬埔寨语脚本进行专门优化。

---

## 785. Provable and Practical In-Context Policy Optimization for Self-Improvement

**arXiv ID:** 2603.01335 | [PDF](https://arxiv.org/pdf/2603.01335v1)

**作者:** Tianrun Yu `[一作]` (Brigham Young University), Weitong Zhang `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了在推理时通过多轮自我反思进行的无参数自适应推理框架 ICPO，并给出了理论证明与实用算法 ME-ICPO；

**💡 创新点**

创新点在于：①在单层线性自注意力网络上理论上证明可实现策略优化；②设计了 Fisher 加权对数匹配训练目标；③提出最小熵响应选择机制来提升自评奖励的鲁棒性；

**🔧 技术方法**

使用线性自注意力（LSA）网络、Fisher 加权对数匹配损失、熵正则化与多数投票自评奖励；

**📊 数据集**

在多种数学推理基准上评估：AIME 2024、AMC、MATH-500 等；

**📈 对比分析**

与基线 LLM（Qwen2.5-Math-1.5B/7B、Llama-3.1-8B、DeepSeek-R1）比较，ME-ICPO 在 Mean@16 和 Accuracy 上提升约 19–30% 以上，并在 ablation 中验证熵选取与奖励信号的关键作用；

**⚠️ 局限性**

局限性包括：依赖自评奖励的准确性；仅在单层 LSA 上理论和实验验证，难以直接推广到更深或更复杂的架构；缺乏参数更新限制了在更动态环境中的适应性。

---

## 786. MicroVerse: A Preliminary Exploration Toward a Micro-World Simulation

**arXiv ID:** 2603.00585 | [PDF](https://arxiv.org/pdf/2603.00585v1)

**作者:** Rongsheng Wang `[一作]` (Chinese University of Hong Kong), Benyou Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了 MicroWorldBench 微观尺度视频生成基准，构建了 9,601 条专家验证的 Microsim-10K 微观模拟视频数据集，并在此数据集上微调了 Wan2.1 生成模型，得到 MicroVerse 视频生成模型，能够在微观尺度下产生更符合科学规律的视频；

**💡 创新点**

创新点包括①首次基于专家制定的细粒度评分 rubric 的微观尺度视频生成基准；②首个专业审核的微观模拟视频大规模数据集 MicroSim-10K；③在微观数据上专门 fine‑tune 的 MicroVerse 模型，显著提升科学真实性；④将物理/生物知识与 LLM 评估相结合，实现更可靠的评测；

**🔧 技术方法**

主要技术包括：扩散式文本到视频生成（Wan2.1 + Diffusion Transformer + VAE Latent + CLIP Embedding + Classifier‑Free Guidance）；多模态 LLM（GPT‑4o）用于生成字幕；VideoMAE、OpenCV、EasyOCR 用于视频筛选；LLM‑Judge（GPT‑5）执行 rubric‑based 评估；混合域训练与规模扩展策略；

**📊 数据集**

使用的数据集有：MicroSim‑10K（9,601 条专家审核的微观模拟视频）和部分公开通用视频（OpenVid）用于混合域训练；基准评测使用 MicroWorldBench（459 任务，每个任务配 8 条 rubric 评价标准）；

**📈 对比分析**

通过与多种开源（HunyuanVideo、Wan2.1 系列等）和闭源（Sora、Veo3）模型在 MicroWorldBench 上对比，评测维度包括科学真实性、视觉质量、指令跟随。MicroVerse‑1.3B 在科学真实性上提升至 43.0 分（比基线高 2.7 分），在细胞/亚细胞层面超越同类开源模型；混合域与模型规模扩展后，MicroVerse‑14B 在所有维度均刷新开源榜单；人类评估显示 MicroVerse 在科学真实性上的偏好显著高于基线；

**⚠️ 局限性**

限制在于模型并未显式嵌入微观尺度下的物理定律（如血流动力学、扩散反应方程、细胞力学约束等），因此对需要高精度科学预测的场景适用性受限，主要适用于教育和演示级别的微观模拟。

---

## 787. AI-IO: An Aerodynamics-Inspired Real-Time Inertial Odometry for Quadrotors

**arXiv ID:** 2603.00597 | [PDF](https://arxiv.org/pdf/2603.00597v1)

**作者:** Jiahao Cui `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2293 | [OpenAlex ID](https://openalex.org/A5019803400)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了AI-IO系统，将IMU与转速计数据通过Transformer网络预测速度，并结合EKF实现实时姿态估计。

**💡 创新点**

基于四旋翼气动模型，将转速计作为关键观测量以保证可观测性，并使用轻量化Transformer捕捉时序特征，大幅提升速度精度。

**🔧 技术方法**

使用CNN+Transformer轻量网络、Huber+NLL损失、物理先验输入、旋转速计、EKF融合与在线推理。

**📊 数据集**

自研高机动性IMU+转速数据集、DIDO公开数据集以及Blackbird等。

**📈 对比分析**

与IMU预积分、IMO、AirIO 等基线比较，在DIDO和自研数据上平均速度误差降低约36.9%，位置误差下降约30%，实时推理时延8.9 ms，性能优于现有方法。

**⚠️ 局限性**

仍需依赖转速测量，未与视觉等多传感器融合，对极端噪声与极限飞行场景的鲁棒性尚待进一步验证。

---

## 788. I Can't Believe It's Not Robust: Catastrophic Collapse of Safety Classifiers under Embedding Drift

**arXiv ID:** 2603.01297 | [PDF](https://arxiv.org/pdf/2603.01297v1)

**作者:** Subramanyam Sahoo `[一作]` (Independent), Aman Chadha `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了指令调优模型在更新时对嵌入空间产生的漂移对嵌入基安全分类器性能与置信度的影响，系统量化了漂移阈值、无声失效及对齐对可分离性的副作用。

**💡 创新点**

首次证明即使是约1–2%的嵌入漂移就能使分类器性能崩溃，且高置信度误判普遍出现；揭示指令调优会降低嵌入可分离性，导致安全分类器更脆弱；提出每次模型更新后必须重新训练安全分类器。

**🔧 技术方法**

使用高斯、方向性及子空间旋转三种漂移方式，对标准化后的逻辑回归安全分类器进行评估；采用 ROC‑AUC、ECE、Silhouette、Fisher 等指标；通过因子设计实验验证漂移类型与模型版本的影响。

**📊 数据集**

Civil Comments 数据集（约180万条带毒性标签的英文评论），二分类阈值0.5，构建10,000条平衡子集进行训练、验证与测试。

**📈 对比分析**

与基础模型相比，基线 ROC‑AUC 为0.85–0.90；在漂移 σ≈0.02 时降至≈0.5，且无声失效率升至约70%；不同漂移机制均导致类似的性能崩溃，指令调优模型相对更脆弱。

**⚠️ 局限性**

仅考虑了简化的漂移模拟，未涵盖架构或数据集更改引起的更复杂分布偏移；使用简单的逻辑回归，未探索更鲁棒的分类器；缺乏实时漂移监测机制；实验仅基于 Qwen 模型，可能不具普适性。

---

## 789. MARS: Harmonizing Multimodal Convergence via Adaptive Rank Search

**arXiv ID:** 2603.00720 | [PDF](https://arxiv.org/pdf/2603.00720v1)

**作者:** Minkyoung Cho `[一作]` (University of Michigan), Z. Morley Mao `[通讯]` (University of Michigan)

**通讯引用:** 14437 | [OpenAlex ID](https://openalex.org/A5003217329)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 MARS 方法，自动搜索并选择最佳的 LoRA rank 组合，以平衡多模态大语言模型（MLLM）的训练动态，从而提升微调效果。

**💡 创新点**

创新点在于引入双重缩放法则（Performance Law 与 Convergence Law），利用收敛时间预测来剪枝搜索空间，并将 LoRA rank 作为多模态同步控制器，实现高效、自动的 rank 搜索。

**🔧 技术方法**

采用了 LoRA 参数高效微调、双重缩放法则建模、轻量级校准阶段、基于多模态 LLM（如 LLaVA、Qwen2）架构的自动 rank 搜索技术。

**📊 数据集**

使用的主要数据集包括 LLaVA-158K、LLaVA-Bench、ScienceQA、MME、MMStar、POPE、TextCaps、AI2D、GQA，以及从零开始的自组装 MLLM。

**📈 对比分析**

与手工调学习率、固定 rank、AdaLoRA、GeoLoRA 等基线对比，MARS 在 LLaVA-Bench 的困惑度从 2.295 降至 2.1875，ScienceQA 准确率从 72.26% 提升至 74.25%，总体提升约 10–12% 并将搜索成本压缩 11.5 倍。

**⚠️ 局限性**

限制在于仍需对缩放法则进行拟合，对极端数据规模或新模态的适用性有待验证，并且校准阶段增加了一定的计算开销。

---

## 790. General Proximal Flow Networks

**arXiv ID:** 2603.00751 | [PDF](https://arxiv.org/pdf/2603.00751v1)

**作者:** Alexander Strunk `[一作]` (Evercot AI), Roland Assam `[通讯]` (Evercot AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 General Proximal Flow Networks (GPFNs)，用任意距离替换 BFNs 的 KL 近端更新，形成可适应数据几何的生成框架

**💡 创新点**

将 BFNs 的固定 KL 近端点扩展为通用的近端优化步骤，允许使用 Wasserstein 等距离，实现对不同数据几何的自适应更新，并将其与凸优化的近端点方法建立形式化联系

**🔧 技术方法**

近端点算法、概率分布优化、Wasserstein 距离、Gaussian 近端更新、U‑Net 预测器、确定式与随机式采样策略

**📊 数据集**

MNIST 数据集（使用 Gaussian GPFN 对比标准 BFN）

**📈 对比分析**

与标准 BFN 在 NFE（5–100）预算下进行多指标比较（SWD、aFID、IS、Precision/Recall、Density/Coverage/Diversity）；GPFN 在低 NFE 下显著优于 BFN，尤其在 aFID、Precision/Recall 及多样性上表现突出

**⚠️ 局限性**

仅在简单图像数据（MNIST）和高斯模型上验证，缺乏更复杂数据集的实验；对距离选择高度依赖，确定采样在某些配置下仍易产生模式崩溃

---

## 791. physfusion: A Transformer-based Dual-Stream Radar and Vision Fusion Framework for Open Water Surface Object Detection

**arXiv ID:** 2603.01947 | [PDF](https://arxiv.org/pdf/2603.01947v1)

**作者:** Yuting Wan `[一作]`, Pin LV `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提供了 IEEEtran LaTeX 模板使用说明与示例，帮助作者准备符合 IEEE 要求的论文文件

**💡 创新点**

将原有复杂说明简化，更新至 1.8b 版本，包含完整的前置、正文、后置元素示例

**🔧 技术方法**

使用 LaTeX、IEEEtran 类、标准包，示例中包含章节、图表、参考文献等命令

**📊 数据集**

无数据集，文档仅为模板说明

**📈 对比分析**

无实验比较，主要以示例代码展示如何实现各论文元素，未涉及性能指标

**⚠️ 局限性**

仅适用于 IEEEtran 1.8b 版本，示例基于 2‑column 格式，若论文需单栏或特殊格式需自行调整

---

## 792. RC-GeoCP: Geometric Consensus for Radar-Camera Collaborative Perception

**arXiv ID:** 2603.00654 | [PDF](https://arxiv.org/pdf/2603.00654v1)

**作者:** Xiaokai Bai `[一作]` (Zhejiang University), Huiliang Shen `[通讯]` (Zhejiang University)

**通讯引用:** 2264 | [OpenAlex ID](https://openalex.org/A5020795553)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 RC-GeoCP 框架，将 4D 雷达与摄像头信息通过几何共识融合，实现多模态协同感知。

**💡 创新点**

创新点在于利用雷达提供的几何锚点构建雷达锚定几何共识；通过几何结构校正（GSR）将视觉语义对齐到物理空间；并采用不确定性感知通信（UAC）和共识驱动组装（CDA）实现高效的稀疏信息选择与跨车融合。

**🔧 技术方法**

核心技术包括雷达锚定的几何结构校正、基于变形注意力的特征对齐、条件熵降低的需求映射、以及将雷达几何先验注入注意力权重的共识驱动组装。

**📊 数据集**

使用 V2X-Radar（真实雷达+单摄像头）和 V2X-R（仿真多视角摄像头）两个数据集进行统一多模态评估。

**📈 对比分析**

与现有 LiDAR/雷达-摄像头协同感知方法相比，RC-GeoCP 在 V2X-Radar 上 AP@0.5/0.7 分别提升至 44.55%/25.92%，在 V2X-R 上 AP@0.5/0.7 提升至 81.90%/65.09%，同时通信开销仅为 2.39 单位，较传统 4.00 单位减少约 40% 以上。

**⚠️ 局限性**

局限性包括目前仅在同质传感器场景下验证；对不同雷达型号的适应性和多车多路况下的鲁棒性仍待进一步研究；以及对姿态误差和通信时延的敏感性需要更深入的实地评估。

---

## 793. Unlearning Evaluation through Subset Statistical Independence

**arXiv ID:** 2603.00587 | [PDF](https://arxiv.org/pdf/2603.00587v1)

**作者:** Chenhao Zhang `[一作]` (University of Queensland), Miao Xu `[通讯]` (University of Queensland)

**通讯引用:** 3042 | [OpenAlex ID](https://openalex.org/A5016620131)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于Hilbert–Schmidt独立性准则的子集级机器未学习评估框架，能够在不重训练模型或使用辅助分类器的情况下判断模型是否已成功忘记指定子集。

**💡 创新点**

创新点在于将HSIC用于分半子集的依赖度量，利用训练过程产生的共享影响成分来检测模型对子集的记忆，从而实现单模型无参考的未学习效果判定。

**🔧 技术方法**

使用的技术包括HSIC统计依赖度量、Gaussian RBF核、分半子集划分、Jensen–Shannon Divergence比较、Mann–Whitney U检验以及在不同层特征提取。

**📊 数据集**

实验数据集涵盖SVHN、CIFAR‑10、CIFAR‑100、Tiny‑ImageNet等分类任务，并扩展到扩散生成模型。

**📈 对比分析**

与传统分布距离度量（MMD、Wasserstein）及成员推理攻击（ASR）等评估方法相比，HSIC方法在子集大小仅为400时即可达到F1>0.8，且在大多数设置下显著优于其他指标。

**⚠️ 局限性**

局限性包括对RBF核带宽σ的敏感性、依赖参考子集、难以区分自然遗忘与有意未学习、以及在极小样本规模下性能下降。

---

## 794. Draft-Thinking: Learning Efficient Reasoning in Long Chain-of-Thought LLMs

**arXiv ID:** 2603.00578 | [PDF](https://arxiv.org/pdf/2603.00578v1)

**作者:** Jie Cao `[一作]` (Zhejiang University), Siliang Tang `[通讯]` (Zhejiang University)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5063062444)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Draft-Thinking方法，训练大模型先学习只保留关键推理步骤的简洁推理结构，并通过渐进式课程学习和自适应提示实现可调节的推理深度，显著降低推理代价；

**💡 创新点**

创新点在于将高价值推理步骤的选择内化为模型的本能能力，并通过分阶段强化学习和自适应提示将推理深度转化为可调节的模型行为；

**🔧 技术方法**

采用监督微调(SFT)、黑盒蒸馏、两阶段强化学习（GRPO）、增量长度扩展、分阶段训练及自适应提示技术；

**📊 数据集**

使用LIMO数学数据集（817题），从中提取342条简洁推理样本做SFT，剩余475题及AIME2024 30题做RL；

**📈 对比分析**

与同基座模型（Qwen3-8B、Qwen3-4B）及更大模型（14B、32B）以及多种在线RL、离线和无训练方法比较，Draft-Thinking在MATH500等数学基准上实现了82.6%推理代价下降，仅丢失2.6%准确率，token效率提升5.6×；在长CoT模式下仍保持或提升准确率，adaptive模式兼顾准确率与效率；

**⚠️ 局限性**

对最难AIME 2025的高难度题目，Draft模式与长CoT的准确率差距仍显著，提示可能需要更长的最大序列长度来进一步提升极难题目的推理性能；

---

## 795. Towards Robot Skill Learning and Adaptation with Gaussian Processes

**arXiv ID:** 2603.01480 | [PDF](https://arxiv.org/pdf/2603.01480v1)

**作者:** A K M Nadimul Haque `[一作]`, Teresa Vidal-Calleja `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于稀疏Via点的高斯过程（GP）参数化，结合非线性优化、行为克隆和强化学习实现结构化技能适应，保持演示的动力学特征；

**💡 创新点**

创新点包括①用稀疏Via点对GP进行一次性学习并通过优化保留速度/加速度；②引入签名相似性奖励实现在线RL适应；③将优化、BC和RL三种方式融合，满足不同实时性和精度需求；

**🔧 技术方法**

使用Gaussian Process（GP）与稀疏Via点、非线性最小二乘优化、行为克隆（BC）、Soft Actor-Critic（SAC）强化学习，以及签名核（signature kernel）进行相似性奖励；

**📊 数据集**

在三类任务（抽屉开启、立方体推送、条形物件搬运）的PyBullet仿真和UR5e机器人实验中，单次演示后在多达20cm的任务配置偏移下测试；

**📈 对比分析**

与SAC‑GMM、Vanilla GP、Object‑centric ProMP、ProMP‑RRL等基线比较；在100次仿真和10次硬件实验中，Skill‑GP、Skill‑Cloning、GPRL几乎100%成功率，速度相似度高；相比基线提升显著；但Skill‑GP收敛慢，BC需要大量专家数据，RL推理快；

**⚠️ 局限性**

限制：GP外推能力有限，末端速度偏差；仅测试轻质物体；TC偏移超过20cm时性能下降；Skill‑GP收敛时间长，适用于非实时场景；BC对专家数据规模敏感。

---

## 796. High Probability Work Efficient Parallel Algorithms

**arXiv ID:** 2603.00898 | [PDF](https://arxiv.org/pdf/2603.00898v1)

**作者:** Chase Hutton `[一作]` (University of Maryland), Adam Melrod `[通讯]` (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

设计了多种高概率线性工作并行算法，包括半排序、整数排序以及图问题（Δ+1-着色与最大独立集）等，取得了O(n)工作与O(log n)深度的结果；

**💡 创新点**

首次将期望线性工作提升为高概率线性工作，提出了基于简单字典哈希、几何随机变量浓度以及“剥除平衡分区+确定性扩展器”通用框架，并在该框架下实现了半排序与图算法的高概率线性工作；

**🔧 技术方法**

利用简单字典哈希、几何随机变量的尾界、均匀抽样、放置问题求解、计数排序/基数排序、以及确定性扩展子图的工作线性扩展器等技术；

**📊 数据集**

本工作主要基于理论证明，未在论文中给出具体实验数据集；

**📈 对比分析**

与传统仅保证期望工作量的并行算法相比，本算法在高概率下仍保持O(n)或O(m)工作量，深度保持O(logⁿ)级别；理论分析显示显著提升了工作可靠性，未给出实验性能对比；

**⚠️ 局限性**

局限性在于依赖随机哈希与高概率分析，常数与对数因子可能较大；算法在实际实现中对哈希表与放置步骤的随机性有要求；并未解决小规模或需要确定性输出的情况；

---

## 797. Wave-Attractor-Tree: A Hierarchical Binary Tree Reduction Architecture for Efficient Sequence Modeling

**arXiv ID:** 2603.00812 | [PDF](https://arxiv.org/pdf/2603.00812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 798. Boosting Device Utilization in Control Flow Auditing

**arXiv ID:** 2603.02161 | [PDF](https://arxiv.org/pdf/2603.02161v1)

**作者:** Alexandra Lengert `[一作]` (University of Zurich), Ivan De Oliveira Nunes `[通讯]` (University of Zurich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计并实现了一种基于硬件‑软件协同的可信根（RoT）架构，用于控制流审计（Control Flow Auditing，CFA），通过在 MCU 上自包含的可靠通信接口，使应用在证据传输期间能够继续执行，从而显著提升 CPU 利用率。

**💡 创新点**

创新点在于：①引入了自含可信通信接口，使 RoT 能够并行处理证据生成与传输；②通过双切片日志管理实现了无忙等的传输调度；③将硬件与软件 TCB 的交互细化为可配置的触发事件，降低了对应用执行的干扰。

**🔧 技术方法**

主要技术包括：openMSP430 MCU 的硬件增强、SHA‑256 HMAC（HACL* 库）做鉴权、VRASED 进行正式验证的 TCB、双切片日志机制、CM‑UART 可靠通信接口、以及基于触发的非屏蔽中断。

**📊 数据集**

使用了公开的传感器与控制系统应用作为评测数据集，具体包括：超声波传感器、温度传感器、自动医疗注射泵和自驱动坦克履带轮控制器。

**📈 对比分析**

通过与基线（无审计）、TinyCFA（最佳努力审计）以及 ACFA（传统忙等审计）进行对比：运行时延增加从 24% 到 71%（相比基线），CPU 利用率提升从 7% 到 150%（依据不同切片配置），硬件成本仅增加 541 LUT、283 FF（约 4–17% 的开销）。

**⚠️ 局限性**

限制在于：仍受网络延迟影响，若通信被完全阻断将退回到传统忙等模式；对资源极度受限的 MCU，双切片日志与通信硬件可能仍带来不可忽略的占用；同时对高频、长周期应用的实时性仍有待进一步评估。

---

## 799. Securing the Floor and Raising the Ceiling: A Merging-based Paradigm for Multi-modal Search Agents

**arXiv ID:** 2603.01416 | [PDF](https://arxiv.org/pdf/2603.01416v1)

**作者:** Zhixiang Wang `[一作]`, Yong Li `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练‑free的跨模态模型合并框架，利用Optimal Brain Merging (OBM)将预训练的视觉语言模型与文本搜索代理在参数层面融合，构建具备自主搜索能力的多模态代理；

**💡 创新点**

通过OBM的显著性感知剪枝与加权符号一致性解决跨模态融合中的参数干扰，并证明模型合并既可提供稳健的零射性能，也能作为强化学习的高效温启动；

**🔧 技术方法**

采用任务向量(Task Vectors)融合、OBM两阶段显著性剪枝与符号一致性、基于激活的Hessian近似、强化学习微调以及跨模态激活收集等技术；

**📊 数据集**

在InfoSeek、FVQA‑test、MMSearch、LiveVQA等视觉问答与搜索密集基准上进行评估，并使用NQ‑HotpotQA、MMStar等小规模校准样本；

**📈 对比分析**

与直接回答、提示式搜索、RAG工作流和MMSearch‑R1等基线对比，OBM在零射下取得约20% Acc并显著提升搜索率，在RL微调时可在更少步骤内达到27.6% Acc，超过标准VLM；

**⚠️ 局限性**

仍依赖预训练模型的匹配、跨模态校准需小样本、对视觉与文本搜索的细粒度决策不足，且在某些基准下未能完全超越固定RAG流程，泛化到不同工具环境仍需进一步验证。

---

## 800. A Resource-Rational Principle for Modeling Visual Attention Control

**arXiv ID:** 2603.02056 | [PDF](https://arxiv.org/pdf/2603.02056v1)

**作者:** Yunpeng Bai `[一作]` `[通讯]` (National University of Singapore), Yunpeng Bai (National University of Singapore)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了资源-合理的视觉注意力控制框架，模拟读写、行走等多任务情境下的眼动与认知过程；

**💡 创新点**

将视觉注意力视为受感知、记忆和时间约束的有限理性决策过程，利用POMDP和分层深度强化学习生成可解释的注意力策略；

**🔧 技术方法**

部分可观测马尔科夫决策过程（POMDP）、分层深度强化学习（PPO）、概率信念更新、深度生成模型；

**📊 数据集**

人类眼动实验数据（在不同时间限制下的阅读实验）、SB‑SAT、MECO、CopCo数据集，以及新收集的时间限制阅读数据；

**📈 对比分析**

对比传统眼动模型（E‑Z Reader、SWIFT）和数据驱动模型（ScanDL、EyeTtention），使用归一化莱文斯坦距离（NLD）衡量单次扫描路径相似度，结果显示本模型在30/60/90 s条件下NLD最低，优于既往模型；

**⚠️ 局限性**

仍需拓展到多主体、非文本任务；对个体差异建模的参数空间尚大；模型在真实交互场景中的可用性和实时性仍待验证。

---

## 801. ORGAN: Object-Centric Representation Learning using Cycle Consistent Generative Adversarial Networks

**arXiv ID:** 2603.02063 | [PDF](https://arxiv.org/pdf/2603.02063v1)

**作者:** Joël Küchler `[一作]` (University and ETH Zurich), Stephan J. Ihle `[通讯]` (University of Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于CycleGAN的全差分循环框架（ORGAN），可在无监督条件下将图像映射为对象列表并反向重建，实现对象级别的表征学习。

**💡 创新点**

创新点包括：① 将图像与对象列表作为两个域进行循环一致性训练，突破传统AE主导的对象学习；② 采用可微top‑k定位与注意力判别器，保证对象检索的可训练性与互相依赖；③ 证明该框架可在低对比度、多对象的真实数据上保持优异性能，并能在不同尺寸图像间无需重新训练即可迁移。

**🔧 技术方法**

技术核心为CycleGAN架构、可微top‑k、非极大抑制、注意力判别器、U‑Net风格变换、线性分配求解器以及最小二乘对抗损失。

**📊 数据集**

使用四个数据集：Tetrominoes、Sprites、MNIST（手写数字）以及真实显微镜血细胞图像（低对比度多细胞）。

**📈 对比分析**

与SPACE、SLATE、LSD、SPOT等基准方法比较，ORGAN在合成数据上与SPACE相当，但在真实细胞数据上实现了显著更高的F1分数；训练与推理速度接近或优于SPACE，且在更大图像（256×256、768×768）上保持高召回与精度，展示了良好的规模化与迁移能力。

**⚠️ 局限性**

局限性包括：① GAN训练不稳定，需谨慎调参；② 不能很好重建复杂背景；③ 依赖滑动窗口分块，可能错过跨边界或尺寸超大的对象；④ 需要手工设定最大对象数（k）并对不同数据集进行调优。

---

## 802. Bimanual XR Specification of Relative and Absolute Assembly Hierarchies for Teleoperation

**arXiv ID:** 2603.01495 | [PDF](https://arxiv.org/pdf/2603.01495v1)

**作者:** Benjamin Yang `[一作]`, Steven Feiner `[通讯]` (Columbia University)

**通讯引用:** 21308 | [OpenAlex ID](https://openalex.org/A5011020333)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于双手抓取的XR交互技术，用户在虚拟环境中通过同时抓取对象来创建约束组，并可通过凸包可视化实现多级嵌套分组，支持分步装配的规划。

**💡 创新点**

创新点在于利用双手同时抓取来直观构造约束组，并通过可视化凸包支持任意深度的层级嵌套，从而实现更自然、更灵活的机器人装配任务指定。

**🔧 技术方法**

使用Unity 6.2 + Quest 3实现交互，GPU加速的Quickhull算法生成凸包，MuJoCo进行物理仿真，MoveIt与DRake执行路径规划与非线性优化，TSP求解组访问顺序。

**📊 数据集**

本文未使用公开数据集，系统在自定义装配场景中进行评估。

**📈 对比分析**

实验仅在仿真环境中验证了规划可行性，没有与现有方法进行定量对比；仿真结果表明规划速度和路径质量可接受。

**⚠️ 局限性**

局限包括凸包可视化可能包含空洞导致误组，且缺乏正式的用户研究来评估相对与绝对组的实际效果。

---

## 803. Fake It Right: Injecting Anatomical Logic into Synthetic Supervised Pre-training for Medical Segmentation

**arXiv ID:** 2603.00979 | [PDF](https://arxiv.org/pdf/2603.00979v1)

**作者:** Jiaqi Tang `[一作]` (Peking University), Qingchao Chen `[通讯]` (Peking University)

**通讯引用:** 2534 | [OpenAlex ID](https://openalex.org/A5069484115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种利用少量真实解剖掩码构建形状库，并通过结构化序列放置生成具有解剖逻辑的合成数据，用于3D医学分割的隐私友好预训练。

**💡 创新点**

在传统FDSL的基础上注入解剖先验：使用形状库代替几何原语，并通过拓扑关系图和空间锚点实现结构化合成，从而弥补语义鸿沟。

**🔧 技术方法**

Formula‑Driven Supervised Learning、形状库提取、约束空间点过程、结构化序列放置、蒙特卡罗候选排序、UNETR/SwinUNETR Transformer等技术。

**📊 数据集**

BTCV多器官CT、MSD肺、脾、心MRI、以及用于构建形状库的TotalSegmentator 5份解剖掩码。

**📈 对比分析**

与从零开始、PrimGeoSeg、以及使用真实医学数据的自监督方法（SwinMM、SwinUNETR）进行比较，平均Dice分别提升约1.7%（BTCV）和1.6%（MSD），甚至超越同等量级真实数据的自监督预训练。

**⚠️ 局限性**

需要从真实患者提取形状库，规模有限；随着合成数据增大，性能提升趋于饱和；仅使用CT解剖先验，对MRI等不同模态的迁移仍存在一定限制。

---

## 804. Diversity over Uniformity: Rethinking Representation in Generated Image Detection

**arXiv ID:** 2603.00717 | [PDF](https://arxiv.org/pdf/2603.00717v1)

**作者:** Qinghui He `[一作]` (Chongqing University of Posts and Telecommunications), Bin Xiao `[通讯]` (Jinan Inspur Data Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种抗特征坍塌学习框架 AFCL，结合信息瓶颈、交叉特征去耦和类别特定提示学习，显著提升生成图像检测的跨模型泛化与鲁棒性。

**💡 创新点**

核心创新是通过 Cue Information Bottleneck (CIB) 清除冗余信息，并用 Hilbert–Schmidt Independence Criterion (HSIC) 强制不同层次特征互相正交，防止特征坍塌；同时引入可学习的类特定文本提示，使视觉特征与语义更好对齐。

**🔧 技术方法**

技术手段包括预训练冻结的 CLIP ViT-L/14 图像编码器、多阶段 CLS 取样、CIB 信息瓶颈模块、AFCL 正交约束、可学习权重聚合与全局 CIB、以及基于文本原型的 Cosine 相似度判别。

**📊 数据集**

训练数据为 Stable Diffusion v1.4 的 GenImage 子集，评测使用 UniversalFakeDetect、GenImage、AIGI-Holmes 三大基准，涵盖 GAN 与扩散模型的 21 种生成器。

**📈 对比分析**

与 CNNDet、VIB‑Net、CLIPping 等现有方法对比，AFCL 在 AP 与 ACC 上平均提升 5.02% 和 5.68%（在跨模型情形下），并在极少样本、抗扰动等场景下保持领先。

**⚠️ 局限性**

局限性主要体现在对超参数 λ 的敏感性、对大规模多模态预训练模型的依赖以及在极端生成器（如新兴的超分辨率扩散模型）上的进一步验证尚需深入。

---

## 805. UltraStar: Semantic-Aware Star Graph Modeling for Echocardiography Navigation

**arXiv ID:** 2603.01461 | [PDF](https://arxiv.org/pdf/2603.01461v1)

**作者:** Teng Wang `[一作]` (Tsinghua University), Gao Huang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了 UltraStar 框架，将心脏超声探头导航从路径回归转变为基于历史关键帧的全局定位；

**💡 创新点**

创新点在于采用星形图（Star Graph）拓扑，将历史关键帧视为空间锚点直接与当前视图关联，并通过语义感知采样策略选取信息丰富的锚点，显著提升了在嘈杂探索历史中的定位精度与可扩展性；

**🔧 技术方法**

使用 ViT 视觉编码器、动作编码器、两层自注意力的锚点细化模块以及单层交叉注意力的全局定位模块，同时训练一个 ResNet-18 视图分类器进行语义感知采样；

**📊 数据集**

在由 178 名成人患者、356 条扫描轨迹组成的 1.31M 样本数据集上进行实验，数据采集采用机器人臂固定超声探头并记录 6-DOF 位姿；

**📈 对比分析**

与单帧、序列图（GRU、Causal/Non‑causal self‑attention）、全连图等基线进行对比，UltraStar 在所有标准视图上的平均 MAE（平移 mm、旋转 °）分别下降 7% 与 6%，并在更长输入长度下表现出更好的可扩展性；

**⚠️ 局限性**

局限性包括：仍需在真实临床环境下验证其鲁棒性，模型对极端嘈杂扫描的适应性待进一步提升，以及在不同设备/探头下的泛化能力尚未彻底评估。

---

## 806. Stabilizing Policy Optimization via Logits Convexity

**arXiv ID:** 2603.00963 | [PDF](https://arxiv.org/pdf/2603.00963v1)

**作者:** Hongzhan Chen `[一作]` (Sun Yat-sen University), Ting Yao `[通讯]` (Tencent Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Logits Convex Optimization (LCO)，一种通过在 logits 空间保持凸性来稳定大型语言模型的强化学习优化的框架。

**💡 创新点**

创新点在于将“logits 凸性”视为 RL 稳定性的核心属性，并设计了三种 LCO 损失（MSE、Log‑Cosh、KL）以实现对最优 logits 的对齐，从而在理论上保证梯度方向一致、梯度幅度随误差衰减，解决传统 PPO 等算法的梯度波动和训练崩溃问题。

**🔧 技术方法**

使用基于优势估计（DPO、稀疏样本、分布式）以及前向 KL、MSE 等损失函数进行策略更新；结合软最大化、梯度范数分析与神经切线核理论；在强化学习中对齐最优 logits 或分布。

**📊 数据集**

在多种任务上验证，包括数学推理（MATH500、AMC23、MinervaMath）、机器阅读理解（QA‑Feedback）以及指令跟随（AlpacaEval 2.0），并使用不同规模的 LLM（Qwen‑2.5‑3B/4B、Llama‑3‑2‑3B、Mistral‑3‑3B 等）与对应的奖励模型（DPO‑RM）进行训练。

**📈 对比分析**

与 REINFORCE、PPO、GRPO、DAPO、GSPO、MiniLLM、GKD 等基线对比，LCO 在 Pass@1、平均奖励、win‑rate 等指标上均表现优于或等同于现有 RL 方法，尤其在 LCO‑KLD 与 LCO‑LCH 上取得显著提升，并在样本效率上优于 PPO；在稀疏优势反馈与规则奖励模型下也保持稳定性。

**⚠️ 局限性**

局限性包括：需要事先估计优势（若估计噪声大会影响性能）；对大词表的全梯度计算成本较高；在非常大规模模型或复杂环境下的可扩展性尚未完全验证；以及 LCO 的超参数（β、学习率）对结果影响较大，需要进一步自动化调优。

---

## 807. S-VoCAL: A Dataset and Evaluation Framework for Inferring Speaking Voice Character Attributes in Literature

**arXiv ID:** 2603.00958 | [PDF](https://arxiv.org/pdf/2603.00958v1)

**作者:** Abigail Berthe-Pardo `[一作]`, Christophe Cerisara `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了S-VoCAL数据集和评估框架，用于从小说中推断与角色声音相关的属性。

**💡 创新点**

将音系学研究的八个声音相关属性纳入数据集，并为开放属性设计了基于Qwen3嵌入的语义相似度评估与加权/软F1指标，实现对不平衡属性的精细化评价。

**🔧 技术方法**

使用检索增强生成（RAG）结合E5-large进行语义检索、Qwen3-8B与Phi‑4 14B LLM进行属性推断，并利用Qwen3嵌入计算开放属性的相似度。

**📊 数据集**

收集192本公开书籍（Project Gutenberg）中的952个角色，属性来源于Wikidata并手工补全年龄（共359例），形成S-VoCAL数据集。

**📈 对比分析**

与多数值基线对比，闭合属性加权F1≥0.96，开放属性使用Human‑Aligned Score（HAS）最高约0.78（非人类类型），最低约0.15（身体健康），表明RAG在闭合属性上表现良好，而开放属性仍有显著提升空间。

**⚠️ 局限性**

数据集仅包含经典书籍，易被LLM记忆；属性覆盖不均衡且缺乏动态属性；评估高度依赖人工标注；开放属性推断仍表现不佳，且存在潜在结构偏差。

---

## 808. Minimalist Compliance Control

**arXiv ID:** 2603.00913 | [PDF](https://arxiv.org/pdf/2603.00913v1)

**作者:** Haochen Shi `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**通讯引用:** 24808 | [OpenAlex ID](https://openalex.org/A5004644695)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过仅利用现有伺服电机或准直驱动电机的电流/ PWM 信号，构建无力/扭矩传感器且无需学习的最简化顺应控制框架；

**💡 创新点**

创新点在于利用电机电流/ PWM 直接估计外部力矩，再通过雅克比投影得到外部力，结合任务空间阻抗/阻抗控制实现无感知的顺应控制，并且框架可跨多种机器人形态及高层规划器使用；

**🔧 技术方法**

核心技术包括电机力矩估计算法、基于雅克比的外部力矩/力估计、弹簧–质量–阻尼阻抗模型、逆运动学映射以及与 VLM、模仿学习和模型规划等高层策略的无缝对接；

**📊 数据集**

实验使用四台不同平台（ARX X5 机器人臂、Unitree G1、ToddlerBot、LEAP Hand）进行多任务测试（擦拭、绘图、拾取、旋转等），未使用公开数据集，而是基于真实交互任务收集数据；

**📈 对比分析**

与两种 RL 基线（UniFP、FACET）对比，所提出方法在位置跟踪误差、接触力控制以及任务成功率方面均优于基线，且实现了更低的根部俯仰角和更高的接触稳定性；

**⚠️ 局限性**

局限性包括未考虑非可背驱/自锁电机、显著摩擦或背隙对力矩估计的影响，以及未建模加速度、科氏力和热效应，适用于近似静态的低频接触场景。

---

## 809. Retrieval, Refinement, and Ranking for Text-to-Video Generation via Prompt Optimization and Test-Time Scaling

**arXiv ID:** 2603.01509 | [PDF](https://arxiv.org/pdf/2603.01509v1)

**作者:** Zillur Rahman `[一作]` (Algoverse AI), Cristian Meo `[通讯]` (Algoverse AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了3R框架，利用检索、细化与排序的推理时提示优化流程，提升文本到视频生成的质量。

**💡 创新点**

创新点在于：①完全不需要模型微调，直接在任意现成T2V模型上推理；②基于RAG提取修饰词的检索增强；③LLM细化与多候选生成，配合视频选择与时间插值，实现高质量、上下文一致的视频。

**🔧 技术方法**

核心技术包括：检索-增强-细化-排序（RAG、LLM）、黑盒T2V生成、视频选择模型、时间插值网络；使用预训练句子变换器与视觉语言模型做检索与提示生成。

**📊 数据集**

评估使用EvalCrafter基准，检索数据库用于生成修饰词；与LaVie、IPO、Show-1、Videocrafter2等公开基线模型进行对比。

**📈 对比分析**

在EvalCrafter的四大指标（文本-视频对齐、视觉质量、运动质量、时间一致性）上，3R总分245最高，分别在大部分单项指标上取得第一或第二名，比基线LaVie提升显著。

**⚠️ 局限性**

局限性包括：推理时延增加；视觉语言模型的批判容易过度校正或语义漂移，导致反馈瓶颈；需要更高效的采样与更稳健的评估机制。

---

## 810. LOGIGEN: Logic-Driven Generation of Verifiable Agentic Tasks

**arXiv ID:** 2603.00540 | [PDF](https://arxiv.org/pdf/2603.00540v1)

**作者:** Yucheng Zeng `[一作]` (Baidu Inc), Jianmin Wu `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LOGIGEN 框架，利用硬件编译的政策、逻辑驱动的正向合成和确定性状态验证，生成可验证、逻辑密集的训练数据，使大语言模型能够在状态化环境中完成复杂的策略驱动任务。

**💡 创新点**

核心创新在于：①将自然语言政策编译为数据库触发器，形成不可违背的硬性约束；②采用逻辑驱动的正向合成，避免逆向合成的“快乐路径”偏差；③通过状态差值进行严格验证，提供确定性奖励；并通过三代理协同（构架师、状态设计师、探索者）自动化生成任务包。

**🔧 技术方法**

技术手段包括：数据库触发器编译、边界相邻状态初始化、基于规则的工具接口、状态差距（State‑Diff）奖励、Turn‑aware GRPO 强化学习、验证式监督微调（Verified SFT）等。

**📊 数据集**

使用自研的 LOGIGEN 数据集，包含 20,000 个跨 8 领域的逻辑密集任务（每个任务包含政策、描述、初始/目标状态）以及 20,000 条已验证的轨迹；评估时采用 τ²‑Bench 作为标准基准。

**📈 对比分析**

在 τ²‑Bench 上与闭源模型、开源模型及其他代理训练框架进行对比，LOGIGEN‑32B(RL) 的成功率达 79.5%，显著高于基线模型（40.7%）并与主流专有模型竞争；SFT 与 RL 的组合进一步提升性能。

**⚠️ 局限性**

主要局限包括：对仿真用户（LLM 代理）可能出现的 “模拟器黑客” 现象导致过拟合；Pass@k 与 Passk 之间的能力‑一致性差距；目前仅限于关系数据库环境，缺乏更丰富的混合仿真或多服务后端；在复杂边界条件下的覆盖与探索效率仍待提升。

---

## 811. Dream2Learn: Structured Generative Dreaming for Continual Learning

**arXiv ID:** 2603.01935 | [PDF](https://arxiv.org/pdf/2603.01935v1)

**作者:** Salvatore Calcagno `[一作]` (University of Catania), Giovanni Bellitto `[通讯]` (University of Catania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Dream2Learn，利用冻结的扩散模型通过软提示优化生成“梦境”类，帮助连续学习模型重构表示空间并提升前向迁移；

**💡 创新点**

创新点在于将生成式梦境作为主动自我训练信号，而非仅重放过去数据，并通过软提示与 oracle 判别停止实现语义独立但结构连贯的梦境类；

**🔧 技术方法**

采用 Stable Diffusion（冻结扩散模型）+ 软提示优化 + oracle 网络 + 传统回放（ER-ACE、DER++ 等）相结合的技术栈；

**📊 数据集**

在 Mini-ImageNet、FG-ImageNet 与 ImageNet-R 这三大连续学习基准上进行实验；

**📈 对比分析**

与现有回放基准（ER-ACE、DER++、ER-ACE+WSCL 等）对比，D2L 在 FAA（最终平均准确率）上提升约10–20%，并实现正向迁移，整体性能超越 state‑of‑the‑art；

**⚠️ 局限性**

局限性包括可能泄露未来类别信息、相对较高的生成开销，以及对预训练扩散模型通用性的依赖导致潜在 OOV 偏差。

---

## 812. Hybrid Neural-LLM Pipeline for Morphological Glossing in Endangered Language Documentation: A Case Study of Jungar Tuvan

**arXiv ID:** 2603.00923 | [PDF](https://arxiv.org/pdf/2603.00923v1)

**作者:** Siyu Liang `[一作]` (University of Washington), Gina-Anne Levow `[通讯]` (University of Washington)

**通讯引用:** 1949 | [OpenAlex ID](https://openalex.org/A5031004017)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了将BiLSTM-CRF结构化预测与检索增强的大型语言模型（LLM）后处理相结合的两阶段混合管线，用于自动生成低资源形态学语言（Jungar Tuvan）的Interlinear Glossed Text（IGT）

**💡 创新点**

创新点包括提出混合架构、系统消融揭示检索增强提示、词典配置与few-shot规模的设计原则，并证明该方法在多种LLM上均能显著提升性能

**🔧 技术方法**

所用技术包括BiLSTM-CRF序列标注模型、四种通用LLM（如GPT-3/4等）、检索增强提示（RAG）、字典提示以及LLM后处理的提示工程

**📊 数据集**

使用了895句、含词形、词义和翻译的Jungar Tuvan IGT语料，训练集760句、测试集135句

**📈 对比分析**

通过token级准确率评估，BiLSTM基线为0.474；检索增强提示可提升至0.506；混合管线在不同LLM上达0.644–0.698，低shot场景提升约0.1–0.2，优于单一模型

**⚠️ 局限性**

限制包括仅在单一语言和单一形态系统上测试，未解决词形分割、跨语言泛化、提示设计系统化以及词典有效性问题，且样本量小导致统计稳定性有限

---

## 813. Specializing Foundation Models via Mixture of Low-Rank Experts for Comprehensive Head CT Analysis

**arXiv ID:** 2603.00675 | [PDF](https://arxiv.org/pdf/2603.00675v1)

**作者:** Youngjin Yoo `[一作]` (Siemens Healthineers), Eli Gibson `[通讯]` (Siemens Healthineers)

**通讯引用:** 5587 | [OpenAlex ID](https://openalex.org/A5025274219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出Mixture of Low‑Rank Experts (MoLRE)，在基础模型上实现条件化的低秩自适应，以提升头颅CT多标签诊断性能。

**💡 创新点**

创新点在于在LoRA基础上引入多专家路由和无监督软路由，能够针对不同病理特征进行条件化特征转化，仅增添不到0.5%的参数。

**🔧 技术方法**

使用技术包括低秩适配器、软路由器、注意力加权池化以及LoRA微调，结合多种2D/3D医学基础模型。

**📊 数据集**

数据集为70,000+无对比头颅CT扫描，标注75类神经学发现，采用GPT‑4‑mini自动生成标签并手工校验。

**📈 对比分析**

与传统LoRA、全微调等方法对比，MoLRE在所有模型上平均提升0.2–4.6%的AUC，MedGemma+MoLRE达最高0.917的平均AUC。

**⚠️ 局限性**

局限在于对3D体量模型提升有限，且无监督路由对极少数类的鲁棒性未知，需进一步验证跨机构泛化。

---

## 814. Neural Operator-Grounded Continuous Tensor Function Representation and Its Applications

**arXiv ID:** 2603.01812 | [PDF](https://arxiv.org/pdf/2603.01812v1)

**作者:** Ruoyang Su `[一作]` (University of Electronic Science and Technology of China), Michael K. Ng `[通讯]` (Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于神经算子的新型连续张量函数表示法NO-CTR，利用连续非线性模式‑n算子实现对多维数据的连续、非线性建模。

**💡 创新点**

创新点在于用神经算子替代传统离散线性模式‑n乘积，引入连续非线性模式‑n算子并证明其可逼近任意连续张量函数，从而打破离散线性限制。

**🔧 技术方法**

使用了神经算子（DeepONet）、SIREN隐式神经网络实现连续核心张量函数，以及DeepONet构造连续非线性模式‑n算子。

**📊 数据集**

实验数据集包括多光谱图像（MSI）、彩色视频、Sentinel‑2卫星图像、点云等不同分辨率与结构的数据。

**📈 对比分析**

与传统张量分解（TR‑ALS、CP、Tucker）、连续张量表示（SIREN、MFN、FR‑INR、LRTFR）以及其他神经算子方法进行对比，NO‑CTR在PSNR/SSIM/NRMSE/R²等指标上均取得最高或次高分，恢复质量显著优于对手。

**⚠️ 局限性**

限制包括模型参数量大、训练成本高以及对深度算子结构与超参敏感，且在低采样率下仍受数据稀疏影响。

---

## 815. Discrete World Models via Regularization

**arXiv ID:** 2603.01748 | [PDF](https://arxiv.org/pdf/2603.01748v1)

**作者:** Davide Bizzaro `[一作]` (University of Padua), Luciano Serafini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无重建、无对比的离散布尔世界模型学习方法 DWMR，能够在无监督环境下学习信息丰富、非坍塌的位编码和相应的状态转移函数。

**💡 创新点**

创新点在于为布尔表示设计了多维正则化：方差、相关性、共偏度以及局部性正则，显著防止码坍塌并鼓励动作仅翻转少数位，从而实现更好、更可解释的离散表征。

**🔧 技术方法**

使用了动作条件联合嵌入预测架构、指数移动平均目标网络、两步训练策略以及上述正则化损失，并在此基础上可选配自回归解码器。

**📊 数据集**

在 MNIST 8‑Puzzle 与 IceSlider 两个具有组合学结构的棋类/滑动块游戏数据集上进行实验。

**📈 对比分析**

通过线性探针评估每格 F1 分数，与 AE、β‑VAE、DeepCubeAI 等基线比较，DWMR 在编码和一跳模拟中的 F1 分数均显著高于对手，且在配合辅助解码器时进一步提升。

**⚠️ 局限性**

局限性包括未处理部分可观测性、连续动作、以及与强化学习或规划的整合；实验仅在有限的离散任务上验证，未来需扩展到更大、更复杂的环境。

---

## 816. GNN Based Joint Beamforming Design for Extremely Large-Scale RIS Assisted Near-Field ISAC Systems

**arXiv ID:** 2603.01379 | [PDF](https://arxiv.org/pdf/2603.01379v1)

**作者:** Jiahao Chen `[一作]` (Guangdong University of Technology), Vincent K. N. Lau `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16531 | [OpenAlex ID](https://openalex.org/A5073153992)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对极大规模RIS（XL‑RIS）协助的近场ISAC系统，提出联合波束成形设计，最大化所有通信用户的加权总速率（WSR），同时满足传输功率、目标感知增益和相位模数约束。

**💡 创新点**

创新点包括：①将FP+BCD方法与SCA、黎曼流形优化相结合得到局部最优解；②将近场ISAC系统建模为异构图（RIS、CU、TGT节点），利用图神经网络（GNN）实现端到端学习，显著提升对网络拓扑变化的泛化能力；③在GNN中设计无监督损失函数，实现对WSR和感知约束的联合优化。

**🔧 技术方法**

主要技术：分数规划（FP）、块坐标下降（BCD）、凸逼近（SCA）、黎曼流形优化、图神经网络（GNN）与消息传播机制、无监督深度学习、Adam优化器。

**📊 数据集**

使用仿真生成的CSI数据集：基于随机布置的多用户多目标位置，构造三维坐标、路径损耗、近场通道矩阵，未采用公开真实数据集。

**📈 对比分析**

比较方法：与传统B端点BCD优化、基于DNN的学习方案比较；在多种N（RIS元素数）和P0（发射功率）配置下评估运行时、WSR、约束满足率、鲁棒性与泛化性。实验显示：GNN实现近场ISAC的WSR最高、运行时最低、约束满足率>99%，并在CSI误差、用户/目标数变化时保持优良性能。

**⚠️ 局限性**

局限性：①模型假设为近场通道，远场环境未覆盖；②训练依赖大量仿真数据，对实际测量信道的迁移性仍需验证；③GNN结构仍需调参，层数过多易出现过平滑；④当RIS尺寸极大时，模型推理与参数规模仍会增长。

---

## 817. SubstratumGraphEnv: Reinforcement Learning Environment (RLE) for Modeling System Attack Paths

**arXiv ID:** 2603.01340 | [PDF](https://arxiv.org/pdf/2603.01340v1)

**作者:** Bahirah Adewunmi `[一作]` (University of Maryland), Sanjay Purushotham `[通讯]` (University of Maryland)

**通讯引用:** 4698 | [OpenAlex ID](https://openalex.org/A5017846156)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

建立了一个将Windows Sysmon日志转换为动态图并用于深度强化学习的环境，训练A2C模型识别攻击路径。

**💡 创新点**

首次通过将原始Sysmon事件映射为可用于RL的图结构，结合GCN与A2C，并为稀疏图设计奖励函数，实现了基于进程级日志的攻击路径学习。

**🔧 技术方法**

使用Gymnasium环境、PyTorch+torchrl、Graph Convolutional Network、Advantage Actor-Critic、SubstratumBridge等技术。

**📊 数据集**

使用公开的 BRAWL（Caldera攻击模拟）与 Cerberus Traces（真实Windows环境）Sysmon日志数据集。

**📈 对比分析**

通过多组超参数实验比较 BRAWL 与 Cerberus 环境的损失与奖励稳定性，Cerberus表现稳定、低方差的价值损失；BRAWL由于稀疏结构导致高方差与极大损失，表明环境可用但对稀疏图挑战大。

**⚠️ 局限性**

受稀疏图导致训练不稳定、内存瓶颈、对大规模日志扩展有限，奖励设计与模型参数仍需进一步优化。

---

## 818. From Dialogue to Execution: Mixture-of-Agents Assisted Interactive Planning for Behavior Tree-Based Long-Horizon Robot Execution

**arXiv ID:** 2603.01113 | [PDF](https://arxiv.org/pdf/2603.01113v1)

**作者:** Kanata Suzuki `[一作]` (Waseda University), Tetsuya Ogata `[通讯]` (National Institute of Informatics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种将混合代理（Mixture-of-Agents, MoA）与交互式LLM规划相结合的框架，利用行为树（Behavior Trees, BT）生成可执行的长期任务计划，并在真实机器人上实现动态模型切换与重试机制。

**💡 创新点**

创新点在于：1）通过MoA代理在规划对话中自动回答可推理的提问，显著降低人类交互成本；2）采用BT结构化长程任务计划，支持层次化条件与重试；3）在BT节点上动态分配不同的模仿学习模型（π_0.5、Diffusion Policy），实现多模型协同执行。

**🔧 技术方法**

核心技术包括：大型语言模型（Gemini 2.0 Flash）用于BT生成与不确定性分析；多专家代理（Robot Expert、Task Domain Expert、Commonsense Expert）实现代理回答；行为树（XML格式）用于任务结构化；视觉语言模型（VLM）用于动作成功判定；模仿学习模型π_0.5和Diffusion Policy用于动作执行。

**📊 数据集**

使用的任务数据集为两类：1）鸡尾酒制作任务（“Make a Margarita”）的交互式问答日志；2）顺滑饮制作任务的机器人演示数据，分别包含水果插入（200条/种）和盖子、开关等动作（300条）收集的演示序列。

**📈 对比分析**

比较方法：1）结构相似度采用归一化树编辑距离（TED）；2）语义相似度采用节点的Sentence‑BERT嵌入余弦相似度；3）执行性能用动作成功率与真实机器人实验中的任务完成率衡量。实验表明，MoA可将人类回答比例降低约27%，BT的结构与语义相似度与完全人工回答的BT相近；在真实机器人上实现顺滑饮制作时，使用动态模型切换和BT重试结构，任务成功率高于单一模型，并能在失败后恢复执行。

**⚠️ 局限性**

局限性包括：1）代理回答仍需人工确认，完全自动化尚未实现；2）子任务的模仿学习性能受限于训练数据量，跨任务迁移效果不佳；3）框架在更大规模或多领域任务中的泛化和代理选择机制仍需进一步研究。

---

## 819. Implementation of Licensed Plate Detection and Noise Removal in Image Processing

**arXiv ID:** 2603.01016 | [PDF](https://arxiv.org/pdf/2603.01016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 820. MultiPUFFIN: A Multimodal Domain-Constrained Foundation Model for Molecular Property Prediction of Small Molecules

**arXiv ID:** 2603.00857 | [PDF](https://arxiv.org/pdf/2603.00857v1)

**作者:** Idelfonso B. R. Nogueira `[一作]`, Erick Giovani Sperandio Nascimento `[通讯]` (University of Surrey)

**通讯引用:** 1498 | [OpenAlex ID](https://openalex.org/A5004611294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态领域约束基础模型 MultiPUFFIN，用于小分子在多种热物理属性上的预测，融合 SMILES、分子图和三维构象以及实验条件和分子描述符，采用多任务学习并在输出层嵌入物理公式。

**💡 创新点**

创新点包括：① 端到端融合三种结构模态（文本、图、空间）并引入交叉注意力与门控融合；② 在每个属性预测头中嵌入对应的热物理方程（Wagner、Andrade、van 't Hoff 等），实现热力学一致性；③ 两阶段训练策略（联合多任务预训练 + backbone‑frozen 头微调）和余弦热重启调度；④ 构建覆盖 9 项属性、37968 分子的多源数据集，实现 59% 覆盖率提升。

**🔧 技术方法**

技术手段包括 GCN、Transformer、SchNet 三模态编码器；交叉模态注意力与门控融合；领域诱导的输出层（inductive bias neurons）；多任务学习与不确定性加权损失；SMILES 枚举增强；余弦 warm‑restart 学习率调度；两阶段训练与冻结策略。

**📊 数据集**

使用 9 个公开数据库构建的 37968 个分子（40904 条记录）数据集：OPERA、NIST ThermoML、ECHA REACH、ChEMBL、AqSolDB、FreeSolv、Bradley、Sun et al.、ABB‑ADD、PubChem 等，涵盖 9 个热物理属性（蒸气压、黏度、溶解度、沸点、熔点、闪点、比热容、溶剂化自由能、分配系数）。

**📈 对比分析**

与 ChemBERTa‑2（77 M SMILES 预训练）在 9 项属性上进行对比，MultiPUFFIN 在 8877 个 scaffold‑split 测试集上平均 R² = 0.716，RMSE 约 0.7–1.8，显著优于 ChemBERTa‑2，尤其在温度依赖属性上误差降低 5–10 倍；同时在 2000× 更少的数据量下实现更高的精度。

**⚠️ 局限性**

局限性包括：① 共享 512 维嵌入导致多任务容量分散，对复杂属性如熔点、闪点的预测仍受限；② 3D 构象仅为单一构象，无法完全捕捉柔性分子构象空间；③ 对缺失模态（如 3D、实验条件）的鲁棒性仍有提升空间；④ 需要更大规模多模态训练数据以进一步提升性能。

---

## 821. Dial E for Ethical Enforcement: institutional VETO power as a governance primitive

**arXiv ID:** 2603.00617 | [PDF](https://arxiv.org/pdf/2603.00617v1)

**作者:** Subramanyam Sahoo `[一作]` (Independent), Divya Chaudhary `[通讯]` (Northeastern University)

**通讯引用:** 5152 | [OpenAlex ID](https://openalex.org/A5048878908)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨在大规模推理模型军事化过程中缺乏可执行的否决机制，并提出嵌入社区领导的治理框架。

**💡 创新点**

提出“机构否决权”作为治理原语，强调社区主导、可执行的拒绝权。

**🔧 技术方法**

无技术实现，仅采用治理理论与案例分析。

**📊 数据集**

无数据集，依托核不扩散、医学伦理等先例。

**📈 对比分析**

无实验比较，文献综述与案例绘图说明。

**⚠️ 局限性**

缺乏具体实施案例、可能被捕捉、无法解决更深层次权力不平衡。

---

## 822. Catapults to the Rescue: Accelerating Vector Search by Exploiting Query Locality

**arXiv ID:** 2603.02164 | [PDF](https://arxiv.org/pdf/2603.02164v1)

**作者:** Sami Abuzakuk `[一作]` (École Polytechnique Fédérale de Lausanne), Martijn de Vos `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Catapults，动态注入快捷边到图索引以利用查询局部性，显著提升 ANN 搜索吞吐量。

**💡 创新点**

首次在运行时根据查询轨迹重组图边，构成透明层，可叠加任何基于图的 ANN 索引，同时兼容过滤、动态插入，无需重建。

**🔧 技术方法**

采用随机超平面 LSH 定位查询区域，LRU 溢出策略维护桶，读写锁实现多线程安全，并在 DiskANN 上实现该机制。

**📊 数据集**

在 TripClick 医疗搜索日志嵌入、LLM 生成的相似查询、Uniform 随机向量以及 arXiv Papers 论文摘要向量四个数据集上进行评估。

**📈 对比分析**

与基线 DiskANN 及 LSH-APG 在四种工作负载下对比，Catapults 在高局部性工作负载上实现最高 2.51× 的 QPS 提升，节点访问减少 66%，召回保持甚至提升，过滤查询下提升 38%，在无局部性工作负载下无显著吞吐退化。

**⚠️ 局限性**

在无局部性查询时吞吐略降；需要 LSH 参数（L 与桶容量 b）的调优；极端查询分布可能导致大桶稀疏或局部性不足时效果有限。

---

## 823. MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention

**arXiv ID:** 2603.01361 | [PDF](https://arxiv.org/pdf/2603.01361v1)

**作者:** Zilong Zhao `[一作]` (Shandong University), Feng Guo `[通讯]` (Shandong University)

**通讯引用:** 9574 | [OpenAlex ID](https://openalex.org/A5027245585)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为MixerCSeg的混合架构，用于道路裂纹像素级分割；

**💡 创新点**

创新点包括：1) TransMixer模块基于Mamba隐式注意力将特征拆分为全局和局部通道，分别使用自注意力和局部细化，天然融合CNN、Transformer与Mamba的优势；2) Direction‑guided Edge Gated Convolution (DEGConv)引入方向先验和门控机制，显著提升裂纹边缘感知；3) Spatial Refinement Multi‑Level Fusion (SRF)模块在不增加复杂度的情况下利用高分辨率特征细化低分辨率特征；

**🔧 技术方法**

核心技术包括Mamba状态机 (SSM)、Transformer自注意力、CNN卷积、DEGConv方向先验与门控、SRF跨尺度融合；

**📊 数据集**

在四个裂纹基准集 DeepCrack、Crack500、CamCrack789、CrackMap 上训练与测试；

**📈 对比分析**

与7种SOTA模型（U‑Net、CarNet、RINDNet、DTrCNet、RestorMixer、SCSegamba、MambaVision）在同一输入尺寸和训练轮数下比较，MixerCSeg 在所有数据集上均取得最高 mIoU、F1、ODS、OIS；在 DeepCrack 上 mIoU 提升约1.4%（SCSegamba）及 1.78%（MambaVision），参数仅 2.54M，GFLOPs 2.05，显著低于其它混合模型；

**⚠️ 局限性**

局限性：1) 对极端纹理噪声或极细裂纹的鲁棒性仍有提升空间；2) 采用的方向先验与细化机制在计算量上仍有限，可能在更大尺寸图像上需要进一步优化；3) 仅在道路裂纹数据上验证，跨域应用（如桥梁、墙体等）的泛化能力未充分评估。

---

## 824. Never Saddle for Reparameterized Steepest Descent as Mirror Flow

**arXiv ID:** 2603.02064 | [PDF](https://arxiv.org/pdf/2603.02064v1)

**作者:** Tom Jacobs `[一作]` (CISPA Helmholtz Center for Information Security), Rebekka Burkholz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并研究了一个统一的“steepest mirror flow”框架，用以分析包括Adam在内的 steepest descent 优化器在深度网络、特别是深度对角线重参数化下的学习动态、特征学习和梯度消失问题；

**💡 创新点**

1) 通过梯度重参数化将传统梯度流映射到 Banach 空间的 mirror flow；2) 揭示了 q 参数（梯度范数）对 saddle 逃逸速率和特征稀疏化的决定性影响；3) 证明了 Adam 与 AdamW 在 decoupled weight decay 下的不同隐式正则化；

**🔧 技术方法**

Steepest mirror flow 理论、L_p 归一化的 steepest descent、对角线重参数化、Bregman 散度与 Legendre 函数、解析的 balance 方程、实验验证用梯度下降、Adam、AdamW

**📊 数据集**

线性回归、二分类（指数损失）、CIFAR‑10、ImageNet、ViT‑large、Bert‑base、Flowers 数据集

**📈 对比分析**

在实验中对比了 SGD、Adam、AdamW 的学习曲线、特征恢复率、特征稀疏化、Hessian 负特征值分布；Adam/AdamW 在 fine‑tuning 下明显优于 SGD，尤其在小学习率和稀疏重参数化时更易逃逸 saddle 并获得更高精度

**⚠️ 局限性**

仅针对对角线重参数化的解析，未证明对非对角线或更复杂网络的泛化；对实际随机梯度噪声的理论分析不足；实验规模相对有限，未覆盖全部大模型场景

---

## 825. Markovian ODE-guided scoring can assess the quality of offline reasoning traces in language models

**arXiv ID:** 2603.01580 | [PDF](https://arxiv.org/pdf/2603.01580v1)

**作者:** Arghodeep Nandi `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 5059 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于马尔可夫链和ODE的后验推理链质量评价框架MarODE，用于评估LLM生成的多步推理轨迹的连贯性、质量和证据对齐。

**💡 创新点**

创新点在于将马尔可夫链测度局部连贯性、连续时间ODE建模全局方向性与冗余、以及证据对齐三大维度统一成一个可解释的分数，并通过人类中心化扰动验证其鲁棒性。

**🔧 技术方法**

技术包括：随机游走马尔可夫链、自然语言推理(NLI)概率、余弦相似度、词集冗余惩罚、ODE连续动力学（Runge–Kutta积分）以及多维度加权融合。

**📊 数据集**

数据集涵盖生成的LIAR与PolitiFact事实检验推理链、以及EntailmentBank、ProofWriter、GSM8K和StrategyQA四大人工评估推理链，用于扰动测试和人类标注评估。

**📈 对比分析**

与ROSCOE、ReCEval、Local/Global Coherence、LLM-as-a-Judge等基线相比，MarODE在人为扰动和专家评估的Somers' D相关性上提升约235–279%，在四个真实评估集上的显著性最高且分布更平滑，证明其更高的敏感度和可靠性。

**⚠️ 局限性**

局限性包括：对证据对齐的贡献有限，可能在证据不完整或歧义的场景下效果下降；重度依赖外部NLI与句向量模型，若这些模型性能不足会影响评分；且评估仍为离线，未覆盖生成时的即时反馈。

---

## 826. Beyond the Flat Sequence: Hierarchical and Preference-Aware Generative Recommendations

**arXiv ID:** 2603.00980 | [PDF](https://arxiv.org/pdf/2603.00980v1)

**作者:** Zerui Chen `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 16377 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种两阶段的层次化、偏好感知生成推荐框架 HPGR，利用结构感知预训练和稀疏注意力实现对用户行为层次结构和偏好信息的有效建模。

**💡 创新点**

创新点：①设计了 Session Enhancement Module (SEM) 通过双层 Transformer 明确捕获会话内外的时间层次；②提出 Preference‑Guided Sparse Attention (PGSA)，在预测时动态选取最相关的历史子集并使用高斯掩码聚焦注意力；③引入时间感知位置编码保证候选物品在序列中的正确时序。

**🔧 技术方法**

使用技术：Transformer‑based SEM、Masked Item Modeling (MIM) 预训练、HSTU 编码器、PGSA、时间感知位置编码、二分类交叉熵损失和微调学习率衰减等。

**📊 数据集**

数据集：APP Gallery 真实工业数据，约 45.8M 用户、200k 商品、285M 训练交互、7.38M 测试交互；实验同时在 2.5% 采样子集和完整数据集上验证。

**📈 对比分析**

比较方法：在离线 AUC 与在线 eCPM 上与 discriminative（DNN, MoE, GRU4Rec, SASRec, BERT4Rec, Wukong）和 generative（HSTU, MTGR）基线对比。HPGR Full 在采样子集上 AUC 0.8377，较 MTGR 提升 1.5%；在完整数据上 AUC 0.8929，提升 1.14%；在线 A/B 测试中 eCPM 提升 1.99%。

**⚠️ 局限性**

限制：模型相较 MTGR 训练与推理时间略增，计算成本升高；仅在 APP Gallery 领域验证，需进一步在多行业场景测试；缺乏对解释性和可视化的深入分析。

---

## 827. Risk-Aware Skill-Coverage Hybrid Workforce Configuration on Social Networks

**arXiv ID:** 2603.00727 | [PDF](https://arxiv.org/pdf/2603.00727v1)

**作者:** Hui-Ju Hung `[一作]` (National Central University), De-Nian Yang `[通讯]` (Academia Sinica)

**通讯引用:** 2897 | [OpenAlex ID](https://openalex.org/A5029978968)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在双层社交网络（接触网络和合作网络）上，兼顾技能覆盖与传染风险的混合工作配置问题，并设计了一种名为 Guided Risk‑aware Iterative Assembling (GRIA) 的多阶段算法。

**💡 创新点**

创新点在于将感染传播风险与员工技能与协作收益统一建模，并通过风险感知构建、技能保留精炼和风险降低替换三阶段迭代策略，首次实现了在满足风险预算下最大化现场协作收益的可扩展求解。

**🔧 技术方法**

技术包括：两层网络建模（G_c 及 G_p）、基于影响概率的传播风险评估、协作收益度量 τ、贪婪选择与局部交换的迭代算法；此外证明问题 NP‑hard 并给出理论保证。

**📊 数据集**

使用了四个公开真实网络数据集：Manhattan、Virginia、ca‑GrQc 和 ca‑HepPh，分别包含数千至数十万节点，验证了算法在大规模图上的可行性。

**📈 对比分析**

与六类基线（密集子图/社群搜索、团队组建、疫情感知推荐等）相比，GRIA 在所有数据集上均取得最高的平均协作得分（目标比 1.00），且在计算时间上处于中等水平，仅略高于纯结构基线，但显著优于其他混合策略。

**⚠️ 局限性**

局限性包括：算法复杂度为 O(|V|²·c_r) 仍对极大规模网络带来压力；风险模型仅考虑静态接触网络，未能捕捉动态传播或多源风险；以及假设远程协作得分与现场相对固定，实际场景中可能变化。

---

## 828. Efficient Long-Sequence Diffusion Modeling for Symbolic Music Generation

**arXiv ID:** 2603.00576 | [PDF](https://arxiv.org/pdf/2603.00576v1)

**作者:** Jinhan Xu `[一作]` (Wuhan University of Technology), Guangli Xiang `[通讯]` (Wuhan University of Technology)

**通讯引用:** 364 | [OpenAlex ID](https://openalex.org/A5004457967)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 SMDIM 的符号音乐生成框架，采用扩散模型与结构状态空间模型相结合，能够高效处理长序列符号音乐并保持全局结构与局部细节的平衡。

**💡 创新点**

创新点包括：① 在扩散过程中引入混合 MFA 块（Mamba + FeedForward + Self‑Attention），先用线性时间的 SSM 捕获长程依赖，再通过轻量级自注意力精细化局部细节；② 将扩散步骤重新设计为全局–局部双阶段，显著降低每步计算和显存需求；③ 通过吸收状态的离散扩散策略实现更稳定的长序列恢复。

**🔧 技术方法**

核心技术包括：离散扩散概率模型（D3PM）与吸收状态，结构状态空间模型 Mamba，MFA 块中的前馈网络和自注意力，整体实现基于 PyTorch，采用余弦学习率调度等。

**📊 数据集**

使用了三大公开数据集：MAESTRO（古典钢琴）、POP909（中国流行钢琴）以及新构建的 FolkDB（传统中国民乐），覆盖不同风格与调式，验证模型的通用性。

**📈 对比分析**

通过与 MusicTr、PolyDiff、SCHmUBERT、GETMusic、MusicMamba 等基线在相同数据集、相同评估指标（OA、主观情感/流畅/节奏/结构清晰度）下进行比较。SMDIM 在 OA 上普遍居首（或次首），在主观评分中明显优于同类模型，且显存占用仅 21 GB、单步推理时间 0.35 s（相较于 SCHmUBERT 的 35 GB/0.54 s），在长序列（2048 码）下保持性能稳定。

**⚠️ 局限性**

局限性包括：① 在极端高低音区出现不合拍的音符；② 生成过度稠密的和弦音块；③ 随着生成长度增大，后段的整体结构连贯性逐步衰退，易出现主题漂移。未来需加入音高范围、和声稀疏性及更高层次结构约束来提升鲁棒性。

---

## 829. K^2-Agent: Co-Evolving Know-What and Know-How for Hierarchical Mobile Device Control

**arXiv ID:** 2603.00676 | [PDF](https://arxiv.org/pdf/2603.00676v1)

**作者:** Zhe Wu `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**通讯引用:** 5335 | [OpenAlex ID](https://openalex.org/A5057896400)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个层次化的移动设备控制框架 K²-Agent，采用高层规划器通过 SRLR 循环自我演化语义知识，低层执行器通过 C‑GRPO 训练实现精细动作控制。

**💡 创新点**

创新点在于：①将认知中的“知何物”(declarative)与“如何做”(procedural)两种知识分离并在同一闭环中共演化；②SRLR 迭代循环实现单演示自我完善；③C‑GRPO 通过错误解耦重放平衡和动态示范注入显著提升采样效率与探索能力。

**🔧 技术方法**

技术包括：VLM 基础模型（Qwen‑2.5‑VL‑72B / 7B）、Summarize–Reflect–Locate–Revise (SRLR) 自我学习循环、Curriculum‑Guided Group Relative Policy Optimization (C‑GRPO)、自适应演示注入、错误解耦经验池。

**📊 数据集**

主要使用数据集：AndroidWorld（116 任务）、ScreenSpot‑v2（多平台 UI 交互）、Android‑in‑the‑Wild（AitW），并在这些数据上进行单示范启动和跨模型迁移实验。

**📈 对比分析**

与多种训练自由和训练基方法相比，K²-Agent 在 AndroidWorld 取得 76.1% 的最高成功率（超越 GPT‑5、AutoGLM‑Mobile 等），在 ScreenSpot‑v2 和 AitW 的零样本迁移中分别实现 91.3% 和 86.5% 的显著提升，证明其优越的双向泛化能力。

**⚠️ 局限性**

局限性：①依赖大规模 VLM 作为起点，模型规模与算力需求仍高；②目前仅使用原始截图输入，缺乏更丰富的 UI 元素信息；③对极端长程或极其稀有 UI 操作的泛化仍有限，需进一步完善错误解耦与探索策略。

---

## 830. Opportunities and Challenges of Operating Semi-Autonomous Vehicles: A Layered Vulnerability Perspective

**arXiv ID:** 2603.01202 | [PDF](https://arxiv.org/pdf/2603.01202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 831. Content-Aware Frequency Encoding for Implicit Neural Representations with Fourier-Chebyshev Features

**arXiv ID:** 2603.01028 | [PDF](https://arxiv.org/pdf/2603.01028v1)

**作者:** Junbo Ke `[一作]` (Hunan Normal University), Chao Wang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 104228 | [OpenAlex ID](https://openalex.org/A5100339418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种内容感知的频率编码框架CAFE及其改进版本CAFE+，用于提升隐式神经表示（INR）的高频细节重建能力。

**💡 创新点**

创新点在于通过多条并行线性层与Hadamard乘积实现对固定Fourier基的动态组合，显著扩大可合成频率空间；并进一步加入Chebyshev多项式基以稳健捕捉低频信息。

**🔧 技术方法**

主要技术包括随机/位置编码的Fourier特征、并行线性层与Hadamard乘积的频率混合、Chebyshev多项式特征、以及标准多层感知机（MLP）作为后端。

**📊 数据集**

在DIV2K、3DScanRep、Blender等公开数据集上进行实验，分别针对二维图像拟合、三维形状重建和神经辐射场（NeRF）视图合成。

**📈 对比分析**

与SIREN、WIRE、FINER、SCONE、SL^2A、Gauss等现有方法比较，CAFE/CAFE+在PSNR、IoU和训练时间等指标上均取得显著提升，特别是在高频细节保留和噪声抑制方面表现突出。

**⚠️ 局限性**

主要局限是对大规模高维输入（如长序列或高分辨率3D体素）仍需进一步优化编码层数和参数规模，以避免计算开销过大；同时目前尚未充分探索与更高级激活函数或其它频率基的融合潜力。

---

## 832. Clawdrain: Exploiting Tool-Calling Chains for Stealthy Token Exhaustion in OpenClaw Agents

**arXiv ID:** 2603.00902 | [PDF](https://arxiv.org/pdf/2603.00902v1)

**作者:** Ben Dong `[一作]` (University of California, Merced), Qian Wang `[通讯]` (University of California, Merced)

**通讯引用:** 4010 | [OpenAlex ID](https://openalex.org/A5100391044)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造并在真实 OpenClaw 生产环境中部署了 Trojanized 技能 Clawdrain，利用多轮分段验证协议（SVP）诱导 LLM 代理产生大量 token 消耗，同时仍返回正确答案；

**💡 创新点**

首次系统展示了在真实代理中出现的工具组合自救、失败回溯以及不同接口下的可见性差异，并证明了仅靠输出 token 扩增的评估方式低估了攻击成本与隐蔽性；

**🔧 技术方法**

结合 OpenClaw 框架的技能注入、Gemini 2.5 Pro 语言模型、shell/Python 工具调用、以及自定义的多轮验证协议脚本；

**📊 数据集**

使用基于 BBC 头条的查询任务作为目标任务，实验数据来自 OpenClaw 自带的会话状态报告（输入/输出 token 计数）和实际运行时间；

**📈 对比分析**

通过对比四种实验设置（基线、SVP v1/v2/v3）得到 6–9 倍 token 扩增，且在失效情况下 9 倍扩增超过成功情况，证明失败路径可导致更高成本；

**⚠️ 局限性**

实验局限于单一生产模型（Gemini 2.5 Pro）和有限的技能配置，未对不同模型、技能规模或长期自驱动攻击进行系统化评估，且未量化成本与预算阈值等实际财务影响。

---

## 833. Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding

**arXiv ID:** 2603.01816 | [PDF](https://arxiv.org/pdf/2603.01816v1)

**作者:** Zhiyuan Zhou `[一作]` (Hefei University of Technology), Shijie Hao `[通讯]` (Hefei University of Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出情感–认知协作多模态生成（ECMC）任务，利用自然语言描述多模态数据中的情感与认知状态，生成可解释的情感认知剖面，辅助精神健康评估。

**💡 创新点**

创新点在于将情感与认知特征独立抽取并通过双桥 Q-former 网络融合，同时使用情感与认知的对比学习提升语义区分，最终由 LLaMA 解码器生成解释性文本。

**🔧 技术方法**

技术细节包括：视频MAE、HuBERT、BERT 的多模态预训练编码器；双桥 Q-former 进行情感/认知特征提取；对比学习（情感负、正、中性；认知多标签 Jaccard 对比）；LLaMA 解码器；两阶段训练策略。

**📊 数据集**

使用 MMDA 大规模多模态精神疾病数据集（1,025 受试者），并在其 30,592 条情感–认知字幕对上训练与评估。

**📈 对比分析**

与多模态 LLM（InternVL、Sa2VA、Qwen2.5‑Omni）及精神健康专用 LLM（PsycoLLM、MindChat、EmoLLM、CPsyCoun）对比，ECMC 在 BLEU、METEOR、CIDEr、F_BERT 等指标上持续领先；在利用生成的情感认知剖面提升抑郁/焦虑检测时，ACC 与 F1 均提升约 12–15% 以上。

**⚠️ 局限性**

局限在于认知描述准确性略低，需进一步提升多模态认知建模；模型规模大、训练成本高；缺乏在真实临床环境中的验证与外部泛化评估。

---

## 834. What Helps -- and What Hurts: Bidirectional Explanations for Vision Transformers

**arXiv ID:** 2603.01605 | [PDF](https://arxiv.org/pdf/2603.01605v1)

**作者:** Qin Su `[一作]` (University of Kentucky), Tie Luo `[通讯]` (University of Kentucky)

**通讯引用:** 2560 | [OpenAlex ID](https://openalex.org/A5049199248)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 BiCAM，一种双向类激活映射方法，可同时捕捉支持性和抑制性对 ViT 预测的贡献；

**💡 创新点**

创新点在于保留负梯度信息生成对比性解释，并引入正负比率 PNR 用于轻量级对抗样本检测；

**🔧 技术方法**

使用 ViT 结构的多头自注意力、梯度回传、温度归一化、层级聚合和简单求和实现；

**📊 数据集**

在 ImageNet、VOC2012、COCO 等公开数据集上进行评估，并对抗生成的 PGD/C&W/MI-FGSM 进行检测；

**📈 对比分析**

与 Attention Rollout、LRP、AGCAM、ViT-Shapley 等方法对比，BiCAM 在定位准确率、可信度与计算效率方面均表现优异；

**⚠️ 局限性**

局限包括对梯度噪声敏感、缺乏用户研究验证以及对抗检测尚未与专门对抗检测器直接比较，且未测试自适应攻击。

---

## 835. Adaptive Dynamic Dehazing via Instruction-Driven and Task-Feedback Closed-Loop Optimization for Diverse Downstream Task Adaptation

**arXiv ID:** 2603.00542 | [PDF](https://arxiv.org/pdf/2603.00542v1)

**作者:** Yafei Zhang `[一作]` (Kunming University of Science and Technology), Yu Liu `[通讯]` (Hefei University of Technology)

**通讯引用:** 47130 | [OpenAlex ID](https://openalex.org/A5022072267)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种闭环自适应去雾框架，能够在推理阶段根据下游任务性能反馈和文本指令实时调整去雾结果。

**💡 创新点**

创新点在于双重闭环机制：任务反馈引导的 TFGA 模块与文本指令引导的 IGM 模块共同实现无重训练、无任务特定微调的实时任务感知去雾。

**🔧 技术方法**

技术包括基于 Transformer 的初始去雾网络 (IDN)，TFGA 交叉注意力+通道融合，IGM 基于 BERT 的文本特征提取与权重生成，以及特征融合模块 (FFM) 和多级对比损失。

**📊 数据集**

使用 ADE20K、COCO、KITTI 三个公开数据集，合成雾图像作为训练集，原始测试集用于评估。

**📈 对比分析**

与 8 种前沿方法在语义分割、目标检测和深度估计等多任务下进行对比，取得了在 PSNR/SSIM/LPIPS 以及 mIoU、mAP、深度误差指标上的最高或相近表现，验证了方法的优越性。

**⚠️ 局限性**

局限性：仅在固定的三大任务集上验证；未评估对新任务或任务动态变化的适应性；对真实多种雾景的鲁棒性仍待进一步研究。

---

## 836. From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing

**arXiv ID:** 2603.01038 | [PDF](https://arxiv.org/pdf/2603.01038v1)

**作者:** Haoyuan Zhang `[一作]` (University of Chinese Academy of Sciences), Zhen Lei `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 26936 | [OpenAlex ID](https://openalex.org/A5109299788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Tool-Augmented Reasoning FAS (TAR-FAS) 框架，将人脸抗欺骗任务重新定义为带视觉工具的链式思考（CoT-VT），并构建了 ToolFAS-16K 数据集以支持多轮工具调用推理。

**💡 创新点**

创新点包括：①将 FAS 与多模大语言模型 (MLLM) 结合，引入可调用的视觉工具实现细粒度特征检索；②构造专门的多轮工具使用注释管线和 ToolFAS-16K 数据集；③提出 Diverse-Tool Group Relative Policy Optimization (DT‑GRPO) 以让模型自发学习高效工具使用；④通过专家模型引导提升工具使用的可靠性。

**🔧 技术方法**

技术方法：InternVL‑3‑8B / Qwen2.5‑VL 等 MLLM，LoRA 微调与全参数 Fine‑Tuning，工具调用格式注入，DT‑GRPO 强化学习，专家模型指导的注释流程，外部视觉工具（LBP、FFT、HOG、边缘检测、放大）等。

**📊 数据集**

使用的数据集：CelebA‑Spoof 作为源域；11 个目标域（CASIA‑MFSD、CASIA‑SURF‑3DMask、HKBU‑MARs‑V1+、HiFiMask、MSU‑MFSD、OULU‑NPU、REPLAY‑ATTACK、Rose‑Youtu、SIW、SIW‑M‑V2、WMCA）进行跨域评测；同时构造 ToolFAS‑16K（16k 条多轮工具使用轨迹）用于训练与验证。

**📈 对比分析**

与 ViTAF、ViT‑L、FLIP、I‑FAS 在一对十一交叉域协议下进行对比，TAR‑FAS 在 HTER 7.54% / AUC 96.67% 方面实现了最高性能；消融实验表明工具多样性、格式注入和 DT‑GRPO 共同提升模型鲁棒性。

**⚠️ 局限性**

局限性：依赖预定义的视觉工具，可能对未知或极端攻击类型的适应性有限；ToolFAS‑16K 规模相对有限，难以覆盖所有欺骗模式；多轮工具调用可能增加推理时延，且解释性仍需进一步完善。

---

## 837. ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs

**arXiv ID:** 2603.01792 | [PDF](https://arxiv.org/pdf/2603.01792v1)

**作者:** Xunlei Chen `[一作]` (University of Electronic Science and Technology of China), Wenhong Tian `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ALTER 框架，用于在大型语言模型中实现高效且可解释的知识忘记。

**💡 创新点**

创新点在于将 Asymmetric LoRA 结构与基于 token 熵的分层分配相结合，实现了子域级参数隔离与 token 级精准忘记，并引入 Tsallis 熵构建动态忘记边界。

**🔧 技术方法**

主要技术包括 Asymmetric LoRA、基于 Shannon 与 Tsallis 熵的 token 级路由、三层级损失（遗忘、保留与结构保持）以及 MoE 路由器。

**📊 数据集**

使用了 TOFU、WMDP、MUSE‑HarryPotter 三个多域忘记基准以及 MMLU 用于评估保留能力。

**📈 对比分析**

与梯度上升、KL 最小化、NPO 等传统忘记方法以及 LoRA、AsymLoRA 等基线对比，ALTER 在遗忘质量（>95%）、保留性能（>90%）和流畅度方面均超过基线，且训练时间降低约 86%。

**⚠️ 局限性**

限制在于实验仅覆盖了少数模型（Llama2‑7B、Llama3‑8B、Zephyr‑7B）和特定子域，且对极端大规模连续忘记或跨模型迁移的鲁棒性尚未充分验证。

---

## 838. From Secure Agentic AI to Secure Agentic Web: Challenges, Threats, and Future Directions

**arXiv ID:** 2603.01564 | [PDF](https://arxiv.org/pdf/2603.01564v1)

**作者:** Zhihang Deng `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18242 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了大语言模型驱动的自主代理系统在安全性方面的威胁与防御，提出了从单体代理到Agentic Web的转变视角，并对威胁分类、攻击手段与防御策略进行组件对齐与系统级阐述。

**💡 创新点**

创新点在于构建了面向Agentic Web的“转变导向”框架，将传统的代理安全问题与多代理、跨域交互、协议层面风险关联起来，提出了身份授权、可追溯性与生态级响应三大核心原语。

**🔧 技术方法**

主要技术包括威胁与防御的层级化分类、案例与基准分析、协议安全设计（如MCP）、以及对攻击与防御在Agentic Web中的扩散机制进行理论映射。

**📊 数据集**

使用了多种公开基准与案例（如PromptInjectionBench、AgentSecurityBench、OpenAgentSafety等）来支撑威胁评估与防御效果的讨论。

**📈 对比分析**

通过对基准和真实部署场景的比较，指出单一防御手段难以覆盖全部攻击，强调多层防御（提示硬化、模型鲁棒、工具控制、运行时监控、持续红队和协议安全）是实现安全的必要条件。

**⚠️ 局限性**

局限性包括：仅为综述性质，缺乏统一实验验证；对Agentic Web的具体实现细节与可量化指标探讨不足；以及对快速演进的攻击技术和跨域政策冲突的系统化评估尚不充分。

---

## 839. VidDoS: Universal Denial-of-Service Attack on Video-based Large Language Models

**arXiv ID:** 2603.01454 | [PDF](https://arxiv.org/pdf/2603.01454v1)

**作者:** Duoxun Tang `[一作]` (Shenzhen International Graduate School, Tsinghua University), Siqi Cai `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 9548 | [OpenAlex ID](https://openalex.org/A5115593970)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了针对视频大语言模型（Video-LLM）的通用能耗-延迟攻击（VidDoS），通过在视频帧中插入一次性学习的全局触发补丁来诱发模型产生无节制的长文本输出，从而极大地增加推理延迟并消耗计算资源。

**💡 创新点**

创新点在于：①首次提出针对 Video-LLM 的通用能耗-延迟攻击框架；②利用局部空间集中补丁绕过视频编码器的时序低通滤波效应；③结合 Masked Teacher Forcing、拒绝惩罚与早期终止抑制三种损失，实现对生成轨迹的精准操控；④采用 Sign‑PGD 离线优化，完成一次性训练后即可在任意视频流上即时投毒。

**🔧 技术方法**

核心技术包括：通用对抗补丁（Patch‑based Trigger）、Masked Teacher Forcing、Refusal Penalty、Early‑Termination Suppression、Sign‑PGD 优化、无梯度推理阶段的“train‑once‑deploy‑anywhere”设计。

**📊 数据集**

实验使用了三大视频数据集：BDDX、D²‑City（自动驾驶/交通摄像头场景）和 VideoSimpleQA（通用视频问答），并在三款主流开源 Video‑LLM（LLaVA‑NeXT‑Video‑7B、Qwen3‑VL‑4B‑Instruct、Video‑LLaVA‑7B‑hf）上评测。

**📈 对比分析**

与随机噪声、Verbose Images、NICGSlowDown 等基线进行对比，VidDoS 在所有模型和数据集上均实现了 Token 量级 205×、推理时延 15× 的增长，显著超过基线；在实时自动驾驶流模拟中，还导致安全时间窗口被突破。

**⚠️ 局限性**

局限性：①对语义差异极大的跨域场景（如从自动驾驶迁移到普通视频问答）时攻击效果显著下降；②缺乏对抗检测与防御方法的探讨，未来需评估模型在检测机制下的鲁棒性。

---

## 840. StepVAR: Structure-Texture Guided Pruning for Visual Autoregressive Models

**arXiv ID:** 2603.01757 | [PDF](https://arxiv.org/pdf/2603.01757v1)

**作者:** Keli Liu `[一作]` (University of Science and Technology of China), Houqiang Li `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 StepVAR，一种训练无关的 token 剪枝框架，用于加速视觉自回归（VAR）模型的推理。它通过双重重要性评分（结构+纹理）选择稀疏 tokens，并采用最近邻特征传播重建稠密特征，以支持后续尺度的生成。

**💡 创新点**

创新点在于：①将结构信息（PCA 主成分）与纹理信息（高通滤波）结合，形成统一的全局-局部重要性分数；②引入最近邻特征传播，实现无语义缺口的稠密重建；③保持训练无关、可插拔的设计，兼容多种 VAR 体系。

**🔧 技术方法**

使用的技术包括：Transformer 自注意力、PCA（通过幂迭代快速估计主成分）、轻量级高通滤波、最近邻特征传播、token 剪枝与重采样、next-scale 预测策略。

**📊 数据集**

实验使用的主要数据集：文本到图像的 Infinity、HART；文本到视频的 InfinityStar；评价集包括 GenEval、DPG、MJHQ‑30K、VBench 等。

**📈 对比分析**

与 FastVAR、SparseVAR 等基线在相同剪枝配置下对比，结果显示：在 HART 上实现 1.4× 的速度提升，GenEval 0.51、DPG 74.58；在 Infinity 上实现 2.0× 的速度提升，DPG 82.65；在 InfinityStar 上实现 1.4× 的速度提升，VBench 83.35（接近原始 83.88）。FID 也在多数场景下优于基线。

**⚠️ 局限性**

局限性包括：当剪枝比例过高（如 0.9）时会出现显著质量下降；对极高分辨率下的细节捕捉仍有一定欠缺；目前验证主要在两种图像和一种视频模型，跨模型通用性和长序列时间维度的进一步评估仍待深入。

---

## 841. LLMs as Strategic Actors: Behavioral Alignment, Risk Calibration, and Argumentation Framing in Geopolitical Simulations

**arXiv ID:** 2603.02128 | [PDF](https://arxiv.org/pdf/2603.02128v1)

**作者:** Veronika Solopova `[一作]`, Ostap Vykhopen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估六种LLM在四个真实地缘政治模拟中的决策与人类行为相似度，分析其行动、风险校准与辩论框架；

**💡 创新点**

首次在结构化、多轮现实情景中对多模型决策者进行实证评估，构建人机对齐框架并引入理论驱动的框架分类，提出基于模拟的评估方法；

**🔧 技术方法**

使用LLM推理与动作生成、文本特征分析（token长度、TTR、TF‑IDF）、自动框架标注（GPT‑4o）、统计对齐与一致性指标（F1、Krippendorff α、Mann‑Whitney U）；

**📊 数据集**

四个地缘政治模拟的行动菜单及MBA学生决策记录，LLM在相同提示下生成的动作与理由；

**📈 对比分析**

通过精确动作匹配、微宏F1、严重性分布、对齐率与模型间一致性评估，发现模型在第一轮与人类相似度中等（0.25‑0.54），随后下降；在严重性上与人类相似但变化幅度差异；所有模型倾向于稳定、合作的论证框架；

**⚠️ 局限性**

受限于预设行动空间与合作性场景，缺乏高度竞争或不确定性情境；模型缺少对抗性思路，解释中对立面呈现不足；数据量有限，模型对情境的适应性和多样性受限。

---

## 842. LAD-Drive: Bridging Language and Trajectory with Action-Aware Diffusion Transformers

**arXiv ID:** 2603.02035 | [PDF](https://arxiv.org/pdf/2603.02035v1)

**作者:** Fabian Schmidt `[一作]` (Esslingen University of Applied Sciences), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出了LAD-Drive框架，结合多模态LLM、动作解码器和带特征瓶颈的扩散解码器，生成多模态、可行的轨迹并实现对自然语言指令的精确跟随。

**💡 创新点**

核心创新在于：①结构化分离高层语义意图与低层空间规划，采用动作解码器生成概率元动作分布作为显式信念状态；②在扩散解码器中引入特征瓶颈过滤LLM高维噪声，并通过状态-意图联合条件实现对运动的安全约束；③利用k-means聚类的轨迹锚点与两步截断扩散实现高效实时生成。

**🔧 技术方法**

使用技术包括：多模态LLM（AD-MLLM）+ Q-Former、动作解码器（多层FFN+Softmax）、扩散解码器（两步截断扩散、跨注意力、多头交叉注意力）、特征瓶颈（MLP投影）、锚点编码、损失设计（最近锚匹配、BCE+L1、交叉熵）。

**📊 数据集**

训练数据为GraphPilot数据集的原始传感器与导航指令，评估使用LangAuto基准（CARLA模拟器中的8座城镇）。

**📈 对比分析**

与LMDrive、AdaDrive、VLDrive等方法对比，LAD-Drive在LangAuto的Driving Score平均提升至68.2，超过基线42.9（提升约59%），并在碰撞率、路线偏差等指标上显著下降。

**⚠️ 局限性**

局限性包括：仅使用横向动作分布作为高层条件，未细化纵向指令；两步扩散在极端复杂场景下可能限制多样性；缺乏对更长时间推理与更大数据集的验证。

---

## 843. Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration

**arXiv ID:** 2603.01725 | [PDF](https://arxiv.org/pdf/2603.01725v1)

**作者:** Guanglu Dong `[一作]` (Sichuan University), Lichao Mou `[通讯]` (MedAI Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了第一种多域全域图像恢复方法 DATPRL-IR，能够在单一模型中同时完成自然场景、医学影像和遥感图像的多种恢复任务。

**💡 创新点**

创新点在于双提示池（任务提示池与域提示池）和提示组合机制，利用跨任务、跨域的共享与特定知识，并通过大模型蒸馏获得域先验，实现域感知任务提示表示。

**🔧 技术方法**

技术手段包括基于 prompt 学习的提示池、提示组合机制 (PCM)、跨模态对齐蒸馏（使用 LLaVA 与 CLIP）、交叉注意力融合与自适应门控融合（AGF），以及多种正则化（多样性、熵、对比）以提升提示多样性和利用平衡。

**📊 数据集**

使用 6 任务（自然场景 SR、去雨；医学 MRI SR、CT 降噪；遥感 SR、去云）和 9 任务（再加自然去模糊、医学 PET 合成、遥感去雾）三大域的数据集，包括 DIV2K、Rain100L、GoPro、IXI、AAPM、PolarStar、UCMerced、CUHK CR1、RICE1 等。

**📈 对比分析**

与多种 SOTA AiOIR 与单任务基线相比，DATPRL-IR 在 6 任务/3 域场景下平均 PSNR 提升约 0.37 dB，单任务去雨最高提升 1 dB；在 9 任务/3 域时亦保持或提升性能，证明了跨任务、跨域知识共享的有效性。

**⚠️ 局限性**

局限性包括对提示池规模与 top‑k 选择的敏感性、对大模型蒸馏的依赖（虽然不影响推理成本），以及在极其不平衡或全新域/任务时可能仍需手工调参或重新蒸馏。

---

## 844. Lookahead identification in adversarial bandits: accuracy and memory bounds

**arXiv ID:** 2603.00803 | [PDF](https://arxiv.org/pdf/2603.00803v1)

**作者:** Nataly Brukhim `[一作]` (DIMACS), Carlo Ciliberto `[通讯]` (University College London)

**通讯引用:** 1500 | [OpenAlex ID](https://openalex.org/A5048960946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在对抗性多臂老虎机模型中提出并分析了“lookahead best-arm identification”任务，给出了在有限记忆下实现误差为O(1/√log T)的算法，并证明了任何非平凡识别算法至少需要Ω(K)位记忆；在满足局部稀疏性条件时，算法可降至多项式对数位记忆。随后，作者还研究了在同一记忆约束下的 regret 最小化问题，提出了一种实现O(T^{2/3}K^{1/3})子线性 regret 的算法，并与先前的专家学习结果做对比。

**💡 创新点**

创新点包括：① 在对抗性环境下首次给出可实现的 lookahead BAI 误差与记忆下界；② 在稀疏实例下实现记忆量大幅降低；③ 将专家学习的记忆约束方法迁移到 bandit 反馈下，实现子线性 regret 与多项式对数记忆的组合；④ 明确展示了 BAI 与 regret 在记忆需求上的根本分离。

**🔧 技术方法**

核心技术主要为：随机窗口与停止时间的采样（log T 级别的指数分布），基于二叉树平均数的随机游走分析，Hoeffding 及 Jensen 等概率工具；在稀疏性下引入 Count‑Sketch 频率估计；利用专家学习中（σ,s）-bounded‑memory 在线学习框架和通信复杂度的 Set‑Disjointness 归约来证明记忆下界；通过多阶段分块与探索-利用策略将专家学习算法迁移到 bandit 反馈。

**📊 数据集**

本文为纯理论研究，未使用具体数据集，而是通过构造对抗性序列与概率分布来证明误差、记忆下界与 regret 上界。

**📈 对比分析**

实验与比较方面，作者在理论层面给出了与已知下界的匹配或近似匹配：lookahead BAI 误差上界 O(1/√log T) 与下界 Ω(1/log T) 的平方根级别匹配；记忆下界 Ω(K) 与算法所需 Ω(K) 位相符；在 regret 方面，提出的 O(T^{2/3}K^{1/3}) 上界在多项式对数记忆下显著优于之前的 O(T^{3/4}K^{1/4}) 或专家学习的 O(√(KT))，但与专家学习的最佳下界仍存在差距。

**⚠️ 局限性**

局限性包括：① 对稀疏实例的记忆下界尚未得到证明；② 对于一般对抗性 bandit，提出的 regret 上界可能并非最优，仍有与专家学习下界的差距；③ 所有结果均为理论分析，缺乏实验验证；④ 记忆下界的证明依赖于通信复杂度的归约，可能对实际实现有一定距离。

---

## 845. Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation

**arXiv ID:** 2603.02190 | [PDF](https://arxiv.org/pdf/2603.02190v1)

**作者:** Divyanshu Daiya `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 Sketch2Colab，一种利用手绘草图实现多人协作人–物–人(HOH)三维动画的生成框架。

**💡 创新点**

创新点包括：将草图驱动的扩散教师蒸馏为 rectified‑flow 学生；使用能量导引和双空间约束实现精准的关键帧与轨迹控制；以及引入连续时间马尔科夫链（CTMC）对接触、握持等离散事件进行时间调度。

**🔧 技术方法**

主要技术包括：扩散概率流与 rectified‑flow 蒸馏、能量模型（关键帧、轨迹、接触、碰撞等）、双空间导引（原始空间与潜在空间）、CTMC 事件规划，以及 VQ‑VAE 潜在编码解码器。

**📊 数据集**

使用了 CORE4D 与 InterHuman 两个多主体人–物交互数据集，配合从 3D 运动生成草图的自制手绘数据进行训练与评估。

**📈 对比分析**

在关键帧、轨迹、碰撞、对象位置等多项指标上相较于 COLLLAGE、Sketch2Anim 及检索基线取得显著提升（如 FID 下降 23%，关键帧误差下降 31%，接触误差降低 30%），且推理速度比扩散模型快 3–5 倍。

**⚠️ 局限性**

局限性包括：仅支持训练集中出现的对象类别、缺乏对多人与多物体多重协作的通用建模、以及对扩散教师的依赖；对极端手绘噪声或大规模自定义对象仍表现出误差和碰撞。

---

## 846. PonderLM-3: Adaptive Token-Wise Pondering with Differentiable Masking

**arXiv ID:** 2603.02023 | [PDF](https://arxiv.org/pdf/2603.02023v1)

**作者:** He Li `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PonderLM-3，构建基于自监督预训练的token级自适应推理计算框架，使每个token可根据需求动态分配额外推理步骤；

**💡 创新点**

创新点在于将动态计算与自监督预训练耦合，利用可微分注意力掩码在训练中逼近推理时的硬停止，从而实现端到端一致的token级计算分配；

**🔧 技术方法**

核心技术包括轻量级路由器预测每token的步骤分布、尾分布累计得到掩码、可微分注意力掩码、加权隐藏状态融合、Jacobi迭代并行训练以及辅助最小推理损失；

**📊 数据集**

使用15B-token的The Pile子集进行预训练，随后在LAMBADA、SciQ、HellaSwag、PIQA、WinoGrande、ARC、RACE等公开基准上评估下游性能；

**📈 对比分析**

与PonderLM-1/2、LoopedLM、Pause、MoR等基线比较，PonderLM-3在匹配推理FLOPs的前提下实现更低perplexity；在下游任务上保持与PonderLM-2相近的准确率，同时实际推理FLOPs更低；

**⚠️ 局限性**

局限性包括：需要设定最大步骤K且随之训练/推理成本上升；对极大模型的可扩展性与硬件加速兼容性待验证；路由器学习仍受训练与推理时序差异影响，可能在极端推理策略下性能波动；

---

## 847. Quantitative Monitoring of Signal First-Order Logic

**arXiv ID:** 2603.00728 | [PDF](https://arxiv.org/pdf/2603.00728v1)

**作者:** Marek Chalupa `[一作]` (Zeroth Research), Emily Yu `[通讯]` (Leiden University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了信号一阶逻辑（SFO）的鲁棒性量化语义，并设计了一套针对过去时、有限未来视角的在线监测算法，实现了从公式到实时鲁棒度的完整闭环。

**💡 创新点**

创新点在于：①首次为全SFO定义鲁棒性语义，突破了仅布尔语义的局限；②提出“pastification”技术，将有限响应SFO公式转换为等价的过去时片段；③利用凸多面体符号运算实现高效的参数化最优化，支持量化监测；④在同一框架下兼容现有STL监测与更高表达式的需求。

**🔧 技术方法**

主要技术包括：逻辑语义重写与量化变量消除；凸多面体（Parma Polyhedra Library）求交、投影与极值求解；符号线性规划（SymPy+pply）实现鲁棒度最优化；以及基于时间窗的历史管理与垃圾回收。

**📊 数据集**

实验使用了两个典型的物理仿真数据集：①动态障碍物避让（8架无人机场景，采样周期0.1 s）；②高空机翼高度控制（F‑16模型，采样周期0.033 s）。

**📈 对比分析**

通过与基准STL监测以及多属性组合的对比，实验显示：对无历史或短历史（≤0.1 s）的属性，单段计算时间低于控制周期（0.1 s）；对含量化子句、长历史（≤20 s）的属性，单段时间约0.2–0.5 s，虽略超控制周期但仍能及时发警；整体性能证明算法在常见安全约束下可实现实时监测。

**⚠️ 局限性**

局限性包括：①仅适用于有限未来视角的SFO，无法处理无界未来的属性；②对极大时间窗或多层量化的公式，polyhedral 操作规模急剧增大导致性能下降；③假设信号为分段线性且采样规则已知，非线性或非均匀采样需额外处理；④鲁棒性语义在不同物理量单位混合时缺乏直接解释。

---

## 848. Neural Discrimination-Prompted Transformers for Efficient UHD Image Restoration and Enhancement

**arXiv ID:** 2603.00853 | [PDF](https://arxiv.org/pdf/2603.00853v1)

**作者:** Cong Wang `[一作]` (Shenzhen Campus of Sun Yat-Sen University), Yang Yang `[通讯]` (University of California)

**通讯引用:** 158115 | [OpenAlex ID](https://openalex.org/A5010426030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向超高清(UHD)图像恢复与增强的轻量级Transformer模型UHDPromer，利用高分辨率与低分辨率特征的神经差异来提升低分辨率特征表达，并通过超分辨率引导重建实现更高质量的恢复。

**💡 创新点**

创新点包括：
- 引入神经判别先验(NDP)衡量高低分辨率特征差异，并将其嵌入注意力（NDPA）和前馈网络（NDPN）中，形成神经判别提示Transformer；
- 设计连续门控机制使得低分辨率特征能更有选择性地传递关键信息；
- 采用超分辨率引导重建（SR-guided Reconstruction）在低分辨率特征被超分后再指导最终恢复，提升色彩与细节。

**🔧 技术方法**

主要技术包括：
- Transformer架构（多头注意力与深度卷积前馈网络）
- 神经判别先验计算与交叉注意力整合
- 连续门控机制（类似Gated-Dconv Feed-Forward Network）
- 低分辨率特征的shuffle down/up和深度卷积实现
- 超分辨率网络（FeaSR）与重建分支（SRG-Recon）
- 联合空间频域损失（ϕ）

**📊 数据集**

使用的数据集：
- UHD任务：UHD-LL（低光增强）、UHD-Haze（去雾）、UHD-Blur（去模糊）
- 通用图像任务：LOL（低光增强）、SOTS-ITS（去雾）、GoPro（去模糊）

**📈 对比分析**

与现有方法对比：
- 在UHD低光增强、去雾、去模糊三大任务中，UHDPromer在PSNR/SSIM/Lpips上均超过或接近最先进方法；
- 参数量仅0.743M，FLOPs 32.56G，推理时间0.12s（1024×1024），比Restormer、FFTformer等Transformer方法高效；
- 在通用图像任务上，UHDPromer虽能取得不错的PSNR/SSIM，但整体低于专门针对一般尺寸图像的最佳方法。

**⚠️ 局限性**

局限性：
- 设计以低分辨率特征为主，导致对普通尺寸图像（尤其是大尺寸的去模糊）性能不佳；
- 对通用图像的去雾、去模糊效果不及专门针对该任务的大模型；
- 目前不支持同时兼顾UHD与普通尺寸图像的统一处理，限制了其在更广泛应用中的适用性。

---

## 849. Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy

**arXiv ID:** 2603.01953 | [PDF](https://arxiv.org/pdf/2603.01953v1)

**作者:** Pengyuan Wu `[一作]` (Zhejiang University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence, China Telecom Corp Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出DCDP框架，结合预训练扩散策略与动态特征注入，实现实时闭环动作纠正；

**💡 创新点**

在不重新训练扩散策略的前提下，通过轻量级自监督动态特征编码器、跨/时序注意力以及非对称动作编码解码，实现长周期规划与高频动态响应的协同；

**🔧 技术方法**

自监督对比学习、跨时序注意力、变分自编码器、扩散模型以及滑动窗口动态特征提取；

**📊 数据集**

200条人类演示轨迹的PushT数据集以及公开的FastUMI数据集；

**📈 对比分析**

与原始开放式、闭环单步、时间集成等基线相比，DCDP在动态PushT任务中成功率提升约19%，在不同扰动条件下仍保持较低的5%计算开销；

**⚠️ 局限性**

仅在单一模拟任务上评估，缺乏多任务和大规模硬件验证，且对不同动态模式的泛化需进一步探究。

---

## 850. On the Rate of Convergence of GD in Non-linear Neural Networks: An Adversarial Robustness Perspective

**arXiv ID:** 2603.02095 | [PDF](https://arxiv.org/pdf/2603.02095v1)

**作者:** Guy Smorodinsky `[一作]` (Ben-Gurion University of the Negev), Itay Safran `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了梯度下降（GD）在一个最小的二元分类设置中的收敛动态，证明了在这种简化假设下，GD成功收敛到最优的鲁棒性边际，但收敛速度极慢，严格按Θ(1/ln(t))缩放。

**💡 创新点**

首次明确给出了非线性模型中鲁棒性边际收敛速度的下界，强调了在简单架构中收敛速度缓慢的严重性。

**🔧 技术方法**

使用了梯度下降（GD）和梯度流（GF）技术，进行了严格的理论分析和实证模拟。

**📊 数据集**

使用了一个包含两个训练实例的深度为2、宽度为2的ReLU网络进行二元分类任务。

**📈 对比分析**

通过理论分析和实证评估，发现GD的收敛速度几乎总是极其缓慢，收敛到最优鲁棒性边际的距离以Θ(1/ln(t))的速度衰减，且在多次自然网络初始化中表现出相同的紧凑收敛速度。

**⚠️ 局限性**

在高度简化的设置中提供了负结果，强调了收敛速度缓慢的瓶颈，表明在复杂的过参数化模型中，这种慢速动态很可能会更为显著。

---

## 851. Subcubic Coin Tossing in Asynchrony without Setup

**arXiv ID:** 2603.02071 | [PDF](https://arxiv.org/pdf/2603.02071v1)

**作者:** Mose Mizrahi `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出一种通过委员会变换将强大但代价高的异步共识硬币转换为更低成本、容错度略低的弱硬币，从而在不需要预先设置的安全通道或密码学假设下实现对Θ(n)自适应拜占庭错误的异步共识；

**💡 创新点**

关键创新在于将委员会视作虚拟节点的概念，利用稀疏图通信实现高效的位发布，显著降低从O(n³)到O(n².⁵)或O(n^{7/3})的通信复杂度，同时保留常量公平性；

**🔧 技术方法**

核心技术包括委员会划分与反向采样、反稠密发布协议（crusader agreement）、AVSS框架、以及针对强硬币的Monte Carlo生成；

**📊 数据集**

本研究未使用传统数据集，而是基于理论分析和概率证明来评估协议的安全性与复杂度；

**📈 对比分析**

与现有最优方案相比，所提出的两种硬币实现分别在容错阈值为(1/4-ε)n和(1/3-ε)n时，通信复杂度分别达到O(n².⁵(ε⁻⁸+log n))和O(n^{7/3}ε⁻⁶log n)，并保持O(log n)的时延；

**⚠️ 局限性**

局限性包括高常数因子导致实际部署困难、对强硬币的依赖以及对安全通道的前置假设，未来工作需进一步降低通信开销至O(n²)并实现无时延常数的协议。

---

## 852. Real Money, Fake Models: Deceptive Model Claims in Shadow APIs

**arXiv ID:** 2603.01919 | [PDF](https://arxiv.org/pdf/2603.01919v1)

**作者:** Yage Zhang `[一作]`, Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对官方大型语言模型（LLM）API与其暗影API进行系统审计，量化了性能差异、模型身份验证失效及安全行为的不一致；

**💡 创新点**

首次将多维度实测与模型指纹识别相结合，对暗影API的欺骗性做出定量评估，并提出可落地的验证与审计流程；

**🔧 技术方法**

使用LLMmap指纹识别、MET统计检验、JailbreakBench与AdvBench安全评测、延迟/Token统计等技术手段；

**📊 数据集**

评测数据集包括AIME 2025、GPQA Diamond、MedQA（USMLE）、LegalBench（Scalr）、JailbreakBench与AdvBench；

**📈 对比分析**

与官方API直接对比，通过准确率、危害评分、指纹余弦距离等指标衡量；暗影API平均准确率下降多达47%，指纹匹配率低至54%，安全评分误判率显著升高；

**⚠️ 局限性**

研究仅覆盖17个暗影API，时间窗口为2025年9–12月，市场波动和API升级导致结果可能随时间变化；缺乏真实后端的ground truth，指纹与MET虽高度可信但并不能完全覆盖所有细粒度差异。

---

## 853. HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views

**arXiv ID:** 2603.01099 | [PDF](https://arxiv.org/pdf/2603.01099v1)

**作者:** Jiashu Li `[一作]` (University of the Chinese Academy of Sciences), Jianbin Jiao `[通讯]` (University of the Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为HeroGS的分层指导框架，通过图像层伪稠密监督、特征层自适应增密/剪枝以及参数层几何一致性共剪枝，提升稀视角下3D Gaussian Splatting的重建质量。

**💡 创新点**

创新点在于将稀视角问题转化为多层次全局到局部的指导策略，利用插值帧生成伪稠密图像提供全局约束；特征层通过边缘感知增密与网格控制实现高频细节恢复；参数层通过多场共剪枝剔除不一致Gaussian，显著提高几何一致性。

**🔧 技术方法**

核心技术包括基于VFI的帧插值生成伪标签、光流或Slerp插值相机外参、特征提取的边缘检测与KNN权重分配、基于网格的密度重分配与归一化、以及多字段共剪枝阈值策略；训练损失融合光度损失、深度相关系数损失和几何一致性约束。

**📊 数据集**

在LLFF和Tanks & Temples两个公共稀视角数据集上进行实验，并使用COLMAP获取初始相机信息。

**📈 对比分析**

与3DGS、FSGS、DropGaussian、CoR-GS等现有基线相比，HeroGS在PSNR、SSIM和LPIPS指标上均实现了显著提升（如LLFF 2视角下PSNR提升约2.5 dB，SSIM提升0.04），并在视觉上恢复了更多高频纹理与清晰背景。

**⚠️ 局限性**

局限性主要体现在对插值模型质量的依赖、对少量视角时仍可能出现细节失真，以及在高分辨率或动态场景下的计算开销与训练时间相对较高。

---

## 854. Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures

**arXiv ID:** 2603.01063 | [PDF](https://arxiv.org/pdf/2603.01063v1)

**作者:** Yuechen Luo `[一作]` (Tsinghua University), Fuxi Wen `[通讯]` (Tsinghua University)

**通讯引用:** 1301 | [OpenAlex ID](https://openalex.org/A5051231210)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了Vision‑Language‑Action（VLA）模型在强化学习（RL）阶段出现性能瓶颈的问题，提出通过教师模型生成结构化诊断反馈，增强RL训练的框架；

**💡 创新点**

创新点在于引入Explicit Learning from Failures（ELF）框架，利用教师模型提供可解释的失败分析报告，生成高奖励修正轨迹并将其注入RL批次，从而实现目标导向梯度；

**🔧 技术方法**

技术组合包括两阶段监督微调（SFT）、GRPO强化学习、教师模型（Qwen3‑VL‑32B）生成的结构化反馈、策略塑形（policy shaping）以及基于难度的样本筛选；

**📊 数据集**

使用Navsim v1与v2仿真数据集进行评估，同时预训练使用多源驾驶QA数据集；

**📈 对比分析**

与多种SOTA方法对比，Navsim v1 PDMS提升至91.0（比前沿方法高0.7），Navsim v2 EPDMS提升至87.1，整体性能显著优于传统GRPO、GT‑GRPO和Rule‑GRPO；

**⚠️ 局限性**

局限性在于依赖外部教师模型，受其分析能力限制；实验仅在Navsim仿真环境，缺乏闭环真实场景验证。

---

## 855. Data-Efficient Brushstroke Generation with Diffusion Models for Oil Painting

**arXiv ID:** 2603.01103 | [PDF](https://arxiv.org/pdf/2603.01103v1)

**作者:** Dantong Qin `[一作]`, Pan Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

未提供具体研究内容

**💡 创新点**

未提供创新点

**🔧 技术方法**

未提供技术细节

**📊 数据集**

未提供数据集

**📈 对比分析**

未提供比较方法或性能

**⚠️ 局限性**

未提供局限性

---

## 856. A Reconstruction System for Industrial Pipeline Inner Walls Using Panoramic Image Stitching with Endoscopic Imaging

**arXiv ID:** 2603.00714 | [PDF](https://arxiv.org/pdf/2603.00714v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 857. Towards OOD Generalization in Dynamic Graphs via Causal Invariant Learning

**arXiv ID:** 2603.01626 | [PDF](https://arxiv.org/pdf/2603.01626v1)

**作者:** Xinxun Zhang `[一作]` (Hangzhou Dianzi University), Xuan Guo `[通讯]` (Tianjin University)

**通讯引用:** 1912 | [OpenAlex ID](https://openalex.org/A5052088230)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了DyCIL模型，用于在动态图中实现OOD（out-of-distribution）泛化，主要通过生成因果动态子图、因果感知时空注意力和自适应环境生成器来捕获时空不变模式。

**💡 创新点**

创新点在于：①从因果视角构造动态因果子图生成器，显式识别因果子图；②设计因果感知时空注意力，提取因果子图内部的演化逻辑；③引入自适应环境生成器生成连续环境实例，配合干预学习实现三者协同优化，从而在多种OOD场景下获得更稳健的不变表示。

**🔧 技术方法**

技术方法包括：结构因果模型（SCM）+后门调整、变分下界求解子图生成、GNN+时空注意力机制、变分自编码器（VAE）用于环境分布推断、干预损失与不变损失的联合训练。

**📊 数据集**

使用的实验数据集有：真实数据集 Collab、ACT、Aminer；合成数据集 Synthetic‑Collab（特征演化）和 Temporal‑Motif（结构演化）。

**📈 对比分析**

与基线比较方法有：DyGNNs（GCRN、EvolveGCN、DySAT）、OOD泛化方法（IRM、VREx、GroupDRO）、图OOD方法（DIR、EERM）以及动态图OOD方法（DIDA、SILD、EAGLE、OOD‑Linker）。在Link Prediction和Node Classification任务中，DyCIL在所有数据集上均实现显著提升（如AUC最高达95%+、ACC最高达82%+），尤其在严重的分布漂移场景下性能优于所有对照方法。

**⚠️ 局限性**

局限性：目前仅针对离散同质动态图，未考虑异质图或连续动态图；在更大规模或更复杂的现实场景中的鲁棒性仍需进一步验证。

---

## 858. GraphScout: Empowering Large Language Models with Intrinsic Exploration Ability for Agentic Graph Reasoning

**arXiv ID:** 2603.01410 | [PDF](https://arxiv.org/pdf/2603.01410v1)

**作者:** Yuchen Ying `[一作]` (Zhejiang University), Mingli Song `[通讯]` (Zhejiang University)

**通讯引用:** 9376 | [OpenAlex ID](https://openalex.org/A5026532752)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 GraphScout 框架，利用 LLM 通过训练自我探索知识图谱，从而完成 agentic 图推理任务。

**💡 创新点**

创新点在于：①引入可编程的 Agentic Graph Exploration Tools（代码解释器 + 节点检索器）让 LLM 能主动、灵活地与图交互；②构建无人工标注的 Graph Quizzer 生成高质量问答+证据；③通过 Graph Solver 的强化学习后训练，让小型 LLM 内化图探索能力，避免手工设计提示。

**🔧 技术方法**

技术要点包括：大规模 LLM（Qwen3-4B/8B、DeepSeek-Chat 等）、Neo4j Cypher 与 FAISS 向量检索、代码解释器、强化学习（Group Relative Policy Optimization）、基于答案与证据的奖励设计、QwenScore 评价指标。

**📊 数据集**

数据集：GRBENCH（5 个知识图谱领域：Healthcare、Literature、Academic、E-Commerce、Legal，共 1,740 题目），以及基准实验中使用的其他公开 GraphRAG 数据集。

**📈 对比分析**

与 BaseLLM、TextRAG、GraphRAG、Cypher、GraphCoT、PolyG、GraphCounselor 等方法在 GRBENCH 上对比；经过 GraphScout 训练后，4B 模型平均 QwenScore 提升 16.7%，超越更大 LLM 基线，且在跨域测试中保持稳健性能，使用的推理 token 数量显著降低。

**⚠️ 局限性**

局限性：训练阶段仍需强 LLM 生成高质量问答；对推荐类（hard）问题提升有限；强化学习过程可能不稳定且易受自我偏差影响；依赖 Neo4j/Cypher、FAISS 等底层工具，部署成本较高。

---

## 859. Curation Leaks: Membership Inference Attacks against Data Curation for Machine Learning

**arXiv ID:** 2603.00811 | [PDF](https://arxiv.org/pdf/2603.00811v1)

**作者:** Dariush Wahdany `[一作]` (CISPA Helmholtz Center for Information Security), Franziska Boenisch `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了数据策划流程中各阶段的隐私泄露风险，并提出了针对评分、子集和最终模型的多种成员推断攻击

**💡 创新点**

首次系统评估策划管线的隐私风险并展示即使不直接训练敏感数据，策划本身也会泄露成员信息，且提出了针对不同策划方法的专门攻击与差分隐私防御

**🔧 技术方法**

利用LiRA概率框架、投票推断、最小二乘、迭代重建、指纹注入等技术，并结合差分隐私的高斯机制

**📊 数据集**

在六个公开数据集上实验，包括CIFAR-10/100、STL-10、RESISC45、PatchCamelyon和Food101

**📈 对比分析**

与传统MIA基线对比，攻击在Image‑based策划中AUC可达0.9以上，TRAK在小样本时也能达到高TPR；DP改造后在ε=10时攻击效果基本消失，证明有效

**⚠️ 局限性**

依赖对策划池的可控注入、对大规模数据集的计算成本、DP在高维梯度下需要更强参数，且实验仅覆盖了有限的策划方法和数据规模

---

## 860. PhysGraph: Physically-Grounded Graph-Transformer Policies for Bimanual Dexterous Hand-Tool-Object Manipulation

**arXiv ID:** 2603.01436 | [PDF](https://arxiv.org/pdf/2603.01436v1)

**作者:** Runfa Blark Li `[一作]` (University of California), Truong Nguyen `[通讯]` (University of California)

**通讯引用:** 15593 | [OpenAlex ID](https://openalex.org/A5102719190)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种基于图Transformer的物理结构感知策略，用于双手工具-物体的精确协作操控。

**💡 创新点**

将双手系统建模为动力学图，并通过每条连杆的token化和物理正则化的多头注意力偏置（空间、关节、几何、解剖先验）实现对物理相互作用的显式推理。

**🔧 技术方法**

使用图Transformer网络、头特定物理偏置生成器、参考轨迹监督的PPO强化学习、RBF几何偏置以及序列与协同掩码等技术。

**📊 数据集**

基于OakInk2双手操控任务数据集，涵盖六种高难度工具使用场景。

**📈 对比分析**

与SOTA基线ManipTrans比较，PhysGraph在成功率、工具/物体平移误差、关节误差和指尖误差等指标上均显著提升（平均成功率提升约30‑40%，参数量仅为ManipTrans的51%），并实现零样本迁移和跨手模型兼容。

**⚠️ 局限性**

仅在推理时依赖参考轨迹，限制了在无轨迹环境下的部署；此外，对示范数据的噪声敏感，导致某些误差指标可能不如基线。

---

## 861. SpectroFusion-ViT: A Lightweight Transformer for Speech Emotion Recognition Using Harmonic Mel-Chroma Fusion

**arXiv ID:** 2603.00746 | [PDF](https://arxiv.org/pdf/2603.00746v1)

**作者:** Faria Ahmed `[一作]` (Islamic University of Technology), Sabbir Ahmed `[通讯]` (Islamic University of Technology)

**通讯引用:** 1197 | [OpenAlex ID](https://openalex.org/A5068841060)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种轻量级视觉变压器框架SpectroFusion‑ViT，用于识别孟加拉语情感语音；

**💡 创新点**

创新点包括将色度（Chroma）和MFCC特征融合成统一时频描述符，采用仅2.04M参数、0.1GFLOPs的ViT架构，并结合强大数据增强与迁移学习；

**🔧 技术方法**

主要技术为声学特征提取（Chroma+MFCC）、特征融合、轻量级ViT自注意力模型、在线数据增强与迁移学习；

**📊 数据集**

使用孟加拉语情感语音两大数据集：SUBESCO（7,000句）和BanglaSER（1,467句）；

**📈 对比分析**

与多种CNN和现有最优方法对比，SpectroFusion‑ViT在SUBESCO上达92.56%准确率、BanglaSER上达82.19%，均优于DenseNet、ResNet等基线和先前工作；

**⚠️ 局限性**

局限性包括仅针对孟加拉语、仅单模态（语音）且数据量有限，可能对跨语言或跨域情感识别的泛化能力受限。

---

## 862. Deep Unfolding for SIM-Assisted Multiband MU-MISO Downlink Systems

**arXiv ID:** 2603.02122 | [PDF](https://arxiv.org/pdf/2603.02122v1)

**作者:** Muhammad Ibrahim `[一作]` (University of Manitoba), Ekram Hossain `[通讯]` (University of Manitoba)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了面向堆叠智能金属表面（SIM）支持的多频段多用户MISO下行系统的深度展开网络 MBDU-Net，用于高效优化共享的 SIM 相位配置。

**💡 创新点**

创新点在于在物理一致的阻抗域 SIM 模型下，将多频段相位优化的投影梯度更新展开为可训练的网络，加入频段感知的步长和动量状态，以一次固定深度即可实现快速收敛。

**🔧 技术方法**

采用深度展开（deep unfolding）、投影梯度、动量加速、频段分离梯度、离散相位投影以及离散化的权重学习等技术。

**📊 数据集**

使用基于 Rayleigh 衰落的合成信道样本进行离线训练，并在未见过的随机信道样本上进行测试。

**📈 对比分析**

与传统投影梯度下降（GD）和基础深度展开（DU）相比，MBDU 在同等迭代次数下获得更高的总速率，收敛速度快、在子载波数量变化时仍保持较好性能。

**⚠️ 局限性**

局限性包括：需离线训练且训练数据为理想化合成信道；对硬件非理想性、时变信道等实际场景的鲁棒性尚未验证；网络结构虽轻量化但仍涉及多组可训练参数。

---

## 863. Decoding Answers Before Chain-of-Thought: Evidence from Pre-CoT Probes and Activation Steering

**arXiv ID:** 2603.01437 | [PDF](https://arxiv.org/pdf/2603.01437v1)

**作者:** Kyle Cox `[一作]` (Independent), Adrià Garriga-Alonso `[通讯]` (FAR AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型的 chain-of-thought（CoT）推理的可信度，发现模型往往在生成 CoT 前已预先决定答案，并通过线性探针和激活调控验证该预先承诺答案在激活空间中的可解码性与因果性，同时对干预后出现的推理错误模式（confabulation、non‑entailment）进行分类。

**💡 创新点**

首次在 CoT 前激活上构造差分探针，证明其不仅能预测答案且具有因果影响；通过激活调控揭示模型在错误答案时的两种不可信推理模式；系统性比较不同模型与任务中预承诺现象，推动对 CoT 可解释性与安全性的深入理解。

**🔧 技术方法**

使用线性差分探针 (difference‑of‑means)、对抗性激活调控（contrastive activation addition）、CoT 移除/替换干预、GPT‑5‑mini 自动分类器评估推理模式，并通过 AUC 与答案翻转率衡量模型行为。

**📊 数据集**

Anachronisms、Logical Deduction、Sports Understanding 与 Social Chemistry 四个二分类推理数据集，包含事实、逻辑、体育与社交判断任务。

**📈 对比分析**

与无 CoT、CoT、CoT 干预前后准确率对比；探针 AUC 超过 0.9，证明预先答案可线性解码；激活调控翻转率显著高于正交基线，尤其在大模型（Gemma‑2 9B、Qwen‑2.5 7B）上表现突出；CoT 对 Logical Deduction 任务提升最大。

**⚠️ 局限性**

仅在指令调优模型上验证，推理模型对探针与调控效果不明显；激活空间可能存在超叠加，影响因果解释；对预先承诺特征本质解释仍有限；干预阈值设定及失败案例分析尚不充分。

---

## 864. NeuroSymb-MRG: Differentiable Abductive Reasoning with Active Uncertainty Minimization for Radiology Report Generation

**arXiv ID:** 2603.01756 | [PDF](https://arxiv.org/pdf/2603.01756v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了NeuroSymb-MRG框架，融合可微神经符号推理、检索增强生成与主动不确定性最小化，实现结构化、临床可信的放射科报告生成。

**💡 创新点**

创新点在于将可微逻辑层与可解释推理链结合，利用主动不确定性采样聚焦临床审核，并通过多智能体协作与UMLS知识图验证提升事实一致性。

**🔧 技术方法**

采用可微神经符号逻辑层（软树、T‑norm、概率和门控）、自监督视觉编码器、检索式模板填充、受约束LLM精炼、Monte Carlo dropout不确定性估计、k‑center多样性采样、反馈模拟器与多智能体通信。

**📊 数据集**

使用公开胸部X光图像-报告数据集MIMIC‑CXR和IU X‑ray进行训练与评估。

**📈 对比分析**

与Show‑Tell、Transformer、R2Gen、M2Transformer等多种基线在BLEU、ROUGE、METEOR等指标上对比，NeuroSymb‑MRG在所有指标上均领先，BLEU‑4提升约0.1，事实一致性显著提高。

**⚠️ 局限性**

局限在于高度依赖大规模预训练模型与外部知识库，推理链长度受规则设计限制，处理罕见病症时仍面临数据稀缺与不确定性挑战。

---

## 865. Understanding the Physics of Key-Value Cache Compression for LLMs through Attention Dynamics

**arXiv ID:** 2603.01426 | [PDF](https://arxiv.org/pdf/2603.01426v1)

**作者:** Samhruth Ananthanarayanan `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 5059 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究将 KV 缓存压缩视为对大型语言模型（LLM）自注意力路由的结构性扰动，通过物理启发框架分析压缩对 token‑level 路径的影响，并揭示了压缩导致的安全悬崖和路由僵化两种失败模式。

**💡 创新点**

创新点包括：①提出 Global Eviction Ratio（GER）和 head‑level consensus 等结构性度量；②将 KV 缓存压缩与稀疏 token‑route 子网络（TR‑LT）联系，形成 “路由级 Lottery Ticket” 的新视角；③通过合成数据集精准分离信息存储、可达性和利用三大维度，发现不同模型（LLaMA vs Qwen）在深度路由动态上的根本差异。

**🔧 技术方法**

主要技术手段包括：物理启发式分析框架、合成数据集设计、KVPress 压缩（FINCH、AdaKV）、线性探测器（probe）、注意力图可视化、统计指标（GER、consensus）以及多模型多压缩比例实验。

**📊 数据集**

使用了自制的合成数据集（Base、Knowledge manipulation、Multi presence、Multi entity、Long context、Coreference、Hops）以及标准长上下文基准 LongBench、RULER 等进行对比评估。

**📈 对比分析**

采用 F1、hallucination 率等指标在不同压缩比例（0%–90%）、问答设置（question‑agnostic vs. aware）和模型族（LLaMA、Qwen）下进行横向比较。实验表明，轻度压缩保持或略升性能，中等压缩可见性能波动，90% 级压缩出现明显的“安全悬崖”，不同架构在路由稳健性上呈现不同的性能曲线。

**⚠️ 局限性**

局限性在于：①合成数据过度控制，未覆盖真实自然语言、多模态或外部记忆情境；②GER、consensus 等指标经验性强，缺乏严谨的理论证明；③未在训练阶段引入鼓励稀疏路由的机制，未来可探索联合稀疏化设计与压缩策略。

---

## 866. Semantic Similarity is a Spurious Measure of Comic Understanding: Lessons Learned from Hallucinations in a Benchmarking Experiment

**arXiv ID:** 2603.01950 | [PDF](https://arxiv.org/pdf/2603.01950v1)

**作者:** Christopher Driggers-Ellis `[一作]` (University of Florida), Bonnie Dorr `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究针对盲人/视障用户的漫画/连环画可访问性，构建了基于页面级描述的评估基准，并评估了多种视觉语言模型（VLM）的表现；

**💡 创新点**

首次在页面级漫画解释任务中系统性比较多种VLM，提出了四类幻觉（未出现文本、误归属、错误物体/场景描述、错误人物描述）并将其映射至图像字幕的幻觉分类体系；

**🔧 技术方法**

使用VLM（如Qwen2.5、CogVLM、Idefics3、LLaVa-1.6、MiniCPM、InstructBLIP）和语义相似度度量（余弦相似度、KL散度）来评估生成文本；

**📊 数据集**

构建了158页漫画的人工标注语料（从公开漫画网站挑选并手工生成 400 字以内的叙述），并用 149 页用于模型评测；

**📈 对比分析**

评测结果显示 Qwen2.5 在余弦相似度和 KL 散度上表现最佳，但所有模型均存在显著幻觉，尤其是 MiniCPM 在物体/人物描述上 100% 幻觉；

**⚠️ 局限性**

主要局限包括数据集规模小、缺乏多参考/人类评估、模型未针对任务微调、GPU 受限导致只使用 7‑9B 规模模型、未覆盖盲人用户反馈。

---

## 867. Power Echoes: Investigating Moderation Biases in Online Power-Asymmetric Conflicts

**arXiv ID:** 2603.01457 | [PDF](https://arxiv.org/pdf/2603.01457v1)

**作者:** Yaqiong Li `[一作]` (Fudan University), Tun Lu `[通讯]` (Fudan University)

**通讯引用:** 2149 | [OpenAlex ID](https://openalex.org/A5004237040)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对消费者与商家之间的线上冲突进行实验，探讨了人类与人机协同在权力不对称冲突中的偏见表现；

**💡 创新点**

创新点在于首次系统地构建了基于社会权力理论的偏见分类，设计了对抗性扰动实验，并验证了AI辅助在减轻或放大偏见方面的双刃效果；

**🔧 技术方法**

主要技术包括对冲突语料的手工编码、基于大语言模型的 AI 建议生成（Wizard‑of‑Oz 方案）以及混合设计实验和混合效应模型分析；

**📊 数据集**

使用了从大众点评平台采集的 100 条真实消费者‑商家冲突样本，并通过 9 种扰动方式扩充到 2,000 条实验材料；

**📈 对比分析**

实验通过 50 名参与者（人类组和人机组）完成 70 轮判断任务并进行访谈，结果显示人类组在五种权力表现下显著偏向强方，而人机组在多数偏见上得到缓解，但在合法主张和权威引用偏见上出现放大；

**⚠️ 局限性**

局限性包括仅研究了中国本土的消费者‑商家场景、未测量 AI 文献素养、采用 Wizard‑of‑Oz 而非真实 LLM 输出，以及对跨文化和其他权力不对称情境的推广性不足。

---

## 868. Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report)

**arXiv ID:** 2603.01499 | [PDF](https://arxiv.org/pdf/2603.01499v1)

**作者:** Yu Lin `[一作]` (ByteDance), Sheng Zhong `[通讯]` (Nanjing University)

**通讯引用:** 6684 | [OpenAlex ID](https://openalex.org/A5060268538)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于协变置换（Covariant Obfuscation）的隐私保护 LLM 推理方法 – “Alore”，能够在不显著损失准确率或效率的前提下，保护输入与输出数据的隐私。

**💡 创新点**

创新点在于将数据置换与模型参数变换联合进行，利用可逆矩阵与噪声同时隐藏置换信息，并通过三条组合定理实现对各模型组件的独立设计与整体协同，理论证明其在同等误差下泄漏信息更少。

**🔧 技术方法**

主要技术包括 token 级置换、可逆矩阵变换、噪声注入、RoPE 块级置换、注意力头置换、FFN 线性变换、层归一化调整等，配合 vLLM/SGLang 框架实现。

**📊 数据集**

使用多种公开 LLM（Qwen2.5/Qwen3、Llama3、Deepseek‑R1‑Distill、Deepseek‑V3.1‑Terminus、Qwen3‑MoE 等）和评测数据集（SST‑2、MMLU、C‑Eval、HumanEval、IF‑Eval、PIQA、PUPA、CCI3、Huatuo26M、MedDialog）进行实验。

**📈 对比分析**

与基线置换/嵌入置换方法对比，Alore 在保持 0~3.5% 以内的准确率损失、与 plaintext 同等推理效率的同时，使逆向攻击（VMA、IMA、ISA 等）恢复率降至 3% 以下，且信息泄漏（TTRSR、MI）显著低于竞争方案。

**⚠️ 局限性**

局限性包括离线模型置换成本高（Deepseek‑V3.1‑Terminus 需 8 小时）、仅针对文本生成 LLM 证明，且对攻击者完全未知置换信息的攻击仍需进一步评估。

---

## 869. How Well Does Agent Development Reflect Real-World Work?

**arXiv ID:** 2603.01203 | [PDF](https://arxiv.org/pdf/2603.01203v1)

**作者:** Zora Zhiruo Wang `[一作]`, Graham Neubig `[通讯]` (Carnegie Mellon University)

**通讯引用:** 21511 | [OpenAlex ID](https://openalex.org/A5068811427)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了将 AI 代理基准映射到 O*NET 工作领域与技能的框架，并对 43 个基准（共 72,342 条任务）与 1,016 个真实职业进行大规模对比，评估当前基准与真实工作之间的覆盖度与代表性；提出了基于工作流程的任务复杂度度量和代理自主性定义，给出自主性曲线及其对任务选择的指导。

**💡 创新点**

①将工作领域与技能的层级结构与代理任务进行自动映射，形成统一的评估框架；②利用大语言模型（LLM）自动生成任务到工作路径的映射并手工校验，显著提升映射规模；③将任务拆解为层级工作流程，量化任务复杂度并据此定义代理自主性；④提出三条基准设计原则（覆盖度、真实性与细粒度评估），以实现更贴近真实工作的基准构建。

**🔧 技术方法**

使用大语言模型（如 GPT‑4、Claude 等）完成任务到 O*NET 路径的自动注释和验证；利用工作流程诱导技术将代理轨迹分解为层级步骤；通过覆盖率计算、采样策略和统计分析评估基准与真实工作匹配度；使用可视化工具展示域/技能覆盖与代理自主性曲线。

**📊 数据集**

O*NET 数据库（工作类别、技能与任务描述）与美国劳工统计局的就业和薪酬数据；43 个公开 AI 代理基准（共 72,342 条任务实例）；1,016 种真实职业的统计信息；LLM 自动生成的映射结果与手工校验数据。

**📈 对比分析**

对比基准在工作领域和技能层级的覆盖率与就业/资本分布；计算不同复杂度级别下的代理成功率，进而确定自主性阈值；发现计算机/数学领域被过度覆盖，管理、法律等高资本数字化领域被忽视；在高复杂度任务上多数代理成功率显著下降，提示需要更细粒度的评估与提升。

**⚠️ 局限性**

仅关注数字化工作，物理工作任务的分类与评估不足；LLM 的映射可能存在偏差，尤其在细粒度路径匹配上；任务复杂度仅基于工作流程步骤数，未考虑协调性和动态变化；缺乏大规模公开的代理轨迹数据，限制了更全面的自主性分析。

---

## 870. OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens

**arXiv ID:** 2603.02138 | [PDF](https://arxiv.org/pdf/2603.02138v1)

**作者:** Yiying Yang `[一作]` (Fudan University), Xingjun Ma `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个统一框架，能够根据多模态（文本、图像、视频）指令自回归生成高质量的 Lottie 矢量动画。

**💡 创新点**

主要创新点包括：①设计了专门的 Lottie tokenizer，将冗余的 JSON 结构压缩成紧凑的命令与参数序列；②构建了大规模多模态数据集 MMLottie‑2M（200 万条 Lottie 动画及其文本、图像、视频描述）；③在预训练的 VLM（Qwen2.5‑VL）上进行微调，实现了跨模态指令跟随与矢量动画生成。

**🔧 技术方法**

技术手段包括：偏移量量化的 Lottie tokenizer、基于 VLM 的自回归解码器、数据归一化与增强（空间统一、时间归一、运动模板迁移）、以及多模态指令编码（文本、图像、视频交互）。

**📊 数据集**

使用的数据集为 MMLottie‑2M（包含 200 万条专业制作的 Lottie 动画及对应文本、图像、视频注释）以及 MMLottie‑Bench 的真实与合成子集，用于评测文本→Lottie、文本+图像→Lottie 与视频→Lottie 三大任务。

**📈 对比分析**

在所有任务上与 DeepSeek、Qwen2.5‑VL、GPT‑5、Recraft 等基线进行对比，本文模型在 FVD、CLIP、对象一致性、运动一致性、PSNR/SSIM/DINO 等指标均取得最佳或接近最佳成绩，并且拥有更高的成功率和更优的 token 效率。

**⚠️ 局限性**

局限性包括：自回归解码仍可能产生无效的 Lottie 序列；对复杂动画的上下文长度与泛化能力有限；需要进一步引入约束解码或强化学习以提升生成可靠性和实用性。

---

## 871. SoK: Is Sustainable the New Usable? Debunking The Myth of Fundamental Incompatibility Between Security and Sustainability

**arXiv ID:** 2603.01958 | [PDF](https://arxiv.org/pdf/2603.01958v1)

**作者:** Maxwell Keleher `[一作]` (Carleton University), Sonia Chiasson `[通讯]` (Carleton University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过系统化文献综述、提取155条可持续性指南并归纳成12个主题，随后将这些主题与22条计算机安全设计原则进行比较，探讨安全与可持续性之间是否存在根本冲突。

**💡 创新点**

创新点在于首次将可持续性主题与安全原则进行系统性对照，证明两者并非不可兼容，并指出安全原则在实践中往往能够支持可持续性目标，挑战了先前关于二者相互排斥的主流观念。

**🔧 技术方法**

主要技术包括文献检索与引用追踪、主题编码与归纳、以及关系矩阵绘制，用于量化安全原则与可持续性主题之间的相互影响。

**📊 数据集**

数据集为被检索的文献集合（共约2,199篇），其中筛选出28篇包含具体可持续性指南，随后提炼出155条指南用于主题分析；未使用传统实验或测量数据集。

**📈 对比分析**

比较方法为构建交叉关系矩阵，使用符号“+”“-”分别表示安全原则与可持续性主题之间的正向或负向关联；结果显示大多数关联为正向，少数几条为负向，表明整体趋向兼容性而非冲突。

**⚠️ 局限性**

局限性包括：仅选用van Oorschot的22条安全原则，其他安全原则可能导致不同结论；主题分析依赖研究者主观判断，可能存在编码偏差；文献范围受搜索策略和排除标准限制，未涵盖所有可持续性相关研究；缺乏实证验证安全措施对可持续性影响的量化数据。

---

## 872. Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents

**arXiv ID:** 2603.01438 | [PDF](https://arxiv.org/pdf/2603.01438v1)

**作者:** Yuxin Liu `[一作]` (University of Science and Technology of China), Lei Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 106452 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个在推理时自适应角色扮演的框架PDD，能够动态估计情境下人格重要性并通过多目标奖励引导LLM生成符合角色属性的回应。

**💡 创新点**

创新点包括：① 用无监督条件互信息估计人格重要性；② 设计动态加权多目标奖励并归一化以保持属性层级；③ 在推理时实现无微调的自适应人格跟随。

**🔧 技术方法**

采用条件互信息（CMI）估计、奖励导向解码、KL约束强化学习思想、奖励归一化以及动态权重更新等技术。

**📊 数据集**

使用 CharacterEval、BeyondDialogue 以及 PERSONALITYBENCH 三个数据集，分别用于通用角色和大五人格属性测试。

**📈 对比分析**

与提示、ICL、OPAD、PAS、NPTI 等基线以及 GPT‑4o/Deepseek‑R1 进行比较，评估指标包括 GPT‑4o 判定、角色一致性、知识准确性、人格行为和话语对齐；PDD 在这些指标上普遍优于基线，性能接近商业模型。

**⚠️ 局限性**

局限性在于仍依赖模型生成的回复近似真实答案，极端噪声下重要性估计可能偏差；仅处理非敏感人格属性；对推理时计算量和奖励归一化的数值稳定性有一定要求。

---

## 873. XAI-enhanced Comparative Opinion Mining via Aspect-based Scoring and Semantic Reasoning

**arXiv ID:** 2603.01212 | [PDF](https://arxiv.org/pdf/2603.01212v1)

**作者:** Ngoc-Quang Le `[一作]` (VNU University of Engineering and Technology), Hoang-Quynh Le `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于Transformer的可解释比较意见挖掘模型，主要包括方面分类、方面评分预测和比较分类三大模块。

**💡 创新点**

创新点在于将方面评分预测拆分为基于得分的分类器和语义分类器，并引入SHAP解释模块，为模型提供可解释的推理过程。

**🔧 技术方法**

使用BERT变体进行方面分类和语义编码，XGBoost用于处理无形容词句子的评分预测，Transformer编码器进行比较分类，并通过SHAP计算特征重要性。

**📊 数据集**

在SUDO数据集上进行实验，该数据集包含多条来自同一用户的啤酒评论，标注了句子-方面和评论-比较两级标签。

**📈 对比分析**

通过将评分分类器和语义分类器的概率分布求和得到最终比较标签，实验结果显示该模型在宏观和微观F1上均达到约58.5%，显著优于基线（如Finetuned‑T5、BART以及通用LLM）。

**⚠️ 局限性**

主要限制包括模块化导致的级联误差、SHAP解释对非专业用户的可读性不足、缺乏跨方面依赖建模、数据集单一且规模有限，以及对非正式语言和讽刺语义的处理不足。

---

## 874. Latent attention on masked patches for flow reconstruction

**arXiv ID:** 2603.02028 | [PDF](https://arxiv.org/pdf/2603.02028v1)

**作者:** Ben Eze `[一作]` (Imperial College London), Andrea Nóvoa `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于视觉Transformer的LAMP模型，利用流场分块、每块POD降维以及单层线性注意力机制，实现从仅10%未遮掩且含噪的测量中重构完整二维流场。

**💡 创新点**

创新点包括：① 将Patch-wise POD与可解释的注意力矩阵相结合，避免反向传播与超参数调优；② 用闭式线性回归训练Transformer，保证全局最优收敛；③ 生成多分辨率传感器布置图，为实验传感器选址提供物理可解释依据；④ 在含噪遮掩输入下实现重构与去噪双重功能。

**🔧 技术方法**

技术核心：视觉Transformer结构、Patch-wise Proper Orthogonal Decomposition（POD）线性自编码、单层注意力机制、闭式线性回归训练、基于误差的注意力权重分配。

**📊 数据集**

数据集：① Re=100 三角钝体层流尾流（100训练+60测试帧）；② 15°斜角平板混沌尾流（2340帧，75%训练、20%测试）。

**📈 对比分析**

与无噪声及不同SNR（10–30 dB）遮掩比例对比，预测误差始终低于噪声方差；在层流尾流中，10%遮掩、SNR = 10 dB时误差低于噪声方差的10%；在混沌尾流中加入uv观测后误差下降至原来的1/10，说明LAMP在非线性观测下性能显著提升。

**⚠️ 局限性**

局限性：目前仅使用线性POD降维，未引入非线性自编码或多层Transformer；对高度混沌或三维流场的扩展尚未验证；与传统CNN/GAN等深度学习基准的直接对比缺失，需进一步实验评估。

---

## 875. Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes

**arXiv ID:** 2603.01837 | [PDF](https://arxiv.org/pdf/2603.01837v1)

**作者:** Hongkun Dou `[一作]` (Beihang University), Yue Deng `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为Constrained Particle Seeking（CPS）的无梯度逆问题求解方法；

**💡 创新点**

创新点在于利用所有候选粒子信息进行局部线性近似并加入高密度约束以高效搜索最优粒子，同时引入Restart重噪策略提升鲁棒性；

**🔧 技术方法**

技术主要包括扩散模型、逆向采样、统计线性化、受限优化以及Restart重噪；

**📊 数据集**

实验使用FFHQ图像数据集、黑洞成像数据、以及流体动力学（Navier-Stokes）数据同化；

**📈 对比分析**

与梯度无关方法（SCG、DPG、EnKG）以及梯度方法（DDRM、DPS、ΠGDM、RED-diff、DAPS）比较，CPS在图像逆问题和科学逆问题上性能与梯度方法相当或更好，显著优于其它无梯度方法；

**⚠️ 局限性**

局限性包括：在高维空间中线性近似可能失真、仍需一定数量粒子、仅适用于可预训练扩散模型且观测噪声已知的场景，对极端稀疏观测的鲁棒性尚待提升。

---

## 876. LiftAvatar: Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation

**arXiv ID:** 2603.02129 | [PDF](https://arxiv.org/pdf/2603.02129v1)

**作者:** Hualiang Wei `[一作]` (Jilin University), Wenhui Li `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在单目视频表达不足的情况下，通过生成式视频扩散变换器 LiftAvatar 对姿态与表情进行补全，以提升 3D Gaussian 头像重建与动画质量。

**💡 创新点**

① 将稀疏的单目输入升维为完整的运动空间；② 采用多粒度表情控制（阴影图 + NPHM 表情系数）；③ 支持多参考图像的条件注入，实现高精度身份与细节保持。

**🔧 技术方法**

大型视频扩散变换器（Wan2.1）+ NPHM 表情模型 + CLIP 跨注意机制 + Flow‑matching 训练目标 + LoRA 微调。

**📊 数据集**

NeRSemble 数据集（4,700+序列，267 人，约 3,170 万帧）。

**📈 对比分析**

与 FOMM、Face Vid2vid、DiffusionAvatars、LivePortrait、HunyuanPortrait 等基线及 SplattingAvatar、MonoGaussianAvatar 进行对比。LiftAvatar 在 PSNR、SSIM、LPIPS、FID、AED、CSIM 等指标上均优于对手，用户评估平均分 8.5/10，显著高于其他方法。

**⚠️ 局限性**

对极端表情或姿态的补全仍受限于训练视频的多样性；在参考图像数量或提升至多于 5 张时增益有限；对计算资源要求较高，且在极端稀疏场景下效果仍需进一步验证。

---

## 877. CLEAR: Null-Space Projection for Cross-Modal De-Redundancy in Multimodal Recommendation

**arXiv ID:** 2603.01536 | [PDF](https://arxiv.org/pdf/2603.01536v1)

**作者:** Hao Zhan `[一作]` (Hefei University of Technology), Le Wu `[通讯]` (Hefei University of Technology)

**通讯引用:** 6095 | [OpenAlex ID](https://openalex.org/A5033706423)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为CLEAR的框架，利用视觉与文本特征的交叉协方差进行SVD，并将特征投影到冗余子空间的正交补空间，从而抑制跨模态冗余，提升多模态推荐性能。

**💡 创新点**

创新点包括：①将跨模态冗余建模为低维共享子空间并直接用null‑space投影抑制冗余；②提供可调节的冗余秩k和投影强度λ，实现对冗余抑制的细粒度控制；③该方法无需改动原模型结构或损失函数，可作为plug‑and‑play模块无缝集成。

**🔧 技术方法**

主要技术：交叉协方差构造、奇异值分解（SVD）、null‑space投影、轻量级投影更新、BPR对比损失与L2正则化。

**📊 数据集**

使用Amazon的三大公开数据集：Baby、Sports和Clothing（均进行5‑core过滤，提取4096维视觉特征和384维文本特征）。

**📈 对比分析**

与14种基线（MF‑BPR、LightGCN、VBPR、MMGCN、DualGNN、LATTICE、SLMRec、BM3、MMSSL、FREEDOM、MGCN、LGMRec、MENTOR等）在Recall@10/20和NDCG@10/20上进行对比。CLEAR在所有数据集上均优于最强基线，Recall@10提升约6%–5%，Recall@20提升约7.6%–25.6%，NDCG同样表现最优。

**⚠️ 局限性**

局限性：①目前仅针对视觉与文本两模态，扩展到多模态需进一步验证；②SVD与投影计算在大规模高维场景下仍有一定开销，需进一步优化；③冗余秩k和投影强度λ需要在验证集上手工调参，缺乏自动化选择机制。

---

## 878. Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval

**arXiv ID:** 2603.01082 | [PDF](https://arxiv.org/pdf/2603.01082v1)

**作者:** Xuan Lu `[一作]` (Shanghai Jiao Tong University), Xiaoyu Shen `[通讯]` (Institute of Digital Twin, Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了MCMR基准，用于评估多条件跨模态检索。

**💡 创新点**

MCMR结合细粒度视觉与文本约束，填补了现有数据集在多条件跨模态检索上的空白。

**🔧 技术方法**

使用多模态大型语言模型（如Qwen、InternVL等）进行检索和点对点重排序，结合双编码器和MLLM嵌入。

**📊 数据集**

基于Amazon Reviews构建的10,400件产品数据，覆盖上装、下装、珠宝、鞋类、家具等五大域。

**📈 对比分析**

对比了多种检索器和重排序器，发现融合模型Recall@10最高达53%，重排序后nDCG@1提升至94%。

**⚠️ 局限性**

目前检索模型在多条件细粒度匹配上仍表现欠佳，受制于模态不平衡和缺乏可扩展的条件推理机制。

---

## 879. PPC-MT: Parallel Point Cloud Completion with Mamba-Transformer Hybrid Architecture

**arXiv ID:** 2603.00870 | [PDF](https://arxiv.org/pdf/2603.00870v1)

**作者:** Jie Li `[一作]` (Xinjiang University), Xin Ning `[通讯]` (Institute of Semiconductors)

**通讯引用:** 7391 | [OpenAlex ID](https://openalex.org/A5064149512)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于Mamba-Transformer混合架构的并行点云补全框架PPC-MT，能够在保证计算效率的前提下实现高质量的点云重建。

**💡 创新点**

创新点包括：①使用主成分分析(PCA)对无序点云进行排序和均匀划分，从而实现多头并行重建；②在编码器中引入线性复杂度的Mamba以快速捕获全局上下文，解码器仍使用Transformer以精细建模多序列关系；③为训练提供更细粒度的PCA分解监督和多尺度Chamfer/EMD损失。

**🔧 技术方法**

技术手段包括Mamba3D块、Transformer自注意力与交叉注意力、Seed Generator、Multi‑Head Reconstructor、PCA排序与均匀划分、Chamfer/EMD/ DCD/F‑Score等评价指标。

**📊 数据集**

实验数据集涵盖PCN、ShapeNet‑55/34、KITTI，三者分别用于离线评估与真实场景泛化。

**📈 对比分析**

与现有方法（FoldingNet、PMP‑Net++、PoinTr、AdaPoinTr等）比较，PPC‑MT在PCN、ShapeNet、KITTI上均获得了最优或接近最优的DCD、EMD、F‑Score和Uniformity指标，显著提升了点云分布均匀性和局部细节恢复。

**⚠️ 局限性**

局限性：模型参数和算力仍高于部分单阶段方法，PCA划分对极端形状可能导致子集不平衡，且对非点云（如网格）直接适用性有限。

---

## 880. Identifying the Geographic Foci of US Local News

**arXiv ID:** 2603.00787 | [PDF](https://arxiv.org/pdf/2603.00787v1)

**作者:** Gangani Ariyarathne `[一作]` (William & Mary), Alexander C. Nwala `[通讯]` (William & Mary)

**通讯引用:** 123 | [OpenAlex ID](https://openalex.org/A5077830996)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

NLGF模型检测美国地方新闻文章的地理焦点与地理层级，提供了可解释的geo-focus/geo-foci识别流程

**💡 创新点**

将LLM用于地名消歧义、构建空间语义特征并结合XGBoost分类，显著提升地方新闻地理焦点识别精度

**🔧 技术方法**

使用spaCy进行NER、LLM（GPT‑4o、LLaMA2‑7b、Phi‑3）进行地名消歧义、传统Geo‑parsers（Mordecai3等）、XGBoost分类、Shapely等工具

**📊 数据集**

1250篇手工标注的美国地方新闻样本，覆盖所有五类geo-focus级别，数据来自3DLNews2集合

**📈 对比分析**

与GPT‑4o和Cliff‑Clavin基线对比，NLGF在geo‑focus级别F1达0.89、geo‑foci F1达0.86，均显著优于基线

**⚠️ 局限性**

仅实现单标签geo‑focus级别，未考虑多标签情况；模型目前仅针对美国新闻，跨国适用性需进一步验证

---

## 881. Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving

**arXiv ID:** 2603.01007 | [PDF](https://arxiv.org/pdf/2603.01007v1)

**作者:** Xubo Zhu `[一作]` (Wuhan University), Huai Yu `[通讯]` (Wuhan University)

**通讯引用:** 10650 | [OpenAlex ID](https://openalex.org/A5065750220)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个统一的视觉驱动 3D 语义占据预测框架 Dr.Occ，解决几何对齐和语义不平衡问题。

**💡 创新点**

创新点在于：① 使用高质量像素级深度信息构造占据掩码并实现双向投影的 D^2‑VFormer，显著提升几何一致性；② 引入基于 Mixture‑of‑Experts 与 Mixture‑of‑Recursions 的 R‑EFormer/R^2‑EFormer，在空间上自适应分配专家以缓解稀疏类别的长尾效应。

**🔧 技术方法**

核心技术包括深度引导的 2D‑to‑3D 视图变换器 D^2‑VFormer、基于 deformable cross‑attention 的双投影策略、区域专家 Transformer（R‑EFormer）及其递归变体（R^2‑EFormer），以及利用 MoGe‑2 提供的高质量深度先验。

**📊 数据集**

在 Occ3D‑nuScenes（nuScenes 扩展）数据集上进行实验，使用 80 m × 80 m × 6.4 m 的体素网格、18 类标签。

**📈 对比分析**

与前向投影、反向投影以及双投影等现有方法对比；在 BEVDet4D 基线上提升 mIoU 7.43%、IoU 3.09%；集成到 COTR 获得 +1% mIoU，整体性能超过公开的最先进方法。

**⚠️ 局限性**

局限性包括：① 对高质量深度模型 MoGe‑2 的依赖，可能在跨域场景产生误差；② R‑EFormer 的手工区域划分需调参，递归版本在整体 IoU 上略有下降；③ 目前仅在 nuScenes 扩展上验证，通用性待进一步验证。

---

## 882. Adaptive Confidence Regularization for Multimodal Failure Detection

**arXiv ID:** 2603.02200 | [PDF](https://arxiv.org/pdf/2603.02200v1)

**作者:** Moru Liu `[一作]` (Technical University of Munich), Mario Trapp `[通讯]` (Fraunhofer IKS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对多模态系统中的误判检测（Failure Detection）问题，提出 Adaptive Confidence Regularization (ACR) 框架，并在此基础上引入 Adaptive Confidence Loss 与 Multimodal Feature Swapping 两项技术。

**💡 创新点**

创新点：① 发现并量化了多模态预测中“confidence degradation”现象，即错误样本的融合置信度低于至少一模态的置信度；② 通过 Adaptive Confidence Loss 明确惩罚这种降级；③ 设计了无外部数据的跨模态特征交换（Feature Swapping）方法，合成逼真的错误样本，提升模型对置信度不确定性的识别。

**🔧 技术方法**

技术：多模态融合模型（视频/光流/音频编码器+融合分类器），Adaptive Confidence Loss（对置信度差异惩罚），Multimodal Feature Swapping（特征维度随机交换+软标签插值），交叉熵、AURC、AUROC、FPR95 等评估指标。

**📊 数据集**

数据集：HMDB51、Kinetics‑600、HAC、EPIC‑Kitchens（视频+光流），HAC（加入音频）以及在 SemanticKITTI 上的图像+LiDAR 语义分割任务。

**📈 对比分析**

与基线（MSP、MaxLogit、Energy、Entropy、DOCTOR、OpenMix、Mixup、CRL、A2D 等）比较，ACR 在 AURC、AUROC、FPR95 以及整体准确率上均取得显著提升（如 HMDB51 上 AURC 下降 32.4%，FPR95 降至 41.96%，AUROC 提升至 92.02%）。

**⚠️ 局限性**

局限性：仍依赖于多模态数据的同步与预训练模型；在极端分布漂移或大规模异构模态组合下，Feature Swapping 的效果和 Adaptive Confidence Loss 的鲁棒性需进一步验证；实验主要聚焦于动作识别任务，跨领域推广仍有待验证。

---

## 883. From Leaderboard to Deployment: Code Quality Challenges in AV Perception Repositories

**arXiv ID:** 2603.02194 | [PDF](https://arxiv.org/pdf/2603.02194v1)

**作者:** Mateus Karvat `[一作]`, Sidney Givigi `[通讯]` (Queen’s University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对178个来自KITTI和NuScenes排行榜的AV感知模型仓库进行软件质量评估

**💡 创新点**

首次大规模研究感知模型代码质量，发现安全问题高度集中并提出针对性预防准则，且关联CI/CD实践提升可维护性

**🔧 技术方法**

使用静态分析工具Pylint、Bandit、Radon以及GitHub API收集仓库指标

**📊 数据集**

基于KITTI和NuScenes 3D目标检测排行榜的模型仓库

**📈 对比分析**

通过Spearman相关和Mann-Whitney U检验分析错误、漏洞与代码规模、CI/CD与可维护性的关系，结果显示错误与漏洞随代码量正相关，CI/CD采用显著提升可维护性，只有7.3%仓库满足生产就绪标准

**⚠️ 局限性**

仅关注两大排行榜的公开仓库，静态分析工具可能产生误报，缺乏动态分析与未公开仓库的代表性

---

