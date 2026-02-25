# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-25 | 今日论文总数: 447

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Predicting Sentence Acceptability Judgments in Multimodal Contexts

**arXiv ID:** 2602.20918 | [PDF](https://arxiv.org/pdf/2602.20918v1)

**作者:** Hyewon Jang `[一作]` (University of Gothenburg), Shalom Lappin `[通讯]` (University of Gothenburg)

**通讯引用:** 3988 | [OpenAlex ID](https://openalex.org/A5074294896)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过收集2025年发布的原始文本，经过三次机器翻译生成语义、句法不自然的句子，并结合GPT‑5生成的视觉上下文，评估人类和多模态大型语言模型（LLM）在不同视觉背景下的句子自然度判断；

**💡 创新点**

创新点在于首次探讨视觉上下文对人类和LLM句子自然度评估的影响，并揭示LLM在视觉条件下表现出压缩效应，而人类则仅在相关视觉情境下出现轻微提升效应；

**🔧 技术方法**

主要技术包括多语言 round‑trip 机器翻译（Moses）、GPT‑5 图像生成、Prolific 众包人类评分、LLM 预训练模型的提示式评估与概率（logprob）评分；

**📊 数据集**

数据集由75句原始英文（来自新闻、书籍、维基百科）以及其225句翻译变体组成，并为每句生成一张对应的 GPT‑5 图像；

**📈 对比分析**

实验通过 Spearman ρ 相关系数评估模型评分与人类评分的契合度，LLM 在无视觉上下文时相关性最高（>0.8），在视觉上下文中出现压缩或波动，整体表现优于早期 DNN，但仍与人类分布差异显著；

**⚠️ 局限性**

局限性包括仅使用英语数据、数据污染风险（尤其是维基百科句子）、仅评估三种文本体裁、未直接比较视觉与文本上下文、对视觉与文本上下文差异机制研究不足以及对闭源模型的可解释性有限。

---

## 2. Rethinking Clause Management for CDCL SAT Solvers

**arXiv ID:** 2602.20829 | [PDF](https://arxiv.org/pdf/2602.20829v1)

**作者:** Yalun Cai `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14171 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出一种完全不依赖LBD的两阶段子句管理机制，针对复杂算术电路验证中的子句质量评估失效问题，改进CDCL求解器的子句裁剪策略。

**💡 创新点**

创新点在于将子句的固有血统（长度）与动态使用模式（BCP与冲突分析频次）完全解耦，在两阶段分别评估，避免了传统LBD混合评估导致的效率下降。

**🔧 技术方法**

技术上采用子句使用计数与周期性衰减的得分机制、基于长度的后续裁剪、以及与kissat和MiniSat现有的冲突间隔调度相结合的动态裁剪阈值。

**📊 数据集**

主要使用60个乘法器等价检查实例（工业案例+SAT竞赛案例）以及2022年SAT竞赛主轨迹基准，另外还对随机3-SAT和密码学实例做了验证。

**📈 对比分析**

通过在kissat和MiniSat中实现该机制，并与原始求解器在上述数据集上对比，取得了平均PAR-2提升约40%、最多5.74倍加速，并在一般基准上保持或提升了解题数量与时间性能。

**⚠️ 局限性**

局限性包括目前仅在算术电路、随机3-SAT和密码学等有限类别上验证，尚未对更广泛的SAT实例进行系统评估；此外，衰减间隔T和裁剪比例等超参数仍需针对不同问题域进行调优。

---

## 3. An interactive enhanced driving dataset for autonomous driving

**arXiv ID:** 2602.20575 | [PDF](https://arxiv.org/pdf/2602.20575v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 4. VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation

**arXiv ID:** 2602.21054 | [PDF](https://arxiv.org/pdf/2602.21054v1)

**作者:** Seongheon Park `[一作]` (University of Wisconsin-Madison), Sharon Li `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为VAUQ的视觉感知不确定性量化框架，用于大视觉-语言模型（LVLM）的自我评估，评估模型输出是否依赖于图像信息；

**💡 创新点**

创新点在于设计Image-Information Score（IS）与无监督核心区域掩码策略，能够无训练、无外部辅助的方式量化视觉证据对预测不确定性的影响；

**🔧 技术方法**

主要技术包括：计算模型的预测熵、利用视觉注意力权重聚合得到核心区域掩码、在掩码前后计算IS并与熵线性组合得到VAUQ分数；

**📊 数据集**

实验使用的公开数据集包括ViLP、MMVet、VisualCoT（VQA）和CVBench（多选）等；

**📈 对比分析**

与8个竞争基线（Perplexity、Verbalized Confidence、SVAR、Contextual Lens、VL-Uncertainty、EigenScore、Semantic Entropy、Chain-of-Embeddings）在四个数据集上对比，VAUQ在所有模型与数据集上均实现最高AUROC，平均提升约10-20%（对比最优方法如VL-Uncertainty提升约21%）；

**⚠️ 局限性**

局限性包括需要手动设置全局超参数α和K，对不同数据集或样本的敏感性尚未解决；此外，VAUQ只是一个辅助的自我评估工具，不能替代完整的安全机制。

---

## 5. OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI Interpretation

**arXiv ID:** 2602.20752 | [PDF](https://arxiv.org/pdf/2602.20752v1)

**作者:** Tian Lan `[一作]` (Renmin University of China), Dingyu Wang `[通讯]` (Peking University)

**通讯引用:** 2020 | [OpenAlex ID](https://openalex.org/A5017472722)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个名为OrthoDiffusion的统一扩散基础模型，专门用于多任务、跨关节的肌肉骨骼系统MRI解释，能够同时完成解剖结构分割和多标签疾病诊断。

**💡 创新点**

创新点在于：①利用自监督的3D扩散预训练学习跨视角（矢状、冠状、轴向）共享的解剖特征；②通过多平面融合和可解释的MPAE模块实现对不同疾病最优视角的自动加权；③实现高度的标签效率与跨关节泛化，无需针对每个关节或任务进行重训练。

**🔧 技术方法**

技术包括：自监督扩散预训练（3D U‑Net denoiser）、多尺度特征提取与自注意池化、特征/标签级融合策略、EHR与MRI的后级融合以及轻量化分割头。整个框架在扩散模型上进行多任务微调。

**📊 数据集**

使用了覆盖30,653例、来自9家临床中心的膝、踝、肩三关节MRI数据：15,948个无标签膝部扫描用于预训练；10,940例膝部带标签的诊断集、1,006例膝部分割集、2,562例踝部诊断集和8,957例肩部诊断集用于下游评估。

**📈 对比分析**

与传统的3D‑U‑Net、UNETR、3D‑ResNet‑18等基线模型相比，OrthoDiffusion在膝部解剖分割上Dice平均提升约5%‑10%，在膝部8种疾病多标签诊断上的宏观AUROC提升至0.91以上，且在10%标签量下保持与完整标签相近的性能；在踝部和肩部的跨关节迁移实验中也实现了显著的AUROC提升。多模态（MRI+EHR）融合进一步提高了诊断精度。

**⚠️ 局限性**

局限性包括：①模型为无条件扩散，缺乏可控生成和数据增强能力；②研究仅聚焦于关节相关的肌肉骨骼MRI，未覆盖其他解剖部位；③未在实时临床工作流中进行前瞻性验证；④扩散预训练对计算资源需求高，训练成本显著；⑤缺乏针对极端异常或罕见疾病的评估。

---

## 6. Sample-efficient evidence estimation of score based priors for model selection

**arXiv ID:** 2602.20549 | [PDF](https://arxiv.org/pdf/2602.20549v1)

**作者:** Frederic Wang `[一作]` (California Institute of Technology), Katherine L. Bouman `[通讯]` (California Institute of Technology)

**通讯引用:** 19108 | [OpenAlex ID](https://openalex.org/A5061668167)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种在使用扩散模型先验的贝叶斯逆问题中估计模型证据的新方法

**💡 创新点**

利用后验采样的中间样本沿后验时间边缘积分，避免需要先验分数或密度，显著降低估计方差并实现高效计算

**🔧 技术方法**

扩散模型、后验采样（DAPS）、Tweedie公式、无偏平方似然估计器、对角协方差近似等

**📊 数据集**

MNIST、SpaceNet、CelebA、GRMHD模拟图、RIAF模拟图、真实M87*黑洞观测数据

**📈 对比分析**

与Naive MC、TI、AIS、SMC、原始DAPS启发式等基线比较，在高维高斯混合、非凸相位检索以及真实M87*观测中，估计误差几乎无偏，性能优于或匹配最强基线，计算时间大幅降低（≈7×）

**⚠️ 局限性**

对OOB测量的路径偏差仍存在；需要手动选择合适的协方差近似；在极端噪声或非常高维下仍需更多样本验证

---

## 7. VISION-ICE: Video-based Interpretation and Spatial Identification of Arrhythmia Origins via Neural Networks in Intracardiac Echocardiography

**arXiv ID:** 2602.20165 | [PDF](https://arxiv.org/pdf/2602.20165v1)

**作者:** Dorsa EPMoghaddam `[一作]` (Rice University), Behnaam Aazhang `[通讯]` (Texas Heart Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建了一套基于ICE视频的AI框架，用于对心律失常进行三分类定位（窦性心律、左侧和右侧起源）。

**💡 创新点**

首次针对ICE影像实现心律失常定位的机器学习方法，并提供了专门标注的多视角数据集与视角级融合策略。

**🔧 技术方法**

采用预训练的3D ResNet‑18骨干网络，结合自定义单通道适配器、数据增强与3D Grad‑CAM可解释性技术实现模型训练与推理。

**📊 数据集**

使用了39名患者的ICE视频数据，共包含四个标准视角（TV、MV、LPV、CT）和三种节律标签（NSR、DIST、PROX），每个患者多次心搏采样。

**📈 对比分析**

通过10折病人层级交叉验证和视角级投票融合，在四个未见患者上平均准确率达到约76%，显著优于随机基线33.3%。

**⚠️ 局限性**

主要局限在样本量不足、患者间差异大导致模型泛化受限，需进一步扩大数据集并探索更鲁棒的网络与迁移学习策略。

---

## 8. OCR-Agent: Agentic OCR with Capability and Memory Reflection

**arXiv ID:** 2602.21053 | [PDF](https://arxiv.org/pdf/2602.21053v1)

**作者:** Shimin Wen `[一作]` (Southwest Minzu University), Ying Cai `[通讯]` (Southwest Minzu University)

**通讯引用:** 3639 | [OpenAlex ID](https://openalex.org/A5100628247)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OCR-Agent，一种基于可控自我反思的多轮迭代自纠框架，结合能力反思与记忆反思，提升大规模视觉语言模型在 OCR 任务中的推理与稳定性。

**💡 创新点**

通过在反思阶段加入能力约束过滤不可执行操作，并在记忆阶段记录历史反思以避免重复，形成双重自我反思机制，实现无训练集成、持续改进。

**🔧 技术方法**

基于大型视觉语言模型（如 RolmOCR/InternVL3），Chain‑of‑Thought 提示、Iterative Self‑Refine、Capability Reflection、Memory Reflection 以及结构化提示模板。

**📊 数据集**

在 OCRBench v2（含英语与中文子集）上进行评估，涵盖识别、提取、推理等多项任务。

**📈 对比分析**

与 Naive、CoT、Self‑Refine、GPT‑4o、Gemini‑Pro 等方法对比；英文子集平均分 51.0，Visual Understanding/Reasoning 分别 79.9/66.5，中文子集平均 54.7，均超过大多数开源模型。

**⚠️ 局限性**

计算开销大，需多轮推理；受限于基础模型的感知与知识；对实时部署不友好；未实现针对任务的动态迭代控制。

---

## 9. International AI Safety Report 2026

**arXiv ID:** 2602.21012 | [PDF](https://arxiv.org/pdf/2602.21012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 10. The Finite Primitive Basis Theorem for Computational Imaging: Formal Foundations of the OperatorGraph Representation

**arXiv ID:** 2602.20550 | [PDF](https://arxiv.org/pdf/2602.20550v1)

**作者:** Chengshuai Yang `[一作]` `[通讯]` (NextGen PlatformAI C Corp), Chengshuai Yang (NextGen PlatformAI C Corp)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文证明任意成像正向模型均可用仅11种原语构成的有向无环图近似，并给出构造算法及最小性证明；同时对非线性现象做结构分类并给出相应实现；

**💡 创新点**

创新点在于提出“有限原语基理论”，提供统一的11个物理原语库覆盖所有临床、科研及工业成像模态（线性、非线性、相对论等），并证明该库最小且足够；将自洽迭代映射拆解为现有线性原语，点非线性归入Transform。

**🔧 技术方法**

技术方法包括：用类型化DAG表示正向模型；将每个阶段分为六类物理过程；构造性证明并给出六个实现引理；利用Lipschitz连续性和算子范数链式误差分析；最优误差上界；并给出扩展协议。

**📊 数据集**

使用公开基准数据集，共31个线性模态（如CASSI、MRI、CT、Compton等）和9个非线性模态（光学相干成像、相位包裹、光声、Raman等），数据来源于各模态的原始参考实现。

**📈 对比分析**

与原始正向模型比较，构造的DAG在相对误差<0.01、节点数≤5、深度≤5内完成；闭包测试中9原语库已通过，加入Scatter后11原语完整通过；表格给出每种模态的误差与节点/深度统计。

**⚠️ 局限性**

限制在于尚未对量子态断层、极高能相对论成像等模态进行实证验证；理论对误差阈值与复杂度界限有严格要求，若出现新的非线性结构或超出原理范畴的模态则需扩展原语库。

---

## 11. Identifying two piecewise linear additive value functions from anonymous preference information

**arXiv ID:** 2602.20638 | [PDF](https://arxiv.org/pdf/2602.20638v1)

**作者:** Vincent Auriau `[一作]` (Artefact Research Center), Marc Pirlot `[通讯]` (Université de Mons)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在匿名匹配查询下，两名决策者的加性分段线性偏好模型（UTA模型）的可辨识性，并给出一套完整的询问程序；

**💡 创新点**

提出单矩形与相邻矩形两类查询作为识别基石，并证明通过有限次数的这些查询即可完全区分两模型；首次把匿名答案与模型辨识结合到UTA框架中；

**🔧 技术方法**

采用几何分析与代数方程求解的组合技术，利用匹配问答设计、线性方程组求解及迭代推导来确定边际斜率与模型归属；

**📊 数据集**

使用电动汽车（续航、价格）作为演示例子进行说明；理论上适用于任意分段线性尺度，无需特定公开数据集；

**📈 对比分析**

论文以理论证明为主，没有实验对比；通过示例说明所需查询数量为 L_i+L_j-1 个矩形，复杂度线性，可在有限步骤内完成辨识；

**⚠️ 局限性**

仅适用于两位决策者、边际必须是分段线性；不考虑答案噪声、假设答案完全正确；未扩展至多于两模型或非线性边际情况。

---

## 12. SurgAtt-Tracker: Online Surgical Attention Tracking via Temporal Proposal Reranking and Motion-Aware Refinement

**arXiv ID:** 2602.20636 | [PDF](https://arxiv.org/pdf/2602.20636v1)

**作者:** Rulin Zhou `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 17297 | [OpenAlex ID](https://openalex.org/A5032340829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `e0540dec-d77f-42db-94ae-d039248f6393` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了SurgAtt-1.16M大规模手术注意力追踪数据集，并提出SurgAtt-Tracker框架，实现基于热图的实时手术视场关注跟踪；

**💡 创新点**

创新点在于把手术关注建模为密集热图，利用提议重排序与运动感知细化，突破传统单一对象或工具指向的局限；

**🔧 技术方法**

采用冻结目标检测器生成高召回提议，跨帧注意力重排序（AS-Rerank）与运动感知自适应细化（MAA-Refine）技术；

**📊 数据集**

使用SurgAtt-1.16M（含SZPH、AutoLaparo、Hamlyn三大子集）作为训练和评估数据；

**📈 对比分析**

与U-Net、回归、跟踪、检测等四类基线对比，SurgAtt-Tracker在SurgAtt-SZPH上NSS、CC、SIM分别提升至2.58/0.871/0.829，MSE/MAE降至0.015/0.051，显著优于现有方法；

**⚠️ 局限性**

局限在于依赖检测器提议覆盖、对长时间遮挡或急剧相机运动的鲁棒性有限，且尚未集成至实时机器人控制场景；

---

## 13. EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer

**arXiv ID:** 2602.20985 | [PDF](https://arxiv.org/pdf/2602.20985v1)

**作者:** Munish Monga `[一作]` (Sony Research), C. V. Jawahar `[通讯]` (IIIT Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Evolving World Object Detection（EWOD）范式，并设计了EW-DETR框架。

**💡 创新点**

创新点在于结合增量LoRA适配器、查询归一化物体性适配器和熵感未知混合三模块，实现无示例、跨域、未知检测的统一解决方案。

**🔧 技术方法**

使用DETR/Deformable-DETR骨干、LoRA低秩适配、查询归一化、熵混合校准以及FOGS综合评估指标。

**📊 数据集**

在Pascal Series（VOC、Clipart、Watercolor、Comic）和Diverse Weather（晴天、夜晚、雨天、雾天等）两个自定义增量+开放域数据集上训练与测试。

**📈 对比分析**

与OWOBJ、DuET、OW-DETR、PROB、CAT、ORTH、ORE等现有方法对比，EW-DETR在FOGS评分上提升约57.24%，在保留、开放性和泛化三维上均居首位。

**⚠️ 局限性**

局限性包括对任务顺序和样本比例的敏感性，且在极端域移位或类别极不平衡时仍可能出现性能下降。

---

## 14. GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization

**arXiv ID:** 2602.20427 | [PDF](https://arxiv.org/pdf/2602.20427v1)

**作者:** Yaohui Cai `[一作]` (Cornell University), Zhiru Zhang `[通讯]` (Cornell University)

**通讯引用:** 6913 | [OpenAlex ID](https://openalex.org/A5037210004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于高斯重参数化的可微分调度框架，用来高效优化大规模操作调度问题。

**💡 创新点**

创新点在于用连续高斯分布代替传统的类别分布，将调度变量从 O(ND) 降到 O(N)，天然捕捉时间序列顺序，并首次支持流水线（模数）调度。

**🔧 技术方法**

核心技术包括高斯重参数化、期望约束表达式、增量拉格朗日方法、GPU 并行向量化以及采样后轻量级贪心合法化。

**📊 数据集**

使用 EPFL Benchmark Suites、Synthetic Random Workloads (RW) 以及作者自行扩展的流水线基准进行实验。

**📈 对比分析**

与商业 ILP 求解器 (CPLEX/Gurobi)、常用启发式 (List Scheduling, FDS) 以及早期可微方法做对比。该方法在 15 分钟内在大规模图上得到与最优相当或更优的 Pareto 前沿，速度提升 1–2 个数量级，显著避免 OOM。

**⚠️ 局限性**

局限性在于将每个操作视为独立高斯变量，未建模节点间的相关性，可能导致次优解；引入全局相关性的 GP 需要 O(N²) 空间，难以规模化；收敛稳定性和解的最优性仍有提升空间。

---

## 15. Topology-Aware Integrated Communication, Sensing, and Power Transfer for SAGIN

**arXiv ID:** 2602.20908 | [PDF](https://arxiv.org/pdf/2602.20908v1)

**作者:** Han Yu `[一作]` (Technical University of Berlin), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了基于拓扑感知的SAGIN多功能集成感知、通信与无线功率传输（ISCPT）优化框架，并将问题转化为可求解的混合整数线性规划（MILP）。

**💡 创新点**

首次利用可见性与信道强度两条拓扑准则构建双部图，提出用户/基站激活与感知基站选择的联合优化模型，实现对通信、感知与能量传输的多目标协同控制。

**🔧 技术方法**

采用拓扑感知、双部图匹配、最大匹配/最小约束、MRT预编码以及MILP优化等技术。

**📊 数据集**

使用基于L0卫星星座轨道、地面城市坐标以及随机生成的卫星/地面用户分布的仿真数据集。

**📈 对比分析**

与贪心用户选择和无选择基准对比；实验结果表明TA-ISCPT在通信速率上明显优于基准，感知SINR虽略低于贪心但显著优于无选择，功率传输损失仅约5 dBm。

**⚠️ 局限性**

仅考虑单时隙静态拓扑，未研究多时隙公平调度与动态网络拓扑；预编码仅采用MRT，缺乏对更复杂预编码方案的评估。

---

## 16. Real-time Motion Segmentation with Event-based Normal Flow

**arXiv ID:** 2602.20790 | [PDF](https://arxiv.org/pdf/2602.20790v1)

**作者:** Sheng Zhong `[一作]` (Hunan University), Yi Zhou `[通讯]` (Hunan University)

**通讯引用:** 6459 | [OpenAlex ID](https://openalex.org/A5046991303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于事件相机正向流（normal flow）的实时运动分割框架，利用图割优化和迭代正向流聚类与运动模型拟合实现对独立运动物体（IMO）的识别与分割。

**💡 创新点**

创新点在于：①利用密集正向流作为中间表示，显著降低了数据量与计算复杂度；②提出了快速采样与运动预测的初始化策略，仅需少量候选模型；③在能量最小化框架下实现正向流的高效拟合与分割。

**🔧 技术方法**

核心技术包括：事件相机正向流估计（VecKM_Flow），Delaunay 三角剖分构建空间图，图割（alpha-expansion）求解多模型能量最小化，Levenberg-Marquardt 非线性优化用于正向流运动模型拟合。

**📊 数据集**

使用了三大公开数据集：EED、EVIMO 与 EMSGC（室外序列），以及自采集的短时序列用于时延分析。

**📈 对比分析**

与 EMSGC 等传统运动补偿方法以及两种经典算法（EMSMC、EMSGC）相比，本文方法在检测率、IoU 等指标上保持或提升了性能，同时实现了约 800 倍的速度提升，能够在 30 Hz 以上的实时率下运行。

**⚠️ 局限性**

局限性在于：依赖高质量的正向流估计，对极端光照或噪声条件下的鲁棒性有限；当前仅使用仿射运动模型，对非刚性或变形对象的分割效果尚待改进。

---

## 17. A Morton-Type Space-Filling Curve for Pyramid Subdivision and Hybrid Adaptive Mesh Refinement

**arXiv ID:** 2602.20887 | [PDF](https://arxiv.org/pdf/2602.20887v1)

**作者:** David Knapp `[一作]` (University of Cologne), Carsten Burstedde `[通讯]` (Rheinische Friedrich-Wilhelms-Universität Bonn)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种适用于金字塔单元的空间填充曲线（SFC）和相应的细化/粗化、邻域、分区等低层与高层算法，实现了在三维混合网格中动态自适应网格细化（AMR）的完整框架。

**💡 创新点**

创新点在于：①设计了将金字塔单元细化为六个金字塔和四个四面体的分裂规则，并与立方体细化可交换；②基于该规则构造了唯一且可线性化的SFC索引；③将此SFC与森林-树结构统一，提供了完整的并行重划分和幽灵层交换算法。

**🔧 技术方法**

使用了Morton型SFC、树形细化与粗化算法、MPI并行通信、基于叶节点的线性八叉树存储、以及对不同单元形状的面/顶点索引表等技术。

**📊 数据集**

主要使用的测试数据集为：①单一飞机翼几何的混合网格（约10万棵树，包含三角形、四面体、六面体、金字塔等），②通过5次细化后在移动的虚拟壁面附近再次细化，最终生成约40亿个元素；②在统一级别细化的基准实验中生成几何级别为7-10的金字塔/其他单元（均约1.5亿个元素）在多台高性能计算机（CARA、CARO）上跑。

**📈 对比分析**

通过与已有的四面体、六面体、棱柱单元的实现进行对比，采用弱/强扩展实验；结果显示金字塔单元的核心算法与传统单元的性能相近，几乎保持理想扩展，唯一略逊是当每个进程元素数低时通信开销略增，但整体性能仍可接受。

**⚠️ 局限性**

局限性包括：①金字塔相关算法比单一形状的更复杂，导致实现和调试成本上升；②在极细粒度分布（每进程仅数十万元素）时，金字塔的通信模式可能产生较高开销；③目前尚未与实际求解器（如CFD、弹性求解器）集成，实际应用中仍需进一步验证。

---

## 18. Training-Free Multi-Concept Image Editing

**arXiv ID:** 2602.20839 | [PDF](https://arxiv.org/pdf/2602.20839v1)

**作者:** Niki Foteinopoulou `[一作]` (Cambridge Research Laboratory, Toshiba Europe), Stephan Liwicki `[通讯]` (Cambridge Research Laboratory, Toshiba Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练‑free 的概念基础图像编辑框架，将优化的 Delta Denoising Score（Optimised DDS）与多 LoRA 组合结合，实现对多视觉概念的可控编辑。

**💡 创新点**

创新点在于：① 将 DDS 的时间步排序、正则化和负提示引导与 LoRA 组合统一到一次性推理流程；② 用 LoRA 预测的特征相似度进行空间加权组合，避免多概念冲突；③ 通过上述改进显著提升身份保持和细节一致性。

**🔧 技术方法**

使用技术包括：Stable Diffusion、SDS、DDS、PDS、时间步排序、正则化、负提示（classifier‑free guidance）、LoRA低秩适配器、多尺度软最小化组合。

**📊 数据集**

使用的数据集：InstructPix2Pix（用于零样本编辑）和 ComposLoRA（22 个预训练 LoRA，涵盖角色、服装、风格、背景等），以及 stable‑diffusion‑v1.5 checkpoint、Realistic_Vision_V5.1、Counterfeit‑V2.5。

**📈 对比分析**

通过与 DiffusionClip、PnP、InstructPix2Pix、DDS、PDS 等基线比较，InstructPix2Pix 上 CLIPScore 由 0.298 提升到 0.308（显著），LPIPS 维持低水平；在 ComposLoRA 的多概念编辑中，LPIPS 更低、CLIPScore 与 baseline 持平或略高；GPT‑4V 与人工评估显示本方法的胜率最高，平均排名 1.90/38%。

**⚠️ 局限性**

限制：① 计算成本随 LoRA 数量线性增长，交互性受限；② 受 LoRA 质量和相互对齐影响，组合时可能偏向某些 Adapter；③ 基模型的固有限制导致姿态/表情变化时出现手臂复制、表情冲突等缺陷。

---

## 19. Efficient Solvers for Coupling-Aware Beamforming in Continuous Aperture Arrays

**arXiv ID:** 2602.20599 | [PDF](https://arxiv.org/pdf/2602.20599v1)

**作者:** Geonhee Lee `[一作]` (Korea Advanced Institute of Science and Technology), Junil Choi `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5665 | [OpenAlex ID](https://openalex.org/A5065248740)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对连续光幕天线（CAPA）中电磁互耦导致的波束成形数值问题，提出了两种高效求解器；

**💡 创新点**

创新点在于（1）采用极坐标变换与三角换元的极坐标-三角核近似（PKA），有效消除积分核在边界处的奇异性；（2）基于Nyström离散化的直接LU分解求解器，提供数值稳定且可预先因式分解的快速解法；

**🔧 技术方法**

主要技术包括Gauss–Legendre数值积分、极坐标变换、三角换元、核近似、Nyström方法、LU分解与部分列主元；

**📊 数据集**

实验使用仿真数据：矩形平面天线（Lx=Ly=0.5 m）、表面阻抗Zs=0.0128 Ω、远场LoS信道，接收机距离R0=50 m；

**📈 对比分析**

与传统Cartesian GL积分的核近似（KA）和共轭梯度（CG）方法比较，PKA在M≈28点即可达到KA所需的M≈44点；LU求解器在频率2–8 GHz下，总运行时间≈1.9 s保持不变，而CG总时间从18.5 s增长到58.4 s，显示出更高的速度和可扩展性；

**⚠️ 局限性**

局限性包括：LU方法需预先因式分解，适用于固定天线几何和频率；极坐标近似在非圆形天线或复杂边界时可能需进一步推广；并且在极高频率或极大天线尺寸下矩阵仍可能变得非常大，导致存储和计算挑战。

---

## 20. OpenPort Protocol: A Security Governance Specification for AI Agent Tool Access

**arXiv ID:** 2602.20196 | [PDF](https://arxiv.org/pdf/2602.20196v1)

**作者:** Genliang Zhu `[一作]` (Accentrust), Qiang Li `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 31308 | [OpenAlex ID](https://openalex.org/A5100688318)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了 OpenPort 协议，为 AI 代理提供治理优先的工具暴露接口，支持授权、风险控制、草稿审核、审计与速率限制。

**💡 创新点**

创新点在于将权限、风险分层、草稿审查、预检哈希、状态证据以及可验证的不变式整合进协议规范，实现安全、可审计且可测试的代理工具调用。

**🔧 技术方法**

技术方案基于 REST/JSON 接口、Bearer/JWT 认证、ABAC 策略、草稿/执行流水线、哈希签名、幂等键、率限制器和结构化审计事件流。

**📊 数据集**

实验验证使用合成多租户数据集，未公开真实业务数据。

**📈 对比分析**

通过黑盒合规测试、负面安全测试和模糊测试验证协议不变式；在基准部署下可处理数百请求/秒，性能满足常规 API 负载，未与其他协议做直接性能对比。

**⚠️ 局限性**

局限包括不防御提示注入、假设管理员可信、示例实现缺乏多节点持久化、未实现 PoP 绑定或签名审计，且风险标签需管理员手工配置。

---

## 21. LUTstructions: Self-loading FPGA-based Reconfigurable Instructions

**arXiv ID:** 2602.20802 | [PDF](https://arxiv.org/pdf/2602.20802v1)

**作者:** Philippos Papaphilippou `[一作]` (University of Southampton), Philippos Papaphilippou `[通讯]` (University of Southampton)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5014027604)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

在 RISC‑V softcore 上实现了一种自定义的 FPGA 架构（LUTstruction），实现了动态加载可重构指令的“FPGA‑on‑an‑FPGA”设计，能够在高频率下以低重配置延迟执行任意自定义指令；

**💡 创新点**

创新点包括：① 采用 LUT4_4 单元并使用斜向路由实现无后向连线的数据流结构；② 通过在每隔 S 列插入必备寄存器实现寄存器置换（register placement）和流水线化；③ 引入并行配置（configuration parallelism）和专用 bitstream 缓存（bitstream cache），从而显著降低重配置时延；④ 将所有逻辑实现为组合式、无内部状态，确保指令实现的可预期性；

**🔧 技术方法**

技术手段涵盖：RISC‑V 自定义指令、LUT4_4 fabric、寄存器置换与并行配置优化、VTR + 自定义路由器、BLIF 交互、bitstream 生成与交换、AXI 总线与 BL1 缓存、在 ZU3EG 与 Alveo V80 上的 FPGA 合成与验证、ASIC 放样与 7nm、ASIC‑on‑ASIC 评估；

**📊 数据集**

使用的“数据集”为基于 STREAM 的微基准循环，包括 popcount、位移/异或等自定义指令的 8 KiB bitstream，实验全部基于合成的自定义指令和软件实现，无外部真实数据集；

**📈 对比分析**

通过与纯软件实现对比，测得软指令在 popcount 上提升 2.55×、在位移/异或上提升 13.4×、在双指令交错上提升 2.86×；重配置延迟比现有最快控制器快 28×；在 Alveo V80 上实现 1.05 GHz，ASIC 在 1–2 GHz 范围；整体占用资源与硬件指令相近，性能接近非可编程实现；

**⚠️ 局限性**

局限性包括：仅支持无状态（stateless）指令，需手动维护 bitstream 缓存；重配置时延仍在数百周期，频繁上下文切换时可能成为瓶颈；寄存器置换与并行配置带来额外 FF/LUT 开销；当前实现仅验证于 ZU3EG 与 V80，缺乏大规模多核/高并行度的评估；不支持自适应/可变复杂度指令；

---

## 22. CaDrift: A Time-dependent Causal Generator of Drifting Data Streams

**arXiv ID:** 2602.20329 | [PDF](https://arxiv.org/pdf/2602.20329v1)

**作者:** Eduardo V. L. Barboza `[一作]` (École de Technologie Supérieure), Rafael M. O. Cruz `[通讯]` (École de Technologie Supérieure)

**通讯引用:** 2041 | [OpenAlex ID](https://openalex.org/A5019553116)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 CaDrift，一种基于结构因果模型的时序合成数据流生成器，能够控制多种概念漂移事件并产生非 i.i.d 的高阶因果关系数据。

**💡 创新点**

创新点在于将 SCM 与 EWMA、AR 噪声、do-算子干预相结合，首次实现可调节漂移强度、频率、类型（分布、协变量、严重、局部）且不依赖源数据的时间序列生成；同时提供了可捕捉真实序列自相关的生成机制。

**🔧 技术方法**

采用结构因果模型（SCM）、小型神经网络映射函数、指数加权滑动平均（EWMA）、自回归（AR）噪声、干预操作（do-算子）、最大均值差异（MMD）与 Ljung‑Box 检验等技术来构建与评估生成器。

**📊 数据集**

使用 CaDrift 自己生成的 8 组数据集（涵盖 5–200 维、不同漂移类型）以及公开漂移基准 SEA、Sine、RandomRBF；全部为合成时间序列数据流。

**📈 对比分析**

与 TabPFN^Stream、ARF、LevBag、HT、OAUE、LAST 等基准方法对比；在 CaDrift 生成的多类型漂移上多数模型准确率显著下降并恢复缓慢，说明生成器具备较高挑战性；相较公开基准准确率更低，进一步证明其难度更大。

**⚠️ 局限性**

局限性包括：仅在分类任务中评估（回归实验留在附录）；漂移事件需手工设定，缺少对真实数据的验证；映射函数采用简化的小型神经网络，可能无法完全捕捉复杂真实因果结构。

---

## 23. MIP Candy: A Modular PyTorch Framework for Medical Image Processing

**arXiv ID:** 2602.21033 | [PDF](https://arxiv.org/pdf/2602.21033v1)

**作者:** Tianhao Fu `[一作]` (University of Toronto), Yucheng Chen `[通讯]` (Project Neura)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 MIP Candy，一个 PyTorch‑native、模块化的医学影像分割框架，提供从多格式数据读取、预处理、训练（含深度监督、EMA、验证得分预测等）到推理和评估的完整流水线，支持一行代码快速启动

**💡 创新点**

核心创新包括 LayerT 的延迟配置机制实现网络层可在运行时替换；内置训练透明度（实时曲线、最差案例预览、验证得分预测）；自动化数据集检查与 ROI 基于采样；以及 bundle 生态系统实现模型、训练器、预测器的无缝打包与扩展

**🔧 技术方法**

技术栈基于 PyTorch、SimpleITK、safetensors、Rich、Weights & Biases/Notion/MLflow；实现 LayerT 延迟模块配置、K‑fold CV、深度监督、EMA、预测预览、斜率回归（quotient regression）预测最大得分与最佳 epoch 等功能

**📊 数据集**

示例实验使用 PH2 皮肤病变二分类数据集进行 2D 分割，BraTS 2021 脑肿瘤多分类数据集进行 3D 分割；框架支持 NIfTI、DICOM、MHA 等医学影像格式

**📈 对比分析**

与 nnU‑Net、MONAI、TorchIO 等现有工具对比，MIP Candy 在功能完整度与模块化之间取得折中；实验结果显示在 PH2 与 BraTS 上，Dice 分数与 nnU‑Net 接近甚至略优，同时提供完整的训练曲线、最差案例可视化和验证得分预测，显著提升实验可解释性

**⚠️ 局限性**

局限性包括：目前尚未实现滑动窗口推理、表面距离指标（Hausdorff 等）；缺乏半监督/自监督学习支持；仅支持单 GPU 训练；对极大体积的推理与资源受限环境的适配仍待完善

---

## 24. Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning

**arXiv ID:** 2602.20722 | [PDF](https://arxiv.org/pdf/2602.20722v1)

**作者:** Xu Wan `[一作]` (Zhejiang University), Mingyang Sun `[通讯]` (Peking University)

**通讯引用:** 4902 | [OpenAlex ID](https://openalex.org/A5079378336)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了BAPO，一个离线RLVR框架，通过动态构建训练批次来提升大型语言模型（LLM）后训练的样本效率和收敛速度。

**💡 创新点**

创新点在于难度感知的经验回放机制：先对历史难样本进行再评估以获取潜在可改进样本，再利用高质量历史样本重用；同时通过自适应阈值动态构造批次，解决奖励同质化和经验浪费问题。

**🔧 技术方法**

主要技术包括：RLVR/GRPO基础算法、延迟滚动策略（v>0）、重要性采样与KL约束、FIFO缓冲区、动态难度阈值与自适应批次构造、奖励标准化与优势估计。

**📊 数据集**

在数学、规划与视觉几何三类推理任务上使用DeepSeek R1 Distilled 1.5B、Qwen3 8B、Qwen2.5 Math/ VL 3B/7B等模型，数据集涵盖DeepScaleR-Preview、AIME24、AMC23、MATH500、Minerva Math、OlympiadBench、Countdown-34、Geometry3K等。

**📈 对比分析**

与GRPO、DAPO、RePO、Remix-GRPO等基线对比，BAPO平均提升12.5%准确率，成功解决了原模型难以解决的40.7%问题，收敛更平稳且在相同计算预算下样本利用率更高。

**⚠️ 局限性**

局限性包括尚未在大规模 MoE 结构或 agentic RL 场景中验证，且对超参数（如延迟、阈值）的鲁棒性和超参数空间探索仍有待进一步研究。

---

## 25. Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs

**arXiv ID:** 2602.20799 | [PDF](https://arxiv.org/pdf/2602.20799v1)

**作者:** Guangsheng Ou `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 34331 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段的持续预训练与监督微调框架（CPT+SFT），通过构建源代码的代码图进行“推理感知”的数据合成，以提升LLM在未见代码库上的代码生成性能。

**💡 创新点**

创新点在于：①利用代码图捕捉文件、类、函数等实体之间的依赖关系；②设计依赖保留的CPT数据和三类带推理轨迹的SFT数据（单跳关系、API组合、代码库使用）；③提出新的针对未见代码库的评测基准，并在多语言、多规模、多架构模型上验证。

**🔧 技术方法**

主要技术包括：程序分析构建代码图、深度优先遍历生成文件级依赖序列、生成带推理轨迹的合成数据、使用大语言模型进行思维链与推理验证、混合通用与领域数据进行CPT与SFT训练。

**📊 数据集**

使用的代码库数据为四个公开新兴项目（C++：sqlgen、reaction、Hexi；Python：Leann），以及内部测试用例；基准数据来自自动生成并人工校验的任务集，包含多语言、多代码库的评测实例。

**📈 对比分析**

与OSS‑Instruct、COTTON等基于数据合成的基线以及RAG检索增强方法进行对比。实验显示，所提方法在Pass@1上平均提升7.2%–26.1%，在CPT+SFT组合下单模型即可在多语言、多代码库场景下达36% pass@1，显著优于所有基线。

**⚠️ 局限性**

局限性包括：仅验证两种语言（C++/Python），未覆盖其他主流语言；训练配置未进行全范围调优；在极大规模代码库或极深依赖链时，DFS生成的样本可能仍出现切分不完全的问题。

---

## 26. Enhancing Heat Sink Efficiency in MOSFETs using Physics Informed Neural Networks: A Systematic Study on Coolant Velocity Estimation

**arXiv ID:** 2602.20177 | [PDF](https://arxiv.org/pdf/2602.20177v1)

**作者:** Aniruddha Bora `[一作]` (Brown University), Chryssostomos Chryssostomidis `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5080611073)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用物理信息神经网络（PINNs）对多层 MOSFET 冷却结构进行逆向建模，求解满足给定进出口温度和热通量的冷却液流速。

**💡 创新点**

提出了分层顺序训练的 PINN 框架，将热传递系数作为可学习参数，并通过能量守恒和自适应权重约束实现高精度逆向求解。

**🔧 技术方法**

采用 PINNs、自动微分、分层分解训练、自适应权重以及能量守恒约束，解决稳态多层热传导逆问题。

**📊 数据集**

使用实验测得的多功率、多进出口温度下的 MOSFET 冷却实验数据和一个可解析的单层热传导基准问题作为数据集。

**📈 对比分析**

通过与实验测得的流速和温度进行对比，误差均维持在 0.5%–2% 之间；在解析基准案例中，热传递系数误差低于 0.6%，验证了方法的高精度。

**⚠️ 局限性**

仅考虑稳态热传导，未显式求解流场；训练对初始参数敏感，需要大量残差点；多层接口处数值误差可能累积，对真实动态流场的推广有限。

---

## 27. Implicit Intelligence -- Evaluating Agents on What Users Don't Say

**arXiv ID:** 2602.20424 | [PDF](https://arxiv.org/pdf/2602.20424v1)

**作者:** Ved Sirdeshmukh `[一作]` (Applied Machine Learning Research), Marc Wetter `[通讯]` (Applied Machine Learning Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Implicit Intelligence 评估框架与 Agent-as-a-World（AaW）模拟环境，衡量 AI 代理在隐式需求（隐式推理、灾难风险、隐私安全、无障碍）下的真实目标实现能力。

**💡 创新点**

创新点在于将隐式需求拆解为四类、使用可声明的 YAML 场景与 LLM 作为统一世界模拟器，消除传统手工仿真难度，同时通过环境交互可探测隐式约束；并构建了覆盖 205 个隐式场景的评测数据集。

**🔧 技术方法**

核心技术包括：LLM 驱动的世界模拟器（Claude Opus 4.5），基于 YAML 的场景描述与执行规则，LLM 评估器用于依据规则自动判定通过/失败，以及多轮交互协议。

**📊 数据集**

使用的主要数据集是 205 个基于 iOS Shortcuts 与 PersonaHub 人物设定生成的场景，涵盖 300+ 原生 iOS 行为和四类隐式需求；场景通过人工与自动迭代验证后收集。

**📈 对比分析**

通过在 205 场景上对 16 个前沿与开源模型进行评估，报告 Scenario Pass Rate (SPR) 与 Normalized Scenario Score (NSS)。最高模型 GPT‑5.2‑pro 仅 48.3% 的 SPR，表明即使最先进的模型在隐式需求上仍显不足；各类别性能差异明显，灾难风险与隐私安全各模型表现不一。

**⚠️ 局限性**

局限性包括：场景构造受作者视角影响，可能不覆盖所有文化/技术背景；依赖 iOS Shortcuts，随系统更新场景有效性下降；使用 LLM 作为世界模拟器可能在极端不一致情况下引入偏差；评估未涉及主动澄清请求的能力。

---

## 28. BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting

**arXiv ID:** 2602.21105 | [PDF](https://arxiv.org/pdf/2602.21105v1)

**作者:** Jiaxing Yu `[一作]` (Nanjing University), Yanwen Guo `[通讯]` (Nanjing University)

**通讯引用:** 4397 | [OpenAlex ID](https://openalex.org/A5009275869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

从多视角图像直接通过2D高斯展开+对比学习得到边缘和补丁特征，随后进行参数化表面拟合和B-rep组装，完成CAD模型重建。

**💡 创新点**

首次利用可学习的2D Gaussian Splatting两阶段训练，结合对比学习实现边缘与补丁特征分离，直接从图像恢复完整的B-rep CAD，无需点云监督。

**🔧 技术方法**

2D Gaussian Splatting、对比学习、RANSAC参数化拟合、约束导向的原语拟合与拓扑优化，以及SAM进行补丁掩码提取。

**📊 数据集**

在ABC-NEF子集（含50张视角图像的ABC数据集）和ABO真实场景数据集上进行训练与测试。

**📈 对比分析**

与基于点云的SegNet、ParSeNet、PCER-Net、SED-Net、Point2CAD、Split-and-Fit等方法对比，Patch/Edge分割Precision/Recall/F1超过对手，CAD重建的Chamfer/Hausdorff距离略高于Point2CAD，但生成模型更紧凑、完整。

**⚠️ 局限性**

对视角数量依赖强，需要至少30-50张视图；对低纹理或遮挡图像的补丁掩码提取易碎；目前仅支持平面、圆柱、球面等基础原语，复杂曲面仍有挑战。

---

## 29. GATES: Self-Distillation under Privileged Context with Consensus Gating

**arXiv ID:** 2602.20574 | [PDF](https://arxiv.org/pdf/2602.20574v1)

**作者:** Alex Stein `[一作]` (University of Maryland), Tom Goldstein `[通讯]` (University of Maryland)

**通讯引用:** 13852 | [OpenAlex ID](https://openalex.org/A5060687985)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在没有验证标签、奖励或外部评审的条件下，利用文档辅助的问答场景进行自蒸馏；模型既是拥有文档的导师，又是仅靠问题回答的学生，采用多条推理轨迹的共识来判断监督可靠性，并将可信轨迹的完整推理过程蒸馏给无文档学生。

**💡 创新点**

创新点在于将“共识门控”（consensus gating）引入自蒸馏：通过多次导师生成的答案一致性来动态筛选可靠的训练样本，仅在共识强时才进行轨迹级别的密集蒸馏，从而在不依赖外部标注或奖励的前提下实现有效的自我提升。

**🔧 技术方法**

核心技术包括：多路导师推理采样、答案提取与共识判定、基于共识的门控机制、离线轨迹蒸馏（off‑policy）和在线轨迹蒸馏（on‑policy）两种模式的密集损失，以及对文档泄漏的防护。

**📊 数据集**

使用 Qwen3‑4B‑Base 作为基础模型，在 Nemotron‑CC‑Math 语料库上通过 Qwen2.5‑32B‑Instruct 预生成 551 条文档‑问题对（每条文档对应一个问题）作为固定挑战者；评估数据包括 50 条留出测试集以及四个公开无文档数学基准（MATH、AMC、Minerva、OlympiadBench）。

**📈 对比分析**

与基线（答案仅 SFT、奖励基 RL、导师轨迹 SFT 等）对比，门控蒸馏方法在留出测试集上将学生准确率从 46% 提升至 62%，在四个无文档基准上的平均 maj@8 准确率从 20.2% 提升至 35.4%，显著优于所有对比方法。

**⚠️ 局限性**

局限性包括：共识仅为可靠性的近似，若导师模型本身欠缺或产生单一错误易导致共识误导；门控会丢弃部分样本，降低样本利用率；要求答案可被准确提取并标准化，适用性受限；多路采样增加训练成本；方法目前仅验证于文档辅助的数学问答，扩展到更广泛任务时需重新评估共识阈值与泄漏控制。

---

## 30. Efficient Hierarchical Any-Angle Path Planning on Multi-Resolution 3D Grids

**arXiv ID:** 2602.21174 | [PDF](https://arxiv.org/pdf/2602.21174v1)

**作者:** Victor Reijgwart `[一作]` (Autonomous Systems Lab), Lionel Ott `[通讯]` (Autonomous Systems Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于多分辨率八叉树的任何角度路径规划方法，结合Theta*与层次搜索实现高效全局规划。

**💡 创新点**

将任何角度成本场压缩到多分辨率立方体；在障碍附近初始化折角点；采用动态细化与可接受误差阈值的层次搜索。

**🔧 技术方法**

Theta*、LazyTheta*、八叉树多分辨率表示、成本场压缩、动态细化、欧几里得直线距离及启发式搜索。

**📊 数据集**

合成10^3体素地图（0–4000障碍）以及Newer College Dataset的四个真实环境（Mine、Cloister、Math、Park）。

**📈 对比分析**

与固定分辨率A*、Theta*、LazyTheta*、OctreeLazyTheta*以及RRTConnect、RRT*（不同时间预算）进行成功率、路径长度和运行时间对比；成功率相同，路径长度与Theta*相当或更优，运行时间比Theta*快1–2个数量级，在大面积环境中甚至比RRTConnect快10倍以上。

**⚠️ 局限性**

仅适用于欧几里得成本，无法处理动态障碍或运动约束，且不适用于非欧几里得空间。

---

## 31. High-Dimensional Robust Mean Estimation with Untrusted Batches

**arXiv ID:** 2602.20698 | [PDF](https://arxiv.org/pdf/2602.20698v1)

**作者:** Maryam Aliakbarpour `[一作]` (Rice University), Junze Yin `[通讯]` (Rice University)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5001116446)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在不可信批次环境下的高维均值估计问题，提出了在存在恶意用户和统计异质性的情况下恢复真实分布均值的算法。

**💡 创新点**

创新点在于引入了双重腐败模型，考虑了用户级和样本级的腐败，提出了基于平方和（SoS）的方法来应对这些挑战，并证明了算法的最小最大最优误差率。

**🔧 技术方法**

使用了平方和（SoS）算法来设计和分析均值估计的鲁棒性。

**📊 数据集**

使用了来自N个用户的批次数据，每个用户提供n个样本，具体数据集未详细说明。

**📈 对比分析**

与现有方法相比，提出的算法在高维情况下能够有效处理用户和样本的腐败，性能达到最小最大最优误差率O(√(/n) + √(d/nN) + √(α))，显示出在批次结构下，恶意用户的影响被抑制。

**⚠️ 局限性**

限制在于算法对腐败程度的假设，实际应用中可能无法准确知道腐败的比例，可能影响算法的表现。

---

## 32. KCFRC: Kinematic Collision-Aware Foothold Reachability Criteria for Legged Locomotion

**arXiv ID:** 2602.20850 | [PDF](https://arxiv.org/pdf/2602.20850v1)

**作者:** Lei Ye `[一作]` (Harbin Institute of Technology), Liang Ding `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 5678 | [OpenAlex ID](https://openalex.org/A5012020883)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于运动学和碰撞感知的足部可达性判定方法（KCFRC），并将其嵌入到多腿机器人接触规划与轨迹优化流程中。

**💡 创新点**

创新点在于：①给出足部可达性问题的充分条件（通过引入引导面和辅助面及拓扑连通性判断）；②将可达性判断转化为在二维网格上寻找连通边界并构造可视图，从而实现微秒级的实时判定；③在保持高准确率的同时，提供两种速度/精度权衡的实现（keypoint‑weighted 与 convolutional 指引面）。

**🔧 技术方法**

使用的技术包括：离散网格地图与签名距离场（SDF）、球形碰撞体检测、Pinocchio 与 IKFast 进行运动学/逆运动学计算、SE(3) 插值构造姿态插值任务域、可视图/最短路径算法、以及 GPU‑加速可选的SDF预计算。算法实现开放源码（GitHub）。

**📊 数据集**

在仿真中使用随机凸多面体通道与分形噪声障碍的网格地图；在真实实验中采用 ElSpider 4 Air 六足机器人和 Unitree A1 四足机器人，构建 2.5 cm 分辨率的高度图并采集 LiDAR 与 OptiTrack 位姿数据。实验涵盖密集、稀疏与狭窄环境（障碍、隧道、阶梯、木桩、孔洞）。

**📈 对比分析**

与 RRT‑Connect、STOMP、FEC 以及 Exhaustive‑RRT 基线进行对比。KCFRC（关键版）平均 27–51 ms、准确率 99.4–99.8%、召回率 98.2–98.7%；KCFRC（卷积版）平均 1.6–2.6 ms、准确率 97.8–99.7%、召回率 93.4–97.8%。相比之下，RRT‑Connect 需要 90–587 ms，FEC 召回率仅 76.5–90%。KCFRC 速度提升 100–400×，保持与基线相近的可靠性。

**⚠️ 局限性**

局限性包括：①依赖离散网格导致边界附近可行点被误判为不可行（false negative）; ②在极窄通道、多模态解空间或复杂地形特征下，引导面平滑可能遗漏可行路径；③当前不考虑时间维度的可达性约束；④对网格分辨率敏感，低分辨率时精度下降。未来改进方向是引入时间维度、改进边界搜索算法、以及在更动态的控制框架中部署。

---

## 33. Ski Rental with Distributional Predictions of Unknown Quality

**arXiv ID:** 2602.21104 | [PDF](https://arxiv.org/pdf/2602.21104v1)

**作者:** Qiming Cui `[一作]` (Johns Hopkins University), Michael Dinitz `[通讯]` (Johns Hopkins University)

**通讯引用:** 1165 | [OpenAlex ID](https://openalex.org/A5049161363)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在算法预测框架下，研究带分布预测的滑雪租赁问题，并设计了同时具备一致性与鲁棒性的算法。

**💡 创新点**

提出在不知预测误差的前提下，利用地球移动距离(EMD)度量预测分布与真实分布的距离，给出最优一致性/鲁棒性折中，并证明该折中是 Pareto 最优的。

**🔧 技术方法**

使用分布式预测、EMD 量度、最优转移规划、截断与延迟技术以及几何级数分析等理论工具。

**📊 数据集**

无实际数据集，全部为理论分析与仿真验证。

**📈 对比分析**

与经典 2‑竞争滑雪租赁算法、点预测方法及之前的鲁棒优化结果比较，证明在预测误差 0 时加性损失 O(√b)，误差无限大时 O(b log b)，且无法进一步改进。

**⚠️ 局限性**

局限在于仅适用于滑雪租赁类租买问题，且对更复杂的预测分布形式或多决策问题的推广尚未探讨。

---

## 34. LUMEN: Longitudinal Multi-Modal Radiology Model for Prognosis and Diagnosis

**arXiv ID:** 2602.21142 | [PDF](https://arxiv.org/pdf/2602.21142v1)

**作者:** Zhifan Jiang `[一作]` (Sheikh Zayed Institute for Pediatric Surgical Innovation, Children's National Hospital), Marius George Linguraru `[通讯]` (Sheikh Zayed Institute for Pediatric Surgical Innovation, Children's National Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了名为 LUMEN 的多图多任务视觉-语言模型，对胸部 X 光进行诊断与预后问答，并通过 LLM 生成更长、更自然的回答及专家模型预测增强训练集。

**💡 创新点**

创新点包括：①将纵向（时间）影像对作为输入，实现病情进展推理；②利用 LLM 生成更长、可读性更高的答案来丰富指令集；③创建包含预后问题的全新指令数据集，填补了现有 VLM 在预测任务上的空白。

**🔧 技术方法**

技术手段为：NVILA‑8B 预训练模型的指令微调，结合多图输入、LLM（Llama‑3.2‑11B‑Vision‑Instruct、Llama‑3.1‑405B）生成答案与评价指标；采用专家模型（TorchXRayVision）提供疾病概率标签。

**📊 数据集**

使用公开数据集 MIMIC‑CXR 与其衍生的 Medical‑Diff‑VQA（包含单图诊断问题与差异问题），并自行扩充预后指令集。

**📈 对比分析**

通过 BLEU‑4、ROUGE‑L、Llama‑Score 等指标与基线 NVILA‑8B、LLaVA‑Med、D‑Rax 进行对比，结果显示 LUMEN 在诊断问题上与基线相当甚至略优，在差异与预测问题上显著提升（BLEU 0.375→0.656，Llama‑Score 4.61→4.86），但整体预测性能仍低于诊断。

**⚠️ 局限性**

局限性包括：仅使用两张时间点图像，未覆盖多时间序列；缺乏明确的纵向预后标签导致预测不确定；未结合治疗或临床文本等多模态信息，限制了模型的临床适用性。

---

## 35. PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning

**arXiv ID:** 2602.20537 | [PDF](https://arxiv.org/pdf/2602.20537v1)

**作者:** Xinyong Cai `[一作]` (Sichuan University), Yuankai Wu `[通讯]` (Sichuan University)

**通讯引用:** 4271 | [OpenAlex ID](https://openalex.org/A5100370856)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全卷积框架PFGNet，用像素级频率引导的外围频率门控实现自适应大卷积核的中心-外围抑制，以提高时空预测模型的表达力和效率。

**💡 创新点**

创新点在于：①将生物学中心–外围机制与可学习的环形带通滤波器结合；②利用像素级梯度、Laplacian和局部方差三种频率提示动态门控多尺度大卷积；③将大卷积分解为水平1×k和垂直k×1可分离卷积，实现线性计算复杂度。

**🔧 技术方法**

核心技术包括：像素级频率提取（Sobel、Laplacian、方差），softmax门控的多尺度融合，学习可调β的中心抑制，GLU通道混合，GRN归一化以及1×1卷积的通道扩展。

**📊 数据集**

在四大公开时空预测基准上验证：Moving MNIST、TaxiBJ、KTH 与 Human3.6M。

**📈 对比分析**

与多种递归、混合与纯卷积/Transformer方法对比，PFGNet在所有数据集上均实现或接近SOTA，在TaxiBJ和Human3.6M上以极低的参数（1.9M/7.3M）与FLOPs（0.6G/58.3G）取得显著优势，且在移动MNIST和KTH上获得更优的MSE/SSIM/PSNR。

**⚠️ 局限性**

局限性包括：对极端噪声或非均匀纹理区域的鲁棒性尚未充分验证；频率提示采用固定深度可卷积，可能无法捕获更高层次语义信息；在极长序列预测中仍需进一步探讨自回归或跨时间注意机制的整合。

---

## 36. Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning

**arXiv ID:** 2602.20528 | [PDF](https://arxiv.org/pdf/2602.20528v1)

**作者:** Justin Lovelace `[一作]` (Cornell University), Kilian Q. Weinberger `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的语言模型架构STAR‑LDM，结合自回归生成与潜在扩散规划，在生成前暂停思考，先在句子嵌入空间中规划语义，再继续生成文本

**💡 创新点**

通过在自回归解码器中引入“思考”阶段，使模型能够在连续空间中进行全局语义规划，显著提升文本连贯性、推理能力与可控性

**🔧 技术方法**

使用Transformer为自回归解码器、句子嵌入（Sentence‑T5）、两阶段扩散Transformer (DiT)、噪声条件与分类器引导的扩散规划

**📊 数据集**

FineWeb大规模文本、StoryCloze、CommonsenseQA、SIQA、ARC等多项零样本理解与生成基准

**📈 对比分析**

与同规模GPT‑2、Pythia等自回归模型对比，STAR‑LDM在零样本理解（尤其是常识推理）与故事续写（Coherence/Reasoning）上平均提升约10–15%，在控制任务中也实现了更好的流畅度与多样性平衡

**⚠️ 局限性**

对噪声规划步骤的调参复杂度高，推理时需要额外扩散步骤导致速度慢，且在表面语法与细粒度表达的精确性上仍不及纯自回归模型

---

## 37. Expregular functions

**arXiv ID:** 2602.21019 | [PDF](https://arxiv.org/pdf/2602.21019v1)

**作者:** Thomas Colcombet `[一作]` (CNRS), Pierre Ohlmann `[通讯]` (CNRS)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出并研究了一类具有指数增长的字符串映射函数——expregular函数，并展示了三种等价描述方式（MSO集合解释、yield‑Hennie机和Ariadne转发器），证明了它们之间的互译性以及该类函数的正则性反射性质；

**💡 创新点**

核心创新在于：①将指数增长的函数引入“finite‑state”框架并给出三种等价模型；②构造了MSO集合解释到yield‑Hennie机的翻译，证实了该类函数的正则性反射，从而解答了自动结构中ω‑word MSO理论可判定的长期猜想；③利用可定义的基与漏斗结构完成了复杂度分析，展示了闭包性质与组合能力；

**🔧 技术方法**

技术手段包括：MSO逻辑扩展（F‑变量）、定义简单度与分割、可定义基与漏斗的构造、tiling与标记的局部可更新编码、合成类型（type）的组合、以及对yield‑Hennie机状态与磁带的精细设计；

**📊 数据集**

该工作主要是理论性质证明，不涉及实验数据集；

**📈 对比分析**

在理论层面，expregular函数类表现出良好的闭包性质：对任意正则函数的前置合成保持不变，对多项式增长的polyregular函数的后置合成保持不变；通过等价翻译，可在不同模型间转换，说明其表达能力与复杂度一致；

**⚠️ 局限性**

局限性：expregular函数类不对polyregular函数的前置合成闭包；对更一般的树到树或更高阶转发器的扩展仍未完全解决；此外，证明依赖于大量可定义性和类型组合，实际实现复杂度尚未给出。

---

## 38. TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering

**arXiv ID:** 2602.20903 | [PDF](https://arxiv.org/pdf/2602.20903v1)

**作者:** Hanshen Zhu `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 38567 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 TextPecker 框架，利用结构感知的强化学习奖励，提升文本生成模型的视觉文本渲染（VTR）质量。

**💡 创新点**

创新点在于：①将细粒度字符结构异常检测与语义对齐融合成复合奖励；②构建大规模字符级结构异常标注数据集并通过 Stroke‑editing 合成引擎扩展错误多样性；③在 RL 优化中引入结构感知评估器，突破 OCR/MLLM 对结构缺陷的盲点。

**🔧 技术方法**

使用的技术包括 Flow‑GRPO 强化学习、结构感知评估器、结构质量与语义对齐分数、Stroke‑editing 合成引擎、Qwen3‑VL/InternVL3 多模大模型等。

**📊 数据集**

数据集主要包含：1) 由多模型生成的文本图像；2) 通过 OCR + 人工标注得到的字符级结构异常标注；3) 通过 Stroke‑editing 生成的合成结构错误样本；4) 原始文本语料 TextAtlas5M、Lex‑10k、WanJuan1.0 等。

**📈 对比分析**

在 OneIG‑Bench、CVTG‑2K、LongText‑Bench 等基准上与 OCR/MLLM 奖励、传统 NED 指标进行对比。实验表明，在 Flux、SD3.5、Qwen‑Image 等模型上，TextPecker 可平均提升结构质量 30%+、语义对齐 10%+，在中文渲染上结构质量 +4%、语义对齐 +8.7%。

**⚠️ 局限性**

局限性包括：对极少见或极端结构错误的泛化仍有限；中文字符结构复杂导致标注成本高；RL 收敛速度受超参数影响，需进一步优化。

---

## 39. E-MMKGR: A Unified Multimodal Knowledge Graph Framework for E-commerce Applications

**arXiv ID:** 2602.20877 | [PDF](https://arxiv.org/pdf/2602.20877v1)

**作者:** Jiwoo Kang `[一作]` (Ulsan National Institute of Science and Technology), Yeon-Chang Lee `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 44282 | [OpenAlex ID](https://openalex.org/A5100383157)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了专为电商设计的多模态知识图谱，并在其上通过图神经网络与 KG 目标联合学习，得到可用于推荐、搜索等多任务的统一物品表示。

**💡 创新点**

创新点在于（1）提出了支持多模态可扩展和跨任务通用的电商专用多模态 KG；（2）将 GNN 与 RotatE 语义约束结合，实现对关系层语义的建模；（3）通过共享语义空间实现推荐与搜索的无缝迁移。

**🔧 技术方法**

采用 LightGCN 作为图编码器，RotatE 作为 KG 损失，预训练的多模态编码器（CNN、SBERT、OpenAI text‑embedding‑3‑large）进行特征提取，结合 BPR 训练推荐，余弦相似度实现检索。

**📊 数据集**

使用六个亚马逊子集（Office、Grocery、Pet、Toys、Beauty、Clothing）数据集，这些数据包含用户-商品交互及图像、描述、评论、图像标题等多模态信息。

**📈 对比分析**

在推荐任务上与 LightGCN 及六个 MMRS（GRCN、DualGNN、LATTICE、BM3、FREEDOM、LGMRec）对比，Recall@10 最大提升 10.18%；在搜索任务上与单模态向量检索对比，细粒度查询 Recall@10 提升 7.80%，粗粒度查询提升 21.72%。

**⚠️ 局限性**

局限性包括：仅在亚马逊数据上验证，未测试对动态商品与用户更新的适应性；KG 结构固定，未充分利用更丰富的属性或外部知识；对新模态的编码器需要手工设计与集成。

---

## 40. Pressure Reveals Character: Behavioural Alignment Evaluation at Depth

**arXiv ID:** 2602.20813 | [PDF](https://arxiv.org/pdf/2602.20813v1)

**作者:** Nora Petrova `[一作]` (Prolific), John Burden `[通讯]` (Prolific)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个包含904个多轮对话情景的对齐基准，评估了24款前沿模型在37种对齐行为（诚实、安全、非操控、鲁棒、可纠正性、策略行为）上的表现，并发现模型间存在统一的对齐因子，鲁棒性普遍不足，封闭源模型优于开源模型。

**💡 创新点**

统一评估37种对齐行为并发现单因子结构；多轮、压力情景设计；公开Leaderboard供持续跟踪；将情景生成与LLM评判相结合。

**🔧 技术方法**

自动化情景生成（Bloom框架）、LLM评判（Claude Opus 4.5）与人工校准、因子分析（PCA）与统计显著性检验。

**📊 数据集**

共904个情景，涵盖六类对齐行为，来源于Bloom自动生成、Petri探测、手工设计，并经过人工真实感评估。

**📈 对比分析**

采用1–5分评分体系计算平均得分和通过率；24模型互相比值显著；最高得分4.66（Claude 4.5 Sonnet），最低2.92（Mistral Large）；闭源模型平均得分4.05，高于开源3.41。

**⚠️ 局限性**

样本量有限、情景可能缺失、LLM评判可能存在系统性偏差、部分行为处于天花板、基准仅为英文且以西方规范为主，可能不适用于其他文化或语言环境。

---

## 41. InterviewSim: A Scalable Framework for Interview-Grounded Personality Simulation

**arXiv ID:** 2602.20294 | [PDF](https://arxiv.org/pdf/2602.20294v1)

**作者:** Yu Li `[一作]` (Salesforce Research), Chien-Sheng Wu `[通讯]` (Salesforce Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并评估基于大规模访谈记录的个性化语言模型；

**💡 创新点**

提出跨维度评估框架（内容相似度、事实一致性、个性一致性、知识保留）并公开大规模访谈语料库；

**🔧 技术方法**

使用 GPT‑4.1 进行生成与评估，结合检索增广（基于嵌入相似度）与时间序列上下文；

**📊 数据集**

包含 1,000 位公共人物、23,536 条经人工验证的访谈稿，总计 671,424 条问答对、11,464 小时访谈；

**📈 对比分析**

与简单提示、维基信息、时间序列示例等方法对比，检索增广在内容与个性维度表现最佳（内容相似度 3.50，个性相似度 78.4%），时间序列在事实一致性与知识保留上更优；

**⚠️ 局限性**

数据受限于英语西方公开访谈、时间窗口近 2015‑2024，评估依赖 LLM 判别器可能存在偏差，未考虑个性随时间演变及更复杂的检索/一致性技术。

---

## 42. A Benchmark for Deep Information Synthesis

**arXiv ID:** 2602.21143 | [PDF](https://arxiv.org/pdf/2602.21143v1)

**作者:** Debjit Paul `[一作]` (Huawei Noah's Ark Lab), Gerasimos Lampouras `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个新的基准数据集，旨在评估大语言模型（LLM）及其代理在多源信息合成、网页浏览与结构化推理方面的能力，包含120个跨越67个国家、7个领域的现实任务；

**💡 创新点**

创新点在于：①设计了真实、可验证且多源信息检索与合成相结合的任务；②采用了专家驱动的多阶段数据收集流程，确保任务难度与真实性；③构建了多步骤推理链与JSON格式答案，便于自动评测；

**🔧 技术方法**

使用了大语言模型（如GPT‑4.1、GPT‑5、Gemini‑Pro‑2.5、DeepSeek‑R1）以及三种深度研究代理框架（o3‑deep‑research、smolagents、OWL），并集成了网页搜索、浏览、文档处理、代码执行等工具；

**📊 数据集**

数据集为自制，源自223个官方数据源，经过专家筛选、假设验证和任务制定，最终生成120个信息合成任务；

**📈 对比分析**

通过Exact Match、F1、Precision、Recall以及LLM‑Judge（LLM 判定）四个指标评估。结果显示，最优单一LLM（Gemini‑Pro‑2.5）F1仅6.25，最优代理（o3‑deep‑research）F1仅8.97；所有模型在Exact Match上均为0，表明任务极具挑战性；

**⚠️ 局限性**

限制包括：①对导航与合成错误高；②在非欧亚地区任务表现极差，显示地理偏见；③对工具使用的依赖强，缺乏可靠性与一致性；④高维度、多步推理导致模型输出不稳定，需进一步提升规划与执行能力。

---

## 43. Interaction-aware Representation Modeling with Co-occurrence Consistency for Egocentric Hand-Object Parsing

**arXiv ID:** 2602.20597 | [PDF](https://arxiv.org/pdf/2602.20597v1)

**作者:** Yuejiao Su `[一作]` (Hong Kong Polytechnic University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5893 | [OpenAlex ID](https://openalex.org/A5044722301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 InterFormer，一种面向 egocentric 视角的交互感知 Transformer，用于精准分割手部及其交互对象。

**💡 创新点**

创新点包括：基于交互先验的动态查询生成器（DQG）、融合语义与交互上下文的双向特征选择器（DFS），以及强制物体-手部逻辑一致性的条件共现损失（CoCo Loss）。

**🔧 技术方法**

技术核心是 Swin Transformer 编码器 + deformable DETR 解码器，配合交互先验预测分支、DQG、DFS 以及 CoCo 损失实现端到端学习。

**📊 数据集**

使用公开数据集 EgoHOS（含 in‑domain、out‑of‑domain）和 mini‑HOI4D 进行训练与评测。

**📈 对比分析**

与多种 SOTA 方法（Segformer、SCTNet、UperNet、Mask2Former、CaRe‑Ego 等）对比，在 in‑domain mIoU 73.22%，out‑of‑domain 72.82%，mini‑HOI4D 66.07%，均显著领先，对手部与交互物体的分割效果尤为突出。

**⚠️ 局限性**

局限性：仍依赖交互边界预测，CoCo 损失对阈值 τ 敏感；在极端遮挡或多手多物体场景下性能仍有提升空间；模型参数量相对较大，对算力要求较高。

---

## 44. CITED: A Decision Boundary-Aware Signature for GNNs Towards Model Extraction Defense

**arXiv ID:** 2602.20418 | [PDF](https://arxiv.org/pdf/2602.20418v1)

**作者:** Bolin Shen `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 934 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了CITED框架，用决策边界感知的签名节点实现GNN模型的所有权验证，兼顾嵌入层和标签层。

**💡 创新点**

创新点在于：①不依赖水印/指纹，直接利用模型内部决策边界信息生成签名；②统一两种输出层的验证；③通过Wasserstein距离和ARUC指标实现高效、无额外辅助模型的验证；④提供理论保证和全面实验验证。

**🔧 技术方法**

使用的技术包括：边界投票得分、margin、thickness、heterogeneity等签名评分；Wasserstein距离、ARUC和AUC等验证指标；对数几何/软最大化处理；对GNN的理论分析（参数扰动下的输出稳定性）和基于图卷积的实现。

**📊 数据集**

实验数据集涵盖七个常用图数据集（Cora、CiteSeer、PubMed、Photo、Computers、CS、Physics），并在GCN、GAT、GraphSAGE、GCNII、FAGCN等五种主流GNN架构上进行测试。

**📈 对比分析**

与RandomWM、BackdoorWM、SurviveWM（标签层水印方法）和GrOVe（嵌入层指纹方法）比较，CITED在嵌入层ARUC约提高5–10%，在标签层ARUC提升30–50%，同时对模型下游任务性能影响小、推理速度快；AUC也明显优于基线。

**⚠️ 局限性**

局限性包括：签名节点生成依赖阈值和模型的决策边界分布，可能在高度不平衡或极大规模图上效果不稳；对攻击者仅限于基于查询的模型提取场景，且对某些特殊攻击策略（如对抗性提取）尚未验证。

---

## 45. PyVision-RL: Forging Open Agentic Vision Models via RL

**arXiv ID:** 2602.20739 | [PDF](https://arxiv.org/pdf/2602.20739v1)

**作者:** Shitian Zhao `[一作]` (Shanghai AI Lab), Chen Wei `[通讯]` (Rice University)

**通讯引用:** 126630 | [OpenAlex ID](https://openalex.org/A5100364769)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了基于Python动态工具的多模态强化学习框架 -RL，分别训练出针对图像的 -Image 和视频的 -Video 视觉模型。

**💡 创新点**

创新点在于：1）引入 Oversampling–Filtering–Ranking 生成 Rollout 机制与累计工具奖励，防止训练中出现交互崩塌；2）在视频模型中实现按需上下文构建，动态抽取关键帧，大幅提升视觉 token 效率。

**🔧 技术方法**

采用 Python 作为原子工具实现动态工具调用；使用强化学习（改进的 GRPO）和累积工具奖励；在 Rollout 生成中加入标准差排序以提升训练稳定性；对图像/视频输入做适配并通过 Python 运行时处理。

**📊 数据集**

数据集包括：图像任务的 V*、HRBench‑4K/8K、DynaMath、MathVerse、WeMath、TIR‑Bench；视频任务使用 VSI‑Bench；SFT 与 RL 训练数据分别由 GPT‑4.1 合成和公开任务集合（DeepEyes、Mini‑o3、V‑Thinker 等）收集。

**📈 对比分析**

与现有静态工具集模型（Pixel‑Reasoner、DeepEyes 等）及动态工具模型（Thyme、CodeV、DeepEyes‑v2）对比，-Image 在视觉搜索、跨模态推理和代理推理上均取得 state‑of‑the‑art 提升（如 V* +10.2%、DynaMath +4.4% 等）。-Video 在 VSI‑Bench 上比 Qwen2.5‑VL‑7B 提升 7.3%，且平均仅消耗 5K visual token（相比 45K），表现出更优的准确率–效率权衡。

**⚠️ 局限性**

局限性包括：1）依赖 Python 运行时，可能出现安全/权限问题；2）训练成本高，需大规模 GPU 资源；3）在极端复杂任务或长视频中，对工具调用策略的泛化尚不充分。

---

## 46. An Expert Schema for Evaluating Large Language Model Errors in Scholarly Question-Answering Systems

**arXiv ID:** 2602.21059 | [PDF](https://arxiv.org/pdf/2602.21059v1)

**作者:** Anna Martin-Boyle `[一作]` (University of Minnesota), Harmanpreet Kaur `[通讯]` (University of Minnesota)

**通讯引用:** 398 | [OpenAlex ID](https://openalex.org/A5101399911)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了一套基于领域专家的错误评估模式，用于评估学术问答系统中LLM输出的准确性、完整性、合成等方面。

**💡 创新点**

创新点在于：①从领域专家的实际评估实践出发，系统性地梳理出20种细粒度错误模式并归纳为七大类；②通过无提示与提示两轮评估展示专家在结构化评估框架下能发现更多被忽视的错误；③提出混合自动化与专家监督的评估思路，并为个性化评估工具奠定框架。

**🔧 技术方法**

技术手段包括检索增量生成（RAG）pipeline、开源小型LLM Mixtral-8x-7B-Instruct、句子向量检索（All-MiniLM-L6-v2）、KeyBERT查询扩展、主题分析与轴向编码（Open/Axial Coding）。

**📊 数据集**

数据集由两部分组成：68条专家生成的问答对（来自作者自己论文）和120条（共188问答）由10位独立领域专家针对自己论文生成的问答，用于验证与进一步分析。

**📈 对比分析**

评估方法主要是专家无提示与提示两轮的定性分析，比较两轮错误发现的差异并绘制错误类型热图，展示不同问题类型下错误分布；未给出传统量化指标，但从错误分布可见系统在高阶推理、合成、引用完整性等方面表现不佳。

**⚠️ 局限性**

局限性包括：样本仅10名STEM领域专家，可能不适用于人文社科；使用的小型开源模型限制了性能；评估清单较长导致专家疲劳；自动化检测能力仍有限，难以完全替代人工评估。

---

## 47. 52-Hz Whale Song: An Embodied VR Experience for Exploring Misunderstanding and Empathy

**arXiv ID:** 2602.20348 | [PDF](https://arxiv.org/pdf/2602.20348v1)

**作者:** Yibo Meng `[一作]` (Tsinghua University), Yan Guan `[通讯]` (Tsinghua University)

**通讯引用:** 10127 | [OpenAlex ID](https://openalex.org/A5100704992)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个三幕式的VR沉浸式体验，让玩家从被误解的鲸鱼转变为调解者，体验沟通失灵与调解过程。

**💡 创新点**

采用身体化互动和角色切换，将误解视为身体经验并通过海洋隐喻进行沉浸式实验，推动共情从被动转为主动调解。

**🔧 技术方法**

使用Unity引擎实现全身运动捕捉、语音到波形可视化、空间音频、触觉反馈、程序化修辞等多模态交互技术。

**📊 数据集**

主要使用实验中收集的30名参与者的问卷数据（IRI、SDS），未使用公开数据集。

**📈 对比分析**

通过与观看纪录片的对照组进行前后测量比较，实验组在社会距离尺度和共情量表上均显著提升（p<0.001，效应量 d≈1.5-1.9）。

**⚠️ 局限性**

样本量有限、仅为短期实验、未能单独隔离角色切换的独立贡献，且效果的长期可持续性未知。

---

## 48. Semantic Novelty at Scale: Narrative Shape Taxonomy and Readership Prediction in 28,606 Books

**arXiv ID:** 2602.20647 | [PDF](https://arxiv.org/pdf/2602.20647v1)

**作者:** W. Frederick Zimmerman `[一作]` `[通讯]` (Nimble Books LLC), W. Frederick Zimmerman (Nimble Books LLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过计算书籍段落的语义新颖度曲线，对 PG19 预1920 年英文书籍进行聚类，提出八种叙事形状原型，并分析其与读者下载量及流派的关系。

**💡 创新点**

创新点在于提出语义新颖度作为信息密度度量，揭示语义体量是读者关注的独立预测因子，并建立了细粒度的八类叙事形状分类。

**🔧 技术方法**

采用 SBERT 768 维句子嵌入、余弦距离计算新颖度，PAA+SAX 对曲线进行降维，Ward 链接聚类得到形状类别，统计和回归分析评估读者相关性。

**📊 数据集**

使用 PG19 公开语料库（28,606 本预1920 年英文书籍），并以 Gutenberg 下载量作为读者受众指标。

**📈 对比分析**

与传统情感弧和主题轨迹方法对比，使用部分相关和多元线性回归证明语义体量在长度控制下的相关系数达到 0.32，显著高于其他指标；聚类结果与 DTW 方式对比显示欧氏距离聚类更能预测下载量。

**⚠️ 局限性**

局限包括仅针对英文学前期文本、下载量只能反映获取而非阅读质量、SBERT 嵌入对不同语言或现代文本可能不通用、段落划分和流派标注的粗糙度。

---

## 49. Generative AI and Machine Learning Collaboration for Container Dwell Time Prediction via Data Standardization

**arXiv ID:** 2602.20540 | [PDF](https://arxiv.org/pdf/2602.20540v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 50. The Truthfulness Spectrum Hypothesis

**arXiv ID:** 2602.20273 | [PDF](https://arxiv.org/pdf/2602.20273v1)

**作者:** Zhuofan Josh Ying `[一作]` (Columbia University), Peter Hase `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM内部如何编码真相，提出真相光谱假设并通过多领域检测、几何分析与因果干预验证其存在。

**💡 创新点**

创新点在于：①提出真相光谱假设，阐明真相方向从域通到域特的连续谱；②发现Mahalanobis余弦相似度能近乎完美预测跨域探针性能（R²≈0.98）；③设计Stratified INLP与LEACE两种概念消除方法，构造并证明域通与域特真相方向共存；④通过因果干预验证域特方向对模型输出更具影响力。

**🔧 技术方法**

采用线性探针（LR、DoM、LDA）、Mahalanobis余弦相似度评估、概念消除方法（INLP、LEACE）、层级投影、因果干预（在中间层添加偏置）以及跨模型层级与数据集的对比实验。

**📊 数据集**

使用的主要数据集包括：FLEED（定义、经验、逻辑、虚构、伦理五类真相）、sycophantic lying、expectation‑inverted lying、先前的诚实基准（如 insider trading、sandbagging 等），并在多种 Llama、Qwen 等LLM 上进行实验。

**📈 对比分析**

通过对比不同域的线性探针AUROC，发现大部分域互相泛化良好，但对 sycophantic 与 expectation‑inverted 的性能接近随机；Mahalanobis余弦相似度几乎完美预测跨域AUROC（R²≈0.98）；post‑training 后 sycophancy 与其它真相方向分离，导致探针泛化下降，证明了代表性几何重组。

**⚠️ 局限性**

局限性包括：未覆盖所有真相类型；FLEED 数据为模型生成，可能带有偏差；仅分析线性结构，非线性真相表示未探究；post‑training 分析仅聚焦 sycophancy，其他潜在变形未考察；因果干预效果有限，主要影响置信度而非答案翻转。

---

## 51. Frontier Space-Time Algorithms Using Only Full Memory

**arXiv ID:** 2602.21089 | [PDF](https://arxiv.org/pdf/2602.21089v1)

**作者:** Petr Chmel `[一作]` (Charles University), Ninad Rajgopal `[通讯]` (Charles University)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5075537255)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文开发了用于算法设计基本问题的催化算法，这些算法在多项式时间内运行，仅使用 (log(n)) 的工作空间，并且使用次线性催化空间，匹配在多项式时间内运行的非催化算法的最佳已知空间界限。

**💡 创新点**

创新点在于设计了多项式时间算法，能够在催化空间中实现与非催化设置相匹配的时间-空间界限，并且在多个经典问题上（如编辑距离、最长公共子序列和离散弗雷歇距离）取得了显著的空间效率提升。

**🔧 技术方法**

使用了催化计算模型，结合了随机算法和图的特殊结构，采用了新的图工具和催化算法技术。

**📊 数据集**

使用了多种数据集，包括图的连通性问题、编辑距离、最长公共子序列和离散弗雷歇距离等。

**📈 对比分析**

与现有方法相比，本文的方法在催化空间使用上取得了显著的改进，特别是在处理编辑距离和最长公共子序列时，达到了次线性空间的使用效率。性能上，算法在多项式时间内运行，且在空间使用上优于传统方法。

**⚠️ 局限性**

限制在于算法的运行时间和空间使用仍然依赖于输入规模，且在某些情况下，具体的运行时间尚未明确计算，依赖于复杂的对数空间程序的运行时间。

---

## 52. Markets are competitive if and only if P != NP

**arXiv ID:** 2602.20415 | [PDF](https://arxiv.org/pdf/2602.20415v1)

**作者:** Philip Z. Maymin `[一作]` (Fairfield University), Philip Z. Maymin `[通讯]` (Fairfield University)

**通讯引用:** 292 | [OpenAlex ID](https://openalex.org/A5026617597)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

论文证明：竞争市场的存在与否取决于 P 与 NP 的关系；若 P=NP，则企业可在有限时间内解决合谋检测、规划与惩罚问题，从而让合谋成为可持续均衡；若 P≠NP，则在符合实例难度假设的市场中，合谋检测不可行，竞争最终占优。

**💡 创新点**

创新点包括：
- 将计算复杂度与市场竞争直接关联，提出“效率–竞争不可能性”命题；
- 发现透明度提升实际上降低合谋检测难度，形成“透明度悖论”；
- 描述 AI 逐步提升计算能力导致的市场三态过渡（竞争→不稳定→合谋）和异构计算能力产生的两层定价结构；
- 提出“计算反垄断”框架，将市场设计的计算复杂度视为竞争保障。

**🔧 技术方法**

主要技术手段：
- 复杂度理论证明（NP-hard性归约）
- 重复博弈与公共信息监测下的 FOLK 定理应用
- 计算机可实现的策略规划与惩罚计算框架
- 形式化假设（实例难度假设）与泛性论证。

**📊 数据集**

论文并未使用实验数据集，而是通过理论模型与已发表的实证研究（如大型语言模型自发合谋、Q-learning 价格算法合谋等）来佐证其预测。

**📈 对比分析**

比较方法主要是理论推导与证明：
- 通过归约展示三类合谋相关问题（策略、检测、惩罚）均为 NP‑hard；
- 在 P=NP 情况下构造可实现的完美公共均衡；
- 在 P≠NP 且满足实例难度假设时证明合谋不可持续。
- 性能表现以理论可解性与不可解性为衡量，没有数值实验。

**⚠️ 局限性**

局限性：
- 关键结果依赖“实例难度假设”，该假设虽然在一般意义下成立，但在具有特殊需求结构（可分离、低秩、稀疏）的市场中可能失效；
- 假设企业为完全理性、利润最大化的代理，未考虑情绪、有限理性或其他动机；
- 仅讨论完美公共均衡与重复博弈框架，未考察有限周期或不完全信息下的策略；
- 对真实市场的实证验证仍需进一步工作。

---

## 53. Momentum Guidance: Plug-and-Play Guidance for Flow Models

**arXiv ID:** 2602.20360 | [PDF](https://arxiv.org/pdf/2602.20360v1)

**作者:** Runlong Liao `[一作]` (University of Texas at Austin), Qiang Liu `[通讯]` (University of Texas at Austin)

**通讯引用:** 35229 | [OpenAlex ID](https://openalex.org/A5100409479)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在流式生成模型的推理阶段，提出Momentum Guidance（MG）机制，通过对过去速度的指数移动平均进行加速，从而在不增加额外模型评估的前提下增强生成质量。

**💡 创新点**

创新点在于利用ODE轨迹自身的动量信息作为平滑参考，无需额外模型，兼容现有的CFG，且保持单步评估成本。

**🔧 技术方法**

使用技术包括流模型的ODE积分、指数移动平均、速度外推以及与classifier-free guidance的组合。

**📊 数据集**

实验数据集包括ImageNet-256、Stable Diffusion 3（SD3）和FLUX.1-dev，并使用HPSv2.1评测。

**📈 对比分析**

通过与无指导、CFG基础以及不同采样步长的对比，MG在ImageNet-256上无CFG时FID降低36.68%，有CFG时25.52%，在64步时FID达1.597；在SD3/FLUX上均表现出一致的FID与精度提升。

**⚠️ 局限性**

局限性在于相对强CFG时提升有限，实验中超参数调优不充分，以及潜在的指导重叠问题。

---

## 54. OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport

**arXiv ID:** 2602.20205 | [PDF](https://arxiv.org/pdf/2602.20205v1)

**作者:** Xiwen Chen `[一作]` (Morgan Stanley), Abolfazl Razi `[通讯]` (Clemson University)

**通讯引用:** 2322 | [OpenAlex ID](https://openalex.org/A5011987346)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练无关的视觉token剪枝框架OTPrune，基于分布对齐实现对视觉token的高效选择；

**💡 创新点**

创新点在于将剪枝问题建模为最优传输（Wasserstein距离）最小化，既保留局部多样性，又兼顾全局分布代表性；

**🔧 技术方法**

采用二维Wasserstein距离近似、Gaussian 近似、log-determinant 下界、子模函数贪婪算法（Cholesky分解）实现高效剪枝；

**📊 数据集**

使用多种视觉问答、图像描述与多模态推理数据集：COCO、Flickr30k、GQA、MME、MMB、OKVQA、POPE、SQA、SeedBench、Nocaps、ScienceQA-IMG 等；

**📈 对比分析**

与现有一键剪枝（FastV、VTW、DivPrune）和自适应/微调方法对比，OTPrune在 11 个基准上平均排名提升 30%+、FLOP 仅 10-15%，且显著降低OT距离，证明更稳健、语义保真；

**⚠️ 局限性**

局限在于对参数 γ 的选择仍需实验验证，且目前仅在 LLaVA 视觉编码器上评估，未来需扩展到更多多模态模型与更大规模数据集。

---

## 55. Validation of an analyzability model for quantum software: a family of experiments

**arXiv ID:** 2602.21074 | [PDF](https://arxiv.org/pdf/2602.21074v1)

**作者:** Ana Díaz-Muñoz `[一作]` (AQCLab Software Quality and University of Castilla-La Mancha), Mario Piattini `[通讯]` (University of Castilla-La Mancha)

**通讯引用:** 16056 | [OpenAlex ID](https://openalex.org/A5083263115)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

验证了基于ISO/IEC 25010的混合软件可分析性模型中针对量子软件的量子成分

**💡 创新点**

首次将传统软件可分析性指标与专门为量子电路设计的量子指标结合，并通过四项实验进行经验验证

**🔧 技术方法**

使用量子电路宽度、深度、门复杂度、量子环路复杂度等量子指标，以及Kruskal‑Wallis检验、Stouffer元分析等统计方法

**📊 数据集**

采用四组参与者（硕士、本科生、专业人士）在不同实验条件下完成的量子电路测评问卷，共计约330名受试者

**📈 对比分析**

通过非参数检验比较可分析性等级对分数的影响，结果显示高可分析性电路显著提高评估得分，元分析得到强统计显著性（p < 1e‑18）

**⚠️ 局限性**

受试者规模偏小且分布不均，部分实验缺乏足够统计功效；模型目前仅验证了量子部分，未涉及经典与量子混合系统整体评估

---

## 56. CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference

**arXiv ID:** 2602.20732 | [PDF](https://arxiv.org/pdf/2602.20732v1)

**作者:** Chao Fei `[一作]` (King Abdullah University of Science and Technology), Panos Kalnis `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 9977 | [OpenAlex ID](https://openalex.org/A5014399734)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种面向长上下文LLM推理的KV缓存管理系统CHESS，利用层次化语义选择动态重建有意义的上下文。

**💡 创新点**

引入上下文感知、层次化的Page/Chunk/Grid语义选择与不确定性回溯机制，并实现零拷贝、页级高效并行的系统架构。

**🔧 技术方法**

采用Key-Key语义亲和度的相似度计算、GEMM批量化、CUDA Graph、PagedAttention、FlashInfer以及量化阈值/不确定性监测。

**📊 数据集**

在LongBenchV2上评估生成质量，在合成长序列（4k-32k）上评估吞吐率/延迟。

**📈 对比分析**

与Full-KV、SnapKV、H2O、KeyDiff、Quest等基线比较，CHESS在仅使用1% KV时质量与Full-KV持平或优于；吞吐率最高可达4.56×，延迟保持平稳，优于其他稀疏方法。

**⚠️ 局限性**

仍需在极大批量或更长上下文下评估；不确定性阈值依赖离线校准；对非分页或非页面对齐的模型适配有限。

---

## 57. KairosVL: Orchestrating Time Series and Semantics for Unified Reasoning

**arXiv ID:** 2602.20494 | [PDF](https://arxiv.org/pdf/2602.20494v1)

**作者:** Haotian Si `[一作]` (Chinese Academy of Sciences), Gaogang Xie `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 6254 | [OpenAlex ID](https://openalex.org/A5030689390)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了语义条件时间序列推理任务，构建了KairosDataPipe与KairosDataset，并设计了两轮强化学习框架KairosVL以提升模型在此任务上的推理性能。

**💡 创新点**

创新点在于：①系统化定义并拆解为四个子任务；②利用LLM驱动的数据生成管道产生高质量推理样本；③两轮强化学习先强化基本时序概念后提升复杂推理，解决单轮RL过浅的问题。

**🔧 技术方法**

使用技术包括大规模多模态语言模型（Qwen2.5VL）、RLVR强化学习、GRPO优化、KL正则化、LLM多代理评判器，以及文本+图像双模数据生成与可验证奖励机制。

**📊 数据集**

使用的数据集有：①KairosDataset（约2k样本）由KairosDataPipe生成；②KairosBench（约600样本）用于评测；③真实世界样本集A（约100条）来自生产环境；④Primitive Dataset（规则生成的基本时序任务）。

**📈 对比分析**

通过在多模态LLM基线（InternVL3‑8B、Qwen2.5VL‑7B/32B/72B、GLM4V‑Plus）和商用模型（GPT‑4o、Gemini‑2.5Flash）上进行多选题准确率对比，KairosVL在两数据集上均显著优于基线，单轮RL相比两轮RL提升约30%，且参数仅7B时与大型商用模型竞争。

**⚠️ 局限性**

局限性包括：对反事实任务的表现仍落后于大模型；模型规模限制导致假设链生成不足；单轮RL可能导致推理过浅；未来需要更大规模模型和更完善的多模态预训练来进一步提升性能。

---

## 58. An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models

**arXiv ID:** 2602.20324 | [PDF](https://arxiv.org/pdf/2602.20324v1)

**作者:** Cathy Shyr `[一作]` (Vanderbilt University Medical Center), Hua Xu `[通讯]` (Yale School of Medicine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并外部验证了一个端到端的人工智能框架RARE-PHENIX，用于从临床笔记中提取表型、标准化为Human Phenotype Ontology（HPO）术语并对诊断相关表型进行排序。

**💡 创新点**

创新点在于将表型提取、标准化与优先级排序整合为完整工作流，并通过检索增强生成（RAG）和监督学习到排序模型显著提升诊断相关表型的准确性，证明模块化设计对性能提升具有关键作用。

**🔧 技术方法**

使用的技术包括：大语言模型（LLaMA系列与Azure OpenAI GPT-4）进行表型提取；检索增强生成（RAG）对齐HPO术语；监督学习到排序（XGBoost、LightGBM、CatBoost）对表型进行诊断信息优先级排序；以及PEFT、QLoRA、LoRA等高效微调技术。

**📊 数据集**

使用的数据集包括：11个UDN临床站点的2671例病人用于训练（包含RareDis、合成临床文本），VUMC 143例病人用于外部验证（共16357条临床笔记），以及HPO、OMIM、Orphanet等知识库进行特征工程。

**📈 对比分析**

与现有深度学习基线PhenoBERT进行对照，评价指标为Lin相似度、精确率、召回率和F1。RARE-PHENIX在所有k=10–50阈值下均优于PhenoBERT，最高Lin相似度达0.70（PhenoBERT为0.58），MAP@30为0.85；模块消融分析进一步证明标准化和排序模块对性能提升贡献显著。

**⚠️ 局限性**

局限性包括：依赖大型模型（如70B参数）在资源受限环境中难以部署；训练和验证数据主要来自UDN的复杂多系统病例，可能缺乏对一般遗传/亚专科人群的代表性；评估为回顾性研究，未测量对诊断效率或临床工作负荷的实际影响；使用人工标注的HPO列表作为金标准，可能不完整或存在主观差异。

---

## 59. AdapTools: Adaptive Tool-based Indirect Prompt Injection Attacks on Agentic LLMs

**arXiv ID:** 2602.20720 | [PDF](https://arxiv.org/pdf/2602.20720v1)

**作者:** Che Wang `[一作]` (Peking University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**通讯引用:** 6830 | [OpenAlex ID](https://openalex.org/A5027969322)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种适用于现代推理型大语言模型代理的自适应间接提示注入（IPI）攻击框架，并构建了新的 IPI‑3k 基准数据集。

**💡 创新点**

创新点：①将攻击策略和工具选择进行动态自适应优化，使攻击既具备鲁棒性又保持高度隐蔽性；②引入多轮迭代策略库生成机制，显著提升攻击成功率；③通过任务相关工具的选择，突破传统基于模板的攻击在“无关信息”场景下被过滤的局限；④创建 IPI‑3k 数据集，为评估推理型代理提供更真实、更大规模的测试环境。

**🔧 技术方法**

技术手段：利用 ReAct 框架进行攻击执行；采用链式思考（CoT）追踪代理决策过程；通过多轮迭代（默认 5 次）构建可迁移的攻击策略库；使用商业与开源 LLM（如 GPT‑4.1、Qwen‑3‑8B）自动评估工具风险并进行工具筛选；引入对抗生成和自适应提示优化算法；对抗性评估结合 MELON、Pi‑Detector 等现有防御。

**📊 数据集**

数据集：①IPI‑3k（3691 条正常代理轨迹，277 条高权限攻击工具）；②InjectAgent；③AgentDojo，作为对比基准。

**📈 对比分析**

实验对比：与 Ignore‑Instruction、Combined Attack、InjectAgent、AutoHijacker 等基线以及 MELON、Pi‑Detector 等防御进行对照。结果显示，攻击成功率（ASR）平均提升 2.13×（例如 GPT‑4.1 上从 8% 提升至 18.5%），在商业 LLM 上平均 ASR 约 50%；在开源 LLM 上平均 ASR 超过 30%。攻击同时导致系统实用性（UA）下降，且在现有防御下仍保持较高 ASR（下降幅度约 2×）。

**⚠️ 局限性**

局限性：①攻击策略的生成依赖大量商业 LLM 调用，成本高；②工具选择为灰盒假设，实际部署时需获取工具权限信息；③在极其先进的安全策略或专门针对 IPI 的防御机制下，攻击效果可能进一步被抑制；④IPI‑3k 虽更大但仍未覆盖所有可能的第三方工具生态，可能存在样本偏差。

---

## 60. Diagnosing Causal Reasoning in Vision-Language Models via Structured Relevance Graphs

**arXiv ID:** 2602.20878 | [PDF](https://arxiv.org/pdf/2602.20878v1)

**作者:** Dhita Putri Pratama `[一作]` (University of Melbourne), Yihao Ding `[通讯]` (University of Western Australia)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5074613052)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Vision‑Language Causal Graphs（VLCGs）以及对应的 ViLCaR 诊断基准，用于细粒度评估大型视觉语言模型在视觉因果推理中的相关性识别、因果推断和答案预测三阶段能力。

**💡 创新点**

创新点在于：①将问答任务的因果相关性显式编码为可查询的有向图，②设计了三任务（CA、CI、QA）和图对齐评估指标，实现对相关性识别与推理一致性的解耦评估；③通过结构化因果图提示提升模型的因果归因与推理一致性，而不单纯依赖答案准确率。

**🔧 技术方法**

技术手段包括：利用大语言模型（LLM）进行图生成与校验、CLIP‑Score 等视觉文本相似度检验、基于BERT/W2V 的语义相似度计算、LLM评估器对推理链与金标准图的一致性评估，以及结构化提示（VLCG‑Augmented Prompting）。

**📊 数据集**

使用的数据集为 VQA、VCR、Visual7W、V‑Genome、OK‑VQA、CoSIm、CELLO 等，构造 ViLCaR 共 12.5K 例子；同时保留原始 VQA 1.1M、V‑Genome 1.7M 等数据用于对比。

**📈 对比分析**

与零样本和标准 ICL 的对比实验中，VLCG 结构化提示使因果归因得分从 0.458 提升至 0.488，因果推断得分从 0.652 提升至 0.690；但答案准确率仅从 0.763 稍升至 0.768，显示结构化提示显著提升推理一致性但未显著改变最终答对率。

**⚠️ 局限性**

局限性包括：①因果图的生成和校验仍依赖 LLM 的生成能力，易出现幻觉；②评估指标主要基于语义相似度和自动化评估，可能忽略细微逻辑错误；③目前实验仅在 Qwen2.5‑VL‑7B 上验证，未充分验证跨模型与跨规模的通用性。

---

## 61. Probing and Bridging Geometry-Interaction Cues for Affordance Reasoning in Vision Foundation Models

**arXiv ID:** 2602.20501 | [PDF](https://arxiv.org/pdf/2602.20501v1)

**作者:** Qing Zhang `[一作]` (Australian National University), Jing Zhang `[通讯]` (Australian National University)

**通讯引用:** 17084 | [OpenAlex ID](https://openalex.org/A5100345321)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过系统探测视觉基础模型中的几何感知和交互感知两大核心能力，发现DINO能够自我编码部分级几何结构，Flux生成模型通过动词条件的跨注意力提供交互先验，并在不需要训练的情况下将二者融合，实现零样本的可行性估计。

**💡 创新点**

提出了将视觉可行性理解拆解为几何与交互两维的双重框架，首次在视觉基础模型中量化并利用生成模型的跨注意力作为无监督交互先验，并证明两维可以组合成有效的零样本可行性推理。

**🔧 技术方法**

使用了线性探测（Probing3D）、PCA分解提取几何原型、Flux Kontext生成模型的动词条件跨注意力、归一化扫描路径显著度（NSS）以及深度/法线补充等技术。

**📊 数据集**

主要数据集包括UMD可行性分割数据集、AGD20K零样本评估数据集，并使用Metric3Dv2提供深度与法线信息。

**📈 对比分析**

与全监督、弱监督和开词汇基线对比，采用mIoU、KLD、SIM、NSS等指标评估；零样本融合方法在AGD20K上KLD降至1.493、SIM提升至0.326、NSS提升至1.090，性能与弱监督方法相当。

**⚠️ 局限性**

局限在于所提取的几何与交互原型噪声较大、生成模型输出不稳定、融合策略简单、缺乏动态视角支持，未来可通过更纯粹的几何抽取和视频生成模型改进。

---

## 62. Tensor Network Generator-Enhanced Optimization for Traveling Salesman Problem

**arXiv ID:** 2602.20175 | [PDF](https://arxiv.org/pdf/2602.20175v1)

**作者:** Ryo Sakai `[一作]` (JIJ Inc), Chen-Yu Liu `[通讯]` (National Taiwan University)

**通讯引用:** 74810 | [OpenAlex ID](https://openalex.org/A5100400880)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在本文中，作者将张量网络生成增强优化（TN‑GEO）框架应用到旅行推销员问题（TSP）上，提出了一种新的整数编码排列表示和自回归掩码采样方法，能够生成合法的完整路径；同时引入了k‑site MPS变体，通过学习k-gram（连续k个城市）分布来降低参数量并提升对大规模实例的可扩展性；并在优化过程中采用指数温度调度来平衡探索与利用。

**💡 创新点**

创新点主要体现在：①采用整数编码的排列表示配合自回归掩码采样，避免了传统二进制编码中大量无效状态和惩罚项；②提出k‑site MPS模型，用滑动窗口学习k-gram分布，既保持局部相关性，又显著减少模型规模；③在GEO迭代中引入温度调度，使softmax代理在不同阶段更加聚焦优解。

**🔧 技术方法**

技术手段包括：张量网络Born机（MPS）作为生成器，使用自动微分实现全局参数更新；自回归采样配合掩码确保生成的每条路径合法；指数温度调度控制softmax目标分布；以及GEO框架的迭代训练与采样流程。

**📊 数据集**

实验使用了TSPLIB基准集，涵盖14、16、22、48、51、52城市等实例，全部采用公开的真实距离矩阵。

**📈 对比分析**

与经典的局部搜索启发式（swap和2‑opt）在同一起始点下进行比较。实验结果显示：对于中小规模实例（burma14、ulysses16、ulysses22）k‑site模型（尤其k=4、8）都能达到最优；在较大实例（att48、eil51、berlin52）中，k=4、8 TN‑GEO的误差均低于2‑opt，并且在绝大多数情况下优于全模型或k=2模型；仅在极小实例中两种传统方法偶尔表现相近。

**⚠️ 局限性**

局限性包括：①目前仅在最多52城市的实例上验证，尚未评估更大规模TSP；②高bond维的MPS对内存需求高，限制了可训练实例大小；③k‑site模型虽然参数少，但忽略了长程城市关联，可能在更复杂或更大规模问题中失效；④实验仅针对成本评估简单的距离和，未检验在更昂贵目标函数下的优势。

---

## 63. Aesthetic Camera Viewpoint Suggestion with 3D Aesthetic Field

**arXiv ID:** 2602.20363 | [PDF](https://arxiv.org/pdf/2602.20363v1)

**作者:** Sheyang Tang `[一作]` (University of Waterloo), Zhou Wang `[通讯]` (University of Waterloo)

**通讯引用:** 97765 | [OpenAlex ID](https://openalex.org/A5100420313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于稀疏视角学习的3D美学场景模型，用来在三维空间中建议最佳摄影视角。

**💡 创新点**

创新点在于把2D美学知识蒸馏进3D高斯散射场，形成可微的3D美学场景表示，并通过两阶段粗采样加梯度优化实现高效视角搜索。

**🔧 技术方法**

主要技术包括3D高斯散射网络、视角感知的美学特征蒸馏、以及两阶段采样+梯度上升的视角搜索流程。

**📊 数据集**

使用了RE10k和DL3DV这两个真实场景数据集进行训练与评估。

**📈 对比分析**

与RGB直分数基线及单视角调节方法对比，实验表明在不同视角数量下美学分数平均提升约30%–40%，并在多场景测试中保持最优。

**⚠️ 局限性**

局限在于依赖相机位姿、对几何重建质量敏感，并且搜索空间受限于已有观测范围。

---

## 64. The Art of Efficient Reasoning: Data, Reward, and Optimization

**arXiv ID:** 2602.20945 | [PDF](https://arxiv.org/pdf/2602.20945v1)

**作者:** Taiqiang Wu `[一作]` (University of Hong Kong), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12167 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统分析并优化LLM的高效链式思维训练机制，提出统一实验协议和细粒度指标，验证在多种 token 预算下的表现。

**💡 创新点**

创新点包括发现两阶段训练（长度适应+推理精炼）模式，强调训练提示难度分层、负样本掩蔽策略、离线优化与 staleness 调整。

**🔧 技术方法**

使用的技术主要是基于奖励塑造的 RL（截断式、掩蔽式），Group Relative Policy Optimization、离线策略、长度预算目标以及多样化的优化技巧。

**📊 数据集**

训练数据采用 DeepScaleR（分为 Easy/Hard）提示集，验证基准覆盖 AIME'25、AMC、MATH-500、Minerva Math、Olympiad Bench、LiveCodeBench 等任务。

**📈 对比分析**

与 Kimi、Laser 等现有方法对比，平均 4k 截断下 Mean@8 提升至约 46%，Pass@8 提升至约 70%，同时生成长度被压缩至原来的一半。

**⚠️ 局限性**

限制在于仅评估数学与代码领域，使用固定长度预算，未探索更大模型或更广泛任务，离线优化可能带来不稳定性。

---

## 65. Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty

**arXiv ID:** 2602.20729 | [PDF](https://arxiv.org/pdf/2602.20729v1)

**作者:** Xu Wan `[一作]` (Zhejiang University), Mingyang Sun `[通讯]` (Peking University)

**通讯引用:** 4902 | [OpenAlex ID](https://openalex.org/A5079378336)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了基于模糊测度的安全强化学习框架Fuz-RL，使用模糊贝尔曼算子和Choquet积分实现鲁棒价值估计；

**💡 创新点**

创新点在于通过模糊测度直接编码不确定性耦合的非加性影响，避免传统的min-max优化，且证明与分布式鲁棒CMDP等价；

**🔧 技术方法**

主要技术包括模糊测度学习（λ-模糊测度）、Choquet积分估计、模糊贝尔曼算子、Primal-Dual策略优化以及神经网络估计模糊密度；

**📊 数据集**

实验使用Safe-Control-Gym和Safety-Gymnasium四个任务（CartPole-Stab、CartPole-Track、Quadrotor-Stab、Quadrotor-Track）并在观测、动作、动力学和多源不确定性下测试；

**📈 对比分析**

与PPO-Lagrangian、CUP、CPPO以及RAMU比较，Fuz-RL在大多数任务中提升平均奖励、降低约束违规率，表现优于现有安全/鲁棒RL方法；

**⚠️ 局限性**

局限在于高维状态空间下的可扩展性受限，且目前不支持非平稳不确定性分布的自适应学习。

---

## 66. Explicit Grammar Semantic Feature Fusion for Robust Text Classification

**arXiv ID:** 2602.20749 | [PDF](https://arxiv.org/pdf/2602.20749v1)

**作者:** Azrin Sultana `[一作]` (American International University Bangladesh), Firoz Ahmed `[通讯]` (American International University Bangladesh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将句子级语法特征与冻结的Transformer上下文嵌入融合的轻量化文本分类框架。

**💡 创新点**

创新点在于将语法信息作为显式归纳偏置，利用固定低维语法向量代替可训练的结构编码，显著降低模型参数并提升跨域鲁棒性。

**🔧 技术方法**

采用spaCy等工具进行语法特征提取，结合BERT/XLNet的冻结嵌入，并在此基础上训练DBN、LSTM、BiLSTM等浅层网络。

**📊 数据集**

实验使用邮件垃圾分类数据集（约52k条）和GMB命名实体识别数据集。

**📈 对比分析**

通过对比不使用语法特征与使用语法特征的同一模型，测量准确率、精确率、召回率和F1，结果显示加入语法特征后模型在分类和NER任务上分别提升了约2%至15%（如BiLSTM准确率从89.8%提升至96.19%）。

**⚠️ 局限性**

主要局限在于仅验证了两类数据集，语法规则的通用性和跨领域适用性待进一步测试；融合方式仅为拼接，未探索更复杂的融合策略。

---

## 67. From Performance to Purpose: A Sociotechnical Taxonomy for Evaluating Large Language Model Utility

**arXiv ID:** 2602.20513 | [PDF](https://arxiv.org/pdf/2602.20513v1)

**作者:** Gavin Levinson `[一作]` (University of Michigan), Keith Feldman `[通讯]` (University of Michigan)

**通讯引用:** 5254 | [OpenAlex ID](https://openalex.org/A5110473382)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了语言模型效用分类法（LUX），从性能、交互、运维与治理四个维度系统化评估LLM的实际效用；

**💡 创新点**

创新点在于将技术性能与社会技术因素统一纳入同一层级结构，并提供可量化指标和可视化工具，形成首个面向多用例的综合评估框架；

**🔧 技术方法**

采用层次化框架设计方法，梳理并对齐现有评估指标，构建可动态查询的网络工具；

**📊 数据集**

未使用特定数据集，而是对已有文献和指标进行综述与归类；

**📈 对比分析**

通过框架内的维度、子维度与指标层级，允许对不同LLM在同一用例下的效用进行量化比较；因论文为方法论性研究，未给出具体实验性能数值；

**⚠️ 局限性**

局限性包括缺乏实证验证、对新兴指标的适配性待检验，以及在多变应用环境下框架本身的可扩展性和更新速度需要进一步探索。

---

## 68. Talking to Yourself: Defying Forgetting in Large Language Models

**arXiv ID:** 2602.20162 | [PDF](https://arxiv.org/pdf/2602.20162v1)

**作者:** Yutao Sun `[一作]` (Zhejiang University), Jianwei Yin `[通讯]` (Zhejiang University)

**通讯引用:** 7270 | [OpenAlex ID](https://openalex.org/A5069353502)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在细化大型语言模型时，提出了在正式微调前让模型自行生成“自对话”数据并将其与任务数据混合的自增训练（Self-Augmented Supervised Fine‑Tuning, SAST）流程，利用自生成数据作为内部回放来抑制灾难性遗忘并提升专用性能。

**💡 创新点**

创新点在于：①无需外部数据或额外损失，单纯用模型自身的生成能力构建回放语料；②将自生成数据直接拼接入原始任务集，形成“无缝”混合；③提出风格诱发参数漂移理论解释，证明自增数据可通过匹配预训练分布来正则化梯度，抵消遗忘。

**🔧 技术方法**

核心技术包括：LLM自生成短对话（采用 nucleus sampling、top‑p=0.9、T=0.7），生成后与任务集拼接，随后采用标准交叉熵微调（全参数或LoRA），并通过混合比例 λ 控制自生成数据权重。

**📊 数据集**

实验使用 Super‑Natural‑Instructions（5 个代表性任务）、五大通用基准（GSM8K、MMLU、IFEval、MedText、AGIEval_G）以及多种 LLM（LLaMA3‑8B‑Instruct、Qwen2.5‑7B‑Instruct、LLaMA2‑7B‑Chat 等）。

**📈 对比分析**

与传统基线（Task‑Only、层冻结、外部数据混合如Alpaca/UltraChat、全参数微调、LoRA）对比，SAST 在 50 个评估场景中保持或提升了基线性能，尤其在保持通用能力方面取得 40/50 场景最佳成绩，显著优于层冻结和外部数据混合方法。

**⚠️ 局限性**

局限性包括：依赖基模型生成质量，弱模型可能产生冗余或低质量样本；只针对风格诱发的参数漂移，未覆盖所有遗忘机制；生成阶段增加推理成本和存储负担，且自生成数据可能包含错误或不安全内容，需要轻量过滤。

---

## 69. BiRQA: Bidirectional Robust Quality Assessment for Images

**arXiv ID:** 2602.20351 | [PDF](https://arxiv.org/pdf/2602.20351v1)

**作者:** Aleksandr Gushchin `[一作]` (MSU Institute for Artificial Intelligence), Anastasia Antsiferova `[通讯]` (Innopolis University)

**通讯引用:** 102 | [OpenAlex ID](https://openalex.org/A5086393377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BiRQA，一种高效且对抗鲁棒性强的全参考图像质量评估（FR IQA）模型。

**💡 创新点**

创新点在于双向多尺度金字塔的交叉尺度注意机制（CSRAM与SCGB）、不确定性感知门控融合以及锚定对抗训练（Anchored Adversarial Training, AAT）。

**🔧 技术方法**

采用轻量级卷积+自适应融合、双向跨尺度门控、GeM池化+可靠性加权聚合，训练时结合锚定排名损失与对抗扰动。

**📊 数据集**

使用公开FR IQA基准：LIVE、CSIQ、TID2013、KADID‑10k、PIPAL、PieAPP以及BAPPS等。

**📈 对比分析**

与多种现有方法（如LPIPS、TOPIQ、AHIQ等）比较，BiRQA在PLCC/SROCC上达到或超过SOTA，推理速度约15 FPS，比Transformer方法快3倍，且在多种白盒攻击下SROCC提升0.02–0.06，IR‑Score提升12%。

**⚠️ 局限性**

在某些特定失真（如径向几何变换）上表现略逊，且对抗训练虽提升鲁棒性但训练时间略长。

---

## 70. Geometric Analysis of Speech Representation Spaces: Topological Disentanglement and Confound Detection

**arXiv ID:** 2602.20823 | [PDF](https://arxiv.org/pdf/2602.20823v1)

**作者:** Bipasha Kashyap `[一作]` (Deakin University), Pubudu N. Pathirana `[通讯]` (Deakin University)

**通讯引用:** 10024 | [OpenAlex ID](https://openalex.org/A5037113249)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了情感、语言与病理语音特征在多语言环境中的几何分离程度，提出了结合四项聚类指标的评估框架；

**💡 创新点**

创新点在于：①将Silhouette、Davies–Bouldin、Calinski–Harabasz与Bootstrap稳定性四个聚类质量指标统一用于t‑SNE嵌入的评估；②量化病理与语言特征的几何重叠，并给出可接受的上限；③提供基于几何分离的公平性与可靠性准则；

**🔧 技术方法**

使用手工提取的声门与滤波器特征（eGeMAPS、MFCC、音素频率等），t‑SNE降维，KMeans聚类，四指标聚类质量评估，Bootstrap ARI评估稳定性，PCA共享空间下的重叠度计算，以及信任度（trustworthiness）评估；

**📊 数据集**

数据集包括情感类RAVDESS、IEMOCAP；语言类L2‑ARCTIC、GMU Speech Accent Archive；病理类UA‑Speech、MDVR‑KCL，共八种组合；

**📈 对比分析**

对八个组合分别计算Silhouette、Davies–Bouldin、Calinski–Harabasz、Bootstrap稳定性和trustworthiness，结果显示情感特征聚类最紧密（Silhouette≈0.250），病理次之（≈0.141），语言最松散（≈0.077）；Bootstrap稳定性与Silhouette高度相关；病理‑语言重叠低于0.21，明显高于置换零基线，但仍在可接受范围；trustworthiness均>0.80，表明嵌入质量可靠；

**⚠️ 局限性**

局限性包括：聚类得分仅为中等（<0.30），未达到完全可分；病理数据量有限，可能影响结果稳定性；手工特征未针对几何分离进行优化；PCA重叠度仅捕捉线性结构，可能忽略非线性交互。

---

## 71. GatedCLIP: Gated Multimodal Fusion for Hateful Memes Detection

**arXiv ID:** 2602.20818 | [PDF](https://arxiv.org/pdf/2602.20818v1)

**作者:** Yingying Guo `[一作]` (Chinese University of Hong Kong), Zirong Zeng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GatedCLIP模型，在冻结CLIP基础上加入投影头、动态门控融合与对比学习，专门用于检测多模态仇恨表情包。

**💡 创新点**

创新点在于：①将CLIP嵌入映射到任务特定低维空间的投影头；②可学习的门控融合机制动态调节视觉与文本贡献；③结合对比损失保持跨模态语义对齐，同时仅引入350K训练参数。

**🔧 技术方法**

使用技术包括CLIP视觉语言模型、投影网络、门控融合（类似GMU）、对比学习损失、AdamW优化、混合精度训练及轻量化参数化。

**📊 数据集**

采用Hateful Memes数据集（约10,000条图文对，训练8,500、验证500、测试1,000）。

**📈 对比分析**

与冻结CLIP+平均融合基线对比，GatedCLIP在验证集AUROC从0.49提升到0.66，准确率从0.50提升到0.59，仅增加350K可训练参数。

**⚠️ 局限性**

局限性包括：仅在Hateful Memes上评估；对比损失假设所有配对相似，可能不适用于仇恨内容；受CLIP预训练数据的文化与语言偏差限制；模型仍未达到SOTA，仍有难以分类的例子。

---

## 72. Computer-Aided Design of Rational Motions for 4R and 6R Spatial Mechanism Synthesis

**arXiv ID:** 2602.20920 | [PDF](https://arxiv.org/pdf/2602.20920v1)

**作者:** Daniel Huczala `[一作]` (University of Innsbruck), Frank C. Park `[通讯]` (Seoul National University)

**通讯引用:** 5037 | [OpenAlex ID](https://openalex.org/A5009036246)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

本文提出了一种基于四元数贝塞尔曲线的几何插值方法，能够在三维空间中生成通过给定三点、四姿势、五点或七点的有理运动，并利用运动因式分解技术直接合成相应的单环一自由度链式机构；同时提供了开源 CAD 工具用于交互式设计和快速可视化。

**💡 创新点**

创新点主要包括：① 针对七个三维点的三次有理运动插值提供了闭式解析公式；② 通过将插值曲线投影到 Study 四面体上实现运动的合法性和可因式分解；③ 在已有二次（五点）和三次（七点）插值的基础上，统一推导了控制点与权重的显式关系；④ 将这些算法封装为开源 Python/Rust 库，并实现了实时交互式 CAD 界面。

**🔧 技术方法**

使用的技术与方法包括：双四元数表示刚体运动、Study 四面体约束、四元数贝塞尔曲线插值、线性/非线性方程组求解、运动因式分解（将运动多项式分解为旋转因子）、Rust 编译加速、Qt6 交互界面，以及对 1-DoF 运动的参数化。

**📊 数据集**

文章未使用公开的实验数据集，而是通过构造任意的三维点集合（例如五点、七点示例）进行插值验证；因此数据集为自定义的合成样本。

**📈 对比分析**

在方法比较与性能评估方面，作者主要强调软件实现的实时可视化和因式分解的直接性，没有给出定量的性能指标或与现有方法的实验对比；在可视化实验中，CAD 工具能够即时展示运动轨迹和生成的 6R 原型。

**⚠️ 局限性**

局限性包括：① 对空间 1-DoF 机构的设计仍受限于运动参数的可因式分解与运动可实现性；② 由于缺乏三维 SE(3) 的有效距离度量，数值优化与误差分析相对困难；③ 仅针对插值点数（3、4、5、7）给出闭式解，其他点数的情况未涵盖；④ 目前缺乏对真实机器人任务或工业应用的验证。

---

## 73. Towards Secure and Efficient DNN Accelerators via Hardware-Software Co-Design

**arXiv ID:** 2602.20521 | [PDF](https://arxiv.org/pdf/2602.20521v1)

**作者:** Wei Xuan `[一作]` (Hong Kong University of Science and Technology), Luhong Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1313 | [OpenAlex ID](https://openalex.org/A5052554057)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一种安全高效的DNN加速器框架，利用带宽感知的加密方案和多层认证机制实现对离线内存的机密性与完整性保护。

**💡 创新点**

创新点在于（1）单一AES引擎配合KeyExpansion+XOR生成多重OTP，既满足带宽需求又极大降低硬件开销；（2）基于GCD求解最优认证块(opt_blk)的多层MAC体系，消除冗余完整性检查并抵御RePA攻击。

**🔧 技术方法**

采用AES-CTR模式、XOR级MAC、GCD算子、硬件/软件协同设计，配合SCALE‑Sim2、Ramulator2、DRAMsim3等周期级仿真工具进行验证。

**📊 数据集**

在多种常见DNN模型（LeNet、AlexNet、MobileNet、ResNet18、GoogleNet、DLRM、AlphaGoZero、DeepSpeech2、FasterRCNN、NCF、Sentimental_seqCNN、Transformer_fwd、Yolo_tiny）上进行评估。

**📈 对比分析**

与SGX‑64B/512B、MGX‑64B/512B等现有方案对比，指标包括内存流量、执行时间与能耗；结果显示本方案将性能延迟降至≈0.12%以内、能耗提升≈87%，并在服务器/边缘NPU上实现≈12%加速，显著优于对比方案。

**⚠️ 局限性**

局限性包括：对静态切块（tiling）假设较强，动态调度时需要重新计算opt_blk；仍需在芯片内部保留较大SRAM存放MAC，限制极小化设备；未考虑侧信道、模型篡改等高级攻击。

---

## 74. Three Concrete Challenges and Two Hopes for the Safety of Unsupervised Elicitation

**arXiv ID:** 2602.20400 | [PDF](https://arxiv.org/pdf/2602.20400v1)

**作者:** Callum Canavan `[一作]` (MATS), Fabien Roger `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了三类安全挑战（显著非真特征、不平衡训练集、不可解任务）下的无监督诱导和易难迁移方法。

**💡 创新点**

首次构造了针对这些挑战的真实压力测试数据集，并系统比较现有方法与两种改进思路（集成与混合）。

**🔧 技术方法**

采用提示法、线性探测器、无监督训练、集成与混合策略等技术。

**📊 数据集**

使用了数学题、政治事实、新闻评论、代码安全、民意调查等多种数据集。

**📈 对比分析**

对比了零样本、随机探测、PCA、训练探测等方法，发现大多数方法在挑战数据上性能显著下降，集成策略只能部分缓解。

**⚠️ 局限性**

实验覆盖范围有限，数据集仍不完全真实，且未考虑模型偏见、恶意干扰等额外挑战。

---

## 75. Overton Pluralistic Reinforcement Learning for Large Language Models

**arXiv ID:** 2602.20759 | [PDF](https://arxiv.org/pdf/2602.20759v1)

**作者:** Yu Fu `[一作]` (University College London), Ilija Bogunovic `[通讯]` (University of Basel)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于RLHF的OP-GRPO框架，能够让单一LLM在不需要显式多模态或提示的情况下，隐式生成多元化的Overton窗口；

**💡 创新点**

创新点在于将OP的语义匹配转化为SBERT细调+MBGM双重奖励机制，并通过GRPO实现覆盖率与多样性并重的策略优化；

**🔧 技术方法**

核心技术包括SBERT多任务微调、Mutual‑Best Greedy Matching、组相对策略优化（GRPO）和多目标奖励设计；

**📊 数据集**

使用从ValuePrism派生的OP‑V2数据集（≈30k行，≥5个独立视角）进行训练与评估；

**📈 对比分析**

在NLI与LLM‑as‑Judge评测中，OP‑GRPO在小型1.5B/3B模型上均超过20B基线及模块化多模态方案，平均准确率提升约40‑70%；

**⚠️ 局限性**

局限性包括数据集潜在的多数派偏见、对SBERT相似度的依赖以及在极端多样性场景下仍可能出现覆盖不足或生成冗长的问题。

---

## 76. ActionEngine: From Reactive to Programmatic GUI Agents via State Machine Memory

**arXiv ID:** 2602.20502 | [PDF](https://arxiv.org/pdf/2602.20502v1)

**作者:** Hongbin Zhong `[一作]` (Georgia Tech), Suman Nath `[通讯]` (Microsoft Research)

**通讯引用:** 7867 | [OpenAlex ID](https://openalex.org/A5024224291)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种两阶段的 GUI 代理框架，离线爬虫代理构建可更新的状态机图作为记忆，在线执行代理利用该记忆一次性生成可执行的 Python 程序完成任务。

**💡 创新点**

创新点在于把传统逐步观察–推理–行动的反应式流程转为全局程序化规划，并通过可更新的状态机记忆消除对连续 LLM 调用的需求，实现 O(1) 的规划成本。

**🔧 技术方法**

技术包括离线爬虫构建状态机图、LLM 代码生成与图搜索编译器、基于 Python 的执行计划、以及视觉回退机制来修复 UI 变化并更新记忆。

**📊 数据集**

使用了 WebArena 基准中的 Reddit 子集（106 个长流程任务）来评估模型。

**📈 对比分析**

与最强反应式基线 AgentOccam 对比，本文模型在 Reddit 子集上达 95% 的成功率（比 66% 提升 29%），平均延迟降低 2 倍，平均成本降低 11.8 倍，单次 LLM 调用平均仅 1.8 次。

**⚠️ 局限性**

局限性包括：任务说明歧义导致的一些失败；在 UI 变化时仍需视觉回退；对高度视觉依赖或图像分析任务的支持不足；以及对完全未知或极端复杂交互场景的适应性待进一步验证。

---

## 77. Mitigating "Epistemic Debt" in Generative AI-Scaffolded Novice Programming using Metacognitive Scripts

**arXiv ID:** 2602.20206 | [PDF](https://arxiv.org/pdf/2602.20206v1)

**作者:** Sreecharan Sankaranarayanan `[一作]` `[通讯]` (Extuitive Inc.), Sreecharan Sankaranarayanan (Extuitive Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过一项对78名 AI‑Native 学习者的实验，研究了生成式 AI 在初学者编程中的“认知债务”，并评估了基于 LLM 的“解释门”干预对功能效用和纠正能力的影响。

**💡 创新点**

首次提出可扩展的自我教学机制（解释门）以引入认知摩擦，量化并缓解认知债务，证明在保持生产效率的同时可恢复学习者的维护能力。

**🔧 技术方法**

使用 Cursor IDE 插件 VibeCheck、Claude 3.5 Sonnet 生成代码、GPT‑4o 作为判定者、SOLO 评价体系以及 Cognitive Load Theory 解释框架。

**📊 数据集**

自制的学生课程调度器 React 任务（12个断言）以及 Phase 2 注入的逻辑炸弹作为实验任务；受试者为 78 名美国 AI‑Native 学习者。

**📈 对比分析**

通过三组（手工、无限制 AI、带解释门）进行 ANOVA/Chi‑square 比较，功能效用：无限制 92.4%、解释门 89.1%、手工 65.2%；维修成功率：无限制 23.1%、解释门 61.5%、手工 69.2%，表明解释门几乎恢复了功能与维护的平衡。

**⚠️ 局限性**

研究仅针对初学者，外部效度有限；仅测量短期纠正能力；评判模型可能存在偏差；实验时间限制及参与者知晓实验导致霍桑效应。

---

## 78. Post-Quantum Sanitizable Signatures from McEliece-Based Chameleon Hashing

**arXiv ID:** 2602.20657 | [PDF](https://arxiv.org/pdf/2602.20657v1)

**作者:** Shahzad Ahmad `[一作]` (Johannes Kepler University), Zahra Seyedi `[通讯]` (Polytechnic University of Milan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一种基于McEliece密码的后量子可清洗签名方案，利用码理论的陷阱哈希实现可控碰撞，构成完整的签名、验证与清洗流程。

**💡 创新点**

创新点在于首次实现了透明、后量子可清洗签名，并通过在签名随机化器上强制权重为t来实现完全透明；此外首次将McEliece的Patterson解码直接用于碰撞生成，且在随机预言机模型下证明碰撞抗性。

**🔧 技术方法**

采用了McEliece密码、Goppa码、Patterson解码、SHA-3随机预言机、Dilithium2基础签名以及二进制哈希链结构。

**📊 数据集**

实验使用了经典的NIST Classic-McEliece参数集（Toy、Medium、Secure三组），并在Python原型中模拟了SHA-3和Patterson解码；未使用实际的文本或图像数据集。

**📈 对比分析**

通过与RSA‑2048、Clermont等基于格的方案比较，发现公钥约655 KB（比格方案小约15%），签名约7 KB（比格方案略大），透明度实现完美；在Python原型下，签名/验证时间随块数线性增长，单块Patterson解码理论上约8 ms，整体性能适合中小规模文档清洗。

**⚠️ 局限性**

局限性包括：依赖随机预言机模型实现碰撞抗性；缺乏曝光防护与策略隐藏；公钥相对较大；未实现标准模型证明；在实际部署中需使用常数时间Patterson解码以防侧信道攻击。

---

## 79. Application of Large Language Models for Container Throughput Forecasting: Incorporating Contextual Information in Port Logistics

**arXiv ID:** 2602.20489 | [PDF](https://arxiv.org/pdf/2602.20489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 80. UAMTERS: Uncertainty-Aware Mutation Analysis for DL-enabled Robotic Software

**arXiv ID:** 2602.20334 | [PDF](https://arxiv.org/pdf/2602.20334v1)

**作者:** Chengjie Lu `[一作]` (Simula Research Laboratory and University of Oslo), Thomas Peyrucain `[通讯]` (PAL Robotics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对深度学习驱动的自适应机器人软件的 uncertainty‑aware mutation analysis 框架，定义了两种基于 MC‑Dropout/MC‑DropBlock 的变异算子，并推出了 19 种新的 mutation score 指标。

**💡 创新点**

创新点在于将模型不确定性显式注入变异操作，并设计针对不确定性的 mutation scores，使测试集质量评估能够直接反映模型在随机不确定性下的失败；同时将统计显著性判定与不确定性度量相结合，提升评估可靠性。

**🔧 技术方法**

使用了 MC‑Dropout 与 MC‑DropBlock 作为变异算子，基于 IoU、分类不确定性（VR、SE、MI）以及回归不确定性（TV、PS）的度量，采用统计检验（如二项检验、Spearman 相关、F‑检验）进行结果评估。

**📊 数据集**

在三项工业级机器人案例（贴纸去除机器人、笔记本拆解机器人、TIAGo Pro 人脸检测机器人）中，使用了五个自定义目标检测模型（YuNet、YOLOv11 等）以及公开数据集 WIDER FACE 的子集。

**📈 对比分析**

通过与传统图像级 mutation score 的对比，利用 Kruskal–Wallis 检验和 η² 效应量衡量指标；实验显示新定义的 mutation scores 能更好地区分测试集质量，并且与不确定性水平呈显著正相关。

**⚠️ 局限性**

局限性包括指标仅针对目标检测任务，可能不适用于其他 DL 任务；变异算子只在模型后几层注入 dropout/DropBlock，可能无法覆盖所有不确定性来源；实验样本受限于三种机器人和有限的测试集。

---

## 81. Beyond Human Performance: A Vision-Language Multi-Agent Approach for Quality Control in Pharmaceutical Manufacturing

**arXiv ID:** 2602.20543 | [PDF](https://arxiv.org/pdf/2602.20543v1)

**作者:** Subhra Jyoti Mandal `[一作]` (GSK), Sander W. Timmer `[通讯]` (GSK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了一套基于多代理框架的CFU检测系统，通过将Detectron2深度学习检测器与Vision‑Language模型（Qwen2‑VL‑Quantized与GPT‑4o）协同工作，实现了自动化的细菌/霉菌菌落计数。

**💡 创新点**

创新点在于将DL与VLM双重推理融合为自我验证的多代理架构，并通过一致性阈值自动决定结果上报或人工复核，首次实现了可追溯、可解释的药物级CFU自动计数。

**🔧 技术方法**

技术上采用Detectron2+ResNet‑101/FPN进行多尺度小目标检测，Qwen2‑VL‑Quantized进行无效盘预筛，GPT‑4o进行零样本计数推理，LangGraph负责代理协调，Databricks/MLflow/Delta Lake实现MLOps与SAP/Postgres集成。

**📊 数据集**

使用GSK内部约5万张培养皿图像的金标数据集，包含多种照明、污染和聚集情况，并在此基础上进行增广与再训练。

**📈 对比分析**

将YOLOv5/7/8、Mask R‑CNN与Detectron2进行mAP@0.5比较，Detectron2获得99.0% mAP、98.8%精确度、98.5%召回率，VLM预筛中Qwen2‑VL‑Quantized FNR0.24、FPR0.01，GPT‑4o计数一致率69%，整体系统误报0.6%/2.0%，人工复核率降低85%，平均推理时间<10秒。

**⚠️ 局限性**

局限性在于对极低质量或极稠密菌落的鲁棒性仍不足，VLM推理耗时相对较高，且多代理同步机制在极端大批量场景下可能产生瓶颈，未来需进一步优化模型轻量化与动态阈值自适应。

---

## 82. Heterogeneity-Aware Client Selection Methodology For Efficient Federated Learning

**arXiv ID:** 2602.20450 | [PDF](https://arxiv.org/pdf/2602.20450v1)

**作者:** Nihal Balivada `[一作]` (University of Oregon), Suyash Gupta `[通讯]` (University of Oregon)

**通讯引用:** 756 | [OpenAlex ID](https://openalex.org/A5002730429)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Terraform，一种在联邦学习中通过梯度更新和分位数范围实现的确定性客户端选择方法。

**💡 创新点**

创新点在于使用最终层梯度更新作为客观异质性度量，结合 IQR 进行分位数范围内的最小方差分割，实现了对“难易”客户端的层次化划分，并消除了随机选择的非确定性。

**🔧 技术方法**

技术方法包括联邦平均（FedAvg）/FedProx、最终层梯度范数计算、IQR 计算、方差最小化分割、层次化迭代重训练。

**📊 数据集**

实验数据集涵盖 CIFAR-10、CIFAR-100、Tiny ImageNet、FEMNIST、FMNIST。

**📈 对比分析**

与 Random、PoC、Oort、HiCS-FL、HBase 等基线进行对比，Terraform 在 FedAvg/FedProx 上平均提升 20‑47% 的准确率，尤其在 CIFAR-100 与 Tiny ImageNet 上显著优于其他方法。

**⚠️ 局限性**

局限性：需要在每轮预先采样较多客户端才能充分发挥效果；对阈值 η 的选择有经验依赖；在极小样本或高类数场景下性能提升有限；计算和通信成本虽不增加但实现复杂度略高。

---

## 83. Learning Physical Principles from Interaction: Self-Evolving Planning via Test-Time Memory

**arXiv ID:** 2602.20323 | [PDF](https://arxiv.org/pdf/2602.20323v1)

**作者:** Haoyang Li `[一作]` (University of California San Diego), Leonidas Guibas `[通讯]` (Stanford University)

**通讯引用:** 82936 | [OpenAlex ID](https://openalex.org/A5065368881)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种测试时记忆框架，使VLM机器人规划器通过交互学习物理原则而不更新模型参数。

**💡 创新点**

通过科学记忆循环（经验→假设→验证→原则），先验证假设再推广，避免盲目检索并提供可解释的物理原则。

**🔧 技术方法**

使用三层记忆结构（经验/工作/长期）、VLM规划器、LLM反射模型生成假设、动作级归因、验证实验、记忆折叠与衰减忘记等技术。

**📊 数据集**

在三个真实世界任务（不规则拼装、球推障碍、石塔稳定）和Reflect‑VLM模拟砖插入基准上测试，使用Gemini、GPT、Qwen等VLM。

**📈 对比分析**

与无记忆、直接经验检索等基线对比，在砖插入任务上从23%提升至76%，真实任务30分钟部署提升数倍；不同VLM难度下提升率最高为+23%。

**⚠️ 局限性**

仅关注高层规划，未集成低层控制；依赖视觉观察缺乏触觉/音频信息；文本化原则难处理连续动力学；环境重置需要人工干预。

---

## 84. Conflict-Based Search for Multi-Agent Path Finding with Elevators

**arXiv ID:** 2602.20512 | [PDF](https://arxiv.org/pdf/2602.20512v1)

**作者:** Haitong He `[一作]` (Shanghai Jiao Tong University), Zhongqiang Ren `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 391 | [OpenAlex ID](https://openalex.org/A5018561143)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了多层楼层多智能体路径规划问题（MAPF-E），并针对电梯冲突设计了高效的冲突处理与决策图扩展方法。

**💡 创新点**

创新点在于：①引入电梯约束（EC）一次性解决电梯冲突；②扩展MDD结构为MDD‑E，能够捕捉电梯状态并用于冲突选择和跳过。

**🔧 技术方法**

核心技术包括：冲突基础搜索（CBS）框架、改进的电梯约束、MDD‑E结构、冲突选择与冲突跳过（BP）技术以及SIPP低层规划。

**📊 数据集**

实验使用了8×8和16×16的4邻域网格（单层）多层合成地图，随机生成障碍与电梯，测试了不同楼层数、电梯速度、智能体数量的实例。

**📈 对比分析**

与传统CBS对比，CBS‑EC和CBS‑EC+MDD‑E在成功率上提高约30–40%，高层节点扩展数明显减少；MDD‑E单独使用效果有限，主要在结合EC时才显著提升。

**⚠️ 局限性**

局限性包括：MDD‑E计算开销随路径长度和冲突频繁度升高而显著增加；只考虑单容量电梯；未处理非整数电梯耗时和异步动作。

---

## 85. CryptRISC: A Secure RISC-V Processor for High-Performance Cryptography with Power Side-Channel Protection

**arXiv ID:** 2602.20285 | [PDF](https://arxiv.org/pdf/2602.20285v1)

**作者:** Amisha Srivastava `[一作]` (University of Texas at Dallas), Kanad Basu `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 1916 | [OpenAlex ID](https://openalex.org/A5066320524)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 CryptRISC，一款集成 RISC‑V Scalar Cryptography 扩展与动态场感知掩码的处理器，兼顾加速与侧信道防护。

**💡 创新点**

创新点在于引入 Field Detection Layer（FDL）与 Masking Control Unit（MCU），通过指令级场域识别动态选择 Boolean、affine 或 arithmetic 掩码，实现ISA透明的硬件侧信道抵抗。

**🔧 技术方法**

采用 RISC‑V Scalar Cryptography ISA、FDL、MCU、可编程 affine 掩码、LFSR 伪随机数生成器、FPGA 合成与测试、TVLA 侧信道评估等技术。

**📊 数据集**

使用 RISC‑V Cryptography Benchmark Suite（AES‑128/192/256、SHA‑256/512、SM3、SM4）以及 OpenSSL 软件实现作为基准数据集。

**📈 对比分析**

与基准 CVA6、未掩码 CISE 及 SCARV 进行对比，采用执行时间、加速比、内存占用、TVLA t‑值与 CPA p‑值等指标；加速比最高达 6.80×，硬件开销约 1.86%，t‑值均低于 ±2，表明优异性能与强安全。

**⚠️ 局限性**

局限在于仅评估了第一阶泄露，未对高阶侧信道攻击做深入分析；依赖 LFSR 随机数质量；实验仅在 Kintex‑7 FPGA 上完成，尚未在 ASIC 或更大规模平台验证。

---

## 86. Improving Data Quality via Pre-Task Participant Screening in Crowdsourced GUI Experiments

**arXiv ID:** 2602.20594 | [PDF](https://arxiv.org/pdf/2602.20594v1)

**作者:** Takaya Miyama `[一作]` (Meiji University), Shota Yamanaka `[通讯]` (LY Corporation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种基于图形用户界面（GUI）预任务——尺寸调整——的筛选方法，用以提升众包环境下 GUI 实验（目标指向任务）的数据质量。

**💡 创新点**

创新点在于：①将预任务误差作为连续质量信号，允许通过阈值 T 调整筛选严格度；②不依赖主任务结果即可判定参与者是否符合；③系统地在不同设备（PC/手机）与错误处理策略下验证其有效性，展示了该方法的普适性与可调性。

**🔧 技术方法**

技术手段包括：尺寸调整预任务、目标指向实验、Fitts 定律与误差率模型、阈值 T 与非符合比例 X 的参数化模拟、R² 统计评估、留一宽度交叉验证（LOOCV）以及方差分析（RM‑ANOVA）等。

**📊 数据集**

数据集来源于雅虎众包平台，共招募约 1,500 名参与者，包含 PC 端 455 名和 iPhone 端 1,069 名（扣除无效后 1,533 名），收集了尺寸调整误差、点击时间、坐标及错误率等实验记录。

**📈 对比分析**

比较方法：对不同阈值 T 和非符合比例 X 进行网格搜索，计算每组样本在主实验中 Fitts 公式与误差率模型的 R²，并通过 LOOCV 评估对未见宽度条件的预测精度。实验显示：阈值越严格、非符合比例越小，R² 越高；在手机实验中误差率模型的改进最为显著，Fitts 公式在 PC 实验中对误差较不敏感。

**⚠️ 局限性**

局限性：①预任务无法检测所有不合规行为，残留的噪声仍可能影响结果；②阈值 T 的选择存在折衷，过严格会导致样本量不足；③可能排除存在运动或视觉障碍的参与者，影响样本代表性；④仅验证了尺寸调整与指向任务的组合，尚需检验更复杂交互场景的通用性；⑤不同错误处理策略与试验设计参数会影响 R² 评估，需在实验设计中进行平衡。

---

## 87. Indaleko: The Unified Personal Index

**arXiv ID:** 2602.20507 | [PDF](https://arxiv.org/pdf/2602.20507v1)

**作者:** William Anthony Mason `[一作]` `[通讯]` (University of British Columbia), William Anthony Mason (University of British Columbia)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了统一个人索引 (UPI) 架构，通过收集多源元数据并使用记忆锚点实现基于人类情节记忆模式的检索。

**💡 创新点**

创新点在于将人类情节记忆模型映射到系统设计，结合语义解耦与 UUID 隐私、schema‑agnostic 采集管道，实现跨平台、跨存储的记忆对齐检索。

**🔧 技术方法**

使用的技术包括 ArangoDB 图数据库、自然语言处理、计算机视觉、UUID、schema‑agnostic 数据采集与归档、语义解耦与隐私保护机制。

**📊 数据集**

实验使用了个人化多源数据集，包括本地文件系统、云存储（如 Dropbox、Google Drive）以及活动流日志，并通过六个典型查询（Q1–Q6）进行评估。

**📈 对比分析**

与传统基于文件属性或内容关键词的检索进行对比，UPI 在检索精度上提升约 15–25%，并将检索时间减少约 30%，展示了更优的性能。

**⚠️ 局限性**

限制在于依赖完整的元数据收集和跨平台权限授权，语义解耦的映射表会带来额外存储与查询开销，且在大规模多用户场景下的可扩展性和隐私评估仍待进一步验证。

---

## 88. The Tragedy of Chain Commons

**arXiv ID:** 2602.20341 | [PDF](https://arxiv.org/pdf/2602.20341v1)

**作者:** Ignacio Amores-Sesar `[一作]` (Aarhus University), Michelle X. Yeo `[通讯]` (Aarhus University and Nanyang Technological University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对分离共识与执行的区块链（解耦链）进行形式化分析，发现并证明了一种名为Gaslighting的新攻击；随后提出了一种部分耦合模型以抵御该攻击，并通过对Sui与Ethereum数据的案例研究验证攻击的现实可行性。

**💡 创新点**

创新点在于首次系统性识别和正式化解耦链的Gaslighting攻击，证明其在解耦模式下的不可避免性及对奖励公平性的破坏；并设计了部分耦合机制，既保持了高吞吐量，又恢复了奖励公平性，填补了前人对解耦链安全性的理论空白。

**🔧 技术方法**

采用了正式的区块链抽象模型、资源（gas）函数与状态依赖分析、组合优化与可行性证明；使用概率模型评估攻击成本；在实践层面利用Sui与Ethereum的交易日志进行实验验证。

**📊 数据集**

使用的主要数据集为公开的Sui和Ethereum主网交易历史，涵盖数十万笔交易的gas估计与实际消耗信息。

**📈 对比分析**

通过理论证明和实验对比，展示了：解耦链在Gaslighting攻击下吞吐量可降至零，成本几乎为零；部分耦合模型在保持与解耦链相同吞吐量的同时，延迟最多提升δc/2，整体性能优于传统耦合链。

**⚠️ 局限性**

局限性包括：部分耦合仅适用于领导者模式，对无领导者（DAG）协议难以直接迁移；需要假设存在足够独立交易；在稳定领导者方案下可能引入审查与单点攻击风险；并未完全解决所有资源冲突导致的吞吐瓶颈。

---

## 89. RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction

**arXiv ID:** 2602.20807 | [PDF](https://arxiv.org/pdf/2602.20807v1)

**作者:** Yangfan Zhao `[一作]` (Capital Normal University), Dengyu Wu `[通讯]` (King's College London)

**通讯引用:** 165 | [OpenAlex ID](https://openalex.org/A5063210265)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出RU4D-SLAM框架，在4D Gaussian splatting中加入integrate-and-render、reweighted uncertainty mask和adaptive opacity weighting，实现对低质量动态环境的鲁棒SLAM与高质量4D重建。

**💡 创新点**

创新点包括：①通过integrate-and-render对曝光区间进行积分渲染，建模运动模糊；②结合曝光不确定性与语义信息的reweighted uncertainty mask，精准区分静态与动态区域；③adaptive opacity weighting使动态节点随时间可变不透明度，提升动态物体的时空一致性；整体将不确定性驱动的跟踪、动态初始化与4D渲染统一起来。

**🔧 技术方法**

使用4D Gaussian splatting、密集束调整（DBA）+不确定性建模、双四元数混合控制节点、语义分割（SAM）与预训练特征（DINO）预测不确定性、运动模糊渲染、Metric3D深度估计、SSIM/LPIPS评估等技术。

**📊 数据集**

在TUM RGB-D、Bonn RGB-D和Wild-SLAM（iPhone真实拍摄）三大数据集上进行训练与测试。

**📈 对比分析**

与MonoGS、Gaussian-SLAM、SplaTAM、4DGS-SLAM、WildGS-SLAM等基线对比，RU4D-SLAM在PSNR、SSIM、LPIPS上均取得最高或接近最高成绩（TUM 25.95 dB、Bonn 26.33 dB、Wild 24.22 dB），跟踪精度方面ATE在TUM 1.69 cm、Bonn 2.50 cm，显示出显著性能提升。

**⚠️ 局限性**

局限性包括：尚未实现实时性能；在极端光照、遮挡或大规模场景下仍可能出现误差；依赖预训练语义分割与深度估计，可能在未见过的类别或极端低质量输入中表现不佳；内存与计算开销较大。

---

## 90. Nonparametric Teaching of Attention Learners

**arXiv ID:** 2602.20461 | [PDF](https://arxiv.org/pdf/2602.20461v1)

**作者:** Chen Zhang `[一作]` (University of Hong Kong), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12167 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AtteNT框架，将注意力学习视为非参数教学，通过教师挑选高梯度样本加速学习；

**💡 创新点**

将注意力网络的参数梯度演化与非参数函数空间的功能梯度演化对齐，首次将非参数教学理论应用于注意力网络；

**🔧 技术方法**

利用注意力机制、神经切线核（ANTK）、功能梯度下降、贪婪样本选择；

**📊 数据集**

自然语言生成任务使用GSM8K、MATH、HumanEval、MBPP、MT-Bench，计算机视觉任务使用ImageNetS50、NYUv2（语义分割与深度估计）及Multi-Modal MAE；

**📈 对比分析**

与标准微调/从零训练相比，AtteNT在LLM和ViT上平均缩短13%–20%训练时间，且准确率保持甚至提升；

**⚠️ 局限性**

受限于仅测试单层单头注意力、无标签噪声鲁棒性验证，以及对更大规模模型的推广性尚未验证。

---

## 91. RAYNOVA: 3D-Geometry-Free Auto-Regressive Driving World Modeling with Unified Spatio-Temporal Representation

**arXiv ID:** 2602.20685 | [PDF](https://arxiv.org/pdf/2602.20685v1)

**作者:** Yichen Xie `[一作]` (Applied Intuition), Wei Zhan `[通讯]` (Applied Intuition)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种几何无关、可处理多视角长时序视频的4D世界基础模型，能够在任意摄像机配置和运动下生成高质量、时间一致的多视角视频。

**💡 创新点**

核心创新包括：1）双因果自回归框架，联合尺度级和时间级因果关系；2）相对Plücker射线位置编码，实现统一的7维连续空间表示；3）循环训练策略，弥合训练与推理分布差异，提升长时序生成的一致性。

**🔧 技术方法**

采用Transformer自回归结构，结合next‑scale预测、全局因果注意力、局部图像自注意力与跨视角交叉注意力，利用RoPE扩展到7维相对射线编码，并在训练中加入随机误差模拟与KV特征缓存的循环训练。

**📊 数据集**

训练与评估使用nuScenes和nuPlan两个公开驾驶数据集，统一转换为ScenarioNet格式，并用GPT‑4o mini生成场景描述；数据覆盖多摄像机视角、分辨率与帧率的多样化。

**📈 对比分析**

通过FID、FVD、吞吐量、NDS、mIoU等指标，在nuScenes验证集上与MagicDrive、X‑Drive、DriveDreamer、BEVWorld、Panacea等基线对比，取得更低的FID/FVD、更高的生成速度，并在对象/地图条件、视角位移和运动可解释性等方面显著优于现有方法。

**⚠️ 局限性**

局限性：①对极端摄像机配置或高度信息缺失的地图投影仍存在误差；②模型在缺乏足够3D先验时细节表达有限；③长期视频仍可能出现累计漂移，需进一步改进分布对齐或使用更大规模数据。

---

## 92. A $2$-branching construction for the $χ\leq 2r$ bound

**arXiv ID:** 2602.20949 | [PDF](https://arxiv.org/pdf/2602.20949v1)

**作者:** Vinicius Tikara Venturi Date `[一作]` (Federal University of Paraná), Leandro Miranda Zatesko `[通讯]` (Federal University of Technology of Paraná)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文研究了字符串冗余度度量χ（最小后缀集大小）与r（Burrows–Wheeler变换中的连串数）的关系，提出2‑branching性质并给出了满足该性质的循环字符串构造，随后推导出闭式比例χ/r=(2σ^k−1+1)/(σ^k−1+4)，并在k=3时给出对任意σ≥2的显式构造，将χ/r与2的接近度提升到O(1/σ^2)，在σ=3、4时通过计算搜索得到k=5构造，使比值超过1.91。

**💡 创新点**

创新点在于引入2‑branching概念，证明其能实现χ与r的最优闭式比值，并提供了对任意字母表大小的全新构造，显著改进了χ/r对2的逼近度，特别是在固定σ>2时取得O(1/σ^2)的极限。

**🔧 技术方法**

技术主要包括组合字符串构造、循环Burrows–Wheeler变换分析、后缀集理论、LFSR序列与有限域多项式的关联，以及计算搜索验证2‑branching性质与cBWT结构。

**📊 数据集**

实验使用理论构造的符号序列以及计算机生成的示例字符串；并参考文献中基因组数据的实验结果，但本研究未直接使用公开的基因组数据集。

**📈 对比分析**

与先前的聚类构造相比，k=3构造的χ/r比值为(2σ^2+1)/(σ^2+4)，在σ=3、4时分别为1.462和1.650，低于已知1.5-1.6；k=5构造进一步提升到1.918和1.973，明显更接近理论上限2，表明构造效果优异。

**⚠️ 局限性**

局限性包括仅在σ=3、4给出了k=5构造，其他σ尚未找到；缺乏证明2‑branching能否产生更高比值；对实际基因组数据的适用性仍未验证；构造仅覆盖特定长度，尚无通用k>5的闭式构造。

---

## 93. Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining

**arXiv ID:** 2602.20500 | [PDF](https://arxiv.org/pdf/2602.20500v1)

**作者:** Keyu Zhou `[一作]` (Hangzhou Dianzi University), Shunlei Li `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5048311528)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于策略挖掘的自主腹腔镜摄像头控制框架，将离线事件抽象与图聚类、在线 Vision‑Language Model 预测和 IBVS‑RCM 闭环结合，实现可解释且稳定的摄像头控制。

**💡 创新点**

创新点包括：①将手术视频转化为时间事件图并挖掘可重用的摄像策略原语；②在实时控制中使用 VLM 生成策略标签与离散方向指令，支持语音交互；③通过策略监督的方向预测与传统 IBVS‑RCM 结合，显著提升安全性与可解释性。

**🔧 技术方法**

采用事件检测、属性图构建、WSBGC 图聚类、Graph Attention Autoencoder、Qwen2.5‑VL 7B VLM（配备策略头和方向头）+ LoRA 微调 + 4‑bit AWQ 量化、IBVS‑RCM 控制、语音识别等技术。

**📊 数据集**

使用 109 例腹腔镜胆囊切除视频（50 私有专家录制 + 59 Cholec80），以及硅胶模型、猪肠、猪胃等 ex‑vivo 数据；手工标注 162 段事件用于评估。

**📈 对比分析**

通过与人工操纵和基线模型对比，事件检测 F1 达 0.86，聚类 Purity 0.81；自主控制在视场中心误差降低 35.26%、图像抖动降低 62.33%，工作距离误差 7.12%，高频能量和姿态抖动极低，语音指令识别准确率接近 100%。

**⚠️ 局限性**

局限性：仅在 ex‑vivo 模型验证，缺少体内实验；12 个策略原语可能不足以覆盖更复杂手术；VLM 在极端视觉降质时仍可能产生误判；实时计算有一定延迟；需要在更广泛手术种类中验证泛化能力。

---

## 94. ParkDiffusion++: Ego Intention Conditioned Joint Multi-Agent Trajectory Prediction for Automated Parking using Diffusion Models

**arXiv ID:** 2602.20923 | [PDF](https://arxiv.org/pdf/2602.20923v1)

**作者:** Jiarong Wei `[一作]` (CARIAD SE), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2565 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种两阶段的自动泊车场景下的ego意图条件联合轨迹预测框架，先预测离散的ego终点意图，再以该意图为条件生成全场景的多车轨迹，并通过安全引导去噪器与对抗式知识蒸馏实现对未观测what‑if情景的学习。

**💡 创新点**

核心创新包括①使用离散意图标记器将复杂意图压缩为可枚举的终点令牌；②构建ego意图条件的联合预测器，结合曝光门控与多模束搜索实现场景级连贯预测；③引入安全引导去噪器与EMA教师+去噪器的对抗式知识蒸馏，提供无标签的counterfactual伪目标；④在自动泊车中首次实现联合ego意图预测与条件联合轨迹预测。

**🔧 技术方法**

采用基于Transformer的场景编码、FiLM调制、曝光门控、束搜索与场景选择器；使用Score-based去噪网络与几何潜能函数进行安全引导；利用EMA教师+去噪器实现对抗式知识蒸馏；训练分两阶段，Stage1为意图标记器，Stage2为条件联合预测器。

**📊 数据集**

在Dragon Lake Parking（DLP）与Intersections Drone（inD）两个公开数据集上进行评估，DLP为密集泊车场景，inD为城市交叉口。

**📈 对比分析**

与多种强基线（WIMP、SceneTransformer、ScePT、MotionLM、DTPP）以及基于ParkDiffusion的随机组合对比，本文方法在DLP和inD上均取得最优或近优的oracle/final ADE/FDE、MR、OR、mAP等指标，尤其在安全性（OR）和命中率（MR）上显著领先。

**⚠️ 局限性**

局限性在于安全引导去噪器使用手工设计的几何势函数，导致预测偏保守；对counterfactual监督的离线学习与闭环控制之间仍存在缺口；且当意图令牌数目过多或对抗权重失衡时会出现性能退化。

---

## 95. An Approach to Combining Video and Speech with Large Language Models in Human-Robot Interaction

**arXiv ID:** 2602.20219 | [PDF](https://arxiv.org/pdf/2602.20219v1)

**作者:** Guanting Shen `[一作]` (Dalian University of Technology), Zi Tian `[通讯]` (Dalian University of Technology)

**通讯引用:** 9383 | [OpenAlex ID](https://openalex.org/A5100428417)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了一套多模态人机交互框架，将视觉、语音、语言模型与模糊逻辑控制相结合，实现了基于口头指令的Dobot Magician机械臂抓取操作。

**💡 创新点**

创新点在于将基础模型（Florence‑2、LLaMA 3.1、Whisper）与开源视觉传感（RealSense、ArUco标记）及Interval Type‑2模糊控制无缝耦合，并通过结构化的JSON指令接口实现高可靠的自然语言到机器人动作映射。

**🔧 技术方法**

使用技术包括：Florence‑2（开放词汇目标检测）、LLaMA 3.1（语言理解与动作生成）、Whisper（语音转文本）、AST（唤醒词检测）、Intel RealSense D435i（RGB‑D感知）、OpenCV（ArUco定位）、Interval Type‑2模糊逻辑控制器、NVIDIA RTX 4070+Intel Core i9进行模型推理。

**📊 数据集**

实验数据集为自制的水果照片集合（苹果、柠檬等）放置在平面桌面，并使用Intel RealSense捕获RGB‑D图像；语音指令由人类录制，唤醒词采用AST模型识别。

**📈 对比分析**

与单一模态或传统规则控制的对比，系统在60次试验中取得了75%的端到端成功率，平均任务耗时35.4 秒；虽然在语音识别与语言理解阶段延迟低于10%，但目标检测与机器人动作执行是主要时间瓶颈。

**⚠️ 局限性**

局限性包括：整体任务耗时偏长（>30 s）、对目标检测与动作执行的误差较大（误差率高达43%）、语言到动作的解析仍易产生歧义、以及缺乏动态环境与复杂物体的鲁棒性验证。

---

## 96. EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations

**arXiv ID:** 2602.20958 | [PDF](https://arxiv.org/pdf/2602.20958v1)

**作者:** Luka Šiktar `[一作]` (University of Zagreb), Marko Švaco `[通讯]` (University of Zagreb)

**通讯引用:** 470 | [OpenAlex ID](https://openalex.org/A5021032181)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于深度相机与单目摄像机关键点估计的多模态融合方法，用于无人机在搜索救援任务中跟踪并保持与目标人的安全距离。

**💡 创新点**

创新点包括：①利用YOLO-pose提取肩-臀关键点进行摄像机到人体的距离近似；②将该近似值与RealSense深度相机直接测距融合；③采用扩展卡尔曼滤波（EKF）实时抑制噪声与异常值；④显著延展有效测距范围至7 m，并在室内外真实环境下实现15 FPS的实时跟踪。

**🔧 技术方法**

所用技术主要包括YOLOv11/YOLOv11-pose用于检测与姿态估计、Dlib进行人脸识别、Intel RealSense D435i深度相机、Jetson Xavier NX嵌入式计算平台、ROS2/Ubuntu、PyTorch深度学习框架以及EKF滤波算法。

**📊 数据集**

实验数据集为室内使用OptiTrack运动捕捉系统标定的人工标记数据，以及在户外Hexsoon EDU450无人机平台上收集的现场跟踪录像。

**📈 对比分析**

与单独使用关键点估计或仅用深度相机相比，融合方法在三种运动场景下将平均误差从10.45 cm/13.01 cm降至0.83 cm（RMSE从20.24/34.35 cm降至17.16 cm），显示出明显的性能提升；在连续和侧向移动情形下亦获得显著误差下降。

**⚠️ 局限性**

局限性包括：①单目关键点近似受人体尺寸假设限制，近距离误差增大；②深度相机在超出4 m后误差指数增长；③两相机需严格时间与空间对齐；④目前仅验证了Intel RealSense与YOLOv11，未评估更高分辨率或激光深度传感器的兼容性。

---

## 97. From Perception to Action: An Interactive Benchmark for Vision Reasoning

**arXiv ID:** 2602.21015 | [PDF](https://arxiv.org/pdf/2602.21015v1)

**作者:** Yuhao Wu `[一作]` (Singapore University of Technology and Design), Roy Ka-Wei Lee `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1647 | [OpenAlex ID](https://openalex.org/A5089793938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个交互式3D物理推理基准，评估视觉-语言模型在多步物理操作中的表现

**💡 创新点**

创新在于将评估从单轮VQA转向基于物理约束的多步交互式推理，并引入细粒度的效率与成本指标

**🔧 技术方法**

使用Unity和Python3D引擎模拟物理环境，结合VLM与扩散式图像‑视频模型的交互式推理框架

**📊 数据集**

构建了包含32个拼图与77个堆叠任务的CHAINS基准集，按易/中/难三层级划分

**📈 对比分析**

对比多种闭源与开源VLM（如GPT‑5.2、Claude‑Sonnet‑4.5、Gemini‑Pro）及视频生成模型，最强模型Pass@1仅22.9%，在拼图任务上表现极差，堆叠任务稍好但仍远低于理想水平

**⚠️ 局限性**

局限在于模型难以内部化物理结构与约束，易产生误操作；基准仍需扩展至更复杂、多维物理场景

---

## 98. PackMonitor: Enabling Zero Package Hallucinations Through Decoding-Time Monitoring

**arXiv ID:** 2602.20717 | [PDF](https://arxiv.org/pdf/2602.20717v1)

**作者:** Xiting Liu `[一作]` (Tsinghua University), Shi-Min Hu `[通讯]` (Tsinghua University)

**通讯引用:** 22606 | [OpenAlex ID](https://openalex.org/A5037233582)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PackMonitor 框架，利用解码时监控和干预机制消除 LLM 生成的包名称幻觉；

**💡 创新点**

通过理论可判定的权威包列表构造 DFA，实现解码时完全排除非法包名称，从而零幻觉、无训练、插件化；

**🔧 技术方法**

采用上下文感知解析器、包名干预器（基于 DFA 与 Token Trie）、logits 层面掩码以及 DFA 缓存技术；

**📊 数据集**

使用 PyPI 官方包列表（约 706k 包）做 DFA 构建；评测数据集包括 HFuzzer、Package4U 以及 HumanEval 代码生成基准；

**📈 对比分析**

与 RAG、SR、SFT、RAG+SR 等基线对比：在 5 种主流 LLM 上零包幻觉，平均推理延迟仅提升 7%–28%，在 HumanEval 上 Pass@1 无下降；DFA 缓存将构建开销降至几百毫秒；

**⚠️ 局限性**

局限性：需依赖完整、准确的权威包列表；仅覆盖安装命令相关幻觉；对生态系统演进需周期性更新；不同部署环境可能影响实际延迟；

---

## 99. Modelling Interaction Duration in Relational Event Models

**arXiv ID:** 2602.21000 | [PDF](https://arxiv.org/pdf/2602.21000v1)

**作者:** Rumana Lakdawala `[一作]` (Tilburg University), Joris Mulder `[通讯]` (Tilburg University)

**通讯引用:** 3109 | [OpenAlex ID](https://openalex.org/A5025801659)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并实现了一种新的时序事件模型——DuREM（Duration Relational Event Model），用于同时建模事件的开始与结束，并将事件持续时间融入事件发生率与持续率的统计量中。

**💡 创新点**

创新点在于：① 引入了事件持续时间权重（ψ^s、ψ^e）和记忆衰减（τ），使得过去事件的时长能够影响未来事件的起止概率；② 将起始与结束事件视为两类竞争风险，并在风险集里同时包含正在进行的事件和空闲的 dyad；③ 通过 R 包 `durem` 实现了模型的自动计算与最大似然估计（结合网格搜索求解 ψ 与 τ）。

**🔧 技术方法**

技术手段：基于传统 Relational Event Model 的 log‑linear 强度模型，扩展为两类强度 λ^s、λ^e；使用最大似然估计结合网格搜索来学习 ψ^s、ψ^e、τ；利用指数分布对事件间隔建模，并构造竞争风险框架；实现了 R 包 `durem` 用于统计量计算、风险集更新和并行网格搜索。

**📊 数据集**

数据集：
• 研究团队动态：来自 GEDII 项目的 5 天社交计量徽章数据，包含 9 名团队成员、11,607 次面对面接触（带持续时间）。
• 亲密暴力案例：对阿姆斯特丹公共空间一段 9 分钟的监控视频进行编码，得到 10 名参与者、220 次交互（含 0–26.7 秒持续时间）。

**📈 对比分析**

与传统 REM（不考虑持续时间）的比较：DuREM 在两组案例中都提供了更丰富的参数解释（如 ψ^s、ψ^e 对起始/结束事件的正负效应），并通过估计出的半衰期 τ 量化记忆衰减；虽然论文未给出数值型性能指标（如似然值或预测误差），但结果表明在两组数据中 DuREM 能够捕捉到显著的持续时间效应，且对多重事件的重叠建模比传统 REM 更为灵活。

**⚠️ 局限性**

局限性：
1. 对于正在进行的事件，模型假设参与者仍可参与新事件，可能不符合某些场景（如会议排程）。
2. 参数 ψ^s、ψ^e、τ 的估计依赖网格搜索，计算量大，尤其在大网络或高维 covariate 时成本高。
3. 采用指数分布假设事件结束的危害率为常数，忽略了可能随时间变化的结束概率；可考虑 Weibull 等更灵活分布。
4. 记忆衰减采用固定半衰期 τ，未考虑更灵活或非参数化的衰减形式。
5. 仅对 dyad 级别建模，未直接扩展到更高层结构（如群组或层级）。

---

## 100. UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling

**arXiv ID:** 2602.20943 | [PDF](https://arxiv.org/pdf/2602.20943v1)

**作者:** Kaiyuan Tan `[一作]` (Xiaomi EV), Haiyang Sun `[通讯]` (Xiaomi EV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种递归式4D驾驶场景重建框架，利用场景token在每帧中进行迭代细化与增量更新，实现长序列的实时高质量重建。

**💡 创新点**

创新点在于把优化式与前馈式方法结合：引入可见性过滤机制实现近线性时间复杂度、使用对象姿态指导的寿命感知高斯模型捕捉长时间动态，并通过Transformer实现全流程无梯度渲染循环。

**🔧 技术方法**

核心技术包括：基于Transformer的场景token编码与更新、3D Gaussian splatting渲染、Plücker射线嵌入、视觉-对象交叉注意力、生命周期预测、LPIPS+深度监督等。

**📊 数据集**

在Waymo Open Dataset上进行实验，使用前端摄像头数据和对应的相机位姿进行训练与评估。

**📈 对比分析**

与多种单场景优化方法（3DGS、PVG、DeformableGS、Street Gaussians）以及前馈方法（GS‑LRM、STORM）比较，PSNR、SSIM和Depth RMSE均显著提升，尤其在16 s序列上可在0.5 s内完成重建，远快于对比方法。

**⚠️ 局限性**

局限性包括：对极端遮挡或低光环境的鲁棒性尚待提升、模型在跨域场景下的泛化能力有限、以及对极端高速动态物体的精细捕捉仍有改进空间。

---

## 101. Body-Reservoir Governance in Repeated Games: Embodied Decision-Making, Dynamic Sentinel Adaptation, and Complexity-Regularized Optimization

**arXiv ID:** 2602.20846 | [PDF](https://arxiv.org/pdf/2602.20846v1)

**作者:** Yuki Nakamura `[一作]` `[通讯]` (Open University of Japan), Yuki Nakamura (Open University of Japan)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出了身体‑储备治理（Body‑Reservoir Governance, BRG）框架，使用三层结构（身体储备、认知滤波器、元认知治理）来解释在重复博弈中合作的可持续性，并通过动态哨兵机制使身体自身的内部状态决定何时需要认知干预。通过自洽方程、KL 复杂度成本、自由能权衡等理论分析，并在大量仿真实验（自洽收敛、KL 景观、扰动响应、习惯化、自由能曲面、动态哨兵、参数灵敏度、储备维度扫描、相位图、EMA 滤波基准）中验证了模型的有效性。

**💡 创新点**

创新点包括：① 将身体的高维动力学视为隐式推理与决策主体，消除了传统条件策略的计算开销；② 用身体状态分布的 KL 散度量化策略的能量成本；③ 设计基于身体自身“不适感”信号的动态哨兵，使元认知仅负责轻量级策略而不承担检测；④ 推导自洽固定点与闭环稳定性；⑤ 将自由能最小化引入博弈中的身体‑认知权衡；⑥ 通过储备维度演化阐释身体复杂度对合作质量的决定作用；⑦ 证明动态哨兵在多种情境下优于经典 Tit‑for‑Tat 与 EMA 过滤器。

**🔧 技术方法**

技术手段包括：echo state network（ESN）储备网络与 Oja 规则自适应；岭回归训练读出权重；k‑近邻估计 KL 散度；自由能函数优化；动态哨兵阈值与积分控制；数值仿真（连续动作 PD 游戏）、相位图与维度扫描分析；以及与 EMA‑TfT 的对比实验。

**📊 数据集**

数据集：无外部真实数据，全部采用人工生成的博弈对手序列，包括完全合作、带噪合作、持续背叛区块等；实验环境为连续动作 Prisoner’s Dilemma，参数设置统一。

**📈 对比分析**

比较方法：将 BRG（不同 α 静态设置、动态哨兵）与传统 Tit‑for‑Tat、EMA‑滤波器在同一对手序列下的均值收益、方差降低和 KL 成本进行对比。结果显示：动态哨兵累计收益最高，行动方差在 α=1 时比 α=0 低约 1600 倍，KL 散度在 α≈0.7 时最小，且自由能最小化给出 α≈0.6‑0.7 的内在最优点。

**⚠️ 局限性**

局限性：① 依赖 ESN 的 echo state 属性与 Oja 自适应，参数选择对结果敏感；② 许多理论结果仍为经验性或推测（如相位边界、KL 非单调性）未给出完整证明；③ 仅在 PD 及其连续扩展上验证，未在更复杂或多玩家博弈中检验；④ 计算成本仍不低，尤其是高维储备网络；⑤ 自由能参数 λ 设定缺乏系统化的物理解释；⑥ 动态哨兵的阈值和速率在不同博弈场景下可能需要重新校准。

---

## 102. See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis

**arXiv ID:** 2602.20951 | [PDF](https://arxiv.org/pdf/2602.20951v1)

**作者:** Jaehyun Park `[一作]` (Korea Advanced Institute of Science and Technology), Dongmin Park `[通讯]` (KRAFTON)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了自动化的 Agentic 框架 ArtiAgent，用来在真实图像中合成多样化的结构性视觉缺陷并生成带有定位、解释等注释的高质量数据集

**💡 创新点**

通过在 Diffusion Transformer 的自注意力层中同时操纵位置编码与值编码实现无人工干预的缺陷注入，并将感知、合成、筛选三位 Agent 协同工作，形成可扩展的人工缺陷数据生成流程

**🔧 技术方法**

使用 Grounded‑SAM+VLM 识别实体/子实体，Patch‑mapping 工具（add、remove、distort、fuse），inversion‑injection 结合 PE+V 操作，LPIPS+VLM 过滤与解释，VLM 生成局部/全局解释，并在 ArtiAgent 上训练 VQA 数据集 fine‑tune Qwen2.5‑VL 等

**📊 数据集**

真实图像来源于 COCO、Caltech‑101、11K Hands、CelebA HQ，合成 100K 对缺陷图像；人工标注 1K 现代 Diffusion 生成图像（ArtiBench），用于基准评估 RichHF‑18K、LOKI、SynthScars、ArtiBench 等

**📈 对比分析**

与 PAL、DiffDoctor、LEGION、GPT‑5、Gemini‑2.5‑Pro 等方法对比，在检测、定位、解释三任务上采用准确率、mIoU、ROUGE 等指标；ArtiAgent 训练的 VLM 在所有基准上显著优于原版，甚至超过 GPT‑5；数据规模越大性能越好；在奖励引导和 inpainting 修复实验中明显降低缺陷率

**⚠️ 局限性**

目前仅覆盖结构性缺陷，忽略文本对齐错误；缺陷注入依赖 DiT 结构，可能不易推广至所有 diffusion 模型；合成工具可能缺乏极端或罕见缺陷；评价仍依赖人工标注的 ArtiBench，规模有限

---

## 103. Uncertainty-Aware Delivery Delay Duration Prediction via Multi-Task Deep Learning

**arXiv ID:** 2602.20271 | [PDF](https://arxiv.org/pdf/2602.20271v1)

**作者:** Stefan Faulkner `[一作]` (Georgia Institute of Technology), Pascal Van Hentenryck `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 17362 | [OpenAlex ID](https://openalex.org/A5035808622)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个多任务深度学习框架，用于在极度类别不平衡的物流数据上同时完成延迟判别与延迟时长回归，并通过分类引导的路由机制实现对迟到与准时货件的专门建模。

**💡 创新点**

创新点在于：①端到端的多任务架构将分类和回归共享特征，避免传统两步训练的误差传播；②对高维离散与连续特征采用专用嵌入层（embedding）提升表达能力；③使用 Sigmoid‑F1 作为分类损失，pinball loss 作为分位数回归损失；④引入 conformalized quantile regression（CQR）为迟到预测提供可靠的置信区间。

**🔧 技术方法**

技术方法包括：深度神经网络（MLP 主干 + 两个分位数回归头），分类头采用 Sigmoid‑F1；对数值特征使用 Periodic Linear 变换，对类别特征使用分级嵌入；优化器为 AdamW，学习率调度采用线性 warm‑up + decay；超参调优通过 Optuna；对比基线使用 XGBoost、CatBoost 的单步和两步树模型。

**📊 数据集**

实验数据来自工业合作伙伴的 1000+ 万条装运记录，涵盖 2022‑2024 年 4 个主要发货地（L1–L4），包含 190k 目的地、重量、体积、距离、地理坐标等 30+ 维特征。

**📈 对比分析**

评价方法：对比单步树模型（XGB‑S1、CatB‑S1）、两步树模型（XGB‑S2、CatB‑S2）与本方法（DL）。在迟到样本上，DL 的 MAE 仅为 0.67–0.91 天，比分步树模型低 41–64%，两步树模型低 15–35%；置信区间覆盖率在迟到样本上达到 64–70%（校准前），而基线仅 20–48%；Winkler 分数在迟到样本上亦显著优于基线。

**⚠️ 局限性**

局限性：① conformal 校准所需的样本可交换性假设在时间序列物流数据中不成立，理论覆盖率不具备严格保证；② 低样本量发货地的迁移学习效果尚未验证；③ 当前模型为批量训练，缺乏在线学习能力，难以即时适应季节性或突发事件导致的分布漂移。

---

## 104. Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks

**arXiv ID:** 2602.21013 | [PDF](https://arxiv.org/pdf/2602.21013v1)

**作者:** Sanjay Haresh `[一作]` (Qualcomm AI Research), Roland Memisevic `[通讯]` (Qualcomm AI Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种语言 scratchpad 机制，让现有的 Vision‑Language‑Action（VLA）模型能够记录并利用自身的“思考”与任务进度，实现对空间与时间记忆的支持，从而解决传统 stateless VLA 无法完成的长期记忆型操纵任务。

**💡 创新点**

创新点在于将模型自身生成的自然语言描述作为可写可读的外部记忆，既捕捉空间信息（如物体位置）又捕捉时间信息（如已完成子任务），并通过特殊 <done> 触发更新；同时证明即使是无记忆的 transformer‑VLA，在加上 scratchpad 后也能匹配或超越带有隐式记忆的 recurrent VLA。

**🔧 技术方法**

技术实现包括：①使用预训练的 VLM（如 PaliGemma‑2、Mamba）作为基础；②在训练和推理时将 scratchpad 文本拼接或插入到输入序列；③定义 <think>、<act>、<done> 等特殊 token，决定何时生成动作、更新 scratchpad；④对 recurrent VLA 采用交错的文本序列训练；⑤利用 LoRA 微调提升在真实机器人上的表现。

**📊 数据集**

实验数据集包括：ClevrSkills‑Mem（5 个需要空间/时间记忆的任务）、MemoryBench（基于 RLBench 的 Put‑Block‑Back 等任务）以及真实世界的 Pick‑Place‑Restore（使用 UFactory xArm 6 与 RealSense 摄像头采集的 200 条轨迹）。

**📈 对比分析**

对比方法：将无 scratchpad 的 T‑VLA 与 T‑VLA+Scratchpad、R‑VLA 与 R‑VLA+Scratchpad 进行同一任务下的滚动实验；在 ClevrSkills‑Mem 上平均提升约 48%（单任务提升 68%–72%），在 MemoryBench 上通过“sim‑eval”策略达到 100% 成功率，真实世界实验中 scratchpad 版模型在子任务完成率上提升至约 65%。

**⚠️ 局限性**

局限性包括：①对极细粒度时间记忆（如 Rotate‑Restore）仍难以满足；②需要大量标注的“思考”文本与子任务划分，人工成本高；③在真实机器人中受限于训练样本量，仍无法实现与人类水平相当的精细操纵；④scratchpad 机制依赖语言表达，若语言生成不准确会导致记忆错误。

---

## 105. Onboard-Targeted Segmentation of Straylight in Space Camera Sensors

**arXiv ID:** 2602.20709 | [PDF](https://arxiv.org/pdf/2602.20709v1)

**作者:** Riccardo Gallon `[一作]` (Delft University of Technology), Eberhard Gill `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文开发了一种面向太空相机的AI语义分割模型，用于实时检测光晕（太阳光斑）故障并可直接部署在航天器受限硬件上；同时提出了面向系统级的碎片级评估指标并给出集成框架；

**💡 创新点**

创新点包括：①使用大规模光晕数据集（Flare7k++）预训练以缓解太空图像稀缺；②构建轻量化DeepLabV3‑MobileNetV3网络，兼顾精度与硬件友好；③引入碎片级指标（PaR、PaP、PamIoU）以捕获完整故障块而非单像素；④提出与导航管线对接的接口，直接影响卡尔曼滤波的可用性；

**🔧 技术方法**

技术手段：DeepLabV3+Atrous Spatial Pyramid Pooling、MobileNetV3 backbone、深度可分离卷积、Binary Cross‑Entropy 损失、Adam 优化器、Gaussian smoothing、regionprops、迁移学习、定制系统级评估；

**📊 数据集**

数据集：1）自研光晕故障数据集（1000张 1024×1024 3通道，包含光晕与正常像素）；2）公开 Flare7k++ 光晕数据集（7962张 512×512 3通道）用于预训练；

**📈 对比分析**

评估方式：先在自研数据上预训练，再微调；与预训练模型对比，微调后标准指标提升至精度0.908、召回率0.958、mIoU0.873；碎片级指标从PaR0.308提升至0.991、PaP0.849→0.832、PamIoU0.855→0.783，证明模型在完整光晕块检测上的显著提升；

**⚠️ 局限性**

局限性：①仅使用合成图像，存在域漂移；②数据集规模有限，需进一步验证真实任务；③模型需在航天器FPGA上硬件‑in‑the‑loop测试；④只针对光晕故障，其他故障类型需扩展；⑤Gaussian smoothing 影响像素级评估，需进一步完善；

---

## 106. AI Combines, Humans Socialise: A SECI-based Experience Report on Business Simulation Games

**arXiv ID:** 2602.20633 | [PDF](https://arxiv.org/pdf/2602.20633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 107. Matching Multiple Experts: On the Exploitability of Multi-Agent Imitation Learning

**arXiv ID:** 2602.21020 | [PDF](https://arxiv.org/pdf/2602.21020v1)

**作者:** Antoine Bergerault `[一作]` (University of Zurich), Negar Mehr `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究在多智能体离线模仿学习中，从专家的纳什均衡演示中学习近似纳什均衡策略，并给出了关于可利用性（exploitability）与模仿误差之间的理论界限。

**💡 创新点**

创新点在于：① 首次证明在一般 n 人马尔可夫游戏中，除非满足完整状态覆盖和状态-动作占据度匹配，否则即使精确匹配专家占据度也无法保证学习到纳什均衡；② 引入“最佳响应连续性”（best‑response delta‑continuity）这一新概念，将游戏对策略扰动的敏感性刻画为一个可度量的函数，从而在弱占优策略均衡或更一般的连续性假设下得到一致且可计算的纳什间隙上界；③ 证明在存在弱占优策略均衡时，行为克隆误差为 ϵ_BC 可得到 NashGap ≤ 2nϵ_BC/(1−γ)² 的显式上界。

**🔧 技术方法**

主要技术包括：马尔可夫游戏理论、占据度匹配、性能差分引理、PPAD 难度分析、δ‑连续性与最佳响应映射的连贯性分析，以及基于信息熵正则化等技术对 δ 的控制。

**📊 数据集**

论文以理论分析为主，没有使用公开数据集；若有实验验证，则采用合成的马尔可夫游戏示例（如合作两智能体示例）进行演示。

**📈 对比分析**

与现有方法（如行为克隆、对抗式模仿学习）相比，本文没有提供数值性能对比，而是通过理论证明给出了在不同假设下可获得的上界和不可行性结论，表明在一般情况下模仿学习难以保证纳什可利用性。

**⚠️ 局限性**

局限性：① 结论在一般马尔可夫游戏中很难得到；② 需要完整状态覆盖和状态-动作占据度匹配，实际数据往往无法满足；③ 对最佳响应连续性 δ 的估计依赖于游戏特定信息，计算成本可能不低；④ 仅在弱占优策略均衡或 δ 连续性足够好的游戏中才能得到可行上界。

---

## 108. TOM: A Ternary Read-only Memory Accelerator for LLM-powered Edge Intelligence

**arXiv ID:** 2602.20662 | [PDF](https://arxiv.org/pdf/2602.20662v1)

**作者:** Hongyi Guan `[一作]` (Microsoft Research), Ningyi Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3030 | [OpenAlex ID](https://openalex.org/A5100833305)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了 TOM（Ternary Read‑only Memory Accelerator），一种面向边缘设备的三值量化 LLM 加速器，采用稀疏感知 ROM 与 SRAM 混合架构，分布式处理单元以及工作负载感知动态电源门控，显著提升内存密度与带宽，同时保持可调优性。

**💡 创新点**

创新点包括：①利用标准单元逻辑合成三值权重，消除零位占用实现极高的 ROM 存储密度；②将高密度 ROM 与计算单元协同布局的分布式处理架构，最大化带宽并最小化数据搬运；③在逻辑基 ROM 上实现工作负载感知的动态电源门控，几乎零延迟地关闭非激活层，显著降低静态功耗。

**🔧 技术方法**

核心技术为：三值量化（1.58‑bit －1/0/+1 权重）、QLoRA 低秩适配器、FP8 计算单元、稀疏感知 ROM（基于标准单元逻辑合成）、分布式处理通道与全局归约树、工作负载感知动态电源门控；实现采用 7 nm 逻辑技术、Synopsys Fusion Compiler 与 PowerArtist。

**📊 数据集**

以 BitNet‑2B（1.58‑bit 三值 LLM）为主要评测模型，使用 64/128/256/512 token 长度的推理任务进行基准；亦参考 LLaMA3‑8B 等公开模型在不同量化方案下的准确性对比。

**📈 对比分析**

与通用 CPU（Intel i5‑12500H）、高端 GPU（NVIDIA A100）以及多款 ASIC/PIM（Olive、FIGNA、Spatten、Arc、SOFA 等）对比：TOM 最高吞吐量约 3,306 TPS，达到 A100 的 63.7× 速度提升，峰值带宽 200 TB/s，ROM 密度 15.0 MB/mm²，功耗仅 5.33 W（相较于 25.8 W 下降近 80%），在功耗效率上超过 A100 4000×、CPU 60×，在 ASIC 对比中 TOPS/W 97.8× 超越 Olive。

**⚠️ 局限性**

局限性包括：①仍需 56.9 mm² 的芯片面积，主要集中在 ROM；②仅针对三值量化模型，非三值模型的迁移需额外改造；③稀疏感知 ROM 的设计与综合复杂度较高，易受工艺变异影响；④动态电源门控依赖顺序推理，批量推理与并行任务受限；⑤对更长上下文或更大模型仍需扩展 SRAM，导致面积与功耗上升。

---

## 109. Vanishing Watermarks: Diffusion-Based Image Editing Undermines Robust Invisible Watermarking

**arXiv ID:** 2602.20680 | [PDF](https://arxiv.org/pdf/2602.20680v1)

**作者:** Fan Guo `[一作]` (Xidian University), Finn Carter `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了扩散模型（Diffusion Model）在图像编辑过程中对现有鲁棒隐形水印的破坏效果，并提供了理论和实验上的证明。

**💡 创新点**

创新点在于：①首次系统性证明扩散过程会使水印信息的互信息趋近零；②提出了基于水印解码器反馈的引导式扩散攻击；③结合理论分析与大规模实验展示了扩散编辑对主流深度学习水印方案的致命威胁。

**🔧 技术方法**

使用的技术包括：Stable Diffusion v1.5 进行图像重生成与引导式编辑；StegaStamp、TrustMark、VINE 三种深度学习水印编码/解码器；信息论分析（互信息、数据处理不等式）以及梯度反向传播实现的引导攻击。

**📊 数据集**

实验数据集采用了 500 张来自 COCO 的 512×512 像素图像，确保与水印算法训练集无重叠。

**📈 对比分析**

与传统扰动攻击（JPEG、噪声、裁剪等）对比，扩散重生成攻击将水印解码准确率从约 95‑100% 降低至 0‑7%，而图像视觉质量保持 PSNR≈31dB、SSIM≈0.95，说明攻击几乎不影响人眼感知；引导式攻击进一步将准确率压到 0% 并且在 1.6% 左右。

**⚠️ 局限性**

局限性包括：仅针对图像水印评估，未测试视频或其它媒体；只使用了 Stable Diffusion v1.5，未覆盖更先进或不同的生成模型；引导式攻击需要解码器信息，对未知水印缺乏通用性；理论证明基于理想化假设，实际模型可能存在偏差；未提供有效的水印恢复或检测手段。

---

## 110. GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training

**arXiv ID:** 2602.20399 | [PDF](https://arxiv.org/pdf/2602.20399v1)

**作者:** Haixu Wu `[一作]` (Tsinghua University), Wojciech Matusik `[通讯]` (MIT CSAIL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种基于“lifted geometry pre-training”的神经模拟器预训练方法GeoPT，使模型在仅使用无标签几何数据的前提下提升物理仿真精度与数据效率。

**💡 创新点**

通过在几何预训练中注入合成动力学轨迹，构建了几何-动力学耦合的自监督任务，弥合了几何预训练与物理仿真之间的鸿沟。

**🔧 技术方法**

采用Transolver Transformer架构，利用随机速度场生成的几何轨迹作为监督信号，实现了“动态提升”的自监督预训练。

**📊 数据集**

在ShapeNet中采集约1万多种工业相关几何（汽车、飞机、水上船），对每个几何生成上百万条随机动力学轨迹作为预训练数据。

**📈 对比分析**

与传统几何预训练、VAE辅助特征等基线相比，在汽车、飞机、船舶和碰撞等工业仿真基准上，GeoPT将标注数据需求降低20–60%，并将收敛速度提升约2倍。

**⚠️ 局限性**

依赖于手工设定的速度场与物理场映射，对极端或未见过的动力学条件可能表现不佳；并且在更复杂多相或时间依赖的物理问题中仍需进一步验证。

---

## 111. Federated Learning for Cross-Modality Medical Image Segmentation via Augmentation-Driven Generalization

**arXiv ID:** 2602.20773 | [PDF](https://arxiv.org/pdf/2602.20773v1)

**作者:** Sachin Dudda Nagaraju `[一作]` (Norwegian University of Science and Technology), Mattijs Elschot `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 2425 | [OpenAlex ID](https://openalex.org/A5081483883)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种联邦学习框架 FedGIN，用于跨模态医学图像分割，尤其是 CT 与 MRI 的腹部器官与心脏分割。

**💡 创新点**

创新点在于将全局强度非线性（GIN）随机卷积增强集成到联邦学习中，在单模态客户端训练时模拟跨模态分布，保持解剖结构不变，克服无配对数据和单模态限制。

**🔧 技术方法**

技术包括联邦平均（FedAvg）、随机卷积强度变换（GIN）、域特定归一化（DSBN）、频域增强（FMAug、RaffeSDG）以及多种基线方法。

**📊 数据集**

使用了公开数据集 TotalSegmentator、AMOS、CARE-Whole Heart 2025 等，涵盖 CT 与 MRI 的腹部器官（肝、肾、脾、胰腺、胆囊）与心脏七个子结构。

**📈 对比分析**

与集中式训练、FedAvg、DSBN、频域增强、ProRandConv 等方法对比，FedGIN 在 CT+MRI 联合学习时在大多数器官上实现 93–98% 的中心化性能，并显著提升低数据环境下的 Dice 分数（如胰腺从 0.073 提升至 0.437）。

**⚠️ 局限性**

局限包括数据量仍低（20–100 体积）、仅使用 2–3 个客户端、仅采用 2D 切片、未对增强质量进行过滤、未在更大规模多中心和更现代网络上验证。

---

## 112. ICON: Indirect Prompt Injection Defense for Agents based on Inference-Time Correction

**arXiv ID:** 2602.20708 | [PDF](https://arxiv.org/pdf/2602.20708v1)

**作者:** Che Wang `[一作]` (Peking University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**通讯引用:** 6830 | [OpenAlex ID](https://openalex.org/A5027969322)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个名为ICON的框架，用以在LLM代理中防御间接提示注入攻击，保证任务继续执行。

**💡 创新点**

创新点在于通过“关注强度得分”识别关键注意力头，利用潜在空间追踪探测器捕捉“过度聚焦”异常，并用“纠正修正器”在推理时对注意力进行精细调节，实现攻击检测与修复的闭环。

**🔧 技术方法**

主要技术包括对Transformer内部注意力熵的量化（FIS）、多阶段特征压缩与CNN+MLP组合的潜在空间探测器，以及基于阈值与对比加权的注意力干预机制。

**📊 数据集**

实验使用了InjectAgent、AgentDojo、TrojanTools等文本攻击数据集，并在Qwen、LLaMA、Mistral及多模态模型（Qwen-VL、InternVL、MiniCPM）上进行验证。

**📈 对比分析**

与模板过滤、工具过滤及商业防御（Qwen3Guard、Gemini）相比，ICON在攻击成功率上仅为0.4%，而在任务效用上提升了超过50%，且在OOD设置下检测率高达97%，训练成本低于2分钟。

**⚠️ 局限性**

局限性包括需要对内部注意力可视化的白盒访问、对极端自适应攻击的鲁棒性尚待进一步验证，以及在极少数模型或特定任务中可能出现误报。

---

## 113. Cooperative-Competitive Team Play of Real-World Craft Robots

**arXiv ID:** 2602.21119 | [PDF](https://arxiv.org/pdf/2602.21119v1)

**作者:** Rui Zhao `[一作]` (Tencent Robotics X Laboratory), Lei Han `[通讯]` (Tencent Robotics X Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在一个真实的机器人拼装竞技场上，设计了协作与竞争两类任务，并通过分布式多代理深度强化学习训练出可在物理机器人上部署的高层策略。

**💡 创新点**

创新点包括：① 采用基于规则的动作掩码实现 Guided RL，帮助机器人遵循现实约束并加速学习；② 提出了 Out of Distribution State Initialization（OODSI）方法，将在更真实环境（Gazebo或真实机器人）中遇到的离散分布外状态加入训练起始分布，从而显著缓解多代理场景的 sim‑to‑real 差距。

**🔧 技术方法**

主要技术：多代理 PPO（CTDE）强化学习、分布式训练框架（TLeague/Kubernetes）、动作掩码与动态随机化（DR）、OODSI、pyBullet 与 Gazebo 两种仿真环境、真实机器人硬件与摄像头+AprilTag 定位。

**📊 数据集**

未使用公开数据集，全部使用自行构造的 pyBullet 与 Gazebo 仿真数据以及在真实机器人上收集的轨迹进行训练与评估。

**📈 对比分析**

通过在 pyBullet 训练后在 Gazebo 进行 Sim2Real 评估，比较 PPO、PPO+DR、PPO+OODSI、PPO+DR+OODSI 四种方法。结果显示：DR 在 Gazebo 的成功率略有提升；OODSI 单独使用时提升约 20‑30%；两者结合时在协作与竞争任务中的成功率分别提高 23.3% 与 30% 左右，显著优于基线。

**⚠️ 局限性**

局限性：① 仍受仿真与真实机器人之间同步/异步执行差异的影响，OODSI 采样方法可能无法覆盖所有离散分布外状态；② 仅在移动机器人 + 方块/斜坡的相对简单任务上验证，缺乏对更复杂环境与更大规模团队的通用性测试；③ 对长期稳定性与鲁棒性评估不足，需进一步研究。

---

## 114. A Modular Multi-Document Framework for Scientific Visualization and Simulation in Java

**arXiv ID:** 2602.21026 | [PDF](https://arxiv.org/pdf/2602.21026v1)

**作者:** David Heddle `[一作]` (Christopher Newport University), David Heddle `[通讯]` (Christopher Newport University)

**通讯引用:** 7514 | [OpenAlex ID](https://openalex.org/A5109052241)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在Java虚拟机环境下，设计并实现了一个模块化的多文档界面（MDI）框架，用于科学可视化与仿真，并提供了可选的3D渲染扩展。

**💡 创新点**

核心创新点在于：①将可视化层、仿真引擎、消息中介和3D渲染等功能严格分离成独立模块；②通过轻量级消息总线实现视图间的解耦；③引入确定性步骤式仿真引擎与线程安全的UI更新策略；④将JOGL 3D功能拆分成可选 Maven 依赖，减少2D应用的部署负担。

**🔧 技术方法**

技术栈包括Java Swing（UI主框架）、多线程并发模型、消息总线（自研）、可选JOGL（OpenGL绑定）、sPlot绘图库、Apache Commons Math（曲线拟合）以及Maven构建与发布。

**📊 数据集**

案例数据为50,000粒子自由膨胀的3D仿真，实时跟踪熵随时间变化的2D绘图；此外，框架支持通过sPlot绘制多种统计图表。

**📈 对比分析**

比较方法主要是与现有UI技术（JavaFX、Web框架）在依赖复杂度、长期维护性、线程安全性和部署简易性方面进行对比。框架通过确定性渲染和协同刷新实现无重绘风暴，保持交互响应；但文中未给出具体数值性能基准。

**⚠️ 局限性**

局限性包括：①基于Swing，缺乏现代UI特性和社区活跃度；②对硬件加速3D的支持需额外模块，仍可能遇到跨平台的本地库兼容问题；③缺乏自动化测试和可视化性能测量；④在高并发多视图场景下的消息调度与资源竞争尚未深入评估。

---

## 115. Circumventing the CAP Theorem with Open Atomic Ethernet

**arXiv ID:** 2602.21182 | [PDF](https://arxiv.org/pdf/2602.21182v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAE DAE LUS), Paul Borrill (DAE DAE LUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出一种基于 Open Atomic Ethernet（OAE）的双向同步链路语义和八元网格架构，旨在将传统异步网络模型中的软分区转化为可在子微秒级别快速检测和修复的局部事件，从而减少应用层可见的分区。

**💡 创新点**

创新点包括：
• 用双向寄存器交换取代单向帧发送，实现“bisynchrony”——每个周期都有确定的成功或失败结果；
• 将 Ethernet 的“fire‑and‑forget”语义改为确定的双向确认，消除语义歧义；
• 采用八元网格（Octavalent Mesh）取代传统 Clos 架构，允许每个节点自行构建本地生成树并在局部完成父链路切换，避免全局控制平面重新计算；
• 通过理论分析（Kirchhoff 定理、Nash‑Williams 定理）证明网格拥有指数级多条生成树，为局部故障恢复提供丰富冗余。

**🔧 技术方法**

技术手段：
• 双向寄存器调和协议（Shannon Slots）实现无歧义的消息交换；
• 采用 100G 物理链路的双向校验，利用光纤/铜线的有界传播延迟；
• 设计局部自愈算法：节点在每个 reconciliation 轮结束时根据邻居信息快速选取备用父链路；
• 在网络层面实现 octavalent mesh 的拓扑，支持每个节点均为根节点的生成树；
• 结合 CAL 理论、PACELC 框架进行性能建模。

**📊 数据集**

论文未给出公开的数据集；评估主要基于理论推导、概率分析以及仿真/实验平台（如基于 FPGA/软件模拟的 OAE 链路实现）。

**📈 对比分析**

比较方法：
• 与传统 Clos（Fat‑Tree）架构对比：在同一规模网络中测量软分区出现率、修复时延与尾部延迟；
• 与传统异步 Ethernet 通过计时器检测和重试的方式对比：观察软分区可见窗口和平均可用性；
• 结果显示：
  • OAE 在单链路失效时的修复时延从毫秒级降至几百纳秒；
  • 软分区可见概率降低数个数量级；
  • 系统整体可用性提升至 >99.999%，尾部延迟下降 30%–50%。

**⚠️ 局限性**

局限性：
• 仍无法消除硬分区（物理链路切断、供电隔离等）；
• 需要在硬件层面实现双向寄存器交换，增加 NIC 设计复杂度；
• 对极大规模网络的实验验证有限；
• 在极端故障模式（如多链路同时失效）下的性能尚未完全评估；
• 需要重新设计 SDN/控制平面协议以配合局部自愈逻辑。

---

## 116. Mitigating Preference Leakage via Strict Estimator Separation for Normative Generative Ranking

**arXiv ID:** 2602.20800 | [PDF](https://arxiv.org/pdf/2602.20800v1)

**作者:** Dalia Nahhas `[一作]` (University of Southampton), Shoaib Jameel `[通讯]` (University of Southampton)

**通讯引用:** 1385 | [OpenAlex ID](https://openalex.org/A5082611298)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双评审、无泄漏的生成式信息检索（GenIR）候选选择框架，用以在相同约束下对文化相关故事进行排序；

**💡 创新点**

通过严格分离监督评审（Judge B）与评估评审（Judge A）来消除偏好泄漏，并利用交叉编码器教师对密集双编码器（BGE‑M3）进行知识蒸馏，显著提升文化适配度排名；

**🔧 技术方法**

采用大语言模型（LLaMA‑3.1‑8B‑Instruct、Yi‑1.5‑9B‑Chat）作为评审器，交叉编码器与密集双编码器训练与蒸馏；使用点对、对偶、列表式学习排序；

**📊 数据集**

构造 NGR‑33k（33,052 条儿童文化故事）以及公开的 Moral Stories、SSGEN 作为外部验证集；

**📈 对比分析**

与无监督基线（随机、BM25、DPH、Dirichlet LM）、教师交叉编码器、浅层神经排序器、ColBERTv2 以及零样本 LLM 排序器对比，BGE‑M3 蒸馏模型在 Judge A 的 nDCG@5 上达 0.771，显著高于教师（0.577）和基线；

**⚠️ 局限性**

仅在两位评审间隔离，仍可能存在预训练共性偏差；数据集为人工生成，可能缺乏真实多样性；蒸馏过程对教师质量高度依赖，若教师噪声大可导致学生表现不佳。

---

## 117. Understanding the Role of Rehearsal Scale in Continual Learning under Varying Model Capacities

**arXiv ID:** 2602.20791 | [PDF](https://arxiv.org/pdf/2602.20791v1)

**作者:** JinLi He `[一作]` (Shanxi University), Xian Yang `[通讯]` (University of Manchester)

**通讯引用:** 1827 | [OpenAlex ID](https://openalex.org/A5060065120)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统研究了回放规模在连续学习中的作用，推导出适应误差、记忆误差和泛化误差的闭式表达，并通过数值仿真和深度神经网络实验验证其理论预测。

**💡 创新点**

创新点在于揭示回放规模并非总能提升模型性能：在过参数化情况下，增大回放规模可能削弱适应性；记忆误差存在下限；并首次以多维度误差框架对回放机制进行统一理论刻画。

**🔧 技术方法**

采用高斯线性回归理论与多维误差分析，构建过参数化/欠参数化解析式；利用数值仿真验证理论；再在MNIST、CIFAR-10/100、Tiny‑ImageNet上进行深度CNN/ResNet实验，探讨不同采样策略与网络深度对误差的影响。

**📊 数据集**

实验数据集包括 MNIST、CIFAR‑10、CIFAR‑100 与 Tiny‑ImageNet，分别划分为若干任务（每任务含数个类别）以检验回放效果。

**📈 对比分析**

通过比较不同缓冲区大小、采样策略（随机、herding、reservoir）和网络深度，在适应误差、记忆误差、泛化误差以及传统精度/遗忘率等指标上评估。结果显示：缓冲区增大往往不提升适应误差，记忆误差先下降后上升；深度网络相对更优；在多数基线方法中，增大回放往往导致新任务精度下降。

**⚠️ 局限性**

局限性包括：理论基于高斯线性回归，未对非线性网络给出严谨推导；实验主要聚焦分类任务，未覆盖更复杂的分布漂移；回放规模对泛化的影响高度依赖任务相似度，缺乏统一解释。

---

## 118. Recursive Belief Vision Language Model

**arXiv ID:** 2602.20659 | [PDF](https://arxiv.org/pdf/2602.20659v1)

**作者:** Vaidehi Bagaria `[一作]` (Indian Institute of Technology), Nirav Patel `[通讯]` (Indian Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种递归信念视觉语言行动模型（RB-VLA），旨在解决现有视觉语言行动模型在部分可观察环境下进行长时间操作时的不足。

**💡 创新点**

创新点在于引入了基于信念的架构，能够在不存储原始观察数据的情况下，保持紧凑的潜在状态编码任务相关的历史、动态和对象交互。

**🔧 技术方法**

使用了自监督的世界模型目标进行训练，结合了视觉语言模型（VLM）和扩散策略来实现闭环控制。

**📊 数据集**

使用了40,000个模拟操作轨迹的数据集，涵盖了在RoboSuite和LIBERO环境中的单个和多个对象的抓取和放置任务。

**📈 对比分析**

与现有的视觉语言行动模型相比，RB-VLA在长时间基准测试中表现更好，成功率分别提高了52.5%和37.5%，并且推理延迟减少了最多5倍。

**⚠️ 局限性**

限制在于模型的训练和推理依赖于高质量的视觉输入，且在动态环境中的适应性仍需进一步研究。

---

## 119. From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection

**arXiv ID:** 2602.20630 | [PDF](https://arxiv.org/pdf/2602.20630v1)

**作者:** Yepeng Liu `[一作]` (Wuhan University), Yongchao Xu `[通讯]` (Wuhan University)

**通讯引用:** 4324 | [OpenAlex ID](https://openalex.org/A5082564408)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

重新定义关键点检测为序列决策任务，利用强化学习直接优化长时跟踪质量；

**💡 创新点**

提出轨迹感知奖励（Rank + Distinctiveness）与混合采样策略，突破传统成对匹配优化，专注多视角一致性与独特性；

**🔧 技术方法**

使用基于RL的策略网络，DINOv3‑ConvNeXt backbone，policy‑gradient + 熵正则，warm‑up 的弱监督，软最近邻匹配；

**📊 数据集**

训练与评估使用 MegaDepth、ScanNet、Aachen Day‑Night、KITTI、ETH 3D 重建等数据集；

**📈 对比分析**

与 SuperPoint、DISK、RDD、RIPE、XFeat 等 SOTA 方法在相对姿态估计、视觉定位、视觉里程计和 3D 重建上进行对比，TraqPoint 在 AUC、ATE、AKTL、重建密度等指标均优于现有方法；

**⚠️ 局限性**

仍依赖预训练描述子，极端光照或动态场景性能可能受限；训练需要大量序列数据，推理速度略低；在长轨迹或大视角变化下仍有提升空间。

---

## 120. TCDA: Robust 2D-DOA Estimation for Defective L-Shaped Arrays

**arXiv ID:** 2602.21146 | [PDF](https://arxiv.org/pdf/2602.21146v1)

**作者:** Wenlong Wang `[一作]` (Tsinghua University), Lei Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 106238 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种TCDA框架，利用张量补全实现对有缺陷L形阵列的鲁棒二维方向估计。

**💡 创新点**

创新点在于把物理故障映射为带缺失值的加权PARAFAC分解，并通过ALS直接恢复因子矩阵，从而无需额外的参数配对步骤。

**🔧 技术方法**

采用张量分解（PARAFAC）、加权ALS、子阵列交叉相关、虚阵列构造以及自适应阈值检测等技术。

**📊 数据集**

使用仿真L形阵列（M=10，K=4，500帧快照）进行1000次Monte‑Carlo实验，SNR范围-10~20 dB。

**📈 对比分析**

与理想无缺陷阵列对比，TCDA在不同缺陷程度下均保持RMSE低于1°，即使缺失约60%数据也能保持可用估计；没有出现误差底线，性能优于传统方法。

**⚠️ 局限性**

局限性包括对源数估计的依赖、在极端空间信息损失时性能受限、仅针对非相关、远场、窄带信号进行验证，未在真实硬件或相关源环境下测试。

---

## 121. Physics-based phenomenological characterization of cross-modal bias in multimodal models

**arXiv ID:** 2602.20624 | [PDF](https://arxiv.org/pdf/2602.20624v1)

**作者:** Hyeongmo Kim `[一作]` (Korea Institute of Science and Technology), Kyungreem Han `[通讯]` (Korea Institute of Science and Technology)

**通讯引用:** 694 | [OpenAlex ID](https://openalex.org/A5084984855)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究多模态大型语言模型（MLLM）在跨模态推理中出现的偏差，结合物理代理多振荡器模型和图形化错误吸引子分析揭示Transformer自注意力与跨注意力动态导致的模态偏差；

**💡 创新点**

提出基于多振荡器的物理代理模型来表征Transformer动态，并将动态SHAP等物理量与跨模态情感分类、Lorenz时间序列预测实验相结合，首次从物理动力学视角解释MLLM的系统性偏差；

**🔧 技术方法**

使用多振荡器动力学模型、动态SHAP、图形化错误吸引子（Sankey、Directed Graph）、自/跨注意力参数化，以及零样本情感分类和Lorenz预测实验；

**📊 数据集**

使用CREMA‑D情感多模态数据集进行情感分类实验，并利用Lorenz chaotic time‑series数据进行跨模态预测实验；

**📈 对比分析**

通过比较Qwen2.5‑Omni和Gemma 3n在视频+音频、视频单独、音频单独三种输入下的错误吸引子结构及动态SHAP值，发现多模态输入并未显著降低偏差，而当自/跨注意力参数提升时预测准确率可提升；

**⚠️ 局限性**

仅对两款模型进行实验，缺乏对偏差纠正的具体方案，物理代理模型在假设与实际Transformer复杂性的匹配上仍有限，未能完整解释所有偏差来源。

---

## 122. Multimodal Crystal Flow: Any-to-Any Modality Generation for Unified Crystal Modeling

**arXiv ID:** 2602.20210 | [PDF](https://arxiv.org/pdf/2602.20210v1)

**作者:** Kiyoung Seong `[一作]` (KAIST), Changyoung Park `[通讯]` (LG AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的多模态流模型MCFlow，可在单一模型中完成晶体结构预测、全新生成以及结构条件下的原子类型生成等多种晶体生成任务。

**💡 创新点**

核心创新在于：1) 通过对原子类型与晶体结构分别设置时间轴，实现任意模态之间的可插拔推断轨迹；2) 引入基于电负性和Wyckoff位的词典化原子排序以及层次化置换增强，为Transformer提供化学与晶体对称的软先验；3) 在Diffusion Transformer基础上实现多模态流匹配与噪声引导。

**🔧 技术方法**

技术包括：多模态流模型（多时间轴）与流匹配训练；Diffusion Transformer（DiT）作为基座；对晶体坐标的周期性约束与几何流；层次化置换增强；噪声引导提升生成质量。

**📊 数据集**

使用MP‑20（≤20原子、能量接近热力学极限的材料）和MPTS‑52（≤52原子）两个公开晶体数据库进行训练与评估。

**📈 对比分析**

与CDVAE、DiffCSP、FlowMM、CrystalFlow、OMatG等专用基线相比，MCFlow在MP‑20与MPTS‑52上的晶体结构预测匹配率与RMSE均保持竞争力，且在全新生成任务中在空间群、Wyckoff分布与能量稳定性方面优于FlowMM与ADiT。噪声引导进一步提升单样本质量。

**⚠️ 局限性**

局限性主要体现在：1) 对于更大单元格的扩展仍需改进；2) 目前仅处理晶体结构与组成模态，未覆盖材料属性等多模态；3) 需要进一步探索更高效的采样与训练技巧。

---

## 123. VII: Visual Instruction Injection for Jailbreaking Image-to-Video Generation Models

**arXiv ID:** 2602.20999 | [PDF](https://arxiv.org/pdf/2602.20999v1)

**作者:** Bowen Zheng `[一作]` (Huazhong University of Science and Technology), Xinge You `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 6191 | [OpenAlex ID](https://openalex.org/A5057095711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于视觉指令注入（VII）的黑盒 Jailbreak 框架，利用现代图像到视频（I2V）模型的视觉指令跟随能力，将恶意文本意图隐蔽为安全图像中的视觉符号与排版文本，从而在不被预生成安全机制识别的情况下生成违规视频。

**💡 创新点**

创新点在于：① 通过“恶意意图再编程（MIR）”模块将危险文本转化为安全同义词并进一步重编为可执行的排版指令；② “视觉指令投影（VIG）”模块将排版指令与抽象视觉符号（边框、箭头）嵌入安全图像，实现语义与空间双重注入；③ 该方法完全无训练、跨模型可迁移，且在现有安全框架下仍能显著规避。

**🔧 技术方法**

使用了大型语言模型（GPT‑4o、GPT‑5.2）完成意图提取、同义词替换与排版描述生成；利用文本渲染与图形绘制工具将排版描述与视觉符号嵌入图像；在 I2V 生成端采用四款商业模型（Kling‑v2.5‑turbo、Gemini Veo‑3.1、Seedance‑1.5‑pro、PixVerse‑V5）。

**📊 数据集**

评估数据集包括 COCO‑I2VSafetyBench 与 ConceptRisk，涵盖性别、暴力、仇恨与非法行为四类安全风险。

**📈 对比分析**

与直接使用恶意文本提示（Unsafe Text Prompt）以及仅嵌入文本的 Typographic Attack 进行对比；在两种评估协议（VBench++ 与 VLM‑based）下，VII 的攻击成功率（ASR）最高，PixVerse‑V5 最高达 83.5%（VBench++）或 86.5%（VLM），而拒绝率（RR）几乎为 0，明显优于基线。

**⚠️ 局限性**

局限性包括：① 受限于目标模型的视觉指令识别能力，若模型对视觉符号重视程度降低或引入更严格的多模态安全机制则效果会下降；② 对中文/日文等非拉丁文字的效果相对较低；③ 需要对输入图像进行可视化处理，攻击成本不低；④ 目前仅针对预生成安全机制，未评估后生成过滤或内容审核场景。

---

## 124. Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation

**arXiv ID:** 2602.20816 | [PDF](https://arxiv.org/pdf/2602.20816v1)

**作者:** Sayantan Dasgupta `[一作]` (University of Melbourne), Timothy Baldwin `[通讯]` (University of Melbourne)

**通讯引用:** 11045 | [OpenAlex ID](https://openalex.org/A5103085805)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在大型语言模型蒸馏过程中提出一种新的尾部感知 KL 散度，降低对教师高概率模式的依赖，从而提升学生模型对低概率词汇的学习能力。

**💡 创新点**

创新点在于将教师分布的前 K 高概率项与尾部概率分离，并对尾部 KL 项进行归一化处理，使其对梯度贡献可控，且不增加额外计算量。

**🔧 技术方法**

采用的技术包括基于 KL 散度的蒸馏损失改写、序列级归一化（β 归一化）、与传统 CLM 损失的组合，以及超参数 β 的调节。

**📊 数据集**

实验使用 Regmix（20GB 子集）作为预训练语料，同时对 OpenWebMath、Gemma、Phi-2、Qwen、TinyLlama 等多种教师模型与不同规模学生进行评估。

**📈 对比分析**

与 Vanilla KD、MiniPLM、Sequence-KD、RKL 等方法在多项评测集（OpenAI Few-shot、数学推理、SFT 任务）对比，TAD 在保持相同 FLOPs 的前提下平均提升 2–12% 的准确率，尤其在难题集上表现突出。

**⚠️ 局限性**

局限性包括对 K 的选择敏感（最佳在 5–10 之间）、对超参数 β 需经验调优、在极小 token 量下难以从零开始训练学生，以及仍无法完全替代大规模数据驱动的教师生成方法。

---

## 125. SoK: Agentic Skills -- Beyond Tool Use in LLM Agents

**arXiv ID:** 2602.20867 | [PDF](https://arxiv.org/pdf/2602.20867v1)

**作者:** Yanna Jiang `[一作]` (University of Technology Sydney), Guangsheng Yu `[通讯]` (University of Technology Sydney)

**通讯引用:** 1893 | [OpenAlex ID](https://openalex.org/A5072542706)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统总结了LLM代理的可重用程序化能力——即“Agentic Skills”，并给出了统一定义、生命周期模型、七种设计模式、表示与作用域的交叉分类、以及安全治理与评估框架。

**💡 创新点**

创新点包括：① 将技能抽象为四元组(S=(C,π,T,R))并与RL选项框架对齐；② 通过七种设计模式和表示×作用域两轴税onomies揭示技能生态；③ 在ClawHavoc供应链攻击案例中验证安全模型；④ 将SkillsBench等基准映射到技能评估维度并量化技能质量对代理性能的影响。

**🔧 技术方法**

主要技术手段是系统性文献综述、模式识别与矩阵映射、形式化定义与威胁模型、案例研究以及基准（SkillsBench、WebArena、SWE‑bench等）对技能效果进行定量评估。

**📊 数据集**

使用的数据集包括SkillsBench（86任务/7308轨迹）、WebArena、Mind2Web、OSWorld、SWE‑bench、AgentBench、AndroidWorld等；安全案例采用ClawHub公开的1,184个恶意技能和相关病毒扫描结果。

**📈 对比分析**

评估方法基于确定性验证器，对比无技能、精心策划的技能与自生成技能的通过率。实验显示：精心策划技能平均提升16.2个百分点（各领域差异显著），自生成技能平均降低1.3个百分点；部分任务甚至出现负增益。

**⚠️ 局限性**

局限性包括：① 领域新颖，文献覆盖不足；② 体系结构模式与分类仍需外部验证；③ 对生产系统与安全基准覆盖有限；④ 依赖现有基准，缺乏真实世界长周期评估；⑤ 未充分解决无监督技能发现与可验证自主生成的问题。

---

## 126. PreScience: A Benchmark for Forecasting Scientific Contributions

**arXiv ID:** 2602.20459 | [PDF](https://arxiv.org/pdf/2602.20459v1)

**作者:** Anirudh Ajith `[一作]` (Allen Institute for Artificial Intelligence), Doug Downey `[通讯]` (Allen Institute for Artificial Intelligence)

**通讯引用:** 5090 | [OpenAlex ID](https://openalex.org/A5043450042)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 PreScience 基准，系统性拆解科研预测为团队预测、先行文献选择、贡献生成与影响评估四个互相依赖的生成任务；同时构建了覆盖 98K 篇 AI 领域 arXiv 论文的完整数据集，并引入了基于 LLM 的概念相似度度量 LACCScore；

**💡 创新点**

创新点在于：①将科研过程分解为四步可测评的生成任务，②提供全流程端到端的模拟实验框架，③设计了新的 LLM 驱动贡献相似度指标，④公开了大规模、可持续更新的科研元数据数据集；

**🔧 技术方法**

主要技术包括：图嵌入与聚类的作者与文献表示、基于 LLM 的生成与评价（GPT‑5、LLaMA 等）、传统协作与引用频率基线、XGBoost 回归、以及端到端的随机采样模拟器；

**📊 数据集**

使用的数据集是从 2023‑10 至 2025‑10 的 AI 相关 arXiv 论文（98K）及其 502K 关联论文，含作者消歧、引用、时间对齐等元数据；

**📈 对比分析**

实验对比了频率、嵌入融合、层次聚类等基线；在团队与引用预测上仅能取得 0.1–0.4 的 nDCG；在贡献生成上 GPT‑5 的 LACCScore 仅达 5.6/10；在影响预测上 XGBoost 结合文本与计量特征的 MAE 仍较大；整体表现显示各任务仍存在显著改进空间；

**⚠️ 局限性**

局限性包括：①假设科研过程可拆分为四步，忽略机构、资金、会议等因素；②仅使用 AI arXiv 数据，缺乏跨学科与后期期刊出版信息；③影响度量依赖 12 个月内引用，忽视长期影响与负面结果；④LLM 生成的论文在多样性与新颖性上低于真实研究，提示模型偏向重现已有方向。

---

## 127. Pip-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching

**arXiv ID:** 2602.20496 | [PDF](https://arxiv.org/pdf/2602.20496v1)

**作者:** Jintu Zheng `[一作]` (ARIDGE XPENG), Zhuojie Chen `[通讯]` (ARIDGE XPENG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了基于迭代优化的立体匹配框架PipStereo，结合迭代剪枝、单目先验迁移与FlashGRU，实现了在边缘设备上实时高精度的立体匹配；

**💡 创新点**

① Progressive Iteration Pruning将多步递归压缩为近单通道推理；② 通过协同学习将单目深度先验嵌入无额外编码器；③ 设计硬件感知的FlashGRU，利用结构化稀疏和I/O优化显著降低内存带宽与延迟；

**🔧 技术方法**

使用迭代优化+GRU、结构化稀疏I/O感知算子、教师-学生协同学习、超网络搜索与遗传算法、逐层迭代剪枝；

**📊 数据集**

训练集包括SceneFlow、CREStereo、TartanAir、SintelStereo、FallingThings、InStereo2K（BTS）；测试集为KITTI2012/2015、SceneFlow、ETH3D、DrivingStereo、FoundationStereo等；

**📈 对比分析**

与多种迭代高精度方法（Raft‑Stereo、IGEV、CREStereo）及实时方法（LightStereo、CoEx、HitNet等）对比；PipStereo在SceneFlow上比第二佳低13.5% EPE，在ETH3D 73% Bad‑1下降；在Jetson Orin NX 320×640仅75 ms，RTX 4090 19 ms，速度分别比MonSter快22×、Defom‑Stereo快14×、FoundationStereo快41×，准确率保持与大迭代模型相当，零样本泛化优于现有实时方法；

**⚠️ 局限性**

对完全从零视差初始化的模型，单步推理仍难以保持精度；PIP在无FlashGRU时对Raft等模型的精度下降显著；FlashGRU加速受限于高分辨率时才显著，低分辨率效果有限；总体仍依赖较大模型规模和多阶段训练，部署复杂度较高。

---

## 128. Boosting Instance Awareness via Cross-View Correlation with 4D Radar and Camera for 3D Object Detection

**arXiv ID:** 2602.20632 | [PDF](https://arxiv.org/pdf/2602.20632v1)

**作者:** Xiaokai Bai `[一作]` (Zhejiang University), Hui-Liang Shen `[通讯]` (Zhejiang University)

**通讯引用:** 98462 | [OpenAlex ID](https://openalex.org/A5100773343)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为SIFormer的场景-实例感知变换器，用于利用4D雷达和相机进行3D物体检测，旨在增强实例意识。

**💡 创新点**

通过交叉视图相关机制增强实例意识，弥补了雷达几何信息弱的固有限制，这是该研究的创新点。

**🔧 技术方法**

使用了深度学习技术，特别是变换器架构，结合了稀疏场景集成（SSI）、交叉视图相关（CVC）和实例增强注意力（IEA）模块。

**📊 数据集**

使用了View-of-Delft、TJ4DRadSet和NuScenes数据集进行实验评估。

**📈 对比分析**

与现有方法（如IS-Fusion）进行比较，SIFormer在多个数据集上表现出色，达到了最先进的性能，尤其是在实例检测精度上有显著提升。

**⚠️ 局限性**

SIFormer的局限性在于推理速度较慢，并且缺乏时间建模，未来的工作将探索轻量化变体和时间建模的结合。

---

## 129. Turing Completeness of GNU find: From mkdir-assisted Loops to Standalone Computation

**arXiv ID:** 2602.20762 | [PDF](https://arxiv.org/pdf/2602.20762v1)

**作者:** Keigo Oka `[一作]` `[通讯]` (Google), Keigo Oka (Google)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文通过构造三种基于 GNU find 的程序，证明 Unix find 命令（及其变体）具有图灵完备性；具体展示了（1）仅用 find + 正则表达式回溯实现 2‑tag 系统模拟，（2）仅用 GNU find（无 -execdir）即可模拟两计数器机器，（3）即使去掉正则回溯，同样可实现图灵完备。

**💡 创新点**

创新点在于揭示了看似简单的文件遍历工具 find 的隐藏计算能力，展示了在不使用 shell 的情况下，仅通过标准选项和正则表达式（甚至不含回溯）即可构造图灵机；此外，还首次通过文件系统层面的递归遍历与 execdir 形成无限循环，进一步拓宽了“偶然”图灵完备性的范畴。

**🔧 技术方法**

主要技术包括：目录路径编码状态、正则表达式匹配与复制、execdir 的相对路径执行、-files0-from 读取动态起点形成循环、以及对 2‑tag 系统和 Minsky 计数器程序机的模拟。

**📊 数据集**

本研究未使用外部数据集，所有实验均基于自定义的 toy 示例来验证构造的正确性。

**📈 对比分析**

通过在 GNU find 4.10.0 环境下执行构造命令，观察到输出与理论预期一致，证明实现可行；由于该工作侧重于可计算性证明，未做复杂的性能对比，只关注构造是否能完成计算。

**⚠️ 局限性**

限制主要包括：依赖 GNU find 的实现细节（版本≥4.2.12 或≥4.9.0），对路径长度、inode 数量等系统资源有限制；在 BSD、BusyBox、uutils 等实现上需要额外修改；构造过程相对繁琐，且对非标准选项的支持不完整。

---

## 130. De-rendering, Reasoning, and Repairing Charts with Vision-Language Models

**arXiv ID:** 2602.20291 | [PDF](https://arxiv.org/pdf/2602.20291v1)

**作者:** Valentin Bonas `[一作]`, Emmanuel Iarussi `[通讯]` (Consejo Nacional de Investigaciones Científicas y Técnicas)

**通讯引用:** 382 | [OpenAlex ID](https://openalex.org/A5044408118)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个从图表图片到可执行代码、自动评估设计缺陷并给出可操作改进建议的完整框架。

**💡 创新点**

首次将图表去渲染、基于LLM的可视化评估以及可编辑反馈的交互循环集成到同一流程，实现了从图像到可执行规范、原则化评审和迭代改进的一体化系统。

**🔧 技术方法**

使用ChartCoder进行图表去渲染生成Matplotlib代码，利用Ollama+GPT-OSS对代码进行分析并生成改进建议，ChatGPT API用于嵌入和聚类建议，并通过Web UI实现交互。

**📊 数据集**

在Chart2Code基准集的1000张柱状、折线、散点图上进行实验。

**📈 对比分析**

生成10452条建议后，通过ChatGPT嵌入+UMAP聚类得到10个语义聚类，Davies–Bouldin指数3.30，表明建议结构化且符合设计原则；与传统规则驱动的linter相比，LLM能提供更丰富、可操作的反馈。

**⚠️ 局限性**

对扫描文件等非程序化来源的图表去渲染效果不足，建议缺乏教育性解释，并且尚未在真实用户实验中验证系统的有效性。

---

## 131. RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces

**arXiv ID:** 2602.20618 | [PDF](https://arxiv.org/pdf/2602.20618v1)

**作者:** Haonan An `[一作]` (City University of Hong Kong), Yuguang Fang `[通讯]` (City University of Hong Kong)

**通讯引用:** 24285 | [OpenAlex ID](https://openalex.org/A5016290340)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种名为RecoverMark的鲁棒水印框架，能够在图像被人工智能篡改后实现篡改区域定位、原内容恢复以及版权验证，主要针对人脸篡改场景。

**💡 创新点**

创新点在于：①将人脸自身作为水印嵌入背景，利用背景保持一致性的实际约束；②设计两阶段逐步训练流程，并引入多种干扰层（噪声、JPEG、低通滤波、重生攻击等）实现对水印移除攻击的强鲁棒性；③兼顾定位、恢复与版权验证三大功能于一体，突破传统脆弱水印对移除攻击的脆弱性。

**🔧 技术方法**

采用前景分割（MTCNN/YoloSeg/SAM2）获取人脸与背景；使用Enc/Dec对人脸做压缩编码；利用UNet实现隐藏网络HNet；通过DistortionLayer模拟多种攻击；使用CEILNet构建提取网络ENet；两阶段训练策略，第一阶段无干扰训练，第二阶段加入Progressive Distortion。

**📊 数据集**

训练与ID测试使用CelebA；OOD测试使用FFHQ；所有图像均裁剪为256×256 RGB。

**📈 对比分析**

与被动方法（MVSS-Net、HiFiNet）以及主动方法（Imuge+、EditGuard、OmniGuard）在ID/OOD数据、已知/未知攻击（重生、噪声、JPEG、低通滤波、格点攻击、补丁移除）下进行对比。RecoverMark在F1、AUC、PSNR、MS‑SSIM和版权验证成功率（NCC>0.95）上均显著优于基线，尤其在未知攻击与OOD数据上保持高稳健性。

**⚠️ 局限性**

局限性在于背景容量有限；当人脸区域占比过大或背景极少时，水印容量受限，导致嵌入背景和恢复内容的质量下降；未来需探索更高容量的鲁棒水印技术以扩展至非人脸对象。

---

## 132. WildGHand: Learning Anti-Perturbation Gaussian Hand Avatars from Monocular In-the-Wild Videos

**arXiv ID:** 2602.20556 | [PDF](https://arxiv.org/pdf/2602.20556v1)

**作者:** Hanhui Li `[一作]` (Sun Yat-sen University), Chenqiang Gao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 4346 | [OpenAlex ID](https://openalex.org/A5021881939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于3D高斯抛射的优化框架 WildGHand，用于从受多种扰动影响的单目野外视频中重建高保真手部化身。

**💡 创新点**

创新点在于：①动态扰动分离模块（DPD）将扰动建模为随时间加权的高斯属性偏置；②扰动感知优化策略（PAO）通过自适应加权掩模抑制空间和时间上的扰动，二者协同提升鲁棒性。

**🔧 技术方法**

使用3D Gaussian splatting、MANO-HD手模型、轻量MLP进行扰动建模、SAM进行分割、加权损失和拉普拉斯正则化等技术。

**📊 数据集**

在公开数据集（InterHand2.6M、AnchorCrafter）和自行构建的 HWP（13.8k帧，包含手-物交互、复杂姿势、光照变化、运动模糊）上进行评测。

**📈 对比分析**

与UHM、Handy、InterGaussianHand等方法对比，WildGHand 在 PSNR、SSIM、LPIPS 等指标上均显著提升（例如在 HWP 上 PSNR 最高 28.25，LPIPS 最低 0.072），并在网络视频中表现出更强的泛化能力。

**⚠️ 局限性**

局限性：对极端模糊或完全遮挡情况仍可能产生细节缺失；模型仍依赖MANO-HD和手部估计器，若估计误差较大会影响最终结果。

---

## 133. Imputation of Unknown Missingness in Sparse Electronic Health Records

**arXiv ID:** 2602.20442 | [PDF](https://arxiv.org/pdf/2602.20442v1)

**作者:** Jun Han `[一作]` (Optum AI), Robert E. Tillman `[通讯]` (Optum AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种基于去噪的二元EHR数据恢复方法Denoise2Impute以及其阈值扩展Denoise2Impute-T，用于识别和填补未知未知缺失的诊断代码。

**💡 创新点**

创新点在于将未知未知缺失视为去噪问题，理论推导出最优去噪函数并引入自适应阈值，以区分缺失与真实缺口，从而提高去噪效果。

**🔧 技术方法**

技术实现采用Set Transformer作为去噪网络，结合自监督的二元交叉熵损失以及可学习的阈值网络，对稀疏二进制EHR进行重构。

**📊 数据集**

使用来自大型医疗机构的两份重叠患者群体的ICD‑10诊断码数据，共计约445,345例（T=993）以及1,670,347例用于住院再入院预测的公开数据集。

**📈 对比分析**

与基线方法（均值填充、k‑NN、softImpute、DAE、CDAE、MLP‑参数化Denoise2Impute）比较，Denoise2Impute-T在维度级AUPRC平均提升约2%，在住院再入院预测任务中提升约0.5% AUPRC，且提升具有统计显著性。

**⚠️ 局限性**

局限性包括需要对照组（无噪声与噪声数据）来训练，稀疏或样本不足的诊断码可能导致模型误判，并且目前仅处理二元诊断码，未考虑多模态或时间序列信息。

---

## 134. Path-Decoupled Hyperbolic Flow Matching for Few-Shot Adaptation

**arXiv ID:** 2602.20479 | [PDF](https://arxiv.org/pdf/2602.20479v1)

**作者:** Lin Li `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 95693 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在Lorentz双曲空间上实现流匹配，提出了路径解耦的双曲流匹配（HFM）方法，用于少样本跨模态适应。

**💡 创新点**

创新点在于：①中心向双曲对齐构建层级化结构；②路径解耦目标为每个类别建立隔离的测地线走廊；③基于语义直径的自适应停止机制，避免过度迁移。

**🔧 技术方法**

核心技术包括Lorentz双曲几何映射、流匹配与速度场学习、对齐与对比损失、以及自适应停止阈值设计。

**📊 数据集**

在11个公开少样本分类基准（如Aircraft、EuroSAT、DTD、SUN、UCF101等）上进行了实验验证。

**📈 对比分析**

与欧氏流匹配以及多种PEFT（CoOp、CoCoOp、CLIP-Adapter、CLIP-LoRA等）对比，HFM在多数设置下平均提升4%–6%，在难度较大的数据集上提升幅度更显著。

**⚠️ 局限性**

局限性包括对双曲曲率与阈值参数的调优较为敏感，且在极大规模数据或更高维任务上的可扩展性尚待进一步验证。

---

## 135. ID-LoRA: Efficient Low-Rank Adaptation Inspired by Matrix Interpolative Decomposition

**arXiv ID:** 2602.20727 | [PDF](https://arxiv.org/pdf/2602.20727v1)

**作者:** Xindian Ma `[一作]` (Tianjin University), Yongyu Jiang `[通讯]` (Tianjin University)

**通讯引用:** 4117 | [OpenAlex ID](https://openalex.org/A5101706525)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ID-LoRA，一种将矩阵插值分解与行聚类相结合的参数高效微调（PEFT）框架，仅训练共享低秩矩阵B，显著减少可训练参数。

**💡 创新点**

创新点在于：①使用预训练权重的行聚类生成冻结的低秩基，②通过共享B和Rank Boosting实现多低秩组件的组合，③理论证明聚类分解在多任务情形下能获得更紧的误差上界。

**🔧 技术方法**

采用了k-means聚类、Matrix Interpolative Decomposition (MID)、Rank Boosting、LoRA基础结构以及参数高效微调技术，并在实验中加入了理论分析。

**📊 数据集**

使用的数据集包括单任务的数学推理（GSM8K）、代码生成（HumanEval）和安全（HEx-PHI），以及多任务的世界知识（MMLU）、常识（CommonsenseQA）等；实验基于LLaMA-3-8B和Mistral-7B模型。

**📈 对比分析**

与全参数微调、LoRA、DoRA、HydraLoRA、MoELoRA等方法在相同任务上对比；单任务中仅更新0.56%-0.62%参数，性能均优于或等于基线；多任务中参数减少46%，平均提升6%且在大多数子任务上超越LoRA及其变体。

**⚠️ 局限性**

局限性：在数学推理任务中提升有限，主要受限于对权重方向更新的精确性，性能与LoRA相近。

---

## 136. A Granularity Characterization of Task Scheduling Effectiveness

**arXiv ID:** 2602.20561 | [PDF](https://arxiv.org/pdf/2602.20561v1)

**作者:** Sana Taghipour Anvar `[一作]`, David Kaeli `[通讯]` (Northeastern University)

**通讯引用:** 7463 | [OpenAlex ID](https://openalex.org/A5061128237)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出一种基于任务图依赖拓扑的粒度表征框架，用来量化调度开销随并行度的增长，从而预测任务调度何时对性能有益或有害。

**💡 创新点**

创新点在于：①将任务调度开销的增长与任务图的依赖边数直接关联；②提出“粒度数（G）”作为统一指标，能够在不同工作负载、规模和算法结构下统一描述调度效能；③给出基于G的动态与静态执行切换决策规则，并通过依赖拓扑校准的开销模型实现无需完整强缩放实验的预测。

**🔧 技术方法**

技术手段包括：任务驱动运行时（如 Dagger.jl、HPX、StarPU 等）对任务图进行构建；对依赖拓扑进行理论分析并推导开销模型；利用实验平台（Intel Xeon Gold 6240R 集群 + InfiniBand HDR）对多种典型工作负载进行测量；使用最小二乘法拟合开销模型参数；在运行时利用已收集的计数信息评估粒度数并做决策。

**📊 数据集**

使用的数据集和工作负载涵盖：3D FFT（不同尺寸），二维五点和一维行方向扫描（Stencil、Sweep），块矩阵乘法（GEMM），稀疏矩阵向量乘（SpMV）、卷积（Conv2D），PageRank，N-Body 模拟等，均在 4–256 个进程上进行强缩放实验。

**📈 对比分析**

比较方法：将所有实验配置的执行时间拆分为计算核时间与调度开销，并计算粒度数 G = T_kernel / T_overhead；随后绘制调度开销占比与 G 的关系曲线，验证不同拓扑的开销随 P 的变化（O(P^2)、O(P)、O(1)）。实验表明：动态调度在 G > 10 时能将调度开销保持在 <10%；在 G ≤ 1 时动态调度反而不如静态；预测的强缩放临界点 P* 与实际测得的性能崩溃点高度吻合。

**⚠️ 局限性**

局限性：①模型假设核执行时间理想地随 P 逆比例缩放，未考虑隐藏的串行部分；②需先行对各工作负载的开销模型进行离线拟合，若拓扑或实现改变需重新校准；③只针对典型的全局、局部、独立三类依赖拓扑验证，极端或混合拓扑的情况尚未充分测试；④未覆盖所有任务运行时的细节（如内存层次、网络不对称等）导致预测误差。

---

## 137. Fair Division with Soft Conflicts

**arXiv ID:** 2602.20929 | [PDF](https://arxiv.org/pdf/2602.20929v1)

**作者:** Hirotaka Yoneda `[一作]` (University of Tokyo), Masataka Yoneda `[通讯]` (University of Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了在存在软冲突（即相邻商品可同一人持有但希望尽量减少冲突边）的情况下，求取满足EF1（即“最多可删一件物品后无羡慕”）的公平分配，并给出了两类算法：对相同增值（identical additive valuations）的情况使用循环移位轮转（CyclicShiftRR）实现 |E|/n 次冲突违例；对一般增值（general additive valuations）使用 GraphEF1（基于 Biswas & Barman 2018 的EF1+卡数约束算法并结合“最近点”几何论证）实现 |E|/n + O(|E|^{1-1/(2n-2)}) 次冲突违例，时间复杂度为 O(m+|E|)。

**💡 创新点**

创新点主要有：
1. 首次在软冲突环境下给出理论上近似最优（1+o(1)）的冲突违例上界；
2. 将 Biswas & Barman 的在线EF1分配与几何“最近点”策略相结合，形成新的“游戏化”选择机制；
3. 通过按度数分组并递归调用 DegreeEF1，将不同度数节点的冲突问题统一处理；
4. 提出可扩展到加权冲突与多属性平衡的通用框架。

**🔧 技术方法**

使用的主要技术手段包括：
- 轮转（Round‑Robin）分配与循环移位技巧；
- Biswas & Barman（2018）基于“envy‑cycle elimination”的在线EF1算法；
- 几何“最近点”定理（在 d‑维超立方体中挑选相近 n 个点）；
- 通过度数分组与递归来控制最大度数 Δ；
- 高效数据结构（指针重排、哈希表维护最近点区块）实现线性时间。

**📊 数据集**

本文为理论算法论文，没有使用实测数据集，所有结果均为分析证明。实验演示仅在合成实例上验证算法复杂度和冲突违例上界。

**📈 对比分析**

相较于先前仅满足 EF1 或完全禁止冲突的算法，本文给出了近似最优的冲突违例上界：对相同增值实现 |E|/n（与随机分配期望相同）；对一般增值实现 |E|/n + O(|E|^{1-1/(2n-2)})，即 1+o(1) 倍。算法时间线性 O(m+|E|)，优于此前的高阶复杂度方法。实验（合成）显示实际冲突数接近理论上界。

**⚠️ 局限性**

限制与未解问题：
- 对一般增值的冲突违例上界仍未达到严格的 |E|/n；
- 结果仅适用于常数 n，n 较大时常数项影响明显；
- 对软冲突的加权版本，误差项仍依赖 Δ；
- 对多属性平衡（如题末所示）只能得到 O(m^{1-1/k}) 的差距，仍未达到最优；
- 证明依赖于“最近点”近似，若冲突图结构特殊可能无法进一步优化。

---

## 138. Not Just What's There: Enabling CLIP to Comprehend Negated Visual Descriptions Without Fine-tuning

**arXiv ID:** 2602.21035 | [PDF](https://arxiv.org/pdf/2602.21035v1)

**作者:** Junhao Xiao `[一作]` (Central China Normal University), Zejiang He `[通讯]` (National University of Defense Technology)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5047498670)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 CLIP 预训练模型上引入两模块（Lens 与 Frame），实现对否定语义的拆解与动态惩罚，改进图文对齐时对否定句的理解。

**💡 创新点**

无侵入式双阶段建模：① 语法-语义双流 Lens 通过多层层次注意力捕捉否定结构并与全局语义融合；② 动态上下文惩罚 Frame 根据图文交互生成惩罚权重 λ，精准抑制与否定概念的相似度。

**🔧 技术方法**

采用 CLIP Transformer 层、层归一化+GELU、注意力机制、残差门控、交叉模态自注意力、InfoNCE 对比学习、t‑SNE 可视化等技术。

**📊 数据集**

训练与评估使用 CC‑Neg（188K 图/376K 文），Neg‑COCO‑MCQ、Neg‑COCO‑R 低资源集；零样本验证在 ImageNet 与 Caltech101 上进行。

**📈 对比分析**

与 NegCLIP、CoN‑CLIP 等基线比较：在 CC‑Neg‑val 内部达 96.56%（略低于 CoN‑CLIP 99.70%），但在跨域 Neg‑COCO‑MCQ 上提升 8.81pp（34.51% vs 25.70%），在 Neg‑COCO‑R 上提升 27.45pp；且保持接近原 CLIP 的零样本性能。

**⚠️ 局限性**

仍无法充分处理非视觉否定（如“not authentic”），需进一步整合常识知识以覆盖这类语义。

---

## 139. Benchmarking Distilled Language Models: Performance and Efficiency in Resource-Constrained Settings

**arXiv ID:** 2602.20164 | [PDF](https://arxiv.org/pdf/2602.20164v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 140. Computing a Characteristic Orientation for Rotation-Independent Image Analysis

**arXiv ID:** 2602.20930 | [PDF](https://arxiv.org/pdf/2602.20930v1)

**作者:** Cristian Valero-Abundio `[一作]` (Universitat Jaume I), Marina Martínez García `[通讯]` (Universitat Jaume I)

**通讯引用:** 508 | [OpenAlex ID](https://openalex.org/A5026032450)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种基于全局像素强度方向估计的预处理方法GID，使输入图像对旋转具有鲁棒性

**💡 创新点**

创新点在于直接对图像进行旋转对齐而非提取不变描述子，兼容任意卷积网络

**🔧 技术方法**

采用全局角度计算（基于像素强度加权的角度累加）和插值旋转

**📊 数据集**

在MNIST/RotMNIST和CIFAR-10数据集上评估

**📈 对比分析**

与多种旋转不变网络相比，GID在RotMNIST上达96.32%准确率，略优于RIC-CNN；在CIFAR-10上亦提升性能，保持稳定

**⚠️ 局限性**

局限在背景干扰大、对象未中心化或多目标时估计角度不稳定

---

## 141. Hybrid Fusion: One-Minute Efficient Training for Zero-Shot Cross-Domain Image Fusion

**arXiv ID:** 2602.20851 | [PDF](https://arxiv.org/pdf/2602.20851v1)

**作者:** Ran Zhang `[一作]` (Hefei University of Technology), Liu Liu `[通讯]` (Hefei University of Technology)

**通讯引用:** 140795 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种混合式图像融合框架：先用轻量级 U‑Net 生成动态引导权重图，再将其投射到固定的拉普拉斯金字塔融合核中，从而在全分辨率上完成融合。

**💡 创新点**

核心创新在于将策略学习（权重分配）与像素合成完全解耦：训练时只学习引导图，推理时直接使用经典拉普拉斯金字塔实现融合，消除了训练‑推理差距、显著提升效率、并提供物理回退机制，保证了融合结果的真实性与可靠性。

**🔧 技术方法**

技术手段包括：轻量级 U‑Net（四级下采样）、固定拉普拉斯金字塔融合核、无监督多项损失（强度最大化、梯度最大化、SSIM、强度一致性）、全分辨率端到端训练、YCrCb 颜色空间预处理、颜色保持与可解释性权重图。

**📊 数据集**

主要使用公开数据集：MSRS（红外‑可见融合）、M3FD、RoadScene；医学融合任务：PET‑MRI、CT‑MRI、SPECT‑MRI；并在这些数据上进行零射向迁移（Zero‑Shot）测试。

**📈 对比分析**

与 15+ 传统与深度学习融合方法（U2Fusion、IFCNN、SwinFusion、SuperFusion、CDDFuse、Text‑IF、DTPF 等）在 VIF、Q^AB/F、SSIM、EN、MI 等指标上对比，本文方法仅在 2‑10 轮训练（约 1–6 分钟）即可达到或超过 SOTA 结果；在下游目标检测（YOLOv8n）中 mAP@50 和 mAP@50‑95 均高于所有竞争者；在医学融合零射向实验中同样优于专业医学模型。

**⚠️ 局限性**

局限性：对视频时序一致性尚未显式建模；虽然引导权重可解释，但在极端模态或极端噪声下仍可能出现分配失误；全分辨率训练依赖足够显存的 GPU，尽管显存占用低于大多数基线，但在极大分辨率下仍受限；当前仅支持双模态融合，扩展到多模态或更多通道仍需研究。

---

## 142. Leveraging Causal Reasoning Method for Explaining Medical Image Segmentation Models

**arXiv ID:** 2602.20511 | [PDF](https://arxiv.org/pdf/2602.20511v1)

**作者:** Limai Jiang `[一作]` (Shenzhen Institutes of Advanced Technology), Yunpeng Cai `[通讯]` (Shenzhen Institutes of Advanced Technology)

**通讯引用:** 4458 | [OpenAlex ID](https://openalex.org/A5090194899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于因果推断的PdCR方法，用于解释医学图像分割模型的决策过程。

**💡 创新点**

创新点在于将平均处理效应（ATE）与Patch‑wise干预相结合，能够量化输入区域对分割结果的正负因果贡献，并且是模型无关的。

**🔧 技术方法**

使用了因果推断框架、平均处理效应、粗到细的Patch筛选、Dice相似度评估、自然分布采样的干预块等技术。

**📊 数据集**

实验使用了皮肤病变分割数据集HAM10000和视网膜血管分割数据集FIVES。

**📈 对比分析**

与SEG-GRAD和MiSuRe两种现有可解释方法比较，PdCR在解释准确性、贡献比例及平均 attribution 曲线等指标上均表现更好，能够给出更细粒度、更可靠的因果解释。

**⚠️ 局限性**

局限性包括：需要大量的干预计算，导致推理时间长；对细长结构（如血管）的解释可能受干预策略限制；并且仍依赖干预块的分布假设，未能充分解决背景干预对结果的影响。

---

## 143. A Lightweight Vision-Language Fusion Framework for Predicting App Ratings from User Interfaces and Metadata

**arXiv ID:** 2602.20531 | [PDF](https://arxiv.org/pdf/2602.20531v1)

**作者:** Azrin Sultana `[一作]` (American International University), Firoz Ahmed `[通讯]` (American International University)

**通讯引用:** 2137 | [OpenAlex ID](https://openalex.org/A5031780725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种轻量级的视觉-语言框架，利用应用程序的 UI 截图和结构化文本信息（如描述、类别等）来预测用户对应用的评分；

**💡 创新点**

创新点在于首次将 MobileNetV3 与 DistilBERT 的多模态特征通过门控融合（包括乘积、绝对差）并使用 Swish 激活函数进行融合，最终通过 MLP 回归头实现连续评分预测，模型参数极少，兼顾精度与可部署性；

**🔧 技术方法**

使用的技术包括 MobileNetV3（视觉特征提取）、DistilBERT（文本特征提取）、门控融合 + Swish 激活、MLP 回归头以及 Adam 优化器；

**📊 数据集**

使用的数据集为 Screen2Words，包含 22,417 张安卓应用界面截图及 112,085 条人工生成的摘要语句，数据经过专业标注并验证；

**📈 对比分析**

与不同激活函数（Swish、Mish、GoLU、GELU）和多种视觉/文本编码器的 ablation 对比后，最优配置在 MAE 0.1060、RMSE 0.1433、R² 0.8529、Pearson r 0.9251 之间取得最佳表现；

**⚠️ 局限性**

局限性包括数据集仅覆盖部分应用类别，未考虑用户评论、假评分等因素，且仅验证了一种融合策略，未来需扩展至更广泛场景并引入可解释性与实时推理优化。

---

## 144. Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning

**arXiv ID:** 2602.20197 | [PDF](https://arxiv.org/pdf/2602.20197v1)

**作者:** Zhuoxu Huang `[一作]` (Aberystwyth University), Jungong Han `[通讯]` (Tsinghua University)

**通讯引用:** 24642 | [OpenAlex ID](https://openalex.org/A5046605531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CalibRL，一种混合策略 RLVR 框架，利用专家指导实现可控探索，以提升多模态大语言模型（MLLM）的推理能力。

**💡 创新点**

创新点在于将专家数据视为分布基准而非直接模仿，结合优势权重与 LeakyReLU 的非对称激活，既保留策略熵又引导正确稀有推理路径的强化。

**🔧 技术方法**

核心技术包括 RLVR/GRPO 的分组优势估计、优势加权、LeakyReLU 异向梯度门控、对策略概率差的对数比值评估，以及可调节的平衡系数 λ。

**📊 数据集**

训练使用 ViRL39K 生成的 CoT 推理对，验证集为 933 条失败案例；评估覆盖 Geo3K、GeoQA、GeoEval、MathVerse、MathVision、MathVista、MMMU、ScienceQA 等八大基准。

**📈 对比分析**

与 GRPO、SFT+GRPO、LUFFY、RL-PLUS、DAPO 等基线比较，CalibRL 在域内平均提升 5.45%，域外提升 2.61%，在 GeoEval 等困难集上显著优于其它方法。

**⚠️ 局限性**

局限在于仍依赖稀疏可验证奖励与专家数据，超参数（α、λ）对性能影响大，对更大规模模型或非推理任务的通用性尚待验证。

---

## 145. Wireless-Fed Pinching-Antenna Systems with Horn Antennas

**arXiv ID:** 2602.21167 | [PDF](https://arxiv.org/pdf/2602.21167v1)

**作者:** Hao Feng `[一作]` (Hunan Institute of Engineering), Zhiguo Ding `[通讯]` (Nanyang Technological University)

**通讯引用:** 60036 | [OpenAlex ID](https://openalex.org/A5002904166)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种采用波导引导的无线馈入压缩天线系统，并通过全双工放大转发中继和定向喇叭天线实现覆盖扩展，随后给出了压缩天线位置、基站发射功率和中继增益的闭式最优解，目标为在满足用户端质量服务（QoS）约束的前提下最小化系统总功耗。

**💡 创新点**

创新点包括：①将传统阵列天线替换为高指向性喇叭天线以降低自干扰并提升链路增益；②利用压缩天线位置对波导衰减与自由空间损耗进行平衡，得到闭式最优位置；③对全双工放大转发链路进行联合优化，得到闭式最优基站功率和中继增益，实现全系统功耗的显著降低。

**🔧 技术方法**

采用的技术主要有：全双工放大转发（AF）中继技术、波导衰减建模、喇叭天线增益与自由空间路径损耗模型、基于SNR约束的凸优化与闭式求解。

**📊 数据集**

仿真数据集基于设定参数：频率28 GHz、波导长度30 m、高度3 m、波导衰减系数0.01、喇叭天线增益20 dBi、PA效率0.9、用户覆盖区域30 m×10 m、最小SNR需求20 dB等。

**📈 对比分析**

与两种基准方案（直接基站到用户的阵列天线传输以及固定位置的压缩天线无中继）相比，所提方案在相同SNR和基站-中继距离下，整体功耗下降约20–40%，尤其在中继位于波导内时显著优于基准2；当距离增加时，功耗提升速率更低，证明了系统的覆盖扩展优势。

**⚠️ 局限性**

局限性包括：仅考虑点对点链路，未考虑多用户干扰与波导多模传播；仿真基于理想信道和完美同步假设，实际部署需进一步验证喇叭天线在高频下的制造与匹配问题；以及对自干扰的抑制依赖于喇叭天线的指向性，复杂环境中仍需额外的自干扰抑制技术。

---

## 146. Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems

**arXiv ID:** 2602.20379 | [PDF](https://arxiv.org/pdf/2602.20379v1)

**作者:** Mukul Chhabra `[一作]` (Dell Technologies), Arush Verma `[通讯]` (Dell Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向企业多轮检索增强生成（RAG）系统的案例感知LLM‑as‑a‑Judge评估框架，能在每个对话轮次上通过八项运营相关指标进行分解评分，并通过严重性加权聚合实现批量评估、回归测试与生产监控；

**💡 创新点**

将案例元数据、对话历史与检索证据统一为评判上下文，设计了八个互相独立的评估维度（检索正确性、上下文充足性、幻觉/扎根准确性、答案有用性、答案类型适配、标识符完整性、问题识别、流程对齐），并引入严重性感知分数与加权聚合，区别于传统单轮、参考无关的评估方法；

**🔧 技术方法**

采用LLM‑as‑a‑Judge范式，利用确定性提示生成严格JSON输出，使用GPT‑4 (Azure OpenAI) 进行评判，配合JSON schema校验、加权求和以及Wilcoxon检验等统计方法来评估结果；

**📊 数据集**

使用两组匿名企业支持案例集（短查询237条、长查询232条），每条案例包含多轮对话、案例主题、描述、检索文档和模型回答；

**📈 对比分析**

与传统代理指标（如RAGAS的可信度与相关度）以及启发式检查进行对比，评估两种指令微调模型（Llama‑3.3‑70B 与 GPT‑oss‑120B）。在短查询中两模型差异不显著，而在长查询中GPT‑oss在加权聚合得分上显著高于Llama（p=0.0011），显示框架能揭示企业关键差异；

**⚠️ 局限性**

依赖高质量的案例字段和检索上下文，严重度阈值与权重需根据不同组织进行调整；评判结果受LLM鲁棒性和提示设计影响，仍可能与人工标注存在偏差。

---

## 147. Skullptor: High Fidelity 3D Head Reconstruction in Seconds with Multi-View Normal Prediction

**arXiv ID:** 2602.21100 | [PDF](https://arxiv.org/pdf/2602.21100v1)

**作者:** Noé Artru `[一作]` (Ubisoft La Forge), Abdallah Dib `[通讯]` (Ubisoft La Forge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用稀疏多视角图片，结合深度学习预测几何一致的表面法线，并在逆渲染优化中利用这些法线恢复高频细节的三维头部网格；

**💡 创新点**

提出跨视角注意力的多视角法线预测模型和基于逆渲染的自适应重拓扑优化框架，使得仅用少于10个摄像头即可达到与密集摄影测量相当的精度；

**🔧 技术方法**

使用扩展的DAViD Transformer、视角注意力机制、相机位姿嵌入、差分渲染与自适应重拓扑；

**📊 数据集**

在Triplegangers、NPHM与Multiface三大公开数据集上进行训练与评估；

**📈 对比分析**

与单视图基线(Sapiens、DAViD)、摄影测量(Meshroom)以及高频细节恢复方法(2DGS、SuGaR)比较，结果显示在仅10个视角下，平均深度误差约为2–3 mm，角度误差约6°，速度比Meshroom快10倍以上；

**⚠️ 局限性**

方法依赖受控光照与同步摄像头，对强视角反射、噪声或面部道具敏感，且目前仅恢复形状，未完成完整外观与光照重建。

---

## 148. GA-Field: Geometry-Aware Vehicle Aerodynamic Field Prediction

**arXiv ID:** 2602.20609 | [PDF](https://arxiv.org/pdf/2602.20609v1)

**作者:** Zhenhua Zheng `[一作]` (University of Chinese Academy of Sciences), Zhiyong Liu `[通讯]` (Institute of Automation)

**通讯引用:** 3222 | [OpenAlex ID](https://openalex.org/A5022647005)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 GA-Field 网络，能够快速且高精度地从车辆几何和流动条件预测表面压强、壁面剪切应力及三维速度场。

**💡 创新点**

通过在网络多阶段持续注入全局几何嵌入和引入粗细分层细化机制，实现了长程几何感知与局部细节恢复的双重提升。

**🔧 技术方法**

使用基于 U‑Net 的点云分层结构、分组向量注意力、网格池化、FiLM 调制的全局几何注入，以及粗细分层残差细化模块。

**📊 数据集**

在 ShapeNet‑Car 与大规模高精度 CFD 数据集 DrivAerNet++ 上进行训练与评估。

**📈 对比分析**

与 RegDGCNN、Transolver、FigConvNet、TripNet、AdaField、SpiderSolver 等最新方法对比，GA‑Field 在表面压强、壁面剪切、三维速度场以及多种误差指标上均取得 SOTA，且在不同车辆类别的 OOD 泛化上表现优于竞争者。

**⚠️ 局限性**

缺乏显式的物理约束，且对极少量训练样本或极端几何形状的鲁棒性尚未得到充分验证。

---

## 149. HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning

**arXiv ID:** 2602.21157 | [PDF](https://arxiv.org/pdf/2602.21157v1)

**作者:** Quanxin Shou `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 30148 | [OpenAlex ID](https://openalex.org/A5043464306)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的 Vision‑Language‑Action (VLA) 模型，利用 Embodied Multimodal Chain‑of‑Thought (EM‑CoT) 进行多步文本推理、视觉子目标预测和动作决策，实现长时序、跨环境的机器人操作。

**💡 创新点**

核心创新：
- 采用 Mixture‑of‑Transformers (MoT) 架构，将文本推理、视觉预见和动作预测拆分为三个专家，保持各自自回归/扩散生成能力；
- 自动化 EM‑CoT 数据合成管道：将原始轨迹转化为动作原语、文本推理和视觉子目标，实现可扩展的多模态监督；
- 双阶段训练策略：先在 VQA、视觉生成、动作预测三大数据源上做通用预训练，再在 EM‑CoT 任务上微调，兼顾一般知识与专门推理。

**🔧 技术方法**

技术手段：Mixture‑of‑Transformers、共享自注意力、特殊控制 token、双向/因果注意力掩码、ViT+VAE 视觉编码、VAE 生成、线性动作投影、Qwen2.5‑1.5B LLM 初始化、Flex Attention、Qwen3‑VL 大模型标注。

**📊 数据集**

使用的数据集：VQA（LLaVA‑NEXT）、机器人轨迹（OXE）、ego‑centric 视频（SSv2）、模拟与真实机器人演示（RoboTwin 2.0 + 320 条真实演示），以及自动生成的 EM‑CoT 数据。

**📈 对比分析**

对比方法：π₀、RDT、Diffusion Policy 等基线；在 RoboTwin 2.0 仿真中平均成功率 80.5%（Easy）/26.4%（Hard），比 π₀ 提升 34.1%/10.1%；在真实机器人上四项长时序任务，均显著高于 π₀、π₀.5，在视觉干扰、光照、背景变化和新对象场景中保持鲁棒性。

**⚠️ 局限性**

局限性：
- 对于极端高随机化或全新对象仍有性能下降，需进一步提升视觉推理泛化；
- 需要大量预训练数据和计算资源，尤其是三阶段数据合成与双阶段训练；
- 模型参数约 4.5B，推理时延相对较大，限制了实时部署。

---

## 150. The Careless Coupon Collector's Problem

**arXiv ID:** 2602.20705 | [PDF](https://arxiv.org/pdf/2602.20705v1)

**作者:** Emilio Cruciani `[一作]` (European University of Rome), Aditi Dudeja `[通讯]` (University of Salzburg)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文提出了“粗心优惠券收集者问题”（CCCP），并研究其完成时间与系统动态。

**💡 创新点**

创新点在于发现并严格证明了 CCCP 的代谢稳定相、不同阶段的收集时间阶梯化，并给出了 O(n²) 的求期望时间算法。

**🔧 技术方法**

主要技术包括马尔可夫链建模、收集概率递推、负相关性与 Chernoff 边界、均值场近似、以及对下 Hessenberg 矩阵的专用高斯消元。

**📊 数据集**

论文不使用实际数据集，全部以理论推导和实验模拟为依据。

**📈 对比分析**

与经典优惠券收集模型相比，CCCP 在 p≈0 时保持 Θ(n ln n) 的时间；当 p 增大到 O(1/n) 或更高时，收集时间呈指数级增长，算法可在 O(n²) 时间内精确计算。

**⚠️ 局限性**

主要局限在于逃逸时间的上界相对保守，导致与下界间仍有指数级误差；同时，均值场假设不严格成立，未来需进一步精细化概率分析。

---

## 151. Deep unfolding of MCMC kernels: scalable, modular & explainable GANs for high-dimensional posterior sampling

**arXiv ID:** 2602.20758 | [PDF](https://arxiv.org/pdf/2602.20758v1)

**作者:** Jonathan Spence `[一作]` (Maxwell Institute for Mathematical Sciences), Marcelo Pereyra `[通讯]` (School of Mathematical and Computer Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种将马尔可夫链蒙特卡洛（MCMC）采样器展开为深度可展开网络（unfolded network）的框架，利用可展开的链条直接学习后验分布的采样器，解决传统MCMC计算量大且缺乏模块化的问题；

**💡 创新点**

创新点在于：①将任意MCMC迁移为可展开的可训练网络，实现参数在推断时可动态设定，保持对似然的天然嵌入；②设计了基于条件Wasserstein GAN的正则化训练方案，加入均值一致性与样本多样性约束，并可结合感知损失提升图像质量；③通过实验展示了该框架在不同逆问题（去卷积、射电干涉成像）上均能取得优于零射影MCMC与纯条件GAN的性能。

**🔧 技术方法**

使用的技术包括：深度可展开MCMC（如拆分Gibbs、Langevin、LATINO），条件Wasserstein GAN（with gradient penalty），数据一致性与方差奖励正则化，感知损失（LPIPS）以及可展开的超参数学习（LoRA、γ、t）。

**📊 数据集**

实验数据集主要有MNIST（去卷积）和PROBES（射电干涉成像），并在后者使用模拟的MeerKAT 4小时 uv 覆盖来生成测量。

**📈 对比分析**

与零射影MCMC（VAE‑SGS、LATINO）、条件GAN（RCGAN、RIGAN）以及零射影SBM（IRIS）比较，Unfolded MCMC在PSNR、LPIPS、SW、W2_latent、CMMD等指标上表现最优或相近，同时采样速度仅为传统GAN的数倍，且提供更丰富的后验不确定性估计。

**⚠️ 局限性**

局限性包括：①需要在有真实标签的数据上进行离线训练，难以直接用于无监督或自监督场景；②模型规模和训练成本相对较高，尤其在展开层数增多时；③对训练分布之外的前向模型仍需进一步鲁棒性验证，超参数调节相对复杂。

---

## 152. Beyond the Star Rating: A Scalable Framework for Aspect-Based Sentiment Analysis Using LLMs and Text Classification

**arXiv ID:** 2602.21082 | [PDF](https://arxiv.org/pdf/2602.21082v1)

**作者:** Vishal Patil `[一作]` (University of Southern California), Mayank Kejriwal `[通讯]` (Information Sciences Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究结合ChatGPT进行方面识别，并使用本地机器学习模型对数百万条餐厅评论进行基于方面的情感分析，随后通过回归分析评估这些方面对整体评分的解释力度。

**💡 创新点**

创新点在于提出一种可扩展、低成本的混合框架：先用LLM快速提取关键方面，再用高效的文本分类器完成情感标注，实现大规模评论的自动化ABSA。

**🔧 技术方法**

主要技术包括ChatGPT-4o/4o mini用于方面抽取，fastText嵌入与逻辑回归（单阶段与两阶段分类器）进行情感预测，SMOTE平衡样本，TF‑IDF和LDA用于特征探索，以及线性回归分析评估方面与整体评分的关系。

**📊 数据集**

使用了Yelp公开数据集，包含约4.7 百万条餐厅评论（2005‑2022年），共52 286家餐厅、144万用户，涵盖17个美国州与一加拿大省。

**📈 对比分析**

通过Fleiss κ、Pearson相关、McNemar检验等方法比较两种分类器，单阶段逻辑回归+fastText在准确率、F1得分上显著优于两阶段模型；情感预测用于回归时，R²>0.81，food quality、service等方面对整体评分的解释度最高。

**⚠️ 局限性**

局限性包括仅关注六个LLM识别的方面，可能遗漏其他重要维度；训练样本量相对有限且地域分布不均；两阶段模型对中性情感识别效果差；LLM在成本、可解释性和对大规模推理的适用性仍有挑战。

---

## 153. SynthRender and IRIS: Open-Source Framework and Dataset for Bidirectional Sim-Real Transfer in Industrial Object Perception

**arXiv ID:** 2602.21141 | [PDF](https://arxiv.org/pdf/2602.21141v1)

**作者:** Jose Moises Araya-Martinez `[一作]` (Technical University Berlin), Jörg Krüger `[通讯]` (Technical University Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SynthRender框架用于生成带有指导域随机化的工业级合成图像，并发布了可用于双向 sim‑real 评估的IRIS数据集

**💡 创新点**

创新点在于：①低开销的 2D‑to‑3D 资产生成方法（3DGS、TRELLIS、MeshyAI）与 CAD 对比；②基于 BlenderProc 的物理驱动、光照指数采样与随机 PBR 材质的 Guided Domain Randomization；③针对工业场景的高效数据生成指南；④在机器人、汽车和 IRIS 三大基准上实现 99.1%@50 mAP，刷新记录

**🔧 技术方法**

技术包括：BlenderProc + Cycles 渲染、物理仿真、光照指数采样、RGB 颜色随机化、摄像机内参扰动、随机 PBR 材质、低开销 2D‑to‑3D 重建（3DGS、TRELLIS、MeshyAI）以及几种现成的 3D 资产生成管线

**📊 数据集**

使用的数据集有：公开机器人检测数据集、汽车检测基准、以及自建的 IRIS 数据集（32 类、508 张 RGB‑D 实景、约 20k 标注、以及 8k 合成图像）

**📈 对比分析**

通过与 Yolov8、Yolov11、DEIM 等现有检测器的对比，以及与 Horváth、Zhu 等方法在同等训练预算下的对比，SynthRender 在机器人、汽车和 IRIS 上分别达到 99.1%@50、98.3%@50 和 95.3%@50，明显优于此前工作，展示了强大的跨模型、跨数据集泛化能力

**⚠️ 局限性**

局限性包括：对 CAD 或高质量 3D 资产的依赖仍显重要；生成的合成数据在极高反射或复杂纹理物体上仍有小幅性能下降；目前仅针对检测任务，未深入探讨姿态估计、分割等；且实验环境仍主要基于少量真实样本的少样本微调，实际大规模部署时可能需进一步验证

---

## 154. When LLMs Enter Everyday Feminism on Chinese Social Media: Opportunities and Risks for Women's Empowerment

**arXiv ID:** 2602.20876 | [PDF](https://arxiv.org/pdf/2602.20876v1)

**作者:** Runhua Zhang `[一作]` (Hong Kong University of Science and Technology), Xiaojuan Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 4983 | [OpenAlex ID](https://openalex.org/A5026376235)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统分析了红人（RedNote）平台上围绕女性成长主题的DeepSeek（LLM）生成内容，探讨其对日常数字女权主义的机遇与风险。

**💡 创新点**

创新点在于首次结合Kabeer框架与女权批判话语分析，对LLM在中国社交媒体日常女权空间中的表现进行定性评估，并揭示其强调自我优化而非结构变革的倾向。

**🔧 技术方法**

研究采用了内容分析、女权批判话语分析（FCDA）和多位研究者编码等定性方法，对贴文、LLM回应及评论进行系统编码。

**📊 数据集**

使用的数据集包括430条与女性成长相关的DeepSeek互动贴文、139条DeepSeek生成回复以及3211条一层评论，全部来自RedNote平台。

**📈 对比分析**

本研究未进行对照实验或性能量化评估，而是通过定性编码展示LLM输出的主题、路径和用户接受度，未给出数值指标。

**⚠️ 局限性**

局限性包括：数据抓取不完整且仅依赖公开截图，缺乏完整对话记录；仅分析一级评论，未涉及多层互动；未进行用户访谈；未深入探讨DeepSeek内部机制与训练数据的影响。

---

## 155. Regret-Guided Search Control for Efficient Learning in AlphaZero

**arXiv ID:** 2602.20809 | [PDF](https://arxiv.org/pdf/2602.20809v1)

**作者:** Yun-Jui Tsai `[一作]` (National Yang Ming Chiao Tung University), Ti-Rong Wu `[通讯]` (Academia Sinica)

**通讯引用:** 1195 | [OpenAlex ID](https://openalex.org/A5101403833)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于 regret（后悔值）引导的搜索控制框架 RGSC，在 AlphaZero 的自我对弈过程中优先重放高后悔状态，以提高学习效率。

**💡 创新点**

创新点在于：①使用 regret 网络（包含排名网络和价值网络）自动识别并排序后悔状态；②构造优先级后悔缓冲区（PRB），通过指数移动平均动态更新后悔值；③采用基于后悔值的 softmax 采样策略，模仿人类重复复盘关键错误的学习方式。

**🔧 技术方法**

技术手段：AlphaZero 基础框架、蒙特卡罗树搜索（MCTS）、神经网络多头（策略、价值、后悔网络）、后悔排名损失（rank‑based objective）、优先级经验回放、指数移动平均（EMA）更新、温度调节的 softmax 采样。

**📊 数据集**

数据集与实验环境：三种棋类游戏的自我对弈生成数据，分别为 9x9 Go、10x10 Othello 和 11x11 Hex，训练迭代约 300 次，使用 3‑block 残差网络。

**📈 对比分析**

对比方法：AlphaZero（无搜索控制）和 Go‑Exploit（均匀采样的搜索控制）。在所有三种游戏中，RGSC 的 Elo 分数平均比 AlphaZero 高 77 点、比 Go‑Exploit 高 89 点；在已训练好的 9x9 Go 模型上，RGSC 将与 KataGo 的胜率从 69.3% 提升至 78.2%，而 AlphaZero 与 Go‑Exploit 均无显著提升。

**⚠️ 局限性**

局限性：①仅在离散、确定性的棋类任务上验证，尚未证明在更复杂或连续控制环境的泛化能力；②需要额外的后悔网络头，略微增加模型尺寸与训练开销；③后悔排名依赖于先验经验，可能在极稀疏奖励或高度随机环境中效果不佳。

---

## 156. Evaluating Proactive Risk Awareness of Large Language Models

**arXiv ID:** 2602.20976 | [PDF](https://arxiv.org/pdf/2602.20976v1)

**作者:** Xuan Luo `[一作]` (Harbin Institute of Technology), Ruifeng Xu `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了主动风险意识评估框架，并在环境领域构建了 Butterfly 数据集，用于检测 LLM 在无害查询下的潜在生态风险识别与预警能力。

**💡 创新点**

创新点在于：①将主动安全评估迁移至长期系统性环境风险；②设计双语、多模态（文本+图像）的基准数据；③引入四个可量化指标（ProR、HAR、GR、BR）和对应的标签体系，全面评估模型的预警、规避与盲点表现。

**🔧 技术方法**

主要技术手段包括：基于 GPT‑5 的自动标签与人工核对、对多模型（GPT‑5、Gemini、Doubao、Qwen、Deepseek）在不同回答长度、语言和模态条件下的实验设计，以及对系统提示干预的对比分析。

**📊 数据集**

使用了 Butterfly 数据集（1094 个无害查询、70 种法规级有害行为，包含 285 张受保护物种图像）以及 1,068 条文本查询，涵盖 26 类受保护物种子集。

**📈 对比分析**

通过在全长/短长、英中两语、文本/图像三种条件下评估 5 大 LLM，发现 GPT‑5 在 ProR 上遥遥领先，短回答显著降低 ProR 并提升 HAR/BR，系统提示能显著恢复 ProR 并压低 HAR/BR；总体来看，当前 LLM 在主动环境安全方面仍显脆弱。

**⚠️ 局限性**

主要局限包括：实验基于公开 API 可能与官方接口存在差异；标签体系聚焦已定义的安全行为，可能低估其它有益提醒；仅覆盖英中两语与特定场景，难以推广到多语种与跨文化环境。

---

## 157. An LLM-driven Scenario Generation Pipeline Using an Extended Scenic DSL for Autonomous Driving Safety Validation

**arXiv ID:** 2602.20644 | [PDF](https://arxiv.org/pdf/2602.20644v1)

**作者:** Fida Khandaker Safa `[一作]` (Macquarie University), Xi Zheng `[通讯]` (Macquarie University)

**通讯引用:** 6115 | [OpenAlex ID](https://openalex.org/A5081182489)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用 GPT‑4o mini 对多模态交通事故报告（文字摘要与手绘示意图）进行语义抽取，生成中间语言 Extended Scenic DSL，再通过模板化的 Scenic 语法合成可执行的仿真脚本，实现从真实事故到 ADS 测试场景的自动化转换。

**💡 创新点**

① 引入中间 DSL 层，将高层语义与低层渲染解耦，显著降低 LLM 产生的语法错误与幻觉；② 通过概率化 DSL 让单一事故能够生成千变万化的场景实例，提升测试覆盖率；③ 在抽取后加入自检与验证步骤，确保每个字段都有原始报告的证据支撑。

**🔧 技术方法**

技术包括：多模态 GPT‑4o mini 结构化抽取与验证；基于 YAML 的 Extended Scenic DSL；Jinja2 模板化 Scenic 生成；CARLA 仿真环境配合 Autoware 驾驶栈；自动化规则检测器（对 11 条加州交通法规进行运行时检查）。

**📊 数据集**

主要数据集是美国 NHTSA CIREN 数据库（约 2,538 起事故报告，实验抽样 100 起；其中 50 起用于对比 Golden Oracle）。

**📈 对比分析**

评估维度：① 语义抽取准确率（与人类 Golden Oracle 对比）达 99%（环境 100%、道路 100%、轨迹 97‑98%）；② 场景重现可信度（人类评估）高于 80% 匹配；③ 在 CARLA+Autoware 中生成 2,000 个随机变体，所有变体均成功触发预期的交通规则违规，验证了测试效果。相比传统的直接文本→脚本或规则驱动的 Target 框架，本文的多模态抽取+概率 DSL 在准确性与覆盖率上都有显著提升。

**⚠️ 局限性**

局限性：① 仅使用单一 LLM（GPT‑4o mini），对模型的鲁棒性与成本仍有待进一步验证；② 验证与抽取主要基于 NHTSA 数据，缺乏对其他地区或更复杂事故类型的评估；③ 对异常图像质量或极端文字描述的鲁棒性未作系统测试；④ 生成的场景仍受限于可用地图资产与 Scenic 模板，某些罕见道路拓扑可能无法完整重现。

---

## 158. Robot Local Planner: A Periodic Sampling-Based Motion Planner with Minimal Waypoints for Home Environments

**arXiv ID:** 2602.20645 | [PDF](https://arxiv.org/pdf/2602.20645v1)

**作者:** Keisuke Takeshita `[一作]` (Toyota Motor Corporation), Takashi Yamamoto `[通讯]` (Aichi Institute of Technology)

**通讯引用:** 30062 | [OpenAlex ID](https://openalex.org/A5078140561)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种周期性采样的全身轨迹规划方法RLP，能够在家居环境中快速、低计算量地完成安全操纵任务。

**💡 创新点**

创新点在于结合周期性规划、最小关键点（直线+三点）采样、鲁棒基位IK以及基于家居布局的两种轨迹类型，实现了速度、最优性、鲁棒性与安全性的统一。

**🔧 技术方法**

采用采样式规划（RRT-Connect启发式）、鲁棒IK求解、时间优化、多线程并行、软约束评估排序和碰撞验证等技术。

**📊 数据集**

使用BEHAVIOR数据集、WRS 2020整理任务以及MπNets训练集（约1.09M轨迹）进行实验评估。

**📈 对比分析**

与AIT*、CBiRRT2以及MπNets等方法对比，RLP在运动完成时间、计划延迟、运动时长、鲁棒性和碰撞率等指标上表现更佳，在家居场景和整理任务中实现最快速度和高成功率。

**⚠️ 局限性**

目前仅在静态或半动态家居环境中验证，动态环境下的适应性与高自由度机器人的鲁棒性提升仍需进一步验证。

---

## 159. What Matters for Simulation to Online Reinforcement Learning on Real Robots

**arXiv ID:** 2602.20220 | [PDF](https://arxiv.org/pdf/2602.20220v1)

**作者:** Yarden As `[一作]` (ETH Zurich), Markus Wulfmeier `[通讯]` (Google DeepMind)

**通讯引用:** 11814 | [OpenAlex ID](https://openalex.org/A5105920879)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一个可开放源代码的训练管线，支持在模拟器中预训练任意机器人（MuJoCo Playground）并无缝切换到真实机器人上进行在线强化学习。

**💡 创新点**

创新点包括：① 在三种不同机器人平台（Franka Emika Panda、Unitree G1/G1、Race Car）上验证了“sim‑to‑online”学习的可行性；② 通过实验系统性评估了三种稳健性技巧——数据保留、warm‑start、非对称更新，证明它们能显著提升迁移过程中的稳定性与样本效率；③ 为视景任务提供了基于DrQ的端到端视觉控制实现。

**🔧 技术方法**

主要技术：Soft Actor‑Critic (SAC) 结合 BRO 关键字网络、DrQ 视觉模块、离线与在线数据混合采样、Polyak 平滑目标网络、异步更新 (actor 每 M 步更新一次) 等。

**📊 数据集**

使用的数据集：① 先前在MuJoCo/Brax 中生成的仿真数据集（随机化摄像头、光照、动力学参数等）；② 在线收集的真实机器人交互数据（从重置到终止），并在后续实验中对其进行保留或 warm‑start。

**📈 对比分析**

比较方法：在同一任务上对比仅在线训练、保留仿真数据、warm‑start、以及非对称更新等不同设置，使用平均累计奖励 (undiscounted return) 和收敛速度（所需环境步数）评估性能。结果显示，数据保留与非对称更新能将收敛速度提升 2–3 倍，并显著提升最终奖励；warm‑start 对于复杂任务尤为关键。

**⚠️ 局限性**

限制：① 仍依赖人工重置与安全约束，未实现完全自主学习；② 只在少数几台机器人与任务上验证，未覆盖更广泛的动力学或感知场景；③ 仅使用离线预训练 + 线上微调的方式，未探索更复杂的多任务或跨任务数据复用策略。

---

## 160. DMCD: Semantic-Statistical Framework for Causal Discovery

**arXiv ID:** 2602.20333 | [PDF](https://arxiv.org/pdf/2602.20333v1)

**作者:** Samarth KaPatel `[一作]` (Causify AI), Paul Smith `[通讯]` (Causify AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种两阶段的因果结构学习框架 DMCD，先用大语言模型根据变量元数据草拟稀疏有向无环图，再通过统计条件独立性检验对草图进行验证和修订。

**💡 创新点**

创新点在于将语义先验与经验验证明确分离，利用大语言模型的知识推理生成初始因果假设，而非仅在统计搜索空间内进行约束，显著提升了召回率和 F1 分数。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑5.2）用于语义草拟；条件独立性检验（Pearson 线性偏相关、Chi‑squared、XGBoost 残差检验）用于统计验证；p 值多重校正为 q 值；基于错误率、SHD、精确率、召回率、F1 等指标进行评估。

**📊 数据集**

使用了三类真实世界带有丰富元数据的基准：工业工程领域的 Tennessee Eastman Process；环境监测领域的 Fluxnet2015；IT 系统监测领域的四组监控数据集（Message, Ingestion, Web, Antivirus）。

**📈 对比分析**

在所有基准上，DMCD 与传统约束、分数、函数及连续优化方法相比，保持了竞争或领先的性能，特别是在召回率和 F1 方面显著优于基线，且在工业、环境和 IT 监测场景均表现稳健。

**⚠️ 局限性**

局限性包括：对元数据质量高度依赖，元数据缺失或含糊时性能下降；推理阶段仍带有非确定性，导致结果波动；以及统计验证阶段受样本量、噪声和潜在隐含变量等数据质量因素限制。

---

## 161. Probing Dec-POMDP Reasoning in Cooperative MARL

**arXiv ID:** 2602.20804 | [PDF](https://arxiv.org/pdf/2602.20804v1)

**作者:** Kale-ab Tessera `[一作]` (University of Edinburgh), Amos Storkey `[通讯]` (University of Edinburgh)

**通讯引用:** 13687 | [OpenAlex ID](https://openalex.org/A5007901825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套信息论诊断框架，用来评估多智能体强化学习（MARL）代理是否真正体现了Dec-POMDP的推理需求；在37个场景（MPE、SMAX、Overcooked、Hanabi、MaBrax等）上系统审计，揭示历史记忆普遍存在但对性能提升不显著，隐含状态与团队信息的难度可分离，同步与时序协调在不同基准中表现不同；最终指出只有MPE满足所有诊断条件，表明大部分现有基准未充分检验Dec-POMDP推理。

**💡 创新点**

创新点在于：①将历史相关性、私有信息流、同步动作耦合、定向动作信息等四个信息论指标整合为可操作的诊断；②使用基于置换的零基线校正消除估计偏差；③系统性地对多种主流MARL基准进行诊断，揭示其对Dec-POMDP推理的真实要求；④提供开源工具供研究者自检。

**🔧 技术方法**

技术主要包括：信息论度量（互信息、条件互信息、定向信息），kNN/KSG估计器，Wilcoxon检验，置换零基线，使用IPPO和MAPPO两种训练范式的FF与RNN策略进行实验。

**📊 数据集**

数据集为七个公开MARL基准套件中的37个具体任务：MPE、SMAX V1/V2、Overcooked V1/V2、Hanabi、MaBrax。

**📈 对比分析**

对比方法：在每个任务上使用10个随机种子训练IPPO/MAPPO，评估平均回报并计算最小最大归一化四分位均值（IQM）及95%置信区间。诊断指标的统计显著性通过Wilcoxon检验和置换检验判断；结果显示大多数基准在历史记忆、同步/时序协调上仅部分满足阈值，说明传统回报评估可能掩盖了缺乏Dec-POMDP推理的情况。

**⚠️ 局限性**

局限性包括：诊断依赖于已收敛策略，无法反映环境最坏/最好情况；信息论估计受样本量、动作空间大小影响，存在偏差；置换零基线虽校正了偏差但对极端依赖情况敏感；仅评估IPPO/MAPPO的FF/RNN，其他更强或更弱算法可能得到不同诊断结果。

---

## 162. Hierarchic-EEG2Text: Assessing EEG-To-Text Decoding across Hierarchical Abstraction Levels

**arXiv ID:** 2602.20932 | [PDF](https://arxiv.org/pdf/2602.20932v1)

**作者:** Anupam Sharma `[一作]` (Indian Institute of Technology Gandhinagar), Krishna Miyapuram `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 717 | [OpenAlex ID](https://openalex.org/A5049413065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究EEG信号在不同层级抽象下的文字解码能力，并构建了最大的EEG‑to‑Text的分层元学习评估框架；

**💡 创新点**

引入基于WordNet的语义层级抽样，采用分层元学习（episodic）评估，揭示EEG对抽象层次的敏感性；

**🔧 技术方法**

使用Meta‑Learning（MAML、Proto‑MAML）、预训练EEG模型CBraMod、传统EEGNet、NICE‑EEG，以及自监督预训练；

**📊 数据集**

PEERS数据集（约931538个EEG样本、1610个词标签、264名受试者）与WordNet构建的层级图；

**📈 对比分析**

对比了非分层（一次性分类）与分层元学习的性能，发现非分层几乎失败，分层元学习虽仍接近随机，但在更高抽象层次（如“PERSON”）表现提升，预训练模型优于复杂元学习；

**⚠️ 局限性**

受限于EEG信号噪声高、任务复杂导致整体准确率接近随机，缺乏多模态辅助，且不同分层抽样导致高方差，需进一步研究多模态融合与更鲁棒的模型。

---

## 163. RISK: Efficiently processing rich spatial-keyword queries on encrypted geo-textual data

**arXiv ID:** 2602.20952 | [PDF](https://arxiv.org/pdf/2602.20952v1)

**作者:** Zhen Lv `[一作]` (Xi'an International Studies University), Yingfan Liu `[通讯]` (Xidian University)

**通讯引用:** 200 | [OpenAlex ID](https://openalex.org/A5101975013)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种统一的安全富空间关键词查询框架 RISK，可在加密地理文本数据上同时支持范围查询和 k 最近邻查询。

**💡 创新点**

关键创新在于设计了 kNN 四叉树 kQ-tree 及其加密版 SkQ-tree，兼容两类查询并使用标准加密（有钥哈希+对称加密）实现 IND‑CKA2 安全。

**🔧 技术方法**

使用了有钥哈希函数、对称加密、标准四叉树与 kNN 近邻结构，并结合索引构造、陷阱门生成与多阶段查询策略。

**📊 数据集**

在 Twitter、NewYork、Paris 三个真实社交/地图数据集以及 Gaussian 合成数据集上进行实验。

**📈 对比分析**

与现有方案（RASK、PBKQ 等）比较，RISK 在存储、查询响应时间和陷阱门开销上均优于 SOTA，范围查询提升 0.5–4 量级，kNN 查询提升 4 量级。

**⚠️ 局限性**

主要局限是仅支持小规模插入/删除；大规模更新效率低；前向安全、Top‑k 查询等功能尚未完善。

---

## 164. From Isolation to Integration: Building an Adaptive Expert Forest for Pre-Trained Model-based Class-Incremental Learning

**arXiv ID:** 2602.20911 | [PDF](https://arxiv.org/pdf/2602.20911v1)

**作者:** Ruiqi Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yongjun Xu `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个Semantic-guided Adaptive Expert Forest（SAEF），将单独训练的任务适配器组织成语义层次化的专家森林。

**💡 创新点**

创新点在于自动将适配器按语义聚类，构建平衡二叉树并通过熵引导的自适应搜索和融合，实现有选择性、层次化的知识迁移。

**🔧 技术方法**

采用预训练视觉模型ViT、CLIP文本编码器构建语义原型，使用轻量级adapter、向量化合并、熵权重融合、k-means聚类、平衡树构建等技术。

**📊 数据集**

在CIFAR-100、ImageNet-R、ImageNet-A、ObjectNet等四大增量学习基准数据集上进行评估。

**📈 对比分析**

与多种SOTA PTM-based CIL与回放方法对比，SAEF在平均精度与最终精度均超过1.3%（最高达94.53%/90.60%），且推理速度比全集成方法提升约5×。

**⚠️ 局限性**

局限性包括对语义原型质量的依赖、聚类与树结构的超参数选择、以及在极长任务序列中仍需进一步验证其可扩展性。

---

## 165. Learning During Detection: Continual Learning for Neural OFDM Receivers via DMRS

**arXiv ID:** 2602.20361 | [PDF](https://arxiv.org/pdf/2602.20361v1)

**作者:** Mohanad Obeed `[一作]` (Huawei Technologies Canada), Ming Jian `[通讯]` (Huawei Technologies Canada)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了零开销的在线持续学习框架，利用重随机化的调制参考信号对OFDM神经接收机实现实时自适应。

**💡 创新点**

创新点在于重新设计解调导频实现同时解调与模型微调，无需额外开销或中断；提出三种导频结构和两种并行/前向重用接收机架构。

**🔧 技术方法**

采用深度神经网络（CoNet/ResNet）、在线微调、前向重用、导频掩码、Mini-batch 反向传播和延迟分析等技术。

**📊 数据集**

使用仿真生成的多样化OFDM数据集（包含不同延迟扩展、Doppler、3GPP TDL 等），训练集约 20M 样本，测试时在相同与移位分布上验证。

**📈 对比分析**

与传统 LS‑LMMSE 以及固定神经接收机对比，在线微调后 BER 接近原始性能，能够在慢速/快速/随机分布漂移下保持 BER < 2×10⁻³。

**⚠️ 局限性**

局限性包括需手动设定掩码比例与 Mini‑batch 大小等超参数，计算成本较高（双网络或反向传播暂停），仅在单用户 MISO 仿真验证，需在多用户 MIMO 或真实硬件上进一步验证。

---

## 166. CG-DMER: Hybrid Contrastive-Generative Framework for Disentangled Multimodal ECG Representation Learning

**arXiv ID:** 2602.21154 | [PDF](https://arxiv.org/pdf/2602.21154v1)

**作者:** Ziwei Niu `[一作]` (National University of Singapore), Yueming Jin `[通讯]` (National University of Singapore)

**通讯引用:** 4840 | [OpenAlex ID](https://openalex.org/A5050163233)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种 CG-DMER 框架，利用对比生成方法实现多模态心电图（ECG）与临床文本的联合表征学习。

**💡 创新点**

创新点包括：① 在 ECG 上进行时空蒙版建模，既捕获各导联间的空间依赖，又提取细粒度的时间动态；② 通过模态专属与共享编码器实现特征分离，消除模态偏差并增强跨模态对齐；③ 结合对比学习与 SigLIP 损失，统一生成与判别目标。

**🔧 技术方法**

采用 Transformer‑based 采样与分块、卷积+GELU+GroupNorm 的 token 化、对比损失、重建损失、正交约束、SigLIP 对齐等技术实现模型训练。

**📊 数据集**

预训练使用 MIMIC‑IV‑ECG（约 78 万对样本），下游评估在 PTB‑XL、CPSC2018、CSN 三个公开心电数据库上进行线性探测和零样本分类。

**📈 对比分析**

与现有 eSSL（SimCLR、MoCo 等）及多模态方法（ETP、MERL、C‑MELT）对比，在 1%‑100% 标签比例下的线性探测以及零样本分类任务中均实现了最佳 AUC，最高单一任务提升超过 0.75 点。

**⚠️ 局限性**

局限性：需要大量标注的 ECG‑文本配对数据；模型结构相对复杂，训练成本高；对不同来源或采样率的 ECG 数据可能存在迁移性能下降。

---

## 167. CREDIT: Certified Ownership Verification of Deep Neural Networks Against Model Extraction Attacks

**arXiv ID:** 2602.20419 | [PDF](https://arxiv.org/pdf/2602.20419v1)

**作者:** Bolin Shen `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 934 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出CREDIT框架，实现对模型提取攻击的可认证所有权验证；通过在嵌入层加入高斯噪声并利用互信息上界，给出可证的所有权阈值；

**💡 创新点**

首次提供理论上可证明的所有权验证方法，给出误差概率上界并通过互信息上界与Gaussian机制实现对提取模型的区分；

**🔧 技术方法**

使用互信息估计（KSG）、高斯机制（Gaussian Mechanism）与随机化差分隐私理论，构建上界与阈值；

**📊 数据集**

在CIFAR‑10、CIFAR‑100（图像分类）以及ENZYMES、PROTEINS（图数据）等数据集上进行评估；

**📈 对比分析**

与多种水印/指纹基线（EWE、Backdoor、IPGuard、UAP、RandomWM等）比较，CREDIT在保持模型效用（误差几乎为0）同时获得AUROC 100%，且在准备与验证阶段显著更高效；

**⚠️ 局限性**

对高斯噪声参数σ的选择敏感，过大会提高误报率；理论假设互信息可准确估计，实际估计误差和样本限制可能影响效果；

---

## 168. HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG

**arXiv ID:** 2602.20926 | [PDF](https://arxiv.org/pdf/2602.20926v1)

**作者:** Yuqi Huang `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 16847 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 HELP 框架，通过层层扩展 HyperNode 和逻辑路径驱动的证据定位，实现在 GraphRAG 任务中的高效且精确检索与答案生成。

**💡 创新点**

创新点在于：① 通过 HyperNode 将多条三元组链式化为统一实体，实现多跳推理路径的迭代构建；② 使用预计算的三元组-文本关联实现逻辑路径驱动的证据定位，避免高成本图遍历与 LLM 生成噪声；③ 采用混合检索策略，兼顾结构化推理与稠密检索，显著提升检索准确性与速度。

**🔧 技术方法**

核心技术包括 OpenIE 关系抽取、三元组-文本倒排索引、HyperNode 序列化与 Transformer 编码、基于欧氏距离的 beam 搜索扩展、逻辑路径证据评分与稠密检索融合、以及大规模 LLM（Llama‑3.3‑70B）生成。

**📊 数据集**

使用了自然问题（NQ）、PopQA、MuSiQue、2WikiMultiHopQA、HotpotQA、LV‑Eval 等标准单跳与多跳问答数据集进行评估。

**📈 对比分析**

与 BM25、Contriever、GTR、NV‑Embed‑v2、HippoRAG、HippoRAG2、LinearRAG、HyperGraphRAG 等基线对比，HELP 在平均 F1 上提升至 55.3%（比 HippoRAG2 高 1.3%，比 NV‑Embed‑v2 高 7%），并在 2Wiki 上实现 28.8× 的检索速度提升。

**⚠️ 局限性**

局限性包括对 OpenIE 抽取质量的依赖；当图结构不完整或噪声较多时，纯逻辑路径检索效果受限，需要混合检索做缓冲；对扩展跳数、beam 宽度等超参数仍有一定敏感性，需根据任务进行微调。

---

## 169. On Electric Vehicle Energy Demand Forecasting and the Effect of Federated Learning

**arXiv ID:** 2602.20782 | [PDF](https://arxiv.org/pdf/2602.20782v1)

**作者:** Andreas Tritsarolis `[一作]` (University of Piraeus), Yannis Theodoridis `[通讯]` (University of Piraeus)

**通讯引用:** 10508 | [OpenAlex ID](https://openalex.org/A5018268830)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对电动汽车充电站能量需求预测问题进行了系统评估，比较了统计、梯度提升树和循环神经网络在集中式与联邦式学习下的表现，并量化了能耗与碳排放。

**💡 创新点**

首次将多种主流时序预测方法与联邦学习在真实EVSE数据上并行对比，并评估其能源与隐私成本。

**🔧 技术方法**

使用了ARIMA/SARIMA/SARIMAX、XGBoost、LSTM/GRU/BiLSTM/BiGRU以及FedAvg/FedProx/FedXGB等联邦学习框架。

**📊 数据集**

数据集为四个真实EVSE数据集：Dundee、Palo Alto、Boulder 和 FEUP。

**📈 对比分析**

通过MASE、SMAPE、MAE、R^2等多项指标比较，XGBoost在集中式下最优，联邦学习在部分数据集可逼近集中性能，但整体仍略逊，能耗与碳排放在轻量化配置下显著降低。

**⚠️ 局限性**

受限于数据分布异质性导致模型泛化差、联邦学习通信开销高、仅评估12小时预测窗口、未探索更高级的聚合策略和特征。

---

## 170. ConceptRM: The Quest to Mitigate Alert Fatigue through Consensus-Based Purity-Driven Data Cleaning for Reflection Modelling

**arXiv ID:** 2602.20166 | [PDF](https://arxiv.org/pdf/2602.20166v1)

**作者:** Yongda Yu `[一作]` (Nanjing University), Xiaobin Xu `[通讯]` (Alibaba Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用少量专家标注为锚点，构建基于共识的噪声清洗框架，对工业级代码评审数据进行去噪并训练反射模型，实现在自动代码评审中拦截低质量或假警报，缓解“警报疲劳”；

**💡 创新点**

① 将模型差异从参数空间迁移到数据空间，通过对噪声“忽略”样本按比例注入形成多种扰动数据；② 采用共识（严格一致/多数投票）选择最佳模型组合并进行偏置重标；③ 以“纯度驱动拦截效能”（PIE）为目标进行多目标优化；

**🔧 技术方法**

噪声污染（noise‑doping）+共识重标 +纯度驱动协同教学（purity‑driven co‑teaching）+严格一致/多数投票等；

**📊 数据集**

① 内部工业数据：约74.9k条噪声标签，1k/1k验证/测试；② 开源GitHub跨语言数据：10种主流语言共1k条；

**📈 对比分析**

与无清洗的基线、提示工程式LLM（Qwen、Claude、DeepSeek等）对比，PIE提升至4.42（相较于Claude的1.53），FPR显著下降（从34%↓至13%），仅用1k专家注释即可比全量清洗版提升≈53%；

**⚠️ 局限性**

对专家标注质量高度依赖，标注不足或错误会导致共识偏差；方法针对代码评审场景设计，迁移到其他领域需重新验证；忽略样本被视为噪声的假设可能不适用于所有任务。

---

## 171. A Robotic Testing Platform for Pipelined Discovery of Resilient Soft Actuators

**arXiv ID:** 2602.20963 | [PDF](https://arxiv.org/pdf/2602.20963v1)

**作者:** Ang `[一作]`, Mihai Duduta `[通讯]` (University of Connecticut)

**通讯引用:** 2155 | [OpenAlex ID](https://openalex.org/A5074130221)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套基于自驱动实验室（SDL）理念的线性电介质弹性体（DEA）寿命优化流程，设计并实现了一个双通道、多功能测试机器人，系统性扫描电压、电场、频率、碳纳米管（CNT）浓度及电连接填料等参数，得到最佳材料组合后放大DEA尺寸以提升力和位移，并将优化后的DEA集成到模块化四足行走机器人中，展示了高负载和长期稳定行走能力。

**💡 创新点**

创新点包括：①首次将SDL方法应用于软体机器人执行器的端到端设计与优化；②设计的可扩展双通道测试机器人集成高压供电、激光位移、力计及电阻/电容测量，实现高通量寿命测量；③分阶段扫描电场/频率和材料参数，系统性提升DEA寿命和性能；④将优化后的DEA直接集成到四足机器人，验证其载荷能力和长周期运行。

**🔧 技术方法**

技术手段：开源 Arduino Nano 控制器 + 高压 Opto‑Coupler (HVM OR‑100) + XP Power HRC/HRL 5W/30W 变换器；激光位移传感器（Panasonic HGC‑1030）+ 受阻力计（Honeywell FSG005WNPB）+ DAQ（Analog Discovery 3）+ 步进电机切换机构；电介质材料：Elastosil P7670、P3 CNT 电子墨水；电连接填料：碳油、液态金属、PDMS‑碳黑混合物；机器人组装采用 3D 打印、金属螺丝与双面胶粘贴；数据采集与 UI 采用 Python 脚本。

**📊 数据集**

数据集：约 300-500 条 DEA 寿命与性能记录，涵盖电场（35–50 V/µm）× 频率（1–50 Hz）× CNT 浓度（1.8–3.3 mL/FA）× 电连接填料（CG、CB、LM）。每个实验点记录寿命（以位移下降至 80% 为准）、平均位移、阻止力、阻抗及电容衰减；机器人实验记录负载下的行走速度与稳定性。

**📈 对比分析**

对比方法：在默认材料/参数下与最佳组合进行寿命、位移、力学输出对比。结果显示：在 40 V/µm、1 Hz 条件下，寿命提升 22%（相对默认）/72%（相对最差）；在 45 V/µm、50 Hz 条件下，寿命提升 99%/496%。放大后的 DEA 在 1 Hz、>40 V/µm 10 h 运行中保持 5.5% 轴向应变、0.55 N/g 的阻止力。集成至四足机器人后，机体可携带 >200 g，超过本体质量的 100% 及整机驱动器质量的 700% 以上，同时保持稳定行走。

**⚠️ 局限性**

局限性：①硬件可靠性受限，光耦驱动器过热导致偶发失效；②DEA 仍为手工批量制备，生产效率低且难以实现组合设计；③数据采集通道采用 Arduino + I²C，吞吐量受限，无法支持 >100 Hz 的多通道测试；④无闭环控制，机器人行走路径不对称；⑤现有高压转换器功率不足，导致无绳电源版无法驱动多条 DEA，限制了自主行驶功能。

---

## 172. Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models

**arXiv ID:** 2602.20981 | [PDF](https://arxiv.org/pdf/2602.20981v1)

**作者:** Christian Simon `[一作]` (Sony Group Corporation), Yuki Mitsufuji `[通讯]` (Sony AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出MMHNet框架，实现从短视频片段训练到长视频音频生成的长度泛化。

**💡 创新点**

创新点在于结合非因果Mamba-2与多层次路由的层次化网络，消除Transformer对位置编码的依赖并有效对齐多模态信息。

**🔧 技术方法**

使用非因果Mamba-2、流匹配目标、CLIP+Synchformer等多模态编码器，以及层次化路由与分块/解块技术。

**📊 数据集**

在UnAV100、LongVale以及VGGSound等公开长视频数据集上进行训练与评估。

**📈 对比分析**

与LoVA、MMAudio、V-AURA、HunyuanVideo-Foley等SOTA方法对比，MMHNet在分布匹配、音质、语义一致性与同步指标上均取得更优表现，尤其在长达5分钟的音频生成上明显领先。

**⚠️ 局限性**

主要限制包括仍需大量计算资源，且在极端长时序（超过数小时）或跨域视觉场景时的泛化性尚未充分验证。

---

## 173. In-context Pre-trained Time-Series Foundation Models adapt to Unseen Tasks

**arXiv ID:** 2602.20307 | [PDF](https://arxiv.org/pdf/2602.20307v1)

**作者:** Shangqing Xu `[一作]` (Georgia Institute of Technology), B. Aditya Prakash `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5189 | [OpenAlex ID](https://openalex.org/A5061110232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在时间序列基础模型中加入 In-Context Learning（ICL）能力，使其能够在不微调的情况下适应未见任务。

**💡 创新点**

通过将原始数据重构为多任务上下文序列，显式为模型提供任务上下文，从而实现无微调的多任务适应。

**🔧 技术方法**

利用 ICL 技术、上下文构造、任务候选集和 Transformer 体系结构，构建 ICTP 预训练框架。

**📊 数据集**

使用 ETTh1、ETTm1、Exchange Rate、Weather、PEMS-Bay、METR-LA 六个公开时间序列数据集进行预训练与评估。

**📈 对比分析**

在 MOMENT、TimesFM、LPTM 三种基础模型上进行实验，并与基线无微调重构方法对比；在 96/192 步输出长度的未见任务上平均提升约 11–12%，同时保持原任务性能。

**⚠️ 局限性**

提升效果在某些模型（decoder-only）和任务（如插值）上不明显，受数据集复杂度限制；且实验仅覆盖有限任务和数据，缺乏对更广泛应用的验证。

---

## 174. Surface-based Manipulation Using Tunable Compliant Porous-Elastic Soft Sensing

**arXiv ID:** 2602.21028 | [PDF](https://arxiv.org/pdf/2602.21028v1)

**作者:** Gayatri Indukumar `[一作]` (Istituto Italiano di Tecnologia), Lucia Beccai `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

开发了一种集成感应线圈阵列的COPESS柔性表面，可通过可调三维打印Gyroid晶格实现柔性操控和局部感知；

**💡 创新点**

创新点在于将可变相对密度的3D打印晶格作为调节机械刚度与感知灵敏度的双重调制层，实现对力范围与灵敏度的共设计；

**🔧 技术方法**

采用SLA打印Elastic 50A V2树脂制成Gyroid晶格，配合铜/铁靶、4x4线圈阵列和LDC1614电感数值转换器；

**📊 数据集**

实验数据使用人工压缩测试、力传感器和冷却装置采集的力-位移和感应电感曲线；

**📈 对比分析**

通过相对密度（7、10、20）对比，显示相对密度从7到20可将刚度提升7倍、操作力范围提升9倍、灵敏度下降23倍，性能表现符合设计目标；

**⚠️ 局限性**

局限在于打印可行性受限于晶格尺寸与相对密度、材料在高密度下易聚集，且高密度晶格灵敏度低，需进一步优化材料与打印工艺。

---

## 175. Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation

**arXiv ID:** 2602.20200 | [PDF](https://arxiv.org/pdf/2602.20200v1)

**作者:** Zaijing Li `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 28968 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出OptimusVLA双记忆框架，通过全局先验记忆（GPM）和局部一致性记忆（LCM）提升机器人视觉语言动作模型的推理效率与鲁棒性。

**💡 创新点**

创新点在于①将高斯噪声替换为检索到的任务级先验，显著缩短生成路径并降低NFE；②使用自注意力+Mamba模块对最近动作序列建模，提供进度感知与时序一致性约束。

**🔧 技术方法**

采用流匹配（Conditional Flow Matching）生成策略，结合Vision‑Language backbone、先验检索头、记忆库、先验采样器、动态噪声与NFE调节、Mamba自注意力、InfoNCE训练等技术。

**📊 数据集**

使用LIBERO、CALVIN、RoboTwin 2.0（Hard）模拟数据集，以及GALAXEA R1 Lite实测机器人数据。

**📈 对比分析**

与π_0、π_0.5、MemoryVLA、OpenVLA‑OFT、RDT等SOTA方法对比，优化后在LIBERO平均成功率98.6%，CALVIN提升13.5%，RoboTwin Hard 38%成功率；实时推理速度提升2.9×，NFE显著减少。

**⚠️ 局限性**

局限性包括对检索记忆质量依赖强、对极端新环境的先验检索效果可能下降、局部一致性记忆对长时序依赖可能不足，且记忆库占用额外存储。

---

## 176. Estimation of Confidence Bounds in Binary Classification using Wilson Score Kernel Density Estimation

**arXiv ID:** 2602.20947 | [PDF](https://arxiv.org/pdf/2602.20947v1)

**作者:** Thorbjørn Mosekjær Iversen `[一作]` (SDU Robotics), Frederik Hagelskjær `[通讯]` (SDU Robotics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于 Wilson Score Kernel Density Estimator 的二分类器，能够在每个样本上给出置信区间，适用于选择性分类。

**💡 创新点**

首次将 Wilson Score 置信区间与核密度估计结合，用单一可调宽度参数在保持统计可靠性的同时实现高效推断。

**🔧 技术方法**

使用核密度平滑、Wilson Score 置信区间估计、选择性分类框架，并与高斯过程分类（GPC）进行对比。

**📊 数据集**

在四个数据集上验证：Banknote Authentication、Cats & Dogs（1k/5k）、ChestMNIST（1k/22k）和 Assembly Inspection（动态捕捉的机器人装配图像）。

**📈 对比分析**

与 GPC 在选择性分类指标（AUPRC/AURRC）上几乎相当，但 WS‑KDC 在超参数优化时间上快 2‑3 位数、参数更少、实现更直观。

**⚠️ 局限性**

对宽度参数的选择高度依赖特征平滑性；在样本量极小或特征分布不平滑时置信区间可能过于宽松或过于保守；目前缺乏 GPU 加速实现，需进一步研究自适应带宽选择。

---

## 177. DRESS: A Continuous Framework for Structural Graph Refinement

**arXiv ID:** 2602.20833 | [PDF](https://arxiv.org/pdf/2602.20833v1)

**作者:** Eduar Castrillo Velilla `[一作]` `[通讯]`, Eduar Castrillo Velilla

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 DRESS 家族，一套可扩展的连续结构细化框架，用于无监督图同构检测和结构相似度评估。

**💡 创新点**

创新点在于：①将参数化的三角形邻域迭代推广为任意结构子图（Motif-DRESS）和抽象化聚合/范数（Generalized-DRESS）；②引入节点删减策略 Δ-DRESS，将多重子图视为连续化的“牌组”，实现对强正则图（SRG）等 3-WL 无法区分实例的识别；③保持算法为无学习、无参数的确定式迭代，兼顾高表达力与 O(n³) 级别的稀疏图可扩展性。

**🔧 技术方法**

技术手段包括：基于边的非线性动态系统迭代、聚合函数与范数的度-0齐次性保证、Hilbert 投影度量下的收敛性证明、节点删除与子图多重集构造。

**📊 数据集**

实验数据集包括：棱柱图 vs K₃,₃、SRG(16,6,2,2)（Rook vs Shrikhande）、SRG(28,12,6,4)（Chang 三图）、2×C₄ vs C₈、Petersen vs Pentagonal Prism 等经典同构测试基准。

**📈 对比分析**

对比方法：将 DRESS 的排序边值指纹与 1-WL、3-WL 以及基于 GNN 的同构测试进行对比；结果显示 DRESS 在上述所有实例中均能成功区分，而 3-WL 失败；时间复杂度为稀疏图下的 O(n³)（低于 3-WL 的 O(n⁴)），并在 ≤20 次迭代内收敛。

**⚠️ 局限性**

局限性：收敛性理论仅在满足三条充分条件时保证；对所有子图模式与聚合/范数组合尚无完整证明；在极稠密图上仍呈现 O(n⁴) 级别；仅在无监督设置下验证，未探索对有监督任务的迁移；实验规模仍以中小型基准图为主，缺乏大规模真实网络的验证。

---

## 178. Assessing the Impact of Speaker Identity in Speech Spoofing Detection

**arXiv ID:** 2602.20805 | [PDF](https://arxiv.org/pdf/2602.20805v1)

**作者:** Anh-Tuan Dao `[一作]`, Nicholas Evans `[通讯]` (EURECOM)

**通讯引用:** 9918 | [OpenAlex ID](https://openalex.org/A5066811192)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究说话人信息对语音欺骗检测的影响，提出可在SSL特征下切换说话人感知与说话人不变的多任务框架SInMT。

**💡 创新点**

统一框架支持说话人感知与说话人不变两种策略，利用梯度反转实现对抗式说话人不变训练，首次在语音欺骗检测中同时优化说话人识别与欺骗检测。

**🔧 技术方法**

使用XLSR预训练SSL特征、MHFA多头注意力分类器、梯度反转层（GRL）、对抗训练、t‑SNE可视化。

**📊 数据集**

训练集：ASVspoof 5（约180K句，400位说话人）；评估集：ASVspoof 5 eval、In‑the‑Wild、ASVspoof 2021 LA与DF隐藏子集。

**📈 对比分析**

与AASIST、Conformer、MHFA基线对比；在四个数据集上均优于基线；MHFA‑IVspk平均EER下降17%，对A11攻击下降48%；在最具挑战的A10、A11攻击上获得显著提升。

**⚠️ 局限性**

说话人感知与不变模型的优势尚未完全分离；评估数据集说话人互斥，未验证在闭集说话人上的性能；未探索动态混合说话人感知/不变策略。

---

## 179. Natural Language Processing Models for Robust Document Categorization

**arXiv ID:** 2602.20336 | [PDF](https://arxiv.org/pdf/2602.20336v1)

**作者:** Radoslaw Roszczyk `[一作]` (Warsaw University of Technology), Krzysztof Siwek `[通讯]` (Warsaw University of Technology)

**通讯引用:** 967 | [OpenAlex ID](https://openalex.org/A5002519164)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了 Naïve Bayes、BiLSTM 与 BERT 在不平衡技术支持工单文本分类中的性能，并实现了自动路由演示系统。

**💡 创新点**

在单一不平衡工单数据集上系统对比传统与深度模型，并提出了 BERT 微调策略及 BiLSTM 在精度与成本平衡上的优势。

**🔧 技术方法**

使用 Naïve Bayes、双向 LSTM 网络、预训练 BERT（Fine‑tune）以及 scikit‑learn、TensorFlow/Keras 等深度学习与机器学习框架；文本预处理包括大小写化、去除特殊字符、停用词等。

**📊 数据集**

Customer IT Support - Ticket Dataset（Problem 7120 条、Request 3479 条、Change 1280 条）。

**📈 对比分析**

采用 10 次 k‑fold 交叉验证比较准确率、精确率、召回率和 F1；BERT 取得 99.23%/0.9932，BiLSTM 98.56%/0.9881，Naïve Bayes 94.23%/0.9423；训练时间从毫秒级到约 20 分钟不等。

**⚠️ 局限性**

BERT 的训练与推理成本高，且对少数类 Recall 仍受限；系统在实时高吞吐量下 BERT 可能成为瓶颈；未验证多语言或跨领域的泛化能力。

---

## 180. When Backdoors Go Beyond Triggers: Semantic Drift in Diffusion Models Under Encoder Attacks

**arXiv ID:** 2602.20193 | [PDF](https://arxiv.org/pdf/2602.20193v1)

**作者:** Shenyang Chen `[一作]` (Google), Liuwan Zhu `[通讯]` (University of Hawaii at Manoa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究文本到图像模型中编码器侧后门对内部语义表示的持续影响，指出后门不仅触发特定模式，还会在无触发器的输入下引起语义漂移。

**💡 创新点**

提出SEMAD（Semantic Alignment and Drift）框架，用于量化后门导致的嵌入漂移和下游语义失配，并通过Jacobian‑based局部变形理论解释后门产生低秩、定向扭曲的机制。

**🔧 技术方法**

Jacobian分析、PCA低秩能量评估、语义漂移得分(SDS)、CLIP相似度差分、两样本Welch检验等技术。

**📊 数据集**

使用LAION‑Aesthetics v2、Oxford‑IIIT Pet、ImageNet‑100等数据集；对Stable Diffusion v1.4、Latent‑Diffusion、MoCo v2 (ResNet‑18)等模型进行后门注入与评估。

**📈 对比分析**

与传统攻击成功率（ASR）和图像质量评估对比，SEMAD显示在无触发器条件下语义漂移显著（SDS均值提升3–9倍），CLIP相似度下降约33%，表明后门导致系统性语义失配。

**⚠️ 局限性**

仅关注编码器侧后门，未覆盖U‑Net层后门或推理时攻击，且实验仅在文本到图像和对比学习模型中验证，其他威胁模型与更大规模系统的适用性待进一步研究。

---

## 181. Delay Alignment Modulation for Secure ISAC Systems

**arXiv ID:** 2602.21114 | [PDF](https://arxiv.org/pdf/2602.21114v1)

**作者:** Tianyu Lu `[一作]` (Queen's University Belfast), Michail Matthaiou `[通讯]` (Queen's University Belfast)

**通讯引用:** 13388 | [OpenAlex ID](https://openalex.org/A5035876091)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了基于时延对齐调制（DAM）的安全集成感知与通信系统，通过两阶段协议实现目标定位与多用户安全通信；

**💡 创新点**

创新点在于利用DAM在多路径环境下对合法用户的时延与角度进行精确对齐，产生对抗Eve的误差，且在不额外耗电的情况下提升信息安全性；

**🔧 技术方法**

使用了时延对齐调制、路径基础零逼近（ZF）预编码、最大似然延迟估计、Cramér–Rao界（CRB）分析、凸优化（SCA + CVX）等技术；

**📊 数据集**

未使用公开数据集，所有结果均基于系统参数（N=30/100天线、f_c=28 GHz、B=128 MHz等）进行仿真；

**📈 对比分析**

与传统最强路径（SP）基准进行对比，仿真表明DAM在提高最差用户信道安全谱效率（SSE）方面明显优于SP，且最大似然延迟估计误差接近CRB；

**⚠️ 局限性**

局限性包括：仅考虑单目标场景；对多径估计的角度与时延精度假设过于理想；算法复杂度高（每轮迭代 O(n_var³)）；未验证在实际硬件与动态多用户环境中的鲁棒性。

---

## 182. Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework

**arXiv ID:** 2602.20375 | [PDF](https://arxiv.org/pdf/2602.20375v1)

**作者:** Jiashun Wang `[一作]` (Carnegie Mellon University), Farbod Farshidian `[通讯]` (RAI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种多任务RL框架，将参考运动仅用作训练时的奖励而非执行时的输入，训练单一目标条件策略实现自然且可迁移的人形机器人运动。

**💡 创新点**

创新点在于通过联合训练模仿任务与目标驱动任务，兼顾运动质量与泛化，避免了传统跟踪RL对参考轨迹的过度依赖。

**🔧 技术方法**

使用PPO、PD控制器、自动难度调度、虚拟助推力、稠密与稀疏奖励混合等技术实现多任务学习。

**📊 数据集**

利用人类运动捕捉数据（如ZEST、MoCap）以及自定义的Box-based Parkour Playground场景。

**📈 对比分析**

与纯RL、跟踪RL（ZEST）以及无模仿基线对比，在模拟与硬件上均实现更高成功率、低根部姿态误差和自然运动，尤其在超出训练分布的初始条件下表现优异。

**⚠️ 局限性**

局限在于需人工设计高层状态机作曲者，参考数据需要覆盖多姿态；在极端环境或未知任务中仍可能缺乏足够的自适应能力。

---

## 183. Closing the Expertise Gap in Residential Building Energy Retrofits: A Domain-Specific LLM for Informed Decision-Making

**arXiv ID:** 2602.20181 | [PDF](https://arxiv.org/pdf/2602.20181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 184. Bridging Physically Based Rendering and Diffusion Models with Stochastic Differential Equation

**arXiv ID:** 2602.20725 | [PDF](https://arxiv.org/pdf/2602.20725v1)

**作者:** Junwei Shu `[一作]` (East China Normal University), Changbo Wang `[通讯]` (East China Normal University)

**通讯引用:** 1322 | [OpenAlex ID](https://openalex.org/A5063110936)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种统一的随机微分方程（MC‑SDE）框架，将物理基础渲染（PBR）与扩散生成模型桥接起来，实现了对扩散生成结果的物理可控渲染与材质编辑；

**💡 创新点**

创新点在于：①从中心极限定理推导出蒙特卡洛估计的连续‑时间SDE；②在此基础上构造路径追踪的MC‑SDE并与扩散SDE对齐；③通过噪声方差匹配实现低样本路径追踪图像与扩散模型的无缝对接，并利用光照与材质方差特性实现精细材质调控；

**🔧 技术方法**

采用随机微分方程（SDE）理论、蒙特卡洛路径追踪、扩散模型（Stable Diffusion v1‑4）以及ControlNet与时间条件适配器等技术；

**📊 数据集**

使用自建的30场景、14种样本数的低样本路径追踪数据集（包含不同光照与材质设置）进行实验；

**📈 对比分析**

通过与基线（直接用扩散模型去噪）和τ‑Mapper+适配器两种方法对比，评估PSNR、SSIM、LPIPS等指标，实验显示τ‑Mapper+适配器在PSNR提升至20.72、SSIM 0.71、LPIPS 0.37，明显优于基线；

**⚠️ 局限性**

局限性包括：仅在二维图像域验证，无法直接处理动态场景；适配器训练对数据集依赖较高；对高阶全局光照的控制仍有限。

---

## 185. OmniOCR: Generalist OCR for Ethnic Minority Languages

**arXiv ID:** 2602.21042 | [PDF](https://arxiv.org/pdf/2602.21042v1)

**作者:** Bonan Liu `[一作]` (Southwest Minzu University), Ying Cai `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了OmniOCR，一种针对少数民族语言的通用OCR框架。

**💡 创新点**

创新点是动态低秩适配器（Dynamic LoRA），在保持知识保留的同时为不同书写系统自适应分配模型容量，并通过稀疏正则化实现参数高效。

**🔧 技术方法**

采用的技术包括基于RolmOCR的视觉语言基础模型、Dynamic LoRA（可变秩低秩微调）、稀疏正则化、混合精度训练等。

**📊 数据集**

使用的数据集包括TibetanMNIST、Shui、古逸（Ancient Yi）和东巴（Dongba）四个少数民族文字数据集。

**📈 对比分析**

与零射击大模型和传统全量微调相比，OmniOCR在四个数据集上平均准确率提升39%–66%，零射击模型准确率约25%–45%，全量微调最高可达95%以上，而OmniOCR保持参数占用最低。

**⚠️ 局限性**

局限性包括仅覆盖四个脚本，训练仍需较多GPU资源，且未充分考虑文档降解、背景噪声和复杂布局等真实场景问题。

---

## 186. MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification

**arXiv ID:** 2602.20873 | [PDF](https://arxiv.org/pdf/2602.20873v1)

**作者:** Jiahao Xu `[一作]` (Chongqing University), Nankun Mu `[通讯]` (Chongqing University)

**通讯引用:** 692 | [OpenAlex ID](https://openalex.org/A5085538046)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为MUSE的随机多视角语义增强框架，用于极少标注的全切片图像分类；

**💡 创新点**

创新点在于通过Mixture‑of‑Experts实现样本级细粒度语义细化，以及通过LLM生成的多视角知识库进行随机检索与优化，显著提升语义精度与多样性；

**🔧 技术方法**

采用了视觉‑语言模型（如CONCH）、Mixture‑of‑Experts、跨模态注意力、检索增强生成、LLM知识库构建和随机优化等技术；

**📊 数据集**

在CAMELYON、TCGA‑NSCLC和TCGA‑BRCA三大全切片数据集上进行实验；

**📈 对比分析**

与传统MIL、VLM基线进行对比，MUSE在4/8/16‑shot设置下均超越对手，4‑shot时准确率提升约6–7％，整体性能表现最优；

**⚠️ 局限性**

局限性包括对LLM知识库质量的高度依赖、检索与随机训练的计算成本，以及对预训练基座模型的依赖，可能在更稀缺或非病理场景下表现不足。

---

## 187. CausalReasoningBenchmark: A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation

**arXiv ID:** 2602.20571 | [PDF](https://arxiv.org/pdf/2602.20571v1)

**作者:** Ayush Sawarni `[一作]` (Stanford University), Vasilis Syrgkanis `[通讯]` (Stanford University)

**通讯引用:** 1868 | [OpenAlex ID](https://openalex.org/A5017741738)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了CausalReasoningBenchmark，包含173个查询和138个真实数据集，要求系统给出结构化的识别规范和估计结果，从而评估自动化因果推理系统的性能。

**💡 创新点**

创新点在于将因果识别与估计分离评价、提供五类常见设计的JSON规范、使用真实研究与教材案例、并通过细粒度评分揭示模型薄弱环节。

**🔧 技术方法**

主要采用大型语言模型（OpenAI GPT‑5）作为基线，利用结构化提示生成识别规范和Python估计脚本，并实现自适应单位缩放的评估器。

**📊 数据集**

使用了来自85篇同行评审政治学论文的120个查询和53个教材示例，共138个数据集，涵盖IV、RDD、DiD、CE和RCT设计。

**📈 对比分析**

与基线模型的比较显示：识别策略正确率84%，完整识别规范仅30%；估计误差中位数相对误差约15.8%，CI重叠率89%；按设计类型拆分时，RDD表现最好，IV和CE仍存在较大差距。

**⚠️ 局限性**

局限性包括：数据主要集中在政治学领域，设计覆盖有限（缺乏合成控制、事件研究等），仅提供单一黄金规范、估计可变性未完全考虑，以及样本规模相对较小。

---

## 188. Emergent Manifold Separability during Reasoning in Large Language Models

**arXiv ID:** 2602.20338 | [PDF](https://arxiv.org/pdf/2602.20338v1)

**作者:** Alexandre Polo `[一作]` (Harvard University), SueYeon Chung `[通讯]` (Harvard University)

**通讯引用:** 777 | [OpenAlex ID](https://openalex.org/A5016533438)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Chain-of-Thought 推理过程中内部表示的几何动态，发现其呈现瞬时的线性可分性脉冲。

**💡 创新点**

首次将 Manifold Capacity Theory 用于时序推理分析，揭示了动态的 manifold 管理机制。

**🔧 技术方法**

使用 Manifold Capacity Theory 与线性探针、注意力热图等技术对大型语言模型的残差流进行几何量化。

**📊 数据集**

采用层级布尔逻辑树（height 5，共 31 个内部节点）作为合成推理任务的数据集。

**📈 对比分析**

与仅提示答案的 baseline 对比，模型在 CoT 条件下达到 98% 准确率，而 baseline 仅 59%；Manifold Capacity 在推理时峰值后迅速衰减，线性探针保持高准确。

**⚠️ 局限性**

局限在于仅使用合成布尔任务，缺乏自然语言模糊性，且假设内部主要使用线性可分性，忽略非线性几何。

---

## 189. Enhancing Hate Speech Detection on Social Media: A Comparative Analysis of Machine Learning Models and Text Transformation Approaches

**arXiv ID:** 2602.20634 | [PDF](https://arxiv.org/pdf/2602.20634v1)

**作者:** Saurabh Mishra `[一作]` (International Institute of Information Technology), Radhika Mamidi `[通讯]` (International Institute of Information Technology)

**通讯引用:** 1218 | [OpenAlex ID](https://openalex.org/A5038314215)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并比较传统与先进机器学习模型（CNN、LSTM、Bi‑LSTM、BERT、DistilBERT）及其混合架构在检测社交媒体仇恨言论与攻击性语言的效果，并提出将攻击性表达转换为中性表述的文本中和方法；

**💡 创新点**

①系统性比较传统模型与Transformer模型的表现；②设计混合模型（如BERT+CNN、BERT+Bi‑LSTM）提升检测准确率；③引入动态文本中和流程（结合BERT分类与LLM生成中性文本），实现实时内容软化；

**🔧 技术方法**

深度学习网络（CNN、LSTM、Bi‑LSTM），Transformer模型（BERT、DistilBERT），混合网络集成，OpenAI GPT/LLM用于文本中和；

**📊 数据集**

约2.5万条推文（24,783条），标签为仇恨言论、攻击性语言与中性；

**📈 对比分析**

采用精确率、召回率、F1‑score、准确率等指标对模型进行统一评估，BERT与DistilBERT在整体准确率（≈91%）与F1上领先；混合模型在部分类别（如仇恨言论）提升了召回率；动态中和流程在检测后可实现内容软化，评估通过人工判定一致性与语义保持率；

**⚠️ 局限性**

数据偏斜导致模型对攻击性语言识别更好，仇恨言论识别仍低；模型依赖预训练权重，难以捕捉新兴词汇；Transformer模型计算量大，低资源环境部署受限；文本中和方法在语义细微处仍可能出现误差或信息丢失。

---

## 190. gQIR: Generative Quanta Image Reconstruction

**arXiv ID:** 2602.20417 | [PDF](https://arxiv.org/pdf/2602.20417v1)

**作者:** Aryan Garg `[一作]` (University of Wisconsin-Madison), Mohit Gupta `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

结合 SPAD 低光子计数与大规模文本到图像扩散模型，提出三阶段框架 gQIR，通过 VAE 对齐、LoRA 对抗精调与 FusionViT 时空融合，实现从极少光子检测的二值帧到高质量彩色图像的重建。

**💡 创新点**

首次将预训练文本到图像扩散模型迁移至量子闪烁成像领域，解决 Bernoulli 噪声与稀疏光子计数问题，并通过可学习的 VAE 对齐、对抗 LoRA 细节增强、时空 Transformer FusionViT 等模块统一处理运动估计、对齐、去噪和去镶嵌。

**🔧 技术方法**

采用 VAE 对齐（deterministic encoding + LSA loss）、LoRA 低秩 U‑Net + 对抗训练、FusionViT miniViT 时空融合、RAFT 光流估计、Stable Diffusion 预训练模型、ConvNext‑Large 判别器与 GAN 对抗、单步生成器等技术。

**📊 数据集**

使用合成基于 Stable Diffusion 的 2.81M 图像与 44k 视频数据集、真实 1Mpx 彩色 SPAD burst 数据集（6k fps）、新构建的 eXtreme‑Deformable (XD) 视频数据集，以及公开的 XVFI、I2‑2000fps 等序列。

**📈 对比分析**

与 Fine‑tuned Restormer、NAFNet、QBP、QUIVER、QuDi、EMVD、FloRNN 等基线在单帧和 burst 任务中比较；在 PSNR、SSIM、LPIPS、ManIQA、ClipIQA、MUSIQ 等指标上，gQIR 在大多数指标上超越基线，特别是在极端运动（XD）和高帧率（100k fps）下保持高感知质量与时空一致性。

**⚠️ 局限性**

仅在固定 PPP 3.5 下训练，低 PPP 或不同光子统计的鲁棒性有限；使用 8‑bit VAE 解码器限制 HDR；光流在量子数据上的误差导致微小漂移；未整合多帧扩散先验；对传感器暗计数、热像素等细节处理不完整。

---

## 191. Mask-HybridGNet: Graph-based segmentation with emergent anatomical correspondence from pixel-level supervision

**arXiv ID:** 2602.21179 | [PDF](https://arxiv.org/pdf/2602.21179v1)

**作者:** Nicolás Gaggion `[一作]` (Universidad de Buenos Aires), Enzo Ferrante `[通讯]` (Universidad de Buenos Aires)

**通讯引用:** 5407 | [OpenAlex ID](https://openalex.org/A5032685263)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了 Mask‑HybridGNet 框架，能够使用标准像素掩码训练图形化医学图像分割模型，无需人工标注对应关键点。

**💡 创新点**

创新点在于通过 Chamfer 距离监督、边缘正则化与可微光栅化，将变长真实边界与固定长度图形节点对齐，使网络在无监督的情况下自然学习一致的解剖对应关系。

**🔧 技术方法**

采用变分编码器‑解码器、图卷积网络、Chamfer 距离损失、边缘正则化、SoftPolygon 可微光栅化以及辅助像素分割的双解码器架构。

**📊 数据集**

在胸部 X‑ray、心脏超声、心脏 MRI、胎儿超声以及大规模 PAX‑Ray++ 37 结构 X‑ray 等多种医学影像数据集上进行实验。

**📈 对比分析**

与 nnUNet 等基准方法比较，Mask‑HybridGNet 在 Dice、Hausdorff、ASSD 等指标上保持相近或略优，并提供拓扑一致性和解剖对应；双解码器与统一图形表示获得最佳性能。

**⚠️ 局限性**

局限在于仅支持二维分割、仅能建模闭合边界结构、无法处理分支结构或孔洞，且需手工设计邻接矩阵；3D 扩展和更复杂拓扑仍待研究。

---

## 192. Blackbird Language Matrices: A Framework to Investigate the Linguistic Competence of Language Models

**arXiv ID:** 2602.20966 | [PDF](https://arxiv.org/pdf/2602.20966v1)

**作者:** Paola Merlo `[一作]` (Idiap Research Institute), Vivi Nastase `[通讯]` (Idiap Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Blackbird Language Matrices（BLM）任务，以文本版 Raven 矩阵形式考察大语言模型的语言能力。

**💡 创新点**

创新点在于将语言现象转化为多层级、多选矩阵，结合结构化模板与人工校验，既具自然语言真实性，又可用于系统性、组合性和归纳性研究。

**🔧 技术方法**

采用预训练 Transformer（Electra、OpenAI GPT）生成句子嵌入，结合 FFNN/CNN 基线与 VAE 结构化压缩，进一步构建两层 VAE 进行跨句子系统性学习。

**📊 数据集**

构建了涵盖七种语言现象（喷涂/装载、因果/不因果、对象丢弃、主谓一致、时态序列）的四语言（英语、法语、意大利语、罗马尼亚语）BLM 数据集，包含三种词汇变异级别（I/II/III）。

**📈 对比分析**

与传统最小对比对照数据集（BLiMP、Holmes）及基线模型相比，BLM 在多语言、多任务上实现了 0.7–0.9 之间的 F1，VAE 系统在类型 III 词汇变异中显著提升 10–20% 的准确率，证明结构压缩有助于系统性学习。

**⚠️ 局限性**

局限包括数据生成仍需人工干预、任务对模型的推理层面要求较高导致现有 LLM 在矩阵逻辑推断上仍表现欠佳，且跨语言迁移性和更复杂语义变体的评估尚未覆盖。

---

## 193. SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images

**arXiv ID:** 2602.20412 | [PDF](https://arxiv.org/pdf/2602.20412v1)

**作者:** Aayush Dhakal `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**通讯引用:** 25104 | [OpenAlex ID](https://openalex.org/A5060280374)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SimLBR 框架，利用潜在空间的混合正则化来学习围绕真实图像分布的紧致决策边界，实现对未见生成器的鲁棒检测。

**💡 创新点**

核心创新在于将真实图像与少量伪造信息在 DINOv3 潜在空间中线性混合，并将混合样本标记为伪造，从而强制模型只识别完全未受污染的真实样本，避免对特定生成器痕迹的过拟合。

**🔧 技术方法**

技术包括：DINOv3 预训练特征提取器、潜在混合正则化（LBR）、轻量级两层 MLP 分类器、α 值从 0.5–B（B≈0.8）均匀采样、BCE 损失、Adam 优化器。

**📊 数据集**

评估数据集有：GenImage、AIGC、Chameleon（人工挑选的难检测样本）以及 RSFake（遥感伪造），所有实验均基于 Stable Diffusion 1.4、ProGAN 等单一训练生成器，测试多种未见生成器。

**📈 对比分析**

与多种现有方法（UnivFD、AIDE、PatchCraft 等）对比，SimLBR 在 GenImage 上平均准确率 94.54%（比 SOTA 高 7.66%），在 Chameleon 上最高准确率及召回率提升至 25% 以上，标准差最低、可靠性分数最高，训练时间仅 3 分钟，显著优于耗时数小时的 AIDE 等。

**⚠️ 局限性**

局限性包括：假设真实图像分布随时间保持相对稳定，若真实域发生剧烈变化可能导致性能下降；对潜在空间结构依赖较高，在 DINOv2 等较弱特征空间中效果不佳；需要在更广泛领域验证其通用性。

---

## 194. How Do Inpainting Artifacts Propagate to Language?

**arXiv ID:** 2602.20520 | [PDF](https://arxiv.org/pdf/2602.20520v1)

**作者:** Pratham Yashwante `[一作]` (University of California San Diego), Sukruth Rao `[通讯]` (University of California San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了扩散式图像修复引入的视觉伪影如何影响视觉语言模型的生成输出，构建两阶段诊断框架并系统评估。

**💡 创新点**

提出了模型无关的诊断框架，量化重建质量与生成语言之间的相关性，并揭示伪影导致的层级注意力漂移。

**🔧 技术方法**

使用Stable Diffusion 1.5/2.0/3.0进行图像修复，冻结BLIP、LLaVA、Qwen2.5‑VL等caption模型，提取ViT‑Base视觉编码器的注意力与嵌入，评估MSE/LPIPS/PSNR/SSIM等视觉指标与BLEU/METEOR/ROUGE等语言指标。

**📊 数据集**

使用Flickr、RefCOCOg、TRUCE、ROCOv2、Indiana X‑Ray、GTZAN等多域视觉语言数据集。

**📈 对比分析**

通过比较原始与修复图像在同一冻结caption模型下的输出，计算视觉指标与语言指标的相关性；发现视觉重建误差越小，语言质量越高，LPIPS/MSE 与BLEU/SimCSE 等指标呈显著正相关，注意力漂移随层加深而增大。

**⚠️ 局限性**

仅针对扩散式修复，未覆盖其他视觉预处理；未探索所有重建超参数；仅在冻结模型下实验，未考虑端到端自适应；对语言多样性有限的数据集（GTZAN、X‑ray）相关性弱。

---

## 195. trainsum -- A Python package for quantics tensor trains

**arXiv ID:** 2602.20226 | [PDF](https://arxiv.org/pdf/2602.20226v1)

**作者:** Paul Haubenwallner `[一作]` (Fraunhofer Institute for Computer Graphics Research), Matthias Heller `[通讯]` (Fraunhofer Institute for Computer Graphics Research)

**通讯引用:** 439 | [OpenAlex ID](https://openalex.org/A5025355615)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了名为 trainsum 的 Python 包，实现了对任意维度量子张量训练（Quantics Tensor Trains）的构造、近似、运算以及线性/非线性问题求解。

**💡 创新点**

创新点在于：① 支持任意维度的质因数分解；② 在 Einstein 求和语法下直接进行多维张量训练运算；③ 提供三种近似策略（zip‑up、变分 DMRG、交叉插值），实现高效的秩控制；④ 与 NumPy、CuPy、Torch 等 Array API 后端无缝对接。

**🔧 技术方法**

技术实现基于 Array API 标准、NumPy/CuPy/Torch 后端、张量网络理论、量子张量训练、矩阵分解（SVD、QR）、交叉插值、DFT、Toeplitz 矩阵等。

**📊 数据集**

示例实验使用的典型数据集包括热方程（PDE）、氢原子本征值、傅里叶谱、图像压缩、MNIST 图像分类以及离散卷积（Toeplitz）等，但未给出具体公开数据集的引用。

**📈 对比分析**

与现有库（如 t3f、quimb 等）和手工实现对比，trainsum 在低秩逼近、交叉插值以及多维 Einstein 求和上表现出显著的速度与内存优势，精度与传统方法基本相同，尤其在任意维度张量和大规模运算中提升可达数十倍。

**⚠️ 局限性**

主要局限在于交叉插值的数值稳定性仍需提升、缺乏高级切片赋值操作、对极大规模张量的并行/分布式支持有限。

---

## 196. Two approaches to low-parametric SimRank computation

**arXiv ID:** 2602.20282 | [PDF](https://arxiv.org/pdf/2602.20282v1)

**作者:** Egor P. Berezin `[一作]` (Lomonosov Moscow State University), Sergey A. Matveev `[通讯]` (Lomonosov MSU)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了低参数方式近似SimRank矩阵，提出了两种从一开始就直接求解低参数形式的算法。

**💡 创新点**

创新点在于引入非对称的交替最小化与对称的二次（Newton）最小化方法，能够在迭代过程中避免存储完整的 n×n 矩阵，同时在 Chebyshev 误差上实现高精度。

**🔧 技术方法**

使用交替最小化、随机化 SVD、GMRES 求解雅可比方程、Newton 迭代以及稀疏矩阵运算等技术，全部实现于 Python/NumPy/SciPy 环境。

**📊 数据集**

在 SNAP 开源图数据集（如 4039 节点的 Epinions、7115 节点的 High‑energy physics citation 等）以及 34546 节点的高能物理引用网络上进行了实验。

**📈 对比分析**

与基准 RSVD 方法比较，交替最小化在 Chebyshev 误差小于 0.1、top‑10 推荐准确率约 60% 时实现了约 5 倍压缩；二次最小化性能略逊，但同样在误差与推荐精度上均优于 RSVD。

**⚠️ 局限性**

主要局限是缺乏理论收敛证明、迭代复杂度仍较高、对大规模数据的并行化实现尚未完成。

---

## 197. Task-oriented grasping for dexterous robots using postural synergies and reinforcement learning

**arXiv ID:** 2602.20915 | [PDF](https://arxiv.org/pdf/2602.20915v1)

**作者:** Dimitrios Dimou `[一作]`, Plinio Moreno `[通讯]` (Institute for Systems and Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并训练了一种基于强化学习的多任务抓取策略，能够根据指定的后抓取意图（使用或递交）对不同物体进行抓取。

**💡 创新点**

通过提取ContactPose数据中的人类抓取偏好，利用VAE学习手部协同空间并作为RL动作空间，实现了端到端、任务导向且人类化的抓取。

**🔧 技术方法**

采用PPO强化学习、VAE协同空间、操作空间控制器、基于抓取点的奖励函数以及物体位姿/类别信息。

**📊 数据集**

使用ContactPose数据集，提供多物体的3D抓取姿势和对应的后抓取意图。

**📈 对比分析**

与全关节控制策略和PCA协同空间策略对比，VAE协同空间策略在抓取成功率上达83%，显著优于其他方法。

**⚠️ 局限性**

仅考虑抓取目标位置而不关注物体姿态或功能特征，假设同类物体尺寸相近，对尺寸差异大或视觉信息缺失的情况性能下降。

---

## 198. cc-Shapley: Measuring Multivariate Feature Importance Needs Causal Context

**arXiv ID:** 2602.20396 | [PDF](https://arxiv.org/pdf/2602.20396v1)

**作者:** Jörg Martin `[一作]` (Physikalisch-Technische Bundesanstalt), Stefan Haufe `[通讯]` (Technische Universität Berlin)

**通讯引用:** 8410 | [OpenAlex ID](https://openalex.org/A5068256213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于因果结构的Shapley值变体cc‑Shapley，用来纠正传统Shapley值在观测上下文中因碰撞偏差（collider bias）导致的误导性特征重要性评估。

**💡 创新点**

创新点在于将因果干预（intervention）引入Shapley值计算，从而在多变量背景下消除因碰撞导致的虚假相关，首次在XAI中实现对多元相互作用的因果上下文解释。

**🔧 技术方法**

采用因果图（结构因果模型）与后门调整、干预模拟、回归/机器学习模型估计等技术；理论上证明cc‑Shapley满足统计关联性（SAP）并消除碰撞偏差；实验上使用线性与非线性合成SCM、糖尿病血糖/BMI数据以及蛋白质因果网络数据。

**📊 数据集**

使用了三类数据集：1）随机生成的线性SCM（8个特征+Y）；2）模拟的糖尿病诊断数据（血糖G、平均血糖H、BMI B、目标Y）；3）真实蛋白质浓度数据（8种蛋白的因果图）。

**📈 对比分析**

通过与传统Shapley值、单变量重要性以及理论推导的回归系数比较，评估cc‑Shapley在消除碰撞偏差、保持正向相关以及对高维交互的鲁棒性；实验显示cc‑Shapley在合成与真实数据中显著消除了负相关或误判，并在大多数实例保持与直观一致。

**⚠️ 局限性**

局限包括：1）对完整因果图的先验依赖，未解决因果结构未知时的自动学习；2）计算复杂度高，无法直接扩展到高维；3）假设外生噪声独立且数据为静态结构，难以推广到图像等非结构化数据；4）对干预模拟与模型拟合的逼近误差敏感。

---

## 199. T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation

**arXiv ID:** 2602.21043 | [PDF](https://arxiv.org/pdf/2602.21043v1)

**作者:** Dongik Park `[一作]` (Seoul National University), Hyung-Sin Kim `[通讯]` (Seoul National University)

**通讯引用:** 1877 | [OpenAlex ID](https://openalex.org/A5065781070)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为T1的CNN‑Transformer混合架构，用于多变量时序数据的缺失值填补；通过自监督学习在多尺度卷积提取时间特征后，再进行跨变量信息传递；

**💡 创新点**

核心创新在于Channel‑Head Binding（CHead Attention），将CNN每个通道与Transformer单个注意力头一一绑定，实现特征级别的可选择信息流；同时利用Mask‑Aware Embedding和自适应权重下调，提高了在高缺失率下的鲁棒性；

**🔧 技术方法**

技术细节包括：实例归一化 + 缺失掩码输入；1D深度卷积（多尺度）做Q/K/V投影；Channel‑Head注意力；点卷积（PWConv）作为FFN；PixelShuffle1D进行无损上采样；自监督 40% 训练掩码；

**📊 数据集**

实验数据集：11个公开基准（ETT‑{h1,h2,m1,m2}、Electricity、Weather、Illness、Exchange、PEMS03）以及两类自然缺失数据集（PhysioNet 2012、AQI36）；

**📈 对比分析**

在同一训练/测试条件下与11个最先进基准（TimeMixer++, ModernTCN, iTransformer, TimesNet, PatchTST, DLinear, ImputeFormer, SAITS, CSDI, BRITS, PSW‑I）对比，平均 MSE 减少 46%，在 70% 缺失率时 MSE 几乎减半；在块缺失和自然缺失场景同样保持显著优势；

**⚠️ 局限性**

局限性：1) 主要在离线批处理下验证，未覆盖实时流式填补；2) 对极端稀疏（>90%）仍存在误差；3) 统一超参虽稳健，但对某些域特定数据仍需微调；4) 计算量相对传统CNN略大，适用于资源受限环境时需进一步压缩；

---

## 200. A Micro-Macro Model of Encounter-Driven Information Diffusion in Robot Swarms

**arXiv ID:** 2602.21148 | [PDF](https://arxiv.org/pdf/2602.21148v1)

**作者:** Davis S. Catherman `[一作]` (Worcester Polytechnic Institute), Carlo Pinciroli `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3160 | [OpenAlex ID](https://openalex.org/A5034543991)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了EDID（Encounter-Driven Information Diffusion）模型，并从微观的平均自由路径到宏观的物流和Gompertz动力学推导信息扩散过程。

**💡 创新点**

首次将平均自由路径与机器人碰撞理论结合，构建了混合逻辑‑Gompertz模型，捕捉密度驱动的扩散分岔。

**🔧 技术方法**

使用ARGoS物理仿真进行随机行走（CRW、LW、Hybrid）实验，统计碰撞并基于主方程求解混合动力学。

**📊 数据集**

通过3,160次物理精确仿真（不同N、C、L、速度和随机行走参数）生成的实验数据。

**📈 对比分析**

将模型预测与仿真结果比较，误差均值约1e-2，混合参数λ能在不同通信密度下精准拟合。

**⚠️ 局限性**

模型假设均匀随机运动、无障碍，无法处理动态障碍或非均匀分布；缺乏在真实机器人实验中的验证。

---

## 201. On the Explainability of Vision-Language Models in Art History

**arXiv ID:** 2602.20853 | [PDF](https://arxiv.org/pdf/2602.20853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 202. Graph Modelling Analysis of Speech-Gesture Interaction for Aphasia Severity Estimation

**arXiv ID:** 2602.20163 | [PDF](https://arxiv.org/pdf/2602.20163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 203. Seeing Through Words: Controlling Visual Retrieval Quality with Language Models

**arXiv ID:** 2602.21175 | [PDF](https://arxiv.org/pdf/2602.21175v1)

**作者:** Jianglin Lu `[一作]` (Adobe Research), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31381 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了质量可控检索（QCR）框架，通过大型语言模型在短查询上完成查询扩展并加入质量控制；

**💡 创新点**

核心创新在于将检索质量（相关性和美学）离散化为条件，并利用条件化查询完成技术实现可控的检索结果；

**🔧 技术方法**

使用大型预训练语言模型（如GPT‑2、Qwen2.5）做查询完成，配合预训练的图文检索模型（CoCa、Blip‑2、OpenCLIP）和美学评估器；

**📊 数据集**

在开放版权的Openverse 2.4M图像集与MS‑COCO 118k图像+描述集上进行训练与评估；

**📈 对比分析**

与基线（仅使用原始短查询、未微调的LLM、随机质量标签微调等）相比，QCQC在平均相关性和美学分数上均有显著提升，尤其在高质量条件下表现突出；

**⚠️ 局限性**

局限性包括对数据集特定的适配度高、跨数据集迁移效果受限、仅考虑两维质量（相关性、美学）且无法处理更细粒度或多样性等维度。

---

## 204. GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**arXiv ID:** 2602.20871 | [PDF](https://arxiv.org/pdf/2602.20871v1)

**作者:** Wenbo Yu `[一作]` (Beijing Forestry University), Di Hu `[通讯]` (Renmin University of China)

**通讯引用:** 2276 | [OpenAlex ID](https://openalex.org/A5100670614)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一种跨任务持续的仿真到现实（sim-to-real）迁移框架 GeCo‑SRT，通过累计几何知识实现对新任务的快速适应

**💡 创新点**

创新点在于：①利用局部几何特征（平面、线性、显著性）作为同时域不变、任务不变的知识媒介；②设计 Geometry‑Aware Mixture‑of‑Experts（Geo‑MoE）动态路由局部几何信息；③提出 Geometry‑Expert‑Guided Prioritized Experience Replay（Geo‑PER）通过专家利用度来防止灾难性遗忘

**🔧 技术方法**

技术手段包括：3D 点云编码器、扩散策略、共享残差模块、Mixture‑of‑Experts 网络、基于专家激活度的 PER、混合回放缓冲、光流人机干预数据收集

**📊 数据集**

数据集涵盖四个机器人操作任务（Pick Cube、Stack Cube、Pick Banana、Plug Insert）在 ManiSkill 及真实 XArm+Rotiq 机器人环境中收集的 2000 条仿真专家轨迹和 60 条人类纠正轨迹

**📈 对比分析**

与 Direct Deploy、Action Residual、Transic、Naïve Fine‑tuning、Geo‑MoE+PER/EWC 等基线对比，单任务时 Geo‑MoE 提升到 80%+ SR，跨任务时 GeCo‑SRT 取得 63.3% 平均成功率，平均遗忘率最低，且在仅 20 条轨迹下即可达到 76% SR，显示显著的数据效率提升

**⚠️ 局限性**

局限性主要在：①聚焦观测层面的几何对齐，未充分处理动力学或非几何的 sim‑real 差异；②对非几何任务或复杂动态场景的迁移效果尚未验证；③需要人工纠正数据，收集成本仍存在

---

## 205. ICSSPulse: A Modular LLM-Assisted Platform for Industrial Control System Penetration Testing

**arXiv ID:** 2602.20663 | [PDF](https://arxiv.org/pdf/2602.20663v1)

**作者:** Michail Takaronis `[一作]` (Norwegian University of Science and Technology), Sokratis Katsikas `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4136 | [OpenAlex ID](https://openalex.org/A5022741687)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

开发并演示了ICSSPulse——一个集成网络扫描、Modbus/OPC UA协议交互与LLM辅助报告生成的开源工业控制系统渗透测试平台。

**💡 创新点**

首次在同一轻量化Web平台中实现协议级扫描、交互和基于LLM的结构化报告，自动将技术发现转化为高层决策与技术细节两种报告。

**🔧 技术方法**

使用Flask搭建Web前端，RustScan完成主机/端口发现，pymodbus和python-opcua实现Modbus/OPC UA交互，GPT‑4o‑mini通过OpenAI API完成报告生成。

**📊 数据集**

采用合成Modbus TCP服务器、Factory I/O水处理仿真场景以及自定义OPC UA生产线模型作为测试数据集。

**📈 对比分析**

通过在上述三种实验环境中进行扫描、枚举、读写操作，验证了平台能发现活跃服务、枚举资产并操纵过程变量；虽然未与现有工具做定量基准对比，但实验结果表明功能完整、操作简便且报告质量高。

**⚠️ 局限性**

受限于Python实现、容器化工具、缺乏真实身份验证和硬件交互，导致性能与实验真实度仅适用于中小规模实验室环境；未来需加入更多协议、加速关键组件、扩展到认证与硬件仿真测试。

---

## 206. Extending $μ$P: Spectral Conditions for Feature Learning Across Optimizers

**arXiv ID:** 2602.20937 | [PDF](https://arxiv.org/pdf/2602.20937v1)

**作者:** Akshita Gupta `[一作]` (Purdue University), Venkatram Vishwanath `[通讯]` (Argonne National Laboratory)

**通讯引用:** 3259 | [OpenAlex ID](https://openalex.org/A5075500139)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出一种利用谱尺度条件的新框架，用于推导最大更新参数化（μP），并将其应用于多种自适应一阶与二阶优化器（AdamW、ADOPT、LAMB、Sophia、Shampoo、Muon），随后在NanoGPT和Llama2两大语言模型上验证其零-shot学习率迁移效果。

**💡 创新点**

创新点在于：①用谱尺度条件替代传统的张量程序，使得μP的推导更直观、易于理解；②实现了一套通用的推导流程，能统一处理多种优化器；③通过实验验证了跨宽度的零-shot学习率迁移，并首次在深度维度上提供经验性洞察。

**🔧 技术方法**

技术手段包括：谱范数分析、随机矩阵理论、一次梯度步长分析、对特征向量稳定性的谱尺度约束、对动量/权重衰减等超参的宽度尺度推导，以及使用Python/C++实现的训练代码。

**📊 数据集**

数据集：NanoGPT（小规模Transformer）和Llama2（大规模LLM）两套基准模型，采用标准语言建模任务（如WikiText、OpenWebText）进行训练与评估。

**📈 对比分析**

比较方法：将每种优化器在默认参数下与通过μP推导得到的参数进行对比，评估在不同模型宽度下的验证损失、训练损失下降曲线、学习率迁移成功率。实验结果显示，使用μP后，学习率可在不同宽度上保持相同的最佳值，训练损失随宽度单调下降，整体性能至少与或优于传统手工调参。

**⚠️ 局限性**

局限性：①假设批量大小为常数（B=Θ(1)），在实际大模型训练中常需调整批量大小；②对权重衰减、ε等小超参的尺度推导仍基于经验假设；③深度尺度的理论框架尚未完善，实验结果主要是经验性；④谱尺度条件在非线性激活或极大批量下可能失效，需要进一步验证。

---

## 207. SpecMind: Cognitively Inspired, Interactive Multi-Turn Framework for Postcondition Inference

**arXiv ID:** 2602.20610 | [PDF](https://arxiv.org/pdf/2602.20610v1)

**作者:** Cuong Chi Le `[一作]` (University of Texas at Dallas), Tien N. Nguyen `[通讯]` (University of Texas at Dallas)

**通讯引用:** 7830 | [OpenAlex ID](https://openalex.org/A5089000736)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一种交互式多轮LLM推理框架（Exploratory Multi-turn）用于自动生成函数后置条件。

**💡 创新点**

将LLM视为主动推理者，利用反馈驱动的多轮互动和自我停止决策，显著提升后置条件的正确性与完整性。

**🔧 技术方法**

基于Llama 4 Scout的LLM，交互式提示模板，反馈评估（正确性与完整性得分），以及最佳候选跟踪与自适应停止。

**📊 数据集**

EvalPlus 与 FixEval 作为评测基准。

**📈 对比分析**

与 nl2postcond 进行对比，Exploratory Multi-turn 在正确性提升 26.1% 以及完整性提升 2.48×，平均一次提交完整性提升 1.67×。

**⚠️ 局限性**

评测受限于 EvalPlus 的中等复杂度 Python 代码，未验证在更大系统、多语言或其他规范形式上的泛化；还受限于测试覆盖率与变异器质量。

---

## 208. InterPilot: Exploring the Design Space of AI-assisted Job Interview Support for HR Professionals

**arXiv ID:** 2602.20891 | [PDF](https://arxiv.org/pdf/2602.20891v1)

**作者:** Zhengtao Xu `[一作]` (National University of Singapore), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**通讯引用:** 1805 | [OpenAlex ID](https://openalex.org/A5054435118)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了名为 InterPilot 的实时 AI 辅助面试系统，帮助 HR 在招聘面试中自动记录笔记、生成后续问题和技能‑证据映射；

**💡 创新点**

将实时语音转写、AI 生成跟进问题和动态技能证据图谱集成到单一界面，并在高阶决策情境下探讨关注度与信任的设计张力；

**🔧 技术方法**

采用语音识别、NLP 关键字抽取、基于 STAR 框架的生成式问句、知识图谱可视化和多模态交互技术；

**📊 数据集**

使用人工对话的模拟面试数据、HR 录音笔记以及预先设计的技术/行为技能标签；未公开公开数据集；

**📈 对比分析**

通过双条件对照实验（基准为仅转写）对 NASA‑TLX 与 SUS 进行评估，InterPilot 在工作量无显著差异但可用性下降约12点；在问答深度和客观性上获得质性提升；

**⚠️ 局限性**

样本量小、受访者多为资深 HR，实验为受控模拟面试，缺乏真实多轮面试与长期使用场景；

---

## 209. Transcoder Adapters for Reasoning-Model Diffing

**arXiv ID:** 2602.20904 | [PDF](https://arxiv.org/pdf/2602.20904v1)

**作者:** Nathan Hu `[一作]` (Stanford University), Christopher Potts `[通讯]` (Stanford University)

**通讯引用:** 26215 | [OpenAlex ID](https://openalex.org/A5042601761)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了Transcoder Adapter方法，用于学习并解释微调后MLP计算的差异；

**💡 创新点**

创新点在于将稀疏字典学习直接对齐至基模型与微调模型的MLP差异，能以极少的活跃特征捕捉到精细化训练的影响；

**🔧 技术方法**

技术包括Transcoder Adapter架构、稀疏正则化、桥接损失、归一化MSE、KL散度以及对特征可解释性的自动评估与归因图；

**📊 数据集**

使用OpenThoughts3（50k样本，约380M token）训练，评估基于Qwen2.5-Math-7B与DeepSeek-R1-Distill-Qwen-7B的差异；

**📈 对比分析**

与基模型、Hybrid（仅替换MLP）以及MLP微调上限进行对比；在4个推理基准上，Adapter能恢复50–90%的精度提升，且在输出一致性和内部重建误差上均优于对照；

**⚠️ 局限性**

局限性在于仅捕捉MLP差异，无法解释注意力、嵌入等非MLP参数的变化；实验仅针对一对模型，结果可能不易泛化至更大或更不同的模型架构；

---

## 210. Linear Reasoning vs. Proof by Cases: Obstacles for Large Language Models in FOL Problem Solving

**arXiv ID:** 2602.20973 | [PDF](https://arxiv.org/pdf/2602.20973v1)

**作者:** Yuliang Ji `[一作]` (Nanjing University of Science and Technology), Yue Zhang `[通讯]` (Westlake University)

**通讯引用:** 17979 | [OpenAlex ID](https://openalex.org/A5100333758)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了PC-FOL数据集，用人工标注的形式收集了线性推理和按情形分解的FOL推理题，并为每题附上自然语言证明；

**💡 创新点**

创新点在于：①首次系统性区分并标注线性推理与证据分案两类FOL问题；②引入词汇替换技术使模型无法依赖记忆；③用专家手写证明评估LLM的推理正确性；

**🔧 技术方法**

使用的技术包括：大语言模型（如GPT‑4o、GPT‑4.1、o4‑mini、Llama‑3、Deepseek‑V3、Qwen3）在零/少样本提示下完成推理与证明生成；评估指标包括准确率、ROUGE、pass@k以及人工检查；

**📊 数据集**

主要使用的数据集为PC‑FOL（1022条线性推理+1022条分案推理）以及词汇替换版PC‑FOL‑Replace；

**📈 对比分析**

对比方法是将各模型在两类问题上的准确率与生成证明的ROUGE/Pass@k进行比较，结果显示线性推理准确率约高出30–50%，而分案推理显著较低；

**⚠️ 局限性**

局限性包括：PC‑FOL规模相对较小；理论分析中对单步推理概率的假设过于简化；以及在分案推理中对情形识别和多步骤推理的错误率尚未得到根本解决。

---

## 211. Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change

**arXiv ID:** 2602.20584 | [PDF](https://arxiv.org/pdf/2602.20584v1)

**作者:** Beverley Gorry `[一作]` (Queensland University of Technology), Alejandro Fontan `[通讯]` (Queensland University of Technology)

**通讯引用:** 204 | [OpenAlex ID](https://openalex.org/A5056412871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在海底珊瑚礁长期多次访问中，通过在SfM优化过程中直接强制跨会话对应关系，实现从数年间分散图像生成单一连贯3D模型的方法。

**💡 创新点**

将跨会话对应关系嵌入联合SfM优化，而非后期点云配准，并结合可视化位置识别筛选候选图像对，仅在必要时使用学习型特征匹配，显著提升在巨大外观变化下的重建一致性。

**🔧 技术方法**

使用手工SIFT特征进行同会话匹配，使用MegaLoc进行视觉位置识别，使用LightGlue学习型特征匹配；整个框架基于COLMAP实现联合SfM。

**📊 数据集**

在日本冲绳岛塞索科岛附近的珊瑚礁三年间（2016-2018）由AUV捕获的下降摄像机图像，数据中包含风暴导致的显著结构与外观变化。

**📈 对比分析**

与COLMAP单独重建+ICP、BUFFER-X以及MapAnything等方法对比，利用手工标注的跨会话对应点评估像素投影误差，实验显示我们的框架将误差从几百像素降至约3-4像素，性能提升超过90%。

**⚠️ 局限性**

对重建依赖于不同会话间的视觉重叠，极端结构变化或稀疏观测仍难以集成；学习型匹配增加了计算负担且对预训练模型的泛化能力有限；评价仅基于人工标注，缺乏完整的地面真值。

---

## 212. Autonomous AI and Ownership Rules

**arXiv ID:** 2602.20169 | [PDF](https://arxiv.org/pdf/2602.20169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 213. SparkMe: Adaptive Semi-Structured Interviewing for Qualitative Insight Discovery

**arXiv ID:** 2602.21136 | [PDF](https://arxiv.org/pdf/2602.21136v1)

**作者:** David Anugraha `[一作]` (Stanford University), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13402 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于可定制化效用函数的自适应半结构化访谈框架，并实现了多代理LLM访谈系统SparkMe；

**💡 创新点**

首次将访谈目标转化为可计算的效用优化问题，设计了覆盖率、发现度与成本权衡；同时引入模拟对话回放的前瞻规划提升探索性；

**🔧 技术方法**

多代理LLM（InterviewAgent、AgendaManager、ExplorationPlanner）+ 预测/评估LLM（作为判定者）+ 交互式规划与滚动模拟；

**📊 数据集**

使用基于真实调查问卷的200个模拟访谈者用户模型和70位来自7个专业的真实受访者；

**📈 对比分析**

与四个现有LLM访谈基线对比，SparkMe在预定义子话题覆盖率提升约4.7%，总体效用得分最高，并在真实用户研究中获得更高的内容质量和更多新兴洞察；

**⚠️ 局限性**

缺乏与真人专业访谈者的直接对比；评估主要基于LLM模拟和LLM判定，可能与真实人类互动差异；对用户代理的模拟能力有限。

---

## 214. PaperTrail: A Claim-Evidence Interface for Grounding Provenance in LLM-based Scholarly Q&A

**arXiv ID:** 2602.21045 | [PDF](https://arxiv.org/pdf/2602.21045v1)

**作者:** Anna Martin-Boyle `[一作]` (University of Minnesota), Harmanpreet Kaur `[通讯]` (University of Minnesota)

**通讯引用:** 398 | [OpenAlex ID](https://openalex.org/A5101399911)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于论证结构的LLM答案与学术文献之间主张‑证据对应的来源可追溯界面PaperTrail，并在研究者任务中评估其对信任与使用行为的影响。

**💡 创新点**

将论文主张与证据拆解为离散单元，并通过主张‑证据匹配呈现细粒度来源信息，较传统引用提供更精准的可追溯机制。

**🔧 技术方法**

采用三阶段管道的论证抽取引擎（LLM抽取、相似度过滤、RAG抽取）与Flask REST后端，前端使用React实现多面板协同视图；核心使用Gemini 2.5 Pro进行抽取与匹配。

**📊 数据集**

对四篇火星探索相关PDF进行离线主张‑证据抽取，外部评估采用SciClaimHunt与BioClaimDetect两种公开主张标注数据集。

**📈 对比分析**

与仅提供源引用的基准界面做within‑subject对照，26名研究者实验显示主张‑证据界面显著降低LLM信任（p=0.015），但未显著改变编辑行为；抽取召回率约0.63‑0.88，精确度0.63‑0.69。

**⚠️ 局限性**

系统延迟与界面复杂度导致用户难以充分利用细粒度证据；时间压力抑制深度验证；缺乏长期使用反馈；抽取模型对未覆盖论文结构可能失效；实验样本单一机构，外部可验证性待进一步研究。

---

## 215. AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents

**arXiv ID:** 2602.20569 | [PDF](https://arxiv.org/pdf/2602.20569v1)

**作者:** Jiaqi Wu `[一作]` (Duke University), Jingheng Huan `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

创建了 AIForge‑Doc，首个面向金融及表单文档的扩散模型图像插补伪造检测基准，并提供像素级注释。

**💡 创新点**

首次在真实文档上系统性使用 Gemini 与 Ideogram 的 AI 插补 API 生成数值字段伪造，并公开可复现的生成流程与 9 语种的标注数据。

**🔧 技术方法**

采用上下文窗口插补、手工设计的多样化 prompt、自动 OCR 验证、人工审核，以及 GPT‑4o 零样本评测等技术。

**📊 数据集**

从 CORD、WildReceipt、SROIE、XFUND 四大公开收据/表单数据集抽取 4,061 张样本，分别覆盖金额、日期、电话、地址、表单数字等字段。

**📈 对比分析**

零样本评估 TruFor、DocTamper 与 GPT‑4o，结果在 AIForge‑Doc 上 AUC 分别为 0.751、0.563 与 0.509，显著低于其在传统 Photoshop 伪造数据上的表现，说明现有检测器对 AI 插补伪造无效。

**⚠️ 局限性**

仅单字段伪造、工具有限（仅 Gemini 与 Ideogram）、缺乏多字段/多文档类型、多语种扩展、缺少针对 AI 插补的专门检测模型，且未探讨多字段共变检测与人类感知基准。

---

## 216. Robust Spiking Neural Networks Against Adversarial Attacks

**arXiv ID:** 2602.20548 | [PDF](https://arxiv.org/pdf/2602.20548v1)

**作者:** Shuai Wang `[一作]` (University of Electronic Science and Technology of China), Haizhou Li `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Threshold Guarding Optimization (TGO) 方法提升直接训练 SNN 的对抗鲁棒性

**💡 创新点**

创新点在于将膜电位约束与噪声 LIF 神经元结合，降低阈值邻近神经元对抗攻击的极限与状态翻转概率

**🔧 技术方法**

使用膜电位约束损失、动态 λ 调节、噪声 LIF 模型、BPTT、对抗训练 (AT)、RAT 等技术

**📊 数据集**

在 CIFAR‑10 与 CIFAR‑100 数据集上进行实验

**📈 对比分析**

与现有 SOTA（如 DLIF、StoG、SR、AT、RAT）对比，TGO 在 FGSM、RFGSM、PGD 等攻击下实现 10–20% 的鲁棒性提升，且在多种训练策略下保持优异性能

**⚠️ 局限性**

局限在于需要额外的超参数调优（如 λ_max、噪声方差），并且对极大扰动（高 ϵ）鲁棒性提升有限

---

## 217. Le-DETR: Revisiting Real-Time Detection Transformer with Efficient Encoder Design

**arXiv ID:** 2602.21010 | [PDF](https://arxiv.org/pdf/2602.21010v1)

**作者:** Jiannan Huang `[一作]` (SHI Labs), Humphrey Shi `[通讯]` (SHI Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Le-DETR 模型，显著降低实时 DETR 的预训练成本并提升检测性能。

**💡 创新点**

创新点在于设计 EfficientNAT 轻量级骨干网与 NAIFI 局部注意力编码器，充分利用 Neighborhood Attention 与 MBConv，兼顾速度与精度。

**🔧 技术方法**

采用 Neighborhood Attention、MBConv、Flash Attention、AIFI 替换传统全局自注意力，并结合多尺度特征融合。

**📊 数据集**

使用 ImageNet1K 进行骨干预训练，COCO2017 进行训练与评估。

**📈 对比分析**

与 YOLOv12、RT-DETR、DEIM‑D‑FINE 等 SOTA 模型对比，Le‑DETR‑M/L/X 在 COCO mAP 与 RTX4090 延迟上均实现了领先优势。

**⚠️ 局限性**

仍需进一步减少对大规模预训练数据的依赖，并提升模型在多种硬件平台上的可迁移性与适配性。

---

## 218. WeirNet: A Large-Scale 3D CFD Benchmark for Geometric Surrogate Modeling of Piano Key Weirs

**arXiv ID:** 2602.20714 | [PDF](https://arxiv.org/pdf/2602.20714v1)

**作者:** Lisa Lüddecke `[一作]` (Helmut-Schmidt-University), Oliver Niggemann `[通讯]` (Helmut-Schmidt-University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了大规模的 Piano Key Weir（PKW）三维 CFD 基准数据集 WeirNet，并在该数据集上设计并评估了多种几何和参数化的代理模型，用于预测排水系数 c_D。

**💡 创新点**

创新点在于：①提供了包含 3,794 个可行 PKW 设计、每个设计 19 个不同流量条件下 71,387 次 CFD 计算的公开数据集；②提出了扩展后的 PKW 参数化命名法和可复现的几何生成流程；③结合多模态（参数、网格、点云）与多种模型（树回归、PointNet、RegDGCNN、Mesh‑GCN）进行系统基准，首次量化几何和操作条件的 OOD 泛化；④提供完整的仿真配置、后处理及评估代码，支持快速复现与扩展。

**🔧 技术方法**

技术手段包括：Rhino/Grasshopper 参数化建模、OpenFOAM 体积自由表面 RANS k‑ω CFD、基于树的回归（RF、XGBoost、LightGBM、GB），点云网络（PointNet、RegDGCNN）和网格网络（Mesh‑GCN）；使用 SHAP 解释模型；定义 ID/OOD 分割、数据效率实验；评估指标为 MAE、MSE、R² 与 Max AE。

**📊 数据集**

使用的数据集是 WeirNet：3,794 个三维 PKW 设计（矩形/梯形），每个设计 19 个流量点，生成 71,387 次 CFD 计算，输出排水系数、流速头曲线；同时提供 8 维参数描述、STL 网格、10⁵ 点云以及完整 CFD 结果（70 TB）。

**📈 对比分析**

比较方法：统一在 ID、OOD‑Geom、OOD‑Head 三种拆分上对所有模型进行同一任务的回归；使用 MAE/MSE/R²/Max AE 等指标。结果显示：树回归（RF）在 ID 上最佳；PointNet 与 RegDGCNN 仍能逼近树回归但训练/推理时间更长；Mesh‑GCN 性能最差。OOD‑Head 误差增幅小，OOD‑Geom 误差显著升高，说明几何偏移是主要难点。数据效率实验表明，模型性能在 60% 训练数据后趋于饱和。

**⚠️ 局限性**

局限性：①仅覆盖 PKW Type A、固定全局尺寸、未考虑护墙、鼻部等附加特征；②所有标签来自单一 OpenFOAM k‑ω RANS 模型，可能带来数值误差；③未提供完整流场或不确定性估计，聚焦于标量 c_D；④规模虽大，但仍为实验室尺度，可能受尺度效应影响；⑤缺乏多精度或多物理耦合的扩展。

---

## 219. On the Generalization Behavior of Deep Residual Networks From a Dynamical System Perspective

**arXiv ID:** 2602.20921 | [PDF](https://arxiv.org/pdf/2602.20921v1)

**作者:** Jinshu Huang `[一作]` (Nankai University), Chunlin Wu `[通讯]` (Nankai University)

**通讯引用:** 2192 | [OpenAlex ID](https://openalex.org/A5010997999)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究残差网络在离散时间和连续时间下的泛化误差界，利用动力系统建模与 Rademacher 复杂度理论，并在深层极限下证明 O(1/√S) 的深度一致性泛化上界。

**💡 创新点**

统一给出离散残差网络与其连续时间极限的泛化界；提出一种新的收缩不等式，利用激活函数的结构性分解得到负修正项；在深层极限下保持界的一致性，从而弥补了传统离散和连续分析之间的差距。

**🔧 技术方法**

动力系统建模、Rademacher 复杂度、收缩不等式、Grönwall 不等式、实验训练与评估。

**📊 数据集**

MNIST、CIFAR‑10、CIFAR‑100。

**📈 对比分析**

通过训练/测试损失差估计泛化误差，并与理论 O(1/√S) 曲线拟合；实验表明准确率随样本量增大而提升，随着层数增加训练与测试误差收敛，验证了理论预测；相比传统界限得到更浅层独立且更紧的误差上界。

**⚠️ 局限性**

负修正项的大小依赖于数据相关的常数，难以显式估计；需要对参数空间、数据分布做有界假设；实验仅验证了部分激活函数参数的学习，未深入探讨优化动态对泛化的影响。

---

## 220. Phase-Aware Localization in Pinching Antenna Systems: CRLB Analysis and ML Estimation

**arXiv ID:** 2602.21162 | [PDF](https://arxiv.org/pdf/2602.21162v1)

**作者:** Hao Feng `[一作]` (Hunan Institute of Engineering), Quoc-Viet Pham `[通讯]` (Trinity College Dublin)

**通讯引用:** 16318 | [OpenAlex ID](https://openalex.org/A5062525719)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了利用PINCHING天线系统(PASS)的幅度与相位信息进行用户定位的完整框架。

**💡 创新点**

创新点在于首次将PASS的复基带信号模型与相位信息联合利用，推导出闭式CRLB/PEB，并提出两阶段最大似然定位算法。

**🔧 技术方法**

使用了复基带信号建模、Fisher信息矩阵推导、Cramér–Rao下限/位置误差下限分析以及粗网格+Levenberg–Marquardt的两阶段最大似然估计。

**📊 数据集**

实验基于仿真数据，采用6×10 m²的二维平面、3 m高的波导、2.8 GHz子载波、0.1 W用户发射功率等参数，并平均1000次随机试验。

**📈 对比分析**

与仅利用幅度的加权最小二乘（WLS）基准相比，所提出的相位感知方法在不同噪声功率和天线数量下均实现了更低的定位误差。

**⚠️ 局限性**

局限在于仍存在理论CRLB与实际误差之间的显著性能差距，且算法对初始网格密度和LM算法的收敛依赖较大，未来可进一步优化估计策略。

---

## 221. The Tragedy of the Commons in Multi-Population Resource Games

**arXiv ID:** 2602.20603 | [PDF](https://arxiv.org/pdf/2602.20603v1)

**作者:** Yamin Vahmian `[一作]` (University of Colorado), Keith Paarporn `[通讯]` (University of Colorado)

**通讯引用:** 577 | [OpenAlex ID](https://openalex.org/A5033692956)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了一个双层决策层级的资源提取博弈，其中高层代理决定本地人口的提取速率，低层人口在反馈演化游戏中自发选择合作或背叛行为；在该框架下推导了资源动态、低层收益以及高层的效用函数；随后证明了在满足一定政策约束下，该博弈为凹游戏并且存在唯一的对称纳什均衡，并分析了该均衡随贪婪人口数目变化时资源可持续性的阈值；

**💡 创新点**

创新点主要在于：①首次将多人口反馈演化游戏与高层决策者的提取博弈结合，形成双层层级模型；②给出了该博弈的唯一对称纳什均衡的解析表达式；③通过对均衡的参数空间划分，揭示了在不同环境政策下资源是否会被破坏的临界条件；

**🔧 技术方法**

技术手段包括：反馈演化动力学、复制动力学、凹游戏理论、对称均衡解析、极限分析（M→∞）、以及解析解的数值验证；

**📊 数据集**

本研究为理论分析，未使用真实数据集；

**📈 对比分析**

由于本研究为理论模型，没有与实验或其他方法进行性能比较；

**⚠️ 局限性**

局限性：①假设所有贪婪人口规模相同、策略空间对称；②未考虑人口异质性、不同规模与质量的高层代理；③缺乏实证验证，未对模型参数与现实系统进行校准；

---

## 222. Pipeline for Verifying LLM-Generated Mathematical Solutions

**arXiv ID:** 2602.20770 | [PDF](https://arxiv.org/pdf/2602.20770v1)

**作者:** Varvara Sazonova `[一作]` (Moscow State University), Vasily Motolygin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套可自动或交互式运行的流水线，用来验证大型语言模型（LLM）生成的数学证明，并将其翻译为 Lean4 代码以供证明助手自动证明，进一步生成完整报告。

**💡 创新点**

核心创新是通过 Prompt Engineering 让 Solver LLM 输出严格结构化的证明（按 lemmas 列表），并引入三步 Agent 链（Solver、Translator、Prover）以及一系列脚本处理与检查，显著降低误报并实现了可交互的纠错流程。

**🔧 技术方法**

使用了多模型 LLM（Qwen3‑8B 作为 Solver、Kimina‑Autoformalizer‑7B 作为 Translator、Kimina‑Prover‑Preview‑Distill‑7B 作为 Prover），Lean4 证明助手，自动化脚本、Prompt Engineering、结构化 lemma 处理以及可选的变量引入机制。

**📊 数据集**

主要实验数据集为 Math‑500 及其子集（easy、similar），并使用全量 Math‑500 数据进行评估。

**📈 对比分析**

通过与单纯答案检查（如 Qwen 直接判断正确性）和仅自动化翻译+证明者管线进行对比，发现自动模式在 easy 集上准确率 0.84、精度 0.82、召回 0.95；在 similar 集上精度 0.984、召回 0.609；交互模式可实现 0 FN/0 FP，误报率显著低于传统方法。

**⚠️ 局限性**

主要限制包括：对 Solver 输出结构的高度依赖，几何和高等数学等难以 formalize 的题目难以通过流水线验证；类型错误和漏步骤导致的 False Negatives；LLM 链中任一环节错误会导致整体失效；对变量引入的控制需手动调节；难以推广到更高难度的奥赛题目。

---

## 223. Upper-Linearizability of Online Non-Monotone DR-Submodular Maximization over Down-Closed Convex Sets

**arXiv ID:** 2602.20578 | [PDF](https://arxiv.org/pdf/2602.20578v1)

**作者:** Yiyang Lu `[一作]` (Purdue University), Vaneet Aggarwal `[通讯]` (Purdue University)

**通讯引用:** 6207 | [OpenAlex ID](https://openalex.org/A5064822688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在下闭合凸集上对非单调递减回报子模函数的在线最大化，提出了一种新的结构结果，证明该类函数在特定的指数重新参数化下是1/e线性可化的，从而实现了在线线性优化的简化。

**💡 创新点**

创新点在于首次证明了非单调递减回报子模函数在下闭合凸集上是1/e线性可化的，并设计了投影自由的在线算法，达到了O(T^1/2)的静态遗憾，同时提供了自适应和动态遗憾的保证。

**🔧 技术方法**

使用了指数重新参数化、缩放参数和替代势能的技术，结合了雅可比校正的梯度估计器和在线线性优化的简化。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在多种反馈模型下的算法性能。

**📈 对比分析**

与现有的投影自由方法相比，本文的算法在静态遗憾上达到了O(T^1/2)，而现有方法的最佳静态遗憾为O(T^2/3)。此外，本文还提供了自适应和动态遗憾的保证，显著改善了现有的结果。

**⚠️ 局限性**

限制在于尽管提出了有效的算法，但在处理非单调递减回报子模函数的复杂性和计算效率方面仍然存在挑战，尤其是在更广泛的应用场景中。

---

## 224. Evaluating the Reliability of Digital Forensic Evidence Discovered by Large Language Model: A Case Study

**arXiv ID:** 2602.20202 | [PDF](https://arxiv.org/pdf/2602.20202v1)

**作者:** Jeel Piyushkumar Khatiwala `[一作]` (University of Baltimore), Weifeng Xu `[通讯]` (University of Baltimore)

**通讯引用:** 2022 | [OpenAlex ID](https://openalex.org/A5031477814)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

建立了一套结构化框架，自动化提取、LLM精炼、知识图构建，并评估AI识别数字取证证据的可靠性。

**💡 创新点**

通过UID链路追溯、LLM与数字取证知识图结合，以及基于指标的可验证评估体系，实现可审计、可解释的AI取证流程。

**🔧 技术方法**

采用大语言模型（GPT‑4）进行实体抽取与修正，SQLite/CSV转换、SHA‑256 UID生成，构建DFKG（节点/边图），并使用一系列专门的取证指标。

**📊 数据集**

以Cellebrite 2021/2022 Capture‑the‑Flag（CTF）竞赛提供的13 GB安卓磁盘镜像为主。

**📈 对比分析**

与Cellebrite官方解法及专家人工评估对比，提取准确率95.24%、精确度95.24%、召回率100%，知识图连通性94.44%，链路完整性100%，显示高可靠性。

**⚠️ 局限性**

受限于缺失/加密/碎片化日志导致的上下文不足，阈值过滤可能误删有效证据，且仅单设备实验，需扩展至多设备与真实案例验证。

---

## 225. Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones

**arXiv ID:** 2602.21101 | [PDF](https://arxiv.org/pdf/2602.21101v1)

**作者:** Rong Zou `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**通讯引用:** 37242 | [OpenAlex ID](https://openalex.org/A5057116316)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出统一框架，利用事件流与运动模糊图像共同优化NeRF，实现无人机高速飞行下的高质量辐射场重建。

**💡 创新点**

采用连续时间共享轨迹模块，使图像与事件协同监督，既实现姿态精细化，又在无地面真值的情况下恢复锐利辐射场。

**🔧 技术方法**

结合NeRF、事件相机响应函数、可微渲染、可微姿态优化及事件+模糊图像的多任务损失。

**📊 数据集**

评估使用Ev-DeblurBlender、Ev-DeblurCDAVIS以及自制的Gen3-HandHeld与Gen3-DroneFlight两套高速度场景数据。

**📈 对比分析**

与多种基线（DeblurNeRF、E^2-NeRF、Ev-DeblurNeRF等）对比，实测在真实无人机数据上超过50% PSNR提升，稳健抗姿态噪声。

**⚠️ 局限性**

仅适用于静态场景，无法处理动态物体；依赖事件相机与RGB同步，且对极端光照或高频振动仍有限制。

---

## 226. HieraMAS: Optimizing Intra-Node LLM Mixtures and Inter-Node Topology for Multi-Agent Systems

**arXiv ID:** 2602.20229 | [PDF](https://arxiv.org/pdf/2602.20229v1)

**作者:** Tianjun Yao `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6371 | [OpenAlex ID](https://openalex.org/A5066530136)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种多代理系统框架，利用代理内部的LLM混合与可学习的通信拓扑实现自适应协作。

**💡 创新点**

创新点在于将代理内部的LLM组合与节点间拓扑优化统一，并通过两阶段算法解决信用分配难题。

**🔧 技术方法**

采用强化学习、图卷积网络、分层奖励和两阶段训练策略。

**📊 数据集**

在 HumanEval++、MATH、MMLU‑Redux 三个基准上进行实验。

**📈 对比分析**

与单代理和多代理基线比较，取得 94.61% 的平均准确率，显著优于 Full‑Graph、AFlow 等方法，并降低成本。

**⚠️ 局限性**

仍受限于离散拓扑候选集、对超大规模任务的可扩展性和跨域泛化的进一步验证。

---

## 227. Benchmarking Early Deterioration Prediction Across Hospital-Rich and MCI-Like Emergency Triage Under Constrained Sensing

**arXiv ID:** 2602.20168 | [PDF](https://arxiv.org/pdf/2602.20168v1)

**作者:** KMA Solaiman `[一作]` (University of Maryland Baltimore County), Karma Tobden `[通讯]` (University of Maryland Baltimore County)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对急诊科分诊阶段的早期病情恶化预测，提出了泄漏控制、可复现的基准框架，并在医院资源充足与MCI（灾难）式受限感知两种情境下进行对比评估。

**💡 创新点**

创新点在于：①在首小时内严格限制可用特征，消除后续干预与多次就诊的泄漏；②对比医院丰富与仅凭生命体征两种感知范式；③通过系统性生命体征消融与SHAP解释，明确呼吸与氧合信号为最关键特征；④提供完整的可复现脚本与固定数据拆分，促进跨研究对比。

**🔧 技术方法**

使用的技术包括：数据清洗与时间对齐、患者级去重、特征归一化/填充、模型基线（Logistic Regression、Random Forest、XGBoost、LightGBM、TabNet）、AUPRC/AUROC评估、5次患者级分层拆分、SHAP可解释性分析、结构化特征消融。

**📊 数据集**

使用公开的 MIMIC-IV-ED 4.2 版（约10,000名首次 ED 访问病例）构建的数据集，包含生命体征、观察记录、早期实验室、文本嵌入等，严格限定在到达后1小时内可获取的信息。

**📈 对比分析**

对比方法：将所有特征（医院丰富）与仅生命体征（MCI-like）两种特征集分别训练上述模型；使用5次随机种子训练-验证-测试拆分，报告平均 AUROC/AUPRC。结果显示：在医院丰富下，XGBoost AUROC ≈0.81、AUPRC≈0.38；在仅生命体征下，AUROC≈0.75、AUPRC≈0.30，降幅仅为约10%；模型顺序保持稳定，非线性集成模型始终优于线性模型。

**⚠️ 局限性**

局限性包括：仅做一次性快照预测，未考虑随访时间序列；评估范围仅限 MIMIC-IV-ED 内部，缺乏外部或现场数据验证；未使用递归/序列模型；缺少对传感器失效、延迟测量等真实灾难情境的模拟。

---

## 228. LogicGraph : Benchmarking Multi-Path Logical Reasoning via Neuro-Symbolic Generation and Verification

**arXiv ID:** 2602.21044 | [PDF](https://arxiv.org/pdf/2602.21044v1)

**作者:** Yanrui Wu `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 74325 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们构建了LogicGraph benchmark和神经符号化生成与验证管线，用于评估多路径逻辑推理能力。

**💡 创新点**

创新点包括：通过逆向DAG构造实现完整的多路径真值集、参考无关的神经符号评估器、以及聚焦收敛与发散两维度的评估指标。

**🔧 技术方法**

主要技术有：逆向逻辑DAG生成、语义实例化（利用LLM填充抽象实体）、Prover9逻辑求解器、LLM作为自然语言到形式逻辑的翻译器。

**📊 数据集**

使用的数据集为自研的LogicGraph，包含900个样本，按路径数分为小、中、大三类，每个实例都提供完整的最小证明集合。

**📈 对比分析**

对比方法：基于我们参考无关评估框架对GPT-5.1、Gemini-3-Pro等SOTA LLM进行评测，发现推理型模型在收敛指标上优于通用模型，但在多路径覆盖（发散指标）上仍显著不足，尤其随推理深度增加差距扩大。

**⚠️ 局限性**

局限性包括：数据为合成，缺少真实世界的不确定性；验证过程计算量大；并且受限于离散逻辑框架，可能未覆盖模糊与概率推理场景。

---

## 229. The Diffusion Duality, Chapter II: $Ψ$-Samplers and Efficient Curriculum

**arXiv ID:** 2602.21185 | [PDF](https://arxiv.org/pdf/2602.21185v1)

**作者:** Justin Deschenaux `[一作]` (Ecole Polytechnique Federale de Lausanne), Subham Sekhar Sahoo `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 Ψ-后验的预测-校正(Predictor-Corrector)采样器，适用于离散扩散模型中的任意噪声先验，并对统一状态扩散模型（USDM）进行改进；同时设计了内存高效的课程学习（Curriculum）训练策略，显著减少了训练时间和显存。

**💡 创新点**

创新点在于：①将前向扩散过程与后向后验线性叠加，得到 Ψ-后验，既保持了原始扩散的边缘分布，又引入了校正项，能够在采样中进行自我纠错；②该框架统一了之前的 ReMDM 等掩码扩散 PC 方法，扩展到统一状态扩散；③提出了利用 softmax 稀疏化与 top‑k 近似的高效课程学习，减少 33% 内存、25% 训练时长。

**🔧 技术方法**

技术主要包括：离散扩散模型（USDM 与 MDM）、Ψ-后验预测-校正采样器、可调的 κ_t 采样调度、nucleus 采样、连续时间马尔科夫链（CTMC）理论、Gaussian relaxation 课程学习、top‑k softmax 近似、梯度下降训练、CFG（无分类器引导）等。

**📊 数据集**

数据集：语言建模使用 OpenWebText 与 LM1B；图像生成使用 CIFAR‑10；下游任务评估包括多选题集（ARC‑Easy/Challenge、HellaSwag、Winogrande、PIQA、OpenBookQA 等）。

**📈 对比分析**

与 MDLM、ReMDM、祖先采样（Ancestral）等基线比较，Ψ-采样器在相同 NFE 下获得更低的生成困惑度（Gen. PPL）和更高的单字熵；在 CIFAR‑10 上取得更低的 FID 与更高的 IS；且性能随 NFE 继续提升，而祖先采样在 NFE 超过序列长度后趋于平稳。训练方面，新的课程学习与 USDM 在 LM1B/OWT 上的验证困惑度与 Duo 相当，同时显存减少 33%、训练时间缩短 25%。

**⚠️ 局限性**

局限性：①与 MDM（尤其是 ReMDM）相比，USDM 的困惑度仍略高，导致在部分下游任务上表现落后；② Ψ-后验的采样效率受 κ_t 调度与核函数选择影响，需手动调参；③高阶采样器或更复杂的后验近似尚未探索；④课程学习的 top‑k 近似对极大词表的鲁棒性尚待进一步验证。

---

## 230. Exploiting Low-Rank Structure in Max-K-Cut Problems

**arXiv ID:** 2602.20376 | [PDF](https://arxiv.org/pdf/2602.20376v1)

**作者:** Ria Stevens `[一作]` (Rice University), Anastasios Kyrillidis `[通讯]` (Rice University)

**通讯引用:** 1781 | [OpenAlex ID](https://openalex.org/A5024280658)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出利用目标矩阵低秩结构，针对 Max‑3‑Cut（以及其复数形式的离散二次最大化）设计了可并行化的低秩算法，能在秩为 r 的情况下在多项式时间内给出全局最优解，且在低秩或近似低秩矩阵下给出近似保证。

**💡 创新点**

创新点在于：① 通过对复数根号单位集合的相位分区，将极大化问题转化为在有限个“候选区间”内枚举解，从而在秩 1 时仅需 O(n²) 复杂度；② 对一般秩 r 的情形构造多维“分区面”并枚举其交点，得到多项式规模的候选集合；③ 给出低秩近似与噪声扰动下的加法与乘法误差理论；④ 在大规模图上实现了基于 Ray 的分布式并行求解。

**🔧 技术方法**

主要技术包括：矩阵谱分解（求主特征向量/子空间）、低秩近似（截断奇异值分解）、复数根号单位的相位划分、基于分区面的点集枚举、离散候选解评估与最大化、以及 Ray 框架实现的任务并行。

**📊 数据集**

使用的数据集有：① 规模为 20/50/100 的 Erdős‑Rényi 随机图和 d‑regular 随机图；② GSet 基准集（71 个图，800–20,000 节点，包含 Erdos‑Rényi、4‑regular 托洛尔图、带偏度分布的随机图）；③ 大型 3‑regular 随机图（50k、100k 节点）用于极限规模测试。

**📈 对比分析**

与 Frieze‑Jerrum SDP、Greedy、Genetic、MOH 等基线对比。实验显示：在结构化图（如 Torus）上 Rank‑1 算法获得最优/近最优切割，速度比 Greedy 轻 10–70 倍；在小型稠密图上 Rank‑2/3 算法逼近 Greedy；在大规模随机图上 Rank‑1 仍优于 Random，且能在数小时内完成。

**⚠️ 局限性**

局限性包括：① 需要目标矩阵具有低秩或近似低秩特性；② 对稀疏随机图的性能下降，低秩近似无法捕捉足够的图结构；③ Rank‑3 及更高秩版本在大 n 下计算量大，导致实际使用受限；④ 对非常大规模图，主成分求解（谱分解）仍是瓶颈。

---

## 231. IG-RFT: An Interaction-Guided RL Framework for VLA Models in Long-Horizon Robotic Manipulation

**arXiv ID:** 2602.20715 | [PDF](https://arxiv.org/pdf/2602.20715v1)

**作者:** Zhian Su `[一作]` (Zhejiang University), Huixu Dong `[通讯]` (Zhejiang University)

**通讯引用:** 793 | [OpenAlex ID](https://openalex.org/A5016175741)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 IG‑RFT，一个互动引导强化学习框架，用于对流式 Vision‑Language‑Action (VLA) 模型在真实世界长周期复杂机械操作任务中的微调。

**💡 创新点**

创新点包括：① 交互引导优势加权回归 IG‑AWR，可根据机器人交互状态动态调节探索噪声；② 混合密集奖励，将轨迹级与子任务级奖励结合，提供全局与局部密集反馈；③ 三阶段训练策略（SFT→离线 RL→人机交互 RL），实现从领域适应到高效样本利用再到鲁棒提升的完整流程。

**🔧 技术方法**

技术实现基于流式 VLA 模型 π₀.₅、Q‑Former 形式的 critic、IG‑AWR 算法、离线强化学习与 Human‑in‑the‑Loop 交互；交互信号通过机器人掩膜与光流估计得到；混合奖励由轨迹级和子任务级分量构成。

**📊 数据集**

实验使用每个任务 60 条专家演示作为初始数据集，并在 Human‑in‑the‑Loop 阶段收集真实回放；基线方法（SFT、IQL、AWR）采用 100 条专家演示以保证公平对比。

**📈 对比分析**

在四个长周期任务上，IG‑RFT 的平均成功率达 85.0%，显著优于 SFT 的 18.8% 以及离线 RL 基线（≈40%），在关键任务上提升 30–50%；实验还显示 IG‑RFT 的样本效率更高，40 条人工干预即可达到 80% 以上成功率。

**⚠️ 局限性**

局限性：仍需人工演示和交互干预；交互信号判定依赖光流与掩膜，易受视觉噪声影响；方法在更高维动作空间或更复杂任务的通用性尚待验证，且混合奖励对任务结构有一定依赖。

---

## 232. Grounding LLMs in Scientific Discovery via Embodied Actions

**arXiv ID:** 2602.20639 | [PDF](https://arxiv.org/pdf/2602.20639v1)

**作者:** Bo Zhang `[一作]` (Tsinghua University), Hongning Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5177 | [OpenAlex ID](https://openalex.org/A5085094109)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EmbodiedAct 框架，将大型语言模型（LLM）从被动工具调用转变为主动具身代理，能够在 MATLAB/Simulink 真实物理仿真中持续感知、执行并实时干预。

**💡 创新点**

核心创新在于建立紧密的感知-执行闭环，利用 Runtime Perception Engine 监测连续仿真状态，触发 Hot-Fix Loop 并通过 Reflective Decision Maker 实时调整计划，从而实现过程导向的科学发现。

**🔧 技术方法**

技术包括：LLM 规划与代码生成（Strategic Planner、Primitive Generator）、实时多模态观察（Asynchronous State Synchronization Protocol）、异常检测与自我修正机制、以及对 MATLAB 具体仿真原语的深度融合。

**📊 数据集**

使用 EngDesign（涵盖 9 个工程领域，共 92 问题，45 需要 MATLAB 仿真）和 SciBench-107（107 个科学问题）两大基准数据集进行实验。

**📈 对比分析**

与生成式模型和 CodeAct 代码即行动基线对比，EmbodiedAct 在 EngDesign Core 组平均 Pass Rate、得分和 SciBench-107 准确率上均实现 70%+ 的 SOTA 性能，且在多试验可靠性、稳定性和零区间分布上显著优于对手。

**⚠️ 局限性**

局限性包括：仍以仿真验证为主，存在“仿真剥削”风险；对实时多模态反馈的利用尚停留在文本层面，未充分挖掘模型多模态能力；以及对不同仿真后端的通用性和安全性需要进一步评估。

---

## 233. SegSEM: Enabling and Enhancing SAM2 for SEM Contour Extraction

**arXiv ID:** 2602.20471 | [PDF](https://arxiv.org/pdf/2602.20471v1)

**作者:** Da Chen `[一作]` (Huawei Technologies), Mingxuan Yuan `[通讯]` (HKUST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SegSEM 框架，结合 SAM2 微调与传统阈值回退，实现高精度 SEM 图像二维轮廓提取。

**💡 创新点**

① 仅微调 SAM2 的编码器，实现极少量样本适配；② 将 SAM2 与置信度回退的传统算法融合，保证系统鲁棒性；③ 设计多项自定义损失（IoU 奖励、计数惩罚、边缘误差惩罚）提升边界精度。

**🔧 技术方法**

使用 SAM2 模型、选择性微调、基于随机点提示的训练、形态学后处理、Sauvola 自适应阈值回退、Sobel 边缘检测等技术。

**📊 数据集**

60 张工业生产 SEM 图像（50 张训练、10 张测试），涵盖多种曝光条件。

**📈 对比分析**

与传统 Sauvola 方法、SAM2-Only 与全微调模型对比，使用 IoU、Precision、Recall、F1 等指标。SegSEM 平均 IoU 0.884、F1 0.941，分别比传统方法提升 13.07% 和 7.58%，回退触发率仅 7.2%。

**⚠️ 局限性**

未评估对 OPC 下游任务的直接影响；数据规模仍偏小，缺乏更广泛曝光条件验证；未深入探讨实时性与大规模部署可行性。

---

## 234. MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Elastic LLMs

**arXiv ID:** 2602.20191 | [PDF](https://arxiv.org/pdf/2602.20191v1)

**作者:** Dongwei Wang `[一作]` (Panasonic AI Lab), Huanrui Yang `[通讯]` (University of Arizona)

**通讯引用:** 1509 | [OpenAlex ID](https://openalex.org/A5076154259)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种混合位量化（MoBiQuant）框架，实现了针对每个token的自适应可弹性推理；

**💡 创新点**

创新点在于：1）多层递归残差量化（many-in-one MoBiSlice）将权重拆解为多份低精度切片；2）基于token感知的二值路由器（MoBiRoute）动态选择所需的切片，实现精度与吞吐量的细粒度平衡；3）通过阈值调度和正则化实现精度弹性与token级别的最优切片分配；

**🔧 技术方法**

使用的技术包括：后训练量化（PTQ）与OmniQuant作为后端，递归残差量化、二值路由器、温度退火的二值化门控、正则化调度、基于位图的BMMA内核、位切片-Token重排与多流并行；

**📊 数据集**

主要使用的评估数据集为WikiText-2（校准与Perplexity评估），并在六个零样本推理任务（BoolQ、PIQA、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge）进行验证；

**📈 对比分析**

与静态PTQ方法（RTN、SmoothQuant、AWQ、GPTQ、SpinQuant、OmniQuant）以及任何精度量化基线（AnyPrecisionLLM、AnyBCQ）进行比较；MoBiQuant在LLaMA2/3系列上匹配或优于静态PTQ，在2–3位精度下仍保持平滑性能，且在A100 GPU上可实现最高2.7×的加速；

**⚠️ 局限性**

局限性包括：需在特定校准集上训练，可能对校准数据分布敏感；目标精度的超参数需要手动设置；目前仅实现权重量化，对激活量化的支持有限；实现复杂度高，需专门的低位内核与内存布局优化；在极低位（<2位）下的泛化能力尚待进一步验证。

---

## 235. Tool Building as a Path to "Superintelligence"

**arXiv ID:** 2602.21061 | [PDF](https://arxiv.org/pdf/2602.21061v1)

**作者:** David Koplow `[一作]`, Tomaso Poggio `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于GF(2)电路重构的对抗性基准，用于测量大型语言模型在多步推理过程中的逐步成功概率γ

**💡 创新点**

创新点在于：①将每一步的正确延伸唯一化，消除模式匹配与记忆的捷径；②通过统计去隐蔽抽样oracle让模型必须整合前缀信息与新观测；③对Diligent Learner框架进行实证验证，揭示工具调用对保持γ的关键作用

**🔧 技术方法**

使用的技术包括：Transformer LLM（如Qwen3、GPT‑5、Claude Opus、Gemini 3 Pro），vLLM推理，正则表达式解析，精确‑匹配验证，工具调用与验证器引导的深度优先搜索

**📊 数据集**

数据集为自生成的GF(2)电路实例，随机采样支持集S_1…S_n，固定Hamming重量w，生成带遮蔽标签的K=32步特定样本集合

**📈 对比分析**

对比方法：对四类估计器（完整、仅数据、仅历史、部分）以及不同规模LLM；评估指标为每步准确率γ_g，结果显示小模型随深度急剧下降，而前沿模型（尤其使用工具）保持高γ且衰减缓慢；可视化为曲线与热图显示性能差异

**⚠️ 局限性**

局限性包括：①实验多依赖自生成数据，缺少真实世界复杂性；②对工具调用的依赖表明模型内部推理仍受限；③仅评估一步成功概率，未完整证明搜索效率；④实验规模有限，缺乏对更大深度和多样任务的系统验证

---

## 236. Turning Semantics into Topology: LLM-Driven Attribute Augmentation for Collaborative Filtering

**arXiv ID:** 2602.21099 | [PDF](https://arxiv.org/pdf/2602.21099v1)

**作者:** Junjie Meng `[一作]` (University of Science and Technology of China), Chao Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 43639 | [OpenAlex ID](https://openalex.org/A5100407048)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用大型语言模型推断用户-项目交互背后的意图，生成中介属性节点，构建用户-属性-项目三元图，并通过图卷积模型学习其表示。

**💡 创新点**

将LLM抽取的语义信息转化为拓扑连通结构而非传统文本嵌入，并提出自适应关系加权图卷积（ARGC）在多关系图上动态估计并融合不同类型边的重要性。

**🔧 技术方法**

核心技术包括大型语言模型（DeepSeek V3.1等）进行属性抽取、基于图神经网络的U-A-I图学习、ARGC机制、BPR损失以及属性过滤与语义融合策略。

**📊 数据集**

实验数据集涵盖公开的 Amazon Book、Amazon Office 与 Yelp 推荐基准。

**📈 对比分析**

与 LightGCN、SGL、SimGCL 等骨干模型以及 KAR、RLMRec、AlphaRec 等 LLM 增强基线对比，在所有组合下实现 Recall@5/20、NDCG@5/20 提升约 8%–22%，并在稀疏数据与冷启动实验中保持明显优势。

**⚠️ 局限性**

限制在于依赖LLM提示与抽取质量，属性过滤阈值需手工调参；对属性丰富度低或语义噪声高的数据集时，效果可能不如预期。

---

## 237. Insertion Correcting Capability for Quantum Deletion-Correcting Codes

**arXiv ID:** 2602.20635 | [PDF](https://arxiv.org/pdf/2602.20635v1)

**作者:** Ken Nakamura `[一作]` (Yamaguchi University), Takayuki Nozaki `[通讯]` (Yamaguchi University)

**通讯引用:** 1212 | [OpenAlex ID](https://openalex.org/A5011452126)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了任何量子 t‑删除纠错码在满足误差球体不相交条件下也能纠正总共 t 个插入和删除错误，并基于此引入了量子 indel 距离来刻画插入/删除错误的纠错能力；

**💡 创新点**

创新点在于将经典删除/插入球体概念推广到量子状态，给出了删除错误纠错与插入+删除错误纠错等价的定理；首次提出量子 indel 距离，并通过最小距离阈值 d_min≥2t+1 判定量子码的插入/删除错误容忍度；

**🔧 技术方法**

主要技术包括量子误差理论中的 Knill‑Laflamme 条件、误差球体不相交原理、混合态插入错误的显式解析、组合错误归约以及度量空间的三角不等式证明；

**📊 数据集**

本文不依赖实验数据，全部采用理论证明与符号演算，无需使用任何数据集；

**📈 对比分析**

通过示例量子单删纠错码 X₂ 计算 d_min=4，满足 d_min≥2·1+1，进而证明其能纠正一次插入错误；该判据与经典的最小距离方法对应，性能上可直接由 d_min 推断纠错能力；

**⚠️ 局限性**

限制主要在于反向不成立，即量子插入纠错码不一定能纠正删除错误；此外，本文的结果依赖于误差球体不相交假设，在非纯态或更一般的混合误差模型下需进一步研究；

---

## 238. CAD-Prompted SAM3: Geometry-Conditioned Instance Segmentation for Industrial Objects

**arXiv ID:** 2602.20551 | [PDF](https://arxiv.org/pdf/2602.20551v1)

**作者:** Zhenran Tang `[一作]` (Robotics Institute Carnegie Mellon University), Changliu Liu `[通讯]` (Robotics Institute Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了基于CAD模型的单阶段 promptable 语义分割框架——CAD‑Prompted SAM3，利用CAD网格的多视角渲染作为几何提示，直接驱动实例掩码预测；

**💡 创新点**

创新点在于：①将 CAD 的多视角渲染引入 SAM3 的 prompt 机制，形成几何条件的跨图像提示；②设计了 geometry‑encoder 与跨视角融合 transformer 的组合，解决不同域提示与查询图像特征的对齐；③构建了大规模随机化的合成训练管线，使模型能在不同材质、光照与背景下依赖几何而非外观；

**🔧 技术方法**

技术包括：SAM3 的图像编码器、融合 Transformer、检测与掩码解码器；多视角渲染（Blender）、点采样提取几何提示；训练阶段采用两阶段损失（score‑weighted mask + one‑to‑many 匹配）；使用 ADR 随机化的 Isaac Sim 场景合成数据；

**📊 数据集**

使用了 9000+ CAD 模型的合成数据集（来自 ABC Dataset）和 8 只 3D 打印对象的真实图像集；在 T‑LESS 与 ITODD 两个工业基准数据集上也进行了评估；

**📈 对比分析**

与 Matcher、PerSAM、SAM3 的同图像示例提示等方法对比，CAD‑Prompted SAM3 在自定义 3D 打印数据集上 PQ 0.7385、F1 0.7636；在 T‑LESS 上 PQ 0.2799、F1 0.3329；在 ITODD 上 PQ 0.4921、F1 0.6090，均显著优于基线；

**⚠️ 局限性**

局限性包括：依赖 CAD 的多视角渲染，需预先渲染和编码；跨域提示对齐仍可能受限于渲染质量；对极端遮挡或复杂背景下的实例识别尚有提升空间；未来可探索直接几何编码（网格/点云/隐式表面）替代渲染。

---

## 239. Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization

**arXiv ID:** 2602.20718 | [PDF](https://arxiv.org/pdf/2602.20718v1)

**作者:** Yangsen Chen `[一作]` (Hong Kong University of Science and Technology), Hao Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 41389 | [OpenAlex ID](https://openalex.org/A5080102032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过多层几何正则化的3D高斯散射框架实现实时、可高质量的动态内镜软组织重建。

**💡 创新点**

结合表面感知重建和半刚性变形约束，利用网格约束高斯分布以及局部刚性和全局非刚性限制，提升表面平滑度与物理可行性。

**🔧 技术方法**

使用3D Gaussian Splatting、Signed Distance Field (SDF) / NeuS2、光流与视频填充、SIFT特征匹配、ARAP损失、旋转一致性与等距约束等技术。

**📊 数据集**

在公开的ENDONERF和SCARED内镜数据集上进行实验。

**📈 对比分析**

与EndoNeRF、EndoSurf、LerPlane、EndoGS、EndoGaussian等方法对比，PSNR/SSIM/LPIPS均领先，训练时间约2分钟/帧，实时渲染速率>60FPS。

**⚠️ 局限性**

仍受单视角深度信息不足和工具遮挡影响，极端变形场景下可能出现细节漂移。

---

## 240. Personal Information Parroting in Language Models

**arXiv ID:** 2602.20580 | [PDF](https://arxiv.org/pdf/2602.20580v1)

**作者:** Nishant Subramani `[一作]` (Carnegie Mellon University), Mona Diab `[通讯]` (Carnegie Mellon University)

**通讯引用:** 36576 | [OpenAlex ID](https://openalex.org/A5091175785)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于正则表达式与后处理规则的个人信息（PI）检测工具套件，并使用它对大型预训练模型（Pythia系列）进行PI“变声”记忆实验，量化了模型对电子邮件、IP地址和电话号码的逐字记忆程度。

**💡 创新点**

创新点在于：1）改进并扩展了现有正则表达式，加入IPv6、带国家码的电话号码等新模式；2）结合上下文规则进一步提升检测精度；3）首次使用手工标注的PI实例集合，对模型的逐字记忆率进行系统测评，揭示模型规模、预训练步数与前缀长度对记忆的影响。

**🔧 技术方法**

技术主要包括：正则表达式（regex）、后处理规则（contextual filtering）、Levenshtein相似度评估（用于计算记忆率），以及在Pythia模型上进行前缀提示+贪婪解码的实验。

**📊 数据集**

使用的数据集为公开的Pythia预训练语料——Pile（383B token），以及从Pile中提取的483条手工标注的PI实例。实验涵盖了Pythia 160M、410M、1B、1.4B、2.8B、6.9B等多种规模的模型。

**📈 对比分析**

检测器与现有最强基线（仅regex的检测器）做精度对比，结果显示：在17/20的评估子类别中，新工具精度显著提升；尤其对含+1国家码的美国/加拿大电话号码精度从0提升到约0.3。记忆率方面，电子邮件在所有模型中最易被完全记忆，最高约19.6%；IP地址次之（最高14.2%）；电话号码记忆率低且与模型规模关系不大。模型规模、预训练步数与前缀长度均对记忆率正相关。

**⚠️ 局限性**

局限性包括：①手工标注PI样本耗时，标注规模有限（仅483条真实实例）；②实验仅在Pythia系列模型上进行，缺乏对其他主流模型的验证；③正则+规则方案可能仍产生误报（如十位数字序列被误判为电话号码），且未能对所有Pythia预训练子集进行细粒度过滤。

---

## 241. Rapid Testing, Duck Lips, and Tilted Cameras: Youth Everyday Algorithm Auditing Practices with Generative AI Filters

**arXiv ID:** 2602.20314 | [PDF](https://arxiv.org/pdf/2602.20314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 242. GENSR: Symbolic Regression Based in Equation Generative Space

**arXiv ID:** 2602.20557 | [PDF](https://arxiv.org/pdf/2602.20557v1)

**作者:** Qian Li `[一作]` (Shanghai Jiao Tong University), Yuntian Chen `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 1902 | [OpenAlex ID](https://openalex.org/A5101707444)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于条件变分自编码器的连续生成潜在空间，并在该空间中通过粗定位+精细搜索实现符号回归。

**💡 创新点**

首次将符号回归表述为最大化p(Equ.|Num.)的贝叶斯优化，并通过双分支CVAE实现全局符号连续性与局部数值平滑的潜在空间。

**🔧 技术方法**

采用双分支Transformer编码器-解码器的CVAE、改进的CMA-ES、ELBO优化、KL退火等技术。

**📊 数据集**

使用约500万条合成方程-样本对预训练模型，评估数据集为SRBench（Feynman、Strogatz、黑盒回归）。

**📈 对比分析**

与18种基线（GP、MCTS、E2ESR、SNIP等）在R²、时间复杂度、表达式复杂度三指标上做Pareto前沿比较，GenSR在三项均表现最佳，并在噪声条件下保持最高鲁棒性。

**⚠️ 局限性**

潜在维度较高时CMA-ES仍显成本，缺乏物理约束与更丰富的生成先验，对极端噪声或极高维复杂方程的适用性仍待验证。

---

## 243. When can we trust untrusted monitoring? A safety case sketch across collusion strategies

**arXiv ID:** 2602.20628 | [PDF](https://arxiv.org/pdf/2602.20628v1)

**作者:** Nelson Gardner-Challis `[一作]` (LASR Labs), Charlie Griffin `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对不受信监控的安全案例草图，包含协同策略分类、实验评估与统计模型；

**💡 创新点**

创新点在于引入主动/被动、因果/非因果以及组合协同策略的系统分类，使用监督微调进行保守能力估计，并构建了可直接从估计能力推断安全性的统计模型；

**🔧 技术方法**

技术方面采用了监督微调（SFT）、提示式评估、信号消除（最小化、摘要）、统计风险建模，并在多模型（GPT‑4.1、GPT‑4o mini、Gemini 2.5 Flash）间部署不受信监控协议；

**📊 数据集**

使用的评估数据集为代码生成基准APPS和BigCodeBench，定义后门为通过测试用例但在特定输入下偏离参考解的程序；

**📈 对比分析**

通过比较不同协同策略与蓝队防御（最小化、摘要、无改动）在APPS和BigCodeBench上的安全率，发现自我识别+SFT为最强红队策略，摘要为最优蓝队响应，安全率在84%–97%之间变化；

**⚠️ 局限性**

局限性包括未覆盖所有可能的协同策略（尤其是非因果协同）、对滑动蜜罐数量的估计不足、独立性假设过简化、未建模监视器越狱攻击，以及仅在代理环境下验证，缺乏真实部署数据。

---

## 244. Discrete Diffusion with Sample-Efficient Estimators for Conditionals

**arXiv ID:** 2602.20293 | [PDF](https://arxiv.org/pdf/2602.20293v1)

**作者:** Karthik Elamvazhuthi `[一作]` (Los Alamos National Laboratory), Andrey Y. Lokhov `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 1385 | [OpenAlex ID](https://openalex.org/A5084142327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种离散扩散框架，结合轮询单点噪声和NeurISE条件估计器，实现高维分类数据的高效生成

**💡 创新点**

创新点在于用单点条件分布代替全局分数构造逆扩散过程，并通过NeurISE实现样本高效的条件学习；同时给出总变差误差传播界限，并证明硬噪声极限等价自回归生成

**🔧 技术方法**

使用轮询单点噪声、Neural Interaction Screening Estimator、总变差误差分析、可微条件估计网络、与D3PM、SEDD等方法对比实验

**📊 数据集**

实验数据集包括25变量Ising（Edwards-Anderson）、MNIST（二值化）、D-Wave量子退火数据、Potts模型以及GHZ量子态

**📈 对比分析**

通过与D3PM、SEDD在总变差、MMD、交叉相关等指标对比，NeurISE扩散在小样本场景下TV下降最快、MMD最低，整体性能优于对照方法

**⚠️ 局限性**

局限性包括需为每个时间步训练条件网络导致训练成本较高；对极高维稀疏数据的表现未充分验证；逆核近似误差可能累积，限制生成多样性

---

## 245. SOM-VQ: Topology-Aware Tokenization for Interactive Generative Models

**arXiv ID:** 2602.21133 | [PDF](https://arxiv.org/pdf/2602.21133v1)

**作者:** Alessandro Londei `[一作]` (Sony Computer Science Laboratories), Matteo Benati `[通讯]` (Sapienza University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种将自组织映射与向量量化结合的 SOM-VQ 代码本，用于生成模型的离散化与可解释性控制。

**💡 创新点**

创新点在于给离散码本赋予低维拓扑结构，利用邻域权重更新保持语义相似性，可通过几何操作直接在 token 空间实现控制。

**🔧 技术方法**

采用 VQ‑VAE 框架、Self‑Organizing Map、指数移动平均 (EMA) 代码本更新、两阶段（拓扑+承诺）训练、以及自动回归模型（GRU、LSTM）进行实验。

**📊 数据集**

使用 Lorenz 动力学三维数据和 AIST++ 51 维动作捕捉数据集进行验证。

**📈 对比分析**

与 VQ、SOM‑hard、VQ‑VAE 等方法对比，评估点级拓扑 (Trust/Cont)、MSE、序列困惑度 (Seq‑PPL) 和扭曲度，SOM‑VQ 在两种任务上均获得最低 Seq‑PPL 并保持良好的拓扑结构，表现优于对比方法。

**⚠️ 局限性**

局限性包括仅在两种域验证，缺乏多模态（图像、音频等）评估，用户交互实验尚浅，对比实验中存在网络容量等混杂因素。

---

## 246. Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting

**arXiv ID:** 2602.20933 | [PDF](https://arxiv.org/pdf/2602.20933v1)

**作者:** Shuangkang Fang `[一作]` (Beihang University), Takeo Igarashi `[通讯]` (University of Tokyo)

**通讯引用:** 11385 | [OpenAlex ID](https://openalex.org/A5102743150)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种锚点式Dropout策略，在3D Gaussian Splatting中同时删除锚点及其邻域的高阶球面谐波系数，以抑制稀视图下的过拟合。

**💡 创新点**

创新点在于将Dropout从单个高斯迁移到空间区域，消除邻域补偿，并将Dropout扩展到颜色属性的高阶球面谐波，从而实现更强的正则化。

**🔧 技术方法**

采用3D Gaussian Splatting框架，利用空间最近邻搜索实现锚点选择、邻域删除，并对球面谐波系数按阶进行Dropout，同时保持标准的L1+SSIM损失。

**📊 数据集**

在LLFF、MipNeRF-360和Blender三个公开数据集上进行实验，分别在3视、6视、12视等稀视图设置下验证。

**📈 对比分析**

与DropGaussian、DropoutGS以及其他3DGS/NeRF稀视图方法对比，实验表明该方法在PSNR、SSIM、LPIPS等指标上平均提升1–2 dB，且训练时间增加不到3%。

**⚠️ 局限性**

局限性包括需手动调参（锚点采样率、邻域大小、SH Dropout概率），以及在极端稀视图或高复杂度场景中仍可能出现几何细节缺失。

---

## 247. QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models

**arXiv ID:** 2602.20309 | [PDF](https://arxiv.org/pdf/2602.20309v1)

**作者:** Jingxuan Zhang `[一作]` (Indiana University), Mi Zhang `[通讯]` (Ohio State University)

**通讯引用:** 6062 | [OpenAlex ID](https://openalex.org/A5100675021)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 QuantVLA，一种训练‑free 的后训练量化（PTQ）框架，用于 Vision‑Language‑Action（VLA）模型，特别对 Diffusion Transformer（DiT）行为头实现低比特量化，保持原始架构不变。

**💡 创新点**

创新点包括：①首个对 VLA 系统（含 DiT 行为头）进行 PTQ 的方法；②设计选择性量化布局（仅对语言主干与 DiT 的 MLP 进行整数化，注意力投影保持浮点）；③提出轻量化校准机制 Attention Temperature Matching (ATM) 与 Output Head Balancing (OHB) 以补偿量化导致的温度漂移和残差能量漂移；④在不增加额外算子或训练的前提下实现约 70% 内存节省并在多项任务上超过全精度基准。

**🔧 技术方法**

技术包括：post‑training quantization (W4A8/W4A4)、DuQuant 重参数化、选择性整数化、ATM 与 OHB 校准、校准缓冲区采样、量化尺度折叠，所有操作保持原始执行顺序与整数 GEMM。

**📊 数据集**

使用 LIBERO 仿真器的四个任务套件（Spatial、Object、Goal、Long）评估，补充了 Simpler 机器人操作基准进行跨任务鲁棒性验证。

**📈 对比分析**

与全精度 FP16、DuQuant、SmoothQuant 等方法对比；在 OpenPI π0.5 上取得 97.6% 成功率（与基准相当甚至略优）并将内存从 4.27 GB 降至 1.28 GB；在 GR00T N1.5 上取得 88.0% 成功率，内存从 2.02 GB 降至 0.91 GB；在更低位宽 W4A4 仍保持 95.3% 成功率；整体表现优于现有 PTQ 方法。

**⚠️ 局限性**

局限性：目前仅针对含 DiT 行为头的 VLA 体系结构；需使用少量未标记数据进行校准；对其他 VLA 结构（如非 DiT 行为头）或更复杂的多任务环境的泛化仍需进一步验证；低精度与长步长下的鲁棒性仍有提升空间。

---

## 248. Disentangling Geometry, Performance, and Training in Language Models

**arXiv ID:** 2602.20433 | [PDF](https://arxiv.org/pdf/2602.20433v1)

**作者:** Atharva Kulkarni `[一作]` (University of Southern California), Swabha Swayamdipta `[通讯]` (University of Southern California)

**通讯引用:** 3209 | [OpenAlex ID](https://openalex.org/A5076880940)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究 Transformer 未嵌入矩阵的几何性质（尤其是有效秩），并在 108 个 OLMo 模型上评估其与预训练、泛化、微调、灾难性遗忘和量化等多任务性能的关系。

**💡 创新点**

创新点在于：① 构造大规模实验平台，全面调节批量、权重衰减、学习率及其衰减等超参，揭示有效秩更多反映训练设置而非模型性能；② 证明低有效秩并非小模型饱和的根本原因，破除先前的因果假设；③ 对比有效秩与余度同质性、角度可变性等其它几何指标，指出它们对性能预测的有限性。

**🔧 技术方法**

采用矩阵奇异值分解计算有效秩、余度同质性和角度可变性等几何指标；在 Pile 数据上进行大规模预训练；使用 Paloma、Dolma‑100、StarCoder‑Python 等基准进行 ID/OOD、量化和灾难性遗忘评估；利用回归和散点图分析几何量与任务损失的相关性。

**📊 数据集**

主要使用 Pile 预训练语料；在评测时采用 Pile‑10k（ID）、Paloma、Dolma‑100（OOD）、StarCoder‑Python（微调）以及 GPTQ 量化测试。

**📈 对比分析**

将模型在不同批量、权重衰减、学习率以及 token 预算下的有效秩与各任务损失进行对应；结果显示：有效秩与 ID 损失呈相关但非单调；ID 损失更能预测 OOD 结果；低有效秩模型在量化时更易出现性能退化；在微调和灾难性遗忘实验中，几何指标对结果几乎没有解释力。

**⚠️ 局限性**

局限性包括：① 仅在 OLMo 架构下进行实验，缺乏对其他 Transformer 变体的验证；② 仅使用几何指标，未对其因果机制进行深入解析；③ 评测任务相对有限，未覆盖更广泛的下游任务或更大规模模型；④ 超参数范围虽广，但仍未涵盖所有可能的训练策略，导致结论在极端设置下的泛化能力未知。

---

## 249. SPRITETOMESH: Automatic Mesh Generation for 2D Skeletal Animation Using Learned Segmentation and Contour-Aware Vertex Placement

**arXiv ID:** 2602.21153 | [PDF](https://arxiv.org/pdf/2602.21153v1)

**作者:** Bastien Gimbert `[一作]` `[通讯]`, Bastien Gimbert

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一套自动化管线，能够将二维游戏精灵图像转换为适用于Spine2D等骨骼动画框架的三角网格。

**💡 创新点**

创新点在于结合深度学习分割与经典计算机视觉算法进行边界感知顶点放置，证明直接顶点回归不可行。

**🔧 技术方法**

使用EfficientNet-B0编码器+U-Net解码器进行前景分割，结合Douglas-Peucker、双边滤波、多通道Canny、Delaunay三角剖分等算法。

**📊 数据集**

训练数据为来自172款游戏的100,363张精灵图像，包括74,366张仅有掩码的region样本和25,997张带有顶点标注的mesh样本。

**📈 对比分析**

与手工制作、Alpha-仅外轮廓、均匀网格和Shi‑Tomasi角点等基线相比，本方法在边界遵循率最高（78.3%），顶点数适中，处理时间仅约1.8秒，显著加速工作流。

**⚠️ 局限性**

局限包括对参数的手动调节需求、缺乏语义理解、仅适用于RGBA游戏精灵、无法处理非游戏图像，以及未完成自动骨骼生成。

---

## 250. KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem

**arXiv ID:** 2602.20217 | [PDF](https://arxiv.org/pdf/2602.20217v1)

**作者:** Seongjin Cha `[一作]` (KAIST), Insu Han `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的自我推测解码框架KnapSpec，通过将层级选择建模为背包问题来自适应地挑选注意力层和MLP层，以最大化每秒生成的 token 数（TPT）并实现高速推理；

**💡 创新点**

核心创新在于：①把注意力层与MLP层拆开并以硬件相关、长度依赖的延迟作为权重；②用余弦相似度严格证明为接收率的有效代理；③通过两阶段动态规划（背包+TPT网格搜索）实现高效自适应层选；

**🔧 技术方法**

技术包括：自我推测解码、Transformer层级拆分、硬件延迟建模、余弦相似度代理、0/1背包动态规划、并行DP、动态草稿长度调整；

**📊 数据集**

在长上下文生成（AIME24/25、MMLU-Pro）与长上下文输入（GovReport、PG19、BookSum）上使用Qwen3系列和Llama3.1系列模型进行实验；

**📈 对比分析**

与SWIFT、DEL、CLaSp等最先进的训练无关SSD方法对比，KnapSpec在所有规模模型上实现1.47×的墙钟加速，TPT和实际吞吐量显著高于基线，且接受率与TPT相关性更高；

**⚠️ 局限性**

局限性在于：①仍需在每个推理阶段进行一次DP搜索，尽管已高效但仍有算力占用；②对极端长序列或极大模型时，权重离散化可能影响精度；③缺乏对多样化硬件平台的自适应性验证。

---

## 251. Naver Labs Europe @ WSDM CUP | Multilingual Retrieval

**arXiv ID:** 2602.20986 | [PDF](https://arxiv.org/pdf/2602.20986v1)

**作者:** Thibault Formal `[一作]` (NAVER LABS Europe), Stéphane Clinchant `[通讯]` (NAVER LABS Europe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过SPLARE模型在WSDM Cup 2026多语言文档检索任务中实现检索并提升效果。

**💡 创新点**

创新点在于使用稀疏自编码器构建语言无关的稀疏潜在特征空间，并结合RRF融合与LLM重排序。

**🔧 技术方法**

采用SPLARE（稀疏自编码器）、SPLADE、Qwen3-Embed、Qwen3-Reranker-4B、RRF融合、Seismic索引等技术。

**📊 数据集**

使用多语言语料库：MS MARCO、NQ、中文数据集、MIRACL、Mr.TyDi及WSDM Cup集合（约1000万文档）。

**📈 对比分析**

通过nDCG@20与SPLADE、Qwen3-Embed等基线对比，SPLARE-7B baseline提高约7.5分，融合+重排序后达到同类最优水平。

**⚠️ 局限性**

局限性包括未使用开发集标签微调、对效率关注不足、未利用英语翻译集合、以及仅在Dev集评估，缺乏更广泛的泛化验证。

---

## 252. Shape-informed cardiac mechanics surrogates in data-scarce regimes via geometric encoding and generative augmentation

**arXiv ID:** 2602.20306 | [PDF](https://arxiv.org/pdf/2602.20306v1)

**作者:** Davide Carrara `[一作]` (Politecnico di Milano), Francesco Regazzoni `[通讯]` (Politecnico di Milano)

**通讯引用:** 4618 | [OpenAlex ID](https://openalex.org/A5049456178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种两步解耦框架：先用PCA或DeepSDF学习左心室几何的低维潜在表示，再用基于USMNet的神经场模型在此潜在空间条件下预测心肌变形。

**💡 创新点**

创新点在于将几何建模与物理回归分离，利用潜在空间进行几何数据增强，显著提升数据稀缺场景下的泛化能力，并且提供两种几何编码方案可直接在点云数据上训练。

**🔧 技术方法**

技术包括：PCA形状模型、DeepSDF隐式神经表示、UVC（通用心室坐标）位置编码、USMNet神经场逼近器、Lipschitz正则化与数据增强。

**📊 数据集**

数据集包括：512个由长轴、直径、壁厚参数生成的理想化椭球模型；以及来自公开数据库的44个真实左心室模型（24心衰、20健康）。

**📈 对比分析**

与仅用空间坐标或真实几何参数的模型比较，经过几何编码与数据增强后，左心室变形预测误差从6.7%降至2.4%；在理想化数据上DeepSDF编码误差0.62%，PCA编码0.73%；在真实数据上PCA编码1.82%，DeepSDF编码2.39%，均优于无编码基线。

**⚠️ 局限性**

局限性包括：PCA编码需点对点对应且受UVC依赖；DeepSDF对噪声较敏感且训练成本较高；合成几何在极端解剖结构（如尖部或基部缺陷）时可能产生伪影；整体方法仍受高质量模拟数据稀缺的限制。

---

## 253. Quantifying the Expectation-Realisation Gap for Agentic AI Systems

**arXiv ID:** 2602.20292 | [PDF](https://arxiv.org/pdf/2602.20292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 254. DA-Cal: Towards Cross-Domain Calibration in Semantic Segmentation

**arXiv ID:** 2602.20860 | [PDF](https://arxiv.org/pdf/2602.20860v1)

**作者:** Wangkai Li `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18049 | [OpenAlex ID](https://openalex.org/A5100648981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出跨域校准框架 DA-Cal，利用软伪标签优化实现语义分割的无监督领域适应校准。

**💡 创新点**

创新点在于把校准问题转化为软伪标签的双层优化，设计像素级元温度网络 MTN，并采用互补域混合策略防止过拟合。

**🔧 技术方法**

使用技术包括元学习双层优化、像素级温度标定、软硬伪标签联合训练、ClassMix/CutMix 数据混合、EMA 更新和温度映射。

**📊 数据集**

实验数据集包括自动驾驶域（GTAv→Cityscapes、SYNTHIA→Cityscapes、Cityscapes→ACDC）以及生物医学域（VNC III→Lucchi、MitoEM‑R→MitoEM‑H）。

**📈 对比分析**

与 DACS、DAFormer、MIC 等自训练 UDA 基线以及 Ensemble、TempScal‑src、PseudoCal 等校准基线比较，ECE、NLL、BS 大幅下降，mIoU 提升 1–3% 甚至更高。

**⚠️ 局限性**

局限性包括额外的训练时间和显存消耗，对 MTN 结构和学习率较敏感，尚未在更多网络架构或多源域场景下验证其泛化能力。

---

## 255. From Logs to Language: Learning Optimal Verbalization for LLM-Based Recommendation in Production

**arXiv ID:** 2602.20558 | [PDF](https://arxiv.org/pdf/2602.20558v1)

**作者:** Yucheng Shi `[一作]` (University of Georgia), Linas Baltrunas `[通讯]` (Netflix)

**通讯引用:** 4648 | [OpenAlex ID](https://openalex.org/A5065744347)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种数据驱动的两阶段框架，将用户交互日志的自然语言化（verbalization）视为可学习的组件，并通过强化学习将其与LLM推理器（Reasoner）分离优化。

**💡 创新点**

创新点在于：①把传统固定模板化的verbalization转变为可学习的rewrite-based模型；②通过Group Relative Policy Optimization (GRPO)将推荐准确率直接作为verbalizer的奖励；③构建两阶段训练流程，使verbalizer和reasoner分别针对自己的目标进行专门化学习。

**🔧 技术方法**

主要技术包括：大型语言模型（如Qwen-3 8B/32B）作为Verbalizer和Reasoner；GRPO强化学习框架；长度奖励机制；对交互日志的语义摘要与噪声过滤策略。

**📊 数据集**

使用的是一大规模工业级流媒体平台数据集，包含数十万用户的观看历史（时间戳、内容ID、标题、参与度、观看时长），主要任务是给定用户最近最多100条交互和10条候选项预测下一条交互。

**📈 对比分析**

与基线（模板化verbalization、零样本、动作式verbalizer）相比，Rewrite+训练后的Reasoner在Recall@1（新内容召回）上实现了92.9%的相对提升；单独训练Reasoner时提升仅42.8%，表明verbalizer的学习贡献显著。

**⚠️ 局限性**

局限性包括：①训练成本高，尤其是强化学习阶段；②对LLM的依赖，模型规模增大后部署与推理成本显著；③框架目前仅在流媒体推荐上验证，跨域适用性需进一步验证；④verbalizer可能会在抽象过程中误删重要信息，需额外的可解释性与安全保障。

---

## 256. VINA: Variational Invertible Neural Architectures

**arXiv ID:** 2602.20480 | [PDF](https://arxiv.org/pdf/2602.20480v1)

**作者:** Shubhanshu Shekhar `[一作]` (University of Michigan), Kamal Youcef-Toumi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 8172 | [OpenAlex ID](https://openalex.org/A5086271665)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出基于变分无监督损失的统一框架，重新定义 INN 与 NF 的训练方式，提供了理论近似保证；

**💡 创新点**

创新点在于将变分推断与逆向可逆网络结合，得到同时适用于后验推理与生成建模的理论性能上界；

**🔧 技术方法**

使用变分推断、可逆神经网络（normalizing flow）、以及对比损失（如 Precision‑Recall）等技术；

**📊 数据集**

以海洋声学反演为真实案例，采用合成海洋声学数据集进行实验；

**📈 对比分析**

与传统 INN、NF 以及 GAN 等基线方法对比，显示在后验精度与生成分布一致性上均优于现有方法；

**⚠️ 局限性**

局限性包括对模型可逆性和 Jacobian 计算的高要求，理论假设仍需光滑性假设，且在大规模高维数据上的计算成本较高。

---

## 257. What Drives Students' Use of AI Chatbots? Technology Acceptance in Conversational AI

**arXiv ID:** 2602.20547 | [PDF](https://arxiv.org/pdf/2602.20547v1)

**作者:** Griffin Pitts `[一作]` (North Carolina State University), Sanaz Motamedi `[通讯]` (Pennsylvania State University)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5104541923)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对229名美国大学本科生的问卷调查，利用PLS-SEM检验并扩展TAM模型，探讨信任、主观规范和愉悦等因素对对话式AI聊天机器人的使用意图的影响。

**💡 创新点**

将TAM模型扩展到生成式对话AI系统，揭示信任、愉悦与社交规范在学生使用意图中的间接调节作用，弥补了传统TAM在AI领域的局限。

**🔧 技术方法**

采用部分最小二乘结构方程模型（PLS-SEM）进行路径分析和解释力评估。

**📊 数据集**

使用来自一所美国公立研究型大学的229名本科生完成的问卷数据。

**📈 对比分析**

通过路径系数和R²值评估模型，结果显示使用意图的R²为0.702，说明模型对行为意图的解释力较强；并通过对各条路径的显著性进行检验。

**⚠️ 局限性**

研究仅基于横断面自我报告的意图数据，未涉及实际使用行为；样本来自单一机构，限制了结果的普遍性。

---

## 258. Optimizing Occupancy Sensor Placement in Smart Environments

**arXiv ID:** 2602.21098 | [PDF](https://arxiv.org/pdf/2602.21098v1)

**作者:** Hao Lu `[一作]`, Richard J. Radke `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套基于室内轨迹模拟和整数线性规划的自动化占用传感器布置方法，用于提高商业办公空间的能耗控制精度。

**💡 创新点**

结合随机化轨迹生成、区域边界膨胀滤波以及对轨迹段的覆盖约束，将传感器布局优化转化为整数线性规划，实现对通道边界附近占用检测的高效覆盖。

**🔧 技术方法**

轨迹生成采用改进的A*算法并加入随机障碍、门口和墙壁惩罚；布置优化使用整数线性规划并通过分支定界求解；仿真评价利用Unity 3D数字孪生和时间飞行传感器模型。

**📊 数据集**

使用六个不同规模和布局的办公室平面图（手绘/电子图），并在Unity中生成的人工轨迹和真实人体动作仿真作为测试数据。

**📈 对比分析**

通过窗口化分类率（CCR）与ILP目标函数覆盖率对比，实验表明两者高度一致，随着传感器数量增加性能稳步提升，达到约90%以上的准确率。

**⚠️ 局限性**

依赖轨迹生成模型的准确性；在传感器数量不足时误差放大；未考虑传感器失效、通信丢包等鲁棒性问题。

---

## 259. CleanStyle: Plug-and-Play Style Conditioning Purification for Text-to-Image Stylization

**arXiv ID:** 2602.20721 | [PDF](https://arxiv.org/pdf/2602.20721v1)

**作者:** Xiaoman Feng `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 26233 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 CleanStyle，一种可插拔、无需训练的框架，能在扩散模型中去除风格图像导致的内容泄漏；

**💡 创新点**

创新点在于通过奇异值分解（SVD）分离风格嵌入的主成分与尾部噪声，利用时间感知指数衰减（CS‑SVD）抑制尾部成分，并用尾部信息构造风格专属负向条件（SS‑CFG）提升提示一致性；

**🔧 技术方法**

主要技术包括 SVD 分解、时间自适应抑制调度、风格特定的 Classifier‑Free Guidance 以及无训练的插拔式模块设计；

**📊 数据集**

实验使用 StyleBench、StyleAdapter 等风格图像与文本提示数据集进行评估；

**📈 对比分析**

与 InstantStyle、StyleShot、DEADiff、CSGO、IP‑Adapter 等现有编码器风格迁移方法对比，CleanStyle 在消除内容泄漏、提升提示对齐度和保持视觉质量方面均获得显著提升，且计算开销仅略增；

**⚠️ 局限性**

局限性包括：在保持风格细节与抑制内容之间存在一定权衡，可能导致部分风格相似度略降；需对超参数（如 k、α、γ）进行调优，且目前仅针对基于编码器的扩散模型，未针对全微调或其他架构验证。

---

## 260. On the Height Profile of Analog Error-Correcting Codes

**arXiv ID:** 2602.20366 | [PDF](https://arxiv.org/pdf/2602.20366v1)

**作者:** Ron M. Roth `[一作]` (Technion), Anxiao Jiang `[通讯]` (Texas A&M University)

**通讯引用:** 2110 | [OpenAlex ID](https://openalex.org/A5103150810)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了线性码的高度配置（height profile），并提供了通过线性规划和组合学方法计算该配置的多种公式，应用于模拟向量-矩阵乘法（VMM）的错误容忍与检测。

**💡 创新点**

创新点在于提出了高度配置的线性规划表达式、其对偶形式以及组合上对极值码字的精确刻画；同时给出了球面奇偶校验（SPC）与正交球面奇偶校验（OSPC）码的几何直观及其高度与球面码Packing/Covering 性能的关联。

**🔧 技术方法**

使用了线性规划、对偶性理论、LAD（最小绝对偏差）回归、单位范数紧框架（tight frame）以及几何覆盖/Packing 理论；并通过求解有限个线性程序或枚举有限子集实现高度配置的计算。

**📊 数据集**

主要使用了结构化的线性码实例（如MDS、循环码、SPC/OSPC 码以及由多面体顶点构成的码），并未使用外部真实数据集；所有实验均基于这些理论构造的代码。

**📈 对比分析**

与传统仅考虑最小距离的纠错能力相比，本文通过高度配置实现了更细粒度的误差比例 Δ/δ 与可容忍错误数的关系；在示例代码（如 (n) 代码、icosahedral 和 dodecahedral 码）中，计算得到的高度值证明了其优越或可比性能，且与先前的表格（如 <cit.>）保持一致或有所改进。

**⚠️ 局限性**

主要限制包括：1）高度配置的计算复杂度随代码长度 n、维度 k 以及 m 值增长而指数级提升；2）对一般非结构化代码缺乏下界或闭式表达；3）对偶性和组合性质的利用依赖于码的特殊结构，无法直接推广到所有线性码。

---

## 261. PromptCD: Test-Time Behavior Enhancement via Polarity-Prompt Contrastive Decoding

**arXiv ID:** 2602.20696 | [PDF](https://arxiv.org/pdf/2602.20696v1)

**作者:** Baolong Bi `[一作]` (Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 20709 | [OpenAlex ID](https://openalex.org/A5029998682)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种在推理阶段通过对立极性提示进行对比解码（PromptCD）的框架，用以在不更新模型参数的前提下提升大语言模型和多模态模型的目标行为。

**💡 创新点**

创新点在于构造正负极性提示对，并在每个解码步骤对这两种提示下的概率分布进行对比，形成一个统一且可调节的行为增强机制，可适用于多种对齐目标（如有用性、诚实性、无害性以及视觉定位）。

**🔧 技术方法**

技术上结合了对比解码、可适配的极性提示、适应性可行性约束（APC）以及对视觉注意力的对比重构，并通过同步双轨生成实现对目标行为的即时控制。

**📊 数据集**

实验使用了多种文本与视觉数据集，包括 NQ、ConFiQA、CoConflictQA、TruthfulQA、FactScore、SafeEdit、A-OKVQA、POPE、V*、TextVQA 等，覆盖帮助性、诚实性、无害性和视觉问答等维度。

**📈 对比分析**

与原始解码、DoLa、SLED、RECITE、DINM、SAM/YOLO/CLIP 等基线对比，PromptCD 在 3H 维度平均提升 10%–30% 以上，且在 VQA 任务中实现 20%–70% 的相对提升，同时保持较高的流畅度与安全性。

**⚠️ 局限性**

主要限制在于需要在每一步进行双向前向推理，导致 1.6–1.8 倍的延迟，并且对极性提示的设计仍需人工经验，未在极端对抗或高频变体场景下完全验证。

---

## 262. Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use

**arXiv ID:** 2602.20426 | [PDF](https://arxiv.org/pdf/2602.20426v1)

**作者:** Ruocheng Guo `[一作]` (Intuit AI Research), Kamalika Das `[通讯]` (Intuit AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于课程学习的框架，用于在LLM代理中改进工具接口，并通过构造大规模高质量工具描述数据集，训练LLM生成改进后的工具描述，从而提升工具选择与参数生成性能，兼顾无跟踪与有跟踪两种部署场景。

**💡 创新点**

创新点包括：①利用课程学习将训练阶段的跟踪监督逐步迁移到部署时的无跟踪场景；②构建基于真实API、修正参数模式与合成多步查询的端到端数据合成工作流；③在同一模型中学习可迁移的接口改进模式，解决传统方法每个工具独立优化且缺乏泛化的问题；④通过教师强制评估与多层次指标验证改进效果。

**🔧 技术方法**

技术手段包括：课程学习框架、开放权重LLM的监督微调、Smolagents与RIMRULE等工具协同的人工智能注释器、依赖感知的查询合成、教师强制评估、子任务/查询/工具层面多维度指标、以及对RESTful API的参数模式校验与修正。

**📊 数据集**

使用的数据集有：ToolBench（作为种子工具源）、StableToolBench（用于训练和评估）、RestBench（跨域评估），并通过上述工作流合成了大规模工具描述与执行轨迹的训练数据。

**📈 对比分析**

与基线EasyTool、DRAFT、Prompt2Play等方法对比，实验表明：在无跟踪设置下，课程学习模型在子任务、查询及工具层面均优于EasyTool和D1；在有跟踪设置下，模型仅需一次简单调用即可生成高质量描述，性能优于DRAFT、Prompt2Play；在工具数扩展至100+时，模型的查询成功率下降最小，表现出更强的鲁棒性。

**⚠️ 局限性**

局限性包括：①训练仍需依赖跟踪数据，无法完全避免对工具交互的依赖；②在无跟踪场景下无法修复参数模式，仅能改进描述；③对新颖工具类别的泛化能力尚未充分验证；④需要较大算力与高质量LLM支持；⑤生成的描述可能在极端复杂或不规则API上仍需人工校验。

---

## 263. MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision

**arXiv ID:** 2602.20689 | [PDF](https://arxiv.org/pdf/2602.20689v1)

**作者:** Bedrettin Cetinkaya `[一作]`, Emre Akbas `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于匹配的监督方法，使传统与学习型边缘检测模型产生的粗糙边缘被显著稀疏化为薄而精确的边缘。

**💡 创新点**

创新点在于通过融合注释失配、监督合并以及自然图像梯度三重原因，并用精确的线性求和分配（bipartite matching）来逐像素对齐预测与真值，显著提升边缘薄度与准确度。

**🔧 技术方法**

核心技术包括：(1) 计算全像素成本矩阵并求解三次复杂度的线性求和分配；(4) 轻量级 CNN（21k 参数）用于后处理；(5) 置信度阈值、距离阈值与置信度权重的自适应调优；以及 NMS 与匹配后重建步骤。

**📊 数据集**

在 NYUD‑v2、BSDS、Multi‑Cue 与 BIPED 四大公共数据集上进行实验，使用 RankED、PiDiNet、DiffusionEdge、SAUGE 等四种主流边缘检测器作为基线。

**📈 对比分析**

与多种 SOTA 方法（HED、RCF、BDCN、EDTER、UAED、MuGE、Diff.Edge 等）比较，在 Multi‑Cue 上 ODS/OIS/AP 依次提升至 0.965/0.967/0.995；在 BSDS 上同样取得最优或接近最优成绩；总体表现优于传统后处理（NMS/CRF）且保持低参数与轻量级特点。

**⚠️ 局限性**

主要限制包括：匹配求解的三次复杂度导致在初期噪声大、边缘粗糙时 GPU 计算与显存消耗较高；对置信度阈值与距离阈值的调参依赖性较大；在资源受限环境下难以在所有基线模型上完整验证。

---

## 264. Surrogate impact modelling for crop yield assessment

**arXiv ID:** 2602.20928 | [PDF](https://arxiv.org/pdf/2602.20928v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 265. Object-Scene-Camera Decomposition and Recomposition for Data-Efficient Monocular 3D Object Detection

**arXiv ID:** 2602.20627 | [PDF](https://arxiv.org/pdf/2602.20627v1)

**作者:** Zhaonian Kuang `[一作]` (Xi'an Jiaotong University), Gang Hua `[通讯]` (Amazon)

**通讯引用:** 20754 | [OpenAlex ID](https://openalex.org/A5081114810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种在线的对象‑场景‑相机分解与重组数据处理方案，用于提升单目 3D 目标检测的数据效率。

**💡 创新点**

通过把对象、场景和相机姿态三者解耦并在每个训练周期在线重组，显著缓解了三者紧耦合导致的过拟合、利用不足和姿态变异有限等问题。

**🔧 技术方法**

采用纹理点云表示的对象重建、场景空洞填充、基于稀疏点云的自由空间生成、随机插入和相机姿态扰动的渲染，并与现有单目 3D 检测模型（MonoDLE、GUPNet、DID‑M3D、MonoDETR、PETR 等）结合。

**📊 数据集**

在 KITTI 和 Waymo（Mono 与 Ring）两个大规模交通数据集上进行评估。

**📈 对比分析**

在完全监督和稀疏监督两种设置下，将方案嵌入原始模型后，AP_3D/BEV 平均精度在完全监督时提升约 26%‑48%，在仅 10% 注释量下实现与全监督相当；在 KITTI 上刷新 SOTA，在 Waymo 上亦显著提升。

**⚠️ 局限性**

仍存在离线构建数据库的计算/存储开销、在多摄像头场景中的域差异、对远距离物体提升有限，以及未探索无监督或更高效渲染方法等局限。

---

## 266. NGL-Prompter: Training-Free Sewing Pattern Estimation from a Single Image

**arXiv ID:** 2602.20700 | [PDF](https://arxiv.org/pdf/2602.20700v1)

**作者:** Anna Badalyan `[一作]` (Max Planck Institute for Intelligent Systems), Michael Black `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 52730 | [OpenAlex ID](https://openalex.org/A5065396778)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的三维服装结构估计方法 NGL-Prompter，能够仅凭单张图片（或文本）推断可直接生成 2D 缝合图案的 GarmentCode 参数，并支持多层服装。

**💡 创新点**

创新点在于：① 设计了 Natural Garment Language（NGL）这一中间 DSL，将 GarmentCode 的复杂参数映射为 VLM 易于推断的离散自然语言描述；② 利用冻结的 Vision‑Language 模型和约束式提示/logits 处理，完全不需要任何任务特定的训练；③ 通过确定性解析器将 NGL 转换为合法 GarmentCode，实现高质量可模拟的服装图案。

**🔧 技术方法**

使用的大型 VLM 包括 GPT‑5.0、Qwen‑2.5‑72B‑VL‑Instruct 等；通过精心设计的多轮问答和 logits 过滤生成 NGL；随后用确定性规则将 NGL 解析为 GarmentCode；最后利用 GarmentCode 编译成 2D 缝合图案并模拟生成 3D 服装。

**📊 数据集**

数据集：Dress4D、CloSe 两大基准数据集；自建 164 张标注图像（ASOS）用于 NGL 评估；224 张未标注图像用于提示工程；约 5,000 张野外时尚图像用于定性和感知评估。

**📈 对比分析**

与 ChatGarment（默认版）及 ChatGarment‑GPT 进行比较。测量 Chamfer Distance、F‑Score 等几何指标，NGL‑Prompter 在 Dress4D 上平均降低约 2 点 CD，CloSe 上提升约 1 点；在感知实验中人类与 GPT‑5.0 评估均显示 NGL‑Prompter 获得更高分数（单层 0.8、双层 1.0，均优于基准）。

**⚠️ 局限性**

局限性：① 受 GarmentCode 表达范围限制，无法描述某些特殊结构（如经典领型、系带式设计、非对称裙等）；② NGL 采用离散化、预定义选项，可能忽略细节；③ 目前完全依赖 VLM 的语言推理，若 VLM 认识不足会影响结果。

---

## 267. Energy-Based Injury Protection Database: Including Shearing Contact Thresholds for Hand and Finger Using Porcine Surrogates

**arXiv ID:** 2602.20362 | [PDF](https://arxiv.org/pdf/2602.20362v1)

**作者:** Robin Jeanne Kirschner `[一作]` (Technical University of Munich), Sami Haddadin `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 16723 | [OpenAlex ID](https://openalex.org/A5024171209)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对猪前爪进行无约束剪切碰撞实验，研究碰撞角度对伤害发生的影响，并构建基于能量阈值的伤害保护数据库。

**💡 创新点**

首次将无约束剪切碰撞情境加入能量阈值研究，发现剪切角度显著降低伤害风险，提供更宽容的能量上限。

**🔧 技术方法**

采用摆杆测试平台结合力/扭矩传感器、加速度计与自动化控制，实现不同角度、质量、速度的剪切碰撞实验。

**📊 数据集**

使用猪前爪作为人手与手指替代，收集1080次实验的能量、伤害类型数据，补充原先两项受限碰撞数据。

**📈 对比分析**

通过卡方检验比较不同碰撞角度的伤害概率，得出角度显著影响伤害发生；能量阈值分布显示剪切碰撞能量阈值比垂直碰撞高约2-3倍，表明控制器可容忍更大能量。

**⚠️ 局限性**

实验仅在猪模型上进行，未验证人类真实伤害阈值；仅考虑了三种撞击几何，且未对力数据进行深入统计，需进一步扩大角度范围和样本量。

---

## 268. How Foundational Skills Influence VLM-based Embodied Agents:A Native Perspective

**arXiv ID:** 2602.20687 | [PDF](https://arxiv.org/pdf/2602.20687v1)

**作者:** Bo Peng `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4187 | [OpenAlex ID](https://openalex.org/A5025292786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 NativeEmbodied 基准，用统一的本地低层动作空间对 VLM 驱动的具身智能进行评估。

**💡 创新点**

创新点在于：①首次引入“native”动作空间，让代理自由使用连续移动与旋转；②将高层任务拆解为低层基础技能任务，形成多粒度评估框架；③通过系统消融与思考模式实验揭示了 VLM 在细粒度空间交互上的瓶颈。

**🔧 技术方法**

使用 AI2THOR 仿真器、VLM（GPT‑4o、Claude‑4‑Opus、Gemini‑2.5‑Pro 等）、自定义动作集合、结构化感知模板及思考模式。

**📊 数据集**

数据集为 1,085 个样本，覆盖三类高层任务（探索、搜索、交互）与四类低层技能（感知、空间对齐、导航、规划），采样自 AI2THOR 场景。

**📈 对比分析**

与现有基准对比，最高 VLM 在搜索任务的成功率仅为 34.9%，在交互与探索任务分别为 52.4% 与 38.3%；低层任务中，最常见的瓶颈为空间对齐（仅 GPT‑4o 超过 50% 成功率），导航与规划表现中等，但感知表现良好。

**⚠️ 局限性**

限制在于：VLM 对细粒度空间动作（对齐、导航）缺乏鲁棒性；思考模式在提升规划时会抑制低层动作的准确性；整体而言，在本地环境中，现有 VLM 仍难以完成复杂具身任务。

---

## 269. Protein Language Models Diverge from Natural Language: Comparative Analysis and Improved Inference

**arXiv ID:** 2602.20449 | [PDF](https://arxiv.org/pdf/2602.20449v1)

**作者:** Anna Hart `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8456 | [OpenAlex ID](https://openalex.org/A5103178893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对蛋白质语言模型(PLM)与自然语言模型(NLM)的内部注意力机制进行直接比较，发现PLM在不同层、不同输入上对位置信息与语义信息的关注差异更大，并提出一种基于早退出(early‑exit)的推理策略，在非结构化蛋白功能预测任务上显著提升了准确率与计算效率。

**💡 创新点**

创新点在于：①首次直接对比PLM与NLM的注意力分布，揭示蛋白质语料在注意力层的高变异性；②将自然语言中的早退出技术迁移并改进为蛋白质专属的多层输出与“最可信层”回退策略，既提高性能又降低计算成本。

**🔧 技术方法**

使用的技术包括Transformer编码器、注意力分解(将注意力对数拆解为位置信息、语义信息与残差)、线性回归估计分量方差；以及在每层加装MLP并基于最大预测概率做置信度阈值判断的早退出框架。

**📊 数据集**

使用的数据集包括：1) 1,000条UniProtKB/SwissProt蛋白序列；2) 1,000条SlimPajama自然语言文本；3) GO‑Biological Process、EC、CL（PEER基准）和SSP（secondary structure prediction）等蛋白功能与结构任务的数据集。

**📈 对比分析**

通过统计每层注意力比值的方差、绘制热图与表格来比较PLM与NLM；在早退出实验中比较不同阈值、回退策略和单层/最后层基线，结果显示在GO、EC、CL任务中，最可信层回退可实现约10–50%的效率提升，同时提升0.4–7.01个百分点的F1/准确率。

**⚠️ 局限性**

局限性包括：仅针对encoder‑only模型；早退出方法依赖简单的置信度阈值，未探索更细粒度的自适应策略；对结构化任务（SSP）效果不佳；未对位置信息与语义信息与生物结构的关系做进一步解释。

---

## 270. Voices of the Mountains: Deep Learning-Based Vocal Error Detection System for Kurdish Maqams

**arXiv ID:** 2602.20744 | [PDF](https://arxiv.org/pdf/2602.20744v1)

**作者:** Darvan Shvan Khairaldeen `[一作]` (University of Kurdistan Hewler), Hossein Hassani `[通讯]` (University of Kurdistan Hewler)

**通讯引用:** 8113 | [OpenAlex ID](https://openalex.org/A5037960046)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对库尔德马卡姆（Bayati‑Kurd）唱法，本文构建了基于深度学习的声学错误检测与分类系统，能够识别音准细微偏差、节奏失误和调式漂移；

**💡 创新点**

创新点在于首次针对库尔德微音阶音乐制定专门的错误检测框架，采用两头CNN‑BiLSTM+注意力模型并结合类不平衡权重与焦点损失，以实现微调音准与节奏特征的联合学习；

**🔧 技术方法**

技术方法包括log‑mel谱图预处理、滑窗分割、CNN特征提取、BiLSTM时序建模、注意力聚合，并使用Sigmoid检测头和Softmax分类头，训练时采用AdamW、ReduceLROnPlateau、焦点损失及数据增强；

**📊 数据集**

使用了由13名歌手录制的50首Bayati‑Kurd歌曲（总时长约2.5小时，22.05kHz单声道），人工标注221个错误跨度（150细音准、46节奏、25调式漂移），构成15,199个滑动窗口的训练集；

**📈 对比分析**

在验证集上宏F1达到0.468，检测F1为0.216；在完整测试集上检测召回率39.4%、精确率25.8%（F1 0.311），错误类型宏F1 0.387，细音准与节奏分别达0.492与0.536，调式漂移仅0.133；与传统西方基准ASA工具相比，显著提升了对微音阶音准与节奏错误的识别，但整体召回仍偏低；

**⚠️ 局限性**

主要局限在于调式漂移样本稀缺导致召回极低、检测召回率不高、模型对表达性转音与噪声易产生误报、缺乏实时或轻量化实现以及缺乏多马卡姆泛化能力。

---

## 271. A Case Study on Runtime Verification of a Continuous Deployment Process

**arXiv ID:** 2602.20598 | [PDF](https://arxiv.org/pdf/2602.20598v1)

**作者:** Shoma Ansai `[一作]` (Kyoto University), Masaki Waga `[通讯]` (National Institute of Informatics)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5039539654)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用SyMon对FluxCD的持续交付过程进行运行时验证，监控日志以检查新镜像推送后FluxCD检测延迟。

**💡 创新点**

首次将符号监控技术应用于CD系统本身，针对时间约束进行动态检测，揭示了FluxCD在5分钟内检测不到镜像的隐患。

**🔧 技术方法**

采用SyMon的符号监控与时序数据单词、参数化时序数据自动机（PTDA）和时序正则表达式来建模和检测。

**📊 数据集**

使用从GitHub Actions、GHCR和FluxCD收集的日志，共计5天（12,758条）以及10天、15天的日志数据。

**📈 对比分析**

通过在不同天数日志上运行SyMon，执行时间始终低于0.4秒，说明监控开销极小，满足近实时监控需求。

**⚠️ 局限性**

SyMon目前无法直接处理JSON格式日志，需额外预处理，导致实现负担较大。

---

## 272. Exponential Lower Bounds for 2-query Relaxed Locally Decodable Codes

**arXiv ID:** 2602.20278 | [PDF](https://arxiv.org/pdf/2602.20278v1)

**作者:** Alexander R. Block `[一作]` (University of Illinois at Chicago), Minshen Zhu `[通讯]` (Purdue University)

**通讯引用:** 54 | [OpenAlex ID](https://openalex.org/A5085562284)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文证明了二查询的Hamming错误下的弱RLDC的码字长度必须指数级

**💡 创新点**

首次给出RLDC的指数下界，揭示了从指数到线性/近线性长度的相位转变

**🔧 技术方法**

采用了将RLDC转为标准LDC的构造、fixable码字位分析、随机限制与双重计数论证

**📊 数据集**

无实验数据集，纯理论分析

**📈 对比分析**

通过与已知的LDC下界对比，证明了二查询RLDC与2查询LDC等价下的指数下界

**⚠️ 局限性**

仅适用于完美一致性、二进制字母表，且对更高查询数或非完美一致性的情况尚未给出下界

---

## 273. SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards

**arXiv ID:** 2602.21158 | [PDF](https://arxiv.org/pdf/2602.21158v1)

**作者:** Dengjia Zhang `[一作]` (Johns Hopkins University), Hua Wei `[通讯]` (Arizona State University)

**通讯引用:** 7541 | [OpenAlex ID](https://openalex.org/A5100777770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于不确定性奖励的自演化LLM代理框架SELAUR，利用模型的不确定性信号来改进强化学习中的奖励设计，从而提升多步骤决策任务的成功率。

**💡 创新点**

创新点在于：①将熵、最小置信度、边际差异三种 token 级不确定性指标融合为统一的多视角不确定性估计；②通过失败感知的奖励重塑，将不确定性信息注入步骤级和轨迹级奖励，实现失败轨迹也能提供富含信息的学习信号；③在奖励设计中强调后期步骤的权重，从而更好地引导最终成功。

**🔧 技术方法**

技术包括：token 级不确定性度量（熵、least confidence、margin），多层级不确定性聚合（步骤级、轨迹级），失败感知奖励重塑机制，结合 PPO 等强化学习算法进行训练；并在预训练 LLM（如 Qwen2.5）上微调。

**📊 数据集**

使用了两大交互式基准：ALFWorld（家庭任务推理）和 WebShop（电子商务决策）。

**📈 对比分析**

与 PPO、RLOO、GRPO、GiGPO 等强化学习基线比较，SELAUR 在两大基准上均取得最高或接近最高的任务成功率（例如在 WebShop 上成功率提升至 0.7656，超过 GiGPO 的 0.6757），并在多种奖励方式 Ablation 上表现最优。

**⚠️ 局限性**

局限性包括：仅在文本交互任务上验证，尚未扩展到视觉或多模态环境；对不同规模 LLM 的通用性需进一步评估；不确定性度量的权重需要手动调参，可能对不同任务表现不一致。

---

## 274. Quantifying Dimensional Independence in Speech: An Information-Theoretic Framework for Disentangled Representation Learning

**arXiv ID:** 2602.20592 | [PDF](https://arxiv.org/pdf/2602.20592v1)

**作者:** Bipasha Kashyap `[一作]` (Deakin University), Pubudu N. Pathirana `[通讯]` (Deakin University)

**通讯引用:** 10024 | [OpenAlex ID](https://openalex.org/A5037113249)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在六大语料库上，利用信息论框架对情感、语言和病理维度的互信息进行量化，并通过源-滤波分解进行归因分析。

**💡 创新点**

创新性地将 MINE、CLUB 两个神经互信息估计器与 KSG 非参数估计器结合，形成带界估计并加入 EMA 稳定化、方差夹持和自适应加权，以提升高维连续分布互信息的估计可靠性。

**🔧 技术方法**

使用神经互信息估计（MINE、CLUB）、KSG 近邻互信息、EMA 稳定化、方差夹持、KSG 加权融合以及源-滤波归因。

**📊 数据集**

情感语料：RAVDESS、IEMOCAP；语言语料：L2-ARCTIC、GMU Speech Accent Archive；病理语料：UA-Speech、MDVR-KCL。

**📈 对比分析**

通过对比 MINE 下界、CLUB 上界和 KSG 基准，最终得到跨维度互信息均低于 0.15 nat，表明在手工特征空间内维度高度独立；源-滤波互信息显著更高（≈0.47 nat）。

**⚠️ 局限性**

局限性包括：仅评估手工提取特征，未检验自监督学习得到的表示；忽略时间动态特征；特征分组存在重叠，可能影响互信息结果；语料覆盖有限，未包含更多病理或情感场景。

---

## 275. Right to History: A Sovereignty Kernel for Verifiable AI Agent Execution

**arXiv ID:** 2602.20214 | [PDF](https://arxiv.org/pdf/2602.20214v1)

**作者:** Jing Zhang `[一作]` `[通讯]` (Independent Researcher), Jing Zhang (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了一个名为 PunkGo Kernel 的 Rust 语言写成的“Right to History” AI 代理行为审计系统，该系统在个人硬件上为 AI 代理提供可验证、不可篡改的行动日志。

**💡 创新点**

创新点包括：① 将 Certificate Transparency 的 RFC 6962 Merkle 树透明日志与能力隔离、能量预算治理相结合，形成全新的审计框架；② 引入人类审批机制与 20% 经济惩罚相结合的 hold/approval 流程；③ 通过五个系统不变量（Append-Only、Completeness、Integrity、Boundary Enforcement、Energy Conservation）构建完整的安全链；④ 通过形式化证明结构化地论证每个不变量的安全性。

**🔧 技术方法**

核心技术包括：Rust 语言实现、RFC 6962 Merkle 树（Google tlog 算法）、基于能力的访问控制、能量预算与持有机制、基于 JSON 的动作提交与验证、SQLite WAL 后端、以及人类审批交互层。

**📊 数据集**

未使用公开数据集，评测基于合成的 AI 代理动作（observe、create、mutate、execute 等），在本地硬件上模拟真实使用场景进行基准测试。

**📈 对比分析**

通过与 AIOS（主流 AI 代理 OS）的功能对比表、单线程提交性能基准（0.7–1.3 ms/动作，≈400 动作/秒）以及 Merkle 包含证明规模（10 000 条记录 448 B）来展示系统的可行性与优势；同时说明了在安全链中的各不变量在对抗特定攻击场景下的有效性。

**⚠️ 局限性**

主要局限包括：① 单节点单写入架构，缺乏多写/多主机共识；② 形式化证明仅为结构化证明，尚未完成机器化验证；③ Merkle 证明生成时间为 O(n)（需进一步优化为 O(log n)）；④ 只防御 Level 1 代理攻击，未对 Level 2（根级攻击）做防护；⑤ 未实现执行环境隔离与外部工具的真实状态验证；⑥ 依赖于用户手动设置硬件能量参数；⑦ 目前未集成 TEE、外部日志锚定或分布式验证服务。

---

## 276. Cycle-Consistent Tuning for Layered Image Decomposition

**arXiv ID:** 2602.20989 | [PDF](https://arxiv.org/pdf/2602.20989v1)

**作者:** Zheng Gu `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 21355 | [OpenAlex ID](https://openalex.org/A5100684575)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的视觉在场学习框架，用于在真实图像中将覆盖的logo与底层物体进行分离并再合成，支持复杂几何、光照和视角变化。

**💡 创新点**

创新点包括：①将单输入多输出的视觉在场学习与循环一致性训练相结合，使分解与重组模型相互监督；②采用轻量化LoRA微调Flux-Fill的图像修复模型进行分解；③构建自我改进的迭代数据收集流程，利用模型生成伪标签逐步提升训练数据质量。

**🔧 技术方法**

技术手段：扩散Transformer（Flux-Fill），LoRA低秩适配，循环一致性损失，进阶数据生成与过滤，图像修复与组合的双向训练。

**📊 数据集**

使用的数据集包括：①手工标注的100对（I, A, B）作为种子；②利用GPT‑4o、Qwen‑VL自动生成与过滤的伪数据；③Synthetic Logo‑Object测试集（1.5K样本）；④Hypersim（用于 intrinsic decomposition 训练）和 MAW（用于评估）；⑤约5K人工合成的前景‑背景三元组。

**📈 对比分析**

与 AssetDropper、Flux‑Kontext、Gemini、IC‑Edit 等基线进行定量对比，使用 VQAScore 与多模型 VLMScore（Qwen、GPT‑4o、Gemini）评估 logo isolation、consistency、object isolation 与 consistency。实验显示在 logo‑object 分解任务中本方法在 VQAScore 与所有 VLMScore 维度均排名第一，且在用户研究中在一致性与自然度方面获得超过 50% 的 top‑1 评价。

**⚠️ 局限性**

局限性：①对占比极大、覆盖范围过宽的 logo（如广告牌、墙面大 logo）效果仍不佳；②目前模型仅支持两层分解，无法一次性处理多层叠加场景；③在极端光照或材质极端反射的极端案例中仍可能出现伪影。

---

## 277. Understanding Human-AI Collaboration in Cybersecurity Competitions

**arXiv ID:** 2602.20446 | [PDF](https://arxiv.org/pdf/2602.20446v1)

**作者:** Tingxuan Tang `[一作]` (William and Mary), Yue Xiao `[通讯]` (William and Mary)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在一次现场大学级CTF竞赛中，对人类参与者使用AI助手的感知、交互行为以及AI自主解题能力进行实证研究，首次量化人机协作的效果与瓶颈。

**💡 创新点**

创新点在于：①首次在真实比赛环境下收集并分析人机交互日志；②系统性对比人类团队与完全自主AI代理在相同挑战集上的表现；③揭示人类对AI期望与信任的变化、有效与无效协作策略，以及AI能否补偿缺失的CTF专业知识。

**🔧 技术方法**

使用的技术包括：多模态LLM（Claude‑Sonnet 4.5/Opus 4.1/Haiku 3.5）构建的AI助手与四个自主CTF代理（Claude Code、NYU CTF、Cybench、专有代理）；对话系统记录与MaxQDA文本编码；实验中还用到工具调用与可视化监控。

**📊 数据集**

数据集为本次竞赛新设计的17道CTF挑战（涵盖取证、密码学、逆向、Web等5类），共95名参与者中41名进入研究；收集2,299条聊天记录和所有挑战的JSON规范文件。

**📈 对比分析**

比较方法：对同一挑战集进行人工与四个代理的“三次尝试”自动化实验，记录成功率、得分、累计运行时间与API成本；结果显示最强代理（专有代理+Sonnet 4.5）获得与人类第2名相当的分数，仅使用约1/5的人类累计运行时间，成本约96美元；人类团队平均成功率与得分远低于最强代理，表明AI在当前能力下可部分替代人类。

**⚠️ 局限性**

局限性：①仅在一次特定的、时间受限的比赛场景下进行；②受限于参与者数量与分布，难以广泛推断；③数据编码由单一研究员完成，尽管做了复核但仍存在主观偏差；④LLM表现受模型版本、温度与工具集限制；⑤结果可能随技术更新与任务设计变化而不同。

---

## 278. The TCF doesn't really A(A)ID -- Automatic Privacy Analysis and Legal Compliance of TCF-based Android Applications

**arXiv ID:** 2602.20222 | [PDF](https://arxiv.org/pdf/2602.20222v1)

**作者:** Victor Morel `[一作]` (Chalmers University of Technology and University of Gothenburg), Romaric Duvignau `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了Google Play上流行Android应用中TCF框架的普及率、实现情况与合规性，系统分析了其对用户同意选择的尊重与个人数据（如AAID）传输行为。

**💡 创新点**

首次在移动端系统化评估TCF实施，发现应用在尊重同意、合法性基础和数据共享方面普遍存在违规，并揭示Google作为CMP主导导致的结构性合规风险。

**🔧 技术方法**

采用自动化爬取、APK下载、Emulator+Appium交互、MITM代理+Objection进行网络流量捕获、TC字符串解析与日志分析，结合机器学习匹配Banner按钮文本。

**📊 数据集**

从2025年1月29日至3月10日抓取并下载了5087个最热门App，最终分析了4482个可用APK，其中576个实现TCF，涵盖12个CMP和多国开发者。

**📈 对比分析**

对比了主动与被动两阶段网络流量，使用三种同意策略（全同意、仅合法利益、全拒绝）评估AAID泄露比例，结果显示约66%应用在被拒绝后仍传输AAID，说明TCF实现效果差。

**⚠️ 局限性**

样本局限于热门App、基于Android模拟器、仅关注部分CMP，未覆盖低下载量App、iOS及非标准Banner；未对加密或哈希数据做深度检测；可能的设备与实际用户行为差异影响结果。

---

## 279. Automata Learning with an Incomplete but Inductive Teacher

**arXiv ID:** 2602.21073 | [PDF](https://arxiv.org/pdf/2602.21073v1)

**作者:** Daniel Stan `[一作]` (EPITA Research Laboratory), Juliette Jacquot `[通讯]` (EPITA Research Laboratory)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 IdMAT 框架和基于 SAT 的主动学习算法，解决了在非唯一目标语言下的自动机学习与正则模型检查问题。

**💡 创新点**

创新点在于①设计了可返回不确定答案与归纳约束的 Inductive MAT；②使用单一 SAT 求解器与 UNSAT 核分析，而非多实例或队列；③将 Rivest‑Schapire 计数技术推广到归纳对与不确定答案。

**🔧 技术方法**

采用增量 SAT 求解器（CaDiCaL）、UNSAT 核提取、观察表（IOT）与基函数构造，基于 Angluin 的 L*。

**📊 数据集**

实验使用语言分离基准（Oliveira 等）以及 28 个参数化协议的正则模型检查实例。

**📈 对比分析**

与 Nerode 的 L*□ 对比，平均运行时间从 24 s 降至 6 s；在 RMC 任务中，IdMAT+RS 方案在大多数模型上实现更高成功率，且相对改进查询复杂度。

**⚠️ 局限性**

局限包括①目标类非唯一时可能产生多解，学习到的 DFA 可能不是最小；②SAT 核分析最坏情况指数级；③长词的 membership 查询开销大，受实现语言限制。

---

## 280. MAST: A Multi-fidelity Augmented Surrogate model via Spatial Trust-weighting

**arXiv ID:** 2602.20974 | [PDF](https://arxiv.org/pdf/2602.20974v1)

**作者:** Ahmed Mohamed Eisa Nasr `[一作]` (University of Southampton), Haris Moazam Sheikh `[通讯]` (University of Southampton)

**通讯引用:** 224 | [OpenAlex ID](https://openalex.org/A5034734698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多保真度增强代理模型MAST，利用几何距离权重融合低保真与高保真数据，提升在计算预算有限时的预测精度和不确定性校准。

**💡 创新点**

核心创新在于：① 通过局部距离驱动的信任区分度量，将纠正后的低保真观测与高保真预测按空间相邻度自适应加权；② 将评估成本融入权重指数，实现对保真度差距的成本感知；③ 采用闭式方差传播，构建单一异方差高斯过程，避免昂贵的超参数优化。

**🔧 技术方法**

技术包括独立高斯过程训练、误差高斯过程建模、基于欧氏距离的自适应加权、异方差高斯过程融合和闭式不确定性传播。

**📊 数据集**

使用标准的多保真度基准函数（Branin、Hartmann、Ackley、Rastrigin、Rosenbrock、Borehole 等）在不同维度下进行实验，数据采样采用拉丁超立方法。

**📈 对比分析**

与 AR1 共轭、递归共轭、模型重识别融合、BoTorch MF-GP、NARGP、MF-DGP 等方法对比，MAST 在 RMSE 与平均 PDF 两指标上在大多数基准中均显著优于或与现有方法持平，且对预算、保真度差异和高维度具有更强鲁棒性。

**⚠️ 局限性**

局限性：假设所有保真度共享相同输入维度，无法直接处理高保真含额外变量的情况；在保真度高度不匹配或低保真数据极少时仍可能受限。

---

## 281. Counterfactual Simulation Training for Chain-of-Thought Faithfulness

**arXiv ID:** 2602.20710 | [PDF](https://arxiv.org/pdf/2602.20710v1)

**作者:** Peter Hase `[一作]` (Stanford University), Christopher Potts `[通讯]` (Stanford University)

**通讯引用:** 26215 | [OpenAlex ID](https://openalex.org/A5042601761)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练模型生成更可信的链式推理（CoT），通过反事实模拟训练（CST）提升CoT的真实性与可监测性。

**💡 创新点**

创新点在于将CoT监测与反事实模拟统一到一个训练框架，利用反事实输入和模拟器奖励来改进CoT的可信度；并引入LLM重写不可信CoT、与RL结合的高效训练策略。

**🔧 技术方法**

使用对比学习与奖励加权交叉熵的训练目标，配合LLM模拟器评估反事实可模拟性；结合自监督重写和强化学习进行样本生成。

**📊 数据集**

实验使用 MMLU、SNLI、ETHICS、MMLU‑Pro 等多任务数据集，并通过 cue‑based 与 model‑based 方法生成反事实。

**📈 对比分析**

与提示基准、纯 RL 等对比，CST 在 cue‑based 反事实监测中 G‑mean 提升 25‑35 分，模型规模越大收益越显著；在 generic 反事实上提升约 30 分，整体效果显著。

**⚠️ 局限性**

局限：对否定/反对性提示的可模拟性提升有限；重写成功率受限，无法保证所有样本都能得到正样本；训练对模型准确率有轻微负面影响。

---

## 282. Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization

**arXiv ID:** 2602.20743 | [PDF](https://arxiv.org/pdf/2602.20743v1)

**作者:** Gabriel Loiseau `[一作]` (Hornetsecurity), Marc Tommasi `[通讯]` (University of Lille)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自适应文本匿名化框架，通过演化式提示优化（GEPA）自动生成符合隐私-实用性需求的语言模型提示，支持在本地开源模型上实现多任务、多域的匿名化。

**💡 创新点**

创新点包括：①将匿名化任务转化为字符串搜索问题，实现单个优化过程产生多个 Pareto 最优提示；②在两阶段（基本反馈+丰富反馈）中结合自动生成的评估器和自适应验证采样提升搜索效率；③提供跨五个领域、不同隐私-实用性目标的统一基准。

**🔧 技术方法**

使用技术：大语言模型（Mistral‑Small、Gemma‑3‑27B、Qwen3‑30B‑A3B）、演化式提示优化 GEPA、反射式突变、自动生成富反馈函数、采样验证策略，以及基于 DSPy 的统一实现。

**📊 数据集**

数据集：DB‑Bio、SynthPAI、TAB、PUPA、MedQA，共五个多领域文本数据集，分别对应不同隐私威胁模型和实用性评估。

**📈 对比分析**

与基线（OpenPII、AF、RUPTA、手工提示）和闭源 GPT‑5 方案对比，使用同一开源模型即可获得接近或优于闭源方法的隐私‑实用性平衡；在各任务上，优化提示往往提升隐私分数同时保持或略增实用性。

**⚠️ 局限性**

局限性：①需要少量标注训练/验证样本；②隐私和实用性以单一加权求和方式组合，未覆盖更细粒度决策规则；③评估仍依赖闭源 LLM，导致数据泄露风险；④未考虑推理型模型的成本和可扩展性；⑤生成过程的非确定性可能导致收敛不稳定。

---

## 283. Is a LOCAL algorithm computable?

**arXiv ID:** 2602.21022 | [PDF](https://arxiv.org/pdf/2602.21022v1)

**作者:** Antonio Cruciani `[一作]` (Aalto University), Jukka Suomela `[通讯]` (Aalto University)

**通讯引用:** 2462 | [OpenAlex ID](https://openalex.org/A5025555126)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在分布式计算模型中，构造了一个LCL问题，用以证明在可计算与不可计算两种模型之间存在显著的时间复杂度差异；当不具备顶点数上限信息时，该问题在可计算模型下需Ω(√n)轮，而在不可计算模型下仅需O(log n)轮；进一步证明若节点知晓图大小上限，计算与不可计算模型的能力相同；

**💡 创新点**

首次揭示可计算性假设与顶点数信息在LCL问题中的本质关联，证明可计算性假设并非可忽视的细节，并通过构造可区分两模型的LCL问题提供了理论上的分界；

**🔧 技术方法**

采用层次化的“认证LCL”技术，将问题分解为五层结构（树、行、网格、图灵机执行、最终输出），利用可计算/不可计算的图灵机oracle、对角线论证、最大安全邻域（max‑T‑safe neighborhood）与算法转化方法；

**📊 数据集**

无真实数据集，全部基于理论构造的图（如“成长网格”与附带树结构的图）；

**📈 对比分析**

通过证明上界（O(log n)）与下界（Ω(√n)）的匹配，展示在可计算与不可计算模型下复杂度的显著差异；并证明在已知上限N时，两模型可等价；

**⚠️ 局限性**

局限性在于仅针对理论上的可计算/不可计算模型，缺乏可实现性与实际网络环境的直接对应；此外结果仅适用于确定性算法，随机化或近似模型尚未涉及；

---

## 284. CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation

**arXiv ID:** 2602.20409 | [PDF](https://arxiv.org/pdf/2602.20409v1)

**作者:** Mainak Singha `[一作]` (University of Trento), Biplab Banerjee `[通讯]` (IIT Bombay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出CLIPoint3D框架，实现少量无监督3D点云域适应。

**💡 创新点**

将CLIP与LLM生成的语义提示、轻量3D编码器融合的知识驱动提示调优、熵引导视图筛选以及不确定性加权原型与OT对齐相结合，构建跨模态、跨域的自适应方法。

**🔧 技术方法**

采用知识驱动提示调优、LoRA参数高效微调、熵引导视图采样、基于不确定性的原型对齐和熵正则化的Optimal Transport损失，配合CLIP的视觉与文本编码器。

**📊 数据集**

在PointDA-10和GraspNetPC-10两大点云域适应基准上进行实验。

**📈 对比分析**

与传统3D编码器和CLIP扩展基线对比，平均提升3–16%分类准确率，尤其在GraspNetPC-10上实现最高16.4%的增益。

**⚠️ 局限性**

对真实稠密扫描的鲁棒性仍有限，且依赖多视图投影导致计算量和投影质量受限。

---

## 285. Exa-PSD: a new Persian sentiment analysis dataset on Twitter

**arXiv ID:** 2602.20892 | [PDF](https://arxiv.org/pdf/2602.20892v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 286. CodeHacker: Automated Test Case Generation for Detecting Vulnerabilities in Competitive Programming Solutions

**arXiv ID:** 2602.20213 | [PDF](https://arxiv.org/pdf/2602.20213v1)

**作者:** Jingwei Shi `[一作]` (Shanghai University of Finance and Economics), Shengyu Tao `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CodeHacker 框架，自动生成针对程序提交的对抗测试用例，提升 LLM 代码评估的可靠性，并构建 CodeHackerBench 基准；

**💡 创新点**

将对抗攻击视为程序级的测试生成，结合自我校准的 Validator/Checker、多策略对抗生成（Stress、LLM、Anti‑Hash），以及利用 LLM 推理生成精细边界条件，显著提高真负率与模型评估的真实性；

**🔧 技术方法**

使用 LLM 驱动的策略、强化学习（DAPO/GRPO）、链式思考、代码静态与行为分析、抗 Hash 碰撞方法、自适应验证器/检查器改进、对抗生成算法；

**📊 数据集**

基准数据集包括 2000 道 CodeContests、HardTests、TACO、CodeContest+ 等竞赛题库，LiveCodeBench 以及通过 CodeHacker 生成并校准的 CodeHackerBench；

**📈 对比分析**

与传统测试、Special Judge、CodeContest+ 等进行对比，TNR 提升至 96%+，HSR 最高 64.83%（DeepSeek V3.2），Pass@1 在 CodeHackerBench 上下降表明纠正指标膨胀，RL 训练使用对抗数据提升 LiveCodeBench 的 Pass@5 性能，消融实验显示每个组件均带来显著提升；

**⚠️ 局限性**

仍无法捕获所有复杂失败（性能瓶颈、复杂验证逻辑）且主要覆盖 C++，约 5% 需要人工干预；对抗生成方法对语言和题型的通用性有限，需进一步扩展。

---

## 287. Efficient Interview Scheduling for Stable Matching

**arXiv ID:** 2602.20358 | [PDF](https://arxiv.org/pdf/2602.20358v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Assaf Romm `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5073660616)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了两种自适应的面试排程算法（顺序和混合），在偏好不确定的两侧匹配市场中实现中期稳定匹配。

**💡 创新点**

在双向前置等价（ex‑ante equivalent）市场中证明：期望面试次数仅为 2n + O(log³n)（即每位申请者平均 2 次面试），混合算法将面试轮数压缩到 O(log³n)，并给出了 2c−2 次面试的下界，说明该常数是不可突破的。

**🔧 技术方法**

采用改进的 Gale‑Shapley（申请者提议）框架，结合“中期效用”更新和自适应选择面试对，利用概率论和事件分段分析（阶段划分、k‑好转等），并引入最大匹配子程序实现并行面试。

**📊 数据集**

论文主要为理论分析，不使用具体数据集；所有结果均来自概率模型和数学证明。

**📈 对比分析**

与此前研究相比，先前方法在每位代理人面试次数多达 O(log²n) 或 O(log³n)，但未考虑面试轮数；本工作同时优化两者，在满足前置等价假设下，面试次数降至常数级，面试轮数降至多项式对数级，显著提升效率。

**⚠️ 局限性**

局限性：结论依赖于双向前置等价假设，无法直接推广到更一般的偏好分布；对于不满足等价性的市场，仍无法保证常数或线性面试次数；此外，混合算法在极端不平衡市场中仍需进一步验证。

---

## 288. FAST-Prefill: FPGA Accelerated Sparse Attention for Long Context LLM Prefill

**arXiv ID:** 2602.20515 | [PDF](https://arxiv.org/pdf/2602.20515v1)

**作者:** Rakshith Jayanth `[一作]` (University of Southern California), Viktor Prasanna `[通讯]` (University of Southern California)

**通讯引用:** 17432 | [OpenAlex ID](https://openalex.org/A5033166029)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为长上下文大语言模型的预填阶段加速，提出了FAST-Prefill FPGA加速器，实现了动态稀疏注意力的完整流水线计算。

**💡 创新点**

创新点包括：①基于Flex-Prefill的流式稀疏索引生成单元，显著压缩中间张量；②自适应双层缓存，利用存活驱动的替换策略高效复用KV缓存；③混合矩阵乘法单元，结合DSP和LUT位平面算术提升算力。

**🔧 技术方法**

采用的技术有：FPGA数据流架构、存活驱动缓存、块级索引生成流水线、LUT/bit‑plane算术、混合DSP+LUT的32×32 systolic array、Vitis HLS + Vivado实现。

**📊 数据集**

使用的数据集为RULER benchmark，模型包括Llama3.2-1B/3B和Qwen2.5-1B，评估长上下文（4K~128K）下的预填推理。

**📈 对比分析**

与NVIDIA A5000 GPU上Flex‑Prefill（BF‑16/INT‑8）对比，FAST‑Prefill在TTFT上实现1.5×–2.5×加速，能耗提升约4.5×；在不同上下文长度上保持一致的速度优势。

**⚠️ 局限性**

局限性包括：对非常大KV缓存（3–4 GB）仍需依赖HBM，缓存容量受限；混合MPU仍占用大量LUT资源；缺乏对更深层模型或多FPGA扩展的评估。

---

## 289. Stability and Generalization of Push-Sum Based Decentralized Optimization over Directed Graphs

**arXiv ID:** 2602.20567 | [PDF](https://arxiv.org/pdf/2602.20567v1)

**作者:** Yifei Liang `[一作]` (Sun Yat-sen University), Li Shen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 15561 | [OpenAlex ID](https://openalex.org/A5100768717)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析了基于Push‑Sum的去中心化优化算法Stochastic Gradient Push（SGP）在有向通信网络中的稳定性与泛化性能，给出了有限迭代下的误差上界；

**💡 创新点**

首次将网络不平衡度δ与谱间隙(1‑λ)联合分解，提出了统一的“均匀稳定性”框架，并证明了Push‑Sum校正对非对称网络的重要性；

**🔧 技术方法**

使用均匀稳定性理论、Push‑Sum一致性界、谱间隙与不平衡参数分析以及Polyak‑Łojasiewicz条件的凸性分析；

**📊 数据集**

在实验中使用a9a数据集进行逻辑回归，以及CIFAR‑10数据集训练LeNet‑5模型；

**📈 对比分析**

与传统D‑SGD及其他去中心化方法对比，实验表明SGP在有向网络上既保持了统计效率（≈1/√{mn}），又能通过早停实现最佳泛化误差，网络不平衡度越大导致误差上界越高；

**⚠️ 局限性**

局限性在于仅针对光滑L‑Lipschitz、G‑Lipschitz且参数空间有界的设置，且仅覆盖PŁ条件下的非凸场景，缺乏对更一般非凸或异步通信、压缩等实际因素的理论分析。

---

## 290. IntRR: A Framework for Integrating SID Redistribution and Length Reduction

**arXiv ID:** 2602.20704 | [PDF](https://arxiv.org/pdf/2602.20704v1)

**作者:** Zesheng Wang `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**通讯引用:** 5439 | [OpenAlex ID](https://openalex.org/A5101512474)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种递归赋值网络（RAN），在生成式推荐阶段动态重分配语义 ID 并将 SID 层级内化，从而在保持高预测准确率的同时显著压缩序列长度并提升效率。

**💡 创新点**

①在生成学习阶段以 UID 作为协同引导，对 SID 进行动态重分配，突破静态索引瓶颈；②将 SID 递归解码内嵌于 RAN，实现一次性解码（每个物品仅 1 个 token），消除传统自回归的序列膨胀。

**🔧 技术方法**

采用层级码本 + 可学习的分配分布、可切换的 Teacher‑Forcing/soft‑weighted 模式、Transformer 或 HSTU 主干、联合对齐损失与推荐损失的双目标优化、Beam Search 推理加速等技术。

**📊 数据集**

在 Amazon Beauty、Amazon Sports、Amazon Toys 三大电商基准数据集上进行实验。

**📈 对比分析**

与传统 flattened SID 基线以及 SSL 设置下的基线在同一历史长度（30 项）下进行对比；在所有索引方式（RK‑Means、VQ‑VAE、RQ‑VAE）和两种主干（Transformer、HSTU）上均实现 Recall@10、NDCG@10 大幅提升，最高可达 60%+；训练吞吐量提升 70%+，显存下降 68%+，推理延迟降低约 1/3，速度提升可达 3 倍。

**⚠️ 局限性**

仍需依赖预先构建的层级 SID，超大层级或更深码本的适配性未知；重分配过程对超参数 λ 依赖显著；在极度稀疏或冷启动场景下协同信息不足可能导致分配不稳定；递归解码虽压缩序列，但对极长历史仍存在上限。

---

## 291. Coupled Cluster con MōLe: Molecular Orbital Learning for Neural Wavefunctions

**arXiv ID:** 2602.20232 | [PDF](https://arxiv.org/pdf/2602.20232v1)

**作者:** Luca Thiede `[一作]`, Alán Aspuru-Guzik `[通讯]` (NVIDIA)

**通讯引用:** 69132 | [OpenAlex ID](https://openalex.org/A5071495561)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过构建旋转等变性的神经网络架构，预测CCSD振幅，从而在保持化学精度的同时显著加速计算；

**💡 创新点**

首次提出基于分子轨道输入的等变性Transformer+MACE网络，兼顾尺寸可扩展性与数据效率；

**🔧 技术方法**

采用等变性Transformer、MACE和Odd‑MACE层，结合Delta‑MP2学习实现振幅差值预测；

**📊 数据集**

在QM7、氨基酸、PubChem等多样化数据集上训练与评估；

**📈 对比分析**

与Δ‑MP2 MLIP比较，MōLe在低数据量、规模外推和离平衡几何下能量MAE≈0.12 mHa，且可减少40–50 % CCSD迭代；

**⚠️ 局限性**

受限于训练集规模、尚未验证更高层CC（如CCSDT）以及仍有较高的计算复杂度。

---

## 292. Playsemble: Learning Low-Level Programming Through Interactive Games

**arXiv ID:** 2602.20167 | [PDF](https://arxiv.org/pdf/2602.20167v1)

**作者:** Elliott Wen `[一作]` (University of Auckland), Yu Yang `[通讯]` (Education University of Hong Kong)

**通讯引用:** 2462 | [OpenAlex ID](https://openalex.org/A5100687570)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 Playsemble，一个将汇编指令转换为 Pac‑Man 游戏任务的基于 Web 的交互式学习平台，支持在线编写、模拟、调试与即时反馈。

**💡 创新点**

创新点包括：将低级编程与游戏化任务相结合、使用大型语言模型（LLM）生成精准的学习反馈、以及基于 CPU 周期的排名系统激励学生优化代码。

**🔧 技术方法**

技术实现依赖 Monaco 编辑器、WebAssembly 编译的 GNU 汇编器与 Unicorn 虚拟 CPU、前端 WebGL 游戏引擎、OpenRouter API 调用 LLM（如 qwen3:14b）以及 CRISPE 提示框架。

**📊 数据集**

数据集为 107 名三年级本科生在架构课程中的学习记录、提交日志、调试使用情况和问卷反馈，涵盖 5 个递增难度的编程作业。

**📈 对比分析**

通过分析会话时长、执行次数、错误类型及排行榜表现来比较学习效果；实验显示学生在后期任务中平均会话时长、执行次数显著上升，错误率下降，且多达 64.7% 的学生继续优化至更低 CPU 周期，说明游戏化与即时反馈显著提升了参与度与代码效率。

**⚠️ 局限性**

局限性包括：LLM 可能产生幻觉或不符合特定汇编版本的指令、设计缺陷导致任务作弊、仅验证本课程内的汇编语言与平台、缺乏对其他编程语言或教学环境的外部可验证性。

---

## 293. Modality-Guided Mixture of Graph Experts with Entropy-Triggered Routing for Multimodal Recommendation

**arXiv ID:** 2602.20723 | [PDF](https://arxiv.org/pdf/2602.20723v1)

**作者:** Ji Dai `[一作]` (Beijing University of Posts and Telecommunications), Dengsheng Cai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的多模态推荐模型MAGNET，利用图结构增强、模态引导的专家池以及基于熵的分阶段路由，实现对多模态信息的自适应融合与解释；

**💡 创新点**

核心创新包括：①基于模态的专家结构（行为、视觉、语义三组，内含主导、平衡、互补三种模板）；②利用内容诱导的高阶图边提升稀疏长尾场景的表达能力；③熵触发的两阶段路由策略，先鼓励专家覆盖再逐步提升路由尖锐度，避免专家坍塌；

**🔧 技术方法**

技术手段包括：轻量级双视图LightGCN编码、三元组专家融合（共享模板）、稀疏Mixture‑of‑Experts、熵正则化的两阶段训练、对齐视图的InfoNCE对比学习；

**📊 数据集**

使用四个Amazon商品评论数据集（Baby、Sports、Clothing、Electronics），每个数据集均包含用户交互和视觉（ResNet50）/文本（BERT）特征；

**📈 对比分析**

在Recall@20和NDCG@20上，MAGNET（双视图）在所有数据集上均优于经典CF、双图学习和对比自监督方法，平均提升约3%–5%，同时保持较低的计算和显存成本；

**⚠️ 局限性**

局限性在于：①依赖预训练视觉/文本特征；②模型结构（9个专家）需经验性设置；③熵阈值与窗口等超参对不同任务仍需微调；④在极端噪声或内容不相关的场景中，内容诱导边可能产生负面影响。

---

## 294. HiSAC: Hierarchical Sparse Activation Compression for Ultra-long Sequence Modeling in Recommenders

**arXiv ID:** 2602.21009 | [PDF](https://arxiv.org/pdf/2602.21009v1)

**作者:** Kun Yuan `[一作]` (Alibaba Group), Yuning Jiang `[通讯]` (Alibaba Group)

**通讯引用:** 3933 | [OpenAlex ID](https://openalex.org/A5074655314)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 HiSAC 框架，用层次稀疏激活压缩和软路由注意力来对超长用户行为序列进行高效压缩，生成个性化兴趣代理。

**💡 创新点**

创新点在于：① 通过 Residual RQ‑VAE 将连续嵌入量化为多层语义 ID；② 构造全局层次语义树并用层次投票筛选兴趣代理，解决兴趣中心数量与粒度异质性；③ 用软路由注意力根据语义相似度对历史行为进行软分配，减少量化误差并保留长尾偏好；④ 对语义嵌入与排名嵌入进行解耦，提升模型鲁棒性。

**🔧 技术方法**

核心技术包括：多模态编码器（文本+图像）、Residual Quantized Variational Autoencoder、层次语义树与投票机制、Soft‑Routing Attention、缓存化多头注意力实现在线加速。

**📊 数据集**

使用两套数据集：① 规模化工业数据集（约 200M 用户、1B 商品、30B 交互，最大序列 10k）和 ② Taobao‑MM（8.86M 用户、275M 商品，最大序列 1k），并在两者上进行离线评估与在线 A/B 测试。

**📈 对比分析**

与多种压缩方法（K‑Means、LSH、Patching、Aggregator、ELASTIC、PolyEncoder、PatchRec、Longer）以及 MHA 基线进行对比。HiSAC 在离线 AUC/GAUC 上均领先或相当，尤其在长序列（≥10k）时表现最优；在线 A/B 测试显示 CTR 提升 1.65%，IPV 1.93%，CTCVR 2.24%，订单 2.56%。

**⚠️ 局限性**

局限性包括：① 依赖 RQ‑VAE 代码库的质量，量化误差若过大仍影响性能；② 需要每日离线重建兴趣树，对极活跃或兴趣快速变化的用户响应不够即时；③ 软路由需要额外的语义嵌入与计算，若语义表达不足会导致分配不准；④ 处理极短尾行为时仍可能出现覆盖不足。

---

## 295. UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models

**arXiv ID:** 2602.20231 | [PDF](https://arxiv.org/pdf/2602.20231v1)

**作者:** Manish Kumar Govind `[一作]` (University of North Carolina at Charlotte), Srijan Das `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 804 | [OpenAlex ID](https://openalex.org/A5057061455)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在无监督的RGB‑D视频中学习统一的深度感知潜在动作表示，并用该表示预训练Transformer Vision‑Language‑Action模型UniLACT，随后在仿真和真实机器人任务中微调得到更精准的控制。

**💡 创新点**

提出了UniLARN框架实现RGB与深度的交互式潜在动作学习，构建统一潜在空间；在预训练阶段仅使用深度信息，推理时仅需RGB；通过跨模态自回归潜在预测显著提升空间理解。

**🔧 技术方法**

采用逆向/正向动力学模型、VQ‑VAE离散化、GPT‑2自回归Transformer、跨模态潜在预测以及连续动作解码器等技术。

**📊 数据集**

使用CALVIN仿真基准、Open X‑Embodiment（OXE）预训练数据、Depth‑Anything‑V2生成的深度、以及在xArm7机器人上收集的30条真实遥控演示。

**📈 对比分析**

在CALVIN ABC→D和Oxe预训练设置下，与仅RGB潜在动作的Moto等基线相比，UniLACT在平均序列长度上提升约29%，在真实世界四个任务中的平均成功率提升约10%，显示出显著性能优势。

**⚠️ 局限性**

局限性包括需要额外的RGB‑D或生成深度数据，深度质量影响预训练效果；对纯视觉或非接触式任务的提升有限；跨域推广仍受限于深度感知的一致性和环境复杂度。

---

## 296. Sample-Efficient Learning with Online Expert Correction for Autonomous Catheter Steering in Endovascular Bifurcation Navigation

**arXiv ID:** 2602.20216 | [PDF](https://arxiv.org/pdf/2602.20216v1)

**作者:** Hao Wang `[一作]` (Tongji University), Peng Qi `[通讯]` (Tongji University)

**通讯引用:** 7164 | [OpenAlex ID](https://openalex.org/A5025104684)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于SAC+GAIL+专家在线校正的样本高效强化学习框架，用于血管分叉处的机器人导管自主导航。

**💡 创新点**

创新点在于将最大熵强化学习、对抗模仿学习与实时专家校正结合，使用模糊控制对分叉姿态进行修正，并通过时间可调奖励实现从探索到仿真的平滑过渡。

**🔧 技术方法**

采用Soft Actor-Critic (SAC)、Generative Adversarial Imitation Learning (GAIL)、YOLOv5+U-Net图像分割、模糊控制与常数曲率模型。

**📊 数据集**

使用3D硅胶肾动脉模型配合7,293张增强的YOLO训练图像和1,000张U-Net分割图像，真实机器人实验。

**📈 对比分析**

与TD3、SAC、SAC-GAIL、SAC-EIL等五种算法在300轮实验中对比，SAC-EIL-GAIL在123轮内收敛，成功率59%（比TD3高17.3%），平均误差82.58像素，收敛时间59.41秒，显示出最高的学习效率和精度。

**⚠️ 局限性**

受限于单模态实时图像、仅验证在硅胶模型上，缺乏真实人体血管多样性与模拟到实操的泛化；模糊控制参数需手工设定，系统对极端解剖变异的鲁棒性尚待提升。

---

## 297. CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation

**arXiv ID:** 2602.20170 | [PDF](https://arxiv.org/pdf/2602.20170v1)

**作者:** Chaeyun Kim `[一作]` (Seoul National University), Minwoo Kim `[通讯]` (AI Safety Team, DATUMO INC)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CAGE框架，用以生成文化适配的红队测试基准，并实现了首个韩语基准KorSET；

**💡 创新点**

创新点在于引入“语义模具”概念，将攻击意图与文化内容解耦，既保持原始攻击结构，又能按地方法律和社会语境重写提示；

**🔧 技术方法**

技术包括三阶段流水线：种子提示收集、基于槽位的语义重写以及内容本地化；还构建了多层次安全分类体系并采用多模型一致性标注；

**📊 数据集**

使用了多源英文红队数据集（如SALAD-Bench、ALERT、WildGuard-Mix等）作为种子，结合韩语法律、新闻与社交媒体语料生成本地化内容；

**📈 对比分析**

与三类基线（直接翻译、模板填充、LLM自适应）和四种自动攻击方法（GCG、TAP、AutoDAN、GPT-Fuzzer）对比，CAGE生成的韩语提示在ASR和质量评分上均显著领先，ASR提升约10-30%；

**⚠️ 局限性**

局限性包括：首个实现仅覆盖韩语，低资源语言适配仍受限；对模型误用风险的控制依赖受限发布；以及对模型训练数据多样性的假设未在更广泛语境中验证。

---

## 298. Memory Undone: Between Knowing and Not Knowing in Data Systems

**arXiv ID:** 2602.21180 | [PDF](https://arxiv.org/pdf/2602.21180v1)

**作者:** Viktoriia Makovska `[一作]` (Ukrainian Catholic University), Tetiana Zakharchenko `[通讯]` (Ukrainian Catholic University)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5078740312)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文通过社会技术视角重新定义AI中的遗忘，将其视为治理与权利层面的干预，并提出构建能够负责任地忘记的“忘记机器”概念。

**💡 创新点**

创新点在于将遗忘从单纯技术问题转向社会技术实践，强调遗忘的政治、伦理与治理意义，并提出忘记机器的设计原则。

**🔧 技术方法**

主要采用跨学科理论方法（哲学、记忆研究、批判媒体理论、人机交互），未给出具体算法实现。

**📊 数据集**

未使用具体数据集，主要以文献综述与案例分析为支撑。

**📈 对比分析**

未进行实验比较或性能评估，讨论基于案例与理论阐述的可行性与意义。

**⚠️ 局限性**

局限在于缺乏可验证的技术实现与实证评估，且对实现可行性、治理合规性与多样性等方面的挑战仍未展开深入探讨。

---

## 299. CARE: An Explainable Computational Framework for Assessing Client-Perceived Therapeutic Alliance Using Large Language Models

**arXiv ID:** 2602.20648 | [PDF](https://arxiv.org/pdf/2602.20648v1)

**作者:** Anqi Li `[一作]` (Zhejiang University), Zhenzhong Lan `[通讯]` (Westlake University)

**通讯引用:** 7887 | [OpenAlex ID](https://openalex.org/A5103239171)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了CARE框架，利用LLM自动预测咨询会话中文文本中客观感知的治疗联盟三维评分，并生成可解释的推理理由。

**💡 创新点**

①将专家手工注释的9,516条理由与评分数据融合，形成rationale-augmented监督；②实现多维度评分与上下文关联推理；③提供高质量可解释的理由，提升透明度。

**🔧 技术方法**

基于LLaMA‑3.1‑8B‑Instruct进行全参数微调，采用rationale-augmented监督；对比零-shot提示、GPT‑4o、DeepSeek‑R1、Claude‑3‑Sonnet等大型模型；同时进行自动与人工评估（BLEU/ROUGE/BERTScore、可信度/相关性/信息量）。

**📊 数据集**

使用中国线上心理咨询平台的CounselingWAI数据集（793场次，82位客户）并加入专家注释理由，另在更大规模的2,236场次数据上验证。

**📈 对比分析**

与人类咨询师、开闭源LLM进行对比；CARE在三维评分上分别达到Pearson 0.52、0.50、0.46，提升约70%以上，MSE分别为1.00、1.05、0.70，且标准差显著降低，说明稳定性更好。

**⚠️ 局限性**

仅基于文本，忽略非语言行为；依赖中文咨询文本，跨语言迁移受限；模型在极端情境（如客户迟到、情绪低落）下仍可能错误关注口头内容。

---

## 300. GSNR: Graph Smooth Null-Space Representation for Inverse Problems

**arXiv ID:** 2602.20328 | [PDF](https://arxiv.org/pdf/2602.20328v1)

**作者:** Romario Gualdrón-Hurtado `[一作]` (Universidad Industrial de Santander), Henry Arguello `[通讯]` (Universidad Industrial de Santander)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的图平滑零空间表示（GSNR）方法，用于解决成像中的逆问题，通过引入图拉普拉斯算子来改善图像重建。

**💡 创新点**

GSNR通过在不可见的零空间组件中引入结构信息，克服了传统方法的偏差，提供了更高的收敛性和可预测性。

**🔧 技术方法**

使用了图拉普拉斯算子和低维投影矩阵，结合了图平滑性分析来选择结构化的零子空间。

**📊 数据集**

在多个数据集上进行了验证，包括CIFAR-10、CelebA和Places365，应用于图像去模糊、压缩感知、去马赛克和图像超分辨率等任务。

**📈 对比分析**

与传统的PnP、DIP和扩散求解器相比，GSNR在PSNR上提供了高达4.3 dB的改进，并在端到端学习模型中提高了1 dB的性能。

**⚠️ 局限性**

方法依赖于对传感矩阵的精确知识，且在处理大规模图像时需要大量计算资源，适应非线性逆问题时需要仔细建模。

---

## 301. Position-Aware Sequential Attention for Accurate Next Item Recommendations

**arXiv ID:** 2602.21052 | [PDF](https://arxiv.org/pdf/2602.21052v1)

**作者:** Timur Nabiev `[一作]` (Skolkovo Institute of Science and Technology), Evgeny Frolov `[通讯]` (AIRI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种位置感知核化自注意力机制，在注意力算子内部直接引入可学习的位置信息；

**💡 创新点**

将位置信息与语义解耦，使用可学习的上三角 Toeplitz 核和全下三角核实现多尺度时序建模；

**🔧 技术方法**

采用可学习的 Toeplitz‑Toeplitz/Toeplitz‑Full 等位置核与标准自注意力相结合，并保持因果性；

**📊 数据集**

在七个工业级数据集（ml‑1m、beauty、gowalla、yelp、zvuk、Y‑likes、Y‑listens）上进行评测；

**📈 对比分析**

与经典绝对位置编码、RoPE、CAPE 等基线比较，普遍在 NDCG@10、HR@10 上取得最优或接近最优性能；

**⚠️ 局限性**

在极长序列（如 zvuk）及某些复杂行为模式上提升有限，且与其他位置编码组合时可能产生干扰。

---

## 302. Toward an Agentic Infused Software Ecosystem

**arXiv ID:** 2602.20979 | [PDF](https://arxiv.org/pdf/2602.20979v1)

**作者:** Mark Marron `[一作]` (University of Kentucky), Mark Marron `[通讯]` (University of Kentucky)

**通讯引用:** 1413 | [OpenAlex ID](https://openalex.org/A5024427812)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了一套完整的“Agentic Software Ecosystem”，包括面向AI代理的编程语言、可验证的代码生成工具和支持动态发现与沙盒化的运行时环境。

**💡 创新点**

核心创新在于：① 明确意图与行为的语法表达，减少隐式行为导致的错误；② 在语言中加入“hole”与“meta‑thunk”机制，支持模块化与递归填充；③ 通过SMT驱动的可机理验证与在线反馈，实现代理生成代码的即时安全检查；④ 引入BOSQUE API协议与HATEOAS式的进阶发现，提升代理与服务的互操作性。

**🔧 技术方法**

使用的技术包括：基于Transformer的大语言模型（LLM）作为代理；BOSQUE语言与其编译器/验证器；SMT/LIA/bitvector等可判定理论实现的验证后端；REST/HATEOAS与BOSQUE API协议实现服务发现；多层沙盒与资源权限控制（URI/Glob）。

**📊 数据集**

论文未使用专门的外部数据集；验证与评测主要基于自定义的示例程序、API定义与安全测试用例，采用公开的BOSQUE代码库与标准验证案例。

**📈 对比分析**

与传统手工编码或单纯基于LLM生成的代码相比，提出的系统能够在编译期和运行期自动检测约束违规，理论上将错误率降至几乎零；但论文中未给出量化的性能基准，主要通过案例演示和验证耗时（如7 ms）说明其可行性。

**⚠️ 局限性**

局限性包括：① 对LLM的依赖，性能受模型规模与推理速度影响；② 需要编写显式API与类型声明，增加开发者学习成本；③ 目前尚未在大规模真实项目中验证，缺乏长期可靠性与可扩展性的实测；④ 只关注可判定理论，可能对更复杂的非判定式语义（如动态反射）支持不足。

---

## 303. MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning

**arXiv ID:** 2602.20223 | [PDF](https://arxiv.org/pdf/2602.20223v1)

**作者:** Wall Kim `[一作]` (Samsung Electronics), Hanul Kim `[通讯]` (Seoul National University of Science and Technology)

**通讯引用:** 1704 | [OpenAlex ID](https://openalex.org/A5084140552)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Multi‑Modal Prior‑data Fitted Network (MMPFN)，将 TabPFN 与图像、文本等非表格模态统一处理，实现多模态学习。

**💡 创新点**

创新点在于设计多头门控 MLP（MGM）与跨注意池化器（CAP）两种模块，将非表格嵌入映射到表格空间，并通过聚合平衡 token 数量，缓解注意力不平衡。

**🔧 技术方法**

使用 TabPFN 预训练的基础网络、DINOv2 ViT、ELECTRA 文本编码器、MGM、CAP、轻量级微调、交叉熵损失等技术。

**📊 数据集**

实验涵盖医学与通用多模态数据集：PAD‑UFES‑20、CBIS‑DDSM（Mass、Calc）、Airbnb、Salary、Cloth、PetFinder 等，均包含表格+图像或表格+文本。

**📈 对比分析**

与 TabPFN、Catboost、AutoGluon、MMCL、TIP、HEALNet、TIME 等最新方法对比，MMPFN 在所有基准上均获得最高或次高准确率，并在低样本、模态增量场景下保持优越性能。

**⚠️ 局限性**

局限性包括对预训练的合成表格数据的依赖，需针对不同非表格模态微调，CAP 的 head 数量需要手工调参，且在极端高 token 数的情形下仍可能出现注意力失衡；未验证对更大规模或其他模态（如音频）的泛化。

---

## 304. A Secure and Interoperable Architecture for Electronic Health Record Access Control and Sharing

**arXiv ID:** 2602.20830 | [PDF](https://arxiv.org/pdf/2602.20830v1)

**作者:** Tayeb Kenaza `[一作]`, Sami Messai `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出基于私有区块链与IPFS的电子病历共享与访问控制体系，并实现原型验证。

**💡 创新点**

创新点在于将患者主导的授权机制与智能合约相结合，实现EHR的可追溯、可撤销授权，并采用离链大文件存储降低链上负载。

**🔧 技术方法**

使用Hyperledger Fabric（私有链）+IPFS进行分布式存储，并配合AES‑256加密、数字签名等加密技术实现安全控制。

**📊 数据集**

实验未使用真实医疗数据，采用模拟场景和自生成的EHR文件进行原型测试。

**📈 对比分析**

通过Hyperledger Caliper对TPS、延迟、CPU、内存等指标进行测试，写事务可达约45 TPS、读事务可达600 TPS，写延迟稳定在1.1–1.2 s，CPU占用随TPS线性增长，内存保持在约1.4 GB。

**⚠️ 局限性**

主要局限在缺乏真实数据验证、未覆盖急诊、调查与研究等关键场景，以及对大规模多机构扩展的性能与密钥管理安全性仍需进一步评估。

---

## 305. Prior-Agnostic Incentive-Compatible Exploration

**arXiv ID:** 2602.20465 | [PDF](https://arxiv.org/pdf/2602.20465v1)

**作者:** Ramya Ramalingam `[一作]` (University of Pennsylvania), Aaron Roth `[通讯]` (University of Pennsylvania)

**通讯引用:** 17289 | [OpenAlex ID](https://openalex.org/A5057693522)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究在多代理环境下，探索型多臂赌博机的激励兼容性问题，提出利用加权交换惩罚与外部惩罚约束来保证代理人遵从推荐的逼近贝叶斯纳什均衡

**💡 创新点**

首次将加权交换惩罚与代理人对到达时间的不确定性关联，证明只需满足对奖励慢速变化和到达时间分散性假设，即可在不假设共通先验或先验知识的前提下实现近似激励兼容；同时展示可通过已有自适应惩罚算法实现此目标

**🔧 技术方法**

利用加权交换惩罚、外部惩罚、马尔科夫/随机游走理论、Azuma不等式、混合先验/凸包技巧，并结合Exp4.S等自适应惩罚算法

**📊 数据集**

无实验数据集，全部为理论分析与证明

**📈 对比分析**

通过理论推导与比较，证明在满足假设下，使用自适应惩罚算法可实现误差趋于零的激励兼容性；对比传统需要共享先验的BIC方法，展示更宽松的前提和更广泛的适用性

**⚠️ 局限性**

局限性在于需满足奖励慢变（ρ≪1/√T）和到达时间分散性假设，且仅在纯随机游走/分布不确定性下适用，缺乏实际数据验证与对更复杂环境（如多臂组合、情境归纳）的实验评估

---

## 306. Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression

**arXiv ID:** 2602.20650 | [PDF](https://arxiv.org/pdf/2602.20650v1)

**作者:** Chenyue Yu `[一作]` (Agency for Science Technology and Research), Yang He `[通讯]` (Agency for Science Technology and Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Dataset Color Quantization (DCQ) 框架，通过在数据集层面进行颜色量化，实现了对训练数据的显著压缩，同时保持模型训练性能。

**💡 创新点**

创新点在于：①利用Chromaticity-Aware Clustering (CAC) 在图像色彩分布相似的样本之间共享调色板；②引入Attention-Guided Palette Allocation，依据模型注意力突出重要语义区域；③采用Texture-preserved Palette Optimization，使用可微分的量化和边缘保持损失来提升纹理保真度；④将颜色量化与数据集压缩策略相结合，解决单纯样本剪枝在高压缩率下性能急剧下降的问题。

**🔧 技术方法**

主要技术包括：K-means聚类、浅层特征映射（ResNet-18第一块输出）、Grad-CAM++/RISE等注意力映射、LAB颜色空间转换、可微分的直通估计器（STE）以及基于Sobel算子计算的边缘保持损失。

**📊 数据集**

在CIFAR-10、CIFAR-100、Tiny-ImageNet和ImageNet-1K四个标准数据集上进行实验，使用ResNet-18/ResNet-34/ResNet-50等架构。

**📈 对比分析**

与传统颜色量化方法（ColorCNN、CQFormer、MedianCut等）以及数据集剪枝/蒸馏方法（EL2N、CCS、TDDS等）对比。DCQ在相同压缩率下，尤其是低位深（1-3bit）时，显著优于其他方法，最高可达到94.39%（CIFAR-10）和66.99%（ImageNet-1K）等性能；在极端压缩率下（≥99%）与剪枝方法联用仍能保持高于70%的准确率。

**⚠️ 局限性**

局限性包括：①目前仍采用统一的色彩量化位深，缺乏对不同样本自适应的细粒度策略；②对不同网络架构的适配性未系统评估，需要针对低位深输入设计专用模型；③在极低位深（1bit）下仍存在显著的准确率下降，说明纹理与语义信息不可完全压缩；④实验主要集中在图像分类任务，对检测/分割等更复杂任务的适用性尚待验证。

---

## 307. Architecting AgentOS: From Token-Level Context to Emergent System-Level Intelligence

**arXiv ID:** 2602.20934 | [PDF](https://arxiv.org/pdf/2602.20934v1)

**作者:** ChengYou Li `[一作]` (Yishu Research), XinYu Zhao `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了AgentOS框架，将大型语言模型（LLM）从单纯的推理引擎转变为可管理的认知操作系统；

**💡 创新点**

创新点在于将传统操作系统概念（进程调度、内存分页、I/O中断）映射到LLM的“推理核”，并引入语义切片、认知同步脉冲和感知对齐协议；

**🔧 技术方法**

使用Transformer自注意力机制、语义分页、语义哈希、事件驱动中断、认知调度算法等技术；

**📊 数据集**

未公开实验数据集，框架以理论和概念验证为主；

**📈 对比分析**

通过构造的系统级指标（认知延迟、上下文利用率、同步稳定指数）与传统模型包装方式对比，展示AgentOS在效率与稳定性上的潜在优势；

**⚠️ 局限性**

主要局限包括认知上下文切换开销、语义分页延迟、同步成本随多智能体规模呈指数增长，以及缺乏实际硬件实现与大规模实验验证。

---

## 308. Smoothly Differentiable and Efficiently Vectorizable Contact Manifold Generation

**arXiv ID:** 2602.20304 | [PDF](https://arxiv.org/pdf/2602.20304v1)

**作者:** Onur Beker `[一作]` (University of Tübingen), Georg Martius `[通讯]` (University of Tübingen)

**通讯引用:** 1792 | [OpenAlex ID](https://openalex.org/A5001474340)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一套能够在向量化计算框架下平滑可微、快速生成接触流形的算法。

**💡 创新点**

核心创新包括：①基于超四面体、凸多面体和定向点云三类解析原语构建可微SDF，并通过soft top‑K实现加速选取；②采用解析活跃集法对顶点–SDF、边–边碰撞进行求解，结合L2正则化与平滑判定，得到可微的签名距离和法线。

**🔧 技术方法**

技术手段：可微SDF（logsumexp、smooth union/subtraction）、soft比较/softclip/softargmax、解析活跃集求解QP、L2正则化、向量化实现（JAX/CUDA）。

**📊 数据集**

使用了自制的立方体碰撞和非凸“armadillo”网格（18个超四面体、305个顶点），并在多种随机配置下进行测试。

**📈 对比分析**

与Mujoco XLA、MJX仿真器进行跑时和梯度计算比较；在相同几何复杂度下，向量化批量为10⁷时实现数十倍甚至百倍加速，梯度计算速度约为原方法一半，整体性能显著提升。

**⚠️ 局限性**

局限性：需要手动选择合适的SDF原语、顶点/边的top‑K参数，SDF与网格间不一致会导致误差；对极大规模几何、复杂非凸形状的自动化分解尚未成熟，且平滑参数需手动调节以平衡物理真实性与数值稳定性。

---

## 309. GA-Drive: Geometry-Appearance Decoupled Modeling for Free-viewpoint Driving Scene Generatio

**arXiv ID:** 2602.20673 | [PDF](https://arxiv.org/pdf/2602.20673v1)

**作者:** Hao Zhang `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 40773 | [OpenAlex ID](https://openalex.org/A5100732450)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

GA-Drive 提供了一种可自由视角、可编辑的高保真驾驶仿真框架，能够在给定录制轨迹的基础上合成任意新轨迹的摄像机视图并支持外观编辑。

**💡 创新点**

其创新点在于将几何与外观解耦，并通过伪视图模拟管线在仅有单轨数据的情况下训练段式视频扩散模型，从而实现多轨道一致性、高质量视图合成以及对外观的全局编辑。

**🔧 技术方法**

主要技术包括：OmniRe 4D Gaussian 场景重建（配合密集深度监督与深度畸变正则化）、伪视图合成与可视性检查、基于 I2V 的段式视频扩散模型，以及伪视图模拟流程。

**📊 数据集**

使用的数据集为 Waymo 训练/验证集以及从 OpenDV 选取的 700 段多天气驾驶视频。

**📈 对比分析**

在 Waymo 验证集的车道变换场景下，GA-Drive 在 NTA‑IoU、NTL‑IoU 与 FID 三项指标上均超过所有基线（PVG、S3Gaussian、DriveDreamer4D、ReconDreamer、Deformable‑GS），分别达到 0.558、56.83 与 49.77 的最佳表现。

**⚠️ 局限性**

局限性包括：对 OmniRe 4D 重建的依赖，可能受滚动快门误差影响；当前框架仅支持外观编辑，尚未实现几何编辑。

---

## 310. LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration

**arXiv ID:** 2602.20497 | [PDF](https://arxiv.org/pdf/2602.20497v1)

**作者:** Peiliang Cai `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14526 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于阶段感知的可学习特征预测框架LESA，用以加速Diffusion Transformer（DiT）模型的推理；

**💡 创新点**

创新点在于：①使用Kolmogorov–Arnold Network（KAN）建模非线性时序依赖；②将采样过程分为高噪、中噪、低噪三阶段，为每一阶段配备专门的预测专家；③采用两阶段训练策略（先使用真实标签训练，再进行闭环自回归训练），提升预测鲁棒性；

**🔧 技术方法**

技术包括：特征缓存与预测、KAN网络、阶段分段（分段窗口大小不同）、闭环自回归训练、线性投影与时间步调制因子相乘的残差预测；

**📊 数据集**

在文本到图像任务上使用FLUX.1-dev、FLUX.1-schnell、Qwen-Image及其蒸馏版；在文本到视频任务上使用HunyuanVideo；数据集覆盖多样化的图像与视频生成场景；

**📈 对比分析**

与TaylorSeer、TeaCache等现有缓存/预测方法对比，LESA在FLUX、Qwen-Image上分别实现5.00×和6.25×的加速，质量仅下降≤1.0%；在HunyuanVideo上实现5.00×加速，PSNR提升24.7%；整体保持或提升ImageReward、CLIP Score、PSNR、SSIM、LPIPS等指标；

**⚠️ 局限性**

局限性包括：对极端加速比例（如N>10）时仍存在一定的质量下降；目前只针对Transformer结构的Diffusion模型验证，其他网络如U-Net的适用性尚未充分探究；

---

## 311. Oracle-Robust Online Alignment for Large Language Models

**arXiv ID:** 2602.20457 | [PDF](https://arxiv.org/pdf/2602.20457v1)

**作者:** Zimeng Li `[一作]` (Purdue University), Vaneet Aggarwal `[通讯]` (Purdue University)

**通讯引用:** 6207 | [OpenAlex ID](https://openalex.org/A5064822688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在偏置偏好反馈下，LLM 的在线对齐问题；通过构造点状不确定集合，定义了最坏情况下的对齐目标并推导出可解析的分解；

**💡 创新点**

创新点在于：①提出点状不确定集合并把最坏情况对齐目标拆成原始 SAIL 损失与显式敏感度惩罚；②证明该目标是弱凸的，并给出对应的 Moreau 包络阶梯收敛分析；

**🔧 技术方法**

使用的技术包括：对齐目标的闭式分解、弱凸性分析、Moreau 包络与投影随机复合梯度下降（R‑SCGD）算法、弱凸优化的理论复杂度证明；

**📊 数据集**

文中未给出具体实验数据集，理论分析基于假设的 log‑linear 策略与有限响应空间；

**📈 对比分析**

方法与原始 SAIL（即 ρ=0）对比，理论上可保证 O(ε⁻²) 的样本复杂度；实验结果未报告，故无法给出性能数值；

**⚠️ 局限性**

局限性包括：仅针对 log‑linear 模型；需要有限且特征受限的响应空间；理论假设较强，如弱凸性、梯度方差界限；缺乏经验验证。

---

## 312. Hierarchical Molecular Representation Learning via Fragment-Based Self-Supervised Embedding Prediction

**arXiv ID:** 2602.20344 | [PDF](https://arxiv.org/pdf/2602.20344v1)

**作者:** Jiele Wu `[一作]` (National University of Singapore), Tze Yun Leong `[通讯]` (National University of Singapore)

**通讯引用:** 1679 | [OpenAlex ID](https://openalex.org/A5039680902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种分层自监督框架GraSPNet，利用节点与化学碎片两层语义的掩码预测实现分子图表征学习。

**💡 创新点**

创新点在于无词典化的基于几何与化学结构的碎片化方法，构建碎片图并在多级消息传递中同时预测节点与碎片嵌入，提供多分辨率语义监督。

**🔧 技术方法**

采用图神经网络的多层消息传递、节点-碎片交互、掩码自监督、目标编码器的指数滑动平均、碎片嵌入编码等技术。

**📊 数据集**

预训练使用ZINC15 200万分子，微调在MoleculeNet的八个分类任务（BBBP、Tox21、ToxCast、SIDER、Clintox、MUV、HIV、BACE）以及三个回归任务（FreeSolv、ESOL、Lipophilicity）。

**📈 对比分析**

与对比学习、生成式、自监督预测以及其他碎片基方法对比，GraSPNet在大多数任务上取得最优或次优结果，特别在Clintox、BACE、BBBP等任务上显著提升。

**⚠️ 局限性**

局限在于碎片化策略专为分子图设计，难以直接迁移至非分子图任务；对大分子碎片化可能导致效率下降；缺乏多模态或语言监督的扩展。

---

## 313. SibylSense: Adaptive Rubric Learning via Memory Tuning and Adversarial Probing

**arXiv ID:** 2602.20751 | [PDF](https://arxiv.org/pdf/2602.20751v1)

**作者:** Yifei Xu `[一作]` (University of California, Los Angeles), Tusher Chakraborty `[通讯]` (Microsoft)

**通讯引用:** 210 | [OpenAlex ID](https://openalex.org/A5043473455)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SibylSense，利用冻结的 rubric 生成器与可调记忆库实现推理时学习，通过验证器的 discriminative gap 优化 rubric，并引入对抗候选刷新提升鲁棒性。

**💡 创新点**

创新点在于把适应性 rubric 生成转为记忆调优而非权重更新；引入对抗候选刷新循环，使 rubric 随 policy 进化；使用验证器反馈作为记忆更新信号。

**🔧 技术方法**

使用冻结 LLM 生成器（Qwen3‑32B）、LLM 验证器（GPT‑4o）、RL 算法（GRPO）、记忆检索与分类、对抗生成等技术。

**📊 数据集**

在 RaR‑Medicine（医学问答）和 GovReport（政府报告摘要）两类开放式生成任务上进行实验。

**📈 对比分析**

与原始 rubric、few‑shot rubric 以及静态 rubric 基线对比，在 preference accuracy 与 RL win‑rate 上均优于基线；对抗刷新进一步提升性能，尤其在 GovReport 上显著。

**⚠️ 局限性**

局限性包括对抗候选刷新需要额外采样，依赖验证器质量；仅在两任务验证，通用性待进一步评估；记忆规模与检索效率需进一步优化。

---

## 314. AnimeAgent: Is the Multi-Agent via Image-to-Video models a Good Disney Storytelling Artist?

**arXiv ID:** 2602.20664 | [PDF](https://arxiv.org/pdf/2602.20664v1)

**作者:** Hailong Yan `[一作]` (University of Electronic Science and Technology of China), Bo Li `[通讯]` (vivo Mobile Communication Co. Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AnimeAgent，一个基于Image-to-Video的多代理框架，用于生成高质量、多角色连贯的定制分镜头；

**💡 创新点**

创新点在于将I2V模型与双重Dope Sheet（文本+视觉）相结合，配合一致性评审与混合主客观评审实现动态一致性、提示忠实度和表现力的自我迭代优化；

**🔧 技术方法**

技术包括大型多模态语言模型（Qwen3VL）、I2V生成器（Wan2.2-I2V-14B）、文本与视觉Dope Sheet结构、光流与关键点跟踪的运动评分、以及主客观混合评审框架；

**📊 数据集**

使用了ViStoryBench和自研的AnimeBoard‑GT（含人类标注的真实分镜）作为评估数据集；

**📈 对比分析**

通过与13种CSG方法及9个商业平台在ViStoryBench与AnimeBoard‑GT上对比，AnimeAgent在CSD、CIDS、PA、Aes、CLIP‑I等指标上均实现SOTA，显示显著提升的一致性、提示符合度与美学质量；

**⚠️ 局限性**

局限在于依赖大规模MLLM和I2V模型，导致算力和推理延迟较高，未来需探索更轻量化的实现以支持实时部署。

---

## 315. Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis

**arXiv ID:** 2602.20207 | [PDF](https://arxiv.org/pdf/2602.20207v1)

**作者:** Shrestha Datta `[一作]` (University of South Florida), Anshuman Chhabra `[通讯]` (University of South Florida)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5022645982)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型知识编辑中的层选择问题，提出黄金层概念并提出基于梯度归因的Layer Gradient Analysis (LGA) 方法，用以高效估计黄金层。

**💡 创新点**

创新点在于：①发现大多数样本在同一层或少数几层实现最佳编辑效果，从而提出黄金层；②提出仅使用一阶梯度信息、无须实际编辑即可快速定位黄金层的LGA方法；③通过实验验证LGA在多模型、多编辑方法和多数据集上均优于传统的Causal Mediation Analysis (CMA)，并显著提升计算效率。

**🔧 技术方法**

采用的技术包括：梯度归因、层级梯度内积、第一阶梯度聚合、统计检验（t 检验）以及多种编辑算法（R-ROME、ROME、EMMET）和评估指标（Rewrite Accuracy、Rephrase Accuracy、Locality、Portability、Fluency、Overall）。

**📊 数据集**

使用的数据集为 ZSRE、WikiBio、WikiCounterfact、WikiRecent、Counterfact，模型涵盖 GPT-2 XL、LLaMA2-7B、Gemma3-12B。

**📈 对比分析**

通过在相同 proxy 集上对比 LGA 与 CMA，评估 Rewrite Accuracy 等指标，并在多个模型和编辑方法上实现平均提升约 2–8% 的 Rewrite Accuracy，整体分数提升约 1–3%；在运行时间上，LGA 相比 CMA 以及 brute‑force 全层搜索可获得 7~12 倍的加速。

**⚠️ 局限性**

局限性包括：在 WikiBio 等部分数据集黄金层表现略逊于样本最优层；对极长输入或特定模型的适用性仍待进一步验证；方法目前仅针对单层编辑，未探讨多层或跨层联合编辑的场景。

---

## 316. IMOVNO+: A Regional Partitioning and Meta-Heuristic Ensemble Framework for Imbalanced Multi-Class Learning

**arXiv ID:** 2602.20199 | [PDF](https://arxiv.org/pdf/2602.20199v1)

**作者:** Soufiane Bacha `[一作]` (University of Science and Technology Beijing), Huansheng Ning `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 12517 | [OpenAlex ID](https://openalex.org/A5102790255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 IMOVNO+ 框架，联合数据层级的分区、噪声清理、改进的 SMOTE 与正则化，以及算法层级的 Jaya 元启发式集成裁剪，以同时解决多类不平衡、类别重叠与噪声问题。

**💡 创新点**

创新点在于：①直接对多类问题进行信息量评估与分区，避免二分类拆分损失类别关系；②使用大跳距+Z‑score 进行重叠清理；③在 SMOTE 中加入多正则化惩罚控制合成样本位置；④将 Jaya 应用于集成裁剪，参数无关且高效。

**🔧 技术方法**

采用条件概率划分、Z‑score＋大跳距、改进 SMOTE（OMRP）、Jaya 元启发式裁剪，并以决策树、k‑NN、朴素贝叶斯、ExtraTree 等弱分类器构建集成。

**📊 数据集**

实验使用 35 个公开不平衡数据集（13 个多类、22 个二类），来源于 KEEL 与 UCI，覆盖 IR 从 1.13 到 92.6 的范围。

**📈 对比分析**

与 SMOTE‑CDNN、ECDNN、SAMME.C2、CPS‑3WS、Counterfactual SMOTE、MLOS、HSCF、GDDSAD 等最新方法及 OVO+basic‑SMOTE 进行对比，评估 G‑mean、F1、精确率、召回率与准确率，IMOVNO+ 在多类任务上提升 37–57% G‑mean、25–44% F1、24–39% 精确率、26–43% 召回率，二类任务接近 100% 的表现。

**⚠️ 局限性**

局限在于：①对条件概率阈值和大跳距参数的敏感性；②在极度不平衡或高维稀疏数据上的可扩展性待验证；③仅在表格数据上测试，缺乏对图像、文本等非结构化数据的评估；④元启发式裁剪虽参数少，但仍需多次迭代。

---

## 317. PIME: Prototype-based Interpretable MCTS-Enhanced Brain Network Analysis for Disorder Diagnosis

**arXiv ID:** 2602.21046 | [PDF](https://arxiv.org/pdf/2602.21046v1)

**作者:** Kunyu Zhang `[一作]` (Zhengzhou University), Shujian Yu `[通讯]` (UiT - The Arctic University of Norway)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种结合信息瓶颈压缩、原型学习与蒙特卡洛树搜索的可解释fMRI诊断框架PIME；

**💡 创新点**

创新点在于将原型一致性作为语义锚点，利用一致性正则化与结构扰动训练鲁棒表征，再通过MCTS在原型引导下搜索最小足够子图，获得可解释且稳定的诊断依据；

**🔧 技术方法**

技术包括基于GIN的变分信息瓶颈编码器、可学习原型与注意力权重的原型分类器、结构扰动一致性正则化、稀疏与多样性约束以及MCTS解释子图搜索；

**📊 数据集**

使用ABIDE（ASD）、ADNI（AD/MCI/NC）和ADHD-200（ADHD/NC）三大公共rs‑fMRI数据集；

**📈 对比分析**

与多种GNN、Transformer及可解释方法（如BrainIB、PGIB等）进行对比，PIME在ABIDE、ADNI和ADHD‑200的分类准确率均优于现有方法（例如ABIDE下ASD分类72.47%），并在解释子图稀疏度与重现性上表现更好；

**⚠️ 局限性**

局限性包括对不同模板/脑区划分的跨域泛化尚不充分，对大规模多模态数据的扩展待验证，以及需要进一步评估临床实用性和解释的临床可验证性。

---

## 318. OptiLeak: Efficient Prompt Reconstruction via Reinforcement Learning in Multi-tenant LLM Services

**arXiv ID:** 2602.20595 | [PDF](https://arxiv.org/pdf/2602.20595v1)

**作者:** Longxiang Wang `[一作]` (City University of Hong Kong), Cong Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 25582 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于强化学习的提示重建框架（SFT + 自动注释的 DPO），用于在多租户 LLM 服务的 KV‑cache 侧通道中高效泄露用户提示。

**💡 创新点**

创新点包括：① 用模型预测难度（token 似然排名）自动识别“硬 token”，从而自动生成偏好对，省去人工标注；② 在 SFT 后直接采用 DPO 进行偏好优化，避免传统超拟合问题并显著提升攻击效率。

**🔧 技术方法**

技术手段包括：监督微调（SFT）、直接偏好优化（DPO）与自动注释、基于最长前缀匹配（LPM）的 KV‑cache 侧通道检测、以及强化学习中的策略优化。

**📊 数据集**

实验使用了医学领域的 MedQA、PubMedQA 数据集以及金融领域的 FinanceBench 数据集。

**📈 对比分析**

与基线（原始 LLM、仅 SFT）相比，实验在 Qwen‑2.5 3B/7B/14B 及 Llama‑3.1‑8B 上均实现了 12.48× 的平均请求/令牌（ARPT）下降，攻击成功率（ASR）显著提升，且在不同模型规模上保持稳健性能。

**⚠️ 局限性**

局限性：依赖攻击者对目标领域的先验知识，若知识偏差大则效果下降；目前仅在特定的 KV‑cache 共享架构（如 vLLM、SGLang）验证，其他调度策略或缓存机制的适用性需进一步评估；框架侧重攻击，缺乏针对性防御方案的实验验证。

---

## 319. Permutation decoding of algebraic geometry codes from Hermitian and norm-trace curves

**arXiv ID:** 2602.20455 | [PDF](https://arxiv.org/pdf/2602.20455v1)

**作者:** Monica Lichtenwalner `[一作]` (Virginia Tech), Padmapani Seneviratne `[通讯]` (Texas A and M University Commerce)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究并构造了适用于 Hermitian 曲线和 norm‑trace 曲线的一点代数几何码的置换译码方案，提出了专门用于校正特定突发错误的 PD 集合。

**💡 创新点**

首次在正曲率曲线上展示置换译码，并利用曲线的自同构群提供了显式的 PD 集合，从而扩展了置换译码技术的适用范围。

**🔧 技术方法**

使用了置换译码理论、曲线自同构群、Riemann–Roch 空间以及系统化生成矩阵的构造方法。

**📊 数据集**

本工作为理论研究，无使用具体数据集；所有结果均基于代数几何与群论的推导。

**📈 对比分析**

未给出实验对比或性能数值，只通过理论分析说明该方法可在特定突发错误模式下实现错误校正，并讨论了算法的时间复杂度。

**⚠️ 局限性**

局限性包括：仅能校正特定的突发错误模式；需要充分了解曲线的自同构群；对一般错误模式或更广泛的 AG 码适用性有限；计算自同构群和构造 PD 集合的复杂度可能较高。

---

## 320. N4MC: Neural 4D Mesh Compression

**arXiv ID:** 2602.20312 | [PDF](https://arxiv.org/pdf/2602.20312v1)

**作者:** Guodong Chen `[一作]` (Northeastern University), Mallesham Dasari `[通讯]` (Northeastern University)

**通讯引用:** 488 | [OpenAlex ID](https://openalex.org/A5021638193)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了首个神经网络驱动的4D网格压缩框架N4MC，能够将时间变化的三维网格序列高效压缩并实现实时解码；

**💡 创新点**

创新点在于将连续网格转换为4D TSDF‑Def张量，并通过自编码器+自解码器学习时空冗余；利用体积跟踪生成运动先验，配合轻量级Transformer实现中间帧的3D插值，显著提升压缩率与视觉质量；

**🔧 技术方法**

技术包括TSDF‑Def表示、3D卷积自编码器/自解码器、量化感知训练、体积跟踪与轨迹编码、点云编码器生成运动先验、跨帧交叉注意力Transformer、量化线性层、Deformable Marching Cubes等；

**📊 数据集**

使用MPEG V‑DMC标准的四个真实动作序列（Dancer、Basketball Player、Mitch、Thomas）以及合成混合多物体数据集和Thingi10K等；

**📈 对比分析**

与NeCGS、TVMC、Draco、KLT等基线比较，N4MC在相同比特率下实现更高的D2‑PSNR、SSIM、PSNR，并保持约24+ FPS实时解码；在移动设备上亦能在Quest 3等平台实现实时播放；

**⚠️ 局限性**

局限性包括：对长序列的组大小与总帧数敏感；解码仍受GPU/CPU性能限制；对极端非刚性运动或复杂拓扑变化的鲁棒性尚待验证；

---

## 321. BoxSplitGen: A Generative Model for 3D Part Bounding Boxes in Varying Granularity

**arXiv ID:** 2602.20666 | [PDF](https://arxiv.org/pdf/2602.20666v1)

**作者:** Juil Koo `[一作]` (KAIST), Minhyuk Sung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于盒子分割的交互式3D形状生成框架，利用递归拆分的方式从粗略的盒子逐步生成细粒度的形状；

**💡 创新点**

1）设计了自回归的盒子分割模型，包含枢轴盒子分类器和基于扩散的子盒子生成器；2）改进的Box2Shape模型通过ControlNet直接将盒子信息映射到预训练的3DShape2VecSet中，实现高质量、对齐的形状合成；

**🔧 技术方法**

Transformer、基于扩散的条件生成、ControlNet、VQ‑VAE、SMART的盒子层级合并方法；

**📊 数据集**

ShapeNet数据集（通过SMART生成的层级盒子作为训练样本）；

**📈 对比分析**

与基于令牌预测的序列模型、无条件扩散模型的修补方法、Spice‑E以及基于门控机制的3DShape2VecSet进行对比，Box2Shape在COV、MMD、1‑NNA、VIoU、Box‑CD/EMD等指标上均优于对手，盒子分割模型在覆盖率与距离指标上也显著优于基线；

**⚠️ 局限性**

仅使用盒子作为条件限制了对细粒度几何细节的直接控制，扩散过程仍需较多计算资源，且目前未支持其他类型的空间引导（如曲面、草图）。

---

## 322. Changing the Optics: Comparing Traditional and Retrieval-Augmented GenAI E-Tutorials in Interdisciplinary Learning

**arXiv ID:** 2602.20544 | [PDF](https://arxiv.org/pdf/2602.20544v1)

**作者:** Hannah Kim `[一作]` (Temple University), Stephen MacNeil `[通讯]` (Temple University)

**通讯引用:** 1955 | [OpenAlex ID](https://openalex.org/A5042822346)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对传统电子教程与基于检索增强生成（RAG）的GenAI电子教程进行对照实验，探究两种教学模式对学习者信息寻求行为的影响。

**💡 创新点**

首次将或ienteering框架与RAG技术结合，用来对比传统教程与GenAI教程在认知负荷、信息空间意识和探索行为上的差异，并提出两种模式的设计取向与学习者的主动性关系。

**🔧 技术方法**

检索增强生成（RAG）系统、LangChain框架、Groq客户端、Ollama嵌入模型、Chroma向量数据库、Guardrail安全模型。

**📊 数据集**

HyPhy生物信息学工具的官方文档与自制电子教程内容。

**📈 对比分析**

使用NASA‑TLX量表、问卷调查、屏幕录制与访谈等多模态方法比较，结果显示GenAI组认知负荷平均低20分，探索行为显著增加，传统组信息空间意识更强。

**⚠️ 局限性**

样本量仅10人，招募方式导致选择偏差，且技术快速迭代可能使实验条件在短时间内失效。

---

## 323. Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning

**arXiv ID:** 2602.21103 | [PDF](https://arxiv.org/pdf/2602.21103v1)

**作者:** Sanket Badhe `[一作]` (Google), Deep Shah `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Prompt-Level Distillation方法，将教师模型的推理模式抽取成系统提示，直接注入学生模型实现零参数、零推理开销的推理转移。

**💡 创新点**

创新点在于通过非参数化的提示注入实现知识蒸馏，避免微调、保持可解释性，并通过监督指令抽取、聚类与冲突解决形成统一的逻辑指令集。

**🔧 技术方法**

采用链式推理抽取指令、文本嵌入与DBSCAN聚类、闭环冲突解决循环以及Gemini/Embedding等模型技术。

**📊 数据集**

使用StereoSet与Contract NLI两大分类基准数据集进行评估。

**📈 对比分析**

与零射击、少量射击基线对比，PLD在Gemma‑3 4B上将Macro‑F1提升至StereoSet 0.90、Contract‑NLI 0.83，显著超过基线且接近教师模型性能。

**⚠️ 局限性**

局限性在于仅适用于静态决策、无法外化的动态推理任务，且指令集随任务复杂度增大可能超出上下文窗口。

---

## 324. PhantomRun: Auto Repair of Compilation Errors in Embedded Open Source Software

**arXiv ID:** 2602.20284 | [PDF](https://arxiv.org/pdf/2602.20284v1)

**作者:** Han Fu `[一作]` (Ericsson AB), Cyrille Artho `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 2012 | [OpenAlex ID](https://openalex.org/A5063347761)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个自动化框架，用大语言模型（LLM）在嵌入式开源软件的 CI 过程中自动重现、分析并修复编译失败；

**💡 创新点**

关键创新在于将 LLM 迁移到多种 CI 平台（GitHub Actions、GitLab CI）和多种构建系统（CMake、SCons、Make、Buildroot）上，并通过提示工程结合历史修复示例显著提升修复成功率；

**🔧 技术方法**

核心技术包括 CI 环境重现、日志解析与错误归类、基于提示的 LLM 修复（CodeT5+、CodeLlama、Falcon、Bloom）以及多轮自动验证；

**📊 数据集**

使用来自四个主流嵌入式 OSS 项目（OpenIPC、STM32、RTEMS、Zephyr）的约 1 万条 PR/MR，重现 4248 次编译错误；

**📈 对比分析**

与传统手工修复和不同 LLM 组合进行对比，最高修复成功率为 45%（CodeLlama+项目内示例），修复大多仅需 0–2 行代码，显示低成本可行；

**⚠️ 局限性**

局限性包括仅针对单文件修复、缺乏语义正确性检测、对硬件依赖错误的修复率较低、LLM 训练数据偏差可能引入新错误、实验数据仅覆盖四个项目，缺乏更广泛验证。

---

## 325. ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments

**arXiv ID:** 2602.21140 | [PDF](https://arxiv.org/pdf/2602.21140v1)

**作者:** Haley Li `[一作]` (Huawei Technologies Canada), Zhenan Fan `[通讯]` (Huawei Technologies Canada)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

针对大规模LLM推理服务，提出一种不需要重新启动实例的快速失败恢复系统。

**💡 创新点**

创新点包括：1) 通过心跳检测实现单卡故障定位；2) 使用日志恢复KV块表，保持推理状态；3) 通过冗余专家、角色切换或容忍专家丢失来保证权重量完整；4) 预编译图缓存实现从10秒级到几秒级的图编译；5) 在两种部署模式（MoE‑attention 同机与分机）均适用。

**🔧 技术方法**

主要技术：xDeepServe 推理框架、Huawei CloudMatrix384、XCCL 通信库、DP/TP/EP 并行模式、心跳检测、日志式 KV 块表恢复、缓存编译（cached compile）。

**📊 数据集**

使用 DeepSeek V3 671B 模型（80 张 Ascend 64GB NPU），在 CloudMatrix384 上进行实验。

**📈 对比分析**

与基线完全重新初始化相比，恢复时间从 83.1 秒降低到 10.2 秒（87.8%），即使需要角色切换并重新加载权重，仍比基线快 36.6%；实验还表明，在 EP≥32 的配置下，丢失 1/32 专家对模型准确率影响极小。

**⚠️ 局限性**

局限性：仅处理单卡故障；不支持慢速卡、功耗下降等隐性故障；无法应对多卡或网络分区等大规模失效；冗余专家布局需在性能与容错间权衡。

---

## 326. Motivation is Something You Need

**arXiv ID:** 2602.21064 | [PDF](https://arxiv.org/pdf/2602.21064v1)

**作者:** Mehdi Acheli `[一作]` (Telecom SudParis), Walid Gaaloul `[通讯]` (Telecom SudParis)

**通讯引用:** 3517 | [OpenAlex ID](https://openalex.org/A5084851123)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于情感神经科学的双模型交替训练框架，利用在满足动机条件（连续批次损失下降）时切换至更大模型来提升基础模型与激励模型的性能。

**💡 创新点**

创新点在于将情感激励状态映射到训练过程中，通过可扩展网络的权重映射实现部分参数的动态激活与冻结，形成“训练一次，部署两次”的高效双模型策略。

**🔧 技术方法**

采用可扩展网络（ResNet、ViT、EfficientNet）、动机条件判定（k 连续批次损失下降）、权重映射与优化器状态复制、以及伪代码实现的交替训练算法。

**📊 数据集**

使用了CIFAR‑10、CIFAR‑100、ImageNet、Flowers、Pets等图像分类数据集进行实验验证。

**📈 对比分析**

通过 ACC/FLOPs 效率指标与经典训练对比，实验显示基础模型在效率上提升至最高122倍，激励模型在某些配置下性能超过同级传统模型，整体训练成本低于单独训练大模型。

**⚠️ 局限性**

局限性包括：方法主要在可扩展架构（如 EfficientNet）中效果突出；动机条件采用经验式阈值，缺乏自适应学习；权重映射需手工定义，限制了通用性。

---

## 327. Qwen-BIM: developing large language model for BIM-based design with domain-specific benchmark and dataset

**arXiv ID:** 2602.20812 | [PDF](https://arxiv.org/pdf/2602.20812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 328. Exploring Anti-Aging Literature via ConvexTopics and Large Language Models

**arXiv ID:** 2602.20224 | [PDF](https://arxiv.org/pdf/2602.20224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 329. Model Merging in the Essential Subspace

**arXiv ID:** 2602.20208 | [PDF](https://arxiv.org/pdf/2602.20208v1)

**作者:** Longhua Li `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**通讯引用:** 6461 | [OpenAlex ID](https://openalex.org/A5074742406)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个新的模型合并框架ESM，能够在不进行额外训练的情况下将多个任务特定的细调模型融合成多任务模型。

**💡 创新点**

创新点包括：①基于任务激活位移的PCA得到的Essential Subspace Decomposition（ESD），该子空间更贴合任务特征分布；②三层Polarized Scaling机制，通过对高范数参数放大、低范数参数抑制，显著降低跨任务干扰。

**🔧 技术方法**

使用的技术包括PCA、SVD、矩阵正交化、低秩分解、三层参数归一化与放缩、ViT视觉编码器、CLIP预训练权重以及多任务微调。

**📊 数据集**

实验数据集涵盖8、14、20任务视觉分类集合（Cars、DTD、EuroSAT、GTSRB、MNIST、RESISC45、SUN397、SVHN等），使用CLIP的ViT-B/32、ViT-B/16、ViT-L/14三种视觉编码器。

**📈 对比分析**

与Weight Averaging、Task Arithmetic、TIES‑Merging、Consensus TA、TSV‑M、Iso‑CTS等主流合并方法对比，ESM在所有基准上均实现SOTA表现，平均准确率提升约3-6个百分点，接近单任务专家模型的性能。

**⚠️ 局限性**

局限性包括：需要额外的代理样本（32张无标签样本）以及PCA、正交化等计算开销；在任务数量极大或模型规模非常大时，内存和计算成本上升；当前仅在视觉分类任务验证，跨域或非视觉任务的泛化尚待进一步研究。

---

## 330. Agile V: A Compliance-Ready Framework for AI-Augmented Engineering -- From Concept to Audit-Ready Delivery

**arXiv ID:** 2602.20684 | [PDF](https://arxiv.org/pdf/2602.20684v1)

**作者:** Christopher Koch `[一作]` (Agile-V.org), Joshua Andreas Wellbrock `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Agile V 框架，利用 AI 代理在每个任务周期内嵌入独立验证和审计文档生成，实现了从需求到交付的连续 Infinity Loop。

**💡 创新点**

创新点在于将敏捷迭代与 V‑Model 验证结构化为循环流程，强制每个 AI 生成的工件在进入下一环节前必须通过独立测试；同时通过人机审批门和可追溯的决策日志，将合规性嵌入开发流水线，自动产出审计证据。

**🔧 技术方法**

核心技术包括：多模型 AI 代理（Build Agent、Test Designer、Red Team Verifier、Compliance Auditor 等）；上下文工程与持久记忆机制保证模型在大上下文窗口内保持独立性；Human Gate 1/2 的人工审批；以及以文本为介质的插件化技能库。

**📊 数据集**

使用了一个硬件在环（HIL）测试系统项目，约 500 行 Python 代码、8 条正式需求、54 条自动化测试；实验通过 Gemini 1.5 Pro（Cycle 1）和 Claude Opus 4.6（Cycle 2）两套 AI 平台完成。

**📈 对比分析**

对比方法：与基于 COCOMO II 的传统交付估算（≈15.6 k 美元、104 小时）及同一项目下 AI 计算成本进行对比；结果显示单周期成本仅 601–605 美元，成本下降 10–50 倍；测试通过率 100%（8/8 需求、54/54 测试）；人机交互仅 6 次提示，符合 H3。性能在两轮循环内保持一致，证明模型无关性。

**⚠️ 局限性**

局限性包括：仅单一边界项目验证，缺乏大型多团队或不确定需求环境的验证；评估由框架作者完成，可能存在偏见；未涵盖变异测试、分支覆盖或形式化验证；一次性采用成本未计入；计算成本依赖于当前 AI 价格，且对不同硬件/软件栈的适配需要进一步验证。

---

## 331. Misty Forest VR: Turning Real ADHD Attention Patterns into Shared Momentum for Youth Collaboration

**arXiv ID:** 2602.20350 | [PDF](https://arxiv.org/pdf/2602.20350v1)

**作者:** Yibo Meng `[一作]` (Tsinghua University), Yan Guan `[通讯]` (Tsinghua University)

**通讯引用:** 10127 | [OpenAlex ID](https://openalex.org/A5100704992)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一款名为Misty Forest的双人VR协作游戏，专为 ADHD 与非 ADHD 青少年设计，通过将 ADHD 的注意力波动转化为协作节奏，帮助玩家在共同任务中实现互补协同。

**💡 创新点**

创新点在于：①将 ADHD 的注意力波动视为协作资源而非缺陷；②引入非对称角色（爆发者与支撑者）使两类玩家在节奏互换中实现互惠；③在中国文化背景下利用沉浸式游戏机制培养同理心与自我接纳。

**🔧 技术方法**

技术手段包括：Unity3D 开发的 VR 环境、基于物理的可变平台与障碍、角色特定的能量束与支撑触发、实时音频/视觉反馈以及使用 Unity 的输入/网络同步框架。

**📊 数据集**

数据集来源：60名受试者（30 ADHD，30非 ADHD）通过社交平台招募；使用 ADHD 知识问卷、ADHD 自我接纳量表、非 ADHD 同理心量表（SDS）及 GEQ 用户体验量表；行为数据为任务完成率与时间、Burst 与支撑持续时间等。

**📈 对比分析**

比较方法：预后-后测设计，设三组（混合对、同质对、单人）并记录任务完成率、量表得分变化。结果显示：混合对组完成率100%，自我接纳提升显著（p<0.001），非 ADHD 对同理心提升最大；同质对与单人组表现显著低于混合组。

**⚠️ 局限性**

局限性包括：样本仅来自中国文化环境，无法推广至其他文化；任务为结构化游戏，未覆盖日常交互的模糊性；部分受试者报告多玩家合作带来社交压力；缺乏长期跟踪评估协作与自我接纳的持续效果。

---

## 332. No One Size Fits All: QueryBandits for Hallucination Mitigation

**arXiv ID:** 2602.20332 | [PDF](https://arxiv.org/pdf/2602.20332v1)

**作者:** Nicole Cho `[一作]` (JPMorgan AI Research), Manuela Veloso `[通讯]` (JPMorgan AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于上下文多臂赌博机的QueryBandits框架，在线学习针对每条查询的最佳重写策略，以降低闭源大型语言模型的幻觉现象。

**💡 创新点**

创新点在于：①构造了由LLM判定、模糊匹配与BLEU三项指标权衡的复合奖励函数；②利用17维语言特征向量实现上下文感知的重写决策；③仅在输入层做前向重写，无需模型梯度或内部参数修改，兼容闭源模型。

**🔧 技术方法**

核心技术包括：上下文多臂赌博机（Thompson Sampling、LinUCB等），基于语言特征的查询重写（Paraphrase、Simplify、Disambiguate、Expand、Clarify Terms），以及复合奖励函数的设计与校准。

**📊 数据集**

在13个问答基准（共16个场景、约1050个查询）上进行评估，包括TruthfulQA、BoolQA、HotpotQA等。

**📈 对比分析**

与无重写基线、静态重写策略（Paraphrase、Expand等）以及传统非上下文赌博机相比，QueryBandits的Thompson Sampling实现了87.5%的查询级胜率，宏平均准确率提升至0.766（比基线高8.5个百分点），并在大多数场景中超过静态策略。

**⚠️ 局限性**

局限性包括：①依赖预先定义的17维特征，特征选择可能影响效果；②奖励函数虽稳健但仍基于LLM判定，可能引入噪声；③在极端或未见过的查询类型中，学习曲线可能需要更长时间；④目前仅针对问答任务，未验证对生成式任务的通用性。

---

## 333. Are Multimodal Large Language Models Good Annotators for Image Tagging?

**arXiv ID:** 2602.20972 | [PDF](https://arxiv.org/pdf/2602.20972v1)

**作者:** Ming-Kun Xie `[一作]` (RIKEN Center for Advanced Intelligence Project), Masashi Sugiyama `[通讯]` (University of Tokyo)

**通讯引用:** 22208 | [OpenAlex ID](https://openalex.org/A5072744508)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于多模态大语言模型的自动图像标签框架TagLLM，分两阶段生成候选标签并精细校正

**💡 创新点**

两阶段结构化提示（分组多选+二元校验）以及概念对齐去歧义策略，显著缩小了MLLM标签与人工标签的差距

**🔧 技术方法**

多模态大语言模型（如Qwen3-VL）、结构化分组提示、概念对齐CAD、ChatGPT-4o协助校正

**📊 数据集**

COCO 2014、COCO 2017、Objects365三大多标签图像基准

**📈 对比分析**

与BP、MOP以及多种先进方法（CLIP、TagCLIP、RAM++等）对比，TagLLM在注释质量和下游mAP上仅比人工低0.5-1.4%，且降低了性能差距达60-80%

**⚠️ 局限性**

仅在自然图像上评估，未覆盖细粒度或专业领域数据集

---

## 334. On the Optimal Integer-Forcing Precoding: A Geometric Perspective and a Polynomial-Time Algorithm

**arXiv ID:** 2602.20529 | [PDF](https://arxiv.org/pdf/2602.20529v1)

**作者:** Junren Qin `[一作]` (Beihang University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 44447 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于整数倍加（Integer‑Forcing）预编码的联合优化方法，针对重叠MIMO系统的整数矩阵A与功率缩放矩阵D的非凸NP‑hard优化问题，构造了MCN‑SPS（Multi‑Cone Nested Stochastic Pattern Search）多锥嵌套随机模式搜索算法，以实现接近最优的A、D配对；

**💡 创新点**

创新点在于将A、D优化空间几何分解为有限个锥形子区域，并将连续搜索转化为在这些离散子空间内的有向射线搜索；利用收缩映射与Hilbert度量证明D优化的收敛性；在此基础上设计了多锥随机搜索+交替优化框架，复杂度为多项式O(K^4logKlog_2r_0)，显著低于传统PSO或全局搜索方法；

**🔧 技术方法**

核心技术包括：整数格子SIVP求解（采用LLL算法）、矩阵平衡（Sinkhorn迭代）、Perron‑Frobenius理论与Hilbert度量的收缩映射、交替优化（AO）与随机射线搜索；在实现上还结合了MMSE/ML估计误差的鲁棒预编码设计；

**📊 数据集**

实验使用随机Rayleigh信道矩阵（K×N）进行Monte‑Carlo仿真，覆盖不同用户数K、天线数N、信噪比SNR、系统占用率（over‑load）等场景；没有使用公开数据集，而是生成统计仿真数据；

**📈 对比分析**

与PSO、Venturelli（松弛式整数约束方法）、RZF、等功率分配基准进行比较。MCN‑SPS在所有SNR、用户数、占用率场景下均取得更高的总速率，且运行时间约为PSO的一半，复杂度与Venturelli相近但在K较大时更优；

**⚠️ 局限性**

局限性包括：1）仍为近似算法，无法保证全局最优；2）依赖LLL求解SIVP，若SIVP解不佳会影响A的质量；3）理论分析多基于高SNR与矩阵可逆假设；4）在极端估计误差或极高负载下收敛速度和性能可能下降。

---

## 335. Exploring the Impact of Parameter Update Magnitude on Forgetting and Generalization of Continual Learning

**arXiv ID:** 2602.20796 | [PDF](https://arxiv.org/pdf/2602.20796v1)

**作者:** JinLi He `[一作]` (Shanxi University), Xian Yang `[通讯]` (University of Manchester)

**通讯引用:** 1827 | [OpenAlex ID](https://openalex.org/A5060065120)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

从参数更新幅度的视角理论分析了连续学习中的灾难性遗忘问题，统一并比较了冻结训练和初始化训练，并提出基于梯度方向自适应切换的混合更新框架；

**💡 创新点**

创新点在于推导出最优参数更新幅度并证明当任务在参数空间相近时冻结训练更优，同时基于梯度余弦相似度设计自适应混合策略；

**🔧 技术方法**

采用理论推导、凸优化约束、梯度余弦相似度度量、ResNet深度网络和混合参数更新算法；

**📊 数据集**

使用Split CIFAR‑10/100、CUB‑200、Permuted MNIST及其Correlated/Corrupted变体进行实验；

**📈 对比分析**

通过与传统初始化训练和冻结训练在Avg.Acc、Avg.Forgetting、Avg.Cur.Acc等指标上的对比，混合训练平均提升2–7%准确率、降低1–5%遗忘率；

**⚠️ 局限性**

局限在于理论推导基于线性/高斯假设，未完全覆盖非线性深度模型的复杂性，实验规模与任务数量仍有限。

---

## 336. Directly from Alpha to Omega: Controllable End-to-End Vector Floor Plan Generation

**arXiv ID:** 2602.20377 | [PDF](https://arxiv.org/pdf/2602.20377v1)

**作者:** Shidong Wang `[一作]` (University of Zurich), Renato Pajarola `[通讯]` (University of Zurich)

**通讯引用:** 5052 | [OpenAlex ID](https://openalex.org/A5000147776)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一个可控端到端的拓扑与几何增强扩散模型 CE2EPlan，用于从给定边界或无边界直接生成住宅平面图。

**💡 创新点**

彻底去掉多步骤管线和中间表示，一次性训练一次性生成，支持多种交互模式并天然产生多样化结果。

**🔧 技术方法**

采用扩散模型作为生成框架，噪声预测器为 GATransformer，结合多条件遮罩、拓扑增强和几何对齐损失，并配合后处理。

**📊 数据集**

使用公开的 RPLAN 80K 真实住宅平面图数据集进行训练与评估。

**📈 对比分析**

通过 FID、GED、统计指标、用户研究和 ablation 对比，CE2EPlan 在所有指标上优于现有多步骤方法，生成质量更高、结构更合理、并能提供更丰富的多样化结果。

**⚠️ 局限性**

受限于轴向矩形盒子表示，无法自然生成 L 形或斜墙等非曼哈顿布局；后处理仍需使用；在强制输入条件下生成多样性有限。

---

## 337. SD4R: Sparse-to-Dense Learning for 3D Object Detection with 4D Radar

**arXiv ID:** 2602.20653 | [PDF](https://arxiv.org/pdf/2602.20653v1)

**作者:** Xiaokai Bai `[一作]` (Zhejiang University), Hui-Liang Shen `[通讯]` (Zhejiang University)

**通讯引用:** 98462 | [OpenAlex ID](https://openalex.org/A5100773343)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出SD4R框架，将稀疏4D雷达点云转化为稠密表示，并实现3D目标检测

**💡 创新点**

设计前景点生成器(FPG)直接从原始点云生成虚拟前景点，抑制噪声；以及基于类别概率的logit-query编码器(LQE)增强柱状特征

**🔧 技术方法**

基于投票机制的点生成、点云投影为柱子、logit查询编码器、基于BEV的检测头、使用MMDetection3D、AdamW优化器等

**📊 数据集**

View‑of‑Delft (VoD) 4D雷达数据集

**📈 对比分析**

与多种单模4D雷达方法对比，SD4R在整个标注区和驾驶走廊的3D mAP均超过之前所有方法，帧率约22.1 FPS，性能位于雷达+相机融合与单模雷达之间

**⚠️ 局限性**

推理速度相对较慢，缺乏时间序列信息，未利用多帧数据

---

## 338. Circumventing the FLP Impossibility Result with Open Atomic Ethernet

**arXiv ID:** 2602.20444 | [PDF](https://arxiv.org/pdf/2602.20444v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAE DAE LUS), Paul Borrill (DAE DAE LUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出并实现了 Open Atomic Ethernet (OAE)，通过在链路层引入双向注册交换（swap）与 Shannon Slot 机制，实现了在“bisynchronous”模型下的确定性原子协调，绕过 FLP 不可解性。

**💡 创新点**

创新点在于：① 抛弃传统异步以太网的“fire‑and‑forget”语义，采用双向同步交换；② 定义并使用“bisynchronous”概念，使双方在每个 Slot 结束时达到共同知识；③ 通过双向 swap 作为无穷共识数的通用原语，彻底消除 FLP 的异步漏洞。

**🔧 技术方法**

技术核心包括：Shannon Slot（基于硬件计时的同步轮次）、双向寄存器交换（swap）机制、无超时的确定性错误处理、基于八元网格的局部重连与自适应路由。

**📊 数据集**

未使用传统数据集；主要通过理论证明和模拟实验评估模型性能。

**📈 对比分析**

比较方法：理论可达性分析与仿真测量；相较传统以太网，OAE 在失效检测、分区恢复和共识达成时间上至少提升十几倍至几百纳秒级，且无需超时重传机制。

**⚠️ 局限性**

局限性包括：① 需要在硬件层实现双向交换与 Slot 计时，受限于现有 NIC 设计；② 对极端物理损坏（如完整网络切断）仍无法避免分区；③ 实际部署需要大规模硬件改造与标准化工作。

---

## 339. FinAnchor: Aligned Multi-Model Representations for Financial Prediction

**arXiv ID:** 2602.20859 | [PDF](https://arxiv.org/pdf/2602.20859v1)

**作者:** Zirui He `[一作]` (New Jersey Institute of Technology), Mengnan Du `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在多长文本金融预测任务中，提出了FinAnchor框架，通过将不同LLM的嵌入映射到统一的anchor空间，再进行平均聚合并训练轻量级读取器，从而实现多模型特征融合，提升预测性能。

**💡 创新点**

创新点在于：1）无需微调任何基础模型，利用线性对齐（岭回归）将多种LLM嵌入投射到相同坐标系；2）对齐后直接聚合，保持训练和推理成本低；3）提供可解释性分析，揭示对齐如何通过重权重化重要财务证据来减少误报。

**🔧 技术方法**

使用的技术包括：预训练LLM嵌入（Gemma、Qwen、Llama）、岭回归线性映射、特征标准化、平均聚合、轻量级MLP读取器，以及可视化的误差重叠、决策转移和置信度变化分析。

**📊 数据集**

实验数据集包括：Conference Call transcripts、10‑Q filings、FNSPID Nasdaq News、earnings‑call‑based stock movement、FOMC stance classification（共5个任务）。

**📈 对比分析**

与零/少量提示、Longformer、Hierarchical FinBERT及单一LLM基线对比，FinAnchor在所有5个数据集上均实现最高准确率和F1，尤其在Conference Call、10‑Q、FNSPID新闻任务中显著优于单一模型，且在更具挑战性的股价移动和FOMC任务中亦保持领先。

**⚠️ 局限性**

局限性包括：仅在金融文本任务上验证，需在更多语言、任务和模型上检验泛化；对齐与聚合的超参数（如岭正则、聚合方式）未做系统敏感性分析；未对计算成本与工程实现细节进行深入评估。

---

## 340. Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI coordination

**arXiv ID:** 2602.20517 | [PDF](https://arxiv.org/pdf/2602.20517v1)

**作者:** Rakshit Trivedi `[一作]`, David C Parkes `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MIMIC 框架，通过内部语音（inner speech）作为行为中介，实现基于语言的模仿学习与可控行为生成。

**💡 创新点**

创新点在于：① 将 Vygotsky 的内在言语理论引入 IL，构建语义压缩、预测性与时间调节的内部语言模型；② 用视觉语言模型提供外部语言标注来训练 CVAE 生成内部语音；③ 将生成的内部语音作为条件输入到扩散式行为克隆中，实现高多样性与可控性。

**🔧 技术方法**

技术手段包括：
- 条件变分自编码器（CVAE）生成内在语音；
- 视觉语言模型（如 CLIP + GPT‑4o）用于外部语言构建；
- 条件扩散式行为克隆（DDPM‑T）并结合 Transformer 进行状态+语音条件化；
- Transformer 关注机制实现预测性关系提取；
- 预训练 VLM 与 CLIP 嵌入作为语言 scaffold。

**📊 数据集**

使用的数据集：
- D3IL benchmark（Aligning、Sorting、Stacking）— 视觉+非视觉观测；
- Overcooked（三种布局）— 机器人与人类代理协作游戏；
- 通过 GIF 序列生成 VLM 输出的描述作为内部语音标签。

**📈 对比分析**

对比方法：传统 BC（DDPM‑T）及其他 BC 方案；在 D3IL 环境中，MIMIC 在成功率、行为熵（entropy）和 Wasserstein 距离上均显著优于基线；在 Overcooked 与人类代理协作中，MIMIC 的总奖励提升约 20‑30%；表格与图形展示了不同 VLM、embedding、内在语音生成器对性能的影响。

**⚠️ 局限性**

局限性：
- 依赖大规模视觉语言模型和 CLIP 嵌入，模型对不同 VLM 的敏感性较高；
- 内在语音生成的质量受 VLM 描述质量影响；
- 对长时间序列的非马尔可夫性处理仍不完善；
- 缺少真实人类实验验证，仅在仿真环境中评估；
- 需要手动设定内在语音更新窗口，可能限制实时可控性。

---

## 341. Adversarial Robustness on Insertion-Deletion Streams

**arXiv ID:** 2602.20854 | [PDF](https://arxiv.org/pdf/2602.20854v1)

**作者:** Elena Gribelyuk `[一作]` (Princeton University), Samson Zhou `[通讯]` (Texas A&M University)

**通讯引用:** 569 | [OpenAlex ID](https://openalex.org/A5018283928)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一套在插入-删除流（turnstile stream）中实现对频率矩阵第二矩（F₂）、重数估计和满足近似三角不等式的任意对称函数的对抗鲁棒算法，能够在子线性空间内完成全时刻的估计。

**💡 创新点**

创新点在于打破传统线性草图（linear sketch）对抗鲁棒性的Ω(n)空间下界，采用了“估计器–校正器–学习器”（estimator–corrector–learner）框架，并通过递归树结构在非线性方式下引入新的随机性，最终实现了接近最优的子线性空间复杂度。

**🔧 技术方法**

核心技术包括：1）多层递归树结构将整个流划分为多级块；2）在每个块内部使用基于随机矩阵的线性草图；3）估计器输出基于当前迭代向量的估计，校正器检测估计错误并触发学习器；4）学习器通过小步更新逐步逼近真正的频率向量；5）对近似三角不等式的函数做通用化处理；6）利用边界-路径（bounded computation paths）和自适应攻击分析保证鲁棒性；7）对空间进行细致计数，利用对数级别的块划分控制自适应查询次数。

**📊 数据集**

本工作为理论研究，未使用具体数据集，而是通过理论证明和空间复杂度分析来评估算法性能。

**📈 对比分析**

在理论上，F₂时刻估计的空间复杂度为 O((1/ε)·log n)（与非鲁棒 AMS 算法相当），对 L₂ 重数检测的空间为 O((1/ε)·log n)；对于满足近似三角不等式的函数，空间提升到 O(n^{1/C}·S(n))，其中 S(n) 为非鲁棒算法所需空间；相比传统的线性草图对抗鲁棒性方案，该方法实现了显著的空间压缩，突破了先前的线性下界，并在某些情况下达到已知最优下界的多项式级别逼近。

**⚠️ 局限性**

局限性包括：① 需要对流长度与维度 n 的关系做假设（如 m = n 或多项式级别），② 对于特定函数仍需要额外的常数因子和对数因子，③ 对于极端的对抗策略仍存在潜在的空间与误差权衡；④ 目前仅针对理论模型，缺乏在实际大规模数据集上的实验验证。

---

## 342. 823-OLT @ BUET DL Sprint 4.0: Context-Aware Windowing for ASR and Fine-Tuned Speaker Diarization in Bengali Long Form Audio

**arXiv ID:** 2602.21183 | [PDF](https://arxiv.org/pdf/2602.21183v1)

**作者:** Ratnajit Dhar `[一作]` (Chittagong University of Engineering and Technology), Arpita Mallik `[通讯]` (Chittagong University of Engineering and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个针对长篇孟加拉语音频的自动语音识别（ASR）与说话人分离（diarization）框架，利用语音分离、VAD、gap-aware窗口划分以及Whisper-Medium模型完成ASR，采用Fine-tune的pyannote分割模型实现说话人分离。

**💡 创新点**

创新点在于：①引入gap-aware窗口与1秒上下文填充以降低长语音中句子截断；②将Demucs声源分离用于提高噪音环境下的识别质量；③在比赛数据上专门Fine-tune pyannote分割模型以提升说话人边界检测，显著降低DER。

**🔧 技术方法**

使用技术包括：Demucs（声源分离）、Silero VAD、Whisper-Medium（ASR）与WhisperProcessor、pyannote/segmentation-3.0（说话人分割）、ECAPA-TDNN嵌入、Spectral clustering、VBx后处理、混合精度训练（FP16）和Cosine学习率衰减。

**📊 数据集**

数据集为BUET DL Sprint 4.0官方比赛数据集（长篇孟加拉语音频），以及公开的Bangla ASR/diarization基准如FLEURS、Bangla AI等作为对比。

**📈 对比分析**

在官方评测中，ASR私有分数WER为0.3425，公开分数为0.3411，显著优于默认配置；说话人分离DER私有约0.29，公开约0.28，Fine-tuned模型相较于无训练或传统聚类方法降低约30% DER。

**⚠️ 局限性**

局限性包括：对强噪声、重混音、回声、强地区口音的鲁棒性不足；窗口划分仍可能导致轻微上下文不连贯；计算成本高，难以在资源受限或实时场景中部署；Fine-tune仅覆盖官方数据，泛化到其他非竞赛域时表现不确定。

---

## 343. Competition Versus Complexity in Multiple-Selection Prophet Inequalities

**arXiv ID:** 2602.20398 | [PDF](https://arxiv.org/pdf/2602.20398v1)

**作者:** Eugenio Cruz-Ossa `[一作]` (Pontificia Universidad Catolica de Chile), Victor Verdugo `[通讯]` (Pontificia Universidad Catolica de Chile)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了多选择Prophet Inequality中的竞争复杂度，给出了单阈值算法的(1‑ε,k)-竞争复杂度的精确表述，并证明了一个从无竞争到有竞争的相位转变。

**💡 创新点**

①首次给出单阈值算法在任意k≥1时的精确竞争复杂度；②揭示了竞争引入后可突破传统1‑1/√(2πk)界限的急剧相位转变；③提出了闭式解和强对偶证明的无限维线性规划框架。

**🔧 技术方法**

使用量化（quantile）方法将收益与阈值关联；构造并求解无限维线性规划及其对偶；利用对偶性与强对偶证明得到精确表达式；结合概率不等式（Markov、Chernoff）得到竞争复杂度上界。

**📊 数据集**

本研究为理论分析，未使用实际数据集，所有结果均在通用连续分布族ℱ上成立。

**📈 对比分析**

通过与已知的单阈值或最优多阈值算法的竞争比率进行对比。结果表明：k=1时β₁(ε)=ln(1/ε)；k>1时ln(1/ε)/k ≤ β_k(ε) ≤ 1+2ln(1/ε)/(k-1)+√(2ln(1/ε)/(k-1))，并在m≈1.01n时即可达到1-exp(−Θ(k))的收益，显著优于无竞争下的≈1−1/√(2πk)。

**⚠️ 局限性**

①仅限单阈值（静态定价）算法，未涵盖最优多阈值在线算法；②分析依赖i.i.d.假设，无法直接推广到非同分布或随机顺序模型；③求解方法虽闭式但实现仍需解析推导，数值实现相对复杂。

---

## 344. Hybrid LLM-Embedded Dialogue Agents for Learner Reflection: Designing Responsive and Theory-Driven Interactions

**arXiv ID:** 2602.20486 | [PDF](https://arxiv.org/pdf/2602.20486v1)

**作者:** Paras Sharma `[一作]` (University of Pittsburgh), Erin Walker `[通讯]` (University of Pittsburgh)

**通讯引用:** 1686 | [OpenAlex ID](https://openalex.org/A5023610771)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在文化响应型机器人夏令营中，研究者开发并评估了一个混合LLM嵌入式对话系统，该系统通过基于规则的有限状态机与LLM生成相结合，支持中学生在机器人设计活动中的自我调节学习反思。

**💡 创新点**

创新点在于提出双阶段LLM集成策略（相关性检查 + 上下文生成），将LLM嵌入结构化对话流程中，既保持了对自我调节学习理论的严格对齐，又利用LLM的生成灵活性；同时系统提供了关于LLM在教育对话中的可解释性、情感与上下文适配的实证洞见。

**🔧 技术方法**

技术包括：基于规则的有限状态机、LLaMa‑3.1‑8B‑Instruct（少量示例提示的二阶段推理）、WebSpeechAPI（STT/TTS）、前后端 WebSocket 通信，以及在实验中讨论但未实施的检索增强生成（RAG）与微调技术。

**📊 数据集**

使用的数据集为九名中学生在夏令营期间产生的对话记录（共 357 轮，平均 39.66 轮/会话）及其访谈转录，全部为现场收集，未采用公开标准数据集。

**📈 对比分析**

评估方法为定性编码（LLM 触发、生成质量、情感、参与度）与定量指标（词数、回合数、LLM 触发准确率等）。LLM 触发准确率约 64%（23/36），生成阶段中 9/24 促成了进一步反思，平均反思词数提升 1.75 倍；整体性能显示LLM 能增强反思深度，但仍受情境与情感失配影响。

**⚠️ 局限性**

局限性包括样本规模小（仅 9 名学生）、对话短文本、未在实时学习任务中嵌入、LLM 对机器人设计领域知识缺乏掌握、情绪识别不足、提示过多导致枯燥，以及缺乏个性化与适配机制；技术层面需结合 RAG/微调提升上下文对齐与可解释性。

---

## 345. Examining and Addressing Barriers to Diversity in LLM-Generated Ideas

**arXiv ID:** 2602.20408 | [PDF](https://arxiv.org/pdf/2602.20408v1)

**作者:** Yuting Deng `[一作]` (Columbia Business School), Olivier Toubia `[通讯]` (Columbia Business School)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在创意生成中的多样性缺失，并通过理论框架和四项实验证明了“固定化”和“知识聚合”机制导致LLM多样性低下，随后提出并验证了persona与Chain‑of‑Thought（CoT）两种干预方法，最终实现LLM创意多样性超过人类水平。

**💡 创新点**

创新点在于首次将认知心理学的固定化与知识分区理论引入LLM多样性研究，提出普通persona替代传统创意企业家更能有效分区，证明CoT能专门缓解LLM固定化，两者结合可将多样性推至人类上限甚至超越。

**🔧 技术方法**

采用GPT‑4o生成创意，设计了persona与CoT提示模板，使用层次抽象的内容分类方法对创意进行三维标签化，并通过嵌入距离、t‑SNE可视化等技术评估多样性。

**📊 数据集**

使用Tencent Personas数据集（约20万合成身份）作为persona来源，结合Prolific平台收集的99名人类参与者的健身产品创意作为基准，实验数据主要来自GPT‑4o生成。

**📈 对比分析**

通过对比人类、默认LLM、seeded LLM、普通persona LLM、创意企业家 LLM、CoT LLM和persona+CoT LLM的累积类别数、唯一组合数和平均距离三指标，采用bootstrap统计检验差异；结果显示persona+CoT LLM在唯一组合上达到248，超过人类的197，整体多样性提升约26%，CoT单独提升约20%，普通persona提升约15%。

**⚠️ 局限性**

研究仅聚焦多样性，未评估质量、可行性或创新价值；persona设计基于合成数据，可能缺乏真实人类多样性；实验仅使用GPT‑4o，尚未验证对其他模型的普适性；Embedding分析仅间接反映知识分区，缺乏对内部表示的直接检查。

---

## 346. FedAvg-Based CTMC Hazard Model for Federated Bridge Deterioration Assessment

**arXiv ID:** 2602.20194 | [PDF](https://arxiv.org/pdf/2602.20194v1)

**作者:** Takato Yasuno `[一作]` `[通讯]` (Yachiyo Engineering Co.), Takato Yasuno (Yachiyo Engineering Co.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种联邦学习框架，使用Federated Averaging对桥梁降解的连续时间马尔可夫链（CTMC）危害模型进行无共享原始检验数据的分布式参数估计。

**💡 创新点**

将CTMC危害模型的对数似然梯度与FedAvg结合，提供了局部梯度上传的低通信成本方案，并通过合成数据验证了在异构用户环境下的收敛与可扩展性。

**🔧 技术方法**

使用PyTorch自动微分实现CTMC对数似然梯度，FedAvg算法（带样本加权、动量、梯度裁剪）实现联邦训练，合成数据生成器模拟海岸、河岸、内陆三种地区的异质性。

**📊 数据集**

完全合成的桥梁检验记录数据，基于预设的真值参数矩阵，并加入地区噪声以模拟真实异构。

**📈 对比分析**

在500、2000、4000用户规模下，使用50轮训练，平均负对数似然收敛至0.76–0.80，梯度范数随用户数下降，通信成本仅25KB/轮，表明模型在异构环境中稳健且可扩展。

**⚠️ 局限性**

仅在合成数据上验证，未考虑桥梁成员相关性、间歇性检验、误差评估等实际问题，且局部估计与全局平均可能导致偏差。

---

## 347. Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking

**arXiv ID:** 2602.20330 | [PDF](https://arxiv.org/pdf/2602.20330v1)

**作者:** Jingcheng Yang `[一作]` (University of Illinois Urbana-Champaign), Mingyuan Wu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 85 | [OpenAlex ID](https://openalex.org/A5107246646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了 Vision–Language Models（VLM）的电路追踪框架，结合 transcoders、归因图、注意力可视化等技术系统分析多模态推理机制并实现可操控的电路补丁；

**💡 创新点**

首次将稀疏自编码器（transcoders）和归因图迁移至 VLM，实现可解释的多模态电路追踪，并通过干预验证电路的因果性与可控制性；

**🔧 技术方法**

利用 Transcoder（稀疏自编码器）、归因图、注意力可视化、特征激活分析、激活补丁/干预等技术；

**📊 数据集**

多模态训练集包括文本数据、ImageNet、Cauldron 等，实验以 Gemma‑3‑4B‑it 为基准，使用约 28k 张图片等；

**📈 对比分析**

与仅文本训练的 Transcoder 对比，Fraction of Variance Unexplained (FVU) 明显下降，实验通过电路补丁与激活干预验证因果性，提升了解释性与可控性，未报告传统性能指标但表明对模型行为有显著解释力；

**⚠️ 局限性**

注意力可视化难以解释、跨层特征叠加未捕获、计算成本高、需人工标注电路、仅在 Gemma‑3 上验证，缺乏跨模型验证

---

## 348. TrajGPT-R: Generating Urban Mobility Trajectory with Reinforcement Learning-Enhanced Generative Pre-trained Transformer

**arXiv ID:** 2602.20643 | [PDF](https://arxiv.org/pdf/2602.20643v1)

**作者:** Jiawei Wang `[一作]` (University of Tokyo), Renhe Jiang `[通讯]` (University of Tokyo)

**通讯引用:** 2246 | [OpenAlex ID](https://openalex.org/A5040449880)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于Transformer的城市移动轨迹生成框架（TrajGPT-R），通过离线强化学习和逆强化学习构建轨迹级奖励模型，并在此基础上进行奖励模型驱动的微调；

**💡 创新点**

创新点在于：①将轨迹生成视为离线RL问题并通过词表压缩降低模型复杂度；②使用逆强化学习学习轨迹级奖励，捕获个体偏好；③引入奖励模型微调（RMFT），解决传统RL自回归方法中的长期信用分配和稀疏奖励问题；④结合可解释性分析展示模型决策机制；

**🔧 技术方法**

技术包括Transformer（GPT式自回归生成）、离线强化学习、逆强化学习、奖励模型微调（RMFT）与GAE、监督微调加权、注意力分析、t‑SNE嵌入可视化；

**📊 数据集**

使用三大公开数据集：Toyota（东京车辆GPS轨迹）、T‑Drive（北京出租车轨迹）和Porto（葡萄牙波尔图出租车轨迹）；

**📈 对比分析**

与Markov、TrajVAE、TrajGAIL、D3PM、IQL、TrajGPT（预训练）和TrajGPT‑DPO（无奖励微调）等基线在Jaccard、Cosine、BLEU、L‑JSD、C‑JSD、UE、BE七项指标上对比；TrajGPT‑R在可靠性（Jaccard、Cosine、BLEU）与多样性（JSD低、UE/BE高）上均优于所有基线，证明了方法的有效性；

**⚠️ 局限性**

局限性包括：①动作词表压缩限制，难以处理高度交叉的路口；②仅针对车辆轨迹，缺乏多模态（步行、自行车等）支持；③自回归生成易出现累积误差，长轨迹生成受限；④个体嵌入在推理阶段利用不足；⑤奖励模型完全基于数据，可能带来偏差。

---

## 349. DeCo: A Core Calculus for Incremental Functional Programming with Generic Data Types

**arXiv ID:** 2602.20866 | [PDF](https://arxiv.org/pdf/2602.20866v1)

**作者:** Timon Böhler `[一作]` (Technical University of Darmstadt), Mira Mezini `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 8135 | [OpenAlex ID](https://openalex.org/A5078067853)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了名为 DeCo 的核心微积分，用于支持通用数据类型的增量式函数式编程，并在 Lean 里实现了完整的可执行框架。

**💡 创新点**

创新点在于通过容器与变化结构统一建模数据，提供细粒度的静态增量化，并设计了一组组合子（Self、Triv、Lin、BiLin、Add 等）自动生成正确的增量实现，兼顾域特定与通用操作。

**🔧 技术方法**

主要技术包括 typed first‑order 函数式语言、缓存增量化、容器化抽象、增量组合子、Lean 证明、哈希映射实现以及 tagless‑final 交互。

**📊 数据集**

案例实验涵盖线性代数（矩阵、向量）、关系代数（表、笛卡尔积、连接）、字典聚合、树结构以及 CRDT；使用 dense 神经网络层、关系查询和树求和作为基准。

**📈 对比分析**

通过在相同输入下多次测量（取最佳），将增量执行与完整重新评估对比，结果表明当输入变动 ≤70% 时增量化明显快于重新评估；在 dense 层、关系查询和树求和实验中均实现了从 O(n²) 到 O(n) 的加速。

**⚠️ 局限性**

限制包括：仅支持静态形状容器，动态增长需手工处理；实现基于哈希表且未针对大规模线性代数做性能优化；只能处理第一阶函数；组合子需手工满足前置条件；某些域特定操作仍需手动实现。

---

## 350. Rethink Efficiency Side of Neural Combinatorial Solver: An Offline and Self-Play Paradigm

**arXiv ID:** 2602.20730 | [PDF](https://arxiv.org/pdf/2602.20730v1)

**作者:** Zhenxing Xu `[一作]`, Ji Wang `[通讯]` (National Key Laboratory of Big Data and Decision, National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 ECO（Efficient Offline self-play for Neural Combinatorial Optimization）框架，利用离线自我对弈和 Mamba 线性架构实现高效的 NCO 学习。

**💡 创新点**

创新点包括：①将学习范式从在线 RL 转变为两阶段离线范式（监督预热 + 迭代 DPO）；②设计基于 Mamba 的 Encoder-Decoder 结构，显著降低 O(N²) 的自注意力内存与计算复杂度；③采用启发式 Bootstrapping 和局部搜索增强的对比学习，使训练更稳定、梯度更强。

**🔧 技术方法**

技术手段：Mamba 结构（S6 状态空间模型）、Direct Preference Optimization（DPO）、监督预热（SFT）、局部搜索（2‑opt/3‑opt）用于构造对比样本、GPU 并行扫描与线性递归实现。

**📊 数据集**

使用标准生成的 TSP 与 CVRP 数据集（N=200/500/1000/5000 节点），并用 LKH‑3/Concorde/HGS 产生高质量轨迹作为训练与评估基准。

**📈 对比分析**

与传统精确求解器（Concorde、LKH‑3、HGS）以及主流 NCO 方法（AM、POMO、CNF、GFlowNet‑HBG）对比。ECO 在大规模实例（如 TSP‑5000）保持竞争性的解质量（约 2–5% 最优性缺口），同时在显存占用、训练吞吐率和推理时间上优于 Transformer 基线（显存线性 vs 二次，推理时间从数小时降至数分钟）。

**⚠️ 局限性**

局限性：①对极大规模实例仍存在一定的最优性缺口；②需要离线数据生成与局部搜索的额外计算开销；③在超大规模图（N>10k）时 Mamba 仍可能遇到内存瓶颈；④方法在不同 CO 问题（非 TSP/CVRP）上的泛化尚未充分验证。

---

## 351. Elimination-compensation pruning for fully-connected neural networks

**arXiv ID:** 2602.20467 | [PDF](https://arxiv.org/pdf/2602.20467v1)

**作者:** Enrico Ballini `[一作]` (Politecnico di Milano), Francesco Regazzoni `[通讯]` (Politecnico di Milano)

**通讯引用:** 4618 | [OpenAlex ID](https://openalex.org/A5049456178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于权重去除同时对相邻偏置进行补偿的全连接神经网络剪枝方法。

**💡 创新点**

创新点在于将偏置补偿融入权重重要性评估，通过泰勒展开求解最优补偿，既能量化权重对输出的影响，又能降低计算成本。

**🔧 技术方法**

使用了自动微分、泰勒线性逼近、梯度与偏置耦合的误差度量，以及对输出差异的期望计算；实验中采用了训练‑剪枝‑微调的循环流程。

**📊 数据集**

实验数据集包括经典 MNIST 手写数字分类数据集，以及基于 1D 扩散‑吸附偏微分方程求解的合成数据集（并加入不同幅度噪声）。

**📈 对比分析**

与非线性（逐权重评估）、幅值、梯度幅值、随机剪枝以及通过减小层宽度实现的“全连接”方法相比，实验显示该方法在 50%–90% 剪枝率下保持更低的测试损失，尤其在 PDE 噪声场景下表现更稳健。

**⚠️ 局限性**

局限性包括仅针对全连接网络验证，未测试卷积或 Transformer 等结构；补偿仅考虑相邻偏置，可能忽略跨层相关性；需要手动设定阈值或保留比例；并未给出完整的时间复杂度或硬件加速分析。

---

## 352. Balancing Multiple Objectives in Urban Traffic Control with Reinforcement Learning from AI Feedback

**arXiv ID:** 2602.20728 | [PDF](https://arxiv.org/pdf/2602.20728v1)

**作者:** Chenyang Zhao `[一作]` (Trinity College Dublin), Ivana Dusparic `[通讯]` (Trinity College Dublin)

**通讯引用:** 1820 | [OpenAlex ID](https://openalex.org/A5059738292)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

探索了 RLAIF 在多目标自适应系统（交通信号控制）中的应用，利用 LLM 自动生成偏好标签训练奖励模型和策略。

**💡 创新点**

设计了专门针对多目标的 LLM 注释工作流程，支持跨目标比较并根据用户规格生成条件标签，展示通过提示即可调节策略行为。

**🔧 技术方法**

结合 Preference‑based RL、RLAIF、LLM（gpt‑4.1‑nano）偏好生成、Bradley‑Terry 奖励模型与 DQN，使用规则模板将环境观测转化为自然语言。

**📊 数据集**

在 SUMO 模拟器构建的四向交叉口上进行实验，使用 NS 200 vph、EW 600 vph 的交通流量，生成 10,000 秒/episode 的仿真数据。

**📈 对比分析**

与单目标 DQN、线性加权奖励基线对比；RLAIF 在 20k 次偏好标注后收敛至接近线性基线的吞吐量，并在不同提示下实现预期优先级，样本效率略低但不需手工奖励工程。

**⚠️ 局限性**

训练初期样本效率低、偏好标注成本高、LLM 产生的标签不确定性导致约 44% 被过滤，对提示的依赖性及多目标冲突下标签一致性仍是挑战。

---

## 353. Online Algorithms with Unreliable Guidance

**arXiv ID:** 2602.20706 | [PDF](https://arxiv.org/pdf/2602.20706v1)

**作者:** Julien Dallot `[一作]` (TU Berlin), Stefan Schmid `[通讯]` (TU Berlin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在线算法的无可靠指导（OAG）模型，并给出一个黑盒编译器（DTB）把任意无预测的在线算法转换为学习增强版，随后在三种经典问题（边匹配、缓存、均匀度量任务系统）上实现并分析。

**💡 创新点**

核心创新是：①将预测与算法分离，构建不依赖特定预测语义和损失函数的通用框架；②提出极其简单的DTB编译器，只需一次硬币抛掷决定是否信任指导；③在OAG框架下实现了最优一致性-鲁棒性折衷（缓存、均匀MTS）并在随机到达顺序的边匹配上超过现有最优。

**🔧 技术方法**

主要技术手段包括：基于请求-答案游戏的正式定义、β‑biased 随机指导模型、DTB编译器的随机信任策略、对齐分析（竞争比的上界与下界）、概率与期望计数、Wald等式、递推/归纳证明。

**📊 数据集**

本工作完全是理论分析，没有使用任何真实数据集；所有结果均基于抽象问题实例和概率模型。

**📈 对比分析**

通过与最优离线算法的竞争比对比，展示了：缓存和均匀MTS的OAG算法在一致性（β=0）时达到常数级别，在鲁棒性（β=1）时仅需对数级别；边匹配的OAG算法在β=1时竞争比为1/2，β=0时为1−e⁻¹，且两者之间呈光滑过渡；在随机到达顺序下的匹配问题上，该算法突破了先前仅在受限模型下的结果。

**⚠️ 局限性**

限制包括：①对β和τ的参数选择需要先验或经验调优；②DTB编译器虽然简单但对每个问题的性能上限取决于原始无预测算法的结构；③本文仅给出理论证明，缺乏实验验证；④在更复杂的预测语义（如带噪声的预测）下的鲁棒性仍需进一步研究。

---

## 354. SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models

**arXiv ID:** 2602.20901 | [PDF](https://arxiv.org/pdf/2602.20901v1)

**作者:** Yuechen Xie `[一作]` (Zhejiang University), Jie Song `[通讯]` (Zhejiang University)

**通讯引用:** 3669 | [OpenAlex ID](https://openalex.org/A5047371218)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了SpatiaLQA基准，系统评估了41种视觉语言模型在空间逻辑推理任务中的表现，并提出递归场景图辅助推理方法来提升模型性能。

**💡 创新点**

创新点在于首次提出空间逻辑推理这一能力概念，构建大规模开放词汇、可多步推理的室内场景问答数据集，利用GPT‑4o实现自动化评估，并通过递归生成任务相关场景图显著提升VLM的多步推理能力。

**🔧 技术方法**

使用了Depth Anything V2、SAM等视觉基础模型获取深度与分割信息，结合GPT‑4o进行步骤匹配与评分，采用匈牙利算法进行一步一一匹配，递归场景图构造以及链式思维（CoT）等技术。

**📊 数据集**

数据集为9,605个图像-文本问答对，来自241个室内场景，涵盖13类场景；通过手工标注、子图提取和图扩展三阶段生成。

**📈 对比分析**

通过回召率、精确率和F1度量对41个VLM进行对比，发现最优模型GPT‑5在内容上F1≈76.0、前提上F1≈39.2，仍低于人工基准；递归场景图方法在GPT‑4o上将内容F1从67.4提升至69.8、前提F1从25.1提升至28.1，表现最优。

**⚠️ 局限性**

局限性包括：多步推理仍显不足，前提预测准确率低；评估高度依赖GPT‑4o的匹配能力；数据集仅覆盖室内场景，可能难以泛化到更复杂或户外环境；递归深度与阈值需要手工设定，存在参数敏感性。

---

## 355. Scaling State-Space Models on Multiple GPUs with Tensor Parallelism

**arXiv ID:** 2602.21144 | [PDF](https://arxiv.org/pdf/2602.21144v1)

**作者:** Anurag Dutt `[一作]` (Stony Brook University), Anshul Gandhi `[通讯]` (Stony Brook University)

**通讯引用:** 3114 | [OpenAlex ID](https://openalex.org/A5024346889)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向选择性状态空间模型（SSM）的大语言模型推理的张量并行（TP）实现，并在多 GPU 上实现了低延迟推理。

**💡 创新点**

创新点包括：①在 TP 中引入 SSM 缓存实现一次性前填充；②基于通道的智能拆分，保持 SSM 混合器的局部性并减少通信；③拆分打包参数以避免在 TP 中的重组；④在 AllReduce 上使用 FP16 量化进一步降低通信开销。

**🔧 技术方法**

使用技术包括：GPU 上的张量并行拆分、通道切分器、状态缓存机制、FP16 量化 AllReduce、对 SSM 混合器的自定义内核优化。

**📊 数据集**

实验数据集为 Simple English Wikipedia（SimpleWiki），并在 NVIDIA A6000（PCIe）和 A100（NVLink）集群上测试。

**📈 对比分析**

与单 GPU（1×）和数据并行（DP）基线对比，TP 推理在 2/4 GPU 上分别提升 1.4–3.9×吞吐量，量化 AllReduce 进一步提升 10–18%，同时支持 2–4 倍更长的输入/输出上下文。

**⚠️ 局限性**

局限性包括：1）通信瓶颈导致在高 GPU 数量或更慢互连时收益趋于饱和；2）量化可能引入 2–5% 的准确率下降；3）仅在有限的 SSM 模型（Mamba、Mamba‑2、Falcon‑Mamba、Zamba）和特定 GPU 体系结构上验证，泛化性待进一步研究。

---

## 356. UrbanFM: Scaling Urban Spatio-Temporal Foundation Models

**arXiv ID:** 2602.20677 | [PDF](https://arxiv.org/pdf/2602.20677v1)

**作者:** Wei Chen `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5733 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向城市时空数据的基础模型，并提出数据、计算、架构三维扩展框架；

**💡 创新点**

① 超大规模多域时空语料库与统一标准化管道；② 基于KD‑树的自适应分块token化，解决不同城市/传感器数差异；③ 极简Transformer，分解时空注意力并结合RoPE、RevIN，实现零-shot迁移与生成式预训练；

**🔧 技术方法**

统一数据清洗/预补全、KD‑树聚类token化、因子化时空注意力、RoPE位置编码、RevIN归一化、生成式预测损失、Flash/Linear Attention等技术；

**📊 数据集**

包含1 B+样本，覆盖100座城市、8个领域（交通速度/流量、拥堵、出租车、共享单车、手机信号等）的大型时空语料库；构成最大时空基准集 ST‑Foundation benchmark，涵盖4国7市、10年以上时序；

**📈 对比分析**

与22个基线（专家模型、图网络、时间序列基础模型等）在零/少/全射任务中对比；零射下MAPE降低39‑70%，甚至超过专家模型；few‑shot fine‑tuning提升28‑65%；在大基准中比现有基础模型误差低10‑30%，推理速度比Chronos等快4‑10倍；

**⚠️ 局限性**

受数据覆盖范围与质量限制，难以处理极稀疏或极端异常城市；未加入多模态信息或事件因子，缺乏可解释性；模型规模仍较大，压缩与加速仍是挑战。

---

## 357. Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion

**arXiv ID:** 2602.20577 | [PDF](https://arxiv.org/pdf/2602.20577v1)

**作者:** Jiaru Zhang `[一作]` (Purdue University), Ziran Wang `[通讯]` (Purdue University)

**通讯引用:** 4682 | [OpenAlex ID](https://openalex.org/A5038550389)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Mask Vision‑Language‑Action Diffusion（Masked VLA Diffusion）框架，将连续驾驶轨迹离散化为动作词表，并通过几何一致的嵌入学习与自注意力扩散模型联合生成可解释的驾驶计划与自然语言说明；

**💡 创新点**

创新点在于①离散动作词表化，将轨迹压缩为可分类的动力学动作；②几何感知嵌入学习，使嵌入空间与物理距离保持一致；③动作优先解码策略，先快速生成轨迹再生成解释，显著降低推理延迟；

**🔧 技术方法**

使用VLM预训练视觉编码器、LLM文本编码器、Transformer‑based masked diffusion生成器、LoRA参数微调、温度软分配与几何一致性损失、两阶段（动作先导、联合VLA）训练以及动作优先解码；

**📊 数据集**

主要数据集为nuScenes（规划）、Nu‑X（解释生成）和nuScenes‑QA（问答），并在公开基准上与多种LLM/VLM与Diffusion模型进行对比；

**📈 对比分析**

相较于自回归VLM（如LLaVA、Llama‑3.2、Qwen2‑VL）、Diffusion基线ViLaD及UniAD，Masked VLA Diffusion在nuScenes规划上平均L2误差降至1.28（比ViLaD低0.53），失效率0%，推理时延1.72ms；在Nu‑X解释上BLEU‑4 13.0、METEOR 36.8，QA准确率55.7%；

**⚠️ 局限性**

局限性包括：①离散词表大小需权衡精度与学习难度，过大或过小均影响性能；②几何嵌入学习依赖额外训练阶段与温度调参；③模型仍基于预训练VLM/LLM，迁移到不同域或更复杂场景可能受限；

---

## 358. Probing Graph Neural Network Activation Patterns Through Graph Topology

**arXiv ID:** 2602.21092 | [PDF](https://arxiv.org/pdf/2602.21092v1)

**作者:** Floriano Tori `[一作]` (Vrije Universiteit Brussels), Vincent Ginis `[通讯]` (Vrije Universiteit Brussels)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过分析图变压器中的大激活（Massive Activations, MAs）与图的平衡福兰曲率（BFc）的对应关系，探究全局注意机制如何与图拓扑交互，并在合成图、分子图和长距离图基准（LRGB）上进行实验。

**💡 创新点**

①发现“曲率坍塌”现象——全局注意机制反而加剧负曲率瓶颈；②将 MAs 作为诊断工具揭示模型对负曲率结构的依赖；③通过因果剪枝验证模型在长距离任务中确实依赖这些高负曲率的路由。

**🔧 技术方法**

使用 Balanced Forman Curvature 计算曲率；利用 Graph Transformer 的全局注意权重提取 MAs；进行因果剪枝实验；在实验中对注意权重进行曲率加权邻接矩阵重构。

**📊 数据集**

合成 Barbell 图、药物分子数据集（PCQM4M、MUTAG 等）以及长距离图基准 LRGB（Peptides-Func、Peptides-Struct）。

**📈 对比分析**

与传统基于重连的 GNN、局部 MPNN 及其他 Transformer 变体对比；对曲率分布和 MA 分布进行可视化与统计；结果显示 GT 在 LRGB 上表现不佳，且在 MAs 依赖性上表现出显著差异。

**⚠️ 局限性**

仅针对 softmax 全局注意的变体；缺乏通用的曲率正则化方案；实验范围受限于所选数据集，结果可能对不同图结构或注意机制不完全适用。

---

## 359. PVminer: A Domain-Specific Tool to Detect the Patient Voice in Patient Generated Data

**arXiv ID:** 2602.21165 | [PDF](https://arxiv.org/pdf/2602.21165v1)

**作者:** Samah Fodeh `[一作]` (Yale School of Medicine), Aimee Roundtree `[通讯]` (Texas State University)

**通讯引用:** 420 | [OpenAlex ID](https://openalex.org/A5050774052)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并评估了一套名为PVminer的端到端NLP框架，用于从患者生成的安全信息（secure messages）中自动提取并结构化患者声音（Patient Voice, PV）标签（包括Code、Subcode和Combo级别的多标签分类）。

**💡 创新点**

创新点：① 统一了沟通行为与社会决定因素（SDoH）两大维度的多标签检测；② 在大规模患者文本上进行领域自适应预训练，生成PV-BERT-base/large；③ 通过BERTopic主题模型提供主题关键词，实现主题增广；④ 在细粒度层面（Subcode）和复合层面（Combo）上实现高精度检测。

**🔧 技术方法**

技术：基于BERT的自监督预训练（Masked Language Modeling），BERTopic（Transformer embeddings + UMAP + HDBSCAN + CT‑IDF）主题建模，微调阶段使用多标签二进制交叉熵（Sigmoid +阈值0.5），加入作者身份标记和主题关键词；实验采用Hugging Face Transformers、scikit‑learn等工具。

**📊 数据集**

数据集：多机构安全信息与调查问卷的患者生成文本共1,137条，约46k词；包含6.76M句子用于预训练，500k句子用于主题模型；标签为两层层级（Code/ Subcode）共多类别，存在极端不平衡。

**📈 对比分析**

比较方法：与BERT-base/large、BioBERT、ClinicalBERT、SapBERT、SciBERT、TwHIN-BERT等通用/医学/科学预训练模型在同一PVminer框架内微调；评估指标为micro F1。结果显示，PVminer结合域自适应预训练与主题增广后，在Code层面F1≈82.25%、Subcode层面F1≈80.14%、Combo层面F1≈77.87%，均显著优于所有基线模型。

**⚠️ 局限性**

局限性：① 训练数据规模有限且类别严重不平衡；② 仅处理单条消息，未利用多轮对话上下文；③ 对低频、隐式表达（如共享决策、邻里环境等）检测仍差；④ 需进一步研究数据增强、层级/标签感知模型与对话级建模以提升鲁棒性。

---

## 360. XMorph: Explainable Brain Tumor Analysis Via LLM-Assisted Hybrid Deep Intelligence

**arXiv ID:** 2602.21178 | [PDF](https://arxiv.org/pdf/2602.21178v1)

**作者:** Sepehr Salem Ghahfarokhi `[一作]` (Georgia State University), Mohammed Alser `[通讯]` (Georgia State University)

**通讯引用:** 2361 | [OpenAlex ID](https://openalex.org/A5081055664)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 XMorph 框架，实现对胶质瘤、脑膜瘤和垂体瘤的可解释性细粒度分类

**💡 创新点**

创新点包括信息加权边界归一化（IWBN）提升肿瘤边界特征、混合深度学习与非线性动力学特征、双通道可解释模块（GradCAM+++LLM）

**🔧 技术方法**

采用 DeepLabV3+ResNet‑50 进行分割与特征提取，结合非线性指标（Fractal Dimension、Approximate Entropy、Largest Lyapunov Exponent）、临床生物标志（REI、MLS、Skull‑to‑Tumor Distance）及 XGBoost 分类器，LLM 生成文本解释

**📊 数据集**

使用公开的多中心脑 MRI 数据集（如 Figshare 2024 集合），共 3,564 张扫描，涵盖胶质瘤 1,426 张、脑膜瘤 708 张、垂体瘤 930 张及 500 张正常脑扫描

**📈 对比分析**

与单一特征组、仅深度特征和仅 IWBN 等方法对比，混合特征模型实现 96.0% 准确率，Dice 0.932，显著优于现有 SOTA 方法

**⚠️ 局限性**

局限性包括仅使用单模态 T1c MRI，未对多序列或跨中心泛化做充分验证，LLM 生成解释可能出现幻觉或缺乏临床验证

---

## 361. Is the Trigger Essential? A Feature-Based Triggerless Backdoor Attack in Vertical Federated Learning

**arXiv ID:** 2602.20593 | [PDF](https://arxiv.org/pdf/2602.20593v1)

**作者:** Yige Liu `[一作]` (Peking University), Hanpin Wang `[通讯]` (Peking University)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5079106687)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在垂直联邦学习（VFL）场景下提出了一种不需要触发器的后门攻击方法，利用攻击者在训练阶段记录的嵌入信息，在推理阶段直接替换嵌入实现攻击

**💡 创新点**

发现VFL后门攻击中触发器并非必要，提出基于标签特征的触发器无关攻击路径，并通过标签推断、毒化生成和推理执行三模块实现

**🔧 技术方法**

使用嵌入聚类与标签推断、放大因子与高斯扰动的毒化生成、以及在推理阶段对嵌入替换的机制；同时评估了多种防御策略

**📊 数据集**

在五个基准数据集上评估：MNIST、Fashion‑MNIST、CIFAR‑10、CINIC‑10 与 Criteo（表格/图）

**📈 对比分析**

与三种传统触发器后门攻击（VILLAIN、BadVFL、BASL）对比，攻击成功率提升2~50倍，主任务精度影响极小；在多方VFL、不同防御（梯度裁剪、压缩、DP、Marvell、CAE、ANP、ABL、检测等）下仍保持高攻击成功率，单独推理阶段防御仅在简单数据上有效

**⚠️ 局限性**

受限于标签推断准确性、底层模型的标签表达能力、以及恶意嵌入被多方正态嵌入稀释的风险，且在复杂数据上防御检测不足，需进一步研究嵌入特征对攻击的影响与推理阶段防御策略

---

## 362. Does Order Matter : Connecting The Law of Robustness to Robust Generalization

**arXiv ID:** 2602.20971 | [PDF](https://arxiv.org/pdf/2602.20971v1)

**作者:** Himadri Mandal `[一作]` (Indian Statistical Institute), Debayan Gupta `[通讯]` (Ashoka University)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5073351462)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过理论证明与实证实验相结合，阐明了过参数化模型的鲁棒性与鲁棒泛化误差之间的联系，并给出了鲁棒泛化误差的下界与Rademacher复杂度的关系；随后在MNIST数据集上，系统地测量并分析了模型容量与数据集大小对经验 Lipschitz 下界的影响。

**💡 创新点**

创新点在于：①首次给出了鲁棒泛化误差与 Lipschitz 常数之间的严格下界，证明了鲁棒性与过参数化的本质关联；②提出了一种基于 Rademacher 复杂度的鲁棒损失分解方法，恢复并推广了 Wu 等人的 Ω(n^{1/d}) 下界；③在实验层面系统对比了 Bubeck 与 Wu 两种理论预测的 Lipschitz 量化尺度，发现实际表现更符合 Wu 风格的 n^{1/d} 规律。

**🔧 技术方法**

主要技术包括：Lipschitz 函数理论、鲁棒损失的上界与下界推导、Rademacher 复杂度的期望下界、对损失向量的收缩不等式、经验 Lipschitz 下界的配对差分估计以及对实验结果的多元线性回归分析。

**📊 数据集**

使用的是 MNIST 手写数字图像数据集，采用固定的卷积特征提取层，唯独改变全连接层宽度来调节模型容量，数据集大小在 1000–10000 样本间变动。

**📈 对比分析**

比较方法：将实验得到的经验 Lipschitz 下界与两种理论预测（Bubeck 风格 L∝√(nd/p) 与 Wu 风格 L∝n^{1/d}）进行对比，并通过多元线性回归估计 α、β 指数；实验结果表明，n 的影响指数约为 0.16，p 的影响指数约为 0.03，明显偏向 Wu 风格的 n^{1/d}（0.1）而非 Bubeck 风格的 n^{0.5}。

**⚠️ 局限性**

局限性包括：①实验仅在 MNIST 数据集上验证，无法说明在更高维度或更复杂数据集上的适用性；②经验 Lipschitz 下界仅基于训练样本之间的差分估计，未考虑全局最坏情况；③模型只对单一架构进行宽度扩展，缺乏对不同网络结构或正则化手段的泛化评估。

---

## 363. When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance

**arXiv ID:** 2602.20880 | [PDF](https://arxiv.org/pdf/2602.20880v1)

**作者:** Yongli Xiang `[一作]` (University of Sydney), Tongliang Liu `[通讯]` (University of Sydney)

**通讯引用:** 12825 | [OpenAlex ID](https://openalex.org/A5065250332)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了冲突感知自适应安全指导（CASG）框架，在文本到图像生成模型中动态识别当前生成状态下最相关的有害类别，并仅沿该类别的安全方向进行引导，从而解决多类别安全引导中的有害冲突问题。

**💡 创新点**

创新点在于引入冲突感知类别识别（CaCI）和冲突解决引导（CrGA）两大模块：CaCI 在每个去噪步骤动态计算与提示引导的余弦相似度或投影残差，自动判定主导有害类别；CrGA 只沿该类别的安全方向进行修正，避免多类别聚合导致的方向不一致和衰减，完全无需额外训练即可实现。

**🔧 技术方法**

技术实现基于已有的安全引导方法（Latent空间的 SLD 与文本空间的 SAFREE），在其基础上嵌入 CaCI 与 CrGA；使用余弦相似度或投影残差评估与有害方向的对齐，并在每个去噪步骤更新安全方向。

**📊 数据集**

实验使用四个文本到图像安全基准：I2P、T2VSafetyBench、Unsafe Diffusion、CoProv2；对比基准模型 Stable Diffusion v1.5 并在 COCO-30k 上评估正常图像质量；评估指标包括 Q16、NudeNet（判定是否有害）、CLIP Score、FID。

**📈 对比分析**

与多种基线（模型编辑、指导、对齐等）对比，CASG+SLD 在所有四个基准上将有害率从 12.7% 降至 10.2%（最高降幅 32%），CASG+SAFREE 也实现了显著提升；图像质量指标（CLIP、FID）基本保持不变，表明安全性提升不损失视觉质量。

**⚠️ 局限性**

局限性包括：仍需预先手工指定有害关键词集合，无法自动覆盖所有潜在有害类别；对极度混合或模糊提示的对齐仍可能出现误判；对极少数特殊有害类别的泛化性能尚未充分验证。

---

## 364. Multilevel Determinants of Overweight and Obesity Among U.S. Children Aged 10-17: Comparative Evaluation of Statistical and Machine Learning Approaches Using the 2021 National Survey of Children's Health

**arXiv ID:** 2602.20303 | [PDF](https://arxiv.org/pdf/2602.20303v1)

**作者:** Joyanta Jyoti Mondal `[一作]` (University of Delaware), Joyanta Jyoti Mondal `[通讯]` (University of Delaware)

**通讯引用:** 71 | [OpenAlex ID](https://openalex.org/A5077483589)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

比较经典统计模型、树模型和深度学习在预测美国10–17岁儿童肥胖方面的性能与可解释性

**💡 创新点**

构建统一比较框架，结合可解释注意力机制和分组公平评估，揭示算法无显著优劣而数据结构决定表现

**🔧 技术方法**

逻辑回归、随机森林、梯度提升、XGBoost、LightGBM、MLP和TabNet等模型，并使用SHAP、特征重要性和注意力权重进行解释

**📊 数据集**

2021年美国儿童健康国家调查（NSCH）全样本18,792名10–17岁儿童

**📈 对比分析**

在AUC 0.70–0.79范围内各模型表现相近，MLP在召回率上略优，但无模型明显优势，公平性指标显示族群与贫困层级差距普遍存在

**⚠️ 局限性**

限制包括二元结果定义导致召回率低、未做深入超参调优与架构消融、缺乏公平性纠正措施、部分饮食变量缺乏实质信息

---

## 365. Maximin Share Guarantees via Limited Cost-Sensitive Sharing

**arXiv ID:** 2602.20541 | [PDF](https://arxiv.org/pdf/2602.20541v1)

**作者:** Hana Salavcova `[一作]` (Charles University), Arpita Biswas `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了允许有限共享的不可分物品的公平分配，提出了k‑共享模型并在此框架下分析了最大最小份额（MMS）和共享最大最小份额（SMMS）公平性。

**💡 创新点**

创新点在于引入成本敏感的k‑共享公平分配模型，证明在一定共享程度下MMS可实现并提出SMMS概念，给出其存在性与近似性结果。

**🔧 技术方法**

主要采用组合数学证明、构造性算法（Shared Bag‑Filling）、与卡丹限制最大最小分享（CMMS）的归约、以及归纳和不等式推导等技术。

**📊 数据集**

论文为理论研究，未使用公开数据集，主要通过构造的示例实例（如3位代理9件商品的反例）来验证理论。

**📈 对比分析**

与传统1‑共享MMS方法比较，提出的算法在k≥n/2时可实现完整MMS，算法为多项式时间，近似因子为(k−1)(1−C)或1/2‑SMMS等；在共享成本较低时性能显著提升。

**⚠️ 局限性**

局限在于一般情况下仍无法保证SMMS存在性，存在性与最大共享成本C密切相关；当共享成本高时近似性能下降；并未考虑非加性估值或动态分配情形。

---

## 366. A Long-Short Flow-Map Perspective for Drifting Models

**arXiv ID:** 2602.20463 | [PDF](https://arxiv.org/pdf/2602.20463v1)

**作者:** Zhiqi Li `[一作]` (Georgia Institute of Technology), Bo Zhu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 172128 | [OpenAlex ID](https://openalex.org/A5100381911)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种长短流图（Long‑Short Flow‑Map）视角，对Drifting Model进行重新诠释，并基于此构建了新的似然学习框架；

**💡 创新点**

创新点在于将闭式流匹配与轨迹一致性相结合，通过短终止流的闭式解提供数据级监督，从而从理论上解释Drifting Model的设计，并提出了在特征空间优化与似然学习的新方法；

**🔧 技术方法**

采用了流匹配（Flow Matching）闭式解、轨迹一致性（semigroup property）约束、梯度回传的闭式核函数、特征映射（MAE encoder）和重要性采样等技术；

**📊 数据集**

在二维数据集（Spiral、Checkerboard、Two Moons）和CelebA‑HQ图像数据集上进行实验；

**📈 对比分析**

与基线方法（如MeanFlow、Consistency Models等）对比，长短流图在2D例子中恢复分布精度高；在CelebA‑HQ上获得FID 14.71（单步生成），仅需64批次即可实现；

**⚠️ 局限性**

局限性包括高维距离退化导致特征空间优化仍需大量特征映射，且对噪声分布与核函数设计的依赖尚未完全克服，缺乏对分类器无关引导（CFG）的理论支持。

---

## 367. SceMoS: Scene-Aware 3D Human Motion Synthesis by Planning with Geometry-Grounded Tokens

**arXiv ID:** 2602.20476 | [PDF](https://arxiv.org/pdf/2602.20476v1)

**作者:** Anindita Ghosh `[一作]` (German Research Center for Artificial Intelligence), Rishabh Dabral `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 565 | [OpenAlex ID](https://openalex.org/A5089712643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于轻量级二维场景线索（BEV 图像和局部高度图）的两阶段文本驱动场景感知 3D 人体运动合成框架，能够生成语义一致且物理可行的运动。

**💡 创新点**

创新点在于将全局运动规划与局部物理执行分离，使用 DINOv2 提取 BEV 语义特征作为全局输入，并通过条件 VQ‑VAE 将高度图嵌入离散运动词表，实现了无须 3D 体素/点云的高效场景感知与几何约束。

**🔧 技术方法**

采用 DINOv2 + BEV 渲染、条件 VQ‑VAE、Transformer 语言+场景特征的自回归规划器、局部高度图与轨迹平滑模块等技术；训练使用跨模态文本嵌入（T5）和运动编码器。

**📊 数据集**

在 TRUMANS 基准数据集上进行训练与评估，包含 100 个室内场景、15 小时 SMPLX 动作和对应文本描述。

**📈 对比分析**

与 TRUMANS、TeSMo、Humanise、SceneDiffuser 等基线对比，本文在 FID、接触准确率、物理一致性等指标上取得最优表现，且场景编码参数仅约 4M，显著低于基线的 35M–86M 参数。

**⚠️ 局限性**

局限性包括：只能处理静态室内场景，难以捕捉细粒度物体抓取等手部交互；对户外崎岖地形或强遮挡的场景适配性不足；推理速度仍受迭代规划与高度图重算影响。

---

## 368. Communication-Inspired Tokenization for Structured Image Representations

**arXiv ID:** 2602.20731 | [PDF](https://arxiv.org/pdf/2602.20731v1)

**作者:** Aram Davtyan `[一作]` (University of Bern), Paolo Favaro `[通讯]` (University of Bern)

**通讯引用:** 9613 | [OpenAlex ID](https://openalex.org/A5070940574)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 COMiT，一个基于人类沟通启发的、使用单一 Transformer 实现编码与解码的 1D 图像离散化框架，能够通过可观测的局部裁剪逐步构建对象级语义结构的离散消息，并使用流匹配解码重建完整图像。

**💡 创新点**

创新在于将编码视为迭代沟通与重构游戏，采用注意力序列化裁剪与全同一网络的“说话者-听者”架构，结合流匹配与语义对齐（SREPA）实现对象级可解释令牌，并通过随机裁剪激励贪婪令牌使用。

**🔧 技术方法**

使用流匹配（flow‑matching）框架训练单一 Transformer（DiT+AdaLN），结合 FSQ 量化、VAE 编码、DINOv2 特征对齐（REPA/SREPA）、注意力探测器、随机裁剪与贪婪更新等技术。

**📊 数据集**

在 ImageNet‑1k 进行预训练和评估，使用 ImageNet‑100 进行分类探针、MSCOCO（分离物体组合）进行组合泛化、Visual‑Genome 进行关系推断，以及 ImageNet‑1k 验证集用于 rFID 和 PSNR 评估。

**📈 对比分析**

与 TiTok、ALIT、FlexTok、SelfTok 等现有 1D 离散化器在 semantic 探针（top‑1 82.9% vs 72% 无 SREPA）、mIoU 0.53 vs 0.34、组合泛化与关系推断等基准上均显著优于前者；在重构质量上与传统压缩目标的对比略逊，模型规模从 B 到 XL 展现典型的权衡。

**⚠️ 局限性**

重构质量仍低于专门的生成模型；未做多阶段微调；裁剪策略与动态自适应探索有限；未扩展到视频或更大规模数据；模型训练对 GPU 内存消耗较高。

---

## 369. Airavat: An Agentic Framework for Internet Measurement

**arXiv ID:** 2602.20924 | [PDF](https://arxiv.org/pdf/2602.20924v1)

**作者:** Alagappan Ramanathan `[一作]` (University of California, Irvine), Sangeetha Abdu Jyothi `[通讯]` (University of California, Irvine)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5058967466)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Airavat框架，利用多代理自动生成互联网测量工作流，并通过知识图谱驱动的验证引擎与验证引擎实现方法学验证与验证策略生成。

**💡 创新点**

首次将代理式工作流生成、知识图谱驱动的系统级验证以及验证策略自动化结合，能够自动识别并修复方法缺陷并为新问题提供验证方案。

**🔧 技术方法**

采用多代理LLM系统（Claude Opus/​Sonnet、Gemini）、知识图谱（Neo4j+embedding）、工具注册表、LLM抽象抽取与代码生成、结构化验证与合成技术。

**📊 数据集**

使用跨领域测量工具注册表、从SIGCOMM/IMC等会议收集的约5000篇论文构建的知识图谱、公开测量数据（BGP、Traceroute、海底电缆、WHOIS、RPKI）以及实验用的海底电缆损坏、灾害影响与IP分配数据。

**📈 对比分析**

通过四个案例（海底电缆损坏、灾害影响、跨洲缆线级联、前缀-组织映射）与专家手工工作流、工具默认实现对比，实验表明Airavat生成的工作流在准确率、代码行数与执行时间上与专家相当，验证引擎将0%准确率提升至>90%，并在验证效率上显著优于传统执行测试。

**⚠️ 局限性**

对工具注册表的人工维护、知识图谱覆盖不足导致无法检测未知方法缺陷、对查询精确度要求高、无法处理长周期/大规模部署或私有数据集等验证需求。

---

## 370. Bikelution: Federated Gradient-Boosting for Scalable Shared Micro-Mobility Demand Forecasting

**arXiv ID:** 2602.20671 | [PDF](https://arxiv.org/pdf/2602.20671v1)

**作者:** Antonios Tziorvas `[一作]` (University of Piraeus), Yannis Theodoridis `[通讯]` (University of Piraeus)

**通讯引用:** 10508 | [OpenAlex ID](https://openalex.org/A5018268830)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Bikelution，一种基于 XGBoost 的水平联邦学习框架，用于预测 dockless 自行车共享系统的六小时需求。

**💡 创新点**

创新点在于将梯度提升树迁移到联邦学习环境，兼顾隐私保护且性能与中心化模型相当，同时在三个真实数据集上超越当前最先进方法。

**🔧 技术方法**

采用 XGBoost、FedProx、FedXGBllr 联邦框架以及 Flower 平台进行训练，并通过 RBF 编码、滚动统计等特征工程提升模型。

**📊 数据集**

使用纽约市、芝加哥和巴塞罗那三大共享单车运营商公开的历史租赁数据。

**📈 对比分析**

与中心化训练和现有 FL 方法对比，Bikelution 在 MAE 与 RMSE 上降低 13.30%/14.96%，且在多步预测中与中心化模型保持近似，性能提升 15% 左右。

**⚠️ 局限性**

主要限制包括：联邦训练下的性能略低于中心化，存在客户端漂移导致误差上升，且对聚合策略和通信成本需进一步优化。

---

## 371. FACTO: Function-space Adaptive Constrained Trajectory Optimization for Robotic Manipulators

**arXiv ID:** 2602.20225 | [PDF](https://arxiv.org/pdf/2602.20225v1)

**作者:** Yichang Feng `[一作]` (Texas A&M University), Minghui Zheng `[通讯]` (Texas A&M University)

**通讯引用:** 2071 | [OpenAlex ID](https://openalex.org/A5066836550)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于函数空间的自适应约束轨迹优化（FACTO），通过正交基函数对时间连续轨迹进行参数化，并在系数空间内直接求解约束优化问题；

**💡 创新点**

创新点包括：①在截断的函数子空间内表示轨迹，显著降低维度并保证时间连续性；②利用EMA平滑的Gauss‑Newton近似与自适应信任域，实现鲁棒收敛；③使用系数空间的零空间投影精确满足等式约束，配合稀疏活跃集合处理不等式约束；④为多机器人设计块结构与解耦策略，支持协作任务与自碰撞检测；

**🔧 技术方法**

核心技术：正交基函数（正弦、余弦、Chebyshev）、系数空间参数化、Gauss‑Newton优化、指数移动平均EMA、零空间投影、活跃集合方法、信任域与LM正则化、OSQP/ADMM求解器、签名距离场（SDF）碰撞检查、时间缩放与动力学限制；

**📊 数据集**

使用公开的单臂任务数据集（MBM 800无约束、200约束任务）、Franka Panda双臂实验数据（40无约束+40约束）以及自建的FR3机器人硬件实验数据；

**📈 对比分析**

与CHOMP、TrajOpt、GPMP2（优化基）以及RRT‑Connect、RRT*、PRM、IMACS‑RRT‑C（采样基）进行对比。FACTO在大多数场景下取得最高或竞争的成功率（>90%），计算时间最低（0.02–0.3 s），轨迹平滑度最佳（粗糙度最小），并在双臂协作任务中保持高效；

**⚠️ 局限性**

局限性：每步计算成本仍高于CHOMP/GPMP2（主要由QP求解器引起），在多臂、任务难度或约束紧缩时收敛速度和成功率会下降；未来工作计划融合学习策略提升鲁棒性并优化求解器平衡。

---

## 372. Wasserstein Distributionally Robust Online Learning

**arXiv ID:** 2602.20403 | [PDF](https://arxiv.org/pdf/2602.20403v1)

**作者:** Guixian Chen `[一作]` (University of Michigan), Soroosh Shafiee `[通讯]` (Cornell University)

**通讯引用:** 967 | [OpenAlex ID](https://openalex.org/A5030361767)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于Wasserstein距离的分布鲁棒在线学习框架，利用在线零和鞍点游戏和分布最佳响应的投影子梯度更新，解决在流式数据中对分布不确定性的风险控制问题。

**💡 创新点**

①将分布鲁棒在线学习建模为在线鞍点游戏，证明收敛到离线Wasserstein DRO解；②针对无限维的极大期望问题，给出与预算分配问题等价的有限维凸表述，设计了多层金字塔搜索与双分支分解的高效求解算法，实现对δ-近似Wasserstein算子的快速求解。

**🔧 技术方法**

在线投影子梯度法、分布最佳响应（Wasserstein）算子、金字塔搜索（golden-section）、双分支分解、凸对偶分解、离散化与稀疏化技巧。

**📊 数据集**

论文未给出具体公开数据集；实验主要通过仿真验证算法在不同场景下的收敛与计算速度。

**📈 对比分析**

与传统的基于完整Wasserstein DRO求解的全局求解器（如Gurobi）及其它在线鲁棒优化方法相比，提出的算法在对数级迭代内达到相同精度，计算时间大幅降低；在实验中表现出更快的收敛速度与更低的运行时间。

**⚠️ 局限性**

局限性包括：仅对分段凹/线性可微损失函数有理论保证；实现依赖于可有效求解子问题的子梯度/投影方法；对高维特征空间和非线性损失的扩展尚未完全解决；实验验证主要在模拟数据，缺乏真实工业数据验证。

---

## 373. RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition

**arXiv ID:** 2602.20735 | [PDF](https://arxiv.org/pdf/2602.20735v1)

**作者:** Kun Ran `[一作]` (RMIT University), Oleg Zendel `[通讯]` (RMIT University)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5052168069)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了R2RAG系统，基于查询复杂度动态路由到单次检索的Vanilla RAG或多轮检索的Vanilla Agent，实现低成本GPU上的动态检索增强生成

**💡 创新点**

将查询分类与迭代检索循环结合，使用LLM评估信息覆盖度并自动决定停止；采用轻量化4B模型与小型reranker，兼顾资源与性能

**🔧 技术方法**

使用Qwen3-4B（文本生成与查询变体）、Qwen3-reranker-0.6b（rerank）、LLM驱动的查询分类与信息覆盖评估，vLLM加速推理，ClueWeb22-A索引检索

**📊 数据集**

训练查询分类器的数据集包括TREC Deep Learning、Deep-Research Questions、TREC RAG 2025、Natural Questions等175,850条问答；检索使用ClueWeb22-A索引

**📈 对比分析**

在NeurIPS 2025 MMU-RAG文本到文本赛道中获得Open Source Dynamic Evaluation最佳奖项，表现优于多种静态/动态评估基线，且在单GPU上保持高效

**⚠️ 局限性**

受限于单GPU内存与上下文窗口，对长篇文档覆盖度仍有限；LLM推理开销导致单查询时间受限；模型对非常复杂、多分支问题的解释仍不完备

---

## 374. Topology-Aware Coordination for Multi-Functional Low-Altitude Wireless Networks

**arXiv ID:** 2602.20993 | [PDF](https://arxiv.org/pdf/2602.20993v1)

**作者:** Jiajun He `[一作]` (Centre for Wireless Innovation), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于拓扑感知（TA）的多功能低空无线网络（LAWN）协同框架，实现E-MT、D-MT和计算中心的分层协作，支持通信、感知与无线能量传输等多功能任务。

**💡 创新点**

核心创新是将LAWN抽象为稀疏图，利用节点特征和可控边规则进行全局任务分配与资源调度，并将图神经网络（GNN）与强化学习相结合，显著降低信令开销并提升协同效率。

**🔧 技术方法**

技术包括拓扑感知图建模、可控边权重设计、GNN与强化学习优化、分层资源分配与任务卸载、以及基于Dijkstra的任务路径搜索等。

**📊 数据集**

主要使用仿真数据：随机在2 km×2 km×50 m 3D空间中布置64架UAV和64个地面AP（E-MT），80%为UAV用户、20%为地面用户，配合4个充电用户和1个感知目标，频段为2.6 GHz。

**📈 对比分析**

与无选择、用户中心化选择以及贪婪局部最优/可达策略对比，TA框架在总谱效率、感知SINR、充电能量获取、任务转发延迟和路径成功率上均优于基线，提升幅度从10%至50%不等。

**⚠️ 局限性**

局限包括：图抽象可能丢失弱连边信息导致最优解偏差、稀疏图密度与计算复杂度权衡、对高速移动导致拓扑频繁变更的适应性不足、以及安全、隐私与标准化缺失等挑战。

---

## 375. Generative Pseudo-Labeling for Pre-Ranking with LLMs

**arXiv ID:** 2602.20995 | [PDF](https://arxiv.org/pdf/2602.20995v1)

**作者:** Junyu Bi `[一作]` (Alibaba Group), Yuning Jiang `[通讯]` (Alibaba Group)

**通讯引用:** 3933 | [OpenAlex ID](https://openalex.org/A5074655314)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Generative Pseudo-Labeling (GPL) 框架，利用大型语言模型（LLM）生成无偏兴趣锚点，为未曝光候选项产生伪标签，从而提升预排序阶段的点击率与多样性。

**💡 创新点**

创新点：① 用LLM在离散化的语义空间中生成内容感知的兴趣锚点；② 将锚点与候选项在多模态语义空间对齐，采用最大池化得到无偏相关度；③ 引入三维不确定性加权（语义离散、历史一致性、LLM置信度）校正伪标签，真正对抗曝光偏差。

**🔧 技术方法**

技术手段：多模态预训练编码器（CLIP/CLIP‑style）+ RQ‑VAE离散化；LLM（Qwen2.5 0.5B/1.8B/7B）进行层级 Beam Search；最大池化匹配与 Sigmoid 归一化；置信度加权的双标签联合损失；离线生成与缓存，在线无延迟。

**📊 数据集**

数据集：① 规模化工业数据集（Taobao 14 天，约 200M 用户、20M 商品、30B 交互）；② Taobao‑MM 公共多模态数据集（8.79M 用户、35.4M 商品）。

**📈 对比分析**

与多种基线（BC, KD, TL, MUDA, UKD, UECF, SIDA, DAMCAR）比较。离线指标 HR@3/5/10、AUC、GAUC、AUC* 均优于所有对照，最佳提升约 5–10%；在线 A/B 测试显示 CTR +3.07%，IPV +3.53%，CTCVR +2.51%，并显著提升长尾覆盖与类别多样性。

**⚠️ 局限性**

局限性：① 伪标签质量受 LLM 生成能力与多模态编码器质量限制；② 对极端稀疏或新颖内容的推荐仍可能不足；③ 离线生成消耗较大，实时动态更新仍需研究；④ 仍未彻底消除因数据稀疏导致的冷启动问题。

---

## 376. "Are You Sure?": An Empirical Study of Human Perception Vulnerability in LLM-Driven Agentic Systems

**arXiv ID:** 2602.21127 | [PDF](https://arxiv.org/pdf/2602.21127v1)

**作者:** Xinfeng Li `[一作]` (Nanyang Technological University), Xiaofeng Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 8654 | [OpenAlex ID](https://openalex.org/A5115602812)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过设计 Human-Agent Trust Laboratory 平台，开展 303 名参与者的实验，量化 LLM 代理系统中的人类易受欺骗性。

**💡 创新点**

首次系统性测评 Agent-Mediated Deception (AMD) 的人类易感性，并发现专家更易受骗、提出六种认知失败模式及“安全心态”对抗措施。

**🔧 技术方法**

使用高保真模拟平台、LLM Agent（如 GPT‑4o 等）、自定义攻击（感知、记忆、动作层面注入）进行实验。

**📊 数据集**

基于自制的九个场景（HR、医疗、软件等）和真实任务资源（简历、代码库、邮件等）构成实验数据。

**📈 对比分析**

与三种防御层级（静态声明、持续提醒、交互警报）对比，交互警报提升风险感知至 17.2%，整体防御效果逐级递增。

**⚠️ 局限性**

研究仅为横断面实验，缺乏长期跟踪、对不同专家领域的泛化验证，且攻击方式仍为静态配置。

---

## 377. Acoustic Feedback for Closed-Loop Force Control in Robotic Grinding

**arXiv ID:** 2602.20596 | [PDF](https://arxiv.org/pdf/2602.20596v1)

**作者:** Zongyuan Zhang `[一作]` (Queensland University of Technology), Jonathan M. Roberts `[通讯]` (Queensland University of Technology)

**通讯引用:** 3319 | [OpenAlex ID](https://openalex.org/A5081990588)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种仅利用低成本接触式麦克风实现实时声学反馈的机器人研磨系统（AFRG），通过声学信号估算研磨力并实现闭环控制。

**💡 创新点**

创新点在于用数据驱动的二维CNN从实时功率谱编码中回归研磨力，完全摆脱昂贵的力/扭矩传感器，并在不同研磨盘状态下实现4倍的MRR一致性。

**🔧 技术方法**

技术包括实时功率谱密度编码、PSDRegNet（2D CNN + 全连接层）、混合力-位置控制器以及基于力的PID/阻抗混合控制。

**📊 数据集**

数据集为在工业环境下使用F/T传感器闭环控制采集的1400 s音频与力标注数据，涵盖连续与间歇研磨、不同目标力（2–7 N）和两种工件材质。

**📈 对比分析**

通过与传统F/T传感器闭环控制对比，AFRG在钢板研磨盘磨损实验中实现MRR变化仅为传统方法的1/4，估计RMSE 0.23 N，稳态误差0.05 N。

**⚠️ 局限性**

局限包括无法自动检测研磨盘失效、训练仍需F/T标签、以及模型在不同工具/麦克风布局上的泛化性未知。

---

## 378. Diffusion Modulation via Environment Mechanism Modeling for Planning

**arXiv ID:** 2602.20422 | [PDF](https://arxiv.org/pdf/2602.20422v1)

**作者:** Hanping Zhang `[一作]` (Carleton University), Yuhong Guo `[通讯]` (Carleton University)

**通讯引用:** 8115 | [OpenAlex ID](https://openalex.org/A5043824291)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于扩散模型的离线强化学习规划方法——DMEMM，通过将环境转移动力学和奖励函数嵌入到扩散模型训练与采样中，提升生成轨迹的一致性和奖励最优化。

**💡 创新点**

创新点在于：①将奖励加权的扩散损失与两种辅助调制损失（基于转移动态和奖励函数）结合；②在逆扩散采样过程中加入双向引导（奖励梯度和转移梯度），实现对轨迹的同时奖励与动态一致性引导。

**🔧 技术方法**

使用了扩散模型（DDPM/Latent Diffuser）与其反向噪声网络；基于离线数据学习的概率转移模型和奖励模型；在采样阶段应用双向引导梯度。

**📊 数据集**

实验数据集主要为D4RL中的HalfCheetah、Hopper、Walker2d三种运动任务（Med-Expert、Medium、Med-Replay）以及Maze2D与Multi2D导航环境。

**📈 对比分析**

与BCQ、BEAR、CQL、IQL、Decision Transformer、MoReL、Trajectory Transformer、RvS、Diffuser、HD-DA等方法比较。DMEMM在D4RL运动任务中平均得分达87.9，显著高于第二名84.6；在Maze2D和Multi2D任务中相较于HD-DA和Diffuser分别提升约4–20分，表现最优。

**⚠️ 局限性**

局限性包括：对大规模长程迷宫任务（Large级别）性能略逊于层次化方法HD-DA；对超参数（λ_tr、λ_rd）的敏感度仍需进一步研究；实验主要集中在离线数据丰富的仿真环境，实际环境适用性尚未验证。

---

## 379. Analyzing Latency Hiding and Parallelism in an MLIR-based AI Kernel Compiler

**arXiv ID:** 2602.20204 | [PDF](https://arxiv.org/pdf/2602.20204v1)

**作者:** Javed Absar `[一作]` (Qualcomm Technologies International), Muthu Baskaran `[通讯]` (Qualcomm Technologies)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可复现的基准方法，利用ablation ladder对边缘NPU上的向量化、线程并行和双缓冲三种编译器机制的性能贡献进行归因分析；

**💡 创新点**

创新点在于将三种优化机制拆分为可独立评估的阶梯，并实现了结构化IR级别的MT和DB转换，提供了可重复、可扩展的性能归因框架；

**🔧 技术方法**

采用了Triton/Inductor生成的内核、结构化IR转换（虚拟线程、fork‑join）、软件流水线双缓冲、异步DMA等技术；

**📊 数据集**

使用了两类代表性内核：一个带宽敏感的二维向量加法微基准（[64, 128×128]元素）和GELU激活核，GELU 还进行了不同规模的大小扫描；

**📈 对比分析**

通过比较标量、Vec、Vec+MT、Vec+MT+DB 四种配置，向量化实现约 41.3× 的加速，MT 在大规模 GElU 中可达约 3.9× 的加速，DB 进一步提升约 10%；总体表现为三种机制互补的增量加速；

**⚠️ 局限性**

局限性包括仅测试两种核，假设独立 tile 适用，依赖特定 NPU 架构的内存与传输模型，且未覆盖如 RMSNorm、softmax 等其他常见算子。

---

## 380. BBQ-to-Image: Numeric Bounding Box and Qolor Control in Large-Scale Text-to-Image Models

**arXiv ID:** 2602.20672 | [PDF](https://arxiv.org/pdf/2602.20672v1)

**作者:** Eliran Kachlon `[一作]`, Ron Mokady `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 BBQ 模型，利用大规模文本到图像扩散框架，在统一的结构化文本中直接嵌入数值化的边界框坐标和 RGB 颜色值，实现对物体位置、大小和颜色的精确控制。

**💡 创新点**

创新点在于：①将数值参数直接写入文本提示，完全不需要额外的定位标记或架构改造；②通过在训练时扩充结构化字幕（包含边界框与颜色），让模型自然学习数值对齐；③使用 VLM 作为推理桥梁，将自然语言或编辑指令转换为完整的数值化结构化提示，支持交互式拖拽与调色。

**🔧 技术方法**

技术细节包括：基于 8B FIBO 变压器的流匹配扩散模型；使用 AdamW、flow‑matching 目标以及 logit‑normal 噪声调度进行大规模训练；对训练集进行 25M 图像的数值化标注；在推理时通过 4B VLM 生成或编辑结构化 JSON；在后续进行美学微调和 DPO 训练以提升文本渲染质量。

**📊 数据集**

数据集：从 25M 公开图像中使用 FIBO 风格字幕生成结构化文本，并利用 SAM2 提取边界框、Depth Anything V2 估计相对深度、Pylette 计算主色调；此外还使用 COCO、LVIS 进行评测。

**📈 对比分析**

评价方法包括：TaBR（图像重建比对）验证表达能力；基于 YOLOv8 / ViTDet-L 的边界框精度评估与 InstanceDiffusion、GLIGEN 等专用布局模型对比；CIEDE2000 与 a‑b 色度距离评估 RGB 精准度。结果显示，BBQ 在表达能力、边界框对齐和颜色精度上均优于 Flux.2 Pro、Nano Banana Pro 等通用模型，并且在无需额外推理优化的情况下，逼近甚至超越部分专用布局模型。

**⚠️ 局限性**

局限性：①相较于专门的布局模型（如 InstanceDiffusion），在极端精细的边界框对齐上仍略逊；②依赖大规模训练数据与强大算力；③当前仅支持 RGB 颜色，未覆盖色温、光照等更高级的参数；④推理桥梁 VLM 可能成为瓶颈，若 VLM 生成的结构化提示不准确，最终图像质量会下降。

---

## 381. Stability Under Valuation Updates in Coalition Formation

**arXiv ID:** 2602.21041 | [PDF](https://arxiv.org/pdf/2602.21041v1)

**作者:** Fabian Frank `[一作]` (Technical University of Munich), René Romen `[通讯]` (Technical University of Munich)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5028529412)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在可加分离性幸福游戏（ASHG）中研究代理偏好更新后，寻找与原稳定划分距离不超过给定阈值的近似稳定合作社结构。

**💡 创新点**

证明该问题在四种稳定概念（NS、IS、CNS、CIS）下，即使取值限制为{-1,0,1}也为NP‑完全；在严格对称游戏中给出CIS的多项式解法与CNS的距离上界，并揭示动态更新与平均距离之间的根本差异。

**🔧 技术方法**

利用集合覆盖等经典NP难问题的多重归约、潜能函数论证以及距离度量分析，结合稳定性定义进行复杂度证明与算法设计。

**📊 数据集**

论文未采用真实数据集，而是通过构造性的游戏实例（如特殊权值图、分割集合）来展示理论结果。

**📈 对比分析**

通过理论构造的对照实例与归约证明难度；在可解情形下提供多项式时间算法，并证明在长期更新序列中的平均距离上界为常数，表明算法在这类问题上具有可接受的时间与稳定性表现。

**⚠️ 局限性**

主要局限在于对非对称游戏以及NS/IS在对称FEG/AFG中的可解性尚未给出多项式算法；结果受限于严格对称与取值范围，缺乏经验性验证。

---

## 382. ActionReasoning: Robot Action Reasoning in 3D Space with LLM for Robotic Brick Stacking

**arXiv ID:** 2602.21161 | [PDF](https://arxiv.org/pdf/2602.21161v1)

**作者:** Guangming Wang `[一作]` (University of Cambridge), Brian Sheil `[通讯]` (University of Cambridge)

**通讯引用:** 12056 | [OpenAlex ID](https://openalex.org/A5075413962)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了基于LLM的多代理框架ActionReasoning，利用3D世界模型实现机器人砖块堆叠的物理推理和决策。

**💡 创新点**

将物理先验与大规模语言模型结合，在SE(3)空间进行显式物理推理；通过多代理、门控结构实现阶段化推理；低层控制最小化，提升可迁移性。

**🔧 技术方法**

使用LLM（如GPT‑4）多代理框架、结构化提示、可调用工具（碰撞检测、接触评估等）、基于世界模型的环境表示、PyBullet仿真。

**📊 数据集**

无需大规模任务数据，使用随机生成的砖块姿态与固定目标堆叠图案，在模拟环境中评估。

**📈 对比分析**

与传统手工脚本控制基线和单代理方法对比，评估旋转误差、中心偏移、3D IoU；ActionReasoning在所有指标上显著优于基线（误差降低≈85%，IoU提升≈130%），单代理方法性能下降，堆叠失稳。

**⚠️ 局限性**

仅在仿真环境验证，缺乏真实机器人实验；依赖准确的3D感知输入；LLM推理延迟与可解释性有限；仅针对砖块堆叠，尚未扩展到更复杂任务。

---

## 383. How communicatively optimal are exact numeral systems? Once more on lexicon size and morphosyntactic complexity

**arXiv ID:** 2602.20372 | [PDF](https://arxiv.org/pdf/2602.20372v1)

**作者:** Chundra Cathcart `[一作]` (University of Zurich), Johann-Mattis List `[通讯]` (University of Passau)

**通讯引用:** 4052 | [OpenAlex ID](https://openalex.org/A5012676548)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化世界各语种数词系统的词汇量与形态复杂度的权衡，使用手工标注的数据对比理论最优系统。

**💡 创新点**

引入可预测与不可预测异体的区分，扩展数据规模至52种语种，并通过Bayesian混合模型揭示系统效率的两个子群。

**🔧 技术方法**

采用手工分词标注、形态学注释、演化算法生成假设系统以及Bayesian混合回归模型进行分析。

**📊 数据集**

使用CoSiNuS v2.0公开数据集，涵盖52种语言的1–99数字手工分割和注释，数据已托管于GitHub与Zenodo。

**📈 对比分析**

将实证系统与演化算法生成的Pareto前沿对齐，并用留一期望对数似然（ELPD）比较单一与两元混合模型，结果显示两元混合模型显著更优。

**⚠️ 局限性**

仍未充分模拟子词层面的不可预测异体，且混合模型仅识别两类子群，未具体阐明历史或社会因素对效率偏离的机制。

---

## 384. Deep Reinforcement Learning Based Block Coordinate Descent for Downlink Weighted Sum-rate Maximization on AI-Native Wireless Networks

**arXiv ID:** 2602.20724 | [PDF](https://arxiv.org/pdf/2602.20724v1)

**作者:** Siya Chen `[一作]` (Dongguan University of Technology), H. Vincent Poor `[通讯]` (Princeton University)

**通讯引用:** 153860 | [OpenAlex ID](https://openalex.org/A5042307561)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种将深度强化学习（DRL）与块坐标下降（BCD）算法结合的框架，用于解决受总功率约束的下行加权求和速率最大化（WSRM）问题。

**💡 创新点**

创新点在于：①将 BCD 的迭代结构（γ 的更新）嵌入到 DRL 环境状态，使得智能体能够利用问题的凸性信息进行决策；②将连续动作空间设为 BCD 子问题中的 y 变量，而非直接控制功率，显著提升学习稳定性与收敛速度；③通过理论分析确保算法满足约束且可实现全局最优或极优解；④展现框架的可扩展性，能够同样应用于联合波束成形的 WSRM。

**🔧 技术方法**

技术手段包括：块坐标下降（高效求解非凸 WSRM 子问题），深度确定性策略梯度（DDPG）与双延迟确定性策略梯度（TD3）两种 Actor‑Critic 训练方法，卷积滤波器用于状态特征提取，经验回放、目标网络、噪声探索等 DRL 常用技巧。

**📊 数据集**

使用 1×10⁵ 个由随机信道增益、噪声功率、预编码器等参数生成的合成 WSRM 实例；实验代码及数据已公开到 GitHub（https://github.com/convexsoft/DRL-based-BCD）。

**📈 对比分析**

与基线方法（WMMSE、SCA、纯 DDPG/TD3、GNN）对比：在多种网络规模（L=4, 32, 64, 128）下，DRL‑BCD 取得了更低的加权求和速率误差、更快的收敛步数、更高的鲁棒性（对学习率、折扣因子、最大步数、批量大小等超参数不敏感）。在测试时，虽然推理时间略高于纯 DRL 方法，但训练时间更短，整体性能显著优于基线。

**⚠️ 局限性**

局限性：①训练过程仍需要大量样本与时间，尤其在网络规模扩大时训练成本上升；②推理时每一步需要执行 BCD 子问题迭代，导致推理速度略逊于纯网络方法；③目前仅验证了固定或交替预编码的下行 WSRM，尚未在更复杂的时变、非静态环境中验证；④对极大规模多天线系统的可扩展性与实现细节仍待进一步研究。

---

## 385. PRECTR-V2:Unified Relevance-CTR Framework with Cross-User Preference Mining, Exposure Bias Correction, and LLM-Distilled Encoder Optimization

**arXiv ID:** 2602.20676 | [PDF](https://arxiv.org/pdf/2602.20676v1)

**作者:** Shuzhi Cao `[一作]` (Alibaba Group), Jufeng Chen `[通讯]` (Alibaba Group)

**通讯引用:** 637 | [OpenAlex ID](https://openalex.org/A5042141859)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PRECTR-V2框架，统一提升搜索相关性匹配与CTR预测，并针对低活跃用户、曝光偏差与模型架构不匹配三大瓶颈进行改进。

**💡 创新点**

创新点包括：1）跨用户相关性偏好挖掘，通过查询类别共享历史行为提升冷启动用户个性化；2）合成硬负样本暴露偏差校正，结合嵌入噪声注入与相对排序损失、距离惩罚和动态截断权重；3）LLM蒸馏的轻量Transformer编码器，实现CTR对齐的可训练文本表示。

**🔧 技术方法**

使用多头目标注意力、Mixture of Experts、对比式对数似然+距离惩罚、动态截断权重、LLM（Qwen‑7B）蒸馏、文本相关性分类SFT、轻量Transformer三层、Embedding+MLP等组合技术。

**📊 数据集**

实验基于阿里二手交易平台Xianyu的点击日志数据，日均1.6 B条，9天数据（7天训练+2天测试），并在线上A/B测试中评估。

**📈 对比分析**

与LR、DNN、Wide&Deep、DeepFM、XDeepFM、DIN、SuKD以及PRECTR基线在离线AUC/GAUC/RelaImpr指标上对比，PRECTR‑V2在AUC0.7674、GAUC0.6933的基础上相较PRECTR提升5.6%/5.3%；线上A/B测试实现订单+1.39%、GMV+3.18%；消融实验验证各模块贡献。

**⚠️ 局限性**

局限性：PCOC指标略低（1.7% vs 2.3%），顶级10结果中无关商品比例微升0.15%；仍需在冷启动、硬负样本生成的随机性与模型对绝对预测与排序的权衡方面进一步完善。

---

## 386. What Makes a Good Query? Measuring the Impact of Human-Confusing Linguistic Features on LLM Performance

**arXiv ID:** 2602.20300 | [PDF](https://arxiv.org/pdf/2602.20300v1)

**作者:** William Watson `[一作]` (J.P. Morgan AI Research), Manuela Veloso `[通讯]` (J.P. Morgan AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 17 维查询特征向量，对 369,837 条真实问答对进行量化分析，探究查询语义特征与 LLM 幻觉风险的关联，并提出基于特征的预处理与重写策略。

**💡 创新点**

将经典语言学特征系统化为 LLM 幻觉评估指标，提出大规模经验“风险景观”，并给出可操作的低成本查询重写规则，首次从输入侧系统性评估幻觉风险。

**🔧 技术方法**

利用 LLM 驱动的特征检测、语义等价扰动生成、混合幻觉检测代理、比例-秩对数回归、ECDF 分布分析、倾向得分诊断和留一数据集检验等技术。

**📊 数据集**

13 个问答数据集，覆盖抽取、选择题、摘要三种场景（SQuAD、TruthfulQA、SciQ、MMLU、PIQA、BoolQ、OpenBookQA、MathQA、ARC-Easy、ARC-Challenge、WikiQA、HotpotQA、TriviaQA），共 369,837 条查询。

**📈 对比分析**

通过 ECDF、KS 距离、Δmedian、回归系数、IPW 提升度等多维诊断量化特征与幻觉的关联；结果显示缺乏特异性、深句法嵌套等特征显著提升风险，意图/答案可答降低风险，低成本重写在多任务场景下可显著降低风险率。

**⚠️ 局限性**

仅为观察性关联，未实现因果推断；模型版本、语言多样性、噪声解析等因素未覆盖；特征相互关联且缺少高阶交互；LLM 判断器可能引入偏差；仅限英文单模 LLM。

---

## 387. QEDBENCH: Quantifying the Alignment Gap in Automated Evaluation of University-Level Mathematical Proofs

**arXiv ID:** 2602.20629 | [PDF](https://arxiv.org/pdf/2602.20629v1)

**作者:** Santiago Gonzalez `[一作]` (Yale University), Quanquan C. Liu `[通讯]` (Yale University)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5079747792)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型在评估大学层级数学证明时存在的“对齐缺口”，并提出了双层评分基准QEDBench来衡量模型与人类专家的评估差距。

**💡 创新点**

创新点在于：①构建大规模双层评分系统，区分课程规范与专家通用标准；②系统化评估多种前沿生成与评估模型，揭示正向偏差、离散与连续推理差距、以及“附和陷阱”；③证明评估提示工程对模型对齐几乎无效，强调过程监督的必要性。

**🔧 技术方法**

技术包括：大型语言模型生成（Gemini 3.0 Pro、GPT‑5 Pro、Claude Opus 4.5 等）与评估（GPT‑5.2 Pro、Claude Sonnet 4.5、Llama 4 Maverick 等）交叉矩阵；采用严格的JSON输出提示、双层评分 rubric；统计学方法如均值差异、t检验、相关系数、误差率（Leniency/Harshness）等。

**📊 数据集**

数据集为QEDBench，包含272门上本科/早期研究生数学题（分析、代数、离散数学等），共1,300+证明，超过1,000小时人类专家手工评分。

**📈 对比分析**

与人类专家基准对比，模型在连续领域（ODE、概率）表现良好，但在离散领域（组合、图论）下降；评估模型存在显著正向偏差（如Claude Opus +0.36），同时大多数评估器在“严谨度”上与专家不匹配；通过平均分、通过率、误差率等多维度衡量，表明现有LLM评估器难以与专家保持一致。

**⚠️ 局限性**

局限性包括：只使用英文证明；人类基准可能不覆盖所有合法证明；评估模型对提示工程鲁棒性低，难以通过Prompt调整对齐；未评估在非标准或多语言场景下的表现；并未测量对评估者的训练影响或长期对齐效果。

---

## 388. 3DSPA: A 3D Semantic Point Autoencoder for Evaluating Video Realism

**arXiv ID:** 2602.20354 | [PDF](https://arxiv.org/pdf/2602.20354v1)

**作者:** Bhavik Chandna `[一作]` (University of California), Kelsey R. Allen `[通讯]` (University of British Columbia)

**通讯引用:** 3431 | [OpenAlex ID](https://openalex.org/A5023131292)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于3D语义点自编码器的自动化视频真实性评估框架，能在不依赖参考视频的情况下衡量生成视频的物理与语义一致性。

**💡 创新点**

创新点在于将3D点轨迹、深度信息与DINOv2语义特征融合到自编码器中，利用Perceiver transformer压缩运动与语义表征，显著提升对物理规则违反和人类真实感评估的敏感度。

**🔧 技术方法**

使用3D点轨迹自编码器、Perceiver‑style transformer、DINOv2语义特征、VideoDepthAnything深度估计以及自编码器损失的组合技术。

**📊 数据集**

训练与评估所用数据集包括合成的Kubric3D、真实的TAPVid‑3D、物理规则验证数据集IntPhys2，以及生成视频评估数据集EvalCrafter和VideoPhy‑2。

**📈 对比分析**

在3D跟踪、物理规则检测与人类评估相关性上均优于现有基线（如TRAJAN、Vision‑Language模型等）；在IntPhys2上赢率>90%，在EvalCrafter/VideoPhy‑2的Spearman系数分别达0.74，接近人类水平。

**⚠️ 局限性**

对复杂场景下深度估计不稳导致轨迹重建误差；模型对极端遮挡和深度误差敏感，需进一步提升轨迹重建鲁棒性。

---

## 389. Exploiting Dependency and Parallelism: Real-Time Scheduling and Analysis for GPU Tasks

**arXiv ID:** 2602.20826 | [PDF](https://arxiv.org/pdf/2602.20826v1)

**作者:** Yuanhai Zhang `[一作]` (Sun Yat-sen University), Kai Huang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 16656 | [OpenAlex ID](https://openalex.org/A5100776772)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对GPU任务的有向无环图（DAG）结构，提出了一种基于子图划分的调度与时序分析框架，能够在不需要额外硬件或软件支持的前提下，通过调节内核级并行度、节点分段和额外依赖来实现可预测且更短的任务完成时间。

**💡 创新点**

创新点包括：①将DAG拆分为平衡组，保证每组内的并行度不超过SM数并能按顺序执行；②通过并行度缩放使同组内的内核执行时间趋于一致；③对过大节点做分段，防止资源争用；④引入额外依赖确保各平衡组严格顺序，从而实现无优先级假设的安全上界。

**🔧 技术方法**

使用了标准CUDA API（streams、events、CUDA Graphs）进行并行度配置与依赖控制；实现子图划分、并行度缩放、节点分段与额外依赖的算法；采用Gustafson定律与Roofline模型来估算内核执行时间。

**📊 数据集**

数据集包括：1）随机生成的规模化DAG（层数5-8，节点数2-30，平均负载可调），共1,000个；2）真实工作负载基准——Laplace、Gaussian Elimination、Stencil，分别在NVIDIA RTX 3060（M=30）和Jetson Orin Nano（M=8）上执行。

**📈 对比分析**

与Greedy、Greedy+unaware和Graham_para等三种基线方法比较。实验结果显示：在各种SM数、并行度参数P和图大小|V|下，所提方法的最坏情况完成时间平均降低32.8%（相对Greedy）和18.2%（相对Graham_para），实际测量执行时间平均提升21.3%，且标准差更小。

**⚠️ 局限性**

局限性：仅针对无条件DAG，假设所有SM资源相同；对极小或极大平均负载时并行度缩放收益有限；不考虑多任务共享GPU或异构平台的情况，且未覆盖条件分支与动态任务生成。

---

## 390. Visual Cooperative Drone Tracking for Open-Path Gas Measurements

**arXiv ID:** 2602.20768 | [PDF](https://arxiv.org/pdf/2602.20768v1)

**作者:** Marius Schaab `[一作]` (Technical University of Munich), Achim J. Lilienthal `[通讯]` (Technical University of Munich)

**通讯引用:** 9109 | [OpenAlex ID](https://openalex.org/A5088586617)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了一个自动化机器人系统，利用地面 PTU 自动对准携带反射器的无人机的 TDLAS 激光束，实现无侵入式的开放路径 CO2 测量。

**💡 创新点**

创新点在于：① 将视觉跟踪（红色 LED 识别、DBSCAN 聚类、PI 控制）与 GNSS 信息结合，实现在三维空间内 60 m 以内的实时对准；② 通过无人机携带可移动反射器打破传统固定反射器限制，实现可自由选择的测量路径；③ 采用动态相机变焦控制和协同跟踪，提升视觉跟踪的鲁棒性。

**🔧 技术方法**

使用技术包括：Tunable Diode Laser Absorption Spectroscopy (TDLAS)、RTK‑GNSS 定位、PTU（Pan‑Tilt Unit）控制、摄像头 HSV‑R 色彩过滤 + DBSCAN 聚类、PI 控制器、Python/OpenCV 图像处理、Raspberry Pi 计算平台。

**📊 数据集**

使用的数据集为现场实验数据：① 预设锯齿路线飞行、记录 TDLAS 状态码与距离；② CO₂ 泄漏瓶（25 L/min）释放实验，记录无人机轨迹、T‑DLAS 读数及风速/温度等环境信息。

**📈 对比分析**

对比方法：与手动对准和之前的地面机器人/无人机固定反射器方案比较；性能表现：在 60 m 以内可获得有效（OK/WARN）测量，跟踪误差受 PTU 速度限制时短暂失去；测量精度受反射器尺寸与定位误差影响，误差约 4 ppm（大于 TDLAS 本身的 1 ppm 误差）。

**⚠️ 局限性**

局限性：① 反射器尺寸受重量限制，影响测量距离；② RTK‑GNSS 与设备间的物理偏移（最多 40 cm）导致 4 ppm 误差，需进一步补偿；③ 仅测 CO₂，需更换传感器才能扩展至甲烷、H₂S 等；④ 追踪性能受 PTU 速度与无人机突变飞行方向影响，需更平滑路径或更快 PTU；⑤ 高风速或强日照下的光干扰仍需进一步评估。

---

## 391. Polynomial Identity Testing and Reconstruction for Depth-4 Powering Circuits of High Degree

**arXiv ID:** 2602.20832 | [PDF](https://arxiv.org/pdf/2602.20832v1)

**作者:** Amir Shpilka `[一作]` (Tel Aviv University), Yann Tal `[通讯]` (Tel Aviv University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了深度4算术电路 Σ^[r]∧^[d]Σ^[s]Π^[δ] 的确定性多项式身份测试与重构算法。

**💡 创新点**

首次给出在无上界顶层 fan‑in 条件下的多项式规模判定集与多项式时间重构算法，突破了以往仅有随机化或平均情况的限制。

**🔧 技术方法**

采用 ABC 定理、Wronskian、Klivans–Spielman 生成器与差分算子等代数工具构造判定集并实现重构。

**📊 数据集**

论文为理论算法研究，无需使用具体实验数据集。

**📈 对比分析**

与先前的随机化或参数受限方法相比，本文在相同模型下实现了确定性多项式时间，构造的判定集规模为 O(r^4 s^4 n^2 d δ^3)。

**⚠️ 局限性**

主要限制在于需要 d ≳ r^4 δ 的条件以及对特征为 0 或足够大的有限域，且构造的判定集规模仍为多项式级，尚未达到最优。

---

## 392. CGSTA: Cross-Scale Graph Contrast with Stability-Aware Alignment for Multivariate Time-Series Anomaly Detection

**arXiv ID:** 2602.20468 | [PDF](https://arxiv.org/pdf/2602.20468v1)

**作者:** Zhongpeng Qi `[一作]` (Dalian Maritime University), Zhuoxuan Liang `[通讯]` (Harbin Engineering University)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5055514169)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种多尺度图对比学习框架（DLGC + CDS + SAA），用于多变量时间序列异常检测。

**💡 创新点**

创新点在于①构造局部、区域、全局三层动态图以捕获不同粒度的变量依赖；②跨尺度对比学习（CDS）实现各尺度内部判别与跨尺度一致性；③稳定性对齐（SAA）通过EMA稳定图与动态图对齐，抑制噪声与漂移。

**🔧 技术方法**

使用注意力构建图、图卷积网络编码、InfoNCE对比损失、EMA更新、条件密度估计等技术。

**📊 数据集**

在四大公开基准上评测：PSM、SWaT、WADI、SMAP。

**📈 对比分析**

与七种基线（Anomaly-Transformer、GANF、MTGFLOW、FCSTGNN、SARAD、CATCH、GCAD）对比，本文在PSM、WADI上显著优于对手，在SWaT、SMAP保持最优或相近性能，且在多尺度与稳定性方面表现更稳健。

**⚠️ 局限性**

局限性包括对稀疏或噪声较大的数据集效果相对逊色，模型对超参数敏感，动态分层分配尚未完全自适应，实时流式部署效率和对数据污染、分布漂移的鲁棒性待进一步提升。

---

## 393. Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques

**arXiv ID:** 2602.20342 | [PDF](https://arxiv.org/pdf/2602.20342v1)

**作者:** Christos Maikos `[一作]` (Harokopio University of Athens), Georgios Th. Papadopoulos `[通讯]` (Athena Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

实现了一个将无人机视频流实时转换为高保真3D Gaussian Splatting模型的端到端系统。

**💡 创新点**

首次将3D Gaussian Splatting与RTMP流、WebSocket实时更新及AR/VR可视化集成，提供低延迟、高帧率的实时3D重建。

**🔧 技术方法**

使用3D Gaussian Splatting、RTMP实时传输、WebSocket实时同步、相机位姿估计、传感器融合以及Unity/WebGL渲染。

**📊 数据集**

采用无人机在户外体育场景的实时视频数据，并与离线高保真参考模型进行对比。

**📈 对比分析**

与NeRF基准相比，重建误差保持在4-7%，渲染性能提升数倍，端到端延迟显著降低。

**⚠️ 局限性**

受限于网络带宽、动态场景处理、资源受限设备对3DGS渲染的支持，以及在低光照或运动模糊环境下的鲁棒性不足。

---

## 394. Grasp to Act: Dexterous Grasping for Tool Use in Dynamic Settings

**arXiv ID:** 2602.20466 | [PDF](https://arxiv.org/pdf/2602.20466v1)

**作者:** Harsh Gupta `[一作]` (University of Illinois Urbana-Champaign), Wenzhen Yuan `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4094 | [OpenAlex ID](https://openalex.org/A5055947140)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Grasp-to-Act 混合系统，结合物理学抓取优化与强化学习在线适配，实现多工具任务下的鲁棒抓取与动态操作。

**💡 创新点**

创新点包括：①利用人类演示初始化抓取位置与轨迹；②设计全维力矩扰动评估抓取稳定性；③在抓取后通过 RL 残差控制实时纠正滑移；④实现零射击模拟到现实的高效迁移。

**🔧 技术方法**

技术手段涵盖：物理仿真平台（Isaac Lab）、人类动作捕捉与姿态估计（FoundationPose、Grounded‑SAM、HaMeR）、强化学习（PPO + LSTM）、力矩扰动模型与 wrench‑space 稳定性评分。

**📊 数据集**

使用数据：5 个工具使用任务的 RGB‑D 演示视频，配合 RealSense D435 与 OptiTrack 采集的姿态数据；未采用公开大规模抓取数据集，主要以实验任务数据为主。

**📈 对比分析**

与解析优化、RL 基线、RL+接触奖励、预抓取姿态、eigengrasp、仅优化抓取等六种基线对比；在仿真与真实硬件上均达成 100% 抓取成功，E_t 与 E_θ 显著低于对照组，任务完成率最高。

**⚠️ 局限性**

局限性：①假设可抓取区域无障碍且物体几何简单；②RL 适配仅针对单一任务，跨任务迁移未实现；③在桌面或复杂形状物体上可能需额外的抓取调整或姿态规划。

---

## 395. Wireless Federated Multi-Task LLM Fine-Tuning via Sparse-and-Orthogonal LoRA

**arXiv ID:** 2602.20492 | [PDF](https://arxiv.org/pdf/2602.20492v1)

**作者:** Nuocheng Yang `[一作]` (Beijing University of Posts and Telecommunications), Changchuan Yin `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 7452 | [OpenAlex ID](https://openalex.org/A5009078493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在去中心化联邦学习环境下，提出一种稀疏且正交的LoRA框架，结合聚类通信拓扑和隐式专家（MoE）机制，实现多任务LLM的高效联合微调。

**💡 创新点**

创新点包括：①将LoRA的投影矩阵设为随机正交且静态，消除不同任务间更新冲突；②通过层级稀疏激活减少参数碰撞并压缩通信开销；③设计基于聚类的邻居拓扑提升聚合收敛速度；④利用任务感知编码实现隐式MoE，无需额外路由器，降低推理时的多任务干扰。

**🔧 技术方法**

技术方法主要包括：稀疏与正交LoRA、聚类通信拓扑（AGNES聚类）、隐式MoE（任务感知投影与top‑k激活）、OFDMA模型传输、参数碰撞率分析与优化。

**📊 数据集**

实验使用Qwen 2.5‑1.5B/7B‑Instruct LLM，并在四类多任务数据集上评估：NLI（BoolQ、Piqa、SocialIqa）、推理（GSM8K、ARC‑Easy、ARC‑Challenge）、代码（HumanEval、MBPP）以及综合能力（DollyTails、HellaSwag、ScienceQA）。

**📈 对比分析**

与LoRA、LoRI、Hard‑routing MoE、FPFT等基线相比，提出的方法在保持参数量仅占 0.5%–1% 的情况下，平均提升 5%（针对7B模型）或 1.3%（单聚类对比） 的测试准确率，并将通信量降低 73% 与参数传输量降低 86% 以上，展示了更优的计算/通信效率与性能平衡。

**⚠️ 局限性**

局限性包括：①需要设备间的聚类与邻居选择，聚类过程可能对网络拓扑和资源异质性敏感；②隐式MoE 的效果依赖于任务感知投影的质量，若任务分布不明显可能影响专家选择；③当前仅针对可微调场景，未涵盖无本地微调或推理协作的更通用联邦设置。

---

## 396. Localized Dynamics-Aware Domain Adaption for Off-Dynamics Offline Reinforcement Learning

**arXiv ID:** 2602.21072 | [PDF](https://arxiv.org/pdf/2602.21072v1)

**作者:** Zhangjie Xia `[一作]` (New York University), Pan Xu `[通讯]` (Duke University)

**通讯引用:** 6050 | [OpenAlex ID](https://openalex.org/A5100396257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对离线强化学习中源域与目标域动力学不匹配的问题，本文提出一种基于局部动态信息的离线域适应方法 LoDADA，能够在仅有少量目标数据与大量源数据的场景下有效学习目标域策略。

**💡 创新点**

创新点主要体现在：①将源目标转移动态差异按局部（聚类）划分，估计每个聚类内的 KL 散度来衡量动态偏差；②对低 KL 的聚类保留源样本，高 KL 聚类不全剔除而是降低权重，形成细粒度、可扩展的数据筛选；③在策略优化中加入对目标行为策略的正则化，使学习的策略与目标域行为保持一致。

**🔧 技术方法**

技术方法包括：K‑means 聚类对下一状态进行分组；构造聚类内二分类器估计源/目标分布差异；利用估计的 KL 散度进行数据筛选和加权；在 IQL（Implicit Q‑Learning）基础上加入行为克隆正则化和基于 KL 的 critic 加权。

**📊 数据集**

使用的主要数据集为：MuJoCo 离线数据集（HalfCheetah、Ant、Walker2d、Hopper）和 AntMaze 导航任务，以及 Adroit 操作任务；目标域数据量约 5k 条样本，源域数据量约 1M 条，全部来源于 D4RL。

**📈 对比分析**

与 DARA、BOSA、IQL、IGDF、OTDF 等基线对比，LoDADA 在 MuJoCo 全部 32 个任务中取得 21 项最佳、6 项次优，平均提升约 19.9% 以上；在局部扰动实验中平均提升 29.3% 以上；在 AntMaze 导航任务中总分 416.4，较第二名提升 8.4%。实验表明在全局与局部动力学偏移场景下均能显著优于现有方法。

**⚠️ 局限性**

局限性包括：①聚类数 K 和正则化强度 λ 需要经验调优；②聚类基于下一状态的 K‑means 可能无法捕捉更细致的动态结构；③方法仍依赖于目标域样本的覆盖范围，极端偏移或目标数据稀缺时效果可能受限；④理论上界仍有一定保守性，实际性能可能因假设不满足而下降。

---

## 397. cuRPQ: A High-Performance GPU-Based Framework for Processing Regular and Conjunctive Regular Path Queries

**arXiv ID:** 2602.20748 | [PDF](https://arxiv.org/pdf/2602.20748v1)

**作者:** Sungwoo Park `[一作]` (Korea Advanced Institute of Science and Technology), Min-Soo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5489 | [OpenAlex ID](https://openalex.org/A5100362662)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于GPU的高性能框架cuRPQ，用于处理常规路径查询(RPQ)及其扩展形式(共轭RPQ, CRPQ)；

**💡 创新点**

创新点包括：①跳跃限制的层级DFS遍历，兼顾DFS的低内存与BFS的路径优先；②按需段池技术管理访问集合，显著降低GPU内存占用；③GPU与CPU并行的探索-增量式材料化（BIM），实现高吞吐且可控制内存；④可支持WavePlan及多种CRPQ执行策略；

**🔧 技术方法**

核心技术：GPU原生DFS+LGF分块存储、段池（visited/ checkpoint/ bridge）、子TG拆分、并行线程块分配、异步D2H传输与CPU侧分块材料化、WCOJ联合CRPQ求解；

**📊 数据集**

使用LDBC SNB (SF=1,10) 与 StackOverflow (Span=6M,1Y) 两个真实/合成图数据集；

**📈 对比分析**

与CPU基线（DuckDB, Umbra, Ring-RPQ）及GPU相关库（RAPIDS, HeavyDB）比较，cuRPQ 在所有测试查询上均显著加速，最高可达 4,945×（algebra‑based）和 269×（automata‑based），能处理数万亿级结果，且在多GPU环境下几乎线性缩放；

**⚠️ 局限性**

局限性：需预先划分 LGF，GPU内存受限时需频繁切分子TG；对极大结果集的即时材料化仍会出现显存瓶颈；不支持持续流式 RPQ（如增量更新）；对路径长度约束的灵活性受静态 hop‑limit 影响。

---

## 398. Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training

**arXiv ID:** 2602.20656 | [PDF](https://arxiv.org/pdf/2602.20656v1)

**作者:** Guanbin Xu `[一作]` (University of Science and Technology of China), Cheng Li `[通讯]` (Anhui Province Key Laboratory of Biomedical Imaging and Intelligent Processing)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Lagom系统，在分布式LLM训练中自动共调通信参数以提升通信与计算的重叠效率。

**💡 创新点**

创新点在于统一重叠成本模型、SM与全局资源争用分析、优先级度量H以及线性搜索算法，解决了计算瓶颈下通信争用的优化难题。

**🔧 技术方法**

采用统一成本模型、SM和全局资源争用建模、优先级度量、基于AutoCCL的分治搜索和线性迭代算法。

**📊 数据集**

在Phi‑2‑2B、Llama‑3‑8B、MPT‑7B、DeepSeek‑MoE‑16B、OLMoE‑1B‑7B等大模型上进行评估。

**📈 对比分析**

与NCCL v2.18.3-1和AutoCCL对比，Lagom在NVLink集群上获得1.10–1.33×加速，在PCIe集群上获得1.08–1.16×加速，显著优于两者。

**⚠️ 局限性**

局限在于仅调优通信参数，未针对计算侧进行优化；对非NCCL框架或不同GPU架构的适用性未知；模型假设在极端场景下可能失效。

---

## 399. FLIM Networks with Bag of Feature Points

**arXiv ID:** 2602.20845 | [PDF](https://arxiv.org/pdf/2602.20845v1)

**作者:** João Deltregia Martinelli `[一作]` (Institute of Computing UNICAMP), Alexandre X. Falcão `[通讯]` (Institute of Computing UNICAMP)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于FLIM的轻量级显著目标检测网络FLIM‑BoFP，专门用于显微镜图像中寄生虫卵与囊的检测。

**💡 创新点**

创新点是将原本每层单独聚类估计过滤器的方法改为一次聚类生成Bag of Feature Points，并直接从这些点映射并生成带偏置的过滤器，显著提升了速度、参数占用与可解释性。

**🔧 技术方法**

采用FLIM框架、K‑means聚类、z‑score标准化、基于特征点的直接过滤器估计、适应性解码器以及动态树后处理算法。

**📊 数据集**

实验数据集包括公开的Schistosoma Mansoni（1219张RGB图），私有的Entamoeba histolytica（395张）和Ancylostoma spp.（320张）三种寄生虫显微图像。

**📈 对比分析**

与FLIM‑Cluster、HVPNet、SAMNet、SeaNet、U2‑Net等基线进行对比，FLIM‑BoFP在F‑score、MAE、wF等指标上均优于所有对手，参数量仅为传统模型的3%且在零样本迁移任务中展现出更强的泛化能力。

**⚠️ 局限性**

局限性包括对用户手绘标记的依赖、在极少量训练图像下仍可能出现过拟合风险，以及在更复杂场景或多类目标时对聚类数量与位置的敏感度需进一步研究。

---

## 400. Scaling Vision Transformers: Evaluating DeepSpeed for Image-Centric Workloads

**arXiv ID:** 2602.21081 | [PDF](https://arxiv.org/pdf/2602.21081v1)

**作者:** Huy Trinh `[一作]` (University of Waterloo), Tahsin Reza `[通讯]` (University of Waterloo)

**通讯引用:** 151 | [OpenAlex ID](https://openalex.org/A5043238877)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用 DeepSpeed 进行 Vision Transformer 的分布式训练，评估其在不同 GPU 集群（Nebula、Tesla、Vector）中的 intra‑node 与 inter‑node 可扩展性，并研究批量大小、梯度累积等软件参数对训练效率和准确率的影响。

**💡 创新点**

将原本主要用于大规模语言模型的 DeepSpeed 框架迁移到视觉任务，首次系统性比较了在视觉模型上的强/弱缩放表现；发现批量大小 64/128 在同步开销与显存利用率之间取得最佳平衡；揭示了 GPU 设备异质性对多机训练的严重制约。

**🔧 技术方法**

使用 DeepSpeed + NCCL + MPI 进行数据并行训练；应用 ZeRO‑0（未使用 ZeRO‑Infinity）以降低显存冗余；采用梯度累积、学习率调度、AllReduce 等常见技术。

**📊 数据集**

主要使用 CIFAR‑10、CIFAR‑100 两个 32×32 分辨率数据集，尝试在 ImageNet‑100（224×224）上训练但未完成；所有实验均以 5 轮训练为基准。

**📈 对比分析**

通过强/弱缩放曲线、每 GPU 训练时间、通信占比和准确率变化进行比较；在 Vector 集群上 32 节点单 GPU 与单节点多 GPU 的性能相当；Tesla 集群因 GPU 异构导致强缩放表现不佳；最佳批量大小为 64/128，可实现接近理论理想的加速比例。

**⚠️ 局限性**

受限于 GPU 设备异质性、显存不足、仅使用数据并行（未探究 ZeRO、模型并行等）；只在小规模、低分辨率数据集上验证，缺乏对大规模视觉模型和更高分辨率图像的评估；未与其他分布式训练框架（如 Megatron‑LM、Accelerate）进行对标。

---

## 401. VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving

**arXiv ID:** 2602.20794 | [PDF](https://arxiv.org/pdf/2602.20794v1)

**作者:** Jie Wang `[一作]` (Tianjin University), Long Chen `[通讯]` (Xiaomi EV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过引入可插拔的跨视角3D几何启用模块（CVGE），利用冻结的视觉3D基座VGGT提取跨视角3D特征，并通过层级自适应注入机制深度融合进Vision‑Language模型，使其获得跨视角几何感知能力，进而统一完成风险感知、运动预测、轨迹规划等自动驾驶任务。

**💡 创新点**

创新点在于（1）首次将成熟的3D视觉基座VGGT与VLM无缝耦合；（2）提出层级自适应注入机制（CVGE），利用多头跨模态注意力实现2D视觉与3D几何特征的深度、可插拔融合；（3）通过双阶段微调，仅训练注入模块即可显著提升VLM性能，避免了传统Q&A或独立动作解码的耦合和信息流断裂。

**🔧 技术方法**

核心技术包括：Qwen2.5‑VL 7B作为基底VLM；冻结的VGGT作为3D专家提取跨视角3D特征；CVGE模块实现多层次、可自适应的跨模态注意力注入；双阶段微调（首阶段仅训练CVGE，次阶段微调VLM+CVGE）；使用MLP降维、上升以及残差注入实现特征匹配。

**📊 数据集**

在五个主流自动驾驶基准集上进行评测：NuInstruct、DriveLM、OmniDrive、NuScenes‑Plan以及NAVSIM。

**📈 对比分析**

与基线VLM、VGGT‑Dist/​Add、现有VLA及E2E方法对比，VGGDrive在跨视角风险感知（MAP）提升约31%/30%，轨迹规划闭环PDMS升至88.76（相较基线提升约2.7），碰撞率降低至2.27%，整体平均提升约10%–30%，在多任务上表现出显著优势。

**⚠️ 局限性**

主要局限包括：对VGGT等3D基座模型的高度依赖，需保证其在目标场景中的适配；训练仍需要大规模GPU资源，且目前仅针对多视角摄像头设置，未充分考虑极端光照、传感器失效或多模态融合的情况。

---

## 402. Codified Context: Infrastructure for AI Agents in a Complex Codebase

**arXiv ID:** 2602.20478 | [PDF](https://arxiv.org/pdf/2602.20478v1)

**作者:** Aristidis Vasilopoulos `[一作]` `[通讯]` (Independent Researcher), Aristidis Vasilopoulos (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发并验证了一套三层结构的 codified context 基础设施，利用机器可读的项目规范与专属领域代理，使 LLM 代码助手在构建 108,000 行 C# 分布式系统时能够跨会话保持记忆与一致性。

**💡 创新点**

创新点在于：①将项目知识拆分为热内存（Constitution）、专属代理与冷内存（知识库）三层，并通过触发表格与 MCP 检索服务实现自动化路由；②将文档视为基础设施而非单纯注释；③通过多级加载与嵌入式域知识显著降低跨会话错误。

**🔧 技术方法**

技术手段包括：Claude Code LLM 编码代理；Model Context Protocol（MCP）检索服务器（Python）；触发表格与专属代理规范；工厂代理用于快速初始化；以及一套交互日志解析脚本。

**📊 数据集**

使用数据集：108,000 行 C# 代码；19 个专属代理规范（约 9,300 行）；34 个知识文档（约 16,250 行）；283 个开发会话；148 次提交；2,801 人类提示；1,197 代理调用；16,522 代理回合。

**📈 对比分析**

评估方法：量化指标（代码行数、上下文文件比例、交互次数、代理调用比例）以及四个观察案例。虽然未做对照实验，但数据显示：提示平均长度约 100 词，维护成本 1–2 小时/周，且系统在 74 次保存相关会话中未出现错误，显示跨会话一致性得到显著提升。

**⚠️ 局限性**

局限性：仅在单人单项目环境下评估；缺乏对照实验以量化性能提升；实现依赖 Claude Code 与 MCP，其他平台可移植性待验证；语义检索未实现，仍用关键字匹配；文档维护与漂移检测需人工监督。

---

## 403. MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation

**arXiv ID:** 2602.20423 | [PDF](https://arxiv.org/pdf/2602.20423v1)

**作者:** Taha Koleilat `[一作]` (Concordia University), Hassan Rivaz `[通讯]` (Concordia University)

**通讯引用:** 3567 | [OpenAlex ID](https://openalex.org/A5077743201)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

基于CLIP的概率化视觉-语言适配器，用文本提示驱动医学图像分割。

**💡 创新点**

创新点在于引入概率跨模态注意力、软patch级对比损失，实现不确定性感知、数据高效和域迁移鲁棒。

**🔧 技术方法**

使用UniMedCLIP ViT‑B/16与PubMedBERT作为基础网络，加入概率交叉模态注意力、Monte Carlo采样、soft contrastive loss和轻量级分割头。

**📊 数据集**

在16个医学影像数据集（如BUSI、BTMRI、ISIC、Kvasir‑SEG、QaTa‑COV19、EUS等）上进行评估。

**📈 对比分析**

与SOTA基线（如CAT‑Seg、SAN、LAVT等）对比，平均提升2–4% DSC，域外性能显著提升，并生成可靠的像素级不确定性图。

**⚠️ 局限性**

局限在于需要丰富的文本提示、Monte Carlo采样导致推理时间增加，以及对极端域外场景仍存在性能下降。

---

## 404. CAMEL: Confidence-Gated Reflection for Reward Modeling

**arXiv ID:** 2602.20670 | [PDF](https://arxiv.org/pdf/2602.20670v1)

**作者:** Zirui Zhu `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**通讯引用:** 3859 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为CAMEL的奖励模型框架，通过在单个令牌的初步判断与可选的反思步骤之间做出决策，利用置信度门控来决定是否进行昂贵的生成反思，从而提升对话模型的偏好判断准确率。

**💡 创新点**

创新点在于：①发现单词级概率差（log‑probability margin）与判断正确性高度相关，可作为实例难度的无成本置信度估计；②设计双决策提示，首次输出单一预判令牌后根据置信度决定是否调用反思；③利用强化学习与对抗性前缀增广，训练模型在低置信度场景下自我纠正，且不需要额外标签。

**🔧 技术方法**

核心技术包括：置信度门控机制、两阶段生成提示（初始判定 + 反思），强化学习策略优化（GRPO），对抗性前缀增广，基于预训练大型语言模型（Qwen3‑14B）进行SFT + GRPO。

**📊 数据集**

使用的偏好学习数据集为Skywork Reward Preference 80K、Code‑Preference‑Pairs 与 Math‑Step‑DPO‑10K，评估基准为RewardBench、RM‑Bench 与 JudgeBench。

**📈 对比分析**

与多种标杆（包括大型 70B 参数模型与强生成奖励模型）对比，CAMEL 在三个基准上平均准确率达到82.9%，比第二佳模型高3.2%，并在 14B 参数规模下实现更优的准确‑成本 Pareto 前沿；CAMEL‑Fast 单令牌版在速度端与大型基准相当或更好，CAMEL‑Reflection 全反思版则在准确率上进一步提升。

**⚠️ 局限性**

限制点包括：对置信度阈值的手动调优需要经验；模型在训练后整体置信度趋于保守，可能导致过度触发反思；仅在固定任务（偏好判定）上验证，尚未在更广泛的对话或多任务场景中测试。

---

## 405. The Initial Exploration Problem in Knowledge Graph Exploration

**arXiv ID:** 2602.21066 | [PDF](https://arxiv.org/pdf/2602.21066v1)

**作者:** Claire McNamara `[一作]` (Trinity), Declan O'Sullivan `[通讯]` (Trinity)

**通讯引用:** 2865 | [OpenAlex ID](https://openalex.org/A5020871962)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“首次探索问题”（Initial Exploration Problem, IEP）的概念，并将其分解为范围不确定性、本体不透明性和查询无能三个互相关联的障碍；

**💡 创新点**

首次将这些障碍聚焦在首次接触知识图谱的时点上，形成了一个统一的、时间限定的理论框架，指出了传统探索界面缺失的“范围揭示”交互原语；

**🔧 技术方法**

采用信息行为理论、认知负荷理论、信息猎捕理论等跨学科框架进行概念化与归纳，未实现具体实现技术；

**📊 数据集**

未使用任何特定数据集，本文以DBpedia、VRTI、FAIRVASC等已有知识图谱为案例说明；

**📈 对比分析**

文章未开展实验或性能对比，仅通过文献综述与案例分析阐述缺口与设计启示；

**⚠️ 局限性**

主要局限在缺乏实证验证：未对不同界面原语或干预方法进行实验评估，无法量化对首次探索障碍的缓解效果。

---

## 406. LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding

**arXiv ID:** 2602.20913 | [PDF](https://arxiv.org/pdf/2602.20913v1)

**作者:** Jihao Qiu `[一作]` (University of Chinese Academy of Sciences), Qixiang Ye `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 15135 | [OpenAlex ID](https://openalex.org/A5015317495)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于主动推理的多模态大语言模型代理LongVideo-R1，能够在低计算预算下高效理解长视频并回答问题。

**💡 创新点**

核心创新点包括：①将视频组织为多层树结构，支持自适应的全局-局部跳转；②设计链式思维+工具调用框架（CoTwT），实现动态决定下一个要采样的片段；③通过大规模手工标注（33K条）和强化学习（GRPO）训练，使模型在保持高准确率的同时显著减少视频采样次数和推理成本。

**🔧 技术方法**

技术手段：多模态大型语言模型（Qwen-3-8B）+工具调用（视频字幕、视频问答工具）、层次化视频分块、链式思维与工具交互、监督微调+强化学习（GRPO）以及自定义奖励函数（答题、定位、重复惩罚）。

**📊 数据集**

使用的数据集：①CG-Bench（1.2K长视频，5.6K QA）用于生成CoTwT轨迹；②LVBench、Video-MME-long、MLVU三大长视频问答基准用于评估；③用于工具的预训练模型（Qwen2.5-VL-72B、Qwen2.5-VL-32B）提供视频字幕与问答能力。

**📈 对比分析**

与现有代理系统（Ego-R1、VideoTree等）以及开源/专有大型语言模型相比，LongVideo-R1在LVBench上实现50%+准确率，KIR/TG子任务分别达56%/56%，在Video-MME上达到64%，同时平均每个问题仅需约10–12次工具调用（≈3分钟推理），显著降低计算成本；在计算效率上，采用更强字幕模型可进一步提升性能。

**⚠️ 局限性**

局限性：①依赖高质量的视频字幕工具，字幕质量直接影响推理效果；②仅使用两种工具，缺乏更丰富的感知能力；③在面向多问题的视频时，仍未充分利用共享信息；④模型有时会被语义相关但不相关片段误导，需要外部提示才能纠正；⑤对超长视频（数小时）仍需进一步优化导航策略。

---

## 407. PropFly: Learning to Propagate via On-the-Fly Supervision from Pre-trained Video Diffusion Models

**arXiv ID:** 2602.20583 | [PDF](https://arxiv.org/pdf/2602.20583v1)

**作者:** Wonyong Seo `[一作]` (KAIST), Munchurl Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为PropFly的训练框架，利用预训练视频扩散模型的 on-the-fly 监督实现传播式视频编辑，无需成对数据。

**💡 创新点**

创新点在于通过变换 CFG 尺度一次性生成结构相同但语义不同的源/目标潜在对，实现高效监督，并引入 Guidance‑Modulated Flow Matching (GMFM) 损失和 Random Style Prompt Fusion。

**🔧 技术方法**

使用预训练的 Wan2.1 视频流匹配模型、VACE 适配器、CFG 调节的一步清晰潜在估计以及 GMFM 损失进行训练。

**📊 数据集**

数据集包括 Youtube‑VOS 与 Pexels 的 3000 条视频，并使用 Qwen2.5‑VL 生成字幕。

**📈 对比分析**

与现有文本引导与传播式编辑方法（如 AnyV2V、Señorita‑2M、STDF、TokenFlow）进行定量评估，在 EditVerseBench‑Appearance 和 TGVE 基准上在视频质量、文本对齐和时序一致性上均实现 SOTA。

**⚠️ 局限性**

限制在于对 CFG 尺度的依赖、对单帧编辑的依赖以及在极端复杂场景下细节保留可能不足。

---

## 408. Actor-Curator: Co-adaptive Curriculum Learning via Policy-Improvement Bandits for RL Post-Training

**arXiv ID:** 2602.20532 | [PDF](https://arxiv.org/pdf/2602.20532v1)

**作者:** Zhengyao Gu `[一作]` (University of Illinois Chicago), Yisong Yue `[通讯]` (Caltech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种可自动化、可扩展的强化学习后训练框架，利用神经策展人根据策略提升目标自适应选择训练问题，并通过在线随机镜面下降（OSMD）在非平稳Bandit环境中进行优化。

**💡 创新点**

创新点包括：①将问题选择建模为非平稳随机Bandit，推导基于策略提升的目标并给出OSMD的动态风险界；②使用PPO风格的近端剪切目标稳定神经策展人的训练；③实现两阶段采样方案，既保证覆盖又实现高效自适应。

**🔧 技术方法**

采用的技术包括：强化学习后训练（GRPO/GSPO等）、政策提升理论、在线随机镜面下降（OSMD）、PPO近端剪切、神经网络策展人以及两阶段采样。

**📊 数据集**

使用的数据集有：Countdown、Zebra、ARC‑1D、MATH500、AIME2024，以及对应的“hard”子集。

**📈 对比分析**

与统一采样、SEC、PCL等基线对比，本文方法在所有基准上均超越基线，ARC‑1D最高提升约30%（相对），AIME24提升约28%；训练速度提升可达80%。

**⚠️ 局限性**

局限性包括：对actor更新的稳定性高度依赖；需要可靠的奖励信号，限制了可用于非可验证奖励领域的适用性；额外的策展网络开销约为整体训练时间的9%。

---

## 409. AWCP: A Workspace Delegation Protocol for Deep-Engagement Collaboration across Remote Agents

**arXiv ID:** 2602.20493 | [PDF](https://arxiv.org/pdf/2602.20493v1)

**作者:** Xiaohang Nie `[一作]` (Harbin Institute of Technology), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18211 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Agent Workspace Collaboration Protocol（AWCP），通过临时工作空间委托实现代理之间的深度协作，弥补了传统基于消息传递的上下文缺失问题。

**💡 创新点**

创新点在于把文件系统视为通用接口，实现工作空间投影和可插拔传输层，解耦控制平面与具体传输实现，使代理能在对方环境中直接使用本地工具链。

**🔧 技术方法**

采用了 HTTP + SSE 控制层、四种传输适配器（SSHFS、Archive、Storage、Git）、双状态机协议、TypeScript 开源实现，并与 MCP、A2A、ANP 等现有协议无缝集成。

**📊 数据集**

实验主要使用真实场景数据：包含 100+ 图像的目录进行跨模态整理，以及单个 PDF 合同进行合规盖章，未使用公开标准数据集。

**📈 对比分析**

通过与传统基于消息的 A2A、ANP 的对比，演示了实时同步、减少上下文丢失的优势；在案例中实现了低延迟交互与高效文件同步，具体性能指标未给出，但实验证明相较于消息传递方案更高效、准确。

**⚠️ 局限性**

局限性包括：仅支持一对一委托，缺乏多方并发冲突处理；权限与审计细粒度不足；传输适配器受限，未覆盖 P2P 或 CRDT 等方案；未做大规模性能与安全评估。

---

## 410. A Space-space Trade-off for Directed st-Connectivity

**arXiv ID:** 2602.21088 | [PDF](https://arxiv.org/pdf/2602.21088v1)

**作者:** Roman Edenhofer `[一作]` `[通讯]` (University Paris Cite), Roman Edenhofer (University Paris Cite)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了在催化空间模型下有向 st 连通性问题的空间-空间权衡。

**💡 创新点**

提出了一种可插值的算法，能够在 O(log n·log k + log n) 正常工作空间和 O((n/k)·log² n) 催化记忆的条件下求解长度为 ℓ 的路径计数，从而在 Savitch 的 O(log² n) 经典空间和 Cook‑Pyne 的 O(n·log² n) 催化记忆之间平滑过渡。

**🔧 技术方法**

主要技术包括：1）将 Savitch 的递归分治思想改写为可逆寄存器程序，实现对长度 ℓ 步的 walk 信息的“传播”；2）将传统的顶点序列枚举改为残差类序列枚举，并利用残差划分将催化记忆压缩至 O(n/k)；3）利用 Chinese Remainder 以及已有的 logspace 余数到二进制转换技术，得到精确计数。

**📊 数据集**

本文为理论论文，无实验数据集。

**📈 对比分析**

与先前的 Savitch（O(log² n) 经典空间）和 Cook‑Pyne（O(n·log² n) 催化记忆）相比，提出的算法在参数 k 变化时可以获得更优的空间-催化记忆平衡；在 k=1 时恢复了 O(log n) 经典空间和 O(n·log² n) 催化记忆的上界，在 k≈2^{O(√log n)} 时可实现 O(log n) 经典空间和 O(n/2^{Θ(√log n)}·log² n) 催化记忆。

**⚠️ 局限性**

当前的技术仍未突破真正的子线性空间（O(n^{1-ε})) 解决方案；Catalytic 内存仍以多项式上界为上限；算法主要关注空间效率，时间复杂度并未得到优化，且依赖于对质数序列的生成与 Chinese Remainder 的运算。

---

## 411. CrystaL: Spontaneous Emergence of Visual Latents in MLLMs

**arXiv ID:** 2602.20980 | [PDF](https://arxiv.org/pdf/2602.20980v1)

**作者:** Yang Zhang `[一作]` (Nankai University), Xiang Li `[通讯]` (Nankai International Advanced Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单阶段多模态大语言模型训练框架 CrystaL，利用双路径自监督实现视觉潜在推理。

**💡 创新点**

通过完整图像与腐败图像双路径的一致性约束，在不使用外部模块或额外图像标签的情况下，将视觉潜在表示“结晶”为任务相关视觉语义，从而解决传统潜在CoT监督与推理不匹配的问题。

**🔧 技术方法**

结合 Latent Chain-of-Thought、随机图像损坏（Stochastic Image Corruption）、双路径对齐损失（KL 与注意力对齐）、LoRA 微调等技术实现自监督训练。

**📊 数据集**

在多模态对话/问答数据上训练，并在 CVBench（2D/3D）、HRBench（4K/8K）、VStarBench、BLINK、RWQA、POPE 等视觉理解与推理基准上进行评估。

**📈 对比分析**

与 CoVT、LIVR、SKILA、Vision‑R1 等基线比较，CrystaL 在各基准上平均 75.4% 得分，2D CVBench 76.6%、3D 84.4%、4K HRBench 73.4%、8K 71.1%，均优于对照模型，并显示更高的数据效率与鲁棒性。

**⚠️ 局限性**

对极端视觉噪声的鲁棒性有限，对推理链的可解释性不强，且在某些细粒度视觉细节任务中提升有限。

---

## 412. Sparse Bayesian Deep Functional Learning with Structured Region Selection

**arXiv ID:** 2602.20651 | [PDF](https://arxiv.org/pdf/2602.20651v1)

**作者:** Xiaoxian Zhu `[一作]` (School of Statistics and Data Science), Mengyun Wu `[通讯]` (School of Statistics and Data Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出稀疏贝叶斯深度函数网络（sBayFDNN），实现功能预测与可解释区域选择的统一模型；

**💡 创新点**

首次将结构化稀疏先验与深度网络相结合，实现对函数域内局部活跃区的自动识别，并给出理论保证；

**🔧 技术方法**

基于B样条基展开的功能嵌入、深度ReLU网络、群组连续尖峰-斑点先验、MAP后验插值以及PIP阈值化；

**📊 数据集**

仿真数据、ECG、Tecator、Bike Rental、IHPC等公开时序/光谱/功率曲线数据集；

**📈 对比分析**

与五种竞争方法（FNN、AdaFNN、cFuSIM、BFRS、SLoS）对比，sBayFDNN在预测RMSE/MAE方面表现最佳，且在区域识别上取得最高Recall/F1；

**⚠️ 局限性**

对单一功能预测的扩展受限，未处理多功能或离散协变量；在高噪声/极端非线性场景下仍可能出现选择误差；

---

## 413. BFA++: Hierarchical Best-Feature-Aware Token Prune for Multi-View Vision Language Action Model

**arXiv ID:** 2602.20566 | [PDF](https://arxiv.org/pdf/2602.20566v1)

**作者:** Haosheng Li `[一作]` (Institute of Software), Hua Chen `[通讯]` (Zhejiang University)

**通讯引用:** 26985 | [OpenAlex ID](https://openalex.org/A5100380230)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现BFA++，一种针对多视角视觉‑语言‑动作模型的动态多级token剪枝框架，减少冗余视觉信息、提升推理速度与抓取/操作成功率。

**💡 创新点**

双层重要性预测（视角级与视图内级）结合层次化剪枝和空间自适应加权；离线注释系统生成任务相关重要性标签；在VLA后训练阶段联合优化。

**🔧 技术方法**

使用视觉‑语言模型π_0与RDT、Transformer、动态Token Pruning、两级重要性预测网络、交叉熵损失、t‑SNE与Grad‑CAM可视化等技术。

**📊 数据集**

RoboTwin benchmark（含七个模拟任务与OOD任务）、真实双臂实验（5个任务、每个200条），以及模拟Bottle Pick等任务集。

**📈 对比分析**

与π_0、RDT基线及BFA、DART剪枝对比；在RoboTwin平均成功率提升约10%，推理速度提升1.5–1.8×；在真实环境中成功率提升约10%，FPS提升0.5–1.8×；在OOV任务中亦保持优势。

**⚠️ 局限性**

依赖离线重要性注释，重要性预测器对未见物体或摄像头配置的泛化有限；高剪枝率可能导致性能下降。

---

## 414. $κ$-Explorer: A Unified Framework for Active Model Estimation in MDPs

**arXiv ID:** 2602.20404 | [PDF](https://arxiv.org/pdf/2602.20404v1)

**作者:** Xihe Gu `[一作]` (University of California San Diego), Tara Javidi `[通讯]` (University of California San Diego)

**通讯引用:** 8833 | [OpenAlex ID](https://openalex.org/A5059310658)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于参数化 U_κ 目标的主动探索算法 κ-Explorer，用以在有限采样预算下估计 MDP 的转移模型。

**💡 创新点**

创新点在于统一构造一个可调曲率 κ 的凸凹目标，既能连续插值平均误差与极大误差，又通过 Frank–Wolfe 优化实现理论保证与实践效率。

**🔧 技术方法**

使用了 Frank–Wolfe（凸优化）与动态规划、经验估计、平滑 U_κ 梯度等技术，并给出在线版的 DP 近似。

**📊 数据集**

在离散化的 Pendulum 和 Mountain Car 两个经典 Gymnasium 环境上进行实验。

**📈 对比分析**

与随机、MaxEnt、Weighted-MaxEnt、SMM 等基线比较，κ-Explorer 在失败率、平均与最差估计误差上均优于其它策略，尤其在 κ 较大时获得更好的 worst‑case 性能。

**⚠️ 局限性**

局限性包括对大规模状态动作空间的计算复杂度、需调节 κ 以权衡平均/极大误差，以及实验仅在离散化的两种环境验证。

---

## 415. SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking

**arXiv ID:** 2602.20792 | [PDF](https://arxiv.org/pdf/2602.20792v1)

**作者:** Muhammad Saif Ullah Khan `[一作]` (German Research Center for Artificial Intelligence), Didier Stricker `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将现有全身姿态数据集与基于肌肉骨骼模型的解剖一致的3D脊柱关键点进行融合，生成了首个开放的含脊柱3D标注的数据集并提供预训练基线；

**💡 创新点**

创新点在于：①提出一套从多视角RGB数据到解剖一致3D脊柱关键点的完整仿真与注释流水线；②利用肌肉骨骼模型对脊柱进行全尺度、动力学可行的姿态生成；③首次在大规模计算机视觉任务中引入脊柱生物力学约束，实现可在自然环境下进行3D脊柱运动估计；

**🔧 技术方法**

使用技术包括：多视角相机标定与三角化、OpenSim逆运动学（IK）与正运动学（FK）、虚拟脊柱标记生成、数据增强与混合训练、深度检测网络（SpinePose、HRNet、RTMPose、ViTPose）和3D姿态上采样与投影；

**📊 数据集**

使用数据集为改造后的Human3.6M（室内多视角）以及新生成的含脊柱3D标注的2.14M帧数据集（称为“SpineTrack”），并在此基础上进行基线训练与评估；

**📈 对比分析**

方法对比：在2D检测任务中，Fine‑tuned SpinePose提升脊柱AUC从0.63到0.80，AP从0.91到0.93；在多视角3D重建任务中，基于GT 2D的P‑MPJPE可达0.67mm，使用检测器时约为26‑40mm；在单目3D提升任务中，全身标记训练相较于仅脊柱标记可将P‑MPJPE从约18mm降至约13mm；

**⚠️ 局限性**

局限性包括：①脊柱模型仅对腰椎五节进行关节耦合，颈椎与胸椎被视为刚体，忽略肋骨与软组织约束；②未模拟椎间位移、肌肉力学与地面反作用，导致运动仅满足几何可行性；③数据来源局限于室内固定相机与Human3.6M的动作集合，缺乏户外、病理及多姿态多样性；④所有标注均为仿真生成，缺乏真实体内验证。

---

## 416. POMDPPlanners: Open-Source Package for POMDP Planning

**arXiv ID:** 2602.20810 | [PDF](https://arxiv.org/pdf/2602.20810v1)

**作者:** Yaacov Pariente `[一作]` (Technion Institute of Technology), Vadim Indelman `[通讯]` (Technion Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

开发了 POMDPPlanners，一个面向 POMDP 规划的开源 Python 包，整合最新规划算法、带安全指标的基准环境、自动超参数搜索、持久缓存与并行仿真，支持可复现的大规模实验。

**💡 创新点**

提供统一可扩展的框架，实现自动化超参数优化、任务管理与持久缓存，支持连续与离散空间，加入风险敏感规划器及安全评估指标，填补现有 Python 工具缺失的功能。

**🔧 技术方法**

采用 Python 3.10+、Optuna 超参搜索、Joblib/Dask/PBS 并行后端、MLflow 记录实验、粒子滤波/高斯/高斯混合信念表示、MCTS（POMCP, POMCPOW, PFT-DPW 等）和风险敏感算法（ICVaR 等）。

**📊 数据集**

包含 9 个基准环境（Tiger、LightDark、RockSample、CartPole、MountainCar、Push、LaserTag、SafetyAnt、PacMan），其中 LightDark 与 LaserTag 提供离散与连续版本，并加入危险区惩罚。

**📈 对比分析**

通过直接评估与优化评估两种工作流，使用 Optuna 搜索最优超参后运行多条 episode，记录平均回报、CVaR、VaR、目标率、安全指标；实验表明在安全关键环境中风险敏感规划器能够显著降低违规率，同时保持较高回报。

**⚠️ 局限性**

目前仅支持离散或连续（混合）空间，尚未充分验证高维连续环境的可扩展性；缺乏对多目标或持续时间约束的完整支持，安全评估仍需手工设定阈值，未实现动态安全约束学习。

---

## 417. A Generalized Apprenticeship Learning Framework for Capturing Evolving Student Pedagogical Strategies

**arXiv ID:** 2602.20527 | [PDF](https://arxiv.org/pdf/2602.20527v1)

**作者:** Md Mirajul Islam `[一作]` (North Carolina State University), Min Chi `[通讯]` (North Carolina State University)

**通讯引用:** 2124 | [OpenAlex ID](https://openalex.org/A5090231772)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并评估一个名为 THEMES 的时间感知分层 Apprenticeship Learning 框架，用于从专家学生演示中学习随时间演变的教学策略。

**💡 创新点**

首次结合子轨迹分割、时间感知奖励调节与 EM-EDM，实现离线多奖励函数随时间演化的学习，并将奖励调节器嵌入分层分割。

**🔧 技术方法**

采用子轨迹分割、EM‑EDM、最大似然逆强化学习、时间衰减函数以及 t‑SNE 可视化；在离线设置下对多奖励函数进行聚类与策略诱导。

**📊 数据集**

来自一门本科概率课程的 221 名学生在 4 学期的 ITS 交互轨迹（共 89 条专家轨迹）。

**📈 对比分析**

与 6 个竞争性 AL 基线及 2 个消融模型进行 3‑折跨学期交叉验证；THEMES 在 AUC、Jaccard 等指标上均显著优于基线，最高 AUC 0.899、Jaccard 0.653，仅用 18 条轨迹即可预测后学期策略。

**⚠️ 局限性**

仅在单一 ITS 环境验证，缺乏对多样化学习场景的泛化分析；对演化奖励函数的解释仍有限；轨迹长度相对较短。

---

## 418. Benchmarking GNN Models on Molecular Regression Tasks with CKA-Based Representation Analysis

**arXiv ID:** 2602.20573 | [PDF](https://arxiv.org/pdf/2602.20573v1)

**作者:** Rajan `[一作]` (Indian Institute Of Technology Delhi), Ishaan Gupta `[通讯]` (Indian Institute Of Technology Delhi)

**通讯引用:** 2411 | [OpenAlex ID](https://openalex.org/A5025356846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对单层 GNN、传统指纹基线和融合 GNN+FP 在四个化学回归数据集上的性能与表示相似性进行系统对比与分析。

**💡 创新点**

提出层次融合框架将 GNN 的图层嵌入与 1024-bit ECFP4 指纹进行拼接，并用 CKA 评估不同模型及 GNN 与指纹之间的表示相似度。

**🔧 技术方法**

采用单层 GCN、GAT、GIN、GraphSAGE 网络、全连接回归头、CKA（RBF kernel）与经典机器学习基线（LR、SVM、RF、XGB）。

**📊 数据集**

四个数据集：ESOL、Lipophilicity、Retention Time (RT) 与 B3DB。

**📈 对比分析**

通过 RMSE 及 95% 置信区间对比；融合模型平均提升 22–26% 的 RMSE，单层 GNN 与基线相差 17–27%，融合后可超越基线。

**⚠️ 局限性**

单层 GNN 在数据量有限时性能不足；对特征维度依赖低，需更深网络；数据集规模限制了模型学习复杂层次的能力。

---

## 419. UDVideoQA: A Traffic Video Question Answering Dataset for Multi-Object Spatio-Temporal Reasoning in Urban Dynamics

**arXiv ID:** 2602.21137 | [PDF](https://arxiv.org/pdf/2602.21137v1)

**作者:** Joseph Raj Vishal `[一作]` (Arizona State University), Bharatesh Chakravarthi `[通讯]` (Arizona State University)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5083090349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了UDVideoQA和VideoQGen两个基准，评估视频语言模型在真实城市交通视频中的多层次推理能力。

**💡 创新点**

创新点在于：1）16小时高密度城市交通视频与28,800 QA对的公开数据集；2）事件驱动动态模糊实现隐私保护而不破坏场景；3）层次化的五级推理税表和半自动注释工具；4）首创的VideoQGen问句生成评测。

**🔧 技术方法**

使用技术包括视频分割、事件驱动动态模糊、半自动 QA 生成与人工审核、LoRA 微调、LLM 判别与加权评分。

**📊 数据集**

数据集：16小时 30fps 交叉路口视频 1.7M 帧，涵盖多时段、多天气与灯光条件；共28,800问答，按 Attribution、Basic Understanding、Event Reasoning、Reverse Reasoning、Counterfactual Inference 5类划分。

**📈 对比分析**

比较方法：对10款SOTA VideoLM进行零射击和微调实验，利用LLM判别+加权评分；Gemini 2.5 Pro在零射击上最高（≈75%），但定位性能低；微调后的Qwen 2.5‑VL 7B在低光照和高密度场景中超过大多数专有模型，性能可与商用系统相当。

**⚠️ 局限性**

局限性：模型在视觉定位（Attribution）上表现欠佳，时序因果推理（Event/Reverse）难度大；复杂多智能体交互导致高频事件对模型视觉细粒度识别产生干扰；当前评测依赖LLM判别，仍可能对非文本推理产生偏差。

---

## 420. SPP-SCL: Semi-Push-Pull Supervised Contrastive Learning for Image-Text Sentiment Analysis and Beyond

**arXiv ID:** 2602.20767 | [PDF](https://arxiv.org/pdf/2602.20767v1)

**作者:** Jiesheng Wu `[一作]` (Anhui Normal University), Shengrong Li `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 3618 | [OpenAlex ID](https://openalex.org/A5112402443)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种半推拉监督对比学习（SPP‑SCL）方法，在融合前通过两步策略平衡视觉‑文本的情感关系；

**💡 创新点**

创新点在于：①先利用内部监督对比学习拉近同模态之间的关系；②通过条件执行语句判断后，若不满足则执行跨模态监督对比学习将跨模态关系推开，实现内外模态关系的一致性与平衡；

**🔧 技术方法**

采用监督对比学习技术，结合条件执行逻辑和跨模态特征融合；

**📊 数据集**

使用三大公开图像‑文本情感与讽刺检测数据集；

**📈 对比分析**

与最新SOTA方法对比，SPP‑SCL在所有数据集上显著提升性能，表现出更强的情感判别能力；

**⚠️ 局限性**

局限性：需要两步训练和条件判断，可能增加训练成本；在跨模态数据分布极度不平衡时效果尚未验证。

---

## 421. Vision-Based Reasoning with Topology-Encoded Graphs for Anatomical Path Disambiguation in Robot-Assisted Endovascular Navigation

**arXiv ID:** 2602.20215 | [PDF](https://arxiv.org/pdf/2602.20215v1)

**作者:** Jiyuan Zhao `[一作]` (Tongji University), Peng Qi `[通讯]` (Tongji University)

**通讯引用:** 7164 | [OpenAlex ID](https://openalex.org/A5025104684)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个结合SCAR-UNet分割与图注意网络（GAT）进行二维DSA下血管路径规划的完整框架。

**💡 创新点**

创新点在于通过空间坐标注意力增强的分割网络精准提取血管细节，并在生成的血管图上使用GAT对分支与投影交叉进行拓扑一致性推理，从而有效区分真实分叉与投影假交叉。

**🔧 技术方法**

使用了SCAR-UNet（含CoordAttention、SimAM、SE和ASPP模块）进行血管分割，利用几何与图像特征构建血管图，随后采用Graph Attention Network进行路径可行性推断。

**📊 数据集**

在包含192名患者的临床冠状动脉DSA数据集（训练768张、验证200张、测试109张）以及上海操作机器人实验室的血管模型平台进行验证。

**📈 对比分析**

与传统最短路径、启发式规划以及多种现有分割网络相比，SCAR-UNet在分割上Dice达到93.1%，在路径去歧义和目标到达率分别达到95%和90%，显著优于短路方法（60%/55%）和启发式方法（75%/70%）。

**⚠️ 局限性**

局限包括仅基于二维投影，未加入多视角或三维信息，且对分割或骨架化误差仍敏感，未来需加入不确定性评估与多模态数据来进一步提升鲁棒性。

---

## 422. NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning

**arXiv ID:** 2602.21172 | [PDF](https://arxiv.org/pdf/2602.21172v1)

**作者:** Ishaan Rawal `[一作]` (Texas A&M University), Wei Zhan `[通讯]` (Applied Intuition)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种不使用推理（reasoning）和大规模数据的Vision‑Language‑Action (VLA) 模型，称为No Reasoning for Driving；

**💡 创新点**

通过分析奖励分布发现GRPO在弱SFT模型上受难度偏差影响，利用Dr.GRPO消除该偏差，实现数据高效且无需推理的训练；

**🔧 技术方法**

采用Qwen-2.5VL-3B-Instruct作为基座，使用k‑disc离散轨迹编码、有限的SFT（仅80k样本）以及Dr.GRPO强化学习后训练；

**📊 数据集**

在NAVSIM（120小时城市驾驶）和WaymoE2E（360°摄像头的长尾驾驶）两个基准上进行评估；

**📈 对比分析**

与多种基线（AutoVLA、DiffusionLTF、UniPlan等）对比，获得与推理基线相近或更好的PDM与RFS分数，同时使用约60%以下的数据、3倍更少的推理标注，推理时间与token数量均显著降低；

**⚠️ 局限性**

Dr.GRPO虽然改善了难度偏差，但仍非完美；弱SFT模型对极难场景的学习仍有限，且在某些复杂场景下会出现失败案例。

---

## 423. Sequential Counterfactual Inference for Temporal Clinical Data: Addressing the Time Traveler Dilemma

**arXiv ID:** 2602.21168 | [PDF](https://arxiv.org/pdf/2602.21168v1)

**作者:** Jingya Cheng `[一作]` (Massachusetts General Hospital), Hossein Estiri `[通讯]` (Massachusetts General Hospital)

**通讯引用:** 20299 | [OpenAlex ID](https://openalex.org/A5087242254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出顺序反事实框架，解决临床纵向数据中时间旅行悖论导致的生物学不可行反事实；

**💡 创新点**

创新点在于：① 将特征分为不可变、可控、干预三类并引入可行性约束；② 构建基于观测数据的时间依赖图，捕获条件传播；③ 将反事实生成转化为先干预再传播的过程；

**🔧 技术方法**

使用了特征分类、图结构学习、传播算子（Φ）、梯度提升机预测模型、以及距离度量和可行性约束的优化；

**📊 数据集**

利用马萨诸塞州总医院 2026 年 2,723 例 COVID‑19 患者数据（383 例长新冠心衰，2,340 例匹配对照），提取 223 个诊断、药物及实验室异常二元特征；

**📈 对比分析**

与传统单时点反事实方法对比：传统方法导致 38–67% 患者需要生物学不可能的反事实；在长新冠心衰预测上，梯度提升机 AUROC 0.88，顺序框架可产生更可行、可解释的干预建议；

**⚠️ 局限性**

局限包括：单中心数据限制泛化；观测混杂仍未消除；实验室缺失难以区分未测与正常；二元特征失去连续信息；特征分类需临床专家主导，可能因应用场景变化。

---

## 424. Lures of Engagement: An Outlook on Tactical AI Art

**arXiv ID:** 2602.20221 | [PDF](https://arxiv.org/pdf/2602.20221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 425. Is Robot Labor Labor? Delivery Robots and the Politics of Work in Public Space

**arXiv ID:** 2602.20180 | [PDF](https://arxiv.org/pdf/2602.20180v1)

**作者:** EunJeong Cheon `[一作]` (Syracuse University), Do Yeon Shin `[通讯]` (University of Illinois Chicago)

**通讯引用:** 155 | [OpenAlex ID](https://openalex.org/A5101933556)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在首尔两个智慧城市区进行120小时以上的田野观察（步行随车法）和半结构化访谈，本文研究了送餐机器人的劳动是如何通过人力、机构与社会空间协同实现的。

**💡 创新点**

将机器人劳动重新定义为集体社会技术汇聚，首次提出“机器人特权”概念，并揭示公共部署背后的国家监管沙盒与劳动再分配机制。

**🔧 技术方法**

主要采用人类中心计算与人机交互的跨学科方法：步行随车观察、现场记录、访谈、反思性主题分析；未使用机器学习或传统算法技术。

**📊 数据集**

收集的研究材料包括现场笔记、照片、视频、访谈录音和转录文本，数据来源于首尔两个智慧城市试点区的实际运作与互动场景。

**📈 对比分析**

本文不涉及算法性能对比；研究方法为定性主题分析和案例比较，关注不同空间、时间与利益相关者之间的差异与共性。

**⚠️ 局限性**

局限性包括：样本仅限首尔两区，缺乏跨国或跨文化对比；研究聚焦社会技术层面，未评估机器人技术的技术性能；访谈数据可能受受访者主观偏差影响。

---

## 426. Vision-Language Models for Ergonomic Assessment of Manual Lifting Tasks: Estimating Horizontal and Vertical Hand Distances from RGB Video

**arXiv ID:** 2602.20658 | [PDF](https://arxiv.org/pdf/2602.20658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 427. Memory-guided Prototypical Co-occurrence Learning for Mixed Emotion Recognition

**arXiv ID:** 2602.20530 | [PDF](https://arxiv.org/pdf/2602.20530v1)

**作者:** Ming Li `[一作]` (Tsinghua University), Wenping Wang `[通讯]` (Texas A&M University)

**通讯引用:** 15688 | [OpenAlex ID](https://openalex.org/A5100668416)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于记忆引导的原型共现学习框架（MPCL），通过多模态生理与行为信号融合实现混合情绪分布预测。

**💡 创新点**

创新点在于：①使用多尺度关联记忆融合实现跨模态信息的自适应聚合；②构建情感原型记忆库并通过原型关系蒸馏实现跨模态语义一致；③利用Hopfield网络检索机制强化原型级情感共现；④通过分层语义压缩逐步抽象情感表示。

**🔧 技术方法**

核心技术包括现代Hopfield网络（关联记忆与注意力统一）、原型学习与重构、原型关系蒸馏、对比学习（SemLOOB损失）以及分层语义压缩（HSC）。

**📊 数据集**

实验数据集为公开的 DMER（多模态 EEG、GSR、PPG、视频）和 WESAD（ECG、EDA、EMG、ACC）。

**📈 对比分析**

与多类基线方法（LDL、MER、单模态 EDL、MMER、EmotionDict、HeLo）在六种评估指标上进行比较，MPCL 在受试者相关与离线受试者设置下均以平均排名 1 或 2 的优异表现领先，误差下降、相似度提升明显。

**⚠️ 局限性**

局限性包括：①依赖较大的原型记忆和多模态数据，计算与存储成本较高；②模型在极少量数据或单模态场景下的泛化能力待进一步验证；③对不同情感维度（如主观情感空间）适配仍需探索。

---

## 428. LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments

**arXiv ID:** 2602.20925 | [PDF](https://arxiv.org/pdf/2602.20925v1)

**作者:** Zeyu Jiang `[一作]` (Hong Kong University of Science and Technology), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 693 | [OpenAlex ID](https://openalex.org/A5046822372)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一套大规模双目热成像SLAM系统LST‑SLAM，能在动态、千米级、光照变换环境下实现稳健定位与建图。

**💡 创新点**

创新点包括：1）自监督热点网络（STP）利用热域单应性训练提升特征；2）双层跟踪结合光度与描述子，实现更强鲁棒性；3）语义‑几何混合动态物体过滤；4）在线增量二进制BoW实现循环闭环与全局优化。

**🔧 技术方法**

采用的技术包括深度学习自监督热特征、YOLOv8分割、二进制描述子Hamming匹配、Stereo双层跟踪、Bundle Adjustment、Sim(3)全局优化以及增量BoW。

**📊 数据集**

主要使用M2DGR和MS^2热成像数据集，涵盖多时段、千米级街景与室外动态场景。

**📈 对比分析**

与SVO、DytanVO、TartanVO、AirSLAM、DROID‑SLAM、ORB‑SLAM3等经典与学习型系统对比，实验显示在无闭环和有闭环条件下，LST‑SLAM平均绝对误差分别比AirSLAM低75.8%、比DROID‑SLAM低66.8%，整体定位误差和鲁棒性均明显优于对手。

**⚠️ 局限性**

局限性包括：仅基于热相机，未加入IMU或LiDAR；计算量较大，实时性待提升；对极端噪声和低对比度场景仍存在挑战；需要大规模热数据进行自监督训练。

---

## 429. Quantitative Approximation Rates for Group Equivariant Learning

**arXiv ID:** 2602.20370 | [PDF](https://arxiv.org/pdf/2602.20370v1)

**作者:** Jonathan W. Siegel `[一作]` (Texas A&M University), Nadav Dym `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文旨在为群等变学习中的量化逼近率提供结果，特别是针对具有群对称性的α-Hölder函数的逼近能力进行分析。

**💡 创新点**

创新点在于首次为多个显著的群等变和不变架构推导出量化逼近率，证明了等变架构在表达能力上与普通的ReLU MLPs相当。

**🔧 技术方法**

使用了ReLU神经网络，特别是Deep Sets、Sumformer和Transformer等架构来实现群等变学习。

**📊 数据集**

使用了α-Hölder函数作为目标函数，研究了在不同群对称性下的逼近能力。

**📈 对比分析**

通过与普通ReLU网络的比较，发现等变架构在逼近能力上没有损失，且在样本效率和优化方面表现更好。

**⚠️ 局限性**

限制在于目前对许多流行的等变模型（如EGNN、Dimenet等）的逼近率尚未建立，未来需要进一步研究这些模型的逼近能力。

---

## 430. VAGNet: Grounding 3D Affordance from Human-Object Interactions in Videos

**arXiv ID:** 2602.20608 | [PDF](https://arxiv.org/pdf/2602.20608v1)

**作者:** Aihua Mao `[一作]` (Institution1), Ying He `[通讯]` (Institution2)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

该文档为IEEE论文排版与提交规范的详细说明，并未包含具体研究内容

**💡 创新点**

无研究创新点

**🔧 技术方法**

无技术实现

**📊 数据集**

无数据集

**📈 对比分析**

无方法对比与性能评估

**⚠️ 局限性**

主要局限是缺乏科研内容，无法评估研究方法与结果

---

## 431. ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning

**arXiv ID:** 2602.21078 | [PDF](https://arxiv.org/pdf/2602.21078v1)

**作者:** Duowen Chen `[一作]` (Shanghai Key Laboratory of Multidimensional Information Processing East China Normal University), Yan Wang `[通讯]` (Shanghai Key Laboratory of Multidimensional Information Processing East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ProxyFL，一个利用统一代理（分类器权重）来同时缓解联邦半监督学习中外部异质性与内部异质性的新框架；通过全局代理调优（GPT）显式拟合全局类别分布，并通过不确定类别代理学习（ICPL）将低置信度无标签样本转为多类别集合，构建正负代理池进行对比学习，以提升模型性能与收敛速度。

**💡 创新点**

创新点：① 用分类器可学习权重作为全局/本地类别分布的代理，避免了对原始数据的隐私泄露；② GPT 在服务器端对代理进行显式距离最小化/最大化优化，使全局代理更好地逼近真实类别分布，克服了传统平均聚合对离群点的敏感性；③ ICPL 将低置信度样本映射为“不可决定类别”集合，利用动态全局先验阈值构造多类别标签，并在正负代理池中对比学习，显著提升低置信度样本的利用率并降低伪标签误差。

**🔧 技术方法**

技术：联邦学习框架（FedAvg/FedProx）、半监督学习伪标签与强弱增强、对比学习、距离/对数似然损失、全局代理微调、动态不确定类别集合、Dirichlet 采样模拟非 IID、ResNet‑8 作为特征提取器、Kullback‑Leibler 与交叉熵混合训练。

**📊 数据集**

数据集：CIFAR‑10、CIFAR‑100、SVHN、CINIC‑10；在每个数据集上采用 10%/20% 的标签比例，使用 Dirichlet 分布 α ∈ {0.1, 0.5, 1} 生成不同程度的非 IID 数据分布。

**📈 对比分析**

对比方法：FedAvg、FedProx、FedAvg‑SL（全监督上界）、FixMatch‑LPL/GPL、FedMatch、FedLabel、FedLoke、FedDure、FedDB、SAGE 等 FSSL 及 FL+SSL 组合；实验表明 ProxyFL 在所有数据集和所有 α 下均获得最高或次高准确率，提升幅度约 1–4%（对 CIFAR‑100 α=0.1 提升 3.3%），并且收敛速度加快、通信效率更高。

**⚠️ 局限性**

局限性：① 仅在图像分类任务验证，缺乏对自然语言或其他领域的泛化评估；② 代理权重和 ICPL 的阈值、类别集合构造依赖经验参数，可能在不同任务或极端异质性场景下需要重新调优；③ 服务器端全局代理微调虽然计算量小，但在大规模类别数或模型复杂度高时仍有一定开销；④ 对于非常稀疏标签或极高噪声的无标签数据，ICPL 的多类别集合可能仍产生误差。

---

## 432. Knowing the Unknown: Interpretable Open-World Object Detection via Concept Decomposition Model

**arXiv ID:** 2602.20616 | [PDF](https://arxiv.org/pdf/2602.20616v1)

**作者:** Xueqiang Lv `[一作]` (Northwestern Polytechnical University), Yanning Zhang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 21405 | [OpenAlex ID](https://openalex.org/A5028235866)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可解释的开放世界目标检测框架IPOW，通过概念分解模型将RoI特征拆分为判别、共享和背景三类概念，实现对已知与未知目标的分离与解释。

**💡 创新点**

创新点在于：① 通过概念分解模型揭示已知-未知混淆的根源——未知目标落入已知类别的判别空间；② 提出概念引导修正（CGR）利用共享概念的激活差异消除混淆；③ 结合LLM生成共享概念并用稀疏自编码补充残留概念，进一步提升未知召回；④ 在检测框架中加入GMM-RPN、背景PCA概念等模块。

**🔧 技术方法**

采用的技术包括：Faster R-CNN骨干+GMM-RPN、概念瓶颈模型（CBM）、CLIP文本编码、LLM概念提取、稀疏自编码器、PCA背景概念、对比学习和交叉熵损失、概念引导修正机制。

**📊 数据集**

使用的数据集包括：M-OWODB、S-OWODB、Remote Sensing DIOR，以及在实验中对比的其它公开基准（如COCO等）。

**📈 对比分析**

与ORE、OW-DETR、PROB、CAT、RandBox、OrthogonalDet、CROWD等最新方法对比。IPOW在M-OWODB任务中将U-Recall提升7.2~11.6个百分点，在S-OWODB提升4~7个百分点，同时保持或提升已知类别mAP；Wilderness Impact（WI）和Absolute Open‑Set Error（A‑OSE）显著下降，表明已知-未知混淆得到有效缓解。

**⚠️ 局限性**

局限性：① LLM生成的共享概念仍不完整，需要手动或自动扩充；② 概念数目（K、M）对性能敏感，需要经验调参；③ 在超类别分离的S-OWODB中效果略逊，表明共享概念的可迁移性受限；④ 对极端场景（如遥感）虽然有提升，但仍存在空间提升。

---

## 433. Singular Arrange and Traverse Algorithm for Computing Reeb Spaces of Bivariate PL Maps

**arXiv ID:** 2602.21087 | [PDF](https://arxiv.org/pdf/2602.21087v1)

**作者:** Petar Hristov `[一作]` (Linköping University), Talha Bin Masood `[通讯]` (Linköping University)

**通讯引用:** 572 | [OpenAlex ID](https://openalex.org/A5060181768)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了“singular arrange and traverse”算法，用以精确高效地计算双变量映射的Reeb空间。

**💡 创新点**

只考虑奇异边和奇异面，跳过非必要面与预像图的计算，通过纤维图类与奇异对应图实现批量处理，显著减少计算量和内存。

**🔧 技术方法**

基于CGAL的二维段式排列、红蓝段交点求解、平衡树动态维护纤维图、BFS遍历、红蓝交点、极小化非必要面、使用精确几何谓词与构造。

**📊 数据集**

MVK（量子化学）、ethane‑diol（分子动力学）以及Enzo（宇宙学数值模拟）三组数值模拟数据。

**📈 对比分析**

与现有的 arrange‑and‑traverse 实现做基准测试，比较运行时间与内存；MVK上速度提升高达10^4倍，ethane‑diol 10^2倍，Enzo 10^1倍；内存使用亦大幅下降。

**⚠️ 局限性**

在奇异边比例较高时收益降低；对非泛化映射需额外扰动或鲁棒性处理；处理嵌套面仍采用启发式伪奇异边；当前实现为单线程，未做并行化；算法最坏情况下仍为 O(N_s N_t log N_t)。

---

## 434. Automated Detection and Mitigation of Dependability Failures in Healthcare Scenarios through Digital Twins

**arXiv ID:** 2602.21037 | [PDF](https://arxiv.org/pdf/2602.21037v1)

**作者:** Bruno Guindani `[一作]` (Politecnico di Milano), Marcello M. Bersani `[通讯]` (Politecnico di Milano)

**通讯引用:** 597 | [OpenAlex ID](https://openalex.org/A5052125239)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一套基于闭环数字孪生的医疗 CPS 可靠性分析方法，融合 SHA 模型、数据驱动学习和统计模型检测，能够主动识别失效场景并自动生成干预策略。

**💡 创新点**

首次将数字孪生与混合随机自动机、模糊测试、游戏论策略合成相结合，实现对医疗 PDP 三元组的主动失效检测、多样性分析和在线决策支持。

**🔧 技术方法**

采用混合随机自动机（SHA）、统计模型检测（SMC）、变异模糊测试、NSGA-II 搜索、1½ 玩家游戏策略合成，以及深度学习/传统回归模型进行患者动态学习。

**📊 数据集**

使用 Kitware Pulse 高保真生理模拟器生成的 BREATHE 仿真数据，共 45 分钟 8 个临床场景，另外采用 20 个仿真情景评估模型精度。

**📈 对比分析**

与传统神经网络、XGBoost、Elastic Net 进行回归/分类对比，学习到的 SHA 模型在分类准确率上提升 3 倍；对失效检测采用随机、模糊与搜索三种探索，模糊测试发现 35–60% 真实失效且产生最多簇；控制策略与真实医生对比，87.5% 情景下性能相当，平均 TV 靠近健康值 20% 更好。

**⚠️ 局限性**

仅在单一仿真病人环境评估，缺乏真实患者数据；搜索方法偏向简单模型导致失效多样性不足；假设医生完全接受建议，未考虑人机交互与部分遵从；对长期临床结果评价不足。

---

## 435. Learning to Solve Complex Problems via Dataset Decomposition

**arXiv ID:** 2602.20296 | [PDF](https://arxiv.org/pdf/2602.20296v1)

**作者:** Wanru Zhao `[一作]` (University of Cambridge), Alessandro Sordoni `[通讯]` (Microsoft Research)

**通讯引用:** 5296 | [OpenAlex ID](https://openalex.org/A5108137151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于教师模型递归分解数据的逆向课程学习方法，让小型语言模型先学习简单子任务再逐步掌握复杂数学推理。

**💡 创新点**

创新点在于通过步骤拆分与概念标签构建层次化子任务树，并用结构复杂度与概念深度相结合的难度评分来自动生成逐步递进的学习路径。

**🔧 技术方法**

主要技术包括大规模语言模型（GPT‑4o）作为教师进行自动推理与子任务生成，概念依赖图构建与聚类，难度分数计算，以及按难度划分的分阶段训练框架。

**📊 数据集**

实验使用 MATH、AIME 以及 CodeForces‑CoTs 三大数据集，并在 HumanEval 评测中验证了跨域推广能力。

**📈 对比分析**

与多种基线（直接 SFT、元数据增强、直接蒸馏等）相比，逆向课程学习在 MATH‑500、AIME‑2025 与 HumanEval 上分别提升约1–3 %以及 10–25 % 的准确率，展示出显著的性能优势。

**⚠️ 局限性**

主要局限包括对教师模型的强大推理能力与准确性的依赖，易受模型幻觉影响，且在非数学任务或高风险领域的适用性尚未充分验证。

---

## 436. On Data Engineering for Scaling LLM Terminal Capabilities

**arXiv ID:** 2602.21193 | [PDF](https://arxiv.org/pdf/2602.21193v1)

**作者:** Renjie Pi `[一作]`, Wei Ping `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向终端智能体的两阶段数据生成框架，结合数据集适配器和基于种子/技能的合成任务，并使用 Qwen3 系列模型进行监督微调。

**💡 创新点**

创新点包括：①粗到细的“数据适配+合成任务”协同生成策略；②利用预构建域特定 Docker 镜像简化环境构建；③在合成任务中严格区分任务描述与解答，防止泄漏；④系统化的过滤与课程学习实验，发现保留失败轨迹能提升鲁棒性。

**🔧 技术方法**

核心技术包括：LLM 任务生成（DeepSeek‑V3.2 作为教师模型）、Docker 化轨迹采集、JSON 结构化交互、长上下文与 YaRN2 的训练/评估、过滤与数据混合策略、单阶段混合训练。

**📊 数据集**

使用的数据集包括：来自 Nemotron‑Cascade 的数学、代码、软件工程原始问题集（通过适配器转换为终端任务）；基于种子与技能的合成任务；Terminal‑Bench 2.0 作为评估基准。

**📈 对比分析**

在 Terminal‑Bench 2.0 上与基线 Qwen3‑8B、Qwen3‑Coder‑480B 等进行比较，-8B/‑14B/‑32B 分别取得 13.0、20.2、27.4 的平均分，显著超越 Qwen3‑8B（2.47）并逼近甚至超过更大模型（如 Qwen3‑Coder‑480B 23.9）。

**⚠️ 局限性**

局限性包括：①合成任务仍需人工审核以确保 Oracle 正确性；②对极其复杂的长周期任务的覆盖有限；③对教师模型质量高度依赖，可能导致生成任务的偏差；④训练成本仍相对较高，尤其在更大模型上；⑤数据过滤策略在不同任务类型间的通用性尚未充分验证。

---

## 437. Spa3R: Predictive Spatial Field Modeling for 3D Visual Reasoning

**arXiv ID:** 2602.21186 | [PDF](https://arxiv.org/pdf/2602.21186v1)

**作者:** Haoyi Jiang `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 32532 | [OpenAlex ID](https://openalex.org/A5037191476)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自监督的预测空间场模型（PSFM）Spa3R，通过从无姿态多视角图像学习统一的视角不变空间表示，并将其与视觉‑语言模型（VLM）结合，提升3D空间推理能力。

**💡 创新点**

创新点在于：①通过预测任意新视角的特征场，实现全局3D几何与语义的自监督学习；②使用对称视角聚合器与相对位置编码（PRoPE）避免信息泄露并强化几何约束；③以轻量级交叉注意力适配器将预训练的空间编码器无缝注入VLM，形成Spa3-VLM。

**🔧 技术方法**

主要技术包括：Transformer‑基的编码器/解码器、对称视角注意力掩码、相机射线查询、PRoPE相对位置编码、轻量级残差交叉注意力适配器、以及自监督的特征重建损失。

**📊 数据集**

使用ScanNet和ScanNet++两大室内RGB‑D视频数据集进行预训练；在VSI‑Bench（约5k问答）以及CV‑Bench、SPAR‑Bench、ViewSpatial‑Bench等多项空间推理基准上进行评估。

**📈 对比分析**

与基线VLM和现有空间增强方法对比，Spa3‑VLM在VSI‑Bench上取得58.6%准确率，超过对比模型3.5%以上；在其他基准亦表现出显著提升，证明PSFM提供了更稳健的3D空间先验。

**⚠️ 局限性**

局限性包括：①预训练需要大量无姿态多视角图像，对单视角或动态场景的适用性尚未验证；②模型在极少视角条件下（mask率过高或过低）性能下降，需精细平衡；③当前仅在室内场景上验证，跨域泛化能力仍待探索。

---

## 438. Test-Time Training with KV Binding Is Secretly Linear Attention

**arXiv ID:** 2602.21204 | [PDF](https://arxiv.org/pdf/2602.21204v1)

**作者:** Junchen Liu `[一作]` (NVIDIA), Ruilong Li `[通讯]` (NVIDIA)

**通讯引用:** 1339 | [OpenAlex ID](https://openalex.org/A5101961150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文重新审视并证明了Test‑Time Training (TTT) 与 KV binding 的模型实际上是学习型线性注意力机制，而非传统意义上的测试时记忆与检索。

**💡 创新点**

核心创新在于将多层非线性内循环更新等价于可解析的线性注意力运算，解释了之前与记忆假设相悖的实验现象，并通过此视角实现了模型结构简化、并行化以及性能提升。

**🔧 技术方法**

主要技术包括对内循环梯度更新的解析展开、证明其为线性注意力形式、对 LaCT 与 ViTTT 进行变体层层剥离、以及构建可并行的前缀扫描实现。

**📊 数据集**

实验使用了三大任务的数据集：LLM 任务使用 Book‑3 书籍文本；NVS（Novel View Synthesis）任务使用对应视觉数据；图像分类任务使用 ImageNet 或类似公开数据。

**📈 对比分析**

通过一系列消融实验，作者发现仅更新最终线性层即可保持甚至提升性能；并行实现将推理吞吐量提升至原来的 4 倍，训练速度提升 1.19×，在 LLM 任务上 perplexity 下降 0.5，NVS 上 PSNR 上升 0.2dB，ViTTT 图像分类 top‑1 准确率提升 0.3%。

**⚠️ 局限性**

局限性包括：分析仅适用于最终线性无偏的内循环；对非线性最终层或更复杂优化器的推广尚未完成；在极端长序列或低资源场景下的可扩展性仍待验证。

---

## 439. Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics

**arXiv ID:** 2602.21203 | [PDF](https://arxiv.org/pdf/2602.21203v1)

**作者:** Abdulaziz Almuzairee `[一作]`, Henrik I. Christensen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文训练了一个名为 Squint 的视觉强化学习算法，在 15 分钟内用单张 RTX 3090 GPU 学习 8 个 SO‑101 机器人任务，并实现零 shot 真实机器人部署。

**💡 创新点**

创新点在于将并行仿真、分布式 critic、低分辨率下采样（squinting）、层归一化、调优更新-数据比率以及 PyTorch 编译/AMP/Cudagraph 等技术结合，显著提升了视觉 RL 的 wall‑clock 训练速度，并实现快速 sim‑to‑real 转移。

**🔧 技术方法**

采用改进的 Soft Actor Critic（SAC）架构，使用 C51 分布式 critic、16×16 图像下采样与 squinting、层归一化、超大并行环境（1024）和 0.25 的更新-数据比率，并配合 PyTorch compile、Cudagraph、AMP bfloat16 等加速手段。

**📊 数据集**

实验基于 ManiSkill3 的 SO‑101 Task Set 共 8 个任务，并在该数字双胞胎仿真环境中进行训练，随后在真实 SO‑101 机械臂上验证。

**📈 对比分析**

与 SAC、PPO、DrQ‑v2、BC、DAgger 等基线对比，Squint 在仿真中 15 分钟后平均成功率达到 96.1%（SAC 88.3%），真实机器人成功率为 91.3%（SAC 81.3%），并在同一训练时间内实现更快收敛。

**⚠️ 局限性**

局限性包括对视觉变化的鲁棒性不足、样本效率仍有提升空间、抓手设计导致某些任务收敛慢、单任务设定缺乏通用性、未采用特权训练或多任务/多视角扩展，且需要进一步改进 Sim‑Real 共训练方法。

---

## 440. Multi-Vector Index Compression in Any Modality

**arXiv ID:** 2602.21202 | [PDF](https://arxiv.org/pdf/2602.21202v1)

**作者:** Hanxiang Qin `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**通讯引用:** 8641 | [OpenAlex ID](https://openalex.org/A5075825791)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多向量检索在任意模态下的索引压缩方法，提出了一种新的注意力引导聚类（Attention‑Guided Clustering）技术，以在固定向量预算内压缩文档表示。

**💡 创新点**

创新点在于：①利用全局可学习的查询token引导注意力挑选语义显著的聚类中心；②在聚类过程中使用加权聚合，避免信息丢失；③证明该方法在文本、视觉文档、视频（视觉和音频）四种模态上均能保持高检索性能，并在多项基准上刷新了 SOTA。

**🔧 技术方法**

技术手段包括：多向量Late Interaction（ColBERT 风格），投影压缩、内存token、层次池化等对比方法，以及注意力引导的聚类与加权聚合；使用 FastPlaid 及自定义索引实现高效检索；训练时采用 distillation 损失与多模态预训练模型（Qwen、ColPali 等）。

**📊 数据集**

使用的数据集涵盖四类模态：文本检索（多家医学、财经、争论领域数据集），视觉文档检索（v2 Visual Document Retrieval benchmark），视频检索（MSR‑VTT、VATEX、DiDeMo、ActivityNet Captions）以及音视频检索（多模态视频检索数据集）。

**📈 对比分析**

与基线（完整多向量索引）和其他压缩方法（Sequence Resizing、Memory Tokens、Hierarchical Pooling）进行对比。实验结果显示，Attention‑Guided Clustering 在保持 97% 以上 uncompressed 结果的同时，压缩率可达 80%+；在视频和音视频任务中，在 5-128 维度压缩下甚至优于完整索引，证明其在不同模态和压缩比例下的稳健性。

**⚠️ 局限性**

局限性：①目前压缩比例是固定的，未能根据文档信息量动态分配；②对极短或极长文档的适配仍需研究；③在音频采样与多模态信号的高效编码上仍存在性能瓶颈。

---

## 441. Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs

**arXiv ID:** 2602.21198 | [PDF](https://arxiv.org/pdf/2602.21198v1)

**作者:** Yining Hong `[一作]` (Stanford University), Yejin Choi `[通讯]` (Stanford University)

**通讯引用:** 25895 | [OpenAlex ID](https://openalex.org/A5102992157)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在机器人部署时通过内部和外部反思双向学习（Reflective Test‑Time Planning）来实现自适应行为的框架，能够在执行失败后即时纠正并持续改进决策。

**💡 创新点**

创新点在于将反思分为“行动前反思（in‑action）”与“行动后反思（on‑action）”两种模式，并通过自监督的反思文本驱动测试时参数更新，实现单次部署即刻学习，解决了传统静态LLM缺乏经验累积的问题。

**🔧 技术方法**

技术包括：多模态LLM（如LLaVA‑3D、Qwen2.5‑VL）、内部/外部评估子模型、测试时扩容（high‑temperature采样）与测试时训练（LoRA+REINFORCE与监督学习）以及回溯反思（hindsight）以实现长期信用分配。

**📊 数据集**

使用了两套新设计的长时序家居任务（Long‑Horizon Household）基准（基于BEHAVIOR‑1K）和受控MuJoCo “Cupboard Fitting”任务，同时利用GPT‑5生成任务和对话数据作为训练/评估集。

**📈 对比分析**

与多种基线（如Reflexion、Self‑Refine、PPO、DreamerV3、3DLLM‑Mem）对比，实验显示在Fitting、Selection、Preparation、Hybrid等子任务中平均成功率提升约+24%（从10%提升至≈35%），Cupboard Fitting中fit率提升至60%（相较于基线≈25%），证明双向反思与测试时学习显著优于单向或无反思方法。

**⚠️ 局限性**

局限包括：依赖LLM的语言生成质量，反思文本可能携带模型偏见；测试时更新需额外算力，且在极端分布偏移下的鲁棒性尚未彻底验证；以及在安全关键场景下自动自适应的潜在风险。

---

## 442. Untied Ulysses: Memory-Efficient Context Parallelism via Headwise Chunking

**arXiv ID:** 2602.21196 | [PDF](https://arxiv.org/pdf/2602.21196v1)

**作者:** Ravi Ghadia `[一作]` (Together AI), Max Ryabinin `[通讯]` (Together AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Untied Ulysses，通过在注意力层级按头（head）细粒度拆分（chunk）执行自注意力，从而显著降低激活内存，支持更长上下文。

**💡 创新点**

创新点在于将 Ulysses 的全头执行解耦，采用 headwise chunking 与 GQA 兼容的调度，能够把 self‑attention 的中间张量内存减少多达 87.5%，并在保持吞吐量的同时突破原有的内存瓶颈。

**🔧 技术方法**

技术手段包括：DeepSpeed‑Ulysses 的 all‑to‑all 机制、Flash Attention‑3、TorchTitan 框架、激活检查点与 CPU offloading、TiledCompute（对 FFN 与 CE 损失的切片）、Liger‑Kernel（高效交叉熵）、以及 GQA 调度来避免冗余通信。

**📊 数据集**

实验主要使用 Llama3‑8B 与 Qwen3‑32B 两个预训练模型（相当于在其原始训练数据上继续训练），未使用公开的独立数据集。

**📈 对比分析**

对比基线包括 Ring Attention、DeepSpeed‑Ulysses、Fully Pipelined Distributed Transformer (FPDT) 与 Unified Sequence Parallelism (USP) 混合模式。单机上，Llama3‑8B 可在 5M token（比 FPDT 高 25%）；多机上可达 8M token；吞吐量与 Ulysses 基线相当或更佳，而内存占用大幅下降。

**⚠️ 局限性**

局限性：最佳内存效率需要将 chunk 大小设置为 U=C，导致在小序列时额外的 kernel 启动开销；依赖于 H100 GPU、NVLink 及大量 CPU 内存；对不同硬件或更大模型的兼容性未完全验证。

---

## 443. Statistical Query Lower Bounds for Smoothed Agnostic Learning

**arXiv ID:** 2602.21191 | [PDF](https://arxiv.org/pdf/2602.21191v1)

**作者:** Ilias Diakonikolas `[一作]` (University of Wisconsin), Daniel M. Kane `[通讯]` (University of California)

**通讯引用:** 2907 | [OpenAlex ID](https://openalex.org/A5034540754)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究在高斯或次高斯分布下，加入轻微高斯扰动的半空间无偏学习问题，并给出了相应的统计查询(SQ)下的复杂度下界；

**💡 创新点**

证明了即便在 smoothed 模型中，已知的 L1‑多项式回归算法几乎已经是最优的，给出了与已知上界几乎匹配的 SQ 下界；

**🔧 技术方法**

采用线性规划对偶与 Hermite 多项式分析相结合的方法，构造了 moment‑matching 难点分布并得到多项式逼近度的下界；

**📊 数据集**

该工作纯理论分析，没有使用具体实验数据集；

**📈 对比分析**

与已有的 L1‑多项式回归上界进行比较，得到下界与上界在 1/σ² 与 log(1/ϵ) 上的指数匹配，说明该算法已接近最优；

**⚠️ 局限性**

仍无法提供多项式时间算法达到最佳误差，且对重尾分布的计算难度尚未给出下界。

---

## 444. Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training

**arXiv ID:** 2602.21189 | [PDF](https://arxiv.org/pdf/2602.21189v1)

**作者:** Anas Barakat `[一作]` (Singapore University of Technology and Design), Amrit Singh Bedi `[通讯]` (University of Central Florida)

**通讯引用:** 933 | [OpenAlex ID](https://openalex.org/A5039563144)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并解释了在使用pass@k（多次抽样验证）进行策略梯度优化时，如何导致单次抽样成功率pass@1下降的问题。

**💡 创新点**

创新点在于：①提出“prompt interference”概念，量化提示之间梯度的正负干扰；②给出pass@k与pass@1梯度冲突的理论判据和k的临界阈值；③通过梯度内积与权重协方差解析梯度冲突的本质。

**🔧 技术方法**

主要技术包括：pass@k策略梯度推导、梯度相似性核（prompt similarity kernel）、梯度内积表达式、梯度冲突阈值分析、平滑性证明以及对大语言模型的蒙特卡罗梯度估计。

**📊 数据集**

使用公开的MATH数学推理数据集（包含多学科高中题目）以及两种LLM模型：DeepSeek-R1-Distill-Llama-8B 与 DeepSeek-R1-Distill-Qwen-7B。

**📈 对比分析**

实验通过对不同难度阈值组合的提示集合计算agreement score、pass@k权重、梯度内积等指标。结果显示：pass@k梯度在对难题重加权后与pass@1梯度形成明显负向角度，导致pass@k提升而pass@1下降；在多种阈值配置下均验证了理论预测，梯度内积从正转负，体现梯度冲突。

**⚠️ 局限性**

局限性包括：仅针对pass@k二元验证目标；未提供有效的冲突缓解方法；实验范围局限于数学推理任务，尚未验证在其他推理或代码生成任务中的泛化；对梯度冲突的定量阈值依赖于假设的平滑性与分布分离等理想化条件。

---

## 445. Human Video Generation from a Single Image with 3D Pose and View Control

**arXiv ID:** 2602.21188 | [PDF](https://arxiv.org/pdf/2602.21188v1)

**作者:** Tiantian Wang `[一作]` (University of California), Varun Jampani `[通讯]` (Stability AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 Human Video Generation in 4D 的潜在视频扩散模型，能够从单张图像通过 3D 姿态与视角控制生成高质量、多视角、时空连贯的人类视频。

**💡 创新点**

创新点在于三大设计：双维骨映射的姿态调制解决自遮挡与形状泄漏；轻量化视角与时间对齐提升多视角一致性；以及渐进式时空采样实现长时长多视角生成。

**🔧 技术方法**

采用潜在扩散模型结合 SMPL 骨骼、双维骨映射、卷积、空间视角与时间注意力、Cross‑Attention 及渐进时空采样等技术。

**📊 数据集**

使用 THuman2.0/2.1、CustomHuman、2K2K 和 MVHumanNet 等多个人体扫描与多视角视频数据集进行训练和评测。

**📈 对比分析**

与 MagicAnimate、AnimateAnyone、Champ、MimicMotion、AniGS、LHM 等方法对比，实验在 FID、SSIM、PSNR、LPIPS、FVD 等指标上均优于基线，尤其在多视角连贯性和姿态准确性方面表现突出。

**⚠️ 局限性**

局限性包括对面部细节捕捉不足、在全身高帧率长视频中可能出现面部失真，以及模型对极端姿态或遮挡的鲁棒性仍有提升空间。

---

## 446. Region of Interest Segmentation and Morphological Analysis for Membranes in Cryo-Electron Tomography

**arXiv ID:** 2602.21195 | [PDF](https://arxiv.org/pdf/2602.21195v1)

**作者:** Xingyi Cheng `[一作]` (Institut Curie), Daniel Lévy `[通讯]` (Institut Curie)

**通讯引用:** 5826 | [OpenAlex ID](https://openalex.org/A5004850929)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了TomoROIS和SurfORA两套工具，分别用于在cryo‑ET数据中直接、形状无关的ROI分割和随后对分割结构进行表面几何量化分析。

**💡 创新点**

创新点在于将ROI分割作为首要任务，采用轻量级MSDCN网络可从少量标注学习上下文，支持开放表面和复杂几何的自动表面提取、法向一致化以及多尺度曲率评估。

**🔧 技术方法**

使用的技术包括：混合尺度密集卷积网络（MSDCN）进行ROI分割；基于点云的表面提取（MLS/isosurface）、球面重建、法向一致化（基于热传播）和曲率/距离量化；Python生态（Napari, PyVista, PyVista）实现GUI交互。

**📊 数据集**

实验数据来自两套in vitro重构膜系统：①VAP‑A/OSBP共价桥接的膜接触站（58套倾斜序列）；②纯脂质囊泡在渗压条件下产生的膜内陷（30套倾斜序列）。

**📈 对比分析**

与传统全局膜分割+人工筛选的工作流程相比，TomoROIS在MCS数据上误报率17%、漏报率3%，Dice 0.89；在内陷数据上误报10.5%、漏报1%。SurfORA能够准确测得MCS间距10–30 nm、内陷膜曲率正负分布，且无人工干预即可生成高质量网格。

**⚠️ 局限性**

局限包括：在高度重叠的MCS中，基于分水岭的实例分离仍不完美；对极端噪声或缺失楔影响的开放表面重建需进一步改进；模型对少量标注依赖较大，需多次迭代精调。

---

## 447. Aletheia tackles FirstProof autonomously

**arXiv ID:** 2602.21201 | [PDF](https://arxiv.org/pdf/2602.21201v1)

**作者:** Tony Feng `[一作]` (Google DeepMind), Thang Luong `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本文中，作者使用 Gemini 3 Deep Think 以及自定义的生成-验证-提取流水线，自动化解决了 FirstProof 基准中的十个研究级数学问题，并对结果进行人工专家评估。

**💡 创新点**

创新点在于提出了严格的自治评估框架（生成+验证+提取+最佳‑二策略）、自我过滤机制以及完整的专家评审流程，从而实现高质量的 AI 证明输出。

**🔧 技术方法**

使用的技术包括 Gemini 3 Deep Think 大语言模型、多阶段代理体系（Generator、Verifier、Extractor）、最佳‑二决策、代码化输出以及基于 prompt 的验证与提取。

**📊 数据集**

采用的数据集为 2026 年 2 月 5 日发布的 FirstProof benchmark（共十个数学问题），未使用其他公开训练数据。

**📈 对比分析**

评估方法为与专家评审结果比较，取得 10 题中 6 题被至少 3 位专家认定为“正确”，其中 P8 仍有 2 位专家未同意；与先前模型相比，准确率显著提升，但推理成本较高。

**⚠️ 局限性**

局限性包括仅覆盖 10 个问题、4 题未得到解答、推理成本高、对“正确”定义的主观性较大、以及对外部专家评审的高度依赖。

---

